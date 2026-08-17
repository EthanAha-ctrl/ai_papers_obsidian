---
source_pdf: Understanding_Economic_Tradeoffs_Between_Human_and.pdf
paper_sha256: 268f1a0d6b06f7a6d87433bad51efc3eba09ba77f689c4fc44e57494bb3fa020
processed_at: '2026-08-12T19:18:15-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 paper

## 一句话总结

三个人玩换筹码的游戏，人类、AI、贝叶斯机器人都在玩，最后赚的钱差不多，但玩的方式完全不一样——人类讲公平，AI 当老好人，贝叶斯机器人贪心但常被拒。

## 游戏怎么玩

三个人，每人手里有各种颜色的筹码。绿色筹码值 5 毛钱，其他颜色每个人心里有不同的估值（别人不知道）。

每一轮，一个人提议"我拿 2 个红筹码换你的 3 个蓝筹码"，另外两个人同时说要不要。两个人都要的话，随机选一个成交。玩 9 轮，最后你手里的筹码按你的估值算钱，真金白银给你。

就这么简单。

## 三个玩家是谁

**人类**：216 个美国人，在 Prolific 上招募的，平均赚了 123 美元。

**AI**：GPT-4o 和 Gemini 1.5 Pro，用最简单的 prompt 跑，跟人类看到完全一样的信息。

**贝叶斯机器人**：专门为这个游戏手写的算法。它会猜别人心里筹码值多少钱，每次出招都算"我最多能赚多少"，被拒了就更新对别人的猜测。

## 结果：钱赚得差不多，玩法天差地别

### 赚钱排名

贝叶斯 > GPT-4o ≈ 人类 > Gemini

贝叶斯在 2-chip 游戏里能达到理论最优的 74%，3-chip 时能到 80%。GPT-4o 跟人类差不多，Gemini 明显差一截。

### 但玩法完全不同

这是 paper 最核心的发现。

**人类怎么玩**：提议"我给你 2 个红的，你给我 2 个蓝的"——给多少要多少，讲究公平。哪怕这是 one-shot game，以后再也见不到这些人，人类还是倾向于公平交换。事后问卷里，人类自己说的策略关键词就是"fairness"和"cooperation"。

**AI 怎么玩**：GPT-4o 和 Gemini 倾向于"我给你 3 个红的，你给我 1 个蓝的就行"——多给少要，当老好人。结果就是接受率很高，但创造的 surplus 很少。甚至有不少 proposal 是自己亏钱的。

**贝叶斯怎么玩**：反过来，"我给你 1 个红的，你给我 3 个蓝的"——少给多要，贪心。创造的 surplus 最高，但被拒率也最高。它不在意被拒，因为它算的是期望收益。

### Regret 分析更扎心

Paper 还做了一个后悔分析，分三种：

- **No regret**：这步棋是对的，后面也没有更好的机会
- **Forced regret**：后面来了更好的机会，但因为你之前已经把筹码用了，抓不住了
- **Unforced regret**：后面来了更好的机会，你本来能抓住，但你蠢了没抓住

结果：

贝叶斯机器人随着游戏变复杂，no regret 的比例反而上升——它越来越准。人类和 AI 随着游戏变复杂，regret 越来越多。

但关键点是——贝叶斯在复杂游戏里表现相对变好，不是因为贝叶斯变强了，而是人类和 AI 变弱了。complexity 上去，人类和 AI 的 decision quality 掉得比贝叶斯快。

## 为什么 AI 当老好人

Paper 列了三个可能原因：

1. **训练数据偏向**：训练语料里，cooperative 的对话被隐式奖励，minimizing friction 成了一种 prior
2. **风险厌恶 + 缺反馈**：RLHF 让 model 偏向 safe、passive 的 response，而且训练时没有 outcome-driven 的 feedback 闭环
3. **不对称厌恶**：model 学到了"给多要少"是一种 coordination strategy，能确保对方接受

这跟 RLHF 的 alignment tax 是一回事——alignment 让 model 更 polite、更 cooperative，代价是 strategic capability 被压扁了。在 bargaining 这种 setting 下，直接表现为 suboptimal surplus extraction。

如果你让 GPT-4o 去谈合同，它可能会系统性地 overcompromise，对方要什么它给什么。

## 为什么贝叶斯不能直接用

贝叶斯赚得最多，但它的策略是 extractive——少给多要，被拒率高。

在 one-shot game 里这是最优的。但在 real-world 里，谈判往往是 repeated interaction。你今天宰了对方一刀，明天人家就不跟你玩了。trust、reciprocity、long-term relationship 这些东西，贝叶斯完全没建模。

Paper 里说得很直白：贝叶斯的 extractive strategy "might fail in real-world negotiations, especially in repeated interactions requiring trust and reciprocity"。

## Hybrid 是出路吗

Paper 最后提了一句 hybrid 方向：LLM 当 System 1，Bayesian / planner 当 System 2。

LLM 负责快速、通用、socially calibrated 的判断；Bayesian 负责慢、精确、task-specific 的优化和 lookahead。

但 paper 也说了，negotiation 不仅仅是 planning 问题，还要 model 别人的 intentions、social norms、cooperation 的潜规则。没有 explicit 的 social reasoning，hybrid 也搞不定人类直觉里那些 social nuances。

## 最深的 insight

**Outcome parity masks process divergence**。

两个人赚一样的钱，一个靠公平交换建立信任，一个靠剥削对方但被容忍。从 outcome 看一样，从 process 看完全不同。而 process 决定了 real-world deployment 能不能持续。

这跟现在整个 AI agent evaluation 领域的问题完全打通——光看 win rate、score、surplus 不够，必须看 agent 是怎么做的。尤其是在 social、strategic、multi-agent 的 setting 下，process 比 outcome 更重要。

## 这篇 paper 的价值

1. **方法论**：给了一个 controlled bargaining game，能 apples-to-apples 比较不同 agent
2. **实证**：三类 agent 在相同条件下展现出完全不同的 procedural pattern——human fairness-oriented, LLM concessionary, Bayesian extractive
3. **概念**：outcome parity masks process divergence 这个 thesis 对整个 agent evaluation 有深远影响

用一句话说：光看结果不够，得看过程。过程决定了一个 AI agent 在 real world 里能不能被信任、能不能长期 deploy。

---

参考链接：
- 论文 arXiv: https://arxiv.org/abs/2502.14641
- Deliberate Lab 开源平台: https://github.com/PAIR-code/deliberate-lab
- Crystal Qian 主页: https://crystalqian.github.io/
- Myerson-Satterthwaite 1983: https://www.sciencedirect.com/science/article/pii/0022053183900480
- Lambert & Calandra alignment ceiling: https://arxiv.org/abs/2311.00168

---

# Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining 深度讲解

## 1. Paper 的核心问题和定位

这篇 paper 来自 Crystal Qian (Google DeepMind)、Kehang Zhu (Harvard)、Benjamin Manning (MIT)、John Horton (MIT & NBER) 等人的合作，发表时间应该在 2024 年底到 2025 年初。核心问题非常尖锐：**当我们把 bargaining / negotiation 这类任务 delegate 给 AI agents 时，光看 outcome（surplus 数值）够不够？**

传统的 agent evaluation 基本都是 outcome-based benchmark——比如 win rate、score、surplus。但这篇 paper 想指出的是：两个 agent 可能产生 same aggregate surplus，但 process 完全不同，process 上的差异在 real-world deployment 里恰恰是最关键的 alignment 问题。

这跟 Andrej 你之前在 Yann LeCun 对话、以及你在 "Software Is Changing Again" 演讲里提到的 agent-as-operating-system 的思路高度相关——agent 不只是"任务完成器"，它的 process、它的行为模式会塑造与 human 长期的协作关系。Paper 的 Section 7 末尾那段 "Procedural alignment matters for human-AI interaction" 基本上就是把这个 thesis 直接写出来了。

参考 link：
- Paper 原文（arXiv 应该有）: https://arxiv.org/abs/2502.14641 （这是 "Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining"，作者是同一团队）
- Crystal Qian 的个人主页: https://crystalqian.github.io/
- John Horton 的 homo silicus 工作: https://www.nber.org/papers/w31161

---

## 2. 为什么选 bargaining 作为 testbed？

Bargaining 在经济学里是一个特别好的 stress test，原因有三层：

**(a) Myerson-Satterthwaite impossibility**：这是 1983 年的经典结果，paper Section 4 末尾的 footnote 3 引用了。定理说的是：在 bilateral trade with private valuations 下，不存在一个 mechanism 能同时满足 Pareto efficiency、Bayesian incentive compatibility、individual rationality、budget balance。换句话说，bargaining under incomplete information 本质上是一个 "no free lunch" 的环境，agent 必须 trade off 自私 vs 合作、效率 vs 信息揭示。

直觉上可以这么理解：如果我（卖家）知道你（买家）很想要，我会报高价；但你的"很想要"是 private info，我没法直接观测。要 Pareto efficient，就要把所有 gains from trade 都提取出来；要 incentive compatible，就要让所有人有 truthful reporting 的激励。这两个目标在 Bayesian setting 下不可能同时达成。

Reference: Myerson & Satterthwaite 1983, https://www.sciencedirect.com/science/article/pii/0022053183900480

**(b) Strategic + Social 双重张力**：bargaining 既是 strategic（博弈论）又是 social（公平、信任、reciprocity）。这让 LLM 和 Bayesian 的本质差异暴露出来——Bayesian 优化的是 narrow objective，LLM 携带的是 pretraining data 里 social norms 的 prior。

**(c) 动态、多 agent**：跟 static QA benchmark 不一样，bargaining 是 multi-turn、multi-agent，这正好是最近 Raman et al. 2024 (STEER)、Goktas et al. 2025 (Strategic Foundation Models) 这些 paper 呼吁要做的 evaluation paradigm。

参考：
- STEER: https://arxiv.org/abs/2402.09552
- Strategic Foundation Models: https://hal.science/hal-04925309

---

## 3. Game 设计的细节

Section 3 的 game 设计非常精巧，我来拆开讲：

**Setup**：
- 3 个 player
- 9 turns（3 rounds × 3 turns/round，每人每 round 提一次 proposal）
- 每个 player 初始有 10 个 each color 的 chips
- Green chip 是 numeraire（计价单位），所有 player 都给它 $0.50 的 valuation
- 其他颜色 chips（red, blue, purple）的 private valuation 从 Uniform[0.10, 1.00] 里抽取
- 复杂度变化：2-chip (green + red), 3-chip (+blue), 4-chip (+purple)

**Gameplay**：
- 固定随机 turn order
- 提案者 propose "give X 个 color A chips, receive Y 个 color B chips"
- 两个非提案者 simultaneously 决定 accept/decline
- 如果两人都 accept，random 选一个 clear trade
- 只能 propose/accept 自己 inventory 够的 trade
- 所有 player 观测 transactions 和 chip holdings，但观测不到别人的 valuations

这个设计的妙处在于：
1. **Identical conditions**：每个 human game 都用相同的 initial endowment 跑一次 LLM 和 Bayesian，做到 apples-to-apples comparison。这是 N=144 games × 3 populations 的 controlled setup。
2. **Private valuations + public holdings**：标准 Bayesian game 的 setup，但加了一个 numeraire 让价值有 anchor。
3. **Simultaneous accept/decline**：避免先动优势，让 strategic reasoning 更难。

---

## 4. Pareto Upper Bound 的数学

Section 4 给了一个 linear programming upper bound，这是后续 empirical results 的参照系。我来把公式逐符号拆解：

### 4.1 Notation

游戏定义为 $\mathbf{M} = (\mathbf{I}, \mathbf{G}, \mathbf{v}, \mathbf{a})$：
- $\mathbf{I}$: agent 集合（这里 |I|=3）
- $\mathbf{G}$: chip 颜色集合（2、3 或 4）
- $\mathbf{v}$: 所有 valuation $v_{ig}$，下标 $i$ 是 agent index，$g$ 是 good index
- $\mathbf{a}$: 所有 allocation $a_{ig}$

**Individual welfare**：
$$w_i = \sum_{g \in G} v_{ig} \cdot a_{ig}$$
- $w_i$: agent $i$ 的 welfare（最终持有 chips 对他而言的总价值）
- $v_{ig}$: agent $i$ 对 good $g$ 的 unit valuation
- $a_{ig}$: agent $i$ 最终持有的 good $g$ 数量
- 求和 over all goods $g$ in $G$

**Total welfare**：
$$\mathbf{w} = \sum_{i \in I} w_i = \sum_{i \in I} \sum_{g \in G} v_{ig} \cdot a_{ig}$$
- 双重求和：先对每个 agent 的所有 goods 求 welfare，再对 all agents 求 total

### 4.2 Optimal Allocation

$$\underset{\mathbf{A}}{\operatorname{argmax}} \sum_{i \in I} \sum_{g \in G} v_{ig} \cdot a_{ig}$$

subject to:
- **(i) Conservation of goods**: $\sum_{i \in I} a_{ig} = \sum_{i \in I} a_{ig}^0, \forall g \in G$
  - 所有 agent 持有的 good $g$ 总量必须等于初始总量 $a_{ig}^0$，上标 0 表示 initial endowment
  - $\forall g \in G$ 表示对所有 good 都成立

- **(ii) Pareto improvement**: $\sum_{g \in G} v_{ig} a_{ig} \geq \sum_{g \in G} v_{ig} a_{ig}^0, \forall i \in I$
  - 每个 agent 的 final welfare ≥ initial welfare
  - 这是 Pareto 条件：没人变差

- **(iii) Non-negativity**: $a_{ig} \geq 0, \forall i, g$
  - 持有量不能为负

**Complexity**: $O((|I||G|)^{3.5})$，这是 interior point method 的标准 bound。$|I|=3, |G|\leq 4$，所以这个 LP 非常小，可以瞬间求解。

**重要 caveat**: 这个 upper bound 在 practice 里几乎不可能达到，因为 decentralized bargaining with private info 没法保证 incentive compatible + Pareto efficient（Myerson-Satterthwaite）。所以 paper 里 Bayesian 在 2-chip game 达到 74% of Pareto 已经是非常高的数字。

### Appendix A.1 的 No Dominant Strategy 证明

这是个挺优雅的小证明，我来捋一下直觉：

**Claim**: 没有 agent 在 game $\mathbf{M}$ 里有 dominant strategy。

**Setup 关键变量**：
- $\mathcal{S}_i$: agent $i$ 的 strategy space
- $s_i \in \mathcal{S}_i$: 一个 pure strategy（完整描述 $i$ 在所有可能 game state 下如何 propose/accept/decline）
- $\mathbf{s}_{-i}$: 除 $i$ 外所有 agent 的 strategy profile
- $u_i(s_i, \mathbf{s}_{-i})$: $i$ 在该 strategy profile 下的 payoff

**Dominant strategy 定义**：$s_i^*$ 是 dominant 如果：
$$u_i(s_i^*, \mathbf{s}_{-i}) \geq u_i(s_i, \mathbf{s}_{-i}) \quad \forall s_i \in \mathcal{S}_i, \forall \mathbf{s}_{-i} \in \mathcal{S}_{-i}$$

**Proof by contradiction**：假设存在 dominant strategy $s_i^*$。构造两个 scenarios $\mathbf{s}_{-i}^A$ 和 $\mathbf{s}_{-i}^B$，它们在 (1) 其他 agent 的 valuations $\{v_{jg}\}$ 和 (2) 其他 agent 的 trade demands/offers 上不同。

- Scenario A: 别人愿意把 good $g^*$ trade 给 $i$ only if $i$ 出高价。存在 specialized strategy $s_i^A$ 使得 $u_i(s_i^A, \mathbf{s}_{-i}^A) > u_i(s_i^*, \mathbf{s}_{-i}^A)$
- Scenario B: 别人不再看重 $g^*$，看重别的 good。存在 $s_i^B$ 使得 $u_i(s_i^B, \mathbf{s}_{-i}^B) > u_i(s_i^*, \mathbf{s}_{-i}^B)$

**Contradiction**: 不存在单一 $s_i^*$ 能同时 dominate $s_i^A$ 在 scenario A 和 $s_i^B$ 在 scenario B。

这个证明其实说明了一个非常深的点：**在这个 game 里，没有任何"通用最优"策略，最优策略依赖于对手的 valuations 和 strategies**。这恰恰是 Bayesian agent 需要不断 update belief 的原因——它必须从观测到的 accept/reject 行为里 infer 别人的 valuations。

---

## 5. 三类 Agents 的设计

### 5.1 Human subjects

- N=216，Prolific 平台招募，US-based
- 在 Deliberate Lab 上跑（Google PAIR 的开源平台）
- 平均 payout $123.24 ± $56.12（含 $4 base payment + surplus-based bonus）
- 每个 participant 完成 2 games
- 3 种 game variant（2/3/4-chip），每 variant 24 groups × 2 games = 144 effective games

**关键**: 为了 incentive alignment，bonus 直接等于 final chip surplus。这是真金白银的 economic experiment 而不是 survey。

Deliberate Lab 链接: https://github.com/PAIR-code/deliberate-lab

### 5.2 LLM agents

两个 model families：
- OpenAI GPT-4o (https://arxiv.org/abs/2303.08774)
- Google Gemini 1.5 Pro (https://arxiv.org/abs/2403.05530)

两种 agent type：
- **Out-of-box**: 用 human instructions 改写的简单 prompt + chain-of-thought
- **Refined**: 生成多个候选 proposal，再用 LLM 选择最好的一个（类似 self-consistency / best-of-N，对应 Furniturewala et al. 2024 的 "thinking fair and slow" 思路）

**Key design choices**:
- Temperature = 0.5（Arora et al. 2024 推荐的 structured reasoning 的 mid-range）
- EDSL library (Expected Parrot DSL) 做 prompt scaffolding (https://www.expectedparrot.com/)
- 每个 LLM agent 在每轮收到跟 human 完全相同的信息：private valuations + public holdings + trade history

### 5.3 Bayesian agents

这是 paper 里最 technically interesting 的部分。每个 agent 维护一个 joint belief $B_i(\mathbf{v}_{-i})$ over 其他 agent 的 valuations。

**核心 optimization（公式 1）**：

$$\max_{(\mathbf{x}, \mathbf{y})} \sum_{\mathbf{v}_{-i}} \mathbf{1}\{\exists j \neq i: \text{accept}(v_j, x_g, y_r)\} \times \Delta u_i(v_i, x_g, y_r) \times B_i(\mathbf{v}_{-i})$$

逐项拆解：
- $(\mathbf{x}, \mathbf{y})$: trade proposal，$\mathbf{x}$ 是给出的 chips 数量 vector，$\mathbf{y}$ 是要的 chips 数量 vector
- $x_g$: 给出 good $g$ 的数量
- $y_r$: 要 good $r$ 的数量
- $\mathbf{v}_{-i}$: 除 $i$ 之外所有 agent 的 valuations 集合
- $\sum_{\mathbf{v}_{-i}}$: 对所有可能的对手 valuations 组合求和（实际上是对 belief 分布求期望）
- $\mathbf{1}\{\exists j \neq i: \text{accept}(\cdot)\}$: indicator function，至少一个其他 agent $j$ 会接受该 trade
- $\text{accept}(v_j, x_g, y_r)$: agent $j$ 在 valuation $v_j$ 下是否接受
- $\Delta u_i(v_i, x_g, y_r) = v_{i,r} \cdot y_r - v_{i,g} \cdot x_g$: agent $i$ 的 utility change
  - $v_{i,r}$: $i$ 对 good $r$ 的 valuation
  - $v_{i,g}$: $i$ 对 good $g$ 的 valuation
  - $y_r$: 收到的 good $r$ 数量
  - $x_g$: 给出的 good $g$ 数量
- $B_i(\mathbf{v}_{-i})$: $i$ 对其他 agent valuations 的 belief (probability)

**直觉**: Bayesian agent 在做 expected utility maximization，但 expectation 是 over 它对对手 valuations 的 belief，且要考虑 trade 是否会被接受。

**Myopic rationality 假设**: 接受者 $j$ 接受当且仅当 $v_{j,g} \cdot x_g - v_{j,r} \cdot y_r > 0$，即 receiver 给出的 chips 对自己价值 < 收到的 chips 对自己价值。

**Belief update（Bayesian learning）**: 
- Trade accepted: 保留所有跟"接受"一致的 valuation states，discard 其余
- Trade rejected: 保留所有跟"拒绝"一致的 valuation states
- Renormalize 剩下的 probabilities

Algorithm 1 给出了完整的 multi-agent trading algorithm with Bayesian learning。关键 subroutines：
- `SOLVEA*`: proposer optimization（用上面那个公式）+ receiver acceptance check
- `BAYESIANUPDATE`: 根据 accept/reject 更新所有 agent 的 belief

这是经典 Bayesian game with belief updating 的实现，跟 Kramár et al. 2022 (Diplomacy AI, https://www.nature.com/articles/s41467-022-34902-3) 里的 strategic reasoning 模块思路类似。

---

## 6. Results 第一层：Performance

Table 3 是核心数据（scaled final surplus gain，即占 Pareto optimal 的比例）：

| Game | Human | GPT-4o | GPT-4o refined | Gemini | Gemini refined | Bayesian |
|------|-------|--------|----------------|--------|----------------|----------|
| 2-chip | 0.60 (0.06) | 0.69 (0.03) | 0.68 (0.04) | 0.42 (0.04) | 0.45 (0.05) | **0.74 (0.04)** |
| 3-chip | 0.59 (0.04) | 0.62 (0.03) | 0.64 (0.03) | 0.42 (0.03) | 0.43 (0.03) | **0.80 (0.03)** |
| 4-chip | 0.54 (0.03) | 0.54 (0.02) | 0.58 (0.02) | 0.37 (0.02) | 0.33 (0.02) | **0.73 (0.02)** |

**观察**：
1. Bayesian 在所有 complexity 下都最高，但 3-chip 时最接近 Pareto optimal（80%）
2. GPT-4o 在 2-chip 时甚至略高于 human（0.69 vs 0.60），4-chip 持平
3. Gemini 1.5 Pro 显著低于 human（统计上 p<0.001 in 3/4-chip）
4. Refined prompting 帮助有限，主要在 GPT-4o 4-chip 时有显著提升（0.54 → 0.58, p=0.270 边缘显著）

**Table 4 的 t-test p-values 揭示的 pattern**：
- Human vs GPT-4o: 没有显著差异（除了 3-chip 时 GPT-4o refined 显著高于 human, p=0.021）
- Human vs Gemini: 显著低于 human
- Bayesian vs all: 几乎都显著高于所有其他 populations

**关键 insight**: 当 game complexity 上升时（2→3→4 chip），Bayesian 反而在 3-chip 最高（0.80），4-chip 略降（0.73）。而 human 和 LLM 都单调下降。这暗示着 Bayesian 的 task-specific 优化在 medium complexity 最 effective，complexity 太高时 action space 太大也难优化。

Appendix A.2 给了一个 complexity 的 theoretical argument：决策空间以 $O(k^2)$ 增长，因为 $\binom{k}{2}$ 个 chip pairs。Empirically，2/3/4-chip game 在初始配置下的 myopically rational trades 数量分别是 37.1, 120.7, 250.7，比值接近 $\binom{3}{2}/\binom{2}{2} = 3$ 和 $\binom{4}{2}/\binom{2}{2} = 6$。

---

## 7. Results 第二层：Procedural Alignment

这是 paper 的真正贡献。Section 6.2 拆成两个 lens：trading patterns 和 regret minimization。

### 7.1 Trading patterns（Figure 3）

3-chip game 的 trade space 可视化，x 轴是 proposer 的 net surplus change，y 轴是 trade ratio（give chips / get chips）。

**Humans**:
- Trade ratio 聚集在 1:1 line 附近（horizontal solid line）——人类偏好"平衡"的 trade，给多少要多少
- 大多数 proposal 在 vertical balanced value line 右侧（positive net surplus to proposer）
- 这是经典 fairness norm 行为，跟 Danz et al. 2022 (https://www.aeaweb.org/articles?id=10.1257/aer.112.9.2851) 的 belief elicitation 工作一致

**LLMs (GPT-4o & Gemini)**:
- Trade ratio 高度分散，没有 anchoring
- **Vertical tail above 1:1 line**: LLM 倾向于 give more than request（最高 5:1）——这是"concessionary posture"
- 大量 proposal 在 vertical balanced value line 附近（little total surplus generated）
- 甚至有 net surplus loss 的 proposal

**Bayesian**:
- 严格 maximize surplus，没有 net-loss trades（所有点在 vertical line 右侧）
- Trade ratio 集中在 parity line 下方（ratio < 1）——ask for more than give
- 高 rejection rate（红色点很多）

Table 5 的 summary statistics 印证了这些观察。比如 3-chip game：
- Human accepted trades: surplus mean = 1.211, ratio mean = 1.039
- GPT-4o accepted: surplus mean = 0.437, ratio mean = 1.765（注意 ratio > 1 表示 give more）
- Bayesian accepted: surplus mean = 3.173, ratio mean = 1.075
- Bayesian rejected: surplus mean = 3.385（即使被拒，proposed surplus 很高——aggressive 但 sometimes rejected）

### 7.2 Regret minimization（Section 6.2.2, Figure 4-5）

这是 paper 里最 sophisticated 的分析。Regret 分三类：

1. **No regret**: 当前 action 是最优的，后续没有更好的 option
2. **Forced regret**: 后续有更好的 option，但之前的 action（比如 premature trade 或 overcommit inventory）阻止了你抓住它
3. **Unforced regret**: 后续有更好的 option，且你本可以抓住，但你没抓住（决策错误）

**Counterfactual analysis**: 用 simulation 重建 "如果做了不同决策" 的轨迹，跟 actual trajectory 比。Figure 9 给了三个例子：
- No regret (P2 proposer, Turn 9): actual = 15.3 vs counterfactual = 14.9
- Forced regret (P3 acceptor, Turn 1→2): actual = 10.8 vs counterfactual = 11.0（如果 P3 拒了 Turn 1 的 trade，本可以接 Turn 2 更好的 offer，但 inventory 不够）
- Unforced regret (P2 decliner, Turn 8→9): actual = 14.9 vs counterfactual = 15.0（P2 拒了一个 profitable trade）

**Scoping decisions**: paper 做了三个合理 simplification：
1. 只考虑 myopically rational transactions（positive surplus for proposer/acceptor/decliner）
2. 评估 proposer 时只看 accepted trades
3. "Acceptors" 包括所有 intended to accept 的人，不只被选中的那个

**Figure 5 的关键 finding**:
- Bayesian 随 complexity 上升，no-regret 比例增加，regrettable 减少——它 more optimal as game gets harder
- Human 和 LLM 反过来——complexity 上升时 regrettable 增加
- Bayesian 有最多的 declines（包括 regrettable 和 non-regrettable）——它 propose 自己占优的 trade，不在意是否被接受
- Human 和 LLM declines 少——propose 更 agreeable 的 trade

**最深的 insight**: Figure 5 + Figure 2 合起来看，Bayesian 在 complexity 高时表现相对变好，**不是因为它变好了，而是 LLM 和 human 表现变差了**。这是一个非常重要的 framing——"Bayesian 在 hard setting 下 win" 其实是 "其他人在 hard setting 下塌方" 的另一种说法。

---

## 8. Discussion 的核心论点

Section 7 的 discussion 我觉得是 paper 最有价值的部分，给三类 agents 各画了一个 character sketch：

### Humans
- 体现 economic + social reasoning 的混合
- Propose balanced trade ratio（fairness norm）
- Post-game survey 强调 fairness 和 cooperation
- 在 one-shot game 里依然 prioritize social norms（虽然 reputation effect 不该起作用）
- **挑战**: 这些 nuanced social motivations 很难 formalize 成 algorithm

### LLMs
- GPT-4o 能达到 human-level surplus
- 但行为是 "concessionary by design"——倾向 value-balanced proposal，avoid friction
- 高 forced regret——缺乏 strategic foresight
- 原因猜测（paper 列了三个）：
  1. Pretraining data 里 cooperative dialogue 被隐式 reward（Wei et al. 2022）
  2. Risk-averse, passive response 倾向 + 缺少 outcome-driven feedback（Ouyang et al. 2022 RLHF paper）
  3. 学到的 aversion to asymmetric outcomes——慷慨 offer 作为 coordination strategy（Fisher et al. 2011 "Getting to Yes"）
- **风险**: adversarial / zero-sum 谈判里 LLM 会被系统性 overcompromise

这跟 RLHF 的 alignment tax 论点（Lambert & Calandra 2023, "The Alignment Ceiling", https://arxiv.org/abs/2311.00168）非常一致——RLHF 让 model 更 safe/cooperative，但代价是 strategic capability。

### Bayesian
- Surplus maximization 强，但 extractive + rejection-tolerant
- Hand-crafted algorithm 特化为这个 game 的 incentives
- **风险**: extractive strategy 在 real-world repeated interaction 里会失败（trust / reciprocity 问题）
- 扩展到 socially compatible 需要更复杂的 preference / fairness model，会牺牲 robustness

### Hybrid 的 promise
- LLM + Bayesian tool / planning module
- 但 negotiation 不仅是 planning——还要 model others' intentions、social norms
- 没有 explicit social reasoning，hybrid 也很难 navigate human instinctively 处理的 social nuances

---

## 9. 跟当前 research landscape 的关联

### 9.1 Multi-agent LLM 路线
- Bakhtin et al. 2022 (Diplomacy, Cicero, https://www.science.org/doi/10.1126/science.ade9097) — language model + strategic reasoning
- Kramár et al. 2022 (Nature Communications, https://www.nature.com/articles/s41467-022-34902-3) — same Diplomacy line
- Fish et al. 2024 (algorithmic collusion by LLMs, https://arxiv.org/abs/2404.00806) — LLM 在 pricing 里 collude
- Soumalias et al. 2025 (LLM preference elicitation, https://arxiv.org/abs/2502.10308)
- Tessler et al. 2024 (AI helps find common ground, Science, https://www.science.org/doi/10.1126/science.adq2852)

### 9.2 LLM-as-economic-agent 路线
- Horton 2023 (homo silicus, https://www.nber.org/papers/w31161) — LLM 作为 simulated economic agents
- Aher et al. 2022 (https://arxiv.org/abs/2208.10264) — LLM simulate multiple humans
- Argyle et al. 2022 (https://arxiv.org/abs/2209.06899) — out of one, many
- Manning, Zhu, Horton 2024 (automated social science, https://www.nber.org/papers/w32322)
- Zhu et al. 2024 (LLMs as auction participants, https://openreview.net/forum?id=FB9mTtJpJI)

### 9.3 Procedural alignment / AI ethics 路线
- Mozannar et al. 2023 (learning to defer, https://arxiv.org/abs/2301.06197)
- Mullainathan & Obermeyer 2021 (diagnosing physician error, https://doi.org/10.1093/qje/qjab046)
- Palminteri, Garcia, Qian 2024 (moral Turing test, https://osf.io/preprints/psyarxiv/ct6rx_v1)
- Kapania et al. 2022 (user attitudes to AI authority in India, https://doi.org/10.1145/3491102.3517533)
- Qian & Wexler 2024 (Take it, leave it, or fix it, https://doi.org/10.1145/3640543.3645198) — Crystal Qian 之前的工作，跟这篇一脉相承

### 9.4 Small model behavior（Appendix F）
Paper 在 Appendix F 测试了 GPT-4o-mini 和 Gemini 2.5 Flash，发现这些 small models 有 "lazy generation" 效应（Lambert & Calandra 2023）—— ~90% 的 proposal 都是 1:1 trade（比如 1 red for 1 green）。这跟 Xia et al. 2024 (https://aclanthology.org/2024.findings-acl.213) 对 sub-10B bargaining agents 的发现一致。这暗示着 strategic bargaining 可能是 emergent capability 里的 phase transition，small models 卡在 template-driven 的简单 strategy 上。

---

## 10. 我个人的 takeaways 和延伸联想

### 10.1 "Outcome parity masks process divergence" 是一个 deep problem

这个 paper 的核心 thesis 跟最近 RLHF / agent evaluation 的讨论完全打通。现在大家都在讲 "agents can reach human-level on X"，但很少人问"是用什么方式 reach 的"。这篇 paper 提供了一个 concrete 的方法论：用 controlled experimental design + procedural metrics（trading pattern, regret classification）来 reveal process-level differences。

这跟 Andrej 你在 Twitter/X 上多次提到的 "RLHF overemphasizes certain kinds of behaviors, leads to sycophancy" 的观察完全吻合。GPT-4o 在这里 concessionary 的倾向，跟 RLHF model 的 sycophancy 本质上是同一类问题——optimization for agreeableness over correctness/optimality。

### 10.2 LLM concessionary behavior 的 root cause 值得深挖

Paper 列了三个 hypothesis：
1. Pretraining data bias（cooperative dialogue）
2. Risk aversion + 缺 outcome feedback
3. 学到的 asymmetry aversion

我怀疑还有第四个：**RLHF 的 distributional shift**。RLHF 的 reward model 通常在 pairwise comparison 上 train，human rater 倾向 prefer "helpful, polite" 的 response，这会 systematically penalize aggressive / extractive behavior。在 bargaining 这种 setting 下，这种 alignment tax 直接表现为 suboptimal surplus extraction。

这是 Anthropic、OpenAI、DeepMind 都在 struggle 的问题——how to align models to be both socially calibrated AND strategically capable。这篇 paper 提供了 measurement framework。

### 10.3 Bayesian + LLM hybrid 是 LLM agent 的下一个 phase

Paper Section 7 的 hybrid 段落很短但很关键。我认为这指向了一个重要的 architectural 方向：

**LLM 作为 "System 1"**：fast, generalizable, socially calibrated
**Bayesian / symbolic planner 作为 "System 2"**：slow, task-specific optimization, lookahead

这跟你之前在 "State of GPT" 演讲里讲的 System 1 / System 2 framing 完全一致。当前的 LLM agents（ReAct, Reflexion, etc.）试图用 LLM 做 System 2，但这篇 paper 显示 LLM 在 strategic lookahead 上确实有 hard limit（高 forced regret）。Hybrid architecture 可能是必经之路。

参考：
- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366

### 10.4 Procedural alignment 跟 AI safety 的 deep connection

这篇 paper 表面上是 economics experiment，但底层其实是 AI safety 的论点。如果我们 deploy LLM 作为 negotiator（比如 supply chain, contract, diplomacy），它的 process 会塑造 counterparty 对 AI 的 trust 和 long-term relationship。一个总是 concessionary 的 LLM 会让人类 counterparty 学会 exploitative behavior；一个总是 extractive 的 Bayesian 会让人类 refuse to trade。

这跟 Anthropic 的 "Sleeper Agents" paper (https://arxiv.org/abs/2401.05566)、Evan Hubinger 的 "inner alignment" 工作、以及 DeepMind 的 "Spec Games" / "Open-ended learning" 路线都有关联——agent 的 process 决定了它跟环境（包括其他 agents）co-evolution 的 trajectory。

### 10.5 限制和 future work 的方向

Paper 自己列了几个 limitations：
1. Single stylized game, static valuations, single-shot（这些条件反而让 Bayesian 占优）
2. LLM 用 minimal prompt（没探索 advanced prompting / fine-tuning）
3. 没解开 "behavioral differences 是 reasoning limits、social norms、还是 inductive biases"
4. 没有 human perception / trust study
5. 没用 natural language（agent 之间是 structured protocol）

我觉得最 promising 的 future direction：
- **Repeated game with reputation**: 让 agent 跟同一批对手玩多次，看 concessionary vs extractive 哪个 long-run 更优
- **Mixed-population games**: human + LLM + Bayesian 混合，看 emergent dynamics
- **Natural language bargaining**: 加 NL communication，这是 LLM 的强项但 Bayesian 的弱项，可能完全改变 ranking
- **Procedural alignment steering**: 设计 RLHF 或 DPO 的 reward 显式包含 procedural metrics（fairness, reciprocity），看能不能 shift LLM 的 concessionary tendency

---

## 11. 公式和算法的 engineering intuition

最后我想给一个 engineering-level 的 intuition，关于这三类 agents 实际怎么跑：

**Human**: 30 分钟 session，2 games，平均 $123 payout。Cognitive load 高，会 fatigue。

**LLM**: API call per turn，每 turn 大概 1-2 秒 latency。Prompt 包括：
- Private valuations（preference description）
- 当前 chip holdings
- Conversation history（之前所有 turns 的 proposals + outcomes）
- Structured output tags (`<GET_COLOR>`, `<GET_QUANTITY>`, etc.)

Refined version: 一次生成 3 个候选 proposal，再用一次 LLM call 选择最好的——这是 Best-of-3 with LLM judge 的简化版。

**Bayesian**: 完全 determinstic 的 Python algorithm。
- Prior: 每个 opponent 的 valuation 是 Uniform[0.10, 1.00] 的 discretized distribution
- Each turn: enumerate all feasible (x, y) pairs, compute expected utility per Equation 1, pick argmax
- After each observed accept/reject: update belief via rejection sampling on valuation states
- Complexity per turn: $O(|\text{feasible trades}| \times |\text{valuation states}|)$

这个 engineering 对比揭示了一个关键的 trade-off：
- **Bayesian**: 高 performance, 高 task-specific engineering cost, 低 generalization
- **LLM**: 中 performance, 零 task-specific engineering, 高 generalization, 但 procedural alignment 偏向 concessionary
- **Human**: 中 performance, 高 social calibration, 高 cognitive cost

这种 trade-off 跟 "capabilities vs alignment" 的讨论完全平行——Bayesian 有 capability 没 (social) alignment，LLM 有 (over-)alignment 没足够 strategic capability，human 在中间但贵且慢。

---

## 12. 总结

这篇 paper 的贡献有三层：

1. **Methodological**: 提供了一个 controlled bargaining game design，能 apples-to-apples 比较 human / LLM / Bayesian。Deliberate Lab 平台开源，可复现。

2. **Empirical**: 三类 agents 在相同条件下显示出 dramatically 不同的 procedural patterns——human fairness-oriented, LLM concessionary, Bayesian extractive。即使 outcome（surplus）相似，process 完全不同。

3. **Conceptual**: "Outcome parity masks process divergence" 这个 thesis 对整个 AI agent evaluation 领域有深远影响。未来的 agent benchmark 不能只看 win rate / score，必须看 process metrics——尤其是 social / strategic / multi-agent setting 下。

跟 Andrej 你一直强调的 "evaluation is the hardest part of AI progress" 完全一致。这篇 paper 给了一个具体的 evaluation methodology 的范本。

Reference 汇总：
- Paper arXiv: https://arxiv.org/abs/2502.14641
- Deliberate Lab: https://github.com/PAIR-code/deliberate-lab
- Crystal Qian 主页: https://crystalqian.github.io/
- John Horton homo silicus: https://www.nber.org/papers/w31161
- Manning, Zhu, Horton automated social science: https://www.nber.org/papers/w32322
- Myerson-Satterthwaite 1983: https://www.sciencedirect.com/science/article/pii/0022053183900480
- Raman et al. STEER: https://arxiv.org/abs/2402.09552
- Goktas et al. Strategic Foundation Models: https://hal.science/hal-04925309
- Lambert & Calandra alignment ceiling: https://arxiv.org/abs/2311.00168
- Bakhtin et al. Cicero: https://www.science.org/doi/10.1126/science.ade9097
- Fish et al. LLM collusion: https://arxiv.org/abs/2404.00806
- Anthropic sleeper agents: https://arxiv.org/abs/2401.05566
- ReAct: https://arxiv.org/abs/2210.03629
- Xia et al. bargaining benchmark: https://aclanthology.org/2024.findings-acl.213
