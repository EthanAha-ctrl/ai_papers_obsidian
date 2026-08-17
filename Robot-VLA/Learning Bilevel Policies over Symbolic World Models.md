---
source_pdf: Learning Bilevel Policies over Symbolic World Models.pdf
paper_sha256: 2ecd83d267e56dc67cc19b7d2521f81c49c3e3980deeda465d365598ababa5a6
processed_at: '2026-08-05T12:45:37-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# BISON 用人话说

Karpathy 我换个更口语的方式讲，重点放在 "为什么这样做 work" 而不是公式堆砌。

---

## 一句话版

**让 symbolic rule 做 reasoning（可 scale 到 1 万个物体），让小 GNN 做 motor control（33k 参数就够），两者通过一个 labelling function 耦合。reasoning 和 control 在 representation 上彻底分离 —— 这就是它能 scale 的根本。**

---

## 为什么这件事 difficult

现在大家都在追 VLA、end-to-end，$\pi_0$、RT-2、OpenVLA、SmolVLA，model 越来越大，motor control 越来越丝滑。但只要 task horizon 一长、object 一多，就歇菜。SmolVLA 在这篇 paper 的 benchmark 上 **直接 0% success rate**，连 fine-tune 在 MetaWorld 上的 SmolVLA$^{MW}$ 都是 0%。原因很 fundamental —— transformer 在 multi-step compositional reasoning 上有天花板（Dziri NeurIPS'23），GNN 在 long-horizon planning 上 extrapolate 不动（WL test limit，Ståhlberg ICAPS'22）。

另一头，classical planner（Fast Downward、LAMA、PRP）做 long-horizon 很猛，但死在两件事上：(a) 需要 closed-world PDDL 假设（不在 state 里的 fact 就是 false），(b) 只能处理离散 symbol，接不上 continuous sensory。这篇 paper 的 Gacha 环境 open-world —— 你不知道 gacha box 会吐什么颜色 block，closed-world planner 直接死。

之前一堆 TAMP 工作（Garrett 2021 review）做法是 **先 search 一个 HL plan，再 LL track**。问题：依赖 search、多数用 closed-world、难处理 partial observability，replan 又慢。

BISON 反过来 —— **不 search，直接 induct 一个 problem-agnostic 的 HL policy**。

---

## 核心 idea —— 两个 representation 的分工

你要 build intuition 的话，最关键一点是 **representation 决定 generalisation bound**：

- HL policy 的 representation 是 **first-order rule**（带变量的 condition→action），比如 `∃x, l. holding(x), clear(l), at^G(x, l) → place(x, l)`。变量 $x, l$ 让 rule 天然 generalise 到任意 object 数 —— 不管有 3 个 block 还是 10000 个 block，rule 一字不改。
- LL policy 的 representation 是 **小 GNN**（<33k 参数），只学 motor skill —— 给定一个 HL action 比如 `pick(block_3, loc_7)`，GNN 输出机械臂下一步 joint torque 或 end-effector delta。

**reasoning 的 complexity bound 在 schema arity 和 subproblem size 上，不 bound 在 object 数上** —— 这是 Theorem 1 的本质。control 的 complexity bound 在 motor skill 数量上（等于 HL action schema 数量），也不 bound 在 object 数上。两边都摆脱了 "object 数量" 这个 dimension。

对比一下 PureNN / PddlNN 这俩 end-to-end baseline：它们把所有 HL facts 也塞进 GNN 让它自己学 HL policy。结果在 n>6 object 上 performance drop，因为 GNN 的 expressive power 受 WL test 限制，extrapolate 不到更长 horizon。BISON 把 reasoning 推给 symbolic，GNN 只学 skill，generalise 得远得多。

---

## HL policy 怎么学 —— 真正的 novelty

分 3 步，每步都很简单：

### Step 1: 把 LL demo 压成 HL trace

你有一串 LL demo：机械臂每一步的 pose、action。通过 labelling function $\mathcal{L}$ 把 LL state 映射成 symbolic facts（`holding(block_3)`, `at(block_3, loc_7)` 等）。找出 abstract state 变更点 —— 即 $\mathcal{L}(\mathbf{s}_j^{ll}) \neq \mathcal{L}(\mathbf{s}_{j-1}^{ll})$ 的点 —— 把 demo 切成一段段 HL action。结果是个 sparse HL action sequence。

这一步本质上在做 **temporal abstraction**：把 200 步的 LL trajectory 压成 5 步的 HL action sequence。

### Step 2: Goal regression —— 倒推 rule

这是核心。Goal regression 问的是：**"在 action $a$ 执行前，state 要满足什么条件，才能保证执行后达到 goal $g$？"**

公式（4）：
$$regr(g, a) = \{(g \setminus add_i(a)) \cup pre(a) \mid i = 1, \ldots, n\}$$

用人话说：
- $g \setminus add_i(a)$：action 的第 i 个 outcome 会自动 add 的 facts 从 goal 里拿掉 —— 剩下的是 action 帮不上忙的，必须执行前就满足
- $\cup pre(a)$：还要加 action 自己的 precondition，让 action applicable
- 对每个 nondeterministic outcome $i$ 算一个 regressor，全部收集

举个具体例子：goal 是 `at(block, loc_B)`，action 是 `move(loc_A, loc_B)`，$add=\{rAt(loc_B)\}$, $del=\{rAt(loc_A)\}$, $pre=\{rAt(loc_A)\}$。
$$regr(\{at(block, loc_B)\}, move(loc_A, loc_B)) = \{at(block, loc_B)\} \cup \{rAt(loc_A)\} = \{at(block, loc_B), rAt(loc_A)\}$$
意思：block 已经在 goal 位置（move 不动 block），机器人要在 loc_A。

Algorithm 1 是 reverse iteration over HL trajectory —— 从最后一个 action 倒着 regress。每 reg一步产生一个 condition→action rule，priority 等于 "到 goal 的距离"。这让 HL policy 像 backward-chaining planner：始终先推离 goal 最近的 subgoal。

### Step 3: Lift 到 first-order

很简单：把所有 ground objects 换成 fresh variables。`holding(block_3), clear(loc_7), at^G(block_3, loc_7) → place(block_3, loc_7)` 变成 `∃x, l. holding(x), clear(l), at^G(x, l) → place(x, l)`。

这一步让 rule 脱离 specific demo instance，generalise 到任意 object。

---

## 为什么 generalisation 有保证 —— Theorem 1

两个 assumption 串起来：

**Assumption A: Goal Independence (GI)** —— goal 的每个 fact 可以独立按任意 order 达成，且实现 $g_i$ 时不 delete 其他 $g_j$。比如 "把 n 个 block 放到各自 goal 位置"，每个 block 独立，先放谁后放谁都行。这在 real-world 是 common case（Simon 1956 [122], Korf 1987 [72]）。

**Assumption B: C-bounded** —— 每个singleton-goal subproblem 至多 $C$ 步能解。比如 pick-and-place 里 $C=2$（pick + place）。

**Assumption C: Object-Renaming Equivalence** —— 两个 HL problem 如果只差 object 重命名（block_1↔block_5），它们的 lifted rule 可以直接 transfer。这个是 first-order rule 的天然性质。

证明思路：
1. Up to object renaming，singleton-goal subproblems 种类有限 —— 因为 singleton goal 由 predicate choice + argument positions 决定，共 $|\mathcal{P}| \cdot M^M$ 种（$M$ = max arity）。
2. 长 $\leq C$ 的 action sequence + singleton goal 提及 $\leq kN+M$ 个 objects，所以 inequivalent rules 数有上限（公式 8）。
3. 取有限 $\mathbb{T}$ 覆盖每个等价类 representative。
4. 任意 test problem 按 GI 拆成 singleton subproblem 序列，每个 subproblem 由 Proposition 1 保证有 applicable rule。

**Intuition**：first-order rule 的变量让 single demo 覆盖所有 renaming-equivalent problems；GI 让 problem 可以 piecewise solve；bounded C 让 rule 数量有限。三个一起 gives zero-shot generalisation to arbitrary object count。

注意 complexity 在 $C$ 和 $N$（schema arity）上 exponential，但**不 bound 在 object 数上**。实际中很少 demo 就够，因为很多 subproblem 共享 rule。

---

## LL policy 怎么学 —— 一个极简 GNN

输入 encoding 很关键。给定 $\mathbf{s}^{ll}$（agent state + 所有 object 相对 state）、$a^{hl} = a(o_1, \ldots, o_n)$、$g^{hl}$，构造 graph：

- **Global node**：agent state $\mathbf{x}_e$ + nullary predicates indicator
- **Action node**：action schema one-hot
- **Object node $o_i$**：object 相对 state $\mathbf{x}_{o_i}$ + 涉及的 unary predicates + **position one-hot in action**（$o_i$ 是 action 第几个 arg）

这个 position one-hot 是关键 —— 让 GNN 知道这个 object 在当前 HL action 里扮演什么角色。`pick(obj, loc)` 里的 obj 和 loc 用同一个 GNN 处理，但 position encoding 不同，路由到不同的 motor skill。

GNN 架构超简单：2 层 message passing，hidden=64，参数 <33k。message passing 是 fully connected（所有 node 互相 aggregate，element-wise max pooling 保持 permutation invariant）。Readout 是 feed-forward 输出 LL action。

训练就是 BC：从 demo 抽 $\langle \mathbf{s}^{ll}, a^{hl}, g^{hl}, \mathbf{a}^{ll}\rangle$ tuples，MSE loss。200 iter, Adam, lr=1e-3。

**Intuition**：LL policy 是 "skill library"。每个 HL action schema 对应一个 LL controller。HL action + object position one-hot 把 GNN 路由到对应 skill。HL reasoning 完全外化到 symbolic policy，GNN 不背 reasoning 的锅。所以 GNN 这么小就够。

---

## 实验关键数据

8 个环境，21,600 episodes，baselines 跨 VLA / end-to-end / planning：

- SmolVLA / SmolVLA$^{MW}$: **0% 全部** —— 即使在 MetaWorld 上 fine-tune 过
- PureNN（GNN no HL info）: 0%
- PddlNN（GNN with HL facts but no HL action）: 30%
- DetPlan（single-shot deterministic planner）: 15%
- DetReplan（replan on failure）: 44%
- NdtReplan（nondeterministic + replan）: 48%
- **BISON: 79%**

最 impressive 的几个点：

**(a) VLA 全死**：呼应 [117] 的报告，VLA 在 hard MetaWorld 上 ≤45%。long-horizon + open-world 是 VLA 的死穴。

**(b) Generalisation to longer horizon**：BISON 在 n=10 object 仍高 SR，PureNN / PddlNN 在 n>6 后 drop。验证 GNN extrapolation limit，也验证 symbolic rule 的 generalisation 优势。

**(c) Gacha 的 opportunistic behavior**：open-world 环境，gacha 会吐 random color block。BISON 学到 9 个 priority level 的 policy（Appendix C.4）。Rules 6-9 是 **opportunistic actions** —— 当在追求 color $c'$ 但 gacha 吐出另一个 needed color $c$ 的 block 时，立刻把它放到对应 tray，不浪费 lucky roll。这是 search-based planner 难以做出来的 implicit multi-goal coordination。

**(d) Reactive 处理 disturbance**：Blocks$^N$ 里已就位的 block 会 random teleport。BISON 学到的 policy 自然 handle —— 因为 goal predicate `at^G(x, l)` 对被 teleport 走的 block 仍为 true，自动 re-pick。**不需要 explicit replanning**。这是 reactive production system 相比 search-based planner 的优势。

**(e) HL scale 到 10,000 objects**：纯 HL planning problem（忽略 LL execution），BISON 几秒解 10,000 个 block 的 problem，LAMA 在 ~100 个 block 就 timeout 100s。因为 $\pi^{hl}$ 是 reactive rule database，执行是 database query，complexity sub-linear in object count。LAMA 要 full search，explode。

**(f) Efficiency**：BISON training + inference 时间 vs success rate 最优。end-to-end GNN baseline 慢得多（GNN 更大），replanning 方法每次失败要重 search 也慢。

---

## Limitations 和我的思考

论文承认的：
1. **依赖 $\mathcal{D}$ 和 $\mathcal{L}$**：HL abstraction quality 决定一切。这跟 Konidaris "symbol emergence" 路线强耦合，是 bilevel planning paradigm 固有限制。
2. **LL policy 无 optimality 保证**：BC 的 covariate shift，failures 主要来自 LL execution 的 covariate shift。作者建议 DAgger 或 RL post-training。
3. **GI$_C$ 假设强**：强 coupled goal（如 "tower 高度 ≥5"）下不成立。

我额外想到几点：

**(a) $\mathcal{L}$ 的 brittleness**：HL state 完全依赖 $\mathcal{L}$。如果 $\mathcal{L}$ 在 continuous state 边界附近 flip（block 刚 hover 在 goal 上方但没放下，`at(block, loc)` 算 true 吗？），整个 HL reasoning 会 collapse。Symbolic abstraction 的 classic granularity issue。

**(b) NDRP 没 enforce**：论文引入 NDRP 作为 "useful HL abstraction" 的定义，但 training 时没 explicit check NDRP。如果 $\mathcal{L}$ / $\mathcal{D}$ / demos 不 satisfy NDRP，BISON 仍会跑，但可能 silently fail。

**(c) Goal regression 假设 schema complete**：regr 要求 $del_i \cap g = \emptyset$，但 demo 是从真实 LL execution 抽出来的，HL action 效果可能比 declared schema 复杂（`pick` 可能 implicitly 也 delete `clear(top_of_stack)`）。Schema 不 complete 时 regression 可能 miss rules。

**(d) LL skill 切分粒度**：每个 HL action 对应一个 LL skill，但 LL skill 没显式 terminate condition。BISON 靠 $\pi^{hl}$ 每步重新 query 并 implicit terminate（HL state 变了就 trigger 下个 action）。如果 LL execution 在 abstract state 边界附近 oscillate，会 dispatch 切换导致抖动。

**(e) 跟 recent predicate invention 路线的关系**：Silver et al. "Predicate invention for bilevel planning" (AAAI'23)、Liang et al. "VisualPredicator" (ICLR'25)、"ExoPredicator" (ICLR'26) 这些 work 在 trying to auto-learn $\mathcal{D}$ 和 $\mathcal{L}$，BISON 假设它们 given。Combine 起来是 obvious next step —— 学 $\mathcal{D}, \mathcal{L}$ 再学 $\pi^{hl}, \pi^{ll}$ 全栈。

**(f) 跟 LLM-as-planner 的对比**：LLM+P [84] 把 LLM 当 PDDL problem translator + symbolic planner 当 solver。BISON 不用 LLM 做 reasoning，用 first-order rule database 做 backward chaining。Rule database 是 explicit、inspectable、verifiable 的（Appendix C.4 用 LLM 解释 rule 那部分挺 cool）。这是 LLM-as-planner 路线的中间产物 —— reasoning 部分外化到 symbolic structure。

**(g) Reaction vs deliberation trade-off**：$\pi^{hl}$ 是 reactive (priority-ordered rules)，没 lookahead search。在 opportunistic 场景（Gacha）效果反而比 search-based 好（不浪费 lucky roll）。但 in adversarial / irreversible setting（Sokoban）reactive 不够。Hybrid（rule + shallow search）是 future direction。

**(h) 验证 neuro-symbolic 的核心 thesis**：PddlNN baseline encode 所有 HL facts 但不用 HL action，让 GNN 自己学 HL policy。它 30% SR 远不如 BISON 79%。说明 **HL reasoning 用 explicit symbolic form 比 implicit neural encoding 优越**。这是 neuro-symbolic 整个领域的核心 thesis 的一个 strong evidence。

---

## 一句话 TL;DR

**representation 决定 generalisation bound**。BISON 把 reasoning 放在 first-order rule（可 scale 到任意 object 数）上，把 control 放在小 GNN（只学 motor skill）上。reasoning 和 control 在 representation 上彻底分离，让整个系统摆脱了 "object 数量" 这个 scaling dimension。10,000 个 object 几秒解，VLA 全死，end-to-end GNN 在 n>6 撑不住。

你可以把它想成 "SOAR/ACT-R 风格的 production system + 一组神经 motor skills"。Production system 用 goal regression 从 demo induct 出来，motor skills 用 BC 学。整个系统的 reasoning complexity bound 在 schema arity 和 subproblem size 上，不 bound 在 object 数上 —— 这就是它能 scale 到 10,000 objects 的根本原因。

References:
- Project page: https://dillonzchen.github.io/bison
- SmolVLA: https://huggingface.co/lerobot/smolvla_base
- MetaWorld: https://arxiv.org/abs/1910.10897
- Konidaris "From skills to symbols": https://www.jair.org/index.php/jair/article/view/11167
- Silver et al. "Predicate invention for bilevel planning": https://arxiv.org/abs/2302.10734
- Garrett et al. TAMP survey: https://arxiv.org/abs/2102.05170
- Reiter "Knowledge in Action": https://mitpress.mit.edu/9780262681572/knowledge-in-action/

---

# BISON: Learning Bilevel Policies over Symbolic World Models — 深度技术解读

Karpathy 你好，这篇论文正好击中当前 embodied AI 一个核心痛点：**纯 neural 的 VLA / end-to-end 方法做不了 long-horizon planning，而纯 symbolic planner 又 handle 不了 continuous LL dynamics 和 open-world**。作者们走了一条非常干净的中间路线 —— 把 long-horizon reasoning 完全外化到一个 symbolic $\pi^{hl}$ 上，把 motor control 完全留给一个 compact 的 GNN $\pi^{ll}$，二者通过 labelling function $\mathcal{L}$ 耦合。下面我尽量把每一个 building block 拆开来讲清楚，build your intuition。

Project page: https://dillonzchen.github.io/bison

---

## 1. 为什么这件事 difficult —— 现状梳理

近年 scaling 路线（RT-2 [153], PaLM-E [32], $\pi_0$/$\pi_{0.5}$ [57,58], OpenVLA [64], SmolVLA [117]）在 fine motor control 上很猛，但 long-horizon planning 仍走不动，原因可参考 [34, 61, 81, 98]：
- **Compositionality gap**：transformer 在 multi-step compositional reasoning 上有 well-documented limits（Dziri et al. NeurIPS'23 [34]）。
- **Horizon scaling**：Park et al. NeurIPS'25 [98] 指出必须做 horizon reduction 才能 scale RL。

而 symbolic planners（Fast Downward [50], LAMA [105], PRP [96]）做 long-horizon 很强，但需要 closed-world 假设、强结构化的 PDDL domain，且无法直接处理 continuous sensory。

之前一系列 bilevel planning / TAMP 工作（[39, 124, 116, 71]）通常做法是 **search-then-refine**：先用 symbolic planner 搜一个 HL plan，再用 LL controller 跟踪。问题在于：(a) 依赖 search；(b) 多数用 closed-world PDDL；(c) 难处理 partial observability。

BISON 反其道而行之 —— **不 search，直接 induct generalise 出一个 problem-agnostic 的 HL policy**。

---

## 2. Bilevel Planning 的形式化

### 2.1 LL problem $\mathbf{P}^{ll} = \langle \mathbf{S}^{ll}, \mathbf{A}^{ll}, \mathbf{s}_0^{ll}, \mathbf{g}^{ll}\rangle$

LL state 采用 **object-centric + ego-centric** 表示：

$$\mathbf{s}^{ll} = \langle \mathbf{x}_e, \{\mathbf{x}_o\}_{o \in \mathbf{O}}\rangle$$

- $\mathbf{x}_e \in \mathbb{R}^n$：agent 自己的状态向量（机械臂末端 pose 等）
- $\mathbf{O}$：当前观测到的物体集合
- $\mathbf{x}_o \in \mathbb{R}^m$：每个物体相对 agent 的状态向量

注意是 **ego-centric**：物体向量是相对 agent 的，这样在 agent 移动时 representation 自然 shift-invariant，对 GNN generalisation 友好。

### 2.2 HL problem $\mathbf{P}^{hl}$ —— STRIPS-style 但 open-world

Domain $\mathcal{D} = \langle \mathcal{P}, \mathcal{A}\rangle$：
- $\mathcal{P}$：predicates，例如 `at(x,y)`, `holding(x)`, `clear(l)`
- $\mathcal{A}$：action schemata，每个 schema $a = \langle var(a), pre(a), \{add_i(a), del_i(a)\}_{i=1}^n\rangle$，注意是 **nondeterministic** 的（有多个 outcome $i$）

HL state $s^{hl}$ 是一组 ground facts。HL goal $g^{hl}$ 是 facts 子集，$s^{hl}$ 是 goal state iff $g^{hl} \subseteq s^{hl}$。

**Successor 函数**（公式 2）：
$$succ(s^{hl}, a^{hl}) = \{(s^{hl} \setminus del_i(a^{hl})) \cup add_i(a^{hl})\}_{i=1}^n$$

- $s^{hl} \setminus del_i(a^{hl})$：从当前 state 删掉第 i 个 outcome 要 delete 的 facts
- $\cup add_i(a^{hl})$：加上第 i 个 outcome 要 add 的 facts
- 返回 n 个可能后继（nondeterministic branching）

**Open-world 假设**：不在 $s^{hl}$ 里的 fact 是 unknown（不是 false）。这点很关键 —— Gacha 环境里你不知道 gacha 会出什么颜色，closed-world PDDL planner 直接死掉。

### 2.3 Labelling function $\mathcal{L}: \mathbf{S}^{ll} \to \mathbf{S}^{hl}$

把 continuous observation 映射到 symbolic facts。论文里 $\mathcal{L}$ 是给的（可以人工、可以 VLA、可以 learned，参考文献 [70, 71, 6, 7, 20, 121, 60, 118, 120, 113, 139, 54, 76, 8, 145, 116, 80] 一大堆）。这是整个 framework 的 "interfaces"，是关键 bottleneck 之一。

### 2.4 Nondeterministic Downward Refinement Property (NDRP)

经典 DRP [9] 说：HL solution 一定能 refine 成 LL solution。论文扩展到 nondeterministic setting（公式 1）：

$$\forall \mathbf{s} \in exec(\mathbf{s}^{ll}, \pi^{ll}), \quad \mathcal{L}(\mathbf{s}) \in exec(\mathcal{L}(\mathbf{s}^{ll}), \pi^{hl}) \cup \{\mathcal{L}(\mathbf{s}^{ll})\}$$

变量解释：
- $exec(\mathbf{s}^{ll}, \pi^{ll})$：在 $\mathbf{s}^{ll}$ 下按 $\pi^{ll}$ 一步能到达的所有 next states
- $exec(\mathcal{L}(\mathbf{s}^{ll}), \pi^{hl})$：在抽象 state 下按 $\pi^{hl}$ 一步能到的所有抽象后继
- $\{\mathcal{L}(\mathbf{s}^{ll})\}$：原地不动也合法

**Intuition**：LL policy 走一步，要么停留在同一个 HL abstract state（仍在执行当前 HL action 的过程中），要么进入 $\pi^{hl}$ 允许的某个合法 HL 后继。LL policy 不能 "偷偷" 跨到 $\pi^{hl}$ 没预期的 HL state。这正是 skill / option [69] 的 abstraction 概念的 formal 版本。

---

## 3. Bilevel Policy 的 composition（公式 3）

$$\pi(\mathbf{a}^{ll} \mid \mathbf{s}^{ll}, g^{hl}) = \sum_{a^{hl}} \pi^{ll}(\mathbf{a}^{ll} \mid \mathbf{s}^{ll}, a^{hl}, g^{hl}) \cdot \pi^{hl}(a^{hl} \mid \mathcal{L}(\mathbf{s}^{ll}), g^{hl})$$

变量：
- $\pi^{hl}(a^{hl} \mid s^{hl}, g^{hl})$：给定 abstract state 和 goal，HL policy 输出 abstract action 的分布
- $\pi^{ll}(\mathbf{a}^{ll} \mid \mathbf{s}^{ll}, a^{hl}, g^{hl})$：给定 LL state、当前 HL action、goal，LL policy 输出 LL action
- 求和 over $a^{hl}$：marginalise 掉 HL action

$\pi^{hl}$ 实际是 deterministic 的（每次 query 返回唯一 lowest-priority applicable rule），所以 $\pi^{hl}(a^{hl} \mid \cdot) \in \{0, 1\}$。这点很重要 —— 把它当 dispatcher 用，不需要在 HL 层 stochastic 探索。

---

## 4. HL Policy Learning —— 这才是论文真正的 novelty

HL policy 学习分 3 步（Fig. 2）：

### 4.1 Step 1：构造 HL traces

给定 LL trajectory $\langle g_i^{hl}, \mathbf{s}_0^{ll}, \mathbf{a}_0^{ll}, \ldots, \mathbf{s}_m^{ll}, \mathbf{a}_m^{ll}\rangle$，通过 $\mathcal{L}$ 做 segmentation：

- 找出 LL state 中 $\mathcal{L}(\mathbf{s}_j^{ll}) \neq \mathcal{L}(\mathbf{s}_{j-1}^{ll})$ 的点 → HL state 变更点
- 在前后两个 HL state 之间找一个 applicable 的 HL action $a_i^{hl}$ 满足 $pre(a_i^{hl}) \subseteq s_{i-1}^{hl}$ 且某个 outcome 能 produce $s_i^{hl}$

输出 HL trace $\langle g_i^{hl}, a_0^{hl}, \ldots, a_n^{hl}\rangle$。这是把 dense LL demo "压缩" 成 sparse HL action sequence。

### 4.2 Step 2：Goal Regression —— 提取 Condition→Action rules

这是核心数学工具。Goal regression [135, 103, 104] 问的是：**"在 action $a^{hl}$ 执行前，state 要满足什么条件，才能保证执行后达到 $g^{hl}$？"**

公式 (4)：

$$regr(g^{hl}, a^{hl}) = \{(g^{hl} \setminus add_i(a^{hl})) \cup pre(a^{hl}) \mid i = 1, \ldots, n\}$$

变量逐一解释：
- $g^{hl}$：要 regressed 的 goal facts 集合
- $a^{hl}$：当前考虑的 HL action
- $add_i(a^{hl})$：action 第 i 个 nondeterministic outcome 的 add list
- $pre(a^{hl})$：action 的 precondition
- $g^{hl} \setminus add_i(a^{hl})$：从 goal 里去掉 action 会自动添加的 facts → 剩下的 facts 是 action 没帮忙实现的，必须执行前就满足
- $\cup pre(a^{hl})$：还要加上 precondition 才让 action applicable
- 集合 comprehension over $i$：对每个 nondeterministic outcome 算一个 regressor，全部收集

**可 regressable 条件**：对每个 outcome $i$，必须 $del_i(a^{hl}) \cap g^{hl} = \emptyset$。即 action 不能 delete 任何 goal fact —— 否则这个 action 不能用来达到 $g^{hl}$。

**Intuition**：regr 是把 "goal $\to$ action 前的 subgoal" 的算子。比如 goal 是 `at(block, loc_B)`，action 是 `move(loc_A, loc_B)` 且 $add = \{rAt(loc_B)\}, del = \{rAt(loc_A)\}, pre = \{rAt(loc_A)\}$，则 $regr = \{at(block, loc_B), rAt(loc_A)\}$ —— block 已经在 goal 位置（move 不动 block），机器人要在 loc_A。

**Algorithm 1** 的核心 loop 是 reverse iteration over HL trajectory：
```
for j = m, ..., 0:           # 从最后一个 action 倒推
    for s^{hl} in S:          # S 是当前要 regress 的 goal 集合
        S' = regr(s^{hl}, a_j^{hl})
        for s^{hl'} in S':
            r = lift(a_j^{hl}, s^{hl'}, g^{hl'})   # 见 4.3
            pi^{hl} = pi^{hl} ∪ {(r, m-j)}         # priority = m-j
    S = S_next
```

priority $val(r) = m-j$ 表示规则到 goal 的 "proximity"：越小越靠近 goal，越优先触发。这让 HL policy 像 backward-chaining planner 一样工作 —— 始终优先推离 goal 最近的 subgoal。

### 4.3 Step 3：Inductive Generalisation —— Lift 到 first-order

公式 (5) lifting operator：

$$lift(a^{hl}, s^{hl}, g^{hl}) = \langle var, sCond, gCond, action\rangle$$

操作很简单：把所有 ground objects $o_1, \ldots, o_q$ 换成 fresh variables $v_1, \ldots, v_q$，通过 mapping $g(o_i) = v_i$。

得到 first-order rule 例如：
```
3: ∃x, l, l1. at(x, l1), free(), rAt(l1), at^G(x, l) → pick(x, l1)
```
（取自 Example 2，下划线表示 goal condition $gCond$，上标 G 表示 goal fact）

Rule 形式化定义：
$$r = \langle val(r), var(r), sCond(r), gCond(r), action(r)\rangle$$

- $val(r)$：priority
- $var(r)$：variable set
- $sCond(r)$：state condition（first-order conjunctive formula over $\mathcal{P}$）
- $gCond(r)$：goal condition，要求 $gCond(r) \subseteq g^{hl} \setminus s^{hl}$，即 **未达成的 goal facts 中** 满足此条件
- $action(r)$：触发的 action schema

**Rule 触发**：对 rule $r$ 做 grounding（把 var 用 objects 实例化），检查 $sCond \subseteq s^{hl}$ AND $gCond \subseteq g^{hl} \setminus s^{hl}$，选 $val(r)$ 最小的 applicable grounded rule。本质上是个 backward-chaining 的 first-order production system。

### 4.4 Generalisation Theorem (Theorem 1)

**陈述**：domain $\mathcal{D}$ 给定，labelling function $\mathcal{L}$ 给定，存在有限 dataset $\mathbb{T}$，使得 Algorithm 1 学到的 $\pi^{hl}$ 能 solve 任何满足 C-bounded goal independence (GI$_C$) 的 HL problem。

**两个关键 assumption**：

**(A) Goal Independence (Definition 2)**：HL problem 的 goal facts 可以按任意 order 逐个独立达成，且在实现 $g_i$ 时不 delete 其他 $g_j$ ($i \neq j$)。

形式上：goal $g^{hl} = \{g_1^{hl}, \ldots, g_n^{hl}\}$，对任意 ordering，按以下流程都能 solve：
- $\pi^{hl} \leftarrow \emptyset$, $S \leftarrow \{init\}$
- 对每个 $i$：(a) 找到 solve $\mathbf{P}_{s^{hl}, \{g_i^{hl}\}}^{hl}$ 的 policy（不删其他 $g_j$），(b) 把 trajectory 中的 states 加入 $S$

**C-bounded**：每个 singleton-goal subproblem 的 preimage size $\leq C$，即每个 subgoal 只需至多 $C$ 步就能达到。

**(B) Object-Renaming Equivalence (Definition 3)**：两个 HL problem等价 iff 存在 object bijection $f: \mathbf{O}_1 \to \mathbf{O}_2$ 使得 $F$ 把 init 和 goal 互推。Proposition 1 证明：等价问题之间 lifted rules 可以直接 transfer —— 因为 first-order rule 只用变量名，对 object renaming 不敏感。

**证明 sketch**：
1. Up to $\sim$，singleton-goal subproblems 数量有限 —— 因为 singleton goal 由 predicate choice $p \in \mathcal{P}$ + 最多 $M = \max_p arity(p)$ 个 argument positions 决定，共 $|\mathcal{P}| \cdot M^M$ 种。
2. 长度 $\leq C$ 的 action sequence 提及 $\leq kN + M$ 个 objects（$N = \max_a |var(a)|$），所以 inequivalent rules 数 $\leq$ 公式 (8)：
$$n \leq |\mathcal{P}| \cdot M^M \cdot \sum_{k=0}^{C} \left(|\mathcal{A}| \cdot (kN+M)^N\right)^k$$
3. 取 $\mathbb{T}$ 覆盖每个等价类的 representative 即可。
4. 任意 test problem按某 order 解 $g_1, \ldots, g_n$，对每个 $g_i$ 由 lifting + Proposition 1 保证有 applicable rule。

**Intuition**：first-order rule 的 variable 让 single demonstration 覆盖所有 object-renaming equivalent problems；GI 保证只要每个 singleton subproblem 有 rule，整个 problem 就能 piecewise solve；bounded C 保证 rule 数量有限。

**Complexity**：$n$ 在 $C$ 和 $N$ 上 exponential，但实际中 demonstration 很少就够了 —— 因为很多 subproblem 共享 regression rules。

---

## 5. LL Policy Learning —— 一个紧凑的 GNN

### 5.1 输入 encoding

给定 $\mathbf{s}^{ll} = \langle \mathbf{x}_e, \{\mathbf{x}_o\}_{o \in \mathbf{O}}\rangle$，$a^{hl} = a(o_1, \ldots, o_n)$，$g^{hl}$，构造 graph：

- **Global node**：$\mathbf{h}_{global} = \mathbf{x}_e \parallel \sum_{p(\cdot) \in s^{hl}} \mathbf{e}_p^{|\mathcal{P}|} \parallel \sum_{p(\cdot) \in g^{hl}} \mathbf{e}_p^{|\mathcal{P}|}$
  - $\mathbf{x}_e$：agent 自己的 vector
  - $\sum \mathbf{e}_p^{|\mathcal{P}|}$：在 $s^{hl}$ 和 $g^{hl}$ 中的 nullary predicates（无参数 predicates）的 one-hot sum
  - $\parallel$：concatenation
  
- **Action node**：$\mathbf{h}_a = \mathbf{e}_a^{|\mathcal{A}|}$
  - 即 action schema 的 one-hot

- **Object node $o_i$**：$\mathbf{h}_{o_i} = \mathbf{x}_{o_i} \parallel \sum_{p(o_i) \in s^{hl}} \mathbf{e}_p^{|\mathcal{P}|} \parallel \sum_{p(o_i) \in g^{hl}} \mathbf{e}_p^{|\mathcal{P}|} \parallel \mathbf{e}_i^M$
  - $\mathbf{x}_{o_i}$：物体相对 agent 的 state vector
  - $\sum \mathbf{e}_p^{|\mathcal{P}|}$：$o_i$ 涉及的 unary predicates 的 indicator
  - $\mathbf{e}_i^M$：$o_i$ 在 HL action 中的 position one-hot（$M = \max_a |var(a)|$）—— **关键**：让 LL policy 知道这个 object 在当前 HL action 里扮演什么角色（如 pick 里的 obj vs. location）

注意这里**不 encode** binary / n-ary facts 的 relational 结构（PddlNN baseline 才 encode，见后文），保持 GNN 极小（<33k params）。

### 5.2 Message passing

3 步 GNN（Fig. 3）：

**(a) Embedding**：$\mathbf{h}_{global}^{(0)} = \mathbf{W}_g^{(0)} \mathbf{h}_{global}$, $\mathbf{h}_a^{(0)} = \mathbf{W}_a^{(0)} \mathbf{h}_a$, $\mathbf{h}_{o_i}^{(0)} = \mathbf{W}_o^{(0)} \mathbf{h}_{o_i}$

**(b) L rounds message passing**：
- 聚合 object 信息：$\mathbf{h}_o^{(l)} = \max_{i=1,\ldots,n} \mathbf{h}_{o_i}^{(l)}$（element-wise max pooling，permutation invariant）
- 更新 global：$\mathbf{h}_{global}^{(l+1)} = \sigma(\mathbf{W}_g^{(l)} (\mathbf{h}_{global}^{(l)} + \mathbf{h}_a^{(l)} + \mathbf{h}_o^{(l)}))$
- 更新 action：$\mathbf{h}_a^{(l+1)} = \sigma(\mathbf{W}_a^{(l)} (\mathbf{h}_{global}^{(l+1)} + \mathbf{h}_a^{(l)} + \mathbf{h}_o^{(l)}))$
- 更新 object $o_i$：$\mathbf{h}_{o_i}^{(l+1)} = \sigma(\mathbf{W}_o^{(l)} (\mathbf{h}_{global}^{(l+1)} + \mathbf{h}_a^{(l)} + \mathbf{h}_{o_i}^{(l)}))$

所有 node 之间 fully connected 加 sum（不是标准 message passing，更像 GraphSage / GIN 风格的全连接 graph）。$\sigma$ 是 nonlinearity。

**(c) Readout**：feed-forward on $\mathbf{h}_{global}^{(L)} + \mathbf{h}_a^{(L)} + \mathbf{h}_o^{(L)}$ 输出 LL action $\mathbf{a}^{ll}$。

### 5.3 训练

标准 BC：从 LL demo 提取 $\langle \mathbf{s}^{ll}, a^{hl}, g^{hl}, \mathbf{a}^{ll}\rangle$ tuples，MSE loss。Hyperparams：200 iter, Adam lr=1e-3, batch=128, cosine annealing, hidden=64, 2 message passing layers, 总参数 < 33k。

**Intuition**：LL policy 本质是 "skill library" —— 每个 $a^{hl}$ 触发对应的 LL controller。HL action + object position one-hot 把 GNN 路由到对应 skill。HL reasoning 完全在 symbolic policy 里，GNN 不背这个锅。

---

## 6. Experiments

### 6.1 环境设计（Table 1）

8 个环境，扩展 MetaWorld [147, 93]：
- **Blocks$^S$/Blocks$^N$**：把 n 个 block 放到同色 goal 位置。N 变体中已就位的 block 会随机 teleport
- **Factory$^S$/Factory$^N$**：每次放好 block 后会 spawn 新 block + 新 goal
- **Colour$^S$/Colour$^N$**：要先用 colourer 给 brown block 上色再放对应颜色 tray
- **Gacha$^S$/Gacha$^N$**：要操作 gacha box 产生 random colour block，open-world（不知道会出什么颜色）

三种 uncertainty 维度：
- **Exogenous**：外部扰动（如 teleport），HL model 不建模
- **Endogenous**：HL 建模的 nondeterminism
- **State / partial observability**：object 属性或位置不可见

Gacha 是 open-world —— closed-world PDDL planner 直接死了。

### 6.2 Baselines (Table 2 关键数据)

| Type | Method | Total SR |
|---|---|---|
| Oracle | hand-coded | upper bound |
| VLA | SmolVLA | 0.00 ± 0.0 |
| VLA | SmolVLA$^{MW}$ (fine-tuned on 2500 MetaWorld) | 0.00 ± 0.0 |
| GNN | PureNN (no HL info) | 0.00 ± 0.0 |
| GNN | PddlNN (HL facts but no HL action) | 0.30 ± 0.2 |
| Planning | DetPlan (single-shot deterministic planner) | 0.15 ± 0.3 |
| Planning | NdtPlan (nondeterministic planner) | 0.25 ± 0.4 |
| Planning | DetReplan (replan on failure) | 0.44 ± 0.4 |
| Planning | NdtReplan | 0.48 ± 0.4 |
| **Ours** | **BISON** | **0.79 ± 0.2** |

关键观察：

**(Q1) VLA 全死**：SmolVLA / SmolVLA$^{MW}$ 在所有 8 个环境都是 0%。哪怕 fine-tuned 在 MetaWorld 上，也只在 n=1 且 block 与 goal 在同一区域时偶尔工作。这与 [117] 的报告一致 —— VLA 在 hard MetaWorld 上 ≤45%。

**(Q2) Generalisation to longer horizon**：BISON 在 n=10 仍保持高 success rate，而 PureNN / PddlNN 在 n>6 后 performance drop。原因 [95, 141, 125, 118, 120]：GNN 的 expressive power 受 WL test 限制，无法 extrapolate 到更长 horizon planning。BISON 把 HL reasoning 推给 symbolic policy（first-order rule 自然 generalise 到任意 object 数），GNN 只学 LL skill。

**(Q3) Robustness to uncertainty**：
- DetPlan 在 exogenous uncertainty 下死得很惨（Blocks$^N$ 仅 0.34，Colour / Gacha 全 0）
- DetReplan 显著提升（Blocks$^N$ 0.97，Factory$^S$ 0.94）但 Colour / Gacha 还是 0 —— 因为 closed-world planner 不认识 colourer / gacha 的 nondeterminism
- BISON 在 Gacha$^S$ / Gacha$^N$ 都能 ~0.6+，因为 open-world symbolic policy + reactive rule triggering

**(Q4) Efficiency (Fig. 5)**：BISON training + inference 时间 vs success rate 最优。PureNN / PddlNN 慢得多 —— 因为它们 GNN 更大（要 encode 所有 HL info）。Replanning 方法每次失败要重 search，慢。

**(Q5) HL generalisation (Fig. 6)**：在纯 HL planning problem 上（忽略 LL execution），BISON 解决 10,000 个 block 的问题用几秒，LAMA [105] 在 ~100 个 block 时就 timeout（100s）。这是因为 BISON 的 $\pi^{hl}$ 是 reactive rule database，执行是 database query（找最低 priority applicable rule），与 object 数成 sub-linear。LAMA 要做 full search，复杂度爆炸。

### 6.3 学到的 HL policies 解读 (Appendix C.4)

非常 illuminating。例如 **Blocks$^S$ / Blocks$^N$** 学到两条 rule：
```
1: ∃x, l. holding(x), clear(l), at^G(x, l) → place(x, l)
2: ∃x, l. clear(x), gripperFree(), at^G(x, l) → pick(x)
```
对 Blocks$^N$ 也适用 —— 因为 goal predicate `at^G(x, l)` 对被 teleport 走的 block 仍为 true，自动触发 re-pick。**Reactive policy 自然处理 exogenous disturbance，不需要 explicit replanning**。

**Gacha** 学到 9 个 priority levels 的 policy，处理 stochastic gacha 产出。特别 Rules 6-9 是 **opportunistic actions**：当在追求 color $c'$ 但 gacha 吐出另一个 needed color $c$ 的 block 时，立刻把 block 放到对应 tray —— 不浪费 lucky roll。这种 implicit multi-goal coordination 是 search-based planner 难以做出来的。

---

## 7. Limitations & 我的思考

论文自己承认：
1. **依赖 $\mathcal{D}$ 和 $\mathcal{L}$**：HL abstraction quality 决定一切。这跟 Konidaris 的 "symbol emergence" 路线 [70, 71] 强耦合，是整个 bilevel planning paradigm 的固有限制。
2. **LL policy 无 optimality / generalisation 保证**：BC 的 covariate shift 问题，实验中 failures 主要来自 LL execution 的 covariate shift。作者建议 DAgger [106] 或 RL post-training。
3. **GI$_C$ assumption 强**：goal facts 要能独立达成。在强 coupled goal（如 "叠 tower 高度 ≥ 5"）下不成立。

我自己额外想到的几点：

**(a) $\mathcal{L}$ 的 brittleness**：HL state 完全依赖 $\mathcal{L}$ 给的 symbolic facts。如果 $\mathcal{L}$ 在某些 continuous state 上 flip（比如 block 刚 hover 在 goal 上方但还没放下，`at(block, loc)` 是否 true？），整个 HL reasoning 会 collapse。这是 symbolic abstraction 的 classic granularity issue。

**(b) NDRP 的 check 缺失**：论文引入了 NDRP 作为 "useful HL abstraction" 的定义，但 training 时没有 explicit enforce NDRP。如果 $\mathcal{L}$ / $\mathcal{D}$ / LL demos 不 satisfy NDRP，BISON 仍会跑，但可能 silently fail。

**(c) Goal regression 假设 deterministic demonstration**：Step 2 的 regr 要求 $del_i \cap g^{hl} = \emptyset$，但 demo 是从真实 LL execution 抽出来的，HL 抽象后的 action 效果可能比 declared schema 复杂（demo 里 `pick` 可能 implicitly 也 delete 了 `clear(top_of_stack)`）。Schema 不 complete 时 regression 可能 miss rules。

**(d) LL policy 的 skill 切分粒度**：每个 HL action 对应一个 LL skill，但 LL skill 没显式 terminate condition —— BISON 靠 $\pi^{hl}$ 在每个 step 重新 query 并 implicit terminate（HL state 变了就 trigger 下一个 action）。如果 LL execution 在 abstract state 边界附近 oscillate，会 dispatch 切换导致抖动。

**(e) 跟 recent "predicate invention" 路线 [120] / "VisualPredicator" [79] / "ExoPredicator" [80] 的关系**：这些工作 trying to auto-learn $\mathcal{D}$ 和 $\mathcal{L}$，BISON 假设它们 given。Combine 起来是 obvious next step —— 学 $\mathcal{D}, \mathcal{L}$ 再学 $\pi^{hl}, \pi^{ll}$ 全栈。

**(f) 与 LLM-as-planner [61, 84] 的对比**：LLM+P [84] 把 LLM 当 PDDL problem translator + symbolic planner 当 solver。BISON 不用 LLM 做 reasoning，而是用 first-order rule database 做 backward chaining。Rule database 是 explicit, inspectable, verifiable 的（Appendix C.4 LLM-interpretation 部分展示 LLM 可以读懂 rule 并解释）。这是 LLM-as-planner 路线的中间产物 —— 把 reasoning 部分外化到 symbolic structure。

**(g) $\pi^{hl}$ 的 reaction vs deliberation trade-off**：BISON 的 $\pi^{hl}$ 是 reactive (priority-ordered rules)，没有 lookahead search。在 opportunistic 场景（Gacha）效果反而比 search-based 好，因为不浪费 lucky rolls。但 in adversarial or irreversible setting（如 Sokoban）reactive policy 不够。Hybrid（rule + shallow search）是 future direction。

**(h) 跟 PDDLPlan 的区别本质**：PddlNN baseline encode 所有 HL facts 但不用 HL action，让 GNN 自己学 HL policy。它在 Colour$^N$ 上 0.36、Gacha$^S$ 上 0.51 表现还行，但远不如 BISON。说明 **HL reasoning 用 explicit symbolic form 比 implicit neural encoding 优越**，验证了 neuro-symbolic 的核心 thesis。

---

## 8. 关键 References

- Project page: https://dillonzchen.github.io/bison
- SmolVLA: https://huggingface.co/lerobot/smolvla_base
- SmolVLA on MetaWorld: https://huggingface.co/jadechoghari/smolvla_metaworld
- MetaWorld: https://arxiv.org/abs/1910.10897
- Fast Downward: https://arxiv.org/abs/1109.6071
- LAMA planner: https://www.jair.org/index.php/jair/article/view/10607
- Goal regression (Waldinger 1977): https://www.sciencedirect.com/science/article/pii/S0022499608000809
- Reiter "Knowledge in Action" book: https://mitpress.mit.edu/9780262681572/knowledge-in-action/
- GNN (Kipf & Welling 2017): https://arxiv.org/abs/1609.02907
- Konidaris "From skills to symbols" (2018): https://www.jair.org/index.php/jair/article/view/11167
- Silver et al. "Predicate invention for bilevel planning": https://arxiv.org/abs/2302.10734
- TAMP survey (Garrett et al. 2021): https://arxiv.org/abs/2102.05170

---

## 9. 一句话总结

BISON 的核心 insight 是：**HL policy 用 first-order symbolic rule 表示可以零成本 generalise 到任意 object 数和 long horizon（Theorem 1），而 LL policy 只需学 skill primitives（<33k params GNN 就够）**。这是把 reasoning 和 control 在 representation 上严格分离的结果 —— reasoning 的 representation 选取决定了能不能 scale，而 neural policy 的 representation 选取决定了能不能 reactive。两者各司其职，比把它们 entangle 进一个大 model 更 sample efficient、更 interpretable、更 long-horizon capable。

对于 build intuition：你可以把 BISON 想成 "一个 production system (SOAR/ACT-R 风格) + 一组神经 motor skills"。Production system 用 goal regression 从 demo induct 出来，motor skills 用 BC 学。整个系统的 reasoning complexity bound 在 schema arity $N$ 和 subproblem size $C$ 上（公式 8），不 bound 在 object 数上 —— 这就是它能 scale 到 10,000 个 objects 的根本原因。
