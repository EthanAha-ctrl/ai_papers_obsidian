---
source_pdf: Aligning Simulation Models A Case Study and Results.pdf
paper_sha256: d7aaaa56a3e9a9b24c95d836a6de1fc659c6338dec71404f1a6927f3bb75bfea
processed_at: '2026-08-18T00:51:11-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话概括

两个 model 长得完全不一样，一个极简、一个全功能，作者们把全功能那个"砍到只剩骨架"，看能不能复刻极简那个的结果——大部分时候能，但一个 20×20 的小细节差点崩掉，最后查出来是 agent activation 的一个微小差异。

---

## 两个 model 是什么

### ACM (Axelrod Culture Model)

你可以把它想成一个棋盘，每个格子里坐着一个 "agent"（村子），每个 agent 有一个 cultural vector，比如 5 个 feature，每个 feature 有 15 种可能的 value：

$$\mathbf{c}_i = (c_{i,1}, c_{i,2}, \ldots, c_{i,F})$$

- $i$ = agent 编号
- $f \in \{1, \ldots, F\}$ = feature 维度（dress、language、religion 等）
- $c_{i,f} \in \{0, 1, \ldots, q-1\}$ = 第 $f$ 个 feature 的取值
- $F$ = feature 数
- $q$ = 每个 feature 的取值数

**核心 rule**（这是整个 paper 的灵魂）：

随机挑一个 agent $i$，再随机挑它的一个 neighbor $j$，算两个 agent 的相似度：

$$s_{ij} = \frac{1}{F}\sum_{f=1}^{F}\mathbb{1}[c_{i,f} = c_{j,f}]$$

- $s_{ij}$ = agent $i$ 与 $j$ 在所有 feature 上相同比例
- $\mathbb{1}[\cdot]$ = indicator function，条件成立为 1 否则 0

然后 **以概率 $s_{ij}$** 进行文化交互——如果交互发生，就随便挑一个两人不同的 feature，把 $i$ 的那个 feature 改成 $j$ 的值。

注意这里的反直觉点：**两个人越像，越可能继续变更像**。这就是 homophily 的 positive feedback。两个人完全不同（$s=0$）就永远不交互，所以 diversity 能稳定存在。

跑到位所有相邻 agent 要么完全相同要么完全不同，就 frozen 了，数有几个 stable cultural region。

Reference: Axelrod 1997 JCR, https://www.jstor.org/stable/2118499

### Sugarscape

这是 Epstein & Axtell 那本 *Growing Artificial Societies* 的 model，是一个**完整的人造社会**：agents 会移动、吃 sugar、代谢、trade、fight、reproduce、得病。文化 diffusion 只是其中一块。

每个 agent：
- 有 vision $v$（能看到多远）
- 有 metabolism $m$（每 step 消耗多少 sugar）
- 有 sugar wealth $w$
- 有 cultural vector（原版 11 个 binary feature）

每个 time step，agent 先 look around → 移动到视野内 sugar 最多的格子 → 吃 sugar → 跟邻居做文化 exchange → 还可能做别的。

Reference: Epstein & Axtell 1996 book, https://mitpress.mit.edu/9780262550253/growing-artificial-societies/

---

## 为什么要把它们 dock

Karpathy 你做 Eureka Labs 肯定知道：任何 evaluation pipeline 想靠谱，必须 cross-implementation consistent。一个 model 单独跑出 surprising result，你不知道是真的 phenomenon 还是 bug。两个 independently designed model 都跑出同样 surprising result，才是 robust 发现。

这就是 paper 说的两个 hallmarks of cumulative science：
1. **Critical experiment** — 用 data reject 一个 model 接受另一个
2. **Subsumption** — 一个 general model 应该在 special case reduce 到 simpler model

类比 Einstein gravity subsumes Newton gravity——在低速弱场极限下，Einstein 公式 reduce 到 Newton 公式。

如果做不到 model-to-model comparison，ABM 永远是 "everyone builds their own toy world，没人能 compare"。作者们指出 1995 年时 social science 里 essentially zero alignment studies。

---

## 怎么 dock

### 砍 Sugarscape

Axtell 在 Ann Arbor 见面时花了约 4 小时把 Sugarscape 改成 ACM 样子：

1. Vision 设成 1（只能看到 von Neumann 4 邻居）
2. Movement range 设成 0（agent 不动）
3. Population density 设成 1（每个格子都有 agent）
4. Topology 从 toroidal 改成 bounded square
5. Features/traits 参数化（不再固定 11 binary）
6. 用 ACM 的 cultural rule 替换 Sugarscape 自己的
7. 关掉所有 metabolism、trade、combat 等 process

Sugarscape 是 Object Pascal 写的，object-oriented，大部分修改都是 "throw switches"。

### 留下的两个 subtle 差异

**差异 1（一开始留着的）**: Activation 方式

- ACM: 每次 step 从 $N$ 个 agent 里 uniform-with-replacement 选一个 active
- Sugarscape: 每次 round 把 agent list 随机 permutation，走完一遍再重新 permutation

差异在统计上是 sampling with vs without replacement。看起来 trivial，后面会爆雷。

**差异 2（两个月后才发现的）**: Cultural rule 里改谁

- ACM: 改 active agent（被选中的那个）
- Sugarscape 原版: 改 neighbor

bounded square 上 edge agent 的 neighbor < 4，所以"改谁"会让 interior 和 edge agent 的 change probability 不一样。修复后重新跑数据。

---

## 主要结果

### Table 1: feature × value 网格（10×10 lattice）

| Features\Values | 5 | 10 | 15 |
|---|---|---|---|
| 5  | 1.0 | 3.2 ± 1.8 | 20.0 ± 10.1 |
| 10 | 1.0 | 1.0 | 1.4 ± 0.5 |
| 15 | 1.0 | 1.0 | 1.2 ± 0.4 |

这是 ACM 原版数据。直觉：
- $q$ ↑（更多 traits）→ cultural space 更大 → 更难 collapse → region 数 ↑
- $F$ ↑（更多 features）→ similarity-based interaction 更易触发 → 更易 homogenize → region 数 ↓

Sugarscape 重做的数据 qualitative pattern 一样，9 个 cell 里只有 3 个不是 1.0，与 ACM 的 4 个不一致有 1 个 cell 的差别。

**统计检验用 Mann-Whitney U**：

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$$

- $n_1 = n_2 = 10$: 两个 sample 的大小
- $R_1$: sample 1 在 pooled rank 里的 rank sum
- $U_{\text{crit}} = 23$ at $\alpha=0.05$ two-sided

所有 9 个对比 $U > 23$，**无法拒绝** $H_0$: 两 sample 来自同一 distribution。即 distributional equivalence 在 Table 1 上达成。

Reference: Mann-Whitney test, https://www.statisticshowto.com/mann-whitney-u-test/

### Figure 1: lattice size 的 non-monotonic 现象

固定 $F=5, q=15$，变化 lattice 大小 $L \in \{5, 10, 20, 50, 100\}$。

**反直觉现象**: stable region 数随 $L$ 先升后降。

直觉应该是单调上升——更大 space 应该有更多 region。但实际：
- 小 $L$: 整个 space 小，邻居多，快速 homogenize
- 中 $L$: 局部 form stable region，全局不连，region 数 peak
- 大 $L$: 中间态时间长，反而让更大 region 有机会形成，最终 stable region 数下降

这是 paper 最 surprising 的 finding，单 model 看不出是不是 artifact，必须靠 alignment 验证。

**统计检验用 Kolmogorov-Smirnov**：

$$D = \sup_x |F_1(x) - F_2(x)|$$

- $F_1(x), F_2(x)$: 两个 sample 的 empirical CDF
- $D$: 两 CDF 的最大 vertical distance
- $n=40, \alpha=0.05$ two-tailed: $D_{\text{crit}} = 0.304$

| Lattice | K-S $D$ | 决定 |
|---------|---------|------|
| 5×5     | 0.225 | 不拒绝 $H_0$（equivalent） |
| 10×10   | 0.175 | 不拒绝 $H_0$（equivalent） |
| **20×20** | **0.5** | **拒绝 $H_0$（NOT equivalent）** |

20×20 case 直接爆了：ACM mean = 16.25，Sugarscape mean = 9.23，差了快一半。

Reference: K-S test, https://www.statisticshowto.com/kolmogorov-smirnov-test/

---

## 谁错了：Activation 方式的蝴蝶效应

作者们回头查，发现就是 Section 3.1 提到但没改的那个 "with vs without replacement" 差异：

- Sugarscape (without replacement): 每个 agent 一个 round 恰好 active 一次，influence 分配均匀
- ACM (with replacement): 一些 agent active 多次，另一些 0 次，influence 不均匀

均匀的 influence 让 cultural diffusion 更彻底，所以 Sugarscape 收敛得更狠，region 数更低。

Axtell 把 Sugarscape 改成 with-replacement activation，所有 3 个 lattice size 都通过 K-S test。

**Takeaway**: 你写 PyTorch code 时 `torch.randperm` vs `torch.randint` 这种选择同理，在 high-dimensional 或 critical regime 下能产生 statistically significant 的 outcome 差异。今天的 RL reproducibility crisis 本质就是这个问题。

Reference: Henderson et al. 2018 "Deep RL That Matters", https://arxiv.org/abs/1709.06560

---

## 三个 Levels of Equivalence

这是 paper 最有方法论价值的 contribution：

### Level 1: Numerical identity
两 model 跑出完全一样的 number。
- 在 stochastic model 里基本不可能（除非 fixed seed 完全算法等价）
- 只在 deterministic model 间有意义

### Level 2: Distributional equivalence
两 model 产生的 distribution 用 statistical test 无法 distinguish。
- 用 K-S / Mann-Whitney 等 nonparametric test
- 这是本文实际采用的标准

### Level 3: Relational equivalence
两 model 产生相同 internal relationship among results。
- 例: region 数随 $L$ non-monotonic
- 不要求 number 对得上，只要求 pattern 对得上
- 最弱但很多 theoretical purpose 已足够

对应到 subsumption 强度：
- Numerical identity → B 完全等于 A 的 special case
- Distributional equivalence → B 在 special configuration 下与 A 统计上 indistinguishable
- Relational equivalence → B preserve A 的 qualitative phenomena

---

## Alignment 的真正 value：Sensitivity Analysis

Dock 成功不是终点，而是起点。作者把 Sugarscape 的 richness 当 ACM 的 extension，做 ACM 单独做不了的实验。

### Mobility Experiment

让 agent 在 50×50 Sugarscape 上能动（vision 5-10，吃 sugar，移动后按 ACM rule 文化交互），看 region 数会怎样。

直觉预判: 移动 = mixing，所以 diversity 应该降低。

结果（10 runs, $F=5, q=15$）：

| Configuration | Mean regions |
|---------------|--------------|
| ACM 10×10 fixed | 18.5 |
| Sugarscape 10×10 fixed | 20.4 |
| Sugarscape 50×50 mobile | **1.1** |
| Sugarscape 50×50 mobile, $q=30$ | 2.2 |

Diversity 暴跌。把 cultural space 翻倍（$q$ 从 15 到 30）只能拉回到 2.2。

### Soup Experiment

极端 mixing: 完全 random pairing，忽略 space。

结果（$F=5, q=15$, 10 runs）: 没有任何一次 region 数 > 1，完全 homogenize。

$q=30$: 7 runs → 1 region, 2 runs → 2 regions, 1 run → 3 regions，mean = 1.4。

**理论结论**: ACM 的 multicultural equilibrium 依赖于"完全不同 agents 之间 interaction probability = 0"这个 strong assumption。任何 nonzero mixing 或 mutation，long-run attractor 都是 single culture。

### Phase Transition Timing 的反直觉

> "The more well-mixed the society, the later is this 'phase transition.'"

直觉预判: mixing 更强 → homogenize 更快。

实际: mixing 更强 → phase transition **更晚**，但 final diversity 更低。

为什么？homophily rule 的 effective update rate $\propto s_{ij}$。Mixing 降低了 average pairwise similarity $\langle s_{ij} \rangle$，所以早期 transition 慢；但 mixing 让每个 agent 最终见到所有其他 agent，attractor basin 是全 1-culture state。

**这个 insight 单 model 看不出来**。必须 cross-model sensitivity sweep 才能浮现。这就是 alignment 的真正 value。

---

## 时间成本

- **Axelrod**: 23 hours（9 design + 14 analysis）
- **Axtell**: 37 hours（4 code + 1 follow-up code + 10 跑 + 9 stats + 2 mobility + 1 soup + 8 re-docking）

37 hours 大头在跑 + re-docking + stats。Actual code change 不到 5 小时。

这个时间预算在 1995 年算合理，今天用 Python/PyTorch 写类似 docking 实验也差不多——critical 的是 face-to-face meeting 解决 ambiguity 的那部分，没法纯 remote 替代。

---

## 让这个 case 相对 easy 的 4 个 factor

1. Sugarscape 用 Object Pascal 写，OOP，generality-driven
2. ACM 极简，prose description 几乎完整
3. ACM 项目 recent，原作者还能提供 raw data
4. 两 model 共享 agent-based framework，没 framework mismatch

---

## 让 case 相对 hard 的 factor

1. Sugarscape 设计时没预期要 dock ACM
2. 没标准化 code modules（除 RNG）
3. **Prose description of model insufficient for replication** — 必须 face-to-face 才能澄清 ambiguity
4. Stochastic model 的 equivalence 标准本身没现成 framework

---

## 关于 Statistical Logic 的 Perverse Incentive

作者注意到一个 methodological landmine：

传统 "fail to reject $H_0$" 框架下，**小 sample** 更易 achieve "equivalence"（因为 power 低）。这是 alignment 的方法论隐患——研究者可以通过跑太少 runs 来"证明"两 model equivalent。

他们 propose 反转 null：

> "Can we confidently reject the claim that the distributions are different?"

两个 complication:
1. Stochastic model 里 sampling fluctuation 永远存在，需要 null hypothesis 形如 "distributions differ by no more than $X\%$"
2. Non-simple null + 未知 distribution 形态 → 需 bootstrap 这种 computational statistics

Reference: Efron & Tibshirani 1993, https://www.tandfonline.com/doi/book/10.1201/9780429246591

这个 observation 在今天 ML reproducibility crisis 仍然高度 relevant。Henderson et al. 2018 指出同一 RL algorithm 不同 implementation 的 reward variance 可 >2x。

---

## 跟现代 research 的 connection

1. **ABM replication standards**: Wilensky & Rand 提出 replication dimensions
   - https://www.jasss.org/JASSS/10/4/2.html

2. **Climate model intercomparison (CMIP)**: 这是 alignment 思想在物理 science 的大规模实践
   - https://www.wcrp-climate.org/wgcm-cmip

3. **Differentiable simulation**: CasADi、JAX 让 alignment 可以 gradient-based
   - CasADi: https://web.casadi.org/
   - JAX: https://github.com/google/jax

4. **AI safety / interpretability**: Anthropic 的 "model stitching" 与此同源
   - https://transformer-circuits.pub/

5. **Differential testing** in software engineering: 同样思想
   - https://en.wikipedia.org/wiki/Differential_testing

6. **CasADi/JAX differentiable ABM**: 现在 alignment 可以是 gradient-based，不再仅 statistical test
   - https://dl.acm.org/doi/10.1145/3580305

---

## 你应该 take away 什么

1. **Alignment ≠ Reimplementation**: reimplementation 从一开始 target reproduce；alignment 是两个 independently designed 的 model 在相同 phenomenon 上的 stress test。

2. **三个 equivalence level** 值得记：
   - Numerical identity（基本不可能）
   - Distributional equivalence（本文 target）
   - Relational equivalence（最弱但常足够）

3. **Stochastic equivalence 的 statistic 要谨慎**: "fail to reject $H_0$" 在小 sample 下虚高，需要反转 null + bootstrap。1995 年提的 issue，今天 ML reproducibility 还没解决。

4. **Subtle implementation differences matter**: with/without replacement、改 active vs neighbor——这种 trivial 选择在某些 parameter regime 爆 statistically significant 差异。今天写 PyTorch 时 `randperm` vs `randint`、`model.eval()` 忘加、`shuffle=True/False` 这种选择同理。

5. **Alignment 的真正 value 是 sensitivity 的 anchor**: dock 完成后，richer model 把 simpler model 当 baseline，问 "如果 relax 这个 assumption 会怎样"。本文 mobility/soup 实验就是这个 pattern 的范例。RL 里等价 "vanilla PPO baseline + 我加的 trick 在此基础上 ablation"。

6. **Non-monotonic region-vs-lattice-size** 这种 counterintuitive 现象是 ABM 的 typical signature——linear intuition 不 work，需要 explicit simulation。这也是为什么 alignment 必要：单 model 的 surprise 可能是 artifact，两个 independent model 都产生同样 surprise 才 robust。

7. **Mathematical structure 记住两个 test**：
   - Mann-Whitney $U$ for rank-based two-sample test
   - K-S $D = \sup_x|F_1-F_2|$ for distributional test
   这两个今天还是 ABM/RL reward distribution comparison 的 baseline 选择。

8. **Phase transition timing 的反直觉**: mixing 推迟 phase transition 但降低 final diversity。单 model 看不出，必须 cross-model sweep 才能浮现。这是 alignment 的方法论价值的核心论证。

---

## 与你 Eureka Labs 的工作的联系

你做 Eureka Labs 的 evaluation pipeline 时，本质就是 alignment 思想：

- 同一 concept 用不同 model（GPT-4、Claude、Gemini）回答
- 同一 model 不同 prompt template
- 同一 prompt 不同 sampling temperature

只有 cross-implementation consistent 的 finding 才算 robust。一个 model 的 surprising capability 可能是 artifact（memorization、prompt exploit、benchmark contamination），多 model 都 produce 同样 surprise 才是 phenomenon。

这正是 Axtell-Axelrod 1995 paper 的精神，只是 LLM eval 比 ABM 复杂几个数量级。

Reference: 
- 你的 Eureka Labs: https://eurekalabs.ai
- 你的 blog "Software 2.0": https://karpathy.medium.com/software-2-0-a64152b37c35

---

最后一句：这篇 1995 年的 paper 在 2026 年读仍然 fresh，因为 alignment/docking 这件事在 deep learning、LLM benchmark、multi-agent simulation 各领域都还没真正标准化。你想 build intuition 的话，把 ACM 的 cultural rule 自己用 NumPy 实现一遍，再用 JAX 重写一遍做不同iable，看两个 implementation 在 $L=20$ 上的 K-S distance 是否还 < 0.304——这是 best way internalize 这篇 paper。

---

# Aligning Simulation Models: A Case Study and Results 深度解析

这篇 paper 是 1995 年 Santa Fe Institute 的 working paper，作者阵容 Robert Axtell (Brookings), Robert Axelrod (Michigan), Joshua M. Epstein (Brookings), Michael D. Cohen (Michigan)。它是 agent-based modeling (ABM) 领域的奠基性方法论 paper，提出了 **model alignment (docking)** 这个概念——即验证两个独立的 computational model 是否能在 equivalent conditions 下产生 equivalent results。这件事在 social science simulation 里基本没人正式做过，作者们认为这是 ABM 成为 cumulative discipline 的必要前提。

Reference link:
- Santa Fe Institute working paper archive: https://www.santafe.edu/research/results/working-papers/aligning-simulation-models-case-study-and-results
- Axelrod 1995 culture model: https://www.jstor.org/stable/2118499 (JCR 1997 published version)
- Sugarscape book (Epstein & Axtell 1996): https://mitpress.mit.edu/9780262550253/growing-artificial-societies/
- Modern docking replication discussion: https://www.jasss.org/JASSS/20/3/12.html

---

## 1. Motivation: 为什么需要 Alignment

科学积累依赖两个 hallmarks：
1. **Critical experiment** — 用实验 reject 一个 model 而接受另一个 fit data 更好的 model
2. **Subsumption** — 像 Einstein 的 gravity subsumes Newton 那样，一个更 general 的 model 应该在 special case 下 reduce 到 simpler model

如果做不到 compare two models 的 outputs，ABM 就永远没法像 mathematical theory 那样给出清晰的 "domain of validity"。作者明确指出：social science 里有大量 ABM，但 essentially zero alignment studies，只有少量 reimplementation（如 Prietula 重写 Cyert & March 1963，Levitt 重写 Cohen-March-Olsen 1972 garbage can model）。Reimplementation 与 alignment 的区别在于：reimplementation 的目标从一开始就是 reproduce，而 alignment 要面对的是**两个 independently designed、mechanism 不同的 model**。

类比 orbital docking（轨道对接）——两个 design 不同的 spacecraft 在轨道上 match 速度与位置。

---

## 2. 两个待 Dock 的 Model

### 2.1 Axelrod Culture Model (ACM)

**Architecture**:
- $\sqrt{N} \times \sqrt{N}$ square lattice（$N$ = agent 数）
- 每个 agent $i$ 有 cultural vector $\mathbf{c}_i = (c_{i,1}, c_{i,2}, \ldots, c_{i,F})$
- $F$ = number of features（本工作用 $F \in \{5, 10, 15\}$）
- 每个 feature $c_{i,f} \in \{0, 1, \ldots, q-1\}$，$q$ = number of traits per feature（用 $q \in \{5, 10, 15\}$）
- 初始每个 $c_{i,f}$ 均匀随机
- Neighborhood: von Neumann 4-neighborhood（North/South/East/West）

**Update rule** (ACM 的核心 mechanism):

1. Random activation: 从 $N$ 个 agents 中 uniform-with-replacement 选 active agent $i$
2. 从 $i$ 的 neighbors 中 uniform 选一个 neighbor $j$
3. 计算文化相似度 $s_{ij} = \frac{1}{F}\sum_{f=1}^{F}\mathbb{1}[c_{i,f} = c_{j,f}]$，$s_{ij} \in [0,1]$
4. 以概率 $s_{ij}$ 交互：
   - 选一个 feature $f^*$ such that $c_{i,f^*} \neq c_{j,f^*}$（uniform 在不同 features 中）
   - 令 $c_{i,f^*} \leftarrow c_{j,f^*}$（**active agent 被修改**，不是 neighbor）

**Key insight**: 相似度越高，越可能进一步交互——这就是 **homophily-driven positive feedback**。这与"unlike neighbors 也能继续 interact"的 model 不同，那种 model 会不可避免地收敛到 homogeneity。

**Termination**: 所有相邻 agents 要么 $s_{ij}=0$（完全不同）要么 $s_{ij}=1$（完全相同），此时 system 在 frozen state，stable cultural regions 数即 count 出来。

**Code-level detail (Axtell 在 Ann Arbor 发现的)**: Axelrod 实际实现中并不是按 $s_{ij}$ 概率交互，而是等价且更高效的"选一个 random feature，如果相同则交互"。两种实现 produce same distribution。

### 2.2 Sugarscape

Sugarscape 是 Epstein & Axtell 1995 那本 *Growing Artificial Societies* 的核心 model——一个完整的 "artificial society" 系统。

**Architecture**:
- Toroidal grid（本工作改成 bounded square 以匹配 ACM）
- 每个 cell 有 sugar level $\sigma(x,y) \in \mathbb{Z}^+$
- Sugar mountain (Gaussian bump) 提供 resource 异质性
- Agents: 有 vision $v$，metabolism $m$，sugar wealth $w$，cultural tag vector

**Standard Sugarscape agent loop** (每一 time step 对每个 active agent):
1. Look around in 4 directions up to vision $v$
2. Move to unoccupied cell with max sugar within vision (rule M)
3. Eat sugar: $w \leftarrow w + \sigma(\text{cell}) - m$
4. Cultural interaction: 选一个 neighbor，randomly 选一个 feature，如果不同则 adopt neighbor 的值
5. 可选: trade, combat, reproduction, disease, etc.

**Cultural representation**: 原版 Sugarscape 有 11 binary features，aggregated 通过 majority vote 得 "Red" or "Blue" tag，tag 进入 trade/combat 等其他 process。

**Differences from ACM (key list)**:

| 维度 | ACM | Sugarscape |
|------|-----|------------|
| Topology | Bounded square | Toroidal |
| Movement | Agents 固定 | Vision-based purposive |
| Population density | 全部 cell 占满 | Sparse |
| Activation | Random with replacement | Random permutation (without replacement) |
| Cultural rule | 修改 active agent | 修改 neighbor |
| Features / traits | $F, q$ 参数化 | 固定 $F=11, q=2$ |
| Other mechanisms | None | Movement, metabolism, trade, combat, sex, disease |

Sugarscape 的 philosophy 是 **sufficiency testing**——证明某组 micro-mechanism 足以 generate 某 macro 现象。ACM 的 philosophy 是 **intensive parameter exploration**——单一 mechanism 极简 model，深入 sweep 参数。

---

## 3. Docking Procedure

### 3.1 修改 Sugarscape 以匹配 ACM

Axtell 在 Ann Arbor 见面时做了如下 code 修改（参见 Appendix 2，total 3:50 hours）：

1. **Vision → 1** (immediate 4 neighbors only)
2. **Movement range → 0** (agents 不动)
3. **Population density → 1** (every cell has an agent)
4. **Topology: toroidal → bounded square**
5. **Features/traits → parameterized** to match ACM 的 $F \in \{5,10,15\}, q \in \{5,10,15\}$
6. **替换 cultural rule** 为 Axelrod 的 rule（修改 active agent，similarity-proportional interaction）
7. **Turn off** 所有其他 Sugarscape processes (metabolism, trade, combat, etc.)

**保留的一个 subtle difference**: 
- Sugarscape activation: random permutation of agent list, 然后 list 走完再 re-permute（**without replacement** within a round）
- ACM activation: 每 step 都 from scratch random selection（**with replacement**）

这个看似 minor 的差异，在 $20\times 20$ case 上爆出了 statistically significant 的 discrepancy，是全文最 informative 的细节之一。

### 3.2 第二次发现并修复的差异

Axtell 在写 paper 阶段（两个月后）意识到：原版 Sugarscape cultural rule 修改 **neighbor**，ACM 修改 **active agent**。在 bounded square 上，edge/corner agent 的 neighbor 数 < 4，所以"谁被修改"会引入 spatial bias——interior agents 比 edge agents 更容易被修改。

修复后，重新 run 所有 experiments。这就是 "re-docking" step（Appendix 2.7，8:30 hours，40 runs）。

---

## 4. Quantitative Results

### 4.1 Table 1: Features × Traits/Feature 网格

10x10 lattice, 10 replications per cell.

**Table 1a (ACM, target)**:

| Features \ Values | 5 | 10 | 15 |
|---|---|---|---|
| 5  | 1.0 | 3.2 ± 1.8 | 20.0 ± 10.1 |
| 10 | 1.0 | 1.0 | 1.4 ± 0.5 |
| 15 | 1.0 | 1.0 | 1.2 ± 0.4 |

**Table 1b (Sugarscape re-docked)**:

| Features \ Values | 5 | 10 | 15 |
|---|---|---|---|
| 5  | 1.0 | 5.3 ± 3.9 | 21.3 ± 12.5 |
| 10 | 1.0 | 1.0 | 1.5 ± 0.7 |
| 15 | 1.0 | 1.0 | 1.0 |

**Pattern**: equilibrium regions 数随 $q$ ↑ 而 ↑（更多 traits 意味着更大 cultural space，更难 collapse），随 $F$ ↑ 而 ↓（更多 features 意味着 similarity-based interaction 更易触发，更易 homogenize）。

**Statistical test — Mann-Whitney U**:

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$$

- $n_1, n_2$: 两个 sample 的大小（这里都是 10）
- $R_1$: sample 1 的所有 observation 在合并 pooled 排序后的 rank sum
- Two-sided critical value at $\alpha=0.05$, $n_1=n_2=10$: $U_{\text{crit}} = 23$
- 拒绝 $H_0$ if $U \leq 23$

对于 Table 1 的 9 个 cells，所有 $U$ 都 > 23，**无法拒绝** $H_0$: 两 samples 来自 same distribution。**Distributional equivalence achieved** on Table 1。

注意 Mann-Whitney 的核心: 它 nonparametric，不要求 normality，只基于 ranks。这正适合 ABM output 这种 distribution 未知、且常常 heavy-tailed 的情况。

### 4.2 Figure 1: Non-monotonic Region Count vs Lattice Size

固定 $F=5, q=15$，变化 lattice size $L \in \{5, 10, 20, 50, 100\}$（ACM 用了 40 runs for 5x5/10x10/20x20，10 runs for 50x50/100x100）。Sugarscape 重做 5x5, 10x10, 20x20，每个 40 runs。

**关键现象**: region 数先升后降，non-monotonic。直觉：
- 小 $L$: 邻居多但整个 space 小，容易 homogenize，region 少
- 中 $L$: 局部 homogenize 形成稳定 region 但全局不相连，region 数 peak
- 大 $L$: region 数应该继续增，但 ACM 显示反而下降

Axelrod 解释: 大 $L$ 时更多"未稳定"中间态有机会形成更大 region，所以最终 stable region 数反而下降。这违背 naive 直觉，是 paper 最 surprising result。

**Kolmogorov-Smirnov test**:

$$D = \sup_x |F_1(x) - F_2(x)|$$

- $F_1(x), F_2(x)$: 两个 sample 的 empirical CDF
- $D$ 是两 CDF 的最大 vertical distance
- Two-tailed, $\alpha = 0.05$, $n=40$: $D_{\text{crit}} = 0.304$

| Lattice | K-S $D$ | Decision |
|---------|---------|----------|
| 5x5     | 0.225 | 不能拒绝 $H_0$ (equivalent) |
| 10x10   | 0.175 | 不能拒绝 $H_0$ (equivalent) |
| 20x20   | **0.5** | **拒绝 $H_0$** (NOT equivalent) |

20x20 case: ACM mean = 16.25, Sugarscape mean = 9.23。差了近一半！

**为何差异？**

回头看 Section 3.1 提到的 activation 差异：
- Sugarscape (without replacement): 每个 agent 在每个 round 恰好 active 一次，influence 分配均匀
- ACM (with replacement): 一些 agent 会 active 多次，另一些 0 次，influence 不均匀

均匀的 influence 让 cultural diffusion 更彻底，导致更多 homogenization，所以 Sugarscape 的 region 数更低（9.23 < 16.25）。

Axtell 把 Sugarscape 改成 with-replacement activation 后，所有 3 个 lattice size 都 achieve distributional equivalence。这就是 Appendix 2.7 那 8:30 hours 的 re-docking。

### 4.3 三个 Levels of Equivalence

Paper 在 conclusion 提出三级 equivalence 概念，这是最有方法论价值的 contribution：

**Level 1: Numerical identity** — 同样的 numbers
- 在 stochastic model 里基本不可能
- 只在 deterministic model 间有意义

**Level 2: Distributional equivalence** — 两 model 产生的 distributions 统计上 indistinguishable
- 用 K-S test 或 Mann-Whitney 等 nonparametric test 检验
- 这是本文实际采用的标准

**Level 3: Relational equivalence** — 两 model 产生相同的 internal relationships among results
- 例: 两 model 都显示 region count 是 lattice size 的 non-monotonic function
- 更弱，但很多 theoretical purpose 已足够
- 比 distributional 容易达到，不需 parametric alignment

这三个 level 直接对应 **subsumption** 的强度：
- Numerical identity → model B 完全等于 A 的 special case
- Distributional equivalence → B 在 special configuration 下 statistical indistinguishable from A
- Relational equivalence → B preserve A 的 qualitative phenomena

---

## 5. Sensitivity Analysis: 两个 Extension Experiments

Docking 成功后，作者利用 Sugarscape 的 richer mechanism 做了两个 ACM 无法做的 sensitivity 实验。这正是 alignment 的最大 value：把 simple model 当 anchor，再用 rich model explore mechanism robustness。

### 5.1 Mobility Experiment

**Setup**:
- 50x50 Sugarscape with central Gaussian sugar mountain
- 100 mobile agents, vision uniform in [5,10]
- $F=5, q=15$（ACM 中产生最多 region 的配置，ACM 平均 18.5, Sugarscape-fixed 平均 20.4）
- 移动后再按 ACM rule 文化交互

**Stopping criterion 修改**: 原本 local (相邻 agents 完全同/异)，现在 global (任意两 agents 完全同/异)，因为 mobile agent 的邻居关系不固定。

**Results** (10 runs):

| Configuration | Mean regions |
|---------------|-------------|
| ACM (10x10 fixed) | 18.5 |
| Sugarscape (10x10 fixed, ACM rule) | 20.4 |
| Sugarscape (50x50 mobile, ACM rule) | **1.1** |
| Sugarscape (50x50 mobile, $q=30$) | 2.2 |

**Intuition**: 移动让任何两个 agents 最终都可能直接 interact，相当于把"非邻居 interact 概率为 0"放宽。Mixing → homogenization，diversity 暴跌。

### 5.2 "Soup" Experiment

**Setup**: 完全 random pairing，忽略 space。这是 mixing 的 extreme case。

**Results** ($F=5, q=15$): 10 runs 中**没有**任何一次 region 数 > 1。完全 homogenize。

**$F=5, q=30$**: 7 runs → 1 region, 2 runs → 2 regions, 1 run → 3 regions。Mean = 1.4。

**Key theoretical takeaway**: ACM 的 multicultural equilibrium 依赖于 "完全不同的 agents 之间 interaction probability = 0" 这一 strong assumption。一旦有任何 nonzero mixing probability 或 mutation rate，long-run attractor 都是 single culture。

### 5.3 Phase Transition Timing 的反直觉发现

> "The more well-mixed the society, the later is this 'phase transition.'"

- ACM (no mixing): local clusters of similar agents 形成 spatial correlation，agreement 快速达到
- Mobility case: agents "hop away" before agreement locks in，phase transition 延迟
- Soup (zero spatial correlation): phase transition 最迟

虽然 phase transition 更晚，但最终 homogenize 更彻底。所以 **mixing 推迟 phase transition 但同时降低最终 diversity**。这是 paper 中最 counterintuitive 的 dynamic insight。

数学上可以这样理解：homophily-driven update rule 的 effective rate $\propto s_{ij}$，mixing 降低了 average pairwise similarity $\langle s_{ij} \rangle$，所以早期 transition 慢；但 mixing 让每个 agent 最终见到所有其他 agents，homogenization 的 attractor basin 是全 1-culture state。

---

## 6. Process Observations: Docking 本身的 metadata

### 6.1 时间成本

- **Axelrod**: 23 hours (9 design + 14 data analysis)
- **Axtell**: 37 hours (3:50 code + 1:10 subsequent code + 10:40 running + 9:00 stats + 2:25 mobility + 1:15 soup + 8:30 re-docking)

### 6.2 让这个 case 相对 easy 的 4 个 factor

1. Sugarscape 用 Object Pascal 写，object-oriented，generality-driven，改动很多是 "throw switches"
2. ACM 极简，prose description 几乎完整
3. ACM 项目 recent，原作者还能提供 raw data
4. 两 model 共享 agent-based framework，没 framework mismatch

### 6.3 让这个 case 相对 hard 的 factor

1. Sugarscape 设计时不预期要 dock ACM
2. 没有 standardized code modules（除 RNG 外）
3. **Prose description of model insufficient** for replication——必须 face-to-face meeting 才能澄清 ambiguity
4. Stochastic model 的 equivalence 标准本身没现成 framework

### 6.4 关于 Statistical Logic 的反思

作者注意到一个 perverse incentive：传统 "fail to reject $H_0$" 框架下，**小 sample** 更易 achieve "equivalence"（因为 power 低）。这是 alignment 的方法论隐患。

他们 propose 反转 null hypothesis：
> "Can we confidently reject the claim that the distributions are different?"

但这有两个 complication:
1. Stochastic model 里 sampling fluctuation 永远存在，需要 null hypothesis 形如 "distributions differ by no more than $X\%$"
2. 这种 non-simple null + 未知 distribution 形态 → 需 bootstrap (Efron & Tibshirani 1993, https://www.tandfonline.com/doi/book/10.1201/9780429246591) 这类 computational statistics

这个 observation 在今天 reinforcement learning reproducibility 危机（Henderson et al. 2018, https://arxiv.org/abs/1709.06560）和 ABM reproducibility 危机（Wilensky & Rand 2007, https://mitpress.mit.edu/books/introduction-agent-based-modeling）背景下仍然高度 relevant。

---

## 7. 与现代 Research 的 Connection

这篇 1995 paper 的思想在多个领域被重新发现/扩展：

1. **ABM replication standard**: Wilensky & Rand 提出 replication dimensions (Woolson sampling, time series, statistical distribution)
   - https://www.jasss.org/JASSS/10/4/2.html

2. **Model-to-Model (M2M) comparison**: 生态学 model validation (Ayllón et al. 2016, https://www.sciencedirect.com/science/article/pii/S0304380016300871)

3. **Software engineering analogy**: 这就是 differential testing / regression testing 在 simulation 上的对应
   - Differential testing: https://en.wikipedia.org/wiki/Differential_testing

4. **Reinforcement learning**: Henderson et al. "Deep RL That Matters" 系列指出相同 algorithm 不同 implementation 的 reward variance 可以 >2x，这是 docking 缺失的代价
   - https://arxiv.org/abs/1709.06560

5. **CasADi / JAX 等可微分 simulation 框架**: 让 alignment 可以是 gradient-based 的，不再仅是统计 test
   - CasADi: https://web.casadi.org/
   - JAX: https://github.com/google/jax

6. **Climate model intercomparison (CMIP)**: 这是 alignment 思想在物理 science 上的大规模实践
   - https://www.wcrp-climate.org/wgcm-cmip

7. **AI safety / interpretability**: Anthropic 的 "model stitching" 与 "circuits across models" 思路与此方法论同源
   - https://transformer-circuits.pub/

---

## 8. Build Intuition: 你应该 take away 什么

1. **Alignment ≠ Reimplementation**: reimplementation 从一开始 target reproduce；alignment 是两个 independent design 的 model 在相同 phenomenon 上 produce equivalent output 的 stress test。

2. **三个 equivalence level** 是值得记住的 mental model:
   - Numerical identity (基本不可能 for stochastic)
   - Distributional equivalence (本文 target)
   - Relational equivalence (最弱但常足够)

3. **Stochastic equivalence 的 statistic 要谨慎**: 传统 "fail to reject $H_0$" 在小 sample 下虚高，需要反转 null + bootstrap 这种 computational 方法。这个 1995 年提出的 issue 在今天 ML reproducibility crisis 仍然没完全解决。

4. **Subtle implementation differences matter**: with/without replacement activation、修改 active agent vs neighbor——这些表面 trivial 的细节会在某些 parameter regime（20x20 case）下爆出 statistically significant 差异。今天写 PyTorch code 时 random shuffle vs torch.randint 这种选择同理。

5. **Alignment 的真正 value 是 sensitivity analysis 的 anchor**: dock 完成后，richer model 可以把 simpler model 当 baseline，问"如果我 relax 这个 assumption 会怎样"。本文 mobility/soup 实验就是这个 pattern 的范例。在 RL 里这等价于 "vanilla PPO baseline + 我加的 trick 在此基础上 ablation"。

6. **Mathematical structure**:
   - Mann-Whitney $U$ 用于 rank-based two-sample test，critical value 23 for $n_1=n_2=10$
   - K-S $D = \sup_x|F_1-F_2|$，critical value 0.304 for $n=40, \alpha=0.05$
   - 当你下次想 compare 两个 ABM 或两个 RL algorithm 的 reward distribution，这两个 test 仍然是 baseline 选择

7. **Non-monotonic region-vs-lattice-size** 这种 counterintuitive 现象是 ABM 的 typical signature——linear intuition 不 work，需要 explicit simulation。这也是为什么 alignment 必要：单一 model 的 surprise result 可能是 artifact，两个 independent model 都产生同样 surprise 才是 robust。

8. **Connection to "phase transition"**: mixing 推迟 phase transition 但降低 final diversity。这个 dynamic insight 是 single-model experiment 看不出来的，只有跨 model 的 sensitivity sweep 才能浮现。

这篇 paper 在 2026 年读仍然 fresh，因为 alignment/docking 这件事在 deep learning、LLM benchmark、multi-agent simulation 各领域都还没真正标准化。Karpathy 自己在做 Eureka Labs 时的 evaluation pipeline 与此方法论精神一致：cross-implementation consistency 是真理性的最低门槛。
