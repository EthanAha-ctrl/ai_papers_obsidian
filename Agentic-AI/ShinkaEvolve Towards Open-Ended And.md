---
source_pdf: ShinkaEvolve Towards Open-Ended And.pdf
paper_sha256: ee287343c53240126e16a444d5ade2cd0ad0f290a400e78b97a040b93d965087
processed_at: '2026-08-12T05:50:01-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ShinkaEvolve 用人话讲

Andrej，我把刚才那篇 paper 的技术细节嚼碎了，用大白话重新讲一遍，但所有公式和变量含义都保留，让你能 build intuition。

---

## 1. 这工作到底在干啥

**一句话版本**：用 LLM 当 mutation operator，跑一个 evolutionary loop，让 program 自动进化变强。核心卖点是用 150 个 samples 就打了 DeepMind AlphaEvolve 需要 thousands of evaluations 才能打的 circle packing benchmark。

**为什么这有价值**：
- AlphaEvolve 是 closed-source，只有 DeepMind 内部能玩
- OpenEvolve 是 community 复现，但 sample efficiency 不好
- ShinkaEvolve 把 sample efficiency 提了一个数量级，还全 open-source

GitHub: [https://github.com/SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve)

---

## 2. 整个 framework 的 mental model

想象你有一个 archive（数据库），里面存着之前跑过的所有 program 及其 fitness score。每个 iteration 做三件事：

**第一件事**：从 archive 里挑一个 parent program，再挑几个 inspiration programs 当参考。怎么挑？这就是 parent sampling 的问题。

**第二件事**：让 LLM 看 parent + inspirations，propose 一个 mutation（改代码）。但 LLM 经常生成跟已有 program 几乎一样的东西，所以要先 reject 掉重复的，这叫 novelty rejection sampling。

**第三件事**：跑新生成的 program，拿 fitness score，存回 archive。同时用 bandit 算法更新每个 LLM 的"信任度"，让表现好的 LLM 下一轮更可能被选中。

就这么循环 150 次，archive 里的 best program 就进化到 SOTA 了。

---

## 3. Parent Sampling：怎么挑 parent

这是第一个核心创新。给定一个 island subpopulation（一堆 candidate programs），你要挑哪个当 parent 去 mutate。

### 3.1 为什么不能纯 random

纯 random（uniform sampling）的问题：浪费 budget 在烂 program 上。Archive 里大部分 program 是 average 水平，随机挑一个大概率是 average，mutate 出来也好不到哪去。

### 3.2 为什么不能纯 greedy

纯 greedy（hill climbing，永远挑 best program）的问题：初期涨得快，但很快 plateau。因为 best program 已经在 local optimum 附近，mutate 它大概率生成更差的或者差不多的，search 就卡住了。

### 3.3 ShinkaEvolve 的解法：weighted sampling

把 performance 和 novelty 两个信号 combine。公式拆开看：

**Performance 部分**：
$$s_i = \sigma(\lambda \cdot (F(P_i) - \alpha_0))$$

变量：
- $F(P_i)$：program $i$ 的 fitness
- $\alpha_0$：island 内所有 program fitness 的 median
- $\lambda$：selection pressure，paper 里设 10.0
- $\sigma(x) = 1/(1+e^{-x})$：sigmoid 函数

Intuition：把每个 program 的 fitness 跟 median 比，比 median 好的 $s_i > 0.5$，比 median 差的 $s_i < 0.5$。Sigmoid 压扁了绝对差异，避免某个 outlier fitness 的 program 把概率全拿走。

**Novelty 部分**：
$$h_i = \frac{1}{1 + N(P_i)}$$

变量：
- $N(P_i)$：program $i$ 已经产生过多少 offspring

Intuition：offspring 越少的 program，novelty 越高。如果某个 program 已经被 mutate 过 10 次，$h_i = 1/11 \approx 0.09$，基本没机会再被选。如果从未被 mutate 过，$h_i = 1$，full novelty。

这防止"赢家通吃"——某个 high-fitness program 不会无限繁殖把 search space 榨干。

**Combine**：
$$w_i = s_i \cdot h_i, \quad p_i = \frac{w_i}{\sum_j w_j}$$

两个信号相乘再归一化。高 fitness 且少 offspring 的 program 最容易被选中。低 fitness 或已经被 exploit 过的 program 机会少。

**跟 classic GA 的区别**：classic GA 用 tournament selection 或 rank-based selection，只看 fitness。ShinkaEvolve 加了 novelty term 来 explicit balance exploration 和 exploitation，灵感来自 [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)。

---

## 4. LLM Mutation：怎么改代码

### 4.1 三种 mutation 方式

**Diff-based edit**：用 SEARCH/REPLACE block，只改局部。优点是 token 省，缺点是 creative freedom 小。

**Full rewrite**：让 LLM rewrite 整个 mutable block。Creative freedom 大，但 token 贵，容易丢之前的好 structure。

**Crossover**：从 archive 再抽一个 program，让 LLM 把两个 parent 的 idea combine。类似 biological crossover，但操作在 code 层面。

Circle packing 实验里概率分布是 `[diff: 0.45, full: 0.45, cross: 0.1]`，diff 和 full 各占一半，crossover 少用。

### 4.2 Immutable block 保护

代码里有 marker：
```python
# EVOLVE-BLOCK-START
def advanced_algo():
    # 这部分会被 evolve
    return solution
# EVOLVE-BLOCK-END

def run_experiment(**kwargs):
    # 这部分固定不变
    result = solve_problem(kwargs)
    return result
```

LLM 如果在 diff/rewrite 里碰了 immutable 部分，framework 会用 [Reflexion](https://arxiv.org/abs/2303.11366) 给 LLM 反馈"你改错了地方，resample"。最多 resample 3 次。

### 4.3 为什么用多个 LLM

不同 LLM 有不同 coding style 和 strength：
- Claude 可能更 careful
- GPT 可能更 creative
- Gemini 可能更长 context 理解
- DeepSeek 可能更 cost-effective

ShinkaEvolve 用 ensemble，但怎么分配 budget 是 bandit 问题（见下一节）。

---

## 5. Novelty Rejection Sampling：第二个核心创新

### 5.1 问题

LLM 在相似 context（相似 parent + inspirations）下会生成相似 mutation。如果直接 evaluate，浪费 budget。

### 5.2 解法

```
1. LLM 生成 proposal code
2. 用 text-embedding-3-small 对 mutable 部分做 embedding
3. 跟 island 里所有已有 program 的 embedding 算 cosine similarity
4. 如果 max similarity > 0.95：
   a. Query 另一个 LLM："这个新 program 跟已有的 meaningfully different 吗？"
   b. 如果不是 → resample 新 proposal
5. 否则接受，进入 evaluation queue
```

### 5.3 为什么 embedding similarity 是 good proxy

Code embedding 编码了 semantic similarity。两段代码如果 logic 结构类似，embedding 就接近。如果 algorithm 完全不同，embedding 远。

Ablation（Figure 9 right）显示：embedding-based rejection 已经 substantial improvement，加 LLM-as-judge 只有 marginal gain。说明 embedding similarity 已经够用，不需要额外 LLM call 来判断 novelty。

### 5.4 跟 classic novelty search 的关系

Classic novelty search（[Lehman & Stanley 2011](https://direct.mit.edu/evco/article-abstract/19/2/189/835/Abandoning-Objectives-Evolution-Through-the-Search)）需要 hand-design behavior characterization 来 measure novelty。ShinkaEvolve 用 code embedding 当 behavior characterization，这是 LLM 时代特有的 shortcut——pretrained embedding space 已经编码了 code semantic。

---

## 6. UCB1 Bandit：第三个核心创新

### 6.1 问题

有多个 LLM 可用，每个 generation 要选一个 LLM 来 generate mutation。怎么选？

如果固定用 uniform sampling，可能浪费 budget 在不适用的 LLM 上。如果固定用 best LLM，可能错过 exploration。

### 6.2 Standard UCB1

[UCB1 (Auer et al. 2002)](https://link.springer.com/article/10.1023/A:1013689704352) 是经典 multi-armed bandit 算法：
$$\text{UCB}_i = \bar{X}_i + \sqrt{\frac{2 \ln N}{n_i}}$$

变量：
- $\bar{X}_i$：arm $i$ 的 empirical mean reward
- $N$：total pulls
- $n_i$：arm $i$ 被 pull 的次数

选 argmax UCB$_i$。第一项 exploit（用历史表现好的），第二项 explore（少被试的给 bonus）。

### 6.3 ShinkaEvolve 的关键改动：relative reward

直接用 absolute fitness $r_i$ 作为 reward 有问题：archive 是 non-stationary 的，baseline fitness 会随时间上升。

举个例子：
- Evolution 初期，archive best fitness = 1.5，LLM A 找到 1.6，absolute reward = 1.6
- Evolution 后期，archive best fitness = 2.5，LLM B 找到 2.51，absolute reward = 2.51

如果直接比较，LLM B 的 2.51 > LLM A 的 1.6，bandit 会过度偏向 LLM B。但 LLM A 的 improvement 是 +0.1（big jump），LLM B 的 improvement 是 +0.01（marginal），LLM A 明显更有价值。

ShinkaEvolve 的解法：
$$r_i^u = \exp(\max(r_i - r_i^b, 0)) - 1$$

变量：
- $r_i$：mutation $i$ 的 absolute fitness
- $r_i^b = \max(\text{parent fitness}, \text{initial program fitness})$：baseline reward
- $r_i^u$：normalized reward for bandit update

逐项拆解：

**$\max(r_i - r_i^b, 0)$**：只取 improvement，clip 负数。如果 mutation 比 parent 还差，reward = 0，不奖励也不惩罚。

**$\exp(\cdot)$**：指数放大。improvement 越大，reward 指数级增长。这鼓励 bold mutation 而非 safe small tweak。

**$-1$**：当 improvement = 0 时，$\exp(0) - 1 = 0$，reward 归零。保证 no-improvement 的 LLM 不被 inflate。

**Normalization**：tracked statistics 用来 normalize $r_i^u$，确保 fitness scale 不影响 bandit 决策。

### 6.4 为什么这 work

用 relative improvement 代替 absolute reward，bandit 比较的是"哪个 LLM 在当前 archive state 下最有效"而非"哪个 LLM 产生的 program fitness 最高"。这在 non-stationary 环境下是关键。

Ablation（Figure 9 middle）显示 bandit-based 显著优于 fixed uniform 和 single LLM。

---

## 7. Meta-Scratchpad：跨 generation 的 knowledge distillation

### 7.1 问题

单个 LLM mutation 只看 local context（一个 parent + 几个 inspirations）。跨 generation 的 pattern（比如"SA + SLSQP hybrid 比 pure SA 好"）不会被单个 mutation call 捕获。

### 7.2 解法

每 K generations（circle packing K=10），meta-agent 做总结：

```
[Recent program summaries] → [Global insights] → [Implementation recommendations]
```

这些 recommendations 追加到后续 mutation prompts，给 LLM high-level guidance。

Intuition：meta-scratchpad 是 slow thinking，把多次 mutation 的 implicit knowledge 显式化。跟 [Chain-of-Thought](https://arxiv.org/abs/2201.11903) 精神类似，但作用在 evolution 层面。

Meta model 在不同实验用不同 LLM：circle packing 用 gpt-5-nano，AIME 用 gpt-4.1，ALE-Bench 用 gpt-5-mini。

---

## 8. Circle Packing 实验详解

### 8.1 任务

把 26 个 circles 放进 unit square，maximize $\sum_i r_i$（半径之和），约束 no overlap 且 fully contained。

这是经典 packing problem，有 multiple local optima，naive approach 会 stuck。

### 8.2 结果

| Method | Samples | Score |
|---|---|---|
| AlphaEvolve | thousands | 2.63597770931127 |
| OpenEvolve | thousands | lower |
| **ShinkaEvolve** | **150** | **2.6359828390115476** |

ShinkaEvolve 用 150 samples 找到更好的 solution，sample efficiency 提升一个数量级。

### 8.3 Discovered Solution 的三个 key innovation

看 Listing 2 代码，拆出三个聪明的设计：

**Innovation 1: Golden-angle spiral initialization**

```python
golden_angle = np.pi * (3 - np.sqrt(5))
for i, idx in enumerate(inner_idx):
    angle = i * golden_angle
    centers_init[idx] = [cx + inner_r * np.cos(angle), cy + inner_r * np.sin(angle)]
```

Golden angle $\approx 137.5°$ 是 [phyllotaxis](https://en.wikipedia.org/wiki/Phyllotaxis)（植物叶序）的最优分布 angle。[Vogel model](https://en.wikipedia.org/wiki/Fermat%27s_spiral) 证明这能在 disk 上达到 high density packing。ShinkaEvolve 重新发现了这个数学结构。

Corner 和 edge 放 8 个 strategic circles，中心 1 个，剩下 17 个用 golden-angle spiral 放两层 ring（inner + outer）。

**Innovation 2: Hybrid SLSQP + Simulated Annealing**

```python
# Initial SLSQP refinement
result = minimize(objective_func, x0, method="SLSQP", constraints=constraints, 
                  options={"maxiter": 600, "ftol": 1e-8})

# SA outer loop
for iter_idx in range(sa_iterations):
    # Perturb centers
    # Refine with shorter SLSQP
    refine_result = minimize(objective_func, x0_candidate, method="SLSQP",
                             options={"maxiter": 150, "ftol": 1e-6})
    # SA acceptance criterion
    if new_score > current_score or np.random.rand() < np.exp((new_score - current_score)/temperature):
        accept
```

[SLSQP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) 是 gradient-based constrained optimizer，擅长 local refinement 但容易 stuck local minima。SA 提供 global exploration 的 acceptance criterion $P(\text{accept}) = \min(1, \exp(\Delta E / T))$。

ShinkaEvolve 发现 SA 当 outer loop（perturbation + accept/reject），SLSQP 当 inner refinement，比单独用任何一个都好。这跟 [Basin Hopping](https://arxiv.org/abs/cond-mat/9803344) 思路类似。

**Innovation 3: Local moves + Global ring rotations**

```python
if np.random.rand() < 0.7:
    # Local move: perturb 2-6 circles
    num_to_move = np.random.randint(2, 6)
    indices = np.random.choice(n, num_to_move, replace=False)
    candidate_centers[indices] += np.random.normal(0, perturb_step, size=(num_to_move, 2))
else:
    # Global move: rotate one ring around center
    idx_to_rotate = inner_idx if np.random.rand() < 0.5 else outer_idx
    # Apply rotation matrix
```

70% local move（扰动几个 circle）+ 30% global rotation（整个 ring 旋转）。Local 探索 fine adjustment，global 保留 spiral 结构整体旋转 escape local optima。这是 structure-preserving mutation，比 naive Gaussian perturbation 高效。

**Innovation 4: Reheating**

```python
if iter_idx - last_improve > stagnation_limit:
    temperature = initial_temperature
    perturb_step = initial_perturb_step
```

长时间没改进时重置 temperature，escape plateau。经典 SA trick。

### 8.4 Evolution Tree

Figure 5 right 展示 program evolution tree。可以看到 multiple branches 探索不同 algorithmic approaches，high-fitness solutions 繁殖更多 offspring，最终收敛到一条最优路径。Stepping stones 现象明显——suboptimal 中间 solution 是 breakthrough 的 building block。

---

## 9. AIME Agent Scaffold 实验详解

### 9.1 任务

AIME 2024 共 30 题数学竞赛题，design agent scaffold，每题最多 10 LLM queries。Base model 用 gpt-4.1-nano，每个 candidate 跑 3 次取平均。Evolution 跑 75 generations。

### 9.2 Discovered Three-Stage Architecture

看 Listing 3 代码：

**Stage 1: Diverse Expert Personas (3 LLM calls, temperature 0.7)**

```python
self.expert_personas = [
    "You are a meticulous and cautious mathematician. 'slow and steady wins the race'...",
    "You are a brilliant and intuitive mathematician, known for finding elegant, non-obvious solutions...",
    "You are a mathematician with a strong background in computer science. You think in terms of states, transitions, and recurrence relations..."
]
```

三个 persona 对应三种 problem-solving style：careful step-by-step、intuitive pattern recognition、algorithmic CS-style。Temperature 0.7 balance creativity 和 reliability。

**Stage 2: Critical Peer Review (3 LLM calls, temperature 0.1)**

```python
reviewer_system_prompt = "You are a skeptical peer reviewer... Do not accept any statement at face value... If the solution relies on a pattern, you MUST test it on several new examples."
```

Reviewer 在低 temperature（0.1）做 critical review，特别强调 pattern verification（防止 LLM hallucinate pattern）。

**Stage 3: Synthesis (1 LLM call, temperature 0.0)**

```python
synthesizer_system_prompt = "You are the master mathematician and editor, synthesizing multiple reviewed solutions into one canonical, correct answer."
```

Editor-in-chief 在 temperature 0.0（完全 deterministic）综合所有 (solution, critique) pairs，构造 canonical solution。

**Fallback mechanism**：synthesis → majority vote on reviewed → majority vote on original → default 0。三层 fallback 保证 robustness。

**Total: 7 LLM calls** (3 generation + 3 review + 1 synthesis)

### 9.3 为什么这个 architecture 好

核心 insight：**diversity 在 generation 阶段（temperature 高），rigor 在 review 阶段（temperature 低），determinism 在 synthesis 阶段（temperature 0）**。

这个 temperature cascade pattern 是 ShinkaEvolve 自动进化出来的，跟 [Self-Consistency](https://arxiv.org/abs/2203.11171) 和 [Self-Critique](https://arxiv.org/abs/2305.02655) 的手工设计思路一致。

### 9.4 Generalization

- AIME 2023：小幅 improvement（可能 training contamination saturation）
- AIME 2025：较大 improvement（证明 generalizable）
- Cross-LLM：scaffold 迁移到 gpt-4.1-mini, gpt-4.1, o4-mini 都 work

---

## 10. ALE-Bench 实验详解

### 10.1 Setup

[ALE-Bench](https://arxiv.org/abs/2506.09050) 是 AtCoder heuristic programming contests。用 [ALE-Agent](https://arxiv.org/abs/2506.09050) 发现的 best solution 作为 initial program，跑 50 generations 改进。LITE subset 10 个 problem。

### 10.2 整体结果

平均改进 ~2.3%。其中 ahc039 从第 5 名升到第 2 名。

### 10.3 ahc039 详细改动

**Task**：找 axis-aligned polygon 最大化 (mackerels contained) - (sardines contained)，给定 grid 上的 fish 位置。

**Base solution (ALE-Agent)**：Simulated Annealing with kd-tree（5th, score 2880）

**ShinkaEvolve improvements (2nd, score 3140)**：

**Improvement 1: kd-tree with cached subtree statistics**

```cpp
struct KDNode {
    Point pt;
    int axis;
    KDNode *left, *right;
    int fish_struct_idx;
    // Subtree bounding box
    int min_x, max_x, min_y, max_y;
    // Subtree counts
    int m_cnt = 0, s_cnt = 0;
};
```

每个 kd-tree node 存子树的 bounding box 和 mackerel/sardine count。Query rectangle 时可以 whole-subtree prune + aggregate，避免遍历到叶子。

Intuition：SA 每次 move 都要 query 一个 rectangle 内的 fish count。如果每次 O(N) 遍历，SA 跑不了多少 iterations。Cached subtree statistics 把 query 降到 O(√N + k)。这是经典 [range tree](https://en.wikipedia.org/wiki/Range_tree) optimization。

**Improvement 2: Targeted edge move**

```cpp
// Find misclassified fish
if ((target_fish.type == 1) == is_inside) {
    // Misclassified! Find nearest edge and move it
    for (size_t i = 0; i < poly.size(); ++i) {
        d_sq = point_segment_dist_sq_ortho(target_fish.p, poly[i], poly[(i+1) % poly.size()]);
        if (d_sq < min_dist_sq) { best_edge_idx = i; }
    }
    // Move that edge to include/exclude the fish
}
```

Targeted move 是 guided mutation：找误分类的 fish，greedily 移最近的 edge 修正。比纯随机 edge move 方向性强。

### 10.4 ahc025 详细改动

**Task**：用 balance scale 比较 item subset weights，分 D 组最小化组间 weight variance。

**Improvements**：faster caching in QueryManager、refined fallback weight estimation、**替换 SA 为 greedy + targeted local search**。

ShinkaEvolve 自己决定 SA 不够好，换成更 targeted 的 local search。这是 evolution 发现 algorithm-level 改动的例子。

---

## 11. MoE Load Balancing Loss 实验

这是最有 scientific discovery 味道的实验。

### 11.1 MoE 背景

[Mixture-of-Experts (MoE)](https://arxiv.org/abs/1701.06538) 把 FFN 替换成多个 expert，router 选 top-K：
$$y_\ell(x) = \sum_{i=1}^{N_E} g_{\ell,i}(x) E_{\ell,i}(x)$$

问题：top-K selection 是 non-differentiable，需要 auxiliary load balancing loss (LBL) 防止 router collapse。

### 11.2 Standard Global-Batch LBL

[Shazeer et al. 2017](https://arxiv.org/abs/1701.06538)，[Qwen3](https://arxiv.org/abs/2505.09388) 在用：
$$L_{LBL} = N_E \cdot \frac{1}{L} \sum_{\ell=1}^{L} \sum_{i=1}^{N_E} f_{\ell,i} \cdot P_{\ell,i}$$

变量：
- $N_E$：expert 数量
- $L$：layer 数量
- $f_{\ell,i}$：expert $i$ 在 layer $\ell$ 的 selection frequency（实际路由比例）
- $P_{\ell,i}$：expert $i$ 在 layer $\ell$ 的 average router probability（soft routing）

Intuition：$f \cdot P$ 鼓励 alignment——soft probability 高的 expert 也应该实际被选得多。

### 11.3 Global-Batch LBL 的 Blind Spot

考虑极端 case：64 个 expert 中 60 个完全没被用过，4 个平分所有 token。

$f \cdot P$ 在 dead experts 上 = 0，在 alive experts 上 = $0.25 \times 0.25 = 0.0625$，sum = $4 \times 0.0625 = 0.25$，乘 $N_E = 64$ 得 16。Loss 表面看起来"balanced"但 60 个 dead experts 没被任何 term 直接 penalize。

更微妙 case：60 个 expert 各被 0.001 的 tokens 选中，4 个 expert 各被 0.24 选中。$f \cdot P$ sum 看起来"almost balanced"但 60 个 expert 接近 dead。

### 11.4 ShinkaEvolve Discovered LBL

在 global-batch LBL 基础上加新 term：
$$L_{LBL} = \underbrace{N_E \cdot \frac{1}{L} \sum_{\ell=1}^{L} \sum_{i=1}^{N_E} f_{\ell,i} P_{\ell,i}}_{\text{Global-batch LBL}} + \underbrace{\frac{0.1}{L} \sum_{\ell=1}^{L} s(P_\ell) \sum_{i=1}^{N_E} \max(0, \tau - f_{\ell,i})}_{\text{ShinkaEvolve new regularization}}$$

新 term 变量：
- $\tau = 0.064 / N_E$：minimum usage threshold（$N_E = 64$ 时 $\tau = 0.001$）
- $s(P_\ell) = 0.5 + (1 - \frac{H(P_\ell)}{\log N_E})$：normalized complement of routing entropy
  - $H(P_\ell) = -\sum_i P_{\ell,i} \log P_{\ell,i}$：routing entropy
  - $\log N_E$：max entropy（uniform routing）
  - Uniform routing: $s = 0.5$（weak push）
  - Concentrated routing: $s \to 1.5$（strong push）

逐项 intuition：

**$\max(0, \tau - f_{\ell,i})$**：hinge loss。如果 expert $i$ 的 selection frequency $f_{\ell,i} < \tau$，penalize 缺口；如果 $f_{\ell,i} \geq \tau$，penalty = 0。Well-used experts 不被 over-regularize。

**$\sum_i \max(0, \tau - f_{\ell,i})$**：layer 内所有 underused expert 的 cumulative gap。

**$s(P_\ell)$ multiplier**：router 已经 concentrated（low entropy）时 push 更强，已经 uniform 时 push 弱。Adaptive regularization——只在真有问题时介入。

**Safety net intuition**：一旦 expert 跨过 floor $\tau$，penalty 对那个 expert vanish，不再 over-regularize。避免 [Switch Transformer](https://arxiv.org/abs/2101.03961) 等 model 中 LBL 过强导致 router 不敢 specialize 的问题。

### 11.5 实验验证

**Evolution setup**：
- Small MoE: 556M params, $N_E = 64$, $K = 8$（82M sparse activated）
- Train 2B fineweb tokens
- Fitness = $-(L_{CE} + L_{imb})$
- $L_{imb} = \frac{1}{2} \sum_i |f_{\ell,i} - 1/N_E|$
- 30 generations（pretraining 太贵）

**Evaluation**：
- Scale 到 2.7B MoE（404M active）
- Train 30B fineweb tokens
- Compare across $\lambda \in \{0.001, 0.01, 0.1\}$
- 7 个 downstream benchmark

**Results**：ShinkaEvolve LBL consistently 优于 global-batch LBL，improvement 随 $\lambda$ 增大而增大（$\lambda$ 大时 standard LBL 的 blind spot 越严重）。

### 11.6 这意味着什么

30 generations 的 evolution 找到了 [Qwen3](https://arxiv.org/abs/2505.09388)、[DeepSeek-MoE](https://arxiv.org/abs/2401.06066) 等 SOTA model 用的 LBL 的改进版。这是 LLM-driven ML architecture discovery 的 proof-of-concept。

ShinkaEvolve 没有 replace global-batch LBL，只是 augment 它。说明 evolution 也"知道"基础 LBL 有用，补它的 blind spot。这种 incremental but principled improvement 是 LLM-as-mutation-operator 的 emergent property。

---

## 12. Ablation Studies

### 12.1 Parent Selection (Figure 9 left)

- **Best-of-N**：忽略 history，用 initial program
- **Hill Climbing**：贪心选 best program
- **Weighted Sampling**：performance + novelty

结果：Weighted > Hill Climbing > Random。Hill climbing 初期强但 plateau 快，weighted sampling 保持 steady improvement。

### 12.2 LLM Ensembling (Figure 9 middle)

- Single LLM (GPT-5-nano)
- Fixed LLM Ensemble (uniform)
- Bandit-Based LLM Ensemble (UCB1)

Bandit 显著最优。Fixed ensemble 比 single 有 moderate improvement，bandit 通过 dynamic prioritization 进一步拉开。

### 12.3 Rejection Sampling (Figure 9 right)

- No Rejection
- Embedding-Based Rejection (cosine sim > 0.95)
- + LLM-as-novelty-judge

Embedding-based 已经 substantial gains。LLM judge 仅 marginal improvement，说明 embedding similarity 已经够用。

---

## 13. 跟 Related Work 的对比

### 13.1 vs AlphaEvolve

[AlphaEvolve (Novikov et al. 2025)](https://arxiv.org/abs/2506.13131)：closed-source，thousands of evaluations，evolutionary + LLM。

ShinkaEvolve 用 150 samples 超过 AlphaEvolve 的 circle packing，关键差异：UCB1 bandit、novelty rejection、weighted parent sampling、meta-scratchpad。

### 13.2 vs OpenEvolve

[OpenEvolve](https://github.com/codelion/openevolve)：open-source baseline，ShinkaEvolve 借鉴 API 接口，但 sample efficiency 更好。

### 13.3 vs AI Scientist v2

[AI Scientist v2](https://arxiv.org/abs/2504.08066) 是 Sakana AI 之前的工作，agentic tree search for scientific discovery。ShinkaEvolve 是它在 code evolution 方向的 refinement。

### 13.4 vs Eureka

[Eureka](https://arxiv.org/abs/2310.12931) 用 LLM 设计 RL reward function，evolutionary loop。ShinkaEvolve 是更 general 的 framework。

### 13.5 vs AI CUDA Engineer

[AI CUDA Engineer](https://sakana.ai/ai-cuda-engineer) 是 Sakana AI 之前的工作，evolutionary CUDA kernel optimization。ShinkaEvolve 是它的 methodology 进化版。

### 13.6 vs Darwin Gödel Machine

[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) 做 self-improving agent 的 open-ended evolution。ShinkaEvolve 借鉴了它的 weighted parent sampling 思路，但更聚焦 fixed task 的 sample-efficient optimization。

---

## 14. 几个关键 Intuition

### 14.1 为什么 evolution 比 pure LLM sampling 强

- LLM 单次 generation 容易 stuck local optimum
- Evolution 提供 iterative refinement with feedback
- Archive 提供 memory，LLM 不需要每次从头生成
- Diversity 机制（island, novelty rejection）保持 exploration

跟 [AlphaCode](https://www.science.org/doi/10.1126/science.abj9343) 等 LLM-for-code 工作的区别：AlphaCode 用 massive sampling + filter，ShinkaEvolve 用 guided evolutionary search。

### 14.2 Sample efficiency 的来源

三个创新各自贡献：
- **Weighted parent sampling**：避免对 top program 过度 exploit
- **Novelty rejection**：避免重复 evaluate 几乎相同的 program
- **UCB1 bandit**：让 best LLM for 当前 phase 更频繁被用

Ablation 显示三者协同——单独任何一个都不够。

### 14.3 跟 Test-Time Compute Scaling 的关系

ShinkaEvolve 本质是 structured test-time compute scaling：用更多 LLM calls 换更好 solution。跟 [OpenAI o1](https://openai.com/o1/) style CoT 的区别：ShinkaEvolve 在 search space 上 explore more，o1 在 reasoning chain 上 think more。两者可以结合——reasoning model 当 mutation operator 在 evolution loop 内 think more。

---

## 15. Limitations 和 Future Work

### 15.1 Limitations

1. **Manual task specification**：objective function 和 evaluation 需要 human expertise
2. **Numerical objectives only**：局限于 well-defined numerical fitness
3. **API cost barrier**：大规模 LLM usage 有经济门槛
4. **Fixed exploration-exploitation balance**：需要 hyperparameter tuning
5. **ALE-Bench 上倾向 local refinement**：可能 overfitting to initialization

### 15.2 Future Directions

1. **Automated task specification**：LLM 自己 generate objectives
2. **True open-endedness**：system 自己 generate objectives（[OMNI-EPIC](https://arxiv.org/abs/2405.15568)）
3. **Self-referential refinement**：framework evolve 自己的 evolution strategy
4. **Online meta-learning**：持续 improve discovery process

---

## 16. Open Questions 和联想

### 16.1 跟 Quality-Diversity (QD) 的关系

Novelty rejection sampling 跟 [MAP-Elites](https://arxiv.org/abs/1504.04909) 等 QD 算法精神类似，但 ShinkaEvolve 用 code embedding 当 behavior characterization，这是 LLM 时代 QD 的新形态。

### 16.2 跟 Neuroevolution 的关系

Classic neuroevolution（[NEAT](https://arxiv.org/abs/1909.13184)）evolve neural network weights 和 topology。ShinkaEvolve evolve 的 program 更广义——algorithm、agent scaffold、loss function。这是 neuroevolution 的 superset。

### 16.3 跟 AutoML 的关系

[AutoML](https://www.automl.org/) 用 search 找 ML architecture。ShinkaEvolve 在 MoE LBL 实验里展示的是 AutoML 的 LLM-driven 版本——LLM 当 mutation operator 替代 random search 或 Bayesian optimization。

### 16.4 Self-Referential Evolution

如果让 ShinkaEvolve evolve 自己的 evolution strategy（parent sampling strategy, novelty threshold, bandit coefficient），会怎么样？这指向 [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) 的 self-improvement 模式，但 applied to meta-evolution。

### 16.5 Multi-Modal Programs

当前 programs 都是 code。如果 program 是 multimodal（code + 自然语言 rationale + 视觉 sketch），evolution 怎么处理？需要 multi-modal LLM 和 multi-modal embedding。

### 16.6 Continual Learning

ShinkaEvolve 每个 task 从头跑。如果 archive 能跨 task transfer（类似 [meta-learning](https://arxiv.org/abs/1904.10690)），新 task 可以 reuse 之前学到的 strategies。这是 future work 里 "self-referential refinement" 的方向。

### 16.7 跟 AlphaEvolve 后续的可能性

DeepMind 的 AlphaEvolve 发现了新数学结果（[Nature paper](https://www.nature.com/articles/s41586-023-06923-3)），ShinkaEvolve 的 sample efficiency 改进如果能让 community 复现这种 discovery level 的 search，可能解锁大量自动数学发现。MoE LBL 实验已经展示了这种潜力。

### 16.8 跟 LLM Inference Time Scaling Law

最近 [OpenAI o1](https://openai.com/o1/)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 推动 inference time scaling。ShinkaEvolve 是另一种 inference time scaling：用更多 LLM calls 做 evolutionary search。两者可以结合——reasoning model 当 mutation operator 在 evolution loop 内 think more。Paper 提到用 o4-mini 在 LLM ensemble 里，但没深入。

---

## 17. 总结

ShinkaEvolve 的核心价值：

1. **Sample efficiency**：150 samples vs thousands，量级提升
2. **Open-source**：打破 AlphaEvolve 的 closed-source 壁垒
3. **Three synergistic innovations**：weighted parent sampling + novelty rejection + UCB1 bandit
4. **Four diverse tasks**：从 mathematical optimization 到 agent design 到 ML architecture discovery
5. **MoE LBL discovery**：证明 LLM 可以做 ML architecture research

真正的价值不在某个具体 trick，在 **整个 framework 的 sample efficiency 让 LLM-driven discovery 从 "big lab only" 变成 "any researcher can play"**。后续 community 在这基础上改进、扩展、应用到新 domain 的潜力很大。

---

## 参考链接

GitHub repo: [https://github.com/SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve)

Sakana AI: [https://sakana.ai/](https://sakana.ai/)

AlphaEvolve paper: [https://arxiv.org/abs/2506.13131](https://arxiv.org/abs/2506.13131)

OpenEvolve: [https://github.com/codelion/openevolve](https://github.com/codelion/openevolve)

ALE-Bench: [https://arxiv.org/abs/2506.09050](https://arxiv.org/abs/2506.09050)

AI Scientist v2: [https://arxiv.org/abs/2504.08066](https://arxiv.org/abs/2504.08066)

Darwin Gödel Machine: [https://arxiv.org/abs/2505.22954](https://arxiv.org/abs/2505.22954)

Eureka: [https://arxiv.org/abs/2310.12931](https://arxiv.org/abs/2310.12931)

LLM4AD: [https://arxiv.org/abs/2412.17287](https://arxiv.org/abs/2412.17287)

UCB1 paper: [https://link.springer.com/article/10.1023/A:1013689704352](https://link.springer.com/article/10.1023/A:1013689704352)

Shazeer MoE LBL: [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)

Switch Transformer: [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)

Qwen3: [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

DeepSeek-MoE: [https://arxiv.org/abs/2401.06066](https://arxiv.org/abs/2401.06066)

Reflexion: [https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)

Self-Consistency: [https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)

Tree of Thoughts: [https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)

Self-Refine: [https://arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651)

MAP-Elites: [https://arxiv.org/abs/1504.04909](https://arxiv.org/abs/1504.04909)

NEAT: [https://arxiv.org/abs/1909.13184](https://arxiv.org/abs/1909.13184)

AlphaCode: [https://www.science.org/doi/10.1126/science.abj9343](https://www.science.org/doi/10.1126/science.abj9343)

OpenAI o1: [https://openai.com/o1/](https://openai.com/o1/)

DeepSeek-R1: [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)

OMNI-EPIC: [https://arxiv.org/abs/2405.15568](https://arxiv.org/abs/2405.15568)

Chain-of-Thought: [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

Novelty Search: [https://direct.mit.edu/evco/article-abstract/19/2/189/835/Abandoning-Objectives-Evolution-Through-the-Search](https://direct.mit.edu/evco/article-abstract/19/2/189/835/Abandoning-Objectives-Evolution-Through-the-Search)

---

# ShinkaEvolve 深度技术解析

Andrej, 这篇 paper 是 Sakana AI 在 LLM-driven evolutionary discovery 路线上的一个 milestone。它直接对标 DeepMind 的 AlphaEvolve，但用 **150 samples** 打了 AlphaEvolve 需要 **thousands of evaluations** 才能解决的问题，sample efficiency 提升了一个数量级。我下面把整个 framework 拆开，从 control flow 到每条 loss 的物理直觉都讲清楚。

---

## 1. 整体定位：这工作在 lineage 上的位置

要先理解 ShinkaEvolve 在做什么，看清楚它的 ancestry：

- **Evolutionary computation + LLM as mutation operator** 这条 line 起源于 Lehman & Stanley 的 novelty search（[LEHMAN2011](https://direct.mit.edu/evco/article-abstract/19/2/189/835/Abandoning-Objectives-Evolution-Through-the-Search)），后续 Meyerson et al. 的 Language Model Crossover（[arXiv:2302.12170](https://arxiv.org/abs/2302.12170)）第一次把 LLM 当 crossover engine 用。
- **Eureka**（[arXiv:2310.12931](https://arxiv.org/abs/2310.12931)）用 LLM 当 reward designer for RL。
- **AlphaEvolve**（[arXiv:2506.13131](https://arxiv.org/abs/2506.13131)）把这条 line 推到 scientific discovery 级别，发现了新的数学结果，但 closed-source。
- **OpenEvolve**（[github.com/codelion/openevolve](https://github.com/codelion/openevolve)）是 community 的开源复现，但 sample efficiency 仍然不够好。
- **AI Scientist v2**（[arXiv:2504.08066](https://arxiv.org/abs/2504.08066)）和 **AI CUDA Engineer**（Sakana AI 自己的工作）已经在这个方向上积累了经验。
- **Darwin Gödel Machine**（[arXiv:2505.22954](https://arxiv.org/abs/2505.22954)）做 self-improving agent 的 open-ended evolution。

ShinkaEvolve 在这条 line 上的核心 contribution 就是：**把 evolutionary search 的 sample efficiency 从 "需要 thousands of evaluations" 压到 "150 evaluations"，并且全 open-source**。

GitHub repo: [https://github.com/SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve)

---

## 2. Control Flow 总览

整个 ShinkaEvolve 的 evolution loop 由三个 phase 组成：

```
[ Archive of island subpopulations ]
        ↓
(1) Parent + inspiration sampling
        ↓
(2) LLM-guided program mutation + novelty rejection sampling
        ↓
(3) Execute program + world feedback
        ↓
[ Update archive, update UCB1 bandit, update meta-scratchpad ]
        ↓
   back to (1)
```

Archive 是固定大小的，带 elite size constraint，用 **island model**（[Tanese 1989](https://deepblue.lib.umich.edu/handle/2027.42/29278)）维护多个独立 subpopulations，island 之间偶尔 migration 来扩散 knowledge，但 **island-specific best program 不允许 migrate** 来保护独特性。这跟 classic GA 的 island model 思路一致，避免 premature convergence。

---

## 3. Parent Sampling：Exploration vs Exploitation 的形式化

这部分是 ShinkaEvolve 第一个核心创新。给定一个 island subpopulation，选 parent 的策略直接影响 search trajectory。

### 3.1 Power Law Sampling

程序按 fitness 排序，rank $r_i = 1$ 对应 best program。selection probability：

$$p_i = \frac{r_i^{-\alpha}}{\sum_{j=1}^n r_j^{-\alpha}}$$

变量含义：
- $r_i \in \{1, 2, \ldots, n\}$：program $i$ 在 island 中的 fitness rank
- $\alpha \geq 0$：exploitation intensity 参数
- $n$：island 大小

边界条件：
- $\alpha = 0$：所有 $r_i^0 = 1$，所以 $p_i = 1/n$，退化为 uniform sampling（pure exploration）
- $\alpha \to \infty$：只有 $r_i = 1$ 的项主导，退化为 hill-climbing（pure exploitation）

这是经典 rank-based selection 的 power law 形式，类似 tournament selection 的连续 relax。

### 3.2 Weighted Sampling（performance + novelty）

这是 ShinkaEvolve 实际推荐的方法，灵感来自 [Darwin Gödel Machine (arXiv:2505.22954)](https://arxiv.org/abs/2505.22954)。给定每个 program $P_i$ 的 offspring count $N(P_i)$：

**Step 1**: 计算 island 内 fitness 的 median 作为 reference：
$$\alpha_0 = \text{median}(\{F(P_1), F(P_2), \ldots, F(P_n)\})$$

**Step 2**: Performance component 用 sigmoid 缩放，把 fitness 相对 median 的偏移压到 $(0, 1)$：
$$s_i = \sigma(\lambda \cdot (F(P_i) - \alpha_0)), \quad \sigma(x) = \frac{1}{1+e^{-x}}$$

变量含义：
- $F(P_i)$：program $i$ 的 fitness
- $\alpha_0$：median fitness，作为"baseline"参考点
- $\lambda$：selection pressure（hyperparameter，paper 里设 $\lambda = 10.0$）

Intuition：sigmoid 把 fitness 差异压扁，比 median 好的程序 $s_i > 0.5$，比 median 差的 $s_i < 0.5$，但不会饱和到 0 或 1，保留梯度信号。

**Step 3**: Novelty component：
$$h_i = \frac{1}{1 + N(P_i)}$$

变量含义：
- $N(P_i)$：program $P_i$ 已经产生的 offspring 数量

Intuition：offspring 越少的 program，novelty 越高。$h_i \in (0, 1]$，从未繁殖过的 program $h_i = 1$，繁殖了很多次的 program $h_i \to 0$。这是 anti-niche-overcrowding 机制，避免某个"赢家"过度繁殖导致 search 陷入 local basin。

**Step 4**: Combine + normalize：
$$w_i = s_i \cdot h_i, \quad p_i = \frac{w_i}{\sum_{j=1}^n w_j}$$

最终 selection probability $p_i$ 同时奖励 high fitness 和 low offspring count。这个 formulation 的核心 insight 是：**fitness 已经高的 program 当然要 exploit，但如果它已经被 explore 过很多次，应该让位给还没 explore 过的 program**。

Figure 2 在 paper 里直观对比了 uniform、hill-climbing、weighted 三种策略下 offspring count 的分布。

---

## 4. LLM-Guided Program Mutation

ShinkaEvolve 支持 GPT、Gemini、Claude、DeepSeek 等 API providers（[Anthropic Claude 3](https://www.anthropic.com/news/claude-3-model-family), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [GPT-4 technical report](https://arxiv.org/abs/2303.08774), [Gemini 2.5](https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025/)）。

### 4.1 三种 mutation 类型

1. **Diff-based edits**: 用 SEARCH/REPLACE blocks，遵循 AlphaEvolve 的做法。LLM 只 propose 局部修改，减少 token 消耗，保留 architecture。
2. **Full rewrites**: 允许 LLM rewrite 整个 program，但通过 EVOLVE-BLOCK-START / EVOLVE-BLOCK-END markers 保护 immutable code。这给 LLM 更大 creative freedom，适合架构级重组。
3. **Crossover mutation**: 从 archive 额外采样一个 program，prompt LLM 去组合两个 parent 的 idea。这是 [Lehman et al. 2022](https://arxiv.org/abs/2206.08896) 和 [Lange et al. 2025 AI CUDA Engineer](https://sakana.ai/ai-cuda-engineer) 用过的思路。

Patch type 的概率分布在 circle packing 实验里是 `[0.45, 0.45, 0.1]`（diff, full, cross），说明 diff 和 full rewrite 是 main workhorse。

### 4.2 Immutable block 保护机制

代码结构：
```python
# EVOLVE-BLOCK-START
def advanced_algo():
    # This will be evolved
    return solution
# EVOLVE-BLOCK-END

def run_experiment(**kwargs):
    # Fixed, not evolved
    result = solve_problem(kwargs)
    return result
```

如果 LLM 提议的 patch 触碰了 immutable 区域，会用 [Reflexion (Shinn et al., 2024)](https://arxiv.org/abs/2303.11366) 给 LLM 反馈，resample 新的 proposal。Max patch resamples = 3。

---

## 5. Code Novelty Rejection Sampling（第二个核心创新）

这是 sample efficiency 提升的关键之一。问题：LLM 在相似 context 下会生成相似的 mutation，浪费 evaluation budget。

### 5.1 算法流程

```
1. 从 LLM 拿到新 proposal code
2. 用 embedding model (text-embedding-3-small) 对 mutable 部分做 embedding
3. 计算 proposal embedding 与 island 内所有 program 的 cosine similarity
4. 如果 max similarity > threshold (e.g., 0.95):
   a. Query 另一个 LLM 评估 "is this meaningfully different?"
   b. 如果不够 different → resample 新 proposal
5. 否则接受 proposal 进入 evaluation queue
```

### 5.2 为什么这能省 sample

intuition 是：**如果 LLM 提议的 mutation 跟 island 里已经 evaluate 过的 program 在 embedding space 几乎一样，那它的 fitness 也大概率一样**。直接 reject 比浪费一次 evaluation 划算。

Embedding model 用的是 OpenAI 的 `text-embedding-3-small`，cosine similarity threshold 0.95 在 circle packing 和 AIME 实验里都用。Ablation 显示 embedding-based rejection 已经 substantial，再加 LLM-as-novelty-judge 只 marginal improvement（Figure 9 right）。

### 5.3 与 novelty search 的关系

经典 novelty search（[Lehman & Stanley 2011](https://direct.mit.edu/evco/article-abstract/19/2/189/835/Abandoning-Objectives-Evolution-Through-the-Search)）需要一个 explicit behavior characterization 来 measure novelty。ShinkaEvolve 用 code embedding 当 proxy，这是 LLM 时代特有的——pretrained embedding space 本身就编码了 semantic similarity。

---

## 6. UCB1 Bandit for LLM Ensemble Selection（第三个核心创新）

ShinkaEvolve 不是固定用一个 LLM，而是用 multi-armed bandit 动态分配 LLM 调用 budget。这部分是 Edoardo Cetin 贡献的。

### 6.1 标准 UCB1 回顾

[UCB1 (Auer et al., 2002)](https://link.springer.com/article/10.1023/A:1013689704352) 标准形式：
$$\text{UCB}_i = \bar{X}_i + \sqrt{\frac{2 \ln N}{n_i}}$$

变量：
- $\bar{X}_i$：arm $i$ 的 empirical mean reward
- $N$：total pulls
- $n_i$：arm $i$ 被 pull 的次数

选择 argmax UCB$_i$。第二项是 exploration bonus，鼓励少被尝试的 arm。

### 6.2 ShinkaEvolve 的改动：relative improvement reward

这是关键改动。Standard UCB1 直接用 absolute reward $r_i$，但在 evolutionary search 里 archive 是 non-stationary 的：archive 里 program 的 baseline fitness 会随时间上升，导致 absolute reward 通胀。

ShinkaEvolve 用 **relative improvement** 代替 absolute reward：

$$r_i^u = \exp(\max(r_i - r_i^b, 0)) - 1$$

变量含义：
- $r_i$：mutation $i$ 的 absolute fitness
- $r_i^b$：baseline reward，定义为 $\max(\text{parent's fitness}, \text{initial program's fitness})$
- $r_i^u$：用于 update bandit 的 normalized reward

逐项 intuition：

1. **$\max(r_i - r_i^b, 0)$**：clip 负数，只考虑 improvement 而非 regression。一个 LLM 如果 mutate 出更差的 program，不应该被惩罚（因为是 exploration），但也不应该被奖励。

2. **$\exp(\cdot)$**：指数放大。如果一个 LLM 找到了大幅 improvement，奖励指数级增长。这鼓励"bold, high-risk, high-reward" mutations 而非"safe, minor improvements"。

3. **$-1$**：当 $\max(\cdot, 0) = 0$ 时，$r_i^u = \exp(0) - 1 = 0$，没有 reward。这保证"no improvement" 的 LLM 不会因为 exp 而被 inflate。

4. **归一化**：tracked statistics 用来 normalize $r_i^u$，确保 fitness scale 不变，跨 domain 都能用同一套 bandit 逻辑。

### 6.3 为什么 absolute reward 不 work

考虑 circle packing 初期 vs 后期：
- 初期 archive 里 best program 可能 fitness = 1.5，一个 LLM 找到 1.6 的 mutation，absolute reward = 1.6
- 后期 archive best 已经 2.5，一个 LLM 找到 2.51 的 mutation，absolute reward = 2.51

如果直接用 absolute reward，后期那个 marginal improvement 会被认为比初期那个 big jump 更好，bandit 会过度偏向"在已经饱和的 archive 上做小修小补"的 LLM。

用 relative reward $r_i - r_i^b$，初期的 0.1 improvement 和后期的 0.01 improvement 在 bandit 看来是公平比较的。

### 6.4 Ablation 数据

Figure 9 middle 显示：bandit-based 显著优于 fixed uniform ensemble，更优于 single LLM (GPT-5-nano)。Improvement 约 5-10% 在 circle packing 上。

---

## 7. Meta-Scratchpad：Online Knowledge Distillation

每 K generations（circle packing 设 K=10，AIME 设 K=10，ALE-Bench 设 K=5），meta-agent 做一次总结：

```
[Program summaries] → [Global insights] → [Implementation recommendations]
```

这些 recommendations 被追加到后续 mutation prompts，形成 high-level guidance。Meta model 在 circle packing 用 gpt-5-nano，在 AIME 用 gpt-4.1，在 ALE-Bench 用 gpt-5-mini。

Intuition：**单次 mutation 看 local context（一个 parent + 几个 inspirations），但跨 generation 的 pattern 不会被单个 LLM call 捕获**。Meta-scratchpad 是一种"slow thinking"，把多次 mutation 的 implicit knowledge 显式化。

跟 [Chain-of-Thought](https://arxiv.org/abs/2201.11903)、[Self-Refine](https://arxiv.org/abs/2303.17651) 等 test-time compute 方法在精神上类似，但作用在 evolution 层面而非单次 inference。

---

## 8. 实验 1：Circle Packing

### 8.1 任务定义

把 26 个 circles 放进 unit square，maximize $\sum_i r_i$（半径之和），约束：
- No overlap：$\|c_i - c_j\|_2 \geq r_i + r_j$ for all $i \neq j$
- Containment：$r_i \leq c_{i,x} \leq 1 - r_i$ 且 $r_i \leq c_{i,y} \leq 1 - r_i$

### 8.2 结果对比

| Method | Samples | Score (relaxed) | Score (exact) |
|---|---|---|---|
| AlphaEvolve | thousands | ~2.6359 | 2.63597770931127 |
| OpenEvolve | thousands | lower | - |
| LLM4AD / EoH | thousands | lower | - |
| **ShinkaEvolve** | **150** | **2.635983099011548** | 2.6359828390115476 |

关键 insight：ShinkaEvolve 在 relaxed verification（允许 $10^{-6}$ slack）下用 150 samples 找到的 solution，可以通过把每个 radius 减 $10^{-8}$ trivially 转成 exact solution，相对变化 < $10^{-6}$。这说明 relaxed task 可以作为 surrogate 加速 evolution，最后 post-process 到 exact。

### 8.3 Discovered Solution 的三个 key innovation

看 Listing 2 完整代码，可以拆出：

**Innovation 1: Structured initialization with golden-angle spiral**

```python
golden_angle = np.pi * (3 - np.sqrt(5))
# Inner ring (idx 9-14) and outer ring (idx 15-25)
for i, idx in enumerate(inner_idx):
    angle = i * golden_angle
    centers_init[idx] = [cx + inner_r * np.cos(angle), cy + inner_r * np.sin(angle)]
for i, idx in enumerate(outer_idx):
    angle = i * golden_angle * 1.003  # slight phase offset
    centers_init[idx] = [cx + outer_r * np.cos(angle), cy + outer_r * np.sin(angle)]
```

Golden angle = $\pi(3 - \sqrt{5}) \approx 137.5°$，是 [phyllotaxis](https://en.wikipedia.org/wiki/Phyllotaxis)（植物叶序）中的最优分布 angle，被 [Vogel model](https://en.wikipedia.org/wiki/Fermat%27s_spiral) 证明能在 disk 上达到高密度 packing。ShinkaEvolve 重新发现了这个数学结构。

Corner 和 edge 上的 8 个 circles 是手工放的战略位置（4 corners + 4 edge midpoints），中心 1 个，剩下 17 个用 spiral 放。

**Innovation 2: Hybrid SLSQP + Simulated Annealing**

```python
# Initial SLSQP
result = minimize(objective_func, x0, method="SLSQP", bounds=bounds,
                  constraints=constraints, options={"maxiter": 600, "ftol": 1e-8})

# SA loop
for iter_idx in range(sa_iterations):
    # Perturb centers
    # Refine with shorter SLSQP
    refine_result = minimize(objective_func, x0_candidate, method="SLSQP",
                             options={"maxiter": 150, "ftol": 1e-6})
    # SA acceptance
    if new_score > current_score or np.random.rand() < np.exp((new_score - current_score)/temperature):
        accept
```

SLSQP ([Sequential Least Squares Programming](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)) 是 gradient-based constrained optimizer，适合 local refinement 但容易 stuck 在 local minima。SA 提供 global exploration 的 acceptance criterion $P(\text{accept}) = \min(1, \exp(\Delta E / T))$。

ShinkaEvolve 发现把 SA 当 outer loop（perturbation + accept/reject），把 SLSQP 当 inner refinement，比单独用任何一个都好。这跟 [Basin Hopping](https://arxiv.org/abs/cond-mat/9803344) 思路类似。

**Innovation 3: Local moves + Global ring rotations**

```python
if np.random.rand() < 0.7:
    # Local move: perturb a few circles (2-6)
    num_to_move = np.random.randint(2, 6)
    indices = np.random.choice(n, num_to_move, replace=False)
    candidate_centers[indices] += np.random.normal(0, perturb_step, size=(num_to_move, 2))
else:
    # Global move: rotate one of the rings around center
    idx_to_rotate = inner_idx if np.random.rand() < 0.5 else outer_idx
    # ... rotation matrix applied to relative positions
```

70% local + 30% global rotation。Local moves 探索 fine-grained adjustment，global rotation 保留 spiral 结构整体旋转来 escape local optima。这是 **structure-preserving mutation**，比 naive Gaussian perturbation 高效得多。

**Innovation 4: Reheating**

```python
stagnation_limit = sa_iterations // 4
if iter_idx - last_improve > stagnation_limit:
    temperature = initial_temperature
    perturb_step = initial_perturb_step
    last_improve = iter_idx
```

长时间没改进时重置 temperature 和 step size，escape plateau。这是经典 SA 的 trick（[Simulated Annealing with Reheating](https://link.springer.com/chapter/10.1007/978-1-4615-6059-7_8)）。

### 8.4 Evolution Tree 分析

Figure 5 right 展示了 program evolution tree。可以看到：
- Multiple evolutionary branches explore 不同 algorithmic approaches
- High-performing solutions（绿色和黄色）作为 parent 繁殖更多 offspring
- 最终收敛到一条高 fitness 路径（黑色高亮）
- Stepping stones 现象：suboptimal 中间 solution 是 breakthrough 的 building block

---

## 9. 实验 2：AIME Math Reasoning Agent Scaffold Design

### 9.1 任务

AIME 2024 共 30 题，要求 design agent scaffold，每个 problem 最多 10 LLM queries。Base model 用 gpt-4.1-nano。每个 candidate 在完整 30 题上跑 3 次取平均（combat LLM stochasticity）。Evolution 跑 75 generations。

### 9.2 Pareto Frontier

Figure 6 left 显示 ShinkaEvolve 发现了一条 Pareto frontier：
- **7 LLM queries** 达到 maximum performance
- **10 queries** 的 alternative scaffold 达到 comparable performance

这暗示 LLM query 数量和 accuracy 不是 monotonically 相关——存在"少即是多"的 sweet spot。

### 9.3 Generalization 实验

- AIME 2023: 小幅 improvement，可能是 training data contamination 导致 saturation
- AIME 2025: 较大 improvement，证明 scaffold 学到的是 generalizable strategy 而非 memorization
- Cross-LLM：把 scaffold 从 gpt-4.1-nano 迁移到 gpt-4.1-mini, gpt-4.1, o4-mini 都 work

### 9.4 Discovered Three-Stage Architecture

看 Listing 3 完整代码。这是进化出来的 agent scaffold：

**Stage 1: Diverse Expert Personas (3 calls, temperature 0.7)**

```python
self.expert_personas = [
    "You are a meticulous and cautious mathematician. Your guiding principle is 'slow and steady wins the race'. You solve problems by breaking them down into the smallest possible steps based on fundamental principles.",
    "You are a brilliant and intuitive mathematician, known for finding elegant, non-obvious solutions. You look for symmetries, invariants, or a change of perspective that radically simplifies the problem.",
    "You are a mathematician with a strong background in computer science. You approach problems by trying to frame them algorithmically. You think in terms of states, transitions, and recurrence relations."
]
```

三个 persona 分别对应三种 problem-solving style：
- Careful step-by-step
- Intuitive pattern recognition
- Algorithmic / CS-style

Temperature 0.7 平衡 creativity 和 reliability。

**Stage 2: Critical Peer Review (3 calls, temperature 0.1)**

```python
reviewer_system_prompt = "You are a skeptical peer reviewer examining a proposed solution to an AIME problem. Your task is to be extremely critical. Do not accept any statement at face value. Re-read the original problem carefully. Check calculations. Scrutinize the logical flow. **Pattern Verification:** If the solution relies on a pattern, you MUST test it on several new examples."
```

Reviewer 在 temperature 0.1（接近 deterministic），prompt 强调：
- Skeptical stance
- Pattern verification on new examples（防止 LLM hallucinate pattern）
- Identify logical flaws + provide corrections

这是 [Self-Consistency](https://arxiv.org/abs/2203.11171) 和 [Self-Critique](https://arxiv.org/abs/2305.02655) 的组合。

**Stage 3: Synthesis (1 call, temperature 0.0)**

```python
synthesizer_system_prompt = "You are the master mathematician and editor, synthesizing multiple reviewed solutions into one canonical, correct answer."
```

Editor-in-chief 在 temperature 0.0（完全 deterministic），分析所有 (solution, critique) pairs，构造 canonical solution。

**Fallback mechanism**：

```python
if self._extract_answer(final_response) is None:
    # First, trust the reviewed answers
    reviewed_answers = [self._extract_answer(c) for c in critiques]
    if valid_reviewed_answers:
        most_common_answer = Counter(valid_reviewed_answers).most_common(1)[0][0]
    # Then check original solutions
    # Ultimate fallback: \boxed{000}
```

三层 fallback：synthesis → majority vote on reviewed → majority vote on original → default 0。这保证 robustness，跟 [Self-Consistency](https://arxiv.org/abs/2203.11171) 的 majority voting 思路一致。

**Total: 7 LLM calls (3 + 3 + 1)**

这个 architecture 的核心 insight：**diversity 在 generation 阶段（temperature 高），rigor 在 review 阶段（temperature 低），determinism 在 synthesis 阶段（temperature 0）**。ShinkaEvolve 进化出这个 temperature cascade pattern 完全自动。

### 9.5 与 hand-designed agent 的对比

Paper 提到 baseline 包括 simple single-query agents 和 sophisticated majority-voting approaches。Discovered scaffold 在 7 queries 就达到 max performance，比 majority-voting 用 10 queries 更 sample efficient。这呼应了 [Automated Design of Agentic Systems (Hu et al., 2024)](https://arxiv.org/abs/2408.08435) 的核心 idea：agent architecture 可以被 evolved 而非 hand-designed。

---

## 10. 实验 3：ALE-Bench Competitive Programming

### 10.1 Setup

[ALE-Bench](https://arxiv.org/abs/2506.09050) 是 AtCoder heuristic programming contests 集合。ShinkaEvolve 用 [ALE-Agent](https://arxiv.org/abs/2506.09050) 发现的 best solution 作为 initial program，跑 50 generations 试图改进。Fitness 用 public test set 的 score，最后在 private test set 报告。

LITE subset 包含 10 个 problem。

### 10.2 整体结果

平均改进 ~2.3%。其中 ahc039 任务从第 5 名升到第 2 名（如果当时提交到 AtCoder leaderboard）。

### 10.3 ahc039 详细分析

**Task**：找 axis-aligned polygon 最大化 (mackerels contained) - (sardines contained)，给定 100,000x100,000 grid 上的 2N 个 fish 位置（N mackerels + N sardines），polygon 顶点数 ≤ 1000，perimeter ≤ 400,000。

**Base solution (ALE-Agent)**：Simulated Annealing with kd-tree（5th, score 2880）

**ShinkaEvolve discovered improvements (2nd, score 3140)**：

看 Listing 4 完整代码。两个 key 改动：

**Improvement 1: kd-tree with cached subtree statistics**

```cpp
struct KDNode {
    Point pt;
    int axis;
    KDNode *left, *right;
    int fish_struct_idx;
    // Subtree bounding box
    int min_x, max_x, min_y, max_y;
    // Subtree counts
    int m_cnt = 0, s_cnt = 0;
};
```

每个 kd-tree node 不仅存自己的 point，还存子树的 bounding box 和 mackerel/sardine count。Query rectangle 时可以 whole-subtree prune + aggregate，避免遍历到叶子。

Intuition：**SA 的每次 move 都需要 query 一个 rectangle 内的 fish count，如果每次都 O(N) 遍历，SA 跑不了多少 iterations**。Cached subtree statistics 把 query 降到 O(√N + k)，k 是返回的 fish 数量。这是经典 [range tree](https://en.wikipedia.org/wiki/Range_tree) optimization。

**Improvement 2: Targeted edge move**

```cpp
// Targeted Edge Move: heuristic 找 misclassified fish, 移 nearest edge 修正
if (target_fish.type == 1) == is_inside:  // misclassified
    // 找最近的 edge
    for (size_t i = 0; i < poly.size(); ++i) {
        d_sq = point_segment_dist_sq_ortho(target_fish.p, poly[i], poly[(i+1) % poly.size()]);
        if (d_sq < min_dist_sq) { best_edge_idx = i; min_dist_sq = d_sq; }
    }
    // Move that edge to include/exclude the fish
```

Targeted move 是 **guided mutation**：找当前 polygon 误分类的 fish（mackerel 在外面 or sardine 在里面），greedily 把最近的 edge 推过去修正。这比纯随机 edge move 方向性强很多。

这两个改进都让 SA search 的方向性更强。Paper 提到 ShinkaEvolve 倾向于 stay close to ALE-Agent's solution，没有 radical architecture change。这暗示 LLM 在已经 high-performing 的 solution 上做 local refinement 比从头 reinvent 更有效。

### 10.4 ahc025 详细分析

**Task**：用 balance scale 比较 item subset weights，固定次数 weighings 后把 items 分成 D 组，最小化组间 weight variance。

**Improvements**：
1. Faster caching in QueryManager (用 `cmp1_flat` 数组存 1v1 comparisons)
2. Refined fallback weight estimation
3. **替换 simulated annealing 为 greedy + targeted local search**

这个改动很 significant——ShinkaEvolve 自己决定 SA 不够好，换成更 targeted 的 local search。

---

## 11. 实验 4：MoE Load Balancing Loss Discovery

这是 ShinkaEvolve 最有 scientific discovery 味道的实验。

### 11.1 MoE 背景

[Mixture-of-Experts (MoE)](https://arxiv.org/abs/1701.06538) 把 FFN 替换成多个 expert，router 选 top-K。Forward pass:

$$y_\ell(x) = \sum_{i=1}^{N_E} g_{\ell,i}(x) E_{\ell,i}(x), \quad g_{\ell,i}(x) = \begin{cases} \frac{e^{h_{\ell,i}(x)}}{\sum_{j \in \mathcal{T}_K(x)} e^{h_{\ell,j}(x)}} & \text{if } i \in \mathcal{T}_K(x) \\ 0 & \text{otherwise} \end{cases}$$

变量：
- $E_{\ell,i}$：layer $\ell$ 的 expert $i$（一个小 FFN）
- $h_{\ell,i}(x)$：router 对 token $x$ 在 expert $i$ 上的 logit
- $\mathcal{T}_K(x)$：top-K logit 的 index 集合
- $g_{\ell,i}(x)$：归一化后的 gating probability（只在 top-K 内归一化）

问题：top-K selection 是 non-differentiable 的，gradient 无法 backprop 到 router。需要 auxiliary load balancing loss (LBL) 防止 router collapse（所有 token 都 routed 到少数 expert）。

### 11.2 Standard Global-Batch LBL

[Shazeer et al. 2017](https://arxiv.org/abs/1701.06538) 提出，[Qwen3](https://arxiv.org/abs/2505.09388) 等开源 LLM 在用：

$$L_{LBL} = N_E \cdot \frac{1}{L} \sum_{\ell=1}^{L} \sum_{i=1}^{N_E} f_{\ell,i} \cdot P_{\ell,i}$$

变量：
- $N_E$：expert 数量
- $L$：layer 数量
- $f_{\ell,i} = \frac{\text{tokens routed to expert } i}{\text{total tokens in layer } \ell}$：selection frequency（实际路由比例）
- $P_{\ell,i} = \frac{\sum_x h_{\ell,i}(x)}{\sum_{x,j} h_{\ell,j}(x)}$：average router probability（soft routing distribution）

Intuition：$f \cdot P$ 鼓励 alignment——soft probability 高的 expert 也应该实际被选中得多。$N_E$ 系数让 minimum value (all uniform) 等于 1。

### 11.3 Global-Batch LBL 的 Blind Spot

考虑一个极端 case：64 个 expert 中，60 个 expert 完全没被用过，4 个 expert 平分所有 token。这种情况下：
- $f_{\ell,i} = 0$ for 60 dead experts
- $f_{\ell,i} = 0.25$ for 4 alive experts
- $P_{\ell,i}$ 类似

$f \cdot P$ 在 dead experts 上 = 0，在 alive experts 上 = $0.25 \times 0.25 = 0.0625$，sum = $4 \times 0.0625 = 0.25$，乘 $N_E = 64$ 得 16。这个值很大，loss 表面看起来"unbalanced"——但 60 个 dead experts 没有被任何 term 直接 penalize。

更微妙的 case：假设 64 个 expert 中 60 个 expert 各被 0.001 的 tokens 选中，4 个 expert 各被 0.24 选中。$f \cdot P$ sum 看起来"almost balanced"但实际 60 个 expert 接近 dead。

### 11.4 ShinkaEvolve Discovered LBL

ShinkaEvolve 在 global-batch LBL 基础上添加了新的 regularization term：

$$L_{LBL} = \underbrace{N_E \cdot \frac{1}{L} \sum_{\ell=1}^{L} \sum_{i=1}^{N_E} f_{\ell,i} P_{\ell,i}}_{\text{Global-batch LBL}} + \underbrace{\frac{0.1}{L} \sum_{\ell=1}^{L} s(P_\ell) \sum_{i=1}^{N_E} \max(0, \tau - f_{\ell,i})}_{\text{ShinkaEvolve new regularization}}$$

新 term 的变量：
- $\tau = 0.064 / N_E$：minimum usage threshold（paper 里 $N_E = 64$ 时 $\tau = 0.001$）
- $s(P_\ell) = 0.5 + (1 - \frac{H(P_\ell)}{\log N_E})$：normalized complement of routing entropy
  - $H(P_\ell) = -\sum_i P_{\ell,i} \log P_{\ell,i}$：layer $\ell$ 的 routing entropy
  - $\log N_E$：max entropy（uniform routing 的 entropy）
  - 当 $H = \log N_E$（uniform）：$s = 0.5 + 0 = 0.5$（weak push）
  - 当 $H \to 0$（concentrated）：$s \to 0.5 + 1 = 1.5$（strong push）

逐项 intuition：

**$\max(0, \tau - f_{\ell,i})$**：hinge loss。如果 expert $i$ 的 selection frequency $f_{\ell,i} < \tau$，penalize 缺口；如果 $f_{\ell,i} \geq \tau$，penalty = 0。这保证 well-used experts 不被 over-regularize。

**$\sum_i \max(0, \tau - f_{\ell,i})$**：layer 内所有 underused expert 的 cumulative gap。

**$s(P_\ell)$ multiplier**：当 router 已经 concentrated（low entropy，少数 expert 主导），push 更强；当 router 已经 uniform，push 弱。这是 **adaptive regularization**——只在真的有问题时介入。

**$\frac{0.1}{L}$**：跨 layer 平均 + 系数 0.1 控制总权重。

**Safety net intuition**：一旦 expert 跨过 floor $\tau$，penalty term 对那个 expert vanish，不再 over-regularize。这避免了 [Switch Transformer](https://arxiv.org/abs/2101.03961) 等模型中 LBL 过强导致 router 不敢 specialize 的问题。

### 11.5 实验验证

**Evolution setup**:
- Small MoE: 556M params, $N_E = 64$, $K = 8$（82M sparse activated）
- Train 2B fineweb tokens
- Fitness = $-(L_{CE} + L_{imb})$
- $L_{imb} = \frac{1}{2} \sum_i |f_{\ell,i} - 1/N_E|$
- 30 generations only（pretraining 太贵）

**Evaluation**:
- Scale 到 2.7B MoE（404M active）
- Train 30B fineweb tokens
- Compare across $\lambda \in \{0.001, 0.01, 0.1\}$
- 7 个 downstream benchmark：CommonSenseQA, HellaSwag, OpenBook QA, PIQA, SIQA, WinoGrande, ARC

**Results (Figure 8)**:
- Left: downstream task performance across 7 benchmarks，ShinkaEvolve LBL consistently 优于 global-batch LBL
- Middle: final perplexity as function of missroute fraction，ShinkaEvolve LBL 在不同 missroute fraction 下都 lower perplexity
- Right: gradient 可视化（2-expert simplified case）显示新 term 的 push 行为

特别值得注意的是：**improvement 随 $\lambda$ 增大而增大**。$\lambda$ 大时 LBL 权重大，standard global-batch LBL 的 blind spot 越严重，ShinkaEvolve 的新 term 越能发挥作用。

### 11.6 这意味着什么

这个实验是 LLM-driven scientific discovery 在 ML architecture design 上的 proof-of-concept。30 generations 的 evolution 就找到了 [Qwen3](https://arxiv.org/abs/2505.09388)、[DeepSeek-MoE](https://arxiv.org/abs/2401.06066) 等 SOTA model 用的 LBL 的一个改进版本。

有意思的是 ShinkaEvolve **没有完全 replace** global-batch LBL，而是 augment 它。这说明 evolution 也"知道"基础 LBL 是有用的，只是补它的 blind spot。这种"incremental but principled improvement"是 LLM-as-mutation-operator 模式的一个 emergent property。

跟 [AlphaEvolve](https://arxiv.org/abs/2506.13131) 发现新 matrix multiplication algorithm、[FlashAttention](https://arxiv.org/abs/2205.14135) 类工作相比，ShinkaEvolve 在 ML architecture design 方向展示了类似潜力。

---

## 12. Ablation Studies 系统分析

### 12.1 Parent Selection (Figure 9 left)

三种对比：
- **Best-of-N**：忽略 evolutionary history，每次用 initial program 作为 parent
- **Hill Climbing**：贪心选 best program 作为 parent
- **Weighted Sampling**：performance + novelty balance

结果：Weighted > Hill Climbing > Random。Hill climbing 初期强（exploit 当前 best）但 plateau 快。Weighted sampling 在整个 evolution 过程保持 steady improvement。Random 最差，证明 fitness-based selection 必要。

### 12.2 LLM Ensembling (Figure 9 middle)

- Single LLM (GPT-5-nano)
- Fixed LLM Ensemble (uniform sampling)
- Bandit-Based LLM Ensemble (UCB1)

Bandit 显著优于其他。Fixed ensemble 比 single LLM 有 moderate improvement，但 bandit 通过 dynamic prioritization 进一步拉开。

### 12.3 Rejection Sampling (Figure 9 right)

- No Rejection
- Embedding-Based Rejection (cosine sim > 0.95)
- + LLM-as-novelty-judge

Embedding-based 已经提供 substantial gains（节省大量 wasted evaluations）。LLM judge 仅 marginal improvement——说明 embedding similarity 已经是 novelty 的有效 proxy，不需要额外 LLM call。

---

## 13. Implementation Details 详解

### 13.1 Queue-based vs Asynchronous

Paper 提到 ShinkaEvolve 用 queue-based implementation：LLMs sequentially generate proposals，加到 evaluation queue，每个 proposal 基于所有 completed jobs。

Asynchronous 实验过：用 proposal queue + job queue 提高 throughput，但引入"off-archiveness"——new proposals 在前一批 evaluation 完成 之前 生成，不是基于最新 archive。这导致 sample efficiency 下降（更多 redundant proposals）。

Trade-off: **throughput vs sample efficiency**。ShinkaEvolve 选择后者，这跟它 sample-efficient 的定位一致。

### 13.2 配置示例

```python
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(eval_program_path="evaluate.py")
db_config = DatabaseConfig()
evo_config = EvolutionConfig(init_program_path="initial.py")

runner = EvolutionRunner(
    evo_config=evo_config,
    job_config=job_config,
    db_config=db_config,
)
runner.run()
```

### 13.3 Evaluate.py 接口

```python
def main(program_path, results_dir):
    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=3,  # Multi-evals to aggregate
        get_experiment_kwargs=get_kwargs,
        aggregate_metrics_fn=aggregate_fn,
        validate_fn=validate_fn,
    )
```

`num_runs=3` 对应 stochastic evaluation，aggregate 把多次 run 的结果 combine 成 single fitness。这跟 AIME 实验里每个 candidate 跑 3 次的设计一致。

### 13.4 Hyperparameter 表

Circle Packing 关键配置（Table 1）：
- Archive size = 40, elite ratio = 0.3
- 2 islands, migration interval = 10, migration rate = 0.0
- Parent selection = weighted, $\lambda = 10.0$
- Patch types = [diff, full, cross], probs = [0.45, 0.45, 0.1]
- 150 generations, 5 max parallel jobs
- Embedding model = text-embedding-3-small, similarity threshold = 0.95
- UCB1 LLM selection, exploration coefficient = 1.0
- LLMs: claude-sonnet-4, o4-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano（4 个 active）
- Temperatures = [0.0, 0.5, 1.0]
- Meta recommendation interval = 10, max 5 recommendations

注意 migration rate = 0.0 在 circle packing，意味着 islands 完全独立。在 AIME 和 ALE-Bench 用 0.1。这个差异可能是 domain-specific tuning。

---

## 14. 与 Related Work 的深度对比

### 14.1 vs AlphaEvolve

[AlphaEvolve (Novikov et al., 2025)](https://arxiv.org/abs/2506.13131)：
- Closed source
- 用 thousands of evaluations
- 也是 evolutionary + LLM
- Google DeepMind 内部有大量 computing resources

ShinkaEvolve 用 150 samples 超过 AlphaEvolve 的 circle packing 结果，关键差异：
1. UCB1 bandit for LLM selection（AlphaEvolve 应该有但未公开）
2. Code novelty rejection sampling（明确 paper 强调这是新）
3. Weighted parent sampling with novelty term
4. Meta-scratchpad online refinement

### 14.2 vs OpenEvolve

[OpenEvolve (Sharma, 2025)](https://github.com/codelion/openevolve)：
- Open-source baseline
- ShinkaEvolve 借鉴了它的 API 接口
- 没有 bandit-based LLM selection
- 没有 embedding-based rejection sampling
- Sample efficiency 比 ShinkaEvolve 差

ShinkaEvolve 借用了 OpenEvolve 的 circle packing verification script，确保 fair comparison。

### 14.3 vs AI Scientist v2

[AI Scientist v2 (Yamada et al., 2025)](https://arxiv.org/abs/2504.08066) 是 Sakana AI 自己之前的工作，agentic tree search for scientific discovery。ShinkaEvolve 可以看作 AI Scientist v2 在 code evolution 方向的 refinement——更聚焦于 program optimization 而非 end-to-end scientific pipeline。

### 14.4 vs Eureka

[Eureka (Ma et al., 2023)](https://arxiv.org/abs/2310.12931) 用 LLM 设计 RL reward function，evolutionary loop。ShinkaEvolve 是更 general 的 framework——reward design 是其中一个 possible task。但 Eureka 更聚焦 RL reward，ShinkaEvolve 是 general program evolution。

### 14.5 vs AI CUDA Engineer

[AI CUDA Engineer (Lange et al., 2025)](https://sakana.ai/ai-cuda-engineer) 是 Sakana AI 之前的工作，evolutionary CUDA kernel optimization。ShinkaEvolve 在 methodology 上是它的直接进化版——增加了 bandit LLM selection、novelty rejection、meta-scratchpad 等创新。

### 14.6 vs Darwin Gödel Machine

[Darwin Gödel Machine (Zhang et al., 2025)](https://arxiv.org/abs/2505.22954) 做 self-improving agent 的 open-ended evolution。ShinkaEvolve 借鉴了它的 weighted parent sampling 思路（performance + novelty），但 ShinkaEvolve 更聚焦 fixed task 的 sample-efficient optimization 而非 open-ended self-improvement。

---

## 15. Limitations 和 Future Work

### 15.1 Limitations

1. **Manual task specification**：objective function 和 evaluation 还需要 human expertise
2. **Numerical objectives only**：framework 局限于 well-defined numerical fitness 的 task
3. **API cost barrier**：大规模 LLM usage 仍有经济门槛
4. **Fixed exploration-exploitation balance**：当前需要 hyperparameter tuning，没 auto-adapt
5. **ALE-Bench 上倾向 local refinement**：可能是 overfitting to initialization

### 15.2 Future Directions

Paper 提到的几个方向：

1. **Automated task specification via LLM**：让 LLM 自己 generate objectives，解锁更多 domain
2. **True open-endedness**：system 自己 generate objectives（如 [OMNI-EPIC](https://arxiv.org/abs/2405.15568)）
3. **Self-referential refinement**：framework evolve 自己的 evolution strategy（meta-evolution）
4. **Online meta-learning**：持续 improve discovery process

---

## 16. 我的几个观察

### 16.1 "150 samples" 的真正含义

150 generations 不是 150 LLM calls——每个 generation 可能涉及多次 LLM call（mutation + novelty check + meta-scratchpad + 多次 resample）。但相对于 AlphaEvolve 的 thousands of evaluations，sample efficiency 提升是真实的，因为 evaluation（跑 program）才是 expensive part。

### 16.2 Evolution vs Pure LLM

为什么不用 LLM 直接生成 solution？因为：
- LLM 单次 generation 容易 stuck 在 local optimum
- Evolution 提供 **iterative refinement with feedback**——每个 mutation 基于前一代的 fitness 和 textual feedback
- Archive 提供 **memory**——LLM 不需要每次从头生成
- Diversity mechanism（island, novelty rejection）保持 **exploration**

这跟 [AlphaCode](https://www.science.org/doi/10.1126/science.abj9343) 等 LLM-for-code 工作的区别：AlphaCode 用 massive sampling + filter，ShinkaEvolve 用 guided evolutionary search。

### 16.3 Sample efficiency 的来源

三个创新各自贡献：
- **Weighted parent sampling**：避免对 top program 过度 exploit，平衡 exploration
- **Novelty rejection sampling**：避免重复 evaluate 几乎相同的 program
- **UCB1 bandit**：让 best LLM for 当前 task phase 更频繁被用

Ablation 显示三者协同——单独任何一个都不够。

### 16.4 与 Test-Time Compute Scaling 的关系

ShinkaEvolve 本质是 **structured test-time compute scaling**：用更多 LLM calls 换更好 solution。但跟 [OpenAI o1 style CoT](https://arxiv.org/abs/2408.03314) 的区别：ShinkaEvolve 不是在 inference time think more，而是在 search space 上 explore more。

这跟 [Self-Refine](https://arxiv.org/abs/2303.17651)、[Tree of Thoughts](https://arxiv.org/abs/2305.10601) 等 test-time compute 方法在精神上类似，但作用在 **program space** 而非 **reasoning chain space**。

### 16.5 Open-Source 的价值

AlphaEvolve 是 closed，OpenEvolve 是 community 复现但 sample efficiency 不够。ShinkaEvolve 在 Apache 2.0 下 release 完整 implementation，包括 interactive visualization tool。这对 democratize 这种 advanced framework 很重要——以前只有 DeepMind 这种大 lab 能玩，现在任何研究者都能 fork 改进。

GitHub: [https://github.com/SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve)

---

## 17. 关联联想和 Open Questions

### 17.1 跟 Quality-Diversity (QD) 的关系

ShinkaEvolve 的 novelty rejection sampling 跟 [MAP-Elites](https://arxiv.org/abs/1504.04909) 等 QD 算法精神类似——保留 behaviorally diverse solutions。但 ShinkaEvolve 用 code embedding 当 behavior characterization，而不是 hand-designed feature。这是 LLM 时代 QD 的新形态。

### 17.2 跟 Neuroevolution 的关系

Classic neuroevolution（[NEAT](https://arxiv.org/abs/1909.13184)）evolve neural network weights 和 topology。ShinkaEvolve evolve 的 program 是更广义的——可以是 algorithm、agent scaffold、loss function。这是 **neuroevolution 的 superset**。

### 17.3 跟 AutoML 的关系

[AutoML](https://www.automl.org/) 用 search 找 ML architecture。ShinkaEvolve 在 MoE LBL 实验里展示的是 AutoML 的 LLM-driven 版本——LLM 当 mutation operator 替代 random search 或 Bayesian optimization。

### 17.4 Open Question: Self-Referential Evolution

如果让 ShinkaEvolve evolve 自己的 evolution strategy（parent sampling strategy, novelty threshold, bandit coefficient），会怎么样？这指向 [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) 的 self-improvement 模式，但 applied to meta-evolution。

### 17.5 Open Question: Multi-Modal Programs

当前 programs 都是 code。如果 program 是 multimodal（code + 自然语言 rationale + 视觉 sketch），evolution 怎么处理？这需要 multi-modal LLM 和 multi-modal embedding。

### 17.6 Open Question: Continual Learning

ShinkaEvolve 每个 task 从头跑。如果 archive 能跨 task transfer（类似 [meta-learning](https://arxiv.org/abs/1904.10690)），新 task 可以 reuse 之前学到的 strategies。这是 future work 里 "self-referential refinement" 的方向。

### 17.7 跟 AlphaEvolve 后续的可能性

DeepMind 的 AlphaEvolve 发现了新数学结果（[Nature paper Romera-Paredes et al. 2024](https://www.nature.com/articles/s41586-023-06923-3)），ShinkaEvolve 的 sample efficiency 改进如果能让社区复现这种 discovery level 的 search，可能解锁大量自动数学发现。MoE LBL 实验已经展示了这种潜力。

### 17.8 跟 LLM Inference Time Scaling Law

最近 [OpenAI o1](https://openai.com/o1/)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 推动 inference time scaling。ShinkaEvolve 是另一种 inference time scaling：用更多 LLM calls 做 evolutionary search。两者可以结合——reasoning model 当 mutation operator 在 evolution loop 内 think more。Paper 提到用 o4-mini 在 LLM ensemble 里，但没深入。

---

## 18. 总结

ShinkaEvolve 在我看来是 **LLM-driven evolutionary discovery 这条 line 上的一个重要 milestone**：

1. **Sample efficiency**：150 samples vs thousands，量级提升
2. **Open-source**：打破 AlphaEvolve 的 closed-source 壁垒
3. **Three synergistic innovations**：weighted parent sampling + novelty rejection + UCB1 bandit
4. **Four diverse tasks**：从 mathematical optimization 到 agent design 到 ML architecture discovery
5. **MoE LBL discovery**：证明 LLM 可以做 ML architecture research

这工作的真正价值不在某个具体 trick，而在 **整个 framework 的 sample efficiency 让 LLM-driven discovery 从 "big lab only" 变成 "any researcher can play"**。后续 community 在这基础上改进、扩展、应用到新 domain 的潜力很大。

GitHub repo: [https://github.com/SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve)

Sakana AI 主页: [https://sakana.ai/](https://sakana.ai/)

AlphaEvolve paper: [https://arxiv.org/abs/2506.13131](https://arxiv.org/abs/2506.13131)

OpenEvolve: [https://github.com/codelion/openevolve](https://github.com/codelion/openevolve)

ALE-Bench: [https://arxiv.org/abs/2506.09050](https://arxiv.org/abs/2506.09050)

AI Scientist v2: [https://arxiv.org/abs/2504.08066](https://arxiv.org/abs/2504.08066)

Darwin Gödel Machine: [https://arxiv.org/abs/2505.22954](https://arxiv.org/abs/2505.22954)

Eureka: [https://arxiv.org/abs/2310.12931](https://arxiv.org/abs/2310.12931)

LLM4AD: [https://arxiv.org/abs/2412.17287](https://arxiv.org/abs/2412.17287)

UCB1 paper (Auer et al. 2002): [https://link.springer.com/article/10.1023/A:1013689704352](https://link.springer.com/article/10.1023/A:1013689704352)

Shazeer MoE LBL: [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)

Switch Transformer: [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)

Qwen3: [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

DeepSeek-MoE: [https://arxiv.org/abs/2401.06066](https://arxiv.org/abs/2401.06066)

Reflexion: [https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)

Self-Consistency: [https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)

Tree of Thoughts: [https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)

Self-Refine: [https://arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651)

MAP-Elites: [https://arxiv.org/abs/1504.04909](https://arxiv.org/abs/1504.04909)

NEAT: [https://arxiv.org/abs/1909.13184](https://arxiv.org/abs/1909.13184)

AlphaCode: [https://www.science.org/doi/10.1126/science.abj9343](https://www.science.org/doi/10.1126/science.abj9343)

OpenAI o1: [https://openai.com/o1/](https://openai.com/o1/)

DeepSeek-R1: [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)

OMNI-EPIC: [https://arxiv.org/abs/2405.15568](https://arxiv.org/abs/2405.15568)

FlashAttention: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
