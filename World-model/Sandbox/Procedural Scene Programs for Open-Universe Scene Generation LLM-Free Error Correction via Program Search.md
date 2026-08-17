---
source_pdf: Procedural Scene Programs for Open-Universe Scene Generation LLM-Free
  Error Correction via Program Search.pdf
paper_sha256: 6c61c3a3f395802aa915472d82c42497a485c5c470243f600e2ff7e9e1072a8b
processed_at: '2026-08-06T06:37:44-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲这篇 paper

Andrej，好，我换个更 colloquial 的方式讲，但 technical 细节不丢。

---

## 一句话说清楚这 paper 在干啥

你给一句 text prompt，比如 "一个大学图书馆，有几排书架，中间一张大桌子，周围一圈椅子"，系统就给你生成一个 3D scene 的 layout——每个 object 的 position、orientation 都摆好。这个 task 叫 scene layout generation。

之前大家都在用 declarative 方法：让 LLM 写一堆 constraint（"chair adjacent to table", "book on shelf"），然后丢给一个 solver 去算具体坐标。这 paper 说：我们用 imperative 方法，让 LLM 直接写 Python 程序来描述每个 object 该放哪——但是用一种特殊设计的 DSL，使得 program 写错了可以自动修。

---

## 背景为什么 complicated

### 1.0 时代的 imperative：LayoutGPT

最早 LayoutGPT 就是让 LLM 直接输出每个 object 的数字坐标。比如：

```python
chair.center.x = 3.2
chair.center.y = 1.5
chair.center.z = 0.0
chair.facing = X_POS
```

LLM 写这种 raw numbers 特别烂——会输出 chair 跟 table 重叠、chair 悬空、chair 跑到房间外面。因为 LLM 对精确数值的 reasoning 很弱，这是你早就反复强调过的点：LLM 不擅长 output raw numbers，擅长 output symbols。

### 2.0 时代的 declarative：Holodeck, DeclBase, FlairGPT

后来大家发现，把 LLM 的任务简化就行：不让它写坐标，让它写关系。

```python
on(book, table)
adjacent(chair1, table)
adjacent(chair2, table)
facing(chair1, table)
```

然后一个 solver 把这些 constraint 转成实际坐标。Solver 本质上是个 optimization：找一组坐标使所有 constraint 的 violation 之和最小。

这招 working 的原因有两个：
1. LLM 不用碰数字了，只预测 high-level relations，它擅长这个
2. Solver 自带 error correction——就算 LLM 写错了一两个 relation，solver 会找到一个 "尽量满足所有 constraint" 的 layout

但 declarative 有两个 problem：
- **慢**：solver 是个全局优化，object 多了 relation 就 O(n²)，40 个 object 的 scene solver 要跑 21 秒
- **DSL 表达能力有限**：如果 DSL 没有 `spiral()` 这个 operator，你就摆不出螺旋形的 layout。DSL 是 hand-designed 的，永远覆盖不全所有可能的 spatial pattern

### 这 paper 的 question

> declarative 的两个 advantage——(1) high-level operations 让 LLM 轻松，(2) solver 自带 error correction——能不能在 imperative 框架里也实现？

如果能，imperative 本身的好处就全回来了：execution 快、理论上能表达任意 configuration。

---

## PSDL：核心 contribution

PSDL = Procedural Scene Description Language。它是个 Python-embedded DSL，意思是 LLM 生成的就是 Python 代码，但里面有几个 domain-specific 的重载 operator。

### 三个关键 design

**1. Object 之间用 parametric relationship 表达**

不要写 `chair.center.x = 3.2`，而是写：

```python
chair.max.x = table.min.x - 0.1
chair.center.y = table.center.y
chair.min.z = scene.min.z
chair.facing = table.facing
```

注意这里赋值号 `=` 是被重载的——左边不是 "修改 chair.max 这个 vector"，而是 "translate chair 使得新的 chair.max.x 等于右边那个值"。所以这是把 declarative 的 constraint 思想塞进了 imperative 的语法里。

好处：任何 parameter perturbation 都会自动 propagate。你把 table 从 0.1 改成 0.15，chair 自动跟着动。关系 *by construction* 保留。

**2. 用变量共享表达式**

```python
d = 2.0
for i, c in enumerate(cols):
    c.center.x = scene.center.x + i * d
```

一个 `d` 控制一整列 object 的间距。Local search 时只需要改一个数字就能移动整列。

**3. Control flow**

`for`, `if` 这些 Python 原生的东西。Loop 隐式共享 stride 常量——一行代码控制 10 把椅子，而不是 10 行重复代码。

### 为什么这三个 design 对 error correction 致关重要

想象一下 LayoutGPT 风格的 flat imperative：fix 一个 overlap 通常意味着把某个 object 单独挪走。比如 chair 和 table 重叠了，修法就是把 chair 挪到房间另一头——overlap 没了，但 chair 和 table 的关系也没了。

PSDL 不一样：fix 一个 overlap 通常是 `table.min.x - 0.1` 改成 `table.min.x - 0.15`。Chair 还是紧贴 table，只是远了一点点。Semantic structure 保留。

这就是为什么后面实验里 gradient descent 能把 error count 降到 0.5（最低）但 preference rate 反而差（61.4%），因为它是 per-object 调整，破坏关系。

---

## Error Correction：不调 LLM 的修复机制

LLM 写完 PSDL program 后，把它 execute 一下，看 loss 是多少。Loss 有四项：

1. **Out-of-Bounds Loss**：每个 object 突出 scene 边界的最大距离
2. **Overlap Loss**：每对 object 的 AABB 交集体积的 cube root
3. **Standing Loss**：STANDING object 底面到最近可支撑面的距离
4. **Mounted Loss**：MOUNTED object 的 mountable face 到最近可支撑墙面的距离

如果 loss 不为零，就要修 program。但修 program 这事不能调 LLM——慢、贵、而且 LLM 自己修经常修不好（Olausson et al. 2024 在 code generation 上验证过 self-repair 通常没用）。

### 目标函数

$$
\arg\min_{P} \big[\text{loss}(L) + d(P, P_0)\big]
$$

变量解释：
- $P_0$: LLM 原始 program
- $P$: 修改后的 candidate program
- $L, L_0$: $P$ 和 $P_0$ 执行后的 layout
- $d(P, P_0)$: program 之间的距离

距离进一步分解：

$$
d(P, P_0) = d_{\text{edit}}(P, P_0) + d_{\text{OT}}(L, L_0)
$$

- $d_{\text{edit}}$: 程序文本的 edit distance。Elementary edits = 改一个 numerical constant 或改一个 facing direction。这是 symbolic similarity。
- $d_{\text{OT}}$: layout 的 optimal transport 距离，geometric similarity。

### Optimal Transport 距离这一项

$$
d_{\text{OT}}(L, L_0) = \min_{f \in \mathcal{F}} \sum_{o \in \mathcal{O}(L)} \text{Vol}(o) \cdot \|\text{center}(o) - \text{center}(f(o))\|_2
$$

变量逐个解释：
- $\mathcal{O}(L)$: layout $L$ 里的 object 集合
- $\mathcal{F}$: 所有保持 object category 的 bijections $f: \mathcal{O}(L) \to \mathcal{O}(L_0)$。Chair 必须映射到 chair，table 映射到 table，避免 degenerate matching
- $\text{Vol}(o)$: object 体积，作为 mass
- $\text{center}(o)$: object 几何中心
- $\|\cdot\|_2$: L2 距离
- 整个公式：找一个 category-preserving 的 object 对应关系，使所有 object 移动距离 × 体积之和最小

这个设计的妙处：
- **体积加权**：大 object（沙发、bed）移动代价高，小 object（书、花瓶）移动代价低。鼓励保留 major elements 的位置，只调整 minor elements
- **Category-preserving**：避免 "把 chair 移到 table 位置，假装 layout 没变" 这种 cheating solution
- 本质是 Wasserstein-1 距离的变种，mass = volume

### 算法：Local search

精确最小化 intractable，用 iterative local search：

$$
P_{i+1} = \arg\min_{P \in \mathcal{N}(P_i)} f(L), \quad f(L) = \log s(L) + d_{\text{OT}}(L, L_0)
$$

变量解释：
- $P_i$: 第 $i$ 次迭代后的 program
- $\mathcal{N}(P_i)$: $P_i$ 的 neighborhood，通过对 $P_i$ 做 elementary edits 得到
- $s(L)$: layout $L$ 的 loss 总和
- $\log s(L)$: 取 log 让 loss 项和 OT 项的量级更平衡

Neighborhood 构造：
- 每个 constant expression 采样 10 个 edits，把常量 $c$ 替换为 $c \cdot \pm 4^\epsilon$
  - $\epsilon \sim \mathcal{U}[-1, 1]$（uniform 分布）
  - sign 均匀取 $\{-1, +1\}$
  - 这意味着 multiplier 在 $[0.25, 4]$ 之间，log-uniform 分布
  - 既包含微调（×0.97）也包含大调（×3.8）
- 每个 direction expression 采样 4 个 edits，即穷举所有 4 个 cardinal directions

终止条件：没有可用 edit 使 $f$ 下降超过某个 small threshold。

### 关键直觉

Local search *不* 优化 object positions，而是优化 *program expressions*。在 PSDL 里因为 variable sharing 和 loops，一个常量可能控制十几个 objects。所以 "改一个数字" 在 PSDL 里等价于 "在 basic imperative 里同时改十几个 objects 的 coordinates 但保持它们之间的关系"。

这就是为什么平均一个 scene 只需要 **7.13 次 adjustments** 就能 resolve 所有 errors——LLM 离 valid program 已经很近，只是数值有点 off。这跟你在 Codex/HumanEval 上观察到的 "LLM 算法正确但 off-by-one" 是同一个现象。

---

## 实验：怎么比较才公平

### Pipeline 隔离

为了只比较 layout generation 这一步，所有方法共享：
- Template generation：LLM 生成 scene dimensions + object list
- Object retrieval：CLIP 从 HypeHype Asset Library（~6000 assets）检索 mesh

只有 layout generation 阶段不同：DeclBase / Holodeck / FlairGPT 用 declarative，Ours 用 PSDL + error correction。

### Benchmark

70 个 prompts，4 个 axes：
- size: small (14) / medium (35) / large (21)
- location: indoor (48) / outdoor (22)
- realism: realistic (50) / fantastical (20)
- structure: chaotic (31) / structured (39)

加上 Holodeck 用过的 52 个 MIT indoor scene prompts + 14 个 Holodeck long prompts = 136 个总共。19 个被标为 'complex'（4+ words）。

---

## 实验结果

### Human Perceptual Study

10 个 Brown 学生，2AFC forced choice，每人 70 个 comparison。Object 渲染成彩色 box，hover 显示 name（剔除 mesh 选择的影响）。Majority vote。

| Comparison | Preference for Ours |
|---|---|
| Ours w/ EC vs. DeclBase | **82.9%** |
| Ours w/ EC vs. Holodeck | **94.3%** |
| Ours w/ EC vs. Ours w/o EC | **74.3%** |

vs Holodeck 有 51/70 task 是 5/5 unanimous for ours。这是非常强的信号。

按 category 看，在 large structured scenes 上优势最大（90.5% vs DeclBase, 95.2% vs Holodeck）。在 fantastical 上 vs Holodeck 是 100%——说明 declarative DSL 在非典型场景下完全失效。

### Automated Evaluation: LLMCompare

先试了现有 text-to-image 的 metric，发现都不行：
- VQAScore [Lin et al. 2024]: 58.6% agreement with human——只比 chance (50%) 好一点。VQA 能识别 "这是 bedroom" 即使 layout 烂，区分度不够
- DSG [Cho et al. 2024]: 50.7%——basically chance。DSG 生成的 yes/no 问题区分不出两个 layout 的差异

提出的 LLMCompare：让 GPT-4o 同时看两张 layout 图 + prompt，先列每个 layout 的 pros/cons，再输出哪个更好。

- LLMCompare: **77.1%** agreement with human
- LLMCompare without pros/cons step (ablation): 70.0%——先列 pros/cons 这步确实有用，符合 chain-of-thought 直觉

按 object 数量拆分 vs Holodeck：
- 10-20 objects: 66.7%
- 20-30: 80.0%
- 30-40: 72.9%
- **40+: 95.2%**

Object 数量越多，imperative + low-dim parameter search 的优势越大。Declarative solver 在 40+ objects 时 quadratic in relations 已经不堪重负。

### Error Correction Ablation

| Error Correction | Pref rate | # Errors | Run time |
|---|---|---|---|
| No solver | 60.0% | 12.3 | 0s |
| Gradient descent | 61.4% | **0.5** | 2.6s |
| LLM self-repair | 68.1% | 5.7 | 106.7s |
| Local search, basic imperative | 71.4% | 0.8 | 25.2s |
| **Local search, PSDL (ours)** | **~80%+** | 1.1 | **9.3s** |

关键 takeaways：

1. **Gradient descent 消除 errors 最多 (0.5) 但 preference rate 最低 (61.4%)**：因为它 per-object 移动，破坏关系。这是论文核心 claim 的实证

2. **LLM self-repair 慢一个数量级 (106.7s vs 9.3s) 且 errors 残留多 (5.7)**：LLM 不擅长 numerical refinement。这跟你和 Olausson et al. 2024 都强调的 "self-repair usually doesn't help much in code generation" 一致。Self-repair 用 LLM 自己反思，但 LLM 对数字不敏感，反思也修不对

3. **PSDL local search 比 basic imperative local search 快 3 倍** (9.3s vs 25.2s)：因为 PSDL 的 loops + variables 降低了 search space 维度。同样的算法，search space 维度不同，速度差 3 倍

### Timing

- Template generation: 9.5s
- PSDL program generation (one LLM call): 19.2s
- Error correction (local search, no LLM): 9.3s
- **Total: ~38s**
- DeclBase: 10s program + 21.3s solver = 40.8s

整体 comparable，但 PSDL error correction 比 declarative solver 快 2 倍多，且差距随 object 数量增长会扩大。

### LLM Backbone Robustness

测试了 Claude-3.5-Sonnet、GPT-4o、o1、Gemini-Exp。Pre-correction errors 都在 14-18 之间，post-correction 都降到 2-3.3。Error correction 对所有 LLM 都 working，是 LLM-agnostic 的 framework。

---

## 跟你 intuition 的连接

### LLM as programmer, not predictor

你一直在 push "LLM 输出 code 而不是 numbers"。这篇是又一个 case study：LLM 直接输出 chair coordinates 时烂，输出 `chair.max.x = table.min.x - 0.1` 时质量大幅提升。

### Verifier in the loop, not LLM in the loop

Error correction 不调 LLM，只用 local search + loss function。这跟你最近讨论 reasoning model / AlphaProof 时强调的 "verifier 比 self-reflection 可靠" 完全一致。Table 6 里 LLM self-repair (106.7s, 5.7 errors) 被无声碾压。

**Constraint satisfaction 是 verifier 的工作，不是 generator 的工作**。

### Compression = Generalization

PSDL 的 variable sharing 和 loops 把 layout 压缩成更短的程序。Search space 从 "每个 object 6 DoF" 压缩到 "十几个关键常量"。这是巨大的维度缩减，跟你在 "A Path Towards Autonomous Machine Intelligence" 里讨论的 world model predictive coding 思路精神上呼应。

### AlphaProof-style search at symbolic level

AlphaProof 的精髓是 *在 Lean program 空间里 MCTS*，不是在 raw proof space。这 paper 是同样的转移：*在 PSDL program 空间里 local search*，不是在 raw layout space。搜索对象是符号结构，每一步 search 都 semantically meaningful——改一个 spacing 常量有明确语义，改一个 chair 的 x 坐标没有。

参考 AlphaProof: https://deepmind.google/discover/blog/alphaaward-awarded-the-rutherford-medal/

### 与 Scene Language 的关系

Scene Language [Zhang et al. 2024] 是 concurrent work，也用 imperative approach 生成 scene programs with control flow，但*没有* parametric relationships、local coordinate frames、或 error correction。它把 program 转化为 detailed scene 是通过 neural rendering。

PSDL 在 layout correctness 这个维度上显著超越 Scene Language——但 Scene Language 在 visual fidelity 上可能更强（用 neural rendering）。两者的结合是 obvious future direction：PSDL 生成 valid skeleton，neural renderer 在 skeleton 上做 detail synthesis。

参考 Scene Language: https://ai.stanford.edu/~yzzhang/projects/scene_language/

### 与 DreamCoder 的连接

PSDL 的 design pattern 思想跟 DreamCoder [Ellis et al. 2023] 的 "learn reusable primitives" 思路相似。区别是 PSDL 的 primitives 是 hand-designed（来自 domain knowledge），DreamCoder 是 learned。

一个 natural follow-up：能不能让 LLM 在生成 PSDL 程序的过程中 learn new primitives？比如 "def dining_set(table, chairs, spacing): ..." 作为 callable，下次 reuse。

参考 DreamCoder: https://arxiv.org/abs/2206.08322

---

## Limitations 和我想到的 Future Work

### Paper 自承的
- Local search 只 edit 数值常量和 facing directions，不能 swap object identities 或 swap x/y 坐标。LLM 如果 misinterpret prompt（把 chair 应该在 table 左边写成右边），correction 救不了
- Loss 没有高阶美学：line-of-sight, walkability, affordances beyond support/collision
- VLM evaluator 本身是 proxy，可能有 bias
- 固定 object retrieval + 4 cardinal orientations——relaxing 这些会暴露新问题

### 我联想到的
1. **Higher-order edits**: 加 "swap two object identities"、"negate a facing"、"swap x/y of an expression" 会让 search space 更 powerful
2. **Learned edit proposals**: 用小 LM 学习 *where to edit*，类似 AlphaGo 的 policy network 减少 MCTS branching
3. **Hierarchical programs**: 支持 function definitions，LLM 可以 reuse 之前生成过的 sub-programs
4. **Differentiable PSDL**: 如果 PSDL 子集是 differentiable（所有 assignments 是 affine combinations），可以用 gradient-based optimization 替代 random local search。这跟 JAX / Dex 思路一致
5. **Dynamic scenes**: PSDL 扩展到 temporal programs 可以生成 animations
6. **Affordance loss**: 把 walkability、line-of-sight、reachability 作为 additional loss terms
7. **Active prompt refinement**: Error correction 失败时让 LLM *clarify* prompt（"did you mean round tables or rectangular tables?"）
8. **PSDL as scene representation for RL**: 生成的 PSDL 程序作为 embodied AI 的 environment spec，比从 mesh 数据集采样更 flexible

---

## 一句话总结

> **Imperative programs with parametric relationships are a compressed representation of layouts. Compression reduces search space dimensionality. Reduced search space makes symbolic local search feasible. Symbolic local search preserves semantic structure by construction.**

四步因果链，每一步 obvious in hindsight，组合起来是 paradigm shift。

从你 "compression is intelligence" 的视角看：*let LLM compress scene into a program; let search decompress the program into a valid layout*。

这种 "LLM compresses, search decompresses" 的分工，比 "LLM compresses, LLM self-repairs" 或 "LLM emits constraints, solver decompresses" 都更 efficient——前者因为 LLM 不擅长 numerical refinement，后者因为 solver 在高维 continuous space 里慢且 DSL-restricted。PSDL 的 middle ground 拿到了两边的优点。

---

### Further reading

- Paper 本身 (SIGGRAPH Asia 2025): https://doi.org/10.1145/3757377.3763930
- Brown Scene Synthesis group: https://scenegen-group.github.io/
- Daniel Ritchie: https://dritchie.github.io/
- DeclBase: https://arxiv.org/abs/2403.09675
- Holodeck: https://arxiv.org/abs/2312.09067
- LayoutGPT: https://openreview.net/forum?id=Xu8aG5Q8M3
- FlairGPT: https://doi.org/10.1111/cgf.70036
- Scene Language: https://arxiv.org/abs/2410.16770
- DreamCoder: https://arxiv.org/abs/2206.08322
- Optimal Transport (Peyré & Cuturi): https://arxiv.org/abs/1803.00567
- VQAScore: https://arxiv.org/abs/2404.01291
- DSG: https://arxiv.org/abs/2405.03253
- Self-Repair Silver Bullet?: https://arxiv.org/abs/2308.15975
- ATISS (same group): https://arxiv.org/abs/2110.04275
- PlanIT (same group): https://cs.stanford.edu/~kaikai/planit/
- DiffuScene (same group): https://arxiv.org/abs/2305.02755
- I-Design: https://arxiv.org/abs/2404.02838
- Objaverse-XL: https://arxiv.org/abs/2307.05663

要不要展开讲 optimal transport 那一项的更多数学，或者 PSDL API 的完整细节，或者跟 AlphaProof 更深的类比，告诉我。

---

# Procedural Scene Programs for Open-Universe Scene Generation — 深度解读

Andrej，这篇 SIGGRAPH Asia 2025 的论文来自 Brown University 的 Daniel Ritchie 团队（也是 ATISS、PlanIT、DiffuScene 这一脉的 scene synthesis 老手）。核心 thesis 是：**在 LLM-based scene layout generation 这件事上，imperative paradigm 不该被 declarative paradigm 淘汰——只要你能给它配上一个 symbolic error-correction module，imperative 反而更优**。这与你一直在强调的 "LLM 写代码 + symbolic verifier/interpreter 反馈" 的 neurosymbolic 套路高度共鸣。

---

## 1. 为什么这篇论文值得你看

你过去几年反复 push 的一个观点是：LLM 真正的力量在于 *writing code that gets executed*，而不是直接输出 raw tensors 或 raw numerical answers。这篇论文正是把这个思想应用到一个之前被 declarative 方法"统治"的领域。它把 LLM-generated scene layout 看成一段 Python 程序，然后在一个 *very small, semantically meaningful search space* 上做 local search 来 fix 物理错误——完全不调用 LLM。

这种 "LLM 生成 program skeleton + symbolic search refine constants" 的范式，和你最近在讨论的 reasoning model / AlphaProof-style 的 "LLM proposes, verifier disposes" 在精神上是一致的，只不过这里的 verifier 是 loss function + local search 而不是 Lean。

参考链接：
- Brown Scene Synthesis group: https://scenegen-group.github.io/
- Daniel Ritchie's page: https://dritchie.github.io/
- 之前同组 DeclBase (Aguina-Kang et al. 2024): https://arxiv.org/abs/2403.09675
- Holodeck (Yang et al. 2023): https://arxiv.org/abs/2312.09067
- LayoutGPT (Feng et al. 2023): https://openreview.net/forum?id=Xu8aG5Q8M3
- FlairGPT (Littlefair et al. 2025): https://doi.org/10.1111/cgf.70036
- Scene Language (Zhang et al. 2024, concurrent imperative work): https://arxiv.org/abs/2410.16770
- Optimal Transport reference (Peyré & Cuturi): https://arxiv.org/abs/1803.00567
- VQAScore: https://arxiv.org/abs/2404.01291
- DSG (Davidsonian Scene Graphs): https://arxiv.org/abs/2405.03253

---

## 2. 背景与动机：imperative vs declarative 之争

### 2.1 两种 paradigm 的本质差异

**Imperative（指令式）**: LLM 直接写出每个 object 的 position、orientation、size。代表：LayoutGPT。
- 优点：execution 极快（直接就是 layout）；理论上能表达任意 configuration。
- 缺点：LLM 对 raw numerical coordinates 的预测极差，经常产生 overlap、floating、out-of-bounds。LayoutGPT 的输出经常物理上不合法。

**Declarative（声明式）**: LLM 只写约束（`on(a,b)`, `adjacent(a,b)`, `aligned(a,b,c)`），然后一个 solver 把约束转化为 layout。代表：Holodeck、DeclBase、FlairGPT、I-Design。
- 优点：LLM 任务简化（predict relations instead of numbers）；solver 自带 error correction（minimize residual constraint violation）。
- 缺点：
  - **慢**：solver 时间随 relation 数量线性增长，而 relation 数通常 O(n²)（n = object 数）。一个大 museum 可能要等几十秒。
  - **DSL 表达能力受限**：如果 DSL 没有 `spiral()`，就摆不出螺旋形。

### 2.2 论文的关键观察

> "Are the above differentiators unique to the declarative setting, or could they be realized in an imperative approach as well?"

Declarative 之所以强，是因为它有 (1) higher-level operations（让 LLM 不必预测数字）和 (2) built-in error correction（solver）。这两个优势其实可以移植到 imperative：
- (1) → 用 *relational* imperative DSL，让 position 是 `table.min.x - 0.1` 而不是裸数字。
- (2) → 加一个 *program-space* error correction module。

这是整篇论文的设计动机。从 your "Software 2.0 / 3.0" 的视角看：declarative 是把约束丢给 solver（GPU-friendly 优化），imperative+search 是把符号程序丢给 local search（CPU-friendly 但 search space 极小，因为参数化关系压缩了自由度）。

---

## 3. PSDL：Procedural Scene Description Language

PSDL 是一个 Python-embedded DSL，意味着 LLM 生成的就是合法 Python 代码，但有几个重载过的 domain-specific operators。这跟你 e.g. 看 DreamCoder / Library Learning 的思路类似：DSL 设计本身就是 inductive bias。

### 3.1 三个核心 feature

| Feature | 作用 | 类比 |
|---|---|---|
| **Explicit Geometric Relationships** | position 写成相对于其他 object 或 scene bounds 的函数 | SVG 的 transform / CSS 的 relative positioning |
| **Expression sharing through variables** | 一个变量控制多个 object | 配色 palette、CSS variable |
| **Control-flow for structured repetition** | `for` loop 隐式共享 stride 常量 | SwiftUI ForEach / React list rendering |

论文 Table 1 给的对比例子非常直观：

```
# Explicit Geometric Relationships          # Use of Loops and Variables
chair.max.x = table.min.x - 0.1             d = 2.0
chair.center.y = table.center.y             for i, c in enum(cols):
chair.min.z = scene.min.z                       c.center.x = scene.center.x + i * d
chair.facing = table.facing
```

左边那一段：chair 紧贴 table 的左侧 0.1m，center 对齐，紧贴地面，朝向和 table 一致。这是 4 条 relational 约束。任何 perturbation（改 0.1、改 table 位置）都会自动 propagate——这正是 imperative 缺失的 "constraint preservation"。

右边那一段：一个 spacing 变量 `d` 控制一整列 object 的间距。local search 只需要调一个数字就能移动整列。

### 3.2 API 设计细节（Table 8, 9）

每个 object 是 cuboid：
- `o.width`（垂直于 facing 方向的尺寸）
- `o.depth`（沿 facing 方向）
- `o.height`（向上）
- `o.center`（几何中心，vec3）
- `o.facing` ∈ {X_NEG, X_POS, Y_NEG, Y_POS}（限制为 4 个 cardinal 方向，这是为了和 prior work 对齐）
- `o.min`, `o.max`（AABB 的两个角，vec3）
- `o.support` ∈ {STANDING, WALL_MOUNTED, FLOATING}

**赋值操作符被重载**——这是整个 DSL 最巧妙的设计：
```
chair.max.x = table.min.x - 0.1
```
这 *不是* 修改 `chair.max` 这个 vector 的 x 分量，而是 *translate chair*（沿 x 轴）使得新的 `chair.max.x` 满足等式。也就是说，DSL 把 LHS 当成一个 *goal constraint*，求解器隐式地反推出 object 的 `center.x` 应该是多少。这其实是把 declarative 的 "constraint" 偷偷塞进了 imperative 的语法里。

**`set_coordinate_frame(o)`**：建立 local frame，y 轴对齐 `o.facing`，x 轴顺时针 90°，z 轴向上。这让 LLM 可以在 canonical 空间里构造 sub-layout，然后整体 transform 进 global scene。这跟 computer graphics 里的 model matrix / view matrix 是同源思想，也跟 scene graph 的 hierarchical transforms 一致。

### 3.3 为什么这些 feature 对 error correction 至关重要

这是 paper Section 4 的核心 argument，也是 build intuition 的关键：

- **裸 imperative**（LayoutGPT 风格）：search space = 每个 object 的 position + orientation。Fix 一个 overlap 通常意味着把某个 object 单独挪走，这就破坏了它和其他 object 的关系（比如把椅子从桌旁挪到房间另一头去避免和桌子重叠）。
- **PSDL**：search space = 程序里的 constant expressions 和 facing expressions。Fix 一个 overlap 通常意味着把 `0.1` 改成 `0.15`——椅子还是紧贴桌子，只是稍微远一点点。关系 *by construction* 被保留。

这就是为什么 Table 6 里 gradient descent 和 basic imperative 的 *error count* 更低（0.5, 0.8），但 *preference rate* 反而比 PSDL local search 差（61.4%, 71.4% vs 80%+）。**消除错误容易，保留语义难**。

---

## 4. Error Correction：形式化与算法

### 4.1 Loss function

`loss(L)` 由四项组成（L 是一个 layout）：

1. **Out-of-Bounds Loss**: 对每个 object，其 bounding cuboid 突出 scene boundary 的最大线性距离。完全在内则为 0。
2. **Overlap Loss**: 对每对 object，AABB 交集体积的 cube root（立方根让量纲是长度，便于和其他项相加）。Doors 和 windows 有 expanded collision boxes（防止被挡住开关/视野）。
3. **Standing Loss**: 对每个 STANDING object，其底面到最近可支撑水平面的距离。
4. **Mounted Loss**: 对每个 MOUNTED object，其 mountable face 到最近可支撑垂直面的距离。

这四项把 declarative 系统里通常作为 hard constraint 的东西转化成 soft loss。注意 doors/windows 的特殊 collision box 是个 domain knowledge hack，但很合理。

### 4.2 目标函数

$$
\arg\min_{P} \big[\text{loss}(L) + d(P, P_0)\big]
$$

变量解释：
- $P_0$: LLM 原始生成的 program
- $P$: 修改后的 candidate program
- $L, L_0$: $P$ 和 $P_0$ 执行后产生的 layouts
- $d(P, P_0)$: program 之间的距离

距离进一步分解：

$$
d(P, P_0) = d_{\text{edit}}(P, P_0) + d_{\text{OT}}(L, L_0)
$$

- $d_{\text{edit}}$: program edit distance，定义为从 $P_0$ 到 $P$ 的最短 elementary edit 序列长度。Elementary edits = 重写 constant expressions + 重写 direction expressions（这是从经验观察得出的——LLM 大多数错误就是数字和方向）。
- $d_{\text{OT}}$: layouts 之间的 optimal transport 距离：

$$
d_{\text{OT}}(L, L_0) = \min_{f \in \mathcal{F}} \sum_{o \in \mathcal{O}(L)} \text{Vol}(o) \cdot \|\text{center}(o) - \text{center}(f(o))\|_2
$$

变量解释：
- $\mathcal{O}(L)$: layout $L$ 中的 object 集合
- $\mathcal{F}$: 所有保持 object category 的 bijections $f: \mathcal{O}(L) \to \mathcal{O}(L_0)$（chair 必须映射到 chair，table 映射到 table，etc.）
- $\text{Vol}(o)$: object 体积
- $\text{center}(o)$: object 几何中心
- $\|\cdot\|_2$: L2 距离

这个 OT 项的设计很 elegant：
- **体积加权**：大 object（沙发、床）移动代价高，小 object（书、花瓶）移动代价低。这鼓励 "保留 major scene elements 的位置，允许 minor elements 调整"——这正好符合人类对 scene 编辑的直觉。
- **Category-preserving bijection**：避免 "把 chair 移到 table 的位置然后说 layout 没变" 这种 degenerate solution。
- 这本质上是 Wasserstein-1 距离的一个变种（with mass = volume）。参考 [Peyré & Cuturi 2019]。

为什么要 $d_{\text{edit}}$ *加* $d_{\text{OT}}$？两者捕捉不同层级的 "similarity to original"：
- $d_{\text{edit}}$ 是 *symbolic* similarity（程序文本相似）。
- $d_{\text{OT}}$ 是 *geometric* similarity（执行结果相似）。
- 一个 program 可能文本差很多但执行结果一样（比如重排了变量声明），也可能文本差不多但执行结果差很多（比如改了一个关键的 0.1）。两者一起约束更鲁棒。

### 4.3 算法：Iterative Local Search

完整目标函数的精确最小化在 program 空间里 intractable（combinatorial explosion）。论文用 iterative local search：

$$
P_{i+1} = \arg\min_{P \in \mathcal{N}(P_i)} f(L), \quad f(L) = \log s(L) + d_{\text{OT}}(L, L_0)
$$

注意：实际迭代时 *不* 显式算 $d_{\text{edit}}$，因为 neighborhood $\mathcal{N}(P_i)$ 本身就是通过 elementary edits 构造的，每个 candidate 的 $d_{\text{edit}}$ 都是常数 1（一个 edit）。所以只需最小化 $\log s(L) + d_{\text{OT}}(L, L_0)$。$\log s(L)$ 是 loss 的 log，估计是为了让不同量级的 loss 项和 OT 项更平衡。

Neighborhood 定义：

$$
\mathcal{N}(P) = \{e(P) \mid e \in E_{\text{fin}}(P)\}
$$

$E_{\text{fin}}(P)$ 是从 $E(P)$ 中随机采样的 finite subset：
- 每个 constant expression 采样 **10 个 edits**：把常量 $c$ 替换为 $c \cdot \pm 4^\epsilon$，sign 均匀取 $\{-1, +1\}$，$\epsilon \sim \mathcal{U}[-1, 1]$。这意味着 multiplier 在 $[0.25, 4]$ 之间（因为 $4^{-1} = 0.25$, $4^{1} = 4$），log-uniform 分布。这是个很合理的 "perturb magnitude" prior——既包含微调（×0.97）也包含大调（×3.8）。
- 每个 direction expression 采样 **4 个 edits**：所有 4 个 cardinal directions（即穷举，因为只有 4 种）。

终止条件：没有可用 edit 使 $f$ 下降超过某个 small threshold。

**关键直觉**：这个 local search *不* 直接优化 object positions，而是优化 *program expressions*。在 PSDL 里，因为 variable sharing 和 loops，一个常量可能控制十几个 objects。所以 "改一个数字" 在 PSDL 里等价于 "在 basic imperative 里同时改十几个 objects 的 coordinates 但保持它们之间的关系"——这就把高维 search 投影到低维 semantic manifold 上了。

平均一个 scene 只需要 **7.13 次 adjustments** 就能 resolve 所有 errors。这是个非常小的数字——说明 LLM 离 "valid program" 已经很近，只是数值有点 off。

---

## 5. 实验设计

### 5.1 公平比较的 pipeline

为了 isolate layout generation 这一阶段，所有方法共享：
- **Template generation**（9.5s）：LLM 生成 scene dimensions + object list（name, dimensions, support type）。
- **Object retrieval**：CLIP similarity 从 HypeHype Asset Library（~6000 assets）检索 mesh。

只在 *layout generation* 阶段不同：DeclBase / Holodeck / FlairGPT 用 declarative，ours 用 PSDL + error correction。Object orientations 都限制为 4 个 cardinal directions（为了和 prior work 对齐）。

### 5.2 Benchmark

70 个 prompts，沿 4 个 axes 标注：
- size: small (14) / medium (35) / large (21)
- location: indoor (48) / outdoor (22)
- realism: realistic (50) / fantastical (20)
- structure: chaotic (31) / structured (39)

加上 Holodeck 用过的 52 个 MIT indoor scene prompts + 14 个 Holodeck qualitative long prompts = 136 个 prompts 总共。其中 19 个被标为 'complex'（4+ words）。

### 5.3 Comparison conditions

- **Ours**: PSDL + local search error correction
- **DeclBase**: [Aguina-Kang et al. 2024]
- **Holodeck**: [Yang et al. 2023] 的 Constraint-based Layout Design Module
- **FlairGPT**: [Littlefair et al. 2025]

LayoutGPT 被显式排除，因为被 Holodeck 和 DeclBase strictly dominated。

---

## 6. 实验结果

### 6.1 Human Perceptual Study（Table 2, 7, Fig. 3）

10 个 Brown 学生参与者，2AFC（two-alternative forced choice），每个 participant 看 70 个 comparison（prompt + 两张随机顺序的 layout 渲染图）。Object 渲染为彩色 box，hover 显示 name（剔除 3D mesh 选择的影响）。Majority vote 作为 final answer。

| Comparison | Preference for Ours |
|---|---|
| Ours w/ EC vs. DeclBase | **82.9%** |
| Ours w/ EC vs. Holodeck | **94.3%** |
| Ours w/ EC vs. Ours w/o EC | **74.3%** |
| Ours w/o EC vs. DeclBase | 61.8% |

按 category 拆分（Table 7）：
- vs Holodeck：fantastical 100%, small 100%, large 95.2%, indoor 95.8%, chaotic 96.8%
- vs DeclBase：structured 89.7%, large 90.5%, outdoor 86.4%

**关键发现**：在 large structured scenes（论文 thesis 的 target）上优势最明显。Holodeck 在 fantastical 和 chaotic 上几乎被碾压，说明 declarative DSL 的表达能力限制在复杂/非典型场景下放大了。

Fig. 3 的 per-task histogram：vs Holodeck 有 51/70 task 是 5/5 unanimous for ours——这是个非常强的信号。

### 6.2 Automated Evaluation：LLMCompare（Table 3, 4, 5）

先发现现有 text-to-image metrics 不行：
- **VQAScore** [Lin et al. 2024]: VQA model 给 "Does image depict text?" 的 'yes' 概率。**58.6% agreement** with human majority——只比 chance (50%) 好一点。原因：VQA 能识别 "这是 bedroom" 即使 layout 烂，只要 objects 在就行。
- **DSG** [Cho et al. 2024]: 生成 yes/no 问题 dependency graph 然后 aggregate。**50.7%**——基本上是 chance。原因：DSG 生成的 yes/no 问题区分不出两个 layouts 的差异，经常 tie。

提出的 **LLMCompare**：prompt multimodal LLM (GPT-4o) 同时看两张 layout 图 + prompt，让它先列每个 layout 的 pros/cons，最后输出哪个更好。
- LLMCompare: **77.1%** agreement with human
- LLMCompare (no pros/cons step, ablation): 70.0%——先列 pros/cons 这一步确实有用，符合 chain-of-thought 的直觉。

Table 4：在自动 metric 下，Ours vs 各 baseline 的 preference rate：
- vs DeclBase: 76.5% (perceptual: 82.9%)
- vs Holodeck: 82.4% (perceptual: 94.3%)
- vs FlairGPT: 88.2%

Table 5：按 object 数量拆分 vs Holodeck：
- 10-20 objects: 66.7%
- 20-30: 80.0%
- 30-40: 72.9%
- **40+: 95.2%**

这非常强烈地支持论文的核心 thesis：**object 数量越多，imperative + low-dimensional parameter search 的优势越大**。Declarative solver 在 40+ objects 时已经不堪重负（21.3s 平均 solver 时间，且 quadratic in relations）。

### 6.3 Error Correction Ablation（Table 6）

| Error Correction Method | Pref rate | # Errors | Run time |
|---|---|---|---|
| No solver | 60.0% | 12.3 | 0s |
| Gradient descent | 61.4% | **0.5** | 2.6s |
| LLM self-repair | 68.1% | 5.7 | 106.7s |
| Local search, basic imperative | 71.4% | 0.8 | 25.2s |
| **Local search, PSDL (ours)** | **~80%** | 1.1 | **9.3s** |

（论文没有直接给出 PSDL 的 preference rate 数字，从 Table 2 的 82.9% / 94.3% 推断大约 80%+）

关键 takeaways：
1. **Gradient descent 消除 errors 最多 (0.5)** 但 preference rate 最低 (61.4%)——因为它把每个 object 独立移动，破坏关系。
2. **LLM self-repair 慢一个数量级** (106.7s vs 9.3s)，且 errors 残留多 (5.7)——LLM 不擅长 numerical refinement。这跟 [Olausson et al. 2024] 在 code generation 上发现 "self-repair usually gives modest or no gain" 一致。
3. **PSDL local search vs basic imperative local search**: 同样是 local search，PSDL 版本 (9.3s) 比 basic imperative 版本 (25.2s) 快近 3 倍，因为 loops + variables 降低了 search space 维度。Basic imperative 的 pref rate (71.4%) 也比 PSDL 低，因为它的 search 空间在 object-level 而非 program-level。

参考：[Olausson et al. 2024 "Is Self-Repair a Silver Bullet for Code Generation?"](https://arxiv.org/abs/2308.15975)

### 6.4 Timing

- Template generation: 9.5s
- PSDL program generation (one LLM call): 19.2s
- Error correction (local search, no LLM): 9.3s
- **Total: ~38s**
- DeclBase: 10s program + 21.3s solver = 40.8s（共享 template stage）

整体 runtime comparable，但 PSDL error correction 比 declarative solver 快 2 倍多，且差距随 object 数量增长会进一步扩大。

### 6.5 LLM Backbone Robustness（Table 10）

测试了 4 个 LLM：claude-3-5-sonnet-20241022、gpt-4o-2024-11-20、o1-2024-12-17、gemini-exp-1206。

Pre-correction errors (ALL / BOUND / OVL)：
- Claude: 17.57 / 1.56 / 10.76
- GPT-4o: 17.80 / 1.14 / 11.19
- o1: 17.29 / 2.36 / 10.70
- Gemini: 14.67 / 1.19 / 10.41

Post-correction errors：
- Claude: 2.10 / 0.54 / 0.56
- GPT-4o: 3.34 / 0.61 / 0.64
- o1: 3.24 / 0.63 / 1.17
- Gemini: 3.14 / 0.61 / 1.14

观察：
1. 所有 LLM pre-correction 都有大量 errors（14-18），证实 "LLM 直接生成 valid layout 很难"。
2. Error correction 对所有 LLM 都有效，把 errors 降到 2-3.3。
3. Claude post-correction errors 最少 (2.10)，但其 pre-correction 也最高 (17.57)——correction 的 *相对* 改善 (88%) 跟其他 LLM 类似。这暗示 PSDL + local search 是个 LLM-agnostic 的修复框架。

---

## 7. Architecture 图解析（Fig. 1, 2）

Fig. 1 是 teaser：展示 6 个生成的 3D scene，从室内（客厅、图书馆）到室外（广场、剧场），结构化程度不一。强调 "open-universe"——不限于固定 room types 或 object categories。

Fig. 2 是 error correction 的 worked example，对比同一个初始 layout 用两种语言（basic imperative vs PSDL）+ 不同 correction 方法的结果：
- 初始：有 out-of-bounds 和 overlap，loss > 0
- Basic imperative + local search：loss = 0 但 table 和 chair 的关系被破坏（chair 被挪到很远去避免和 table 重叠）
- PSDL + local search：loss = 0 且 table-chair 关系保留
- Gradient descent：同样破坏关系

这个图是论文核心 claim 的 visual proof。

---

## 8. 与你的 intuition 的连接点

### 8.1 LLM as programmer, not predictor

你一直在强调 LLM 应该输出 *code*，而不是 *numbers*。这篇论文是又一个 case study：LLM 直接输出 chair coordinates 时表现很差，但当它输出 `chair.max.x = table.min.x - 0.1` 这种 relational program时，结果质量大幅提升。这跟 Meta 的 Toolformer、你的 nanoGPT + calculator 思路是同源的：把 LLM 限制在它擅长的 symbolic reasoning 上，把 numerical precision 交给外部机制。

### 8.2 Verifier in the loop, not LLM in the loop

Error correction 模块 *不调用 LLM*，只用 local search + loss function。这跟你最近在 reasoning models / AlphaProof 讨论中强调的 "verifier 比 self-reflection 更可靠" 完全一致。Table 6 里 LLM self-repair (106.7s, 5.7 errors) 被无声地碾压。

这也让人想到 Lean / Coq 的 tactic search：LLM propose tactics，prover 验证。这里 LLM propose program skeleton，local search refine constants to satisfy physical constraints。**Constraint satisfaction 是 verifier 的工作，不是 generator 的工作**。

### 8.3 Compression = Generalization

PSDL 的 variable sharing 和 loops 把 layout 压缩成更短的程序。这跟你在 "A Path Towards Autonomous Machine Intelligence" 演讲里讨论的 world model 的 predictive coding 思路有精神上的呼应：*低维 latent* 更容易 search、更容易 generalize。Search space 从 "每个 object 6 DoF" 压缩到 "十几个关键常量"，这是巨大的维度缩减。

### 8.4 AlphaProof-style search at the symbolic level

你的 tweet 早就指出：AlphaProof 的精髓是 *在 Lean program 空间里 MCTS*，不是在 raw proof space。这篇论文是同样的转移：*在 PSDL program 空间里 local search*，不是在 raw layout space。搜索对象是符号结构，不是连续参数。这使得 search 的每一步都 *semantically meaningful*——改一个 spacing 常量有明确的语义，改一个 chair 的 x 坐标没有。

### 8.5 与 Scene Language [Zhang et al. 2024] 的关系

Scene Language 是个 concurrent work，也用 imperative approach（生成 scene programs with control flow），但**没有** parametric relationships、local coordinate frames、或 error correction。它把 program 转化为 detailed scene 是通过 neural rendering，不是通过 correction。所以 PSDL 在 *layout correctness* 这个维度上显著超越 Scene Language——但 Scene Language 在 *visual fidelity* 上可能更强（因为它用 neural rendering）。两者的结合是个明显的 future direction：PSDL 生成 valid skeleton，neural renderer 在 skeleton 上做 detail synthesis。

参考：https://ai.stanford.edu/~yzzhang/projects/scene_language/

### 8.6 与 DreamCoder / Library Learning 的连接

PSDL 的 design pattern 思想（relational primitives, control flow, variable sharing）跟 DreamCoder 的 "learn reusable primitives" 思路有相似性。区别是 PSDL 的 primitives 是 *hand-designed*（来自 domain knowledge），DreamCoder 是 *learned*。一个自然的 follow-up：能不能让 LLM 在生成 PSDL 程序的过程中 *learn new primitives*（比如"place a row of chairs around a table"作为一个 callable）？

参考 DreamCoder: https://arxiv.org/abs/2206.08322

---

## 9. Limitations 和 Future Work（论文自承 + 我的联想）

### 9.1 论文自承
- Local search 只 edit 数值常量和 facing directions——不能 swap object identities、不能 swap x/y 坐标。这意味着 LLM 如果 *misinterpret* 了 prompt（把 chair 应该在 table 左边写成右边），error correction 救不了。
- Loss 没有高阶美学（line-of-sight, walkability, affordances beyond support/collision）。比如生成的 layout 可能物理 valid 但 walkable 很差（家具挡走廊）。
- VLM evaluator 本身是 proxy，可能有 bias，未来系统可能 overfit LLMCompare 的 blind spots。
- 固定 object retrieval + 4 cardinal orientations——relaxing 这些会暴露新问题。

### 9.2 我的联想
1. **Higher-order edits**: 现在的 elementary edits 太弱。能不能加 "swap two object identities"、"negate a facing"、"swap x/y of an expression"？这会让 search space 大但更 powerful。
2. **Learned edit proposals**: 现在 edits 是 random perturbation。可以让一个小 LM 学习 *where to edit*——类似 AlphaGo 的 policy network 减少 MCTS branching。这跟 "Self-Debugging" / "Reflexion" 的区别是：仍然不调用 LLM 自己，而是用一个小的 *specialized* edit-proposer。
3. **Hierarchical programs**: PSDL 现在是 flat Python。如果支持 function definitions（"def dining_set(table, chairs, spacing): ..."），LLM 可以 reuse 之前生成过的 sub-programs。这跟 CodeLlama / HumanEval 的 function-level generation 更对齐。
4. **Differentiable PSDL**: 现在的 loss 是 non-differentiable through program（program 是离散的）。但如果 PSDL 子集是 *differentiable*（比如所有 assignments 都是 affine combinations），就可以用 gradient-based optimization 替代 random local search。这跟 JAX / Dex 的 differentiable programming 思路一致。
5. **Dynamic scenes**: 论文 future work 提到。PSDL 扩展到 temporal programs（`for t in range(T): chair.center.x = t * velocity`）可以生成 animations。
6. **Affordance loss**: 把 walkability、line-of-sight、reachability 作为 additional loss terms。可以做 path-planning-based loss（A* 从门到每个 chair 的路径长度）。
7. **Active prompt refinement**: 如果 error correction 失败（loss 降不下来），可以让 LLM *clarify* prompt（"did you mean round tables or rectangular tables?"）。这是 LLM-in-the-loop 的合理使用场景，跟单纯 self-repair 不同。
8. **PSDL as scene representation for RL**: 生成的 PSDL 程序可以作为 embodied AI 的 environment spec。比如 AI2-THOR / Habitat 可以从 PSDL 直接 spawn scenes，比从 mesh 数据集采样更 flexible。
9. **Token-level LLM sampling + program-level verification**: 现在 LLM 一次性生成整个 PSDL program。能不能用 best-of-N + verifier filter？或者 token-level rejection sampling？这跟 STaR / RLAIF 思路一致。

---

## 10. 一些 critical takeaways

1. **DSL design is the real contribution**。PSDL 的三个 feature（relational, variables, control flow）才是让 imperative + search work 的关键。Error correction 算法本身（local search）并不新颖，novelty 在于 *search space 的设计*。
2. **OT distance with volume weighting** 是个 elegant trick，值得记住。在任何 "modify while preserving structure" 的任务里都适用（比如 3D pose refinement, image editing）。
3. **LLMCompare eval method** 简单到令人惊讶：让 VLM 列 pros/cons 然后选。但 77.1% vs 50.7% (DSG) 的提升说明 *structured prompting* 对 evaluation 很重要。这跟 chain-of-thought prompting 在 generation 上有效是平行的发现。
4. **Open-universe 仍然是 undersold**。Benchmark 70 prompts 还是偏 indoor realistic。真正的 open-universe test 应该包括 e.g. "a Mars colony with hydroponic farms and airlocks" 这种——fantastical category 只有 20 个 prompts，且 vs Holodeck 是 100%（说明 Holodeck 完全没法做这种）。
5. **7.13 edits per scene** 这个数字很 striking。LLM 其实已经基本 "懂" scene structure，只是数字不太对。这跟程序合成的 "skeleton correct, constants off" 现象一致，让人想起 Codex/HumanEval 上 LLM 经常正确算法 + 错误 off-by-one。
6. **38s end-to-end** 对 scene synthesis 来说是可用的。但 19.2s 给 LLM 写 PSDL program 这一步是瓶颈——如果未来用更小的 specialized LM 替代 Claude/GPT-4o 写 PSDL，能大幅加速。

---

## 11. 最终的 intuition

这篇论文真正的洞察可以浓缩成一句：

> **Imperative programs with parametric relationships are a *compressed* representation of layouts. Compression reduces search space dimensionality. Reduced search space makes symbolic local search feasible. Symbolic local search preserves semantic structure by construction.**

四步因果链，每一步都 *obvious in hindsight*，但组合起来是这个领域的 paradigm shift。从 your "compression is intelligence" 的视角看，这篇论文其实是在说：*let LLM compress scene into a program; let search decompress the program into a valid layout*。

这种 "LLM compresses, search decompresses" 的分工，比 "LLM compresses, LLM self-repairs" 或 "LLM emits constraints, solver decompresses" 都更 efficient——前者因为 LLM 不擅长 numerical refinement，后者因为 solver 在高维 continuous space 里慢且 DSL-restricted。PSDL 的 middle ground 拿到了两边的优点。

期待看到这个思路扩散到其他 LLM-for-graphics 任务（character rigging, particle systems, shader generation）和其他 LLM+symbolic reasoning 任务（math, theorem proving, query planning）。

---

### 进一步阅读

- Holodeck: https://arxiv.org/abs/2312.09067
- LayoutGPT: https://openreview.net/forum?id=Xu8aG5Q8M3
- DeclBase / Open-Universe Indoor Scene Generation: https://arxiv.org/abs/2403.09675
- Scene Language: https://arxiv.org/abs/2410.16770
- DreamCoder: https://arxiv.org/abs/2206.08322
- Optimal Transport (Peyré & Cuturi): https://arxiv.org/abs/1803.00567
- VQAScore: https://arxiv.org/abs/2404.01291
- DSG: https://arxiv.org/abs/2405.03253
- Self-Repair Silver Bullet?: https://arxiv.org/abs/2308.15975
- Objaverse-XL: https://arxiv.org/abs/2307.05663
- ATISS (same group): https://arxiv.org/abs/2110.04275
- PlanIT (same group): https://web.archive.org/web/20190821193324/https://cs.stanford.edu/~kaikai/planit/
- DiffuScene (same group): https://arxiv.org/abs/2305.02755
- I-Design: https://arxiv.org/abs/2404.02838
- SIGGRAPH Asia 2025: https://sa2025.siggraph.org/

如果你想要我 deep dive 某个具体方面——比如 optimal transport 那一项的更多数学细节、PSDL API 的完整边界 case、或者跟 AlphaProof 的更深类比——告诉我，我可以再展开。
