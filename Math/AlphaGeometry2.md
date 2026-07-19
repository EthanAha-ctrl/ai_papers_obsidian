---
source_pdf: AlphaGeometry2.pdf
paper_sha256: 54ea0a3c557cbf19f57a9c255e096cd4241c94d8563939a174f8a17a7ecefd28
processed_at: '2026-07-18T07:58:55-07:00'
target_folder: Math
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AlphaGeometry2 深度讲解

 Andrej，这篇 DeepMind 的 AG2 paper 信息密度极高，我尽量把每条技术线都展开到公式、架构、实验数据级别，并夹带一些我的 intuition 和相关联想。

---

## 1. 宏观定位：为什么需要 AG2

AG1 (Trinh et al., 2024, Nature) 是一个 neuro-symbolic 系统：LM 负责猜 auxiliary constructions，symbolic engine DDAR 负责 deduction closure。AG1 在 IMO-AG-30 上做到 25/30，但扩展到 2000–2024 全部 IMO geometry（IMO-AG-50）只有 **27/50 (54%)**。瓶颈集中在三处：

1. **Domain language 太窄**：只有 9 个 predicates，无法表达 "Find x"、linear equations of angles/distances、locus problems、non-constructive problems。
2. **Symbolic engine 太慢**：Python + polynomial-complexity rule matching，最坏 O(N⁸) for similar triangle search，限制了 synthetic data 规模和 inference-time search 深度。
3. **LM 太弱 + 搜索太单调**：AG1 用单一 beam search，模型是 custom transformer。

AG2 把这三条线分别升级，最终在 IMO-AG-50 上做到 **42/50 (84%)**，超过 average gold medalist 的 40.9/50。这是 neuro-symbolic 第一次在 IMO geometry 上"defeat"金牌选手。

相关链接：
- AG1 Nature paper: https://www.nature.com/articles/s41586-023-06747-5
- AG2 code: https://github.com/google-deepmind/alphageometry2
- DeepMind IMO silver blog: https://dpmd.ai/imo-silver
- IMO 2024 P4 solution: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/imo-2024-solutions/P4/index.html

---

## 2. Domain Language 扩展（Section 2）

### 2.1 AG1 的 9 个 predicates（Table 1）

`cong, perp, para, coll, cyclic, eqangle, eqratio, aconst, rconst` —— 只能描述**静态**几何关系。

### 2.2 AG2 新增的"compute" predicates

```
acompute a b c d  → "Find angle between AB and CD"
rcompute a b c d  → "Find ratio AB/CD"
```

这解锁了像 IMO 2009 P4 这种 "Find the angle..." 的题型。

### 2.3 Linear equations over geometric quantities

这是 IMO 2024 P4（关于 ∠KIL + ∠YPX = 180°）这类问题能被 formalize 的关键。三个新 predicates：

$$
\text{distmeq } a_1 b_1 \dots a_n b_n\ t_1 t_2 \dots t_n\ y \ \Longleftrightarrow\ t_1 \log(A_1 B_1) + t_2 \log(A_2 B_2) + \dots + t_n \log(A_n B_n) + y = 0
$$

$$
\text{distseq } a_1 b_1 \dots a_n b_n\ t_1 t_2 \dots t_n \ \Longleftrightarrow\ t_1 A_1 B_1 + \dots + t_n A_n B_n = 0
$$

$$
\text{angeq } a_1 b_1 \dots a_n b_n\ t_1 \dots t_n\ y \ \Longleftrightarrow\ t_1 d(A_1 B_1) + \dots + t_n d(A_n B_n) + y = 0
$$

其中：
- 下标 $i \in \{1, \dots, n\}$ 索引不同的几何量；
- $t_i \in \mathbb{Z}$ 是该几何量的系数；
- $y \in \mathbb{R}$ 是常数项；
- $d(AB)$ 指 line AB 与 horizontal line 之间的**无方向**夹角；
- `distmeq` 用 log 是为了把"乘积/比例"关系转化为线性形式（log 的 linear 化技巧，等同于 working in log-space for ratios）。

**Intuition**: `distmeq` 的设计让我想起 Mass point / barycentric coordinate 的 trick —— 乘法律 → 加法律。把 AB·CD = EF·GH 这种 power-of-point / similar-triangle 推论直接写成 log-sum = 0，AR sub-engine 就能用 Gaussian elimination 一次性解决所有 ratio chasing。

### 2.4 Locus predicates（Table 2，11 种 case）

引入 placeholder token `*` 表示 fixed point。例如：

```
?cyclic a b c * : x  → "When x moves, the circumcircle of abc always passes through a fixed point *"
?cong b c a * : x     → "When x moves, |BC| = |A*|, i.e. x traces a circle centered at * with radius BC"
?coll a b * : x       → "When x moves, x stays on line ab"
```

这是 AG2 区别于 AG1 的本质飞跃 —— 系统现在能 reason about **movement** 和 **fixed-locus** 而不只是 fixed snapshots。

### 2.5 Diagram topology predicates

- `sameclock a b c d e f`: orientation(A→B→C) = orientation(D→E→F)
- `noverlap a b`: A ≠ B
- `lessthan a b c d`: AB < CD（for SSA congruence）
- `overlap a b`: A ≡ B（允许 coincidence reasoning）
- `cyclic_with_center a_1 ... a_n x`: a_1 = ... = a_x = center, a_{x+1}...a_n on the circle

### 2.6 Non-constructive problem relaxation

AG1 强制每个 point 至多由 2 个 predicates 定义（intersection of 2 objects），AG2 放宽到 ≥3 个 predicates，配合 Section Appendix G 的自动 diagram 生成。覆盖率从 **66% → 88%**。

剩下 12% 是 3D、inequalities、非线性方程、variable number of points（n 任意），Appendix H 给了 inequality rules 的草稿，但未集成。

---

## 3. Symbolic Engine：DDAR2（Section 3）

### 3.1 Double points handling（Section 3.1）

这是 IMO hard problems 最关键的能力。Figure 1 的例子：要证 a ∩ b = X ∈ ω 很难，但 LM 可以 propose 一个 auxiliary point X' = a ∩ ω，然后 DDAR 证 X' ∈ b ⇒ X = X'。

这个 "reformulation via auxiliary double-point" 等价于人类在 inversion / projective geometry 里的"换 working point"的直觉。AG1 完全不能这样思考，因为 DDAR1 拒绝同名不同坐标的 point。

### 3.2 Faster algorithm

AG1 的两个 bottleneck：
1. **Similar triangle search**：worst case O(N⁸)（每边取 2 个 point = 4 个，再找两个三角形 = 8 个 point）。
2. **Clause matching**：exponential in clauses-per-premise。

AG2 的 trick：
- 对所有 point triples，hash 它们的"shape"（symbolic normal form via AR sub-engine），重复即相似。把 O(N⁸) 降到 O(N³) + hash lookup。
- 对 cyclic quadrilaterals：对 (point X, segment AB) hash `(A, B, ∠AXB)` 的 normal form。重复 ⇒ X, A, B, Y 共圆。
- AR sub-module 保持 known linear equations（angles, distances, log-distances）的 closure，把任何 linear expression reduce 到 normal form。

### 3.3 Faster implementation

Gaussian elimination 的核心在 C++ via pybind11 (https://github.com/pybind/pybind11)。

Benchmark（25 道 DDAR 解不掉的 IMO 题，跑 50 次，AMD EPYC 7B13 64-core）：

| Engine | Runtime (s) |
|---|---|
| DDAR1 (Python) | 1179.57 ± 8.055 |
| DDAR2 (C++) | 3.44711 ± 0.05476 |

**~340× speedup**。这允许 AG2 在 inference 时跑更多 search tree branches，data generation 时跑更复杂的 random diagrams。

**Intuition**: 在 neuro-symbolic 系统里，symbolic engine 的 speedup 是 multiplicative 的 —— 它既加速 data generation（→ bigger/better LM），又加速 inference search（→ deeper beam）。一个 300× speedup 可以在端到端 solve rate 上贡献 10–20%。

---

## 4. Synthetic Data Generation（Section 4）

### 4.1 Pipeline 概览

1. 随机采样 random diagram；
2. DDAR 跑 deduction closure；
3. 对每个 deduced fact，跑 traceback algorithm 提取 minimal premises + aux points + deduction steps。

AG2 坚持从 **random diagrams** 出发（不用人类题），理由是避免 contamination 并探索超出人类知识分布。这与 TongGeometry (Zhang et al., 2024, https://arxiv.org/abs/2412.10673) 形成对比 —— 后者用 human expertise + 现有 problem diagrams 引导生成。

### 4.2 Scale-up 对比（Figure 2）

| 维度 | AG1 → AG2 |
|---|---|
| Random diagram size | 2× larger |
| Theorem complexity (points + premises) | up to 2× |
| Proof length | up to 10× |
| Aux vs no-aux 比例 | 9:91 → 50:50 |
| Question type balance | imbalanced → balanced |

50:50 这个数字特别关键 —— AG1 几乎全是 no-aux 证明，导致 LM 在 aux proposal 上严重欠拟合。

### 4.3 Movement function P(X)

为生成 locus problems，AG2 定义：

$$
P(A) := \text{set of points that control the movement of } A
$$

Table 3 例子：
- `a = midpoint b c, d = midpoint a c` ⇒ P(d) = {b, c}（d 的运动源是 b, c，因为 a 完全由 b, c 决定）
- `a = on_line b c` ⇒ P(a) = {a, b, c}（a 自己也可以在 b,c 上任意移动，所以 a 也算自己的 source）

Table 5 列了 **17 个 detection cases**，对应 11 种 locus statements。例如：
- `cong a b c d` + `P(b,c,d) − P(a) ≠ ∅` ⇒ "circle center b radius cd passes through fixed point a" (Case 2)
- `cyclic a b c d` + `P(b,c,d) − P(a) ≠ ∅` ⇒ "circumcircle of bcd passes through fixed point a" (Case 1)
- `para a b c d` + `P(c,d) − P(a,b) ≠ ∅` ⇒ "line cd is always parallel to a fixed line" (Case 10)

**Intuition**: P(X) 是一种 dependency-DAG 上的"free-variable closure"。这个 trick 让 AG2 能自动 detect 哪些 predicates 在 movement 下保持不变 —— 这是 IMO locus 题的本质（locus 是 movement 下的 invariant）。

### 4.4 Greedy minimal-point pruning（Figure 3）

AG1 的 traceback 用 exponential subset search 找 minimal premises set，对大 diagram 不可行。AG2 改用 greedy：

```python
def prune_points(points, check_provable):
    pruned = set(points)
    for p in reverse_topological(points):
        if check_provable(pruned - {p}):
            pruned = pruned - {p}
    return pruned
```

**Key insight**: 只要 `check_provable` 是 monotonic（A ⊆ B ⇒ provable(B) ⇒ provable(A)），greedy 保证找到 inclusion-minimal set。但 construction dependency 让 check 不 monotonic，AG2 通过 reverse-topological order 处理：先考虑 leaves（不依赖其他点的），最后考虑 root construction points，保证每一步删除后剩余 set 仍然 constructible。

---

## 5. Search Algorithm: SKEST（Section 5）

### 5.1 核心思想

**Shared Knowledge Ensemble of Search Trees**：多个不同配置的 beam search 并行跑，失败 node 把已证 facts 写入 shared database，成功则全部 terminate。

### 5.2 架构图解析（Figure 4）

```
   ┌─────────────────────────────────────┐
   │       Shared Facts Database          │
   └────┬───────┬───────┬───────┬─────────┘
        │       │       │       │
   ┌────▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐
   │Tree 1 │ │Tree2│ │Tree3│ │Tree4│  ... different configs
   │Classic│ │Multi│ │Uni  │ │Deep │
   └───────┘ └────┘ └────┘ └────┘
        │       │       │       │
        └───────┴───────┴───────┘
                 │
         multiple Gemini LM replicas (TPUv4)
                 │
         single DDAR worker pool (shared across problems)
```

### 5.3 五种 search tree 类型

1. **Classic** (AG1-style)：每 node 1 个 aux point。
2. **Multi-aux**：LM 一次 propose 多个 aux points。trained-on-full-proof 的好处。有效加深 search depth。
3. **Uniform aux type**：强行 prompt `x00 a : cong`, `x00 a : coll`, `x00 a : cyclic`, `x00 a : perp` 等首 4 tokens，强制均匀分布，避免 LM 陷入 prior mode collapse。
4. **Deep-narrow**：beam=64, depth=10。
5. **Shallow-wide**：beam=512, depth=4。

### 5.4 System design 细节

- TPUv4，多 replicas per model；
- LM workers 和 DDAR workers **异步**；
- LM workers 写 node content 到 DB，DDAR workers 异步 pick up；
- DDAR workers 内部 self-balance workload；
- 多 problem 并行时，提前完成的 problem 释放 DDAR 资源给其他 problem。

**Intuition**: SKEST 本质上是 **population-based search + memory sharing**，类似 evolution strategies 或 PBT (Population Based Training, Jaderberg et al. 2017, https://arxiv.org/abs/1711.09846)。每个 tree 是一种 search heuristic 的"individual"，shared facts DB 是 population-level memory。这个设计也呼应 AlphaZero 的 self-play ensemble，但这里 ensemble 是 search policy 而非 RL policy。

---

## 6. Language Model（Section 6）

### 6.1 Architecture

- **Gemini-based** sparse mixture-of-experts Transformer (https://arxiv.org/abs/2403.05530)
- 多 size：从 ~million 到 billion 参数（Figure 5 标注如 "3p3B" = 3.3B params）
- 单阶段 unsupervised training on all data（简化 AG1 两阶段）
- LR schedule：linear warmup → cosine anneal，hyperparams 来自 scaling laws
- 数据规模：~300M theorems（数量级大于 AG1）

### 6.2 三种 training setup

1. **From scratch + custom tokenizer**（AG1 setup，baseline）
2. **Fine-tune math-specialized Gemini** in natural language（Appendix B）
3. **Multimodal from scratch** with diagram image input（Appendix C）

### 6.3 三个 evaluation sets

- `eval`：synthetic, with/without aux
- `eval_aux`：synthetic, aux-only
- `imo_eval`：IMO 2000–2024 AG-solvable subset

注意这些都是 **proxy metrics**，因为 perplexity 算全 proof，但 inference 只用 aux proposals。

### 6.4 Inference setup

- **Top-k sampling** with `t = 1.0`, `k = 32`
- Greedy decoding (`t=0, k=1`) 无 tree search：仅 2/26 with-aux 问题可解
- `t=1.0, k=32` 无 tree search：9/26
- 低 t 缺乏 diversity，高 t 产生 syntax error（Figure 6）

### 6.5 Analysis string（neuro-symbolic interface enrichment）

AG1: `<problem statement>`

AG2: `<problem statement> serialized(S1) serialized(S2−S1) serialized(S3−S2)`

其中：
- $S_1$ = DDAR 从原 premises 能推的所有 facts
- $S_2$ = DDAR 从原 premises + 假设 goal 为真 推出的所有 facts（counterfactual reasoning）
- $S_3$ = numerical diagram inspection 给出的所有（可能 true 但 DDAR 还没证的）facts
- 显然 $S_1 \subseteq S_2 \subseteq S_3$

**Intuition**: 这是一个非常漂亮的 "contextualized state injection"。$S_2 - S_1$ 告诉 LM "如果 goal 成立，那这些 facts 应该成立 —— 你能不能构造 aux point 让它们 reachable from premises?"。这模拟了人类数学家常用的"假定结论反推"技巧。$S_3 - S_2$ 提供 diagram-level hints，类似 human 用 GeoGebra 拖一下点观察 invariant。

---

## 7. Results（Section 7）

### 7.1 主表（Table 4）

| System | IMO-AG-50 solved | IMO-AG-30 solved |
|---|---|---|
| OpenAI o1 | 0 | 0 |
| Gemini thinking | 0 | 0 |
| AG1 DDAR only | 14 | 14 |
| AG2 DDAR only | 16 | 15 |
| TongGeometry DD | – | 18 |
| Average bronze medalist | 27.1 | 19.3 |
| Wu with AG1 DDAR (Sinha et al., 2024) | – | 21 |
| Average silver medalist | 33.9 | 22.9 |
| AG1 (full) | 27 | 25 |
| Average gold medalist | 40.9 | 25.9 |
| Wu + AG1 (Sinha et al., 2024) | – | 27 |
| TongGeometry w/o value | – | 28 |
| AG2 with AG1 setup (single tree) | 38 | 28 |
| TongGeometry full | – | 30 |
| **AG2 full (SKEST)** | **42** | **30** |

**值得注意的对照**：
- 纯 LM 系统（o1, Gemini thinking）在 IMO-AG-50 上 0 分 —— 这强有力地证明 **symbolic engine 不可替代** for long-horizon geometric reasoning。这也呼应 Stechly et al. 2025 (https://arxiv.org/abs/2402.14903 类似系列) 关于 LLM self-verification 局限。
- AG2 DDAR-only 比 AG1 DDAR-only 仅多 2 题，但 AG2 full 比 AG1 full 多 15 题 ⇒ **LM + search 才是主战场**。
- AG2 单 tree 已经 38，比 gold medalist 40.9 仅差 ~3 题。SKEST 再加 4 题 ⇒ **ensemble 的边际收益在递减但仍显著**。
- Wu's method (Sinha et al., 2024, https://arxiv.org/abs/2404.06405) + AG1 达 27/30，与 AG2 AG1-setup 28/30 接近 ⇒ algebraic bashing 和 synthetic reasoning 在 IMO-AG-30 上接近 ceiling，但 AG2 在 IMO-AG-50（更广义集合）上的优势源于 language 覆盖率。

### 7.2 Training curve（Figure 7）

仅 250 training steps（batch=256，~200M tokens），AG2 就能 solve 27/50 —— 已经超过 AG1。这说明 **架构升级 + 数据质量** 在 small scale 就见效。

### 7.3 Inference ablation（Figure 9）

最优 single-tree config：`beam_size=128, beam_depth=4, k=32 samples`。继续加 sample 或 beam size 不再提升 —— 典型 saturation 现象。

### 7.4 未解决问题分析

- **2 题 attempted but not solved**：IMO 2018 P6, IMO 2023 P6 —— 涉及 inversion, projective geometry, radical axis，DDAR 尚未实现这些 machinery。
- **6 题 unformalizable**：inequalities + variable number of points。

---

## 8. Featured Solutions（Appendix D）

### 8.1 IMO 2024 P4

构造 E 在 BI 上使得 ∠AEB = 90°，创造两对相似三角形 △ABE ~ △YBI 和 △ALE ~ △IPC。30 秒内得到 Joseph Myers（两届 IMO 金牌、2024 IMO 命题组主席）给的满分 7 分。

### 8.2 IMO 2013 P3

仅构造一个点 D = midpoint of arc $\widehat{ABC}$ containing B（**非对称**构造，非常 unnatural for humans），推出 B, A₁, D, I_a 共圆 ⇔ AB ⊥ AC。

### 8.3 IMO 2014 P3

构造 4 个 reflection points E, F, G, I（S 关于 OH, H 关于 AT, H 关于 AS, H 关于 ST 的反射）。AG2 证明了一个**更强的**结论 OH ⊥ BD，蕴含原命题。

### 8.4 IMOSL 2009 G7（最强 superhuman 例子）

题目：△XYZ equilateral ⇒ △ABC equilateral（其中 X, Y, Z 是 BIC, CIA, AIB 的 incenter）。

AG2 构造了 **11 个 auxiliary points**，几乎全是各种 circumcenter：

```
D       = circumcenter(BXC)
E       = circumcenter(AYZ)
X_1     = circumcenter(BIX)
X_2     = circumcenter(AIY)
X_3     = circumcenter(CIX)
X_4     = circumcenter(ABZ)
X_5     = circumcenter(ACY)
X_6     = circumcenter(AXZ)  [after showing A,C,X,Z cyclic]
X_7     = reflection of I w.r.t. BZ
X_8     = circumcenter(AXY)  [after showing A,B,X,Y cyclic]
X_9,X_10 = points making △IZX_9, △IZX_10 equilateral
X_11    = reflection of Z w.r.t. BI  [≡ X via double-point trick]
```

这是 AG2 "superhuman creativity" 的最有力证据 —— 人类几乎不会这样构造，但 angle/ratio chasing 在大量 circumcenter 之间自动 propagate 出答案。

**Intuition**: 这让我联想到 AlphaGo 的 Move 37 —— "creative" 其实是 search 在足够大 depth 下发现 human-prior 之外的 useful move。AG2 的 circumcenter bombing 策略本质是：circumcenter creates cyclic quadrilaterals ⇒ free equal-angle relations ⇒ AR engine 一次性 reduce。

---

## 9. Tokenizer & DSL 的重要性（Appendix B）—— 一个 surprising finding

### 9.1 Tokenizer ablation

- Custom word-level tokenizer（vocab ~thousands）
- Large LLM tokenizer（vocab 300k）

⇒ **IMO solve rate 完全相同**。这与 Singh & Strouse 2024 (https://arxiv.org/abs/2402.14903) 关于 tokenizer 是 LLM 算术瓶颈的假设相悖。

### 9.2 DSL vs natural language

把 AG2 数据全部翻译成自然语言（如 "Construct points d e f g such that a d g are collinear, ..."）训练同等规模模型 ⇒ **IMO solve rate 仍然相同**。

⇒ 这说明 **AG 数据的 combinatorial structure 远比表征形式重要**。打开了一条路：直接 fine-tune 大型 math-pretrained Gemini。

### 9.3 Fine-tune vs from scratch

- Fine-tune 3.3B math Gemini → 与 from-scratch 3.3B 在 IMO-AG-50 上**持平**（Figure 10）；
- 但两者 aux proposal 分布**slightly different**，在 SKEST ensemble 里能互补。

**Intuition**: 这是非常 deep 的发现。它意味着 AG 任务的 bottleneck 不在 "语言理解" 而在 "组合搜索的 prior"。一旦 prior 学到了，无论怎么 import 都行。这呼应了 "skill → policy" 的解耦。

---

## 10. Multimodal（Appendix C）—— 一个 negative result

训练带 diagram image 输入的 Gemini 模型。**单 model 不提升 solve rate**，但加入 SKEST ensemble 后由于 diversity 互补而轻微提升。

原因分析：
1. IMO diagrams 太 crowded，image tokenization 把图拆成 sequential patches 丢失 spatial info；
2. diagram 信息已部分由 sameclock 等 topology predicates 提供；
3. Vision-language models 在 atomic visual skills 上弱（Chae et al. 2024, https://openreview.net/forum?id=nFU4xCyoe0）；
4. Geometry solving 本质是 algebraic reasoning —— 复数法、barycentric、trig bashing 都不需要图。

**Intuition**: 这与 "geometry 是 visual task" 的 naive 直觉相悖。顶级 IMO 选手也很少真正"看图"，他们看的是 symbolic relations。这给未来 VLM for math 的方向敲了一个警钟。

---

## 11. Auto-formalization & Diagram Generation（Appendix G）

### 11.1 Auto-formalization

用 Gemini 写 few-shot prompt，5 次并行 query + 1 次 combine。对 IMO 2000–2024 44 道 formalizable 题成功 **33/44**。简单题几乎无错。这套系统在 **IMO 2025 P2** 上 20 秒内 end-to-end 解出（从自然语言输入）。

### 11.2 Automated diagram generation（non-constructive problems）

由于 AG2 允许 ≥3 predicates 定义 point，diagram 构造变成 nonlinear constraint solving。

**三阶段优化**：

1. **Initialization**（三种交替）：
   - Random normal distribution
   - Construction in order
   - Heuristically chosen order

2. **Adam gradient descent** on loss:
$$
\mathcal{L}(\bar{x}) = \sum_{c \in \text{exact}} f_c(\bar{x})^2 + \sum_{c \in C_<} \text{softplus}(g_c(\bar{x})) + \sum_{c \in C_\neq} \text{softplus}(\min(h_c(\bar{x}), -h_c(\bar{x}))) + \lambda_1 \|\bar{x}\|_2 + \sum_{A \neq B} \frac{\lambda_2}{|AB|^2 + \epsilon}
$$

   - $\bar{x} \in \mathbb{R}^{2n}$ 是所有点坐标的 flat vector；
   - $f_c(\bar{x}) = 0$：精确几何约束（cong, perp, etc.）；
   - $g_c(\bar{x}) < 0$：拓扑约束（如 same side）；
   - $h_c(\bar{x}) \neq 0$：不相等约束（如 noverlap）；
   - 后两项 non-degeneracy loss 防止点 collapse 或全部跑到原点。

3. **Gauss-Newton-Levenberg**：在 loss 收敛到 threshold 后切换，求解 under/over-determined nonlinear system 的精确 numerical solution。

**Benchmark**: 43/44 IMO problems within 1 hour，自动生成 diagram。

这与 Krueger et al. 2021 (https://arxiv.org/abs/2106.11328 类似 CAD 方向) 的方法相似但加了第三阶段解决 gradient descent 难以达到精度的问题。

---

## 12. LM 直接生成 full proof 的探查（Appendix F）

训练时 LM 学的是 full proof，但 inference 只用 aux proposals。Appendix F 测试：能否不要 symbolic engine？

工具：逐步骤验证 —— 把 step 的 premises 加入 fresh DDAR，跑 deduction closure with 该 step 用的 rule，看 conclusion 是否被证 + numerical check。

错误类别：
- Wrong grammar
- Theorem name error
- Step reference error
- Point not found
- Numerical error
- Unverified (premises 不蕴含 conclusion)
- Invalid aux point

**Finding**（Figure 16）：模型几乎不犯语法错；大多数 generated steps 要么 fully verified 要么 correct-but-unverified；小模型 vs 大模型 performance 相似。

⇒ 现代 LM 已具备 self-sufficient 生成 partial proof 的能力，但 hallucination 和长程 consistency 仍需 symbolic engine 兜底。

**Intuition**: 这与 Lightman et al. (PRM800K, https://arxiv.org/abs/2305.20050) 的发现一致 —— step-level correctness 是 achievable 的，但 full-trajectory correctness 仍 hard。AG2 把这个问题"作弊"绕过了：只让 LM 做 prior proposal，让 symbolic engine 做 verification。

---

## 13. Inequality Rules（Appendix H）

虽然未集成进 DDAR，但 Appendix H 给出了完整的 inequality rule set 草稿。核心定义：

- $P(A_1 A_2 \dots A_n)$：polygon
- $\omega(ABC) \in \{+1, -1\}$：orientation（clockwise / counter-clockwise）
- $\eta(ABC) \in \{+1, -1\}$：betweenness（B 在 A, C 之间 vs 同侧）
- $\angle_0(ABC) \in [0, \pi]$：传统无方向角
- $\angle_1(ABC) \in [0, \pi]$：AB 到 BC 逆时针
- $\angle_2(ABC) \in [-\pi, \pi]$：BA 到 BC，正表逆时针

关键公式：
$$
\angle_1(ABC) = \angle_2(ABC) + \frac{\pi(1 - \omega(ABC))}{2}
$$

这条公式把 orientation 直接 link 到 angle representation，是 inequality reasoning 的 algebraic backbone。

Rule H.8 几个经典：
- $\eta(ABC) = 1 \Rightarrow AC > AB$（B 在 AC 之间 ⇒ AC = AB + BC > AB）
- $AB + BC \geq AC$（triangle inequality）
- $AB \perp BC \Rightarrow AC > AB, AC > BC$

整个 Appendix H 像是一个"未来 DDAR3"的 specification —— 把 inequality, orientation, betweenness 形式化为 closure rules，从而让 symbolic engine 能处理剩下 12% 中的 inequality 类型题。参考 Li et al. 2025 (https://openreview.net/forum?id=FiyS0ecSm0) 在 olympiad inequalities 上的 LLM+symbolic 工作可作为补充。

---

## 14. 我的几个 cross-cutting intuitions

### 14.1 为什么 AG2 的 neuro-symbolic 比纯 LM 强这么多

AG2 的成功本质是 **labor division**：
- LM 是"intuition generator"：给出 aux point 的 prior 分布（高 entropy 但 high recall）；
- DDAR 是"verifier + closure computer"：低 entropy 但 high precision。

这两个 component 的 interface 是 aux point proposal。Analysis string 把 DDAR 的"已知/反推/数值"三层状态喂回 LM，是一个 **state-conditional policy** 的设计，本质上是把 LM 训练成 $\pi_\theta(\text{aux} \mid \text{problem}, \text{DDAR state})$。

这让我想起 AlphaGo 的 policy network + value network + MCTS —— AG2 的 DDAR 类似 MCTS 的 simulator，LM 类似 policy network。但 AG2 缺一个显式 value network —— SKEST 的 shared facts DB 算是一种 implicit value signal。

### 14.2 SKEST 与 Population-Based Methods

SKEST 的 multi-tree + knowledge sharing 在 RL 领域有 echoes：
- PBT (Jaderberg et al. 2017, https://arxiv.org/abs/1711.09846)
- Population-based PG (Q-prop, ACER 类)
- Multi-agent search in Hanabi (https://arxiv.org/abs/1907.03169)

把每个 search tree 当作一个 agent，shared facts DB 是 common pool resource。失败的 agent 把"已学到的"贡献给 pool，成功的 agent 终结整个 episode。这是 cooperation 而非 competition。

### 14.3 "Creative" 的来源

AG2 的 superhuman solutions（如 IMOSL 2009 G7 的 11 个 circumcenter）说明：在 combinatorial search 空间足够大 + verifier 足够快时，"creativity" 等价于 **non-obvious aux point sequences that pass verification**。人类选手的"intuition"是 high-precision low-recall prior，AG2 是 low-precision high-recall prior + fast verifier。两条路都能找到 proof，但 AG2 找到的是 human prior 之外的。

这跟 AlphaGo 的 Move 37 是同一类现象 —— "AI creativity" 来自 search + verification，不是来自"灵感"。

### 14.4 关于未来工作

Paper 明确指出两个方向：
1. **扩展 language**：variable number of points、non-linear equations、inequalities（Appendix H 已起草）。
2. **RL**：把当前 SKEST 当作 rollout policy，引入 RL fine-tune（类似 AlphaZero）。这能直接 attack 仍未解决的 IMO 2018 P6, 2023 P6（需 inversion / projective geometry 的"长程"aux constructions）。

Jha et al. 2024 (https://arxiv.org/abs/2405.16661) 的 RLSF (Reinforcement Learning via Symbolic Feedback) 是一个直接可借鉴的 framework。AG2 目前没做 RL，这是一个明显的 next step。

### 14.5 关于"language doesn't matter" finding 的深层含义

AG2 在 Appendix B 证明 custom tokenizer 和 DSL 都不关键。这暗示：**AG 任务的本质是 combinatorial search over geometric constructions**，表征层是"skin-deep"。这与 Chollet 的 ABSTRACT AND REASONING CORPUS (ARC) 哲学一致 —— 真正的 reasoning 通用性应该对 input/output 表征 invariance。

对 Karpathy 你可能特别 relevant：这呼应了你在 "Software 2.0" 里的观点 —— 一旦数据足够，"program"是 emergent 的，不需要 hand-designed DSL。但 AG2 又证明 symbolic verifier 不可缺，所以更准确的 framing 是 **"Software 1.5"**：neural 生成 + symbolic verify。

---

## 15. 主要参考链接汇总

- AG2 GitHub: https://github.com/google-deepmind/alphageometry2
- AG1 Nature paper: https://www.nature.com/articles/s41586-023-06747-5
- IMO 2024 silver blog: https://dpmd.ai/imo-silver
- IMO 2024 P4 full solution: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/imo-2024-solutions/P4/index.html
- Gemini 1.5 paper: https://arxiv.org/abs/2403.05530
- pybind11: https://github.com/pybind/pybind11
- TongGeometry: https://arxiv.org/abs/2412.10673
- Wu + AG1 (Sinha et al. 2024): https://arxiv.org/abs/2404.06405
- RLSF (Jha et al. 2024): https://arxiv.org/abs/2405.16661
- Tokenizer counts (Singh & Strouse 2024): https://arxiv.org/abs/2402.14903
- Autoformalization (Wu et al. 2022): https://arxiv.org/abs/2205.12615
- Olympiad inequalities (Li et al. 2025): https://openreview.net/forum?id=FiyS0ecSm0
- PRM800K (Lightman et al. 2024): https://arxiv.org/abs/2305.20050
- Autoformalizing Euclidean geometry (Murphy et al. 2024): https://openreview.net/forum?id=bylZbZOsGA
- Visual skills of VLMs (Chae et al. 2024): https://openreview.net/forum?id=nFU4xCyoe0
- PBT (Jaderberg et al. 2017): https://arxiv.org/abs/1711.09846
- Geometric reasoning gap in LLMs (Mouselinos et al. 2024): https://aclanthology.org/2024.findings-emnlp.360/

---

如果非要 pick 一个最 important 的 contribution，我会选 **SKEST + analysis string** —— 它把 neuro-symbolic 从"LM 提案 + engine 验证"升级成"LM 在 engine 状态条件下的 multi-agent 集成搜索"，这是 AG2 把 solve rate 从 38 推到 42、越过 gold medalist 阈值的真正 driver。其他改进（language、DDAR2、Gemini LM）都是必要的 enabler，但 SKEST 是 tipping point。
