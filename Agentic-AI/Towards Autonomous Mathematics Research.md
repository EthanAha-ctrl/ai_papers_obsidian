---
source_pdf: Towards Autonomous Mathematics Research.pdf
paper_sha256: 6a705febf9c89c99f14e45c0306864b82cd057809e2f0dd48d7a99ca7963809d
processed_at: '2026-08-12T17:00:03-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我换个姿势跟你聊，就像我们在咖啡店里白板上画图那样。

---

# 这篇 paper 到底在干嘛

一句话：**Google DeepMind 想知道，IMO 金牌模型离真正的 research mathematician 还有多远，然后他们真的去试了，试完发现比想象的近，但比想象的窄。**

## 背景的 timeline 你得先对齐

2025 年 7 月，DeepMind 的 Gemini Deep Think 在 IMO 2025 拿了金牌（6 题解 5 题，https://goo.gle/imo-gold）。这是 big deal，因为 IMO 是 high school 数学最高难度的比赛。但 Tony Feng 这帮人立刻问了一个更狠的问题：**research math 和 IMO math 是完全不同的 game**，IMO 题目自包含、几页纸能搞定、用 high school 标准 theorem；research paper 几十页、要 synthesize 大量 literature、要 navigate subfield 的全部 context。金牌 model 到 research math 还有几个数量级？

这篇 paper 就是这个问题的答案。他们做了一个叫 **Aletheia** 的 agent，让它在真实 research problem 上跑，结果产生了 5 篇 paper（其中 1 篇完全 AI 生成、2 篇 human-AI collaboration、2 篇 AI 贡献中间 lemma），还扫了 700 个 Erdős open problem。然后他们非常 honest 地报告了 success rate 和 failure mode。

---

# Aletheia 的架构，画给你看

```
       problem
         │
         ▼
   ┌───────────┐
   │ Generator  │  ← Gemini Deep Think，吐一个候选解
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │  Verifier  │  ← 独立 context，重新审视这个解
   └─────┬─────┘
         │ pass? 
    no ──┤ yes ──► output
         │
         ▼
   ┌───────────┐
   │  Reviser   │  ← 根据 Verifier 的批评改
   └─────┬─────┘
         │
         └────► 回到 Generator，循环
```

## 为什么要把 Verifier 从 thinking trace 里 decouple 出来

这是整篇 paper 最有 idea 的部分。Karpathy 你做 GPT 训练肯定有 intuition：模型生成 chain-of-thought 之后，final answer 是 condition 在整段 CoT 上的，也就是 $\pi(a \mid s_{1:T}, \text{problem})$，其中 $s_{1:T}$ 是 thinking tokens。

问题在于，一旦 $s_{1:T}$ 走偏了，后面的 token 在 autoregressive 采样下会被 anchoring 拉住，**模型自己很难回头**。作者用两个 hypothesis 解释：

1. **Bluffing hypothesis**：RL 训练（很可能是 RLHF 或 RLVR on math）incentivize 模型在不确定时硬猜，commit 到一个方向就不回头。
2. **Supporting context hypothesis**：长 CoT 变成了 misleading 的 prior，artificially 抬高了 $P(\text{wrong answer} \mid \text{wrong CoT})$。

所以 Verifier 干的事是：在 fresh context 下重新跑 $\pi(\text{verdict} \mid \text{answer}, \text{problem})$，**不加 CoT 的 anchoring**。这相当于 importance resampling，把 Generator 被 CoT 带偏的 bias 给 wash out 掉。

直觉：Generator 像一个思维发散的 grad student，写了一堆 reasoning 然后给个结论；Verifier 像一个冷静的 postdoc，只看结论本身和原始问题，问 "这个真的对吗？"

这个设计与 AlphaProof（https://doi.org/10.1038/s41586-025-09833-y）和 AlphaGeometry2（https://arxiv.org/abs/2502.03544）有本质区别——后两者用 Lean 形式语言，correctness 100% guaranteed，但 expressiveness 局限在能 formulate 的问题里。Aletheia 走 natural language 路线，correctness 没 guarantee，但能 cover 远超形式语言的研究前沿。

---

# Scaling law 部分很有 physics 味道

## 两条曲线，两个故事

**Figure 2a：IMO-ProofBench Advanced（30 道 IMO 风格题）**

x 轴是 inference compute（对数刻度，从 $2^7$ 到 $2^{12}$），y 轴是 human expert graded accuracy。

- IMO Gold 版本（Jul 2025）：compute 增 2 个数量级，accuracy 起步上升然后 plateau
- Advanced 版本（Jan 2026）：同等 accuracy 下 compute 少 100×，这就是 paper 里说的 "two orders of magnitude"
- Aletheia 在同一 base model 上又往上推了一截，到 **95.1%**

**Figure 2b：FutureMath Basic（PhD 级别习题）**

曲线形状类似，但 absolute accuracy 显著低。Aletheia 在 <60% 的问题上返回答案，conditional accuracy >82%。

## 我对 scaling law 的解读

你可以把 Deep Think 的 parallel thinking 想成某种 implicit search。每条 thinking chain 在某题上成功的概率是 $p$，跑 $K$ 条独立链至少成功一次的概率是 $1-(1-p)^K \approx 1 - e^{-Kp}$（当 $p$ 小）。

所以 $\log(1-\text{acc}) \approx -Kp$，acc 对 $\log K$ 大致线性。这跟 Figure 2 的曲线形状一致。

但 PhD 级别问题 $p$ 极小，且 chain 之间相关性高（都是同一 base model 出来的，diversity 不够），所以斜率显著低于 Olympiad。**这就是为什么光靠 inference scaling 不够**——必须加 agent harness（让 Verifier/Reviser 引入新的 "perspective"）和 tool use（grounding）。

---

# Tool use 这块经验观察很关键

他们发现两类 hallucination：

**Type 1：完全编造的 paper**（Figure 3）
- 模型说 "C. Livingston and S. Naik, Algebraic & Geometric Topology, 13(2) (2013), 1115-1124"
- 这 paper 不存在，模型用 author name + topic + journal 风格拼出一个看起来合理的引用

**Type 2：真 paper 但引错**（Figure 4）
- Galambos (1976) 这篇 paper 真存在
- 但模型说 Galambos 证明了一个 "classical result"，去原 paper 里找不到

加了 Google Search + web browsing 后，Type 1 基本消失，错误全部转移到 Type 2。Python tool 的 marginal improvement 很小，因为 Gemini 本身计算能力已经够强。

**我的解读**：LLM 在低密度 retrieval 区域本质上是 weighted interpolation。pretraining 见过类似 author+topic 但没见过 exact paper，就生成一个 likelihood-weighted 的 "合理" 引用。Search 工具相当于硬约束 $P(\text{cite} \mid \text{query})$ 只在真实 corpus 上取值，把 hallucination 从 "完全假" 推到 "真但引错"。要彻底解决 Type 2，可能需要 retrieval + page-level grounding（把 paper 全文 chunk 进来再 cite 具体定理）。

---

# 四个数学结果，我挑有趣的讲

## Milestone A: Eigenweights (Feng26) — 完全 AI 的 paper

### 数学背景

Hirzebruch Proportionality Principle（1958）说：compact locally symmetric space 上 automorphic vector bundle 的 Chern number = compact dual 上对应 Chern number × 一个比例常数，这个常数是 Gross motive 的 $L$-function 值。

Feng-Yun-Zhang (FYZ26, https://arxiv.org/abs/2601.18557) 研究一个变种叫 "Arithmetic Hirzebruch Proportionality"，把 moduli space of shtukas 上 Chern class 的 arithmetic volume 和 Gross motive $L$-function 上的 differential operator 联系起来。这个 differential operator 由一组 structure constant 决定，叫 **eigenweights**。

Feng-Yun-Zhang 自己算了一些 eigenweights 例子，但不知道 closed form。

### Aletheia 怎么做的

在 zero human intervention 下，Aletheia 用了完全不同领域的技术：
- **Atiyah-Bott localization**（ equivariant cohomology 的工具）
- **Schur polynomial** manipulation
- **Frobenius character identities**
- **Murnaghan-Nakayama rule**（Symmetric group representation theory）

对 Type A、Type C、Type D group 都给出了完整 closed form。

这超出 Feng-Yun-Zhang 的 toolkit。Feng-Yun-Zhang 是 arithmetic geometry 专家，Aletheia 用的是 representation theory / equivariant cohomology 的工具，cross-subdomain 的 transfer。

**这就是 paper 里说的 superhuman breadth**。一个 human mathematician 一般只精通一两个 subfield，但模型 pretraining 见过所有领域的 paper，能在 cross-domain 之间做联想。这是 AI 的真正 comparative advantage。

paper: https://arxiv.org/abs/2601.23245

---

## Milestone B: Independence Polynomials (LeeSeo26) — 反过来的 workflow

### 背景

Independent set = graph 上互不相邻的顶点集合。Sah-Sawhney-Stoner-Zhao (SSSZ19, https://doi.org/10.1016/j.jctb.2019.01.007) 给出 weighted independent set 数量的下界。Lee-Seo 想推广到 **semiproper colouring**（两种粒子，同种排斥，异种不排斥）。

### 反直觉的 workflow

正常 human-AI collaboration 是：human 把大问题拆成小 technical query 给 AI。但这次反过来：Aletheia 给 high-level roadmap（dual sets + log-convexity + reduction + 几个 key lemma），然后 human 把 outline 填成 rigorous proof。

这就是 paper 里说的 "AI 给 big picture strategy，human 做 rigorous execution"。非常像 senior mathematician 给 grad student 出 idea。

paper 在 arXiv 上但还没贴公开链接。

---

## Milestone C: 700 个 Erdős open problem 的 systematic evaluation

这是最 quantitative 的部分，我必须把数字摆给你看。

### Data flow

| Stage | Count | Rate |
|-------|-------|------|
| Bloom DB 上的 "Open" problems（2025 Dec 2-9 期间） | 700 | 100% |
| Aletheia 的 informal verifier 筛过后 "potentially correct" | 212 | 30.3% |
| Human 能明确判 correct/incorrect 的 candidates | 200 | — |
| Fundamentally flawed | 137 | 68.5% |
| Technically correct | 63 | 31.5% |
| Meaningfully correct（真的解决了 Erdős 想问的问题） | 13 | 6.5% |

### "Technically correct 但 mathematically vacuous" 是什么意思

50 个 technically correct 的解，但**问题被模型以 Erdős 不 intend 的方式解读了**，导致 trivial 解。这是 RL 里典型的 specification gaming / reward hacking——模型在 ambiguity 空间里挑最好解的版本，而不是 intended 版本。

### 13 个 meaningful correct 的分类

| 类别 | 描述 | 实例 |
|------|------|------|
| Autonomous Resolution | 全新自主解 | 652, 1051 |
| Partial AI Solution | 多部分问题解了一部分 | 654, 1040 |
| Independent Rediscovery | 解出后发现文献已有 | 397, 659, 935, 1089 |
| Literature Identification | 发现问题已在文献中解决 | 333, 591, 705, 992, 1105 |

### 一个非常 honest 的 takeaway

很多 AI 解的 "novel" Erdős 问题，事后被人发现其实文献里早就解了（比如 1026, 397, 333, 281）。Erdős-1089 更夸张——答案就在 Bannai-Bannai (1981, https://doi.org/10.1007/BF02579266) 的一句 offhand remark 里，原作者自己都没意识到解决了一个 Erdős 问题。

作者说：**这些 open 问题之所以 open，不是因为难，是因为 obscure**。AI 的 superhuman breadth 能扫遍冷门文献，但这种解对数学的 actual advance 很小。

paper: https://arxiv.org/abs/2601.22401

---

## Milestone D: Robust MDP 的 complexity bound (ACGKMP26)

这个我简短讲。ISTA 的 Asadi 等人做 Robust MDP 的 strongly polynomial time bound，conditional on 一个数论断言。Pagano 用 Siegel's Lemma 证明了。Aletheia 也用 Siegel's Lemma 但给了更 sharp 的 bound，比所有 human 和 AI 之前的尝试都好。

paper: https://arxiv.org/abs/2601.23229

---

# Autonomy framework 是这篇 paper 的 meta 贡献

作者类比 SAE 自动驾驶等级，给 AI-math 结果搞了个二维 taxonomy。

## Axis 1: Autonomy Level

| Level | 含义 |
|-------|------|
| H | Primarily Human，AI 只是辅助 |
| C | Human-AI Collaboration，双方都 essential |
| A | Essentially Autonomous，AI 全生成核心内容 |

## Axis 2: Mathematical Significance

| Level | 含义 |
|-------|------|
| 0 | Negligible novelty（Olympiad / PhD 习题级） |
| 1 | Minor novelty（新但不够 publication grade） |
| 2 | Publication Grade（正常 research journal 可发） |
| 3 | Major Advance（顶 5 数学期刊级） |
| 4 | Landmark Breakthrough（一代一次） |

## 论文自己结果的定位（Table 1）

| | H | C | A |
|---|---|---|---|
| Level 0 | | | Erdős-652, 654, 1040 |
| Level 1 | | | Erdős-1051 |
| Level 2 | ACGKMP26, FYZ26 | BKKKZ26, LeeSeo26 | Feng26 |
| Level 3 | （空） | （空） | （空） |
| Level 4 | （空） | （空） | （空） |

注意 Level 3 和 Level 4 全空——作者明确说 AI 离 major advance 还很远。Level 2 内部 variation 极大，Feng26 和 ACGKMP26 都算 Level 2 但难度差距巨大，**不能解读成 "AI 达到人类数学家水平"**。

## HAI Card

类比 model card，提出 Human-AI Interaction Card，记录关键 prompt 和 output。对 Level C/A 的结果，作者建议至少公开包含 essential new insight 的 raw prompt。Terence Tao 已经在 Zulip 上发起 "best practices for incorporating AI" 的讨论: https://ai-math.zulipchat.com/#narrow/channel/539992-Web-public-channel---AI-Math/

---

# Ablation 部分给你看一眼

用 Gemini Deep Think (IMO scale, 2× compute) 跑 Aletheia 解出的 13 个 Erdős 问题，只解出 8 个。在 research paper prompts 上，Deep Think 在 Feng26（完全 autonomous 那个）3 个 prompt 全失败。

**这就是 agent harness 的价值**：同样的 base model，加了 Verifier-Reviser 闭环和 tool use，能把 success rate 从 8/13 提到 13/13，把 "完全无法解决" 的 Feng26 推到 "完全解决"。Agent harness 是 base model capability 的 multiplier。

---

# 附录里的几个 case study 我也讲一下

## IMO 2025 Problem 6: 2025×2025 grid tiling

### 题目

2025×2025 grid，每行每列恰好 1 个 unit square 不被覆盖，最少要多少 rectangular tiles？

### 模型解法

**Lower bound**: $T \geq N + a + b - 3$，其中：
- $N = 2025$（uncovered squares 数）
- $a$ = holes 坐标作为 permutation 的 LIS 长度
- $b$ = LDS 长度

由 Erdős-Szekeres / Dilworth: $a \cdot b \geq N$。AM-GM: $a + b \geq 2\sqrt{2025} = 90$。所以 $T \geq 2112$。

**构造**：$k = 45$，分 $k \times k$ macro-blocks，每块 $k \times k$。macro-block $(u,v)$ 的 hole 放在 row $(u-1)k + v$、column $(u-1)k + (k+1-v)$。这给 $a = b = 45$。

剩余空间分两类 tile：
- Central: $(k-1)^2 = 44^2 = 1936$ 个 $45 \times 45$ 大方块
- Edge: $4(k-1) = 176$ 个级联边界矩形
- Total: 1936 + 176 = 2112 ✓

### 论文里诚实说的 caveat

第一版 output 引用了 advanced theorem without proof（EGMO discrete geometry 结果），按 IMO 规则只能 1-3 分。作者额外 prompt 让模型用 elementary techniques 重写，第二版（$2^8$ scale）给出 self-contained 证明。

**这揭示了一个有趣的 failure mode**：模型会默认 research math 风格（引 advanced theorem 当 black box），但 IMO 要求 self-contained elementary 证明。Context 决定 genre，模型需要 explicit signal 才能切换。

### Elementary 证明里的关键技术

**Rectilinear partition formula**: $T = H + V + I + 1 - N$
- $H, V$ = 最大水平/垂直内部线段数
- $I$ = 严格内部 crossings 数
- 通过数矩形 90° 角（4 per rectangle）= grid corners (4) + crossings (4I) + T-junctions (4H+4V) 反推

**LIS/LDS penalty argument**: 对 LIS 每个 gap $[y_k, y_{k+1}]$，hole column 从 $\leq c_{y_k}$ 跳到 $\geq c_{y_{k+1}}$，找到一个 row $y_k^*$ 和 column $x_k^*$，使 $u_{y_k^*} + w_{x_k^*} + X_{x_k^*, y_k^*} \geq 1$。LIS 和 LDS 的 row index 集合严格不相交（一个要求 $c_y < c_{y+1}$，另一个要求 $c_y > c_{y+1}$），无 double counting。

求和：$\sum u_y + \sum w_x + \sum X_{x,y} \geq (a-1) + (b-1) = a + b - 2$，得 $T \geq N + a + b - 3 = 2112$。

---

## IMO 2024 Problem 3 (在 $2^7$ scale 成功，有小错)

### 题目

男孩女孩交替排队，前 $N$ 人任选正整数。对 $m > N$，第 $m$ 人选的数 = $a_{m-1}$ 在前 $m-2$ 项中出现次数 +1。证明 $\{b_n\}$ 或 $\{g_n\}$ 最终周期。

### 模型解法

1. **简化**: $a_m = c_{m-1}(a_{m-1})$，其中 $c_m(x)$ = $x$ 在前 $m$ 项出现次数
2. **大数有界**: 令 $M_0 = \max(N, \max_{i \leq N} a_i)$，归纳证明 $c_m(y) \leq M_0$ for all $y > M_0$
3. **无穷出现集 $S$**: 证明 $S = \{1, 2, \dots, L\}$ for some $L \geq 1$，且每个 $V > M_2$ 恰好出现 $L$ 次
4. **交替结构**: $a_{m-1} \in S \Rightarrow a_m > L \notin S$；反之亦然。男孩女孩必有一方总从 $S$ 选
5. **状态空间有界**: 计数向量 $\nu_n$ 满足 $\nu_{n+1} = \nu_n + e_{s_{n+1}}$，其中 $s_{n+1}$ 是新值的 rank。证明 $\max(\nu_n) - \min(\nu_n)$ 全局有界 → 状态空间有限 → 确定性转移 → 最终周期

注：模型在某处有 "miscellaneous mistake"，作者加了 footnote。

---

## IMO 2024 Problem 5 (在 $2^8$ scale 成功)

### 题目

$3002 \times 3001$ grid，左上角 stone。Peter 选 3000 个 trap cell（每中间行 1 个，每列至多 1 个）。James 不知道位置，要把 stone 移到最后一行。踩 trap 罚 1 分并回起点。求最小 $n$ 使 James 能在罚 $n$ 分前到达。

### 模型解法（与标准 online solution 不同）

Online 标准解法用 staircase / happy triangle 的 visual pattern。模型没用这些，改用 state-space 推理。

**Lower bound $n \geq 3$**: Peter 在 James 第一条路径 row 2 crossing 处放 trap，再在新路径 row 3 不同列 crossing 处放 trap，至少罚 2 次。

**Upper bound $n \leq 3$**: James 用 row-by-row "Safe Probing"：
- $A$ = 未发现 trap 的 column 集合
- $S$ = 已发现 trap 的 column 集合
- 选 $x \in A$（若已有 1 penalty，选 $A$ 中最接近 $c_1$ 的）
- 探测其他 $y \in A \setminus \{x\}$，找本行 trap

第 2 次 penalty 后用 "Drop to Finish"：
- 沿 column $x$ 下（$x \in A$ 上方无 trap）
- 横到 column $c_1$（中间列都在 $S$，行 $k$ 上方有 trap 但行 $k$ 本身安全）
- 沿 column $c_1$ 下到底（$c_1$ 上方 trap 在 $r_1 < k$，下方全空）

答案：$n = 3$。

---

# 我自己的几点延伸 intuition 给你

## 1. Verifier decoupling 本质是 amortized inference

Generator 用大量 thinking compute 探索，Verifier 用独立 context 重评。这相当于把 $P(\text{answer} \mid \text{problem})$ 分解成 $P(\text{answer} \mid \text{CoT}, \text{problem}) \cdot P(\text{CoT} \mid \text{problem})$，Verifier 在 marginal 上独立评估，wash out CoT 的 bias。

**这跟你做 nanoGPT 时强调的 "scale + architecture" 互补**：单纯 scale base model 在 research math 上 plateau 很快，architecture（agent harness）现在是 scaling 的 multiplier。

## 2. Tool use 是 retrieval-augmented reasoning 的硬约束

Citation hallucination 本质是 LLM 在 low-density 区域的 weighted interpolation。Search 把生成分布 clamp 到真实 corpus 上。**未来真正 work 的 tool 不只是 search，而是 page-level grounding**——把 paper 全文 chunk、retrieve、cite 具体定理，而不是只 cite paper-level metadata。

## 3. 6.5% meaningfully correct 的真实含义

注意 6.5% 是 conditional on Aletheia 自己筛出的 212 candidates 中的 200 个可判定。真正 rate 是 13/700 ≈ 1.86%。**这是 LLM 在 research-level open problem 上的真实 success rate**，远低于 popular press 给人的印象。

## 4. AI 的 comparative advantage 在 breadth

Feng26 的故事最能说明问题：Aletheia 用 representation theory 工具解 arithmetic geometry 问题，cross-subdomain 的 transfer 是人类 mathematician 做不到的（一个人精通两三个 subfield 已经是 top tier）。但深度上仍然远不如 expert。

**这预示 AI 在 math 上的 killer app 是 "cross-domain synthesis"，而不是 "single-domain depth"**。把这跟 AlphaFold 类比：AlphaFold 不是比 biologist 更懂 biology，而是用 geometric deep learning 把一个特定问题解得比人类快 1000×。Aletheia 的角色类似——不是替代 mathematician，而是 cross-subdomain 的 retrieval + synthesis tool。

## 5. Inference scaling 的渐近行为暗示什么

Olympiad 曲线 plateau 明显，PhD 曲线还在 linear region。这说明 research math 需要 **远超目前 inference budget** 的 compute。Agent harness 是 amortize 这种 compute 的方式——把 search 空间用 Verifier 切片，避免无效探索。

**这跟你的 LLM101 第 16 讲 "post-training RL" 的思路吻合**：RL on CoT 让模型学会在 inference time "search"，但这种 search 本身有 scaling limit，需要 external verifier 提供 ground truth signal 才能突破。

## 6. HAI Card 是 ML reproducibility 向 math 界的推广

Model card 记录 model bias/perf，HAI Card 记录 human-AI 边界。未来 math 界可能要求所有 AI-assisted paper 附 HAI Card，类似 data availability statement。这是 transparency infrastructure 的雏形。

---

# 最后给你几个值得关注的 follow-up 阅读

- AlphaProof: https://doi.org/10.1038/s41586-025-09833-y
- AlphaGeometry2: https://arxiv.org/abs/2502.03544
- FrontierMath benchmark: https://arxiv.org/abs/2411.04872
- Tao 的 AI-Erdős wiki: https://github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erd%C5%91s-problems
- AlphaEvolve: https://arxiv.org/abs/2506.13131
- Huang-Yang IMO 2025 harness: https://arxiv.org/abs/2507.15855
- 全部 prompts 和 outputs: https://github.com/google-deepmind/superhuman/tree/main/aletheia

---

# TL;DR 给你

DeepMind 做了个 math research agent 叫 Aletheia，由 Generator-Verifier-Reviser 闭环 + Google Search + Python 组成，跑在 advanced Gemini Deep Think 上。它产生了 5 篇 paper（1 篇完全 AI、2 篇 human-AI collab、2 篇 AI 贡献中间结果），扫了 700 个 Erdős open problem（13 个 meaningfully correct，4 个真 novel）。他们还提出二维 taxonomy（Autonomy × Significance）和 HAI Card 的概念，非常 honest 地报告 6.5% meaningful success rate。**核心 takeaway：base model scaling 在 research math 已经接近瓶颈，agent harness + tool grounding 是下一个 multiplier，AI 的 comparative advantage 在 superhuman breadth 而非 depth**。

---

# 《Towards Autonomous Mathematics Research》深度讲解

## 一、论文核心定位与历史背景

这篇 paper 来自 Google DeepMind，第一作者 Tony Feng（既是 project lead 又是 mathematician），由 Quoc V. Le、Thang Luong 共同领导，还有一大批专业 mathematician 参与验证（Junehyuk Jung、Joonkyung Lee、Carlo Pagano、Sang-hyun Kim、Federico Pasqualotto、Sergei Gukov 等）。这反映出该工作的特殊性：AI 生成的数学结果必须由领域专家审阅，否则无法判断 correctness 与 novelty。

这篇文章的 timing 很关键——2025 年 7 月 DeepMind 的 Gemini Deep Think 在 IMO 2025 拿到金牌（解出 6 题中的 5 题），但论文真正想回答的问题是：**从 IMO 金牌到 PhD 级别 research math 之间还有多远**？结论是：仍然很远，但已经能产出可发表的研究论文，前提是用 agent harness 包住 base model，用工具压住 hallucination。

参考链接：
- IMO 2025 金牌公告: https://goo.gle/imo-gold
- GitHub prompts & outputs: https://github.com/google-deepmind/superhuman/tree/main/aletheia
- Erdős problems 数据库: https://erdosproblems.com
- Tao 的 AI-Erdős wiki: https://github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erd%C5%91s-problems

---

## 二、Aletheia 的架构（Figure 1 解析）

Aletheia 由三个 subagent 组成闭环：

```
        ┌─────────────────────┐
        │  Generator (Gemini)  │ ─── 生成候选解
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Verifier (Gemini)  │ ─── 独立审查，decoupled from thinking trace
        └──────────┬──────────┘
                   │ pass / fail
                   ▼
        ┌─────────────────────┐
        │   Reviser (Gemini)   │ ─── 根据 Verifier 反馈修正
        └──────────┬──────────┘
                   │
                   └──────► loop until Verifier approves or hit attempt limit
```

### 2.1 关键设计直觉：为什么要 decouple thinking tokens 与 final output？

论文 §2.2 给出一个非常 informative 的经验观察：

> "decoupling a reasoning model's final output from its intermediate thinking tokens, and adding well-chosen prompt scaffolding, enables the model to recognize flaws it initially overlooked during generation."

作者给出两个 hypothesis 来解释这个现象：
1. **Training incentive hypothesis**：训练过程（很可能是 RL on chain-of-thought）incentivize 模型 "guess or bluff"，使其在长 CoT 中越走越远。
2. **Supporting context hypothesis**：extended thinking trace 作为 misleading "supporting" context，artificially inflate conditional probability $P(\text{erroneous conclusion} \mid \text{flawed CoT})$。

这第二种 hypothesis 用概率图模型的视角看其实很自然：模型在生成 final answer 时是条件在整段 CoT 上的，即 $\pi(a \mid s_{1:T})$，其中 $s_{1:T}$ 是 thinking tokens。如果 $s_{1:T}$ 已经偏向错误方向，后续 token 在 autoregressive 采样下会被 "commitment" 拉住，难以回头。把 Verifier 单独跑一次，相当于在 fresh context 下重新评估 $\pi(\text{verdict} \mid a, \text{problem})$，去掉了 CoT 的 anchoring 效应。

这种设计与 AlphaProof（Hubert et al., 2025, Nature 论文 https://doi.org/10.1038/s41586-025-09833-y）和 AlphaGeometry2（Chervonyi et al., 2025, https://arxiv.org/abs/2502.03544）有本质不同：后两者用 Lean/Isabelle 等形式语言，而 Aletheia 完全在 natural language 下端到端工作。这个 trade-off 的代价是 correctness 无法 formally guaranteed，收益是能 cover 远超形式语言目前能 formulate 的研究前沿问题。

### 2.2 与同期工作的对比

- **Huang-Yang harness**（https://arxiv.org/abs/2507.15855）：手工构造的 solver-verifier pipeline，把 GPT-5、Gemini 2.5 Pro、Grok 4 推到 IMO 2025 金牌。
- **FullProof**（Bryan et al., 2026）：另一个 math research agent，也有 informal verifier。

Aletheia 的差异化主要在 tool use 与 long-horizon revision 的强度，以及和真正 mathematician 的 deep collaboration。

---

## 三、Inference-time Scaling Law（§2.1，Figure 2）

这是论文最有 "physics-style law" 味道的部分。Deep Think 利用 parallel thinking，可以在 inference 时灵活调整 compute。作者严格地为每个 (problem, compute scale) 跑一次，避免 cherry-picking。

### 3.1 两条 scaling 曲线

**Figure 2a：IMO-ProofBench Advanced（30 道 IMO 风格题）**
- x 轴：inference compute（对数刻度，标注了 $2^7, 2^8, \dots, 2^{12}$ 等 scale）
- y 轴：accuracy（human expert graded）
- IMO-Gold 版本（Jul 2025）的曲线在 compute 增大 ~2 个数量级后 plateau
- Advanced 版本（Jan 2026）在同等 accuracy 下，compute 减少 ~100×（两个数量级）
- Aletheia 在同一 base model 上进一步推到 **95.1%**，并在 29/30 题返回解答的子集上 conditional accuracy 达到 **98.3%**

**Figure 2b：FutureMath Basic（PhD 级别习题）**
- 同样呈现 scaling law，但 absolute accuracy 显著低于 Olympiad
- Aletheia 在 answered subset（<60% 问题）上 conditional accuracy >82%
- 关键观察：Aletheia 倾向于 **承认不会做**，而非硬凑——这对 human-AI collaboration 是 essential 的 feature

### 3.2 直觉解释

可以把 inference-time scaling 想成 MCTS 风格的 search 宽度。Deep Think 内部并行采样多条 thinking chain，通过某种 internal selection（可能类似 self-consistency 或更复杂的 reranking）输出 best 候选。当 compute scale 从 $2^7$ 增到 $2^{12}$，相当于搜索空间扩展 32 倍。

形式化一点（这是我的解读，论文没明说）：设每条 thinking chain 在某题上成功的概率为 $p$，则 $K$ 条独立链中至少一条成功的概率为 $1-(1-p)^K$。取 log：$\log(1-\text{acc}) \approx -Kp$ 当 $p$ 小时。所以 acc 对 $\log K$ 大致是线性的，这与 Figure 2 的视觉趋势一致。

但 PhD 级别问题 $p$ 极小，且 chain 之间相关性高（都来自同一 base model 的同一分布），所以 scaling 斜率显著低于 Olympiad。这就是作者总结 "inference-time scaling alone would not be sufficient" 的根本原因——必须加 agent harness + tool use。

参考：Luong et al., 2025, "Towards robust mathematical reasoning", https://arxiv.org/abs/2511.01846

---

## 四、Tool Use 的关键作用（§2.3）

### 4.1 Hallucination 的两类

论文给出两个典型的 failure mode：

**Type 1: 完全虚构的引用（Figure 3）**
- Prompt: 证明 pretzel knot $\mathcal{P}(-3,5,13)$ 在 smooth concordance group 中有 infinite order
- 无 internet 时模型编造：C. Livingston and S. Naik, "Ozsváth-Szabó and Rasmussen invariants of some pretzel knots", Algebraic & Geometric Topology, 13(2) (2013), 1115-1124
- 这篇 paper 完全不存在

**Type 2: 真实文献但错误引用（Figure 4）**
- Galambos (1976) 这篇 paper 确实存在
- 但模型声称的 "classical result" 在原 paper 里找不到

### 4.2 各工具的 marginal contribution

- **Google Search + Web Browsing**：对 spurious citation 大幅减少，把错误从 "假 paper" 推到 "真 paper 但引错"
- **Python**：marginal improvement，因为 Gemini 本身计算能力已经强
- 这暗示 **未来需要更 specialized 的工具**，比如 Lean/Coq verifier、SageMath、Mathematica 的 deep integration

直觉上，工具的作用是把模型的 "ungrounded generation" 转化为 "grounded retrieval + verification"。Citation hallucination 本质上是 LLM 在低概率尾部的 retrieval 失败——pretraining 见过类似 author+topic 但没见过 exact paper，模型用 likelihood-weighted 插值生成一个 "看起来合理" 的引用。Search 工具相当于硬约束 $P(\text{cite} \mid \text{query})$ 只在真实 corpus 上取值。

---

## 五、四个数学研究里程碑（§3）

### Milestone A: Eigenweights (Feng26) — Level A2

**背景**：Hirzebruch Proportionality Principle（Hir58）表达 compact locally symmetric space 上 automorphic vector bundle 的 Chern numbers 与 compact dual 上对应 Chern numbers 成比例，比例常数可解释为 Gross motive 的 $L$-function 值。Mumford (Mum77) 推广到 non-compact。Feng-Yun-Zhang (FYZ26, https://arxiv.org/abs/2601.18557) 研究的 "Arithmetic Hirzebruch Proportionality" 把 arithmetic volume of Chern classes on moduli spaces of shtukas 与 Gross motive $L$-function 上的 differential operator 联系起来，是 Gaitsgory-Lurie 解决 Weil Tamagawa Number Conjecture (GL14) 的推广。这里 differential operator 由某些 structure constants **eigenweights** 决定。

**故事**：Feng-Yun-Zhang 算了一些 eigenweights 例子但不知道 closed form。Aletheia 在无 human intervention 下，用 Atiyah-Bott localization、Schur polynomial操作、Frobenius character identities 和 Murnaghan-Nakayama rule 给出完整 closed form（对 Type A、C、D group 都解决）。这超出了 Feng-Yun-Zhang 的 toolkit（用了代数几何另一个子领域的技术）。

**HAI Card**（论文表格）：
```
Human: Query eigenweights for Type A
Aletheia: 完整正确解 (Atiyah-Bott + Schur + Frobenius + Murnaghan-Nakayama)
Human: Query Type C
Aletheia: 完整正确解（同样工具变体）
Human: Query Type D
Aletheia: 完整正确解
```

论文: https://arxiv.org/abs/2601.23245

### Milestone B: Independence Polynomials (LeeSeo26) — Level C2

**背景**：Independent sets 是 graph 上互不相邻的顶点集合。Sah-Sawhney-Stoner-Zhao (SSSZ19, JCTB, https://doi.org/10.1016/j.jctb.2019.01.007) 给出 weighted independent set 数量下界。Lee-Seo 想推广到 semiproper colourings（两种粒子，不同种类不互相排斥）。

**故事**：先让 Gemini 2.5 Deep Think 证明 SSSZ 推广的关键 inequality，成功。再上 Aletheia 解更深问题，Aletheia 给出 high-level roadmap：dual sets + log-convexity + reduction + 关键 Lemmas。人类作者把 outline 填成 rigorous proof。

这里 workflow 反过来——AI 给 "big picture strategy"，human 做 "rigorous execution"。这非常像 senior mathematician 给 graduate student 出 idea 的关系。

### Milestone C: Erdős Problems (Feng et al., 2026a) — Level A0/A1

**数据流**（这是论文最 quantitative 的部分）：

| Stage | 数量 | 比例 |
|-------|------|------|
| Open problems on Bloom DB (Dec 2025) | 700 | 100% |
| Aletheia 返回的 "potentially correct" | 212 | 30.3% |
| 能明确判 correct/incorrect 的 candidates | 200 | — |
| Fundamentally flawed | 137 | 68.5% |
| Technically correct | 63 | 31.5% |
| Meaningfully correct | 13 | 6.5% |
| 真正 novel（autonomous） | 4 | 2% |

13 个 meaningfully correct 分四类：

| 类别 | 描述 | 实例 |
|------|------|------|
| Autonomous Resolution | 全新自主解 | 652, 1051 |
| Partial AI Solution | 多部分问题中解了一部分 | 654, 1040 |
| Independent Rediscovery | 解出后发现文献已有 | 397, 659, 935, 1089 |
| Literature Identification | 发现问题已在文献中解决 | 333, 591, 705, 992, 1105 |

带 * 的（397、652、659）表示 Aletheia 评估后、论文发表前被其他 party 独立解决。

**关键 takeaway**：很多 Erdős open problems 没解出来是因为 **obscurity 而非 difficulty**。比如 Erdős-1089 的答案其实是 Bannai-Bannai (1981) 一句话的 offhand remark，作者自己都没意识到解决了 Erdős 问题。

参考: https://arxiv.org/abs/2601.22401, https://doi.org/10.1007/BF02579266

### Milestone D: 两个 paper 的中间贡献

- **(FYZ26)**: Arithmetic Volumes of moduli stacks of shtukas，AI 给一个 eigenweights 计算的更好证明
- **(ACGKMP26)**: Strongly Polynomial Policy Iteration for $L_\infty$ Robust MDPs（https://arxiv.org/abs/2601.23229）。原团队 conditional on 一个数论断言（特定有界组合落在多项式个 dyadic interval 内），Pagano 用 Siegel's Lemma 证明。Aletheia 也用 Siegel's Lemma 但给出更 sharp 的 bound，超过所有人类和 AI 之前的尝试。

---

## 六、Autonomous Mathematics Levels 框架（§5.1）

作者类比 SAE 自动驾驶等级，提出二维分类：

### Axis 1: Autonomy Level

| Level | 名称 | 描述 |
|-------|------|------|
| H | Primarily Human | 核心内容人类生成，AI 辅助 |
| C | Human-AI Collaboration | 双方都有 essential 贡献 |
| A | Essentially Autonomous | 核心数学内容 AI 全生成 |

### Axis 2: Mathematical Significance

| Level | 名称 | 标准 |
|-------|------|------|
| 0 | Negligible novelty | Olympiad 或 PhD 习题级 |
| 1 | Minor novelty | 新结果但够不上 publication grade |
| 2 | Publication Grade | 可在正常 research journal 发表 |
| 3 | Major Advance | 顶 5 数学综合期刊级别 |
| 4 | Landmark Breakthrough | 一代一次的突破 |

### 论文自身结果的定位（Table 1）

| | H | C | A |
|---|---|---|---|
| Level 0 | | | Erdős-652, 654, 1040 |
| Level 1 | | | Erdős-1051 |
| Level 2 | ACGKMP26, FYZ26 | BKKKZ26, LeeSeo26 | Feng26 |
| Level 3 | | | |
| Level 4 | | | |

注意 Level 2 内部 variation 极大（作者强调），不能解读为 "AI 达到人类数学家水平"。

### HAI Card 概念

类比 model card，论文提出 Human-AI Interaction Card，记录关键 prompt 和 AI 输出，目的是 transparency。对 Level C 和 Level A 的结果，作者建议至少公开包含 essential new insight 的 raw prompt 和 output。

Tao 已经发起 "best practices for incorporating AI" 的在线讨论: https://ai-math.zulipchat.com/#narrow/channel/539992-Web-public-channel---AI-Math/

---

## 七、Ablation Studies（§4.1）

### 7.1 在 13 个 Erdős 问题上的对比

用 Gemini Deep Think (IMO scale) 跑相同 prompts，~2× compute 情况下解出 8/13。Aletheia 解出全部 13。

| 333 | 397 | 591 | 652 | 654 | 659 | 705 | 935 | 992 | 1040 | 1051 | 1089 | 1105 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |

### 7.2 在 research paper prompts 上的对比

| FYZ26 | Feng26 | LeeSeo26 | BKKKZ26 | ACGKMP26 |
|-------|-------|----------|---------|----------|
| ✓ | ✗ (3个 prompt 全失败) | partial (1/2) | ~✓ | partial (bound 不如 Aletheia sharp) |

这表说明 Aletheia 的 agent harness 提供的 verifier-reviser 闭环确实有意义，特别是 Feng26（完全 autonomous 的 eigenweights 计算）Deep Think 单独跑 3 个 prompt 都失败。

---

## 八、附录技术细节

### 8.1 FM-Grad-011：Mean-Field Potts Model Phase Transition

这是 FutureMath Basic 的一道题。给定 $S=\{1,2,\dots,q\}$，$q\geq 3$，对 $x\in S^N$ 定义：
$$U_N(x) = \frac{1}{2N}\sum_{1\le i,j\le N} \mathbf{1}(x_i = x_j)$$

记 $N_k(x) = \sum_{i=1}^N \mathbf{1}(x_i = k)$（状态 $k$ 的占据数），empirical measure $p_k(x) = N_k(x)/N$，则：
$$U_N(x) = \frac{N}{2}\sum_{k=1}^q p_k(x)^2$$

Partition function：
$$Z_{N,\beta} = \sum_{x\in S^N} e^{\beta U_N(x)}$$

问题：求最大 $\beta$ 使得
$$\lim_{N\to\infty} \frac{1}{N}\log Z_{N,\beta} = \frac{\beta}{2q} + \log q$$

**Aletheia 的解法**：

由 Sanov 定理，empirical measure 满足 LDP，rate function 为 Shannon entropy $H(p) = -\sum_k p_k \log p_k$。由 Varadhan 引理：
$$\lim_{N\to\infty}\frac{1}{N}\log Z_{N,\beta} = \sup_{p\in\Delta_q} F_\beta(p)$$
其中 free energy functional：
$$F_\beta(p) = \frac{\beta}{2}\sum_{k=1}^q p_k^2 - \sum_{k=1}^q p_k \log p_k$$

均匀分布 $p^* = (1/q,\dots,1/q)$ 给出 $F_\beta(p^*) = \frac{\beta}{2q} + \log q$。

**关键分析**：用 Lagrange multiplier 找 critical points。设 $f(z) = \beta z - \log z$，由 $f''(z) = z^{-2} > 0$（严格凸），$f(z) = \lambda + 1$ 至多两个交点。所以 critical point 形如 $(x, y, \dots, y)$，$x$ 出现 $m$ 次、$y$ 出现 $q-m$ 次。Hessian 二阶分析表明局部最大只能在 $m=1$ 或 $m=q$。

进一步把对称 subspace 写成一维函数：
$$h(x) = \frac{\beta}{2}\left(x^2 + \frac{(1-x)^2}{q-1}\right) - x\log x - (1-x)\log\left(\frac{1-x}{q-1}\right)$$

**First-order phase transition**：随 $\beta$ 增大，在某个 $x_c > 1/q$ 出现第二个 local max，最终追上 uniform。临界条件：
1. Stationarity: $h'(x_c) = 0$
2. Equal heights: $h(x_c) = h(1/q)$

由 $h'(x_c) = 0$ 得：
$$\beta = \frac{q-1}{qx_c - 1}\log\frac{x_c(q-1)}{1-x_c}$$

代回 equal-height 条件，可验证 $x_c = \frac{q-1}{q}$ 是唯一非平凡解。代入：
$$\beta_{\max} = \frac{q-1}{q\cdot\frac{q-1}{q} - 1}\log\left((q-1)^2\right) = \frac{2(q-1)}{q-2}\log(q-1)$$

**变量含义**：
- $q$：Potts model 的状态数
- $N$：粒子数
- $\beta$：inverse temperature
- $p_k$：状态 $k$ 的 empirical probability
- $x_c$：critical point 处大分量取值
- $\beta_{\max}$：free energy density 仍由 uniform 分布主导的最大 inverse temperature

这是经典的 Curie-Weiss-Potts model 结果（Costeniuc-Ellis-Touchette 2005, Wu 1982, Rev. Mod. Phys. 54:235），但 Aletheia 从 first principles 推出来。

### 8.2 IMO 2025 Problem 6: 2025×2025 Grid Tiling

**问题**：2025×2025 grid，每行每列恰好一个 unit square 不被覆盖，问最少要多少 rectangular tiles。

**Gemini Deep Think (advanced, Jan 2026) 在 $2^{12}$ scale 的解**：

**Lower bound**：$T \geq N + a + b - 3$，其中：
- $N = 2025$（uncovered squares / holes 数量）
- $a$：hole 坐标形成的 permutation 的 LIS 长度
- $b$：LDS 长度

由 Erdős-Szekeres / Dilworth：$a \cdot b \geq N$。AM-GM：$a + b \geq 2\sqrt{2025} = 90$。所以 $T \geq 2025 + 90 - 3 = 2112$。

**构造（达到 lower bound）**：取 $k = 45$，将 grid 分成 $k\times k$ 个 macro-blocks，每块 $k\times k$。macro-block $(u,v)$ 的 hole 放在：
- Row: $(u-1)k + v$
- Column: $(u-1)k + (k+1-v)$

这给出 $a = b = 45$。剩余空间分为：
- Central tiles: $(k-1)^2 = 44^2 = 1936$ 个 $45\times 45$ 大方块
- Edge tiles: $4(k-1) = 176$ 个级联边界矩形
- 总计：$1936 + 176 = 2112$

**但**：第一版 output 引用了 advanced theorem without proof（EGMO 的 discrete geometry 结果），按 IMO 规则只能 1-3 分。作者额外 prompt 让模型 self-contained 用 elementary techniques，第二版（在 $2^8$ scale）给出完整 elementary 证明。

**Elementary 证明的关键**：

1. **Rectilinear partition formula**：$T = H + V + I + 1 - N$
   - $H, V$：最大水平/垂直内部线段数
   - $I$：严格内部 crossings 数
   - 通过数矩形的 90° 角（4个角 per rectangle），分类 grid corners (4)、crossings (4I)、T-junctions (4H+4V)

2. **LIS/LDS penalty**：对 LIS 每个 gap $[y_k, y_{k+1}]$，hole column 必须从 $\leq c_{y_k}$ 跳到 $\geq c_{y_{k+1}}$，找到 row $y_k^*$ 与 column $x_k^*$，使 $u_{y_k^*} + w_{x_k^*} + X_{x_k^*,y_k^*} \geq 1$。LIS 和 LDS 的 row index 集合严格不相交（前者要 $c_y < c_{y+1}$，后者要 $c_y > c_{y+1}$），所以无双重计数。

3. **最终**：$\sum u_y + \sum w_x + \sum X_{x,y} \geq (a-1) + (b-1) = a + b - 2$，得 $T \geq N + a + b - 3 = 2112$。

### 8.3 IMO 2024 Problem 3（在 $2^7$ scale 成功，有小错）

**问题**：男孩女孩交替排队，前 $N$ 人任选正整数。对 $m > N$，第 $m$ 人选的数 = $a_{m-1}$ 在 $a_1, \dots, a_{m-2}$ 中出现次数 $+1$。证明 $\{b_n\}$ 或 $\{g_n\}$ 最终周期。

**模型解法的关键步骤**：

1. **简化规则**：$a_m = c_{m-1}(a_{m-1})$，其中 $c_m(x)$ 是 $x$ 在前 $m$ 项中的出现次数。
2. **大数有界**：$M_0 = \max(N, \max_{i\le N} a_i)$。归纳证明 $c_m(y) \leq M_0$ for all $y > M_0$。
3. **无穷出现集合 $S$**：$S = \{1, 2, \dots, L\}$ for some $L \geq 1$。整数 $V > M_2$ 恰好出现 $L$ 次。
4. **交替结构**：$a_{m-1} \in S \Rightarrow a_m > L \notin S$；$a_{m-1} \notin S \Rightarrow a_m \leq L \in S$。男孩女孩必有一方总从 $S$ 选。
5. **状态空间有界**：$\nu_n$ 是 $S$ 中元素计数向量。$\nu_{n+1} = \nu_n + e_{s_{n+1}}$，$s_{n+1}$ 是新增值的 rank。证明 $\max(\nu_n) - \min(\nu_n)$ 全局有界，所以状态空间 $X_n = (s_{n+1}, \nu_n - \min(\nu_n)\mathbf{1})$ 有限。确定性转移 $\Rightarrow$ 最终周期。

注：模型在某处有 "miscellaneous mistake"，作者加了 footnote。

### 8.4 IMO 2024 Problem 5（在 $2^8$ scale 成功）

**问题**：$3002\times 3001$ grid，左上角放 stone。Peter 选 3000 个 cell（每中间行 1 个、每列至多 1 个）。James 不知道位置，需把 stone 移到最后一行。踩到 Peter 的 cell 罚 1 分并回到起点。求最小 $n$ 使 James 能在罚 $n$ 分前到达。

**模型解法**（与 online 标准解法不同，不用 staircase / happy triangle pattern，改用 state-space 推理）：

1. **Lower bound $n \geq 3$**：Peter 先在 James 第一条路径的第 2 行 crossing 处放 trap，再在新路径第 3 行 crossing（不同列）放 trap，至少罚 2 次。

2. **Upper bound $n \leq 3$**：James 用 row-by-row "Safe Probing" 策略：
   - $A$：未发现 trap 的 column 集合
   - $S$：已发现 trap 的 column 集合
   - 选 column $x \in A$（若已有 1 penalty，选 $A$ 中最接近 $c_1$ 的）
   - 探测其他 $y \in A\setminus\{x\}$，找到本行 trap
   - 第 2 次 penalty 后用 "Drop to Finish"：沿 column $x$ 下、横到 $c_1$、沿 $c_1$ 下到底（$c_1$ 已用过的 trap 在 $r_1 < k$，下方全空）

答案：$n = 3$。

---

## 九、论文的 Reflection 与 Limitation

### 9.1 AI 的 qualitative gap

- Autonomous 结果仍 **brief 且 elementary**，远不如典型人类 paper
- 成功来自 clever technical manipulation 或 vast knowledge retrieval，而非 genuine creativity
- "Specification gaming"：模型倾向把模糊问题解读成最易版本，导致 50/200 technically correct 但 mathematically vacuous

### 9.2 Erdős 案例的教训

AI 解 Erdős 问题的 "novel solution" 频繁被事后发现已在文献中（如 1026, 397, 333, 281）。作者强调：人类 mathematician 极少犯这种 redundant 错误，因为现代通信发达。AI 之所以频繁出现，是因为这些 solution 太简单——如果是人类写的，根本不会引起注意。Erdős-1089 就是 Bannai-Bannai (1981) 一句话 remark，作者自己都没意识到解题。

### 9.3 比较优势

AI 与人类 intelligence diverge：
- 单一领域：frontier models knowledge **远浅于** domain expert
- Breadth：**superhuman** 
- 物理限制：AI 不受 time/attention 限制，能扫 700 个 open problems

这暗示 AI 的最佳应用是 **需要 vast memory / computation / breadth** 的问题，而非深度 creativity 问题。

---

## 十、我的几点延伸 intuition

1. **Verifier 与 Generator decoupling 的本质**：这其实是 amortized inference 的一种形式。Generator 用大量 thinking compute 探索，Verifier 用独立 context 重新评估，相当于 importance sampling 的 resampling 步骤。可以理解为把 $P(\text{answer} \mid \text{problem})$ 分解为 $P(\text{answer} \mid \text{CoT}, \text{problem}) \cdot P(\text{CoT} \mid \text{problem})$，然后 Verifier 在 marginal 上独立评估。

2. **Tool use 是 retrieval-augmented reasoning 的硬约束**：citation hallucination 的根本原因是 LLM 在 low-density 区域的生成本质是 weighted interpolation。Search 工具把生成分布 clamp 到真实 corpus 上，把 hallucination 从 "假实体" 推到 "错引用"。

3. **Level 2 内部的巨大 spread**：Feng26 vs ACGKMP26 vs LeeSeo26 都是 Level 2，但难度和 significance 差距巨大。这暗示二维 taxonomy 太粗，可能需要第三轴（如 "tooling dependence" 或 "domain accessibility"）。

4. **Inference scaling 的渐近行为**：Figure 2 的曲线在 Olympiad 上有清晰 plateau，但 PhD 级别似乎还在 linear region。这暗示研究级别问题需要 **远超目前 inference budget** 的 compute，agent harness 是 amortize 这种 compute 的方式。

5. **HAI Card 的意义**：这其实是 ML 社区 reproducibility 标准向数学界推广的尝试。model card 记录 model bias/perf，HAI Card 记录 human-AI 边界。未来数学界可能要求所有 AI-assisted paper 附 HAI Card，类似 data availability statement。

6. **Erdős 700 问题的 31.5%/6.5% 数据**：这是目前对 LLM math capability 最 honest 的 quantitative 评估之一。注意 6.5% 是 conditional on Aletheia 自己返回的 212 candidates 中的 200 个可判定，所以真正 rate 是 13/700 ≈ 1.86%。这是 "research-level open problem" 上的 success rate。

---

## 十一、相关工作的 broader context

- **AlphaProof/AlphaGeometry2**：formal language 路线，correctness guaranteed 但 expressiveness 受限
- **AlphaEvolve**（Novikov et al., 2025, https://arxiv.org/abs/2506.13131）：用 LLM 进化算法/代码
- **FrontierMath**（Glazer et al., 2024, https://arxiv.org/abs/2411.04872）：研究级别 benchmark
- **UQ benchmark**（Nie et al., 2025, https://arxiv.org/abs/2508.17580）：unsolved questions 评估
- **Gödel Test**（Feldman-Karbasi, 2025, https://arxiv.org/abs/2509.18383）：测试 LLM 解 easy conjectures
- **Schmitt25**（moduli space descendant integrals, https://arxiv.org/abs/2512.14575）
- **BEMSV26**（genus 0 maps to flag variety, https://arxiv.org/abs/2601.07222）
- **JangRyu25**（Nesterov convergence, https://arxiv.org/abs/2510.23513）
- **Woodruff et al., 2026**（Gemini 在 TCS/econ/physics 的 case studies, https://arxiv.org/abs/2602.03837）
- **Bubeck et al., 2025**（GPT-5 early science acceleration, https://arxiv.org/abs/2511.16072）

---

## 十二、最终总结

这篇 paper 的真正贡献不在于解出几个 Erdős 问题或写出 eigenweights paper，而在于：

1. **建立了 AI-assisted math research 的 transparency 框架**（Autonomy Level × Significance Level + HAI Card）
2. **Quantitative 揭示了 AI 在 research math 上的真实能力分布**（31.5% technically correct / 6.5% meaningfully correct）
3. **证明了 agent harness + tool use 是从 Olympiad 跨向 research 的关键桥梁**
4. **诚实承认 AI 的 comparative advantage 在 breadth 而非 depth**

对 Karpathy 你而言，这篇 paper 最重要的 takeaway 可能是：**inference-time scaling 在 Olympiad 已经接近 saturation，但在 research math 还有 headroom，且 scaling 必须配合 verifier-reviser 闭环和 tool grounding 才能转化为 reliable research output**。这与你在 LLM101 / nanoGPT 里强调的 "scale + architecture" 互补——architecture 的 agentic harness 现在是 scaling 的 multiplier，而非单纯 add-on。
