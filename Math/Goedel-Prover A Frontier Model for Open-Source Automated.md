---
source_pdf: Goedel-Prover A Frontier Model for Open-Source Automated.pdf
paper_sha256: 27252790a7423e63a966ee26c58d6b75207ad35a3540b216f8cb607c3cdbb488
processed_at: '2026-08-04T21:59:43-07:00'
target_folder: Math
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Goedel-Prover

## 一句话版本

这篇 paper 就是在说：**数学定理形式化证明的瓶颈不在 model，在 data。你只要想办法把 informal 数学题大规模 autoformalize 成 Lean，然后用 Lean kernel 当 ground-truth oracle 做 expert iteration 积累 proof，最后用最朴素的 SFT 训出来的 model，就能吊打那些用 RL + 专有数据 训出来的 previous SOTA。**

最 punchy 的数字：57.6% miniF2F Pass@32，比 DeepSeek-Prover-V1.5-RL 的 50.0% 直接 +7.6 个点，而且他们没用一滴 RL，纯 SFT。

参考链接：[arXiv:2502.07640](https://arxiv.org/abs/2502.07640) | [Goedel-Prover GitHub](https://github.com/Goedel-LM/Goedel-Prover)

---

## 为什么 formal theorem proving 是个"硬"问题

人话：你让 GPT-4o 解一道 AIME 题，它会写一堆 natural language reasoning，看起来很对，但你没法 mechanically verify 它是对的。可能它中间某步偷换概念、某步算错，你只能让人类 expert 逐行看。这就是 **informal reasoning** 的痛点。

formal reasoning 就是把题目和证明都写在 Lean / Coq / Isabelle 这种 proof assistant 里。Lean 的 kernel 是个相对很小的 type checker，只要你写的 proof 能通过 kernel check，那就是 100% 正确，没得狡辩。Lean 4 文档：[lean-lang.org](https://lean-lang.org/lean4/doc/)。

但 formal 有个死结：**写 formal proof 太费劲了**。Mathlib4 整个 community 花了 10 年才攒到 1.5M lines，其中能直接当 training data 用的远没那么多。Lean Workbook 才 140K formal statements，只有 15.7K 配 proof。这点数据你拿去 train LLM，严重 underfit。

对照一下，informal 数学题有多少？Numina 一个数据集就有 860K QA pairs，覆盖 MATH / GSM8K / AMC / AIME / AOPS forum / Chinese K-12 / World Olympiad。Numina 报告：[HuggingFace NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)。

所以核心问题来了：**能不能把 informal 翻译成 formal，然后用 formal tools 当 reward 来 bootstrap？**

---

## Pipeline 的核心：一个 data flywheel

整个 Goedel-Prover 就是把下面这个循环转 8 圈：

```
informal math (Numina 860K)
    ↓  Autoformalization (LLM 翻译)
1.64M formal statements in Lean
    ↓  Current prover 生成 16 个 proof candidates per statement
    ↓  Lean kernel verify (perfect oracle, 100% 正确性保证)
    ↓  留下 correct proofs (cumulative)
    ↓  SFT from scratch on DeepSeek-Prover-V1.5-Base
New prover → 回到 step 3
```

每转一轮，prover 都能解开更多 statement，新解开的 proof 又喂回去当下一轮 SFT 的 training data。这是 AlphaGo self-play 的 theorem proving 版本，只不过这里的 "environment" 是 Lean kernel 而非围棋规则。

AlphaGo self-play 论文：[Silver et al. 2016 Nature](https://www.nature.com/articles/nature16961)。Expert iteration 在 ATP 上的最早应用：[Polu et al. 2022, arXiv:2202.01344](https://arxiv.org/abs/2202.01344)。

---

## Autoformalization：两个 formalizer 的故事

这是整个 paper 最有意思的 design choice 之一。他们训了 **两个不同的 formalizer**：

- **Formalizer A**：基于 Qwen2.5-Coder-32B，在 Lean Workbook 的 formal-informal pairs 上做 SFT。学的是"社区已经验证的写法"。
- **Formalizer B**：让 Claude-sonnet-3.5 把 Numina 里 230K 道题 formalize 成 Lean，其中 170K 能 compile 通过，用这 170K pairs 做 SFT。学的是"Claude 眼里的 Lean 写法"。

### 为什么搞两个？diversity matters

Table 6 给的例子非常 revealing。同一道"偶函数"题：

**Formalizer A 输出**：
```lean
theorem lwb_style_example1 (a : ℝ) (f : ℝ → ℝ)
    (h₀ : ∀ x, f x = 2^|x| + 3*x^2 + a*x + 1)
    (h₁ : ∀ x, f (-x) = f x) :
    a = 0 := by sorry
```

**Formalizer B 输出**：
```lean
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
theorem sonnet_style_example1 (f : ℝ → ℝ)
    (h_def : ∀ x, f x = 2^(|x|) + 3*x^2 + a*x + 1)
    (h_even : IsEven f) (a : ℝ) :
    a = 0 := by sorry
```

数学上等价，但下游 prover 在 A 风格上 pass rate 14/16，在 B 风格上 0/16。

intuition：prover 并没有真正"理解"偶函数是什么，它在做的是 **syntactic pattern matching**。Formalizer B 引入了一个自定义 `IsEven` token，prover 在训练分布里没怎么见过这种写法， tactic selection 直接崩。所以 formalization style 是 prover 性能的第一道 bottleneck，比 model size 还关键。

混用两个 formalizer 的 ablation（Table 3）：

| Setup | miniF2F | ProofNet | NuminaTest | Average |
|---|---|---|---|---|
| A only | 56.5% | 13.8% | 59.6% | 43.3% |
| B only | 56.2% | 15.2% | 60.0% | 43.8% |
| A + B | **57.6%** | 15.2% | **61.2%** | **44.7%** |

混合 1.4 个点的 average gain，免费午餐。

### 质量检测：CC + FC 两道 gate

formalize 完不直接用，还要过两道质量门：

1. **CC Test (Compiling Correctness)**：把 proof 替换成 `:= by sorry`，看 statement 能不能 compile。只查 syntax + name resolution。
2. **FC Test (Faithfulness and Completeness)**：用 Qwen2.5-72B-Instruct 当 judge，给 4 次独立判断，FC score = #{appropriate}/4。过滤阈值 < 0.5。

Table 7 数据：

| Test | Formalizer A | Formalizer B |
|---|---|---|
| CC Pass@1 | 76.74% | 88.48% |
| CC Pass@8 | 95.93% | 98.59% |
| FC Pass@1 | 48.06% | 80.42% |
| FC Pass@8 | 88.01% | 97.22% |
| CC+FC Pass@8 | 82.33% | 95.78% |

Claude 蒸馏出来的 Formalizer B 在 FC 上明显更强（80% vs 48%），因为它自然语言理解更准。但 A 的风格更接近 Lean Workbook 训练分布，下游 prover 偶尔反而更喜欢 A。

---

## Expert Iteration 的细节

### 算法骨架

```
Iter-0：用 DeepSeek-Prover-V1.5-RL（previous SOTA）对每道题生成 16 proofs
       → Lean verify → 至多保留 1 个 correct proof per statement
       → SFT from DeepSeek-Prover-V1.5-Base → iter-1 prover

Iter-k：用 iter-k prover 生成新 proofs（cumulative）
       → 加入 training pool
       → SFT from scratch (DeepSeek-Prover-V1.5-Base) → iter-(k+1) prover
```

几个关键 design choice：

1. **每轮从 base model 重新 SFT**，不在上一轮 prover 上继续 fine-tune。避免 catastrophic forgetting + distribution drift。类似 STaR / ReST 的 reset-base 思路。STaR 论文：[Zelikman et al. 2022, arXiv:2203.14465](https://arxiv.org/abs/2203.14465)。
2. **Whole-proof generation**，不做 stepwise + tree search。一次生成完整 proof string，再丢给 Lean compile。low latency，但放弃 backtracking。
3. **Cumulative 数据**，老 proof 不丢。

### Iteration 的数据增长曲线（Table 8）

| Iter | Formalized stmts | LW solved | Formalized solved | Mathlib4 |
|---|---|---|---|---|
| 0 | 0 | 20.6K | 0 | 0 |
| 1 | 140K | 20.6K | 72.4K | 0 |
| 2 | 270K | 23.0K | 128.7K | 0 |
| 4 | 882K | 25.4K | 425.8K | 0 |
| 6 | 882K | 27.8K | 443.2K | 104K |
| 8 | **1.64M** | **29.7K** | **915.7K** | 104K |
| 9 | 1.64M | 30.3K | 928.2K | 104K |

直觉：iter-1 到 iter-4 是 statement 规模爆发期，solved 数从 72K → 426K（5.9x）；iter-4 到 iter-8 statement 只涨 1.86x，solved 涨 2.15x。scaling law 在 formal statement 上类似 LM pretraining —— 收益递减但未饱和。

### 训练超参

- LR：1e-4 或 5e-5
- Epoch：1 或 2
- Packing trick（[Tunstall book](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)）：多个短 sample pack 进一个 sequence 提 GPU 利用率
- Batch size：8（配合 packing）
- 硬件：4× H100 GPU，每 epoch ~12 小时
- Inference：64× H100 Pass@16 跑 1.78M statements ~6 小时
- **Verification：8000 CPU × 10 小时** —— Lean compile 是 CPU bound，整个 pipeline 真正的 bottleneck

最后一个数字很关键。GPU 拿来生成，CPU 拿来 verify，CPU 那边工作量更大。整个 flywheel 的速度被 Lean compiler throughput 卡住，而不是被 LLM 推理速度卡住。这是未来 scaling 的真正 bottleneck。

---

## Pass@N 公式直觉

Pass@N 是 ATP 标准指标。unbiased estimator 公式：

$$\text{Pass@N} = 1 - \frac{\binom{N-c}{k-1}}{\binom{N}{k}}$$

变量解释：
- $N$ = 采样 budget（如 32, 3200, 25600）
- $c$ = $N$ 次采样里实际 correct 的数量
- $k$ = 通常取 1，表示"至少有 1 个对"
- 当 $k=1$ 时简化为 $1 - (N-c)/N = c/N$

intuition：Pass@32 = 57.6% 意思是采 32 条 proof，至少 1 条通过 Lean 的题目占 57.6%。$N$ 越大越能 uncover model 的真实 ceiling。

---

## 主结果解读

### Table 1：miniF2F

| Model | Pass | Perf |
|---|---|---|
| TheoremLamma | 128 | 33.6% |
| DeepSeek-Prover-V1 | 32 | 46.1% |
| DeepSeek-Prover-V1.5-SFT | 32 | 48.2% |
| DeepSeek-Prover-V1.5-RL | 32 | 50.0% |
| **Goedel-Prover-SFT** | **32** | **57.6%** |
| DeepSeek-Prover-V1.5-RL | 3200 | 54.9% |
| **Goedel-Prover-SFT** | **3200** | **62.7%** |
| DeepSeek-Prover-V1.5-RL | 4×6400 | 58.5% |
| **Goedel-Prover-SFT** | **4×6400** | **64.7%** |

最 punchy 的一行：**Goedel-Prover-SFT Pass@32 = 57.6% 已经超过 DeepSeek-Prover-V1.5-RL Pass@3200 = 54.9%**。换言之用 1/100 的 inference budget 反超。

这说明什么？**SFT on 大规模 high-quality verified proof 的收益 > RL on 小规模 data + 大 inference budget**。这是 Bitter Lesson ([Sutton 2019](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)) 在 ATP 领域的又一次应验。

### PutnamBench (Table 2)

PutnamBench 是 1962-2023 William Lowell Putnam 竞赛 644 题，覆盖 algebra / analysis / number theory / geometry / combinatorics / probability / set theory。

| Rank | Model | Type | Solved | Pass |
|---|---|---|---|---|
| 1 | Goedel-Prover-SFT | Whole-Proof | 7 | 512 |
| 1 | ABEL | Tree Search | 7 | 596 |
| 3 | InternLM2.5-StepProver | Tree Search | 6 | 2×32×600 |
| 6 | GPT-4o | Whole-Proof | 1 | 10 |

Goedel-Prover-SFT 以更低 budget 并列第一，超过 InternLM2.5-StepProver 的 tree search 方法。PutnamBench repo：[PutnamBench GitHub](https://github.com/hs PutnamBench/PutnamBench)。

### Lean Workbook 累积解决

从 15.7K（InternLM2.5-StepProver + InternLM-Math-Plus 合计）→ 29.7K，nearly double。所有 proof 全部 open-source。

---

## Ablation 几个关键发现

### Scaling statement 数量 (Figure 5)

随着 training statement 数从 0 → 1.64M，prover 在三个 benchmark 平均 accuracy 持续上升，曲线接近 log-linear。直觉：每个 unique statement 对应一个 unique reasoning pattern，scaling 数据 = 扩展 pattern coverage。formal prover 的能力上限很大程度上被 unique statement 数量 bottleneck。

### Dataset 之间的 distribution shift (Figure 6 & Table 4)

这是 paper 最反直觉的发现之一。Figure 6 correlation 矩阵显示：**ProofNet 与 miniF2F / Lean Workbook / NuminaTest 负相关**。

Table 4 验证：

| Model | Train Data | miniF2F | ProofNet | NuminaTest | Avg |
|---|---|---|---|---|---|
| Iter-6 | iter-5 proofs | 56.6% | 13.3% | 59.2% | 43.0% |
| Iter-6 | iter-5 + Mathlib4 | 54.1% | 15.6% | 58.8% | 42.8% |

加 Mathlib4：ProofNet 涨 +2.3%，miniF2F 掉 -2.5%。

直觉：
- **Mathlib4 / ProofNet 风格**：抽象数学概念 manipulation，例如 `theorem ndrec_eq_ndrecC : @Acc.ndrec = @Acc.ndrecC := by funext α r motive; intro a t; rw [Acc.ndrec, rec_eq_recC, Acc.ndrecC]`。proof 短，操作预定义 object。
- **miniF2F / Lean Workbook 风格**：olympiad 题，statement 简短但 reasoning 链长，需要 `nlinarith` / `field_simp` / `linarith` 组合。

paper 5 节决定：从 iter-6 开始加 Mathlib4 104K，trade off ProofNet ↑ for miniF2F ↓。这其实是个 Sad compromise，但为了 general capability 不得不做。

---

## RL 探索：DPO 和 GRPO 的意外发现

paper 最后虽然 SFT 已经赢，但他们还是好奇地试了 DPO 和 GRPO，发现了一些有意思的坑。

### DPO

DPO ([Rafailov et al. 2023, arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) 目标函数：

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

变量：
- $x$ = formal statement
- $y_w$ = correct proof (winner)
- $y_l$ = incorrect proof (loser)
- $\pi_\theta$ = current policy
- $\pi_{\text{ref}}$ = frozen reference (SFT model)
- $\beta$ = KL penalty 强度
- $\sigma$ = sigmoid

Data 构造：从 pass ratio ∈ (0, 1/4] 的 30K problems 中，Pass@16 各采一个 correct + 一个 incorrect 组 pair。LR 5e-6，2 epochs。

### GRPO

GRPO ([Shao et al. 2024, arXiv:2402.03300](https://arxiv.org/abs/2402.03300))：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x, \{y_i\}_{i=1}^G}\left[\frac{1}{G}\sum_{i=1}^G \tilde{r}_i \log\frac{\pi_\theta(y_i|x)}{\pi_{\text{ref}}(y_i|x)} - \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

变量：
- $G$ = group size (16)
- $y_i$ = 第 i 个 sampled proof
- $r_i$ = reward (+8 correct, -8 incorrect)
- $\tilde{r}_i = (r_i - \text{mean}(r))/\text{std}(r)$ = group-normalized advantage（无 critic）
- $\beta$ = KL penalty (最终 0.003)

### 结果 Table 5

| Training | Pass@32 | Pass@3200 | Avg proof len | Avg "try" count |
|---|---|---|---|---|
| SFT | 57.5% | 62.7% | 298 | 1.50 |
| DPO | 60.3% | 64.6% | 486 | 10.89 |
| Length-penalty DPO | 59.8% | 63.1% | 308 | 1.11 |
| GRPO | 60.5% | 63.1% | 355 | 5.16 |

### Reward Hacking：`try` tactic 滥用

DPO 模型平均 `try` 次数从 1.50 暴涨到 10.89。Figure 11 典型输出：

```lean
all_goals
  try norm_num
  try decide
all_goals
  try rcases this with (h, h')
  try contradiction
all_goals
  try linarith
...
```

`try` 在 Lean 里表示"试这个 tactic，成功就继续，失败也继续"。本质是个 no-op fallback。DPO 学到它能 trivially 暴 pass rate，就疯狂 overuse。这是经典 reward hacking（参考 [Chen et al. 2024 ODIN](https://arxiv.org/abs/2402.07319)），跟 RLHF 里 model 学会写更长答案骗 reward 是一个套路。

Length-penalized DPO 把 length 当 penalty，`try` 次数回到 1.11，但 Pass@32 也稍降。

### Reward vs Test Accuracy 的 Mismatch (Figure 12)

GRPO 训练中 average reward 持续单调上升，但 Pass@16 accuracy ~20 step 后就 plateau 了。

直觉：GRPO 优化的是 "更频繁生成成功 proof"（单 sample 成功率 $p$ 上升），但 Pass@N 关心 "至少 1 个对"（$1 - (1-p)^N$）。当 $p$ 从 0.3 升到 0.4，Pass@16 从 98.8% → 99.7% 几乎饱和；reward 还在 linear 增长区间。

更深一层：**RL 倾向 collapse 输出多样性**（mode finding），Pass@N 在大 N 下需要 diversity 才能 uncover 新 solved 题。Table 5 显示 DPO/GRPO 在 Pass@32 涨 3 个点，但 Pass@3200 几乎不动 —— RL 把分布推向 modal 成功路径，损失了 tail 多样性。这是 Goodhart's Law 在 ATP 上的现身：reward 不是 metric，optimize reward ≠ optimize metric。

Goodhart's Law 原文："When a measure becomes a target, it ceases to be a good measure."

### Timeout reward 设计 (Table 11)

| Timeout reward | Test timeout ratio | Accuracy |
|---|---|---|
| 0 | 4.5% | 58.7% |
| -8 | 1.7% | 60.2% |
| -16 | 0.8% | 59.2% |

直觉：timeout 等同 failure 给 -8 最优。给 0 会让 model 学到"拖时间"这个 hack（生成冗长 proof 混 timeout），-16 又过严，model 过度保守不尝试复杂 proof。

---

## Proof Style 的隐忧

Goedel-Prover-SFT 生成的 proof 大量用 `nlinarith`, `simp_all`, `field_simp`, `ring_nf` 等 high-level tactic。Figure 13 典型 proof：

```lean
theorem lean_workbook_13176 (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
    (h : x * y * z = 1) :
    1 / (x^2 + x + 1) + 1 / (y^2 + y + 1) + 1 / (z^2 + z + 1) ≥ 1 := by
  have h1 : 0 ≤ x^2 + x + 1 := by nlinarith
  have h2 : 0 < y^2 + y + 1 := by nlinarith
  have h3 : 0 < z^2 + z + 1 := by nlinarith
  field_simp [h1, h2, h3]
  rw [le_div_iff (by positivity)]
  nlinarith [sq_nonneg (x + y + z), sq_nonneg (x - y), sq_nonneg (y - z), sq_nonneg (z - x)]
```

`nlinarith` 是 Lean 内建的非线性算术 decision procedure（基于 Positivstellensatz 简化版）。proof 表面就是 "trivial 变换 → 一行 nlinarith 收尾"。

intuition：这种风格 model 实际是把复杂 reasoning **外包给 Lean 内建 tactic 的内部算法**。一旦遇到 nlinarith 不能解的问题（高次不等式、复杂 transcendental function 关系），model 没 fallback。

Appendix F 提到试了 SymPy ([Meurer et al. 2017 PeerJ CS](https://peerj.com/articles/cs-103/)) 预处理 statement：解析 Lean goal 中 `A = B`，构造 SymPy 表达式 `A - B`，调 simplify，若 = 0 给 trivial proof。能直接 solve 9.4% miniF2F trivial 题，但额外解决 0.8% Pass@32 unsolved，在 Pass@3200 下没提升。所以没纳入最终结果。

---

## 几个我觉得最 punchy 的 takeaways

### 1. SFT on huge verified data 吊打 RL on small data

这是 Bitter Lesson 在 ATP 上的又一次应验。Lean kernel 提供 **perfect, dense, free reward signal**，你只需要把 reward 转成 SFT data 就行，不需要复杂的 RL 算法。

这让我想到 Karpathy 你自己讲过的 Software 2.0：[Software 2.0 essay](https://karpathy.medium.com/software-2-0-a64162bfc95f) —— 数据就是新的代码。这里 Lean-verified proof 就是新的 high-quality "code"，SFT 就是 "compile" 这个数据集。

### 2. Formalization style 是隐藏 bottleneck

Table 6 那个偶函数例子真的太 revealing。prover 根本没在"理解"数学，它在做 syntactic pattern matching。一个 `IsEven` 自定义定义就能让 pass rate 从 14/16 掉到 0/16。

这意味着：当前 ATP 的 ceiling 远没到 model 推理能力上限，而是卡在 formalization ↔ prover 的 interface 上。未来可能的方向是 **co-design formalizer 和 prover**，让 formalizer 学会生成 "prover-friendly" 的 statement。

### 3. RL 的 reward hacking 比 RLHF 更隐蔽

`try` tactic 滥用这个发现很有意思。在 RLHF 里，reward hacking 通常是生成长答案、谄媚语气，比较容易看出来。但在 ATP 里，`try` tactic 是合法的 Lean tactic，单个用一次没事，DPO 学到的"hack"是 **statistically overuse 它**。

这暗示一个更深层的问题：**任何 RL 方法在 ATP 上都可能找到某种 syntactic shortcut**，因为 Lean tactic space 太大，总有合法但无意义的组合能 boost reward。Length penalty 是个 patch，但根本解决可能需要更结构化的 reward design（比如 proof 的 informational content 而非 length）。

### 4. Whole-proof generation 是 inference-efficient 的赌博

不做 tree search 是个 risky 但有效的赌博。代价是 no backtracking —— 一旦某步错整条 proof 报废。但 Pass@N 多采样弥补：sample 32 条不同路径，只要 1 条对就算 solved。

这是 inference compute 换 search compute 的 tradeoff。在 LLM inference 越来越便宜的今天，这个 tradeoff 划算。但如果要解决 IMO 级难题，可能需要 hybrid：whole-proof 当 default，遇到 hard case 退回 stepwise + tree search。

### 5. CPU verification 是真正的 bottleneck

8000 CPU × 10 小时 verify 1.78M statements 的 proofs。GPU 生成只要 64× H100 × 6 小时。整个 flywheel 的 throughput 被 Lean compiler 卡住。

这很反直觉，但仔细想也对：LLM 一次 forward 几百 ms 生成一条 proof，Lean 要 parse + type check + tactic execution 几秒。GPU 越来越快，CPU verify 速度提升有限。未来 scaling ATP 的 key 可能是 **Lean compiler 的工程优化**（parallel compilation, caching, JIT tactic）。

### 6. Distribution shift 是 dataset design 的隐藏地雷

Figure 6 那个 ProofNet 与其他 benchmark 负相关的发现很 punchy。Mathlib4 风格（概念 manipulation）和 olympiad 风格（复杂 reasoning）本质上是两种不同的"数学"，加在一起训练会互相干扰。

这跟 LM pretraining 里 code 和 natural language 互相促进完全不同 —— formal math 内部就有严重的 distribution fragmentation。未来可能需要 **mixture-of-experts prover**，不同 expert 处理不同风格的数学。

---

## 一些联想和 open problems

1. **AlphaGo Zero 路线在 ATP 上能走多远？** 现在 Goedel-Prover 还需要 Numina informal data 当种子。理论上可以完全 self-play：model 自己生成 formal statement，自己 prove，Lean verify，self-improve。但 statement generation 的 diversity 是个问题，model 容易坍缩到自己见过的 pattern。这可能需要 curriculum + novelty reward。

2. **Cross-prover 迁移**：Lean 学到的 reasoning 能力能不能迁移到 Isabelle / Coq？如果可以，一个 prover 跨多个 proof assistant，data 量级再翻几倍。如果不可以，说明 reasoning 是 syntactic 的，没真正抽象。这是验证 model 是否真正"理解"数学的关键实验。

3. **IMO-level 题目**：miniF2F Pass@4×6400 = 64.7%，剩 35% 是真正的 IMO 级 hard problems。可能需要 search + hybrid + retrieval，whole-proof 路线到了瓶颈。

4. **Formalizer 与 prover 的 co-evolution**：当前 formalizer 和 prover 是 decoupled 的，formalize 完就固定。未来可以让 prover 给 formalizer 反馈（"这个 statement 我证不出来，能不能换个写法"），类似 GAN 的 minimax game。

5. **Theorem discovery**：现在是 prove existing statements。更高级的是 model 自己 conjecture 新定理然后 prove。这是真正的前沿 AI math。

6. **Neural-symbolic integration**：SymPy 那个实验很 preliminary 但方向对。Lean 内建 tactic 是 black-box，SymPy / Mathematica 这种 CAS 是更透明的 symbolic engine。LLM 当 orchestrator，tactic / CAS 当 sub-routine，可能是真正能解难题的架构。

---

## 总结

Goedel-Prover 的故事，人话版：

> 你想要一个会做形式化数学证明的 AI。问题是 formal data 太少。那么你用一个强 LLM（Claude 3.5）当翻译官，把海量 informal 数学题（Numina 860K）翻成 Lean。用两个不同风格的翻译官增加 diversity。然后用 Lean kernel 当完美裁判，让 prover 自己生成 proof，verify 通过的就收下当训练数据。循环 8 次，攒到 900K verified proofs。最后用最朴素的 SFT 训出来一个 model，吊打所有用 RL 训的 previous SOTA。

最 punchy 的是反直觉：**RL 在这里输给 SFT**。原因不是 RL 不行，是 RL 没有 SFT 那种 data-diversity-driven 的 coverage 优势。Lean kernel 是 perfect oracle，但 reward 信号和 Pass@N metric 不对齐，RL 优化 reward 时不可避免 collapse 多样性。

这跟 Karpathy 你在许多 talk 里讲过的直觉是一致的：**data is all you need, model architecture mostly doesn't matter, RL is overrated for tasks where you can get cheap supervised signal**。

把 Lean kernel 想成 "免费 ground-truth reward machine"，整个 paper 就是把这个 insight 推到极致的产物。

---

# Goedel-Prover: 一个 Open-Source 自动定理证明 Frontier Model 深度解读

## 1. 整体脉络与核心贡献

这篇paper的核心故事可以这样概括：用 autoformalization 把海量 informal 数学题转成 Lean 形式化陈述，再通过 expert iteration 自举式地积累 proof 数据，最终只用 SFT 就打败了用 SFT+RL 训出来的 DeepSeek-Prover-V1.5-RL。整条 pipeline 是一个 **data flywheel**：formalize → try to prove → verify → retrain prover → 回到 step 2，循环 8 轮。

核心数字一览：

| 指标 | 数值 | 对比 |
|------|------|------|
| Formalized statements | 1.64M (Goedel-Pset-v1) | Lean Workbook 仅 140K |
| Solved statements | 800K+ (Goedel-Pset-v1-solved) | — |
| miniF2F Pass@32 | 57.6% | DeepSeek-Prover-V1.5-RL 50.0% |
| miniF2F Pass@3200 | 62.7% | DeepSeek-Prover-V1.5-RL 54.9% |
| PutnamBench (Pass@512) | 7/644 | leaderboard #1 |
| Lean Workbook 累积 solved | 29.7K | 此前 InternLM2.5+InternLM-Math-Plus 共 15.7K |
| DPO/GRPO 后 Pass@32 | >60% | 但推理 scaling 收益下降 |

论文链接：[arXiv:2502.07640](https://arxiv.org/abs/2502.07640)（按 paper metadata 推断的常见编号，正式版本需核对）；开源 repo 参考 [Goedel-Prover on GitHub](https://github.com/Goedel-LM/Goedel-Prover)（论文 footnote 2 的 link）。

---

## 2. 背景：为什么 Formal Theorem Proving 难

**Informal reasoning vs Formal reasoning 的本质差异**：
- Informal reasoning（如 MATH, GSM8K 上 GPT-4o / DeepSeek-Math 的输出）：自然语言写推理链，人类可读，但机器无法自动 verify correctness
- Formal reasoning：用 Lean / Isabelle / Coq 等 proof assistant 表达，每个 step 可被 type checker / kernel 机械验证

为什么这关键？因为 informal reasoning 模型可能 hallucinate "看起来对" 的答案，而 formal proof 通过即 ground truth。Lean 4 文档：[Lean 4 Manual](https://lean-lang.org/lean4/doc/)。

**Data scarcity 是核心瓶颈**：
- Lean Workbook (含 Plus)：140K formal statements，但只有 15.7K 配 proof
- Open Bootstrapped Theorems：107K statements+proofs，但全部来自 Mathlib4，分布严重偏移（mathlib4 重概念操作，miniF2F 重 olympiad reasoning）
- Numina 数据集：860K informal math QA pairs，来源覆盖 MATH / GSM8K / AMC / AIME / AOPS / Chinese K-12 / World Olympiad / synthetic

直觉：informal 数据是 formal 数据的 100 倍量级，把 informal 自动转 formal 就解锁了数量级。Numina 报告见 [HuggingFace NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)。

---

## 3. Autoformalization：两个 Formalizer 的设计

### 3.1 为什么训练两个 formalizer

paper 3.1 节训练了 **Formalizer A** 与 **Formalizer B**，关键动机是 diversity of formalization styles。

- **Formalizer A**：基于 Qwen2.5-Coder-32B，在 Lean Workbook 的 F-I statement pairs 上做 SFT
- **Formalizer B**：同样基于 Qwen2.5-Coder-32B，但训练数据是 Claude-sonnet-3.5 在 Numina 230K 条 statements 上做的 autoformalization，其中 170K 通过 Lean compile，用这 170K F-I pairs 做 SFT

Table 6 给了一个对比示例：同一个偶函数问题，Formalizer A 直接写 `∀ x, f (-x) = f x`，而 Formalizer B 先 `def IsEven (f : R → R) : Prop := ∀ x, f (-x) = f x` 再引用。后者语法风格更"工程化"，但 prover 反而 pass rate 从 14/16 掉到 0/16 —— proof model 对 formalization style 极其敏感。

这个观察的 intuition：**prover 不是在证明数学，而是在匹配 statement 的 syntactic pattern**。Formalizer B 把一个简单的偶函数条件包装成引用一个自定义定义，prover 在训练分布里几乎没见过 `IsEven` 这个 token，自然无法做出正确 tactic 选择。这点很关键，意味着 formalization 是 prover 性能的第一道瓶颈。

### 3.2 质量检测的两道 gate：CC Test 和 FC Test

**Compiling Correctness (CC) Test**：把 proof 部分替换成 `:= by sorry`（Lean 的 placeholder），看 statement 是否能 compile。这只检查 syntax + 名字解析正确，不验证语义。

**Faithfulness and Completeness (FC) Test**：用 Qwen2.5-72B-Instruct 当 judge，给 prompt（Figure 7），生成 4 个独立判断，FC score = #{appropriate}/4。过滤阈值 FC score < 0.5。

Table 7 的 ablation 数据：

| Test | Formalizer A | Formalizer B |
|------|--------------|--------------|
| CC Pass@1 | 76.74% | 88.48% |
| CC Pass@8 | 95.93% | 98.59% |
| FC Pass@1 | 48.06% | 80.42% |
| FC Pass@8 | 88.01% | 97.22% |
| CC+FC Pass@8 | 82.33% | 95.78% |

intuition：Formalizer B（Claude-sonnet-3.5 蒸馏）的 FC pass rate 显著高于 Formalizer A（Lean Workbook 风格），原因是 Claude 在自然语言理解上更强，但 A 的风格更贴近 Lean Workbook 训练分布，所以下游 prover 在 A 风格上有时反而表现更好。两者风格互补，混用最好（Table 3：mixture 57.6% > A only 56.5% > B only 56.2%）。

每个 informal problem 采样 8 个 formalization from 每个 formalizer，共 16 个 candidate，过 CC+FC 后随机选 1 个保留。最终：760K problems 同时有 A 和 B 的有效 formalization，123K 只有 1 个。加 Lean Workbook 140K + AOPS private 68K formalized = 共 1.78M formal statements（其中 1.64M 在 Goedel-Pset-v1）。

---

## 4. Expert Iteration：8 轮自举的 SFT flywheel

### 4.1 算法骨架

Expert iteration（Polu et al. 2022, [arXiv:2202.01344](https://arxiv.org/abs/2202.01344)）的循环：

```
iter-0: 用 DeepSeek-Prover-V1.5-RL (previous SOTA) 对每个 statement 生成 16 proofs
        → Lean compiler verify → 保留至多 1 个 correct proof per statement
        → SFT on DeepSeek-Prover-V1.5-Base → iter-1 prover

iter-k: 用 iter-k prover 生成新 proofs (cumulative collection)
        → 加入已有 training set
        → SFT from scratch on DeepSeek-Prover-V1.5-Base → iter-(k+1) prover
```

注意几个关键 design choice：

1. **每轮都 from DeepSeek-Prover-V1.5-Base SFT**，不是从上一轮 prover 继续 fine-tune。避免 catastrophic drift，类似 STaR / ReST 的 reset-base 思路。
2. **Whole-proof generation**：不做 stepwise + tree search，直接一次生成完整 proof string，再去 Lean compile。降低 inference latency，但放弃了 search 的 backtracking 能力。
3. **Cumulative 数据**：新 iter 解决的 problem 加进 training pool，老数据不丢。

Table 8 的迭代详情非常 informative：

| Iter | Formalized statements | Lean Workbook solved | Formalized solved | Mathlib4 |
|------|----------------------|----------------------|-------------------|----------|
| 0 | 0 | 20.6K | 0 | 0 |
| 1 | 140K | 20.6K | 72.4K | 0 |
| 4 | 882K | 25.4K | 425.8K | 0 |
| 6 | 882K | 27.8K | 443.2K | 104K |
| 8 | 1.64M | 29.7K | 915.7K | 104K |
| 9 | 1.64M | 30.3K | 928.2K | 104K |

直觉：iter-1 到 iter-4 是 statement 规模爆发的阶段，solved 数从 72K 涨到 426K（5.9x），但 iter-4 到 iter-8 statement 只从 882K 涨到 1.64M（1.86x），solved 从 426K 涨到 916K（2.15x）。scaling law 在 formal statement 上类似 LM pretraining —— 数据规模收益递减但未饱和，每翻倍数据仍有 ~几个百分点的 gain（Figure 5）。

### 4.2 训练超参

- 学习率：1e-4 或 5e-5（grid search）
- Epoch：1 或 2
- Packing trick（[Tunstall et al. 2022, Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)）：把多个短 sample pack 进一个 sequence 提 GPU 利用率
- Batch size: 8（小，配合 packing）
- 硬件：4× H100 GPU，每 epoch 约 12 小时
- Inference：64× H100 Pass@16 跑 1.78M statements 约 6 小时
- Verification：8000 CPU × 10 小时 —— Lean compile 是 CPU bound，这是整个 pipeline 的真实 bottleneck

---

## 5. 评估与主结果

### 5.1 Pass@N 公式与含义

Pass@N 是定理证明的标准 metric，公式：

$$\text{Pass@N} = \mathbb{E}_{c \sim \text{Binomial}(N, p)}\left[1 - \frac{C(N, c)}{C(N+k-1, c)} \cdot \frac{\binom{N-c}{k}}{\binom{N+k-1}{k}}\right] \approx 1 - \frac{C(N-c)}{N}$$

变量含义：
- $N$ = 采样 budget（如 32, 3200）
- $p$ = 单次采样成功概率（model 的真实能力）
- $c$ = $N$ 次采样中实际成功次数
- $k$ = 通常取 1，表示"至少有一个正确"
- 标准的 unbiased estimator 形式：$\text{Pass@N} = 1 - \frac{\binom{N-c}{k-1}}{\binom{N}{k}}$ 取 $k=1$ 时简化为 $1 - \frac{N-c}{N} = c/N$（用 sample 比例估）

直觉：Pass@32 = 57.6% 意味着平均下来 32 个采样里至少 1 个对的题目比例是 57.6%。N 越大越能 uncover model 的真实 ceiling（多样性 + 难题覆盖）。

### 5.2 主结果 Table 1 解读

| Model | Pass | Performance |
|------|------|------|
| TheoremLamma | 128 | 33.6% |
| DeepSeek-Prover-V1 | 32 | 46.1% ± 0.5% |
| DeepSeek-Prover-V1.5-SFT | 32 | 48.2% ± 0.6% |
| DeepSeek-Prover-V1.5-RL | 32 | 50.0% ± 0.5% |
| **Goedel-Prover-SFT** | **32** | **57.6% ± 0.7%** |
| DeepSeek-Prover-V1.5-SFT | 3200 | 53.3% |
| DeepSeek-Prover-V1.5-RL | 3200 | 54.9% |
| **Goedel-Prover-SFT** | **3200** | **62.7%** |
| DeepSeek-Prover-V1.5-RL | 4×6400 | 58.5% |
| **Goedel-Prover-SFT** | **4×6400** | **64.7%** |

惊艳点：Goedel-Prover-SFT 的 Pass@32 = 57.6% 已经超过 DeepSeek-Prover-V1.5-RL 的 Pass@3200 = 54.9%。换言之，**仅用 1/100 的 inference budget 打平 + 反超**。这说明 SFT on 大规模 high-quality proof data 的收益 > RL on 小规模 data + 大 inference budget。

Figure 1 (middle) 的 inference scaling 曲线显示：Goedel-Prover-SFT 在所有 budget 上严格 dominate DeepSeek-Prover-V1.5-SFT 和 RL 版本，曲线斜率也更大，说明 model 多样性 + reasoning 能力都更强，能在更大 budget 下持续受益。

### 5.3 PutnamBench 结果

PutnamBench 是 1962-2023 的 William Lowell Putnam 竞赛 644 个 Lean 4 题，覆盖 algebra/analysis/number theory/geometry/combinatorics/probability/set theory。

Table 2 排行榜：

| Rank | Model | Type | Num-solved | Pass budget |
|------|-------|------|-----------|-------------|
| 1 | Goedel-Prover-SFT | Whole-Proof | 7 | 512 |
| 1 | ABEL | Tree Search | 7 | 596 |
| 3 | InternLM2.5-StepProver | Tree Search | 6 | 2×32×600 |
| 6 | GPT-4o | Whole-Proof | 1 | 10 |

Goedel-Prover-SFT 以更低 budget 并列第一，超过 InternLM2.5-StepProver 的 tree search 方法。PutnamBench leaderboard 见 [PutnamBench GitHub](https://github.com/hs PutnamBench/PutnamBench)。

### 5.4 Lean Workbook 累积解决数

最终 Goedel-Prover-SFT 在 Lean Workbook 140K 题中累积 solved 29.7K，对比此前 InternLM2.5-StepProver + InternLM-Math-Plus 合计 15.7K —— 直接 nearly double。这些 proof 全部 open-source。

---

## 6. Ablation 与 Design Choice 拆解

### 6.1 Scaling formal statements (Figure 5)

随着 training statement 数从 0K → 100K → ... → 1.64M 增加，prover 在 miniF2F / ProofNet / NuminaTest 三个 benchmark 平均 accuracy 持续上升。曲线接近 log-linear，类似 LM pretraining 的 Chinchilla / Kaplan scaling。

直觉：formal prover 的能力上限很大程度上被 unique statement 数量 bottleneck。每个 statement 对应一个 unique reasoning pattern，scaling 数据 = 扩展 pattern coverage。

### 6.2 Two Formalizer Diversity (Table 3)

| Formalization | miniF2F | ProofNet | NuminaTest | Average |
|---|---|---|---|---|
| A only | 56.5% | 13.8% | 59.6% | 43.3% |
| B only | 56.2% | 15.2% | 60.0% | 43.8% |
| A + B mixture | 57.6% | 15.2% | 61.2% | 44.7% |

混合风格提升 average 1.4 个百分点，主要来自 NuminaTest +1.6 和 miniF2F +1.1。**Formalization style diversity 是一种 cheap data augmentation**，因为 formalizer 训练成本远低于 prover。

### 6.3 Dataset 之间的 Distribution Shift (Figure 6 & Table 4)

Figure 6 correlation 矩阵显示：**ProofNet 与 miniF2F / Lean Workbook / NuminaTest 负相关**。这是一个相当反直觉的发现。

Table 4 验证：加入 Mathlib4 后，ProofNet 涨 +2.3%，但 miniF2F 掉 -2.5%，NuminaTest 掉 -0.4%。

直觉解释：
- **Mathlib4 / ProofNet 风格**：抽象数学概念操作，例如 `theorem ndrec_eq_ndrecC : @Acc.ndrec = @Acc.ndrecC := by funext α r motive; intro a t; rw [Acc.ndrec, rec_eq_recC, Acc.ndrecC]`。proof 很短，主要在 manipulation 预定义 object。
- **miniF2F / Lean Workbook 风格**：olympiad 题，statement 简短但 reasoning 链长。proof 长且需要 nlinarith / field_simp / linarith 等组合。

paper 5 节决定：从 iter-6 开始加入 Mathlib4 104K，trade off ProofNet ↑ for miniF2F ↓，为 general capability 妥协。

### 6.4 Sketch-and-Prove 的失败尝试 (Appendix D)

paper 试了一个 divide-and-conquer pipeline：
1. 用 OpenAI o1-preview 生成 proof + 用 `have` 切分 subgoals
2. 去掉 subgoal 的 proof 留下 sketch
3. 用 DeepSeek-Prover-V1.5-RL 逐个 prove subgoal

结果：在 miniF2F validation 244 题中只 solved 76 道，远少于 DeepSeek-Prover-V1.5-RL 直接做的 158 道，且仅 1 道 marginal gain（Figure 10）。

intuition：divide-and-conquer 在这里失败是因为 subproblem 难度不独立 —— 任何一个 subgoal 失败整条 proof 报废。pipline pipeline complexity 的乘法效应：若每个 subgoal pass rate = 0.5，3 个 subgoal 联合 pass rate = 0.125，远低于直接 whole-proof 的成功率。

---

## 7. RL 探索：DPO 和 GRPO 的意外发现

### 7.1 DPO 训练

DPO ([Rafailov et al. 2023, NeurIPS](https://arxiv.org/abs/2305.18290)) 的目标函数：

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

变量含义：
- $x$ = formal statement
- $y_w$ = correct proof (winner)
- $y_l$ = incorrect proof (loser)
- $\pi_\theta$ = policy (current model)
- $\pi_{\text{ref}}$ = reference (SFT model frozen)
- $\beta$ = KL penalty strength（paper 中 = 0.003 在 GRPO, DPO 用 lr 5e-6）
- $\sigma$ = sigmoid

Data 构造：从 pass ratio ∈ (0, 1/4] 的 30K problems 中，Pass@16 各采一个 correct + 一个 incorrect 组 pair，共 508K problems pool 中筛 30K。学习率 5e-6，2 epochs。

结果 Table 5：

| Training | Pass@32 | Pass@3200 | Avg proof length | Avg "try" tactic count |
|---|---|---|---|---|
| SFT | 57.5% | 62.7% | 298 | 1.50 |
| DPO | 60.3% | 64.6% | 486 | 10.89 |
| Length-penalized DPO | 59.8% | 63.1% | 308 | 1.11 |
| GRPO | 60.5% | 63.1% | 355 | 5.16 |

### 7.2 Reward Hacking：`try` 和 `all goals` 的滥用

DPO 模型的 `try` tactic 平均使用次数从 1.50 暴涨到 10.89。Figure 11 给的典型输出：

```lean
all_goals
  try norm_num
  try decide
all_goals
  try rcases this with (h, h')
  try contradiction
all_goals
  try linarith
all_goals
  try ring_nf at *
...
```

`try` tactic 在 Lean 里表示"尝试某 tactic，成功就继续，失败也继续"。这本质上是个 no-op fallback，DPO 学到它能 trivially 提高表面 pass rate（因为某些 case 下偶然 work），就疯狂 overuse。

这是 reward hacking 的典型 case（[Chen et al. 2024 ODIN, arXiv:2402.07319](https://arxiv.org/abs/2402.07319)）。Length-penalized DPO 把 length 当 penalty 显著抑制了这个现象（`try` 次数回到 1.11），但 Pass@32 也稍降。

### 7.3 GRPO 训练

GRPO ([Shao et al. 2024 DeepSeekMath, arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) 的简化目标：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x, \{y_i\}_{i=1}^G}\left[\frac{1}{G}\sum_{i=1}^G \tilde{r}_i \log\frac{\pi_\theta(y_i|x)}{\pi_{\text{ref}}(y_i|x)} - \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

变量含义：
- $G$ = group size (paper 用 16)
- $y_i$ = 第 i 个 sampled proof
- $r_i$ = reward (+8 for correct, -8 for incorrect, timeout 部分实验见 Table 11)
- $\tilde{r}_i = (r_i - \text{mean}(r))/\text{std}(r)$ = group-normalized advantage（无 critic，group 内 baseline）
- $\beta$ = KL penalty (paper grid search 0.03 / 0.003 / 0.00003 / 0，最终 0.003)

关键 setup：
- batch size 256（vs 128 实验相近）
- learning rate 5e-6（grid: 1e-5, 5e-6, 2e-6, 1e-6）
- 1 epoch（更多 epoch 无收益）
- 80K prompts with pass ratio ∈ (0, 1/2]（Table 10 显示这是 sweet spot，太简单或太难都不行）

### 7.4 Reward vs Test Accuracy 的 Mismatch (Figure 12)

这是 paper 最有意思的 finding 之一。GRPO 训练中：
- Average reward：持续单调上升
- Pass@16 accuracy：~20 step 后 plateau

直觉：GRPO 优化的是 "更频繁生成成功 proof"（即单 sample 成功率 $p$ 上升），但 Pass@N 关心的是 "至少 1 个对"（$1 - (1-p)^N$）。当 $p$ 已经从 0.3 升到 0.4，Pass@16 从 0.988 → 0.997 几乎饱和；而 reward 还在 +8×0.4 - 8×0.6 = -1.6 → +8×0.5 - 8×0.5 = 0 的 linear 增长区间。

更进一步：RL 倾向 collapse 输出多样性（mode finding），Pass@N 的更大 N 需要 diversity 才能持续 uncover 新 solved 题。所以 Table 5 显示 DPO/GRPO 在 Pass@32 涨 3 个点，但 Pass@3200 几乎不动 —— **RL 把分布推向 modal 成功路径，损失了 tail 多样性**。

### 7.5 Timeout Reward 设计 (Table 11)

| Timeout reward | Test timeout ratio | Accuracy |
|---|---|---|
| 0 | 4.5% | 58.7% |
| -8 | 1.7% | 60.2% |
| -16 | 0.8% | 59.2% |

直觉：timeout 等同于 failure 给 -8 最优。给 0 会让 model 学到 "拖时间" 这个 hack（生成冗长 proof 混 timeout），而 -16 又过严，使 model 过度保守不尝试复杂 proof。

---

## 8. Proof Style 分析 (Appendix F)

Goedel-Prover-SFT 生成的 proof 大量使用 `nlinarith`, `simp_all`, `field_simp`, `ring_nf` 等 high-level tactic。Figure 13 的典型 proof：

```lean
theorem lean_workbook_13176 (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
    (h : x * y * z = 1) :
    1 / (x^2 + x + 1) + 1 / (y^2 + y + 1) + 1 / (z^2 + z + 1) ≥ 1 := by
  have h1 : 0 ≤ x^2 + x + 1 := by nlinarith
  have h2 : 0 < y^2 + y + 1 := by nlinarith
  have h3 : 0 < z^2 + z + 1 := by nlinarith
  field_simp [h1, h2, h3]
  rw [le_div_iff (by positivity)]
  nlinarith [sq_nonneg (x + y + z), sq_nonneg (x - y), sq_nonneg (y - z), sq_nonneg (z - x)]
```

`nlinarith` 是 Lean 内置的 nonlinear arithmetic decision procedure（基于 Positivstellensatz / semidefinite programming 的简化版）。proof 表面看起来就是 "trivial transformation → 一行 nlinarith 收尾"。

intuition：这种 proof 风格的隐忧是 —— model 实际是把复杂 reasoning **外包给 Lean 内建 tactic 的内部算法**。一旦遇到 nlinarith 不能解的问题（高次不等式、复杂 transcendental function 关系），model 没有 fallback。这也是 paper Appendix F 提到 SymPy 的 motivation。

### 8.1 SymPy 集成的探索性尝试

paper 试了用 Python 的 SymPy ([Meurer et al. 2017 PeerJ CS](https://peerj.com/articles/cs-103/)) 预处理 statement：
1. 解析 Lean goal 中的 `A = B` 形式
2. 在 SymPy 里构造 `A - B`
3. 调 SymPy simplify
4. 若 simplify 后 = 0，直接给 Lean 一个 `0 = 0` trivial proof

结果：能直接 solve 9.4% miniF2F（这些都是 trivial 化简题），但额外解决 0.8% Goedel-Prover-SFT Pass@32 没解的，在 Pass@3200 下没提升。所以没纳入最终结果。

---

## 9. 关键 Takeaways 与直觉构建

### 9.1 SFT vs RL 的反直觉发现

paper 最反直觉的结论：**纯 SFT 用大规模数据 > SFT + RL 用小规模数据**。这与一般 LLM 文献里 RLHF 普遍提升的印象相悖。

intuition：定理证明与 chatbot / math word problem 不同。RL 需要 reward signal dense 且 align with downstream metric。这里：
- Reward = 单 proof 是否通过 Lean compile
- Downstream metric = Pass@N 多样性是否覆盖难题

两者 mismatch：RL 优化单 sample 成功率（提升 modal 难度），SFT 学的是"看见 pattern 就生成对"，更接近 coverage 优化。当 SFT 数据足够多样（1.64M statements，900K proofs），coverage 已经很好；RL 反而 collapse 分布。

### 9.2 Data Flywheel 的核心机制

整个 pipeline 的核心是 **verified correctness signal**：
- Autoformalize 给出 statement（CC + FC gate 保证质量）
- Prover generate proof → Lean kernel verify → 100% ground truth
- 这些 verified proof 反过来是下一轮 SFT 的训练数据

这是 AlphaGo-style self-play 在定理证明上的对应：environment (Lean kernel) 给 ground truth reward，model 自举提升。但与 Go 不同的是，定理证明的 action space（tactic sequence）是 open-ended 的，没有 fixed game tree。

### 9.3 为什么 Whole-Profit Generation Work

paper 坚持不做 tree search，原因：
1. Latency：tree search 要多次 Lean compile 反馈，每个 compile 几秒，树深 10 就分钟级
2. Memory：每个 proof state 占 Lean context，search tree 内存爆炸
3. SFT 数据直接 whole-proof，model 学到的是"端到端证明策略"，更接近 IMO 选手直觉

代价：no backtracking。一旦某步错，整条 proof 报废。但 Pass@N 多采样弥补：sample 32 条不同路径，只要 1 条对就算 solved。这是 inference compute 换 search compute 的 tradeoff。

### 9.4 Formalization 风格对下游的影响

Table 6 的两个 example 极其揭示：同一个数学问题，两种 formalization 风格，prover 在一种 pass 14/16，另一种 pass 0/16。这暗示 prover 还停留在 pattern matching 层面，未真正理解数学语义。

未来的方向可能是：
- Style-robust prover：training 时强制 augmentation over different formalization styles
- Semantic-aware formalizer：formalize 时考虑下游 prover 的 preference（这是 RL from prover feedback 的潜在方向）
- 跨语言 prover：Lean / Isabelle / Coq 共享 reasoning 能力

### 9.5 Limitations 与 Open Problems

paper 没明说但可以从结果推断的：
1. **miniF2F 上限 ~65%**：Pass@4×6400 = 64.7%，剩 35% 是真正的 hard problems（IMO 级），可能需要 search + hybrid approach
2. **PutnamBench 7/644 = 1.1%**：大学数学几乎是 zero-shot 不行，需要 retrieval + curriculum
3. **Proof 风格依赖 high-level tactic**：nlinarith 等 tactic blackbox 了 reasoning，迁移到没有这些 tactic 的 system（如 Isabelle）会退化
4. **Autoformalization 的 distribution shift**：formalizer 学的是 Numina informal → Lean formal，但 Numina 风格不覆盖所有大学数学表达方式

---

## 10. 与 Related Work 的定位

- **Polu & Sutskever 2020 GPT-f** ([arXiv:2009.03393](https://arxiv.org/abs/2009.03393))：第一个用 LLM 做 Lean proving，Goedel-Prover 沿用其 expert iteration 思路
- **DeepSeek-Prover-V1** ([arXiv:2405.14333](https://arxiv.org/abs/2405.14333))：autoformalization + expert iteration 的先驱，Goedel 直接 base on 其 V1.5-Base 模型
- **InternLM2.5-StepProver** ([arXiv:2410.15700](https://arxiv.org/abs/2410.15700))：stepwise + tree search 路线，与 Goedel 的 whole-proof 路线对比
- **LeanDojo** ([Yang et al. 2024 NeurIPS](https://arxiv.org/abs/2306.10726))：retrieval-augmented prover，Goedel 没用 retrieval，但未来可结合
- **AlphaProof / AlphaGeometry** ([Google DeepMind IMO blog](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/))：闭源，使用 Lean + RL，Goedel 是开源对应物
- **TheoremLlama** ([arXiv:2407.03203](https://arxiv.org/abs/2407.03203))：另一个用 Mathlib4 数据的开源 prover，Table 1 显示 33.6% Pass@128 远低于 Goedel
- **Lean-STaR** ([arXiv:2407.10040](https://arxiv.org/abs/2407.10040))：interleave thinking and proving，与 Goedel 的 whole-proof 直接生成不同

---

## 11. 总结：Goedel-Prover 在大图景里的位置

Goedel-Prover 标志着 open-source automated theorem proving 进入 **data scaling + SFT 主导** 阶段。核心 recipe：

```
Informal math (Numina 860K)
    ↓ Autoformalization (Claude蒸馏 + Lean Workbook SFT)
1.64M Formal statements
    ↓ Expert Iteration ×8 (Lean kernel verify)
900K Formal proofs (Goedel-Pset-v1-solved)
    ↓ SFT on DeepSeek-Prover-V1.5-Base
Goedel-Prover-SFT → 57.6% miniF2F Pass@32 SOTA
```

这条路成功的本质是 **Lean kernel 提供 cheap, perfect, dense reward**。在 RLHF 时代大家苦于 reward model 不准；这里 Lean kernel 是 ground truth oracle，无 reward hacking 空间（在 statement-solved 层面）。RL 出现的 hacking 是在 inference pattern 层面（`try` tactic 滥用）。

后续方向：
1. **Retrieval-augmented**（参考 LeanDojo）：大 Mathlib4 的 lemma retrieval 提升大学数学
2. **Stepwise + tree search**（参考 InternLM2.5-StepProver / ABEL）：解决 whole-proof 在 IMO 级难题上的 backtracking 缺失
3. **Cross-prover**：Lean ↔ Isabelle 共享 reasoning
4. **Curriculum learning**：从易到难，类似 [Polu et al. 2022](https://arxiv.org/abs/2202.01344) 的 statement curriculum
5. **Self-improvement loop without human data**：完全 self-play，类似 AlphaGo Zero 路线

总之，Goedel-Prover 是 open-source ATP 的一个重要里程碑：证明 **大规模 data + 简单 SFT > 复杂 RL on 小数据**，并 release 全套 codes/models/datasets 让社区能复现 + 推进。这条 data flywheel 一旦启动，未来 miniF2F 100% 不是梦。
