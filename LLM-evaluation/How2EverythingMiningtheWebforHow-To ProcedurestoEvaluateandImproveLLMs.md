---
source_pdf: How2EverythingMiningtheWebforHow-To ProcedurestoEvaluateandImproveLLMs.pdf
paper_sha256: 507a8350a3ee33ef4e29a980ca770e21758cb39639925f454dcabd2698ead542
processed_at: '2026-08-05T07:22:39-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# How2Everything 人话版

链接放前面方便查：
- Paper: https://arxiv.org/abs/2507.15532
- Code: https://github.com/allenai/how2everything
- How2Bench/How2Train: https://huggingface.co/datasets/how2everything
- WebOrganizer: https://openreview.net/forum?id=boSqwdvJVC
- DCLM: https://arxiv.org/abs/2406.11794
- GRPO: https://arxiv.org/abs/2402.03300
- PRM (Let's verify step by step): https://arxiv.org/abs/2305.20050

---

## 1. 这篇 paper 到底想解决什么问题

假设你问 ChatGPT "how to sell my apartment share" 或者 "how to repair a faucet"，模型给一堆 steps。你怎么知道这堆 steps 是真的能达成 goal 的？你怎么 scale 评测到 thousands of examples？

如果这是数学题，简单——run 代码看答案对不对。如果这是 code，简单——跑 unit test。但 "how to file taxes" 这种 task 根本没有 executor，没法 execute。

以前的做法有几种，每种都有问题：

| 做法 | 问题 |
|---|---|
| 只看 WikiHow 上那几个特定 source | topic 窄、domain 偏 |
| 只做 cooking recipe | 太 niche，不 general |
| 用 BLEU 等 string overlap | 根本不反映 procedural validity |
| 用 human eval | 准但太贵，没法 scale |
| 用 GPT-4 当 judge | 贵，不可复现 |
| 数学 PRM 那套 first-failure localization | open-world steps 不可执行，没法定位"first wrong step" |

paper 想做的事：**用 web 上天然存在的 how-to 文档作为 reference anchor，把"validity"这件事变得 scalable 可测，且把 reward 蒸馏到开源小模型上，让 RL 能用**。

---

## 2. 核心 idea 的三个 moves

把整个 paper 压缩成三句话：

1. **Web 是 reference 的 unbounded supply**。execution-free 设定下没有 oracle，但 web 上有海量真实人写的 how-to 文档，可以当 anchor。
2. **Critical failure detection 是 validity 的 proxy**。放弃"step-by-step correctness localization"，改问 binary 问题："这堆 steps 里有没有一个 critical failure 让 goal 达不成？"——这个 binary question 的 inter-annotator agreement 比 first-failure localization 高得多（α=0.593 vs 0.307）。
3. **Distillation 让 reward 可复现**。把 GPT 5 的判断蒸馏到 Qwen 3 8B（How2Judge），agreement 80.5%，让大规模 eval 和 RL reward 都能 cheap + reproducible。

整个 framework 叫 How2Everything，里面 5 个 component：

| Component | 作用 |
|---|---|
| How2Mine | 从 web 挖 how-to procedures |
| How2Bench | 7K evaluation set |
| How2Score | critical failure detection 协议 |
| How2Judge | distilled 8B judge，让 How2Score cheap |
| How2Train | 100K 用来 RL training |

---

## 3. How2Mine: 怎么从 web 挖数据

### 3.1 数据源

DCLM web corpus（Li et al. 2025）。用 WebOrganizer（Wettig et al. 2025）两层 classifier 筛：
- **Format filter**：只保留 "Tutorial & How-to Guide" 标签的文档
- **Topic stratification**：跨 14 个 topic 平衡采样

为什么挑 tutorial 格式？Table 6 的实验：tutorial 类文档经过三阶段过滤后还剩 24.46% valid procedures，其他 format（personal blog / Q&A forum / academic writing）只剩 14-17%。

最终：980K 文档 → 351K structured procedures，跨 189K unique domains。

### 3.2 五阶段 pipeline

每阶段都有 reason，不是随便堆 filter 的。

**Stage 1: Procedure Extraction**

用 GPT-4.1 从文档抽 (goal, ordered steps)。要求 sequential、imperative、atomic、concrete。这个阶段 980K 文档中 ~87% 被认为含 valid procedure（在 tutorial 格式下）。

**Stage 2: Heuristics Filter**

- Step count: 5 ≤ steps ≤ 15
- N-gram repetition: 2-gram ≥0.40, 3-gram ≥0.35, 4-gram ≥0.30 就 reject

主要去除 degenerate extraction。

**Stage 3: LLM Filter**

排除 6 类不适合的 procedure。这个 stage 是 24.46% vs 70.46% 的卡口，是最严的过滤。每类都有清楚的理由：

- **Named-entity focused**：比如 "在 Microsoft Excel 做 pivot table"——正确性依赖特定软件，没 entity context 没法判断
- **Pure math**：math 的 correctness 是 numerical result，不是 instruction validity，应该用 math benchmark 测
- **UI interaction**：UI 会变化、state 隐含，execution-free 设定下不可验证
- **Open-ended creative**：比如 "写首秋天的诗"——endpoint 多种 valid，边界由 taste 决定
- **Non-sequential**：listicle 类型，steps 顺序无关，对 step-by-step validity 评估 ill-posed
- **Unreasonable**：内部矛盾或不能达成 goal，作为 reference 会污染下游

**Stage 4: Post-processing + Resource Extraction**

两件事：
1. Rewrite goal 让它 specific 且 deterministic——明确 constraints 和 expected outcome，缩小 valid solution 空间
2. 提取 resources list（tools/ingredients/equipment）

为什么 resources 重要？同一个 goal 可能有多条 valid procedure，列出 resources 帮 narrow solution space。比如 goal "做 spiced glazed nuts with a Crock Pot"——不写 resources 的话，模型可能用 oven 做，paper 不能简单判它错。但 reference 是 Crock Pot-based，resources list 里也有 Crock Pot，那模型用 oven 就是 deviation。

**Stage 5: Final Validation**

LLM sanity check，再清一遍 nonsensical examples。

### 3.3 成本

980K 文档，252K requests via OpenAI batch API，$5,717。比人工 curation 便宜几个数量级，且 scalable 到更大 corpus。

---

## 4. How2Score: 评测的核心

这是 paper 最有创新性的部分。

### 4.1 为什么不能直接用 math PRM 的思路

math PRM（Lightman et al. 2023）找"first incorrect step"作为 failure point，因为 math steps 可执行、可验证。但 open-world procedure 不可执行——比如 "Wait 30 days for co-owners to respond" 这种 step，没法自动判断它是否被正确执行。

paper 测了一下：让 human annotators 找 first failure 的 Krippendorf's α = 0.307，让 human annotators 只回答"有没有 critical failure"的 α = 0.593。Binary 问法的 agreement 显著更高，所以采用 binary。

### 4.2 Critical Failure 的定义

> An omission, extraneous action, contradiction, severe vagueness, or other deviation from the reference that is severe enough to prevent achieving the goal, or to make the procedure unusable as instructions.

关键 nuance：
- 用 reference 作为 anchor，但 **不**惩罚 alternative valid procedures
- 例：goal 是做 terracotta pot 当 gift，gift message 写啥不算 critical failure

Table 1 给的两个 example 很直观：

**Example 1 (Crime & Law)**：Gemini 2.5 Pro 输出 sell apartment share 的 steps，跳过了 "wait 30 days for co-owners to respond" 这一步。这是 critical failure——法律上少了这个 waiting period 整个 procedure 就无效。

**Example 2 (Food & Dining)**：Qwen 3 8B Instruct 输出做 spiced glazed nuts 的 steps，但没指定 Crock Pot 用 high 还是 low、lid 开还是关。这是 severe vagueness——没法可靠执行。

### 4.3 形式化

给定 evaluation set $D$，每个 example $x = (g, R, S^*, \hat{S})$：

- $g$ = goal
- $R$ = extracted resource list
- $S^*$ = reference procedure
- $\hat{S}$ = model-generated procedure

LLM judge $J(\cdot)$ 输出 binary label：`has_failure` if 至少一个 critical failure，否则 `no_failure`。

$$\text{Score}(D) = \frac{1}{|D|} \sum_{x \in D} \mathbb{I}[J(g, R, S^*, \hat{S}) = \text{no\_failure}]$$

变量解释：
- $|D|$：evaluation set 大小
- $\mathbb{I}[\cdot]$：indicator function，条件成立为 1 否则 0
- $J(\cdot)$：judge function，返回 `no_failure` 或 `has_failure`

报告 success rate = no_failure 的比例。

虽然 binary 报告，paper 仍然让 judge 列出所有 critical failures + 对应的 reference steps 和 generation steps，方便 transparency。

### 4.4 Human Annotation

- 3 个 Prolific annotators
- 200 examples（预筛掉需要 specialized domain knowledge 的）
- 时薪 $28，total $3,600
- 多轮 pilot：initial α=0.273 → final α=0.593

annotator training 很关键。早期 pilot 中 annotator 要么把任何和 reference 的 difference 标 critical，要么被 coherent surface form 误导忽略 real failure。所以 training materials 反复 clarify 边界。

### 4.5 LLM Judge 对比 Human

Figure 3：5 个 LLM judge 对比 human majority：

| Judge | Overall agreement | has_failure | no_failure |
|---|---|---|---|
| GPT 5 | 83.0% | 83.7% | 82.4% |
| Gemini 2.5 Pro | ~80% | - | - |
| Claude Opus 4.5 | ~80% | - | - |
| Gemini 3 Pro Preview | ~80% | - | - |
| GPT 5.1 | ~80% | - | - |

Human leave-one-out agreement: 84.7%-88.5%。GPT 5 落在 human annotator 范围内。

### 4.6 How2Judge: 蒸馏到 8B

GPT 5 评 7K examples 要 $15，太贵且不可复现。所以蒸馏：

**Training data**：73K GPT 5 annotations，generators 包括 Gemini 2.5 flash/pro, GPT-4.1, GPT 5, Qwen2.5-7B-inst, OLMo-2 1B/7B/32B 各 stage。每个 example 跑 GPT 5 两次，只保留 binary judgment 一致的（denoise）。

**Training**：Qwen 3 8B，3 epochs，lr=5e-6，batch size=64。

**结果**：
- 90.5% agreement with GPT 5
- 80.5% agreement with human majority
- 79.6% on has_failure, 81.4% on no_failure（balanced）

80.5% 接近 GPT 5 的 83.0%，但 cost 接近零、完全 reproducible。这是闭环的关键。

---

## 5. How2Bench: 评测结果

How2Bench = 7K examples（每 topic 500）。inference 时模型收到 (goal, resources, n)，被要求输出 exactly n steps。

### 5.1 Length Control

LLM judge 普遍有 length bias——模型写更长容易得更高分。How2Bench 的应对：

- 3-shot prompt，例子展示 expected format 和 detail level
- 强制每 step 是 single concise sentence with one main action
- 要求模型 closely follow examples 的 concision level

Figure 12 显示加这 control 后 generation length 分布显著接近 reference distribution。

### 5.2 Scaling Trend (Table 8)

跨 OLMo 2/3 的 1B/7B/32B 各 pretraining checkpoint 跑 How2Bench。挑几个关键点：

| Suite | Size | Stage | Step | How2Score |
|---|---|---|---|---|
| OLMo-2-0425 | 1B | Pretrain step 20K | 0.06 |
| OLMo-2-0425 | 1B | Pretrain step 1.9M | 1.59 |
| OLMo-2-0425 | 1B | Midtrain | 6.39 |
| OLMo-2-0425 | 1B | Posttrain | 5.96 |
| OLMo-3-1025 | 7B | Pretrain step 14K | 4.13 |
| OLMo-3-1025 | 7B | Pretrain step 1.4M | 21.59 |
| OLMo-3-1025 | 7B | Midtrain | 24.91 |
| OLMo-3-1025 | 7B | Posttrain | 30.23 |
| OLMo-3-1125 | 32B | Pretrain step 6K | 6.21 |
| OLMo-3-1125 | 32B | Pretrain step 579K | 31.00 |
| OLMo-3-1125 | 32B | Midtrain | 38.31 |
| OLMo-3-1125 | 32B | Posttrain | 43.16 |

观察：
- 1B model 在 ~10^21 FLOPs（约 5% pretraining）开始出现 non-trivial performance
- 跨 stage（pretrain → midtrain → posttrain）单调提升
- 跨 size（1B → 7B → 32B）单调提升

这是 benchmark 的好性质——能 rank models across compute budgets，不像许多 benchmark 要么早早 saturate 要么在小 model 上 near-zero。

### 5.3 Format vs Validity 的解耦 (Table 9)

跨 checkpoints 跟踪三个 formatting proxy：
- Step-count mismatch
- Duplicate-step frequency
- Dup n-gram rate

观察：formatting errors 在 early pretraining 快速 plateau，但 How2Score 继续提升。

**Intuition**：surface format 早期就稳定了，后期 How2Score 的 gains 反映的是 end-to-end procedural validity 的真实提升，不是修表面格式。

### 5.4 How2Score 不是 Perplexity (Table 16)

每个 OLMo trajectory 内，比较 checkpoint 按 How2Score 排序 vs 按条件 PPL 排序：

| Trajectory | Spearman ρ |
|---|---|
| OLMo-2-0425 (1B) | 0.917 |
| OLMo-2-1124 (7B) | 0.667 |
| OLMo-2-0325 (32B) | 0.233 |
| OLMo-3-1025 (7B) | 0.850 |
| OLMo-3-1125 (32B) | 0.483 |

1B 两者高度相关（因为 1B 还在学 reference 的 surface form），32B 相关性低到 0.233。这表明对 strong model，How2Score 测的不再是"是否见过 reference"，而是 validity。

### 5.5 Step Count 是 Difficulty Knob (Table 10)

logistic regression 预测 no_failure：

$$\text{logit}(p(\text{no\_failure})) = \beta_0 + \beta_{\text{steps}} |S^*| + \beta_{\text{res}} |R| + \beta_{\text{ratio}} \rho + \sum_{t \neq t_0} \gamma_t \mathbb{I}[\text{topic} = t]$$

变量：
- $|S^*|$：reference step count（也等于 requested output length）
- $|R|$：reference resource count
- $\rho = 100 \cdot |\text{gen}|/|\text{ref}|$：generation/reference token ratio（百分比）
- $T$：14 topics，$t_0$ = Art & Design 作 baseline
- $\gamma_t$：topic 固定效应

报告 odds ratios $\text{OR} = \exp(\beta)$：

| Model | OR per +1 step | OR per +1 resource | OR per +1pp gen/ref |
|---|---|---|---|
| OLMo-3-7B-Inst | 0.756 | 1.009 | 1.012 |
| Qwen3-8B-Inst | 0.737 | 1.020 | 1.015 |
| OLMo-3.1-32B-Inst | 0.751 | 1.043 | 1.013 |
| Qwen3-32B-Inst | 0.765 | 1.018 | 1.014 |
| Gemini-2.5-Pro | 0.795 | 1.062 | 1.018 |
| Claude-Opus-4.5 | 0.813 | 1.060 | 1.017 |
| GPT 5 | 0.846 | 1.022 | 1.014 |
| Overall | 0.803 | 1.032 | 1.015 |

**解读**：
- OR_steps ≈ 0.74-0.85：每多 1 step，no_failure 的 odds 下降 15-26%。Step count 是 dominant predictor。这很合理——更多 steps 意味着更多出错机会。
- OR_ratio ≈ 1.01-1.02 per +1pp：即使有 explicit length control，residual verbose bias 还在。这就是为什么 paper 坚持 report avg generated tokens alongside score。
- Frontier model 系数更接近 1（ceiling effect：高分时 residual variance 少，简单 predictor 解释力下降）。

### 5.6 Topic Effects (Figure 13)

控制 $|S^*|$、$|R|$、$\rho$ 后，topic 仍然有大的 fixed effects，且 cross-model 一致：
- Education & Jobs、Food & Dining 比 baseline 容易
- Electronics & Hardware、Industrial 比 baseline 难

**Intuition**：Electronics/Industrial 涉及更多 critical parameters（电压、时间、温度）和精确顺序依赖，错了就 critical failure。Education/Food 更 forgiving，alternative procedure 也常 valid。

---

## 6. RL 训练

### 6.1 Setup

- 训练数据：100K from How2Train，跨 14 topic 平衡，embedding similarity filter 与 How2Bench 去重（τ=0.65，Qwen3-Embedding-0.6B）
- Algorithm：GRPO（Shao et al. 2024）
- 1000 optimizer steps
- lr = 5e-7
- 4 prompts/batch，8 completions/prompt
- 三个 model：Qwen3-4B-Inst, Qwen3-8B-Inst, OLMo-3-7B-Think

### 6.2 三个 Reward 组件

GRPO 的 reward 是三者之和。

**Reward 1: How2Score from How2Judge**

Binary 0/1。

**Reward 2: Step-format verifier**

检查 final answer 是连续编号列表、step count 匹配。Return 1 if 满足，0 otherwise。

**Reward 3: Reference-calibrated length reward**

防止 length gaming。设 $r = |\text{gen}|/|\text{ref}|$，$\tau=0.2$，$\alpha=5$：

$$R_{\text{len}}(r) = \begin{cases} 1 & |r-1| \leq \tau \\ \exp\left(-\alpha \cdot \frac{|r-1| - \tau}{1 - \tau}\right) & \text{otherwise}\end{cases}$$

变量：
- $r$：generation 与 reference 的 token 长度比
- $\tau$：tolerance band，允许 ±20% 偏差
- $\alpha$：超出 tolerance 后的衰减率

**Intuition**：LLM judge 普遍有 verbose bias。如果不控制 length，模型会膨胀长度骗 reward。Table 12 证实：

| Model | RL reward | How2Bench | Avg gen/ref |
|---|---|---|---|
| Qwen3-4B-Inst | + length reward | 43.52 | 1.01 |
| Qwen3-4B-Inst | no length reward | 54.41 | 1.34 |
| Qwen3-8B-Inst | + length reward | 48.62 | 1.00 |
| Qwen3-8B-Inst | no length reward | 67.00 | 1.53 |

去掉 length reward，How2Score 涨了 11 points，但 generation 膨胀 34-53%。这 11 points 是虚的。

### 6.3 主要结果 (Table 2)

| Benchmark | Qwen3-4B-Inst ∆ | Qwen3-8B-Inst ∆ | OLMo-3-7B-Think ∆ |
|---|---|---|---|
| **How2Bench** | **+13.23** | **+10.10** | **+10.59** |
| MMLU-Pro | +1.54 | +0.95 | +5.07 |
| GPQA | -0.23 | -0.23 | +0.89 |
| ZebraLogic | -1.2 | +0.5 | -2.3 |
| AlpacaEval | +2.95 | +0.32 | +1.44 |
| HumanEval+ | +3.48 | -1.71 | -1.04 |
| LiveCodeBench | -0.22 | -0.21 | -2.45 |
| MBPP+ | -0.48 | +0.66 | -0.52 |
| GSM8K | -0.31 | -0.38 | +0.38 |
| Minerva | +0.07 | +0.72 | +0.18 |
| Omega | -2.8 | 0.00 | +2.4 |
| AIME24 | 0.00 | -2.09 | +3.13 |
| AIME25 | +3.44 | +2.19 | +5.42 |
| $\overline{\Delta}_{\text{OOD}}$ | +0.52 | +0.06 | +1.05 |

观察：
- How2Bench 一致 +10 以上
- 12 个 OOD benchmark 平均小幅正向（+0.52, +0.06, +1.05）
- 个别 benchmark 退化，但没 systematic regression
- OLMo-3-7B-Think 在 AIME25 上 +5.42——RL on procedural data 居然 improve 数学竞赛，暗示 procedure generation 和 multi-step reasoning 共享 latent skill

### 6.4 SFT 失败 (Table 11)

SFT 在 instruct checkpoint 上无效甚至有害：

| Model | Stage | Before | After | ∆ |
|---|---|---|---|---|
| Qwen3-4B | Base | 32.00 | 33.11 | +1.11 |
| Qwen3-4B | Instruct | 29.70 | 28.47 | -1.23 |
| Qwen3-8B | Base | 35.54 | 35.20 | -0.34 |
| Qwen3-8B | Instruct | 35.34 | 32.45 | -2.89 |
| OLMo-3-7B | Base | 24.91 | 26.13 | +1.22 |
| OLMo-3-7B | Instruct | 30.23 | 22.07 | **-8.16** |

**为什么 SFT 有害**：SFT 的 objective 是 maximum likelihood of single reference per goal。但 How2Score reward 任何 valid procedure 只要 no critical failure。这是 **objective mismatch**（Stiennon et al. 2022, Xie et al. 2025）。Imitate 单个 reference 不必然减少 critical failures，反而 overfit 到一种 specific realization，破坏 general procedural capability。

### 6.5 External Judge 验证 (Table 13)

用 GPT 5 和 Gemini 2.5 Pro 重新评测 RL 后的模型：

| Model | How2Judge ∆ | GPT 5 judge ∆ | Gemini judge ∆ |
|---|---|---|---|
| Qwen3-4B-Inst | +13.23 (43.69%) | +9.15 (33.72%) | +9.17 (58.58%) |
| Qwen3-8B-Inst | +10.10 (26.23%) | +8.76 (26.84%) | +8.03 (39.94%) |
| OLMo-3-7B-Think | +10.58 (38.77%) | +11.09 (53.74%) | +6.77 (50.05%) |

Gains 在外部 judge 下持续存在——不是 How2Judge 的 artifact。

---

## 7. 排除 Confounds

paper 最严谨的地方——主动 stress-test 两个 confound。

### 7.1 Confound 1: Implicit Task Format

如果 gains 只是学了"输出格式"，那么从弱 pretraining checkpoint 也应该得到类似 gains。

**Diagnostic 1: Pretraining Maturity (Figure 6)**

固定 post-training recipe (SFT + RL)，从不同 OLMo 3 7B pretraining checkpoint 出发：
- SFT gains：3.39 到 5.36（基本不变）
- RL gains：3.56 at 10% pretraining → 20.33 at 100% pretraining

**Intuition**：SFT 主要塑造 surface behavior（任何 pretraining level 都能学 format），RL 才放大 pretrained capabilities（Ouyang et al. 2022, Zhao et al. 2025）。RL gains 随 pretraining FLOPs 增长，说明 gains driven by underlying capability 而非 format learning。

**Diagnostic 2: Data Topic Coverage (Table 15)**

用 Qwen3-8B 做 topic-restricted RL：

| Train | Overall ∆ | Art & Design | Crime & Law | Education & Jobs | ... | Food & Dining |
|---|---|---|---|---|---|---|
| All topics | +10.10 | +8.84 | +15.65 | +10.36 | ... | +9.94 |
| Science only | +9.41 | +11.24 | +14.94 | +11.56 | ... | +7.53 |
| Dining only | +5.55 | +2.74 | +8.72 | +7.96 | ... | +8.13 |

Science topic 在 embedding space 分散（Figure 14 PCA），dining 是 specialized cluster。

**Intuition**：如果 gains 纯 format-driven，各 train set 应该 transfer 一样好。但 broad coverage yields largest gains，说明 improvement driven by content coverage 不是 generic format。Science-only 比 dining-only transfer 更广，因为 Science embedding 分散覆盖更多 semantic space。

### 7.2 Confound 2: Memorization

How2Bench examples 来自 web，LLM 也 pretrain on web。可能 memorization confound。

**实验设计**：
- Midtraining 时控制文档曝光频率 (0, 1, 3, 6, 10 次)
- Midtrain 10B tokens
- 从这些文档 extract 13.5K evaluation examples
- 测 How2Score

**结果 (Table 3)**：

| Model | Metric | 0 | 1 | 3 | 6 | 10 |
|---|---|---|---|---|---|---|
| OLMo 3 7B | Doc PPL | 10.4 | 8.5 | 6.1 | 3.0 | 1.4 |
| OLMo 3 7B | How2Score | 14.0 | 17.3 | 15.8 | 15.7 | 16.5 |
| OLMo 3 32B | Doc PPL | 8.0 | 6.0 | 3.5 | 1.4 | 1.2 |
| OLMo 3 32B | How2Score | 33.3 | 39.3 | 39.4 | 38.1 | 37.9 |

**关键 finding**：
- Doc PPL 从 10.4 暴跌到 1.4（memorization 强烈发生）
- How2Score 只 modest 提升（7B: +3.3，32B: +6.1）
- **非单调**：32B 在 10 次 exposure 时 How2Score 反而下降到 37.9

**Intuition**：即使 aggressively contaminate training data，How2Score 不会大幅提升。How2Score 测的不是"是否见过这个文档"，而是 procedural generation capability。

---

## 8. 失败模式 (Section G)

paper 做了 qualitative 分析，发现 common failure patterns：

| Failure type | 例子 |
|---|---|
| Critical omissions | Gemini 2.5 Pro 在 sell apartment 例子中跳过 30-day waiting period |
| Missing parameters | Qwen 3 8B 在 Crock Pot 例子中没说 high/low、lid on/off |
| Wrong parameter values | （未具体举例） |
| Unsafe/invalid actions | （未具体举例） |
| Internal contradictions | （未具体举例） |
| Refusals | GPT 5 在 whole-plasmid PCR 例子中拒绝给 step-by-step，只给 high-level guidance——这被判为 critical failure 因为不可执行 |

Refusals 是 frontier model 的常见问题——paper 把 refusal 也算 critical failure 因为 relative to reference 它"not executable"。

---

## 9. 与相关工作的对比

| 工作 | 与 How2Everything 的区别 |
|---|---|
| Math PRMs (Lightman 2023) | math 可执行能找 first failure；open-world 不可执行，How2Score 用 binary |
| WikiHow datasets (Zhang 2020, Yuan 2023) | 局限单一 source；How2Mine 从 arbitrary web 挖，14 topics/189K domains |
| Cooking-only (Lal 2024, Toyooka 2025) | domain 窄；How2Bench 跨 14 topics |
| Generic LLM-as-judge (Zheng 2023, Dubois 2025) | generic preference 偏 surface quality；How2Score target task-level validity |
| Plasma (Brahman 2024) | 用 BLEU 和 human eval；How2Score 用 distilled judge 做 critical failure detection |

---

## 10. 我的几点观察

### 10.1 关于 Reference 的 Noise

paper 坦白：reference 不是 perfect ground-truth（96.6% 被 GPT-4.1 接受为 valid，3.4% 可能有问题）。这种 noise 会 propagate 到 evaluation 和 RL reward。Binary aggregation 比 first-failure localization 在 noise 下更 robust，因为不需要就具体哪一步出错达成一致。

### 10.2 关于 RL 的 Generalization

Table 2 最 striking 的数字是 OLMo-3-7B-Think 在 AIME25 上 +5.42。RL on procedural data 居然 improve 数学竞赛。这暗示 procedural generation 和 multi-step reasoning 共享某种 latent skill——比如"分步思考"、"每步检查依赖"、"保持 consistency"。当然 sample size 小（AIME25 几十题），需要谨慎。

### 10.3 Verbose Bias 的顽固性

即使有 explicit length control + length reward，Table 10 仍然显示 OR_ratio ≈ 1.012-1.018 per +1pp。LLM judge 的 verbose bias 是结构性的。任何 LLM-as-judge benchmark 都应该 report avg generated tokens alongside score，否则 gains 可能是 length gaming 的 artifact。

### 10.4 SFT 失败的深层原因

SFT 在 instruct checkpoint 上 -8.16 的原因是 objective mismatch。这和 RLHF 文献中"reward over-optimization" vs "imitation"的对比一致——SFT 是 imitate 单个 reference，RL 是 optimize 一个承认多种 valid solution 的 reward function。当 task 本身允许多种 valid solution 时（procedure generation 正是这种 task），RL 的 formulation 比 SFT 更合适。

### 10.5 Pretraining Maturity vs RL Gains 的 Pattern

Figure 6 是这 paper 最 informative 的图之一：SFT gains 跨 pretraining checkpoint 基本不变，RL gains 从 10% pretraining 的 3.56 增长到 100% 的 20.33。这印证了"RL amplifies pretraining, doesn't inject new capability"的 framing（Zhao et al. 2025 的 "echo chamber" hypothesis）。这也解释了为什么 RL 在弱 base model 上 gains 小——base capability 不够，RL 没东西可放大。

### 10.6 Distillation 的 Cost-Benefit

How2Judge 的 80.5% human agreement 比 GPT 5 的 83.0% 只低 2.5 个百分点，但 cost 几乎为零且完全 reproducible。这是 distillation 在 evaluation pipeline 中的典范用法——把 frontier judge 蒸馏到 small open model，让大规模 eval 和 RL reward 都 practical。

---

## 11. 整体框架的 Big Picture

```
[Web Documents]
    ↓ WebOrganizer format + topic classifier
[Topic-stratified tutorial docs]
    ↓ How2Mine (5 stages)
[351K structured procedures]
    ↓ Split
[How2Bench (7K eval)] + [How2Train (344K train)]
    ↓
[Generator model] → (g, R, n) → Ŝ
    ↓
[How2Score: detect critical failures]
    ↓ How2Judge (distilled 8B)
[Binary label: has_failure / no_failure]
    ↓
[Success rate as eval metric] OR [Reward signal for GRPO]
    ↓
[Improved model, validated on OOD suite]
```

整个 framework 的 elegance 在于：
- web data 同时 drive evaluation 和 improvement
- reference 作为 anchor 把 execution-free verification 变成可能
- critical failure detection 用 binary 换取 reliability
- distillation 让 reward 可复现
- RL gains 在外部 judge 和 OOD suite 上 persist

这种 framing 把 pretraining data curation 和 capability evaluation 统一起来——pretraining data 不只是模型训练的 input，也能作为 capability measurement 的 reference。这是 future LLM development pipeline 的方法论模板。

---

## 12. 一句话总结

paper 用 web 上天然存在的 how-to 文档作为 reference anchor，通过 critical failure detection 把 procedural validity 变成 scalable 可测的指标，蒸馏 frontier judge 到开源 8B model 让 evaluation 和 RL reward 都 cost-effective，最后证明 RL on procedural data 能 +10 points on How2Bench 且不退化 OOD benchmark。整个 framework 是"用 pretraining web data 支持 capability evaluation 和 improvement 的闭环"的 worked example。

---

# How2Everything: 从 Web 挖掘 How-To Procedures 来评测和改进 LLMs

Paper from AI2 + UMD，作者 Yapei Chang, Kyle Lo, Mohit Iyyer, Luca Soldaini。核心 idea：用 web 上自然发生的 how-to procedure 文档作为 reference anchor，构建一个闭环的 capability evaluation + improvement framework。这是一种 "execution-free verification" 的思路——当 real-world procedure 没有任务专属的 executor 时，用 mined reference 作为 ground truth proxy。

参考链接：
- Paper: https://arxiv.org/abs/2507.15532 (How2Everything)
- Code: https://github.com/allenai/how2everything
- Data: https://huggingface.co/datasets/how2everything/How2Bench, How2Train
- Model: How2Judge on HuggingFace
- WebOrganizer: https://openreview.net/forum?id=boSqwdvJVC
- DCLM: https://arxiv.org/abs/2406.11794
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- Process Reward Models (Lightman et al.): https://arxiv.org/abs/2305.20050

---

## 1. Motivation: 为什么 "how-to procedure generation" 值得单独研究

ChatGPT 的 query type 分布（Chatterji et al. 2025）显示 "How-To Advice" 占 8.5%，排第 4。每天数千万次交互都是 how-to 类型。同时 procedural content 在 pretraining/midtraining/posttraining 各阶段都能 improve downstream reasoning（Ruis et al. 2025 pretraining; Zhang et al. 2020 midtraining; Brahman et al. 2024 post-training）。

但 procedural validity 的 end-to-end evaluation 很难，原因：
- **Diversity**：既有工作局限于 cooking recipes (Lal et al. 2024; Toyooka et al. 2025) 或者特定 source 如 WikiHow (Zhang et al. 2020; Yuan et al. 2023)
- **End-to-end**：之前工作多关注 subtask，如 graph edge prediction (Sakaguchi et al. 2021) 或 step ordering (Zhang et al. 2020)
- **Reliability–scalability tradeoff**：BLEU 之类的 string-overlap metric 快但不准；human eval 准但贵

**Key intuition**：web 上有 virtually unbounded supply 的 naturally occurring real-world how-to 文档。这些文档可以作为 "reference anchors"，把 execution-based verification 不可行的任务转成可以大规模 measure 的 task。

---

## 2. How2Mine: 从 Web 挖掘 Procedures 的 Pipeline

### 2.1 数据源与采样

数据来自 DCLM web corpus (Li et al. 2025)。两层筛选：
1. **Format filter**：用 WebOrganizer (Wettig et al. 2025) format classifier，只保留标记为 "Tutorial & How-to Guide" 的文档（这类文档高密度地包含 imperative steps）
2. **Topic stratification**：用 WebOrganizer topic classifier，跨 14 个 topic 做 stratified sampling

14 个 topic 覆盖很广：Art & Design, Crime & Law, Education & Jobs, Electronics & Hardware, Fashion & Beauty, Food & Dining, Health, Home & Hobbies, Industrial, Religion, Science/Math/Tech, Sports & Fitness, Transportation, Travel & Tourism。

最终 980K 文档 → 351K procedures，跨 189K unique domains。

### 2.2 五阶段 Pipeline

每个阶段都有精心设计的 rationale。

**Stage 1: Procedure Extraction**
用 GPT-4.1 从 candidate web document 提取 (goal, ordered steps)。要求：
- Sequential：steps 有依赖关系
- Imperative & atomic：每个 step 是单个 action
- Concrete：说"做什么"不说"为什么"

输出 JSON：`{"has_valid_process": bool, "goal": str, "steps": list[str]}`

**Stage 2: Heuristics Filter**
两个简单规则：
- Step count：5 ≤ steps ≤ 15（避免 trivial 或 overly complex）
- N-gram repetition filter：2-gram ≥ 0.40, 3-gram ≥ 0.35, 4-gram ≥ 0.30 就 reject。主要去除 degenerate extraction。

**Stage 3: LLM Filter**
这个 stage 排除 6 类不适合的 procedure：

| 类别 | 为什么排除 |
|------|----------|
| Named-entity focused | 依赖特定 person/organization/website/brand，需要外部上下文才能判断正确性 |
| Pure math | 数学题的 correctness 由 numerical result 决定，不是 instruction validity |
| UI interaction | UI 会变化，state 隐含，execution-free 设定下不可验证 |
| Open-ended creative | 多种 valid endpoint，边界由 taste 决定，不是 missing prerequisites |
| Non-sequential | listicle 类型，steps 顺序无关，对 step-by-step validity 评估 ill-posed |
| Unreasonable | 内部矛盾或不能达成 goal，作为 reference 会污染 evaluation/training |

**Stage 4: Post-processing & Resource Extraction**
两件事：
1. Rewrite goal 让它 specific 且 deterministic——明确 constraints 和 expected outcome，缩小 valid solution 空间
2. 提取 resources list——steps 引用的 external tools/ingredients/equipment

为什么 resources 重要：同一个 goal 可能有多条 valid procedure，列出 resources 帮助 narrow solution space，让 evaluation 更 deterministic。

**Stage 5: Final Validation**
LLM sanity check，去除 remaining nonsensical examples。

### 2.3 各 Format 的 Yield 分析

Table 6 是很 informative 的实验：对 6 种 WebOrganizer format 各采样 140K 文档，跑 stage 1-3 测 yield：

| Source format | After extraction | After heuristics | After LLM filter |
|---|---|---|---|
| Tutorial & how-to guide | 86.87% | 70.46% | 24.46% |
| Personal blog | 34.48% | 25.41% | 14.26% |
| Knowledge articles | 35.47% | 23.72% | 16.54% |
| Non-fiction writing | 33.65% | 24.80% | 13.65% |
| Q&A forum | 49.02% | 31.75% | 17.67% |
| Academic writing | 29.20% | 23.21% | 16.29% |

**Intuition**：tutorial 类 format yield 最高（24.46%），所以作为 primary source 是合理的。但其他 format 也能产出 14-17% 的 valid procedures，说明 valid procedure 不是 tutorial 独有。

### 2.4 成本

跑 980K 文档，用 OpenAI batch API，252K requests，$5,717 USD。比人工 curation 便宜几个数量级。

---

## 3. How2Score: Critical Failure Detection Protocol

这是 paper 的核心创新。

### 3.1 关键定义：Critical Failure

> An omission, extraneous action, contradiction, severe vagueness, or other deviation from the reference that is severe enough to prevent achieving the goal, or to make the procedure unusable as instructions.

**重要 nuance**：
- 用 reference 作为 anchor，**不**惩罚 alternative valid procedures 或 superficial differences
- 例：goal 是做 terracotta pot 作为 gift，不同 gift message 不算 critical failure

**Inspiration 来自 Process Reward Models (PRM)**：
数学 PRM (Lightman et al. 2023) 中，verifier 找出 "first incorrect step" 作为 failure point。但 open-world procedures 的 steps 不可执行，自动定位 "first failure" 很难。所以 How2Score 采用 **binary aggregation** 而不是 first-failure localization。

**Reference correctness 的假设**：
用 GPT-4.1 验证 How2Bench references，96.6% 被接受为 valid。reference 不是 perfect ground-truth，而是让 task 更 deterministic 的 anchor。

### 3.2 形式化

Evaluation set $D$，每个 example $x = (g, R, S^*, \hat{S})$：
- $g$ = goal
- $R$ = extracted resource list
- $S^*$ = reference procedure
- $\hat{S}$ = model-generated procedure

LLM judge $J(\cdot)$ 输出 binary label：
- `has_failure` if 至少一个 critical failure
- `no_failure` otherwise

$$\text{Score}(D) = \frac{1}{|D|} \sum_{x \in D} \mathbb{I}[J(g, R, S^*, \hat{S}) = \text{no\_failure}]$$

变量解释：
- $|D|$ = evaluation set 大小
- $\mathbb{I}[\cdot]$ = indicator function，条件成立为 1 否则 0
- $J(\cdot)$ = judge function，返回 `no_failure` 或 `has_failure`

报告 success rate = no_failure 的比例。

**为什么 binary 而非 first-failure**：
human annotation 中，binary aggregation 的 Krippendorf's α = 0.593，first-failure localization 只有 α = 0.307。Binary 让 annotator 不必就"哪一步是 first failure"达成一致——他们只需就"是否有 critical failure"达成一致。

### 3.3 Human Annotation 验证

- 3 个 annotators via Prolific
- 200 examples（预筛掉需要 specialized domain knowledge 的）
- 时薪 $28，total $3,600 USD
- 多轮 pilot + training（initial α=0.273 → final α=0.593）

**Annotator training 的关键**：
早期 pilot 中，annotator 要么把任何和 reference 的 difference 标为 critical，要么被 coherent surface form 误导忽视实际 failure。所以 training materials 反复 clarify "non-critical variation" vs "critical failure" 的边界。

### 3.4 LLM Judge 对比 Human

Figure 3 的结果：

| Judge | Overall agreement | has_failure | no_failure |
|---|---|---|---|
| GPT 5 | 83.0% | 83.7% | 82.4% |
| Gemini 2.5 Pro | ~80% | - | - |
| Claude Opus 4.5 | ~80% | - | - |
| Gemini 3 Pro Preview | ~80% | - | - |
| GPT 5.1 | ~80% | - | - |

Human leave-one-out agreement: 84.7%-88.5%。GPT 5 接近 individual annotator 水平。

### 3.5 How2Judge: Distilled 8B Judge

为了低成本、可复现的大规模评测，蒸馏 GPT 5 到 Qwen 3 8B：

**Training data**：
- 73K GPT 5 annotations on outputs from diverse generators
- Generators 包括：Gemini 2.5 flash/pro, GPT-4.1, GPT 5, Qwen2.5-7B-inst, OLMo-2 1B/7B/32B 各 stage
- 跑 GPT 5 两次 per example，只保留 binary judgment 一致的（denoise）
- 与 human-annotated set 去重

**训练**：
- Qwen 3 8B
- 3 epochs
- lr = 5e-6
- batch size = 64

**结果**：
- 90.5% agreement with GPT 5
- 80.5% agreement with human majority
- 79.6% on has_failure, 81.4% on no_failure（balanced）
- 评测 7K examples 用 GPT 5 需要 $15，用 How2Judge 接近免费

**Intuition**：distillation 让 frontier judge 的能力 transfer 到可复现的小模型上，这是把"reliability–scalability tradeoff"打破的关键。蒸馏的 cost-effective 性质让 RL 用 How2Score 作为 reward 变得 practical。

---

## 4. How2Bench: 评测结果

How2Bench = 7K examples（每 topic 500）。Inference setup：模型接收 (goal, resources, n)，被要求输出 exactly n steps。

### 4.1 Length Control

LLM judge 普遍有 length bias。How2Bench 的应对：
- 3-shot prompt，例子展示 expected format 和 detail level
- 强制每 step 是 single concise sentence with one main action
- 要求模型 closely follow examples 的 concision

Figure 12 显示加这个 control 后，generation length 分布显著接近 reference distribution。

### 4.2 Decoding Setup

- Non-reasoning endpoints：greedy decoding，stop sequence `\n\n`
- Reasoning endpoints：T=0.6 stochastic decoding，用 provider 默认 thinking budget

### 4.3 主要结果 (Figure 4)

跨模型 family 和 size 都有清晰的 scaling trend。Frontier closed models 显著领先 open models。

### 4.4 Self-Preference Bias 检查 (Figure 5)

用 GPT/Gemini/Claude family 各自的 model 当 judge，重新评分 same generations。绝对分数变化，但 **relative ranking 不变**。这是 LLM-as-judge 的一个重要 robustness check。

### 4.5 Pretraining Trajectory 上的 Scaling (Table 8, Figure 1b)

在 OLMo 2/3 的 1B/7B/32B 各 pretraining checkpoint 上跑 How2Bench：

| Suite | Size | Stage | Step | How2Score |
|---|---|---|---|---|
| OLMo-2-0425 | 1B | Pretrain | 20000 | 0.06 |
| OLMo-2-0425 | 1B | Pretrain | 1907359 | 1.59 |
| OLMo-2-0425 | 1B | Midtrain | - | 6.39 |
| OLMo-2-0425 | 1B | Posttrain | - | 5.96 |
| OLMo-3-1025 | 7B | Pretrain | 14000 | 4.13 |
| OLMo-3-1025 | 7B | Pretrain | 1413814 | 21.59 |
| OLMo-3-1025 | 7B | Midtrain | - | 24.91 |
| OLMo-3-1025 | 7B | Posttrain | - | 30.23 |
| OLMo-3-1125 | 32B | Pretrain | 6000 | 6.21 |
| OLMo-3-1125 | 32B | Pretrain | 579120 | 31.00 |
| OLMo-3-1125 | 32B | Midtrain | - | 38.31 |
| OLMo-3-1125 | 32B | Posttrain | - | 43.16 |

**关键 observations**：
- 1B model 在 ~10^21 FLOPs (~5% pretraining) 处开始出现 non-trivial performance
- 跨 stage (pretrain → midtrain → posttrain) 单调提升
- 跨 size (1B → 7B → 32B) 单调提升

这使 How2Bench 适合做 performance forecasting (Xu et al. 2025)，不像许多 benchmark 要么早 saturate 要么在小 model 上 near-zero。

### 4.6 Format Emergence vs Validity (Table 9)

跟踪三个 formatting proxy metrics 跨 checkpoints：
- Step-count mismatch
- Duplicate-step frequency
- Dup n-gram rate (1-4 gram)

观察：formatting errors 在 early pretraining 快速 plateau，但 How2Score 继续提升。这是 emergence-like pattern：surface format 早期稳定，procedural validity 持续改善。后期 gains 不主要来自修 surface format，而是 end-to-end procedural validity。

### 4.7 How2Score vs Perplexity (Table 16)

在每个 OLMo trajectory 内，比较 checkpoint 按 How2Score 的 rank 和按 conditional perplexity on reference 的 rank：

| Trajectory | Spearman ρ |
|---|---|
| OLMo-2-0425 (1B) | 0.917 |
| OLMo-2-1124 (7B) | 0.667 |
| OLMo-2-0325 (32B) | 0.233 |
| OLMo-3-1025 (7B) | 0.850 |
| OLMo-3-1125 (32B) | 0.483 |

**Intuition**：1B model 两者高度相关（因为 1B 还在学 reference 的 surface form），但 32B model 相关性低（0.233），表明对 strong model，How2Score 测的不是 likelihood 而是 validity。

### 4.8 Instance-Level Correlates: Step Count 是 Difficulty Knob (Table 10)

用 logistic regression 分析 instance-level 属性如何预测 no_failure：

$$\text{logit}(p(\text{no\_failure})) = \beta_0 + \beta_{\text{steps}} \cdot |S^*| + \beta_{\text{res}} \cdot |R| + \beta_{\text{ratio}} \cdot \rho + \sum_{t \in T \setminus \{t_0\}} \gamma_t \mathbb{I}[\text{topic} = t]$$

变量解释：
- $|S^*|$ = reference step count（也等于 requested output length）
- $|R|$ = reference resource count
- $\rho = 100 \cdot |\text{gen}|/|\text{ref}|$ = generation/reference token ratio（百分比）
- $T$ = 14 topics
- $t_0$ = baseline topic (Art & Design)
- $\gamma_t$ = topic 固定效应，控制 topic 差异

报告 odds ratios $\text{OR} = \exp(\beta)$：

| Model | OR per +1 step | OR per +1 resource | OR per +1pp gen/ref |
|---|---|---|---|
| OLMo-3-7B-Inst | 0.756 [0.730, 0.783] | 1.009 [0.990, 1.028] | 1.012 |
| Qwen3-8B-Inst | 0.737 | 1.020 | 1.015 |
| OLMo-3.1-32B-Inst | 0.751 | 1.043 | 1.013 |
| Qwen3-32B-Inst | 0.765 | 1.018 | 1.014 |
| Gemini-2.5-Pro | 0.795 | 1.062 | 1.018 |
| Claude-Opus-4.5 | 0.813 | 1.060 | 1.017 |
| GPT 5 | 0.846 | 1.022 | 1.014 |
| Overall | 0.803 | 1.032 | 1.015 |

**解读**：
- OR_steps ≈ 0.74-0.85：每增加 1 step，no_failure 的 odds 下降 15-26%。Step count 是 dominant predictor。
- OR_ratio ≈ 1.01-1.02 per +1pp：residual verbosity 仍然 positively associated with no_failure，即使有 explicit length control。这就是为什么要 report avg generated tokens alongside How2Score。
- Topic effects 大但 cross-model 一致（Figure 13）：Education & Jobs 和 Food & Dining 比 baseline 容易，Electronics & Hardware 和 Industrial 更难。
- Frontier model 系数更接近 1（ceiling effect：高分时 residual variance 少）。

---

## 5. RL 训练：用 How2Score 作为 Reward

### 5.1 Setup

- 训练数据：100K from How2Train，跨 14 topic 平衡
- 用 embedding similarity filter 与 How2Bench 去重（threshold τ=0.65，Qwen3-Embedding-0.6B）
- Algorithm：GRPO (Shao et al. 2024)
- 1000 optimizer steps
- lr = 5e-7
- 4 prompts/batch，8 completions/prompt
- 三个 model：Qwen3-4B-Inst, Qwen3-8B-Inst, OLMo-3-7B-Think

### 5.2 三个 Reward 组件

**Reward 1: How2Score from How2Judge**
Binary 0/1，检测是否有 critical failure。

**Reward 2: Step-format verifier**
检查 final answer 是连续编号的列表（1,2,3,...）且 step count 匹配。Return 1 if 满足，0 otherwise。

**Reward 3: Reference-calibrated length reward**
防止 length gaming。设 $r = |\text{gen}|/|\text{ref}|$，$\tau = 0.2$，$\alpha = 5$：

$$R_{\text{len}}(r) = \begin{cases} 1, & |r - 1| \leq \tau \\ \exp\left(-\alpha \cdot \frac{|r - 1| - \tau}{1 - \tau}\right), & \text{otherwise} \end{cases}$$

变量解释：
- $r$ = generation 与 reference 的 token 长度比
- $\tau$ = tolerance band，允许 ±20% 偏差
- $\alpha$ = 衰减率，控制超出 tolerance 后 reward 下降速度
- 当 $r=1$，$R_{\text{len}}=1$（完美长度）
- 当 $|r-1|=\tau$（边界），$R_{\text{len}}=1$
- 当 $|r-1| > \tau$，指数衰减

**Intuition**：LLM judge 普遍 verbosity bias。如果不控制 length，模型会膨胀长度获取虚假 reward。Table 12 显示：Qwen3-4B-Inst 无 length reward 时 ratio 1.34，score 54.41（虚高）；有 length reward 时 ratio 1.01，score 43.52（真实）。

### 5.3 主要结果 (Table 2)

| Benchmark | Qwen3-4B-Inst Base | +RL | ∆ | Qwen3-8B-Inst Base | +RL | ∆ | OLMo-3-7B-Think Base | +RL | ∆ |
|---|---|---|---|---|---|---|---|---|---|
| How2Bench | 30.29 | 43.52 | **+13.23** | 38.52 | 48.62 | **+10.10** | 27.30 | 37.89 | **+10.59** |
| MMLU-Pro | 60.16 | 61.70 | +1.54 | 62.16 | 63.11 | +0.95 | 44.54 | 49.61 | +5.07 |
| GPQA | 44.87 | 44.64 | -0.23 | 54.02 | 53.79 | -0.23 | 46.21 | 47.10 | +0.89 |
| ZebraLogic | 82.4 | 81.2 | -1.2 | 85.2 | 85.7 | +0.5 | 65.6 | 63.3 | -2.3 |
| AlpacaEval | 44.78 | 47.73 | +2.95 | 58.44 | 58.76 | +0.32 | 49.75 | 51.19 | +1.44 |
| HumanEval+ | 71.95 | 75.43 | +3.48 | 81.28 | 79.57 | -1.71 | 90.49 | 89.45 | -1.04 |
| LiveCodeBench | 85.6 | 85.38 | -0.22 | 86.32 | 86.11 | -0.21 | 74.85 | 72.40 | -2.45 |
| MBPP+ | 67.46 | 66.98 | -0.48 | 68.65 | 69.31 | +0.66 | 64.81 | 64.29 | -0.52 |
| GSM8K | 94.09 | 93.78 | -0.31 | 95.68 | 95.30 | -0.38 | 94.92 | 95.30 | +0.38 |
| Minerva | 90.38 | 90.45 | +0.07 | 91.20 | 91.92 | +0.72 | 94.44 | 94.62 | +0.18 |
| Omega | 42.2 | 39.4 | -2.8 | 44.4 | 44.4 | 0.00 | 44.6 | 47.0 | +2.4 |
| AIME24 | 60.42 | 60.42 | 0.00 | 61.15 | 59.06 | -2.09 | 55.52 | 58.65 | +3.13 |
| AIME25 | 46.04 | 49.48 | +3.44 | 47.29 | 49.48 | +2.19 | 38.54 | 43.96 | +5.42 |
| $\overline{\Delta}_{\text{OOD}}$ | - | - | **+0.52** | - | - | **+0.06** | - | - | **+1.05** |

**关键 finding**：
- How2Bench 一致提升 >10 points
- 12 个 OOD benchmark 平均小幅正向 (+0.52, +0.06, +1.05)
- 个别 benchmark 退化（如 HumanEval+ 退化 -1.71 on Qwen3-8B），但没有 systematic regression
- 这暗示 procedural generation 是一种 broad capability，improvement 会 transfer

### 5.4 SFT vs RL (Table 11)

SFT 的结果很有启发性：

| Model | Stage | Before | After | ∆ |
|---|---|---|---|---|
| Qwen3-4B | Base | 32.00 | 33.11 | +1.11 |
| Qwen3-4B | Instruct | 29.70 | 28.47 | -1.23 |
| Qwen3-8B | Base | 35.54 | 35.20 | -0.34 |
| Qwen3-8B | Instruct | 35.34 | 32.45 | -2.89 |
| OLMo-3-7B | Base | 24.91 | 26.13 | +1.22 |
| OLMo-3-7B | Instruct | 30.23 | 22.07 | **-8.16** |

**为什么 SFT 在 instruct checkpoint 上无效甚至有害**：
SFT 的 objective 是 maximize likelihood of single reference per goal。但 How2Score reward 任何 valid procedure，只要 no critical failure。这是 **objective mismatch** (Stiennon et al. 2022; Xie et al. 2025)。Imitate 单个 reference 不必然减少 critical failures，反而可能 overfit 到一种 specific realization。

### 5.5 External Judge Robustness (Table 13)

用 GPT 5 和 Gemini 2.5 Pro 重新评测 RL 训练后的模型：

| Model | How2Judge ∆ | GPT 5 judge ∆ | Gemini judge ∆ |
|---|---|---|---|
| Qwen3-4B-Inst | +13.23 (43.69%) | +9.15 (33.72%) | +9.17 (58.58%) |
| Qwen3-8B-Inst | +10.10 (26.23%) | +8.76 (26.84%) | +8.03 (39.94%) |
| OLMo-3-7B-Think | +10.58 (38.77%) | +11.09 (53.74%) | +6.77 (50.05%) |

**Intuition**：Gains 在外部 judge 下持续存在。这表明 gains 不是 How2Judge 的 artifact，而是 procedural capability 真实提升。

---

## 6. Robustness 分析：排除 Confounds

这是 paper 严谨之处——主动 stress-test 两个可能的 confound。

### 6.1 Confound 1: Implicit Task Format Compliance

如果 gains 主要来自学了"输出格式"，那么从弱 pretraining checkpoint 应该也能得到类似 gains。

**Diagnostic 1: Pretraining Maturity (Figure 6)**

固定 post-training recipe (SFT + RL)，从不同 OLMo 3 7B pretraining checkpoint 出发：
- SFT gains：3.39 到 5.36（基本不变）
- RL gains：3.56 at 10% pretraining → 20.33 at 100% pretraining

**Intuition**：SFT 主要塑造 surface-level behavior（任何 pretraining level 都能学会 format），RL 才放大 pretrained capabilities。这与 Ouyang et al. 2022, Zhao et al. 2025 的发现一致——RL amplifies pretraining，不是 inject 全新能力。

**Diagnostic 2: Data Topic Coverage (Table 15)**

用 Qwen3-8B 做 topic-restricted RL：
- All topics: +10.10 overall
- Science only: +9.41 overall（broad transfer）
- Dining only: +5.55 overall（弱 transfer）

**Intuition**：Science topic 在 embedding space 分散（PCA Figure 14），dining 是 specialized cluster。如果 gains 纯 format-driven，应该各 topic 间 transfer 一样好。但 broad coverage yields largest gains，说明 improvement driven by content coverage 不是 generic format。

### 6.2 Confound 2: Memorization of Source Documents

How2Bench 的 evaluation examples 来自 web，而 LLM 也 pretrain on web。可能有 memorization confound。

**实验设计**：
- Midtraining 时控制文档曝光频率 (0, 1, 3, 6, 10 次)
- Midtrain 10B tokens
- 从这些文档 extract 13.5K evaluation examples
- 测 How2Score

**结果 (Table 3)**：

| Model | Metric | 0 | 1 | 3 | 6 | 10 |
|---|---|---|---|---|---|---|
| OLMo 3 7B | Doc PPL | 10.4 | 8.5 | 6.1 | 3.0 | 1.4 |
| OLMo 3 7B | How2Score | 14.0 | 17.3 | 15.8 | 15.7 | 16.5 |
| OLMo 3 32B | Doc PPL | 8.0 | 6.0 | 3.5 | 1.4 | 1.2 |
| OLMo 3 32B | How2Score | 33.3 | 39.3 | 39.4 | 38.1 | 37.9 |

**关键 finding**：
- Doc perplexity 从 10.4 暴跌到 1.4（memorization 强烈发生）
- How2Score 只 modest 提升（7B: +3.3，32B: +6.1），且 **非单调**
- 32B 在 10 次 exposure 时 How2Score 反而下降到 37.9

**Intuition**：即使 aggressively contaminate training data，How2Score 不会大幅提升。这表明 How2Score 测的不是"是否见过这个文档"，而是 procedural generation capability。

---

## 7. 整体框架的 Intuition

把所有 piece 串起来：

```
[Web Documents]
    ↓ WebOrganizer format + topic classifier
[Topic-stratified tutorial docs]
    ↓ How2Mine (5 stages: extract → heuristics → LLM filter → post-process → validate)
[351K structured procedures]
    ↓ Split
[How2Bench (7K eval)] + [How2Train (344K train)]
    ↓
[Generator model] → (g, R, n) → Ŝ
    ↓
[How2Score: detect critical failures]
    ↓ How2Judge (distilled 8B)
[Binary label: has_failure / no_failure]
    ↓
[Success rate as eval metric] OR [Reward signal for GRPO]
    ↓
[Improved model, validated on OOD suite]
```

**关键 intuition**：web 提供 unbounded supply 的 naturally occurring reference anchors。当 execution-based verification 不可行时，把这些 reference standardize 成 evaluable format + 用 critical failure detection 作为 validity proxy，就把 hard-to-measure behavior 转成 practical development loop。

---

## 8. 一些 Critical Observations

### 8.1 关于 Reference 作为 Anchor 的局限性

Paper 坦白：reference 不是 perfect ground-truth。96.6% 被 GPT-4.1 接受为 valid，但仍 3.4% 可能有问题。这种 noise 会 propagate 到 evaluation 和 RL reward。这是一个 inherent limitation，但 binary aggregation 比 first-failure localization 在 noise 下更 robust。

### 8.2 关于 RL 的 Generalization

Table 2 显示 OLMo-3-7B-Think 在 AIME25 上 +5.42 points，这是 RL on procedural data 居然 improve 数学竞赛成绩！这暗示 procedure generation 是一种 general capability，与 multi-step reasoning 共享某种 latent skill。但 sample size 小（AIME25 只有几十题），需要谨慎解读。

### 8.3 关于 Verbose Bias 的顽固性

即使有 explicit length control + length reward，Table 10 仍然显示 OR_ratio ≈ 1.012-1.018 per +1pp。这说明 LLM judge 的 verbose bias 是结构性的，需要持续 vigilance。任何 LLM-as-judge benchmark 都应该 report avg generated tokens alongside score。

### 8.4 关于 SFT 失败的深层原因

SFT 在 instruct checkpoint 上无效甚至有害（OLMo-3-7B -8.16）的原因是 **objective mismatch**。SFT 的 maximum likelihood 与 How2Score 的 minimum critical failures 不是一回事。多个 valid procedures 都满足 How2Score，但 SFT 只模仿一个。这和 RLHF 文献中 "reward over-optimization" vs "imitation" 的对比一致（Stiennon et al. 2022）。

### 8.5 关于 Topic Effects 的 Cross-Model 一致性

Figure 13 显示 topic fixed effects 在不同 model 上方向一致：Education & Jobs 和 Food & Dining 容易，Electronics & Hardware 和 Industrial 难。这说明 topic 难度有 intrinsic structure——可能因为 Electronics/Industrial 涉及更多 critical parameters（电压、时间、温度）和精确顺序依赖，而 Education/Food 更 forgiving。这是 future work 可以深挖的方向。

---

## 9. 公式与符号 Summary

**How2Score aggregation**:
$$\text{Score}(D) = \frac{1}{|D|} \sum_{x \in D} \mathbb{I}[J(g, R, S^*, \hat{S}) = \text{no\_failure}]$$

**Logistic regression for instance-level correlates**:
$$\text{logit}(p(\text{no\_failure})) = \beta_0 + \beta_{\text{steps}} |S^*| + \beta_{\text{res}} |R| + \beta_{\text{ratio}} \rho + \sum_{t \neq t_0} \gamma_t \mathbb{I}[\text{topic} = t]$$

**Length reward**:
$$R_{\text{len}}(r) = \begin{cases} 1 & |r-1| \leq \tau \\ \exp\left(-\alpha \cdot \frac{|r-1| - \tau}{1 - \tau}\right) & \text{otherwise}\end{cases}$$

**Krippendorf's α**：
$$\alpha = 1 - \frac{D_o}{D_e}$$
其中 $D_o$ = observed disagreement，$D_e$ = expected disagreement by chance。α=1 完美一致，α=0 等于 chance，α<0 systematic disagreement。

**Odds Ratio**:
$$\text{OR} = \exp(\beta)$$
解释：covariate 增 1 单位时 odds 的乘性变化。OR<1 表示该变量增加使 no_failure 更不易发生。

---

## 10. 与相关工作的 Position

- **vs Math PRMs (Lightman et al. 2023)**：math PRM 能定位 first incorrect step，但 open-world procedure 不可执行，自动 first-failure 定位太难。How2Score 用 binary aggregation 换取 reliability。
- **vs WikiHow-based datasets (Zhang et al. 2020; Yuan et al. 2023)**：那些局限于单一 source，How2Mine 从 arbitrary web 挖掘，覆盖 14 topics / 189K domains。
- **vs Cooking-only benchmarks (Lal et al. 2024; Toyooka et al. 2025)**：那些局限于窄 domain，How2Bench 跨 14 topics。
- **vs Generic LLM-as-judge (Zheng et al. 2023; Dubois et al. 2025)**：generic preference judging 偏 surface quality，How2Score target task-level validity。
- **vs Plasma (Brahman et al. 2024)**：Plasma 用 BLEU 和 human eval，How2Score 用 distilled judge 做 critical failure detection，且 cost-effective。

---

## 11. Final Intuition

这篇 paper 是 "把不可测的 capability 变成可测"的 worked example。三个关键 moves：

1. **Web 作为 unbounded reference supply**：execution-free 设定下，web 文档提供 naturally occurring 的 reference anchor，绕开"没有 executor"的难题。

2. **Critical failure detection 作为 validity proxy**：放弃"step-by-step correctness localization"，用 binary "is there any critical failure" 换取 reliability 和 scalability。这是 math PRM 思路在 open-world 上的退化版本，但 trade-off 合理。

3. **Distillation 让 reward 可复现**：把 frontier judge 蒸馏到 8B open model，让 evaluation 和 RL reward 都 cost-effective 且 reproducible。这是闭环的关键。

整个 framework 的 elegance 在于：用 web data 同时 drive evaluation 和 improvement，且每个 component 都可 scale。Pretraining data 不只是模型训练的 input，也能作为 capability measurement 的 reference。这种 framing 把 pretraining data curation 和 capability evaluation 统一起来，对 future LLM development pipeline 有方法论意义。
