---
source_pdf: The Magic Correlations Understanding Knowledge Transfer from Pretraining
  to Supervised Fine-Tuning.pdf
paper_sha256: 7b46e4896268c37fc4be0e8d952ca1ca0ce0cf5ec966bac6239fd4efcba127e2
processed_at: '2026-08-12T14:28:26-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

Andrej，我把那些公式和术语都扔掉，用大白话重新过一遍。

---

## 这篇 paper 到底在问什么？

你训练一个 LLM，分两步走：先 pretrain 吃海量文本，再 SFT 学会听指令。现在大家做 pretrain 的时候，通常拿一个小模型（比如 240M）当"试验田"，试不同的 data mixture 配比，看哪个表现好，然后就拿这个结论去训练大模型。

**核心问题就是：你在小模型上做的决策，到大模型上还管用吗？**

这帮人系统性地做实验回答了这个问题，答案很有意思——**有时候管用，有时候完全反转**。

---

## 他们做了什么实验？

很简单的 setup：

**两个模型大小**：240M 和 1B 参数

**9 种 data mixture**：3 种 web 数据源 × 3 种 code 占比
- Web 数据三种：RefinedWeb（基本不过滤）、FineWeb-Edu（专门筛教育内容）、DCLM（用模型打分筛质量）
- Code 占比三档：25%、35%、45%

**20 个 benchmark**，分四类：
- **Commonsense**：HellaSwag、PIQA 这些"物理常识"题
- **Science**：ARC、SciQ 这些科学知识题
- **NLI**：判断两句话的关系（蕴含、矛盾、中立）
- **Semantic**：判断两句话意思是否相同、代词指代等

然后他们 pretrain 完跑一遍 benchmark，SFT 完再跑一遍，看两个阶段的**排名**是否一致。

---

## 五个发现，用大白话

### 发现一：有些 task 的 pretrain 成绩能预测 SFT 后成绩，有些不能

**Science 和 Commonsense**：pretrain 阶段哪个 mixture 好，SFT 后还是那个好。排名基本不变。

**NLI 和 Semantic**：pretrain 阶段的排名到 SFT 后就乱了。SFT 基本上**重新组织**了模型处理这类 task 的方式。

直觉解释：Science 和 Commonsense 主要是"记住了多少事实知识"，这个在 pretrain 阶段就固化了，SFT 只是打磨一下输出格式。但 NLI 和 Semantic 需要的是"理解语言细微差别"的能力，SFT 的 instruction-following 训练让模型学会了一种全新的处理方式，把 pretrain 阶段学的 surface pattern 给覆盖了。

### 发现二：模型的"自信程度"在 SFT 后基本保留

这个挺反直觉的。模型对某个 benchmark "自信"还是"不自信"，这个倾向在 SFT 后基本保留。240M 模型上 79% 的 benchmark 都显示出这种 confidence persistence。

特别是 Commonsense 和 Science，SFT 前后 confidence 排名高度一致（相关系数 0.87 和 0.82）。NLI 又是例外，confidence 也被 SFT 重新洗牌了。

**人话**：SFT 改变了模型"说什么"，但没怎么改变模型"对什么有把握"。

### 发现三：大模型和小模型的行为模式完全相反

这是整篇 paper 最 striking 的发现。

**Accuracy transfer**：大模型更好。1B 的 pretrain→SFT accuracy 相关性 0.60，240M 是 0.49。这符合直觉——大模型学到的东西更"可迁移"。

**Confidence transfer**：大模型反而更差。240M 的 confidence 相关性 0.68，1B 掉到 0.39。

**为什么会这样？** 小模型容量有限，它用一套"统一的自信模式"应对所有 task，这个模式在 SFT 后不容易变。大模型容量大，SFT 时它会为不同 task 各自发展出不同的自信模式——比如对 Science 题很自信、对 NLI 题不自信——这种 task-specific 的 calibration profile 跟 pretrain 阶段就不一样了。

**实际影响**：你用小模型做 pretrain 阶段的 calibration 检查，结论到大模型就不靠谱了。大模型必须 SFT 之后单独重新做 calibration。

### 发现四：同一类 benchmark 内部会"互相打架"

你可能会想，HellaSwag 和 PIQA 都是 Commonsense，那我优化 HellaSwag 的 data mixture，PIQA 应该也变好吧？

**不是的。** 在 240M 和 1B 上，Commonsense 内部的 benchmark 经常显示负相关——提升一个会损害另一个。Semantic 也一样。

这意味着"Commonsense"这个标签下的 benchmark 其实测的是**不同的东西**。你提升 HellaSwag 不能代表你提升了"Commonsense reasoning"整体。

**好消息**：Science 在 1B 上首次出现正相关——大模型终于开始把科学知识当作一个统一的东西来学了。小模型还在"优化一个就得牺牲另一个"的阶段。

### 发现五：有些 benchmark 是"害群之马"

- **WinoGrande** 在 Commonsense 内部跟其他成员负相关（240M 上 -0.18）
- **WSC** 在 Semantic 内部跟其他成员相关性不到 0.15

即使你单独看 WinoGrande 的 transfer 还行，它的提升也不能 generalize 到同类的其他 task。**这类 benchmark 不能用来做 data curation 决策**，因为它给你的信号是误导性的。

还有几个 benchmark 在 240M 上是**负 transfer**——pretrain 好反而 SFT 后差：
- WiC: -0.07（240M）→ -0.43（1B）
- WinoGrande: -0.31（240M）→ +0.32（1B）
- MNLI: -0.34（240M）→ +0.53（1B）

注意 WinoGrande 和 MNLI 在 1B 上又变正了！**这种 scale-dependent 的翻转是最危险的**——你在 240M 上看 MNLI 选了一个 mixture，到 1B 上结论完全相反。

### 发现六：模型在不同 task 上的"校准质量"差异巨大

**Science**：模型自信的时候基本是对的（confidence-accuracy 相关性 0.7-0.8），非常 well-calibrated。

**Commonsense**：相关性接近 0，模型对错题和 对题的自信程度差不多。

**Semantic**：相关性是负的！模型在错的时候反而更自信。这是 systematic miscalibration。

**直觉**：Science 题有明确的对错答案，模型学到的是"我知道这个 or 我不知道"。Commonsense 和 Semantic 是更模糊的 task，模型容易学会 surface pattern matching——看起来 plausible 就自信，即使底层没真正理解。这跟 LLM hallucination 现象是同一个根源。

### 发现七：这种 calibration pattern 在 SFT 后基本保留

Science 的好 calibration 在 SFT 后还是好，Semantic 的差 calibration 在 SFT 后还是差。

**这意味着你不能指望 SFT 来修 calibration 问题**。Calibration 在 pretrain 阶段就定型了，SFT 只是继承。

唯一的例外是 NLI——它的 calibration persistence 是负的（-0.40），说明 SFT 确实在 NLI 上做了 fundamental reorganization。Pretrain 时 calibration 好的 mixture，SFT 后反而变差。

### 发现八：教育内容过滤的效果在大小模型上完全反转

**FineWeb-Edu vs RefinedWeb 在 NLI 上**：

- **240M 模型**：FineWeb-Edu 让 NLI 准确率提升 5 个百分点，但 calibration 从 0.68 掉到 -0.12。准确率上去了，模型却变得"错的时候更自信"。
  
- **1B 模型**：完全反转。FineWeb-Edu 让 NLI 准确率下降 4.4 个百分点，但 calibration 略微改善。

**为什么会这样？** FineWeb-Edu 把 web 数据筛成了"教育内容"，语言模式变窄了。小模型需要这种语言多样性来学习 NLI 的 surface pattern，所以筛掉反而变差。大模型已经隐式学到了这些 pattern，窄内容反而引入更少 noise。

**Science 上 FineWeb-Edu 一致地好**：两个 scale 上都提升 2-2.5 个百分点，calibration 基本不变。说明教育内容对"结构化事实知识"是 scale-invariant beneficial，但对"细微语言理解"是 scale-dependent harmful。

### 发现九：Code 数据对不同 task 的影响天差地别

- **HellaSwag（物理常识）**：code 占比越高越差（相关性 -0.94 at 240M）
- **PIQA（物理常识）**：同样负面（-0.77 at 1B）
- **SIQA（社交推理）**：code 占比越高越好（+0.51 at 1B）
- **BoolQ**：同样正面（+0.35 at 1B）

**直觉**：Code 数据里没有"物理世界"的描述——没有"杯子掉地上会碎"这种内容。所以 code 越多，模型的物理常识越差。但 code 的逻辑结构有助于抽象推理，包括"社交逻辑"（社交规则本质上也是一套抽象逻辑）。

NLI 内部也分裂：MNLI 和 QNLI 跟 code 负相关，RTE 正相关。因为 RTE 是简单的二分类（蕴含/不蕴含），code 的逻辑结构直接有用；MNLI 是三分类（蕴含/中立/矛盾），"中立"这个类别需要更细微的语言理解，code 数据帮不上忙。

---

## 三个 Practical Implications

### 1. 选 benchmark 要看 category

不能随便挑一个 Commonsense benchmark 来做 pretrain 阶段的 proxy。HellaSwag、PIQA、COPA 是 reliable predictor。WiC、WSC、MultiRC 是 unreliable 的，用了会误导。

WinoGrande 和 MNLI 在 240M 上是负 transfer，**绝对不能用来做小模型的 data curation 决策**。

### 2. 模型变大后 confidence 不再 transfer

小模型上你可以在 pretrain 阶段就预测 SFT 后的 calibration 质量。大模型上不行了，必须 SFT 之后单独做 calibration。

这解释了为什么 production LLM（都是大模型）需要那么多 post-hoc calibration 工作——temperature scaling、verbalized confidence、human feedback calibration 等等。小模型没这个问题。

### 3. 小模型上的 data curation 决策不能直接 extrapolate 到大模型

**特别是 NLI**：FineWeb-Edu 在 240M 上 +5pp，1B 上 -4.4pp。如果你用 240M proxy 选了 FineWeb-Edu 来优化 NLI，到 1B 上你的准确率反而降低了 4.4 个百分点。

正确做法：**在多个 scale 上验证 data curation 决策**，至少 2 个 scale，再 commit 到 production training。

---

## 我觉得最值得记住的几点

1. **Pretrain 决定 calibration fingerprint，SFT 决定 task-specific output adaptation**。这两个是分离的维度，不能混为一谈。

2. **Scale 对 accuracy 和 confidence 的作用方向相反**。大模型 accuracy 更可迁移，但 confidence 反而不行。这是反直觉的，scaling laws 文献从来没提过这点。

3. **Calibration 在 pretrain 就定型了，SFT 修不了**（NLI 例外）。想 deploy well-calibrated model，要在 pretrain data 上下功夫，不能指望 SFT 救场。

4. **同一 category 内的 benchmark 不一定测同一个东西**。HellaSwag 好不代表 Commonsense 整体好。做 data curation 要看 category-level 而非 single benchmark。

5. **小模型 proxy 的决策在大模型上可能完全反转**。这是对 DOReMi 等 small-proxy-based data selection 方法的直接挑战。

---

## 一句话总结

**你以为 pretrain 好的东西 SFT 后也会好，大模型上更好——但其实 accuracy 和 confidence 是两条独立的线，scale 让它们朝相反方向走，calibration 在 pretrain 就锁死了，SFT 修不了，小模型上的 data 决策到大模型可能完全翻转。**

这篇 paper 最大的贡献是把这些"隐藏的复杂性"用一套清晰的 correlation protocol 暴露出来了。具体的数值在更大 scale 上可能会变，但这个 methodology 和 qualitative findings 我觉得是 lasting 的。

---

# The Magic Correlations: Pretraining→SFT Knowledge Transfer 深度解析

Andrej 你好, 这篇 paper 触及了一个我长期关注的问题: **当我们用 small proxy model 在 pretraining 阶段做 data curation 决策时, 这些决策真的能 extrapolate 到 large-scale post-SFT 吗?** 论文用 correlation-based protocols 给出了相当 nuanced 的答案, 我会尽量把所有直觉、公式含义、实验细节、以及与你以往工作 (e.g., nanoGPT, llm.c, Education LLM 思路) 的关联都挖掘出来。

---

## 1. Paper 的 Core Thesis 与 Motivation

现代 LLM training 分两阶段: **pretraining** (PT) on massive corpora, 然后 **supervised fine-tuning** (SFT) on instruction data。在 PT 阶段, 我们通常在 small-scale proxy models 上做 data mixture 选择, 依赖一个隐含假设:

> Pretraining benchmark performance reliably predicts post-SFT performance, 且 internal representations remain stable across training stages.

这篇 paper 系统性地 scrutinize 这个 assumption, 通过 **5 个 correlation protocols** 在 240M 和 1B 两个 scale、9 种 data mixtures、20 个 benchmarks 上的实验, 揭示 transfer dynamics 远比想象中复杂。

参考:
- Kaplan et al. 2020 Scaling Laws: https://arxiv.org/abs/2001.08361
- Hoffmann et al. 2022 Chinchilla: https://arxiv.org/abs/2203.15556
- Hernandez et al. 2021 Transfer Scaling: https://arxiv.org/abs/2102.01293
- Lourie et al. 2025 (downstream scaling laws unreliable): https://arxiv.org/abs/2507.00885
- Schaeffer et al. 2025 (frontier prediction elusive): https://openreview.net/forum?id=I1NtlLvJal

---

## 2. Experimental Setup 细节

### 2.1 Model Architecture

| Hyperparameter | 240M | 1B |
|---|---|---|
| Model dimension | 768 | 1680 |
| Per head dim | 256 | 256 |
| Heads | 8 | 8 |
| Layers | 8 | 12 |
| Expand factor | 8 | 8 |
| Context length | 4096 | 4096 |
| Vocab | 100,864 | 100,864 |

注意 per-head dim 是 256, 相对较大 (typical GPT-2 style 是 64), 这暗示 design 倾向于 fewer heads with larger per-head dim。Vocab 100,864 也不是常见的 50k, 暗示用了 SentencePiece-style tokenizer (可能与 T5/Titan 谱系相关)。

### 2.2 Pretraining Data Mixtures (9 total)

**Factorial design**: 3 web sources × 3 code proportions, 加上固定的 curated knowledge。

**Web sources (3)**:
- **RefinedWeb** (V1) - minimal filtering, https://arxiv.org/abs/2306.01116
- **FineWeb-Edu** (V2) - aggressive educational filtering, https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **DCLM** (V3) - model-based quality scoring, https://arxiv.org/abs/2406.11794

**Code sources**:
- StarCoder (with V1)
- The Stack v2 (with V2, V3), https://arxiv.org/abs/2402.19173

**Curated knowledge (固定)**: RedPajama-v2 中的 Wikipedia, ArXiv, StackExchange, Books, https://arxiv.org/abs/2411.12372

**Proportion configs (P0/P1/P2)**:
- P0 (web-heavy): 65% web / 25% code / 10% curated
- P1 (balanced): 45% web / 35% code / 20% curated
- P2 (code-heavy): 25% web / 45% code / 30% curated

这个设计干净地 **disentangle** data source quality 和 data composition effects, 这是很多 earlier scaling papers 没做到的。

### 2.3 SFT Setup

- Dataset: **Tulu-v2-mix** (Ivison et al. 2023), https://arxiv.org/abs/2311.10702
- 包含 FLAN-v2, Open Assistant, ShareGPT 等
- 240M: 5 epochs, 6365 steps, lr=2e-5
- 1B: 10 epochs, 12730 steps, lr=1e-5
- Batch size 256, seq_len 4096, cosine schedule

注意 1B 用了 10 epochs 而 240M 用了 5 epochs, 这是一个 **slightly non-iso-compute** setup, 不过考虑到 token 数差异, 这是 reasonable。

### 2.4 Pretraining Compute

- 240M: 12B tokens (200k steps × 256 batch × 4096 seq ≈ 210B tokens? 实际是 12B)
- 1B: 52B tokens

这两个数和 Chinchilla optimal (≈ 20 tokens/parameter) 略有差距, 240M 被略微 over-trained, 1B 略微 under-trained。这会影响 transfer dynamics 的解读 (下面会展开)。

---

## 3. 五个 Correlation Protocols 公式详解

这是这篇 paper 最有意思的部分 — 把 "transferability" 用 5 个不同的 correlation lens 来看。我逐一拆解公式中每个变量。

### 3.1 Cross-Stage Accuracy Correlation

$$r_{\text{acc}}^{\text{stage}} = \text{corr}(\mathbf{a}^{\text{PT}}, \mathbf{a}^{\text{SFT}})$$

- $\mathbf{a}^{\text{PT}} \in \mathbb{R}^M$: 在 $M=9$ 个 data mixtures 上, 某 benchmark 的 PT accuracy scores 向量
- $\mathbf{a}^{\text{SFT}} \in \mathbb{R}^M$: 同一 benchmark 在 SFT 后, 在同样 9 个 mixtures 上的 accuracy 向量
- $r_{\text{acc}}^{\text{stage}}$: Pearson correlation, 衡量 "PT 阶段 data mixture 排名是否预测 SFT 阶段排名"
- **直觉**: 若 $r > 0.7$, 则 "PT 上表现好的 mixture 在 SFT 后也好" 这个 ranking hypothesis 成立; 若 $r < 0.3$ 或 negative, 则 PT 选择不可靠

注意: 这里 corr 是 **across mixtures** (不是 across samples), 所以度量的是 data mixture 层面的 ranking 稳定性, 不是 per-sample 的 transferability。

### 3.2 Cross-Stage Confidence Correlation

$$r_{\text{conf}}^{\text{stage}} = \text{corr}(\mathbf{c}^{\text{PT}}, \mathbf{c}^{\text{SFT}})$$

- $\mathbf{c}^{\text{PT}}, \mathbf{c}^{\text{SFT}}$: 平均 confidence scores (i.e., selected answer 的 softmax probability 在 eval set 上平均)
- 这度量的是 **calibration fingerprint persistence** — 模型在 PT 阶段对某 benchmark 表现出"自信"或"不自信"的倾向是否在 SFT 后保留

### 3.3 Pretraining Coherence (Intra-Category)

$$r^{\text{PT}}(i, j) = \text{corr}(\mathbf{a}_i^{\text{PT}}, \mathbf{a}_j^{\text{PT}}), \quad i < j$$

- $i, j$ 是同一 category $C$ 内的两个 benchmark (e.g., HellaSwag 和 PIQA 在 Commonsense 内)
- 度量: 优化一个 benchmark 的 mixture 是否也改善另一个
- $r > 0$: synergy (良性协同)
- $r < 0$: **competition** (data mixture 改善 A 反而损害 B, 警告 "intra-category competition")

### 3.4 SFT Coherence

$$r_{\text{intra}}^{\text{SFT}}(i, j) = \text{corr}(\mathbf{a}_i^{\text{SFT}}, \mathbf{a}_j^{\text{SFT}})$$

比较 $r_{\text{intra}}^{\text{SFT}}$ 与 $r^{\text{PT}}$ 可以揭示 SFT 是 **resolve** 还是 **introduce** intra-category competition。

### 3.5 Cross-Stage Intra-Category Coherence

$$r^{\text{cross}}_{\text{intra}}(i, j) = \text{corr}(\mathbf{a}_i^{\text{PT}}, \mathbf{a}_j^{\text{SFT}}), \quad i \neq j$$

这个是最 subtle 的: PT 阶段对 benchmark $i$ 的提升, 是否能预测 SFT 后对 **不同** benchmark $j$ (同 category 内) 的提升? 这是真正的 **cross-benchmark transfer**。

### 3.6 Category-Level Average

$$\bar{r}_{\text{intra}}(C) = \frac{2}{|C|(|C|-1)} \sum_{i<j, i,j \in C} r_{\text{intra}}(i, j)$$

- $|C|$: category 中 benchmark 数量 (e.g., Commonsense 有 7 个, Science 有 4 个)
- $2/(|C|(|C|-1))$: 归一化, 因为有 $\binom{|C|}{2}$ 个 unique pairs
- 这给出 category-level 的 coherence summary

### 3.7 Accuracy-Confidence Alignment (Calibration Proxy)

$$r_{\text{align}} = \text{corr}(\mathbf{a}, \mathbf{c})$$

- $\mathbf{a}, \mathbf{c}$: 某 model configuration 下, 在 benchmark suite 内的 accuracy 和 confidence 向量
- **High positive**: well-calibrated (confident when correct, uncertain when wrong)
- **Negative**: systematic overconfidence on errors (miscalibration)
- 这是一个 **cheap proxy** for calibration quality, 比 ECE/Brier score 简单很多, 但论文没有做正式 calibration metric 对比

直觉上, 这个 metric 类似于 **rank correlation between per-benchmark accuracy and per-benchmark confidence**, 它并不是 per-sample calibration, 而是粗粒度的 global pattern。

参考: Guo et al. 2017 (temperature scaling): https://arxiv.org/abs/1706.04599
参考: Kadavath et al. 2022 (LMs know what they know): https://arxiv.org/abs/2207.05221
参考: Tian et al. 2023 (just ask for calibration): https://arxiv.org/abs/2305.14975
参考: Geng et al. 2024 survey: https://aclanthology.org/2024.naacl-long.366/

---

## 4. 主要 Findings 深度分析

### Finding 1: Transfer Reliability Strongly Category-Dependent

从 Figure 1(a) 看:
- **Science 和 Commonsense**: 高 cross-stage accuracy correlation ($\bar{r} > 0.5$), 说明 PT 性能可预测 SFT 性能
- **Semantic 和 NLI**: weak transferability, 说明 SFT **fundamentally reorganize** 模型对这些 task 的处理方式

**Intuition**: Commonsense 和 Science 主要依赖 **factual knowledge storage** + pattern matching, 这类 capability 在 PT 阶段固化得很深, SFT 是 "polishing" 而非 "restructuring"。而 Semantic 和 NLI 需要 **linguistic nuance understanding** (paraphrase, entailment, coreference), 这些 capability在 SFT 时被 instruction-following format 重新组织 — 比如 SFT 让模型学会用 chain-of-thought 处理 entailment, 而非依赖 PT 学到的 surface pattern。

这与 Desai & Durrett 2020 的发现一致 (pretrained transformers 需要在 downstream 重新 calibration): https://aclanthology.org/2020.emnlp-main.21/

### Finding 2: Confidence Patterns Persist More Strongly Than Accuracy

在 240M 模型上, **mean benchmark-wise confidence correlation $\bar{r} = 0.68$**, 79% benchmarks 超过 $r > 0.5$。Commonsense (0.87) 和 Science (0.82) 显示出**强烈的 cross-stage confidence persistence**, 而 NLI (0.21) 弱。

**直觉**: 模型的 "uncertainty fingerprint" 在 SFT 后大部分保留, 说明 SFT 改变的是 **decision boundary**, 不是 model's internal sense of certainty。这呼应 Kadavath et al. 2022 的观察 (large models know what they know)。

但这里的 caveat 是: confidence persistence 是 **rank-based**, 不是 absolute confidence level 的保留。SFT 通常会 lower confidence on instruction-formatted prompts (因为 format mismatch), 但 ranking 在 mixture 间是稳定的。

### Finding 3: Cross-Benchmark Confidence Structure Persists

PT-PT heatmap (Figure 4a) 和 SFT-SFT heatmap (Figure 4b) 在 240M 上结构相似度 $r = 0.73$, $p < 10^{-32}$。Commonsense–Science 形成 "confidence block", 这个 block 通过 SFT 不仅保留还略微加强。

**直觉**: 这暗示 SFT 是 **reinforcing** existing calibration structure, 而非 disrupting。这与" catastrophic forgetting" 担忧相反 — 至少在 confidence 层面, SFT 是 graceful 的。

### Finding 4: Certain Benchmarks Are Unreliable Cross-Stage Predictors

具体数据 (Table 5):
- **WiC**: $r_{\text{acc}}^{\text{stage}} = -0.07$ (240M), $-0.43$ (1B) — **negative transfer** at 1B
- **MultiRC**: $0.11$ (240M), $0.26$ (1B) — weak
- **WinoGrande**: $-0.31$ (240M), $0.32$ (1B) — 在 240M **negative**, 但 1B 时转正
- **MNLI**: $-0.34$ (240M), $0.53$ (1B) — 同样 scale-dependent reversal

**Common characteristic**: 这些 benchmarks 都需要 nuanced linguistic understanding, 而 SFT 显著 reorganize 这类能力。

**关键实用警告**: 如果你用 small proxy model 评估 WinoGrande 或 MNLI 来做 data curation 决策, **你的 ranking 可能在 1B 时翻转**!

### Finding 5: Intra-Category Competition Undermines Single-Benchmark Evaluation

Commonsense 和 Semantic category 在两个 scale 上都显示 **negative intra-category coherence on accuracy**。意思是: 提升 HellaSwag 的 mixture 经常 degrade 其他 Commonsense benchmarks。

这与直觉相反 — 我们一直假设 Commonsense benchmarks 都测一个东西。事实上, **data mixture 决策必须考虑 category-level 而非 single benchmark**。

Science category 在 1B 上首次出现 positive intra-category coherence ($\bar{r}^{\text{PT}} = 0.50$, $\bar{r}^{\text{SFT}} = 0.17$, $\bar{r}^{\text{cross}} = 0.25$), 说明 larger models 学到更 **unified** 的 scientific reasoning representation。

### Finding 6: "Black Sheep" Benchmarks

- **WinoGrande**: 在 Commonsense 内, 与其他成员的 mean pairwise $r = -0.18$ (240M)
- **WSC**: 在 Semantic 内, $r < 0.15$ 与其他成员

这些 benchmark 即使 individual transfer 可接受, 也无法 generalize 到同 category 其他 task。Practical implication: 即使你只关心 Commonsense 总体, 也不要仅看 WinoGrande。

### Finding 7: Inverse Scaling of Accuracy and Confidence Transfer

| Metric | 240M | 1B | Direction |
|---|---|---|---|
| Accuracy $\bar{r}_{\text{acc}}^{\text{stage}}$ | 0.49 | 0.60 | ↑ with scale |
| Confidence $\bar{r}_{\text{conf}}^{\text{stage}}$ | 0.68 | 0.39 | ↓ with scale |

**这是 paper 最 striking 的发现**。Larger models 在 accuracy 上 transfer 更好, 但 confidence 反而 transfer 更差。

**直觉解释 (Karpathy-style)**: 当模型 scale up, 它的 representational capacity 增加, SFT 时能 develop **task-specific calibration profiles** — 即, 对不同 task 学会不同的 uncertainty pattern。Small models capacity 受限, 倾向于 **monolithic, uniform calibration pattern**, 这个 pattern 通过 SFT 不容易改变。

这让我想到 "grokking" 现象 — larger models 在 SFT 时确实有更多 freedom 去 reorganize 内部 representation。这个发现可能解释了为什么 production-scale LLM 在 SFT 后经常需要重新做 calibration (Kadavath et al., Tian et al. 都观察到这点)。

### Finding 8: Scaling Induces Intra-Category Competition → Synergy Transition

Science category 的 coherence 变化 (Figure 5):
- PT: $0.24 \to 0.50$
- SFT: $-0.19 \to 0.17$
- Cross-stage: $-0.15 \to 0.25$

**Intuition**: Small models 的 capacity 是 zero-sum — 优化一个 Science benchmark 会 steal capacity from 其他 Science benchmarks。Larger models 有 capacity buffer, 可以**同时**优化多个相关 task。这与 "skill mixing" 在 small models 上的 trade-off 一致。

### Finding 9: Confidence Coherence Degrades with Scale (While Positive)

| Category | 240M | 1B |
|---|---|---|
| Commonsense | 0.73 | 0.24 |
| Science | 0.85 | 0.58 |

这进一步支持 Finding 7 — larger models develop **task-specific** calibration, 而 small models 用 **universal** calibration。

### Finding 10: NLI Anomalous Scaling Pattern

NLI 是唯一一个 1B 比 240M 显示更高 confidence coherence 的 category — 与其他 category 趋势相反。

**假说**: Larger models develop 更 consistent 的 inferential strategies 专门 for NLI, 即使其他 category 在 scale up 时变得更 task-specific。这可能反映 NLI 在 large models 中被作为 "distinct inferential mode" 处理。

### Finding 11: Alignment Varies Dramatically by Category

Figure 6 和 Table 6 的数据:

| Scale | Stage | Commonsense | Science | NLI | Semantic |
|---|---|---|---|---|---|
| 240M | PT | 0.05 | 0.75 | 0.35 | -0.18 |
| 240M | SFT | 0.04 | 0.80 | 0.25 | -0.37 |
| 1B | PT | -0.12 | 0.81 | -0.12 | -0.06 |
| 1B | SFT | -0.16 | 0.83 | 0.39 | -0.16 |

- **Science**: consistently well-calibrated ($r_{\text{align}} \approx 0.7-0.8$) — 模型 confident when correct
- **Commonsense**: weak near zero, slightly negative at 1B
- **Semantic**: consistent negative alignment — systematic **overconfidence on errors**!

**直觉**: Science questions 有明确的 factual answer, model 学到 "I know this or I don't" 的 confidence pattern。Commonsense 和 Semantic 是更模糊的 task, 模型容易学会 surface pattern matching, 在错误时也表现自信。

这呼应了 LM hallucination 现象 — 模型对 "looks plausible" 的答案自信, 即使底层 understanding 不存在。

### Finding 12: Alignment Patterns Persist Through SFT

Table 10 的 persistence correlation:
- Commonsense: 0.73 (240M), 0.54 (1B)
- Science: 0.62, 0.31
- **NLI: -0.41, -0.40** ← negative persistence!
- Semantic: 0.50, 0.27

**Striking**: NLI 的 alignment persistence 是负的 — PT 阶段 calibration 好的 mixture, SFT 后反而变差。这意味着 NLI 经历 fundamental reorganization, 与 Finding 10 一致。

其他 category (Commonsense, Science, Semantic) 的 alignment 在 SFT 后 largely persist。Practical implication: **calibration quality 由 pretraining 决定, SFT 不能 fix miscalibration**。这反驳了 "SFT 可以修 calibration" 的常见假设。

### Finding 13: Educational Filtering Exhibits Scale-Dependent Accuracy-Calibration Trade-offs

这是 paper 中最 actionable 的 finding。

FineWeb-Edu vs RefinedWeb 在 NLI 上:

| Scale | Stage | Metric | RefinedWeb | FineWeb-Edu | Δ |
|---|---|---|---|---|---|
| 240M | PT | Accuracy | baseline | +5.0pp | +5.0 |
| 240M | PT | Alignment | 0.68 | -0.12 | **-0.80** |
| 1B | PT | Accuracy | baseline | -4.4pp | -4.4 |
| 1B | PT | Alignment | -0.26 | -0.10 | +0.16 |

**惊人的 reversal**: 在 240M 上 FineWeb-Edu 提升 NLI accuracy 5pp 但严重破坏 calibration (从 0.68 到 -0.12, Δ=-0.80), 而在 1B 上完全反转 — FineWeb-Edu degrade accuracy 但 improve alignment。

**直觉**: Educational content filtering 移除了 diverse linguistic patterns, 这些 patterns 对 small model 的 NLI 推理有帮助 (因为 small model 需要 surface diversity 来 infer entailment), 但在 large model 中, 这种 diversity 反而引入 noise。

**Practical warning**: 你用 240M proxy model 选 FineWeb-Edu for NLI, 实际 deploy 到 1B 时, accuracy 反而**降低 4.4pp**, 而且你的 calibration 改善也只是 marginal (Δ=+0.16 vs PT 时 -0.80 的灾难)。

Science category 的 FineWeb-Edu benefit 是 robust across scales:
- 240M: +2.1pp accuracy, -0.14 alignment
- 1B: +2.5pp accuracy, -0.03 alignment

这说明 educational content 对 **structured factual knowledge** 是 scale-invariant beneficial, 但对 nuanced linguistic reasoning 不是。

---

## 5. Code Proportion Effects (Appendix C.4 细节)

Figure 14 和 15 揭示 code data 的 heterogeneous effects:

**Physical reasoning tasks** (HellaSwag, PIQA): 强烈 negative correlation with code proportion
- HellaSwag: $r = -0.94$ at 240M, $-0.80$ at 1B
- PIQA: $r = -0.77$ at 1B

**Social reasoning tasks** (SIQA, CommonsenseQA, BoolQ): positive correlation
- SIQA: $r = +0.51$ at 1B
- BoolQ: $r = +0.35$ at 1B

**Intuition**: Code data 缺乏 physical world descriptions, 所以 degrade grounded physical reasoning。但 code 的 logical structure 改善 abstract logical reasoning — 包括 social reasoning (它本质上是 abstract logic of social norms)。

**NLI heterogeneity**: MNLI 和 QNLI 与 code 负相关 ($r=-0.68, -0.53$ at 1B), 但 RTE 正相关 ($+0.25$)。RTE 是 binary classification (entail/not), 比 MNLI 的 3-way (entail/neutral/contradiction) 简单, 可能更容易受益于 code 的 logical structure。

**Scale interactions**: MNLI 在 240M 上 $r=0.12$ (weak positive), 1B 上 $r=-0.68$ (strong negative)。又一次证明 **small-scale 结论不能 extrapolate 到 large-scale**。

参考: Code data 影响 pretraining 的相关工作
- StarCoder: https://arxiv.org/abs/2305.06161
- The Stack v2: https://arxiv.org/abs/2402.19173

---

## 6. 我对 paper 的 Intuition 和 Connection to Your Work

### 6.1 "Magic Correlations" — 为什么叫 Magic?

Paper title 的 "magic" 应该是指这些 correlation patterns 的 non-obvious 性质。特别是 **inverse scaling** (accuracy transfer ↑ with scale, confidence transfer ↓ with scale) 这个 finding 违反直觉, 因为 scaling laws 文献通常假设 "more parameters = more transferable everything"。

### 6.2 联系到 nanoGPT / llm.c 视角

nanoGPT 训练 GPT-2 (124M) on TinyStories 或 FineWeb-Edu 时, 实际上你观察到的 phenomenon 可能正属于这篇 paper 的 240M regime:
- Confidence patterns transfer reliably through SFT
- Intra-category competition 在 Commonsense 内可见
- Educational filtering 在 small-scale 上对某些 task 有显著效果

但如果你 scale 到 1B+, **inverse scaling** 会发生 — confidence transfer breakdown, 模型开始 develop task-specific calibration。这可能解释了为什么 production LLM 需要 RLHF + 多轮 calibration tuning, 而 small model 一轮 SFT 就够。

### 6.3 联系到 Education LLM 思路

你提到过 "教育一个 LLM 像 teaching a child"。这篇 paper 实证支持了这一点 — FineWeb-Edu (educational content) 在 Science 上 **scale-invariant beneficial**, 暗示 educational content 提供 structured factual knowledge, 与 model capacity 无关。

但 NLI 的 reversal 暗示: educational content 的"窄谱"性质对 nuanced linguistic reasoning 是 scale-dependent harmful。Small model 需要它来 learn surface patterns, 但 large model 已经 implicit 学到这些 patterns, 反而被 narrow content 限制。

### 6.4 与 Scaling Laws 的关系

Hernandez et al. 2021 transfer scaling laws 提出了 $L_C \approx (C_T + \gamma C_N)^{-0.17}$ 等 formula, 其中 $C_T$ 是 transfer compute, $C_N$ 是 pretraining compute, $\gamma$ 是 transfer efficiency multiplier。但 Lourie et al. 2025 (https://arxiv.org/abs/2507.00885) 和 Schaeffer et al. 2025 (https://openreview.net/forum?id=I1NtlLvJal) 都 challenge 这些 laws的 reliability for downstream tasks。

这篇 paper 提供了一个 **correlation-based lens** — 不预测 absolute performance, 而是测 ranking 稳定性。这是一个 fundamentally 不同的 framing, 可能为 future scaling law research 提供更 nuanced 的 angle。

### 6.5 与 DataAttribution / TRAK / DOReMi 的对比

DOReMi (Xie et al. 2023, https://arxiv.org/abs/2305.10429) 用 small proxy model 的 loss distribution 来重 weight data sources。TRAK (Park et al. 2023, https://arxiv.org/abs/2303.14186) 和 influence functions (Grosse et al. 2023, https://arxiv.org/abs/2308.03296) 做更 fine-grained attribution。

这篇 paper 取 **aggregate approach**: 不 attribute individual samples, 而是 characterize broad mixture choices 的 cross-stage transferability。这两类方法互补 — DOReMi 优化 mixture weights, 这篇 paper 评估 mixture effects 是否 persist。

### 6.6 Benchmark Contamination 视角

Sainz et al. 2023 (https://aclanthology.org/2023.findings-emnlp.722/) 和 Deng et al. 2024 (https://arxiv.org/abs/2311.09783) 都 documented benchmark contamination。WiC 和 MNLI 的 negative transfer at small scale 可能部分由 contamination 解释 — contaminated data 让 PT 阶段看似 high accuracy, 但 SFT reformat 后这些 contamination 失效。

### 6.7 与你的 "Building Intuition" 视角

构建 intuition 的关键 takeaways:

1. **Pretraining 决定 calibration fingerprint, SFT 决定 task-specific output adaptation** — 这两个是 dissociable dimensions
2. **Capacity 决定 calibration flexibility** — small model 用 universal calibration, large model 用 task-specific calibration
3. **Task type 决定 transfer reliability** — factual knowledge (Science) transfer well, nuanced linguistic reasoning (NLI, Semantic) 被 SFT reorganize
4. **Proxy model decisions 需要 multi-scale validation** — 240M 上的 data mixture 选择可能在 1B 上反转 (NLI 是 extreme case)
5. **Benchmark 不是 category representative** — 单个 benchmark (e.g., HellaSwag) 不能代表 category, 必须 suite-level evaluation

---

## 7. 实用建议汇总

### 7.1 Benchmark Selection (Implication 1)

**Reliable early-stage predictors**:
- Commonsense: HellaSwag, PIQA, COPA
- Science: ARC-Easy, SciQ (moderately reliable), ARC-Challenge at 1B
- Semantic: QQP

**Unreliable (avoid for proxy decisions)**:
- WiC, WSC, MultiRC (negative or near-zero transfer)
- WinoGrande, MNLI at 240M (negative, but recover at 1B)

### 7.2 Confidence as Complementary Signal (Implication 2)

At 240M, confidence patterns transfer reliably — 可以用 PT confidence 选 SFT-ready models。At 1B+, **必须做 explicit task-specific calibration** during/after SFT, 因为 PT confidence 不再 transfer。

### 7.3 Multi-Scale Data Curation (Implication 3)

Don't extrapolate data curation decisions from small proxy models. Validate at minimum 2 scales before committing to production.

特别 for NLI 和 Semantic 类 task, FineWeb-Edu 在 240M 和 1B 上效果反转, 你需要做 multi-scale check 才能确定 educational filtering 是否 beneficial。

### 7.4 Calibration-Aware Data Curation

如果你想 deploy 一个 well-calibrated model:
- **Science**: 任何 mixture 都 OK, calibration 普遍高
- **Commonsense / Semantic**: 需要 post-hoc calibration (temperature scaling)
- **NLI**: SFT 后必须重新做 calibration, 因为 PT calibration 不 persist

### 7.5 Code Data Trade-offs

- Physical reasoning tasks (HellaSwag, PIQA) — code data **hurts**
- Abstract logical / social reasoning — code data **helps**
- 考虑到 code 通常占 pretraining 大头 (e.g., 30-50%), 这个 trade-off 在 production LLM 中很重要

---

## 8. Paper 局限和 Future Directions

### 8.1 Scale 局限

240M 和 1B 还在 small LM regime。Paper 自己也提到 inverse scaling trend 在 7B+ 是否 persist/amplify/reverse 是 open question。我猜测: 7B+ 可能看到 confidence transfer 进一步 degrade, 因为模型 capacity 更大, task-specific calibration profile 更 distinct。

### 8.2 SFT Data 局限

只用 Tulu-v2-mix, 没测 multi-stage post-training (SFT → DPO, SFT → RLHF)。你的 work 中提到过 RLHF 改变 model behavior 显著, 这个 paper 没覆盖这部分, 一个重要 extension 是测 RLHF 后 transfer 是否更不稳定。

参考: DPO (Rafailov et al.)
参考: RLHF 综述 (Wang et al. 2025 RL for reasoning): https://arxiv.org/abs/2504.20571
参考: DeepSeek-R1 (Guo et al. 2025 Nature): https://www.nature.com/articles/s41586-025-09422-z

### 8.3 Benchmark Coverage 局限

没覆盖 long-context reasoning, code generation, multi-turn dialogue, safety。这些是 production LLM 的 critical capability, transfer dynamics 可能完全不同。

### 8.4 Mixture Granularity 局限

9 个 mixtures 是 sparse sampling of mixture space。Finer-grained variations (e.g., 5% increment in code proportion) 可能 reveal non-monotonic effects — Table 9 已经在 NLI 上看到这一点 (35% code 比 25% 和 45% 都差)。

### 8.5 Missing Calibration Metrics

Paper 用 $r_{\text{align}}$ 作为 calibration proxy, 但没对比 standard metrics:
- **ECE** (Expected Calibration Error)
- **Brier score**
- **NLL**
- **Selective prediction AURC**

$r_{\text{align}}$ 是 rank-based across benchmarks, 而 ECE 等是 per-sample based。这两类 metric 的 divergence 可能解释一些 finding。

### 8.6 Theoretical Connection to Mechanistic Interpretability

Paper 的 correlation lens 是 behavior-level。Mechanistic interpretability 视角 (e.g., circuits, attention head specialization) 可能解释:
- 为什么 Commonsense 和 Science 形成 "confidence block"
- 为什么 NLI 经历 fundamental reorganization during SFT
- 为什么 WSC 是 "black sheep"

参考: Anthropic circuits work, Olah et al.
参考: Kunstner et al. attention attribution

---

## 9. 与最近相关 work 的 connection (hallucination-tolerant associations)

### 9.1 DataComp-LM (Li et al. 2025)

DCLM 是 paper 中三个 web source 之一 (V3)。DataComp-LM 强调 systematic data curation pipeline, 这篇 paper 的发现 (DCLM 在多 category 上 competitive with FineWeb-Edu) 支持 model-based quality scoring 不必 restrict 到 educational content。

### 9.2 DoReMi / DoGE (Fan et al. 2024)

DOGE (https://arxiv.org/abs/2310.15393) 用 generalization estimation 重 weight domains。这篇 paper 的发现 (cross-stage transfer task-dependent) 暗示 DOReMi 的 optimal weights 可能也 scale-dependent — 一个 future work 方向。

### 9.3 Textbooks Are All You Need (Gunasekar et al. 2023)

https://arxiv.org/abs/2306.11644 — 小模型用 textbook 数据学 reasoning 更好。这与 paper 中 FineWeb-Edu 在 240M 上对 Science +2.1pp 一致。但 textbook 数据在 large scale 是否还有 marginal benefit 是 open。

### 9.4 Phi-series (Microsoft)

Phi-1, Phi-1.5, Phi-2 都用 synthetic + educational data。这篇 paper 的 scale-dependent findings 暗示 Phi 在 ~1B scale 可能正经历 transfer dynamics 的 transition point。

### 9.5 Synthetic Data Scaling (Kang et al. 2025, DatologyAI 2025)

https://arxiv.org/abs/2510.01631 (Kang et al.): synthetic data scaling laws
https://arxiv.org/abs/2508.10975 (DatologyAI): trillion-scale synthetic pretraining

这两篇都涉及 synthetic data 的 scale-dependent effects。Magic Correlations paper 的 methodology (5 个 correlation protocols) 可以直接 applied 到 synthetic vs real data 的 cross-stage transfer analysis。

### 9.6 Cheng et al. 2025 RL Reasoning Cross-Domain

https://arxiv.org/abs/2506.14965 — RL for LLM reasoning 的 cross-domain analysis。Magic Correlations 的方法可以 extended 到 RL stage (PT → SFT → RL 三阶段 transfer dynamics)。

### 9.7 Bowman & Dahl 2021 Benchmark Construct Validity

https://aclanthology.org/2021.naacl-main.385/ — NLU benchmark 的 construct validity 问题。这篇 paper 的 "black sheep" findings (WSC in Semantic) 直接呼应这个 critique。

### 9.8 HELM Framework (Liang et al. 2023)

https://arxiv.org/abs/2211.09110 — multi-dimensional evaluation。Magic Correlations paper 是 HELM 的一个 complementary lens: HELM 测 absolute performance 多维, 这篇 paper 测 cross-stage ranking 稳定性。

---

## 10. 总结: 这篇 paper 改变了什么 intuition

| Pre-paper Intuition | Post-paper Intuition |
|---|---|
| PT performance reliably predicts SFT performance | **Category-dependent**: Science/Commonsense yes, NLI/Semantic no |
| Scaling improves everything | **Inverse scaling**: accuracy transfer ↑ but confidence transfer ↓ |
| SFT can fix PT calibration issues | **SFT largely preserves PT calibration fingerprint**, except NLI |
| Educational filtering is universally good | **Scale-dependent**: helps factual tasks, hurts linguistic tasks at small scale, reverses at large scale |
| Single benchmark represents category | **Intra-category competition**: HellaSwag ≠ Commonsense; WiC ≠ Semantic |
| Proxy model decisions extrapolate | **Don't**: MNLI effect reverses 240M→1B |
| Larger models = more transferable representations | **True for accuracy, false for calibration** |

---

## 11. 我对 paper 的最终评价

**Strengths**:
1. 干净的 factorial experimental design (3 sources × 3 proportions)
2. 5 个 complementary correlation protocols 揭示 transfer 的 multi-dimensional nature
3. Striking finding (inverse scaling, alignment persistence) 都是 actionable
4. Honest about limitations (small scale, single SFT dataset)

**Weaknesses**:
1. Scale 太小 (240M, 1B) — 在 Llama-class scale (7B+) 不一定 generalize
2. $r_{\text{align}}$ 不是 standard calibration metric, 与 ECE 关系未明
3. Per-sample analysis 缺失 (e.g., per-sample confidence shift through SFT)
4. 没有 mechanistic interpretation — 为什么 NLI 经历 reorganization?
5. 单一 SFT dataset — 不同 SFT data composition 可能完全改变 transfer dynamics

**My take**: 这是一篇 **foundational methodology paper**, 5 个 correlation protocols 是 reusable framework。具体 findings 在更大 scale 上可能 shift, 但 methodology 是 lasting contribution。最 actionable 的 finding 是 **multi-scale validation for data curation**, 这直接挑战了 DOReMi 等 small-proxy-based 数据 selection 方法的 validity。

---

参考链接汇总:
- Paper: https://arxiv.org/abs/2507.00885 (Lourie et al. 2025, related)
- Kaplan Scaling Laws: https://arxiv.org/abs/2001.08361
- Chinchilla: https://arxiv.org/abs/2203.15556
- Hernandez Transfer Scaling: https://arxiv.org/abs/2102.01293
- RefinedWeb: https://arxiv.org/abs/2306.01116
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- DCLM: https://arxiv.org/abs/2406.11794
- StarCoder: https://arxiv.org/abs/2305.06161
- The Stack v2: https://arxiv.org/abs/2402.19173
- RedPajama-v2: https://arxiv.org/abs/2411.12372
- Tulu-v2-mix: https://arxiv.org/abs/2311.10702
- Guo Calibration NN: https://arxiv.org/abs/1706.04599
- Kadavath LMs know: https://arxiv.org/abs/2207.05221
- Tian Just Ask: https://arxiv.org/abs/2305.14975
- Geng Survey Calibration: https://aclanthology.org/2024.naacl-long.366/
- Desai Durrett: https://aclanthology.org/2020.emnlp-main.21/
- HELM: https://arxiv.org/abs/2211.09110
- Bowman Dahl NLU benchmark: https://aclanthology.org/2021.naacl-main.385/
- Sainz Contamination: https://aclanthology.org/2023.findings-emnlp.722/
- Deng Contamination: https://arxiv.org/abs/2311.09783
- TRAK: https://arxiv.org/abs/2303.14186
- Influence Functions: https://arxiv.org/abs/2308.03296
- DOReMi: https://arxiv.org/abs/2305.10429
- DOGE: https://arxiv.org/abs/2310.15393
- Textbooks: https://arxiv.org/abs/2306.11644
- Lourie Scaling unreliable: https://arxiv.org/abs/2507.00885
- Schaeffer Frontier: https://openreview.net/forum?id=I1NtlLvJal
- Cheng RL Cross-Domain: https://arxiv.org/abs/2506.14965
- Kang Synthetic: https://arxiv.org/abs/2510.01631
- DatologyAI Synthetic: https://arxiv.org/abs/2508.10975
- DeepSeek-R1: https://www.nature.com/articles/s41586-025-09422-z
- RL for Reasoning (Wang): https://arxiv.org/abs/2504.20571

希望这个深度解读能 build 起你的 intuition, Andrej。最 striking 的 takeaway 我认为是 **inverse scaling between accuracy and confidence transfer** — 这是一个真正 fundamental 的发现, 暗示我们一直用单一 metric (accuracy) 来理解 LLM capability transfer 是 oversimplified。Confidence / calibration 维度独立演化, 而且 scale 的作用方向相反。
