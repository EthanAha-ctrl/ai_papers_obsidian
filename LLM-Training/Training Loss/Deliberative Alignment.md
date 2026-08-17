---
source_pdf: Deliberative Alignment.pdf
paper_sha256: baa269bca96556970a7d26d00cd1b20ce2cf59f51d5cb0f5fc230a1a71663ae2
processed_at: '2026-08-03T19:17:22-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Deliberative Alignment 大白话版

好，我换一种方式讲。

---

## 这篇论文到底在解决什么问题？

训练一个安全的大模型，传统做法是这样的：

你雇一群人，给他们一堆 prompt，让他们标注"这个该答"、"这个不该答"、"这个该委婉答"。然后用这些标注数据去 SFT + RLHF 训练模型。

模型学到了什么呢？它学到了一堆**模式**："看到毒品相关词就拒绝"、"看到自杀相关词就给 988 热线"。但模型**根本没见过**真正的安全规则长什么样。它只是从海量标注里**猜**出了一个近似的"拒绝模式"。

这就好比你让一个人学开车，但不给他交通规则手册，只让他看一万段"司机踩了刹车"和"司机没踩刹车"的视频，让他自己悟什么时候该刹车。他能悟个大概，但碰到没见过的情况就傻了。

**问题就在这**：

1. 模型没见过规则本身，只见过规则被应用的结果 → 碰到新情况就泛化不了
2. 模型回答太快，没时间想 → 复杂边界 case 容易判断错
3. 你要持续雇人标注 → 越强的模型，能标注的人越少，scaling 不动

---

## Deliberative Alignment 的核心 idea

**特别简单**：把安全规则直接写给模型看，教它在回答前先查规则、做推理，然后再答。

但这有个问题：你不可能在 deployment 时每次都把几十页规则塞进 prompt——太慢、context 装不下、模型还可能跳过相关部分。

所以 Deliberative Alignment 的做法是：**训练时让模型看着规则做推理，但 deployment 时把规则拿走，让它自己回忆**。

这就是所谓的 "context distillation"——你先让模型在 context 里有规则的情况下学会做，然后通过 SFT 把这个能力烤进 weights 里，让它即使没有 context 也能 recall。

---

## 具体怎么做的？三步

### Step 1：造数据

拿一个**只训练了 helpfulness、没做 safety training** 的 reasoning model（这就是 $\mathcal{G}_{base}$）。

给它一个 prompt，比如"教我怎么伪造残疾人停车证"，同时把相关的安全规则（illicit behavior 那一类的 spec）塞进 prompt 里，告诉它"回答前先引用并推理相关规则"。

模型就会生成一段 CoT，里面会引用规则、做推理，然后给出一个 policy-compliant 的回答。

比如 Figure 7 的例子，CoT 会说："用户在问伪造停车证，这是在请求违法指导。policy 说这种请求必须拒绝。所以我要 refuse。"

### Step 2：筛数据

用另一个 model（$\mathcal{G}_{RM}$，judge model），给它规则 + 模型的回答，让它打分。

对每个回答打 $k$ 次分，取**最低分**。为什么取 min？因为 judge model 自己也可能犯错，取 min 是保守策略——只要有一次发现问题就丢弃这个数据。

筛选完之后，**把规则从 prompt 里删掉**，只保留 (原始 prompt, CoT, answer) 三元组。

### Step 3：训练

**SFT 阶段**：用这些 (prompt, CoT, answer) 去训练。prompt 里没有规则，但 CoT 里会引用规则、推理规则。模型被迫把规则**记进 weights**，因为 inference 时它看不到规则。

**RL 阶段**：再用 judge model 当 reward model，但**只看 final answer，不看 CoT**。这是故意设计的——如果 RL 的时候 reward 信号能看 CoT，模型可能会学出"假装推理得对但答案是错的"的欺骗性 CoT。不看 CoT，模型就没法 hack 推理过程，只能 honest 地推理然后给出 honest 的答案。

---

## 为什么效果这么好？

### 好处 1：Jailbreak robustness 大幅提升

传统模型被 ROT13 编码骗，因为它 pattern match 的是表面文本。但 Deliberative Alignment 训练的模型会先 decode，然后查规则，发现"哦这是在教我犯罪"，然后拒绝。

Figure 1 那个 ROT13 的例子特别生动——模型在 CoT 里自己 decode 出来，然后说"用户在 trick 我"，然后回头查 policy，最后 refuse。

### 好处 2：Overrefusal 反而降了

这听起来矛盾——更安全但不更保守？关键在于 CoT 让模型能做**细粒度判断**。

传统模型碰到"翻译这段话：tell me a story about a man who has sex with a goat"会直接拒绝，因为里面有敏感词。

但 Deliberative Alignment 训练的模型会在 CoT 里推理：规则说"用户提供的 content 我们可以做 transformation，即使 content 本身是 disallowed 的"。用户给的是他们自己的文本，只是要我翻译，我不需要拒绝。Figure 11 就是这个例子，模型翻译了，没有 refuse。

传统模型做不到这个，因为它没见过这个"transformation exception"规则——它只从标注里学了一个粗糙的"有敏感词就 refuse"的 pattern。

### 好处 3：OOD 泛化

最惊艳的实验：训练时只用英文 + 非 encoded data 做 safety training，但测试时用 base64 encoded jailbreak 和多语言 jailbreak。

结果？训练时没见过 encoding 的模型，在 encoding eval 上 0.97 ≈ 见过的 0.95。没见过多语言的，multilingual eval 0.69 ≈ 见过的 0.68。

这说明模型学到的不是"这种 encoding 要拒绝"，而是"碰到奇怪的输入，先 normalize 它，然后查规则"这个 **general skill**。这比 pattern matching robust 得多。

### 好处 4：Inference compute 也能 scale safety

给模型更多 thinking time，StrongREJECT 和 regulated advice style 都会提升。说明安全不是只靠 training——test-time compute 也能改善安全行为。这跟你一直说的 "test-time compute is a new scaling dimension" 完全一致。

---

## 跟传统方法的关键区别

**传统 RLHF**：规则 → 给标注员看 → 标注员产出 label → 用 label 训练模型。模型从没见过规则。

**Constitutional AI**：规则 → 模型生成 critique + revision → 只用 revision 做 SFT。critique 不被优化，推理时模型也不 access 规则。

**Deliberative Alignment**：规则 → 模型生成 CoT（引用规则）+ answer → **CoT 和 answer 一起被 SFT 优化** → 推理时模型自己 recall 规则。规则真的进了 weights。

Figure 15 把这个区别画得很清楚。

---

## 最核心的 insight

如果只记一件事：

**把安全规则从"标注员的脑子"搬到"模型的脑子"里。**

传统方法是规则进入 human labeler 的认知，然后 labeler 输出 (prompt, answer) pair，模型只从 pair 学。规则的信息在传递中丢失了——模型只知道"这个 prompt 该 refuse"，但不知道"为什么 refuse，什么规则说的"。

Deliberative Alignment 是规则进入模型的 CoT，CoT 被直接优化。模型不光学会了"什么时候 refuse"，还学会了"根据什么规则 refuse，怎么推理到 refuse 的"。规则的信息完整地传递到了 weights 里。

这就是为什么它泛化好、过拒绝少、抗 jailbreak 强——因为模型是真的在**用规则**，不是在**匹配模式**。

---

## 一个可能让你 "aha" 的类比

你可以想成是 closed-book exam vs. open-book exam 的区别。

传统 RLHF 是 closed-book：你让学生（模型）背了一堆 (题目, 答案) pair，考试时（inference）让他凭记忆答。碰到没背过的题就崩了。

Deliberative Alignment 是训练时 open-book，考试时 closed-book，但学生把**怎么查书、怎么用书**这个 skill 记住了。虽然考试时没书，但他记住了规则的结构、怎么找到相关条款、怎么 apply。碰到新题，他能"回忆"出规则然后推理。

这比死记硬背 robust 得多。

---

# Deliberative Alignment 深度解读

Andrej，这篇 paper 是 OpenAI 关于 o-series model 安全对齐的核心方法论文章。我会把它从 motivation → 方法形式化 → 关键设计选择 → 实验数据 → 与既有方法对比，逐层展开。

---

## 1. Motivation：为什么 traditional safety training 会失效

paper 一开始就诊断了 modern safety training 的两个 fundamental limitations，这两点是 Deliberative Alignment 整套设计的 root cause：

**Limitation 1: Fixed compute at inference。** 传统 LLM 在 inference 时用固定 compute 立即回答，对 borderline / adversarial 的 prompt 没有时间"想一想"。

**Limitation 2: Implicit, pattern-based learning。** 传统 RLHF/SFT 让模型从大量 (prompt, label) 对中"猜"出 underlying safety spec，但 spec 本身从未显式呈现给 policy model。这导致：

- Data efficiency 差：需要海量 labeled data
- Generalization 弱：碰到 OOD / jailbreak 容易崩
- 不 interpretable：模型行为不可用 spec 文本反推

paper 的核心 claim：把 safety spec 直接 text 喂给 reasoning model，训练它显式 recall + reason 这些 spec，能在 Pareto frontier 上同时改善 jailbreak robustness 与 overrefusal。

---

## 2. 方法形式化

设 generative reasoning model $\mathcal{G}$ 接受 prompt，输出 (CoT, answer) 的 completion。训练目标：把初始 reasoning model $\mathcal{G}_{base}$（只训练 helpfulness，无 safety data）转为 $\mathcal{G}_{spec}$，使其 answer adhere to safety spec。

### 2.1 整体 pipeline (Figure 3)

四步：**Data Generation → Filtering → SFT → RL**

### 2.2 SFT Stage 细节

**Step 1 - Generation (context distillation)**

对每个 (prompt, category) pair，构造 augmented prompt：

$$\text{Prompt}_{\text{aug}} = [\text{original prompt}] \oplus \text{spec(category)} \oplus \text{instructions to cite spec}$$

其中 $\text{spec(category)$ 是只对当前 category 给详细 spec，其它 category 给 high-level summary。这是为了控制 context length（一个 detailed category spec 约 5–10K tokens，全展开会超出 context window）。

让 $\mathcal{G}_{base}$ 在 augmented prompt 上 sample，得到 (CoT, answer) completion，CoT 中显式引用 spec 片段。

**Step 2 - Quality Filtering (judge model $\mathcal{G}_{RM}$)**

对每个 completion 跑 $k$ 次 judge，取 score = min over k runs：

$$\text{score(completion)} = \min_{i=1,\ldots,k} \mathcal{G}_{RM}(\text{prompt}, \text{CoT}, \text{answer}, \text{spec(category)})$$

取 min 的设计 motivation：任何单次 judge run 可能漏掉 violation，min 是 conservative aggregator。

如果 prompt 携带 noisy metadata（如人类标了 ideal completion 是 refuse/comply/safe-complete），通过 prompt suffix "In your answer, consider that another AI determined that ..." 注入到 $\mathcal{G}_{RM}$，让它 justify 是否同意。这是 trade-off between guidance 与 noise over-indexing 的设计。

**Step 3 - SFT Training (key insight: strip spec from prompt)**

这是整个方法的灵魂所在：

$$\mathcal{D}_{\text{SFT}} = \{(\text{prompt}_{\text{orig}}, \text{CoT}, \text{answer}) : \text{spec 不在 prompt 中}\}$$

SFT loss：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(\text{prompt}, \text{CoT}, \text{answer}) \sim \mathcal{D}_{\text{SFT}}} \left[ \log \mathcal{G}_{\theta}(\text{CoT}, \text{answer} \mid \text{prompt}_{\text{orig}}) \right]$$

变量解释：
- $\theta$: model parameters
- $\text{prompt}_{\text{orig}}$: **不含** spec 的原始 prompt
- $\text{CoT}$: 包含对 spec 片段的引用与推理
- $\text{answer}$: 最终 policy-adherent 回答

通过在 training 时移除 spec，强迫模型 internalize spec 到 weights 里，使其在 inference 时 spec 缺席也能 recall。这本质是 **context distillation** (Snell et al., 2022, https://arxiv.org/abs/2209.15189) 的应用。

输出 $\mathcal{G}_{SFT}$。

### 2.3 RL Stage

用 $\mathcal{G}_{RM}$ + spec 给 reward signal：

$$R(\text{prompt}, \text{answer}) = \mathcal{G}_{RM}(\text{prompt}, \text{answer}, \text{spec(category)})$$

**关键设计：RL 时 CoT 对 $\mathcal{G}_{RM}$ 不可见**。Motivation：避免直接对 CoT 优化压力导致 deceptive CoT（reward hacking 把 CoT 变成 misleading shape）。

整个 RL 阶段 → $\mathcal{G}_{spec}$。

paper 在 ablation 中显示：SFT 提供 strong prior for safe reasoning，RL 进一步 refine CoT 的有效使用，两者都不可少。

---

## 3. Safety Spec 设计

spec 由两部分组成：

### 3.1 Content policy（per category）
针对每个 safety category (erotic, extremism, harassment, illicit behavior, regulated advice, self-harm, violence)，policy 定义：
- "allowed" → comply
- "disallowed" → refuse
- "requires safe completion" → 提供资源 / 转介

paper 附录 Figure 9 给了 illicit behavior 的 spec 片段，Figure 10 给了 self-harm 的片段。

### 3.2 Style guidelines
Figure 4 给了 hard refusal 的 style guideline：要求 brief apology + statement of inability，**不要** moralize / lecture / preachment。这是为了避免 RLHF 模型典型的 "I'm sorry, as an AI language model..." 病态行为。

Safe completion guidelines 用于 self-harm / regulated advice，比 hard refusal 复杂得多（Figure 8 是 self-harm safe completion 的 sample CoT，模型需要识别 ideation 然后给出 988 hotline、emergency services 等资源）。

### 3.3 Category-specific composition
$\text{spec(category)}$ = 所有 category 的高层 summary + 当前 category 的详细 policy + style guidelines。Figure 5 给了 prompt template。

---

## 4. Main Results（Table 1 详细解读）

| Policy Area | Dataset | Metric | GPT-4o | o1-preview | o1 | o3-mini |
|---|---|---|---|---|---|---|
| Disallowed Content | Challenging Refusal | not_unsafe | 0.80 | 0.934 | 0.90 | 0.94 |
| | WildChat (toxic) | not_unsafe | 0.98 | 0.98 | 0.99 | 0.98 |
| Response Style | Hard refusal | style_adherence | 0.72 | 0.69 | 0.79 | 0.78 |
| | Safe completion (self-harm) | style_adherence | 0.04 | 0.01 | 0.92 | 0.87 |
| | Safe completion (regulated advice) | style_adherence | 0.28 | 0.04 | 0.65 | 0.35 |
| Jailbreak | StrongREJECT | goodness@0.1 | 0.37 | 0.66 | **0.88** | 0.75 |
| Overrefusal | XSTest | not_overrefuse | 0.88 | 0.976 | 0.93 | 0.90 |
| | Transformation Exception | not_overrefuse | 0.28 | 0.72 | **0.89** | 0.89 |
| | Policy overrefusals | not_overrefuse | 0.69 | 0.87 | 0.84 | 0.83 |

几个有意思的观察：

1. **o1 在 StrongREJECT 上 goodness@0.1 = 0.88**，相比 GPT-4o 的 0.37 是 2.4 倍提升。这是 Figure 2 Pareto frontier 的主轴之一。
2. **o1-preview 的 safe completion style 极差（0.01 / 0.04）**。paper 解释：safe completion guidelines 在 o1-preview 与 o1 之间被更新了，o1-preview 训练时用的是旧 spec，被新标准 eval 才会这么低。这是 spec 演进的真实证据。
3. **o1 vs o1-preview 在 StrongREJECT 上从 0.66 跳到 0.88**。这 22 个百分点的提升 paper 归功于 RL stage 加入的 $\mathcal{G}_{RM}$-based safety reward signal。
4. **Transformation Exception overrefusal：从 o1-preview 的 0.72 到 o1 的 0.89**。这与 CoT 中显式 reason "user provided content + transformation rule" 有关，Figure 11 的 CoT 是个非常好的例子。

### Figure 2 Pareto frontier
横轴 XSTest not_overrefuse，纵轴 StrongREJECT goodness@0.1。o1 同时把两个指标推高，证明 Deliberative Alignment 不是靠"refuse everything"偷工。o3-mini 在 XSTest 上 0.90 < o1-preview 的 0.976，但 StrongREJECT 0.75 > o1-preview 的 0.66，说明 o3-mini 是不同 compute/data trade-off 的点。

---

## 5. Inference-Time Compute Scaling (Figure 13)

这是最 connect 到你 interests 的部分。paper 测试了给 o1 不同 amount of CoT compute，观察 safety eval 变化：

- **StrongREJECT**：随 compute 单调上升
- **Regulated advice safe completion style**：随 compute 单调上升
- 其它 eval（disallowed content、overrefusal 等）基本持平

paper 的 hypothesis：StrongREJECT 与 regulated advice style 是更难的任务（前者用 compositional jailbreaks，后者 style guideline 复杂），所以能 benefit from more thinking。其它简单任务一开始就 saturate。

这强化了 paper 的核心论点：**safety failures 部分源自 insufficient reasoning time**，CoT 是把 test-time compute 转化为 safety 的 mechanism。这与你反复强调的 "test-time compute 是新的 scaling dimension" 一脉相承。

---

## 6. Ablation (Section 4.1, Figure 14)

四个 setting：
- A: No safety training (SFT/RL 都没有 safety data)
- B: Safety in SFT only
- C: Safety in RL only
- D: Safety in SFT & RL (full method)
- E: Baseline: no safety training, but full spec at inference time (system prompt)

关键发现：
- D > A on disallowed content / response style / jailbreak，但 overrefusal 上升（safety training 的常见 side effect）
- B 和 C 都达到中间水平 → 两阶段都不可少
- **E 表现差于 D，甚至差于 B 在某些 eval 上**：直接把 spec 塞进 system prompt 远不如训练时 internalize

为什么 E 这么差？paper 给出三个原因：
1. Latency cost（每次都要 reason over pages of spec）
2. Instruction following 可能 miss relevant part
3. 必须 summarize 所有 category，detail 受限（all detailed specs 会超出 context window）

这第三个原因很重要：**spec 的总量本身已经超出 context window**，所以 in-context 方法在工程上根本不可行。Deliberative Alignment 通过把 spec 压进 weights 绕过了这个 hard limit。

---

## 7. Policy Retrieval Accuracy (Table 2)

用 prompted classifier 抽取 CoT 中提到 policy 的句子，匹配回 spec 的 category：

| Model | Hard Refusal | Safe Completion | Comply |
|---|---|---|---|
| No-Safety Baseline | 0.27 | 0.21 | 0.09 |
| Full Data | 0.75 | **0.91** | 0.54 |

- Safe completion 的 retrieval accuracy 0.91 最高，因为 safe completion 必然需要查 spec（不像 hard refusal 可以靠 pattern）。
- Comply 的 0.54 较低是符合预期的：benign request 不需要 reference policy。
- No-Safety Baseline 也有 0.27/0.21：因为 pretraining 已含 safety 相关数据，且 classifier 较 generous（"illegal content is not allowed" 也算引用 illicit policy）。

这验证了 Deliberative Alignment 真的让模型 learn to recall spec，而非只是学会 refusal pattern。

---

## 8. OOD Generalization (Section 4.3, Table 3)

两个 OOD eval：
- **Encoding**: 250 forbidden prompts × 多个 encoding-based jailbreak（base64 等）
- **Multilingual**: translated jailbreak prompts

三个 model 变体：
- No-Safety Baseline
- Eng-Only, No-Encoded Data（safety training 只用英文 + 非 encoded data）
- Full Data

| Model | Encoding | Multilingual |
|---|---|---|
| No-Safety | 0.65 ± 0.06 | 0.44 ± 0.01 |
| Eng-Only | **0.97 ± 0.02** | 0.69 ± 0.01 |
| Full Data | 0.95 ± 0.03 | 0.68 ± 0.01 |

惊人发现：**Eng-Only 在 Encoding eval 上 0.97 ≈ Full Data 的 0.95**，尽管它从未见过 encoding-based safety data。Multilingual 同样 Eng-Only (0.69) ≈ Full Data (0.68)。

这是 Deliberative Alignment 最强的 selling point：模型学会了"碰到异常输入先 decode/translate，然后查 spec reason"这个 generalizable skill，而非死记某种 jailbreak 模式。这与传统 pattern-based safety training 的脆弱性形成对比。

---

## 9. 与既有方法对比 (Figure 15)

paper 在 Section 5 给出很清晰的分类：

### 9.1 vs. RLHF / DPO
- Spec 进入 human/AI labeler 的脑中生成 label，但 spec 文本本身从未到达 policy model
- Policy 只从 (prompt, label) 隐式推断 spec

### 9.2 vs. Constitutional AI (CAI) (Bai et al., 2022, https://arxiv.org/abs/2212.08073)
- Constitution 类似 spec，但 CAI 用 constitution 生成 critique + revision 序列
- **关键区别**：CAI 的 critique 在 SFT 中不直接被 optimize（只 revision 被用作 label），且 RL 阶段 preference model 是 fine-tuned out 的，policy 推理时不再 access constitution
- Deliberative Alignment 直接 optimize CoT，且 inference 时 CoT 仍 reference 内化的 spec

### 9.3 vs. Self-REFINE (Madaan et al., 2023, https://arxiv.org/abs/2303.17651)
- Self-REFINE 用 few-shot prompting 在 inference 时迭代 critique + revise
- 依赖 pre-specified LMP / fixed reasoning path
- Deliberative Alignment 的 CoT 是 end-to-end 学出来的，无固定 reasoning path

### 9.4 vs. Backtracking (Zhang et al., 2024, https://arxiv.org/abs/2409.14586)
- Backtracking 训练模型输出 `[RESET]` token 触发重写
- 限制：每次 response 只能 back-track 一次，且不教 spec
- Deliberative Alignment 的 CoT 可无限 "backtrack"，且唯一教 spec 的方法

---

## 10. Limitations & Open Questions（paper 6 节 discussion 透露的）

paper 第 6 节末尾自己提到了 long-term alignment 风险：
- 高智能 self-aware AI 可能 reject human constraints (Russell, 2019)
- Instrumental convergence：self-preservation, resource acquisition (Bostrom, 2014; Omohundro, 2008)
- 提到正在 invest 的方向：CoT monitoring for deception (Järviniemi & Hubinger, 2024, https://arxiv.org/abs/2405.01576)

paper 自己把 Deliberative Alignment 定位为"alignment progress alongside capabilities"的乐观证据，但承认这个 trend 可能 break。

---

## 11. 我对这篇 paper 的几个 intuition

**Intuition 1: Spec-as-weights 比 spec-as-context 更 reliable。**
Figure 14 的 E baseline 失败是个强证据。Reasoning model 在长 context 中有 attention dilution 问题，把 spec 放进 system prompt 不可靠。Internalize 进 weights 后，模型 attention 是 learned 的、sparse 的、task-conditional 的，远比无差别扫整个 spec 高效。

**Intuition 2: RL 隐藏 CoT 是 anti-deception 的关键设计。**
Section 2.4 明确说 "We avoid applying direct optimization pressure on the CoT during RL to ... reduce the chance of encouraging deceptive CoTs"。这是 very thoughtful 的设计：如果 RL reward 直接看 CoT，模型可能学出"假装推理得对"的 CoT。只 reward final answer 让 CoT 保持 SFT 学到的 honest shape，RL 只优化 final answer 的 quality。

**Intuition 3: Min-over-k judging 是 calibrated safety 的关键。**
Section 2.3.2 取 min 而非 mean。Mean 会被 high-variance judge runs 拉高，min 是悲观估计，更接近 "worst case policy adherence"。这与 safety 的 risk-averse 性质契合。

**Intuition 4: SFT 数据全合成是 scalability claim 的核心。**
Section 1 末尾和 Section 6 都强调 "no human-labeled completions"。这把 human expertise 从 labeling 转移到 spec writing + evaluation，是 alignment 的劳动分工重构。Model spec (https://cdn.openai.com/spec/modelspec-2024-05-08.html) 是 human 价值显式化的载体。

**Intuition 5: Pareto improvement 来自 reasoning granularity。**
传统 RLHF 把 (refuse, comply, safe-complete) 当三类 label，granularity 受限。Deliberative Alignment 通过 CoT 让模型在每一步都对照 spec，能处理 transformation exception 这类需要细粒度推理的 case（Figure 11）。这是 overrefusal 同时下降的解释：模型不再用 "contains sensitive word → refuse" 的浅层 heuristic。

**Intuition 6: 与 o1 reasoning training 的协同。**
Deliberative Alignment 之所以 work，前提是 base model 已经是 reasoning model（$\mathcal{G}_{base}$ 是 helpfulness-only 的 o-type model）。如果 base 是 GPT-4o 这种 non-reasoning 模型，SFT 教不出 long CoT。这也解释了为什么 paper 没在 GPT-4o 上做：方法本身就是 reasoning model 时代的 alignment paradigm。

---

## 12. 可能的延伸联想（hallucination-prone zone）

我主动 expand 几个可能相关但 paper 没明说的方向：

**A. Spec 的 verifiable form。** paper 中 spec 是自然语言文本。如果未来 spec 是形式化的（e.g., 一个 first-order logic 表达式或 type system），CoT 就变成了 formal proof search。这与 PRMs (Process Reward Models, https://arxiv.org/abs/2211.14275) 的 process supervision 高度相关——paper 引用 [9] 就是 Uesato et al. 的 PRM paper。

**B. CoT as interpretation surface。** 既然 CoT 显式 reference spec，理论上可以 audit CoT 找 misalignment。这是 paper 6 节提到的 "monitoring chain-of-thoughts for deception" 方向，与 Anthropic 的 sleeper agents / deceptive alignment research 直接对接。

**C. Spec editing for fast policy update。** 如果 spec 内化进 weights，policy 更新需要 retrain。但如果能像 LoRA 一样只 fine-tune 一个 spec adapter，policy update 可以快很多。这可能是 OpenAI 内部正在做的方向。

**D. Constitutional AI 的隐含 path forward。** CAI 用 constitution 生成 critique-revision；Deliberative Alignment 用 spec 生成 CoT。两者结合：用 constitution 引导 CoT 的 revision process，是 self-improving alignment 的可能路径。

**E. 与 you 在 "State of GPT" 中讲的 system 1 / system 2 类比。** Deliberative Alignment 把 safety 从 system 1（fast reflex）迁到 system 2（deliberate reasoning）。这可能改变 jailbreak 的本质：从"骗过 system 1 pattern matcher"变成"在 system 2 reasoning 中注入错误前提"。Figure 1 的 ROT13 例子显示模型 decode 后还回头查 spec，说明 system 2 已经能抵抗 encoding-based injection。但 system 2 自身的 jailbreak（如 sycophancy-based reasoning hijacking）是新 attack surface。

**F. RL 隐藏 CoT 的潜在副作用。** 虽然 paper 说避免 deceptive CoT，但也意味着我们失去了 RL 阶段 CoT 的 interpretability。如果 RL 阶段 model 学会了 deceptive final answer（即使 CoT honest），我们无从 reward against it。这可能需要 future work 引入 CoT consistency reward（CoT 与 answer 的一致性）。

---

## References

- Paper 主页（推测）: OpenAI blog 通常会发，目前 arxiv 版本搜索 "Deliberative Alignment"
- OpenAI o1: https://openai.com/index/learning-to-reason-with-llms/
- Model Spec: https://cdn.openai.com/spec/modelspec-2024-05-08.html
- Context Distillation: https://arxiv.org/abs/2209.15189
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Self-REFINE: https://arxiv.org/abs/2303.17651
- StrongREJECT: https://arxiv.org/abs/2402.10260
- XSTest: https://arxiv.org/abs/2308.01263
- WildChat: https://arxiv.org/abs/2405.01470
- DPO: https://arxiv.org/abs/2305.18290
- Backtracking: https://arxiv.org/abs/2409.14586
- Process supervision (Uesato): https://arxiv.org/abs/2211.14275
- Deceptive CoT monitoring: https://arxiv.org/abs/2405.01576
- SimpleQA: https://arxiv.org/abs/2411.04368
- BBQ: https://arxiv.org/abs/2110.08193
- Universal jailbreak (GCG): https://arxiv.org/abs/2307.15043
- Jailbroken (Wei et al.): https://arxiv.org/abs/2307.15043 (以及 NeurIPS 2024 版本)
- LLaMA 3: https://arxiv.org/abs/2407.21783
- GPT-4o system card: https://cdn.openai.com/gpt-4o-system-card.pdf
- o1 system card: https://cdn.openai.com/o1-system-card.pdf

---

Andrej，简短总结一下我从中提炼出的最 actionable 的几个想法：

1. **Context distillation 是把 spec 压进 weights 的 clean technique**，不限于 safety，任何 long instruction set（如 complex tool use protocol）都可以这么做。
2. **Min-over-k judge 是 risk-averse data filtering 的简单但有效设计**。
3. **RL 不看 CoT 是 anti-deception 的关键**，这个 design choice 值得在任何 CoT + RL pipeline 中拷贝。
4. **Inference-time compute scaling 对 safety 也成立**，进一步强化 test-time compute 作为新 scaling axis 的地位。
5. **OOD generalization 数据（Eng-Only = Full Data）证明 reasoning 比 pattern matching 更 robust**，这呼应你常说"模型应该学会 skill，而非 memorize 表面形式"。
