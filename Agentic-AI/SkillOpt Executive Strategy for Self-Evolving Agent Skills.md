---
source_pdf: SkillOpt Executive Strategy for Self-Evolving Agent Skills.pdf
paper_sha256: 87f7f0f323b1671e9202b3ebb1596e909e507c71ecd1b360b0075a5ee1727fe3
processed_at: '2026-08-12T07:11:55-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SkillOpt 人话版

## 一句话总结

**现在大家给agent写skill都是拍脑袋，写完就扔。SkillOpt说：skill本身就该像个可训练的参数，用一套类似SGD的训练流程来优化它——有learning rate、有validation set、有momentum——只不过优化的是一段text而不是weights。**

---

## 问题出在哪

想象你有一个很强的LLM agent，你要让它做spreadsheet操作。你写了一段skill document，告诉它"用Python操作excel，保留格式"。

问题来了：

- 你写的skill可能漏掉了关键细节——比如grader其实是读cell的static value，不是读formula
- 你写完就完了，agent跑砸了你也不知道该怎么改
- 就算让LLM自己改，它可能一改就把之前有用的rule给删了，或者overfit到某个specific case

现有的几种做法都有问题：

**Hand-written skill**：专家写，写得好就好用，写得差就完蛋。而且写完不会变。

**One-shot LLM生成**：让GPT写一段，写完就固定。没法根据实际跑的结果迭代。

**Trace2Skill / EvoSkill这种self-revision**：让模型看自己的trajectory然后自己改skill。听起来不错，但**没有约束**——模型可以随便改，可能越改越烂。没有validation set来check改了之后到底有没有变好。

这就像你做SGD但**没有learning rate限制，没有early stopping，没有validation set**——模型weights可能diverge，可能overfit，可能oscillate。

SkillOpt的核心insight：**skill document应该被当成一个可训练的external state来对待，用weight optimization同等严谨度的方法来优化它。**

---

## 核心类比：Skill = Weights

这个类比是整篇paper的灵魂。

| 深度学习概念 | SkillOpt对应 |
|-------------|--------------|
| Weights $\theta$ | Skill document $s$（一段text） |
| Forward pass | Agent用当前skill执行task，产生trajectory和score |
| Backward pass | Optimizer model看trajectory，分析success/failure，提出edits |
| Learning rate $\eta$ | Edit budget $L_t$（每step最多改几处） |
| Mini-batch | Rollout batch + reflection minibatch |
| Validation set | Selection split $D_{\mathrm{sel}}$ |
| Early stopping / checkpoint selection | Validation gate（只accept严格改善的edit） |
| Momentum | Epoch-wise slow/meta update |
| Negative gradient signal | Rejected-edit buffer |
| Cosine LR schedule | Cosine edit budget schedule |
| Gradient accumulation | 多个rollout batch独立reflect后合并 |

这个类比**不只是装饰性的**（operational rather than decorative）。每一个深度学习的control mechanism都有text-space的对应物，而且都起着类似的stabilization作用。

---

## Forward Pass

$$
(\tau(s), r(s)) = h(M, x, s), \quad r(s) \in [0,1]
$$

翻译成人话：

- $h$ = harness（你用什么方式跑这个agent，比如direct chat、Codex CLI、Claude Code CLI）
- $M$ = frozen target model（你不动它的weights，比如GPT-5.5）
- $x$ = 一个task实例，比如一个spreadsheet题目
- $s$ = 当前skill document
- $\tau(s)$ = 跑出来的trajectory（所有的messages、tool calls、observations、final answer）
- $r(s)$ = 分数，0到1之间

这就是forward pass——你拿当前skill让agent跑一批task，收集trajectory和score。相当于在当前weights下做一次forward，得到loss。

---

## Backward Pass

这是最精妙的部分。Optimizer model（另一个LLM，比如GPT-5.5当teacher）看这批trajectory，做三件事：

### 1. 分离success和failure

把rollout batch分成成功的trajectory和失败的trajectory。然后各自分成reflection minibatch（default size = 8）。

**为什么要分minibatch而不是看单个trajectory？**

单个trajectory只能告诉你"这个case怎么修"——这是anecdotal fix。minibatch能暴露**recurring patterns**：比如agent在8个失败的case里都犯了同样的错误——总是search wrong source，或者总是write wrong format。这才是你想要encode到skill里的东西。

### 2. 分别分析

- **Failure minibatch** → 提出corrective rules（"agent应该做X而不是Y"）
- **Success minibatch** → 提出preserving rules（"agent在成功case里做了Z，这个behavior应该保留"）

每个analysis返回structured edits：add、delete、replace。

### 3. Hierarchical merging

多个analyst worker并行分析不同minibatch，产生多组edits。然后hierarchical merge：

1. 先merge所有failure edits → 一组consolidated failure edits
2. 再merge所有success edits → 一组consolidated success edits
3. Final merge：**failure patches take priority**，如果failure和success edit说同一件事，保留failure版本

这就像ensemble——多个独立的分析师看不同batch，最后vote出consensus。

---

## Bounded Updates = Learning Rate

合并完edits后，optimizer按expected utility排序，然后**只保留top-$L_t$个edits**。

$$
L_t = L_{\min} + \frac{1}{2}(L_0 - L_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

变量解释：
- $L_t$ = step $t$的edit budget（这一step最多应用几个edit）
- $L_0$ = 初始budget（比如4）
- $L_{\min}$ = 最小budget（比如2，cosine decay的floor）
- $t$ = 当前step
- $T$ = 总step数

这就是cosine schedule——开头允许大改，后期收敛到小改。和深度学习的cosine LR decay一模一样。

**为什么bounded这么重要？**

如果你允许unbounded rewrite（就是让模型随便重写整个skill），会发生三件事：
1. **Erase useful rules**：模型可能把之前验证过有用的rule删了
2. **Introduce incompatible instructions**：新加的rule可能和已有rule矛盾
3. **Overfit to local failure**：模型看到几个case犯同样错误就加一个overly specific的rule

Bounded update保证每step只做incremental change，preserve continuity。这就像small learning rate——每step只挪一小步，避免overshoot。

---

## Validation Gate = Early Stopping

Candidate skill生成后，在selection split $D_{\mathrm{sel}}$上评估：

$$
s_{\mathrm{sel}}^{\star} = \arg\max_{s \in \mathcal{C}(D_{\mathrm{tr}})} \frac{1}{|D_{\mathrm{sel}}|} \sum_{x \in D_{\mathrm{sel}}} r(s)
$$

翻译：

- $\mathcal{C}(D_{\mathrm{tr}})$ = 在train split上生成的所有候选skill
- $D_{\mathrm{sel}}$ = selection split（validation set）
- $r(s)$ = skill $s$在task $x$上的score
- $s_{\mathrm{sel}}^{\star}$ = 在validation set上表现最好的skill

**Gate的规则极其严格**：candidate skill必须**strictly greater than** current score才被accept。Ties被rejected。

为什么这么严格？因为**plausible textual diagnoses可能hurt实际target model**。Optimizer model看trajectory觉得"这个edit应该能修好这个failure"，但实际上：
- Optimizer model ≠ Target model——optimizer认为好的edit可能不适合target的attention pattern
- Fix specific failure可能hurt general performance
- Selection split虽小但真实反映distribution

这个gate是**propose-and-test optimization**而非unconditional self-editing。这是SkillOpt和EvoSkill最大的区别——EvoSkill让模型自己改但没有validation gate。

---

## Rejected-Edit Buffer = Negative Feedback

被reject的edit不会直接扔掉。它们被存到epoch-local buffer $\mathcal{B}$里，记录：
- 被reject的edit是什么
- 它导致了多大的score drop
- 观察到的failure pattern

后续的reflection call会看到这个buffer，optimizer model可以：
1. **避免重复failed edits**——"上次你加了这条rule，score掉了3分，别再试了"
2. **聚焦unresolved failures**——"这些failure你试过修但没修好，换个方向"

这给loop提供**negative feedback**，就像深度学习中negative gradient告诉你"往这个方向走是错的"。

最妙的是这个buffer**只在training时用**，deployment时zero cost——deployed skill就是那个compact的`best_skill.md`，不附带任何buffer。

---

## Epoch-Wise Slow/Meta Update = Momentum

这是ablation中影响最大的component（去掉后SpreadsheetBench掉22.5分）。

**Fast update**从current batch学习，可能noisy。**Slow update**从adjacent epochs学习，capture durable directions。

机制：
1. 每个epoch结束时，sample相同的training items
2. 分别用previous epoch的skill和current skill执行
3. 分成四类：improvements、regressions、persistent failures、stable successes
4. Optimizer model写出longitudinal guidance，注入到skill document的**protected section**

这个protected section被`<!-- SLOW_UPDATE_START -->`和`<!-- SLOW_UPDATE_END -->`标记包围，step-level edits**不能修改它**。只有epoch boundary的slow update process可以overwrite。

**Meta skill**是另一个东西——它完全在optimizer side，不ship给target model。它总结：
- 哪种edit pattern在这个environment里tend to help
- 哪种edit pattern tend to be too vague / redundant / brittle / harmful
- 什么level的abstraction work best
- 什么failure-repair pattern应该prioritize

这个meta skill被prepend到future optimizer prompts里，帮助optimizer产生更好的edits。

**为什么这个这么重要？**

没有slow update时，每个epoch基本独立学习。你第一个epoch学到的durable lesson到第二个epoch可能被step-level edits覆盖掉。Slow update把这些durable lessons锁在protected region里，就像momentum把stable gradient directions跨step保留下来。

---

## 数据Split的严肃性

三个split：

$$
D = D_{\mathrm{tr}} \cup D_{\mathrm{sel}} \cup D_{\mathrm{test}}
$$

- $D_{\mathrm{tr}}$（train）：产生rollout evidence，optimizer从这里学习
- $D_{\mathrm{sel}}$（selection）：validation gate，只在这里accept/reject candidate skills
- $D_{\mathrm{test}}$（test）：**只在最后report用一次**，不参与任何training或selection

最终report的公式：

$$
\mathrm{Test}(s_{\mathrm{sel}}^{\star}) = \frac{1}{|D_{\mathrm{test}}|} \sum_{x \in D_{\mathrm{test}}} r(s_{\mathrm{sel}}^{\star})
$$

翻译：拿在selection split上选出的最佳skill $s_{\mathrm{sel}}^{\star}$，在disjoint的test split上跑一遍，算平均score。

这种三段式split确保了报告的数字测量的是**generalization而非validation-set fit**。这在prompt optimization领域极其罕见——很多paper直接在training set上report，没有held-out test。

---

## 实验结果有多强

### 主结果Table 1的核心数字

**52/52 cells best or tied-best**。

这是什么概念？6个benchmark × 7个model + 2个harness = 52个evaluation cells。SkillOpt在**每一个cell**上都是best or tied-best。没有一次输给任何baseline。

### GPT-5.5的提升幅度

| Benchmark | No skill | SkillOpt | Δ |
|-----------|----------|----------|---|
| SearchQA | 77.7 | 87.3 | +9.6 |
| SpreadsheetBench | 41.8 | 80.7 | +38.9 |
| OfficeQA | 33.1 | 72.1 | +39.0 |
| DocVQA | 78.8 | 91.2 | +12.4 |
| LiveMath | 37.6 | 66.9 | +29.3 |
| ALFWorld | 83.6 | 95.5 | +11.9 |
| **Average** | **58.8** | **82.3** | **+23.5** |

Average提升23.5分。而且这**不改任何weights**——只加了一段300-2000 token的text。

### 对比oracle baseline

Oracle baseline = 每个cell从6个competitor中挑最好的那个。SkillOpt平均超过这个oracle 5.4分。也就是说，哪怕你有一个oracle帮你从所有baseline里挑最好的，SkillOpt还是更好5.4分。

### 小模型benefit更大

| Model | Average Δ |
|-------|-----------|
| GPT-5.4-nano | +26.7 |
| Qwen3.5-4B | +19.2 |
| GPT-5.2 | +16.6 |
| GPT-5.4-mini | +15.4 |
| GPT-5.4 | +12.7 |
| GPT-5.5 | +23.5 |

GPT-5.4-nano在ALFWorld上从34.3涨到69.4（×2.0），在DocVQA上从30.8涨到80.2（×2.6）。

**Intuition**：小模型weights里缺乏特定domain的procedural knowledge。一段compact skill artifact可以supply这些knowledge，相当于给小模型外挂了一个procedural memory。

### Harness结果

| Harness | Δ over no skill | Δ over EvoSkill |
|---------|-----------------|-----------------|
| Codex | +24.8 | +14.0 |
| Claude Code | +19.1 | +3.2 |

EvoSkill在Claude Code上已经很强了（57.8 → 73.7），SkillOpt还能再加3.2分。

---

## Ablation：什么真正重要

### Table 2 Hyperparameter Sweep

最关键的发现是**robustness**——大部分hyperparameter在很大范围内都work：

**Training set size**（Panel a）：
- SpreadsheetBench从1 example的47.5涨到100% train的78.0
- LiveMath从59.1涨到70.5
- SearchQA在20%后就saturate（因为ceiling effect）

**Reflection minibatch size** $B_m$（Panel b）：
- 1到32都基本stable
- Default $B_m=8$基本处处near top

**Rollout batch size** $B$（Panel c）：
- 8到full epoch都work
- 但**full epoch反而下降**（Spreadsheet 75.0, LiveMath 53.2）——说明stochasticity actually helps

**Textual learning rate** $L_t$（Panel d）：
- 1到16都competitive
- LiveMath在$L_t=8$时反而最好（66.9）——不同benchmark最优step size不同

**Schedule**（Panel e）：
- Constant: 87.3/80.7/62.1
- Cosine: 87.1/77.5/61.3
- Linear: 87.2/72.9/62.9
- Bounded-update story不依赖特定scheduler

### Table 3 Component Ablation

| Component | SearchQA | Spreadsheet | LiveMath |
|-----------|----------|-------------|----------|
| Full SkillOpt | 87.1 | 77.5 | 61.3 |
| Without lr (unbounded) | 84.6 | 75.7 | 57.3 |
| Without rejected buffer | 85.5 | 72.9 | 58.9 |
| Without meta skill | 85.1 | 75.7 | 58.1 |
| **Without both meta & slow** | 86.3 | **55.0** | 59.7 |

**Without both meta & slow update**：SpreadsheetBench从77.5暴跌到55.0（−22.5 points）。这是整个ablation suite中最大的degradation。

**Intuition**：没有slow update时，每个epoch基本从零开始学。第一个epoch学到的durable lesson到第二个epoch被step-level edits覆盖。就像做SGD但没有momentum——每step方向noisy且inconsistent，无法build up stable direction。

---

## Transfer Experiments：为什么这很重要

### Cross-Model Transfer

在GPT-5.4上训练的skill直接用到GPT-5.4-mini和GPT-5.4-nano上：

| Direction | Baseline | Direct | Transferred |
|-----------|----------|--------|-------------|
| 5.4 → 5.4-mini (Spreadsheet) | 36.1 | 47.5 | 45.5 (+9.4) |
| 5.4 → 5.4-nano (Spreadsheet) | 23.5 | 42.5 | 26.5 (+3.0) |
| 5.4 → 5.4-mini (LiveMath) | 14.7 | 32.8 | 19.2 (+4.5) |
| 5.4 → 5.4-nano (LiveMath) | 23.2 | 27.2 | 28.8 (+5.6) |

所有transfer都是positive。LiveMath的5.4→5.4-nano transfer（28.8）甚至超过in-domain（27.2）。

**这说明learned rules有model-agnostic的成分**——procedural knowledge不依赖于特定model的attention pattern。

### Cross-Harness Transfer

这是最impressive的结果。

| Direction | Benchmark | Baseline | Direct | Transferred |
|-----------|-----------|----------|--------|-------------|
| Codex → Claude Code | Spreadsheet | 22.1 | 80.4 | **81.8** (+59.7) |
| Claude Code → Codex | Spreadsheet | 27.5 | 85.0 | 71.1 (+43.6) |
| Codex → Claude Code | LiveMath | 40.8 | 56.5 | 42.4 (+1.6) |
| Claude Code → Codex | LiveMath | 35.2 | 78.4 | 48.0 (+12.8) |

**Codex→Claude Code的SpreadsheetBench transfer（81.8）甚至超过in-domain Claude Code SkillOpt（80.4）**。

这说明learned rules encode的是**harness-agnostic procedures**——structure-first inspection、formula-aware verification、static-value materialization这些principle不依赖于特定tool API。在Codex里学到的procedure可以无缝迁移到Claude Code。

**应用价值**：你可以在一个execution environment里花成本训练skill，然后amortize到其他deployment environment。

### Cross-Benchmark Transfer

OlympiadBench上训练的skill用到Omni-MATH上：

| Model | Baseline | Transferred |
|-------|----------|-------------|
| GPT-5.4 | 56.6 | 60.3 (+3.7) |
| GPT-5.4-mini | 34.8 | 36.6 (+1.8) |
| GPT-5.4-nano | 38.8 | 40.1 (+1.3) |

所有transfer都positive但gain较小——因为source和target benchmark只share broad task family（math），test instances和answer-format conventions都变了。

**这说明optimized skill encode的是reusable mathematical procedure而非memorized benchmark-specific formatting**。

---

## Optimizer Strength实验

| Benchmark | Target | Strong (GPT-5.5) | Target-matched |
|-----------|--------|------------------|----------------|
| Spreadsheet | 5.4-mini | +11.4 | +7.1 |
| Spreadsheet | 5.4-nano | +19.0 | +11.9 |
| SearchQA | 5.4-mini | +4.3 | +2.4 |
| SearchQA | 5.4-nano | +19.0 | +14.1 |

**Stronger optimizer always wins**，但target-matched optimizer仍recover 56-74% of strong-optimizer gain。

**关键insight**：SkillOpt不是简单的distillation pipeline。如果只是distillation，target-matched optimizer应该接近0 gain（因为teacher和student一样强）。但实际上target-matched optimizer仍有substantial gain——**optimization loop本身**贡献了很大value，不只是optimizer capacity。

---

## Learned Skill长什么样

### Compactness

| Benchmark | Final tokens | Edits accepted |
|-----------|-------------|----------------|
| SearchQA | 857 | 4 |
| SpreadsheetBench | 1,995 | 4 |
| OfficeQA | 883 | 1 |
| DocVQA | 959 | 3 |
| LiveMath | 379 | 1 |
| ALFWorld | 1,321 | 2 |

**最长的skill也就1,995 tokens**——远低于modern system prompt budget。最短的LiveMath只有379 tokens。

**OfficeQA的+39.0 points来自single accepted edit**。LiveMath的+29.3 points也来自single edit。这是validation gate在做real work的直接证据——optimizer propose了大量edits，但绝大多数被reject，只有1-4个pass了gate并survive到deployed skill。

### Representative Rules

看几条实际学到的rule：

**SearchQA**："Infer the expected answer type from clue wording, then choose the shortest canonical entity supported by co-occurring distinctive evidence."

→ 这是procedural guidance——先infer answer type，再选canonical entity。不是specific question的answer。

**SpreadsheetBench**："Inspect workbook structure and formulas, then write evaluated static values across the full requested target range instead of relying on Excel recalculation."

→ 这是关键的procedural insight——grader读static value不读formula，所以agent必须compute并write evaluated values。这种discipline frontier models zero-shot不具备。

**ALFWorld**："Keep a horizon-aware visited/frontier ledger, diversify search after repeated same-type failures, and avoid revisiting the destination until holding the target."

→ 这是search-frontier management——maintain visited memory，diversify after failures，avoid premature destination revisiting。这是finite-state execution policy的描述。

**关键观察**：每条rule都是procedural而非instance-specific。没有任何rule提到specific question、file、entity。它们读起来像一个thoughtful human practitioner写一天benchmark后会写的——但它们是optimizer自动产生且edit-by-edit validated的。

---

## ALFWorld案例：Skill Evolution的完整故事

**Initial skill**：generic household plan——search target object, pick up, transform if needed, place at destination。

**问题**：agent经常loop——反复检查同一个location，把mugs当成cups，到了destination但还没pick up目标就试图place。

**Optimized skill加了什么**：
1. **Exact object-name matching**——mugs不是cups，pans不是pots，不能互相替代
2. **Visited-location memory**——prefer未visited receptacles over repeatedly检查likely但exhausted locations
3. **Destination memory**——记住destination在哪
4. **Pick-two progress locks**——如果需要pick两个object，pick完第一个才能pick第二个
5. **Direct completion rules**——once agent can clean/heat/cool/place/complete next subgoal, take that admissible action instead of examining/closing/verifying again

**质性转变**：从general search-transform-place strategy → **finite-state execution policy** with object identity、search memory、progress locks、loop breakers。

性能：49.3 → 74.6。

---

## SpreadsheetBench案例

**Initial skill**：use Python spreadsheet libraries + preserve unrelated workbook content。

**问题**：agent rely on previews而不是inspect actual workbook；不知道grader读static value；不fill complete target ranges。

**Optimized skill加了什么**：
1. **Inspect actual workbook**而非rely on previews
2. **Locate headers和target ranges** across multiple sheets
3. **Normalize keys和cell types** before lookup/aggregation
4. **Preserve formatting** during structural edits
5. **Key rule**：when grader reads cell values, agent应compute并write evaluated static values，即使prompt mentions formulas如INDEX/MATCH或XLOOKUP
6. **Fill complete target ranges**包括currently blank result cells
7. **Keep helper computations in Python**而非adding workbook artifacts
8. **Reopen saved workbook**检查boundary rows和remaining blanks

**质性转变**：从generic automation workflow → **workbook-forensics policy**。

性能：40.4 → 78.9。

---

## 为什么这套设计work：深层Intuition

### 1. Skill = External Procedural Memory

Frontier models在weights中encode了大量procedural knowledge，但zero-shot deployment无法激活task-specific procedures。Skill document作为external state，直接进入attention计算，无需改变weights就能steer model behavior。

对小模型尤其有效——它们weights中缺乏特定domain的procedural knowledge，compact skill artifact相当于外挂procedural memory。

### 2. Bounded Updates > Unbounded Rewriting

Small learning rate + many steps > large learning rate + few steps。这是deep learning的基本lesson。SkillOpt把这个lesson搬到text space——bounded edits preserve continuity，avoid overshoot，avoid erasing useful rules。

### 3. Validation Gate是Real Work

Edit economy（1-4 accepted edits）证明gate在filter大量proposed edits。Plausible textual diagnoses可能hurt实际target model——因为optimizer model ≠ target model，因为fix specific failure可能hurt general performance。

Gate把这个gap给bridge了——只accept在held-out data上**严格改善**的edit。

### 4. Slow/Meta Update = Momentum

Fast updates from current batch可能noisy。Slow update aggregates跨epoch的stable directions。没有slow update时每个epoch基本独立学习，无法build up durable procedural lessons。

这就是为什么去掉slow/meta update后SpreadsheetBench掉22.5分——它是整个system的momentum term。

### 5. Procedural > Instance-Specific

Cross-transfer成功的根本原因是learned rules是procedural而非instance-specific。"Inspect workbook structure"这个principle的carrier invariance远高于"在cell A1写=XLOOKUP(...)"这种specific instruction。

Validation gate和bounded edits共同ensure了这一点——overly specific的edits会在selection split上overfit而被reject。

---

## Algorithm 1完整流程

```
输入：frozen M, optimizer O, harness h, splits D_train/D_sel/D_test
      initial skill s_0, epochs E, edit schedule L_t
      rollout batch B, accumulation A, reflection minibatch B_m
输出：best_skill.md + test score

1. 初始化：s_cur = s_0, s_best = s_0, cache = {}, buffer = [], meta = ""
2. 在selection split上评估s_0，得到初始score
3. 缓存s_0的hash → score

4. for each epoch e = 1 to E:
5.   shuffle D_train成rollout batches；清空buffer
   
6.   for each optimization step:
7.     收集A个rollout batch（用s_cur执行h(M, x, s_cur)）
8.     分成failures/successes，再分成size B_m的minibatches
9.     O分析failure minibatches → failure patch proposals
10.    O分析success minibatches → success patch proposals
11.    O merge failure proposals, merge success proposals,
        final failure-prioritized merge
12.    O rank merged edits, 保留top L_t个
13.    Apply这L_t个edits → candidate skill s̃
14.    if s̃的hash在cache里：
15.      直接取cached score
16.    else:
17.      在D_sel上评估s̃
18.      cache s̃的hash → score
19.    end if
20.    if score > current score (严格大于):
21.      s_cur = s̃
22.      if score > best score:
23.        s_best = s̃
24.      end if
25.    else:
26.      把rejected edits + failure patterns存到buffer
27.    end if
28.  end for (optimization steps)
  
29.  if e >= 2 and slow update enabled:
30.    sample相同tasks，用上个epoch和这个epoch的skill分别跑
31.    O写longitudinal guidance到protected section
32.    通过D_sel验证这个guidance
33.  end if
  
34.  if e >= 2 and meta skill enabled:
35.    O更新meta_skill，用于未来edit generation
36.  end if
37. end for (epochs)

38. 在D_test上评估s_best → 最终report
39. return s_best, test_score
```

---

## Optimizer Prompt的Design

### Failure Analysis Prompt

要求optimizer model：
1. Read ALL trajectories in minibatch
2. Identify most prevalent, systematic failure patterns
3. Classify failure type
4. Propose edits addressing COMMON patterns（不是individual edge cases）
5. Edits必须generalizable——do not hardcode task-specific values
6. Only patch gaps——don't duplicate existing content
7. 最多L个edits

输出strict JSON格式：
```json
{
  "batch_size": <number>,
  "failure_summary": [
    {"failure_type": "<type>", "count": <int>, "description": "<one-line>"}
  ],
  "patch": {
    "reasoning": "<why these edits>",
    "edits": [
      {"op": "append", "content": "<markdown>"},
      {"op": "insert_after", "target": "<heading>", "content": "<markdown>"},
      {"op": "replace", "target": "<old text>", "content": "<new text>"},
      {"op": "delete", "target": "<text to remove>"}
    ]
  }
}
```

**Protected section rule**：`<!-- SLOW_UPDATE_START -->`和`<!-- SLOW_UPDATE_END -->`之间的内容step-level edits绝对不能碰。

### Ranking Criteria

按优先级：
1. **Systematic impact**：fix 50% failures的rule > fix single edge case的rule
2. **Complementarity**：fill gaps > duplicate existing content
3. **Generality**：general principles > specific question types
4. **Actionability**：clear concrete guidance > vague advice

### Slow Update Prompt

这个prompt的角色和per-step analyst完全不同。Per-step analyst看individual trajectories，slow update看**skill如何across entire epoch演化**。

它收到：
1. Previous epoch的skill和current epoch的skill
2. 在相同sampled tasks上两个skill的longitudinal comparison
3. Previous slow update guidance（如果有）

它要写strategic guidance block，OVERWRITE protected section。要求：
- Written as direct, actionable instructions to training model
- Prioritize：preventing regressions → fixing persistent failures → reinforcing successful patterns
- NOT duplicate main skill body
- Address training model directly："When you encounter X, always do Y."

### Meta Skill Prompt

Address **FUTURE OPTIMIZER**直接，不是training model。Captures：
- Which kinds of edits tend to help in this environment
- Which kinds of edits tend to be too vague / redundant / brittle / harmful
- What level of abstraction works best
- What failure-repair patterns should be prioritized
- What regression risks to guard against

这个meta skill **不ship给target model**——只存在于optimizer side。

---

## Limitations

1. **依赖scored trajectories**：对open-ended domains where success is subjective，validation gate需更强evaluation
2. **Training cost**：需rollout computation + optimizer model calls；amortized当skill被reused时
3. **Single skill per domain**：highly heterogeneous domains可能需要多个disjoint procedures
4. **Domain-specific heuristics风险**：careful held-out evaluation必要

---

## Outlook

- **Skill libraries** sharing infrastructure across domains
- **Reuse of optimizer-side meta skills** across benchmarks
- **Reward-free或preference-driven validation gates** for open-ended tasks
- **Self-distillation of optimized skills back into target model**作为weight-level adaptation的stepping stone

---

## 我的延伸思考

### SkillOpt本质上是什么

SkillOpt是**text-space architecture search**。Skill document是一种"architecture"，optimizer model在text space搜索optimal architecture。这和DARTS、ENAS等NAS方法的motivation相似——区别是SkillOpt在natural language space搜索。

### 和Distillation的区别

Table 5证明SkillOpt不是简单distillation。Target-matched optimizer（teacher和student一样强）仍recover 56-74% gain。**Optimization loop本身**是关键，optimizer只是loop的engine。

### 和Chain-of-Thought的关系

Learned skills某种程度上是**externalized, persistent chain-of-thought**。CoT在inference时临时展开reasoning，skill则将procedural reasoning固化到text artifact中。SkillOpt可视为**automated CoT distillation**——从rollouts中提取procedural lessons并固化。

### Adaptation的Hierarchy

SkillOpt建立了adaptation的hierarchy：

| Adaptation Level | Cost | Generality | Controllability |
|-----------------|------|------------|-----------------|
| Weight space (fine-tuning) | Highest | Highest | Hardest |
| Skill space (SkillOpt) | Moderate | Moderate | Good |
| Prompt space (TextGrad/GEPA) | Lowest | Lowest | Hardest to validate |

Skill space是sweet spot——compact enough to be auditable，rich enough to encode procedural knowledge，validation gate使其可靠。

### 对未来Frontier Model部署的影响

这种text-space optimization范式可能成为frontier model部署后的主流adaptation方式。比weight fine-tuning更轻量、比prompt engineering更可靠、比RLHF更易于控制。

特别是**cross-harness transfer的成功**意味着：你可以在一个cheap environment里训练skill，然后deploy到expensive production environment。这在实际工程中价值巨大。

参考：
- [TextGrad: Automatic Differentiation via Text](https://arxiv.org/abs/2406.07496)
- [GEPA: Reflective Prompt Evolution](https://arxiv.org/abs/2507.19457)
- [Voyager: Open-Ended Embodied Agent](https://arxiv.org/abs/2305.16291)
- [Reflexion: Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Self-Refine: Iterative Refinement](https://arxiv.org/abs/2303.17651)
- [DSPy: Compiling Declarative LM Calls](https://arxiv.org/abs/2310.03714)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [SWE-Agent](https://arxiv.org/abs/2405.15793)
- [SkillsBench](https://arxiv.org/abs/2602.12670)
- [SoK: Agentic Skills](https://arxiv.org/abs/2602.20867)
- [Trace2Skill](https://arxiv.org/abs/2603.25158)
- [EvoSkill](https://arxiv.org/abs/2603.02766)
- [SkillFoundry](https://arxiv.org/abs/2604.03964)
- [AutoSkill](https://arxiv.org/abs/2603.01145)
- [Omni-MATH](https://arxiv.org/abs/2410.07985)
- [OlympiadBench](https://arxiv.org/abs/2402.14008)
- [SpreadsheetBench](https://arxiv.org/abs/2412.10656)
- [ALFWorld](https://openreview.net/forum?id=0IOX0YcCdTn)
- [OPRO: LLMs as Optimizers](https://arxiv.org/abs/2310.03714)
- [Codex](https://openai.com/index/introducing-codex/)
- [Claude Code](https://www.anthropic.com/claude-code)

---

# SkillOpt: 将Agent Skills作为可训练外部状态

## 1. 核心问题与动机

当前agent skills的三种主流构建方式各有致命缺陷：

- **Hand-crafted skills**：依赖专家手工编写，脆弱且缺乏反馈机制
- **One-shot LLM generation**：一次性生成，无法根据rollout反馈迭代修正
- **Loosely controlled self-revision**：如Trace2Skill、EvoSkill等系统允许无约束的skill改写，容易erase useful rules、引入incompatible instructions、overfit到局部failure

这些方法都缺乏深度学习中weight-space optimization所具备的**discipline**：可复现性、step size控制、validation gating、momentum等。SkillOpt的核心insight是：**skill document本身应该被视为一个可训练的external state**，用与weight optimization同等的严谨度来优化它。

这种视角的深层动机来自frontier models的部署现实——weight adaptation对closed frontier models不可用，对open models成本高昂。而agent skills作为procedural adaptation的interface，正好提供了一个可审计、可移植、可复用的adaptation layer。

参考：
- [TextGrad: Automatic Differentiation via Text](https://arxiv.org/abs/2406.07496)
- [GEPA: Reflective Prompt Evolution](https://arxiv.org/abs/2507.19457)
- [Voyager: Open-Ended Embodied Agent](https://arxiv.org/abs/2305.16291)

## 2. 核心方法形式化

### 2.1 问题设定

Skill $s$ 是一个natural-language policy，在execution前注入到agent context中。在direct-chat benchmarks中prepend到system/developer instruction；在tool-use harnesses中作为persistent procedural memory。

核心公式（式1）定义了forward pass：

$$
(\tau(s), r(s)) = h(M, x, s), \quad r(s) \in [0,1]
$$

变量解释：
- $h$：execution harness（如direct chat、Codex、Claude Code）
- $M$：frozen target model（被适配的模型，weights不动）
- $x$：单个task实例
- $s$：当前skill document
- $\tau(s)$：执行产生的trajectory（包括messages、tool calls、observations等）
- $r(s)$：scalar score，归一化到$[0,1]$区间

数据split为三部分：$D_{\mathrm{tr}}$（train）、$D_{\mathrm{sel}}$（selection）、$D_{\mathrm{test}}$（test）。这个划分非常关键——**selection split扮演validation set的角色，用于gating updates**，与深度学习中的held-out validation完全对应。

训练目标（式2、3）：

$$
s_{\mathrm{sel}}^{\star} = \arg\max_{s \in \mathcal{C}(D_{\mathrm{tr}})} \frac{1}{|D_{\mathrm{sel}}|} \sum_{x \in D_{\mathrm{sel}}} r(s)
$$

$$
\mathrm{Test}(s_{\mathrm{sel}}^{\star}) = \frac{1}{|D_{\mathrm{test}}|} \sum_{x \in D_{\mathrm{test}}} r(s_{\mathrm{sel}}^{\star})
$$

其中$\mathcal{C}(D_{\mathrm{tr}})$表示在训练split上生成的候选skill集合。$s_{\mathrm{sel}}^{\star}$是通过selection split筛选出的最佳skill，最终在test split上报告性能。

这种三段式split确保了报告的数字测量的是**generalization而非validation-set fit**——与深度学习中train/val/test split的严格原则完全一致。

### 2.2 Optimizer状态

Optimizer state包含：
- 当前skill $s_{\mathrm{cur}}$
- 最佳validation-gated skill $s_{\mathrm{best}}$
- 缓存的skill hashes $\mathcal{C}$（用于避免重复evaluation）
- Epoch-local rejected-step buffer $\mathcal{B}$
- Optional slow/meta-update state $m_{\mathrm{meta}}$

只有最佳accepted skill被export为`best_skill.md`。

## 3. Pipeline深度解析（Figure 2对应）

### 3.1 Forward Pass: Rollout Evidence

每个optimization step，target model用当前skill在$D_{\mathrm{tr}}$上执行一个rollout batch。Harness记录：
- Task metadata
- Messages
- Tool calls
- Observations
- Command outputs
- Final answers
- Verifier feedback
- Benchmark-specific context（如spreadsheet previews、document references、compact execution traces）

**Batch size的作用**：小batch更新快但噪声大；大batch能暴露更多recurring patterns后再改变skill。这种trade-off与SGD中mini-batch size的作用完全对应。

支持**accumulation**：多个rollout batch独立reflect后合并为一个update——这与gradient accumulation在深度学习中的作用完全一致，decouple了execution throughput和update frequency。

### 3.2 Backward Pass: Minibatch Reflection

这是与反向传播对应的机制。Optimizer model将trajectories转化为skill edits：

1. **分离failures与successes**，分别partition成reflection minibatches
2. 单个trajectory往往只产生anecdotal fixes，minibatch能暴露**reusable procedural errors**（如agent consistently searches wrong source、writes wrong format、fails to verify tool results）
3. **Failure minibatches** propose missing/corrective rules
4. **Success minibatches** preserve behaviors that already work
5. 每次reflection返回structured add/delete/replace edits（或rewrite mode下的rewrite suggestions）

**Hierarchical merging**：先独立consolidate failure-driven和success-driven edits，再以failure correction优先级进行最终合并。此步骤过滤duplicate、contradictory、example-specific suggestions。

这种两阶段merge的动机类似ensemble averaging——独立的分析师worker并行处理不同minibatch，避免单点偏差。

### 3.3 Bounded Text Updates（学习率类比）

Textual learning rate = edit budget $L_t$，即step $t$最多应用的skill edits数量。

流程：
1. Aggregation后，optimizer model按expected utility对merged edit pool排序
2. Clip到top-$L_t$ edits
3. 生成candidate skill

**关键区别**：ad hoc prompt rewriting允许unbounded rewrites，会erase useful rules、引入incompatible instructions、overfit到local failure。Bounded updates保留continuity，同时允许skill acquire新procedures。

支持的schedule：
- **Constant**：$L_t = L_0$
- **Linear**：$L_t = L_0 - \alpha \cdot t$
- **Cosine**：$L_t = L_{\min} + \frac{1}{2}(L_0 - L_{\min})(1 + \cos(\pi t / T))$
- **Autonomous**：根据reflection自动决定

默认cosine schedule，开头大edit后期收敛——与深度学习cosine learning rate decay的intuition完全一致。

**两种edit mode**：
- **Patch mode**：localized operations（append、insert_after、replace、delete）
- **Rewrite mode**：suggestions condition一个full skill rewrite

Step-level edits**不能overwrite protected slow-update field**——这种separation类似ResNet中skip connection保护low-level features的思路。

### 3.4 Validation Gate和Rejected-Edit Buffer

每个candidate skill在$D_{\mathrm{sel}}$上用同一frozen target model和harness评估：
- 若**严格大于**current selection score → 成为new current skill
- 若也**严格大于**best score so far → 成为`best_skill.md`
- 否则rejected

**严格大于**的设计：ties被rejected，防止deployed skill silently drift。这是保守但重要的设计——与深度学习中early stopping的保守性对应。

**Rejected edits仍有价值**：epoch-local buffer记录：
- Observed failure patterns
- 对rejected steps：tried edits + caused score drop

后续reflection calls接收此buffer，optimizer model可：
- 避免repeating failed edits
- 聚焦unresolved failures

这给loop提供**negative feedback**，且**zero inference-time cost**——因为只在training时使用。

### 3.5 Epoch-Wise Slow/Meta Update

Fast updates从current batch学习；epoch-wise slow/meta update从adjacent epochs学习。

机制：
1. Epoch结束时，sample相同的training items
2. 分别在previous epoch的skill和current skill下执行
3. 分组为：improvements、regressions、persistent failures、stable successes
4. Optimizer model写出concise longitudinal guidance block到protected slow-update field
5. 此candidate仍需通过validation gate

**Meta skill（optimizer-side only）**：
- 总结哪些edit patterns helped、哪些rejected、哪些failures persisted across epochs
- Prepended到future optimizer prompts for reflection、merging、ranking
- **不随target model shipped**

这种separation of concerns的设计很精妙：deployed skill保持compact和portable，training则受益于editing process的richer record。

直觉上，这与深度学习中的**momentum**对应——slow update将stable editing directions跨epoch保留下来，避免每epoch重新学习。

## 4. Algorithm 1完整解析

Algorithm 1给出了完整的optimization procedure：

```
Input: Frozen M, optimizer O, harness h, splits D_train/D_sel/D_test
       initial skill s_0, epochs E, edit-budget schedule L_t
       rollout batch size B, accumulation factor A, reflection minibatch B_m
Output: s_best, test score

1: s_cur ← s_0, s_best ← s_0, C ← ∅, B ← [], m_meta ← ∅
2: score_cur ← EVALUATE(M, h, s_0, D_sel); score_best ← score_cur
3: C[Hash(s_0)] ← score_cur
4: for e = 1 to E do
5:   Shuffle D_train into rollout batches; reset B ← []
6:   for each optimization step in epoch e do
7:     Collect A rollout batches via h(M, x, s_cur)
8:     Split into failures/successes, then into minibatches of size B_m
9:     Ask O to analyze failure minibatches → failure patch proposals
10:    Ask O to analyze success minibatches → success patch proposals
11:    Ask O to merge failure proposals, merge success proposals, 
        then final failure-prioritized merge
12:    Ask O to rank merged edits, keep at most L_t edits
13:    Apply selected edits → candidate skill s̃
14:    if Hash(s̃) ∈ C then
15:      score_cand ← C[Hash(s̃)]    # 缓存命中，避免重复评估
16:    else
17:      score_cand ← EVALUATE(M, h, s̃, D_sel)
18:      C[Hash(s̃)] ← score_cand
19:    end if
20:    if score_cand > score_cur then
21:      s_cur ← s̃; score_cur ← score_cand
22:      if score_cand > score_best then
23:        s_best ← s̃; score_best ← score_cand
24:      end if
25:    else
26:      Add rejected edits + observed failure patterns to B
27:    end if
28:  end for
29:  if e ≥ 2 and slow update enabled then
30:    Compare same sampled tasks under previous/current epoch-end skills
31:    Ask O for protected longitudinal guidance; validate via D_sel
32:  end if
33:  if e ≥ 2 and optimizer memory enabled then
34:    Ask O to update m_meta for future edit generation/selection
35:  end if
36: end for
37: score_test ← EVALUATE(M, h, s_best, D_test)
38: return s_best, score_test
```

注意几个关键设计：
- **Line 14-15**：Hash缓存避免对相同candidate skill重复评估——这是计算优化的细节
- **Line 20**：**严格大于**，ties被rejected
- **Line 29-31**：Slow update从第二个epoch开始，因为需要previous epoch作为baseline
- **Line 33-35**：Meta skill也是从第二个epoch开始更新

## 5. Harness-Agnostic Deployment

通过lightweight adapter interface实现harness-agnostic：
- Adapter constructs train/evaluation batches
- Injects当前skill到agent context
- Runs native harness
- Returns scored trajectories

同一个optimizer因此适用于：
- Direct QA
- Spreadsheet execution
- Document reasoning
- Multimodal QA
- Embodied environments
- Codex-style execution loops
- Claude Code-style execution loops

**实际优势**：更强的optimizer model可offline训练reusable skill artifact，结果`best_skill.md`可跨target models、harnesses、nearby benchmarks部署，**不改变model weights**。

## 6. 实验设计深度解读

### 6.1 Benchmark Suite

6个benchmark覆盖多样任务：
- **SearchQA**：extractive QA with noisy retrieval
- **SpreadsheetBench**：multi-round codegen with openpyxl/pandas runtime（up to 30 turns）
- **OfficeQA**：multi-turn tool loops（up to 24 tool calls）
- **DocVQA**：multimodal-document reasoning
- **LiveMathematicianBench**：mathematical MCQ reasoning
- **ALFWorld**：persistent embodied interaction（up to 50 steps/episode）

### 6.2 Target Models

7个target models：
- GPT-5.5（frontier-scale）
- GPT-5.4
- GPT-5.4-mini
- GPT-5.4-nano
- GPT-5.2
- Qwen3.5-4B（small-scale）
- Qwen3.6-35B-A3B

### 6.3 Execution Harnesses

3种execution mode：
- **Direct chat**：single chat completion call with skill prepended to system prompt
- **Codex harness**：drives target through codex CLI in workspace-write sandbox；SkillOpt renders当前skill到per-task `SKILL.md`，读回`codex_trace_summary.txt`作为teacher reflection context
- **Claude Code harness**：通过claude CLI mirror相同workspace contract

参考：
- [SWE-Agent](https://arxiv.org/abs/2405.15793)
- [Codex介绍](https://openai.com/index/introducing-codex/)
- [Claude Code](https://www.anthropic.com/claude-code)

### 6.4 Baselines

7个baseline涵盖no-adaptation、hand-written、one-shot、learning families：
- **No skill**：frozen target with default system prompt
- **Human skill**：expert-written skill curated per benchmark（145–516 tokens）
- **One-shot LLM skill**：single skill generated by GPT-5.5，never updated
- **Trace2Skill**：trajectory-level skill distillation
- **TextGrad**：gradient-style natural-language prompt optimization
- **GEPA**：Pareto reflective prompt evolution
- **EvoSkill**：skill-folder evolution under failure analysis（harness-side competitor）

## 7. 主结果Table 1深度分析

### 7.1 Headline Numbers

- **52/52 cells best or tied-best**——这是压倒性结果
- **GPT-5.5 average improvement**：
  - Direct chat: +23.5 points（58.8 → 82.3）
  - Codex: +24.8 points
  - Claude Code: +19.1 points
- **Oracle-baseline gap**：+5.4 points（即SkillOpt平均超过"每cell最强baseline"5.4分）

### 7.2 Per-Benchmark Highlights (GPT-5.5 direct chat)

| Benchmark | No skill | SkillOpt | Δ |
|-----------|----------|----------|---|
| SearchQA | 77.7 | 87.3 | +9.6 |
| SpreadsheetBench | 41.8 | 80.7 | +38.9 |
| OfficeQA | 33.1 | 72.1 | +39.0 |
| DocVQA | 78.8 | 91.2 | +12.4 |
| LiveMath | 37.6 | 66.9 | +29.3 |
| ALFWorld | 83.6 | 95.5 | +11.9 |

**关键观察**：
1. Procedural benchmarks（SpreadsheetBench、OfficeQA）gain最大（+38.9、+39.0）——这些有strict procedural和answer-format requirements
2. Near-ceiling benchmarks（SearchQA）gain相对小但仍positive（+9.6）
3. Math reasoning（LiveMath）+29.3是显著提升

### 7.3 跨模型Scale

| Model | Average improvement |
|-------|---------------------|
| GPT-5.5 | +23.5 |
| GPT-5.4 | +12.7 |
| GPT-5.4-mini | +15.4 |
| GPT-5.4-nano | +26.7 |
| GPT-5.2 | +16.6 |
| Qwen3.5-4B | +19.2 |
| Qwen3.6-35B-A3B | +9.1 |

**重要洞察**：Small and weak target models benefit the most in relative terms。GPT-5.4-nano在DocVQA上near doubles，在ALFWorld上triples（34.3 → 69.4）。这与"compact skill artifact supplies procedural knowledge small models don't hold in weights"的假说一致。

### 7.4 工具执行Harness

| Harness | Average Δ over no skill | Δ over next-best (EvoSkill) |
|---------|--------------------------|----------------------------|
| Codex | +24.8 | +14.0 |
| Claude Code | +19.1 | +3.2 |

EvoSkill在Claude Code上已经很强（57.8 → 73.7），但SkillOpt仍能在此基础上+3.2。

### 7.5 反驳Alternative Explanations

论文特别反驳了三种alternative explanations：

1. **"只是prompt length"**：Human skills已是145-516 tokens，learned artifacts更compact仍胜过——所以不是prompt length在起作用
2. **"只是optimizer capacity"**：GPT-5.4-nano也胜过所有baseline；Table 5显示target-matched optimizer也能recover大部分gain
3. **"只是skill format"**：EvoSkill已improve Codex SpreadsheetBench 27.5→67.5，SkillOpt再加+17.5（67.5→85.0）

## 8. Hyperparameter Analysis (Table 2)

### 8.1 Training Set Size (Panel a)

| Setting | SearchQA | Spreadsheet | LiveMath |
|---------|----------|-------------|----------|
| 1 example | 81.0 | 47.5 | 59.1 |
| 20% train | 84.1 | 69.0 | 65.9 |
| 40% train | 86.1 | 73.5 | 64.8 |
| 80% train | 86.1 | 77.6 | 67.0 |
| 100% train | 84.1 | 78.0 | 70.5 |

**洞察**：Procedural benchmarks reward更多training evidence——SpreadsheetBench从47.5涨到78.0，LiveMath从59.1涨到70.5。SearchQA在20%后saturate，因为ceiling效应。

### 8.2 Reflection Minibatch Size B_m (Panel b)

| B_m | SearchQA | Spreadsheet | LiveMath |
|-----|----------|-------------|----------|
| 1 | 85.9 | 75.4 | 60.5 |
| 4 | 86.3 | 77.1 | 54.8 |
| 8 (default) | 86.9 | 75.4 | 64.5 |
| 16 | 87.1 | 77.5 | 61.3 |
| 32 | 87.0 | 77.9 | 61.3 |

**洞察**：Robustness in 1-32 range。Default B_m=8基本处于或接近top。

### 8.3 Rollout Batch Size B (Panel c)

| B | SearchQA | Spreadsheet | LiveMath |
|---|----------|-------------|----------|
| 8 | 85.1 | 76.8 | 58.1 |
| 16 | 86.4 | 77.1 | 62.9 |
| 40 (default) | 87.1 | 77.5 | 61.3 |
| 56 | 86.5 | 76.8 | 56.5 |
| Full epoch | 87.2 | 75.0 | 53.2 |

**洞察**：Full epoch反而下降（SpreadsheetBench 75.0, LiveMath 53.2）——说明stochasticity actually helps，过度aggregate反而overfit到当前skill。

### 8.4 Textual Learning Rate L_t (Panel d)

| L_t | SearchQA | Spreadsheet | LiveMath |
|-----|----------|-------------|----------|
| 1 | 85.5 | 77.5 | 62.1 |
| 2 | 86.7 | 77.5 | 60.5 |
| 4 (default) | 86.5 | 78.2 | 56.5 |
| 8 | 87.0 | 73.6 | 66.9 |
| 16 | 86.8 | 78.2 | 65.3 |

**洞察**：Small/moderate edit budgets都competitive。L_t=8在LiveMath上反而更好（66.9）——说明不同benchmark的最优step size不同。

### 8.5 Scheduler (Panel e)

| Schedule | SearchQA | Spreadsheet | LiveMath |
|----------|----------|-------------|----------|
| Constant | 87.3 | 80.7 | 62.1 |
| Cosine (default) | 87.1 | 77.5 | 61.3 |
| Linear | 87.2 | 72.9 | 62.9 |

**洞察**：Constant schedule在SpreadsheetBench上最好（80.7），但linear较差（72.9）。Bounded-update story不依赖特定scheduler。

### 8.6 Slow-update Samples (Panel f)

| Samples | SearchQA | Spreadsheet | LiveMath |
|---------|----------|-------------|----------|
| 5 | 86.8 | 76.4 | 64.5 |
| 10 | 86.4 | 74.3 | 65.3 |
| 20 (default) | 87.1 | 77.5 | 61.3 |
| 40 | 86.9 | 75.4 | 54.8 |

**洞察**：20 examples是sweet spot。40反而下降（Spreadsheet 75.4, LiveMath 54.8）——可能因为太多samples引入noise。

## 9. Component Ablations (Table 3)

| Component | Setting | SearchQA | Spreadsheet | LiveMath |
|-----------|---------|----------|-------------|----------|
| Learning-rate form | lr=4 (default) | 87.1 | 77.5 | 61.3 |
| | dynamic lr | 85.8 | 71.8 | 54.0 |
| | without lr | 84.6 | 75.7 | 57.3 |
| Rejected buffer | with buffer (default) | 87.1 | 77.5 | 61.3 |
| | without buffer | 85.5 | 72.9 | 58.9 |
| Slow/meta update | meta + slow (default) | 87.1 | 77.5 | 61.3 |
| | without meta | 85.1 | 75.7 | 58.1 |
| | without both | 86.3 | **55.0** | 59.7 |

**关键发现**：
1. **Without lr**（即unbounded rewrite）：Spreadsheet降到75.7——证明bounded update的必要性
2. **Without rejected buffer**：Spreadsheet降4.6，证明negative feedback的价值
3. **Without both meta & slow update**：Spreadsheet从77.5暴跌到55.0（−22.5 points）——这是ablation suite中最大的degradation，证明epoch-wise slow/meta update的核心作用

## 10. Transfer Experiments (Table 4)

这是paper最有应用价值的部分。

### 10.1 Cross-Model Transfer

| Source | Target | Benchmark | Baseline | Direct | Transferred |
|--------|--------|-----------|----------|--------|-------------|
| GPT-5.4 | GPT-5.4-mini | Spreadsheet | 36.1 | 47.5 | 45.5 (+9.4) |
| GPT-5.4 | GPT-5.4-nano | Spreadsheet | 23.5 | 42.5 | 26.5 (+3.0) |
| GPT-5.4 | GPT-5.4-mini | LiveMath | 14.7 | 32.8 | 19.2 (+4.5) |
| GPT-5.4 | GPT-5.4-nano | LiveMath | 23.2 | 27.2 | 28.8 (+5.6) |

**洞察**：所有cross-model transfers为positive。值得注意的是LiveMath GPT-5.4→GPT-5.4-nano transfer（28.8）甚至超过in-domain（27.2），suggesting某些learned procedures是target-model agnostic。Spreadsheet GPT-5.4-mini保留82%的in-domain gain（+9.4 of +11.4）。

### 10.2 Cross-Harness Transfer

| Source | Target | Benchmark | Baseline | Direct | Transferred |
|--------|--------|-----------|----------|--------|-------------|
| Codex | Claude Code | Spreadsheet | 22.1 | 80.4 | 81.8 (+59.7) |
| Claude Code | Codex | Spreadsheet | 27.5 | 85.0 | 71.1 (+43.6) |
| Codex | Claude Code | LiveMath | 40.8 | 56.5 | 42.4 (+1.6) |
| Claude Code | Codex | LiveMath | 35.2 | 78.4 | 48.0 (+12.8) |

**重要发现**：Codex→Claude Code transfer在SpreadsheetBench上甚至slightly exceeds in-domain Claude Code SkillOpt reference（81.8 vs 80.4）。这suggests learned rules encode **harness-agnostic procedures** like structure-first inspection、formula-aware verification、static-value materialization，不是harness-specific command recipes。

### 10.3 Cross-Benchmark Transfer

| Source | Target | Model | Baseline | Direct | Transferred |
|--------|--------|-------|----------|--------|-------------|
| OlympiadBench | Omni-MATH | GPT-5.4 | 56.6 | - | 60.3 (+3.7) |
| OlympiadBench | Omni-MATH | GPT-5.4-mini | 34.8 | - | 36.6 (+1.8) |
| OlympiadBench | Omni-MATH | GPT-5.4-nano | 38.8 | - | 40.1 (+1.3) |

**洞察**：Cross-benchmark transfer是strictest shift，所有transfers仍positive。说明optimized skill encodes reusable mathematical procedure，而非memorized benchmark-specific formatting。

## 11. Optimizer Strength (Table 5)

| Benchmark | Target | Baseline | Strong (GPT-5.5) | Target-matched |
|-----------|--------|----------|------------------|----------------|
| Spreadsheet | GPT-5.4-mini | 36.1 | 47.5 (+11.4) | 43.2 (+7.1) |
| Spreadsheet | GPT-5.4-nano | 23.5 | 42.5 (+19.0) | 35.4 (+11.9) |
| SearchQA | GPT-5.4-mini | 75.9 | 80.2 (+4.3) | 78.3 (+2.4) |
| SearchQA | GPT-5.4-nano | 55.8 | 74.8 (+19.0) | 69.9 (+14.1) |

**关键insight**：Stronger optimizer produces larger gains on every cell。但target-matched optimizer仍recover 56-74% of strong-optimizer gain。**这证明SkillOpt不是简单的distillation pipeline from stronger teacher to weaker student**——optimization loop itself contributes substantial value on top of optimizer capacity。

## 12. Learned Skill分析 (Table 6 & Figure 4)

### 12.1 Compactness与Edit Economy

| Benchmark | Initial (tok) | Final (tok) | Edits | Train tokens | Cost/pt |
|-----------|---------------|-------------|-------|--------------|---------|
| SearchQA | 16 | 857 | 4 | 213.8M | 37.9M |
| SpreadsheetBench | 224 | 1,995 | 4 | 21.4M | 0.6M |
| OfficeQA | 145 | 883 | 1 | 20.8M | 1.1M |
| DocVQA | 81 | 959 | 3 | 188.2M | 46.4M |
| LiveMath | 154 | 379 | 1 | 23.2M | 3.6M |
| ALFWorld | 516 | 1,321 | 2 | 59.3M | 15.9M |

**惊人发现**：
1. **Final skills都 < 2,000 tokens**（最大1,995 tok for SpreadsheetBench）
2. **仅1-4个accepted edits**就产生巨大gain——OfficeQA的+39.0 points来自single accepted edit，LiveMath的+29.3来自single edit
3. 这是validation gate在做real work的直接证据——大量proposed edits被rejected

### 12.2 Learned Rules (Figure 4)

| Benchmark | Representative Rule |
|-----------|---------------------|
| SearchQA | "Infer the expected answer type from clue wording, then choose the shortest canonical entity supported by co-occurring distinctive evidence." |
| SpreadsheetBench | "Inspect workbook structure and formulas, then write evaluated static values across the full requested target range instead of relying on Excel recalculation." |
| OfficeQA | "Treat oracle parsed pages as primary evidence, lock table/date/unit context, and output exactly the requested rounded value without extra labels." |
| DocVQA | "For tables, forms, charts, and legends, first bind the question to the exact visual row/header/field, then copy only the aligned answer span." |
| LiveMath | "In strongest-statement MCQs, rank choices by theorem strength and prefer a justified stronger-result option over true but weaker corollaries." |
| ALFWorld | "Keep a horizon-aware visited/frontier ledger, diversify search after repeated same-type failures, and avoid revisiting the destination until holding the target." |

**三个关键观察**：
1. **Procedural而非instance-specific**：没有rule提到specific question、file、entity
2. **Encode frontier models缺乏的discipline**：answer-format constraints、evidence binding、workbook-structure-first reasoning、search-frontier discipline、canonical-entity choice
3. **可读性极强**：这些rule读起来像一个thoughtful human practitioner写一天benchmark后会写的——但它们是optimizer自动产生且edit-by-edit validated的

## 13. Qualitative Skill Evolution

### 13.1 ALFWorld案例

**Initial skill**：generic household plan——search target object, pick up, transform if needed, place at destination

**Optimized skill**：
- Exact object-name matching（防止mugs/cups/pans/pots互相替代）
- Visited-location memory（prefer未visited receptacles over repeatedly检查likely但exhausted locations）
- Destination memory
- Pick-two progress locks
- Direct completion rules（once agent can clean/heat/cool/place/complete next subgoal, take that admissible action instead of examining/closing/verifying）

**质性转变**：从general search-transform-place strategy → finite-state execution policy with object identity、search memory、progress locks、loop breakers。在这个run中，性能从49.3提升到74.6。

### 13.2 SpreadsheetBench案例

**Initial skill**：use Python spreadsheet libraries + preserve unrelated workbook content

**Optimized skill**：
- Inspect actual workbook而非rely on previews
- Locate headers和target ranges across multiple sheets
- Normalize keys和cell types before lookup/aggregation
- Preserve formatting during structural edits
- **Key rule for formula-style prompts**：when grader reads cell values, agent应compute和write evaluated static values，即使prompt mentions formulas如INDEX/MATCH或XLOOKUP
- Fill complete target ranges包括currently blank result cells
- Keep helper computations in Python而非adding workbook artifacts
- Reopen saved workbook检查boundary rows和remaining blanks

**质性转变**：从generic automation workflow → workbook-forensics policy。性能从40.4提升到78.9。

## 14. 与相关工作的深度对比

### 14.1 Prompt Auto-tuning家族

- **GEPA**：trajectory feedback → reflective prompt evolution，outperforms RL on several tasks。但mainly targets prompts
- **ABSTRAL**：multi-agent design documents
- **EvoTest**：test-time agentic system evolution

SkillOpt的不同：optimizes **persistent skill document** that can be trained、validated、exported、reused with adapted model，applying language-level controllability to a stable procedural skill state。

### 14.2 Skill Construction和Evolution

- **SkillsBench / SoK**：frame skills as reusable procedural knowledge
- **Prior systems**：from lifelong experience、trajectory lessons、skill knowledge bases、heterogeneous domain resources
- **Refinement via**：failure analysis、creation-evaluation-revision loops、co-evolving generators和verifiers、collective updates、RL

SkillOpt studies narrower problem：how to train one compact domain skill with deep-learning-style controls。Yields controlled和auditable procedure for producing portable `best_skill.md` without changing model weights。

参考：
- [Reflexion: Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Self-Refine: Iterative Refinement](https://arxiv.org/abs/2303.17651)
- [DSPy: Compiling Declarative LM Calls](https://arxiv.org/abs/2310.03714)

## 15. Optimizer Prompt Contracts（Appendix C）

### 15.1 Failure Analysis (analyst_error.md)

要求：
- Read ALL trajectories in minibatch
- Identify most prevalent, systematic failure patterns
- Classify failure type
- Propose skill edits addressing COMMON patterns
- Edits must generalize beyond specific tasks
- Only patch gaps, don't duplicate existing content
- Produce AT MOST L edits

输出JSON格式包含：
- `batch_size`：analysed trajectories数量
- `failure_summary`：failure_type + count + description
- `patch`：reasoning + edits（each edit has op + target + content）

**Protected section**：`<!-- SLOW_UPDATE_START -->`和`<!-- SLOW_UPDATE_END -->`之间内容step-level edits不能修改。

### 15.2 Success Analysis (analyst_success.md)

类似但focus on generalizable behavior patterns COMMON across batch。`edits` may be empty if skill already covers all patterns。

### 15.3 Hierarchical Merging

- **merge_failure.md**：merge failure patches into ONE coherent, non-redundant patch
- **merge_success.md**：merge success patches
- **merge_final.md**：FINAL merge with **FAILURE PATCHES TAKE PRIORITY**原则：
  - Failure-driven edits should be preserved unless directly conflict with well-supported success pattern
  - Deduplicate: if failure和success edit cover same point, keep failure version
  - Include success edits covering patterns NOT addressed by failure edits
  - Higher-level merges represent broader consensus

### 15.4 Ranking (ranking.md)

排序criteria（按优先级）：
1. **Systematic impact**：edits addressing widespread, recurring failure patterns rank highest
2. **Complementarity**：edits filling gaps in current skill
3. **Generality**：general principles > specific question types/entities
4. **Actionability**：clear, concrete guidance > vague advice

### 15.5 Slow Update (slow_update.md)

这是per-step analyst的补充——per-step sees individual trajectories, slow update sees **how skill evolved across entire epoch**。Process：
1. Reflect on previous guidance（哪些effective、哪些backfired、哪些blind spots）
2. Write updated guidance:
   - Retain和strengthen effective parts
   - Revise或remove ineffective/counterproductive parts
   - Add new instructions for newly observed regressions和persistent failures

输出written as **direct, actionable instructions to training model**，prioritizing：preventing regressions → fixing persistent failures → reinforcing successful patterns。

### 15.6 Optimizer Memory (meta_skill.md)

Address **FUTURE OPTIMIZER**直接，not training model。Captures：
- Which kinds of edits tend to help in this environment
- Which kinds of edits tend to be too vague, redundant, brittle, or harmful
- What level of abstraction works best for rules here
- What failure-repair patterns should be prioritized
- What regression risks future optimizer calls should guard against

## 16. 深层Intuition构建

### 16.1 为什么Skill Optimization有效

从信息论角度看，frontier models在weights中已经encode了大量procedural knowledge，但zero-shot deployment无法激活task-specific procedures。Skill document作为**external state**：
1. **Attention steering**：注入的skill document直接进入attention计算，无需改变weights就能steer model behavior
2. **Procedural memory**：对small models尤其有效，因为它们weights中缺乏特定domain的procedural knowledge
3. **Format constraints**：很多failure不是capability gap而是format/execution discipline gap——skill能encode这些discipline

### 16.2 为什么Bounded Updates胜过Unbounded Rewriting

这与deep learning中**small learning rate + many steps > large learning rate + few steps**的intuition一致：
- Unbounded rewriting = large learning rate：容易overshoot、erase useful information
- Bounded updates = small learning rate：preserve continuity，每个update是incremental improvement
- Validation gate = early stopping + checkpoint selection

### 16.3 为什么Slow/Meta Update如此重要（ablation中-22.5 points）

Epoch-wise slow update扮演**momentum**角色：
- Fast updates from current batch可能noisy和directional inconsistent
- Slow update aggregates跨epoch的stable directions
- 没有slow update时，每个epoch基本独立学习，无法build up durable procedural lessons

这与深度学习中**BatchNorm或EMA weights**的作用类似——stabilize训练过程。

### 16.4 为什么Validation Gate是Real Work而非Trivial

Edit economy数据（1-4 accepted edits）证明gate在filter大量proposed edits。Plausible textual diagnoses可能hurt实际target model——因为：
1. Optimizer model ≠ target model：optimizer认为好的edit可能不适合target的attention pattern
2. Specificity vs generality trade-off：某些edits fix specific failures但hurt general performance
3. Distribution shift：selection split虽小但真实反映distribution

### 16.5 为什么Cross-Transfer Works

Cross-model/harness/benchmark transfers成功的根本原因是**learned rules是procedural而非instance-specific**。Figure 4的rules都是abstract procedural principles——这些principles的"carrier invariance"远高于specific tool commands或model-specific quirks。

这与"skills are reusable procedural knowledge"的定义完全契合。

## 17. Limitations和Outlook

### 17.1 Limitations

1. **依赖scored trajectories和held-out selection split**：对open-ended domains where success is subjective/multi-dimensional/costly to judge, validation gate需更强human或model-based evaluation
2. **Training cost**：需additional rollout computation和optimizer model calls；amortized当skill被reused时
3. **Single skill per domain**：highly heterogeneous domains需多个disjoint procedures时可能insufficient
4. **Domain-specific heuristics风险**：careful held-out evaluation必要

### 17.2 Outlook

- **Skill libraries** sharing infrastructure across domains
- **Reuse of optimizer-side meta skills** across benchmarks
- **Reward-free或preference-driven validation gates** for open-ended tasks
- **Self-distillation of optimized skills back into target model**作为weight-level adaptation的stepping stone

## 18. 我的延伸思考

### 18.1 与Neural Architecture Search的类比

SkillOpt本质上是**text-space architecture search**：skill document是一种"architecture"，optimizer model在text space搜索optimal architecture。这与DARTS、ENAS等NAS方法的motivation相似——区别是SkillOpt在natural language space而非neural network space搜索。

### 18.2 与Continual Learning的关联

Rejected-edit buffer和slow update mechanism都与continual learning中的**experience replay**和**elastic weight consolidation**思想相通：
- Rejected-edit buffer防止repeating catastrophic edits（类似experience replay防止forgetting）
- Slow update保护durable lessons（类似EWC保护important parameters）

### 18.3 与Distillation的不同

Table 5的target-matched optimizer实验证明SkillOpt不是简单distillation。Stronger optimizer帮助更大，但target-matched也recover 56-74% gain。这说明**optimization loop本身**是关键，optimizer只是loop的engine。

### 18.4 与Chain-of-Thought的关系

Learned skills某种程度上是**externalized, persistent chain-of-thought**。CoT在inference时临时展开reasoning，skill则将procedural reasoning固化到text artifact中。SkillOpt可视为**automated CoT distillation**——从rollouts中提取procedural lessons并固化。

### 18.5 与Weight-Level Adaptation的Hierarchy

SkillOpt建立了adaptation的hierarchy：
- **Weight space**：most expensive, most general, hardest to control
- **Skill space**（SkillOpt）：moderate cost, portable, auditable
- **Prompt space**（TextGrad/GEPA）：cheapest, most brittle, hardest to validate

Skill space的sweet spot：compact enough to be auditable，rich enough to encode procedural knowledge，validation gate使其可靠。

参考相关延伸工作：
- [Large Language Models as Optimizers (OPRO)](https://arxiv.org/abs/2310.03714)
- [SkillsBench](https://arxiv.org/abs/2602.12670)
- [SoK: Agentic Skills](https://arxiv.org/abs/2602.20867)
- [Trace2Skill](https://arxiv.org/abs/2603.25158)
- [EvoSkill](https://arxiv.org/abs/2603.02766)
- [SkillFoundry](https://arxiv.org/abs/2604.03964)
- [AutoSkill](https://arxiv.org/abs/2603.01145)
- [Omni-MATH](https://arxiv.org/abs/2410.07985)
- [OlympiadBench](https://arxiv.org/abs/2402.14008)
- [SpreadsheetBench](https://arxiv.org/abs/2412.10656)
- [ALFWorld](https://openreview.net/forum?id=0IOX0YcCdTn)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)

## 19. 总结

SkillOpt的核心贡献是将**deep learning optimization的discipline**引入**text-space skill editing**。关键设计choices的深层intuition：

1. **Skill = trainable external state**：类比weights为internal trainable state
2. **Bounded edits = learning rate**：控制每step移动距离
3. **Validation gate = early stopping**：只accept严格改善的updates
4. **Rejected-edit buffer = negative feedback**：避免重复失败
5. **Slow/meta update = momentum**：跨epoch保留stable directions
6. **Optimizer ≠ target**：decouple optimization capacity和deployment cost
7. **Harness-agnostic**：通过adapter interface实现deployment portability

52/52 cells的dominance和1-4 edits的edit economy证明这套设计确实将skill editing变成了controlled learning process。最impressive的是cross-harness transfer（Codex→Claude Code甚至超过in-domain）证明learned rules确实encode **harness-agnostic procedural knowledge**。

这种text-space optimization范式可能成为frontier model部署后的主流adaptation方式——比weight fine-tuning更轻量、比prompt engineering更可靠、比RLHF更易于控制。
