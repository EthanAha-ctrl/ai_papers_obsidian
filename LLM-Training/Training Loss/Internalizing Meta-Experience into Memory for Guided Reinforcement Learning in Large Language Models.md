---
source_pdf: Internalizing Meta-Experience into Memory for Guided Reinforcement Learning
  in Large Language Models.pdf
paper_sha256: 0b0faec0eed6cb073043899f44f8e13a6655a6f2643b28fe1df179d02dbedaf4
processed_at: '2026-08-05T10:09:40-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 MEL

## 一句话版本

让 LLM 自己给自己写"错题本"，然后把错题本背进脑子里，下次做题时这些教训就自动起作用。

---

## 1. 这 paper 在解决什么"人"的问题

想象一个学生（LLM）做数学题。Standard RLVR 的做法是：

- 学生做 8 遍同一题
- 老师只批改最后答案对不对（对就 1 分，错就 0 分）
- 学生根据这 8 个分数调整自己的解题策略

问题在哪？学生只知道 "第 3 次做对了，第 7 次做错了"，但**完全不知道第 7 次到底从哪一步开始走偏的，也不知道为什么走偏**。下次见到类似题，照样会掉进同一个坑。

人类学习不是这样的。人类会：

1. 做题 → 对答案（practice + verification）
2. 错了就分析 "我在哪一步开始想错的？为什么想错？"（error attribution）
3. 把教训写成一条 rule，比如 "见到积分题要先检查被积函数的定义域"（experience internalization）
4. 下次做题时这条 rule 自动在脑子里跳出来（reuse）

MEL 就是让 LLM 走完这四个阶段。

参考：
- Human learning cycle 在教育心理学的经典框架：https://en.wikipedia.org/wiki/Experiential_learning
- Meta-cognition in learning：https://arxiv.org/abs/2306.11590

---

## 2. 为什么之前的方法没做到

### PRM (Process Reward Model) 路线

找另一个 LLM 当 "批改老师"，给每一步打分。问题是：
- 这个 "老师" 本身会犯错（reward hacking）
- 跟 RLVR "用规则验证、不依赖 learned proxy" 的初衷违背了
- 训练这个 teacher model 又要标注数据、又要训练，成本高

参考 Let's Verify Step by Step (PRM 开山之作)：https://arxiv.org/abs/2305.20050

### External Memory 路线

把经验存到一个外部数据库，inference 时检索拼接。问题是：
- 经验没进 weights，model 本身没变强
- Context 越拼越长，inference 越来越慢
- 到了新场景、新 distribution，老经验可能不 work

参考 ReasoningBank：https://arxiv.org/abs/2509.25140

### Hint Injection 路线

训练时把高手的解法塞进 prompt 当 prefix。问题是：
- 训练时有 prefix，inference 时没有（distribution mismatch）
- Model 学到的是 "模仿这条 trajectory"，没学到 trajectory 背后的 thinking pattern
- 还是 trajectory-level imitation，没抽象成 knowledge-level rule

参考 Scaf-GRPO：https://arxiv.org/abs/2510.19807

MEL 的核心区别：**经验进 weights，而且进的是抽象 rule 而不是具体 trajectory**。

---

## 3. MEL 怎么做的——四个阶段像 "错题本流程"

### Stage 1: 做题 + 批改（Rollout + Verify）

跟 GRPO 完全一样：一道题采样 8 条 trajectory，用 rule-based verifier 判断每条对错。

**关键 filter**：只有 "8 条里既有对的又有错的" 这道题才进入下一阶段。全对（已经会了）和全错（还不会）的题都没信息量。

### Stage 2: 写错题本（Meta-Experience Construction）

挑出一对 (正确 trajectory, 错误 trajectory)，让 LLM 自己做三件事：

**第一件：定位分叉点**

把两条 trajectory 并排放，让 LLM 找出 "从第几步开始，错误那条走偏了"。这个点叫 **bifurcation point** $s^*$。

为什么 LLM 能做这个？因为 **discrimination 比 generation 容易**。让 LLM 从零解题难，让 LLM 在两条已写好的解法里找差异，简单得多。这跟人能轻松看出别人解题哪步错、但自己解题照样犯错是一个道理。

参考 self-critique asymmetry：https://arxiv.org/abs/2206.05802

**第二件：深度诊断**

在分叉点附近，让 LLM 写一段 critique，要回答：
- 错的 trajectory 这里为什么走偏？（assumption 错了？constraint 漏了？定理用错了？）
- 对的 trajectory 为什么没掉进去？（先检查了什么？用了什么稳健的 representation？中途 self-correct 了？）

**第三件：抽象成 rule**

强制 LLM 把诊断抽象成 "If [抽象条件], then [抽象动作]" 的形式。Prompt 里明确 forbid 提具体数字，比如不允许写 "这道题答案是 $\sqrt{2}$ 所以要平方"，必须写成 "做积分题时先验证被积函数的定义域"。

这一步是把 "这道题的教训" 提升成 "这类题的 rule"。

### Stage 3: 验证错题本有没有用（Empirical Validation via Replay）

LLM 自己写的 rule 不一定真的有用——可能写得头头是道，但下次照样犯错。怎么过滤？

做法：把这条 rule 当 hint 拼到 prompt 里，让 LLM 重新解同一道题。如果这次解对了，保留这条 rule；解错了，丢弃。

这就是 **"用结果说话"**：你说这条经验有用，那你用它把题做对给我看。

这个 filter 非常关键。Paper Figure 9 显示 14B model 的 retention ratio 比 4B 高得多——大 model 写的 rule 质量高，通过率高。这也是 MEL 在大 model 上 gain 更大的原因（scaling law）。

### Stage 4: 把错题本背进脑子（Internalization）

通过验证的 rule 集合 $\mathcal{D}_{\mathcal{M}^*}$ 怎么进 weights？

用 NLL（negative log-likelihood）训练，让 model 在给定 retrospective context（原始题 + 两条对照 trajectory）的情况下，能高概率生成这条 rule。

**这里的 magic**：NLL 的 gradient 数学上等价于 "reward = +1 的 policy gradient"。意思是每条 rule 的每一个 token 都在受到正向强化，这是 dense reward，远比 outcome-level 的 0/1 signal 信息量大。

所以最终的训练 loss 是：

```
Total Loss = GRPO Loss (学解题) + NLL Loss (学写错题本)
```

两个目标同步优化。Model 一边学怎么解题，一边学怎么反思自己的错误，反思的产物又反过来 help 解题。

---

## 4. 为什么这个设计 work

### 4.1 把经验放进了 weights 而不是 context

External memory 方法 inference 时要拼长 context，又慢又贵。MEL 在训练时一次性 "消化" 进 weights，inference 时 model 本身就变强了，零额外开销。

类比：错题本背进脑子 vs 错题本带在身上随时翻。前者随时随地可用，后者要花时间翻找。

### 4.2 Dense reward 解决了 RLVR 早期 cold start

Vanilla GRPO 早期训不动，因为 outcome reward 太 sparse——model 还弱的时候几乎全错，全错 = 全 0 reward = 没有 gradient signal。

MEL 的 NLL reward 是 dense 的，每个 token 都有 +1 reward。即使 outcome 全错，只要能抽出一条 valid meta-experience，gradient 照样更新。这给早期训练一个 "bootstrap" 信号。

Paper Figure 3 显示 MEL 早期 ascent 陡峭得多，就是这个原因。

### 4.3 Knowledge-level reuse 跨任务泛化

一条 trajectory 只对一道题有用。一条 rule 对所有同结构题目有用。

比如抽象出 "处理几何题要先建立坐标系，避免纯代数推导" 这种 rule，下次见到任何几何题都可能 activate。这是 trajectory-level 方法做不到的。

### 4.4 Self-distillation 天然 scale

Teacher = student，所以 student 越强，teacher 越强，生成的 supervision 越干净。这解释了 Figure 9 的 retention ratio 随 scale 上升，以及主实验中 MEL 的 gain 随 scale 上升。

参考 self-distillation 的经典讨论：https://arxiv.org/abs/1503.02531

---

## 5. 实验数据怎么说

### 14B model 上 MEL vs GRPO 的 Pass@1 提升

| Benchmark | GRPO | MEL | 提升 |
|-----------|------|-----|------|
| AIME24 | 30.00 | 33.33 | +3.33 |
| AIME25 | 33.33 | 36.67 | +3.34 |
| AMC23 | 75.00 | 82.50 | +7.50 |
| MATH500 | 85.00 | 90.80 | +5.80 |
| OlympiadBench | 58.16 | 61.87 | +3.71 |

平均 4-5 个百分点，对 RLVR 这种已经 optimize 得很厉害的 baseline 来说相当显著。

### 三个 metric 各自说明什么

- **Pass@1**：greedy decoding 时能不能一次做对。MEL 提升说明经验已经进 weights，不需要外部 hint 也能 work。
- **Avg@8**：采样 8 次的平均。MEL 提升说明输出 consistency 提升、variance 下降。
- **Pass@8**：8 次里至少一次对。MEL 提升说明 exploration 没被压缩，反而探索到了更复杂的解法。

### 跨 paradigm 泛化

MEL 不只能加到 GRPO 上，加到 RFT（rejection sampling fine-tuning）和 REINFORCE++ 上都有提升。说明 meta-experience 是个 universal enhancement，不是绑定特定算法的。

参考 REINFORCE++：https://arxiv.org/abs/2501.03262

---

## 6. 一个具体例子帮你建立 intuition

### 假设原题是一道积分题

$$\int_{-1}^{1} \frac{1}{x} \, dx$$

### Stage 1: Rollout

8 条 trajectory，比如：
- $y_1$ (对): 检查 $\frac{1}{x}$ 在 $x=0$ 处无定义，积分发散，return "diverges"
- $y_2$ (错): 直接用 $\ln|x|$ 代入上下界，得到 $\ln 1 - \ln 1 = 0$，return "0"
- ...

### Stage 2: 写错题本

**Bifurcation point**：$y_2$ 在 "直接用 $\ln|x|$ 代入" 这一步开始走偏，没检查被积函数在积分区间内是否 well-defined。

**Critique**：错误 trajectory 把奇点积分当常规积分处理，忽略了被积函数在区间内有无定义域 hole。正确 trajectory 先做定义域检查，发现 $x=0$ 是奇点，积分发散。

**Heuristic**（抽象后）："When evaluating definite integrals, I must first check whether the integrand is well-defined across the entire integration interval. If a singularity exists within the interval, the integral diverges and standard antiderivative substitution does not apply."

注意：没提具体函数 $\frac{1}{x}$、没提具体区间 $[-1, 1]$、没提具体答案。这是一条普适 rule。

### Stage 3: Replay 验证

把这条 heuristic 拼进 prompt，让 model 重新解原题。如果这次 model 写出 "积分发散"，保留；如果还是算出 0，丢弃。

### Stage 4: Internalize

用 NLL 训练，让 model 在 "见到类似 retrospective context" 时能自动生成这种 "先检查定义域" 的反思。

---

## 7. 跟你熟悉的几件事的类比

### 7.1 跟 AlphaZero 的类比

AlphaZero:
- Self-play 探索棋局 → MCTS 评估 → 用 improved policy 训练 network

MEL:
- Rollout 探索解法 → Verifier 评估 → 用 contrastive analysis 抽取 improved cognitive pattern → NLL 训练

结构高度相似，区别是 MEL 用 natural language pattern 替代了 MCTS move distribution。

### 7.2 跟 STaR 的类比

STaR (Self-Taught Reasoner): 让 model 自己生成 reasoning，用 answer correctness 过滤，SFT 训练。

MEL: 让 model 自己生成 meta-experience（reasoning 的 reasoning），用 replay correctness 过滤，NLL 训练。

差一层 abstraction。STaR 训练 "怎么解题"，MEL 训练 "怎么反思解题"。

参考 STaR：https://arxiv.org/abs/2203.14465

### 7.3 跟 Self-Rewarding LM 的类比

Self-Rewarding LM: model 自己给自己打分（LLM-as-judge）。

MEL: model 自己给自己生成经验，用 rule-based verifier 间接验证（replay 解题）。

MEL 更可靠，因为最终 judge 还是 rule-based verifier，避免了 LLM judge 的 bias。

参考 Self-Rewarding LM：https://arxiv.org/abs/2401.10020

### 7.4 跟 "bitter lesson" 的呼应

Rich Sutton 的 bitter lesson 说：scale + general method 总会赢过 hand-crafted knowledge。

MEL 的 meta-experience 是 general method 自动 generate 的 knowledge，不是 human hand-crafted 的。它 scale：model 越大，meta-experience 质量越高，gain 越大。这跟 bitter lesson 的精神一致。

参考 The Bitter Lesson：http://www.incompleteideas.net/IncIdeas/BitterLesson.html

---

## 8. 实操上要注意什么

### 8.1 Compute 开销

每个 contrastive pair 要跑：
1. 定位 bifurcation point（1 次 forward）
2. 生成 critique + heuristic（1 次 forward）
3. Replay validation（1 次 forward）

3× 额外 forward。如果 batch 内一半样本有 contrastive pair，总 compute 大概比 vanilla GRPO 多 1.5-2×。

Paper 没明确 report 这个 overhead，但训练时间显著变长是肯定的。

### 8.2 Model 太小会退化

4B model 写的 critique 质量低，retention ratio 低，NLL 训练反而可能 internalize 错的 rule，污染 weights。

建议：在小 model 上要更严格的 validation（比如 replay 多次取多数），或者只在 model 能力达到某阈值后再启用 MEL。

### 8.3 Domain generalization

Paper 只测了 math reasoning。在 code、logic、tool use 等其他有 verifiable reward 的 domain 应该也 work，但 heuristic 的抽象难度不一样——math 的 rule 容易抽象成 "检查 X 约束"，code 的 rule 可能更复杂。

### 8.4 Catastrophic forgetting 风险

把新 meta-experience 持续 NLL 进 weights，可能 overwrite 旧能力。可以借鉴 experience replay 的思路，定期 revisit 旧 meta-experience。

参考 catastrophic forgetting 综述：https://arxiv.org/abs/1612.00796

---

## 9. 我的 take

### 9.1 这 paper 真正的 contribution

不是某个 trick，而是把 **"经验从 trajectory-level 提升到 knowledge-level"** 这个 framing 立起来了。一旦立起来，后面所有设计（contrastive pair, bifurcation localization, abstraction constraint, replay validation, NLL internalization）都是自然推导出来的。

### 9.2 最 elegant 的点

**NLL = reward=1 的 policy gradient** 这个数学等价性。让整套 method 在 implementation 上极简（GRPO loss + NLL loss 相加），但在 interpretation 上极丰富（dense process reward without trained proxy）。

参考这个数学等价性的讨论：https://arxiv.org/abs/2503.01067

### 9.3 跟 AGI 路线的关系

如果 LLM 要自我进化，必须有 "从自己的错误中抽象 reusable knowledge" 的能力。纯 RLVR 只能做 trial-and-error，不能做 reflection-and-abstraction。MEL 补上了这一环。

可以想象一个 iterated MEL：每轮训练后 model 变强，能抽出更深的 meta-experience，下一轮再 internalize，bootstrap 出更高 capability。这是 self-improvement loop 的雏形。

### 9.4 一个潜在的 "暗坑"

MEL 假设 LLM 写的 critique 即使经过一次 replay validation 也足够 robust。但一次 validation 通过不等于永远正确。某条 rule 在 math A 子领域有效，在 math B 子领域可能有害。如果无差别 internalize，可能在 A 提升、在 B 退化。

更严格的做法可能是：把 meta-experience 在多个相似 query 上分别 replay，要求多个 query 都通过才保留。这能减少 single-query overfit。

### 9.5 推荐你关注的方向

如果顺着 MEL 的思路往下做，几个有潜力的方向：

1. **Iterated MEL**: 多轮 self-distillation，每轮基于上轮 updated model 抽更深的 meta-experience
2. **Cross-domain MEL**: meta-experience 从 math 迁移到 code/logic，测试 abstraction 的 domain transferability
3. **Hierarchical meta-experience**: 不只抽 "具体 rule"，还抽 "怎么写 rule 的 meta-rule"（更高一层 abstraction）
4. **Meta-experience curriculum**: 按难度/类型排序 meta-experience，先学简单 rule 再学复杂 rule
5. **Active meta-experience generation**: 不依赖 rollout 自然产生的 contrastive pair，主动构造 "model 容易错的边界 case" 来定向生成 meta-experience

---

## 10. 终极人话版

LLM 现在 learning 的方式是 "做题→对答案"，没有 "错题本" 环节。

MEL 让 LLM 自己写错题本：找自己哪步错的、为什么错、抽象成普适 rule、用 "重做一遍能不能做对" 来验证 rule 有没有用、用 NLL 把 rule 背进 weights。

效果：4B/8B/14B 三个 size 上，Pass@1 一致提升 4-5 个百分点，而且 scale 越大提升越明显。

核心 insight：experience 应该是 knowledge-level 而不是 trajectory-level，应该进 weights 而不是 context，应该 self-distilled 而不是 externally provided。

参考与延伸阅读：
- MEL 论文本身（arXiv 搜索 "Meta-Experience Learning"）
- GRPO: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DAPO: https://arxiv.org/abs/2503.14476
- STaR: https://arxiv.org/abs/2203.14465
- Self-Rewarding LM: https://arxiv.org/abs/2401.10020
- Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- Self-critique: https://arxiv.org/abs/2206.05802
- Knowledge Distillation: https://arxiv.org/abs/1503.02531

---

# MEL: Meta-Experience Learning 详解

## 1. Paper 一句话核心

MEL (Meta-Experience Learning) 在 standard RLVR 之上加了一个 **knowledge-level learning loop**：让 LLM 通过对比自己的正确/错误 trajectory，定位 **bifurcation point**（推理分叉点），蒸馏出 natural language 形式的 reusable heuristic（即 "meta-experience"），通过 empirical validation via replay 过滤后用 NLL loss 把它 internalize 进 parametric memory，从而把它当作 dense process-level reward 注入 GRPO 训练。

参考链接：
- arXiv: https://arxiv.org/abs/2503.14476 (DAPO/RLVR)
- GRPO: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Let's Verify Step by Step (PRM 起源): https://arxiv.org/abs/2305.20050

---

## 2. Build Intuition: 为什么 RLVR 不够

### 2.1 Human Learning Cycle 的三阶段

人类学习的认知循环：
1. **Practice and verification**：做题 + 对答案
2. **Error attribution**：分析 "我哪一步开始走偏了" "为什么走偏"
3. **Experience internalization**：把教训抽象成原则、规则、pattern，下次直接调用

### 2.2 RLVR 实际只做了第一件事

Standard RLVR (GRPO/REINFORCE++/DAPO/GSPO) 的 reward signal：

$$r_i = \mathbb{I}[V(y_i, y^*)] \in \{0, 1\}$$

变量含义：
- $r_i$: 第 $i$ 个 rollout trajectory 的 binary outcome reward
- $\mathbb{I}[\cdot]$: indicator function（条件成立为 1，否则 0）
- $V(\cdot, \cdot)$: rule-based verifier，比对 extracted answer 和 ground truth
- $y_i$: 第 $i$ 条 generated reasoning trajectory
- $y^*$: ground-truth answer

这个 reward 只告诉你 "对了/错了"，**完全没有 attribution 信息**。这意味着：
- Gradient 对 trajectory 上的每个 token 一视同仁（advantage 是 trajectory-level scalar，broadcast 到所有 token）
- Model 学到的是 "整体行为像 successful trajectory"，学不到 "下次见到这种 constraint 一定要先检查 X"
- 等同于学生只对答案，从来不做错题本

### 2.3 已有改进路线的瓶颈

| 方向 | 代表方法 | 问题 |
|------|----------|------|
| Process Reward Model (PRM) | Lightman et al. 2023, Khalifa et al. 2025 | Trained proxy 容易 reward hacking；违背 RLVR "programmatically verifiable" 的初衷 |
| External memory (test-time scaling) | SpeedupLLM, Training-Free GRPO, ReasoningBank | 经验没进 weights，inference-time context 越堆越长，intrinsic capability 没提升 |
| Hint injection (training-time) | StepHint, Scaf-GRPO, LUFFY, SRFT | 经验作为 prefix/hint，要么 off-policy distribution shift（StepHint），要么 inference-time 不可用（Scaf-GRPO），仍然是 trajectory-level imitation，没有抽象到 knowledge-level |

参考链接：
- StepHint: https://arxiv.org/abs/2507.02841
- Scaf-GRPO: https://arxiv.org/abs/2510.19807
- LUFFY: https://arxiv.org/abs/2504.14945
- SRFT: https://arxiv.org/abs/2506.19767
- Training-Free GRPO: https://arxiv.org/abs/2510.08191
- ReasoningBank: https://arxiv.org/abs/2509.25140
- SpeedupLLM: https://arxiv.org/abs/2505.20643

---

## 3. Meta-Experience 的核心定义

$$\mathcal{M} = (s^*, \mathcal{C}, \mathcal{H})$$

变量含义：
- $\mathcal{M}$: 一条 meta-experience tuple
- $s^*$: bifurcation point，正确/错误 trajectory 开始分叉的 reasoning step
- $\mathcal{C}$: critique，即 "deep diagnosis" 的文本输出，包含 error attribution、comparative strategic gap、corrective principle
- $\mathcal{H}$: heuristic，即抽象后的 generalizable rule，类似 "If [abstract condition], then [abstract action]" 的形式

**关键 insight**：MEL 把 experience 从 "trajectory-level instance" 提升到 "knowledge-level representation"。一条 trajectory 只能用一次，一条 heuristic 可以在成千上万道相似结构题目上复用。

---

## 4. MEL 的四阶段流水线

### 4.1 Stage 1: Explorative Rollout（§3.1）

GRPO 式 group rollout，对每个 query $x$ 采样 $G$ 条 trajectory：

$$\mathcal{Y} = \{y_1, y_2, \dots, y_G\}, \quad \mathcal{Y}^+ = \{y_i \mid r_i = 1\}, \quad \mathcal{Y}^- = \{y_i \mid r_i = 0\}$$

变量含义：
- $\mathcal{Y}$: 一个 group 内的全部 $G$ 条 trajectory
- $\mathcal{Y}^+$: 正确 trajectory 子集
- $\mathcal{Y}^-$: 错误 trajectory 子集

**Important constraint**: 只有 $\mathcal{Y}^+ \neq \emptyset$ **且** $\mathcal{Y}^- \neq \emptyset$ 的样本才会进入下一步。这是因为 contrastive analysis 需要正负对照。换句话说，"全错" 的题目没有信息量（model 还没能力解），"全对" 的题目也没有信息量（已经掌握了），只有 "model 有能力解对但有时会解错" 的题目才最 informative——这是 active learning 里典型的 "boundary 案例"。

实验里 batch size 128，每 prompt 8 个 rollout，所以理论上每 batch 最多能产生 16 个 contrastive pair（如果一半对一半错）。

### 4.2 Stage 2: Meta-Experience Construction（§3.2）

#### 4.2.1 Locating the Bifurcation Point

$$s^* \sim \pi_\theta(\cdot \mid I, x, y^+, y^-)$$

变量含义：
- $s^*$: bifurcation point（reasoning step index 或 step 内容）
- $\pi_\theta$: 当前 policy model（self-verification）
- $I$: structured instruction，引导 introspective analysis 的 prompt
- $x$: 原始 query
- $y^+, y^-$: 一对 correct/incorrect trajectory

**Intuition**: "Discrimination is easier than generation" —— 给定两条 trajectory 让 model 找出哪里分叉，比让 model 从头解题要简单得多。这是 Saunders et al. 2022 和 Swamy et al. 2025 已经验证过的 self-critique asymmetry。所以 model 自己生成的 $s^*$ 可信度比直接生成正确答案要高。

参考：Saunders et al. self-critique: https://arxiv.org/abs/2206.05802

#### 4.2.2 Deep Diagnosis and Abstraction

接着生成 critique $\mathcal{C}$：

$$\mathcal{C} \sim \pi_\theta(\cdot \mid I, x, y^+, y^-, s^*)$$

Critique 包含三部分：
1. **Error attribution**: $s^*$ 处为什么走偏（violated assumptions / erroneous sub-goals / overlooked constraints / misused principles）
2. **Comparative strategic gap**: 为什么正确 trajectory 没掉进去（precise knowledge application / explicit constraint verification / coherent knowledge representations / emergent self-correction）
3. **Corrective principle**: 修复建议

然后抽象成 heuristic $\mathcal{H}$：

$$\mathcal{H} \sim \pi_\theta(\cdot \mid I, x, y^+, y^-, s^*, \mathcal{C})$$

**Strict Generalization Constraint**（见 Appendix C 的 prompt）：
- Forbidden: 提到本题具体数字、变量、答案
- Required: 抽象成 "If [condition], then [action]" 或 "When dealing with [Concept X], I should [verify constraint]"

这一步很关键：如果不强制抽象，model 很容易写出 "本题答案是 $\sqrt{2}$，所以要先平方" 这种 instance-specific recap，这种东西对未来题目毫无泛化价值。强制抽象迫使 model 把具体 case 提升到 concept-level pattern，类似 human 写错题本时把 "这道题" 抽象成 "这类题"。

#### 4.2.3 Empirical Validation via Replay

这一步是为了解决 self-generated critique 的 hallucination 问题。直接用 LLM 自己生成的 critique 不一定真的有用——可能 model 写得头头是道，但下次还是会犯同样的错。

做法：把 $\mathcal{M}$ 作为 in-context hint 注入 prompt，让 model 重新解 $x$：

$$y_{\text{val}} \sim \pi_\theta(\cdot \mid x, \mathcal{M})$$

如果 $V(y_{\text{val}}, y^*) = 1$，保留这条 $\mathcal{M}$，否则丢弃：

$$\mathcal{D}_{\mathcal{M}^*} = \{(x, y^+, y^-, \mathcal{M}) \in \mathcal{D}_\mathcal{M} \mid \mathbb{I}[V(y_{\text{val}}, y^*) = 1]\}$$

变量含义：
- $y_{\text{val}}$: replay 生成的 trajectory
- $\mathcal{D}_\mathcal{M}$: 全部候选 meta-experience pool
- $\mathcal{D}_{\mathcal{M}^*}$: 通过验证的 meta-experience pool

**Intuition**: 这是 MEL 的 "natural selection" 环节。空谈 critique 谁都会，但要能让 model 在 inference 时确实避开错误并解对题，这条 meta-experience 才有 "生存权"。这跟 constitutional AI 中 self-revision、self-rewarding LM 中 self-reward validation 是一类思路：用 model 自己的下游表现来 filter 它自己上游的产物。

参考 Figure 9 的 retention ratio 数据：14B model 的 retention ratio 显著高于 4B，说明 model capacity 越大，self-generated critique 的 quality 越高，validation 通过率越高——这跟 paper §4.5 讲的 scaling law 一致。

### 4.3 Stage 3: Internalization（§3.3）

#### 4.3.1 NLL Loss 形式

把 validated meta-experience 通过 token-averaged NLL 编进 weights：

$$\mathcal{L}_{\text{NLL}}(\theta) = -\mathbb{E}_{(x, y^+, y^-, \mathcal{M}^*) \sim \mathcal{D}_{\mathcal{M}^*}}\left[\frac{1}{|\mathcal{M}^*|}\sum_{t=1}^{|\mathcal{M}^*|} \log \pi_\theta(\mathcal{M}^*_t \mid C_{\text{retro}}, \mathcal{M}^*_{<t})\right]$$

变量含义：
- $\theta$: policy model 参数
- $\mathcal{D}_{\mathcal{M}^*}$: validated meta-experience pool
- $\mathcal{M}^*$: 一条 validated meta-experience 的 token sequence
- $|\mathcal{M}^*|$: 该 sequence 的长度
- $\mathcal{M}^*_t$: 第 $t$ 个 token
- $\mathcal{M}^*_{<t}$: 前 $t-1$ 个 token 的 prefix
- $C_{\text{retro}} = [I, x, y^+, y^-]$: retrospective context（包括原始 instruction、query 和对照 trajectory）
- $\pi_\theta(\cdot \mid \cdot)$: policy 的 next-token probability

第二行展开是对 rollout 的 marginal expectation，$\tau(\cdot)$ 表示 §3.2 中构造 meta-experience 的随机过程（包括 bifurcation localization、critique generation、abstraction、validation）。

#### 4.3.2 为什么 NLL 等价于 Process Reward

把 $\mathcal{L}_{\text{NLL}}$ 取负、看作 reward，定义 **Meta-Experience Return**：

$$\mathcal{R}_{\text{MEL}} = \mathbb{E}_{(y^+, y^-, \mathcal{M}^*) \sim \tau(x, \{y_i\}_{i=1}^G)}\left[\frac{1}{|\mathcal{M}^*|}\sum_{t=1}^{|\mathcal{M}^*|} \log \pi_\theta(\mathcal{M}^*_t \mid C_{\text{retro}}, \mathcal{M}^*_{<t})\right]$$

**关键 observation**：NLL 的 gradient 关于 policy parameter $\theta$ 来说，等价于一个 reward 恒为 +1 的 policy gradient：

$$\nabla_\theta \mathcal{R}_{\text{MEL}} = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(\mathcal{M}^*_t \mid \cdots)\right] = \mathbb{E}\left[\sum_t \frac{\nabla_\theta \pi_\theta(\mathcal{M}^*_t \mid \cdots)}{\pi_\theta(\mathcal{M}^*_t \mid \cdots)}\right]$$

这正好是 REINFORCE 在 reward $R \equiv 1$、state 是 $(C_{\text{retro}}, \mathcal{M}^*_{<t})$、action 是 $\mathcal{M}^*_t$ 时的 policy gradient estimator。

**Intuition**: 这意味着每个 $\mathcal{M}^*_t$ token 都受到 +1 reward 的 positive reinforcement。与 RLVR outcome reward 只在 trajectory 末尾给一次 0/1 不同，$\mathcal{R}_{\text{MEL}}$ 是 **dense, step-level reward**，对 meta-experience 的每一个 token 都给信号。

更深一层：$\mathcal{M}^*$ 是 model 自己生成的、经过 validation 验证有用的 "high-quality reasoning trace"，所以最大化它的 likelihood 等同于在 "what model should output when doing meta-reflection" 这个 action space 上做 positive shaping。换句话说，MEL 在训练 model **怎么做错题本**，而 RLVR 在训练 model **怎么解题**，两者同步进行。

参考：Swamy et al. "All roads lead to likelihood" 关于 NLL 与 RL gradient 等价性的讨论：https://arxiv.org/abs/2503.01067

#### 4.3.3 Parametric Memory vs Context Window

Paper 强调：把 meta-experience 写进 weights 而不是 context，有两个好处：
1. **Capacity unlimited**: parametric memory 几乎无限，context window 有限
2. **Inference-time zero overhead**: inference 时不需要 prepend 长 context

这一点跟 RAG-based experience retrieval 形成鲜明对比。Training-Free GRPO、ReasoningBank 这类方法 inference 时要检索并拼接 memory，context 越长越慢越贵。MEL 把这些一次性 "消化" 进 weights，inference 时是免费的。

### 4.4 Stage 4: Joint Training Objective（§3.4）

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{RLVR}}(\theta) + \mathcal{L}_{\text{MEL}}(\theta)$$

展开：

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min\left(\rho_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) + \mathcal{R}_{\text{MEL}}\right]$$

变量含义：
- $x \sim \mathcal{D}$: 从训练集采样 query
- $\{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)$: 从 old policy 采样 $G$ 条 trajectory
- $|y_i|$: 第 $i$ 条 trajectory 的 token 长度
- $\rho_{i,t}(\theta) = \pi_\theta(y_{i,t} \mid y_{i,<t}) / \pi_{\theta_{\text{old}}}(y_{i,t} \mid y_{i,<t})$: importance sampling ratio
- $\hat{A}_{i,t}$: group-normalized advantage（GRPO: within-group reward standardization, broadcast to each token）
- $\epsilon$: PPO clipping range
- $\mathcal{R}_{\text{MEL}}$: 公式 (8) 的 meta-experience return

**Unifying view**: 整个 $\mathcal{L}(\theta)$ 可以看作 **maximize expected cumulative return of a hybrid reward function**。Outcome reward 来自 verifier（sparse, terminal），process reward 来自 $\mathcal{R}_{\text{MEL}}$（dense, every token）。

### 4.5 整体架构图解析

Figure 2 的 pipeline：

```
Query x
   │
   ├─→ Group Rollout (G trajectories) ──→ Verifier ──→ Y+, Y-
   │                                                      │
   │                                                      ▼
   │                                       Construct Contrastive Pairs (y+, y-)
   │                                                      │
   │                                                      ▼
   │                                       Bifurcation Localization (s*)
   │                                                      │
   │                                                      ▼
   │                                       Deep Diagnosis (C) → Abstraction (H)
   │                                                      │
   │                                                      ▼
   │                                       Meta-Experience M = (s*, C, H)
   │                                                      │
   │                                                      ▼
   │                                       Empirical Validation via Replay
   │                                                      │
   │                                       ┌──────┴──────┐
   │                                  Pass                Fail
   │                                       │              │
   │                                       ▼              ▼
   │                                  D_M*             Discard
   │                                       │
   │                                       ▼
   │                          NLL Internalization (R_MEL)
   │                                       │
   │                                       ▼
   └─→ GRPO Loss (L_RLVR) ──→  Joint L = L_RLVR + L_MEL  ──→ Update θ
```

---

## 5. 实验结果详细解析

### 5.1 主实验数据（Table 1）

数据集：DAPO-Math-17k 训练，评估在 5 个 math benchmark：AIME24, AIME25, AMC23, MATH500, OlympiadBench。

硬件：8×H20 GPU，VERL framework，Math-Verify 做 rule-based verification。训练 hyperparameter：8 rollouts/prompt, temperature 1.0, batch 128, lr $1 \times 10^{-6}$。评估：Pass@1 用 temperature 0，Avg@8 和 Pass@8 用 temperature 0.6。

#### Qwen3-14B-Base 关键数据点

| Benchmark | Baseline Pass@1 | GRPO Pass@1 | MEL Pass@1 | Δ vs GRPO |
|-----------|-----------------|-------------|------------|-----------|
| AIME24 | 13.33 | 30.00 | 33.33 | +3.33 |
| AIME25 | 6.66 | 33.33 | 36.67 | +3.34 |
| AMC23 | 60.00 | 75.00 | 82.50 | +7.50 |
| MATH500 | 80.80 | 85.00 | 90.80 | +5.80 |
| OlympiadBench | 45.25 | 58.16 | 61.87 | +3.71 |

#### Avg@8 (Qwen3-14B-Base)

| Benchmark | Baseline | GRPO | MEL |
|-----------|----------|------|-----|
| AIME24 | 10.83 | 35.41 | 35.83 |
| AIME25 | 9.58 | 24.17 | 30.00 |
| AMC23 | 51.25 | 75.94 | 82.81 |
| MATH500 | 74.15 | 88.35 | 90.80 |
| OlympiadBench | 40.50 | 58.46 | 60.90 |

#### Pass@8 (Qwen3-14B-Base)

| Benchmark | Baseline | GRPO | MEL |
|-----------|----------|------|-----|
| AIME24 | 36.67 | 56.67 | 60.00 |
| AIME25 | 33.33 | 43.33 | 46.67 |
| AMC23 | 82.50 | 95.00 | 95.00 |
| MATH500 | 93.60 | 96.40 | 97.20 |
| OlympiadBench | 65.58 | 74.78 | 75.82 |

### 5.2 三个 metric 的语义解读

- **Pass@1** (temperature 0): one-shot 可靠性。MEL 的 gain 说明 internalized meta-experience 让 model 在 greedy decoding 下更倾向走正确路径——也就是说，**这些经验已经压进了 weights，不需要外部提示也能起作用**。
- **Avg@8** (temperature 0.6): 8 次采样的平均。MEL gain 说明 reasoning consistency 提升、output variance 下降——meta-experience 充当 "intrinsic prior"，把 generation distribution 收窄到 valid logic 区域。
- **Pass@8** (temperature 0.6): 8 次里至少一次对。MEL gain 说明 **exploration 没被压缩**，反而 upper bound 被抬高——internalization 没让 model 变保守，反而让 model 探索到更复杂的 long-horizon solution。

### 5.3 Scaling 行为

Figure 3 + Figure 9 显示：
- 4B → 8B → 14B，MEL 相对 GRPO 的 margin 递增
- 14B 的 retention ratio 显著高于 4B（critique quality 高）

这是 self-distillation 的固有特性：teacher = student 时，student 能力越强，self-generated supervision 质量越高，self-distillation gain 越大。这跟 Hinton et al. 在 dark knowledge 上的发现一致，也跟 STaR、Self-Rewarding LM 的观察吻合。

### 5.4 Training Dynamics（§4.2）

Figure 3 的训练曲线显示一个有趣现象：
- **Vanilla GRPO**: 早期 ascent 缓慢，因为 outcome reward sparse，model 早期 performance 低时几乎拿不到 positive reward，gradient signal 微弱
- **MEL**: 早期就有 sharp ascent，因为 $\mathcal{R}_{\text{MEL}}$ 是 dense reward，即使 outcome reward 全 0，meta-experience token 仍然贡献正梯度，"bootstrap" 了早期训练

这是 process reward 相对 outcome reward 的天然优势，PRM 方法也有类似效果，但 PRM 需要 trained proxy，MEL 用 self-distilled natural language 规避了 reward hacking。

### 5.5 跨 paradigm 泛化（§4.4, Figure 5）

Paper 把 MEL 加到 RFT (Rejection Sampling Fine-Tuning) 和 REINFORCE++ 上，发现：
- RFT + ME: 缓解 rote memorization 和 overfitting（RFT 本质是 SFT on filtered trajectories，容易过拟合到 specific sample，加入 meta-experience 后变成 "学 pattern 而非学 answer"）
- REINFORCE++ + ME: 显著抬高 performance ceiling

这印证了 paper §1 的 claim："meta-experience is a universal enhancement, not limited to the GRPO framework"。

参考 REINFORCE++: https://arxiv.org/abs/2501.03262

### 5.6 Reasoning Pattern 的质性变化（§4.3, Figure 4）

Case study 对比 GRPO 和 MEL：
- GRPO: 直接 numerical operation，缺 holistic view，复杂题容易错
- MEL: 先 explicit outline relevant theorems/formulas，再 execute；apply theorem 时会自发 "activate" internalized bitter lesson，做 constraint check 和 self-correction

这种 "structured preparatory strategy" + "emergent self-correction" 是 meta-experience 起作用的直接表现。Model 不是简单记住了某条 trajectory，而是 internalize 了 "在做 X 类型题前应该先做 Y 检查" 的 procedure。

---

## 6. 与 Related Work 的位置图

```
                      Outcome Reward         Process Reward
                     ┌─────────────────┬─────────────────────┐
   Verifiable        │  GRPO, DAPO,    │  MEL (this paper)   │
   (rule-based)      │  REINFORCE++,   │  - Self-distilled    │
                     │  GSPO           │  - NLL as process    │
                     │                 │    reward            │
                     ├─────────────────┼─────────────────────┤
   Learned Proxy     │  RLHF (PPO)     │  PRM (Lightman et    │
                     │                 │  al., Khalifa et al.)│
                     ├─────────────────┼─────────────────────┤
   External Memory   │  RAG-style      │  Training-Free GRPO, │
                     │                 │  ReasoningBank,      │
                     │                 │  SpeedupLLM          │
                     ├─────────────────┼─────────────────────┤
   Hint Injection    │  StepHint,      │  Scaf-GRPO, LUFFY,   │
                     │  (off-policy)   │  SRFT                │
                     └─────────────────┴─────────────────────┘
```

MEL 的独特位置：**Verifiable + Process Reward + Parametric Internalization**——这是 RLVR 范式下第一个 dense process reward 而不引入 trained proxy 的方法。

---

## 7. 公式 10 的梯度直觉

把 $\mathcal{L}(\theta)$ 拆成两项看 gradient：

$$\nabla_\theta \mathcal{L}(\theta) = \underbrace{\nabla_\theta \mathcal{L}_{\text{GRPO}}(\theta)}_{\text{outcome-driven exploration}} + \underbrace{\nabla_\theta \mathcal{R}_{\text{MEL}}(\theta)}_{\text{process-driven consolidation}}$$

第一项让 model 探索更广的 solution space（expansion），第二项让 model 巩固已发现的 cognitive pattern（consolidation）。这跟 awake/sleep 双阶段学习、exploration/exploitation trade-off、hippocampus/neocortex dual system memory 有 conceptual parallel。

更细一点看 $\nabla_\theta \mathcal{R}_{\text{MEL}}$：

$$\nabla_\theta \mathcal{R}_{\text{MEL}} = \mathbb{E}\left[\sum_{t=1}^{|\mathcal{M}^*|} \nabla_\theta \log \pi_\theta(\mathcal{M}^*_t \mid C_{\text{retro}}, \mathcal{M}^*_{<t})\right]$$

- Gradient 推 model 提高 "given retrospective context, generate this validated meta-experience token sequence" 的概率
- 等同于：在 "反思任务" 上做 SFT，但 reflection 本身是 model 自己生成的、经过 validation 过滤的
- 这正是 Self-Distillation 的标准形式

---

## 8. Prompt Template 解析（Appendix C）

### 8.1 Meta-Experience Prompt 的结构

Prompt 强制 model 输出 4 个 section：

1. **Failure Resolution Path & Error Pattern Recognition**：定位 bifurcation point，揭示 latent cognitive pattern（bias / missing prerequisite / prompt misunderstanding）
2. **Analysis of Success Factors**：从 correct trajectory 提取 robustness factor / reasoning pivot
3. **Meta-Cognitive Reflection (First-person)**：第一人称反思，"我" 视角
4. **Subject Heuristics (Internalized Experience)**：抽象 rule，强制 "If [abstract condition], then [abstract action]" 格式

**关键设计**：
- 第一人称强制让 model 把 critique 当 "自己的经验" 而非 "外部反馈"，这与 internalization 阶段的 NLL 训练目标一致——model 在 train 时学的是 "given context, I generate this self-reflection"
- "Strict Generalization Constraint" 显式 forbid 提具体数字，强制抽象
- "Deep Dive into Correct Trajectories" 而非只看 incorrect —— 这避免 model 只学到 "避开错误"，也要学到 "成功的关键是什么"

### 8.2 Empirical Validation Prompt

```
Prior study has provided some internal reference information...
{experience}
Now, please fully internalize this information as your own experience,
then independently think through the problem in detail and produce a complete answer.
```

注意措辞："fully internalize as your own experience" + "independently think" —— 既鼓励 model 用 meta-experience，又要求它独立推理（不能直接抄 meta-experience 里的 hint）。这模拟 "学生读错题本后合上书自己重做" 的场景。

---

## 9. Limitations 与 Open Questions

Paper 没有显式 limitations section，但从 method 本身能推断几个潜在问题：

1. **Compute overhead**: 每个 contrastive pair 要跑 3 次 model forward（localize bifurcation, generate critique+heuristic, replay validation）。如果 batch 内一半样本有 contrastive pair，每个 query 至少多 3×compute。Paper 没明确 report overhead，但训练时间应该显著高于 vanilla GRPO。

2. **Critique quality 上限 = model capability 上限**: 14B retention ratio 高于 4B 说明 critique quality 跟 model capacity 强相关。这意味着 MEL 在 weak model 上的增益有限（model 写不出有用的 critique）。这是 self-distillation 的 inherent limitation。

3. **Generalization across domain**: 实验全在 math reasoning。Code generation、logical reasoning、tool use 等其他 verifiable domain 上是否同样有效未知。Math 的 verifier 简单（答案对错），code 的 verifier 也类似（test pass），但 logical reasoning 的 verifier 设计更难。

4. **Contrastive pair 退化**: 如果某 query 的 rollout 全对或全错，无法形成 contrastive pair。随着 training 推进，model 越来越强，全对的 query 会越来越多，contrastive pair 越来越稀缺——这可能解释 §4.2 中后期 plateau。

5. **Distribution shift between train and inference**: Training 时 $C_{\text{retro}}$ 包含 $y^+, y^-$，inference 时只有 $x$。NLL 训练让 model 在 "见过 $y^+, y^-$ 的 context 下" 生成 $\mathcal{M}^*$，但 inference 时没有这个 context。Paper 没有显式分析这个 shift 的影响。

6. **Memory pollution risk**: 把低质量 meta-experience internalize 进 weights 可能 "污染" 已有能力。虽然 replay validation 过滤了一部分，但通过一次 validation 不代表永远正确——某条 heuristic 可能在某类题上有效，在另一类题上有害。Paper 没有讨论 catastrophic interference。

参考相关方向：
- Self-Rewarding LM: https://arxiv.org/abs/2401.10020
- STaR: https://arxiv.org/abs/2203.14465
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Dark Knowledge / Knowledge Distillation: https://arxiv.org/abs/1503.02531

---

## 10. 给 Karpathy 的 Intuition Takeaways

### 10.1 核心 metaphor

**RLVR = practice + grading; MEL = practice + grading + 错题本 + 内化错题本到 muscle memory**。

错题本不是 "把错题再背一遍"，而是 "提炼出 '见到 X 类型题要先检查 Y' 的 abstract rule"。这种 rule 在 weights 里，下次做题时自动 activate。

### 10.2 三层 abstraction 的提升

| 层次 | 代表方法 | Reusability |
|------|----------|-------------|
| Token-level | Vanilla SFT | 仅当前 sample |
| Trajectory-level | GRPO, Scaf-GRPO, LUFFY | 当前任务类 |
| Knowledge-level | MEL | 跨任务同结构 |

MEL 把 experience 从 trajectory-level 提升到 knowledge-level，这是 abstraction 维度上的根本跃迁。

### 10.3 NLL as Process Reward 的 elegance

最 elegant 的部分：NLL 的 gradient 数学上等价于 reward = +1 的 policy gradient。这意味着 MEL 不需要新的 optimizer、不需要新的 reward model、不需要 dual network——只是 GRPO loss + 一项 NLL loss 的简单相加。Implementation cost 极低，conceptual contribution 极高。

### 10.4 Self-Play 的另一种形态

MEL 可以看作一种 self-play：
- Player: rollout + solve
- Critic: bifurcation localization + critique
- Validator: replay validation
- Student: NLL internalization

四个角色都是同一个 $\pi_\theta$，通过 curriculum（contrastive pair selection）和 filter（replay validation）保证 self-generated signal 的质量。

### 10.5 与 Self-Play Preference Optimization (SPPO)、Self-Rewarding LM 的关系

MEL 的 self-validation 机制跟 Self-Rewarding LM 的 self-judging 机制 conceptually 类似，但 reward signal 不同：
- Self-Rewarding: model 自己打分（LLM-as-judge）
- MEL: model 自己解题（replay），用 rule-based verifier 验证

后者更可靠，因为最终 judge 仍然是 rule-based verifier，避免了 LLM judge 的 bias。

参考 SPPO: https://arxiv.org/abs/2405.15070

### 10.6 跟 AlphaZero 的隐喻对比

AlphaZero 的 self-play 流程：
1. MCTS 探索（rollout）
2. Value network 评估
3. Policy network 更新（学 MCTS improved distribution）

MEL 的流程：
1. Group rollout（探索）
2. Verifier 评估（binary outcome）
3. Contrastive analysis 抽取 meta-experience（"improved cognitive distribution"）
4. NLL internalization（policy update toward improved distribution）

结构上很相似：探索 → 改进 supervision → 更新 policy。MEL 用 natural language meta-experience 替代了 AlphaZero 中的 MCTS improved move distribution。

### 10.7 一个可能的研究方向

MEL 当前在 single-step internalization（一次 NLL 更新）。可以设想 multi-round 版本：
- Round 1: 抽取 meta-experience $\mathcal{M}_1$, internalize
- Round 2: 用 updated model 抽取更深的 meta-experience $\mathcal{M}_2$ (model 现在能看到更深层次的 error)
- Round 3: ...

这是 iterated self-distillation，类似 STaR 的 bootstrapping。理论上能逼近 model 的 capability frontier。

参考 STaR: https://arxiv.org/abs/2203.14465

---

## 11. 实操建议

如果要在自己的 RLVR pipeline 上复现 MEL：

1. **Verifier 选择**: Math 用 Math-Verify，code 用 sandbox execution，logic 用 constraint checker。Verifier 越可靠，contrastive pair 越干净。
2. **Rollout G 选择**: 太小（G=4）contrastive pair 稀缺，太大（G=32）compute 浪费。G=8 是 paper 用的 sweet spot。
3. **Replay validation temperature**: 应该跟 training temperature（1.0）一致，避免 overfit 到 greedy decoding。
4. **NLL loss weight**: Paper 公式 (10) 中 $\mathcal{R}_{\text{MEL}}$ 跟 GRPO loss 直接相加（权重 1:1）。可以考虑加 weight $\lambda$ 调节 consolidation vs exploration 的平衡。
5. **Curriculum**: 早期 model 能力低，contrastive pair 质量差，可以 late-start MEL（前 N steps 只跑 GRPO，N step 后再开 MEL）。
6. **Memory replay**: 防止 catastrophic forgetting，可以把 $\mathcal{D}_{\mathcal{M}^*}$ 当 replay buffer，定期 revisit 旧 meta-experience。

---

## 12. 总结

MEL 的贡献可以浓缩为三点：

1. **Concept**: 把 RLVR 的 outcome-level learning 扩展成 knowledge-level learning loop，模拟 human 的 error attribution + experience internalization。
2. **Method**: Self-distilled meta-experience via contrastive analysis + empirical validation via replay + NLL internalization。整套 pipeline 不引入外部 model、不引入 trained proxy、不引入 inference-time overhead。
3. **Empirical**: 在 4B/8B/14B 三个 scale 上 consistent gain 3.92%-4.73% Pass@1，跨 RFT/GRPO/REINFORCE++ 三种 paradigm 都有效。

最值得 internalize 的设计哲学：**让 model 自己生成、自己验证、自己消化自己的学习信号**，这是 self-improvement 的最朴素也最 robust 的形式。

参考链接汇总：
- VERL framework: https://arxiv.org/abs/2409.19256
- DAPO: https://arxiv.org/abs/2503.14476
- GSPO: https://arxiv.org/abs/2507.18071
- Qwen3: https://arxiv.org/abs/2505.09388
- MATH dataset: https://arxiv.org/abs/2103.03874
- OlympiadBench: https://aclanthology.org/2024.acl-long.211/
- NuminaMath: https://huggingface.co/datasets/AI-MO/NuminaMath-CoT
- Tulu 3 (RLVR survey): https://arxiv.org/abs/2411.15124
- PPO: https://arxiv.org/abs/1707.06347
- Original MEL paper (本 paper 本身可能在 arXiv 上有 version): 建议直接搜索 "Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models"
