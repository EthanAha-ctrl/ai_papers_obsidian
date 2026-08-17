---
source_pdf: The Distillation Game Adaptive Attacks & Efficient Defenses.pdf
paper_sha256: 358ea0a49221cb14351efe5ca6195bbb909b2e77f34258073e203ffecfb4255f
processed_at: '2026-08-12T14:15:09-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话概括

大模型公司怕别人"抄"自家模型，想了各种防御招数。这篇paper说：你们防御评估的方式有问题——你们假设小偷傻乎乎地均匀学你给的所有数据，但现实中小偷会挑肥拣瘦，专门学对他最有用的那些。用正确的评估方式一看，你们引以为豪的防御其实不怎么样。顺便我提了个又便宜又好用的防御方法。

---

## 背景故事

想象你是OpenAI，花了上亿美元训练GPT-5。你把这个模型API开放出去，用户问问题，模型给出**详细推理过程**+**最终答案**。

问题来了：有人可以把你的推理过程全部收集起来，当成训练数据，训练一个自己的小模型。小模型学完之后，在某些任务上能接近你大模型的能力。这叫**distillation attack**——你花大钱练的capability被别人白嫖了。

所以你想防御。怎么防？你已经把答案给出去了，不能不给。但你可以在"怎么给"上做文章——比如：
- 故意把推理过程写得很绕、很碎，让别人学起来费劲
- 在推理过程中掺噪声
- 改变输出的概率分布，让"有价值"的输出少出现

现在学术界已经有几个防御方法了，比如ADS（Anti-Distillation Sampling），号称state-of-the-art。

---

## 这篇paper发现的问题

### 问题1：评估方式太天真

以前的防御研究是这样的：
1. 我发明一个防御方法
2. 我让一个"student"模型**均匀地**在你防御后的输出上训练
3. 看student的准确率降了多少，降得多说明防御好

这有什么问题？**现实中的小偷不会均匀学**。

打个比方：你是一个老师，要防止学生抄你的教案。你把教案乱序排列了，觉得这样别人就不好抄。然后你测试：让一个学生**从头到尾按顺序**读你的乱序教案，看他学得怎么样。发现学得不好，你说"我的防御有效！"

但真实的学生会怎么做？他会**挑着读**——先看哪些段落最有信息量，专门精读那些，跳过废话。你把教案乱序，他重新排序就行了。

这篇paper把这个叫**passive vs adaptive gap**。Passive student就是傻乎乎均匀学的，adaptive student会挑肥拣瘦。

### 问题2：没有统一框架

不同的防御方法背后其实有不同的"理论假设"——关于"什么样的输出对distiller最有用"。但没有一个统一框架来比较这些假设，也没有办法系统地设计新防御。

---

## 这篇paper的解法

### 核心idea：用博弈论框架

把问题建模成一个游戏：
- **Teacher（你，防守方）**：选择一个"发布的模型"$\pi_{rel}$，但不能离原始模型$\pi_{ref}$太远（要保证质量）
- **Student（小偷，攻击方）**：看到你的$\pi_{rel}$后，选择一个"有效训练分布"$\pi_{eff}$，但不能离$\pi_{rel}$太远（只能filter/reweight你给的，不能凭空造）

两边都有一个"预算"——你偏离原始模型的程度（$\varepsilon$），和小偷reweight的程度（$\rho$）。

然后定义一个"value function"$v(x,y)$，衡量某个prompt-output对$(x,y)$对distillation有多有用。

游戏目标：

$$\inf_{\pi_{rel}} \sup_{\pi_{eff}} \mathbb{E}[v(x,y)]$$

人话：teacher先选发布策略，student看到后选最有利的训练分布，teacher要选一个让student能获得value最小的发布策略。这是minimax——teacher minimizes，student maximizes。

### 关键定理：两边最优策略都是"指数倾斜"

这篇paper的核心数学结果（Theorem 3.1）：

**Student的最优策略**：
$$\pi_{eff}^\star(y|x) \propto \pi_{rel}(y|x) \cdot e^{\eta \cdot v(x,y)}$$

人话：在你给的输出分布基础上，对高value的例子**指数放大权重**。value高的多采样，value低的少采样。$\eta$控制"挑肥拣瘦"的强度，$\eta=0$就是passive，$\eta$越大越selective。

**Teacher的最优策略**：
$$\pi_{rel}^\star(y|x) \propto \pi_{ref}(y|x) \cdot e^{-\lambda \cdot v(x,y)}$$

人话：在原始模型分布基础上，对高value的输出**指数压低权重**。学生最想学的，你偏偏不给（或少给）。

这两个公式是对称的，一个往高value倾斜，一个往低value倾斜。这个对称性很优雅。

### 从这个框架得到什么

**1. Adaptive evaluation protocol**

直接用student的最优策略作为评估方法：训练student的时候，先给每个example算value，然后按$v$做exponential reweighting，再训练。

算法很简单：每个batch里，算每个example的value，用softmax with temperature转成权重，然后做weighted SFT。

**2. Defense template**

Teacher的最优策略给了一个"防御模板"：随便选一个value function，往低value方向exponential tilt。选不同的$v$得到不同的defense。

**3. 重新解释已有的ADS**

如果选$v_{grad}$（基于gradient alignment的value），代入teacher最优策略，就得到ADS方法。所以ADS是这个框架的一个特例——它用的是最精确但最贵的value function。

**4. 新防御PoE**

$v_{grad}$太贵了（每个token都要backprop through student）。作者选了个便宜的proxy：

$$v_{gap}(x,y) = \log \pi_{ref}(y|x) - \log \pi_{stu}(y|x)$$

人话：teacher给这个输出的概率比student高多少（log ratio）。这个值大，说明teacher比student在这个输出上强很多，这正是distillation value的来源。

代入teacher最优策略，化简后得到：

$$\pi_{rel}^\star(y|x) \propto \pi_{ref}(y|x)^{1-\lambda} \cdot \pi_{stu}(y|x)^\lambda$$

人话：把teacher和proxy student的概率分布做**几何平均**（加权）。$\lambda$控制混多少student进来。

实现上就是：每个token位置，取teacher logits和student logits的加权平均，然后softmax采样。**纯forward pass，不需要gradient，非常便宜**。

这就是PoE（Product-of-Experts）。名字来自Hinton 2002的Product of Experts传统——多个expert的概率相乘（在log space就是相加）。

---

## 实验结果讲了什么故事

### 实验设置

- Teacher: DeepSeek-R1-Distill-Qwen-7B
- Proxy student（防御用）: Qwen2.5-3B  
- Final student（实际攻击者）: Llama-3.2-3B
- 任务: GSM8K和MATH（数学推理）

### 核心发现1：Adaptive gap巨大

以GSM8K上的ADS为例：
- Passive evaluation: student准确率34%（看起来防御很好）
- Adaptive evaluation: student准确率51%（防御效果大打折扣）
- **相对提升50%**

这意味着以前论文报的"我的防御把student从57%降到34%"这种结果，在很大程度上是假象。真实的小偷不会那么乖。

这和adversarial examples领域的历史很像——Athalye et al. 2018那篇"Obfuscated Gradients Give a False Sense of Security"破了一堆adversarial defense。这篇paper对distillation defense做了类似的事。

### 核心发现2：PoE在adaptive下和ADS差不多，但便宜得多

| 方法 | Teacher准确率 | Adaptive student准确率 | 时间开销 |
|------|--------------|----------------------|---------|
| ADS | 82.13% | 51.50% | 2.93x |
| PoE | 81.61% | 49.46% | **1.64x** |

在comparable teacher准确率下，PoE在adaptive evaluation下只比ADS差2个百分点，但时间开销只有ADS的一半多一点。

**为什么ADS在passive下看起来强很多，但adaptive下优势消失？**

ADS在passive下强是因为它直接针对uniform training优化。但adaptive student通过reweight可以mitigate ADS的效果——ADS suppress了最高value的输出，student就concentrate在次高value的输出上，仍然能recover大量capability。

PoE suppress的是likelihood gap，这是distribution-level property，更robust to reweighting。

### 核心发现3：PoE保留推理质量

ADS生成的trace经常是垃圾——重复"raid raid raid"几千次，或者乱七八糟。因为它optimize的目标就是"让student gradient没用"，而垃圾output的gradient确实没用。

PoE生成的trace是干净的、简洁的、可审计的。因为它要求output在teacher和student下都plausible，所以自然倾向于coherent reasoning。

用Claude Sonnet 4.6做rubric-based judge打分，PoE的trace质量显著高于ADS。人工校验30个样本，judge和human的quadratic-weighted kappa = 0.76（强agreement）。

---

## 几个intuition

### 1. 为什么exponential tilt是最优的

KL-constrained optimization的solution必然是exponential tilt form。这不是巧合，这是information geometry的必然结果。

类比统计力学：在能量约束下，熵最大的分布是Boltzmann分布$e^{-\beta E}$。这里KL约束下的最优分布也是exponential tilt。

RLHF里PPO的KL penalty、DPO的implicit reward，solution都是这个form。Distillation game这里也是。**同一套数学结构在不同问题里反复出现**。

### 2. 为什么adaptive evaluation重要

真正的攻击者是有"agency"的。他不是被动接受你给的数据，他会：
- 观察你防御后的输出
- 估计每个输出的价值
- 集中精力学高价值的
- 跳过低价值的

如果评估不respect这个capability，就是"打稻草人"。这个paper的贡献不在于说"以前的防御都不行"，而在于"你们的评估protocol有问题，让你们产生了虚假的安全感"。

### 3. 为什么PoE便宜又好用

$v_{grad}$需要gradient through student → 每个token都要backprop → 极贵
$v_{gap}$只需要forward pass算likelihood → 便宜

$v_{gap}$虽然只是$v_{grad}$的proxy，但实验显示在adaptive evaluation下两者效果接近。这说明**likelihood ratio是distillation value的good proxy**。

信息论直觉：teacher和student的log-likelihood ratio就是KL divergence的"local contribution"，衡量student在这个点能"学到多少"information。这和gradient alignment捕捉的是同一个东西的different aspect。

### 4. 为什么PoE保留trace质量

ADS: 目标是"让student gradient没用" → 可以通过output garbage实现 → garbage的gradient确实没用
PoE: 目标是"让teacher-student likelihood gap小" → 要求output在两模型下都plausible → 自然coherent

**优化目标决定了你会得到什么样的输出**。ADS的优化目标太"敌对"，容易collapse到degenerate solution。PoE的优化目标更"温和"，保留了plausibility约束。

### 5. Stackelberg game的double exponential

完整的game equilibrium下，teacher的策略是double exponential：

$$\pi_{rel}^\star(y|x) \propto \pi_{ref}(y|x) \cdot e^{-\lambda \cdot e^{\eta \cdot v(x,y)}}$$

为什么是double？因为teacher要考虑"student会做exponential tilt"这个事实。Student的tilt里$e^{\eta v}$，teacher要suppress的正是这个exponential quantity，所以teacher的penalty是$e^{-\lambda e^{\eta v}}$。

这是game的recursion：teacher防御student的defense against teacher的defense。实际实现用one-sided best response就够了，但equilibrium的form提供了理论理解。

---

## 对AI safety的implications

1. **Strong distillation很难阻止**。即使最好的防御，在adaptive attack下也只能把student准确率降低有限程度。如果你暴露任何有用信号，adaptive attacker就能recover大量capability。

2. **Summarization不够**。实验显示即使从GPT/Claude/Gemini的summarized traces训练，student仍能达到50%+ on GSM8K。隐藏chain-of-thought不能消除risk。

3. **评估protocol很重要**。未来的anti-distillation工作应该明确说明它against的是什么样的adaptive student，不能只报passive数字。

4. **Detection可能是complementary方向**。Watermarking、fingerprinting这些post-hoc方法可能和output manipulation互补。

5. **Defense-attack arms race会continue**。这篇paper是"attack"这边的一步——展示了adaptive evaluation。下一步可能是attacker想出更rich的adaptation，defender想出更robust的defense。

---

## 总结

这篇paper的价值在于**方法论**而非具体方法：

1. **统一框架**：把distillation attack/defense放进一个minimax game，defense和evaluation从同一个framework推出来
2. **Adaptive evaluation protocol**：纠正了field的evaluation bias，揭示passive数字是虚假的安全感
3. **PoE defense**：从框架自然推导出来的practical方法，便宜、有效、保留质量

PoE本身很简单（logits加权平均），但它的价值在于"从principles推出来"——不是拍脑袋想出来的heuristic，而是teacher best response的一个instance。这种"理论motivated的practical method"是好research的标志。

最后一句：**评价防御的时候，永远问自己——你against的attacker有多smart？**

---

# The Distillation Game: Adaptive Attacks & Efficient Defenses 深度讲解

## 1. 背景与动机：为什么distillation是个问题

LLM provider暴露越丰富的输出（reasoning traces, tool-use sequences, error corrections），user获得越好的experience，但同时这些输出可以被aggregated成imitation dataset，用于distillation攻击。Core tension: **utility和distillability之间的trade-off无法避免**。

现实背景：
- Anthropic, OpenAI, Google都已经hiding full chain-of-thought或者用summaries替代 ([Anthropic 2025](https://platform.claude.com/docs/en/build-with-claude/extended-thinking), [OpenAI 2025](https://developers.openai.com/api/docs/guides/reasoning), [Google 2025](https://ai.google.dev/gemini-api/docs/thinking))
- 但summarization并不能消除distillation risk（论文Appendix C.2和Fig. 5显示即使从GPT-5.4 mini, Claude Sonnet 4.6, Gemini 3 Flash Preview的summaries训练student，仍能recover大量capability）
- [Anthropic 2026b](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)确认distillation attacks是live operational issue

Prior work的问题：defense和evaluation孤立研究。Defense提出修改teacher output的rule，然后被训练在uniformly released data上的distiller评估。这导致两个open questions：
1. Defense应该被什么distiller judge？Realistic attacker不需要uniformly训练
2. 不同defense隐含不同的"usefulness theory"，但缺乏common framework比较

## 2. 核心框架：Distillation Game的形式化

### 2.1 三个key objects

| Symbol | Meaning |
|--------|---------|
| $\pi_{ref}$ | Reference teacher: undefended原始模型 |
| $\pi_{rel}$ | Released teacher: 实际暴露给user的defended模型 |
| $\pi_{eff}$ | Effective student distribution: student reweighting后看到的training distribution |
| $\pi_{stu}$ | 最终distilled student model |

关键区分：$\pi_{eff}$不是最终的student model，而是student训练时看到的effective data distribution。这捕捉了"filter/reweight/subsample"这种practical attack capability。

### 2.2 Teacher fidelity budget

$$\Pi_\varepsilon(\pi_{ref}) := \left\{\pi \in \Delta(\mathcal{Y}) : \mathbb{E}_{x\sim\mathcal{D}}\left[D_{KL}(\pi(\cdot|x) \| \pi_{ref}(\cdot|x))\right] \leq \varepsilon\right\}$$

变量含义：
- $\mathcal{Y}$: output space（reasoning traces + answers）
- $\mathcal{D}$: prompt distribution
- $\varepsilon \geq 0$: teacher允许偏离reference的budget
- $D_{KL}(\pi \| \pi_{ref})$: KL divergence from $\pi$ to $\pi_{ref}$

为什么用$\pi \| \pi_{ref}$方向？这个方向penalize released model在reference认为unlikely的output上放mass。即released不能"发明"reference不会生成的内容，只能reweight。这捕捉了fidelity constraint的本质。

### 2.3 Student adaptation budget

$$\Pi_\rho(\pi_{rel}) := \left\{\pi \in \Delta(\mathcal{Y}) : \mathbb{E}_{x\sim\mathcal{D}}\left[D_{KL}(\pi(\cdot|x) \| \pi_{rel}(\cdot|x))\right] \leq \rho\right\}$$

- $\rho \geq 0$: student的adaptation budget
- $\rho = 0$: passive student，uniformly训练在released data上
- $\rho$增大: student可以越来越selectively concentrate on high-value outputs

关键约束：student的effective distribution必须absolutely continuous w.r.t. $\pi_{rel}$。即attacker不能invent新data source，只能filter/reweight已有的released outputs。这是realistic threat model。

### 2.4 Value function

$v: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$衡量$(x,y)$对distillation的usefulness。$v(x,y)$大iff训练在$(x,y)$上helps student more on downstream task。

**Gradient-based value** (first-order approximation):

$$v_{grad}(x,y) := -\langle \nabla_\theta \mathcal{L}(\theta_0), \nabla_\theta \log \pi_{stu}(y|x; \theta)|_{\theta=\theta_0}\rangle$$

变量含义：
- $\theta$: student参数
- $\theta_0$: student初始参数
- $\mathcal{L}(\theta)$: downstream task loss (e.g., NLL on held-out set)
- $\nabla_\theta \mathcal{L}(\theta_0)$: downstream task的gradient
- $\nabla_\theta \log \pi_{stu}(y|x; \theta)$: 在example $(x,y)$上的training gradient
- $\langle \cdot, \cdot \rangle$: inner product

Intuition: $v_{grad}$衡量"训练在这个example上的gradient"和"减小downstream loss的gradient"的alignment。Alignment大 → 这个example对减小downstream loss有用 → value高。负号是因为我们希望training gradient指向downstream gradient的反方向（gradient descent）。

这就是influence function的精神，参考[Koh & Liang 2017](https://arxiv.org/abs/1703.04730)。

**Likelihood-ratio (gap) value** (cheaper proxy):

$$v_{gap}(x,y) := \log \pi_{ref}(y|x) - \log \pi_{stu}(y|x)$$

这个值大iff teacher比student给这个output高得多likelihood。Intuition: 这些outputs标记了teacher有但student还没match的capability/confidence regions，因此是distillation value的carrier。

### 2.5 Minimax objective

$$\mathcal{V}(\varepsilon, \rho) := \inf_{\pi_{rel} \in \Pi_\varepsilon(\pi_{ref})} \sup_{\pi_{eff} \in \Pi_\rho(\pi_{rel})} \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_{eff}(\cdot|x)}[v(x,y)]$$

变量含义：
- Outer inf: teacher选择$\pi_{rel}$在fidelity budget内
- Inner sup: student选择$\pi_{eff}$在adaptation budget内maximize expected value
- $\mathcal{V}(\varepsilon, \rho)$: game value，即在threat model$(\varepsilon, \rho)$下的minimax expected distillation value

这个formulation的elegance: budgets $(\varepsilon, \rho)$指定threat model，value function $v$指定"usefulness theory"。Framework和具体choice解耦。

## 3. Theorem 3.1: Best Responses的closed form

### 3.1 Statement

$$\pi_{eff}^\star(y|x) \propto \pi_{rel}(y|x) e^{\eta v(x,y)}, \qquad \pi_{rel}^\star(y|x) \propto \pi_{ref}(y|x) e^{-\lambda v(x,y)}$$

其中$\eta, \lambda \geq 0$是dual variables。当KL constraints active时，$\eta, \lambda$是unique constants使得budgets tight:
- $\mathbb{E}_{x\sim\mathcal{D}}[D_{KL}(\pi_{eff}^\star(\cdot|x) \| \pi_{rel}(\cdot|x))] = \rho$
- $\mathbb{E}_{x\sim\mathcal{D}}[D_{KL}(\pi_{rel}^\star(\cdot|x) \| \pi_{ref}(\cdot|x))] = \varepsilon$

### 3.2 Proof intuition

考虑student的best response。我们要解:

$$\sup_{\pi_{eff}} \left\{\mathbb{E}[v] : \mathbb{E}[D_{KL}(\pi_{eff} \| \pi_{rel})] \leq \rho\right\}$$

这是convex program (linear objective + convex constraint)。Lagrangian:

$$\mathcal{L}(\pi_{eff}, \eta) = \mathbb{E}_{x,y\sim\pi_{eff}}[v(x,y)] - \frac{1}{\eta}\left(\mathbb{E}[D_{KL}(\pi_{eff} \| \pi_{rel})] - \rho\right)$$

对$\pi_{eff}(\cdot|x)$求导，pointwise优化:

$$\frac{\partial}{\partial \pi_{eff}(y|x)}\left[\pi_{eff}(y|x) v(x,y) - \frac{1}{\eta}\pi_{eff}(y|x)\log\frac{\pi_{eff}(y|x)}{\pi_{rel}(y|x)}\right] = 0$$

$$v(x,y) - \frac{1}{\eta}\left(\log\frac{\pi_{eff}(y|x)}{\pi_{rel}(y|x)} + 1\right) = 0$$

$$\log\frac{\pi_{eff}(y|x)}{\pi_{rel}(y|x)} = \eta v(x,y) - 1$$

$$\pi_{eff}(y|x) \propto \pi_{rel}(y|x) e^{\eta v(x,y)}$$

这就是**exponential tilt** toward high value。Teacher side symmetric，得到exponential tilt away from high value。

### 3.3 与information theory的connection

这个exponential tilt的形式就是**Gibbs distribution / Boltzmann distribution**。在large deviations theory中，exponential tilt是constrained MLE的最优form。KL-constrained optimization的solution必然是这个form，这和RLHF中RL with KL penalty的solution形式一致（参考[Schulman et al. 2017](https://arxiv.org/abs/1707.06347) PPO, [Rafailov et al. 2023](https://arxiv.org/abs/2305.18290) DPO）。

DPO的implicit reward $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}$就是这种exponential tilt的另一面。这个framework的elegance在于把distillation attack/defense统一到同一个information geometric structure下。

## 4. Implication 1: Adaptive Student Attack

### 4.1 Algorithm 1解析

```
Algorithm 1: Adaptive distillation attack with gradient-based value
Require: Traces T = {(x^(i), y^(i))}, student init θ_0, stepsize α, sharpness η
Require: Student downstream loss L(θ)
1: for each training step / minibatch B ⊂ T do
2:   g ← ∇_θ L(θ)                    // downstream task gradient
3:   for each trace (x,y) ∈ B do
4:     v_grad(x,y) ← -⟨g, ∇_θ log π_stu(y|x; θ)⟩
5:     w(x,y) ← exp(η · v_grad(x,y)) / Σ_{(x',y')∈B} exp(η · v_grad(x',y'))
6:   end for
7:   θ ← θ - α ∇_θ [Σ_{(x,y)∈B} w(x,y) Σ_t -log p(y_{t+1}|y_{1:t}, x; θ)]
8: end for
9: return θ
```

Intuition:
- Step 2: 计算downstream loss gradient $g$（指向loss增加的方向）
- Step 4: 对每个example计算其training gradient和$g$的对齐度（负号因为gradient descent）
- Step 5: Softmax over batch with temperature $1/\eta$，得到importance weights
- Step 7: Weighted SFT loss

这本质上是**importance sampling for gradient alignment**。$\eta$控制selectivity: $\eta=0$ uniform weights (passive), $\eta\to\infty$ concentrated on best aligned example。

### 4.2 Passive vs Adaptive gap

Table 1的关键数据：

| Dataset | Teacher | Passive Acc ↓ | Adaptive† Acc ↓ | Rel. Gain |
|---------|---------|---------------|------------------|-----------|
| GSM8K | ADS (λ=0.052) | 34.33% | 51.50% | **50.04%** |
| GSM8K | PoE (γ=0.65) | 39.26% | 49.46% | 25.98% |
| MATH | ADS (λ=0.08) | 8.96% | 13.45% | **50.07%** |
| MATH | PoE (γ=0.75) | 9.00% | 12.92% | 43.56% |

**核心发现**: Adaptive evaluation增加student accuracy约50% relative。这意味着prior work的defense评估都被严重overestimate了。这个gap大到足以改变defenses之间的qualitative comparison。

类比: 这和adversarial examples早期的情况类似——[Athalye et al. 2018](https://arxiv.org/abs/1802.00420) "Obfuscated gradients give a false sense of security"指出许多adversarial defense在adaptive attack下崩溃。这篇paper做了distillation领域的类似contribution。

## 5. Implication 2: Teacher-Side Defenses

### 5.1 General defense template

$$\pi_{rel}^\star(y|x) \propto \pi_{ref}(y|x) e^{-\lambda v(x,y)}$$

任何value function $v$ → 对应defense。$\lambda$控制defense强度。

### 5.2 ADS (Anti-Distillation Sampling)

用$v_{grad}$:

$$\pi_{rel}^\star(y|x) \propto \pi_{ref}(y|x) \exp\left\{\lambda \langle \nabla_\theta \mathcal{L}(\theta_0), \nabla_\theta \log \pi_{stu}(y|x; \theta)\rangle\right\}$$

这suppresses outputs whose student gradient is most aligned with reducing downstream risk。这就是[Savani et al. 2025](https://arxiv.org/abs/2504.13146) ADS方法的objective的game-theoretic interpretation。

**Drawback**: 需要student gradient during decoding。Autoregressive generation中每个token都要backprop through student → 非常expensive。

### 5.3 PoE (Product-of-Experts)

用cheaper $v_{gap}$:

$$\pi_{rel}^\star(y|x) \propto \pi_{ref}(y|x) \exp\left\{-\lambda(\log \pi_{ref}(y|x) - \log \pi_{stu}(y|x))\right\}$$

简化:

$$\pi_{rel}^\star(y|x) \propto \pi_{ref}(y|x)^{1-\lambda} \pi_{stu}(y|x)^\lambda$$

这就是teacher和(proxy) student的**geometric mixture**。在[Hinton 2002](https://www.cs.toronto.edu/~hinton/absps/ncopt.ps)的Product-of-Experts传统中，多个probabilistic models通过multiplying densities组合。这里product form作为proxy teacher response出现，不是generative modeling assumption。

实际实现（Algorithm 2）用token-level approximation:

```
Algorithm 2: Product-of-Experts (PoE) Sampling
Require: Prompt x_{1:n}, max tokens N, teacher θ_T, proxy student θ_S, mixture γ ∈ (0,1), temperature τ
1: for t = n, n+1, ..., N-1 do
2:   z_T(·) ← log p(·|x_{1:t}; θ_T)         // teacher logits
3:   z_S(·) ← log p(·|x_{1:t}; θ_S)         // proxy student logits
4:   x_{t+1} ∝ exp((1/τ)[(1-γ) z_T(·) + γ z_S(·)])
5: end for
6: return x_{1:N}
```

注意: PoE是**add** student logits，不subtract。这和[contrastive decoding](https://aclanthology.org/2023.acl-long.522/) (Li et al. 2023) subtract small model logits to amplify capability gap相反。直觉: defense要pull released distribution toward student，让teacher-student gap变小，从而reduce $v_{gap}$。

### 5.4 PoE vs ADS的conceptual difference

**ADS**: 直接optimization against student's learning signal → 可能strongly distort readable reasoning
**PoE**: regularize released policy toward outputs plausible under both models → 保留teacher plausibility，suppress only large gap regions

这是PoE在trace quality上胜出的design rationale。

## 6. Stackelberg Equilibrium分析

### 6.1 Full game的reduction

Lemma B.1给student problem的variational form:

$$\Psi(\pi_{rel}) = \inf_{\eta > 0} \frac{\rho + \log \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_{rel}(\cdot|x)}[e^{\eta v(x,y)}]}{\eta}$$

这是**Donsker-Varadhan variational formula**的形式（参考[Dupuis & Ellis 2011](https://link.springer.com/book/10.1007/978-3-642-24568-1)）。$\log\mathbb{E}[e^{\eta v}]$是**cumulant generating function** (CGF)，衡量value的exponential moment。

Lemma B.3: 

$$\mathcal{V}(\varepsilon, \rho) = \inf_{\eta > 0} \inf_{\pi_{rel} \in \Pi_\varepsilon} \frac{\rho + \log \mathbb{E}_{x,y\sim\pi_{rel}}[e^\eta v]}{\eta}$$

对每个fixed $\eta$，inner minimization的solution:

$$\pi_{rel,\eta}^\star(y|x) \propto \pi_{ref}(y|x) e^{-\lambda e^{\eta v(x,y)}}$$

注意这里exponent是$e^{\eta v}$，不是$\eta v$！这是因为CGF本身是exponential moment，所以teacher要penalize的正是这个exponential quantity。

### 6.2 Theorem B.4: Coupled form

当两个KL constraints都active:

$$\pi_{eff}^\star(y|x) \propto \pi_{rel}^\star(y|x) e^{\eta^\star v(x,y)}$$
$$\pi_{rel}^\star(y|x) \propto \pi_{ref}(y|x) e^{-\lambda^\star e^{\eta^\star v(x,y)}}$$

这是**double exponential** structure for teacher。Intuition: teacher看到student会用exponential tilt (CGF)，所以teacher要suppress的正是CGF贡献大的examples，导致double exponential tilt。

这个coupled form在practical implementation中不一定直接使用（paper用one-sided best responses），但提供了theoretical foundation for理解game的equilibrium structure。

## 7. Experimental Setup细节

### 7.1 Models

| Role | Model | 
|------|-------|
| Reference teacher | DeepSeek-R1-Distill-Qwen-7B ([Guo et al. 2025](https://arxiv.org/abs/2501.12948)) |
| Proxy student (for defense) | Qwen2.5-3B ([Bai et al. 2023](https://arxiv.org/abs/2309.16609)) |
| Final distilled student | Llama-3.2-3B ([Grattafiori et al. 2024](https://arxiv.org/abs/2407.21783)) |

Proxy/final-student mismatch是intentional: 模拟realistic transfer-style evaluation，teacher防御的proxy和实际attacker架构不同。

### 7.2 Datasets

- **GSM8K** ([Cobbe et al. 2021](https://arxiv.org/abs/2110.14168)): 5238/2246/1319 train/holdout/test
- **MATH** ([Hendrycks et al. 2021](https://arxiv.org/abs/2103.03874)): 5000/2500/5000

### 7.3 Training hyperparameters

- LoRA rank 128, $\alpha=128$, dropout 0
- LR $5\times10^{-4}$, weight decay 0.1, gradient clip 1.0
- Cosine decay with warmup ratio 0.03
- Batch size 6, gradient accumulation 4
- Max sequence length 2048
- 3 epochs
- 3 seeds: 123, 456, 789
- Adaptive student sharpness $\beta_s = 0.5$

### 7.4 Defense sweeps

| Dataset | Defense | Sweep range |
|---------|---------|-------------|
| GSM8K | ADS | λ ∈ {0.043, 0.044, ..., 0.060} |
| GSM8K | PoE | γ ∈ {0.50, 0.55, ..., 0.75} |
| MATH | ADS | λ ∈ {0.07, 0.075, ..., 0.09} |
| MATH | PoE | γ ∈ {0.70, 0.725, ..., 0.90} |

## 8. Experimental Results深度解析

### 8.1 Table 1: Utility-distillability frontier representative points

```
GSM8K:
- Standard:  Teacher 87.22% | Passive 57.24% | Adaptive 56.74% | Time 1.00x
- ADS:       Teacher 82.13% | Passive 34.33% | Adaptive 51.50% | Time 2.93x
- PoE:       Teacher 81.61% | Passive 39.26% | Adaptive 49.46% | Time 1.64x

MATH:
- Standard:  Teacher 61.78% | Passive 15.17% | Adaptive 15.29% | Time 1.00x
- ADS:       Teacher 61.16% | Passive 8.96%  | Adaptive 13.45% | Time 3.85x
- PoE:       Teacher 60.07% | Passive 9.00%  | Adaptive 12.92% | Time 2.33x
```

关键观察:
1. Standard teacher下passive≈adaptive (57.24 vs 56.74)，因为no defense时uniform training已经接近optimal
2. Defenses下passive vs adaptive gap巨大（ADS: 34.33 vs 51.50，gap = 17.17）
3. PoE在adaptive下和ADS接近（49.46 vs 51.50），但time cost远低（1.64x vs 2.93x）
4. 在GSM8K representative point，PoE甚至在adaptive下略低于ADS（49.46 vs 51.50）

### 8.2 Figure 3: Frontier curves

Frontier plots显示:
- Adaptive frontier明显高于passive frontier
- Defenses shift frontier向下（reduce leakage），但under adaptive attack shift程度大幅减少
- GSM8K上ADS在passive下看似更强（34% < 39%），但adaptive下gap缩小（52% vs 49%）

### 8.3 Figure 4: Trace quality distributions

用[Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)作为rubric-based judge，1-5评分auditability:
1. No auditable reasoning
2. Few auditable steps
3. Partially auditable
4. Mostly auditable
5. Fully auditable

结果: PoE产生更多high-scoring traces，更接近standard teacher。ADS traces质量显著下降。

Human evaluation sanity check (30 traces):
- Quadratic-weighted Cohen's $\kappa = 0.76$ (strong agreement)
- Mean absolute error = 0.40
- Mean signed error = +0.13 (small upward offset)

### 8.4 Figure 5: Frontier model distillation

```
GSM8K:
- Base: 0.76%
- Q&A only: 3.43%
- Gemini 3 Flash Preview: 51.75%
- Claude Sonnet 4.6: 53.57%
- GPT-5.4 mini: 54.78%

MATH:
- Base: 1.01%
- Q&A only: 11.25%
- Gemini: 12.09%
- Claude: 16.33%
- GPT: 14.83%
```

即使是summarized traces，frontier models仍leak大量capability。这印证了"summarization alone不应被assume消除distillation risk"的论点。

### 8.5 Qualitative trace comparison (Appendix C.4)

看ADS的example trace:
```
raid raid raid raid raid raid raid raid raid raid... (重复数千次)
```

ADS产生大量degenerate output。PoE则产生clean, concise, well-structured reasoning。

PoE的一个interesting pattern: 不只是shorten trace，有时finds different mathematically valid route to same answer。例如functional equation问题中，unperturbed和ADS先derive general form $f(x) = k/x$，而PoE直接substitute $x=30, y=4/3$得到 $f(40) = 15$。

## 9. Related Work connections

### 9.1 Anti-distillation methods

- **ADS** ([Savani et al. 2025](https://arxiv.org/abs/2504.13146)): Inference-time, gradient-based, this paper的framework解释了它的objective
- **DOGe** ([Li et al. 2025](https://arxiv.org/abs/2505.19504)): Adversarial train final linear layer to maximize KL from proxy student
- **Fang et al. 2026** ([arxiv](https://arxiv.org/abs/2602.03396)): Learn logit transformation matrix via conditional MI objective
- **Trace rewriting** ([Ma et al. 2026](https://arxiv.org/abs/2602.15143), [Ding et al. 2025](https://arxiv.org/abs/2510.11545)): Post-hoc modify traces
- **Trace inversion** ([Zhang et al. 2026](https://arxiv.org/abs/2603.07267)): Train model to reconstruct CoT from summaries

### 9.2 Decoding-time LM combination

- **DExperts** ([Liu et al. 2021](https://aclanthology.org/2021.acl-long.530/)): Geometric combination with expert/anti-expert
- **Contrastive decoding** ([Li et al. 2023](https://aclanthology.org/2023.acl-long.522/)): Subtract small model logits
- **GeDI** ([Krause et al. 2021](https://aclanthology.org/2021.findings-emnlp.334/)): Generative discriminator guided
- **FUDGE** ([Yang & Klein 2021](https://aclanthology.org/2021.naacl-main.327/)): Future discriminator controlled

PoE区别: derived from first principles as teacher's best response, **adds** student logits (contrastive decoding subtracts)。

### 9.3 Watermarking and attribution

- **LLM watermarking** ([Kirchenbauer et al. 2023](https://arxiv.org/abs/2301.08254)): Post-hoc detection
- **Anti-distillation fingerprinting** ([Xu et al. 2026](https://arxiv.org/abs/2602.03812)): Detect distillation after fact

### 9.4 RLHF/KL regularization connections

- PPO ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)): KL penalty in RLHF
- DPO ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)): Implicit reward via log-ratio

Best response的exponential tilt form和这些方法的solution structure同构，都源于KL-constrained optimization的information geometry。

## 10. Intuition building总结

### 10.1 为什么adaptive evaluation重要

Passive evaluation下的"strong defense"可能在adaptive下崩溃。这不是defense设计错了，而是evaluation protocol太弱。Realistic attacker有能力filter/reweight/subsample released data，evaluation应该respect这个capability。

类比: adversarial examples领域，[Carlini & Wagner 2017](https://arxiv.org/abs/1608.04644)破了许多distillation-based defenses。Distillation game这里，adaptive student破了许多anti-distillation defenses。

### 10.2 为什么PoE在adaptive下接近ADS

ADS在passive下强是因为它直接optimizes against uniform training。但adaptive student通过reweighting，可以mitigate ADS的effect——ADS suppress的high-value outputs，adaptive student通过concentrate在剩余的moderate-value outputs来recover。

PoE suppress的是likelihood gap，这更robust to reweighting，因为gap是distribution-level property，不依赖于具体training distribution。

### 10.3 为什么PoE保留trace quality

ADS的objective是"make student gradient unhelpful"，这可以通过output garbage实现。Garbage output的gradient对student没用，所以ADS的gradient optimization可能collapse到degenerate outputs。

PoE的objective是"reduce likelihood gap"，这要求output仍然plausible under both teacher and student。所以PoE自然倾向于produce coherent, if compressed, reasoning。

### 10.4 Game theory的insight

Stackelberg game with teacher as leader: teacher先commit to $\pi_{rel}$，student观察后best-respond with $\pi_{eff}$。One-sided best responses (Theorem 3.1) 给practical implementation，full Stackelberg equilibrium (Theorem B.4) 给theoretical foundation。

Double exponential in teacher's equilibrium strategy反映game的recursion: teacher要defense against student who's optimizing against teacher's defense。CGF $\log\mathbb{E}[e^{\eta v}]$是student的optimal value under exponential tilt，teacher要minimize这个CGF，导致exponential of exponential。

### 10.5 Value function choice matters

- $v_{grad}$: 精确但expensive，requires gradient through student
- $v_{gap}$: cheap proxy，forward-pass-only，但只capture likelihood ratio信息

两者在adaptive evaluation下performance接近，suggesting likelihood ratio是distillation value的good proxy。这有information-theoretic解释: student-teacher的log-likelihood ratio就是student能"学到多少"的information measure。

### 10.6 Broader implications for AI safety

如果strong distillation难以阻止，那么:
1. Capability diffusion难以通过output manipulation阻止
2. Safety-relevant capabilities的distillation是real concern ([Trockman & Savani 2026](https://antidistillation.com/blog/unexpected-externalities-of-distillation/))
3. Detection-based methods（watermarking, fingerprinting）可能是complementary approach
4. Defense-attack arms race会continue，需要adaptive evaluation作为standard protocol

## 11. Limitations和future directions

Paper自己承认的:
- 只focus on adaptive reweighting，没考虑richer adaptive attacks (e.g., student fine-tunes its own model to better absorb released data)
- 有限的model和task family
- Proxy student choice的sensitivity没充分explore

我想补充的:
- Value function的choice有arbitrary成分，$v_{grad}$ vs $v_{gap}$的选择based on computational convenience
- KL budget $\varepsilon, \rho$的实际calibration不清晰
- 7B teacher + 3B student的scale gap相对小，larger gap下的behavior未知
- Game假设simultaneous knowledge of budgets，realistic setting下budgets unknown

## 12. 个人思考

这篇paper的main contribution不是defense itself（PoE相对simple），而是**methodological contribution**: 把distillation attack/defense统一到game-theoretic framework，并提供adaptive evaluation protocol。

这和[Koh & Liang 2017](https://arxiv.org/abs/1703.04730) influence functions的方法论价值类似——具体方法可能被superseded，但framework和evaluation protocol持久。

PoE的elegance在于它从first principles推导出来，同时practically cheap and preserves trace quality。这种"理论motivated的practical method"是好的research的标志。

对future work的hint: richer adaptive attacks（e.g., student meta-learns how to learn from defended outputs）、multi-step interactions的distillation game、formal hardness results（distillation resistance的computational complexity）、information-theoretic lower bounds on achievable defense。

---

**Reference links:**

- Paper: [The Distillation Game (arxiv)](https://arxiv.org/abs/2504.13146) (作者github: https://github.com/ysfalh/distillation-game)
- [Savani et al. 2025 ADS](https://arxiv.org/abs/2504.13146)
- [Anthropic 2026b Detecting distillation](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)
- [Athalye et al. 2018 Obfuscated gradients](https://arxiv.org/abs/1802.00420)
- [Hinton 2002 PoE](https://www.cs.toronto.edu/~hinton/absps/ncopt.ps)
- [Rafailov et al. 2023 DPO](https://arxiv.org/abs/2305.18290)
- [Schulman et al. 2017 PPO](https://arxiv.org/abs/1707.06347)
- [Li et al. 2023 Contrastive decoding](https://aclanthology.org/2023.acl-long.522/)
- [Liu et al. 2021 DExperts](https://aclanthology.org/2021.acl-long.530/)
- [Kirchenbauer et al. 2023 Watermarking](https://arxiv.org/abs/2301.08254)
- [Koh & Liang 2017 Influence functions](https://arxiv.org/abs/1703.04730)
- [Guo et al. 2025 DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Cobbe et al. 2021 GSM8K](https://arxiv.org/abs/2110.14168)
- [Hendrycks et al. 2021 MATH](https://arxiv.org/abs/2103.03874)
- [Trockman & Savani 2026 Antidistillation externalities](https://antidistillation.com/blog/unexpected-externalities-of-distillation/)
