---
source_pdf: Parallel-R1 Towards Parallel Thinking via Reinforcement Learning.pdf
paper_sha256: 0f08fb26acc3d887a1784c1f3bfec57ca89f2f2083c96e777961ceaa6992d650
processed_at: '2026-08-06T02:12:34-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Parallel-R1 大白话版

Andrej,好嘞,我用更接地气的方式重新讲一遍这篇paper。

## 一句话总结

**教LLM在thinking过程中学会"开多个脑洞同时想",而且是intrinsic的能力,靠RL训练出来的。**

---

## 为什么这事有意思

想象你在做一道数学题,卡住了。你不会傻乎乎地一条路走到黑,你脑子里会同时冒出几个想法:"要不试试代数法?""要不用几何法?""要不反过来推?" 然后这几个idea各自往下走一段,你再看看哪个靠谱,合一下继续做。

这就是parallel thinking。

Google的Gemini在IMO拿金牌就靠这招 ([Gemini blog](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/))。

问题来了:**怎么让LLM天生就会这招?**

---

## 现有方法的坑

### SFT方法的问题

之前的人([Multiverse](https://arxiv.org/abs/2506.09991)那拨)是这么干的:先用某个强模型生成一堆"parallel thinking"的解题过程,然后让小模型模仿(SFT)。

问题是这就像让一个学生背标准答案——他只是在**模仿表面pattern**,没有真正学会**什么时候、为什么**要开多个脑洞。一旦换个题型,立马歇菜。而且生成那种训练数据成本极高。

### Test-time方法的问题

像[Tree of Thoughts](https://arxiv.org/abs/2305.10601)、[Self-Consistency](https://arxiv.org/abs/2203.11171)这些,都是在推理时搞并行,开销巨大,而且分支点和合并点都是**写死的规则**,不adaptive。

---

## 这篇paper的核心idea

**能不能用RL让模型自己学会什么时候开脑洞、怎么合并脑洞?**

这听起来简单,实际有几个大坑:

### 坑1: Cold-start问题

LLM在pre-training里从没见过`<Parallel>`这种tag。你直接上RL,模型根本生成不出这种trajectory来学习。就像让一个从没见过游泳的人,通过"游得好给奖励"来学游泳——他连水都没下过,怎么开始?

### 坑2: Reward设计两难

- 你只用"答案对不对"做reward → 模型会take shortcut,直接走单线把题做对就完事,根本不用parallel thinking
- 你强制"必须用parallel thinking" → 模型会在简单题上也硬开脑洞,纯属浪费

### 坑3: 这玩意是个黑盒

就算学会了,模型脑子里到底怎么用parallel thinking的?战略是什么?完全不清楚。

---

## 关键发现1: 简单题能生成,难题完全不行

这是整个paper最关键的empirical observation (Table 1):

| 数据集 | 用prompt让DeepSeek-R1-0528-Qwen-3-8B生成parallel thinking | 成功率 |
|--------|-------------------------------------------------------------|--------|
| GSM8K (简单) | 同样的prompt和sampling | **83.7%** |
| DAPO (难) | 同样的prompt和sampling | **0.0%** |

**同一个模型,同样的prompt,简单题83%能生成valid的parallel thinking格式,难题一个都生成不出来。**

这个发现超级重要,它直接启发了整个训练策略:

**既然简单题能生成parallel thinking data,那就先用简单题的data教会模型格式,然后用RL把这种能力generalize到难题上。**

这绕开了"在难题上合成parallel thinking data"这个几乎不可能的任务。

---

## 训练流程: 三段式

### Stage 0: Cold-Start SFT (教格式)

用GSM8K (7k道简单数学题),让DeepSeek-R1生成parallel thinking过程,提取出来做SFT。

**目标非常明确: 只教格式,不教解题**。模型学会在什么时候吐出 `<Parallel>`, `<Path>`, `<Summary>` 这些tag就够了。

- Batch size: 128
- LR: 1e-5
- Weight decay: 0.01
- Warm-up ratio: 0.1, cosine schedule
- Steps: 58 (Seen) / 230 (Unseen)

### Stage 1: RL on Easy Math (稳定格式)

光SFT还不够,因为 `<Parallel>` 这种token在pre-training里从没出现过,模型会生成但**不稳定**。

用GRPO在GSM8K上做小规模RL,把格式彻底稳定住。

**Reward公式**: $R_{final} = R_{\langle Parallel \rangle} \times \bar{R}_{acc}$

- $R_{\langle Parallel \rangle}$: 有没有parallel thinking unit (有=1, 无=0)
- $\bar{R}_{acc}$: group内平均accuracy
- 这是**乘法**,意味着两者必须同时满足
- 具体是binary strict: 有parallel block AND 答案对 → +1,否则-1

- Batch size: 1024, 5 rollouts, LR 1e-6, 5 epochs, 35 steps

### Stage 2: RL on Hard Math (真正学解题)

在DAPO上用GRPO,只用accuracy reward,让模型在难题上学会**什么时候该用parallel thinking**。

**关键: 这里只用accuracy reward,不用parallel reward**。因为primary goal是解题性能,让模型自己决定要不要开脑洞。

- Batch size: 512, 8 rollouts, LR 1e-6, 300 steps

---

## GRPO算法到底在干啥

GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300))是DeepSeek搞的RL算法,paper里有公式,我拆开讲:

### Importance Ratio

$$\rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$$

- $\pi_\theta$: 当前policy (正在update的model)
- $\pi_{\theta_{old}}$: 采样时的old policy
- $o_i$: 第i个采样response
- $q$: question
- $\rho_i$: 新policy对旧policy的probability ratio,>1说明新policy更喜欢这个response

### Advantage (group-normalized)

$$A_i = \frac{r_i - \bar{r}}{\sqrt{\frac{1}{G}\sum_{j=1}^G(r_j - \bar{r})^2} + \varepsilon_{stab}}$$

- $r_i$: 第i个response的reward
- $\bar{r} = \frac{1}{G}\sum_{j=1}^G r_j$: G个response的平均reward
- 分母: group的标准差 + $\varepsilon_{stab}$ (防除零)
- 意思: 这个response相对于同组其他response,好多少或差多少,做normalization

### Loss

$$\mathcal{L}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min(\rho_i A_i, \text{clip}(\rho_i, 1-\alpha, 1+\alpha)A_i) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

- $\min(\rho_i A_i, \text{clip}(\cdot))$: PPO的clipped surrogate,$\alpha$限制ratio波动幅度
- $\beta D_{KL}(\pi_\theta \| \pi_{ref})$: KL penalty to reference policy,防止模型跑太远
- $\pi_{ref}$: 参考policy (通常是SFT后的model)

**直觉**: 对每个question采样G个answer,看哪些reward高哪些低,高的就增加其概率,低的就降低,用group内比较来减少variance。

---

## 两个模型variant

### Parallel-Seen (Causal model)

不改动模型架构,就是标准causal LLM。不同的path之间可能有information leakage,但RL训练更stable。

### Parallel-Unseen (Structured model)

改attention mask和position encoding,强制不同path之间**物理隔离**:

**Path-window masking**: `<Path>` block内的token只能attend同一path的token和shared context,不能看其他path

**Multiverse position encoding**: 给每个path分配不同的position index区间,避免位置重叠导致混淆

但`<Summary>` block能看到所有path,这样才能aggregate insights。

### 为什么Unseen要不同recipe

实验发现attention mask从简单题学到难题**transfer不好**。所以Unseen去掉了Stage 1 RL,改用alternating reward。

---

## Reward设计的ablation (Table 4)

这个ablation非常informative:

| Reward策略 | Parallel Ratio | AIME25 | AMC23 | MATH |
|-----------|----------------|--------|-------|------|
| Accuracy only | 13.6% | 17.7 | 69.7 | 82.6 |
| Parallel only | 80.3% | 17.7 | 59.4 | 81.7 |
| **Alternating** | 63.0% | **19.0** | 67.5 | **84.5** |

### 三个takeaway

1. **只reward accuracy**: 模型压根不用parallel thinking (13.6%),虽然某些benchmark还行,但没学会新技能
2. **只reward parallel**: 模型疯狂开脑洞 (80.3%),但为了开脑洞而开脑洞,正确率反而掉 (AMC从69.7掉到59.4)
3. **Alternating (S2)**: 80% steps用accuracy reward,20% steps用tiered reward (有parallel+对=1.2, 无parallel+对=1.0, 其他=-1.0),balance得最好

**直觉**: 你不能完全不管parallel usage (模型会偷懒),也不能硬逼着用 (模型会form over substance)。交替reward就像"偶尔提醒一下要用新技能,但大部分时间还是看结果"。

---

## 最有意思的发现: 模型行为的evolution

### Finding 2 (Figure 3)

**测量方法**: 统计 `<Parallel>` block在整个response中的相对位置 (starting token index / total length)。

**发现**: 随着RL训练进行,`<Parallel>` block的位置**稳步后移**——从reasoning chain早期移到末尾。

### 这意味着什么

**训练早期**: 模型reasoning能力弱,在reasoning chain早期就开parallel block,用来探索不同的解题路径。这是high-variance的exploration strategy。

**训练后期**: 模型reasoning能力变强,它发现早期开parallel反而容易引入错误。于是它改成: 先用单条high-confidence路径把题做出来,然后在最后开parallel block做**verification**——从多个角度验证答案对不对。

**为什么会这样**: 因为reward是final-answer-dominated的。模型为了maximize reward,会采取risk-averse策略。早期exploration风险高,后期verification几乎零风险,只增不减。

### Case study对比

**Figure 5 (早期模型)**: 一道进制转换题,模型刚把base转换完就开 `<Parallel>`,两条path分别用factor theorem和base arithmetic去探索解法。

**Figure 6 (后期模型)**: 同一道题,模型先用remainder theorem一条道走到黑算出答案,然后才在最后开 `<Parallel>` 做verification,两条path只是确认一下"divisors和base validity"。

---

## 最exciting的发现: Parallel thinking作为mid-training scaffold

### Finding 3 (Figure 4)

**Hypothesis**: parallel thinking本身是一种structured exploration mechanism,可以帮RL training escape local optima。

### 实验设计

**Stage-1 (step 0-200)**: 用alternating ACC/PAR reward,强制模型维持high parallel ratio,做broad exploration。相当于"到处乱试,广撒网"。

**Stage-2 (step 200+)**: 切换到accuracy-only reward,让模型exploit Stage-1发现的好策略。parallel ratio自然下降。

### 结果

- **AIME25 peak accuracy: 25.6%**,超过baseline GRPO model
- **关键**: performance提升发生在parallel ratio**下降**的时候

### 这说明什么

Parallel thinking的价值有两层:
1. 作为reasoning structure本身有用 (这已经超过baseline)
2. **更重要的是**: 它作为exploration scaffold,帮模型在policy space里找到一个更好的region。一旦找到了,模型可以drop掉parallel structure,用更简洁的单线reasoning也能保持高性能

**类比**: 就像学骑自行车用辅助轮。辅助轮 (parallel thinking) 帮你学会平衡和蹬踏的feel,但学会之后你可以拆掉辅助轮,骑得反而更快更灵活。

---

## 主实验结果 (Table 2)

| Method | AIME25 | AIME24 | AMC23 | MATH | Avg |
|--------|--------|--------|-------|------|-----|
| Qwen3-4B-Base | 1.3 | 2.9 | 8.1 | 13.9 | 6.6 |
| SFT only | 8.0 | 10.6 | 48.9 | 76.6 | 36.0 |
| GRPO (标准RL) | 14.8 | 18.5 | 63.6 | 83.5 | 45.1 |
| **Parallel-R1-Seen** | **19.2** | 19.4 | **70.5** | **86.7** | **48.9** |
| Parallel-R1-Unseen (S2) | 19.0 | 16.3 | 67.5 | 84.5 | 46.8 |

几个关键点:

1. **SFT only (36.0) 远不如标准GRPO (45.1)**: 证明光模仿parallel thinking format没用,必须靠RL探索
2. **Parallel-R1-Seen比标准GRPO提升8.4%**: 这是核心gain,证明parallel thinking确实有用
3. **Structured variant (Unseen) 略差**: 说明explicit architecture modification可能detrimental,至少在RL training下
4. **AIME25上15.4% → 25.6%**: 用mid-training scaffold trick后,从15.4%飙升到25.6% (+42.9%)

---

## Ablation: Stage 1 RL到底有没有用 (Table 3)

| Config | AIME25 | AIME24 | AMC23 | MATH | Avg |
|--------|--------|--------|-------|------|-----|
| Parallel-R1-Seen (full) | 19.2 | 19.4 | 70.5 | 86.7 | 48.9 |
| - w/o RL on GSM8K | 17.9 | 19.0 | 65.0 | 84.5 | 46.6 |

Causal model去掉Stage 1 (GSM8K上的RL),掉了2.3%。

| Config | AIME25 | AIME24 | AMC23 | MATH | Avg |
|--------|--------|--------|-------|------|-----|
| Parallel-R1-Unseen (S1) | 17.7 | 18.3 | 69.7 | 82.6 | 47.1 |
| + with RL on GSM8K | 14.4 | 12.9 | 52.3 | 74.4 | 38.5 |

Structured model**反而**加了Stage 1 RL后暴跌8.6%!

**解释**: Structured model的attention mask在GSM8K上学的pattern transfer不到DAPO的难题上,导致overfit到superficial pattern。

这呼应了[Yang et al., 2025c](https://arxiv.org/abs/2503.06749)的发现: structured attention mask的generalization是问题。

---

## 我的intuition和思考

### 为什么progressive curriculum work

核心在于**解耦**。传统方法想一步到位: 既教格式,又教解题,又教什么时候用。这太难了,模型在cold-start时既没格式也没解题能力。

Parallel-R1把它拆开:
- Stage 0: 只教格式 (在简单题上)
- Stage 1: 稳定格式 (还是在简单题上)
- Stage 2: 只教解题 (在难题上,格式已经会了)

每个stage的learning burden都小,所以能work。

### Behavior evolution是feature还是bug

这个问题很深。我认为**在outcome-based reward下,这是必然的adaptation**:

- Early stage: 模型弱,需要parallel探索才能偶尔做对 → exploration
- Late stage: 模型强了,单线就能做对,parallel反而引入风险 → verification

这就像一个新手厨师做饭会同时开几个炉子试不同做法,但大厨一道菜闭着眼都能做,不需要并行,最多最后尝一口确认下 (verification)。

**如果想要maintain exploration behavior**,可能需要:
1. Process-based reward (reward中间步骤,不只看最终答案)
2. Explicit exploration bonus (像count-based exploration in RL)
3. Curriculum design让模型始终面对有挑战的task

### Mid-training scaffold的deep implication

这个发现最让我兴奋。它暗示:

**Parallel thinking不仅是reasoning capability,更是一种exploration mechanism**。

这和RL里的几个经典idea都有关联:
- **Entropy regularization**: 强制high entropy = 鼓励diverse exploration,parallel thinking是structured version
- **Curiosity-driven exploration**: parallel paths之间的disagreement可以作为intrinsic reward
- **Population-based training**: 多个path相当于population members

更进一步想,**任何structured exploration mechanism都可能作为mid-training scaffold**:
- Tree search
- Counterfactual reasoning
- Hypothetical scenarios ("what if...")

这paper打开了"用structured exploration作为RL training scaffold"这扇门,后续可能有很多follow-up。

### Structured vs Causal的paradox

理论上structured model (Unseen)应该更好,因为:
- 路径隔离避免信息泄露
- Position encoding不重叠避免混淆
- 更clean的inductive bias

但实验结果Unseen < Seen。我觉得原因可能是:

1. **Gradient interference**: Path isolation导致不同path的gradient无法互相regularize
2. **Generalization问题**: Attention mask在easy→hard的distribution shift上不transfer
3. **Exploration受限**: Causal model的"软隔离"反而允许更多cross-path learning,某种意义上是soft ensemble

### 和test-time scaling的关系

Parallel thinking作为intrinsic capability vs test-time methods (CoT, ToT, Self-consistency),关系是:

- **Intrinsic parallel thinking**: 训练时学会,推理时自动用,开销小,但受限于训练时见过的pattern
- **Test-time methods**: 灵活但开销大,依赖hand-crafted rules

最理想可能是两者combine: model有intrinsic parallel thinking能力,test-time再放大它 (像self-consistency over parallel paths)。

---

## Web Links汇总

- **Paper GitHub**: [https://github.com/zhengkid/Parallel-R1](https://github.com/zhengkid/Parallel-R1)
- **Gemini IMO blog**: [Gemini Deep Think](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/)
- **GRPO (DeepSeekMath)**: [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
- **Multiverse (prior SFT work)**: [https://arxiv.org/abs/2506.09991](https://arxiv.org/abs/2506.09991)
- **Tree of Thoughts**: [https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)
- **Self-Consistency**: [https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
- **Countdown task (prior RL on toy)**: [https://arxiv.org/abs/2504.15466](https://arxiv.org/abs/2504.15466)
- **DAPO**: [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)
- **DeepSeek-R1**: [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- **Qwen3 Technical Report**: [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
- **VERL codebase**: [https://arxiv.org/abs/2409.19256](https://arxiv.org/abs/2409.19256)

---

## 最终思考

这篇paper我最看重的三点:

1. **Key Finding 1的实用性**: "简单题能生成parallel thinking data,难题不行" 这个observation太有价值了,它直接告诉你cold-start该怎么做

2. **Behavior evolution的洞察**: 从exploration到verification的shift,揭示了outcome-based reward的深层影响。这对你做RLVR的人应该很有共鸣——reward design不只影响performance,还shape model的strategic behavior

3. **Mid-training scaffold的概念**: 这可能是paper里最novel的contribution。它把parallel thinking从"最终能力"重新定位为"训练时的探索工具"。一旦学到了好的policy region,可以drop掉这个scaffold。这就像你之前讲过的"learning to learn"的思路

Andrej,你觉得如果把这个mid-training scaffold idea用到code generation或scientific reasoning上,会有什么结果? 特别是code,因为code有natural verification signal (能不能跑通)。

---

# Parallel-R1 深度解析:通过RL让LLM学会Parallel Thinking

Andrej, 这篇paper非常有意思,它试图解决一个根本性问题:**如何通过训练(而非test-time tricks)让LLM intrinsic地具备parallel thinking能力**。让我详细拆解。

## 1. 核心Motivation与Problem Framing

### 1.1 为什么Parallel Thinking重要

Google的Gemini在IMO上的成功部分归功于parallel thinking ([Luong & Lockhart, 2025](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/))。从cognitive science角度,人类在面对uncertainty时会同时考虑multiple possibilities再synthesize,这避免了premature lock-in到suboptimal solution ([Clark, 1989](https://mitpress.mit.edu/9780262530736/microcognition/); [Jackendoff, 2011](https://doi.org/10.1515/9783110245749.17))。

### 1.2 现有方法的局限

**SFT-based方法** ([Yang et al., 2025b - Multiverse](https://arxiv.org/abs/2506.09991); [Macfarlane et al., 2025](https://openreview.net/forum?id=a3o4b3hkwp); [Chen et al., 2025a](https://arxiv.org/abs/2508.08895))本质上是behavioral cloning on pre-generated trajectories:
- 依赖costly data pipeline合成parallel thinking data
- 导致superficial pattern matching而非deep reasoning skill
- Generalization能力severely limited

**Test-time methods** ([Yao et al., 2023 - Tree of Thoughts](https://arxiv.org/abs/2305.10601); [Wang et al., 2022 - Self-Consistency](https://arxiv.org/abs/2203.11171); [Brown et al., 2024](https://arxiv.org/abs/2407.21787)):
- Branching/aggregating points由pre-defined schedule决定,lacks adaptivity
- 或依赖hand-crafted heuristics和external verifiers
- 推理开销高,且非intrinsic capability

### 1.3 RL的核心挑战

1. **Cold-start problem**: LLM在pre-training和SFT中没见过parallel thinking行为,无法在RL exploration中生成这样的trajectories来学习
2. **Reward design dilemma**:
   - 仅用final correctness → 模型take shortcuts,forget parallel thinking
   - 强制parallel thinking → 模型在unnecessary场景也用
3. **Black box**: 即使学会了,strategic role和mechanisms不清楚

---

## 2. Parallel Thinking的Formulation

### 2.1 行为定义

Paper将parallel thinking formalize为两阶段循环:

**Exploration阶段**: 模型检测到critical step时,suspend主chain,launch N个independent trajectories同时探索

**Summary阶段**: 聚合outcomes,distill key insights,resolve conflicts,得到most promising conclusion,然后resume主reasoning chain

### 2.2 Control Tags

引入三个control tags:
- `<Parallel>...</Parallel>`: 标记exploration phase
- `<Path>...</Path>`: 隔离reasoning threads
- `<Summary>...</Summary>`: 总结parallel thinking

### 2.3 Inference Workflow

```
Main reasoning chain → <Parallel> token → 
spawn multiple <Path> blocks concurrently → 
<Summary> block integrates insights → 
resume main reasoning with summarized conclusion
```

这个过程可以repeat多次,实现adaptive dynamic parallel inference。

---

## 3. Data Pipeline:一个Key Finding

### 3.1 Finding 1 (Table 1)

这是一个非常striking的发现:

| Data | # Samples | Parallel Thinking Format (%) |
|------|-----------|------------------------------|
| GSM8K | 7472 | 83.7 |
| DAPO | 17916 | 0.0 |

**同一个model (DeepSeek-R1-0528-Qwen-3-8B)在identical prompts和sampling settings下**:
- 简单的GSM8K: 83.7%能生成valid parallel-thinking traces
- 难的DAPO: 0.0%成功率

### 3.2 这个发现的战略意义

这启发了一个关键策略:**用简单task的parallel thinking data做cold-start,只teach format,然后用RL generalize到难task**。

这bypass了在challenging problems上生成high-quality parallel thinking data的困难(这正是prior SFT方法如[Multiverse](https://arxiv.org/abs/2506.09991)的bottleneck)。

### 3.3 Format Check (Algorithm 1)

用stack-based check验证tag pairs的well-formedness:
```
for each token t:
    if t is opening tag: push to stack S
    elif t is closing tag: 
        if S empty or top mismatch: invalid
        else: pop S
return valid if S empty
```

这对structured model variant尤其重要,因为path-window attention mask需要strict format adherence。

---

## 4. RL训练框架

### 4.1 GRPO算法

Paper使用[Group Relative Policy Optimization (GRPO)](https://arxiv.org/abs/2402.03300)作为RL算法。

**变量定义**:
- $q$: question
- $\{o_i\}_{i=1}^G$: G个candidate responses从old policy $\pi_{\theta_{old}}(\cdot|q)$采样
- $r_i$: response $o_i$的reward

**Importance ratio**:
$$\rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$$

这里 $\rho_i$ 衡量new policy相对于old policy对 $o_i$ 的likelihood ratio。

**Group baseline**:
$$\bar{r} = \frac{1}{G}\sum_{j=1}^G r_j$$

这是group mean reward,作为baseline。

**Advantage**:
$$A_i = \frac{r_i - \bar{r}}{\sqrt{\frac{1}{G}\sum_{j=1}^G(r_j - \bar{r})^2} + \varepsilon_{stab}}$$

- 分子 $r_i - \bar{r}$: response相对于group mean的relative performance
- 分母: group standard deviation + $\varepsilon_{stab}$(numerical stability constant)
- 这做了group-wise normalization

**GRPO Loss**:
$$\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{\{o_i\} \sim \pi_{\theta_{old}}}\left[\frac{1}{G}\sum_{i=1}^G \min(\rho_i A_i, \text{clip}(\rho_i, 1-\alpha, 1+\alpha)A_i) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

- $\min(\rho_i A_i, \text{clip}(\cdot))$: PPO-style clipped surrogate objective,$\alpha$是clip ratio
- $\beta D_{KL}(\pi_\theta \| \pi_{ref})$: KL penalty to reference policy $\pi_{ref}$,防止deviation过大
- $\beta$: KL penalty coefficient

### 4.2 Progressive Training Recipe (Causal Model - Parallel-Seen)

**三阶段curriculum**:

#### Stage 0: Cold-Start SFT
- Dataset: Parallel-GSM8K (7k samples)
- 用distilled Qwen3-8B生成,只extract non-thinking parts (final short CoT)作为gold annotations
- 目标: teach basic format of parallel thinking
- Hyperparameters: batch size 128, lr 1e-5, weight decay 0.01, warm-up ratio 0.1, cosine schedule

#### Stage 1: RL on Easy Math (GSM8K)
- 在cold-start SFT后,model能生成tags但behavior不稳定(因为这些special tokens从没在pre-training中出现)
- 用GRPO on GSM8K做small-scale RL增强format learning
- **Reward**: $R_{final} = R_{\langle Parallel \rangle} \times \bar{R}_{acc}$
  - $R_{\langle Parallel \rangle}$: Parallel Reward,incentivize parallel reasoning paths
  - $R_{acc}$: Accuracy Reward,evaluate final response correctness
  - $\bar{R}_{acc}$: group mean accuracy
- **Binary strict**: +1 if (至少一个parallel thinking unit AND 答案正确),否则-1
- Hyperparameters: batch size 1024, 5 rollouts, lr 1e-6, 5 epochs, 35 gradient steps

#### Stage 2: RL on General Math (DAPO)
- Generalize parallel thinking ability到challenging tasks
- 用GRPO with **accuracy reward $R_{acc}$ only**
- 不再用parallel reward因为primary goal是improve task performance
- Dataset: [DAPO training set](https://arxiv.org/abs/2503.14476)
- Hyperparameters: batch size 512, 8 rollouts, lr 1e-6, 300 gradient steps

---

## 5. Structured Model Variant (Parallel-Unseen)

### 5.1 Motivation

Causal variant (Parallel-Seen)不显式isolate reasoning paths,导致:
- Hidden representations从一个path leak到另一个path
- Gradients across paths互相interfere

### 5.2 Structured Attention Mechanism (Figure 2)

**Path-window masking**:
- 每个`<Path>` block内的token只能attend:
  - 同一path内的tokens
  - Shared context (parallel block之前的main reasoning)
- 防止cross-path information leakage

**Multiverse position encodings**:
- 给每个path分配disjoint set of position indices
- 确保positional embedding space不overlap
- 如果position overlap,模型会confuse不同path的内容

**Summary block的visibility**:
- 保留从shared `<Summary>` block到所有paths的visibility
- 这对integrating insights across paths至关重要

### 5.3 为什么Structured Model需要不同Recipe

**Key insight**: attention mask从easy math到hard math的generalization差([Yang et al., 2025c](https://arxiv.org/abs/2503.06749)有类似发现)。

直接用Parallel-Seen的progressive recipe到structured variant prove ineffective,所以:
- **Remove Stage 1 RL** (避免在GSM8K上overfit到superficial patterns)
- **Redesign reward schedule**

### 5.4 Reward Schemes for Parallel-Unseen

**S1: Accuracy-only**
- 只optimize task correctness
- 移除parallel usage的direct incentives

**S2: Alternating accuracy and parallel** (推荐)
- 在fixed windows $W=10$ steps内交替:
  - **80% steps**: standard accuracy-only reward $R_{acc}$
  - **20% steps**: tiered reward system:
    - **+1.2**: 有parallel thinking unit AND 答案正确
    - **+1.0**: 没parallel thinking unit 但答案正确
    - **-1.0**: 其他(包括错误答案)

这种schedule reintroduce了calibrated incentive for parallel usage,但不让它dominate training。

---

## 6. 实验结果分析

### 6.1 Main Results (Table 2)

| Method | # Parallel | AIME25 Mean@16 | AIME24 Mean@16 | AMC23 Mean@16 | MATH Mean@1 | Avg |
|--------|------------|----------------|----------------|---------------|--------------|-----|
| Qwen3-4B-Base | 0.0 | 1.3 | 2.9 | 8.1 | 13.9 | 6.6 |
| Parallel-SFT-Seen | 95.6 | 8.0 | 10.6 | 48.9 | 76.6 | 36.0 |
| GRPO (DAPO) | 0.0 | 14.8 | 18.5 | 63.6 | 83.5 | 45.1 |
| **Parallel-R1-Seen** | 27.3 | **19.2** | 19.4 | **70.5** | **86.7** | **48.9** |
| Parallel-R1-Unseen (S2) | 63.0 | 19.0 | 16.3 | 67.5 | 84.5 | 46.8 |

**关键观察**:
1. Parallel-R1-Seen比GRPO baseline提升 **8.4%** average (48.9 vs 45.1)
2. SFT alone (Parallel-SFT-Seen 36.0)远不如GRPO (45.1),证明SFT insufficient for advanced reasoning
3. Naive额外RL on easier data只marginal benefit (45.3 vs 45.1)
4. **Structured variant (Unseen)性能稍差**,说明explicit architectural modifications可能detrimental to RL training
5. Parallel ratio: Seen 27.3% vs Unseen S2 63.0% - Unseen更frequently使用parallel thinking

### 6.2 Ablation: Training Stages (Table 3)

**Causal variant**:
- Parallel-R1-Seen: Avg 48.9
- w/o RL on GSM8K: Avg 46.6 (-2.3%)
- → Stage 1 RL crucial for bootstrap adaptive parallel thinking

**Structured variant** (相反趋势!):
- Parallel-R1-Unseen (S1): Avg 47.1
- + with RL on GSM8K: Avg 38.5 (-8.6%!)
- → Attention mask learned on easy math不transfer到hard math distribution

### 6.3 Ablation: Reward Modeling (Table 4)

| Training Config | Parallel Ratio | AIME25 | AIME24 | AMC23 | MATH |
|------------------|----------------|--------|--------|-------|------|
| Accuracy | 13.6 | 17.7 | 18.3 | **69.7** | **82.6** |
| Parallel | 80.3 | 17.7 | 15.2 | 59.4 | 81.7 |
| **Alternating Acc./Parallel** | 63.0 | **19.0** | 16.3 | 67.5 | 84.5 |

**三个critical findings**:

1. **Accuracy-only insufficient**: parallel ratio 13.6,模型不naturally adopt parallel reasoning

2. **Parallel-only detrimental**: parallel ratio 80.3但性能drop (AMC23 59.4 vs 69.7),模型prioritize structural form over logical correctness

3. **Alternating achieves balance**: parallel ratio 63.0,AIME25甚至超过accuracy-only (19.0 vs 17.7)

---

## 7. 最重要的发现:Behavior Evolution

### 7.1 Finding 2 (Figure 3)

**测量方法**: 用 `<Parallel>` block的starting token index除以total sequence length得到relative position。

**观察**: 随RL训练进行,`<Parallel>` block的average relative position steadily increases。

**Interpretation**: 
- **Early stage**: parallel thinking用于**computational exploration** (在reasoning chain早期)
  - 模型reasoning能力弱,需要high-variance strategy discover potential solution
- **Late stage**: parallel thinking用于**multi-perspective verification** (在reasoning chain末尾)
  - 模型reasoning能力improve,early exploration成为liability (可能引入错误)
  - 学到risk-averse strategy:先用single high-confidence path得到answer,再用parallel做verification

这是**final-answer-dominated reward design**直接shape的行为。

### 7.2 Case Studies对比

**Figure 5 (Early-stage model)**: 
- 在reasoning早期(刚转换base后)就插入`<Parallel>`
- 两个path分别用factor theorem和base arithmetic
- 这是exploration-oriented strategy

**Figure 6 (Late-stage model)**:
- 先用single path完成整个求解(remainder theorem)
- 得到答案后才插入`<Parallel>`做verification
- 两个path只是confirm divisors和base validity
- 这是verification-oriented strategy

---

## 8. 最Exciting的发现:Mid-Training Exploration Scaffold

### 8.1 Finding 3 (Figure 4)

**Hypothesis**: parallel thinking本身可以作为structured exploration mechanism改善RL training。

**RL的经典挑战**: ensuring sufficient exploration to avoid local optima。

**Key insight**: 强制模型在critical step生成multiple parallel thought blocks = injecting strong inductive bias for structured, diverse exploration。

### 8.2 Two-Stage Training Curriculum

**Stage-1 (Exploration Phase, steps 0-200)**:
- 用Parallel-R1-Unseen (S2)的训练方法
- Alternating ACC/PAR reward
- 维持high parallel ratio,强制broad exploration
- 绿色dashed line显示parallel ratio保持高位

**Stage-2 (Exploitation Phase, after step 200)**:
- 切换到accuracy-only reward
- 从exploration转向exploitation
- Refine和exploit Stage-1发现的有效策略

### 8.3 结果

- Stage-2进入后,performance (red line)继续improve
- **Peak AIME25 accuracy: 25.6%**,超过Baseline GRPO model
- **Critical**: performance gain发生在parallel ratio **decrease**的同时

**Interpretation**: parallel thinking的价值:
1. 不仅是parallel structure本身的有效性
2. 更重要的是它通过exploration帮助discover的robust policy space

Initial forced exploration = scaffold,引导model到policy space的更effective region,然后learn final policy。

这让我联想到:
- **Curriculum learning**: 先学简单format再generalize
- **Self-play / opponent modeling**: 多个path可看作weak opponents
- **MCTS的learned version**: 这里是learned而非hand-crafted
- **Entropy regularization**: parallel thinking注入diversity

---

## 9. 我的Intuition和思考

### 9.1 为什么这个方法work

1. **Curriculum design的优雅**: 
   - 不试图直接在难task上学parallel thinking (cold-start会失败)
   - 而是先在easy task学format,再RL generalize
   - 这解耦了format learning和reasoning skill learning

2. **Reward design的tension**:
   - Multiplicative reward ($R_{\langle Parallel \rangle} \times \bar{R}_{acc}$)的strictness让模型必须combine两种能力
   - Alternating reward避免single reward type的failure mode

3. **Behavior evolution的必然性**:
   - Final-answer-dominated reward自然drive risk-averse behavior
   - 从exploration到verification的shift是rational adaptation

### 9.2 Open Questions

1. **Verification vs Exploration的trade-off**: 
   - Late-stage verification strategy是否真的是optimal?
   - 如果想要真正的parallel exploration,需要什么reward design?
   - 可能需要process-based reward而非outcome-based

2. **Structured vs Causal的paradox**:
   - Structured model理论上有更好的inductive bias但实际性能差
   - 这可能因为attention mask的generalization问题
   - 或因为explicit isolation限制了cross-path learning的某些benefits

3. **Mid-training scaffold的generalization**:
   - 这个idea能否extend到其他domains (code, science)?
   - 最优的Stage-1/Stage-2切换点在哪?
   - 是否可以用其他structured exploration替代parallel thinking?

4. **与Test-time scaling的关系**:
   - Parallel thinking作为intrinsic capability vs test-time methods
   - 这两者能否combine? 比如intrinsic parallel thinking + test-time self-consistency

### 9.3 与其他工作的联系

- **[Multiverse (Yang et al., 2025b)](https://arxiv.org/abs/2506.09991)**: SFT-based parallel thinking,这篇paper的RL方法更generic
- **[Countdown task (Pan et al., 2025)](https://arxiv.org/abs/2504.15466)**: RL on toy task,这篇extend到real-world math
- **[Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)**: Test-time method with hand-crafted heuristics
- **[DeepSeek-R1 (Guo et al., 2025)](https://arxiv.org/abs/2501.12948)**: RLVR foundation
- **[DAPO (Yu et al., 2025)](https://arxiv.org/abs/2503.14476)**: Training dataset和RL system

---

## 10. 代码和资源

- **GitHub**: [https://github.com/zhengkid/Parallel-R1](https://github.com/zhengkid/Parallel-R1)
- **Codebase**: Adapted from [VERL](https://arxiv.org/abs/2409.19256)
- **Backbone**: Qwen-3-4B-Base ([Yang et al., 2025a](https://arxiv.org/abs/2505.09388))
- **Cold-start model**: DeepSeek-R1-0528-Qwen-3-8B

---

## 总结

这篇paper的核心贡献:

1. **第一个RL framework**让LLM从scratch学会parallel thinking on real-world math tasks
2. **Progressive curriculum**解耦format learning和reasoning skill learning
3. **Alternating reward**平衡parallel usage和performance
4. **Behavior evolution分析**揭示exploration→verification的strategic shift
5. **Mid-training scaffold概念**用parallel thinking做structured exploration提升RL性能

最让我excited的是Finding 3的mid-training scaffold idea - 这暗示parallel thinking不仅是最终的reasoning capability,更是一种powerful的exploration mechanism,可以用来escape RL training的local optima。这个idea可能有更broad的implications。

Andrej,你觉得这种behavior evolution (exploration→verification)是feature还是bug? 如果是bug,什么样的reward design能maintain exploration behavior?
