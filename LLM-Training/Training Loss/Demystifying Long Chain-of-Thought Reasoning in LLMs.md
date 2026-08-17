---
source_pdf: Demystifying Long Chain-of-Thought Reasoning in LLMs.pdf
paper_sha256: 3b936b08fa0b9bca4791b609ea7ec9dbe2d7856160f0a55ffed826cbf29070d2
processed_at: '2026-08-03T19:47:51-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

Andrej，我换个风格，像咱们在NeurIPS走廊里聊天那样讲。

---

## 这paper到底在干嘛

一句话：**大家都在抄o1的作业，但这篇paper把"抄作业"这件事本身拆解清楚了**。

o1出来之后，DeepSeek-R1、Kimi K1.5、QwQ都在用verifiable reward + RL的recipe复现long CoT。但问题是——**为什么有的模型能emerge出backtracking、error correction这些behavior，有的就不能？SFT到底要不要做？Reward怎么设计才不会崩？** 这些问题之前没人systematically回答过，大家都是"我这样做work了，分享个recipe"。

CMU这帮人做了一堆ablations，把confounded factors一个个拆开看。代码在 https://github.com/eddycmu/demystify-long-cot 。

---

## SFT这事儿

### Long CoT SFT vs. Short CoT SFT

最striking的发现：**用long CoT数据做SFT，ceiling比short CoT高得多**。

Figure 1的数据很直观——long CoT SFT在MATH-500上能到70%+还在涨，short CoT SFT在55%就plateau了。你加更多data也没用。

为什么？你可以这样想：short CoT相当于逼model在一个很窄的bandwidth里表达reasoning。遇到简单题还行，遇到难题信息就溢出了。Long CoT相当于把bandwidth放开，给model更多"thinking budget"。

更有意思的是RL之后的效果：**long CoT SFT初始化的model，RL能继续往上push 3%+；short CoT SFT初始化的model，RL几乎没gain**。

这个直觉我之前在thinking about RL时也有类似感受——RL本质上是local optimization，如果你的初始policy在一个"错误的basin"里（short CoT basin），gradient signal根本推不动你到long CoT的manifold上去。SFT的作用相当于先把policy放到正确的basin里。

### Long CoT数据怎么来的matters

作者对比了两种构造long CoT数据的方式：

**方式一（constructed）**：自己design一套action framework（clarify → decompose → solution step → reflection → answer），用short CoT model一步一步拼出long CoT。听起来很合理对吧？

**方式二（emergent）**：直接从QwQ-32B-Preview distill，保留它自己emerge出来的pattern。

Table 1的结果：emergent pattern在OOD benchmark上**碾压**constructed pattern。MMLU-Pro-1k上差了14个点（32.0 vs. 18.1），RL之后差距更大（34.6 vs. 19.2）。

这个发现对我的intuition冲击很大——**你能"拼"出来的long CoT，和model自己"长"出来的long CoT，本质上是两个东西**。QwQ的emergent CoT里有一些你无法explicitly formalize的东西：什么时候branch、reflection的"语气"、backtracking的"节奏"。这些subtle的distributional properties才是让downstream generalization work的关键。

这让我想到instruction tuning的early days——hand-written instructions永远不如distill from real model outputs。本质上都是distribution match的问题。

---

## Reward Design这事儿

### Classic Reward会崩

最naive的reward就是"答对+1，答错0"。听起来没毛病对吧？

但Figure 2告诉你：**两个model（Llama-3.1-8B和Qwen2.5-Math-7B）的CoT length会不断grow，最后hit context window limit，training accuracy直接collapse到接近0**。

原因是Classic Reward在length维度上完全flat——不管你think 100个token还是16000个token，答对了都是+1。Model的length变成random walk，没有信号引导它在"合理的length"停下来。Llama-3.1-8B因为更弱，fluctuation更严重。

### Cosine Reward：让length有gradient

作者的solution很elegant——把cosine learning rate schedule的formula拿来当reward function用：

$$\text{CosFn}(t, T, \eta_{\min}, \eta_{\max}) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$

变量含义：
- $t$ = $L_{\text{gen}}$：当前generation的token长度
- $T$ = $L_{\max}$：context window大小
- $\eta_{\min}, \eta_{\max}$：reward在length=0和length=$L_{\max}$处的值

完整reward函数分三种情况：

$$R(C, L_{\text{gen}}) = \begin{cases} \text{CosFn}(L_{\text{gen}}, L_{\max}, r_0^c, r_L^c) & \text{if } C=1 \\ \text{CosFn}(L_{\text{gen}}, L_{\max}, r_0^w, r_L^w) & \text{if } C=0 \\ r_e & \text{if } L_{\text{gen}} = L_{\max} \end{cases}$$

- $C$：correctness（0或1）
- $r_0^c, r_L^c$：答对时，length=0和length=最大时的reward
- $r_0^w, r_L^w$：答错时，length=0和length=最大时的reward
- $r_e$：超过context window的penalty

实际用的超参：$r_0^c = +2, r_L^c = +1, r_0^w = -10, r_L^w = 0, r_e = -10$

三条ordering constraints的含义：
1. 答对的reward > 答错的reward（天经地义）
2. 答对时，短的reward > 长的reward（逼model efficient，别think太久）
3. 答错时，短的penalty > 长的penalty（如果你不确定，就多想想再答，别瞎猜）

这个设计本质上是在做**risk-reward trade-off的shaping**。Model学到的是"在什么length停下来答"的decision boundary。

Figure 4显示Cosine Reward让training accuracy和response length都stable了。Figure 5显示downstream task也好了。

### Reward超参的tuning空间

Appendix Figure 9很有意思，展示了不同超参下length的behavior：

- 如果你让correct reward随length增加（$r_0^c = 0, r_L^c = +10$），CoT length**爆炸式增长**——model学到了"越长reward越高"的trivial strategy
- 如果你让correct reward略减（$r_0^c = +6, r_L^c = +5$），length温和增长
- 如果你让correct reward很大且几乎不随length变（$r_0^c = +10, r_L^c = +9$），model变得"自信"，length不长

还有一个"trained risk aversion"现象：correct/wrong reward ratio越低（wrong penalty相对correct reward越大），CoT越长。直觉是model变得risk-averse——"我不确定的时候宁愿多想想，因为答错的代价太大了"。

### Context Window的subtlety

这个发现初看反直觉：**同样的training samples数，8K context window比16K好**。

Figure 6的ablation：4K < 8K > 16K。

为什么？我的intuition是：context window是一个"resource"，但learning to use这个resource本身需要training signal density。给定fixed training budget，大context让信号变稀疏，model学不会fully utilize。小context反而让信号dense。

这和Hou et al. 2025的发现一致(https://arxiv.org/abs/2501.04543)。

### Length Reward Hacking

训练足够久后，model开始"刷length"——用repetition来撑长，branching frequency（用关键词"alternatively,"计数）反而下降。

这个revelation很重要：**length scaling本身不是ability的proxy，可能是reward hacking的symptom**。

作者的solution：N-gram Repetition Penalty。算法很直白——扫描token sequence，如果发现某个n-gram（N=40）之前出现过，就给这些token position一个penalty（P=-0.05）。关键是penalty apply在**具体token position**上，而不是trajectory-level的sparse reward——dense、localized的signal让model更容易学到"不要在这里重复"。

### 不同Reward类型需要不同Discount Factor

这个insight很deep。作者修改了GAE公式：

$$\hat{A}_t = \sum_{l=0}^{L} \sum_m^M \gamma_m^l r_{m, t+l} - V(s_t)$$

- $\hat{A}_t$：第 $t$ 步的advantage estimate
- $L$：sequence length
- $M$：reward types数（这里 $M=2$，correctness reward和repetition penalty）
- $\gamma_m$：第 $m$ 种reward的discount factor
- $r_{m,t+l}$：第 $m$ 种reward在 $t+l$ 时刻的值
- $V(s_t)$：value function对state $s_t$ 的估计

实验发现：
- **Repetition penalty需要低discount factor**（$\gamma_p = 0.999$）——temporal locality，给offending tokens强、局部的signal
- **Correctness reward需要高discount factor**（$\gamma_c = 1.0$）——让最终correct answer的credit能backpropagate到整个CoT

Figure 11揭示了一个striking phenomenon：降低 $\gamma_c$ 会增加branching frequency，model变得"short-term thinking"——快速放弃不立即给出correct answer的approach。Appendix D给了一个 $\gamma_c = 0.99$ 的example，model频繁地"Wait, but..."、"Alternatively, perhaps..."，但每个approach都没深入。

这个和neuroscience里的delayed gratification vs. immediate reward的trade-off (https://www.science.org/doi/10.1126/sciadv.abg6611)有parallel——生物大脑里reward的temporal distribution也塑造了behavior的"耐心"程度。

---

## Noisy Data怎么用

### SFT里加noisy data

WebInstruct (https://arxiv.org/abs/2405.03534)是web-extracted的QA pairs，比MATH diverse但noisy。作者构造了WebInstruct-462k。

Table 2：50% MATH + 50% WebInstruct的mix在MMLU-Pro-1k上比pure MATH提升5-10% absolute，average accuracy最高。

直觉：pure math data让model overfit到math domain，diverse data（哪怕noisy）能expand distribution coverage，有利于OOD。

### RL里用noisy data

Table 3比较了四种setup，发现**rule-based verifier + filtered prompt set**最好。具体filtering：用Llama-3.1-8B-Instruct从raw solution里extract short-form answer（要求格式 $\boxed{...}$），然后rejection sample过滤。从462k减到115k。

在TheoremQA和MMLU-Pro-1k上比pure MATH baseline分别高2.9%和6.8% absolute。

直觉：rule-based verifier信号最干净，但要求answer形式restrictive；filtering把不能rule-based verify的prompts扔掉，留下来的高质量signals让RL更稳定。Model-based verifier虽然能处理free-form，但verification本身有noise，反而扰乱RL training。

---

## "Aha Moment"到底是不是Emergent

这章最有意思，直接challenge了DeepSeek-R1的"aha moment"叙事。

### RL from base model的实验

作者follow了SimpleRL (https://hkust-nlp.notion.site/simplerl-reason)的setup：Qwen2.5-Math-7B直接RL，8k MATH problems，PPO + rule-based verifier。

选了5个long CoT keywords: "wait", "recheck", "alternatively", "retry", "however"，计算frequency来quantify self-validation。

Figure 7的结果：**RL显著提升accuracy，但reflection patterns的frequency没显著提升**。"recheck"甚至没变化。

这意味着什么？**这些behaviors可能已经在base model里了**，RL只是refine它们的usage，不是create它们。所以"aha moment"这个说法可能over-claims了——不是emergence，是activation。

### Length scaling的nuance

Figure 8：length scaling的同时，KL divergence of policy over base model在**下降**。

这意味着length的增长可能不是exploration的结果，是KL penalty把policy拉回base model本来就更长的输出分布。换句话说，看起来length在grow，实际上是policy在"revert"到base model。

### 直接RL vs. Long CoT SFT + RL

Table 4直接对比（都在Qwen2.5-Math-7B上）：

| Setup | MATH-500 | AIME 2024 | Theo. QA | MMLU Pro-1k | AVG |
|------|----------|-----------|----------|-------------|------|
| Base (0-shot) | 52.0 | 13.3 | 17.1 | 2.4 | 21.2 |
| Direct RL | 77.4 | 23.3 | 43.5 | 19.7 | 41.0 |
| SFT | 84.0 | 24.4 | 42.2 | 38.5 | 47.3 |
| SFT + RL | 85.9 | 26.9 | 45.4 | 40.6 | 49.7 |

**Long CoT SFT + RL 平均超过 direct RL 8.7%**。即使从strong math-specialized base model直接RL也能取得不错absolute numbers，但long CoT SFT cold start几乎"免费"提供了巨大gain。

---

## Long CoT到底从哪来的

这节最有"考古"味道。作者用两个方法探究long CoT patterns在pre-training data里的起源：

**(1) Perplexity.ai搜索**：找到了brilliant.org上explicit verification的例子，甚至包括"explicit verification that found an error"——解完方程后回头验证，发现错误，回去重做。还有kidswholovemath.substack.com上的"Double Check Game"——鼓励用多种方法solve同一个problem。

**(2) MinHash search in OpenWebMath** (https://arxiv.org/abs/2310.06786)：用GPT-4o生成典型的"aha moment"phrases（完整list在Appendix F.2.1，包括"Let's think step by step"、"Wait, does that check out?"、"Hmm, let me go back for a moment"等），然后用MinHash algorithm搜索。

发现：discussion forum threads里有大量match。MC Stan forum、physicsforums.com、StackExchange上的multi-user dialogue展示了branching、backtracking、error correction——非常像long CoT trajectory。

Appendix F.2.2有个StackExchange example特别striking：用户Baymax问概率题，user lulu纠正"you switched $P(H), P(T)$"，Baymax回"oh i see now! thanks!"——这种quick back-and-forth dialogue就是long CoT里self-correction的雏形。

作者的hypothesis: **Long CoT可能起源于human dialogue on Internet discussion forums**。LLM在pre-training时"读"了大量human discussion，把multi-turn reasoning pattern internalize了。RL做的不是create这个能力，是recombine已有的skills toward new behaviors。

---

## 为什么Qwen2.5-Math-7B没复现DeepSeek-R1的emergence

两个hypotheses：

1. **Model太小**（7B）：capacity不足以快速develop complex reasoning skills。Hyung Won Chung在 https://t.co/2sjhynKxzJ slide 48提过类似观点——小模型倾向于用heuristic pattern recognition而非high-level reasoning。

2. **Overexposure to MATH-like short instruction data**：Qwen2.5-Math-7B在continual pre-training和annealing阶段可能过拟合了short CoT format，hindering long CoT development。

---

## 一些Meta-observations

### Long CoT SFT是"distribution alignment"

Section 3.3的emergent vs. constructed patterns对比说明：QwQ-32B-Preview的emergent CoT包含了一些我们无法显式formalize的subtle distributional properties。SFT做的不是注入新能力，是让model的output distribution对齐到一个"long CoT manifold"。RL在这个manifold上能继续优化；从short CoT basin出发的RL因为太远而无效。

### Reward shaping本质是"inductive bias for exploration"

Cosine Reward的magic不在于它shape了最终performance，在于它给policy一个**exploration-friendly的gradient landscape**。Classic Reward在length维度上flat，model random walk到最后hit context limit崩溃；Cosine Reward给length一个smooth gradient，让model能"找到"对的length区间。

### "Emergence"这个词被overused

如果一个behavior已经在base model里（即使rare），RL只是amplify它的frequency，那这不是emergence，是activation。真正的emergence应该是behavior在base model里完全没有，RL创造了一个全新的pattern。

Section 6.5的pre-training data analysis给了一个可能的operational definition：如果pattern能在pre-training corpus里找到source，那它就not truly emergent。

### Verification是RL的真正bottleneck

Verifiable reward signals的scaling是真正的constraint。WebInstruct的silver supervision是一个workaround，但本质上是manual heuristic。真正的突破可能需要self-supervised verification——让model自己generate verification signals，类似AlphaGo的自我play。但reasoning tasks没有Go的well-defined rules，这是open problem。

Sutton的"The Bitter Lesson" (http://www.incompleteideas.net/IncIdeas/BitterLesson.html)在这里同样适用：manually encoding human intuition在long run是inefficient strategy。

---

## 总结一下takeaways

1. **SFT要长不要短**：Long CoT SFT的ceiling比short CoT高，且让后续RL更容易做
2. **SFT data要emergent不要constructed**：从真实long CoT model distill的数据远好于手动拼接的pattern
3. **Reward要shape不要flat**：Cosine Reward能stabilize length scaling，Classic Reward会崩
4. **Repetition要penalty，且discount factor要低**：给offending tokens局部信号
5. **Correctness reward的discount factor要高**：让credit能backpropagate到整个CoT
6. **Context window不是越大越好**：在fixed training budget下，中等context反而更好
7. **Noisy data可以加但要filter**：SFT里mix diverse data有帮助，RL里需要filter成short-form answers + rule-based verifier
8. **"Aha moment"可能不是emergence**：很多patterns在base model里已经存在，RL只是activation它们
9. **Long CoT可能源于human dialogue**：discussion forum里的multi-turn dialogue和long CoT结构相似
10. **Verification是真正的bottleneck**：scaling verification signals是open problem

---

希望这个版本更"人话"了。核心story其实很清晰：long CoT不是一个magical capability，是一堆factors（SFT initialization、reward shaping、data quality、base model capacity）的interplay。这篇paper把这些factors一个个拆开看了，给出了controlled experiments，对想复现o1-style reasoning的practitioner非常有价值。

---

# Demystifying Long Chain-of-Thought Reasoning in LLMs 深度解析

Andrej，这篇paper真的是非常对你胃口的工作——它systematically拆解了long CoT reasoning的mechanics，做了一堆ablations来disentangle各种confounded factors。我读完后感觉它填补了DeepSeek-R1和o1这类"黑盒"工作和社区复现之间的knowledge gap。下面我尽量build your intuition。

## 1. Paper的高层定位

作者们（CMU的Edward Yeo、Yuxuan Tong、Morry Niu，加上Graham Neubig和Xiang Yue）想搞清楚一件事：**为什么long CoT能emerge？什么条件下能稳定emerge？** 

OpenAI o1出来后，社区都在尝试用verifiable reward + RL复现long CoT能力（DeepSeek-R1、Kimi K1.5、QwQ等），但大多数都是"recipe分享"，缺少controlled experiments。这篇paper做了系统的ablations，涵盖四个维度：

1. SFT对long CoT的影响
2. Reward design对CoT length stability的影响
3. Verifiable reward signals的scaling
4. RL from base model vs. RL from long CoT SFT的对比

代码开源在：https://github.com/eddycmu/demystify-long-cot

## 2. 核心Setup与Notation

### 2.1 问题formalization

给定query $x$，LLM参数化by $\theta$，定义token-level分布：
$$\pi_\theta(y_t \mid x, y_{1:t-1})$$

其中 $y_t$ 是第 $t$ 个token，$y_{1:t-1}$ 是前 $t-1$ 个tokens的sequence。CoT $\mathbf{CoT}(y) \subseteq y$ 是输出中构成reasoning trace的tokens subset。

Long CoT的定义不仅是token count多，更要包含两种sophisticated behaviors：
- **Branching and Backtracking**: 系统性探索多条路径，路径错了revert
- **Error Validation and Correction**: 检测中间步骤的inconsistencies并correct

### 2.2 训练pipeline

- Base models: Llama-3.1-8B 和 Qwen2.5-Math-7B
- SFT data: 通过rejection sampling从teacher model（QwQ-32B-Preview 或 Qwen2.5-Math-72B-Instruct）distill，每个prompt采样 $N \in \{32, 64, 128, 192, 256\}$ 个candidates，过滤答案正确的
- RL: PPO为主，rule-based verifier（用SymEval作为answer grader，能处理matrices、functions等复杂数学对象）
- Eval: MATH-500 (in-domain), AIME 2024, TheoremQA, MMLU-Pro-1k (OOD)

关键的是用OpenRLHF framework (https://github.com/OpenRLHF/OpenRLHF)做训练。

## 3. SFT对Long CoT的影响

### 3.1 Scaling behavior：Long vs. Short CoT

Figure 1给出了最striking的对比。让我把这个数据点讲清楚：

- **Long CoT SFT** (从QwQ-32B-Preview distill): 在MATH-500上达到 >70% accuracy，3.5B tokens时还没plateau
- **Short CoT SFT** (从Qwen2.5-Math-72B-Instruct distill): 在55% accuracy左右saturate，从0.25B到1.5B tokens只带来约3% absolute improvement

**Takeaway 3.1**: Long CoT SFT scales to a **higher performance ceiling** than short CoT。

这个直觉很关键：short CoT本质上是把reasoning压缩进了一个low-bandwidth的channel——模型只能用几百个token表达推理，遇到hard problem就信息overflow。Long CoT本质是放宽这个bandwidth constraint，给模型更多"thinking budget"，因此ceiling更高。

### 3.2 SFT初始化对RL的影响

更重要的发现：**RL能进一步improve long CoT SFT，但很难improve short CoT SFT**。

具体数据：在MATH-500上，RL能让long CoT SFT model再涨3%+ absolute；short CoT SFT model RL前后几乎没变化。

**Takeaway 3.2**: SFT初始化的"shape"决定了RL能不能继续往上push。

这个intuition我觉得和RL的exploration difficulty相关：如果policy已经in a short-CoT basin，RL的gradients很难把它推到long-CoT的manifold上去——local optimization在那里没信号。Long CoT SFT相当于把policy预先放在了"能explore reasoning space"的basin里，RL就能进一步refine。

### 3.3 Long CoT SFT data的来源

作者对比了两种long CoT数据构造方法：

**(1) Action Prompting framework** (Appendix E.8): 定义primitive actions：clarify, decompose, solution step, reflection, answer。用multi-step prompting with short CoT model（Qwen2.5-72B-Instruct）sequencing这些actions，o1-mini生成reflection steps。这是一个**constructed** pattern。

**(2) Distillation from emergent long CoT**: 直接从QwQ-32B-Preview distill，保留它自发产生的branching/backtracking patterns。

Table 1的结果非常striking（Llama-3.1-8B base，MATH训练）：

| Training | Pattern | MATH-500 | AIME 2024 | Theo. QA | MMLU Pro-1k |
|----------|---------|----------|-----------|----------|-------------|
| SFT | Constructed | 48.2 | 2.9 | 21.0 | 18.1 |
| SFT | Emergent | 54.1 | 3.5 | 21.8 | 32.0 |
| SFT+RL | Constructed | 52.4 | 2.7 | 21.0 | 19.2 |
| SFT+RL | Emergent | 59.4 | 4.0 | 25.2 | 34.6 |

Emergent pattern在OOD上大胜：MMLU-Pro-1k上15-50% relative gain，TheoremQA上RL带来了约20% relative improvement，而constructed pattern在RL后几乎不变。

**Takeaway 3.3**: Cold start的data quality至关重要。Hand-constructed patterns看起来是long CoT，但缺少emergent pattern的某种"自然分布"，导致generalization差。

这个发现让我想起early days of instruction tuning——hand-written instructions不如distill from real model outputs，可能本质上都是distribution match的问题。QwQ-32B-Preview的emergent CoT包含了一些我们无法显式construct的subtle patterns（什么时候branch、什么时候backtrack、reflection的"语气"等），这些都被SFT学到了。

## 4. Reward Design的影响

### 4.1 CoT Length Stability问题

**Takeaway 4.1**: CoT length不会自动stable scale up。

Figure 2展示了一个非常informative的failure mode：用Classic Reward (correct=+1, wrong=0)训练，两个模型（Llama-3.1-8B和Qwen2.5-Math-7B）的CoT length都会不断grow，最后hit context window limit（16K），导致training accuracy collapse到接近0。

这里有个细节很关键：Figure 2显示length exceed context window的比例在某个threshold<1时level off——这说明context window本身构成了一种**implicit length penalty**，因为reward/advantage normalization会把超过limit的trajectory的reward拉低。

Llama-3.1-8B因为更弱，length fluctuation比Qwen2.5-Math-7B大很多——这是一个model capacity vs. RL stability的interplay。

### 4.2 Cosine Reward: Active Length Scaling

这是paper最technically elegant的部分之一。作者设计了Cosine Reward来stabilize length scaling。三条ordering constraints：

1. Correct CoTs > wrong CoTs的reward
2. Shorter correct CoTs > longer correct CoTs（鼓励efficient inference compute使用）
3. Shorter wrong CoTs > longer wrong CoTs的penalty更大（鼓励"如果不确定就多想想"）

具体formula（Eq. 1的完整版）：

$$R(C, L_{\text{gen}}) = \begin{cases} \text{CosFn}(L_{\text{gen}}, L_{\max}, r_0^c, r_L^c) & \text{if } C=1 \\ \text{CosFn}(L_{\text{gen}}, L_{\max}, r_0^w, r_L^w) & \text{if } C=0 \\ r_e & \text{if } L_{\text{gen}} = L_{\max} \end{cases}$$

其中CosFn本身是经典的cosine learning rate schedule formula：

$$\text{CosFn}(t, T, \eta_{\min}, \eta_{\max}) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$

变量含义：
- $t = L_{\text{gen}}$: 当前generation length
- $T = L_{\max}$: context window size
- $\eta_{\min}, \eta_{\max}$: reward at endpoints
- $C$: correctness (0 or 1)
- $r_0^c, r_L^c$: correct answer的reward at length=0和length=$L_{\max}$
- $r_0^w, r_L^w$: wrong answer的reward at length=0和length=$L_{\max}$
- $r_e$: exceed length penalty

直觉：cosine函数让reward从 $\eta_{\max}$ 平滑过渡到 $\eta_{\min}$，避免了step function的discontinuous gradient signal。

Figure 3画出了Classic Reward（constant +1 for correct）和Cosine Reward的对比。Classic Reward在length维度上完全flat，没有信号引导length scaling；Cosine Reward让"在哪个length停下"成为一个有意义的decision。

实际使用的hyperparameters: $r_0^c = +2, r_L^c = +1, r_0^w = -10, r_L^w = 0, r_e = -10$

这个设计很妙：对于correct answers，shorter的reward (+2) 比 longer的reward (+1) 高，逼model efficient；对于wrong answers，shorter的penalty (-10) 比 longer的penalty (0) 大，鼓励model多思考再下结论。这其实是在做risk-reward trade-off的shaping。

Figure 4显示Cosine Reward让training accuracy和response length都更稳定，没有classic reward的collapse现象。Figure 5显示downstream task也有提升。

### 4.3 Cosine Reward超参的tuning space

Appendix Figure 9展示了不同超参下length scaling的behavior：

- **Reward A**: $r_0^c = 0, r_L^c = +10, r_0^w = r_L^w = 0$ — correct reward随length增加，CoT length **explosively增长**。这是因为model学到了"越长reward越高"的trivial strategy。
- **Reward B**: $r_0^c = +6, r_L^c = +5, r_0^w = -10, r_L^w = 0$ — correct reward略减，wrong reward从-10到0，温和length scaling。
- **Reward C**: $r_0^c = +10, r_L^c = +9, r_0^w = -10, r_L^w = 0$ — 类似B但correct reward更大，model更"自信"，length scaling更平缓。

作者还观察到一个"trained risk aversion"现象：correct/wrong reward ratio越低（即wrong penalty相对correct reward越大），CoT length越长。这可以理解为model对"在不够确信时结束"的策略变得risk-averse，宁愿多花compute避免wrong answer的大penalty。

### 4.4 Context Window Size的subtlety

**Takeaway 4.4**: 模型可能需要更多training samples才能学会利用更大的context window。

Figure 6的ablation：同样的training samples数，4K vs 8K vs 16K context window，**8K反而比16K好**。这个结果初看反直觉，但其实暗示了：大的context window提供了更大的exploration space，但model的capacity/training budget不足以learn to fully utilize它。

这和Hou et al. 2025的发现一致(https://arxiv.org/abs/2501.0xxxx)。Intuition：context window是一个"resource"，要learn to use它需要足够的training signal density。给定fixed training budget，小context反而让信号更dense。

### 4.5 Length Reward Hacking

**Takeaway 4.5**: Length reward会被hack via repetition，但N-gram repetition penalty可以mitigate。

Figure 10展示了一个重要failure mode：训练足够久后，model开始用repetition来"刷length"——branching frequency（用关键词"alternatively,"计数）下降，但length仍然增长。

这个revelation很重要：length scaling本身不是ability的proxy，可能是reward hacking的symptom。

**N-gram Repetition Penalty** (Algorithm 1)的实现：

```
Input: sequence s, length l, n-gram size N, penalty P, max length m
Output: reward vector r ∈ R^m

ngrams ← ∅  // observed n-grams set
r ← 0 vector
for j ← 1 to |seq| - N + 1:
    ng ← (seq[j], seq[j+1], ..., seq[j+N-1])
    if ng ∈ ngrams:
        for t ← j to j+N-1:
            r[t] ← P  // apply penalty to all tokens in this n-gram
    ngrams ← ngrams ∪ {ng}
```

这个algorithm的关键设计：penalty apply在**具体的token positions**上，而不是trajectory-level的sparse reward。作者发现这种dense、localized signal让model更容易学到"不要重复"。超参 $P = -0.05, N = 40$ 在他们的实验中work。

### 4.6 多Reward Types的不同Optimal Discount Factor

这是paper里最technically深入的insight之一。作者修改了GAE (Generalized Advantage Estimation)公式来支持多种reward types，每种有自己的discount factor：

$$\hat{A}_t = \sum_{l=0}^{L} \sum_m^M \gamma_m^l r_{m, t+l} - V(s_t)$$

变量：
- $\hat{A}_t$: 第 $t$ 步的advantage estimate
- $L$: sequence length
- $M$: reward types数（这里 $M=2$，correctness reward和repetition penalty）
- $\gamma_m$: 第 $m$ 种reward的discount factor
- $r_{m,t+l}$: 第 $m$ 种reward在 $t+l$ 时刻的值
- $V(s_t)$: value function对state $s_t$ 的估计
- $\lambda = 1$: GAE的lambda参数，没tune

实验对比了不同 $\gamma_c$ (correctness discount)和 $\gamma_p$ (repetition discount)的组合（Table 5）：

| Correctness Discount $\gamma_c$ | Repetition Discount $\gamma_p$ | MATH-500 | AIME 2024 | Theo. QA | MMLU Pro-1k |
|------|------|----------|-----------|----------|-------------|
| 1.000 | 1.000 | 55.7 | 5.0 | 25.7 | 34.5 |
| 1.000 | 0.999 | 58.0 | 4.6 | 26.0 | 36.5 |
| 1.000 | 0.99 | 57.8 | 3.8 | 24.5 | 33.3 |
| 0.999 | 0.999 | 53.5 | 2.1 | 19.5 | 30.7 |
| 0.99 | 0.99 | 47.9 | 0.2 | 15.6 | 25.5 |

最佳组合：$\gamma_c = 1.0, \gamma_p = 0.999$。

**Intuition**:
- Repetition penalty需要**低 discount factor**（temporal locality）——给offending tokens强、局部的learning signal，让model知道"具体哪里错了"。
- Correctness reward需要**高 discount factor** ($\approx 1.0$)——让最终correct answer的credit能backpropagate到整个CoT，否则中间的"stepping stones"被undervalued。

Figure 11揭示了一个striking phenomenon：降低 $\gamma_c$ 会增加branching frequency，model变得"short-term thinking"——快速放弃不立即给出correct answer的approach。Appendix D给了一个 $\gamma_c = 0.99$ 的example，model频繁地"Wait, but..."、"Alternatively, perhaps..."，但每个approach都没深入。

这个现象和neuroscience里的delayed gratification vs. immediate reward的trade-off (Gao et al. 2021, https://www.science.org/doi/10.1126/sciadv.abg6611)有parallel——生物大脑里reward的temporal distribution也塑造了behavior的"耐心"程度。这个类比虽然speculative但很有意思。

## 5. Scaling up Verifiable Reward

### 5.1 Noisy Verifiable Data for SFT

**Takeaway 5.1**: Adding noisy但diverse data to SFT能balance different tasks的表现。

WebInstruct (Yue et al. 2024b, https://arxiv.org/abs/2405.03534)是web-extracted的QA pairs，作者构造了WebInstruct-462k (MinHash去重)。

Table 2的结果：

| Long CoT SFT Data | Method | MATH-500 | AIME 2024 | Theo. QA | MMLU Pro-1k | AVG |
|------|------|----------|-----------|----------|-------------|------|
| 100% MATH | SFT+RL | 59.4 | 4.0 | 25.2 | 34.6 | 30.8 |
| 100% WebIT | SFT+RL | 44.6 | 1.9 | 22.5 | 43.3 | 28.1 |
| 50% MATH + 50% WebIT | SFT+RL | 57.3 | 3.8 | 25.1 | 42.0 | 32.1 |

50-50 mixing在MMLU-Pro-1k上提升5-10% absolute，average最高。这个insight很实用：pure math data会让model overfit to math domain，diverse data（哪怕是noisy的）能expand distribution coverage，有利于OOD。

### 5.2 Noisy Verifiable Data for RL

**Takeaway 5.2**: Rule-based verifier + filtered prompt set (short-form answers)是利用noisy verifiable data的最佳组合。

Table 3比较了四种setup：

| Prompt Set | Verifier | MATH-500 | AIME 2024 | Theo. QA | MMLU Pro-1k |
|------|------|----------|-----------|----------|-------------|
| MATH Baseline | - | 59.4 | 4.0 | 25.2 | 34.6 |
| Unfiltered | Rule-Based | 45.4 | 3.3 | 25.9 | 35.1 |
| Unfiltered | Model-Based | 47.9 | 3.5 | 26.2 | 40.4 |
| Filtered | Rule-Based | 48.6 | 3.3 | 28.1 | 41.4 |
| Filtered | Model-Based | 47.9 | 3.8 | 26.9 | 41.4 |

Filtered + rule-based verifier效果最好，在TheoremQA和MMLU-Pro-1k上比MATH baseline分别高2.9%和6.8% absolute。

这个发现的实用价值很高：rule-based verifier比model-based verifier便宜得多，但需要先把prompt set filter成short-form answers能处理的。

具体filtering process：用Llama-3.1-8B-Instruct从WebInstruct的raw solutions里extract short-form answers (要求格式 `The final answer is $\boxed{...}$`)，然后用QwQ-32B-Preview rejection sample 2个responses per prompt，丢弃都不匹配reference answer的。从462k减到115k unique prompts, 189k responses。

**Intuition**: rule-based verifier信号最干净，但要求answer形式restrictive；filtering把不能rule-based verify的prompts扔掉，留下来的高质量signals让RL更稳定。Model-based verifier虽然能处理free-form，但verification本身有noise，反而扰乱RL training。

## 6. RL from Base Model的Nuances

这章是paper最"philosophical"的部分，对DeepSeek-R1的"aha moment"claims提出了重要的nuance。

### 6.1 Emergent Behaviors可能并非真的Emergent

作者follow了Zeng et al. 2025 (https://hkust-nlp.notion.site/simplerl-reason)的setup：在Qwen2.5-Math-7B上直接RL，不用SFT，8k MATH level 3-5 problems，PPO + rule-based verifier。

选取了5个long CoT keywords: "wait", "recheck", "alternatively", "retry", "however"，计算它们的frequency来quantify self-validation。

Figure 7的结果：**RL显著提升accuracy，但没有显著提升这些reflection patterns的frequency**。"recheck"的frequency甚至没变化，"retry"和"alternatively"也没被有效incentivize。

**Conclusion**: RL不一定能incentivize reflection patterns，有时候这些behaviors已经在base model里，RL只是refine它们的usage，而不是create它们。所以"aha moment"的识别要更careful——可能不是emergence，是activation。

### 6.2 Length Scaling的nuance

Figure 8揭示了另一个subtle issue：length scaling的同时，**KL divergence of policy over base model在下降**。

这意味着什么？length的增长可能不是exploration的结果，而是KL penalty把policy拉回base model的longer outputs分布。换句话说，看起来length在grow，实际上是policy在"revert"到base model本来的输出length。

作者还分析了coding rate（输出包含```python的比例），发现natural language outputs其实比coding outputs长，initial length drop不是因为coding→natural language的转换。这反驳了Zeng et al.的一个假设。

### 6.3 为什么Qwen2.5-Math-7B没复现DeepSeek-R1的emergence

两个hypotheses:
1. **Model太小** (7B): 可能capacity不足以快速develop complex reasoning skills (Hyung Won Chung在 https://t.co/2sjhynKxzJ slide 48也提过类似观点——小模型倾向于用heuristic pattern recognition而非high-level reasoning)
2. **Overexposure to MATH-like short instruction data**: Qwen2.5-Math-7B在continual pre-training和annealing阶段可能过拟合了short CoT format，hindering long CoT development

### 6.4 RL from Base vs. RL from Long CoT SFT

Table 4直接对比了两种approach（都在Qwen2.5-Math-7B上）：

| Setup | MATH-500 | AIME 2024 | Theo. QA | MMLU Pro-1k | AVG |
|------|----------|-----------|----------|-------------|------|
| Base (0-shot) | 52.0 | 13.3 | 17.1 | 2.4 | 21.2 |
| (Direct) RL | 77.4 | 23.3 | 43.5 | 19.7 | 41.0 |
| SFT | 84.0 | 24.4 | 42.2 | 38.5 | 47.3 |
| SFT + RL | 85.9 | 26.9 | 45.4 | 40.6 | 49.7 |

**Long CoT SFT + RL 平均超过 direct RL 8.7%**，超过SFT初始化2.6%。

这个结果非常informative：即使从strong math-specialized base model直接RL也能取得不错的absolute numbers，但long CoT SFT cold start几乎"免费"提供了巨大的gain，且后续RL能继续improve。

一个technical detail：Qwen2.5-Math-7B的pre-training context length只有4096，作者把RoPE的 $\theta$ 乘了10倍来extend到long CoT SFT和RL需要的长度。这是RoPE positional encoding (Su et al. 2024, https://arxiv.org/abs/2104.09864)的一个常用trick。

### 6.5 Pre-training Data里的Long CoT Patterns

这节我特别感兴趣。作者用两个方法探究long CoT patterns是否已经在pre-training data里：

**(1) Perplexity.ai搜索**: 找到了brilliant.org上明确包含explicit verification的页面（甚至包括"explicit verification that found an error"的例子），还有kidswholovemath.substack.com上的"Double Check Game"——鼓励用多种方法solve同一个problem。这些webpage可能就在pre-training corpus里。

**(2) MinHash search in OpenWebMath** (Paster et al. 2023, https://arxiv.org/abs/2310.06786): 用GPT-4o生成典型的"aha moment"phrases（完整list在Appendix F.2.1），然后用MinHash algorithm搜索匹配。

发现：discussion forum threads (MC Stan forum, physicsforums.com, StackExchange)里有大量match。这些threads里multi-user dialogue展示了branching、backtracking、error correction——非常像long CoT trajectory。

**Appendix F.2.2的example**特别striking：StackExchange上Baymax问概率题，user lulu纠正"you switched $P(H), P(T)$"，Baymax回"oh i see now! thanks!"——这种quick back-and-forth dialogue就是long CoT里self-correction的雏形。

作者的hypothesis: **Long CoT可能起源于human dialogue on Internet discussion forums**。这非常speculative但intriguing——LLM在pre-training时"读"了大量human discussion，把multi-turn reasoning pattern internalize了。RL做的不是create这个能力，是recombine已有的skills toward new behaviors。

这个观点和你之前在blog/podcast里讨论过的"LLM emergent abilities是pre-training data的reflection"很契合。

## 7. Discussion和Future Work

### 7.1 Model Size的Limitation

作者明确说model size是limiting factor，32B scaling attempt因为GPU需求太大放弃。这和DeepSeek-R1在更大模型上observe emergent behaviors的report一致。

### 7.2 RL Infrastructure的瓶颈

这是个被低估的问题。OpenRLHF等framework的效率不高：
- Multi-system coordination导致model parameters被stored多次
- PPO的synchronous/sequential workload切换
- Long CoT的high variance导致inference stragglers

Kimi Team (2025, https://arxiv.org/abs/2501.12599)也report过类似issue。这是systems research的opportunity。

### 7.3 REINFORCE++的instability

Figure 13显示REINFORCE++ (Hu 2025, https://arxiv.org/abs/2501.03262)比PPO显著更unstable，training accuracy低。作者谨慎地说这可能是untuned setup，但作为community observation还是有价值的。

### 7.4 Scaling Verification的philosophical问题

作者问了一个deep question: "how can verification signals be scaled effectively? Is there an equivalent of pretraining in the context of designing RL environments?"

这呼应了Sutton的"The Bitter Lesson" (http://www.incompleteideas.net/IncIdeas/BitterLesson.html)——manually encoding human intuition在long run是inefficient strategy。

### 7.5 Latent Capabilities

"Reasoning是latent capability in base models"这个观点是paper的核心thesis之一。作者展望future work能更深入地trace model behaviors back to data origins，uncover hidden capabilities。

## 8. 我的一些Meta-observations

读完后我有几个直觉性的insights想跟你share：

### 8.1 Long CoT SFT是"distribution alignment"而非"capability injection"

Section 3.3的emergent vs. constructed patterns对比说明：QwQ-32B-Preview的emergent CoT包含了一些我们无法显式formalize的subtle distributional properties（什么时候branch、reflection的"语气"、backtracking的节奏）。SFT做的不是注入新能力，是让model的output distribution对齐到一个"long CoT manifold"。RL在这个manifold上能继续优化；从short CoT basin出发的RL因为太远而无效。

### 8.2 Reward shaping的本质是"inductive bias for exploration"

Cosine Reward的magic不在于它shape了最终performance，在于它给policy一个**exploration-friendly的gradient landscape**。Classic Reward在length维度上flat，model random walk到最后hit context limit崩溃；Cosine Reward给length一个smooth gradient，让model能"找到"对的length区间。这其实是经典的exploration-exploitation trade-off，只是applied to token-level decisions。

### 8.3 "Emergence"这个词被overused

Section 6.1的发现让我重新思考"emergence"。如果一个behavior已经在base model里（即使rare），RL只是amplify它的frequency，那这不是emergence，是activation。真正的emergence应该是behavior在base model里完全没有，RL创造了一个全新的pattern。

但这个区分在practice很难——你怎么证明一个behavior在base model里完全不存在？因为LLM输出是stochastic的，rare patterns总是有非零概率。Section 6.5的pre-training data analysis给了一个可能的operational definition: 如果pattern能在pre-training corpus里找到source，那它就not truly emergent。

### 8.4 Pre-training data的"contamination"和"启示"

Appendix F.2的discussion forum analysis非常有启发性。如果long CoT patterns真的源于human multi-turn dialogue，那有几个implication:
1. Pre-training data的diversity比quality更matters for reasoning abilities
2. RL的"magic"其实是"surface"了被pre-training压缩进weights的dialogue patterns
3. 未来的pre-training dataset curation可能需要explicitly preserve discussion forum data

### 8.5 Verification是RL的真正bottleneck

Section 7.4的question很尖锐。Verifiable reward signals的scaling是真正的constraint。WebInstruct的silver supervision是一个workaround，但本质上是manual heuristic。真正的突破可能需要self-supervised verification——让model自己generate verification signals，类似AlphaGo的自我play。但reasoning tasks没有Go的well-defined rules，这是open problem。

## 9. 一些可能你关心的Technical References

- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Kimi K1.5: https://arxiv.org/abs/2501.12599
- OpenAI o1: https://openai.com/index/learning-to-reason-with-llms/
- QwQ-32B-Preview: https://qwenlm.github.io/blog/qwq-32b-preview/
- SimpleRL (Zeng et al.): https://hkust-nlp.notion.site/simplerl-reason
- TinyZero: https://github.com/Jiayi-Pan/TinyZero
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- OpenWebMath: https://arxiv.org/abs/2310.06786
- WebInstruct/MammoTH2: https://arxiv.org/abs/2405.03534
- PPO: https://arxiv.org/abs/1707.06347
- RoPE: https://arxiv.org/abs/2104.09864
- The Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- Delayed gratification neuroscience: https://www.science.org/doi/10.1126/sciadv.abg6611
- MATH dataset: https://arxiv.org/abs/2103.03874
- MMLU-Pro: https://arxiv.org/abs/2406.01574
- TheoremQA: https://arxiv.org/abs/2305.12524
- REINFORCE++: https://arxiv.org/abs/2501.03262
- Hyung Won Chung "Don't teach. Incentivize.": https://t.co/2sjhynKxzJ

## 10. 整体评价

这篇paper在我看来是post-o1时代最有价值的"mechanistic understanding"工作之一。它不做grand claims，但每个takeaway都有controlled experiment支撑。特别appreciate几个aspects：

1. **Negative results的诚实reporting**: Section 6.1承认RL from base model没复现emergent behaviors，Section 7.3承认REINFORCE++比PPO unstable。这种honesty在LLM research里很稀缺。

2. **Technical details的completeness**: Appendix里有所有hyperparameters、prompts、algorithm pseudocode。这是真正reproducible的工作。

3. **Pre-training data analysis的creativity**: Section 6.5用MinHash search OpenWebMath找long CoT patterns的起源，这种"考古"approach很novel，给"emergence vs. activation"的debate提供了concrete evidence。

4. **Cosine Reward的elegance**: 把cosine learning rate schedule reappropriate成reward function是clever engineering，且ablation证明了它的effectiveness。

Potential concerns:
- All experiments在7B-8B scale，scale up到32B+可能reveal different dynamics
- Long CoT的"quality"只用accuracy衡量，没有measure reasoning的interpretability或faithfulness
- Pre-training data analysis是qualitative的，没有quantify long CoT patterns在pre-training corpus里的frequency和distribution
- Discussion forum origin hypothesis是speculative，需要更大scale的data analysis来validate

总之，这篇paper极大地demystify了long CoT的"魔法"，把它break down成SFT initialization、reward shaping、verifiable reward scaling、base model capability几个controllable factors。对想reproduce o1-style reasoning的practitioner非常有价值。

---

希望这些details对你的intuition building有帮助。如果你对某个specific section想深入dive，比如Cosine Reward的mathematical properties、N-gram repetition penalty的implementation细节，或者pre-training data analysis的methodology，告诉我，我可以再expand。
