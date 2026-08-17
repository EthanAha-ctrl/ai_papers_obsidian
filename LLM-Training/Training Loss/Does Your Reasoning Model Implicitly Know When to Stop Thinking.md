---
source_pdf: Does Your Reasoning Model Implicitly Know When to Stop Thinking.pdf
paper_sha256: 2dc633a448cf8475ff158470e1fe692f27f02b254422f936e32a28ded11253ef
processed_at: '2026-08-03T23:06:38-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话版本

大模型reasoning的时候其实心里有数什么时候该停，但现在的采样方式把这个"心里有数"给埋没了，作者发明了一个采样方法把模型的这个本能给"挖"出来，然后通过RL让模型把这个能力固化下来，结果accuracy涨了、token少了，皆大欢喜。

---

## 这个problem到底有多烦人

你给DeepSeek-R1一道AIME题，它"想"出来的token数量是Claude 3.7 Sonnet的5倍，但accuracy差不多。你给QwQ-32B一道题，随机采样出来的答案比"最短的那批答案"还差2个百分点，而最短那批用的token少了31%。

换句话说：**模型想太多，想太多反而更容易错**。

这不是个别case。作者定义了一个特别直观的指标叫RFCS（Ratio of First Correct Step）：

$$\text{RFCS} = \frac{\text{第一次算对的step位置}}{\text{总step数}}$$

比如一道题模型分10步思考，第3步就把答案算对了，但后面7步还在反复检查、自我安慰、重新推导，RFCS = 3/10 = 0.3。

他们用DS-1.5B、DeepScaleR、Qwen3-8B三个模型在MATH-500上跑统计，发现**超过一半的正确回答里RFCS < 1**。也就是说模型明明已经答对了，还在那墨迹。

更扎心的是：DeepScaleR和Qwen3-8B这种"更强大"的模型，RFCS并没有比DS-1.5B好多少。post-training做得越多、reasoning能力越强，这个"墨迹病"反而越顽固。

---

## 为什么会这样？一个直觉解释

你想象一个学生在考试，做一道数学题。他脑子里其实3步就解出来了，但他不敢相信，于是在草稿纸上反复验算、换个方法再做一遍、自言自语"让我再确认一下"。

这个学生"知道"答案，但他的"行为"不是停在"知道答案"那一刻，而是停在他"心理上觉得安全"的那一刻。

大模型的greedy sampling和random sampling就是这种"心理安全"的proxy——它们看的是**next-token probability**，即"下一个token我有多自信"。但问题是：

**模型在path level的自信 ≠ token level的自信**。

作者发现了一个非常striking的现象：当一个reasoning branch在宏观上"该结束了"的时候，`` 这个token的next-token probability其实可能很低。为什么？因为周围的token distribution很分散（模型在想"是总结呢还是再验算呢还是换个角度呢"），`` 只是众多选项中的一个。

但从整条path看，这条path的累积信心很高，`` 是"顺理成章"的收尾。

这就是paper的核心insight：**模型心里有数，但greedy/random只看眼前一步，看不到全局，于是错过了该停的时刻**。

---

## Φ 这个东西是什么，为什么重要

作者定义了一个叫 Φ 的东西，说白了就是"整条path的平均信心"：

$$\Phi(\mathbf{y}_{\le k}) = \frac{1}{k}\sum_{i=1}^{k} \phi(y_i; \mathbf{y}_{<i})$$

- $k$：当前生成了多少个token
- $\phi(y_i; \mathbf{y}_{<i})$：第i个token的log-probability
- $\Phi$：把所有token的log-prob加起来除以长度，相当于"这条path平均每一步有多自信"

对比一下：
- **φ（next-token prob）**：只看下一步，像近视眼
- **Φ（cumulative avg prob）**：看整条path，像望远镜

作者做了个ablation实验特别有意思。他们搞了两个版本的search：
- TSearch w/ Φ：用"望远镜"来决定保留哪条branch
- TSearch w/ φ：用"近视眼"来决定保留哪条branch

然后逐步增大exploration width（搜索宽度）m：

| 增大m | TSearch w/ Φ | TSearch w/ φ |
|-------|---------------|---------------|
| response长度 | 变短 | 急剧collapse |
| accuracy | 提升 | 急剧下降 |

也就是说：用Φ搜索，越搜越精越搜越短；用φ搜索，越搜越短但越搜越烂，直接collapse成写废话。

**这说明是Φ在做"该停就停"的智能判断，φ只会贪婪地选最短的，不管对不对**。

---

## `` token的rank诡异现象

这个发现我觉得是paper最漂亮的observation。

作者记录了 `` token在搜索过程中出现时的rank（在候选token列表里排第几）。结果是：

- **TSearch w/ Φ**：`` 出现时，在按Φ排序的候选里**总是排第一**
- **TSearch w/ φ**：`` 出现时，在按φ排序的候选里**rank越来越靠后**

翻译成人话：

当你用"整条path的信心"来评判时，模型说"该停了"的时候，它确实是非常confident的——这条path就是最该停的那条。

但当你用"下一步的prob"来评判时，模型说"该停了"的那个token，恰恰是它"下一步最不确定"的时候——因为下一步有很多选择（继续推理？总结？换方法？），`` 只是一个"勉强够格"的选项。

**这就是为什么greedy和random sampling会miss掉这些高效path**：它们在 `` 该出现的时候，看到的 `` prob不够高，于是选了别的token继续走，结果越走越长。

---

## SAGE算法到底干了啥

知道了上面的insight，SAGE的算法其实非常简单。

### 核心idea

既然模型用Φ判断时 `` 总是排第一，那我们就不用做token-level的精细搜索了，直接做**step-level的搜索**就行。

### 具体步骤

1. 给模型一个query，先采样2m个"第一步"（m是exploration width，比如m=2就采样4个第一步）
2. 对每个candidate，计算它的 Φ
3. 保留 Φ 最高的m个candidate
4. 对这m个candidate，每个再采样2m个"下一步"
5. 重复，直到某个candidate的step以 `` 结尾
6. 把这个candidate加入"完成集"
7. 当"完成集"里有r个candidate时，停止

就这么简单。不用设TR（tolerance ratio），不用做token-level的beam search，就是一个**step-wise的best-first search，用 Φ 当score function**。

### 为什么比beam search好

beam search的问题是：它会一直expand所有beam，最后可能把"早早遇到 `` 的短branch"给挤掉，因为其他branch虽然长但累积prob高。

SAGE的做法是：**一旦看到 `` 就立即接受这个branch作为完成**，不再跟其他branch竞争。这保证了"该停的branch不会被长的branch挤掉"。

---

## SAGE-RL：怎么把efficient pattern"焊"进模型

SAGE是一个inference-time的方法，你用它来生成答案会更高效。但每次inference都要跑SAGE太麻烦了。能不能让模型"学会"SAGE发现的pattern，在普通pass@1 sampling下也这么高效？

SAGE-RL的思路超级简单：**在RL训练的rollout阶段，混入SAGE采样的样本**。

具体来说，GRPO每个query采样G=8个response。SAGE-RL的做法是：
- 其中2个用SAGE采样（高质量、短）
- 另外6个用普通random sampling（照旧）
- 然后照常计算advantage、更新policy

就这么简单。**不修改reward function，不增加length penalty，只改采样方式**。

### 为什么这样能work

RL的advantage机制是group-relative的：
$$\hat{A}_i = \frac{r(x, y_i) - \text{mean}(\text{group})}{\text{std}(\text{group})}$$

如果SAGE采样的2个response又短又对（reward=1），而random采样的6个又长又可能错，那么：
- SAGE样本的advantage很高（reward高 + group平均被拉低）
- policy会"朝着SAGE的方向"更新
- 长期下来，model的random sampling也会越来越像SAGE发现的pattern

这就像是：**在group里放两个"学霸"，让其他六个"普通学生"通过对比学会学霸的学习方法**。

---

## 实验效果到底多好

我挑几个最striking的数据点：

### DS-1.5B（小模型）+ SAGE-GRPO

| Benchmark | Pass@1变化 | Length变化 | Token Efficiency变化 |
|-----------|------------|------------|---------------------|
| MATH-500 | 83.2→84.8 (+1.6%) | 4882→2915 (-40%) | +70.7% |
| AIME 2025 | 20.9→26.5 (+5.6%) | 11669→7479 (-36%) | +97.8% |
| AMC23 | 60.1→66.3 (+6.2%) | 8250→5091 (-38%) | +78.9% |

小模型 + 数学题，accuracy涨5-6个百分点，token少40%，efficiency翻倍。

### Qwen3-8B（强模型）+ SAGE-GSPO

| Benchmark | Pass@1变化 | Length变化 | Token Efficiency变化 |
|-----------|------------|------------|---------------------|
| Minerva | 51.8→53.7 (+1.9%) | 7358→3363 (-54%) | +126.8% |
| AIME 2025 | 67.3→66.0 (-1.3%) | 18342→9183 (-50%) | +95.9% |

强模型 + 难题，accuracy基本持平或微涨，但token砍半。efficiency几乎翻倍。

### 对比其他efficient reasoning方法

现有的方法比如LC-R1、ThinkPrune、AdaptThink，它们的pattern是：**压缩token但牺牲accuracy**。比如AdaptThink在MATH-500上把DS-1.5B的token从4882压到2563（-48%），但pass@1从83.2掉到80.4（-2.8%）。

SAGE-RL是**同时涨accuracy和压token**。这在efficient reasoning领域是很少见的，因为通常accuracy和efficiency是trade-off。

---

## 一个特别有意思的pattern

作者在Section 5.2总结了一个规律：

**强模型 + 难题 → SAGE主要涨accuracy**
**弱模型 + 易题 → SAGE主要压length**

直觉是：
- 弱模型本来就overthinking严重，SAGE帮它"别墨迹"，主要收益在省token
- 强模型的capability ceiling高，SAGE帮它"找到更好的解法"，主要收益在accuracy
- 难题需要更多必要token，SAGE在保证必要长度的前提下找到更准的path
- 简单题大部分token都是冗余的，SAGE直接砍掉冗余

这个pattern让SAGE-RL在不同场景下都能work，只是收益的"侧重点"不同。

---

## Training Dynamics的有趣发现

Figure 9展示了一个有意思的现象：

SAGE-RL相比vanilla GRPO：
- **Entropy下降更快**：model变得更"笃定"
- **KL divergence上升更快**：model偏离initial policy更多

这说明什么？**SAGE发现的efficient reasoning pattern和model原本的distribution差距很大**，model需要"大改"才能学会。这反过来证明：random sampling确实"学不到"这些pattern，因为它们太偏离model的comfort zone了。

只有通过SAGE"强行"把这些pattern放进rollout，再通过RL的advantage机制"强行"让model学，model才能跨越这个gap。

---

## 我的intuition和一堆联想

### 联想1：System 1 vs System 2

Kahneman的System 1/System 2框架。大模型的reasoning是System 2（慢思考），但model其实也有"System 1的直觉"——它在path level知道答案。SAGE像是把System 1的直觉提取出来，用System 2的方式表达。

这让我想到Anthropic的"Distilling System 2 into System 1"那篇paper，思路相似：把慢思考的pattern蒸馏成快思考的reflex。

### 联想2：Sparse retrieval里的dense vs sparse

Φ vs φ 有点像dense retrieval vs sparse retrieval：
- φ是local的、token-level的，像BM25看关键词匹配
- Φ是global的、path-level的，像dense embedding看语义相似度

SAGE用Φ做pruning，本质上是从"local view"切换到"global view"。这个思路在NLP很多地方都work：sentence-level vs token-level、document-level vs passage-level。

### 联想3：MCTS里的value function

AlphaGo里policy network告诉你"下一步走哪"，value network告诉你"这个局面值多少分"。SAGE里 φ 像 policy network（局部决策），Φ 像 value function（全局评估）。

但SAGE的巧妙之处是：它没有单独训一个value function，而是用policy自己的累积log-prob当value的proxy。这说明policy model的内部representation已经包含了path-level的value信息，只是我们一直没用对方式提取。

### 联想4：Speculative decoding的verification

Speculative decoding里，draft model生成token，target model验证。SAGE有点像：policy自己生成step，policy自己的 Φ 验证。只不过speculative decoding是token-level的验证，SAGE是step-level的验证。

如果能做一个"step-level speculative decoding"——draft model生成step，target model用 Φ 验证step是否"该结束"——可能会很有意思。

### 联想5：Information bottleneck

从信息论角度，冗余token是那些不增加answer信息的token。Φ 可能是information gain的implicit proxy：高 Φ path意味着每token都在有效推进，低 Φ path有大量"原地踏步"。

这跟Information Bottleneck theory、Minimum Description Length都有哲学上的联系。最理想的reasoning chain应该是MDL——用最少的token表达完整的推导。

### 联想6：为什么RLVR能work

这篇paper间接回答了一个问题：**为什么RLVR能提升reasoning**？

Wen et al. 2025b那篇"RLVR implicitly incentivizes correct reasoning"说RLVR让model学到正确的reasoning pattern。SAGE-RL进一步说明：**只要rollout里有"好的pattern"，RL的advantage机制会自动学到它**。

SAGE-RL的本质就是：用SAGE在rollout里"制造"好的pattern，让RL去学。这个思路可以推广——任何能在rollout里注入"高质量样本"的方法，理论上都能通过RL让model学到。

### 联想7：Self-play和self-distillation

SAGE-RL有点像self-play：
- SAGE用policy自己的 Φ 发现"更好的path"
- RL让policy学习这些"更好的path"
- policy变强后，SAGE能发现更好的path
- 循环迭代

这跟AlphaGo的self-play、Constitutional AI的self-improvement都是同一种哲学：**用model自己的能力来提升model自己**。

### 联想8：为什么 `` 的next-token prob会低

这个现象值得深挖。我的猜测是：

当model reason到一个"该总结"的点时，它的attention分布其实很分散——它在考虑"要不要总结"、"总结什么"、"要不要再验证"。这时候 `` 只是众多选项之一，prob自然被稀释。

但如果model在这条path上的每一步都很"笃定"（高 Φ），说明它的"内部状态"已经收敛了，它"知道"该停了。只是这个"知道"没充分反映在next-token prob上。

这有点像人的直觉：你"知道"答案了，但如果让你"说出"下一个该想什么，你会犹豫。你的"知道"是global的，但"说下一步"是local的。

### 联想9：跟length penalty方法的本质区别

LC-R1、ThinkPrune这些方法加length penalty到reward里。问题是：
- Length penalty是**全局施加**的，不管这道题需不需要长
- Model可能学会"写废话凑数"或"偷工减料"
- Reward hacking风险高

SAGE-RL的哲学完全不同：**不改reward，改sampling**。让model通过对比"高质量短样本"vs"低质量长样本"自己学会什么时候该长什么时候该短。

这就像是：不告诉学生"答案要短"，而是让学生看"学霸的短答案"和"自己的长答案"对比，自己悟出来。

### 联想10：可能的failure mode

我担心的几点：

**第一**：如果SAGE找到的"短path"其实是错的shortcut怎么办？比如model用某个错误的heuristic快速得到答案，SAGE的 Φ 很高（因为model很"笃定"），但答案是错的。RL的rule-based reward会惩罚这个，但如果错误很subtle，可能需要PRM。

**第二**：SAGE的time complexity在训练时比random sampling高。虽然paper说tuned model在inference时更快，但训练成本是个问题。特别是在大规模RL训练里，rollout是bottleneck。

**第三**：只在math上验证了。code reasoning、commonsense reasoning、multi-hop QA——这些任务的"step"定义不清晰，SAGE的step-wise exploration可能不适用。

---

## 最后的大图景intuition

这篇paper让我对大模型reasoning有了新的mental model：

**大模型的reasoning是分层的，每一层都有自己的"confidence"，但下一层的sampling noise会mask上一层的confidence**。

- Token level有 φ（next-token prob）
- Step level有 Φ（cumulative avg prob）
- Path level可能还有更macro的信号我们没发现

我们一直被困在token level，greedy和random都只看 φ。SAGE让我们看到了step level的 Φ。未来可能需要更macro的sampler来看path level甚至problem level的confidence。

**每一层"知道"的东西，都被下一层的"noise"给mask了**。Sampling paradigm的革新，本质上就是"穿透更多层的noise，提取更深层的知道"。

这可能是test-time compute scaling的下一个frontier：不是sample更多，而是sample更smart；不是看更多token，而是看更多layer的signal。

---

## 关键References

- Paper project page: [https://hzx122.github.io/sage-rl/](https://hzx122.github.io/sage-rl/)
- DeepSeek-R1: [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- GRPO (DeepSeekMath): [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
- GSPO: [https://arxiv.org/abs/2507.18071](https://arxiv.org/abs/2507.18071)
- Don't Overthink It: [https://arxiv.org/abs/2505.17813](https://arxiv.org/abs/2505.17813)
- Distilling System 2 into System 1: [https://arxiv.org/abs/2407.06023](https://arxiv.org/abs/2407.06023)
- Let's verify step by step (PRM): [https://arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050)
- RLVR incentivizes correct reasoning: [https://arxiv.org/abs/2506.14245](https://arxiv.org/abs/2506.14245)
- DeepScaleR: [https://github.com/agentica-project/rational-lm](https://github.com/agentica-project/rational-lm)
- Qwen3: [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
- AdaptThink: [https://arxiv.org/abs/2505.13417](https://arxiv.org/abs/2505.13417)
- ThinkPrune: [https://arxiv.org/abs/2504.01296](https://arxiv.org/abs/2504.01296)
- GFPO: [https://arxiv.org/abs/2508.09726](https://arxiv.org/abs/2508.09726)
- Inference-time scaling: [https://arxiv.org/abs/2504.00294](https://arxiv.org/abs/2504.00294)
- Best-first beam search (Meister et al.): [https://aclanthology.org/2020.tacl-1.16/](https://aclanthology.org/2020.tacl-1.16/)
- verl framework: [https://arxiv.org/abs/2409.19256](https://arxiv.org/abs/2409.19256)

---

# Does Your Reasoning Model Implicitly Know When to Stop Thinking - 深度解析

## 1. 核心Thesis与直觉构建

这篇paper的核心insight非常优美：**LRMs内部其实"知道"什么时候该停，但当前pass@1和pass@k的sampling paradigm把这个能力给mask掉了**。

让我先build一些intuition。考虑一个LRM在MATH-500上的典型行为：模型用500 tokens就推导出了正确答案，但还要继续生成452个冗余tokens才停止。这种"overthinking"现象不是偶然的，是系统性的。

**RFCS指标**（Ratio of First Correct Step）定义为：
$$\text{RFCS} = \frac{\text{first correct step index}}{\text{total reasoning steps}}$$

变量说明：
- numerator：first correct step index 表示模型第一次产生正确答案的step序号
- denominator：total reasoning steps 表示总reasoning step数
- step通过"\n\n"分割

RFCS < 1 意味着模型在中间某step就已经答对了，但还继续reasoning。Figure 3的统计显示：
- DS-1.5B：超过一半的correct responses中RFCS < 1
- DeepScaleR：即使经过大量post-training，RFCS仍未显著改善
- Qwen3-8B：更advanced的reasoning capability也没能解决这个问题

这说明overthinking不是"模型不够强"的问题，而是**sampling paradigm本身的结构性缺陷**。

## 2. Pass@k vs Pass@1的Dilemma

### Pass@k的观察
先前工作的关键发现（来自Balachandran et al. 2025, Hassid et al. 2025）：
- DeepSeek-R1在AIME 2025上的response比Claude 3.7 Sonnet长5×，但accuracy相当
- QwQ-32B的shortest responses比random sampling高2 percentage points，但用token数少31%
- AIME 2025上，72%的问题中longer response更可能错误

这告诉我们：**CoT length超过某个threshold后，accuracy不再提升，反而下降**。换句话说，模型capability的upper bound早就达到了，只是sampling没找到这些"最优的短chain"。

### Pass@1的根本问题
当前pass@1 inference（无论是greedy还是random sampling）都是基于**next-token probability**来决策的。但paper在Section 4.2揭示了一个关键现象：

**high-confidence reasoning path的ending token可能next-token probability较低**。

这点很重要。直觉是：模型在reasoning过程中累积的"宏观信心"（cumulative log-probability Φ）和单步的"微观信心"（next-token log-probability φ）可能不一致。当一个reasoning branch在宏观上是高信心的，模型可能很自然地想到""应该出现，但因为next-token probability被周围的token distribution稀释，""的rank可能不高。

## 3. Φ的数学定义与角色

**Cumulative log-probability** 定义：
$$\Phi(\mathbf{y}_{\le k}) = \frac{1}{k} \sum_{i=1}^{k} \phi(y_i; \mathbf{y}_{<i})$$

其中：
- $\mathbf{y}_{\le k}$：从开始到第k个token的序列
- $k$：当前token位置（序列长度）
- $\phi(y_i; \mathbf{y}_{<i})$：第i个token $y_i$在给定前缀 $\mathbf{y}_{<i}$ 下的log-probability
- 求和从1到k，除以k做长度归一化

**Next-token log-probability**：
$$\phi(y_i; \mathbf{y}_{<i}) = \log \pi_\theta(y_i \mid \mathbf{y}_{<i}, \mathbf{x})$$

- $\pi_\theta$：参数为θ的policy（language model）
- $\mathbf{x}$：输入query
- $\mathbf{y}_{<i}$：第i个token之前的所有token

**关键intuition**：Φ衡量的是整条path的"平均信心"，而φ只看下一步。在beam search中我们用φ做pruning，但 Φ 才能捕获path-level的confidence。

## 4. TSearch算法详解（Token-wise Exploration）

### 算法核心
TSearch(m, r) 是paper首先提出的token-wise exploration算法：
- $m$：exploration width (EW)，beam数量
- $r$：最终返回的completion数量
- $T_{max}$：max step budget

**Algorithm流程**：

**Step 1: Candidate token generation**
给定 $m$ 个candidates $Y_{i-1} = \{\mathbf{y}_{\le i-1}^{(1)}, \ldots, \mathbf{y}_{\le i-1}^{(m)}\}$，对每个candidate选top-2m tokens：

$$\mathcal{T}^{(j)} = \text{Top}_{2m}(\{y_i \mid y_i \in \mathcal{V}\}; \phi(\cdot; \mathbf{y}_{\le i-1}^{(j)}))$$

- $\mathcal{V}$：vocabulary
- $\text{Top}_{2m}$：按φ score降序排列，取前2m个
- $j \in [1, m]$：第j个beam

**Step 2: Expand candidates**
产生 $2m \times m = 2m^2$ 个candidates：
$$\hat{Y}_i = \{\mathbf{y}_{\le i}^{(j,k)} \mid j \in [m], k \in [2m]\}$$
$$\mathbf{y}_{\le i}^{(j,k)} = \mathbf{y}_{\le i-1}^{(j)} \oplus y_i^{(j,k)}$$

- $\oplus$：concatenation
- $y_i^{(j,k)} \in \mathcal{T}^{(j)}$：第j个beam的第k个token候选

**Step 3: Retain top-m by Φ**
$$Y_i = \text{Top}_m(\{\mathbf{y}_{\le i}^{(j,k)} \mid j \in [m], k \in [2m]\}; \Phi)$$

注意：这里用 Φ 而不是 φ 来prune。

### Exploration Termination with TR
**Tolerance accept rank Ratio (TR)** = $h/2m$，其中 $h \in \{1, 2, \ldots, 2m\}$ 是hyperparameter。

当candidate $\mathbf{y}_{\le i}^{(j,k)}$ 的token $y_i^{(j,k)}$ 是 ``，且其rank在top-h内时，加入completion set $\mathcal{O}$；否则discard。

当 $|\mathcal{O}| \ge r$ 时终止。如果到 $T_{max}$ 仍然 $|\mathcal{O}| < r$，则补足top-$(r - |\mathcal{O}|)$ 的unfinished sequences。

### TSearch w/ φ vs TSearch w/ Φ的对比
**TSearch w/ φ**（ablation）：用 φ 替代 Φ 做pruning：
$$Y_i = \{\mathbf{y}_{\le i}^{(j,k)} \mid (j,k) \in \arg\text{Top}_m(\{y_i^{(j,k)} \mid j \in [m], k \in [2m]\}; \phi(\cdot; \mathbf{y}_{\le i-1}^{(j)}))\}$$

Figure 4的实验结果非常striking：
- **TSearch w/ Φ**：增大m → response变短 + accuracy提升（同步发生！）
- **TSearch w/ φ**：增大m → response快速collapse + accuracy急剧下降

这个对比证明：**是 Φ 在做"该停就停"的判断**，而单步 φ 只会"贪婪地选最短路径"，导致length collapse。

## 5. Observation 2的精细分析

Figure 6展示了一个非常counterintuitive的现象：

当一个reasoning branch被 Φ 保留下来（说明整条path的cumulative confidence高），它结束时 `` token **在next-token probability排序中可能很低**。

但在candidate set $\mathcal{T}$ 中按 Φ 排序时，`` **总是排第一**！

这个现象解释了为什么greedy/random sampling会miss掉这些高效path：
- 它们只看next-token，看到 `` 的probability低就不选
- 但从path-level看，这正是该停的时候

Figure 5进一步证实：TSearch w/ Φ 发现的 `` 在candidate set中的average rank ratio 接近0（即top rank），而 TSearch w/ φ 发现的 `` rank ratio逐渐上升（说明越搜索越不自信）。

## 6. SAGE算法（Step-wise版本）

### 从TSearch到SAGE的简化
基于Section 4.2的observation（Φ present时 `` 总是top rank），TSearch可以做大幅简化：

**简化1**：从token-wise expansion改为**step-wise expansion**
$$\mathbf{y}_{\le i}^{(j,k)} = \mathbf{y}_{\le i-1}^{(j)} \oplus \mathbf{r}_i^{(j,k)}, \quad \mathbf{r}_i^{(j,k)} \in \mathcal{R}^{(j)}$$

- $\mathbf{r}_i^{(j,k)}$：完整的reasoning step（一个或多个token）
- $\mathcal{R}^{(j)}$：从policy $\pi_\theta$ 用vanilla random sampling采样的2m个steps

**简化2**：不再需要TR hyperparameter
- 终止条件简化为：如果某个step以 `` 结尾，直接加入completion set $\mathcal{O}$
- 不再做rank tolerance判断

### SAGE的伪代码逻辑
从Appendix E的代码看：
1. **Initialization**：每个prompt生成一个ExplorationInstance
2. **Step-wise exploration** (max_step_num次循环)：
   - 对每个active instance，每个candidate采样2m个next steps
   - 用 `_score_candidate` 函数（即 Φ）排序
   - 保留top-m candidates
3. **Completion detection**：检查step的最后一个token是否是 ``
4. **Final answer generation**：对top candidates greedy生成answer

```python
def _score_candidate(self, candidate):
    return candidate.cum_logprob / len(candidate.tokens)
```

这就是 Φ 的实现：cumulative log-prob除以token数。

## 7. SAGE-RL训练框架

### Mixed Sampling的核心idea
SAGE-RL只修改RLVR的rollout阶段：

给定group size $G$（paper用 $G=8$），SAGE-RL用 SAGE(m, r) 生成 $r$ 个completions（paper用 SAGE(2,2)，即2个），其余 $G-r$ 个用standard random sampling。

### GRPO与SAGE-GRPO的目标函数对比

**GRPO objective**：
$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|x)} \left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min(w_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(w_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,t})\right]$$

变量说明：
- $\mathcal{D}$：training data distribution
- $G$：group size
- $|y_i|$：第i个response的token长度
- $w_{i,t}(\theta)$：importance ratio（详见下式）
- $\hat{A}_{i,t}$：advantage estimate
- $\varepsilon$：clip range（PPO-style）
- $\min(\cdot, \cdot)$：pessimistic bound

**Importance ratio与advantage**：
$$w_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}$$
$$\hat{A}_{i,t} = \hat{A}_i = \frac{r(x, y_i) - \text{mean}(\{r(x, y_i)\}_{i=1}^G)}{\text{std}(\{r(x, y_i)\}_{i=1}^G)}$$

- $\pi_\theta, \pi_{\theta_{\text{old}}}$：new和old policy
- $r(x, y_i)$：rule-based reward
- advantage是group-relative的归一化reward

**SAGE-GRPO objective**：
$$\mathcal{J}_{\text{SAGE-GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\left(\underbrace{\sum_{i=1}^r \cdots}_{\text{SAGE}(m,r)} + \underbrace{\sum_{i=r+1}^G \cdots}_{\text{Random sampling}}\right)\right]$$

- 前 $r$ 个用SAGE采样的responses
- 后 $G-r$ 个用random sampling的responses
- 其他部分与GRPO完全一致

### GSPO与SAGE-GSPO
GSPO（Zheng et al. 2025a）用**sequence-level** importance ratio：

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}\right)^{\frac{1}{|y_i|}} = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}\right)$$

- 整个sequence的likelihood ratio，再开 $|y_i|$ 次方
- 相当于sequence-level geometric mean of token-level ratios

paper发现SAGE-GSPO比SAGE-GRPO略好，原因是：
- GRPO的token-level importance sampling在SAGE采样的sequence上容易触发clipping
- 因为SAGE采样的sequence的old policy probability可能比random sampling低（毕竟是用 Φ 选的，不是greedy）
- GSPO的sequence-level averaging缓解了这个问题

## 8. 关键实验结果深度分析

### Table 2/4 - 主实验结果

让我重点看几个关键数据点：

**DS-1.5B + SAGE-GRPO on MATH-500**：
- Pass@1: 83.2 → 84.8 (+1.6%)
- LEN: 4882 → 2915 (-40.3%)
- TE: 17.0 → 29.1 (+70.7%)

**Qwen3-8B + SAGE-GSPO on AIME 2025**：
- Pass@1: 67.3 → 66.0 (-1.3%)
- LEN: 18342 → 9183 (-49.9%)
- TE: 3.67 → 7.19 (+95.9%)

**Qwen3-8B + SAGE-GSPO on Minerva**：
- Pass@1: 51.8 → 53.7 (+1.9%)
- LEN: 7358 → 3363 (-54.3%)
- TE: 7.04 → 16.0 (+126.8%)

**Pattern分析**：
1. **Weak model (DS-1.5B)** + **easy dataset (MATH-500)**：主要收益在length reduction
2. **Strong model (Qwen3-8B)** + **hard dataset (AIME)**：pass@1几乎不变，主要收益在efficiency
3. **Strong model** + **medium dataset (Minerva)**：pass@1和efficiency双提升

这个pattern与paper Section 5.2的结论一致："SAGE prioritizes performance for strong models and hard datasets, and efficiency for weaker models and simple datasets."

### Training Dynamics (Figure 9)
SAGE-RL vs vanilla RLVR的关键差异：
- **pass@1**：SAGE-RL提升更快且更高
- **response length**：SAGE-RL下降更明显
- **entropy**：SAGE-RL下降更显著（说明policy更confident）
- **KL divergence**：SAGE-RL上升更快（说明policy deviate from initial更多）

最后一点很有趣：**更大的KL意味着SAGE-RL让model做了更大的"更新"**。直觉是：SAGE发现的efficient reasoning pattern与原始model的distribution差距较大，需要更大的update才能学到。这也解释了为什么SAGE-RL能学到的pattern是random sampling学不到的。

### Hyperparameter Sensitivity (Table 5)

| Setting | MATH-500 Pass@1 | MATH-500 LEN | AIME 2024 Pass@1 | AIME 2024 LEN |
|---------|-----------------|--------------|------------------|---------------|
| GRPO | 83.6 | 3907 | 28.3 | 8767 |
| SAGE(1,1)-GRPO | 84.0 | 3416 | 28.3 | 7979 |
| SAGE(2,1)-GRPO | 84.2 | 2952 | 28.5 | 7308 |
| SAGE(2,2)-GRPO | 84.8 | 2915 | 28.8 | 7243 |

**关键发现**：
- **r的影响小**：从1到2，performance提升很小。因为相似trajectory带来的额外信息少。
- **m的影响大**：从1到2，performance和length reduction都有显著提升。m=1基本退化为vanilla GRPO。

Figure 12的training dynamics也证实：SAGE(2,1)和SAGE(2,2)的entropy/KL曲线几乎重合，但都与SAGE(1,1)和vanilla GRPO明显不同。

### Time Complexity Analysis (Figure 14)
- SAGE本身比Degrade SAGE慢（受限于8 GPU的memory constraint）
- 但**SAGE-RL tuned models在pass@1 inference时显著更快**：reduce latency 28.7%~40%+
- 因为tuned model的response更短，且KV cache预填充后每token延迟恒定

## 9. Beam Search vs TSearch的关键差异

Appendix B详细对比了TSearch w/ Φ 和vanilla beam search：

| Method | DS-1.5B ACC | DS-1.5B LEN | Qwen3-8B ACC | Qwen3-8B LEN |
|--------|-------------|-------------|--------------|--------------|
| Greedy | 0.81 | 4216 | 0.82 | 4505 |
| Random | 0.81 | 4142 | 0.82 | 4526 |
| Beam Search (4,4) | 0.82 | 4472 | 0.84 | 4655 |
| TSearch w/Φ (4,4) | 0.84 | 2972 | 0.89 | 2946 |

TSearch w/ Φ 同时实现更高ACC和更短LEN。Figure 11解释了原因：

**Case A**：beam search在早期discarded一个 `` 出现的branch（因为整体 Φ 不够高），但实际上这个branch是最优的。

**Case B**：一个 `` branch被保留了一段时间，但后续expansion时被其他更"长"但 Φ 更高的branch挤掉了。

TSearch的核心区别：**遇到 `` 就立即接受**，不再继续expand，避免了"被长sequence挤掉"的问题。

## 10. Case Study深度分析

### Case 1 (Figure 19) - 买领带问题
**Original DS-1.5B** (957 tokens)：包含完整推导 + 大段double-check + 重复确认
**SAGE-GRPO-DS-1.5B** (467 tokens)：直接推导 + 直接结论

SAGE-RL学到的pattern是：**完成必要推导后立即stop，不再反复验证**。

### Case 2 (Figure 20) - 极坐标转换
**Original** (712 tokens)：推导 + 双重验证 + 重复
**SAGE-GRPO** (短得多)：直接推导到结论

这两case说明：**overthinking的本质是"不必要的self-verification"**，SAGE-RL通过RL的advantage机制让model学到"足够confident时不需重复验证"。

## 11. 我的intuition构建与相关联想

### (1) 这与mode collapse/length collapse的关系
SAGE通过 Φ 而非 φ 做pruning，避免了单纯的length minimization collapse。这让我想到：
- **Constrained sequence generation**中的length normalization问题
- **CTC**中的blank token机制有类似哲学："够了就停"
- **Speculative decoding**中的draft model verification也有类似confidence判断

### (2) 与Process Reward Model (PRM)的关系
SAGE本质上是一种**implicit process reward**：用model自己的 Φ 作为process-level confidence。这与Lightman et al. 2023 (Let's verify step by step)的PRM有哲学相似性，但：
- PRM需要额外训练一个reward model
- SAGE用policy model自己的 Φ 作为proxy
- 这可能解释了为什么SAGE-RL稳定：没有引入额外的model bias

### (3) 与CTO/QC (Constrained Thinking Optimization)的关系
近期有paper用explicit length penalty（如L1, LC-R1, ThinkPrune）。这些方法的问题：
- Length penalty是global的，无法区分"必要长"和"冗余长"
- 容易reward hacking：model学会写废话filler

SAGE-RL的优势：**不修改reward function，只改rollout**。让model自己通过advantage学习"何时该长何时该短"。

### (4) 与OOD detection/Selective prediction的关联
SAGE的 Φ 与OOD detection中的**confidence calibration**有深层联系：
- High Φ 但low next-token `` probability：类似"known unknowns"
- Model在path level知道该停，但在token level被noise掩盖

这与DeepMind的Selective Prediction, Anthropic的Constitutional AI中的self-reflection机制都相关。

### (5) 与Test-time compute scaling的关系
最近test-time scaling的研究（Snell et al. 2024, OpenAI o1/o3）显示：通过更多sampling可以得到更好的答案。SAGE从另一角度证明：**不是sampling更多，而是smarter sampling**。

这让我想到：
- **Best-of-N** vs **Self-consistency** vs **SAGE**
- SAGE是"在搜索空间内找high-confidence short path"
- 本质上是**Relevance-based pruning**，而不是uniform exploration

### (6) 与Speculative Decoding的有趣parallel
Speculative decoding用draft model产生token，target model验证。SAGE用policy的 Φ 作为"验证信号"，在path level做accept/reject决策。
- Speculative: token-level accept/reject
- SAGE: step-level accept/reject基于path confidence

### (7) Information Theory视角
一条reasoning chain的信息量可以看作：
$$I(y; \text{answer}) = H(\text{answer}) - H(\text{answer} \mid y)$$

冗余tokens是那些 $H(\text{answer} \mid y_{<t}) \approx H(\text{answer} \mid y_{<t-1})$ 的tokens，即不进一步减少answer不确定性的tokens。

SAGE的 Φ 可能是这种"information gain"的implicit proxy：高 Φ path意味着每token都在有效推进，低 Φ path有大量"无效推进"。

### (8) 与Curriculum Learning的关联
Figure 13显示：SAGE-GRPO在hard level (4-5)上比GRPO有显著更大的improvement。这让我联想到：
- Easy problem：model本来就confident，SAGE收益主要在length
- Hard problem：model的"何时停"判断更uncertain，SAGE帮助model识别真正confident path
- 这与Curriculum Learning中"hard example mining"有相似哲学

### (9) 与EM (Expectation-Maximization)的隐喻
SAGE-RL的训练过程有点像EM：
- **E-step**：SAGE找出"最优latent variable"（即efficient reasoning path）
- **M-step**：RL更新policy让random sampling也能sample到这些path

这个视角下，SAGE-RL是一种**self-distillation via RL**，但distillation的不是answer而是reasoning pattern。

### (10) 限制与潜在问题
paper没深入讨论的：
- **Computational cost**：虽然paper说tuned model快，但training时SAGE sampling比random慢
- **Generalization**：只在math上验证，code, commonsense reasoning是否同样work？
- **Failure mode**：当SAGE找到的"短path"实际是错误的shortcut怎么办？paper的rule-based reward可能不够capture这种subtle error
- **Φ的局限性**：cumulative log-prob是均匀加权的，是否应该用discounted sum（recent tokens权重更高）？

## 12. 总结：核心贡献与限制

### 核心贡献
1. **Empirical finding**：LRMs隐式知道何时停，但被sampling paradigm掩盖
2. **Methodological**：SAGE用 Φ 做step-level exploration，无需修改training objective
3. **Practical**：SAGE-RL在6个benchmark上consistently improve efficiency + accuracy

### 可能的extension方向
1. **Multi-modal reasoning**：vision-language model的CoT是否也有此现象？
2. **Code generation**：是否可以用step-wise SAGE on code reasoning?
3. **Adaptive SAGE**：根据problem difficulty动态调整m
4. **Combination with PRM**：用external PRM增强 Φ 的判断
5. **Theoretical analysis**：为什么LRM会"隐式知道"何时停？是pre-training data的pattern还是RL训练的emergent behavior？

## References

- Paper本身: [https://hzx122.github.io/sage-rl/](https://hzx122.github.io/sage-rl/)
- DeepSeek-R1: [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- GRPO (DeepSeekMath): [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
- GSPO: [https://arxiv.org/abs/2507.18071](https://arxiv.org/abs/2507.18071)
- Let's verify step by step (PRM): [https://arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050)
- Don't Overthink It: [https://arxiv.org/abs/2505.17813](https://arxiv.org/abs/2505.17813)
- DeepScaleR: [https://github.com/agentica-project/rational-lm](https://github.com/agentica-project/rational-lm)
- Qwen3: [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
- DAPO: [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)
- Best-first beam search: [https://aclanthology.org/2020.tacl-1.16/](https://aclanthology.org/2020.tacl-1.16/)
- verl framework: [https://arxiv.org/abs/2409.19256](https://arxiv.org/abs/2409.19256)
- vLLM: [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- AdaptThink: [https://arxiv.org/abs/2505.13417](https://arxiv.org/abs/2505.13417)
- ThinkPrune: [https://arxiv.org/abs/2504.01296](https://arxiv.org/abs/2504.01296)
- LC-R1: [https://arxiv.org/abs/2506.14755](https://arxiv.org/abs/2506.14755)
- Efficient Reasoning: [https://arxiv.org/abs/2502.04463](https://arxiv.org/abs/2502.04463)
- GFPO: [https://arxiv.org/abs/2508.09726](https://arxiv.org/abs/2508.09726)
- Inference-time scaling: [https://arxiv.org/abs/2504.00294](https://arxiv.org/abs/2504.00294)
- RLVR incentivizes correct reasoning: [https://arxiv.org/abs/2506.14245](https://arxiv.org/abs/2506.14245)

---

**最终intuition**：这篇paper让我对LRM的"internal representation"有了新认识。Policy model的next-token distribution表面上看起来很noisy，但path-level的cumulative confidence却很clean。这暗示LRM的"decision making"可能在path level而非token level发生，而我们的sampling paradigm一直被困在token-level的局部视野中。SAGE本质上是一个**macro-scale sampler**，它把modeling的"宏观意图"从"微观noise"中提取出来。

更深一层：这可能暗示LRM的reasoning是**hierarchical**的——token level的uncertainty掩盖了step level的confidence，step level的confidence又掩盖了path level的intent。每一层的"knowing"都被下一层的"sampling noise"给mask掉了。SAGE只unmask了step level，path level的intent可能还需要更macro的sampler来unveil。这可能是test-time compute scaling的下一个frontier。
