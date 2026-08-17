---
source_pdf: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs
  Beyond the Base Model.pdf
paper_sha256: 76139ea0ba0e9155569c7aa7b8153502f6a5b23fed717af0ca9934ec9d764f9a
processed_at: '2026-08-03T23:00:19-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲清楚这篇paper

## 一句话版本

**RLVR没教会模型任何新东西，它只是把模型本来就会的东西"挑出来放大"了。**

---

## 1. 先建立intuition：模型是个什么"东西"

你训练一个LLM，pretrain完之后，模型有一个"sampling distribution"——给它一道题，它会在所有可能的token sequence上有个概率分布。这个分布里：
- 有些sequence是correct reasoning path
- 有些是incorrect
- 有些是nonsense

关键是：**这个分布本身，定义了模型"知道什么"**。分布里有的path，模型"会"；分布里没有的path，模型"不会"。

现在你用RLVR训这个模型。question是：训完之后，模型"会"的东西变多了吗？还是只是把本来就会的东西"放大"了？

---

## 2. 怎么测"模型会什么"——pass@k的intuition

最naive的测法：给模型一道题，sample 1次，看对不对。这是pass@1。但pass@1有个问题：模型可能"会"这道题，只是1次采样没采到correct path。

更好的测法：给模型256次机会，只要其中1次对了，就算"会"。这是pass@256。

**pass@k的intuition**：pass@k衡量的不是"模型平均能做对多少"，而是"模型的reasoning boundary有多宽"。pass@1高，说明模型sampling效率高；pass@k高（k大），说明模型reasoning coverage广。

类比：pass@1是"这个学生考试平均能考多少分"，pass@256是"这个学生如果允许重考256次，能做对多少道题"——后者反映的是学生的"潜力边界"。

---

## 3. 核心实验发现

### 发现1：pass@1上升，pass@256下降

Figure 2的核心pattern，我用语言描述：

给一道数学题，
- base model采样1次，对的概率是30%
- RLVR model采样1次，对的概率是60%（RLVR"看起来"有用）
- 但采样256次，
- base model能解80%的题
- RLVR model只能解70%的题

**RLVR把"1次就做对"的概率提高了，但把"256次里至少做对1次"的coverage降低了**。

### 发现2：RLVR解的题，base model都能解

Table 2：AIME24上，把30道题分成4类——
- base能解 + RLVR能解：63.3%
- base能解 + RLVR不能解：13.3%
- base不能解 + RLVR能解：**0.0%**
- base不能解 + RLVR不能解：23.3%

**第三行是0%——RLVR没有解出任何base model解不了的题**。RLVR解的题，是base model本来就能解的题的subset。

### 发现3：RLVR的输出在base model眼里很"自然"

用base model去算RLVR输出的perplexity——发现RLVR生成的response，在base model的分布里是高likelihood的。意思是：**RLVR生成的那些reasoning path，base model本来就容易生成**。RLVR只是"偏好"了base model分布里的某个region，没有创造新region。

对比：拿OpenAI-o1的输出算base model的perplexity——显著高于base model自己的输出。o1的reasoning pattern在Qwen2.5-7B的prior里是low-likelihood的——这才是"genuinely new reasoning"。

---

## 4. 为什么会这样——一句话intuition

**RLVR用binary reward做policy gradient，本质是在prior内部做probability redistribution，不是prior外部的exploration**。

具体机制：
- policy gradient对每个sampled response，根据reward调整likelihood
- 如果response correct（reward=1）：增加这个response的likelihood
- 如果response incorrect（reward=0）：降低这个response的likelihood

问题在于：**能被sampled到的response，都是prior里high-likelihood的**。prior里low-likelihood的correct path，几乎不会被sample到，所以拿不到gradient signal，likelihood不会被抬高。

结果：RLVR把prior里high-likelihood的region重新分配——把mass从incorrect region移到correct region。这是个**zero-sum game within prior**：correct region的mass增加，是以incorrect region的mass减少为代价的。但prior的整体形状没变——prior外部的correct path仍然是low-likelihood。

**类比**：base model是个图书馆，里面有100万本书。RLVR做的事情是——把"正确答案"的书从角落搬到前台，让读者更容易拿到。但RLVR没有写任何新书。而且搬运的过程中，有些原本在前台的"正确但冷门"的书，反而被挤到角落去了——这就是pass@256下降的原因。

---

## 5. Distillation为什么不一样

Distillation用的是cross-entropy loss on teacher的output sequence：

$$\mathcal{L}_{\text{distill}} = -\sum_t \log P_{\text{student}}(y_t^{\text{teacher}} | x, y_{<t}^{\text{teacher}})$$

这里$y_t^{\text{teacher}}$是teacher model生成的token，$P_{\text{student}}$是student model的概率。

这个loss做的事情：**强行抬高teacher output的likelihood**，即使teacher output在student的prior里是low-likelihood的。

Figure 7的实验：distill model的pass@k曲线**始终高于**base model——即使$k$很大也不被反超。distillation真正"教会"了student一些base model不会的东西。

**直觉**：RLVR只能"放大"prior里已有的signal；distillation能"注入"prior里没有的signal。这是质的区别。

---

## 6. 为什么传统RL能发现新策略，RLVR不能

传统RL（AlphaGo Zero）和RLVR有两个key differences：

### Difference 1：Action space大小

- Go：action space是361（棋盘361个点）
- Atari：action space是~18（离散低维）
- LLM：action space是$|V|^T$，$|V|$是vocab size（~32k），$T$是sequence length（~1000+）

LLM的action space是**exponentially vast**。在这种空间里，naive token-level exploration几乎不可能找到correct path——随机sample一个1000-token的sequence，对的概率是$|V|^{-1000}$，astronomically small。

### Difference 2：Starting point

- AlphaGo Zero：从random init开始，没有prior。agent必须自己explore，任何discovered strategy都是"new"。
- RLVR：从pretrained base model开始，有strong prior。prior引导exploration走向"reasonable"region，使得RL tractable。**但prior也cap了exploration的范围**——偏离prior的sample大概率是nonsense，拿不到reward，被policy gradient压低。

**Prior是double-edged sword**：它让RLVR可行（没有prior，RLVR根本train不起来），但它也cap了RLVR的ceiling（RLVR只能在prior内部做redistribution）。

---

## 7. 各种ablation都在确认这个intuition

### KL penalty让事情更糟

加KL penalty（防止policy偏离base model太远）——pass@256更低。因为KL explicit惩罚偏离，连"偏离但正确"的paths也被压低了。**KL把prior的"边界"焊死了**。

### 加大rollout number n

把n从8增到32——pass@k略升，但仍被base model反超。更多rollouts = 更好的group statistics = advantage估计更准 = 探索略广。但scale rollout alone不够——prior的cap还在。

### 加temperature match entropy

把RLVR model的temperature调高，让它的output entropy匹配base model——pass@k比低temperature时略好，但仍低于base model。

**Intuition**：entropy reduction是coverage narrowing的**一部分原因**，但不是全部。RLVR还做了"distribution reshape"——把probability mass从incorrect region移到correct region，这种reshape即使entropy不变也减少coverage。因为"correct region"的面积本来就比"整个reasonable region"小，concentrate mass到correct region必然牺牲coverage。

### Training dynamics

Table 4：从step 150到step 450，pass@1从26.1→42.5（+16.4），pass@256从66.3→64.3（-2.0）。

**训练越久，sampling efficiency越好，reasoning boundary越窄**。这是trade-off的direct evidence。RLVR本质上在做"分布sharpening"——这是个持续的过程，越sharpen越窄。

### Frontier scale也成立

Magistral-Medium（Mistral的pure-RL reasoning model，接近DeepSeek-R1水平）vs. Mistral-Medium-3 base：
- pass@1：Magistral在AIME24多解~7题
- pass@k增大：gap持续缩小

**结论在frontier-scale model上仍然成立**，不是small model的artifact。

---

## 8. 各种RL算法都一样

paper测了6种RL算法：PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO。

定义Sampling Efficiency Gap：

$$\Delta_{\text{SE}} = \text{pass@256}_{\text{base}} - \text{pass@1}_{\text{RL}}$$

这是"RL model的pass@1离base model的pass@256 upper bound有多远"。

结果：所有算法的$\Delta_{\text{SE}}$都在40+个百分点，彼此差异<2个百分点。

**Intuition**：算法层面的tweak（PPO vs GRPO vs Reinforce++ vs ...）改变的是"如何估计advantage"和"如何更新policy"的细节，但不改变一个fundamental fact——**所有这些都是policy gradient方法，都在prior内部做redistribution**。换个advantage estimator不会让suddenly能explore prior外部。

---

## 9. 那怎么办——Future Directions的intuition

paper最后提了4个方向，我用intuition讲：

### Direction 1：High-level abstraction exploration

AlphaEvolve在program-level做evolution——action space从token-level变成program-level，drastically smaller，exploration tractable。

类比reasoning：与其在token level explore，不如在"reasoning strategy level" explore。比如explore不同的problem decomposition方式、不同的subgoal setting方式。这相当于在更abstract的空间里search，action space小很多。

### Direction 2：Curriculum learning

paper观察到：current RLVR data里偶尔有hierarchical relationship（easy subproblem → hard parent problem），但没有被deliberately exploit。

Intuition：如果model在hard problem上的success rate是0，RLVR拿不到任何positive reward，学不到任何东西。但如果先让model在easy subproblem上train，学到meta-skill（比如"set up equation"），再transfer到hard problem，success rate可能从0变成non-zero——这时RLVR才能拿到meaningful reward，开始学习。

这是hierarchical reduction of exploration space。

### Direction 3：Process reward

Binary outcome reward的credit assignment problem：一个200-token的CoT，只有最终答案对不对一个signal。model不知道"哪一步做对了""哪一步做错了"。

Process reward model（PRM）提供step-level credit。好处：
- "Prior内部但low-likelihood的正确path"能拿到中间reward，从而被reinforce
- Model能学到"哪类step是productive的"，而不仅是"哪类final answer是对的"

### Direction 4：Agentic RL / Era of Experience

Silver & Sutton的"Welcome to the era of experience"论点：current RLVR是single-turn的——给prompt，生成response，结束。但IMO-level reasoning需要iterative refinement with feedback。

Multi-turn agentic RL：model可以
- 用tool（calculator, search, code interpreter）
- 做hypothesis testing
- 根据feedback修改approach

这些interaction能generate novel experience——这些experience是prior里没有的。这才是能escape prior的path。

**Intuition**：single-turn RLVR像是一个学生闭卷考试，只能从脑子里"调取"已有知识。multi-turn agentic RL像是一个学生开卷考试，可以查资料、做实验、修改答案——后者能"学到"前者学不到的东西。

---

## 10. 我（作为Andrej）会怎么think about这个

这篇paper让我想到几个更深的问题：

### 问题1：Pretraining本身就是"distillation from internet"

Pretrain一个LLM，本质是next-token prediction on internet text。Internet text里有mathematical reasoning、code、logic——pretrain把这些patterns distill进model的weights。

所以base model的"reasoning capacity"，本质上是internet text里reasoning patterns的reflection。base model能解AIME24的63.3%——意味着internet上已经有足够多的olympiad-level math reasoning text，让Qwen2.5-7B pretrain后就能sample出correct path。

**RLVR不能超越base model，等于说RLVR不能超越internet text的reasoning patterns**。这很reasonable——token-level RLVR没有新的information source。

### 问题2：RLVR的"value"到底在哪

paper说RLVR没创造新reasoning，但paper也说RLVR大幅提升pass@1。这两个finding不矛盾——**RLVR的value是"sampling efficiency"**。

实际deploy LLM时，你不会sample 256次取最好——太贵。你sample 1次或几次。RLVR让这1次sample更可能hit correct path。这是巨大的practical value。

但paper的point是：**别把"sampling efficiency"和"reasoning capacity"混淆**。RLVR提升的是前者，不是后者。DeepSeek-R1 paper的narrative说RLVR让model"self-improve and acquire novel reasoning"——这个narrative在mechanism层面是misleading的。

### 问题3：GPT-o1/DeepSeek-R1的"reasoning ability"从哪来

paper的finding暗示：这些frontier reasoning model的"reasoning ability"，主要来自pretrain + distillation，不是来自RLVR。

DeepSeek-R1的pipeline：
1. Pretrain base model（internet reasoning patterns → weights）
2. Distill from DeepSeek-R1的long CoT data（teacher reasoning patterns → weights）
3. RLVR（sharpen distribution towards correct paths）

Step 1和2是"manifold expanding"——真正增加reasoning capacity。Step 3是"manifold-internal optimization"——提升sampling efficiency。

**所以如果要build更强的reasoning model，投资在step 1和2的ROI可能比step 3高**。更好的pretrain data（更多reasoning text）、更好的distillation teacher（更强的reasoning model），可能比更好的RLVR algorithm更有用。

### 问题4：有没有办法让RLVR真的"discover new reasoning"

paper的诊断是：token-level RLVR被prior cap住。要escape prior，需要：
- 更小的effective action space（high-level abstraction）
- 更细的reward signal（process reward）
- 更多的information source（multi-turn interaction with environment）

**Silver & Sutton的"era of experience"可能是正确的方向**：让agent在environment里interact，collect experience，这些experience是prior里没有的。这类似于AlphaGo Zero self-play——agent通过self-play generate novel game states，这些states是human knowledge里没有的。

类比LLM：让LLM在"reasoning environment"里interact——比如
- 和theorem verifier interact（prove a theorem, get feedback on which steps are valid）
- 和code interpreter interact（write code, run it, see if it works, debug）
- 和scientific simulator interact（hypothesize, simulate, observe, refine）

这些interaction generate的experience，是pretrain data里没有的——这才是能escape prior的path。

### 问题5：这和"test-time compute scaling"的关系

Brown et al. 2024（Large Language Monkeys）和Snell et al. 2024（Scaling test-time compute）的工作显示：base model通过更多sampling，能解决更多问题——pass@k随k增长。

paper的finding：base model的pass@256 > RLVR model的pass@1。这意味着：**用base model + 更多test-time sampling，可能比RLVR model + 少量sampling更便宜且更effective**。

这给了一个alternative path：与其invest在RLVR training上，不如invest在test-time compute上——用inference-time search（比如MCTS, beam search with verifier）来explore base model的distribution。这可能比RLVR更高效，因为：
- 不需要expensive RL training
- 能explore base model的full distribution，不受RLVR的narrowing影响
- 可以根据具体problem dynamically allocate compute

---

## 11. 一图总结

我画一个mental model：

```
Base Model Distribution (prior)
┌─────────────────────────────────────────┐
│  ┌─────┐    ┌─────┐    ┌─────┐         │
│  │correct│   │incorrect│  │correct│        │
│  │ path │    │ path │    │ path │        │
│  │ (high│    │ (high│    │ (low │        │
│  │  lik)│    │  lik)│    │  lik)│        │
│  └─────┘    └─────┘    └─────┘         │
│      ↑          ↓          ↑(unchanged)  │
│      │          │          │             │
│   RLVR放大    RLVR压低    RLVR够不到     │
│  (pass@1↑)  (pass@1↑)   (stays low)    │
│                                          │
│  ┌─────┐                                │
│  │correct│                              │
│  │ path │                               │
│  │ (low │                               │
│  │  lik)│                               │
│  └─────┘                                │
│      ↑(unchanged, RLVR够不到)            │
│                                          │
│  ┌─────┐    ┌─────┐                    │
│  │correct│   │correct│                   │
│  │ path │    │ path │                    │
│  │(prior│    │(prior│                    │
│  │ 外部)│    │ 外部)│                    │
│  └─────┘    └─────┘                     │
│  (RLVR完全碰不到，distillation能碰到)       │
└─────────────────────────────────────────┘

RLVR Model Distribution (after training)
┌─────────────────────────────────────────┐
│  ┌─────┐                                │
│  │correct│                              │
│  │ path │                               │
│  │ (更高 │                               │
│  │  lik)│                               │
│  └─────┘                                │
│      ↑pass@1大幅提升                     │
│                                          │
│  其他region都被压低了                      │
│  → pass@256下降                          │
│  → coverage变窄                          │
│                                          │
│  整个distribution还是在prior范围内          │
└─────────────────────────────────────────┘

Distillation Model Distribution
┌─────────────────────────────────────────┐
│  ┌─────┐    ┌─────┐    ┌─────┐         │
│  │correct│   │correct│   │correct│        │
│  │ path │    │ path │    │ path │        │
│  │ (high│    │(原本低│    │(prior│        │
│  │  lik)│    │ 被抬高)│    │ 外部 │        │
│  │       │    │       │    │被注入)│        │
│  └─────┘    └─────┘    └─────┘         │
│                                          │
│  distribution形状变了                     │
│  → pass@1和pass@256都提升                │
│  → coverage扩大                          │
│  → 真正的新reasoning capacity             │
└─────────────────────────────────────────┘
```

---

## 12. 我的take-away

1. **RLVR是"sampling efficiency optimizer"，不是"reasoning capacity expander"**。这两件事要分开claim。

2. **Base model的reasoning capacity被严重underestimated**。我们平时评估LLM用pass@1，看到base model pass@1低，以为它"不会"。但pass@256显示base model其实"会"很多——只是sampling效率低。base model像个很聪明但很absent-minded的professor，你得多问几次才能问出correct answer。

3. **Reasoning model的"reasoning ability"主要来自pretrain和distillation，不是RLVR**。RLVR是锦上添花，不是雪中送炭。

4. **要build真正能"discover new reasoning"的RL，需要paradigm shift**：从token-level到strategy-level，从binary reward到process reward，从single-turn到multi-turn agentic。token-level RLVR可能是个local optimum。

5. **Test-time compute scaling可能是比RLVR更好的investment**。base model + inference-time search，可能比RLVR model + greedy decoding更便宜且更effective。

6. **Distillation被低估了**。在RLVR hype中，distillation被认为是"次等"的方法——只是模仿teacher。但paper显示distillation能做RLVR做不到的事：expand reasoning boundary。如果要build更强的student model，找更强的teacher做distill，可能比用RLVR train更有效。

7. **这和"bitter lesson"有关吗**。Sutton的bitter lesson说：general methods that leverage computation are ultimately the most effective。RLVR是general method + computation，但paper显示它被prior cap住——所以RLVR不是"general method that leverages computation"的正确形式。正确的形式可能是：让computation去explore更大的空间（high-level abstraction, multi-turn interaction），而不是在prior内部做redistribution。

---

## References

- **Paper Project Page**: https://limit-of-RLVR.github.io
- **DeepSeek-R1**: https://arxiv.org/abs/2501.12948
- **Large Language Monkeys (pass@k scaling)**: https://arxiv.org/abs/2407.21787
- **Scaling test-time compute (Snell et al.)**: https://arxiv.org/abs/2408.03314
- **PPO**: https://arxiv.org/abs/1707.06347
- **GRPO / DeepSeek-Math**: https://arxiv.org/abs/2402.03300
- **Echo Chamber**: https://arxiv.org/abs/2504.07912
- **OAT-Zero**: https://oatllm.notion.site/oat-zero
- **DAPO**: https://arxiv.org/abs/2503.14476
- **SimpleRL-Zoo**: https://arxiv.org/abs/2503.18892
- **Silver & Sutton "Era of Experience"**: https://silverlab.media.filed.gracehalloffame.org/2025/03/Rich-Sutton-Reward-Hypothesis-Talk-2025.pdf
- **AlphaEvolve**: https://arxiv.org/abs/2506.13131
- **OpenAI o1 system card**: https://arxiv.org/abs/2412.16720
- **Kimi k1.5**: https://arxiv.org/abs/2501.12599
- **Dang et al. (diversity collapse)**: https://openreview.net/forum?id=AMiKsHLjQh
- **VeRL framework**: https://arxiv.org/abs/2409.19256
- **Reinforce++**: https://arxiv.org/abs/2501.03262
- **RLOO**: https://arxiv.org/abs/2402.14740
- **ReMax**: https://arxiv.org/abs/2310.10505
- **Magistral**: https://arxiv.org/abs/2506.10910
- **Tulu 3**: https://arxiv.org/abs/2411.15124
- **DPO**: https://arxiv.org/abs/2305.18290
- **STAR**: https://arxiv.org/abs/2203.14465
- **REST**: https://arxiv.org/abs/2308.08998
- **Absolute Zero**: https://arxiv.org/abs/2505.03386
- **AceReason Nemotron**: https://arxiv.org/abs/2505.16400
- **StepHint**: https://arxiv.org/abs/2507.02841
- **Questa**: https://arxiv.org/abs/2507.13266
- **IMO 2025 verification-and-refinement**: https://arxiv.org/abs/2507.15855
- **Rethinking Reflection (Shah et al.)**: https://arxiv.org/abs/2504.04022
- **DeepCoder**: https://arxiv.org/abs/2506.07390
- **Code-R1**: https://github.com/ganler/code-r1
- **EasyR1**: https://github.com/hiyouga/EasyR1
- **LiveCodeBench**: https://arxiv.org/abs/2403.07974
- **Omni-MATH**: arXiv:2510.04556
- **Qwen2.5**: https://arxiv.org/abs/2412.15115
- **Qwen3**: https://arxiv.org/abs/2505.09388

---

# Deep Dive: Does RLVR Really Incentivize Reasoning Beyond the Base Model?

这是清华大学LeapLab（Gao Huang组）的一篇非常扎实的empirical study，第一作者Yang Yue（乐洋）。核心claim很反直觉：**当前RLVR训练的reasoning ability，本质上是base model sampling distribution的一个sharpened subset，并没有产生真正novel的reasoning patterns**。让我系统地把intuition建起来。

---

## 1. Motivation: 为什么这个问题值得问

传统RL的success story（AlphaGo Zero, Atari DQN）里，agent通过self-play能discover超越人类的策略——这是RL的"圣杯"：exploration带来genuinely new capability。DeepSeek-R1把这套narrative搬到了LLM上，宣称RLVR能让LLM"自我进化"，emerge出reflection、enumeration、iterative refinement等行为。

但是这里有一个conceptual gap：**Go的action space是~361，Atari是离散低维，而LLM的action space是|V|^T，exponentially vast**。在这种combinatorial explosion下，naive token-level sampling exploration几乎不可能escape base model的prior——任何偏离prior的sample大概率是nonsense，拿不到positive reward。policy gradient算法只能maximize prior内部拿到positive reward的sample的log-likelihood，minimize prior外部拿到negative reward的sample的log-likelihood，结果就是policy被"压"回prior内部。

这个intuition paper用大量实验rigorously验证了。

---

## 2. 评估reasoning boundary的metric：pass@k

这是全文的核心方法论选择。greedy decoding或nucleus sampling的average score只反映average-case behavior，会underestimate model的true potential。

### 2.1 Unbiased pass@k estimator

对于问题$x_i$，采样$n$次（$n \geq k$），其中$c_i$次正确。无偏估计为：

$$\text{pass@k} := \mathbb{E}_{x_i \sim \mathcal{D}}\left[1 - \frac{\binom{n - c_i}{k}}{\binom{n}{k}}\right]$$

**变量解释**：
- $n$：每个问题采样的总次数（实验中通常是128、256或1024）
- $c_i$：问题$x_i$在$n$次采样中正确的次数，$c_i \in \{0, 1, \ldots, n\}$
- $k$：我们评估的"尝试次数"上界
- $\binom{n-c_i}{k}$：从$n-c_i$个错误sample中选$k$个的组合数——这代表"全部$k$次都错"的方式数
- $\binom{n}{k}$：从$n$次采样中选$k$次的总组合数
- 整个分数$\frac{\binom{n-c_i}{k}}{\binom{n}{k}}$就是"随机选$k$次都错"的概率
- 1减去它，就是"至少有一次对"的概率

**Intuition**：pass@k衡量的不是"模型平均能做对多少"，而是"模型在$k$次尝试内，能解决多少比例的问题"。这个值高，意味着模型的reasoning boundary宽；低，意味着boundary窄。

### 2.2 为什么不用Best-of-N或Majority Voting

Best-of-N需要一个verifier去select——但这就把"boundary评估"和"selection能力"混在一起了。Majority voting依赖frequency，对于"只有1次正确但其他都错"的hard problem会被vote掉。pass@k纯粹看"potential coverage"，这正是我们想测的reasoning capacity。

---

## 3. RLVR背景：核心算法

### 3.1 PPO的clipped surrogate

$$\mathcal{L}_{\text{CLIP}} = \mathbb{E}\left[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)\right]$$

**变量解释**：
- $r_t(\theta) = \pi_\theta(y_t | x, \mathbf{y}_{<t}) / \pi_{\theta_{\text{old}}}(y_t | x, \mathbf{y}_{<t})$：importance sampling ratio，新policy与old policy在token $y_t$处的概率比
- $A_t$：advantage，由value network $V_\phi$估计
- $\epsilon$：clip范围，通常0.1~0.2，限制policy update幅度
- $\min$操作：取clipped和unclipped的较小值，保证pessimistic bound

### 3.2 GRPO的advantage（critic-free）

$$A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

其中$\mathbf{r} = \{r_1, \ldots, r_G\}$是同一prompt的$G$个sample的reward vector。

**关键insight**：GRPO用group statistics做baseline，省去了value network。但paper后面会展示，所有这些algorithm（PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO）的$\Delta_{\text{SE}}$差异都很小——**算法层面的innovation并不能改变"被prior bound住"的本质**。

### 3.3 RLOO的leave-one-out baseline

$$A_i = r_i - \frac{1}{|\mathcal{B}|-1}\sum_{j \neq i} r_j$$

leave-one-out减小了baseline variance，但仍然是within-prior的signal。

---

## 4. 核心实验发现

### 4.1 Figure 1的搜索树图解析

左边panel画了两个problem的search tree：
- **Problem A**：base model的sampling distribution覆盖广（黑色paths多），其中包含correct paths（绿色）。RLVR训练后，distribution被"挤"向correct paths，sampling efficiency上升，但其他paths变grey（unlikely to be sampled）。这是RLVR"正面"的作用。
- **Problem B**：base model原本包含correct path，但RLVR训练后这个path被"挤没了"——distribution变narrow，correct path反而sampling不到了。这是RLVR"负面"的副作用：**coverage缩小**。

右边panel是asymptotic dynamics：横轴training steps，纵轴两个metric。pass@1（蓝色）上升，pass@256（红色）下降——**采样效率提升的代价是reasoning boundary收缩**。

### 4.2 Figure 2的pass@k曲线

横轴是$k$（log scale），纵轴是pass@k。每张子图都是同一个pattern：
- $k=1$附近：RLVR model（红/橙线）显著高于base model（蓝线）
- $k$增大到tens/hundreds：两条线交叉
- $k$继续增大：base model反超RLVR model

例如Minerva benchmark上32B model，$k=128$时base model比RLVR高约9%——意味着base model能多解9%的problems。

**这个crossover现象的intuition**：RLVR把probability mass集中到了"高reward paths"上，所以单次采样hit正确答案的概率高。但集中意味着牺牲了diversity，base model虽然单次hit概率低，但$k$次里总能"碰上"正确path的coverage更广。

### 4.3 Table 2的coverage分析

AIME24（$k=1024$）和MATH500（$k=128$）上，把problems分成4类：

| Base | RLVR | AIME24 | MATH500 |
|------|------|--------|---------|
| ✓ | ✓ | 63.3% | 92.4% |
| ✓ | ✗ | 13.3% | 3.6% |
| ✗ | ✓ | 0.0% | 1.0% |
| ✗ | ✗ | 23.3% | 3.0% |

**关键观察**：第三行（base解不出但RLVR解出）在AIME24上是**0.0%**，在MATH500上仅1.0%——**RLVR几乎从未解出base model解不了的problem**。第二行（base能解但RLVR不能）则显著存在。

这就是"subset relationship"的直接证据：**RLVR的solvable set几乎是base model solvable set的子集**。

---

## 5. Perplexity Analysis：核心证据

这是paper最elegant的分析。给定model $m$、problem $x$、response $\mathbf{Y} = (y_1, \ldots, y_T)$：

$$\text{PPL}_m(\mathbf{Y} | x) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(y_t | x, y_1, \ldots, y_{t-1})\right)$$

**变量解释**：
- $m$：计算perplexity的model（这里是base model）
- $\mathbf{Y}$：被评估的response sequence（可以来自任何source）
- $T$：response长度
- $y_t$：第$t$个token
- $P(y_t | x, y_1, \ldots, y_{t-1})$：model $m$在给定prompt和前$t-1$个token后，预测$y_t$的概率
- 整个sum是average negative log-likelihood
- exp把它从log空间拉回probability空间

**实验设计**：
- $\mathbf{Y}_{\text{base}}$：base model生成的16个responses
- $\mathbf{Y}_{\text{RL}}$：RL model生成的16个responses
- $\mathbf{Y}_{\text{GT}}$：OpenAI-o1生成的8个responses（作为"out-of-prior" reference）

计算$\text{PPL}_{\text{Base}}(\mathbf{Y}_{\text{RL}} | x)$，看RL model的输出在base model眼里"有多natural"。

**Figure 6的结果**：
- $\text{PPL}_{\text{Base}}(\mathbf{Y}_{\text{RL}} | x)$的分布**紧密贴合**$\text{PPL}_{\text{Base}}(\mathbf{Y}_{\text{base}} | x)$分布的**lower portion**
- 这意味着：**RL model生成的response，在base model的视角下，就是base model本来就容易生成的那些response**——RL只是在base model的high-likelihood region里做了selection
- 对比之下，$\text{PPL}_{\text{Base}}(\mathbf{Y}_{\text{GT}} | x)$（o1的输出）显著高于$\mathbf{Y}_{\text{base}}$的分布——这才是"out-of-prior"的pattern

### 5.1 Section C.4的perplexity evolution

paper还检查了RL训练过程中$\text{PPL}_{\text{Base}}(\mathbf{Y}_{\text{RL}} | x)$的演化：随着training steps推进，这个perplexity**逐渐下降**——意味着RLVR model的输出越来越"靠近"base model的高likelihood region，即**RLVR在sharpen prior内部的distribution，而不是expanding到prior之外**。

---

## 6. Distillation vs. RLVR：决定性对比

Figure 7比较了4个model的pass@k：
- Qwen2.5-Math-7B（base）
- Qwen2.5-Math-7B-Instruct
- Qwen2.5-Math-7B-Oat-Zero（RLVR）
- DeepSeek-R1-Distill-Qwen-7B（distillation）

**关键发现**：distill model的pass@k曲线**始终且显著**高于base model的曲线——即使$k$很大也不被反超。这意味着distillation真正把teacher（DeepSeek-R1）的reasoning patterns注入了student，**扩展了reasoning boundary**。

**Intuition**：distillation用的是cross-entropy loss on long CoT tokens，本质上是maximum likelihood on teacher的output distribution。如果teacher的某些CoT paths在student的prior里是low-likelihood的，distillation会**强行抬高这些paths的likelihood**，这就是"expanding beyond prior"。而RLVR用的是binary outcome reward，无法提供fine-grained signal去reinforce那些"prior里low-likelihood但correct"的paths。

---

## 7. Sampling Efficiency Gap $\Delta_{\text{SE}}$

paper定义了一个quantitative metric来衡量"RL算法离optimal有多远"：

$$\Delta_{\text{SE}} = \text{pass@256}_{\text{base}} - \text{pass@1}_{\text{RL}}$$

**变量解释**：
- $\text{pass@256}_{\text{base}}$：base model在$k=256$时的pass rate，作为"reasoning capacity upper bound"的proxy
- $\text{pass@1}_{\text{RL}}$：RL model的pass@1，即average-case performance
- 差值越小，意味着RL算法越接近"optimal sampling efficiency"——即用1次采样就能达到base model用256次采样的coverage

### 7.1 Table 3的算法对比

| Model | Omni-MATH-Train pass@1 | pass@256 | Omni-MATH-Test pass@1 | pass@256 | MATH500 pass@1 | pass@256 |
|-------|------------------------|----------|------------------------|----------|----------------|----------|
| Qwen2.5-7B (base) | 9.9 | 67.2 | 10.2 | 69.1 | 34.5 | 96.2 |
| GRPO | 26.1 | 66.3 | 25.1 | 68.3 | 74.4 | 97.2 |
| PPO | 27.2 | 65.8 | 26.8 | 69.2 | 75.2 | 97.2 |
| ReMax | 24.4 | 65.5 | 23.8 | 67.5 | 73.5 | 96.6 |
| RLOO | 28.6 | 66.4 | 28.1 | 69.2 | 75.0 | 97.4 |
| Reinforce++ | 28.2 | 67.7 | 28.0 | 69.7 | 75.4 | 96.8 |
| DAPO | 31.4 | 66.1 | 26.5 | 67.0 | 75.6 | 96.4 |

**Intuition分析**：
- 在Omni-MATH-Test上，base的pass@256是69.1，最好的RL算法RLOO的pass@1是28.1——$\Delta_{\text{SE}} = 69.1 - 28.1 = 41.0$，**还差41个百分点**
- 所有RL算法的$\Delta_{\text{SE}}$都在40以上，彼此差异<2个百分点
- DAPO在pass@1上略高（31.4），但它的dynamic sampling用了3~6×的samples per batch，性价比存疑
- ReMax表现最差——因为它的baseline是greedy response的binary reward，variance太大导致gradient不稳定

**结论**：**算法层面的tweak无法弥合这个gap**，需要的是paradigm shift。

---

## 8. Ablation Studies

### 8.1 KL penalty的作用（Figure 16）

加KL=0.001的模型：
- pass@1与无KL的GRPO相似
- pass@128显著更低

**Intuition**：KL penalty本来是为了防止policy偏离base model太远。但这里发现KL反而**进一步压窄了coverage**——因为它explicitly惩罚偏离，连那些"偏离但正确"的paths也被压低了。

### 8.2 Rollout number $n$的作用

把$n$从8增到32：
- pass@k在小$k$略降（因为没train够，只220 steps）
- pass@128略升

**Intuition**：更多rollouts = 更好的group statistics = advantage估计更准 = 探索更广。但即便如此，RL model仍被base model反超——**scale rollout alone不够**。

### 8.3 Temperature/Entropy matching（Figure 18）

把RLVR model的temperature调高到match base model的output entropy（比如AMC23上$T=0.9$让$E_{\text{RL}}=0.47 \approx E_{\text{base}}$）：
- pass@k比$T=0.6$时略好
- 但仍低于base model

**Intuition**：entropy reduction确实贡献了一部分coverage narrowing，但不是全部原因。即使把entropy强行拉回来，RL model still underperforms——说明RL训练还做了**分布重塑**而不仅是entropy reduction：它把probability mass从"低reward region"移到了"高reward region"，这种重塑即使entropy不变也减少coverage。

### 8.4 Training dynamics（Table 4）

| Model | Omni-MATH-Train pass@1 | pass@256 | Omni-MATH-Test pass@1 | pass@256 |
|-------|------------------------|----------|------------------------|----------|
| Qwen2.5-7B | 9.9 | 67.2 | 10.2 | 69.1 |
| GRPO-step150 | 26.1 | 66.3 | 25.1 | 68.3 |
| GRPO-step300 | 33.6 | 65.3 | 27.1 | 66.6 |
| GRPO-step450 | 42.5 | 64.3 | 28.3 | 63.9 |

**关键观察**：从step150到step450，pass@1从26.1→42.5（+16.4），但pass@256从66.3→64.3（-2.0）。

**Intuition**：训练越久，sampling efficiency越好，但reasoning boundary越窄。这是**trade-off的direct evidence**——RLVR本质上在做"分布sharpening"，这是一个zero-sum game（在fixed prior下）。

### 8.5 Scale到frontier model（Figure 9）

Magistral-Medium（Mistral的pure-RL reasoning model，接近DeepSeek-R1水平）vs. Mistral-Medium-3：
- $k=1$：Magistral在AIME24多解~7题，AIME25多解~8题
- $k$增大：gap持续缩小

**说明**：这个conclusion在frontier-scale model上仍然成立，不是small model的artifact。

---

## 9. Discussion部分的核心论证

### 9.1 为什么传统RL能超越prior，RLVR不能

传统RL（AlphaGo Zero, DQN）的两个特性：
1. **Action space小且离散**：Go是361，Atari是低维discrete。exploration tractable。
2. **从scratch开始**：没有prior，agent必须自己explore，所以discovery的任何strategy都是"novel"。

RLVR的两个特性：
1. **Action space是$|V|^T$**：exponentially vast。token-level exploration几乎不可能escape nonsense。
2. **从pretrained prior开始**：prior提供了"useful initialization"，使exploration tractable——但**也正是prior constrain了exploration范围**。任何偏离prior的sample都是nonsense，拿不到reward，被policy gradient压低。

**Double-edged sword**：prior让RLVR可行，但prior也cap了RLVR的ceiling。

### 9.2 数学视角的policy gradient分析

policy gradient的update direction本质是：

$$\nabla_\theta J \propto \mathbb{E}_{\mathbf{y} \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(y_t | \cdot) \cdot A\right]$$

- $A > 0$（correct response）：增加$\log \pi_\theta(y_t)$，即prior里high-likelihood的correct paths被进一步reinforce
- $A < 0$（incorrect response）：减少$\log \pi_\theta(y_t)$，即prior里high-likelihood的incorrect paths被压低

**但prior里low-likelihood的correct paths呢？** 它们被采样到的概率本来就低，贡献的gradient signal微弱。即使偶尔被采样到，单次gradient update也难以系统性地抬高它们的likelihood。

**结论**：policy gradient在RLVR下，是一个**prior-internal redistribution**过程，不是prior-external expansion。

---

## 10. Future Directions的深入思考

### 10.1 High-level abstraction exploration

AlphaEvolve在program-level abstraction space做evolution——这相当于把action space从token-level提高到program-level，drastically reduced action space，使得exploration tractable。类比到reasoning：是否能在"reasoning strategy level"而非"token level"做exploration？比如explore不同的problem decomposition strategy。

### 10.2 Curriculum learning

paper提到hierarchical relationship在current RLVR data里偶尔出现，但没有被deliberately exploited。intuition：如果先让model在easy subproblem上学到meta-skill（比如"set up equation"），再transfer到hard parent problem，hard problem的success rate可能从0变成non-zero——这是RLVR拿到meaningful reward的前提。

### 10.3 Process reward

binary outcome reward的credit assignment problem很严重：一个200-token的CoT，只有最终答案对不对一个signal。process reward model（PRM）能提供step-level credit，让"prior-internal但low-likelihood的正确path"也能拿到中间reward，从而被reinforce。

### 10.4 Agentic RL / Era of Experience

Silver & Sutton的"Welcome to the era of experience"论点：current RLVR是single-turn的，但IMO-level reasoning需要iterative refinement with feedback。multi-turn agent-environment interaction能让model generate novel experience（tool use, hypothesis testing, experiment），这些experience是prior里没有的——这才是真正能escape prior的path。

---

## 11. 与Related Work的对比

### 11.1 Echo Chamber（Zhao et al., 2025b）

arXiv:2504.07912  
这篇也从理论角度论证"RL post-training amplifies behaviors learned in pretraining"——和本文的finding互补：本文提供了extensive empirical evidence。

### 11.2 "There may not be aha moment in R1-zero-like training"（Liu et al., 2025a）

OAT-Zero团队的pilot study，发现reflective behaviors在base model里已经存在。本文进一步把这个finding generalize到"all reasoning paths"。

### 11.3 Dang et al.（2025）

观察到pass@k在RLVR后deteriorate，但只在Qwen-2.5-0.5B on GSM8K上做了limited experiment，没有base vs. RL的subset analysis。本文做了systematic investigation。

### 11.4 DeepSeek-Math（Shao et al., 2024）

arXiv:2402.03300  
观察到类似trend但只在single instruction-tuned model和两个math benchmark上。本文大幅扩展了实验范围。

### 11.5 Large Language Monkeys（Brown et al., 2024）

arXiv:2407.21787  
这篇是pass@k scaling inference compute的工作，启发了本文用pass@k来评估reasoning boundary。

---

## 12. Limitations和open questions

1. **Proprietary model无法eval**：GPT-o1的base model不公开，无法做base vs. RL对比。Deepseek-R1-Zero的API throughput只有~50 tokens/s，pass@1024不现实。Magistral-Medium是best proxy，但scale仍有限。
2. **Scale是否改变结论**：随着pretraining scale和RL scale都增加，结论是否hold？这是critical open question。
3. **RLVR algorithms在rapid evolution**：current paper eval的是2024末~2025初的algorithms。新的exploration机制（比如tree search augmented RL）可能改变结论。

---

## 13. 我的Intuition总结

这篇paper让我想到一个**信息论视角的reframe**：

Base model的sampling distribution $\pi_{\text{base}}$定义了一个"reasoning manifold"——所有high-likelihood的reasoning paths都在这个manifold上。RLVR做的事情是：在这个manifold上重新分配probability mass，把mass从"incorrect region"移到"correct region"。这提升了sampling efficiency（pass@1↑），但manifold本身没变——manifold外部的"correct but low-likelihood"paths依然low-likelihood，甚至因为mass redistribution而变得更low-likelihood（pass@k↓）。

Distillation做的事情不同：它通过supervised cross-entropy loss on teacher outputs，**改变了manifold本身**——把teacher的reasoning patterns"刻"进了model。即使teacher的某些paths在student的original manifold外部，distillation也强行把它们拉进新manifold。

所以**RLVR是manifold-internal optimization，distillation是manifold-expanding operation**。这解释了为什么两者有本质区别。

要实现"genuinely new reasoning"的RLVR，需要：
- 改变exploration的"层次"——从token level到strategy level（缩小effective action space）
- 改变reward的"粒度"——从binary outcome到step-level process（让low-likelihood correct paths能拿到signal）
- 改变interaction的"turn数"——从single-turn到multi-turn agentic（让model能collect novel experience）

这些都是open research directions，paper没有给出solution，但rigorously诊断了problem。

---

## References

- **Paper Project Page**: https://limit-of-RLVR.github.io
- **DeepSeek-R1**: https://arxiv.org/abs/2501.12948
- **AlphaEvolve**: https://arxiv.org/abs/2506.13131
- **Large Language Monkeys (pass@k scaling)**: https://arxiv.org/abs/2407.21787
- **PPO原paper**: https://arxiv.org/abs/1707.06347
- **GRPO / DeepSeek-Math**: https://arxiv.org/abs/2402.03300
- **Echo Chamber (RL amplifies pretraining)**: https://arxiv.org/abs/2504.07912
- **OAT-Zero (no aha moment)**: https://oatllm.notion.site/oat-zero
- **DAPO**: https://arxiv.org/abs/2503.14476
- **SimpleRL-Zoo**: https://arxiv.org/abs/2503.18892
- **Silver & Sutton "Era of Experience"**: https://silverlab.media.filed.gracehalloffame.org/2025/03/Rich-Sutton-Reward-Hypothesis-Talk-2025.pdf
- **VeRL framework**: https://arxiv.org/abs/2409.19256
- **Reinforce++**: https://arxiv.org/abs/2501.03262
- **ReMax**: https://arxiv.org/abs/2310.10505
- **RLOO**: https://arxiv.org/abs/2402.14740
- **Magistral**: https://arxiv.org/abs/2506.10910
- **Code-R1**: https://github.com/ganler/code-r1
- **EasyR1**: https://github.com/hiyouga/EasyR1
- **OpenAI o1 system card**: https://arxiv.org/abs/2412.16720
- **Kimi k1.5**: https://arxiv.org/abs/2501.12599
- **Tulu 3 (RLVR)**: https://arxiv.org/abs/2411.15124
- **HumanEval+ / MBPP+**: https://arxiv.org/abs/2303.04910
- **LiveCodeBench**: https://arxiv.org/abs/2403.07974
- **Omni-MATH**: arXiv:2510.04556 (Gao et al., 2025)
- **DeepCoder**: https://arxiv.org/abs/2506.07390
- **Dang et al. (diversity collapse)**: https://openreview.net/forum?id=AMiKsHLjQh
- **Rethinking Reflection in Pre-training (Shah et al.)**: https://arxiv.org/abs/2504.04022
- **Qwen2.5**: https://arxiv.org/abs/2412.15115
- **Qwen3**: https://arxiv.org/abs/2505.09388
- **AceReason Nemotron (curriculum)**: https://arxiv.org/abs/2505.16400
- **StepHint (multi-level hints)**: https://arxiv.org/abs/2507.02841
- **Questa (question augmentation)**: https://arxiv.org/abs/2507.13266
- **IMO 2025 verification-and-refinement**: https://arxiv.org/abs/2507.15855
- **Absolute Zero (self-play reasoning)**: https://arxiv.org/abs/2505.03386
- **REST (reinforced self-training)**: https://arxiv.org/abs/2308.08998
- **STAR (bootstrap reasoning)**: https://arxiv.org/abs/2203.14465
- **DPO**: https://arxiv.org/abs/2305.18290

---

这篇paper的value在于：在RLVR hype cycle里，提供了一个rigorous的reality check——它没有否定RLVR的empirical utility（pass@1确实大幅提升），但precisely界定了RLVR的mechanism边界。对于你（Andrej）关心的问题——如何让LLM做真正的reasoning——这篇paper暗示token-level RLVR可能是个local optimum，需要paradigm shift到hierarchical、agentic、process-rewarded的RL才能unlock下一个台阶。
