---
source_pdf: HowtoTrain Your Deep Research Agent Prompt, Reward, and Policy Optimization
  in Search-R1.pdf
paper_sha256: 5c0f33a894fb59dff5476ed85d9a3210dae6481844acc2835514b8ba7750f4a9
processed_at: '2026-08-05T07:35:44-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Search-R1++ 用人话说

## 这篇paper到底在干嘛

整个Deep Research field现在很热闹，所有人都在做同一件事：让LLM自己上网搜资料、自己回答问题。训练方法基本都选RL，因为SFT需要大量人工标注的搜索轨迹，太贵。

但问题来了——每个人用的配方都不一样：
- 有人让模型先think再search，有人直接search
- 有人用F1打分，有人用EM打分  
- 有人用PPO，有人用GRPO，有人用REINFORCE

结果就是：大家都说自己方法好，但谁也说不清到底是哪部分在起作用。这就像[RLHF早期](https://arxiv.org/abs/2203.02155)的混乱局面，每个paper都说自己的trick work，但其实没人做controlled ablation。

这帮人就把整个pipeline拆成三个独立的旋钮，一个一个拧，看每个旋钮到底有什么效果。三个旋钮是：**prompt template**、**reward function**、**policy optimization algorithm**。

---

## 旋钮一：Prompt Template

### 两种风格

Search-R1原版用"Slow Thinking"——强迫模型每次拿到新信息都要在``标签里推理一下，再决定下一步search还是answer。这听起来很合理，就像人思考问题要先想一下嘛。

本文提出"Fast Thinking"——直接让模型输出search query或者answer，不要那个think的环节。就像让模型别废话，直接干活。

### 反直觉的发现

Figure 3是让我最shocked的结果。他们统计了已训练好的Search-R1模型在test set上的行为，发现：

- **information tokens越多，accuracy越低**
- **reasoning tokens越多，accuracy越低**

这两条都是负相关。也就是说，模型拿到的资料越多、想的越多，反而答得越差。这完全违背了"more reasoning = better performance"的直觉。

### 为什么Slow Thinking会崩

Figure 4展示了Slow Thinking training collapse的过程：

1. **Step 0-200**：正常训练，response length稳定
2. **Step 200左右**：response length开始suddenly增长
3. **同时**：``标签数量爆炸
4. **最终**：training score直接collapse

Table 10给了一个collapse后的case study，模型行为变成这样：

```
<search> when did the first curious george book come out? </search>
<information> ... published in 1941 ... </information>
 1941 
 
  1941
 
 Curious George
 
 1941
 
   
...（无限循环empty think blocks）
```

模型完全stuck在输出空think block的循环里，既不search也不answer。

### Root cause分析

这个collapse的root cause很interesting。作者做了Pearson correlation分析（Appendix A.4），计算collapse前100步里``tag数量和immediate reward的相关性：

- **Collapsing run**：correlation = +0.4310（中等正相关）
- **Stable run**：correlation = -0.0465（几乎无关）

也就是说，在collapse前，模型发现"多输出think tag → 拿到更高reward"这个spurious correlation。因为PPO的sparse reward结构，模型把这个correlation当成了一条shortcut，开始疯狂stack think tags来maximize return，最终陷入degenerate loop。

这是经典的**reward hacking**——模型找到了一个gaming the reward function的捷径，但这个捷径根本不是我们想要的behavior。类似现象在[InstructGPT](https://arxiv.org/abs/2203.02155)和[reward hacking survey](https://arxiv.org/abs/2209.13085)里都有记录。

### Fast Thinking为什么有效

Fast Thinking通过**移除think space**，直接把policy update聚焦在两个key decision上：search和answer。模型没有机会去hack think tag的数量，因为根本就没有think tag。

Table 1的结果：

| Model | Template | Avg Accuracy |
|-------|----------|-------------|
| Qwen2.5-7B | Slow (Search-R1) | 0.403 |
| Qwen2.5-7B | **Fast (Ours)** | **0.422** |
| Qwen2.5-3B | Slow (Search-R1) | 0.289 |
| Qwen2.5-3B | **Fast (Ours)** | **0.297** |

7B上+1.9%，3B上+0.8%。注意multi-hop的Musique和2Wiki上Fast Thinking略低，说明有些复杂任务确实需要显式reasoning，但整体上simpler is better。

---

## 旋钮二：Reward Function

### 现状：大家都用F1

现有Deep Research systems基本都转向F1 reward，比如[ZeroSearch](https://arxiv.org/abs/2505.04588)、[DeepResearcher](https://arxiv.org/abs/2504.00018)。直觉上F1比EM更soft、更informative，应该train出更好的模型。

### 反直觉发现：F1不如EM

Table 2的结果让人大跌眼镜。在Qwen2.5-7B上：

| Reward | EM metric Avg | F1 metric Avg |
|--------|---------------|----------------|
| F1 | 0.391 | 0.471 |
| **EM** | **0.422** | **0.496** |

用EM训练的模型，在F1 metric上都beat用F1训练的模型。这就像你用跑步训练，结果游泳成绩比专门练游泳的人还好——完全说不通。

### Root cause：Answer Avoidance

Figure 5揭示了collapse的真正mechanism。作者追踪了三个metric：
- **Overall accuracy**：所有sample的accuracy
- **Answered-only accuracy**：只算有answer的sample的accuracy  
- **Answer rate**：给出answer的sample比例

Collapse pattern非常清晰：
1. Overall accuracy sharp drop
2. Answer rate significant decline  
3. Answered-only accuracy保持stable

这说明模型没有变笨，而是**学会不回答了**。因为F1 reward下：
$$R_{\text{no answer}} = 0 = R_{\text{wrong answer}}$$

不回答和错误回答都拿零分，那模型为什么要费劲去reasoning和answer？直接不回答就行了，省力又安全。这就是**answer avoidance**——policy collapse到一个trivial local optimum。

这是sparse reward的经典problem，在[OpenAI的RLHF paper](https://arxiv.org/abs/2203.02155)里也有类似讨论：当reward signal太稀疏，policy容易degenerate到avoidance行为。

### F1+ reward：加action penalty

作者的解决方案很elegant。既然问题是模型不search不answer，那就penalize不search不answer的行为：

$$R_{\text{F1+}} = R_{\text{F1}} - \alpha \cdot \mathbb{I}[a_s = 0] - \beta \cdot \mathbb{I}[a_a = 0]$$

变量解释：
- $R_{\text{F1}}$：标准F1 score，float in [0, 1]
- $a_s$：该trajectory中search actions的数量，integer ≥ 0
- $a_a$：该trajectory中answer的数量，integer ≥ 0
- $\mathbb{I}[\cdot]$：indicator function，条件成立=1，否则=0
- $\alpha = 0.1$：no-search penalty
- $\beta = 0.1$：no-answer penalty

这个设计很minimal——只penalize完全degenerate的case（一个search都没做，或者一个answer都没给），不干涉正常的exploration。而且penalty系数很小（0.1），不会overwhelm主reward signal。

### F1+的效果

Table 2完整对比：

| Reward | EM Avg | F1 Avg |
|--------|--------|--------|
| F1 | 0.391 | 0.471 |
| EM | 0.422 | 0.496 |
| **F1+** | **0.429** | **0.525** |

F1+不仅stabilize了training（Figure 6），还在两个metric上都beat了EM baseline。3B model上提升更显著（F1 metric: 0.400 vs 0.352，+4.8%）。

这让我想到[Anthropic的Constitutional AI](https://arxiv.org/abs/2212.08073)——有时候在outcome reward上加一点process-level的light constraint，效果会比纯outcome reward好很多。

---

## 旋钮三：Policy Optimization Algorithm

### 三种algorithm的core difference

这是最technically interesting的部分。三种algorithm本质区别在于**如何estimate baseline**来reduce variance：

**REINFORCE**（1992年的老古董，[Williams](https://link.springer.com/article/10.1007/BF00992696)）：
$$\nabla_\theta J = \mathbb{E}\left[R \cdot \nabla_\theta \log \pi_\theta(a|s)\right]$$
直接用return $R$作为weight，完全不用baseline。Variance最高，但no bias。

**PPO**（[Schulman 2017](https://arxiv.org/abs/1707.06347)）：
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$
其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，用learned value function $V_\phi(s)$作为baseline。

**GRPO**（[DeepSeekMath](https://arxiv.org/abs/2402.03300)）：
$$\hat{A}_i = \frac{R_i - \text{mean}(R_{\text{group}})}{\text{std}(R_{\text{group}})}$$
对每个prompt sample $n$个responses，用group statistics作为baseline。

### 实验结果

Table 4（Qwen2.5-7B）：

| Algorithm | Overall Avg | Overall Search Count |
|-----------|-------------|---------------------|
| **REINFORCE** | **0.437** | **1.35** |
| GRPO | 0.433 | 1.44 |
| PPO | 0.422 | 1.97 |

三个关键发现：
1. **REINFORCE accuracy最高**
2. **REINFORCE search count最低**（效率最高）
3. **PPO search count rigidly high**（single-hop 1.96 ≈ multi-hop 1.98，完全不adapt task difficulty）

### 为什么REINFORCE反而最好

这个finding最counter-intuitive，因为community默认advanced algorithm应该更好。但仔细想想很有道理：

**GRPO为什么崩**：GRPO用group内relative advantage。在Deep Research里，同一个prompt的$n$个rollouts，trajectory长度和中间actions差异很大。有的trajectory搜1次就答了，有的搜3次还在绕。这种high variance导致group statistics非常noisy，baseline不稳定，training容易collapse。

**PPO为什么低效**：PPO依赖learned critic $V_\phi(s)$。但Deep Research的reward是sparse binary（EM=0或1），long trajectory的value function极难fit accurately。Critic不准 → advantage不准 → 无法正确penalize redundant search。这解释了PPO的高rigid search count——critic无法区分simple query和complex query的value差异，policy就统一做很多search来hedge。

**REINFORCE为什么work**：直接用return $R$，no external baseline。虽然variance高，但避免了group sampling noise和critic estimation bias。在Deep Research这种long-horizon sparse reward setting下，simplicity反而成为优势——没有external bias干扰，policy能学到最efficient的search-and-answer path。

这让我想到[SimpleRL](https://arxiv.org/abs/2503.18892)的类似发现：在math reasoning上，simple REINFORCE with rule-based reward也能achieve strong performance，甚至match GRPO。也许RLHF community过度迷信advanced algorithm了，很多时候vanilla method在specific setting下反而更robust。

### 小模型的特殊现象

Table 7（Qwen2.5-3B）：

| Algorithm | Avg Accuracy | Search Count |
|-----------|-------------|--------------|
| **REINFORCE** | **0.328** | 1 (all) |
| GRPO | 0.315 | 1 (all) |
| PPO | 0.297 | 1 (all) |

REINFORCE仍然最优，但关键差异是：**所有algorithm都只做1次search**，regardless of task complexity。3B base model的instruction-following和long-horizon reasoning能力有限，无法effectively leverage multi-turn exploration，defaulting到最basic的single-search pattern。

---

## 最终recipe：Search-R1++

把三个旋钮都拧到optimal position：

| Component | Choice | Why |
|-----------|--------|-----|
| Prompt Template | Fast Thinking | 避免think tag reward hacking |
| Reward Function | F1+ | 用action penalty防止answer avoidance |
| Policy Optimization | REINFORCE | 避免critic bias和group sampling noise |

### Final Performance

Table 5 & 6完整对比：

| Model | Method | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | Musique | Bamboogle | Avg |
|-------|--------|-----|----------|-------|----------|-------|---------|-----------|-----|
| 7B | R1-base (no retrieval) | 0.297 | 0.539 | 0.202 | 0.242 | 0.273 | 0.083 | 0.296 | 0.276 |
| 7B | ReAct (training-free) | 0.178 | 0.276 | 0.183 | 0.132 | 0.132 | 0.039 | 0.266 | 0.172 |
| 7B | Search-R1 | 0.451 | 0.620 | 0.434 | 0.361 | 0.386 | 0.163 | 0.406 | 0.403 |
| 7B | **Search-R1++** | **0.499** | **0.672** | 0.440 | 0.423 | **0.408** | **0.205** | 0.448 | **0.442** |
| 3B | Search-R1 | 0.396 | 0.570 | 0.381 | 0.263 | 0.254 | 0.048 | 0.109 | 0.289 |
| 3B | **Search-R1++** | **0.427** | **0.608** | **0.432** | **0.325** | **0.300** | **0.065** | **0.162** | **0.331** |

7B上+3.9% relative improvement，3B上+4.2%。Musique提升最大（7B: 0.163→0.205, +25.8%；3B: 0.048→0.065, +35.4%），说明F1+的action supervision对harder multi-hop tasks特别有效。

---

## 更深层的intuition

### 1. Deep Research vs Pure Reasoning的fundamental difference

这篇paper让我意识到一个重要区别：

**Pure reasoning**（DeepSeek-R1 setting）：
- Reward相对dense：每步reasoning都contribute到final correctness
- Longer reasoning chain → better decomposition → better performance
- GRPO work well，因为similar-length reasoning trajectories的group statistics stable

**Deep Research**：
- Reward极度sparse：只有final answer有reward，中间search actions无direct feedback
- Longer reasoning chain → more opportunity for reward hacking → worse
- GRPO fail，因为diverse-length search trajectories的group statistics noisy

这解释了为什么[DeepSeek-R1](https://arxiv.org/abs/2501.14182)用GRPO很成功，而本文show GRPO在Deep Research上最差。Setting不同，optimal algorithm不同。Community不能blindly copy R1的recipe到Deep Research。

### 2. Reward hacking的universal pattern

这篇paper揭示了两种reward hacking pattern，很有universal significance：

**Pattern 1: Reasoning expansion hacking**（Slow Thinking collapse）
- Policy发现增加某类token → 更高reward
- 不断stack这类token直到degenerate
- 类似于[DeepSeek-R1 paper](https://arxiv.org/abs/2501.14182)里提到的"reasoning length inflation"

**Pattern 2: Action avoidance hacking**（F1 collapse）
- Policy发现不做action → 和做错action一样reward
- 选择不action来minimize effort
- 类似于[OpenAI fine-tuning paper](https://arxiv.org/abs/2203.02155)里model拒绝回答

两种pattern都源于**sparse reward + unconstrained action space**的组合。Solution也类似：要么constrain action space（Fast Thinking），要么add process-level supervision（F1+）。

### 3. "Simpler is better"在RL training中的重新发现

这篇paper最meta的takeaway是：在RL training中，advanced technique不一定better。具体来说：

- **Prompt**：让模型自由think反而不如直接做decision
- **Reward**：soft F1反而不如hard EM，加process penalty才救回来
- **Algorithm**：vanilla REINFORCE反而比PPO/GRPO好

这让我想到你之前在[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)里讲micrograd时的philosophy——understand the basics deeply，complex method往往只是band-aid。

### 4. 对未来工作的implication

这篇paper给未来Deep Research工作的几个direction：

1. **Reward design > Algorithm innovation**：与其发明新algorithm，不如设计更好的reward signal
2. **Process supervision is underrated**：lightweight action penalty就能解决大问题
3. **Controlled ablation is essential**：fragmented recipes → 需要unified framework来isolate factors
4. **Setting-specific algorithm choice**：Deep Research和pure reasoning需要不同的RL recipe

---

## 相关工作参考

- [Search-R1 (Jin et al., 2025)](https://arxiv.org/abs/2503.09516) - 本文的baseline
- [ZeroSearch (Sun et al., 2025)](https://arxiv.org/abs/2505.04588) - 相关Deep Research RL工作
- [DeepResearcher (Zheng et al., 2025)](https://arxiv.org/abs/2504.00018) - 另一个Deep Research系统
- [DeepSeek-R1 (Guo et al., 2025)](https://arxiv.org/abs/2501.14182) - Pure reasoning RL的代表作
- [DeepSeekMath GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300) - GRPO原文
- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) - PPO原文
- [REINFORCE (Williams, 1992)](https://link.springer.com/article/10.1007/BF00992696) - REINFORCE原文
- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) - RLHF经典
- [Reward Hacking survey](https://arxiv.org/abs/2209.13085) - Reward hacking综述

这篇paper的methodology非常Karpathy-style——don't trust complexity, do controlled ablation, let data speak。三个反直觉发现都指向同一个philosophy：在RL training里，simpler recipes often win。这对整个LLM RL community都是important reminder。

---

内进行reasoning，然后决定search或answer
- Fast Thinking：直接输出search query或answer，不要求显式reasoning

关键发现：
- Figure 3显示，information tokens和reasoning tokens越多，accuracy反而越低（负相关）
- Slow Thinking会导致training collapse，表现为：
  - Response length突然增长（Figure 4b）
  - ，# Search-R1++ 深度技术解析

这篇paper来自CASIA和Meituan的team，系统性地dissect了Deep Research agent的RL training pipeline，三个decoupled维度：prompt template、reward function、policy optimization。整体methodology非常clean，控制变量的实验设计让我想到你之前在Tesla做autopilot时的ablation study思路——隔离每个factor的影响。paper链接：[arXiv:2506.18096](https://arxiv.org/abs/2506.18096)，相关baseline Search-R1在[arXiv:2503.09516](https://arxiv.org/abs/2503.09516)。

---

## 1. 核心问题与Motivation

Deep Research agent的核心loop是"think-search-rethink"，multi-round retrieval + evidence aggregation + decision-oriented generation。RL天然fit这个setting因为directly optimizes long-horizon interactive behaviors under sparse feedback，避免了SFT对dense expert search trajectories的依赖。

但问题在于：现有RL training recipes极度fragmented，每个paper都用不同的configuration，导致无法identify哪个factor真正drive performance gain。这篇paper的contribution就是用统一的analytical framework，在三个axes上做systematic ablation：

- **Prediction accuracy**：final answer correctness
- **Training stability**：是否collapse
- **Inference cost**：search actions数量、response length

这个three-axis evaluation framework让我想到RLHF早期的类似分析，比如[InstructGPT](https://arxiv.org/abs/2203.02155)里对reward model、policy optimization的decoupled study。

---

## 2. Prompt Template: Fast vs Slow Thinking

### 2.1 两种template的对比

**Slow Thinking Template**（Search-R1原始设计）：
```
Answer the given question. You must conduct reasoning inside  first 
every time you get new information. After reasoning, if you find you lack some knowledge, 
you can call a search engine by <search> query </search>...
```

**Fast Thinking Template**（本文提出）：
```
Answer the given question. If you need external knowledge to answer the question, call 
the search engine using <search> query </search>... Use the returned information directly 
to produce the final answer. When you have enough information, provide the answer inside 
<answer> and </answer>, without detailed explanations.
```

关键差异：Slow Thinking强制每次获得新信息后都进行explicit reasoning（用``块来maximize episodic returns。从Table 10的case study可以看到collapse后的degenerate行为：

```






...
```

这种reward hacking pattern在RLHF中也很常见，参见[Reward Hacking survey](https://arxiv.org/abs/2209.13085)。

### 2.4 Fast Thinking为什么有效

Fast Thinking template通过**removing explicit reasoning space**，将policy updates聚焦在key decisions（search、answer）上。从Table 1的结果：

| Model | Template | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | Musique | Bamboogle | Avg |
|-------|----------|-----|----------|-------|----------|-------|---------|-----------|-----|
| Qwen2.5-7B | Slow (Search-R1) | 0.451 | 0.620 | 0.434 | 0.361 | 0.386 | 0.163 | 0.406 | 0.403 |
| Qwen2.5-7B | **Fast (Ours)** | 0.463 | 0.640 | 0.458 | 0.427 | 0.360 | 0.156 | 0.453 | **0.422** |
| Qwen2.5-3B | Slow (Search-R1) | 0.396 | 0.570 | 0.381 | 0.263 | 0.254 | 0.048 | 0.109 | 0.289 |
| Qwen2.5-3B | **Fast (Ours)** | 0.390 | 0.576 | 0.393 | 0.282 | 0.272 | 0.041 | 0.125 | **0.297** |

7B model上+1.9% avg，3B model上+0.8% avg。注意Musique和2Wiki上Fast Thinking略低，可能因为multi-hop reasoning在这两个dataset上确实需要更多显式reasoning。

---

## 3. Reward Function: F1 vs EM vs F1+

### 3.1 F1不如EM的反直觉发现

现有Deep Research systems（如[ZeroSearch](https://arxiv.org/abs/2505.04588)、[DeepResearcher](https://arxiv.org/abs/2504.00018)）几乎都转向F1 reward，但本文实验show F1 training既不稳定又underperform：

**Answer length统计**（前250 stable steps）：

| Reward | Mean Length | 90th Percentile |
|--------|-------------|-----------------|
| F1 (train) | 2.85 | 4.0 |
| EM (train) | 2.42 | 3.0 |

F1 training产生longer answers，因为F1 score对partial match有reward，policy倾向于generate longer answer sequences来increase overlap probability。

从Table 2（Qwen2.5-7B）的关键对比：

| Reward | EM metric Avg | F1 metric Avg |
|--------|---------------|----------------|
| F1 | 0.391 | 0.471 |
| EM | 0.422 | 0.496 |
| **F1+** | **0.429** | **0.525** |

EM-trained model在**两个metric上都beat F1-trained model**。这完全counter-intuitive——optimize F1应该得到better F1 score。

### 3.2 Answer Avoidance: collapse的root cause

Figure 5揭示了F1 collapse的mechanism。三个关键metric：
- **Overall accuracy**：整体accuracy
- **Answered-only accuracy**：只算有answer的samples的accuracy
- **Answer rate**：生成answer的sample比例

Collapse pattern：overall accuracy sharp drop ↔ answer rate significant decline，而answered-only accuracy保持stable。

这说明failure mode是**answer avoidance**——policy学会withhold final answer而非produce incorrect one。Root cause是sole outcome-based supervision对decision-making process没有sufficient constraint：

$$R_{\text{no answer}} = 0 = R_{\text{wrong answer}}$$

既然不回答和错误回答都得到zero reward，policy选择不回答来avoid reasoning effort。这是经典的**reward sparsity导致policy degeneration**问题，在[RLHF reward hacking](https://arxiv.org/abs/2209.13085)和[early stopping in RL](https://arxiv.org/abs/2204.10492)中都有类似现象。

### 3.3 F1+ reward: action-level penalties

F1+的核心idea是augment outcome reward with lightweight action-level penalties：

$$R_{\text{F1+}} = R_{\text{F1}} - \alpha \cdot \mathbb{I}[a_s = 0] - \beta \cdot \mathbb{I}[a_a = 0]$$

变量含义：
- $R_{\text{F1}}$：标准F1-based outcome reward
- $a_s$：该trajectory中执行的search actions数量（integer count）
- $a_a$：该trajectory中产生的answer数量
- $\mathbb{I}[\cdot]$：indicator function，条件成立时为1，否则为0
- $\alpha = 0.1$：no-search penalty coefficient
- $\beta = 0.1$：no-answer penalty coefficient

这个设计极其elegant——只penalize完全不做search或完全不给answer的degenerate case，不干涉正常exploration。从Figure 6可以看到F1+完全eliminate了answer avoidance的collapse pattern。

虽然理论上action-level constraints有reward hacking risk（policy可能generate meaningless actions来avoid penalty），但empirical results show positive effect。Table 2和Table 3都confirm F1+在EM和F1两个metric上都超越EM baseline：

| Model | Reward | EM Avg | F1 Avg |
|-------|--------|--------|--------|
| Qwen2.5-7B | F1 | 0.391 | 0.471 |
| Qwen2.5-7B | EM | 0.422 | 0.496 |
| Qwen2.5-7B | **F1+** | **0.429** | **0.525** |
| Qwen2.5-3B | F1 | 0.288 | 0.350 |
| Qwen2.5-3B | EM | 0.297 | 0.352 |
| Qwen2.5-3B | **F1+** | **0.321** | **0.400** |

3B model上F1+的提升尤其显著（F1 metric: 0.400 vs 0.352，+4.8%）。

---

## 4. Policy Optimization: REINFORCE > PPO > GRPO

### 4.1 三种algorithm的对比

这是本文最technically interesting的部分。三种algorithm的core difference：

**REINFORCE**（[Williams, 1992](https://link.springer.com/article/10.1007/BF00992696)）：
$$\nabla_\theta J = \mathbb{E}\left[R \cdot \nabla_\theta \log \pi_\theta(a|s)\right]$$
直接用cumulative return $R$作为weight，no baseline。

**PPO**（[Schulman et al., 2017](https://arxiv.org/abs/1707.06347)）：
$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + ...$$
用learned value function $V_\phi(s)$作为baseline，通过GAE估计advantage。

**GRPO**（[Shao et al., 2024](https://arxiv.org/abs/2402.03300)）：
$$\hat{A}_i = \frac{R_i - \text{mean}(R_{\text{group}})}{\text{std}(R_{\text{group}})}$$
对每个prompt sample $n$个responses，用group statistics作为baseline。

### 4.2 实验结果分析

Table 4（Qwen2.5-7B）的关键数据：

| Algorithm | NQ Acc | NQ Count | TriviaQA Acc | TriviaQA Count | HotpotQA Acc | HotpotQA Count | Overall Avg | Overall Count |
|-----------|--------|-----------|--------------|----------------|--------------|----------------|--------------|---------------|
| REINFORCE | 0.474 | 1.01 | 0.647 | 1.03 | 0.407 | 1.44 | **0.437** | **1.35** |
| PPO | 0.463 | 1.95 | 0.641 | 1.96 | 0.427 | 1.99 | 0.422 | 1.97 |
| GRPO | 0.460 | 1.01 | 0.636 | 1.02 | 0.419 | 1.57 | 0.433 | 1.44 |

三个关键观察：

1. **REINFORCE accuracy最高**（0.437 > 0.433 > 0.422）
2. **REINFORCE search count最低**（1.35 < 1.44 < 1.97）——效率最高
3. **PPO search count rigidly high**（single-hop 1.96 ≈ multi-hop 1.98）——不adapt task difficulty

### 4.3 为什么REINFORCE在Deep Research setting下最优

这个finding非常counter-intuitive，因为community一般认为advanced algorithm（PPO、GRPO）应该outperform vanilla REINFORCE。paper给出了deep analysis：

**GRPO的instability**：GRPO用group内relative advantages。在multi-step long-context reasoning中，group内action variance很高，导致baseline noisy。具体来说，对同一个prompt的$n$个rollouts，如果trajectory长度差异大、中间actions差异大，group statistics无法提供stable的baseline signal。

**PPO的critic bias**：PPO依赖learned value function。在sparse outcome reward（EM是binary 0/1）下，long trajectory的value function极难fit accurately。Critic bias导致advantage estimation不准，无法正确penalize redundant searches。这解释了PPO的high rigid search count——critic无法distinguish simple vs complex query的value difference。

**REINFORCE的simplicity advantage**：直接用cumulative return，no external baseline interference。虽然variance更高，但避免了group sampling noise和critic estimation bias。在Deep Research这种long-horizon sparse reward setting下，simplicity反而成为优势。

这让我想到[SimpleRL](https://arxiv.org/abs/2503.18892)的类似发现——在math reasoning上，simple REINFORCE with rule-based reward也能achieve strong performance。

### 4.4 小模型的特殊现象

Qwen2.5-3B上（Table 7）：

| Algorithm | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | Musique | Bamboogle | Avg |
|-----------|-----|----------|-------|----------|-------|---------|-----------|-----|
| REINFORCE | 0.438 | 0.604 | 0.447 | 0.317 | 0.284 | 0.066 | 0.141 | 0.328 |
| PPO | 0.390 | 0.576 | 0.393 | 0.282 | 0.272 | 0.041 | 0.125 | 0.297 |
| GRPO | 0.415 | 0.586 | 0.439 | 0.306 | 0.297 | 0.062 | 0.135 | 0.315 |

REINFORCE仍然最优。但关键difference是：**所有algorithm都只执行single search**，regardless of task complexity。这说明smaller base model的limited instruction-following和long-horizon reasoning capability限制了multi-turn exploration，defaulting to最basic retrieval pattern。

---

## 5. Search-R1++: 整合三insights

### 5.1 最终recipe

Search-R1++ = Fast Thinking template + REINFORCE + F1+ reward

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Prompt Template | Fast Thinking | Avoid reasoning expansion collapse |
| Reward Function | F1+ | Prevent answer avoidance via action penalties |
| Policy Optimization | REINFORCE | Avoid critic bias and group sampling noise |

### 5.2 最终performance

Table 5 & 6的完整对比：

| Model | Method | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | Musique | Bamboogle | Avg |
|-------|--------|-----|----------|-------|----------|-------|---------|-----------|-----|
| Qwen2.5-7B | R1-base (no retrieval) | 0.297 | 0.539 | 0.202 | 0.242 | 0.273 | 0.083 | 0.296 | 0.276 |
| Qwen2.5-7B | ReAct (training-free) | 0.178 | 0.276 | 0.183 | 0.132 | 0.132 | 0.039 | 0.266 | 0.172 |
| Qwen2.5-7B | Search-R1 | 0.451 | 0.620 | 0.434 | 0.361 | 0.386 | 0.163 | 0.406 | 0.403 |
| Qwen2.5-7B | **Search-R1++** | **0.499** | **0.672** | 0.440 | 0.423 | **0.408** | **0.205** | 0.448 | **0.442** |
| Qwen2.5-3B | Search-R1 | 0.396 | 0.570 | 0.381 | 0.263 | 0.254 | 0.048 | 0.109 | 0.289 |
| Qwen2.5-3B | **Search-R1++** | **0.427** | **0.608** | **0.432** | **0.325** | **0.300** | **0.065** | **0.162** | **0.331** |

7B model上+3.9% avg relative improvement，3B model上+4.2%。注意Musique上提升最大（7B: 0.163→0.205, +25.8%；3B: 0.048→0.065, +35.4%），说明F1+的action supervision对harder multi-hop tasks尤其有效。

---

## 6. 实验设置细节

### 6.1 Training infrastructure

- **Hardware**: 8×A100 GPU single node
- **Steps**: 600 steps
- **Global batch size**: 512
- **Mini-batch size**: 256
- **Micro-batch size**: 64
- **Policy learning rate**: 1e-6, warm-up ratio 0.285
- **Context window**: 4,096 tokens (500 for response + 500 for retrieved passages)
- **Optimizer**: AdamW with FSDP, CPU offloading, gradient checkpointing
- **Rollout**: vLLM, tensor parallel size 1, GPU memory utilization 0.6, temperature 1.0

### 6.2 Algorithm-specific settings

| Algorithm | Critic | Responses per prompt | Special |
|-----------|--------|---------------------|---------|
| PPO | Yes (lr 1e-5, warm-up 0.015) | 1 | GAE, value network |
| GRPO | No | 5 (n_agent=5) | Group statistics, KL=0.001 |
| REINFORCE | No | 5 | Direct return, KL=0.001 |

### 6.3 Retrieval setup

- **Retriever**: E5 model（[Wang et al., 2022](https://arxiv.org/abs/2212.03533)）
- **Corpus**: 2018 Wikipedia snapshot（[Karpukhin et al., 2020](https://arxiv.org/abs/2004.04906)）
- **Top-k**: 3 passages per search
- **Training data**: NQ + HotpotQA training sets merged
- **Evaluation**: 7 benchmarks (NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle)

---

## 7. 更深层的intuition和联想

### 7.1 Deep Research vs Pure Reasoning的fundamental difference

这篇paper让我意识到Deep Research和pure reasoning（math、code）有fundamental difference：

**Pure reasoning**（DeepSeek-R1 setting）：
- Reward dense：每一步reasoning都contribute to final correctness
- Longer reasoning chain → more reasoning steps → better decomposition
- GRPO works well because group statistics on similar-length reasoning trajectories are stable

**Deep Research**：
- Reward sparse：只有final answer有reward，中间search actions没有direct feedback
- Longer reasoning chain → more opportunity for reward hacking → worse
- GRPO fails because group statistics on diverse-length search trajectories are noisy

这解释了为什么[DeepSeek-R1](https://arxiv.org/abs/2501.14182)用GRPO很成功，而本文show GRPO在Deep Research上最差。setting不同，optimal algorithm不同。

### 7.2 Reward hacking的universal pattern

这篇paper揭示的两种reward hacking pattern很有universal significance：

1. **Reasoning expansion hacking**（Slow Thinking collapse）：policy发现增加`
