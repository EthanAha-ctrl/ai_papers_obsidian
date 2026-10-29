---
source_pdf: EMERGENT HIERARCHICAL REASONING IN LLMS.pdf
paper_sha256: d67b03b9650dc684cc574798bf31b7c27f52f677ea54150f07dc9845b836eab1
processed_at: '2026-08-04T03:55:34-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再讲一遍

---

## 这篇paper到底在说什么

一句话：**RL 训练 LLM reasoning，真正起作用的地方不是让模型"算得更准"，而是让模型"想得更对"**。

他们发现了一个现象，然后基于这个现象改了一下训练算法，效果就变好了。

---

## 现象是什么

先说背景。你拿一个 base model，用 GRPO 这种 RL 方法去训练它做数学题。训练过程中会观察到一些奇怪的现象：

- **"aha moment"**：模型训练到某个阶段，突然"开窍"了，会开始 self-reflection，会说"wait, let me reconsider"
- **length scaling**：模型输出的 reasoning trace 越来越长，而且越长越准
- **token entropy 的奇怪动态**：有时候 entropy 下降，但模型还在变好

这些现象之前大家都观察到了，但没人能解释清楚为什么。

这篇paper给了一个统一的解释：**这些现象都是同一个底层机制的不同表现——模型在 RL 训练中自发地形成了一个"分层推理"结构**。

---

## 什么是"分层推理"

类比人。你解一道数学竞赛题的时候，你的脑子里其实同时在干两件事：

**高层**：你在做 strategy 决策。"这题应该用 substitution"、"这条路走不通，换一个 approach"、"我需要分 case 讨论"。这些是慢思考，是 deliberate 的。

**低层**：你在做具体操作。"把 5 加到两边"、"展开这个 polynomial"、"算 23 乘 47"。这些是快思考，是 automatic 的。

人脑就是这么分层的——前额叶做 high-level planning，运动皮层 / 后部皮层做 low-level execution。神经科学的 literature 早就讲过这个（Murray et al. 2014 那篇 Nature Neuroscience）。

这篇paper发现：**LLM 在 RL 训练中也自发地形成了类似的两层结构**。不是你设计的，是 emergent 的。

---

## 怎么发现的两层结构

难点是：你拿到一段 reasoning trace，哪些 token 是"高层 planning"，哪些是"低层 execution"？这没法靠语法规则区分——"let me"可以是 strategic move，也可以是 filler word。

他们的解法很聪明。他们注意到一个 statistical signature：

**高层的 strategic 语言有一个特点：跨很多不同题目反复出现，但在单道题里用得很少。**

你想想，"let's try a different approach" 这句话，会在成百上千道不同题目的 solution 里出现，但在一道题的 solution 里可能只出现一两次。而具体的计算步骤（"23 × 47 = 1081"）在一道题里可能出现很多次，但跨题目的重复率低。

所以他们做了一个 pipeline：
1. 从大量成功 solution 里抽 n-gram（3-5 个 token 的短语）
2. 用 sentence embedding 把语义相似的 n-gram 聚类（"let's try another approach" 和 "let's try a different angle" 归到一起）
3. 看哪些 cluster 在很多不同 solution 里都出现过（Cluster Document Frequency 高的）
4. 取 top 20%，这些就是 **Strategic Grams (SGs)**

附录里列了 250 多个 SGs，比如 "let's backtrack"、"on second thought"、"the key insight is"、"wait, that's not right"、"let me reconsider"。

一旦你有了 SG set，你就能把每条 reasoning trace 里的 token 分成两类：planning token（属于某个 SG）和 execution token（其他）。

---

## 训练过程中发生了什么

他们在 8 个不同 model 上跑 RL 训练（Qwen、Llama、还有 VLM），都观察到同样的 **two-phase dynamic**：

### Phase ①：先把低层搞稳

训练刚开始，模型的主要矛盾是"算不对"。一个计算错误就毁掉整条 solution。所以学习信号逼着模型先把 procedural skill 搞稳。

观察到的指标变化：
- Execution token 的 perplexity **急剧下降** → 模型对"23×47=1081"这种操作越来越 confident
- Execution token 的 token entropy **显著低于** planning token → 模型在低层放弃 exploration，converge 到可靠操作

这个 phase 持续不长，模型很快就 build 好了一个 reliable 的 "toolbox"。

### Phase ②：开始探索高层 strategy

低层稳了之后，performance gain 的来源就 shift 了。模型开始探索各种 high-level strategy。

观察到的指标变化：
- Strategic Grams 的 **semantic entropy 稳定上升** → 模型在用越来越多样的策略
- Reasoning trace 越来越长 → 因为 sophisticated strategy 本身就需要更多篇幅
- Accuracy 持续上升 → 这些新策略真的有用

关键发现：**conditional entropy of procedural n-grams given a preceding SG 保持稳定**。意思是，一旦你决定了"我要用 substitution"，具体怎么做 substitution 没什么变化空间。variety 在 strategy 层，不在 execution 层。

---

## 这解释了之前那些"谜"

**"aha moment"** = 模型 discover 并 internalize 了一个新的 powerful strategy（比如 self-reflection / backtracking）的那个瞬间。

**"length scaling"** = sophisticated strategy 需要更长的 trace 来表达。length 是 semantic diversity 的 by-product，不是 cause。

**token entropy 的奇怪动态** = token entropy 被海量低层 execution token 主导。低层 entropy 下降（模型变 confident）拉低了整体平均，但这不代表 exploration 停了——strategic 层的 exploration 还在进行。

---

## 现有算法的问题

GRPO 的 credit assignment 是这样的：一条 trajectory 拿到一个 reward（比如 0 或 1），算出一个 advantage，然后这条 trajectory 里**所有 token 都拿同一个 advantage**。

问题来了：如果 learning frontier 已经 shift 到 strategic planning 了，你把同样的 gradient signal 投到 execution token 上，是在浪费优化 budget。更糟的是，execution token 的数量远多于 planning token，所以 gradient 被**稀释**了——真正重要的 strategic decision 拿到的有效 signal 很弱。

类比：你在公司里，CEO 做的战略决策和实习生做的数据录入，你给同样的绩效权重。结果 CEO 的战略改进信号被淹没了。

---

## HICRA 的解法

非常简单的一行公式：

对 planning token，把 advantage 放大；对 execution token，不动。

具体来说：
- 成功的 trajectory（advantage > 0）：planning token 的 advantage 乘以 $(1+\alpha)$，$\alpha=0.2$
- 失败的 trajectory（advantage < 0）：planning token 的 advantage 乘以 $(1-\alpha)$

注意这个 **asymmetric design**：
- 成功了 → planning token 拿更多 credit（"这次成功主要是因为 strategy 选得好"）
- 失败了 → planning token 的 penalty 被衰减（"你尝试了新 strategy 但 execution 出错了，不该把锅扣在 strategy 上"）

这其实就是 **optimism in the face of uncertainty** 的一个变体——鼓励 strategic exploration，减少 false negative 的惩罚。

---

## 为什么 entropy regularization 不 work

有人试过另一种方法：给所有 token 加 entropy bonus，鼓励 exploration。结果：

- Token entropy 确实升上去了 ✓
- 但 accuracy 没涨甚至下降 ✗
- Length 失控增长 ✗

为什么？因为 indiscriminate entropy bonus 主要膨胀的是低层 token 的多样性。"5+3" 可以写成 "five plus three" / "adding 5 and 3" / "the sum of 5 and 3 is"... entropy 涨了，reasoning 没变好。

HICRA 反过来：semantic entropy（strategy 多样性）涨上去，accuracy 跟着涨。**关键不是 explore more，是 explore in the right direction**。

---

## 实验结果怎么样

一句话总结：**在大多数 model 上，HICRA 比 GRPO 好，而且越弱的 base model 提升越大**。

亮眼的数据点：
- Qwen3-4B-Base 在 AIME24 上：GRPO 24.9 → HICRA 31.0（+6.1）
- Qwen3-4B-Instruct 在 AIME25 上：GRPO 60.0 → HICRA 65.1（+5.1）
- MiMO-VL 在 MathVista 上：GRPO 73.7 → HICRA 80.7（+7.0）

AIME25 是 out-of-distribution benchmark，提升显著说明学到的是 **generalizable strategic skill**，不是过拟合。

---

## 一个重要的 negative result

Llama-3.1-8B-Instruct 上，HICRA 反而比 GRPO **差**（AIME24: 8.9 → 8.3）。

为什么？因为 Llama 的 procedural foundation 不够稳。HICRA 鼓励它 explore strategy，但它 explore 出来的 plan 没法可靠地 execute，结果 advantage signal 变成噪声。

直觉：HICRA 是个 **amplifier**，它假设 underlying signal 存在。如果 base model 还在"算术都不稳"的阶段，你放大 strategic dimension 的 gradient 反而 inject 噪声。

这说明 HICRA 有个 **dependency**：base model 需要先有一定 procedural reliability。对 Llama 这种 model，可能需要先用 vanilla GRPO 跑一段，再切到 HICRA。

---

## Semantic entropy 比 token entropy 好在哪

这是 paper 里一个很 useful 的工具贡献。

Token entropy 的问题是：它被海量低层 execution token 主导。模型训练过程中，低层 token 变得越来越 confident（entropy 下降），把整体平均拉下来。你看到 token entropy 下降，可能以为 exploration 停了，但实际上 strategic 层还在积极 explore。

Semantic entropy 的做法：不看 individual token 的不确定性，看 **strategic gram 的频率分布的多样性**。这直接测的是 high-level action space 的覆盖。

实验证据：在 MiMO-VL 上，token entropy 在 HICRA 和 GRPO 上都 collapse（看不出差异），Pass@8 早早饱和（看不出差异），但 semantic entropy 保持高位且 HICRA > GRPO，这个 gap 与最终 accuracy 的 gap 一一对应。

**对 RL practitioner 的实操建议：用 semantic entropy 判断 exploration 健康度，别用 token entropy**。

---

## 还有一个有意思的对比

最近有人提出用 "high-entropy token"（也叫 "fork token"）作为 decision point 的 proxy。

paper 做了对比：
- 70%+ 的 planning token 确实是 high-entropy 的 ✓
- 但只有 <10% 的 high-entropy token 是 planning token ✗

意思是：**用 high entropy 找 planning token，recall 高但 precision 极低**。大量高熵 token 是 phrasing variants（"therefore" / "thus" / "so"）或计算中间步骤，它们熵高但不是 strategic decision。

functional definition（基于语义角色）比 statistical definition（基于熵）更可靠。

---

## 我读完的几个 takeaways

1. **RL for LLM reasoning 的主要杠杆是 strategy，不是 execution**。Fig 5 的 error decomposition 图是最直接的证据——训练后低层错误几乎不降，但 accuracy 大涨，所有 gains 来自高层 strategy 的修复。

2. **Pre-training 给了 model 一个 hierarchical prior**，RL 不是从 0 发明 strategy，是在已有 scaffold 的子空间里做选择。这跟 Sutton 的 "Bitter Lesson" 哲学一致——你不需要 hand-engineer reasoning module，你只需要让 RL 去 discover 已有的 structure。

3. **Credit assignment 应该是 functional-aware 的**。所有 token 一视同仁是 brute approximation。HICRA 用一个简单的 binary mask + scalar amplification 就获得了显著提升，说明这个方向还有大量 low-hanging fruit。

4. **Semantic entropy 是一个 better diagnostic tool**。对所有在跑 LLM RL 的人都有 immediate 的工具价值。

5. **HICRA 有适用边界**：base model 需要先有 procedural foundation。对 weak base model，可能需要 curriculum（先 GRPO 后 HICRA）。

---

## 更大的图景

这篇 paper 让我想到一个更大的 narrative：

LLM 的 reasoning 能力，本质上是一种 **"用语言编程"的能力**。Strategic Grams 就是 natural language 里的 **control flow keywords**——"let's backtrack" 是 goto，"let's try another approach" 是 branch，"the key insight is" 是 function definition。

Pre-training 让模型继承了这些 control flow keyword 的 vocabulary。RL 让模型学会在什么时候用哪个 control flow。

HICRA 做的事情，就是给这些 control flow keyword 一个更大的 gradient signal——"这些 keyword 很重要，多学学怎么用它们"。

这跟 Karpathy 你之前讲的 "Software 2.0 / 3.0" 的视角很契合：natural language 正在变成一种 programming language，而 RL 是编译 + 优化这个过程的关键。

---

## 一句话总结

**RL 训练 LLM reasoning 的核心机制是 rediscover 并 amplify pre-training 继承的 hierarchical reasoning structure。学习 frontier 会从 low-level procedural consolidation shift 到 high-level strategic exploration。HICRA 通过给 planning token 更大的 advantage 来集中优化 budget，效果显著。Semantic entropy 是比 token entropy 更好的 exploration diagnostic。**

这个 paper 的价值不在于算法多复杂（其实很简单），在于**揭示了一个可复现的 phenomenon**，并给出了一个可操作的 intervention + 一个有用的 diagnostic tool。这种 "phenomenon + intervention + tool" 的组合拳，是好的 research 的标志。

---

# Emergent Hierarchical Reasoning in LLMs via RL — 一份深度讲解

## 一、Core thesis 一句话

Pre-training 给了 LLM 一套 "hierarchical prior"（类似人类前额叶分层处理的结构），RL fine-tuning 真正在做的事是 *rediscover* 并 *amplify* 这套 prior：先在低层 procedural skill 上压稳 variance，再把优化压力"迁移"到高层的 strategic planning 子空间。GRPO 这类算法对所有 token 等价施加 gradient，本质上把这个稀疏、高杠杆的 strategic 维度给 *dilute* 了。HICRA 用一个简短的 advantage 修正把梯度集中投到 planning token 上，结果在多个 base model 上获得持续提升。

project page: https://tiger-ai-lab.github.io/Hierarchical-Reasoner/

---

## 二、为什么是 hierarchical？cognitive neuroscience 的类比

Karpathy 你肯定熟悉 Kahneman 的 System 1 / System 2 切分。这篇 paper 借用了另一条神经科学线：primate cortex 的 *intrinsic timescale hierarchy*（Murray et al., 2014, *Nature Neuroscience*）。从 V1 到 prefrontal cortex，神经元的 time constant 沿着 cortical hierarchy 单调上升——低层做快速、stereotyped 的 computation（毫秒级），高层做 slow、deliberate 的 planning（秒级）。Zeraati et al., 2023 也证明这种 hierarchy 在 attention task 中会动态 shift。

paper 把这个类比搬到 LLM 上：

- **High-level Planning Tokens**：orchestrate 推理流程的 logical maneuver，包括 *deduction*（"we can use the fact that"）、*branching*（"let's try a different approach"）、*backtracing*（"but the problem mentions that"）。
- **Low-level Execution Tokens**：arithmetic、variable substitution、formula application、formatting 等具体 ops。

这个 decomposition 不是语言学层面的（不是 noun/verb），是 *functional* 的：一个 token 的角色由 context 决定。

参考：
- Murray et al., 2014: https://www.nature.com/articles/nn.3862
- Zeraati et al., 2023: https://www.nature.com/articles/s41467-023-37437-3
- Huntenburg et al., 2018: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(17)30207-5

---

## 三、Strategic Grams (SGs)：functional proxy 的构造

直接的难点：你没法用 syntactic rule 区分 planning vs execution token，"let me" 可以是 high-level 也可以是低层填充词。作者的解法是 *statistical signature*——SGs 是"reusable scaffolding"：跨多种问题反复出现，但在单个 solution 里使用频率低。

### 三步 pipeline

**Step 1: Semantic Clustering**
- 从一个 *large corpus of successful reasoning solutions*（rollouts 中拿到正 reward 的 trajectories）抽所有 $n$-grams，$n \in [3,5]$。
- 用 `sentence_transformers`（pre-trained sentence transformer，例如 `all-MiniLM-L6-v2` 这种）把每个 $n$-gram 投到 embedding space。
- 在 embedding space 上做 clustering，把 lexically diverse 但 semantically equivalent 的 $n$-grams（"let's try another approach"、"let's try a different angle"、"perhaps we can consider another method"）合并到一个 cluster。
  - 这一步绕开了 linguistic diversity 的难题：同一个 strategic intent 可能有几十种表达。

**Step 2: Identification by Frequency**
对每个 cluster $c$，定义 **Cluster Document Frequency (Cluster DF)**：

$$\text{ClusterDF}(c) = \#\{\text{solutions } s \mid \exists g \in c,\, g \subset s\}$$

- $c$：semantic cluster
- $g$：cluster 内某个具体 $n$-gram
- $s$：一条 solution
- $\#$：count of unique solutions containing at least one member of $c$

这相当于把 TF-IDF 里的 DF 抬到 cluster 层面。

**Step 3: SG Construction**
取 Cluster DF 排在前 20% 的 clusters，把这些 clusters 里所有 $n$-grams 的 union 作为最终 SG set。附录 Listing 1 给了完整 list（约 250+ 个 SGs），含 `"let's backtrack"`、`"on second thought"`、`"wait, that's not right"`、`"the key insight is"` 等。

### Robustness 验证
random drop 30% SGs → semantic entropy curve *qualitatively identical*（Fig 11, 12）。这间接说明 SG 是一个 *sufficient statistic* for strategic movement，不需要 exhaustive。

### 我对 SG 的几个思考
1. SG set 其实是 pre-training 分布的 emergent signature——人类 solution 里就这么写的，pretraining 让模型继承了这套 *verbal scaffold*。RL 不是从 0 发明 strategy，是在已有 scaffold 的子空间里做选择。
2. 这个方法对 *non-English* 推理需要重新构建 SG set；但 pipeline 本身是 language-agnostic 的。
3. 对 *code reasoning* 可能失效：code 的 "strategic move" 多是结构性 token（`def`、`return`、`if __name__`），而不是 phrasal n-gram。作者在 conclusion 也提到这个 future direction。

参考：
- SentenceTransformers: https://www.sbert.net/
- Listing 1 完整 SG list 在 paper 附录

---

## 四、Emergent two-phase dynamics：实证证据

作者跑了 8 个 model（Qwen2.5-7B、Qwen3-4B、Llama-3.1-8B、Qwen2.5-VL-7B、MiMO-VL-7B 等），每个都展现同一个 two-phase pattern。

### Phase ① Procedural Consolidation

**Relative Perplexity**：把 PPL 用初始值 normalize，比较 planning token vs execution token 的变化速率。

$$\text{RelPPL}_t = \frac{\text{PPL}_t}{\text{PPL}_0}$$

观察：execution tokens（灰线）的 RelPPL 在训练早期 *急剧下降*，然后 plateau（Fig 3 column 1）。这说明模型在 procedural 层面迅速变得 confidently correct。

**Token-Level Entropy**（next-token policy 的 Shannon entropy）：

$$H\big(\pi(\cdot \mid \mathbf{x}_{<t})\big) = -\sum_v \pi(v \mid \mathbf{x}_{<t}) \log \pi(v \mid \mathbf{x}_{<t})$$

- $\mathbf{x}_{<t}$：timestep $t$ 之前的 context
- $v$：vocabulary 中的 token
- $\pi(v \mid \mathbf{x}_{<t})$：policy 在该 context 下输出 $v$ 的概率

观察：execution tokens 的 token entropy 持续且显著低于 planning tokens（Fig 3 column 2）。模型在低层 *主动放弃 exploration*，converge 到可靠 ops。

**Takeaway 1**: Procedural consolidation 表现为 execution token 的 perplexity + token entropy 双降。模型先 build reliable toolbox，再让 frontier shift 到 strategy。

特别地，对于已经具备较强 procedural foundation 的 model（MiMO-VL-Instruct、Qwen3-4B-Instruct），Phase ① 几乎缺失，直接进入 Phase ②——这反向佐证了"低层不是瓶颈，strategy 才是"。

### Phase ② Strategic Exploration

**Semantic Entropy of Strategic Grams**：定义在 SGs 的频率分布上的 Shannon entropy：

$$H_{\text{sem}} = -\sum_{g \in \text{SGs}} p(g) \log p(g), \quad p(g) = \frac{\text{count}(g) \text{ in batch}}{\sum_{g'} \text{count}(g')}$$

- $g$：某个 Strategic Gram
- $p(g)$：该 SG 在 batch 中出现的相对频率

这跟 token entropy 的核心区别（Fig 4）：token entropy 按 token 个体聚合，被海量低层 token 主导；semantic entropy 按 *语义单元* 聚合，直接测 high-level action space 的覆盖。

观察（Fig 3 column 3，红曲线 ②）：
- Semantic entropy 稳定上升（Qwen 系列）或先降后升（Llama）
- **Conditional entropy of procedural n-grams given a preceding SG 保持稳定**——一旦 procedural skill 掌握，没有动力去探索"多种加法"，动力只在"如何组合 strategy"
- Semantic entropy 上升与 length scaling、validation accuracy 上升同步（column 4）

**Takeaway 2**: Performance gains 在 Phase ② 由 strategic exploration 驱动，标志是 SG 的 semantic entropy 上升，伴随 length scaling 与 sustained accuracy gain。

### 统一解释 puzzling phenomena

- **"Aha moment"** = 模型 discover 并 internalize 一个 powerful strategy（如 self-reflection / backtracking）的 behavioral signature。
- **"Length scaling"** = sophisticated strategy（case analysis、self-reflection）本身就需要更长 trace，所以 length 是 semantic diversity 的 by-product，不是 cause。
- **Token entropy 的复杂动态** = 被 vast majority of execution tokens 主导，掩盖了 strategic 层的 exploration。

**Takeaway 3**: Aggregate token-level entropy 是 *misleading compass* for exploration。Semantic entropy 才是正确的指标。

---

## 五、HICRA 算法

### Motivation
如果 learning frontier 在 Phase ② 已 shift 到 strategic planning，那 GRPO 这种 agnostic credit assignment（所有 token 拿同一个 trajectory-level advantage）是 *inefficient* 的——梯度被稀释到对 final reward 贡献微弱的 procedural token 上。

### GRPO baseline
给定 query $\mathbf{q}$，policy $\pi_\theta$ 生成 $G$ 个 trajectories $\{\mathbf{o}_1, \dots, \mathbf{o}_G\}$。token $o_{i,t}$ 的 advantage：

$$\hat{A}_{i,t} = R(\mathbf{q}, \mathbf{o}_i) - \frac{1}{G}\sum_{j=1}^{G} R(\mathbf{q}, \mathbf{o}_j)$$

- $\mathbf{q}$：query
- $\mathbf{o}_i$：第 $i$ 条 trajectory
- $o_{i,t}$：trajectory $i$ 在 timestep $t$ 的 token
- $R(\mathbf{q}, \mathbf{o}_i)$：trajectory $i$ 的 reward（sparse，通常 0/1）
- $G$：group size
- $\hat{A}_{i,t}$：group-normalized advantage，对 trajectory 内所有 token 相同

参考 GRPO：
- DeepSeek-R1 / GRPO paper: https://arxiv.org/abs/2501.12948
- DAPO: https://arxiv.org/abs/2503.14476

### HICRA 的 modification

令 $S_i$ = trajectory $\mathbf{o}_i$ 中 planning token 的 index 集合（用 SG 检测器标定）。HICRA advantage：

$$\hat{A}_{i,t}^{\text{HICRA}} = \begin{cases} \hat{A}_{i,t} + \alpha \cdot |\hat{A}_{i,t}| & \text{if } t \in S_i \\ \hat{A}_{i,t} & \text{if } t \notin S_i \end{cases}$$

变量解释：
- $\hat{A}_{i,t}$：原 GRPO advantage
- $\alpha \in (0,1)$：amplification 强度超参，paper 用 $\alpha = 0.2$
- $|\hat{A}_{i,t}|$：advantage 的绝对值
- $S_i$：trajectory $i$ 中 planning token 的 index set

RL objective 与 policy gradient：

$$\mathcal{J}(\theta) = \mathbb{E}_{\mathbf{q}\sim\mathcal{D},\, \mathbf{o}_i\sim\pi_\theta}\Big[\hat{A}_{i,t}^{\text{HICRA}}\Big]$$

$$\nabla \mathcal{J}(\theta) = \mathbb{E}\Big[\hat{A}_{i,t}^{\text{HICRA}} \cdot \nabla \log \pi_\theta(o_{i,t} \mid \mathbf{q}, \mathbf{o}_{i,<t})\Big]$$

（paper 里写的是简化版，没带 PPO clipping；实际实现仍带 clip-higher，参照 DAPO）。

### 这个设计的 intuition

分两种情况拆开看：

**成功 trajectory（$\hat{A}_{i,t} > 0$）**：
$$\hat{A}_{i,t}^{\text{HICRA}} = \hat{A}_{i,t} + \alpha \cdot \hat{A}_{i,t} = (1+\alpha) \hat{A}_{i,t}$$
→ planning token 拿到 *放大* 的正向 credit，把"这次成功主要是因为用了某个 strategy"这个信号更显式地注入 policy gradient。

**失败 trajectory（$\hat{A}_{i,t} < 0$）**：
$$\hat{A}_{i,t}^{\text{HICRA}} = \hat{A}_{i,t} + \alpha \cdot (-\hat{A}_{i,t}) = (1-\alpha) \hat{A}_{i,t}$$
→ planning token 的 *penalty 被衰减*。这避免了在失败 trajectory 里把策略性 token 过度打压——因为你 explore 了 "let me try another approach" 但 execution 出错导致整条 trajectory 拿负 reward，不应该把锅扣在"尝试新策略"上。

这个 asymmetric design 实际上就是 *optimism in the face of uncertainty* 的一个变体，鼓励 strategic exploration 同时减少 false negative 的惩罚。

### 与 implicit target distribution 的关系

paper appendix D 给了一个优雅的视角：standard policy gradient 等价于把 policy 推向一个 advantage-shaped target distribution：

$$\pi^*(o_{i,t} \mid \mathbf{q}, \mathbf{o}_{i,<t}) \propto \pi_{\theta_{old}}(o_{i,t} \mid \mathbf{q}, \mathbf{o}_{i,<t}) \exp\Big(\hat{A}_{i,t}\Big)$$

- $\pi_{\theta_{old}}$：更新前的 policy
- $\exp(\hat{A}_{i,t})$：把 advantage 转成 multiplicative reweighting

带温度 $\beta$ 的版本：

$$\pi^*(a \mid s) = \frac{1}{Z(s)} \pi_{\theta_{old}}(a \mid s) \exp\Big(\hat{A}(a,s)/\beta\Big)$$

- $Z(s)$：partition function，归一化
- $\beta$：温度

minimize $KL(\pi^* \| \pi_\theta)$ 等价于 maximize $\mathbb{E}[\hat{A} - \beta KL(\pi_\theta \| \pi_{\theta_{old}})]$，这就是 PPO-KL objective。

HICRA 把 $\hat{A}$ 替换成 $\hat{A}^{\text{HICRA}}$，等于把 target distribution **anisotropically** 拉向 strategic dimension：planning token 的概率被更剧烈地 boost（正 trajectory）或更轻微 suppress（负 trajectory）。作者把这个称作 "anisotropic reshaping"，在 action space 的 strategic 子空间上施加更高的 KL 压力。

参考：
- PPO: https://arxiv.org/abs/1707.06347
- Williams REINFORCE: https://link.springer.com/article/10.1007/BF00992696

### HICRA vs Entropy Regularization 的关键差异

Entropy Regularization（Cheng et al., 2025, https://arxiv.org/abs/2506.14758）在所有 token 上加 entropy bonus，效果是：

- Token entropy 升上去 ✓
- 但 validation accuracy 不涨甚至下降 ✗
- Length uncontrolled scaling ✗

为什么？因为 indiscriminate entropy bonus 主要膨胀的是低层 execution token 的多样性，"5+3" 可以写成 "5 plus 3" / "adding 5 and 3" / "5 and 3 sum to"，entropy 涨了，reasoning 没变好。

HICRA 反过来：semantic entropy 涨上去（strategy 多样性），validation accuracy 跟着涨。Fig 7 是这个对比最干净的证据。

---

## 六、实验结果

### Text-only benchmarks（Table 1）

亮点数据：

| Model | Benchmark | GRPO | HICRA | Δ |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | AIME24 | 68.5 | 73.1 | **+4.6** |
| Qwen3-4B-Instruct-2507 | AIME25 | 60.0 | 65.1 | **+5.1** |
| Qwen3-4B-Base | AIME24 | 24.9 | 31.0 | **+6.1** |
| Qwen3-4B-Base | Math500 | 83.0 | 89.0 | **+6.0** |
| Qwen2.5-7B-Base | AIME24 | 16.3 | 18.8 | +2.5 |
| Qwen2.5-7B-Base | AMC23 | 46.7 | 55.1 | **+8.4** |
| Llama-3.1-8B-Instruct | AIME24 | 8.9 | 8.3 | **-0.6** ❌ |

观察：
1. **越弱的 base model（base < instruct）提升越大**——因为弱模型还有大量 strategic frontier 没探索。
2. **Llama-3.1-Instruct 不涨反跌**——这是个 *negative result*，下节分析。
3. HICRA 在 Qwen2.5-7B-Base 上与 ORZ 持平（AIME24 都是 18.8），但 ORZ 是 Open-Reasoner-Zero（用了更多 trick）；HICRA 仅靠一个 advantage 修正就追平。
4. AIME25 这种 *out-of-distribution* benchmark 的提升尤其显著（+5.1 on Qwen3-4B-Instruct）——暗示 HICRA 学到的是 *generalizable strategic skill*，而不是过拟合到 training distribution。

参考：
- ORZ: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
- SimpleRL: https://hkust-nlp.notion.site/simplerl-reason
- DeepScaler: https://github.com/agentica-project/deepscaler

### Multimodal benchmarks（Table 2）

| VLM | Benchmark | GRPO | HICRA | Δ |
|---|---|---|---|---|
| MiMO-VL-Instruct-2508 | MathVista | 73.7 | 80.7 | **+7.0** |
| MiMO-VL-Instruct-2508 | MathVision | 42.8 | 48.9 | **+6.1** |
| Qwen2.5-VL-7B-Instruct | MathVision | 25.8 | 28.7 | +2.9 |

MiMO-VL 的提升幅度比 text model 还大——作者的解释是 VL 任务里 base model 已经快速把 visual grounding 这种 procedural skill 学会，strategic frontier 完全没被 exploit。

### Error type 分析（Fig 5）

用 GPT-4o 把失败 trajectory 分成 4 类（A: dummy、B: low-level execution、C: high-level plan flaw、D: actually correct），再合并成 "Planning & Strategy" vs "Others"。

观察：训练过程中 *Planning & Strategy 错误下降幅度远大于 Others*。对 Qwen2.5-7B-Base 尤其明显——low-level error 几乎不降，但 model accuracy 仍涨，说明 RL 的杠杆主要在 strategic 层。

这条证据很关键，因为它直接反驳了"RL 主要让模型做对算术"的直觉。

### HICRA vs high-entropy "fork tokens"（Fig 9, 10）

最近 Wang et al., 2025d（"Beyond the 80/20 rule"，https://arxiv.org/abs/2506.01939）提出把 top-entropy token 当 "fork token" 作为 decision point 的 proxy。

paper Fig 10 给的对照：
- 70%+ 的 planning token 在 top-30% high-entropy token 里（左图）→ planning token 通常确实是高熵点，对。
- 但只有 <10% 的 high-entropy token 是 planning token（右图）→ 用 high entropy 作为 strategic proxy **recall 高但 precision 极低**。

为什么？大量高熵 token 是 *phrasing variants*（"therefore" / "thus" / "so" / "hence"）或 *calculation intermediate*（多位数计算的下一步选择空间大），它们熵高但 *not* strategic decision point。functional definition 比 statistical definition 更可靠。

这个对比也间接回答了一个有意思的问题：**是不是所有"重要决策点"都伴随高熵？** 答案倾向于"是"，但反命题不成立——高熵 ≠ 重要决策点。

---

## 七、Semantic Entropy 作为 exploration compass

Fig 8（MiMO-VL-7B）的实验是关键：
- Token entropy 在 HICRA 和 GRPO 上都 *collapse*（被 low-level token 主导）
- Pass@8 早早饱和，分不出方法差异
- Semantic entropy 保持高位，且 HICRA > GRPO，这个 gap 与最终 validation accuracy 的 gap 一一对应

这对 RL practitioner 的实操意义：
1. **别再用 token entropy 判断 exploration 是否还在进行**——它在多数 LLM RL setting 下是 broken metric。
2. **Pass@K 在大 K 下 saturate 太快**，对小模型 + 简单任务可能有用，对 reasoning 任务不再 informative。
3. **Semantic entropy（或者更广义的 semantic-level diversity metric）是 better compass**。

这与最近 Farina et al. 在 process reward model 上的观察类似——*outcome-level reward signal 太稀疏，需要 process-level / semantic-level signal*。

---

## 八、Llama-3.1 的 negative result 与 HICRA 的依赖

Appendix E.2 揭示一个重要 caveat：HICRA *presumes a procedural foundation*。

Llama-3.1-8B-Instruct 在 HICRA 下反而比 GRPO 差（AIME24: GRPO 8.9 vs HICRA 8.3）。Fig 14 显示 semantic entropy 出现"反向 trend"——HICRA 鼓励 Llama explore strategic plan，但 Llama 的 procedural execution 不够稳，explore 出来的 plan 没法 execute 出正 reward，结果 advantage noise 反而干扰 training。

直觉：HICRA 是个 *amplifier*，它假设 underlying signal 存在。如果 base model 还在 procedural consolidation phase，把 strategic dimension 的 gradient 放大反而 inject 噪声。

对实践的指导：
- **Procedural foundation 不足的 model**（base model 算术能力 < 60%？）应该先用 SFT 或 vanilla GRPO 跑一段，再切到 HICRA。
- **curriculum / phased training** 可能是自然的延伸：先 GRPO 再 HICRA。

---

## 九、与相关工作的脉络

### Hierarchical RL (HRL) 的脉络
经典 HRL（Options Framework, Sutton-Precup-Barto 1999；MAXQ Dietterich 2000）把 policy 分成 high-level manager（选 option）和 low-level worker（执行 option）。HICRA 不引入显式 hierarchy，而是 *在 flat policy gradient 里通过 advantage shaping 隐式分离 hierarchy*——这更像 Feudal Networks (Vezhnevets et al., 2017) 的思路，但作用在 token 级。

参考：
- Options Framework: https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-1999.pdf
- FeUdal Networks: https://arxiv.org/abs/1703.01161

### Hierarchical Reasoning Model (HRM)
Wang et al., 2025a 的 HRM（https://arxiv.org/abs/2506.21734）显式构造一个 two-level neural architecture（thought generator + stepwise reasoner）。HICRA 跟它哲学相似（都把 hierarchy explicit）但实现 orthogonal——HRM 改架构，HICRA 改 objective。

### System 1/System 2 在 LLM 上的实现
Kahneman 的 dual process theory 已经被多个工作引用（Anthropic 的 "thinking" model、OpenAI o1、Sutton 的 "bitter lesson" 谈到的 search）。HICRA 提供了一个 *training-side* 的实现：通过 reward shaping 让 System 2 的 strategic move 更显式。

参考：
- Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- Sutton on System 1/2: https://richsutton.com/2024/08/13/the-era-of-experienced-system-2-is-coming.html

### Process Reward Model (PRM) 视角
PRM（Lightman et al., 2023, https://arxiv.org/abs/2305.20050）给每个 step 估 value。HICRA 不估 step value，只做 binary mask（planning vs not）+ scalar amplification——更轻量、不需要 trained reward model。可不可以把两者结合？比如用 PRM 给每个 planning token 不同的 $\alpha_t$？这是个 obvious next step。

### Credit Assignment Problem
经典 RL 里 credit assignment 是个核心难题（Sutton 1984）。在 LLM RL 里，trajectory-level reward 给所有 token 同一个 advantage 是个 brutal approximation。HICRA 是 *rule-based credit assignment*，借鉴 hierarchical decomposition。未来一个方向是 *learned credit assignment*：训一个 critic 估计每个 token 的 marginal contribution。

参考：
- Process Reward Models: https://arxiv.org/abs/2305.20050

### Advantage Shaping 的 related family
- **Clip-higher** (DAPO): 把 PPO clip 的上限抬高，鼓励 exploration。
- **Entropy Regularization** (Maximum Entropy RL, Haarnoja 2017): 给所有 token entropy bonus。
- **Token-level Reward / ReMEN** (Wang et al.): 用 implicit value function 分配 token-level credit。
- **HICRA**: hierarchical-aware credit，做 strategic token 的 asymmetric amplification。

HICRA 在这个 family 里独特之处是 *functional prior*——基于 semantic role 而不是统计量（entropy、frequency）分配 credit。

参考：
- DAPO / clip-higher: https://arxiv.org/abs/2503.14476
- Soft Actor-Critic: https://arxiv.org/abs/1801.01290

### R1-Zero-like Training
Liu et al., 2025b（https://arxiv.org/abs/2503.20783）对 R1-zero training 做了 critical perspective，观察到 entropy collapse、length scaling 等。HICRA 的 two-phase dynamic 给了这些现象一个统一解释，并可操作地缓解 entropy collapse 的 negative consequence。

---

## 十、对 paper 的 critique 与 open questions

### 我觉得强的部分
1. **Phenomenon-driven 的分析框架**——8 个 model 都展现同一个 two-phase pattern，cross-model consistency 说服力强。
2. **Semantic entropy 这个 metric** 比 token entropy / Pass@K 更 informative，是个 useful diagnostic 工具，对社区有工具价值。
3. **Asymmetric advantage shaping** 这个设计很 elegant，单独看公式非常简单，但 *functional prior* 让它 distinguish 于 entropy regularization。
4. **AIME25 上 OOD 提升**说明学到的是 generalizable strategic skill，不是过拟合。

### 我觉得弱的 / open 的部分

1. **SG 构造的 sample efficiency**：依赖一个 large corpus of successful solutions。对 cold-start 场景（base model 在某 task 上几乎全错）不友好。可不可以做 *online SG discovery*？
2. **SG 是英文的**：迁移到中文 / 多语言需要重新构建。paper 完全没讨论。
3. **$\alpha = 0.2$ 是 magic number**：没给 ablation。会不会与模型 size、task difficulty 有 interaction？
4. **Planning token 的 binary mask 过于粗**：所有 SG 一视同仁，"let me think" 和 "let's backtrack and reconsider our entire approach" 给同样的 amplification。一个 PRM-like 的 per-token $\alpha_t$ 可能更好。
5. **Phase boundary 不清晰**：paper 说 phase overlapping，但没给一个 quantitative criterion 判断 "现在该不该切 HICRA"。这是 practical deployment 的痛点。
6. **Llama 的 negative result 处理得有点轻**：只说"procedural foundation 不足"，但没给一个 base model quality 的 diagnostic metric 来预判 HICRA 是否适用。
7. **没比较 Hybrid**：先 GRPO 后 HICRA 的 curriculum 策略没试过，直觉上对 Llama 这种 weak-procedural base 应该 work。

### 与你（Karpathy）过往工作的联系
你最近在讲 "Software 2.0 / 3.0"、"natural language as programming language"。HICRA 实际上是在 RL training 时给 *natural language program 的 control flow keyword* 一个放大信号——某种意义上是把 "let's reconsider" 这种 phrase 当成 implicit *goto / break / continue* 指令，并通过 reward shaping 强化这些 control flow 的学习。这与你的"用自然语言写程序"的视角很契合：HICRA 让 LLM 学会用 *语言* 来表达 *control flow*，而非仅用语言表达 *content*。

你的 micrograd / nanoGPT 的教学也强调过 "看 gradient flow" 的 intuition。HICRA 改的就是 gradient flow 的方向：把 gradient mass 从 procedural 维度抽出来，投到 strategic 维度。

参考：
- Karpathy Software 2.0: https://karpathy.medium.com/software-2-0-a64152b37c35
- nanoGPT: https://github.com/karpathy/nanoGPT
- micrograd: https://github.com/karpathy/micrograd

### 与 "AI OS" / agentic 工具调用的关联
作者在 conclusion 里提到未来方向：code generation 与 agentic tool use。直觉上 agentic setting 更需要 hierarchical credit assignment：
- High-level = "现在该调用搜索 / 该写代码 / 该问用户"
- Low-level = 具体工具参数

GRPO 在 agentic setting 下做 trajectory-level reward 时，"该不该调用工具"这个决策与"调用的具体参数"被同权重处理，这与本 paper 的 thesis 完全 mirror。HICRA 在 agentic RL 上的延伸几乎是 inevitable 的下一步。

---

## 十一、Build intuition 的 mental model

我读完之后的一个 mental model：

想象 LLM 的 reasoning 是一棵 *decision tree*——
- 叶节点 = low-level execution（"算 23 × 47"）
- 内部节点 = strategic move（"let's try a different approach"）

Pre-training 让这棵树有了大量 "潜在节点"。RL 是在树上做 policy iteration：
- Phase ①: 先把每个叶节点的"算对率"拉到 1.0。这是快速的，因为 leaf space 小。
- Phase ②: 在内部节点做 explore-exploit。这是慢的，因为 branching factor 高。

GRPO 把同样的 advantage 推给叶子和内部节点——叶子上 gradient 浪费在"如何更 confidently 算 23×47"，内部节点上 gradient 又被稀释。

HICRA 在内部节点上 *加大 gradient step*，叶子节点不动。结果：树往上长（更多 strategy），不在叶子上空耗 compute。

Llama 失败的原因：Llama 的叶子节点本身不稳，把内部节点推出去探索反而让叶子更不稳，树长歪了。

这个 mental model 跟 AlphaGo 的 policy iteration 也有点像——rollout policy（叶节点估值）和 move policy（内部节点选择）的迭代。MCTS 里两个 policy 分别训，LLM RL 里两个 policy 共享一个网络，HICRA 通过 advantage shaping 把 gradient 分别投到两套"虚拟参数"上。

参考：
- AlphaGo: https://www.nature.com/articles/nature16961
- Silver et al. on policy iteration: https://www.nature.com/articles/nature14236

---

## 十二、对自家训练的可能 implication

如果你（Karpathy）在做 eureka labs / 自己的 reasoning model 训练，HICRA 的 takeaway 可能是：

1. **监控指标改造**：用 semantic entropy 而非 token entropy 判断 exploration 健康度。
2. **训前诊断**：先在小 train set 上跑 vanilla GRPO，看 procedural error vs strategic error 的比例，决定 HICRA 何时切入。
3. **Generalizable SG set**：构建一个跨 domain 的 SG library（math + code + tool use），让 HICRA 一开始就作用于全 strategic subspace。
4. **与 PRM 结合**：用 PRM 给每个 planning token 不同的 $\alpha_t$，替代固定 $\alpha = 0.2$。
5. **Self-distillation 视角**：HICRA 学到的 strategic policy 可以 distill 回 base model 作为 improved prior，给下一轮 RL 用。

---

## 总结

这篇 paper 的核心 contribution 不是 HICRA 算法本身（公式简单到一行），而是 *揭示了 LLM RL 训练中 emergent 的 functional hierarchy*——一个能在 8 个模型上复现的 *现象*。这个现象的诊断工具（semantic entropy、SG-based token classification）+ 一个对应 intervention（HICRA）的组合拳，让 paper 既有 analysis 又有 method 又有 experiment。

最让我个人震撼的不是 HICRA 的效果，而是 Fig 5 那张 error decomposition 图：**RL 训练后低层错误几乎不降，但 accuracy 大涨——所有 gains 来自高层 strategy 的修复**。这把 RL for LLM reasoning 的机理从"让模型算得更准"重新定位到"让模型想得更对"。

如果未来有 LLM RL 的 *mechanistic interpretability* 方向（mech interp on RL dynamics, 而不仅是 forward pass），这篇 paper 提供了一个很好的 phenomenon 和 metric 框架。

完整 reference list：
- Paper project page: https://tiger-ai-lab.github.io/Hierarchical-Reasoner/
- DeepSeek-R1 / GRPO: https://arxiv.org/abs/2501.12948
- DAPO: https://arxiv.org/abs/2503.14476
- PPO: https://arxiv.org/abs/1707.06347
- Fork tokens: https://arxiv.org/abs/2506.01939
- HRM: https://arxiv.org/abs/2506.21734
- Entropy Regularization in RL reasoning: https://arxiv.org/abs/2506.14758
- Murray et al. cortical timescale hierarchy: https://www.nature.com/articles/nn.3862
- Zeraati et al. attention and intrinsic timescales: https://www.nature.com/articles/s41467-023-37437-3
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Llama-3: https://arxiv.org/abs/2407.21783
- Sentence Transformers: https://www.sbert.net/
- Open-Reasoner-Zero: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
- SimpleRL: https://hkust-nlp.notion.site/simplerl-reason
- DeepScaler: https://github.com/agentica-project/deepscaler
- Process Reward Models / Let's Verify Step by Step: https://arxiv.org/abs/2305.20050
- AlphaGo Nature paper: https://www.nature.com/articles/nature16961
- Feudal Networks: https://arxiv.org/abs/1703.01161
- Soft Actor-Critic: https://arxiv.org/abs/1801.01290
- R1-Zero-like training critical perspective: https://arxiv.org/abs/2503.20783
- Karpathy Software 2.0: https://karpathy.medium.com/software-2-0-a64152b37c35
- The Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
