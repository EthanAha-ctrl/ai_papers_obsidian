---
source_pdf: Self-Consolidation for Self-Evolving Agents.pdf
paper_sha256: 4a349f2ae03063d205274f7587ac23f783ed99a3302bfa55965c1eb783954071
processed_at: '2026-08-12T04:36:06-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 EvoSC

## 一句话版本

让 LLM agent 干活儿的时候别每次都从零开始——**把成功和失败的经验攒下来，但攒的方式很聪明：先总结成几条文字 tips，再把这些 tips 进一步压成 20 个"虚拟词"，塞进 prompt 前面当直觉用**。这样既不爆 context window，又能让 agent 越干越熟练。

---

## 为什么需要这东西

想象你雇了个实习生，每次给他布置任务他都忘掉上次干过啥，每次都重新摸索。你肯定想揍他对吧。

现在 LLM agent 基本就这状态。完成一个 task，下个 task 来了，记忆清零。

有人想了个办法：**把过去成功的对话存起来，下次遇到类似 task 就翻出来塞 prompt 里当范例**。这就是 AWM (https://arxiv.org/abs/2409.07429) 和 TER (https://arxiv.org/abs/2505.11942) 干的事。

但这招有两个坑：

**坑一：只看成功，不看失败。** 实习生光看你表扬他"这次做对了"，但你不告诉他"上次那个搞砸了是因为 X 原因"，他下次还会犯同样的错。失败的 trajectory 里藏着很值钱的"别这么干"信号，白白浪费了。

**坑二：经验越攒越多，prompt 越来越长，最后爆了。** LLM 的 context window 是有限的（8B 模型一般 128k token，Qwen 2.5-7B 更紧）。你存了 30 条 trajectory，每条几千 token，光历史经验就吃掉一半 context，留给当前任务的就没多少了。而且塞太多 demo 进去，LLM 反而 attention 分散、精度下降（"Lost in the middle" 现象，参见 Liu et al. https://arxiv.org/abs/2307.03172）。

实验里这俩坑直接体现为：AWM、TER 这些 baseline 在 KG 任务上 Exp=16 就 **OOM** 跑不了了。

---

## EvoSC 怎么解决

### 第一招：对比着学（contrastive reflection）

不只存成功的对话，**同时拿一条成功 + 一条失败，让 LLM 自己 diff 一下**，问它："这俩过程哪一步开始走岔了？是什么导致了失败？提炼出几条避坑指南。"

公式（论文 eq.3）：
$$\mathrm{Exp}_c = \mathrm{LLM}(\mathcal{P}_c \cup \mathcal{C}_s \cup \mathcal{C}_f)$$

- $\mathcal{P}_c$：contrastive prompt template，就是问 LLM "请对比这两段对话找差异"
- $\mathcal{C}_s$：一条成功对话
- $\mathcal{C}_f$：一条失败对话
- $\mathrm{Exp}_c$：输出，几条 error-prone insight，比如 KG 任务里就是"用 `get_neighbors()` 时 relation type 要精确匹配图结构"、"遍历顺序要先 `get_relations()` 再 `get_neighbors()` 再 `get_attributes()` 最后聚合"

这就比单纯存原始 trajectory 高效多了——**你把 2000 token 的对话压成 50 token 的精华 tips**。

类似的，成功经验也抽（eq.4）：
$$\mathrm{Exp}_s = \mathrm{LLM}(\mathcal{P}_s \cup \mathcal{C}_s^{(i)} \cup \mathcal{C}_s^{(j)})$$

注意这里**喂两条不同的成功对话** $\mathcal{C}_s^{(i)}$ 和 $\mathcal{C}_s^{(j)}$，逼 LLM 找共同点，避免它只是 memorize 单次 trace。这跟 ExpeL (https://arxiv.org/abs/2308.10144) 思路一致。

这俩 insight 各自进一个 FIFO queue，用新的挤掉旧的，保证 relevance。

### 第二招：把经验压进参数（parametric trajectory consolidation）

这是真正的核心招。

文字 tips 再精炼也是文字，还是占 context slot。如果历史有 100 条 trajectory，你哪怕抽成 tips 也不少。

所以作者想：**能不能把"看过 20 条 trajectory 后的推理能力"直接压进几个 virtual token 里？**

做法分 teacher / student 两步：

**Teacher**：给 LLM 看 20 条成功 trajectory 当 in-context demo，让它做当前任务：
$$A_{k,s}^* = \mathrm{LLM}(\mathcal{E}_{\mathrm{many}} \cup \mathcal{H}_{k,s-1} \cup \mathcal{T}_k)$$
$\mathcal{E}_{\mathrm{many}}$ 是 20 条 trajectory，$\mathcal{H}_{k,s-1}$ 是当前 task 已经进行的历史，$\mathcal{T}_k$ 是 task 描述。$A_{k,s}^*$ 就是"看了一堆范例后做出的聪明决策"。

**Student**：只给 LLM 看 8 条 trajectory，但在 prompt 最前面加 20 个 **learnable soft token** $\mathcal{P}_\theta$（就是 20 个连续向量，可以梯度更新）：
$$\hat{A}_{k,s} = \mathrm{LLM}(\mathcal{P}_\theta \cup \mathcal{E}_{\mathrm{few}} \cup \mathcal{H}_{k,s-1} \cup \mathcal{T}_k)$$

训练目标（eq.7）就是让 student 模仿 teacher：
$$\mathcal{L} = -\sum_s \sum_j \log P_\theta(A_{k,s,j}^* \mid \mathcal{P}_\theta, \mathcal{T}_k, \mathcal{H}_{k,s-1}, A_{k,s,<j})$$

- $A_{k,s,j}^*$：teacher 输出的第 j 个 token
- $A_{k,s,<j}$：teacher-forcing 下已经生成的前 j-1 个 token
- 只更新 $\mathcal{P}_\theta$ 这 20 个 token 的 embedding，**LLM 主干冻结**

**直觉**：当 student 用 (8 条 demo + 20 个 soft token) 能复现 teacher 用 20 条 demo 给出的答案时，那 20 个 soft token 就"吸收"了剩下 12 条 demo 的信息。这相当于把"长 context 里的知识"蒸馏进了"短 context + 一小段 latent code"。

这就是 **context distillation** 思想，Anthropic 2021 年就玩过（https://arxiv.org/abs/2104.08773），把长 prompt 的效果压进 model weights。EvoSC 没动 weights（成本太高），改压进 prefix soft prompt，思路同源。

人脑类比：这就像你看了 100 本烹饪书，一开始做菜还得翻书（in-context），后来看多了，不用翻书也能凭"手感"做（parametric memory）。 Hippocampus 负责"记住具体哪本书哪页讲了啥"，neocortex 负责"内化成手艺"。EvoSC 的 $\mathcal{P}_\theta$ 就是那个"内化的手艺"。神经科学背景参见 Spens & Burgess 2024 Nature Human Behaviour (https://www.nature.com/articles/s41562-024-01838-7)。

### 第三招：推理时两层一起上

新任务来了，prompt 这样拼：
$$\mathcal{T}_k = \mathcal{P}_\theta \oplus \mathcal{P}_{\mathrm{sys}} \oplus \mathrm{Exp}_c \oplus \mathrm{Exp}_s \oplus \mathcal{C}_s \oplus t_k$$

从最 implicit 到最 explicit：
- $\mathcal{P}_\theta$：20 个 soft token，**直觉**（参数化、长期、压缩过的经验）
- $\mathcal{P}_{\mathrm{sys}}$：domain 规则（静态）
- $\mathrm{Exp}_c$：避坑 tips（最近从失败里抽的）
- $\mathrm{Exp}_s$：成功模式（最近从成功里抽的）
- $\mathcal{C}_s$：最近几条成功对话原文（in-context demo）
- $t_k$：当前任务

相当于人同时用"多年手感 + 临场查的笔记 + 翻出来几页书"三种 memory 协同工作。

---

## 实验结果怎么读

三个 benchmark（来自 LifelongAgentBench https://arxiv.org/abs/2505.11942）：
- DB（500 个 SQL 任务，最多 3 round）
- OS（500 个 shell 任务，最多 5 round）
- KG（396 个知识图谱 API 任务，最多 15 round，long-horizon）

**DB on Llama 3.1-8B**：EvoSC avg 65.1，最强 baseline A-MEM 58.4，**+6.7**

**KG on Qwen 2.5-7B**：EvoSC avg 38.4，最强 baseline TER 27.8，**+10.6**

KG 提升最大，因为 15 round 的 long-horizon 任务里，textual replay 爆 context 最快，PTC 压缩价值最大。

**OOM 现象**是关键卖点：所有 baseline 在 Exp=32 (Qwen DB/OS) 或 Exp=16 (KG) 时直接跑不了，因为 prompt 超过 context window。EvoSC 能一直 scale 上去，因为多出的 trajectory 被 PTC 吃进 $\mathcal{P}_\theta$ 了，context 占用恒定。

**Ablation 亮点**：EE（error experience）和 SE（success experience）单独去掉，Avg 都掉 3-4 个点，说明**失败的教训和成功的经验一样值钱**，contrastive 思想被实验验证。

---

## 这篇 paper 真正聪明的地方

1. **不强行 fine-tune LLM weights**。fine-tune 8B 模型成本高、有 catastrophic forgetting 风险、不 plug-and-play。EvoSC 只训 20 个 soft token，GPU 几分钟搞定，主干冻结，可以随便换 backbone。

2. **Failure 信号被显式利用**。之前的工作（AWM、TER、ExpeL）几乎都只看 success，浪费了一半数据。EvoSC 用 contrastive 把 failure 变成 "anti-pattern" 写进 prompt。

3. **Hybrid memory 架构符合认知科学**。working memory（in-context tips + demos）+ long-term memory（soft prompt），跟人脑 hippocampus + neocortex 分工对应。这不是硬套 analogy，是真的在工程上 work。

---

## 真正的弱点

1. **Retrieval 太朴素**。FIFO + top-K recent，完全没用 semantic retrieval。如果换成 embedding similarity + recency + importance 三权重打分（Generative Agents 那套 https://arxiv.org/abs/2304.03442），应该还能涨。

2. **Length=20 没扫 ablation**。为什么不是 50、100？capacity 和 noise 的 tradeoff 没研究透。

3. **没在 70B+ 验证**。7B/8B 上 in-context 能力弱，soft prompt 增益明显；70B+ in-context 本身就强，soft prompt 边际收益可能衰减。这是 scaling law 角度的 open question。

4. **Cross-domain forgetting 没测**。$\mathcal{P}_\theta$ 只有 20 个 token，如果先学 DB 再学 OS，DB 的经验会被覆盖掉。论文里三个 domain 是独立跑的，没测 lifelong cross-domain。

5. **PTC 训练成本没量化**。teacher forward + student forward + backward，实际 GPU hours 和 token consumption 没列，reproduce 有点难。

---

## 对你的 intuition building 价值

如果你在做 agent 系统，这篇 paper 给的核心启发是：

**textual memory 是 working memory，parametric memory 是 long-term memory，两者必须分层**。纯 textual 走不远（OOM），纯 parametric 太贵且 forget，hybrid 才是 pragmatic 路径。

context distillation 这个工具被严重低估了。它比 RLHF 便宜得多，比 in-context 优雅得多，但社区用得很少。EvoSC 把它用在了 agent lifelong learning 这个具体场景，其实这工具能用在很多地方：把 RAG 的长 context 压进 prompt token、把 chain-of-thought 蒸进 prefix、把 multi-turn dialog history 压成 latent。Anthropic 早就这么干了，但学术界跟进得慢。

参考 link：
- Context Distillation 原始 paper: https://arxiv.org/abs/2104.08773
- Prefix Tuning: https://arxiv.org/abs/2101.00190
- Prompt Tuning: https://arxiv.org/abs/2104.08691
- Constitutional AI（也用了 distillation）: https://arxiv.org/abs/2212.08073
- LifelongAgentBench: https://arxiv.org/abs/2505.11942
- AWM: https://arxiv.org/abs/2409.07429
- A-Mem: https://arxiv.org/abs/2502.12110
- ExpeL: https://arxiv.org/abs/2308.10144
- Reflexion: https://arxiv.org/abs/2303.11366
- Generative Agents: https://arxiv.org/abs/2304.03442
- Lost in the middle: https://arxiv.org/abs/2307.03172
- Memory consolidation 神经科学: https://www.nature.com/articles/s41562-024-01838-7

---

# EvoSC: Self-Consolidation for Self-Evolving Agents 深度解析

## 1. 一句话定位与核心 motivation

这篇 paper 想做的事情非常清晰: 给 LLM agent 装一个**终生学习**机制, 让它在 sequential task stream 上不断积累经验, 同时避免 context window 爆炸。作者的核心 insight 来自认知科学里的 **memory consolidation** 现象——人脑不会把所有 episodic experience 原样堆在 working memory 里, 会在睡眠 / 反思时把 high-value 的信息压缩成 semantic memory, 然后写入 neocortex 的 synaptic weights (Spens & Burgess, 2024, *Nature Human Behaviour* https://www.nature.com/articles/s41562-024-01838-7)。

EvoSC 把这套机制映射成两段式 pipeline:
- **Non-parametric contrastive extraction**: working-memory 层, 显式提取 error-prone insights 和 success patterns, 用 textual prompt 形式注入。
- **Parametric trajectory consolidation**: long-memory 层, 把大量 textual trajectory 蒸馏成一段 length=20 的 learnable soft prompt $\mathcal{P}_\theta$, 让"过去看过的 100 个 trajectory"被压缩进 20 个 virtual token。

这与 Anthropic 早期提出的 **context distillation** 思路同源 (Askell et al. https://arxiv.org/abs/2112.08895), 也与 prefix tuning / prompt tuning (Lester et al. 2021 https://arxiv.org/abs/2104.08691) 的 parameter-efficient 思想一脉相承, 只是把训练目标从"拟合下游 task label"换成了"拟合一个看多了 context 的 teacher 的输出分布"。

---

## 2. Problem Formulation 详解

作者用 very formal 的 RL-style 语言定义 lifelong agent:

- Domain: $\mathcal{D} = \langle \mathcal{P}_{\mathrm{sys}}, \mathcal{T} \rangle$, 其中
  - $\mathcal{P}_{\mathrm{sys}}$: universal system prompt, encode domain rules (e.g. SQL 语法、shell 命令)
  - $\mathcal{T} = \{t_1, t_2, \dots, t_N\}$: 顺序到达的 N 个 task instances
- 第 k 个 task 的 input context: $\mathcal{T}_k = \mathcal{P}_{\mathrm{sys}} \oplus t_k$, 这里 $\oplus$ 就是 string concatenation。
- 在每个 task 内 agent 最多交互 r rounds (DB=3, OS=5, KG=15), 第 s 步的 policy:
  $$A_{k,s} \sim \pi_\theta\left( A_{k,s} \mid \mathcal{H}_{k,s-1}, \mathcal{T}_k \right) \tag{1}$$
  变量含义: $A_{k,s}$ 是第 k 个 task 第 s 步的 action, $\pi_\theta$ 是 parameterized policy (LLM), $\mathcal{H}_{k,s-1}$ 是历史 interaction list, $\mathcal{T}_k$ 是 task prompt。
- 环境返回 feedback: $F_{k,s} \in \Omega$, history 更新为 $\mathcal{H}_{k,s} = \mathcal{H}_{k,s-1} \cup \{(A_{k,s}, F_{k,s})\}$。
- 二值 reward: $R(t_k) \in \{0,1\}$, lifetime objective:
  $$\max_{\pi_\theta} \sum_{k=1}^{N} \mathbb{E}_{\xi^{(k)} \sim \pi_\theta}\left[ R(t_k) \right] \tag{2}$$
  $\xi^{(k)} = (I_k, A_{k,1}, F_{k,1}, \dots, A_{k,T}, F_{k,T})$ 是一个完整 trajectory。

**Intuition**: 这个 formulation 把 agent lifelong learning 做成了一个 **online contextual bandit / RL** 问题, reward 稀疏 (只在 task 结束才给 0/1), 没有显式 gradient signal 流回 LLM weights。所以作者选择走 **in-context learning + soft prompt tuning** 这条路, 避开 PPO / REINFORCE 那种高方差 policy gradient 训练, 同时又不被局限于纯 textual replay。这是个相当务实的设计选择。

---

## 3. Method: 三个核心模块深度拆解

### 3.1 Non-parametric Contrastive Experience Extraction

**Error-prone experience**:
$$\mathrm{Exp}_c = \mathrm{LLM}\left( \mathcal{P}_c \cup \mathcal{C}_s \cup \mathcal{C}_f \right) \tag{3}$$
变量: $\mathcal{P}_c$ 是 contrastive prompt template (Figure 4 给了 KG 的例子), $\mathcal{C}_s$ 是一条 successful dialog, $\mathcal{C}_f$ 是一条 failed dialog。LLM 被要求做的是 **diff 操作**: 找出 success 和 failure 在哪个 reasoning step 分叉, 提炼出"该 task category 下的 error-prone patterns + 规避策略"。

这是 contrastive learning 思想的 textual 化版本, 类似 SimCLR / MoCo 在 representation space 做的 positive-negative 对比, 但这里对比的是 **reasoning trajectory**, 输出是 **symbolic rule**。它比纯 retrieval 的好处是: failure 蕴含的"反向信息"被显式表达出来, 而不是被 success-only 的 retrieval 给淹没。

**Successful experience**:
$$\mathrm{Exp}_s = \mathrm{LLM}\left( \mathcal{P}_s \cup \mathcal{C}_s^{(i)} \cup \mathcal{C}_s^{(j)} \right) \tag{4}$$
注意这里**用两个不同的 successful instances**做 multi-shot extraction, 而不是一个。这个设计的精妙之处在于强迫 LLM 找 **common pattern** 而不是 memorize 单次 trace, 起到了 in-context generalization 的作用。这跟 ExpeL (Zhao et al. https://arxiv.org/abs/2308.10144) 里用多条 success 抽 insight 的思路一致。

两个 queue 都是 **FIFO**, 这是个简化的 recency prior, 类似 RL 里的 replay buffer, 自动 prune 过时信息。作者在 limitations 里也承认 retrieval 太朴素, 未来可以上 semantic retrieval (e.g. embedding-based top-K, 类似 RAG)。

### 3.2 Parametric Trajectory Consolidation (最核心创新)

这是整篇 paper 的灵魂。我们慢一点讲。

**Setup**:
- $\mathcal{E} = \{\mathcal{C}_{\mathrm{succ}}^i\}_{i=1}^K$: 历史成功 trajectory 集合, K 可以很大。
- 当前 task $t_k$, 输入 $\mathcal{T}_k = \mathcal{P}_{\mathrm{sys}} \oplus t_k$。
- 该 task 的一条成功 trajectory $\mathcal{H}_k = \{(A_{k,s}, F_{k,s})\}_{s=1}^r$, r rounds。
- 引入一段 **learnable soft prompt** $\mathcal{P}_\theta$, length=20 (论文实验配置), 形状就是 `[20, d_model]` 的 continuous embedding 序列, 接在 input embedding 前面。

**Teacher forward (用 many trajectories 做 in-context reasoning)**:
$$A_{k,s}^* = \mathrm{LLM}\left( \mathcal{E}_{\mathrm{many}} \cup \mathcal{H}_{k,s-1} \cup \mathcal{T}_k \right) \tag{5}$$
$\mathcal{E}_{\mathrm{many}} \subset \mathcal{E}$: 给 teacher 喂很多 trajectory 当 in-context demos。这个 teacher action $A_{k,s}^*$ 就是"看过大量历史后做出的高 informed 决策"。

**Student forward (用 few trajectories + learnable prompt 模仿)**:
$$\hat{A}_{k,s} = \mathrm{LLM}\left( \mathcal{P}_\theta \cup \mathcal{E}_{\mathrm{few}} \cup \mathcal{H}_{k,s-1} \cup \mathcal{T}_k \right) \tag{6}$$
$\mathcal{E}_{\mathrm{few}} \subset \mathcal{E}_{\mathrm{many}}$: student 只看少量 trajectory, 但多了 $\mathcal{P}_\theta$ 这 20 个 soft token, 期望这 20 个 token 把 teacher 看到的"剩下的 12 个 trajectory 信息"压缩进去。

**Consolidation loss**:
$$\mathcal{L}_{\mathrm{consolid}} = -\sum_{s=1}^{r} \sum_j \log P_\theta\left( A_{k,s,j}^* \mid \mathcal{P}_\theta, \mathcal{T}_k, \mathcal{H}_{k,s-1}, A_{k,s,<j} \right) \tag{7}$$
变量逐项解释:
- 外层 $\sum_{s=1}^r$: 遍历 task 内 r 个 interaction rounds。
- 内层 $\sum_j$: 遍历 expert action token 序列的第 j 个 token。
- $A_{k,s,j}^*$: teacher 在第 s round 给出的 action 的第 j 个 token (ground truth target)。
- $A_{k,s,<j}$: student autoregressive decoding 已生成的前 j-1 个 token (teacher-forcing 时直接用 teacher 的 prefix)。
- $P_\theta(\cdot)$: student LLM 在 given context 下的 next-token probability, $\theta$ 只更新 $\mathcal{P}_\theta$ 这 20 个 token 的 embedding, LLM 主体冻结。
- 负号 + log: 标准 cross-entropy, 让 student 分布逼近 teacher 的 action token。

**这就是 context distillation 的经典配方**, 类似 DistilBERT / TinyBERT 把大模型知识压到小模型, 但这里压的不是"大模型→小模型", 而是"长 context→短 context + soft prompt"。Anthropic 在 Constitutional AI: Harmlessness via RLHF 里也用过类似手法把长的 chain-of-thought 蒸馏到模型本身 (https://arxiv.org/abs/2212.08073)。

**关键 intuition**: 当 student 用 (few trajectories + 20 soft tokens) 能 reproduce teacher 用 many trajectories 给出的 action 分布时, 这 20 个 soft token 就成了"压缩后的经验 latent code"。在 inference 时, 哪怕给 student 一个全新的 task, 它也能从这 20 个 token 里 "intuition-like" 地受益, 类似人脑在没显式 recall 细节的情况下凭直觉做决策。

**为什么 length=20 够用?** 这是个有趣问题。作者没详细讨论, 但我的猜测是: 在 narrow domain (DB/OS/KG) 内, 真正需要 internalize 的"domain-specific procedural knowledge"信息量本来就不大, 几百个 trajectory 之间高度冗余, 压到 20×d_model (Llama 8B 的 d_model=4096, 所以 20×4096≈82k float, ~330KB 参数) 完全装得下。这是个 information bottleneck, 反而帮助泛化 (类似 autoencoder bottleneck 强迫 latent 抽 feature)。

### 3.3 Experience-Enhanced Inference

推理时把两层 memory 拼起来:
$$\mathcal{T}_k = \mathcal{P}_\theta \oplus \mathcal{P}_{\mathrm{sys}} \oplus \mathrm{Exp}_c \oplus \mathrm{Exp}_s \oplus \mathcal{C}_s \oplus t_k \tag{8}$$
层次解读 (从最 implicit 到最 explicit):
- $\mathcal{P}_\theta$: parametric, long-term, consolidated, "intuition"。
- $\mathcal{P}_{\mathrm{sys}}$: static domain rule。
- $\mathrm{Exp}_c, \mathrm{Exp}_s$: working memory, contrastive 提取的 symbolic insights。
- $\mathcal{C}_s$: recent successful demonstration, in-context replay。
- $t_k$: 当前 task。

这种 hierarchical memory 设计在 cognitive architecture 文献里很常见, 比如 SOAR 的 procedural memory + working memory (https://soar.sourceforge.net/), ACT-R 的 chunks + production rules。EvoSC 算是把这套思想用 LLM 时代的语言重新实现了。

### 3.4 Algorithm 1 整体流程

算法 1 的伪代码就是上面三块的组合, 关键的几个 step:
1. 对每个 task $t_k$, 先 retrieve top-K recent success dialogs $\mathcal{C}_{\mathrm{succ}}^{\mathrm{rec}}$ from $\mathcal{R}_{\mathrm{succ}}$。
2. 用 (3) (4) 在线计算 $\mathrm{Exp}_c, \mathrm{Exp}_s$。
3. 拼成 augmented prompt $\mathcal{T}_k$, 跑 r round interaction。
4. 若 success: 把 dialog 存进 $\mathcal{R}_{\mathrm{succ}}$, 把 $\mathrm{Exp}_s$ 推进 $\mathcal{Q}_{\mathrm{succ}}$ FIFO; 若 fail: 把 $\mathrm{Exp}_c$ 推进 $\mathcal{Q}_{\mathrm{err}}$ FIFO。

注意 Algorithm 1 里**没显式写 PTC 训练时机**, 但根据上下文推测应该是在后台 periodic trigger (e.g. 每隔若干 task 触发一次 distillation), 不是每个 task 都训。这是合理的, 因为 PTC 训练有 GPU 计算开销, 不能在线阻塞。

---

## 4. Experiments 详细分析

### 4.1 Benchmark: LifelongAgentBench

来源: Zheng et al. 2025b (https://arxiv.org/abs/2505.11942), 三个 domain:
- **Database (DB)**: 500 tasks, SQL-style 交互, max 3 rounds。
- **Operating System (OS)**: 500 tasks, shell 交互, max 5 rounds。
- **Knowledge Graph (KG)**: 396 tasks, API 调用, max 15 rounds (long-horizon)。

KG 的 15 rounds 特别关键——这是 long-horizon 场景, 是 PTC 这种 parametric consolidation 真正能 shine 的地方, 因为 textual replay 在这种场景下 context 爆炸最快。

### 4.2 Main Results Table 1 (DB & OS)

Llama 3.1-8B 在 DB 上:
| Method | Exp=0 | Exp=1 | Exp=4 | Exp=16 | Exp=32 | Avg |
|---|---|---|---|---|---|---|
| AWM | 19.0 | 45.4 | 71.6 | 66.7 | 74.2 | 55.4 |
| TER | 19.8 | 41.6 | 68.2 | 69.0 | 70.2 | 53.8 |
| SCM | 19.8 | 23.4 | 63.0 | 61.0 | 68.4 | 47.1 |
| A-MEM | 19.8 | 57.0 | 67.0 | 74.8 | 73.4 | 58.4 |
| EvoSC | 24.8 | 71.2 | 74.4 | 77.2 | 77.8 | **65.1 (+6.7 vs A-MEM)** |

几个观察:
1. **Exp=0 (cold start)**: EvoSC 24.8 > 所有 baseline 的 19.x, 说明仅靠 contrastive reflection + PTC pretrain 就有增益。这是个 sanity check, 说明 framework 真的让 agent "进化了", 不是单纯吃更多 demo。
2. **Exp=1**: EvoSC 71.2 vs A-MEM 57.0, +14.2 巨大差距, 说明 contrastive 抽取的 insight 比简单存 dialog 高质量得多。
3. **Exp=32**: EvoSC 77.8, 所有 baseline 都掉到 70-74, 说明 textual-only 方法在 history 大时被 noise 拖累, 而 EvoSC 的 PTC 把多余 trajectory 压进 $\mathcal{P}_\theta$, 留出来的 context slot 给最相关的 explicit memory。

OS 上 Llama 3.1-8B:
| Method | Avg |
|---|---|
| AWM | 47.1 |
| A-MEM | 48.4 |
| EvoSC | 50.1 (+1.7) |

OS 增益小一些, 我的解读是: OS 命令空间更离散、state transition 更 deterministic, contrastive reflection 能挖出来的"insight"边际收益没那么高, 而且很多 OS task 一两步就能搞定, multi-round reasoning 价值低。

### 4.3 Table 2 (KG) — EvoSC 优势最明显

Llama 3.1-8B on KG:
| Method | Exp=0 | Exp=1 | Exp=4 | Exp=16 | Avg |
|---|---|---|---|---|---|
| AWM | 12.6 | 26.5 | 32.6 | OOM | 23.9 |
| TER | 28.0 | 35.1 | 32.8 | OOM | 32.0 |
| SCM | 28.0 | 28.0 | 31.1 | OOM | 29.0 |
| A-MEM | 28.0 | 31.8 | 19.9 | OOM | 26.6 |
| EvoSC | 32.1 | 39.4 | 36.7 | **42.7** | **37.7 (+5.7)** |

Qwen 2.5-7B on KG:
| Method | Avg |
|---|---|
| AWM | 15.6 |
| TER | 27.8 |
| SCM | 25.6 |
| A-MEM | 15.1 |
| EvoSC | **38.4 (+10.6 vs TER)** |

KG 上 +10.6 是质的飞跃。原因: KG 任务需要长 reasoning chain (15 rounds), 大量 API 调用顺序约束 (e.g. `get_relations() -> get_neighbors() -> get_attributes() -> count()`), 这种 procedural knowledge 非常适合 parametric consolidation——因为 textual demo 长且冗余, 压成 20 个 soft token 反而捕捉到了"workflow 的抽象结构"。

**OOM 现象**: 注意所有 baseline 在 Exp=16 (KG) 或 Exp=32 (Qwen DB/OS) 时全部 OOM, 因为 Qwen 2.5-7B 的 context window 比 Llama 3.1-8B 紧。EvoSC 唯一能 scale 上去的方法, 这是 PTC 的关键价值——**bypass context window limit via parameterization**。

### 4.4 Ablation Table 3

Llama 3.1-8B on DB:
| EE | SE | PTC | Avg |
|---|---|---|---|
| ✗ | ✓ | ✗ | 61.7 |
| ✓ | ✗ | ✗ | 62.2 |
| ✓ | ✓ | ✗ | 65.2 |
| ✓ | ✓ | ✓ | **65.1** |

注意 **加 PTC 反而 Avg 没变化 (65.2 → 65.1)**! 但看 Exp=32 那列: 77.8 vs 77.8 一样。这看起来反直觉。其实关键在 Table 里 "OOM" 那几格——不加 PTC, 在 Exp=32 时直接 OOM 没法跑, 所以那个"65.2"是用更小的 Exp 数 (0,1,4,16) 算的 avg, 数量上能凑齐但实际不可 scale。加 PTC 才能在 Exp=32 上跑出 77.8。这一行 ablation 真正展示的是**PTC 的可扩展性**而非性能。

EE (error-prone experience) 单独 vs SE (successful experience) 单独: 61.7 vs 62.2, 几乎打平, 说明 failure 信号和 success 信号一样重要, contrastive 思想在 ablation 上得到验证。

### 4.5 Figure 5 — Learning Curve

作者画了 cumulative correct count 在 DB 上的曲线 (window=100, 1 trajectory)。EvoSC 曲线斜率更陡且持续上升, baseline 在中后期趋于平台。这是 lifelong learning 最该看的指标——能不能持续提升而不是 plateau。这条曲线形态很像人脑 learning curve 的 power law, 很符合 intuition。

---

## 5. 与相关工作的对照 (build intuition)

| Concept | EvoSC 对应 | 关系 |
|---|---|---|
| Hippocampus → Neocortex consolidation | $\mathcal{E}_{\mathrm{many}}$ teacher → $\mathcal{P}_\theta$ student | 直接 inspiration |
| Episodic memory | $\mathcal{R}_{\mathrm{succ}}$ 原始 trajectory 库 | 短期、verbatim |
| Semantic memory | $\mathcal{P}_\theta$ soft prompt | 压缩、抽象、long-term |
| Working memory | $\mathrm{Exp}_c, \mathrm{Exp}_s, \mathcal{C}_s$ in-context | 容量受限、易失 |
| Reflexion (Shinn et al. https://arxiv.org/abs/2303.11366) | contrastive reflection | Reflexion 是 task-level self-correction, EvoSC 是 cross-task |
| ExpeL (Zhao et al. https://arxiv.org/abs/2308.10144) | success experience extraction | ExpeL 用 LLM 抽 insight 后写回 prompt, 没做 parametric consolidation |
| AWM (Wang et al. https://arxiv.org/abs/2409.07429) | workflow extraction | AWM 只抽 success workflow, 没 failure 信号, 没参数化 |
| Voyager (Wang et al. https://arxiv.org/abs/2305.16291) | skill library | Voyager 在 Minecraft 里建 code skill library, 是 symbolic memory, 没做 latent consolidation |
| Generative Agents (Park et al. https://arxiv.org/abs/2304.03442) | reflection + memory | Stanford 小镇 agent, 用 reflection 把 memory 压成 insight, 但都是 textual |
| Prefix Tuning (Li & Liang 2021 https://arxiv.org/abs/2101.00190) | $\mathcal{P}_\theta$ 形式 | EvoSC 借用了 prefix tuning 的训练形式 |
| Context Distillation (Anthropic https://arxiv.org/abs/2112.08895) | PTC 训练目标 | 直接同源思想, Anthropic 把 long prompt 蒸馏进 weights, EvoSC 蒸馏进 soft prompt |
| DPO (Rafailov et al. https://arxiv.org/abs/2305.18290) | contrastive reflection | 都用 pref/dispref 对比, 但 DPO 训 weights, EvoSC 训 prompt + textual insight |

---

## 6. 我对这篇 paper 的 critique 与扩展思考

### 6.1 强点
1. **问题定义清晰**: context window 爆炸是 lifelong agent 的真实瓶颈, PTC 直击痛点。
2. **Hybrid memory 架构合理**: 不强行 all-parametric (那会 catastrophic forget) 也不 all-textual (那会 OOM), 两层并存有工程美感。
3. **Contrastive reflection 的抽象**: 用 LLM 自己做 "diff" 提取 insight, 比 reward model 训练成本低得多, 而且产物可读 (symbolic insight) 可解释。
4. **Model-agnostic**: 不动 LLM 主干, plug-and-play, 适配任何 frozen LLM。

### 6.2 弱点 / 可改进
1. **Retrieval 太朴素**: FIFO + top-K recent, 完全没 semantic retrieval。如果换 embedding similarity + recency prior (类似 RAG + EMA), 应该能再涨。Generative Agents 那套 recency-importance-relevance 三个 score 的检索 (https://arxiv.org/abs/2304.03442) 完全可以挪过来。
2. **PTC 训练成本未量化**: 每个 round 的 student forward + teacher forward + backward, GPU cost 多少? 论文没列。Exp=32 设置下 teacher 用 20 traj, student 用 8 traj, 但训练数据规模没说。这是个 reproducibility 短板。
3. **没在 70B+ 上验证**: 作者在 limitations 里承认, 但 PTC 是否随 scale 保持增益是 open question。直觉上 model 越大 in-context 越强, soft prompt 边际收益可能下降 (类似 scaling law 下 prompt tuning gain 减少)。
4. **PTC 的 catastrophic forgetting**: 一段 length=20 的 soft prompt 要 encode 全部历史 trajectory 的 distilled knowledge, 当 task domain shift 时怎么处理? 论文里 domain 是固定的 (DB/OS/KG 各自独立), 没测 cross-domain。如果做 lifelong cross-domain, $\mathcal{P}_\theta$ 会被新 domain 覆盖。
5. **没对比 ExpeL / Reflexion / Voyager 这条 strong baseline**: 只跟 AWM/TER/SCM/A-MEM 比, 后者都是 memory-heavy 方法, 缺少 reflection-heavy 的对比。
6. **Length=20 的 ablation 缺失**: 为什么是 20 不是 50 不是 100? length 对 capacity 的影响没扫, 这是个 hyperparam sensitivity 缺口。

### 6.3 可以延展的研究方向

**A. Hierarchical / Mixture-of-Prompts**:
不同 task category 学不同 $\mathcal{P}_\theta^{(c)}$, 推理时用 router 选。类似 Mixture-of-Experts 但 expert 是 soft prompt。这能解决 cross-domain forgetting 问题, 同时增加 capacity。

**B. Online PTC update**:
现在 PTC 是 periodic batch 训, 可以做成 **continual PTC**: 每个 task 完成后用 REINFORCE-style gradient 微调 $\mathcal{P}_\theta$。但这需要可微分 reward 或 surrogate loss, 实现复杂度高。

**C. Metacognitive gating**:
agent 自己评估哪些 trajectory 值得 consolidate, 不值得的丢掉。类似人类 sleep 里 memory triage 机制 (only consolidate emotionally salient or surprising experiences)。

**D. PTC + LoRA hybrid**:
把 $\mathcal{P}_\theta$ 从 input embedding 扩展到 attention layers 的 LoRA, capacity 大幅提升。代价是参数量上去了, 但仍比 full FT 省。

**E. Distillation target 升级**:
现在 teacher target 是 single best action token sequence。可以改成 **teacher's full distribution** (软标签 distillation, Hinton 2015 https://arxiv.org/abs/1503.02531), 把 teacher uncertainty 也传过去, 可能改善 calibration。

**F. 跟 RLHF / DPO 整合**:
contrastive reflection 出来的 textual insight 可以转化为 preference pair (success vs failure trajectory), 直接 DPO 训 LLM weights (而不是 soft prompt)。这样 parametric memory 不局限于 20 token, 而是真正的 weight update。但代价是 catastrophic forgetting 风险上升, 需要加 EWC / LoRA 之类的 regularization。

**G. Test-time compute scaling**:
现在 inference 是单次 forward, 可以加 test-time search (MCTS, beam search), 让 $\mathcal{P}_\theta$ 在搜索树展开时给 prior。这就跟 Sutton 的 "learning to search" 思路接上了 (AlphaGo 风格)。

**H. 与 continual pretraining 对比**:
真正的 lifelong learning 应该是 weights 一直 update, 而不是 frozen LLM + soft prompt。如果对比 full model continual SFT (带 replay buffer 防 forget), EvoSC 的相对位置在哪? 这个 baseline 缺失让我略遗憾。

---

## 7. 关键 web links 汇总

- Paper 本身 (按文件名推测可能尚未上 arXiv, 这是 markdown 草稿): 可以搜 "Self-Consolidation for Self-Evolving Agents" + 第一作者 Hongzhuo Yu (UCAS-Terminus AI Lab)
- LifelongAgentBench: https://arxiv.org/abs/2505.11942
- AWM (Agent Workflow Memory): https://arxiv.org/abs/2409.07429
- A-Mem (Agentic Memory): https://arxiv.org/abs/2502.12110
- Reflexion: https://arxiv.org/abs/2303.11366
- ExpeL: https://arxiv.org/abs/2308.10144
- Voyager: https://arxiv.org/abs/2305.16291
- Generative Agents (Stanford 小镇): https://arxiv.org/abs/2304.03442
- Prefix Tuning: https://arxiv.org/abs/2101.00190
- Prompt Tuning (Lester et al.): https://arxiv.org/abs/2104.08691
- Context Distillation (Anthropic, "A General Language Assistant"): https://arxiv.org/abs/2104.08773
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Knowledge Distillation (Hinton): https://arxiv.org/abs/1503.02531
- DPO: https://arxiv.org/abs/2305.18290
- Memory consolidation in neuroscience (Spens & Burgess): https://www.nature.com/articles/s41562-024-01838-7
- SOAR cognitive architecture: https://soar.sourceforge.net/
- ACT-R: http://act-r.psy.cmu.edu/
- Llama 3 report: https://arxiv.org/abs/2407.21787
- Qwen 2.5 report: https://arxiv.org/abs/2412.15115

---

## 8. 一句话总结给 Karpathy

EvoSC 把 cognitive science 里的 hippocampus-to-neocortex consolidation 用 LLM 时代的语言重新表达: **contrastive reflection 抽 symbolic insight (working memory), context distillation 把 trajectory 蒸成 length=20 的 soft prompt (semantic memory), inference 时 hierarchical 拼起来**。工程上 pragmatic, 思想上和 Anthropic 的 context distillation 同源, 实验里在 long-horizon KG 任务上 +10.6 是个有说服力的数字。最大的 open question 是 scale 到 70B+ 之后 soft prompt 的 marginal gain 是否还在, 以及怎么把 retrieval 从 FIFO 升级到 semantic, 怎么处理 cross-domain consolidation 的 interference。
