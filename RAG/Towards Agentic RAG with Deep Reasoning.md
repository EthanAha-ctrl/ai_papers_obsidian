---
source_pdf: Towards Agentic RAG with Deep Reasoning.pdf
paper_sha256: a4eeea0d4964303774b8416b0006ed73d7c0424416d564b5b063c58a9189ef8d
processed_at: '2026-08-12T16:58:23-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 — 这篇 survey 到底在说什么

---

## 一句话总结

这篇 paper 在讲一件事: **让 LLM 学会像人类研究员一样工作 — 边查资料边思考,查到新东西又激发新问题,一直循环到想明白为止**。

---

## 故事背景 — LLM 的两个老毛病

LLM 有两个一直没根治的毛病:

**毛病一: 记不住事儿**。LLM 的知识是训练时灌进去的,训练完就冻结了。你问它 2024 年 12 月的某个 news,它不知道;你问它某个 long-tail 的小众 fact,它也会瞎编。这叫 hallucination。

**毛病二: 不会深想**。复杂问题需要一步一步推理,但 LLM 一次 forward pass 就吐答案,中间步骤全靠 implicit 的 hidden state,容易在多步推理时崩掉。

以前大家开了两个药方:
- **RAG** 治毛病一 — 给 LLM 外接一个数据库,回答前先查一下
- **CoT / RL reasoning** 治毛病二 — 让 LLM 显式写出中间推理步骤,或者用 RL 训它深度思考

---

## 关键 Insight — 这两个毛病其实是一个毛病

这篇 survey 的核心 observation 是: **这俩毛病是纠缠在一起的**。

举个例子,你问 LLM "收购 WhatsApp 的那家公司的创始人是谁?"。

- 如果纯靠 parametric memory,LLM 可能记得 Facebook 收购了 WhatsApp,但不确定 Zuckerberg 是不是创始人,于是开始 hallucinate。
- 如果先 RAG 一下,搜到 "Facebook acquired WhatsApp in 2014",但 LLM reasoning 不够深,可能直接答 "Mark Zuckerberg" 而不验证,或者被 retrieved context 里的 noise 带偏。

反过来,如果 LLM reasoning 能力强,能拆成两个 sub-question ("谁收购了 WhatsApp?" → "那个人的创始人是谁?"),但每一步都需要外部知识补充,否则 reasoning chain 在第一步就断了。

所以 paper 的 thesis 是: **reasoning 产生 information need,retrieval 满足 need,新 evidence 又驱动更深的 reasoning**。这是同一个 inference loop 的两面,不能拆开治。

---

## 传统 RAG 为什么不够用 — "先查后答" 的三个坑

传统 RAG 是 "Retrieval-Then-Reasoning" (RTR) — 先搜一次,把结果塞进 prompt,再让 LLM 答。这有三个坑:

**坑一: 搜不准**。你在 reasoning 开始前根本不知道自己需要什么知识。比如 "收购 WhatsApp 的公司的创始人",如果直接搜整句话,搜到的可能是关于 WhatsApp 的一堆 news,而不是 Facebook 的创始信息。真正需要的 knowledge gap 要在推理到 "Facebook 收购了 WhatsApp" 这一步之后才浮现。

**坑二: 搜回来的东西反而干扰**。retrieved context 里如果有错误信息或 conflict,LLM 会被带偏。实验数据表明 misleading context 能让 accuracy 掉 15-20%。这就像你写论文查文献,查到一篇错的 paper 然后整个 reasoning 都跑偏了。

**坑三: 没法动态调整**。推理到一半发现证据不够,传统 RAG 没法 "再搜一次"。这就像侦探破案,线索查到一半发现要补查,但系统不允许补查。

---

## 这篇 survey 的三层结构

paper 把整个领域分成三层,像洋葱一样:

### 第一层: Reasoning-Enhanced RAG — 用推理优化检索

思路: 既然检索那么重要,那就让 reasoning 能力渗透到 RAG 的每个 stage。

**检索前 — Query 改写**:
- 把复杂问题拆成小问题: "收购 WhatsApp 的公司的创始人" → ["谁收购了 WhatsApp?", "那个人是谁创立的?"]
- 用 CoT 先想一下再改写 query,让 query 更精准
- 用 RL 训练 query rewriter,直接用 retrieval recall 和 answer correctness 当 reward

**检索决策 — 要不要搜**:
- 用一个 classifier 先判断问题难度。简单问题直接答,中等问题搜一次,复杂问题搜多次。这能节省大量 API call。
- 高级版本用 LLM 先做一个 full retrieval plan,规划好整个搜索路径

**检索器本身 — 让 retriever 有 reasoning 能力**:
- GNN-RAG 用 graph neural network 在 knowledge graph 上做 implicit multi-hop reasoning,然后把聚合的 representation 喂给 LLM
- RuleRAG 在检索时加入 symbolic rule,保证逻辑一致性

**检索后 — Integration**:
- 搜回来一堆 passage,先用 NLI model 过滤掉不 entail 的
- SEER 用多个 assessor expert 从 faithful/helpful/concise 三维度评分,prune 低分
- Beam-AggR 把 sub-question 的 answer 做概率聚合: $P(a|q) = \sum_{\{a_i\}} P(a|\{a_i\}) \prod_i P(a_i|q_i)$,其中 $a_i$ 是第 $i$ 个 sub-question 的答案

**生成时 — Grounded generation**:
- Self-RAG 训练 LLM 在 decoding 时 emit reflection token,比如 `[Retrieve]`、`[Relevant]`、`[No Support]`,触发 critical review。这就像让 LLM 自言自语 "这里我需要查一下" "这个证据不靠谱,要修正"
- RARR 先 generate,再 retrieve contradiction,再 revise,同时插入 citation

### 第二层: RAG-Enhanced Reasoning — 用知识支撑推理

反过来,reasoning 过程也需要外部知识。

**外部知识库**:
- 数学推理: 从 Lean/Coq theorem library 里 retrieve lemma,形式化证明时 $lemma \vdash goal$ 必须靠外部知识
- 法律推理: retrieve judicial precedent 做类比推理 (analogical reasoning)
- 代码生成: 从 GitHub repository retrieve API signature 和 snippet

**Web search**:
- Fact-checking: 多步验证,每步从 news/social media 取证据
- Agentic QA: MindSearch 模仿人类搜索行为,先 decompose query,再 parallel 读 snippet,再 synthesize
- 医疗: 从文献库 retrieve diagnosis 依据

**Tool using**:
- ToolkenGPT 把 tool 当成 vocabulary token 一起 decode: $P(\text{tool}_i | h) = \text{softmax}(W_{tool} \cdot h)$,其中 $h$ 是 hidden state, $W_{tool}$ 是 tool embedding matrix
- 这就像让 LLM 把 "调用计算器" 当成一个普通 word 来预测

**In-context retrieval (不查外部,查自己记忆)**:
- RA-DT 是 Decision Transformer + retrieval,把 past trajectory 存成 memory,retrieve 来 guide 未来决策: $\pi(a|s, \text{retrieved trajectory}) = \text{DT}(s \oplus \text{traj}_{retrieved})$
- EM-LLM 用 surprise metric 做 episodic memory segmentation,把 infinite context 压成 episode chunks
- ICL example retrieval: UPRISE 训 universal prompt retriever,从 training data 里挑最匹配的 demonstration

### 第三层: Synergized RAG-Reasoning — 双向驱动 (核心!)

这是 paper 的 main course。前面两层都是 one-way,这一层是 bidirectional iterative loop。

paper 分两个 axis: **Reasoning Workflow** (推理的结构) 和 **Agent Orchestration** (agent 怎么协作)。

#### Reasoning Workflow — 推理长什么样

**(a) Chain-based — 一条线走到底**

代表是 IRCoT 和 RAT。形式化:

```
r_0 = ""
for t = 1 to T:
    d_t = Retrieve(q | r_{t-1})   # 根据当前推理状态搜
    r_t = LLM(r_{t-1} ⊕ d_t ⊕ q)  # 基于新证据继续推
```

关键: 检索 query 随着推理演化,每一步的 query 都基于前一步的 reasoning state。

改进版:
- CoV-RAG 在每步加 verification: $v_t = \text{Verify}(r_t, d_t)$,如果 fail 就 re-retrieve 或 revise
- Chain-of-Note 让 LLM 对每个 retrieved doc 写 "reading note",filter unhelpful 信息,final answer 基于_notes_ 而非 raw docs
- RAFT 训练时主动加入 distractor docs,让 LLM 学会 ignore noise

Chain-based 的好处是 low latency、易 cache,坏处是 error propagation — 第一步走错,后面全错。

**(b) Tree-based — 分叉探索**

ToT (Tree-of-Thought) 把 CoT 扩展成 tree,每个 node 是一个 thought,可以 branch 多条路同时探索。用 BFS/DFS + self-evaluation $\text{Eval}(\text{thought}) \in [0,1]$ 决定 expand 哪个 node。

MCTS (Monte Carlo Tree Search) 更聪明,用 UCB1 公式做 selection:
$$\text{UCB}(s, a) = Q(s, a) + c \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

变量解释:
- $s$ = 当前 tree node (一个 reasoning state)
- $a$ = action (一个 retrieval 或 reasoning step)
- $Q(s, a)$ = 这个 action 的 average value (来自 self-evaluation)
- $N(s)$ = parent node 的 visit count
- $N(s, a)$ = 从 $s$ 走 $a$ 到 child 的 visit count
- $c$ = exploration constant (通常 $\sqrt{2}$)

第一项 $Q(s,a)$ 是 exploitation (选价值高的),第二项是 exploration (选没怎么探索过的)。MCTS 的好处是 budget-aware — API call 不够时可以集中探索 promising branch,实现 anytime stopping。代表: AirRAG、MCTS-RAG、SeRTS。

Tree-based 的好处是 high recall、transparent,坏处是 quadratic cost (很多 branch 都要 retrieval call)。

**(c) Graph-based — 在图上走**

两个分支:

**Walk-on-Graph**: 用 GNN 在 knowledge graph 上做 message passing,$h_v^{(l+1)} = \text{AGG}(\{h_u^{(l)} : u \in \mathcal{N}(v)\})$,其中 $h_v^{(l)}$ 是 node $v$ 在第 $l$ 层的 hidden state, $\mathcal{N}(v)$ 是邻居。代表: PullNet、QA-GNN、GreaseLM、LightRAG、StructRAG。

**Think-on-Graph**: LLM 直接 drive graph traversal。ToG 让 LLM 把 KG 当 "reasoning playground",每步决定 explore 哪个 entity/relation,逐渐 build 出一条 path 到 answer。Graph-CoT 是三阶段 loop (reasoning → graph interaction → execution)。GraphReader coupling LLM reasoning with explicit subgraph retrieval + evidence anchoring。

Graph-based 的好处是 verifiable citation (每步都能 anchor 到 KG node),坏处是 high latency (很多 micro-tool call) 和 search space explosion without pruning。

#### Agent Orchestration — 谁来干活

**(a) Single-Agent — 一个人全包**

**Prompt-based**:
- ReAct 是经典: Thought-Action-Observation 三元组循环
  ```
  Thought: "我需要知道谁收购了 WhatsApp"
  Action: Search["Who acquired WhatsApp"]
  Observation: "Facebook acquired WhatsApp in 2014"
  Thought: "现在需要知道 Facebook 的创始人"
  Action: Search["Facebook founder"]
  Observation: "Mark Zuckerberg founded Facebook"
  Thought: "答案找到了"
  Action: Finish["Mark Zuckerberg"]
  ```
- Self-Ask、IR-CoT 用 recursive sub-question formulation,不是显式 action command
- DeepRAG、Self-RAG 加 self-reflection,让 LLM 自己判断何时需要 retrieve

**SFT-based**:
- Toolformer self-supervised,LLM 自己 decide 何时 call API,在 synthetic data 上 fine-tune
- INTERS 用 20 个 task、43 个 dataset 的 instruction tuning 数据,manually written template

**RL-based — 这是最近的 frontier**:

Search-R1 训 LLM 在 reasoning 中 emit `<search>` token 触发检索。Reward = exact match + format reward:
$$R = R_{em} + \beta \cdot R_{format}$$
其中 $R_{em} = \mathbb{1}[\hat{a} = a^*]$ 是 exact match indicator, $R_{format}$ 检查 `<search>...</search>` tag 是否 well-formed, $\beta$ 是权重。

用 GRPO 优化 (DeepSeek-R1 推广的 RL algorithm):
$$\mathcal{L}_{GRPO}(\theta) = -\mathbb{E}_{q, \{o_i\} \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \hat{A}_i \right]$$

变量解释:
- $q$ = question
- $o_i$ = 第 $i$ 个 sampled output (group of G 个)
- $\pi_\theta$ = current policy
- $\pi_{\theta_{old}}$ = old policy (importance sampling)
- $\hat{A}_i$ = advantage,在 group 内归一化 (不需要 critic network!)

GRPO 比 PPO 适合 long-context agent scenario,因为不需要训 critic,计算更省。

**DeepResearcher** 是 first end-to-end RL-trained research agent 直接和 open web 交互。Reward 只有 final answer correctness,但 agent 学到了 emergent capability: decomposition、iterative verification、retrieval planning — 这些 SFT 难以 instill。

**ZeroSearch** 更有趣: 用 LLM 自己 simulate search engine,不真调 API,降低训练 cost。这对应 "internalized retrieval" 方向 — 模型把搜索行为压进 parametric memory。

**(b) Multi-Agent — 分工合作**

**Decentralized** (去中心化): 多个 agent 各自从不同 source retrieve,然后聚合。代表: Collab-RAG (white-box + black-box LLM 合作)、MDocAgent (文本 agent + 图像 agent 处理多模态 document QA)。

**Centralized/Hierarchical** (中心化层级): manager-worker 架构。HM-RAG 用 decomposer-retriever-decider 三层。好处是 budget-efficient,manager 避免 duplicate search,保证 provenance chain 清晰。坏处是 manager 可能成为 bottleneck。

---

## Deep Research 产品对比 — Appendix B 的干货

Table 5 列出当前 SOTA deep research system,我提炼几个 trend:

**Optimization 分布**:
- **Prompting-based**: gpt-researcher、deep-searcher、nanoDeepResearch、DeerFlow、Open Deep Search — 这些用 prompt engineering + ReAct loop 驱动 commercial LLM (GPT/Claude/Gemini/Deepseek),工程化实现
- **DPO-based**: Webthinker 用 preference pair 训 Deepseek-R1、QwQ-32B
- **GRPO-based**: Search-R1、ZeroSearch、DeepResearcher、R1-Searcher、ReSearch — 这是 research 主流,因为 GRPO 不需要 critic,适合 agent scenario

**Agent Architecture**: 大多数是 single-agent (context sharing 容易),少数 decentralized (DeerFlow) 或 centralized (Agentic Reasoning、gpt-researcher)。

**Retriever**: 绝大多数用 web search (需要实时 open-domain 信息),少数支持 local retrieval。

**Evaluation**: 经典 QA (NQ、TriviaQA、HotpotQA、MuSiQue) + 高难度 (GPQA、GAIA、HLE、BrowseComp)。Webthinker 在 GPQA+GAIA+WebWalkerQA+HLE 上 eval,是目前覆盖最难 benchmark 的 system 之一。

---

## Benchmark — 评估还差什么

paper 列了 46 个 benchmark 覆盖 13 个 task 类别。我重点说三个 gap:

**Gap 一: Domain 太窄**。大部分是 general/academic,缺 industrial、vertical-domain、personalized 场景。

**Gap 二: Reasoning 类型不平衡**。当前主要 test deductive reasoning,counterfactual、causal、analogical reasoning 严重 underexplored。

**Gap 三: Trajectory evaluation 缺失**。没有 standardized metric 评估整个 reasoning-retrieval trajectory — 比如 retrieval step 效率、intermediate query 质量、multi-step chain 一致性。现在只看 final answer 对不对,但中间过程是黑盒。

**Gap 四: Trust 维度缺失**。没有 benchmark 系统评估 robustness against noisy/evolving/conflicting 信息,也没有 multi-dimensional trust metric。

---

## 我的几点 critical reflection

**(1) 范式转移的本质**: 从 "先查后答" 到 "边查边想边答"。这本质上是把人类研究员的工作方式编码进了 LLM 的 inference loop。你做 literature review 不会先把所有 paper 读完再开始写,而是边读边想边查边写,reasoning 和 retrieval 交织。这篇 survey 把这个 intuition 形式化了。

**(2) RL-RAG 是新 frontier**: Search-R1、R1-Searcher、DeepResearcher 这些 RL-trained system 在 multi-hop QA 上展现的 emergent capability 是 SFT 难以复现的。这呼应 DeepSeek-R1 的 thesis: RL can teach reasoning without SFT on CoT data。在 RAG 场景,RL 让模型自己摸索出 "什么时候搜、搜什么、怎么整合" 的策略,而不是人工设计。

**(3) Benchmark 落后于 method**: HLE、BrowseComp 是 2024 底/2025 初才出,evaluation 严重滞后。特别是 trajectory-level 和 trustworthiness 的评估是 open gap。这个 gap 对 industry 落地很 critical — 你不能只看 final answer 对不对,还得看 reasoning 过程是否 faithful、citation 是否 accurate、retrieval 是否 efficient。

**(4) Multimodal 是 obvious next step**: 目前绝大多数 system 仍 text-only。MDocAgent、SurgRAW 开始探索 multimodal multi-agent RAG,但 unified multimodal retriever (image + table + text + heterogeneous doc) 还不成熟。这个方向和你 eureka labs 教学中提到的 "multimodal understanding is the future" 高度 align。

**(5) 理论 foundation 缺失**: 为什么 iterative retrieval-reasoning 比 single-pass 更有效?信息论角度 (mutual information gain at each step)、计算复杂度分析、最优 stopping time 的理论都还很少。这是 academic 上的 fertile ground。

**(6) Human-in-the-loop 方向被低估**: Section 7 提到 human-agent collaboration,这个方向其实非常重要。很多 open-ended task (literature review、interactive programming、education) 天然需要 human feedback 来 steer reasoning。PersonaAgent 让 agent adapt reasoning strategy based on user expertise,这和 eureka labs "education as killer app of LLMs" 的愿景 align — 教育场景需要 agent 理解学生状态、adaptive 调整推理深度。

---

## 用一个比喻收尾

想象 LLM 是个实习生。传统 RAG 是: 你给他一堆资料,让他读完写报告。问题是他可能读不完、读不懂、或者读了错的资料。

Synergized RAG-Reasoning 是: 你给他一个研究问题,让他自己决定查什么、怎么查、查到之后怎么想、想不明白再查什么。这就是 OpenAI Deep Research、Perplexity Deep Research 在做的事 — 把人类研究员的工作方式 encode 进 LLM 的 inference loop。

RL-RAG 进一步: 不是你教他怎么查,而是让他自己摸索出高效的研究策略,通过 reward signal (final answer 对不对) 来 learn。这就像培养一个真正的研究员,而不是一个执行固定流程的 worker。

这就是这篇 survey 画的 trajectory: **从 "检索工具" 到 "研究伙伴"**。

---

# Towards Agentic RAG with Deep Reasoning — 深度技术解读

Hey Andrej, 这篇由 Tsinghua、UIC、UTokyo、PKU 等机构联合出的 survey 把过去两年 RAG 和 LLM reasoning 两条线 finally weave 在一起,核心 narrative 非常清晰: **从 one-way enhancement 走向 synergized iterative loop**,这其实就是你 eureka labs 教学里反复强调的 "let the model think while it retrieves" 的范式。下面我尽量 build your intuition,把每个模块的 mechanism、failure mode、以及和 SOTA system(DeepSeek-R1, OpenAI Deep Research, Gemini Deep Research)的对应关系都铺开。

---

## 1. 为什么需要这篇 survey — The Twin Bottlenecks

LLM 有两个 intrinsic limitation:

**(a) Knowledge hallucination** — parametric knowledge 是 static 的, training cutoff 后无法更新;并且 parametric memory 在 long-tail fact 上 recall 极差 (SimpleQA、BrowseComp 这类 benchmark 上 GPT-4o 也只有 ~40% accuracy)。

**(b) Complex reasoning struggle** — 单次 forward pass 难以支撑 multistep deductive/abductive chain,尤其当中间步骤需要 explicit symbolic manipulation。

传统解决方案是两条平行线:
- **RAG** 解决 (a),通过 external corpus injection
- **CoT / RL-reasoning** 解决 (b),通过 explicit intermediate tokens

但 paper 在 Section 1 指出关键的 insight: **这两个 limitation 是 intertwined 的**。missing knowledge 会阻塞 reasoning chain (你 reason 到一半发现 parametric memory 没有 fact 支撑),而 flawed reasoning 又会让 retrieved knowledge 无法被正确 utilize (retrieved context 里 relevant passage 被 LLM 忽略)。这就是 paper 想要打通的 fundamental motivation。

Paper 归纳了传统 RTR (Retrieval-Then-Reasoning) 的三个 failure modes,我觉得这个归纳非常 tight,值得记住:

1. **Retrieval Adequacy & Accuracy**: pre-retrieved knowledge 和 reasoning 过程中真正需要的 knowledge gap 不 align,尤其在 multi-hop QA 上
2. **Reasoning Depth**: retrieved knowledge 里有 noise/conflict 时反而会 pollute LLM 的 inherent reasoning capability (Li et al. 2025b; Chen et al. 2025a 的实验表明 retrieved misleading context 会让 accuracy 下降 ~15-20%)
3. **System Adaptivity**: 没有 feedback loop,无法在 reasoning 过程中 dynamic re-retrieve

---

## 2. Background — RAG 三阶段 + CoT 形式化

### 2.1 RAG Pipeline 的三个 stage

```
Query q → [Retrieval] → {d_1, ..., d_k} 
       → [Integration] → C (curated context) 
       → [Generation] → Answer a
```

形式化:
- Retrieval: $\mathcal{R}(q; \mathcal{D}) = \{d_i\}_{i=1}^k$ where $\mathcal{D}$ is the corpus, typically using bi-encoder score $\text{sim}(E_q, E_{d_i})$
- Integration: $\mathcal{I}(\{d_i\}, q) = C$,涉及 deduplication, conflict resolution, re-ranking
- Generation: $\mathcal{G}(q, C) = a$,LLM 在 context $C \oplus q$ 上做 next-token prediction

### 2.2 CoT 的本质

CoT (Wei et al. 2022) 把 generation 分解成 $a = (r_1, r_2, ..., r_T, a_{final})$,其中 $r_t$ 是 intermediate reasoning token。从 inference-time compute scaling 角度看,这相当于把 single-pass 的 computation budget 扩展成 T-step,允许 model 在每一步 "consume" 之前的 reasoning state。

Key insight: **RAG-Reasoning 的 synergized 版本相当于把 retrieval operation 插入到 CoT 的中间步骤**,即:

$$a = (r_1, \text{retrieve}(q_1), r_2, \text{retrieve}(q_2), ..., r_T, a_{final})$$

其中 $q_t$ 是从 $r_1, ..., r_t$ 动态生成的 query。

---

## 3. Reasoning-Enhanced RAG — 用 reasoning 优化 RAG 各 stage

这一节对应 Figure 1 左侧的 one-way arrow (Reasoning → RAG),核心是让 reasoning capability 渗透到 RAG 的 retrieval/integration/generation 三个 stage。

### 3.1 Retrieval Optimization

#### 3.1.1 Query Reformulation

三个 sub-strategy:

**(a) Query Decomposition**: 把 complex query $q$ 拆成 $\{q^{(1)}, ..., q^{(n)}\}$,例如 "Who founded the company that acquired WhatsApp?" → ["Who acquired WhatsApp?", "Who founded {answer_1}?"]。对应 Press et al. Self-Ask。

**(b) Query Reformulation**: 用 RL signal 训练 rewriter,典型工作 Wang et al. 2025c (MaFeRW) 用 multi-aspect feedback:
$$\mathcal{L}_{rewriter} = -\mathbb{E}_{q \sim \mathcal{D}}[\sum_i \alpha_i \cdot f_i(q, q')]$$
其中 $f_i$ 是 retrieval recall、answer correctness 等 aspect, $\alpha_i$ 是权重。

**(c) Query Expansion via CoT**: Dhuliawala et al. 2024 (CoVe, Chain-of-Verification) 让 LLM 先 generate CoT 然后从 CoT 中 extract expansion terms。Lee et al. 2024 (RadCoT) 用 distillation 方式把 expansion 能力压到 small model。

#### 3.1.2 Retrieval Strategy & Planning

- **Advance Planning**: PAR-RAG (Zhang et al. 2025d) 在 retrieval 之前用 CoT 生成一个 full retrieval blueprint,避免 greedy local optima。LPKG (Wang et al. 2024b) 用 KG-augmented LLM 编码 relational structure 来 plan。
- **Adaptive Retrieval Decision**: FIND (Jia et al. 2025) 和 Adaptive-RAG (Jeong et al. 2024) 用 classifier 预测 query complexity $\in \{easy, medium, hard\}$,分别决定 single-retrieval / multi-retrieval / no-retrieval。Marina et al. 2025 加入 entity popularity、question type 等 feature。

#### 3.1.3 Retrieval Model Enhancement

- **Structured Knowledge**: GNN-RAG (Mavromatis & Karypis 2024) 用 GNN 在 KG 上做 implicit multi-hop reasoning 然后把 aggregated representation 喂给 LLM。形式化: $h_v^{(l+1)} = \text{AGG}(\{h_u^{(l)} : u \in \mathcal{N}(v)\})$,其中 $h_v^{(l)}$ 是 node $v$ 在第 $l$ 层的 hidden state, $\mathcal{N}(v)$ 是邻居。RuleRAG (Chen et al. 2024c) 在 retrieval 时 append symbolic rules $r \in \mathcal{R}$ 保证 logical consistency。
- **Explicit Reasoning**: Ji et al. 2024 把 CoT 和 query concat 来 improve multi-hop recall。

### 3.2 Integration Enhancement

这一 stage 处理 retrieved evidence 的 filtering 和 fusion。

#### 3.2.1 Relevance Assessment & Filtering

- **SEER** (Zhao et al. 2024c): 用 assessor experts 评分 evidence 的 faithful/helpful/concise 三维度,符合 $\text{score}(d_i, q) = w_1 \cdot \text{faithful}(d_i) + w_2 \cdot \text{helpful}(d_i, q) + w_3 \cdot \text{concise}(d_i)$,然后 prune 低分项。
- **Yoran et al. 2024**: 用 NLI model 判断 passage 是否 entail query,过滤 non-entailing passages;再 fine-tune LLM 在 mixed relevant/irrelevant context 上,让 model 学会 ignore residual noise。这个 idea 和 RAFT (Zhang et al. 2024a) 思路相似 — 后者主动在 training data 中加入 distractor documents,训练 LLM 在 noisy context 中保持 faithful。

#### 3.2.2 Information Synthesis & Fusion

- **Beam-AggR** (Chu et al. 2024): enumerate sub-question answer combinations 然后 probabilistic aggregate:
$$P(a | q) = \sum_{\{a_i\}} P(a | \{a_i\}) \prod_i P(a_i | q_i)$$
其中 $q_i$ 是 sub-question, $a_i$ 是对应的 answer。Beam search 在这个联合分布上做。

- **DualRAG** (Cheng et al. 2025): dual-process framework,System 1 做 fast retrieval-augmented querying,System 2 做 progressive knowledge aggregation,生成 evolving outline。

- **CRP-RAG** (Xu et al. 2024): 构建一个 reasoning graph $\mathcal{G}_R = (V, E)$,每个 node $v \in V$ 是 reasoning step,边 $e = (u, v)$ 表示 step $u$ 推出 step $v$。在每个 node 做 retrieve/evaluate/aggregate,然后用 knowledge-sufficiency check 决定 path 是否终止。

### 3.3 Generation Enhancement

#### 3.3.1 Context-Aware Synthesis

- **Open-RAG** (Islam et al. 2024): MoE 架构,sparse experts 动态选择 knowledge module。形式化:$\text{output} = \sum_{i \in \text{TopK}} g_i \cdot \text{Expert}_i(x)$,其中 $g_i$ 是 gating score。
- **RARE** (Wang et al. 2025d): 在 prompt 中加入 domain knowledge 来 incentivize model 依赖 external context 而非 memorization。
- **Self-Reasoning** (Xia et al. 2025b): 三阶段 — evidence selection、evidence verification、inference synthesis,构建 structured reasoning chain。

#### 3.3.2 Grounded Generation Control

- **Self-RAG** (Asai et al. 2023/2024): 训练 LLM 在 decoding 时 emit reflection markers `[Retrieve]`, `[Relevant]`, `[No Support]`, `[Partially Support]`,这些 markers 触发 critical review 和 correction。这个 work 是后续 RL-RAG 的奠基。
- **RARR** (Gao et al. 2023a): post-hoc revision,先 generate 再 retrieve contradiction 然后 revise,同时插入 citation 保持 stylistic coherence。
- **AlignRAG** (Wei et al. 2025b): criticism alignment,训 reward model 评估 reasoning path 的 quality,然后 RL fine-tune。
- **TRACE** (Fang et al. 2024): 构建 KG 来 form coherent evidence chain,每条 reasoning step 都 anchor 到 KG node。

---

## 4. RAG-Enhanced Reasoning — 用 retrieved knowledge 支撑 reasoning

对应 Figure 1 右侧 one-way arrow (RAG → Reasoning)。

### 4.1 External Knowledge Retrieval

#### 4.1.1 Knowledge Base

- **General QA Reasoning**: AlignRAG、MultiHop-RAG (Tang & Yang 2024)、CRP-RAG 从 KB retrieve interconnected factual entries。
- **Mathematical Reasoning**: Premise-Retrieval (Tao et al. 2025)、ReaRAG (Lee et al. 2025) 从 theorem library retrieve formal lemmas,例如在 Lean/Coq 形式化证明场景下, retrieve $lemma$ such that $lemma \vdash goal$。这对 miniF2F 这种 olympiad-level benchmark 很关键。
- **Legal Reasoning**: CASEGPT (Yang 2024)、CBR-RAG (Wiratunga et al. 2024) retrieve judicial precedent 做案例推理 (analogical reasoning)。
- **Code Generation**: CodeRAG (Li et al. 2025a) 从 code repository retrieve API snippet 和 signature,Koziolek et al. 2024 retrieve control code pattern。

#### 4.1.2 Web Retrieval

这是最近 agentic search 的核心。

- **Fact-checking**: VeraCT Scan (Niu et al. 2024)、Ragar (Khaliq et al. 2024) 多模态政治事实核查;PACAR (Zhao et al. 2024b) 加入 planning + customized action reasoning;STEEL (Li et al. 2024b) 做 step-by-step evidence verification。
- **Agentic QA**: RARE (Tran et al. 2024)、RAG-Star (Jiang et al. 2024)、MindSearch (Chen et al. 2024b) iteratively refine reasoning with broad web content。MindSearch 模仿人类 search behavior,做 query decomposition + parallel snippet reading + synthesis。
- **Medical Reasoning**: FRVA (Fan et al. 2024b) 做 fact-retrieval + verification augmented entailment tree;ALR² (Li et al. 2024d) 是 retrieve-then-reason framework for long-context medical QA。

#### 4.1.3 Tool Using

- **Re-Invoke** (Chen et al. 2024a): tool invocation rewriting for zero-shot tool retrieval。
- **AVATAR** (Wu et al. 2024): contrastive reasoning 优化 tool usage。
- **ToolkenGPT** (Hao et al. 2023): 把 tool 当成 token embedding,和 vocabulary token 一起 decode。形式化: $P(\text{tool}_i | h) = \text{softmax}(W_{tool} \cdot h)$,其中 $h$ 是 hidden state, $W_{tool}$ 是 tool embedding matrix。
- **Tool-LLM** (Qin et al. 2023): 16000+ real-world APIs 的 instruction tuning。
- **SCIAGENT** (Ma et al. 2024b)、**TRICE** (Qiao et al. 2024): 集成 WolframAlpha 等 symbolic computation tool。
- **RAR** (Dutta et al. 2024): low-resource language 的 code generation,retrieve OSCAT library docs。

### 4.2 In-context Retrieval

这一类用 model 内部的 experience 或者 retrieved examples,不涉及 external tool 调用。

#### 4.2.1 Prior Experience

- **RAHL** (Sun et al. 2024a)、**RA-DT** (Schmied et al. 2024): 把 past decision trajectory 和 RL reward signal 存成 memory,retrieve 来 guide future reasoning。RA-DT 是 Decision Transformer + retrieval: $\pi(a | s, \text{retrieved trajectory}) = \text{DT}(s \oplus \text{traj}_{retrieved})$。
- **JARVIS-1** (Wang et al. 2024f): multimodal memory-augmented agent,dynamically recall interaction history。
- **EM-LLM** (Fountas et al. 2024): human-like episodic memory,用 surprise metric 触发 segmentation,把 infinite context 压成 episode chunks。
- **CoPS** (Yang et al. 2024a): cross-task experience sharing,retrieve structured prior cases。

#### 4.2.2 Example / Training Data Retrieval

- **RE4** (Li et al. 2024c): relation extraction,用 annotated sentence pairs 做 demonstration。
- **OpenRAG** (Zhou & Chen 2025)、**UPRISE** (Cheng et al. 2023)、**MoD** (Wang et al. 2024c)、**Dr.ICL** (Luo et al. 2023): ICL demonstration retrieval。UPRISE 训 universal prompt retriever。
- **PERC** (Yoo et al. 2025): 用 pseudocode 做 plan-as-query example retrieval,保证 alignment with target code semantics。

---

## 5. Synergized RAG-Reasoning — The Core Contribution

这是 paper 的 main course。Section 5 提出从 isolated enhancement 走向 iterative interplay,对应 Figure 1 中间的双向箭头 (RAG ⇔ Reasoning)。这一章分两条 axis: **Reasoning Workflow** (结构化 inference format) 和 **Agent Orchestration** (agent 如何交互)。

### 5.1 Reasoning Workflow

#### 5.1.1 Chain-based

**IRCoT** (Trivedi et al. 2023) 和 **RAT** (Wang et al. 2024g) 是奠基性工作。形式化:

```
r_0 = ""
for t = 1, ..., T:
    d_t ~ Retrieve(q | r_{t-1})
    r_t = LLM(r_{t-1}, d_t, q)
```

这里 $r_t$ 是第 $t$ 步的 reasoning state, $d_t$ 是 retrieved evidence。关键 design choice 是 retrieval query 随着 reasoning 演化。

**CoV-RAG** (He et al. 2024a): 在每步加入 verification:
$$v_t = \text{Verify}(r_t, d_t)$$
如果 $v_t = \text{fail}$,re-retrieve 或者 revise $r_t$。这个 verification chain 直接对应 Self-RAG 的 reflection marker 但用在 multi-step 场景。

**RAFT** (Zhang et al. 2024a): 训练时主动加入 distractor documents,让 LLM 学会 ignore noise。Loss 是标准 LM loss 但是 context $C$ 包含 $\text{relevant} \cup \text{distractors}$。

**Chain-of-Note** (Yu et al. 2024): 让 LLM 对每个 retrieved doc 生成 sequential "reading note",filter out unhelpful info。形式化: $n_i = \text{LLM}(q, d_i)$,$n_i$ 是 note,然后 final answer 基于 $\{n_i\}$ 而非 raw $\{d_i\}$。

#### 5.1.2 Tree-based

**ToT** (Yao et al. 2023a): 把 CoT 扩展成 deterministic tree,branch multiple logical pathways。每个 node 是一个 thought,children 是不同的 continuation。用 BFS 或 DFS 加上 self-evaluation $\text{Eval}(\text{thought}) \in [0,1]$ 决定 expand 哪个 node。

**RATT** (Zhang et al. 2025a): retrieval-augmented thought tree,同时 evaluate 多个 reasoning trajectory,避免 early mistake trap。应用于 ambiguous question (Kim et al. 2023)、medical diagnosis (Yang & Huang 2025)、complex story generation (Wen et al. 2023)。

**MCTS-based**:
- **AirRAG** (Feng et al. 2025): 用 self-consistency check 保证 retrieval 和 reasoning quality。
- **MCTS-RAG** (Hu et al. 2025): adaptive MCTS retrieval refine evidence。
- **SeRTS** (Hu et al. 2024): self-rewarding tree search for biomedical RAG。

MCTS 的核心公式 (UCB1 selection):
$$\text{UCB}(s, a) = Q(s, a) + c \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

其中:
- $s$ 是 tree node (一个 reasoning state)
- $a$ 是 action (一个 retrieval 或 reasoning step)
- $Q(s, a)$ 是该 action 的 average value (来自 self-evaluation 或 reward model)
- $N(s)$ 是 parent node 的 visit count
- $N(s, a)$ 是从 $s$ 走 action $a$ 到 child 的 visit count
- $c$ 是 exploration constant (通常 $c = \sqrt{2}$)

MCTS 的 budget-aware 特性 (Table 6 提到) 是关键 — 在 tight API-call budget 下可以集中 exploration 在 promising branch,实现 graceful anytime stopping。

#### 5.1.3 Graph-based

两个分支:

**(a) Walk-on-Graph**: 用 graph learning 技术,不直接让 LLM 推理。
- **PullNet** (Sun et al. 2019)、**QA-GNN** (Yasunaga et al. 2021)、**GreaseLM** (Zhang et al. 2022b): GNN-based iterative aggregation。
- **LightRAG** (Guo et al. 2024)、**StructRAG** (Li et al. 2024h): lightweight graph 技术 (vector index + PageRank) 做 efficient multi-hop retrieval。

**(b) Think-on-Graph**: 让 LLM 直接 drive graph traversal。
- **ToG** (Sun et al. 2024b): LLM 用 KG 当 "reasoning playground",每步决定 explore 哪个 entity/relation。
- **Graph-CoT** (Jin et al. 2024): 三阶段 iterative loop (reasoning → graph interaction → execution)。
- **KGP** (Wang et al. 2024d): 先构建 document-level KG,然后 LLM-driven graph traversal agent。
- **GraphReader** (Li et al. 2024f): coupling LLM reasoning with explicit subgraph retrieval + evidence anchoring。

Graph-based 的好处是 verifiable citation,坏处是 high latency (很多 micro-tool calls) 和 search space explosion without pruning。

### 5.2 Agent Orchestration — The Agentic Paradigm

这是 Section 5 的灵魂,也是 OpenAI Deep Research / Perplexity Deep Research / Gemini Deep Research 的技术基础。

#### 5.2.1 Single-Agent

**Prompt-based**:
- **ReAct** (Yao et al. 2023b): 经典的 Thought-Action-Observation 三元组循环:
```
Thought_i: LLM reason about current state
Action_i: tool call (e.g., Search["query"])
Observation_i: tool return
```
直到 LLM emit "Finish" action。

- **Self-Ask** (Press et al. 2023)、**IR-CoT** (Trivedi et al. 2023): 不是显式 action command,而是 recursive sub-question formulation。

- **DeepRAG** (Guan et al. 2025)、**Self-RAG** (Asai et al. 2024): self-reflection 让 LLM 自己判断何时 retrieve。

**SFT-based**:
- **Toolformer** (Schick et al. 2023): self-supervised,LLM 自己 decide 何时 call API,在 synthetic data 上 fine-tune。
- **INTERS** (Zhu et al. 2024): 20 个 task、43 个 dataset 的 instruction tuning 数据,manually written template,reformulate 成 instructional format。

**RL-based** — 这是最近的 frontier:

- **WebGPT** (Nakano et al. 2021): 早期 work,用 human feedback 训 browser-assisted QA,reward 来自 human preference。
- **RAG-RL** (Huang et al. 2025a): curriculum learning + RL,在 factual correctness reward 上优化。
- **Search-R1** (Jin et al. 2025): 训练 LLM 在 reasoning 过程中 emit `<search>` token,trigger retrieval。Reward 是 exact match + format reward:
$$R = R_{em} + \beta \cdot R_{format}$$
其中 $R_{em} = \mathbb{1}[\hat{a} = a^*]$ 是 exact match indicator,$R_{format}$ 检查 `<search>...</search>` tag 是否 well-formed。用 GRPO 优化:
$$\mathcal{L}_{GRPO}(\theta) = -\mathbb{E}_{q \sim \mathcal{D}, \{o_i\} \sim \pi_{\theta_{old}}(\cdot | q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{\pi_\theta(o_i | q)}{\pi_{\theta_{old}}(o_i | q)} \hat{A}_i \right]$$
其中 $G$ 是 group size, $\hat{A}_i$ 是 advantage (在 group 内归一化), $\pi_\theta$ 是 current policy。GRPO 不需要 critic network,比 PPO 更适合 long-context agent scenario。

- **R1-Searcher** (Song et al. 2025): 类似 Search-R1,加入 retrieval in-context signal,generalization 跨 domain 更强。
- **DeepResearcher** (Zheng et al. 2025): first end-to-end RL-trained research agent,直接和 open web 交互。Reward 来自 final answer correctness,但 agent 学到了 emergent 的 decomposition、iterative verification、retrieval planning capability — 这是 SFT 难以 instill 的。
- **ReSearch** (Chen et al. 2025b)、**ReARTeR** (Sun et al. 2025c): 强调 reasoning step 既要 factual 又要 interpretable,引入 process reward model (PRM) 评估每一步 quality:
$$R_{process}(s_t) = \text{PRM}(s_t, a_t)$$
$$R_{total} = \sum_t \gamma^t R_{process}(s_t) + R_{outcome}(a_{final})$$
其中 $\gamma \in [0,1]$ 是 discount factor。

**ZeroSearch** (Sun et al. 2025a): 一个 interesting 的工作 — 用 LLM 自己 simulate search 而不真的 call search engine,降低 training cost。这对应 "internalized retrieval" 的方向。

#### 5.2.2 Multi-Agent

**Decentralized**: 多个 agent 各自 retrieve from partitioned data source。
- **Wang et al. 2024e**、**Salve et al. 2024**: 多 agent 从不同 DB partition retrieve。
- **Collab-RAG** (Xu et al. 2025b): white-box + black-box LLM collaboration。
- **RAG-KG-IL** (Yu & McQuade 2025): RAG + incremental KG learning 双 agent。
- **MDocAgent** (Han et al. 2025): 多模态,文本 agent + 图像 agent 协同处理 document QA。
- **Agentic Reasoning** (Wu et al. 2025c): 通用 framework,unite tool-using agents for search、computation、structured reasoning。

**Centralized / Hierarchical**: manager-worker 架构。
- **HM-RAG** (Liu et al. 2025)、**SurgRAW** (Low et al. 2025): decomposer-retriever-decider 三层架构。
- **Wu et al. 2025a**: dynamic routing based on task relevance。
- **Iannelli et al. 2024**: SLA-aware reconfigurable multi-agent RAG。
- **Chain of Agents** (Zhang et al. 2024c): 层级 pipeline,用于 long-context summarization。

Table 6 给出 trade-off 矩阵,这里我展开几个关键对比:

| Category | Strengths | Limitations |
|---|---|---|
| Chain-based | Low latency, easy cache | Error propagation, context bloat |
| Tree-based (ToT) | High recall, transparent | Quadratic cost in API calls |
| Tree-based (MCTS) | Budget-aware, anytime stopping | Tuning-heavy, suboptimal convergence |
| Graph (Walk-on-Graph) | Efficient on curated KG | Needs high-quality KG |
| Graph (Think-on-Graph) | Adaptive, verifiable citation | High latency, search explosion |
| Single-agent (RL) | Adaptive, high recall | Hard reward design, expensive training |
| Multi-agent (Decentralized) | High recall, robust | Communication overhead, consensus needed |
| Multi-agent (Centralized) | Budget-efficient, provenance | Manager bottleneck |

---

## 6. Deep Research Implementations — Appendix B 的干货

Table 5 列出当前 SOTA deep research system 的对比,我提炼几个关键 trend:

### Optimization Method 分布
- **Prompting-based**: gpt-researcher, deep-searcher, nanoDeepResearch, DeerFlow, open-deep-research, node-DeepResearch, deep-research, Open Deep Search, Agentic Reasoning, r1-reasoning-rag。这些是工程化实现,用 prompt engineering + ReAct loop 驱动 commercial LLM (GPT、Claude、Gemini、Deepseek)。
- **DPO-based**: Webthinker (Li et al. 2025c) 用 preference pair 训 Deepseek-R1、QwQ-32B。
- **GRPO-based**: Search-R1, ZeroSearch, DeepResearcher, R1-Searcher, ReSearch。GRPO 是 DeepSeek-R1 推广的 RL algorithm,在 RAG-reasoning 上成为主流。

### Agent Architecture 分布
- **Single-agent**: 占大多数,因为 context sharing 容易,但难处理 specialized task。
- **Decentralized**: DeerFlow (Qwen)。
- **Centralized/Hierarchical**: Agentic Reasoning, gpt-researcher, deep-searcher, nanoDeepResearch。

### Retriever
绝大多数用 **Web Search**,因为 deep research 需要实时、open-domain 信息。少数支持 Local Retrieval (gpt-researcher, deep-research, R1-Searcher, r1-reasoning-rag)。

### Evaluation Data
- 经典 QA: NQ, TriviaQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle, PopQA
- High-difficulty: GPQA, GAIA, HLE (Humanity's Last Exam), MATH500, AMC2023, AIME2024, LiveCodeBench, SimpleQA, FRAME, WebWalkerQA

Webthinker 在 GPQA + GAIA + WebWalkerQA + HLE 上 eval,是目前覆盖最难 benchmark 的 system 之一。

---

## 7. Benchmarks — 系统性分析

Table 1 和 Table 2 (Appendix A) 列出 46 个 benchmark 覆盖 13 个 task 类别。

### Benchmark 的核心 Retrieval & Reasoning Challenge (Table 4)

| Benchmark | Retrieval Challenge | Reasoning Challenge |
|---|---|---|
| TriviaQA, NQ | Scale & Noise from massive corpus | Ambiguity in underspecified queries |
| HotpotQA, MuSiQue, HLE | Multi-document synthesis across Wikipedia | Multi-hop deduction linking discrete facts |
| MMLU-Pro, QuALITY | Expert-level retrieval from academic/narrative | Complex long-form reasoning (QuALITY >5k tokens) |
| MATH, AQuA-RAT | Formal knowledge retrieval (theorems) | Symbolic & deductive multi-step |
| LiveCodeBench | Heterogeneous source (repo, docs, SO) | Tool use + self-correction via test execution |
| BrowseComp, WebWalkerQA | Agentic planning in live web | Strategic multi-step (search, click, extract) |

### 关键 insight

1. **Domain gap**: 大多数 benchmark 集中在 general/academic,缺乏 industrial、vertical-domain、personalized 场景。
2. **Reasoning 类型 imbalance**: 当前 benchmark 主要 test deductive reasoning,counterfactual、causal、analogical reasoning 严重 underexplored。
3. **Trajectory evaluation 缺失**: 没有 standardized metric 评估 entire reasoning-retrieval trajectory (retrieval step efficiency、intermediate query quality、multi-step chain consistency)。
4. **Trust dimension 缺失**: 没有 benchmark 系统评估 robustness against noisy/evolving/conflicting 信息。

---

## 8. Future Work — 关键 Research Direction

Paper Section 7 列出 6 个方向,我重点 expand 三个:

### 8.1 Reasoning Efficiency

Deep research 一个 query 经常需要 10+ 分钟,这是 production 落地的 critical bottleneck。三个方向:

**(a) Latent Reasoning**: 不在 token space 展开 reasoning,而在 continuous hidden state 上做 "thinking",类似 coconut (Training LLMs to Reason in Continuous Latent Space) 思路。

**(b) Thought Distillation + Length Penalty**: TokenSkip (Xia et al. 2025a)、LightThinker (Zhang et al. 2025b) 压缩 CoT,加 length penalty:
$$\mathcal{L} = \mathcal{L}_{LM} + \lambda \cdot \mathbb{E}[|r|]$$
其中 $|r|$ 是 reasoning chain length, $\lambda$ 是 penalty 系数。

**(c) Model Compression**: quantization、pruning、knowledge distillation 做高效 small RAG-reasoning model。这个方向 Qwen2.5-3B 系列 + Search-R1 已经在做。

### 8.2 Human-Agent Collaboration

Open-ended task (literature review、interactive programming) 需要 human-in-the-loop:
- Modeling user intent under uncertainty (Zhang et al. 2025e, Yang et al. 2025)
- Interactive clarification interface
- Adaptive reasoning strategy based on user expertise (PersonaAgent, Zhang et al. 2025g)
- Human-LLM symbiotic collaboration (SymbioticRAG, Sun et al. 2025b)

这和你 eureka labs 教学中强调的 "education as the killer app of LLMs" 高度 align — 教育场景天然需要 human-agent collaboration。

### 8.3 Retrieval Trustworthiness

Adversarial attack 通过 poisoned corpus 污染 retrieval:
- Watermarking、digital fingerprinting 增强 traceability
- Uncertainty quantification (Shorinwa et al. 2025)
- Robust generation under noisy context
- Multi-dimensional trust metrics beyond accuracy

---

## 9. 一些 Critical Reflection

我读完这篇 survey 有几个 meta-level 的观察:

**(1) 从 RTR 到 RAG⇔Reasoning 的范式转移**:这是 LLM 应用层最重要的 evolution 之一。本质上对应人类 "System 2 thinking" 的 emulative — 不是 single pass,而是 iterative hypothesize-test-revise。你之前在斯坦福 CS25 讲的 "reasoning model" lecture 里提到 test-time compute scaling,这篇 survey 把这个概念和 RAG 自然结合。

**(2) RL-based RAG 是新 frontier**:Search-R1、R1-Searcher、DeepResearcher 这些 RL-trained system 在 multi-hop QA 上展现出的 emergent capability 是 SFT 难以复现的。这呼应了 DeepSeek-R1 论文里 "RL can teach reasoning without SFT on CoT data" 的 thesis。

**(3) Benchmarks 落后于 method**:HLE、BrowseComp 是 2024 年底 / 2025 年初才出的,说明 evaluation 严重滞后。特别是缺乏 trajectory-level evaluation 和 trustworthiness evaluation,这是 open research gap。

**(4) Multimodal extension 是 obvious next step**:目前绝大多数 system 仍 text-only。MDocAgent、SurgRAW 已经开始探索 multimodal multi-agent RAG,但 unified multimodal retriever (image + table + text + heterogeneous doc) 还未成熟。

**(5) Theoretical foundation 缺失**:为什么 iterative retrieval-reasoning 比 single-pass 更有效?信息论角度的解释 (mutual information gain at each step)、计算复杂度分析、最优 stopping time 的理论都还很少看到。这是 academic 上 future 的 fertile ground。

---

## Reference Links

- Paper GitHub: https://github.com/DavidZWZ/Awesome-RAG-Reasoning
- Search-R1: https://arxiv.org/abs/2503.09516
- R1-Searcher: https://arxiv.org/abs/2503.05592
- DeepResearcher: https://arxiv.org/abs/2504.03160
- Webthinker: https://arxiv.org/abs/2504.21776
- ZeroSearch: https://arxiv.org/abs/2505.04588
- ReSearch: https://arxiv.org/abs/2503.19470
- ReARTeR: https://arxiv.org/abs/2501.07861
- DeepRAG: https://arxiv.org/abs/2502.01142
- AirRAG: https://arxiv.org/abs/2501.10053
- MCTS-RAG: https://arxiv.org/abs/2503.20757
- Self-RAG: https://arxiv.org/abs/2310.11511 (original paper)
- ReAct: https://arxiv.org/abs/2210.03629
- IRCoT: https://arxiv.org/abs/2212.10509
- ToT: https://arxiv.org/abs/2305.10601
- Toolformer: https://arxiv.org/abs/2302.04761
- BrowseComp: https://arxiv.org/abs/2504.12516
- HLE: https://arxiv.org/abs/2501.14249
- LightRAG: https://arxiv.org/abs/2410.05779
- GraphRAG / ToG: https://arxiv.org/abs/2407.10805
- GNN-RAG: https://arxiv.org/abs/2405.20139
- Open-RAG: https://arxiv.org/abs/2410.01784
- RAG-RL: https://arxiv.org/abs/2503.12759
- INTERS: https://arxiv.org/abs/2501.01715
- PersonaAgent: https://arxiv.org/abs/2506.06254
- Agentic Reasoning (Wu et al.): https://arxiv.org/abs/2502.04644
- MDocAgent: https://arxiv.org/abs/2503.13964
- HM-RAG: https://arxiv.org/abs/2504.12330
- SurgRAW: https://arxiv.org/abs/2503.10265
- Collab-RAG: https://arxiv.org/abs/2504.04915
- RAG-KG-IL: https://arxiv.org/abs/2503.13514
- DeepSeek-R1 (RL background): https://arxiv.org/abs/2501.12948
- Coconut (continuous reasoning): https://arxiv.org/abs/2412.06769
- TokenSkip: https://arxiv.org/abs/2502.12067
- LightThinker: https://arxiv.org/abs/2502.15589
- BrowseComp-ZH: https://arxiv.org/abs/2504.19314
- Awesome-LLM-Reasoning-RL (相关 RL survey): https://github.com/Agent-ML/awesome-llm-rl

---

## Final Intuition

这篇 survey 的核心 mental model 可以浓缩成一句话: **Reasoning 和 Retrieval 是同一个 inference loop 的 two sides — reasoning 产生 information need,retrieval 满足 need,新 evidence 又驱动更深 reasoning,如此迭代直到 convergence**。这就是当前 deep research system 的 design principle,也是 future agentic AI 的 blueprint。

对你 eureka labs 的教学启发:学生学 LLM 时不应只学 single-pass inference,而应学这种 iterative think-search-think 的 reasoning pattern — 这才是 future-proof 的 mental model。同时,human-in-the-loop collaboration 的方向 (Section 7) 和 eureka labs "education-first LLM" 的愿景高度 align,值得 deep dive。
