---
source_pdf: AgentFold Long-Horizon Web Agents with.pdf
paper_sha256: 87a0f274e46cc47da4fa50e308c2e9cc09baf9bf382e5a090acfb2910d286ed1
processed_at: '2026-07-18T04:55:46-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AgentFold:为 long-horizon web agent 做 proactive context management

这是 Tongyi Lab (Alibaba) 的一篇工作,作者 Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin 等。核心 claim:**让 agent 在每一步 reasoning 的同时,主动对自己的 context 做一次 multi-scale 的"折叠"(fold),从而在 100~500 turn 的 long-horizon 任务上既不爆 context、也不丢关键信息。** 30B-A3B 的 MoE 模型在 BrowseComp 上拿到 36.2%,打败 DeepSeek-V3.1-671B-A37B (30.0%) 和 OpenAI o4-mini (28.3%)。这个结果本身就值得仔细看。

我会按 paper 的章节顺序展开,但每节都加上技术细节、公式解释、以及我自己的直觉与联想。

---

## 1. Motivation:ReAct 与 summarization 之间的 trade-off

paper 一上来就把领域的两种极端做法摆出来,并指出它们各自的病:

**(A) ReAct (Yao et al. 2023)**:context = 完整 history of (reasoning, action, observation) triplets,append-only。信息保真度高,但 turn 数一多,context 单调线性膨胀,signal 被 noise 淹没 → **context saturation**。所有当前主流的 open-source agent(WebThinker, WebDancer, WebSailor, ASearcher, MiroThinker, WebExplorer, DeepDive, DeepDiver-V2)都 build on 这个 paradigm。

**(B) MEM1 (Zhou et al. 2025b) / MemAgent (Yu et al. 2025)**:每一步把整段 history 强制重新 summarize。context 干净,但**单次 summarization 就可能 irreversibly 丢掉关键 detail**,而且越往后丢得越多。

paper 用一个简单的概率 argument 把 (B) 的病量化。假设每次 summarize 时某个关键 detail 有概率 $p$ 被丢掉,经过 $T$ 次 summarization 后该 detail 存活概率:

$$P_{\text{survive}}(T) = (1 - p)^T$$

取 $p = 0.01$:
- $T = 100 \Rightarrow P \approx 0.99^{100} \approx 36.6\%$
- $T = 500 \Rightarrow P \approx 0.99^{500} \approx 0.66\%$

这里 $p$ 是 single-step loss probability,$T$ 是 summarization 次数(等于 turn 数)。**指数衰减**意味着 full-history summarization 在长任务上是致命的。

AgentFold 的 motivation 概括成一句话:**别让 summarization 是机械的、强制每步的;让 agent 自己决定何时 fold、fold 多少、保留什么、丢掉什么**。这个动机非常符合 "context engineering is the new programming" 这一直觉。

---

## 2. 核心 idea:Multi-Scale State Summaries + Proactive Fold

paper 借鉴 cognitive science 里的 **retrospective consolidation**(Newell & Simon 1972; Miller 1956)——人类解决问题时既不是把所有记忆原样保留,也不是机械地每步总结;而是在 critical point 上"回头看一眼",discard irrelevant steps,distill intermediate findings,abstract key insights。

AgentFold 把这个机制搬进 LLM agent:在 step t 的 response 里同时输出 (1) 一个 **folding directive** 决定怎么折叠历史,(2) 一个 explanation,(3) 一个 next action。fold 是 learned 的、显式的、与 action 共生成的。

关键设计有两个 scale:
- **Granular Condensation**($k = t-1$):只把当前 Latest Interaction 压成单个 fine-grained summary,保留最高分辨率
- **Deep Consolidation**($k < t-1$):把当前 Latest Interaction + 一串 prior summaries 合并成单个 coarse summary,scale 跳变

这让我想到几个类比:

1. **Video coding 的 I-frame / P-frame**:关键帧全保,非关键帧只存差分。Granular condensation ≈ 关键帧,deep consolidation ≈ 把一堆 P-frame 折叠回一个新的 I-frame。
2. **Git squash commit**:把一串细碎 commit 合并成一个 atomic commit。Deep consolidation 就是 squash。
3. **Hippocampus → neocortex memory consolidation**:海马体的 episodic memory 在 sleep 中 replay 并 integrate 进 neocortex 的 semantic memory。AgentFold 把这个机制在线化(online during inference)。
4. **LSTM forget gate 的 symbolic 版本**:LSTM forget gate 是 token-level、dense、continuous;AgentFold 的 fold 是 step-level、sparse、discrete、symbolic。从 connectionist memory 到 symbolic memory 的抽象层跳。
5. **RL 中的 backtracking / pruning**:dead-end 被 deep consolidate 后,context 反而缩短 —— paper 后面在 500-turn 实验里观察到 non-monotonic context growth,这就是 self-correcting context management。

---

## 3. 架构细节:Context 与 Response 的 dual design

### 3.1 Context 结构 — 公式 (1) 与 (2)

agent 在 step t 的 context 写成(其实是四元组,因为 Q 和 T 都 invariant):

$$C_t = (Q, T, S_{t-2}, I_{t-1}) \quad \text{(1)}$$

变量解释:
- $Q$:invariant user question,作为 anchor 一直留在 context 里,提醒 agent 终极目标
- $T$:工具列表,包含每个 tool 的 name / description / parameter schema
- $S_{t-2}$:Multi-Scale State Summaries,代表 curated long-term memory
- $I_{t-1}$:Latest Interaction,代表 high-fidelity working memory

**注意 indexing 的微妙**:long-term memory 是 $S_{t-2}$,working memory 是 $I_{t-1}$,差一拍。原因是 fold directive 要作用于 $I_{t-1}$ 之后才生成新的 summary 进入 $S_{t-1}$。也就是说,step t 时,working memory 还是上一步完整记录 $I_{t-1}$,而 long-term memory 是 t-2 及更早的 summary 集合。等到 agent 输出 fold directive,才把 $I_{t-1}$ 转化进 $S_{t-1}$,供 step t+1 使用。

**初始化**:
- $t = 1$:$C_1 = (Q, T, \emptyset, \emptyset)$,只有 question
- $t = 2$:$C_2 = (Q, T, \emptyset, I_1)$,有 question 和第一次 interaction,但还没有 fold 出来的 summary

$S_{t-2}$ 的形式化定义,公式 (2):

$$S_t = (s_{x_1, y_1}, s_{x_2, y_2}, \dots, s_{x_m, y_m}) \quad \text{(2)}$$

变量:
- 每个 $s_{x,y}$ 是一段 textual summary,覆盖从 step $x$ 到 step $y$ 的连续步
- $m$:当前 summary 序列的 block 数

三个约束让这个序列始终是一个对历史的"不相交划分":
- $x_1 = 1$(从最早开始)
- $y_m = t-2$(到上一步为止)
- $x_{i+1} = y_i + 1$ for all $i$(无重叠、无遗漏地覆盖整段历史)

**这就是 "Multi-Scale" 的精髓**:单个 step 可以被压缩成 $s_{x,x}$ ($y = x$),而多个 step 可以被合并成 $s_{x,y}$ ($y > x$)。整个历史被切成不相交的区间,每个区间的"分辨率"由 agent 自己决定。

Latest Interaction 的定义:
$$I_{t-1} = (e_{t-1}, a_{t-1}, o_{t-1})$$

也就是上一步的 explanation + action + observation 三段拼接,**完整保留**,不做任何压缩。这给 agent 提供 situational awareness。

这种 context 设计直接对应人类的认知结构:**stable goal (Q) + consolidated knowledge (S) + volatile working memory (I)**。Working memory 容量小但清晰;long-term memory 容量大但抽象。两者协同。

### 3.2 Response 结构 — 公式 (3)

每一步 agent 的输出是一个**可被 parser 切成四块的单段 text**:

$$R_t = \text{AgentFold}(C_t; \theta) \rightarrow (th_t, f_t, e_t, a_t) \quad \text{(3)}$$

变量:
- $th_t$:thinking,长 chain-of-thought monologue,内部分析 context、权衡 fold 范围与 next action
- $f_t$:folding directive,JSON 格式
- $e_t$:explanation,简洁说明 next action 的动机
- $a_t$:action(tool call 或 final answer)

最关键的是 $f_t$ 的形式:

$$f_t = \{\text{"range"}: [k, t-1], \text{"summary"}: \sigma_t\}$$

- $k$:folding 起始 step id,由 agent 自己填
- $\sigma_t$:新的 summary 文本,由 agent 自己写

**单一格式,两种 mode**:

| Mode | 条件 | 作用 | 典型例子 |
|---|---|---|---|
| Granular Condensation | $k = t-1$ | 把 $I_{t-1}$ 压成新 summary block | "[Compressed Step 5] Found new candidate XYZ that needs further exploration." |
| Deep Consolidation | $k < t-1$ | 把 $I_{t-1}$ + prior summaries $s_{k,\cdot}, \dots, s_{\cdot,t-2}$ 合并成单个 coarse summary | "[Compressed Step 5 to 9] Confirmed XYZ does not fit all criteria after checking several sources." |

执行 fold 的过程在 paper 第 3.3 节描述:把所有 step 落在 $[k, t-1]$ 区间内的 summary blocks 撤回,替换成单个新 block $s_{k, t-1} = \sigma_t$。这等价于在 summary 序列上做一次"区间替换",非常像 git rebase -i 里的 squash。

执行完 fold 后:
- 新 long-term memory:$S_{t-1}$(把 $[k, t-1]$ 区间替换成 $\sigma_t$)
- 新 working memory:$I_t = (e_t, a_t, o_t)$(当前步的 explanation + action + observation)
- 下一步 context:$C_{t+1} = (Q, T, S_{t-1}, I_t)$

这就完成了 perceive → reason → fold → act 循环。

### 3.3 为什么 thinking 和 folding 要耦合

paper 在 3.3 末尾给了一个非常精辟的认知论点,值得单独拎出来:

> **要求 agent 显式输出 folding directive,等于强制它对自己的 trajectory 做一次反思**;这个反思本身就 sharpens 它对当前 state 的理解,从而让 next action 更好。反过来,**要决定 next action,agent 必须审视 recent history 找出关键 clue**;这个审视过程天然提供了"什么值得 fold 保留"的 signal。Fold 和 action 互为 mirror,互相 sharpen。

这种 coupling 我觉得是 paper 最深刻的 design choice。它把 context curation 从 passive byproduct 升级为 reasoning 的内在组成部分。这一点跟 Karpathy 之前讲过的 "agent 应该把 context window 当作自己的 working memory 主动 edit" 的直觉完全吻合。

---

## 4. 数据与训练:Fold-Generator pipeline

paper 没给 Fold-Generator 太多公式,但流程是清楚的:

1. **Question set**:复用 WebSailor-V2 (Li et al. 2025a) 的 question set,保证 fair comparison
2. **Generator LLM**:用强大的 open-source LLM(具体哪个 paper 没明说,推测是更大的 Qwen3 或 DeepSeek)生成 trajectory,每一步要严格输出 AgentFold 要求的四元组格式
3. **Rejection sampling**:
   - 丢弃任何 step 不严格符合格式的 trajectory
   - 丢弃任何 environmental error 太多的 trajectory
4. **Output**:$\{(C_t, R_t^*)\}_N$,其中 $C_t$ 是结构化 context,$R_t^*$ 是验证过的 gold response,$N$ 是所有 question 的总 interaction step 数
5. **SFT** on Qwen3-30B-A3B-Instruct-2507 (Yang et al. 2025)

关键 insight paper 反复强调:

> "Even the most advanced LLMs cannot reliably produce AgentFold's accurate, structured, multi-part responses through prompt engineering alone."

也就是说,**fold 是一个需要训练才能稳定的 skill**,prompt engineering 不够。这跟 agentic RL 的 "SFT bootstrap" 思路一致 —— 先用 rejection sampling 蒸馏出能稳定产出 format 的 base policy,再考虑 RL refine。

但 AgentFold **只做了 SFT,没做 RL**。结尾明确承认:

> "The clear next step is to leverage reinforcement learning (RL) to enable the agent to autonomously discover optimal and potentially non-obvious folding policies by directly optimizing for task success."

这是一个清晰的 opening。我自己觉得 fold policy 的搜索空间非常大($k \in [1, t-1]$ 是离散选择,$\sigma_t$ 是 free-form text),在这种 combinatorial + open-ended 空间里 RL converge 是 non-trivial challenge,但 reward signal 直接(任务成功率),credit assignment 也不算太难(fold 的好坏可以通过"后续 action 是否还需要被折叠掉的 information"来判断)。半年内大概率会有 AgentFold + GRPO/PPO 的工作出来。

---

## 5. 实验结果:30B 怎么打过 671B

### 5.1 主结果表(Table 1)

我把关键几行抽出来:

| Agent | BrowseComp | BrowseComp-ZH | WideSearch | GAIA |
|---|---|---|---|---|
| Claude-4-Sonnet | 14.7 | 22.5 | 62.0 | 68.3 |
| Claude-4-Opus | 18.8 | 37.4 | – | – |
| OpenAI o4-mini | 28.3 | 44.3 | – | – |
| OpenAI o3 | 49.7 | 58.1 | 60.0 | 70.5 |
| OpenAI Deep Research | 51.5 | 42.9 | – | 67.4 |
| DeepSeek-V3.1-671B-A37B | 30.0 | 49.2 | – | 63.1 |
| GLM-4.5-355B-A32B | 26.4 | 37.5 | – | 66.0 |
| Kimi-K2-Instruct-1T | 14.1 | 28.8 | 59.9 | 57.3 |
| WebSailor-72B | 12.0 | 30.1 | – | 55.4 |
| WebSailor-32B | 10.5 | 25.5 | – | 53.2 |
| WebDancer-32B | 3.8 | 18.0 | – | 51.5 |
| WebThinker-32B | 2.8 | 7.3 | – | 48.5 |
| WebExplorer-8B | 15.7 | 32.0 | – | 50.0 |
| MiroThinker-32B-DPO-v0.2 | 13.0 | 17.0 | – | 64.1 |
| DeepDive-32B | 14.8 | 25.6 | – | – |
| DeepDiver-V2-38B | 13.4 | 34.6 | – | – |
| **AgentFold-30B-A3B (Ours)** | **36.2** | **47.3** | **62.1** | **67.0** |

几个 takeaway:

1. **BrowseComp 上 30B 打过 671B (DeepSeek-V3.1) 与 355B (GLM-4.5)**:这是 context engineering 相对 model scale 的胜利,某种意义上是"算法"对"暴力规模"的胜利。
2. **WideSearch 上 62.1 是所有 method 最高的**,包括 OpenAI o3 (60.0) 和 Claude-4-Sonnet (62.0)。
3. **GAIA 上 67.0**:跟 GLM-4.5 (66.0)、OpenAI Deep Research (67.4) 接近,不是最高但很有竞争力。
4. BrowseComp-ZH 上 47.3 输给 DeepSeek-V3.1 (49.2) 一点点,但仍超过所有更小或同等规模的开源 model。可能是因为中文 search 环境 dead-end detection 更难,或 Qwen3-30B-A3B 中文 base 能力限制了 summary 质量。

**为什么 fold 这么有用?** 我自己解读:BrowseComp 的 task 设计就是要 agent 探索很多 dead end,然后 pivot。ReAct 在这里特别吃亏,因为 dead-end 的 noise 一直留在 context 里;MEM1-style summarization 又会过早把"也许以后还有用"的细节揉掉。AgentFold 可以**等到确认 dead end 之后再 deep consolidate 这一段**,把"失败教训"提炼成一句话结论,然后 pivot 到新方向。这正好对应 Figure 5 case study 的 step 17→18 行为。

### 5.2 Context dynamics(Figure 3a / 3b)

paper 在 BrowseComp 上采样 200 条 trajectory,画出两条曲线。

**Token count** 公式:

$$A_t = \frac{1}{|\mathcal{T}_t|} \sum_{j \in \mathcal{T}_t} \text{TokenCount}(C_{j,t})$$

变量:
- $\mathcal{T}_t$:turn 数 > $t$ 的存活 trajectory 集合
- $C_{j,t}$:第 $j$ 条 trajectory 在 turn $t$ 的 context
- $|\mathcal{T}_t|$:存活 trajectory 数量

结果(图 3a):
- AgentFold:100 turns 时从 ~3.5k token 增长到 ~7k token,**sub-linear**
- ReAct:100 turns 时已经 ~91k token
- **AgentFold 比 ReAct 平均少 84k token,~92% reduction**
- 工程换算:7GB per inference instance 的 KV-cache memory saving

**Block count**(图 3b):
- AgentFold:sub-linear 增长,deep consolidation 不断把多个 block 合并成一个
- ReAct:严格 linear,每个 turn 加一个 block

paper 还观察到一个很有意思的现象:在 500-turn 扩展实验里,AgentFold 的 context 大多 < 20k token,**且非单调增长** —— 当 long line of inquiry 被识别为 dead end 后 deep consolidate 整段失败子轨迹,context 反而缩短。这是 self-correcting context management 的 emergent 行为。

### 5.3 Scaling turns(Figure 4)

paper 把 turn limit 从 64 → 128 → 256 → 500 依次增加,跟 GLM-4.5-355B 对比:

- GLM-4.5-355B 在 > 64 turns 时 saturate 并 fail(append-only context 撑爆)
- AgentFold-30B 在 256 turns 时 accuracy 仍 steady 提升
- 500 turns 扩展实验:context 大多 < 20k token,且非单调

这个结果对 long-horizon agent 的 future 方向意义重大。它意味着 fold 不只是"省 context"的优化,而是**让 agent 真正能 scale 到 hundreds of turns 的 enabler**。对做 deep research、long-form exploration 类产品的人,这是直接相关的工程信号。

### 5.4 Survival curve 的小细节

图 3a 的灰色 bar 是 $|\mathcal{T}_t|$(存活 trajectory 数),它单调下降。paper 指出在 100 turn 上有 20%+ 任务被强制终止 —— 这些通常被标为 failure。但这些被终止 trajectory 的 context 在终止点只有 ~7k token,远低于 128k 上限。这暗示:

> (1) 单纯 increase max turns 大概率能 unlock higher success rate;
> (2) AgentFold 在极端 long-horizon 任务上有巨大潜力。

这是个很"诚实"的观察 —— paper 没有把 100-turn 的 failure 全部归因于 model 能力,而是承认一部分是 turn limit 造成的。

---

## 6. Case study(Figure 5 + Table 2 + Table 3)

paper 给了一个 step 17 的 case,展示 deep consolidation 的关键时刻。

**step 17 时的 context 已经是 multi-scale**:

```
[Compressed Step 0]
[Compressed Step 1 to 2]
[Compressed Step 3]
[Compressed Step 4]
[Compressed Step 5]
[Compressed Step 6 to 8]
[Compressed Step 9]
[Compressed Step 10]
[Compressed Step 11]
[Compressed Step 12]
[Compressed Step 13]
[Compressed Step 14]
[Compressed Step 15]
[Step 16]   ← Latest Interaction
```

注意:`[Compressed Step 1 to 2]` 和 `[Compressed Step 6 to 8]` 是 deep consolidation 的产物,把多个 step 合并成一行;其他 `[Compressed Step X]` 是 granular condensation,把单步压成一行;最后一行 `[Step 16]` 是 working memory,未压缩。

**step 17→18 的 fold**:
- Agent 识别 step 6 到 step 16 是 11 步的失败探索
- 执行 deep consolidation:整个 11 步压成 `[Compressed Step 6 to 16]`
- 基于这个失败结论,重新规划新的 search direction
- 输出 next tool call

Table 2 给出这条 trajectory 完整 35 turns 的 context 演化,可以清楚看到 block 数量随 deep consolidation 时不时回缩。比如 **turn 19 突然从 17 个 block 缩成 7 个 block** —— 那一步就是 step 6→16 的 deep consolidation。

Table 3 给第二条 case,更长(60 turns),里面 step 0→9、step 0→22、step 0→34、step 0→44 等多次大范围 deep consolidation,可以看出 agent 在 different exploration phases 之间频繁 pivot + squash。

---

## 7. 与相关工作的位置

paper 在 Related Works 里把自己定位得很清楚,我加上一些 paper 之外的联想:

### 7.1 Web Agents
- **ReAct (Yao et al. 2023)**:append-only paradigm 的鼻祖。WebThinker, WebDancer, WebSailor, WebSailor-V2, WebShaper, WebExplorer, ASearcher, MiroThinker, DeepDive, DeepDiver-V2 都 build on 它。
- **Test-time scaling**:X-Master (Chai et al. 2025), BrowseMaster (Pang et al. 2025) 在 inference 时做 scaling。
- **OpenAI Deep Research**:prop agent 的代表,但 closed。
- AgentFold 的位置:**保留 ReAct 的 reasoning-experimentation 循环,但在 context 层面引入 proactive multi-scale management**。

### 7.2 Context Management / Context Engineering
两条线:

**External Context Augmentation**:
- Mem0 (Chhikara et al. 2025):production-ready external memory store
- A-MEM (Xu et al. 2025):agentic memory
- Memory3 (Yang et al. 2024):explicit memory 模块直接改 model architecture
- Memos (Li et al. 2025d):memory-augmented generation OS
- 这些都是"需要时 retrieve 进来"或"在 model 内部加 memory module"

**Intra-Task Context Curation**(AgentFold 属于这条线):
- MEM1 (Zhou et al. 2025b):每步强制 full-history summarization
- MemAgent (Yu et al. 2025):类似思路,multi-conv RL-based
- 这些方法在 HotpotQA (Yang et al. 2018) 这种 simpler retrieval-focused task 上 evaluate,在 BrowseComp / WideSearch 这种 long-horizon complex task 上没充分验证

AgentFold 相对 MEM1/MemAgent 的关键差异:**flexible look-back mechanism,避免 rigid step-wise compression,允许在 multi-step scale 上选择性 fold**。这是从"机械 summarizer"到"self-aware knowledge manager"的概念跃迁。

### 7.3 更远的联想

- **Neural Turing Machine / Differentiable Neural Computer (Graves 2014, 2016)**:differentiable external memory,但 connectionist、不可解释。AgentFold 是 symbolic 版本,memory 内容是自然语言 summary,可读可编辑。
- **REINFORCEMENT learning 中的 episodic memory**:MFEC, MERLIN 等用 non-differentiable memory + RL 学习 read/write policy。AgentFold 用 SFT 学习 fold policy,本质相似,但 memory unit 是 text。
- **Retrieval-Augmented Generation (RAG)**:RAG 是"需要时 retrieve 进来",AgentFold 是"已经在 context 里的主动决定丢/留/浓缩"。两者正交,完全可以 RAG + AgentFold 结合。
- **Karpathy 自己讲过的 "micrograd / llm.c" 哲学**:把复杂机制用最简洁的代码实现清楚。AgentFold 的 fold operation 在代码上其实非常简单 —— 一个区间替换 —— 但效果显著。这跟 "small thing done right" 的工程哲学吻合。
- **"Context engineering is the new programming"**:AgentFold 就是这个理念的具体实现 —— agent 不只是生成 token,还生成对自己 context window 的修改操作。这跟 self-modifying code / metaprogramming 是同一类思想,只是迁移到 LLM agent 层面。
- **AlphaGo 的 MCTS pruning**:在 search tree 上主动 prune 无用分支。AgentFold 在 trajectory 上做类似的事,但 prune 的 unit 是 "summarized conclusion",而不是"丢弃"。
- **Software Engineering 里的 working set management**:操作系统的 working set 算法决定哪些 page 留在 memory、哪些 swap out。AgentFold 是 symbolic working set manager。

---

## 8. 我的一些 critical thoughts

读完这篇 paper,我有几个想法:

1. **Fold 是 irreversible 的**。一旦 deep consolidate,原始 details 从 context 消失。如果后续发现需要回看,只能重新 search。这跟人类记忆一致,但带来额外 search cost。一个可能改进是"lazy deep consolidation":先标记为"待 consolidate",等到第 N 步确认无用才真正 fold。但 paper 没做这个。

2. **SFT data 的 quality bottleneck**。Fold-Generator 用 rejection sampling,意味着 base LLM 得有一定能力才能产生合格 trajectory。在更难 domain 上,rejection rate 可能高到无法 scale。这也是为什么 paper 用 WebSailor-V2 的 question set —— 那个 set 是为 SFT-friendly task 设计的。

3. **No RL yet**。Fold 的"好坏"目前由 SFT data 决定,没有直接对 task success 优化。这是 paper 自己承认的"next step"。我预期半年内会有 AgentFold + GRPO 的工作出来。一个有意思的 RL reward design 思路:folding policy 的 reward 可以是"后续 N 步 action 是否还需要被折叠掉的 information" —— 如果一直不需要,说明 fold 成功;如果需要但已经 fold 掉了,说明 fold 过激。这种 intrinsic reward 可以与 task success reward 组合。

4. **Summary 文本 $\sigma_t$ 是 free-form**。模型可能写出 ambiguous 或 information 密度不高的 summary。一个可能改进是结构化 schema,比如强制 field:`{hypothesis, evidence, conclusion, next_action_hint}`。这会增加 parser 复杂度,但提升 summary quality。

5. **BrowseComp-ZH 略输给 DeepSeek-V3.1**。可能是因为 (a) 中文 search 环境 dead-end detection 更难,(b) Qwen3-30B-A3B 中文 base 能力限制了 summary 质量,(c) Fold-Generator 的中文 SFT data 质量不够高。这是一个待解释的细节。

6. **Block count 与 attention 计算成本**。即使 token 数控制在 7k,block 数 sub-linear 增长,attention 的计算还是 $O(n^2)$。但 7k token 的 attention cost 在 30B-A3B 的 inference 上基本可忽略。真正瓶颈是 KV-cache memory,paper 给的 7GB saving 数字很 practical。

7. **Multi-scale 与 hierarchical RL**。AgentFold 的 fold policy 隐式地学到了一个 hierarchical decomposition:sub-task → epilogue summary。这跟 Options framework (Sutton, Precup, Singh 1999) 在 RL 里的 temporal abstraction 是同一个 idea,只是 apply 到 context management 而不是 action space。一个研究方向是把 fold policy 显式分层:low-level 决定 granular condensation,high-level 决定 deep consolidation。这可能让 RL 更容易 converge。

8. **与 chain-of-thought 的关系**。AgentFold 的 thinking $th_t$ 是 chain-of-thought,fold directive 是 CoT 的"产物"。这跟 DeepSeek R1, OpenAI o1 的 "reasoning then answer" 范式同源,但加了"reasoning then fold + answer"的双产物。这是把 CoT 从"内部 reasoning"扩展到"对自己 memory 的 reasoning"。

9. **Generalizability**。paper 只在 web agent 上 validate,但 fold 这个机制原则上对任何 long-horizon agent 都适用:tool-using agent, code agent, robot agent, multi-modal agent。一个有意思的 follow-up 是把 AgentFold 搬到 code editing agent(类似 Devin / SWE-Agent)上,看 fold 对 long-horizon code refactor 任务的帮助。

10. **The "RL next step" 的具体形式**。我猜测最自然的 RL formulation 是:state = $(S_{t-2}, I_{t-1})$,action = $(k, \sigma_t, a_t)$,reward = task success + 中间 intrinsic reward(fold 后 context token reduction, fold 后是否还能回答关于被 fold 内容的 probe question)。这跟 standard RLHF 区别在于 action space 是结构化的。GRPO / PPO 都可以,但需要 careful 设计 token-level advantage 的分配。

---

## 9. 参考链接

**Paper 本身**:
- Project blog: https://tongyi-agent.github.io/blog
- Code: https://github.com/Alibaba-NLP/DeepResearch

**Benchmarks**:
- BrowseComp: https://arxiv.org/abs/2504.12516
- BrowseComp-ZH: https://arxiv.org/abs/2504.19314
- WideSearch: https://arxiv.org/abs/2508.07999
- GAIA: https://gaia-benchmark.github.io/

**ReAct lineage**:
- ReAct (Yao et al. 2023): https://arxiv.org/abs/2210.03629
- WebThinker: https://arxiv.org/abs/2504.21776
- WebDancer: https://arxiv.org/abs/2505.22648
- WebSailor: https://arxiv.org/abs/2507.02592
- WebSailor-V2: https://arxiv.org/abs/2509.13305
- WebShaper: https://arxiv.org/abs/2507.15061
- WebExplorer: https://arxiv.org/abs/2509.06501
- ASearcher: https://arxiv.org/abs/2508.07976
- MiroThinker: https://github.com/MiroMindAI/MiroThinker
- DeepDive: https://arxiv.org/abs/2509.10446
- DeepDiver-V2: https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-DeepDiver
- X-Master: https://arxiv.org/abs/2507.05241
- BrowseMaster: https://arxiv.org/abs/2508.09129
- WebResearcher: https://arxiv.org/abs/2509.13309

**Context management**:
- MEM1: https://arxiv.org/abs/2506.15841
- MemAgent: https://arxiv.org/abs/2507.02259
- Mem0: https://arxiv.org/abs/2504.19413
- A-MEM: https://arxiv.org/abs/2502.12110
- Memory3: https://arxiv.org/abs/2407.01178
- Memos: https://arxiv.org/abs/2505.22101
- Context Engineering survey: https://arxiv.org/abs/2507.13334

**Base / competing models**:
- Qwen3: https://arxiv.org/abs/2505.09388
- DeepSeek-V3.1: https://api-docs.deepseek.com/news/news250821
- GLM-4.5: https://arxiv.org/abs/2508.06471
- Kimi K2: https://arxiv.org/abs/2507.20534
- Claude 4: https://www.anthropic.com/news/claude-4
- OpenAI o3/o4-mini: https://openai.com/index/introducing-o3-and-o4-mini/
- OpenAI Deep Research: https://cdn.openai.com/deep-research-system-card.pdf
- Gemini 2.5: https://arxiv.org/abs/2507.06261

**Cognitive science background**:
- Miller 1956 (magical number 7±2): https://psycnet.apa.org/record/1957-02914-001
- Newell & Simon 1972 (Human Problem Solving): https://books.google.com/books?id=3uwNAQAAIAAJ
- Marchionini 1995 (Information seeking in electronic environments): https://www.cambridge.org/core/books/information-seeking-in-electronic-environments/9C5A0F8E5F5D8E5A1F8E5D5F5A5E5F5D

**Surveys**:
- Agentic RL survey: https://arxiv.org/abs/2509.02547
- MAS-GPT (Ye et al. 2025): https://arxiv.org/abs/2507.04223

---

## 10. 一句话 take-away

AgentFold 把 agent 的 context 从 "append-only log" 升级为 "dynamic cognitive workspace with multi-scale summary blocks + high-fidelity working memory";agent 在每一步同时输出 fold directive、explanation、action,通过 granular condensation 保留关键 fine-grained detail,通过 deep consolidation 把已完成或失败的 sub-task 抽象成单条结论。这个 mechanism 用纯 SFT 蒸馏进 30B-A3B 的 Qwen3,在 BrowseComp 上打败 671B 的 DeepSeek-V3.1 和 OpenAI o4-mini,context 在 100 turn 后只占 7k token,500 turn 后仍 < 20k token 且非单调增长。**Context engineering 在 long-horizon agent 上的杠杆比 model scale 大得多** —— 这是这篇 paper 给我的最强烈的 intuition。
