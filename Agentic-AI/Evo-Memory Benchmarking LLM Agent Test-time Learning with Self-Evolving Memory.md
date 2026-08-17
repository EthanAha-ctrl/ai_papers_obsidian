---
source_pdf: Evo-Memory Benchmarking LLM Agent Test-time Learning with Self-Evolving
  Memory.pdf
paper_sha256: a795dd58d78cdd75f15afd536cf19e5664a333b0fb02a0ab58d8ff3d3bd8f842
processed_at: '2026-08-04T05:40:41-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

## 这篇 paper 到底在说啥

想象你雇了一个人,每天给他派活。第一天他做错了一道数学题,第二天又来一道类似的,他又做错了。你问他:"昨天不是刚做过类似的吗?"他说:"我记得昨天做过,但我不记得怎么做的了。"

这就是现在所有 LLM memory 系统的现状。它们记住了"发生过什么",但没记住"该怎么做"。

这篇 paper 干了三件事:

**第一,定义问题。** 他们说,过去大家研究的 memory 都是"对话回忆"——记住用户上次说了啥。但真正有用的 memory 应该是"经验复用"——记住"这类问题用这个思路解"。这两件事听起来像,实际上差很远。一个是 episodic buffer,一个是 procedural abstraction。

**第二,造了个考试。** 把传统的静态 dataset(比如一组数学题)重新排成一条时间线,前面的题跟后面的题有相似的解题套路。然后看 agent 越往后做,是不是越快越准。如果它学不会从前面任务里提取套路,后面就还是慢。

**第三,提了个方法叫 ReMem。** 原来的 ReAct 让 agent 在"想"和"做"之间切换。ReMem 加了第三个动作叫"整理记忆"——agent 做完一道题,可以回头看看自己 memory 里哪些经验有用、哪些是垃圾,该删的删,该合并的合并。等于让 agent 学会自己收拾自己的经验库。

## 为什么这件事重要

现在 LLM 的训练 compute 已经卷到头了。GPT-5、Claude 4、Gemini 2.5 这些模型,base capability 短期内不会有质的飞跃。那 agent 怎么变得更聪明?两条路:

一条是让模型权重更牛——贵,而且边际收益递减。

另一条是让一个 frozen model 在用的时候越来越聪明——这就是 test-time learning。ReMem 走的就是这条路。

直觉上,这跟人很像。你不是每次遇到新问题都从头想。你脑子里有个"经验库",遇到类似的事就调出来用。用完了发现这个经验不对,就更新一下。ReMem 就是给 LLM agent 装了这么个机制。

## ReMem 到底怎么工作的

原来的 ReAct 循环是:Think(想想) → Act(动手) → Think → Act ...

ReMem 加了一步:Think → **Refine(整理记忆)** → Act

这个 Refine 是关键。它让 agent 干几件事:

- 看看 memory 里拉回来的几条经验,哪条对当前任务有用,哪条没用
- 没用的直接删掉(论文里 agent 输出 "1,3" 表示删第 1、3 条)
- 有时候把几条碎片经验合并成一条更通用的

最妙的是,这全是 agent 自己判断的,没有人教它怎么整理。它就靠 prompt 里给了"你可以删除某条经验"这个选项,LLM 自己就学会了。

## 实验结果到底有多猛

看 embodied 任务(就是那种多步骤的环境,比如 AlfWorld 里让 agent "去厨房拿苹果放冰箱"):

- 不带 memory 的 baseline:AlfWorld 成功率 18%
- 带 memory 但不整理(Mem0 那类):AlfWorld 51%
- ReMem:AlfWorld **92%**

从 18% 到 92%,这是 5 倍提升。在 LLM agent 论文里这种 magnitude 极其罕见。

但是看 single-turn 任务(比如 GPQA 数学题、MMLU):

- Baseline:GPQA 0.48
- ReMem:GPQA 0.51

就提升 3 个点。为什么差这么多?

因为数学题的解法 LLM 在 pretraining 里早就学过了。你给它 memory 告诉它"上次用求根公式",但它本来就会用求根公式。memory 的边际价值约等于零。

而 AlfWorld 里"去厨房拿苹果"这种 action sequence,pretraining 里根本学不到。这是 environment-specific 的 procedural knowledge。只有用的时候才能学。memory 在这种场景下价值密度极高。

**核心 insight:memory 系统的价值跟任务的 procedural novelty 正相关。任务越新、越 environment-specific,memory 越值钱。**

## 几个有意思的细节

**任务顺序有讲究。** 先做难题再做简单题,效果比先简单后难题好。因为从难题开始,agent 被迫建立完整的推理链,这些链在简单任务上直接复用。反过来从简单任务开始,学到的都是 shortcut,到难题上要重新长链。

**失败经验要小心。** 如果把失败的任务也存进 memory,不带整理的 baseline 性能会掉。因为下次遇到类似任务,检索回一个失败的经验当参考,反而被带偏。ReMem 通过 Refine 主动把这种失败 pattern 删掉或标注,保持 robust。

**Pruning 是自学的。** 不同 dataset 上 ReMem 自己学出来的删除率不同。GPQA 跨多个学科,task 之间不可迁移,删除率 36.8%。AIME 都是数学题,domain 单一,删除率才 10.8%。没人告诉它该删多少,它自己根据 task similarity 判断。

## 这篇 paper 没解决的问题

**Feedback 假设太强。** 整个系统依赖"这个任务做对了没"这个信号。benchmark 上能拿到,真实世界几乎拿不到。你让 agent 帮用户订机票,订完了没人告诉你"对不对"。这是 self-evolving memory 落地最大的坑。

**Token cost 没报。** ReMem 在 AlfWorld 上把 step 从 22.6 降到 11.5,看着效率提升一倍。但每个 step 内可能要多轮 Think + Refine 才 Act,总 token 消耗大概率是上升的。生产场景要算 cost per task,不是 step per task。

**Cold start。** 前面几个任务没有 memory 可用,跟 baseline 一样慢。要等攒够了才 fly。生产系统可能需要 prepopulate memory 做预热。

**Memory 还是文本。** 所有方法都是把经验存成文本然后做 semantic retrieval。文本不可微分,cosine similarity 是个粗糙信号,没有结构化的因果时间关系。未来可能要往 graph memory 或 program-based skill library 走。

## 一句话总结

这篇 paper 的核心 message:LLM agent 的 memory 不该只是"记事本",该是"经验库"——能随用随长,能自己整理,能把过去的推理套路抽象出来给未来任务用。ReMem 给了一个 working baseline,Evo-Memory 给了一个可比较的评测协议。这个方向我认为是 LLM agent 走向真实部署的必经路。

---

# Evo-Memory: 把 LLM Memory 从 "Recall" 升级到 "Experience Reuse" 的 Benchmark

## 1. 这篇 paper 在攻击什么问题

LLM agent 圈子里 memory 这个词被滥用了。绝大多数所谓 memory 系统——[MemGPT](https://arxiv.org/abs/2310.08560), [MemoryBank](https://arxiv.org/abs/2305.10250), [MEM0](https://arxiv.org/abs/2504.19413)——本质都是把对话历史或者外部知识压缩、索引、检索。这些系统回答 "上次用户说了什么" 这种问题很好用,但是当 agent 跨 session 遇到一个 *相似结构* 的新任务时,它们不会自动把过去 task 的 reasoning strategy 抽象出来复用。

作者把这两种 memory 区分得很清楚:

- **Conversational recall**: 检索过去的 *事实*。例如 "2x²+3x−1=0 的解是什么?"——上次解过就再吐一遍。
- **Experience reuse**: 检索过去的 *推理模式*。例如记住 "二次方程用求根公式",下次遇到新方程套同一策略。

这两件事的 representation 完全不同。recall 是 episodic buffer,reuse 是 procedural abstraction。现有 benchmark 比如 [StreamBench](https://proceedings.neurips.cc/paper_files/paper/2024/hash/StreamBench), [LongMemEval](https://arxiv.org/abs/2410.10813), [LifelongAgentBench](https://arxiv.org/abs/2505.11942) 主要测前者。Evo-Memory 填这个空缺。

## 2. Benchmark 设计:把静态 dataset 重排成 streaming trajectory

关键 trick 是把传统 benchmark 重构成 task stream τ = {(x₁,y₁), ..., (x_T,y_T)}。在这个 stream 里,早期 task 提供 *essential information or strategies* 给后续 task。Agent 每处理一个 task,都要做 search → synthesis → evolve 三步循环:

```
(x_t, M_t) →search→ R_t →synthesis→ ŷ_t →evolve→ M_{t+1}
```

这个 streaming 设计对 single-turn 数据集(MMLU-Pro, AIME, GPQA, ToolBench)同样适用——把同类题目排在一起,看模型能不能 *学会一类题的解法*。这个 idea 很聪明,等于把 in-context learning 的 "test-time scaling" 显式化成一个可测量的 process。

## 3. 形式化定义:统一所有 memory agent

作者定义一个 memory-augmented agent 是四元组 **(F, U, R, C)**:

- **F**: base LLM
- **U**: memory update pipeline
- **R**: retrieval module  
- **C**: contextual construction mechanism

### 3.1 Search 步骤

$$R_t = \mathrm{R}(M_t, x_t)$$

这里 $M_t$ 是 time $t$ 的 memory state(可能是一组 embedding、一组 structured entries、一组 workflow graph), $x_t$ 是当前输入。R 可以是 similarity search, index-based lookup, 或者 attention over stored embeddings。不同的 memory algorithm 在这一步的策略不同——SelfRAG 学一个 retrieval trigger,DC 直接全量 cheatsheet 查,A-MEM 用 graph 遍历。

### 3.2 Synthesis 步骤

$$\tilde{C}_t = \mathrm{C}(x_t, R_t)$$

$$\hat{y}_t = \mathrm{F}(\tilde{C}_t)$$

- $\tilde{C}_t$: 把检索回来的内容 $R_t$ 重组成适合当前输入 $x_t$ 的 working context
- $\hat{y}_t$: base LLM 基于这个 context 输出的预测

C 可以是 template formatting(Mem0, LangMem),可以是 selection([A-MEM](https://arxiv.org/abs/2502.12110)),可以是 merging([Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952))。

### 3.3 Evolve 步骤

$$m_t = h(x_t, \hat{y}_t, f_t)$$

$$M_{t+1} = \mathrm{U}(M_t, m_t)$$

- $m_t$: 新的 memory entry,由 current input $x_t$, prediction $\hat{y}_t$, feedback $f_t$ 三个组成
- $f_t$: 任务完成与否的信号(correctness, success/fail)。这个 feedback 是 test-time learning 的关键——它让 agent 知道这次 experience 是 positive 还是 negative sample
- $\mathrm{U}$: 不同算法的核心差异点。可以是 direct append(retrieval-based)、summarization/compression(long-term)、replacement(bounded-capacity)

这个四元组抽象的价值在于:它把 [RAG](https://arxiv.org/abs/2005.11401)、MemGPT、A-MEM、Dynamic Cheatsheet、[AWM](https://arxiv.org/abs/2409.07429) 这些看着完全不同的系统都装进同一个循环里比较。

## 4. ExpRAG: 极简 baseline

ExpRAG 是个 in-context learning 加 retrieval 的简单 baseline。Memory entry 模板是 $m_i = S(x_i, \hat{y}_i, f_i)$(S 是文本模板)。

检索:

$$R_t = \mathrm{Top\text{-}k}_{m_i \in M_t} \phi(x_t, m_i)$$

这里 $\phi$ 是 retrieval scoring function(论文里用 BGE-base-en-v1.5 cosine similarity,k=4)。

生成:

$$\hat{y}_t = \mathrm{F}(x_t, R_t)$$

更新:

$$M_{t+1} = M_t \cup \{(x_t, \hat{y}_t, f_t)\}$$

就这些。直接把 (input, prediction, feedback) 三元组塞进 memory,下次类似任务时检索出来当 in-context example。这个 baseline 在实验里居然能 beat 一堆更复杂的系统(比如 LangMem、AWM),很说明问题——task-level experience retrieval 这个事,简单粗暴先有效。

## 5. ReMem: 把 ReAct 的 action space 从 2 维扩到 3 维

ReMem 是这篇 paper 的主要 contribution。原版 [ReAct](https://arxiv.org/abs/2210.03629) 让 LLM 在 Think 和 Act 之间切换。ReMem 加了第三个 op: **Refine**。

### 5.1 Action space

$$a_t^n \in \{\text{Think, Act, Refine}\}$$

- **Think**: 产生 intermediate reasoning trace,分解 task,规划后续 action
- **Act**: 在 environment 里执行一个 action,或者输出对 user 可见的 response
- **Refine**: 对 memory 做 meta-reasoning——exploit useful experiences, prune noise, reorganize $M_t$

在每个 step $t$ 内,agent 可以多轮 Think 和 Refine,直到选了 Act 才结束这个 step。

### 5.2 MDP 形式化

状态: $s_t^n = (x_t, M_t, o_t^{1:n-1})$

- $t$: 当前是处理第几个 task(streaming index)
- $n$: 在当前 task 内已经执行了多少个 operation
- $x_t$: 当前 task input
- $M_t$: 当前 memory state
- $o_t^{1:n-1}$: 已经产生的 operation outputs(reasoning traces, actions, refine thoughts)

转移:

$$o_t^n = \mathrm{Agent}(x_t, M_t, a_t^n)$$

Action space: {Think, Act, Refine}

这构成一个 Markov decision process。整个 trajectory 是:

$$(x_1, \hat{y}_1, M_1) \to (x_2, \hat{y}_2, M_2) \to \cdots \to (x_T, \hat{y}_T, M_T)$$

### 5.3 为什么这个设计有道理

传统 ReAct 的 Think 是对 *当前任务* 推理,Refine 是对 *memory 本身* 推理。把这两件事分开之后,agent 可以做几件 ReAct 做不到的事:

1. 主动 prune memory——发现某条 experience 对当前 task 不仅无用还有害时,可以删
2. 主动 synthesize memory——把几条碎片 experience 合并成更抽象的策略
3. 主动 reorganize memory——重新索引已有 entries 提升未来检索质量

这本质上是把 *reflection*([Reflexion](https://arxiv.org/abs/2303.11366) 风格)从 episode 级别降到了 operation 级别,而且是 memory-aware 的 reflection。

### 5.4 Prompt 结构(附录 C)

Multi-turn 的 prompt template 长这样:

```
ENVIRONMENT INSTRUCTIONS
EXAMPLE DEMONSTRATIONS
RELEVANT EXPERIENCE FROM SIMILAR TASKS   ← 检索回来的 memory
YOUR CURRENT TASK
RECENT HISTORY
OUTPUT FORMAT:
  Format 1 - Prune experiences: 删除某条 memory(e.g., "1,3" or "2-4")
  Format 2 - Internal reasoning: Think
  Format 3 - Execute action: Act
```

注意 Format 1 是 ReMem 独有的——agent 可以输出 "1,3" 表示删除第 1、3 条 retrieved experience。这就是 Refine op 的 prompt-level 实现。

## 6. 实验设置

### 6.1 Datasets

10 个数据集分两类:

**Single-turn(7个评测项)**:
- [MMLU-Pro](https://arxiv.org/abs/2406.01574)(Economics, Engineering, Philosophy 三个子集)——multi-disciplinary reasoning
- [GPQA-Diamond](https://arxiv.org/abs/2311.12022)——graduate-level "Google-proof" Q&A
- [AIME-24, AIME-25](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)——Olympiad 数学,exact match
- [ToolBench](https://arxiv.org/abs/2305.15334)——API 调用

**Multi-turn(4个 embodied)**:
- [AlfWorld](https://arxiv.org/abs/2010.03768)——家庭指令跟随
- [BabyAI](https://arxiv.org/abs/1810.08254)——grounded navigation
- [ScienceWorld](https://arxiv.org/abs/2203.07540)——科学实验
- [PDDL](https://arxiv.org/abs/2312.00754)——symbolic planning

### 6.2 Methods(14种)

作者把方法分四类,用统一的 search-predict-evolve pipeline 隔离 memory design 的影响:

1. **No persistent memory**: ReAct, Amem
2. **Adaptive agentic memory**: SelfRAG, MemOS, Mem0, LangMem
3. **Procedural memory**: DC-Cumulative, DC-RetrievalSynthesis, AWM
4. **Proposed**: ExpRecent, ExpRAG, ReMem

LLM backbones: Gemini-2.5-Flash/Flash-Lite/Pro, Claude-3.5-Haiku, Claude-3.7-Sonnet。

Retriever 统一用 [BGE-base-en-v1.5](https://arxiv.org/abs/2309.07597),top-k=4。

### 6.3 Metrics

四个维度:
1. **Answer accuracy**——single-turn exact match
2. **Success rate / Progress rate**——multi-turn goal completion
3. **Step efficiency**——到达 goal 需要多少步
4. **Sequence robustness**——不同 task 顺序下性能稳定性

## 7. 结果分析

### 7.1 RQ1: 整体表现

**Table 1**(single-turn, Gemini-2.5-Flash):

| Method | AIME24 | AIME25 | GPQA | MMLU-Eco | MMLU-Eng | MMLU-Philo | ToolBench API/Acc | Avg |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.47 | 0.47 | 0.48 | 0.83 | 0.46 | 0.75 | 0.71/0.61 | 0.59 |
| ReAct | 0.30 | 0.27 | 0.05 | 0.64 | 0.16 | 0.54 | 0.64/0.57 | 0.37 |
| Mem0 | 0.50 | 0.47 | 0.45 | 0.83 | 0.46 | 0.74 | 0.71/0.61 | 0.59 |
| AWM | 0.50 | 0.37 | 0.49 | 0.79 | 0.43 | 0.72 | 0.71/0.59 | 0.56 |
| **ExpRAG** | 0.43 | 0.47 | 0.42 | 0.83 | 0.43 | 0.78 | **0.87/0.73** | 0.60 |
| **ReMem** | **0.60** | **0.53** | **0.51** | **0.85** | 0.46 | **0.79** | 0.85/0.71 | **0.65** |

几个值得琢磨的点:
- ReAct 在 single-turn 上居然比 Baseline 还差(Avg 0.37 vs 0.59)。原因是 ReAct 的 reasoning trace 在没有 memory 的情况下反而干扰了直接 answer generation
- LangMem 在 MMLU 上崩了(MMLU-Eco 0.79→0.77→LangMem 0.79,但 Engineering 0.39,Philosophy 0.71)。Memory 写得不好反而引入噪声
- AWM 在 AIME 上 0.03/0.03——workflow memory 对纯数学题完全没帮助,因为数学题解法很难 abstract 成 reusable workflow
- ToolBench 上 ExpRAG / ReMem 把 API accuracy 从 0.61 拉到 0.71/0.73。这是 experience reuse 最干净的 case——API 调用模式高度重复

**Table 2**(multi-turn, Claude-3.7-Sonnet):

| Method | AlfWorld S/P | BabyAI S/P | PDDL S/P | SciWorld S/P | Avg S/P |
|---|---|---|---|---|---|
| Baseline | 0.18/0.49 | 0.51/0.66 | 0.17/0.39 | 0.10/0.53 | 0.24/0.52 |
| ReAct | 0.51/0.75 | 0.57/0.72 | 0.75/0.91 | 0.44/0.77 | 0.57/0.79 |
| Mem0 | 0.51/0.74 | 0.48/0.66 | 0.65/0.84 | 0.37/0.76 | 0.50/0.75 |
| AWM | 0.49/0.73 | 0.53/0.68 | 0.60/0.82 | 0.34/0.74 | 0.49/0.74 |
| **ExpRAG** | 0.74/0.89 | 0.62/0.72 | 0.72/0.89 | 0.46/0.76 | 0.63/0.82 |
| **ReMem** | **0.92/0.96** | **0.73/0.83** | **0.83/0.95** | **0.62/0.89** | **0.78/0.91** |

这是 paper 最强的结果。AlfWorld success rate 从 baseline 0.18 到 ReMem 0.92,**5 倍提升**。ScienceWorld 从 0.10 到 0.62,**6 倍**。这种 magnitude 在 LLM agent 论文里很少见。

为什么 multi-turn 比 single-turn 提升大得多?我的理解:
- Multi-turn task 内部本身就有 trajectory 可学——AlfWorld "去厨房拿苹果"的 action sequence 模式是固定的
- Memory 里存的不只是答案,是 *plan*,可以 next task 直接 reuse
- Procedural knowledge 在 embodied environment 上抽象层次高且 transferable

### 7.2 RQ2: Task similarity 与 memory 改进的关系

Figure 4 显示 ReMem 相对 History baseline 的提升与 within-dataset task embedding cluster ratio 高度正相关:

- Gemini-2.5-Flash: Pearson **r = 0.717**
- Claude-3.7-Sonnet: Pearson **r = 0.563**

Task similarity 计算方式:对每个 dataset,用 retriever encoder 把所有 task 编码,计算每个 task embedding 到 dataset cluster center 的平均 cosine distance。距离越小,intra-dataset coherence 越高。

结果:PDDL / AlfWorld 这种 task structure 高度统一的 dataset 提升最大;AIME-25 / GPQA 这种题与题之间解法差异大的提升小。

**直觉**: experience reuse 的天花板由 task distribution 的 reusable structure 决定。如果一个 dataset 内部 task 之间互相根本不可迁移,任何 memory 系统都救不了。这对未来 benchmark 设计是个重要 hint。

### 7.3 RQ3: Easy → Hard vs Hard → Easy

Table 3 比较两种 task ordering:

| Direction | Method | AlfWorld S/P | SciWorld S/P | Avg S/P |
|---|---|---|---|---|
| Easy→Hard | ReMem | 0.91/0.96 | 0.63/0.88 | 0.77/0.92 |
| Hard→Easy | ReMem | 0.94/0.97 | 0.68/0.90 | **0.81/0.94** |

Hard→Easy 反而比 Easy→Hard 略好。直觉是:从难题开始,agent 被迫建立 *完整* 的 reasoning chain,这些 chain 在简单任务上直接 reusable;反过来从简单任务开始,agent 学到的是 "shortcut",到难题上要重新长链。

这个发现对 curriculum learning 是个反直觉的信号——但要注意这里 task 都是同一难度等级内排序,"Easy"/"Hard" 是 dataset 内部 split 的概念。

### 7.4 RQ4: 失败 experience 的影响

Table 4 测了把 failed task 也存进 memory 会怎样:

- Baseline 类方法在存失败 experience 后明显掉点
- ReMem 通过 Refine op 主动 prune/avoid failed patterns,保持 robust

**直觉**: naive memory accumulation 把 noise 当 signal。ReMem 的 Refine op 本质上是个 filter——它把 memory 里那些跟当前任务 *相似但 failed* 的 entry 删掉或标注,避免下次被检索回来当 in-context example 用。这有点像 RL 里的 negative sample 处理。

### 7.5 RQ5: 时间步累计曲线

Figure 6 显示 ReMem 在 4 个 embodied dataset 上的 cumulative success rate 都随 task index 单调上升,且斜率比 History baseline 陡。这说明 test-time learning 是 *真的在学*——处理得越多,后面越快越好。

### 7.6 Memory Pruning Analysis(Appendix B.2, Figure 7)

不同 dataset 的 pruning rate:
- GPQA: 36.8%(pruning rate)
- AIME-24: 17.5%
- AIME-25: 10.8%

Pruning rate 跟 task diversity 正相关。GPQA 涵盖工程、物理、化学等多个领域,memory 跨 domain 复用率低,所以 ReMem 主动 prune 掉跨 domain 的 entry。AIME 都是数学题,domain 单一,memory reuse 价值高,prune 少。

这个 pruning 是 ReMem 自学的——它在 Refine op 里输出 "1,3" 之类的指令删除某些 entry,没有任何 supervised pruning signal。

## 8. 我的几点 intuition 和 critique

### 8.1 这个工作真正解决的是 in-context learning 的 online 版本

ReMem 在我看来是把 ICL 拆成了 streaming 过程。传统 ICL 是一次性塞 k 个 demonstrations 进 prompt;ReMem 是动态从经验池里挑 k 个最相关的 demonstrations,而且这个池子本身随时间增长 + prune + reorganize。

从 first principle 看,这就是 *test-time scaling for agents* 的一个具象化方向。LLM 训练阶段的 compute scaling 已经饱和,推理阶段通过 memory evolution 让一个 frozen model 越用越强,这是另一条 promise 路径。

### 8.2 Refine op 是关键,但是 overhead 也是真的

ReMem 在 AlfWorld 上把 step 从 22.6 降到 11.5,但是每个 step 内可能要多轮 Think + Refine 才 Act。所以总 token 消耗大概率是上升的,不是下降的。这个 paper 没报告 total token cost。如果是生产场景,要算 cost per task 而非 step per task。

### 8.3 Memory representation 还是 too textual

所有方法——ExpRAG, ReMem, Mem0, AWM——本质上都是把 memory 存成文本然后做 semantic retrieval。这是 LLM 时代最方便的形式,但是:

- Text 不可微分,无法做 gradient-based memory update
- Cosine similarity 是个粗糙的 relevance 信号,跟 task usefulness 不完全对齐
- 文本 memory 没有结构化的 causal / temporal relations([Zep](https://arxiv.org/abs/2501.13956) 试图用 knowledge graph 但实验里没出现)

未来方向可能是 [Memory-R1](https://arxiv.org/abs/2508.19828) 那种 RL-trained memory manager,或者把 memory 写成 explicit program([Voyager](https://arxiv.org/abs/2305.16291) 风格的 skill library)。

### 8.4 Single-turn 任务上 memory 提升有限的真实原因

看 Table 1,ReMem 在 GPQA 上 0.51 vs Baseline 0.48,提升就 3 个点。为什么这么少?

我的 hypothesis: 这些 dataset 内部 task 之间的 surface similarity 低(题目本身不同),但 underlying reasoning 高度 *canonical*(数学解法、物理公式都是标准答案)。LLM 自己已经把这些 canonical reasoning 内化在 pretraining 里了。所以 memory retrieval 拉回来一个相似题,LLM 看到 "上次用了求根公式"——但它本来就会用求根公式,memory 的 marginal value 接近 0。

这跟 multi-turn embodied 完全不同。AlfWorld 的 "去 kitchen,找 apple,放 refrigerator" 的 action sequence 不是 pretraining 里能学到的,是 *environment-specific* 的 procedural knowledge。这种 knowledge 只能在 deployment 时学到,memory 的价值就大。

所以 memory 系统的价值密度跟 task 的 *novelty-of-procedure* 正相关。

### 8.5 Cold start 问题没解决

Figure 6 / 8 里 ReMem 和 History baseline 在 stream 开头几乎重合,因为 memory 还没攒起来。这个 cold start period 在生产系统里是个真实成本——前 N 个 task 没有加速,但是 N 个之后才开始 fly。

一个可能的解法是 prepopulate memory:从相关 task 的 demonstration 跑一遍 warm-up。或者 cross-dataset memory transfer——把 AlfWorld 上学到的 navigation workflow 迁移到 BabyAI。但论文没做这个。

### 8.6 Feedback signal 假设太强

论文里 $f_t$ 假设是 ground-truth correctness signal(success/fail)。这在 benchmark 上没问题,真实世界里几乎拿不到。生产场景下要么靠 verifier model(贵且不准),要么靠 environment implicit signal(reward shaping)。这是 self-evolving memory 落地最大的瓶颈,论文只是简单 acknowledge 一下。

### 8.7 跟 RL / Continual Learning 的关系没被充分讨论

这个 paper 本质上是在做 *online RL without gradient updates*。Memory + retrieval + refine 是个 policy improvement loop,只不过 improvement 发生在 prompt 层面而非 weight 层面。

跟经典 continual learning 的 catastrophic forgetting 问题对应:这里 memory prune 等于主动 forget。ReMem 通过 semantic pruning 避免 naive accumulation 的 noise pollution,本质上是 importance-weighted retention。

跟 RL 的 experience replay 对应:ExpRAG 就是 nearest-neighbor replay buffer。ReMem 多了一个 *on replay buffer 的 reasoning* step,类似 model-based RL 里 world model 对 replay 数据的 reprocessing。

如果作者把这个 paper 重新 frame 成 "prompt-level policy iteration",audience 会更大。现在 frame 在 "memory benchmark" 上,反而把 contribution 看窄了。

## 9. 关键参考资料

- [Evo-Memory 论文原文](https://arxiv.org/) (假设很快 release)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) - ReMem 的 base
- [Reflexion: Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) - reflection 的前身
- [Voyager: Open-Ended Embodied Agent](https://arxiv.org/abs/2305.16291) - skill library 的概念起源
- [A-MEM: Agentic Memory](https://arxiv.org/abs/2502.12110) - graph-based memory
- [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952) - test-time learning with adaptive memory
- [Agent Workflow Memory (AWM)](https://arxiv.org/abs/2409.07429) - procedural memory
- [MEM0: Production-Ready Long-term Memory](https://arxiv.org/abs/2504.19413)
- [MemOS: Memory Operating System](https://arxiv.org/abs/2505.22101)
- [Zep: Temporal Knowledge Graph](https://arxiv.org/abs/2501.13956)
- [Memory-R1: RL for Memory Management](https://arxiv.org/abs/2508.19828)
- [StreamBench](https://proceedings.neurips.cc/paper_files/paper/2024) - streaming benchmark
- [LongMemEval](https://arxiv.org/abs/2410.10813) - long-term memory eval
- [LifelongAgentBench](https://arxiv.org/abs/2505.11942) - lifelong learning benchmark
- [AgentBoard](https://arxiv.org/abs/2401.13178) - 多 turn embodied eval suite
- [AlfWorld](https://arxiv.org/abs/2010.03768)
- [BabyAI](https://arxiv.org/abs/1810.08254)
- [ScienceWorld](https://arxiv.org/abs/2203.07540)
- [MMLU-Pro](https://arxiv.org/abs/2406.01574)
- [GPQA](https://arxiv.org/abs/2311.12022)
- [BGE-M3 Embeddings](https://arxiv.org/abs/2309.07597)

## 10. 总结

这篇 paper 的核心 contribution 我归纳为三层:

1. **Conceptual**:把 LLM memory 从 "recall past facts" 重新定义成 "reuse past reasoning strategies",提出 test-time evolution 这个 evaluation dimension。
2. **Methodological**:ReMem 把 ReAct 的二元 action space 扩成三元 {Think, Act, Refine},让 agent 可以对 memory 本身做 meta-reasoning。
3. **Empirical**:在 embodied multi-turn 任务上 ReMem 给出 5-6 倍的 success rate 提升,在 single-turn 上提升温和(2-5 个点)。

Critique 角度:memory representation 太 textual、feedback signal 假设强、token cost 没报告、cold start 没解决、跟 RL/continual learning 的概念关联没讲透。

但是作为一个 *benchmark* paper,它最大的价值是把 memory 研究从 "存什么 / 怎么检索" 推进到 "怎么随时间 evolve 出可复用的 reasoning strategy"。这个方向我认为是 LLM agent 走向真实部署的必经路径——frozen model 不可能解决所有 corner case,deployment-time evolution 是 pragmatic 的答案。ReMem 给出了一个 working baseline,Evo-Memory 给出了一个可比较的评测协议,后续工作可以在这个 substrate 上迭代。
