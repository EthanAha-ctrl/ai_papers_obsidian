---
source_pdf: Evo-Memory.pdf
paper_sha256: a795dd58d78cdd75f15afd536cf19e5664a333b0fb02a0ab58d8ff3d3bd8f842
processed_at: '2026-08-04T05:56:34-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Evo-Memory 人话版

## 一句话总结

这篇 paper 说的事情很 simple：**现有 LLM memory 研究都在测 "你记不记得过去说了啥"，没人测 "你能不能从过去学到经验让未来做得更好"**。Evo-Memory 就是来填这个坑的 benchmark，顺便塞了两个 method（ExpRAG 和 ReMem）证明 experience reuse 这个方向真的 work。

## 1. 为什么这篇 paper 存在

你去翻所有 LLM memory 的 paper——MemGPT ([Packer et al., 2023](https://arxiv.org/abs/2310.08560))、MemoryBank ([Zhong et al., 2023](https://arxiv.org/abs/2305.10250))、LongMemEval ([Wu et al., 2024](https://arxiv.org/abs/2410.10813))——它们测的全是 *conversational recall*：用户上周说他狗叫 Max，这周你还能记得吗？这种 memory 本质上是 **fact retrieval**。

但真正的 agent 用例里，你要的是 **experience reuse**：昨天你在 AlfWorld 里花了 20 步才把 apple 放进 bag，今天再做一个类似 task，你应该 5 步搞定，因为你已经 *学到* 了 "先 go to kitchen 再 pick up 再 go to destination" 这个 workflow。这跟记 fact 完全是两回事。

Evo-Memory 的 contribution 就是把这件事 formalize 成一个 streaming evaluation：把 dataset 重排成 task sequence τ = {(x₁, y₁), ..., (x_T, y_T)}，让 early task 的 solution 给 later task 用。Agent 每做完一个 task 必须更新 memory M_t，下游 task 的表现直接反映 memory quality。

## 2. 三个核心概念，用最直觉的方式讲

### 2.1 Conversational Recall vs Experience Reuse

Figure 1 那个例子特别好：解方程 `2x² + 3x − 1 = 0`。

- **Conversational recall**：你过去解过这题，答案是 `x = (−3 ± √17) / 4`，现在再问，你 retrieve 出来——这是 fact lookup
- **Experience reuse**：你过去解过二次方程，学到了 "碰到 ax² + bx + c = 0 就用求根公式 x = (−b ± √(b²−4ac)) / 2a"——这是 strategy abstraction

两者本质区别：recall 是 retrieve **specific instance**，reuse 是 retrieve **generalizable pattern**。Evo-Memory 测后者。

### 2.2 Search-Synthesize-Evolve 循环

所有 memory-augmented agent 都能 abstract 成这个 loop：

```
Input x_t → Search memory → Synthesize context → Generate ŷ_t → Evolve memory
```

形式化：

$$R_t = \mathrm{R}(M_t, x_t)$$
$$\tilde{C}_t = \mathbf{C}(x_t, R_t), \quad \hat{y}_t = \mathrm{F}(\tilde{C}_t)$$
$$m_t = h(x_t, \hat{y}_t, f_t), \quad M_{t+1} = \mathrm{U}(M_t, m_t)$$

变量解释：
- **M_t**：第 t 步的 memory state（一个 set / list / structured store）
- **R_t**：retrieved memory entries（top-k 个）
- **x_t**：当前 task input
- **\tilde{C}_t**：合成的 working context（prompt）
- **ŷ_t**：LLM F 生成的 prediction
- **f_t**：feedback signal（对错、成功失败）
- **m_t**：本步产生的 memory entry
- **U(·)**：memory update 算子

不同 method 就是在 R / C / U 三个位置用不同实现。RAG 用 cosine retrieval + append；Mem0 用 LLM-as-judge decide add/update/delete；ReMem 在 loop 里多塞一个 Refine action。

### 2.3 Test-time Evolution

这是 paper 的关键词。Agent 在 deployment 期间 *持续* 更新 memory，每个 task 都让 memory 更聪明一点。这跟 test-time training (TTT, [Sun et al., 2020](https://arxiv.org/abs/1909.13255)) 在 spirit 上类似——都是 "test 时改东西"——但 TTT 改的是 weight，Evo-Memory 改的是 memory store。好处：不用 backprop，不用 gradient，pure inference-time。

## 3. ExpRAG：简单到夸张的 baseline

ExpRAG 的 idea 一句话说完：**把过去做过的 task (x_i, ŷ_i, f_i) 全存下来，新 task 来了用 embedding similarity retrieve top-4 个最相似的，塞 prompt 里做 ICL**。

就这么简单。公式：

$$R_t = \mathrm{Top\text{-}k}_{m_i \in M_t}\, \phi(x_t, m_i)$$
$$\hat{y}_t = \mathrm{F}(x_t, R_t)$$
$$M_{t+1} = M_t \cup \{(x_t, \hat{y}_t, f_t)\}$$

- **φ(x_t, m_i)**：cosine similarity on bge embedding
- **Top-k**：k=4
- **F**：base LLM，input 是 (x_t, R_t) 拼接

**就这么个东西，在 Table 1 里 beat 了 Mem0、SelfRAG、AWM 几乎所有 baseline**。Gemini-2.5-Flash 上 ExpRAG 平均 0.60，Mem0 0.59，SelfRAG 0.59，AWM 0.56。Claude 3.7-Sonnet 上 ExpRAG 0.59 vs Mem0 0.55。

这个结果非常说明问题：**当前的 adaptive memory method 大部分 over-engineered**。它们搞了复杂的 structured memory、LLM-as-judge、workflow induction，结果还不如 "存 raw experience + embedding retrieval + ICL" 这个最 dumb 的方案。说明 task-level experience reuse 这个 axis 在当前 research 里被严重 underexplored，简单方法就能拿大头的 gain。

## 4. ReMem：在 ReAct 里塞一个 Refine action

ReAct ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)) 的 action space 是 {Think, Act}：你要么内部推理，要么执行 action。ReMem 扩成 {Think, Act, Refine}：

$$a_t^n \in \{\text{Think, Act, Refine}\}$$
$$o_t^n = \mathrm{Agent}(x_t, M_t, a_t^n)$$

- **a_t^n**：第 t 个 task 第 n 次操作
- **o_t^n**：这次操作的 output

State 是 Markov 的：

$$s_t^n = (x_t, M_t, o_t^{1:n-1})$$

终止条件：一旦选 Act，step 结束。所以一个 step 内可以有任意轮 Think + Refine 交替，最后才 Act。

**Refine 具体干啥**？看 Appendix C 的 prompt，它让 agent 输出一个 memory entry 的 index list 要删掉（比如 "1,3" 或 "2-4"），原因是冗余、不相关、或者误导。本质就是 **agent 主动 prune 自己的 memory**。

这个 design 的 intuition 是：naive append-only memory（ExpRAG 那种）会 accumulate noise——失败 trajectory、误导性的 reasoning、redundant entries。光 retrieve 不够，你得 *清理* memory。但清理这个动作本身需要 reasoning——所以把它做成 agent 的一个 action，让 LLM 自己决定保留什么删除什么。

**跟 Reflexion ([Shinn et al., 2023](https://arxiv.org/abs/2303.11366)) 的区别**：Reflexion 的 reflection 作用在 trajectory 上（"刚才那步我走错了，下次注意"），ReMem 的 Refine 作用在 memory store 上（"memory 里第 3 条 experience 是误导的，删掉"）。一个是 verbal RL over actions，一个是 verbal management over memory entries。

**跟 A-Mem ([Xu et al., 2025](https://arxiv.org/abs/2502.12110)) 的区别**：A-Mem 把 memory 操作做成独立 agent（memory 自己有 read/write/organize 能力），ReMem 把 memory refine 内嵌进主 agent 的 decision loop。A-Mem 是 "memory as separate module"，ReMem 是 "memory operation as first-class action"。

**跟 MEM1 ([Zhou et al., 2025](https://arxiv.org/abs/2506.15841)) 的区别**：MEM1 用 RL 训练 memory read/write policy，memory 是 latent hidden state；ReMem 用 prompting 实现，memory 是 explicit text buffer。MEM1 是 learned policy，ReMem 是 prompted policy。

## 5. 实验结果的人话版

### 5.1 Single-turn task（Table 1）：提升 moderate

在 AIME-24/25、GPQA、MMLU-Pro、ToolBench 这些 single-turn reasoning benchmark 上：

- Gemini-2.5-Flash：ReMem 平均 0.65 vs Baseline 0.59 vs ReAct 0.37
- Claude 3.7-Sonnet：ReMem 0.58 vs Baseline 0.54 vs ReAct 0.54

提升大概 3-5 个点，有但不大。**原因直觉**：single-turn task 的 reasoning complexity 有限，ICL 本身就能 handle 大部分，memory reuse 的边际收益小。

### 5.2 Multi-turn task（Table 2）：提升巨大

这才是 Evo-Memory 的主战场。Claude 3.7-Sonnet 上：

| Dataset | Baseline S/P | ReMem S/P | Gain |
|---------|--------------|-----------|------|
| AlfWorld | 0.18 / 0.49 | 0.92 / 0.96 | **+74 / +47** |
| BabyAI | 0.51 / 0.66 | 0.73 / 0.83 | +22 / +17 |
| PDDL | 0.17 / 0.39 | 0.83 / 0.95 | **+66 / +56** |
| ScienceWorld | 0.10 / 0.53 | 0.62 / 0.89 | **+52 / +36** |

PDDL 从 essentially random 推到 0.95 progress，这是巨大的 jump。**原因直觉**：multi-turn long-horizon task 里，agent 需要 *累积 procedural knowledge*——怎么 plan、怎么从 failure 改 workflow。这种 knowledge 单次 ICL 抓不到，必须靠 memory 跨 task 累积。task horizon 越长，memory reuse 的 ROI 越高。

### 5.3 Task similarity 与 improvement 的相关性（Figure 4）

论文用 retriever embedding 计算 intra-dataset task similarity：

$$\text{sim}(D) = \frac{1}{|D|}\sum_{x \in D} \cos(\text{emb}(x), \mu_D)$$

- **μ_D**：dataset D 的 embedding cluster center
- 距离越小 = task 越同质

ReMem improvement 和 task similarity 的 Pearson correlation：
- Gemini-2.5-Flash：**r = 0.717**
- Claude 3.7-Sonnet：**r = 0.563**

**强正相关**。PDDL / AlfWorld 这种 task structure 重复性高的 dataset，memory reuse 收益巨大；AIME-25 / GPQA 这种 task diversity 高的 dataset，收益 marginal。

**直觉**：ReMem 本质还是 *in-distribution experience retrieval* 系统。它在 PDDL 上 work 是因为 PDDL 的 plan template 可重用；它在 AIME 上不 work 是因为每题 trick 不同。这暗示一个方向——**memory 的 abstraction level 需要自适应**。homogeneous task 用 low-level workflow memory 就够；heterogeneous task 需要 high-level strategy abstraction。ReMem 现在用同一个 template 处理所有，这里有 mismatch。

### 5.4 Task sequence 顺序：Hard→Easy 居然比 Easy→Hard 好（Table 3）

Claude 3.7-Sonnet：
- Base：AlfWorld 0.50/0.73, ScienceWorld 0.32/0.74
- Easy→Hard：ReMem 0.91/0.96
- Hard→Easy：ReMem 0.94/0.97

**Hard→Easy 略好**。这反直觉——curriculum learning 通常主张 Easy→Hard。解释：Hard task 的 failure 包含更多 *negative examples*，agent 通过 Refine 学到 "什么不能做"，这种 negative knowledge 在 Easy task 上 transfer 更稳。Karpathy 你之前讲 "learning from failure" 就是这个道理，跟 Reflexion 的 spirit 也一致。

### 5.5 Feedback type：失败的 experience 会污染 memory（Table 4）

当 memory 同时存 success 和 failure experience 时，baseline 方法明显退化——failure trajectory 进 memory 干扰 retrieval。ReMem 保持 robust——靠 Refine 操作主动 prune 失败 trajectory 的 misleading 部分，只保留可学习的 negative lesson。

**关键 implication**：naive RAG-style memory（ExpRAG）在 failure 进入后会退化；只有带 Refine 的 ReMem 能稳定 handle failure。这说明 memory management policy 在 long-horizon agent 里是 *必需的*。

### 5.6 Step efficiency：ReMem 用更少 step 完成任务（Figure 5）

AlfWorld 上平均 step 从 22.6 降到 11.5，**约 50% 减少**。memory 不仅提升 success rate，还让 agent 更 economical——把 "trial-and-error" 转化成 "informed action"。这是 procedural memory 的本质。

### 5.7 Cumulative performance（Figure 6）

ReMem 在 4 个 multi-turn environment 上的 cumulative success rate 曲线比 History baseline 上升得快且稳定。**monotonic continual improvement**——这是 test-time learning 应该有的 signature，early burst 然后 decay 就不对了。

## 6. 你应该 build 的 intuition

### 6.1 Memory 是 test-time compute 的另一个 axis

ReMem 的 Refine operation 本质是在 spend additional test-time compute on memory organization。跟 Snell et al. ([2024](https://arxiv.org/abs/2408.03314)) 讲的 test-time compute scaling 同向，但 axis 不同——不是 spend on solution search，是 spend on knowledge organization。这是 orthogonal 的 test-time scaling 方向。

### 6.2 Streaming evaluation 是 test-time learning 的正确 paradigm

Evo-Memory 真正的 contribution 是 **evaluation paradigm**。把 static dataset 变成 streaming task sequence，evaluation 从 pointwise accuracy 升级到 trajectory-level continual improvement。这跟你讲 Software 2.0 的延伸很自然——dataset 不再是 training data，是 experience stream。

### 6.3 Memory 作为 POMDP 的 belief state

ReMem 的 (state, action, transition)：
- State: s_t^n = (x_t, M_t, o_t^{1:n-1})
- Action: {Think, Act, Refine}
- Reward: f_t = task correctness

这是 prompting-based MDP，结构上是 POMDP（M_t 是 partial observation over 全部 history）。**Memory 就是 POMDP 里的 belief state**。如果要 RL train 这个 policy，就是 Memory-R1 ([Yan et al., 2025](https://arxiv.org/abs/2508.19828)) 的方向。Evo-Memory 给这类工作提供了 standardized eval suite。

### 6.4 简单方法 beat 复杂方法这个信号

ExpRAG 这个 dumb baseline beat 了 Mem0、SelfRAG、AWM 这些复杂方法。这个信号很重要：**当前的 adaptive memory method 大部分 over-engineered**。structured memory、LLM-as-judge、workflow induction 这些复杂机制，在 task-level experience reuse 这个 axis 上还不如 raw experience + embedding retrieval + ICL。

这暗示一个研究方向：先把简单方法做到极致（better retrieval、better experience encoding、better ICL prompt），再加 complexity。现在很多 method 在简单方法还没 optimize 的情况下就堆 complexity，导致 baseline 都 beat 不了。

### 6.5 Prompting-based 的 ceiling

ReMem 全靠 prompting 实现 Refine，这意味着：
1. Refine quality 依赖 LLM 的 self-reflection 能力（小模型会崩）
2. Refine cost 是额外 LLM call（per-step cost 高）
3. Pruning 决策可能 sub-optimal（错 prune critical experience 会损害下游）

把 Refine policy 用 RL 训练应该能推过这个 ceiling。Memory-R1 已经开始做这件事，Evo-Memory 给这类工作提供了 perfect eval suite。

## 7. 跟你最近思考的几个轴的 connection

### 7.1 跟 TTT (Test-Time Training) 的关系

TTT ([Sun et al., 2020](https://arxiv.org/abs/1909.13255)) 改 weight，Evo-Memory 改 memory store。两者都是 test-time 改东西，但 TTT 需要 backprop（expensive），Evo-Memory 只需要 retrieval + append + LLM-as-judge（cheap）。对于 deployment 场景，Evo-Memory 的 approach 更 practical。

### 7.2 跟 o1 / R1 style reasoning 的关系

o1 / DeepSeek-R1 style 的 long CoT 是 spend test-time compute on *solution search*。ReMem 是 spend test-time compute on *memory organization*。两者可以 stack——agent 可以在 long CoT 里穿插 Refine 操作，一边 search solution 一边 update memory。

### 7.3 跟 skill library 的关系

Voyager ([Wang et al., 2023](https://arxiv.org/abs/2305.16291)) 的 skill library 是 pre-abstracted skill，ReMem 的 memory 是 raw experience。两者中间有个 spectrum——memory 的 abstraction level 可以从 raw trajectory → summarized experience → workflow → skill → strategy 逐层升高。ReMem 现在在 summarized experience 这层，未来可以往更高 abstraction 推。

### 7.4 跟 in-context learning 的关系

ExpRAG 本质是 ICL over past experience。这跟你讲的 "context window is all you need" 派系有 connection——但 Evo-Memory 证明：context window 不够时，external memory + retrieval 能 extend ICL 的有效 horizon。这是 ICL 的 streaming 版本。

## 8. Reference

核心 method paper：
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [Voyager](https://arxiv.org/abs/2305.16291)
- [Self-Refine](https://arxiv.org/abs/2303.17651)
- [Self-RAG](https://openreview.net/forum?id=hSyW5go0v8)

Memory system paper：
- [MemGPT](https://arxiv.org/abs/2310.08560)
- [Mem0](https://arxiv.org/abs/2504.19413)
- [A-Mem](https://arxiv.org/abs/2502.12110)
- [MEM1](https://arxiv.org/abs/2506.15841)
- [Memory-R1](https://arxiv.org/abs/2508.19828)
- [MemOS](https://arxiv.org/abs/2505.22101)
- [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952)
- [Agent Workflow Memory](https://arxiv.org/abs/2409.07429)
- [Zep](https://arxiv.org/abs/2501.13956)

Benchmark paper：
- [StreamBench (NeurIPS 2024)](https://arxiv.org/abs/2401.07643)
- [LongMemEval](https://arxiv.org/abs/2410.10813)
- [LifelongAgentBench](https://arxiv.org/abs/2505.11942)
- [AgentBoard](https://arxiv.org/abs/2401.13178)

Test-time learning：
- [TTT (Sun et al., 2020)](https://arxiv.org/abs/1909.13255)
- [TENT (Wang et al., ICLR 2021)](https://arxiv.org/abs/2006.10926)
- [MEMO (Zhang et al., NeurIPS 2023)](https://arxiv.org/abs/2310.13331)
- [Test-time Compute Scaling (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)

Dataset：
- [MMLU-Pro](https://arxiv.org/abs/2406.01574)
- [GPQA](https://arxiv.org/abs/2311.12022)
- [ToolBench](https://arxiv.org/abs/2305.15334)
- [AlfWorld](https://arxiv.org/abs/2010.03768)
- [BabyAI](https://arxiv.org/abs/1810.08272)
- [ScienceWorld](https://arxiv.org/abs/2205.07559)
- [PDDLBench](https://arxiv.org/abs/2312.00754)

---

**Bottom line**：Evo-Memory 告诉我们三件事——（1）multi-turn long-horizon task 上，self-evolving memory 是 game-changer（PDDL 0.17 → 0.83）；（2）简单 ExpRAG 就能 beat 大部分复杂方法，说明 task-level experience reuse 被 underinvested；（3）Memory 和 reasoning 应该 co-evolve，Refine operation 是处理 failure experience 的关键。对于你在 build 的 test-time learning intuition，memory 作为 explicit、agent-operated object 是比 in-context recall 更 scalable 的 test-time computation carrier。

---

# Evo-Memory：一个面向 LLM Agent Test-time Learning 的 Self-evolving Memory Benchmark

## 1. 论文要解决的真正问题：Conversational Recall ≠ Experience Reuse

这篇 paper 的核心 motivation 来自一个非常 sharp 的区分：

- **Conversational recall**：从 dialogue history 里 passively 检索 "过去说过什么"（比如 recall 出 `2x² + 3x − 1 = 0` 的解）
- **Experience reuse**：从过去 trajectory 里抽象出 *reasoning strategy*（比如 "碰到二次方程就用求根公式"），然后把 strategy transfer 到未来同分布或 near-distribution 的 task 上

Karpathy 你应该一眼就看出这其实就是 **test-time scaling 的两种范式之争**：第一种是 retrieval-augmented in-context recall（很 o1-preview 的 "lookup 思维"），第二种是真正意义上的 *test-time learning*（更接近 TTT / TTSL 的 spirit）。所有现有 benchmark（LongMemEval [Wu et al., 2024](https://arxiv.org/abs/2410.10813), StreamBench [Wu et al., NeurIPS 2024](https://arxiv.org/abs/2401.07643), LifelongAgentBench [Zheng et al., 2025](https://arxiv.org/abs/2505.11942)）几乎都掉在第一种坑里——它们测的是 "你能不能记住过去 fact"，而不是 "你能不能用过去 trajectory 改进当前 policy"。Evo-Memory 想把 evaluation 推到第二种。

**Evo-Memory 的核心设计：把静态 dataset 重排成 streaming task stream** τ = {(x₁, y₁), …, (x_T, y_T)}，使得 early task 的 solution / strategy 是 later task 的 prerequisite。Agent 每解完一个 task，memory state M_t 都必须 update，下游 task 才能从中受益。这就把 evaluation 从 *pointwise accuracy* 升级到 *trajectory-level continual improvement*。

## 2. 统一形式的 Memory-augmented Agent 形式化

论文把任何 memory-augmented agent 抽象成四元组 (F, U, R, C)：

- **F**：base LLM（生成器）
- **R**：retrieval module
- **C**：context construction（把 retrieved content 拼到 prompt 里）
- **U**：memory update pipeline

在 streaming 设定下，t 时刻 agent 接收 x_t，维护 evolving memory M_t，执行三步循环：

### Search
$$R_t = \mathrm{R}(M_t, x_t)$$

其中 R 可以是 similarity search（cosine / dot product on embedding）、index-based lookup、或者 attention over stored embeddings。论文统一用 `BAAI/bge-base-en-v1.5` ([Chen et al., 2023](https://arxiv.org/abs/2309.07597)) 作为 retriever，top-k=4，保证 baseline 之间 retrieval budget 公平。

### Synthesis
$$\tilde{C}_t = \mathbf{C}(x_t, R_t), \quad \hat{y}_t = \mathrm{F}(\tilde{C}_t)$$

- **C_t**：contextualized prompt
- **R_t**：retrieved memory entries（一个 set/list）
- **x_t**：当前 task input
- **\tilde{C}_t**：合成后的 working context（structured prompt / selected key items / merged summary）
- **ŷ_t**：当前 step 预测

这一步在不同 method 里有不同 instantiation：
- ExpRAG / Mem0 / Self-RAG：直接拼接 retrieved text 到 prompt
- Dynamic Cheatsheet：把所有 retrieved cheatsheet 合并成一个 summary
- AWM (Agent Workflow Memory [Wang et al., 2024](https://arxiv.org/abs/2409.07429))：把 workflow schema 注入 prompt
- ReMem：Refine 操作可以在这里动态重组 memory，相当于 C_t 在多轮 think/refine 后才 freeze

### Evolve
$$m_t = h(x_t, \hat{y}_t, f_t), \quad M_{t+1} = \mathrm{U}(M_t, m_t)$$

- **m_t**：当前 step 产生的 memory entry（experience）
- **h(·)**：experience 编码函数（template / summarization / structured extraction）
- **f_t**：feedback signal（task correctness / success / progress rate）
- **U(·)**：memory update算子——可以是 append（retrieval-based）、summarize/compress（long-term）、replace（bounded capacity）、或者 prune-and-merge（ReMem）

整个 trajectory 是：

$$(x_1, \hat{y}_1, M_1) \rightarrow (x_2, \hat{y}_2, M_2) \rightarrow \cdots \rightarrow (x_T, \hat{y}_T, M_T)$$

这个 unification 把 RAG、MEM0、Dynamic Cheatsheet、AWM、Reflexion 全部塞进同一个 search-predict-evolve loop，唯一不同的就是 R / C / U 三个算子的实现。**这是个相当 elegant 的 abstraction**，Karpathy 你可能会觉得它太抽象——但好处是所有 baseline 都跑同一个 harness，差异完全归因于 memory design。

## 3. ExpRAG：一个最简 experience reuse baseline

ExpRAG = Experience Retrieval + Aggregation，是 retrieval-augmented ICL 的 task-level 版本。

每个 memory entry 编码成一个结构化文本：

$$m_i = S(x_i, \hat{y}_i, f_i)$$

其中 **S(·)** 是一个固定 template（包含 task description, solution, success/failure tag）。

检索 top-k：

$$R_t = \mathrm{Top\text{-}k}_{m_i \in M_t}\, \phi(x_t, m_i)$$

- **φ(x_t, m_i)**：retrieval score，论文用 cosine similarity on bge embedding
- **Top-k**：取 k=4 个最相似 experience

生成：
$$\hat{y}_t = \mathrm{F}(x_t, R_t)$$

注意这里 F 的 input 是 **(x_t, R_t)** 的拼接，没有 C_t 这个独立 module——直接 ICL 风格。

更新：
$$M_{t+1} = M_t \cup \{(x_t, \hat{y}_t, f_t)\}$$

直接 append，没有任何压缩、merge、reflection。这是最 minimal viable 的 experience reuse。

**为什么这个 baseline 重要**：它告诉我们一个 lower bound——"如果只是简单 ICL over 过去 trajectory，能到什么水平"。在 Table 1 里，ExpRAG 在 Gemini-2.5-Flash 上 AIME24/25 拿到 0.43/0.47，GPQA 0.42，ToolBench API/Acc 0.87/0.73；在 Claude 3.7 Sonnet 上 GPQA 0.70、MMLU-Pro(Eng.) 0.67、ToolBench 0.88/0.72。**ExpRAG 在所有 dataset 上都超过 Mem0 / SelfRAG / AWM**，这是相当 surprising 的结果——说明 "task-level experience reuse" 在当前 adaptive memory method 里被严重低估。

## 4. ReMem：Think-Act-Refine 三元操作

ReMem 是论文主推的 method。关键 insight：**把 ReAct 的 binary action space {Think, Act} 扩展成 ternary {Think, Act, Refine}**，让 agent 能在推理过程中主动 reorganize 自己的 memory。

在 step t，第 n 次 operation，agent 选择：

$$a_t^n \in \{\text{Think, Act, Refine}\}$$

执行后输出：

$$o_t^n = \mathrm{Agent}(x_t, M_t, a_t^n)$$

- **a_t^n**：第 t 个 task 的第 n 次操作（Think / Act / Refine）
- **o_t^n**：operation 产生的 output（reasoning trace / environment action / memory refinement thought）

**状态空间**是 Markovian 的：

$$s_t^n = (x_t, M_t, o_t^{1:n-1})$$

- **s_t^n**：当前 state
- **x_t**：当前 task input
- **M_t**：当前 memory
- **o_t^{1:n-1}**：本 step 已经生成的 reasoning/action/refine history

终止条件：**一旦选了 Act，step 就结束**。所以一个 step 内可以有 *任意多轮* Think 和 Refine 交替，但只有一次 Act。

三个 operation 的角色：

| Operation | 作用 | 类比 |
|-----------|------|------|
| **Think** | 内部 reasoning，分解 task，规划下一步 | ReAct 的 "Thought" |
| **Act** | 在 environment 执行 action 或给 user final answer | ReAct 的 "Action" |
| **Refine** | meta-reasoning over M_t：prune noise、merge redundant、reorganize structure | **ReMem 独有**——类似 Reflexion [Shinn et al., 2023](https://arxiv.org/abs/2303.11366) 但作用在 memory 上而非 trajectory 上 |

**Refine 的具体形式**（从 Appendix C 的 prompt 看）：
1. Prune: agent 输出一个 list of memory entry index 要删除（"1,3" 或 "2-4"），原因是冗余/不相关/误导
2. Internal reasoning: 在 Think 模式下评估哪些 experience 有用
3. Execute action: 把 refined 后的 M_t 作为 context 给 Act

这种设计把 memory 从 "passive context" 变成 "active, learned object"——agent 不仅 read memory，还在 read 之后 *修改* memory，再 read。这是一种很 DeepMind 风格的 test-time computation——把 thinking budget 部分用在 "组织自己的知识库" 上。

**与 MEM1 [Zhou et al., 2025](https://arxiv.org/abs/2506.15841) 的关系**：MEM1 也是 RL-trained memory+reasoning synergy，但 MEM1 是把 memory 作为 hidden state 的 functional form，ReMem 是把 memory 作为 explicit text buffer + agent 操作。两条路线对应了 "memory-as-weights vs memory-as-text" 这个 Karpathy 你一直在思考的轴。

**与 A-Mem [Xu et al., 2025](https://arxiv.org/abs/2502.12110) 的关系**：A-Mem 也是 agentic memory（memory 自己能 read/write/organize），但 A-Mem 把 memory operation 作为独立 agent，ReMem 把 memory refine 内嵌进同一个 decision loop。ReMem 更像 "memory operation as first-class action"。

**与 Memory-R1 [Yan et al., 2025](https://arxiv.org/abs/2508.19828) 的关系**：Memory-R1 用 RL 优化 memory management policy，ReMem 是 prompting-based。但两者的 action space abstraction 是同一类思想——memory management 是可学习的 policy。

## 5. 实验设置深度拆解

### 5.1 Datasets

**Single-turn reasoning/QA（5 个）**：
- **AIME-24/25** ([HuggingFaceH4](https://huggingface.co/datasets/HuggingFaceH4/aime_2024))：奥赛级数学，exact-match evaluation，符号推理
- **GPQA-Diamond** ([Rein et al., 2024](https://arxiv.org/abs/2311.12022))：研究生级 "Google-proof" 物理/科学问答
- **MMLU-Pro** ([Zheng et al., 2024](https://arxiv.org/abs/2406.01574))：MMLU 的 robust 版本，覆盖 Economics / Engineering / Philosophy 三个子集
- **ToolBench** ([Patil et al., 2023](https://arxiv.org/abs/2305.15334))：API 调用，测 tool-use grounding

**Multi-turn goal-oriented（5 个，来自 AgentBoard [Zhuang et al., 2024](https://arxiv.org/abs/2401.13178)）**：
- **AlfWorld** ([Shridhar et al., 2021](https://arxiv.org/abs/2010.03768))：text-based household instruction following
- **BabyAI** ([Chevalier-Boisvert et al., 2019](https://arxiv.org/abs/1810.08272))：grounded language navigation + compositional reasoning
- **ScienceWorld** ([Wang et al., 2022](https://arxiv.org/abs/2205.07559))：开放式科学实验
- **Jericho** ([Hausknecht et al., 2020](https://arxiv.org/abs/1908.11539))：text-based game
- **PDDL** ([Yang et al., 2023](https://arxiv.org/abs/2312.00754))：symbolic planning

### 5.2 LLM Backbones

- **Gemini 2.5** series ([Comanici et al., 2025](https://arxiv.org/abs/2507.06261))：Flash, Flash-Lite, Pro
- **Claude** family ([Anthropic](https://www.anthropic.com/news/claude-4))：3.5-Haiku, 3.7-Sonnet

### 5.3 Evaluation Metrics（四个维度）

1. **Answer accuracy**：single-turn correctness
2. **Success rate (S) + Progress rate (P)**：multi-turn goal completion
3. **Step efficiency**：完成 goal 需要的 step 数
4. **Sequence robustness**：不同 task order下稳定性

### 5.4 对比方法（四类，14 个 method）

| 类别 | Method | 特点 |
|------|--------|------|
| **No persistent memory** | ReAct, Amem | 短期 context 或轻量 cache |
| **Adaptive agentic memory** | SelfRAG, MemOS, Mem0, LangMem | dynamic retrieval + continual update |
| **Procedural memory** | DC-Cu, DC-RS, AWM | reusable workflows |
| **Proposed** | ExpRecent, ExpRAG, **ReMem** | reasoning + action + memory refine |

## 6. 关键实验结果深度解读

### 6.1 RQ1：整体性能（Table 1, 2）

**Single-turn（Table 1）**：

Gemini-2.5-Flash 上，ReMem 平均 0.65，超过 ReAct 的 0.37、Mem0 的 0.59、AWM 的 0.56。Claude 3.7-Sonnet 上 ReMem 0.58，超过 ReAct 0.54、Mem0 0.55。**ReMem 提升幅度在 single-turn 上 moderate（~3-5 个点）**，但在 multi-turn 上巨大。

**Multi-turn（Table 2）——这才是 Evo-Memory 真正发力的地方**：

Gemini-2.5-Flash：
- AlfWorld：ReMem S=0.66/P=0.81 vs Baseline 0.12/0.34（**+54 points！**）
- PDDL：ReMem 0.22/0.33 vs Baseline 0.12/0.20
- ScienceWorld：ReMem 0.58/0.81 vs Baseline 0.24/0.59
- 平均 S/P：ReMem 0.50/0.64 vs Baseline 0.27/0.46

Claude 3.7-Sonnet 上更夸张：
- AlfWorld：ReMem 0.92/0.96 vs Baseline 0.18/0.49
- BabyAI：ReMem 0.73/0.83 vs Baseline 0.51/0.66
- PDDL：ReMem 0.83/0.95 vs Baseline 0.17/0.39（**+66 points progress！**）
- ScienceWorld：ReMem 0.62/0.89 vs Baseline 0.10/0.53

**核心 insight：task horizon 越长，continual adaptation 的价值越大。** 在 PDDL 这种需要 long-horizon planning 的环境，ReMem 把 baseline 从 essentially random 推到 0.95 progress。这非常符合 intuition：单步 reasoning task 里，ICL 已经能 in-context 解决大部分；但 multi-turn task 里，"如何 plan, 如何从 failure 中改 workflow" 是需要 *累积* 的，single-turn ICL 抓不到这种 procedural knowledge。

### 6.2 RQ2：Memory improvement 与 task similarity 的相关性（Figure 4）

论文用 retriever embedding 计算 intra-dataset task similarity：

$$\text{sim}(D) = \frac{1}{|D|}\sum_{x \in D} \cos(\text{emb}(x), \mu_D)$$

其中 μ_D 是 dataset cluster center。距离越小，coherence 越高。

ReMem 的 improvement 和 task similarity 的 Pearson correlation：
- Gemini-2.5-Flash：r = 0.717
- Claude 3.7-Sonnet：r = 0.563

**强正相关**——说明 experience reuse 的有效性高度依赖 task distribution 的同质性。PDDL / AlfWorld 这种 task structure 重复性高的 dataset，memory reuse 收益巨大；AIME-25 / GPQA 这种 task diversity 高的 dataset，memory reuse 收益 marginal。这非常符合 transfer learning 的经典规律——"in-distribution transfer is easy, out-of-distribution transfer is hard"。

**对这个结果的 deep read**：ReMem 在本质上还是个 *in-distribution experience retrieval* 系统。它在 PDDL 上 work 是因为 PDDL 的 plan template 可重用；它在 AIME 上不 work 是因为每道题的 trick 不同。这暗示了一个研究方向——**memory 的 abstraction level 需要自适应**：对于 homogeneous task，low-level workflow memory 够了；对于 heterogeneous task，需要 high-level strategy abstraction（类似 Voyager [Wang et al., 2023](https://arxiv.org/abs/2305.16291) 的 skill library 但要更抽象）。

### 6.3 RQ3：Task sequence 顺序（Easy→Hard vs Hard→Easy, Table 3）

Claude 3.7-Sonnet 上：
- Base：AlfWorld 0.50/0.73, ScienceWorld 0.32/0.74
- Easy→Hard: ExpRecent 0.66/0.82, ExpRAG 0.77/0.87, **ReMem 0.91/0.96**
- Hard→Easy: ExpRecent 0.72/0.85, ExpRAG 0.87/0.92, **ReMem 0.94/0.97**

**Hard→Easy 略好于 Easy→Hard**。这其实反直觉——通常 curriculum learning 主张 Easy→Hard，但这里反过来。原因可能是：Hard task 的 failure 包含更多 *negative examples*，agent 通过 Refine 学到 "什么不能做"，这种 negative knowledge 在 Easy task 上 transfer 更稳。这跟 Karpathy 你提的 "learning from failure" 思路很契合，也呼应了 Reflexion [Shinn et al., 2023](https://arxiv.org/abs/2303.11366) 的核心 idea。

### 6.4 RQ4：Feedback type（Table 4）

当 memory 里同时存 success 和 failure experience 时，baseline 方法明显退化（noise 进 memory 干扰 retrieval）。ReMem 保持 robust——靠 Refine 操作 *主动 prune 失败 trajectory 的 misleading 部分*，只保留可学习的 negative lesson。

**这里有一个相当关键的设计 implication**：naive RAG-style memory（ExpRAG）在 failure 进入 memory 后会退化；只有带 Refine 的 ReMem 能稳定 handle failure。这说明 "memory management policy" 在 long-horizon agent 里是必需的，不是 nice-to-have。

### 6.5 RQ5：Cumulative performance（Figure 6）

ReMem 在 4 个 multi-turn environment 上的 cumulative success rate 曲线都比 History baseline 上升得快且稳定。这表明 ReMem 不是 "early burst 然后 decay"，而是真正的 *monotonic continual improvement*——这正是 test-time learning 应该有的 signature。

### 6.6 Step Efficiency（Figure 5）

ReMem 在 AlfWorld 上把平均 step 数从 22.6 降到 11.5（**约 50% 减少**）。这意味着 memory 不仅提升 success rate，还让 agent 更 *economical*。这是 procedural memory 的本质——把 "trial-and-error" 转化成 "informed action"。

## 7. 几个你想 build intuition 的关键点

### 7.1 Memory as Test-time Compute Budget

ReMem 的 Refine operation 本质上是在 *spend additional test-time compute* on memory organization。这跟你最近讲的 "test-time compute scaling" [Snell et al., 2024](https://arxiv.org/abs/2408.03314) 思想一致——但 ReMem 的 compute 不是花在 *search over solution space*，而是花在 *search over memory structure*。这是一种 orthogonal 的 test-time scaling axis。

### 7.2 Streaming Evaluation 是 Test-time Learning 的正确 evaluation paradigm

Evo-Memory 的核心 contribution 其实是 **evaluation paradigm**——把 static dataset 变成 streaming task sequence。这跟 StreamBench [Wu et al., NeurIPS 2024](https://arxiv.org/abs/2401.07643) 同向，但 Evo-Memory 更进一步：
- StreamBench 测 factual retention
- Evo-Memory 测 *procedural reuse + memory evolution*

Karpathy，这跟你 2022 年讲 "Software 2.0" 的延伸很自然——dataset 不再是 *training data*，而是 *experience stream*。evaluation 也要跟着升级。

### 7.3 ReMem 的 MDP 视角

ReMem 的 (state, action, transition) 是：

- State: s_t^n = (x_t, M_t, o_t^{1:n-1})
- Action space: {Think, Act, Refine}
- Transition: o_t^n = Agent(x_t, M_t, a_t^n)
- Reward（implicit）：f_t = task correctness at end of step t

这是一个 *prompting-based MDP*，没有 RL training，但结构上是 POMDP（M_t 是 partial observation over 全部 history）。**Memory 就是 POMDP 里的 belief state**。如果你愿意，可以加 RL 来优化这个 policy——Memory-R1 [Yan et al., 2025](https://arxiv.org/abs/2508.19828) 已经开始做这件事。Evo-Memory 提供了 standardized eval suite 给这类 RL-trained memory agent。

### 7.4 与 MEM1 / Mem0 / A-Mem 的本质区别

| 方法 | Memory 形式 | Update 机制 | Optimization |
|------|-------------|-------------|--------------|
| **Mem0** ([Chhikara et al., 2025](https://arxiv.org/abs/2504.19413)) | Structured text entries | LLM-as-judge decide add/update/delete | Prompting |
| **A-Mem** ([Xu et al., 2025](https://arxiv.org/abs/2502.12110)) | Zettel-like notes | Agentic self-organize | Prompting |
| **MEM1** ([Zhou et al., 2025](https://arxiv.org/abs/2506.15841)) | Latent hidden state | RL-learned read/write | RL |
| **Memory-R1** ([Yan et al., 2025](https://arxiv.org/abs/2508.19828)) | Structured text | RL-learned memory ops | RL |
| **ReMem** (本文) | Structured text | Think/Act/Refine loop | Prompting |

**ReMem 的特殊之处**：把 memory refine 作为 agent 的 *in-loop action*，而不是 *post-hoc cleanup*。这是 procedural memory 的关键——memory 的 organization 不是事后整理，而是 reasoning 的一部分。

### 7.5 一个潜在的 concern：Prompting-based 的 ceiling

ReMem 全靠 prompting 实现 Refine，这意味着：
1. Refine 的 quality 依赖 LLM 的 self-reflection 能力（Gemini-2.5 / Claude 3.7 都还行，小模型会崩）
2. Refine 的 cost 是额外的 LLM call——在 step budget 计算里，ReMem 的"高效"可能掩盖了它的 "expensive per step"
3. Pruning 决策可能 sub-optimal——如果 LLM 错误地 prune 了 critical experience，下游 task 会受损

这给未来工作留了空间：用 RL 或 supervised learning to optimize the Refine policy（这恰恰是 Memory-R1 的方向）。

## 8. 与 broader test-time learning 文献的连接

Evo-Memory 的定位在三条 research line 的交叉点：

1. **Test-time Adaptation (TTA)** → Test-time Training (TTT) → Test-time Self-Learning (TTSL)
   - TENT [Wang et al., ICLR 2021](https://arxiv.org/abs/2006.10926)：entropy minimization for distribution shift
   - TTT [Sun et al., 2020](https://arxiv.org/abs/1909.13255)：把 test sample 当 mini-training data
   - MEMO [Zhang et al., NeurIPS 2023](https://arxiv.org/abs/2310.13331)：augmentation + adaptation
   - Evo-Memory 把这个 line 从 *parameter adaptation* 推到 *memory adaptation*——不动 weight，动 memory store

2. **Self-evolving Agents**
   - Voyager [Wang et al., NeurIPS 2023](https://arxiv.org/abs/2305.16291)：skill library via code generation
   - Reflexion [Shinn et al., 2023](https://arxiv.org/abs/2303.11366)：verbal RL via reflection
   - Self-Refine [Madaan et al., 2023](https://arxiv.org/abs/2303.17651)：iterative self-improvement
   - Evo-Memory 把这些方法都囊括进 unified eval framework，并提供 streaming setting 下的 apples-to-apples 比较

3. **Long-term Memory Systems**
   - MemGPT [Packer et al., 2023](https://arxiv.org/abs/2310.08560)：OS-style memory hierarchy
   - MemoryBank [Zhong et al., 2023](https://arxiv.org/abs/2305.10250)：long-term conversational memory
   - LongMemEval [Wu et al., 2024](https://arxiv.org/abs/2410.10813)：benchmark for chat memory
   - Evo-Memory 强调 *procedural* memory over *episodic* memory——这是新角度

## 9. 论文的 Limitations 和 Future Direction

作者自己承认（Appendix D）：
1. 只测了 Gemini / Claude 两家，没测 open-weight（Llama / Mistral）
2. 只测了 text + goal-oriented，没测 multimodal
3. 没测非常 long-horizon（>100 task）的 stability

我会加几个：
4. **Prompting-based Refine 的 robustness**：Refine 错误的 recovery 机制缺失
5. **Memory 的 abstraction level 是固定的**：homogeneous task 用 low-level memory OK，heterogeneous task 需要 high-level abstraction，但 ReMem 用同一个 template——这里有个 *abstraction level mismatch* 的问题
6. **Cost analysis 缺失**：Refine 的额外 LLM call 带来的 latency / token cost 没有报告
7. **Generalization 到 OOD task stream**：如果 task stream 突然切换 domain，memory 怎么 handle？这是 lifelong learning 的 core challenge，但论文没测

**最有意思的 future direction**：把 ReMem 的 Refine policy 用 RL 训练，类似 Memory-R1。这能把 prompting-based 的 ceiling 推高，并且让 Refine 决策 *data-driven* 而非 *prompt-engineered*。Evo-Memory 提供了 perfect eval suite 给这类工作。

## 10. Reference & Further Reading

- Evo-Memory 本身（arXiv 待发，目前从文本看是 Google DeepMind + UIUC 合作）
- [Mem0](https://arxiv.org/abs/2504.19413)
- [A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)
- [Memory-R1: RL for Memory Management](https://arxiv.org/abs/2508.19828)
- [MEM1: Synergize Memory and Reasoning](https://arxiv.org/abs/2506.15841)
- [MemOS: Memory Operating System](https://arxiv.org/abs/2505.22101)
- [Self-RAG](https://openreview.net/forum?id=hSyW5go0v8)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Reflexion: Verbal RL](https://arxiv.org/abs/2303.11366)
- [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952)
- [Agent Workflow Memory](https://arxiv.org/abs/2409.07429)
- [Voyager: Open-Ended Embodied Agent](https://arxiv.org/abs/2305.16291)
- [StreamBench (NeurIPS 2024)](https://arxiv.org/abs/2401.07643)
- [LongMemEval](https://arxiv.org/abs/2410.10813)
- [LifelongAgentBench](https://arxiv.org/abs/2505.11942)
- [AgentBoard](https://arxiv.org/abs/2401.13178)
- [Zep: Temporal Knowledge Graph Memory](https://arxiv.org/abs/2501.13956)
- [MemGPT](https://arxiv.org/abs/2310.08560)
- [Gemini 2.5 Technical Report](https://arxiv.org/abs/2507.06261)
- [Test-time Compute Scaling (Snell et al.)](https://arxiv.org/abs/2408.03314)
- [TENT: Test-time Entropy Minimization](https://arxiv.org/abs/2006.10926)
- [Self-Refine](https://arxiv.org/abs/2303.17651)
- [A Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046)
- [GPQA](https://arxiv.org/abs/2311.12022)
- [MMLU-Pro](https://arxiv.org/abs/2406.01574)
- [ToolBench / Gorilla](https://arxiv.org/abs/2305.15334)
- [AlfWorld](https://arxiv.org/abs/2010.03768)
- [BabyAI](https://arxiv.org/abs/1810.08272)
- [ScienceWorld](https://arxiv.org/abs/2205.07559)
- [PDDLBench](https://arxiv.org/abs/2312.00754)
- [Jericho](https://arxiv.org/abs/1908.11539)

---

**Bottom line for intuition building**：Evo-Memory 的核心 contribution 是把 LLM memory evaluation 从 *static recall* 推到 *streaming experience reuse*。它证明了：
1. Multi-turn long-horizon task 上，self-evolving memory 是 game-changer（PDDL 从 0.17 → 0.83）
2. Refine operation（in-loop memory reorganization）是处理 failure experience 的关键
3. 简单的 ExpRAG（task-level RAG over past experience）已经能 beat 大部分复杂的 adaptive memory method——这暗示 "task-level experience reuse" 在 current LLM agent design 里被严重低 invested
4. Memory 和 reasoning 应该是 *co-evolving* 而非 *sequential*（retrieval then generation）

这是个值得 follow 的方向，特别对你在 build 的 test-time learning intuition——memory 作为 *explicit, agent-operated* object 是比 in-context recall 更 scalable 的 test-time computation carrier。下一步 RL-train Refine policy 应该能把 prompting-based ReMem 推过现在的 ceiling。
