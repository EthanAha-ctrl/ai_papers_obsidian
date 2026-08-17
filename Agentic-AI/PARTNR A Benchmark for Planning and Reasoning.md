---
source_pdf: PARTNR A Benchmark for Planning and Reasoning.pdf
paper_sha256: 012863a9401bfd4519f94451113ff628ad0443e958f3a19ff6fe0e4232a78a4f
processed_at: '2026-08-06T02:24:35-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PARTNR 用人话讲讲

好，我换一种方式。假设我们坐下来喝咖啡聊这篇 paper，我会这么跟你讲：

---

## 这篇 paper 到底在干嘛

Meta FAIR 的一帮人想搞一个 benchmark，专门测试 "机器人能不能跟人一起做家务"。听起来简单，但实际上之前没有人把这件事做对。

之前做 embodied AI benchmark 的人要么是让 robot 一个人在房子里干活（ALFRED、Habitat 那些），要么是搞 multi-agent 但任务不用自然语言描述（Overcooked、FurnMove）。PARTNR 是第一个把 **"两个 agent + 自然语言任务 + 3D 多房间 + 10 万个任务"** 这几样同时塞进一个 benchmark 的工作。

网站: https://aihabitat.org/partnr  
代码: https://github.com/facebookresearch/partnr-planner

---

## 为什么这件事难

你想啊，如果我说一句 "帮我把盘子收一下，然后把书放到书架上，洗完碗再放进柜子"，这话说给真人听没问题。但说给 robot 听，它需要做几件事：

1. **理解这句话**：盘子在哪？书架是哪个？洗完碗是什么意思？
2. **分工**：robot 能搬东西但不会洗碗（heterogeneous），它得知道把洗碗这步让给 human
3. **顺序**：先洗碗再放柜子（temporal），顺序反了就失败了
4. **空间关系**：书要"放在一起"（spatial），不是随便摆

PARTNR 把日常家务的约束拆成四类，我自己的直觉是这其实对应四种不同的 reasoning 能力：

- **Constraint-free**：终态匹配就行，最像传统 planning
- **Spatial**：几何关系推理，`is_next_to` 这类
- **Temporal**：state tracking across time，LLM 的弱项
- **Heterogeneous**：theory of mind，要推理 partner 能干啥

测试集里大约 50% 的任务同时有多个约束叠加，这就是为什么这个问题难。

---

## 10 万个任务怎么造出来的

这是工程上最硬的部分。手工造 10 万个任务不现实，纯 LLM 生成又会 hallucinate 一堆不存在的物体。他们走了个很聪明的路子，分四步：

**第一步**：让 LLaMA-3-70B 看一个 Habitat 3.0 里的房子（room list + furniture list + 物体 list），生成 5 个 task。每个 task 带初始状态和目标状态。同时让它生成一些 clutter 物体增加复杂度。

**第二步**：把这些 task 在 simulator 里实际跑一遍，过滤掉不可行的。比如 "Clear dishes from the living room" 但这房子没有 living room，直接扔掉。这一步过滤掉 90% 的生成结果——你想想，纯 LLM 生成 embodied task 的 hallucination 率是 90%，这个数字本身就说明 grounding 有多重要。

然后人工标注 1000 个任务，确保四种类型分布均衡，物体和房间多样化。这 1000 个就是"种子"。

**第三步**：拿这 1000 个种子当 in-context example，让 LLM 把它们"移植"到其他 59 个房子里。比如 "Clear all dishes from the living room" 改成 "Clear all toys from the bedroom"。这一步只有 10% 被过滤，质量高多了。

**第四步**：让真人用户用 HITL 工具实际尝试解任务，6 次都解不了的就判为 infeasible 扔掉。

我的直觉：**simulation-in-the-loop 是关键**。LLM 在纯文本里说 "move the washing machine" 没问题，但 simulator 里没有 washing machine 这步就被堵住了。simulator 在这里当 "executable verifier" 用。这思路其实跟 GenSim、RoboCasa 一样，但 PARTNR 的规模和严格度更高。

---

## 评估函数怎么写

每个 task 都要有个 evaluation function 来判断"任务算不算完成"。手工写 10 万个 evaluation 不现实，所以用 CodeLLaMA-70B 生成 predicate-based 的 Python 程序。

predicate 词汇表很精简，Table 6 里列的：

- **Rearrange**: `is_on_top(o1, o2)`, `is_inside(o1, o2)`, `is_in_room(o1, r1)`, `is_on_floor(o1)`
- **Spatial**: `is_next_to(o1, o2)`（bounding box 垂直 overlap 且水平 L2 距离 < 0.50m）
- **State**: `is_clean`, `is_filled`, `is_powered_on` 等等

这里 $o_1$ 是 object，$o_2$ 是 furniture，$r_1$ 是 room。下标 1、2 就是 first argument 和 second argument，没有更深的含义。

然后还有三个层级表达更复杂的语义：

**Proposition**：基本判断单元，比如
```python
is_on_top([spoon_1, spoon_2, spoon_3], [table_1, table_2], number=2, arg_match=True)
```
方括号里的列表是 OR 关系（任一都行），`number=2` 是要满足 2 个，`arg_match=True` 是这 2 个要落在同一个 table 上。这对应 "bring two spoons to the same table" 这句话。

**Dependency**：`after_satisfied` / `after_unsatisfied` / `while_satisfied`，表达 "X 之后做 Y" 这类时序。

**Constraint**：四种
- `TemporalConstraint`: DAG over propositions，定义满足顺序
- `SameArgConstraint`: 强制某些 propositions 用同一个 argument
- `DifferentArgConstraint`: 强制用不同 argument（比如 "each candle on its own table"）
- `TerminalSatisfactionConstraint`: 终态必须满足的 propositions

最终指标：

$$PC = \frac{\sum_{i=1}^{N} \mathbb{1}[p_i \text{ satisfied w.r.t. constraints}]}{N}$$

$N$ 是 propositions 总数，$p_i$ 是第 $i$ 个 proposition，$\mathbb{1}[\cdot]$ 是 indicator function（满足为 1，不满足为 0）。$PC \in [0, 1]$，$PC = 1$ 就算 success。

这个设计我特别喜欢的地方在于它把"任务完成度"变成了一个**可执行程序**。不是让人判断"这个任务算不算完成"，而是有一个明确的 predicate 程序在 simulator 里跑，输出 0/1。这比 BDDL（Srivastava et al., 2022, https://behavior.stanford.edu/behavior-1k）的 expressivity 高，BDDL 不支持 time-varying evaluation。

LLM 生成 evaluation 的准确率从 50% 提升到 92%（通过 retrieval-augmented prompting，用 1000 个手工标注的 evaluation 当 example），整体联合准确率 $90\% \times 92\% = 83\%$。

---

## LLM 怎么当 planner 用

架构很 standard，Figure 5 里画了：

- **High-level**: LLM，从 skill library 里选 skill
- **Low-level**: 实际执行 skill 的 neural network（或 oracle）

skill 包括 navigate、pick、place、open、close，robot 只能干这些。human 额外能干 clean、fill、pour、poweron、poweroff。这就是 heterogeneous 任务的根源。

LLM 看到的不是 RGB，是一个 **world graph**——三层结构：rooms → furniture → objects。每个 node 存 semantic category、3D 位置、state。这跟 SayPlan（Rana et al., CoRL 2023, https://sayplan.github.io）的思路一样。

有个很关键的工程细节：**constrained generation**。LLM 输出被强制约束成合法的 skill call。grammar 是 state-conditioned 的，每一步根据当前 world graph 重新构造。这确保 LLM 不会输出 "Navigate to the washing machine" 这种 hallucination，因为 washing_machine 不在 grammar 里就直接被 decode 阶段挡掉。这跟 Outlines、guidance、JSON mode 是一个思路，但这里 grammar 是动态的。

---

## Baseline 设计的五个轴

这部分设计很干净，对应五个研究问题：

1. **Planner variant**: Heuristic Expert（用 GT eval function 的 oracle）/ ReAct / ReAct+RAG / Finetuned
2. **Centralized vs Decentralized**: 一个 LLM 控两个 agent vs 每个 agent 一个 LLM
3. **Partial vs Full observability**: 不知道物体在哪 vs 知道
4. **Learned vs Oracle skills**: 神经网络 skill vs 特权 skill
5. **Privileged vs Non-privileged perception**: simulator GT vs ConceptGraphs（Gu et al., ICRA 2024, https://concept-graphs.github.io）

这五个轴是干净的 ablation 设计。其中 partial observability 是不知道物体位置（需要 ExploreRoom 找），non-privileged perception 是检测物体本身可能错（misclassification）。这是两种不同的"感知降级"。

---

## ReAct-RAG 这部分有点反直觉

作者从 2000 个 train task 上跑 ReAct，收集 925 条成功 trace。测试时用 sentence embedding 找最相似的 trace，作为 in-context example 加进 prompt。

结果在 Table 2 里，ReAct-RAG success 0.71 比 ReAct 0.73 略低，但 sim steps 更高。**RAG 在这个场景下没帮上忙**，反而引入的 example trace 干扰了 LLM 在新任务上的 reasoning。

这跟 Pang et al. 2024、Madaan et al. 2024 的 self-refine 思路相关，但 PARTNR 的发现是 RAG 在 multi-agent 协作场景下不一定提升主指标。我自己猜的原因是：multi-agent 下 partner 的 action 引入额外的不确定性，example trace 的迁移性变差。

---

## Finetuned 8B ≈ ReAct 70B，但快 8.6 倍

这是最有意思的发现。用 LoRA fine-tune Llama-3.1-8B：

**LoRA 配置**：
- rank $r = 132$（低秩矩阵的秩）
- $\alpha = 128$（scaling factor，实际加的扰动是 $\frac{\alpha}{r} \cdot BA = \frac{128}{132} \cdot BA \approx 0.97 \cdot BA$）
- dropout = 0.01
- 只在 $W^V$（value projection）和 $W^Q$（query projection）上加 adapter

数据构造：从 ReAct baseline 收集 success rate 1.0 的 episode，拆成单步 (world_graph, action_history) → action 对。过滤掉失败的 action。最终 1226 episodes，15889 training samples。在 4 × A100 上训 40000 步，24 小时。

**关键发现（Table 9）**：只用 ReAct 数据 fine-tune 效果最好（success 0.70），加入 Heuristic-Expert 数据反而下降（0.63）。这反直觉——用 GT eval function 训的 expert trace 理论上信息更丰富，但 distill 出来的 model 在 partial observation 下的 noise 上泛化不好。

这跟 Hsieh et al. 2023 的 "distilling step-by-step" 发现一致——分布外的 trace 反而是更好的 teacher，因为它包含了真实推理过程中的 error recovery。

---

## 实验结果里几个让我震惊的数字

Table 2 的主结果：

| Row | 方法 | 设置 | Success | Sim Steps |
|-----|------|------|---------|-----------|
| a | ReAct-Single | 单 agent, oracle, partial | 0.73 | 2519 |
| e | ReAct | 双 agent decentralized, oracle | 0.73 | 3295 |
| h | ReAct | learned skills | 0.57 | 6494 |
| i | ReAct | ConceptGraph perception | 0.30 | 12490 |

**三个让我震惊的发现**：

**1. Coordination burden 是负的**：两个 LLM 一起干比一个 LLM 单干还慢（3295 > 2519）。这说明 LLM 在 multi-agent coordination 上的能力是**负贡献**——额外开销超过了并行收益。

对比 human pair：multi-user 2369 < single-user 3046。人类 pair 比单人快 22%。所以这个 coordination burden 是 LLM 特有的，不是任务本身的内在属性。

**2. Skill failure 让 LLM 崩溃**：从 oracle skill 换成 learned skill，success 从 0.73 跌到 0.57，sim steps 翻倍。LLM **不会从 skill failure 中恢复**——pick 失败了它就继续往下走，不会重新规划。

**3. Perception error 是真正的灾难**：从 privileged perception 换成 ConceptGraphs，success 从 0.57 跌到 0.30。LLM 对 perception error **几乎零容忍**——shelf 被误识别成 table，整个 plan 就跑偏了。

---

## Human-in-the-loop 结果更扎心

Table 3，让 129 个真人参与：

| 设置 | Success | Sim Steps | Task Offloading |
|------|---------|-----------|-----------------|
| Single human | 0.93 | 3046 | N/A |
| Two humans | 0.93 | 2369 | 0.59 |
| Human + ReAct 70B | 0.91 | 4267 | 0.16 |
| Human + Finetuned 8B | 0.92 | 3443 | 0.26 |

**几个 sharp 的观察**：

1. **人类 93% vs. LLM 30%**——3 倍 gap。这是 LLM planning 能力当前水平的真实标尺。

2. **Human pair 比单 human 快 22%**，但 **Human-LLM pair 比单 human 慢 13%**。LLM 是个 **negative coordination partner**——它拖累了人类。

3. **Finetuned 在 HITL 下比 ReAct 好**，即使 automated eval 下两者差不多。原因是 latency：Finetuned 8B 每步推理 0.53s，ReAct 70B 每步 4.55s。一个 episode 平均 36 分钟（ReAct）vs 25 分钟（Finetuned）。**人类能适应 LLM 的错误，但不能适应等待**。

这个 latency 的重要性在 robotics 部署里被严重低估。我自己的直觉是：HITL 场景下，inference latency 和 task success rate 几乎同等重要。一个 95% 准确但 10s/step 的 planner 不如一个 85% 准确但 0.5s/step 的 planner。

4. **Task offloading 数字很扎心**：Human-ReAct 只有 16%，Human-Finetuned 26%，而 Human-human 是 59%。LLM 实际只承担了 1/6 到 1/4 的工作。人类实际上"哄着" LLM 干活。

---

## 任务类型 breakdown 也很有意思

Table 13：

| 类型 | Constraint-free | Spatial | Temporal | Heterogeneous |
|------|----------------|---------|----------|---------------|
| Decentralized | 0.82 | 0.82 | 0.60 | 0.66 |

Temporal 任务 success 比 constraint-free 低 27%，heterogeneous 低 20%。这两类是 LLM 的真正弱项：

- **Temporal** 要求 state tracking across time，LLM 在长 rollout 上会丢失中间状态
- **Heterogeneous** 要求 theory of mind（推理 partner 的 capability），LLM 没有显式建模 partner

---

## Coordination 问题的量化

Table 12，有个叫 "extraneous effort" 的指标，意思是"做了但没用的 action 比例"：

| 设置 | Extraneous Effort |
|------|-------------------|
| Single agent | 0.047 |
| Centralized | 0.04 |
| Decentralized | 0.21 |

**Decentralized 下 extraneous effort 暴涨 300%**。这意味着 LLM 30% 的 action 是浪费的——重复 partner 做过的事，或者做了无关的 rearrange。

我自己的联想：这个 extraneous effort 的概念其实可以推广到所有 multi-LLM-agent 系统。我怀疑 AutoGen、CrewAI、Camel 这些 multi-agent 框架里都存在类似的"无意义 actions"问题，只是没人量化。PARTNR 提供了一个很好的 metric template。

---

## ConceptGraphs 集成的工程细节

非特权感知用 modified ConceptGraphs。pipeline 改造（Appendix A.7.2）：

- Meta-CLIP（ViT-H-14-quickgelu）做 object 分类
- YOLO（yolov8x-worldv2）+ SAM 做分割
- LLaMA-3.1-70B 做 room label 标注（基于 10 个最近 furniture 的 categories）

room annotation prompt 很有意思——让 LLM 看一个 furniture 的 10 个最近邻居，输出 7 类 room label 之一（bedroom / living_room / kitchen / dining_room / hallway / bathroom / unknown）。

这暴露了一个深层矛盾：open-vocab perception system 在 closed-world benchmark 上不一定更好。ConceptGraphs 原本是 open-vocab 的，会检测出 benchmark 里没定义的物体，反而增加 LLM 混淆。所以作者把它改成 closed-vocab，用 CLIP embedding 直接分类到 155 个 object type。

---

## 我自己读完的几个 takeaway

**1. LLM planning 当前真实水平是 30%**。在有特权信息（oracle skill + privileged perception + centralized）下能到 73%，说明 LLM 的 reasoning 本身不差，差的是 grounding 和 coordination。

**2. Coordination 是 LLM 的死穴**。两个 LLM 一起比一个 LLM 单干慢，extraneous effort 暴涨 300%。当前 multi-agent LLM 框架（AutoGen 那类）在 embodied long-horizon 任务下的"多 agent 比单 agent 好"假设不成立。

**3. Latency 在 HITL 下几乎跟 success rate 一样重要**。Finetuned 8B 能在 HITL 下击败 ReAct 70B，主要靠快 8.6 倍的推理速度。这对实际 robotics 部署的启示是：**小模型 + 快推理 > 大模型 + 慢推理**。

**4. Perception robustness 是当前 embodied LLM 的最大瓶颈**。ConceptGraphs 引入的 perception error 让 success 从 0.57 跌到 0.30。这暗示 VLM 接入可能会大幅改善——VLM 能直接看 RGB，绕过 ConceptGraphs 的 detection error 累积。

**5. 半自动化 benchmark 生成是个 meta-trend**。LLM 生成 task + simulator 验证 + 人工 quality gate + in-context scale 这个 pipeline 是可复用的。RoboCasa、GenSim 是早期版本，PARTNR 把它推到了 10 万规模。

---

## 一些可能的延伸方向

我自己的几个 idea：

1. **Theory of mind module**：让 LLM 显式建模 partner 的 state（"它在做什么，它看到什么，它接下来会做什么"）。当前 decentralized 下 extraneous effort 300% 增长就是缺这个。

2. **Skill failure recovery**：把 skill failure signal 作为 prompt 一部分反馈给 LLM，让它重新规划。当前 learned skill 下 success 跌 16% 全是因为这个。

3. **VLM-based perception**：ConceptGraphs 的 room label 错误率太高，VLM 可能更 robust。Llama-3.2-Vision 或 GPT-4o 直接看 RGB 而不是 text-summarized graph，可能部分解决 0.30 → 0.73 的 gap。

4. **HITL data → DAgger**：HITL 数据本身就是 distillation 金矿。当前 fine-tune 只用 ReAct trace，未来可以用 human-LLM pair trace 做 DAgger-style imitation。

5. **Hierarchical planning with explicit task divider**：当前 LLM 在 temporal + heterogeneous task 上弱，可能需要一个 explicit high-level task divider（把 task 拆给 human 和 robot）+ low-level executor 的两级架构。

---

## 一句话总结

PARTNR 是第一个把 multi-agent + language + 3D + 10 万任务拼到一起的 embodied benchmark，揭示了当前 LLM 在 embodied multi-agent 任务上的三个核心 bottleneck：**grounding robustness、coordination、skill failure recovery**。Fine-tuned small model 是工程上最 actionable 的短期方向，长期还是得靠 perception 和 coordination 上的根本性改进。

---

## 参考链接

- PARTNR 项目: https://aihabitat.org/partnr
- PARTNR 代码: https://github.com/facebookresearch/partnr-planner
- Habitat 3.0: https://aihabitat.org/habitat3/
- Habitat 3.0 论文: https://arxiv.org/abs/2402.04235
- ALFRED: https://askforalfred.com
- ConceptGraphs: https://concept-graphs.github.io
- SayPlan: https://sayplan.github.io
- RoboCasa: https://robocasa.ai
- HSSD: https://huggingface.co/datasets/hssd/hssd-data
- OVMM (HomeRobot): https://homerobot.github.io
- CoELA: https://github.com/THUDM/CoELA
- LoRA: https://arxiv.org/abs/2106.09685
- ReAct: https://react-lm.github.io
- Llama 3: https://arxiv.org/abs/2407.21783
- Overcooked-AI: https://github.com/HumanCompatibleAI/overcooked_ai
- VirtualHome: http://virtualhome.org
- BDDL: https://behavior.stanford.edu/behavior-1k
- Grammar-constrained decoding: https://arxiv.org/abs/2305.13971

---

# PARTNR：人类-机器人协作中的规划与推理 Benchmark 深度解析

Andrej，这篇来自 FAIR Meta 的 PARTNR paper 我看了好几遍，从多个角度来剖析给你。我会从 motivation、benchmark 构造机制、评估系统、实验结论这几个层面，把技术细节摊开讲，并加入我自己的联想和直觉构建。

---

## 1. 这个工作的核心定位

PARTNR 要解决的问题很清晰：现有的 embodied AI benchmark 存在一个明显的空白。要么 robot 单独操作（如 ALFRED、ALFWorld、BEHAVIOR-1K、Habitat 2.0 这类），要么任务用自然语言描述但不是 multi-agent（如 RoboGen、GenSim），要么是 multi-agent 但没有自然语言（如 FurnMove、Overcooked、WAH）。PARTNR 是第一个把 **3D-M（multi-room）+ Multi-Agent + Language + HL+LL（高阶+低阶动作）+ C/S/T/H（constraint-free/spatial/temporal/heterogeneous）+ 100,000 tasks** 同时拼在一起的工作。Table 1 里那个对比矩阵非常清楚地显示了这点。

我个人的直觉是：这其实是把 **Habitat 3.0** (Puig et al., 2024, ICLR) 已经具备的 human-avatar-robot co-habitat 能力，扩展到一个大规模、可重复、可系统评估的 planning & reasoning 测试场。代码和网站分别在 https://github.com/facebookresearch/partnr-planner 和 https://aihabitat.org/partnr。

---

## 2. 四种任务类型的语义结构

PARTNR 把任务沿着四个正交维度切片，每个 task 都落在这个分类里：

- **Constraint-free (C)**: 比如 "Let's move all dirty plates to the sink." 子任务对顺序和分工都没有约束，两个 agent 可以任意切分。
- **Spatial (S)**: 比如 "Let's place the books on the shelf next to each other." 这类任务的核心是 `is_next_to`、`is_on_top` 这类 spatial predicate，要求最终状态满足几何关系。
- **Temporal (T)**: 比如 "Let's remove the candles from the dining table before bringing the plates to the table." 关键是要追踪整个 rollout 时间序列上的状态变化，不能只看终态。
- **Heterogeneous (H)**: 比如 "Let's wash the dishes before putting them in shelves." 这里 robot 的 skill set 不包含 clean/fill/pour/poweron/poweroff，必须把这部分 action 让给 human agent。

这四种类型覆盖了日常家务协作里几乎所有典型约束结构。我注意到 heterogeneous 这类非常关键，因为它强制 LLM 去 **reason about agent capabilities**，这种能力推理在大多数 planning benchmark 里是缺失的。

### 2.1 测试集的任务分布（Figure 4 解读）

论文里 Figure 4 展示了 test split 的任务类型分布。constraint-free 大约占 24%，剩余 76% 是包含 spatial/temporal/heterogeneous 特征的任务。更细看，大约 50% 的任务至少有两个特征交叉。这个分布很关键——它意味着 PARTNR 评估的不只是单一约束类型上的能力，更多是 **组合约束** 下的 reasoning。

我自己的联想：这其实有点像强化学习里把 reward shaping 拆成 sparse + dense 的成分。constraint-free 是最 sparse 的 reward（终态匹配即可），temporal 要求 tracking 整条轨迹的 intermediate state，heterogeneous 要求 capability-aware assignment。把这几个 reward 信号叠加，问题难度成倍上升。

---

## 3. 半自动化生成 Pipeline——这个才是工程上最硬的部分

这部分是整篇 paper 工程量最大的地方。100,000 个任务不可能手工构造，但又不能纯 LLM 生成（会 hallucinate）。所以作者走了一条 **LLM 生成 + simulation-in-the-loop grounding + human annotation + scale via in-context prompting** 的路径，看 Figure 2。

### 3.1 四步生成流水线（Appendix A.4）

**Step 1: 小规模 free-form 生成**
- 在 Habitat 3.0 里加载一个 HSSD 房子，解析出 room list + furniture list + 可用 object list（OVMM 数据集）。
- 把这些信息塞进 LLM（LLaMA-3-70B-Instruct），让它生成 5 个 task，每个 task 包含 `instruction`、`initial_state`、`final_state`。
- 同时 LLM 还要生成 clutter（额外物体）来增加环境复杂度。

**Step 2: Simulation-in-the-loop 过滤 + 人工标注**
- 把生成的 task 在 simulator 里实例化，过滤掉不可行的（比如 "Clear dishes from the living room" 但这个房子没有 living room）。
- 这一步过滤掉了大约 90% 的 free-form 生成指令（这数字非常惊人，说明纯 LLM 生成 task 在 embodied 场景下的 hallucination 率极高）。
- 人工标注 1,000 个任务，确保 task type 分布平衡，物体和房间多样化。

**Step 3: 大规模生成**
- 把 1,000 个 human-annotated tasks 作为 in-context examples。
- 给 LLM 一个 new house description + 一个 example task，让它把 task "transplant" 到新房子里。比如 "Clear all dishes from the living room" → "Clear all toys from the bedroom"。
- 这一步只有约 10% 被过滤，质量大幅提升。

**Step 4: Human-in-the-loop 过滤**
- 让真人用户用 HITL 工具尝试解任务，6 次重试（3 single + 3 multi）都解不了的任务判为 infeasible，从数据集移除。

这里我的直觉是：**simulation-in-the-loop 是降低 LLM hallucination 的关键 grounding 机制**。LLM 在纯文本里说 "move the washing machine" 没问题，但如果 simulator 里没有 washing machine，这一步就被过滤掉了。这种把 simulator 当 "executable verifier" 来用，和之前 GenSim、RoboGen、RoboCasa 的思路是一脉相承的，但 PARTNR 的规模和过滤严格度更高。

### 3.2 评估函数生成（Section 3.2 + Appendix A.5）

这部分我觉得是最精巧的。每个 task 都要配一个 evaluation function 来判定是否完成。手工写 100,000 个 evaluation 不现实，所以用 LLM（CodeLLaMA-70B-Instruct）生成 predicate-based Python 程序。

**Predicate vocabulary**（Table 6）非常精简但表达力强：
- **Rearrange predicates**: `is_on_top(o1, o2)`, `is_inside(o1, o2)`, `is_in_room(o1, r1)`, `is_on_floor(o1)`
- **Spatial predicates**: `is_next_to(o1, o2)`（bounding box 垂直 overlap 且水平 L2 < 0.50m）, `is_clustered(o1,...,on)`
- **State predicates**: `is_clean`, `is_dirty`, `is_filled`, `is_empty`, `is_powered_on`, `is_powered_off`

**Proposition + Dependency + Constraint 三层结构**：

```python
is_on_top([spoon_1, spoon_2, spoon_3], [table_1, table_2], number=2, arg_match=True)
```
这里 `[spoon_1, spoon_2, spoon_3]` 表示 OR 关系（任意一个都行），`number=2` 表示需要满足 2 个，`arg_match=True` 表示这两个要落在同一个 table 上（实现 "bring two spoons to the same table" 的语义）。

**PropositionDependency** 支持 `after_satisfied` / `after_unsatisfied` / `while_satisfied` 三种 relation，用来表达 "after X then Y" 这类时序约束。

**Constraints** 包含四种：
- `TemporalConstraint(graph_edges)`: DAG over proposition indices，定义满足顺序。
- `SameArgConstraint(proposition_indices, arg_names)`: 强制某些 propositions 用相同 argument。
- `DifferentArgConstraint(proposition_indices, arg_names)`: 强制某些 propositions 用不同 argument（比如 "place each candle on its own table"）。
- `TerminalSatisfactionConstraint(proposition_indices)`: 终态必须满足的 propositions。

**评估指标公式**：

$$PC = \frac{\sum_{i=1}^{N} \mathbb{1}[p_i \text{ satisfied w.r.t. constraints}]}{N}$$

其中 $N$ 是 propositions 总数，$p_i$ 是第 $i$ 个 proposition。$S := (PC = 1)$，即所有 propositions 都满足。

第三步是预测 SameArgConstraint / DifferentArgConstraint。

最后 retrieval-augmented prompting 把准确率从 50% 提升到 92%（Appendix A.6.2）。整体联合准确率 $90\% \times 92\% = 83\%$。

我的联想：这其实有点像 **program synthesis for evaluation** 的路子，把 task specification 从自然语言编译成一个 executable predicate program。这跟 BDDL（Srivastava et al., 2022）和 PDDL 的思路类似，但 PARTNR 选择 Python-based 是为了 expressivity（BDDL 不支持 time-varying evaluation）和 human/LLM interpretability。我注意到 Table 8 里 evaluation 最常见的 failure mode 是 `Incorrect Temporal Grouping` 和 `Incorrect Predicate (Room vs Furniture)`，这暴露了 LLM 在 spatial reference resolution 上的系统性弱点——它经常把 "move to bedroom" 错误地具体化为 "move to bed"。

### 3.3 PrediViz 可视化工具（Appendix A.6.2）

这是个很巧妙的 annotation 工具。它把 task 和 evaluation function 渲染成 2D 图形，room 是 box，object 是带颜色的小框，receptacle 是 25 个 bespoke icons，箭头表示 rearrangement（solid 是 AND，dotted 是 OR），时间约束拆成多帧。

实测下来 PrediViz 比纯文本 annotation **快 2.6 倍、准确率高 8%、感知难度低 24%**（n=22 的小规模实验）。这是非常典型的"可视化降低 cognitive load"的 case，跟 AutoML 工具里把 pipeline 可视化是同一个道理。

---

## 4. 数据集特性与多样性

PARTNR 数据集切分：
- 37 train scenes，100,000 episodes
- 13 validation scenes，1,000 episodes
- 10 test scenes，1,000 episodes（全部人工标注）

任务平均 4.7 个 propositions（暗示平均步数）。覆盖 155 个 unique object type、20 个 furniture class、13 个 room type。

**Linguistic phenomena 分析**（Table 4，50 个 test episode 手工标注）：
- Class reference（"the table"）: 100% 出现，平均每 episode 6.38 次
- Instance reference: 12%，平均 0.14 次
- Co-reference（"that", "it"）: 50%，平均 0.64 次
- Passive voice: 14%，主动 voice 占主导（92%）
- High-level goal spec: 50%
- Agentic reference: 14%（"You do X while I do Y"）

**Secondary task characteristics**（Table 5，全 test 自动标注）：
- Subset count（"bring two cups" 当有 ≥3 个 cups）: 1%
- Resolvable ambiguity（"any chair"）: 68%，平均 1.68 次
- Dependent rearrange（"same table"）: 7%
- Multi-step rearrange（cup→sink→cabinet）: 6%，平均 2 次/任务

这个分析我特别喜欢，因为它精确指出了 LLM 需要解决的语言理解问题。Co-reference resolution 在 50% 的任务里出现，意味着 agent 必须能跟踪 "it" 指代什么；resolvable ambiguity 在 68% 的任务里出现，意味着 agent 需要处理 "any chair" 这种开放集合。这些都是 LLM 在纯 NLP 上做得到，但在 grounded embodied 场景下容易出错的点。

---

## 5. ReAct Planner 架构与基线

### 5.1 两层 hierarchical control（Figure 5）

- **High-level planner**: LLM，从 skill library 里选 skill。
- **Low-level skills**: navigate, pick, place, open, close（robot）；额外加 clean, fill, pour, poweron, poweroff（human）。

**World graph 是三层层级结构**（Appendix A.7, Figure 10）：
- Level 1: rooms
- Level 2: furniture
- Level 3: objects + agents

每个 node 存 semantic category、3D info（position/bbox）、state（clean/on/open 等）。这个 world graph 是 ReAct planner 的核心 grounding 机制，类似 SayPlan（Rana et al., 2023）的思路——把 environment 编码成 graph，用 tool 提取信息，再用 motor skill 执行。

### 5.2 五个评估维度的设计

作者沿着五个轴设计 baseline，每个轴都对应一个具体的研究问题：

1. **Planner variant**: Heuristic Expert / Zero-shot ReAct / ReAct-RAG / Finetuned
2. **Centralized vs. Decentralized**: 单一 LLM 控制 vs. 各自一个 LLM
3. **Partial vs. Full observability**: 不知道 vs. 知道物体位置
4. **Learned vs. Oracle skills**: 神经网络 skill vs. 特权 skill
5. **Privileged vs. Non-privileged perception**: simulator GT vs. ConceptGraphs

这五个轴是非常干净的 ablation 设计。其中 partial observability 和 non-privileged perception 是两个不同的"感知降级"——partial observability 是不知道物体在哪儿（需要 ExploreRoom），non-privileged perception 是检测物体本身可能错（misclassification, 错误 room assignment）。

### 5.3 Constrained Generation（Appendix A.9.2）

这是个关键技术细节。LLM 输出要被强制约束成合法的 skill call。作者用 Geng et al. 2023 的 grammar-constrained decoding，把 skill API schema 编码成 CFG：

```
root ::= Navigate | Pick | Place | Open | Close | ...
Place ::= "Place[" object "," WS spatial_relation "," WS furniture ...
```

object / furniture / room 规则 **动态**地根据当前 world graph 生成。这确保 LLM 只会输出它观察到的实体上的合法 action。这是一个非常重要的工程优化——它把 LLM 的 "hallucination" 问题在 decoding 阶段直接堵住，而不是让 LLM 输出后再 reject。

我的联想：这跟 Outlines、JSON-mode、guidance 这些 structured generation 库的思路一致，但这里 grammar 是 **state-conditioned** 的，每一步都重新构造，这是 embodied planning 里特别需要的。

### 5.4 ReAct-RAG（Appendix A.9.3）

作者从 2,000 个 train task 上跑 ReAct，收集了 925 条成功 trace，作为 RAG 数据库。测试时用 `all-mpnet-base-v2`（Reimers & Gurevych 2019）算 sentence embedding，做 cosine similarity，把最相似 instruction 的 trace 作为 in-context example 加进 prompt。

这是个很朴素但有效的 self-distillation 思路——用 LLM 自己的成功经验当 few-shot example。我注意到 ReAct-RAG 在 Table 2 里的成功率（0.71）反而略低于 ReAct（0.73），但 sim steps 更高。这说明 RAG 引入的 example trace 可能干扰了 LLM 在新任务上的 reasoning。这跟 Pang et al. 2024、Madaan et al. 2024 的 self-refine 路线相关，但 PARTNR 的发现是 RAG 在 multi-agent 协作场景下不一定提升主指标。

### 5.5 Finetuned LLM（Appendix A.10）

这是最有意思的部分。作者用 LoRA（Hu et al., 2021, ICLR）在 Llama-3.1-8B 上 fine-tune：

**LoRA 配置**：
- rank $r = 132$
- $\alpha = 128$
- dropout = 0.01
- 只在 $W^V$ 和 $W^Q$ 上加 adapter

这里 $r$ 是低秩矩阵的秩，$\alpha$ 是 scaling factor（实际加的扰动是 $\frac{\alpha}{r} \cdot BA$，所以 $\alpha/r = 128/132 \approx 0.97$，接近 1）。只调 V 和 Q 是 LoRA 论文里建议的"够用又省"配置，E 和 O 不动。

**数据构造**：
- 从 ReAct baseline 收集 success trace（success rate 1.0 的 episode）
- 把每个 episode 拆成单步 (world_graph, action_history) → action 对
- 过滤掉失败的 action
- 如果 episode 中途 success 但最终 fail，截断到 success 点，最后一个 action 替换成 `Done[]`
- 最终 1,226 episodes，15,889 training samples

**训练设置**：
- 4 × A100 GPU，batch size 2/GPU
- 40,000 steps，约 24 小时

**关键发现（Table 9）**：只用 ReAct 数据 fine-tune 效果最好（success 0.70），加入 Heuristic-Expert 数据反而下降（0.69 或 0.63）。这有点反直觉——用 GT evaluation function 训练的 expert trace 理论上信息更丰富，但可能因为它 "太完美"，distill 出来的 model 在面对 partial observation 下的 noise 时泛化不好。这跟 Hsieh et al. 2023 的 "distilling step-by-step" 发现一致——分布外的 trace 反而是更好的 teacher。

---

## 6. 实验结果深度解读

### 6.1 Table 2 主结果

| Row | Method | Controllability | Skills | Observability | Sim Steps | Success | PC | Plan Cycles |
|-----|--------|-----------------|--------|---------------|----------|---------|----|----|
| a | ReAct-Single | Single | Oracle | Partial | 2519 | 0.73 | 0.85 | 18.68 |
| b | Heuristic-Expert | Centralized | Oracle | Full | 1260 | 0.84 | 0.94 | N/A |
| c | ReAct | Centralized | Oracle | Full | 1347 | 0.74 | 0.88 | 17.49 |
| d | ReAct | Centralized | Oracle | Partial | 2298 | 0.74 | 0.85 | 20.73 |
| e | ReAct | Decentralized | Oracle | Partial | 3295 | 0.73 | 0.86 | 15.24 |
| f | ReAct+RAG | Decentralized | Oracle | Partial | 3467 | 0.71 | 0.84 | 14.75 |
| g | Finetuned | Decentralized | Oracle | Partial | 3228 | 0.70 | 0.84 | 12.85 |
| h | ReAct | Decentralized | Learned | Partial | 6494 | 0.57 | 0.76 | 22.72 |
| i | ReAct | Decentralized | Learned | ConceptGraph | 12490 | 0.30 | 0.56 | 23.84 |

**关键观察**：

1. **Coordination burden**（row e vs. row a）：multi-agent decentralized 3295 steps > single-agent 2519 steps。**两个 LLM 协作反而比一个 LLM 单干慢**。这是个非常重要的发现——它说明当前 LLM 在 multi-agent coordination 上的能力是负的（额外开销超过了并行收益）。而人类 pair（multi-user）反而比 single human 更快（2369 < 3046），说明这个 coordination burden 是 LLM 特有的问题，不是任务本身的内在属性。

2. **Learned skills 灾难性下降**（row h vs. e）：success 从 0.73 跌到 0.57，sim steps 从 3295 涨到 6494。这表明 LLM **不会从 skill failure 中恢复**。当 pick 失败或 navigation 不完整，LLM 没有重新规划的能力，只是继续往下走。

3. **Non-privileged perception 是真正的瓶颈**（row i）：success 从 0.57 跌到 0.30，sim steps 涨到 12490。这说明当前 LLM 极度依赖准确的 world graph，对 perception error 几乎零容忍。ConceptGraphs 的 misclassification（shelf 被认成 table）会让 LLM 完全跑偏。

4. **Finetuned 8B ≈ ReAct 70B**（row g vs. e）：success 0.70 vs. 0.73，几乎持平。但 inference 速度 8.6 倍快。这验证了 distillation 在这个任务上的有效性。

### 6.2 Table 3 Human-in-the-loop 结果

| Setting | SR | PC | Sim Steps | Task Offloading | Expl. Eff. | Extraneous |
|---------|----|----|-----------|-----------------|------------|------------|
| Single-user | 0.93 | 0.96 | 3046 | N/A | 2459 | 0.09 |
| Multi-user | 0.93 | 0.96 | 2369 | 0.59 | 1762 | 0.15 |
| Human-ReAct | 0.91 | 0.96 | 4267 | 0.16 | 2624 | 0.12 |
| Human-Finetuned | 0.92 | 0.96 | 3443 | 0.26 | 2164 | 0.13 |

**几个非常 sharp 的发现**：

1. **人类 93% vs. LLM 30%**——3 倍 gap。这是 LLM planning 能力当前水平的真实标尺。

2. **Human pair 比 single human 快 22%**（2369 vs. 3046），但 **Human-LLM pair 比 single human 慢 13%**（3443 vs. 3046 for finetuned；4267 vs. 3046 for ReAct）。LLM 是个 **negative coordination partner**——它拖累了人类。

3. **Finetuned 比 ReAct 在 HITL 下更优**：sim steps 3443 < 4267，task offloading 26% > 16%。即使 automated eval 下两者差不多，HITL 下 finetuned 更好。原因是 inference latency：Finetuned 8B 推理快，人类等待时间短，体验流畅。ReAct 70B 每步推理 4.55s，整个 episode 平均 36 分钟，人类等不起。

这个 latency 的重要性其实在 robotics 部署里被严重低估。我自己的直觉是：**HITL 场景下，inference latency 和 task success rate 几乎同等重要**。一个 95% 准确但 10s/step 的 planner 不如一个 85% 准确但 0.5s/step 的 planner。

### 6.3 Task-type breakdown（Table 13）

| Method | Constraint-free | Spatial | Temporal | Heterogeneous |
|--------|----------------|---------|----------|---------------|
| Decentralized | 0.82 | 0.82 | 0.60 | 0.66 |
| Centralized | 0.84 | 0.85 | 0.59 | 0.66 |
| Single | 0.85 | 0.81 | 0.58 | 0.68 |

Temporal 任务 success 比 constraint-free 低 27%，heterogeneous 低 20%。这两类是 LLM 的真正弱项。Temporal 要求 **state tracking across time**，LLM 在长 rollout 上会丢失中间状态。Heterogeneous 要求 **theory of mind**（推理 partner 的 capability），LLM 没有显式建模 partner。

### 6.4 Coordination 分析（Table 12）

| Setting | Task Offloading | Extraneous Effort | Exploration Eff. |
|---------|-----------------|-------------------|------------------|
| Decentralized | 0.596 | 0.21 | 994 |
| Centralized | 0.49 | 0.04 | 684 |
| Single | - | 0.047 | 1121 |

**Extraneous effort 暴涨 300%**（decentralized 0.21 vs. single 0.047）。这意味着 decentralized LLM 30% 的 action 是浪费的——重复 partner 做过的事，或者做了无关的 rearrange。这是个非常具体的 coordination failure 量化指标。

我的联想：这个 extraneous effort 的概念其实可以推广到所有 multi-LLM-agent 系统。我怀疑在大部分 LLM-based multi-agent 框架（AutoGen、CrewAI、Camel）里都存在类似的"无意义 actions"问题，只是没人量化。PARTNR 提供了一个很好的 metric template。

---

## 7. ConceptGraphs 集成（Appendix A.7.2）

非特权感知用 modified ConceptGraphs（Gu et al., 2024, ICRA）。pipeline 改造：
- 用 Meta-CLIP（ViT-H-14-quickgelu, metaclip_fcc）做 object 分类
- YOLO（yolov8x-worldv2）+ SAM 做分割
- LLaMA-3.1-70B 做 room label 标注（基于 10 个最近 furniture 的 categories）
- 不用 LLaVA / GPT 做 open-vocab 名称（用 CLIP embedding 直接分类 closed vocab）

这种工程选择很有意思——它把 ConceptGraphs 从 open-vocab 改成 closed-vocab，是为了和 PARTNR 的 155 个 object type 对齐。这暴露了一个深层矛盾：open-vocab perception system 在 closed-world benchmark 上不一定更好，因为它会检测出 benchmark 里没定义的物体，反而增加 LLM 的混淆。

Room annotation prompt 很有意思——它让 LLM 看一个 furniture 的 10 个最近邻居，然后输出 7 类 room label 之一（bedroom / living_room / kitchen / dining_room / hallway / bathroom / unknown）。这是个非常简单的 spatial reasoning 任务，但实验显示这个步骤的错误率不低，导致后续 planner 误判。

---

## 8. Learned Low-level Skills（Appendix A.8）

两个 base skill：

**Navigate skill**：
- Observation: arm depth camera (224×171, hFOV 55°) + relative target pose (2D polar)
- Action: linear/angular velocity ∈ [-10, 10] m/s
- Reward: forward + orientation bonus + success bonus + collision penalty + slack
- 训练: DD-PPO（distributed）

**Manipulation skill**：
- Observation: depth + 3D Cartesian relative pose + 7-dim arm joint + holding detector + end-effector relative pose
- Action: linear/angular base velocity + delta arm joint angles ∈ [-5e-2, 5e-2] + binary grasp
- Reward: arm move toward target + success + slack
- 训练: DD-PPO

基于这两个 base skill，组合出 Navigate / Explore / OpenFurniture / CloseFurniture / PickObject / PlaceObject。

我注意到 manipulation skill 的 action space 有 11 维（base 2 + arm 7 + grasp 1），这在 RL 上是中等难度，但需要 sim 速度足够快。Habitat 3.0 的 DD-PPO 分布式训练能跑到这个规模，但训练这些 skill 本身就是几天的 GPU 时间。

---

## 9. 一些我没有完全搞清楚的问题

1. **ReAct-Tools 的表现**（Table 10）：ReAct-Tools 在 partial obs 下 success 0.71，比 ReAct 的 0.73 略低。这意味着让 LLM **主动调用 perception tool**（FindObjectTool 等）反而比直接给它 summarised world graph 更差。这跟我的直觉相反——我以为 active perception 会更好。可能的原因是 LLM 不会很好地决定 "什么时候该 query"，反而 query 太多浪费 plan cycles。

2. **Finetuned 在 test set 上更差**（Table 11）：val 上 Finetuned success 0.70，test 上 0.51。但 ReAct 在 test 上是 0.63。**Finetuned 在 test 上 generalization 不如 ReAct**。这暗示 LoRA fine-tune 在 distribution shift 下会 overfit train set 的 task pattern。但 HITL 下（Table 14）Finetuned 又比 ReAct 好。所以这是个矛盾：automated eval 上 ReAct 更好，HITL 上 Finetuned 更好。我的解释是 latency——HITL 下人类能适应 LLM 的错误，但不能适应 latency。

3. **Task offloading 在 LLM pair 里很低**：Human-ReAct 只有 16%，Human-Finetuned 26%，而 Human-human 是 59%。这意味着 LLM 实际只承担了 1/6 到 1/4 的工作。这个数字看起来很低，但考虑到 LLM 30% 的 success rate，其实它是合理——LLM 接手的部分它做不好，所以人类倾向于自己做。

---

## 10. 我自己的联想与延伸思考

### 10.1 跟其他工作的关系

- **Habitat 3.0**（ICLR 2024, https://aihabitat.org/habitat3/）是底层 sim。PARTNR 是 Habitat 3.0 之上的一个 benchmark layer。
- **ALFRED**（CVPR 2020, https://askforalfred.com）是 single-agent 语言的早期标杆，PARTNR 在某种意义上是 multi-agent + LLM-native 的 ALFRED。
- **Overcooked-AI**（Carroll et al., 2019, https://github.com/HumanCompatibleAI/overcooked_ai）是 multi-agent coordination 的经典 2D benchmark，PARTNR 是 3D + language 版。
- **CoELA**（Zhang et al., ICLR 2024）是 multi-agent LLM 协作的早期工作，PARTNR 的 baseline 部分借鉴了它的设计哲学，但任务多样性大得多。
- **RoboCasa**（RSS 2024, https://robocasa.ai）也是 LLM-generated task 的工作，但 single-agent。
- **VirtualHome**（Puig et al., CVPR 2018, http://virtualhome.org）是早期的 household activity simulation，PARTNR 继承了它的 activity 描述思路但用 LLM 替代 templated programs。

### 10.2 跟 SayPlan、ConceptGraphs 的关系

PARTNR 的 world graph + tool 调用架构跟 SayPlan（Rana et al., CoRL 2023, https://sayplan.github.io）非常接近。SayPlan 用 3D scene graph 做 scalable robot task planning。PARTNR 把这个思路扩展到 multi-agent，并加入了 LLM-as-planner 的 ReAct 框架。

ConceptGraphs（Gu et al., ICRA 2024, https://concept-graphs.github.io）是 open-vocab 3D scene graph，PARTNR 把它当作 non-privileged perception 的 backend。这个组合很有意思——ConceptGraphs 提供感知，world graph 提供结构化 grounding，LLM 提供 reasoning。三者组合后 success 30%，比 privileged 的 73% 低很多，说明 perception error 是当前 LLM-based embodied agent 的最大 bottleneck。

### 10.3 跟 VLA / VLM 的关系

PARTNR 当前只用 LLM（text-only）作为 planner，world graph 是文本化的。但未来显然应该有 VLA（vision-language-action）或 VLM 接入。我自己的直觉是：把 Llama-3.2-Vision 或 GPT-4o 直接接入，让 VLM 看 RGB 而不是 text-summarized graph，可能会部分解决 non-privileged perception 下的 0.30 → 0.73 gap。但 VLM 的 inference latency 比 LLM 更高，HITL 下不一定划算。

### 10.4 关于 LLM planning 能力的判断

PARTNR 给出的最 sharp 的数据是：**human 93% vs. LLM 30% under no privileged info**。这是个 3 倍 gap。但我注意到一个细节：在 privileged + full obs + centralized 下（row c），ReAct 能到 0.74，接近 Heuristic-Expert 的 0.84。这说明 LLM 的 reasoning 本身不差，差的是 **grounding + coordination + recovery**。如果我能把 perception 和 skill 做 robust，LLM planning 的上限其实可以接近人类。

### 10.5 关于 multi-LLM-agent 的反思

decentralized 下两个 LLM 一起工作比单个 LLM 慢，这个发现对整个 multi-agent LLM 领域是个 warning。当前很多 multi-agent LLM 框架假设"多 LLM 协作 > 单 LLM"，但 PARTNR 数据显示这个假设在 embodied long-horizon 任务下不成立。我的猜测是：在短 horizon + 信息完全共享的 chat 场景下 multi-LLM 有用，但在长 horizon + partial observability + 实物 action 的场景下，coordination overhead 会超过 parallelism 收益。

---

## 11. 一些可能的延伸方向

1. **Theory of mind module**：让 LLM 显式建模 partner 的 state（"它在做什么，它看到什么，它接下来会做什么"）。当前 decentralized 下 extraneous effort 300% 增长就是缺这个。

2. **Skill failure recovery**：把 skill failure signal 作为 prompt 一部分反馈给 LLM，让它重新规划。当前 row h → row e 的 success 跌幅 16% 全是因为这个。

3. **VLM-based perception 替代 ConceptGraphs**：ConceptGraphs 的 room label 错误率太高，VLM 可能更 robust。

4. **Active perception with VLM**：ReAct-Tools 显示 active perception 不一定好，但如果用 VLM 做 grounded active perception（"看一下桌子那边的物体"），可能更好。

5. **Iterative self-improvement with HITL data**：HITL 数据本身就是 distillation 的金矿。当前 fine-tune 只用 ReAct trace，未来可以用 human-LLM pair trace 做 DAgger-style imitation。

6. **Hierarchical planning with explicit task division**：当前 LLM 在 temporal + heterogeneous task 上弱，可能需要一个 explicit high-level task divider（把 task 拆给 human 和 robot）+ low-level executor 的两级架构。

---

## 12. 总结

PARTNR 是个非常 solid 的 benchmark 工作。它的核心贡献：

1. 把 multi-agent embodied AI 的语言化、规模化、多样化拼到一起，规模 100k 任务。
2. 半自动 LLM 生成 pipeline，关键在 simulation-in-the-loop grounding + 人工 quality gate + scale via in-context prompting。
3. Predicate-based evaluation system，支持 temporal + spatial + heterogeneous 约束。
4. 系统性的 LLM baseline 评估，揭示了 coordination burden、skill failure recovery、perception robustness 三个核心 bottleneck。
5. HITL 评估揭示 LLM 是 negative coordination partner，但 fine-tuned small model 可以部分缓解 latency 问题。

我自己的 takeaway 是：当前 LLM 在 embodied multi-agent 上的真实水平是 30% success（在没特权时），主要瓶颈是 **grounding robustness** 和 **multi-agent coordination**。Fine-tuning small model 是工程上最 actionable 的方向，但要从 30% 推到 90% 还需要 perception 和 coordination 上的根本性改进。

---

## 参考链接

- PARTNR 项目主页: https://aihabitat.org/partnr
- PARTNR 代码: https://github.com/facebookresearch/partnr-planner
- Habitat 3.0: https://aihabitat.org/habitat3/
- Habitat 3.0 论文: https://arxiv.org/abs/2402.04235
- ALFRED: https://askforalfred.com
- ConceptGraphs: https://concept-graphs.github.io
- SayPlan: https://sayplan.github.io
- RoboCasa: https://robocasa.ai
- HSSD: https://huggingface.co/datasets/hssd/hssd-data
- OVMM (HomeRobot): https://homerobot.github.io
- CoELA: https://github.com/THUDM/CoELA
- LoRA: https://arxiv.org/abs/2106.09685
- ReAct: https://react-lm.github.io
- Llama 3: https://arxiv.org/abs/2407.21783
- Overcooked-AI: https://github.com/HumanCompatibleAI/overcooked_ai
- VirtualHome: http://virtualhome.org
- BDDL: https://behavior.stanford.edu/behavior-1k
- PrediViz 工具: 论文 Appendix A.6.2 描述
- Grammar-constrained decoding: https://arxiv.org/abs/2305.13971
