---
source_pdf: LaMMA-P.pdf
paper_sha256: 749b64c27371a4b01a0de98533c7b1a6f203f6350c736696cc38c1bc62d5c628
processed_at: '2026-08-05T11:40:34-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LaMMA-P

## 一句话总结

让 GPT-4o 当"项目经理"，把一句模糊的人话拆成小任务分给不同机器人，然后每个小任务交给一个老派的确定性 planner 去算具体动作序列，算不下去就让 GPT 再改一遍。

---

## 这论文到底在解决什么问题

家里有俩仨机器人，人对它们喊一句："帮我把鸡蛋放冰箱，顺便关个灯，再切个苹果"。

这句话听着简单，但对机器人系统来说有一堆麻烦：

1. **鸡蛋在哪儿？冰箱在哪儿？** 需要把"鸡蛋"这种模糊词映射到环境里的具体 object。
2. **谁去拿鸡蛋？谁去关灯？** 不同机器人技能不同，可能 Robot 1 没法开冰箱门，得 Robot 2 先开。
3. **关灯和切苹果能不能同时干？** 能并行就并行，省时间。
4. **从"走到鸡蛋→拿起→走到冰箱→打开冰箱→放进去→关门"** 这么长的 action chain，任何一步错了后面全崩。

之前人们试过两条路：

- **纯 LLM 路线**（SMART-LLM 这类）：让 GPT 直接吐出一串 action。简单任务还行，任务一长就开始 hallucinate，第 8 步忘了第 3 步干了啥，格式也容易乱。
- **纯 PDDL 路线**：用 1998 年的 planning language 把任务写成形式化问题，交给 Fast Downward 这种经典 planner 算。完全可靠，但完全听不懂人话，每个新任务都得人手写 problem file，而且根本没法处理"顺便"这种模糊措辞。

LaMMA-P 的核心 idea 就是——**让 LLM 干它擅长的（听人话、拆任务、分配），让 PDDL planner 干它擅长的（保证 action 序列正确）**。中间用 PDDL 文件当接口。就这么简单。

---

## 六个模块的实际工作流

拿"把鸡蛋放盘子上"举例：

### 第 1 步：Precondition Identifier (P)

GPT 被喂一个 few-shot prompt，输出类似：

```
Subtask: Put egg on plate
- GoToObject: robot 走到 egg 旁边
  - 需要: robot 当前没在干活
  - 效果: robot 在 egg 旁边
- PickupObject: robot 拿起 egg
  - 需要: robot 在 egg 旁边, robot 手是空的
  - 效果: robot 拿着 egg
- PutObject: robot 把 egg 放到 plate 上
  - 需要: robot 拿着 egg, robot 在 plate 旁边
  - 效果: egg 在 plate 上
```

这一步的关键 trick 是**把每个 action 的 precondition 简化**。原始 PDDL operator 可能要求 5 个 precondition 同时成立，LLM 看太多约束容易晕，所以只保留最核心的 2-3 个，让 LLM 推理更顺。论文里把这包装成一个概率公式，其实意思就是"LLM 在更宽松的约束下生成更靠谱的 action 序列"。

### 第 2 步：Task Allocator

GPT 看一眼每个 robot 的技能清单，把 sub-task 分下去。"Robot 1 你去拿鸡蛋，Robot 2 你去关灯，能并行的就并行"。

### 第 3 步：Problem Generator (G)

GPT 把每个 robot 分到的 sub-task 翻译成一个 PDDL problem 文件。比如 Robot 2 的：

```lisp
(define (problem put-egg-on-plate)
  (:domain robot2)
  (:objects Robot2 Egg Plate Loc1 Loc2 - object)
  (:init
    (at Robot2 InitLoc)
    (at-location Egg Loc1)
    (at-location Plate Loc2))
  (:goal (at-location Egg Plate)))
```

这一步 LLM 经常出错——比如忘了写 `(not (holding Robot2 Egg))` 这种 negative literal，或者 hallucinate 出不存在的 object。所以下一步专门 catch 这些。

### 第 4 步：PDDL Validator (V)

一个解析器，检查 problem file 格式对不对、object 类型对不对、init/goal 里的 predicate 是不是在 domain 里定义过。错了就退回去让 GPT 重写。

### 第 5 步：Fast Downward Planner

把验证过的 problem 文件喂给经典 planner。Fast Downward 用 delete-relaxation heuristic 做 A* 搜索，保证返回的 action 序列在 PDDL 语义下 100% valid。如果 planner 报告"unsolvable"（比如 init state 写错了导致 goal 不可达），又退回去让 GPT 改 problem file。

**这一步是 LaMMA-P 比 SMART-LLM 强的根本原因**——SMART-LLM 直接相信 GPT 吐的 action 序列，LaMMA-P 不信，必须过 planner 这道关。

### 第 6 步：Sub-Plan Combiner

每个 robot 都有自己的 sub-plan，现在要把它们合并成一个全局 schedule。GPT 看一眼哪些可以并行（Robot 1 拿鸡蛋 和 Robot 2 关灯 互不干扰），哪些必须串行（Robot 2 必须等 Robot 1 打开抽屉才能放东西进去）。

合并完用 regex 把 plan 转成 AI2-THOR 仿真器能执行的 Python API 调用。结束。

---

## 为什么这招有效

### 核心洞察

LLM 的 hallucination 不可怕，**可怕的是没人检查**。

纯 LLM 路线里，GPT 说"已经把鸡蛋放冰箱了"，系统就信了。但实际上可能 GPT 第 5 步就忘了 robot 手里还拿着鸡蛋。没人查，错就错到底。

LaMMA-P 的设计是——**让 LLM 在一个被严格约束的"沙箱"里干活**。LLM 输出的不是 action 序列，是 PDDL problem 描述。这个描述对不对，Validator 查一次，Planner 查第二次。两次都过了才执行。错了就反馈给 LLM 重写。这本质上是一个 closed-loop 的 self-correction。

### 实验数据说话

看 Complex 任务（6+ sub-task，强制依赖）：

| 方法 | Success Rate |
|---|---|
| CoT (GPT-4o) 直接生成 | **0%** |
| SMART-LLM (GPT-4o) | 20% |
| LaMMA-P (GPT-4o) | **77%** |

Vague Command 任务（模糊指令）：

| 方法 | Success Rate |
|---|---|
| 所有 baseline | **0%** |
| LaMMA-P (GPT-4o) | 45% |

注意 baseline 在 vague task 上全军覆没。GPT-4o 直接生成 action 序列时，遇到"随便弄点吃的"这种指令完全抓瞎。LaMMA-P 之所以还能拿 45%，是因为 PDDL 框架强制 LLM 把模糊指令 concrete 化成结构化 goal——就算指令模糊，goal 必须写成 `(at-location Egg Plate)` 这种确定形式，逼着 LLM 去推断。

### 小模型的惊喜

LaMMA-P 用 Llama 3.1-8B 在 Complex 任务上拿到 0.15 SR，只比 SMART-LLM 用 GPT-4o 的 0.20 差一点。**8B 小模型 + 好架构 ≈ 大模型 + 差架构**。这说明 LaMMA-P 的架构本身承担了大量 reasoning burden，对 LLM 大小的依赖降低了。这对实际部署很重要——你可以在机器人上跑小模型，不用调 GPT-4o API。

---

## Ablation 告诉我们什么

逐步加组件看 SR 变化（Complex 任务）：

- 什么都没有，纯 LLM：**0.10**
- + 预定义的 robot domain (D)：**0.52** ← 最大跃迁
- + Problem Generator (G)：0.58
- + Validator (V)：0.68
- + Precondition Identifier (P)：**0.77**

**D 单项贡献最大**。意思是——如果你只能做一件事，就把每个机器人的技能写成结构化 schema 喂给 LLM。这比任何 prompt trick 都管用。

直觉上这很合理：LLM 不知道 Robot 1 能干啥、不能干啥，就只能瞎猜。一旦给它一个明确的 skill list，它分配任务就靠谱了。

---

## 这论文的弱点

1. **PDDL domain 得人手写**。每个机器人的每个 skill 都要写成 PDDL operator，precondition 和 effect 全部明确。这是巨大的 engineering effort，跟"LLM zero-shot"的卖点矛盾。

2. **完全假设环境静态且全可观**。鸡蛋位置必须提前告诉系统，不能靠相机识别。真实场景里 state estimation 出错整个 plan 就崩。

3. **Vague command SR 只有 45%**。绝对数字还是很低，模糊指令处理远未解决。

4. **Sub-Plan Combiner 还是靠 LLM**。如果 GPT 把并行/串行关系搞错，可能死锁。这模块没单独 ablation。

5. **执行时没有 re-planning**。如果某个 action 在仿真器里执行失败，没有 reactive 机制重新规划。

6. **重试次数上限没说**。LLM 反复生成 invalid problem 怎么办？论文没测 robustness。

---

## 对你的 intuition 有什么用

如果你将来要做 LLM + robotics 的系统，这论文给的关键 takeaway：

1. **LLM 别直接输出 action，输出 problem description，让外部 solver 求解**。这是 neuro-symbolic 的核心 pattern。

2. **Validator + planner 失败信号是天然的 feedback loop**。比让 LLM 自己 "reflect" 靠谱得多，因为 solver 的反馈是 ground truth。

3. **结构化的 robot skill schema 比任何 prompt engineering 都值钱**。先把 skill API 写清楚，再调 LLM。

4. **多 agent 场景的并行/串行判断仍然依赖 LLM**，这是个 open weak point，值得改进。

5. **小模型 + 好 architecture 能逼近大模型**。如果你要在 edge device 部署，architecture design 比模型大小重要。

---

## 相关工作如果你感兴趣

- **LLM+P** (Liu et al. 2023) - LaMMA-P 的 single-agent 前身，思路一模一样但只 1 个 agent
- **SMART-LLM** (Kannan et al. 2024) - 主要 baseline，纯 LLM 路线
- **TwoStep** (Singh et al. 2024) - 2 agent 版本，LaMMA-P 是它的泛化
- **SayCan** (Google 2022) - 用 affordance grounding 选 action
- **Code as Policies** (Google 2022) - LLM 直接生成 robot code
- **PlanBench** (ASU 2022) - 证明纯 LLM 在 planning 上 reliability 极差

项目网站: https://lamma-p.github.io
SMART-LLM: https://arxiv.org/abs/2407.09092
LLM+P: https://arxiv.org/abs/2304.11477
AI2-THOR: https://ai2thor.allenai.org/

---

# LaMMA-P: Language Model-Driven Multi-Agent PDDL Planner 深度解析

## 一、核心动机与 intuition

这篇论文要解决的核心问题是 **long-horizon multi-robot task allocation + planning**。在 household 场景里，异构机器人团队（heterogeneous robot team）需要协作完成一个跨越很多 step 的复合任务，比如 "准备一份带鸡蛋的餐点并关掉灯" —— Robot 1 擅长 pick/place，Robot 2 擅长 toggle，需要把任务分下去并且并行化执行。

纯 LLM planning 的问题在于：**长 horizon 下 hallucination 累积、action 序列格式漂移、不可验证**；纯 PDDL planner 的问题在于：**无法处理 vague natural language instruction、domain 固化、不能泛化到新任务**。LaMMA-P 的核心 insight 就是——**让 LLM 做 semantic decomposition + task allocation + precondition simplification（擅长的事），让 Fast Downward 做 guaranteed-valid symbolic planning（擅长的事），中间用 PDDL 作为 interface**。这个思路本质上和 [LLM+P (Liu et al. 2023)](https://arxiv.org/abs/2304.11477) 是同一血脉，但 LaMMA-P 把它扩展到 **arbitrary number of heterogeneous agents**，这是关键贡献。

参考链接：
- LLM+P: https://arxiv.org/abs/2304.11477
- LLM-as-Planner survey: https://arxiv.org/abs/2402.01817
- Fast Downward: https://www.fast-downward.org/
- AI2-THOR: https://ai2thor.allenai.org/

---

## 二、Problem Formulation 细节

任务被 formalize 为 cooperative Multi-Agent Planning (MAP) task：

$$\langle \mathcal{AG}, \mathcal{D}, \{A^i\}_{i=1}^n, \mathcal{P}, \mathcal{T}, \mathcal{G} \rangle$$

变量含义：
- $\mathcal{AG}$: agent 集合，共 $n$ 个 agent
- $\mathcal{D}$: domain 集合，每个 agent $i$ 在自己的 domain $d_i \in \mathcal{D}$ 内操作（异构性的来源）
- $A^i$: agent $i$ 可用的 action 集合（不同 robot 技能不同）
- $\mathcal{P}$: world state 的原子（atom / ground predicate）集合
- $\mathcal{T} \subseteq \mathcal{P}$: initial state（注意论文里 $\mathcal{T}$ 和 $\mathcal{I}$ 混用，原文公式里写 $\mathbb{Z}$ 和 $\mathcal{T}$，是 typo，应当理解为 init state）
- $\mathcal{G} \subseteq \mathcal{P}$: goal state

解的形式：

$$\Pi_g = \langle \Delta, \prec \rangle$$

- $\Delta \subseteq A$: 被选中的 actions
- $\prec$: actions 之间的偏序关系（partial order），允许并行执行

这里 partial-order plan 的表示方法值得一提——它比 total-order 更接近真实多机器人调度，因为不同 robot 的 action 在时间轴上可以交错。

---

## 三、架构图解析（Fig. 2）

LaMMA-P 是一个 **六模块 pipeline**，每个模块都是一次 LLM call 或 PDDL tool call：

```
Natural Language Instruction
        │
        ▼
┌──────────────────────────┐
│ 1. Precondition Identifier (P)│  ← LLM: few-shot prompt
│    识别 sub-task + 每个 action 的 P_a, E_a          │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│ 2. Task Allocator             │  ← LLM: 把 sub-task 分给 robot
│    依据 robot skill set 分配                       │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│ 3. Problem Generator (G)       │  ← LLM: 输出 PDDL problem.pddl
│    生成 init state / goal / objects                │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│ 4. PDDL Validator (V)         │  ← 解析器: format check
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│ 5. Fast Downward / LLM Planner │  ← 经典 planner 求解
│    失败 → fallback LLM 重写                        │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│ 6. Sub-Plan Combiner          │  ← LLM: 合并并行/串行
│    + Plan-to-Code Converter (regex) → AI2-THOR      │
└──────────────────────────┘
```

这个设计的关键 intuition：**LLM 不直接生成最终 action 序列，而是生成 PDDL problem file，然后让 guaranteed-correct 的 planner 做最后一步求解**。LLM 的 hallucination 被限制在 "problem 描述" 层面，而 problem 描述错误可以被 Validator + Planner 失败信号捕获并反馈给 LLM 修复。这类似于 LLM+P 的 iterative correction loop，但引入了 multi-agent decomposition 层。

---

## 四、Precondition Identifier 的数学核心（最 interesting 的部分）

这是论文最值得细看的一节，但写得比较晦涩。我把它重新展开。

### 4.1 经典 relaxed plan heuristic

Fast Downward 用的是 delete-relaxation heuristic：

$$h(\mathcal{I}, \mathcal{G}) = \min_{\Pi \in \Pi(\mathcal{I}, \mathcal{G})} \left( \sum_{a \in \Pi} \text{cost}(a) \right)$$

- $\Pi(\mathcal{I}, \mathcal{G})$: 所有从 $\mathcal{I}$ 到 $\mathcal{G}$ 的 valid relaxed plan 集合
- "relaxed" 意味着 **ignore delete effects**——只看 action 的 add effects（新增的 atom），不考虑 action 让哪些 atom 变 false
- $\text{cost}(a)$: action $a$ 的 cost（通常 =1）
- 这个 $h$ 是 admissible heuristic 的基础（$h^*$ 的下界），用于 $A^*$ 类搜索

### 4.2 LLM 的隐式 "概率启发式"

LLM autoregressive sampling 在 action 空间上诱导一个分布：

$$p(a_1, \dots, a_n \mid \mathcal{I}, \mathcal{G}) = \prod_{i=1}^{n} p(a_i \mid a_1, \dots, a_{i-1}, \mathcal{I}, \mathcal{G})$$

- $a_1, \dots, a_n$: 生成的 action 序列
- 条件是 $(\mathcal{I}, \mathcal{G})$: 任务上下文
- 这是 chain rule 分解，LLM 本质上在 action token 序列上建模

### 4.3 Precondition Simplification

论文的核心 trick：**LLM 把每个 action 的 preconditions $P_a$ 和 effects $E_a$ 投影到子集**：

$$P'_a \subseteq P_a, \quad E'_a \subseteq E_a$$

然后让 LLM 在简化后的 precondition 上生成：

$$p(a \mid P'_a, E'_a, \mathcal{I}, \mathcal{G})$$

### 4.4 把 LLM 输出视作 heuristic

论文把这个 probability 下的期望 cost 类比为 relaxed plan heuristic：

$$\hat{h}(\mathcal{I}, \mathcal{G}) = \mathbb{E}\left[ \sum_a \text{cost}(a) \mid p(a_1, \dots, a_n \mid P'_a, E'_a, \mathcal{I}, \mathcal{G}) \right]$$

- $\hat{h}$: LLM 隐式提供的 "heuristic estimate"
- 关键 insight: LLM 的 in-context prior 大致对应了一种 "delete-relaxed common-sense cost"，这和 $h(\mathcal{I}, \mathcal{G})$ 在 spirit 上同构

**但这里我要 critically 补充**：这个公式其实是一种 metaphor 而非严格等价。LLM 输出的不是 admissible heuristic，而是 mode-seeking 的 high-probability 路径。论文用这个数学包装只是为了说 "LLM 的概率分布可以理解成另一种形式的 relaxed heuristic"。真正的算法 value 不在公式上，而在 **precondition simplification 让 LLM 不被过多 ground atom 约束束缚**——LLM 直接生成完整 PDDL problem 时常常 hallucinate 出 initial state 错误（论文也提到 "The generated initial state is often flawed"），所以先用 precondition list 做 grounding 再交给 Problem Generator。

直觉上你可以理解为：**Precondition Identifier 是一个 "soft parser"**，把 high-level instruction 映射到一个 sub-goal sequence，每个 sub-goal 有粗粒度的 precondition，类似 HTN (hierarchical task network) 的 method precondition，但是是 LLM soft 生成的。

---

## 五、PDDL Domain 与 Problem 文件结构

### Domain file 片段（pick up 例子）：

```lisp
(:action PickupObject
  :parameters (?robot - robot
               ?object - object
               ?location - object)
  :precondition (and
    (at-location ?object ?location)
    (at ?robot ?location)
    (not (inaction ?robot)))
  :effect (and
    (holding ?robot ?object)
    (not (inaction ?robot))))
```

每个 robot type 有自己的 domain，定义了它能执行的 operators。异构性体现在：Robot 1 的 domain 可能包含 `OpenObject`、`CloseObject`，而 Robot 2 没有——这就强制 Task Allocator 必须按 skill 分配。

### Problem file 例子：

```lisp
(define (problem prepare-plate-with-egg)
  (:domain robot2)
  (:objects Robot2 - robot
             Egg Plate Location1 Location2 - object)
  (:init (at Robot2 InitLocation)
         (at-location Egg Location1)
         (at-location Plate Location2)
         (not (inaction Robot2)))
  (:goal (and (at-location Egg Plate)
              (not (holding Robot2 Egg))
              (not (holding Robot2 Plate)))))
```

`(:goal ...)` 用 closed-world assumption：未在 init 列出的 atom 默认 false。这个 PDDL 标准假设很重要，因为 LLM 经常忘记某个 negative literal 是否需要明列——这是 Validator 要 catch 的典型错误。

---

## 六、MAT-THOR Benchmark

这是论文的第二个贡献。基于 [SMART-LLM benchmark](https://arxiv.org/abs/2407.09092) 扩展，使用 [AI2-THOR](https://ai2thor.allenai.org/) 仿真器。

- **70 个任务** 跨越 **5 个 floor plan**
- **2-4 个 robot**，技能不同
- 三个难度等级：
  - **Compound Tasks (30)**: 2-4 sub-tasks，每个 robot 独立可完成，可并行
  - **Complex Tasks (20)**: 6+ sub-tasks，单个 robot 不具备全部技能，必须协作（例如只有 Robot 1 能开抽屉，Robot 2 必须等）
  - **Vague Command Tasks (20)**: 自然语言指令含模糊表述，需 inference

### 评估指标定义：

- **SR (Success Rate)** = 成功执行任务数 / 总任务数
- **GCR (Goal Condition Recall)** = $|\text{achieved goals} \cap \text{ground truth goals}| / |\text{ground truth goals}|$
- **RU (Robot Utilization)** = 实际 transition count / ground truth transition count（衡量 plan 是否 "浪费" 步骤）
- **Exe (Executability)** = 可执行 action 比例，不管是否相关
- **Eff (Efficiency)** = ground truth time steps / 实际 time steps（>1 表示比理想慢）

**RU 和 Eff 仅在 success 上评估**，避免失败任务污染数据。

---

## 七、实验数据表深度解读

### Table I: 主结果

| Method | Compound SR | Complex SR | Vague SR |
|---|---|---|---|
| CoT (GPT-4o) | 0.32 | 0.00 | 0.00 |
| SMART-LLM (GPT-4o) | 0.70 | 0.20 | 0.00 |
| Ours (Llama 2-13B) | 0.36 | 0.05 | 0.00 |
| Ours (Llama 3.1-8B) | 0.45 | 0.15 | 0.00 |
| **Ours (GPT-4o)** | **0.93** | **0.77** | **0.45** |

几个 critical observation：

1. **Vague Command 上 baseline 全军覆没 (SR=0)**，CoT 和 SMART-LLM 完全无法处理模糊指令。LaMMA-P (GPT-4o) 仍能拿到 0.45 SR，说明 PDDL grounding 提供了 "scaffold" 让 LLM 把模糊指令 hard-ify 成结构化 problem。

2. **Complex 任务上 SMART-LLM 只有 0.20 SR**，因为 sub-task ≥6 且强制依赖，纯 LLM 调度崩溃。LaMMA-P 拿到 0.77，是 ~4x 提升。

3. **小模型 (Llama 3.1-8B) 在 Complex 上接近 SMART-LLM (GPT-4o)**：0.15 vs 0.20。这说明 LaMMA-P 的架构本身能 compensate 模型大小，PDDL 部分承担了 reasoning burden。

4. **Compound 任务上小模型反而不如 SMART-LLM (GPT-4o)**：0.45 vs 0.70。因为 compound 任务简单，LLM 直接生成 action 即可，PDDL overhead 反而拖累。这是个很有意思的 negative result——架构 complexity 在简单任务上有 tax。

5. **平均提升**: SR +105%, Eff +36% (vs SMART-LLM GPT-4o)。Eff 只提升 36% 是因为成功率大幅提升的同时，新解决的任务往往 plan 较长，拉低了效率均值。

### Table II: Ablation Study

逐步移除组件：
- **w/o P & V & G & D**: SR 0.50 / 0.10 / 0.10 — 移除所有结构，纯 LLM
- **+ D (pre-defined domain)**: 复杂任务 SR 从 0.10 → 0.52 — 最大跃迁，证明 **domain knowledge grounding** 是核心
- **+ G (Problem Generator)**: Complex SR 0.52 → 0.58 (轻微提升)
- **+ V (Validator)**: Complex SR 0.58 → 0.68 — Validator 显著降低 task failure
- **+ P (Precondition Identifier)**: Complex SR 0.68 → 0.77 — 最后的 push

**最重要的 ablation insight**：D（pre-defined PDDL domain）单独贡献最大。这意味着如果你只能加一个东西，应该是 **结构化的 robot skill domain description**，而不是 LLM prompt trick。

---

## 八、与相关工作的 positioning

### 8.1 vs LLM+P (Liu et al. 2023)

LLM+P 是 single-agent，把 natural language task 翻译成 PDDL problem 然后用 Fast Downward 求解。LaMMA-P 继承了这个思路，但加了：
- Multi-agent decomposition
- Precondition Identifier 模块
- Sub-Plan Combiner 处理并行/串行
- Iterative re-planning fallback

### 8.2 vs SMART-LLM (Kannan et al. 2024, IROS)

SMART-LLM 完全依赖 LLM 做端到端 task allocation + plan generation，没有 symbolic planner verify。优势是 zero domain engineering，劣势是 long-horizon 不可靠。

论文链接: https://arxiv.org/abs/2407.09092

### 8.3 vs TwoStep (Singh et al. 2024)

[TwoStep](https://arxiv.org/abs/2403.17246) 限制在 2 agent，LaMMA-P 支持任意 agent 数量，并且 PDDL planning 比 teacher-student prompting 更 rigorous。

### 8.4 vs CoP, SayPlan 等 LLM planner

[SayCan](https://say-can.github.io/) 用 affordance grounding；[Inner Monologue](https://innermonologue.github.io/) 用 feedback loop；它们都不引入 symbolic planner verify。LaMMA-P 在 "neuro-symbolic" 这条线上走得更远。

### 8.5 vs PlanBench (Valmeekam et al.)

[PlanBench](https://arxiv.org/abs/2206.10498) 显示纯 LLM 在 Blocksworld 等任务上 reliability 极差。LaMMA-P 的实验间接印证了这点：CoT baseline 在 Complex 任务上 SR=0。

---

## 九、Failure Mode 与 limitation

论文 conclusion 里诚实承认 **assumes fully observable, static environment**。但还有几个 implicit limitation：

1. **PDDL Domain 必须手工定义**：每个 robot 的 skill 需要写成 PDDL operator。这是 significant engineering effort，违背了 "LLM zero-shot" 的卖点。Future work 应该让 LLM 自动 generate domain file from robot API description——但这是另一个 open problem（参考 [Mahdavi et al. 2024](https://arxiv.org/abs/2407.12979)）。

2. **No perception loop**：完全依赖 symbolic state，真实场景下 state estimation error 会让 plan 失败。论文 future work 提到 "incorporating vision-language models"。

3. **Iterative re-planning 的最大重试次数**没有 ablation——如果 LLM 反复生成 invalid problem，loop 会终止于多少次？这个 robustness 没测。

4. **Vague command SR 仍只有 0.45**——这是绝对数字低，意味着模糊指令处理远未 solved。

5. **Sub-Plan Combiner 仍依赖 LLM**：合并 sub-plan 时如果 LLM 弄错了 parallel/sequential 关系，整个 plan 可能死锁。论文没有 ablation Combiner 模块。

6. **No dynamic re-planning during execution**：执行中如果某个 action 失败，没有 reactive re-plan 机制。

---

## 十、关键 takeaways for building intuition

1. **Neuro-symbolic 的 sweet spot**：LLM 负责 fuzzy → structured translation，symbolic planner 负责 guaranteed correctness。LaMMA-P 是这个 pattern 的 multi-agent extension。

2. **Precondition simplification 是个有意思的 trick**：减少 symbolic constraint 让 LLM 生成更顺畅，类似在 LLM+P 中加 "hint"，但形式化为概率子集。

3. **Domain knowledge 是最大杠杆**：ablation 显示 +D 单项就拿到主要 performance gain。意味着在 robotics 应用上，**先把 robot skill 写成结构化 schema** 比调 LLM prompt 更重要。

4. **Validator loop 是 robustness 关键**：LLM 错了不可怕，可怕的是错了没人 catch。Validator + planner 失败信号给 LLM 一个 closed-loop feedback。

5. **Heterogeneity 必须显式建模**：每个 robot 一个 domain file 是简单粗暴但有效的方案。未来如果 robot 类型很多，需要 ontology 而非 per-robot domain。

6. **PDDL 作为 LLM 之间的 interface language**：让多个 LLM call 之间通过 PDDL 而非 free text 通信，减少 hallucination 传播。

---

## 十一、相关延伸阅读

- [LLM+P](https://arxiv.org/abs/2304.11477) - 原始 LLM-PDDL 桥接工作
- [PlanBench](https://arxiv.org/abs/2206.10498) - LLM planning benchmark
- [SMART-LLM](https://arxiv.org/abs/2407.09092) - 主要 baseline
- [TwoStep](https://arxiv.org/abs/2403.17246) - 两 agent 版本
- [SayCan](https://say-can.github.io/) - affordance-based LLM robot control
- [Code as Policies](https://code-as-policies.github.io/) - LLM 直接生成 robot code
- [Inner Monologue](https://innermonologue.github.io/) - LLM + feedback
- [Fast Downward](https://www.fast-downward.org/) - PDDL solver
- [PDDL Editor](https://editor.planning.domains/) - 在线 PDDL 编辑器
- [AI2-THOR](https://ai2thor.allenai.org/) - 仿真环境
- [PROGPROMPT](https://progprompt.github.io/) - programmatic LLM prompt
- [LLM-Plan survey](https://arxiv.org/abs/2402.01817) - LLM planning 综述
- [Reasoning with LLMs as Planning](https://arxiv.org/abs/2305.15771) - LLM 推理 vs 规划
- [LLM agents survey](https://arxiv.org/abs/2308.11432) - LLM agent 综述
- [Reflexion](https://arxiv.org/abs/2303.11366) - LLM self-reflection loop，思想与 LaMMA-P 的 validator feedback 类似

---

## 十二、可以追问的 open question

1. 如果 PDDL domain 自动从 robot URDF + skill API 生成，性能会怎样？这正是 [Mahdavi et al. 2024](https://arxiv.org/abs/2407.12979) 探索的方向。

2.Validator 给 LLM 的 feedback 是 binary (valid / invalid)，如果传回 **具体 violated precondition**，会不会更快收敛？类似 [Reflexion](https://arxiv.org/abs/2303.11366) 的 verbal reinforcement。

3. 把 LaMMA-P 中的 PDDL planner 换成 **MCTS** 或 **Monte-Carlo sampling over LLM** (类似 [Tree of Thoughts](https://arxiv.org/abs/2305.10601)) 会不会在 vague command 上更强？

4. 多 robot 之间的 **communication protocol** 完全缺失——目前是 central planner 全局调度。如果改成 decentralized + LLM-based communication（类似 [Mandi et al. ROCO](https://arxiv.org/abs/2310.01524)），效率会不会更高？

5. PDDL 是 fully observable Markovian，扩展到 POMDP（部分可观）需要 belief state tracking——LLM 是否可以充当 belief state approximator？这是 [LLM as world model](https://arxiv.org/abs/2310.01072) 的 research direction。

6. 任务 horizon 进一步延长到 100+ steps 时，PDDL state space 爆炸 vs LLM context length 限制，哪个先崩？这是 long-horizon scaling 的关键问题。

---

希望这个解析帮助你 build 起对 LaMMA-P 的 intuition。如果你想更深入某个具体模块（比如 Precondition Identifier 的 prompt 设计，或者 Sub-Plan Combiner 如何决定 parallel/sequential），可以继续问。
