---
source_pdf: embodied agent interface.pdf
paper_sha256: d10c7fee3594c88ff9f6a061a68b64a380a82d36c6652f3b08c42179044e2a9f
processed_at: '2026-08-04T03:26:53-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好, 用人话讲就是: 这篇paper在给LLM做"体检", 而且是分科室的那种体检, 不是给个总分让你自己猜哪里有病。

## 一句话总结

以前大家用LLM做机器人, 跑完任务看成功率, 高了就吹, 低了就重来。这篇paper说: 不行, 你得知道LLM到底在哪个环节掉链子 — 是听不懂人话, 还是规划动作顺序错了, 还是搞不清动作的precondition, 还是理解不了环境状态变化。于是作者搞了一套标准化"体检套餐", 把embodied decision making拆成四个独立模块, 每个模块都能单独测, 单独打分, 单独诊断。

## 为什么需要这套体检

之前领域里有个尴尬: 你说GPT-4o做embodied agent成功率60%, 我说Claude做同样任务成功率70%, 但这俩数字根本没法比。因为:

你的60%可能是: LLM只负责生成action sequence, 其他module用的是传统planner
我的70%可能是: LLM负责所有事情, 从理解goal到planning到transition modeling

你这叫什么比较? 就像比两个人谁身体好, 一个只跑了800米, 另一个跑了铁人三项, 然后你拿心率比。没意义。

更糟的是, 就算都是60%失败, 你也不知道为啥失败 — 是LLM把goal理解错了, 还是action顺序排错了, 还是没搞清动作precondition? 不知道原因就没法改进。

## 三个核心创新, 用大白话

### 创新一: 用LTL当"通用语言"

LTL就是Linear Temporal Logic, 线性时序逻辑。听起来吓人, 其实就是一套能同时描述"状态要求"和"先后顺序"的formal language。

比如"做饭"这个任务, 你可以写成:
```
准备好食材 then 点火 then 放食材 then 等熟 then 装盘
```

这个`then`就是关键 — 它说"前面那步必须先发生"。普通的goal state formula只能说"最后食物要熟, 要在盘子里", 但没法说"得先开火再放菜"。

更妙的是, LTL还能表达"选择" — 用∨(或)连接。比如"用抹布擦冰箱"还是"用海绵擦冰箱", 都行, LTL能写出两种path都合法的spec。

**为啥这个统一语言重要**: 因为BEHAVIOR simulator用BDDL格式描述goal, VirtualHome用自然语言, 两套完全不同的representation。没法跨simulator比较LLM能力。现在大家都翻译成LTL, 就像大家都说英语, 沟通无障碍了。

### 创新二: 四个模块独立体检

作者把"LLM做embodied agent"这件事拆成四个根本能力:

**模块1 - Goal Interpretation: 听懂人话**
- 输入: "用抹布擦冰箱"
- 输出: `not_stained(fridge.97) ∧ soaked(rag.0) ∧ ...` 这种symbolic goal
- 测的是: LLM能不能把自然语言grounding到环境里的具体object和state

**模块2 - Subgoal Decomposition: 拆解目标**
- 输入: symbolic goal
- 输出: 一串中间状态, 比如 `抹布在水槽旁 → 水槽开了 → 抹布湿了 → 水槽关了 → 冰箱开了 → 冰箱干净了`
- 测的是: LLM能不能declaratively把大目标拆成有逻辑顺序的小目标
- 评估方法: 用BFS search把这些state转成action sequence, 再看action sequence能不能execute

**模块3 - Action Sequencing: 排动作顺序**
- 输入: goal + transition model
- 输出: 一串action, 比如 `[GRASP(rag), WALK(sink), TOGGLE_ON(sink), SOAK(rag), TOGGLE_OFF(sink), OPEN(fridge), CLEAN(fridge)]`
- 测的是: LLM能不能imperatively排好每一步动作

**模块4 - Transition Modeling: 理解动作后果**
- 输入: action name, 比如`SOAK`
- 输出: PDDL preconditions和effects
  - pre: `holding(?obj) ∧ next_to(?sink, ?agent) ∧ toggled_on(sink)`
  - eff: `soaked(?obj)`
- 测的是: LLM能不能当"world model builder", 写出动作的precondition和effect

这四个模块对应MDP的四个核心component: goal input, trajectory output (两种形式), transition model。非常elegant的decomposition。

### 创新三: 细粒度错误分类

这可能是最实用的部分。以前失败就失败了, 现在失败要分类:

**Grammar Errors (语法错误)**:
- Parsing error: 输出格式不对
- Hallucination: action name或object name不存在
- Predicate-Arg Num: 参数个数错了, 比如`grab`只需要1个参数你给了2个

**Runtime Errors (运行时错误)** — 这个最有诊断价值:
- **Affordance error**: 你要`open(shelf)`, 但shelf根本不能开
- **Missing step**: 你要`release(book)` 但book从来没被grab过 — 前面漏了关键步骤
- **Wrong order**: 你先`release(book)` 再`grab(book)`, 顺序反了
- **Additional step**: 你又`toggle_on(light)` 但light已经on了, 多此一举

有了这种分类, 你就能精准定位LLM的问题。比如o1-preview在BEHAVIOR上的missing step error只有6%, 但GPT-4o有36%, 说明o1在precondition reasoning上强很多。这种insight是单纯success rate给不了的。

## 几个最让我兴奋的实验发现

### 发现一: o1-preview碾压, 但有个反转

在BEHAVIOR上, o1-preview avg performance 74.9%, 第二名Claude-3.5 Sonnet只有64.2%, gap巨大。原因: BEHAVIOR任务平均14.6步, 是long-horizon planning, 需要深度推理。o1的reasoning tokens让它能"想更多", 所以在long-horizon上优势明显。

但! 在VirtualHome上, Mistral Large和Gemini 1.5 Pro在action sequencing上居然超过o1-preview。因为VirtualHome任务虽然object多, 但action sequence只有3-5步, 是short-horizon。这种情况下"想太多"反而会被无关object干扰, 直觉型模型反而更准。

这个发现的implication: **不是所有任务都需要最强LLM**。Short-horizon任务用"想得少"的模型反而更好, long-horizon才需要o1这种深度reasoning。这对system design很有指导意义。

### 发现二: LLM有"报道偏差"

任务"serve a meal", ground truth是 `ontop(chicken, plate) ∧ ontop(plate, table)` — 鸡肉在盘子上, 盘子在桌子上。

但所有LLM都预测 `ontop(chicken, table)` — 鸡肉直接在桌子上。

为啥? 因为人类日常说"把鸡肉放桌上"时, 语境上默认鸡肉在盘子里。但在物理机器人执行时, 这种隐含信息至关重要。LLM学了太多日常对话, 把这种"理所当然"的spatial relationship给丢了。

这种"reporting bias"是LLM做embodied agent的systematic问题 — 日常语言省略的信息, 在物理执行时不能省。

### 发现三: Subgoal Decomposition不比Action Sequencing简单

直觉上, subgoal decomposition (拆成中间状态) 应该比action sequencing (排具体动作) 简单, 因为状态更抽象, 约束更少。

但实验发现两者难度相当! 因为在abstract action space里, 要declaratively拆出feasible的state sequence, LLM必须理解state之间的依赖关系 — 这跟理解action之间的依赖关系一样难。

这个发现打破了"先拆subgoal再搜action"这个pipeline的效率假设。如果subgoal decomposition的难度跟action sequencing一样, 那这个pipeline并没有simplify问题。

### 发现四: Replanning超有用

GPT-4o在BEHAVIOR上, 给3次replan机会:
- 不replan: 65.2% task success
- replan: 77.4% task success
- 提升12个百分点!

而且对stochastic action (动作有失败概率) 更夸张:
- 动作失败率20%, 不replan: 10% success
- 动作失败率20%, replan: 65% success
- 提升55个百分点!

这说明LLM其实有能力从错误feedback里学习并调整plan, 只是之前没人给它机会。这是一个非常low-hanging的fruit — 加个replan loop就能大幅提升performance。

### 发现五: Action Space一致性陷阱

这个发现非常subtle但重要。用GPT-4o预测的transition model + PDDL planner, 能找到plan。用ground truth transition model + planner, 也能找到plan。

但! 如果你混用 — 拿GPT-4o预测的`plug_in` + ground truth的`walk_towards`和`switch_on`, planner就找不着解了!

原因: GPT-4o的`plug_in`漏了`has_switch`的case, 但它的`switch_on`也不要求`plug_in` precondition, 所以两个不完整的定义互相compensate, 整体action space是自洽的。一旦你插入ground truth的`switch_on` (要求`plug_in`), 就break了。

**启示**: LLM生成的action definitions有内在self-consistency。用的时候要么全用LLM的, 要么全用ground truth的, 不要混用。这对modular system design是个重要警告。

### 发现六: LLM在Subgoal Decomposition和Transition Modeling上超越人类!

对比GPT-4o和人类标注:
- Goal Interpretation: 人80.6 F1, GPT-4o 37.6 — 人完胜
- Action Sequencing执行: 人85.7%, GPT-4o 57.1% — 人完胜
- Subgoal Decomposition: 人60%, GPT-4o 70% — **GPT-4o胜**
- Transition Modeling planner: 人66.7%, GPT-4o 100% — **GPT-4o完胜**

这非常有意思。LLM在short-horizon、需要精确物理理解的task上不如人类, 但在long-context logical reasoning和大scale scene graph tracking上反而更强。这暗示LLM的优势在"处理大量信息做logical reasoning", 而非"精细的物理直觉"。

## 对未来的启示

### 1. LLM在physical reasoning上有systematic gaps
- 不懂commonsense preconditions (开箱子才能拿里面的东西)
- 长序列state tracking会失败 (忘了前面已经做过什么)
- 有reporting bias (丢失了日常语言省略的物理细节)
- 对quantifier (forall/forpairs) 推理很弱

### 2. 系统设计要selective use LLM
- 不是所有module都该用LLM。Goal interpretation和transition modeling用LLM有优势
- Action sequencing在short-horizon上, 弱模型反而可能更准
- Modular composition有error accumulation, 但replanning可以mitigate

### 3. Action space一致性
- 如果用LLM生成transition model, 要么全用LLM的, 要么全用ground truth, 不要混用
- LLM生成的action definitions有内在self-consistency, 被打断会出问题

### 4. Replanning是low-hanging fruit
- 简单的error feedback + replan loop就能提升10%+ success
- 对stochastic environment提升更显著 (从10%到65%)
- 但要小心action over-generation

### 5. VLM还需要很长的路
- 纯VLM end-to-end做planning几乎不可用 (9.1% F1)
- 加上scene graph后接近LLM水平
- 当前VLM在long-horizon reasoning上能力不足
- 但LLM上的improvement strategy可以transfer到VLM

## 最后一句

这篇paper的价值不在于提出新method, 而在于提供了一个**诊断框架**。它告诉我们: 不要再只看success rate了, 要知道LLM在哪里、为什么失败。只有诊断清楚了, 才能对症下药。这就像从"我感觉不舒服"进化到"血常规显示白细胞高, 可能是感染" — 医学的进步不在于药多, 而在于诊断精准。Embodied AI也需要这种精准诊断。

参考链接:
- 项目主页: https://embodied-agent-interface.github.io/
- 代码: https://github.com/embodied-agent-interface/embodied-agent-interface/
- 数据: https://huggingface.co/datasets/Inevitablevalor/EmbodiedAgentInterface
- BEHAVIOR benchmark: https://behavior.stanford.edu/
- VirtualHome: https://virtual-home.org/
- o1 technical overview: https://openai.com/o1/

---

# Embodied Agent Interface: 深度解析

Andrej, 这篇paper的核心问题非常清晰: **当前LLM用于embodied agent的评价都是一团乱麻** — 不同domain,不同input-output spec,最后只看一个success rate,完全无法诊断LLM到底在哪个环节出问题。作者们做的事情本质上是为这个领域建立一个"可分解的诊断框架"。

项目主页: https://embodied-agent-interface.github.io/
代码: https://github.com/embodied-agent-interface/embodied-agent-interface/
数据: https://huggingface.co/datasets/Inevitablevalor/EmbodiedAgentInterface

## 1. 核心设计直觉: 把embodied decision making映射到MDP

最关键的intuition藏在Appendix B.3里。作者把整个问题映射到Markov Decision Process ⟨U, S, A, M, R, g⟩:

- **U**: universe of objects (环境中所有物体的集合)
- **S**: state space,每个state s = ⟨U, F⟩,F是relational Boolean features的集合
- **A**: action space,每个action a = ⟨name, args⟩
- **M: S × A → S**: deterministic transition function
- **R: S × A × g → ℝ**: reward function,R(s, a, g) = 1 if eval(g, s) = 1
- **g**: goal specification

这个MDP视角下,LLM在embodied agent中需要承担四个fundamental ability,这正好对应四个module:

| MDP component | 对应的ability module |
|---|---|
| Goal input | Goal Interpretation 𝒢 |
| Trajectory output (actions) | Action Sequencing 𝒬 |
| Trajectory output (states) | Subgoal Decomposition Φ |
| Transition model 𝓜 | Transition Modeling 𝒯 |

这种分解的好处是: 你可以isolate单个module用LLM,其他module用ground truth,这样就能精准诊断"LLM在哪个环节拉胯"。

## 2. LTL (Linear Temporal Logic) 作为统一接口

这是paper最巧妙的设计。为什么不用first-order logic on goal states,也不用reward function? 

**First-order logic的局限**: 只能描述final state的要求,无法表达"先做A再做B"这种temporal ordering。

**Reward function的局限**: 虽然表达力强,但numeric nature让它无法compact representation。

**LTL的优势**: 既 expressive又 compact,可以统一描述 state constraints, action constraints, 和 temporal ordering。

### 2.1 LTL语法定义

paper用的是LTL的一个fragment,定义在finite trajectory上:

$$\phi := p \mid \neg\phi \mid \phi_1 \wedge \phi_2 \mid \phi_1 \vee \phi_2 \mid \phi_1 \Rightarrow \phi_2 \mid \forall x\,\phi(x) \mid \exists x\,\phi(x) \mid \exists^{=n} x\,\phi(x) \mid (\phi) \mid \phi_1 \text{ then } \phi_2$$

变量解释:
- $p$: atomic proposition,可以是state proposition (如 ontop(book1, chair1)) 或 action proposition (如 touch(cat))
- $\phi, \phi_1, \phi_2$: LTL formulas
- $\neg, \wedge, \vee, \Rightarrow$: 标准逻辑连接词
- $\forall x, \exists x$: 一阶逻辑量词
- $\exists^{=n} x$: 计数量词,"恰好有n个x满足φ(x)"
- **then**: 关键的temporal connective

注意:**then**是standard LTL里"next"和"eventually"的组合替代。paper没有用"globally"和"until",因为这个fragment已经足够描述所有task spec。

### 2.2 LTL语义

LTL formula本质上是一个trajectory classifier。给定state-action trajectory $\bar{T} = [s_0, a_1, s_1, \ldots, a_n, s_n]$, $T_i = (s_i, a_i)$:

对于state formula φ (不含then):
$$eval(\phi, \bar{T}) = \exists t. \phi(s_t)$$
意思是"eventually"目标在某一步被满足。

对于then连接的formula:
$$eval(\phi_1 \text{ then } \phi_2, \bar{T}) = \exists k. \phi_1(\bar{T}_{\leq k}) \wedge \phi_2(\bar{T}_{>k})$$

这里 $\bar{T}_{\leq k}$ 是前k个state-action pair (prefix), $\bar{T}_{> k}$ 是k之后的suffix。直观理解: 存在一个分割点k,使得前半段满足φ₁,后半段满足φ₂。

### 2.3 一个具体例子

任务"browse Internet"的subgoal plan用LTL表达:
```
ontop(character, chair) 
then holds_rh(character, mouse) ∧ holds_lh(character, keyboard) 
then facing(character, computer)
```

这个formula清晰地编码了: 先坐下 → 然后双手拿鼠标键盘 → 然后面对电脑。三段时序关系一目了然。

### 2.4 为什么LTL是game changer

考虑BEHAVIOR的BDDL goal: `(forall (?jar ?apple) (inside ?apple ?jar))`
和VirtualHome的自然语言: "take everything out of the fridge, throw anything outdated..."

这两个simulator原本用完全不同的goal representation,无法统一评估。LTL把它们都映射到同一套formalism,这样:
1. 可以用同一个metric (F1 set matching) 衡量不同module的输出
2. 可以cross-simulator比较LLM能力
3. 可以表达alternative goals (用∨) 和temporal ordering (用then)

## 3. 四个能力模块的深度拆解

### 3.1 Goal Interpretation 𝒢: ⟨s₀, l_g⟩ → g

**输入**: initial state $s_0$ + 自然语言指令 $l_g$
**输出**: LTL goal $\hat{g}$

这个module本质上是"语言到符号的grounding"。比如 "use the rag to clean the refrigerator" 要被翻译成具体的object IDs和state conditions:
- fridge (ID: 97)
- rag (ID: 0)  
- not_stained(fridge.97)
- soaked(rag.0)
- 等等

**Evaluation metric**用F1 set-matching:
$$F_1(g, \hat{g}) = \max_{g_i \in \mathcal{G}(g, \mathcal{U})} F_1(set(g_i), set(\hat{g}))$$

变量解释:
- $g$: ground truth goal
- $\hat{g}$: LLM预测的goal
- $\mathcal{G}(g, \mathcal{U})$: 把复杂goal $g$ 翻译成一组simple LTL goals $\{g_0, g_1, \ldots, g_k\}$,每个$g_i$都entail $g$
- $set(g)$: 把goal $g = a_1 \text{ then } \ldots \text{ then } a_k \text{ then } (p_1 \wedge \ldots \wedge p_\ell)$ 转成集合 $\{\{a_i\}_{i=1}^k\} \cup \{p_i\}_{i=1}^\ell$,action序列作为一个整体元素

**关键发现**: LLMs普遍把intermediate subgoal当成final goal。比如VirtualHome的"Drink"任务,GPT-4o预测了 `open(freezer)` 和 `inside(water, glass)` 作为final goal的一部分 — 这些其实是中间步骤,不是最终目标。这暗示LLM在内部做chain-of-thought时,把reasoning的中间状态泄漏到了输出里。

### 3.2 Subgoal Decomposition Φ: ⟨s₀, g⟩ → φ̄

**输入**: initial state + LTL goal
**输出**: subgoal sequence $\bar{\phi} = \{\phi_i\}_{i=1}^k$,每个$\phi_i$是LTL formula

这个module输出declarative states(声明性状态),imperative actions(命令性动作)。比如:
```
next_to(rag.0, sink.82) → toggled_on(sink.82) → soaked(rag.0) → toggled_off(sink.82) → open(fridge.97) → not_stained(fridge.97)
```

**Evaluation challenge**: 没有统一的reference decomposition。解决方案是用BFS把subgoal sequence refine成action sequence,然后用action sequencing的metrics评估。

subgoal-action mapping函数 $\mathcal{AM}(\bar{\phi}, s_0)$:
```
Initialize empty state-action sequence t̄
Set s_curr = s_0
For each subgoal φ_i in φ̄:
    Perform BFS from s_curr to find s_goal satisfying φ_i
    Extract path from s_curr to s_goal, append to t̄
    Set s_curr = s_goal
Return t̄
```

**关键发现**: Subgoal decomposition并不比action sequencing简单! 因为LLM必须declaratively break goals into feasible steps,这要求它理解state之间的依赖关系。在abstract action space里,declarative decomposition的难度和imperative sequencing相当。

### 3.3 Action Sequencing 𝒬: ⟨s₀, g⟩, 𝓜 → ā

**输入**: initial state + goal + transition model
**输出**: action sequence $\bar{a} = \{a_i\}_{i=1}^n$

这个module的evaluation有两个维度:

**Trajectory feasibility**: action sequence能否在simulator里执行。如果不可执行,错误分为:
- **Grammar errors**: Parsing error, Hallucination (action/object/predicate name不存在), Predicate-Arg Num (参数个数不对)
- **Runtime errors**: 
  - **Affordance error**: 物体属性不允许该action (如 open(shelf),shelf不能开)
  - **Additional step**: action的effect已经在当前state满足 (如 toggle_on(light) 但light已经on)
  - **Missing step**: precondition未满足且从未在历史state中满足过 (如 release(book) 但book从未被grasp)
  - **Wrong order**: precondition未满足但曾经在历史state中满足过 (说明action顺序错了)

**Goal satisfaction**: 执行完ā后,用 $eval(g, \bar{T})$ 检查goal是否满足。还有partial success rate:
$$\text{PartialSucc}(\bar{a}, g) = \max_{g_i \in \mathcal{G}(g, \mathcal{U})} \text{PartialSucc}(\bar{a}, g_i)$$

### 3.4 Transition Modeling 𝒯: ⟨s₀, g⟩, o → ⟨pre, eff⟩

**输入**: task + operator definition set $\{o_i\}$
**输出**: PDDL operator definition,包括precondition和effect

这个module本质上让LLM当"world model builder"。比如对于action `soak`:
- **Precondition**: holding(?obj1) ∧ next_to(?sink, ?agent) ∧ toggled_on(?sink)
- **Effect**: soaked(?obj1)

**Evaluation**有两种:
1. **Logic form accuracy**: 用bipartite matching比较predicted和ground truth的pre/eff。对conjunctive formulas,就是set matching;对复杂formulas (如 ∀x.φ(x)) 递归评估。

2. **Planner success rate**: 用PDDL planner (Fast Downward) 基于LLM生成的operator definitions来planning,看能否找到plan。

**关键发现**: LLMs tend to overstate object states in effects while understating them in preconditions; 反之,overstate spatial relationships in preconditions and understate them in effects。这导致即使planner找到plan,plan在真实环境里可能不可行。

## 4. 实验数据深度分析

### 4.1 整体性能对比 (Table 3关键数据)

| Model | Goal Interp (B) | Action Seq Task SR (B) | Subgoal Task SR (B) | Transition Planner SR (B) | Avg Perf (B) |
|---|---|---|---|---|---|
| o1-preview | 81.6 | 81.0 | 57.0 | 89.0 | 74.9 |
| Claude-3.5 Sonnet | 82.7 | 60.0 | 39.0 | 82.0 | 64.2 |
| GPT-4o | 79.2 | 47.0 | 49.0 | 67.0 | 59.8 |
| Gemini 1.5 Pro | 79.6 | 42.0 | 31.0 | 39.0 | 48.8 |
| Llama 3 70B | 70.9 | 34.0 | 20.0 | 78.0 | 48.1 |
| Llama 3 8B | 28.3 | 10.0 | 21.0 | 29.0 | 23.1 |

o1-preview在BEHAVIOR上碾压,74.9% vs第二名64.2%。这个gap非常显著。原因:o1的reasoning tokens让它能"think more"应对long-horizon planning (BEHAVIOR平均14.6步)。

**但有趣的是**: 在VirtualHome上,Mistral Large (73.4%, 83.6%) 和 Gemini 1.5 Pro (73.1%, 83.3%) 在action sequencing上超越了o1-preview (71.1%, 78.4%)。作者推测: VirtualHome task虽然object多但action sequence短(3-5步),需要的是"抓住关键信息"而非"深度推理",所以"想太多"的模型反而会被无关object干扰。

### 4.2 错误分布 (Table 6 BEHAVIOR数据)

对Action Sequencing的runtime errors:
| Model | Wrong Order | Missing Step | Affordance | Additional Step |
|---|---|---|---|---|
| Claude-3.5 Sonnet | 1.0 | 25.0 | 1.3 | 2.0 |
| GPT-4o | 1.0 | 36.0 | 2.3 | 0.0 |
| o1-preview | 2.0 | 6.0 | 2.0 | 3.0 |

o1-preview的missing step只有6.0%,远低于GPT-4o的36.0%。这说明o1的深度推理确实帮它更好地处理precondition。

### 4.3 Goal Interpretation的precision/recall分解 (Table 4)

对Relation Goal在BEHAVIOR上:
| Model | Precision | Recall | F1 |
|---|---|---|---|
| o1-preview | 78.4 | 82.9 | 82.7 |
| Claude-3.5 Sonnet | 83.1 | 81.3 | 82.9 |
| GPT-4o | 78.6 | 78.5 | 79.8 |

**Reporting bias的体现**: "serve a meal"任务,ground truth是 `ontop(chicken.0, plate.2) ∧ ontop(plate.2, table.1)`,但所有LLM都预测 `ontop(chicken.0, table.1)`。因为自然语言里说"put the chicken on the table"是conversationally acceptable的,但物理上chicken应该在plate上,plate在table上。这种omission导致recall下降。

### 4.4 Transition Modeling的category breakdown (Table 7)

BEHAVIOR上的F1 scores:
| Model | Object States | Spatial Relations | Non-Spatial Relations |
|---|---|---|---|
| o1-preview | 78.3 | 56.3 | 83.5 |
| Claude-3.5 Sonnet | 78.8 | 58.6 | 73.6 |
| GPT-4o | 71.3 | 45.9 | 73.0 |

Non-spatial relations是最难的。在VirtualHome上,最好的o1-preview也只有11.8% F1! 原因: 非空间关系(如holding)涉及complex logic和corner cases。比如predict `grab`的precondition时,很少有LLM会想到 "not in a closed container" 或 "both hands are empty" 这种条件。

### 4.5 Replanning的效果 (Table 17)

GPT-4o在BEHAVIOR上:
- Without replanning: Task SR 65.2%, Execution SR 71.8%
- With replanning (最多3次): Task SR 77.4%, Execution SR 83.3%

提升超过10%! 但Additional step error从0.0%升到0.7% — replanning有时会导致action over-generation。

对stochastic actions (Table 18)更显著:
| Fail Prob | Method | Execution SR |
|---|---|---|
| 0.2 | w/o replanning | 10% |
| 0.2 | w/ replanning | 65% |

失败概率20%时,replanning把execution SR从10%拉到65%,提升55个点!

## 5. 最值得玩味的发现

### 5.1 Action space consistency的陷阱

Figure 33-35展示了一个非常微妙的现象。如果用GPT-4o预测的`plug_in` + ground truth的`walk_towards`和`switch_on`,PDDL planner找不到解。但用全GPT-4o预测的或全ground truth的,都能找到解。

原因: GPT-4o的`plug_in`只处理`has_plug`的情况,漏了`has_switch`。但GPT-4o的`switch_on`也不要求`plug_in` precondition,所以两个"不完整"的定义互相compensate,整体action space是consistent的。一旦混入ground truth的`switch_on`(要求`plug_in`),就break了。

这个发现对实际系统设计非常重要: **LLM-generated的action definitions有内在的self-consistency,不要随意和human-written的混用**。

### 5.2 Pipeline vs Modularized (Table 16)

G+Q pipeline在BEHAVIOR上:
- Modularized: Task SR 47.0%, Execution SR 53.0%
- Pipeline-based: Task SR 42.0%, Execution SR 55.0%

Execution SR相近(53 vs 55),但Task SR下降(47 vs 42)。这说明: Goal interpretation的错误会propagate到下游,但trajectory的executable性主要由action语义决定,不受goal interpretation影响太多。

### 5.3 人类对比 (Table 19)

| Method | Goal Interp F1 | Action Seq Exec SR | Subgoal Task SR | Transition Planner SR |
|---|---|---|---|---|
| GPT-4o | 37.6 | 57.1 | 70.0 | 100.0 |
| Human | 80.6 | 85.7 | 60.0 | 66.7 |

GPT-4o在Subgoal Decomposition和Transition Modeling上超越人类! 但在Goal Interpretation和Action Sequencing执行上远不如人类。这暗示LLM擅长long-context的logical reasoning,但short-horizon的精细物理执行不如人类。

### 5.4 VLM vs LLM (Table 20, 21)

| Model | Goal Interp F1 | Action Seq SR |
|---|---|---|
| Llama 3 (scene graph) | 31.5 | 11.1 |
| LLaVA (no scene graph) | 9.1 | 2.5 |
| LLaVA (with scene graph) | 25.8 | 11.0 |

纯VLM end-to-end几乎不可用(9.1% F1)。加上scene graph后接近LLM水平。这说明: **当前VLM在long-horizon planning上还远不够好,perception和decision-making entangled会complicate diagnosis**。但LLM上的improvement strategy可以transfer到VLM(Exp.3→Exp.4显示相同pattern)。

## 6. 对未来研究的启示

### 6.1 LLM在physical reasoning上的systematic gaps

- **Precondition hallucination**: LLMs经常忽略commonsense preconditions,如"open the box before fetching items inside"
- **State tracking over long horizon**: trajectory越长,missing step error越多
- **Reporting bias**: conversationally elided的信息(如plate在table和chicken之间)对physical precision至关重要
- **Quantifier reasoning**: BEHAVIOR的`forall`/`forpairs`让state/relation goal success rate显著下降

### 6.2 系统设计启示

1. **Selective use of LLMs**: 不是所有module都适合LLM。Goal interpretation和transition modeling上LLM有优势,但action sequencing在short-horizon上可能不需要最强LLM。

2. **Modular composition**: pipeline会有error accumulation,但modularized + replanning可以mitigate。

3. **Action space consistency**: 如果用LLM生成transition model,要么全用LLM的,要么全用ground truth的,不要混用。

4. **Replanning is powerful**: 简单的error feedback + replanning就能提升10%+ success rate,这是低-hanging fruit。

### 6.3 我的思考

这个benchmark最valuable的地方是**diagnostic granularity**。以前我们只知道"LLM做embodied agent成功率不高",现在可以精确说"o1-preview在BEHAVIOR的missing step error只有6%,但GPT-4o有36%,差距在precondition reasoning"。

这种diagnostic能力对引导research direction至关重要。比如:
- 如果missing step是主要问题,应该focus on precondition learning
- 如果additional step是主要问题,应该focus on state tracking memory
- 如果affordance error多,应该focus on object property grounding

paper的limitation也很honest: 没有visual input,没有low-level control,没有geometric reasoning。这些都是future work。但作为一个symbolic层面的diagnostic framework,它已经做得非常complete了。

最后,从Karpathy你关心的"build intuition"角度: 这个paper的核心intuition是 **"要诊断LLM在embodied AI上的能力,必须先把问题decompose到MDP的fundamental modules,然后用formal language (LTL) 统一interface,最后用fine-grained metrics定位error type"**。这种decompose-diagnose的methodology比单纯的end-to-end success rate有用得多,因为它告诉我们"接下来该improve什么"。

参考链接:
- BEHAVIOR benchmark: https://behavior.stanford.edu/
- VirtualHome: https://virtual-home.org/
- LTL原始paper (Pnueli 1977): https://dl.acm.org/doi/10.1109/SFCS.1977.32
- Fast Downward planner: https://www.fast-downward.org/
- PDDL: http://editor.planning.domains/
- o1 technical overview: https://openai.com/o1/
- SayCan (相关work): https://say-can.github.io/
- Code as Policies: https://code-as-policies.github.io/
- LLM+P: https://arxiv.org/abs/2304.11477
