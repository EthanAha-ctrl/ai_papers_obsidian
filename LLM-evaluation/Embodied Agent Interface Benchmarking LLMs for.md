---
source_pdf: Embodied Agent Interface Benchmarking LLMs for.pdf
paper_sha256: d10c7fee3594c88ff9f6a061a68b64a380a82d36c6652f3b08c42179044e2a9f
processed_at: '2026-08-18T10:33:35-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我用人话讲一遍。

---

## 这篇paper到底在干啥

想象你在debug一个神经网络。早年我们只看test accuracy——"模型87%，好像还行"。但这没法告诉你**哪里坏了**。后来大家学会看loss curve、gradient norm、per-layer activation、confusion matrix。这时候你才发现："哦原来layer 7 dead了"或者"gradient在layer 12爆炸"。

**这篇paper就是对 embodied LLM 做同样的事**。

之前所有让LLM做机器人规划的工作（SayCan、Code as Policies、Voyager...）都只报告一个success rate。"GPT-4在BEHAVIOR上40%成功"。然后呢？然后没了。你完全不知道：

- 是LLM没听懂人话？
- 是LLM听懂了但plan顺序错了？
- 是LLM plan对了但忘了precondition？
- 是LLM连"open fridge之前fridge得是closed的"这种常识都没有？

这篇paper说：**够了，我们拆开看**。

---

## 他们怎么拆的

把"LLM做embodied agent"这件事切成四个独立小考：

**1. 翻译官**（Goal Interpretation）  
给LLM一句自然语言"把冰箱擦干净"，看它能不能翻译成机器能懂的形式化goal。这考的是语言理解。

**2. 切分者**（Subgoal Decomposition）  
给它一个最终goal，让它拆成中间milestone。比如"擦冰箱" → "先拿抹布 → 浸湿 → 走到冰箱旁 → 擦"。考的是抽象规划。

**3. 排序员**（Action Sequencing）  
给它goal，直接输出一串action。`GRASP(rag), WALK(sink), SOAK(rag), ...` 考的是具体执行规划。

**4. 物理学家**（Transition Modeling）  
给它一个action名字（比如`soak`），让它预测precondition和effect。"soak之前得holding对象、得在toggled-on的sink旁边；soak之后对象变soaked"。考的是world model。

**关键设计**：评估每个模块时，其他模块都用ground truth。这样错误不会传染——你看到的就是这个模块自己的能力上限。这跟neural network里做ablation study是一回事。

---

## 他们发现了啥

### 发现1：LLM经常把中间步骤当成最终目标

任务："drinking water"  
LLM预测的final goal里有：`open(freezer)`

兄弟，开冰箱只是手段，不是目的。LLM搞不清"goal"和"subgoal"的边界。它学到的是"看到drink water，文本里常出现open freezer"，就全塞进final goal了。

### 发现2：LLM学的是人话的shortcut，不是物理精确

任务："serve a meal"  
正确goal：`ontop(chicken, plate) ∧ ontop(plate, table)`  
所有LLM预测：`ontop(chicken, table)`

人话说"把鸡肉放桌上"——大家都懂，因为盘子是隐含的。但物理世界没有"隐含"——你不放盘子，鸡肉就掉地上了。LLM把language的省略当成了physical truth。

这个叫**reporting bias**：训练语料里大家都不啰嗦，但机器人不能不啰嗦。

### 发现3：LLM不理解action的precondition

任务："clean fridge"  
LLM输出：`CLEAN(fridge)`  
但CLEAN的precondition是"得holding cleaning tool"。LLM经常直接跳过这步。

更搞笑的pattern：agent坐在沙发上，LLM让它`WALK`——但你得先`STANDUP`啊。LLM不知道"坐着"这个状态是blocking的。这跟ChatGPT写代码忘`import`一个味儿——它学到的是surface pattern，不是dependency graph。

### 发现4：LLM记不住"已经做完了"

VirtualHome里很多任务的初始状态里，有些goal**已经满足了**。比如light已经on了。LLM还是会输出`TOGGLE_ON(light)`。

它学到的training data里plan都是从零开始的，所以它默认"啥都没干过"，输出一个完整plan。这就像一个always重置memory的agent。

### 发现5：subgoal分解竟然不比直接action sequencing简单

这反直觉。你以为是：先拆成小目标再执行，比直接跳到action简单。但数据显示两个一样难。因为LLM要在抽象state空间里搜，而state空间比action空间大得多——你可以做啥是有限的，但状态可以是啥是无限的。

### 发现6：o1-preview甩开所有人

在BEHAVIOR上：o1-preview **74.9%** vs 第二名Claude-3.5 Sonnet **64.2%**。差10个点。

原因：o1有hidden reasoning tokens。它可以在输出前"思考"——这种test-time compute对长horizon planning确实是杀器。embodied planning本质是search，search需要compute。o1把search放到了inference time。

这跟你最近常说的"reasoning model开始overfit to test-time compute"完全吻合。

### 发现7：错误的分类很有意思

把trajectory error分成四类：

- **Affordance错误**：让LLM `open(shelf)`——shelf根本不能open。物理常识不够。
- **Additional step**：goal已经满足了，LLM还输出动作。记不住现状。
- **Missing step**：precondition不满足，且历史上也没满足过。比如没grasp就release。
- **Wrong order**：precondition不满足，但历史上满足过。时序乱了。

BEHAVIOR上missing step是大头（36%），因为precondition严格。VirtualHome上additional step多（15.7%），因为初始状态里goal经常已经满足一半。

**这个分类的价值**：不同错误要不同fix。Missing step要多给LLM precondition hint；Additional step要让LLM能观察当前状态；Wrong order要加强temporal reasoning。一个success rate给不了这些信息。

---

## 为什么这个工作重要

我之前说"诊断式benchmark"是核心。再说一遍为什么。

之前的embodied AI evaluation像是个黑盒评分：你提交一个agent，系统告诉你"42% success"。你不知道是perception错了、还是planner错了、还是controller错了。你只能盲目调prompt、换model、加CoT，然后祈祷。

这篇paper打开黑盒。它告诉你：GPT-4o在"理解goal"上还行（79% F1），但在"规划action时漏precondition"上有36%的失败率。这下你知道该往哪使劲了——不是换更大的model，而是让prompt显式问"这个action的precondition是啥"。

类比一下：这就像当年从"test accuracy"进化到"confusion matrix + per-class metrics"。分类问题不再是"准不准"，而是"哪类容易错"。embodied AI现在也到了这个stage。

---

## 我对这篇paper的吐槽

它诊断了，但**没开药**。它告诉你LLM漏precondition，但没做实验证明"在prompt里加'list preconditions first'能降多少error"。诊断完应该有remediation experiments。

另外它整个benchmark是symbolic的——假设你有完美的scene graph、完美的object localization。真实机器人的perception会错，会hallucinate物体，会误判状态。这个noise会放大planning error。Appendix K讨论了VLM但主benchmark没纳入。

但作为"第一步：先搞清楚LLM到底在embodied setting哪里不行"，这paper够好了。它至少把"LLM做embodied agent"这件事从玄学变成了工程——可以系统地、模块化地debug了。

---

## 一句话总结

> 这篇paper不是告诉你"LLM能不能做机器人"，而是告诉你"LLM在哪个环节、以什么方式、为什么做不好机器人"。它是个CT scan，不是个治疗。

embodied AI评估正在从"看success rate"的dark age走向"模块化诊断"的现代阶段。这篇是那条路上的里程碑。

---

# Embodied Agent Interface: 深度技术讲解

Karpathy，这篇paper做的本质是为 LLM 在 embodied decision making 中提供一个**诊断式**的benchmarking framework。类似你之前讲过的"microscope"概念——不只看success rate这个标量，而是打开黑盒，看 LLM 到底在哪个环节、以什么方式失败。这很像神经网络早期debugging时只看test accuracy，后来才学会看gradient flow、attention pattern、per-layer activation statistics。Embodied AI 评估现在正在经历类似的成熟化过程。

---

## 1. 核心问题：为什么需要这个Interface?

### 1.1 现有评估的三大缺陷

现有 embodied LLM 工作（SayCan [[2]](https://arxiv.org/abs/2204.01691)、Code as Policies [[6]](https://arxiv.org/abs/2209.07753)、LLM+P [[7]](https://arxiv.org/abs/2304.11477)、Voyager [[35]](https://arxiv.org/abs/2305.16291)）存在三个标准化缺失：

**(1) Goal specification不统一**：BEHAVIOR用BDDL描述 state goals（如`not_stained(fridge)`），VirtualHome用temporally extended goals描述action ordering。同一个"clean fridge"任务，在不同simulator里success criteria完全不同。

**(2) Module边界模糊**：不同framework假设LLM做不同的事情。Code as Policies让LLM做action sequencing；LLM+P让LLM做goal interpretation + PDDL planning；Ada [[8]](https://arxiv.org/abs/2312.08566) 让LLM做transition modeling生成PDDL domain。这些工作不可比。

**(3) Metrics过粗**：只看最终success rate，不知道是goal理解错了、还是action排序错了、还是transition model错了。

### 1.2 论文的三大贡献

```
EMBODIED AGENT INTERFACE
├── (1) Standardized goal specification (LTL-based)
├── (2) Four formalized ability modules (G, Φ, Q, T)  
└── (3) Fine-grained error taxonomy
    ├── Grammar errors: parsing / hallucination / arg-num
    └── Runtime errors: affordance / missing-step / 
                       additional-step / wrong-order
```

这个分层让我想起你做nanoGPT时的经验：要理解model behavior，必须把pipeline拆开，单独评估每个component。Embodied agent其实也是一系列 module composition，需要**modular evaluation**。

参考：[embodied-agent-interface.github.io](https://embodied-agent-interface.github.io/)

---

## 2. LTL作为统一接口：为什么这是关键设计选择

### 2.1 LTL的formal定义

论文用的是LTL的一个**fragment**（on finite trajectories）：

$$\phi := p \mid \neg\phi \mid \phi_1 \wedge \phi_2 \mid \phi_1 \vee \phi_2 \mid \phi_1 \Rightarrow \phi_2 \mid \forall x\, \phi(x) \mid \exists x\, \phi(x) \mid \exists^{=n} x\, \phi(x) \mid (\phi) \mid \phi_1 \text{ then } \phi_2$$

变量含义：
- $p$：atomic proposition（state predicate如`ontop(book1, chair1)` 或 action predicate如`touch(cat)`）
- $\phi, \phi_1, \phi_2$：LTL formulas
- $\neg, \wedge, \vee, \Rightarrow$：标准逻辑连接词
- $\forall, \exists$：一阶逻辑量词
- $\exists^{=n}$：counting quantifier，表示"恰好存在$n$个$x$满足条件"
- `then`：时序连接词，论文自创的简写

注意论文**没用** standard LTL 的 `next` / `eventually` / `globally` / `until`，而是用`then`作为复合算子，更贴合task planning的实际需求。

### 2.2 `then`的语义：trajectory切分

这是论文最优雅的地方。`then`的本质是把trajectory切成两段：

$$eval(\phi_1 \text{ then } \phi_2, \bar{T}) = \exists k.\, \phi_1(\bar{T}_{\leq k}) \wedge \phi_2(\bar{T}_{>k})$$

其中：
- $\bar{T} = [s_0, a_1, s_1, \ldots, a_n, s_n]$ 是state-action trajectory
- $\bar{T}_{\leq k}$：前$k$个state-action pairs
- $\bar{T}_{> k}$：从$k+1$开始的suffix

**Intuition**：`then`说"存在某个切分点，使得前半段轨迹满足$\phi_1$，后半段轨迹满足$\phi_2$"。这其实就是`eventually`的连续复合。比如：

```
browse Internet:
  ontop(character, chair) then 
  (holds_rh(character, mouse) ∧ holds_lh(character, keyboard)) then 
  facing(character, computer)
```

这描述了三阶段的时序约束：先坐椅子 → 再双手持设备 → 最后面向电脑。

### 2.3 为什么LTL而不是reward function？

我想强调一个subtle的point。Reward function $R: S \times A \times g \to \mathbb{R}$ 表达力更强，但有几个问题：
1. **不可组合**：两个reward functions相加不等于两个goals的合取
2. **不可解释**：dense reward无法告诉你"哪一步错"
3. **不可比较**：LLM A的reward输出 vs LLM B的reward输出怎么比较？

LTL是symbolic、compositional的，可以直接做set-matchingF1 score。

参考：Pnueli 1977的原始LTL paper [[44]](https://ieeexplore.ieee.org/document/812441)

---

## 3. MDP Grounding：四个模块与MDP的对应

### 3.1 Embodied MDP定义

论文把embodied agent形式化为：

$$\text{MDP} = \langle \mathcal{U}, \mathcal{S}, \mathcal{A}, \mathcal{M}, \mathcal{R}, g \rangle$$

- $\mathcal{U}$：物体universe，固定有限集
- $\mathcal{S}$：state space，$s = \langle \mathcal{U}, \mathcal{F} \rangle \in \mathcal{S}$，其中$\mathcal{F}$是relational Boolean features的集合
- $\mathcal{A}$：action space，$a = \langle name, args \rangle \in \mathcal{A}$
- $\mathcal{M}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$：deterministic transition function
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \times g \to \{0, 1\}$：reward，$\mathcal{R}(s, a, g) = 1 \iff eval(g, s) = 1$
- $g$：goal specification（LTL formula）

每个feature $f \in \mathcal{F}$是一个table，entry是$(o_1, \ldots, o_k)$ tuple，$k$是arity。比如二元predicate $on(x, y)$的arity是2。

### 3.2 四个能力模块的形式化

```
┌─────────────────────────────────────────────────────────┐
│  Goal Interpretation:                                    │
│  G: ⟨s₀, l_g⟩ → ĝ                                       │
│  (NL instruction → LTL goal)                             │
├─────────────────────────────────────────────────────────┤
│  Subgoal Decomposition:                                  │
│  Φ: ⟨s₀, g⟩ → φ̄                                         │
│  (LTL goal → sequence of LTL subgoals)                   │
├─────────────────────────────────────────────────────────┤
│  Action Sequencing:                                      │
│  Q: ⟨s₀, g⟩, M → ā                                      │
│  (LTL goal + dynamics → action sequence)                 │
├─────────────────────────────────────────────────────────┤
│  Transition Modeling:                                    │
│  T: ⟨s₀, g⟩, o → ⟨pre, eff⟩                              │
│  (operator → PDDL precondition & effect)                 │
└─────────────────────────────────────────────────────────┘
```

**关键设计insight**：这四个模块对应MDP的不同部分：
- G对应**input**（goal specification processing）
- Φ和Q对应**policy output**（一个是declarative states，一个是imperative actions）
- T对应**environment dynamics learning**

这让我想到你之前讲modularity时强调的：好的system design要让每个component可以独立替换、独立评估。这里就是同一个原则在embodied AI上的应用。

### 3.3 Subgoal Decomposition vs Action Sequencing

这两个模块输出都是trajectory，但有根本差异：

| | Subgoal Decomposition (Φ) | Action Sequencing (Q) |
|---|---|---|
| Output type | Declarative states | Imperative actions |
| Example | `soaked(rag) ∧ next_to(agent, fridge)` | `RIGHT_GRASP(rag.0), WALK(sink), ...` |
| Search space | 状态空间（巨大） | 动作空间（受限） |
| 评估方式 | BFS grounding成action sequence | 直接simulator执行 |

subgoal decomposition其实更难，因为它要在抽象状态空间里规划，而action sequencing只在有限的action vocabulary里组合。论文发现**subgoal decomposition不比action sequencing简单**，这点counterintuitive。

---

## 4. Fine-grained Metrics：错误分类学

### 4.1 Goal Interpretation的评估

给定ground truth goal $g$，先生成multiple equivalent simple LTL goals $\{g_0, g_1, \ldots, g_k\}$（因为复杂LTL可以有多种等价的simple形式）。

然后F1 set-matching：

$$F_1(g, \hat{g}) = \max_{g_i \in \mathcal{G}(g, \mathcal{U})} F_1(set(g_i), set(\hat{g}))$$

其中 $set(g) = \{a_1, \ldots, a_k\} \cup \{p_1, \ldots, p_\ell\}$，即把action sequence作为一个元素，加上所有final state propositions。

### 4.2 Trajectory Error Taxonomy（最重要的部分）

论文设计了一个decision tree来自动分类trajectory errors：

```
Trajectory Error Detection Pseudocode:

function check_runtime_errors(action, current_state, historical_states):
    # Step 1: Check affordance
    if not is_affordable(action, current_state):
        return "Affordance Error"
    
    # Step 2: Check redundancy (already achieved)
    if is_effect_redundant(action, current_state):
        return "Additional Step"
    
    # Step 3: Check precondition satisfaction
    if not is_precondition_satisfied(action, current_state):
        # Step 4: Was precondition ever satisfied?
        if not precondition_satisfied_in_history(action, historical_states):
            return "Missing Step"
        else:
            return "Wrong Order"
    
    return "No Runtime Error"
```

**Intuition解释这四类error**：

- **Affordance Error**：物体属性不允许这个动作。比如`open(shelf)`——shelf不能open。这是LLM不理解物理affordance。
- **Additional Step**：effect已经在当前state里了。比如`toggle_on(light)`但light已经on。这是LLM不记住/不观察当前状态。
- **Missing Step**：precondition不满足，且历史上也没满足过。比如`release(book)`但从来没grasp过。这是LLM不理解action dependency。
- **Wrong Order**：precondition不满足，但历史上满足过。比如先`grasp(book)` → `place(book, table)` → `release(book)`——release时book已经不在手里了。这是时序混乱。

这个分类让我想到 debugging code：syntax error / runtime error / logic error 的分层。这里Grammar Errors相当于syntax errors，Runtime Errors相当于logic errors。

### 4.3 Transition Modeling的Bipartite Matching评估

这是最技术性的部分。给定预测的PDDL operator和ground truth operator，比较它们的preconditions和effects：

```
function match_expressions(expr1, expr2):
    if both are literals:
        return expr1 == expr2  # 直接比较
    
    if type(expr1) != type(expr2):
        return 0
    
    if isinstance(Not):
        return match(expr1.expression, expr2.expression)
    
    if isinstance(When):
        return match(expr1.condition, expr2.condition) and 
               match(expr1.consequence, expr2.consequence)
    
    if isinstance((Exists, Forall)):
        return match(expr1.body, expr2.body)  # 量词匹配
    
    if isinstance((And, Or)):
        # 二分图匹配
        adj_matrix = build_adjacency(predicted_clauses, gt_clauses)
        match_result = maximum_bipartite_matching(adj_matrix)
        return total_match / max_possible_match
```

**Intuition**：precondition `and(A, B, C)` vs `and(A, D, E)`，不应该按位置比，应该按set来比。但是嵌套结构（如`when(...)`、`forall(x, ...)`）需要递归匹配。这是把graph matching用到logic formula comparison上。

参考：[scipy.sparse.csgraph.maximum_bipartite_matching](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_bipartite_matching.html)

---

## 5. 实验数据深度解读

### 5.1 总体模型对比（Table 3的核心pattern）

```
Model              | B-Avg | V-Avg | 备注
-------------------|-------|-------|------
o1-preview         | 74.9  | 64.4  | reasoning token确实有用
Claude-3.5 Sonnet  | 64.2  | 65.7  | BEHAVIOR goal interp强
Gemini 1.5 Pro     | 48.8  | 65.7  | VirtualHome goal interp强
GPT-4o             | 59.8  | 63.3  | balanced
Mistral Large      | 50.4  | 55.8  | V的action seq强
Llama-3-70B        | 48.1  | 47.3  | 最佳开源
Llama-3-8B         | 23.1  | 28.4  | 太小
```

**关键发现**：o1-preview在BEHAVIOR上74.9% vs 第二名64.2%，gap超过10%。这说明reasoning token（CoT hidden tokens）对长horizon planning确实有效。这跟你之前观察到的"reasoning models开始overfit to test-time compute"的现象一致。

### 5.2 错误分布的Pattern（Table 6 - Trajectory Evaluation）

让我把GPT-4o在BEHAVIOR上的错误分布拎出来：

```
GPT-4o on BEHAVIOR (Action Sequencing):
- Parsing Error:        0.0%
- Hallucination:       1.0%  
- Predicate-Arg Num:    0.0%
- Wrong Order:         9.0%
- Missing Step:       36.0%  ← 大头！
- Affordance:          1.0%
- Additional Step:     2.3%

GPT-4o on VirtualHome:
- Missing Step:       15.1%
- Additional Step:    15.7%  ← 比B多
- Wrong Order:         0.3%
```

**关键insight**：BEHAVIOR的missing step多（36%），VirtualHome的additional step多（15.7%）。原因：
- BEHAVIOR precondition严格（必须holding cleaning tool才能clean），LLM容易遗漏
- VirtualHome初始state里很多goal已经满足，LLM还是重复输出

### 5.3 Goal Interpretation的reporting bias（Table 4）

```
任务: "serve a meal"
Ground truth: ontop(chicken, plate) ∧ ontop(plate, table)
所有LLM预测: ontop(chicken, table)

任务: "cleaning sneakers"  
Ground truth: onfloor(gym_shoe, floor)
所有LLM预测: 缺少onfloor关系
```

这是**reporting bias**的经典案例：自然语言里"put chicken on table"是conversational shorthand，但物理世界必须经过plate这个中间物。LLM学到了language的shortcut，没学到physical precision。

这跟你讲LLM时提到的"language model learns statistical patterns of text, not world model"完全吻合。Reference: [On the Dangers of Stochastic Parrots](https://dl.acm.org/doi/10.1145/3442188.3445922)

### 5.4 Transition Modeling的Category分析（Table 7）

```
                    | Object States | Spatial Relations | Non-Spatial Relations
Claude-3.5 Sonnet  |     78.8      |       58.6        |        73.6       (B)
o1-preview         |     78.3      |       56.3        |        83.5       (B)
GPT-4o             |     71.3      |       45.9        |        73.0       (B)
Llama-3-70B        |     66.3      |       47.2        |        58.9       (B)
```

**Pattern**：spatial relations最难，non-spatial relations次之，object states最简单。但VirtualHome里non-spatial relations崩溃（o1-preview只有11.9%）——因为VirtualHome的non-spatial relation涉及holding、facing这种复杂多物体关系，BFS展开后precondition非常多。

---

## 6. 敏感性分析：哪些action最脆弱（Section F）

### 6.1 Per-action Success Rate

```
VirtualHome sensitivity:
- plug_in:        0.09  ← 最难！
- walk_towards:   0.63
- put_inside:     0.75
- close/grab:     1.00  ← 简单

BEHAVIOR sensitivity:
- slice_carvingknife: 0.00  ← 完全失败
- slice:               0.00
- place_inside:        低
- grasp:               0.95
```

**为什么plug_in这么难？** VirtualHome的`plug_in` precondition是OR结构：

```pddl
:precondition (or 
    (and (next_to ?char ?obj) (has_plug ?obj) (plugged_out ?obj))
    (and (next_to ?char ?obj) (has_switch ?obj) (plugged_out ?obj))
)
```

LLM几乎总是只预测`has_plug`分支，漏掉`has_switch`分支。这是disjunctive preconditions的失败模式。

### 6.2 Pipeline vs Modularized（Section G）

```
GPT-4o BEHAVIOR G+Q pipeline:
- Modularized task SR:    47.0%
- Pipeline-based task SR: 42.0%  (-5pp)

GPT-4o BEHAVIOR G+Φ pipeline:  
- Modularized task SR:    48.0%
- Pipeline-based task SR: 38.0%  (-10pp)
```

**Intuition**：modularized评估时给LLM ground truth goal作为input；pipeline-based时让LLM先做goal interpretation，再feed给下游。Goal interpretation的错误会propagate，导致downstream 5-10pp的degradation。这跟你讲error propagation in deep networks时强调的"early layer error compounds"完全一致。

### 6.3 Replanning的效果（Section H）

```
GPT-4o BEHAVIOR action sequencing:
                | Task SR | Exec SR | Missing Step | Additional Step
Original        |  47.0   |  53.0   |    36.0      |     0.0
With replanning |  59.0   |  63.0   |    14.1      |     0.7  (↑0.7)
```

Replanning让success rate +12pp，但additional step也轻微上升。**这暴露了LLM的over-correction倾向**：看到error message后倾向于添加动作而不是删除。这是agent design的重要警示——feedback loop不能让agent变得"越来越啰嗦"。

Stochastic action实验更dramatic：

```
Failure prob = 0.20:
                Execution SR  | Goal SR
w/o replanning:     10%       |   5%
w/ replanning:      65%       |  45%    (+55pp / +40pp)
```

action失败概率高时，replanning的相对收益巨大。这是robustness的关键来源。

---

## 7. 关键Architectural Insights

### 7.1 LLM在embodied setting的四大失败模式

我从论文里提取出最核心的失败模式：

**Failure Mode 1: Intermediate goal vs Final goal confusion**
LLM倾向于把intermediate state当作final goal。比如"drinking water"任务，预测`open(freezer)`作为final goal——但open freezer只是中间步骤。这暴露了LLM对"goal"和"subgoal"的边界感知模糊。

**Failure Mode 2: Reporting bias / Imprecise spatial relations**
"put X on table"被直译为`ontop(X, table)`，但实际应该是`ontop(X, plate) ∧ ontop(plate, table)`。LLM学的是conversational shorthand，不是physical precision。

**Failure Mode 3: Missing preconditions**
LLM不理解action的implicit precondition。比如"clean fridge"需要先holding cleaning tool，但LLM经常直接输出`CLEAN(fridge)`。这跟ChatGPT在webapp里能写代码但经常忘import是一样的——它学到的是surface pattern，不是dependency graph。

**Failure Mode 4: Forgetting already-achieved state**
LLM倾向于输出"完整"的plan，即使某些subgoal已经在initial state里满足了。这是training data bias——训练数据里plan总是从头开始的。

### 7.2 为什么o1-preview这么强？

o1-preview在BEHAVIOR上74.9% vs 其他模型~60%。我推测三个原因：

1. **Hidden reasoning tokens**：可以在output前显式推理，模拟planning过程
2. **Longer effective context**：reasoning tokens相当于working memory
3. **Self-correction during reasoning**：可以在reasoning里发现"我刚才漏了precondition"

这跟你在[DeepSeek-R1的讨论里提到的](https://www.interconnects.ai/p/deepseek-r1) "test-time compute scaling"完全一致。Embodied planning本质是search problem，search需要compute，o1把search放到inference time。

### 7.3 为什么BEHAVIOR比VirtualHome难？

```
                | VirtualHome | BEHAVIOR
Avg action len  |    8.76      |   14.6   (B长1.7x)
Task categories |     26       |   100    (B丰富4x)
Quantifiers     |     无       |   有     (B有forall/forpairs)
```

BEHAVIOR用BDDL的`forpairs`等高阶量词，比如`forpairs(?jar ?apple) (inside ?apple ?jar)`——所有apple-jar对都要满足inside关系。这种quantified goals的evaluation很难，single failure就导致整个quantifier失败。

VirtualHome更简单但state space更大（更多objects），所以Goal Interpretation在V上更难（要在大量objects里找对相关的那几个）。

---

## 8. 我对论文的批评性思考

### 8.1 优点

1. **Diagnostic value高**：四个模块解耦评估，可以精准定位LLM failure
2. **Symbolic abstraction合理**：用scene graph + LTL避免了perception noise的confound
3. **Error taxonomy实用**：runtime error的四分类对应不同fix策略
4. **覆盖广**：18个LLM × 2 simulators × 4 modules = 大规模systematic evaluation

### 8.2 局限

1. **纯symbolic，no perception**：Appendix K讨论了VLM，但主benchmark假设perfect scene graph。真实机器人的perception error会放大planning error。
2. **No low-level control**：actions是离散high-level的`GRASP`, `PLACE`，不涉及连续控制
3. **Deterministic transitions**：$\mathcal{M}$是deterministic的，没考虑stochastic dynamics
4. **Limited LTL fragment**：没有`globally`、`until`，不能描述"safety constraint"（比如"never put knife on bed"）
5. **Annotation cost高**：人工标注PDDL transition models 30+33个，scalability有限

### 8.3 跟你work的connection

你之前在[building LLM-from-scratch](https://github.com/karpathy/llm.c)和[makemore/nanoGPT系列](https://github.com/karpathy/makemore)里强调"understand by building"。这篇paper的哲学是"understand by decomposing"。两者都反对"端到端崇拜"——直接看loss/success rate而不分析内部机制。

但paper的limitation是它**没有提供fix**。它告诉我们LLM在goal interpretation上hallucinate，但没说怎么fix。下一代的benchmark应该包括remediation experiments——比如在prompt里加"reason about preconditions explicitly"看错误率是否下降。

---

## 9. Future Directions（我自己的思考）

基于这篇paper，我认为embodied LLM evaluation下一步应该做：

1. **Active perception**：让agent主动query visual system获取信息，而不是被动接收scene graph
2. **Counterfactual evaluation**：评估"如果precondition X满足，LLM会做什么"——sensitivity analysis的方向
3. **Cross-embodiment generalization**：同一个LLM在双臂机器人vs单臂机器人vs移动机器人上的能力差异
4. **Memory integration**：把episodic memory作为额外input，看LLM能否avoid重复error
5. **Hierarchical evaluation**：subgoal decomposition之上再加一层"task decomposition"——把"clean house"分解成"clean kitchen, clean bathroom, ..."然后再分解

---

## 10. 实操代码层面的key insights

如果你想复现或者扩展这个benchmark，关键技术点：

```python
# LTL evaluation的核心：递归+DP
def eval_ltl(phi, trajectory):
    if isinstance(phi, AtomicProp):
        return any(phi.satisfied_at(t) for t in trajectory)
    
    if isinstance(phi, And):
        return all(eval_ltl(sub, trajectory) for sub in phi.args)
    
    if isinstance(phi, Or):
        return any(eval_ltl(sub, trajectory) for sub in phi.args)
    
    if isinstance(phi, Not):
        return not eval_ltl(phi.arg, trajectory)
    
    if isinstance(phi, Then):  # φ₁ then φ₂
        for k in range(len(trajectory)):
            if (eval_ltl(phi.phi1, trajectory[:k]) and 
                eval_ltl(phi.phi2, trajectory[k:])):
                return True
        return False
    
    if isinstance(phi, Forall):
        return all(eval_ltl(phi.body.subst(x, obj), trajectory) 
                   for obj in universe)
```

BFS grounding subgoal to action：

```python
def subgoal_to_actions(subgoals, initial_state, action_space):
    """每个subgoal之间用BFS找action sequence"""
    current_state = initial_state
    full_trajectory = []
    
    for subgoal in subgoals:
        # BFS search
        queue = deque([(current_state, [])])
        visited = {current_state}
        
        while queue:
            state, path = queue.popleft()
            if subgoal.satisfied_at(state):
                full_trajectory.extend(path)
                current_state = state
                break
            
            for action in action_space:
                if action.precondition(state):
                    next_state = action.apply(state)
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [action]))
    
    return full_trajectory
```

更多代码细节参考：[github.com/embodied-agent-interface](https://github.com/embodied-agent-interface/embodied-agent-interface/)

---

## 11. 总结：这篇paper的真正贡献

Karpathy，我认为这篇paper的最大价值不在benchmark本身，而在它建立的**diagnostic framework**。它告诉我们：

> LLM在embodied decision making上的失败**不是单一能力不足**，而是四种distinct能力的差异化失败：
> - Goal Interpretation: 报告偏差 + intermediate/final goal混淆
> - Subgoal Decomposition: 与Action Sequencing一样难（counterintuitive）
> - Action Sequencing: precondition reasoning是主要瓶颈（45%错误）
> - Transition Modeling: non-spatial relations普遍失败

这就像医学诊断——知道"病人死了"vs知道"病人死于肝衰竭导致的multiple organ failure"是两回事。前者无能为力，后者可以靶向治疗。

Embodied AI 现在正从"看success rate"阶段进入"看per-module breakdown + per-error-type distribution"阶段。这跟 deep learning 早年从"看test accuracy"到"看 loss landscape / gradient flow / activation distribution"的演化完全一样。

References:
- [Paper PDF](https://embodied-agent-interface.github.io/)
- [Code & Data](https://github.com/embodied-agent-interface/embodied-agent-interface)
- [HuggingFace Dataset](https://huggingface.co/datasets/Inevitablevalor/EmbodiedAgentInterface)
- [BEHAVIOR-1K](https://behavior.stanford.edu/)
- [VirtualHome](https://virtualhome.org/)
- [LLM+P](https://arxiv.org/abs/2304.11477)
- [SayCan](https://say-can.com/)
- [Code as Policies](https://code-as-policies.github.io/)
- [Voyager](https://voyager.minedojo.tech/)
- [PlanBench](https://github.com/autonomousagents/planbench)
- [Karpathy: Recipe for training GPT](https://github.com/karpathy/nanoGPT)
- [Karpathy: State of GPT talk](https://www.youtube.com/watch?v=bZQun8Y4B2Y)

希望这个讲解帮你build了足够的intuition。如果你想深入某个module或者某个error type的细节，告诉我，我可以再展开。
