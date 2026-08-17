---
source_pdf: Seeing is Believing.pdf
paper_sha256: c4042d58cedc219f7ed4b0eb17ab8339969484f4415ab36de90845ecb2bdafea
processed_at: '2026-08-12T04:30:04-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 到底在干嘛

## 先说问题

想象你让机器人"把抽屉里的东西扔进垃圾桶"。听起来简单，但实际操作时会遇到一连串麻烦：

- 抽屉关着，你不知道里面有啥
- 可能是空的，可能有个 block，可能有三个 block
- 你得先**开抽屉看一眼**才知道
- 看完之后还要**拿起来**，**移动到垃圾桶**，**放进去**

这里头有个微妙的事：**"看一眼"这个动作本身也要 plan**。你得先走到抽屉前，伸手拉开，低头往里看——这其实是三四步 action 才能完成一次"观察"。而且你不知道看完会看到什么，可能是空的，可能不是。

这种"我先不知道，但可以主动去查清楚"的场景，在 robotics 里叫 **partial observability**，处理起来特别头疼。

---

## 以前怎么做的，为啥不行

### 老路子 1: 让 VLM 直接当 planner

你给 GPT-4o 看图片，问"下一步干啥"。问题在于：

**长 horizon 会失忆**。你开了抽屉，看到 block，拿起来，移动到垃圾桶——到第五步时，VLM 早就忘了第一步看到啥了。context window 撑不住。

**不知道什么时候该"先看一眼"**。VLM 倾向于直接行动，不会主动说"等等，我先确认一下这个杯子是不是空的再决定"。它缺乏 strategic information gathering 的能力。

[Valmeekam et al. 2023](https://arxiv.org/abs/2302.06706) 专门做了 benchmark 证明 LLM 在 systematic planning 上很拉胯。

### 老路子 2: 传统 belief-space planning

这帮 MIT 的人（Kaelbling, Lozano-Pérez）搞了十几年 [belief-space TAMP](https://journals.sagepub.com/doi/10.1177/0278364913484072)，思路是：维护一个 belief state（"我对世界的认知"），在 belief 上做 planning。

问题是：**perception module 要手写**。你得 hand-code 一个 classifier 判断"这个 drawer 是 empty 吗"，换个场景就废了。而且 object set 是预先定义好的，遇到新物体就傻眼。

---

## 这篇 paper 的 core insight

**一句话**：VLM 当 perceiver，symbolic planner 当 thinker，中间用 three-valued logic 桥接。

具体来说：

### Step 1: VLM 只负责回答 yes/no 问题

别让 VLM 规划动作，让它专注做它擅长的事——看图回答问题。

比如你问 GPT-4o："这张图里 cup 是 empty 吗？"它回答 yes 或 no。就这么简单。

但关键来了：**有时候它也不确定**。比如从侧面看杯子，看不清里面有没有水。这时候怎么办？

### Step 2: 引入"三值逻辑"

普通逻辑只有 true/false。这篇 paper 用 Kleene 的三值逻辑（[Kleene 1952](https://en.wikipedia.org/wiki/Three-valued_logic)），加一个 **unknown**：

| VLM 说的 | 系统理解的 |
|---------|----------|
| "Yes, empty" | **Known-True** |
| "No, not empty" | **Known-False** |
| "我不确定" / 判断不了 | **Unknown** |

技术上怎么实现？用两个 binary predicate 替代一个三值 predicate：

$$K_P(x) = \text{True} \Rightarrow \text{知道 } P(x) \text{ 为真}$$
$$K_{\neg P}(x) = \text{True} \Rightarrow \text{知道 } P(x) \text{ 为假}$$

两个都 false 就是 unknown。这个 trick 来自 [Bonet & Geffner IJCAI 2011](https://www.ijcai.org/Proceedings/11/Papers/135.pdf)。

### Step 3: 让 planner 会"主动去查"

现在 planner 看到 `Empty(cup1)` 是 unknown，它可以选择执行一个 `ObserveEmptiness(cup1)` 的 action 去查清楚。

但这里有个麻烦：**observe action 的结果是不确定的**——查完可能是 empty 也可能不是。经典 planner 处理不了 nondeterministic action。

**Solution: optimistic determinization**。把一个 observe action 拆成两个：

```
ObserveEmptiness+ → 假设查完是 empty
ObserveEmptiness- → 假设查完不是 empty
```

Planner **乐观地选一个它想要的 outcome**，然后 plan 后续动作。执行时如果实际结果跟假设不一样，**replan** 就行。

Bonet & Geffner 证明了：只要你每次执行后 replan，optimistic determinization 的解等价于真正的 nondeterministic 解。这是个很优雅的 reduction——把难问题化简成经典 planner 能解的问题。

### Step 4: 执行循环

```
观察环境 → VLM 评估 predicates → 更新 belief state
    → planner 规划 → 执行第一步 → 如果结果跟预期不一样就 replan
    → 重复直到 goal 满足
```

---

## 为什么这个设计 work

### VLM 的角色定位很对

VLM 强在哪？看图理解，回答 atomic question。

VLM 弱在哪？long-horizon reasoning, systematic search, uncertainty tracking。

BKLVA 把 VLM 限制在它擅长的 atomic perception query 上，把 systematic reasoning 交给 classical planner（Fast Downward）。这是个很好的 **separation of concerns**。

### Symbolic belief state 解决了"失忆"问题

VLM end-to-end 失败因为 state 藏在 context window 里，长了就丢。BKLVA 把 state 显式存成 symbolic predicate set：

```
{ On(cup1, table), Holding(robot, None), K_Empty+(cup1), 
  K_Empty-(cup2), ¬K_Empty+(cup3) ∧ ¬K_Empty-(cup3) }
```

这个 representation 永远不会"忘记"，而且 planner 可以直接看到哪些 predicate 还是 unknown，主动去 resolve。

### Three-valued logic 让 planner 天然支持 info gathering

经典 planner 只会"改变世界"。加上 K-fluents 后，planner 还能"获取信息"——因为 observe action 的 precondition 要求 `¬K_P ∧ ¬K_¬P`（即 unknown），effect 是设成 known。这就把 information gathering 自然纳入 planning 框架。

---

## 实验告诉我们什么

看 [Table I 的数据](https://arxiv.org/abs/2403.10454)：

### Cup Pick-Place（简单任务，fully observable）

| Method | Success | SPL |
|--------|---------|-----|
| VLM Captioning + LLM | 100% | 0.49 |
| VLM Labeling + LLM | 90% | 0.69 |
| **BKLVA** | **100%** | **1.00** |

Success rate 都差不多，但 **SPL 差很多**。SPL 衡量的是"路径效率"——1.0 是最优，越小说明走了多余的路。

为什么 baseline 的 SPL 低？因为 LLM 用 commonsense 推理，觉得"应该先确认一下"，于是做了不必要的 observe action。但这个 task 是 fully observable 的，根本不需要 observe。

BKLVA 为什么满分？因为 symbolic planner 看到所有 predicate 都是 known，observe action 的 precondition `¬K_P ∧ ¬K_¬P` 不满足，**根本不会去执行 observe**。这是 symbolic planner 比 LLM "聪明"的地方——它严格按逻辑走，不会因为"common sense"做多余的事。

### Drawer Cleaning（必须开抽屉看）

| Method | Success |
|--------|---------|
| VLM End-to-End | 0% |
| VLM Captioning + LLM | 10% |
| **BKLVA** | **80%** |

这个 task 必须 open drawer → observe content → pick object → place in bin。Baselines 基本全崩，因为：

1. **Long-horizon state tracking 失败**：开了 drawer1，拿了 block，还要记得 drawer1 已经开过了，里面是空的——VLM 到第三步就忘了
2. **不会 conditional replanning**：看到 drawer 里没东西不知道该怎么办

BKLVA 把这些都写进 symbolic belief state，planner 永远知道当前状态。

### Sort Weight（14 步长 horizon）

| Method | Success |
|--------|---------|
| 所有 baseline | 0% |
| **BKLVA** | **70%** |

需要：find cabinet → open → pick box → put on scale → read weight → judge empty → if empty: pick → dispose。这种 14 步的长 horizon，baselines 完全做不来。

---

## 一些细节值得品

### "看一眼"可能很贵

Paper 里强调的一点：**一次 observation 可能需要多步 action 才能完成**。

比如要判断 cup 是不是 empty，从正面看不清，你需要：
1. `MoveToReachObject` 走过去
2. `MoveToHandViewObjectFromTop` 调整到俯视角度
3. `ObserveCupContentFindEmpty` 才能真正 observe

这就把 perception 和 motion planning 耦合在一起了。BKLVA 的 PDDL operator 里 `ObserveCupContentFindEmpty` 的 precondition 要求 `InHandViewFromTop`，planner 会自动先规划出获取这个视角的动作序列。

这是这篇 paper 相比纯 object search 的独特之处——它关注 **property-level uncertainty**，而 property 的 observation 可能需要 complex action sequence。

### Lifted goal representation

Goal 用 first-order logic 表示成 $\forall x. \text{Empty}(x) \to \text{InBin}(x)$，而不是 grounded 成 `InBin(cup1)`。

好处：执行过程中发现新 object（比如开 drawer 发现里面有个 block），新 object 自动被 $\forall x$ 量化覆盖，不需要重新 translate goal。

### Incidental object discovery

不是主动 search 未知位置，而是"顺手发现"。开 drawer 发现里面有 block → block 加入 object set。这比 active object search 简单很多，但够用。

---

## 我觉得这篇 paper 最漂亮的地方

### 1. Role assignment 很 clean

VLM 做感知，LLM 做 goal translation，symbolic planner 做 reasoning。每个组件做它擅长的事，不越界。

对比 [SayCan](https://arxiv.org/abs/2204.01691) 让 LLM 当 planner + affordance scorer，或 [Code as Policies](https://arxiv.org/abs/2209.07753) 让 LLM 生成代码——那些方法把 LLM 推到它不擅长的位置。

### 2. Three-valued logic + optimistic determinization 是教科书级 trick

这个 trick 本身不新（[Bonet & Geffner 2011](https://www.ijcai.org/Proceedings/11/Papers/135.pdf)），但跟 VLM 结合得很自然。你 essentially 把 VLM 的 "I don't know" 升格为 first-class citizen，让 planner 能 reasoning about it。

### 3. Monotonic knowledge assumption

简化得很大胆，但 work。假设"一旦 known 就永远 known"——现实中如果你把 cup 拿走再放回来，VLM 可能又判断不出来了。但 paper 假设 quasi-static environment，大部分 mobile manipulation task 这个假设 hold 得还可以。

---

## 我觉得弱的地方

### 1. 实验规模小

10 seeds，Sort Weight 70% ± 0.32——方差太大，70% 跟 40% 统计上可能没区别。需要更大 scale 的 evaluation。

### 2. VLM error 没建模

VLM 说 "empty" 但实际不 empty 怎么办？Paper 假设 observation perfectly accurate（Section III.A 明说）。这假设太强，GPT-4o 在真实图片上判断 cup empty 的准确率多少？没给数据。

[这句 paper 里也承认了](https://arxiv.org/abs/2403.10454)："our approach relies on a perception system based on pretrained vision-language models whose performance bounds the overall success of our planning framework."

### 3. 跟 SOTA belief-space TAMP 没比

[Curtis et al. PONT-TAMP 2024](https://arxiv.org/abs/2403.10454) 也是处理 partial observability 的 TAMP，还带 risk awareness。BKLVA 没跟它直接比，只跟 VLM-based baselines 比。这有点避重就轻。

### 4. Operator 还是要手写

PDDL operator 定义还是 human-engineered。换到新场景要重新写 operator。Paper 在 Limitations 里承认了，说未来希望能从 foundation model 提取。

---

## 给你的 intuition

如果你只想记一句话：**这篇 paper 的贡献是把 VLM 当 "uncertainty estimator"，用 three-valued predicate 把 VLM 的感知能力包装成 symbolic belief state，然后让 classical planner 在 belief space 上做 systematic reasoning with information gathering**。

更深层 intuition：**partial observability 的本质是 state representation 问题**。VLM end-to-end 失败因为 state 在 context window 里漂移；BKLVA 成功因为 state 是 explicit symbolic belief，planner 能"看见"哪些 predicate 还 unknown 并主动 plan 去 resolve。

这跟 "explicit memory beats implicit context" 的大趋势一致。Neural network 擅长 pattern matching，symbol 擅长 long-horizon reasoning with compositionality。把两者用对的 interface 串起来，比硬让一个模块干所有事强。

类比一下：你让一个聪明但健忘的人（VLM）做长流程任务，他会忘东忘西。你给他一个笔记本（symbolic belief state）让他随时记下关键信息，他就能靠谱地完成任务。BKLVA 就是给 VLM 配了个笔记本，还教他什么时候该翻笔记本看看缺什么信息。

---

## 相关阅读

如果你想深挖：

- [Bonet & Geffner IJCAI 2011](https://www.ijcai.org/Proceedings/11/Papers/135.pdf) - K-fluents 原始论文，很短很 elegant
- [Kaelbling & Lozano-Pérez 2013](https://journals.sagepub.com/doi/10.1177/0278364913484072) - belief-space TAMP 经典
- [Curtis et al. 2024 PONT-TAMP](https://arxiv.org/abs/2403.10454) - 更现代的 partial observable TAMP with risk
- [Silver et al. CoRL 2022](https://openreview.net/forum?id=OIaJRUo5UXy) - neuro-symbolic bilevel planning
- [MOLMO](https://arxiv.org/abs/2409.17146) - pointing-based VLM，比 bounding box 更适合 robotics
- [Fast Downward](https://www.jair.org/index.php/jair/article/view/10457/25068) - 他们用的 planner

---

# Seeing is Believing: 用 VLM 当 Uncertainty Estimator 的 Belief-Space Planning

这篇 paper 是 Northeastern + MIT + Princeton 的工作（Linfeng Zhao, Willie McClinton, Aidan Curtis, Nishanth Kumar, Tom Silver, Leslie Pack Kaelbling, Lawson Wong），把 VLM 塞进 belief-space planning 当 perception module，专门解决 partially observable mobile manipulation。我会一层层剥开讲，先 build intuition，再讲技术细节、公式、实验数据。

---

## 1. 一句话直觉

Long-horizon mobile manipulation 的难点在于 **partial observability** + **unknown objects/properties**。常见两种 strategy：

- **Strategy 1**: end-to-end policy over observation history → 数据需求随 horizon 指数增长，且 VLM/LLM 在 systematic reasoning + uncertainty modeling 上很弱（参考 [Valmeekam et al. 2023](https://arxiv.org/abs/2302.06706)）
- **Strategy 2**: belief-space planning → 传统方法需要 hand-crafted state estimator + belief transition model

这篇 paper 的核心 insight：**VLM 适合做 on-demand predicate evaluator**（即"这个 cup 是 empty 吗？"这种 query），把 perception 模块化；然后用 **three-valued logic (Kleene logic)** 表示 belief（known-true / known-false / unknown），再通过 **K-fluents + optimistic determinization** 把 belief-space 的 nondeterministic planning 化简成 classical deterministic planning。这样既能 leverage foundation model 的 perception 能力，又能享受 symbolic planner 的 systematic reasoning。

作者把这个 pipeline 称为 **BKLVA**：
- **B**elief-space planning
- **K**-fluents (three-valued predicates)
- **L**LM-based goal grounding
- **V**LM-based perception
- **A**ctions for information gathering

---

## 2. 为什么这个组合有意思

VLM 直接做 end-to-end planning 的问题在于：

1. **No explicit belief state**：history 全靠 context window 撑住，long-horizon 会丢失
2. **No strategic info gathering**：不知道什么时候应该"先看一眼"再行动
3. **No systematic reasoning**：[Silver et al. 2022](https://openreview.net/forum?id=OIaJRUo5UXy) 和 [Huang et al. 2022](https://arxiv.org/abs/2207.05608) 都指出 LLM/VLM 在 combinatorial planning 上表现差

而传统 belief-space TAMP 的问题在于：

1. 需要 hand-coded perception predicates
2. 需要 hand-coded belief transition model
3. object set 是 closed-world assumption

BKLVA 的 trade-off：**保留 symbolic planner 的 systematic reasoning**，但把 perception 模块外包给 VLM，用 three-valued logic 让 symbolic planner 直接能 reasoning about uncertainty。

---

## 3. 数学公式：Three-Valued Belief + K-fluents

这是 paper 的技术核心。设 predicate $P(x)$ 表示某个 object $x$ 的某个 property（比如 `Empty(cup1)`）。

### 3.1 三值逻辑

定义两个 binary **K-fluents**：

$$K_P(x), \quad K_{\neg P}(x)$$

含义：
- $K_P(x) = \text{True}$: agent **知道** $P(x)$ 是 True
- $K_{\neg P}(x) = \text{True}$: agent **知道** $P(x)$ 是 False
- 两者都 False：$P(x)$ **未知**

完整对应关系：

| $K_P$ | $K_{\neg P}$ | Belief about $P(x)$ |
|-------|--------------|---------------------|
| True  | False        | Known-True          |
| False | True         | Known-False         |
| False | False        | Unknown             |
| True  | True         | (矛盾，不允许)       |

这套思路来自 [Bonet & Geffner IJCAI 2011](https://www.ijcai.org/Proceedings/11/Papers/135.pdf) 和 Kleene 的三值逻辑。

### 3.2 Information-Gathering Action 的 PDDL

一个 observe action 本身是 **nondeterministic**，因为结果可能是 $K_P$ 也可能是 $K_{\neg P}$：

```
(:action ObserveEmptiness
:parameters (?o - object ?s - surface)
:precondition (and (On ?o ?s) (HandEmpty)
                   (¬KEmpty+ ?o) (¬KEmpty- ?o))   ; 只有 unknown 时才能 observe
:effects (oneof (KEmpty+ ?o) (KEmpty- ?o)))       ; nondeterministic
```

### 3.3 Optimistic Determinization

经典 trick：把 nondeterministic action 拆成两个 deterministic action，让 planner **乐观地选择想要的 outcome**：

$$
A \xrightarrow{\text{nondet}} \{K_P, K_{\neg P}\} \quad \Longrightarrow \quad A_+ \to K_P, \quad A_- \to K_{\neg P}
$$

```
(:action ObserveEmptiness+       ; 乐观假设 cup 是 empty
:precondition (and (On ?o ?s) (HandEmpty) (¬KEmpty+ ?o) (¬KEmpty- ?o))
:effects (and (KEmpty+ ?o)))

(:action ObserveEmptiness-       ; 乐观假设 cup 不是 empty
:precondition (and (On ?o ?s) (HandEmpty) (¬KEmpty+ ?o) (¬KEmpty- ?o))
:effects (and (KEmpty- ?o)))
```

**为什么这样做能 work？** 关键 theorem（Bonet & Geffner）：**replanning after each execution step** 的条件下，optimistic determinization 解出来的 plan 等价于 nondeterministic 解。直觉：planner 假设最有利情况规划；如果实际 observation 跟假设不符（比如假设 cup empty 但实际有水），就触发 replan，从更新后的 belief state 重新规划。

---

## 4. 整体 Pipeline（Algorithm 1）

```
Require: b_0, g_text, predicates P, actions A
1: g ← Translate(g_text, P, O_0)        ; LLM 把 NL goal 翻译成 first-order logic
2: b ← b_0
3: while ¬Satisfied(g, b) do
4:    p ← Plan(b, g, A)                  ; Fast Downward 在 determinized domain 上规划
5:    if p = None then return False
6:    for a in p do
7:       o ← Execute(a)                  ; 执行第一个 action, 得到 observation
8:       b ← BeliefUpdate(b, a, o)       ; VLM 评估 predicates, 更新 K-fluents
9:       if ¬ExpectedEffects(a, b) then
10:         break                        ; 不符合乐观假设 → replan
11:     end if
12:   end for
13: end while
14: return True
```

### 4.1 Goal Translation

NL goal → lifted first-order logic。比如 "put any object in the drawer into a paper bin" → 

$$\forall x. \text{InBin}(x) \land K_{\text{Empty+}}(\text{drawer})$$

关键：用 **lifted**（含变量）而不是 grounded，因为执行过程中可能 discover 新 object。如果用 `InBin(cup1)` 这种 grounded form，新发现 cup2 时还要重新 ground，麻烦。

### 4.2 Perception：VLM 怎么当 Predicate Evaluator

Pipeline 由两部分组成：

**(a) Object Detection & Localization**
- **MOLMO**（[Deitke et al. 2024](https://arxiv.org/abs/2409.17146)）：pointing-based localization，给 textual prompt 返回 pixel location
- **SAM**（[Kirillov et al. 2023](https://arxiv.org/abs/2304.02643)）：从 pointing 生成 segmentation mask
- Depth + SLAM odometry → 把 pixel 转成 global coordinate 的 spatial extent $S$ 和 location $L$

**(b) Predicate Evaluation**
- **GPT-4o**（[OpenAI 2023](https://arxiv.org/abs/2303.08774)）批量 evaluate 所有 predicate over 所有 visible objects
- 一次 query 评估多个 predicate（Spot 6 个 camera，全部 objects × predicates 一起问）
- 返回 Yes/No，对应 $K_P$ / $K_{\neg P}$；如果 VLM 自己说"不确定"就视为 unknown（保持 $\neg K_P \land \neg K_{\neg P}$）

Prompt 设计很有意思——见 paper Appendix X.A，比如 `Empty` 的 prompt：
> "This predicate is true if the object is not inside any container..."

### 4.3 Belief State Update

两个原则：
1. **Monotonic knowledge acquisition**：unknown → known 是单向的，known 不退回 unknown
2. **Quasi-static assumption**：predicate 值除非被 observation 或 action 改变，否则保持不变

Incidental object discovery：当 `EmptyContainer(drawer)` 从 unknown 变成 False（即发现 drawer 里有东西），新发现的 object 加入 object set $O$。

---

## 5. 实验数据深度解读

### 5.1 三个 synthetic tasks + 一个 real task

| Task | Horizon | 关键挑战 |
|------|---------|----------|
| Cup Pick-Place | 短 | fully observable，测试是否过度 gather info |
| Drawer Cleaning | 中 | 必须开 drawer 才能知道 contents |
| Sort Weight | 长（14 steps optimal） | 必须用 scale 称重才知道 box empty |
| Empty Cup Removal (Spot) | 中 | 必须 close-up top-view 才能判断 cup empty |

### 5.2 主结果 Table I

| Method | Cup Pick-Place (Succ/SPL) | Drawer Cleaning | Sort Weight |
|--------|---------------------------|-----------------|-------------|
| Random | 0% / 0.00 | 0% / 0.00 | 0% / 0.00 |
| VLM End-to-End | 30% / 0.15 | 0% / 0.00 | 0% / 0.00 |
| VLM (Captioning) + LLM | 100% / 0.49 | 10% / 0.04 | 0% / 0.00 |
| VLM (Labeling) + LLM | 90% / 0.69 | 0% / 0.00 | 0% / 0.00 |
| **BKLVA (Ours)** | **100% / 1.00** | **80% / 0.32** | **70% / 0.46** |

几个观察：

**(1) Cup Pick-Place 上 SPL 差异暴露的问题**

这个 task fully observable，所有 info 都给了。理想 agent 应该直接执行 manipulation，**跳过 information-gathering**。但 VLM+LLM baselines 的 SPL 只有 0.49/0.69，说明它们**做了不必要的 information gathering**——因为 LLM 用 commonsense 推理，倾向于"先确认一下"。

BKLVA 拿满分 1.00，因为 symbolic planner 知道 $\neg K_P \land \neg K_{\neg P}$ 不成立时（已经 known），observe action precondition 不满足，不会去执行。

**(2) Drawer Cleaning 上 baselines 全崩**

VLM End-to-End 0%，VLM Captioning 10%。原因：执行过程中需要 maintain "开了哪个 drawer、drawer 里有什么" 这种 long-horizon state，VLM/LLM 在 history 很长时丢失 context。BKLVA 用 symbolic belief state 持久化，80% 成功率。

**(3) Sort Weight 上 baselines 全 0%**

需要 14 步最优 plan：开 cabinet → pick box → 放 scale 上 → 读数 → 判断 empty → 如果 empty 拿起 → 丢 bin。这种长 horizon + 需要 conditional info gathering，baselines 完全做不来。

### 5.3 Baseline 失败的具体原因（paper 列出来的）

1. **Commonsense-driven extra steps**：LLM 觉得"应该先 X 再 Y"，实际不需要
2. **Subtle state 差别捕捉不到**：VLM captioning 把 "cup on table" 和 "cup in hand" 都说成 "cup is visible"
3. **History 不理解**：同一 state 重复访问，LLM 会 retry 同样 action
4. **Output format 错误**：state 描述不规范，下一步推理出错

---

## 6. 系统架构图解析（Fig. 3）

Pipeline 是 **Observe → Update → Plan → Execute** 循环：

```
                ┌──────────────────────────────────┐
                │  Goal Translation (offline)       │
                │  "move empty cups to bin"          │
                │       ↓ (LLM)                     │
                │  ∀x. Empty(x) → InBin(x)          │
                │  + Determinize observation ops    │
                └──────────────────────────────────┘
                            │
   ┌────────────────────────┴───────────────────────┐
   │ Runtime loop:                                    │
   │                                                  │
   │  Observation (images + sensors)                   │
   │       ↓                                          │
   │  ┌──────────┬─────────────────────┐              │
   │  │ MOLMO    │ VLM predicate eval   │              │
   │  │ + SAM    │ (GPT-4o, batched)    │              │
   │  ↓          ↓                      │              │
   │  Objects O  K_P, K_¬P              │              │
   │  + spatial  predicate values       │              │
   │  info                            │              │
   │  └──────────┴─────────────────────┘              │
   │       ↓                                          │
   │  BeliefUpdate (monotonic)                         │
   │       ↓                                          │
   │  Symbolic Belief State b                          │
   │       ↓                                          │
   │  Fast Downward Planner (determinized domain)      │
   │       ↓                                          │
   │  Plan p = [a_1, a_2, ...]                         │
   │       ↓                                          │
   │  Execute a_1 → new observation                    │
   │       ↓                                          │
   │  if ¬ExpectedEffects: replan                      │
   └─────────────────────────────────────────────────┘
```

---

## 7. PDDL Operators 解析（Appendix X.B）

举两个关键 operator：

### 7.1 `ObserveCupContentFindEmpty`（乐观 determinization + 版）

```
(:action ObserveCupContentFindEmpty
:parameters (?robot - Robot ?cup - Container ?surface - Immovable)
:precondition (and
    (On ?cup ?surface)
    (InHandViewFromTop ?robot ?cup)      ; 必须从上方看
    (HandEmpty ?robot)
    (NotHolding ?robot ?cup)
    (Unknown_ContainerEmpty ?cup))      ; 必须 unknown 才能 observe
:effect (and
    (Known_ContainerEmpty ?cup)
    (BelieveTrue_ContainerEmpty ?cup)    ; 乐观假设 empty
    (not (Unknown_ContainerEmpty ?cup))))
```

注意： precondition 要求 `InHandViewFromTop`——这是关键的 **subgoal**！Planner 必须先规划出 `MoveToHandViewObjectFromTop` 等动作序列，才能执行 observe。这就是 paper 强调的"看一眼"可能需要 multiple actions：navigate close → 调整 camera angle → look from top。

### 7.2 `PickObjectFromTop`

```
(:action PickObjectFromTop
:precondition (and
    (On ?object ?surface)
    (HandEmpty ?robot)
    (InHandView ?robot ?object)
    (NotInsideAnyContainer ?object)
    (IsPlaceable ?object)
    (HasFlatTopSurface ?surface))
:effect (and
    (Holding ?robot ?object)
    (not (On ?object ?surface))
    ...))
```

注意 `NotInsideAnyContainer` 这个 precondition：如果 object 在 drawer 里，必须先 open drawer 取出才能 pick。这就把 "drawer 里有什么" 的 uncertainty 和 manipulation 自然耦合到一起了。

---

## 8. Real Robot Demo：Empty Cup Removal on Spot

Spot 上有 6 个 RGBD camera：1 个 arm-mounted（fine manipulation）+ 5 个 body cameras（360° view）。3 个 cup 放在不同桌子上，robot 不知道哪些 empty。流程：

1. **Body view** 看到杯口侧面 → VLM 评估 `Empty` predicate → 通常 unknown（侧面看不出）
2. 触发 planner 规划：`MoveToHandViewObjectFromTop` → `ObserveCupContentFindEmpty`（乐观假设 empty）
3. 如果实际 empty：继续 `PickObjectFromTop` → `MoveToBin` → `PlaceObject`
4. 如果实际 not empty：replan，跳过这个 cup，去检查下一个

这里关键 benefit：robot **不会盲目对所有 cup 都做 close-up 检查**，而是按 goal 需求 selectively gather info。

---

## 9. 与 Related Work 的关系

### 9.1 Belief-Space Planning 谱系

- [Kaelbling & Lozano-Pérez 2013](https://journals.sagepub.com/doi/10.1177/0278364913484072) BHPN：hand-coded perception + belief-space TAMP
- [Garrett et al. 2020](https://arxiv.org/abs/1911.04577)：online replanning in belief space
- [Curtis et al. 2024](https://arxiv.org/abs/2403.10454) PONT-TAMP：uncertainty + risk awareness
- BKLVA 与这些工作的差别：**用 VLM 替代 hand-coded perception**，让 belief-space planning 落地到 open-world

### 9.2 Foundation Models for Planning

- [SayCan (Ahn et al.)](https://arxiv.org/abs/2204.01691)：LLM + affordance
- [Code as Policies (Liang et al.)](https://arxiv.org/abs/2209.07753)：LLM 生成 code
- [PDDLPlan with LLM (Silver et al.)](https://arxiv.org/abs/2302.06706)：LLM 当 planner，但无 uncertainty handling
- BKLVA 把 LLM 限制在 **goal translation** 这个 LLM 真正擅长的任务，把 systematic reasoning 留给 symbolic planner

### 9.3 Object Search

- [Wong et al. 2013](https://ieeexplore.ieee.org/document/6630966/)：manipulation-based active search
- [Nie, Wong, Kaelbling 2016](https://ieeexplore.ieee.org/document/7487752)：partial known env search
- BKLVA 关注 **property-level uncertainty**，而不是 object existence uncertainty

---

## 10. Limitations（作者自承认）

1. **Hand-defined operators**：PDDL operators 还是 human-coded，未来希望从 foundation model 提取或学习
2. **VLM perception 是 bottleneck**：整个系统 success 上限受限于 VLM predicate evaluation 准确率
3. **No active object search**：只能 incidental discovery（开 drawer 顺手发现），不能主动 search unknown locations
4. **Low-level skill integration 简化**：没深究 motion planning under physical constraint
5. **Monotonic knowledge assumption**：known → unknown 不允许，现实中如果有 information loss（机器人移走了又看不到）会失效

---

## 11. 我对这篇 paper 的评价

**Strong points**:
- 把 VLM 当 predicate evaluator 而不是 planner，这个 role assignment 非常合理——VLM 强在 perception，弱在 systematic reasoning
- Three-valued logic + K-fluents + optimistic determinization 是 well-established trick（Bonet & Geffner），跟 foundation model 结合得很自然
- Lifted goal representation 处理 open-world object discovery，比 grounded 优雅
- Real robot demo 验证 pipeline 可落地

**Weak points**:
- Synthetic 实验 10 seeds 偏少，Sort Weight 70% ± 0.32 方差很大
- VLM predicate evaluation 的 failure mode 没有详细分析（GPT-4o 误判率多少？哪些 predicate 难？）
- 跟 baseline 比较时，baseline 的 prompt engineering 是否 fair？VLM End-to-End 是否给了足够 context？
- "14 steps optimal" 这个 claim 没有 formal proof
- 没有跟 [Curtis et al. PONT-TAMP](https://arxiv.org/abs/2403.10454) 这种 SOTA belief-space TAMP 直接对比
- Monotonic knowledge 假设太强，real-world 物体可能被遮挡后重新变 unknown

**值得 follow-up 的方向**:
- VLM uncertainty calibration：让 VLM 输出 confidence 而不只是 Yes/No
- 学习 belief-space operators：从 demonstration 或 foundation model 提取
- 跟 neuro-symbolic bilevel planning（[Silver et al. CoRL 2022](https://openreview.net/forum?id=OIaJRUo5UXy)）结合
- 把 POMDP 的 transition uncertainty 也纳入（现在只考虑 observation uncertainty）

---

## 12. 核心参考链接

- Paper PDF（推测在 arXiv 上有 preprint，搜索 "Seeing is Believing Belief-Space Planning Foundation Models"）
- [MOLMO](https://arxiv.org/abs/2409.17146) - pointing-based VLM
- [SAM](https://arxiv.org/abs/2304.02643) - segmentation
- [GPT-4o tech report](https://arxiv.org/abs/2303.08774)
- [Bonet & Geffner IJCAI 2011](https://www.ijcai.org/Proceedings/11/Papers/135.pdf) - K-fluents 原始论文
- [Kaelbling & Lozano-Pérez 2013](https://journals.sagepub.com/doi/10.1177/0278364913484072) - belief-space TAMP 经典
- [Curtis et al. 2024 PONT-TAMP](https://arxiv.org/abs/2403.10454) - 部分可观察 TAMP with risk
- [Valmeekam et al. 2023](https://arxiv.org/abs/2302.06706) - LLM planning benchmark
- [SayCan](https://arxiv.org/abs/2204.01691)
- [Fast Downward planner](https://www.jair.org/index.php/jair/article/view/10457/25068)
- [PDDL 2.1](https://arxiv.org/pdf/1106.4561.pdf)
- [Kleene three-valued logic](https://en.wikipedia.org/wiki/Three-valued_logic)

---

## 13. 给你的 Intuition 总结

如果你要记住这篇 paper 一件事，那就是：**VLM 的合适定位是 "predicate evaluator" 而不是 "planner"**。Planner 需要 systematic search over long horizon with uncertainty，这是 VLM/LLM 不擅长的；但 VLM 在 "看图判断 cup 是不是 empty" 这种 atomic perception query 上很强。用 three-valued logic + K-fluents 把 VLM 的输出 wrap 成 symbolic belief state，然后交给 classical planner 做 systematic reasoning，最后用 optimistic determinization + replanning 处理 observation 的 nondeterminism——这是一个非常 clean 的 factorization。

更深一层的 intuition：**partial observability 的问题本质是 state representation 问题**。VLM End-to-End 失败因为 state 在 context window 里漂移；BKLVA 成功因为 state 是 explicit symbolic belief，planner 可以"看见"哪些 predicate 还是 unknown 并主动 plan 去 resolve 它们。这跟 "explicit memory beats implicit context" 的 broader trend 一致。

如果你还想 build 更深的 intuition，建议看 Bonet & Geffner 的原 paper（很短，10 页），然后想象把他们的 hand-coded perception 换成 VLM query——其实就是 BKLVA 的核心 move。其余的工程（MOLMO + SAM + SLAM + Spot SDK）都是让这个 move 落地到 real robot 的胶水代码。
