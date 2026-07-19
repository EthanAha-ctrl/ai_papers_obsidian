---
source_pdf: A Real-Time PDDL-Based Planning Component for Video Games.pdf
paper_sha256: e2d1ec9214d6906681112cbafc732a0b4c1bcefaedd8d0c7ea84ba7815c3ffa9
processed_at: '2026-07-17T21:06:50-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# A Real-Time PDDL-Based Planning Component for Video Games - 深度解析

## Paper 核心论点

这篇 paper 探讨的核心问题是：能否将 **PDDL (Planning Domain Definition Language)** 这种来自 AI planning 学术界的 expressive language 直接接入 commercial video game 的 game loop，并实现 real-time playability。作者通过两个 case study (Iceblox 和 VBS2) 给出了 affirmative answer，并 reverse engineering 出一个通用的 planning component architecture。

参考: [PDDL 标准](https://helios.hud.ac.uk/scommv/IPC-2008/), [FF Planner](https://fai.cs.uni-saarland.de/hoffmann/ff.html), [GOAP / F.E.A.R. AI](https://www.gamedeveloper.com/programming/three-states-and-a-plan-the-a-i-of-f-e-a-r)

---

## 1. 背景与动机

### 1.1 为什么需要 planning 而不是 hard-coded FSM

传统 game AI 多采用 **Finite State Machine (FSM)** 或 **Hierarchical FSM (HFSM)**，将 goal 与唯一的 action sequence 绑定。问题在于 dynamic environment 下 unexpected events（enemy random move、player 行为）会让 pre-baked sequence 失效。

**GOAP (Orkin 2003, 2006)** 在 F.E.A.R. 中首次工业化地 decouple 了 goal 和 action sequence：planner 根据 current world state 动态 search 出一条满足 precondition-effect chain 的 action 序列。这是 STRIPS-style planning 在 game 中的成功落地。

PDDL 是 IPC (International Planning Competition) 自 1998 年起的 mandatory language，其 expressiveness 远超 GOAP 所用的 stripped-down version：支持 **resources, preferences, constraints, timed literals, numeric fluents, durative actions** 等。但 expressiveness 带来 computational complexity 爆炸（Erol, Nau, Subrahmanian 1995 证明即使 simple PDDL 也很 hard）。这就是 Orkin 选择降低 operator expressiveness 的原因。

本文的反向论点是：通过 engineering decisions，full PDDL 也可以 real-time。

### 1.2 Planning 的两阶段本质

```
Phase 1: Plan construction
  Input: (s_init, s_goal, Operators)
  Output: π = ⟨a_1, a_2, ..., a_n⟩ s.t. s_init ⊨ pre(a_1), 
                                          result(a_i) ⊨ pre(a_{i+1}),
                                          result(a_n) ⊨ s_goal

Phase 2: Plan execution + monitoring
  Execute a_i in game world;
  If deviation detected → re-plan or repair
```

两阶段都需要在 time budget 内完成，这是 real-time 的本质难点。

---

## 2. Case Study 1: Iceblox

### 2.1 Game mechanics

Iceblox (Hornell 1996) 是一个 2D arcade game：
- Player 控制 penguin 在 maze 中 move (horizontal/vertical)
- 收集 iced coins（被冻在 ice block 里的 coin，需要 7 次 push 才能 crack 开取出）
- Flames 随机 patrol，碰到 penguin 就 kill
- Ice block 可以 push 滑动，slide 过 flame 可以 kill flame
- 收集所有 coin 进入下一关

参考: [Iceblox 原版](http://www.javaonthebrain.com/java/iceblox/)

### 2.2 Real-time budget 推导（关键 engineering 推理）

这一段是全文最精彩的 engineering reasoning，我详细展开：

**Frame rate analysis:**
- Playable frame rate ≈ 30 FPS
- Sprite size = 30 × 30 pixels
- Flame speed = 30 pixels/frame × 30 frame/s ÷ 30 pixels/crossroad = 1 crossroad/second

**Randomness analysis:**
- Flame 在每个 crossroad 有 1/4 概率改方向
- 因此 flame 可以认为 "stable direction" 持续约 4 crossroads（worst case 才会立即转）

**Danger threshold:**
- 当 penguin 距 flame ≥ 4 crossroads 时仍然 safe
- < 4 crossroads 时需要立即 action → 触发 re-plan

**Time budget:**
$$T_{plan} \leq T_{flame\_travel}(4 \text{ crossroads}) - T_{safety\_margin}$$

$$T_{plan} \leq 4 \text{ s} - 1 \text{ s} = 3 \text{ s}$$

减去 1 秒是因为：
- Plan 找到后需要 trigger execution (有 latency)
- Flame 在 crossroad 间移动不可中断，所以不能让它走到第 3 个 crossroad

这就是为什么作者说 "limit is when the penguin is 4 crossroads away from a flame" 然后 conservative 地取 3 秒。

**Intuition building:** 这里的核心 insight 是 — real-time planning 的 budget 不取决于 planner 本身性能，而取决于 game world 中最快 adversarial entity 的可达时间。这是 game-aware planning 的关键。

### 2.3 Operator abstraction (核心 engineering 决策)

如果用 naïve 的 "move-up, move-down, move-left, move-right" 这些 primitive operators，plan length 会爆炸（一个 maze 20×20 就可能产生几十步 move），FF 这种 heuristic planner 在 3 秒内 search 不完。

**Solution: Abstract Move operator**
$$Move(s_{from}, s_{to}) : \text{pre} = at(s_{from}) \land reachable(s_{to}) \quad \text{eff} = at(s_{to}) \land \neg at(s_{from})$$

将 pathfinding 委托给 plan execution module（A* 之类），planner 只关心 high-level decisions。

这继承了 STRIPS 的 GoThru operator (Fikes, Hart, Nilsson 1972)。同样地，7 次 push crack coin 压缩成单个 **extract** operator。

### 2.4 Predicates 设计

| Predicate | 含义 |
|-----------|------|
| `(at i j)` | sprite 位于 crossroad_{i,j} |
| `(extracted i j)` | coin_{i,j} 已被收集 |
| `(guard i1 j1 i2 j2)` | flame_{i1,j1} 守卫 coin_{i2,j2}（< 3 crossroads 距离） |
| `(iced-coin i j)` | (i,j) 处有含 coin 的 ice block |
| `(protected-cell i j)` | 存在 safe path 到 (i,j) |
| `(reachable-cell i j)` | 存在 path 但有 flame 危险 |
| `(weapon i1 j1 i2 j2 i3 j3)` | 武器在 (i2,j2)，应从 (i1,j1) 推，停在 (i3,j3) |
| `(blocked-path i j)` | path 到 (i,j) 上有 ice block |
| `(blocked-by-weapon i1 j1 i2 j2)` | (i2,j2) 的 weapon 在 (i1,j1) 的 path 上 |

下标变量解释：
- `i, j` — crossroad 的 row/column 坐标
- `i1, j1` — 起点 / weapon 出发位置
- `i2, j2` — weapon 当前位置
- `i3, j3` — weapon 滑动停止位置
- `guardx, guardy` — flame 位置
- `coinx, coiny` — coin 位置
- `blockedx, blockedy` — 被 block 的目标位置

### 2.5 关键 operator 详解

#### extract operator
```lisp
(:action extract
  :parameters (?coinx - coord-i ?coiny - coord-j)
  :precondition (and (protected-cell ?coinx ?coiny)
                      (iced-coin ?coinx ?coiny)
                      (at ?coinx ?coiny)
                      (reachable-cell ?coinx ?coiny))
  :effect (and (extracted ?coinx ?coiny)
               (not (iced-coin ?coinx ?coiny))
               (not (protected-cell ?coinx ?coiny))
               (not (reachable-cell ?coinx ?coiny))))
```

**Intuition:** 这里 `protected-cell` 和 `reachable-cell` 同时为 true 看起来矛盾（一个说 safe，一个说有 danger）。实际上这是 PDDL 表达 "存在 path" 的方式，需要 plan 在执行时选择 safe path。这种 over-approximation 让 planner 在 search 时不被 low-level path 细节困扰。

#### kick-to-kill-guard operator
这是最复杂的 operator，有 9 个 parameters，覆盖：
- 起始 reachable 位置
- weapon 位置
- weapon 滑动终点
- guard 位置
- coin 位置
- 被 block 的位置

**Effects** 同时维护：
- weapon 移动后的新位置
- 移除 guard predicate
- 更新 protected/reachable cell 状态
- 新增 blocked-by-weapon

**Why so many effects?** 因为 PDDL 不允许 side-effect，所有 state change 必须 explicit。这暴露了 PDDL 在 game 应用中的 verbosity 问题。

### 2.6 Plan length 公式

$$L_{plan} = 2 + 2 \cdot n_{danger} + n_{destroy} + n_{align}$$

变量含义：
- `2` — baseline: move to coin + extract
- `n_{danger}` — 危险 flame 数量（每个需要 move+push 两步 kill）
- `n_{destroy}` — 需 destroy 的 blocking ice block 数
- `n_{align}` — 需 primary push 来 align weapon 的次数

随机生成关卡平均每个 coin 1-2 个 dangerous flame，所以典型 plan length 在 4-8 之间。这是 FF planner 在 3 秒内可处理范围。

### 2.7 Planner 对比

| Planner | 语言 | 算法 | 实时表现 |
|---------|------|------|----------|
| **FF** (Hoffmann 2001) | C | Heuristic forward search + relaxed plan heuristic | ✅ 大多数情况满足 3s |
| **Qweak** (Bartheye & Jacopin 2005) | Prolog | Partial order planning with arithmetic constraints over time intervals | ✅ 但代码复杂 |
| Other IPC planners | Various | Various | ❌ PDDL 理解失败或 real-time 不达标 |

**FF 算法核心:** 用 relaxed planning graph (忽略 delete effects) 计算 heuristic h_FF，用 enforced hill-climbing + best-first search。这是为什么它能 fast — delete relaxation 让 heuristic computation 多项式时间。

**Qweak 算法核心:** 给 predicates 和 operators 关联 time intervals [t_low, t_up]，operator 应用产生 arithmetic constraints，当 bounds 满足时 plan found。这是一种 constraint-based temporal planning。

参考: [FF paper](https://fai.cs.uni-saarland.de/hoffmann/papers/aij01.pdf), [Qweak paper](https://planning.cis.strath.ac.uk/UKPlanSIG2005/papers/bartheye.pdf)

---

## 3. Case Study 2: VBS2

### 3.1 VBS2 简介

**Virtual Battle Space 2** (Bohemia Interactive Australia 2006-2009) 是军事 training 用的 serious game，类似 FPS 视角，支持 scenario editing 和 scripting。被多国军队用于训练。

参考: [VBS2 (archive)](https://web.archive.org/web/2009/http://www.vbs2.com/), [Bohemia Interactive Simulations](https://bisimulations.com/)

### 3.2 Real-time budget

VBS2 比 Iceblox 更宽松，因为 **take cover** 是合法的 military 行为。所以：
- 允许 troops 安全时才触发 planning
- 同样设 4 秒上限

### 3.3 Hook 选择

VBS2 提供两种 hook：
1. `OnSimulationStep` — 每帧调用 (类似 Iceblox)
2. `PluginFunction` — script 内调用，只接受 string 参数

作者选 `OnSimulationStep` 的三个理由：
- 与 Iceblox 实现一致
- 无 script 启动 overhead
- `PluginFunction` string 序列化有 overhead

Planner 打包成 **DLL** 放在 `plugins/` 文件夹，VBS2 启动时 load — 零 runtime activation overhead。

### 3.4 VBS2 High-level actions → PDDL operators

VBS2 提供 `move-unit`, `escort-unit` 等 scriptable high-level actions。这 backward validates Iceblox 中的 abstract operator 设计 — game engine 本身就在 abstract 层面思考。

关键决策：让 VBS2 自己 engine 处理 pathfinding，PDDL operators 只用 waypoints 作 parameter。这与 Iceblox 中 delegate path planning to execution module 完全一致。

### 3.5 Tactical planning example

**Scenario:** 2 名 blue force soldier (hero=machine gunner=player, companion=grenadier) 救 hostage，需 neutralize 1 个 enemy。

**PDDL problem:**
```lisp
(define (problem plan-auto)
  (:domain vbs2-strips)
  (:objects companion - unit hero - unit hostage - unit 
            enemy - unit desert-car - vehicle)
  (:init (near-object hero hero)
        (soldier-unit hero) (soldier-unit companion)
        (hostage-unit hostage) (enemy-unit enemy)
        (has-grenade companion)
        (guard-unit enemy hostage)
        (unsafe-unit hostage)
        (reachable-unit hostage) (reachable-unit hero)
        (safe-unit hostage))
  (:goal (and (liberated-unit hostage)
              (near-object hostage desert-car))))
```

**escort-unit operator:**
```lisp
(:action escort-unit
  :parameters (?soldier - unit ?hostage - unit ?vehicle - vehicle)
  :precondition (and (near-object ?soldier ?hostage)
                      (soldier-unit ?soldier)
                      (hostage-unit ?hostage)
                      (safe-unit ?hostage)
                      (reachable-unit ?hostage))
  :effect (and (liberated-unit ?hostage)
               (near-object ?hostage ?vehicle)
               (not (near-object ?soldier ?hostage))
               (not (hostage-unit ?hostage))
               (not (safe-unit ?hostage))
               (not (reachable-unit ?hostage))))
```

**Intuition:** 注意 `liberated-unit` 是作者自定义的 "mission-completion" predicate，不属于 VBS2 原生 API。这是把 tactical semantics 编码进 PDDL effect 的 pattern — 让 planner 直接 search 到 mission goal 而非 low-level action 完成。

### 3.6 自动生成 SQF scripts

Plan 找到后，系统自动生成 SQF (VBS2 scripting language) 代码：

```sqf
[companion, enemy] execVM "kill unit.sqf"
selectPlayer companion
[hero, position hostage, str hostage] execVM "move.sqf"
trg = createTrigger["EmptyDetector", position hostage]
trg setTriggerArea[3,3,0,false]
trg setTriggerActivation["WEST","PRESENT",true]
statement = '[hero, hostage, desert car] execVM "escort unit.sqf"'
trg setTriggerStatements["this", statement, ""]
```

**关键 pattern:** 用 **trigger** 串联 plan steps — 不是 sequentially execute，而是 event-driven。`createTrigger` 设定 spatial trigger area (3×3 meters)，当 WEST 单位进入时 fire `escort unit.sqf`。这让 plan execution 自动等待 precondition satisfaction（hero 必须先到 hostage 位置）。

这是 **plan execution as event-driven state machine** 的实现，比 sequential dispatch 更 robust。

### 3.7 Generated grenade throw script

```sqf
Soldier = this select 0;
Enemy = this select 1;
hint "kill-guard companion enemy";
Soldier switchmove "AwopPercMstpSgthWnonDnon_end";
array = [getPos Enemy, "VBS2 ammo G_40x46mm_HE", 15, 4, 3, 1, 0.75, 1, 5];
height = array select 2;
coords = array select 0;
radius = array select 8;
tmp setPos [(coords select 0) + (random radius) - (radius/2),
            (coords select 1) + (random radius) - (radius/2),
            height];
```

**细节分析:**
- `switchmove` 触发 throw 动画
- `VBS2 ammo G_40x46mm HE` 是 40mm 高爆榴弹（仿真美军 M433 榴弹）
- `random radius` 模拟投掷散布 — 这对应真实军事训练中的 accuracy variability
- `height = 15` 模拟抛物线峰值高度

---

## 4. Reverse Engineering: 通用 Planning Component

### 4.1 Pipeline architecture (Figure 1)

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐
│ Goal        │ -> │ Goal         │ -> │ Plan       │ -> │ Plan         │
│ Confirmation│    │ Selection    │    │ Search     │    │ Execution    │
└─────────────┘    └──────────────┘    └────────────┘    └──────────────┘
                                                              │
                                                              v
                                                       Success? -> remove goal
                                                       Failure? -> re-plan or abort
```

每个圆角矩形是一个 **thread**，classical pipeline architecture：
- 每个 thread 等前一个 thread 输出
- 整个 pipeline 由 "goals confirmation" 信号启用
- Goal set 空或 no plan found 时终止

**Goal Selection 启发式:**
- Iceblox: 按 penguin 距离 + 危险 flame 数排序，每次只处理 1 个 coin
- VBS2: 类似 distance + danger criteria

**Intuition:** Goal selection 是 reduce search space 的关键 step。Full PDDL planner 在大 goal set 上会爆炸，所以必须先 hierarchical decomposition。这与 HTN (Hierarchical Task Network) planning 思路相通。

### 4.2 Plan Execution thread (Figure 2)

```
Plan ──> [Compile high-level action] ──> [Execute low-level instructions] 
                                              │
                                              v
                                         Emergency check
                                              │
                                    ┌─────────┴─────────┐
                                    v                   v
                              Continue            Abort & re-plan
```

**关键设计:** "on the fly, one high-level action at a time" — 不预编译整个 plan。原因：
- 如果 emergency 出现，后续 compilation 浪费
- Compilation time 省下用于 emergency response
- 提供每步 emergency 检查机会

例如 Iceblox 中：`Move(s1, s2)` 在 execution 时才 A* path planning，`extract` 才触发 7 次 push 序列。VBS2 中：`escort-unit` 才生成 waypoint + trigger。

### 4.3 Take Cover (未实现 extension)

作者承认没实现 take cover procedure — 让 troops 在 planning 期间保持 safe。这是 future work。Intuition: 这是 real-time planning 的"buy time"策略，对应游戏中的 "planning stance"（如 XCOM 的 overwatch, Dragon Age: Origins 的 tactical pause）。

---

## 5. 实验数据 (隐性 data table)

虽然 paper 没有正式 benchmark table，但可以 reconstruct:

| 维度 | Iceblox | VBS2 |
|------|---------|------|
| Genre | 2D arcade | 3D serious FPS |
| Frame rate | 30 FPS | (game-dependent) |
| Time budget | 3s | 4s |
| Plan length | 2 + 2·n_danger (+n_destroy +n_align) | small (2-4 ops) |
| Planner used | FF, Qweak | FF, Qweak |
| Path planning | delegated to executor | delegated to VBS2 engine |
| Hook | custom (rewrote in C++) | OnSimulationStep (DLL) |
| Execution model | sequential + interrupt | event-driven (triggers) |
| Goal selection | per-coin | per-mission-objective |
| Take cover | ❌ | ❌ (allowed by design but not implemented) |

---

## 6. Critical Analysis 与 Intuition Building

### 6.1 Contribution 的本质

这篇 paper 的核心 contribution 不是新算法，而是一组 **engineering patterns** 让 academic PDDL planner 可用于 game。这些 patterns：
1. **Abstract operators** — 把 low-level 减弱 search complexity
2. **Delegate path planning** — 让 planner 不处理 geometric reasoning
3. **Goal selection** — hierarchical 减小 search space
4. **On-the-fly compilation** — lazy evaluation 节省 emergency 响应时间
5. **Event-driven execution** — robustness over sequential dispatch
6. **Pipeline threading** — overlap planning 与 execution

### 6.2 与现代 game AI 的关系

- **Mount & Blade, Total War, XCOM** 都用过类似 GOAP / HTN 系统
- **CryEngine** 有 native planning integration
- 现代趋势是 **behavior trees + utility AI** 而非纯 PDDL，因为 PDDL verbosity 仍然痛点
- **HTN (Hierarchical Task Network)** planners 如 SHOP, PANDA 在 game 中更流行，因为 decomposition 更直观

参考: [HTN Planning](https://ojs.aaai.org/index.php/aimagazine/article/view/2638), [Behavior Trees vs GOAP](https://www.gamedeveloper.com/programming/behavior-trees-for-ai-how-they-work)

### 6.3 与现代 LLM-based planning 的联想

作为 Karpathy 你可能感兴趣：当下 LLM 也能做 "planning" — ReAct, Tree of Thoughts, Plan-and-Execute。但 LLM planning 与 PDDL planning 本质区别：
- PDDL 是 **symbolic, sound, complete** (within search bounds)
- LLM 是 **subsymbolic, approximate, heuristic**

可能的 hybrid：LLM 作 goal selection + operator crafting，PDDL planner 作 sound plan search。这与 paper 中 "goal selection" thread 完美契合 — LLM 可以基于 game state narrative 选择 next goal，传统 planner 保证找到满足 goal 的 action chain。

### 6.4 PDDL 的真实 expressiveness cost

paper 没讨论 PDDL 2.1+ 的 numeric fluents 和 durative actions。如果加进 weapon 滑动时间、flame 移动 timing，planner 需要 temporal reasoning，复杂度从 PSPACE 跳到 EXPTIME。这是为什么作者严格保持 STRIPS-level (PDDL 1.0) subset。

### 6.5 Iceblox → VBS2 的 transfer learning insight

最 striking 的 claim: "everything from Iceblox was reused or adapted for VBS2"。这说明 abstract operator + delegate execution + goal selection 这个 pattern 是 genre-agnostic 的。从 2D grid maze 到 3D military sim 都适用。这是 paper 最强的 generalization argument。

### 6.6 失败的对比

paper 提到 "Other planners available from the IPC web site either failed to understand our PDDL problems or else failed to satisfy our real-time need"。这暗示 IPC planner 对 game-derived PDDL (有大量 predicates、specific domain structure) 并不 robust — 学术 benchmark domain (blocks, logistics, gripper) 与 game domain distribution 不同。这是重要的 negative result。

---

## 7. 个人 Intuition 总结

读完这篇 paper 我 build 出的 mental model：

1. **Real-time AI planning ≠ fast planner** — real-time 是 game-world-relative concept，budget 由 fastest adversary 决定，与 planner 实测 speed 是间接关系。

2. **Abstraction 是一切** — abstract operators 不是 "妥协"，而是正确的 engineering boundary。Low-level reasoning (pathfinding, animation) 应该在 execution 层，因为那是 game engine 已经 optimized 的领域。

3. **PDDL 的 verbosity 是 feature 也是 bug** — feature 因为 explicit 且 sound；bug 因为 game designer 不愿写 9 个 parameter 的 operator。这是 PDDL 没在 game industry 普及的根本原因，而非 performance。

4. **Plan execution 比 plan search 更难** — Figure 2 的 complexity 反映了这点。Re-plan vs repair、emergency handling、event-driven sequencing 都在 execution 层。

5. **Academic AI planning 与 game AI 的 gap 主要是 vocabulary gap** — paper 通过 "operator mapping" 桥接，但这需要大量 manual engineering。未来方向可能是 LLM-assisted PDDL domain authoring。

参考链接:
- [PDDL specifications](https://planning.github.io/pddl-spec/)
- [IPC competition archives](http://www.icaps-conference.org/index.php/Main/Competitions)
- [GOAP original paper (Orkin)](https://www.gamedeveloper.com/programming/three-states-and-a-plan-the-a-i-of-f-e-a-r)
- [FF planner homepage](https://fai.cs.uni-saarland.de/hoffmann/ff.html)
- [STRIPS original paper (Fikes & Nilsson 1971)](https://ai.stanford.edu/users/nilsson/OnlinePubs-Nils/PublishedPapers/strips.pdf)
- [HTN Planning survey](https://arxiv.org/abs/1406.1938)
- [Behavior trees in game AI](https://www.gamedeveloper.com/programming/behavior-trees-for-ai-how-they-work)
