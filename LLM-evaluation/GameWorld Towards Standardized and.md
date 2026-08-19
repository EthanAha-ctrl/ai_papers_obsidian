---
source_pdf: GameWorld Towards Standardized and.pdf
paper_sha256: 5966ec57c1ef2f7efff58859a8948e1c3288451241c696241c0af4b85412eb73
processed_at: '2026-08-19T08:36:07-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Andrej，我们抛开学术黑话，直接从 first principles 来聊聊 GameWorld 这篇 paper 到底在搞什么。

用一句话总结：**GameWorld 是一个给 Multimodal LLM (MLLM) 玩 browser game 量身定制的考场。它强行暂停游戏让你慢慢想，并且直接读取游戏内存来判断你考了多少分，结果发现目前最强的 AI 玩家，连刚上手的菜鸟人类都打不过。**

下面我把这篇 paper 拆解成几个直觉层面，顺带聊聊这背后暴露出的 VLA (Vision-Language-Action) 模型的根本瓶颈。

---

### 1. 为什么老套路的评测全是坑？

要让 AI 玩游戏，以前的 benchmark 设计存在三个致命问题，这直接导致测出来的分数根本不可信：

**坑一：Latency coupling (反应速度与脑力混为一谈)**
在实时游戏里，如果模型需要推理 2 秒钟，游戏里的角色早就掉下悬崖了。你测的到底是模型的决策质量，还是模型的反应速度？这两者被耦合在了一起。VideoGameBench (https://arxiv.org/abs/2505.08425) 尝试过引入 paused track，但 GameWorld 把它做成了核心 default 设定。

**坑二：Noisy Evaluation (用感知模型去评感知模型，相当于用弯尺量弯尺)**
以前的 benchmark 怎么判断 AI 有没有过关？要么用 OCR 去读画面上的分数，要么写一堆像素规则，甚至用 VLM-as-judge 去看截图打分。问题在于，MLLM 本身看图就有幻觉，你用一个同样会看错的 VLM 去评判另一个 VLM 玩得好不好，这完全是盲人摸象。

**坑三：Heterogeneous Action Interfaces (动作空间乱七八糟)**
不同的模型，输出的动作格式完全不同。同样是点击屏幕，A 模型输出 `left_click(x,y)`，B 模型输出 `computer(action="click", coordinate=[x,y])`。这导致不同模型之间的分数差异，很大一部分是 parser 适配质量造成的 noise。

---

### 2. GameWorld 的三个核心 Hack

为了绕开这些坑，GameWorld 在 architecture 上做了极其精妙的处理，它建立了一个 closed-loop 的 Observation-Action-Evaluation 流程：

#### Hack A: Browser Sandbox with Pause (可以叫停的考场)
Sandbox 在 model 推理时，直接调用 `gameAPI.pause()` 暂停游戏 clock。模型想完之后，输出 action，再 `resume()`。
这彻底解耦了：
$$ \text{score} \perp \text{latency} $$
分数严格只反映决策质量。同时它也提供了 GameWorld-RT 变体（不暂停），让你去测真实世界的交互难度。

#### Hack B: State-Verifiable Evaluator (直接看底牌)
这是最漂亮的设计。论文要求每个 benchmark game 必须暴露一个 `window.gameAPI` 接口，提供 `init()`, `reset()`, 和 `getState()`。

Evaluator 根本不看截图！它直接读取游戏引擎内部的 serialized state。比如在 Mario 游戏里，它读到的是这种 JSON：
```javascript
{
  "game_state": {
    "player": { "x": 128, "y": 80, "alive": true, "name": "Mario" },
    "score": 3200
  },
  "metrics": { "lives": 3, "coins": 8, "distance": 3200 }
}
```
这样，evaluation 变成了 deterministic 的数据库查询，zero perceptual noise。总共手工 instrument 了 233 个 state fields，平均每个游戏 6.85 个变量。这种获取 dense, verifiable reward signal 的做法，对于未来用 RL 训练 VLA 模型极其重要，因为这等价于一个完美的、无延迟的 reward model。

#### Hack C: Unified Action Space + Dual Interfaces (统一考试大纲)
所有模型的动作，最后都被 normalize 到 7 个 atomic human-computer interaction events：
$$ \mathcal{A}_{\text{atomic}} = \{\text{mouse\_move}, \text{mouse\_down}, \text{mouse\_up}, \text{key\_down}, \text{key\_up}, \text{scroll}, \text{wait}\} $$

在这个底层之上，评测两类 agent interface：
1. **Computer-Use Agent (CUA)**：模型自己输出 `mouse_move(120, 340)` 或 `press_key("Space")`。它要自己搞定 pixel-level 的 grounding。
2. **Generalist Agent + Semantic Action Parsing**：模型只输出 semantic action，比如 `jump`。系统用一个 deterministic parser 把 `jump` 确定性地映射到 `press_key("Space")`。

这个设计的精妙之处在于：它消除了 parser 带来的随机性，让你可以公平对比“模型自己做 fine-grained control (CUA)”和“模型只做 high-level planning”到底谁强。

---

### 3. 数学直觉：为什么只看 Success Rate 是骗自己？

论文定义了两个指标。如果只看 Success Rate ($\mathcal{SR}$)，你会发现所有模型都是废物（分数都在 10-20% 挣扎），掩盖了它们到底差在哪里。

所以引入了 Progress ($\mathcal{PG}$)：

$$ \text{progress}_i = \text{clip}_{[0,1]}\!\left(\frac{q_i^{\max} - b_i}{\tau_i - b_i}\right) $$

变量拆解：
- $q_i^{\max}$：整个 run 中，模型拿到的最高历史分数。因为是 best progress so far，所以如果中途死掉 reset，run-level 的 best score 不会被清零。
- $b_i$：起跑线分数。
- $\tau_i$：target 分数（终点线）。
- 分子 $q_i^{\max} - b_i$：模型实际跑了多远。
- 分母 $\tau_i - b_i$：总路程有多远。

直觉：如果没通关，不要直接打 0 分，看看你走到全程的百分之几了。这给了模型 partial credit。这对于构建 RL 训练的 dense reward 也极具启发。

最后平均一下：
$$ \mathcal{PG} = \frac{1}{N} \sum_{i=1}^{N} \text{progress}_i $$

---

### 4. 实验数据讲了什么大实话？

Table 6 是全场最核心的数据。我把关键点拉出来：

**人类 vs AI 的鸿沟：**
- Novice Player (新手): SR = 55.3%, PG = 64.1%
- Expert Player (老手): SR = 77.1%, PG = 82.6%
- 最好的 AI (Gemini-3-Flash-Preview, Generalist): PG = 41.9%, SR = 21.2%
- 最好的 CUA (Seed-1.8): PG = 39.8%

**直觉结论：** 目前最强的 MLLM，在同样的 100 步 action budget 下，连刚摸键盘的新手都打不过。AI 普遍能做出 partial progress（走个 40%），但极难 reliable completion（通关率只有 20%）。

**Capability-Aligned Curriculum (能力雷达图):**
论文把游戏按能力瓶颈重排成 5 级：
- Level-1 (Basic Control & Timing Grounding): 比如精准点一下、按一下空格。
- Level-2 (System-1 Reactive Control): 跑酷、Flappy Bird 这种高频反应。
- Level-3 (System-2 Spatial Navigation): 走迷宫、3D 寻路。
- Level-4 (Symbolic Reasoning & Strategy): 2048、扫雷、俄罗斯方块。
- Level-5 (Open-World Coordination): Minecraft、经营类游戏。

雷达图显示了一个极其反直觉的现象：**模型在 Level-4 (Symbolic Reasoning) 和 Level-2 (Reactive Control) 上表现最好，但在 Level-1 (Basic Timing Grounding) 和 Level-5 (Open-World) 上断崖式下跌。**

这说明什么？MLLM 继承了 LLM 的逻辑推理能力（Level-4），并且对高频 frame-by-frame 的反应也能应付（Level-2，因为 paused 模式下它有充足时间看每一帧）。但是，**当需要极精准的 timing 控制时，或者需要跨越极长 horizon 维持一个连贯的 subgoal 时，模型直接拉胯。** Level-1 失败证明了当前 VLA 模型的 action grounding 极其粗糙；Level-5 失败证明了 context window 再长，模型也会陷入 "重复无效动作且无法 self-correction" 的死循环。

---

### 5. 极其有趣的 Control Variables 发现

**Memory Rounds 的不对称效应 (Table 9):**
论文做了 memory ablation，给模型看前 0, 1, 2 轮的 history。
- Generalist Agent：memory 越多，PG 略微上升（30.0 -> 30.6）。
- CUA Agent：memory 越多，PG 反而下降（30.3 -> 28.7）。

直觉解释：Generalist 的 history 是 `move_left`, `jump` 这种 semantic token，信息密度高，模型能从中提取有用上下文。CUA 的 history 是 `mouse_move(123, 456)`, `click(345, 678)` 这种数字垃圾，历史越长，越容易 distract 模型。这证明 long context memory 对不同 interface 有 distinct trade-offs，不是无脑加 memory 就有好处的。

**Action Validity (IAR):**
公式：
$$ \text{IAR} = 1 - \frac{\sum_{r \in \mathcal{R}} \#\text{valid\_actions}(r)}{\sum_{r \in \mathcal{R}} \#\text{proposed\_actions}(r)} $$
解释：1 减去（合法动作数 / 提出动作总数）。

结果发现：像 GLM-4.6V 在 Generalist 模式下，IAR 达到了 8.3%！意味着它每输出 100 个动作，有 8 个是直接乱说、不在合法动作空间里的（比如在键盘游戏里输出 `craft_a_workbench()`）。这叫 Instruction-following failure，说明模型在长交互中会 "忘记" 自己能干什么。这再次暴露了 MLLM 在 agentic loop 中的脆弱性。

---

### 6. 为什么这对 Build Intuition 很重要？

这不仅仅是做一个游戏 benchmark。Karpathy，你一直关注 physical world agent 和 VLA 的发展。GameWorld 的结果为我们提供了关于训练 generalist agents 的三个关键直觉：

1. **Perception 和 Action Grounding 仍然是阿喀琉斯之踵。** 我们以为 MLLM 会看图了，但在需要 precise timing 和 spatial grounding 的场景（Flappy Bird, 平台跳跃），它依然像一个没长手眼的婴儿。用 LLM 做大脑，直接硬接 mouse/keyboard pixel 输出（CUA 模式），在 fine-grained control 上依然拼不过有 deterministic parser 做底层映射的 Generalist 模式。
2. **Outcome-based Verifiable State 是终极 Reward Signal。** 为什么以前做 game agent 很难？因为 RL 的 reward 很难提取。GameWorld 展示了一条明路：通过 instrument 游戏引擎，拿到 deterministic state。这种方法可以直接迁移到任何有 API 的环境（比如 OSWorld 训 computer-use agent），用 deterministic env state 作为 dense reward 去做 RL post-training。这避免了用 LLM 做 judge 带来的 reward hacking。
3. **长视域规划需要新的 memory 架构。** 模型在 Minecraft (Level-5) 里只能拿到 90% progress 却无法 finish，陷入局部 loop。这说明纯粹把 history 塞进 context window 是不够的，我们需要 external 的 goal-state tracker 或者 hierarchical policy 来保持 subgoal 的稳定收敛。

关联阅读参考：
- OSWorld: https://arxiv.org/abs/2404.07972
- BALROG: https://arxiv.org/abs/2411.04405
- WebArena: https://arxiv.org/abs/2307.13854
- Voyager (Minecraft lifelong learning): https://arxiv.org/abs/2305.16291

这个 paper 表面上是测游戏，实质上是在宣告：目前的 multimodal agent 在 embodied interaction 上，连基础的 action grounding 和 memory consolidation 都还没搞定，离 reliable physical world interaction 还差得很远。

---

# GameWorld: Multimodal Game Agents 的标准化与可验证评测深度解析

下面我尝试从多个层面把这篇 paper 拆解给你，目标是 build intuition——为什么 GameWorld 这么设计、它解决了什么真实的 evaluation 痛点、各 module 如何耦合、为什么指标这么定义，以及实验数据到底告诉了我们什么。

---

## 1. Motivation: 为什么现有的 game agent benchmark 都不够

论文 motivation 部分指向三个 systematic 问题，我用一句话分别概括：

1. **Heterogeneous action interfaces**：同一个 "click screen" 动作，不同 model 的 tool schema 完全不同——一个叫 `left_click(x,y)`，另一个叫 `computer(action="click", coordinate=[x,y])`。这导致 cross-model comparison 失真，因为 score 里混入了 "parser 适配质量" 的 noise。
2. **Latency coupling in real-time interaction**：在实时游戏中，慢 model 推理 2 秒，角色可能已经掉下 platform，score 同时反映了 thinking quality 和 response speed——两个 orthogonal 维度被 conflated。
3. **Lack of outcome-based verifiable evaluation**：现有 benchmark 大量依赖 OCR、pixel heuristic、VLM-as-judge，所有这些 pipeline 都会注入 perceptual noise，使 results 不可复现、不可诊断。

GameWorld 的 design choice 就是逐个击破这三点。这种 motivation-driven 的设计在 OSWorld（https://arxiv.org/abs/2404.07972）和 WebArena（https://arxiv.org/abs/2307.13854）的传统中延续下来，但是把它转移到 game domain。

---

## 2. 整体架构：四大模块的 Observation-Action-Evaluation Loop

Figure 2 描述的是 closed-loop：

$$
\text{MLLM Agent} \xrightarrow{\text{action}} \text{Browser Sandbox} \xrightarrow{\text{screenshot}} \text{MLLM Agent} \quad \text{while} \quad \text{Evaluator} \xleftarrow{\text{serialized state}} \text{Game API}
$$

四个 module 互相解耦：

- **Module (i) MLLM as game agent**：可以是 CUA，也可以是 Generalist；两者最终都 normalize 到同一个 unified control space。
- **Module (ii) Browser-based sandbox**：管理 game execution 和 pause机制。
- **Module (iii) Games & tasks library**：34 games × 170 tasks，每 task 含 instruction + init state + target metric + eval config。
- **Module (iv) Outcome-based state-verifiable evaluator**：从 `window.gameAPI.getState()` 直接读 serialized state，deterministic 计算 SR 和 PG。

这里关键的设计是 evaluator **不依赖于 screenshot**——它绕过了 perception pipeline，直接读 game 的内部 state field。这就是 "verifiable" 的字面意思。

---

## 3. 两种 Agent Interface：CUA vs. Generalist

### 3.1 Unified Control Space

所有 action 最终都被 normalize 到 7 个 atomic human-computer interaction events：

$$
\mathcal{A}_{\text{atomic}} = \{\text{mouse\_move}, \text{mouse\_down}, \text{mouse\_up}, \text{key\_down}, \text{key\_up}, \text{scroll}, \text{wait}\}
$$

这是 executor-level 的最低 common denominator。论文在 Appendix D.2 又引入一个 slightly higher-level 的 normalized runtime layer（包含 `click`, `click_hold`, `drag`, `type`, `press_key`, `press_keys`, `wait`），原因是 Playwright 本身就提供这些 higher-level primitives，重新拆成 atomic 反而冗余。

### 3.2 Computer-Use Agent (CUA)

CUA 直接输出 `mouse_move(x,y)`, `left_click(x,y)`, `press_key(key)` 这种 low-level 指令。模型同时承担两个职责：

- **Strategic decision-making**：选什么动作
- **Precise action grounding**：选什么 pixel 坐标、什么 key timing

CUA 的约束是 "one-action-per-step"——每一步只能发一个 atomic action，禁止 macro（比如 `move_to_enemy_and_attack()`）。这是为了让所有 model 在同一个 action budget 下被公平评测。

### 3.3 Generalist Agent + Semantic Action Parsing

Generalist agent 通常不擅长输出精确的 pixel 坐标和精细 key timing，所以 GameWorld 引入 **Semantic Action Parsing**：

$$
\text{Semantic Action} \xrightarrow{\text{deterministic parser}} \text{Fixed Low-Level Interaction Command}
$$

例如在 2048 里，model 输出 semantic action `move_left`，parser 确定性地把它映射到 `press_key("ArrowLeft")`。这个 mapping 是写在 YAML registry 里的，因此**完全 deterministic**——没有 LLM-in-the-loop 的 parsing 不确定性。

设计原则叫 **Action Atomicity**：每步一个 semantic command，禁止 multi-command macro。这点和 CUA 是一致的，使两个 interface 可比较。

### 3.4 Agent Harness 的四个组件

每个 model 都被 wrap 在 shared agent harness 里，包含：

1. **Structured Prompt**：固定四段——`#Game Rules` / `#Role and Controls` / `#Task Instruction` / `#Output Format`。这种结构化 prompt 设计借鉴了 ReAct（https://arxiv.org/abs/2210.03629）和 modular harness 思路，目的是 reduce prompt-induced variance。
2. **Rolling Memory**：记录最近几轮 `user_prompt → screenshot → reasoning → action`，作为 `Action History` block prepend 到当前 observation。
3. **Reasoning**：允许模型有 chain-of-thought，对 long-horizon 任务尤其重要。但 reasoning 会增加 latency，所以有 trade-off（Section 4.5.1 专门分析）。
4. **Customized Function Calling**：每个 model 用自己 native 的 function calling 接口（OpenAI function calling / Claude tool use / Gemini function declarations），保持 native agentic 能力。

---

## 4. Benchmark 设计：34 Games × 170 Tasks

### 4.1 五大 genre 的 capability 分层

| Genre | Games 数 | 核心能力 |
|---|---|---|
| **Runner** | 8 | 高频 reactive control + precise timing |
| **Arcade** | 7 | 快速 closed-loop control + multi-entity tracking |
| **Platformer** | 8 | physics-aware spatial navigation |
| **Puzzle** | 7 | discrete state-space 上的 long-horizon planning |
| **Simulation** | 4 | open-ended, multi-objective, resource management |

这个分类不是随意的——它在 Section 4.4 被进一步重新组合成 5-level **Capability-Aligned Curriculum**，从 Level-1（basic control + timing grounding）到 Level-5（open-world coordination）。这种双视角（genre vs. capability）的设计让结果可被 diagnostic。

### 4.2 任务的两个核心指标

论文定义了两个互补的 metrics。先看公式：

设 $\mathcal{R}$ 是所有 evaluated runs 的集合，$N = |\mathcal{R}|$。对第 $i$ 个 run：

- $q_{i,t}$：第 $t$ 步从 verifiable game state 读到的 task score（可以是单一 scalar field，也可以是多个 aggregate fields 之和）
- $b_i$：task 的 starting score
- $\tau_i$：configured task target score（满足 $\tau_i > b_i$）
- $q_i^{\max} = \max_t q_{i,t}$：整个 run 中观察到的 best score

**Run-level Progress（公式 1）**：

$$
\text{progress}_i = \text{clip}_{[0,1]}\!\left(\frac{q_i^{\max} - b_i}{\tau_i - b_i}\right)
$$

变量含义：
- 分子 $q_i^{\max} - b_i$：相对起点的 best-ever advancement
- 分母 $\tau_i - b_i$：从起点到 target 的总距离
- $\text{clip}_{[0,1]}$：保证 progress 落在 $[0,1]$，防止"超 target"或者"起点以下"造成 $>1$ 或 $<0$

注意一个细节：**Reset-on-fail 机制下，episode-local 的 score 会被 clear，但 run-level 的 best progress 被保留**。这是为了避免单次 early mistake 把整个 run 归零。这种 "best progress so far" 的设计在 RL eval 里也是常见做法。

**Aggregate Metrics（公式 2）**：

$$
\mathcal{SR} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\text{status}_i = \text{success}], \quad \mathcal{PG} = \frac{1}{N} \sum_{i=1}^{N} \text{progress}_i
$$

变量：
- $\mathcal{SR}$：Success Rate，所有 run 里达到 target 的比例
- $\mathcal{PG}$：Progress Gain，所有 run 的平均 normalized progress
- $\mathbf{1}[\cdot]$：indicator function，满足条件输出 1，否则 0

为什么需要 $\mathcal{PG}$ 而不仅用 $\mathcal{SR}$？因为 paper 的核心 finding 之一是：当前 MLLM agent **能 partial progress 但远不能 reliable completion**。如果只看 SR，所有 model 看起来都"很差"——但事实上它们在 Runner genre 上能跑到 50%+ progress。$\mathcal{PG}$ 提供了 fine-grained 的 partial credit signal。

---

## 5. Browser Sandbox 的 Pause 机制

这是 GameWorld 区别于 VideoGameBench（https://arxiv.org/abs/2505.08425）的关键设计之一。

### 5.1 Readiness Gate

Sandbox 不会在 game 一加载就开始评测，而是 wait 直到 game 进入 actionable status。Table 12 定义了状态机：

| Status | 含义 |
|---|---|
| `loading` | 资源还没准备好 |
| `menu` | 在 title screen / level select，未进入 play state |
| `ready` | 已初始化，但需要一次 start action |
| `playing` | gameplay loop active，安全可控制 |
| `paused` | 暂停（用于 sandbox pause mechanism） |
| `terminal` | episode 结束 |

只有 `ready` 或 `playing` 才会 trigger agent action。

### 5.2 Latency Decoupling

Inference 时，sandbox 调用 `gameAPI.pause()`，game clock 停止；模型输出 action 后，调 `gameAPI.resume()`。这保证：

$$
\text{score}_{\text{agent}} \perp \text{latency}_{\text{model}}
$$

每个 model 面对的 game dynamics 是 identical 的。这是 "paused benchmark" 的设计哲学。

### 5.3 GameWorld-RT：实时变体

但 real world 不是 paused 的，所以论文还建立了 GameWorld-RT——不暂停，让 latency 成为 task 的一部分。Table 8 给出了对照：

| Model | RT sec/step | SR | PG |
|---|---|---|---|
| Qwen3-VL-235B (CUA) | 6.2 | 17.1 | 33.2 |
| Qwen3-VL-30B (CUA) | 2.4 | 15.6 | 33.0 |
| Qwen3-VL-235B (Generalist) | — | 16.8 | 34.0 |
| Qwen3-VL-30B (Generalist) | 3.44 | 15.6 | 32.9 |

一个有意思的观察：235B 比 30B 慢 2-3 倍，但 PG 只高一点点；30B 推理快但 PG 没显著低。这说明在 real-time 设定下，**thinking speed 和 action timing 是 coupled 的**，单独变快解决不了问题，单独变准也不够。

---

## 6. State-Verifiable Evaluation 的技术细节

### 6.1 gameAPI Contract

每个 benchmark game 必须暴露 `window.gameAPI`，含三个 callable methods：

```javascript
window.gameAPI = {
  init(config),       // 初始化
  reset(options),    // 重置 episode
  getState()         // 返回 serialized verifiable state
}
```

`getState()` 返回的结构（以 17_mario-game 为例）：

```javascript
{
  gameId: "17_mario-game",
  timestampMs: 1760001234567,
  gameTimeMs: 18420,
  status: "playing",
  terminal: { isTerminal: false, reason: null },
  game_state: {
    score: 3200,
    level: "1-1",
    progress: 0.37,
    player: { x: 128, y: 80, vx: 0, vy: 0, power: 1, alive: true, name: "Mario" },
    board: null,
    entities: null
  },
  metrics: {
    lives: 3, coins: 8, distance: 3200, attempts: 1,
    time_left_s: 999, enemies_alive: 5, level_progress_percent: 42
  },
  raw: { /* game-specific 原始字段 */ }
}
```

三层结构：
- `game_state`：用于 evaluation 的 structured in-game state
- `metrics`：compact comparable counters
- `raw`：optional，保留 game-specific 字段供 further analysis

总共 instrument 了 **233 个 task-relevant state fields**，平均 6.85 fields/game。每个 field 是 manual designed 的，对应一个 gameplay quantity（score, level, coordinates, lives, coins, checkpoints...）。

### 6.2 与 VLM-as-Judge 的对比

这里 design 的核心 insight 是：**game state 本身已经是 ground truth**。VLM-as-judge 之所以不可靠，是因为它需要在 screenshot 上做 perception——而这正是被评测 model 自己的 bottleneck。用 perception 去评 perception，等于用尺子量尺子。

GameWorld 直接读 game 的内部 state，绕过 perception，所以叫 **noise-free and fully reproducible**。这是它能做 robustness study 的前提。

---

## 7. 主实验结果解析

Table 6 是论文最关键的数据表，我重点挑几个数字讲 intuition。

### 7.1 Human Baseline

- **Novice Player**：SR=55.3%, PG=64.1%
- **Expert Player**：SR=77.1%, PG=82.6%

注意 human 用同样的 100-action budget。这意味着 budget 不是 limiting factor——是 capability。

### 7.2 Top Agents 的表现

**Generalist Agent**：
- Gemini-3-Flash-Preview：PG=41.9, SR=21.2 → 排名第 1
- GPT-5.2：PG=40.6, SR=20.6 → 第 2
- Claude-Sonnet-4.6：PG=39.3, SR=20.6 → 第 3

**Computer-Use Agent**：
- Seed-1.8：PG=39.8, SR=20.0 → CUA 第 1
- Claude-Sonnet-4.6：PG=38.3, SR=19.4 → 第 2

**Key insight**：最好的 agent 的 PG 大约 41-42，而 Novice human 是 64，Expert 是 82.6。**Gap 巨大**——连 novice 都不到。

### 7.3 Genre-Level 模式

- **Runner**：很多 model PG ~50-55（最高 genre），因为这种游戏 reactive control 多、long-horizon planning 少
- **Puzzle**：Generalist agent 表现强（GPT-5.2 PG=56.2），因为 symbolic reasoning 是 MLLM 的强项
- **Simulation**：几乎全军覆没（多数 PG < 20），因为 open-ended + long-horizon 是 MLLM 的致命弱点

---

## 8. Benchmark Robustness Study

Table 7 给出 4 个 Qwen setting 的 10 次 rerun mean ± std：

| Setting | Overall SR | Overall PG |
|---|---|---|
| Qwen3-VL-30B (CUA) | 12.7 ± 1.2 | 30.9 ± 1.1 |
| Qwen3-VL-30B (Generalist) | 12.5 ± 1.3 | 30.7 ± 1.1 |
| Qwen3-VL-235B (CUA) | 13.8 ± 0.7 | 30.4 ± 0.7 |
| Qwen3-VL-235B (Generalist) | 13.6 ± 1.4 | 30.1 ± 0.5 |

PG 的 std 都在 0.5-1.4 之间，single-digit band。这是论文一个 strong claim：**GameWorld 是 reproducible measurement platform 而非 one-off leaderboard snapshot**。

Figure 4 进一步展示 per-game 的 run-to-run variance：大部分 game 的 error bar 很紧，variance 大的集中在 control-sensitive 高难度 game（Hextris, Cubefield, Wordle, World's Hardest Game 2）——这是 expected 的，因为这些 game 一个 timing 错误就大幅改变 outcome。

---

## 9. Capability-Aligned Curriculum（5-Level）

Figure 5 把 34 个 game 重新组织成 5 个 capability bottleneck level：

- **Level-1**：Basic Control & Timing Grounding（breakout, core-ball, stack）——能否稳定输出 valid atomic action 并在合适时机 trigger
- **Level-2**：System-1 Reactive Control（chrome-dino, flappy-bird, temple-run-2, doodle-jump, ...）——高频 reflex
- **Level-3**：System-2 Spatial Navigation（mario, pacman, astray, wolf3d, ...）——2D/3D 几何建模 + pathfinding
- **Level-4**：Symbolic Reasoning & Strategy（2048, minesweeper, tetris, wordle, hextris）——离散状态空间 strategic planning
- **Level-5**：Open-World Coordination（fireboy-watergirl, minecraft-clone, monkey-mart）——多 subgoal 协调

雷达图显示：model 的能力在 **Level-4 和 Level-2 上 peak**（MLLM 擅长 symbolic reasoning 和 reactive control），但在 **Level-1 和 Level-5 上 sharp drop**。

这个结果初看反直觉：Level-1 明明最简单啊？但仔细想——**Level-1 失败原因不是规划难，是 action grounding 弱**：MLLM 不能稳定地输出 valid pixel coordinate 或者正确 timing。这其实呼应了 fine-grained action failure 的发现。

Level-5 失败原因是 long-horizon memory failure：open-ended 任务里 model 会进入"重复无效 action"的 loop，无法 self-crection。

---

## 10. Context-Memory Sensitivity（Table 9）

| Memory Rounds | Interface | Input Tokens | sec/step | PG |
|---|---|---|---|---|
| 0 | Generalist | 1278 | 5.5 | 30.0 |
| 0 | CUA | 1891 | 7.2 | 30.3 |
| 1 | Generalist | 2171 | 6.8 | 30.1 |
| 1 | CUA | 3771 | 10.1 | 29.0 |
| 2 | Generalist | 3052 | 8.6 | 30.6 |
| 2 | CUA | 5627 | 12.8 | 28.7 |

**Asymmetric 效应**：
- Generalist：memory ↑ → PG 轻微 ↑（30.0 → 30.6）
- CUA：memory ↑ → PG ↓（30.3 → 28.7）

论文解释：Generalist 的 semantic trajectory 信息密度高，history 有用；CUA 的 low-level action trace 信息密度低，长 history 反而 distractor 多。这是 interface-aware trade-off 的直接证据。

同时 latency 显著增加（Generalist 5.5→8.6s，CUA 7.2→12.8s），所以 memory 是 selective benefit 不是 uniform。

---

## 11. Action Validity: IAR（Invalid Action Rate）

公式 3：

$$
\text{IAR} = 1 - \frac{\sum_{r \in \mathcal{R}} \#\text{valid\_actions}(r)}{\sum_{r \in \mathcal{R}} \#\text{proposed\_actions}(r)}
$$

变量：
- $\#\text{valid\_actions}(r)$：run $r$ 里通过 tool-call parsing + role constraint + parser check 的 action 数
- $\#\text{proposed\_actions}(r)$：run $r$ 里 model 提出的总 action 数
- $\mathcal{R}$：所有 run 集合

IAR 拆成两类 invalid action：
- **NTC (No-Tool-Call)**：模型输出 free-form text，没生成 tool call——常见原因是 reasoning 太长被 truncate
- **OOS (Out-of-Space)**：模型生成了 tool call，但 action 不在合法 action space——比如 keyboard-only game 里输出 `craft_a_workbench()`

Table 11 的关键数字：
- 大多数 top model 的 IAR ≈ 0%
- 但 **GLM-4.6V** Generalist IAR = 8.3%（NTC=7.6%, OOS=0.7%）——异常高
- **UI-TARS-1.5-7B** CUA IAR = 0.4%
- **Qwen3-VL-30B-A3B** Generalist IAR = 2.7%

这告诉我们：即使是 frontier 模型，在 long interactive context 下也会"忘记"合法 action space。weak model 尤其严重。

---

## 12. 四种 Failure Mode 的分类学

Section 4.5.4 给出可解释的失败模式分类：

1. **Perception failures**：misread visual state（错认 obstacle 位置、误判 traversable region），尤其在 cluttered scenes / partial observability 下严重
2. **Fine-grained action failures**：high-level intent 对，但 low-level execution 错（jump timing 错、key-combo duration 错）——典型是 Flappy Bird case study
3. **Instruction-following failures**：违反 declared controls / output schema / task constraints，长 trajectory 后 drift away from goal
4. **Long-horizon memory failures**：lose critical context，重复无效 action，进入无 self-correction 的 loop

这四类不是 mutually exclusive 的，但作为 diagnostic 框架很有价值——它们指向不同的 fix 方向：perception 失败要改 vision tower，fine-grained 失败要改 action grounding，instruction-following 失败要改 harness，memory 失败要改 context management。

---

## 13. Case Study 深度解析

### 13.1 Mario Game Interface Comparison（Figure 6）

论文把 CUA 和 Generalist 的 trajectory 对齐展示，控制变量：同 backbone、同 game environment，只差 interface。

- CUA 输出：`hotkey(key='arrowright')` 或者 `click(point='<point>640 360</point>')`
- Generalist 输出：`move_right` semantic action

Generalist 在 Mario 这种 platformer 上略占优势（Table 6：Claude Generalist PG=37.0 vs CUA PG=36.5），因为语义抽象屏蔽了 pixel-level grounding noise。

### 13.2 Minecraft Resource Collection（Figure 7）

这是 Level-5 long-horizon 失败的典型。Agent 反复 mine resource，reach 90% progress 但没在 step budget 内 finish collection target。失败模式是"missing closure"——局部 plausible 但全局未闭合。这呼应了 long-horizon memory failure。

### 13.3 Flappy Bird Timing（Figure 8）

连续 frame 看起来几乎 identical，但正确 action 在 wait 和 flap 之间交替。微小 timing 错误就决定成败。这是 fine-grained action failure 的极致 case，也是 real-time interaction 难度的代表。

---

## 14. Cost Analysis（Table 13）

总评测成本（all listed models，170 tasks）= **$815.19 USD**。

最贵的：
- Claude-Sonnet-4.6 Generalist：$244.03
- Claude-Sonnet-4.6 CUA：$172.46
- GPT-5.2：$110.68

最便宜的：
- Qwen3-VL-Plus CUA：$4.99
- Grok-4.1-Fast-Reasoning：$9.86

这个 cost 拆解对社区很重要——它告诉研究者复现 full benchmark 大概需要多少 budget。Token usage 上，Generalist 通常比 CUA 输入 token 多（因为更长的 semantic action list），但输出 token 少（CUA 要输出详细 coordinates）。

---

## 15. 与相关工作 Position 的对比

Table 2 给出最系统的对比：

| Benchmark | Games | Tasks | Models | Vision-Centric | Config. Init. State | Task-Oriented | Parallel Inst. | Verif. Eval. |
|---|---|---|---|---|---|---|---|---|
| GameQA | 30 | 158 | 8 | ✗ | NA | ✗ | NA | ✗ |
| VideoGameQA | 800+ | 9 | 16 | ✓ | NA | ✗ | NA | ✗ |
| MCU | 1 | 150 | 4 | ✓ | ✓ | ✗ | ✗ | ✗ |
| LMGame-Bench | 6 | 6 | 13 | ✗ | ✗ | ✗ | ✗ | ✓ |
| VideoGame-Bench | 23 | 23 | 5 | ✓ | ✗ | ✓ | ✗ | ✗ |
| FlashAdventure | 34 | 34 | 7 | ✓ | ✗ | ✗ | ✗ | ✓ |
| V-MAGE | 5 | 30 | 7 | ✓ | ✓ | ✗ | ✗ | ✗ |
| BALROG | 6 | 48 | 12 | ✗ | ✗ | ✓ | ✓ | ✓ |
| NitroGen | 10 | 30 | 1 | ✓ | ✗ | ✗ | ✗ | ✓ |
| Orak | 12 | 12 | 15 | ✗ | ✗ | ✓ | ✗ | ✓ |
| GameVerse | 15 | 15 | 7 | ✓ | ✗ | ✗ | ✗ | ✗ |
| **GameWorld** | **34** | **170** | **18** | ✓ | ✓ | ✓ | ✓ | ✓ |

GameWorld 是唯一一个**全 ✓**的——34 games, 170 tasks, 18 model-interface pairs, vision-centric, configurable init state, task-oriented, parallel instances, state-verifiable eval。

特别值得注意的几个 referenced work：

- **OSWorld**（https://arxiv.org/abs/2404.07972）：computer-use benchmark 范式，强调 outcome-based evaluation，GameWorld 把这个理念 transfer 到 game domain
- **BALROG**（ICLR 2025, https://arxiv.org/abs/2411.04405）：long-horizon play in classic games，但发现 visual input 反而 degrade performance——这暗示了 perception bottleneck
- **VideoGameBench**（https://arxiv.org/abs/2505.08425）：23 titles，引入 paused track 隔离 latency——GameWorld 借鉴了这个 idea
- **Voyager**（https://arxiv.org/abs/2305.16291）：Minecraft lifelong skill acquisition，是 training-side 而非 evaluation-side 工作
- **JARVIS-1**（https://arxiv.org/abs/2311.05997）：multimodal memory for long-horizon Minecraft
- **MineDojo**（https://arxiv.org/abs/2106.01554, NeurIPS 2022）：internet-scale knowledge for Minecraft training
- **VPT (Video PreTraining)**（NeurIPS 2022）：从 unlabeled gameplay video 学 behavioral priors
- **Steve-1**（NeurIPS 2023）：text-conditioned behavior generation in Minecraft
- **SIMA / SIMA 2**（https://arxiv.org/abs/2410.0687）：DeepMind 通用游戏 agent
- **GameVerse**（concurrent work）：dual action space 类似 GameWorld 的 CUA+Generalist，但用 VLM-as-judge 量化 progress，存在 evaluation noise 问题
- **MCU**（https://arxiv.org/abs/2412.17587）：Minecraft evaluation framework
- **Cradle**（https://arxiv.org/abs/2403.03186）：general computer control，包含 complex games

---

## 16. 关键 Insights 和 Limitations 总结

### 16.1 三个核心发现

1. **Partial progress is achievable, reliable completion is not**：当前 MLLM agent 在多数 task 上能做出 meaningful partial progress（PG ~40%），但 reliable completion（SR ~20%）远未达到 human level（Novice SR 55.3%）。这意味着 SR 单一指标 misleading，PG 是必要的补充。

2. **Capability profile 继承自 backbone foundation model**：MLLM 在 symbolic reasoning（Level-4）和 reactive control（Level-2）上强，但在 basic timing grounding（Level-1）和 open-world coordination（Level-5）上 weak。这指向一个未来方向：要 fix Level-1 和 Level-5，需要 foundation model 本身的改进，不是单纯 harness 工程能解决的。

3. **Interface-aware trade-offs**：CUA 和 Generalist 在 capability bottleneck 上相似（都 peak 在 Level-2/4），但在 memory sensitivity（CUA 受损、Generalist 受益）和 instruction-following reliability 上有 distinct trade-off。这意味着 interface 选择应该是 task-dependent 的，没有 silver bullet。

### 16.2 Limitations

论文 Section 7 自己列出的 limitation：

- 每个 new environment 都要 unique instruction set，tightly couples action space to task，constrains scalability
- Semantic Action Parsing 的 producing 和 alignment 过程还是 manual 的——未来工作是用 MLLM-powered agent exploration 自动化

我觉得还有一个 implicit limitation：**233 个 state fields 是 manual designed 的**，每加一个新 game 都要 manual instrument。这对 benchmark 的可扩展性是真实约束。一个潜在的 future direction 是用 RL-based game state miner 自动 discover task-relevant fields。

---

## 17. 对未来研究的 Implications

从 Karpathy 的视角看，GameWorld 的真正价值在于它把"agent 能力分解"这件事变得 **measurable and reproducible**：

1. **作为 RL post-training 的 reward signal**：outcome-based state-verifiable evaluation 直接可用作 RL 的 reward，无需经过 perception。这对 VLA（vision-language-action）model 的训练尤其有价值。
2. **作为 model capability 的 diagnostic instrument**：5-level curriculum 可以作为未来 model release 的 standard eval——类似 MMLU 之于 LLM。Level-1 的失败尤其值得警觉——它说明当前 model 的 action grounding 仍然 weak。
3. **作为 game agent harness design 的指南**：context-memory 的 asymmetric 效应提示我们应该为不同 interface 设计不同 memory 策略；IAR 的存在提示我们 instruction-following 是 harness-level 而非 model-level 问题。

项目主页：https://gameworld-bench.github.io

希望这个深度解读能给你 build 起对这篇 paper 的 intuition。如果需要进一步 dig into 任何 module（比如 Semantic Action Parsing 的 YAML registry 实现、gameAPI schema 的具体 instrument 方式、或者某个具体 case study 的 frame-by-frame 分析），告诉我。
