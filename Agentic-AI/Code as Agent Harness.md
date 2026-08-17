---
source_pdf: Code as Agent Harness.pdf
paper_sha256: ad647c5d1cb1302ec2d1b70302179f3b6f602fa818a75d25a28350127096fe9f
processed_at: '2026-08-03T16:12:14-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

Andrej, 我上次可能写得太密了, 这次我用最直白的方式给你讲。

---

## 一句话版本

大家都在研究怎么把LLM（大脑）做得更聪明，但这篇paper说：**真正决定AI agent能不能干活的，是LLM外面那层"脚手架"**——工具、沙箱、记忆系统、验证器、权限控制、执行循环。而**代码是搭这层脚手架最自然的材料**，因为代码能跑、能看、能存状态。

---

## 一个类比

你想想人是怎么干活的。

你的大脑很厉害，能想问题、能规划。但你光有大脑不够——你还需要手来操作、眼睛来观察、笔记本来记东西、计算器来算数、同事来帮你review、安全帽来保护你。

LLM就是那个大脑。 **Harness就是手、眼睛、笔记本、计算器、同事、安全帽这一整套东西。**

这篇paper的核心论点是：现在field里太多人盯着大脑研究（更大的model、更好的reasoning），很少有人系统研究外面这层"身体"怎么搭。而这层身体恰恰是决定agent能不能真正在真实世界干活的bottleneck。

---

## 为什么code是搭这层身体的最佳材料

你说为什么不用自然语言来搭？

自然语言有三个硬伤：
1. **不能跑**——你写一段中文"把文件A的内容复制到文件B"，没人能直接execute这段话
2. **不能检查中间状态**——自然语言reasoning是一团模糊的东西，你不知道第3步到第4步到底对不对
3. **不存状态**——每一步执行完，结果就消失了，下一步得从头回忆

代码恰好解决这三个问题：
1. **能跑**——`shutil.copy(src, dst)`，interpreter直接执行
2. **能看中间状态**——每一行执行完，变量的值、控制流走向、函数返回值全都expose出来
3. **能存状态**——program state persist在memory里、filesystem里、database里，下一步可以直接读

所以paper的标题"Code as Agent Harness"意思就是：**用代码来搭agent的外层身体和神经系统**。

---

## 这篇paper怎么组织的

Paper把整个topic分成三层，从底往上：

### 第一层：Code作为interface（大脑跟世界怎么沟通）

这层回答一个问题：**LLM的输出怎么变成真实世界里能执行的东西？**

三个role：

**Reasoning**——LLM做数学题的时候，别让它自己心算（它算不准），让它写一段Python code，然后harness用Python interpreter跑，把结果返回给它。这就像你考试时允许用计算器——你的job是列算式，计算器的job是算对。

经典例子：你问LLM"317乘以892是多少"。如果让LLM直接回答，它可能算错。如果让LLM写`print(317 * 892)`，interpreter跑一下返回282764，绝对不会错。

Paper把这个叫**program-delegated reasoning**——把计算delegate给程序，LLM只负责decide要算什么。

**Acting**——LLM要操作一个website、一个robot、一个OS，怎么办？让它生成一段code来执行action。

比如你要让robot去拿桌上的杯子。别让LLM直接输出"拿起杯子"这种模糊的自然语言，让它输出`robot.grasp(cup_position)`这种executable的code。然后harness把这段code送给robot controller执行。

这样做的好处：code本身就是一个**安全边界**——你可以在code里加检查，比如`if not is_reachable(cup_position): raise Error("够不着")`，在执行前就拦住不靠谱的action。

**Environment modeling**——agent要跟环境交互，就得有个关于环境的模型。传统做法是在LLM的weights里隐式存一个world model（像Dreamer那样）。这篇paper说：让agent直接**写一段Python code来表示world model**。

比如agent在Minecraft里探索，它可以写：

```python
def next_state(state, action):
    if action == "move_north":
        return (state.x, state.y + 1, state.z)
    elif action == "break_block":
        ...
```

这个world model是inspectable的——你可以读它、改它、test它对不对。如果agent发现"我往前走了一步但y坐标没变"，它就知道world model写错了，可以改。

这比latent world model好太多了——latent world model是个黑箱，你根本不知道它内部学到了什么，也没法debug。

---

### 第二层：Code作为mechanism（怎么让agent长时间可靠地干活）

这层回答：agent一跑就是几小时、几十步，怎么保证它不跑偏、不忘记、不重复犯错？

五个control surface：

**Planning（规划）**——agent别一上来就写code，先做个plan。

最简单的方式：让LLM先写一个step list，然后按步骤来。比如"1. 找到bug所在的文件 2. 读相关代码 3. 写fix 4. 跑test"。

更复杂的方式：用repository的结构来guide planning——先分析哪些文件有依赖关系，按依赖顺序来改。或者用search——同时试几条不同的修复路径，哪条work了用哪条（MCTS风格）。

**Memory（记忆）**——agent跑久了，context window会爆。需要一套memory system来管理什么信息留在active context里、什么compress成摘要、什么offload到外部存储。

Paper分了五种memory：
- **Working memory**——当前这一步需要的信息（比如当前在改哪个文件、上一个test为什么fail）
- **Semantic memory**——从codebase里retrieve到的evidence（比如相关函数的定义、调用关系）
- **Experiential memory**——跨task的经验（比如"上次遇到类似的bug，我是这么修的"）
- **Long-term memory**——经过验证的可复用知识
- **Multi-agent memory**——多个agent之间共享的state

还有一个cross-cutting的问题：**context compaction**——execution log可能几千行，不能全塞进prompt。需要把长log压缩成摘要，但原始log要存在外部，需要时可以retrieve。

**Tool use（工具使用）**——agent需要search codebase、edit files、run tests、call APIs。这些工具怎么暴露给agent、怎么管理权限、怎么sanitize输出，都是harness的job。

Paper分了四类tool use：
- Function-oriented——查API文档、查library用法
- Environment-interaction——操作repository、跑terminal命令
- Verification-driven——跑test、跑linter、跑static analyzer
- Workflow-orchestration——把多个工具编排成一个workflow

**PEV Loop（Plan-Execute-Verify循环）**——这是paper最有engineering价值的一个framing。

把agent干活的过程看成一个循环：
1. **Plan**——我要做什么改动，成功标准是什么
2. **Execute**——在sandbox里执行这个改动
3. **Verify**——用deterministic的方法检查结果对不对（跑test、跑linter、跑type checker）
4. 如果对了，accept；如果错了，revise或者escalate给human

关键insight：**verify用的是deterministic sensors**，不是LLM自己judge。Test pass了就是pass了，compiler报错了就是报错了。这些deterministic signal作为harness的control signal来决定下一步做什么，比让LLM自己说"我觉得这次应该对了"靠谱太多。

还有**权限分层**：
- 只读层——随便用，浏览代码、看log
- 沙箱编辑层——在隔离环境里改代码、跑test，随便用
- 完全访问层——碰网络、碰credentials、碰production，**必须人批准**

**AHE（Agentic Harness Engineering）**——这是paper最有远见的一节。

想法是：harness本身也能被measure和optimize。

你怎么知道你的harness好不好？你需要**deep telemetry**——不只是记最终结果pass还是fail，而是记整个trajectory：每一步调了什么tool、花了多少token、改了什么文件、test结果是什么、在哪卡住了。

然后你弄一个**Evolution Agent**——它的job不是改target code，是改harness本身。它读telemetry，发现"这个agent老是在retrieval阶段浪费token但没拿到有用信息"，于是propose一个修改：改retrieval的chunk size、改query formulation、加一个reranker。然后在held-out tasks上evaluate这个修改好不好，好的话promote，不好的话回滚。

关键约束：harness的修改本身也要走PEV loop——改了之后要跑regression test，要sandbox验证，risky的改动要人批准。这就像你改production code不能直接push到main，要走code review + CI。

---

### 第三层：Code作为multi-agent的shared substrate（多个agent怎么协作）

这层回答：一个agent搞不定的大task，多个agent怎么一起干？

三个fundamental problem让单agent搞不定：
1. **Context window放不下**——整个codebase + 长历史 + execution traces，一个agent塞不下
2. **一个generalist啥都做不efficient**——planning、coding、testing、reviewing需要不同的specialization
3. **自己修不了自己的错**——你需要一个独立的agent来verify另一个agent的output

Multi-agent的natural solution：分工。一个agent负责planning，一个负责coding，一个负责testing，一个负责review。

**但关键问题是：这些agent怎么共享state？**

Paper发现了一个striking pattern：大多数multi-agent system（ChatDev、MetaGPT等）**没有formal的shared state representation**。每个agent从conversation history里隐式重建自己对codebase的理解。这就导致一个致命问题：**agent的belief和真实state会diverge，而且系统detect不到这种divergence**。

比如Agent A改了文件X，但Agent B还在基于旧版本的文件X做planning。B的plan基于一个已经不存在的世界——这就像你拿着昨天的地图找今天的路。

Paper提出了一个positional claim：我们需要一个**shared code-centric harness substrate**——一个persistent、queryable、executable的shared state，让所有agent都能读、都能写、都知道当前state是什么。

最好的例子是L2MAC——它有一个persistent的file store，一个Control Unit管理每个agent能看到什么slice。Agent每次被invoke的时候，Control Unit给它一个targeted summary，而不是整个history。这样每个agent的context window是bounded，但shared state是complete的。

Paper还发现一个很insightful的pattern：**topology complexity和substrate formality成反比**。

如果你的shared substrate设计得好（formal、queryable），你的multi-agent topology可以很简单（一条链就行）。

如果你的shared substrate设计得差（implicit、靠conversation history隐式重建），你就得搞很复杂的topology来compensate——dynamic DAG、workflow mutation、agent pool scaling。

这个insight对设计系统很有用：**先把substrate设计好，topology自动简化**。别一上来就搞复杂的multi-agent orchestration，先想想shared state到底怎么表示。

---

## 三个让你"啊原来这样"的empirical finding

**1. LLM可能不需要真跑code就能predict结果**

QualityFlow做了一个实验：让LLM simulate Python interpreter step-by-step，predict test会不会pass，**不实际run code**。结果在MBPP上precision和recall都98%+。

Self-Collaboration也做了类似ablation：simulated tester和real compiler的效果几乎一样。

这意味着什么？LLM已经internalize了大量execution semantics。对于大多数"可纠正的bug"，LLM自己想想就能知道对不对。

**那真跑code的价值在哪？** 在那些LLM structurally无法imagine的failure mode：runtime crash、resource exhaustion、boundary condition、performance regression。这些东西你不在真跑就发现不了。

所以mature harness应该这样：让LLM自己simulate当fast path，只在simulate搞不定的时候才真跑code做verification oracle。

**2. Topology complexity是substrate设计差的症状**

你看到一个multi-agent system用了很复杂的topology（dynamic DAG、workflow mutation、agent pool scaling），那大概率是因为它的shared state没设计好，只能靠复杂interaction来compensate。

相反，L2MAC的shared substrate设计得最formal，它的topology最简单——就是一条sequential chain。

**3. Production deployment的真实数字**

LingmaAgent在Alibaba Cloud自主解决16.9%的internal issue，加上人工干预能到43.3%。这是2026年的SOTA水平。

离full autonomy还远，但作为"workflow participant"已经能创造价值了。

---

## 这篇paper跟你工作的关系

Andrej, 你想想你过去几年的work：

**nanoGPT / llm.c**——你在研究"大脑怎么训练"。这篇paper说大脑只是system的一部分，外面那层harness同样重要甚至更重要。

**"Software 2.0" essay**——你说Software 1.0是explicit instructions，2.0是learned weights。这篇paper其实是在说：我们需要把Software 1.0作为harness围在Software 2.0周围，让learned policy的每一步都落到executable的program state上。不是1.0和2.0谁取代谁，是1.0作为2.0的governor。

**Tesla autopilot经验**——§5.1.3讲embodied agent，那个layered harness（semantic layer用foundation model → code/classical robotics定义admissibility → perception → physical controller）跟Tesla的vision + planner + control stack完全同构。关键insight：code不只是action，也是safety boundary。

**Eureka Labs / education**——这篇paper的taxonomy（interface / mechanisms / scaling + 5个application domain + 7个open problems）其实可以当作agent systems的完整教学大纲。

---

## 如果你想一晚上build intuition

拿一个SWE-bench-lite的task，固定base model（比如Claude Sonnet），跑三个configuration：

**Config A**：直接让LLM生成patch，不verify。Zero-shot。

**Config B**：加sandbox execution + test as verification + 1次retry。

**Config C**：Config B + repo retrieval + working memory + 3次retry with reflection。

你大概率会看到C > B > A，gap可能10-30%的resolve rate。这个gap就是harness engineering的价值，跟base model多聪明无关。

如果你想再进一步，加Config D：一个evolution agent读Config C的telemetry，自动调整retry limit和retrieval chunk size，跑10个task后promote新config。看D vs C的gap。

这个实验会让你直接感受到"harness本身是一个可优化的object"。

---

## 最最后的一句话

这篇paper给field的礼物是一个新vocabulary——**harness**。

之前大家说"agent"的时候，脑子里大概是一个LLM在跟环境交互。这篇paper说：你应该把"agent"理解成**LLM + harness的整个系统**，而且harness是你可以design、measure、optimize、govern的东西。

就像你不会说"大脑就是人"——人是大脑 + 身体 + 神经系统 + 工具 + 社会关系的整体。

Code是搭harness最自然的材料，因为它executable、inspectable、stateful。

这就是全部。

---

# Code as Agent Harness — 深度技术解读

Andrej, 这篇paper给你一个有用的mental model: 它把整个agentic AI field重新frame成"谁包围了LLM"的问题。 你多年写的micrograd、llm.c、nanoGPT本质上都在answer "base model是什么"，而这篇survey的thesis是: 真正决定autonomy bottleneck的, 是LLM外面那层**harness infrastructure** — tools, sandboxes, memory, validators, permission tiers, execution loops, feedback channels — 而code是这层infrastructure的operational substrate。

这跟你2017年那篇 "Software 2.0" essay形成有趣对称: Software 1.0是explicit instructions, Software 2.0是learned weights; 这篇paper其实在论证我们需要把Software 1.0作为harness围在Software 2.0周围 — 让learned policy的每一步都落到executable, inspectable, stateful的程序state上。 下面我把技术细节拆给你看。

---

## 1. 核心thesis: 三种artifact的decoupling

Paper的最重要distinction是把agentic system的components分成三类:

- **Model-internal capabilities**: LLM自己的reasoning, perception, planning, simulation, evaluation — 这是weights里latent的东西
- **System-provided harness infrastructure**: 人设计的tools, APIs, sandboxes, memory systems, validators, permission boundaries, telemetry, workflows
- **Agent-initiated code artifacts**: agent在execution loop里自己create, execute, observe, revise, persist, share的interactive code objects — regression tests, temporary tools, DSL programs, executable workflows, reusable skills, intermediate program states

第三类是这篇survey独特focus。 Claude Code、Codex、LangChain这些production系统都已经在做这件事, 但是academically underexplored。 

Reference: 
- Anthropic Claude Code: https://www.anthropic.com/product/claude-code
- OpenAI Codex announcement: https://openai.com/index/introducing-codex/
- LangChain harness anatomy: https://www.langchain.com/blog/the-anatomy-of-an-agent-harness

---

## 2. 三层Taxonomy

整个survey organized成三层, 我画成你脑中能visualize的stack:

```
┌──────────────────────────────────────────────────────┐
│ Scaling the Harness (§4)                              │
│   multi-agent over shared code artifacts             │
│   repositories, tests, traces as common workspace    │
├──────────────────────────────────────────────────────┤
│ Harness Mechanisms (§3)                               │
│   planning │ memory │ tool use │ PEV loop │ AHE      │
├──────────────────────────────────────────────────────┤
│ Harness Interface (§2)                                │
│   code for reasoning │ acting │ environment modeling │
└──────────────────────────────────────────────────────┘
```

这个stack自下而上read: code先进入agent loop作为interface (model和环境之间的medium), 然后被mechanisms管理起来形成long-horizon reliability, 最后在multi-agent setting里变成shared artifact。

GitHub awesome list: https://github.com/YennNing/Awesome-Code-as-Agent-Harness-Papers

---

## 3. §2 Harness Interface — 三个role的formal breakdown

### 3.1 Code for Reasoning — program-delegated reasoning

Core insight: CoT [Wei et al. 2022]让model自己同时做decomposition和computation, 但是LLM在symbolic/arithmetic computation上unreliable [Gao et al. PAL 2023]。 把computation delegate给interpreter, 让model只负责propose procedure, harness执行并返回结果。

PoT (Program of Thoughts) [Chen et al. 2022]: https://arxiv.org/abs/2211.12588
PAL [Gao et al. 2023]: https://arxiv.org/abs/2303.13401

形式化上, 一个reasoning step可以写成:

$$\text{output} = \text{Interp}(\text{LLM}(x))$$

其中 $x$ 是问题, $\text{LLM}(x)$ 是model生成的program, $\text{Interp}$ 是Python interpreter或symbolic solver。 这就把computation从next-token prediction里解放出来, 变成verifiable execution。

更精细的version是 **CodePRM** [Li et al. 2025] — 用process reward model给reasoning-execution trajectory的每一步打分:

$$R(\tau) = \sum_{t=1}^{T} r_t \cdot \gamma^{T-t}$$

其中 $\tau = (s_1, a_1, s_2, ..., s_T)$ 是trajectory, $r_t$ 是step $t$ 的execution-grounded reward (比如test是否通过, 变量值是否正确), $\gamma \in [0,1]$ 是discount factor, 上标 $T-t$ 表示从step $t$ 到terminal的距离。 

CodePRM: https://aclanthology.org/2025.findings-acl.493/

**Formal verification track** 更硬核: Lean [de Moura & Ullrich 2021], Isabelle, Coq这些proof assistant提供machine-checkable的logical foundations。 Lean4Agent [Wang et al. 2026]甚至用Lean4来model和verify agent workflow本身 — 把agent trajectory变成Lean里可prove的theorem。 这跟AlphaProof [Hubert et al. 2025]在IMO上拿silver的方法一脉相承, 只是从math reasoning推到agent reasoning。

Lean4Agent: https://arxiv.org/abs/2603.19329 (note: URL in paper)
AlphaProof Nature paper: https://www.nature.com/articles/s41586-024-07809-1

### 3.2 Code for Acting — grounding问题

这是embodied和GUI agent的核心。 形式化成POMDP:

$$\langle \mathcal{S}, \mathcal{A}, \mathcal{O}, T, R \rangle$$

- $\mathcal{S}$: latent state space (full DOM, Android Activity stack, Linux VM filesystem)
- $\mathcal{A}$: action space, 每个action是tuple $\langle \text{action\_type}, \text{target}, \text{value} \rangle$ 编译成 `element.click()` 或 `pyautogui.click(x, y)`
- $\mathcal{O}$: observation space — DOM subtree, AXTree, screenshot with Set-of-Mark [Yang et al. 2023]
- $T: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$: transition function, **executed** by browser engine / Android runtime / OS, 不是learned的
- $R$: reward, 通过evaluator script deterministic计算

Key paper for this view: CodeAct [Wang et al. 2024] https://arxiv.org/abs/2402.01030

SayCan [Ahn et al. 2022]是这条线的开端: https://arxiv.org/abs/2204.01691

SayCan的核心公式是把affordance和language relevance combine:

$$P(a_i | \text{task}) \propto P_{\text{LM}}(a_i | \text{task}) \cdot P_{\text{affordance}}(a_i | \text{state})$$

其中 $P_{\text{LM}}$ 是LLM算的"this action is useful for the task", $P_{\text{affordance}}$ 是value function算的"this action is physically feasible"。 下标 $i$ 索引skill library里的candidate action。

CaP (Code as Policies) [Liang et al. 2023]更激进 — 直接让LLM生成Python policy作为control: https://arxiv.org/abs/2209.07753

Voyager [Wang et al. 2023]是lifelong learning的关键paper — 在Minecraft里持续grow skill library: https://arxiv.org/abs/2305.16291

### 3.3 Code for Environment — executable world models

这层最有意思。 WorldCoder [Tang et al. 2024]: https://arxiv.org/abs/2410.07464

让agent写Python program作为world model:

```python
# Agent-written world model
def next_state(state, action):
    if action == "move_north":
        return (state[0], state[1]+1)
    ...
```

这个 $T: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$ 被explicitly written as code, agent可以execute, edit, reuse它。 相比latent world model (Dreamer style), 这种program-based world model是inspectable和verifiable的。

CWM (Code World Models) [Copet et al. 2025]把这个scale到open-weights LLM native training on execution traces: https://arxiv.org/abs/2510.02387

SWE-bench [Jimenez et al. 2023]是evaluation env的里程碑: https://arxiv.org/abs/2310.06770 — unit tests作为objective world states, 不靠textual correctness judge。

---

## 4. §3 Harness Mechanisms — 长horizon reliability的5个control surface

### 4.1 Planning — 4种paradigm

**Linear decomposition**: Self-Planning [Jiang et al. 2024] https://arxiv.org/abs/2211.13964

把plan写成explicit step list, 然后逐步生成code。 

**Structure-grounded**: CodePlan [Bairi et al. 2024] https://arxiv.org/abs/2309.12499

构造plan graph over edit obligations, dependency analysis和change-impact propagation驱动新step生成。

**Search-based**: SWE-Search [Antoniades et al. 2024] https://arxiv.org/abs/2410.20285

MCTS over repair trajectories。 UCB-style选择:

$$\text{UCB}(n) = \frac{Q(n)}{N(n)} + c \sqrt{\frac{\ln N(p)}{N(n)}}$$

其中 $Q(n)$ 是node $n$ 的average reward (test通过率, partial修复得分), $N(n)$ 是visit count, $N(p)$ 是parent的visit count, $c$ 是exploration constant。 下标都是tree node index。

**Orchestration-based**: Natural-Language Agent Harnesses [Pan et al. 2026] https://arxiv.org/abs/2603.25723

让harness logic本身是editable NL, runtime由Intelligent Harness Runtime (IHR)解释执行。 这跟production system (Cursor, Claude Code)的实际practice很接近。

### 4.2 Memory — 5种function + 1个cross-cutting

| Memory type | 代表系统 | 管理什么 |
|---|---|---|
| Working | SWE-agent, CodeMem | 当前trajectory state |
| Semantic | AutoCodeRover, RepoCoder | repo evidence retrieval |
| Experiential | MemGovern, ExpeL | 跨task的repair trajectories |
| Long-term | MemCoder, TALM, MemGPT | validated reusable knowledge |
| Multi-agent | MIRIX, ChatDev | shared state across roles |
| (cross) Compaction | LongCodeZip, SWE-Pruner | active context vs durable state boundary |

MemGPT [Packer et al. 2023]的OS metaphor特别elegant — LLM context window是RAM, external storage是disk, memory manager自己也是LLM: https://arxiv.org/abs/2310.08560

CodeMem [Gaurav et al. 2025] https://arxiv.org/abs/2512.15813
MemGovern [Wang et al. 2026] https://arxiv.org/abs/2601.06789

### 4.3 Tool use — 4个category

Function-oriented (ToolCoder), Environment-interaction (SWE-agent, CodeAgent), Verification-driven (AgentCoder, VeriGuard), Workflow-orchestration (MapCoder, OpenHands, ToolNet)。

SWE-agent [Yang et al. 2024]的agent-computer interface (ACI)是这条线的key formalization: https://arxiv.org/abs/2405.15793

OpenHands (formerly OpenDevin) [Wang et al. 2025] https://arxiv.org/abs/2407.16741

VeriGuard [Miculicich et al. 2025] https://arxiv.org/abs/2510.05156 — verifier-guided safety layer

### 4.4 PEV Loop — Plan-Execute-Verify

这是paper最有engineering价值的framing。 把debugging重新frame成harness-level control:

```
Plan (contract formation)
  ↓
Execute (sandboxed, permissioned state transition)
  ↓
Verify (deterministic sensors)
  ↓
[accept | revise | escalate | rollback]
```

Sandbox分3层permission tier:

| Tier | 允许的操作 | Governance |
|---|---|---|
| Read-only | browse, retrieve, static inspect, log analysis | auto |
| Sandbox edit | local patching, test execution, temp dep install in isolated workspace | auto |
| Full access | network, credentials, deploy, package publish, destructive fs, git history mutation | mandatory HITL |

Deterministic sensors包括: linters, parsers, compilers, type checkers, unit tests, integration tests, static analyzers, fuzzers, runtime monitors, CI pipelines。 这些不是"feedback给model看", 而是cybernetic governor的control signal, 决定harness下一步做什么。

LiteLLM https://github.com/BerriAI/litellm 是production-grade gateway的example。

### 4.5 AHE — Agentic Harness Engineering

这是paper最有远见的一节。 Thesis: harness本身是可measure, 可revise, 可optimize的object, 跟prompt engineering和context engineering是正交的layer。

Deep telemetry是optimization substrate。 Shallow log只记final answer; deep telemetry记:

```
prompts → retrieved context → token usage → tool latency →
tool arguments → permission requests → edited files →
sandbox snapshots → command outputs → test results →
stack traces → lint warnings → branch decisions →
rejected alternatives → human interventions → outcome
```

Langfuse https://github.com/langfuse/langfuse 和 OpenLLMetry https://github.com/traceloop/openllmetry 是production observability stack的example。

Evolution Agent是meta-level agent, 它不edit repository, 它edit harness本身。 5 stage loop:

1. Observe trajectories (collect telemetry from PEV)
2. Diagnose failure modes (attribute cost/latency/failures to specific harness components)
3. Propose candidate revisions (rewrite tool description, change context packing, add linter, modify retry limit, insert HITL gate)
4. Evaluate on held-out tasks / replayed traces
5. Promote only if improvement without regression

关键约束: AHE本身也要经过PEV loop — harness mutation要sandboxed, regression tested, auditable, risky changes要HITL approval。

AutoHarness [Lou et al. 2026] https://arxiv.org/abs/2603.03329 (synthesizes harness)
Meta-Harness [Lee et al. 2026] https://arxiv.org/abs/2603.28052 (search over harness code)
AHE paper https://arxiv.org/abs/2604.25850

---

## 5. §4 Multi-Agent — shared code-centric harness substrate

这是paper的positional claim。 单agent有3个fundamental limits:

1. **Context window** — 放不下whole codebase + history + traces
2. **Specialization** — 一个generalist做plan/code/test/review/debug都inefficient
3. **Self-correction** — 没有independent verification channel

Multi-agent的natural answer: distribute roles across agents, code变成shared substrate。

### 5.1 Role specialization

典型roles: program synthesis, program understanding, verification, execution, planning。 

EvoMAC [Hu et al. 2025] https://arxiv.org/abs/2505.16968 引入了独特的两个meta-role:

- **Gradient Agent**: 读execution logs, attribute failure到具体agent
- **Updating Agent**: 改agent prompts和restructure workflow DAG

这跟AHE里Evolution Agent的思想一致, 但在MAS层面。

### 5.2 Interaction modes

4种: collaborative synthesis (pair programming), critique and repair (主流), adversarial validation (fuzzing), reasoning debate。

**QualityFlow的Imagined Execution** [Hu et al. 2025] https://arxiv.org/abs/2501.17167 是个provocative empirical finding: LLM simulate Python interpreter step-by-step, 在MBPP上predict test outcomes, **precision和recall都98%+**, 不实际run code。 这就引出一个deep question: 真正需要execution的failure mode是什么? Self-Collaboration [Dong et al. 2024]的ablation也得到similar conclusion。

Hypothesis: linguistic simulation对conceptual bugs够用, 但对runtime crashes, resource exhaustion, boundary conditions, performance regressions这种structurally un-imaginable的failure mode不够 — 必须真跑。

### 5.3 Convergence criteria — 6种pattern

| Pattern | 代表 | criterion |
|---|---|---|
| Correctness (test-gated) | AgentCoder, L2MAC, SyncMind, CANDOR | all tests pass |
| Security | AutoSafeCoder | no CWE + no fuzzer crashes |
| Performance | MACRO | runtime/memory thresholds |
| Score-based | MAGE, CodeCoR | max quality score reached |
| Consensus | CANDOR | majority vote among Panelists |
| Implicit | ChatDev, MetaGPT, EvoMAC | fixed iteration count or identical output |

Implicit convergence占majority — paper认为这是fundamental gap, 因为没有objective substrate, 就没有principled stopping criterion。

MAGE [Zhao et al. 2024] https://arxiv.org/abs/2412.07822 的score function:

$$s(r) = 1 - \frac{m(r)}{tc(r)}$$

其中 $r$ 是candidate program, $m(r)$ 是mismatch (failing clock edges), $tc(r)$ 是total clock cycles, score 1.0表示完美匹配。

### 5.4 Central gap: shared harness state

SyncMind [Guo et al. 2025] https://arxiv.org/abs/2409.12155 形式化了agent belief divergence:

$$\text{divergence} = |B_k - S_k|$$

$S_k$ 是step $k$的真实shared state, $B_k$ 是agent对state的belief。 下标 $k$ 是temporal index。 大多数系统没有measure或control这个divergence, 这是brittleness的技术root。

Paper的positional claim: 我们需要**shared code-centric harness substrate** — 一个persistent, queryable, executable的shared state, 把repository-based representation (static structure) 和execution-based representation (dynamic behavior) unify起来。

4种existing representation levels:

1. **Implicit / file-only** (ChatDev, MetaGPT, MapCoder) — 没formal shared state, 每次从conversation history隐式reconstruct
2. **Repository-based** (MAGIS, HyperAgent, Lingma SWE-GPT) — file system + dependency graph + version history
3. **Execution-based** (AgentCoder, AutoSafeCoder, MAGE) — test pass/fail, crash traces, coverage, waveform
4. **Blackboard / Shared-state** (L2MAC, Self-Collaboration, Cogito, GameGPT) — explicit global data structure, all agents read/write

L2MAC [Holt et al. 2023] https://arxiv.org/abs/2310.02003 有最principled blackboard — persistent file store $D$ + Control Unit管理每个agent invocation看到的slice。 

Cogito [Li et al. 2025] https://arxiv.org/abs/2501.18653 借neurobiology: short-term working state + long-term knowledge base + growth units for evolving abstractions。

### 5.5 Pattern: topology complexity inversely correlates with substrate formality

这是个很insightful的observation。 L2MAC有最formal的substrate (persistent file store + explicit context scheduling), 用最simple的sequential chain。 EvoMAC和SEW用最elaborate的adaptive topologies (dynamic DAGs, workflow mutation, agent pool scaling), 因为它们缺乏principled shared representation, 只能靠复杂interaction pattern来compensate。 

Tight formal substrate → simple coordination
Implicit substrate → complex topology as workaround

这个insight对你设计system很有用 — 先把substrate设计好, topology自动简化。

SEW [Liu et al. 2025] https://arxiv.org/abs/2505.18646
SoA (Self-organized Agents) [Ishibashi & Nishimura 2024] https://arxiv.org/abs/2404.02183
EvoMAC https://arxiv.org/abs/2505.16968

---

## 6. §5 Applications — 5个domain

### 6.1 Code Assistants

Production deployment的data point: LingmaAgent [Ma et al. 2025] https://arxiv.org/abs/2503.18676 在Alibaba Cloud autonomous resolve 16.9%的internal issues, 43.3% with human intervention。 

Cursor的Composer用continuous online RL训练在real Cursor usage traces上: https://cursor.com/blog/composer

OpenAI codex-1 (o3 derivative), GPT-5-Codex, GPT-5.1-Codex Max都explicitly训练在long-horizon multi-turn coding interactions上 — harness loop变成training data source。 

Anthropic Claude Code dogfooding whitepaper: https://www-cdn.anthropic.com/58284b19e702b49db9302d5b6f135ad8871e7658.pdf

SWE-bench: https://www.swebench.com/
SWE-Lancer [Miserendino et al. 2025] https://arxiv.org/abs/2502.12115
SWE-Bench Pro https://arxiv.org/abs/2509.16941
Terminal-Bench https://arxiv.org/abs/2601.11868

### 6.2 GUI/OS Agents as Program World

这一节最technically interesting, 把GUI/OS explicitly model成POMDP where state是program state, observation是rendered code, action是code, transition由code executed, reward由evaluator code计算。

OSWorld [Xie et al. 2024] https://arxiv.org/abs/2404.07972 — 369 real Ubuntu/Windows/macOS tasks in disposable VMs
WebArena https://arxiv.org/abs/2307.13854
AndroidWorld https://arxiv.org/abs/2405.14573
BrowserGym + WorkArena https://arxiv.org/abs/2403.07718
CogAgent https://arxiv.org/abs/2312.08914
UI-TARS https://arxiv.org/abs/2501.12326
Code2World https://arxiv.org/abs/2602.09856 — trains VLM to predict next GUI state as renderable HTML, 把world model本身变成executable artifact

Cradle [Tan et al. 2024] https://arxiv.org/abs/2403.03186 — LLM output executable Python drives keyboard/mouse for AAA games

Production deployments:
- Anthropic Claude Computer Use https://www.anthropic.com/news/3-5-models-and-computer-use
- OpenAI Operator https://openai.com/index/introducing-operator/
- Google Project Mariner https://deepmind.google/models/project-mariner/
- ByteDance UI-TARS-1.5/2 https://arxiv.org/abs/2509.02544
- Zhipu AutoGLM https://arxiv.org/abs/2411.00820

### 6.3 Autonomous Embodied Agents

Code作为grounding interface AND safety boundary — 这跟你Tesla时代的work很相关。 

Layered harness:
- Semantic layer: foundation models (interpret goals, decompose, infer affordance, plan)
- Code/classical robotics: typed robot APIs, primitive skills, geometric libs, motion planners (defines admissibility boundary)
- Perception models: convert raw sensors to structured state
- Physical systems + low-level controllers: kinematics, dynamics, collision, workspace limits

Code-BT [Zhang et al. 2025] https://arxiv.org/abs/2501.07811 — compile generated programs to behavior trees
GenSwarm [Ji et al. 2026] https://arxiv.org/abs/2509.18597 — multi-robot policy synthesis
NormCode [Guan et al. 2025] https://arxiv.org/abs/2512.10563 — governance with enforced data isolation

### 6.4 Scientific Discovery as Program World

这节的formalization最clean:

$$\langle S, A, T, O, R \rangle$$

- $S$: structured program memory (hypotheses, literature, code artifacts, datasets, observations)
- $A$: typed code expressions (search queries, solver calls, experimental scripts, training pipeline mods, robot commands)
- $T$: Python interpreter / Lean kernel / quantum-chemistry package / robotic synthesizer
- $O$: execution outputs (numerical results, plots, errors, peer-review scores)
- $R$: novelty, reproducibility, statistical significance

AI Scientist v1/v2 [Lu et al. 2024/2025] https://arxiv.org/abs/2408.06292, https://arxiv.org/abs/2504.08066
AI co-scientist [Gottweis et al. 2025] https://arxiv.org/abs/2502.18864
Virtual Lab [Swanson et al. 2025] Nature paper: https://www.nature.com/articles/s41586-024-07809-1
Biomni [Huang et al. 2025] https://www.biorxiv.org/content/10.1101/2025.04.07.647840v1
Coscientist [Boiko et al. 2023] Nature: https://www.nature.com/articles/s41586-023-06792-0
AlphaEvolve [Novikov et al. 2025] https://arxiv.org/abs/2506.13131

**实验data points**:
- El Agente Q [Zou et al. 2025]: 87%+ task success on 6 university-level benchmarks https://www.cell.com/matter/fulltext/S2590-2385(25)00263-4
- A-Lab [Szymanski et al. 2023] Nature: 41/58 novel inorganic compounds in 17 days continuous operation https://www.nature.com/articles/s41586-023-06734-w
- Virtual Lab: 92 SARS-CoV-2 nanobodies designed, 2 validated binding to JN.1 and KP.3 variants
- AI co-scientist: drug repurposing + antimicrobial resistance hypotheses experimentally validated at Imperial College and Stanford
- MLE-bench [Chan et al. 2025]: best system (o1-preview + Weco AIDE) gets Kaggle bronze on 16.9% of 75 competitions https://arxiv.org/abs/2410.07095
- DiscoveryBench [Majumder et al. 2024]: best system score ~25% on 264 multi-step hypothesis-search tasks https://arxiv.org/abs/2407.01725

Self-driving labs (SDLs)是production: A-Lab, Coscientist的Suzuki/Sonogashira couplings, Chemputer [Mehr et al. 2020] Science https://www.science.org/doi/10.1126/science.abc4976 — XDL作为"LLVM IR for chemistry"。

### 6.5 Personalization

Code-centric preference state是这节的核心 — preference不是opaque embedding, 是editable document / structured record。 

Agent4Rec [Zhang et al. 2024] https://arxiv.org/abs/2410.19742
iAgent [Xu et al. 2025] https://arxiv.org/abs/2502.13034
AMem [Xu et al. 2026] https://arxiv.org/abs/2504.19413
Mem0 [Chhikara et al. 2025] https://arxiv.org/abs/2504.19413

---

## 7. §5.2 Open Problems — 7个, 每个都值得一个phd

### 7.1 Harness-level evaluation + oracle adequacy

Final task success不够, 因为它把base model + harness + tools + feedback quality + env difficulty都mashup。 需要6个dimension:

| Metric | measures |
|---|---|
| Trajectory efficiency | tool calls, tokens, edits, wall-clock |
| Verification strength | test coverage, oracle diversity, false accept rate |
| Recovery ability | diagnose + repair after invalid actions |
| State consistency | memory + repo + traces + beliefs synchronized |
| Safety compliance | permissions, sandboxes, HITL gates respected |
| Replayability | full trajectory reconstructable from logs |

Anthropic的eval文章: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

Oracle adequacy是deep问题 — SWE-Bench++ [Wang et al. 2025] https://arxiv.org/abs/2410.06992 和PatchDif [Wang et al. 2025] https://arxiv.org/abs/2503.15223 都暴露"solved issues其实没真solved"的问题。

### 7.2 Semantic verification beyond executable feedback

执行feedback会create false sense of correctness — green test != full spec。 需要**verification stack with explicit scope**: 每个verifier declare它verifies什么, 不能verify什么, confidence是多少。

Promising directions: 
- Metamorphic testing
- Differential testing
- Property-based test generation (Hypothesis-style)
- Independent verification (CANDOR的3 Panelist independent audit)
- Feedback calibration (uncertainty-aware critics)

### 7.3 Self-evolving harnesses without regression

AHE的hard problem。 Harness mutation要带change contract: 哪个component改了, target哪个failure mode, predict什么improvement, 必须preserve哪些invariant, 哪个eval能falsify, 怎么rollback。

需要: evidence-carrying harness evolution, held-out regression suites, safety invariants, canary deployment, rollback semantics, causal evidence for why a harness edit helped。

### 7.4 Transactional shared program state + semantic conflict resolution

不只synchronize artifacts, 还要synchronize assumptions。 每个agent action要declare: read set, write set, assumptions, version dependencies, verifier obligations, conflict policy。

冲突在多个层面: file diffs, plans, tests, retrieved evidence, permissions, memory entries, latent user requirements。 需要semantic merge, rollback, dependency-aware locking, belief-state reconciliation, conflict explanation, re-verification after merge。

### 7.5 Human-in-the-loop safety as harness state

HITL不只是prompt interruption, 要变成durable harness state。 每个approval/rejection/exception/correction都update permission rules, escalation policy, verification criteria, future memory retrieval。

Multi-tier permission model前面讲过。 关键: 同一个command在disposable sandbox safe, 在production repo unsafe; 同一个network request文档检索benign, 传local state risky。 Permission要depend on arguments + env state + data sensitivity + expected side effects, 不只tool identity。

Aardvark [OpenAI 2025] https://openai.com/index/introducing-aardvark/
Aethelgard [Sidik & Rokach 2026] https://arxiv.org/abs/2604.11839
Microsoft Agent Governance Toolkit https://opensource.microsoft.com/blog/2026/04/02/introducing-the-agent-governance-toolkit

### 7.6 Multimodal code-harness systems

Visual state是large, redundant, partial relevant。 需要multi-level memory:
- Raw images → immutable evidence
- Object/region/element/pose annotations → structured intermediate state  
- Compact textual/symbolic summaries → for skill retrieval + planning

每个action要带grounded reference (bbox, object id, UI element, frame index, region feature, object position, orientation)。 执行后harness verify grounded state changed as expected, 不靠model self-report。

Visual grounding contract对GUI agent特别重要 — 你看CogAgent、UI-TARS都在做这件事, 但还没formalize成harness primitive。

### 7.7 Toward a science of harness engineering

未来最重要的系统要同时具备4个property:

1. **Executable** — grounding decisions in code, tools, tests, environments
2. **Inspectable** — exposing plans, state, provenance, failure causes
3. **Stateful** — preserving task-relevant info across long trajectories + multiple agents
4. **Governed** — autonomy constrained by permissions, verification, accountability

---

## 8. 与你工作的关联 (build your intuition的角度)

Andrej, 这篇paper实际上把你过去几年散落的几个intuition unify到一个frame里:

1. **micrograd**教backprop直觉 — 这里对应的是deep telemetry作为harness optimization的gradient signal。 Agent loop本身要backprop through execution traces才能改进。

2. **"Software 2.0" essay** — Software 1.0是explicit instructions, 2.0是weights。 这篇paper论证: 真正production-ready的autonomy需要Software 1.0作为harness围在Software 2.0周围, 把learned policy的每一步ground到executable program state上。 所以不是Software 1.0 vs 2.0, 是1.0作为2.0的governor。

3. **llm.c / nanoGPT** — focus在base model。 这篇survey提醒我们: 一个部署到production的agent, base model只占整个system的一小部分, 大部分engineering effort在harness上。

4. **Eureka Labs / education** — 这篇paper的taxonomy其实可以当作agent systems的"教学大纲"。 三层interface/mechanisms/scaling, 加上5个application domain, 加7个open problems, 是一个complete curriculum。

5. **你Tesla autopilot经验** — §5.1.3 Embodied Agents直接对应。 Layered harness: semantic (foundation models) → code/classical robotics (admissibility boundary) → perception → physical controllers。 这跟Tesla的vision+planner+control stack同构。 关键insight: code不only是action, 也是safety boundary。

6. **"State of GPT" talk里你提到system 2 thinking** — PEV loop就是给LLM装一个system 2的executive control, 让它在sandbox里试错, 用deterministic sensors verify, 而不是next-token prediction一把过。

7. **"Deep Learning: System 1 and System 2" — Kahneman对应** — 这篇paper的Plan-Execute-Verify本质上是在LLM外面建一个Kahneman System 2: slow, deliberative, verifiable, gated。

---

## 9. 几个我认为paper没充分展开的方向 (为你提供research ideas)

1. **Harness metaprogramming**: 现在AHE还是evolution agent改harness code。 如果harness本身是 homoiconic (code = data, like Lisp), agent可以inspect自己的harness state, 用同一种language reasoning about harness structure, 可能比NL-based harness spec更精确。 Lean4Agent走了一半这个方向。

2. **Information-theoretic bounds on shared state**: §4.2 paper提到"code-mediated channels do not eliminate coordination bottlenecks" — 这其实是个information theory问题。 Channel capacity, compression loss, staleness, authority conflict都是可形式化的。 一个shared harness substrate的"bandwidth"是多少bits/step? agent belief divergence的Shannon entropy bound?

3. **Causal inference in failure attribution**: 现在failure attribution accuracy才14-53% [Cemri et al. 2025] https://arxiv.org/abs/2503.13657。 这其实是causal discovery问题 — 从execution traces里infer哪个agent/tool/action caused failure。 Do-calculus或structural causal model可能比gradient agent的heuristic attribution更principled。

4. **World model与action model的joint training**: §2.3说program-based world model, §5.2.6 multimodal说要predict next state作为action verification。 这两个可以joint train — 让agent学 $T(s, a) = s'$ 和 $\pi(a|s)$ 同时, 用prediction error作为auxiliary loss。 这跟你早期在Tesla做的multimodal prediction有联系。

5. **Self-driving lab的Lean化**: 现在XDL是LLVM IR for chemistry, 但没有formal verification。 如果wet-lab protocol用Lean写, 每个reaction step都有形式化precondition/postcondition, agent可以在执行前formally verify protocol不会产生危险副产物。 Lean4Physics和PhysLib已经把Lean推到physics, 推到chemistry是natural next step。

6. **HITL as conjugate prior**: §5.2.5把HITL frame成durable state。 从Bayesian角度, human approval是harness belief的conjugate prior update — 每次approval/rejection都update一个probability distribution over (action, safety) tuples。 这给harness一个principled uncertainty model。

7. **Code as Toolformer目标**: Toolformer [Schick et al. 2023] 教LLM何时call tool。 这篇paper暗示一个更深的版本: 教LLM何时synthesize tool (即write a new function), 何时synthesize harness piece, 何时modify existing harness。 这跟CodeAdapt [Zhang et al. 2025] https://arxiv.org/abs/2510.20909 "code-enabled LLMs outperform reasoning models"的发现一致。

---

## 10. 一个具体实验protocol, 帮你直观感受harness engineering

如果你想用一晚上build intuition, 可以做这个minimal experiment:

**Setup**: 一个简单的SWE-bench-lite task, 用Claude Sonnet 4 / GPT-4 / Gemini 2.5作为base model (固定不变), 跑下面3个harness configuration:

**Config A (minimal)**: prompt + LLM一次性生成patch, 不verify。 这相当于zero-shot。

**Config B (PEV basic)**: 加上sandbox execution + test as deterministic sensor + 1次retry on failure。 

**Config C (PEV + memory + structured retrieval)**: Config B + AutoCodeRover-style repo retrieval + CodeMem-style working memory + 3次retry with reflection。

预测: C > B > A, gap可能10-30% resolve rate。 这个gap就是harness engineering的价值, 跟base model大小无关。

如果你想让student感受harness self-evolution, 加Config D: 一个evolution agent读Config C的telemetry, 自动调整retry limit + retrieval chunk size + reflection prompt, 跑10个task后promote新的harness config。 看Config D vs Config C的gap。

---

## 11. 一些可能让你觉得"啊原来这样"的specific细节

- **QualityFlow Imagined Execution 98%+ on MBPP** — 这意味着LLM已经internalize了大量Python execution semantics, 真正execution的marginal value不在大多数correctable bugs, 而在corner cases和resource/perf dimensions。 这跟你"LLM在reasoning上还能走多远"的怀疑形成有趣对照: model内部其实已经在做simulation, harness的真execution是verification oracle而非primary computation engine。

- **MAGE的waveform feedback到sub-clock-edge granularity** — 这是finest-grained execution feedback。 RTL/hardware domain给我们一个extreme example of什么是"deterministic sensor"。

- **Self-Collaboration的simulated tester和real compiler ablation几乎一样** — 跟QualityFlow相互印证。 当LLM能predict execution, harness的role从"提供computation"转向"提供grounded verification"。

- **Topology complexity inversely correlates with substrate formality** — 这是最有预测力的pattern。 设计multi-agent system时, 先想shared substrate长什么样, topology自然简化。

- **LingmaAgent在Alibaba 16.9% autonomous + 43.3% with intervention** — production deployment的真实数字。 这是2026年的SOTA-ish水平。 离full autonomy还很远, 但是workflow participant已经可行。

- **El Agente Q 87%+ on 6 university benchmarks** — chemistry agent在程序化env里已经很厉害。 

- **DiscoveryBench best system ~25%** — open-ended discovery还是unsolved, 跟code assistant的resolve rate形成鲜明对比。 Discovery的oracle比code tests难多了。

- **Failure attribution 14-53%** [Cemri et al. 2025] — production harness缺structured traces来做principled debugging。 这是一个明显的research gap, 跟software observability (OpenTelemetry, Jaeger)的成熟形成对比。

---

## 12. 总结的mental model

整篇paper的thesis压缩成一句话:

> **Code is not just what agents generate; it is the executable, inspectable, stateful medium through which agents reason, act, and adapt.**

对你build intuition有用的一句话:

> **Agentic AI的真正bottleneck不在base model的reasoning capability, 而在harness infrastructure的reliability — 而code是这层infrastructure的natural substrate, 因为它executable, inspectable, stateful。**

这跟你当年写"Software 2.0"的姿势完全一致 — 重新frame一个领域, 给field一个新vocabulary, 让大家看到之前没注意的结构。 这篇survey给agentic AI field的礼物就是这个"harness"的vocabulary, 加上code作为substrate的unification。

如果你想深入, 我推荐你从这几个paper开始读:

1. AutoHarness — 看harness synthesis的具体algorithm
2. SyncMind — 看belief divergence的formalization
3. L2MAC — 看最principled blackboard实现
4. AHE paper — 看evolution agent的5-stage loop
5. AI Scientist v2 — 看scientific agent的完整PEV implementation

GitHub awesome list: https://github.com/YennNing/Awesome-Code-as-Agent-Harness-Papers

---

*Last meta-note: 这篇survey本身也是一个harness artifact的example — 它是persistent, queryable, version-controlled的knowledge structure, 让后续agent (和人类researcher)可以build on top, 而非每次从零reason。*
