---
source_pdf: METAGPT.pdf
paper_sha256: 17582e47647fac8eaa3681feade41c8408310c4133acc5c6823565fbe880e1ac
processed_at: '2026-08-05T18:01:39-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MetaGPT 人话版

## 一句话说清楚

让一堆 LLM agent 自由对话做软件 → 会崩，因为每多一轮对话就多一次 hallucination sampling，像传话游戏一样信息越来越歪。MetaGPT 的解法：**别让它们聊天，让它们像真实软件公司一样按 SOP 流水线交接文档**。

---

## 问题出在哪

naive multi-agent 的画风是这样的：

> Agent A (PM): "Hi, how are you?"
> Agent B (Architect): "Great! Had lunch?"

这种 idle chatter 在 token 上烧钱，在信息上注入 noise。更糟的是 cascading hallucination：A 产出一句话带点幻觉，B 基于这句幻觉再生成，C 再基于 B 的输出...error 像 compound interest 一样累积。

LLM 本质是 next-token sampler，你让它们自由 chat 越多轮，joint distribution 偏离正确解的概率就越大。这不是 prompt engineering 能救的，是 system design 的问题。

---

## MetaGPT 的 insight

人类软件公司其实早就解决了这个问题——用 **Standardized Operating Procedures (SOPs)**。Product Manager 不会跟 Engineer 闲聊 "今天天气怎么样"，而是交一份 PRD 文档。Architect 不会口述设计，而是画 UML 图、写 API spec。

MetaGPT 把这套 human workflow 直接编码进 multi-agent system：

```
用户一句话需求
   ↓
Product Manager → 产出 PRD（需求文档，含 user stories, requirement pool）
   ↓
Architect → 产出 System Design（文件列表, 数据结构, 接口定义, 流程图）
   ↓
Project Manager → 产出 Task List（按文件拆任务）
   ↓
Engineer → 产出 Code（带 executable feedback 的迭代）
   ↓
QA Engineer → 产出 Unit Tests
   ↓
能跑的软件
```

每一步交接的是 **结构化文档**，不是自由对话。PRD 有 schema，file list 是 Python list 形式，interface definition 是标准格式——下一个 agent 可以直接 parse，不需要 LLM 去 "理解" 上一个 agent 的闲聊。

---

## 两个关键技术 trick

### Trick 1: Shared Message Pool + Subscribe

之前的工作（CAMEL、ChatDev）让 agent 两两对话：PM 跟 Architect 说，Architect 跟 Engineer 说...这是 N×N 的通信拓扑，效率差，信息容易丢。

MetaGPT 搞了个公共消息池：

- 每个 agent 把自己的产出 **publish** 到池子里
- 每个 agent 根据自己角色 **subscribe** 相关消息
- Architect 只关心 PRD，不看 QA 的 test report
- Engineer 关心 system design 和 task list，不看 competitive analysis

这本质是 blackboard architecture（80 年代 AI 经典架构）的 LLM 版本。通信复杂度从 O(N²) 降到 O(N)，而且信息过滤避免了 information overload。

### Trick 2: Executable Feedback

之前 self-reflection 类工作（Reflexion、ChatDev）让 LLM 自己 review 自己的代码。问题是 LLM review LLM 还是 hallucination——自己很难发现自己逻辑错在哪。

MetaGPT 让 Engineer 写完代码 **直接 run**：

```
写代码 → 跑 unit test → 挂了看 traceback → 修 → 再跑
（最多重试 3 次）
```

runtime error 是 ground truth signal，不骗人。TypeError 就是 TypeError，ModuleNotFoundError 就是 ModuleNotFoundError。用 execution signal 当 verifier，比 LLM self-judge 靠谱得多。

这跟 Execution-Guided Neural Program Synthesis (Chen et al., 2018) 的思想一脉相承，只不过用 LLM agent 实现。

---

## 结果有多猛

### Code Generation Benchmark

| Method | HumanEval Pass@1 | MBPP Pass@1 |
|---|---|---|
| GPT-4 单独 | 67.0% | - |
| CodeLlama | 53.7% | - |
| WizardCoder | 73.2% | - |
| **MetaGPT + GPT-4** | **85.9%** | **87.7%** |
| MetaGPT w/o feedback | 81.7% | 82.3% |

Executable feedback 单独贡献 +4.2%（HumanEval）和 +5.4%（MBPP）。

### 真实软件开发任务

他们自建了 70 个 software task（小游戏、CRUD、数据处理等），对比 AutoGPT、LangChain、AgentVerse、ChatDev：

| Framework | 平均 Executability (1-4) |
|---|---|
| AutoGPT | 1.0（全挂） |
| LangChain | 1.0（全挂） |
| AgentVerse | 1.0（全挂） |
| ChatDev | 2.1（勉强能跑） |
| **MetaGPT** | **3.9（几乎完美）** |

MetaGPT 在 7 个主实验任务上 **100% completion rate**。

### 效率对比（vs ChatDev）

| 指标 | ChatDev | MetaGPT |
|---|---|---|
| 生成代码总行数 | 77.5 | **251.4** |
| 每行代码消耗 token | 248.9 | **124.3** |
| 人需要手动改的地方 | 2.5 处 | **0.83 处** |

MetaGPT 生成的代码量是 ChatDev 的 3 倍，但每行代码反而省一半 token，而且人几乎不用改——直接能跑。

---

## 为什么这个设计 work

构建直觉：

**1. SOP 是 inductive bias**
人类几十年软件工程总结出 PM → Architect → Engineer → QA 这个流程是有原因的。MetaGPT 把这个 prior 注入 LLM agent system，相当于给了一个很强的 curriculum。不需要 LLM 自己摸索怎么协作，human knowledge 已经告诉它了。

**2. Structured output 是 anti-hallucination 工具**
自然语言 "我们用一个 GUI 库吧" 模糊得很，但 `"Tkinter"` 是 determinstic token。强制 schema 输出逼 LLM commit 到具体实体，减少 hedging 和 ambiguity。这跟 chain-of-thought 思路一致，但更严格——schema 是 hard constraint。

**3. Document 交接过滤了 hallucination**
PRD 写错了，Architect 基于 PRD 做设计时会被 schema 检验——如果 PRD 里 requirement pool 漏了关键需求，Architect 的 file list 就对不上。每一步的 structured artifact 都是对上一步的 implicit verification。

**4. Ablation 证明了 role 的价值**
单 Engineer 生成代码 executability 只有 1.0（全挂）。加 PM 到 2.0，加 Architect 到 2.5，全角色到齐到 4.0。每个 role 都在 suppress 不同类型的 hallucination——PM 抑制需求模糊，Architect 抑制技术选型幻觉，QA 抑制代码 bug。

---

## 跟你工作的 connection

这让我想到 micrograd 的哲学：explicit, readable, step-by-step。MetaGPT 在 multi-agent 层面做的事类似——每个 agent 的产出都是人类可读的 document（PRD、design、code），整个流程透明可审计。

这跟 "Software 2.0"（weights 不可读）形成有趣对比。MetaGPT 在 "Software 3.0"（natural language programming）中保留了 Software 1.0 的 readability——agent 之间的通信是文档，不是黑盒 weights 的梯度更新。

Executable feedback 那块跟你 "Practical ML for Coders" 的思路一致：用 execution 当 ground truth，别让 LLM 自我评价。runtime 不会骗人。

---

## 局限性我也得说

1. **SOP 是 hardcoded 的**：真实 agile 团队的流程会动态调整，MetaGPT 的 pipeline 是固定的
2. **规模天花板**：实验最大 251 LOC，100k LOC 的 enterprise 项目能不能 scale 不知道
3. **Self-improvement 只改 constraint prompt**：Appendix A 讨论的 recursive self-improvement 只动 role 的 constraint，没动 communication schema，learning 上限有限
4. **Token cost 仍然高**：31k token 生成 251 行 code，比人类 programmer 慢很多——但这是 LLM 本质限制，不是 MetaGPT 的问题

---

## 总结

MetaGPT 的核心贡献就三句话：

1. **用 SOP 替代自由对话** → 把 human domain knowledge 作为 inductive bias 注入
2. **用 structured document 替代 natural language chat** → suppress cascading hallucination
3. **用 executable feedback 替代 LLM self-review** → 用 runtime signal 当 ground truth

本质上是在 multi-agent system 里把人类软件工程 best practice 显式编码进去，而不是让 LLM 自己摸索怎么协作。结果就是这个 approach 在 code generation 和 software development 上都打了 SOTA。

> Paper: [arxiv.org/abs/2308.00352](https://arxiv.org/abs/2308.00352)
> Code: [github.com/geekan/MetaGPT](https://github.com/geekan/MetaGPT)

---

# MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework 深度解析

Andrej，这篇 paper 的核心 insight 非常有意思：把人类软件公司的 Standardized Operating Procedures (SOPs) 显式编码进 LLM-based multi-agent system，让 agent 之间通过 structured documents 通信，从而 suppress cascading hallucination。我分层给你 build intuition。

---

## 1. 核心动机：为什么 naive multi-agent 会失败

之前的 multi-agent framework（ChatDev、AutoGPT、AgentVerse）采用 unconstrained natural language dialogue 让 agent 互相对话。问题在于 telephone game / Chinese whispers 效应：每一轮 LLM 生成都会引入 hallucination，naively chaining LLMs 会导致 error 累积放大 (cascading hallucinations)。

举个 paper 中提到的反例：
> "Hi, hello and how are you?" – Alice (Product Manager)
> "Great! Have you had lunch?" – Bob (Architect)

这种 idle chatter 消耗 token 同时引入 noise。MetaGPT 的 thesis 是：**如果让 agent 输出结构化的 artifact（PRD、design doc、API spec、flowchart），并强制 SOP 工作流，hallucination 会被中间结构化产物过滤掉**。

---

## 2. 架构解析：SOP-Driven Multi-Agent

### 2.1 Role Specialization（角色分工）

MetaGPT 模拟一家 software company，定义 5 个 role：

| Role | 职责 | 输出 artifact | 工具 |
|---|---|---|---|
| Product Manager | business analysis, 竞品分析 | PRD (含 User Stories, Requirement Pool, quadrant chart) | web search |
| Architect | technical translation | System design (File List, Data Structures, Interface Definitions, sequence flow) | - |
| Project Manager | task decomposition | Task list, shared knowledge | - |
| Engineer | implementation | Code files | code execution |
| QA Engineer | quality assurance | Unit tests, bug reports | - |

每个 agent 遵循 ReAct (Yao et al., 2022) 的 reasoning-acting loop，monitor 环境（message pool）→ observe relevant message → take action。

### 2.2 SOP Workflow（流水线范式）

整体 pipeline 严格 sequential：

```
User Requirement
    ↓
[Product Manager] → PRD (User Stories + Requirement Pool + Competitive Analysis)
    ↓
[Architect] → System Design (File List + Data Structures + Interface Defs + Sequence Flow)
    ↓
[Project Manager] → Task List (per-file task decomposition)
    ↓
[Engineer] → Code Implementation (iterative with executable feedback)
    ↓
[QA Engineer] → Unit Tests + Bug Fixes
    ↓
Final Software
```

这是 assembly line paradigm：每个 role 输出的 structured artifact 直接作为下一个 role 的 input，**不存在自由对话**。

---

## 3. Communication Protocol：信息流的核心设计

### 3.1 Structured Communication Interfaces

每个 role 有自己的 output schema。例如 Architect 必须输出：
- `## Implementation approach`
- `## Python package name`
- `## File list` (Python list 形式)
- `## Anything UNCLEAR`

这种强制 schema 有两个好处：
1. **消除 ambiguity**：natural language 中 "我们用一个 GUI 库吧" 这种表述太模糊，而 `"Tkinter"` 是 determinstic token
2. **可 machine-parse**：下一个 agent 可以直接 JSON/CSV parse 上游输出，不需要 LLM 再去理解

### 3.2 Publish-Subscribe Mechanism + Shared Message Pool

这是 paper 中我认为最 elegant 的设计。所有 agent 共享一个 global message pool：

```
┌─────────────────────────────────────────────┐
│         Shared Message Pool                  │
│  [msg1: PRD from PM]                         │
│  [msg2: SystemDesign from Architect]         │
│  [msg3: TaskList from ProjMgr]               │
│  [msg4: Code from Engineer]                  │
│  [msg5: TestResult from QA]                  │
└─────────────────────────────────────────────┘
       ↑ publish         ↑ subscribe (filtered by role)
   [any agent]        [any agent]
```

- **Publish**：agent 完成任务后把 structured output push 到 pool
- **Subscribe**：agent 根据自己 role profile 只订阅相关消息，过滤无关信息避免 information overload

例如 Architect 主要 subscribe PRD，对 QA 的 test report 不感兴趣。这避免了 Li et al. (2023) 那种 1-to-1 dialogue 的 N×N 通信拓扑。

**Intuition**: 这本质上是把 blackboard architecture（1980s AI 经典架构）+ observer pattern 用 LLM agent 重新实现。比自由 chat 高效得多，因为 N 个 agent 的通信复杂度从 O(N²) 降到 O(N)。

---

## 4. Iterative Programming with Executable Feedback

这是 paper 中另一关键技术贡献，针对 code hallucination 问题。

### 4.1 机制

Engineer agent 写完 code 后：
1. 执行 unit test
2. 如果 fail，从 memory 中 retrieve 过往 message（PRD + system design + 历史 code）
3. 对比 error 信息与 spec，进行 debug
4. 重新生成 code
5. 重复直到 pass 或达到 **max 3 retries**

```
while not test_pass and retries < 3:
    code = generate_code(prd, design, past_errors)
    test_result = execute_unit_test(code)
    if test_result.pass:
        break
    memory.append({code, test_result})
```

### 4.2 为什么有效

之前的 self-reflection 工作（Reflexion、ChatDev 的 code review）是 **non-executable** 的——LLM 自己 review 自己的 code，本质还是 hallucination。MetaGPT 引入 **ground truth signal**：执行结果（traceback、test pass/fail）作为外部 oracle 反馈给 LLM。这跟 Execution-Guided Neural Program Synthesis (Chen et al., 2018) 的思想一脉相承，但用 LLM agent 实现。

---

## 5. 实验：SOTA 结果与详细数据

### 5.1 Code Generation Benchmarks

**Pass@k 公式**：

$$\text{Pass@}k = \mathbb{E}_{\text{Problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

变量解释：
- $n$ = 每个 problem 生成的总 sample 数（通常 n=200）
- $c$ = 这 n 个 sample 中通过 test 的正确 sample 数
- $k$ = 我们取前 k 个 sample 评估（如 k=1 表示单次生成）
- $\binom{n-c}{k}$ = 从 c 个错误 sample 中选 k 个的组合数
- $\binom{n}{k}$ = 从 n 个 sample 中选 k 个的组合数
- 整个表达式表示：随机抽 k 个 sample，**至少有一个正确** 的概率期望

这个 unbiased estimator 修正了原始 GPT-3 paper 中 high variance 的问题。

**HumanEval & MBPP 结果**：

| Method | HumanEval Pass@1 | MBPP Pass@1 |
|---|---|---|
| CodeX | 28.8% | - |
| AlphaCode | - | - |
| CodeT | 65.8% | - |
| PaLM | 26.2% | - |
| GPT-4 | 67.0% | - |
| CodeLlama | 53.7% | - |
| WizardCoder | 73.2% | - |
| **MetaGPT (GPT-4)** | **85.9%** | **87.7%** |
| MetaGPT w/o feedback | 81.7% | 82.3% |

Executable feedback 单独贡献 **+4.2% (HumanEval)** 和 **+5.4% (MBPP)** 的绝对提升。

### 5.2 SoftwareDev Benchmark（自建）

70 个 representative software tasks，主实验用 7 个：Snake game, Brick breaker, 2048, Flappy bird, Tank battle, Excel data process, CRUD manage。

**Executability 评分 1-4**：
- 1: non-functional
- 2: runnable but imperfect
- 3: nearly perfect
- 4: flawless

| Framework | AutoGPT | LangChain | AgentVerse | ChatDev | **MetaGPT** |
|---|---|---|---|---|---|
| Avg Executability | 1.0 | 1.0 | 1.0 | 2.1 | **3.9** |

MetaGPT 在 7 个 task 上 100% completion rate，AutoGPT/LangChain/AgentVerse 全部 fail（1.0）。

### 5.3 Cost & Productivity 对比（vs ChatDev）

| Statistical Index | ChatDev | MetaGPT w/o Feedback | MetaGPT |
|---|---|---|---|
| (A) Executability | 2.25 | 3.67 | **3.75** |
| (B) Running Time (s) | 762 | 503 | 541 |
| (B) Token Usage | 19,292 | 24,613 | 31,255 |
| (C) Code Files | 1.9 | 4.6 | **5.1** |
| (C) Lines per File | 40.8 | 42.3 | **49.3** |
| (C) Total Code Lines | 77.5 | 194.6 | **251.4** |
| (D) Productivity (token/line) | 248.9 | 126.5 | **124.3** |
| (E) Human Revision Cost | 2.5 | 2.25 | **0.83** |

**关键 intuition**：MetaGPT 用更多 token（31k vs 19k），但每行 code 只耗 124.3 token（ChatDev 是 248.9），**token efficiency 翻倍**。Human revision 从 2.5 降到 0.83——意味着生成的 code 几乎可以直接跑。

### 5.4 Ablation: Roles 的贡献

| Engineer | Product | Architect | Project | #Agents | Executability | Revisions |
|---|---|---|---|---|---|---|
| ✓ | | | | 1 | 1.0 | 10 |
| ✓ | ✓ | | | 2 | 2.0 | 6.5 |
| ✓ | ✓ | ✓ | | 3 | 2.5 | 4.0 |
| ✓ | ✓ | | ✓ | 3 | 2.0 | 3.5 |
| ✓ | ✓ | ✓ | ✓ | 4 | **4.0** | 2.5 |

**Intuition**: 单 Engineer 只能产生 1.0（完全 fail）。加 Product Manager 跳到 2.0。完整 4 role 才能达到 4.0 flawless。每个 role 都在 suppress 不同类型的 hallucination。

### 5.5 Instruction Level Impact

| Prompt Type | #Words | Executability | Productivity |
|---|---|---|---|
| High-level ("create a brick breaker game") | 13.2 | 3.8 | 163.8 |
| Detailed | 42.2 | 4.0 | 118.0 |

**惊人发现**：即使 high-level prompt，MetaGPT 也能达到 3.8/4.0。这意味着 SOPs 自身承担了 requirement elaboration 的工作，**降低了用户 burden**。

---

## 6. Demo: 完整 SOP 流程示例

以 "write a python3 GUI app such that you can draw an image with it" 为例：

### 6.1 Product Manager 输出 PRD
```python
# Product Goals
["Create a user-friendly GUI color meter",
 "Ensure accurate RGB output",
 "Real-time RGB updates"]

# User Stories  
["As a user, I want to select any color on screen...",
 "As a user, I want real-time RGB updates..."]

# Requirement Pool
[("Design user-friendly GUI", "P0"),
 ("Implement color selection", "P0"),
 ("Display RGB values", "P0"),
 ("Real-time updates", "P0"),
 ("Test accuracy", "P1")]
```

PM 还生成了 competitive quadrant chart（Figure 7），定位 Color Cop、Just Color Picker 等竞品。

### 6.2 Architect 输出 System Design
```python
# Implementation approach
"We will use Python's Tkinter library...
The color selection uses PIL (Pillow)...
threading for real-time updates..."

# File list
["main.py", "color_picker.py", "gui.py", "tests.py"]
```

Architect 还要生成 data structure UML (Figure 8) 和 sequence flow chart (Figure 9)。

### 6.3 Engineer 实现
```python
class ColorPicker:
    def __init__(self, root: tk.Tk, color: str = 'black'):
        self.frame = tk.Frame(root)
        self.color = color
        self.color_button = tk.Button(self.frame, text='Color', 
                                       command=self.select_color)
    def select_color(self):
        color = colorchooser.askcolor()[1]
        if color is not None:
            self.color = color
```

### 6.4 QA Engineer 测试
```python
class TestColorPicker(unittest.TestCase):
    def test_initial_color(self):
        self.assertEqual(self.cp.get_color(), 'black')
    def test_set_and_get_color(self):
        new_color = '#ffffff'
        self.cp.color = new_color
        self.assertEqual(self.cp.get_color(), new_color)
```

最终输出 Figure 10 的 functional "Drawing App"。

---

## 7. Capabilities 对比

| Capability | AutoGPT | LangChain | AgentVerse | ChatDev | MetaGPT |
|---|---|---|---|---|---|
| PRD generation | ✗ | ✗ | ✗ | ✗ | ✓ |
| Technical design | ✗ | ✗ | ✗ | ✗ | ✓ |
| API interface gen | ✗ | ✓ | ✗ | ✗ | ✓ |
| Code generation | ✓ | ✓ | ✓ | ✓ | ✓ |
| Precompilation execution | ✗ | ✗ | ✗ | ✗ | ✓ |
| Role-based task mgmt | ✗ | ✓ | ✗ | ✓ | ✓ |
| Code review | ✗ | ✓ | ✓ | ✓ | ✓ |

MetaGPT 是唯一一个全 ✓ 的 framework。

---

## 8. Outlook: Self-Improvement & Multi-Agent Economy

### 8.1 Recursive Self-Improvement
Paper Appendix A 讨论 future direction：每个 agent 在 project 结束后 review 历史 feedback，**修改自己的 constraint prompt**。这呼应 Schmidhuber 的 Gödel Machines (2003, 2009) 思想——self-referential self-improvement。

```
for project in projects:
    for agent in company:
        feedback = summarize(project_history)
        agent.constraint_prompt = update(agent.constraint_prompt, feedback)
```

### 8.2 Economy of Minds (EOM)
基于 NL SOM (Zhuge et al., 2023) 提出的 free market 信用分配：agents 通过 supply/demand 定价服务，贡献 reward 的 agent 获得 credit。DeepWisdom 的 AgentStore 已经实现了这个雏形——agents 可以购买彼此的服务。

---

## 9. Limitations & Ethics

### Limitations
- **System side**: 缺少 UI/frontend agent 和 multimodal tools
- **User side**: 无法 checkpoint/interrupt agent 运行

### Ethics
- **Unemployment concern**: paper 论证 natural language programming 降低了 programming 门槛，反而增加 demand（类比 assembly → C → Python 的演化）
- **Transparency**: open-source, 本地运行，所有 reasoning 可观察
- **Privacy**: 不收集 user data，支持 open-source LLM backend

---

## 10. 我的 Critical Analysis（build your intuition）

### Strengths
1. **SOP 是 inductive bias**: 把人类几十年软件工程 best practice 作为 prior 注入 LLM agent system，相当于 curriculum design
2. **Structured output 作为 anti-hallucination 工具**: 这跟 scratchpad、chain-of-thought 思路一致，但更严格——schema 强制 LLM 输出 determinstic structure
3. **Executable feedback 是 RL-lite**: 用 runtime signal 而非 LLM judgment 作 verifier

### Open Questions
1. **Scalability**: 5 个 role 是否足以应对 100k LOC 项目？paper 实验最大 251 LOC
2. **SOP rigidity**: real-world agile 团队的 SOP 是动态的，MetaGPT 的 SOP 是 hardcoded
3. **Token cost**: 31k token 生成 251 行 code——比人类 programmer 的 token efficiency 低很多，但这是 LLM 本质限制
4. **Self-improvement 只改 constraint prompt**: 没有改 communication schema 本身，limit 了 learning 上限

### Connection to 你的工作
MetaGPT 的 PRD → Design → Code pipeline 本质上是把 micrograd 那种 "explicit, readable, step-by-step" 的 philosophy 用到 multi-agent system 上。每个 agent 输出的 artifact 都是人类可读的 document，这跟 "software 2.0" 中 weights 不可读形成对比——MetaGPT 在 software 3.0（natural language programming）中保留了 software 1.0 的 readability。

Executable feedback 机制让我想到你的 "Practical ML for Coders" 思路：用 execution 作为 ground truth，避免 LLM self-evaluation 的循环偏差。

---

## References

- MetaGPT Paper: [https://arxiv.org/abs/2308.00352](https://arxiv.org/abs/2308.00352)
- GitHub: [https://github.com/geekan/MetaGPT](https://github.com/geekan/MetaGPT)
- ReAct: [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- Reflexion: [https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
- ChatDev: [https://arxiv.org/abs/2307.07924](https://arxiv.org/abs/2307.07924)
- Generative Agents: [https://arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
- NL SOM: [https://arxiv.org/abs/2305.17066](https://arxiv.org/abs/2305.17066)
- HumanEval: [https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)
- MBPP: [https://arxiv.org/abs/2108.07732](https://arxiv.org/abs/2108.07732)
- Execution-Guided Synthesis: [https://arxiv.org/abs/1806.10453](https://arxiv.org/abs/1806.10453)
- Gödel Machines: [https://arxiv.org/abs/cs.LO/0309048](https://arxiv.org/abs/cs.LO/0309048)
- Chain-of-Thought: [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
- AgentVerse: [https://arxiv.org/abs/2308.10848](https://arxiv.org/abs/2308.10848)
- ToolLLM: [https://arxiv.org/abs/2307.16789](https://arxiv.org/abs/2307.16789)
- CAMEL: [https://arxiv.org/abs/2303.17760](https://arxiv.org/abs/2303.17760)
- Schmidhuber Meta-learning: [https://people.idsia.ch/~juergen/diploma1987cc.pdf](https://people.idsia.ch/~juergen/diploma1987cc.pdf)

总结一句: MetaGPT 的本质贡献是把 **structured human workflow 作为 LLM multi-agent 的 inductive bias**，用 document artifact 替代 free-form dialogue 作为 inter-agent 通信媒介，用 executable feedback 作为外部 verifier——三管齐下 suppress cascading hallucination。
