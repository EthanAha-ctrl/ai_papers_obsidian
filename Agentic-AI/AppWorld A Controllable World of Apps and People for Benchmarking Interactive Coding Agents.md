---
source_pdf: AppWorld A Controllable World of Apps and People for Benchmarking Interactive
  Coding Agents.pdf
paper_sha256: 2ca1feab1559d678c28706acca11caab54a8a76c64dcf9e14d46f7bd33f21487
processed_at: '2026-08-18T01:07:49-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# AppWorld 用人话讲

## 一句话总结

现在的 agent benchmark 都是"玩具级"的，这篇paper搞了个真实的"app世界"来测试 agent 能不能像人一样用手机app干活，结果发现 GPT-4o 连一半任务都做不完。

## 为什么要搞这个

你想象一下你的日常：早上起来看 Simple Note 里的 workout plan，然后 Spotify 放个时长够的 playlist；中午跟室友商量吃饭，Venmo 转账；晚上 Amazon 下单买东西，还要查 Gmail 里的 confirmation email。

这些任务有个关键特点：**你没法提前知道所有信息，必须边干边查**。

举个例子，"放一个时长够今天锻炼的playlist"这个任务，agent 必须先去 Simple Note 读那个 note，看到它是按天列的（Monday: 30mins, Tuesday: 45mins...），才能知道今天是多少分钟。然后还要去 Spotify 把每个 playlist 的所有 song duration 加起来，找一个 >= workout duration 的。这不是一次写完代码就能搞定的。

但是你看看现有的 benchmark：

- **API-Bank**: 1-4 个 API call，线性序列
- **ToolBench**: 也是浅层的 API 调用
- **Gorilla**: 就是 API retrieval，不用 compose

这些 benchmark 跟真实任务的差距，就像"加法练习"和"解微分方程"的差距。现有的 benchmark 测不出"rich code generation + iterative interaction with environment"这种能力，而这恰恰是 daily task automation 的核心。

所以作者花了14个月，写了100K+ lines of code，搞了 AppWorld。

## AppWorld 到底是什么

### AppWorld Engine：一个假的但很真的 app 世界

9个 app，每个都模拟真实 app 的功能：

```
Gmail      - 收发邮件
Venmo      - 转账、收款
Amazon     - 购物、退货、wishlist
Spotify    - 播放音乐、playlist管理
Phone      - 通讯录、短信
Simple Note - 笔记
Splitwise  - 分账
File System - 文件操作
Calendar   - 日程
```

外加2个 helper app：
- **ApiDocs**: 让 agent 动态查 API 文档（因为457个API的文档太长，塞不进context）
- **Supervisor**: 提供 task assigner 的信息（地址、信用卡、各 app 的密码）

总共 **457 个 API**，1470 个参数，101 张数据库表，~370K 行数据。

里面住着106个虚构的人，他们之间有各种关系：室友、朋友、家人、同事。每个人的每个 app 都有两三年的活动历史——Venmo 上有跟室友分房租的记录，Gmail 里有跟同事约会议的邮件，等等。

**关键技术决策：全部塞进一个 Python process**

这个很重要。通常 web app 要跑 database server、search engine server、web app server、client——四个 process。AppWorld 用 SQLite（serverless）+ FastAPI TestClient（server和client同一process），全塞一起。

好处是：你做实验的时候不用 start/stop/restart servers，加载一个新 task <0.5秒。听起来是小事，但当你要跑几百个 task、每个 task 要 reset 环境重跑的时候，这个决定了实验可行性。

### AppWorld Benchmark：750 个真实复杂的任务

250 个 scenario，每个 scenario 实例化出3个 task，总共750个。

举个 scenario 例子：
> "I like the last {color} {apparel} I bought on Amazon, repurchase the same in that size. Prefer {preferred-color}, if available; otherwise, go with the same color."

通过换 {color}, {apparel}, {preferred-color} 和初始 DB state，变出3个 task。这3个 task 形成 **contrast set**——一个 task 里 T-shirt 有 preferred color，另一个只有 original color，测的是 agent 能不能在不同条件下都正确处理。

每个 task 要满足四个性质：

**1. Well-Defined**: 任务的前提高在 DB state 里成立。说"repurchase last T-shirt"，DB 里得真有 past T-shirt orders。

**2. Has Distractors**: 放一堆干扰项。supervisor 有多个 past T-shirt orders in different colors and sizes，agent 得 reasoning 才能找到 correct one。如果跳过 interaction 直接猜，大概率猜错。

**3. Has realistic hurdles**: 加自然障碍。比如 default payment card expired，agent 得 try another。这测的是 agent 能不能处理 deviation from typical route。

**4. Forms contrast set**: 同一 scenario 的不同 task 覆盖 diverse conditions。

### Evaluation：怎么判断任务做对了

这是 paper 最有意思的部分之一。

复杂任务有很多 valid solution path——你可以从 Amazon API 下 receipt，也可以从 confirmation email 下载。传统的 process-based evaluation（对比 reference solution）根本没法处理。

而且 agent 可能造成 **collateral damage**——比如你让它买T-shirt，它买了T-shirt但顺便删了你的 wishlist，这算成功吗？

作者的 solution：**state-based programmatic evaluation**。

定义两个 change set：
- $C_i^{\text{expect}}$: 必须发生的 DB changes（比如 exactly 1 order placed, 只有1个 product of type T-shirt）
- $C_i^{\text{allow}}$: 允许但非 mandatory 的 changes（比如下单后 cart 可以清空也可以留着）

Evaluation 条件用公式写就是：

$$C_i^{\text{expect}} \subseteq D_i^{\Delta} \subseteq C_i^{\text{expect}} \cup C_i^{\text{allow}}$$

这里 $D_i^{\Delta}$ 是 final DB state $D_i^f$ 和 start DB state $D_i^s$ 的 diff——哪些表的哪些行哪些列变了。

- 左边 $C_i^{\text{expect}} \subseteq D_i^{\Delta}$: 所有必须的 changes 都发生了
- 右边 $D_i^{\Delta} \subseteq C_i^{\text{expect}} \cup C_i^{\text{allow}}$: 没有意外的 changes（collateral damage detection）

**Hash-based diff 加速**

101张表360K行数据，naive 逐行比较太慢。他们在 DB 里额外维护：
- `record_hash`: 每行内容的 hash（排除ID列），内容变就变
- table counter: 这张表有多少行 hash 变过

Diff 时分三层做：
1. Tables: counter 没变的表直接跳过
2. Rows: 同 ID 同 record_hash 的行没变；end 有 start 没的是 added；反之 deleted
3. Columns: 只对 changed rows 逐列比较

最终 evaluation 平均 <0.6 秒/task。

## 两个核心 Metric

- **TGC (Task Goal Completion)**: 通过所有 evaluation tests 的 task 百分比
- **SGC (Scenario Goal Completion)**: 同一 scenario 的所有 tasks 都通过的 scenario 百分比

SGC 是个 consistency metric——你可以在某些 variant 上成功但在其他 variant 上失败，SGC 会暴露这个问题。

## 实验：6种方法

### Decision-making methods（动态探索文档）

**ReAct**: Thought → Code 循环，最多100次 LLM call。因为 trajectory 很长，只用 one-shot prompt。教 agent 用 ApiDocs 探索：`show_app_descriptions` → `show_api_descriptions` → `show_api_doc`。

**PlanExec**: 先生成 high-level plan，再每步用 ReAct 执行。Planner 用3-shot，executor 用5-shot。每步能看到之前 executed steps 的 history 和 code。

### Code-based methods（预先 retrieve relevant APIs）

因为 full documentation ~142K tokens 塞不下，先用一个 LLM 预测 relevant APIs（最多16个）。

**FullCodeRefl**: 一次生成全部代码，失败后看 error stacktrace 反思 retry，最多5次。用3-shot demonstrations（用自己写的 validation solutions）。

**IPFunCall**: 用 parallel function calling ability，每 turn 生成 batch of independent API calls。最多15 turns。因为能 batch call，相当于 implicit 的 for loop。

**ToolLLaMA**: fine-tuned LLaMA 生成 API call sequence。

**CodeAct**: Mistral-7B fine-tuned on CodeActInstruct。

### Context Truncation Strategy

ReAct 和 PlanExec 的 trajectory 会爆长（JSON outputs 是主要 contributor）。策略是：

```
threshold = 20000 chars
if trajectory > threshold:
    保留 last two steps 的 outputs
    从 oldest 开始用 [NOT SHOWN FOR BREVITY] 替换
    如果还不够，从 oldest 开始删整个 (thought, action, observation) blocks
    加 [TRIMMED HISTORY] 标记
```

## 结果：GPT-4o 也只做对一半

| Base LLM | Method | Test-N TGC | Test-N SGC | Test-C TGC | Test-C SGC |
|----------|--------|-----------|-----------|-----------|-----------|
| GPT-4o | ReAct | **48.8** | 32.1 | **30.2** | 13.0 |
| GPT-4o | PlanExec | 44.6 | 23.2 | 19.7 | 7.9 |
| GPT-4o | FullCodeRefl | 33.9 | 26.8 | 19.2 | 12.2 |
| GPT-4o | IPFunCall | 32.1 | 16.1 | 18.0 | 10.1 |
| GPT-4Turbo | ReAct | 32.7 | 12.5 | 17.5 | 5.8 |
| LLaMA3 | FullCodeRefl | 24.4 | 17.9 | 7.0 | 4.3 |
| DeepSeek | FullCodeRefl | 13.1 | 8.9 | 5.8 | 2.9 |
| Mistral-7B | CodeAct | 0.0 | 0.0 | 0.0 | 0.0 |
| LLaMA | ToolLLaMA | 0.0 | 0.0 | 0.0 | 0.0 |

几个关键观察：

**1. GPT-4o + ReAct 最好，但 normal task 也只做对 ~49%，challenge task ~30%**

这是在 controlled sandbox 里，真实世界只会更难。如果你让 GPT-4o 帮你干日常 digital task，它有超过一半概率失败或造成 collateral damage。

**2. TGC → SGC 掉30-50%**

比如 GPT-4o + ReAct 在 Test-N 上 TGC=48.8，但 SGC 只有32.1。这说明 models 在某些 variant 上成功但在 similar variant 上失败——**consistency 比 peak performance 差得多**。对于真正 autonomous 的 daily task automation，consistency 才是关键。

**3. GPT-4Turbo 显著落后 GPT-4o（16+ points）**

GPT-4o 比 GPT-4Turbo 强很多，这个 gap 比我预期的大。

**4. Open models 很差**

最好的 FullCodeRefl + LLaMA3 在 Test-C 上只有7.0 TGC。DeepSeek 更差。CodeAct 和 ToolLLaMA 直接0分——它们的 specialized narrow-domain training 跟 AppWorld 的复杂度不匹配。

### Oracle API 实验

用 validation solution 中的 Oracle APIs 替换 predicted APIs：

| Method | Test-N Pred. | Test-N Oracle | Test-C Pred. | Test-C Oracle |
|--------|-------------|--------------|-------------|--------------|
| ReAct | 48.8 | 54.8 | 30.2 | 35.2 |
| FullCodeRefl | 33.9 | 41.7 | 19.2 | 29.0 |

API retrieval 提升6-10 points，但远没解决。**主要 difficulty 不在 retrieve relevant APIs，而在 compose them in complex code, extract info from outputs, adapt based on errors**。

### Hardness 分析

GPT-4o 在所有 hardness indicators 上都随 difficulty 下降：
- Difficulty 1→3: ReAct 从58.3→21.0 TGC
- 60+ lines of code 的 tasks: TGC < 20
- 更多 unique APIs: performance 显著下降

### 错误分析

手动分析 GPT-4o 的常见 failures：

**a) Does not Interact**: agent hallucinates 而不是与环境交互。比如要给 roommate 发邮件，不去查 Phone app 找 email，直接编一个 email address。

**b) API Understanding & Usage**: hallucinate fields in arguments 或 output schemas。比如 `song_id` 字段不存在但 agent 当它存在。或者用错 API：用 `sent_payment_requests` 而不是 `received_payment_requests`。

**c) Instruction following**: 只 partial follow instruction。比如让它只 approve roommates 的 requests，它 approve all requests。

**d) Commonsense errors**: 混淆 song added to playlist 的 date 和 song release date。

**e) State tracking**: 忘记 current state 和 previous actions，重复工作，耗尽 budget。

## 我的 Intuition

### 1. Interactive coding 的真正难点在 interaction，不在 code generation

Oracle API 实验证明，即使把 perfect API list 给 GPT-4o，TGC 也只提升6-10 points。真正难的是：
- 从 environment 的 unstructured outputs 里 extract needed info
- 看 output structure 才知道下一步怎么做（format discovery）
- handle unexpected errors and hurdles
- 在长 trajectory 里 maintain state without forgetting

这些都是"与世界交互"的能力，不是"写代码"的能力。当前 LLM 在写 code snippet 上已经很强了，但在"看着世界反应，调整下一步"上还很弱。

### 2. Evaluation methodology 决定 benchmark 质量

State-based programmatic evaluation with database diff 是关键创新。它允许 many valid solution paths 同时严格检查 collateral damage。这比 process-based comparison（用 LLM 或 human judge 对比 reference solution）robust 得多。

Hash-based diff 让这个 approach 计算上 tractable（<0.6s/task）。没有这个加速，750个 task × 多个 model × 多个 method 的实验规模根本不可行。

### 3. Contrast sets 暴露 consistency 问题

SGC 比 TGC 低30-50% 是最重要的发现之一。这说明 models 可以在 isolated tasks 上成功，但在 same scenario 的 similar variants 上失败——**局部成功不等于全局可靠**。

如果你要 deploy 一个 agent 帮你干日常 task，你不关心 peak performance（"它能不能做到"），你关心的是 consistency（"它每次都能做到吗"）。SGC 是个更好的实用 metric。

### 4. Strong interaction requirement 是试金石

能一次性写代码解决的 task（weak interaction）和必须与环境交互才能推进的 task（strong interaction）有质的区别。后者要求：
- **Information foraging**: 从 free-form text 里找 info
- **Format discovery**: 看 output structure 才知道下一步
- **Error recovery**: handle unexpected hurdles
- **Adaptive planning**: 根据 intermediate results 调整

AppWorld 的 harder levels 大部分是 strong interaction。这预示着未来 agent research 应该 focus 在这上面，而不是 API retrieval 或 single-shot code generation。

### 5. Engineering quality 是 benchmark 可信度的基础

100K+ lines of code over 14 months，由 experienced researchers（不是 crowdworkers 或 LLMs）写。1780 unit tests with 98% code coverage。FastAPI auto-generate 部分 documentation 避免人为错误。

这些细节决定了 benchmark 能否承受 agents 的 arbitrary exploration。如果 API 实现有 bug，evaluation 结果就不可信。AppWorld 在这方面下了重本，这也是它能成为可靠 benchmark 的基础。

### 6. 当前 SOTA 离实用还很远

GPT-4o 在 normal tasks 上 ~49% TGC，在 challenge tasks 上 ~30%。这还是在 controlled sandbox 里，真实世界只会更难。这意味着：
- 我们离"让 agent 帮你干日常 digital task"还有很长的路
- 改进空间巨大，这个 benchmark 会推动 research
- consistency、state tracking、error recovery 是关键 open problems

### 7. 对未来的 hint

AppWorld Engine 是 modular 和 extensible 的。作者提到几个 future direction：
- 加 UI-based control（有些 app 只能通过 UI 操作）
- Multi-agent tasks（一个 supervisor 的 assistant 要 coordinate 另一个 supervisor 的 assistant，比如约会议）
- Privacy 和 safety risks 的 sandbox 研究

这些方向都很有意思。特别是 multi-agent——很多真实 task 是 collaborative 的，现在的 single-assistant 设定限制了这个维度。

## 参考链接

- Paper: https://arxiv.org/abs/2407.18901
- GitHub: https://github.com/stonybrooknlp/appworld
- 项目主页: https://appworld.dev/
- ReAct paper: https://arxiv.org/abs/2210.03629
- Reflexion paper: https://arxiv.org/abs/2303.11366
- ToolLLM paper: https://arxiv.org/abs/2307.16789
- CodeAct paper: https://arxiv.org/abs/2402.01030
- SWE-Bench: https://arxiv.org/abs/2310.06770
- WebArena: https://arxiv.org/abs/2307.13854
- FastAPI: https://fastapi.tiangolo.com/
- freezegun: https://github.com/spulec/freezegun
- gVisor: https://gvisor.dev/
- SQLite FTS: https://www.sqlite.org/fts5.html

---

# AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents

## 核心动机与定位

这篇paper来自Stony Brook University和Allen Institute for AI，核心想解决的问题是：现有tool-use benchmarks (如ToolBench, API-Bank, Gorilla等) 都太"浅"了——它们只需要1-4个API call的线性序列，而真实的daily digital task自动化需要agent写出**rich code with complex control flow**，并且要**iteratively interact with environment**。

考虑一个直观例子：用户说"play my Spotify playlist with enough songs for today's workout, the workout plan is in Simple Note"。这个task理论上**无法一次性写代码解决**，因为agent必须先去Simple Note里读workout note，看到它的free-form text结构（哪天锻炼多久），才能知道today的workout duration，然后再去Spotify iterate playlists计算每个playlist的总时长，最后选一个>= workout duration的来播放。这就是paper定义的**strong interaction requirement**——必须通过language understanding step与环境交互才能推进。

## AppWorld Engine架构

### 整体设计

AppWorld Engine是60K lines of code的高质量execution environment，包含：

- **9个real-world apps**: Gmail, Venmo, Amazon, Spotify, Phone, Simple Note, Splitwise, File System, Calendar
- **2个helper apps**: ApiDocs (动态查API文档), Supervisor (提供task assigner的信息如addresses, payment cards, account passwords)
- **457 APIs** (avg 50 per app), 1470 arguments
- **101 database tables**, 726 columns, ~370K rows

关键设计决策是用**SQLite + FastAPI TestClient**实现"serverless"架构——所有web app server、database、search engine、client全部跑在**同一个Python process**里。这意味着做实验时不需要manage multiple processes (start/stop/restart servers)，加载一个新task只需要<0.5s。这对大规模benchmarking极其重要。

### API实现的真实性

API实现遵循REST design principles，模拟真实app的feature set，例如：
- **Consistent state changes**: place_order会自动clear cart
- **Cross-app interactions**: 下单会自动发email confirmation
- **Authentication**: 需要access token，防止unauthorized access
- **Pagination**: search results分页返回
- **Informative errors**: 提供结构化error messages

### 可控性与可复现性

对每个task $T_i$，环境关联一个specific starting state：
- **Task DB** $D_i^s$: Base DB的一个copy，经过task-specific modifications
- **Current timestamp** $t_i^s$: frozen date and time (用freezegun库)

这个设计允许：(1) reset到identical starting state保证reproducibility; (2) 写programmatic evaluation suites直接查DB state。

### 数据填充方法

直接用SQL populate 101张表太易错。他们用了一个procedural data population方法：
1. 创建106个fictitious people (ages 19-60)，有home/work addresses和app accounts
2. 建立relationships: friends, family, coworkers, roommates等
3. 用rigorously tested APIs "in the past" 来populate每个person每个app的活动（如Venmo上两年的payment history）
4. 确保consistency: house rent split among roommates但不在coworkers之间

对于precise semantics重要的text entries (如workout note内容)手动写；对于amount/message这类用ChatGPT生成并human-in-the-loop review。

## AppWorld Benchmark构建

### Task Scenario与Task Generator

每个scenario是一个blueprint，例如：
> "I like the last {last-color} {apparel} I bought on Amazon, repurchase the same in that size. Prefer {preferred-color}, if available; otherwise, go with the same color."

通过varying placeholder values和starting states，每个scenario实例化出3个tasks。总共**250 scenarios × 3 = 750 tasks**，分为Train(105), Dev(60), Test-N(168), Test-C(417)。

### 四个关键属性

每个task必须满足：

**1. Is Well-Defined**: task的pre-suppositions在DB state中成立。例如"repurchase last T-shirt"要求supervisor确实有past T-shirt orders且latest order在给定colors里可用。

**2. Has Distractors**: 添加大量task-relevant distractors。例如supervisor有multiple past T-shirt orders in different colors and sizes，迫使agent必须reasoning才能找到correct order。

**3. Has realistic hurdles**: 添加natural hurdles测试agent处理deviation的能力。例如default payment card expired，agent必须try another。

**4. Forms contrast set**: 同一scenario的不同task覆盖diverse conditions形成contrast set。例如一个task里T-shirt在preferred color可用，另一个只在original color可用。

### Setup程序的数学形式

每个instantiated task $T_i$包含三元组：
- **Task input**: Supervisor $S_i$ + instruction $I_i$
- **Environment State**: $(D_i^s, t_i^s)$ — starting DB state和timestamp
- **Evaluation Data**: expected values $E_i$ — 反映correct solution应该导致的DB changes

### Evaluation: State-based Programmatic Testing

这是这篇paper的核心创新之一。复杂task有many valid solution paths，且collateral damage可以以many ways发生。他们提出**state-based**而非**process-based** evaluation。

核心idea：检查final DB state $D_i^f$ 是否属于valid gold states集合 $\mathcal{D}_i^*$。但直接枚举 $\mathcal{D}_i^*$ 不可行，所以用**database diff** $D_i^\Delta$ 高效检查。

定义两个change sets：
- $C_i^{\text{expect}}$: 必须发生的changes (e.g., exactly 1 order placed, has only 1 product of type T-shirt)
- $C_i^{\text{allow}}$: 允许但非mandatory的changes (e.g., 下单后cart可以恢复原状或留空)

Evaluation条件：
$$C_i^{\text{expect}} \subseteq D_i^\Delta \quad \text{AND} \quad D_i^\Delta \subseteq C_i^{\text{expect}} \cup C_i^{\text{allow}}$$

第一个不等式确保所有expected changes都发生了；第二个确保没有unexpected changes (collateral damage)。

### Hash-based Database Diff加速

Naively逐行比较101张表360K rows极慢。他们的backend额外维护：
- **record_hash**: 每行内容的hash (排除ID列)
- **table counter**: 每张表的简单counter，row hash变化时increment

这样diff可以hierarchical进行：
1. **Tables**: counter没变的table直接跳过
2. **Rows**: 同ID同record_hash的row没变；end有start没有的是added；start有end没有的是deleted
3. **Columns**: 只对changed rows做column-by-column比较

这使得evaluation平均<0.6s per task。

### 两个核心Metrics

- **Task Goal Completion (TGC)**: 通过所有evaluation tests的task百分比
- **Scenario Goal Completion (SGC)**: 同一scenario的所有tasks都通过的scenario百分比——这是一种consistency metric，测试agent能否在varying requirements下reliably achieve goal

### Validation Solutions

为每个task写fully programmatic solution做end-to-end testing。对于strong interaction requirement的task，他们用minimal internal knowledge绕过interaction hurdle (例如用regex提取workout duration，因为知道note的内部结构)。这确保tasks确实solvable且evaluation suites正确。

## Dataset统计

| 指标 | Test-N avg | Test-N max | Test-C avg | Test-C max |
|------|-----------|-----------|-----------|-----------|
| Num. Apps | 1.51 | 3 | 2.01 | 6 |
| Num. Unique APIs | 8.2 | 17 | 10.5 | 26 |
| Num. API calls | 42.5 | 244 | 46.8 | 649 |
| Num. Solution Code Lines | 41.3 | 134 | 56.9 | 128 |
| Num. Evaluation Tests | 5.9 | 19 | 8.0 | 24 |

Test-C ("challenge" set)要求至少用一个designated unseen app (Amazon或Gmail)，防止model靠memorized actions作弊，强迫它真正读API documentation。Test-N与Train/Dev同分布。

## 实验方法

### 6种方法

**Decision-making methods** (用ApiDocs动态探索):
1. **ReAct**: Thought→Code循环，最多100次LLM calls
2. **Plan & Execute (PlanExec)**: 先生成high-level plan，再每步用ReAct-styled agent执行

**Code-based methods** (用API predictor预先retrieve relevant APIs):
3. **FullCodeRefl**: 一次生成全部代码，失败后用Reflexion-style反思retry，最多5次
4. **IPFunCall**: 用parallel function calling ability，每turn生成batch of independent API calls，最多15 turns
5. **ToolLLaMA**: fine-tuned LLaMA生成API call sequence
6. **CodeAct**: Mistral-7B fine-tuned on CodeActInstruct

### Context Truncation Strategy

ReAct和PlanExec的trajectory会变得很长（JSON outputs是主要contributor）。他们的truncation策略：
- 当trajectory超过20000 chars时，保留last two steps的outputs
- 从oldest开始用`[NOT SHOWN FOR BREVITY]`替换outputs
- 如果还不够，从oldest开始移除整个(thought, action, observation) blocks
- 添加`[TRIMMED HISTORY]`标记

### API Predictor

由于full documentation ~142K tokens太大，用一个LLM prompted预测relevant APIs (最多16个)。Table 4显示GPT-4o的API retrieval F1在Test-N上87，Test-C上71——相当高，说明**API retrieval不是主要bottleneck**。

## 主要结果

| Base LLM | Method | Test-N TGC | Test-N SGC | Test-C TGC | Test-C SGC |
|----------|--------|-----------|-----------|-----------|-----------|
| GPT-4o | ReAct | **48.8** | **32.1** | **30.2** | **13.0** |
| GPT-4o | PlanExec | 44.6 | 23.2 | 19.7 | 7.9 |
| GPT-4o | FullCodeRefl | 33.9 | 26.8 | 19.2 | 12.2 |
| GPT-4o | IPFunCall | 32.1 | 16.1 | 18.0 | 10.1 |
| GPT-4Turbo | ReAct | 32.7 | 12.5 | 17.5 | 5.8 |
| LLaMA3 | FullCodeRefl | 24.4 | 17.9 | 7.0 | 4.3 |
| DeepSeek | FullCodeRefl | 13.1 | 8.9 | 5.8 | 2.9 |
| Mistral-7B | CodeAct | 0.0 | 0.0 | 0.0 | 0.0 |
| LLaMA | ToolLLaMA | 0.0 | 0.0 | 0.0 | 0.0 |

关键观察：
1. **最强的GPT-4o + ReAct也只解决~49%的normal tasks和~30%的challenge tasks**
2. **TGC到SGC有30-50%的drop**——models不能consistently完成同一scenario的所有variants
3. GPT-4Turbo显著落后GPT-4o (16+ points差距)
4. Open models (LLaMA3, DeepSeek)更差，最好的FullCodeRefl+LLaMA3在Test-C只有7.0 TGC
5. **CodeAct和ToolLLaMA在所有tasks上都fail**——likely因为它们的specialized narrow-domain training与AppWorld的复杂度不匹配

### Oracle API实验

用validation solution中的Oracle APIs替换predicted APIs：
- ReAct在Test-N上从48.8→54.8 (+6.0)
- ReAct在Test-C上从30.2→35.2 (+5.0)
- FullCodeRefl在Test-C上从19.2→29.0 (+9.8)

这证实**主要difficulty不来自API retrieval，而来自在complex code中interactive使用APIs并适应errors**。

### Hardness分析

Figure 4显示GPT-4o在所有hardness indicators上都随difficulty增加而performance下降：
- Difficulty level 1→3: ReAct从58.3→21.0 TGC
- 60+ lines of code的tasks: TGC < 20
- 更多unique APIs的tasks: performance显著下降

### 错误分析

手动分析GPT-4o的常见failures：
- **a) Does not Interact**: agent hallucinates而不是与环境交互获取信息 (如roommate的email address)
- **b) API Understanding & Usage**: hallucinate fields in arguments或output schemas; 用错API (如sent_payment_requests vs received_payment_requests)
- **c) Instruction following**: 只partial follow instruction (如approve all requests而不是只approve roommates的)
- **d) Commonsense errors**: 混淆song added to playlist的date与release date
- **e) State tracking**: 忘记current state和previous actions，重复工作耗尽budget

## 技术细节深挖

### Execution Shell设计

IPython-based stateful execution，类似Jupyter Notebook：
- **Stateful**: 变量在code blocks间可复用
- **Function or REST**: 支持直接function calls (`apis.spotify.login()`) 和REST calls (`request.post("/spotify/auth/token", {...})`)
- **Safe execution**: 禁用os.write, shutil.rmtree, subprocess.call等; 提供Docker + gVisor runtime
- **Error stacktraces**: 对Python errors和failed API requests都提供informative traces
- **Frozen datetime**: datetime.now()返回task设置的timestamp

### Task Generator代码结构

以workout playlist task (Listing 1-3)为例：

**Setup** (`_setup`方法):
```python
data.datetime = DateTime.now().next(go_to_next_day).set_time("morning")
# 确保supervisor有>=MIN_PLAYLISTS个playlists
# 找到titled "Weekly Workout Plan"的note
# 从note.data["day_plans"]解析workout plans
# 确保至少4个distinct workout durations
# 获取current day的workout_duration_mins
# 确保它不是min或max (避免trivial case)
# 清空所有playlists的songs
# 随机选一个playlist作为long_enough_playlist，填入songs直到duration >= workout_duration_mins
# 其余playlists填入songs使其duration在[workout_duration_mins - 15, workout_duration_mins)
# pause music player, clear queue
# rename note用rolling_get选择标题
```

**Validation Solution** (`solution`函数):
```python
# 从supervisor获取profile和passwords
# login到simple_note, search "workout" notes
# 找到title含"workout"或"exercise"的note
# show_note获取content, parse出current day的duration_mins
# login到spotify, paginated获取所有playlists
# for each playlist, sum所有songs的duration
# 找到第一个total_duration >= workout_duration_mins的playlist, play它
# complete_task
```

**Evaluation** (`evaluate`函数):
```python
# assert task_completed status == "success"
# assert answer匹配ground_truth
# assert only spotify.MusicPlayer model changed (无collateral damage)
# assert music_player.is_playing is truthy
# assert queue_songs总duration >= workout_duration_mins
# assert queue_song_ids == private_data.long_enough_playlist_song_ids
```

### Prompt设计细节

**ReAct prompt** (Listing 7):
- One-shot demonstration (因为trajectory长)
- 教agent用ApiDocs探索: `show_app_descriptions`, `show_api_descriptions`, `show_api_doc`
- 7条key instructions: small chunks, pagination looping, always check API specs before calling, etc.

**FullCodeRefl prompt** (Listing 5):
- 3-shot demonstrations (用validation solutions)
- 提供可用imports列表 (builtins, datetime, json, pendulum等)
- 明确APIs allowed to use列表
- 失败时 (Listing 6)显示error stacktrace要求reflect并retry

**PlanExec** (Listing 8-9):
- Planner: 3-shot生成high-level plan, 每步clear且independently executable
- Executor: 5-shot ReAct-styled, 每步给定previously executed steps的history和code

### Cost分析

GPT-4o在Test-N上的per-example cost:
- ReAct: $0.7
- PlanExec: $1.33
- IPFunCall: $0.33
- FullCodeRefl: $0.02

GPT-4Turbo约2x cost，GPT-4约6x cost。整个实验约$10K。

## 我的Intuition构建

读完这篇paper，我build的intuition是：

**1. Interactive coding agent的真正难点不在"知道有哪些API"**。Oracle API实验证明，即使把perfect API list给GPT-4o，TGC也只提升~6-10 points。真正难的是：(a) 在rich code中compose这些API calls with control flow; (b) 从environment的unstructured outputs中extract needed information; (c) adapt behavior based on errors and intermediate results; (d) maintain state across long trajectories without forgetting。

**2. Evaluation methodology决定了benchmark的quality**。State-based programmatic evaluation with database diff是关键创新——它允许many valid solution paths同时严格检查collateral damage。这比process-based comparison (用LLM或human judge对比reference solution)robust得多，尤其是对于complex tasks。Hash-based diff让这个approach在计算上tractable。

**3. Contrast sets暴露了LLM的consistency问题**。SGC比TGC低30-50%说明models可以在某些variants上succeed但在其他similar variants上fail——这是"局部成功但全局不可靠"的信号。要实现真正autonomous的daily task automation，consistency比peak performance更重要。

**4. Strong interaction requirement是真正的试金石**。能一次性写代码解决的task (weak interaction)和必须与环境交互才能推进的task (strong interaction)有质的区别。后者要求agent具备: information foraging (从free-form text中找info), format discovery (看output structure才知道下一步怎么做), error recovery (handle unexpected hurdles)。

**5. Engineering quality是benchmark可信度的基础**。100K+ lines of code over 14 months由experienced researchers (不是crowdworkers或LLMs)写，1780 unit tests with 98% code coverage，FastAPI auto-generate部分documentation——这些细节决定了benchmark能否承受agents的arbitrary exploration。如果API实现有bug，evaluation结果就不可信。

**6. 当前SOTA离实用还很远**。GPT-4o在normal tasks上只有~49% TGC，在challenge tasks上~30%——这意味着如果你让GPT-4o帮你做日常digital tasks，它有超过一半概率会失败或造成collateral damage。而且这是在controlled sandbox环境里，真实世界的complexity更高。

## 参考链接

- Paper PDF: https://arxiv.org/abs/2407.18901
- AppWorld GitHub: https://github.com/stonybrooknlp/appworld
- AppWorld项目主页: https://appworld.dev/
- ReAct paper: https://arxiv.org/abs/2210.03629
- Reflexion paper: https://arxiv.org/abs/2303.11366
- ToolLLM paper: https://arxiv.org/abs/2307.16789
- CodeAct paper: https://arxiv.org/abs/2402.01030
- SWE-Bench: https://arxiv.org/abs/2310.06770
- WebArena: https://arxiv.org/abs/2307.13854
- FastAPI: https://fastapi.tiangolo.com/
- freezegun: https://github.com/spulec/freezegun
- gVisor: https://gvisor.dev/
- SQLite FTS: https://www.sqlite.org/fts5.html

如果你想进一步深挖某个方面——比如task generator的具体设计模式、evaluation assertion的写法、某个specific scenario的contrast set构造、或者不同prompting strategy的ablation——我可以展开讲。
