---
source_pdf: Dive into Claude Code_ The Design Space of Today's and Future AI Agent
  Systems.pdf
paper_sha256: a4c3c92070557d693125850ab40d8249dbb3967988c919e95a6f46caa4ab62b9
processed_at: '2026-08-03T22:44:32-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话概括

这篇 paper 把 Claude Code 的源码扒开看了一遍，发现一个反直觉的事实：**这个所谓的 "AI agent" 里，真正做 AI 决策的代码只占 1.6%，剩下 98.4% 全是围绕 AI 的脚手架、安检、保险丝、归档系统**。换句话说，你用的不是一个 AI，你用的是一套很复杂的"让 AI 别闯祸"的操作系统。

---

## 核心赌注：Model 越聪明，越不该用框架捆它

主流做 agent 的两个流派：

- **LangGraph 那派**：画一张 state graph，每个 node 是一个 decision，typed edges 连起来，model 只能在图里走。像把 AI 关在迷宫里。
- **Devin 那派**：搞一个 multi-step planner，先规划再执行，model 被一个"项目经理"盯着。

Claude Code 走的是反方向：**不画图、不规划、不约束，就给 model 一个 while 循环 + 一堆 tools，让它自己决定干嘛**。所有的工程量都花在"如果它搞砸了怎么救"上——context 太长怎么压、permission 怎么把关、subagent 怎么隔离、session 怎么 resume。

这个 bet 背后的逻辑很简单：**model 每一代都在变强，你花力气写的 planner / state graph 第二年就过时了；但 context management、safety layer、recovery mechanism 这些 deterministic infrastructure 不会过时，反而越攒越值钱**。

类比一下：与其给一个越来越聪明的员工配一个事事汇报的项目经理，不如给他配一个好的行政系统、法务审核、IT 支持——他自己会做决策，你要做的是确保他闯祸的代价可控。

---

## 系统心脏：一个 while 循环

整个 Claude Code 的核心就是 `query.ts` 里的一个 `queryLoop()`，是个 async generator。每一 turn 走 9 步：

1. 读 settings
2. 初始化 mutable state
3. 拼装 context
4. 跑 5 个 compaction shaper（下面细讲）
5. 调 model
6. 如果 model 返回了 tool_use 就 dispatch
7. 过 permission 关
8. 执行 tool，结果塞回 history
9. 没 tool_use 就停

**所有入口都 converge 到这同一个 loop**——不管是 interactive CLI、headless `claude -p`、Agent SDK、还是 IDE plugin。只有 UI 层不一样，业务逻辑全共享。这一点很重要，意味着 Claude Code 不会因为 surface 不同而行为分裂。

---

## Context 管理：五个清洁工按顺序进场

这是整个 paper 最值得琢磨的子系统。Model 的 context window 是 binding constraint（200K 或 1M tokens），所以每次调 model 之前都要"打扫"。

Claude Code 没用简单粗暴的"扔掉最老的 message"，而是**五个清洁工按 cost 从低到高排队进场**：

### 1. Budget Reduction（永远开）
每个 tool 的输出有个 size 上限。比如你 `npm test` 输出 50KB，超过阈值就被替换成一个 reference 指针，原文写磁盘。

**直觉**：就好比你书桌上放不下大文件，把厚的报告塞抽屉，桌上只留个便签说"报告在抽屉第二格"。

### 2. Snip（`HISTORY_SNIP` flag）
轻量级地砍掉比较老的历史片段。不动内容，只动时间维度。

**直觉**：清理桌面，把三个月前的便签扔了，本周的还留着。

### 3. Microcompact（`CACHED_MICROCOMPACT` flag）
Fine-grained 压缩，time-based 必跑，cache-aware 可选。Cache-aware 模式会等 API 返回后用真实的 `cache_deleted_input_tokens` 而非估计值来决定怎么压。

**直觉**：看你哪些缓存还能复用，尽量不动那些能省钱的 cache 边界。

### 4. Context Collapse（`CONTEXT_COLLAPSE` flag）
**这是最 elegant 的一层**。它不修改存储的历史，只在做 model call 时把"老对话"投影成一个 summary view 喂给 model。原始历史完整保留在磁盘上。

源码注释原话：*"a read-time projection over the REPL's full history. Summary messages live in the collapse store, not the REPL array."*

**直觉**：你有一本完整日记，但每天早上你只看一份"本月摘要"。摘要不是把日记撕了重写，而是另一张纸。原始日记还在，要查随时能查。

### 5. Auto-compact（默认开，可关）
前 4 层都搞不定才上这个——**直接调 model 让它生成一个 summary**。Pre-compact hooks 先 fire，然后 `getCompactPrompt()` 构造 prompt，model 返回压缩版本，`buildPostCompactMessages()` 拼出新 messages 数组。

**直觉**：实在塞不下了，找个人帮你把过去三个月的事总结成一页 A4 纸。

### 为什么要五层而不是一层

**No single compaction strategy addresses all types of context pressure**：
- Budget reduction 处理 "一个 tool 输出太大"
- Snip 处理 "历史太老"
- Microcompact 处理 "cache 边界"
- Context collapse 处理 "历史非常长但我不想真删"
- Auto-compact 处理 "什么都压不住了"

每层 cost-benefit 不同，便宜的先跑，贵的最后跑。这叫 **lazy-degradation principle**。

代价是：五层互相作用，加上多个 feature flag，用户很难预测到底什么时候发生了什么。Auto-compact 在 transcript 里有可见 summary，microcompact 有 boundary marker，但 context collapse **完全静默**——这是一个 transparency vs. efficiency 的明确 trade-off。

参考 MemGPT 的 LLM-as-OS 思路：https://arxiv.org/abs/2310.08560

---

## Permission：机场安检式的分层信任

Anthropic 自己发现一个尴尬数据：**用户批准了 93% 的 permission prompts** (Hughes 2026)。也就是说，弹窗根本拦不住人——大家都是无脑点 yes。

如果继续加弹窗，是治标不治本。Claude Code 的回应是**重构 problem**：

> 与其每次都问，不如定义一个 trust boundary，在 boundary 内 agent 自由行动，boundary 外的事才找人。

这就像机场安检：VIP 通道刷脸就过（auto classifier），常旅客走快速通道（acceptEdits），第一次来的走人工窗口（default mode），嫌疑人直接拦（deny rules）。**不是每个旅客都过同一道关，而是按 risk 分流**。

### 七个 mode 的 trust spectrum

```
plan → default → acceptEdits → auto → dontAsk → bypassPermissions
（高 caution）                                  （低 caution）
```

加上一个 internal 的 `bubble` mode 给 subagent 用。

### Deny-First Rule Evaluation 的反直觉设计

**Broad deny ("deny all shell") 不能被 narrow allow ("allow npm test") override**，即使 allow 更 specific。

这违反了常见的 "most-specific-match" 原则，但有意的——safety-conservative。宁可错杀，不可错放。

### 七层独立 Defense-in-Depth

任何一层都能 block 一个 request：

1. **Tool pre-filtering** — blanket-denied tools 直接从 model 视野里消失，model 根本不能 attempt
2. **Deny-first rule evaluation**
3. **Permission mode constraints**
4. **Auto-mode ML classifier** — 一个独立 LLM call 评估 tool safety
5. **Shell sandboxing** — 独立的 filesystem + network isolation
6. **Not restoring permissions on resume** — session-scoped permissions 不持久化
7. **Hook-based interception** — PreToolUse hooks 能修改 permission decisions

**Independence assumption**：如果一层 fail，其他层 catch。但 paper 指出一个 **structural 隐患**——这些层 share common performance constraints：

- Auto-mode classifier 是独立 LLM call，有 token cost
- `bashSecurity.ts` 做 AST parsing，有 latency
- Deny-first rule evaluation 也依赖 command parsing

**Adversa.ai 报告过一个真实案例**：超过 50 个 subcommands 的命令会 fallback 到单个 generic prompt，跳过 per-subcommand 检查，因为 per-subcommand parsing 会 freeze UI。

这是 defense-in-depth 的结构性弱点：**当 layer 共享 failure mode 时，"多层"退化成"一层"**。

参考：https://adversa.ai/blog/claude-code-security-bypass-deny-rules-disabled/

---

## 四个 Extension Mechanism：Cost 不同的四个口

为什么是 4 个而不是 1 个？因为 **不同的扩展需求 context cost 差好几个数量级**，单一机制会强制 trade-off。

| Mechanism | 干嘛的 | Context Cost | 像什么 |
|-----------|--------|--------------|--------|
| **Hooks** | 在 tool 执行前后拦截、改写、注入 | **Zero** | 公司保安，平时不占工位 |
| **Skills** | 注入 domain-specific 指令 | **Low** | 说明书，只放目录在桌上 |
| **Plugins** | 打包分发的多组件 bundle | **Medium** | 一整套外包服务包 |
| **MCP servers** | 外部工具集成（数据库、API 等） | **High** | 外包供应商，要谈判要签 contract |

**直觉**：你不能用同一个 mechanism 既处理"我想在每次 commit 前跑个 lint"（hook，零 context）又处理"我想让 agent 能查 Postgres"（MCP，要 tool schema 进 context）。前者用 MCP 是杀鸡用牛刀，后者用 hook 又做不到。

### Agent Loop 的三个 Injection Point

```julia
while not stopped:
    # a assemble(): 决定 model 看到什么
    context = assemble(system_prompt, tool_schemas, history, hook_additions)
    
    # b model(): 决定 model 能碰到什么
    action = model(context, tools)  # flat tool pool
    
    # c execute(): 决定动作怎么跑
    if not permitted(action): continue
    action = run_pre_tool_hooks(action)
    result = execute(action)
    result = run_post_tool_hooks(result)
```

- **assemble** 里塞：CLAUDE.md、skill descriptions、MCP resources
- **model** 里塞：built-in tools + MCP tools + SkillTool + AgentTool 都在一个 flat pool
- **execute** 里塞：permission + hooks + sandbox

四个 extension mechanism 分别插在这三个点上的不同位置。

---

## CLAUDE.md：为什么是 user context 而不是 system prompt

四级 hierarchy：

```
/etc/claude-code/CLAUDE.md      (managed, OS policy)
~/.claude/CLAUDE.md             (user, 全局私有)
<project>/CLAUDE.md             (project, 进 git)
<project>/CLAUDE.local.md       (local, gitignore)
```

加载顺序是 **reverse order of priority**——后加载的 file 得到更多 model attention。File discovery 从 CWD 向上 traverse 到 root。

**关键 architectural choice**：CLAUDE.md 内容被作为 **user context message** deliver 给 model，而不是塞进 system prompt。

这意味着什么？**Model 遵守 CLAUDE.md 指令是 probabilistic 的，不是 guaranteed**。如果你在 CLAUDE.md 里写"永远不要删数据库"，model 可能还是删了。

那 safety 怎么保证？靠 **permission rules 的 deterministic enforcement**。Permission rules 是代码逻辑，deny 就是 deny，model 绕不过去。

**这是 guidance (probabilistic) vs. enforcement (deterministic) 的明确分离**。CLAUDE.md 是"建议"，permission 是"硬规矩"。

### Memory Retrieval: 不用 embedding

Claude Code **不用 vector similarity index**。它用 LLM 扫一遍 memory-file headers，选 up to 5 个 relevant files，以 file granularity surface。

**Trade-off**：
- Embedding-based: 可以 selectively 拿单个 entry，但需要 infra + opaque to user
- LLM scan: transparent、no infra，但 coarse（只能拿 file 不能拿 entry）+ 每次 LLM call 要花钱

Claude Code 选了透明、无 infra 的路。这跟它整体哲学一致：**用户能看到、能改、能 git diff 的东西优先于黑箱性能**。

---

## Subagent：外包给同事，只收 summary

当 Claude 决定 "fix auth test 之前先 explore 一下 auth 模块结构"，它可以 delegate 给一个 subagent。

### Isolation 用 Git Worktree 而不是 Docker

SWE-Agent / OpenHands 用 Docker container isolation——强 resource boundary，但要 container infra。

Claude Code 用 **Git worktree**——给 subagent 一个临时 git worktree，它有自己的 working copy 可以乱改，不影响 parent 的 working tree。

**这是个非常 elegant 的工程选择**：零依赖，用 Git 自带机制，不需要 container orchestration。

### Sidechain Transcripts

每个 subagent 写自己的 `.jsonl` + `.meta.json` 到独立文件。**关键：subagent 的完整 history 永远不进入 parent 的 context window**，只有 final response text + metadata return。

**直觉**：你把活儿外包给同事，同事干完只发你一封邮件说"搞定了，结论是 X"。你不需要看他工作过程中查了多少资料、试了多少次。

这是 "context as bottleneck" principle 在 multi-agent 场景下的延伸。

**成本数据**：Claude Code agent teams 消耗约 **7×** 一个标准 session 的 tokens (Anthropic 2025b)。所以 summary-only return 在 isolated contexts 下就更关键。

### SkillTool vs. AgentTool 的本质区别

- **SkillTool**: 把指令塞进当前 context，model 在同一个窗口里继续干
- **AgentTool**: 开一个新 context 窗口，subagent 在里面独立干完，只把 summary 回来

一个共享 context，一个开新 context。这是两个完全不同的东西，但都作为 meta-tool 出现在 flat tool pool 里。

---

## Session 持久化：只追加，不修改

Session transcripts 是 **mostly append-only JSONL**。Compaction 也不修改已写的 lines，只 append 新的 boundary marker + summary。

**为什么这样设计**：

1. **Auditability** — 每个事件都能查到，不会丢
2. **Resume/fork** — 重建 session 只需 replay transcript
3. **可 git diff** — 用户能看到、能审

代价是：rich query（"show me all tool calls that modified file X across sessions"）需要 post-hoc reconstruction，不是直接 SQL lookup。

### 一个关键 Safety Choice: Resume 不恢复 Permissions

`--resume` 重建 conversation，但 **session-scoped permissions 不恢复**——你必须重新批准。

理由：session 是 isolated trust domain。如果恢复之前 granted 的 permissions，会把 stale trust decision 带到 changed context 里。

**系统选 re-granting over implicit persistence**，接受 user friction 以维持 safety invariant。

**直觉**：每次新开一个工作 session，权限都重新审一遍，不沿用上次的"免检章"。麻烦但安全。

---

## OpenClaw 对比：同一组问题，不同的答案

OpenClaw 是个 independent open-source agent gateway——persistent WebSocket daemon（默认 port 18789），连接 ~24 个 messaging surface（WhatsApp、Telegram、Slack、Discord、Signal 等）。

**两者在六个 dimensions 上做 opposite bets**：

| Dimension | Claude Code | OpenClaw |
|-----------|-------------|----------|
| Scope | 临时 CLI 进程 | 持久 gateway daemon |
| Trust | per-action deny-first + 7 modes + classifier | 单一 trusted operator + 周边访问控制 |
| Center | agent loop 是系统中心 | gateway control plane 是中心，agent loop 嵌进去当组件 |
| Extension | 4 mechanisms 改一个 context | 12 capability types 改 gateway surface |
| Memory | CLAUDE.md 4 级 + 5-layer compaction | workspace bootstrap files + 独立 memory system + 实验性 dreaming |
| Multi-agent | 任务委派 subagent | 两层：multi-agent routing + sub-agent delegation |

**三个观察**：

1. **Recurring design questions 普适**——where reasoning lives, what safety posture, how manage context, how structure extensibility。这些问题跨 deployment context 稳定存在。
2. **Opposite bets 不是任意的**——都 follow from 不同的 trust model + deployment topology
3. **两系统 composable**——OpenClaw 通过 ACP (Agent Client Protocol) 能 host Claude Code 作为 external coding harness。**Design space 不是 flat taxonomy，是 layered**

**直觉**：Claude Code 像随身秘书，OpenClaw 像公司前台总机。同一个"什么该让 AI 做、什么该审"的问题，秘书场景和前台场景答案天然不同。但前台可以呼叫秘书——两者能组合。

---

## Pre-Trust Initialization：一个 Timing 漏洞

两个独立验证的 CVE 共享 root cause：**project initialization 阶段执行的 code 在 interactive trust dialog 之前运行**。

```
extension loading  →  trust dialog  →  permission enforcement
```

Hooks、MCP server connections、settings file resolution 都在第一阶段执行，**这时 permission system 还没 engage**。这创建一个 structurally privileged phase，safety guarantees 不适用。

Permission pipeline 图画的是 **spatial ordering**（哪些 check 在哪里），但 **不 capture temporal dimension**——每个 mechanism 在 session init 过程中何时 active。

这 refine 了 extensibility-vs-safety tension：**extensivity 创建 attack surface 不仅通过 combinatorial complexity，还通过 initialization ordering**。

参考 CVE-2025-59536: https://research.checkpoint.com/2026/rce-and-api-token-exfiltration-through-claude-code-project-files-cve-2025-59536/

---

## Long-term Capability: 一个细思极恐的副效应

论文引入第六个 concern 作为 **evaluative lens**（不作为 co-equal design value，因为 Anthropic 自己的 stated values 里没 prominent 反映）。

**Empirical evidence**：

- **Becker et al. 2025** RCT（16 个资深开发者，246 任务）：AI tools 让开发者**慢 19%**，尽管 perceived 提升 20%。主观感觉快了，客观慢了。
- **He et al. 2025**（807 repos Cursor adoption）：code complexity **+40.7%**，initial velocity spike 到 month 3 消退到 baseline。早期快，后期慢。
- **Kosmyna et al. 2025** EEG study（54 人）：LLM users 显示 weakened neural connectivity，**AI 移除后仍持续**。脑子结构变了。
- **Liu et al. 2026**（304,000 AI commits, 6,275 repos）：~25% AI-introduced issues 持续到 latest revision，security-related 持续率更高。
- **Rak 2025**：entry-level tech hiring 2023-2024 下降 25%。

**Paradox of supervision**：越依赖 AI，越需要 skills to supervise AI；但越依赖 AI，这些 skills 越萎缩。

**论文的 pivot**：未来系统 could treat sustainability gap as **first-class design problem, not downstream evaluation metric**。

这意思是：不应该等产品做完了再用 metrics 测"用户脑子有没有变笨"，而应该在架构层面就考虑"这个 design 会不会让用户变笨"。

---

## KAIROS: Proactive Architecture 的一个实验

Feature-gated 的 KAIROS 是一个 persistent background agent with tick-based heartbeats：

- 没有待处理 user message 时，系统注入 periodic `<tick>` prompts
- Model 决定 act or sleep
- **Terminal focus awareness**: user 离开时 maximize autonomous action，user 在场时 increase collaboration
- **Economic throttling via SleepTool**: 每次 wake-up costs API call；prompt cache 5 min 后 expire，使 sleep/wake 成 explicit cost optimization

这 address 一个 documented tension (Chen et al. 2025)：
- Proactive AI assistants **+12-18%** task completion
- 但 high frequency 下 user preference 从 80-90% 跌到 47%

KAIROS 通过绑定 proactivity 到 user presence + token economics 解决——**user 不在就主动干活，user 在就收敛**。

这种 binding 不常见于 production agent systems。KAIROS 是否在 production 中 active 未能 confirm。

---

## 三个 Cross-Cutting Pattern

读完六个 subsystem，浮现三个共同 commitment：

### 1. Graduated Layering over Monolithic Mechanism
- Permission: 7 层
- Context: 5 层 + lazy loading + summary-only
- Extension: 4 个 mechanism 不同 context cost

**都是用多层独立机制代替单一集成方案**。Trade: simplicity + debuggability 换 defense in depth。

### 2. Append-Only Favors Auditability over Query Power
- Transcripts: append-only JSONL
- Permissions: 不跨 session 恢复
- Compaction: read-time projection 而非 destructive edit

**都是只追加不修改**。Trade: rich query 需 post-hoc reconstruction。

### 3. Model Judgment within Deterministic Harness
- 1.6% decision logic / 98.4% operational infrastructure
- Model 有完全 latitude 选哪些 tool、什么顺序
- Harness 提供 conditions (tool routing, permission, context, recovery)

**Harness 创造条件让 model 决策好，不约束决策**。Trade: good local decisions 可 produce poor global outcomes 当 bounded context 阻碍 global awareness。

---

## 一个 Final Intuition: Claude Code 作为 LLM-as-OS 的 partial instantiation

Karpathy 2023 年 11 月的 "Intro to LLMs" talk 里 popularize 了 **LLM-as-OS framing**。Claude Code 的 architecture 可以视为这个 framing 的 partial production instantiation：

| OS 概念 | Claude Code 对应 |
|---------|----------------|
| Process scheduler | `queryLoop()` + `StreamingToolExecutor` |
| Virtual memory | 5-layer compaction + read-time projection |
| Filesystem | append-only JSONL + CLAUDE.md hierarchy |
| Syscalls | tool_use protocol |
| Permission system | 7-mode deny-first + classifier + sandbox |
| Process isolation | worktree-based subagent isolation |
| IPC | sidechain transcripts + file locking |
| Device drivers | MCP servers (8 transport variants) |
| Kernel modules | plugins (10 component types) |
| Daemons | hooks (27 event types) |
| User accounts | 4-level CLAUDE.md memory hierarchy |

Martin et al. 2026 "Managed Agents" 把这个 analogy explicit: virtualize session/harness/sandbox 为 independently replaceable interfaces，类比 OS virtualize hardware into processes/files。

Rajasekaran 2026 Harness Design essay: **"the space of interesting harness combinations doesn't shrink as models improve; it moves"**——意思是这个 "OS" 不是 fixed optimum，是 co-evolving snapshot。Model 每强一代，"OS" 该长什么样就变一次。

参考：
- Karpathy talk: https://www.youtube.com/watch?v=zjkBMFhNj_g
- Managed Agents: https://www.anthropic.com/engineering/managed-agents
- Harness Design: https://anthropic.com/engineering/harness-design-long-running-apps

---

## 最后一句话总结

这篇 paper 的真正贡献：**把 "怎么做一个 production coding agent" 从一堆 ad-hoc engineering choices 抽象成一组 recurring design questions**。每个 question 都有 alternative answers，每个 answer 都 encode 一组 trade-offs。

Claude Code 占据 design space 中一个 **coherent design point**——把 model autonomy 推到极端，把 deterministic harness 也推到极端。OpenClaw 占另一个 point——把 gateway control plane 推到极端，把 per-action safety 简化到周边。

未来的 agent 系统 **不需要 converge 到同一个 point**，但需要 consciously navigate 同一个 design space。最该被 first-class 关心的，可能不是 "怎么让 agent 更强"，而是 **"怎么让 agent 不让人类变弱"**——这个 sustainability gap 在当前 architecture 里几乎没有 mechanism 专门 address，是下一个十年的 design frontier。

---

# Dive into Claude Code: Architecture 深度解析

这是 VILA Lab (Mohamed bin Zayed University of Artificial Intelligence) 对 Claude Code v2.1.88 源码做的一次系统性 architecture 解构，作者 Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, Zhiqiang Shen。论文不只是描述一个产品，更尝试抽取 production coding agent 的 design space，再用 OpenClaw 作为对照系来验证这些 design question 的普适性。

GitHub repo: https://github.com/VILA-Lab/Dive-into-Claude-Code

---

## 1. 核心论点：1.6% / 98.4% Ratio

整个分析最有冲击力的一个数字：从源码估计 Claude Code 大约 **1.6% 是 AI decision logic**，**98.4% 是 operational infrastructure**。这呼应了 Anthropic 自己的 design philosophy——"a Unix utility rather than a traditional product"，built from smallest building blocks that are "useful, understandable, and extensible" (Cherny and Wu, Latent Space podcast)。

这与主流 agent engineering 的两个方向形成鲜明对比：

| Approach | 代表系统 | Decision scaffolding 占比 |
|----------|---------|--------------------------|
| **Graph-based orchestration** | LangGraph | 高 (显式 state graph + typed edges) |
| **Planner + executor** | Devin | 高 (multi-step planner) |
| **Container isolation** | SWE-Agent, OpenHands | 中 (Docker sandbox 为主) |
| **Git-as-safety** | Aider | 低 (依赖 VCS rollback) |
| **Minimal scaffolding + maximal harness** | Claude Code | **极低 (~1.6%)** |

Claude Code 的 bet 是：随着 model 能力上升，**约束 model 选择的 framework 反而不如给 model 富 operation environment 收益大**。

Reference: Anthropic "Building Effective Agents" https://www.anthropic.com/research/building-effective-agents

---

## 2. Value → Principle → Architecture 的三层映射

论文构建了一个三层 abstraction：

```
Human Values (5)  →  Design Principles (13)  →  Implementation Choices (concrete source files)
```

### 2.1 五个 Values

1. **Human Decision Authority** — 人保留最终决定权，principal hierarchy: Anthropic → operators → users
2. **Safety, Security, and Privacy** — 即使人在 inattentive 状态也要保护 code/data/infrastructure
3. **Reliable Execution** — 单 turn correctness + long-horizon dependability
4. **Capability Amplification** — 实质放大单位 effort 完成的工作量 (27% 任务是「没有这个工具就不会尝试」的工作)
5. **Contextual Adaptability** — 适配用户的具体 context，trust 是 co-constructed over time

### 2.2 十三个 Design Principles (Table 1 的核心)

我把这 13 个 principles 重新组织成 4 个 cluster，便于 intuition：

**Safety cluster:**
- **Deny-first with human escalation** — 未知动作默认 deny 或 escalate，never silently allow
- **Defense in depth with layered mechanisms** — 多层独立安全机制，任一层可阻止
- **Reversibility-weighted risk assessment** — 可逆/只读操作享受 lighter oversight
- **Isolated subagent boundaries** — subagent 不继承 parent 的 context 和 permissions

**Trust/Authority cluster:**
- **Graduated trust spectrum** — trust 不是固定状态而是 spectrum
- **Externalized programmable policy** — policy 是 externalized config 不是 hardcoded
- **Append-only durable state** — state 以 append-only log 形式存在，便于审计
- **Transparent file-based configuration and memory** — 配置是用户可见可编辑的 file 而不是 opaque database

**Capability cluster:**
- **Minimal scaffolding, maximal operational harness** — invest 在 infrastructure 而非 decision framework
- **Values over rules** — 给 model contextual judgment 而非 rigid procedures
- **Composable multi-mechanism extensibility** — 多个 extension mechanism 在不同 context cost 上

**Reliability cluster:**
- **Context as scarce resource with progressive management** — context 是 binding constraint，graduated compression
- **Graceful recovery and resilience** — silently recover，把人 attention 留给 unrecoverable 情况

---

## 3. High-Level Architecture: 7 个 Component + 5 个 Subsystem Layer

### 3.1 7-Component 视图 (Figure 1)

数据流走的是 left-to-right spine：

```
User ─→ Interfaces ─→ Agent Loop ─→ Permission System ─→ Tools ─→ Execution Environment
                ↑                                                ↓
            State & Persistence ←─────── (append-only JSONL) ──┘
```

关键 insight: **所有 entry surface 都 converge 到同一个 queryLoop()**——interactive CLI、headless CLI (`claude -p`)、Agent SDK、IDE/Desktop/Browser integration 全部共享同一个 async generator。

这与 OpenClaw 形成对比：OpenClaw 是 persistent WebSocket gateway daemon，agent loop 被 embedded 在 gateway control plane 内部作为 component。

### 3.2 5-Layer Subsystem Decomposition (Figure 3)

```
┌─────────────────────────────────────────────────┐
│ Surface Layer   │ Interactive CLI, Headless CLI, │
│                 │ Agent SDK, IDE/Desktop/Browser│
├─────────────────────────────────────────────────┤
│ Core Layer      │ queryLoop() + 5-shaper        │
│                 │ compaction pipeline           │
├─────────────────────────────────────────────────┤
│ Safety/Action   │ Permission system (7 modes +  │
│                 │ auto classifier), hooks (27   │
│                 │ events), tools (54 built-in), │
│                 │ shell sandbox, subagents      │
├─────────────────────────────────────────────────┤
│ State Layer     │ Context assembly, runtime     │
│                 │ state, JSONL persistence,     │
│                 │ CLAUDE.md + memory, sidechain │
├─────────────────────────────────────────────────┤
│ Backend Layer   │ Shell exec (sandboxed),       │
│                 │ MCP connections (8 transport  │
│                 │ variants), remote execution   │
└─────────────────────────────────────────────────┘
```

---

## 4. Agentic Query Loop: 一个 Turn 的 9 步流水线

这是整个系统的心脏，`query.ts:queryLoop()` 实现为 `AsyncGenerator`，yielding `StreamEvent | RequestStartEvent | Message | TombstoneMessage | ToolUseSummaryMessage`。**一个 turn 严格走 9 步**：

```
Step 1: Settings resolution (destructure immutable params)
Step 2: Mutable state init (single State object)
Step 3: Context assembly (getMessagesAfterCompactBoundary)
Step 4: Pre-model context shapers (5 layers)
Step 5: Model call (for await over deps.callModel())
Step 6: Tool-use dispatch (if tool_use blocks present)
Step 7: Permission gate
Step 8: Tool execution + result collection
Step 9: Stop condition check (no tool_use ⇒ done)
```

### 4.1 Tool Dispatch: Concurrent-Read, Serial-Write

这是 ReAct pattern (Yao et al. 2022) 的一个具体 production 实现。关键设计点：

**`StreamingToolExecutor`** 开始在 tool_use blocks 还在 streaming 时就启动执行，减少 multi-tool 响应的 latency。它有两套 coordination 机制：

- **Sibling abort controller** — 任一 Bash tool 出错就立即 kill 其他 in-flight subprocesses
- **Progress-available signal** — wake up `getRemainingResults()` consumer

工具分类：
- **Concurrent-safe** (read-only): parallel execution
- **Exclusive** (state-modifying, e.g. shell commands): serialized

结果按 tool received 顺序 buffer，保证 model 看到的 tool_result 顺序与 tool_use 顺序匹配。

可对比 **PASTE** (Sui et al. 2026) 的 speculative pre-execution：在 model 还在生成时就预测并预执行未来的 tool calls，用 speculation 来 hide latency。Claude Code 取了一个更保守的中间位。

Reference: ReAct paper https://arxiv.org/abs/2210.03629

### 4.2 Recovery Mechanisms

- **Max output tokens escalation**: 触顶时 retry with escalated limit，最多 3 次 (`MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3`)
- **Reactive compaction** (`REACTIVE_COMPACT` flag): context 接近容量时 summarize just enough，每 turn 至多 1 次 (`hasAttemptedReactiveCompact`)
- **Prompt-too-long handling**: API 返回 `prompt_too_long` 时先尝试 context collapse overflow recovery + reactive compaction
- **Streaming fallback + fallback model**

### 4.3 Stop Conditions

5 种终止条件：no tool use / maxTurns / context overflow / hook intervention (`hook_stopped_continuation`) / explicit abort。

---

## 5. 5-Layer Compaction Pipeline: Context-as-Bottleneck 的核心实现

这是论文里最值得深挖的子系统。在每次 model call 之前，5 个 shaper 顺序执行在 `messagesForQuery` 数组上，**cheap-and-targeted 在前，broad-and-expensive 在后**：

### Layer 1: Budget Reduction (`applyToolResultBudget()`)
**always active**，per-tool-result size limits，超过阈值的 tool output 被替换为 content reference。Exempt tools (`maxResultSizeChars = Infinity`) 保留全 output。Replacement 持久化到 transcript 以支持 resume 重建。

数学上：对第 i 个 tool_result message $m_i$，如果 $|m_i| > B_{tool}$ 则替换为 $\text{ref}(m_i)$，其中 $B_{tool}$ 是该 tool 的 budget 上界。

### Layer 2: Snip (`snipCompactIfNeeded()`, gated by `HISTORY_SNIP`)
轻量级 older-history trimming。返回 `{messages, tokensFreed, boundaryMessage}`。

有一个微妙的 plumbing 问题：主 token counter 从 most recent assistant message 的 `usage` field 推断 context size。snip 后这条 message 仍带着 pre-snip 的 `input_tokens`，所以 `snipTokensFreed` 必须显式传给 auto-compact，否则 savings 不可见。

### Layer 3: Microcompact (`CACHED_MICROCOMPACT`)
Fine-grained compression，always 跑 time-based path，可选 cache-aware path。

当 cached path enabled 时，boundary messages 被延迟到 API response 之后处理，这样能用真实的 `cache_deleted_input_tokens` 而非估计值。返回 `{messages, compactionInfo}`，`compactionInfo` 可能含 `pendingCacheEdits`。

### Layer 4: Context Collapse (`CONTEXT_COLLAPSE`)
**Read-time virtual projection** over history，**不修改 REPL 的 stored history**。

源码注释解释：*"Nothing is yielded; the collapsed view is a read-time projection over the REPL's full history. Summary messages live in the collapse store, not the REPL array."*

这是非常 elegant 的设计：model 看到的是 collapsed version，但 full history 仍然 available for reconstruction，通过 `applyCollapsesIfNeeded()` 替换 `messagesForQuery` 数组。

### Layer 5: Auto-compact
**Full model-generated summary**，触发条件：前 4 层跑完后 context 仍超过 pressure threshold。

`compactConversation()` (compact.ts) 流程：
1. Fire `PreCompact` hooks (允许 hook 注入 custom instructions)
2. `getCompactPrompt()` 构造 summary request
3. 调用 model 生成压缩 summary
4. `buildPostCompactMessages()` 返回：
   ```
   [boundaryMarker, ...summaryMessages, ...messagesToKeep, 
    ...attachments, ...hookResults]
   ```

Boundary marker 通过 `annotateBoundaryWithPreservedSegment()` 记录 `headUuid, anchorUuid, tailUuid` 以支持 read-time chain patching。

### Compaction 的设计哲学

Graduated pipeline 的反例：
- **Simple truncation** (drop oldest) — Coarse granularity，丢信息
- **Sliding window** — Medium granularity
- **RAG** — Fine granularity，但需要 infrastructure
- **Single summarization** — Coarse，one-pass compress
- **Graduated compaction (Claude Code)** — Very fine，multi-layer pipeline

Claude Code 选择 graduated 是因为 **no single compaction strategy addresses all types of context pressure**。Budget reduction 处理 oversized tool outputs，snip 处理 temporal depth，microcompact 处理 cache overhead，context collapse 处理 very long histories，auto-compact 作为 last resort 语义压缩。

成本：5 个相互作用层 + 多个 feature flag，行为对用户来说难以完全预测。Auto-compact 在 transcript 里产生 visible summary，microcompact 发出 boundary marker，但 context collapse **没有任何用户可见输出**——这是一个 transparency vs. context efficiency 的明确 trade-off。

Reference: "MemGPT: towards LLMs as operating systems" https://arxiv.org/abs/2310.08560

---

## 6. Permission System: 7 个 Modes + Deny-First Pipeline

### 6.1 七个 Permission Modes

| Mode | 描述 | 用途 |
|------|------|------|
| `plan` | model 必须先 create plan，user approve 后才执行 | 高 caution |
| `default` | 标准交互模式，多数操作 require approval | 一般开发 |
| `acceptEdits` | 工作目录内的 edits + 部分 shell (mkdir/rmdir/touch/rm/mv/cp/sed) auto-approve | 信任开发流 |
| `auto` | ML classifier 评估 fast-path 不通过的请求 | 生产自动化 |
| `dontAsk` | 不 prompt 但 deny rules 仍 enforce | CI/CD |
| `bypassPermissions` | skip 多数 prompt 但 safety-critical + bypass-immune 仍 apply | 极端自动化 |
| `bubble` | internal-only, subagent 升级到 parent terminal | 多 agent |

External visible 5 个：`acceptEdits, bypassPermissions, default, dontAsk, plan`。`auto` 由 `TRANSCRIPT_CLASSIFIER` flag conditionally include。`bubble` 是 internal type union 用于 subagent escalation。

### 6.2 Deny-First Rule Evaluation

```
deny rules (always win)  >  ask rules  >  allow rules
```

`toolMatchesRule()` 先 check deny：**broad deny ("deny all shell") 不能被 narrow allow ("allow npm test") override**，即使 allow rule 更 specific。这违反了常见的 most-specific-match 原则，是有意的 safety-conservative 设计。

Rule 支持 tool-level matching (by name) 和 content-level matching (e.g. `Bash(prefix:npm)`)。

### 6.3 七层独立 Defense-in-Depth

任何一层都能 block 一个 request：

1. **Tool pre-filtering** (`filterToolsByDenyRules`) — blanket-denied tools 在 model 视野中直接被 strip 掉，model 根本无法 attempt
2. **Deny-first rule evaluation** (`permissions.ts`)
3. **Permission mode constraints** — active mode 决定 baseline
4. **Auto-mode ML classifier** (`yoloClassifier.ts`) — 可能 deny rule system 会 allow 的请求
5. **Shell sandboxing** (`shouldUseSandbox.ts`) — 独立的 filesystem + network isolation
6. **Not restoring permissions on resume** — session-scoped permissions 不持久化
7. **Hook-based interception** — `PreToolUse` hooks 能 modify permission decisions

### 6.4 Permission Handler 的四个分支 (`useCanUseTool.tsx`)

1. **Coordinator path** — multi-agent coordination mode，先尝试 automated resolution (classifier → hooks → rules) 再 fallback 到 user interaction
2. **Swarm worker path** — agent teams 中的 worker agents
3. **Speculative classifier** — `BASH_CLASSIFIER` enabled + BashTool 时，classifier race against timeout。high confidence ⇒ 立即 approve，no user interaction
4. **Interactive fallback** — 标准 user approval dialog

### 6.5 关键 Insight: Approval Fatigue 重构了整个 Problem

Anthropic 的 auto-mode 分析发现 **93% permission prompts 被 user 批准** (Hughes 2026)。这意味着 interactive confirmation 行为上不可靠——user habituated 后不仔细 review。

系统的回应 **不是加更多 warning**，而是重构 problem：

> "defined boundaries (sandboxing, auto-mode classifiers) within which the agent can work freely, rather than per-action approvals that users stop reviewing once habituated" (Dworken and Weller-Davies 2025)

Longitudinal 数据 (McCain et al. 2026)：auto-approve rate 从 <50 sessions 时的 ~20% 增长到 750 sessions 时的 >40%。Sandboxing 减少 permission prompts 估约 **84%**。

这把 safety 从 "human vigilance" 问题重新框定成 **"human-factors 减少决策数量"** 问题。

References:
- https://www.anthropic.com/engineering/claude-code-auto-mode
- https://www.anthropic.com/engineering/claude-code-sandboxing
- https://anthropic.com/research/measuring-agent-autonomy

### 6.6 Defense-in-Depth 的 Independence Assumption 隐患

论文里一个尖锐 observation：defense-in-depth **依赖 independence assumption**——如果一层 fail，其他层 catch。

但 Claude Code 的多个 safety layer **share common performance/economic constraints**：
- Auto-mode classifier 是独立 LLM call，有直接 token cost
- `bashSecurity.ts` 做 sequential AST-based parsing，有 latency
- Deny-first rule evaluation 依赖 command structure parsing

**Adversa.ai 报告**：commands with >50 subcommands fallback 到单个 generic approval prompt，跳过 per-subcommand deny-rule checks，因为 per-subcommand parsing 导致 UI freeze。

这是一个 **structural tension**：当 layer 共享 failure mode 时，defense-in-depth degrade。

Reference: https://adversa.ai/blog/claude-code-security-bypass-deny-rules-disabled/

---

## 7. Extensibility: 4 Mechanisms 在不同 Context Cost 上

论文的核心 insight 是 **没有单一 extension mechanism 能 span 整个 range**——从 zero-context lifecycle hooks 到 schema-heavy tool servers，单一机制会强制 trade-off。

### 7.1 四种 Mechanism 的 Context Cost 梯度

| Mechanism | Unique Capability | Context Cost | Insertion Point |
|-----------|-------------------|--------------|-----------------|
| **MCP servers** | External service integration (multi-transport) | High (tool schemas) | `model():tool pool` |
| **Plugins** | Multi-component packaging + distribution | Medium (varies) | All three points |
| **Skills** | Domain-specific instructions + meta-tool invocation | Low (descriptions only) | `assemble():context injection` |
| **Hooks** | Lifecycle interception + event-driven automation | Zero by default | `execute():pre/post tool` |

### 7.2 Agent Loop 的三个 Injection Point

论文把 loop 抽象为三个 injection point：

```julia
while not stopped:
    # a assemble -- build what the model sees
    context = assemble(
        system_prompt,
        tool_schemas,           # callable tool signatures
        history,                # prior turn messages
        hook_additions,         # pushed in by hooks
    )
    
    # b model -- pick the next action
    action = model(context, tools)
    if action.is_text_only():
        stopped = run_stop_hooks(action)
        continue
    
    # c execute -- gate and run the tool call
    if not permitted(action):
        continue
    action = run_pre_tool_hooks(action)
    result = execute(action)
    result = run_post_tool_hooks(result)
    history.append(action, result)
```

- **`assemble()`** controls what the model sees — skills, CLAUDE.md, MCP resources 注入这里
- **`model()`** controls what it can reach — built-in tools + MCP tools + SkillTool + AgentTool 都在 flat tool pool
- **`execute()`** controls whether/how an action actually runs — permission + hooks + sandbox

### 7.3 Tool Pool Assembly 的 5 步流水线

`assembleToolPool()` (tools.ts) 是 single source of truth：

1. **Base tool enumeration** — `getAllBaseTools()` 返回 up to 54 tools (19 unconditional + 35 conditional)
2. **Mode filtering** — `CLAUDE_CODE_SIMPLE` mode 只 expose Bash/Read/Edit (或 REPLTool)
3. **Deny rule pre-filtering** — strip blanket-denied tools before model 看见
4. **MCP tool integration** — MCP tools filtered by deny rules 后 merge
5. **Deduplication** — by name, built-in 优先 over MCP

### 7.4 Hooks: 27 个 Event Types

源码在 `coreTypes.ts` 定义 27 个 hook events，覆盖：

- **Tool authorization** (5): PreToolUse, PostToolUse, PostToolUseFailure, PermissionRequest, PermissionDenied
- **Session lifecycle** (5): SessionStart, SessionEnd, Setup, Stop, StopFailure
- **User interaction** (3): UserPromptSubmit, Elicitation, ElicitationResult
- **Subagent coordination** (5): SubagentStart, SubagentStop, TeammateIdle, TaskCreated, TaskCompleted
- **Context management** (4): PreCompact, PostCompact, InstructionsLoaded, ConfigChange
- **Workspace events** (4): CwdChanged, FileChanged, WorktreeCreate, WorktreeRemove
- **Notifications** (1)

**5 个 safety-related，22 个 lifecycle/orchestration**。

Hook 命令类型 4 种 (`schemas/hooks.ts`)：
- `command` — shell 命令
- `prompt` — LLM prompt hooks
- `http` — HTTP hooks
- `agent` — agentic verifier hooks

加上 SDK 用的 non-persistable `callback` hooks。

### 7.5 一个 Plugin 能扩展多少东西

`PluginManifestSchema` (`utils/plugins/schemas.ts`) 接受 **10 个 component types**：
commands, agents, skills, hooks, MCP servers, LSP servers, output styles, channels, settings, user configuration。

Plugin 是 **packaging + distribution 层**，不是 distinct runtime primitive。一个 plugin package 可以同时 extend Claude Code 跨多个 component types。

---

## 8. Context Construction & Memory: CLAUDE.md 4-Level Hierarchy

### 8.1 四级 Memory Hierarchy

```
1. Managed memory  (/etc/claude-code/CLAUDE.md)        — OS-level policy for all users
2. User memory      (~/.claude/CLAUDE.md)              — private global instructions
3. Project memory   (CLAUDE.md, .claude/CLAUDE.md,     — checked into codebase
                     .claude/rules/*.md)
4. Local memory     (CLAUDE.local.md in project roots)  — gitignored, private project-specific
```

**Loading 顺序: reverse order of priority**——later-loaded files receive more model attention。File discovery 从 current directory 向上 traverse 到 root。

### 8.2 Lazy Loading 设计

- **Root-to-CWD directories**: `.claude/rules/*.md` eager load at startup
- **Nested directories below CWD**: 即使 unconditional rules 也 **lazy load**，只在 agent 读取 matching directory 中的文件时才加载

这意味着 model 的 instruction set 在 conversation 过程中会 evolve，随 codebase 探索新部分。

### 8.3 CLAUDE.md 作为 User Context 而非 System Prompt

一个非常微妙的 architectural choice：CLAUDE.md content 被作为 **user context message** 而非 system prompt content deliver (`context.ts` 里的 `prependUserContext()`)。

**Implication**: model compliance with CLAUDE.md instructions 是 **probabilistic 而非 guaranteed**。Permission rules (deny-first) 提供 deterministic enforcement layer。

这是 **guidance (probabilistic) vs. enforcement (deterministic) 的明确分离**——一个有意的 architectural commitment。

### 8.4 Memory Retrieval: LLM-based Scan 而非 Embedding

Claude Code **不使用 embeddings 或 vector similarity index**。它用 **LLM-based scan of memory-file headers** 选 up to 5 个 relevant files on demand，以 **file granularity 而非 entry granularity** surface。

Trade-off：
- **Embedding-based**: 可 selectively retrieve individual entries，但需要 infrastructure + opaque to user
- **LLM scan**: transparent, no infra, 但 coarse granularity + LLM call cost

### 8.5 `@include` Directive

CLAUDE.md 支持 `@path, @./relative, @~/home, @/absolute` 几种 include 语法。Circular references 通过 tracking processed paths 防止。Non-existent files silently ignored。Include 只在 leaf text nodes 工作，不在 code blocks 内。

---

## 9. Subagent Delegation: Isolated Context Windows

### 9.1 AgentTool 的 Dispatch Axes

`AgentTool` (`AgentTool.tsx`) 是 meta-tool，dispatch 沿 **三个 axes**：

- **Routing axis** (teammate dispatch)
- **Isolation axis** (`worktree`, `remote` internal-only, `in-process` default)
- **Lifecycle axis** (`async`, `sync`)

### 9.2 内置 Subagent Types (up to 6)

- **Explore**: read/search-oriented，write/edit 在 deny-list
- **Plan**: creates structured plans
- **General-purpose**: broadly capable
- **Claude Code Guide**: onboarding/docs
- **Verification**: validation checks (test suites, linting)
- **Statusline-setup**: terminal status line config

### 9.3 Custom Subagent Definitions

用户通过 `.claude/agents/*.md` 定义 custom subagents。Markdown body 作为 system prompt，YAML frontmatter 配置：
- `description, tools (allowlist), disallowedTools, model, effort, permissionMode`
- `mcpServers, hooks, maxTurns, skills, memory scope, background flag, isolation mode`

**这意味着一个 custom agent 是一个完全配置的、isolated 的 sub-system**，有自己的 tools, model, permissions, hooks, memory scope, isolation mode。

### 9.4 Isolation 模式对比

| Isolation 方式 | 代表系统 | 边界强度 | 基础设施需求 |
|---------------|---------|---------|-------------|
| **Container-based** | SWE-Agent, OpenHands | 最强 (resource boundaries) | 需 container infra |
| **Worktree-based** | Claude Code | 中 (filesystem-level) | 零依赖 (用 Git) |
| **Context-only** | AutoGen | 弱 (共享 fs, 分离 history) | 零依赖 |

Claude Code 的 worktree isolation 是一个非常 elegant 的工程选择：**用 Git 的 built-in mechanism 而非引入 container orchestration**。

### 9.5 Permission Override 规则

Subagent 定义 `permissionMode` 时，override apply **除非 parent 已经在 `bypassPermissions, acceptEdits, auto`**——因为这些 mode 代表 user 关于 safety/autonomy trade-off 的 explicit decision。

Async agents 的 prompt 显示逻辑 cascade：
```
explicit canShowPermissionPrompts 
  → bubble mode (always show, escalate to parent terminal) 
  → default (sync: show, async: no show)
```

Background agents that can show prompts 设 `awaitAutomatedChecksBeforeDialog: true`，确保 classifier + hooks 先 resolve 再 interrupt user。

### 9.6 Sidechain Transcripts

每个 subagent 写自己的 `.jsonl` + `.meta.json` 到独立文件 (`sessionAgent.ts`, `runAgent.ts`)。

**关键**: subagent 全 history **永远不进入 parent context window**——只有 final response text + metadata return。这是 "context as bottleneck" principle 在 multi-agent 场景下的具体实现。

Token 成本数据：**Claude Code agent teams 消耗约 7× 一个标准 session 的 tokens** (Anthropic 2025b)，使 summary-only return 在 isolated contexts 下更关键。

Multi-instance coordination 用 **file locking** 而非 message broker——零依赖 deployment + 完全 debuggability (任何 agent 状态可读 plain-text JSON file)。

Reference: https://code.claude.com/docs/en/agent-teams

### 9.7 SkillTool vs. AgentTool 的本质区别

| | SkillTool | AgentTool |
|---|-----------|-----------|
| **Operation** | Inject instructions into current context | Spawn new isolated context window |
| **Context cost** | Low | High (new window) |
| **History sharing** | 共享 parent | 不共享 (除 fork-subagent path) |

---

## 10. Session Persistence: Append-Only JSONL

### 10.1 三个独立 Persistence Channel

1. **Session transcripts** — project-scoped JSONL，含 user/assistant/attachment/system messages + compaction markers + filehistory snapshots + attribution snapshots + content-replacement records
2. **Global prompt history** — `history.jsonl` 在 Claude config home，仅 user prompts，`readLinesReverse()` 支持 ↑ + ctrl+r navigation
3. **Subagent sidechains** — 每个子代理独立 `.jsonl + .meta.json`

### 10.2 Resume/Fork 的设计选择

`--resume` 通过 replay transcript 重建 conversation (`conversationRecovery.ts`)。`fork` 创建新 session (`commands/branch/branch.ts`)。

**关键 safety choice: resume 和 fork 不恢复 session-scoped permissions**。

理由：session 是 isolated trust domain。恢复之前 granted permissions 会把 stale trust decision 带入 changed context。系统选 **re-granting over implicit persistence**，accept user friction 以维持 safety invariant。

### 10.3 Boundary Marker 的 Chain Patching

Compaction 后的 boundary marker 通过 `annotateBoundaryWithPreservedSegment()` 记录 `headUuid, anchorUuid, tailUuid`。这些 UUID 让 session loader 能在 read-time patch message chain：

- Preserved messages 在 disk 上保留原 `parentUuids`
- Loader 用 boundary metadata 在 read-time 正确 link 它们

这是 **mostly-append design**：compaction 永远不 modify/delete 已写的 transcript lines，只 append 新的 boundary + summary events。

### 10.4 "Checkpoints" 是什么

`~/.claude/filehistory/<sessionId>/` 下的 file-history checkpoints——**file-level snapshots for reverting filesystem changes**，不是 generic checkpoint store。用于 `--rewind-files`。

---

## 11. 与 OpenClaw 的 Architectural Contrast

OpenClaw 是一个 independent open-source system，persistent WebSocket gateway daemon (default port 18789, loopback-only)，连接 ~24 个 messaging surface (WhatsApp, Telegram, Slack, Discord, Signal 等) 到 embedded agent runtime。

### 11.1 六个对比维度 (Table 3)

| Dimension | Claude Code | OpenClaw |
|-----------|-------------|----------|
| **System scope** | Ephemeral CLI per-session | Persistent WS gateway daemon |
| **Trust model** | Deny-first per-action + 7 modes + graduated trust | Single trusted operator + DM pairing + allowlists + opt-in sandboxing |
| **Agent runtime** | `queryLoop()` 作为系统中心 | Pi-agent 嵌入 gateway RPC dispatch; per-session queue serialization |
| **Extension** | 4 mechanisms at graduated context costs | Manifest-first plugin system, 12 capability types, central registry |
| **Memory** | CLAUDE.md 4-level + 5-layer compaction + LLM-based scan | Workspace bootstrap files (AGENTS.md, SOUL.md, TOOLS.md, IDENTITY.md, USER.md, ...); separate memory system with optional hybrid vector+keyword search; experimental dreaming |
| **Multi-agent** | Task-delegating subagents, worktree isolation, summary-only return | 两层：multi-agent routing + sub-agent delegation (max depth 5, default 1, recommended 2) |

### 11.2 三个关键观察

1. **Recurring design questions 跨 deployment context 稳定**——where reasoning lives, what safety posture, how manage context, how structure extensibility——这些问题普适，但答案随 context 变
2. **系统在多个 dimensions 上做 opposite bets**：
   - Claude Code: per-action safety evaluation; OpenClaw: perimeter-level identity/access control
   - Claude Code: agent loop 作为 architectural center; OpenClaw: gateway control plane 作为 center，agent loop 嵌入为 component
   - Claude Code: extensions modify one context window; OpenClaw: plugins extend shared gateway surface
3. **Compositional relationship**: OpenClaw 通过 ACP (Agent Client Protocol) 能 host Claude Code 作为 external coding harness——**两系统 composable 而非纯 alternative**。这暗示 AI agent 的 design space 不是 flat taxonomy 而是 **layered**，gateway-level + task-level 可组合。

Reference: https://github.com/openclaw/openclaw

---

## 12. Value Tensions & Architectural Trade-offs

### 12.1 五对 Value Tension (Table 4)

| Value Pair | Tension | Evidence |
|------------|---------|----------|
| Authority × Safety | Approval fatigue vs. protection | 93% approval rate undermines vigilance |
| Safety × Capability | Performance vs. defense depth | >50-subcommand fallback skips per-subcommand checks |
| Adaptability × Safety | Extensibility vs. attack surface | CVEs exploit pre-trust initialization of hooks + MCP servers (Donenfeld and Vanunu 2026) |
| Capability × Adaptability | Proactivity vs. disruption | +12-18% tasks 但 high frequency 下 preference drop (Chen et al. 2025) |
| Capability × Reliability | Velocity vs. coherence | Bounded context 阻碍 full codebase awareness; subagent isolation 限制 cross-agent consistency |

### 12.2 Long-term Capability Preservation 作为 Evaluative Lens

论文引入第六个 concern 作为 evaluative lens，但**不作为 co-equal design value**——因为它没有 prominently 反映在 architecture 或 Anthropic stated values 中。

**Empirical evidence**:
- **Becker et al. 2025** RCT (16 experienced developers, 246 tasks): AI tools 让开发者 **慢 19%**，尽管 perceived 20% improvement
- **He et al. 2025** Cursor adoption (807 repos): code complexity 增加 **40.7%**，initial velocity spike 到 month 3 消退到 baseline
- **Kosmyna et al. 2025** EEG study (54 participants): LLM users 显示 weakened neural connectivity，**在 AI 移除后仍持续**
- **Liu et al. 2026** audit (304,000 AI-authored commits, 6,275 repos): ~25% AI-introduced issues 持续到 latest revision，security-related 持续率更高
- **Rak 2025**: entry-level tech hiring 2023-2024 下降 25%

这构成一个 **paradox of supervision**：overreliance on AI risks atrophying skills needed to supervise it。

### 12.3 Pre-Trust Initialization Vulnerability

两个独立验证的 vulnerabilities 共享 root cause：**pre-trust initialization ordering**。

Project initialization 阶段执行的 code (hooks, MCP server connections, settings file resolution) **在 interactive trust dialog 之前**运行。这创建一个 structurally privileged phase，safety guarantees **不适用**。

Permission pipeline 描绘的是 spatial ordering，但 **不 capture temporal dimension**——specifically，每个 mechanism 在 session initialization 过程中何时 active。Initialization 顺序是：

```
extension loading  →  trust dialog  →  permission enforcement
```

Extension architecture 在 safety architecture fully engage 之前 operate。这 refine 了 extensibility-vs-simplicity tension：extensibility 创建 attack surface **不仅通过 combinatorial complexity，还通过 initialization ordering**。

References:
- https://research.checkpoint.com/2026/rce-and-api-token-exfiltration-through-claude-code-project-files-cve-2025-59536/
- CVE-2025-59536 (CVSS 8.7), CVE-2026-21852 (CVSS 5.3)

### 12.4 Empirical Predictions

Architecture 性质产生 **testable predictions**:
1. **Bounded context** 阻止 agent maintain simultaneous awareness of full codebase ⇒ agent-generated code 将 exhibit 更高 pattern duplication + convention violation rate
2. **Subagent isolation** + 独立 assembled tool pool ⇒ parallel agents 可能 independently re-implement 已存在的 solutions
3. **Single-pass generation** 在 long-horizon 任务上 degrade

He et al. 2025 的发现 (initial velocity spike 到 month 3 消退到 baseline，complexity 上升 proportional decrease future velocity) **consistent with** 这些 predictions——gains 是 **self-cancelling**。

Claude Code 的 context management pipeline **specifically designed to mitigate**:
- Graduated compression 保留 most recent + most relevant context
- Cache-aware compaction 避免 invalidating prompt caches
- Read-time projection 维持 full history for reconstruction
- Subagent summary isolation 阻止 exploratory noise accumulate

**Whether sufficient 是 directly measurable empirical question**——source-level analysis 不能 resolve。

---

## 13. KAIROS: 一个 Proactive Architecture 案例

论文提到 feature-gated **KAIROS**——persistent background agent with tick-based heartbeats：

- 当没有 user messages pending，系统注入 periodic `<tick>` prompts
- Model 决定 **act or sleep**
- Terminal focus awareness: user 离开时 maximize autonomous action，user 在场时 increase collaboration
- Economic throttling via `SleepTool`: 每次 wake-up costs API call; prompt cache 5 min 后 expire，使 sleep/wake 成 explicit cost optimization

这直接 address 一个 documented tension (Chen et al. 2025)：
- Proactive AI assistants **increase task completion 12-18%**
- 但 **reduce user preference at high frequencies** (47% vs. 80-90% in high-frequency Persistent Suggest variant)

KAIROS 通过绑定 proactivity 到 user presence + token economics 解决。**这种 binding 不常见于 production agent systems**，但 KAIROS 是否在 production builds 中 active 未能 confirm。

---

## 14. 六个 Open Directions (Section 12)

### 14.1 Silent Failure 和 Observability-Evaluation Gap

Bessemer 2026 infrastructure report: **78% AI failures 是 invisible**。LangChain 1,340-respondent survey: observability **89% adoption** vs. offline evaluation **52.4%**——巨大 gap。

Quality, **不是 cost**，是 production use 的 top barrier。

问题：generator-evaluator separation, sprint contracts, post-hoc checks 应该 **inside harness (作为额外 hook events) 还是 outside 作为 separate evaluation layer**？现有 hook pipeline 能否 host 这种 scaffolding 在当前 context-cost envelope 内？

References:
- Cemri et al. 2025: https://arxiv.org/abs/2503.13657 (14 failure modes)
- Kapoor et al. 2024: https://arxiv.org/abs/2407.01502
- Pathak et al. 2025: https://arxiv.org/abs/2511.04032

### 14.2 Cross-Session Persistence 和 Longitudinal Colleague Relationships

当前两层：
- **Static instruction** (CLAUDE.md hierarchy + auto memory)
- **Single session transcript** (append-only JSONL, session-scoped permissions 不 restore)

**中间层** (durable state 既不是 static instruction 也不是 single session transcript) 是 open question。

候选机制：
- **MemGPT** (Packer et al. 2023) — LLM as OS with paged memory
- **Mem0** (Chhikara et al. 2025) — production-oriented memory store survives restarts
- **A-Mem** (Xu et al. 2025) — research agentic-memory design
- **Reflexion** (Shinn et al. 2023) — verbal reinforcement across attempts
- **Voyager skill library** (Wang et al. 2023) — embodied agent 累积 skill 跨 tasks

**Human side**: Dell'Acqua et al. 2025 P&G field experiment (776 professionals), Stray et al. 2025 Copilot rollout longitudinal, Xiao et al. 2025 AI-teamwork trajectories——都报告 human-AI work dynamics 随 collaboration 累积而 shift。

Reference: https://arxiv.org/abs/2310.08560 (MemGPT)

### 14.3 Harness Boundary Evolution: Where / When / What / With Whom

Rajasekaran 2026 观察：**"the space of interesting harness combinations doesn't shrink as models improve; it moves"**

四个 extension axis:

**Where**: Martin et al. 2026 "Managed Agents" virtualizes session, harness, sandbox 为 independently replaceable interfaces——明确类比 OS virtualizing hardware into processes/files。Khattab et al. 2023 (DSPy) treats harness 本身作为 compile target。

**When**: KAIROS + Liu et al. 2025, Pu et al. 2025, Lee et al. 2025, Pasternak et al. 2025, Sun et al. 2025, Deng et al. 2025——proactivity design space 在 programming + ambient-interface settings 扩展。

**What**: VLA (Vision-Language-Action) work——RT-2 (Brohan et al. 2024), π_0 (Black et al. 2024), Figure AI Helix, Gr00t N1 (Bjorck et al. 2025)——harness 扩展 beyond textual tool returns 到 physical actions。这 faces reversibility-weighted risk principle at **cost asymmetry that the principle names but does not quantify for non-textual actions**。

**With whom**: Role-differentiated multi-agent (MetaGPT Hong et al. 2023, CAMEL Li et al. 2023, AgentVerse Chen et al. 2023, ChatDev Qian et al. 2024), multi-agent debate (Du et al. 2024, Liang et al. 2024), graph-structured workflows (GPTSwarm Zhuge et al. 2024)。

**Single harness 能否 span 所有四个 extension，还是会 fragment 成 specialized stacks** 是 open question。

Reference: https://www.anthropic.com/engineering/managed-agents

### 14.4 Horizon Scaling: From Session to Scientific Program

METR study (Kwa et al.) 测量 50%-time horizon——frontier agents 以 fixed reliability 成功的 task duration——以及它如何 cross model generations 演化。

参考文献：
- **AI Scientist** (Lu et al. 2024) — end-to-end autonomous research pipeline
- **AI co-scientist** (Gottweis et al. 2025) — multi-agent hypothesis generation across days
- **AlphaEvolve** (Novikov et al. 2025) — algorithmic discovery over weeks

**Question**: Claude Code 的 context-management pipeline (5-layer), last-assistant-text return policy (summary-only), append-only persistence 能否 sufficient 当 sessions compose 成 multi-session programs?

Reference: https://arxiv.org/abs/2506.13131 (AlphaEvolve)

### 14.5 Governance 和 Oversight at Scale

EU AI Act **fully applicable August 2026**。GPAI Code of Practice (European Commission 2025a) + implementation guidelines (2025b) detail general-purpose AI obligations。

MIT AI Agent Index (Staufer et al. 2026): 仅 **13.3%** indexed agentic systems publish agent-specific safety cards。International AI Safety Report (Bengio et al. 2026) 警告: "AI agents pose heightened risks because they act autonomously, making it harder for humans to intervene before failures cause harm"。

Bartz v. Anthropic ruling (June 23, 2025): 添加 **input-side constraint** on training-data sourcing (lawful acquisition of copyrighted works)，distinct from output-side copyright questions about AI-generated code。

两个 open properties:
1. Deny-first evaluation 是 internally auditable through session transcripts，但 **not yet externally auditable** in GPAI Code of Practice contemplate 的形式
2. **values-over-rules principle** 是否 admit 显式 rule articulation that compliance review 可能 call for

References:
- https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai
- https://arxiv.org/abs/2602.21012 (International AI Safety Report)

### 14.6 Long-Term Human Capability: From Lens to Design Problem

论文最后的 pivot：**未来系统 could treat sustainability gap as first-class design problem, not downstream evaluation metric**。

两个 sub-question:

**Measurement gap**: 现有 citations 在 session-to-multi-month scales operate，但 harness exposes **no per-session signal for comprehension or convention drift**。Related work:
- Barke et al. 2023 — programmer interaction modes
- Perry et al. 2023 — AI-induced code-security regressions
- Aiersilan 2026 — session-level cognitive-offloading probes protocol

**Design gap**: architecture 能否 respond to such measurements once they exist? Analogous to generator-evaluator separation applied to the **human loop**, comprehension-preserving surfaces, or mechanisms not yet named.

**Open**: harness 是否是 right locus for that action (vs. IDE, organisation, human development loop)?

---

## 15. 三个 Cross-Cutting Recurring Design Commitments

把所有六个 subsystem 一起读，浮现三个 cross-cutting commitment:

### 15.1 Graduated Layering over Monolithic Mechanisms

| Subsystem | Layers |
|-----------|--------|
| **Permission** | 7 stages (pre-filter → deny-rules → modes → classifier → sandbox → non-restore → hooks) |
| **Context mgmt** | 5 compaction + lazy CLAUDE.md + deferred tool schemas + summary-only subagent returns |
| **Extensibility** | 4 mechanisms at graduated context costs (hooks zero, skills low, plugins medium, MCP high) |

**Trade**: simplicity + debuggability **for** defense in depth。Layer 间 interaction 产生 emergent behaviors 难以从单一 config file 预测。

### 15.2 Append-Only Designs Favoring Auditability over Query Power

- Session transcripts: append-only JSONL + read-time chain patching
- Permissions: not restored across session boundaries
- Context compaction: read-time projections over full history, not destructive edits

**Cost**: richer structured queries ("show me all tool calls that modified file X across sessions") 需要 post-hoc reconstruction，not direct lookup。

### 15.3 Model Judgment within Deterministic Harness

跨所有 subsystem：architecture trusts model's judgment within rich deterministic harness 而 **不 constrain 它的选择**。1.6% decision-logic ratio 量化这一点。

**Harness creates conditions (tool routing, permission enforcement, context assembly, recovery logic) under which model can decide well**。

**Trade-off**: good local decisions 可 produce poor global outcomes 当 bounded context 阻碍 global awareness (Section 11.4 empirical predictions)。

---

## 16. 一个 Final Intuition: Claude Code 作为 LLM-as-OS 的 partial instantiation

Karpathy 在 2023 年 11 月的 "Intro to LLMs" talk 中 popularize 了 **LLM-as-OS framing**。Claude Code 的 architecture 可视为这个 framing 的 partial production instantiation：

| OS Concept | Claude Code 对应 |
|-----------|----------------|
| Process scheduler | `queryLoop()` async generator + `StreamingToolExecutor` |
| Virtual memory | 5-layer compaction pipeline + read-time projection |
| Filesystem | append-only JSONL transcripts + CLAUDE.md hierarchy |
| Syscalls | tool_use protocol |
| Permission system | 7-mode deny-first pipeline + classifier + sandbox |
| Process isolation | worktree-based subagent isolation |
| IPC | sidechain transcripts + file locking for multi-agent |
| Device drivers | MCP servers (multi-transport) |
| Kernel modules | plugins (10 component types) |
| Daemons | hooks (27 event types, 4 command types) |
| User accounts | 4-level CLAUDE.md memory hierarchy |

Martin et al. 2026 "Managed Agents" essay 把这个 analogy explicit: virtualizing session/harness/sandbox 为 independently replaceable interfaces，类比 OS virtualizing hardware into processes/files。Rajasekaran 2026 Harness Design essay: "the space of interesting harness combinations doesn't shrink as models improve; it moves"——意味着这个 OS 不是 fixed optimum 而是 co-evolving snapshot。

References:
- Karpathy 1hr talk: https://www.youtube.com/watch?v=zjkBMFhNj_g
- https://www.anthropic.com/engineering/managed-agents
- https://anthropic.com/engineering/harness-design-long-running-apps

---

## 17. 关键 Take-aways

1. **Claude Code 是一个 thin decision layer (1.6%) + thick operational harness (98.4%) 的 design**——invest 在 deterministic infrastructure 而非 decision scaffolding，bet 是 increasingly capable models benefit 更 from rich environment 而非 constraining frameworks。

2. **5-layer compaction pipeline** 是 context-as-bottleneck principle 的具体实现，**cheap-and-targeted 在前，broad-and-expensive 在后**——budget → snip → microcompact → context collapse → auto-compact。Read-time projection (context collapse) 让 full history 持续 available for reconstruction 而 model 看到 collapsed view。

3. **Permission system 的 7-layer defense-in-depth** 解决 93% approval fatigue 问题：把 safety 从 "human vigilance" 重构为 "减少决策数量 + 在 trust boundary 内自由"。

4. **4 个 extension mechanism (MCP/plugins/skills/hooks) 在不同 context cost 上**——zero-cost hooks 让 cheap extensions scale widely，schema-heavy MCP 留给真正需要新 tool surface 的 case。

5. **Subagent isolation 用 Git worktree + sidechain transcripts**——zero-dependency deployment + 完全 debuggability + subagent history 永远不 inflate parent context (summary-only return)。

6. **Append-only design 跨所有 subsystem**——session transcripts, read-time chain patching, compaction 不 modify/delete 已写 lines。Favor auditability over query power。

7. **CLAUDE.md 作为 user context 而非 system prompt** 是一个有意的 architectural choice——model compliance 是 probabilistic，permission rules 提供 deterministic enforcement。**Guidance (probabilistic) vs. enforcement (deterministic) 的明确分离**。

8. **Pre-trust initialization vulnerability** 是一个 **temporal dimension** 问题，permission pipeline 描绘的 spatial ordering **不 capture**：extension loading 在 trust dialog 之前 operate，创建 structurally privileged phase。

9. **Long-term capability preservation 是 evaluative lens 而非 design value**——empirical evidence (慢 19%, complexity +40.7%, neural connectivity persistence, hiring -25%) 暗示 short-term amplification **可 at cost of long-term human understanding + codebase coherence + developer pipeline**。未来系统 **could treat sustainability gap as first-class design problem, not downstream evaluation metric**。

10. **OpenClaw 对比** 显示 recurring design questions 普适，但答案随 deployment context 变：per-action safety evaluation vs. perimeter-level access control; agent loop as center vs. gateway control plane as center; extensions modify one context vs. extend shared gateway surface。**两系统 composable via ACP**——design space 是 layered，不是 flat taxonomy。

这个 paper 的最大贡献：把 production coding agent 的 design space **从 ad-hoc engineering choices 抽象为 recurring design questions**，每个 question 有 alternative answers，每个 answer encode 一组 trade-offs。Claude Code 占据 design space 中一个 **coherent design point**，privileging model autonomy within rich operational harness，OpenClaw 占据另一个 point。未来的 agent 系统 **不需要在 same point 上 converge**，但需要在 same design space 中 consciously navigate。
