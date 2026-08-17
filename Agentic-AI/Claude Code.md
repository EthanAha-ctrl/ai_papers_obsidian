---
source_pdf: Claude Code.pdf
paper_sha256: f36fba71726ccde93ec03bab4911b2bed81da413a46aeafac112f915f93a7513
processed_at: '2026-08-03T15:50:21-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Claude Code 源码泄露

Karpathy, 我上一版讲得太学术了。重来一次,用大白话。

---

## 先说这事怎么发生的

Anthropic 发 npm 包的时候,`package.json` 的 `files` 字段没做白名单,把 `.map` 文件一起发了出去。这个 map 文件里有个 URL,指向他们 R2 存储桶上的源码 zip 包,而且这个 URL 没鉴权。点开就能下。

zip 解压,1903 个文件,51 万行 TypeScript,全部裸奔。

讽刺的地方在于:他们在 binary 层面做了大量防泄漏设计——编译时 dead code elimination、宠物名 hex 编码、`excluded-strings.txt` 黑名单扫描——结果被一个忘删的 `.map` 文件全端掉了。

安全这件事,100 个环节做对 99 个,漏一个等于没做。jailbreaker 攻击 LLM 也一样,总找最弱的轴打,不会硬刚你最强的那面。

参考: [npm files field 文档](https://docs.npmjs.com/cli/v8/configuring-npm/package-json#files), [source map security 讨论](https://www.scien.cx/2023/01/30/source-map-security/)

---

## 技术栈一句话

Bun + TypeScript + React + Ink。Ink 就是让 React 跑在终端里的库,2017 年就有了,Gatsby CLI、Prisma CLI 都在用。

Claude Code 的 UI 复杂度比一般 CLI 高一个量级:多 Agent 并行、流式输出、工具执行中用户随时中断、权限弹窗。状态复杂到这个程度,用 React 比 hand-rolled state machine 靠谱。

参考: [Ink GitHub](https://github.com/vadimdemedes/ink)

---

## 核心:一个 `while(true)` 循环

整个 Claude Code 的"大脑"在 `src/query.ts`,1729 行。外面那层 `QueryEngine.ts` 只是会话管理,真正的 agent loop 在 `query.ts` 里。

伪代码:

```
async function* queryLoop(params) {
  let state = { messages, toolUseContext, turnCount: 1, ... }
  while (true) {
    // 1) 预处理:裁历史、压缩上下文、预取 memory 和 skills
    // 2) 调 Claude API(流式)
    // 3) 一边收流一边看有没有 tool_use block
    // 4) 有 tool_use → 检查权限 → 执行 → 结果塞回 messages → 继续 while
    // 5) 没有 tool_use → 退出
  }
}
```

退出条件:模型这一轮没调工具,纯输出 text,就算 done。

这个循环就是 ReAct 范式的工程化极致。从 [ReAct 论文](https://arxiv.org/abs/2210.03629) 到 [Toolformer](https://arxiv.org/abs/2302.04761) 到这里,本质没变——模型输出 action,环境返回 observation,append 回 context,再喂给模型。Claude Code 把这个循环的每个环节都优化到了极致。

为什么用 `async function*` generator?因为流式。Claude API 用 SSE 流式返回,每个 `tool_use` block 在流到 `content_block_stop` 时就能解析,立刻 dispatch 给 executor,不用等整个 assistant turn 结束。Generator 让调用方(React/Ink 层)能 incremental 渲染,同时 loop 内部保持 single-threaded 状态机语义。

有个隐式 invariant:**`tool_use_id` 是 executor 的 primary key**。Claude API 要求每个 `tool_use` 必须有对应的 `tool_result`,否则下一轮 400 报错。所以 loop 必须保证不丢 block。源码里有个 `MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3`,撞到 max_output_tokens 时悄悄重试最多 3 次,对用户无感——就是为了避免输出被截断导致 `tool_use` JSON 不完整,污染 state。

这段代码上面有段注释,模仿中世纪巫师口吻:

> Heed these rules well, young wizard. For they are the rules of thinking, and the rules of thinking are the rules of the universe. If ye does not heed these rules, ye will be punished with an entire day of debugging and hair pulling.

维护这块的人显然被坑过很多次。

---

## 上下文压缩:四把手术刀,一刀切

这是整篇源码里我最喜欢的部分。

用过 Claude Code 的人都知道,长对话到后面它会自动"压缩"。我之前以为就是把早期对话摘要一下。读了源码才发现,它有四种不同粒度的压缩机制同时工作,按顺序触发,前面能搞定就不触发后面。

### 第一层:HISTORY_SNIP —— 直接删

最粗暴的一层。某个工具返回了 500 行搜索结果,模型只用了其中 3 行。剩下 497 行就是纯噪声。

留着?浪费 token。摘要它?也浪费 token——summarizer 要调一次 LLM,丢掉原始信息熵,换回来一段可能还用不上的摘要。

直接删最划算。删的是"模型已经消费过且不会再 query 的 chunk"。

这里有个信息论直觉:**一块 token 的 expected future query probability 还有多高?** 低就 snip,高就保留或 collapse。

### 第二层:Microcompact —— 改 cache,不改 message

这层最聪明。利用 Anthropic API 的 prompt caching 能力。

Anthropic 的 cache 机制:你在 message 里打 `cache_control: {type: "ephemeral"}` breakpoint,API 侧会把前面的内容 cache 住,下次命中就便宜 10 倍。cache 有 TTL(5 分钟或 1 小时)。

计费分四类:
- `cache_creation_input_tokens`:首次写入
- `cache_read_input_tokens`:命中
- `input_tokens`:未 cache 的部分
- **`cache_deleted_input_tokens`**:曾经 cache 但本次请求里没出现的部分

Microcompact 不改 message content,而是改 `cache_control` breakpoint 的位置。把某些 block 从 "active cache" 里标记为 deleted,API 不再 charge 这些 token,但 model 端依然能"看到"完整 history——因为 cache 里有,只是计费层面不收钱了。

这是 **billing-side optimization,context-side 没动**。短期会话里 cache 还活着,把 token 费用降下来,model 行为不变。

参考: [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

### 第三层:CONTEXT_COLLAPSE —— 归档,保留结构

把旧的对话轮次"归档"成摘要,但维护一个类似 git log 的结构。哪一轮做了什么事、结论是什么,都还在,不是一坨糊在一起的摘要。

这非常像 [MemGPT](https://arxiv.org/abs/2310.08560) 的 hierarchical memory,也像 [LLM compilers](https://arxiv.org/abs/2312.02291) 里 "graph of prompts" 的思路。保留一个 DAG,节点是 turn summary,边是因果或时序关系。每次新 query 时从某个 checkpoint replay。

直觉:**collapse 是 "archive with structure",autocompact 是 "compress with loss"**。前者保留可恢复性,后者是 destructive。

### 第四层:Autocompact —— 兜底,整体压缩

最后的兜底。调一次 model 把整个历史压成一段话。到这层说明前面三层都没压到阈值以下,context window 快爆了。

大部分时候这层根本不需要跑。前面三层按信息"保质期"分层处理,已经把体积控制住了。

### 四层总结

| Layer | 做什么 | 触发条件 | 信息损失 |
|-------|--------|----------|----------|
| HISTORY_SNIP | 直接删掉模型没用的工具输出 | 工具输出大但只用了一小部分 | 删的是"消费过的噪声" |
| Microcompact | 改 cache breakpoint,不改 content | cache 命中率下降 | 零(model 还能看到) |
| CONTEXT_COLLAPSE | 归档为结构化摘要 | 旧 turn 累积体积过大 | 保留结构,丢失细节 |
| Autocompact | 整体调一次 LLM 压成一段话 | 接近 context window 上限 | 丢失结构 + 细节 |

做 agent 的上下文管理不能只有一种策略。工具的中间输出可能之后就没用了,但用户描述的需求背景可能整个会话都要保留。信息的"保质期"不一样,处理方式也该不一样。

---

## 流式工具并行:模型还在说话,工具就已经在跑了

一般 agent 的实现:等模型说完 → 看有没有工具调用 → 执行 → 结果返回 → 下一轮。中间有明显等待。

Claude Code 的 `StreamingToolExecutor` 做法不同:模型流式吐出一个 `tool_use` block,立刻开始执行。模型还在流后面的内容,前面的工具已经在跑了。

```
时间线:
t=0 ──────────────────────────────────────────►
   │ text tokens │ tool_use_1 │ text │ tool_use_2 │
                  ▲                    ▲
                  │                    │
              spawn T1              spawn T2 (如果只读)
                  │                    │
              T1 执行中 ──────┐    T2 执行中 ──────┐
                                 ▼                    ▼
                          result_1 (排队)     result_2 (排队)
```

每个工具有个 `isConcurrencySafe` 标记:
- `true`:只读操作(Read、Grep、Glob),可以并行跑
- `false`:写操作(Edit、Write、Bash),独占,必须串行

结果按 `tool_use_id` 接收顺序 buffer,保证 message 顺序 deterministic。

延迟对比:

$$L_{\text{naive}} = L_{\text{model}} + L_{\text{tool}}$$

$$L_{\text{streaming}} = \max(L_{\text{model}}, L_{\text{tool}})$$

变量含义:
- $L_{\text{model}}$:模型这一轮推理时间(典型 8 秒)
- $L_{\text{tool}}$:工具执行时间(典型 grep 30ms,bash 可能几秒)

当 $L_{\text{tool}} \ll L_{\text{model}}$,用户感知不到工具延迟。工具执行的时间被藏在了模型推理的时间里。这就是为什么 Claude Code 用户反馈"工具执行响应比较快"。

这个 pattern 在 [vLLM 的 continuous batching](https://arxiv.org/abs/2309.06180) 里有类似影子:把慢的部分(GPU forward)和快的部分(CPU scheduling)overlap 起来。

---

## 工具系统:40 多个工具,零继承

做过 agent 框架的人都习惯写一个 `BaseTool` 基类然后继承。Claude Code 完全没有继承,40 多个工具全是纯函数式的 `buildTool()` 工厂函数。

每个 tool 是一个 self-contained record:

```typescript
type ToolDef<T> = {
  name: string
  description: string
  inputSchema: ZodSchema<T>        // Zod v4 做校验 + 自动生成 JSON Schema
  call(input: T, ctx: ToolUseContext): AsyncGenerator<...>
  isReadOnly(): boolean
  getPermissions(): ToolPermission[]
  renderToolUse?(input: T): ReactNode          // 直接渲染到终端
  getToolUseSummary?(input, result): string    // 压缩上下文时的摘要
}
```

每个工具完全自包含:schema、权限、执行逻辑、UI 渲染、压缩摘要,全在一个文件。没有全局注册表,每个 session 动态组装工具池——静态工具、MCP 工具、Agent 定义混在一起用。

为什么选这个设计?因为 [MCP(Model Context Protocol)](https://modelcontextprotocol.io/) 天然就是 JSON Schema-based 的。MCP server 暴露的 tool 就是 `{name, description, inputSchema, handler}`。Claude Code 把内部 tool 和 MCP tool 用同一个 `ToolDef` 类型表示,session 启动时混合组装。

[Zod](https://zod.dev) v4 在这里干两件事:runtime validation + 自动 JSON Schema 生成。一条 schema 定义,两端用——本地校验、API 侧 model tool definition。

直觉:**ADT + factory 是 "tool as data",把 tool 变成可序列化、可传输、可远端化的 first-class value**。这正是 MCP 设计哲学的延伸。

---

## Feature Flag 双层:编译时消失 vs 运行时灰度

### 编译时:DCE(Dead Code Elimination)

```javascript
import { feature } from 'bun:bundle'
const voiceModule = feature('VOICE_MODE')
  ? require('./voice/index.js')
  : null
```

`feature()` 是 Bun 的 compile-time macro,build 时替换成 `true` 或 `false` 字面量。然后 minifier 做 reachability analysis,把 `false` 分支物理删除——从 binary 里抹掉,连字符串字面量都不剩。

为什么这么极端?因为 reverse engineering 的第一步是 `strings` 命令。安全研究员会反编译你的 binary 找隐藏功能。运行时 flag 再怎么关,字符串还在那里。编译时 DCE 才是真的"不存在"。

Anthropic 还有个 `excluded-strings.txt` 黑名单,构建系统会 grep binary 产物里有没有黑名单里的字符串。宠物系统里 18 个物种名,其中 `capybara` 用 hex 编码绕过这个 scan——因为 `capybara` 恰好是某个未公开模型的内部代号,不能出现在 binary 里。

讽刺的是,他们在 binary 层面做了这么多防护,最后被一个忘删的 `.map` 文件全部端掉了。

参考: [Bun bundler macros](https://bun.sh/docs/bundler/macros)

### 运行时:GrowthBook / Statsig

```javascript
const enabled = checkStatsigFeatureGate_CACHED_MAY_BE_STALE(
  'tengu_streaming_tool_execution'
)
```

所有 gate 名都以 `tengu_` 前缀。**tengu(天狗)** 是 Claude Code 项目的内部代号。

`_CACHED_MAY_BE_STALE` 后缀说明从本地磁盘 cache 读,接受 stale read,不阻塞 startup。Statsig SDK 默认会做 network fetch,会拖慢启动。Claude Code 为冷启动优化,接受脏读。

### 两层分工

| 层级 | 防什么 | 切换粒度 | 部署成本 |
|------|--------|----------|----------|
| 编译时 | 防被外人看见未发布功能 | 整个模块 | 重新 build + 发 npm |
| 运行时 | 防上线后出问题 | 单个 gate | Statsig 后台秒级生效 |

编译时 flag 是 security boundary,运行时 flag 是 operational boundary。

---

## ABLATION_BASELINE:把 ML 研究方法搬进产品代码

```javascript
if (feature('ABLATION_BASELINE') && process.env.CLAUDE_CODE_ABLATION_BASELINE) {
  for (const k of [
    'CLAUDE_CODE_DISABLE_THINKING',
    'DISABLE_COMPACT',
    'DISABLE_AUTO_COMPACT',
    'CLAUDE_CODE_DISABLE_AUTO_MEMORY',
    'CLAUDE_CODE_DISABLE_BACKGROUND_TASKS',
  ]) { process.env[k] = '1' }
}
```

一次性关掉五个核心子系统:thinking、compact、auto-compact、auto-memory、background-tasks。

做过 ML 研究的人都熟悉 ablation study:逐个关掉组件看对最终效果的影响。但把这个方法论搬到产品工程上,在工业代码里我也是第一次见。

这意味着 Anthropic 每上线一个新功能,都可以跑一组对照实验量化它的价值。形式化:

$$\Delta_i = \text{score}(\text{full}) - \text{score}(\text{full} \setminus \{i\})$$

变量含义:
- $\text{score}(\cdot)$:某个量化指标(任务完成率、用户满意度、token 效率等)
- $\text{full}$:所有功能全开
- $\text{full} \setminus \{i\}$:关掉第 $i$ 个功能
- $\Delta_i$:第 $i$ 个功能的边际贡献

$\Delta_i$ 就是第 $i$ 个子系统的边际贡献。LLM 应用的 feature 之间耦合太多,thinking + compact 的交互效应可能比单独加 thinking 还大。没有 ablation infrastructure,就没法做 attribution。这是 Anthropic 把 ML research rigor 带到 production engineering 的标志。

参考: [Ablation studies in ML](https://arxiv.org/abs/2102.11450), [Statsig experimentation](https://docs.statsig.com/experiments)

---

## BashTool:一个工具顶一个小型框架

文章说 BashTool 1143 行。一个工具为什么这么复杂?因为它要处理:

- **shell quoting**:POSIX 词法分析,处理引号、转义、变量替换
- **复合命令拆分**:`ls && git push` 会被拆开逐段判定安全性
- **命令分类**:read-only vs write,决定能否并发
- **沙箱规则生成**:根据命令内容生成 sandbox profile
- **输出截断**:大输出存磁盘,只给 model 一个文件路径引用
- **后台化**:超过 15 秒的阻塞命令自动转后台
- **内置 sed 解析器**:检测到 `sed -i` 时 UI 从 "Bash" 变成文件编辑样式,提取 file path 给用户预览 diff

### 沙箱实现

- **macOS**:`sandbox-exec`,基于 Apple Seatbelt sandbox,profile 是 Scheme-like DSL
- **Linux**:`seccomp-BPF`,用 eBPF filter 在 syscall 层拦截

macOS 的 profile 大致:

```
(allow file-read* (subpath "/usr"))
(deny file-write* (subpath "/System"))
(allow process-exec (literal "/bin/ls"))
(deny process-exec (literal "/bin/rm"))
```

Linux seccomp 的 filter 大致:

```
ALLOW: read, write, open, close, mmap, brk, ...
DENY:  ptrace, mount, unshare, setuid, ...
DENY:  socket, connect, bind, listen (除非白名单)
```

seccomp 比 namespace 更轻量,只做 syscall filter 不做隔离。Claude Code 选 seccomp 可能是因为它不需要 process 隔离(不跑 untrusted code),只需要阻止"危险动作"。

参考: [Apple Sandbox profiles](https://developer.apple.com/library/archive/technotes/tn2206/), [seccomp-BPF](https://www.kernel.org/doc/Documentation/prctl/seccomp_filter.txt)

---

## 隐藏功能:源码里还没发布的东西

编译时 flag 门控住的功能,虽然公开版 binary 看不到,但源码全暴露了。

### Voice Mode(代号 Amber Quartz)

`src/voice/` 目录确认了语音模式:
- 只支持 Claude.ai OAuth 认证(API key、Bedrock、Vertex 都不行)
- 走专门的 `voice_stream` 端点
- 有紧急 kill switch: `tengu_amber_quartz_disabled`
- 从注释看已经开发完成,只是还没公开

### Bridge Mode:把你的电脑变成 Claude 的远程终端

`src/bridge/` 有 31 个文件,实现了一个完整的远程控制系统。运行 `claude remote-control` 之后,本地环境就变成一个可以被 claude.ai 远程操控的"桥接环境"。

最多支持 32 个并发会话,JWT 认证加可信设备机制,企业管理员可以通过策略禁用。架构大概:

```
claude.ai (browser) ──HTTPS──► Anthropic relay ──WSS──► local bridge (your machine)
                                                                   │
                                                                   ▼
                                                            local filesystem
                                                            local shell
                                                            local tools
```

这是 [reverse tunnel](https://ngrok.com/docs) 模式。让 claude.ai 网页版能直接操作你本地的开发环境,不用再手动复制粘贴代码。

### Buddy:终端里的电子宠物

最出人意料的发现。Claude Code 内置了一个完整的虚拟宠物系统,而且没有用 feature flag 门控,已经在每个用户的 binary 里了:

```javascript
// 18 种宠物
export const SPECIES = [
  duck, goose, blob, cat, dragon, octopus, owl, penguin,
  turtle, snail, ghost, axolotl, capybara, cactus, robot,
  rabbit, mushroom, chonk
] as const

// 5 级稀有度
export const RARITY_WEIGHTS = {
  common: 60, uncommon: 25, rare: 10, epic: 4, legendary: 1,
}

// RPG 式属性
export const STAT_NAMES = ['DEBUGGING', 'PATIENCE', 'CHAOS', 'WISDOM', 'SNARK']
```

伪随机数生成器从用户 ID 确定性计算,每个用户绑定一只,不能刷。

在一个 51 万行的严肃工程项目里发现一套完整的抽卡养宠系统,挺有意思的。

### 宠物名 hex 编码的秘密

18 个物种名全部用 hex 编码,一个都没用明文:

```javascript
const c = String.fromCharCode
export const duck = c(0x64, 0x75, 0x63, 0x6b) as 'duck'
export const capybara = c(0x63, 0x61, 0x70, 0x79, 0x62, 0x61, 0x72, 0x61) as 'capybara'
```

注释写着:"One species name collides with a model-codename canary in excluded-strings.txt."

其中一个宠物名恰好是 Anthropic 某个未公开模型的内部代号。构建系统会 grep binary 里有没有黑名单字符串,所以必须 hex 编码绕过检测。

到底是哪一个?结合另一次泄露来看。

参考: [Bun compile](https://bun.sh/docs/bundler/compile)

---

## 模型代号:两次泄露拼出完整拼图

### 第一次泄露(3 月 28 日)

npm 泄露的三天前,Anthropic 还出了另一件事:CMS 数据库权限没关,被 Fortune 记者翻出了近 3000 份内部文件。其中提到一个从未公开的模型叫 **Claude Mythos**,内部代号 **Capybara(水豚)**,定位在 Opus 之上的全新层级。

### 第二次泄露(3 月 31 日)

Claude Code 源码的 `prompts.ts` 里,大量 `@[MODEL LAUNCH]` 注释(给新模型发布准备的 TODO checklist),反复出现这个名字:

```javascript
// @[MODEL LAUNCH]: Update comment writing for Capybara
// - remove or soften once the model stops over-commenting by default

// @[MODEL LAUNCH]: False-claims mitigation for Capybara v8
// (29-30% FC rate vs v4's 16.7%)
```

这些 TODO 还没完成,说明 Capybara 还没公开发布。但 `main.tsx` 里已经有 `capybara-fast`、`capybara-v2-fast[1m]` 这些模型别名,说明 Anthropic 内部员工已经在日常使用了。

### 拼图

- **Capybara** = Claude Mythos,比 Opus 更强的下一代旗舰
- Opus 4.6 在代码里叫 `claude-opus-4-6`,没有动物名
- 宠物系统里撞车的就是这只水豚,所以 hex 编码

还泄露了一个内部数据:**Capybara v8 的 false-claim rate 29-30%,比上一版 v4 的 16.7% 翻了一倍**。Anthropic 没有回退版本,而是在 prompt 层加指令做修补,内部员工先当白鼠验证效果。

这说明 v8 在某些核心 axis(reasoning、coding、long-context)显著优于 v4,足以补偿 FC 的退化。也说明 Anthropic 把 FC 当作可调 axis 而非 hard constraint——这与 [Constitutional AI](https://arxiv.org/abs/2212.08073) 的 trade-off 思路一致。

源码里还出现了另一个模型代号:

```javascript
// @[MODEL LAUNCH]: Remove this section when we launch numbat.
```

**Numbat(袋食蚁兽)**,又一个待发布的模型。和 Capybara 的关系目前还不清楚。

### 动物命名谱系

Anthropic 的模型代号:Haiku(俳句)、Sonnet(十四行诗)、Opus(作品)——都是诗体。但内部研发代号转成动物:Capybara、Numbat、Mythos。**对外是文学意象,对内是动物代号**。

参考: [Claude model spec](https://www.anthropic.com/news/claude-model-spec), [Constitutional AI](https://arxiv.org/abs/2212.08073)

---

## Skills 系统:约定优于配置

`.claude/skills/` 目录下放 `.md` 文件,YAML frontmatter 里写描述、触发条件、允许的工具、用哪个模型。Claude Code 读文件时如果发现目录下有 skills,自动加载,不用显式注册。

这是 [Rails convention over configuration](https://rubyonrails.org/doctrine) 在 LLM agent 里的对应。Skills 不需要注册中心——filesystem 就是 registry。

skill 的 frontmatter 大致:

```yaml
---
name: refactor-react-component
description: Refactor a React component to use hooks
triggers:
  - "refactor this component"
  - "convert class to hooks"
allowed_tools: [Read, Edit, Grep]
model: capybara-fast
---
```

Claude Code 在 system prompt 里把所有 skill 的 `name + description` 列出来(类似 tool description),model 决定调用哪个时,把对应 `.md` 的 body 注入到 context。这是 **lazy skill loading**——只 load 触发的那个 skill 的 full content。

直觉:这是 [RAG over prompt library](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) 的特例。skill 描述是 retrieval key,skill body 是 retrieved document。

---

## Coordinator 多 Agent:极简工具集

Coordinator 模式下主 Agent 只有三个工具:
1. **TeamCreate**:生成 worker
2. **SendMessage**:给 worker 发消息
3. **StopWorker**:停止 worker

Worker 拿不到 `TeamCreate` 和 `SendMessage`,防止套娃——避免 worker spawn 子 worker,无限递归。

后端三种 worker runtime:
1. **tmux pane**:用 tmux 的 split-window + send-keys,让 worker 的 output 可见于终端
2. **in-process**:在同一个 Bun process 里 spawn query sub-loop
3. **remote**:通过 Bridge Mode 在另一台机器跑

这三种是 latency vs isolation 的 trade-off:tmux 最快但隔离最弱,remote 最慢但隔离最强。

参考: [AutoGen](https://arxiv.org/abs/2308.08155), [MetaGPT](https://arxiv.org/abs/2308.00352)

---

## 投机执行:CPU branch predictor 的 LLM 版

AppState 里有 `speculationState`,追踪每一轮的结束方式(bash / 文件编辑 / 正常结束 / 权限拒绝),用来预判下一步操作并提前执行。

这解释了为什么 Claude Code 有时候"想"完就瞬间开始干活——它在 model 还没输出完时就预 spawn 了 sandbox,或者预 read 了目标文件到 cache。

形式化:state $s_t \in \{\text{bash}, \text{edit}, \text{chat}, \text{deny}\}$。预测器 $\hat{a}_{t+1} = g(s_{t-k:t})$。

变量含义:
- $s_t$:第 $t$ 轮的 ending modality
- $g(\cdot)$:预测函数,可能是个简单 n-gram 或 Markov model
- $\hat{a}_{t+1}$:对下一轮 action 类别的预测

如果 $\hat{a}_{t+1} = \text{bash}$,预 spawn sandbox;如果 $\hat{a}_{t+1} = \text{edit}$,预 read 目标文件。

hit rate 高时 latency $\approx 0$。prediction miss 时 fallback 到正常路径,有点 overhead 但可接受。这是 [speculative decoding](https://arxiv.org/abs/2211.17192) 在 system layer 的对应——predict 的对象从 token 变成了 action。

---

## 冷启动优化:`--version` 零 import

`--version` 路径做到零 import,直接读编译时内联的版本号,一个模块都不加载就退出。其他子命令走独立的 `import()` 路径。只有最终进主循环才加载完整的 React 应用。

| 路径 | 加载内容 | 启动时间(估计) |
|------|----------|------------------|
| `claude --version` | 0 import | < 20ms |
| `claude --help` | 仅 dispatch module | ~50ms |
| `claude <prompt>` | React + Ink + tools + MCP | 200-500ms |

800KB 的 React app 不用就不加载。这对 CLI UX 至关重要——shell completion、脚本调用都依赖快速 `--version`。

参考: [Bun compile](https://bun.sh/docs/bundler/compile), [CLI startup time best practices](https://blog.rust-lang.org/inside-rust/2020/01/20/cli-startup-time.html)

---

## 隐私保护的类型名技巧

```typescript
type AnalyticsMetadata_I_VERIFIED_THIS_IS_NOT_CODE_OR_FILEPATHS = { ... }
```

用类型名本身做 developer nudge。TypeScript 编译后类型名消失(不像 runtime value),不影响 binary 体积。但 IDE hover、compiler error 里会强制显示这句话。

这是 **"naming as lint"** 的极致。等价于在 code review 里加一个无法 skip 的 checkbox。简单有效,zero runtime cost。

---

## 泄露原因总结 + npm 发布最佳实践

```
npm publish → .map 文件未删 → map 引用 R2 zip → R2 URL 无 auth → 全部源码下载
```

给所有发 npm 包的人提个醒:

1. **`package.json` 的 `files` 字段白名单制**,只包含你想发布的东西
2. **CI 里加一步检查发布产物有没有 `.map` 文件**:
   ```bash
   npm pack --dry-run | grep -E '\.map$' && exit 1
   ```
3. **源码归档 URL 要有鉴权**,别裸挂在 CDN 上
4. **构建产物和源码的访问控制应该独立管理**

---

## 给 Karpathy 的几个 takeaway

1. **Agentic loop 的本质是 message state machine**。$M_t \to M_{t+1}$ 的转移由 model + tool executor 协同完成。所有"工程优化"都是在减少这个转移的 cost 或增加其信息密度。

2. **Context compression 应该是 multi-strategy 的**。单一 summarizer 是 dead simple 但 suboptimal 的。Claude Code 的四层策略对应四种不同的 "information lifetime" 假设,每层做最 cost-effective 的操作。这和 GPU memory hierarchy(register / L1 / L2 / HBM)的思路惊人相似——离 model 越近的越贵越小越快。

3. **Streaming + concurrency 是 latency hiding 的核心**。把 $\max$ 替代 $\sum$,是 distributed systems 的经典操作。Claude Code 在 LLM layer 重现了这个 pattern。

4. **Feature flag 双层是 security + operational 的分离**。编译时 flag 防 reverse engineering,运行时 flag 防运营事故。这两件事不能混。

5. **Ablation infrastructure 是 LLM 工程成熟度的标志**。能跑 leave-one-out 的 team 才能做 attribution。否则永远在"感觉有用就上"的混沌状态。

6. **最简单的事故往往击穿最复杂的防御**。`.map` 文件 vs 编译时 DCE + hex 编码 + binary scan。这是 security 的 [weakest link principle](https://en.wikipedia.org/wiki/Weakest_link)。

---

## 相关开源资源

- 何宇峰用 950 行 Python 重写 Claude Code 核心:[he-yufeng/CoreCoder](https://github.com/he-yufeng/CoreCoder)
- Windy 的源码导读 + 从零构建教程:[Windy3f3f3f3f/how-claude-code-works](https://github.com/Windy3f3f3f3f/how-claude-code-works) 和 [Windy3f3f3f3f/claude-code-from-scratch](https://github.com/Windy3f3f3f3f/claude-code-from-scratch)
- Ink (TUI React):[vadimdemedes/ink](https://github.com/vadimdemedes/ink)
- Bun bundler macros:[bun.sh/docs/bundler/macros](https://bun.sh/docs/bundler/macros)
- Anthropic Prompt Caching:[docs.anthropic.com/en/docs/build-with-claude/prompt-caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- Model Context Protocol:[modelcontextprotocol.io](https://modelcontextprotocol.io/)
- Zod v4:[zod.dev/v4](https://zod.dev/v4)
- Statsig experimentation:[docs.statsig.com/experiments](https://docs.statsig.com/experiments)

---

## 还想深挖的点

1. **`prompts.ts` 全文**。文章只引了几行 `@[MODEL LAUNCH]` TODO,但整个 system prompt 的结构(persona、tool description 注入、skill description 注入、permission rule 注入)才是 Claude Code 行为的真正源头。我怀疑里面有大量 [prompt scaffolding](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) 的工程经验。

2. **`CONTEXT_COLLAPSE` 的数据结构**。文章说"维护一个类似 git log 的结构",但具体是 linked list、tree、还是 DAG?replay 时的 token cost 是多少?这个数据结构的设计直接决定了 long-context agent 的可恢复性。

3. **`speculationState` 的预测器形式**。是简单的 last-ending-modality n-gram,还是有更复杂的 Markov model?prediction miss 时的 fallback cost 是多少?这关系到 speculative execution 是否值得——如果 hit rate 不高,speculation overhead 反而是负优化。

4. **BashTool 1143 行的完整沙箱 profile**。文章只提了 sandbox-exec 和 seccomp,但具体 profile 长什么样?白名单怎么维护?命令分类的规则引擎是什么?

如果有这些 module 的源码或逆向分析,再聊。这次的泄露让 Anthropic 的 Claude Code 从"黑盒魔法"变成了"可学习的工程范例",对整个 agent engineering 社区都是好事。

---

# Claude Code 源码泄露：一份给 Karpathy 的技术深读

Andrej, 这篇泄露分析里真正有意思的，不是那些花边新闻（Capybara 代号、宠物系统、Bridge Mode），而是 Anthropic 在工程上展现的几处 "反直觉但正确" 的设计决策。我把每一条都拆给你看，并尽量补上对应的技术联想。

---

## 1. Agentic Loop：一个 `while(true)` 为什么能撑住 51 万行

文章说核心在 `src/query.ts`，1729 行，本质是一个 generator-based while loop。这其实是从 [ReAct](https://arxiv.org/abs/2210.03629) 到 [Toolformer](https://arxiv.org/abs/2302.04761) 路线的工程化极致。

用形式化的语言，每一轮的状态转移是：

$$M_{t+1} = M_t \oplus \text{concat}(a_t, r_t), \quad a_t \sim f_\theta(\cdot | M_t), \quad r_t = \mathcal{T}(a_t)$$

变量含义：
- $M_t \in \mathcal{M}$：第 $t$ 轮的 message list（Anthropic Messages API 格式）
- $a_t$：模型采样出的 assistant turn，可能包含若干 `text` block 和零或多个 `tool_use` block
- $r_t$：tool executor 对 $a_t$ 中每个 `tool_use` 的执行结果（`tool_result` block）
- $\oplus \text{concat}$：append 操作，保留顺序
- $f_\theta$：Claude 模型（推理时 $\theta$ 固定）
- $\mathcal{T}$：工具执行器，可能含沙箱、超时、并发调度

退出条件：$a_t$ 中 `tool_use` block 数量 $= 0$。这是 ReAct 的 stop condition 的工程化版本——纯 text 输出即 "I'm done"。

**为什么是 `async function*` generator 而非普通 `async`?** 因为流式。Claude API 用 SSE 流，每个 `tool_use` block 在 partial JSON 完整解析后（content_block_start → content_block_delta × N → content_block_stop）就可以立刻 dispatch 给 executor，无需等整个 assistant turn 结束。Generator 让调用方（React/Ink 层）能 incremental 渲染，同时让 query loop 内部保持 single-threaded 的状态机语义。

这里有个隐式的 invariant：**`tool_use_id` 是 executor 的 primary key**。Claude API 要求每个 `tool_use` 必须有对应的 `tool_result`，否则下一轮会 400。所以 loop 必须保证不丢 block。文章里那段 "扣留错误消息、悄悄重试 3 次" (`MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3`) 就是为了在 max_output_tokens 截断导致 `tool_use` JSON 不完整时，避免 orphan tool_use 污染 state。

参考：[Anthropic Messages API - tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use), [Anthropic streaming](https://docs.anthropic.com/en/api/messages-streaming)

---

## 2. 四层上下文压缩：信息论视角

这是整篇里我最喜欢的一段。先放一张表：

| Layer | 机制 | 触发条件 | 信息保留度 | 计算成本 |
|-------|------|----------|------------|----------|
| $L_1$ | HISTORY_SNIP | 工具输出 > 阈值且模型只用了一小部分 | 删除原 chunk，不摘要 | $O(1)$，纯截断 |
| $L_2$ | Microcompact | cache 命中率下降、token 接近 budget | 改 cache breakpoint，不改 message | $O(n)$，API 侧处理 |
| $L_3$ | CONTEXT_COLLAPSE | 旧 turn 完整但累积体积过大 | 归档为结构化摘要，保留 turn 拓扑 | $O(n)$，本地结构化 |
| $L_4$ | Autocompact | 接近 context window 上限 | 整体调一次 model 压成一段话 | $O(n)$ + 一次 LLM call |

waterfall 触发：$L_1 \to L_2 \to L_3 \to L_4$，前面能压到阈值以下就不触发后面。

### 2.1 HISTORY_SNIP 的信息论依据

工具返回 $r_t$ 是一段 text。模型在后续 turn 中引用它的概率 $P(\text{ref} | M_{t+k})$ 随 $k$ 衰减。被实际消费的部分的 self-information：

$$I_{\text{consumed}} = -\sum_i p_i \log p_i, \quad i \in \text{used lines}$$

未消费部分的 conditional entropy $H(r_t | \text{used})$ 在很多 case 下接近原值，即信息还在但模型不会 query 它。这时摘要反而是负优化——summarizer 会丢失原始信息熵，且消耗 LLM call。所以直接 snip 是 minimum-cost / maximum-information-retention 的 Pareto 最优。

**直觉：snip vs summarize 的选择，本质是问"这块 token 的 expected future query probability 还有多高"**。低就 snip，高就 collapse。

### 2.2 Microcompact 与 Anthropic Prompt Caching

Anthropic 的 prompt caching 在 API 侧把 prompt 切成 cache block，每个 block 有 `cache_control: {type: "ephemeral"}`。计费上分四类：
- `cache_creation_input_tokens`：首次写入
- `cache_read_input_tokens`：命中
- `input_tokens`：未 cache 的部分
- **`cache_deleted_input_tokens`**：曾经 cache 但本次请求中没出现的部分（这是关键）

Claude Code 的 Microcompact 不改 message content，而是改 `cache_control` breakpoint 的位置——把某些 block 从 "active cache" 里"标记为 deleted"，让 API 不再 charge 它们，但 model 端依然能"看到"完整 history（因为 cache 里有，只是不重新 charge）。

这里有个 subtle 的点：Anthropic 的 cache 是 ephemeral（5 分钟或 1 小时 TTL），不是永久存储。Microcompact 利用了这个窗口——短期会话里 cache 还活着，把 token 计费降下来，model 行为不变。这是 **billing-side optimization 而非 context-side optimization**。

参考：[Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), [Cache deletion behavior](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#cache-deletion)

### 2.3 CONTEXT_COLLAPSE vs Autocompact：结构保留的差异

Autocompact 是 $M \to \text{LLM} \to \text{paragraph}$，丢失结构。

CONTEXT_COLLAPSE 保留一个 DAG $\mathcal{G} = (V, E)$，其中 $V$ 是 turn 节点（含 summary），$E$ 是因果或时序边。每次新 query 时从某个 checkpoint replay。这非常像 [LLM compilers](https://arxiv.org/abs/2312.02291) 里 "graph of prompts" 的思路，也像 [MemGPT](https://arxiv.org/abs/2310.08560) 的 hierarchical memory。

**直觉：collapse 是 "archive with structure"，autocompact 是 "compress with loss"**。前者保留可恢复性，后者是 destructive。把 autocompact 放最后一道防线，意味着 Anthropic 把 "信息损失" 当作 last resort。

---

## 3. StreamingToolExecutor：Latency Hiding 的工程化

文章给的核心代码片段：

```typescript
class StreamingToolExecutor {
  addTool(block: ToolUseBlock, message: AssistantMessage): void { ... }
  async *getRemainingResults(): AsyncGenerator<MessageUpdate> { ... }
}
```

这是经典的 **pipeline parallelism + ordered buffer**。用 token stream 的时间线拆开：

```
t=0 ───────────────────────────────────────────────────►
   │ text tokens │ tool_use_1 │ text │ tool_use_2 │ text │
                  ▲                    ▲
                  │                    │
              spawn T1              spawn T2 (if concurrency-safe)
                  │                    │
              ── T1 executing ──┐   ── T2 executing ──┐
                                  ▼                      ▼
                          result_1 (ordered)    result_2 (ordered)
```

关键 invariant：
1. `tool_use` block 在 SSE `content_block_stop` 时即可解析（partial JSON parser），不必等整个 assistant turn 结束
2. `isConcurrencySafe = true` 的工具（read-only: `Read`, `Grep`, `Glob`）可并发跑
3. `isConcurrencySafe = false` 的工具（write: `Edit`, `Write`, `Bash`）独占，必须串行
4. 结果按 `tool_use_id` 接收顺序 buffer，保证 message 顺序 deterministic

延迟公式：

$$L_{\text{naive}} = L_{\text{model}} + L_{\text{tool}}, \quad L_{\text{streaming}} = \max(L_{\text{model}}, L_{\text{tool}})$$

当 $L_{\text{tool}} \ll L_{\text{model}}$（典型：grep 30ms vs Claude 8s），用户感知不到工具延迟。这就是为什么 Claude Code 用户反馈"工具执行响应比较快"——延迟被 model 的 reasoning 时间吸收了。

这种 pattern 在 [vLLM 的 continuous batching](https://arxiv.org/abs/2309.06180) 里有类似的影子：把慢的部分（GPU forward）和快的部分（CPU scheduling）overlap 起来。

---

## 4. Tool 系统的函数式 ADT：为什么零继承

文章给的核心 type：

```typescript
type ToolDef<T> = {
  name: string
  description: string
  inputSchema: ZodSchema<T>
  call(input: T, ctx: ToolUseContext): AsyncGenerator<...>
  isReadOnly(): boolean
  getPermissions(): ToolPermission[]
  renderToolUse?(input: T): ReactNode
  getToolUseSummary?(input, result): string
}
```

这是 **Product type**（不是 Sum type），每个 tool 是 self-contained record。对比传统 OOP 的 `class BashTool extends BaseTool`：

| 维度 | OOP 继承 | 函数式 record |
|------|----------|---------------|
| 共享逻辑 | `super` 调用，隐式 | 组合（compose），显式 |
| 类型安全 | `this` 类型模糊 | T 泛型精确 |
| 序列化 | 实例不能直接 JSON | record 可直接 to JSON Schema |
| Hot reload | class 重新加载麻烦 | factory function 容易 |
| MCP 集成 | 需要适配器层 | 直接构造 record |

为什么 Anthropic 选后者？因为 **MCP（Model Context Protocol）天然就是 JSON Schema-based 的**。MCP server 暴露的 tool 就是 `{name, description, inputSchema, handler}`。Claude Code 把内部 tool 和 MCP tool 用同一个 `ToolDef` 类型表示，session 启动时混合组装：

```
internal tools (静态) ─┐
MCP tools (动态发现) ─┼──► ToolPool[session_id]
Agent-defined tools ──┘
```

[Zod](https://zod.dev) v4 在这里干两件事：runtime validation + 自动 JSON Schema 生成。一条 schema 定义，两端用——本地校验、API 侧 model tool definition。

**直觉：ADT + factory 是 "tool as data" 而非 "tool as object"。这让 tool 变成可序列化、可传输、可远端化的 first-class value**。这正是 MCP 设计哲学的延伸。

参考：[Model Context Protocol spec](https://modelcontextprotocol.io/), [Zod v4 release](https://zod.dev/v4)

---

## 5. Feature Flag 双层：编译时 DCE vs 运行时 Gate

### 5.1 编译时 DCE（Dead Code Elimination）

```javascript
import { feature } from 'bun:bundle'
const voiceModule = feature('VOICE_MODE')
  ? require('./voice/index.js')
  : null
```

`feature()` 是 Bun 的 compile-time macro，build 时替换为 `true` / `false` 字面量。然后 Bun 的 minifier 做 reachability analysis，把 `false ? ... : null` 的 dead branch 物理删除——从 binary 里抹掉，连字符串字面量都不剩。

为什么这么极端？因为 **reverse engineering 的第一步是 `strings` 命令**。Anthropic 的 `excluded-strings.txt` 黑名单 + binary 字符串扫描，确保未发布功能的任何痕迹（函数名、字符串、配置 key）都不出现在公开 binary 里。Hex 编码宠物名 `capybara` 就是为了绕过自家 binary scanner——`excluded-strings.txt` 里有 "capybara"，但 hex 编码后的 `0x63, 0x61, ...` 不会被 grep 到。

参考：[Bun bundler macros](https://bun.sh/docs/bundler/macros), [Bun compile-time features](https://bun.sh/docs/bundler/compile)

### 5.2 运行时 GrowthBook/Statsig Gate

```javascript
const enabled = checkStatsigFeatureGate_CACHED_MAY_BE_STALE('tengu_streaming_tool_execution')
```

所有 gate 名都以 `tengu_` 前缀——**天狗** 是 Claude Code 的内部代号（"tengu" 在日本神话里是长鼻子的山神，可能暗指 long-context 或某种内部梗）。

`_CACHED_MAY_BE_STALE` 后缀说明从本地磁盘 cache 读，接受 stale read，不阻塞 startup。这是 **cold start optimization** 的一部分——Statsig SDK 默认会做 network fetch，会拖慢启动。

### 5.3 两层 flag 的分工

| 层级 | 用途 | 切换粒度 | 部署成本 |
|------|------|----------|----------|
| 编译时 | 完全未发布、不可泄露的功能 | 整个模块 | 重新 build + 发布 npm |
| 运行时 | 灰度发布、紧急 kill switch | 单个 gate | Statsig 后台配置，秒级生效 |

**直觉：编译时 flag 是 security boundary，运行时 flag 是 operational boundary。前者防"被外人看见"，后者防"上线后出问题"**。

---

## 6. ABLATION_BASELINE：把 ML 研究方法搬到产品工程

```javascript
if (feature('ABLATION_BASELINE') && process.env.CLAUDE_CODE_ABLATION_BASELINE) {
  for (const k of [
    'CLAUDE_CODE_DISABLE_THINKING',
    'DISABLE_COMPACT',
    'DISABLE_AUTO_COMPACT',
    'CLAUDE_CODE_DISABLE_AUTO_MEMORY',
    'CLAUDE_CODE_DISABLE_BACKGROUND_TASKS',
  ]) { process.env[k] = '1' }
}
```

一次性关掉五个核心子系统。这意味着 Anthropic 内部可以跑：

$$\text{score}(\text{full}) \quad \text{vs} \quad \text{score}(\text{ablation})$$

然后逐个开回去做 leave-one-out：

$$\Delta_i = \text{score}(\text{full}) - \text{score}(\text{full} \setminus \{i\})$$

$\Delta_i$ 就是第 $i$ 个子系统（thinking / compact / memory / background）的边际贡献。这是 ML ablation study 的标准方法，Anthropic 把它工程化到产品代码里。

**为什么这件事重要？** 因为 LLM 应用的 feature 系统太多了，彼此耦合。thinking + compact 的交互效应可能比单独加 thinking 还大。没有 ablation infrastructure，就没法做 attribution。这是 Anthropic 把 ML research rigor 带到 production engineering 的标志。

参考：[Ablation studies in ML](https://arxiv.org/abs/2102.11450), [Statsig experimentation](https://docs.statsig.com/experiments)

---

## 7. BashTool：沙箱的操作系统级实现

文章提到：
- macOS 走 `sandbox-exec`
- Linux 走 `seccomp`
- > 15 秒阻塞命令转后台
- 大输出存磁盘
- 内置 `sed` 解析器
- 复合命令拆段判安全

### 7.1 sandbox-exec（macOS）

macOS 的 `sandbox-exec` 是 Seatbelt sandbox 的 CLI 入口，基于 [Apple Sandbox](https://developer.apple.com/library/archive/technotes/tn2206/) profile。Profile 是 Scheme-like DSL：

```
(allow file-read* (subpath "/usr"))
(deny file-write* (subpath "/System"))
(allow process-exec (literal "/bin/ls"))
(deny process-exec (literal "/bin/rm"))
```

Claude Code 大概率用了一个白名单 profile：允许读项目目录、读系统 lib、禁止写系统目录、禁止 network egress（除非白名单）。

### 7.2 seccomp（Linux）

[seccomp-BPF](https://man7.org/linux/man-pages/man2/seccomp.2.html) 用 eBPF filter 在 syscall 层拦截。典型 Claude Code 的 filter 应该是：

```c
// 伪码
ALLOW: read, write, open, close, mmap, brk, ...
DENY:  ptrace, mount, unshare, setuid, ...
DENY:  socket, connect, bind, listen (除非白名单 host)
```

seccomp 比 namespace 更轻量，只做 syscall filter 不做隔离。Claude Code 选 seccomp 可能是因为它不需要 process 隔离（不跑 untrusted code），只需要阻止"危险动作"。

### 7.3 内置 sed 解析器

为什么自己写 sed parser？因为 `sed -i` 会改文件，UI 要从 "Bash" 切到 "Edit" 样式，且要 extract 出 file path 给用户预览 diff。正则匹配 sed 命令提取 `s/pattern/replacement/flags` 和目标文件，是 AST-level 的解析。

文章说 BashTool 1143 行，这个复杂度合理——它要处理：
- shell quoting（POSIX 词法分析）
- pipe / && / ; 拆分
- 环境变量替换
- 命令分类（read-only vs write）
- 沙箱规则生成
- 输出截断
- 后台化

参考：[Apple Sandbox profiles](https://developer.apple.com/library/archive/technotes/tn2206/), [seccomp-BPF](https://www.kernel.org/doc/Documentation/prctl/seccomp_filter.txt), [POSIX shell grammar](https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V7_chap2.html)

---

## 8. 投机执行 speculationState

> AppState 里有 speculationState，追踪每一轮的结束方式（bash / 文件编辑 / 正常结束 / 权限拒绝），用来预判下一步操作并提前执行。

这是 **CPU branch predictor 的 LLM 版**。把每轮的 "ending modality" 当作 branch history，预测下一轮的 action 类别。

形式化：state $s_t \in \{\text{bash}, \text{edit}, \text{chat}, \text{deny}\}$。预测器 $\hat{a}_{t+1} = g(s_{t-k:t})$。

如果 $\hat{a}_{t+1} = \text{bash}$，可以预 spawn sandbox；如果 $\hat{a}_{t+1} = \text{edit}$，可以预 read 目标文件到 cache。

hit rate 高时 latency $\approx 0$。这是 [speculative decoding](https://arxiv.org/abs/2211.17192) 在 system layer 的对应——不是 predict token，是 predict action。

---

## 9. Bridge Mode：claude.ai 操作本地环境

> 运行 `claude remote-control` 之后，本地变成 claude.ai 远程操控的"桥接环境"。最多 32 个并发会话，JWT 认证 + 可信设备机制。

这是 [reverse tunnel](https://github.com/rgrok/s) 模式：本地起 server，claude.ai 通过 Anthropic 中继连过来。架构图大致：

```
claude.ai (browser) ──HTTPS──► Anthropic relay ──WSS──► local bridge (your machine)
                                                                   │
                                                                   ▼
                                                            local filesystem
                                                            local shell
                                                            local tools
```

32 并发会话上限可能是因为 Anthropic 中继的 connection pool 限制，或者单机 file descriptor 限制。

这种设计的安全考量：
1. JWT 短 TTL（防 replay）
2. Trusted device 注册（防 MITM）
3. 管理员 policy 可禁用（企业场景）
4. 所有操作走本地 Claude Code 的同一套 permission system

参考：[ngrok 类 reverse tunnel](https://ngrok.com/docs), [JWT RFC 7519](https://datatracker.ietf.org/doc/html/rfc7519)

---

## 10. 隐藏模型代号：Capybara / Numbat

文章拼出两条信息：

1. **Capybara（水豚）= Claude Mythos**，定位 Opus 之上的新旗舰。内部别名 `capybara-fast`、`capybara-v2-fast[1m]` 已在 `main.tsx`。Capybara v8 false-claim rate 29-30%，v4 是 16.7%。
2. **Numbat（袋食蚁兽）= 另一个待发布模型**。

### 10.1 false-claim rate 翻倍但没回退——为什么？

v4 FC=16.7%, v8 FC=29-30%。翻倍。Anthropic 选择不回退 v4，而是：
- 在 `prompts.ts` 加 `@[MODEL LAUNCH]: False-claims mitigation for Capybara v8` TODO
- 用 prompt 层指令"软化"模型行为
- 内部员工日常用 v8 做 dogfooding 验证

这说明 v8 在某些核心 axis（可能是 reasoning、coding、long-context）显著优于 v4，足以补偿 FC 的退化。也说明 Anthropic 把 FC 当作可调 axis 而非 hard constraint——这与 [Constitutional AI](https://arxiv.org/abs/2212.08073) 的 trade-off 思路一致。

### 10.2 动物命名谱系

Anthropic 的模型代号：Haiku（俳句）、Sonnet（十四行诗）、Opus（作品）——都是诗体。但内部研发代号转成动物：Capybara、Numbat、Mythos。这是 **two-tier naming**：对外是文学意象，对内是动物代号。

宠物系统里 18 个物种，其中 `capybara` 与 model codename 碰撞，所以 hex 编码绕 binary scan。其他 17 个动物明文，唯独 capybara hex——这间接证实了 capybara 是高敏感词。

参考：[Claude model naming](https://www.anthropic.com/news/claude-model-spec), [Constitutional AI](https://arxiv.org/abs/2212.08073), [Anthropic model card](https://www.anthropic.com/news/claude-3-model-card)

---

## 11. 冷启动优化：`--version` 零 import

> `--version` 路径做到了零 import，直接读编译时内联的版本号，一个模块都不加载就退出。

这是经典的 [lazy initialization](https://blog.cloudflare.com/cloudflare-architecture-and-how-bpf-eats-the-world/) pattern。Bun 编译时把 version 字符串 inline 到 binary entry point，subcommand dispatch 用 process.argv 直接 switch，不加载 React/Ink/Zod。

启动时间分解：

| 路径 | 加载内容 | 启动时间（估计） |
|------|----------|------------------|
| `claude --version` | 0 import | < 20ms |
| `claude --help` | 仅 dispatch module | ~50ms |
| `claude <prompt>` | React + Ink + tools + MCP | 200-500ms |

800KB 的 React app 不用就不加载。这对 CLI UX 至关重要——shell completion、脚本调用都依赖快速 `--version`。

参考：[Bun compile](https://bun.sh/docs/bundler/compile), [CLI startup time best practices](https://blog.rust-lang.org/inside-rust/2020/01/20/cli-startup-time.html)

---

## 12. 隐私保护的类型名技巧

```typescript
type AnalyticsMetadata_I_VERIFIED_THIS_IS_NOT_CODE_OR_FILEPATHS = { ... }
```

用类型名本身做 developer nudge。TypeScript 编译后类型名消失（不像 runtime value），不影响 binary 体积。但 IDE hover、compiler error 里会强制显示这句话。

这是 **"naming as lint"** 的极致。等价于在 code review 里加一个无法 skip 的 checkbox。简单有效，且 zero runtime cost。

---

## 13. Skills 系统：约定优于配置

> `.claude/skills/` 目录下放 `.md` 文件，YAML frontmatter 写描述、触发条件、允许的工具、用哪个模型。Claude Code 读到目录下有 skills 自动加载。

这是 [Rails convention over configuration](https://rubyonrails.org/doctrine) 在 LLM agent 里的对应。Skills 不需要注册中心——filesystem 就是 registry。

skill 的 frontmatter 大致是：

```yaml
---
name: refactor-react-component
description: Refactor a React component to use hooks
triggers:
  - "refactor this component"
  - "convert class to hooks"
allowed_tools: [Read, Edit, Grep]
model: capybara-fast
---
```

Claude Code 在 system prompt 里把所有 skill 的 `name + description` 列出来（类似 tool description），model 决定调用哪个时，Claude Code 把对应 `.md` 的 body 注入到 context。这是 **lazy skill loading**——只 load 触发的那个 skill 的 full content。

直觉：这是 [RAG over prompt library](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) 的特例。skill 描述是 retrieval key，skill body 是 retrieved document。

---

## 14. Coordinator 多 Agent：极简工具集

> Coordinator 模式下主 Agent 只有三个工具：生成 worker、给 worker 发消息、停止 worker。Worker 拿不到 TeamCreate 和 SendMessage，防止套娃。

形式化：
- Coordinator $C$ 的工具集 $\mathcal{T}_C = \{\text{TeamCreate}, \text{SendMessage}, \text{StopWorker}\}$
- Worker $W$ 的工具集 $\mathcal{T}_W = \mathcal{T}_{\text{default}} \setminus \{\text{TeamCreate}, \text{SendMessage}\}$

防止 $W$ spawn 子 $W$，避免无限递归。

后端三种 worker runtime：
1. **tmux pane**：用 tmux 的 split-window + send-keys，让 worker 的 output 可见于终端
2. **in-process**：在同一个 Bun process 里 spawn query sub-loop
3. **remote**：通过 Bridge Mode 在另一台机器跑

这三种是 latency vs isolation 的 trade-off：tmux > in-process > remote，依次延迟递增、隔离递增。

参考：[Multi-agent LLM systems](https://arxiv.org/abs/2402.01680), [AutoGen](https://arxiv.org/abs/2308.08155), [MetaGPT](https://arxiv.org/abs/2308.00352)

---

## 15. 泄露的工程教训：安全是木桶

```
npm publish → .map 文件未删 → map 引用 R2 zip → R2 URL 无 auth → 全部源码下载
```

100 个环节做对 99 个，最后一个 .map 文件把全部防泄漏设计（编译时 DCE、hex 编码、excluded-strings.txt、binary 字符串扫描）归零。

**直觉：安全是 min-over-cost 的，攻击者总找最薄弱环节**。这跟 LLM adversarial robustness 完全同构——jailbreaker 也总找最弱的 axis 攻击，不是最难的那个。

npm publish 的最佳实践：

```json
// package.json
{
  "files": ["dist/index.js", "dist/index.d.ts"]
  // 白名单制，只列要发布的
}
```

CI 加：
```bash
# 检查发布产物
npm pack --dry-run | grep -E '\.map$' && exit 1
```

参考：[npm files field](https://docs.npmjs.com/cli/v8/configuring-npm/package-json#files), [Source maps security](https://www.scien.cx/2023/01/30/source-map-security/)

---

## 16. 给 Karpathy 的几个 takeaway

1. **Agentic loop 的本质是 message state machine**。$M_t \to M_{t+1}$ 的转移由 model + tool executor 协同完成。所有"工程优化"都是在减少这个转移的 cost 或增加其信息密度。

2. **Context compression 应该是 multi-strategy 的**。单一 summarizer 是 dead simple 但 suboptimal 的。Claude Code 的四层策略对应四种不同的 "information lifetime" 假设，每层做最 cost-effective 的操作。这和 [VRAM hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware)（register / L1 / L2 / HBM）的思路惊人相似——离 model 越近的越贵越小越快。

3. **Streaming + concurrency 是 latency hiding 的核心**。把 $\max$ 替代 $\sum$，是 distributed systems 的经典操作。Claude Code 在 LLM layer 重现了这个 pattern。

4. **Feature flag 双层是 security + operational 的分离**。编译时 flag 防 reverse engineering，运行时 flag 防运营事故。这两件事不能混。

5. **Ablation infrastructure 是 LLM 工程成熟度的标志**。能跑 leave-one-out 的 team 才能做 attribution。否则永远在"感觉有用就上"的混沌状态。

6. **最简单的事故往往击穿最复杂的防御**。.map 文件 vs 编译时 DCE + hex 编码 + binary scan。这是 security 的 [weakest link principle](https://en.wikipedia.org/wiki/Weakest_link)。

---

## 17. 相关的几个开源资源

- 何宇峰用 950 行 Python 重写 Claude Code 核心：[he-yufeng/CoreCoder](https://github.com/he-yufeng/CoreCoder)
- Windy 的源码导读 + 从零构建教程：[Windy3f3f3f3f/how-claude-code-works](https://github.com/Windy3f3f3f3f/how-claude-code-works) 和 [Windy3f3f3f3f/claude-code-from-scratch](https://github.com/Windy3f3f3f3f/claude-code-from-scratch)
- Ink (TUI React)：[vadimdemedes/ink](https://github.com/vadimdemedes/ink)
- Bun bundler macros：[bun.sh/docs/bundler/macros](https://bun.sh/docs/bundler/macros)
- Anthropic Prompt Caching：[docs.anthropic.com/en/docs/build-with-claude/prompt-caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- Model Context Protocol：[modelcontextprotocol.io](https://modelcontextprotocol.io/)
- Zod v4：[zod.dev/v4](https://zod.dev/v4)
- Statsig experimentation：[docs.statsig.com/experiments](https://docs.statsig.com/experiments)

---

## 18. 我还想深挖的几个点

文章作者问读者想看哪个模块深挖——BashTool 1143 行沙箱机制 or Coordinator 多 Agent 编排。我个人最想看的其实是另外三个：

1. **`prompts.ts` 全文**。文章只引了几行 `@[MODEL LAUNCH]` TODO，但整个 system prompt 的结构（persona、tool description 注入、skill description 注入、permission rule 注入）才是 Claude Code 行为的真正源头。我怀疑里面有大量 [prompt scaffolding](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) 的工程经验。

2. **`CONTEXT_COLLAPSE` 的数据结构**。它说"维护一个类似 git log 的结构"，但具体是 linked list、tree、还是 DAG？replay 时的 token cost 是多少？这个数据结构的设计直接决定了 long-context agent 的可恢复性。

3. **`speculationState` 的预测器形式**。是简单的 last-ending-modality n-gram，还是有更复杂的 Markov model？prediction miss 时的 fallback cost 是多少？这关系到 speculative execution 是否值得——如果 hit rate 不高，speculation overhead 反而是负优化。

如果有这些 module 的源码或逆向分析，再聊。这次的泄露让 Anthropic 的 Claude Code 从"黑盒魔法"变成了"可学习的工程范例"，对整个 agent engineering 社区都是好事。
