# 深度解读：OpenAI 的 "Harness Engineering" 实践

这篇文章记录了 OpenAI 团队在 2025 年 8 月至 2026 年 1 月（5个月）间进行的一项激进实验：**完全零手写代码**，由 Codex agents 驱动整个软件开发流程。让我从多个维度详细拆解。

---

## 一、实验概述：核心数据与背景

### 1.1 实验参数表

| 指标 | 数值 | 说明 |
|------|------|------|
| 实验周期 | 5个月 (Aug 2025 - Jan 2026) | 从空仓库开始 |
| 代码规模 | ~1,000,000 行 | 包含应用逻辑、测试、CI配置、文档、observability、内部工具 |
| Pull Requests | ~1,500 个 | 全部由 Codex 生成 |
| 驱动工程师 | 3人 → 7人 | 初期3人，后期增长到7人 |
| 人均PR产出 | 3.5 PR/engineer/day | 且随团队扩大而提升 |
| 效率提升 | ~10x | 相比手写代码的估算 |
| 用户规模 | 数百内部用户 + 外部 alpha testers | 真实产品，非 demo |

### 1.2 核心约束

```
Constraint: ∀ code ∈ repository, ∄ human_written(code)
```

即：仓库中**任何一行代码**都不是人类直接编写的，包括：
- Application logic & tests
- CI configuration & release tooling  
- Internal developer tools
- Documentation & design history
- Evaluation harnesses
- Review comments & responses
- Repository management scripts
- Production dashboard definitions

---

## 二、核心理念：Humans Steer, Agents Execute

这是一个**控制论框架**的体现。让我用数学形式描述：

### 2.1 传统软件工程模型

```
Human → [Write Code] → [Review] → [Merge] → [Deploy]
```

人类深度介入每个环节，bottleneck 在人类时间。

### 2.2 Agent-First 模型

```
Human → [Specify Intent] → Agent → [Execute] → [Feedback Loop] → Agent → ...
                              ↑__________________|
```

形式化描述：

$$\text{Throughput}_{\text{agent-first}} = \frac{\text{Human Intent Specification}}{\text{Human Attention Time}} \times \text{Agent Parallelism Factor}$$

其中：
- **Human Intent Specification**: 高层目标描述（prompts）
- **Human Attention Time**: 稀缺资源，记为 $T_H$
- **Agent Parallelism Factor**: 可同时运行的 agent 数量，记为 $P_A$

在传统模式下，throughput 受限于：

$$\text{Throughput}_{\text{traditional}} \approx \frac{\text{Code Output}}{\text{Human Coding Time}}$$

而在 agent-first 模式下：

$$\text{Throughput}_{\text{agent-first}} \approx P_A \times \frac{\text{Task Completion Rate}}{T_H}$$

关键洞察：**$P_A$ 可以很大（文章提到单次 Codex 运行可持续6小时），而 $T_H$ 保持恒定**。

---

## 三、技术架构详解

### 3.1 Layered Domain Architecture（分层领域架构）

文章给出了一张核心架构图，让我解析：

```
┌─────────────────────────────────────────────────────────────┐
│                    Business Domain                          │
│  (e.g., App Settings, User Management, Billing)            │
│                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐    │
│  │  Types  │ → │ Config  │ → │  Repo   │ → │ Service │    │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘    │
│        ↓                                        ↓           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Providers                         │   │
│  │  (auth, connectors, telemetry, feature flags)       │   │
│  └─────────────────────────────────────────────────────┘   │
│        ↓                                                    │
│  ┌─────────┐   ┌─────────┐                                │
│  │ Runtime │ → │   UI    │                                │
│  └─────────┘   └─────────┘                                │
└─────────────────────────────────────────────────────────────┘
                              ↑
                        ┌─────────┐
                        │  Utils  │ (cross-cutting utilities)
                        └─────────┘
```

**Dependency Rule（依赖规则）**：

$$\forall l_i, l_j \in \text{Layers}, \quad \text{if } i < j, \text{ then } l_i \rightarrow l_j \text{ is allowed, but } l_j \rightarrow l_i \text{ is forbidden}$$

其中 Layer 序号定义为：
- $l_1$ = Types (类型定义)
- $l_2$ = Config (配置)
- $l_3$ = Repo (数据访问层)
- $l_4$ = Service (业务逻辑层)
- $l_5$ = Runtime (运行时)
- $l_6$ = UI (用户界面)

**Cross-cutting Concerns（横切关注点）** 通过 Providers 模块单一接口进入：

$$\text{CrossCutting} = \{\text{auth}, \text{connectors}, \text{telemetry}, \text{feature\_flags}\}$$

$$\forall c \in \text{CrossCutting}, \quad \text{Interface}(c) \subseteq \text{Providers}$$

### 3.2 为什么这种架构对 Agent 友好？

文章提到一个关键洞察：

> "This is the kind of architecture you usually postpone until you have hundreds of engineers. With coding agents, it's an early prerequisite."

原因在于**约束的可执行性**：

1. **Structural Constraints**（结构约束）：依赖方向可由 linter 机械检查
2. **Boundary Constraints**（边界约束）：每个 domain 有明确边界
3. **Predictable Structure**（可预测结构）：agent 可以依赖稳定的模式

**对比分析**：

| 特性 | 传统微服务架构 | Agent-First 架构 |
|------|---------------|-----------------|
| 边界定义 | 往往模糊，依赖团队共识 | 机械强制执行 |
| 依赖检查 | 事后 code review | CI 时自动拦截 |
| 理解成本 | 需要人类经验积累 | 可从代码结构直接推理 |
| 变更风险 | 边界破坏可能长期潜伏 | 立即触发 lint 错误 |

---

## 四、Context Management：知识管理策略

这是文章最精华的部分之一。

### 4.1 失败模式："One Big AGENTS.md"

文章描述了他们尝试过的失败方案：

```markdown
# AGENTS.md (Anti-pattern)
- 1000+ lines of instructions
- Everything marked as "important"
- Rules become stale over time
- No mechanical verification
```

**失败原因的数学描述**：

设 Context Window 大小为 $C$，任务描述占用 $T$，代码占用 $K$，相关文档占用 $D$。

传统大文件方案：

$$T + K + D_{\text{big}} \approx C \implies D_{\text{effective}} \rightarrow 0$$

即：巨大的 instruction 文件挤占了任务和代码的 context，导致 agent 无法有效推理。

### 4.2 成功模式：Progressive Disclosure（渐进式披露）

新方案的核心：

```
AGENTS.md (≈100 lines) → Table of Contents
        ↓
docs/
├── design-docs/
│   ├── index.md
│   ├── core-beliefs.md
│   └── ...
├── exec-plans/
│   ├── active/
│   ├── completed/
│   └── tech-debt-tracker.md
├── generated/
│   └── db-schema.md
├── product-specs/
│   ├── index.md
│   └── ...
├── references/
│   ├── design-system-reference-llms.txt
│   └── ...
├── DESIGN.md
├── FRONTEND.md
├── PLANS.md
└── ...
```

**Progressive Disclosure 公式**：

$$D_{\text{effective}} = \sum_{i=1}^{n} D_i \times \mathbb{1}[\text{relevant}(D_i, \text{task})]$$

其中：
- $D_i$ 是第 $i$ 个文档
- $\mathbb{1}[\cdot]$ 是指示函数，当文档与任务相关时为 1
- Agent 通过 AGENTS.md 中的"地图"找到相关文档

**实现机制**：

1. **Short AGENTS.md**: ~100 lines，作为 entry point
2. **Indexed docs/**: 结构化目录，有 index.md
3. **Verification Status**: 文档有验证状态标记
4. **Mechanical Enforcement**: CI 检查文档新鲜度、交叉链接、结构正确性
5. **Doc-Gardening Agent**: 定期扫描过期文档，自动创建修复 PR

### 4.3 Knowledge Visibility Principle（知识可见性原则）

文章用了一张图说明核心问题：

```
┌─────────────────────────────────────────────────┐
│           Codex's Knowledge Bubble              │
│                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│   │  Code    │  │  Docs    │  │  Schemas │    │
│   └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────┘
                    ↑
                    │ What Codex can see
                    │
┌─────────────────────────────────────────────────┐
│           Outside Codex's View                  │
│                                                 │
│   ┌────────────┐  ┌────────────┐  ┌──────────┐│
│   │ Google Docs│  │   Slack    │  │  Human   ││
│   │            │  │  Messages  │  │  Heads   ││
│   └────────────┘  └────────────┘  └──────────┘│
└─────────────────────────────────────────────────┘
```

**原则**：

$$\forall \text{knowledge } k, \quad k \notin \text{repo} \implies k \notin \text{Agent's context}$$

这类似于**新员工 onboarding 问题**：如果信息只存在于 Slack 讨论或 Google Docs 中，新员工（或 agent）无法获取。

**解决方案**：将所有决策、架构讨论、产品原则编码到仓库中的 markdown 文件。

---

## 五、Observability & Testing Infrastructure

### 5.1 Per-Worktree Isolation（按工作树隔离）

这是一个精妙的设计：

```
git worktree → isolated app instance → isolated observability stack
```

每个 Codex 任务运行在自己的 git worktree 中，拥有：
- 独立的应用实例
- 独立的日志/指标/追踪栈
- 任务完成后销毁

**好处**：

1. **Parallelism**: 多个 agent 可同时工作，互不干扰
2. **Isolation**: 一个任务的失败不影响其他任务
3. **Observability**: agent 可以查询自己实例的 logs/metrics

### 5.2 Chrome DevTools Protocol Integration

文章描述了一个关键能力增强：

```
┌──────────────────────────────────────────────────────────────┐
│                    Codex Agent                               │
│                                                              │
│  1. Select Target (e.g., "user onboarding flow")            │
│  2. Snapshot Before State (DOM + screenshot)                 │
│  3. Trigger UI Path (click buttons, fill forms)              │
│  4. Observe Runtime Events (via Chrome DevTools MCP)         │
│  5. Apply Fix                                                │
│  6. Restart App                                              │
│  7. Re-run Validation                                        │
│  8. Loop until clean                                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**技术栈**：
- Chrome DevTools Protocol (CDP)
- DOM snapshots
- Screenshots
- Navigation events

**应用场景**：
- Bug reproduction
- Fix validation
- UI behavior reasoning

### 5.3 Observability Stack

```
┌─────────────────────────────────────────────────────────────┐
│                        App                                   │
│                          │                                   │
│                          ▼                                   │
│                    ┌─────────┐                              │
│                    │  Vector │ (telemetry collector)        │
│                    └─────────┘                              │
│                          │                                   │
│         ┌────────────────┼────────────────┐                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Victoria  │ │   Victoria  │ │   Victoria  │           │
│  │    Logs     │ │   Metrics   │ │   Traces    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│      LogQL           PromQL          TraceQL               │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                   │
│                    ┌─────────┐                              │
│                    │  Codex  │ (query, correlate, reason)   │
│                    └─────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

**可执行的 SLO（Service Level Objectives）**：

现在可以这样提示 Codex：
- "Ensure service startup completes in under 800ms"
- "No span in these four critical user journeys exceeds two seconds"

这些 SLO 被转化为可验证的指标查询。

---

## 六、Enforcement Mechanisms：约束执行机制

### 6.1 Custom Linters（自定义 Linter）

文章强调：约束必须**机械执行**，而非依赖人类 review。

**示例约束**：
1. Structured logging（结构化日志）
2. Naming conventions for schemas and types
3. File size limits
4. Platform-specific reliability requirements

**关键创新**：Lint 错误信息包含修复指导：

```python
# 传统 lint 错误
"File too large: 500 lines exceeds limit of 300"

# Agent-friendly lint 错误
"File too large: 500 lines exceeds limit of 300. 
 Remediation: Split into smaller modules following the 
 pattern in docs/architecture/module-split-guide.md"
```

这使 agent 能够自我修复，而非等待人类指导。

### 6.2 Taste Invariants（品味不变量）

文章提到 "a small set of taste invariants"。这些是**风格偏好**被编码为规则：

| Invariant | Rationale |
|-----------|-----------|
| Shared utils over hand-rolled helpers | Keep invariants centralized |
| No "YOLO-style" data probing | Validate boundaries or use typed SDKs |
| Explicit over implicit | Agent can't guess shapes |

### 6.3 Golden Principles（黄金原则）

文章描述了一个"garbage collection"过程：

```
┌─────────────────────────────────────────────────────────┐
│                Technical Debt Lifecycle                  │
│                                                         │
│  Code Change → Potential Drift → Detection → GC Agent  │
│       ↑                                    │            │
│       └────────── Fix PR ←─────────────────┘            │
│                                                         │
│  Frequency: Daily (continuous)                         │
│  Review Time: < 1 minute per PR                        │
│  Auto-merge: Yes (for low-risk changes)                │
└─────────────────────────────────────────────────────────┘
```

**类比**：技术债务就像高利率贷款：

$$\text{Debt}_{t+1} = \text{Debt}_t \times (1 + r) - \text{Payment}$$

其中 $r$ 是"利率"（债务复利率）。**连续小额支付比一次性大额偿还更优**。

---

## 七、工作流详解

### 7.1 Human-Agent Interaction Pattern

```
Human Engineer
      │
      │ 1. Describe task (prompt)
      ▼
┌─────────────┐
│    Codex    │ ← 2. Run agent
└─────────────┘
      │
      │ 3. Open PR
      ▼
┌─────────────────────────────────────────────┐
│            Self-Review Loop                 │
│                                             │
│  ┌─────────┐    ┌─────────┐    ┌────────┐ │
│  │ Local   │ → │ Cloud   │ → │ Iterate│ │
│  │ Review  │    │ Agent   │    │        │ │
│  └─────────┘    │ Review  │    └────────┘ │
│                 └─────────┘                 │
│                      │                      │
│                      ▼                      │
│              ┌──────────────┐              │
│              │ All reviewers│              │
│              │ satisfied?   │              │
│              └──────────────┘              │
│                    │ Yes                    │
│                    ▼                        │
│              ┌──────────────┐              │
│              │  Human Review│              │
│              │  (optional)  │              │
│              └──────────────┘              │
└─────────────────────────────────────────────┘
      │
      │ 4. Merge (often by agent itself)
      ▼
   Deployed
```

**关键观察**：
- Human review is **optional**
- Most review effort is **agent-to-agent**
- Agent can **squash and merge its own PRs**

### 7.2 End-to-End Feature Delivery

文章描述了一个突破性能力：给定单个 prompt，Codex 可以：

```
Input: Single prompt describing a feature/bug

Codex Execution Flow:
1. Validate current codebase state
2. Reproduce bug (if applicable)
3. Record video of failure
4. Implement fix
5. Validate fix by driving app
6. Record video of resolution
7. Open PR
8. Respond to agent/human feedback
9. Detect & remediate build failures
10. Escalate to human (if judgment required)
11. Merge change

Output: Merged PR with evidence
```

**视频录制**是一个有趣细节——为人类提供可验证的证据。

---

## 八、失败与学习

### 8.1 早期失败：Underspecified Environment

文章坦诚：

> "Early progress was slower than we expected, not because Codex was incapable, but because the environment was underspecified."

这是一个重要洞察：**agent 失败往往是环境问题，而非模型能力问题**。

**问题诊断框架**：

```
Agent Failure
      │
      ├── Model limitation? → Rarely the root cause
      │
      ├── Missing capability? → What tool/abstraction is missing?
      │
      ├── Underspecified intent? → Improve prompt/docs
      │
      └── Missing guardrail? → Add lint/test
```

**修复模式**：

$$\text{Fix} \neq \text{"Try harder"}$$

$$\text{Fix} = \text{Identify missing capability} \rightarrow \text{Make it legible \& enforceable}$$

### 8.2 AI Slop Problem

文章描述了一个严重问题：

> "Our team used to spend every Friday (20% of the week) cleaning up 'AI slop.'"

**原因**：Codex 会复制仓库中已有的模式——包括不均匀或次优的模式。

**解决方案**：Golden Principles + GC Agent

1. 将"品味"编码为可执行的规则
2. 定期运行 cleanup agent 扫描偏差
3. 自动创建重构 PR
4. 大部分可在1分钟内 review 并自动合并

---

## 九、关键洞察总结

### 9.1 架构决策

| 决策 | 传统考量 | Agent-First 考量 |
|------|----------|-----------------|
| 分层架构 | 数百人团队才需要 | 早期就必须，约束即速度 |
| 依赖方向 | 靠共识维持 | 机械强制执行 |
| 文档策略 | 外部工具 | 仓库内版本化 |
| 技术选型 | 最新/最酷 | 训练数据中常见的"无聊"技术 |

### 9.2 工程角色转变

```
Traditional Engineer:
  - Write code
  - Review code
  - Debug issues
  - Maintain documentation

Agent-First Engineer:
  - Design environments
  - Specify intent
  - Build feedback loops
  - Encode constraints
  - Prioritize human attention
```

### 9.3 效率公式

$$\text{Efficiency} = \frac{\text{Output}}{\text{Human Attention Time}}$$

传统模式：Output ∝ Human Coding Time

Agent-First 模式：Output ∝ Agent Parallelism × Human Specification Quality

**关键**：Human Attention Time 是固定稀缺资源，必须最大化 leverage。

---

## 十、开放问题与未来方向

文章承认仍有许多未知：

1. **长期架构演化**：完全 agent 生成的系统，多年后架构一致性如何保持？
2. **Human judgment leverage**：人类判断在哪些地方最有价值？
3. **Judgment compounding**：如何编码判断使其复利增长？
4. **Model evolution**：更强大的模型会如何改变这个系统？

---

## 参考资料

1. **原文**: [Harness engineering: leveraging Codex in an agent-first world | OpenAI](https://openai.com/index/harness-engineering-leveraging-codex-in-an-agent-first-world/) (Feb 11, 2026)

2. **相关文章**:
   - [AGENTS.md pattern](https://openai.com/index/) - 文章中提到的 agent 指导文件模式
   - [Architecture documentation](https://openai.com/index/) - 架构文档最佳实践
   - [Execution plans](https://openai.com/index/) - 执行计划模式

3. **技术参考**:
   - Chrome DevTools Protocol: https://chromedevtools.github.io/devtools-protocol/
   - Victoria Metrics: https://victoriametrics.com/
   - LogQL/PromQL/TraceQL 查询语言

4. **概念背景**:
   - Clean Architecture (Uncle Bob) - 分层架构理论
   - Domain-Driven Design - Bounded Context 概念
   - Progressive Disclosure - HCI 中的信息呈现原则

---

## 总结

这篇文章展示了一个**范式转变**：从"人类写代码"到"人类设计环境，agents 执行代码"。核心洞察是：

1. **约束即速度**：严格的架构约束让 agents 能快速移动而不破坏基础
2. **Context management is everything**：知识的组织方式决定 agent 效能
3. **Human attention is the scarcest resource**：所有工具设计都围绕最大化人类注意力 leverage
4. **Mechanical enforcement over human judgment**：能编码为规则的，就不依赖人类判断

这不仅是工具的升级，更是软件工程方法论的重构。