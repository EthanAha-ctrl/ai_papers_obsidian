## Google Agent2Agent (A2A) 协议 —— 文章总结

### 📌 一句话概括

Google Cloud 于 **2025年4月9日** 发布了一个名为 **Agent2Agent (A2A)** 的开放协议，旨在让不同厂商、不同框架构建的 AI agent 能够互相通信、安全交换信息、协调行动，从而实现跨企业平台的多 agent 协作。

---

### 🧩 核心问题与动机

当前企业中 AI agent 的痛点：

| 痛点 | 描述 |
|------|------|
| **数据孤岛** | Agent 各自运行在隔离的应用和数据系统中 |
| **互操作性缺失** | 不同 vendor、不同 framework 的 agent 无法直接协作 |
| **被当作"工具"使用** | 现有方案往往把 agent 降级为 tool call，丧失了 agent 的自主性和协作潜力 |

A2A 的目标就是解决这些问题，让 agent 以 **真正的 agent 身份**（而非简单 tool）进行协作。

---

### 🏗️ 五大设计原则

| 原则 | 含义 | 技术细节 |
|------|------|----------|
| **Embrace agentic capabilities** | Agent 以其自然、非结构化的方式协作，即使不共享 memory、tools、context 也能工作 | 这意味着协议层不假设 agent 内部状态一致性，而是通过消息传递来实现协作 |
| **Build on existing standards** | 基于 HTTP、SSE (Server-Sent Events)、JSON-RPC 构建 | $\text{A2A transport} = \text{HTTP} + \text{SSE} + \text{JSON-RPC 2.0}$，降低集成门槛，兼容现有 IT 栈 |
| **Secure by default** | 企业级认证授权，与 OpenAPI 的 authentication schemes 对齐 | 支持 Bearer token、API Key、OAuth2 等方案，与 OpenAPI security spec parity |
| **Support for long-running tasks** | 从快速任务到耗时数天的研究任务均支持，可实时反馈 | 使用 SSE 实现流式状态推送，task lifecycle 包含 `submitted → working → completed/failed/canceled` 等状态 |
| **Modality agnostic** | 不仅支持文本，还支持音频、视频流等多模态 | "Part" 对象带有 content type 标识，支持 `text/plain`、`image/png`、`audio/wav`、`video/mp4` 等 MIME 类型协商 |

---

### 🔧 A2A 工作机制（核心架构）

A2A 采用了 **Client-Remote** 模型：

```
┌──────────────┐                          ┌──────────────┐
│  Client Agent │◄────── A2A Protocol ────►│ Remote Agent │
│  (发起任务方) │     HTTP/SSE/JSON-RPC    │  (执行任务方) │
└──────────────┘                          └──────────────┘
```

#### 四大核心能力：

**1. Capability Discovery（能力发现）**
- Agent 通过 **"Agent Card"**（JSON 格式）声明自己的能力
- 类似于 OpenAPI spec 描述 API 的能力，Agent Card 描述 agent 的能力
- Client agent 可以据此发现最佳匹配的 remote agent

**2. Task Management（任务管理）**
- 通信以 **Task** 为核心单位
- Task 有明确的生命周期：`created → in-progress → completed / failed / canceled`
- Task 的输出称为 **"Artifact"**
- 支持短时任务（秒级完成）和长时任务（小时/天级，人参与其中）

**3. Collaboration（协作）**
- Agent 之间通过 **Message** 交换上下文、回复、artifact 或用户指令
- Message 包含多个 **"Part"**，每个 Part 是一段完整内容（如生成的图片、文本等）

**4. User Experience Negotiation（用户体验协商）**
- 每个 Part 有指定的 content type
- Client 和 Remote agent 协商正确的展示格式
- 明确协商用户 UI 能力：iframe、video、web forms 等
- 公式化表示：
$$\text{Part} = \{type: \text{ContentType}, content: \text{Payload}, metadata: \text{Map}\}$$
$$\text{Message} = \{role: \text{Role}, parts: [\text{Part}], metadata: \text{Map}\}$$

---

### 🤝 与 MCP 的关系

| | A2A | MCP (Model Context Protocol) |
|---|---|---|
| **创建者** | Google | Anthropic |
| **定位** | Agent ↔ Agent 互操作 | Agent ↔ Tool/Context 互操作 |
| **层次** | Agent 间协作层 | Agent 的工具和上下文提供层 |
| **关系** | **互补关系**，不是竞争 | 为 agent 提供 tools 和 context |

可以用一个分层模型来理解：

```
┌─────────────────────────────────┐
│         A2A Layer               │  ← Agent-to-Agent 协作
│   (跨 agent 任务编排与通信)      │
├─────────────────────────────────┤
│         MCP Layer               │  ← Agent-to-Tool/Context
│   (agent 获取工具和数据上下文)    │
├─────────────────────────────────┤
│    Foundation: HTTP/SSE/JSON-RPC │
└─────────────────────────────────┘
```

---

### 📋 实际案例：候选人招聘

文章给出了一个招聘场景的端到端流程：

```
Hiring Manager
      │
      ▼
┌─────────────┐    A2A     ┌──────────────────┐
│ Client Agent │──────────►│ Sourcing Agent   │  → 从各平台筛选候选人
│ (Agentspace) │           └──────────────────┘
│              │    A2A     ┌──────────────────┐
│              │──────────►│ Scheduling Agent │  → 安排面试
│              │           └──────────────────┘
│              │    A2A     ┌──────────────────┐
│              │──────────►│ Background Check  │  → 背景调查
│              │           │ Agent            │
└─────────────┘           └──────────────────┘
```

关键点：所有这些 agent 可能来自 **不同供应商**（SAP、Workday、ServiceNow 等），运行在 **不同平台** 上，A2A 让它们无缝协作。

---

### 🌐 生态伙伴

文章列出了 **50+ 技术合作伙伴** 和 **多家服务提供商**，包括：

| 类别 | 代表伙伴 |
|------|---------|
| **技术/平台** | Atlassian, Salesforce, SAP, ServiceNow, PayPal, MongoDB, Cohere, LangChain, Neo4j, Elastic, Datadog, UiPath, Box, Intuit |
| **服务/咨询** | Accenture, BCG, Capgemini, Deloitte, KPMG, McKinsey, PwC, TCS, Wipro, HCLTech, Cognizant, Infosys |

值得注意的是 **Salesforce** 和 **SAP** 这两个企业软件巨头的加入——意味着 A2A 有望打通两大企业应用生态。

---

### 🔮 路线图

- **当前**：发布 draft specification + 代码示例 + 开源
- **2025年下半年**：与合作伙伴推出 production-ready 版本
- 社区开放贡献

---

### 💡 第一性原理视角的思考

从第一性原理来看 A2A 解决的核心问题：

1. **为什么需要 agent 互操作协议？** 因为复杂任务天然跨越系统边界。单个 agent 无法拥有所有数据和工具，协作是必然。

2. **为什么不直接用 API？** 传统 API 是确定性的 request-response，而 agent 交互是非确定性的、需要状态管理、需要协商——A2A 在 HTTP/JSON-RPC 之上增加了 **Task lifecycle** 和 **Capability discovery** 语义层。

3. **A2A 的本质是什么？** 它是一个 **agent 间的应用层协议**，类似于 HTTP 是服务间的应用层协议。正如 HTTP 统一了 Web 服务的通信方式，A2A 试图统一 agent 间的通信方式。

4. **风险与挑战**：协议的采纳率是关键。如果主流 agent framework（LangChain、CrewAI、AutoGen 等）不原生支持，A2A 可能沦为 "Google-only" 的生态。但目前 LangChain 的 Harrison Chase 明确表示支持，这是一个积极信号。

---

### 📚 参考链接

- A2A 官网与规范：[https://google.github.io/A2A/](https://google.github.io/A2A/)
- A2A GitHub 仓库：[https://github.com/google/A2A](https://github.com/google/A2A)
- MCP 协议：[https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)
- Google Cloud 博客原文：[https://cloud.google.com/blog/products/application-development/announcing-the-agent2agent-protocol](https://cloud.google.com/blog/products/application-development/announcing-the-agent2agent-protocol)