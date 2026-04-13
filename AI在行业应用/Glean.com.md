



## Glean.com 是做什么的？

**Glean** 是一家专注于 **Enterprise AI（企业级人工智能）** 的美国科技公司，核心产品是一个 **全栈式企业搜索 + AI Work Assistant（工作助手）平台**。简单来说，它做的事情可以概括为一句话：

> **把公司里散落在各种工具中的数据连接起来，用一个 AI 搜索/助手帮你找到答案、创建内容、自动化工作流程。**

---

### 🎯 核心定位

| 维度 | 描述 |
|------|------|
| **产品名** | Glean — Work AI Platform |
| **Slogan** | "The Work AI platform connected to all your data. Find, create, and automate anything." |
| **核心功能** | Enterprise Search + AI Assistant + AI Agents + Automation |
| **创始人** | Arvind Jain（前 Google 工程副总裁） |
| **估值** | $7.2B（2025年 Series F，融资 $150M） |
| **ARR** | ~$200M（2025年底数据） |

**参考链接：**
- [Glean 官网](https://www.glean.com/)
- [Glean Technologies - Wikipedia](https://en.wikipedia.org/wiki/Glean_Technologies)
- [Glean 融资报道 - CNBC](https://www.cnbc.com/2025/06/10/glean-gen-ai-search-startup-raises-150-million-at-7-billion-value.html)
- [Glean ARR $200M - Fortune](https://fortune.com/2025/12/08/exclusive-glean-hits-200-million-arr-up-from-100-million-nine-months-back/)

---

### 🏗️ 技术架构（第一性原理拆解）

从第一性原理出发，Glean 要解决的根本问题是：

> **企业知识是碎片化的（fragmented），人需要花大量时间在多个 app 之间切换来寻找信息。**

要解决这个问题，需要三个核心能力：
1. **连接** — 把所有数据源接进来
2. **理解** — 理解数据之间的关系和上下文
3. **生成/行动** — 基于理解给出答案或执行动作

Glean 的架构就是围绕这三个能力构建的：

```
┌─────────────────────────────────────────────────┐
│              User Interface Layer                │
│   (Search Bar / Chat / Assistant / Agents)       │
├─────────────────────────────────────────────────┤
│            Agentic Engine Layer                  │
│   (AI Agents: Create, Automate, Reason)          │
├─────────────────────────────────────────────────┤
│            Model Hub Layer                       │
│   (Latest LLMs: GPT, Claude, Gemini, etc.)       │
├─────────────────────────────────────────────────┤
│       Hybrid Search + System of Context          │
│   (Semantic Search + Keyword + Personal Graph)   │
├─────────────────────────────────────────────────┤
│          Enterprise Graph (Knowledge Graph)       │
│   (Entities, Relationships, Permissions, Activity)│
├─────────────────────────────────────────────────┤
│          Data Connector Layer                    │
│   (100+ Integrations: Slack, Drive, Jira, etc.)  │
└─────────────────────────────────────────────────┘
```

**参考链接：**
- [How Glean Search Works](https://www.glean.com/resources/guides/how-glean-search-works)
- [The Glean Knowledge Graph](https://www.glean.com/resources/guides/glean-knowledge-graph)
- [Enterprise AI & Knowledge Graphs Blog](https://www.glean.com/blog/enterprise-ai-knowledge-graph)

---

### 🔑 核心技术组件详解

#### 1. Enterprise Graph（企业知识图谱）— Glean 的核心壁垒

这是 Glean 区别于普通 RAG 系统的关键。传统的 RAG 只做 "文档 → embedding → 检索"，而 Glean 构建了一个完整的 **Knowledge Graph**：

$$G = (V, E, W)$$

其中：
- $V$ = Vertices（节点）：包括 **Documents**（文档）、**People**（人员）、**Teams**（团队）、**Projects**（项目）、**Messages**（消息）
- $E$ = Edges（边）：表示节点之间的关系，如 "作者-文档"、"回复-消息"、"成员-团队"
- $W$ = Weights（权重）：边的权重，反映关系的强度（如交互频率、访问频率）

**为什么 Knowledge Graph 比纯 vector search 更强？**

因为企业搜索的核心挑战不是 "找到关键词匹配的文档"，而是 **理解上下文**：
- 谁写的这个文档？（authorship）
- 这个文档被谁查看过？（access patterns）
- 这个文档属于哪个项目？（organizational hierarchy）
- 当前用户有权访问哪些文档？（permissions）

Glean 的 Enterprise Graph 把这些全部编码进图中，使得搜索结果不仅语义相关，还 **上下文精准 + 权限安全**。

#### 2. Personal Graph（个人图谱）

这是 Enterprise Graph 的子图，为每个用户个性化定制：

$$G_{personal}(u) = \{v \in V \mid \text{relevance}(v, u) > \theta\} \cup \{e \in E \mid \text{relevance}(e, u) > \theta\}$$

其中 $\text{relevance}(v, u)$ 基于以下信号计算：
- 用户 $u$ 过去的搜索/点击行为
- 用户 $u$ 的组织关系（team membership, manager chain）
- 用户 $u$ 最近的工作活动（recently viewed, edited, shared）
- $\theta$ 是相关性阈值

**效果**：同一个 query，不同的人搜到的结果不同，因为他们的上下文不同。

#### 3. Hybrid Search（混合搜索）

Glean 不是只用 semantic search，而是结合了多种检索策略：

$$\text{Score}(q, d) = \alpha \cdot S_{\text{keyword}}(q, d) + \beta \cdot S_{\text{semantic}}(q, d) + \gamma \cdot S_{\text{graph}}(q, d) + \delta \cdot S_{\text{freshness}}(q, d)$$

其中：
- $S_{\text{keyword}}(q, d)$：传统关键词匹配分数（BM25 等）
- $S_{\text{semantic}}(q, d)$：语义相似度分数（embedding cosine similarity）
- $S_{\text{graph}}(q, d)$：图谱关系分数（基于 Enterprise Graph 的路径距离和关系强度）
- $S_{\text{freshness}}(q, d)$：时效性分数（最近更新的文档权重更高）
- $\alpha, \beta, \gamma, \delta$：可学习的权重超参数

#### 4. Model Hub

Glean 不绑定单一 LLM，而是提供一个 **Model Hub**，可以接入最新的模型：
- GPT-4o / GPT-4.1
- Claude 3.5 / 4
- Gemini
- 自研模型

这样企业可以根据不同场景选择最适合的模型（速度 vs 质量 vs 成本）。

#### 5. Glean Agents（AI 代理）

这是 Glean 最新的进化方向 — 从 "被动搜索" 变成 "主动执行"：

- **Search Agent**：回答问题，引用来源
- **Create Agent**：生成文档、邮件、代码
- **Automate Agent**：自动化工作流（如 "每周一汇总这个项目的进展发给团队"）
- **Custom Agent**：企业可以自定义 agent

---

### 📊 数据连接能力

Glean 支持 **100+ 数据源**的连接，包括但不限于：

| 类别 | 工具示例 |
|------|---------|
| Communication | Slack, Microsoft Teams, Zoom |
| Document | Google Drive, SharePoint, Confluence, Notion |
| Code | GitHub, GitLab, Bitbucket |
| Project | Jira, Asana, Linear |
| CRM | Salesforce, HubSpot |
| Email | Gmail, Outlook |
| Database | BigQuery, Snowflake, PostgreSQL |

**关键点**：Glean 不仅仅是 "读" 这些数据，它还 **尊重每个数据源的权限系统**。用户搜索时只能看到自己有权访问的内容，这是企业级搜索和 consumer 搜索的本质区别。

**参考链接：**
- [Glean + Google Cloud Blog](https://cloud.google.com/blog/products/data-analytics/glean-uses-bigquery-and-google-ai-to-enhance-enterprise-search)
- [Glean + AWS Marketplace](https://aws.amazon.com/blogs/awsmarketplace/transform-enterprise-search-knowledge-discovery-glean-amazon-bedrock/)

---

### 💡 与竞品对比

| 竞品 | 核心差异 |
|------|---------|
| **Microsoft Copilot** | 深度绑定 Microsoft 365 生态，但对非 MS 工具支持较弱 |
| **Google Workspace AI** | 深度绑定 Google 生态，类似局限 |
| **Perplexity Enterprise** | 更偏通用搜索 + RAG，缺少企业 Knowledge Graph 和权限管理 |
| **Atlassian Intelligence** | 只覆盖 Atlassian 自家产品 |
| **Glean** | **跨平台 + Knowledge Graph + 权限安全 + Agents**，这是核心差异化 |

---

### 🧠 直觉总结

用一句话 build 你的直觉：

> **Glean = Google for your company × Knowledge Graph × AI Agents**

就像 Google 把整个互联网变得可搜索一样，Glean 把企业内部的所有数据变得可搜索、可理解、可行动。而它的核心壁垒不是 LLM（别人也有），而是那个 **Enterprise Knowledge Graph** — 它理解谁在做什么、什么和什么有关系、谁有权看什么，这些是纯 LLM 做不到的。