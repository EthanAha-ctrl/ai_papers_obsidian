





Glean 是一个 **Work AI platform**，主要提供企业级人工智能和搜索能力。它通过连接并理解企业所有数据，生成可信答案并自动化工作流程。下面从多个技术层面详细解析：

---

## **一、核心定位**  
Glean 定位为 "AI-native" 架构的 **enterprise search platform**，与传统搜索引擎不同，它专注于企业内部结构化/非结构化数据的智能检索与生成。其价值主张是：
- **统一数据接入**：连接企业内部的 CRM、文档、代码库、聊天记录等。
- **语义理解**：通过 embedding 和 graph 技术捕捉实体关系。
- **生成式回答**：基于 RAG 生成自然语言答案而非仅返回链接。

参考链接：  
https://www.glean.com/  
https://en.wikipedia.org/wiki/Glean_Technologies

---

## **二、技术架构深度解析**

### **2.1 AI-native 架构**  
Glean 从底层设计就围绕 AI 构建，而非在传统搜索上叠加 AI 功能。其架构包含以下核心层：

1. **Data Ingestion Layer**  
   支持多种数据源 connector，包括：
   - **SaaS 产品**：Salesforce、ServiceNow、Jira、Confluence  
   - **代码仓库**：GitHub、GitLab  
   - **通信工具**：Slack、Microsoft Teams  
   - **文档存储**：Google Drive、SharePoint  
   每个 connector 采用 **增量同步** 和 **权限继承**，确保数据实时且符合访问控制。

2. **Indexing & Enrichment Layer**  
   - **文档分块（Chunking）**：  
     长文档被切分为语义连贯的 chunk，常用方法包括：
     - 固定大小分块（Fixed-size chunking）  
     - 语义分块（Semantic chunking，基于 embedding 相似度）  
     - 递归分块（Recursive chunking，按章节层级递归切割）
   - **向量化（Embedding）**：  
     使用文本 embedding 模型（如 OpenAI `text-embedding-ada-002` 或开源模型）将 chunk 转换为向量：  
     \[
     \mathbf{v}_i = f_{\text{embed}}(c_i) \in \mathbb{R}^d
     \]
     其中 \( d \) 为 embedding 维度（通常 1536 或 768），\( c_i \) 为第 \( i \) 个 chunk。
   - **元数据标签**：提取文档中的实体、时间、作者等，并附加权限标签（如 `department: engineering`）。

3. **Enterprise Graph（企业图谱）**  
   Glean 的核心差异化在于构建 **知识图谱**，将分散的文档、用户、实体通过关系连接：
   - **节点**：文档、员工、项目、客户、代码库等  
   - **边**：`created_by`、`mentions`、`belongs_to`、`depends_on` 等关系  
   图谱通过 **图神经网络（GNN）** 或 **transitive closure** 实现跨文档推理，例如：
   > "项目 Alpha 的 PR 涉及数据库迁移，因此相关文档可能包含 'PostgreSQL' 和 '数据迁移'"

4. **Retrieval Layer**  
   查询时执行 **混合检索（Hybrid Retrieval）**：
   - **向量检索**：计算 query embedding 与文档 chunk embedding 的余弦相似度：
     \[
     \text{score}_{\text{vec}}(q, d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \|\mathbf{v}_d\|}
     \]
   - **关键词检索（BM25）**：基于词频和文档长度加权  
   - **图检索**：从 query 中提取实体，在 graph 上遍历邻居节点  
   最终采用 **加权融合** 或 **学习排序（Learning to Rank）** 模型合并多个检索结果。

5. **RAG（Retrieval-Augmented Generation）层**  
   将检索到的 top-k 个 chunk 作为上下文，输入 LLM 生成答案：
   - **Prompt 设计**：  
     ```
     请基于以下上下文回答用户问题。如果信息不足，请说明。

     上下文：
     {retrieved_chunk_1}
     {retrieved_chunk_2}
     ...

     问题：{user_query}

     答案：
     ```
   - **生成模型**：可选用 GPT-4、Claude 或开源模型（Llama 3、Mistral）。  
   - **引用追踪**：答案中标注来源文档，确保可验证性。  
   - **防止幻觉**：通过 **上下文约束** 和 **事实一致性检查** 降低 LLM 幻觉风险。

6. **Agent Layer**  
   Glean Agents 是可编排的 AI 代理，支持：
   - **结构化查询代理**：将自然语言转换为 SQL（类似 Text-to-SQL），直接查询 BigQuery 等数据仓库。  
   - **工作流代理**：触发多步操作，如 "提交 PR" → "通知团队" → "更新 Jira 状态"。  
   - **自定义代理**：用户可通过低代码界面定义权限、工具调用和记忆机制。

7. **Governance & Security Layer**  
   - **权限继承**：检索和生成时严格检查用户权限，避免越权访问。  
   - **审计日志**：记录所有 query、retrieved 文档和生成答案。  
   - **内容过滤**：拦截敏感信息或不当内容。

参考链接：  
https://www.glean.com/blog/7-core-components-of-an-ai-agent-architecture-explained  
https://www.glean.com/blog/rag-for-llms

---

### **2.2 Google Cloud 集成**  
Glean 深度集成 Google Cloud，利用其数据和 AI 服务：
- **BigQuery**：存储结构化数据，支持 SQL 查询和实时分析。  
  Glean 的 **Structured Query Agents** 可将自然语言转换为 SQL：
  \[
  \text{SQL} = f_{\text{text2sql}}(\text{query}, \text{schema})
  \]
  其中 `schema` 包含表名、列名和数据类型。
- **Google AI**：使用 Vertex AI 托管 embedding 和生成模型，实现微调和高可用推理。  
- **Dataflow**：用于大规模生成训练数据，例如从文档中抽取实体-关系三元组。  
- **Google Search**：可能利用 Google 的检索算法优化跨文档排序。

参考链接：  
https://cloud.google.com/blog/products/data-analytics/glean-uses-bigquery-and-google-ai-to-enhance-enterprise-search

---

## **三、关键技术创新**

### **3.1 Enterprise Graph 的优势**  
传统 RAG 仅依赖向量检索，而 Glean 的 graph 提供：
- **关系推理**：通过多跳查询（multi-hop）回答复杂问题。  
  例如：  
  Query: "哪些工程师在去年修改了数据库迁移相关的代码？"  
  Graph 路径：`工程师 → 提交的 PR → PR 描述 → 包含 '数据库迁移' → 时间筛选`
- **去重与消歧**：同一实体在不同文档中的引用被聚合，减少冗余。
- **权限传播**：基于图的权限继承，确保敏感关系不被泄露。

### **3.2 防止幻觉的机制**  
- **引用强制**：要求 LLM 引用检索到的 chunk，避免捏造。  
- **答案一致性校验**：使用 **cross-encoder** 模型检查生成答案与上下文的事实一致性：  
  \[
  \text{consistency score} = f_{\text{cross-encoder}}(\text{answer}, \text{context})
  \]
  若分数低于阈值，则拒绝回答或提示“信息不足”。  
- **用户反馈闭环**：用户可标注答案质量，用于强化学习（RLHF）改进模型。

### **3.3 权限粒度控制**  
Glean 的权限模型基于 **ABAC（Attribute-Based Access Control）**：
- 用户属性：`role=engineer`, `team=infra`  
- 文档属性：`owner=team:infra`, `sensitivity=high`  
- 检索时动态计算 `is_allowed(user, document)`，过滤无权限内容。

---

## **四、与竞品对比联想**

### **4.1 Glean vs. Microsoft Copilot**  
| 维度 | Glean | Copilot |  
|------|-------|---------|  
| 数据范围 | 多 SaaS 数据源 | 主要 Microsoft 365 生态 |  
| Graph | 自建 Enterprise Graph | 依赖 Microsoft Graph |  
| LLM | 多模型支持（Google AI + 开源） | 主要依赖 OpenAI（Azure） |  
| 开放代理 | 提供 Agents 平台 | 主要在 Office 应用内 |  

### **4.2 Glean vs. Google Agentspace**  
- **Agentspace**：Google 的企业搜索代理，重点在与大模型深度集成和 Google Workspace 原生。  
- **Glean**：独立平台，强调多源数据和自定义代理，与 Google Cloud 合作但非绑定。  
  两者都使用 **RAG + agent** 架构，但 Glean 的 graph 更成体系。

参考链接：  
https://www.credal.ai/blog/glean-vs-agentspace

---

## **五、应用场景举例**

1. **技术支持自动化**：  
   - 输入："客户报告 API 限流错误，如何解决？"  
   - Glean 检索内部文档、代码库、Slack 讨论，生成包含 **配置调整步骤** 和 **监控建议** 的答案。

2. **新员工入职**：  
   - 输入："如何配置本地开发环境？"  
   - 聚合多个团队的 README、Wiki、教程视频，生成个性化清单。

3. **合规问答**：  
   - 输入："GDPR 关于数据保留的规定是什么？"  
   - 检索法律文件、内部政策，并标注最新修订日期。

---

## **六、技术趋势与未来方向**

1. **多模态扩展**：支持图片、音频、视频的 embedding 检索（如会议录音转文本后检索）。  
2. **主动推送**：基于用户角色和项目，主动推送相关知识更新（类似 **alerting**）。  
3. **联邦学习**：在不共享原始数据的前提下训练模型，满足数据驻留要求。  
4. **AI Agent 编排**：支持跨代理的 **或chestration**，例如 "分析 Q3 销售趋势 → 生成 PPT → 邮件发送给管理层"。

---

## **总结**  
Glean 的本质是一个 **企业知识大脑**，通过 **graph + RAG + agents** 的融合，将分散的企业数据转化为可理解、可操作的知识。其技术栈与 Google Cloud 深度绑定，同时保持 openness。对于企业用户，它解决了 **数据孤岛** 和 **信息检索低效** 的核心痛点，是 **First-principle** 上对 "enterprise search" 的重构。

参考链接汇总：  
- 官方介绍：https://www.glean.com/  
- Google Cloud 合作：https://cloud.google.com/blog/products/data-analytics/glean-uses-bigquery-and-google-ai-to-enhance-enterprise-search  
- AI 代理架构：https://www.glean.com/blog/7-core-components-of-an-ai-agent-architecture-explained  
- RAG 指南：https://www.glean.com/blog/rag-for-llms  
- Agentic Reasoning：https://www.glean.com/blog/agentic-reasoning-future-ai  
- 对比分析：https://www.credal.ai/blog/glean-vs-agentspace