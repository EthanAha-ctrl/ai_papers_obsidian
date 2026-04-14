深入探讨了 **Codex CLI** 的核心架构——**Agent Loop（智能体循环）**
文章聚焦于 **Codex Harness**——这是所有Codex产品共享的**核心Agent循环和执行逻辑**
## 二、核心概念

### 2.1 什么是Agent Loop？
**Agent Loop（智能体循环）** 是所有AI Agent的核心逻辑，负责编排三方交互：

```
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT LOOP ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌──────────┐      ┌──────────┐      ┌──────────────────┐     │
│    │   USER   │─────▶│  AGENT   │◀────▶│  MODEL (LLM)     │     │
│    │  INPUT   │      │  HARNESS │      │  (Inference)     │     │
│    └──────────┘      └────┬─────┘      └──────────────────┘     │
│                           │                                     │
│                           ▼                                     │
│                    ┌──────────────┐                             │
│                    │    TOOLS     │                             │
│                    │  (shell,     │                             │
│                    │   web_search,│                             │
│                    │   MCP, etc.) │                             │
│                    └──────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Loop 的数学形式化

可以将Agent Loop形式化为一个**迭代过程**：

$$
\text{Let } \mathcal{P}_0 = \text{UserInput} + \text{SystemInstructions} + \text{Tools}
$$
$$
\text{For } t = 0, 1, 2, \ldots:
$$

$$
\text{Output}_t = \text{Model}(\mathcal{P}_t) = \begin{cases}
\text{FinalResponse} & \text{if terminal state} \\
\text{ToolCall}(name, args) & \text{if tool needed}
\end{cases}
$$

$$
\text{If tool call: } \mathcal{P}_{t+1} = \mathcal{P}_t \oplus \text{ToolOutput}
$$

其中：
- $\mathcal{P}_t$ = 第 $t$ 轮迭代的 prompt
- $\oplus$ = append操作（追加到prompt末尾）
- $\text{Model}(\cdot)$ = LLM推理函数
- $t$ = iteration counter（迭代计数器）

### 2.3 Turn vs Iteration 的区别

| 概念 | 定义 | 包含内容 |
|------|------|---------|
| **Turn（对话轮次）** | 从用户输入到Agent响应的完整周期 | 可能包含多个model inference + tool call迭代 |
| **Iteration（迭代）** | 单次inference + tool execution | 一次模型调用 + 工具执行 |

```
TURN STRUCTURE:
┌────────────────────────────────────────────────────────────┐
│                        ONE TURN                            │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ User    │──▶│ Infer 1  │──▶│ Tool 1   │──▶│ Infer 2  │  │
│  │ Message │   │          │   │ Execute  │   │          │  │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                   │                        │
│                                   ▼                        │
│                              ┌──────────┐                  │
│                              │ Tool 2   │                  │
│                              │ Execute  │                  │
│                              └──────────┘                  │
│                                   │                        │
│                                   ▼                        │
│                              ┌──────────┐                  │
│                              │ Infer 3  │                  │
│                              │ (Final)  │                  │
│                              └──────────┘                  │
│                                   │                        │
│                                   ▼                        │
│                              ┌──────────┐                  │
│                              │Assistant │                  │
│                              │ Message  │                  │
│                              └──────────┘                  │
└────────────────────────────────────────────────────────────┘
```

---

## 三、Model Inference 流程详解

### 3.1 Responses API 架构

Codex CLI 通过 **Responses API** 与模型交互，这是一个统一的HTTP端点设计：

```
RESPONSES API ENDPOINTS:
┌───────────────────────────────────────────────────────────────┐
│  Authentication Method     │  Endpoint URL                    │
├────────────────────────────┼──────────────────────────────────┤
│  ChatGPT Login             │  https://chatgpt.com/backend-api/│
│                            │  codex/responses                 │
├────────────────────────────┼──────────────────────────────────┤
│  API Key (OpenAI hosted)   │  https://api.openai.com/v1/      │
│                            │  responses                       │
├────────────────────────────┼──────────────────────────────────┤
│  Local OSS (ollama/LM      │  http://localhost:11434/v1/      │
│  Studio)                   │  responses                       │
├────────────────────────────┼──────────────────────────────────┤
│  Cloud Provider (Azure)    │  {provider_endpoint}/v1/         │
│                            │  responses                       │
└───────────────────────────────────────────────────────────────┘
```

**关键洞察**：Responses API是一个**抽象层**，允许Codex CLI与任何兼容的后端交互，实现了"write once, run anywhere"的灵活性。

### 3.2 Tokenization 与 Streaming

文章揭示了inference的底层过程：

$$
\text{Text} \xrightarrow{\text{Tokenizer}} \text{Tokens} = [t_1, t_2, \ldots, t_n]
$$

$$
\text{Tokens} \xrightarrow{\text{Model}} \text{OutputTokens} = [o_1, o_2, \ldots, o_m]
$$

$$
\text{OutputTokens} \xrightarrow{\text{Detokenizer}} \text{ResponseText}
$$

**Streaming的实现原理**：由于tokens是**增量生成**的（autoregressive generation），detokenization可以并行进行：

$$
\text{At step } i: \text{display } \text{Detokenize}([o_1, \ldots, o_i])
$$

这解释了为什么LLM应用可以显示流式输出——模型一边生成，前端一边解码显示。

---

## 四、Prompt 构建机制

### 4.1 Prompt 的结构化表示

Prompt不是一段纯文本，而是一个**结构化的items列表**：

```json
PROMPT STRUCTURE (as list of items):
[
  { "role": "system", "content": "..." },      // 最高优先级
  { "role": "developer", "content": "..." },   // 次高优先级
  { "role": "user", "content": "..." },        // 用户输入
  { "role": "assistant", "content": "..." },   // 助手回复
  { "type": "function_call", ... },            // 工具调用
  { "type": "function_call_output", ... },     // 工具输出
  { "type": "reasoning", ... },                // 推理过程
  { "type": "compaction", ... }                // 压缩标记
]
```

### 4.2 Role 优先级体系

文章首次公开了OpenAI模型的**Role优先级体系**：

$$
\text{Priority}(\text{system}) > \text{Priority}(\text{developer}) > \text{Priority}(\text{user}) > \text{Priority}(\text{assistant})
$$

这意味着：
- **system role**：服务器控制，最高权重（模型核心行为约束）
- **developer role**：客户端可指定，次高权重（应用层指令）
- **user role**：用户输入
- **assistant role**：模型历史输出

### 4.3 Initial Prompt 的构建流程

```
INITIAL PROMPT CONSTRUCTION FLOW:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. SYSTEM MESSAGE (server-controlled)                          │
│     └── Model-specific base instructions                        │
│                                                                 │
│  2. TOOLS DEFINITION                                            │
│     ├── Codex built-in tools (shell, update_plan)               │
│     ├── Responses API tools (web_search)                        │
│     └── MCP server tools (user-configured)                      │
│                                                                 │
│  3. DEVELOPER INSTRUCTIONS (from config.toml)                   │
│     ├── model_instructions_file                                 │
│     └── developer_instructions                                  │
│                                                                 │
│  4. USER INSTRUCTIONS (aggregated from multiple sources)        │
│     ├── AGENTS.override.md / AGENTS.md in $CODEX_HOME          │
│     ├── Project-level AGENTS.md files                           │
│     └── Skill metadata (if configured)                          │
│                                                                 │
│  5. ENVIRONMENT CONTEXT                                         │
│     ├── Current working directory                               │
│     └── User's shell (zsh, bash, etc.)                          │
│                                                                 │
│  6. USER MESSAGE                                                │
│     └── The actual user request                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Tools Schema 定义

文章展示了tools的JSON Schema定义：

```json
{
  "type": "function",
  "name": "shell",
  "description": "Runs a shell command and returns its output...",
  "strict": false,  // 非严格模式，允许灵活参数
  "parameters": {
    "type": "object",
    "properties": {
      "command": {"type": "array", "description": "The command to execute"},
      "workdir": {"description": "The working directory..."},
      "timeout_ms": {"description": "The timeout for the command..."}
    },
    "required": ["command"]
  }
}
```

**关键参数解释**：
- `strict: false`：允许模型生成schema中未定义的参数，增加灵活性
- `parameters`：符合JSON Schema规范，定义工具输入的结构

---

## 五、Performance Considerations（性能考量）

### 5.1 Quadratic vs Linear Complexity

这是文章最精彩的技术洞察之一。

#### 问题：Agent Loop的复杂度分析

假设一个conversation有 $n$ 个turns，每个turn平均有 $m$ 次tool iterations：

**Naive Approach（不使用缓存）**：

$$
\text{TotalTokensSent} = \sum_{i=1}^{n} \sum_{j=1}^{m_i} |\mathcal{P}_{i,j}|
$$

由于 $\mathcal{P}_{i,j} \supseteq \mathcal{P}_{i,j-1}$（prompt只增不减），最坏情况下：

$$
\text{Complexity} = O(n^2 \cdot m \cdot \bar{L})
$$

其中 $\bar{L}$ 是平均prompt长度。

**With Prompt Caching**：

$$
\text{If cache hit: } \text{Cost}(\mathcal{P}_{t+1}) = \text{Cost}(\mathcal{P}_t) + \text{Cost}(\Delta \mathcal{P})
$$

因为旧prompt是exact prefix，可以复用计算：

$$
\text{Complexity}_{\text{cached}} = O(n \cdot m \cdot \bar{L}) \quad \text{(Linear!)}
$$

```
COMPLEXITY COMPARISON:
│
│  Tokens      ────────────────────────────────
│  Sent        │\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\  Quadratic (no cache)
│              │\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
│              │\\\\\\\\\\\\\\\\\\
│              │\\\\\\\\\\\\
│              │\\\\\\
│              │\\
│              ────────────────────────────────  Linear (with cache)
│              │
│              └────────────────────────────────▶ Time/Turns
```

### 5.2 Why Not Use `previous_response_id`？

Responses API提供 `previous_response_id` 参数来避免重复传输，但Codex**选择不使用**：

| 考量因素 | 决策 |
|---------|------|
| **Stateless Requests** | ✅ 每个请求独立，简化服务端实现 |
| **Zero Data Retention (ZDR)** | ✅ 支持"零数据留存"客户需求 |
| **Encrypted Reasoning** | ✅ 推理内容通过 `encrypted_content` 传递，服务端可解密但不存储 |
| **Cost Trade-off** | ❌ 网络传输成本高于推理成本 |

**ZDR的实现细节**：

$$
\text{ZDR Customer Data Flow}:
$$

$$
\text{Reasoning} \xrightarrow{\text{Encrypt with } K_{\text{customer}}} \text{encrypted\_content}
$$

$$
\text{OpenAI stores } K_{\text{customer}} \text{ but NOT the data itself}
$$

$$
\text{On next request: } \text{encrypted\_content} \xrightarrow{\text{Decrypt with } K_{\text{customer}}} \text{Reasoning}
$$

参考PR：#642 和 #1641（文章中提及）

### 5.3 Cache Miss 的触发条件

文章列举了会导致缓存失效的操作：

```python
# Cache Miss Triggers:
class CacheMissTrigger:
    def __init__(self):
        self.triggers = [
            "Changing tools mid-conversation",      # 工具列表变化
            "Switching model",                       # 模型切换（改变system message）
            "Modifying sandbox config",              # 沙箱配置变化
            "Changing approval mode",                # 审批模式变化
            "Changing working directory",            # 工作目录变化
            "MCP tools list change notification",    # MCP工具动态变化
        ]
```

**MCP Tools的特殊问题**：

MCP (Model Context Protocol) 服务器可以通过 `notifications/tools/list_changed` 动态通知工具列表变化。如果在长对话中处理此通知，会导致**昂贵的cache miss**。

### 5.4 Cache-Preserving 设计模式

Codex采用的策略：**Append而非Modify**

```python
# Instead of modifying earlier message:
def handle_config_change(change_type, new_value):
    # ❌ BAD: Modify existing message
    # prompt[2]["content"] = new_value  # Causes cache miss!
    
    # ✅ GOOD: Append new message
    if change_type == "sandbox_config":
        prompt.append({
            "role": "developer",
            "content": f"<permissions instructions>\n{new_value}\n</permissions_instructions>"
        })
    elif change_type == "working_directory":
        prompt.append({
            "role": "user", 
            "content": f"<environment_context>\n<cwd>{new_value}</cwd>\n</environment_context>"
        })
```

---

## 六、Context Window Management（上下文窗口管理）

### 6.1 Context Window 约束

每个模型都有**context window**限制：

$$
\text{ContextWindow} = \text{MaxInputTokens} + \text{MaxOutputTokens}
$$

Agent可能在一个turn中执行数百次tool calls，快速消耗context window：

$$
\text{Risk: } \sum_{i=1}^{n} |\text{ToolOutput}_i| > \text{ContextWindow}
$$

### 6.2 Compaction（压缩）机制

Codex采用**compaction**策略来管理上下文：

```
COMPACTION EVOLUTION:
┌─────────────────────────────────────────────────────────────┐
│  V1: Manual Compaction                                      │
│  ├── User invokes /compact command                          │
│  ├── Query model with summarization instructions            │
│  └── Use assistant message as new input                     │
├─────────────────────────────────────────────────────────────┤
│  V2: Automatic Compaction via /responses/compact            │
│  ├── Triggered when auto_compact_limit exceeded             │
│  ├── Dedicated API endpoint for efficiency                  │
│  └── Returns compacted item list                            │
└─────────────────────────────────────────────────────────────┘
```

**Compaction的输出结构**：

```json
[
  {
    "type": "compaction",
    "encrypted_content": "opaque_data_preserving_latent_understanding"
  },
  {
    "type": "message",
    "role": "assistant", 
    "content": "Summary of conversation so far..."
  }
]
```

**关键创新**：`encrypted_content` 保留了模型的**latent understanding（潜在理解）**，即使原始对话被压缩，模型仍能"记住"之前的推理过程。

---

## 七、SSE（Server-Sent Events）流式响应

### 7.1 Event Types

文章展示了Responses API返回的SSE事件流：

```
SSE EVENT STREAM EXAMPLE:
data: {"type":"response.reasoning_summary_text.delta","delta":"ah ", ...}
data: {"type":"response.reasoning_summary_text.delta","delta":"ha!", ...}
data: {"type":"response.reasoning_summary_text.done", "item_id":...}
data: {"type":"response.output_item.added", "item":{...}}
data: {"type":"response.output_text.delta", "delta":"forty-", ...}
data: {"type":"response.output_text.delta", "delta":"two!", ...}
data: {"type":"response.completed","response":{...}}
```

**Event Taxonomy**：

| Event Type | Purpose | Usage |
|-----------|---------|-------|
| `response.reasoning_summary_text.delta` | 推理过程增量 | 显示模型思考过程 |
| `response.output_text.delta` | 输出文本增量 | UI streaming显示 |
| `response.output_item.added` | 新输出项添加 | 追加到input数组 |
| `response.completed` | 响应完成 | 标记turn结束 |

### 7.2 Incremental State Building

每次tool call后，prompt结构增长：

```json
// After tool call iteration:
[
  /* ... original input ... */,
  {
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "I need to..."}],
    "encrypted_content": "gAAAAABpaDWNMxMeLw..."
  },
  {
    "type": "function_call",
    "name": "shell",
    "arguments": "{\"command\":\"cat README.md\"}",
    "call_id": "call_8675309..."
  },
  {
    "type": "function_call_output",
    "call_id": "call_8675309...",
    "output": "<p align=\"center\">..."
  }
]
```

---

## 八、关键技术创新总结

### 8.1 Exact Prefix Matching for Cache

$$
\mathcal{P}_{t+1} = \mathcal{P}_t \parallel \Delta \mathcal{P}
$$

$$
\text{where } \mathcal{P}_t \text{ is an exact prefix of } \mathcal{P}_{t+1}
$$

这确保了每次请求都能利用之前的计算结果。

### 8.2 Encrypted Reasoning Preservation

$$
\text{Reasoning}_t \xrightarrow{\text{Encrypt}} \text{encrypted\_content}_t \xrightarrow{\text{Decrypt at } t+1} \text{Reasoning}_t
$$

即使采用ZDR，模型仍能在后续turn中访问之前的推理过程。

### 8.3 Stateless Request Design

$$
\text{Request}_{t+1} = \text{FullPrompt}_{t+1}
$$

$$
\not\exists \text{ServerState}_t \text{ that needs to be referenced}
$$

这简化了服务端实现，支持水平扩展和ZDR合规。

---

## 九、参考链接

1. **Codex GitHub Repository**: https://github.com/openai/codex
2. **Responses API Documentation**: https://platform.openai.com/docs/api-reference/responses
3. **Prompt Caching Documentation**: https://platform.openai.com/docs/guides/prompt-caching
4. **Zero Data Retention (ZDR)**: https://platform.openai.com/docs/guides/enterprise-data-privacy
5. **MCP (Model Context Protocol)**: https://modelcontextprotocol.io/
6. **Server-Sent Events (SSE) Spec**: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events

---

## 十、对Agent开发的启示

1. **Prompt结构化至关重要**：不要把prompt当作纯文本，而应视为结构化数据结构
2. **缓存友好设计**：将静态内容前置，动态内容后置，确保prefix匹配
3. **增量状态管理**：通过append而非modify来保持状态一致性
4. **Context window主动管理**：实现自动compaction机制，避免被动溢出
5. **抽象层设计**：通过统一API（如Responses API）实现后端无关的架构

这篇文章是OpenAI首次如此详细地公开其Agent系统的内部实现，对于任何想要构建production-grade AI Agent的开发者来说，都是必读的参考资料。