

我来帮你详细了解 OpenAI 的 Responses API。让我先搜索最新信息并解析官方文档。











看起来搜索工具暂时无法返回结果。不过，基于我对 OpenAI Responses API 的深入了解（截至我的训练数据），让我为你做一个**极其详尽**的技术讲解。

---

# OpenAI Responses API — 全面深度解析

> 参考链接: https://developers.openai.com/api/reference/responses/overview
> https://platform.openai.com/docs/api-reference/responses

---

## 一、第一性原理：为什么需要 Responses API？

### 1.1 从 Chat Completions API 的局限说起

传统的 **Chat Completions API** (`POST /v1/chat/completions`) 是一个 **stateless（无状态）** 的接口：

```
你发送 messages[] → 模型返回一个 completion → 结束
```

这意味着：
- 每次调用你都要**手动管理整个 conversation history**
- 如果模型需要调用 **tool（function calling）**，你必须：
  1. 收到 `tool_calls` 的 response
  2. 自己执行 tool
  3. 把 tool result 拼回 `messages[]`
  4. 再次调用 API
- **没有内置的 web search、file search、code interpreter** 等能力——这些只存在于 Assistants API 中
- **Assistants API** 虽然有这些能力，但它又太重了（需要 Thread、Run、polling 等概念）

**第一性原理问题**：能不能有一个 API，既像 Chat Completions 一样**简洁（single endpoint call）**，又像 Assistants API 一样**强大（built-in tools + stateful）**？

→ 答案就是 **Responses API**。

### 1.2 设计哲学

Responses API 的核心设计理念是：

```
Primitive enough to be flexible, powerful enough to be agentic.
（足够原始以保持灵活，足够强大以支持 agentic workflow）
```

---

## 二、Responses API 的 Endpoint 与核心结构

### 2.1 基本 Endpoint

```
POST /v1/responses
```

这是一个**单一 endpoint**，取代了：
- `POST /v1/chat/completions`（简单对话）
- `POST /v1/threads/runs`（Assistants 的 agentic 执行）

### 2.2 Request Body 核心参数

```json
{
  "model": "gpt-4o",
  "input": "...",           // 可以是 string 或 structured items[]
  "instructions": "...",     // system-level instructions
  "tools": [...],            // built-in tools + custom function tools
  "tool_choice": "auto",
  "temperature": 0.7,
  "max_output_tokens": 4096,
  "metadata": {},
  "previous_response_id": "resp_abc123",  // 用于 multi-turn 对话
  "stream": true,
  "store": true              // 是否在 OpenAI 侧持久化
}
```

各参数深度解析如下：

---

### 2.3 `input` 参数

`input` 可以是两种形式：

**简单 string 形式**（类似给 ChatGPT 发消息）：
```json
{
  "input": "What is the weather in Tokyo?"
}
```

**结构化 items 形式**（类似 Chat Completions 的 `messages[]`，但更灵活）：
```json
{
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": "Analyze this image"
    },
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "https://..."}
      ]
    }
  ]
}
```

关键区别：`input` 中的 items 不仅仅是 `message`，还可以包含 **tool call results**、**tool outputs** 等，这使得 **multi-step agentic loop** 可以在一个 request 中表达。

---

### 2.4 `instructions` 参数

等价于 Chat Completions 中的 `system` message，但被提升为**顶层参数**：

```json
{
  "instructions": "You are a helpful financial analyst. Always cite sources.",
  "input": "What's happening with NVIDIA stock?"
}
```

**为什么分离出来？** 因为在 agentic workflow 中，`instructions` 是跨 turn 稳定的，而 `input` 每次变化。分离后 OpenAI 可以在 server 端做更好的 **prompt caching**。

---

## 三、Built-in Tools — Responses API 的杀手锏

这是 Responses API 和 Chat Completions API 最大的区别。Responses API 内置了若干 **server-side tools**，模型可以**自主调用，无需你写任何执行代码**。

### 3.1 Tool 类型总览

| Tool Type | `type` 值 | 描述 |
|---|---|---|
| **Web Search** | `web_search_preview` | 实时搜索互联网 |
| **File Search** | `file_search` | 在你上传的 Vector Store 中做 semantic search |
| **Code Interpreter** | `code_interpreter` | 在 sandbox 中执行 Python 代码 |
| **Computer Use** | `computer_use_preview` | 控制虚拟桌面（screenshot → action loop）|
| **Image Generation** | `image_generation` | 调用 DALL·E / gpt-image 生成图片 |
| **Custom Function** | `function` | 你自定义的 function（需要你自己执行）|
| **MCP** | `mcp` | Model Context Protocol 连接外部 tool servers |

### 3.2 Web Search Tool 详解

```json
{
  "tools": [
    {
      "type": "web_search_preview",
      "search_context_size": "medium",  // "low" | "medium" | "high"
      "user_location": {
        "type": "approximate",
        "country": "US",
        "region": "California",
        "city": "San Francisco"
      }
    }
  ]
}
```

**工作原理**（从第一性原理）：

```
User Input
    ↓
Model 决定是否需要搜索（基于 tool_choice）
    ↓
Model 生成 search query（内部 query rewriting）
    ↓
OpenAI server 执行搜索（使用 Bing 或自建 index）
    ↓
搜索结果注入 context（RAG 模式）
    ↓
Model 基于搜索结果生成最终 response
```

`search_context_size` 控制注入多少搜索结果到 context：
- `"low"` → ~少量 snippets，token 省，速度快
- `"medium"` → 平衡
- `"high"` → 大量内容，更准确但更慢更贵

**在 response 中你会看到**：
```json
{
  "output": [
    {
      "type": "web_search_call",
      "id": "ws_abc123",
      "status": "completed"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "According to recent reports...",
          "annotations": [
            {
              "type": "url_citation",
              "start_index": 29,
              "end_index": 45,
              "url": "https://example.com/article",
              "title": "Article Title"
            }
          ]
        }
      ]
    }
  ]
}
```

注意 `annotations` 里的 **url_citation** — 这提供了**可追溯的引用**。

### 3.3 File Search Tool 详解

```json
{
  "tools": [
    {
      "type": "file_search",
      "vector_store_ids": ["vs_abc123"],
      "max_num_results": 20,
      "ranking_options": {
        "ranker": "auto",
        "score_threshold": 0.5
      }
    }
  ]
}
```

**底层架构**：

```
上传文件 → chunking（自动分块）→ embedding（text-embedding-3-large）→ 存入 Vector Store
                                                                          ↓
User Query → embedding → cosine similarity search → top-K chunks → 注入 context → 生成回答
```

**Chunking 策略**：OpenAI 使用 **semantic chunking**，会考虑文档结构（标题、段落边界），默认 chunk size ~800 tokens，overlap ~400 tokens。

**Ranking Options**：
- `ranker: "auto"` — OpenAI 会先做 vector search，再用 **cross-encoder reranker** 对 top candidates 做 re-ranking
- `score_threshold` — 只返回相似度分数高于此阈值的 chunks，公式：

$$\text{score} = \text{cosine\_sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{||\mathbf{q}|| \cdot ||\mathbf{d}||}$$

其中：
- **q** = query 的 embedding vector
- **d** = document chunk 的 embedding vector
- **||·||** = L2 norm（欧几里得范数）

### 3.4 Code Interpreter Tool

```json
{
  "tools": [
    {
      "type": "code_interpreter",
      "container": {
        "type": "auto",
        "file_ids": ["file_abc123"]  // 可注入文件
      }
    }
  ]
}
```

模型可以自主编写和执行 Python 代码，运行在 **sandboxed container** 中。适用于：
- 数据分析（pandas、numpy）
- 绘图（matplotlib）
- 数学计算
- 文件格式转换

### 3.5 Computer Use Tool (Preview)

```json
{
  "tools": [
    {
      "type": "computer_use_preview",
      "display_width": 1024,
      "display_height": 768,
      "environment": "browser"  // "browser" | "mac" | "windows" | "linux"
    }
  ]
}
```

这是最 **agentic** 的 tool。工作循环：

```
Screenshot（base64 image）
    ↓
Model 分析屏幕内容（vision 能力）
    ↓
Model 输出 action：
  - click(x, y)
  - type("text")
  - scroll(direction)
  - keypress("Enter")
  - screenshot()  // 再截一次
    ↓
环境执行 action
    ↓
新的 Screenshot → 回到 Model
    ↓
... 循环直到任务完成
```

### 3.6 Custom Function Tool

```json
{
  "tools": [
    {
      "type": "function",
      "name": "get_stock_price",
      "description": "Get the current stock price for a given ticker symbol",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol, e.g., AAPL"
          }
        },
        "required": ["ticker"]
      },
      "strict": true  // 启用 structured outputs，保证 JSON 严格匹配 schema
    }
  ]
}
```

**`strict: true`** 的底层原理：OpenAI 使用 **constrained decoding**（约束解码），在 token sampling 时，通过 **finite state machine (FSM)** 或 **context-free grammar (CFG)** 来 mask 掉不符合 JSON Schema 的 tokens，从而**100% 保证**输出的 JSON 结构正确。

### 3.7 MCP (Model Context Protocol) Tool

```json
{
  "tools": [
    {
      "type": "mcp",
      "server_label": "my_mcp_server",
      "server_url": "https://my-server.example.com/mcp",
      "allowed_tools": ["tool_a", "tool_b"],  // 白名单
      "require_approval": "always"  // "always" | "never" | 特定 tools
    }
  ]
}
```

**MCP** 是 Anthropic 提出的开放协议，OpenAI 也支持了。它让模型可以通过标准化协议连接任意外部 tool server。

---

## 四、Response Object — 返回结构详解

```json
{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1710000000,
  "model": "gpt-4o-2025-03-01",
  "status": "completed",     // "completed" | "failed" | "incomplete"
  "output": [
    // 按时间顺序排列的 output items
    {
      "type": "web_search_call",
      "id": "ws_001",
      "status": "completed"
    },
    {
      "type": "message",
      "id": "msg_001",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The answer is...",
          "annotations": [...]
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 1500,
    "input_tokens_details": {
      "cached_tokens": 500     // prompt caching 命中的 tokens
    },
    "output_tokens": 800,
    "output_tokens_details": {
      "reasoning_tokens": 200  // o1/o3 等 reasoning model 的思考 tokens
    },
    "total_tokens": 2300
  },
  "incomplete_details": null,
  "error": null
}
```

### 4.1 `output` 是一个 **ordered list of items**

这是关键设计。不像 Chat Completions 只返回一个 `message`，Responses API 的 `output` 是**多个 items 的序列**，可以包含：

| Item Type | 说明 |
|---|---|
| `message` | 文本/图片等输出 |
| `web_search_call` | 执行了一次 web search |
| `file_search_call` | 执行了一次 file search |
| `code_interpreter_call` | 执行了一次代码 |
| `computer_call` | 执行了一次 computer action |
| `function_call` | 请求调用一个 custom function |
| `reasoning` | reasoning model 的思考过程（可选暴露）|
| `image_generation_call` | 生成了一张图片 |

**这意味着一次 response 可以包含多步操作**，例如：

```
[web_search_call] → [code_interpreter_call] → [message]
模型先搜索 → 再用代码处理数据 → 最后给出回答
```

### 4.2 `usage` 中的 Token 计量

```
input_tokens_details.cached_tokens
```
这是 **Prompt Caching** 的体现。当 `previous_response_id` 被使用时，OpenAI 会在 server 端缓存前序对话的 KV cache，后续 turn 的 input 中被缓存的部分**只收 50% 费用**（某些 model 甚至更低）。

```
output_tokens_details.reasoning_tokens
```
对于 **o1、o3、o4-mini** 等 reasoning models，模型会产生内部 chain-of-thought tokens。这些 tokens 在 `output` 中可能不直接可见（除非设置了 `reasoning.effort` 或相关参数），但**会被计费**。

---

## 五、Multi-Turn Conversation — `previous_response_id` 机制

这是 Responses API 最优雅的设计之一。

### 传统方式（Chat Completions）：

```python
# 你必须手动维护 messages 列表
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},      # 手动拼接
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "4"},             # 手动拼接
    {"role": "user", "content": "And multiply by 3?"}, # 新消息
]
# 每次都要发送 ENTIRE history
```

### Responses API 方式：

```python
# Turn 1
response1 = client.responses.create(
    model="gpt-4o",
    input="Hi"
)
# response1.id = "resp_001"

# Turn 2 — 只需传 previous_response_id + 新 input
response2 = client.responses.create(
    model="gpt-4o",
    input="What's 2+2?",
    previous_response_id="resp_001"  # 引用上一轮
)
# response2.id = "resp_002"

# Turn 3
response3 = client.responses.create(
    model="gpt-4o",
    input="And multiply by 3?",
    previous_response_id="resp_002"
)
```

**底层原理**：

```
当 store=true 时（默认），OpenAI 在 server 端保存了每个 response 的完整 context。

当你传入 previous_response_id 时：
1. Server 从存储中加载 resp_002 的完整对话历史
2. 加上你的新 input
3. 组合成完整 prompt 送入模型
4. 同时利用 KV Cache（如果前缀未变，大量 tokens 可以从 cache 中读取）
```

**好处**：
- 客户端无需维护 conversation state
- **Server-side KV cache** 减少延迟和成本
- 对话历史存储在 OpenAI 侧，可通过 `GET /v1/responses/{id}` 检索

---

## 六、Streaming

```python
stream = client.responses.create(
    model="gpt-4o",
    input="Explain quantum computing",
    stream=True
)

for event in stream:
    print(event)
```

### Server-Sent Events (SSE) 事件类型：

```
response.created           → response 对象创建
response.in_progress       → 开始处理

response.output_item.added → 新 output item 开始
response.output_text.delta → 文本增量
response.output_text.done  → 文本完成

response.web_search_call.in_progress → web search 正在执行
response.web_search_call.completed   → web search 完成

response.function_call.arguments.delta → function call 参数增量
response.function_call.arguments.done  → function call 参数完成

response.completed         → 整个 response 完成
response.failed            → 失败
```

这比 Chat Completions 的 streaming 粒度更细，尤其是**你可以实时看到模型调用了哪些 built-in tools**。

---

## 七、与 Chat Completions API 的对比

| 维度 | Chat Completions | Responses API |
|---|---|---|
| **Endpoint** | `/v1/chat/completions` | `/v1/responses` |
| **State** | Stateless | Stateful（via `previous_response_id`）|
| **Input** | `messages[]` | `input`（string 或 items[]）|
| **System Prompt** | `messages[0].role="system"` | `instructions`（顶层参数）|
| **Built-in Tools** | ❌ 无 | ✅ web_search, file_search, code_interpreter, computer_use, image_generation |
| **Custom Functions** | ✅ `tools[]` + `function_calling` | ✅ `tools[].type="function"` |
| **Multi-step Agent Loop** | 手动循环 | Server 端自动执行 built-in tools |
| **Prompt Caching** | 有限 | 原生支持（`previous_response_id` + `cached_tokens`）|
| **Structured Output** | `response_format` | `text.format` 或 function `strict:true` |
| **Deprecation** | 计划在未来逐步淡出 | **新的标准** |

---

## 八、Agentic Loop — Function Calling 的 Multi-Step 执行

当使用 **custom function tools** 时，Responses API 仍然需要你手动执行函数（因为 OpenAI 不知道你的函数逻辑）。流程如下：

```python
# Step 1: 初始调用
response = client.responses.create(
    model="gpt-4o",
    input="What's the weather in Tokyo and New York?",
    tools=[{
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }]
)

# Step 2: response.output 可能包含多个 function_call items
# [function_call(id="fc_1", name="get_weather", arguments='{"city":"Tokyo"}'),
#  function_call(id="fc_2", name="get_weather", arguments='{"city":"New York"}')]

# Step 3: 执行函数，把结果传回
response2 = client.responses.create(
    model="gpt-4o",
    previous_response_id=response.id,
    input=[
        {
            "type": "function_call_output",
            "call_id": "fc_1",
            "output": '{"temp": 22, "condition": "sunny"}'
        },
        {
            "type": "function_call_output", 
            "call_id": "fc_2",
            "output": '{"temp": 15, "condition": "rainy"}'
        }
    ]
)

# Step 4: response2 包含最终的 message
```

注意：**built-in tools（如 web_search）不需要这个循环**，OpenAI server 自己就会执行搜索并把结果注入。只有 **custom function** 需要你参与 loop。

---

## 九、Reasoning Models 的支持

对于 **o1, o3, o3-mini, o4-mini** 等 reasoning models：

```json
{
  "model": "o4-mini",
  "input": "Solve this math problem: ...",
  "reasoning": {
    "effort": "high",        // "low" | "medium" | "high"
    "summary": "auto"        // 是否在 output 中暴露 reasoning 摘要
  }
}
```

**`reasoning.effort`** 控制模型花多少"思考 tokens"在推理上：

| Effort | 行为 |
|---|---|
| `low` | 快速回答，少量 reasoning tokens |
| `medium` | 平衡 |
| `high` | 深度思考，大量 reasoning tokens（更贵但更准）|

**`reasoning.summary`**：当设为 `"auto"` 时，`output` 中会包含一个 `type: "reasoning"` 的 item，展示模型的思考过程摘要。

---

## 十、Structured Output

```json
{
  "model": "gpt-4o",
  "input": "Extract the name and age from: John is 30 years old",
  "text": {
    "format": {
      "type": "json_schema",
      "name": "person_info",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        },
        "required": ["name", "age"],
        "additionalProperties": false
      },
      "strict": true
    }
  }
}
```

**`strict: true` 的技术原理**：

在 autoregressive decoding 中，每一步 token 生成时：

$$P(t_i | t_1, ..., t_{i-1}) = \text{softmax}(\mathbf{z}_i \odot \mathbf{m}_i)$$

其中：
- **t_i** = 第 i 个 token
- **z_i** = logits vector（模型原始输出）
- **m_i** = mask vector，由 JSON Schema 对应的 FSM 状态决定
  - 如果 token t_j 在当前 FSM 状态下合法，m_i[j] = 1
  - 否则 m_i[j] = -∞（即该 token 概率变为 0）
- **⊙** = element-wise multiplication（或 additive masking）

这保证了**每一个 token 都符合 schema 约束**，不可能产生非法 JSON。

---

## 十一、`store` 参数与 Privacy

```json
{
  "store": true,     // 默认 true
  "metadata": {
    "user_id": "user_123",
    "session_id": "sess_456"
  }
}
```

- `store: true` → 对话存储在 OpenAI 侧，可以被 `previous_response_id` 引用，也可以通过 `GET /v1/responses/{id}` 检索
- `store: false` → 对话不持久化，**但也无法使用 `previous_response_id`**
- `metadata` → 你可以附加自定义键值对，方便后续检索和管理

---

## 十二、Background Mode (异步执行)

```json
{
  "model": "o3",
  "input": "Write a comprehensive 50-page research report on...",
  "background": true
}
```

当 `background: true` 时：
- API 立即返回 `response` 对象，`status: "in_progress"`
- 任务在后台执行（可能需要几分钟甚至更长）
- 你可以通过 `GET /v1/responses/{id}` 轮询状态
- 适合**长时间运行的 agentic 任务**

---

## 十三、架构图解

```
┌─────────────────────────────────────────────────────────┐
│                      Client                              │
│                                                          │
│   POST /v1/responses                                     │
│   {                                                      │
│     model, input, instructions, tools,                   │
│     previous_response_id, stream, store                  │
│   }                                                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                 OpenAI API Gateway                        │
│                                                          │
│  ┌─────────────┐   ┌──────────────────────────────────┐ │
│  │ Context      │   │ Response Store                    │ │
│  │ Assembly     │   │ (previous_response_id lookup)     │ │
│  │              │   │ + KV Cache for Prompt Caching     │ │
│  └──────┬──────┘   └──────────────────────────────────┘ │
│         │                                                │
│         ▼                                                │
│  ┌─────────────────────────────────────────┐            │
│  │           LLM Inference                  │            │
│  │  (GPT-4o / o3 / o4-mini / etc.)         │            │
│  │                                          │            │
│  │  Output: text / tool_calls               │            │
│  └──────────┬──────────────────────────────┘            │
│             │                                            │
│     ┌───────┼────────┬──────────┬──────────┐            │
│     ▼       ▼        ▼          ▼          ▼            │
│  ┌──────┐┌──────┐┌────────┐┌────────┐┌─────────┐       │
│  │Web   ││File  ││Code    ││Computer││Image    │       │
│  │Search││Search││Interp. ││Use     ││Gen      │       │
│  │(Bing)││(Vec- ││(Python ││(VM +   ││(DALL·E) │       │
│  │      ││tor   ││sandbox)││Screen) ││         │       │
│  │      ││Store)││        ││        ││         │       │
│  └──┬───┘└──┬───┘└───┬────┘└───┬────┘└────┬────┘       │
│     │       │        │         │          │             │
│     └───────┴────────┴─────────┴──────────┘             │
│                      │                                   │
│                      ▼                                   │
│              Re-inject results into context               │
│              → Model generates final response             │
│                                                          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
              Response Object returned to Client
              (with output[], usage, annotations)
```

---

## 十四、Pricing 模型（概念层面）

Responses API 的定价和 Chat Completions 相同的 per-token pricing，但有额外项：

| 项目 | 费用说明 |
|---|---|
| **Input Tokens** | 标准价 |
| **Cached Input Tokens** | ~50% 折扣 |
| **Output Tokens** | 标准价 |
| **Reasoning Tokens** | 等同 output token 价格 |
| **Web Search** | 按次额外计费（search context size 影响）|
| **File Search** | 按 Vector Store 存储量 + 查询次数 |
| **Code Interpreter** | 按 session 时间计费 |
| **Image Generation** | 按张计费 |

---

## 十五、总结——第一性原理视角

从第一性原理来看，Responses API 解决的核心问题是：

### **Stateless → Stateful 的演进**
LLM API 本质上是 `f(context) → text`。Chat Completions 要求你每次传入完整 context；Responses API 让 server 帮你管理 context，你只需传增量。

### **单一能力 → 复合能力**
传统 API 只做 text generation。现实世界的 agent 需要 search + code + browse + read files。Responses API 把这些能力统一在一个 endpoint 下，模型自主决定何时使用何种工具。

### **Manual orchestration → Server-side orchestration**
对于 built-in tools，server 自动执行 tool → 注入结果 → 继续推理。客户端代码大幅简化。

### **本质上，Responses API 就是 OpenAI 对 "Agent-as-an-API" 的标准化**。

它不再只是一个 "text completion" 接口，而是一个 **"完成任务"** 的接口——模型会自主规划、使用工具、多步推理，最终交付结果。

---

**参考链接**：
- 官方文档: https://developers.openai.com/api/reference/responses/overview
- 官方指南: https://platform.openai.com/docs/guides/responses
- Responses vs Chat Completions: https://platform.openai.com/docs/guides/responses-vs-chat-completions
- Built-in Tools: https://platform.openai.com/docs/guides/tools
- MCP Support: https://platform.openai.com/docs/guides/tools-mcp
- Prompt Caching: https://platform.openai.com/docs/guides/prompt-caching
- Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs