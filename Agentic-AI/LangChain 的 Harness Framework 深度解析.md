## 核心概念定位

LangChain 的 **Harness framework** 实际上指的是 **DeepAgents**，这是一个更高层次的 **agent harness** 架构。在 LangChain 生态系统中，存在三个不同层次的抽象：

| Layer | Component | Value Add |
|-------|-----------|-----------|
| Framework | LangChain | Abstractions, Integrations |
| Runtime | LangGraph | Durable execution, Streaming, HITL, Persistence |
| **Harness** | **DeepAgents** | Predefined tools, Prompts, Subagents |

**DeepAgents** 被定义为"batteries-included"的 **agent harness**，它构建在 LangGraph 之上，提供了默认的 prompts、opinionated 的 tool calls 处理、planning tools、filesystem access 等完整功能集。

## 架构分层详解

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepAgents Harness                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Planning   │  │  Filesystem  │  │   Subagents  │      │
│  │   Middleware │  │  Middleware  │  │  Middleware  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                           │                                  │
│                           ▼                                  │
│                    ┌──────────────┐                          │
│                    │  LangGraph   │  ← Agent Runtime         │
│                    │   Runtime    │                          │
│                    └──────────────┘                          │
│                           │                                  │
│                           ▼                                  │
│                    ┌──────────────┐                          │
│                    │   LangChain  │  ← Agent Framework       │
│                    │  Abstractions│                          │
│                    └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## 核心技术组件

### 1. Filesystem Access System

DeepAgents 将 filesystem 作为 **first-class citizen**，提供六个核心工具：

| Tool | 功能描述 | 技术实现 |
|------|----------|----------|
| `ls` | 列出目录文件及 metadata | 返回 `FileInfo` 对象列表，包含 `path`, `size`, `modified_at` |
| `read_file` | 读取文件内容，支持 offset/limit | 分页读取大文件，避免 context 溢出 |
| `write_file` | 创建新文件 | 原子写入操作，返回 `WriteResult(path, files_update)` |
| `edit_file` | 精确字符串替换 | 支持全局替换模式 `replace_all=True` |
| `glob` | 模式匹配文件查找 | 使用 `**/*.py` 等模式表达式 |
| `grep` | 文件内容搜索 | 多种输出模式：files only, content with context, counts |

#### Backend 架构

DeepAgents 采用 **pluggable backend** 设计，实现 `BackendProtocol` 接口：

```python
class BackendProtocol:
    def ls_info(self, path: str) -> list[FileInfo]:
        """列出路径下的文件信息"""
        
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """读取文件内容，支持分页"""
        
    def write(self, file_path: str, content: str) -> WriteResult:
        """写入文件，返回路径和状态更新"""
        
    def edit(self, file_path: str, old_string: str, new_string: str, 
             replace_all: bool = False) -> EditResult:
        """编辑文件内容"""
        
    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """模式匹配查找文件"""
        
    def grep_raw(self, pattern: str, path: str | None = None, 
                 glob: str | None = None) -> list[GrepMatch] | str:
        """正则表达式搜索文件内容"""
```

#### 四种 Backend 实现

**1. StateBackend (Ephemeral)**

```python
# 默认实现
agent = create_deep_agent()
# 等价于
from deepagents.backends import StateBackend
agent = create_deep_agent(backend=lambda rt: StateBackend(rt))
```

- 存储位置：LangGraph agent state
- 生命周期：单个 thread 内持久化
- 使用场景：scratch pad、中间结果存储
- 特性：supervisor agent 和 subagents 共享状态

**2. FilesystemBackend (Local Disk)**

```python
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir=".", virtual_mode=True)
)
```

- 存储位置：真实文件系统
- 安全机制：`virtual_mode=True` 启用 sandbox
- 路径验证：阻止 `..`、`~` 和绝对路径逃逸
- 使用场景：本地开发、CI/CD pipeline

**3. StoreBackend (Durable)**

```python
from langgraph.store.memory import InMemoryStore
from deepagents.backends import StoreBackend

agent = create_deep_agent(
    backend=lambda rt: StoreBackend(rt),
    store=InMemoryStore()
)
```

- 存储位置：LangGraph BaseStore
- 持久化：跨 thread 保存
- 命名空间：按 `assistant_id` 隔离
- 使用场景：长期记忆、知识库

**4. CompositeBackend (Router)**

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

composite_backend = lambda rt: CompositeBackend(
    default=StateBackend(rt),
    routes={
        "/memories/": StoreBackend(rt),
        "/docs/": CustomDocBackend()
    }
)

agent = create_deep_agent(
    backend=composite_backend,
    store=InMemoryStore()
)
```

- 路由规则：longest-prefix matching
- 示例路由：
  - `/workspace/plan.md` → `StateBackend`
  - `/memories/agent.md` → `StoreBackend`
  - `/docs/api.md` → `CustomDocBackend`

### 2. Context Management Engine

DeepAgents 实现了三层次的 **context compression** 机制：

#### Token Threshold 公式

```
T_evict = 20,000 tokens  # Large tool result eviction threshold
T_summarize = 0.85 × T_max  # Summarization trigger threshold
T_keep = 0.10 × T_max  # Recent context retention ratio
```

其中：
- `T_evict`：tool result 自动驱逐阈值
- `T_max`：模型的 `max_input_tokens`（从 model profile 获取）
- `T_summarize`：历史记录压缩触发点
- `T_keep`：保留的最近 context 比例

#### Compression 策略

**1. Offloading Large Tool Results**

```python
# FilesystemMiddleware 的核心逻辑
if token_count(result) > T_evict:
    file_path = write_to_backend(result)
    truncated_preview = get_preview(result, lines=10)
    return f"[Result offloaded to {file_path}]\n{truncated_preview}"
```

- 触发时机：tool result > 20,000 tokens
- 存储位置：根据 backend 配置
- 替换内容：文件引用 + 前 10 行预览
- 可配置参数：`tool_token_limit_before_evict`

**2. Offloading Large Tool Inputs**

当 context 超过 85% 时，自动清理旧的 write/edit tool calls：

```
Condition: token_usage > 0.85 × T_max
Action: Truncate old tool call arguments
Result: Replace with file pointer
```

**3. Summarization**

当 offloading 无法释放足够空间时：

```
Input: Full conversation history M = [m₁, m₂, ..., mₙ]
Process:
    1. Generate structured summary S = LLM(M)
       S = {
           "session_intent": string,
           "artifacts_created": list,
           "next_steps": list
       }
    2. Write M to filesystem as canonical record
    3. Replace M with S in working memory
Output: Compressed history H = [system_msg(S), recent_messages]
```

#### 实验数据

在 **terminal-bench** 上的性能表现：

| Metric | Value |
|--------|-------|
| Token reduction (summarization) | ~85% (turn 20) |
| Token reduction (tool eviction) | ~40% (turn 40) |
| Compression trigger frequency | 1-3 events per task (at 85% threshold) |
| Goal preservation rate | >95% (targeted evals) |

### 3. Task Delegation (Subagents)

**SubAgentMiddleware** 实现了 **context isolation** 和 **parallel execution**：

```python
from deepagents.middleware.subagents import SubAgentMiddleware

SubAgentMiddleware(
    default_model="claude-sonnet-4-5-20250929",
    default_tools=[],
    subagents=[
        {
            "name": "code-reviewer",
            "description": "Review code quality and suggest improvements",
            "system_prompt": "You are a senior code reviewer...",
            "tools": [lint_tool, security_scan_tool],
            "model": "claude-sonnet-4-5-20250929",
            "middleware": [CustomLoggingMiddleware()]
        }
    ]
)
```

#### Subagent 工作流

```
Main Agent
    │
    ├─► task tool invocation
    │   ├─► Create ephemeral subagent
    │   ├─► Execute with isolated context
    │   └─► Return single final report
    │
    └─► Compress result into main context
```

#### 优势分析

| Feature | Benefit |
|---------|---------|
| Context isolation | 防止 subagent 工作污染主 agent context |
| Parallel execution | 多个 subagents 可并发运行 |
| Specialization | 不同 subagents 配备不同 tools |
| Token efficiency | 大型 subtask context 压缩为单个结果 |

#### Custom Subagent 实现

```python
from deepagents import CompiledSubAgent
from langgraph.graph import StateGraph

# 构建自定义 LangGraph graph
def create_weather_graph():
    workflow = StateGraph(...)
    # 构建自定义 workflow
    return workflow.compile()

weather_graph = create_weather_graph()
weather_subagent = CompiledSubAgent(
    name="weather",
    description="This subagent can get weather in cities.",
    runnable=weather_graph
)

agent = create_deep_agent(
    middleware=[
        SubAgentMiddleware(
            default_model="claude-sonnet-4-5-20250929",
            subagents=[weather_subagent]
        )
    ]
)
```

### 4. To-Do List Tracking

**TodoListMiddleware** 提供结构化任务管理：

```python
from langchain.agents.middleware import TodoListMiddleware

TodoListMiddleware(
    system_prompt="Use the write_todos tool to track your progress"
)
```

#### Task 状态模型

```python
TodoState = {
    "tasks": [
        {
            "id": str,
            "description": str,
            "status": Literal["pending", "in_progress", "completed"],
            "created_at": datetime,
            "updated_at": datetime
        }
    ]
}
```

#### 工作流

```
1. Agent receives complex task
    ↓
2. Invokes write_todos tool
    ↓
3. Generates structured task list
    ↓
4. Executes tasks sequentially/parallel
    ↓
5. Updates status dynamically
    ↓
6. Reports completion
```

### 5. Human-in-the-Loop (HITL)

通过 `interrupt_on` 参数实现：

```python
agent = create_deep_agent(
    interrupt_on={
        "edit_file": True,  # 每次编辑前暂停
        "write_file": True  # 每次写入前暂停
    }
)
```

#### 工作流程

```
Agent Decision
    ↓
Tool Call Triggered
    ↓
Check interrupt_on
    ↓
[If True] → Pause execution
    ↓
Human Review/Modify
    ↓
Resume/Cancel
```

### 6. Prompt Caching (Anthropic)

针对 Anthropic 模型的优化：

```
Cache Strategy:
    1. Identify repeated prompt segments
    2. Cache system prompts (5k+ tokens)
    3. Skip caching for non-Anthropic models
    4. Performance: ~10x speedup, ~10x cost reduction
```

### 7. Streaming Architecture

基于 LangGraph streaming system：

```
Events:
    ├─► tool_call_start
    ├─► tool_call_result  
    ├─► llm_token_stream
    ├─► subagent_event
    └─► agent_step_complete
```

## 实际应用示例

### 配置完整的 DeepAgent

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
from langgraph.store.memory import InMemoryStore

# 混合存储后端
composite_backend = lambda rt: CompositeBackend(
    default=StateBackend(rt),
    routes={
        "/memories/": StoreBackend(rt),
        "/workspace/": FilesystemBackend(root_dir="./workspace", virtual_mode=True)
    }
)

# 创建 agent
agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    backend=composite_backend,
    store=InMemoryStore(),
    system_prompt="""\
You are an expert coding assistant. Your job is to:
1. Understand the user's requirements
2. Plan the implementation using write_todos
3. Execute tasks step by step
4. Maintain context using the filesystem
5. Delegate specialized tasks to subagents
    """,
    tools=[custom_search_tool, code_execution_tool],
    skills=["./skills/", "./docs/"],
    memory=["./AGENTS.md"],
    interrupt_on={
        "edit_file": True,
        "write_file": False
    }
)
```

### 执行流程

```python
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Build a REST API for user management"}
        ]
    },
    config={"configurable": {"thread_id": "user-api-123"}}
)

# Streaming version
async for event in agent.astream_events(
    {
        "messages": [
            {"role": "user", "content": "Build a REST API for user management"}
        ]
    },
    config={"configurable": {"thread_id": "user-api-123"}}
):
    print(f"Event: {event}")
```

## 性能优化策略

### Context Management 优化

```python
# 激进压缩配置（用于 stress testing）
aggressive_config = {
    "tool_token_limit_before_evict": 5000,  # 降低驱逐阈值
    "summarization_threshold": 0.25,  # 更早触发压缩
    "keep_ratio": 0.15  # 保留更多最近 context
}
```

### Backend 选择策略

| 场景 | 推荐 Backend | 理由 |
|------|--------------|------|
| 临时计算 | StateBackend | 零开销，自动清理 |
| 长期记忆 | StoreBackend | 跨 thread 持久化 |
| 本地开发 | FilesystemBackend | 直接访问文件系统 |
| 混合需求 | CompositeBackend | 灵活路由 |

## 最佳实践建议

1. **Context Management**: 从真实 benchmark 开始，然后 stress-test 单个特性
2. **Recoverability Testing**: 实现 needle-in-haystack 评估验证信息恢复能力
3. **Goal Drift Monitoring**: 监控 summarization 后的目标保持率
4. **Backend Security**: 生产环境始终使用 `virtual_mode=True`
5. **Skill Organization**: 按功能分组 skills，实现 progressive disclosure

## 参考链接

- [Agent harness capabilities - Docs by LangChain](https://docs.langchain.com/oss/python/deepagents/harness)
- [Frameworks, runtimes, and harnesses - Docs by LangChain](https://docs.langchain.com/oss/python/concepts/products)
- [Agent Frameworks, Runtimes, and Harnesses- oh my! - LangChain Blog](https://www.blog.langchain.com/agent-frameworks-runtimes-and-harnesses-oh-my/)
- [Customize Deep Agents - Docs by LangChain](https://docs.langchain.com/oss/python/deepagents/customization)
- [Backends - Docs by LangChain](https://docs.langchain.com/oss/python/deepagents/backends)
- [Deep Agents Middleware - Docs by LangChain](https://docs.langchain.com/oss/python/deepagents/middleware)
- [Context Management for Deep Agents - LangChain Blog](https://blog.langchain.com/context-management-for-deepagents/)
- [GitHub - langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)
- [Build a Research Agent with Deep Agents - YouTube](https://www.youtube.com/watch?v=5tn6O0uXYEg)
- [What are Deep Agents? - YouTube](https://www.youtube.com/watch?v=IVts6ztrkFg)
- [Deep Agents CLI - Interactive CLI](https://docs.langchain.com/oss/python/deepagents/cli)