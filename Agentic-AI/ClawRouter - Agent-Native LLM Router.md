
















## ClawRouter: Agent-Native LLM Router 深度解析

### 一、Project 一句话概括

**ClawRouter** 是一个开源（MIT）的智能 LLM 路由器，专为自主 AI Agent（特别是 OpenClaw）设计。它对每个请求在本地进行 14-15 维加权评分（<1ms），自动路由到"最便宜且能胜任"的模型，并通过 x402 协议用 USDC 在 Base/Solana 链上完成微支付。声称可将 AI API 成本降低最高 **92%**。

**GitHub**: [https://github.com/BlockRunAI/ClawRouter](https://github.com/BlockRunAI/ClawRouter)

---

### 二、要解决的问题（第一性原理分析）

从第一性原理出发，LLM API 成本的核心矛盾是：

$$C_{total} = \sum_{i=1}^{N} P_{model_i} \times T_{tokens_i}$$

其中：
- $C_{total}$ = 总成本
- $P_{model_i}$ = 第 $i$ 次请求所使用模型的每 token 价格
- $T_{tokens_i}$ = 第 $i$ 次请求消耗的 token 数
- $N$ = 请求总数

**问题核心**：大多数 Agent 应用对所有请求都使用同一个"最强大"的模型（如 Claude Opus 4.6、GPT-4o），但实际上：

> Opus 4.6 运行成本是 $5/$25（input/output per million tokens），而 Haiku 4.5 只需 $1/$5。对于简单的心跳检测、格式转换等任务，用 Opus 就像"请律师查信箱"。

ClawRouter 的核心思想：**不同复杂度的请求应该用不同级别的模型**。通过将每个请求分类到合适的 tier，自动选择性价比最高的模型，从而大幅降低 $P_{model_i}$ 的平均值。

---

### 三、核心架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client / OpenClaw Agent                   │
│                                                                  │
│  HTTP Request (OpenAI-compatible format)                         │
│  POST /v1/chat/completions                                       │
│  { model: "blockrun/auto", messages: [...] }                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ClawRouter (localhost)                         │
│                                                                  │
│  ┌─────────────┐   ┌──────────────────────┐   ┌──────────────┐ │
│  │  Input      │──▶│  14-Dimension Weighted│──▶│  Tier Mapper │ │
│  │  Parser     │   │  Scorer (<1ms)        │   │  (4 tiers)   │ │
│  └─────────────┘   └──────────────────────┘   └──────┬───────┘ │
│                                                       │         │
│  ┌────────────────────────────────────────────────────┘         │
│  │  Tier → Model Selection (configurable)                       │
│  │  budget    → flash/haiku models                              │
│  │  standard  → sonnet-level models                             │
│  │  premium   → opus-level models                               │
│  │  ultra     → most powerful models                            │
│  └──────────────────────────────────────────────────────────────┘
│                           │                                      │
│  ┌────────────────────────┘                                      │
│  │  x402 Payment Layer                                           │
│  │  USDC on Base / Solana                                        │
│  │  (HTTP 402 Payment Required → crypto micropayment)            │
│  └───────────────────────────────────────────────────────────────┘
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LLM Provider APIs                              │
│   Anthropic · OpenAI · Google Gemini · 41+ models                │
└──────────────────────────────────────────────────────────────────┘
```

**关键设计原则**：
1. **100% 本地路由** — 模型选择不需要任何外部 API 调用，无 LLM inference
2. **Client-side only** — 你的钱包私钥永远不会离开本机
3. **Non-destructive** — 不修改 prompt 内容，只做路由决策
4. **<1ms 决策延迟** — 整个分类流程（14 维 + tier 映射 + 模型选择）在 1 毫秒内完成

---

### 四、14 维加权评分系统（核心技术）

这是 ClawRouter 的灵魂。核心代码在 `src/router/rules.ts`，注释写道：

> *Rule-Based Classifier (v2 — Weighted Scoring)*
> *Scores a request across 14 weighted dimensions and maps the aggregate score to a tier using configurable thresholds.*

#### 4.1 评分公式

$$S_{aggregate} = \sum_{d=1}^{14} w_d \cdot s_d(x)$$

其中：
- $S_{aggregate}$ = 聚合分数，决定请求的 tier
- $w_d$ = 第 $d$ 维的权重（可在 `src/router/config.ts` 中配置）
- $s_d(x)$ = 第 $d$ 维对请求 $x$ 的评分函数（0 或 1，或连续值）
- 14 个维度各自独立评估请求的某个特征

#### 4.2 已知的维度列表

根据多个来源（Reddit、Medium 博文、Facebook 帖子）整理出的 14 维度包括：

| # | 维度 | 检测内容 | 权重倾向 |
|---|------|---------|---------|
| 1 | **Token Count** | prompt + system prompt 的总 token 数 | 高 token → 高 tier |
| 2 | **Reasoning Markers** | 是否包含 "think step by step"、"analyze"、"reason about" 等推理触发词 | 有 → 倾向 premium |
| 3 | **Code Presence** | 是否包含代码块（```标记、函数定义等） | 有 → 倾向 standard+ |
| 4 | **Math Presence** | 是否包含数学公式、计算请求 | 有 → 倾向 premium |
| 5 | **Complexity Score** | 基于 prompt 结构的复杂度评估（嵌套层级、条件数量等） | 高 → 高 tier |
| 6-14 | **其他维度** | 可能包括：多语言检测、长上下文需求、工具调用检测、creative writing 标记、safety/合规标记、agentic loop 检测、structured output 需求、对话轮次等 | 各有倾向 |

> 注意：完整的 14 维列表在 `docs/smart-llm-router-14-dimension-classifier.md` 中有详细定义，但该文件需要直接访问 repo 查看。

#### 4.3 Tier 映射

聚合分数 $S_{aggregate}$ 通过可配置的阈值映射到 4 个 tier：

```
S < θ₁  →  budget     →  Claude Haiku 4.5 / GPT-4o-mini / Gemini Flash
θ₁ ≤ S < θ₂  →  standard  →  Claude Sonnet 4.6 / GPT-4o / Gemini Pro
θ₂ ≤ S < θ₃  →  premium   →  Claude Opus 4.6 / GPT-4.5 / Gemini Ultra
S ≥ θ₃  →  ultra      →  最强大的可用模型
```

其中 $\theta_1, \theta_2, \theta_3$ 是可配置的阈值参数。

---

### 五、x402 支付协议（Crypto 微支付层）

ClawRouter 集成了 **x402 协议**，这是一个基于 HTTP 402 (Payment Required) 状态码的开放支付标准，由 Coinbase 推动。

#### 5.1 工作流程

```
Agent → POST /v1/chat/completions
        ↓
Provider 返回 HTTP 402 + 支付信息（链上地址、金额）
        ↓
ClawRouter 自动用 USDC 在 Base/Solana 上发起微支付
        ↓
带着支付证明重新发送请求
        ↓
Provider 验证链上支付 → 返回 LLM 响应
```

#### 5.2 x402 核心特性

| 特性 | 说明 |
|------|------|
| **无账户** | 不需要信用卡、API key 账户 |
| **即时结算** | USDC on-chain 结算，秒级确认 |
| **微支付** | 每次请求可能只需 $0.001 级别的支付 |
| **跨链** | 支持 Base（L2）和 Solana 两条链 |
| **AI Agent 友好** | Agent 可以自主持有钱包、自主支付 |

参考: [x402 Protocol Explained](https://stablecoininsider.org/x402-protocol/), [Sherlock x402 分析](https://sherlock.xyz/post/x402-explained-the-http-402-payment-protocol)

---

### 六、与竞品对比

| 特性 | ClawRouter | OpenRouter | LiteLLM |
|------|-----------|------------|---------|
| **类型** | Agent-native 本地路由器 | SaaS 网关 | 开源代理/SDK |
| **路由决策** | 14 维加权评分，本地 <1ms | 基于用户选择或 fallback | 基于配置的 fallback |
| **智能路由** | ✅ 自动分析 prompt 复杂度 | ❌ 需手动指定模型 | ❌ 需手动配置 |
| **支付方式** | x402 USDC（crypto） | 信用卡/预充值 | 自带 API key |
| **Agent 优化** | ✅ 专为 agent 行为模式设计 | ❌ 面向人类应用 | ❌ 通用代理 |
| **数据隐私** | 100% 本地，私钥不出设备 | 请求经过第三方服务器 | 取决于配置 |
| **Failover** | Agent 感知的错误处理 | 429 错误处理不够友好 | 有限的重试逻辑 |

参考: [Why ClawRouter Is the Natural Choice for OpenClaw](https://thamizhelango.medium.com/why-clawrouter-is-the-natural-choice-for-openclaw-and-where-openrouter-and-litellm-fall-short-6edc0a77748d), [clawrouter-vs-openrouter comparison](https://github.com/BlockRunAI/ClawRouter/blob/main/docs/clawrouter-vs-openrouter-llm-routing-comparison.md)

---

### 七、成本节省数据

根据项目文档和社区报告：

| 场景 | 无路由（全用 Opus） | 有路由（ClawRouter） | 节省率 |
|------|-------------------|--------------------|--------|
| Anthropic 日常使用 | ~$4,660/月 | ~$1,400/月 | **~70%** |
| Agent 工作负载（高频简单请求） | ~$200/月 | ~$16/月 | **~92%** |
| 混合工作负载 | 取决于分布 | 自动优化 | **60-92%** |

核心原因：Agent 的工作负载中，大量请求是"心跳检测"、"状态查询"、"格式转换"等简单任务，这些完全可以用 Haiku/Flash 级别的模型处理，而不需要 Opus。

---

### 八、OpenClaw 是什么

OpenClaw 是一个开源的自主 AI Agent，可以：
- 在本地硬件上运行
- 连接到 LLM（Claude、GPT 等）
- 通过消息应用（Discord、Telegram 等）交互
- 执行真实任务（100+ 内置技能）
- 无需人工持续干预

ClawRouter 就是专门为 OpenClaw 设计的路由组件，解决 OpenClaw 在高频 agent 循环中"烧钱"的问题。

参考: [What Is OpenClaw?](https://www.mindstudio.ai/blog/what-is-openclaw-ai-agent/), [OpenClaw Explained](https://www.kdnuggets.com/openclaw-explained-the-free-ai-agent-tool-going-viral-already-in-2026)

---

### 九、如何使用

```bash
# 安装
npm install -g clawrouter

# 或 Docker
docker run -p 8402:8402 blockrun/clawrouter

# 配置（设置 API keys 和钱包）
clawrouter config set anthropic-key sk-ant-...
clawrouter config set wallet-key <your-private-key>

# 使用（将 OpenClaw 的 API endpoint 指向 localhost:8402）
# model 设为 "blockrun/auto" 即可启用智能路由
```

OpenClaw 配置中：
```json
{
  "apiEndpoint": "http://localhost:8402/v1",
  "model": "blockrun/auto"
}
```

当 `model = "blockrun/auto"` 时，ClawRouter 的路由引擎自动介入。

---

### 十、总结 & Intuition Builder

**一句话直觉**：ClawRouter 就像是 LLM 的"交通调度员"——它不是让你坐最贵的出租车去任何地方，而是根据你的目的地复杂度自动给你分配最合适的交通工具：去楼下买咖啡骑共享单车就行，跨国出差才坐头等舱。

**核心洞察**：
1. **Agent ≠ Human App**：Agent 的工作负载分布与人类应用完全不同，Agent 有大量低复杂度、高频率的请求
2. **Local > Cloud**：路由决策完全在本地做，延迟 <1ms，不泄露数据
3. **Crypto-native 支付**：x402 让 AI Agent 能自主持有钱包、自主支付，这是 Agent 经济的基础设施
4. **Rule-based > LLM-based routing**：用 LLM 来决定用哪个 LLM 是反模式（额外成本+延迟），用规则评分才是正道