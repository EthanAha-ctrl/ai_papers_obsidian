# Anthropic PR Blog 深度解析：Effective Context Engineering for AI Agents

**发布日期：2025年9月29日 | 来源：Anthropic Applied AI Team**

---

## 一、核心论点：从 Prompt Engineering 到 Context Engineering 的范式转移

这篇 blog 的核心主张是：**随着 AI agent 从单轮推理走向多轮自主循环，"写好 prompt" 已经不够了，我们需要 "engineer 好 context"。**

### Prompt Engineering vs. Context Engineering 的本质区别

| 维度 | Prompt Engineering | Context Engineering |
|------|-------------------|---------------------|
| **关注焦点** | 如何写好 system prompt | 如何在每一步推理中 curate 整体 context state |
| **时间维度** | 一次性（one-shot） | 迭代式（iterative），每轮 inference 都需要重新 curate |
| **信息范围** | 主要关注 prompt 本身 | system instructions + tools + MCP + external data + message history + ... |
| **适用场景** | 单次 classification/generation | 多轮、长时 horizon 的 agentic workflow |

**直觉理解**：Prompt engineering 像是写一份好的菜谱（静态的），Context engineering 像是在烹饪过程中不断调整火候、食材比例和顺序（动态的）。

---

## 二、为什么 Context Engineering 如此重要？——从 Transformer 架构的第一性原理出发

### 2.1 Context Rot 与 Attention Scarcity

Blog 引用了 **needle-in-a-haystack** benchmarking 的发现，提出了 **context rot** 的概念：

$$\text{Retrieval Accuracy}(n) \propto f(n), \quad \text{where } f'(n) < 0$$

即：随着 context window 中 token 数 $n$ 的增加，模型从 context 中准确检索信息的能力递减。

### 2.2 从 Transformer 架构理解这个现象

Transformer 的核心是 **self-attention mechanism**。对于 $n$ 个 token 的序列，self-attention 计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$：Query matrix，每个 token 的"提问"
- $K \in \mathbb{R}^{n \times d_k}$：Key matrix，每个 token 的"索引"
- $V \in \mathbb{R}^{n \times d_v}$：Value matrix，每个 token 的"内容"
- $d_k$：key/query 的维度（head dimension）

**关键洞察**：$QK^T$ 产生一个 $n \times n$ 的 attention matrix，包含 **$n^2$ 个 pairwise relationship**。这意味着：

1. **计算复杂度 $O(n^2)$**：token 数翻倍，attention 计算量变为 4 倍
2. **Attention 分布稀释**：softmax 归一化意味着 $\sum_{j=1}^{n} \alpha_{ij} = 1$，当 $n$ 增大时，每个 token 获得的 "attention budget" $\alpha_{ij}$ 被分配到更多候选 token 上，单 token 获得的 attention 权重变薄
3. **训练数据分布偏差**：LLM 的 attention pattern 是从训练数据中学到的，而训练数据中短序列的分布密度远高于长序列，导致模型在长 context 上的 attention 模式不够精准

### 2.3 Position Encoding Interpolation 的影响

Blog 提到了 **position encoding interpolation** 技术允许模型处理比训练时更长的序列：

原始 RoPE (Rotary Position Embedding) 定义位置 $m$ 的编码为：

$$f(\mathbf{q}, m) = \mathbf{q} \cdot e^{im\theta}$$

当需要将 context 从训练长度 $L_{\text{train}}$ 扩展到 $L_{\text{target}}$ 时，interpolation 将位置索引缩放：

$$m' = m \cdot \frac{L_{\text{train}}}{L_{\text{target}}}$$

这意味着原本在整数位置 $m$ 的 token 被映射到分数位置 $m'$，模型从未见过这样的位置编码，因此虽然可以工作，但 **position understanding 会 degradation**。这也是 context rot 的一个结构性原因。

### 2.4 性能是 Gradient 而非 Cliff

Blog 强调这是一个 **性能梯度** 而非 **硬性断崖**：

$$\text{Performance}(n) = P_{\text{max}} - \lambda \cdot g(n)$$

其中 $g(n)$ 是一个关于 $n$ 单调递增的函数，$\lambda$ 是 degradation 系数（不同模型不同）。模型在长 context 上仍然 highly capable，但 precision 降低。

**直觉**：就像人的工作记忆，给你一本 1000 页的书，你能理解每一页，但要你精确回忆第 487 页第三段的某个细节——难度显著上升。

---

## 三、Effective Context 的解剖——核心原则

Blog 给出了指导性公式：

$$\text{Good Context Engineering} = \arg\min_{\text{tokens}} |\text{tokens}| \quad \text{s.t.} \quad P(\text{desired outcome} | \text{tokens}) \to \max$$

**即：找到最小的高信号 token 集，最大化期望结果的概率。**

### 3.1 System Prompt 的 "Goldilocks Zone"

Blog 描述了两个极端的 failure mode：

```
过于 brittle (左极端)          ←→    Optimal Zone    ←→    过于 vague (右极端)
├── if-else 硬编码逻辑                 ├── 具体到能引导行为         ├── 模糊的高层指导
├── 脆弱、难以维护                      ├── 灵活到提供强 heuristic    ├── 假设了共享 context
└── 试图用 prompt 控制一切              └── 像 "好管理者的指令"        └── LLM 无法获得具体信号
```

**结构建议**：
- 使用 XML tagging 或 Markdown headers 组织 prompt section：`<background_information>`, `<instructions>`, `## Tool guidance`, `## Output description`
- 从 **minimal prompt** 开始，用最好的模型测试，然后基于 failure mode 逐步添加指令和 examples

**注意**：minimal ≠ short。你需要给 agent 足够的 upfront information。

### 3.2 Tool Design 的效率原则

Blog 引用了他们的另一篇文章 [Writing tools for AI agents – with AI agents](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)。

核心原则：
1. **Self-contained**：每个 tool 功能独立
2. **Minimal overlap**：工具之间功能重叠最小化
3. **Token-efficient return**：工具返回值要精简
4. **Clear intent**：工具名和参数描述要极度清晰

**最常见的 failure mode：bloated tool set**。如果人类工程师都无法确定在某个场景下该用哪个 tool，AI agent 更不可能做对。

### 3.3 Few-shot Examples 的正确用法

**错误做法**：塞入 laundry list of edge cases，试图用规则覆盖所有情况。
**正确做法**：curate 一组 **diverse, canonical examples**，有效展示期望行为。

$$\text{Value of Examples} = \text{Diversity} \times \text{Representativeness} \neq \text{Quantity of Edge Cases}$$

Blog 的金句：**"For an LLM, examples are the 'pictures' worth a thousand words."**

---

## 四、Context Retrieval 与 Agentic Search——最核心的架构创新部分

### 4.1 从 Pre-inference Retrieval 到 Just-in-Time Retrieval

Blog 描述了一个重要的范式转移：

| 策略 | Pre-inference Retrieval | Just-in-Time Retrieval |
|------|------------------------|----------------------|
| **时机** | 推理前一次性获取 | 推理时按需动态获取 |
| **实现** | Embedding + Vector Search (RAG) | Agent 自主使用 tool 查询 |
| **Context 占用** | 所有 retrieved data 都在 context 中 | 仅保持轻量 identifier (file path, query, URL) |
| **类比** | 考前背完整本教材 | 带着参考书目进考场，需要时翻阅 |

**Claude Code 的实际做法**：
- 不把整个 database 加载到 context
- Agent 写 targeted SQL query → 存储结果 → 用 `head`, `tail` 等 Bash 命令分析
- **从不加载完整 data object 到 context**

### 4.2 Metadata as Implicit Signal

这是一个非常精妙的洞察：**轻量 identifier 的 metadata 本身就是高信号**。

```
tests/test_utils.py       →  暗示这是测试工具
src/core_logic/test_utils.py  →  暗示这是核心逻辑中的测试相关代码
```

文件路径、命名规范、时间戳——这些都是 **无需加载内容就能获得的 context signal**，对 agent 理解信息 landscape 极其有用。

### 4.3 Progressive Disclosure

这个概念来自 UI/UX 设计，被应用到 agent 的 context 策略中：

$$\text{Context Understanding}_t = f(\text{Context Understanding}_{t-1}, \text{New Discovery}_t)$$

Agent 逐层发现相关信息：
1. **File size** → 暗示复杂度
2. **Naming convention** → 暗示 purpose
3. **Timestamp** → 暗示 relevance（更新 = 可能更相关）
4. **Content** → 确认或修正假设

### 4.4 Hybrid Strategy（混合策略）

Blog 推荐 **hybrid model**：

```
┌─────────────────────────────────────────────────┐
│              Hybrid Context Strategy              │
├─────────────────────────────────────────────────┤
│                                                   │
│  Up-front (Pre-inference):                       │
│  ├── CLAUDE.md files (static project context)    │
│  └── Essential background info                    │
│                                                   │
│  Just-in-Time (Runtime):                          │
│  ├── glob tool → discover files                   │
│  ├── grep tool → search content                   │
│  └── Agent 自主探索环境                            │
│                                                   │
└─────────────────────────────────────────────────┘
```

**决策边界取决于任务特性**：
- **Less dynamic content**（法律、金融）→ 更多 up-front retrieval
- **Highly dynamic content**（coding、research）→ 更多 just-in-time exploration

Blog 的务实建议：**"Do the simplest thing that works."**

---

## 五、Long-Horizon Tasks 的三大核心技术

这是这篇 blog 最具工程价值的部分。当任务跨越数分钟到数小时（如大型 codebase migration 或 comprehensive research），token 数会超出 context window，需要专门技术。

### 5.1 Compaction（压缩）

**定义**：当 conversation 接近 context window 上限时，将内容 summarizing，然后用 summary 重新初始化 context window。

**Claude Code 的实现**：
1. 将 message history 传给模型进行 summary
2. 保留：architectural decisions、unresolved bugs、implementation details
3. 丢弃：冗余 tool outputs、重复 messages
4. 继续时：compressed context + 5 个最近访问的文件

**Compaction 的 art**：选择保留什么 vs 丢弃什么。过度 aggressive 的 compaction 会丢失看似不重要但后续至关重要的 context。

**调优方法**：
1. 先最大化 **recall**（确保 capture 所有相关信息）
2. 再迭代优化 **precision**（消除冗余内容）

**最简单有效的 compaction**：**Tool result clearing**——一旦 tool 被调用且结果已在 deep message history 中，raw result 对后续推理没有价值，可以安全清除。这已经在 [Claude Developer Platform](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) 上作为 feature 发布。

**数学直觉**：

$$\text{Compaction Ratio} = \frac{|\text{Summary Tokens}|}{|\text{Original Tokens}|}$$

目标是最小化这个 ratio 同时最大化：

$$\text{Information Fidelity} = \frac{I(\text{Summary})}{I(\text{Original})}$$

其中 $I(\cdot)$ 表示对后续推理任务的信息效用（information utility）。

### 5.2 Structured Note-Taking / Agentic Memory

**定义**：Agent 定期将 notes 写到 context window 外的持久化存储中，后续需要时再加载回 context。

**Claude Code 实例**：创建 to-do list、维护 NOTES.md 文件。

**Claude Playing Pokémon 的精彩案例**：

Agent 在数千步游戏过程中维护精确记录：
- "过去 1,234 步我一直在 Route 1 训练 Pokémon，Pikachu 已经升了 8 级，目标是 10 级"
- 绘制已探索区域的地图
- 记录已解锁的关键成就
- 维护战斗策略笔记（哪种攻击对哪种对手最有效）

**Context reset 后**，agent 读取自己的 notes 继续数小时的训练序列或地牢探索。

**数学模型**：

$$\text{Effective Context}_t = \text{Working Memory}_t \cup \text{Notes Loaded}_t$$

$$|\text{Working Memory}_t| \ll |\text{Total Accumulated Info}_t|$$

Agent 在 working memory 中只保持当前需要的最小信息集，其余持久化到 notes。

Anthropic 在 Sonnet 4.5 launch 时发布了 [memory tool public beta](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)，通过 file-based system 让 agent 在 context window 外存储和查阅信息。

### 5.3 Sub-Agent Architectures（子代理架构）

**定义**：主 agent 协调高层计划，专业化 sub-agent 处理 focused tasks，每个 sub-agent 有独立的 clean context window。

**核心机制**：
```
Main Agent (Coordinator)
│   Context: High-level plan, synthesis
│   Token budget: ~5,000-10,000
│
├── Sub-Agent A (Deep Search)
│   Context: Full search results, tool outputs
│   Token budget: ~50,000+
│   Returns: ~1,000-2,000 tokens summary
│
├── Sub-Agent B (Code Analysis)
│   Context: Full codebase analysis
│   Token budget: ~50,000+
│   Returns: ~1,000-2,000 tokens summary
│
└── Sub-Agent C (Data Processing)
    Context: Full dataset operations
    Token budget: ~50,000+
    Returns: ~1,000-2,000 tokens summary
```

**Separation of Concerns**：
- 详细搜索 context 被隔离在 sub-agent 内部
- 主 agent 只看到 distilled summary
- Sub-agent 的 context 污染不会影响主 agent

Blog 引用了 [How we built our multi-agent research system](https://www.anthropic.com/engineering/claiude-multi-agent-research-system)，显示在复杂 research 任务上 substantial improvement。

**Compression Rate**：
$$\text{Sub-Agent Compression} = \frac{\text{Summary Tokens}}{\text{Sub-Agent Total Tokens}} \approx \frac{1{,}000\text{-}2{,}000}{50{,}000+} \approx 2\%\text{-}4\%$$

这是极其高效的 context 压缩比。

### 5.4 三种技术的适用场景对比

| 技术 | 最适场景 | 核心优势 | 核心风险 |
|------|---------|---------|---------|
| **Compaction** | 需要大量 back-and-forth 的对话式任务 | 维持 conversational flow | 过度压缩丢失关键 context |
| **Structured Note-Taking** | 有清晰里程碑的迭代开发 | 跨 session 持久化 | Notes 结构设计需要工程判断 |
| **Multi-Agent** | 复杂 research 和 parallel exploration | Clean context isolation + 并行 | Coordination overhead + sub-agent 间的信息 gap |

---

## 六、整体框架总结

Blog 的整体思路可以用一个统一的框架来理解：

```
┌──────────────────────────────────────────────────────────────┐
│                 CONTEXT ENGINEERING FRAMEWORK                 │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  1. CONTEXT IS FINITE                                          │
│     ├── Attention budget: ∑α_ij = 1 per token                │
│     ├── Context rot: accuracy ↓ as n ↑                       │
│     └── O(n²) pairwise relationships → diminishing returns    │
│                                                                │
│  2. MINIMIZE HIGH-SIGNAL TOKENS                                │
│     ├── System prompt: Goldilocks zone                        │
│     ├── Tools: minimal viable set, token-efficient returns    │
│     └── Examples: diverse + canonical > edge case laundry list│
│                                                                │
│  3. DYNAMIC RETRIEVAL                                          │
│     ├── Pre-inference (RAG) ↔ Just-in-time (agentic search)   │
│     ├── Metadata as signal                                     │
│     ├── Progressive disclosure                                 │
│     └── Hybrid: up-front static + runtime dynamic              │
│                                                                │
│  4. LONG-HORIZON TECHNIQUES                                    │
│     ├── Compaction: summarize + restart context               │
│     ├── Structured note-taking: external persistent memory    │
│     └── Sub-agents: isolated context + distilled summaries    │
│                                                                │
│  5. GUIDING PRINCIPLE                                          │
│     "Find the smallest set of high-signal tokens               │
│      that maximize P(desired outcome)"                        │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 七、我的评论与延伸思考

### 7.1 这篇 blog 的定位

这是一篇 **Anthropic 的平台营销 + 工程方法论文章**，目的是：
1. 推广 "Context Engineering" 作为新范式（建立认知框架）
2. 推销 Claude Developer Platform 的 feature（memory tool, tool result clearing, MCP）
3. 为 Claude Code 提供设计哲学的背书
4. 对抗 "just use longer context windows" 的简单化思路

### 7.2 未被讨论的重要维度

1. **Cost optimization**：context engineering 直接影响 API pricing（token 计费），但 blog 没有量化讨论 cost-performance tradeoff
2. **Latency**：just-in-time retrieval 增加 inference latency（多轮 tool call），何时 up-front 更好应该有 latency budget 的考量
3. **Compaction 的信息损失量化**：blog 说 "minimal performance degradation" 但没有给出 benchmark data
4. **Sub-agent coordination 的 failure mode**：当 sub-agent 返回的 summary 遗漏了主 agent 需要的关键信息时怎么办？
5. **Security & privacy**：sub-agent 架构中，不同 sub-agent 可能有不同的 data access 权限需求，blog 完全未提及

### 7.3 与学术文献的关联

- **Context rot** 对应学术界研究的 "Lost in the Middle" 现象（[Liu et al., 2023](https://arxiv.org/abs/2307.03172)）
- **Compaction** 对应 "context window management" / "conversation compression"（[Wu et al., 2023](https://arxiv.org/abs/2305.14726)）
- **Sub-agent architecture** 对应 "LLM-based multi-agent systems"（[Park et al., 2023 - Generative Agents](https://arxiv.org/abs/2304.03442)）
- **Progressive disclosure** 借鉴自 HCI 领域的经典设计原则
- **Attention scarcity** 的 $O(n^2)$ 分析是 Transformer 的基本性质，可追溯至 [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

### 7.4 关键 Takeaway

> **Context engineering is to prompt engineering what software engineering is to writing a script.**

Prompt engineering 是写一个 script——一次性、静态、有限 scope。Context engineering 是构建一个 software system——需要考虑 state management、resource allocation、modular design、error recovery、和 long-running process 的 coherence。

这篇 blog 最重要的贡献是给出了一个 **可操作的 mental model**：把 context 当作有限资源来 engineer，而不是试图用更长 context window 或更 clever prompt 来绕过限制。

---

**相关资源链接**：
- 原文：[Effective context engineering for AI agents](https://www.anthropic.com/engineering/context-engineering-for-ai-agents)
- Anthropic Prompt Engineering 文档：[docs.anthropic.com/en/docs/build-with-claude/prompt-engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
- Claude Developer Platform Memory Tool：[docs.anthropic.com/en/docs/build-with-claude/extended-thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- Memory and Context Management Cookbook：[docs.anthropic.com/en/docs/build-with-claude/memory-and-context](https://docs.anthropic.com/en/docs/build-with-claude/memory-and-context-management-cookbook)
- Multi-Agent Research System：[anthropic.com/engineering/claiude-multi-agent-research-system](https://www.anthropic.com/engineering/claiude-multi-agent-research-system)
- Lost in the Middle 论文：[arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)
- Attention Is All You Need：[arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)