# 用人话说：这篇文章在唠叨啥？

---

## 先从一个真实场景说起

想象你在用 Word 写文档：

**场景 A**：你选中一段文字，按"加粗"按钮 → Word 自动帮你加上格式标记
**场景 B**：你直接手敲 `<b>标题</b>` → Word 显示成 `<b>标题</b>`，不会自动变粗

这就是两种编辑思维的区别。

---

## 文章讨论的核心问题

程序员写代码时，也面临两种选择：

### 选择 1：Structured Editing（结构化编辑）

> 编辑器"懂"你写的代码，知道哪里是函数、哪里是变量。

**好比**：你用乐高积木搭房子，只能按卡扣的位置拼接，不会搭错。

**优点**：
- 编辑器能告诉你"下一个该写变量名了"
- 选中一个函数复制，100% 准确，不会多选或少选
- 语法错误？编辑器立刻标红

**缺点**：
- 你想"违规操作"？比如一次性在 5 行代码前面加空格 → 编辑器说：不行！
- 很多人觉得被束缚住了手脚

### 选择 2：普通文本编辑

> 编辑器把代码当成普通文字，你想怎么改就怎么改。

**好比**：你用橡皮泥捏东西，想怎么捏怎么捏。

**优点**：
- 自由！想怎么改怎么改
- 块编辑、多光标、正则替换... 随便用

**缺点**：
- 编辑器"不懂"你的代码
- 选一个函数？靠猜，经常选错
- 语法错误？等你运行了才知道

---

## 问题来了

**两种方式各有利弊，能不能兼得？**

作者说：**能！答案就是 Incremental Parsing（增量解析）。**

---

## 什么是 Incremental Parsing？

用人话说：

> 你尽情地改代码，当成普通文本改。
> 
> 后台有个"小助手"默默帮你分析代码结构。
> 
> 哪怕你改到一半代码语法错了，小助手也不慌，等着你改完再更新。

**打个比方**：

你开车（编辑代码），导航仪（解析器）实时更新路线。

- 传统解析：你拐个弯，导航仪要重新算一遍全程路线 → 慢！
- 增量解析：你拐个弯，导航仪只更新受影响的那段路线 → 快！

---

## 文章作者的 frustration（吐槽）

作者说，他已经吐槽过很多次了：

> "又有人搞 structured editor，又不支持自由编辑，我都懒得说了。但问题是——**明明有解决方案啊！**"

这个解决方案就是 Tim Wagner 的博士论文，里面有完整的算法。

而且，**Tree-sitter**（现在很多编辑器都在用）就是这套理论的实践证明。

---

## 一句话总结

> **Structured editing 很好，但不该限制用户的编辑自由。Incremental parsing 是"两全其美"的技术方案——让你既能自由编辑，又能享受编辑器"懂你代码"的好处。**

作者的呼吁：做工具的人，别闭门造车，去看看 Tim Wagner 和 Lukas Diekmann 的论文，站在巨人肩膀上做事。

---

## 为什么这事儿重要？

你用的这些工具，背后都涉及这个问题：

| 工具 | 用到了什么 |
|------|-----------|
| VS Code 语法高亮 | Tree-sitter (incremental parsing) |
| JetBrains IDE | 部分结构化编辑 |
| GitHub Copilot | 需要理解代码结构 |
| 未来的 AI 编辑器 | 更需要这个问题解决好 |

---

# 文章概述：Incremental Parsing 与 Structured Editing 的博弈

这篇文章由 parsing 领域专家 **Laurence Tratt** 撰写，核心探讨了一个困扰编程工具开发多年的问题：**如何在保持结构化编辑优势的同时，允许自由形式的文本编辑？**

---

## 一、核心矛盾

### Structured Editing（结构化编辑）的理想与现实

| 维度 | 理想状态 | 现实困境 |
|------|----------|----------|
| **核心理念** | 编辑器完全理解语言的 syntactic structure | 程序员习惯"违反"语法结构的编辑方式 |
| **优势** | 即时语法反馈、精确语义提示、100% 准确的代码选择 | "Square block editing"（块编辑）等操作被禁止 |
| **历史** | 1960s Lisp 编辑器起步，现代代表 JetBrains MPS | 始终未能"take over the world" |

### 矛盾根源

> "Most of us... sometimes edit programs in **deliberate violation** of their syntactic structure."

作者举例：在行 7-10 的第 3-5 列进行块编辑，可能瞬间将一个语法正确的程序变成多个 bizarre 的语法错误状态。传统 structured editors **直接禁止**这类操作。

---

## 二、解决方案：Incremental Parsing

### 核心思想

$$\text{Incremental Parsing} = \underbrace{\text{自由编辑}}_{\text{UTF-8 sequence}} + \underbrace{\text{后台维护 parse tree}}_{\text{允许"broken"状态}}$$

**关键突破**：parse tree 可以在用户输入语法不正确时变得 **arbitrarily "broken"**，而不约束用户的编辑行为。

---

## 三、Tim Wagner 的三算法架构

Wagner 的博士论文提供了 **incremental parsing** 的系统性解决方案，包含三个核心算法：

### Algorithm 1: Incremental Lexing

**传统 lexing 问题**：
- 修改一个字符可能改变所有后续 token 的边界
- 时间复杂度：$O(n)$，其中 $n$ 为文件长度

**增量词法分析核心**：
$$\Delta_{\text{lex}} = f(\text{edit\_position}, \text{edit\_length}, \text{old\_tokens})$$

关键洞察：
- Token 边界的变化是 **locally bounded** 的
- 只有当 token 边界跨越编辑位置时才需要重新词法分析
- 使用 **delta encoding** 记录变化

**变量说明**：
- $\Delta_{\text{lex}}$：词法分析的增量变化
- `edit_position`：编辑发生的位置
- `edit_length`：编辑的长度（正数表示插入，负数表示删除）

---

### Algorithm 2: Incremental LR Parsing

**LR Parsing 基础回顾**：
$$\text{LR}(k): S \rightarrow \alpha \quad \text{where } \alpha \in (V \cup T)^*$$

传统 LR parsing 使用：
- **ACTION 表**：$GOTO[\text{state}, \text{symbol}]$
- **GOTO 表**：$ACTION[\text{state}, \text{terminal}]$

**增量式更新策略**：

```
Parse Tree Nodes:
├── maintain reference to: 
│   ├── production rule used
│   ├── children nodes
│   └── span in source code [start, end]
└── on edit:
    ├── identify affected nodes (those whose span overlaps edit)
    ├── reuse unaffected subtrees
    └── re-parse only the "dirty" region
```

**核心技术**：
1. **Node Reuse**：未受影响的子树直接复用
2. **State Recovery**：从最近的 valid state 恢复解析
3. **Error Tolerance**：允许 parse tree 包含 "error nodes"

---

### Algorithm 3: Incremental GLR Parsing

**为什么需要 GLR？**
- LR 只能处理 $LR(k)$ 文法
- GLR（Generalized LR）可处理所有 **Context-Free Grammars (CFG)**

**CFG 定义**：
$$G = (V, T, P, S)$$
其中：
- $V$ = 非终结符集合
- $T$ = 终结符集合  
- $P$ = 产生式规则集合，形如 $A \rightarrow \alpha$
- $S$ = 起始符号

**GLR 增量解析**：

GLR 使用 **Graph-Structured Stack (GSS)** 处理歧义：

```
传统 LR stack:    [s0, s1, s2, s3, ...]  (线性)
GLR GSS:          多条路径并行探索，形成 DAG 结构
```

**增量 GSS 更新**：
$$\text{GSS}_{\text{new}} = \text{GSS}_{\text{old}} - \text{affected\_paths} + \text{recomputed\_paths}$$

**Wagner 的关键贡献**：
- 在 GSS 中标记 "pinned" 和 "unpinned" 节点
- 只有 unpinned 节点可能被更新
- 通过 **memoization** 缓存中间结果

---

## 四、实际应用：Tree-sitter

作者特别指出：

> "Max Brunsfeld, the dynamo behind **Tree-sitter**, has been clear about the influence of Tim's work on Tree-sitter."

### Tree-sitter 架构图解

```
┌─────────────────────────────────────────────────────┐
│                    Source Code                       │
│                 (UTF-8 sequence)                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Incremental Lexer                       │
│  ┌─────────────────────────────────────────┐        │
│  │  Token Stream (with caching)             │        │
│  └─────────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Incremental Parser                      │
│  ┌─────────────────────────────────────────┐        │
│  │  Concrete Syntax Tree (CST)              │        │
│  │  - preserves all syntax details          │        │
│  │  - allows "error nodes"                  │        │
│  └─────────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Query System                            │
│  - Pattern matching on CST                          │
│  - Syntax highlighting, code folding, etc.          │
└─────────────────────────────────────────────────────┘
```

### Tree-sitter 的 CST 特性

与传统 AST 不同，Tree-sitter 使用 **Concrete Syntax Tree**：

| 特性 | AST | CST (Tree-sitter) |
|------|-----|-------------------|
| 注释 | 通常丢弃 | 保留为节点 |
| 括号 | 通常丢弃 | 保留为节点 |
| 错误处理 | 通常失败 | 生成 "ERROR" 节点，继续解析 |
| 增量更新 | 需要完整重解析 | 只更新受影响部分 |

---

## 五、后续发展：Polyglot Editing

文章提到 **Lukas Diekmann** 扩展了 Wagner 的工作：

### Composed/Polyglot Programs

现代编程常涉及多语言混合：
- HTML + CSS + JavaScript
- SQL embedded in Python
- JSX (JavaScript + XML-like syntax)

**挑战**：
$$\text{Parse}_{\text{polyglot}} = \bigcup_{i=1}^{n} \text{Parse}_{\text{language}_i}(\text{region}_i)$$

不同语言的语法规则可能冲突，需要：
1. **Language Boundary Detection**：识别语言切换点
2. **Nested Incremental Parsing**：内嵌语言的独立增量解析
3. **Cross-Language References**：跨语言的语义引用

---

## 六、第一性原理分析

从第一性原理出发，incremental parsing 解决的根本问题是：

### 问题本质

$$\underbrace{\text{Human Intent}}_{\text{编辑意图}} \xrightarrow{\text{gap}} \underbrace{\text{Syntactic Structure}}_{\text{语法结构}}$$

传统 structured editing 强制：
$$\text{Intent} \subseteq \text{Syntactic Structure}$$

但这违反了人类编辑的真实模式。

### Incremental Parsing 的解决方案

它将问题解耦为两个独立通道：

$$
\begin{cases}
\text{Channel 1: Editing} & \rightarrow \text{自由操作字符序列} \\
\text{Channel 2: Analysis} & \rightarrow \text{best-effort 维护结构信息}
\end{cases}
$$

**"Best-effort" 是关键**：
- 允许结构信息暂时 incomplete 或 incorrect
- 在用户停止编辑时逐渐收敛到 correct state

---

## 七、文章的核心呼吁

作者表达了明确的 **frustration**：

> "I thus must admit to some frustration when I'm pointed at yet another attempt to make structured editing work, that either doesn't take this problem into account, or does so in an ad hoc manner."

**呼吁内容**：
1. 承认 incremental parsing 作为 published solution 的存在
2. 阅读 Tim Wagner 和 Lukas Diekmann 的论文
3. 在此基础上构建工具，而非"reinvent the wheel"

---

## 八、相关资源链接

| 资源 | 链接 |
|------|------|
| Tim Wagner's Thesis | [Incremental Parsing, University of Washington, 1998](https://dl.acm.org/doi/book/10.5555/931235) |
| Tree-sitter | [https://tree-sitter.github.io/tree-sitter/](https://tree-sitter.github.io/tree-sitter/) |
| Laurence Tratt's Blog | [https://tratt.net/laurie/blog/](https://tratt.net/laurie/blog/) |
| Lukas Diekmann's Work | [University of Kent, Software Development Team](https://www.cs.kent.ac.uk/people/staff/lb514/) |
| JetBrains MPS | [https://www.jetbrains.com/mps/](https://www.jetbrains.com/mps/) |

---

## 九、总结

这篇文章本质上是一篇 **"technical advocacy"**：

1. **问题**：Structured editing 与自由编辑的矛盾
2. **方案**：Incremental parsing（以 Tim Wagner 论文为理论基础）
3. **验证**：Tree-sitter 的成功应用
4. **呼吁**：不要忽视已有成果，站在巨人肩膀上前进

作者的态度既学术又务实——他不是否定 structured editing，而是认为 incremental parsing 提供了 **两全其美** 的路径：既享受结构化编辑的好处，又不牺牲编辑自由度。