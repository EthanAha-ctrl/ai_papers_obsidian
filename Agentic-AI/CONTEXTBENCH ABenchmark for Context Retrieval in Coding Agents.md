---
source_pdf: CONTEXTBENCH ABenchmark for Context Retrieval in Coding Agents.pdf
paper_sha256: 517140a2504959bfb94c7c875f6e80a031e899bc362dcc73c446536b45e18a46
processed_at: '2026-08-03T17:06:54-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我们把这篇 paper 揉碎了用最直白的话聊一聊。你做 LLM 这么久，肯定遇到过这种憋屈情况：看 SWE-bench leaderboard，各家 agent 的 Pass@1 分数涨得飞快，但你真去 debug 它的 trajectory，发现它纯属瞎猫碰死耗子，随便 grep 一下改了两行代码刚好通过了测试。整个 pipeline 是个黑盒。CONTEXTBENCH 这篇论文就是来砸这个黑盒的。

### 1. 他们到底干了什么事

他们花了 4 个月，找专家给 1,136 个真实的 GitHub issue 人工标注了所谓的 "Gold Context"。简单来说就是：为了修好这个 bug，人类专家觉得你**必须**看懂哪几段代码。他们总共标了 522,115 行代码。

有了这个 ground truth，我们就能直接量化 agent 中间到底看了什么、看对了多少。他们搞了一套全自动的 evaluation framework，把 agent 跑过的轨迹全记下来，跟人类的 Gold Context 去做对齐比对。

### 2. 技术层面怎么对齐的

我们怎么比较 agent 看的代码跟人类标的代码？这里有个很 nice 的工程细节。代码在 repo 里是有层次的。File level 比较简单，就是路径匹配。到了 Block level，他们用 Tree-sitter 把代码抽成 AST，只取 definition 级的 node（比如 `function_definition`, `class_declaration`）。这样跨 8 种编程语言才具有可比性。最细的 Line level 直接拿 byte offset 算区间重叠。

然后是算分数。最基础的是 Recall 和 Precision。

Recall 就是：人类标的金标准里，你 agent 找回来了多少？
公式是 $Recall(C^A, C^G) = \frac{|C^A \cap C^G|}{|C^G|}$。这里 $C^A$ 是 agent 最终交出来的 context，$C^G$ 是 Gold Context，分子是交集的绝对大小，分母是 Gold Context 的大小。

Precision 就是：你 agent 找回来的这一堆代码里，有多少是真正有用的？
公式是 $Precision(C^A, C^G) = \frac{|C^A \cap C^G|}{|C^A|}$。分母变成了 agent 检索出来的 context 大小。

### 3. 最让人震惊的发现：The Bitter Lesson 重演

看论文 Table 2 的实验结果。他们拿最猛的 GPT-5 当 backbone，跑了 5 个不同的 agent 框架。那个最土的、连个花哨 tool 都没有、纯靠 bash 敲命令的 `mini-SWE-agent`，在 File-level F1 和 Line-level F1 上把所有花里胡哨的框架按在地上摩擦！SWE-agent、OpenHands、Prometheus 全军覆没。

这简直完美印证了 Rich Sutton 的 [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)。我们在 agent scaffolding 上堆了无数的工程复杂度，搞 embedding 检索、搞 graph 知识图谱，结果在纯 retrieval 质量上，打不过最原始的 LLM 驱动的 shell 交互。复杂的设计往往引入了额外的检索噪音，扼杀了 LLM 自身的 exploration 能力。

### 4. 极具启发性的新指标：Evidence Drop

论文里还有一个我觉得极具商业和工程价值的新 metric，叫 Evidence Drop，对应公式 (10)。

$$Drop = 1 - \frac{|C^A \cap C^G|}{|(\bigcup_{t=1}^T C_t^A) \cap C^G|}$$

这里的 $C_t^A$ 代表在第 $t$ 步 agent 探索到的 context。$\bigcup_{t=1}^T C_t^A$ 就是整个轨迹里 agent 所有看过的东西的并集。分母的意思是：在整个跑的过程中，agent **曾经看到过**的、且属于 Gold Context 的代码总量。分子是最终 patch 提交时保留的 Gold Context 数量。

结果你猜测试出来什么？Gemini 2.5 Pro 和 Devstral 2 的 Drop 指标高达 0.43！这说明啥？说明 agent 在探索 repo 的时候，明明已经扒到了正确的、能修 bug 的关键代码，但在最后做 patch 提交时，它把之前看到的证据直接给丢了！没有用上！

这直接点出了当前 coding agent 最大的 bottleneck：**检索成功了，但 Consolidation 失败了**。信息从 working memory 传递到最终推理时断裂了。

### 5. LLM 们的奇怪性格

看 Table 3 和 Table 4 里 LLM 的对比极度有意思。

GPT-5 的策略是 "few big bites"：平均只跑 5.87 步，每步疯狂看 119.29 行代码。这导致它 Recall 极高，但 Precision 垃圾，因为引入了太多无关代码干扰了最后的推理。
Claude Sonnet 4.5 采取 "moderate" 策略，步数和代码量都适中，最后 Pass@1 拿了第一。
Devstral 2 最神经质，跑 22 步，每次只看 11 行，不仅成本最贵，而且经常陷入 hallucination。论文 Appendix K 怀疑它被 SWE-bench 的训练数据污染了，直接盲猜整个解题过程，跳过 tool call 直接提交。

### 6. 我们能顺着这篇 paper 干点什么

我们可以从这篇论文顺藤摸瓜做特别多有意思的事。

**第一，拿这些 Gold Context 做 Supervised Fine-tuning。** 522,115 行高质量标注数据啊！完全可以训练一个 Context Retrieval Expert 小模型，输入 issue 描述，直接输出你需要去读哪几个文件的哪几行。彻底 bypass 那种长程的、容易 tunneling 的 agentic 探索。

**第二，基于 Process Reward 做 RL。** Evidence Drop 这个现象给了我们做 RL 的绝妙切入点。我们在 agent 的探索阶段给 dense reward，奖励它发现了新的 Gold Context（对应优化 AUC-Cov）；在提交阶段也给 reward，惩罚它丢弃了已经发现的证据（优化 Drop）。这就把纯 outcome 的稀疏 reward 变成了 process supervision 的密集 reward。

**第三，Architectural 改造。** 既然有 Usage Drop 的断裂，我们应该在 agent 里设计一个 Evidence Notebook 机制。每跑完一次 grep 或者 cat，强制 LLM 生成一句 "Evidence Summary"，最后打 patch 之前，强制把所有的 Summary 拿出来做一次 Consolidation Pass。这样能极大地压低 Drop 率，逼迫 agent 把看到的证据用起来。

### 7. Case Study 折射出的 Agent 智商缺陷

Appendix I 里三个 case 极度值得看。

**Case 1 (Prometheus):** 它去检索一个 class，只捞到了 methods，漏掉了 `__init__`。导致它传参数时直接 crash，因为它不懂这个 class 的初始化协议。这说明细粒度检索破坏了 OOP 的整体语义。
**Case 2 (Agentless):** Issue 提到 "db table collision"，Agentless 顺着 keyword 找到了 models 模块，但真正的 bug 藏在 core/checks 校验层。Agent 不会从 error code 反向 trace 架构，顺着表象 keyword 一路走到黑。
**Case 3 (OpenHands):** 这个叫 "Search-Induced Context Tunneling"。它 grep "TIME_ZONE" 第一个命中在 MySQL 模块，结果它的注意力完全被锚定在 MySQL 上，完全忽略了平行的 SQLite 和 Oracle 模块。哪怕它用 SQLite 复现了 bug，最后还是去改了 MySQL。这是典型的 context tunneling bias。

### 参考链接

*   [CONTEXTBENCH 官网](https://contextbench.github.io/)
*   [The Bitter Lesson by Rich Sutton](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
*   [SWE-bench Original Paper](https://arxiv.org/abs/2310.06770)
*   [SWE-agent Original Paper](https://arxiv.org/abs/2405.15793)

---

# CONTEXTBENCH 深度解读：打开 Coding Agent 的黑盒

Karpathy 你好，这篇论文切中了一个你长期关注的核心问题——**process supervision vs. outcome supervision**。NeurIPS 之前 AlphaProof 系的工作就在数学领域证明了 process reward 的价值，而 CONTEXTBENCH 把这种思路移植到 SWE agent 上：不再只看 Pass@1，而是 unbox agent 中间到底 retrieve 了什么 code context。下面我从 motivation、data construction、alignment 坐标系、metrics、实验发现、case study 几个层面拆解，重点放在能 build intuition 的技术细节上。

---

## 1. Motivation：为什么 end-to-end benchmark 不够用

SWE-bench [Jimenez et al. 2024](https://arxiv.org/abs/2310.06770) 系列只看 `Pass@k`——agent 提交 patch，跑 fail-to-pass + pass-to-pass tests，过即过。这种 outcome-only 评估存在三个根本问题：

1. **Lucky trial-and-error**：agent 可能靠 grep 撞对一个文件名，改两行就 pass test，但它的 reasoning chain 是错的，换一个相似 issue 就崩。
2. **Test overfitting**：SWE-bench 的 test cases 公开，agent scaffolding 可能隐式 fit 测试模式。
3. **无法归因 failure**：一个 agent 在 Pass@1 = 0.4 上失败，到底是 file localization 错了？element localization 错了？还是 explore 到了 gold context 但没用到？end-to-end 指标完全区分不了。

CONTEXTBENCH 引入的中间信号 `Gold Context` 就是回答第三问的钥匙。它的 formulation 是：

> Given issue $T$, the **gold context** $C^G$ is a compact set of code regions (files / AST blocks / lines) such that ∃ patch $p$ with $p \models \text{Tests}(T)$ and $p$ is generatable from $C^G$ alone (by a strong LLM).

注意 "compact sufficient" 的定义避开了 **global minimal sufficient set** 的 NP-hard 问题（在 repo 级 codebase 上是不可行的 combinatorial search），所以论文在 Appendix D 明确说：不要把 precision 解释成 "retrieve 任何额外 context 都该扣分"，而是 "retrieve 显然冗余才扣分"。

---

## 2. Data Construction Pipeline：从 4,497 到 1,136

整个 funnel 在 Figure 2 里画得很清楚，三步走：

### Step 1: Task Deduplication (4,497 → 3,100)

- **ID-based dedup**：repo name + issue ID 完全相同 → 删。4,497 → 3,981。
- **Semantic dedup**：用 embedding 模型把每个 issue description 编码成向量，计算 cosine similarity，threshold = 0.90 视为 near-duplicate → 删。3,981 → 3,100。
- 人工 review borderline cases 防误删。

为什么选 0.90 而不是 0.85 或 0.95？这是 trade-off：太低删得多但会丢 diversity，太高留得多但 redundancy 多。论文没给 sensitivity analysis，是个小遗憾。

### Step 2: Task Selection (3,100 → 1,136)

用三个 difficulty metrics 排序：

#### (a) Agent Solvability
爬公开 leaderboard，记录每个 task 被多少 agent 解决过。优先选 0-few shot 解决的 task——这些 task 对 retrieval 评估有区分度。

#### (b) Edit Scope
Gold patch 修改的文件数 $|F_{edit}|$。$|F_{edit}|$ 越大意味着 retrieval 需要跨越更多文件，越难。

#### (c) Edit Dispersion
用 tree-sitter 解析 repo，把所有 edited regions 映射到 repo tree 上，计算 edited regions 之间的平均 structural distance：

$$D_{disp} = \frac{1}{|F_{edit}|^2} \sum_{i,j \in F_{edit}} \text{treeDist}(i, j)$$

这里 $\text{treeDist}(i, j)$ 是 repo tree 上 file $i$ 到 file $j$ 的最短路径长度。$D_{disp}$ 越大说明修改跨多个 module，retrieval 越需要 cross-module reasoning。

经过 manual review 去掉 "看似难但语义 trivial" 的 case（比如 bulk rename、formatting only），1,500 → 1,136。

### Step 3: Expert Annotation (1,136 → 1,136 verified)

这是工作量最大的一步，6 个作者 + 一组 expert developer，4 个月，每 task 平均 40 分钟（range: 20 min ~ 1.5h）。

**Annotation Protocol**：
1. 从 gold patch 的 modified regions 出发。
2. Trace 这些 regions 涉及的：
   - function/class invocations
   - inheritance relations
   - control-flow & data-flow paths
   - same-file/module 内 semantically relevant 的 components
3. **Minimal context principle**：只保留 strictly necessary 的 region，剔除 redundant/irrelevant。
4. 多轮 blind annotation 防止 anchoring bias（annotator 互相不知道对方 annotation）。

**Verification (Appendix G 的细节很关键)**：
用 GPT-5 作为 verifier，**仅**把 annotated context 喂给 GPT-5（不给 repo access、不给 retrieval tool），让它生成 5 个 candidate patches。Sufficiency criterion：

$$\text{Sufficient}(C^G) \iff \exists p \in \{p_1, ..., p_5\}: p \models \text{Tests}(T)$$

这是 **existence-based** 而非 consistency-based 的判据——不要求 GPT-5 5 次都对，只要求至少 1 次对。这样的好处是：避免 GPT-5 自己 capability 不足被误判为 context 不足。坏处是：context 可能 too generous（包含一些实际不必要的 region），所以需要 Refinement 步骤。

**Refinement**：通过 verification 的 context 给另一个 annotator 做 compactness checking，去掉冗余 region，然后两人 jointly reconcile。未通过 verification 的 context 进入最多两轮 extra annotation，每轮换新 annotator。最终 3 个 annotator 一起 reconcile。

### Robustness Check (RQ5)

同一个 issue 可能有多个 semantically equivalent patches（实现不同但都 pass tests）。他们做了 82 个 case study，每个 case 收集 2-3 个等价 patches，对每个 patch 都 derive 一个 patch-conditioned gold context $G^{(k)}$，然后计算 pairwise Jaccard similarity：

$$\text{Jaccard}(G^{(i)}, G^{(j)}) = \frac{|G^{(i)} \cap G^{(j)}|}{|G^{(i)} \cup G^{(j)}|}$$

平均 Jaccard = **0.9518**，距离 = 0.0482。这个数字非常关键——它说明 gold context 不依赖特定 patch 的 syntax，而是 capture 了 issue 的语义本质。这也间接支持了 CONTEXTBENCH evaluation 的 validity：用单一 gold context 作 reference 不会严重低估那些走 "另一条合法路径" 的 agent。

---

## 3. Alignment 坐标系：tree-sitter 三层 granularity

这是我觉得工程上最 elegant 的部分。问题本质是：agent retrieved 的 context 和 human annotated 的 gold context 都是 `(file_path, line_range)` 的集合，但两者的 granularity 不一致（agent 可能看了整个 file，gold 可能只标了某几行）。需要在统一坐标系上算 overlap。

### File-Level Alignment
最简单，按 file path match。

### Block-Level (AST) Alignment
用 tree-sitter 解析每个文件，提取 **definition-level** AST nodes 作为 block 单位（详见 Table 6）：

| Language | Node types |
|---|---|
| Python | `function_definition`, `class_definition`, `async_function_definition` |
| JavaScript | `function_declaration`, `class_declaration`, `method_definition`, `arrow_function` |
| TypeScript | + `interface_declaration` |
| Java | `class_declaration`, `interface_declaration`, `method_declaration`, `constructor_declaration` |
| Go | `function_declaration`, `method_declaration`, `type_declaration` |
| Rust | `function_item`, `impl_item`, `struct_item`, `trait_item` |
| C/C++ | `function_definition`, `class_specifier`, `struct_specifier` |
| ... | ... |

注意这里**刻意不用任意 AST node**，只用 definition-level symbol。为什么？因为低层 syntax node 的 granularity 在不同 language parser 之间差异巨大（比如 `expression_statement` 在 Python 和 Go 的语义完全不同）。用 definition-level 保证 cross-language comparability。

每个 block 用 `(file_path, start_line, end_line)` 唯一标识，start/end line 由 tree-sitter 的 byte offset 转换得到。

### Line-Level Alignment
按 byte offset 算 interval overlap。例如 gold context 是 `[10, 50]`，agent retrieved 是 `[30, 70]`，overlap 是 `[30, 50]`。

这种三层 granularity 设计的妙处：file-level 给一个粗 overview，block-level 对齐到 API 语义边界，line-level 最严格——它能区分 agent 是看了整个 function 还是只看了 docstring。

---

## 4. Metrics 全景：从静态到动态

### 4.1 静态 metrics（针对 final retrieved context $C^A$）

$$\text{Recall}(C^A, C^G) = \frac{|C^A \cap C^G|}{|C^G|} \quad (1)$$

$$\text{Precision}(C^A, C^G) = \frac{|C^A \cap C^G|}{|C^A|} \quad (2)$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \quad (3)$$

其中 $|C^A \cap C^G|$ 在三层 granularity 上分别用 path match / block span overlap / line interval overlap 算。

### 4.2 动态 metrics（针对 trajectory $\{C_t^A\}_{t=1}^T$）

这是论文最有创新的部分，把 retrieval 当 process 而不只是 outcome。

#### Cumulative Explored Context

$$A^{(t)} = \bigcup_{i=1}^{t} C_i^A \quad (4)$$

到第 $t$ 步为止 agent 看过的所有 region 的 union。

#### AUC-Cov (Retrieval Efficiency)

$$\text{AUC-Cov} = \frac{1}{T} \sum_{t=1}^{T} \text{Recall}(A^{(t)}, C^G) \quad (5)$$

这是 cumulative gold coverage curve 下方的归一化面积。**直觉**：AUC-Cov 高意味着 agent 早期就 explore 到了大部分 gold context；AUC-Cov 低意味着 agent 早期都在看无关代码，很晚才"撞上" gold。

#### Redundancy

$$\text{Redun}_t = \frac{|C_t^A \cap (\bigcup_{i=1}^{t-1} C_i^A)|}{|C_t^A|} \quad (6)$$

$$\text{Redun} = \frac{1}{T-1} \sum_{t=2}^{T} \text{Redun}_t \quad (7)$$

$t$ 步看过的代码中有多大比例是前 $t-1$ 步已经看过的。**直觉**：高 redundancy 说明 agent 在 loop——反复 re-read 同一文件，陷入 cycle。低 redundancy 说明每一步都在 acquire 新 context。

#### Evidence Drop (Explored ≠ Used)

这是最 insightful 的 metric。

$$G_{seen} = \left(\bigcup_{t=1}^{T} C_t^A\right) \cap C^G \quad (8)$$

整个 trajectory 中至少被 explore 过一次的 gold context。

$$\text{Keep} = \frac{|C^A \cap C^G|}{|G_{seen}|} \quad (9)$$

$$\text{Drop} := 1 - \text{Keep} = 1 - \frac{|C^A \cap C^G|}{|(\bigcup_{t=1}^T C_t^A) \cap C^G|} \quad (10)$$

**直觉**：Drop 衡量"明明看到了 gold evidence 却没在 final patch context 中保留"的比例。Drop 高说明 agent 的 bottleneck 不是 retrieval 本身，而是 **consolidation**——把 explore 到的证据合理 retain 进 final reasoning 的能力。

这个 metric 直接指向一个关键 design insight：current agent 的 failure mode 里，"retrieval success → patch failure" 占了很大比例。如果你只在 retrieval stage 加 RL reward，根本解决不了这个 gap——需要在 consolidation stage 也加 process reward。

---

## 5. Agent Context Tracing 的工程细节

Appendix C 给了很多 implementation 细节，值得细看。

### 5.1 mini-SWE-agent 的 two-stage submission protocol

agent 提交前必须先输出一个 `<PATCH CONTEXT>` block：

```
<PATCH_CONTEXT>
File: /testbed/src/core/handler.ext
Lines: 34-56
File: /testbed/lib/utils.ext
Lines: 128-145
</PATCH_CONTEXT>
```

实现上是用一个 `ContextRequested` exception 在 agent 想调用 submit 时拦下来，要求先 emit context，verify format 通过后才允许 submit。这种 exception-based control flow 在 agentic framework 里很 hacky 但很 effective。

### 5.2 Cross-agent adaptation

不同 agent 的 prompt template 不一样，但都用同样的"output `<PATCH CONTEXT>` before submit"约束：
- **SWE-agent**：加 `Pre-Submission Context Enforcement Specification` 到 system prompt。
- **OpenHands**：额外加 `<EXPLORE_CONTEXT>` block 在每个 read command 之前 emit（不只是最终提交时）——这能记录 intermediate exploration step。
- **Agentless**：改 prompt 让它支持 multi-language（dynamic file extension detection + language-agnostic "Hello World" DIFF example）。
- **Prometheus**：基于 knowledge graph 的 retrieval，同样 enforce `<PATCH_CONTEXT>` 输出。

### 5.3 Devstral 2 的异常行为 (Appendix K)

Devstral 2 在 mini-SWE-agent 上出现一个非常有意思的失败模式：它读 issue 后**单次 response** 里就把整个 debug+patch 流程规划出来，**hallucinate 中间 system state**（仿佛已经看到了 bash 输出）。当 framework 返回 "Please provide EXACTLY ONE action" 错误时，它直接跳到 final submit。

论文 hypothesis 是 **data contamination**——Devstral 2 训练时可能见过 SWE-bench 的 instances 和 patches，导致 benchmark-specific memorization。这是 LLM benchmark 评估一个越来越严重的 hygiene 问题，也是为什么我觉得 SWE-bench Verified 之外需要像 SWE-bench Live [Zhang et al. 2025](https://arxiv.org/abs/2505.21658) 这种动态更新的 benchmark。

---

## 6. 实验结果深度解读

### 6.1 RQ1: Coding Agent 对比 (Table 2)

所有 agent 都用 GPT-5 作为 backbone，在 CONTEXTBENCH Lite (500 tasks) 上跑。

| Agent | File Recall | File Prec | Block Recall | Block Prec | Line Recall | Line Prec | Pass@1 |
|---|---|---|---|---|---|---|---|
| **mini-SWE-agent** | 0.682 | **0.709** | 0.645 | 0.369 | **0.606** | 0.301 | 0.472 |
| Agentless | 0.609 | 0.352 | 0.328 | 0.344 | 0.461 | 0.318 | 0.452 |
| SWE-agent | **0.726** | 0.537 | **0.625** | 0.312 | 0.476 | 0.228 | 0.490 |
| OpenHands | 0.733 | 0.400 | 0.505 | 0.283 | 0.472 | 0.203 | 0.490 |
| Prometheus | 0.717 | 0.336 | 0.646 | 0.258 | 0.584 | 0.195 | **0.512** |

**最 shocking 的发现**：mini-SWE-agent（最简单的、只用 bash 命令的 baseline）在 file-level F1 (0.634) 和 line-level F1 (0.312) 上**都超过**了四个 state-of-the-art scaffolds。这就是论文标题里的 "The Bitter Lesson of Coding Agents"——致敬 Rich Sutton [2019](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) 那篇经典 essay。

为什么复杂 scaffold 反而退化？我的理解：
- Agentless 用 embedding-based semantic retrieval，但 embedding 在 SWE 任务上常常 retrieve 表面相似但语义无关的代码，把 precision 拖垮（0.352 vs mini-SWE 0.709）。
- SWE-agent 的 ACI (agent-computer interface) 把 file navigation 包装成 specialized tool，agent 反而失去了 raw shell 的灵活性——比如 grep + sed + awk 这种组合。
- OpenHands 的 file editor 让 agent edit 时容易引入 noise context。
- Prometheus 用 knowledge graph retrieval，graph 覆盖率本身限制 recall。

但 Pass@1 上 Prometheus 最高 (0.512)。这就引出一个有意思的 paradox：retrieval metric 好不代表 Pass@1 好，反过来也是。这可能是因为 gold context 是 "necessary but not sufficient"——agent 还需要 reasoning、planning、edit 等能力，而 Prometheus 的 graph-based retrieval 可能在 reasoning 上提供了额外 signal。

### 6.2 RQ2: LLM 对比 (Table 3)

固定 agent scaffold 为 mini-SWE-agent，换不同 LLM backbone。

| LLM | File F1 | Block F1 | Line F1 | Pass@1 |
|---|---|---|---|---|
| GPT-5 | 0.634 | 0.375 | 0.312 | 0.472 |
| **Claude Sonnet 4.5** | 0.624 | 0.420 | **0.344** | **0.530** |
| Gemini 2.5 Pro | 0.600 | 0.403 | 0.311 | 0.364 |
| Devstral 2 | 0.615 | 0.422 | 0.332 | 0.402 |

几个关键观察：

1. **All LLMs favor recall over precision**。看 Figure 1b 雷达图，所有模型在所有 granularity 上的 recall 都高于 precision。在 line level 上 recall 是 precision 的 2 倍。这反映 LLM 内在的 "exploration-first" 倾向——宁可多看一点也不要漏。

2. **GPT-5 aggressive retrieve**：line recall 0.606（最高），但 line precision 0.301（次低），导致 line F1 0.312 反而不如 Claude。这种策略在 retrieval 上看 coverage 好看，但引入太多 noise 影响 final patch reasoning。

3. **Claude Sonnet 4.5 平衡策略胜出**：line F1 0.344 最高，Pass@1 0.530 最高。它的 block precision 0.449 是所有模型里最高的，说明它 retrieval 更精确。

4. **Gemini 2.5 Pro 保守**：line recall 只 0.313（最低），但 line precision 0.529（最高）。这种 "看少看准" 的策略 Pass@1 只 0.364，因为漏掉了关键 gold context。

### 6.3 RQ3: Retrieval Pattern (Table 4)

| LLM | Avg Steps | Avg Lines/Step | Avg Cost ($) |
|---|---|---|---|
| GPT-5 | **5.87** (↓) | **119.29** (↑) | 0.45 |
| Claude Sonnet 4.5 | 14.38 | 29.74 | 0.76 |
| Gemini 2.5 Pro | 7.57 | 26.29 | **0.38** (↓) |
| Devstral 2 | **22.16** (↑) | **11.98** (↓) | **0.91** (↑) |

GPT-5 是 "few big bites"——5.87 步，每步看 119 行。Devstral 2 是 "many small bites"——22 步，每步 12 行。Claude 居中。**Claude 居中策略 → 最好 line F1 + 最好 Pass@1**。

这给我一个很强的 intuition：retrieval pattern 的 sweet spot 是 **moderate rounds × moderate granularity**。太少太大 → context 稀释（precision 损失）；太多太小 → trajectory 拖长 cost 增加 redundancy。

Cost 上 Gemini 最便宜 0.38/instance，因为它 step 少 + 每 step 看的不算多。Devstral 最贵 0.91，因为 22 步的 output token 成本累积。**Reducing queries = reducing cost**——这个结论对部署 agent 的人很 actionable。

### 6.4 RQ4: Retrieval Dynamics (Table 5)

| LLM | Efficiency ↑ | Redundancy ↓ | Usage Drop ↓ |
|---|---|---|---|
| GPT-5 | 0.591 | **0.487** (↓) | **0.179** (↓) |
| Claude Sonnet 4.5 | **0.658** (↑) | 0.708 | 0.196 |
| Gemini 2.5 Pro | 0.529 | 0.558 | 0.431 |
| Devstral 2 | 0.616 | 0.672 | **0.435** (↑) |

- **Claude efficiency 最高 0.658**：很早就 reach 高 gold coverage。但 redundancy 0.708 也最高——它会反复 re-read 同一文件保持 coverage，有点像 "anchoring"。
- **GPT-5 redundancy 最低 0.487**：每一步都在看新东西，符合它 "few big bites" 的策略。
- **Gemini 和 Devstral 的 Usage Drop 惊人地高 (0.43+)**：意味着它们 explore 到的 gold context 有 43% 在 final patch 中没用到！这是 consolidation bottleneck 的直接证据。

**Usage Drop 这个 metric 对 agent design 有重要意义**：current agent 把 explore 和 consolidate 当作两个 stage，但中间没有 mechanism 强制 agent "retain explore 到的 gold evidence"。可能的改进方向：
- 让 agent 在每个 explore step 后做一次 "evidence summary"，把看过的关键 code snippet 提炼成 short note。
- 在 final patch generation 前 force agent 做一次 "evidence consolidation pass"，对照之前 explore 的所有 context。
- 用 process reward 在 explore step 时奖励 "discovered new gold evidence"，在 submit step 时奖励 "retained gold evidence"。

### 6.5 RQ5: Gold Context Robustness

82 个 case，每个 2-3 个 semantically equivalent patches，pairwise Jaccard = 0.9518。这告诉我们 gold context 在 "what code is relevant to this issue" 这个维度上是 stable 的——无论你选哪条合法修复路径，需要看的代码差不多。

---

## 7. Case Studies：失败模式分类

Appendix I 三个 case study 非常有教学意义，我详细拆。

### Case 1: Prometheus — Incomplete Class Semantics Retrieval

**Task**: `psf/requests#1921`，setting session headers to None 时应该 omit 而不是发 literal "None"。

**Failure**: Prometheus retrieve 到了 `CaseInsensitiveDict.__setitem__` 和 `__getitem__`（lines 71-78），但漏掉了 `__init__` 和 `update` 方法。它的 patch 是：

```python
# agent's buggy patch
merged_setting = dict_class((k, v) for k, v in ... if v is not None)
```

它把 generator expression 传给 `CaseInsensitiveDict`，但 `CaseInsensitiveDict.__init__` 要求 `.items()` 接口（mapping-like），不接受 generator。

**Gold patch**:
```python
merged_setting = dict((k, v) for (k, v) in merged_setting.items() if v is not None)
```

**Root cause**: Element-level slicing insufficiency——graph-based retrieval 在 class 这个粒度上把 methods 当独立 node，丢失了 class 作为整体的 constructor semantics。

**Intuition**: 当 retrieval 粒度太细（method-level），agent 会拿到"操作"但漏掉"初始化协议"。这种 partial retrieval 在 OOP 重构场景里很常见。

### Case 2: Agentless — File Localization Failure

**Task**: `django/django#11630`。

**Failure**: Agentless 在 file localization stage 选了 10 个 files，5 个来自 `django/db/models/*`，5 个 framework config。但 gold patch 修改的是 `django/core/checks/model_checks.py`——这个文件**从未被 retrieve**。

**Cascade**: 错误 file set → Stage 2 element localization 在错的 class 上 → Stage 3 edit 在错的 file 上。

**Root cause**: Information architecture gap。Issue description 提 "db table collision"（symptom-level），agent keyword match 找到 models 模块。但 issue 的实际 root cause 在 `core/checks/` 这个 validation framework 里——issue 文本里根本没提 checks。Agent 没有"从 error code (E028) 反向 trace 到 source file"的能力。

**Intuition**: 在大型 framework 里，symptom 和 fix 经常在不同 architectural layer。Agent 需要 "backward dependency tracing"（从 error code → validator → 调用方）而不是纯 forward keyword search。

### Case 3: OpenHands — Cross-Context Exploration Gap

**Task**: `django/django#11138`，TIME_ZONE 在 multi-backend 配置下被忽略。

**Failure**: OpenHands 用 grep 搜 "TIME_ZONE"，结果锚定到 MySQL backend（因为这个 SQL function 在 MySQL 独有）。它甚至在 SQLite 环境里成功 reproduce 了 bug，但 fix 时回到 MySQL 路径，完全 miss 了 SQLite 和 Oracle backends 的并行 modules。

**Root cause**: Search-Induced Context Tunneling。Grep 第一个 hit 充当 anchor，抑制了对 parallel modules 的 horizontal exploration。`ls -R` 列出了 `sqlite3/` 和 `oracle/` 目录，但 agent 没触发 follow-up view。

**Intuition**: Keyword-driven search 在 modular architecture（同 semantic、不同 syntax 的多文件）上系统性失败。需要的不是更好 search，而是 search 后的"horizontal generalization"——发现多个相似 module 并对每个都 audit。

---

## 8. Limitations & Open Questions

论文有几个值得讨论的 limitation：

1. **Gold context 是 patch-conditioned**：它从 gold patch 出发 trace 依赖。但有些 issue 可能存在完全不同的修复路径，gold context 无法覆盖所有可能性。虽然 RQ5 的 Jaccard 0.95 显示这种 bias 不大，但 0.05 的 distance 在 critical path 上仍可能影响 evaluation。

2. **Verification 用 GPT-5**：把 GPT-5 当 oracle 验证 context sufficient。但 GPT-5 自己可能也 fail 在某些 task 上，导致 context 被错误标为 insufficient 进入 refinement loop，引入 annotation cost。这是 chicken-and-egg 问题。

3. **Block-level alignment 用 definition-level AST nodes**：跨语言 comparable，但忽略了 statement-level 的 granularity。对 bug-fixing 任务，关键的可能是某一行 if 条件而不是整个 function。Line-level metric 弥补了这点，但 line-level F1 普遍低（0.31-0.34）说明 agents 在细粒度上确实 struggle。

4. **No process-level reward training**：论文只做了 evaluation，没有用这些 intermediate signal 训练 agent。如果能拿 `AUC-Cov` 或 `Drop` 做 RL reward 训练 agent，可能直接验证 process supervision 在 SWE 上的效果，呼应 OpenAI o1 / AlphaProof 那条线。

5. **"Bitter Lesson" 的 caveat**：mini-SWE-agent 在 retrieval metric 上赢，但 Pass@1 上 Prometheus 赢（0.512 vs 0.472）。这说明 retrieval quality 不是 Pass@1 的充分条件。复杂 scaffold 可能别处有用（比如 edit planning, test generation）。论文承认了这点但没深挖。

---

## 9. 对你的 research 直觉的可能启发

我猜你读完这篇会想到几个方向：

**(a) Process Reward Model for SWE**：把 `AUC-Cov` 和 `Drop` 作为 dense reward 训练 PRM，对 agent trajectory 每个 step 打分。这比纯 outcome reward signal dense 多了，可能解决 SWE-RL 的 sparse reward 问题。可以参考 [SWE-Gym](https://arxiv.org/abs/2412.21139) 的思路加 process supervision。

**(b) Retrieval-Consolidation Decoupling**：Usage Drop 0.43 说明 explore 和 consolidate 应该是两个 separate module。可以设计 agent 让一个 sub-agent 专门做 explore + evidence notebook，另一个 sub-agent 专门做 consolidation + patch generation，中间用 structured note 传递。

**(c) Test-time compute trade-off**：Table 4 里 cost 从 0.38 到 0.91 跨 2.4 倍，Pass@1 从 0.364 到 0.530。这条 scaling curve 在 agent 上还没被好好 characterizing——能不能像 LLM scaling law 那样写出 `Pass@1 = f(test_time_compute)` 的公式？

**(d) Anti-tunneling retrieval policy**：Case 3 的 "Search-Induced Context Tunneling" 是个 systematic failure mode。可以设计 "first-hit diversification"——grep 第一个 hit 后强制 explore 其他 matching file，或者 "module-graph-aware retrieval"——按 repo module graph 做 BFS 而不是 keyword DFS。

**(e) Gold Context as Distillation Target**：522,115 行 gold context 是个 huge dataset，可以用来 SFT 一个 "context retrieval expert" model，让它学会从 issue + repo 直接 predict gold context。这就是 retrieval 的 imitation learning 路线，可以完全 bypass agentic loop。

---

## 10. 参考链接

- [CONTEXTBENCH 项目主页](https://contextbench.github.io/)
- [SWE-bench (Jimenez et al. 2024)](https://arxiv.org/abs/2310.06770)
- [SWE-agent (Yang et al. 2024)](https://arxiv.org/abs/2405.15793)
- [OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [Agentless (Xia et al. 2024)](https://arxiv.org/abs/2407.01489)
- [Prometheus (Chen et al. 2025)](https://arxiv.org/abs/2507.19942)
- [Multi-SWE-bench (Zan et al. 2025)](https://arxiv.org/abs/2504.02605)
- [SWE-PolyBench](https://arxiv.org/abs/2504.08703)
- [SWE-bench Pro](https://arxiv.org/abs/2509.16941)
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
- [The Bitter Lesson (Rich Sutton)](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
- [GPT-5 System Card](https://arxiv.org/abs/2601.03267)
- [Claude Sonnet 4.5 Announcement](https://www.anthropic.com/news/claude-sonnet-4-5)
- [Gemini 2.5 Technical Report](https://arxiv.org/abs/2507.06261)
- [SWE-Gym (Pan et al. 2025)](https://arxiv.org/abs/2412.21139)
- [SWE-bench Live](https://arxiv.org/abs/2505.21658)

---

**一句话总结这篇论文**：它把 SWE-bench 的 binary outcome 指标升级成 retrieval process 的 dense signal，发现 agent 在 "看到 gold context" 和 "用上 gold context" 之间存在显著 consolidation gap，且复杂 scaffold 在 retrieval 上并没胜过简单 bash baseline——这给 process reward training、retrieval-consolidation 解耦、scaling law for agentic test-time compute 都提供了新的 evaluation 基础。

如果你下一步想推 SWE agent 的 RL 训练，CONTEXTBENCH 的 `AUC-Cov` 和 `Drop` 是天然的 dense reward 信号；如果想推 retrieval 路线，那 522K 行 gold context 是个绝佳的 distillation dataset。这两个方向我觉得都直接 build on 这篇工作。
