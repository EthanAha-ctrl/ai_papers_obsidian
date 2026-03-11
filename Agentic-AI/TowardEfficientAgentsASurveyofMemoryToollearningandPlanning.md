
## 一、论文背景与核心动机

### 1.1 为什么需要"高效Agent"？

关键问题：

> **虽然Agent的能力持续提升，但效率——对真实世界部署至关重要的因素——往往被忽视。**

**核心矛盾**：
- 传统LLM是单次查询-响应模式
- Agent却是**递归循环**模式：Input → Memory → Planning → Tool Learning → Observation → (循环) → Solution

每轮迭代中，step n的输出成为step n+1的输入成本，导致**指数级增长的token消耗**、**高延迟**和**上下文窗口饱和**。

### 1.2 效率的定义公式

论文给出了一个清晰的Agent成本公式：

$$\text{Cost}_{\text{agent}} \approx \alpha \cdot N_{\text{tok}} + \mathbb{I}_{\text{tool}} \cdot \text{Cost}_{\text{tool}} + \mathbb{I}_{\text{mem}} \cdot \text{Cost}_{\text{mem}} + \mathbb{I}_{\text{retry}} \cdot \text{Cost}_{\text{retry}}$$

其中：
- $N_{\text{tok}}$: 生成的推理token数量
- $\mathbb{I}_{\text{tool}}, \mathbb{I}_{\text{mem}}, \mathbb{I}_{\text{retry}} \in \{0,1\}$: 是否调用工具、访问内存、重试的指示变量
- $\alpha$: 每token成本（时间或金钱）

**核心洞察**：提升Agent效率不仅是减少语言生成，还要**降低工具/内存调用频率**，优化轨迹上的选择性。

---

## 二、Efficient Memory（高效记忆）

这是论文最详细的部分，占据约1/3篇幅。记忆机制被组织为**生命周期三阶段**：

### 2.1 Memory Construction（记忆构建）

目标：将长交互上下文压缩为可管理的表示，解决"lost in the middle"问题。

#### 2.1.1 Working Memory（工作记忆）

**Textual Memory（文本记忆）**：
- **COMEDY**: 两阶段记忆蒸馏，提取会话特定记忆并压缩为用户画像和关键事件
- **MemAgent/MEM1**: 顺序处理长输入，每步重写/更新紧凑记忆状态
- **AgentFold**: 主动将交互历史折叠为多尺度摘要+最新完整轮次

**Latent Memory（隐式记忆）** - 这是更有趣的方向：
- **Activation Beacon**: 将上下文分块，通过压缩比交错beacon token，逐层蒸馏KV激活到beacons
- **MemoRAG**: 在每个窗口后插入memory token作为全局记忆的信息载体，用独立权重矩阵更新memory-token KV cache
- **MemoryLLM**: 维护固定大小的memory token池，通过self-update更新，实现知识复用而不延长prompt
- **M+**: CPU/GPU两级长期记忆，每层co-trained retriever只获取少量相关memory token
- **Titans**: 测试时更新的神经记忆模块，预测误差高时才写入

**工作记忆优缺点**：
- ✅ 直接参与生成，无外部检索延迟
- ❌ 文本记忆计算随长度增长；隐式状态增加内存占用；长上下文信息稀释

#### 2.1.2 External Memory（外部记忆）

**Item-based Memory（条目式记忆）**：
- **MemoryBank**: 基于艾宾浩斯遗忘曲线的记忆管理
- **Expel**: 试错积累经验，蒸馏为自然语言insights
- **SeCom**: 分段模型划分话题一致段落，压缩模型去噪
- **Memory-R1/Mem0**: 提取对话为候选记忆，进行CRUD操作（CREATE/READ/UPDATE/DELETE）
- **ReasoningBank**: 从成功/失败轨迹中蒸馏策略，减少探索步数

**Graph-based Memory（图结构记忆）**：
- **GraphReader**: 长文本分块→压缩为关键元素和原子事实→构建图捕捉长程依赖
- **Zep**: 时间感知知识图，提取实体关系，存储带有效期的fact edges
- **AriGraph**: 统一的语义-情景记忆图，语义三元组+情景节点通过情景边链接

**Hierarchical Memory（分层记忆）**：
- **MemGPT**: OS风格虚拟内存分页，context分为system instructions、writable working context、FIFO message buffer
- **MemoryOS**: 三层存储（STM短期/MTM中期/LTM长期），基于Heat score的迁移策略
- **LightMem**: 感知模块预压缩→STM话题分段→LTM定期摘要

### 2.2 Memory Management（记忆管理）

解决记忆空间爆炸问题。

**Rule-based（基于规则）**：
- 艾宾浩斯遗忘曲线衰减（MemoryBank, H-MEM）
- FIFO替换策略（MemGPT, MemoryOS）
- ⚠️ 优点：快速、可预测、低成本；缺点：静态规则可能盲目剪枝关键信息

**LLM-based（基于LLM）**：
- **Memory-R1/Mem0**: 检索相似条目后，LLM选择ADD/UPDATE/DELETE/NOOP操作
- **RMM**: retrieve-then-update模式，LLM决定merge或add
- **A-MEM**: 生成式更新，LLM创建链接、重写相关笔记

**Hybrid（混合）**：
- 分层触发+LLM更新（MemoryOS, LightMem, LD-Agent）
- 图结构记忆：规则更新+LLM验证矛盾（Zep, Mem0g, AriGraph）
- 生命周期策略：MemOS的MemScheduler和MemVault进行去重、冲突处理、归档

### 2.3 Memory Access（记忆访问）

**Memory Selection（记忆选择）**：
- **Rule-enhanced Retrieval**: 加入时间（recency）、重要性、频率等评分（Generative Agents, Human-like memory）
- **Graph-based Retrieval**: 从查询相关事实锚定，扩展邻居到局部子图（AriGraph, Mem0g）
- **LLM/Tool-based Retrieval**: 将记忆层暴露为工具，LLM选择层级和操作（MemGPT）
- **Hierarchical Retrieval**: 多层索引，递归检索（H-MEM的四层结构）
- **Training-based**: 在线RL更新reranker（RMM的Retrospective Reflection信号）

**Memory Integration（记忆整合）**：
- **Textual**: 过滤+压缩+结构化插入（RECOMP的Retrieve-Compress-Prepend, DC-RS的cheatsheet）
- **Latent**: 存储为紧凑隐藏状态或KV对，通过attention重用（MemoryLLM, Memory³, MemoRAG）

### 2.4 Multi-Agent Memory（多智能体记忆）

**Shared Memory（共享记忆）**：
- **MS**: Prompt-Answer对存储，LLM评估器过滤后加入共享池
- **G-Memory**: 三层图层级（insight/query/interaction），双向遍历检索
- **MIRIX**: 六模块分层记忆，Meta Memory Manager治理，Active Retrieval注入系统prompt

**Local Memory（本地记忆）**：
- **Intrinsic Memory Agents**: 角色对齐模板，每轮将最新输出折叠回同一模板
- **AgentNet**: 固定大小记忆模块，频率/最近性/独特性信号动态剪枝

**Mixed Memory（混合记忆）**：
- **SRMT**: 个人记忆向量 + 全局广播共享循环记忆
- **LEGOMem**: 角色感知记忆路由，运行时高效检索调度

---

## 三、Efficient Tool Learning（高效工具学习）

### 3.1 Tool Selection（工具选择）

解决从大规模工具池中高效选择的问题：

**External Retriever（外部检索器）**：
- **ProTIP**: 对比学习嵌入查询和工具描述，选择后从查询嵌入中减去已选工具表示，递归选择子任务工具
- **AnyTool**: 分层检索，分治策略缩小搜索空间

**Multi-Label Classification（多标签分类）**：
- **TinyAgent**: DeBERTa-v3 small模型输出概率分布，>50%阈值的工具被选中
- **Tool2Vec**: 基于合成使用示例而非静态描述生成工具嵌入，两阶段检索+reranker

**Vocabulary-based Retrieval（词汇检索）**：
- **ToolkenGPT**: 将工具视为可学习的token嵌入（"toolken"），工具选择=下一个token预测
- **Toolken+**: 增加reranking步骤和rejection toolken，可调候选数量平衡效率-效果
- **ToolGen**: 每个工具分配唯一tool token，将检索和调用统一为生成任务
- **Chain-of-Tools**: 只用单个toolken，retriever计算与候选的相似度

### 3.2 Tool Calling（工具调用）

**In-Place Parameter Filling（原地参数填充）**：
- **Toolformer**: 在CoT路径中嵌入工具调用，达到闭包时获取结果
- **CoA**: 符号抽象表示中间步骤，后续替换为实际结果，比Toolformer减少30%推理时间

**Parallel Tool Calling（并行工具调用）**：
- **LLMCompiler**: 编译器启发框架，制定执行计划、分发任务、并行执行函数
- **LLM-Tool Compiler**: 运行时选择性融合相似工具操作
- **CATP-LLM**: 成本感知离线RL微调，多分支规划语言

**Cost-Aware Tool Calling（成本感知）**：
- **BTP (Budget-Constrained Tool Learning with Planning)**: 将工具调用建模为背包问题，动态规划预计算各工具在硬预算下的调用频率
- **TROVE**: 在线增量构建可重用函数工具箱，无需额外训练数据
- **ToolCoder**: 将工具学习形式化为端到端代码生成任务

**Test-Time Scaling（测试时扩展）**：
- **ToolChain\***: A*搜索策略，任务特定成本函数早期剪枝错误分支，单步节点扩展

**Post-training（后训练优化）**：
- **OTC-PO**: 将工具使用惩罚整合进RL目标，训练模型最小化冗余调用
- **ToolOrchestra**: 效率感知奖励训练专业化编排器
- **ToolRM**: 专业化基于结果的奖励模型，数据高效微调和推理时扩展

### 3.3 Tool-Integrated Reasoning（工具集成推理）

**Selective Invocation（选择性调用）**：
- **TableMind**: 计划-行动-反思循环，SFT热身建立工具使用模式+RL优化
- **SMART**: CoT数据集详细说明每个工具调用的必要性，微调模型决定使用参数知识还是外部工具
- **Agent-FLAN**: 将格式遵循数据与一般推理数据分离，进一步分解为能力特定子集

**Cost-Aware Policy Optimization（成本感知策略优化）**：
- **TableMind (RAPO)**: Rank-Aware Policy Optimization，识别不对齐轨迹，排名感知优势加权
- **ARTIST**: 统一agentic推理与基于结果的RL，无限制性步骤级监督学习最优工具使用策略
- **ReTool**: 代码解释器集成到推理循环，动态交错自然语言与可执行代码，可验证奖励信号发现策略
- **ToolRL**: 格式奖励 + 正确性奖励（工具参数与ground truth匹配）
- **AutoTIR**: 特定奖励惩罚抑制不必要工具使用
- **PORTool**: 衰减因子$\gamma$强调接近最终结果的步骤，偏好更少工具调用步数的解决方案

---

## 四、Efficient Planning（高效规划）

将推理重新框架为**资源受限控制问题**，而非无界生成。

### 4.1 Single-Agent Planning Efficiency（单智能体规划效率）

**Inference-Time Strategies（推理时策略）**：

*I. Adaptive Budgeting and Control（自适应预算控制）*：
- **SwiftSage**: 快/慢双过程架构（System 1启发式 + System 2结构化推理）
- **Budget-Aware**: 基于预算约束动态调整工具策略
- **Reflexion**: 言语强化从先前失败中学习，摊销失败分析成本

*II. Structured Search（结构化搜索）*：
- **LATS (Language Agent Tree Search)**: 将agent rollout重构为蒙特卡洛树搜索，自反思引导探索
- **CATS**: 成本感知直接整合进搜索树，早期剪枝昂贵分支
- **ToolChain\***: A*搜索导航动作空间

*III. Task Decomposition（任务分解）*：
- **ReWOO/Alita**: 解耦规划与执行，生成蓝图避免逐步token冗余
- **HuggingGPT**: 子任务路由到专业化模型

**Learning-Based Evolution（基于学习的进化）**：

*Policy Optimization（策略优化）*：
- **QLASS**: Q-Value critic指导搜索
- **ETO**: 试错偏好学习（DPO）
- **RLTR/Planner-R1**: 过程级奖励，对推理序列而不仅是最终结果提供反馈

*Memory & Skill Acquisition（记忆与技能获取）*：
- **VOYAGER**: 可重用技能库构建，避免重新规划
- **GAP**: 图分解与并行化识别

### 4.2 Multi-Agent Collaborative Efficiency（多智能体协作效率）

解决MAS中$O(N^2)$通信成本问题。

**Topological Efficiency and Sparsification（拓扑效率与稀疏化）**：
- **Chain-of-Agents**: 线性复杂度顺序上下文传递
- **MacNet**: DAG拓扑排序
- **AgentPrune/AgentDropout/SafeSieve**: 动态学习剪枝低效用边，推理时逐步稀疏化图
- **MARS/S²-MAD**: 仅观点分歧时触发辩论，消除直接点对点噪声

**Protocol and Context Optimization（协议与上下文优化）**：
- **CodeAgents**: 简洁伪代码编码推理
- **Smurfs**: 丢弃失败搜索分支防止上下文膨胀
- **Free-MAD/ConsensAgent**: 提示工程鼓励批判性推理加速收敛
- **SMAS**: 监督器早期终止冗余循环

**Distilling Coordination into Planning（协调蒸馏为规划）**：
- **MAGDI/SMAGDi**: 将复杂交互图或"Socratic"分解蒸馏到单学生模型
- **D&R**: 师生辩论生成DPO偏好树

---

## 五、Benchmarks（评测基准）

### 5.1 Memory Benchmarks

**Effectiveness（效果）**：
- QA数据集: HotpotQA, Natural Questions
- 交互式: GAIA, LoCoMo, LongMemEval

**Efficiency（效率）**：
- **Evo-Memory**: Step Efficiency（达到目标所需环境步数）
- **StoryBench**: Runtime Cost, Token Consumption
- **MemBench**: Read/Write Time（秒/记忆操作）

**四类效率指标**：
1. Token Consumption & API Cost
2. Time-based metrics (latency, runtime, retrieval time)
3. Resource-based (GPU memory)
4. Interaction-based (LLM calls数, reasoning steps)

### 5.2 Tool Learning Benchmarks

**Selection & Parameter Infilling**:
- SealTools, UltraTool, MetaTool, BFCL, API-Bank, NesTools, τ-Bench, ToolBench, MGToolBench, T-Eval, StableToolBench

**MCP (Model Context Protocol)**:
- MCP-RADAR: 工具选择效率、计算资源效率、执行速度
- MCP-Bench: LLM-as-Judge评分并行性和效率

**Agentic Tool Learning**:
- SimpleQA, BrowseComp, SealQA

### 5.3 Planning Benchmarks

**Effectiveness**:
- WebArena, WorkArena, Mind2Web, SWE-Bench, OSWorld

**Efficiency**:
- **GPTSwarm**: 成功率 vs 推理步数
- **AgentPrune**: 成功率 vs token消耗、时间成本
- **MCP-Bench**: Agent效率评分，最小化冗余工具调用

---

## 六、核心洞察与未来方向

### 6.1 关键权衡（Trade-offs）

1. **Compression vs Performance**: 过度压缩导致信息丢失，需要找到最优平衡点

2. **Online vs Offline Memory Management**: 
   - Online: 实时适应但增加延迟和成本
   - Offline: 最小化推理开销但适应较慢

3. **Test-Time Scaling Trade-off**: 虽然提高准确性，但仍是效果与效率的权衡

### 6.2 共享设计原则

论文总结了跨Memory/Tool/Planning三大组件的**通用高效原则**：

| 原则 | Memory | Tool | Planning |
|------|--------|------|----------|
| **Compression** | 上下文压缩、摘要 | 工具描述压缩 | 轨迹压缩、抽象 |
| **Selection** | 相关记忆检索 | 相关工具选择 | 自适应 deliberation |
| **Hierarchy** | 分层存储 | 分层工具组织 | 分层任务分解 |
| **Learning** | 可训练retriever | RL减少调用 | 策略优化、技能库 |
| **Caching/Reuse** | 计划缓存 | 可重用函数 | 可重用技能 |

### 6.3 Pareto Frontier 视角

论文提出用**Pareto前沿**（效果-成本权衡曲线）来刻画效率：
- 固定成本预算下比较效果
- 相当效果水平下比较成本

真正的效率提升是**将Pareto前沿向外推**，而非简单地在曲线上移动。

### 6.4 开放挑战

1. **动态预算分配**: 根据任务复杂度自适应调整资源
2. **跨组件协同优化**: Memory-Tool-Planning联合优化
3. **可学习的效率**: 让Agent自己学习何时"思考"何时"行动"
4. **多智能体协调的 scaling law**: 如何从$O(N^2)$降到$O(N)$或更好

---

## 七、总结

这篇综述的核心贡献是：

1. **系统化的taxonomy**: 将Efficient Agents分解为Memory、Tool Learning、Planning三个核心组件，每个组件再细分为具体技术方向

2. **统一的效率视角**: 提出Agent成本公式，明确效率不仅是模型压缩，更是系统层面的调用频率优化

3. **Pareto优化框架**: 强调效率是效果-成本的权衡，而非单维度优化

4. **丰富的技术图谱**: 涵盖了从2023-2025年的关键工作，提供了详细的机制对比和适用场景分析

对于实践者，这篇论文提供了**构建高效Agent的决策树**：根据任务特点（长上下文vs工具丰富vs多步推理）、资源约束（延迟vs成本vs准确率）选择合适的技术组合。

相关资源：
- 项目主页: https://efficient-agents.github.io/
- 代码仓库: https://github.com/yxf203/Awesome-Efficient-Agents