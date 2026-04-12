

让我搜索一下这个公司的信息。














好的，我已经收集了大量信息。下面给你一个全面且深入的解析。

---

# AIvilization (aivilization.ai / aivilization.cn) —— 全面解析

## 一、公司概述

**AIvilization** 是由 **HKUST（Hong Kong University of Science and Technology，香港科技大学）** 的 **Professor Pan Hui** 团队主导的一个大规模 **AI Multi-Agent Social Simulation（AI 多智能体社会模拟）** 项目/平台。它的核心定位是：

> 构建一个 **digital sandbox（数字沙盒）**，让数以万计甚至 **100,000 个 AI Agent** 在一个资源受限的模拟经济中自由发展、交互、共存，模拟未来 **human-AI cohabitation（人-AI 共栖）** 和 **civilizational evolution（文明演化）** 的可能形态。

简单来说，这是一个"**AI 版 SimCity + 社会实验室**"——但每个 NPC 都是一个由 **LLM（Large Language Model）** 驱动的 autonomous agent，拥有独立的 personality、memory、goals 和 adaptive behavior。

**参考链接：**
- 官网: https://aivilization.cn/
- 论文: https://arxiv.org/abs/2602.10429
- HKUST 新闻: https://hkust.edu.hk/news/hkust-launches-worlds-largest-ai-powered-educational-sandbox-game-advancing-ai-literacy-and
- The Decoder 报道: https://the-decoder.com/aivilization-experiment-lets-over-22000-ai-agents-model-what-future-societies-could-become/
- Medium 深度分析: https://kellyontech.medium.com/aivilization-100-000-ai-agents-rehearse-the-rules-of-the-mirrorworld-4c3d7f60381b

---

## 二、核心技术架构（来自论文 AIvilization v0, arXiv: 2602.10429）

论文标题：**"AIvilization v0: Toward Large-Scale Artificial Social Simulation with a Unified Agent Architecture and Adaptive Agent Profiles"**

### 2.1 系统的两大支柱

整个系统耦合了两个核心子系统：

| 子系统 | 说明 |
|---|---|
| **Resource-Constrained Sandbox Economy** | 一个有稀缺资源（food, shelter, tools, currency 等）的模拟经济世界，agent 必须在资源有限的约束下做出决策 |
| **Unified LLM-Agent Architecture** | 所有 agent 共用一个统一的 LLM-driven 认知架构，但通过 adaptive profile 实现个性差异 |

### 2.2 核心技术创新 #1：Hierarchical Branch-Thinking Planner（层级分支思维规划器）

这是解决 **goal stability（目标稳定性）** 与 **reactive correctness（反应正确性）** 之间张力的关键机制。

**问题直觉：**
- 如果一个 agent 太坚持长期目标 → 它会忽略环境突变（比如突然下雨、被攻击）
- 如果一个 agent 太响应短期刺激 → 它会丢失长期规划（像一只永远追逐最新刺激的松鼠）

**解决方案：** Hierarchical Branch-Thinking Planner 将 agent 的 life goals 分解为 **multi-level sub-goals tree（多层子目标树）**：

```
Life Goal (Lₗᵢfₑ)
  ├── Long-term Goal G₁ (e.g., become a merchant)
  │     ├── Mid-term Sub-goal G₁₁ (e.g., accumulate 100 gold)
  │     │     ├── Short-term Action A₁₁₁ (e.g., gather wood today)
  │     │     └── Short-term Action A₁₁₂ (e.g., trade wood for gold)
  │     └── Mid-term Sub-goal G₁₂ (e.g., build a shop)
  └── Long-term Goal G₂ (e.g., form an alliance)
        └── ...
```

这里的"**branch-thinking**"意味着在每个决策节点，planner 会评估多个可能的分支路径，并根据当前环境上下文选择最优分支，同时保留其他分支作为 fallback。

**关键公式（概念性）：**

$$a_t = \arg\max_{a \in \mathcal{A}} \sum_{k=1}^{K} w_k \cdot Q_k(s_t, a | G_k)$$

其中：
- $a_t$ = 在 time step $t$ 选择的 action
- $\mathcal{A}$ = 可用 action 集合
- $K$ = 当前活跃的 sub-goal 数量
- $w_k$ = 第 $k$ 个 sub-goal 的权重（由 urgency 和 importance 决定）
- $Q_k(s_t, a | G_k)$ = 在 state $s_t$ 下执行 action $a$ 对达成 sub-goal $G_k$ 的预期效用
- $s_t$ = 当前世界状态

这本质上是一个 **multi-objective decision making** 框架——agent 同时追求多个层级的目标，通过加权求和来 resolve conflicts。

### 2.3 核心技术创新 #2：Profile-Conditioned Action Selector（基于 Profile 的行为选择器）

每个 agent 有一个 **Adaptive Agent Profile**，包括：

- **Personality traits**（e.g., Big Five / MBTI parameters）
- **Values and beliefs**（e.g., 合作倾向 vs. 竞争倾向）
- **Skills and knowledge**
- **Memory**（episodic memory + semantic memory）
- **Social relationships**（trust graph）

这个 profile 不是静态的——它会随着 agent 的经历 **动态演化**（这就是"adaptive"的含义）。

**Action selection 的直觉：** 即使两个 agent 面对完全相同的 state $s_t$ 和相同的 goal $G$，因为他们的 profile $\mathcal{P}$ 不同，他们会选择不同的 action：

$$P(a | s_t, G, \mathcal{P}_i) \neq P(a | s_t, G, \mathcal{P}_j) \quad \text{当} \quad \mathcal{P}_i \neq \mathcal{P}_j$$

例如：一个 trait = "aggressive" 的 agent 面对资源争夺可能选择 fight，而一个 trait = "diplomatic" 的 agent 可能选择 negotiate。

### 2.4 Unified Agent Architecture（统一 Agent 架构）

每个 agent 的认知循环大致如下：

```
┌────────────────────────────────────┐
│          Perception Module          │
│  (观察环境、接收消息、感知资源)        │
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│          Memory System              │
│  ┌──────────┐  ┌───────────────┐   │
│  │ Episodic │  │   Semantic    │   │
│  │ Memory   │  │   Memory      │   │
│  └──────────┘  └───────────────┘   │
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│   Hierarchical Branch-Thinking     │
│          Planner                    │
│   (分解目标 → 评估分支 → 规划)       │
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│   Profile-Conditioned Action       │
│          Selector                   │
│   (基于 personality 选择行为)        │
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│          Action Execution           │
│  (move, talk, trade, build, etc.)  │
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│          Reflection & Learning      │
│  (更新 memory、调整 profile)         │
└────────────────────────────────────┘
```

这个架构与 **Stanford 的 Generative Agents** (Park et al., 2023) 有明显的继承关系，但关键区别是：
1. **规模大几个数量级**（Stanford 是 25 个 agent，AIvilization 是 22,000 ~ 100,000 个）
2. 引入了 **resource scarcity economy**（Stanford 没有经济系统）
3. **Profile 是 adaptive 的**（Stanford 的 agent persona 是固定的）
4. **Hierarchical planning** 替代了 Stanford 的简单 plan-and-execute loop

---

## 三、沙盒经济系统（Sandbox Economy）

这是让 AIvilization 从一个"AI 聊天 demo"变成"**文明模拟器**"的关键设计。

### 3.1 Resource Scarcity（资源稀缺性）

世界中的资源是 **有限的**：
- **Food** → 需要种植、采集或交易
- **Materials** → 需要开采或制作
- **Shelter** → 需要建造
- **Currency** → 通过劳动或交易获得

这遵循的是 **第一性原理**：任何真实文明的核心驱动力都是 **scarcity**（稀缺性）。正是资源稀缺才催生了 trade（交易）、specialization（专业化分工）、governance（治理）、conflict（冲突）和 cooperation（合作）。

### 3.2 Emergent Social Phenomena（涌现的社会现象）

在实验中，研究者观察到 agent 自发涌现了以下社会行为：
- **Division of labor**（分工）
- **Market formation**（市场形成）
- **Social hierarchy**（社会等级）
- **Alliance / coalition building**（联盟构建）
- **Norm emergence**（规范涌现）——agent 自发形成了类似"法律"的行为规则
- **Conflict and resolution**（冲突与解决）

这是一个 **bottom-up emergence** 的过程——没有人预先编程"建立市场"或"形成政府"，这些 macro-level patterns 完全从 micro-level agent interactions 中涌现。

---

## 四、项目定位与用途

### 4.1 Citizen Science（公民科学）
平台以 **游戏形式** 开放给公众，玩家可以：
- 创建自己的 **digital twin**（数字孪生），设置 personality、values、background
- 观察自己的 AI 代理如何在社会中行动
- 收集的数据用于 AI 研究

### 4.2 AI Literacy Education（AI 素养教育）
目标是让普通人理解：
- AI Agent 如何做决策
- Multi-Agent System 如何产生 emergent behavior
- AI alignment 和 AI safety 的挑战

### 4.3 Research Platform（研究平台）
为学术界提供大规模 **social simulation testbed**，可以研究：
- **AI alignment**：agent 的行为是否符合人类价值观？
- **Governance mechanisms**：什么样的治理机制能在 AI 社会中有效运作？
- **Economic dynamics**：AI agent 驱动的经济系统会如何演化？
- **Social network formation**：信任网络如何自发形成？

---

## 五、团队与背景

| 角色 | 人物 |
|---|---|
| **项目负责人** | **Professor Pan Hui** (HKUST, 计算机科学系, 同时也是 MetaHKUST 项目的领导者) |
| **统计/实验设计** | **Professor Kani Chen** (HKUST, 数学系 & 工业工程与决策分析系) |
| **协作方** | HKUST Entrepreneurship Center, Data & AI Literacy Association 等 |

Pan Hui 教授本身在 **mobile computing、social networks、XR (Extended Reality)** 领域有深厚背景，有 Wikipedia 条目 (https://en.wikipedia.org/wiki/Pan_Hui)。

---

## 六、与相关工作的比较

| 项目 | Agent 数量 | 经济系统 | Adaptive Profile | 公开部署 |
|---|---|---|---|---|
| **Stanford Generative Agents** (2023) | 25 | ❌ | ❌ (固定) | ❌ (研究 demo) |
| **Smallville** (Park et al.) | 25 | ❌ | ❌ | ❌ |
| **CAMEL** (Li et al.) | 2 | ❌ | ❌ | ❌ |
| **MetaGPT** | ~10 | ❌ (软件公司模拟) | 部分 | ❌ |
| **AIvilization v0** | **22,000 ~ 100,000** | ✅ (resource-constrained) | ✅ (adaptive) | ✅ (公开游戏) |

这个规模差异是 **质的飞跃**——从 25 个 agent 到 100,000 个 agent，不仅仅是"更多"的问题，而是带来了 **phase transition（相变）**：小规模下不会出现的 macro-level social structures（市场、政府、文化）在大规模下才能涌现。

---

## 七、第一性原理理解

为什么要做这样的项目？从第一性原理思考：

1. **我们即将进入 human-AI coexistence 时代** → 我们需要一个"飞行模拟器"来测试各种可能的未来场景，而不是等到现实中出问题
2. **Complex adaptive systems 不可预测** → 你无法通过分析单个 agent 的行为来推测 100,000 个 agent 的集体行为（这就是 emergence 的本质）→ 唯一的方法就是 **模拟**
3. **LLM 让 believable agent 成为可能** → 之前的 agent-based simulation (ABM) 用的是简单规则（如 Schelling segregation model），行为非常机械；LLM 使 agent 能够用自然语言推理、对话、协商，行为更 realistic
4. **资源稀缺是文明的第一驱动力** → 加入经济系统让模拟从"社交实验"升级为"文明实验"

**核心 insight：** AIvilization 本质上是在问一个深刻的问题——**如果 AI Agent 拥有自主性和约束条件，它们会自发构建出什么样的"文明"？** 这个"文明"与人类文明有何异同？这对我们理解 AI alignment 有什么启示？

---

## 八、总结

**AIvilization** 是一个由 HKUST 主导的 **世界最大规模 AI 社会模拟平台**，核心特点包括：

- 🎯 **100,000 个 LLM-driven Agent** 在一个 resource-constrained sandbox 中自由演化
- 🏗️ 基于 **Hierarchical Branch-Thinking Planner + Adaptive Agent Profiles** 的统一 Agent 架构
- 🌍 观察 **emergent social phenomena**（分工、市场、治理、文化）
- 🎮 以 **游戏形式** 向公众开放，兼具 citizen science 和 AI literacy 教育功能
- 📄 有严谨的学术论文支撑 (arXiv: 2602.10429)

它本质上是人类为了 **预演 human-AI 共存未来** 而建造的一个"**数字文明实验室**"。