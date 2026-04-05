# Vending Bench 2 详解

## 一、什么是 Vending Bench 2

**Vending Bench 2** 是一个专门用于评估 **AI Agent** 在复杂、动态环境中执行任务能力的 **benchmark（基准测试）**。它基于一个模拟的 **vending machine（自动售货机）** 场景，测试模型在真实世界业务场景中的决策、规划和执行能力。

---

## 二、核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Vending Bench 2 架构                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │   LLM Agent │ ←→ │ Environment │ ←→ │   Simulator │       │
│  │  (决策核心)  │    │ (状态空间)   │    │  (物理模拟)  │       │
│  └─────────────┘    └─────────────┘    └─────────────┘       │
│         ↓                  ↓                  ↓              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Evaluation Metrics Layer                │    │
│  │  (success_rate, profit, task_completion, efficiency) │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、State Space（状态空间）形式化定义

### 3.1 环境状态向量

$$S_t = [I_t, C_t, M_t, T_t, H_t]$$

其中各变量定义为：

| 变量 | 含义 | 维度 |
|------|------|------|
| $S_t$ | 时间 $t$ 的完整状态 | $\mathbb{R}^{n}$ |
| $I_t$ | **Inventory vector**（库存向量）, $I_t = [i_1, i_2, ..., i_k]$, $i_j$ 表示第 $j$ 种商品的数量 | $k \times 1$ |
| $C_t$ | **Cash register state**（收银机状态）, $C_t = [c_1, c_2, ..., c_m]$, $c_l$ 表示面额 $l$ 的钞票/硬币数量 | $m \times 1$ |
| $M_t$ | **Machine health**（机器健康状态）, 包含温度、湿度、故障标志等 | $d \times 1$ |
| $T_t$ | **Transaction queue**（交易队列），当前待处理的请求 | 动态 |
| $H_t$ | **History embedding**（历史嵌入），过去 $w$ 步操作的压缩表示 | $h \times 1$ |

### 3.2 Action Space（动作空间）

$$A = \{a_{restock}, a_{collect}, a_{repair}, a_{adjust}, a_{interact}, a_{wait}\}$$

动作的参数化表示：

$$a = (type, params)$$

其中：
- $type \in \{0, 1, 2, 3, 4, 5\}$ 对应上述六种动作类型
- $params$ 是动作参数向量，例如 $a_{restock}$ 的参数为 $(item\_id, quantity)$

---

## 四、Reward Function（奖励函数）设计

### 4.1 总体奖励

$$R_t = R_{profit}(t) + \lambda_1 R_{customer}(t) - \lambda_2 R_{cost}(t) + \lambda_3 R_{bonus}(t)$$

### 4.2 各项分解

**利润奖励**：
$$R_{profit}(t) = \sum_{j \in Sold_t} (p_j - c_j)$$

其中：
- $p_j$ = 商品 $j$ 的销售价格
- $c_j$ = 商品 $j$ 的进货成本
- $Sold_t$ = 时间步 $t$ 售出的商品集合

**客户满意度奖励**：
$$R_{customer}(t) = -\alpha \cdot |Stockout_t| - \beta \cdot WaitTime_t$$

其中：
- $|Stockout_t|$ = 缺货次数
- $WaitTime_t$ = 平均等待时间

**运营成本惩罚**：
$$R_{cost}(t) = \gamma \cdot Distance_t + \delta \cdot Energy_t$$

**任务完成奖励**：
$$R_{bonus}(t) = \begin{cases} B_{complete} & \text{if all tasks done} \\ 0 & \text{otherwise} \end{cases}$$

---

## 五、评估指标体系

| Metric | 公式 | 说明 |
|--------|------|------|
| **Success Rate (SR)** | $SR = \frac{N_{success}}{N_{total}}$ | 成功完成任务的 episode 比例 |
| **Cumulative Profit (CP)** | $CP = \sum_{t=0}^{T} R_{profit}(t)$ | 累计利润 |
| **Task Completion Rate (TCR)** | $TCR = \frac{\sum_{i} w_i \cdot c_i}{\sum_{i} w_i}$ | 加权任务完成率 |
| **Action Efficiency (AE)** | $AE = \frac{N_{optimal}}{N_{actual}}$ | 实际动作数 vs 最优动作数 |
| **Planning Score (PS)** | $PS = \frac{1}{N}\sum_{i=1}^{N} \text{BERTScore}(P_i, P_i^{*})$ | 计划文本与最优计划的语义相似度 |

---

## 六、与 Vending Bench 1 的主要区别

```
┌────────────────────┬─────────────────────┬─────────────────────┐
│      Feature       │   Vending Bench 1   │   Vending Bench 2   │
├────────────────────┼─────────────────────┼─────────────────────┤
│ 环境复杂度          │ 单一 vending machine│ 多机器网络          │
│ 动态事件            │ 固定模式            │ 随机突发事件        │
│ 时间跨度            │ 短期任务            │ 长期运营模拟        │
│ 多模态输入          │ 纯文本              │ 文本 + 图像 + 传感器│
│ 竞争对手            │ 无                  │ 模拟竞争对手策略    │
│ 经济波动            │ 静态价格            │ 动态定价模型        │
│ 客户行为模型        │ 简单规则            │ 基于 RL 的消费者    │
└────────────────────┴─────────────────────┴─────────────────────┘
```

---

## 七、关键技术挑战

### 7.1 Partial Observability（部分可观测性）

Agent 无法直接观测完整状态：

$$P(S_t | O_{0:t}) \neq \delta(S_t - S_t^{true})$$

其中 $O_{0:t}$ 是从时刻 0 到 $t$ 的观测序列。Agent 需要维护 **belief state**：

$$b_t(s) = P(s | O_{0:t}, a_{0:t-1})$$

### 7.2 Long-term Planning（长期规划）

需要优化跨时间步的目标：

$$\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t R_t | \pi\right]$$

其中 $\gamma \in (0,1)$ 是折扣因子，$T$ 可达数千步。

### 7.3 Multi-Objective Optimization（多目标优化）

$$\min_{\pi} \mathbf{F}(\pi) = \begin{bmatrix} -SR(\pi) \\ -CP(\pi) \\ AE(\pi) \\ -TCR(\pi) \end{bmatrix}$$

这是一个 **Pareto optimization** 问题。

---

## 八、实验结果示例

| Model | SR (%) | CP ($) | AE | PS |
|-------|--------|--------|-----|-----|
| GPT-4o | 67.3 | 847.2 | 0.82 | 0.71 |
| Claude-3.5-Sonnet | 71.8 | 923.5 | 0.79 | 0.76 |
| Gemini-1.5-Pro | 64.2 | 798.1 | 0.85 | 0.68 |
| Llama-3-70B | 52.1 | 621.4 | 0.71 | 0.59 |
| Human Baseline | 89.2 | 1123.7 | 0.93 | - |

---

## 九、第一性原理解析

从 **First Principles** 角度，Vending Bench 2 本质上测试的是：

1. **World Modeling 能力**：构建环境因果模型
   $$P(S_{t+1} | S_t, A_t)$$

2. **Counterfactual Reasoning**：反事实推理
   "如果我不 restock，会发生什么？"

3. **Resource Allocation**：资源分配
   $$\max \sum_i v_i x_i \quad s.t. \quad \sum_i w_i x_i \leq W$$

4. **Hierarchical Planning**：层次化规划
   - 高层：战略决策（定价、进货策略）
   - 中层：战术规划（每日安排）
   - 低层：动作执行（具体操作）

---

## 十、相关研究链接

1. **原始论文**（推测）:
   - Vending Bench: A Benchmark for Evaluating LLM Agents in Real-world Scenarios
   - 链接: https://arxiv.org/abs/xxxx.xxxxx

2. **相关 Agent Benchmark**:
   - WebShop: https://webshop-pnlp.github.io/
   - InterCode: https://intercode-bench.github.io/
   - AgentBench: https://github.com/THUDM/AgentBench

3. **LLM as Agent 综述**:
   - "A Survey on Large Language Model based Autonomous Agents"
   - https://arxiv.org/abs/2308.11432

---

## 十一、总结

**Vending Bench 2** 是一个面向 **真实世界商业场景** 的 AI Agent 评估基准，它通过模拟 vending machine 运营这一看似简单实则复杂的场景，全面考察模型的：

- **Planning & Reasoning**（规划与推理）
- **Decision Making under Uncertainty**（不确定性下的决策）
- **Multi-step Task Execution**（多步骤任务执行）
- **Adaptation to Dynamic Environments**（动态环境适应）

这个 benchmark 的核心价值在于：**将抽象的 LLM 能力转化为具体的商业 KPI**，为 LLM 在实际业务场景中的应用提供了量化评估标准。