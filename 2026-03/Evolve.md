# 到底什么是 Evolve？——从第一性原理讲起

## 一句话概括

**Evolve = 让 LLM 当"变异算子"，自动写代码、跑评估、留优汰劣，循环迭代，最终"进化"出更好的算法/程序。**

---

## 第一性原理拆解

### 1. 传统 Genetic Programming (GP) 的局限

经典的 genetic programming 用**人工设计的变异算子**（crossover, mutation）来修改程序树：

$$\text{offspring} = \text{mutate}(\text{parent}, \text{random\_perturbation})$$

问题：变异是**盲目的、局部的**——只能做小改动（交换子树、改变常数），无法做语义级别的"创造性跳跃"。

### 2. Evolve 的核心洞见

**用 LLM 替代人工变异算子**。LLM 能理解代码语义，做"聪明的变异"——不仅是改参数，而是**重写算法结构**。

| 维度 | 传统 GP | Evolve |
|------|---------|--------|
| 变异算子 | 人工规则（随机） | LLM（语义感知） |
| 变异粒度 | 子树/节点级 | 整个 code block 级 |
| 能否发明新算法？ | ❌ 几乎不能 | ✅ 可以（如 random search → simulated annealing） |
| 搜索空间 | 语法树空间 | 自然语言+代码的联合空间 |

---

## 四大组件详解

```
┌─────────────────────────────────────────────────┐
│                   Controller (异步调度)            │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │  Prompt   │  │   LLM    │  │Evaluator │  │ Program  │
│  │ Sampler  │→ │ Ensemble │→ │  Pool    │→ │ Database │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘
│       ↑                                          │
│       └──────────── 反馈循环 ─────────────────────┘
└─────────────────────────────────────────────────┘
```

### ① Prompt Sampler（提示采样器）

**做什么**：从 Program Database 里选"精英父代"，构建上下文丰富的 prompt。

核心选择逻辑：

$$P(\text{select } p_i) \propto \exp\left(\frac{\text{score}_i}{\tau}\right)$$

其中 $\text{score}_i$ 是程序 $p_i$ 的评估分数，$\tau$ 是温度参数（exploitation_ratio 控制其等价物）。

**exploitation_ratio = 0.7** 意味着：70% 概率偏向选高分程序（exploitation），30% 探索多样性（exploration）。

Prompt 里包含：
- 问题描述
- 过去的程序代码 + 对应分数
- 指令："修改这个程序以获得更高分数"

### ② LLM Ensemble（语言模型集成）

**做什么**：多个 LLM 并行生成代码变体。

为什么要 ensemble？
- **不同模型有不同"变异风格"**：Gemini-Flash 偏快偏广，Claude-Sonnet 偏深偏精
- **速度 vs 质量平衡**：大部分 generation 用快模型（Gemini-Flash-2.0-lite），偶尔用强模型（Claude-Sonnet-3.7）做"大胆突变"

$$\text{ensemble output} = \bigcup_{m \in \mathcal{M}} \text{generate}_m(\text{prompt})$$

每个模型独立生成，产出的候选全部送入评估。

### ③ Evaluator Pool（评估器池）

**做什么**：自动运行程序，量化打分。

评估函数长这样：

```python
def evaluate(program_output) -> dict:
    return {
        "sum_of_radii": 2.634,   # 越大越好
        "valid": True,           # 是否满足约束
        "runtime_ms": 120        # 可作为多目标之一
    }
```

关键：**评估必须是自动化的、可编程的**。人工打分不行。

### ④ Program Database（程序数据库）

**做什么**：存所有历史程序 + 分数，相当于"基因库"。

使用 **Island Model**（岛屿模型）来维持多样性：

$$\text{Population} = \bigcup_{k=1}^{K} \text{Island}_k$$

- 每个 island 独立进化，定期 **migrate**（迁移）精英个体到其他 island
- 防止所有程序收敛到同一个局部最优
- 配置示例：5 个 islands，每 island ~100 个体，总 population = 500

---

## Evolve 到底"进化"了什么？——Circle Packing 的例子

这是最直观的演示。目标：把 26 个圆塞进一个 unit square，最大化半径之和。

### 进化轨迹

```
Generation   策略                          sum(r)
─────────────────────────────────────────────────
0            同心圆环放置                     1.87
10           六角密排                        2.18
100          网格交错排列                     2.32
~500         scipy.optimize SLSQP 求解器      2.634  ← 接近理论最优!
```

**关键观察**：Evolve 不仅"调参数"，它**发明了新算法**！

- 从"手动摆位置" → "用数学优化器求解" 这种跨越，是传统 GP 做不到的
- LLM 知道 scipy.optimize 的存在，能在合适时机"引入"这个工具

### 第二个例子：Random Search → Simulated Annealing

```
Generation   算法                    最优值
────────────────────────────────────────────
0            Random Search            -0.8xx
~50          梯度近似 + 局部搜索        -1.2xx
~200         Simulated Annealing      -1.519  ← 全局最优!
```

LLM 自主"重新发现"了 simulated annealing 的三个核心要素：
1. **局部扰动**：$x_{new} = x_{current} + \mathcal{U}(-\delta, \delta)$
2. **Metropolis 准则**：接受劣解的概率 $P = \exp\left(\frac{f_{current} - f_{new}}{T}\right)$
3. **冷却调度**：$T_{k+1} = \alpha \cdot T_k$，$\alpha = 0.99$

---

## 为什么 Evolve 能 Work？——理论直觉

1. **LLM 是"结构化变异"**：它不是随机扰动，而是基于海量代码语料的**分布采样**，相当于一个极度丰富的"先验"
2. **评估函数是"适应度"**：selection pressure 引导搜索方向
3. **Island + exploitation ratio 是"多样性维护"**：防止早熟收敛
4. **整个 codebase 级别进化**：不只是调超参，是重写算法本身

用进化计算的语言说：

$$\text{Evolve} = (\mu + \lambda)\text{-ES with LLM as mutation operator}$$

其中 $\mu$ = 父代种群大小，$\lambda$ = 每代子代数，ES = Evolution Strategy。LLM 替代了传统的 Gaussian mutation $\mathcal{N}(0, \sigma^2)$，变成了**语义空间的条件生成**。

---

## 一句话再总结

> **Evolve = 用 LLM 做"聪明的基因突变"，用自动评估做"自然选择"，用 island 模型维护"物种多样性"，循环迭代直到收敛。本质上是用 LLM 的代码先验知识大幅压缩了搜索空间，让进化从"盲搜"变成"有方向的创造性搜索"。**

---

## 参考链接

- **OpenEvolve GitHub**: [https://github.com/codelion/openevolve](https://github.com/codelion/openevolve)
- **AlphaEvolve 原始论文**: [Google DeepMind Blog](https://deepmind.google/research/publications/alphaevolve/)
- **Circle Packing 示例**: [https://github.com/codelion/openevolve/tree/main/examples/circle_packing](https://github.com/codelion/openevolve/tree/main/examples/circle_packing)
- **Function Minimization 示例**: [https://github.com/codelion/openevolve/tree/main/examples/function_minimization](https://github.com/codelion/openevolve/tree/main/examples/function_minimization)
- **Evolution Strategies 基础**: [https://arxiv.org/abs/1912.01850](https://arxiv.org/abs/1912.01850)
- **LLM as Optimizer (OPRO)**: [https://arxiv.org/abs/2309.03409](https://arxiv.org/abs/2309.03409)