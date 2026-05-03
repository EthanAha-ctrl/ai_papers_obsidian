# FLAN: Fine-tuned LAnguage Net 深度解析

## 1. 核心直觉：从第一性原理出发

### 1.1 问题的本质

大语言模型（LLM）在 pre-training 阶段通过 next-token prediction 学习了海量世界知识，但这些知识以一种**隐式分布**存储在参数中。关键问题是：

> **如何将预训练模型中的隐式知识"解锁"并迁移到具体下游任务？**

传统范式有两种路径：

| 范式 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| **Fine-tuning** | 在标注数据上更新权重 | 任务性能强 | 每个任务需存储独立权重、需要大量标注数据 |
| **Prompting (zero/few-shot)** | 设计 prompt 让模型续写 | 无需训练 | 需要精心设计 prompt，不直观，部分任务效果差 |

FLAN 提出了**第三条路径**：**Instruction Tuning** —— 不是为了某个特定任务而微调，而是让模型学会"遵循指令"这个通用能力本身。

### 1.2 核心洞察的数学表述

从贝叶斯视角理解：

- 传统 fine-tuning 学习的是：$P(y | x; \theta_{\text{task}})$，即针对特定任务的条件分布
- Instruction tuning 学习的是：$P(y | \text{instruction}, x; \theta_{\text{FLAN}})$，即**以指令为条件**的更通用分布

关键变量：
- $x$：输入文本
- $y$：输出标签/文本
- $\text{instruction}$：自然语言任务描述（如 "Translate this sentence to Danish"）
- $\theta$：模型参数

FLAN 的假设是：

$$P(y | \text{instruction}_{\text{unseen}}, x; \theta_{\text{FLAN}}) \approx P(y | \text{instruction}_{\text{unseen}}, x; \theta^{*})$$

其中 $\theta^{*}$ 是一个能泛化到**未见指令**的理想参数。即：只要训练中见过**足够多样的指令类型**，模型就能对**新的指令类型**产生正确的行为。

---

## 2. 方法论详解

### 2.1 Instruction Tuning Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Existing NLP   │────▶│  Template        │────▶│  Instructional  │
│  Datasets       │     │  Transformation  │     │  Training Data  │
│  (e.g., SNLI,   │     │  (Multiple       │     │  (Instruction + │
│   SST-2, SQuAD) │     │   Templates per  │     │   Input → Output)│
│                 │     │   Dataset)        │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │  Fine-tune LLM  │
                                                │  (e.g., 137B    │
                                                │   LaMDA-PT)     │
                                                └─────────────────┘
```

### 2.2 Template Transformation（模板转换）

这是 FLAN 方法中最精巧的设计之一。每个数据集被转化为**指令格式**：

**原始数据**：
- Premise: "A man is playing guitar"
- Hypothesis: "A man is making music"
- Label: Entailment

**转换后（多种模板）**：

| Template ID | Instruction Format |
|-------------|-------------------|
| T1 | "Given the premise 'A man is playing guitar', does it entail the hypothesis 'A man is making music'? Yes or No?" |
| T2 | "Is the hypothesis 'A man is making music' entailed by the premise 'A man is playing guitar'?" |
| T3 | "Read the following and determine if the second sentence is entailed by the first. Sentence 1: A man is playing guitar. Sentence 2: A man is making music." |

每个数据集使用**约 10 个模板**，增加指令的多样性，防止模型过拟合到特定措辞。

**数学表达**：

设数据集 $\mathcal{D}_k$ 的第 $i$ 个样本为 $(x_i, y_i)$，模板集合为 $\mathcal{T}_k = \{t_1, t_2, ..., t_M\}$，则转换后的训练样本为：

$$(\tilde{x}_i, \tilde{y}_i) = (t_m(x_i), f(y_i)), \quad m \sim \text{Uniform}(\{1,...,M\})$$

其中 $t_m$ 是第 $m$ 个模板函数，$f$ 是标签到目标文本的映射（如 "entailment" → "Yes"）。

### 2.3 Task Clustering（任务聚类）与 Hold-out 策略

这是 FLAN 实验设计中最关键的创新——为了公平评估**零样本泛化能力**，需要确保评估任务与训练任务**在类型上不重叠**：

| Cluster | 包含的数据集 |
|---------|-------------|
| **NLI** | ANLI R1-R3, CB, RTE |
| **Reading Comprehension** | BoolQ, MultiRC, OpenbookQA |
| **Closed-book QA** | ARC, NQ, TriviaQA |
| **Translation** | WMT, ... |
| **Sentiment** | SST-2, ... |
| **...** | ... |

评估某个 cluster 时，**整个 cluster 的所有数据集都从训练中移除**。这确保了模型不是在测试"同类型任务的迁移"，而是在测试**跨任务类型的泛化**。

用公式表述：

$$\mathcal{D}_{\text{train}} = \bigcup_{k \notin C_{\text{eval}}} \mathcal{D}_k$$

$$\mathcal{D}_{\text{eval}} = \bigcup_{k \in C_{\text{eval}}} \mathcal{D}_k$$

其中 $C_{\text{eval}}$ 是被 hold-out 的 task cluster。

---

## 3. 实验结果详解

### 3.1 核心结果

FLAN 在 **25 个任务**上评估，核心发现：

- **25 个任务中 21 个**优于 zero-shot prompting
- **25 个任务中 20 个**优于 zero-shot GPT-3
- 部分任务甚至**超过 few-shot GPT-3**

具体数据（以 NLI、Reading Comprehension、Closed-book QA 为例）：

| 任务 Cluster | 数据集 | Zero-shot GPT-3 (175B) | Few-shot GPT-3 | FLAN (137B) |
|-------------|--------|----------------------|----------------|-------------|
| **NLI** | ANLI R1 | ~33% | ~38% | **~49%** |
| | ANLI R2 | ~33% | ~35% | **~40%** |
| | CB | ~41% | ~49% | **~75%** |
| | RTE | ~51% | ~63% | **~74%** |
| **Reading Comp.** | BoolQ | ~58% | ~61% | **~73%** |
| | MultiRC | ~5% | ~24% | **~33%** |
| **Closed-book QA** | ARC | ~23% | ~24% | **~35%** |
| | NQ | ~6% | ~9% | **~14%** |

> 注：具体数值来自论文，上表为近似值，精确数据请参考原论文 Table 2-4。

### 3.2 Scale 依赖性——最关键的发现

FLAN 发现了一个**涌现现象（Emergent Ability）**：

$$\text{Instruction Tuning 效果} = \begin{cases} \text{负效应（性能下降）} & \text{if } |\theta| < \theta_{\text{threshold}} \\ \text{正效应（性能提升）} & \text{if } |\theta| \geq \theta_{\text{threshold}} \end{cases}$$

具体来说：
- **8B 参数模型**：instruction tuning 后性能**下降**
- **62B 参数模型**：instruction tuning 后略有提升
- **137B 参数模型**：instruction tuning 后**显著提升**

**直觉解释**：

小模型的参数容量（capacity）有限，已经在 pre-training 阶段学到的知识勉强够用。Instruction tuning 相当于用有限的参数去"记忆"多种指令格式，反而**挤占**了原有知识的容量——这是一种**catastrophic forgetting** 的变体。只有当模型足够大时，才有**冗余容量**来学习"遵循指令"这一元能力，而不损失原有知识。

这类似于一个类比：
- 🧒 小学生学了很多学科基础知识，此时让他同时学"如何读题目"，反而会让他困惑
- 🎓 大学生已有丰富知识储备，学会"如何理解考题要求"后反而能更好发挥

---

## 4. 与相关方法的对比

### 4.1 FLAN vs. Prompt Engineering

| 维度 | Prompt Engineering | FLAN |
|------|-------------------|------|
| 交互方式 | 模型需适应人的 prompt 风格 | 模型学会适应人的自然语言指令 |
| 适用性 | 需要领域专家反复调试 | 任何人都能用自然语言下达指令 |
| 成本 | 推理时无额外成本，但设计成本高 | 一次 training，多次 zero-shot 使用 |
| 泛化性 | 对特定 prompt 格式敏感 | 对指令变体鲁棒 |

### 4.2 FLAN vs. T0

同期工作 **T0**（Sanh et al., 2021）也提出了类似思路：
- T0 使用 T5 架构，FLAN 使用 LaMDA-PT
- T0 使用了更多的数据集（约 60 个），FLAN 约 62 个
- 两者核心结论一致：**instruction tuning 能显著提升 zero-shot 性能**

### 4.3 FLAN 的演进路线

```
FLAN (2022)
  │
  ├─▶ FLAN-T5 / FLAN-PaLM (Chung et al., 2022)
  │     • 增加更多 task 数量（~1800+）
  │     • 加入 Chain-of-Thought 数据
  │     • 提出 "mixture of instructions" 策略
  │
  ├─▶ FLAN v2 / FLAN-T5-XXL
  │     • 进一步扩大规模
  │     • 加入 CoT + 直接回答的混合训练
  │
  └─▶ 影响：InstructGPT / ChatGPT 的 RLHF 流程
        • FLAN 是 "SFT 阶段" 的先驱
        • InstructGPT = FLAN-style SFT + RLHF
```

---

## 5. 训练细节

### 5.1 基座模型

FLAN 使用 **LaMDA-PT (137B)** 作为基座模型：
- 参数量：137B
- Pre-training 数据：Web 文本 + 代码
- Pre-training 目标：Causal language modeling

### 5.2 Instruction Tuning 训练配置

| 配置项 | 值 |
|--------|-----|
| 优化器 | Adafactor |
| Learning rate | $3 \times 10^{-5}$ |
| LR schedule | Constant |
| Batch size | 8192 tokens |
| 训练步数 | ~12,000 steps |
| 总 token 数 | ~98M |
| 训练数据 | 62 个数据集 |
| 每个数据集模板数 | ~10 |

**关键观察**：训练只需 ~98M tokens，相较于 pre-training 的数万亿 tokens，确实是"dessert to the main course"——计算量不到 pre-training 的 **0.001%**。

### 5.3 损失函数

标准的 next-token prediction loss：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, \text{instruction}, x; \theta)$$

其中 $y_t$ 是目标序列的第 $t$ 个 token，$T$ 是目标序列长度。

---

## 6. 深层原理：为什么 Instruction Tuning 有效？

### 6.1 任务多样性假设

设指令分布为 $P(\text{instruction})$。Instruction tuning 本质上是在最大化：

$$\mathbb{E}_{(\text{inst}, x, y) \sim \mathcal{D}_{\text{inst}}} [\log P(y | \text{inst}, x; \theta)]$$

当指令类型足够多样时（即 $\mathcal{D}_{\text{inst}}$ 覆盖了足够广的任务空间），模型学到的不是某个特定任务的解法，而是**"理解指令 → 执行对应操作"**这一元策略。

### 6.2 与 Multi-task Learning 的关系

Instruction tuning 可以视为一种**结构化的 multi-task learning**：

- 传统 multi-task learning：共享底层的参数，不同任务用不同的输出头
- Instruction tuning：**用自然语言统一了不同任务的头**，消除了对任务特定架构的依赖

这带来的好处是：新任务不需要新的输出头，只需要**用自然语言描述任务**即可。

### 6.3 Distribution Shift 的视角

Pre-training 时，模型看到的是"续写文本"的分布：

$$P_{\text{pretrain}}(x)$$

而下游任务需要的是"根据指令回答"的分布：

$$P_{\text{instruct}}(y | \text{instruction}, x)$$

这两个分布之间存在 **distribution shift**。Instruction tuning 的作用就是用少量计算弥补这个 shift，将模型从"续写者"校准为"回答者"。

---

## 7. 局限性与后续改进

### 7.1 FLAN 的局限

1. **Scale 依赖性**：小模型无法受益，限制了应用范围
2. **人工模板设计**：模板仍然需要人工设计，存在主观性
3. **任务覆盖**：62 个数据集虽多，但相比真实世界的任务空间仍然有限
4. **没有 CoT**：原始 FLAN 没有引入 chain-of-thought 数据，推理能力有限

### 7.2 FLAN-T5/PaLM 的改进

Chung et al. (2022) 在 "[Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)" 中做了重大改进：

| 改进 | 详情 |
|------|------|
| **更多任务** | 62 → 1836 tasks |
| **CoT 数据** | 加入 9 个 CoT 数据集 |
| **混合策略** | CoT 和 non-CoT 数据混合训练 |
| **多模型验证** | T5, PaLM 等多个架构 |

关键发现：

$$\text{性能} \propto f(\text{任务数量}, \text{模型规模}, \text{CoT数据比例})$$

三者协同提升效果最为显著。

---

## 8. FLAN 的历史意义

FLAN 是现代 LLM 训练范式的**奠基性工作**之一，它的影响路径：

```
FLAN (2022) ──── Instruction Tuning 的概念验证
    │
    ▼
InstructGPT (2022) ──── SFT + RLHF
    │
    ▼
ChatGPT (2022) ──── 对话式 AI 的商业突破
    │
    ▼
GPT-4 / Claude / Gemini ──── 现代指令遵循模型
```

核心贡献：**证明了"教会模型遵循指令"比"教会模型某个特定任务"更有价值**——这是一次从 **task-specific** 到 **task-agnostic** 的范式转换。

---

## 9. 参考

- 📄 **FLAN 原论文**：[Fine-tuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (Wei et al., 2022)
- 📄 **FLAN-T5/PaLM**：[Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) (Chung et al., 2022)
- 📄 **T0**：[Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207) (Sanh et al., 2021)
- 📄 **InstructGPT**：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022)
- 📄 **Emergent Abilities**：[Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) (Wei et al., 2022)
- 🔗 **Google Blog**：[https://ai.googleblog.com/2022/02/finetuned-language-models-are-zero-shot.html](https://ai.googleblog.com/2022/02/finetuned-language-models-are-zero-shot.html)