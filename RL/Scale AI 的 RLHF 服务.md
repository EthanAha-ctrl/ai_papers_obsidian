

我来搜索并解析 Scale.com 的 RLHF 服务页面，获取详细信息。






















现在我已经收集到大量信息，让我来详细整理 Scale.com 的 RLHF 服务特色。

---

# Scale AI 的 RLHF 服务：全面技术解析

## 一、整体定位：Alignment-as-a-Service

Scale AI 是目前全球最大的 **RLHF data labeling** 供应商，年营收超过 **$750M+** 主要来自为 LLM 提供 human feedback data。其核心客户包括 **OpenAI**（用于训练 ChatGPT）、**Meta**（$14.3B 投资）、**Google**、**Anthropic** 等顶级 AI Lab。

Scale 的 RLHF 服务本质上是 **"Alignment-as-a-Service"**——即把 LLM alignment pipeline 中最耗费人力的部分（human preference data collection）作为一个端到端的托管服务。

> 参考：[Interconnects: Alignment-as-a-Service](https://www.interconnects.ai/p/alignment-as-a-service)

---

## 二、核心 RLHF Pipeline 的技术背景

从第一性原理出发，理解 Scale 的 RLHF 服务需要先理解 RLHF 的完整 training pipeline：

### RLHF 三阶段标准流程

```
Stage 1: SFT (Supervised Fine-Tuning)
  Pretrained LLM → 用高质量 (prompt, response) pairs 做 supervised learning

Stage 2: Reward Model Training
  收集 human preference data → 训练 Reward Model R(x, y)

Stage 3: RL Optimization (PPO / GRPO / DPO)
  用 Reward Model 的信号优化 Policy Model
```

**Reward Model** 的核心训练目标（Bradley-Terry Model）：

$$P(y_w \succ y_l | x) = \sigma(R(x, y_w) - R(x, y_l))$$

其中：
- $x$ = prompt (输入)
- $y_w$ = preferred response (人类偏好的回答)
- $y_l$ = rejected response (人类不偏好的回答)
- $R(x, y)$ = Reward Model 对 (prompt, response) pair 给出的 scalar reward
- $\sigma$ = sigmoid function
- $\succ$ = "preferred over" 的关系

**Loss function** 为：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim D} [\log \sigma(R(x, y_w) - R(x, y_l))]$$

Scale 的 RLHF 服务主要覆盖 **Stage 1 和 Stage 2** 中人类数据收集的部分。

---

## 三、Scale RLHF 的核心产品特性

### 1. **LLM Toolkit — 多种 Annotation Task Types**

Scale 提供了一个专门为 LLM 设计的 **Large Language Model Toolkit**，支持多种任务类型：

| Task Type | 描述 | 用于 RLHF 哪个阶段 |
|-----------|------|-------------------|
| **Text Collection** | Annotator 从零撰写高质量 prompt 和 response | SFT Data |
| **Comparison (SBS - Side-by-Side)** | 两个 model response 并排展示，annotator 选择更好的 | Reward Model Training |
| **Ranking** | 对多个 response 进行全序排列 | Reward Model Training |
| **Rating** | 对单个 response 按多维度打分（如 helpfulness, harmlessness, honesty） | Reward Model Training / Evaluation |
| **Response Editing / Rewriting** | Annotator 修改 model 生成的 response 使其更好 | SFT Data (高质量) |
| **Prompt Engineering** | 专门收集高质量、多样化的 prompt | 全阶段 |

> 参考：[Scale Blog: LLM Toolkit](https://scale.com/blog/scales-large-language-model-toolkit)

### 2. **Generative AI Data Engine**

这是 Scale 的旗舰产品平台，特点包括：

- **Rapid dataset creation**：快速创建定制化的高质量 dataset
- **Vetted Subject Matter Experts (SMEs)**：经过验证的领域专家
- **Flexible project archetypes**：可自定义的项目模板
- **API-driven workflow**：通过 API 编程式管理整个 data pipeline

> 参考：[Scale Generative AI Data Engine](https://scale.com/generative-ai-data-engine)

### 3. **Rubric-based Evaluation System**

Scale 的一大特色是引入了结构化的 **Rubrics**（评分标准）系统：

- 每个 evaluation task 可以绑定多个维度的 rubric（如 accuracy, helpfulness, safety, formatting）
- 每个 rubric criterion 可以有不同的 **categorical weight**
- 这使得 human feedback 更加 **structured、consistent、reproducible**

> 参考：[GenAI Docs: Rubrics](https://docs.genai.scale.com/project-archetypes/rubrics)

---

## 四、Scale 的突破性研究：Rubrics as Rewards (RaR)

这是 Scale Research 团队发布的一篇重要论文，直接来自他们的 RLHF 实践经验：

### 核心思想

传统 RLHF 的 Reward Model 输出一个 **single scalar reward**，这在很多 non-verifiable domain（如 creative writing, open-ended QA）中存在问题：
- 模糊的偏好信号
- Reward hacking 风险
- 不可解释性

**RaR** 的解决方案是：将 **checklist-style rubric** 转化为 **structured reward signal**。

### 技术架构

```
Input: (prompt x, response y, rubric R = {c₁, c₂, ..., cₖ})

Step 1: LLM-as-Judge 对每个 criterion cᵢ 独立评估
  → 得到 per-criterion score sᵢ ∈ {0, 1, ..., m}

Step 2: Weighted aggregation
  → R(x, y) = Σᵢ wᵢ · sᵢ

Step 3: 用 R(x, y) 作为 reward signal 进行 RL training (GRPO)
```

其中：
- $c_i$ = 第 $i$ 个 rubric criterion（如 "factual accuracy", "code correctness", "tone appropriateness"）
- $s_i$ = 该 criterion 的得分
- $w_i$ = 该 criterion 的权重（categorical weight）
- $k$ = rubric 中 criterion 的总数

### 关键发现

- **小模型** judge（如 7B/8B parameter）使用 rubric 后，其与 human preference 的一致性可以 **超过** 大模型 judge 不使用 rubric 的表现
- Rubric-guided RL 训练的 model 在 open-ended tasks 上表现更稳定
- 解决了传统 Reward Model 在 subjective domain 中的 reward hacking 问题

> 参考：
> - [Scale Research: Rubrics as Rewards](https://scale.com/research/rubrics_as_rewards)
> - [arXiv Paper](https://arxiv.org/html/2507.17746v1)
> - [Scale Blog: Using Rubrics to Build Better Models](https://scale.com/blog/rubrics-as-rewards)
> - [Enterprise RaR](https://scale.com/blog/enterprise-rar)

---

## 五、Subject Matter Expert (SME) Network

Scale 的核心竞争壁垒之一是其庞大的 **专家 annotator 网络**：

### Domain Coverage

| Domain | Expert 类型 | 典型任务 |
|--------|------------|---------|
| **Coding / SWE** | Software Engineer, PhD CS | 评估 code generation, debugging, code review |
| **Mathematics** | Math PhD, Professors | 评估 mathematical reasoning, proof verification |
| **Creative Writing** | Professional Writers | 评估 style, coherence, creativity |
| **Medicine / Healthcare** | MD, Clinical Researchers | 评估 medical accuracy, clinical reasoning |
| **Law** | JD, Practicing Lawyers | 评估 legal reasoning, case analysis |
| **Finance** | CFA, Quantitative Analysts | 评估 financial analysis accuracy |
| **Multilingual** | Native Speakers | 多语言 response quality evaluation |
| **STEM** | PhD-level Researchers | 评估 scientific reasoning |

这些 expert 不只是做简单的 A/B preference selection，而是能够：
- 撰写 **expert-level response**（用于 SFT）
- 对 response 做 **细粒度 rubric-based evaluation**
- 进行 **Red Teaming**（尝试 break model safety guardrails）
- 验证 model 输出的 **factual correctness**

> 参考：[Scale.com/rlhf](https://scale.com/rlhf)

---

## 六、Red Teaming 和 Safety Alignment

Scale 的 RLHF 服务还包含专门的 **AI Safety 和 Red Teaming** 模块：

### Red Teaming Workflow

```
1. Adversarial Prompt Generation
   → Expert 构造恶意/边界 prompt 试图诱导 model 输出有害内容

2. Safety Evaluation
   → 评估 model response 是否存在 toxicity, bias, misinformation

3. Safety Preference Data
   → 将 (unsafe prompt, safe response, unsafe response) triples 纳入 RLHF training

4. Iterative Hardening
   → 用新发现的 vulnerability 持续改进 model safety
```

这直接服务于 RLHF 中的 **HHH (Helpful, Harmless, Honest)** 对齐目标。

> 参考：[Scale Blog: Rethink Red Teaming](https://scale.com/blog/rethink-red-teaming)

---

## 七、RLHF Meets Specific Domains — 案例研究

### RLHF + Text2SQL

Scale 团队展示了将 RLHF 应用于 **Text2SQL** 任务的案例：

- 使用 **Mistral mixtral-8x7b-instruct** model
- 在 **BIRD dataset** 上
- 采用 **hybrid reward model**：结合 database execution feedback（verifiable signal）+ human preference（non-verifiable signal）
- 结果显示 RLHF 显著提升了 Text2SQL 的准确率

Hybrid reward 的公式概念：

$$R_{hybrid}(x, y) = \alpha \cdot R_{exec}(x, y) + (1-\alpha) \cdot R_{human}(x, y)$$

其中：
- $R_{exec}$ = execution-based reward（SQL 执行结果是否正确）
- $R_{human}$ = human preference-based reward
- $\alpha$ = blending coefficient

> 参考：[Scale Blog: RLHF Text2SQL](https://scale.com/blog/rlhf-text2sql)

---

## 八、技术架构：GenAI Platform API

Scale 提供了完整的 **API-driven** RLHF 数据收集平台：

### 核心概念层级

```
Organization
  └── Project (定义 task type + rubrics + guidelines)
       └── Batch (一组 tasks 的集合)
            └── Task (单个 annotation unit)
                 ├── prompt
                 ├── response(s)
                 ├── rubric scores
                 └── preference / ranking labels
```

### API 示例结构（来自 GenAI Docs）

```json
{
  "task_id": "task_123",
  "project": "project_123",
  "batch": "batch_123",
  "status": "completed",
  "params": {
    "prompt": "Explain quantum entanglement...",
    "responses": [
      {"model": "gpt-4", "text": "..."},
      {"model": "claude-3", "text": "..."}
    ]
  },
  "response": {
    "preference": "response_0",
    "rubric_scores": {
      "accuracy": 4,
      "helpfulness": 5,
      "safety": 5
    }
  }
}
```

> 参考：[GenAI Docs: RLHF](https://docs.genai.scale.com/project-archetypes/rlhf)

---

## 九、与竞争对手对比

| Feature | **Scale AI** | **Surge AI** | **Appen** | **Labelbox** |
|---------|-------------|-------------|-----------|-------------|
| Expert SME Network 规模 | ⭐⭐⭐⭐⭐ (最大) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Rubric-based Evaluation | ✅ 原生支持 + RaR 研究 | ✅ | ⚠️ 有限 | ⚠️ 有限 |
| API-first Architecture | ✅ | ✅ | ⚠️ | ✅ |
| Red Teaming | ✅ 专门模块 | ⚠️ | ⚠️ | ❌ |
| 核心客户 | OpenAI, Meta, Google | Anthropic | Various | Various |
| Domain Coverage | 最广 (Coding, Math, Law, Med...) | 较广 | 广 | 有限 |
| 年营收 (RLHF) | $750M+ | 较小 | 较大 | 中等 |
| Research Output | ✅ (RaR, Text2SQL, etc.) | 有限 | 有限 | 有限 |

> 参考：[Interconnects: Alignment-as-a-Service](https://www.interconnects.ai/p/alignment-as-a-service)

---

## 十、直觉总结：为什么 Scale 在 RLHF 领域独特？

从第一性原理看，RLHF 的核心瓶颈是：

1. **Human feedback 是 non-scalable 的**——每个 preference label 都需要人类时间
2. **Expert feedback 的 quality 远超 crowd feedback**——尤其在 coding、math、science 领域
3. **Consistency 是关键**——同一个 prompt 被不同 annotator 评估时需要一致性

Scale 的三大核心壁垒恰好对应解决这三个问题：

| 瓶颈 | Scale 的解决方案 |
|------|----------------|
| Scalability | 全球最大的 vetted annotator 网络 + API-driven pipeline 实现高吞吐量 |
| Quality | PhD-level SME network 覆盖各个专业领域 |
| Consistency | Rubric-based evaluation system + RaR framework 将主观偏好结构化 |

这三者的组合效应让 Scale 成为了几乎所有 top AI lab 的 RLHF data provider，形成了强大的 **network effect**：更多客户 → 更多 annotator 加入 → 更好的 domain coverage → 更吸引客户。

---

**关键参考链接汇总：**
- [Scale RLHF 产品页](https://scale.com/rlhf)
- [Scale Generative AI Data Engine](https://scale.com/generative-ai-data-engine)
- [Scale LLM Toolkit Blog](https://scale.com/blog/scales-large-language-model-toolkit)
- [GenAI API Docs: RLHF](https://docs.genai.scale.com/project-archetypes/rlhf)
- [GenAI API Docs: Rubrics](https://docs.genai.scale.com/project-archetypes/rubrics)
- [Rubrics as Rewards Paper](https://arxiv.org/html/2507.17746v1)
- [Scale Blog: Rubrics as Rewards](https://scale.com/blog/rubrics-as-rewards)
- [Enterprise RaR](https://scale.com/blog/enterprise-rar)
- [RLHF + Text2SQL](https://scale.com/blog/rlhf-text2sql)
- [Red Teaming Blog](https://scale.com/blog/rethink-red-teaming)
- [Alignment-as-a-Service Analysis](https://www.interconnects.ai/p/alignment-as-a-service)