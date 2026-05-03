# 深度解读：Mantic 的 RL Fine-tuning 预测系统

这篇文章由 Mantic 撰写，发布于 Tinker 博客，核心论点是：**通过 RL fine-tuning，一个中等规模的开源模型 (gpt-oss-120b) 可以在判断性预测任务上达到前沿 LLM 的水平，并且在集成中具有不可替代的价值。** 下面我从第一性原理出发，逐层拆解。

---

## 一、问题设定：判断性预测 (Judgmental Forecasting)

### 1.1 什么是判断性预测？

与时间序列外推不同，判断性预测处理的是**缺乏充足历史数据**的二元事件预测问题：

$$\text{Question}: \quad \text{Will } [E] \text{ occur before } [T]?$$

其中 $E$ 是某个事件，$T$ 是截止日期。模型需要输出一个概率 $p \in [0, 1]$，表示事件 $E$ 在时间 $T$ 之前发生的概率。

这类问题的特点：
- **数据稀缺**：无法用经典统计方法（如 ARIMA、指数平滑）
- **需要研究推理**：需要搜集上下文、权衡不同情景
- **应用广泛**：地缘政治、商业决策、经济政策

> 参考：Philip Tetlock 的 *Superforecasting* (2015) 是这个领域的开山之作，https://www.penguinrandomhouse.com/books/316739/superforecasting-by-philip-tetlock-and-dan-gardner/

### 1.2 评估指标：Metaculus Baseline Score

文章使用 Metaculus 平台的 **baseline score**，本质是 **log scoring rule** 的重新缩放：

$$S = \frac{\ln(p_{\text{assigned}}) - \ln(0.5)}{\ln(1) - \ln(0.5)} \times 100 = \frac{\ln(p_{\text{assigned}}) - \ln(0.5)}{-\ln(0.5)} \times 100$$

其中：
- $p_{\text{assigned}}$ 是模型赋予真实结果的概率
- 如果真实结果是 "Yes"，则 $p_{\text{assigned}} = p(\text{Yes})$
- 如果真实结果是 "No"，则 $p_{\text{assigned}} = 1 - p(\text{Yes})$
- 完美预测（$p_{\text{assigned}} = 1$）得 100 分
- 均匀预测（$p_{\text{assigned}} = 0.5$）得 0 分

**关键直觉**：log score 严厉惩罚过度自信的错误预测。如果你说 $p = 0.99$ 但事件没发生，你会得到 $\ln(0.01) \approx -4.6$ 的极大负值。

---

## 二、系统架构：两阶段流水线

### 2.1 研究阶段 (Research Phase)

```
Question → Deep Research Agents → Collected Context → Summarized Prompt
```

- 多个搜索 Agent 并行工作，搜集与问题相关的信息
- 例如 "Will the US attack Venezuela before 2026?"，Agent 会搜索：
  - 加勒比海军事集结情况
  - Trump 总统的声明
  - 委内瑞拉经济健康状况
- 搜集到的研究结果被汇总成一个 prompt，输入预测阶段

### 2.2 预测阶段 (Prediction Phase) — 混合模型 (Mixture Model)

这是文章最精巧的技术设计。模型不是直接输出一个概率 $p$，而是**参数化一个混合模型**来描述事件发生时间的分布。

设事件首次发生的时间为 $\tau$，模型输出：

$$f(\tau) = \sum_{k=1}^{K} w_k \cdot f_k(\tau | \theta_k)$$

其中：
- $K$：混合成分数量（由 LLM 选择）
- $w_k$：第 $k$ 个成分的权重，$\sum_k w_k = 1$
- $f_k(\tau | \theta_k)$：第 $k$ 个成分的概率密度函数，参数为 $\theta_k$
- 每个成分代表一种**不同的情景/叙事**导致事件发生

对应的累积分布函数 (CDF)：

$$F(t) = P(\tau \leq t) = \sum_{k=1}^{K} w_k \cdot F_k(t | \theta_k)$$

最终预测概率：

$$p(\text{event before } T) = F(T) = \sum_{k=1}^{K} w_k \cdot F_k(T | \theta_k)$$

**为什么用混合模型？** 这是一种 **结构化的不确定性表达**。不同成分对应不同的因果路径/情景。例如：

- 成分 1（权重 $w_1 = 0.3$）：外交谈判破裂导致军事行动，中位时间 $\sim$ 2个月
- 成分 2（权重 $w_2 = 0.7$）：局势逐步缓和，事件不太可能发生

这种方法比直接输出单个概率值具有更好的 **校准性 (calibration)** 和 **可解释性 (interpretability)**。

> 参考：Halawi et al. (2024) "Approaching Human-Level Forecasting with Language Models" https://arxiv.org/abs/2402.18563

---

## 三、RL Fine-tuning：核心训练细节

### 3.1 训练数据

- **~10,000 个二元预测问题**
- 时间范围：2024年8月 — 2025年12月
- 模型的知识截止日期在这些问题之前，所以**解决方案已知但模型不知道**
- 训练前先运行研究阶段，**缓存静态 prompts**（降低训练成本）

### 3.2 训练算法：GRPO 变体

文章使用 **policy gradient** 算法，基于 **GRPO-style advantage normalization**：

**标准 GRPO (Group Relative Policy Optimization)** 的核心思想：

$$\hat{A}_i = \frac{R_i - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$$

但文章做了一个关键修改：**不做除以标准差的归一化**（no division by the standard deviation）：

$$\hat{A}_i = R_i - \text{mean}(\mathbf{R})$$

原因：Brier score 已经是有界的 $[0, 1]$，方差本身就比较低，再除以标准差会过度放大微小差异。

策略梯度更新：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\hat{A}_i \cdot \nabla_\theta \log \pi_\theta(a_i | s_i)\right]$$

其中：
- $\pi_\theta$ 是当前策略（LLM 的 token 生成策略）
- $s_i$ 是第 $i$ 个 rollout 的 prompt（包含研究上下文）
- $a_i$ 是生成的 token 序列（包括推理链和最终概率）
- $R_i$ 是 Brier score reward

### 3.3 Reward 函数选择：Brier Score vs Log Score

**Brier Score**：

$$R_{\text{Brier}} = 1 - (p - o)^2$$

其中 $o \in \{0, 1\}$ 是真实结果（1 = Yes, 0 = No），$p$ 是预测概率。

- **严格proper** (strictly proper)：唯一最优策略是诚实报告真实信念
- 有界于 $[0, 1]$，产生**低方差的策略梯度估计**
- 对于固定结果，关于 $p$ 严格单调

**Log Score**：

$$R_{\text{Log}} = \ln(p \cdot o + (1-p) \cdot (1-o))$$

- 也是 strictly proper
- **无界**：当 $p \to 0$ 但 $o = 1$ 时，$R_{\text{Log}} \to -\infty$
- 产生**高方差**的梯度估计，训练不稳定

文章发现 Brier score 训练更稳定，这是一个重要的实践洞察。

### 3.4 关键工程细节

| 超参数 | 值 | 备注 |
|---------|-----|------|
| Group size | 8 | 每个问题生成 8 个 rollout |
| Batch size | 64 | 更大的 batch size 会导致训练不稳定 |
| Reward | Brier Score | 严格 proper + 有界 |
| Advantage | $R_i - \text{mean}(\mathbf{R})$ | 无标准差归一化 |

**Importance Sampling Correction**：

文章指出 vLLM（采样后端）和 FSDP（训练后端）对**相同策略**会产生不同的 token 概率。这会引入策略梯度的偏差。解决方案是对 advantage 做重要性采样校正：

$$\hat{A}_i^{\text{corrected}} = \hat{A}_i \cdot \frac{\pi_\theta^{\text{train}}(a_i|s_i)}{\pi_\theta^{\text{sample}}(a_i|s_i)}$$

这是一个非常实用的工程技巧，解决了 RLHF 基础设施中的常见问题。

**为什么 group size = 8 就够了？** 因为 Brier score 对不同 $p$ 值严格单调，所以同一个 prompt 的 8 个 rollout 几乎不可能产生相同的 reward，不需要处理 tie-breaking。

---

## 四、实验结果深度解析

### 4.1 核心结果：Fine-tuning 的提升

| 模型配置 | Baseline Score |
|----------|---------------|
| gpt-oss-120b (原始, 无 research) | ~35 |
| gpt-oss-120b (原始, 有 research+tools) | 38.6 |
| gpt-oss-120b (fine-tuned, 无 research) | ~38 (+3) |
| gpt-oss-120b (fine-tuned, 有 research+tools) | **45.8 (+7)** |
| Gemini 3 Pro | ~45 |
| Grok 4 | ~44 |
| GPT-5 | ~43 |

**关键发现**：
1. Fine-tuning **单独使用**只带来 +3 分提升
2. Fine-tuning **配合 research + tools** 带来 +7 分提升
3. 这说明 **research 和 tools 不仅提供更好的初始化，还改善了优化动力学**

直觉：如果 prompt 中没有足够的上下文信息，模型即使学到了更好的"预测策略"，也无法弥补信息匮乏。Research 提供了"原材料"，tools 提供了"表达工具"，fine-tuning 则让模型学会更有效地使用它们。

### 4.2 集成分析：多样性 > 纯精度

最优 5 样本集成：
- **fine-tuned gpt-oss-120b × 2** (40%)
- **Gemini 3 Pro × 1** (20%)
- **GPT-5 × 1** (20%)
- **Grok 4 × 1** (20%)

**不可替代性排名**（移除后性能下降幅度）：
1. 🥇 **Grok 4** — 最不可替代
2. 🥈 **fine-tuned gpt-oss-120b** — 第二不可替代
3. GPT-5 / Gemini 3 Pro — 高度可替代

为什么？用 **Jensen-Shannon Divergence (JSD)** 来衡量模型间预测分布的差异：

$$\text{JSD}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M)$$

其中 $M = \frac{1}{2}(P + Q)$。

- GPT-5 和 Gemini 3 Pro 的 JSD 很低 → 预测高度相关 → 互为冗余
- Grok 4 和 fine-tuned gpt-oss-120b 与其他模型 JSD 较高 → 提供多样性

**三集成实验进一步验证**：

| 集成组合 | 权重分配 |
|----------|---------|
| fine-tuned + Gemini 3 Pro + GPT-5 | 56% / 26% / 18% |
| fine-tuned + Gemini 3 Pro + Grok 4 | 48% / 26% / 26% |

fine-tuned model 在两种配置中都获得了**最大权重**，说明它既准确又独特。

---

## 五、第一性原理分析：为什么 RL Fine-tuning 有效？

从第一性原理出发，通用 LLM 和专业预测模型之间的差距来自三个层面：

### 5.1 认知层面：校准 (Calibration)

通用 LLM 在预测任务上存在 **系统性偏差**：
- **过度自信**：倾向于输出极端概率（接近 0 或 1）
- **基数率忽视**：不充分考虑先验基础概率

RL fine-tuning 通过 Brier score reward 间接校正了这些偏差，因为 Brier score 的 strictly proper 性质确保**诚实报告是最优策略**。

### 5.2 行为层面：策略优化

通用 LLM 的"预测策略"是通过预训练和通用 RLHF 学到的，不是针对预测任务优化的。Fine-tuning 让模型学会：

- 如何更好地利用研究上下文中的信号
- 如何更合理地参数化混合模型
- 如何在不确定时保持适度的概率

### 5.3 多样性层面：正交化

最有趣的发现是 **fine-tuned model 与前沿 LLM 预测去相关**。这可能因为：

1. **训练数据差异**：fine-tuned model 在 ~10k 预测问题上做了专门的策略优化
2. **模型规模差异**：120B vs 数万亿参数的前沿模型，内在表征不同
3. **推理路径差异**：RL 训练可能发展出了不同于 SFT 模型的推理模式

---

## 六、局限性与未来方向

### 6.1 当前局限

1. **仅限二元问题**：尚未覆盖数值型问题（如经济指标）和多选题（如选举结果）— 但文章提到已在进行中
2. **静态研究阶段**：research 是预计算并缓存的，不在 RL 训练循环内
3. **模型规模**：gpt-oss-120b 相对较小，更大模型（如 Kimi K2.5）的 fine-tuning 效果未知
4. **问题质量**：随着模型变强，需要更难的预测问题才能区分性能

### 6.2 最有前景的下一步

**Information Retrieval inside the loop** — 这可能是最大的突破方向。如果预测 LLM 在 RL 训练中可以使用搜索工具，模型可以学到：

- **何时搜索**：识别自己知识不足的时刻
- **搜索什么**：选择最能减少不确定性的查询
- **如何整合**：将搜索结果与已有推理融合

这本质上是在训练一个 **自主学习的研究-预测 Agent**，远比静态研究阶段强大。

> 相关工作：Turtel et al. (2025) "Outcome-based Reinforcement Learning to Predict the Future" https://arxiv.org/abs/2503.03255

---

## 七、与更广泛 AI 趋势的联系

### 7.1 RL > SFT for Reasoning

这篇文章是 **RL fine-tuning 提升推理能力** 的又一证据，呼应了 DeepSeek-R1、OpenAI o1/o3 的工作范式。关键区别是：这里用 **真实世界结果** 作为 reward signal，而不是数学验证或代码执行。

### 7.2 预测市场与 AI Safety

准确的 AI 预测系统对 **AI safety** 有深远意义：
- 可以用来预测 AI capability milestones
- 可以评估不同安全策略的有效性
- 是一种 ** scalable oversight** 机制

> 参考：Tetlock et al. 的超级预测研究 https://goodjudgment.com/

### 7.3 Ensemble Diversity 的理论

这篇文章的集成分析实践中验证了一个理论洞察：**在集成中，多样性 (diversity) 和精度 (accuracy) 同等重要**。数学上，集成的误差可以分解为：

$$\text{Ensemble Error} = \bar{e} - \bar{d}$$

其中 $\bar{e}$ 是成员模型的平均误差，$\bar{d}$ 是成员模型间的平均多样性。Fine-tuned model 之所以不可替代，是因为它**保持低 $\bar{e}$ 的同时贡献高 $\bar{d}$**。

> 参考：Krogh & Vedelsby (1995) "Neural network ensembles, cross validation, and active learning" https://papers.nips.cc/paper/1994/hash/c3d5e9055b6dc5969b5c8b2a6c5e8e6e-Abstract.html

---

## 八、总结

| 维度 | 核心结论 |
|------|---------|
| **可行性** | RL fine-tuning 可将中等模型的预测能力提升到前沿 LLM 水平 |
| **机制** | Brier score + GRPO + importance sampling correction 是实用的训练配方 |
| **协同效应** | Research + Tools 与 fine-tuning 存在正交互作用（+3 vs +7） |
| **多样性价值** | Fine-tuned model 的预测与前沿 LLM 去相关，在集成中不可替代 |
| **核心洞察** | On-task training 可以突破 off-the-shelf LLM 的预测天花板 |

这篇文章最重要的takeaway不是"小模型可以打败大模型"（它只是 marginally 超过），而是：**专门的任务训练可以产生与通用大模型正交的预测能力**，这种正交性在集成中具有巨大价值。这暗示了 AI 预测的未来不在于单一超级模型，而在于**多样化专家模型的生态系统**。