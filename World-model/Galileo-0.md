# Galileo-0: 面向 World Model 的可扩展 World Critic 深度解析

---

## 1. 核心动机与问题定义

### 1.1 从 Physion-Eval 的发现出发

文章从 Physion-Eval [1] 的核心发现出发：**视频生成模型越来越"好看"，但并不越来越"物理正确"**。具体数据触目惊心：

| 模型视角 | 含物理 glitch 的视频比例 |
|---------|----------------------|
| Exocentric (第三人称) | **83.3%** |
| Egocentric (第一人称) | **93.5%** |

这意味着几乎所有 SOTA 视频生成模型（Veo3, Kling, Sora2, Wan2.6, Seedance 等）的输出都包含人类可辨识的物理错误。

### 1.2 为什么需要 World Critic？——第一性原理推导

从第一性原理出发，我们可以将视频生成模型的优化目标分解为两个层面：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{appearance}} + \mathcal{L}_{\text{physics}}$$

当前范式主要优化 $\mathcal{L}_{\text{appearance}}$（视觉保真度），而 $\mathcal{L}_{\text{physics}}$（物理合理性）几乎没有显式优化信号。

文章类比了 LLM 领域的发展路径：

| LLM 发展路径 | Video Generation 对应 |
|-------------|---------------------|
| Human preference → scalar reward (RLHF) | Binary preference → "哪个视频更好" |
| Structured feedback → detailed critique | **World Critic → what/when/where/why failed** |

关键洞察：**标量偏好只能告诉你哪个视频"更好"，但不能告诉你哪里错了、何时开始错、为何违背物理规则**。这就像一个只会打分的老师 vs 一个能详细批改的老师——后者对学习才有真正价值。

### 1.3 World Critic 的双重价值

文章指出 World Critic 有两个关键应用场景：

**① Alignment Signal（对齐信号）**
- 为 RLHF / DPO 等提供结构化的 reward signal
- 不再是 $r(\text{video}) \in \mathbb{R}$，而是 $r(\text{video}) = \{(t_i, \mathbf{b}_i, c_i, e_i)\}_{i=1}^{N}$，其中：
  - $t_i$：glitch 发生的时间戳
  - $\mathbf{b}_i$：空间边界框
  - $c_i$：glitch 类别
  - $e_i$：自然语言解释

**② Inference-time Refinement（推理时迭代优化）**
- 单一分数只能 rerank，无法指导如何改进
- 结构化诊断可作为 agentic refinement loop 的控制信号
- 类比：iterative preference optimization [7]、reward-guided video generation [8]、agentic image refinement [9]

---

## 2. 人类 Critic 的可扩展性瓶颈

### 2.1 标注时间成本分析

这是文章中一个非常关键的实证数据（Figure 1）：

| 任务类型 | Easy (0 glitches) | Medium (1–5 glitches) | Hard (>5 glitches) |
|---------|-------------------|----------------------|-------------------|
| Binary preference | ~20s | ~20s | ~20s |
| Fine-grained spatiotemporal reasoning | ~2.5 min | ~18 min | **>45 min** |

**关键观察**：
- Binary preference 的时间几乎恒定（~20s），与 glitch 数量无关
- Spatiotemporal reasoning 的时间随 glitch 数量**非线性增长**
- 即使是**无 glitch 的视频**，也需要 ~2.5 min 来确认（因为需要多轮审视排除 subtle glitches）
- 中位数：每个 SOTA 模型生成的视频包含 **3 个 glitches**

### 2.2 为什么人类 Critic 不可扩展？

用第一性原理分解 fine-grained spatiotemporal reasoning 的认知负荷：

$$T_{\text{critic}} = \underbrace{T_{\text{scan}}}_{\text{全局扫描}} + N_{\text{glitch}} \times \underbrace{(T_{\text{detect}} + T_{\text{localize}} + T_{\text{explain}})}_{\text{per-glitch}} + T_{\text{verify\_negatives}}$$

其中：
- $T_{\text{scan}}$：多遍扫描视频以寻找候选区域
- $T_{\text{detect}}$：确定 glitch 是否真实存在（vs 正常遮挡等）
- $T_{\text{localize}}$：精确定位时空区域
- $T_{\text{explain}}$：生成自然语言解释
- $T_{\text{verify\_negatives}}$：确认其他区域没有遗漏的 glitch

这解释了为什么无 glitch 视频也要 2.5 min——$T_{\text{verify\_negatives}}$ 是主要开销。而随着 $N_{\text{glitch}}$ 增加，per-glitch 成本线性叠加，且 glitch 之间可能相互干扰，导致 $T_{\text{detect}}$ 进一步上升。

---

## 3. 构建 World Critic 的核心挑战

### 3.1 Glitch 的本质：时序演化异常，非静态语义模式

这是文章最深刻的洞察之一。标准视频理解任务 vs glitch 检测的对比：

| 维度 | 标准视频理解 | Glitch 检测 |
|------|-----------|-----------|
| 目标 | 已知类别（动作/物体） | 未知异常模式 |
| 特征 | 稳定的视觉模式 | **时序演化中的违例** |
| 方法 | 语义匹配即可 | 需要物理推理 |
| 类内变化 | "跑步"看起来都差不多 | "物体消失"在不同场景完全不同 |

**举例**：同样是"物体消失"：
- 杯子从桌上消失
- 球在飞行中途消失
- 物体从人手中脱落

这三者视觉表现完全不同，但都属于 **object existence discontinuity**。模型不能靠"记忆消失是什么样子"来解决，必须联合推理：

$$\text{Glitch}(o, t) = f\Big(\underbrace{\text{Persistence}(o, t)}_{\text{物体持久性}}, \underbrace{\text{Occlusion}(o, t)}_{\text{遮挡推理}}, \underbrace{\text{Motion}(o, t)}_{\text{运动连续性}}, \underbrace{\text{Interaction}(o, t)}_{\text{交互动力学}}\Big)$$

### 3.2 高频感知信号的双重挑战

| 挑战类型 | 描述 | 例子 |
|---------|------|------|
| **时间短暂性** | Glitch 仅持续几帧 | 物体消失几帧后重现 |
| **空间细微性** | Glitch 仅涉及小区域 | 手轻微未触碰物体、小部件短暂错位 |

这意味着 World Critic 需要同时具备：
- **细粒度感知**（fine-grained perception）：捕捉帧级别的细微变化
- **结构化推理**（structured reasoning）：判断变化是否违背物理预期

---

## 4. Galileo-0 的架构：两阶段 Spatiotemporal Reasoning Pipeline

### 4.1 架构概览

Galileo-0 采用 **Proposal + Verification** 的两阶段设计，类比目标检测领域的经典范式：

| 领域 | Proposal 阶段 | Verification 阶段 |
|------|-------------|-----------------|
| Object Detection | Region proposals (Selective Search / RPN) | Classification + Regression (Fast R-CNN) |
| Temporal Action Localization | Temporal proposals (SSN) | Action classification |
| Spatiotemporal Action Detection | Tube proposals (STEP) | Action classification |
| **Galileo-0** | **Spatiotemporal glitch proposals** | **Physical violation reasoning** |

### 4.2 Stage 1: Spatiotemporal Perception（时空感知）

**目标**：在时空体积（spatiotemporal volume）中提出可能包含物理不一致行为的候选区域。

形式化地，给定视频 $V \in \mathbb{R}^{T \times H \times W \times 3}$，Stage 1 输出：

$$\mathcal{P} = \{(t_i^s, t_i^e, \mathbf{b}_i)\}_{i=1}^{K}$$

其中：
- $t_i^s, t_i^e$：候选 glitch 的起始和结束时间
- $\mathbf{b}_i \in \mathbb{R}^{4}$：空间边界框 $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$
- $K$：候选数量（需平衡 recall vs 计算量）

**设计思路**：高 recall 为先——宁可多提一些候选（false positives），也不要遗漏真实 glitch（false negatives）。这与目标检测中 RPN 的设计哲学一致。

### 4.3 Stage 2: Spatiotemporal Reasoning（时空推理）

**目标**：对每个候选区域进行深入检查，判断其是否真正违背物理预期。

关键区别于简单分类的是：**需要结合上下文进行因果推理**。

| 场景 | 视觉表现 | 是否为 Glitch |
|------|---------|-------------|
| 猫消失在遮挡物后 | 物体从视野中消失 | ❌ 正常遮挡 |
| 公交车静止但轮子转动 | 轮子在转但车不动 | ✅ 物理违例 |

Stage 2 需要显式推理对象的时空状态变化：

$$\text{State}(o, t) = \{\text{position}, \text{velocity}, \text{visibility}, \text{structural\_integrity}, \text{identity}\}$$

然后判断状态转移是否物理合理：

$$\text{Violation} = \mathbb{1}\Big[\text{State}(o, t+\Delta t) \not\approx \text{Predict}(\text{State}(o, t), \text{Physics})\Big]$$

### 4.4 训练数据与监督

- 使用人类专家的 reasoning traces 作为监督信号
- 每个 glitch 标注包含：timestamps、structured labels、natural-language explanations
- 基于自研 MLLM（Multimodal Large Language Model）进行训练
- 监督信号的结构化使得模型不仅学习"有没有 glitch"，还学习"如何推理 glitch"

### 4.5 架构的 First-Principles 优势

为什么两阶段而非端到端？

1. **搜索空间分解**：视频的时空体积 $T \times H \times W$ 非常大，端到端直接分类每个时空位置的计算成本为 $O(T \times H \times W)$。两阶段将其分解为 $O(K)$ 个小区域的精细推理。

2. **感知与推理的解耦**：感知需要高分辨率、细粒度的视觉特征；推理需要长上下文、因果链的语义推理。两者对模型能力的要求不同，解耦后各自优化。

3. **可解释性**：每个 Stage 的输出都有明确语义——候选区域 + 推理结果，便于人类审查和调试。

---

## 5. 实验结果详解

### 5.1 数据集构建

**覆盖的 Glitch 类别**：

| 类别 | 英文名 | 描述 |
|------|--------|------|
| 物体存在性不连续 | Object Existence Discontinuity | 物体突然出现/消失 |
| 物体身份或属性漂移 | Object Identity/Attribute Drift | 物体颜色、形状等属性在视频中发生变化 |
| 物体结构或部件不一致 | Object Structural/Part Inconsistency | 物体的部件错位、脱离等 |
| 文本或语言不一致 | Textual/Language Inconsistency | 视频中的文字内容错误 |

前三个归为 **Object-related glitch**，最后一个归为 **Text-related glitch**。

**视频来源**：Veo3, Kling, Sora2, Wan2.6, Seedance 等（闭源 + 开源）

### 5.2 评估指标：Glitch-level Micro F1

文章选择 **glitch-level micro F1** 而非 mAP，理由明确：

| 指标 | 适用场景 | Galileo-0 的选择 |
|------|---------|-----------------|
| mAP | 跨阈值排序性能 | ❌ 不适合，因为目标是固定操作点的端到端 glitch 恢复 |
| Micro F1 | 固定操作点的精确恢复 | ✅ 更符合实际需求 |

**匹配规则**（pred ↔ GT 的 one-to-one matching）：

$$\text{Match}(p, g) = \mathbb{1}\Big[\underbrace{\text{tIoU}(p, g) \geq 0.3}_{\text{时间重叠}} \wedge \underbrace{\text{STS}(p, g) \geq 0.5}_{\text{语义相似度}}\Big]$$

其中：
- $\text{tIoU}(p, g) = \frac{|t_p \cap t_g|}{|t_p \cup t_g|}$：temporal Intersection over Union
- $\text{STS}(p, g)$：Semantic Textual Similarity [13]，衡量两个 glitch 原因描述的语义相似度

匹配后：
- Matched pairs → **True Positives (TP)**
- Unmatched predictions → **False Positives (FP)**
- Unmatched ground-truth → **False Negatives (FN)**

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F1 = \frac{2 \cdot P \cdot R}{P + R}$$

### 5.3 核心实验结果（Figure 3）

| 模型 | All Categories F1 | Text-related F1 | Object-related F1 |
|------|------------------|-----------------|-------------------|
| **Galileo-0** | **63.26%** | **84.10%** | **37.17%** |
| GPT-5.4 | 38.89% | 53.20% | — |
| Gemini 3.1 Pro | 36.48% | — | — |
| Qwen3.5-Plus | 35.15% | — | — |
| GLM-5V Turbo | 25.42% | — | — |
| Pegasus 1.2 | 3.40% | — | — |

**关键发现**：

1. **Galileo-0 全面领先**：在所有设定下 F1 最高，尤其 Text-related 类别大幅领先（84.10% vs 53.20%）。

2. **Text-related 远比 Object-related 容易**：
   - Galileo-0: 84.10% vs 37.17%
   - GPT-5.4: 53.20% vs (推测 ~30%)
   - 原因：文字 glitch 相对明确（字符变形、拼写错误），而物体 glitch 种类繁多、时空定位更难

3. **通用 MLLM 表现不佳**：即使是最强的 GPT-5.4，F1 也只有 38.89%，说明通用视觉理解能力 ≠ 物理推理能力

4. **Pegasus 1.2 几乎无法工作**（3.40%），可能因为其视频采样设置不够精细

### 5.4 评估设置细节

| 模型 | 视频采样 | 输入方式 | 分辨率 |
|------|---------|---------|--------|
| Gemini 3.1 Pro | 10 FPS | 原生视频输入 | ≤1080p |
| Qwen3.5-Plus | 10 FPS | 原生视频输入 | ≤1080p |
| GPT-5.4 | 10 FPS | 图像序列输入 | ≤1080p |
| GLM-5V Turbo | 默认设置 | 原生视频输入 | ≤1080p |
| Pegasus 1.2 | 默认设置 | 原生视频输入 | ≤1080p |

10 FPS 的选择是为了更好地捕捉短暂时序 glitch——以 10 FPS 采样 8 秒视频 = 80 帧，足以捕捉持续 2-3 帧的短暂 glitch。

---

## 6. 深层分析与未来展望

### 6.1 为什么 Object-related Glitch 这么难？

用信息论视角分析，Object-related glitch 检测的难度源于：

$$I(\text{Glitch}; \text{Observation}) = \underbrace{H(\text{Glitch})}_{\text{高：种类繁多}} - \underbrace{H(\text{Glitch} \mid \text{Observation})}_{\text{高：证据稀疏}}$$

- $H(\text{Glitch})$ 高：物体消失、属性漂移、部件不一致……每种又有无数变体
- $H(\text{Glitch} \mid \text{Observation})$ 高：关键证据可能仅在几帧的小区域内

相比之下，Text-related glitch：
- $H(\text{Glitch})$ 低：文字错误类型有限
- $H(\text{Glitch} \mid \text{Observation})$ 低：文字区域明确，错误通常显而易见

### 6.2 两阶段设计的局限

| 局限 | 描述 |
|------|------|
| Stage 1 recall 不足 | 如果 proposal 阶段遗漏了某个 glitch，Stage 2 无法弥补 |
| 长时序依赖 | 当前设计可能难以处理跨越长时间跨度的 glitch（如物体属性渐变） |
| 类别覆盖 | 目前仅覆盖 4 类 glitch，物理世界的违例远不止于此 |

### 6.3 World Critic → World Model 的闭环

文章展望的核心问题是：**如何将 structured critique 转化为 world model 的优化信号？**

可能的路径：

1. **Reward Shaping**：将 structured critique 转化为 dense reward signal，用于 RL fine-tuning
   $$r_t = \begin{cases} -\lambda & \text{if glitch detected at } t \\ 0 & \text{otherwise} \end{cases}$$

2. **Conditioned Generation**：将 critique 作为 condition，引导模型在特定时空区域避免已知错误

3. **Iterative Refinement**：Generate → Critique → Refine → Critique → ... 的 agentic loop

4. **Curriculum Learning**：从易到难，先修复 text-related glitch，再逐步处理 object-related

### 6.4 更广阔的愿景

文章最后提到 Galileo-0 的未来方向：

- **更准确**：提升 Object-related glitch 的 F1
- **更鲁棒**：处理 subtle 和 long-horizon failures
- **更广泛**：覆盖更多 glitch 类型
- **超越现实物理**：支持自定义物理规则（如游戏世界、科幻场景中的"合理性"定义）

最后一点特别有趣——这意味着 World Critic 不需要绑定地球物理，而是可以适应**任意物理规则**的世界，这对于游戏、VR、影视特效等领域有巨大价值。

---

## 7. 与相关工作的联系

| 相关方向 | 代表工作 | 与 Galileo-0 的关系 |
|---------|---------|-------------------|
| Video quality assessment | DOVER, FAST-VQA | 评估整体质量，不诊断具体物理错误 |
| RLHF for video generation | [2-6] | 从标量偏好到结构化反馈的演进 |
| Iterative refinement | [7-9] | World Critic 的输出可作为 refinement 的控制信号 |
| Object detection proposals | Fast R-CNN [10] | Galileo-0 的两阶段设计灵感来源 |
| Temporal action localization | SSN [11], STEP [12] | 时空定位的方法论参考 |
| Physion benchmark | [1] | 发现问题的起点 |

---

## 8. 总结

Galileo-0 的核心贡献可以用一个公式概括：

$$\text{World Critic}: V \xrightarrow{\text{Stage 1: Perception}} \mathcal{P} \xrightarrow{\text{Stage 2: Reasoning}} \{(t_i, \mathbf{b}_i, c_i, e_i)\}_{i=1}^{N_{\text{glitch}}}$$

它将视频物理错误检测从"给个分数"提升到"精确诊断 what-when-where-why"，这不仅是评估工具，更是未来视频世界模型自我改进的关键基础设施。

正如文章所言：**A scalar preference may say which video is better overall, but it cannot tell the model what physically failed, when the failure started, where it occurred, or why it violated the rules of the scene.** 这句话精准地概括了 World Critic 存在的意义。

---

**参考链接**：
- Physion benchmark: https://physion.github.io/
- Fast R-CNN: https://arxiv.org/abs/1504.08083
- RLHF 原始论文: https://arxiv.org/abs/2203.02155
- Video generation RLHF 相关: https://arxiv.org/abs/2312.00857