来自 Alibaba 的 AI Hardware Division
在 function calling 方面的能力有效地蒸馏到超小模型 0.6B参数

Function Calling 是 LLM 作为智能体与外部工具/API交互的核心能力，典型的输出格式是 JSON：

```
{"name": "function_name", "arguments": {"arg1": "value1", ...}}
```

论文指出现有的 **SFT+RL**（Supervised Fine-Tuning + Reinforcement Learning）范式在超小模型上存在三个核心问题：

| 问题                             | 描述                                                |
| ------------------------------ | ------------------------------------------------- |
| **Overfitting**                | 小模型容量有限，在有限的高质量数据上通过 SFT 容易过拟合，记忆特定的工具使用模式而不是泛化   |
| **Training Instability**       | 直接对小模型应用 RL 以其训练不稳定且效率低下                          |
| **Ineffective Binary Rewards** | 对于多解问题（如 function calling），标准的离散/二值奖励会过度惩罚有效的替代方案 |

### 1.3 KD+RL 范式的挑战

作者提出使用 **Knowledge Distillation (KD)** 来为 RL 提供稳定、可泛化的初始化，但这引入了新挑战：

1. **KD instability and constrained exploration**：标准 KD 使用 top-k 截断，学生模型的长尾分布缺乏监督，导致训练不稳定和模型崩溃
2. **Ineffective RL rewards**：多解问题中，离散奖励过度惩罚替代有效解
3. **Synergistic integration challenges**：实现 KD 和 RL 的真正协同而非干扰

---

## 2. STAR 方法详解

STAR 包含两个核心技术：
- **CKD (Constrained Knowledge Distillation)**
- **Sim-RL (Similarity-guided RL)**

### 2.1 Constrained Knowledge Distillation (CKD)

#### 2.1.1 背景知识：FKL vs RKL

在 KD 中，常用的两个损失函数：

**Forward KL Divergence (FKL)** - 分布覆盖型：
```
ℒFKL = Σx∈D DKL(P_T(y|x) || P_S(y|x))
     = Σx∈D Σy P_T(y|x) log[P_T(y|x) / P_S(y|x)]  --- (1)
```

**Reverse KL Divergence (RKL)** - 模态搜索型：
```
ℒRKL = Σx∈D DKL(P_S(y|x) || P_T(y|x))
     = Σx∈D Σy P_S(y|x) log[P_S(y|x) / P_T(y|x)]  --- (2)
```

**变量说明：**
- `x`：输入上下文（包含用户查询和可用函数集合 F = {f1, f2, ..., fN}）
- `y`：输出 token
- `P_T`：Teacher 模型的输出概率分布
- `P_S`：Student 模型的输出概率分布
- `DKL(· || ·)`：KL 散度

**FKL vs RKL 的关键区别：**
- **FKL**：鼓励学生模型覆盖教师分布的全部概率质量
- **RKL**：强迫学生模型专注于教师模型的高概率 token，忽略长尾分布

#### 2.1.2 Top-k 截断下的不稳定性

为了计算效率，KD 通常使用 **top-k 截断**，只在教师的前 k 个 token 上计算损失。

**关键发现**：RKL 与 top-k 截断结合会导致**灾难性训练崩溃**！

理论分析（Appendix A.3）：

**Top-k FKL 的梯度**：
```
∂ℒFKL-TopK / ∂zS_j = qj · Σi∈Ik pi - pj · 𝟏j∈Ik  --- (14)
```

其中：
- `zS_j`：学生模型对 token j 的 logit
- `qj`：学生对 token j 的概率
- `pi`：教师对 token i 的概率
- `Ik`：教师分布的 top-k 索引集
- `𝟏j∈Ik`：指示函数，j 在 Ik 中为 1，否则为 0

**分析**：
- 对于非 top-k token (j ∉ Ik)：梯度 = `qj · Σi∈Ik pi`（恒为正，抑制）
- 对于 top-k token (j ∈ Ik)：梯度 = `qj · Σi∈Ik pi - pj`（较小或负）

这创造了稳定的学习动态：非 top-k 的 logits 被强抑制，top-k 的 logits 被鼓励或弱抑制。

**Top-k RKL 的梯度**：
```
∂ℒRKL-TopK / ∂zS_j = qj · [log(qj/pj) + 1 - S]  --- (20)
```
其中 `S = Σi∈Ik qi · [log(qi/pi) + 1]`

**问题**：当教师对某些 top-k 项分配很小的概率（pj → 0）或学生过度自信时，梯度会不稳定，可能激励非 top-k 项！

#### 2.1.3 CKD 的核心思想

CKD 从稳定的 top-k FKL 出发，引入一个**目标化的正则化项**来控制学生长尾分布中最有问题的部分。

**CKD 损失函数**：
```
ℒCKD = ℒFKL-k + λtail · ℒtail  --- (3)
```

其中：
```
ℒFKL-k = Σx∈D Σv∈Vk(x) P_T(v|x) log[P_T(v|x) / P_S(v|x)]  --- (4)

ℒtail = Σx∈D Σv∈Vm(x) \ Vk(x) P_S(v|x)  --- (5)
```

**变量说明**：
- `Vk(x)`：教师模型的前 k 个高概率 token 集合（trusted set）
- `Vm(x)`：学生模型的前 m 个高概率 token 集合
- `Vm(x) \ Vk(x)`：学生认为可能（在前 m 中）但教师认为不重要（不在前 k 中）的 token——"confidently incorrect" 预测
- `λtail`：平衡超参数

**核心洞察**：`ℒtail` 仅对学生认为可能但教师认为不重要的 token 施加 L1 惩罚，这直接抑制学生对"置信但错误"的预测。

#### 2.1.4 CKD 的梯度分析

CKD 对 token j logit 的梯度：
```
对于"置信但错误"的 token (j ∈ J'm = Vm \ Vk):
∂ℒCKD / ∂zS_j = qj · [Σi∈Ik pi + λ(1 - Σi∈J'm qi)] - pj  --- (26)

对于其他非 top-k token:
∂ℒCKD / ∂zS_j = qj · (Σi∈Ik pi - λ Σi∈J'm qi)  --- (27)
```

**重平衡机制**：

| Token 类型 | 梯度效果 | 含义 |
|------------|----------|------|
| 置信但错误 (j ∈ J'm) | **显著增强抑制** | 专门惩罚最有可能是错误的预测 |
| 其他非 top-k | **减弱抑制** | 不浪费容量抑制已经低概率的类 |

这种机制防止模型将所有概率质量都坍缩到 top-k 集合，鼓励更健康、不那么尖峰的学生分布。

#### 2.1.5 直觉理解

想象一个学生在学习回答问题：
- **FKL**：告诉学生"老师认为可能的所有答案"
- **RKL**：告诉学生"只关注老师认为最可能的答案"，但学生可能过度自信
- **CKD**：告诉学生"关注老师认为可能的答案，但如果你自己很确信的答案老师没提到，那就要小心了！"

这就像一个老师既要指导学生关注重点，又要防止学生形成"钻牛角尖"式的过度自信。

---

### 2.2 Similarity-guided RL (Sim-RL)

Sim-RL 引入了细粒度的、基于相似度的奖励信号，为多解问题提供更丰富的学习信号。

#### 2.2.1 奖励设计

**1. Format Reward (Rformat)**

一个成功的 function call 的前提是正确格式：

```
Rformat = {1, if all format rules are satisfied; 0, otherwise}  --- (6)
```

**Qwen 工具调用模板的规则**：
1. 输出必须包含恰好一对 `
` 标签，封装推理过程
2. 如果调用函数，每个调用必须包裹在 `<tool_call>...</tool_call>` 标签中
3. 内容必须是单个 JSON 对象，包含 `"name"` 和 `"arguments"` 键
4. `"name"` 的值必须在可用函数集 F 中
5. `"arguments"` 对象的所有键必须是该函数定义键的子集

**2. Function Call Reward (Rfc)**

基于 **Intersection over Union (IoU)** 原理，比较预测的函数调用序列 P = {p1, ..., pm} 和真实序列 G = {g1, ..., gn}：

```
Rfc = Σi=1^min(m,n) sim(pi, gσ(i)) / (|P| + |G| - |P ∩ G|)  --- (7)
```

**变量说明**：
- `σ`：贪婪匹配方案，建立 P 和 G 元素的一一对应（见 Algorithm 2）
- `sim(p, g)`：预测调用 p 和真实调用 g 之间的参数级相似度

**参数级相似度**：
```
sim(p, g) = Σk∈keys(p)∩keys(g) s(pk, gk) / |keys(p) ∪ keys(g)|  --- (8)
```

`s(pk, gk)` 的定义取决于参数类型：
- **String**: ROUGE-L F1 score
- **Numeric/Boolean**: 精确匹配（相等为1，否则0）
- **其他**: 字符串转换后的精确匹配

**3. Response Reward (Rresponse)**

对于纯文本响应（不调用函数）：
```
Rresponse = ROUGE-L(p, g)  --- (9)
```
其中 `p` 是预测响应，`g` 是真实响应。

**4. Total Reward**

```
R = (Rformat - 1) + Rformat · (Rfc + Rresponse)  --- (10)
```

**结构解释**：
- 如果格式错误（Rformat = 0）：`R = -1`（强惩罚）
- 如果格式正确（Rformat = 1）：`R = Rfc + Rresponse`（评估内容正确性）

总奖励范围是 **[-1, 1]**。

#### 2.2.2 优化方法：GRPO

使用 **GRPO (Group Relative Policy Optimization)** 作为 RL 算法：

```
𝒥GRPO(θ) = 𝔼(q,a)~D,{oi}i=1^G~πθold(·|q) [
  (1/G) Σi=1^G (1/|oi|) Σt=1^|oi| (
    min(ri,t(θ) Âi,t, clip(ri,t(θ), 1-ε, 1+ε) Âi,t)
    - β DKL(πθ || πref)
  )
]  --- (11)
```

**变量说明**：
- `θ`：策略参数
- `q`：查询/提示
- `a`：行动/响应
- `G`：每组 rollout 数量（论文中设为 8）
- `oi`：第 i 个 rollout（token 序列）
- `ri,t(θ) = πθ(ai,t|q, ai,<t) / πθold(ai,t|q, ai,<t)`：概率比率
- `Âi,t`：优势估计
- `ε`：剪切参数（通常 0.1-0.2）
- `β`：KL 散度系数
- `πref`：参考策略（SFT 后的模型）

**优势计算**（通过奖励标准化）：
```
Âi,t = (ri - mean({Ri}i=1^G)) / std({Ri}i=1^G)  --- (12)
```

**启发式过滤**：如果组内所有 rollout 完全正确或完全错误（mean 为 1 或 0），优势为 0，丢弃该组以避免浪费计算。

---

### 2.3 STAR 训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAR Training Curriculum                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐      ┌─────────────────┐                   │
│  │   Teacher (8B)  │      │   Student (0.6B) │                   │
│  │   Qwen3-8B      │      │   Qwen3-0.6B    │                   │
│  └────────┬────────┘      └────────┬────────┘                   │
│           │                         │                            │
│           │  Sim-RL Refinement      │                            │
│           ▼                         ▼                            │
│  ┌─────────────────┐                                           │
│  │ Refined Teacher│                                           │
│  │ (Sim-RL trained)│                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           │  CKD Distillation                                   │
│           │  (ℒCKD = ℒFKL-k + λtail·ℒtail)                    │
│           ▼                                                     │
│  ┌─────────────────┐      ┌─────────────────┐                   │
│  │ Distilled       │─────▶│ Student         │                   │
│  │ Student (0.6B)  │      │ (CKD-trained)   │                   │
│  └─────────────────┘      └────────┬────────┘                   │
│                                    │                            │
│                                    │  Sim-RL Refinement        │
│                                    ▼                            │
│                           ┌─────────────────┐                   │
│                           │ STAR Model      │                   │
│                           │ (CKD+Sim-RL)    │                   │
│                           └─────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**三阶段训练**：

1. **Teacher Refinement**: 用 Sim-RL 微调教师模型（Qwen3-8B）适应蒸馏数据集
2. **Model Distillation (CKD)**: 用 CKD 将教师知识蒸馏到学生模型（Qwen3-0.6B）
3. **Model Refinement (Sim-RL)**: 用 Sim-RL 进一步优化学生模型

---

## 3. 实验结果

### 3.1 主要结果

**BFCLv3 Benchmark (0.6B 模型)**：

| Method | Overall Acc | Non-Live Acc | Live Acc | Multi Turn Acc |
|--------|-------------|--------------|----------|-----------------|
| Base-model | 47.33 | 71.81 | 65.66 | 1.88 |
| SFT | 44.58 | 66.29 | 62.15 | 1.62 |
| SFT-think | 47.59 | 71.54 | 64.46 | 4.50 |
| FKL | 49.51 | 76.44 | 65.93 | 5.12 |
| ToolRL | 47.35 | 64.81 | 66.55 | 6.75 |
| LUFFY | 49.23 | 76.75 | 64.59 | 5.48 |
| GKD | 47.32 | 67.62 | 67.61 | 3.25 |
| **CKD** | 49.84 | 75.92 | 66.15 | 5.62 |
| **Sim-RL** | 49.35 | 75.21 | 67.39 | 3.25 |
| **CKD+Sim-RL (STAR)** | **51.70** | **78.65** | **68.19** | **7.00** |

**ACEBench Normal (0.6B 模型)**：

| Method | Summary | Atom | Single-Turn | Multi-Turn | Similar API | Preference |
|--------|---------|------|-------------|------------|-------------|------------|
| Base-model | 27.20 | 37.70 | 19.50 | 10.00 | 36.00 | 6.00 |
| SFT | 2.10 | 1.70 | 0.50 | 0.00 | 14.00 | 0.00 |
| FKL | 36.80 | 52.30 | 16.00 | 16.00 | 42.00 | 22.00 |
| LUFFY | 44.40 | 59.30 | 26.50 | 26.00 | 50.00 | 22.00 |
| GKD | 40.10 | 54.00 | 21.50 | 23.00 | 46.00 | 22.00 |
| **CKD** | 39.00 | 55.00 | 21.00 | 19.00 | 48.00 | 10.00 |
| **CKD+Sim-RL (STAR)** | **53.00** | **69.30** | **35.00** | **32.00** | **62.00** | 20.00 |

**关键观察**：
- STAR (CKD+Sim-RL) 在 BFCLv3 上达到 **51.70**，相对增益 **9.2%**
- STAR 在 ACEBench 上达到 **53.00**，相对增益超过 **50%**
- SFT 在 ACEBench 上性能崩溃（从 27.20 降到 2.10），因为模型过度拟合 JSON 格式，无法适应 Python 风格的函数调用语法
- STAR 展现了强大的泛化能力，即使训练数据格式不同

### 3.2 跨尺度性能

| Model | Size | BFCLv3 Overall | ACEBench Normal |
|-------|------|----------------|------------------|
| STAR-0.6B | 0.6B | **51.70** | **53.00** |
| Qwen3-0.6B | 0.6B | 47.33 | 27.20 |
| Llama3.1-8B | 8B | 49.57 | 46.60 |
| STAR-0.6B **超过** Llama3.1-8B | ✓ | ✓ |
| STAR-1.7B | 1.7B | 56.05 | 60.90 |
| STAR-4B | 4B | 65.24 | 74.10 |
| Qwen3-8B | 8B | 66.34 | 72.90 |
| STAR-4B **接近** Qwen3-8B | ✓ | **超过** |

**核心结论**：STAR-0.6B 在 ACEBench 上超过了 Llama3.1-8B，STAR-4B 在 ACEBench 上超过了 Qwen3-8B！

### 3.3 消融研究

**KD 策略比较**：

| Method | BFCLv3 (w/o RL) | BFCLv3 (w/ RL) | ACEBench (w/o RL) | ACEBench (w/ RL) |
|--------|-----------------|----------------|-------------------|------------------|
| CE | 47.59 | 50.41 | 28.70 | 38.90 |
| FKL | 49.51 | 51.46 | 36.80 | 50.00 |
| RSKD | 49.03 | 50.65 | 35.40 | 49.80 |
| RKL | 49.26 | 50.49 | 35.30 | 41.30 |
| AKL | 49.47 | 50.29 | 44.20 | 49.00 |
| **CKD** | **49.56** | **51.70** | **39.00** | **53.00** |

**Reward 设计比较**：

| Method | BFCLv3 Overall | ACEBench Normal |
|--------|----------------|-----------------|
| CKD + Binary Reward | 51.05 | 35.70 |
| CKD + ToolRL | 48.59 | 40.50 |
| CKD + SwiRL | 51.10 | 40.30 |
| **CKD + Sim-RL** | **51.70** | **53.00** |

**关键发现**：
- Binary Reward 过于僵硬，在 ACEBench 上表现差
- Sim-RL 提供细粒度的连续奖励信号，显著改善泛化能力

### 3.4 超参数敏感性

| k | λtail | BFCLv3 (w/o RL) | BFCLv3 (w/ RL) | ACEBench (w/o RL) | ACEBench (w/ RL) |
|---|-------|-----------------|----------------|-------------------|------------------|
| 10 | 10 | 49.58 | 51.48 | 43.20 | 49.20 |
| **100** | **10** | **49.56** | **51.70** | **39.00** | **53.00** |
| 1000 | 10 | 49.84 | 51.59 | 36.70 | 52.20 |

**分析**：
- k 太小（10）过度约束学生，损害泛化
- k 太大（1000）接近标准 FKL，减少 tail penalty 的影响
- k = 100 提供了良好的平衡

---

## 4. 案例研究：Sim-RL vs Binary Reward

### 案例 1：缺少默认参数

| Function | check_wordpress |
|----------|-----------------|
| Query | "Can you check if https://example.com is running WordPress?" |
| Model Rollout | {"name": "check_wordpress", "arguments": {"url": "https://example.com"}} |
| Ground Truth | {"name": "check_wordpress", "arguments": {"url": "https://example.com", "user_agent": "Mozilla/5.0"}} |
| Binary RL Score | **0** (Mismatch) |
| Sim-RL Score | **0.5** (正确函数和主要参数的部分信用) |

### 案例 2：琐碎的格式差异

| Function | label_template_brands |
|----------|-----------------------|
| Query | "Can you list the brands available for A4 size blank label sheets?" |
| Model Rollout | {"name": "label_template_brands", "arguments": {"format": "a4"}} |
| Ground Truth | {"name": "label_template_brands", "arguments": {"format": "A4"}} |
| Binary RL Score | **0** (Mismatch) |
| Sim-RL Score | **1.0** (ROUGE-L 不区分大小写) |

### 案例 3：Reward Hacking（冗余工具调用）

| Context | 之前的对话中模型已经查询了 "SFO" 机场的信息 |
|---------|----------------------------------------|
| Query | "What is the ICAO code for SFO airport, and how many runways does it have?" |
| Model Rollout | <tool_call> {"name": "airportstatistics", "arguments": {"iata": "SFO"}} </tool_call> |
| Ground Truth | "The ICAO code for SFO is KSFO, and it has 4 runways." |
| SwiRL Score | **1.0** (奖励看起来有效的工具调用，忽略上下文) |
| Sim-RL Score | **0.0** (惩罚相比最优响应的不必要调用) |

---

## 5. 理论深度分析

### 5.1 为什么 KD+RL 比 SFT+RL 更适合超小模型？

**SFT 的问题**：
- 硬标签监督：强迫小模型学习特定输出模式
- 容量有限：容易过拟合到训练数据的格式
- 初始化差：RL 起点质量低，限制优化潜力

**KD 的优势**：
- 软标签监督：学生模仿教师的完整概率分布
- 学习不确定性：学习教师的推理和不确定性
- 更好的初始化：为 RL 提供更强的基础

**直觉类比**：
- SFT 就像给学生"标准答案"，学生死记硬背
- KD 就像让学生看到老师的"思考过程"，理解为什么这样答

### 5.2 CKD 的关键洞察：Pass@k 指标

论文强调 **Pass@k** 作为模型潜力的重要指标：
- 衡量模型生成多样化正确解的能力
- 高 Pass@k 意味着模型有更丰富的策略空间
- RL 需要探索能力，高熵是前提

CKD 通过重平衡学习信号：
1. 保留教师的 top-k 概率
2. 引入目标化抑制项，惩罚"置信但错误"的 logits
3. 在 RL 开始时提供更高的策略熵

### 5.3 RKL vs FKL 的数学直觉

**KL 散度的不对称性**：
```
DKL(P || Q) = Σx P(x) log[P(x) / Q(x)]
DKL(Q || P) = Σx Q(x) log[Q(x) / P(x)]
```

- **DKL(P || Q) (FKL)**：如果 P 赋予某点概率但 Q 没有，惩罚很大 → 鼓励 Q 覆盖 P 的支持
- **DKL(Q || P) (RKL)**：如果 Q 赋予某点概率但 P 没有，惩罚很大 → 鼓励 Q 专注于 P 的高概率区域

**在 Function Calling 中的含义**：
- FKL：鼓励学生学习教师认为"可能"的所有解
- RKL：鼓励学生只专注于教师认为"最可能"的解

**问题**：RKL 过度修剪分布，降低熵，不利于 RL 探索。

---

## 6. 架构图解析

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STAR Framework                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────┐      ┌──────────────────────────────────────────────┐ │
│  │   Problem Space   │      │          Solution Space                      │ │
│  ├───────────────────┤      ├──────────────────────────────────────────────┤ │
│  │                   │      │                                              │ │
│  │  • Overfitting    │      │  ┌────────────────────────────────────────┐  │ │
│  │  • Instability    │      │  │   CKD: Constrained Knowledge            │  │ │
│  │  • Binary Rewards │      │  │   Distillation                           │  │ │
│  │                   │      │  │                                        │  │ │
│  │                   │      │  │  ℒCKD = ℒFKL-k + λtail·ℒtail          │  │ │
│  │                   │      │  │                                        │  │ │
│  │                   │      │  │  • Stable FKL foundation              │  │ │
│  │                   │      │  │  • Targeted tail suppression           │  │ │
│  │                   │      │  │  • Preserve exploration capacity       │  │ │
│  │                   │      │  └────────────────────────────────────────┘  │ │
│  │                   │      │                    ↓                        │ │
│  │                   │      │  ┌────────────────────────────────────────┐  │ │
│  │                   │      │  │   Sim-RL: Similarity-guided RL         │  │ │
│  │                   │      │  │                                        │  │ │
│  │                   │      │  │  R = (Rformat - 1) + Rformat·(Rfc+Rres)│ │ │
│  │                   │      │  │                                        │  │ │
│  │                   │      │  │  • Fine-grained similarity reward     │  │ │
│  │                   │      │  │  • Handle multiple valid solutions    │  │ │
│  │                   │      │  │  • GRPO optimization                  │  │ │
│  │                   │      │  └────────────────────────────────────────┘  │ │
│  └───────────────────┘      └──────────────────────────────────────────────┘ │
│                                                                             │
│                        Training Curriculum                                  │
│                                                                             │
│  Teacher(8B) ──Sim-RL──▶ RefinedTeacher ──CKD──▶ DistilledStudent ──Sim-RL──▶ │
│                                                         STAR Model          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 相关工作

| 领域 | 相关方法 | STAR 的创新点 |
|------|----------|--------------|
| **LLM for Function Calling** | Gorilla, ToolACE, xLAM | 针对**超小模型**的专门训练框架 |
| **Knowledge Distillation** | MiniLLM, RSKD | **CKD**：稳定性 + 探索能力保留 |
| **Reinforcement Learning** | PPO, GRPO, DAPO | **Sim-RL**：细粒度相似度奖励 |

---

## 8. 局限性与未来工作

### 当前局限

1. **任务范围**：仅在 function calling 上验证，但框架可能泛化到其他任务（如 SQL 生成、数学推理）
2. **相似度度量**：探索了相似度引导的奖励，但可能有更复杂的相似度度量

### 未来方向

1. **多教师策略**：探索使用多个教师模型的集成
2. **更丰富的奖励设计**：探索更细粒度的反馈
3. **部署感知约束**：考虑实际部署的限制

---

## 9. 直觉总结

### 核心思想类比

想象你在教一个**初学者**（0.6B 模型）下国际象棋：

**传统 SFT 方法**：
- 给初学者"开局库"和"残局定式"
- 初学者死记硬背，遇到陌生开局就懵了
- 即使学了很多，也不理解背后的原理

**传统 RL 方法**：
- 直接让初学者通过输赢来学习
- 初学者随机尝试，大部分时间输，学习极其缓慢
- 容易陷入局部最优（只学会某些特定开局）

**STAR 方法**：

1. **Teacher Refinement**：先让一位大师（8B 模型）在特定棋局上练得更好

2. **CKD Distillation**：
   - 不是给初学者"标准答案"，而是让大师展示他的"思考概率"
   - "我觉得这步棋有 60% 可能，那步棋有 30% 可能..."
   - 同时警告初学者："如果你觉得某步棋很确定，但我根本没考虑过，那你要小心！"
   - 这样初学者学会了大师的思考模式，同时保持探索空间

3. **Sim-RL Refinement**：
   - 不是简单的"赢=1，输=0"
   - 而是：你的棋和大师的棋"相似度"是多少？
   - 即使不完全一样，如果是合理的变化，也给部分奖励
   - 这样初学者学习了更多有效的策略

**结果**：初学者虽然只有有限的"计算资源"（模型容量），但通过正确的方法，达到了接近大师的水平！

---

## 10. 技术要点总结

### CKD 公式汇总

```
ℒCKD = ℒFKL-k + λtail·ℒtail

其中：
ℒFKL-k = Σx∈D Σv∈Vk(x) PT(v|x) log[PT(v|x) / PS(v|x)]
ℒtail = Σx∈D Σv∈Vm(x)\Vk(x) PS(v|x)

梯度：
∂ℒCKD/∂zS_j = qj·[Σi∈Ik pi + λ(1-Σi∈J'm qi)] - pj  (j ∈ J'm)
∂ℒCKD/∂zS_j = qj·(Σi∈Ik pi - λΣi∈J'm qi)        (j ∉ Ik ∪ J'm)
```

### Sim-RL 奖励汇总

```
总奖励：R = (Rformat - 1) + Rformat·(Rfc + Rresponse)

-1 ≤ R ≤ 1

Rformat ∈ {0, 1}：格式正确性
Rfc：IoU 风格的函数调用相似度
Rresponse：ROUGE-L F1（纯文本响应）
```

---

## 11. 实际应用意义

STAR 证明了：
1. **有效训练框架**可以将大模型能力转移到超小模型
2. **0.6B 模型**可以在 function calling 上达到甚至超过 **8B 模型**
3. 为**端侧部署**和**大规模服务**提供了可行的解决方案

---

## 参考链接

- **论文**: https://arxiv.org/html/2602.03022v1
- **代码**: https://github.com/Qwen-Applications/STAR
- **BFCL Benchmark**: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- **ACEBench**: https://github.com/chenchen0103/ACEBench
- **Qwen Models**: https://arxiv.org/abs/2505.09388
- **GRPO**: https://arxiv.org/abs/2402.03300
- **MiniLLM (KD)**: https://openreview.net/forum?id=5h0qf7IBZZ
- **DAPO**: https://arxiv.org/abs/2503.14476