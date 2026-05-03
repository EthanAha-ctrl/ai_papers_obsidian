## 1. 论文概述

2026年1月23日 字节跳动 Seed 团队

**核心贡献**：提出了 **Stable-DiffCoder**，这是一个基于块扩散的代码语言模型，证明了在相同数据和架构条件下，扩散训练范式可以超越自回归训练。

**项目链接**：
- GitHub: https://github.com/ByteDance-Seed/Stable-DiffCoder
- HuggingFace: https://huggingface.co/collections/ByteDance-Seed/stable-diffcoder

--- 

## 2. 背景与动机

### 2.1 自回归模型 (AR Models) 的局限性

自回归模型通过从左到右建模序列，其联合分布因式分解为：

$$p_\theta(x_{1:N}) = p_\theta(x_1) \prod_{i=2}^N p_\theta(x_i|x_{<i})$$

**训练损失函数**：
$$L_{AR}(\theta) = -\mathbb{E}_{x \sim p_{data}} \left[ \log p_\theta(x_1) + \sum_{i=2}^N \frac{\log p_\theta(x_i|x_{<i})}{i} \right]$$

**局限性**：
- 严格的顺序解码无法利用代码的非自回归本质
- 缺乏全局规划和并行推理能力
- 计算密集型，无法并行生成多个代码块

### 2.2 扩散语言模型 (DLLMs) 的优势

**前向噪声过程**：通过 [MASK] token 逐步替换原始 token

$$h(\hat{x}_i|x^{UM}_t) = \alpha(t) p_0(\hat{x}_i|x^{UM}_t)$$

其中 $x^{UM}_t$ 表示时间步 t 的未屏蔽 token。

**扩散目标函数**：
$$L_{DLLM}(\theta) = -\mathbb{E}_{x_0 \sim p_{data}, t \sim U(0,1), x_t \sim q(x_t|x_0)} \left[ w(t) \sum_{i=1}^N \mathbf{1}[x_t^i = \text{MASK}] \log p_\theta(x_0^i|x_t^{1:N}) \right]$$

**优势**：
- 非顺序、块级别生成
- 通过随机掩码提供丰富的数据增强
- 特别适合处理高质量但稀缺的代码样本

---

## 3. 核心理论分析：Token推理知识

### 3.1 问题定义

给定干净训练序列 $x$ 和用于预测单个 token $x_i$ 的上下文 $c$，定义候选集：

$$C(c) = \{ v \in V : p_0(v|c) \geq \varepsilon \}, \quad K(c) = |C(c)|$$

**Token推理知识**：上下文 c 中所包含的关于 $x_i$ 的条件分布，即 $p_0(C(c)|c)$。

### 3.2 知识学习三阶段

| 阶段 | K(c) 大小 | 特征 |
|------|-----------|------|
| **推理阶段** | 小 | 映射 c→x_i 几乎确定性，梯度高度一致 |
| **关联阶段** | 中等/大 | x_i 是多个候选之一，梯度部分抵消 |
| **噪声阶段** | 很大 | p₀(·|c) 接近均匀，只能记忆特定配对 |

### 3.3 问题示例

考虑干净序列：
```
a = 1, b = 2, a + b = 3; a = 3, b = 4, a + b = 7.
```

如果重度掩码中间和尾部部分得到：
```
a = 1, b = 2, [MASK₁] ... [MASK₂] a + b = [MASKₙ]
```

**问题**：模型从这种掩码视图学习到的是一些数字倾向于出现在 `a + b =` 之后，以及 `(a=3, b=4, 7)` 的共现模式，而不是正确的加法规则！

**推理阶段示例**：
```
a = 1, b = 2, a + b = [MASK]
```
上下文强约束答案，模型可以稳定学习到推理规则。

### 3.4 训练-推理对齐原则

为有效学习新知识并确保数据增强有效性，需要满足两个条件：

1. **清洁推理证据原则**：模型应暴露于清洁可靠的推理证据
2. **上下文对齐原则**：$C_{train}$ 和 $C_{infer}$ 应尽可能接近

对于块大小为 B 的块扩散和左到右块解码：推理上下文与训练上下文高度匹配。

---

## 4. 训练课程设计

### 4.1 2.5B规模实验设置

从通用域数据训练的 2.5B AR 模型开始，使用代码数据作为新的 CPT 数据，考虑以下三种课程策略：

#### 策略对比

| 策略 | 描述 | 性能排名 |
|------|------|----------|
| (1) AR→BiDLLM | 纯 AR 训练 → 双向 DLLM | ★★★★★ |
| (2) ARDLLM→BiDLLM | 因果式 DLLM → 双向 DLLM | ★★★★☆ |
| (3) BiDLLM | 直接 CPT 到双向 DLLM | ★★★☆☆ |

**关键发现**：
- 小块解码（block size 1 或 2）：策略 (1) 表现最佳
- 大块解码（block size 32）：策略 (3) 表现更好
- 策略 (2) 在 block-32 解码下表现优于策略 (3)

### 4.2 推荐训练流程（三层课程架构）

```
1. [基础层] 新知识 → AR 训练 → 高效压缩
      ↓
2. [增强层] 小块扩散 CPT → 数据增强特性 → 提升模型质量
      ↓
3. [探索层] 可选：大块扩散 CPT → 扩展训练模式
```

**架构图**：
```
┌─────────────────────────────────────────────────┐
│           Pretraining Phase                     │
│  (General Domain: 2.5B AR Pre-trained Model)    │
└───────────────────┬─────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────┐
│      Code Continual Pretraining (CPT)           │
│  ┌───────────────────────────────────────────┐  │
│  │  Step 1: Code Data AR Training           │  │
│  │       (Efficient Knowledge Compression)  │  │
│  └───────────────┬───────────────────────────┘  │
│                  ↓                                │
│  ┌───────────────────────────────────────────┐  │
│  │  Step 2: Small-Block Diffusion (B=4)     │  │
│  │       (Data Augmentation)                │  │
│  │       - Tailored Warmup                │  │
│  │       - Block-wise Clipped Noise Schedule│  │
│  └───────────────────────────────────────────┘  │
└───────────────────┬─────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────┐
│         Supervised Fine-tuning (SFT)            │
│    (Instruction Tuning with same dataset)       │
└─────────────────────────────────────────────────┘
```

---

## 5. 核心技术创新

### 5.1 Warmup 机制

**问题**：AR→DLLM 的 CPT 过程存在严重的不稳定性：
- (i) 注意力掩码变化导致内部表示的结构分布偏移
- (ii) 当噪声过程掩码大量 token 时任务难度更高
- (iii) ELBO 启发的损失权重 w(t) 在低掩码比率时可达到 10×

**Warmup 设计**：

```
腐蚀水平采样：t ∼ U(0, u_max(s))
u_max(s) = u_init + (1 - u_init) × (s / S_warmup)
```

其中：
- u_init = 10⁻³（初始值）
- S_warmup：warmup 步数
- s = 0, 1, ..., S_warmup

**Warmup 期损失函数**（移除 w(t)）：
$$L_{warmup}^{DLLM}(\theta) = -\mathbb{E}_{x_0 \sim p_{data}, t \sim U(0, u_{max}), x_t \sim q(x_t|x_0)} \left[ \sum_{i=1}^N \mathbf{1}[x_t^i = \text{MASK}] \log p_\theta(x_0^i|x_t^{1:N}) \right]$$

**效果对比**：

```
无 Warmup：                      有 Warmup：
┌─────────────────┐            ┌─────────────────┐
│ 梯度范数峰值极高 │     →      │ 梯度范数稳定下降 │
│ 训练损失剧烈波动 │            │ 平滑的√形曲线   │
│ 收敛不稳定       │            │ 收敛稳定且快速  │
└─────────────────┘            └─────────────────┘
```

### 5.2 Block-wise Clipped Noise Scheduling

**问题**：对于小块大小 B，许多训练步骤产生弱或零学习信号。

**数学推导**：

在时间步 t，大小为 B 的块中期望掩码 token 数：
$$E[m|t] = B \cdot u(t)$$

无掩码 token 的概率：
$$\Pr[m=0|t] = (1-u(t))^B$$

在标准全局线性调度 $u(t) = 1-t$，$t \sim \text{Unif}[0,1]$ 下：

$$E_t[(1-t)^B] = \int_0^1 (1-t)^B dt = \frac{1}{B+1}$$

**典型大小**：
- B=2: $1/3 ≈ 33.3\%$ 的步骤无效
- B=4: $1/5 = 20\%$ 的步骤无效  
- B=8: $1/9 ≈ 11.1\%$ 的步骤无效

**解决方案**：块感知裁剪掩码率

$$u_{blk}(t) = \min\{1, \max(u(t), 1/B)\}$$

**保底规则**：如果采样后块中无 token 被掩码，强制均匀采样一个位置进行掩码。

**保证**：
$$E[m|t] = B \cdot u_{blk}(t) \geq 1, \quad \forall t$$

同时确保损失权重：
$$w(t) = \frac{1}{u_{blk}(t)} \leq B$$

防止梯度爆炸。

---

## 6. 实验设置

### 6.1 模型配置

| 参数 | 配置 |
|------|------|
| 基础架构 | Seed-Coder（复用） |
| 模型规模 | 8B 参数 |
| 上下文长度 | 8192 tokens |
| 块大小 (Block Size) | 4 |
| CPT 数据量 | 1.3T tokens |
| Logit Shift | 无（与 absorbing diffusion 一致） |
| Attention | Packed sequences，样本间相互可见 |

### 6.2 打包策略

**CPT 阶段**：
- 使用 packed sequences
- 样本间 attention 相互可见（避免 flex attention 重复编译）

**SFT 阶段**：
- 保持相同打包策略
- 每个样本后随机追加 1-4 个 `<eos>` tokens
- 保持变长输出生成能力

### 6.3 评估基准

代码生成基准：
- HumanEval / HumanEval+
- MBPP / MBPP+
- MultiPL-E (8 种语言)
- BigCodeBench (Full / Hard)
- LiveCodeBench (v5)
- MBXP (13 种语言)
- NaturalCodeBench

代码推理基准：
- CRUXEval (Input-CoT / Output-CoT)

代码编辑基准：
- CanItEdit
- Aider

---

## 7. 实验结果详解

### 7.1 基础模型性能：HumanEval(+) 和 MBPP(+)

**表1关键数据**：

| 模型 | 类型 | 规模 | HumanEval | HE+ | MBPP | MBPP+ |
|------|------|------|-----------|-----|------|-------|
| StarCoder2-7B | AR | 7B | 35.4 | 29.9 | 54.4 | 45.6 |
| DeepSeek-Coder-6.7B | AR | 6.7B | 47.6 | 39.6 | 70.2 | 56.6 |
| Qwen2.5-Coder-7B | AR | 7B | 72.0 | 67.1 | 79.4 | 68.3 |
| **Seed-Coder-8B** | **AR** | **8B** | **77.4** | **68.3** | **82.0** | **69.0** |
| LLaDA-8B | DLLM | 8B | 35.4 | 30.5 | 50.1 | 42.1 |
| Dream-7B | DLLM | 7B | 56.7 | 50.0 | 68.7 | 57.4 |
| DiffuCoder-7B | DLLM | 7B | 67.1 | 60.4 | 74.2 | 60.9 |
| Dream-Coder-7B | DLLM | 7B | 66.5 | 60.4 | 75.9 | 61.6 |
| WeDLM-8B | DLLM | 8B | 75.0 | 68.9 | 67.0 | - |
| **Stable-DiffCoder-8B** | **DLLM** | **8B** | **79.3** | **73.8** | **83.6** | **67.7** |

**相对提升**（对比 Seed-Coder-8B）：
- HumanEval: $79.3 - 77.4 = +1.9$ 分 (+2.5%)
- HumanEval+: $73.8 - 68.3 = +5.5$ 分 (+8.1%) ⭐
- MBPP: $83.6 - 82.0 = +1.6$ 分 (+2.0%)
- MBPP+: $67.7 - 69.0 = -1.3$ 分 (-1.9%)

### 7.2 多语言性能：MultiPL-E

**表2关键数据 (8B 模型)**：

| 模型 | Python | C++ | Java | PHP | TS | C# | Bash | JS | 平均 |
|------|--------|-----|------|-----|----|-----|------|----|----|
| StarCoder2-7B | 35.4 | 40.4 | 38.0 | 30.4 | 34.0 | 46.2 | 13.9 | 36.0 | 34.3 |
| Qwen2.5-Coder-7B | 72.0 | 62.1 | 53.2 | 59.0 | 64.2 | 60.8 | 38.6 | 60.3 | 58.8 |
| Seed-Coder-8B | 77.4 | 69.6 | 72.8 | 63.9 | 77.4 | 53.8 | 48.1 | 77.6 | 67.6 |
| **Stable-DiffCoder-8B** | **80.5** | **69.4** | **74.1** | **74.4** | **74.8** | **70.3** | **53.2** | **73.1** | **71.2** |

**显著优势**：
- PHP: $74.4 - 63.9 = +10.5$ 分 (+16.4%) ⭐⭐⭐
- C#: $70.3 - 53.8 = +16.5$ 分 (+30.7%) ⭐⭐⭐
- 平均: $71.2 - 67.6 = +3.6$ 分 (+5.3%)

**分析**：在低资源语言（PHP、C#）上，扩散风格的数据增强效果显著！

### 7.3 代码推理性能：CRUXEval

**表3关键数据 (8B 模型)**：

| 模型 | Input-CoT | Output-CoT |
|------|-----------|------------|
| DeepSeek-Coder-6.7B | 39.0 | 41.0 |
| OpenCoder-8B | 43.3 | 43.9 |
| Qwen2.5-Coder-7B | 56.5 | 56.0 |
| **Seed-Coder-8B** | **52.0** | **54.8** |
| **Stable-DiffCoder-8B** | **53.8** | **60.0** |

**提升**：
- Input-CoT: $53.8 - 52.0 = +1.8$ 分 (+3.5%)
- Output-CoT: $60.0 - 54.8 = +5.2$ 分 (+9.5%) ⭐

**分析**：引入适度的随机掩码目标有效增强了模型的推理能力。

### 7.4 指令调优模型：关键指标汇总

**表4核心模型**：

| 模型 | 类型 | 规模 | HumanEval | HE+ | MBPP | MBPP+ |
|------|------|------|-----------|-----|------|-------|
| Qwen2.5-Coder-7B-Instruct | AR | 7B | 88.4 | 84.1 | 83.5 | 71.7 |
| Qwen3-8B-Instruct | AR | 8B | 84.8 | 80.5 | 77.0 | 67.2 |
| Seed-Coder-8B-Instruct | AR | 8B | 84.8 | 78.7 | 85.2 | 71.2 |
| **Stable-DiffCoder-8B-Instruct** | **DLLM** | **8B** | **86.6** | **82.3** | **85.7** | **72.8** |

**与 Seed-Coder 对比**：
- HumanEval+: $82.3 - 78.7 = +3.6$ 分 (+4.6%)
- MBPP+: $72.8 - 71.2 = +1.6$ 分 (+2.2%)

**表5：高难度基准**：

| 模型 | MHPP | BigCodeBench-Full | BigCodeBench-Hard | LiveCodeBench |
|------|------|-------------------|-------------------|---------------|
| Qwen2.5-Coder-7B-Instruct | 26.7 | 48.8 | 20.3 | 17.3 |
| Qwen3-8B-Instruct | 32.8 | 51.7 | 23.0 | 23.5 |
| Seed-Coder-8B-Instruct | 36.2 | 53.3 | 26.4 | 24.7 |
| **Stable-DiffCoder-8B-Instruct** | **42.4** | **54.8** | **31.8** | **23.5** |

**重大突破**：
- MHPP: $42.4 - 36.2 = +6.2$ 分 (+17.1%) ⭐⭐⭐
- BigCodeBench-Hard: $31.8 - 26.4 = +5.4$ 分 (+20.5%) ⭐⭐⭐
- 达到 Qwen2.5-Coder-32B-Instruct 的 MHPP 水平！

### 7.5 代码编辑性能

**表9：编辑基准**：

| 模型 | CanItEdit | Aider (tries=2) |
|------|-----------|-----------------|
| Qwen2.5-Coder-7B-Instruct | 49.5 | 57.9 |
| Qwen3-8B-Instruct | 45.7 | 55.6 |
| Seed-Coder-8B-Instruct | 50.5 | 57.1 |
| **Stable-DiffCoder-8B-Instruct** | **60.0** | **54.9** |

**突出表现**：
- CanItEdit: $60.0 - 50.5 = +9.5$ 分 (+18.8%) ⭐⭐⭐

**分析**：DLLM 的去噪本质天生训练模型的编辑模式，使模型能更好地利用编辑监督。

### 7.6 性能对比图总结

从论文图1可以概括主要竞争模型在核心基准的得分（8B级别 vs 更大或外部API）：

| 基准 | Stable-DiffCoder-8B (DLLM) | OpenCoder-8B (AR) | Qwen2.5-Coder-7B (AR) | Gemini Diffusion / Mercury Coder (DLLM, 非约8B) |
|------|---------------------------|-------------------|----------------------|-------------------------------------|
| HumanEval | 86.6 (S) / 79.3 (B) | 83.5 (S) / 66.5 (B) | 88.4 (S) / 72.0 (B) | ≈89-90 (S) / 未给 (B) |
| MBPP | 85.7 (S) / 83.6 (B) | 79.1 (S) / 79.9 (B) | 83.5 (S) / 79.4 (B) | ≈77 (S) / 未给 (B) |
| BigCodeBench (Full) | 54.8 (S) | 50.9 (S) | 48.8 (S) | ≈45-46 (S) |
| BigCodeBench (Hard) | 31.8 (S) | 18.9 (S) | 20.3 (S) | ≈14-15 (S) |
| MHPP | 42.4 (S) | 30.5 (S) | 26.7 (S) | ≈18-19 (S) |
| LiveCodeBench (v5) | 23.5 (S) | 17.1 (S) | 17.3 (S) | ≈14-15 (S) |
| CanItEdit | 60.0 (S) | 39.0 (S) | 49.5 (S) | 未给出或显著低于60 |

（注：S=Instruct 模型，B=Base 模型；HumanEval/MBPP 栏对比中，HumanEval+、MBPP+ 的 S/B 版本在论文中的详细表格分别给出，此表以 HumanEval/MBPP 的 Instruct 为主基准并交叉参考各模型）

---

## 8. 技术架构图与实现细节

### 8.1 整体架构示意

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stable-DiffCoder Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Layer                                                     │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐                  │
│  │ Token   │      │ Packing │      │ Position│                  │
│  │ Embed   │─────▶│  &      │─────▶│  Encode │                  │
│  │         │      │ Context │      │         │                  │
│  └─────────┘      └─────────┘      └─────────┘                  │
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Transformer Backbone (Shared)               │   │
│  │                                                         │   │
│  │  Layer 1  ──▶ Layer 2  ──▶ ... ──▶ Layer L              │   │
│  │    │           │                     │                  │   │
│  │    ▼           ▼                     ▼                  │   │
│  │  [Self-Attention]  [Self-Attention]        ...          │   │
│  │      │                 │                                 │   │
│  │      ▼                 ▼                                 │   │
│  │  [FFN]            [FFN]                      ...          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Output Layers                          │   │
│  │                                                         │   │
│  │  LM Head No-Logit-Shift (Absorbing Diffusion Design)   │   │
│  │                                                         │   │
│  │  Mask Prediction:                                       │   │
│  │  p_θ(x_0^i | x_t^{1:N}) for each [MASK] position        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

Diffusion Training Process:
┌────────────────┐      ┌──────────────────┐      ┌────────────────┐
│  Clean Input   │      │  Forward Process │      │ Reverse Process │
│  x₀ ∈ Vⁿ       │────▶│  q(x_t | x₀)      │────▶│ p_θ(x₀ | x_t)   │
│                │      │  Random Masking   │      │  Denoising     │
└────────────────┘      └──────────────────┘      └────────────────┘
                               │                         │
                               ▼                         ▼
                        Block-wise               Block-wise
                      Clipped Noise         Decoding Order
                         Schedule              (B=4 tokens)
```

### 8.2 注意力掩码模式对比

**AR 模型（因果下三角掩码）**：
```
     t₀   t₁   t₂   t₃   t₄
t₀   ✓    ✗    ✗    ✗    ✗
t₁   ✓    ✓    ✗    ✗    ✗
t₂   ✓    ✓    ✓    ✗    ✗
t₃   ✓    ✓    ✓    ✓    ✗
t₄   ✓    ✓    ✓    ✓    ✓
```

**Bidirectional DLLM（完全可见掩码）**：
```
     t₀   t₁   t₂   t₃   t₄
t₀   ✓    ✓    ✓    ✓    ✓
t₁   ✓    ✓    ✓    ✓    ✓
t₂   ✓    ✓    ✓    ✓    ✓
t₃   ✓    ✓    ✓    ✓    ✓
t₄   ✓    ✓    ✓    ✓    ✓
```

**Block Diffusion（块扩散，B=4示例）**：
```
     t₀   t₁   t₂   t₃   t₄   t₅   t₆   t₇
t₀   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓  ── Context
t₁   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓  ── Context
t₂   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓  ── Context
t₃   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓  ── Context
t₄   [M]  [M]  [M]  [M]  ✓    ✓    ✓    ✓  ──扩散块 (4 tokens)
t₅   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓  ── Context
t₆   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓  ── Context
t₇   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓  ── Context
```
*注：[M] 表示被掩码的 token*

### 8.3 No-Logit-Shift 设计

与传统吸收扩散一致性，每个掩码位置预测自身：

**传统 Logit-Shift**：
$$\text{logits}' = \text{logits} - \log P_{\text{shift}}$$

**No-Logit-Shift**：
$$\text{logits}' = \text{logits}$$

**优势**：
- 输入和预测目标在 token 和句子层面都能对齐
- 更符合吸收扩散范式
- 训练更稳定

---

## 9. 实验详细分析

### 9.1 训练稳定性对比

**图4分析**（根据论文所述）：

| 配置 | 初始行为 | 中期行为 | 最终结果 |
|------|----------|----------|----------|
| BiDLLM 无 Warmup | 损失极高，梯度尖峰 | 波动剧烈 | 收敛困难 |
| BiDLLM 有 Warmup | 损失平滑上升 | √形曲线 | 稳定收敛 |
| BlockDLLM 有 Warmup (no-shift) | 渐进式适应 | 平滑下降 | 与 AR CPT 相当 |

---

## 10. 局限性与未来工作

### 10.1 当前局限

1. **领域限制**：主要关注代码领域，缺乏其他领域的大规模训练数据
2. **数学推理**：在数学推理和通用文本任务上性能可能相对有限
3. **上下文窗口**：Aider 任务中长上下文（>8192 tokens）性能下降

### 10.2 未来方向

1. **跨域扩展**：探索文本扩散采样在更广泛领域（数学、通用NLP）的益处
2. **更大规模**：扩展到更大模型规模（如 16B、32B）
3. **混合训练**：探索 AR + DLLM 的联合训练范式
4. **推理加速**：充分利用扩散模型的并行推理潜力

---

## 11. 方法要点与公式回顾

### 11.1 AR 损失函数
$$L_{AR}(\theta) = -\mathbb{E}_{x \sim p_{data}} \left[ \log p_\theta(x_1) + \sum_{i=2}^N \frac{\log p_\theta(x_i|x_{<i})}{i} \right]$$

### 11.2 DLLM 损失函数
$$L_{DLLM}(\theta) = -\mathbb{E}[ w(t) \sum_{i=1}^N \mathbf{1}[x_t^i = \text{MASK}] \log p_\theta(x_0^i|x_t^{1:N}) ]$$

### 11.3 Warmup 期损失函数
$$L_{warmup}^{DLLM}(\theta) = -\mathbb{E}[ \sum_{i=1}^N \mathbf{1}[x_t^i = \text{MASK}] \log p_\theta(x_0^i|x_t^{1:N}) ]$$

其中 $t \sim U(0, u_{max})$，$u_{max}(s) = u_{init} + (1-u_{init})\frac{s}{S_{warmup}}$

### 11.4 Block-wise Clipped Noise Schedule
$$u_{blk}(t) = \min\{1, \max(u(t), 1/B)\}$$

保证 $E[m|t] = B \cdot u_{blk}(t) \geq 1$，$w(t) = 1/u_{blk}(t) \leq B$

---

## 12. 性能数据表汇总

### 12.1 基础模型 vs Seed-Coder (8B) 提升

| 基准 | Seed-Coder-8B | Stable-DiffCoder-8B | 提升 | 提升率 |
|------|---------------|---------------------|------|--------|
| HumanEval | 77.4 | 79.3 | +1.9 | +2.5% |
| HumanEval+ | 68.3 | 73.8 | +5.5 | +8.1% |
| MBPP | 82.0 | 83.6 | +1.6 | +2.0% |
| MBPP+ | 69.0 | 67.7 | -1.3 | -1.9% |
| MultiPL-E (平均) | 67.6 | 71.2 | +3.6 | +5.3% |
| PHP | 63.9 | 74.4 | +10.5 | +16.4% |
| C# | 53.8 | 70.3 | +16.5 | +30.7% |
| CRUXEval-Input | 52.0 | 53.8 | +1.8 | +3.5% |
| CRUXEval-Output | 54.8 | 60.0 | +5.2 | +9.5% |

### 12.2 指令模型 vs Seed-Coder (8B) 提升

| 基准 | Seed-Coder-8B-Instruct | Stable-DiffCoder-8B-Instruct | 提升 | 提升率 |
|------|-------------------------|------------------------------|------|--------|
| HumanEval | 84.8 | 86.6 | +1.8 | +2.1% |
| HumanEval+ | 78.7 | 82.3 | +3.6 | +4.6% |
| MBPP | 85.2 | 85.7 | +0.5 | +0.6% |
| MBPP+ | 71.2 | 72.8 | +1.6 | +2.2% |
| MHPP | 36.2 | 42.4 | +6.2 | +17.1% |
| BigCodeBench-Full | 53.3 | 54.8 | +1.5 | +2.8% |
| BigCodeBench-Hard | 26.4 | 31.8 | +5.4 | +20.5% |
| CanItEdit | 50.5 | 60.0 | +9.5 | +18.8% |

### 12.3 跨模型对比（8B 规模）

| 模型 | 类型 | HumanEval | MBPP | MultiPL-E | MHPP | CanItEdit |
|------|------|-----------|------|-----------|------|-----------|
| StarCoder2-7B | AR | 35.4 | 54.4 | 34.3 | - | - |
| DeepSeek-Coder-6.7B | AR | 47.6 | 70.2 | 44.7 | 20.0 | 44.4 |
| Qwen2.5-Coder-7B | AR | 72.0 | 79.4 | 58.8 | 26.7 | 49.5 |
| Seed-Coder-8B | AR | 77.4 | 82.0 | 67.6 | 36.2 | 50.5 |
| LLaDA-8B | DLLM | 35.4 | 50.1 | - | - | 41.0 |
| Dream-7B | DLLM | 56.7 | 68.7 | - | - | 68.3 |
| DiffuCoder-7B | DLLM | 67.1 | 74.2 | - | 75.1 | - |
| WeDLM-8B | DLLM | 75.0 | 67.0 | - | - | - |
| **Stable-DiffCoder-8B** | **DLLM** | **79.3** | **83.6** | **71.2** | **42.4** | **60.0** |

---

## 13. 参考模型链接

**AR Baselines**：
- StarCoder2: https://huggingface.co/bigcode/starcoder2-7b
- DeepSeek-Coder: https://github.com/deepseek-ai/DeepSeek-Coder
- CodeQwen1.5: https://qwenlm.github.io/blog/codeqwen1.5/
- OpenCoder: https://github.com/OpenCoder-Project/OpenCoder
- Qwen2.5-Coder: https://arxiv.org/abs/2409.12186
- Seed-Coder: https://arxiv.org/abs/2506.03524

**DLLM Baselines**：
- LLaDA: https://arxiv.org/abs/2502.09992
- Dream: https://arxiv.org/abs/2506.20639
- DSD: https://arxiv.org/abs/2508.02193
- SDAR: https://arxiv.org/abs/2510.06303
- WeDLM: https://arxiv.org/abs/2512.22737
- Mercury Coder: https://www.inceptionlabs.ai/introducing-mercury
- Gemini Diffusion: https://deepmind.google/models/gemini-diffusion

**相关工作**：
- Absorbing Discrete Diffusion: https://arxiv.org/abs/2406.03736
- Block Diffusion: https://arxiv.org/abs/2503.09573
- Fast-dLLMv2: https://arxiv.org/abs/2509.26328
- Dinfer: https://arxiv.org/abs/2510.08666

---

## 总结

**Stable-DiffCoder** 的核心价值在于：

1. **理论突破**：系统分析了 DLLM 中 token 推理知识的学习机制，提出推理/关联/噪声三阶段理论

2. **训练创新**：设计了 warmup + block-wise clipped noise schedule，解决了 AR→DLLM CPT 的稳定性问题

3. **实践验证**：在相同数据和架构下，扩散训练超越 AR 训练，证明扩散采样可作为有效的数据增强方法

4. **性能领先**：在多项代码基准上达到 SOTA，特别是低资源语言（PHP、C#）、高难度推理（MHPP、BigCodeBench-Hard）和代码编辑任务

未来方向是探索这种训练范式在更广泛领域的适用性。