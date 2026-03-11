# DeepSeek-OCR 2: Visual Causal Flow

## 一、核心动机与背景

### 1.1 传统VLM的局限性问题分析

现有vision-language models (VLMs) 存在一个根本性缺陷：它们始终以**刚性的光栅扫描顺序**（从左上到右下）处理视觉tokens，并使用固定的位置编码。这与人类视觉感知机制形成鲜明对比。

**人类视觉感知的因果性特征**：
```
人眼扫描模式：
┌─────────────────────────────────────┐
│  人眼不按固定顺序扫描               │
│  而是跟随语义逻辑的因果流          │
│                                     │
│  示例：追踪螺旋时，每个注视点     │
│        都因果依赖前一个注视点      │
└─────────────────────────────────────┘
```

### 1.2 2D vs 1D 的结构性矛盾

| 维度 | 图像本质 | LLM处理能力 | 传统方法 | 问题 |
|------|----------|-------------|----------|------|
| 空间结构 | 2D网格 | 1D序列化 | 光栅扫描展开 | 引入不合理的归纳偏置 |
| 语义关系 | 多维度复杂布局 | 线性因果链 | 固定位置编码 | 忽略语义依赖 |
| 文档特征 | 非线性布局(表格/公式) | 序列推理 | 强制线性化 | 破坏逻辑结构 |

---

## 二、DeepEncoder V2 架构设计

### 2.1 整体架构图解析

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSeek-OCR 2 整体架构                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入图像 (1024×1024)                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │  Vision Tokenizer (80M参数)             │                   │
│  │  - SAM-base + 2层卷积                   │                   │
│  │  - 16×压缩比                            │                   │
│  │  - 输出: 256个visual tokens (D=896)    │                   │
│  └─────────────────────────────────────────┘                   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │  DeepEncoder V2 (Qwen2-0.5B, 500M参数)  │                   │
│  │  ┌─────────────────────────────────┐   │                   │
│  │  │ 双流注意力机制                  │   │                   │
│  │  │ ├─ Visual tokens: 双向attention│   │                   │
│  │  │ └─ Causal queries: 因果attention│   │                   │
│  │  └─────────────────────────────────┘   │                   │
│  │  输出: 256个重新排序的causal tokens     │                   │
│  └─────────────────────────────────────────┘                   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │  DeepSeek-MoE Decoder (3B参数)          │                   │
│  │  - ~500M active parameters (MoE-A570M) │                   │
│  └─────────────────────────────────────────┘                   │
│       │                                                         │
│       ▼                                                         │
│  输出 (OCR结果/文档解析)                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键创新点详解

#### **创新1: Language Model as Vision Encoder**

DeepEncoder V2 用紧凑的LLM架构替换了DeepEncoder中的CLIP组件：

```python
# 架构对比
DeepEncoder (V1):
    Vision Tokenizer → CLIP ViT (300M) → Visual Tokens

DeepEncoder V2:
    Vision Tokenizer → Qwen2-0.5B (500M) [双流注意力] → Causal Tokens
```

**为什么选择Qwen2-0.5B？**
- 参数量与CLIP ViT (300M)相当，不引入过大计算开销
- 继承LLM的因果推理能力
- 可自然继承LLM社区的优化（MoE、高效attention等）

#### **创新2: 双流注意力机制 (Dual-Stream Attention)**

这是DeepEncoder V2的核心设计，通过定制attention mask实现：

```
Attention Mask 架构:

                Visual Tokens (m个)       Causal Queries (n个)
                ┌──────────────────┐     ┌──────────────────┐
                │ 1 1 1 1 ... 1 1  │     │ 0 0 0 0 ... 0 0  │
Visual Tokens  │ 1 1 1 1 ... 1 1  │     │ 1 1 1 1 ... 1 1  │  ← 视觉tokens
(m个)          │ 1 1 1 1 ... 1 1  │     │ 1 1 1 1 ... 1 1  │     可以互相看到
                │ ...              │     │ ...              │
                │ 1 1 1 1 ... 1 1  │     │ 1 1 1 1 ... 1 1  │
                ├──────────────────┤     ├──────────────────┤
Causal Queries │ 1 1 1 1 ... 1 1  │     │ 1 1 1 1 ... 1 1  │
(n个)          │ 1 1 1 1 ... 1 1  │     │ 0 1 1 1 ... 1 1  │  ← Causal queries
                │ 1 1 1 1 ... 1 1  │     │ 0 0 1 1 ... 1 1  │     只能看到前面
                │ ...              │     │ ...              │
                │ 1 1 1 1 ... 1 1  │     │ 0 0 0 0 ... 1 1  │
                └──────────────────┘     └──────────────────┘
                  ↑                      ↑
             双向attention           因果(三角)attention
```

**Attention Mask 数学定义**：

```
M = ┌───────────────────────────────────┐
    │     1_{m×m}     │    0_{m×n}    │
    ├───────────────────────────────────┤
    │     1_{n×m}     │ LowerTri(n)   │
    └───────────────────────────────────┘

其中：
- m: visual tokens数量
- n: causal queries数量 (设计为 n = m)
- LowerTri(n): 下三角矩阵 (对角线及以下为1，以上为0)
```

#### **创新3: Causal Flow Query 机制**

每个causal query token可以：
1. **attend to 所有visual tokens**（获得全局视觉信息）
2. **attend to 前面的causal queries**（建立因果依赖链）

```
Causal Flow 传播示意:

Visual Tokens → [V₁, V₂, V₃, ..., Vₘ]
                    │    │    │         │
                    ▼    ▼    ▼         ▼
Causal Queries → [Q₁, Q₂, Q₃, ..., Qₙ]
                    │    │    │         │
                    └────┴────┴─────────┘
                    因果依赖链

Q₁ = f([V₁, V₂, ..., Vₘ])           # 可看所有visual tokens
Q₂ = f([V₁, V₂, ..., Vₘ, Q₁])       # 可看visual tokens + Q₁
Q₃ = f([V₁, V₂, ..., Vₘ, Q₁, Q₂])  # 可看visual tokens + Q₁,Q₂
...
```

**Token数量计算公式**：

```
总token数 = k × 144 + 256

其中：
- k: local crops数量 (0 ≤ k ≤ 6)
- 256: global view 产生的tokens (1024×1024分辨率)
- 144: 每个local view产生的tokens (768×768分辨率)

范围: [256, 1120] tokens
```

### 2.3 Vision Tokenizer 设计

```
Vision Tokenizer 组件:

输入图像 (1024×1024)
        │
        ▼
┌─────────────────────────┐
│  SAM-base (80M参数)      │
│  - 类似VITDET架构         │
│  - Conv 16×              │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  2层卷积层                │
│  - 进一步特征提取          │
│  - 通道压缩: 1024→896     │
└─────────────────────────┘
        │
        ▼
输出: n×16×16 patches → m visual tokens (m = n²/16)
```

**关键特性**：
- 16×压缩比（通过window attention实现）
- 参数量80M，与LLM文本embedding层相当
- 可被简单的patch embedding替代（非必须）

---

## 三、技术细节深度解析

### 3.1 前向传播公式

```
核心公式:

O = D(π_Q(T_L(E(I) ⊕ Q₀; M))))

其中：
- I ∈ R^{H×W×3}: 输入图像
- E: vision tokenizer, 映射到 m 个visual tokens V ∈ R^{m×d}
- Q₀ ∈ R^{n×d}: 可学习的causal query embeddings
- ⊕: 序列拼接 (concatenation)
- T_L: L层的Transformer (with masked attention)
- M ∈ {0,1}^{2n×2n}: block causal attention mask (Equation 1)
- π_Q: 投影操作，提取最后n个tokens (Z = X_{m+1:m+n})
- D: 语言decoder
- O ∈ R^{n×|V|}: LLM vocabulary上的输出logits
```

### 3.2 为什么Decoder-only设计有效？

论文提到关键发现：**使用cross-attention的mBART-style encoder-decoder结构无法收敛**。

```
失败的架构:
Visual Tokens → [Encoder] → Cross-attention → [Decoder] → Output

成功的架构:
[Visual Tokens + Causal Queries] → [Prefix-concatenation] → [Decoder-only LLM] → Output
                                ↑
                        视觉tokens始终活跃，与causal queries充分交互
```

**原因分析**：
- 独立encoder中的视觉tokens交互不足
- Prefix设计让visual tokens在所有layer中保持活跃
- 促进visual信息与causal queries的有效交换

### 3.3 两级级联因果推理

```
第一级因果推理 (Encoder):
    Visual Tokens → Causal Queries
    [语义重新排序]

第二级因果推理 (Decoder):
    Ordered Causal Queries → Autoregressive Generation
    [自回归推理]

┌─────────────────────────────────────────────────────┐
│  2D空间理解 = 两个1D因果推理器的级联                │
│                                                     │
│  Encoder: 处理阅读逻辑推理                           │
│  (通过query tokens因果地重排序视觉信息)              │
│                                                     │
│  Decoder: 执行视觉任务推理                           │
│  (在因果排序的表示上进行推理)                        │
└─────────────────────────────────────────────────────┘
```

---

## 四、训练流程详解

### 4.1 三阶段训练流程

```
┌─────────────────────────────────────────────────────────┐
│                  训练流程 (三阶段)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Stage 1: Encoder Pretraining                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ • Vision Tokenizer + LLM-style Encoder          │   │
│  │ • 目标: 特征提取、token压缩、token重排序        │   │
│  │ • Decoder: 轻量级decoder用于next-token预测      │   │
│  │ • 分辨率: 768×768, 1024×1024                   │   │
│  │ • 初始化:                                       │   │
│  │   - Vision Tokenizer: 来自DeepEncoder          │   │
│  │   - LLM Encoder: Qwen2-0.5B-base               │   │
│  │ • 优化器: AdamW                                 │   │
│  │ • LR: 1e-4 → 1e-6 (cosine decay)              │   │
│  │ • 硬件: 160×A100 (20 nodes × 8 GPUs)          │   │
│  │ • Batch size: 640                              │   │
│  │ • 迭代: 40k (约100M图像-文本对)                │   │
│  └─────────────────────────────────────────────────┘   │
│                      │                                  │
│                      ▼                                  │
│  Stage 2: Query Enhancement                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ • 冻结Vision Tokenizer                          │   │
│  │ • 联合优化LLM Encoder + LLM Decoder            │   │
│  │ • 统一分辨率: Multi-crop策略                    │   │
│  │ • Pipeline并行 (4-stage):                       │   │
│  │   PP0: Vision Tokenizer                         │   │
│  │   PP1: LLM-style Encoder                        │   │
│  │   PP2-3: DeepSeek-LLM layers (6 layers/PP)     │   │
│  │ • 硬件: 160×A100 40GB                          │   │
│  │ • Data replicas: 40 (4 GPUs/replica)           │   │
│  │ • Global batch size: 1280                       │   │
│  │ • LR: 5e-5 → 1e-6                               │   │
│  │ • 迭代: 15k                                     │   │
│  └─────────────────────────────────────────────────┘   │
│                      │                                  │
│                      ▼                                  │
│  Stage 3: Continue-training LLM                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │ • 冻结所有DeepEncoder V2参数                    │   │
│  │ • 只更新DeepSeek-LLM参数                        │   │
│  │ • 加速训练 (相同batch下速度翻倍)                 │   │
│  │ • 帮助LLM更好理解重排序的视觉tokens             │   │
│  │ • LR: 1e-6 → 5e-8                               │   │
│  │ • 迭代: 20k                                     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.2 数据引擎

```
训练数据组成:

OCR 1.0 + OCR 2.0: 80%
  ├─ 文本
  ├─ 公式
  └─ 表格
  └─ 采样比例: 3:1:1 (更平衡)

General Vision Data: 20%

标签优化:
  - 布局检测标签精炼
  - 合并语义相似类别 (如"figure caption"和"figure title")
```

---

## 五、实验结果与性能分析

### 5.1 OmniDocBench v1.5 主要结果

| 模型 | V-token_max | Overall↑ | Text Edit↓ | Formula CDM↑ | Table TEDs↑ | R-order Edit↓ |
|------|-------------|----------|-------------|--------------|-------------|---------------|
| PaddleOCR-VL | - | 92.86% | 0.035 | 91.22 | 90.89 | 94.76 |
| Qwen2.5-VL-72B | >6000 | 87.02% | 0.094 | 88.27 | 82.15 | 86.22 |
| Qwen3-VL-235B | >6000 | 89.15% | 0.069 | 88.14 | 86.21 | 90.55 |
| Gemini-2.5 Pro | - | 88.03% | 0.075 | 85.82 | 85.71 | 90.29 |
| DeepSeek-OCR (9-crops) | 1156 | 87.36% | 0.073 | 84.14 | 85.25 | 89.01 |
| **DeepSeek-OCR 2** | **1120** | **91.09%** | **0.048** | **90.31** | **87.75** | **92.06** |
| **提升** | - | **+3.73%** | **-0.025** | **+6.17** | **+2.5** | **+3.05** | **-0.028** |

**关键发现**：
1. **以最小的视觉token预算（1120）实现了最佳性能（91.09%）**
2. 阅读顺序的编辑距离显著降低（0.085 → 0.057），证明了causal flow的有效性

### 5.2 详细文档类型对比

```
DeepSeek-OCR 2 vs DeepSeek-OCR 在9类文档上的Edit Distance对比:

文档类型      │ Text ED (DS-OCR→DS-OCR2) │ R-order ED (DS-OCR→DS-OCR2)
─────────────┼──────────────────────────┼────────────────────────────
PPT          │ 0.052 → 0.031 ✓          │ 0.052 → 0.025 ✓
Academic Paper│ 0.028 → 0.013 ✓          │ 0.021 → 0.013 ✓
Book         │ 0.022 → 0.033 ✗          │ 0.040 → 0.027 ✓
Colorful Textbook│ 0.130 → 0.053 ✓       │ 0.125 → 0.066 ✓
Exam Paper   │ 0.074 → 0.047 ✓          │ 0.083 → 0.048 ✓
Magazine     │ 0.049 → 0.026 ✓          │ 0.101 → 0.100 ≈
Newspaper    │ 0.131 → 0.139 ✗          │ 0.217 → 0.176 ✓
Note         │ 0.145 → 0.068 ✓          │ 0.089 → 0.035 ✓
Research Report│ 0.015 → 0.008 ✓         │ 0.016 → 0.011 ✓

说明:
✓ 改进, ✗ 下降, ≈ 持平
```

**分析**：
- 在阅读顺序（R-order）上，DeepSeek-OCR 2 **全面优于** DeepSeek-OCR
- 在文本识别上，除了Newspaper和Book外，均有改进
- Newspaper表现下降的可能原因：
  1. 视觉token上限限制（超密集文本）
  2. 训练数据不足（仅250k newspaper样本）

### 5.3 与其他模型在编辑距离上的对比

| 模型 | V-token_max | Text Edit↓ | Formula Edit↓ | Table Edit↓ | R-order Edit↓ | Overall Edit↓ |
|------|-------------|-------------|---------------|-------------|---------------|---------------|
| Gemini-3 Pro | 1120 | - | - | - | 0.115 | - |
| Seed-1.8 | 5120 | - | - | - | 0.106 | - |
| DeepSeek-OCR | 1156 | 0.073 | 0.236 | 0.123 | 0.085 | 0.129 |
| **DeepSeek-OCR 2** | **1120** | **0.048** | **0.198** | **0.096** | **0.057** | **0.100** |

**在相似视觉token预算下，DeepSeek-OCR 2的阅读顺序ED显著低于Gemini-3 Pro（0.057 vs 0.115）**

### 5.4 生产环境性能

```
生产环境重复率对比:

场景                     │ DeepSeek-OCR │ DeepSeek-OCR 2 │ 改进
─────────────────────────┼──────────────┼────────────────┼──────────
在线用户日志 (image)      │ 6.25%        │ 4.17%          │ -2.08%
预训练数据 (PDF)          │ 3.69%        │ 2.88%          │ -0.81%

说明: 重复率是OCR模型在无ground truth的生产环境中
     主要可观测的质量指标
```

---

## 六、讨论与未来工作

### 6.1 Towards Genuine 2D Reasoning

DeepSeek-OCR 2提出了一种新颖的架构范式：**两个级联的1D因果推理器实现真正的2D推理**

```
Genuine 2D Reasoning 路径:

2D空间理解
    │
    ├───┬──────────────────────────────────────────────┐
    │                                                │
    ▼                                                ▼
分解为两个互补的1D因果推理子任务                    目标
    │                                                │
    ├── Encoder: 阅读逻辑推理                        │ 真正的2D推理
    │   (通过query tokens因果地重排序视觉信息)        │
    │                                                │
    └── Decoder: 视觉任务推理                        │
        (在因果排序的表示上进行推理)                  │
                                                     │
未来方向:                                            │
• 支持多重重新审视                                   │
• 多跳重排序                                         │
• 更长的causal flow tokens                          │
```

### 6.2 Towards Native Multimodality

DeepEncoder V2为统一的多模态编码器提供了初步验证：

```
统一多模态编码器构想:

Single Encoder with:
├─ Shared Parameters:                              │
│   ├─ W_k, W_v (projections)                      │
│   ├─ Attention mechanisms                        │
│   └─ FFNs                                         │
│                                                    │
└─ Modality-specific Learnable Queries:            │
    ├─ Text Queries → 文本压缩                      │
    ├─ Audio Queries → 语音特征提取                 │
    └─ Visual Queries → 视觉内容重组                │
         (如DeepSeek-OCR 2的causal flow tokens)    │

优势:
• 在同一参数空间处理多种模态
• 继承LLM社区的高级优化 (MoE, Efficient Attention等)
• DeepSeek-OCR的光学压缩是初步探索
• LLM-style encoder架构是进一步进展
```

---

## 七、关键技术要点总结

### 7.1 核心创新点

| 创新点 | 技术细节 | 优势 |
|--------|----------|------|
| **LM-style Vision Encoder** | 用Qwen2-0.5B替换CLIP ViT | 获得因果推理能力，继承LLM优化 |
| **双流注意力机制** | Visual tokens双向 + Causal queries因果 | 全局视野 + 因果依赖 |
| **Causal Flow Query** | 与visual tokens等数量的可学习查询 | 提供足够重新定位容量 |
| **Cascade Causal Reasoning** | Encoder排序 + Decoder推理 | 桥接2D空间与1D因果建模 |
| **高压缩率** | 16×视觉token压缩 | 计算效率高 |

### 7.2 性能优势

1. **在OmniDocBench v1.5上达到91.09%**
2. **使用最小的视觉token预算（1120）实现最佳性能**
3. **阅读顺序编辑距离降低32.9%（0.085 → 0.057）**
4. **生产环境重复率降低33%（6.25% → 4.17%）**

---

## 八、相关技术对比

### 8.1 与其他并行查询设计对比

| 模型 | 查询类型 | Attention方式 | 应用场景 |
|------|----------|---------------|----------|
| DETR | Object queries (n=100) | 双向self-attention | 目标检测 |
| BLIP-2 Q-former | Token queries (n=32) | 双向self-attention | 视觉token压缩 |
| **DeepEncoder V2** | Causal flow queries (n=m) | 因果attention | 视觉语义重排序 |

### 8.2 与LLM-based多模态初始化对比

| 方法 | 特点 | 代表模型 |
|------|------|----------|
| Frozen LLM layers | 冻结LLM层增强视觉任务 | Pang et al. |
| Encoder-free/Lightweight | 无或轻量编码器 | Fuyu, Chameleon |
| **LM-style Encoder** | LLM架构作为视觉编码器 | DeepEncoder V2 |

---

## 九、技术细节补充

### 9.1 Multi-crop策略详解

```
Multi-crop策略:

输入图像 → 根据尺寸决定裁剪

如果 W < 768 且 H < 768:
    只使用 global view (1024×1024)
    → 256 tokens

如果 W ≥ 768 或 H ≥ 768:
    global view (1024×1024) + k个local views (768×768)
    → 256 + k×144 tokens

其中 k ∈ [0, 6]:
    k = 0: 无local crops
    k = 6: 最多6个local crops

总token数范围: [256, 1120]
```

### 9.2 参数规模对比

```
参数规模对比:

Vision Tokenizer:
  - SAM-base + Conv layers: 80M参数
  - 输出维度: 896 (从1024压缩)

DeepEncoder V2:
  - Qwen2-0.5B: 500M参数
  - 与CLIP ViT (300M)相当

DeepSeek-MoE Decoder:
  - 总参数: 3B
  - 活跃参数: ~500M (MoE-A570M)

视觉token压缩:
  - 16×压缩比
  - 最小256 tokens (global only)
  - 最大1120 tokens (global + 6 locals)
  - 对应Gemini-3 Pro的视觉token预算
```

---

## 十、技术启示与联想

### 10.1 对VLM设计的启示

1. **打破固定扫描顺序的限制**
   - 传统VLM的刚性光栅扫描顺序不符合人类视觉感知
   - 引入语义驱动的动态重排序

2. **1D因果模型的2D理解**
   - 两个级联的1D因果推理器可能实现真正的2D推理
   - Encoder处理阅读逻辑，Decoder处理任务推理

3. **LLM架构的多模态泛化**
   - LLM不仅可用于文本，也可作为视觉编码器
   - 统一架构便于优化迁移（MoE、高效attention）

### 10.2 可能的应用扩展

```
技术扩展方向:

1. 通用视觉理解
   ├─ 场景理解
   ├─ 图像推理
   └─ 视觉问答

2. 多模态统一编码
   ├─ 文本 + 图像
   ├─ 音频 + 文本
   └─ 视频 + 音频 + 文本

3. 长上下文理解
   ├─ 多页文档
   ├─ 长序列推理
   └─ 跨页关联
```

### 10.3 与其他工作的联系

- **General OCR Theory (Wei et al., 2024)**: DeepSeek-OCR的理论基础
- **Context Cascade Compression (Liu & Qiu, 2025)**: 文本压缩的上限探索
- **DeepSeek-V3 (Liu et al., 2024)**: MoE架构的借鉴
- **InternVL3.5 (Wang et al., 2025)**: 开源多模态模型进展

---

## 十一、代码与资源

**官方代码和模型权重**：
- GitHub: http://github.com/deepseek-ai/DeepSeek-OCR-2

**相关论文**：
- DeepSeek-OCR: https://arxiv.org/abs/2510.18234
- General OCR Theory: https://arxiv.org/abs/2409.01704
- Qwen2-VL: https://arxiv.org/abs/2409.12191

---

## 十二、总结

DeepSeek-OCR 2通过**DeepEncoder V2**创新性地解决了传统VLM的刚性扫描顺序问题，其核心贡献包括：

1. **LM-style视觉编码器**：用紧凑LLM替换CLIP，获得因果推理能力
2. **双流注意力机制**：结合双向和因果attention，实现语义驱动的视觉token重排序
3. **级联因果推理**：两个1D因果推理器协同工作，向真正的2D理解迈进
4. **高效压缩**：保持16×视觉token压缩，性能提升3.73%

这项工作不仅为文档OCR提供了更优解决方案，更为通用视觉理解和原生多模态建模提供了新的技术路径。