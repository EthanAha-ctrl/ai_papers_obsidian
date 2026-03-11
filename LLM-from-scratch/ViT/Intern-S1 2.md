 让我继续阅读这篇论文的其余部分，以获取完整的技术细节。


这篇论文介绍的是 **Intern-S1**，由上海 AI Lab 开发的一个专门针对 **科学领域 (scientific domain)** 的多模态 (multimodal) 大模型。让我从技术架构、训练策略到实验结果，给你一个全面的技术拆解。

---

## 1. 核心定位与动机 (Problem Motivation)

### 图2揭示的关键问题
论文指出了一个现象：现有的开源模型在数学、代码等"热门任务"上进步飞快，但**在科学领域（低资源 but 高价值）的能力增长却严重滞后**。

这是因为：
- 科学数据稀缺且分散（molecular structures, spectrograms, time-series 等）
- 需要长程严谨推理（hypothesis validation, experimental design）
- 不同科学模态差异巨大（从蛋白质序列到地震波信号）

**Intern-S1 的目标**：做一个既能处理通用任务，又具备专业科学理解能力的"通才+专家"模型。

---

## 2. 模型架构 (Architecture)

Intern-S1 是一个 **MoE (Mixture-of-Experts)** 架构：
- **激活参数量**：28B
- **总参数量**：241B
- 基于 **Qwen3-235B** (Intern-S1) 和 **Qwen3-8B** (Intern-S1-mini)

### 2.1 多模态编码器设计 (图3)

论文把科学模态分为三类，每种用不同策略编码：

#### **(1) 视觉模态 (Visualizable)**
- 使用 **InternViT-6B** (Intern-S1) 或 **InternViT-300M** (mini)
- 支持动态分辨率 (dynamic resolution)，最大 448×448
- **pixel unshuffle** 技术：把视觉 token 数量减少 4 倍（448×448 → 256 tokens）
- 通过 MLP projector 对齐到 LLM 的 embedding 空间

#### **(2) 线性化离散表示 (Linearizable Discrete)**
- 这是论文的**核心创新**之一：**Dynamic Tokenizer**
- 处理分子式 (SMILES)、蛋白质序列 (FASTA) 等
- 传统 tokenizer 对所有序列用同一套切分策略，导致：
  - SMILES 在通用语料中罕见，压缩率差
  - 同一个字符 (如"C") 在不同模态中语义不同（碳元素 vs 选项C），但被迫共享 embedding

**Dynamic Tokenizer 的工作流程**（图4左）：
```
输入字符串 → 规则检测器/特殊标签识别 → 分段 → 不同切分策略 → 正交嵌入空间 → 拼接 → Transformer输入
```

**正交嵌入 (Orthogonal Embedding)**：
- 不同模态的 token embedding 空间互相正交
- 避免语义干扰，让模型精确区分"C"是碳元素还是选择题选项

**压缩率结果**（图4右）：
在 SMILES 数据集上，Intern-S1 的压缩率比 GPT-4、DeepSeek-R1、Qwen3 系列**高 70%**！这意味着更少的 token 表示相同分子结构，节省计算开销。

#### **(3) 时序信号 (Time Series)**
- 专门设计的 **Time Series Encoder**
- 处理地震波、引力波、天文光变曲线、脑电信号 (EEG) 等
- 特点：
  - 输入原始数值信号（而非 tokenized）
  - 采样率差异巨大（每天一次到 GHz 级别）
  - 序列长度变化大（几十到数百万时间步）
- 架构：自适应下采样模块 (adaptive downsampling) + Transformer blocks

---

## 3. 训练基础设施 (Infrastructure)

### 3.1 预训练与 SFT 基础设施

**FP8 训练**：
- 矩阵乘法 (GEMM) 使用 FP8 精度，采用 per-tile (1×128) 动态缩放
- Forward: tile-wise scaling for inputs, block-wise for weights
- Backward: 两类 GEMM（gradients+weights 和 gradients+inputs）
- Vision tower 保持 BF16 确保稳定性

**优化 Kernel**：
1. **TMA-Adaptive FP8 Grouped GEMM**：处理 MoE 的动态 group size 问题，减少内存和计算开销
2. **Liger-kernel**：融合 linear 和 cross entropy 层
3. **Flash Attention-3**：支持可变长度序列

**Variable-Length Balanced Strategy (VLBS)**：
解决 FSDP 在变长训练中的负载不均衡问题：
1. 随机打包文档到 buckets，记录最大序列长度
2. 应用滑动窗口 (sliding window, SS) 分组 buckets
3. 在每个窗口内按最大长度排序
效果：**平均 2 倍加速**

### 3.2 RL 基础设施

**并行策略**：
- FSDP + 1-way Expert Parallelism (EP)
- 避免 dropless MoE 在长序列训练时的内存爆炸

**FP8 推理与训练**：
统一使用 FP8，最大化 rollout 吞吐量

**Colocated Design**：
训练和推理引擎共置 (colocate) 在同一设备上：
- 每个 RL step 开始时，模型从 training mesh 透明地重分布到 rollout mesh
- 收集轨迹后，再重分布回去，optimizer 状态保持不变
- 轻量级重分布和集体同步，无需资源分区

**Rollout**：
使用 LMDeploy 进行推理，8-way Expert Parallel (EP8)
- PyTorch 实现，权重存储为 FP8
- CPU offloading + continuous batching
- 动态负载重平衡防止 stragglers

---

## 4. 持续预训练 (Continue Pre-training)

### 4.1 科学数据 (图6)

**数据规模**：
- 总计 **5T tokens** 的持续预训练数据
- 其中 **2.5T+ tokens** 来自科学领域（占一半以上）
- 六个核心科学领域：Mathematics, Physics, Chemistry, Life Science, Earth Science, Materials Science

### 4.1.1 文本数据处理

**三个关键 pipeline**：

**1. Page-level PDF 文档解析 (图7)**
这是处理科学文献的核心：
- 低成本的 parser (MinerU) 先处理所有页面
- **启发式检测器**检查页面中是否有大量公式、符号标记
- 有问题的页面送入高成本的 VLM parser (InternVL, Qwen-VL)
- VLM 结果再通过规则或小 LLM 清洗
- 全局页面级去重（移除版权页等通用内容）

效果：
- 即使是干净的 PDF 数据，也要去除约 20% 的低质量 token
- 网页爬取的 PDF 经过教育水平筛选后，保留率约 50%
- 对于存档图书馆数据：5% 页面用高成本 parser
- 网页爬取 PDF：3% 页面用高成本 parser

**2. Domain-centric web 数据解析 (图8)**
处理网页数据的新方法：
- 同 URL domain 的页面共享特征
- 从每个 domain 采样数百页面，送入 LLM-based classifier
- 根据分类结果，对整个 domain 做出决策：
  - **丢弃**：质量低且不具信息性
  - **重写**：质量低但内容有价值，用 LLM 重写
  - **保留**：作为训练候选

动机：同一 domain 的页面常有共同的解析问题（如代码片段提取失败、难以检测的导航栏）。Domain-level 解析能识别轻量级分类器无法捕捉的结构模式，同时控制成本。

**3. Scientific data recall and filtering (图9)**
从开源预训练语料和 Common Crawl 中召回科学数据：
- 构建三级 domain taxonomy（类似 Du et al., 2025）
- 针对六个科学领域细粒度处理
- 流程：
  1. 用强 LLM 标注子集数据
  2. 训练轻量级分类器 (fastText, 1.5B param LLM)
  3. 构建 in-domain 和 OOD (out-of-domain) validation sets
  4. 用 validation sets 优化标注 prompts
  5. 分类器过滤 web data pool，召回目标数据

效果：人工评估显示，目标 domain 数据比例从约 2% 提升到 50%，**提升 25 倍**！

### 4.1.2 多模态数据

**数据构成**：
- 交错图文数据集
- 纯文本数据集
- 总训练量：约 **250B tokens**
  - 70B 语言数据
  - 180B 交错图文数据（其中 30B 科学数据）

**三个数据源**：
1. InternVL3 的多模态预训练语料
2. 4.1.1 节的文本语料（保持文本理解能力）
3. 多模态科学数据（专门领域数据增强）

**多模态科学数据处理 pipeline**：
目标：(1) 保留精细科学结构（图、公式、符号、表格、图表）；(2) 图文对齐；(3) 生成指令式和考试式监督数据

- **考试式问题**：检查结构完整性（题干、选项、答案、解析），用规则和 LLM 评估
- **公式渲染验证**：用 VLM 判断 LaTeX/Markdown 公式渲染结果是否正确
- **通用图文对**：规则过滤（空白图、模糊图、图文不对应）

### 4.2 训练策略

#### 4.2.1 Batch Size Warmup（图10）

核心发现：**batch size 的最优值随训练阶段变化**

- 小 batch size (4M tokens)：前期训练质量更好（前 700B tokens）
- 大 batch size (10M tokens)：后期训练效率更高

**策略**：分两阶段
- 前 400B tokens：batch size = 66M
- 400B 后：batch size = 132M

理论依据：基于 Scaling Law 分析，找到 Critical Batch Size 与训练 loss 的关系，确定最优切换点。

#### 4.2.2 Starting Point Choice（图11）

**问题**：继续预训练 (CPT) 应该基于 base model 还是 instruction model？

实验结论：
1. 最终性能差异不大（Post-training 主要是"激活"预训练已学能力，而非"新增"能力）
2. 只有在 coding 等特定领域（Post-training 真正新增能力的领域），instruction model 才有优势
3. 输出多样性差异（entropy：base 0.19 vs CPT-on-instruct 0.15）可通过 RL 超参数调节

**实践选择**：使用 instruction model 进行 CPT 是可接受的，且在某些领域（如 coding）更优。

#### 4.2.3 Hyper Parameters（关键公式）

**公式 (1) - Gradient Noise 与 Batch Size 的关系**：

$$1 - \frac{1}{2}\eta\frac{\mathcal{B}_{\text{simple}}}{B} > 0, \quad \mathcal{B}_{\text{simple}} = \frac{tr(\Sigma)}{|G|^{2}}$$

变量解释：
- $\eta$：learning rate
- $B$：batch size
- $\mathcal{B}_{\text{simple}}$：gradient noise，来自 McCandlish et al. (2018) 的噪声度量
- $tr(\Sigma)$：梯度协方差矩阵的迹
- $|G|$：梯度范数

物理意义：该不等式定义了"Critical Batch Size"的条件，确保梯度噪声不会导致训练不稳定。

**公式 (2) - Learning Rate 的优化问题**：

$$\min_{\Omega} L_{\theta}(\Omega), \quad s.t. \Omega = \{\eta_i | 0 \leq i \leq T, 0 \leq \eta_i \leq \mu\}, \phi(\Omega)$$

变量解释：
- $\Omega$：整个训练过程的 learning rate 集合 $\{\eta_i\}$
- $L_{\theta}(\Omega)$：关于 learning rate  schedule 的训练 loss
- $T$：总训练步数
- $\mu$：learning rate 上限
- $\phi(\Omega)$：对 learning rate 的额外约束

这是一个**基于 Scaling Law 的 learning rate 优化框架**，通过拟合 $L(\Omega)$ 的关系，求解最优的 learning rate schedule。

**Predictable Loss**：
论文报告最终训练 loss 预测值约 1.16，实际落在 1.17-1.18，精度达到 **0.02 级别**，展示了对预训练质量的精确控制。

#### 4.2.4 Multi-modal Training

**联合训练所有参数**：
不同于传统 MLLM 冻结部分层，Intern-S1 在多模态 CPT 中**联合更新所有参数**（LLM、ViT、projectors）。

**Loss Function（公式 3）**：

$$\mathcal{L}(\theta) = -\sum_{\substack{i=2 \\ x_i \in \text{Text}}}^{L} w_i \cdot \log p_{\theta}(x_i | x_1, \dots, x_{i-1})$$

变量解释：
- $\theta$：模型参数
- $L$：序列总长度
- $x_i$：第 $i$ 个 token（只计算 Text token 的 loss，Visual token 只作为 condition）
- $w_i$：第 $i$ 个 token 的 loss weight
- $p_{\theta}$：模型预测的概率分布

**Square-averaging scheme（公式 4）**：
$$w_i = l^{-1/2}$$

其中 $l$ 是参与 loss 计算的 token 数。这种加权方案用于缓解梯度偏差。

---

## 5. 后训练 (Post-training): 强化学习框架

这是 Intern-S1 的**核心创新之一**，采用了 **两阶段 RL**：Offline RL (SFT with reward) → Online RL (Mixture-of-Rewards)

### 5.1 Offline Reinforcement Learning

虽然通常叫 SFT，但论文将其视为 **Offline RL**，因为所有训练响应都通过 **Best-of-N (BoN) sampling** 选择（基于准确率、流畅性、安全性等标准）。

### 5.2 Online Reinforcement Learning: Mixture-of-Rewards (MoR)

这是论文的**关键技术贡献**之一（图12）。

#### 挑战
同时训练 1000+ 任务，每个任务反馈形式不同：
- 可验证任务（数学题、编程）：有 ground truth，用规则验证
- 难验证任务（创意写作、对话）：需要 reward model 评估质量

#### MoR 框架设计

MoR 将各种异构反馈统一为**单一标量 reward**：

1. **对于难验证任务**：使用 **POLAR (Dou et al., 2025)**
   - 基于 Policy Discriminative Learning
   - 学习两个 policy 的相对差异，而非绝对分数
   - 提供距离期望分布的标量距离作为 reward

2. **对于易验证任务**：使用 **组合验证器**
   - CompassVerifier (Liu et al., 2025a)：轻量级生成式验证器
   - 规则验证
   - 环境反馈（如代码执行结果）
   - 生成精确指示准确率的 reward 标量

#### Offline-Online 混合数据过滤策略

为解决多任务训练中难度差异和收敛速度差异的问题：

1. **Offline 阶段**：对每个任务 domain 单独 rollouts 和训练，测量难度和收敛速度，确定最终数据混合比例
2. **Online 阶段**：rollout 的 prompt 数量大于训练 batch size，根据准确率、输出质量等标准在线过滤，生成足够数据后即停止生成

### 5.3 Policy Optimization：MoE 适配的 RL 算法

#### 挑战：MoE + RL 的不稳定性
直接应用 GRPO 等算法到大规模 MoE 会导致**训练崩溃**，根源是：
- Inference 和 Training 使用不同 kernel，存在数值差异
- MoE 的动态 expert routing + FP8 量化放大这种差异
- 导致 inference 和 training 激活的 experts 不匹配，policy 严重 off-policy
- Token-level clipping 进一步加剧问题（experts 差异导致 ratio 不可靠）

#### 解决方案：OREAL + KL-Cov
论文采用 **OREAL (Lyu et al., 2025)** 算法：
- 对正样本用 SFT loss（behavior cloning）
- 对负样本用 Policy Gradient
- **不使用**基于新旧 policy log-prob ratio 的 token-level clipping，天然避免 MoE 崩溃问题

但 OREAL 需要在线训练 token-level reward model 进行 credit assignment，计算开销大。为加速训练，论文**移除了 token-level reward model**，但这导致 entropy 快速下降，policy 失去探索能力。

为此，引入 **KL-Cov 策略**（来自 Cui et al., 2025）：

**公式 (5) - KL-Cov Loss**：
$$\mathcal{L}_{\text{KL-Cov}}(\theta) = \begin{cases} 0, & t \notin I \\ \mathbb{E}_{t}\left[-\beta D_{\text{KL}}\!\left(\pi_{\theta_{\text{old}}}(y_{t} \mid y_{<t}) \| \pi_{\theta}(y_{t} \mid y_{<t})\right)\right], & t \in I \end{cases}$$

变量解释：
- $t$：token 索引
- $I = \{i \mid \text{Rank}(\text{Cov}(y_i)) \leq k \cdot N\}$：token 协方差排名在前 $k \cdot N$ 的集合（$N$ 为序列长度）
- $\beta$：KL 系数 (0.01)
- $\pi_{\theta_{\text{old}}}$：旧 policy
- $\pi_{\theta}$：新 policy

**作用**：只对高方差（高不确定性）的 token 施加 KL 约束，防止 entropy collapse，同时保持模型探索能力。

**整体 Loss Function（公式 6）**：
$$\mathcal{L}(\theta) = \lambda_{\text{sft}} \mathbb{E}_{\mathcal{D}^{+}}[L_{\text{sft}}(x,y;\theta)] + \lambda_{\text{pg}} \mathbb{E}_{\mathcal{D}^{-}}[L_{\text{pg}}(x,y;\theta)] + \mathcal{L}_{\text{KL-Cov}}(\theta)$$

- $\mathcal{D}^{+}$：正样本集（成功解决的任务）
- $\mathcal{D}^{-}$：负样本集（失败的任务）
- $L_{\text{sft}}$：SFT loss（behavior cloning）
- $L_{\text{pg}}$：Policy Gradient loss
- $\lambda_{\text{sft}}$, $\lambda_{\text{pg}}$：权重系数

---

## 6. 数据工程细节 (Data Curation)

### 6.1 PDF 文档解析 (图7)

科学文献主要是 PDF，论文开发了**页面级混合解析 pipeline**：

```
PDF → 分页 → MinerU (低成本 parser) → 启发式检测 (公式/符号密度)
  ↓ 问题页面                      ↓ 干净页面
高成本 VLM parser (InternVL/Qwen-VL)   直接后处理
  ↓
合并页面 + 全局去重 → 最终语料
```

关键洞察：**没有单一工具能完美解析所有 PDF**，根据内容复杂度动态选择 parser 是性价比最优解。

- 存档图书馆 PDF：5% 页面需 VLM
- 网页爬取 PDF：3% 页面需 VLM
- VLM 比 MinerU 慢 20 倍，但精准度必要

### 6.2 Domain-centric Web 解析 (图8)

网页数据按 URL domain 分组处理：
- 从每个 domain 采样数百页面
- LLM-based agent 打标签
- 对整个 domain 决策：**丢弃 / 重写 / 保留**

优势：识别 domain 级别的结构模式（如特定网站的导航栏、代码片段格式问题），比页面级启发式规则更精准，同时控制成本。

### 6.3 Scientific Data Recall (图9)

从 Common Crawl 等开源语料中召回科学数据：
- 构建三级 taxonomy（类似 SuperGPQA）
- 针对六个科学领域（Math, Physics, Chemistry, Life Science, Earth Science, Materials Science）
- **流程**：
  1. LLM 标注小样本来训练轻量分类器 (fastText, 1.5B param LLM)
  2. 构建 in-domain 和 OOD validation sets
  3. 验证集用于自动优化 prompts
  4. 分类器过滤 web data pool，召回目标数据

**效果**：目标 domain 数据比例从 **2% 提升到 50%**（25倍提升）！

---

## 7. 评估结果 (Evaluation)

### 7.1 通用推理基准 (表2)

| Benchmark | Intern-S1 | 开源最佳 | 闭源最佳 (o3/Gemini) |
|-----------|-----------|----------|---------------------|
| MMLU-Pro  | **83.5**  | 83.4 (DeepSeek-R1) | 86.0 (Gemini) |
| GPQA      | **77.3**  | 80.6 (DeepSeek-R1) | 83.8 (Gemini) |
| AIME2025  | **86.0**  | 87.5 (DeepSeek-R1) | 91.7 (Grok-4) |
| MathVista | **81.5**  | 79.0 (InternVL3)   | 80.3 (Gemini) |
| MMMU      | **77.7**  | 72.2 (InternVL3)   | 81.9 (Gemini) |

分析：
- Intern-S1 在多模态通用推理上**领先所有开源模型**
- 在 MathVista、MMMU 等视觉推理任务上有显著优势（MathVista +2.5 于 InternVL3-78B）
- 与闭源模型仍有差距（GPQA、AIME2025），但已相当接近

### 7.2 科学推理基准 - 纯文本 (表3)

| Benchmark | Intern-S1 | 开源最佳 | 闭源最佳 |
|-----------|-----------|----------|----------|
| SmolInstruct | **51.0** | 48.1 (Kimi-K2) | 47.3 (Grok-4) |
| ChemBench    | **83.4** | 75.8 (Qwen3)    | 83.3 (Grok-4) |
| MatBench     | **75.0** | 61.7 (Kimi-K2)  | 67.9 (Grok-4) |
| ProteinLMBench| 63.1   | 66.7 (Kimi-K2)  | 67.7 (o3) |

**重大突破**：
- **SmolInstruct (化学)**：超越所有闭源模型（包括 o3, Gemini, Grok-4）！
- **ChemBench (化学知识)**：与 Grok-4 持平 (83.4 vs 83.3)，碾压其他开源模型 (+7.6~24.1)
- **MatBench (材料科学)**：大幅超越所有模型，包括闭源（75.0 vs 67.9）

**ProteinLMBench 落后分析**：蛋白质语言处理是一个高度专业化的子领域，可能需要 domain-specific 的预训练或架构调整。

### 7.3 科学推理基准 - 多模态 (表4)

| Benchmark | Intern-S1 | 开源最佳 | 闭源最佳 |
|-----------|-----------|----------|----------|
| SFE (科学家初级考试) | **44.3** | 43.9 (MiMo) | 43.0 (Gemini) |
| Physics (物理博士考题) | 44.0 | 28.3 (GLM) | **47.9** (o3) |
| MicroVQA (显微镜图像) | **63.9** | 59.1 (InternVL3) | 63.1 (Gemini) |
| MSEarthMCQ (地球科学) | **65.7** | 58.0 (Grok-4) | 61.0 (o3) |
| XLRS-Bench (超高分辨率遥感) | **55.0** | 51.6 (MiMo) | 45.4 (Grok-4) |

**关键突破**：
- **几乎全面领先开源模型**，在 5/5 个 benchmark 上取得最佳或接近最佳
- **SFE**：首次超越闭源模型（44.3 vs 43.0 Gemini）
- **MSEarthMCQ**：大幅超越闭源模型（65.7 vs 61.0 o3）
- **XLRS-Bench**：在超高分辨率遥感图像理解上显著领先（55.0 vs 45.4 Grok-4）

---

## 8. Intern-S1-mini 结果 (表5-7)

轻量版 Intern-S1-mini (8B 激活) 同样表现强劲：

**通用推理**（表5）：
- MMLU-Pro: 74.8 (超越 Qwen3-8B 的 73.7)
- GPQA: 65.2 (领先 Qwen3-8B 的 62.0)

**科学推理**（表6）：
- SmolInstruct: 32.2 (超越所有基线，包括 Intern-S1 在某些设置下的表现)
- ChemBench: 76.5 (远超 Qwen3-8B 的 61.1)
- MatBench: 61.6 (领先 Qwen3-8B 的 45.2)

---

## 9. 核心技术贡献总结

### 9.1 Dynamic Tokenizer (动态分词器)
- 解决多模态（SMILES、FASTA、自然语言）的编码冲突
- 正交嵌入空间让不同模态互不干扰
- **70% 的压缩率提升**，大幅减少科学数据的 token 数量

### 9.2 Mixture-of-Rewards (MoR)
- 统一处理 1000+ 任务的异构反馈
- POLAR 处理开放式任务（相对排序奖励）
- CompassVerifier + 规则处理可验证任务（二元准确率奖励）
- **10x RL 训练效率提升**（相比 DeepSeek-R1 等公开基线）

### 9.3 MoE-specific RL 优化
- 发现传统 GRPO/DAPO 的 token-level clipping 在 MoE 中不可靠
- OREAL + KL-Cov 规避 token-level ratio 问题，防止 entropy collapse
- FP8 统一训练和推理，减少数值不一致

### 9.4 数据工程创新
- Page-level PDF parsing：动态选择低成本/高成本 parser，质量提升 25 倍数据纯度
- Domain-centric web parsing：URL domain 级别策略，识别结构模式
- Scientific recall：分类器自动进化，科学数据比例 2% → 50%

---

## 10. 参考链接

- **论文**: https://arxiv.org/abs/2508.15763
- **模型权重**: https://huggingface.co/internlm/Intern-S1

---

Intern-S1 代表了一次重要的技术突破：**首次在科学专业领域，开源模型不仅追平而且超越了闭源 SOTA 模型**（尤其是在化学和材料科学相关任务上）。这得益于其在多模态科学数据表示（Dynamic Tokenizer）、大规模异构任务强化学习（MoR）、以及精细数据工程上的系统性创新。对于从事 AI for Science 的研究者来说，这是一个极具参考价值的开源成果。