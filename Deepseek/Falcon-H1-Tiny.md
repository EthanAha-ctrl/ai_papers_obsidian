让我先完整阅读这篇论文的内容，然后给你详细解析。








# Falcon-H1-Tiny: 极小规模语言模型的范式突破 —— 全方位深度解析

---

## 一、论文全景概览

这篇论文来自 **Technology Innovation Institute (TII)** 的 **Falcon-LLM Team**，发表于 2026 年 1 月 15 日。核心主张是：**在 ≤100M 参数的极端小规模下，通过系统性地探索 data strategy（数据策略）与 architecture/optimization（架构与优化）两个正交轴，可以构建出在特定目标领域具有卓越性能的专门化语言模型系列**。

论文的哲学内核可以用一句话概括：

> **"A future that might rely on a series of tiny specialized models rather than bigger and more generalist models."**

这是一个从"大而全"到"小而精"的范式转变。

### 发布的模型家族

| 模型名 | 参数量 | 定位 |
|--------|--------|------|
| Falcon-H1-Tiny-90M-Base | 90M | English base model |
| Falcon-H1-Tiny-90M-Instruct-Curriculum | 90M | 传统 Curriculum SFT + DPO |
| Falcon-H1-Tiny-90M-Instruct (SFT-pretrain) | 90M | Anti-curriculum SFT pretrain + DPO |
| Falcon-H1-Tiny-Multilingual-100M-Base | 100M | 多语言 base |
| Falcon-H1-Tiny-Multilingual-100M-Instruct | 100M | 多语言 instruct |
| **Falcon-H1-Tiny-R-0.6B** | 600M | **SoTA 小规模推理模型** |
| Falcon-H1-Tiny-R-90M | 90M | 推理预训练小模型 |
| Falcon-H1-Tiny-Coder-90M | 90M | Python 代码生成 + FIM |
| Falcon-H1-Tiny-Tool-Calling | 90M | Function calling 专精 |

---

## 二、核心创新点：三大技术支柱

### 支柱 1: Memorization-Aware Repetition（记忆感知重复）

#### 第一性原理分析

传统 pretraining 的隐含假设是：**数据只能过一遍**（single-epoch）。这个约束来自于对 overfitting 的恐惧。但让我们从第一性原理重新审视：

**问题建模：** 设我们有高质量数据源包含 $D_{HQ}$ tokens，占总数据混合的比例为 $p$。则该 HQ 数据的 epoch size 为：

$$D_{ep} = \frac{D_{HQ}}{p}$$

在 single-epoch 训练中，HQ 数据比例的上界为：

$$p \leq p_{\max} \equiv \frac{D_{ep}}{T}$$

其中 $T$ 是总训练 token 数。这意味着 **HQ 数据的比例被训练时长所束缚**。

**关键发现：** 模型存在一个 **memorization window** $M$，即模型"遗忘"训练 token 所需的 token 数量。通过 FalconMamba-7B 的实验观测：

$$M \sim 100\text{GT} \quad (\text{乐观估计}), \quad M \sim 500\text{GT} \quad (\text{保守估计})$$

如果数据源的 epoch size 满足 $D_{ep} \lesssim M$，重复该数据会导致有害的记忆化；反之若 $D_{ep} \gtrsim M$，则重复是无害的。这直接解除了 HQ 比例与训练时长的耦合！

**直觉构建：** 把模型想象成一个"有漏的桶"——水（信息）从一端注入，从另一端流出。memorization window 就是桶的"容量"。只要两次注入同一滴水的间隔超过桶的容量（即水已经流走），就不会溢出（不会 overfitting）。

#### 实验验证

在 Falcon-H1-Tiny 的 800GT SFT-pretraining 中，SFT 数据源（如 Tulu3）被重复了多次，但未观察到 memorization 或 overfitting 的迹象。

---

### 支柱 2: Anti-Curriculum Strategy（反课程策略）

这是本文最核心的策略性创新。先理解什么是 "Curriculum" 和 "Anti-Curriculum"：

| 策略 | 阶段 1 | 阶段 2 | 直觉 |
|------|--------|--------|------|
| **Curriculum** | General pretrain（长） | SFT finetune（短，~4 epochs） | 先学基础，再学技能 |
| **Anti-Curriculum** | 混合 SFT + general data 的单阶段 pretrain | （可选的轻量 DPO） | 从一开始就浸入目标技能 |

#### 为什么 Anti-Curriculum 对小模型有效？

关键在于 **memorization window 与模型规模的缩放关系**：

- 100B 参数模型：$M \approx 5000\text{GT}$（线性缩放估计），SFT 数据 5GT → $p_{\max} = \frac{5}{5000} = 0.1\%$，比例太低，anti-curriculum 不够有效
- **100M 参数模型**：$M \approx 5\text{GT}$（线性缩放估计），SFT 数据 5GT → $p_{\max} = \frac{5}{5} = 100\%$，**可以完全用 SFT 数据预训练！**

**第一性原理解释：** 小模型的"遗忘"更快，因此可以更激进地重复数据而不 overfitting。这反过来意味着，小模型可以承受将目标领域数据从一开始就混入 pretraining，而不用担心过拟合——这正是 anti-curriculum 的理论根基。

#### 实验数据对比

**English Instruct 模型 IFEval 对比：**

| 模型 | 策略 | IFEval |
|------|------|--------|
| Falcon-H1-Tiny-90M SFT | Curriculum | 40.77 |
| Falcon-H1-Tiny-90M SFT + DPO | Curriculum + DPO | 53.47 |
| Falcon-H1-Tiny-90M SFT-pretrain | Anti-curriculum | 50.11 |
| **Falcon-H1-Tiny-90M SFT-pretrain + DPO** | **Anti-curriculum + DPO** | **66.08** |

Anti-curriculum + DPO 的 IFEval 比 Curriculum + DPO 高出 **~13 分**！

**Reasoning 模型对比（90M）：**

| Benchmark | Reasoning SFT (Curriculum) | Reasoning Pretraining (Anti-curriculum) |
|-----------|---------------------------|----------------------------------------|
| AIME24 pass@16 | 3/30 | **6/30** |
| AIME25 pass@16 | 2/30 | **9/30** |
| MATH500 | 0.2 | **0.4** |

Reasoning pretraining 的优势是压倒性的。

---

### 支柱 3: Learnable Multipliers (LRM) + Muon Optimizer

#### Muon Optimizer

Muon 是 AdamW 的替代优化器，基于矩阵正交化的动量方法。本文采用 Liu et al. (2025) 的修改版本：
- 应用 weight decay
- 缩放 update 的 RMS norm 以匹配 AdamW 的值

观测结果：**训练稳定，最优 LR 与 AdamW 几乎相同，但模型评估结果更优**。

#### Learnable Multipliers (LRM)

这是来自 Velikanov et al. (2026) 的核心创新。让我从第一性原理解释：

**问题：Noise-WD Equilibrium Trap**

在标准训练中，权重矩阵 $\mathbf{W}$ 的行的范数被锁定在一个由 LR 和 WD 决定的平衡点：

$$\|\mathbf{w}_i\| \approx \sqrt{\frac{\eta \cdot \sigma^2}{\lambda}}$$

其中 $\eta$ 是 learning rate，$\sigma^2$ 是梯度噪声方差，$\lambda$ 是 weight decay 系数。这意味着 **矩阵行/列的范数是由超参数决定的，而不是从数据中学到的**——这是一个"陷阱"！

**解决方案：** 给权重矩阵的每一行和每一列附加一个可学习的标量乘子（learnable multiplier）：

$$\mathbf{W}' = \text{diag}(\mathbf{r}) \cdot \mathbf{W} \cdot \text{diag}(\mathbf{c})$$

其中 $\mathbf{r} \in \mathbb{R}^{m}$ 是行乘子，$\mathbf{c} \in \mathbb{R}^{n}$ 是列乘子，都是可学习参数。这样矩阵的行/列范数就可以自由地从数据中学习，突破 noise-WD 平衡的束缚。

**实验验证（200GT 训练）：**

| Benchmark | Muon Baseline | Muon + LRM | 相对提升 |
|-----------|---------------|------------|----------|
| MMLU | baseline | ↑ up to 20% | 显著 |
| BBH | baseline | ↑ up to 20% | 显著 |
| GSM8K | baseline | ↑ up to 20% | 显著 |

**直觉：** 想象一栋建筑，标准训练好比所有房间的天花板高度都由建筑规范（超参数）统一规定。LRM 则允许每个房间根据自己的需求（数据）来决定天花板高度——有的需要挑高，有的只需要标准高度。这种自由度使得模型可以更高效地分配参数容量。

---

## 三、Model Architecture Ablations（架构消融实验）

所有模型基于 **Falcon-H1 架构**——一个将 **parallel Mamba + Attention heads** 结合在 mixer block 中的混合架构。

基础配置：
- Vocab size: 32768
- Embedding 与 projection head 共享参数（tied embedding）
- Tokenizer 包含常见 LaTeX token、数字和标点分割

### Exploration 1: Depth vs Width

固定 90M 参数预算，比较三种配置：

| 配置 | 层数 | 特点 |
|------|------|------|
| Shallow | 少 | 宽 |
| Mid | 27 | 中等 |
| Deep | 50 | 窄 |

**关键发现：**
- Mid-to-deep 模型在 Hellaswag（English commonsense）上明显更好
- Shallow 模型在 BBH（reasoning）上更好
- 但 50 层 vs 27 层带来约 **2x 的训练吞吐量下降**
- **最终选择 Mid 架构（24层）**——在性能和吞吐量间取得平衡

**直觉：** Depth 带来的信息流动路径更长，有利于需要多步推理的任务，但对于极小模型，过深的网络会导致严重的训练效率问题和梯度传播困难。24 层是"甜蜜点"。

### Exploration 2: MLP Factor

研究了在固定参数预算下，SSM（State Space Model）维度与 MLP 维度的 trade-off：

| 配置 | d_ssm | d_mlp | hidden_size | 结论 |
|------|-------|-------|-------------|------|
| cfg2 (baseline) | 768 | 768 | 512 | **最优** |
| cfg5 | 256 | 2360 | 368 | 差 |
| cfg6 | 256 | 1700 | 448 | 差 |
| cfg7 | 256 | 1280 | 512 | 略差 |
| cfg8 | 256 | 700 | 640 | 略差 |

**核心结论：**
1. **High-SSM 配置比 High-MLP/Low-SSM 配置 loss 下降更快** → SSM 容量对 tiny model 更有价值
2. 对于固定的 d_ssm，存在最优的 hidden_size-to-mlp 比率
3. Moderate mlp_size + high d_ssm + moderate hidden_size 是最佳组合

**第一性原理解释：** SSM（Mamba）层本质上是 **序列建模引擎**，通过隐状态压缩长程依赖；MLP 层是 **知识存储引擎**，通过非线性变换记忆事实。对于 tiny model，参数极其有限，序列建模能力（SSM）是更稀缺的资源——因为一个不足的 SSM 意味着模型连"看到"上下文都做不到，更不用说利用上下文了。

### Exploration 3: Attention Channels

增加 KV heads 并将节省的参数补偿到 MLP 中：

**发现：** 增加 KV heads 有帮助，但存在最优的 total_heads / kv_heads 比率。Baseline 配置仍然是最优的。

---

## 四、Falcon-H1-Tiny-English：完整生命周期

### Base Model 训练配置

| 参数 | 值 |
|------|-----|
| 总训练 | 800 GT |
| LR | 256e-5 |
| Weight Decay | 0.1 |
| Batch Size | 4 MT（million tokens） |
| LR Schedule | WSD (Warmup-Stable-Decay) |
| Warmup | 100 MT |
| Decay | x64 exponential decay over 100 GT |
| Power Scheduler | Square root LR decay from 100GT |
| Batch Rampup | 40GT, LR 按 √(batch_size) 缩放 |
| μP | 35 个 tuned multipliers 从 Falcon-H1 迁移 |

### Web Data 比例消融

| Web data 比例 | STEM 性能 | English Commonsense |
|---------------|-----------|---------------------|
| 10% | 基准 | 基准 |
| 20% | 基本不变 | **更好** |

选择 20% web data。

### SFT Duration Sweep（Curriculum 模型）

| 实验 | LR | Decay Duration | Total SFT Duration |
|------|-----|----------------|-------------------|
| lr-256-1gt-2gt | 256e-4 | 1GT | 2GT |
| lr-256-2gt-4gt | 256e-4 | 2GT | 4GT |
| lr-256-8gt-16gt | 256e-4 | 8GT | 16GT |
| lr-256-24gt-32gt | 256e-4 | 24GT | 32GT |
| lr-128-2gt-4gt | 128e-4 | 2GT | 4GT |
| lr-512-2gt-4gt | 512e-4 | 2GT | 4GT |

**最优选择：** LR = 256e-4，SFT duration = 10GT（从 lr-256-8gt-16gt 的 best checkpoint 取出）。

### Anti-Curriculum 模型的 SFT 比例消融

测试了 0%, 25%, 50%, 75%, 100% 的 SFT 数据比例：

| SFT 比例 | Base 性能 | IFEval | 结论 |
|----------|-----------|--------|------|
| 0% | 最好 | 低 | 纯 base |
| 25% | 接近 0% | 好 | **最佳平衡** |
| 50% | 略低 | 好 | 与 25% 相当 |
| 75% | 低 | 中 | 过多 SFT 有害 |
| 100% | 低 | 中 | 缺乏 pretrain 基础 |

**最终选择：25% SFT + 75% base mixture**，训练 800GT。

### DPO 对 Tiny 模型的效果

LR sweep：1e-5, 3e-7, 3e-6, 1e-6

**关键发现：**
- **超过 1 epoch 的 DPO 会导致严重的性能退化**，尽管 DPO reward 在增加（典型的 reward hacking 信号！）
- 最优 LR：1e-6 到 3e-6
- **1 epoch DPO 就能带来 IFEval 从 ~50 跳到 65+ 的巨大提升**

### 最终 English 模型性能对比

**Base Models：**

| Benchmark | Falcon-H1-Tiny-English-Base | Mobile-LLM-140m-R1 | Smol-LM-135M |
|-----------|---------------------------|---------------------|--------------|
| MMLU | 32.3 | 24.4 | 24.2 |
| GSM8K | 2.6 | 20.2 | 1.2 |
| IFEVAL | 30.07 | 18.6 | 18.4 |

**Instruct Models：**

| Benchmark | Curriculum+DPO | **SFT-pretrain+DPO** | SmolLM2-135M-Instruct |
|-----------|---------------|---------------------|----------------------|
| IFEVAL | 53.47 | **66.08** | 30.69 |
| Alpaca Eval (win rate) | 10.44 | **9.43** | 1.52 |
| MT Bench (avg) | 4.4 | **4.33** | 2.68 |
| LiveBench (global) | 12.4 | **15.69** | 8.25 |

**SFT-pretrain + DPO 模型在 IFEval 上以 66.08 大幅领先**，在 LiveBench 上也是最强的。

---

## 五、Falcon-H1-Tiny-Multilingual：多语言挑战

### 架构调整

- Vocab size 增加到 65K（支持 17+ 种语言）
- 参数量增加到 100M
- 支持：Czech, German, Spanish, French, Hindi, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Romanian, Russian, Swedish, Urdu, Chinese

### 数据构成

**Multilingual Pretrain Corpus：**
- 50% Common Crawl 多语言混合
- 33% Wikipedia
- 17% Textbooks

**Anti-Curriculum (SFT-pretrain) 配置：**
- 40% English pretrain
- 20% English SFT
- 20% Multilingual pretrain（10% CC + 6.67% Wiki + 3.33% textbooks）
- 20% Multilingual SFT

### 关键发现

1. **Anti-curriculum 在多语言场景中没有带来明显优势**——与 English 实验结论不同！
   - 假设：多语言 SFT 数据质量不够高，或 100M 的容量限制使得模型无法从额外的训练数据中受益
   
2. **DPO 对多语言 tiny 模型同样有效**——是指令跟随能力的主要提升来源

3. **多语言能力在 100M 规模上是有限的**——虽然相对同等规模的模型有竞争力，但绝对性能依然 modest

**Instruct Models 对比：**

| Benchmark | SFT-pretrain | Curriculum-SFT | Curriculum-SFT-DPO | SmolLM2-135M |
|-----------|-------------|----------------|-------------------|--------------|
| IFEVAL | 46.50 | 43.83 | **52.00** | 30.69 |
| multilingual_mmlu | **55.00** | 51.00 | 45.00 | 25.63 |
| multilingual_mgsm | 4.07 | 4.73 | **4.67** | 1.40 |

---

## 六、Falcon-H1-Tiny-R：推理模型的预训练范式革命

这是本文最有野心的部分：**直接在推理数据上预训练，而不是先 general pretrain 再 reasoning SFT**。

### 方法论

**传统推理模型三阶段：**
1. General pretraining
2. Reasoning SFT
3. RL (GRPO)

**本文的 Anti-curriculum 推理模型：**
1. 直接在 reasoning data 上 pretraining（合并阶段 1+2）
2. GRPO RL

### 训练策略

采用 WSD schedule，但特意延长了 decay 阶段：

| 阶段 | Token 数 |
|------|----------|
| Warmup | 100M |
| Constant LR | 500 GT |
| Decay Stage 1 | x4 exponential decay over 50 GT |
| Decay Stage 2 | x256 exponential decay over 350 GT |
| **总计** | **900 GT** |

**关键观察：** 大部分性能增益来自 **decay 阶段**！这与一般 pretraining 的经验不同，说明推理能力可能在训练后期才"涌现"。

### 核心结果

| Benchmark | Falcon-H1-Tiny-R-0.6B (post-GRPO) | Falcon-H1-Tiny-R-0.6B (pre-GRPO) | Falcon-H1-Tiny-R-0.09B | Qwen3-1.7B | OpenReasoning-Nemotron-1.5B | DeepSeek-R1-Distill-Qwen-1.5B |
|-----------|-----------------------------------|-----------------------------------|------------------------|-------------|---------------------------|-------------------------------|
| **AIME24 (pass@1)** | **75.0** | 67.5 | 5.0 | 47.0 | 49.7 | 29.13 |
| **AIME25 (pass@1)** | **67.3** | 60.0 | 7.9 | 37.0 | 40.4 | 23.43 |
| **LCBv6** | **39.0** | 35.0 | 4.5 | 29.8 | 28.3 | 19.92 |
| **Math500** | **94.0** | 92.5 | 39.7 | 89.4 | 83.4 | 83.28 |

**令人震惊的结果：** 0.6B 参数的 Falcon-H1-Tiny-R 在 AIME24/AIME25 上 **大幅超越** 1.5B-1.7B 的竞争模型！

### GRPO 的效果

60 步 GRPO（32K context length）带来的提升：

| Benchmark | Pre-GRPO | Post-GRPO | Δ |
|-----------|----------|-----------|---|
| AIME24 pass@1 | 67.5 | 75.0 | +7.5 |
| AIME25 pass@1 | 60.0 | 67.3 | +7.3 |
| LCBv6 accuracy | 35.0 | 39.0 | +4.0 |
| Math500 accuracy | 92.5 | 94.0 | +1.5 |

**附带发现：** GRPO 还缩短了生成长度——从约 16K tokens 降到约 8K tokens，相当于模型学会了"更高效地思考"。

### 90M 推理模型的局限

- 90M 模型虽然超过了 Mobile-LLM-R1 系列，但仍然存在 **repetition trap（重复陷阱）**——倾向于重复相同的 tokens
- 这与 Pipis et al. (2025) 的发现一致：更小的模型更容易陷入循环行为
- **Repetition penalty 可以缓解但无法根治**
- 推理能力的"涌现阈值"被推低到了 0.6B，但 90M 仍在阈值之下

---

## 七、Falcon-H1-Tiny-Function-Calling：工具调用专精

### Reasoning Loop Problem（推理循环问题）

这是本文的一个重要发现，值得深入讨论：

**现象：** 当训练数据中包含 chain-of-thought reasoning traces 时，90M 模型会产生 **无限生成循环**——不断重复相同的文本片段，而不是生成干净的函数调用。

**根因分析（基于 Pipis et al., 2025）：**
- 当训练分布中包含超过模型学习能力的复杂推理模式时
- 模型学到了"最简单的可用行为"——重复
- 这不是在"推理"，而是在 **模仿推理的表面模式**

**解决方案：** 从训练数据中过滤掉所有 reasoning 和 thinking 内容，只保留直接的 tool calling 示例。**效果立竿见影。**

### 训练策略对比

与 English 实验不同，**Anti-curriculum 和 Curriculum-SFT 在 function calling 上表现几乎相同**！

**解释：** Tool calling 的能力（JSON 语法、schema 匹配、参数提取）受限于模型的 **最大特征学习容量**，而非数据暴露的时机或重复次数。

### Tool Calling Data 比例缩放

| Tool Calling % | Global BFCL v3 Score |
|----------------|---------------------|
| 20% | 32.1% |
| 50% | 36.8% |
| 75% | 39.4% |
| 85% | **41.2%** |

单调递增但有递减收益。

### 与大模型对比

| 模型 | Size | Non-Live AST | Relevance | Multi-Turn | Global |
|------|------|-------------|-----------|------------|--------|
| Qwen3-0.6B | 600M | 71.79% | 56.62% | 80.84% | 57.57% |
| Function Gemma | 270M | 48.40% | 61.10% | 70.60% | 41.30% |
| **Falcon-H1-Tiny-FC** | **90M** | 36.06% | **94.44%** | 61.37% | **41.23%** |

**惊人发现：** 90M 模型的 **Relevance Detection 达到 94.44%**，远超 Function Gemma 的 61.10%！但在 AST accuracy 上落后。这说明小模型的容量被更好地用于理解"何时调用"，而非"如何完美格式化"。

---

## 八、Falcon-H1-Tiny-Coder：口袋里的 Python 程序员

### FIM（Fill-in-the-Middle）格式

采用 PSM（Prefix-Suffix-Middle）格式：

```
<|prefix|>{prefix}<|suffix|>{suffix}<|middle|>{middle}
```

### 关键技术发现

1. **缩进问题：** 模型在从缩进行自动补全时，会预测一个全新的函数而不是继续当前函数。**解决方案：** 构造 50% 的 FIM 样本需要预测缩进。

2. **Masking vs Un-masking：** 是否对非 FIM tokens（prefix/suffix）应用 loss masking？
   - **结论：Un-masking 更好！** 不遮蔽非 FIM tokens 在给定 token/compute 预算下更高效。

3. **Dropout 对数据重复的作用：** 在极端数据重复场景下，dropout（p=0.1）可以缓解 HumanEval-FIM 上的退化，虽然其他任务略有损失。

### 最终性能

| Benchmark | Falcon-H1-Tiny-Coder-90M | Qwen2.5-Coder-0.5B |
|-----------|-------------------------|---------------------|
| HumanEval+ (@1) | 14.63 | 23.17 |
| MBPP (@1) | 41.26 | 54.76 |
| HumanEval-FIM-RS (@1) | 30.96 | 31.76 |
| HumanEval-FIM-RS (@10) | 56.7 | 56.7 |

**亮点：** 在 HumanEval-FIM-RS (@10) 上，90M 模型 **追平了** 0.5B 的 Qwen2.5-Coder！

---

## 九、统一框架：Tiny Model 的设计原则

从整篇论文中，我可以提炼出以下设计原则：

### 原则 1: Memorization Window 是小模型的超能力
$$M_{\text{small}} \ll M_{\text{large}} \implies p_{\text{mem}}^{\text{small}} \gg p_{\text{mem}}^{\text{large}}$$

小模型的遗忘更快 → 可以更激进地重复数据 → 可以从 pretraining 阶段就注入目标领域数据。

### 原则 2: Anti-Curriculum 对小模型有效，但不是万能的
- ✅ English instruct（IFEval +13 分）
- ✅ Reasoning（AIME 翻倍）
- ❌ Multilingual（无明显优势）
- ❌ Function Calling（与 Curriculum 持平）

**条件：** Anti-curriculum 的效果取决于目标数据的质量和模型是否有足够容量来吸收它。

### 原则 3: SSM 容量 > MLP 容量（对 Tiny Models）
在参数预算有限时，优先分配给 SSM（序列建模）而非 MLP（知识存储）。

### 原则 4: CoT 对极小模型是有毒的
Chain-of-thought 推理 traces 超过小模型的压缩能力 → 导致 repetition loop。**对 90M 模型，直接输出比"思考"更有效。**

### 原则 5: DPO 对 Tiny Models 有效但极其脆弱
- 只需 1 epoch
- LR 需要精确调整
- 超过 1 epoch 会导致 reward hacking 式的退化

### 原则 6: GRPO 可以显著提升推理能力
即使在 0.6B 规模，GRPO 也能带来 AIME +7.5 的提升，同时缩短生成长度。

---

## 十、论文的局限与未来方向

### 局限

1. **Memorization window 的线性缩放假设** 未经严格验证——从 7B 线性缩放到 100M 可能过于简化
2. **多语言 anti-curriculum 失效的原因** 未深入分析
3. **90M 推理模型的 repetition trap** 只有缓解方案，无根治方法
4. **缺乏与同规模的 closed-source 模型对比**
5. **Function Calling 的 AST accuracy 低**——实际部署中格式错误是致命的

### 未来方向（论文提及）

1. **Model Merging**：多个 tiny specialized models 能否通过 merge 组合成更强大的模型？
2. **Data Repetition 的深入研究**：Dropout 等正则化技术与数据重复的交互
3. **Extreme Quantization**：4-bit 量化在 90M 规模的影响
4. **Merging RL with Pre-training**：将 RL 阶段也合并到 pretraining 中
5. **Scaling Laws for Tiny Models**：90M-100M 是否是某些任务（如多语言）的理想规模？

---

## 十一、关联工作与延伸阅读

| 主题 | 相关工作 | 链接 |
|------|---------|------|
| Falcon-H1 主模型 | Zuo et al. (2025) | https://arxiv.org/abs/2507.22448 |
| Learnable Multipliers | Velikanov et al. (2026) | https://arxiv.org/abs/2601.04890 |
| Muon Optimizer | Liu et al. (2025) | https://arxiv.org/abs/2502.16982 |
| Reasoning Models Loop | Pipis et al. (2025) | https://arxiv.org/abs/2512.12895 |
| Mobile-LLM-R1 | Zhao et al. (2025) | https://arxiv.org/abs/2509.24945 |
| SmolLM2 | Allal et al. (2025) | https://arxiv.org/abs/2502.02737 |
| GRPO / DeepSeekMath | Shao et al. (2024) | https://arxiv.org/abs/2402.03300 |
| Falcon-H1R | Team et al. (2026) | https://arxiv.org/abs/2601.02346 |
| FIM Pretraining | Bavarian et al. (2022) | https://arxiv.org/abs/2207.14255 |
| Qwen2.5-Coder | Hui et al. (2024) | https://arxiv.org/abs/2409.12186 |
| Qwen3 | Yang et al. (2025) | https://arxiv.org/abs/2505.09388 |
| BFCL | Patil et al. (2025) | ICML 2025 |
| Structure-Aware FIM | Gong et al. (2025) | https://arxiv.org/abs/2506.00204 |

---

## 十二、总结：这篇论文的真正价值

这篇论文的核心贡献不在于某个单一的技术突破，而在于 **一个完整的、可复现的 tiny model 训练方法论**：

1. **Memorization-aware repetition** 打破了"数据只能过一遍"的教条
2. **Anti-curriculum** 颠覆了"先基础后技能"的训练范式
3. **LRM + Muon** 提供了优化层面的新工具
4. **SSM-prioritized architecture** 给出了 tiny model 的架构设计原则
5. **CoT-toxicity finding** 揭示了小模型训练中一个容易被忽视的陷阱

最重要的是，论文展示了一个令人兴奋的未来方向：**不是追求一个万能的大模型，而是部署一组各有所长的小模型**。每个 90M 模型只需要 ~90MB 的内存（8-bit 量化后），可以在任何设备上实时运行。这种"小而精"的哲学，可能会重新定义我们思考和部署 AI 的方式。