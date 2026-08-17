---
source_pdf: Scaling Law for Quantization-Aware Training.pdf
paper_sha256: 2a89042b70bde87ad5caa8038719a1e620f1e9d87f51818e68a4dc42059366ea
processed_at: '2026-08-12T03:35:17-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 Paper

好，我换个讲法，不堆公式，讲讲这帮人到底发现了啥。

---

## 这篇 Paper 在纠结什么

一句话：**给 quantization-aware training（QAT）建了个 scaling law，让你能预测 4-bit 模型到底能跑多差。**

背景是这样的。你想把 LLM 部署到 GPU 上，但 70B 模型一张卡塞不下，推理又慢，所以大家想 quantization——把 weight 和 activation 从 16-bit 压到 4-bit。PTQ（训练完再量化）在 8-bit 下还行，但 4-bit 基本崩了。所以只能 QAT——训练的时候就把量化塞进去，让模型自己适应低 bit。

问题来了：QAT 在 4-bit 下到底表现怎么样？它会随 model size、训练 data、quantization granularity 怎么变？你总不能每次都真训一个 70B 模型试一试，那烧不起。所以需要个 scaling law 来预测。

之前其实有人搞过 QAT scaling law，但这帮作者一看，之前的工作有个大 bug：**它们假设 quantization error 只跟 model size N 有关，跟训练 data D 无关**。这帮作者跑了两百多个实验，发现这假设是错的——**训练 token 越多，quantization error 反而越大**。这个发现挺反直觉的，因为常识里"训练越多越好"嘛，但 quantization 的世界里不是。

---

## 三大核心发现

### 发现一：模型越大，quantization 误差越小

这个挺好理解的。大模型参数多，单个 weight 抖一下对 output 影响小，因为它有 redundancy。就像你删掉 Google 一个工程师没事，删掉 5 个人的 startup 就完蛋。模型大了，4-bit 表示带来的噪声被稀释掉了。

74M → 595M，误差平均降 34%。

### 发现二：训练数据越多，quantization 误差越大

**这是这篇 paper 最反直觉、最重要的发现。**

10B → 100B tokens，误差平均涨 22%。

直觉解释有两种，我倾向第一种：

**Underfitting 假说**：BF16 模型一直在学新东西，loss 一直降。QAT 模型因为被 4-bit 锁住了表示能力，学不动那么细的 pattern 了，loss 降得慢。所以 gap 越拉越大。就像两个人跑步，BF16 一直加速，QAT 到某个速度就上不去了，距离越来越远。

**Weight 复杂化假说**：训练越久，weight 分布越复杂——更多 outliers、更尖锐的 spectrum、更窄的有效 rank。4-bit 表示装不下这种复杂性。

实际验证下来，weight error 对 D 的敏感度（γ_D=0.16）是 activation error 对 D 敏感度（γ_D=0.03）的 5 倍。所以**第二个发现主要讲的是 weight 越训越复杂**。

这个发现跟 PTQ 的结论方向一致（[Ouyang 那篇 PTQ scaling law](https://arxiv.org/abs/2411.17691) 也说 over-trained 模型 PTQ 更难），但 QAT 的增长比 PTQ 慢一个数量级，因为 QAT 在训练时就 adapt 了。

### 发现三：quantization granularity 越粗，误差越大

Granularity 就是"几个 element 共享一个 scale"。G=32 就是每 32 个数一个 scale，G=per-tensor 就是整个矩阵一个 scale。

直觉：activation 里有 outliers（少数特别大的值）。如果一组里有个 outlier，整个组的 scale 被它拉爆，正常的值全被挤到低 bit 的几个码字附近，精度全毁。组越小，outlier 只影响自己那几个邻居，正常值的精度保住。

最粗和最细粒度之间误差差 0.037，这个数大概是粗粒度本身误差的一半，说明 granularity 影响巨大。

---

## 把误差拆开：Weight vs Activation

光知道总误差不够，得知道是 weight 量化坏的事还是 activation 量化坏的事，不然算法没方向。

但有个坑：你训完一个 W4A4 模型，不能简单把 quantization 关掉看每个的 contribution，因为训练时 weight 已经 adapt 了。所以他们用了**三个独立训练**的招：

- W4A4 模型（weight 和 activation 都 4-bit）
- W4A16 模型（只 weight 4-bit，activation 保留 16-bit）→ 这个误差就是 weight 误差
- W16A4 模型（只 activation 4-bit，weight 保留 16-bit）→ 这个误差就是 activation 误差

发现：**δ_W4A4 ≈ 0.906 × (δ_W4A16 + δ_W16A4)**。0.906 说明联合量化时两者有点互相抵消（因为 W4A4 训练时模型学会同时适应两边），但近似加性。

**关键对比**：

| 误差来源 | 对 N 敏感 | 对 D 敏感 | 对 G 敏感 |
|---------|----------|----------|----------|
| Weight | 强（γ_N=0.36） | 超强（γ_D=0.16） | 弱（γ_G=0.35） |
| Activation | 弱（γ_N=0.18） | 弱（γ_D=0.03） | 超强（γ_G=0.98） |

翻译成人话：
- **Weight error 怕大数据**——D 一大 weight error 涨得快
- **Activation error 怕粗粒度**——G 一粗 activation error 涨得飞快，因为 outliers
- **两个都怕小模型**，但 weight 更怕

那到底哪个更大？在大多数 setting 下，**activation error 大于 weight error**（比值 R 在 1.2-1.7 之间）。但 R 随 D/N 增大而下降——**over-training 时 weight error 会追上来**。

---

## FC2 的 SwiGLU 是元凶

Activation error 大，但具体哪层最坏？作者量了每层 input 的 kurtosis（峰度，衡量尾部厚度，越大说明 outlier 越多）。

发现 QAT 训练能把 QKV、O、FC1 三层的 kurtosis 都压下来，但 **FC2 input 的 kurtosis 死活压不下来**，从 123 降到 89，其他层都降到 10 以下了。

**为什么 FC2 这么特殊？** 看 LLaMA 的 architecture：FC2 的 input 是 SwiGLU 的输出，SwiGLU = `Swish(x·W_gate) ⊙ (x·W_up)`。这个 ⊙（element-wise 乘）是关键——gating 让某些 channel 被放大，element-wise 乘让放大效应相乘，分布尾部被指数级拉伸。

数学上讲，两个独立分布相乘，结果的 kurtosis 是两个分布 kurtosis 的乘积。SwiGLU 等于把两个激活分布乘起来，kurtosis 必爆。这是结构性的，QAT 的 regularizer 抑制不了。

**解法很暴力**：FC2 input 用 8-bit，其他层还是 4-bit。8-bit 是 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 证明过的 near-lossless 精度。

效果：
- G=32 时总误差降 20.5%
- G=256 时降 42.9%（粗粒度受益更大，因为 outliers 在粗粒度下最致命）
- Activation error 对 granularity 的敏感度 γ_G 从 0.98 降到 0.45，基本不敏感了
- R 比值从 1.2-1.7 降到 0.85-1.10，**weight 和 activation 误差打平**

---

## 对实践的意义

我总结几条 actionable 的：

1. **别在 over-trained 模型上指望 4-bit QAT 奇迹**。D/N 越大，weight quantization 越成问题，过去大家只盯 activation outliers，现在 weight 也得管。

2. **FC2 input 用 8-bit，性价比极高**。其他层 4-bit 推理快，FC2 input 8-bit 推理慢一点点但精度大涨。这是 mixed-precision 的 sweet spot。

3. **Granularity 一定要 fine**。G=32 比 per-token 好太多，activation error 几乎减半。但太细也不划算——因为 log_2(G) 项是 sublinear 的，G=32 到 G=8 收益递减，且推理时 scale storage 成本变高。

4. **大模型 quantization 更友好，但不是"救世主"**。N 增大能让 δ 降，但衰减速度比 BF16 loss 衰减慢，所以**大模型的 quantization 相对损失反而更大**。这点跟"大模型更 robust to quantization"的直觉不完全一致——大模型绝对损失小，但相对损失在变大。

5. **EPM 指标算下来，4-bit 仍然是 Pareto optimal**，比 8-bit 划算（前提是 EPM > 0.5）。这跟 [Kumar 那篇](https://arxiv.org/abs/2411.04330) 之前说 8-bit optimal 的结论矛盾，原因是 Kumar 用了 per-tensor activation quantization，outliers 把 4-bit 搞崩了，本文用 fine-grained 后 4-bit 就翻身了。

---

## 我对这篇 Paper 的看法

**优点**：
- 268 个实验，5 个 model size × 5 个 token 数 × 6 个 granularity × 多种 precision 设置，量级够大，结论可信
- 数学形式简洁，公式只有 4 个参数，实用
- FC2 + SwiGLU 的归因分析做得很到位，找到了数学结构上的根因

**局限**（作者自己承认）：
- 最大只到 595M / 100B tokens，外推到 70B+ 模型还有不确定性
- 只测 dense 模型，MoE 没测。MoE 的 weight 多 activation 一样多，weight/activation 误差比例会不一样
- 没测 ternary（1.58-bit），BitNet 系列的 scaling law 还得另建
- FC2 用 8-bit 是 upper bound 方案，更精细的 outlier suppression（[QuaRot](https://arxiv.org/abs/2404.00456) 的 rotation、[PrefixQuant](https://arxiv.org/abs/2410.05265) 的 prefix token）能不能达到类似效果，需要再验证

**值得 follow up 的方向**：
- 把 scaling law 扩展到 MoE 和 ternary
- 研究 rotation-based 方法（QuaRot、SpinQuant）能不能数学上把 SwiGLU 的 kurtosis 项消除掉，从而改变 γ_G
- Backward 也 quantize 的 FQT scaling law——本文只 quantize forward，backward quantization 是新的 error source
- Mixed-precision 的最优分配问题：哪些层 8-bit、哪些 4-bit，应该是个 constrained optimization，本文只试了 FC2 一层

---

**一句话总结**：这篇 paper 把 QAT 的 quantization error 拆解成 N、D、G 三个变量的可预测函数，发现**数据多了 weight 误差会爆**和**SwiGLU 的 FC2 是 activation outlier 的死穴**两个核心 insight，给未来 QAT 算法指了方向：别只盯 activation outlier，weight 和 activation 都得管，FC2 这种乘法结构得用 mixed precision 兜底。

---

# QAT Scaling Law 深度解析

## 1. 背景：为什么需要 QAT 的 Scaling Law

LLM 部署的核心矛盾在于 model size 与 compute cost 之间的张力。Quantization 是缓解这个矛盾的主要手段，其中 W8A8 已经被广泛证明 near-lossless（见 [SmoothQuant](https://arxiv.org/abs/2211.01038)、[FP8-LM](https://arxiv.org/abs/2310.18313)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)），但 W4A4 在 PTQ 下 accuracy 严重下降（[Quantization hurts reasoning](https://arxiv.org/abs/2504.04823)）。QAT 通过把 quantization 嵌入训练过程让模型自适应低 bit 表示，是 W4A4 场景下唯一可行的路径（[BitNet](https://arxiv.org/abs/2402.17764)、[EfficientQAT](https://arxiv.org/abs/2407.11062)、[QUEST](https://arxiv.org/abs/2502.05003)）。

Scaling law 的意义在于：给定 budget 优化训练策略、预测 large scale 行为、识别 bottleneck。已有的 QAT scaling law 主要有两类：
- [Frantar et al. 2025](https://arxiv.org/abs/2502.16440) 提出 compression scaling law，把 sparsity 与 quantization 统一建模，核心思想是用 effective parameter multiplier (EPM) 修改 N 项；
- [Kumar et al. 2024](https://arxiv.org/abs/2411.04330) 提出 scaling laws for precision，研究 bit-width 的 Pareto frontier，结论是 8-bit Pareto-optimal。

**这两个工作的关键缺陷**：它们都假设 quantization error 只依赖 N，与 D 无关。本文通过 268 个实验证伪了这个假设，并加入 quantization granularity G 这个被忽略的关键变量。

---

## 2. 核心 Scaling Law 公式

### 2.1 Chinchilla 基准

[Chinchilla](https://arxiv.org/abs/2203.15556) 给出：

$$L(N, D) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + E$$

变量含义：
- **N**：model parameter count
- **D**：number of training tokens
- **A, B**：prefactor 常数，控制 irreducible 部分以外的 scaling 强度
- **α, β**：scaling exponent，描述 loss 随 N 和 D 衰减的速率
- **E**：entropy constant，代表数据集本身的不可压缩 loss（token-level entropy）

本文拟合得到：α = β = 0.3022（与 Chinchilla 原文 α=0.34, β=0.28 接近，[Kaplan](https://arxiv.org/abs/2001.08361) 给的 α=0.076 对 N 更悲观，已被 Chinchilla 修正）。这里强行约束 α = β 是为了对齐 Chinchilla 的"compute-optimal 时 N 和 D 等比例 scaling"结论。

### 2.2 既有 QAT Scaling Law（EPM 形式）

[Frantar](https://arxiv.org/abs/2502.16440) 与 [Kumar](https://arxiv.org/abs/2411.04330) 通过 EPM 改写 Chinchilla：

$$L(N, D) = \frac{A}{(N \cdot \text{eff}(\mathbf{C}))^{\alpha}} + \frac{B}{D^{\beta}} + E$$

其中 eff(C) ∈ [0,1] 是 effective parameter multiplier，依赖 architecture C。**关键假设**：eff(C) 是 N、D 的常数函数。这等价于说 quantization 只是把模型"等效缩小"为原来的 eff(C) 倍，loss curve 整体平移。

### 2.3 本文提出的 Quantization Error Scaling Law

本文直接建模 **quantization error** δ_p = loss_QAT - loss_BF16，而非改写 Chinchilla。观察三个 trend 后提出：

$$\boxed{\delta_p(N, D, G) = \frac{k \cdot D^{\gamma_D} \cdot (\log_2 G)^{\gamma_G}}{N^{\gamma_N}}}$$

最终 loss：

$$L(N, D, G) = \underbrace{\frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + E}_{\text{Chinchilla loss}} + \underbrace{\delta_p(N, D, G)}_{\text{low-bit QAT effect}}$$

**变量与符号**：
- **N**：model parameter count
- **D**：number of training tokens
- **G**：quantization group size（每组内共享 scale 的元素个数；G=32 即每 32 个 element 一个 scale，G 越小 granularity 越细）
- **k**：prefactor，反映 quantizer 本身的 baseline error
- **γ_N**：model size 敏感度 exponent（γ_N > 0 表示 N 增大时 δ 减小）
- **γ_D**：data 敏感度 exponent（γ_D > 0 表示 D 增大时 δ 增大）
- **γ_G**：granularity 敏感度 exponent（γ_G > 0 表示 G 增大即粗粒度时 δ 增大）
- **log_2(G)**：用对数是因为 G=1（无 quantization）时 δ=0 的边界条件要被满足；log 形式让 G 的影响是 sublinear 的

### 2.4 Fitted Parameters 深度解读

| Precision | k | γ_N | γ_D | γ_G |
|-----------|---|------|------|------|
| δ_W4A4 | 0.1582 | 0.2186 | 0.0745 | 0.7779 |
| δ_W4A16 | 0.2522 | 0.3589 | 0.1610 | 0.3533 |
| δ_W16A4 | 0.1004 | 0.1816 | 0.0331 | 0.9812 |
| δ_W4A4 (FC2 8-bit) | 0.3519 | 0.2637 | 0.0964 | 0.3407 |
| δ_W16A4 (FC2 8-bit) | 0.1273 | 0.2347 | 0.0827 | 0.4491 |

**直觉构建**：
1. **γ_N 比较**：δ_W4A16 的 γ_N=0.3589 远大于 δ_W16A4 的 0.1816。这意味着 weight quantization error 随 model 增大衰减更快。**直觉**：大模型有更多 redundancy，weight 的低 bit 表示损失被稀释；activation 是数据流，redundancy 不随 N 线性增长。
2. **γ_D 比较**：δ_W4A16 的 γ_D=0.1610 是 δ_W16A4 的 0.0331 的 5 倍。**这是本文最重要的发现之一**：训练 token 越多，weight quantization error 增长越快。**直觉**：更多数据让 weight 学到更精细、更复杂的 distribution（例如更尖锐的 outlier pattern、更窄的有效 rank），4-bit 容纳不下这种复杂性；而 activation distribution 在收敛后相对稳定。
3. **γ_G 比较**：δ_W16A4 的 γ_G=0.9812 远高于 δ_W4A16 的 0.3533。**直觉**：activation 有 outliers，粗粒度 quantization 用一个 scale 覆盖整组，outliers 把 scale 拉大导致 normal value 量化精度劣化；weight 没有强 outliers，所以 granularity 影响小。
4. **k 比较**：δ_W16A4 的 k=0.1004 比 δ_W4A16 的 0.2522 小，说明在 baseline 情况下 activation quantization error 比 weight 大。这与 R = δ_W16A4 / δ_W4A16 > 1 一致。

### 2.5 Contour 的几何含义（Appendix G 的精华）

固定 G 时，δ_p = C · D^{γ_D} · N^{-γ_N}，取 log10：

$$\gamma_D \log_{10} D - \gamma_N \log_{10} N = \text{const}$$

在 (log N, log D) 平面上是斜率为 γ_N/γ_D 的直线。对 W4A4，γ_N/γ_D = 0.2186/0.0745 ≈ 2.93。**这意味着等 error 线的斜率比 Chinchilla 的等 compute 线（斜率 1）陡得多**：要维持相同 quantization error，N 增加 1 个数量级只需要 D 增加 1/2.93 个数量级。换言之，**用大模型抵消更多训练数据带来的 quantization 退化是非常高效的**。

---

## 3. 实验设置详解

### 3.1 Model & Data

- Architecture：[LLaMA-3](https://arxiv.org/abs/2407.21783) style，含 GQA + SwiGLU
- Model size：74M / 145M / 297M / 595M / 973M（973M 用于 extrapolation 验证）
- Tokens：10B / 20B / 50B / 100B / 200B
- Dataset：[OLMo2-Mix-1124](https://arxiv.org/abs/2501.00656)
- 268 QAT experiments，276K A100 GPU-hours

| Model | Layers | Hidden | FFN | Heads | KV Heads | Max LR |
|-------|--------|--------|-----|-------|----------|--------|
| 74M | 12 | 768 | 2048 | 16 | 4 | 1.5e-3 |
| 145M | 12 | 1024 | 3072 | 16 | 4 | 1.0e-3 |
| 297M | 12 | 1536 | 4096 | 24 | 6 | 8e-4 |
| 595M | 24 | 1536 | 4096 | 24 | 6 | 6e-4 |
| 973M | 16 | 2048 | 8192 | 32 | 8 | 6e-4 |

### 3.2 三个 Quantization 设置

- **W4A4**：weight 与 activation 都 4-bit
- **W4A16**：仅 weight 4-bit，activation 保留 16-bit（用来 isolate weight error）
- **W16A4**：仅 activation 4-bit，weight 保留 16-bit（用来 isolate activation error）

### 3.3 Quantization 格式选择

INT4 vs FP4 对比（Figure 2，297M/50B）：
- group-wise quantization 下两者性能相当
- per-channel/token 下 INT4 比 FP4 好 0.015 loss
- 原因：INT4 有 16 个 representable value，FP4 (E2M1) 只有 15 个（[Wang et al. 2025](https://arxiv.org/abs/2501.17116)）

最终选 INT4。假设 INT 与 FP scaling behavior 一致，Figure 13 用 INT4 拟合的 law 验证可预测 FP4。

### 3.4 Quantizer 选择

| Quantizer | 公式 | 用途 |
|-----------|------|------|
| AbsMax | s = M / max(|X|) | weight 全部，activation G<256 |
| [LWC](https://arxiv.org/abs/2308.13137) | s = M / (max(|X|)·γ)，γ 可学 per weight group | weight 备选 |
| [LAC](https://arxiv.org/abs/2410.05265) | s = M / (max(|X|)·γ)，γ 共享 group index 跨 token | activation G≥256 |
| [LSQ](https://arxiv.org/abs/1902.08153) | s 直接可学 | 备选 |

**关键发现**（Figure 14）：activation 对 quantizer 极其敏感（outliers 影响），weight 不敏感。所以 weight 用最简单的 AbsMax，activation 按 granularity 切换。

### 3.5 Learning Rate 不敏感

Figure 3 显示 W4A4 在 LR ∈ [5e-4, 4e-3] 范围内 quantization error 几乎恒定（0.6-0.65）。这与 [BitNet](https://arxiv.org/abs/2402.17764) 的 ternary 需要 high LR 不同——4-bit 不够 aggressive，所以 LR 调整收益有限。这简化了实验：QAT 与 BF16 用相同 hyperparameter。

---

## 4. 三大核心 Trend 详解

### 4.1 N 增大 → δ 减小（Figure 4a）

74M → 595M，δ_W4A4 平均下降 34%。

**直觉**：模型变大意味着 weight 矩阵的 singular value spectrum 更平、column 之间正交性更好、单个 weight 的扰动对 output 的影响被分散。这与 [k-Bit inference scaling law](https://arxiv.org/abs/2302.02689)（Dettmers & Zettlemoyer）的结论一致：大模型在 low-bit 下表现更好。但本文进一步把衰减率 γ_N 量化出来，γ_N=0.2186，比 α=0.3022 小，意味着 quantization error 比 BF16 loss 本身衰减得慢——**scaling 时 quantization 的相对损失会越来越大**。

### 4.2 D 增大 → δ 增大（Figure 4b）

10B → 100B tokens，δ_W4A4 平均增加 22%。

**直觉**：这个现象 PTQ scaling law 也观察到过（[Low-bit quantization favors undertrained LLMs](https://arxiv.org/abs/2411.17691)），但 PTQ 增长更快。QAT 的 δ 增长更慢是因为 QAT 在训练时就适应了 quantization。但仍然存在增长，原因有二：
1. **Underfitting hypothesis**：更多数据让 BF16 模型继续学到更精细的 pattern，QAT 模型受 4-bit 容量限制无法学得一样精细，gap 拉大；
2. **Weight specialization hypothesis**：训练越久，weight distribution 越偏离 initialization，越复杂，4-bit 表示能力不够。

Appendix I 的 ablation（Table 4）：去掉 D 后，W4A4 相对预测误差从 4.7% 升到 8.6%，W4A16 从 5.2% 升到 13.8%——**W4A16 对 D 更敏感**，说明 weight error 是 D 项的主要驱动力。

### 4.3 G 增大 → δ 增大（Figure 4c）

最粗（per-token/channel）与最细（G=32）的 δ 差 0.037，约为最粗粒度本身 error 的一半。

**直觉**：quantization 本质是把连续值映射到有限 codebook。G 越大，一组内的 value range 越宽，outliers 越可能存在，scale 越被 outliers 主导，normal values 的量化噪声越大。G=1 是 per-element quantization，等价于不量化（δ=0），所以 log_2(G) 项满足边界条件。

---

## 5. Weight vs. Activation 误差分解

### 5.1 分解方法（Figure 6）

直接在 W4A4 模型里关闭 quantization 无法分离贡献（QAT 训练时 weight 已 adapt quantization）。**解决方案**：单独训练 W4A16 和 W16A4 模型，用：

$$\delta_{W4A4} \approx k \cdot (\delta_{W4A16} + \delta_{W16A4})$$

实测 k=0.906，强相关。**为什么 k<1？** 因为 W4A4 训练时 weight 与 activation 同时 quantize，模型可能学到 partially compensate 两者误差，所以联合 error 小于独立和。

### 5.2 三组 Scaling 参数对比（参见 Section 2.4 表格）

**核心 insight**：
- weight error：对 N 敏感（γ_N=0.36）、对 D 极敏感（γ_D=0.16）、对 G 弱敏感（γ_G=0.35）
- activation error：对 N 弱敏感（γ_N=0.18）、对 D 弱敏感（γ_D=0.03）、对 G 极敏感（γ_G=0.98）

**这意味着两个误差的相对大小会随配置而变化**。

### 5.3 R 比值的动态变化（Figure 8a）

$$R = \frac{\delta_{W16A4}}{\delta_{W4A16}}$$

- 所有 setting 下 R > 1：activation error 普遍大于 weight error
- R 随 D/N 增大而减小：D/N=100 时 R=1.67（G=32），D/N=1000 时 R=1.20。**因为 weight error 增长更快**
- R 随 G 增大而增大：D/N=1000 时 G=32 给 R=1.20，G=256 给 R=1.62。**因为 activation error 对 G 更敏感**

**实践含义**：
- Compute-optimal 训练（D/N≈20）下，activation error 主导
- Over-training（D/N=1000，参见 [Gadre et al.](https://arxiv.org/abs/2403.08540)）下，weight error 接近 activation error
- 粗粒度 quantization 下，activation error 始终主导

---

## 6. FC2 Bottleneck 深度分析

### 6.1 Kurtosis 度量 outliers（Figure 9a）

Kurtosis（[DeCarlo 1997](https://doi.org/10.1037/1082-989X.2.3.292)）衡量分布尾部厚度。BF16 模型各层 input activation 的 kurtosis：

| Layer | BF16 | W4A4 QAT |
|-------|------|----------|
| QKV Proj input | 高 | 显著降低 |
| O Proj input | 高 | 显著降低 |
| FC1 Proj input | 高 | 显著降低 |
| **FC2 Proj input** | **123** | **89** |

**关键观察**：QAT 对前三层有效降低 kurtosis（QAT 本身作为 regularizer 抑制 outliers，参见 [Nrusimha et al.](https://arxiv.org/abs/2404.03605)），但 FC2 input 的 kurtosis=89 仍远高于其他层。

### 6.2 为什么 FC2 input 有持续 outliers

Architecture 解析（Figure 15，LLaMA-3 block）：

```
input → QKV Proj → attention → O Proj → + residual → norm
     → FC1 Proj → SwiGLU(FC1 gate, FC1 up) → FC2 Proj → + residual
```

[SwiGLU](https://arxiv.org/abs/2002.05202) 定义：

$$\text{SwiGLU}(x) = (\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$$

其中 Swish(z) = z · σ(βz)。FC2 的 input 是 SwiGLU 的输出，即 `Swish(x W_gate) ⊙ (x W_up)`。

**Outlier 产生机制**：
1. Gating 让某些 channel 被显著放大（Swish 在大正值时近似 linear，gate 输出大正值）
2. Element-wise 乘法让放大效应相乘，分布尾部被指数级拉伸
3. ReLU 类激活的稀疏性 + 乘法 → 重尾分布（[Zhang et al. 2025](https://arxiv.org/abs/2503.08040)）

**为什么 QAT 抑制不了？** 因为 SwiGLU 的 gating 是模型表达力的核心，强制压缩 outlier 等价于损失 expressivity。QAT 的 regularizer 不足以既保 expression 又压 outlier。

### 6.3 Mixed-precision 方案

将 FC2 input 用 8-bit 量化（其余仍 W4A4）。8-bit 是 [FP8 训练](https://arxiv.org/abs/2310.18313) 与 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 验证的 near-lossless 精度。

**效果**（Figure 9b）：
- G=32 时 δ_W4A4 降 20.5%
- G=256 时降 42.9%（粗粒度受益更大，因为 outliers 是粗粒度的主要 pain point）

**γ_G 大幅下降**：δ_W16A4 的 γ_G 从 0.9812 → 0.4491，说明 8-bit FC2 输入让 activation error 对 granularity 几乎不再敏感。

### 6.4 新的 R 比值（Figure 8b）

加入 FC2 8-bit 后，R 从 (1.20-1.67) 降到 (0.85-1.10)，**weight error 与 activation error 量级持平**。

**重大实践意义**：
- 过去 QAT 算法（[SmoothQuant](https://arxiv.org/abs/2211.01038)、[QuaRot](https://arxiv.org/abs/2404.00456)、[SpinQuant](https://arxiv.org/abs/2405.16406)、[PrefixQuant](https://arxiv.org/abs/2410.05265)）都聚焦 activation outlier
- 本文证明：解决 FC2 bottleneck 后，weight error 同等重要
- D/N 增大时 weight error 甚至超过 activation error——**未来 QAT 算法应同等关注 weight quantization**

---

## 7. 与 EPM 的连接（Appendix H）

通过 Eq.(11) 把 δ_p 转回 EPM：

$$\text{eff}(\mathbf{C}) = \left(\frac{A}{A + k \cdot D^{\gamma_D} \cdot (\log_2 G)^{\gamma_G} \cdot N^{\alpha - \gamma_N}}\right)^{1/\alpha}$$

**关键 insight**：eff(C) 与 N 的关系取决于 (α - γ_N) 的符号。
- W4A4 时 α=0.3022 > γ_N=0.2186，所以 eff(C) 随 N 增大**减小**
- 直觉：大模型 BF16 loss 衰减快（α 主导），但 quantization error 衰减慢（γ_N 主导），所以 quantization 的"等效容量损失"在大模型上**相对更显著**

EPM contour（Figure 16）：
- W4A4 的 EPM 普遍 >0.5，说明 W4A4 比 W8A8（假设 cost 翻倍且 lossless）更 Pareto 优
- FC2 8-bit 把 EPM 提升 0.06-0.14，进一步拉开差距

这与 [BitNet b1.58](https://arxiv.org/abs/2402.17764)、[ParetoQ](https://arxiv.org/abs/2502.02631) 的"4-bit 是 Pareto optimal"结论一致，但 [Kumar](https://arxiv.org/abs/2411.04330) 之前得出 8-bit optimal。本文 Appendix I 解释：Kumar 用 per-tensor activation quantization，outliers 严重 degrade W4A4 性能；用 fine-grained 或 LAC 后 W4A4 才显示优势。

---

## 8. 与 PTQ Scaling Law 的区别

[Ouyang et al. 2024](https://arxiv.org/abs/2411.17691) 与 [Kumar](https://arxiv.org/abs/2411.04330) 都发现 PTQ 下 quantization error 随 D 增大而快速增长——训练越多数据，模型越 undertrained 时 PTQ 误差越小，反而是 over-trained 模型 PTQ 误差大。

**QAT 与 PTQ 的关键区别**：
- PTQ：training 后才 quantize，error 增长快
- QAT：训练中 quantize，weight adapt 量化，error 增长慢但**仍非零**

本文 δ_W4A4 的 γ_D=0.0745 比相同 setting PTQ 的 γ_D 小一个数量级，定量证实 QAT 在 D-axis 上的优势。

---

## 9. Intuition Building：综合图景

把所有发现拼成一幅图：

1. **W4A4 quantization error 由三件事决定**：模型够大能 absorb error（N）、数据够少避免 weight 复杂化（D）、granularity 够细避免 outlier 拖累（G）。
2. **三个 axis 的"性价比"对比**：从 contour 斜率看，N 的抵消能力最强（γ_N/γ_D≈2.93），其次 G（log 形式增长慢），D 增长最危险的。
3. **Weight vs activation 的跷跷板**：
   - 默认 setting：activation error > weight error（因为 FC2 outlier）
   - 移除 FC2 bottleneck：两者持平
   - Over-training（高 D/N）：weight error 反超
4. **算法设计启示**：QAT 算法需要 **layer-specific**、**precision-mixed**、**balance weight and activation** 的设计。
5. **Scaling law 实用价值**：
   - 预测 large scale QAT 性能（用小模型拟合后外推）
   - 决定是否值得用 fine-grained quantization（EPM 提升 vs 推理开销）
   - 决定何时切换到 mixed precision（看 R 比值与 D/N）

---

## 10. 局限性与未来方向

作者自己指出（Appendix A）：
1. 只测 dense model，MoE 未涉及。MoE 的 weight 多但 activation 同 dense，weight/activation error 比值会不同。
2. 只测 4-bit，ternary（[BitNet](https://arxiv.org/abs/2402.17764)、[QUEST](https://arxiv.org/abs/2502.05003)）的 scaling law 未建立。
3. 最大只到 595M/100B，外推到真正 LLM scale（70B+）需要更多验证。

可以延伸思考的方向：
- **FQT (Fully Quantized Training)**：本文只 quantize forward。backward 也 quantize 会引入新的 error source，scaling law 形式可能变（[FP8 training](https://arxiv.org/abs/2310.18313)、[mxfp4 training](https://arxiv.org/abs/2502.20586)、[FP4 training](https://arxiv.org/abs/2501.17116)）。
- **Rotation-based method**：[QuaRot](https://arxiv.org/abs/2404.00456)、[SpinQuant](https://arxiv.org/abs/2405.16406) 通过正交变换消除 outliers，理论上可改写 FC2 的 kurtosis，应该能改变 γ_G 项。
- **Outlier suppression 的 scaling law**：如果让 FC2 kurtosis 降到与其他层持平，δ_W16A4 的 γ_G 应该收敛到与 δ_W4A16 类似量级，可以验证。
- **Mixed-precision 的最优分配**：本文只测 FC2 用 8-bit，最优 mixed-precision 策略（哪些层 8-bit、哪些 4-bit）应该是一个 constrained optimization 问题。
- **数据类型 scaling law**：MXFP4、NVFP4 等 block-scaling 格式（[microscaling](https://arxiv.org/abs/2310.10537)）有 per-block scale，本质上等价于 fine-grained quantization，可以用 G 项统一建模。

---

## 11. 关键 References

- [Chinchilla (Hoffmann et al. 2022)](https://arxiv.org/abs/2203.15556) — compute-optimal scaling law
- [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) — 最早的 LLM scaling law
- [Kumar et al. 2024 — Scaling laws for precision](https://arxiv.org/abs/2411.04330) — 既有 QAT scaling law，本文主要对照
- [Frantar et al. 2025 — Compression scaling laws](https://arxiv.org/abs/2502.16440) — EPM 框架出处
- [Ouyang et al. 2024 — PTQ scaling law](https://arxiv.org/abs/2411.17691) — PTQ 视角对照
- [BitNet b1.58 (Ma et al. 2024)](https://arxiv.org/abs/2402.17764) — ternary QAT
- [BitNet a4.8 (Wang et al. 2024)](https://arxiv.org/abs/2411.04965) — 4-bit activation ternary
- [EfficientQAT (Chen et al. 2024)](https://arxiv.org/abs/2407.11062) — QAT 训练效率
- [QUEST (Panferov et al. 2025)](https://arxiv.org/abs/2502.05003) — 1-bit stable training
- [SmoothQuant (Xiao et al. 2022)](https://arxiv.org/abs/2211.01038) — activation outlier migration
- [QuaRot (Ashkboos et al. 2024)](https://arxiv.org/abs/2404.00456) — rotation-based outlier removal
- [SpinQuant (Liu et al. 2024)](https://arxiv.org/abs/2405.16406) — learned rotation
- [PrefixQuant / LAC (Chen et al. 2024)](https://arxiv.org/abs/2410.05265) — learnable clipping
- [OmniQuant / LWC (Shao et al. 2023)](https://arxiv.org/abs/2308.13137) — omnidirectional quantization
- [LSQ (Esser et al. 2019)](https://arxiv.org/abs/1902.08153) — learnable step size
- [LLaMA-3 (Grattafiori et al. 2024)](https://arxiv.org/abs/2407.21783) — 模型架构来源
- [OLMo2 (Team OLMo 2024)](https://arxiv.org/abs/2501.00656) — 训练数据与超参来源
- [SwiGLU (Shazeer 2020)](https://arxiv.org/abs/2002.05202) — FFN 激活函数
- [GQA (Ainslie et al. 2023)](https://arxiv.org/abs/2305.13245) — group query attention
- [Gadre et al. 2024 — over-training scaling](https://arxiv.org/abs/2403.08540) — D/N 比值的讨论
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — FP8 训练大规模验证
- [FP8-LM (Peng et al. 2023)](https://arxiv.org/abs/2310.18313) — FP8 LLM 训练
- [Nrusimha et al. 2024 — activation regularization](https://arxiv.org/abs/2404.03605) — QAT 抑制 outliers
- [Zhang et al. 2025 — block-level fallback](https://arxiv.org/abs/2503.08040) — INT8 训练 outlier 问题
- [Dettmers & Zettlemoyer — k-bit inference scaling](https://arxiv.org/abs/2302.02689) — 大模型在低 bit 表现更好的早期证据
- [ParetoQ (Liu et al. 2025)](https://arxiv.org/abs/2502.02631) — 极低 bit QAT scaling
- [DeCarlo 1997 — Kurtosis](https://doi.org/10.1037/1082-989X.2.3.292) — kurtosis 含义
- [Microscaling formats](https://arxiv.org/abs/2310.10537) — block-scaling 数据格式

---

## 12. 总结性 Takeaway

这篇 paper 的核心贡献是把 QAT 的 quantization error 从一个"模糊的工程问题"变成一个**可预测的数学对象**。最反直觉的发现是 **D 增大 → δ 增大**——意味着 quantization 与 over-training 的组合会变得越来越不友好，over-trained 模型（[Gadre](https://arxiv.org/abs/2403.08540) 推荐的 D/N=100-1000）在做 4-bit QAT 时需要特别关注 weight quantization。其次是 **FC2 input 的 persistent outlier**——SwiGLU 的乘法 gating 是数学结构上的"先天缺陷"，QAT 的 regularizer 无法根除，必须 mixed-precision 或 rotation 来处理。这两个发现合起来给未来 QAT 算法画了路线图：layer-specific precision allocation + weight/activation 联合优化 + 处理 SwiGLU 数学结构。
