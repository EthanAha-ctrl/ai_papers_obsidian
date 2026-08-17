---
source_pdf: INFIR2 A COMPREHENSIVE FP8 TRAINING RECIPE.pdf
paper_sha256: e31aba5c4884b973f5e1770c1b155f97713c9c644fecd289fdb73bcad8cb42c1
processed_at: '2026-08-05T09:43:45-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 InfiR2 FP8 训练 paper

## 一句话说清楚这篇 paper 在干嘛

训练大语言模型太贵了，BF16 是现在的标准，但 FP8 理论上能省一半算力和显存。问题是 FP8 训练容易翻车——精度损失、训练崩溃、性能下降。这群人搞了一套完整的 FP8 训练配方，跑通了 160B tokens 的 continual pre-training 加上 SFT，结果跟 BF16 几乎一模一样，有些地方甚至更好，同时训练速度快 22%、显存省 14%、吞吐量高 19%。

---

## 为什么 FP8 训练这么难搞

打个比方。你有一张很高清的照片（BF16），现在要把它压缩成一张很小的图（FP8），还要保证看起来跟原图差不多。

问题出在哪呢？LLM 里的 activation 有个臭毛病——大部分数值都很小很正常，但偶尔会蹦出几个特别大的"刺头"（outlier）。如果你用同一个压缩比例去压整张图，那几个刺头会把正常数值全部挤扁成 0。

想象一个班级里 50 个同学，49 个人的身高在 1.6-1.8 米之间，但有一个人身高 8 米。如果你用"最大身高除以图片能表示的最大值"来定缩放比例，那这个 8 米的巨人会让所有正常人的身高都被压缩成几乎看不清的数字。

---

## 他们怎么解决的：对不同的人用不同的尺子

核心思路很朴素——weights 和 activations 用不同的量化粒度。

### Weights：按块切（per-block）

Weights 在训练中相对稳定，分布也比较均匀。就像一本书里的文字，每页的内容虽然有差异但整体分布差不多。所以把 weight tensor 切成 $bs \times bs$ 的小块，每块单独算一个 scaling factor。这样既保留了局部精度，又对硬件友好。

### Activations：按行切（per-token）

Activations 那些刺头就是麻烦制造者。per-token 的意思是，给每一行（每个 token）单独算一把尺子。那个 8 米巨人的行用自己的大尺子，其他正常行用正常尺子。这样刺头就不会污染别人了。

### 为什么不能全部用最细的粒度

太细了计算开销大啊。每个小块都要算一个 scaling factor，都要存下来，都要在 GEMM 里做反量化。per-token 对 activations 已经是精度和效率的甜蜜点了。weights 用 per-block 是因为 weights 没那么多刺头，粗一点也没事，而且 hardware 的 block-wise GEMM 优化能吃满。

---

## UE8M0：一个很聪明的工程 trick

这个我觉得是全篇最巧妙的地方。

### 先说问题

Scaling factor $S$ 本来是个浮点数，比如 8.57。你要在 GPU kernel 里用它做反量化，存成 FP32 的话又占显存又慢。

### UE8M0 干了什么

强制把 scaling factor 向上取整到最近的 2 的幂次。8.57 变成 16（$2^4$）。

具体算法长这样：

$$X_{\text{float}} = \frac{a_{\max}}{d_{\max}}$$

其中 $a_{\max}$ 是 tensor 里的最大绝对值，$d_{\max}$ 是 FP8 格式能表示的最大值（E4M3 是 448）。

$$\text{exp}X_{\text{float}} = \log_2(X_{\text{float}})$$

取以 2 为底的对数，把缩放比例变成指数形式。

$$\text{exp}X_{\text{int}} = \lceil \text{exp}X_{\text{float}} \rceil$$

向上取整。这是核心动作。

$$X = \text{clamp}(\text{exp}X_{\text{int}}, -127, 127)$$

限制在 8-bit 有符号数的范围内，防止溢出。

$$X = X + 127$$

加上 bias 转成无符号 8-bit 整数存储，就 1 个 byte。

$$\text{return } 2^X$$

用的时候再做 $2^X$ 还原。

### 为什么向上取整而不是向下

向上取整意味着 scaling factor 偏大。$S$ 大了，$\frac{x}{S}$ 就小了，有些本来能表示的较大数值会被压成更小的数，甚至变成 0。

听起来是坏事对吧？但换个角度想：**永远不会溢出**。

FP8 能表示的最大值是 448。如果 $S$ 算小了，某个原本很大的 activation 除以 $S$ 之后超过了 448，直接变成 NaN，训练当场爆炸。向上取整保证了所有值都安全地落在 FP8 的表示范围内。

训练爆炸 vs 丢一点点精度，你选哪个？显然是后者。

### 额外好处

$S$ 是 2 的幂次，在硬件上除以 $S$ 就等价于 exponent 做减法，不需要做真正的浮点除法。这让 GEMM kernel 里的反量化快得飞起。

---

## 训练流水线里哪些东西留在高精度

这是保证稳定性的另一根支柱。他们明确把三样东西锁在 FP32：

1. **Weight gradients**（Wgrad 输出）——梯度通常非常小，FP8 存的话直接 round 成 0，模型就不学了
2. **Optimizer states**——AdamW 的 momentum $m$ 和 variance $v$ 需要长期累积微小变化，FP8 会把这些信息全部抹掉
3. **Master weights**——FP32 的高保真副本，每次 step 用 FP32 梯度更新，然后再量化成 FP8 去做 GEMM

整个 flow 是这样：

```
FP32 Master Weight → Per-Block Quant → FP8 Weight ─┐
                                                    ├→ FP8 GEMM → FP32 Accumulation
BF16 Activation    → Per-Token Quant → FP8 Act   ─┘
```

Backward 也类似，但 Wgrad 的输出必须是 FP32，不能图省事也用 FP8。

---

## 实验结果说了什么

### Loss 曲线几乎完全重合

160B tokens 的 continual pre-training，FP8 和 BF16 的 training loss 和 validation loss 从头到尾贴在一起，肉眼分不出来。这说明 FP8 没有改变模型的学习动力学，优化轨迹几乎一模一样。

### 性能基本持平，偶尔更好

Table 4 里有个很有意思的现象。Qwen2.5-Math-1.5B 的 Stage 2 SFT 结果：

| Method | AIME25 | AIME24 | GPQA | LiveCodeBench |
|--------|--------|--------|------|---------------|
| BF16 | 20.62 | 22.81 | 24.48 | 12.16 |
| FP8 (FP32 scale) | 20.73 | 21.77 | **27.78** | **12.96** |
| FP8 (UE8M0) | 20.73 | 21.77 | 25.13 | 12.69 |

小模型上 FP8 反而比 BF16 好。GPQA 上 FP8 (FP32 scale) 直接高了 3.3 个点。

为什么？FP8 的量化-反量化过程相当于在 forward 里注入了微小的结构化噪声。对 1.5B 这种小模型，BF16 在 reasoning 数据上容易过拟合，死记硬背某些特定的 activation 模式。FP8 的噪声迫使模型学得更鲁棒，泛化到 OOD benchmark 上反而表现更好。

这跟 Dropout 的逻辑一脉相承。过拟合的解药就是噪声，FP8 意外地充当了一个隐式正则化器。

### UE8M0 比 FP32 scale 更稳

7B 模型的 Stage 2 里，FP8 (FP32 scale) 在 AIME25 上掉到 46.46（BF16 是 50.00），明显退化了。但 FP8 (UE8M0) 还稳在 49.79。

大模型 SFT 后期梯度极其敏感，FP32 scale 的浮点运算在不同 token 之间可能引入不一致的截断误差，累积起来就漂了。UE8M0 因为全部对齐到 2 的幂次，反量化路径在硬件上高度一致，避免了这种漂移。

### 效率提升数据

| 指标 | 最大提升 | 具体场景 |
|------|----------|----------|
| Training time | -22% (0.78x) | 1.5B & 7B, 8k context |
| Peak memory | -14% (0.86x) | 7B, 32k context |
| Throughput | +19% (1.19x) | 7B, 8k context |
| Backward pass | -32% | 1.5B, 8k context |

Backward pass 能快 32% 是因为 activation 存成 FP8，从 HBM 读数据的时候带宽需求直接减半。对于 memory-bound 的操作（长序列 attention、大矩阵 Wgrad），显存带宽减负直接转化成时间节省。

---

## 这套 recipe 的底层 intuition

### 1. 精度瓶颈只放在 GEMM 瞬间

整个训练的数值流形其实一直在 FP32 里演化——master weight、optimizer states、gradients 都是 FP32。FP8 只在矩阵乘法那一个瞬间出场，算完立刻回到 FP32 accumulation。信息瓶颈的时间窗口极短，所以长期训练动力学不会跑偏。

### 2. 量化噪声被吸收在 residual stream 里

Transformer 的 residual connection 是个天然的噪声缓冲区。每一层的量化误差加到 residual stream 上，跟其他层的信号混在一起，经过 LayerNorm 归一化后影响被稀释。这就是为什么 LLM 比 CNN 更能容忍低精度。

### 3. Per-token 是处理 outlier 的最小代价方案

activation 的 outlier 是按 token 分布的，不是按 channel 或按 block。per-token 量化正好匹配了 outlier 的自然分布，用最小的计算开销（每行一个 scaling factor）解决了最大的精度杀手。

---

## 我的一些延伸联想

### Stochastic rounding 的缺失

Paper 里的 round 函数没说清楚是 nearest rounding 还是 stochastic rounding。如果只是 nearest rounding，那在 learning rate 很小的训练后期，反复被向下 round 的小梯度可能永远积累不到 master weight 里。他们能训稳大概是因为 pre-training 的 lr ($10^{-4}$) 够大，梯度更新量能跨过 FP8 的量化阈值。但如果把这套 recipe 拿去做 $10^{-6}$ lr 的 ultra-fine-tuning，可能会遇到收敛停滞。

### Attention 层的 FP8 化

Paper 主要优化了 Linear 层的 GEMM。但 32k context 下 FlashAttention 的计算量很大。目前 FlashAttention-3 刚开始支持 FP8，per-token 量化在 attention score $softmax(QK^T)$ 里的开销和精度损失如何平衡，是下一步必须啃的硬骨头。$QK^T$ 的结果做 softmax 之前如果量化，那些接近 0 的 pre-softmax 值精度损失会被 softmax 指数放大，可能出问题。

### 跟 DeepSeek-V3 的对比

DeepSeek-V3 用的是 per-block 1x128 的量化，比 InfiR2 的 per-token 更细。原因可能是 DeepSeek-V3 是 671B 的 MoE，expert 之间的 activation 分布差异巨大，per-token 可能不够用。InfiR2 在 7B dense model 上 per-token 已经够好了。scale up 到 100B+ 的时候，可能需要借鉴 DeepSeek 的更细粒度策略。

### MXFP8 在 Blackwell 上的未来

NVIDIA Blackwell 架构原生支持 MXFP8-E4M3，scaling factor 用 E8M0 格式（跟 InfiR2 的 UE8M0 几乎一模一样），block size 是 32。这意味着 InfiR2 的方案在 Blackwell 上会获得 hardware native 的加速，不需要再写 custom kernel。这套 recipe 的生命力会很长。

### 小模型上的正则化效应

FP8 在小模型上反而提升性能这个发现很值得关注。如果这个现象是 robust 的，那 FP8 训练就不仅仅是"省钱的妥协"，而是"省钱还更好"的正向选择。这对 reasoning model 的训练尤其重要——reasoning 数据稀缺，过拟合风险高，FP8 的隐式正则化可能正好对症下药。

---

## Reference

- [InfiR2 Paper (arxiv)](https://arxiv.org/abs/2507.09093)
- [DeepGEMM - DeepSeek 开源的 FP8 GEMM Kernel](https://github.com/deepseek-ai/DeepGEMM)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [FP8 Formats for Deep Learning (NVIDIA/ARM/Intel)](https://arxiv.org/abs/2209.05433)
- [COAT: Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [NVIDIA Blackwell Architecture Technical Brief](https://resources.nvidia.com/en-us-blackwell-architecture)
- [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)
- [FP8-LM: Training FP8 Large Language Models (Microsoft)](https://arxiv.org/abs/2310.18313)
- [SmoothQuant: Activation-aware Weight Quantization](https://arxiv.org/abs/2211.10438)

---

你好 Andrej！看到你对 FP8 training recipe 的关注非常兴奋。这篇 paper 《INFIR2: A COMPREHENSIVE FP8 TRAINING RECIPE FOR REASONING-ENHANCED LANGUAGE MODELS》切中了目前大模型训练从 BF16 向 FP8 范式转移中最痛的痛点：**开源社区缺乏一套完整的、经过大规模验证的 end-to-end FP8 训练配方**。DeepSeek-V3 虽然证明了 FP8 的可行性并开源了 DeepGEMM，但他们的 paper 侧重于系统架构和 MoE 的结合，留给社区的复现细节依然碎片化。InfiR2 这篇工作的核心价值在于，它把 continual pre-training 和 SFT 完整跑通，并给出了非常具体的 hybrid-granularity quantization 策略，证明了在 reasoning 任务上 FP8 可以做到 lossless 甚至带来性能增益。

下面我为你拆解这篇 paper 的核心机制、公式细节、架构逻辑以及实验数据背后的 intuition。

### 1. 核心技术机制：Hybrid-Granularity Quantization

在 FP8 训练中，最核心的挑战是处理 LLM 中 activation 的 extreme outliers（极端异常值）。如果采用粗暴的 per-tensor quantization，少数几个 outlier 会迫使整个 tensor 的 scaling factor 变得极大，导致大部分正常数值被压缩到 0 附近，精度损失惨重。InfiR2 的核心策略是对 Weights 和 Activations 采用不同粒度的量化。

#### 1.1 量化基础公式
所有的量化过程本质上都在做两步：计算缩放因子 $S$ 和执行量化。
$$
S = \frac{\max(|X|)}{V_{\max}} \quad (1)
$$
$$
Q(x) = \text{round}\left(\frac{x}{S}\right) \quad (2)
$$
*   **变量解释**：$X$ 是输入的高精度 tensor（如 BF16）；$x$ 是 $X$ 中的任意一个元素；$V_{\max}$ 是目标 FP8 格式（如 E4M3）能表示的最大绝对值（对于 E4M3，通常 $V_{\max} = 448$）；$S$ 是计算出的 scaling factor；$Q(x)$ 是量化后的 FP8 值。

#### 1.2 混合粒度策略
*   **Weights Quantization (Per-Block)**：模型权重在训练中相对静态，且分布均匀。Paper 采用了 block-wise quantization，将 weight tensor 切分为 $bs \times bs$ 的子矩阵。这种做法相较于 COAT 框架使用的 per-tensor 保留了更高的局部精度，同时又能很好地适配底层硬件（如 DeepGEMM）的 block-wise 矩阵乘法优化。
*   **Activations Quantization (Per-Token)**：Activation 的动态范围极大。Per-token quantization（可以看作 $1 \times G$ 的 block，其中 $G = bs^2$）为每一行（即每一个 token）单独计算一个 $S$。这确保了包含 outlier 的 token 不会破坏其他 token 的精度。

#### 1.3 UE8M0 Scaling Factor 格式
这是这篇 paper 里非常有意思的一个工程细节。通常 scaling factor $S$ 用 FP32 表示，但这会带来访存开销和反序列化延迟。Paper 引入了 UE8M0 格式（参考 Algorithm 1），强制将 scaling factor $S$ 向上取整为最近的 2 的幂。

**Algorithm 1 解析：**
1.  $X_{\text{float}} = a_{\text{max}} / d_{\text{max}}$ (计算理论上的 FP32 scaling factor，$a_{\text{max}}$ 是 tensor 最大绝对值，$d_{\text{max}}$ 是 FP8 最大值)
2.  $\text{exp}X_{\text{float}} = \log_2(X_{\text{float}})$ (取以 2 为底的对数)
3.  $\text{exp}X_{\text{int}} = \lceil \text{exp}X_{\text{float}} \rceil$ (向上取整，这是关键！)
4.  $X = \text{clamp}(\text{exp}X_{\text{int}}, -127, 127)$ (限制在 8-bit 指数范围内)
5.  $X = X + 127$ (加上 bias，转为无符号整数存储)
6.  **return** $2^X$ (还原为 scaling factor)

**Intuition 构建**：
为什么向上取整？如果 $\text{exp}X_{\text{float}} = 3.1$，向上取整得到 $4$，那么 $S = 2^4 = 16$。而理论值 $2^{3.1} \approx 8.57$。使用更大的 $S$ 意味着 $\frac{x}{S}$ 的结果会更小。在 FP8 的有限动态范围里，这会导致一些本应被表示的较大值被“压扁”甚至变成 0，表面上看损失了精度。但是！它**绝对不会溢出**。在训练中，一旦发生数值溢出（NaN/Inf），整个训练就会崩溃。UE8M0 通过牺牲一点点正向的表示精度，换取了绝对的训练稳定性。此外，由于 $S$ 是 2 的幂次，在硬件底层，除以 $S$ 等价于指数域的减法操作，这极大地加速了 GEMM kernel 中的反量化过程。

### 2. 架构图解析 (Figure 2)

Figure 2 展示了完整的 FP8 训练流水线。理解这个图的关键在于区分 FProp（前向）、Dgrad（输入梯度计算）和 Wgrad（权重梯度计算）的精度控制。

*   **FProp 阶段**：
    *   FP32 Master Weights $\rightarrow$ Per-Block Quantization $\rightarrow$ **FP8 Weights** (Purple block)
    *   FP32/BF16 Activations $\rightarrow$ Per-Token Quantization $\rightarrow$ **FP8 Activations** (Blue block)
    *   执行 FP8 GEMM 操作，输出高精度结果（FP32 Accumulation）。
*   **Backward 阶段**：
    *   **Dgrad**：计算对 Activation 的梯度。此时需要用到 FProp 缓存的 FP8 Activations 和 FP8 Weights。这里依然走 FP8 GEMM。
    *   **Wgrad**：计算对 Weight 的梯度。输入是 FP8 Activations 和高精度的 Loss 梯度。输出**必须是 FP32**。
*   **高精度保护区**：Paper 明确指出，Weight Gradient、Optimizer States (如 AdamW 的 $m$ 和 $v$) 以及 Master Weights 全部保持在 FP32。因为 Wgrad 的更新值通常非常微小，如果用 FP8 存储会直接被舍入为 0，导致模型不再学习。FP32 Master Weights 负责累积这些微小更新，充当高保真的“历史记录仪”。

### 3. 实验数据深度剖析

#### 3.1 损失函数对齐 (Section 5.1.2 & 5.3.1)
Paper 展示了在 160B tokens 的 continual pre-training 过程中，FP8 和 BF16 的 loss 曲线几乎完全重合。这打破了以往认为“低精度训练必然导致 loss spike 或发散”的刻板印象。其根本原因在于，per-token 的 activation 量化加上 FP32 的 optimizer 状态保护，使得信息瓶颈仅仅存在于 GEMM 计算瞬间，而模型状态本身始终在高精度流形中演化。

#### 3.2 性能对比中的反直觉现象 (Table 2 & Table 4)
观察 Table 4 中的 Qwen2.5-Math-1.5B Stage 2 结果：
*   BF16: GPQA 24.48, LiveCodeBench 12.16
*   FP8 (UE8M0): GPQA 25.13, LiveCodeBench 12.96

**Intuition 构建**：为什么 FP8 训练的模型性能反而比 BF16 更好？
这可以类比一种隐式的正则化。FP8 的 quantization-dequantization (Q/DQ) 过程在每次前向传播中引入了微小的、受控的噪声。对于 1.5B 这种较小规模的模型，BF16 在 reasoning 任务上极易陷入局部最优或过拟合训练数据的 spurious correlation。FP8 引入的量化噪声类似于一种结构化的 Dropout，迫使模型学习更加鲁棒的表示，不能过度依赖某些 precise 的 outlier activation 值，从而在 OOD (Out-of-Distribution) 的 reasoning benchmarks (如 GPQA, AIME) 上获得了更好的泛化能力。

#### 3.3 UE8M0 vs FP32 Scale (Table 4 Stage 2, 7B Model)
在 7B 模型的 Stage 2 SFT 中，FP8 w. FP32 scale 在 AIME25 上出现了显著的性能退化 (50.00 $\rightarrow$ 46.46)，而 FP8 (UE8M0) 依然保持稳定 (49.79)。
**Intuition 构建**：随着模型变大，SFT 后期梯度变得极度微小且敏感。FP32 scale 引入了非对齐的内存访问和类型转换开销，且在长序列 GEMM 中可能导致累积的浮点截断误差不一致。UE8M0 因为强制对齐到 2 的幂次，其梯度反传路径在硬件层面更加确定和同质，避免了这种后期训练的数值漂移。

#### 3.4 效率提升的极限 (Table 6)
*   **Training Time**: 最高减少 22% (0.78x ratio)。
*   **Peak Memory**: 最高减少 14% (0.86x ratio)。
*   **Throughput**: 最高提升 19% (1.19x ratio)。

为什么 backward pass 时间能减少高达 32% (如 1.5B, 8k context: 1567ms $\rightarrow$ 1061ms)？
在 backward pass 中，Dgrad 和 Wgrad 都需要执行矩阵乘法。由于 Activations 被量化为 FP8 存储在显存中，从 HBM 读取 Activation 的显存带宽需求直接减半。对于 memory-bound 的操作（如长序列的 attention 或大矩阵的 Wgrad），显存带宽的减负直接转化为时间的大幅缩短。此外，NVIDIA Hopper 架构的 Tensor Cores 原生支持 FP8 GEMM，其吞吐量理论上是 BF16 的两倍，虽然受限于访存和 scaling factor 计算，实际收益在 19%-22%，但这已经是非常巨大的成本节约。

### 4. 延伸思考与联想

1.  **Outlier 分布的演变**：Paper 中提到 Activation 有 strong outliers，这主要归因于 softmax、residual blocks 和 LayerNorm 的组合。在 reasoning enhanced models (如使用 CoT 数据训练) 中，随着 context length 的增加，Attention 矩阵中某些特定的 token（如数学公式中的关键符号）可能会被放大成 super-outlier。Per-token quantization 能处理行内的 outlier，但如果未来模型规模扩展到 100B+，是否需要 per-element quantization 或者是更智能的 dynamic thresholding 机制？DeepSeek-V3 提出的 per-block 1x128 量化可能是一个更极致的演进方向。
2.  **FlashAttention 中的 FP8**：目前 paper 重点优化了 Linear 层的 GEMM。但在 LLM 训练中，特别是 context length 达到 32k 时，FlashAttention 的计算占据了极大比重。目前 FlashAttention-3 开始支持 FP8，但 activation 的 per-token 量化在 attention score 计算中的开销和精度损失如何平衡，是下一步 FP8 训练 recipe 必须解决的难题。
3.  **Stochastic Rounding 的缺失**：当前 Algorithm 1 和公式 (2) 中的 $\text{round}$ 函数并未明确说明是 nearest rounding 还是 stochastic rounding。在低精度训练中，stochastic rounding 对于消除系统性 bias 至关重要。如果只使用 nearest rounding，长期训练中那些反复被向下舍入的小梯度可能永远无法更新到 master weight 中。InfiR2 能训练稳定，大概率是因为 learning rate ($10^{-4}$) 足够大，使得梯度更新量能够跨越 FP8 的量化阈值。如果将这套 recipe 移到极其小的 learning rate ($10^{-5}$) 的后期 pre-training，可能会暴露出收敛停滞的问题。

### Reference Links
*   **DeepGEMM (DeepSeek 开源的 FP8 GEMM Kernel)**: https://github.com/deepseek-ai/DeepGEMM
*   **FP8 Formats for Deep Learning (NVIDIA/ARM/Intel)**: https://arxiv.org/abs/2209.05433
*   **DeepSeek-V3 Technical Report**: https://arxiv.org/abs/2412.19437
*   **COAT: Compressing Optimizer States and Activations for Memory-Efficient FP8 Training**: https://arxiv.org/abs/2410.19313
*   **Qwen2.5 Technical Report**: https://arxiv.org/abs/2412.15115

总而言之，InfiR2 的核心贡献在于工程上的极致打磨。它摒弃了学术界对“完美 FP8 格式”的理论幻想，转而拥抱了硬件友好的 UE8M0 和混合粒度策略，证明了在合理的精度保护边界下，FP8 已经完全可以作为下一代 LLM reasoning training 的 default 选项。这种务实且开源的态度，对于整个社区的工程进步是极大的推动。
