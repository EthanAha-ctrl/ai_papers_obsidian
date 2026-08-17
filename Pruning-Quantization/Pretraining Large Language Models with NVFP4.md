---
source_pdf: Pretraining Large Language Models with NVFP4.pdf
paper_sha256: 8f0a2eeab9cdd71c36af759922f7e41a43c00cc6fc3cd8525e23b2c8d0d82565
processed_at: '2026-08-06T06:02:18-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 NVFP4 Pretraining

## 1. 一句话讲清楚这篇paper在干嘛

NVIDIA搞了个新的4-bit浮点数格式叫NVFP4，然后配了一堆训练技巧，成功用这个格式训练了一个 **12B参数** 的模型，喂了 **10T tokens**，结果跟用FP8训练的baseline几乎一样好。这是第一次有人把FP4训练跑到这个规模还能work。

## 2. 为什么FP4训练这么难？一个生活化的比喻

想象你有一把只有 **8个刻度** 的尺子（FP4能表示的正数：0.5, 1, 1.5, 2, 3, 4, 6）。现在让你用这把尺子去量一堆数字，然后做精确计算。

问题来了：
- 数字 **5.7** 怎么办？只能round到 **6**（差0.3）
- 数字 **0.3** 怎么办？只能round到 **0**（直接丢了，因为最小非零是0.5）
- 数字 **100** 怎么办？saturate到 **6**（直接截断）

如果只是存几个数还好，但训练时要存 **几百亿个weight**，每个都被这样"糊弄"一下，累积起来bias会把训练带崩。

FP8能work是因为有256个刻度，FP4只有8个——精度差了32倍，所以问题严重得多。

## 3. NVFP4 vs MXFP4：格式设计上的关键差异

### 3.1 先说MXFP4的问题

MXFP4是OCP标准的microscaling格式，做法是：**32个元素一组**，共享一个scale factor。这个scale factor是 **power-of-two**（只能取 $2^n$），没有mantissa。

问题出在这：假设一个block里最大值是 **3.7**，理想情况下你想把它映射到FP4的max（6），所以scale应该是 $3.7/6 = 0.617$。但power-of-two只能取 $0.5$ 或 $1.0$。你只能选 **1.0**（往大round防止saturate），结果3.7除以1.0还是3.7，quantize到FP4的 **3**。

这一来，FP4能表示的 **4和6两个刻度彻底浪费**——因为这个block里没有值能scale到4或6。Dynamic range从3.58 binades退化到2.58 binades，白白损失了一个binade。

### 3.2 NVFP4的三个改进

**改进一：block size从32减到16**
更小的block意味着每个block里的值更"相似"，dynamic range更窄，更容易塞进FP4的表示范围。

**改进二：block scale用E4M3（FP8）而不是UE8M0（power-of-two）**
E4M3有3-bit mantissa，能把3.7精确映射到接近6的位置，不浪费4和6这两个刻度。

**改进三：两级scaling——FP32全局 + E4M3局部**

公式：
$$s_{enc} = \frac{6 \cdot 448}{amax_x}$$

变量解释：
- $amax_x$：整个tensor的绝对最大值
- 6：FP4（E2M1）能表示的最大值
- 448：FP8（E4M3）能表示的最大值
- $s_{enc}$：global encode scale，目的是把tensor里"最大的block amax"remap到E4M3的max

然后每个block内的local scale：
$$s_{dec,b} = \frac{amax_b}{6}$$
$$s_{dec,b,e4m3} = \text{E4M3}(s_{dec,b} \cdot s_{enc})$$

变量解释：
- $amax_b$：block b内的绝对最大值
- $s_{dec,b}$：理想的decode scale
- $s_{dec,b,e4m3}$：quantize到E4M3存储的实际decode scale

**Intuition**：FP32管全局范围（防止E4M3 scale溢出），E4M3管block局部范围（精确映射），FP4管具体值（压缩存储）。这是一个 **hierarchical精度分配** ——每个block里最大那个值实际上以接近FP8精度存储，其余15个值在FP4。

## 4. 训练方法学：四个技巧的"鸡尾酒疗法"

光有格式不够，paper里明确说：如果只用base method（无额外技巧），模型 **早期就diverge**。需要四个技巧配合。

### 4.1 Mixed precision：保留敏感层在BF16

**发现**：不是所有层都 equally sensitive。最后几个block的weight gradient quantization error特别大，容易把训练带崩。

**做法**：前2个block + 后8个block（约16%线性层）保持BF16。中间的用FP4。

**为什么是后8个block**：末端层往往处理更高层抽象，dynamic range更大，FP4装不下。

**为什么还要前2个block**：仅保留末尾block时训练仍不稳定，前面几个block作为"buffer"稳定输入分布。

**其他保留高精度的部分**：
- Embedding（lookup table用BF16）
- Output projection head（跟vocab size有关，太敏感）
- LayerNorm / RMSNorm
- Softmax
- Attention的QK和SV GEMM
- Optimizer states（master weights、momentum、variance都在FP32）
- Weight gradients（用于跨microbatch和DP replica accumulate）

### 4.2 Random Hadamard Transforms (RHT)：把outlier"摊平"

**outlier问题**：LLM里有些weight/activation值比其他大几十倍。FP4只有8个刻度，一个outlier就能"霸占"整个block的dynamic range，让其他小值全挤到0附近。

**Hadamard变换的intuition**：想象你有一堆柱子，有几根特别高。Hadamard变换就像把这些柱子"摊平"——高的变矮，矮的变高，整体趋向Gaussian分布。这样FP4就容易表示了。

数学上，Hadamard矩阵 $H$ 满足 $HH^T = I$（正交），所以：
$$C = (AH)(H^T B) = AB$$

两个operand的变换在dot-product里互相抵消，计算结果不变。

**Random版本**：$H = S_d H_d$，其中 $S_d$ 是对角元素随机取 $\{-1, 1\}$ 的对角矩阵。这会随机翻转某些行/列的符号，防止"结构化outlier"在Hadamard basis下幸存。

**关键发现**：
- 只对 **Wgrad inputs** 用RHT有效
- 对Fprop和Dgrad用RHT反而 **有害**——RHT本身引入quantization error，在Fprop/Dgrad上这个error超过outlier removal的收益
- 矩阵大小16×16是sweet spot：4×4太小无法Gaussian化，128×128略好但cost高
- 用 **单一固定seed**（整个训练一个random sign vector，全网络共享）就够，per-instance随机化无额外收益
- 小模型上randomization无差异，但12B/10T上明显重要—— **scale越大，技巧越关键**

### 4.3 2D block scaling：修复chain rule violation

这是paper里最精妙的发现。

**问题**：标准NVFP4 scaling沿 **dot-product dimension** 应用。但backward pass会transpose weight，导致dot-product dimension变化。同一个weight在forward和backward里有 **不同的quantized representation**。

具体说：
- Forward：$y_{\text{fprop}} = w_{\text{fprop}} \cdot x$，weight按input channel分组
- Backward：$\partial x = w_{\text{bprop}}^T \partial y$，weight按output channel分组（因为transpose了）
- 但 $w_{\text{fprop}} \neq w_{\text{bprop}}$——它们是 **不同的量化函数**

**这破坏了chain rule**：backward不再differentiate forward用的同一个函数。Gradient descent假设你优化的函数是你forward的函数的微分，如果这个假设不成立，convergence guarantee失效。

**解决方案**：weight用 **16×16 2D block scaling**（同时覆盖input channel和output channel）。这样无论forward还是backward，weight的quantized representation都一致。

公式上，每个16×16 block共享一个scale，而不是每1×16或16×1共享。

**为什么activation不用2D**：ablation显示activation对inconsistency更tolerant。而且activation的更细granularity（1×16）对accuracy更好。

**为什么RHT只用于Wgrad**：如果对Fprop/Dgrad用RHT，需要weight也RHT才能invert。但weight一旦RHT就引入forward/backward inconsistency，得不偿失。

### 4.4 Stochastic rounding：只对gradient用

**deterministic rounding的bias来源**：
1. Mantissa分布偏向某个方向
2. 小值underflow到0
3. 大值saturate到max representable

**Stochastic rounding**：以距离反比概率round到两个最近representable之一。

数学上，对于值 $x$，两个最近representable是 $\lfloor x \rfloor$ 和 $\lceil x \rceil$：
$$\text{round}(x) = \begin{cases} \lfloor x \rfloor & \text{w.p. } \lceil x \rceil - x \\ \lceil x \rceil & \text{w.p. } x - \lfloor x \rfloor \end{cases}$$

变量解释：
- $\lfloor x \rfloor$：下界representable
- $\lceil x \rceil$：上界representable
- 概率跟距离成反比——离谁近更可能round到谁

**期望上无偏**：$E[\text{round}(x)] = x$。

**关键发现**：
- 只对 **gradient** 用stochastic rounding有效
- 对activation或weight用stochastic rounding会 **diverge**——stochastic增加variance，对非梯度tensor反而放大quantization error
- 必须同时应用到Dgrad和Wgrad的输入梯度

**Intuition**：Forward的quantization error是deterministic mapping，可以用更多training step修正。但gradient本身就是stochastic estimation，deterministic rounding的bias跟SGD noise耦合，无法被average out。Stochastic rounding把bias转成variance，让SGD的inherent averaging机制处理。

## 5. 核心实验结果

### 5.1 12B / 10T tokens主结果

| Task | FP8 | NVFP4 | Gap |
|---|---|---|---|
| MMLU-Pro | 62.62% | 62.58% | -0.04% |
| MMLU | 77.36% | 76.57% | -0.79% |
| GSM8k CoT | 89.08% | 92.27% | +3.19% |
| MATH | 83.32% | 81.48% | -1.84% |
| HumanEval+ | 59.93% | 57.43% | -2.50% |
| MBPP+ | 59.11% | 55.91% | -3.20% |
| ARC Challenge | 91.81% | 91.81% | 0% |
| HellaSwag | 83.83% | 83.09% | -0.74% |
| MGSM | 81.87% | 85.53% | +3.66% |

Validation loss相对误差：stable阶段 < 1%，decay阶段 ~1.5%。

**关键观察**：
- 大部分task跟FP8几乎对齐
- Coding略差（MBPP+/HumanEval+），作者怀疑是evaluation noise
- Math和multilingual甚至更好（可能是noise或FP4的regularization effect）

### 5.2 NVFP4 vs MXFP4对比（8B / 1T）

| Format | 相对误差 | 需要的tokens |
|---|---|---|
| MXFP4 | ~2.5% | 1.36T（多36%）|
| NVFP4 | ~1.5% | 1T |

MXFP4需要 **多36%的tokens** 才能match NVFP4的final loss。这直接量化了格式设计的重要性。

### 5.3 Ablation：四个技巧都必要

Figure 4的ablation从完整方法开始，每次去掉一个组件：
- 去掉stochastic rounding → loss worsen
- 去掉RHT → loss worsen
- 去掉2D scaling → loss worsen
- 减少BF16层 → loss worsen

**每个都必要**，且重要性随scale增长。

### 5.4 Loss gap的缓解

**Forward quantization是主因**。在8.2T tokens后把forward切到BF16（仅forward），gap从1.5%降到0.5%。

**实用建议**：
- 在LR decay之前切到high precision（约18%训练时间）能完全recover
- 在末尾切（<1%训练时间）能显著改善但不完全

## 6. 更深的Intuition

### 6.1 为什么FP4比FP8难这么多？

FP8有256个刻度，FP4只有8个。相邻大值间距达1.5（从4到6），任何中间值都被round到这些"灯塔"。Weight gradient的statistical bias如果systematic，会在trillion-level token累积中放大成catastrophic drift。

### 6.2 为什么2D scaling如此重要？

Standard 1D scaling破坏chain rule的本质：gradient descent假设你优化的函数是你forward的函数的微分。如果weight在forward和backward里是"两个不同的quantized function"，optimizer实际上在解一个 **mismatched game**，convergence guarantee失效。2D scaling让weight成为一个 **统一的quantized object**，恢复chain rule的一致性。

### 6.3 为什么RHT只在Wgrad有用？

Fprop/Dgrad里，activation/gradient的outlier影响相对小，因为它们的scale可以online recompute。但Wgrad里，accumulation over microbatch使outlier累积放大。RHT把outlier摊薄让quantization error成零均值noise，配合stochastic rounding被averaging out。

### 6.4 为什么stochastic rounding只对gradient好？

Forward的quantization error是deterministic mapping，可以用更多training修正。但gradient是stochastic estimation本身，deterministic rounding的bias跟SGD noise耦合，无法被average out。Stochastic rounding把bias转成variance，让SGD的inherent averaging机制处理。

### 6.5 scale依赖性：小模型ablation结论不能外推

1.2B上Hadamard matrix大小、randomization无明显差异。12B上4×4明显比16×16差；不用random sign vector明显worse。Outlier的"结构化"特性在大模型里更明显，需要更强的mitigation。这意味着 **FP4训练研究必须在大scale验证**，小scale的ablation可能误导。

### 6.6 训练方法学各组件的相互依赖

这四个技巧不是独立的，而是相互配合：
- 2D scaling让weight一致 → 让RHT能只用于Wgrad
- RHT把outlier摊薄 → 让stochastic rounding的bias更可控
- Stochastic rounding消除gradient bias → 让2D scaling的一致性更有意义
- Mixed precision保留敏感层 → 给其他技巧"喘息空间"

去掉任何一个，整个stack崩塌。这是典型的 **co-design** ——格式、硬件、算法必须协同。

## 7. 我的延伸联想

### 7.1 与inference量化的区别

Inference量化（如AWQ、GPTQ、SmoothQuant）是 **post-training**，weight固定，只需要处理activation outlier。Training量化要同时处理weight、activation、gradient三者的dynamic distribution，且必须保证convergence。难度高一个量级。

### 7.2 为什么FP8比FP4早成熟？

FP8有256个刻度，dynamic range足够，基本只需要block scaling就能work（如DeepSeek-V3）。FP4的8个刻度逼出了RHT、2D scaling、stochastic rounding这一整套技巧。FP4的成功反过来说明FP8可能还有改进空间——比如FP8 training是否也能从2D scaling受益？

### 7.3 硬件协同设计的重要性

NVFP4的成功离不开Blackwell Tensor Core的native support：
- 16元素block + E4M3 scale的硬件加速
- Stochastic rounding的硬件指令
- FP4 GEMM的2×（GB200）到3×（GB300）throughput

这展示了 **算法-硬件co-design** 的力量。OCP标准（MXFP4）走通用路线，NVIDIA走专用路线，专用路线在这个case下赢了（少36% tokens）。

### 7.4 Mamba-2 hybrid架构的选择

为什么用hybrid Mamba-Transformer而不是纯Transformer？可能原因：
- Mamba-2的state space model对quantization更tolerant（线性递归 vs attention的非线性）
- 混合架构的memory footprint更小，FP4的memory saving收益更明显
- Mamba-2的GEMM pattern更适合FP4 block scaling

### 7.5 未来方向

Paper自己列的未来工作：
- 全FP4（去掉剩余15% BF16层）
- Attention的QK/SV GEMM量化
- Communication量化（TP/DP all-reduce）
- MoE架构
- >12B参数验证scaling law
- Post-training场景（SFT、RLHF）
- System-level throughput数据

### 7.6 潜在的问题

- **只测了一个模型家族**（Nemotron-H），其他架构（纯Transformer、MoE）是否work未知
- **只测了一个scale**（12B），70B+是否work未知
- **coding task略差**，是否系统性的？
- **stochastic rounding的throughput cost**未报告，硬件指令是否真的free？
- **RHT的memory overhead**（16×16矩阵 + 变换计算）

### 7.7 与其他FP4训练工作的对比

- **Quartet**（Castro et al. 2025）：native FP4 training，方法学类似但细节不同
- **FP4 All the Way**（Chmiel et al. 2025）：全量化训练，结论说backward切精度能recover loss（与本文相反）
- **Training LLMs with MXFP4**（Tseng et al. 2025）：MXFP4训练，本文的MXFP4 baseline
- **Optimizing LLM Training Using FP4**（Wang et al. 2025）：另一个FP4训练方法

这些工作的差异点：
- 哪些层保留高精度
- RHT用于哪些GEMM
- scaling的维度策略
- stochastic rounding的apply范围

本文的独特贡献： **规模最大**（12B/10T）+ **四技巧组合验证** + **NVFP4格式设计**。

## 8. 一句话总结

NVFP4 pretraining的可行性来自一个 **精密的stack**：硬件提供的E4M3 block scale + FP32两级scaling让FP4 sample被充分利用；2D weight scaling保持forward/backward一致性；RHT抑制Wgrad outlier；stochastic rounding消除gradient bias；最后保留~15% sensitive layers在BF16。四者缺一不可，且重要性随scale增长。

这工作本质上证明： **4-bit pretraining at frontier scale is no longer an algorithmic question, but an engineering one**。

## 9. 相关阅读链接

- NVFP4 official blog: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- MX Formats spec (OCP): https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- DeepSeek-V3 FP8 Technical Report: https://arxiv.org/abs/2412.19437
- QuaRot (Hadamard for inference): https://arxiv.org/abs/2404.00456
- Quartet (native FP4 training): https://arxiv.org/abs/2505.14669
- Training LLMs with MXFP4 (Tseng et al.): https://arxiv.org/abs/2502.20586
- FP4 All the Way (Chmiel et al.): https://arxiv.org/abs/2505.19115
- Blackwell architecture brief: https://resources.nvidia.com/en-us-blackwell-architecture
- Nemotron-H: https://arxiv.org/abs/2504.03624
- SmoothQuant (outlier handling classic): https://arxiv.org/abs/2211.10438
- QSGD (stochastic gradient quantization): https://arxiv.org/abs/2305.11360
- Transformer Engine (code): https://github.com/NVIDIA/TransformerEngine
- FlashAttention-3 (Hadamard in attention): https://arxiv.org/abs/2407.08608
- QSGD original (NeurIPS 2017): https://papers.nips.cc/paper/2017/hash/6c34f465793fa6e7e4c5e5e5e5e5e5e5-Abstract.html

## 10. 最终takeaway

如果你只记一件事： **FP4训练在frontier scale是可行的，但需要格式设计（NVFP4的两级scaling）+ 训练方法学（2D scaling + RHT + stochastic rounding + mixed precision）的协同co-design**。这不是单点突破，是系统工程的胜利。

如果你想深挖一个技术点： **2D block scaling修复chain rule violation** 是最精妙的insight，它揭示了quantization-aware training里一个被忽视的问题——forward和backward必须用同一个quantized function，否则gradient descent的mathematical foundation崩塌。

如果你想质疑一个点： **只在Nemotron-H架构上验证**，其他架构（纯Transformer、MoE、different model family）是否work仍未知。且coding task的gap是否系统性也需进一步验证。

如果你想做相关研究： **scale依赖性** 是最大的open question——小模型ablation结论不能外推，这意味着FP4训练研究必须在大scale验证，门槛很高。

---

# NVFP4 Pretraining：技术深度解析

## 1. 核心动机与背景

这篇paper来自NVIDIA团队，核心贡献在于：首次在 **12B参数模型** 上、 **10T tokens** 规模上成功完成 **4-bit floating point (FP4) pretraining**，并且loss/下游精度与FP8 baseline几乎对齐。这是一个工程意义重大的milestone——FP4相比FP8能提供2-3x arithmetic throughput、memory减半。

需要强调的关键insight在于：FP4的可行性强烈依赖于 **格式设计 + 训练方法学** 的协同。仅靠硬件native support不够，paper里展示了如果只用base method（无任何额外技术）模型会早期diverge。

## 2. NVFP4格式：相对MXFP4的三个关键改进

### 2.1 E2M1 element format
每个element是1 sign bit + 2 exponent bits + 1 mantissa bit，可表示的值为：±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6。注意这里的非均匀分布—— **binade间距** 在大值处更宽，这意味着outlier可以被"压缩"但小值容易塌缩到0。

### 2.2 Block size：32 → 16
MXFP4每个block 32个元素共享一个scale，NVFP4减到16。这直接narrow了block内的dynamic range，让outlier更局部化。

### 2.3 Scale factor：UE8M0 → E4M3 + FP32两级scaling
这是最关键的设计。MXFP4的scale是power-of-two（无mantissa），导致 **binade浪费** ：

考虑一个block，amax = 3 + δ（δ小）。理想scale = amax/6 = 0.5 + δ/6。但UE8M0只能round up到1.0，结果amax在scaled后变成3+δ，quantize到FP4的"3"—— **±4和±6两个sample完全被浪费**。Dynamic range从 $\log_2(6/0.5) = 3.58$ binades退化到 $\log_2(3/0.5) = 2.58$ binades。

NVFP4用E4M3（FP8）做block scale，有3-bit mantissa，能把amax精确映射到6附近，充分利用所有FP4 sample。

### 2.4 两级scaling的公式细节

**Global tensor-level scale (FP32)**：

$$s_{enc} = \frac{6 \cdot 448}{amax_x}$$

- $amax_x = \max_i(|x_i|)$：整个tensor的绝对最大值
- 6：E2M1的最大表示值
- 448：E4M3的最大表示值
- 目的：把tensor里最大的block amax（= $amax_x/6$）remap到E4M3的max representable（448），这样block scale不会溢出E4M3

**Local block-level scale (E4M3)**：

$$s_{dec,b} = \frac{amax_b}{6}$$
$$s_{dec,b,e4m3} = \text{E4M3}(s_{dec,b} \cdot s_{enc})$$

- $amax_b = \max_{i \in b}(|x_i|)$：block b内的绝对最大值
- 第二步：把local scale乘上global encode scale后quantize到E4M3
- Round-to-nearest-even用于这一步

**最终quantization**：

$$\hat{x}_i = q(x_i \cdot s_{enc,b})$$

其中 $s_{enc,b} = 1/(\text{fp32}(s_{dec,b,e4m3}) \cdot s_{dec})$，目标是让 $s_{enc,b} \cdot s_{dec} \cdot s_{dec,b,e4m3} \approx 1$，即能精确recover原始值。

**Tensor Core上的decode**：

$$s_{dec,b,e4m3}^x \cdot s_{dec,b,e4m3}^y \cdot \sum_{k \in b}(x_k \cdot y_k)$$

partial dot-product over block，再乘两个block scale，最后再apply global decode scales $s_{dec}^x$ 和 $s_{dec}^y$。

**Intuition**：这本质上是一个 **hierarchical normalization**——FP32抓全局范围，FP8抓局部范围，FP4抓精细值。约6.25%的元素（每个block的amax）实际上以接近FP8精度存储，剩余93.75%在FP4。

## 3. 训练方法学的四个支柱

### 3.1 Mixed precision（保留sensitive layers）

- 前2个block + 后8个block（共16%线性层）保持BF16
- 为何要保留 **后8个block**：ablation显示末端层的weight gradient有更大的quantization error，FP4的range不够
- 为何要保留 **前2个block**：仅保留末尾block时training不稳定，前几层作为"buffer"
- Embedding、output head、layernorm、softmax、attention QK/SV GEMM都保持高精度
- Optimizer states、weight gradients（用于跨microbatch/DP accumulate）、master weights保持FP32

### 3.2 Random Hadamard Transforms (RHT)

核心思想：用正交变换把outlier "摊薄"到Gaussian-like分布，让FP4更容易represent。

**Hadamard matrix**：$H_d = (1/\sqrt{2}) H_2 \otimes H_{d/2}$，元素全为±1，正交：$HH^T = I$。

**Random版本**：$H = S_d H_d$，其中 $S_d$ 是对角元素从 $\{-1, 1\}$ 随机采样的矩阵——随机翻转某些行/列的符号，防止"结构化outlier"在Hadamard basis下幸存。

**应用到GEMM**：

$$C = (AH)(H^T B) = AB$$

由于正交性，两个operand的变换在dot-product中互相抵消。

**关键发现**：
- 只对 **Wgrad inputs** 应用RHT；对Fprop和Dgrad inputs反而 **有害**（quantization error超过outlier removal的收益）
- 矩阵大小16×16：4×4太差（无法Gaussian化），128×128略好但不值得cost，16是sweet spot
- **单一固定seed**（整个训练用一个random sign vector，全网络共享）就够；per-instance随机化无额外收益
- 在小模型（1.2B）上randomization无差异，但在12B/10T上明显重要—— **scale越大，越依赖这些技巧**

### 3.3 2D block scaling（解决chain rule violation）

这是个很精妙的发现。标准NVFP4 scaling沿 **dot-product dimension** 应用。但backward pass会transpose tensor，导致dot-product dimension变化——同一个weight tensor在forward和backward里有 **不同的quantized representation**。

数学上：

- Forward：$y_{\text{fprop}} = w_{\text{fprop}} \cdot x$，其中 $w_{\text{fprop}}$ 是按某种block分组quantize的w
- Backward：$\partial x = w_{\text{bprop}}^T \partial y$，但 $w_{\text{bprop}} \neq w_{\text{fprop}}$，因为scaling维度变了
- Chain rule被 **破坏**：backward不再differentiate forward用的同一个函数

**解决方案**：weights用 **16×16 2D block scaling**（16 input channels × 16 output channels），覆盖两个维度，forward和backward里quantized representation一致。代价是block granularity变粗，但weights能"adapt"到FP4值，对粗粒度更tolerant。

Activations和gradients仍用1×16（标准NVFP4），因为：
- 它们对inconsistency更不敏感（ablation验证）
- 更细的granularity对accuracy更好

注意：RHT也因此只用于Wgrad——如果对Fprop/Dgrad用RHT，需要weight也RHT才能invert，但weight一旦RHT就引入inconsistency，得不偿失。

### 3.4 Stochastic rounding（仅用于gradients）

**确定性round-to-nearest-even的bias来源**：
1. Mantissa分布偏向某个方向
2. 小值underflow到0
3. 大值saturate到max representable

Stochastic rounding：以距离反比概率round到两个最近representable之一，期望上无偏。

$$\text{round}(x) = \begin{cases} \lfloor x \rfloor & \text{w.p. } 1 - (x - \lfloor x \rfloor) \\ \lceil x \rceil & \text{w.p. } x - \lfloor x \rfloor \end{cases}$$

**关键发现**：
- 仅对 **gradients** 用stochastic rounding有效
- 对activations或weights用stochastic rounding会 **diverge**——stochastic增加variance，对非梯度tensor反而放大quantization error
- 必须同时应用到Dgrad和Wgrad的输入梯度

## 4. 核心实验：12B / 10T tokens

### 4.1 模型
- Hybrid Mamba-Transformer（Nemotron-H家族）
- 62 blocks：6 Self-Attention + 28 FFN + 28 Mamba-2
- d_model = 5120，FFN dim = 20480，40 query heads / 8 KV heads
- Mamba-2：8 groups，state dim 128，head dim 64，expansion 2
- 8192 seq len，batch 736，WSD schedule（80% constant + 20% decay），LR = 4.5e-4 → 4.5e-6

### 4.2 主结果

| Metric | FP8 | NVFP4 | Gap |
|---|---|---|---|
| MMLU-Pro | 62.62% | 62.58% | -0.04% |
| MMLU | 77.36% | 76.57% | -0.79% |
| GSM8k CoT | 89.08% | 92.27% | +3.19% |
| HumanEval+ | 59.93% | 57.43% | -2.50% |
| MGSM | 81.87% | 85.53% | +3.66% |

Validation loss相对误差：stable阶段 < 1%，decay阶段 ~1.5%。NVFP4在coding（MBPP+/HumanEval+）略差，作者怀疑是evaluation noise。

### 4.3 Loss gap的成因与缓解

**Forward quantization是loss gap的主因**。在8.2T tokens后切到BF16（仅forward），gap从1.5%降到0.5%。这跟Chmiel et al. 2025的结论相反（他们说backward切精度能recover loss）——可能与具体模型/setup有关。

**实用建议**：在LR decay之前切到high precision（约18%训练时间）能完全recover；在末尾切（<1%训练时间）能显著改善但不完全。

## 5. NVFP4 vs MXFP4（8B / 1T）

| Format | Block | Scale | Block scale format | 相对误差 |
|---|---|---|---|---|
| MXFP4 | 32 | UE8M0 | Power-of-two | ~2.5% |
| NVFP4 | 16 | E4M3 + FP32 | FP8 | ~1.5% |

MXFP4需要 **多36%的tokens**（1.36T vs 1T）才能match NVFP4的final loss——这意味着NVFP4在固定token预算下efficiency更高，或在固定loss目标下train time更短。

## 6. Ablation的关键insight

四个组件全部necessary，去掉任何一个loss都worsen（Figure 4）。但 **scale依赖性** 强烈：

- 1.2B上：Hadamard matrix大小、randomization无明显差异
- 12B上：4×4矩阵明显比16×16差；不用random sign vector明显worse

**Lesson**：小scale上的ablation结论 **不能直接外推** 到大模型/长horizon。Outlier的"结构化"特性在大模型里更明显，需要更强的mitigation。

## 7. Intuition Building

### 7.1 为什么FP4训练这么难？
FP4只有8个正数值（含0则是9个），相邻大值间距达1.5（从4到6）——任何中间值都被round到这些"灯塔"。Weight gradient的statistical bias如果systematic，会在trillion-level token累积中放大成catastrophic drift。

### 7.2 为什么2D scaling如此重要？
Standard 1D scaling破坏chain rule的本质：gradient descent假设你优化的函数是你forward的函数的微分。如果weight在forward和backward里是"两个不同的quantized function"，optimizer实际上在解一个 **mismatched game**，convergence guarantees失效。2D scaling让weight成为一个 **统一的quantized object**，恢复chain rule的一致性。

### 7.3 为什么RHT只在Wgrad有用？
Fprop/Dgrad里，activation/gradient的outlier影响相对小，因为它们的scale可以online recompute；但Wgrad里，accumulation over microbatch使outlier累积放大。RHT把outlier摊薄让quantization error成零均值noise，配合stochastic rounding被averaging out。

### 7.4 为什么stochastic rounding只对gradient好？
Forward的quantization error是 **deterministic mapping**，可以用more training修正；但gradient是 **stochastic estimation** 本身，deterministic rounding的bias跟SGD noise耦合，无法被average out。Stochastic rounding把bias转成variance，让SGD的inherent averaging机制处理。

## 8. 局限与未来方向

- 仍需 ~15% layers在BF16，未来目标全FP4
- Attention的QK/SV GEMM未量化
- Communication（TP/DP all-reduce）仍在BF16
- 未测MoE
- 未在 >12B 验证scaling law
- System-level throughput数据未报告（只说algorithm/methodology）

## 9. 相关延伸阅读

- NVFP4 official blog: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- MX Formats spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- DeepSeek-V3 FP8: https://arxiv.org/abs/2412.19437
- QuaRot (Hadamard for inference): https://arxiv.org/abs/2404.00456
- Quartet (native FP4 training): https://arxiv.org/abs/2505.14669
- Training LLMs with MXFP4 (Tseng et al.): https://arxiv.org/abs/2502.20586
- FP4 All the Way (Chmiel et al.): https://arxiv.org/abs/2505.19115
- Blackwell architecture brief: https://resources.nvidia.com/en-us-blackwell-architecture
- Nemotron-H: https://arxiv.org/abs/2504.03624
- SmoothQuant (outlier handling classic): https://arxiv.org/abs/2211.10438
- QSGD (stochastic gradient quantization classic): NeurIPS 2017
- Transformer Engine (code): https://github.com/NVIDIA/TransformerEngine

## 10. 一句话总结

NVFP4 pretraining的可行性来自一个 **精密的stack**：硬件提供的E4M3 block scale + FP32两级scaling让FP4 sample被充分利用；2D weight scaling保持forward/backward一致性；RHT抑制Wgrad outlier；stochastic rounding消除gradient bias；最后保留~15% sensitive layers在BF16。四者缺一不可，且重要性随scale增长。

这工作本质上证明： **4-bit pretraining at frontier scale is no longer an algorithmic question, but an engineering one**。
