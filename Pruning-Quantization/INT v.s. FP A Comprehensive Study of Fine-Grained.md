---
source_pdf: INT v.s. FP A Comprehensive Study of Fine-Grained.pdf
paper_sha256: 4ddbb7a38ad9a146410ff621f44f9f67f1dd48d58622c35225009bcc57994f67
processed_at: '2026-08-05T10:01:55-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话总结

**Industry 觉得 FP8/FP4 是未来，但这篇 paper 说：等等，你们没仔细测过 fine-grained INT。实测下来 8-bit 场景 INT 按着 FP 打，4-bit 场景配合 Hadamard rotation INT 也能反杀 FP，而且 hardware 还便宜 30%+。**

Reference: [Paper GitHub](https://github.com/ChenMnZ/INT_vs_FP) | [NVIDIA Blackwell](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)

---

## 1. 背景：Industry 为什么在追 FP？

LLM 有个臭名昭著的问题叫 **activation outliers**。简单说就是大部分 activation 值都很小（比如在 -1 到 1 之间），但偶尔蹦出几个特别大的值（比如 100+）。这种 heavy-tailed distribution 让传统的 INT quantization 很头疼。

为什么头疼？因为 INT 是**均匀网格**，相邻 quantization level 之间的距离是固定的。假设你有 8 bits，能表示 256 个 level，从 -127 到 127，step 是 1。如果你 block 里有个 outlier 是 100，你的 scale 就得设成 100/127 ≈ 0.79，那么所有 0 到 1 之间的值都会被 round 成 1 或 0，精度全毁了。

FP 的妙处在于它是**非均匀网格**，呈 logarithmic spacing。靠近 0 的地方 step 很小（精度高），靠近 max 的地方 step 很大（精度低）。这天然契合 heavy-tailed distribution。所以 NVIDIA Blackwell 原生支持 MXFP8, MXFP4, NVFP4，Industry 跟着走。

但 paper 的作者说：**你们只看了 coarse-grained 的 case，没看 fine-grained block-wise quantization 的 case**。

---

## 2. 关键 Insight：Crest Factor κ 决定一切

### 2.1 什么是 Crest Factor？

$$\kappa := \frac{\max(|\mathbf{X}|)}{\sigma}$$

- $\mathbf{X}$: 一个 block 里的 vector
- $\max(|\mathbf{X}|)$: block 里的最大绝对值（outlier 的 size）
- $\sigma = \mathrm{RMS}(\mathbf{X})$: block 的 root-mean-square（普通值的大小）
- $\kappa$: 就是"outlier 有多 outlier"

**人话**：$\kappa$ 越大，说明 block 里有个特别突兀的值，把整个 scale 拉大了。$\kappa$ 越小，说明 block 里的值比较均匀。

### 2.2 Block Size 越小，κ 越小

这是整个 paper 的核心 leverage。Table 2 的实测数据：

| Block Size | Q3 Crest Factor κ (75% 分位) |
|------------|----------------------------|
| per-channel (无 block) | 11.97 |
| block 32 (MX format) | 2.96 |
| block 16 (NV format) | 2.39 |
| block 32 + Hadamard rotation | 2.39 |
| block 16 + Hadamard rotation | 2.11 |

**人话**：如果你用 per-channel quantization（传统做法），一个 channel 里有个 outlier，整个 channel 都被它污染，$\kappa = 12$。但如果你把 channel 切成 32 个一组的 block，那个 outlier 只影响它所在的 block，其他 31 个 block 的 $\kappa$ 只有 2-3。如果再加 Hadamard rotation 把 outlier "打散"，$\kappa$ 降到 2 出头。

### 2.3 为什么这很重要？

因为 INT 和 FP 的 QSNR 公式里，对 $\kappa$ 的 sensitivity 完全不同：

- **INT QSNR**: $\propto -20 \log_{10}(\kappa)$，$\kappa$ 每翻倍 QSNR 掉 6 dB。INT 极度怕 outlier
- **FP QSNR**: 受 $\kappa$ 影响小得多，因为 FP 的 logarithmic spacing 天然 handle 大值

所以当 $\kappa$ 很大时（coarse-grained），FP 完胜 INT。当 $\kappa$ 很小时（fine-grained），INT 的 uniform grid 反而更 efficient，因为 FP 浪费了很多 exponent bits 去表示你根本不需要的 dynamic range。

---

## 3. 理论框架：QSNR 公式拆解

### 3.1 INT QSNR (Theorem 1, Eq. 13)

$$\mathrm{QSNR_{INT}} \approx 4.78 + 6.02 b - 20 \log_{10}(\rho) - 20 \log_{10}(\kappa)$$

变量解释：
- $b$: bit width（8, 6, 4）
- $\rho \in [1, 2)$: UE8M0 scale 的 power-of-two overhead。因为 MX format 的 scale 只能是 $2^n$，所以实际 scale $s'$ 要 round up 到最近的 power-of-two，相当于最多损失 1 bit
- $\kappa$: crest factor
- $4.78$: 常数，来自 $\log_{10}(12) \times 10 \approx 10.79$，但前面有个负号，所以 $-(-10 \log_{10}(12/4)) = 4.78$。这是 uniform quantization 的经典 Bennett 噪声模型
- $6.02 b$: 每多 1 bit 增加 6.02 dB，经典结论
- $-20 \log_{10}(\rho)$: scale 量化损失，最多 6.02 dB
- $-20 \log_{10}(\kappa)$: **outlier 惩罚**，$\kappa=10$ 就掉 20 dB

**E4M3 scale 情形**（NV format 用）：
$$\mathrm{QSNR_{INT}} \approx 4.78 + 6.02 b - 20 \log_{10}(\kappa) + 10 \log_{10}\left(\frac{g}{g-1}\right)$$

- $g$: block size（NV 是 16）
- 多了一项 $10 \log_{10}(g/(g-1))$，因为 E4M3 scale 精度高，block max 元素几乎 zero error，这一项是 bonus。$g=16$ 时 +0.28 dB，不大但不是 0

### 3.2 FP QSNR (Theorem 2, Eq. 14)

$$\mathrm{QSNR_{FP}} \approx -10 \log_{10}\left(\alpha_M w_{\mathrm{norm}} + \beta (\rho \kappa)^2 p_{\mathrm{sub}}\right)$$

变量解释：
- $M$: mantissa bit width（E4M3 是 3，E2M1 是 1）
- $\alpha_M = \frac{1}{24 \cdot 2^{2M}}$: mantissa resolution term。代表 normal region 的 relative quantization error
- $w_{\mathrm{norm}}$: signal energy 落在 normal region 的 fraction。理想情况 ≈ 1
- $\beta = \frac{2^{2(1-B-M)}}{12 Q_{\max}^2}$: subnormal region 的 fixed-step error 系数
  - $B$: exponent bias（E4M3 是 7）
  - $Q_{\max}$: FP format 的最大 normal magnitude（E4M3 是 448）
- $p_{\mathrm{sub}}$: 值落在 subnormal region 的 probability
- $\rho \in [1, 2)$: scale overhead，同 INT

**关键 insight**：当 dynamic range 充足（$w_{\mathrm{norm}} \approx 1$, $p_{\mathrm{sub}} \approx 0$）时：

$$\mathrm{QSNR_{FP}} \approx 13.80 + 6.02 M \text{ dB}$$

**这是 FP 的天花板**，只取决于 mantissa bits，与 block granularity 和 distribution 无关！

- FP8 (E4M3, M=3): 天花板 31.86 dB
- FP6 (E2M3, M=3): 天花板 31.86 dB（和 FP8 一样！因为 mantissa 都是 3）
- FP4 (E2M1, M=1): 天花板 19.82 dB

### 3.3 Crossover Points

把 INT 和 FP 的 QSNR 公式画在一起（Figure 3），交点就是 crossover κ：

| Format Pair | Crossover κ | 实测 κ (Q3, block 32) | 谁赢？ |
|-------------|-------------|----------------------|--------|
| MXINT8 vs MXFP8 | 7.55 | 2.96 | **INT 碾压** |
| MXINT6 vs MXFP6 | 1.96 | 2.96 | FP 赢 |
| MXINT4 vs MXFP4 | 2.04 | 2.96 | FP 赢 |
| NVINT4 vs NVFP4 | 2.39 | 2.39 (block 16) | 边界 case |
| NVINT4 vs NVFP4 (w/ Hadamard) | 2.39 | 2.11 | **INT 反杀** |

**人话总结**：
- 8-bit：INT 的天花板 48.16 dB（$6.02 \times 8$）远高于 FP8 的 31.86 dB。只要 $\kappa$ 不超过 7.55，INT 都赢。而 fine-grained block 保证了 $\kappa$ 不会那么大
- 4-bit：INT 的天花板 24.08 dB 接近 FP4 的 19.82 dB，差距小。FP 的 logarithmic spacing 优势凸显，所以 FP4 默认占优
- 4-bit + Hadamard：把 $\kappa$ 从 2.39 压到 2.11，刚好跨过 crossover 点 2.39，INT 反杀

---

## 4. 为什么 Hadamard Rotation 能压 κ？

### 4.1 Hadamard Rotation 是什么？

给定一个 hidden dimension $h$，随机选一个 Hadamard matrix $\mathbf{R} \in \mathbb{R}^{h \times h}$（满足 $\mathbf{R}^T \mathbf{R} = h \mathbf{I}$，正交矩阵），然后：

- Forward: 把 activation $\mathbf{X}$ 变成 $\mathbf{X}\mathbf{R}$
- Weight: 把 $\mathbf{W}$ 变成 $\mathbf{R}^T \mathbf{W}$
- 结果: $\mathbf{X}\mathbf{R} \cdot \mathbf{R}^T \mathbf{W} = \mathbf{X} \mathbf{W}$（数学等价）

**人话**：Hadamard rotation 是一个 lossless 的线性变换，它把 "几个特别大的值" 分散到 "所有值都稍微大一点"。outlier 被摊平了。

### 4.2 为什么 FP 反而变差？

这是 paper 里最反直觉的发现。看 Table 13 的 Llama-3.1-8B：
- NVFP4 无 rotation: KL = 3718
- NVFP4 有 rotation: KL = 4752 ← **变差了！**
- NVINT4 无 rotation: KL = 4224
- NVINT4 有 rotation: KL = 3609 ← 变好了

为什么 FP4 rotation 后变差？回到 Theorem 2 的公式。当 $\kappa < 4$ 时，NVFP4 的 QSNR 实际上**随 $\kappa$ 增大而增大**。这是因为：

- $\kappa$ 小 → 更多值落入 subnormal region → $p_{\mathrm{sub}}$ 大 → subnormal error 增大
- $\kappa$ 小 → normal region 的 relative error 减小（因为 max 值更接近 mean）
- 在 $\kappa < 4$ regime，normal region 的 error 占主导，所以 $\kappa$ 减小反而让总 error 增大

**人话**：FP4 的 subnormal region 是个坑。值太小掉进 subnormal 会用 fixed step 量化，误差很大。Hadamard rotation 把 outlier 摊平后，更多值掉进 subnormal region，反而让 FP4 变差。INT 没有这个问题，因为它没有 subnormal region。

Reference: [QuaRot: Outlier-free 4-bit Inference](https://arxiv.org/abs/2404.00456)

---

## 5. Symmetric Clipping：被忽视的 Training 杀手

### 5.1 问题

标准 INT8 two's complement 是 $[-128, 127]$，比 symmetric 多一个 -128。Inference 时无所谓，training 时这个多余的 -128 会造成 **persistent negative gradient bias**。

Figure 2 的实验：用 $[-128, 127]$ 时，block size 越小 loss 越差。block 32 比 block 256 还差！这完全违反"fine-grained 应该更好"的直觉。

### 5.2 BFloat16 的精度陷阱

作者写了个 Algorithm 1 测试：生成 $\mathcal{N}(0,1)$ 矩阵，用 BFloat16 计算 scale 然后 round，看有多少值被错误 map 到 -128。

结果（Table 11）：
| Precision | 被错误 map 到 -128 的比例 |
|-----------|--------------------------|
| BFloat16 | **16.82%** |
| Float16 | 0.02% |
| Float32 | 0% |

**人话**：BFloat16 只有 7 bits mantissa，算 scale 时精度不够，导致 round 后 overflow 到 -128。16.82% 的值被错杀！这就是为什么 fine-grained INT training 会崩。

### 5.3 解决方案

强制 symmetric range：
$$Q_{\min} = -(2^{b-1} - 1), \quad Q_{\max} = 2^{b-1} - 1$$

对于 INT8: $[-127, 127]$。这样即使 BFloat16 精度不够 overflow 了，最多到 127 不会到 -128，gradient bias 消除。

Table 10 的 ablation 证实：symmetric $[-127, 127]$ 在所有 block size 下都优于 asymmetric，fine-grained 时差距最明显（block 32: 3.1354 → 3.1251）。

---

## 6. 实验结果详解

### 6.1 Direct-Cast Inference (Table 3)

12 个 model（Qwen3 0.6B 到 235B-A22B，Llama-3.2 1B 到 3.1-70B），KL divergence metric：

**无 rotation**：
- MXINT8 vs MXFP8: INT **12/12 赢**
- MXINT6 vs MXFP6: FP 12/12 赢
- MXINT4 vs MXFP4: FP 12/12 赢
- NVINT4 vs NVFP4: FP 12/12 赢

**有 Hadamard rotation**：
- MXINT8 vs MXFP8: INT 12/12 赢
- NVINT4 vs NVFP4: INT **12/12 赢**（反转！）
- MXINT6 vs MXFP6: FP 11/12 赢
- MXINT4 vs MXFP4: FP 12/12 赢

具体数字看 Table 12-13，比如 Llama-3.1-8B：
- MXINT8: 82 → 65（rotation 后更好）
- MXFP8: 359 → 409（rotation 后更差！）
- NVINT4: 4224 → 3609
- NVFP4: 3718 → 4752（rotation 后更差！）

### 6.2 Training (Table 4, Figure 5)

Llama-1B 100B tokens 和 Llama-3B 200B tokens 训练：

**1B 模型**：
| Precision | Loss | Avg Accuracy (6 tasks) |
|-----------|------|------------------------|
| BF16 | 2.6727 | 56.89 |
| MXFP8 | 2.6767 | 56.86 |
| MXINT8 | 2.6758 | 57.02 |

**3B 模型**：
| Precision | Loss | Avg Accuracy |
|-----------|------|--------------|
| BF16 | 2.4794 | 64.45 |
| MXFP8 | 2.4821 | 64.05 |
| MXINT8 | 2.4812 | 64.30 |

**人话**：MXINT8 训练 loss 比 MXFP8 低约 0.001，accuracy 略高，**几乎 lossless**。这是重要 contribution，因为 prior work（DeepSeek-V3, NVIDIA recipes）都 focus FP8 training。

Reference: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [MXFP8 Training Recipes](https://arxiv.org/abs/2506.08027)

---

## 7. Hardware Cost：INT 便宜多少？

### 7.1 为什么 INT 在 hardware 上便宜？

FP 的 MAC unit 需要这些 INT 不需要的东西：
1. **Exponent adder**（乘法时 exponent 相加）
2. **Exponent subtractor + comparator**（加法时找大的 exponent）
3. **Mantissa aligner / barrel shifter**（加法时把小 exponent 的 mantissa 右移对齐）
4. **Normalizer**（加法结果重新 normalize）

这些 logic 在 gate count 上很贵，尤其 barrel shifter 是 $O(n \log n)$ 的 MUX 阵列。

### 7.2 数字对比 (Table 5)

| Format | Energy | Area |
|--------|--------|------|
| MXFP8 | 1.0x | 1.0x |
| **MXINT8** | **0.63x** | **0.79x** |
| NVFP4 | 0.55x | 0.54x |
| **NVINT4** | **0.34x** | **0.38x** |

**MXINT8 比 MXFP8 节省 37% energy, 21% area**。

Mixed-format（8-bit + 4-bit，throughput 1:2，模拟 Blackwell）：
| Configuration | Energy | Area |
|---------------|--------|------|
| MXFP8 + NVFP4 | 1.0x | 1.0x |
| **MXINT8 + NVINT4** | **0.75x** | **0.66x** |

**节省 25% energy, 34% area**。主要因为 INT 的 circuit reuse 更简单（Table 7: INT 可以用 2 个 int8×int4 lane 复用，FP 需要更复杂的 reuse scheme）。

---

## 8. 我的理解和 Extension

### 8.1 核心洞察

paper 揭示了一个被忽视的维度：**quantization format 和 granularity 是耦合的**。不能脱离 granularity 谈 format。coarse-grained FP 好是正确的，但 fine-grained 时 INT 的 uniform grid 反而更 efficient，因为 local dynamic range 小了，FP 浪费 exponent bits。

### 8.2 Crest Factor 作为统一 metric 的价值

paper 最大的 contribution 是把 INT vs FP 的 trade-off 归约到单一变量 $\kappa$ 上。Practical 意义：**先测你 data 的 $\kappa$ distribution，再选 format**。如果 Q3 $\kappa < 7.55$，无脑选 MXINT8。

### 8.3 Hadamard Rotation 改变 format 选择

这很有意思。rotation 不仅是 outlier suppression 工具，它改变了 INT 和 FP 的相对优势。Future work 可以探索：
- [SpinQuant](https://arxiv.org/abs/2405.16406) 的 learned rotation 与 format selection 的 joint optimization
- Hardware-native Hadamard transform unit 的成本
- 不同 rotation matrix（DCT, DFT, random orthogonal）对 $\kappa$ 的影响

### 8.4 FP4 的 subnormal 陷阱

NVFP4 在 $\kappa < 4$ 时 QSNR 随 $\kappa$ 增大而增大，这个反直觉现象的本质是 FP 的 subnormal region 是 fixed-step quantization，error 很大。rotation 把值摊平后更多值掉进 subnormal，反而变差。这暗示 future FP format 设计应该考虑：
- Flush-to-zero (FTZ) 模式
- Non-uniform mantissa allocation
- 更小的 subnormal region

### 8.5 Hybrid Format 策略

[ICME 2024](https://ieeexplore.ieee.org/document/10687605) 已经探索过 mixed INT/FP 的 PTQ。这篇 paper 的理论框架可以指导 hybrid 策略：根据每层的 $\kappa$ distribution 动态选择 format。比如 attention 的 pre-softmax 用 FP（outlier 多），FFN 用 INT（outlier 少）。

### 8.6 Training 的 further questions

- MXINT8 training 在更大 model（7B+）上是否仍然 nearly lossless？
- Optimizer state（AdamW 的 m, v）能否也用 MXINT8？
- MXINT8 与 LoRA / QAT 的结合效果？
- [Quartet](https://arxiv.org/abs/2505.14669) 说 native FP4 training 可以 optimal，那 native INT4 training 呢？

### 8.7 与 Scaling Laws 的关联

[Kumar et al. 2024](https://arxiv.org/abs/2411.04330) 的 scaling laws for precision 说 optimal bit width 随 model size 变化。这篇 paper 的 framework 可以与之结合，推导不同 model size 下的 optimal (format, block size, bit width) triple。

---

## 9. 对 Industry 的 Implications

### 9.1 NVIDIA Blackwell 选错了吗？

Blackwell 原生支持 MXFP8, MXFP4, NVFP4，但不原生支持 INT variants。这篇 paper 说这个选择是 suboptimal 的：

1. **8-bit 场景**：MXINT8 在 accuracy 和 hardware efficiency 上双杀 MXFP8。继续走 FP8 是 pure loss
2. **4-bit 场景**：FP4 默认占优，但配合 Hadamard rotation 后 NVINT4 反超 NVFP4，且 hardware cost 只有 34%
3. **Training 场景**：MXINT8 training 是 nearly lossless 的

### 9.2 下一步 hardware 应该怎么做？

paper 呼吁 algorithm-hardware co-design 重新评估 fine-grained INT formats。具体：
- Native support for MXINT8, NVINT4
- Hardware-native Hadamard transform unit
- Symmetric clipping 的原生支持

---

## 10. Take-aways

1. **INT vs FP 不是简单的谁好谁坏，是 $\kappa$ 的函数**
2. **8-bit fine-grained: MXINT8 完胜 MXFP8**（accuracy + hardware 双优）
3. **4-bit: FP4 默认占优，但 Hadamard rotation 后 NVINT4 反超**
4. **Symmetric clipping $[-127, 127]$ 是 INT training 的必需品**，BFloat16 精度问题导致 16.82% overflow
5. **Hardware cost: INT 在 multiplier 和 adder 上都有显著优势**，mixed-format 下节省 34% area
6. **Industry 的 FP-centric 路线值得 re-evaluate**

---

## References

- [Paper Code](https://github.com/ChenMnZ/INT_vs_FP)
- [OCP Microscaling Formats](https://arxiv.org/abs/2310.10537)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)
- [NVIDIA TensorRT Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)
- [QuaRot: Outlier-free 4-bit Inference](https://arxiv.org/abs/2404.00456)
- [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406)
- [Scaling Laws for Precision](https://arxiv.org/abs/2411.04330)
- [Integer or Floating Point? (ICME 2024)](https://ieeexplore.ieee.org/document/10687605)
- [Massive Activations in LLMs](https://arxiv.org/abs/2402.17762)
- [SmoothQuant](https://arxiv.org/abs/2210.17323)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [OmniQuant](https://arxiv.org/abs/2308.13137)
- [PrefixQuant](https://arxiv.org/abs/2410.05265)
- [EfficientQAT](https://arxiv.org/abs/2407.11062)
- [Scaling Law for QAT](https://arxiv.org/abs/2505.14302)
- [Quartet: Native FP4 Training](https://arxiv.org/abs/2505.14669)
- [LLM Inference Survey](https://arxiv.org/abs/2402.16363)
- [Bennett 1948: Spectra of Quantized Signals](https://ieeexplore.ieee.org/document/6773024)

---

# INT v.s. FP: Fine-Grained Low-bit Quantization Formats 深度解读

## 1. 这篇论文要解决的核心矛盾

Industry 正在向 low-precision FP 靠拢（NVIDIA Blackwell原生支持 MXFP8/MXFP4/NVFP4），理由是 FP 的 dynamic range 更适合 handle LLM activation outliers。但是这篇 paper 揭示了一个被忽视的关键 issue：**当 quantization granularity 变细（block-wise）之后，INT 和 FP 的 trade-off 会发生 crossover**。Coarse-grained 时 FP 确实占优，但 fine-grained 时情况翻转，尤其 8-bit 场景下 MXINT8 全面碾压 MXFP8。

Reference: [OCP Microscaling formats](https://arxiv.org/abs/2310.10537) | [NVIDIA Blackwell architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)

---

## 2. 量化的数学基础

### 2.1 INT Quantization (Eq. 1)

$$\mathbf{X_q} = \mathrm{clip}\left(\left\lfloor \frac{\mathbf{X}}{s} \right\rceil, Q_{\min}, Q_{\max}\right) \cdot s$$

- $\mathbf{X}$: high-precision input tensor (BF16/FP32)
- $s$: scale factor，把 X 的 range 映射到 integer 表示范围
- $\lfloor \cdot \rceil$: round-to-nearest 操作
- $Q_{\min}, Q_{\max}$: integer 表示的边界。Symmetric 时 $Q_{\min} = -(2^{b-1}-1)$, $Q_{\max} = 2^{b-1}-1$，其中 $b$ 是 bit width
- $\mathbf{X_q}$: dequantized 后的 tensor

**Intuition**: INT 是 uniform grid，相邻 quantization level 间距恒定为 $s$，所以一旦 data 有 outlier，整个 scale 被拉大，normal 区域的 resolution 急剧下降。

### 2.2 FP Quantization (Eq. 2-3)

$$\mathbb{C}_{\mathrm{FP}} = \begin{cases} (-1)^s \times (1.m)_2 \times 2^{e-\mathrm{bias}} & \text{if } e \neq 0 \text{ (Normal)} \\ (-1)^s \times (0.m)_2 \times 2^{1-\mathrm{bias}} & \text{if } e = 0, m \neq 0 \text{ (Subnormal)} \end{cases}$$

- $s$: sign bit (1 bit)
- $e$: exponent bits（E4M3 中 $e$ 有 4 bits）
- $m$: mantissa bits（E4M3 中 $m$ 有 3 bits）
- $\mathrm{bias}$: exponent bias，E4M3 中 bias=7
- $(1.m)_2$: normal 数的隐式 leading 1
- $(0.m)_2$: subnormal 数无隐式 leading 1

FP 是**非均匀 grid**，靠近 0 的地方 step 小（resolution 高），靠近 max 的地方 step 大（resolution 低）。这种 logarithmic spacing 天然契合 heavy-tailed distribution。

### 2.3 Block-wise Quantization 与 Scale Formats

- **MX format**: block size = 32，每个 block 共享一个 UE8M0 scale（8-bit unsigned exponent only，即 power-of-two）
- **NV format**: block size = 16，scale 用 E4M3（更精确，有 mantissa），还有第二层 per-tensor FP32 scale 防止 overflow

UE8M0 scale 的 overhead 关键：因为只能是 power-of-two，所以 $s' = 2^{\lceil \log_2 s \rceil} = \rho s$，其中 $\rho \in [1, 2)$。这意味着最多损失 1 bit 的 effective precision（6.02 dB QSNR）。

---

## 3. 理论框架：QSNR 推导

### 3.1 QSNR Metric (Eq. 10)

$$\mathrm{QSNR} = -10 \log_{10}\left(\frac{\|\mathbf{X} - \mathbf{X_q}\|^2}{\|\mathbf{X}\|^2}\right)$$

这是 relative MSE 的 dB 形式，越高代表 quantization noise 越小。

### 3.2 Crest Factor κ (Eq. 11)

$$\kappa := \frac{\max(|\mathbf{X}|)}{\sigma}$$

- $\mathbf{X} \in \mathbb{R}^k$: 一个 block 内的 vector
- $\sigma = \mathrm{RMS}(\mathbf{X})$: block 的 root-mean-square
- $\kappa$ 衡量"outlier 程度"，越大代表 block 内有 extreme outlier

**这是论文最核心的变量**。一切 trade-off 都围绕 $\kappa$ 旋转。Block size 越小，$\kappa$ 越小（因为 outlier 被 local化了）。Table 2 显示：
- per-channel (block size ∞): Q3 κ = 11.97（巨大 outlier）
- block size 32: Q3 κ = 2.96
- block size 16: Q3 κ = 2.39
- block size 32 + Hadamard rotation: Q3 κ = 2.39
- block size 16 + Hadamard rotation: Q3 κ = 2.11

### 3.3 Theorem 1: INT QSNR (Eq. 13)

**UE8M0 scale 情形**：
$$\mathrm{QSNR_{INT}} \approx 4.78 + 6.02 b - 20 \log_{10}(\rho) - 20 \log_{10}(\kappa)$$

**E4M3 scale 情形**：
$$\mathrm{QSNR_{INT}} \approx 4.78 + 6.02 b - 20 \log_{10}(\kappa) + 10 \log_{10}\left(\frac{g}{g-1}\right)$$

变量解释：
- $b$: bit width（8, 6, 4）
- $\rho \in [1, 2)$: UE8M0 scale 的 power-of-two overhead。$\rho=1.5$ 是 average case
- $\kappa$: crest factor
- $g$: block size（32 for MX, 16 for NV）
- $10 \log_{10}(g/(g-1))$: E4M3 scale 没有 $\rho$ overhead，且 block max 元素几乎 zero error，所以扣除这一项贡献

**Intuition 分解**：
1. **6.02 b**: 每多 1 bit 增加 6.02 dB QSNR（经典量化理论）
2. **-20 log₁₀(ρ)**: UE8M0 scale 的 power-of-two quantization 最多损失 6.02 dB（当 $\rho \to 2$）
3. **-20 log₁₀(κ)**: 这是 INT 的**阿喀琉斯之踵**。$\kappa$ 越大，scale 越被 outlier 拉大，正常值 resolution 越差。$\kappa=10$ 时损失 20 dB
4. **$10 \log_{10}(g/(g-1))$**: E4M3 的 bonus。block 越小这个 bonus 越大（$g=32$ 时 +0.135 dB，$g=16$ 时 +0.28 dB），因为 block max 元素是 zero-error 的，相对贡献更大

### 3.4 Theorem 2: FP QSNR (Eq. 14)

**UE8M0 scale 情形**：
$$\mathrm{QSNR_{FP}} \approx -10 \log_{10}\left(\alpha_M w_{\mathrm{norm}} + \beta (\rho \kappa)^2 p_{\mathrm{sub}}\right)$$

**E4M3 scale 情形**：
$$\mathrm{QSNR_{FP}} \approx -10 \log_{10}\left(\alpha_M \left(w_{\mathrm{norm}} - \frac{\kappa^2}{g}\right) + \beta \kappa^2 p_{\mathrm{sub}}\right)$$

变量解释：
- $M$: mantissa bit width
- $\alpha_M = \frac{1}{24 \cdot 2^{2M}}$: **mantissa resolution term**。代表 normal region 的相对量化误差。每多 1 mantissa bit 增加 6.02 dB
- $B$: exponent bias
- $Q_{\max}$: FP format 的最大 normal magnitude（E4M3 是 448）
- $\beta = \frac{2^{2(1-B-M)}}{12 Q_{\max}^2}$: subnormal region 的 fixed-step error 系数
- $w_{\mathrm{norm}}$: signal energy 落在 normal region 的 fraction（理想情况 ≈ 1）
- $p_{\mathrm{sub}}$: value 落在 subnormal region 的 probability
- $\kappa^2/g$: block max 元素的能量 fraction，E4M3 scale 时可以减去

**Intuition 分解**：
1. **Upper bound**: 当 dynamic range 充足（$w_{\mathrm{norm}} \approx 1$, $p_{\mathrm{sub}} \approx 0$）时，QSNR ≈ $13.80 + 6.02 M$ dB，**与 block granularity 和 distribution 无关**！这就是 FP 的"天花板"特性
2. **κ 的影响是双面的**：κ 增大 → 更多 values 落入 subnormal → $p_{\mathrm{sub}}$ 增大 → QSNR 下降。但同时 normal region 的 relative error 减小（因为 $\kappa^2/g$ 项）
3. **为什么 NVFP4 在 κ < 4 时 QSNR 反而随 κ 增大而增大**：因为此时 normal region 的 error 占主导，而 larger κ 让 subnormal region 占比减小，反而减少 error。这是一个反直觉的现象

### 3.5 Crossover Points (Figure 3)

论文通过比较 Eq. 13 和 Eq. 14 得到 critical crossover points：

| Format Pair | Crossover κ | Interpretation |
|-------------|-------------|----------------|
| MXINT8 vs MXFP8 | **7.55** | INT8 几乎总是赢（实测 κ 远低于 7.55） |
| MXINT6 vs MXFP6 | **1.96** | FP6 几乎总是赢 |
| MXINT4 vs MXFP4 | **2.04** | FP4 几乎总是赢 |
| NVINT4 vs NVFP4 | **2.39** | 边界 case，Hadamard 后 INT4 赢 |

**核心 insight**：
- **8-bit 时 INT 占优**：因为 6.02×8 = 48.16 dB 远高于 FP8 的 13.80 + 6.02×3 = 31.86 dB。INT 8-bit 的 dynamic range 足够覆盖 block 内 κ 较小的 data
- **4-bit 时 FP 占优**：因为 6.02×4 = 24.08 dB 接近 FP4 的 13.80 + 6.02×1 = 19.82 dB，差距小，但 FP4 的 logarithmic spacing 在低 bit 时优势明显
- **6-bit 是 middle ground**：FP6 略胜，因为 mantissa=3 提供了 decent resolution

---

## 4. Compute Flow 与 Six Quantization Operations (Figure 1)

Linear layer 的 forward 和 backward 共有 6 个 quantization 点：

**Forward**:
$$\mathbf{Y} = \underbrace{\mathrm{Quantize}(\mathbf{X})}_{①} \underbrace{\mathrm{Quantize}(\mathbf{W})}_{②}$$

**Backward (dX)**:
$$d\mathbf{X} = \underbrace{\mathrm{Quantize}(d\mathbf{Y})}_{③} \underbrace{\mathrm{Quantize}(\mathbf{W}^T)}_{④}$$

**Backward (dW)**:
$$d\mathbf{W} = \underbrace{\mathrm{Quantize}(\mathbf{X}^T)}_{⑤} \underbrace{\mathrm{Quantize}(d\mathbf{Y}^T)}_{⑥}$$

关键点：block quantization 沿 GEMM reduction dimension 进行。所以 ①⑤（X 方向）、②④（W 方向）、③⑥（dY 方向）是不同 axis 的 quantization。Tensor-wise analysis 在 10752 个 tensors（224 linear layers × 6 operations）上测量 QSNR，验证理论。

---

## 5. 实验结果深度解析

### 5.1 Tensor-wise QSNR (Figure 4)

| Format Pair | Average QSNR (INT) | Average QSNR (FP) | INT Win Rate |
|-------------|-------------------|-------------------|--------------|
| MXINT8 vs MXFP8 | 40.35 | 31.50 | ~100% |
| MXINT6 vs MXFP6 | lower | higher | < 50% |
| MXINT4 vs MXFP4 | lower | higher | < 50% |
| NVINT4 vs NVFP4 | 20.55 | 20.60 | 64.3% |
| NVINT4 vs NVFP4 (w/ Hadamard) | 21.65 | 20.35 | ~100% |

**MXFP8 QSNR 恒定 31.50 dB**：完全符合 Theorem 2 的 prediction（13.80 + 6.02×3 = 31.86，略低是因为 $w_{\mathrm{norm}}$ 略小于 1）。

### 5.2 Direct-Cast Inference (Table 3, 12 models)

**Without Hadamard rotation**:
- MXINT8 vs MXFP8: INT 赢 12/12
- MXINT6 vs MXFP6: FP 赢 12/12
- MXINT4 vs MXFP4: FP 赢 12/12
- NVINT4 vs NVFP4: FP 赢 12/12

**With Hadamard rotation**:
- MXINT8 vs MXFP8: INT 赢 12/12
- NVINT4 vs NVFP4: INT 赢 12/12 ← **关键反转**
- MXINT6 vs MXFP6: FP 赢 11/12
- MXINT4 vs MXFP4: FP 赢 12/12

KL divergence 详细数据（Table 12, Llama-3.1-8B 为例）：
- MXINT8: 82 → 65 (with rotation)
- MXFP8: 359 → 409 (with rotation，FP 反而变差！)
- NVINT4: 4224 → 3609 (with rotation)
- NVFP4: 3718 → 4752 (with rotation，FP 反而变差)

**为什么 Hadamard rotation 让 FP 反而变差？** 因为 NVFP4 在 κ < 4 时 QSNR 随 κ 增大而增大（Theorem 2 的反直觉现象），所以 rotation 降低 κ 反而降低 NVFP4 的 QSNR。

### 5.3 Training Results (Table 4, Figure 5)

Llama-1B, 100B tokens 训练：
| Precision | Loss | Avg Accuracy (6 tasks) |
|-----------|------|------------------------|
| BF16 | 2.6727 | 56.89 |
| MXFP8 | 2.6767 | 56.86 |
| **MXINT8** | **2.6758** | **57.02** |

Llama-3B, 200B tokens：
| Precision | Loss | Avg Accuracy |
|-----------|------|--------------|
| BF16 | 2.4794 | 64.45 |
| MXFP8 | 2.4821 | 64.05 |
| **MXINT8** | **2.4812** | **64.30** |

MXINT8 训练 loss 比 MXFP8 低约 0.001，accuracy 略高，**几乎 lossless**。这是 paper 的重要 contribution，因为 prior work（DeepSeek-V3, MXFP8 training）主要 focus FP8 training。

Reference: [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) | [Recipes for pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)

---

## 6. Symmetric Clipping 的必要性 (Section 3.2, D.2)

### 6.1 问题：Asymmetric Integer Range 导致 Gradient Bias

标准 INT8 的 two's complement 表示是 $[-128, 127]$，多出一个 -128。在 inference 时影响不大，但 training 时会产生 persistent negative gradient bias。

Figure 2 显示：使用 asymmetric $[-128, 127]$ 时，block size 越小 training loss 越差（block 32 比 block 256 还差）。原因是：**更细的 granularity → 更多 block → 更多 block min 值被 map 到 -128**。

### 6.2 BFloat16 的精度陷阱 (Table 11, Algorithm 1)

即使理论上 scale $s = \mathrm{AbsMax}(\mathbf{X})/127$，BFloat16 精度不够，约 **16.82%** 的值会被错误 map 到 -128！Float16 是 0.02%，Float32 是 0%。

```
Algorithm 1 的核心逻辑：
1. 生成 N×N 的 N(0,1) 矩阵 D
2. 计算 S = D / 127
3. D_norm = round(D / S)  ← 这里应该得到 127，但 BFloat16 精度问题导致 overflow 到 -128
4. 统计等于 128 的元素比例
```

### 6.3 解决方案

强制使用 symmetric range $[-127, 127]$ for INT quantization（所有 bit width）：
$$Q_{\min} = -(2^{b-1} - 1), \quad Q_{\max} = 2^{b-1} - 1$$

Table 10 的 ablation 证明：symmetric $[-127, 127]$ 在所有 block size 下都优于 asymmetric $[-128, 127]$，尤其 fine-grained 时差距明显（block 32: 3.1354 → 3.1251 on BF16 scale）。

---

## 7. Hardware Cost Analysis (Section 6, Appendix C)

### 7.1 MAC Unit Gate-Level Model (Table 6)

Paper 建模了 k-lane MAC unit 的主要 sub-blocks：
- **Multiplier**: INT $k(x+y+1)^2$ vs FP $k(y+1)^2$（FP 只乘 mantissa）
- **Adder**: INT $2k(x+y+1)$ vs FP $kn$（FP 需要 alignment）
- **Exponent adder/subtractor/comparator**: FP 独有
- **Aligner (barrel)**: FP 独有，$kn \log_2 n$
- **Normalizer**: FP 独有，shared across k lanes

其中 aligner width $n$ (Eq. 44):
$$n = \min(2^{x+1} + 2y, \mathrm{psum\_bit\_width})$$
- $x$: exponent width
- $y$: mantissa width
- INT 时 $x = 0$

### 7.2 Energy 和 Area 对比 (Table 5)

| Format | Energy | Area |
|--------|--------|------|
| MXFP8 (baseline) | 1.0x | 1.0x |
| **MXINT8** | **0.63x** | **0.79x** |
| NVFP4 | 0.55x | 0.54x |
| **NVINT4** | **0.34x** | **0.38x** |
| MXFP8+NVFP4 (baseline) | 1.0x | 1.0x |
| **MXINT8+NVINT4** | **0.75x** | **0.66x** |

**MXINT8 比 MXFP8 节省 37% energy, 21% area**。主要因为：
1. INT multiplier 不需要 exponent 处理
2. INT adder 不需要 alignment 和 normalization
3. INT pipeline 的 circuit reuse 更简单（Table 7: INT reuse scheme 2 用 2 个 int8×int4 lane 即可配置，FP 需要更复杂的 reuse scheme）

Mixed-format（8-bit + 4-bit，throughput ratio 1:2）下，INT 组合节省 34% area。

---

## 8. 对当前 Hardware Trajectory 的挑战

NVIDIA Blackwell 选择 FP-centric 路线（原生支持 MXFP8/MXFP4/NVFP4 但不原生支持 INT variants）。这篇 paper 论证这个选择是 suboptimal 的：

1. **8-bit 场景**：MXINT8 在 accuracy 和 hardware efficiency 上**双杀** MXFP8。继续走 FP8 路线是 pure loss
2. **4-bit 场景**：FP4 默认占优，但配合 Hadamard rotation 后 NVINT4 反超 NVFP4，且 hardware cost 只有 34%
3. **Training 场景**：MXINT8 training 是 nearly lossless 的，不需要 FP8 的 complexity

Paper 呼吁 algorithm-hardware co-design 重新评估 fine-grained INT formats。

---

## 9. 我的 Critical Thoughts 和 Extension

### 9.1 Crest Factor 作为统一 metric 的优雅性

这篇 paper 最大的 theoretical contribution 是把 INT vs FP 的 trade-off 归约到单一变量 $\kappa$ 上。这给了 practitioner 一个 actionable 的判断标准：**先测量你 data 的 crest factor，再选 format**。

### 9.2 Hadamard Rotation 的双重作用

Random Hadamard rotation ([QuaRot](https://arxiv.org/abs/2404.00456)) 不仅是 outlier suppression 工具，它实际上改变了 INT 和 FP 的相对优势。这是一个 under-explored 的 algorithm-hardware co-design 维度。可以想象 future work 探索：
- Learned rotation（[SpinQuant](https://arxiv.org/abs/2405.16406)）与 format selection 的 joint optimization
- Hardware-native Hadamard transform unit 的成本
- 不同 rotation matrix（DCT, DFT）对 κ 的影响

### 9.3 Subnormal Region 的处理

FP 的 subnormal region 是 $p_{\mathrm{sub}}$ 项的来源，是 FP QSNR 的主要 degradation source。可以考虑：
- Flush-to-zero (FTZ) 模式：直接把 subnormal 当 0，减少 hardware cost 但增加 error
- Non-uniform mantissa allocation：在小值区域分配更多 mantissa bits

### 9.4 与 Scaling Laws for Precision 的关联

[Kumar et al. 2024](https://arxiv.org/abs/2411.04330) 的 scaling laws for precision 提出 optimal bit width 随 model size 变化。这篇 paper 的 framework 可以与之结合，推导不同 model size 下的 optimal (format, block size, bit width) triple。

### 9.5 MXINT8 Training 的 further questions

- MXINT8 training 在更大 model（7B+）上是否仍然 nearly lossless？
- Optimizer state（AdamW 的 m, v）能否也用 MXINT8？
- MXINT8 与 LoRA / QAT 的结合效果？

### 9.6 Hybrid Format 探索

[Zhang et al. 2024](https://ieeexplore.ieee.org/document/10687605) 已经探索过 mixed INT/FP 的 PTQ。这篇 paper 的理论框架可以指导 hybrid 策略：根据每层的 $\kappa$ distribution 动态选择 format。例如 attention layer 的 softmax 之前用 FP，FFN 用 INT。

---

## 10. Summary 的 Take-aways

1. **INT vs FP 不是简单的谁好谁坏，而是 crest factor κ 的函数**
2. **8-bit fine-grained: MXINT8 完胜 MXFP8**（accuracy + hardware 双优）
3. **4-bit: FP4 默认占优，但 Hadamard rotation 后 NVINT4 反超**
4. **Symmetric clipping $[-127, 127]$ 是 INT training 的必需品**，解决 BFloat16 精度导致的 16.82% overflow 问题
5. **Hardware cost: INT 在 multiplier 和 adder 上都有显著优势**，mixed-format 下节省 34% area
6. **Industry 的 FP-centric 路线值得 re-evaluate**

Code: [https://github.com/ChenMnZ/INT_vs_FP](https://github.com/ChenMnZ/INT_vs_FP)

---

## References

- [Microscaling Data Formats for Deep Learning (OCP MX)](https://arxiv.org/abs/2310.10537)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)
- [NVIDIA TensorRT Quantized Types Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [DeepSeek-V3 Technical Report (MXFP8 training)](https://arxiv.org/abs/2412.19437)
- [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)
- [QuaRot: Outlier-free 4-bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
- [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406)
- [Scaling Laws for Precision](https://arxiv.org/abs/2411.04330)
- [Integer or Floating Point? New Outlooks for Low-bit Quantization on LLMs](https://ieeexplore.ieee.org/document/10687605)
- [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762)
- [SmoothQuant: Accurate and Efficient PTQ for LLMs](https://arxiv.org/abs/2210.17323)
- [GPTQ: Accurate PTQ for GPTs](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [OmniQuant: Omnidirectionally Calibrated Quantization](https://arxiv.org/abs/2308.13137)
- [PrefixQuant: Eliminating Outliers by Prefixed Tokens](https://arxiv.org/abs/2410.05265)
- [EfficientQAT: Efficient Quantization-Aware Training for LLMs](https://arxiv.org/abs/2407.11062)
- [Scaling Law for Quantization-Aware Training](https://arxiv.org/abs/2505.14302)
- [Quartet: Native FP4 Training Can Be Optimal for LLMs](https://arxiv.org/abs/2505.14669)
- [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363)
- [The New IEEE-754 Standard for Floating Point Arithmetic](https://link.springer.com/chapter/10.1007/978-3-540-85521-7_1)
- [Bennett 1948: Spectra of Quantized Signals](https://ieeexplore.ieee.org/document/6773024)
- [W. R. Bennett经典论文](https://onlinelibrary.wiley.com/doi/10.1002/j.1538-7305.1948.tb01364.x)
