---
source_pdf: Four Over Six.pdf
paper_sha256: 76d62efa49143a7b2fca3ca79d53858e442226dc0c8c7696d791502cdea55898
processed_at: '2026-08-04T10:16:30-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，我们用最直白的 human language 来拆解这篇 paper。核心 intuition 其实非常优美，它抓住了 floating point 量化中一个极其反直觉的几何特征。

---

### 一、 问题的几何直觉：FP4 的“断层”

我们要做 LLM 的 quantization，把 BF16 压成 4-bit。NVIDIA Blackwell 硬件原生支持的格式叫 NVFP4。FP4 E2M1 总共只能表示 16 个值，正数部分是：
`{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`

你在数轴上画一下这些点，立刻就能发现一个巨大的“断层”：从 4.0 到 6.0 之间，步长突然变成了 2.0。

在标准的 NVFP4 quantization 中，算法会找到一个 tensor 里的最大值，然后把它 scale 到 6.0（即 $M^{FP4} = 6$）。假设一个 block 里有 16 个数字，最大值是 40，那么 scale factor 就是 $40 / 6 \approx 6.67$。

此时，如果 block 里有个值是 30，它除以 scale 后变成 $30 / 6.67 = 4.5$。
你看向 FP4 的数轴：4.5 刚好掉在 4.0 和 6.0 之间的断层里！它只能被 round 到 4.0，反量化后变成 26。原本是 30，现在变 26，误差高达 13.3%。

这就是 NVFP4 精度受损的罪魁祸首：**近最大值（near-maximal values）掉进了 4 到 6 的断层里**。Paper 里的 Figure 2b 做了个极其漂亮的 ablation 实验：只对 scaled 后大于 5 的值进行量化，模型 perplexity 瞬间崩盘；而小于 4 的值量化，几乎无害。

---

### 二、 核心 Insight：换个尺子量

既然 4 到 6 之间有断层，那我们能不能不用 6 这把尺子？

如果我把最大值 40 scale 到 4.0（即 $M^{FP4} = 4$），scale factor 变成 $40 / 4 = 10$。
那个原本是 30 的值，现在变成 $30 / 10 = 3.0$。
3.0 在 FP4 里有精确表示！误差为 0。

这就是 "Four Over Six" (4/6) 的核心思想：**与其无脑把所有 block 的最大值拉到 6，不如有些 block 拉到 4**。拉到 4 的时候，你放弃了表示 6.0 的能力，但换来了 3.0 这个极其有用的点。因为 30/40 = 0.75，恰好等于 3/4 = 0.75，很多真实数据的分布刚好在这个区间里能被完美命中。

---

### 三、 方法论：Adaptive Block Scaling

你不能把所有 block 都拉到 4。如果某个 block 里真有个极端 outlier，拉到 4 会把小值全压死。所以必须 adaptive。

算法极其直接：对每个 block 量化两次。
1. 算一次 $M=6$ 的结果 $\bar{\mathbf{X}}^{(6)}$
2. 算一次 $M=4$ 的结果 $\bar{\mathbf{X}}^{(4)}$
3. 分别计算它们的 Mean Squared Error (MSE)
4. 谁的 MSE 小，就用谁的结果。

**公式与变量解析：**

标准 NVFP4 的 quantization 公式（Eq. 3）是一个 piecewise function，根据归一化后的值 $\frac{\mathbf{X}}{\alpha \Delta}$ 的大小决定 rounding 逻辑：
- 如果 $|\frac{\mathbf{X}}{\alpha \Delta}| < 2$，步长是 0.5，所以乘以 2 后 round 再除以 2：$\frac{1}{2} \lceil \frac{2\mathbf{X}}{\alpha \Delta} \rfloor$
- 如果 $|\frac{\mathbf{X}}{\alpha \Delta}| < 4$，步长是 1，直接 round：$\lceil \frac{\mathbf{X}}{\alpha \Delta} \rfloor$
- 如果 $|\frac{\mathbf{X}}{\alpha \Delta}| \leq 6$，步长是 2，除以 2 后 round 再乘以 2：$2 \lceil \frac{\mathbf{X}}{2\alpha \Delta} \rfloor$

变量解释：
- $\mathbf{X}$: 原始高精度 tensor
- $\alpha$: tensor 级别的 FP32 scale factor
- $\Delta$: block 级别的 FP8 E4M3 scale factor
- $\lceil \cdot \rfloor$: rounding function (可以是 round-to-nearest 或 stochastic rounding)

4/6 就是分别用 $M=6$ 和 $M=4$ 跑这个流程，然后选 MSE 低的。

---

### 四、 工程实现里最骚的操作：修补 Tensor Scale

这里有个非常精妙的工程细节。NVFP4 是一个三级结构：FP32 tensor scale $\alpha$ -> FP8 block scale $\Delta_i$ -> FP4 value。

标准 NVFP4 算 tensor scale 的公式（Eq. 1）：
$$\alpha^{\mathrm{FP32}} = \frac{\max(|\mathbf{X}|)}{M^{\mathrm{FP4}} \times M^{\mathrm{FP8}}}$$

在标准设定下，$M^{\mathrm{FP4}} = 6$，FP8 E4M3 的最大值 $M^{\mathrm{FP8}} = 448$。这意味着全局最大值会被压缩到 $6 \times 448 = 2688$ 的范围内。

问题来了：如果某个包含全局最大值的 block 想用 $M=4$，它的 block scale $\Delta$ 需要变成多少？
原本对应 6 的 $\Delta$ 是 448，现在对应 4，$\Delta$ 需要放大 $6/4 = 1.5$ 倍，也就是 $448 \times 1.5 = 672$。
但是，FP8 E4M3 **最大只能表示 448**！Overflow 了！

**4/6 的解决方案：**
把全局的 $M^{\mathrm{FP8}}$ 从 448 强制降到 **256**。
此时，如果要用 $M=4$，需要的 $\Delta = 256 \times 1.5 = 384$。384 完美落在 E4M3 的表示范围内（448以内），可以精确表示。

这相当于牺牲了一点点全局的动态范围（从 2688 降到 1536），换取了 block 级别在 4 和 6 之间自由切换的灵活性。这个 trade-off 极其划算。

---

### 五、 硬件 Kernel 实现：为什么 Overhead 只有 15%？

你要对一个 block 量化两次，听起来计算量翻倍了。但在 NVIDIA Blackwell GPU 上，他们把 overhead 压到了 15% 以内。

秘诀在于 PTX 指令和 Register File。
- Blackwell 有原生的 `cvt` 指令，可以直接把 FP16 打包成 packed FP4，也可以把 FP4 反量化回 FP16。
- Block size 是 16。16 个 FP4 值只要 8 bytes。两次量化的结果、反量化结果、MSE 累加器，全部可以塞进 GPU 的 Register File 里，不需要访问 Shared Memory 或 Global Memory。
- 所有的 double quantization 和 error 比较都在寄存器里闭环完成。

这意味着 4/6 纯粹是一个 software 层面的 algorithmic trick，完全兼容现有的 NVFP4 硬件格式，存储格式没变，依然是 FP4 + FP8 scale。

---

### 六、 实验结果的 Intuition

**Pre-training:**
NVIDIA 官方的 NVFP4 训练 recipe 其实很笨重，需要 Random Hadamard Transform (RHT)、Stochastic Rounding (SR)、甚至训练末期还要切回高精度 "healing"。4/6 把这些开销全省了，单纯靠更精确的量化，就让 pre-training loss 逼近了带 healing 的 NVFP4，距离 BF16 baseline 缩近了 22.3%。

**Post-Training Quantization (PTQ):**
4/6 和现有的 PTQ 方法是正交的。
- **AWQ + 4/6** 效果最好。AWQ 是通过 activation 权重调整来保护重要 weight，4/6 是优化 block 内部的 rounding。两者完美互补。
- **GPTQ + 4/6** 效果反而有时变差。因为 GPTQ 依赖 Hessian matrix 做二阶误差补偿，它的补偿是基于标准量化误差计算的。你中途换成了 4/6 的量化规则，GPTQ 预计算的补偿就不准了。Paper 里也提到，未来需要把 4/6 嵌入到 GPTQ 的 optimization loop 里才行。

---

### 七、 为什么 MXFP4 用不了这个方法？

这是个极其关键的 limitation。AMD 或其他硬件支持的 MXFP4 格式，它的 block scale 是 FP8 **E8M0**。
E8M0 只有 exponent，没有 mantissa。它只能表示 2 的幂次方：1, 2, 4, 8, 16...
你没法把一个 scale factor 乘以 1.5 倍（从 6 缩放到 4 需要 1.5 倍的 scale 变化）。所以 4/6 这个 trick 物理上就无法应用于 MXFP4。这是 NVFP4 (E4M3 scale) 独享的红利。

---

### 八、 总结你的 Intuition

这篇 paper 给我最大的启发是：**浮点量化的非均匀性既是天堂也是地狱**。

INT4 是均匀的，步长永远一样，所以它极度害怕 outliers，必须用 SmoothQuant 之类的方法去抹平分布。
FP4 引入了非均匀步长，小值密集，大值稀疏。它天然不怕 outliers（因为有 per-block scale 把 outlier 单独隔离），但它引入了新的敌人：**near-maximal values 的断层**。

4/6 告诉我们，对付这种断层，不需要发明新硬件，不需要加 bit，只要在 algorithm 层面“自适应地选择尺子”，就能把硬件的榨干到极限。

**Relevant Reference Links:**
- Four Over Six GitHub: https://github.com/mit-han-lab/fouroversix
- NVIDIA NVFP4 Pre-training Paper: https://arxiv.org/abs/2509.25149
- Microscaling (MXFP4) Formats: https://arxiv.org/abs/2310.10537
- AWQ Paper: https://arxiv.org/abs/2306.00978
- GPTQ Paper: https://arxiv.org/abs/2210.17323
- SmoothQuant Paper: https://arxiv.org/abs/2211.10438

---

# Four Over Six: NVFP4 量化精度提升方法深度解析

## 一、Paper 核心动机与 Intuition Building

这篇 paper 来自 MIT Han Lab（Jack Cook, Song Han 等），解决的是 NVIDIA Blackwell GPU 上 NVFP4 量化格式的精度瓶颈问题。核心 insight 非常 elegant：**FP4 浮点格式的非均匀步长导致 near-maximal values 产生巨大量化误差，而通过 adaptively 把某些 block 的 scale 从 6 改成 4，可以更精确地表示这些值**。

GitHub repo: https://github.com/mit-han-lab/fouroversix

---

## 二、FP4 E2M1 格式的几何结构

要理解 4/6 方法，必须先 build FP4 数值格式的 intuition。FP4 E2M1 总共只有 16 个值（含符号位），正数部分为：

```
FP4 正数值: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
```

**关键 observation**：步长是非均匀的
- 区间 [0, 2]: 步长 = 0.5
- 区间 [2, 4]: 步长 = 1.0
- 区间 [4, 6]: 步长 = 2.0

动态范围 = 最大值 / 最小正值 = 6 / 0.5 = **12**，相比 INT4 的 7/1 = 7 要大，这是 FP4 的优势。但是步长在 4 到 6 之间跳到 2，意味着**任何落在 4-6 之间的值只能 round 到 4 或 6，最大误差 1.0**。

---

## 三、NVFP4 Block-Scaled Quantization 详解

### 3.1 三层 scale 结构

NVFP4 采用三层 scaling 结构：

```
Tensor (FP32 α)  →  Block (FP8 E4M3 Δ_i, 16 values)  →  Value (FP4 E2M1)
```

### 3.2 公式逐项解析

**Tensor scale (Eq. 1):**

$$\alpha^{\mathrm{FP32}} = \frac{\max(|\mathbf{X}|)}{M^{\mathrm{FP4}} \times M^{\mathrm{FP8}}}$$

变量解释：
- $\mathbf{X}$: 高精度输入 tensor（BF16 或 FP32）
- $\alpha^{\mathrm{FP32}}$: tensor 级别 scale factor，存储在 FP32
- $M^{\mathrm{FP4}}$: FP4 可表示最大值，standard NVFP4 中 = 6
- $M^{\mathrm{FP8}}$: FP8 E4M3 可表示最大值，standard NVFP4 中 = 448

所以标准 NVFP4: $\alpha = \max(|\mathbf{X}|) / (6 \times 448) = \max(|\mathbf{X}|) / 2688$

**Block scale (Eq. 2):**

$$\Delta_i^{\mathrm{FP8}} = \frac{\max(|\mathbf{X}_{16i \ldots 16(i+1)}|)}{\alpha M^{\mathrm{FP4}}}$$

变量解释：
- $\Delta_i^{\mathrm{FP8}}$: 第 $i$ 个 block 的 scale factor，存储为 FP8 E4M3（这里引入 quantization error）
- $\mathbf{X}_{16i \ldots 16(i+1)}$: 第 $i$ 个 block 的 16 个连续值
- 分母 $\alpha M^{\mathrm{FP4}}$ 确保量化后值不超过 FP4 最大值

**Quantization function (Eq. 3):**

$$\bar{\mathbf{X}}^{\mathrm{FP4}} = \begin{cases} 
\frac{1}{2}\left\lceil \frac{2\mathbf{X}}{\alpha \Delta} \right\rfloor, & \left|\frac{\mathbf{X}}{\alpha \Delta}\right| < 2 \\
\left\lceil \frac{\mathbf{X}}{\alpha \Delta} \right\rfloor, & \left|\frac{\mathbf{X}}{\alpha \Delta}\right| < 4 \\
2\left\lceil \frac{\mathbf{X}}{2\alpha \Delta} \right\rfloor, & \left|\frac{\mathbf{X}}{\alpha \Delta}\right| \leq 6 
\end{cases}$$

这个 piecewise function 反映了 FP4 的非均匀步长：
- 第一段：归一化值在 [0, 2)，步长 0.5，所以乘以 2 后 round 再除以 2
- 第二段：归一化值在 [2, 4)，步长 1.0，直接 round
- 第三段：归一化值在 [4, 6]，步长 2.0，除以 2 后 round 再乘以 2

$\lceil \cdot \rfloor$ 是 rounding function（可以是 deterministic round-to-nearest 或 stochastic rounding）。

### 3.3 两个误差来源

1. **FP8 block-level scale factor 误差**: $\Delta_i$ 被 cast 到 E4M3，影响整个 block
2. **FP4 value 误差**: 单个值 round 到 8 个 FP4 值之一

Paper 的 Figure 2a 实验表明：**scale factor 误差对性能影响极小，而 FP4 value 误差是性能下降的全部原因**。当把 values 保持高精度时，模型性能完全恢复。

---

## 四、核心问题：Near-Maximal Values 的 Rounding Catastrophe

### 4.1 关键实验（Figure 2b）

在 Llama-3.1-8B 上做 simulated quantization：只对 magnitude ≥ $x$ 的 scaled values 量化到 FP4，观察 WikiText-2 PPL 变化：

| $x$ 阈值 | 含义 | 性能影响 |
|----------|------|---------|
| 0 | 全 BF16 | baseline |
| 1-4 | 量化小值 | PPL 稳定缓慢上升 |
| 5 | 量化 ~5 的值 | **PPL 急剧恶化** |
| 6 | 全 FP4 | 最大退化 |

**Intuition**: 值在 4.5 附近（scaled 后）无法被 FP4 表示（4 和 6 之间步长 2），rounding error 最大。同理 2.5 和 3.5 附近也有小 spike（步长 1）。

### 4.2 表格 1 的具体例子

原始值 $\mathbf{X} = [10, 20, 30, 40]$：

**Standard NVFP4 (M=6):**
- $\Delta = 40/6 \approx 6.67$，cast 到 FP8 E4M3 → 6.5
- Scaled values: $[1.54, 3.08, 4.62, 6.15]$
- Round to FP4: $[1.5, 3, 4, 6]$（注意 4.62 → 4，误差 0.62）
- Dequantized: $[9.75, 19.5, 26, 39]$
- MSE = $((10-9.75)^2 + (20-19.5)^2 + (30-26)^2 + (40-39)^2)/4 = (0.0625 + 0.25 + 16 + 1)/4 = 4.33$

**4/6 方法 (M=4):**
- $\Delta = 40/4 = 10$
- Scaled values: $[1, 2, 3, 4]$
- Round to FP4: $[1, 2, 3, 4]$（全部精确表示！）
- Dequantized: $[10, 20, 30, 40]$
- MSE = 0

这里 30/40 = 0.75，恰好是 FP4 中 3/4 = 0.75 能精确表示的比例。而 standard NVFP4 中 4/6 = 0.667 是 4 和 6 之间的"断层"。

---

## 五、Four Over Six 方法详解

### 5.1 核心思想

对每个 block，计算两种量化结果：
- $\bar{\mathbf{X}}^{(6)}$: scale 到 M=6（standard NVFP4）
- $\bar{\mathbf{X}}^{(4)}$: scale 到 M=4

然后选择 quantization error 更小的那个。

### 5.2 Scale Selection Rules

Paper 比较了三种选择规则（Table 4）：

1. **Abs-Max error**: $\max_i(|\bar{\mathbf{X}}_i^{(M)} - \mathbf{X}_i|)$ 最小者
2. **L1 norm**: $\sum_i |\bar{\mathbf{X}}_i^{(M)} - \mathbf{X}_i|$ 最小者
3. **MSE**: $\frac{1}{n}\sum_i (\bar{\mathbf{X}}_i^{(M)} - \mathbf{X}_i)^2$ 最小者

实验结论：**MSE 规则在大多数情况下最优**。

Table 4 关键数据（WikiText-2 PPL，Llama-3-8B）：
- RTN baseline: 8.43
- + 4/6 (MSE): **8.30** ← 最佳
- + 4/6 (L1): 8.33
- + 4/6 (Abs-Max): 8.36

### 5.3 Tensor Scale 的关键修改

标准 NVFP4: $M^{\mathrm{FP8}} = 448$，则 $\alpha = \max(|\mathbf{X}|) / (6 \times 448)$

**问题**: 如果某个 block 想用 M=4，需要 $\Delta = \max(\text{block}) / (4\alpha)$。对于包含 tensor 最大值的 block，这个 $\Delta$ 需要 = $448 \times (6/4) = 672$，但 E4M3 最大只能表示 448，**overflow**。

**4/6 的修改**: 把 $M^{\mathrm{FP8}}$ 从 448 降到 **256**：
$$\alpha = \frac{\max(|\mathbf{X}|)}{6 \times 256}$$

这样 M=4 的 block 需要 $\Delta = 256 \times (6/4) = 384$，在 E4M3 范围内（448）可精确表示。

**Trade-off**: tensor-level 动态范围从 $6 \times 448 = 2688$ 降到 $6 \times 256 = 1536$，但换来 block-level 的灵活性。实验表明这个 trade-off 是值得的。

### 5.4 为什么不能全部用 M=4？

Table 3 的实验：如果把所有 block 都 scale 到 4：

| Model | NVFP4 (M=6) | NVFP4 (M=4) |
|-------|-------------|-------------|
| Llama-3-1B | 14.27 | 14.75 (更差) |
| Llama-3-8B | 8.43 | 8.63 (更差) |
| Llama-3-70B | 4.00 | 4.48 (更差) |

**原因**: M=4 的动态范围是 4/0.5 = 8，而 M=6 是 6/0.5 = 12，少了 33%。对于有 outliers 的 block，M=6 更好。所以必须 **adaptive** 选择。

### 5.5 为什么只有 4 和 6？

- M=3: 丢失 FP4 值 {4, 6}，只剩 {0, 0.5, 1, 1.5, 2, 3}
- M=2: 丢失 {3, 4, 6}，只剩 {0, 0.5, 1, 1.5, 2}
- M=4 是能保留 {0.5, 1, 1.5, 2, 3, 4} 的最小 scale（6 个值）
- M=6 保留全部 8 个值

所以 {4, 6} 是仅有的两个有意义的候选。

---

## 六、实现细节：Blackwell GPU 上的高效 Kernel

### 6.1 PTX 指令

使用 `cvt` 指令族：
- `cvt.rn.fp4x2.f16`: 把 2 个 FP16 值 pack 成 FP4
- 反量化：FP4 → FP16（用于计算 error）

### 6.2 Register File 优化

Table 2 的伪代码需要计算：
1. $\bar{\mathbf{X}}^{(6)}$ 和 $\bar{\mathbf{X}}^{(4)}$（两组 quantized values）
2. $\mathbf{D}^{(6)}$ 和 $\mathbf{D}^{(4)}$（两组 dequantized values）
3. $E^{(6)}$ 和 $E^{(4)}$（两组 error）

如果这些数据在 global memory 或 shared memory 之间搬运，overhead 会很大。Paper 把所有中间结果保持在 **register file** 中，overhead < **15%**。

### 6.3 为什么 overhead 可以这么低？

关键在于 block size = 16 非常小：
- 16 个 FP4 值 = 8 bytes
- 2 组 = 16 bytes
- 加上 dequantized (FP16) = 16 × 2 × 2 = 64 bytes
- 总共 < 100 bytes，完全可以放进 register

---

## 七、Pre-Training 实验详解

### 7.1 模型架构

Nemotron 3 Nano 30B-A3B（hybrid Mamba-Transformer MoE）：
- 总参数: 30B
- Active 参数: 3B
- 52 blocks: 6 Self-Attention + 23 MoE + 23 Mamba-2
- Hidden dim: 2688
- GQA: 32 query heads, 2 KV heads
- MoE: 128 experts, 6 active + 2 shared, squared ReLU, expert hidden = 1856

### 7.2 训练配置

- 数据: 1T tokens（curated + synthetic）
- Optimizer: AdamW, $\beta_1 = 0.9, \beta_2 = 0.95$
- Weight decay: 0.1
- Gradient clipping: 1.0
- Sequence length: 8192
- Global batch size: 3072
- LR schedule: WSD, constant $10^{-3}$ → decay to $10^{-5}$ over last 20%
- Hardware: 384 × B200 180GB GPUs

### 7.3 NVFP4 Training Recipe (Figure 3)

```
FPROP:
  Activation (BF16) → Q(4/6) → NVFP4
  Weight (FP32) → Q(4/6) → NVFP4
  NVFP4 × NVFP4 → FP32 accumulate → BF16 output

DGRAD:
  Output Gradient (BF16) → Q(4/6) → NVFP4
  Weight (FP32) → Q(4/6) → NVFP4
  NVFP4 × NVFP4 → FP32 → BF16

WGRAD:
  Output Gradient (BF16) → RHT → Q(4/6) → NVFP4
  Activation (BF16) → RHT → Q(4/6) → NVFP4
  NVFP4 × NVFP4 → FP32 → BF16 → SR → Weight Gradient (FP32)
```

关键组件：
- **SR (Stochastic Rounding)**: 仅用于 gradient，保持无偏性
- **RHT (Random Hadamard Transform)**: 仅用于 WGRAD 的两个输入，分散 outliers
- **2D block quantization**: weight matrices 用 2D block 结构
- **High precision 保留**: Attention、output projection head、norm layers、non-linearities、Mamba-2 output projection (MXFP8)

### 7.4 结果

4/6 让 training loss 比 standard NVFP4 recipe **接近 BF16 22.3%**。

---

## 八、Post-Training Quantization 实验详解

### 8.1 主结果表 (Table 5)

WikiText-2 Word PPL（部分关键数据）：

| Method | Llama-3-1B | Llama-3-8B | Qwen-3-1.7B | Qwen-3-8B |
|--------|-----------|-----------|-------------|-----------|
| BF16 | 11.98 | 7.54 | 21.06 | 12.22 |
| RTN | 14.27 | 8.43 | 23.06 | 12.68 |
| RTN + 4/6 | 13.84 | 8.30 | 23.60 | 12.56 |
| GPTQ | 13.67 | 8.30 | 22.70 | 12.65 |
| GPTQ + 4/6 | 13.73 | 8.33 | 21.48 | 12.50 |
| AWQ | 13.67 | 8.24 | 21.67 | 12.57 |
| AWQ + 4/6 | **13.67** | **8.24** | **21.67** | **12.57** |
| SmoothQuant | 14.03 | 8.32 | 21.97 | 12.62 |
| SmoothQuant + 4/6 | 14.03 | 8.32 | 21.97 | 12.62 |

### 8.2 为什么 4/6 与 AWQ 配合最好？

**Intuition**:
- **AWQ** 是 pre-quantization transformation：通过 activation-aware scaling factor $s$ 调整 weight 和 activation 的分布，使重要 weight 落在 FP4 可精确表示的区间
- **4/6** 修改 quantization function 本身：让 block adaptively 选择更优的 scale
- 两者正交，可叠加

**为什么 GPTQ + 4/6 反而变差？**
- **GPTQ** 是 sequential column-wise 量化，依赖 Hessian-based error compensation
- GPTQ 在量化第 $i$ 列时，会根据前 $i-1$ 列的累积误差调整
- **4/6 改变了量化规则**，使得 GPTQ 预计算的 Hessian inverse 不再准确匹配实际量化误差
- 论文提到 "Modifying the GPTQ optimization process in a way that incorporates Four Over Six is likely to deliver performance improvements in future work"——即需要把 4/6 集成进 GPTQ 的 optimization loop，而非简单 post-hoc 替换

### 8.3 Downstream Tasks (Table 6, 7)

Llama-3-8B 平均准确率：
- BF16: 75.0
- RTN: 72.0
- RTN + 4/6: 72.2
- AWQ: 72.6
- AWQ + 4/6: **73.1** ← 最佳改善

---

## 九、Outliers 与 Block-Scaled Format 的深层关系

### 9.1 传统 INT4 的痛点

INT4 量化中，outliers 是大敌：
- 单个 outlier 会拉伸整个 tensor 的 scale
- 导致其他正常值被压缩到极少数 INT4 levels
- 需要 SmoothQuant、AWQ 等方法平滑 outliers

### 9.2 NVFP4 的范式转变

Block-scaled format (NVFP4, MXFP4) 通过 per-block scale 天然处理 outliers：
- Outlier 所在的 block 有自己的 scale，不影响其他 block
- **Outlier 本身几乎零误差**

所以 NVFP4 的主要误差来源转移到了 **near-maximal values**——即 block 内部第二大、第三大的值。4/6 正是针对这个新痛点。

### 9.3 Paper Section 5.1 的洞见

> "most of the quantization error in these formats comes from near-maximal values"

这意味着传统 quantization literature（针对 INT4）关注的 outlier mitigation 在 FP4 时代不再是主要矛盾。新的研究方向应该是 **intra-block value distribution optimization**。

---

## 十、Limitations 与未来方向

### 10.1 MXFP4 不兼容

MXFP4 用 **FP8 E8M0** 作为 scale factor：
- E8M0 只有 exponent，无 mantissa
- 相邻可表示值是 2 的倍数（如 1, 2, 4, 8, ...）
- 无法表示 $448 \times (6/4) = 672$ 或 $256 \times (6/4) = 384$（不是 2 的幂）

所以 4/6 无法应用于 MXFP4。这限制了 4/6 在 AMD 或其他硬件上的适用性。

### 10.2 精度增加后收益递减

> "the benefits fade quickly as the precision used to store values increases"

对于 FP6、FP8 等，步长更小，near-maximal error 问题不再严重，4/6 的收益消失。

### 10.3 未来方向猜想

基于这篇 paper 的思路，可能的延伸：
1. **Learned scale selection**: 用小网络预测每个 block 应该用 M=4 还是 M=6，避免 double quantization
2. **更多 scale 候选**: 如果 FP5 或 FP6 格式出现，可能有 M ∈ {3, 4, 5, 6} 的多候选
3. **与 rotation 方法结合**: SpinQuant、QuaRot 等 rotation 让分布更 uniform，可能减少需要 M=4 的 block 比例，进一步降低 overhead
4. **GPTQ 集成**: 把 4/6 的量化规则嵌入 GPTQ 的 Hessian compensation loop

---

## 十一、相关 Reference 与延伸阅读

### 核心 Paper
- **Four Over Six (本篇)**: https://github.com/mit-han-lab/fouroversix
- **NVFP4 Pretraining (NVIDIA)**: https://arxiv.org/abs/2509.25149
- **MXFP4 Microscaling Formats**: https://arxiv.org/abs/2310.10537
- **Quartet: Native FP4 Training**: https://arxiv.org/abs/2505.14669
- **FP4 All the Way**: https://arxiv.org/abs/2505.19115
- **Training LLMs with MXFP4**: https://arxiv.org/abs/2502.20586

### PTQ 方法
- **GPTQ**: https://arxiv.org/abs/2210.17323
- **AWQ**: https://arxiv.org/abs/2306.00978
- **SmoothQuant**: https://arxiv.org/abs/2211.10438
- **SpinQuant**: https://arxiv.org/abs/2405.16406
- **QuaRot**: https://arxiv.org/abs/2404.00456
- **SVDQuant**: https://arxiv.org/abs/2411.05007

### 硬件与格式
- **NVIDIA Blackwell Architecture**: https://www.nvidia.com/en-us/data-center/blackwell-architecture/
- **DeepSeek-V3 (FP8 training)**: https://arxiv.org/abs/2412.19437
- **Llama 4 (FP4 native)**: Meta AI 技术报告
- **INT vs FP Quantization Formats**: https://arxiv.org/abs/2510.25602

### 理论背景
- **A White Paper on Neural Network Quantization**: https://arxiv.org/abs/2106.08295
- **Deep Compression (Song Han)**: https://arxiv.org/abs/1510.00149
- **k-bit Inference Scaling Laws**: https://arxiv.org/abs/2212.09720

### Outlier 研究
- **StreamingLLM (Attention Sinks)**: https://arxiv.org/abs/2309.17453
- **Quantizable Transformers**: https://arxiv.org/abs/2306.12929
- **Systematic Outliers in LLMs**: https://arxiv.org/abs/2502.06415

---

## 十二、Build Intuition 的总结

4/6 方法的 elegant 之处在于：**它没有改变硬件格式（仍然是 NVFP4），没有增加存储（仍然是 FP4 value + FP8 scale），仅仅是修改了 quantization algorithm 本身**。这个修改的 cost 是 < 15% 的 kernel overhead，但换来 22.3% 的 pre-training loss 改善和广泛的 PTQ 提升。

更深层的 intuition 是：**浮点量化的非均匀性既是优势也是诅咒**。优势在于小值密集表示（适合 weight/activation 的尖峰分布），诅咒在于大值区间步长过大。4/6 通过 adaptive scale selection 在两者间找到 per-block 的最优平衡。

这让我想到一个更 general 的 principle：**量化算法的设计应该 match 数据的局部分布特征，而非假设全局均匀**。Block-scaled format 已经迈出了第一步（per-block scale），4/6 迈出了第二步（per-block scale range selection）。未来或许会有 per-block step size optimization、per-block rounding mode 等更细粒度的 adaptivity。

---

*附注：Nemotron 3 Nano 的 hybrid Mamba-Transformer 架构很值得关注，paper 引用 [20] [23] 指出 23 个 Mamba-2 block + 23 个 MoE block + 6 个 Attention block 的混合设计。Mamba-2 在长序列上的线性复杂度与 Attention 的 expressivity 结合，是当前 architecture search 的前沿方向之一。*
