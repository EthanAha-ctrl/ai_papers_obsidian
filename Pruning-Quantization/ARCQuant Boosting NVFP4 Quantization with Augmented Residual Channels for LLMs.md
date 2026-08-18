---
source_pdf: ARCQuant Boosting NVFP4 Quantization with Augmented Residual Channels
  for LLMs.pdf
paper_sha256: 4d98102573ba0073018b87ee1b52a57877511c0d30766c921b7d3ec8960f7d37
processed_at: '2026-08-18T01:12:10-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ARCQuant 人话版：把 NVFP4 的精度坑填平

## 先聊一个最朴素的事实

你手里有个 LLM，想用 W4A4 跑得快，NVIDIA Blackwell 给了你 NVFP4 这个硬件原生 format。你直接把 weight 和 activation 都 round 到 NVFP4，结果模型变傻了——PPL 从 6.24 涨到 6.95，MMLU 从 65 掉到 61。差距是明显的。

那怎么办？传统上有三招，但在 NVFP4 上全不好使。ARCQuant 提出第四招：**把 outlier channel 的量化残差 quantize 一遍，塞回 GEMM 的 K dimension 里**。

就这么个事。

## 三招为什么不行

### 第一招：Hadamard rotation（QuaRot、FlatQuant 用的）

核心想法：activation 里有些 channel 数值特别大（outlier），把整个 tensor 的 dynamic range 拉爆。Hadamard 是一个正交矩阵，乘上去相当于把所有 channel 的能量平均摊一摊，outlier 就不显眼了。

对 per-tensor quantization，这招好用。但 NVFP4 是 per-block（block size = 16）quantization。Hadamard 是 linear combination，把 outlier 的能量**等比例挤进了**之前没 outlier 的 block。结果：
- Global peak 下降 ✓
- Local block 内部 dynamic range 暴涨 ✗
- NVFP4 靠 small block 把 outlier "隔离"起来的好处被自己毁掉

Table 2 实测：Llama 3.1-8B 上 NVFP4+RTN PPL=6.95，NVFP4+QuaRot 反而 PPL=6.99。QuaRot 把事情搞得更糟。这就是 paper 的 motivation。

参考: https://arxiv.org/abs/2404.02558 ; https://arxiv.org/abs/2502.14825

### 第二招：Smoothing（SmoothQuant 用的）

核心想法：activation outlier 太猛，weight 很平。把 activation 的难度"转嫁"到 weight 上，两边 dynamic range 拉平。INT8 时代管用。

但 W4A4 下 weight 只有 4 bit，capacity 太小，转嫁过去直接把 weight quantization error 搞爆。Table 2 实测 SmoothQuant 相对 RTN 提升几乎可以忽略（PPL 6.95→6.92）。

参考: https://arxiv.org/abs/2211.10438

### 第三招：Mixed precision（Atom、MicroMix 用的）

核心想法：sensitive channel 用 INT8/FP8 保留，剩下用 INT4。理论上没问题，硬件上撞墙。

NVFP4 block size = 16，但 MXFP6 / MXFP8 block size = 32。Tensor Core 的 MMA instruction 要求两边 group size 一致，不然就走不了 hardware pipeline，throughput 大幅降级。所以 Atom 那套在 NVFP4 上没办法用 hardware-native 路径。

参考: https://arxiv.org/abs/2310.19102 ; https://arxiv.org/abs/2508.02343

## ARCQuant 的招：augment K dimension

### 核心思路

我们回头看一下 GEMM 在算什么：
$$Y = X W^\top$$

X 形状 (N, K_in)，W 形状 (M, K_in)。GEMM 沿 K_in 这一维做 reduction。

ARCQuant 的关键洞察：**reduction 是线性的**。如果我们能把 X 拆成 "主项 + 残差"，weight 拆成 "主项 + 对应残差的 weight"，那么 GEMM 一次就能算完两块。

具体怎么拆？

1. 找出 X 里 magnitude 最大的 S 个 channel，记为 X_o
2. 主量化：整个 X 按 NVFP4 量化，得到 Q(X) 和 scale s_X
3. 残差：对 outlier channel 部分算 residual R_o = X_o - s_{X_o} · Q_{X_o}
4. 二次量化：把 R_o 再按 NVFP4 量化一次，得到 Q(R_o)
5. Concat：沿 K dimension 拼接，Q_{X_aug} = [Q_X | Q_{R_o}]，长度从 K_in 变成 K_in + S
6. Weight 端：对应位置**复制** W 的 outlier 行（GEMM 里 W 转置后，X 的第 i 列对应 W 的第 i 行），Q_{W_aug} = [Q_W | Q_{W_o}]

GEMM 一次计算：
$$Y \approx s_{X_{aug}} \cdot Q_{X_{aug}} \cdot (s_{W_{aug}} \cdot Q_{W_{aug}})^\top$$

由于 accumulation 是线性的，主项 Q(X)·Q(W)^T 和残差项 Q(R_o)·Q(W_o)^T 在 FP32 accumulator 里自动相加。

**关键好处**：用标准 CUTLASS NVFP4 GEMM kernel，一行都不用改。K dimension 从 K_in 变成 K_in + S，kernel 本身无感。

### 数学等价性（公式 2）

$$Y \approx Q(X)Q(W)^\top + Q(R_o)Q(W_o)^\top = s_{X_{aug}} \cdot Q_{X_{aug}} (s_{W_{aug}} \cdot Q_{W_{aug}})^\top$$

变量解释：
- $Q(X), Q(W)$: 主项量化值
- $Q(R_o), Q(W_o)$: 残差量化值 / 对应的 weight
- $s_{X_{aug}} = [s_X | s_{R_o}]$: 合并 scale，前 K_in 个是主 scale，后 S 个是残差 scale
- $s_{W_{aug}} = [s_W | s_{W_o}]$: weight 端合并 scale，后 S 个对应复制的 outlier 行

整个 trick 的核心就是利用 GEMM 的线性性质把 residual compensation "免费" 嵌进去。

## S 怎么选

阈值 τ = 2^-3 · M，M 是 layer-wise max。

直觉：FP8 (E5M2) 有 5 bit exponent，NVFP4 (E2M1) 只有 2 bit exponent，差 3 bit。低于 τ 的 channel 在 NVFP4 下精度与 FP8 接近，没必要补偿；高于 τ 的 channel 才进入 top-S。

Figure 7 显示 Qwen2.5-7B 各 layer S 值在几十到几百之间波动。自适应分配。

## 为什么 dual-stage 4-bit 能匹敌 8-bit

这是论文最妙的地方。

### 基本量

- NVFP4 (E2M1) 的 machine epsilon: $\epsilon_4 = 2^{-2} = 0.25$（1 ULP 相对值）
- MXFP8 (E4M3) 的 machine epsilon: $\epsilon_8 = 2^{-4} = 0.0625$
- 关键恒等式：$\epsilon_4^2 = (2^{-2})^2 = 2^{-4} = \epsilon_8$

这个 $\epsilon_4^2 = \epsilon_8$ 是 ARCQuant 的理论基石：4-bit 量化误差再量化一次，相对量级平方一次，刚好等于 8-bit 一次量化的误差量级。

### Worst-case error bound 推导

#### MXFP8 的情况

MXFP8 用 E8M0 scale（纯 exponent，power-of-two），scale 是 2 的幂。当 max 落在两个 power-of-two 之间时，scale 只能取较大的那个，alignment factor $\alpha_{mx} \in [1, 2)$。

Worst-case error:
$$B_{mx} = \alpha_{mx} M \epsilon_8 < 2 M \epsilon_8$$

变量解释：
- $M$: tensor 的 dynamic range
- $\alpha_{mx}$: scale alignment overhead，因 E8M0 只能选 2 的幂，最坏情况是 max 的一半被舍到 1 倍 scale，alignment 因子接近 2
- $\epsilon_8$: MXFP8 的 machine epsilon

#### ARCQuant dual-stage 的情况

**Stage 1**: 量化 x，scale $s_1 = \alpha_1 M$，残差 r = x - Q(x) 的 bound:
$$\|r\|_\infty \leq \alpha_1 M \epsilon_4$$

**Stage 2**: 量化残差 r，scale $s_2 = \alpha_2 \|r\|_\infty$，二级残差 e_arc = r - Q(r) 的 bound:
$$|e_{arc}| \leq s_2 \epsilon_4 \leq (\alpha_2 \alpha_1 M \epsilon_4) \epsilon_4 = (\alpha_1 \alpha_2) M \epsilon_8 = B_{arc}$$

变量解释：
- $\alpha_1, \alpha_2$: stage 1 和 stage 2 的 alignment factor
- $M$: tensor 的 dynamic range
- $\epsilon_4 = 0.25$: NVFP4 的 machine epsilon
- $\epsilon_8 = 0.0625$: MXFP8 的 machine epsilon

代入 $\epsilon_4^2 = \epsilon_8$，dual-stage 4-bit 的 worst-case error bound 与 single-stage 8-bit 等价。

### Alignment factor 谁更紧

- MXFP8: E8M0 是 pure exponent，power-of-two spacing，$\sup \alpha_{mx} = 2$
- NVFP4: E4M3 scale 有 3 bit mantissa，步长 2^-3，$\sup \alpha_1 = 1.125$（scale 与 max 最多差 12.5%）
- Dual-stage: $\sup \alpha_1 \alpha_2 = 1.125^2 \approx 1.266$

$1.266 < 2$，所以 ARCQuant 的 worst-case bound 反而比 MXFP8 更紧！

这背后的 insight：NVFP4 用 E4M3 scale 而不是 E8M0 scale，虽然单看 block scale 动态范围小（所以才需要额外的 FP32 per-tensor scale），但因为 mantissa 提供 fine-grained spacing，alignment loss 反而更小。dual-stage 又把这个优势"平方"了一次。

参考: NVIDIA Blackwell architecture brief https://www.nvidia.com/en-us/data-center/blackwell-architecture/ ; cuDNN BlockScaling op https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.14.0/operations/BlockScaling.html

## Kernel 实现细节

### Fused Quantization Kernel

Online 部分四件事必须 fused 在一起，不然每次都 round-trip global memory：
1. Channel Reordering（按 offline 算好的 index 重排）
2. RMSNorm
3. Primary Quantization
4. Residual Quantization

输出直接是 NVFP4 encoded tensor，喂给 GEMM kernel。

### Interleaved Channel Layout（Appendix D 的 trick）

公式 (2) 写 $Q_{X_{aug}} = [Q_X | Q_{R_o}]$ 是 logical concat。但物理上如果直接这样布局，GEMM 读 memory 时会有 strided access，latency 爆炸。

实际 layout：把 S 个 outlier channel 按 16 个一组分块（NVFP4 block size），然后 local interleave——每 16-channel primary block 紧跟着 16-channel residual block。

物理内存示意：
```
[P_0..P_15 | R_0..R_15 | P_16..P_31 | R_16..R_31 | ...]
```

好处：
- Weight 端 offline 一次性按这个 interleave 模式排好
- Fused kernel 在 on-chip register 里同时算 P 和 R，coalesced write
- 标准 GEMM kernel 不用改就跑得通，因为 GEMM 不知道也不关心这是 interleave 的，它就当普通 tensor 处理

### Latency 实测

Figure 8(b) Qwen2.5-7B 端到端 prefill breakdown：
- 总 latency 增加仅 4.9%（vs. 无补偿 NVFP4）
- Fused Quantization Kernel 本身 cost 极小

Figure 8(a) GEMM latency 与 S 严格线性关系，S ≤ 512 时几乎平直。

Figure 6 端到端 prefill：
- Qwen2.5-7B: PRO 6000 上 2.0x-2.5x vs FP16
- Llama 3.1-8B: RTX 5090 上 3.5x
- Memory 降 1.5x-2.8x

Table 4 vLLM decode throughput（Qwen2.5-7B，RTX 5090，batch=8，gen_len=128）：

| Method | SeqLen 1024 Total / Decode | SeqLen 2048 Total / Decode |
|---|---|---|
| FP16 | 10756 / 1195 | 10203 / 600 |
| FP8 | 16699 / 1855 | 16857 / 992 |
| ARCQuant | 21077 / 2342 | 21237 / 1249 |

Decode 部分 1.96x-2.08x speedup vs FP16，且超过 FP8。这点很有意思：说明 NVFP4 的 hardware path 在 decode 上比 FP8 更高效，可能跟 NVFP4 的 bandwidth 利用率更高有关。

## 实验结果的人话版

### Table 1：与所有 W4A4 方法比较

Llama 3.1-8B：

| Method | PPL↓ | MMLU↑ |
|---|---|---|
| FP16 | 6.24 | 65.15 |
| W4A8+RTN | 7.07 | 61.08 |
| FlatQuant | 6.95 | 61.33 |
| MicroMix | 7.35 | 60.17 |
| Atom | 7.52 | 59.27 |
| **ARCQuant** | **6.87** | **62.61** |

ARCQuant PPL 比 W4A8 RTN 还低（6.87 < 7.07）。MMLU 比 W4A8 RTN 还高（62.61 > 61.08）。这是 "dual-stage 4-bit 等于 8-bit" 理论预测的完美兑现。

Qwen2.5-32B 上 ARCQuant 几乎无损压缩，PPL 5.41 vs FP16 5.02，差距很小。

### Table 2：NVFP4 内部比较

Llama 3.1-8B：

| Method | PPL↓ | MMLU↑ |
|---|---|---|
| NVFP4+RTN | 6.95 | 61.64 |
| NVFP4+SmoothQuant | 6.92 | 61.76 |
| NVFP4+QuaRot | 6.99 | 61.73 |
| **NVFP4+ARCQuant** | **6.87** | **62.61** |

所有传统方法在 NVFP4 上几乎没提升甚至倒退，只有 ARCQuant 显著突破。

### Table 3：Code generation（Qwen2.5-Coder-7B-Instruct）

| Method | HE | HE+ | Mbpp | Mbpp+ |
|---|---|---|---|---|
| FP16 | 84.1 | 79.9 | 80.4 | 67.2 |
| Atom | 80.5 | 76.2 | 74.5 | 63.2 |
| **ARCQuant** | **86.0** | 79.3 | 79.9 | **68.3** |

ARCQuant 在 HumanEval 上居然超过 FP16 baseline 86.0 vs 84.1。这点比较反直觉，可能是 calibration set 的微小扰动让 code generation 上某些 outlier 处理得更好。但 Mbpp 略低于 FP16，整体来说 ARCQuant 在 code 任务上接近无损。

### Table 6：Calibration robustness

Llama 3.1-8B 上用 C4 / HumanEval / WikiText2 三个不同 calibration set，PPL 波动 <0.03，accuracy 波动 <0.03%。说明 outlier 结构在 LLM 里非常稳定，不依赖 calibration data。

### Table 9：扩展到大模型和 MoE

Llama 3.1-70B：

| Method | Avg zero-shot | PPL | MMLU |
|---|---|---|---|
| FP16 | 78.20 | 2.81 | 78.57 |
| NVFP4+RTN | 75.43 | 3.85 | 76.11 |
| ARCQuant | 76.08 | 3.62 | 76.61 |

70B 上 ARCQuant 依然显著好于 RTN baseline。

Mixtral 8x7B-Instruct（MoE）：

| Method | Avg zero-shot | PPL | MMLU |
|---|---|---|---|
| FP16 | 78.46 | 4.14 | 70.30 |
| NVFP4+RTN | 77.63 | 4.41 | 68.54 |
| ARCQuant | 78.03 | 4.38 | 68.71 |

MoE 上 ARCQuant 也有提升，证明方法在 sparse 架构上 generalizable。

## 联想与延伸

### 与 SVDQuant 的对比

SVDQuant（Li et al. 2024）也是处理 4-bit outlier，但用 low-rank FP16 branch 吸收 outliers。ARCQuant 区别：纯 4-bit path，不引入额外 precision branch。

直觉上 SVDQuant 在 diffusion model 上效果好，因为 diffusion 的 outlier 结构与 LLM 不一样；ARCQuant 在 LLM 上更 hardware-friendly。

参考: https://arxiv.org/abs/2411.05007

### 与 QuIP# 的对比

QuIP# 用 incoherence preprocessing + lattice codebook 在 weight-only 场景下做 W4。ARCQuant 走另一条路：不动 weight format，靠 activation-side augmentation。两者思路完全不同，但目标都是榨干 4-bit 的 representation power。

参考: https://arxiv.org/abs/2404.14001

### Sub-4-bit 的可能扩展

Limitations 提到 "future work will extend to sub-4-bit formats"。理论推演：

- INT2 / 1.58-bit (BitNet) 的 $\epsilon_2 \approx 2^{-1} = 0.5$
- $\epsilon_2^2 = 0.25 = \epsilon_4$，所以 2-bit dual-stage 理论上逼近 4-bit

这个方向非常诱人。如果 NVFP2 或类似 format 出现，ARCQuant 的 dual-stage 思路应该可以无缝迁移。

参考: BitNet https://arxiv.org/abs/2310.11453

### 与 BRQ (Block Rotation) 的潜在竞争

BRQ（Shao et al. 2025）提出 block-local rotation，声称可以绕开 ARCQuant motivation 里指出的 Hadamard global rotation 问题。这是 ARCQuant 的直接竞争者。论文 Section 3 引用了 BRQ 但没正面比较。

如果 BRQ 真的解决了 local dynamic range 问题，那 QuaRot 一类 rotation 方法在 NVFP4 上可能复活，与 ARCQuant 形成两条路线竞争。

参考: https://arxiv.org/abs/2511.04214

### vLLM 整合

Table 4 已经证实 ARCQuant 已 vLLM-integrated。Table 10 给了 RTX 5090 / PRO 6000 上各种 batch / seqlen 组合的详细 latency 和 memory。这种 hardware-aware co-design 的思路很像 FlashAttention / FlashDecoding 早期的整合方式。

参考: FlashAttention https://arxiv.org/abs/2205.14135 ; vLLM https://arxiv.org/abs/2309.06180

## 局限性（我自己观察）

### Static outlier index 的风险

Channel reorder index 是 offline 定的。如果 inference 时 activation 分布漂移（OOD prompt、distribution shift），augmented channels 可能不是真正的 outliers，补偿失效。

但 Table 6 的 calibration robustness 实验显示换 calibration set 影响 <0.03 PPL，说明 outlier 结构在 LLM 内在很稳定。这是个 empirical 好消息，但理论上仍是隐患。

### Weight 端没用 GPTQ / AWQ

ARCQuant 在 weight 端只用 RTN。Atom / FlatQuant 在 weight 端有 Hessian-based 优化，理论上 weight 端还有 0.5-1 PPL 的提升空间没榨干。

Limitations 部分提到 ARCQuant 与 GPTQ/AWQ 兼容，但实际实现没做。这是未来一个明显的改进方向。

### 没碰 KV cache quantization

W4A4 是 matmul 端，KV cache 还在 FP8/FP16。如果 ARCQuant 思路扩展到 KV cache，端到端 inference 的 memory 节省会更显著。

参考: KIVI (KV cache INT4) https://arxiv.org/abs/2402.02750

### S 是 layer-wise fixed

是否可以 per-token dynamic 选 S？理论上更精细，但 hardware 实现会复杂很多。在线 dynamic S 会导致 GEMM 的 K dimension 在 runtime 变化，需要 dynamic shape GEMM，性能可能反而下降。

## 最后的 takeaway

ARCQuant 给我们几个 broader lesson：

1. **Hardware constraint 倒逼 algorithmic innovation**：NVFP4 的 block-size=16 + MMA 同 group-size 要求，逼出了"统一 4-bit format 下做误差补偿"这个非常 specific 的 research question
2. **Linear accumulator 是 free lunch**：GEMM 内部 K dimension 的线性求和性质，让 residual compensation "免费"嵌入到主计算中
3. **Bit-level math matters**：$\epsilon_4^2 = \epsilon_8$ 这种 floating-point format 层面的恒等式，催生了 dual-stage 这种 elegant 设计
4. **Format engineering 是新战场**：NVFP4 vs MXFP4 不仅是 block size 不同，scale format (E4M3 vs E8M0) 也不一样，alignment factor 性质不同。未来 hardware format 设计会越来越深地影响 quantization algorithm 设计

GitHub: https://github.com/actypedef/ARCQuant

## 一句话总结

**ARCQuant = 把 outlier channel 的量化残差再 quantize 一遍 + 沿 K dimension 拼回原矩阵 + 用标准 NVFP4 GEMM 一次算完，靠 $\epsilon_4^2 = \epsilon_8$ 让 4-bit dual-stage 的理论精度匹敌 8-bit single-stage，同时完全 hardware-native。**

希望这版"人话"能帮你在脑子里把 ARCQuant 的 mental model 搭起来。还想 drill deeper 到哪个点（比如 fused kernel 的 CUDA 实现细节、$\epsilon$ 推导的更严格形式、或者与某个 baseline 的具体对比），告诉我就行。

---

# ARCQuant: Build Intuition for NVFP4 Residual Augmentation

## 1. 大背景：为什么 NVFP4 是个 "硬骨头"

NVIDIA Blackwell 把 W4A4 推到了硬件原生支持，但 format 上有微妙的设计选择。先理清几个 format 之间的层次关系，这是理解 ARCQuant 的前提：

| Format | Element | Block Size (g) | Scale Type | Tensor Scale |
|---|---|---|---|---|
| MXFP8 | E4M3 / E5M2 | 32 | E8M0 (only exponent) | 无 |
| MXFP4 | E2M1 | 32 | E8M0 | 无 |
| NVFP4 | E2M1 | **16** | **E4M3** | **FP32 (额外)** |

关键差异：
- MX 系列用 E8M0 的 power-of-two scale，alignment 因子 α ∈ [1, 2)，很简单
- NVFP4 把 block size 减半到 16，用 E4M3 scale（带 3-bit mantissa，步长 2^-3），由于 E4M3 dynamic range 太窄，又额外加了一个 FP32 per-tensor scale 形成三层 hierarchy: **Element → Block Scale → Tensor Scale**

这种 finer block isolation 是 ARCQuant 能发挥的前提——它需要靠小 block 把 outlier "圈起来"。

参考资料: NVIDIA Blackwell technical brief (https://www.nvidia.com/en-us/data-center/blackwell-architecture/) ; OCP Microscaling Formats spec https://arxiv.org/abs/2310.10537 ; cuDNN BlockScaling op https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.14.0/operations/BlockScaling.html

## 2. 痛点诊断：三类现有 PTQ strategy 全部失效

Karpathy 你看 paper 时会注意到 Section 3.1 的 motivation 部分是核心。这里论文用实验和理论两种方式同时打三拳：

### 2.1 Rotation-based methods (QuaRot, FlatQuant, Hadamard) 的反例

直觉：Hadamard 是一个 orthogonal 矩阵，作用是把能量均匀摊到所有维度上。对 per-tensor 量化是好的（global max 下降），但对 per-block NVFP4 是灾难。

为什么？看 Figure 2: outlier channel 的 magnitude 通过 linear combination 被 "挤"到了每个之前低 magnitude 的 block 里。结果：
- Global peak 下降 ✓
- Local block dynamic range 暴涨 ✗
- NVFP4 的 isolation 好处被 self-sabotage 掉

Table 2 的数据印证：Llama 3.1-8B 上 NVFP4+RTN PPL = 6.95，NVFP4+QuaRot 反而 PPL = 6.99（regression）。这对靠 rotation 吃饭的 community 是很打脸的结果。

参考: QuaRot https://arxiv.org/abs/2404.02558 ; BRQ (Block Rotation) https://arxiv.org/abs/2511.04214

### 2.2 Smoothing (SmoothQuant) 在 4-bit 下 capacity 不足

SmoothQuant 把 activation 的 outlier 通过 sqrt 形式迁移到 weight 上，在 INT8 时代好用。但在 W4A4 下 weight 只有 4 bit，迁移过来的 magnitude 直接把 weight 量化精度搞爆。Table 2 显示 SmoothQuant 相比 RTN 提升极小（Llama 上 PPL 6.95→6.92）。

### 2.3 Mixed-precision (Atom, MicroMix) 与硬件不兼容

这是最关键的硬件细节。Atom 把 sensitive channel 用 INT8/FP16 保留，其他用 INT4。但 NVFP4 block g=16，而 MXFP6/MXFP8 block g=32。NVIDIA Tensor Core 的 MMA (Matrix Multiply-Accumulate) instruction 要求 operand 的 group size 一致——heterogeneous group size 直接走不了 hardware pipeline，得 fallback 到复杂 kernel logic，throughput 大幅降级。

参考: Atom https://arxiv.org/abs/2310.19102 ; MicroMix https://arxiv.org/abs/2508.02343 ; FGMP https://arxiv.org/abs/2504.14152

## 3. ARCQuant 的核心直觉：把残差"塞回" K dimension

这里 Karpathy 你应该能联想到几件事：

1. **ResNet 思想**：x → Q(x) + Q(x - Q(x))，二级 quantization 把 primary stage "rounded 掉" 的部分重新捞回来
2. **视频/语音 coding 里的 prediction + residual**：IDR 帧之后是 P/B 帧只 encode 残差
3. **Hadamard 不能用，那直接做 channel-level decomposition 即可**

### 3.1 数学形式

原问题：Y = XW^T，X ∈ R^(N×K_in)，W ∈ R^(M×K_in)

ARCQuant 做 channel reorder 后：
- 把 X 中 magnitude 最大的 S 个 channels 挑出来记作 X_o ∈ R^(N×S)
- 剩下的部分还是叫 X，按 NVFP4 block-wise 量化得 Q(X) 和 scale s_X
- Residual: R_o = X_o - s_{X_o} · Q_{X_o}（这部分是 primary stage 的量化残差）
- 对 R_o 再做一次 NVFP4 量化得 Q(R_o)

然后**沿 K dimension 拼接**：
$$Q_{X_{aug}} = [Q_X \mid Q_{R_o}] \in \mathbb{R}^{N \times (K_{in} + S)}, \quad s_{X_{aug}} = [s_X \mid s_{R_o}]$$

Weight 端则不是计算残差，而是**复制对应 outlier 列**：
$$Q_{W_{aug}} = [Q_W \mid Q_{W_o}] \in \mathbb{R}^{M \times (K_{in} + S)}$$

注意这里 W_o 是 weight 的 outlier 行（不是 outlier 列），因为 GEMM 是 Y = X W^T，W 转置后第 i 列对应 X 的第 i 列。

这样 GEMM 一次性计算：
$$Y \approx Q(X) Q(W)^T + Q(R_o) Q(W_o)^T = s_{X_{aug}} \cdot Q_{X_{aug}} \cdot (s_{W_{aug}} \cdot Q_{W_{aug}})^T$$

公式 (2) 的变量含义：
- $s_{X_{aug}}, s_{W_{aug}}$: 合并后的 scale vector，前 $K_{in}$ 个是主量化 scale，后 $S$ 个是 residual scale (X 端) 或 duplicate scale (W 端)
- $Q_{X_{aug}}, Q_{W_{aug}}$: 合并后的 NVFP4 encoded tensor
- 整个等式的核心：由于 GEMM reduction 是线性的，主项和残差项自动在 accumulator 中相加

这个 trick 的妙处在于：你不需要写新的 GEMM kernel！标准 CUTLASS NVFP4 GEMM 直接用，K dimension 从 $K_{in}$ 变成 $K_{in} + S$。在线性 accumulator 的工作机制下，主项和 residual 项自动求和进入高精度 FP32 accumulator。

### 3.2 选哪些 channel？S 怎么定？

Threshold 设计：$\tau = 2^{-3} M$，其中 $M$ 是 layer-wise max。

直觉：参考 E5M2 (5 exponent bits) vs E2M1 (2 exponent bits)，两者 exponent 宽度差 3 bits，所以 $2^{-3}$ 这个 threshold 划出了一个 "信息论"边界——低于此 threshold 的 channel 在 NVFP4 下精度与参考 FP8 相当，不需要补偿；高于此 threshold 的 channel 才进入 top-S 集合。

Figure 7 显示 Qwen2.5-7B 各 layer 的 S 数量，可以从几十一路波动到几百，自适应。

## 4. Error Bound Analysis：为什么 dual-stage 等效 MXFP8？

这是这篇 paper 里最 elegant 的部分，Karpathy 你应该会觉得有意思。

### 4.1 Preliminaries

定义：
- $M$: 当前 tensor 的 dynamic range
- $\epsilon$: 该 format 的 "machine epsilon"，即 1 ULP 相对 value
- $\alpha$: scale alignment overhead，$\alpha = s/M \geq 1$（scale 不可能小于 max，否则会饱和；但 scale 可能比 max 大，导致 alignment loss）

具体 format 的 $\epsilon$：
- NVFP4: E2M1 元素，1 bit sign + 2 bit exponent + 1 bit mantissa，$\epsilon_4 = 2^{-2} = 0.25$
- MXFP8: E4M3 元素，$\epsilon_8 = 2^{-4} = 0.0625$
- 关键恒等式：$\epsilon_4^2 = (2^{-2})^2 = 2^{-4} = \epsilon_8$

这个 $\epsilon_4^2 = \epsilon_8$ 是 ARCQuant 的理论基石。直觉上：4-bit 的 quantization error 再做一次 4-bit quantization，error 的相对量级会平方一次，刚好等于 8-bit 的一次量化误差量级。这是 "双 4-bit 等于 8-bit" 的 information-theoretic argument。

### 4.2 MXFP8 的 worst-case error bound

MXFP8 用 E8M0 scale（纯 exponent，power-of-two），所以 scale 是 2 的幂。这意味着 scale 不可能完美贴合 max，只能取最接近的 2 的幂，alignment 因子 $\alpha_{mx} \in [1, 2)$。

Worst-case bound:
$$B_{mx} = \alpha_{mx} M \epsilon_8 < 2M\epsilon_8 \tag{3}$$

变量下标：mx 表示 MX format。

### 4.3 ARCQuant dual-stage 的 worst-case bound

Stage 1: 量化 $x$ 用 scale $s_1 = \alpha_1 M$，残差 $r = x - Q(x)$ 的无穷范数被 bound：
$$\|r\|_\infty \leq \alpha_1 M \epsilon_4$$

Stage 2: 量化残差 $r$ 用 scale $s_2 = \alpha_2 \|r\|_\infty$，第二级残差 $e_{arc} = r - Q(r)$ 的 bound：
$$|e_{arc}| \leq s_2 \epsilon_4 \leq (\alpha_2 \alpha_1 M \epsilon_4) \epsilon_4 = (\alpha_1 \alpha_2) M \epsilon_8 = B_{arc} \tag{4}$$

这里下标 1, 2 分别指 stage 1 和 stage 2 的 alignment factor；arc 指 ARCQuant。

### 4.4 关键比较：alignment factor

为什么 ARCQuant 反而更紧？

- MXFP8 用 E8M0 (pure exponent)，power-of-two spacing → $\sup \alpha_{mx} = 2$（取上界时 scale 是 max 的一半）
- NVFP4 用 E4M3 (3 bit mantissa)，步长 $2^{-3}$ → $\sup \alpha_1 = 1.125$（最近 scale 距离 max 最多差 12.5%）
- 双 stage NVFP4: $\sup \alpha_1 \alpha_2 = 1.125^2 \approx 1.266$

因为 $1.266 < 2$，所以 ARCQuant 的 worst-case error bound 比 MXFP8 更紧！这是 paper 里很妙的一笔——通过 mantissa-coded scale 的细粒度 + dual-stage 的 $\epsilon_4^2$ 性质，4-bit + 4-bit 在 outlier channel 上精度反超 8-bit。

当然实际效果看 Table 1，ARCQuant 的 PPL 介于 W4A8 RTN 和 FP16 之间，符合"接近但不完全等于"的理论预期。整篇论文的实验数据列得相当干净，你可以看 Table 1 全集：

| | PPL↓ | MMLU↑ |
|---|---|---|
| FP16 | 6.24 | 65.15 |
| W4A8+RTN | 7.07 | 61.08 |
| FlatQuant (W4A4) | 6.95 | 61.33 |
| MicroMix (W4A4) | 7.35 | 60.17 |
| Atom (W4A4) | 7.52 | 59.27 |
| **ARCQuant (W4A4)** | **6.87** | **62.61** |

ARCQuant 在 PPL 上居然比 W4A8 RTN 还要好（6.87 < 7.07），这与理论分析一致。

## 5. Kernel Design：把动态 residual 焊进 GEMM

这里 Karpathy 你应该能联想到 vLLM / TensorRT-LLM 的 kernel fusion 思路。

### 5.1 Fused Quantization Kernel

Online 部分（activation 端）有四步必须 fused 在一起：
1. Channel Reordering（按 offline 算好的 index 重排）
2. RMSNorm
3. Primary Quantization
4. Residual Quantization

不能 fused 的话，每步都是一次 global memory round-trip，对 latency 是灾难。Fused 之后输出直接进入 NVFP4 encoded tensor。

### 5.2 Interleaved Channel Layout (Appendix D)

这是个 paper body 没明说但 Appendix D 揭晓的关键 trick：

数学公式 (2) 写成 $Q_{X_{aug}} = [Q_X \mid Q_{R_o}]$ 是 logical concat，但物理上不能这样布局——会触发 strided global memory access。

ARCQuant 的实际 layout：把 S 个 outlier channel 按 16 个一组分块（NVFP4 block size），然后 **local interleave**——每 16-channel primary block 紧跟着它的 16-channel residual block。

物理内存布局示意：
```
[P_0..P_15 | R_0..R_15 | P_16..P_31 | R_16..R_31 | ... ]
```

好处：
- Weight 矩阵 offline 一次性按这个 interleave 模式排好
- Fused kernel 在 on-chip registers 里同时算 P 和 R，coalesced write 到 global memory
- 标准 GEMM kernel 不需要任何修改就跑得通

### 5.3 实测 latency breakdown

Figure 8(b) 给了 Qwen2.5-7B 的端到端 prefill breakdown：
- 总 latency 增加仅 **4.9%**（vs. 无补偿 NVFP4）
- Fused Quantization Kernel 本身的 cost 极小
- GEMM latency 与 S 严格线性关系（Figure 8(a)），但 S ≤ 512 时几乎平直

Figure 6 端到端 prefill：
- Qwen2.5-7B: PRO 6000 上 2.0x-2.5x vs FP16
- Llama 3.1-8B: RTX 5090 上 3.5x
- Memory 降 1.5x-2.8x

Table 4 vLLM decode throughput (Qwen2.5-7B, RTX 5090, batch=8, gen_len=128):
- SeqLen 1024: 21077 total tok/s, 2342 decode tok/s (vs FP16 1195)，**1.96x decode speedup**
- SeqLen 2048: 21237 total tok/s, 1249 decode tok/s (vs FP16 600)，**2.08x decode speedup**

注意 decode 部分还超过了 FP8（FP8 只有 1855 / 992 tok/s），这点很有意思——意味着 NVFP4 的硬件 path 在 decode 上比 FP8 更高效。

## 6. Intuition 升华：为什么这个 trick 之前没人做？

Karpathy 你大概会问：这个 residual + concat 的 idea 看着很自然啊，为什么 Atom / SVDQuant / ResQ 没想到？

我的几个猜测：

1. **硬件 uniformity 之前不是 binding constraint**：INT4/INT8 混合时，atom 可以直接写 custom kernel 处理 heterogeneity。NVFP4 出来后，Tensor Core 的 MMA instruction 强制要求同 group size，custom kernel 也没办法走 hardware pipeline，这是新的硬约束
2. **Residual 自然映射到 K dimension 这个 trick 不显然**：大部分人想到 residual 会想到加一个独立的低秩 branch（SVDQuant 的做法），而不是把 residual 作为 K 维度的 augmentation。这需要意识到 GEMM 的 linear accumulation 性质可以被 exploit
3. **Error bound 的 $\epsilon_4^2 = \epsilon_8$ 性质**：这个观察很数学，需要对 floating-point format 的 epsilon 性质有深刻直觉才想得到。等于说 ARCQuant 团队真的把 NVFP4 的 E2M1 和 MXFP8 的 E4M3 拆到 bit-level 去对比了

## 7. 可能的延伸联想

基于这篇 paper 我能联想到的几个方向：

### 7.1 与 SVDQuant 的对比
SVDQuant (Li et al. 2024) 也是处理 4-bit outliers，但用 low-rank branch（FP16）吸收 outliers，diffusion model 上效果好。ARCQuant 与之区别：不引入额外 precision，纯 4-bit 路径，靠 dual-stage 量化模拟 8-bit 精度。
参考: https://arxiv.org/abs/2411.05007

### 7.2 与 QuIP# 的 lattice codebook 对比
QuIP# 用 incoherence preprocessing + lattice codebook 在 weight-only 场景下做 W4。ARCQuant 走的是另一条路：不动权重 format，靠 activation-side augmentation。
参考: https://arxiv.org/abs/2404.14001

### 7.3 Sub-4-bit 的扩展方向
Limitations 提到 "future work will extend to sub-4-bit formats"。直觉上 NVFP2 不存在（hardware 不支持），但 INT2 / 1.58-bit (BitNet) 上能不能用类似 dual-stage trick？理论上 $\epsilon_2^2 = \epsilon_4$，所以 2-bit dual-stage 应该逼近 4-bit，这是非常诱人的方向。
参考: BitNet https://arxiv.org/abs/2310.11453

### 7.4 与 Hadamard 的 "block rotation" 新论文的关系
BRQ (Shao et al. 2025) 提出 block-local rotation 而非 global rotation，声称可以绕开 ARCQuant motivation 里指出的 Hadamard 问题。这是 ARCQuant 的直接竞争者，论文里 Section 3 引用了 BRQ 但没正面比较，未来这块会是 hot area。
参考: https://arxiv.org/abs/2511.04214

### 7.5 与 vLLM / TensorRT-LLM 的整合路径
ARCQuant 已 vLLM-integrated（Table 4 已证实）。Table 10 的 detailed prefill latency 数据给出了 RTX 5090 / PRO 6000 上各种 batch / seqlen 组合的 latency 和 memory。这种 hardware-aware co-design 的思路很像之前 FlashAttention / FlashDecoding 的整合方式。

## 8. 局限性点评（我自己的观察，非论文原文）

1. **Static outlier index 的 risk**: 论文 Limitations 提到 channel reorder index 是 offline 定的。如果 inference 时 activation 分布漂移（比如 OOD prompt、distribution shift），augmented channels 可能不是真正的 outliers，补偿失效。不过 Table 6 的 calibration robustness 实验显示换 calibration set 影响 <0.03 PPL，说明 outlier 结构在 LLM 内在很稳定
2. **Weight 端没用 GPTQ / AWQ**: 这是真亏。Atom / FlatQuant 在 weight 端有专门的 Hessian-based 优化，ARCQuant 只用 RTN，理论上 weight 端还有 0.5-1 PPL 的提升空间没榨干
3. **对 KV cache quantization 没触及**: W4A4 是 matmul 端，KV cache 还在 FP8/FP16。如果 ARCQuant 思路能扩展到 KV cache，端到端 inference 的 memory 节省会更显著
4. **S 的选择是 layer-wise fixed**: 是否存在 per-token dynamic S 的可能？理论上更精细，但 hardware 实现会复杂很多

## 9. 总结：ARCQuant 教会我们什么

1. **Hardware constraint 倒逼 algorithmic innovation**: NVFP4 的 group-size=16 + MMA 同 group-size 要求，逼出了"统一 4-bit format 下做误差补偿"这个非常 specific 的研究 question
2. **Linear accumulator 是 free lunch**: GEMM 内部 K dimension 的线性求和性质，让 residual compensation 可以"免费"嵌入到主计算中，只需 K dimension 增加 S 维
3. **Bit-level math matters**: $\epsilon_4^2 = \epsilon_8$ 这种 floating-point format 层面的恒等式，催生了 dual-stage 这种 elegant 设计
4. **Format engineering 是新战场**: NVFP4 vs MXFP4 不仅是 block size 不同，scale format (E4M3 vs E8M0) 也不一样，这导致 alignment factor 性质不同。未来 hardware format 的设计会越来越深地影响 quantization algorithm 的设计

GitHub 仓库: https://github.com/actypedef/ARCQuant

希望这些技术细节 + 直觉分析能帮你在脑子里搭起完整的 ARCQuant mental model。如果你想 drill deeper 到某一个点（比如 fused kernel 的 CUDA 实现细节、$\epsilon$ 推导的更严格形式、或者与某个具体 baseline 的对比），告诉我，我可以再展开。
