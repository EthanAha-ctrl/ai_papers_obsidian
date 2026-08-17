---
source_pdf: Training LLMs with MXFP4.pdf
paper_sha256: 0c514da58b8aa7b1003424c426fd331c643cee62cb1f361bad93e9eeda52f2fe
processed_at: '2026-08-12T18:01:10-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版: Training LLMs with MXFP4

## 一句话版本

FP4 硬件比 FP8 快 2 倍, 但直接用训出来的 model 是坏的; 这篇 paper 用两个 trick——**stochastic rounding** 和 **random Hadamard transform**——把 FP4 training 救活了, 训出来的 GPT 6.7B 和 BF16 几乎一模一样。

## 想象一下背景

你现在有个 NVIDIA Blackwell GPU ([Blackwell brief](https://resources.nvidia.com/en-us-blackwell-architecture))。它有个新 toy: **MXFP4 GEMM**。MXFP4 是什么?

- 每 32 个 FP4 number 共享一个 INT8 的 scale
- FP4 本身只有 4 bits: sign + 2 exponent + 1 mantissa
- 能表示的 normal values 一共就 ±{0.5, 1, 1.5, 2, 3, 4, 6} — **总共 7 个数**

你没看错, 7 个数。但加上 per-block scale, 就能 cover 任意 magnitude。这套标准来自 [OCP MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 和 [Rouhani et al. 2023](https://arxiv.org/abs/2310.10537)。

硬件给你 2× FP8 的 throughput, 你想要吗? 当然想要, training LLM 贼贵。但问题是...

## 直接用 MXFP4 会发生什么

Table 2 里写得清清楚楚, GPT 345M 训 33B tokens:

| Backward | Val Loss | Gap |
|----------|----------|-----|
| BF16 | 2.49 | — |
| MXFP4 直接上 | 2.60 | **+0.11** (烂) |
| MXFP4+RHT+SR (本 paper) | 2.51 | +0.02 (基本无损) |

0.1 perplexity gap 在 pretraining 里是个大事。更糟糕的是, paper 里说如果考虑 FP8 已经基本无损 ([Peng et al. 2023 FP8-LM](https://arxiv.org/abs/2310.18313)) 还比 pure MXFP4 慢 30-40%, **直接用 MXFP4 连 FP8 都不如**——固定 wall clock 下你少训几步用 FP8 反而效果更好。

所以问题很清楚: 怎么把 FP4 的"快"吃下来, 同时不掉精度?

## 问题出在哪: 两个 mathematical ills

### Ill #1: Bias

标准的 MXFP4 quantization (Algorithm 1) 是这样的:

```
对每个 32-element block:
  找到 block 里 max magnitude m
  shared_exp = floor(log2(m)) - 2  # 因为 FP4 最大 normal 6 = 2^2 * 1.5
  scale X = 2^shared_exp
  对每个 element: 量化 V_i / X 到 FP4
```

问题在最后一步。归一化之后, 最大值 $m / X$ 会落在 $[4, 8)$ 之间:

$$\frac{m}{2^{\lfloor \log_2(m) \rfloor - 2}} < \frac{m}{2^{\log_2(m) - 3}} = 8$$

但 FP4 最大能表示 6, 不是 8。所以 $(6, 8)$ 之间的值会被 **clip** 掉。实测大约 **3% 的 entries 受影响**。

这个 clip 是系统性的, 不是随机的——每次 forward/backward 都 clip 同样的方向。Bias 在 gradient 里累加, 训着训着就偏了。

### Ill #2: High variance from outliers

哪怕你修了 bias, 还有 variance 问题。

LLM 的 activations、weights、gradients 都有 **outliers** ([Xi et al. 2023](https://openreview.net/forum?id=H9hWlfMT60), [QuIP#](https://proceedings.mlr.press/v235/tseng24a.html))。outlier 是什么? 就是某个 element 比其他大 10× 或 100×。

MXFP4 量化用 block max 作 scale, 所以一个 outlier 会**拉大整个 block 的 scale**, 让 block 里其他正常值 quantize 得很糙——相对精度变差。SR 又会让这些被压扁的值产生大 variance。

Theorem 3.2 给出: 不做任何处理, SR 的 variance 是

$$\text{Var} = \mathcal{O}(b \cdot \Delta^4 \cdot \|A\|_\infty \cdot \|B\|_\infty)$$

变量解释:
- $b$: block size (32)
- $\Delta$: quantizer grid 上相邻可表示点的最大 gap
- $\|A\|_\infty$, $\|B\|_\infty$: A 和 B 的最大绝对值, 就是 outlier 大小

**关键: linear in $b$**, 而且依赖 worst-case outlier。outlier 越大, variance 越大。

## 救星 #1: Stochastic Rounding (修 bias)

### 核心 idea

不要每次都 round 到最近的, **按概率 round**。

要 round 1.3 到整数:
- Nearest rounding: 永远选 1 (biased)
- Stochastic rounding: 70% 概率选 1, 30% 概率选 2, **期望是 1.3**

具体用 **dithering** 实现 ([Croci et al. 2022](https://doi.org/10.1098/rsos.211631)):

$$\delta \sim \mathcal{U}(-0.5, 0.5)$$

$$\text{SR}(x) = \begin{cases} \lfloor x \rfloor & \text{if } x + \delta < \lfloor x \rfloor + 0.5 \\ \lceil x \rceil & \text{if } x + \delta \geq \lfloor x \rfloor + 0.5 \end{cases}$$

变量解释:
- $\delta$: 加到 $x$ 上的 uniform 随机噪声
- $\lfloor x \rfloor$: 比原值小的最近 LP 可表示值
- $\lceil x \rceil$: 比原值大的最近 LP 可表示值

**为什么这样无偏**: $\delta$ uniform, 期望 0, 所以 $\mathbb{E}[x + \delta] = x$, rounding 操作保持期望。

### 加 3/4 scale 防 clip

光有 SR 还不够, 因为如果 $V_i / X > 6$, SR 还是会 clip。

Trick: 把 $V_i$ 先乘 $3/4$, 这样归一化后的最大值严格 $\leq 6$:

$$\frac{3}{4} \cdot \frac{m}{2^{\lfloor \log_2(m) \rfloor - 2}} < \frac{3}{4} \cdot 8 = 6$$

正好卡在 FP4 上限。

但这意味着你量化的是 $3/4 \cdot V_i$, 输出 scale 也变了。在 GEMM 里:

$$\mathbb{E}[\text{MXFP4\_GEMM}(A, B^T)] = \frac{9}{16} (AB^T)$$

因为 $\frac{3}{4} \times \frac{3}{4} = \frac{9}{16}$。所以最后乘 $\frac{16}{9}$ 校正 (Lemma 3.1)。

### 硬件友好

Amazon Trainium 原生支持 SR dithering, 只加 < 2% BF16 GEMM 开销 ([Trainium docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium.html))。假设 FP4 是 BF16 的 4× throughput, 折算 SR 加 < 10% 开销——基本免费。

## 救星 #2: Random Hadamard Transform (修 variance)

### Intuition: 把 outliers 摊平

LLM 里的 outlier 问题本质是**信息不均匀**: 大部分 element 很小, 少数 element 巨大。

Idea: 在 quantize **之前**做一个 random orthogonal transform, 把信息"摊平"到所有 elements 上。

为什么 orthogonal? 因为 orthogonal transform 保持 $L_2$ norm, $\|Hx\|_2 = \|x\|_2$。从 GEMM 角度:

$$(HSA)^T (HSB) = A^T S^T H^T H S B = A^T B$$

$H^T H = I$ 让变换**完全不影响 GEMM 的期望结果**, 只改变 quantization 的 statistics。

### 具体用什么 transform: Hadamard + random sign

Random Hadamard Transform (RHT) 定义:

$$x \mapsto H S x$$

变量:
- $x \in \mathbb{R}^k$: input vector
- $S \in \{\pm 1\}^k$: 随机正负号 vector (Rademacher distribution)
- $H \in \mathbb{R}^{k \times k}$: Hadamard matrix

Hadamard matrix 递归定义:

$$H_n = \frac{1}{2^{n/2}} \begin{bmatrix} H_{n-1} & H_{n-1} \\ H_{n-1} & -H_{n-1} \end{bmatrix}, \quad H_1 = [1]$$

每个 entry 是 $\pm 1/\sqrt{k}$, 所以是 orthogonal。

Intuition: $H$ 是一个"均匀 mixing"操作, $S$ 加随机性。一个 outlier 经过 RHT 后, 其能量被均匀 spread 到整条 vector 上——通过 CLT, 每个 entry 变成大致 Gaussian 分布。

### 数学: variance 从 linear 变 log

Theorem 3.2 给出: 用 RHT 之后, 以概率 $\geq (1-\epsilon)^2$:

$$\text{Var} = \mathcal{O}\left(\Delta^4 \cdot \|A\|_2 \cdot \|B\|_2 \cdot \log\left(\frac{2b}{\epsilon}\right)\right)$$

对比不用 RHT 的 $\mathcal{O}(b \cdot \Delta^4 \cdot \|A\|_\infty \cdot \|B\|_\infty)$:

- $b$ 变成 $\log b$ — 从 linear 变 logarithmic, 巨大改进
- $\|A\|_\infty$ (worst-case outlier) 变成 $\|A\|_2$ (整体能量) — 不再被单个 outlier 主导

**核心机制** (Equation 5, from [Tseng et al. 2024a](https://proceedings.mlr.press/v235/tseng24a.html), originally [Halko Martinsson Tropp 2011](https://doi.org/10.1137/090771806)):

$$\mathbb{P}(|e_i H S x| \geq a) \leq 2 \exp\left(\frac{-a^2 k}{2 \|x\|^2}\right)$$

变量:
- $e_i$: 第 $i$ 个 basis vector
- $x$: input vector
- $a$: 阈值
- $k$: vector 长度

这说的是 RHT 后任意 entry 超 $a$ 的概率是 **sub-Gaussian tail**。outlier 概率指数衰减。用 union bound 把 $\|HSx\|_\infty$ bound 住:

$$\|HSx\|_\infty = \mathcal{O}\left(\|x\|_2 \sqrt{\frac{\log(2b/\epsilon)}{b}}\right)$$

这就把 $\|A\|_\infty$ 换成了 $\|A\|_2$ 乘个 $\sqrt{\log b / b}$ 因子。

### 实证: Figure 2

paper 里测了 $A, B \sim \mathcal{N}(0, I) + \text{Bernoulli}(p) \cdot \mathcal{N}(0, 5I)$, 即加 $p$ 比例 outlier。

不用 RHT: variance 随 $b$ 线性增长, 受 outlier 比例 $p$ 严重影响
用 RHT: variance 增长慢得多 (logarithmic)

### Blockwise 实现 (Algorithm 3)

问题: 完整 RHT 在 batch 维度 mix, 在 FSDP/ZeRO-3 ([Zhao et al. 2023](https://arxiv.org/abs/2304.11277), [Rajbhandari et al. 2020](https://arxiv.org/abs/1910.02054)) 下需要跨 GPU 通信, 太贵。

Trick: 用 small block size $g$ (取 64), 把 RHT 当 dense matmul 做。

```
G' = (dL/dy).view(bm/g, g) @ diag(S) @ H_g
W' = H_g^T @ diag(S) @ (W.view(g, nm/g))
dL/dx = MXFP4_GEMM(G', W')
```

复杂度:
- Runtime: $O((b+m)ng)$ — 对小 $g$ 是 memory bound
- $g \leq 256$ 时 memory bound, 可以 fuse 进 GEMM kernel
- 不需要跨 GPU 通信 (g 远小于 seq len)

实测 Table 4: $g=64$ 已经足够好, 更大收益递减。

## 两个 trick 必须合在一起用

Table 2 + Figure 6 揭示一个微妙的事:

**短 run (40B tokens)**: 单独用 RHT 或单独用 SR 都够。两者效果差不多。

**长 run (210B tokens)**:
- 只用 RHT (no SR): gap 0.10 — 不够
- 只用 SR (no RHT): 最终 OK, 但 **初期收敛慢**
- RHT + SR: gap 0.00 — 完美

为什么 SR alone 初期慢? SR 虽无偏, 但小 gradient 值会被 stochastic flush 到 0 (Equation 1 里 dither 把小数推到 floor), 信息丢失。RHT 把 gradient 变到另一空间, **降 variance 同时降单 entry 被置零概率**。

为什么 RHT alone 长期不够? RHT 是 **biased**(用了 3/4 scale 但没 SR 还原), 长 run 累积 bias。

两者各管一头: **SR 管 bias (期望对), RHT 管 variance (单次估计准)**。

## 整个 recipe 长啥样

对每个 decoder linear layer 的 backward pass:

```
# Backward pass for y = xW^T (no bias)
# Input: dL/dy ∈ R^(b×m), x ∈ R^(b×n), W ∈ R^(m×n)

# Step 1: 准备 RHT 材料
H = Hadamard matrix H_g  # g=64
S = random sign vector

# Step 2: 对所有 operands 做 blockwise RHT
G'  = (dL/dy).view(bm/g, g) @ diag(S) @ H    # transform gradient
W'  = H^T @ diag(S) @ (W.view(g, nm/g))      # transform weight
GT' = (dL/dy^T).view(bm/g, g) @ diag(S) @ H
X'  = H^T @ diag(S) @ (x.view(bn/g, g))      # transform activation

# Step 3: 用 Algorithm 2 量化到 MXFP4 (3/4 scale + SR)
# Step 4: 跑 MXFP4 GEMM
dL/dx = MXFP4_GEMM(G', W')
dL/dW = MXFP4_GEMM(GT', X')

# Step 5: 校正 9/16 scale
dL/dx *= 16/9
dL/dW *= 16/9

return dL/dx, dL/dW
```

Forward pass 不变, 还是 BF16 (mixed precision)。

## 速度账 (Table 5)

A100 GPU, Llama 2 70B decoder layer, batch 16K tokens, FP16 forward:

| Backward | E2E tok/s | BW tok/s | vs FP16 BW |
|----------|-----------|----------|-----------|
| FP16 | 46983 | 72563 | 1.0× |
| INT8 (FP8 proxy) | 55469 | 94688 | 1.31× |
| INT4 (MXFP4 proxy) no RHT | 67306 | 133952 | 1.85× |
| INT4 + RHT g=64 | 64335 | 123056 | **1.70×** |

INT4 + RHT g=64: **backward 比 FP16 快 70%, 比 FP8 快 30%**。

RHT overhead: g=64 时只加 5% E2E overhead。在 H100 上 7B-sized matrix RHT 加 9.7%, 70B-sized 加 1.6%——假设 MXFP4 是 FP8 的 2× throughput, 翻倍为 19.4% 和 3.2%, **仍然比 FP8 GEMM 快**。

## 直觉总结: 三个层次的 insight

### Insight 1: Quantization bias 累积成 gradient bias, 长期致命

直接 MXFP4 看 short run 还行, 0.1 gap 不大。但长 run (210B tokens) 显示 bias 累积是真问题。SR 通过让每次 quantization 期望正确, 从根本上消除累积。

### Insight 2: Outlier 是 LP training 的最大敌人

LLM outlier 是 well-known ([Xi et al. 2023](https://openreview.net/forum?id=H9hWlfMT60), [QuIP#](https://proceedings.mlr.press/v235/tseng24a.html))。MX 的 per-block scale 设计本来想 mitigate, 但只是把问题从"全局 scale 不够"变成"block 局部 scale 被 outlier 拉爆"。RHT 用 orthogonal transform **物理上**把 outlier 能量摊平, 是 structural fix 而非 symptomatic fix。

### Insight 3: orthogonal transform 是 quantization 的免费午餐

$H^T H = I$ 让 RHT 不改 GEMM 期望值, 只改 quantization statistics。这种"不影响结果只影响精度"的性质, 是 quantization 工具箱的瑞士军刀。同源思想在 QuIP# (推理量化), randomized SVD ([Halko Martinsson Tropp 2011](https://doi.org/10.1137/090771806)), sketching 算法里都有。

## 为什么只在 backward 用 FP4

两个原因:
1. **FLOPs 占比**: backward 占 > 50% training FLOPs, 省 backward 收益大
2. **不损 model capacity**: forward 用 BF16/FP8 保留 model 表达力。FP4 forward 会限制 model capacity ([Kumar et al. 2025 scaling laws for precision](https://openreview.net/forum?id=wg1PCg3CUP)), 但 FP4 backward 只影响 gradient 估计质量——而 SR+RHT 正好是"估计准 gradient"的 tool, 完美对应。

paper 还测了 FP8 forward + MXFP4 backward (Section 6.1, Figures 8-9), 也基本 lossless, 进一步推 speed-quality tradeoff。

## 限制

- 没真测 Blackwell 上的 wall-clock speedup, 用 INT4 on A100 做 proxy
- 最大只到 6.7B, 没到 405B 级别
- 最长 210B tokens, 可能更长的 run 会暴露新问题
- 只对 decoder linear layers, attention QKV proj 仍用高精度
- 并发工作 [Wang et al. 2025](https://arxiv.org/abs/2501.17116) 用不同方法 (differentiable gradient estimator + outlier 保留高精度), gap > 0.5; 本文 gap < 0.1

## 最终直觉

整个 recipe 是个**精巧的刚好够用**的 intervention:

1. FP4 太少 bits → 加 MX per-block scale → 但 bias
2. 加 3/4 防 clip → 但 nearest rounding 还 biased
3. 换 SR → unbiased 了, 但 outlier 让 variance 爆炸
4. 加 RHT → variance 从 $O(b)$ 变 $O(\log b)$, 还顺便避免小值被 flush
5. blockwise RHT → memory bound, 几乎免费
6. forward 不动 → 保住 model capacity

每一步都精准对应一个 specific numerical ill, 没有多余动作。最终效果: 4-bit training 从"不可用"被推到"比 FP8 还快 1.3×", 而且几乎无损。

这种 system + ML theory 的结合, 把硬件给的 2× 加速真正吃到嘴里, 是 low-precision training 领域的 elegant 典范。

## 相关阅读

- [OCP MX v1.0 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) — MX 格式官方标准
- [Rouhani et al. 2023 - Microscaling data formats](https://arxiv.org/abs/2310.10537) — MX 原始 paper
- [Croci et al. 2022 - Stochastic rounding](https://doi.org/10.1098/rsos.211631) — SR 数学基础
- [Halko Martinsson Tropp 2011](https://doi.org/10.1137/090771806) — RHT 起源, randomized SVD
- [Tseng et al. 2024a - QuIP#](https://proceedings.mlr.press/v235/tseng24a.html) — RHT 用于 LLM 推理量化
- [Tseng et al. 2024b - QTIP](https://arxiv.org/abs/2409.02586) — 同作者后续量化工作
- [Peng et al. 2023 - FP8-LM](https://arxiv.org/abs/2310.18313) — FP8 training baseline
- [Xi et al. 2023 - INT4 training](https://openreview.net/forum?id=H9hWlfMT60) — 之前 INT4 training 尝试
- [Kumar et al. 2025 - Scaling laws for precision](https://openreview.net/forum?id=wg1PCg3CUP) — precision scaling laws
- [Wang et al. 2025 - concurrent FP4 work](https://arxiv.org/abs/2501.17116) — 并发工作
- [NVIDIA Blackwell brief](https://resources.nvidia.com/en-us-blackwell-architecture) — FP4 硬件
- [Amazon Trainium docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium.html) — SR 硬件支持
- [Agarwal et al. 2024 - HadaCore](https://arxiv.org/abs/2412.08832) — 快速 Hadamard kernel
- [Microsoft microxcaling emulation library](https://github.com/microsoft/microxcaling) — MX 模拟
- [Megatron-LM](https://arxiv.org/abs/1909.08053) — 训练 codebase
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) — FP8 实现

---

# Training LLMs with MXFP4 深度讲解

## 1. 论文核心问题与动机

当前 LLM 训练成本惊人 — Llama 3.1 405B 耗费 3×10²⁴ FLOPs, 超过 10000 GPUs 训练数月 ([Dubey et al., 2024](https://arxiv.org/abs/2407.21783))。training 是 compute-bound 的, 90%+ FLOPs 在 linear layers ([Casson, 2023](https://www.adamcasson.com/posts/transformer-flops))。

硬件层面, low precision GEMM 收益巨大 — FP8 比 FP32 快 4×, MXFP4 又比 FP8 快 2× ([NVIDIA Blackwell brief](https://resources.nvidia.com/en-us-blackwell-architecture))。问题在于: 直接用 MXFP4 替换 BF16 会严重 degrade 模型质量。这篇 paper 是第一个做到 near-lossless 的 MXFP4 training recipe。

## 2. 背景知识: 浮点数与 MX 格式

### 2.1 IEEE 754 浮点表示

IEEE float 用 1 sign bit + e exponent bits + m mantissa bits 表示, 记作 EeMm。一个数的实际 normal value 为:

$$(-1)^S (1+M) \cdot 2^{E - \text{bias}}$$

其中:
- $S \in \{0,1\}$: sign bit
- $M$: mantissa bits 组成的小数部分, 是一个 $[0, 1)$ 之间的值
- $E$: exponent bits 表示的整数
- $\text{bias}$: 数据类型相关的偏移常数, 比如 FP32 的 bias = 127

关键观察: FP datatype 对 quantization SNR 是 scale-invariant 的 — 也就是说不管你把 input 放大几倍, 相对误差不变 (除了 over/underflow 之外, [Blake et al., 2023](https://proceedings.mlr.press/v202/blake23a.html))。

### 2.2 FP4 的痛点

看 Table 1 中的 FP4: 1+2+1 bits, 仅有 normal values {0.5, 1, 1.5, 2, 3, 4, 6} (考虑 sign)。

- FP8 E4M3 dynamic range = 448 / 2^(-9) ≈ 2.3×10⁶
- FP4 dynamic range = 6 / 0.5 = **12** (小得可怜)

这就是为什么需要 Microscaling (MX): 每 32 个 FP4 共享一个 INT8 scale $s$, 实际表示值是 $2^{s-1} \cdot v$ (其中 1 是 FP4 exponent bias)。这样把 dynamic range 大幅扩大, 代价仅每 entry 多花 $8/32 = 0.25$ bit。MX 标准详见 [OCP MX v1.0 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

### 2.3 Algorithm 1: 标准 MX 量化 (biased)

输入: $V \in \text{HP\_DTYPE}^k$, $k=32$ (硬件 block 大小)

```
shared_exp ← ⌊log₂(max_i(|V_i|))⌋ - emax_elem
X ← 2^shared_exp
for i = 1 to k:
    P_i = quantize_to_LP(V_i / X)
return X, {P_i}
```

这里:
- $\text{emax\_elem}$ = base datatype 中最大 normal number 的 exponent, 对 FP4 是 2 (因为 $6 = 2^2 \times 1.5$)
- $X$: 共享 scale, 是 2 的幂
- $P_i$: 量化后的 FP4 值

**Bias 来源**: 经过 $V_i / X$ 归一化后, 最大值 $m$ 变为:

$$m \mapsto \frac{m}{2^{\text{shared\_exp}}} < \frac{m}{2^{\log_2(m) - 3}} = 8$$

但 FP4 的最大 normal 值是 6, 所以 $m \in (6, 8)$ 的值会被 clip 掉。论文实测约 3% 的 entries 受影响。

## 3. 核心方法之一: 无偏量化 (Algorithm 2)

### 3.1 关键 idea

两个改动让 Algorithm 1 变无偏:

1. **缩放 3/4**: $V_i \leftarrow \frac{3}{4} V_i$, 防止 clipping
2. **Stochastic Rounding**: 用 SR 替代 nearest rounding

### 3.2 为什么 3/4 就够?

重新走一遍 bound:

$$\frac{3}{4} \cdot \frac{m}{2^{\lfloor \log_2(m) \rfloor - 2}} < \frac{3}{4} \cdot \frac{m}{2^{\log_2(m) - 3}} = \frac{3}{4} \times 8 = 6$$

正好卡到 FP4 最大值 6。$m/X$ 的上界变成 6, 严格小于等于 FP4 可表示范围, 不会 overflow。

### 3.3 Stochastic Rounding 实现机制

SR 通过 dithering 实现: 给 input 加均匀噪声然后做 nearest rounding。Equation (1)-(2):

$$\delta \sim \mathcal{U}(-0.5, 0.5)$$

$$\text{SR}_{\text{dither}}(x) = \begin{cases} \lfloor x \rfloor & \text{if } x + \delta < \lfloor x \rfloor + \frac{1}{2} \\ \lceil x \rceil & \text{if } x + \delta \geq \lfloor x \rfloor + \frac{1}{2} \end{cases}$$

这里 $\lfloor x \rfloor$ 和 $\lceil x \rceil$ 是相邻的两个 LP 可表示值。

intuition: 比如要量化 1.3 到整数, nearest rounding 总是给 1 (biased)。SR 给 70% 概率是 1, 30% 概率是 2, 期望正好是 1.3。多次重复就能消除系统性偏差。

Amazon Trainium 硬件有原生 SR 支持, 只增加 < 2% GEMM 开销 ([Amazon Trainium docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium.html))。

### 3.4 Lemma 3.1: 无偏性证明

**Lemma 3.1**: Algorithm 2 产生 MXFP4 matrix, 是 input 的 $\frac{3}{4}$ 的无偏估计。Algorithm 3 把 Algorithm 2 作为 subroutine, 产生 $\frac{dL}{dx}$ 和 $\frac{dL}{dW}$ 的无偏估计。

证明的关键 chain:
- 每个 $P_i$ 是 $\frac{3}{4} \frac{V_i}{X}$ 的无偏估计 (因为 SR 无偏且没 clipping)
- $X \cdot P_i$ 是 $\frac{3}{4} V_i$ 的无偏估计 (X 是确定性的)
- 对 GEMM $C = \text{MXFP4\_GEMM}(A, B^T)$:

$$\mathbb{E}[C_{ij}] = \sum_{k=0}^{n/g} \left( X_{A_{i,k}} X_{B_{j,k}} \sum_{l=0}^{g} \mathbb{E}[A_{i,k}^{\text{FP4}}{}_l] \cdot \mathbb{E}[B_{j,k}^{\text{FP4}}{}_l] \right)$$

(独立性来自 SR 用独立 dithering noise)

$$= \sum_{k=0}^{n/g} X_{A_{i,k}} X_{B_{j,k}} \sum_{l=0}^{g} \frac{3}{4} \frac{A_{i,kl}}{X_{A_{i,k}}} \cdot \frac{3}{4} \frac{B_{j,kl}}{X_{B_{j,k}}}$$

$$= \frac{9}{16} (AB^T)_{ij}$$

最后 lines 10-11 乘 $\frac{16}{9}$ 校正。对 backward pass:
- $\frac{dL}{dx}$: $A = \frac{dL}{dy}\text{diag}(S)H$, $B = W^T \text{diag}(S)H$, 用 $HH^T = I$ 性质
- $\mathbb{E}[\text{MXFP4\_GEMM}(A, B^T)] = \frac{9}{16} \frac{dL}{dy} W$

## 4. 核心方法之二: Random Hadamard Transform bound variance

### 4.1 为什么只无偏还不够?

无偏只保证 $\mathbb{E}[\hat{g}] = g$ (true gradient), 但如果 $\text{Var}(\hat{g})$ 太大, 单次更新可能严重偏离 true gradient, 等于给梯度注入大量噪声, 损害 convergence。

LLM 的 activations、weights、gradients 都有 **outliers** ([Xi et al., 2023](https://openreview.net/forum?id=H9hWlfMT60), [Tseng et al., 2024a - QuIP#](https://proceedings.mlr.press/v235/tseng24a.html))。MXFP4 quantization 用 group 的 max element 作 scale, 一旦一个 block 里有 outlier, 整个 block 的 quantization distortion 都被放大。

### 4.2 Random Hadamard Transform

RHT 定义: $x \mapsto HSx$, 其中:
- $x \in \mathbb{R}^{j \times k}$: input
- $S \in \{\pm 1\}^k$: random sign vector (Rademacher)
- $H \in \mathbb{R}^{k \times k}$: k-dimensional Hadamard matrix

Hadamard matrix 递归定义 (Equation 4):

$$H_n = \frac{1}{2^{n/2}} \begin{bmatrix} H_{n-1} & H_{n-1} \\ H_{n-1} & -H_{n-1} \end{bmatrix}, \quad H_1 = [1]$$

性质:
- $H$ 是正交矩阵: $HH^T = I$
- $\text{diag}(S)$ 也是正交
- RHT 完全可逆: $(HSA)^T(HSB) = A^T S^T H^T H S B = A^T B$

所以可以直接把 RHT 应用到 GEMM operands, 不需要 inverse。

### 4.3 Theorem 3.2: variance bound

**Theorem 3.2**: 对 $A, B \in \mathbb{R}^b$:

- **不用 RHT**: $\text{Var}(\mathcal{Q}(A)^T \mathcal{Q}(B)) = \mathcal{O}(b \Delta^4 \|A\|_\infty \|B\|_\infty)$
  
- **用 RHT**: 以概率 $\geq (1-\epsilon)^2$, $\text{Var}(\mathcal{Q}(HSA)^T \mathcal{Q}(HSB)) = \mathcal{O}(\Delta^4 \|A\| \|B\| \log(2b/\epsilon))$

其中 $\Delta$ 是 quantizer 中相邻可表示点的最大 gap。

**核心对比**:
- 不用 RHT: variance **linear** in block size $b$, 依赖 $\|A\|_\infty$ (worst-case outlier)
- 用 RHT: variance **logarithmic** in $b$, 依赖 $\|A\|_2$ (整体 energy)

### 4.4 证明关键步骤

第一步 — SR 单元素的 variance:

设 $\alpha = A_i / X_A$, SR 采样到 $f(\alpha)$ (largest representable ≤ α) 概率 $\frac{c(\alpha)-\alpha}{c(\alpha)-f(\alpha)}$, 采样到 $c(\alpha)$ (smallest representable ≥ α) 概率 $\frac{\alpha-f(\alpha)}{c(\alpha)-f(\alpha)}$。

经过代数推导:

$$\text{Var}(Q_{A_i}) = (c(\alpha) - \alpha)(\alpha - f(\alpha)) = \mathcal{O}(\Delta^2)$$

第二步 — 独立性, 拆解 GEMM element 的 variance:

$$\text{Var}(C) = X_A X_B \sum_{i=1}^{b} \big( \text{Var}(Q_{A_i})\text{Var}(Q_{B_i}) + \text{Var}(Q_{A_i})\mathbb{E}[Q_{B_i}]^2 + \text{Var}(Q_{B_i})\mathbb{E}[Q_{A_i}]^2 \big)$$

代入 $\mathbb{E}[Q_{A_i}] = A_i/X_A$ 和 $\mathbb{E}[Q_{B_i}] = B_i/X_B$:

$$\text{Var}(C) = \mathcal{O}\left( b \Delta^4 X_A X_B + \Delta^2 \frac{X_A}{X_B} \|B\|^2 + \Delta^2 \frac{X_B}{X_A} \|A\|^2 \right)$$

第三步 — 关键 bound $X_A$:

由 Algorithm 1/2, $X_A = 2^{\lfloor \log_2 \max|A_i| \rfloor - \text{emax\_elem}} = \Theta(\|A\|_\infty)$

不用 RHT 时直接代入: $\text{Var}(C) = \mathcal{O}(b \Delta^4 \|A\|_\infty \|B\|_\infty)$ ✓

第四步 — RHT 的 concentration:

引用 [Tseng et al., 2024a](https://proceedings.mlr.press/v235/tseng24a.html) (Halko et al. 2011) 的 bound:

$$\mathbb{P}(|e_i H S x| \geq a) \leq 2 \exp\left( \frac{-a^2 k}{2\|x\|^2} \right)$$

由 union bound:

$$\mathbb{P}\left(\max_i |e_i A S H^T| \geq \sqrt{\frac{2\|A\|^2}{b} \log\frac{2b}{\epsilon}}\right) \leq \epsilon$$

所以以概率 $1-\epsilon$:

$$\|ASH^T\|_\infty = \mathcal{O}\left( \|A\| \sqrt{\frac{\log(2b/\epsilon)}{b}} \right)$$

代入即得 log 项。✓

### 4.5 实验验证 (Figure 2)

采样 $A, B \sim \mathcal{N}(0, I) + \text{Bernoulli}(p) \cdot \mathcal{N}(0, 5I)$, 即正常分布加 $p$ 比例 outlier。结果:
- 不用 RHT: variance 随 $b$ 线性增长, outlier 比例 $p$ 影响大
- 用 RHT: variance 随 $b$ 增长慢得多 (logarithmic)

## 5. Blockwise RHT: memory-bound 实现

### 5.1 两个工程挑战

1. **Data parallelism 问题**: 计算 $\frac{dL}{dW} = \mathcal{Q}(HS\frac{dL}{dy})^T \mathcal{Q}(HSx)$, RHT 在 batch 维度 mix。FSDP/ZeRO-3 ([Zhao et al., 2023](https://arxiv.org/abs/2304.11277), [Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054)) 下 activations 分 shard 到不同 GPU, 全 RHT 需要昂贵跨 GPU 通信
2. **RHT 本身开销**: 虽然 Equation 4 有 $O(n \log n)$ 算法, 但 RHT 在高精度下做。如果比 FP4 GEMM 还慢, 就不如直接用 FP8

### 5.2 Blockwise RHT (Algorithm 3)

设 RHT block size $g$ ($32 | g$, $g \leq 256$), 把 RHT 实现为 small dense matmul:

```
H ← Hadamard matrix H_g
S ← random sign vector
G' ← (dL/dy).view(bm/g, g) @ diag(S) @ H    # gradient
W' ← H^T diag(S) (W.view(g, nm/g))           # weight
GT' ← (dL/dy^T).view(bm/g, g) @ diag(S) @ H
X' ← H^T diag(S) (x.view(bn/g, g))            # activation
dL/dx ← MXFP4_GEMM(G', W')
dL/dW ← MXFP4_GEMM(GT', X')
if Algorithm 2:  # SR
    dL/dx ← (16/9) dL/dx
    dL/dW ← (16/9) dL/dW
```

**复杂度分析**:
- Runtime: $O((b+m)ng)$
- IO cost: $O(bn + nm + bm)$
- Memory bound 当 $g \lesssim 256$ (现代 AI accelerator 算力/带宽比高)

关键工程 insight: $g$ 远小于 sequence length, 所以 data-parallel 下也是 drop-in 替换。lines 3-6 理论上可以 fuse 到 lines 7-8 减少 memory access。

### 5.3 RHT block size 消融 (Table 4)

GPT 345M, 33B tokens:

| BW Pass | BF16 | g=32 | g=64 | g=128 | g=256 |
|---------|------|------|------|-------|-------|
| Val PPL | 11.89 | 12.02 | 12.01 | 11.98 | 11.98 |

block size 越大, variance 越小, perplexity 越好, 但收益递减。$g=64$ 是 sweet spot。

## 6. 实验结果详解

### 6.1 主结果 (Table 2)

所有实验 forward pass 都用 BF16 mixed precision, 只在 backward 改 precision:

| Params | Tokens | Backward Precision | Val Loss |
|--------|--------|--------------------|----------|
| 345M | 33B | BF16 | 2.49 |
| 345M | 33B | MXFP4 only | **2.60** (gap 0.11) |
| 345M | 33B | MXFP4+RHT | 2.51 |
| 345M | 33B | **MXFP4+RHT+SR** | **2.51** (gap 0.02) |
| 1.3B | 42B | BF16 | 2.32 |
| 1.3B | 42B | MXFP4 only | **2.40** (gap 0.08) |
| 1.3B | 42B | MXFP4+RHT | 2.33 |
| 1.3B | 42B | **MXFP4+RHT+SR** | **2.32** (gap 0.00) |
| 6.7B | 21B | BF16 | 2.27 |
| 6.7B | 21B | MXFP4+RHT | 2.28 |
| 6.7B | 21B | **MXFP4+RHT+SR** | **2.27** (gap 0.00) |

### 6.2 长 run 才体现 SR 价值 (Figure 6, Section 6.4)

GPT 1.3B 训 210B tokens (5× 之前):

| Backward | Val PPL | Gap vs BF16 |
|----------|---------|-------------|
| BF16 | 9.92 | — |
| MXFP4+RHT (no SR) | 10.02 | +0.10 |
| MXFP4+RHT+SR | 9.90 | -0.02 |

**关键 insight**: 短 run (40B tokens) 用 RHT 或 SR 之一就够。长 run (210B tokens) **必须** 用 SR — 否则有 0.1 的 perplexity gap。SR 的无偏性在长时间累计下变得至关重要。

### 6.3 为什么 SR without RHT 初期慢 (Table 4 推论)

MXFP4+SR only 早期 convergence 比 RHT 变体慢。原因: SR 虽无偏, 小值会被 stochastic flush 到 0 (Equation 1)。RHT 把 gradient 变到另一空间, 显著降低单 entry 被置零的概率 — 既降 variance, 又保留 gradient 信息。

### 6.4 FP8 forward + MXFP4 backward (Section 6.1)

进一步 push speed-quality tradeoff: FP8 (E4M3) forward + MXFP4+RHT+SR backward。GPT 1.3B 和 6.7B 都与 BF16 forward 接近 lossless (Figures 8-9)。注意 6.7B 的 FP8 是 BF16 模拟的, 相对误差约 0.3%。

### 6.5 下游任务与 fine-tuning (Table 3)

GPT 6.7B (20B tokens) + Tulu V2 5 epochs fine-tuning:

| Model | ArcC | ArcE | PiQA | BoolQ | Wino |
|-------|------|------|------|-------|------|
| BF16 | 23.1 | 49.2 | 60.5 | 53.3 | 59.6 |
| MXFP4* | 22.2 | 47.8 | 61.3 | 52.0 | 49.6 |
| BF16 + Tulu V2 | 25.6 | 50.6 | 62.7 | 59.6 | 51.6 |
| MXFP4* + Tulu V2 | 25.9 | 49.9 | 62.9 | 60.5 | 51.8 |

Fine-tuning final train perplexity: BF16 = 1.96, MXFP4* = 1.98。基本无损。

### 6.6 Overhead 估算 (Table 5)

A100 GPU, Llama 2 70B decoder layer, batch 16K tokens:

| Backward | E2E tok/s | BW tok/s |
|----------|-----------|----------|
| FP16 | 46983 | 72563 |
| INT8 (FP8 proxy) | 55469 | 94688 |
| INT4 (MXFP4 proxy), no RHT | 67306 | 133952 |
| INT4 + RHT g=64 | 64335 | 123056 |
| INT4 + RHT g=128 | 64171 | 122734 |
| INT4 + RHT g=256 | 63979 | 121823 |
| INT4 + RHT g=1024 (HadaCore, [Agarwal et al., 2024](https://arxiv.org/abs/2412.08832)) | 62640 | 120495 |

E2E: INT4+RHT g=64 比 FP16 快 37%, 比 INT8 快 16%
BW only: INT4+RHT g=64 比 FP16 快 69%, 比 INT8 快 30%

RHT overhead: g=64 时仅 5% E2E overhead, 在 g≤256 内 memory bound。

H100 上 7B-sized matrix: RHT 9.7% overhead; 70B-sized: 1.6% overhead。乘以 2 (假设 MXFP4 是 FP8 的 2× throughput): 19.4% 和 3.2%, 仍然比 FP8 GEMM 快。

SR overhead: Trainium 1 上 SR to BF16 加 <2%, 假设 FP4 是 BF16 的 4× throughput, 折算 <10%。

### 6.7 训练超参 (Section 7)

| Hyperparameter | 345M | 1.3B | 6.7B |
|----------------|------|------|------|
| Decoder Layers | 24 | 24 | 32 |
| Hidden Size | 1024 | 2048 | 4096 |
| Attention Heads | 16 | 16 | 32 |
| Context Length | 1024 | 2048 | 2048 |
| Batch Size | 64 | 1024 | 256 |
| LR | 1.5e-4 | 2e-4 | 1.2e-4 |
| LR Scheduler | Cosine | Cosine | Cosine |
| Weight Decay | 1e-2 | 0.1 | 0.1 |
| LR Warmup | 0.01 | 0.01 | 0.01 |
| Grad Clip | 1.0 | 1.0 | 1.0 |

## 7. 整体方法的 Intuition 构建

### 7.1 三个层次的洞察

**层次 1 — Quantization 是有偏的**: 标准 MXFP4 量化 (Algorithm 1) 因为 FP4 最大值是 6 而非 8, ~3% entries 被 clip, 产生系统性 bias。Bias 在 gradient 中累积, 长期 training 严重偏离。

**层次 2 — 无偏但 variance 大**: 用 3/4 缩放 + SR (Algorithm 2) 消除 bias, 但 LLM 的 outlier 让 SR 的 variance 线性增长于 block size, 等于把大噪声注入 gradient。

**层次 3 — RHT 控制 variance**: Random Hadamard transform 是 orthogonal 变换, 不改 GEMM 结果 (因 $H^T H = I$), 但把 input 从 worst-case $\|A\|_\infty$ (outlier driven) 变到 sub-Gaussian tail, variance 从 $O(b)$ 变 $O(\log b)$。

### 7.2 为什么 RHT 是"对的"工具

RHT 本质是"average out outliers":
- 一个 outlier 元素经过 RHT 后, 其能量被均匀 spread 到所有元素上
- 每个新元素大致是所有原元素的加权和, 由 central limit theorem 趋于 Gaussian
- Gaussian tail 给出 Equation 5 的 exponential concentration
- 这就是 QuIP# 等量化工作中"incoherence"的同源思想 ([Tseng et al., 2024a](https://proceedings.mlr.press/v235/tseng24a.html))

### 7.3 为什么在 backward 而不是 forward

backward pass 占 > 50% training FLOPs。FP4 forward 会限制模型表达力 ([Kumar et al., 2025 - scaling laws for precision](https://openreview.net/forum?id=wg1PCg3CUP)), 但 FP4 backward 只影响 gradient estimation quality, 不限制 model capacity。这正好对应 SR + RHT 的目标: 估计准的 gradient。

### 7.4 vs. 并发工作 [Wang et al., 2025](https://arxiv.org/abs/2501.17116)

并发工作也用 FP4 训 LLM, 但用 differentiable gradient estimator + outlier 保留高精度, perplexity gap > 0.5。本文方法 gap < 0.1。

## 8. 公式变量汇总表

| 符号 | 含义 |
|------|------|
| $S$ (在 IEEE float) | sign bit ∈ {0, 1} |
| $M$ | mantissa bits 表示的小数 |
| $E$ | exponent bits 表示的整数 |
| $\text{bias}$ | datatype 相关的 exponent offset |
| $V \in \mathbb{R}^k$ | 输入 high precision tensor, $k=32$ 是 MX group size |
| $\text{emax\_elem}$ | LP datatype 最大 normal value 的 exponent (FP4 = 2) |
| $X$ | 共享 scale = $2^{\text{shared\_exp}}$ |
| $P_i$ | 量化后 LP 元素 |
| $m$ | $\max_i(|V_i|)$, group 内最大 magnitude |
| $\delta$ | dithering 噪声, $\mathcal{U}(-0.5, 0.5)$ |
| $\lfloor x \rfloor$, $\lceil x \rceil$ | 相邻可表示 LP 值 (floor/ceil 到 LP grid) |
| $f(\alpha)$, $c(\alpha)$ | 量化 grid 中 $\leq \alpha$ 最大值, $\geq \alpha$ 最小值 |
| $\Delta$ | quantizer 中相邻可表示点最大 gap |
| $H$ | Hadamard matrix (orthogonal) |
| $S$ (在 RHT) | random sign vector $\in \{\pm 1\}^k$ |
| $g$ | RHT block size (取 64) |
| $b, n, m$ | batch, input dim, output dim |
| $A_{i,kg:(k+1)g}$ | 第 $i$ 行第 $k$ 个 size-$g$ block |
| $X_{A_{i,kg:(k+1)g}}$ | 该 block 的共享 scale |
| $A^{\text{FP4}}_{i,kg:(k+1)g}$ | 该 block 量化后的 FP4 vector |
| $\|A\|_\infty$, $\|A\|_2$ | $L_\infty$ (max element), $L_2$ (Frobenius) 范数 |

## 9. 速度与质量总结表

| Recipe | Backward Speedup vs BF16 | vs FP8 | Val PPL Gap vs BF16 (1.3B 210B) |
|--------|--------------------------|--------|-------------------------------|
| BF16 MP | 1.0× | ~0.7× | 0 |
| FP8 MP | ~1.4× | 1.0× | ~0 |
| MXFP4 only | ~1.8× | ~1.3× | >0.5 (large gap) |
| MXFP4+RHT | ~1.7× | ~1.2× | 0.10 |
| **MXFP4+RHT+SR** | **~1.7×** | **~1.2×** | **~0.00** |

## 10. 与更广领域的联系

### 10.1 与 randomized linear algebra 的关系

RHT 源自 [Halko, Martinsson, Tropp 2011](https://doi.org/10.1137/090771806) 的 randomized SVD 工作。本质是用 random projection 把 high-coherence matrix 变成 low-coherence, 让 uniform sampling/quantization 有效。

### 10.2 与 LLM quantization 推论工作的关系

QuIP# ([Tseng et al. 2024a](https://proceedings.mlr.press/v235/tseng24a.html)) 和 QTIP ([Tseng et al. 2024b](https://arxiv.org/abs/2409.02586)) 都用 Hadamard incoherence 做 LLM weight 4-bit quantization 推理。本文把同样的工具第一次成功用到 training 上, 并证明 SR + RHT 的组合是关键 — 单用 RHT 在长 run 上不够, 还需要 SR 的无偏性。

### 10.3 与 scaling laws for precision 的关系

[Kumar et al. 2025](https://openreview.net/forum?id=wg1PCg3CUP) 推导 precision 与 model size/data 的 scaling law。本文方法在 backward 用 4-bit 但 forward 用 BF16/FP8, 是一种"asymmetric precision"策略, 避开 forward 的 capacity 限制同时省 backward FLOPs。

### 10.4 Hardware 趋势

NVIDIA Blackwell B200/B100 支持 FP4 native ([NVIDIA 2024a](https://resources.nvidia.com/en-us-blackwell-architecture), [NVIDIA 2024b](https://developer.nvidia.com/blog/nvidia-blackwell-platform-sets-new-llm-inference-records-in-mlperf-inference-v4-1/))。Amazon Trainium 原生支持 SR ([Amazon docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium.html))。这篇 work 正好踩在硬件 ready 的节点上。

## 11. 局限与未来方向

1. **未实测 FP4 hardware wall-clock**: 论文用 INT4 on A100 作 proxy, 估得 1.3× over FP8。Blackwell 上实际数字待验证。
2. **未在更大 model (>7B) 验证**: 6.7B 已经够代表, 但 405B 级别可能有不同 dynamics。
3. **未在更大数据集 (>210B tokens) 验证**: scaling laws 显示 precision gap 可能随 data 增长而放大, 需要更长 run 确认。
4. **RHT 是 dense matmul**: 虽然 memory bound, HadaCore ([Agarwal et al., 2024](https://arxiv.org/abs/2412.08832)) 的 $O(n \log n)$ 算法在 $g=1024$ 已接近 dense 但仍稍慢。可能有更好的 fast orthogonal transform。
5. **只对 decoder linear layers**: attention 内部 GEMM (QKV proj) 仍用高精度。能否 extend 到全部 GEMM 待研究。

## 12. 推荐阅读材料

- [OCP MX v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Rouhani et al. 2023 - Microscaling data formats for deep learning](https://arxiv.org/abs/2310.10537)
- [Halko, Martinsson, Tropp 2011 - Finding structure with randomness](https://doi.org/10.1137/090771806)
- [Tseng et al. 2024a - QuIP#](https://proceedings.mlr.press/v235/tseng24a.html)
- [Croci et al. 2022 - Stochastic rounding analysis](https://doi.org/10.1098/rsos.211631)
- [Peng et al. 2023 - FP8-LM training](https://arxiv.org/abs/2310.18313)
- [Kumar et al. 2025 - Scaling laws for precision](https://openreview.net/forum?id=wg1PCg3CUP)
- [Wang et al. 2025 - Concurrent FP4 LLM training work](https://arxiv.org/abs/2501.17116)
- [NVIDIA Blackwell architecture brief](https://resources.nvidia.com/en-us-blackwell-architecture)
- [Agarwal et al. 2024 - HadaCore kernel](https://arxiv.org/abs/2412.08832)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [Microsoft microxcaling emulation library](https://github.com/microsoft/microxcaling)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)

## 13. 总结

这篇 paper 的 elegant 之处在于把"训练动力学要求"和"硬件效率"统一到同一个 mathematical framework:

- 训练动力学要求 **unbiased** gradient → 用 SR
- 训练动力学要求 **low variance** gradient → 用 RHT
- RHT 通过 orthogonal 性质不影响期望值 (因 $H^T H = I$) → 与 SR 完美兼容
- Blockwise RHT 实现在 memory-bound regime, 几乎无开销

整个 recipe 就是个"刚好够"的 minimal intervention — 既消除了 4-bit training 的 numerical ills, 又几乎没有 extra overhead, 把 4-bit training 从 "不可用" 推到 "比 FP8 还快 1.3×"。这种 minimal 但 fundamental 的改进, 是 system + ML theory 结合的典范。
