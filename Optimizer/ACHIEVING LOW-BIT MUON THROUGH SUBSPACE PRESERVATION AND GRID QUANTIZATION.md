---
source_pdf: ACHIEVING LOW-BIT MUON THROUGH SUBSPACE PRESERVATION AND GRID QUANTIZATION.pdf
paper_sha256: a7869a3839f1c520917cf913c8be0f83f52f35adc1a0dc87b8865edc8ec9ab9a
processed_at: '2026-07-18T00:23:28-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# 这篇 paper 在做什么：一个 1-paragraph TL;DR

作者把 Muon optimizer 的 state 成功压到 4-bit (4-bit-Muon-GRASP = GRid And Subspace Preserving)。核心 insight 是 Muon 的 Newton-Schulz orthogonalization 是一个 nonlinear spectral mapping,会**不对称地**放大 top singular subspace 上的 quantization noise(40×)而对 residual subspace 只放大 5×。所以他们对 top subspace 用 8-bit 保精度(因为 low-rank,overhead 可忽略),对 residual 用 4-bit,再加一个 grid quantization 处理两个方向都出现的 outlier pattern。在 LLaMA 130M/350M/1.1B pretraining 和 Qwen2.5-7B fine-tuning 上几乎不掉点,optimizer state 相比 fp32-Muon 省 28%、相比 fp32-AdamW 省 48%。代码在 https://github.com/wuhuaijin/lowbit-Muon 。

---

# 背景:Muon optimizer 的 update rule 和它为什么 memory-friendly

Karpathy 你应该熟 Keller Jordan 的 Muon (https://kellerjordan.github.io/posts/muon/),它在 2024 年底火起来,被 Kimi-K2 (https://arxiv.org/abs/2502.16982, https://arxiv.org/abs/2507.20534) 等大模型采用,Liu et al. 2025 报告 Muon 把 AdamW 的 sample efficiency 翻倍。Update rule 三步:

$$
\mathbf{M}_t = \mu \mathbf{M}_{t-1} + \nabla \mathcal{L}_t(\mathbf{W}_{t-1}) \tag{1}
$$

- $\mathbf{M}_t \in \mathbb{R}^{m \times n}$:first moment buffer,初始化为 0
- $\mu$:momentum coefficient(常见 0.95)
- $\nabla \mathcal{L}_t$:当前 step 的 gradient
- $\mathbf{W}_{t-1}$:当前 weight

$$
\mathbf{O}_t = \text{Newton-Schulz}_p(\mathbf{M}_t, T) \tag{2}
$$

- $p$:polynomial degree(=5)
- $T$:iteration steps(=5)
- $\mathbf{O}_t$:近似 orthogonalized update

$$
\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t \mathbf{O}_t \tag{3}
$$

NS iteration 用一个 5 阶 polynomial 来近似 $\text{sign}(\Sigma)$ 等价的 orthogonalization。形式上:

$$
\mathbf{X}_0 = \frac{\mathbf{M}_t}{\|\mathbf{M}_t\|_F}, \quad \mathbf{X}_k = a \mathbf{X}_{k-1} + b (\mathbf{X}_{k-1} \mathbf{X}_{k-1}^\top) \mathbf{X}_{k-1} + c (\mathbf{X}_{k-1} \mathbf{X}_{k-1}^\top)^2 \mathbf{X}_{k-1} \tag{4}
$$

- $\mathbf{X}_0$:Frobenius-normalized moment
- $\mathbf{X}_k$:第 k 步迭代结果
- $a=3.4445, b=-4.7750, c=2.0315$:让 polynomial $f(x) = ax + bx^3 + cx^5$ 在 $x \approx 1$ 处有一个 attracting fixed point,把 normalized singular value 都推到 1
- 直觉:如果 $\mathbf{M}_t = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 是 SVD,那 NS iteration 近似输出 $\mathbf{U} \cdot \text{sign}(\boldsymbol{\Sigma}) \cdot \mathbf{V}^\top$,即把所有 singular value 都映射到 1。等价于一个 spectral "trust-region"——所有 update direction 被等同对待,避免 top singular direction over-dominate。

**Memory advantage**:AdamW 要存 first moment + second moment(2× weight size),Muon 只存 first moment,所以省 50%。但是 LLM 训练 memory 还是吃紧,5B 模型用 fp32 Muon optimizer state 已经 40GB+,所以进一步 compress 是有意义的。

---

# 核心 challenge:为什么直接用 4-bit AdamW 的方法压 Muon 不行

Naive baseline(论文叫 4-bit-Muon-base):直接把 group quantization (Dettmers et al. 2021, https://arxiv.org/abs/2110.02861) 套到 $\mathbf{M}_t$ 上。问题在于 Muon 的 update rule 不是 element-wise 的(SGD/AdamW 都是 element-wise),中间有一个矩阵乘法 $\mathbf{X}\mathbf{X}^\top\mathbf{X}$,这让 quantization noise 在 spectral domain 被放大。

## 实验现象(paper §3.1, Figure 1)

定义 relative error:

$$
\text{RE}(\mathbf{A}, \mathbf{B}) = \frac{\|\mathbf{A} - \mathbf{B}\|_F}{\|\mathbf{B}\|_F} \tag{9}
$$

- $\mathbf{A}$:量化或扰动后的矩阵
- $\mathbf{B}$:原始矩阵
- $\|\cdot\|_F$:Frobenius norm

把 $\mathbf{M}_t$ 量化到 4-bit 得到 $\hat{\mathbf{M}}_t$,在 quantization 前后做 NS iteration:

| 阶段 | RE |
|---|---|
| $\text{RE}(\hat{\mathbf{M}}_t, \mathbf{M}_t)$ | 0.07 |
| $\text{RE}(\text{NS}(\hat{\mathbf{M}}_t), \text{NS}(\mathbf{M}_t))$ | **1.78** |

量化本身只引入 7% 的 noise,NS iteration 之后变成 178%——放大了 **25×**。这就是为什么直接套 4-bit AdamW 的方法在 Muon 上炸了。

## 为什么 NS iteration 会放大 error?(intuition)

NS iteration 是 spectral operation。考虑 $\mathbf{M} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$,简化分析假设 quantization 只扰动 $\boldsymbol{\Sigma}$,得到 $\hat{\boldsymbol{\Sigma}} = \boldsymbol{\Sigma} + \boldsymbol{\Delta}$。NS 把每个 singular value 通过 $f(\sigma_i)$ 映射:

$$
\text{output}_i \approx f(\sigma_i)
$$

那么第 i 个 singular value 的扰动:

$$
\Delta f(\sigma_i) \approx f'(\sigma_i) \cdot \Delta \sigma_i
$$

放大因子就是 $|f'(\sigma_i)|$。Muon 用的 polynomial $f(x) = 3.4445x - 4.7750x^3 + 2.0315x^5$ 在不同 $\sigma$ 上的导数差异很大,所以 noise 在 spectral domain 上被**非均匀放大**。

Figure 2 的实验进一步确认:增加 NS iteration 步数 T 或 polynomial degree p **不能 fix** error,反而让 error 更大。这说明问题不是 "iteration 不够收敛",而是 quantization 让矩阵的 singular spectrum 系统性偏移(Figure 1c→1d 显示量化后所有 singular value 都增大),最终收敛到错误的 fixed point。

---

# Top singular subspace 的 differential error amplification(paper §3.1, Table 1)

作者的核心发现:NS iteration 的 noise amplification 主要发生在 **top singular subspace**。

把 $\mathbf{M}$ 做 SVD,拆成两部分:

$$
\mathbf{M}_{\text{top}} := \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top, \quad \mathbf{M}_{\text{res}} := \mathbf{M} - \mathbf{M}_{\text{top}} \tag{10}
$$

- $\mathbf{U}_k \in \mathbb{R}^{m \times k}$, $\mathbf{V}_k \in \mathbb{R}^{n \times k}$:top-k 左/右 singular vectors
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{k \times k}$:top-k singular values
- $k$:截断 rank

然后分别量化 $\mathbf{M}_{\text{top}}$ 和 $\mathbf{M}_{\text{res}}$,测量 NS 前后的 RE:

| $k$ | $\text{RE}(\mathbf{M}_{\text{top}})$ NS 前 → NS 后 | $\text{RE}(\mathbf{M}_{\text{res}})$ NS 前 → NS 后 |
|---|---|---|
| 64 | 0.08 → **3.31** (40×) | 0.09 → 0.47 (5×) |
| 128 | 0.08 → 2.42 (30×) | 0.09 → 0.63 |
| 256 | 0.08 → 1.76 (22×) | 0.09 → 0.35 |
| 512 | 0.08 → 1.26 (16×) | 0.09 → 0.42 |

Top subspace 的 quantization noise 被放大 **16~40 倍**,residual 只放大 5 倍。

## 为什么?直觉上三种解释可以叠加

1. **Fixed-point curvature**:normalized $\mathbf{M}_t$ 的 top singular value 接近 1,落在 NS polynomial 的 fixed-point neighborhood,这里的 $f'(\sigma)$ 大(曲率高),所以 $\Delta \sigma$ 被放大。Residual 的 $\sigma$ 小,落在 $f(x) \approx ax$ 的 linear region,而且 $f$ 把 small $\sigma$ 推到 0,noise 被 "压扁"。

2. **NS iteration 把所有 singular value 都推向 1,所以 residual 的 small singular value 经过 NS 后贡献变小;但 top singular value 已经在 1 附近,扰动会保持**。换句话说,NS 是一个 "spectral high-pass",而 quantization noise 在 top subspace 的 energy 占比本来就高。

3. **Quantization 引入 systematic bias**:Figure 1d 显示量化后所有 singular value 都变大。这是因为 group quantization 用 max normalize, outlier 会把整个 group 的 scale 拉大,大部分值被压缩到小的 range,rounding error 系统性偏向 0 或者向外。这种 bias 在 top singular direction 上表现最明显。

---

# 方法 1:Subspace Preservation(paper §3.2)

观察:top subspace 是 low-rank 的(rank k 远小于 min(m,n)),所以即使存 8-bit,total overhead 也很小。

具体做法:
- 用 Power Iteration(下一节)近似 top singular vectors $\mathbf{P}_t \in \mathbb{R}^{m \times k}, \mathbf{R}_t \in \mathbb{R}^{n \times k}$,使得 $\mathbf{P}_t \mathbf{R}_t^\top \approx \mathbf{M}_{\text{top}}$
- $\mathbf{P}_t, \mathbf{R}_t$ 用 **8-bit** 量化(精度足够,放大不显著)
- Residual $\mathbf{M}_{\text{res}, t} = \mathbf{M}_t - \mathbf{P}_t \mathbf{R}_t^\top$ 用 **4-bit** 量化
- 内存计算:$k(m+n) \cdot 1\text{byte} + mn \cdot 0.5\text{byte}$。如果 $k = \frac{1}{16}\min(m,n)$,那 8-bit 部分占总内存的 $\frac{2k}{\min(m,n)} \cdot 2 = \frac{1}{4}$ 量级但实际是 $\frac{k(m+n) \cdot 1}{mn \cdot 0.5 + k(m+n) \cdot 1} \approx$ 几个百分点的 overhead

关键 trick:为什么要保留 residual?paper §4.3 ablation Figure 7c 做了实验——如果只保留 top subspace,discard residual,即使 rank = 1/2,training accuracy loss > 2%。原因是 NS iteration 把 residual 的小 singular value 也 amplify 到 1,这部分在 update 中不可忽略。所以 Muon 不允许直接 low-rank 近似(对比 GaLore,https://arxiv.org/abs/2403.03507,GaLore 用 low-rank gradient projection 是因为 AdamW 不做 spectral 操作)。

---

# 方法 2:Power Iteration with warm-start(paper §3.2, Algorithm 1)

直接做 SVD 拿 top singular vectors 太贵($O(mn \min(m,n))$)。作者用 **1-step Power Iteration + warm-start**:

```
function PowerIter(B, Q):  # B = M_t, Q = warm-start
    P = B @ Q              # m×k
    P = QR_orthogonalize(P)  # m×k, 正交化
    R = B.T @ P            # n×k
    return P, R
```

Warm-start:从 $\mathbf{R}_{t-1}$ 出发,先做 column normalize 得到 $\mathbf{Q}_t = \text{ColumnNormalize}(\mathbf{R}_{t-1})$,然后跑 1 步 power iteration。

为什么 1 步够?直觉:$\mathbf{M}_t = \mu \mathbf{M}_{t-1} + \nabla \mathcal{L}_t$,而 momentum 让 $\mathbf{M}_t$ 和 $\mathbf{M}_{t-1}$ 高度相似(top singular subspace 漂移很小)。Ablation Figure 8 显示 1-step 已经达到 RE ≈ 0.01,2-step、3-step 提升微乎其微。

这个 trick 来自 PowerSGD (Vogels et al. 2019, https://arxiv.org/abs/1905.12726) 和 DION (Ahn et al. 2025, https://arxiv.org/abs/2504.05295),都用了 warm-start 的 power iteration 来 cheaply 追踪 top singular subspace。PowerSGD 是用来 compress gradient communication,GaLore 用来 project gradient 到 low-rank subspace,这里用来 cheaply factorize optimizer state——idea 类似但 application 不同。

Power iteration cost breakdown(paper Table 8):
- 130M rank=1/16: 46ms (power iteration) vs 98ms (NS iteration) → 总 optimizer 时间 < 0.5s,远小于 forward/backward
- 1.1B rank=1/16: 242ms (power) vs 378ms (NS) → 总增加 0.2~0.6s/step

Rank 1/16 是个 sweet spot,power iteration 比 NS iteration 还快;rank 1/4 时 QR decomposition 变贵,power iteration 反而比 NS 慢。

---

# 方法 3:Grid Quantization(paper §3.3, Figure 3)

观察:Muon 的 moment tensor $\mathbf{M}_t$ 的 outlier pattern **在两个维度都出现**——既有 per-channel outlier 又有 per-token outlier。传统 group quantization(per-channel 或 per-token)只能在一个维度 normalize,outlier 会污染整个 group 的 scale。

Grid quantization:把矩阵分成 $s \times s$ 的 block(默认 $s=128$),对每个 block 同时算 row scale 和 column scale:

$$
\text{scale}_{r_i} = \max_{r_1 \leq j \leq r_2} |x_{i,j}|, \quad \text{scale}_{c_j} = \max_{c_1 \leq i \leq c_2} |x_{i,j}| \tag{12}
$$

- $\text{scale}_{r_i}$:第 $i$ 行的 max absolute value(在 block 内)
- $\text{scale}_{c_j}$:第 $j$ 列的 max absolute value(在 block 内)
- $r_1, r_2, c_1, c_2$:block 的边界 index

然后每个 element 用 **两者中较小的一个**做 normalization:

$$
\mathcal{N}_{\text{grid}}(x_{i,j}) = \frac{x_{i,j}}{\min(\text{scale}_{r_i}, \text{scale}_{c_j})} \tag{13}
$$

## Intuition:为什么用 min?

考虑一个 outlier 在 $(i^*, j^*)$ 位置,值很大。它会拉大 $\text{scale}_{r_{i^*}}$ 和 $\text{scale}_{c_{j^*}}$。但 element $(i^*, j)$ for $j \neq j^*$ 用 $\min(\text{scale}_{r_{i^*}}, \text{scale}_{c_j}) = \text{scale}_{c_j}$(假设 $c_j$ 没有 outlier),所以不受 $(i^*, j^*)$ 的 outlier 污染。同理 $(i, j^*)$ 用 $\text{scale}_{r_i}$。

所以 grid quantization 等价于给每个 element 一个 **adaptive scale**:如果它所在的 row 有 outlier 但 col 没有,用 col scale;反之用 row scale。Outlier 只在它真正的"十字交叉"位置影响精度,不会扩散。

Memory overhead:scale 数量翻 2×,但 scale 占总内存的比例小(block size 128 → scale bytes = 2×128×1byte per 128×128 block,vs 128×128×0.5byte quantized values,占比 2/64 ≈ 3%)。

Ablation Figure 7b:grid quantization vs group quantization 直接压 moment matrix(不用 subspace preservation),grid 把 accuracy loss 减半。

---

# 完整 Algorithm(paper Algo. 1)

```
Init: W, lr η, momentum μ, weight decay λ, rank k, quantizer QUANT, dequantizer DEQUANT

# step 0: random init Q_0, M_0 = 0

# step t:
1. Dequant: M_res_{t-1} ← DEQUANT(M_res_{t-1}^q)
            P_{t-1}, R_{t-1} ← DEQUANT(P_{t-1}^q, R_{t-1}^q)
2. Reconstruct: M_{t-1} ← M_res_{t-1} + P_{t-1} R_{t-1}^T
3. Warm-start: Q_t ← ColumnNormalize(R_{t-1})
4. Update moment: M_t ← μ M_{t-1} + ∇L_t(W_{t-1})
5. Power iteration: P_t, R_t ← PowerIter(M_t, Q_t)  # 1-step
6. Residual: M_res,t ← M_t - P_t R_t^T
7. Quantize: M_res,t^q ← QUANT_4(M_res,t)
            P_t^q, R_t^q ← QUANT_8(P_t, R_t)
8. Orthogonalize: O_t ← Newton-Schulz(M_t)  # 在 full M_t 上,不是 quantized!
9. Update: W_t ← W_{t-1} - η_t (O_t + λW_{t-1})
```

注意一个 subtle 点:line 8 的 orthogonalization 是在 **full precision** 的 $\mathbf{M}_t$ 上做,quantization 只用于 storage。所以 NS iteration 看到的是 dequantized 的 $\mathbf{M}_{t-1}$,加上当前 step 的 gradient。

---

# 实验

## Pretraining(paper §4.1, Table 2, 3)

LLaMA architecture (RMSNorm + SwiGLU) on SlimPajama,三个尺寸:130M / 350M / 1.1B。Optimizer 比较:fp32-Muon / 8-bit-Muon / 4-bit-Muon-base / 4-bit-Muon-GRASP。

Downstream zero-shot (HellaSwag / ARC-c / ARC-e / boolQ / OBQA / PIQA / SciQ):

| Model | Optimizer | Avg |
|---|---|---|
| 130M | fp32-Muon | 41.8 |
| 130M | 4-bit-Muon-base | 41.9 |
| 130M | 4-bit-Muon-GRASP | 41.9 |
| 350M | fp32-Muon | 44.6 |
| 350M | 4-bit-Muon-base | 43.7 (-0.9) |
| 350M | 4-bit-Muon-GRASP | 44.5 (-0.1) |
| 1.1B | fp32-Muon | 48.0 |
| 1.1B | 4-bit-Muon-base | 47.6 (-0.4) |
| 1.1B | 4-bit-Muon-GRASP | 48.2 (+0.2) |

4-bit-Muon-base 在大模型上明显掉点,4-bit-Muon-GRASP 几乎无损,1.1B 上甚至略高(随机种子 noise)。

## Memory & Time(paper Table 3, Figure 6)

| Model | Optimizer | time (s/step) | mem (GB) | PPL |
|---|---|---|---|---|
| 1.1B | fp32-Muon | 61.3 | 13.22 | 12.48 |
| 1.1B | 8-bit-Muon | 61.5 | 13.22 | 12.46 |
| 1.1B | 4-bit-Muon-base | 61.5 | 10.54 | 12.76 |
| 1.1B | 4-bit-Muon-GRASP | 61.9 | **10.14** | 12.48 |

Time overhead 几乎为 0(0.6s/step),memory 省 23%。在 5B 模型上 vs fp32-AdamW 省 48%,vs fp32-Muon 省 28%。是当前最 memory-efficient 的 low-bit optimizer。

## Fine-tuning 7B(paper §4.2, Table 4)

Qwen2.5-7B (general SFT on tulu-3) 和 Qwen2.5-7B-Math (NuminaMath CoT) 上做 SFT,distributed 实现(verl framework,partition-shape quantization + global subspace preservation)。

| Model | Origin | fp32-SFT | 4bit-base | 4bit-GRASP |
|---|---|---|---|---|
| Qwen2.5-7B Avg (MMLU/HumanEval/MBPP/GSM8K) | 68.2 | 76.3 | 75.9 | 76.2 |
| Qwen2.5-7B-Math Avg (MATH/Minerva/Olympiad) | 35.0 | 44.4 | 44.5 | 45.1 |

GRASP 在 math reasoning 上甚至比 fp32 略好(45.1 vs 44.4),可能是 low-bit 起到 slight regularization 作用(类似 8-bit optimizer 报告过的 implicit regularization)。

## CPU offloading 对比(paper Table 6, 7)

5B 模型:CPU offload 的 fp32-Muon optimizer time = 17.93s/step(GPU↔CPU transfer 主导),而 4-bit-Muon-GRASP (rank 1/16) = 1.93s/step + memory 只占 1/5。CPU offload 在大模型上 time overhead 太大,low-bit 是更好的 trade-off。

---

# Ablation 关键点(paper §4.3)

1. **Rank selection**(Figure 7a):rank 从 1/64 → 1/2,training loss gap 从 ~1% 缩小到 0。Rank 1/16 是 reasonable sweet spot。

2. **Residual 必要性**(Figure 7c):只保留 top subspace,discard residual,即使 rank=1/2,loss 仍 > 2%。Muon 的 NS iteration 不允许直接 low-rank approximation。

3. **Grid vs Group quantization**(Figure 7b):grid quantization 把 accuracy loss 减半。

4. **Power iteration steps**(Figure 8):1-step vs 2-step vs 3-step 的 approximation error 都 ≈ 0.01,1-step 足够。warm-start 是关键。

5. **INT4 vs FP4**(Appendix B.4, Figure 9 left):两者 training curve 几乎相同,FP4 需要额外 lookup table 略慢。INT4 简单 round() 即可,工程更友好。

6. **Learning rate sensitivity**(Figure 9 right):lr ∈ {3e-4, 6e-4, 1e-3, 3e-3} 都能 converge 到类似 level。

---

# 联想和 critical thinking

## 跟相关工作的关系

1. **PowerSGD** (Vogels et al. 2019, https://arxiv.org/abs/1905.12726):warm-start power iteration 这个 trick 的 origin。PowerSGD 用来 compress gradient communication,这里用来 compress optimizer state storage。一个有趣的差别:PowerSGD 在 distributed 设置下用 error feedback 来修正 compression bias,Muon-GRASP 没有显式 error feedback。是否需要?作者没讨论,但 momentum 本身已经是一种 error feedback($\mathbf{M}_t = \mu \mathbf{M}_{t-1} + g_t$ 携带历史)。

2. **GaLore** (Zhao et al. 2024, https://arxiv.org/abs/2403.03507):也是用 power iteration + warm-start 找 top singular subspace,然后 project gradient 到 low-rank。差别:GaLore 是 project gradient 来节省 AdamW state,Muon-GRASP 是 factorize Muon state 来分别量化。GaLore 之所以能直接用 low-rank(不需要 residual)是因为 AdamW 是 element-wise,不放大 spectral error;Muon 因为有 NS iteration,必须保留 residual。

3. **8-bit Optimizer** (Dettmers et al. 2021, https://arxiv.org/abs/2110.02861):block-wise dynamic quantization,核心 insight 是 outlier 在 row 方向,所以 per-row block quantization。Muon 的 outlier 在两个方向都出现,所以需要 grid quantization。

4. **4-bit AdamW** (Li et al. 2023, NeurIPS):用 finer-grained quantization + 移除 zero point。直接套到 Muon 上是 4-bit-Muon-base,作者证明不够。

5. **4-bit Shampoo** (Wang et al. 2024):Shampoo 也是 spectral optimizer(用 L^{-1} R^{-1} precond),4-bit Shampoo 也是相关工作。Comparison 这篇 paper 没做。

6. **DION** (Ahn et al. 2025, https://arxiv.org/abs/2504.05295):distributed Muon,作者在 7B fine-tuning 实验里参考了 DION 的实现。

7. **Adafactor** (Shazeer & Stern 2018):factorize second moment 用 row/col statistics,Muon-GRASP 的 grid quantization 也有点这个味道——两个维度的 statistic 都用上。

## 一些可能延伸的方向

1. **Rank 自动选择**:paper §6 提到目前 rank 是手动选的(1/16)。可能可以根据 singular value decay 自动决定——spectral entropy 或者 cumulative energy ratio threshold。也可以参考 GaLore 的 schedule(定期 update subspace)。

2. **2-bit 或 1-bit Muon**:paper 只做了 4-bit。如果 top subspace 8-bit,residual 用 2-bit 或 1-bit(binary)呢?需要更激进的 quantization scheme,可能需要 LFQ/PWLF 类的 lookup-free 量化。

3. **Distributed 设置的 communication cost**:大模型训练 optimizer state 是 sharded 的(FSDP/ZeRO),subspace preservation 是 global 的,跨 device 的 power iteration 通信开销如何?作者在 7B 用 verl 实现 distributed,但没有详细 profiling。

4. **Activation quantization 结合**:paper §6 提到可以和 activation reduction 方法结合。如果 backward 也用 4-bit,memory 还能再降一截。这和 "1-bit Adam"(deterra et al.)或者 "Sophia" 之类的 training-time quantization 工作可以连起来。

5. **为什么 quantization 让 singular value 系统性变大?**(Figure 1d) 这个现象 paper 提到但没解释清楚。我猜可能是 group quantization 的 rounding behavior——max normalize 后,大部分值集中在 0 附近,round 到 0,但偶尔 round 出来的非零值相对 scale 较大,导致 singular spectrum 整体上移。这值得理论分析,可能和 randomized rounding vs nearest rounding 的 bias 有关。

6. **NS iteration 的替代品**:如果用更便宜的 orthogonalization approximator(比如 Chebyshev iteration,或者 polar decomposition via Newton),quantization noise 的放大 pattern 会不同。这是个 open direction。

7. **Muon 本身的更便宜实现**:NS iteration 5 步 × 5 阶 polynomial,在 GPU 上是 matmul-heavy 的。能不能用更少的 iteration(2-3 步)就达到足够的 orthogonalization?Liu et al. 2025 (https://arxiv.org/abs/2502.16982) 在 scale Muon 时已经在调这个。

8. **Muon 之外的应用**:paper §C 提到可以拓展到 CV、combinatorial optimization、bioinformatics、generative modeling(diffusion / autoregressive / molecular generation)。Low-bit optimizer 在这些 memory-constrained 场景都有用。

9. **Quantization schedule**:目前是 uniform 4-bit/8-bit throughout training。WSD 之类的 learning rate schedule 启发:training 不同阶段 quantization 需求不同(warmup 期 gradient 大、不稳定;stable 期平稳;decay 期 fine-tune),是否可以做 dynamic bit-width schedule?早期 8-bit、后期 4-bit?

10. **Top subspace 的物理意义**:training 后期,top singular direction 代表 "informed direction"(gradient 一致指向的方向)。保住它就是保住 informative update signal;residual 是 "noise"。这跟 LM itself 的 attention head rank collapse 现象异曲同工——training 推进,signal 集中到少数 direction。

## Critical points

- 1.1B 是 paper 的最大 pretraining 规模,7B 只做 fine-tuning。在更大规模(70B+)的 pretraining 上效果未知。Muon 在 Kimi-K2 上是 trillion scale,但 GRASP 在那个 scale 上是否还成立?
- Distributed 实现的细节没充分展开,但工程上是关键。
- NS iteration 的 5 步 × 5 阶 polynomial 是 fixed 的,作者没探索这部分的 quantization-aware design。如果专门设计一个 quantization-robust 的 NS polynomial,error amplification 可能更小。

---

# 总结

这篇 paper 是 "把 Muon 压到 4-bit" 的 first work,核心贡献是发现 NS iteration 对 top singular subspace 的 quantization noise 放大远大于 residual subspace,然后针对性地用 8-bit 保 top、4-bit 压 residual + grid quantization 处理 two-way outlier,几乎无损。工程上 Triton kernel + verl distributed 实现,实测 28% memory reduction vs fp32-Muon,几乎 free lunch。Open question 在更大 scale 的 pretraining、distributed 通信、和更激进 bit-width 上。

代码:https://github.com/wuhuaijin/lowbit-Muon
Muon 原文:https://kellerjordan.github.io/posts/muon/
Muon scalable (Liu et al. 2025):https://arxiv.org/abs/2502.16982
Kimi-K2:https://arxiv.org/abs/2507.20534
8-bit Optimizer:https://arxiv.org/abs/2110.02861
4-bit Adam:https://proceedings.neurips.cc/paper_files/paper/2023/hash/3f7e96f64d33a9f33f5b2d8b45c2cfab-Abstract-Conference.html
PowerSGD:https://arxiv.org/abs/1905.12726
GaLore:https://arxiv.org/abs/2403.03507
DION:https://arxiv.org/abs/2504.05295
4-bit Shampoo (Wang et al. 2024):https://papers.nips.cc/paper_files/paper/2024/hash/4b2e9d8f4e0f5d2a8e0e7c4e3a4c6a5b-Abstract-Conference.html
Adafactor:https://arxiv.org/abs/1804.04235
Shampoo:https://arxiv.org/abs/1802.09468
