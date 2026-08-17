---
source_pdf: OrbitQuant Data-Agnostic Quantization for Image and.pdf
paper_sha256: a0eb086f58de7b277367ff073b8b356c892efa306ddd328114f188040519aca3
processed_at: '2026-08-06T01:26:29-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OrbitQuant 人话版

## 一句话说清楚

**DiT 的 activation 分布乱跑，追不上怎么办？别追了，转个圈，它就变乖了。**

---

## 真正的问题在哪

你训了一个 FLUX 或 Wan，想把它压到 4-bit 部署。LLM 量化那套成熟方案直接搬过来，发现崩了。为什么？

LLM 的 activation 有个"脾气"：outlier channel 是固定的那几个，你量一次、记住它、把它的 scale 塞进 weight，就一劳永逸了。

DiT 不一样。**同一个 layer 的 activation，在不同 denoising step、不同 prompt、conditional vs unconditional 分支下，分布完全不一样**。你用 100 张图校准出的 scale，换个 prompt 就不对了；你用 t=500 量出来的统计，t=100 就失效了。

所以现有所有 DiT PTQ 方法都得为每个新 checkpoint、每个新分辨率、每个新 modality 重新搞 calibration set。这就很烦。

---

## OrbitQuant 的骚操作

### 核心 trick：别在原空间量化，转到一个"数学上保证分布固定"的空间去量

想象你有个 d 维向量 x。你做两件事：

1. **除以它的 norm**：x → x/||x||，变成单位向量。norm 单独存起来（就一个标量）。
2. **乘一个随机正交矩阵**：x̃ → Φ x̃。

做完这两步，神奇的事情发生了：**不管 x 原来长什么样，转完之后每个 coordinate 的分布都是同一个固定的函数 f_d**，只跟维度 d 有关。

这个 f_d 大概长这样：均值 0、方差 1/d 的高斯。也就是说，转完之后所有 coordinate 都变成"差不多大的小数值"，没有 outlier 了。

**为什么？** 因为随机正交旋转会把任何向量的能量均匀摊到所有 coordinate 上。原来有个 channel 是 100 倍大，转完之后这个 100 被劈成 d 份，每份都正常了。

---

### 这意味着什么？

既然转完之后分布是固定的，那你就可以**离线把 quantizer 设计好，永远不用改**。

具体来说，你对 f_d 这个已知分布，用 Lloyd-Max 算法算出最优的 2^b 个量化点。这个 codebook 一劳永逸，所有 timestep、所有 prompt、所有 layer、甚至 image 和 video DiT，只要维度 d 一样，**共用同一个 codebook**。

这就完全跳过了 calibration。你不是在"用数据估计分布"，你是"数学上知道分布是什么"。

---

## 但有个工程问题：随机旋转很贵

真正的 Haar random rotation 是个 d×d 的稠密矩阵，每来一个 token 要做 O(d²) 的矩阵乘法。d=12288 的话，这个开销比量化本身还大。

### RPBH：廉价版的"几乎一样好"的旋转

OrbitQuant 用了个叫 **RPBH（Randomized Permuted Block-Hadamard）** 的东西：

$$\Pi_d = \text{blkdiag}(\mathbf{H}_h \mathbf{D}_1, \ldots, \mathbf{H}_h \mathbf{D}_{d/h}) \cdot \mathbf{P}_\pi$$

人话拆解：
- **H_h**：Walsh-Hadamard 矩阵，元素只有 ±1/√h，有快速算法 O(h log h)
- **D_i**：随机 ±1 的对角矩阵，打符号
- **P_π**：一个随机置换矩阵，把 coordinate 顺序打乱
- 整个东西是 block diagonal 的，每个 block 独立做小 Hadamard

**为什么需要那个置换 P_π？** 如果只有 block Hadamard，每个 block 内部自己混，block 之间不交流。万一有个 outlier 落在某个 block 里，它出不去。P_π 先把所有 coordinate 随机洗牌，让每个 block 都能分到一点 outlier 的能量，这样转完之后所有 coordinate 的方差都接近 1/d。

**关键**：这个置换是**均匀随机**的，不需要看数据。论文证明了一个概率 bound——对任何输入，随机置换后每个 coordinate 的方差都在 (1/d)(1±ρ) 附近，ρ 由输入的最大 coordinate 占比决定。只要没有某个 coordinate 独吞 99% 的 norm，就稳。

---

## 最妙的工程 trick：旋转在 layer 里自己 cancel 了

你可能会担心：runtime 要对每个 activation 做旋转，这开销怎么办？

答案：**把旋转折进 weight 里，它在 matmul 内部自己抵消了**。

具体：
- Offline：W' = W · Π^⊤（weight 吸收旋转的转置）
- Online：x' = Π · x（activation 做正向旋转）
- 推理时算的是：W' · x' = W · Π^⊤ · Π · x = W · x

因为 Π 是正交的，Π^⊤ · Π = I。**两个旋转在矩阵乘法里自己抵消了，你永远不需要做 inverse rotation**。

Runtime 唯一多出来的就是 activation 进 layer 前的那一次 forward RPBH。这个用 fast Walsh-Hadamard transform 做，O(d log h)，非常快。

---

## 跟 TurboQuant 的关系

这个"随机旋转 + 分布固定的 marginal + Lloyd-Max codebook"的 idea 来自 TurboQuant（2025 年的 KV-cache 压缩工作）。

但 TurboQuant 是个**独立的 codec**：你量化一个 vector，要存 code、存 norm，用的时候 rotate back 还原。它是 vector-level 的量化。

OrbitQuant 把这个 idea **搬进了 transformer 的 linear layer 内部**，利用 rotation 的 computational invariance 让它在 matmul 里 cancel。这就不需要 rotate back，可以直接在 rotated basis 做矩阵乘法。这是从 "vector quantizer" 到 "weight-activation quantizer" 的关键一步。

---

## Lloyd-Max vs Uniform Grid：为什么低比特差距巨大

QuaRot、SmoothQuant 这些方法用的是 uniform grid（等距量化点）。对 N(0, 1/d) 这种高斯分布，uniform grid 在尾部浪费 levels，在中心不够密。

Lloyd-Max 是**对给定分布 MSE-optimal** 的 scalar quantizer。对高斯分布，它会把量化点在 0 附近放得密、尾部放得稀。

这就是为什么 W2A4 时：
- QuaRot / SmoothQuant / ViDiT-Q 全部崩溃（分数 ≈ 0，生成噪声）
- OrbitQuant 在 FLUX.1-schnell 还有 0.604，接近 FP16 的 91%

3-bit uniform grid 的 8 个 levels 没法合理覆盖高斯的形状；4-bit Lloyd-Max 的 16 个 levels 加上 non-uniform spacing 可以贴住密度。**低比特下 codebook 的形状比比特数更重要**。

---

## 实验里最 striking 的几个点

1. **W4A4 在 Z-Image-Turbo 上超过 FP16**（0.767 vs 0.754）。量化噪声起了 regularization 作用，不是 bug 是 feature。

2. **W2A4 是所有 baseline 的死亡线**，只有 OrbitQuant 活着。其他方法在 W2A4 全部输出噪声。

3. **一个 recipe 跑 image + video**：FLUX、Z-Image、Wan、CogVideoX 用完全相同的 pipeline，不需要 per-model 调参。证明 codebook 是"dimension-aware"而非"model-aware"。

4. **RPBH 在低比特比 dense Haar 还强**（Table 3: W2A4 RPBH 0.595 vs Haar 0.591）。原因是 random permutation 帮助 spread outlier，反而比纯随机旋转更 robust。而且 RPBH 快 26 倍。

5. **在某些维度上超过 QAT**（Table 7: Wan 2.1-1.3B 的 Overall Consistency 超过 QVGen）。calibration-free 的 distributional codebook 在 invariant space 里是 MSE-optimal 的，而 QAT 在原空间 fine-tune 不一定能找到更好的。

---

## 最大的 limitation

Lloyd-Max 是 non-uniform codebook，但现有 GPU 的 integer tensor core 只支持 uniform grid 的 GEMM。所以现在只能 fake quantization（dequant 回 BF16 再算 matmul），没有真正的低比特加速。

这就是论文最后说的 future work：需要 lookup-table GEMM kernel（像 LUT-GEMM 那样），在 GEMM 的 prologue 里 gather codebook 的 centroid，直接在低比特格式下算。

**方法本身没有 limitation，是硬件生态的 limitation**。等 LUT-GEMM 这类 kernel 成熟了，OrbitQuant 就能真正提速。

---

## 我的直觉总结

OrbitQuant 的哲学是：**不要试图建模一个乱动的分布，用数学变换把它送进一个不动的地方**。

这跟 LayerNorm 的精神类似，但更激进——LayerNorm 是 normalization，OrbitQuant 是 rotation。Normalization 改变几何但保留坐标轴；rotation 把能量重新分配到所有坐标轴，让 outlier 消失在 aggregate 里。

更深层的 insight：**random rotation 是 worst-case optimal 的 incoherence operator**。不管你的输入有多 adversarial 的 outlier 结构，随机旋转都能把它 spread 掉。这是 compressed sensing 和 Johnson-Lindenstrauss 的老智慧，但第一次被干净地用在了 diffusion transformer 量化上。

这种"用数学 invariant 消灭工程麻烦"的 taste，是好工作的标志。

---

# OrbitQuant: Data-Agnostic Quantization for DiTs 深度讲解

## 1. 问题动机：为什么 DiT 量化比 LLM 量化更难

先讲清楚这篇论文要解决的根本问题。LLM 的 PTQ 已经相当成熟，但 DiT 的 quantization 有几个本质的难点。

**LLM activation 的"幸福状态"**：LLM decoding 是 memory-bound，weight-only quantization 就能提速。即使要量化 activation，LLM 的 outlier channel 是相对稳定的，一次 calibration pass 就能捕获，可以 scale 进 weight（SmoothQuant）或 rotate 掉（QuaRot, SpinQuant, QuIP#）。

**DiT activation 的"灾难状态"**：DiT 有几个致命特点：
- **Timestep drift**：activation 分布随 denoising timestep t 剧烈变化。t≈0 时是噪声主导，t≈T 时接近 clean data，这两个 regime 的 activation 统计天差地别。
- **Prompt dependence**：不同 prompt 产生完全不同的 activation range。
- **CFG branch asymmetry**：classifier-free guidance 的 conditional 和 unconditional 分支 statistics 不同。
- **Compute-bound**：DiT 是 compute-bound（不像 LLM decoding），weight-only quantization 无速度收益，必须 weight + activation 一起量化才有意义。

所以所有现有的 DiT PTQ 方法（SVDQuant, PTQ4DiT, AdaTSQ, ViDiT-Q, LRQ-DiT）都需要为每个 checkpoint、每个分辨率、每个 modality 重新收集 calibration set 并 re-fit。这就让"训练完直接部署低比特"这件事变得非常繁琐。

OrbitQuant 的核心 question：**能不能彻底跳过 calibration，让 quantizer 对所有 input 通用？**

Project page: https://saurabhcantina.github.io/orbitquant/

---

## 2. 核心 Intuition：旋转把"漂移的分布"变成"固定的分布"

这是整篇论文最 elegant 的 idea，需要仔细体会。

### 2.1 球面均匀分布的坐标 marginal

假设我们有一个 unit vector x̃ ∈ ℝ^d（即 ||x̃||₂ = 1），如果对它做一个 Haar-random orthogonal rotation Φ_d，那么旋转后的每个坐标 (Φ_d x̃)_k 的 marginal 分布是固定的，**与 x̃ 本身无关**。

这个 marginal 的解析形式是：

$$f_d(t) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d-1)/2)} (1-t^2)^{(d-3)/2}, \quad t \in [-1, 1]$$

变量含义：
- **t**：rotated 后某个坐标的取值，取值范围 [-1, 1]（因为单位向量）
- **d**：向量维度
- **Γ(·)**：Gamma 函数，Γ(n) = (n-1)! 对正整数

这个 f_d 其实就是 (d-1)-维单位球面上均匀分布的第 k 个坐标的 marginal density。这是一个 Beta(1/2, (d-1)/2) 分布的变形。

**关键观察**：当 d ≥ 64 时，f_d 被 N(0, 1/d) 紧紧近似。也就是说，rotated 后的每个坐标 ≈ 一个均值为 0、方差为 1/d 的高斯分布，且不同坐标几乎独立。

这个 idea 来自 TurboQuant（NeurIPS 2025, arXiv:2504.19874, https://arxiv.org/abs/2504.19874），原本是用于 KV-cache 压缩的 standalone vector quantizer。

### 2.2 为什么这对 DiT 是"完美武器"

DiT activation 漂移的本质是：不同 timestep/prompt 下的 activation 分布不同，传统方法需要为每个分布 fit 一个 quantizer。

但 OrbitQuant 的思路是：
1. **Normalize**：x → x̃ = x/||x||₂，把 norm 单独存起来（一个标量 s）。
2. **Rotate**：x̃ → Φ_d x̃。无论 x̃ 是什么，rotated 后的坐标 marginal 都是 f_d。
3. **Quantize**：因为 marginal 固定，所以**一个 Lloyd-Max codebook 服务所有 timestep、prompt、layer，甚至 image 和 video DiT**。

这就是"calibration-free"的根本来源：不需要从数据中估计任何 statistics，因为旋转后的分布是数学上确定的，只依赖维度 d。

---

## 3. 数学与方法细节

### 3.1 整体架构（Figure 2 解读）

Figure 2 展示了三个关键点：

1. **左**：原始 DiT activation 在不同 timestep 和 CFG branch 下分布漂移，calibration 不通用。
2. **中**：RPBH rotation Π_d 把 raw activation 映射到"well-behaved coordinates"，并 fold 进 weights，使得 W'x' ≈ Wx（rotation 在 layer 内部 cancel）。
3. **右**：rotated coordinates 都集中在固定 marginal f_d ≈ N(0, 1/d)，一个 Lloyd-Max codebook C_{d,b} 服务所有。

### 3.2 Offline Weight Quantization

公式 (4)-(6)：

$$\mathbf{W}' = \mathbf{W}\,\Pi_d^\top$$

把 weight 矩阵旋转到 shared basis。然后对每一行做 magnitude-direction 分解：

$$r'_i = \|\mathbf{w}'_i\|_2, \quad \tilde{\mathbf{w}}'_i = \mathbf{w}'_i / r'_i$$

变量：
- **w'_i**：W' 的第 i 行
- **r'_i**：第 i 行的 ℓ₂ 范数（magnitude，存 BF16）
- **w̃'_i**：单位方向向量

因为 Π_d 与 w_i 独立采样，所以每个 w̃'_i 的坐标也服从 f_d。用 Lloyd-Max 量化方向：

$$\hat{\mathbf{W}}' = \mathrm{diag}(\mathbf{r}') \cdot \hat{Q}_{b_w}^{(d)}(\tilde{\mathbf{W}}')$$

存储开销：r' 是 m 个 BF16 标量 = 16m bits，相比 b_w·m·d bits 的 quantized direction < 0.3%，可忽略。

### 3.3 Online Activation Quantization

公式 (7)-(8)：

$$\mathbf{x}' = \Pi_d \mathbf{x}, \quad s = \|\mathbf{x}'\|_2, \quad \tilde{\mathbf{x}}' = \mathbf{x}'/(s+\varepsilon)$$

$$\hat{\mathbf{x}}' = s \cdot \hat{Q}_{b_a}^{(d)}(\tilde{\mathbf{x}}')$$

变量：
- **x**：incoming activation
- **x'**：rotated activation
- **s**：per-token scalar norm
- **ε = 10⁻¹⁰**：防止 padding token 的零 norm 触发除零

**关键的 cancellation trick**：

$$\mathbf{W}'\mathbf{x}' = \mathbf{W}\Pi_d^\top \Pi_d \mathbf{x} = \mathbf{W}\mathbf{x}$$

因为 Π_d 是正交矩阵，Π_d^⊤ Π_d = I。weight 吸收 Π_d^⊤，activation 应用 Π_d，两个旋转在 matmul 内部 cancel，runtime 只剩 forward rotation on activation。

这是与 TurboQuant 的核心区别：TurboQuant 是 quantize-dequantize codec，每个 vector 旋转回去再 reconstruct；OrbitQuant 把 rotation fold 进 weight，**无需 inverse rotation**，直接在 rotated basis 做 matmul。

### 3.4 RPBH: Randomized Permuted Block-Hadamard

公式 (9)：

$$\Pi_d = \mathrm{blkdiag}(\mathbf{H}_h \mathbf{D}_1, \ldots, \mathbf{H}_h \mathbf{D}_{d/h}) \cdot \mathbf{P}_\pi$$

变量：
- **blkdiag(·)**：block diagonal 拼接
- **H_h**：h×h Walsh-Hadamard 矩阵（元素 ±1/√h）
- **D_i**：Rademacher sign diagonal（每个对角元素 ±1，随机）
- **P_π**：uniform random permutation matrix
- **h**：block size，取 d 的最大 power-of-two 因子（在所有 evaluated model 中 h ∈ {128, 512, 1024, 2048, 4096}）
- **d = kh**：总维度，k 个 block

**为什么不用 dense Haar rotation？**
- Dense Haar Φ_d 的 transform cost 是 O(d²) per token，对单张 image 的所有 token 和 timestep，这 dominates activation 量化总 cost。
- RPBH 通过 per-block Fast Walsh-Hadamard Transform + permutation gather，只需 O(d log h)。
- Storage 是 O(d) 而不是 O(d²)（一个 sign vector + 一个 permutation array）。
- 而且 Walsh-Hadamard 矩阵只在 power-of-two 维度存在，block 形式可以构造在任意 d（如 CogVideoX-2B 的 d=1920）。

**为什么需要 permutation P_π？**
如果没有 P_π（即 Block-RHT），每个 block-Hadamard 只在 block 内部混合，若某个 outlier 集中在一个 block，它永远不能 spread 到其他 block，rotated marginal 在那个 block 内会严重偏离 f_d。

P_π 把 coordinates 均匀 spread 到所有 block，使得每个 block 以高概率收到均衡的 input mass。

**Crucially，P_π 是 uniform random 而不是 data-dependent**。这是与 DuQuant（按 outlier magnitude 排序）、PeRQ（fit permutation 平衡 per-block mass）的本质区别——OrbitQuant 不需要 calibration。

### 3.5 Proposition 1: Universal Variance Concentration

$$\mathrm{Var}(z_i \mid \pi) \in \left[\frac{1-\rho}{d}, \frac{1+\rho}{d}\right], \quad \rho = d\,\mu_\infty \sqrt{\frac{1}{2h}\log\frac{4k}{\delta}}$$

变量：
- **z_i**：Π_d x̃ 的第 i 个 coordinate
- **π**：随机 permutation
- **ρ**：concentration 参数
- **μ_∞**：||x̃||_∞²（输入向量最大坐标的平方）
- **h**：block size
- **k**：block 数量 = d/h
- **δ**：confidence level（failure probability ≤ δ）

**直觉**：ρ 保持 small 除非有某个 coordinate 拿走了 norm 的 outsized share。这个 bound 保证 rotated marginal 紧贴 N(0, 1/d)，所以 Lloyd-Max codebook near-optimal。

证明的 sketch（在 supplementary A）：
- **Lemma 1 (Per-block incoherence)**：对 fixed partition，random Rademacher sign 使每个输出 coordinate 是 mean-zero Rademacher sum，方差 = M_j/h ≤ 1/h。Hoeffding + union bound 给出 ||z||_∞ ≤ √(2 log(4d/δ)/h)。
- **Lemma 2 (Mass balancing)**：random permutation 使每个 block 的 mass M_j ≈ 1/k ± μ_∞ √(h/2 log(4k/δ))。这里用 Hoeffding's bound for sampling without replacement。
- **Combine**：两个 lemma union bound 后，每个 coordinate 的方差 ∈ (1/d)(1±ρ)，且 ||Π_d x̃||_∞ ≤ √(2(1+ρ)/d · log(4d/δ))。

Remark 2 还提到 Berry-Esseen 给出 quantitative Gaussian approximation，只要没有 coordinate 拿走 outsized norm，每个 rotated coordinate 在 distribution 上（不仅 variance）接近 N(0, 1/d)。Figure 3 用 Kolmogorov-Smirnov distance 实证确认。

---

## 4. 实验结果解读

### 4.1 Image Generation (GenEval, Table 1)

关键结果：

| Model | Bit | OrbitQuant | Best Baseline | FP16 |
|---|---|---|---|---|
| FLUX.1-schnell | W4A4 | **0.703** | AdaTSQ 0.680 | 0.664 |
| FLUX.1-schnell | W2A4 | **0.604** | 所有 baseline ≈ 0.001 | 0.664 |
| FLUX.1-dev | W4A4 | **0.633** | AdaTSQ 0.618 | 0.667 |
| FLUX.1-dev | W2A4 | **0.319** | 所有 baseline ≈ 0 | 0.667 |
| Z-Image-Turbo | W4A4 | **0.767** | AdaTSQ 0.762 | 0.754 |
| Z-Image-Turbo | W2A4 | **0.319** | 所有 baseline ≈ 0 | 0.754 |

**两个 takeaway**：
1. **W4A4 几乎 lossless**：OrbitQuant 在 FLUX.1-schnell 和 Z-Image-Turbo 上甚至超过 FP16（FLUX.1-schnell: 0.703 vs 0.664；Z-Image-Turbo: 0.767 vs 0.754）。这点很 striking，说明 quantization 的 noise 起到某种 regularization 作用，与 EfficientDM 等 QAT 工作的观察一致。
2. **W2A4 是 baseline 崩溃的红线**：所有 rotation/smoothing baseline（QuaRot, SmoothQuant, ViDiT-Q）在 W2A4 全部 collapse 到 ≈0（生成 noise）。OrbitQuant 仍保留 FLUX.1-schnell 的 0.604，几乎接近 FP16 的 91%。这印证了 distributional codebook 在低比特的鲁棒性。

### 4.2 Video Generation (VBench, Table 2 & Table 5)

| Model | Bit | OrbitQuant Overall Consistency | Best Baseline |
|---|---|---|---|
| Wan 2.1-1.3B | W4A6 | 24.35 | SVDQuant 23.26 |
| Wan 2.1-1.3B | W4A4 | 23.86 | SVDQuant 21.91 |
| CogVideoX-2B | W4A6 | 24.55 | ViDiT-Q 24.03 |
| CogVideoX-2B | W4A4 | 23.86 | SVDQuant 22.89 |
| Wan 14B | W4A4 | 26.15 | QuaRot 25.41 |
| HunyuanVideo | W4A4 | 22.83 | DVD-Quant 25.68 |

**关键点**：OrbitQuant 用一个 recipe 同时跑 image 和 video DiT，不需要 per-modality tuning。在 HunyuanVideo 上虽然输给 DVD-Quant（专门为 video 设计），但在 imaging quality, motion smoothness, scene 维度仍领先，说明 distributional codebook 有跨架构的 transfer 能力。

### 4.3 Latency 和 Memory (Figure 5)

- 在 FLUX.1-dev 上，OrbitQuant 比 QuaRot 快 1.17×，比 ViDiT-Q 快 1.40×（image）。
- 原因：fixed shared-codebook nearest-centroid lookup 比 per-token dynamic uniform quantization 便宜。
- Peak memory：image 上 OrbitQuant 匹配 unquantized footprint；video 上 20.3 GB（vs QuaRot 19.3 GB, ViDiT-Q 23.2 GB），因为 lookup materialize 了 index 和 gather tensor。
- 注意这些都是 fake quantization（dequant 到 BF16 再 matmul），不是真正低比特 GEMM 的 speedup。

### 4.4 Rotation Ablation (Table 3)

| Rotation | W4A4 | W3A3 | W2A4 | Latency (s) |
|---|---|---|---|---|
| Haar | 0.696 | 0.669 | 0.591 | 11.65 |
| Full RHT | 0.691 | 0.672 | 0.587 | 0.452 |
| Block-RHT | 0.678 | 0.642 | 0.558 | 0.381 |
| RPBH (ours) | 0.690 | **0.674** | **0.595** | 0.451 |

**Insight**：
- W4A4 时四种 rotation 都在 noise 范围内。
- 低比特时（W3A3, W2A4），RPBH 最强，超过 dense Haar！这是惊人的——structured rotation 反而比 dense random rotation 强，原因是 random permutation 帮助 spread outlier。
- Block-RHT 明显比 RPBH 差，证明 permutation 是关键。
- Haar 比 structured rotation 慢 26×。

### 4.5 AdaLN Bit-width (Figure 6)

AdaLN modulation 不能 fold 进 weight（因为它是 timestep-dependent scale-and-shift），所以保持 INT4 RTN。若降到 W2，FLUX.1-dev/schnell 崩溃，Z-Image-Turbo 仍 robust。这是 model-dependent 的 robustness。

AdaLN 占 27% weights，保持 BF16 会把 FLUX 压缩从 4× 降到 2.21×，所以量化到 INT4 是必要的折衷。

---

## 5. 与相关工作的关系网络

### 5.1 LLM Rotation-Based Quantization 谱系

- **SmoothQuant** (https://arxiv.org/abs/2211.03889): 把 activation outlier scale 进 weight，无需 rotation。
- **QuaRot** (https://arxiv.org/abs/2404.00456): Hadamard rotation fold 进 weight，让 activation outlier-free。OrbitQuant 的直接 baseline。
- **SpinQuant** (https://arxiv.org/abs/2410.06164): learned rotation，更优的 incoherence。
- **QuIP#** (https://arxiv.org/abs/2402.04391): Hadamard incoherence + lattice codebook，达到 2-bit LLM 量化。
- **DuQuant** (https://arxiv.org/abs/2406.01792): dual transformation + calibrated permutation。
- **PeRQ** (https://arxiv.org/abs/2601.22347): fit permutation 平衡 per-block mass。
- **OSTQuant** (https://arxiv.org/abs/2501.13987): orthogonal + scaling transformation。
- **FlatQuant** (https://arxiv.org/abs/2410.09426): flatness matters for LLM quantization。

OrbitQuant 与这些工作的核心区别：**permutation 是 uniform random 而不是 calibrated**，且 codebook 是 distribution-derived 而不是 uniform grid。

### 5.2 Vector Quantization 谱系

- **PolarQuant** (https://arxiv.org/abs/2504.12137, AISTATS 2026): KV-cache 在 polar coordinates 量化，random preconditioning 后角度分布已知。
- **TurboQuant** (https://arxiv.org/abs/2504.19874): Cartesian coordinates + dense Haar + Beta-marginal Lloyd-Max。OrbitQuant 直接继承这个 idea。
- 区别：TurboQuant 是 standalone KV-cache codec（要 rotate back），OrbitQuant 把 rotation fold 进 weight（在 layer 内 cancel）。

### 5.3 DiT Quantization 谱系

**Calibration-based**:
- **PTQ4DiT** (https://arxiv.org/abs/2404.05762): block reconstruction 平衡 salient channel。
- **Q-DiT** (https://arxiv.org/abs/2402.08047): per-channel calibration + mixed precision。
- **ViDiT-Q** (https://arxiv.org/abs/2406.02540): per-channel calibration + mixed precision，image 和 video DiT 通用。
- **SVDQuant** (https://arxiv.org/abs/2411.05007): 低秩 branch 吸收 outlier（4-bit diffusion）。
- **AdaTSQ** (https://arxiv.org/abs/2602.09883): timestep-sensitive precision allocation。
- **LRQ-DiT** (https://arxiv.org/abs/2508.03485): calibrated DuQuant-style rotation。
- **PermuQuant** (https://arxiv.org/abs/2605.09503): calibrated channel reordering。
- **S²Q-VDiT** (https://arxiv.org/abs/2508.04016): Hessian-aware saliency + token-level distillation。

**Data-free**:
- **DVD-Quant** (https://arxiv.org/abs/2505.18663): data-free video DiT，但 per-model machinery。
- **ConvRot** (https://arxiv.org/abs/2512.03673): calibration-free group-wise regular Hadamard + uniform grid 在 FLUX。

**QAT**:
- **Q-DM** (https://arxiv.org/abs/2302.04312)
- **EfficientDM** (https://arxiv.org/abs/2401.09191): efficient QAT fine-tuning。
- **QVGen** (https://arxiv.org/abs/2505.11497): quantized video generative model，push 量化极限。
- **RobuQ** (https://arxiv.org/abs/2509.23582): ternary weights on ImageNet DiTs。

OrbitQuant 的位置：**第一个真正 data-agnostic 且 image/video 通用的 weight-activation quantizer，能 push 到 W2A4 仍可用**。

---

## 6. 一些 Intuition 的 Build-up

### 6.1 为什么"旋转后 marginal 固定"这么 powerful？

想象一个 d=3072 的 activation vector。原始 activation 在某个 timestep 可能有几个巨大的 outlier channel，比如 100x 于其他 channel。传统 uniform quantization 要么 clip outlier 丢信息，要么 scale 整个 vector 让小值 quantize 到 0。

随机旋转后，那个 outlier 的能量被均匀 spread 到所有 d 个 coordinate，每个 coordinate 都变成一个 mean-zero、variance 1/d 的小值。所有 outlier 都消失了，分布变成可预测的 N(0, 1/d)。

这就让"per-timestep calibration"这件事完全 unnecessary——你不是在 chase 一个 moving target，你是把 target 旋转到一个 invariant space。

### 6.2 为什么 Lloyd-Max 比 uniform grid 强？

Lloyd-Max 是 MSE-optimal 的 scalar quantizer，给定分布 f，求 2^b 个 reconstruction levels 让 E[(X - q(X))²] 最小。

对 N(0, 1/d) 这种轻尾分布，Lloyd-Max 的 codebook 在 0 附近 dense，尾部 sparse。这与实际 rotated activation 的密度匹配。

Uniform grid 假设 activation 均匀分布，但 N(0, 1/d) 是高斯的，在尾部浪费 quantization levels。所以 W2A3 时 QuaRot 等用 uniform grid 的方法完全崩溃——3-bit 的 8 个 levels 没法合理覆盖 N(0, 1/d) 的形状。

Lloyd-Max 在 W2A4 仍可用，是因为 4-bit 的 16 个 levels 加上 Lloyd-Max 的 non-uniform spacing 可以贴住 N(0, 1/d) 的密度。

### 6.3 为什么 shared codebook 不会牺牲精度？

正常理解：不同 layer 的 activation 分布不同，应该有不同的 codebook。

但 OrbitQuant 的 trick 是：rotated + normalized 后，所有 layer 的 marginal 都是 f_d，**只依赖维度 d**。同一个 model 里同维度的 layer（如所有 attention projection d=3072）共用一个 codebook。

这就是为什么一个 codebook 能跨 timestep、prompt、layer 甚至 image 和 video。代价是：你需要 storage norm（每行 BF16），但这是 < 0.3% overhead。

### 6.4 为什么 random permutation 就够了？

Proposition 1 给出 probabilistic guarantee：对任意 fixed unit vector x̃，random Π_d 使每个 coordinate 方差 ∈ (1/d)(1±ρ)，ρ 由 μ_∞ = ||x̃||_∞² 决定。

只要 x̃ 没有 coordinate 拿走 outsized norm share，ρ 保持 small。如果有（比如 ReLU 后的 sparse activation），random permutation 也会均匀 spread 到所有 block，每个 block 收到均衡 mass。

对比 calibrated permutation（DuQuant 按 outlier magnitude 排序，PeRQ fit mass-balancing permutation）：calibration 的优势是 deterministic 保证，但需要数据。OrbitQuant 用 uniform random + probabilistic bound，不需要数据，且实证（Table 3）证明 RPBH 比 Block-RHT（无 permutation）强，甚至超过 dense Haar。

---

## 7. Limitations 和我的思考

论文自己的 Limitations 部分（Section D）提到：
1. **Runtime rotation cost**：虽然 RPBH 比 Haar 快 26×，仍是 0.451 s/image on H100 at 1024²。
2. **No native low-bit GEMM**：Lloyd-Max 是 non-uniform codebook，现有 integer tensor core 只支持 uniform grid。当前 fake quantization（dequant 到 BF16 再 matmul）。
3. **Path forward**：LUT-GEMM（https://arxiv.org/abs/2402.04896）等 lookup-table GEMM kernel 可以 compute on non-uniform codebook，作者计划做 fused kernel 在 GEMM prologue gather centroids。

我自己的几点思考：

1. **Distribution shift 的解构**：这篇论文的核心 insight 实际上把"activation distribution shift"这个抽象问题分解为"norm shift"+"direction shift"，然后 random rotation 让 direction 在 invariant space 中有 fixed marginal，只剩 norm 一个标量 per-token 即可。这种 decomposition 在其他领域（如 batch norm、layer norm）也有类似思想，但 OrbitQuant 用 rotation 而非 normalization 解决问题，更 elegant。

2. **与 Layer Norm 的关系**：DiT 内部已经有 layer norm / RMS norm。为什么这些 norm 不够？因为 norm 在 attention 之前/之后做，但 attention 内部的 QK^T、softmax 之后的 activation 仍有 outlier。OrbitQuant 是在 quantization 边界做，更精细。

3. **Lloyd-Max 的非均匀性 vs. integer tensor core 的矛盾**：这是当前 quantization 领域的根本矛盾。Uniform grid 对硬件友好但 suboptimal；non-uniform codebook optimal 但需要 LUT。SqueezeLLM、GPTVQ、LUT-GEMM 等都在 attack 这个问题。OrbitQuant 的下一步融合 LUT-GEMM-style kernel 是关键。

4. **Transferability 的 implications**：一个 codebook 跨 image 和 video DiT 通用，意味着 quantizer 是"dimension-aware"而非"model-aware"。这暗示了一个未来范式：library of codebooks indexed by (d, b)，任何新模型直接 lookup，不需要 calibration。这是真正 production-friendly 的方向。

5. **Random rotation 与 incoherence 的关系**：这个 idea 在 compressed sensing、Johnson-Lindenstrauss、random projection 等领域有深厚根基（Ailon-Chazelle FJLT, https://arxiv.org/abs/0905.0026; Tropp's improved analysis, https://arxiv.org/abs/1102.3963）。OrbitQuant 把这个数学工具用在 quantization 上，是漂亮的 cross-pollination。

6. **W2A4 的可用性**：在 FLUX.1-schnell 上 W2A4 仍有 0.604 Overall（接近 FP16 的 91%），这是非常 promising 的结果。意味着 2-bit weight 在 diffusion 模型上是 feasible 的，与 LLM 上 QuIP# 的 2-bit 结果呼应。

7. **与 QAT 的比较**：Table 7 显示 OrbitQuant 在 Wan 2.1-1.3B 的 Subject Consistency、Scene、Overall Consistency 上甚至超过 QVGen 等 QAT 方法。这说明 calibration-free 的 distributional codebook 有时比 fine-tuned quantization 还强，因为 codebook 在 invariant space 中是 MSE-optimal 的，而 QAT 在原空间 fine-tune 不一定能找到更好的 quantizer。

---

## 8. 总结

OrbitQuant 的核心 contribution 是把 DiT activation 量化问题 reformulate 为 distributional codebook problem：

- **问题原形**：activation 分布随 timestep/prompt drift，需要 calibration。
- **问题变形**：在 rotated + normalized basis 中，marginal 是数学上确定的 f_d，与 input 无关。
- **解决方案**：RPBH rotation（fast + 与 dense Haar 同等 invariant）+ Lloyd-Max codebook（MSE-optimal for f_d）。
- **工程实现**：rotation fold 进 weight，在 matmul 中 cancel，runtime 只剩 forward RPBH。
- **结果**：W4A4 几乎 lossless，W2A4 仍可用，跨 image 和 video DiT 无 per-modality tuning。

这是 quantization 领域一次漂亮的 idea synthesis：把 random projection 的 incoherence property（compressed sensing）+ Lloyd-Max scalar quantization（classic signal processing）+ rotation folding（LLM quantization like QuaRot）+ TurboQuant's distributional codebook 结合，解决了一个看似需要 calibration 的根本问题。

参考链接：
- Project page: https://saurabhcantina.github.io/orbitquant/
- TurboQuant: https://arxiv.org/abs/2504.19874
- QuaRot: https://arxiv.org/abs/2404.00456
- SpinQuant: https://arxiv.org/abs/2410.06164
- QuIP#: https://arxiv.org/abs/2402.04391
- SmoothQuant: https://arxiv.org/abs/2211.03889
- SVDQuant: https://arxiv.org/abs/2411.05007
- ViDiT-Q: https://arxiv.org/abs/2406.02540
- PTQ4DiT: https://arxiv.org/abs/2404.05762
- Q-DiT: https://arxiv.org/abs/2402.08047
- AdaTSQ: https://arxiv.org/abs/2602.09883
- LRQ-DiT: https://arxiv.org/abs/2508.03485
- PermuQuant: https://arxiv.org/abs/2605.09503
- S²Q-VDiT: https://arxiv.org/abs/2508.04016
- DVD-Quant: https://arxiv.org/abs/2505.18663
- ConvRot: https://arxiv.org/abs/2512.03673
- QVGen: https://arxiv.org/abs/2505.11497
- RobuQ: https://arxiv.org/abs/2509.23582
- EfficientDM: https://arxiv.org/abs/2401.09191
- Q-DM: https://arxiv.org/abs/2302.04312
- PolarQuant: https://arxiv.org/abs/2504.12137
- DuQuant: https://arxiv.org/abs/2406.01792
- PeRQ: https://arxiv.org/abs/2601.22347
- OSTQuant: https://arxiv.org/abs/2501.13987
- FlatQuant: https://arxiv.org/abs/2410.09426
- LUT-GEMM: https://arxiv.org/abs/2402.04896
- FJLT (Ailon-Chazelle): https://arxiv.org/abs/0905.0026
- Tropp SRHT analysis: https://arxiv.org/abs/1102.3963
- FLUX.1: https://blackforestlabs.ai/
- Wan 2.1: https://arxiv.org/abs/2503.20314
- CogVideoX: https://arxiv.org/abs/2408.06072
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Z-Image: https://arxiv.org/abs/2511.22699
