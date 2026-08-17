---
source_pdf: Rethinking Shrinkage Bias in LLM FP4 Pretraining.pdf
paper_sha256: 78016e3014ef7a4ffd1424d6f0c303588d7c2dcb771fa229d858ed863a263c88
processed_at: '2026-08-11T23:37:41-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

---

## 一句话总结

大家都在用 E2M1 这个 4-bit 格式训练大模型,但这篇 paper 发现:**E2M1 的数字排列方式天生有"缩水" bug**,训练越深越严重,而且大家普遍用的 RHT 旋转技巧反而让这个 bug 更糟。解决办法是换成 E1M2 这种均匀排列的格式。

---

## 故事背景

现在训练 LLM 太贵了,大家都想用 4-bit 来算。NVIDIA Blackwell、AMD MI350 都主推 E2M1 格式。E2M1 长这样:

$$\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$$

你看,前面 0 到 2 之间每个间隔 0.5,到了 2 到 3 变成 1,3 到 4 也是 1,4 到 6 变成 2。**间距不均匀**。

这看起来没什么大不了,但 paper 发现这是个坑。

---

## 核心发现:Shrinkage Bias

### 直觉解释

想象你站在数轴上某个刻度 $q_i$,左边邻居距离 $\ell_i$,右边邻居距离 $r_i$。RTNE 规则是"就近舍入"。

如果右边比左边远($r_i > \ell_i$),那你这个 bin 里"被舍入到 $q_i$"的数字,平均下来会偏小——因为右边那块区域更大,采到更多比 $q_i$ 大的数字,但它们都被舍入成 $q_i$,相当于把大数字往下拽。

公式:

$$\mathbb{E}[\text{误差}] = \frac{\ell_i - r_i}{2}$$

右宽 > 左宽,结果为负,这就是 Shrinkage Bias。

### 具体例子

E2M1 里 $q_i = 2$ 这个点:
- 左邻居 1.5,距离 0.5
- 右邻居 3,距离 1
- 期望误差 = $(0.25 - 0.5) / 2 = -0.125$

也就是落在 2 这个 bin 的数字,平均被砍掉 0.125。均匀网格(E1M2 / INT4)左右等距,这个误差永远是 0。

---

## 为什么这事儿严重:乘性累积

单看一层 GEMM,误差可能就 $\delta_k \approx 0.01$ 这种小数,看起来无所谓。

但 K 层叠起来,衰减是**乘性**的:

$$\prod_{k=1}^{K}(1 - \delta_k) \approx \exp\left(-\sum_{k=1}^{K} \delta_k\right)$$

假设每层衰减 1%,80 层之后:

$$0.99^{80} \approx 0.45$$

信号被砍掉一半。这跟零均值噪声完全不同——零均值噪声会相互抵消,Shrinkage Bias 只会越叠越深,指数级衰减。

这就是为什么 FP4 训练到深层就崩——不是某个 quantizer 写得差,是格式本身把信号一点点吃掉了。

---

## RHT 的反效果:最反直觉的部分

RHT(Random Hadamard Transform)本来是个好东西——通过随机正交旋转,把 outlier 能量摊到所有维度上,让 tensor 分布变"扁",提高 codebook 利用率。

大家一直以为"RHT + E2M1"是黄金组合。但 paper 发现这俩其实是**冲突**的。

### 原因

RHT 之前,tensor 有 outlier,动态范围大,E2M1 的宽范围(0.5 到 6)正合适。

RHT 之后,outlier 被摊平了,大多数值都挤到中等幅值附近。这时候:
- E2M1 的宽动态范围用不上了,白浪费
- 更要命的是,数据被推到 E2M1 **最不对称的 bin**(2 和 4 那两个点),Shrinkage Bias 反而被放大

paper 实测:对 outlier 重的 tensor(linear_fc2/fwd_x),RHT 前 E2M1 比 E1M2 好(21.90 vs 19.94 dB),RHT 后 E1M2 反超(23.19 vs 20.00 dB)。**格式排名直接反转**。

均匀网格的 E1M2 则完美匹配 RHT 后的扁平分布,把提升的 codebook 利用率转化为更高的 SQNR。

---

## UFP4 Recipe

既然问题出在 E2M1 不均匀,那就换均匀网格。UFP4 的做法:

1. **格式换成 E1M2/INT4-style 均匀网格**
2. **RHT 应用到全部三个 GEMM 路径**:forward(fwd_y)、data gradient(bwd_dx)、weight gradient(bwd_dw)
3. **Stochastic rounding 只在 dY 上用**

为什么 SR 只在 dY?因为 dY 是梯度入口,SR 能保住梯度期望。其他 operand 上 SR 反而增加方差,paper 实测在 E2M1 上加更多 SR 反而损失变大。

为什么 E2M1 recipe 不敢把 RHT 用到全部三路?因为在 E2M1 下,旋转 fwd_y 和 bwd_dx 会把数据推进最坏的 bin,损失更大。NVFP4 recipe 只敢在 bwd_dw 上用 RHT,这就是被 Shrinkage Bias 逼的妥协。UFP4 换均匀网格后,这个限制就解除了。

---

## 实验结果

### Q1:RHT 是否改变首选 grid?

会。对表现良好的 tensor,RHT 中性。对 outlier 重的 tensor,**RHT 直接反转格式排名**,E2M1 从领先变落后。

### Q2:训练 loss 改善多少?

BF16-relative loss error:

| Model | E2M1 baseline | UFP4 (E1M2) | 改善 |
|---|---|---|---|
| Dense 1.5B | 1.26% | 0.97% | -23% |
| MoE 7.9B | 2.36% | 1.85% | -22% |
| MoE 124B | 1.73% | 1.39% | -20% |

跨尺度都稳定有效。

### Q3:Scale law 怎么样?

从 10M 到 324M MoE 训练,E1M2 曲线始终低于 E2M1,且 gap 随 compute 缩小,不会随规模放大恶化。

### Q4:组件 ablation

Dense 1.5B 上:

| RHT scope | Loss | $\Delta$ |
|---|---|---|
| No RHT | 1.8920 | 0 |
| bwd_dw only | 1.8872 | -0.0048 |
| fwd_y + bwd_dw | 1.8856 | -0.0064 |
| bwd_dx + bwd_dw | 1.8891 | -0.0029 |
| **Full RHT + SR (UFP4)** | **1.8808** | **-0.0112** |

Full RHT 比 no RHT 多减 0.011 loss。SR 在 full RHT 基础上再减 0.0046。

**有意思的细节**:fwd_y + bwd_dw 比 bwd_dx + bwd_dw 更有用。前向路径每个 token 都走一次,影响所有后续层;反向 bwd_dx 误差虽然也级联,但梯度信号已经过非线性,分布更驯服。

### Q4 续:E2M1 能否通过限幅模拟均匀网格?

把 E2M1 的 max 限制到 2.0,只保留 {0, 0.5, 1, 1.5, 2}。结果在 Dense 1.5B 和 MoE 7.9B 上都比 E2M1 baseline 差。**限制范围救不了 E2M1,动态范围丢了,utilization 也丢了**。原生 E1M2/INT4 是不可替代的。

### Q5:RHT 融合效率

把 RHT 和 quantization 融合到一个 kernel:
- SM90:1.06× standalone quant latency
- SM100:1.07× standalone quant latency

不融合:1.62×(SM90)、1.41×(SM100)。融合后开销几乎可忽略。

---

## 这篇 paper 的深层洞察

### 洞察一:几何不对称是系统偏差的根源

非均匀量化网格的 bin 不对称,在 RTNE 下天然产生负偏差。这不是噪声,是**系统性信号衰减**。

### 洞察二:outlier 处理和格式选择是耦合的

RHT 这种 outlier mitigation 技术本身没错,错在和 E2M1 搭配。RHT 把分布推向"扁平",而 E2M1 的优势在"动态范围",两者目标错位。换成均匀网格,RHT 的优势才能完全发挥。

paper 原话:"full-RHT training is not inherently harmful; rather, the issue lies in the mismatch between E2M1 and the post-RHT tensor regime."

### 洞察三:硬件设计需要反思

现在所有 FP4 硬件都围绕 E2M1 设计。这篇 paper 说:这不够,应该把 E1M2/INT4-style 均匀网格也作为 first-class training primitive。E2M1 适合 inference 和 raw outlier-heavy tensor,但训练应该有均匀网格选项。

特别提到 Huawei Ascend 960 + HiFloat4(均匀 S1P2 格式)可能天然适合 UFP4。

---

## 一句话再总结

E2M1 数字排列不均匀 → RTNE bin 左右不对称 → 落进去的数字平均被砍一点 → K 层叠乘指数衰减 → 训练崩。RHT 本来想帮忙,结果把数据推进最不对称的 bin,火上浇油。换 E1M2 均匀网格,bin 左右对称,偏差归零,RHT 可以放心用到全部三路 GEMM,训练 loss 显著降低。

就这么个事,但分析得很透。

---

# Rethinking Shrinkage Bias in LLM FP4 Pretraining 深度解读

Andrej 你好，这篇论文非常有意思。它揭示了 FP4 训练中一个被忽视的格式几何问题。让我来详细拆解。

---

## 1. 论文核心洞察：从数据格式几何出发

当前 FP4 训练硬件生态（NVIDIA Blackwell NVFP4、AMD MI350、MXFP4）几乎都默认使用 E2M1 格式（2 exponent bits + 1 mantissa bit），总共只有 16 个可表示值。

这篇论文的核心发现是：**E2M1 的非均匀量化网格本身就包含系统性的 Shrinkage Bias**，这是一种由 RTNE（Round-to-Nearest-Even）bin 几何不对称引起的负向期望误差。

### 1.1 E2M1 的可表示值

E2M1 非负幅值集合是：
$$G_+ = \{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$$

注意这里面的间距是不均匀的：
- 0 到 2 之间间距是 0.5
- 2 到 3 间距是 1
- 3 到 4 间距是 1  
- 4 到 6 间距是 2

这些间距过渡点（如 $q_i = 2$ 和 $q_i = 4$）就是 Shrinkage Bias 的根源。

---

## 2. Shrinkage Bias 的几何起源

### 2.1 数学推导

对于 codebook 中的某个内部量化级别 $q_i$，其 RTNE bin 定义为：

$$\mathcal{B}_i = \left(\frac{q_{i-1} + q_i}{2}, \frac{q_i + q_{i+1}}{2}\right)$$

其中：
- $q_{i-1}$: 前一个量化级别（下标 $i-1$ 表示前一个）
- $q_{i+1}$: 后一个量化级别
- $\ell_i = \frac{q_i - q_{i-1}}{2}$: bin 的左半宽度（left width）
- $r_i = \frac{q_{i+1} - q_i}{2}$: bin 的右半宽度（right width）

假设 bin 内密度局部均匀，通过变量替换 $u = t - q_i$，条件期望误差为：

$$\mathbb{E}[\rho_G(t) - t \mid t \in \mathcal{B}_i] = \frac{\ell_i - r_i}{2} = \frac{2q_i - q_{i-1} - q_{i+1}}{4}$$

**关键结论**：当 $r_i > \ell_i$（即右半 bin 比左半宽），期望误差必然为负。这就是 Shrinkage Bias。

### 2.2 具体例子

以 E2M1 中 $q_i = 2$ 为例：
- $q_{i-1} = 1.5$，$q_{i+1} = 3$
- $\ell_i = (2 - 1.5)/2 = 0.25$
- $r_i = (3 - 2)/2 = 0.5$
- 期望误差 = $(0.25 - 0.5)/2 = -0.125$

同样 $q_i = 4$ 也有 -0.125 的偏差（尺度翻倍）。

### 2.3 对比：均匀网格

E1M2 / INT4 的均匀网格满足 $\ell_i = r_i$ 对所有 bin 成立，因此：
$$\frac{\ell_i - r_i}{2} = 0$$

完全消除了这个几何源头的偏差。

---

## 3. 系统性影响：乘性累积与 RHT 放大

### 3.1 乘性累积（核心公式）

这是论文最重要的洞察之一。对于 GEMM $Z = AB^\top$，将量化后的 $\hat{A}$ 投影到 BF16 原值 $A$ 上：

$$\alpha_A = \frac{\langle \hat{A}, A \rangle_F}{\|A\|_F^2}$$

- $\alpha_A$: signal attenuation factor（信号衰减因子）
- $\langle \cdot, \cdot \rangle_F$: Frobenius 内积
- $\|A\|_F$: Frobenius 范数
- 当 $\alpha_A < 1$ 表示与 $A$ 对齐的信号分量被衰减

正交分解 $\hat{A} = \alpha_A A + R_A$（其中 $R_A$ 与 $A$ 正交），代入 GEMM：

$$Z_q = \hat{A}\hat{B}^\top = \underbrace{\alpha_A \alpha_B AB^\top}_{\text{scaled signal}} + \underbrace{\alpha_A A R_B^\top + \alpha_B R_A B^\top + R_A R_B^\top}_{\text{residual noise}}$$

关键参数 $\eta \approx \alpha_A \alpha_B < 1$ 是单次 GEMM 的相干衰减因子。

对于 K 层 GEMM，累积衰减为：

$$\prod_{k=1}^{K} \eta_k = \prod_{k=1}^{K}(1 - \delta_k) \approx \exp\left(-\sum_{k=1}^{K} \delta_k\right)$$

- $k$: GEMM 索引，从 1 到 $K$
- $\eta_k$: 第 $k$ 次 GEMM 的相干衰减因子
- $\delta_k = 1 - \eta_k$: 第 $k$ 次的乘性损失

**这个公式告诉你**：即使每层只有很小的 $\delta_k$，但因为是乘性累积，$K$ 层后会变成指数衰减。这与零均值噪声不同——零均值噪声会相互抵消，而 Shrinkage Bias 会指数放大。

附录 C 给出了 Taylor 展开：
$$\log(1 - \delta_k) = -\delta_k + O(\delta_k^2), \quad |\delta_k| \ll 1$$

只有当二阶项 $O(\sum_k \delta_k^2)$ 相对一阶累积项很小时，近似才成立。

### 3.2 RHT 的反效果

**这部分非常反直觉**。RHT（Random Hadamard Transform）通常用来分散 outlier，改善 codebook 利用率。但在 E2M1 上，RHT 反而恶化了情况。

定义有效 bucket ratio：
$$B_{\text{eff}}(G, T) = \frac{\exp(\mathcal{E}(G, T))}{K}$$

其中：
- $\mathcal{E}(G, T) = -\sum_{i=1}^{K} p_i \log(p_i + \epsilon)$: bucket 熵
- $p_i$: 第 $i$ 个 bucket 的经验占比
- $K$: bucket 总数
- $\epsilon$: 防止 log(0) 的小常数
- $B_{\text{eff}} \in [1/K, 1]$：从单 bucket 坍缩到均匀利用

RHT 的 SQNR 变化：
$$\Delta \text{SQNR} = 10 \log_{10} \frac{\text{NMSE}_I}{\text{NMSE}_{\mathbf{H}_{16}}}$$

- $\text{NMSE}_A(G, T) = \|Q_G(TA) - TA\|_F^2 / \|TA\|_F^2$: Normalized MSE
- $A = I$: 无旋转
- $A = \mathbf{H}_{16}$: 16 维 Hadamard 旋转
- $\Delta\text{SQNR} > 0$ 表示旋转提升保真度

**RHT 把 tensor 从 dynamic-range-limited 转变成 local-resolution-limited**。这意味着瓶颈从表示极端 outlier 变成了准确表示典型幅值密集分布。

RHT 将能量从极端 tail 推到 mid-magnitude 区域，恰好落入 E2M1 最不对称的 bin（如 $q_i = 2, 4$），导致 $\Delta\text{SQNR} < 0$。而 E1M2 的均匀网格能安全地将这种扁平化分布转化为更高保真度（$\Delta\text{SQNR} > 0$）。

---

## 4. UFP4 Recipe

基于上述分析，论文提出 UFP4（Uniform FP4）recipe：

| Configuration | E2M1-based | UFP4 (E1M2-based) |
|---|---|---|
| Format | E2M1 | E1M2/INT4-style uniform grid |
| Quant block size | 1×16 | 1×16 |
| Scale hierarchy | FP32 single-level | FP32 single-level |
| RHT scope | bwd_dw only | fwd_y, bwd_dx, bwd_dw (全部) |
| RHT block size | 16 | 16 |
| SR scope | dY | dY only |
| 2D weight scaling | ✗ | ✗ |

**设计原则**：
1. 用均匀网格消除 Shrinkage Bias 的几何源头
2. 因为网格无偏，可以安全地把 RHT 扩展到所有三个 GEMM 路径（FPROP fwd_y、DGRAD bwd_dx、WGRAD bwd_dw）
3. 只在 dY 上使用 stochastic rounding，保留梯度期望

**关键对比**：现有 NVFP4 recipe 通常只在 bwd_dw 上用 RHT，避免对 fwd_y 和 bwd_dx 旋转——因为在 E2M1 下这样做有害。UFP4 证明了"full-RHT 不是天生有害的，问题在于 E2M1 与 post-RHT 分布的失配"。

---

## 5. 实验：5 个关键问题

### Q1: RHT 是否改变首选 4-bit grid？

**单 tensor 量化**：
- 对表现良好的 linear_fc1/fwd_x：RHT 近乎中性，$\Delta$SQNR 在 E1M2 是 -0.008 dB，在 E2M1 是 +0.007 dB
- 对 outlier 重的 linear_fc2/fwd_x：**RHT 反转了格式排名**——旋转前 E2M1 领先（21.90 vs 19.94 dB），旋转后 E1M2 领先（23.19 vs 20.00 dB）

这验证了核心机制：RHT 把 outlier-heavy tensor 从动态范围限制推到局部分辨率限制，改变了首选网格。

### Q2: UFP4 是否减少 BF16 相对训练 loss gap？

在 Dense 1.5B、MoE 7.9B、MoE 124B 上的 BF16-relative LM loss error：

| Model | E2M1 reference | UFP4 (E1M2) | Improvement |
|---|---|---|---|
| Dense 1.5B | 1.2570% | 0.9673% | -23.0% |
| MoE 7.9B | 2.3596% | 1.8469% | -21.7% |
| MoE 124B | 1.7308% | 1.3863% | -19.9% |

### Q3: 优势是否跨尺度保持？

使用 Ling scaling-law protocol（Tian et al., 2026），训练 10M–324M MoE 模型。E1M2 曲线始终低于 E2M1 曲线，且 fitted FP4-to-BF16 gap 随 compute 下降。

### Q4: 哪些 recipe 组件重要？

**RHT scope ablation（Dense 1.5B，>100B tokens）**：

| Setting | Mean LM Loss | $\Delta$ loss |
|---|---|---|
| No RHT | 1.89202 | 0.00000 |
| RHT on bwd_dw | 1.88721 | -0.00481 |
| RHT on bwd_dx, bwd_dw | 1.88912 | -0.00290 |
| RHT on fwd_y, bwd_dw | 1.88558 | -0.00644 |
| Full RHT w/ SR (UFP4) | 1.88079 | -0.01123 |

**SR ablation under full RHT**：
| Setting | Mean LM Loss | $\Delta$ loss |
|---|---|---|
| Full RHT w/o SR | 1.88535 | 0.00000 |
| Full RHT w/ SR (UFP4) | 1.88079 | -0.00456 |

观察：
- Full RHT 比 no RHT 减少 0.01123 loss
- fwd_y + bwd_dw 比 bwd_dx + bwd_dw 更有用（-0.00644 vs -0.00290），说明前向路径的 RHT 收益更大
- SR 在 full RHT 基础上额外减少 0.00456

**Range-restricted E2M1 能否模拟均匀网格？**
测试 max_fpx = 2.0（只保留 {0, 0.5, 1.0, 1.5, 2.0}），但这牺牲了动态范围和 bucket 利用率，在 Dense 1.5B 和 MoE 7.9B 上都劣于 E2M1 reference。结论：**E2M1 range restriction 不能替代原生 E1M2/INT4 支持**。

### Q5: RHT 能否高效融合进 FP4 量化？

- Fused RHT + quantization：SM90 上 1.06×、SM100 上 1.07× 的 standalone quantization latency
- Unfused RHT + quantization：SM90 上 1.62×、SM100 上 1.41× 的 fused latency

当 Hadamard block size 与 quantization block 匹配时，transform 可以融合到 scale estimation 和 packing 之前，避免中间 rotated tensor。

---

## 6. 技术直觉构建

### 6.1 为什么 E2M1 被选为默认？

E2M1 的优势在于动态范围广（0.5 到 6，覆盖 12× 的幅值范围）。对于原始 outlier-heavy tensor，这看起来很合理——能容纳极端值而不饱和。

### 6.2 为什么 RHT 改变了游戏？

RHT 通过正交旋转（Sylvester Hadamard 矩阵 $\mathbf{H}_n$，加上随机符号矩阵 $\mathbf{S}_n = \text{diag}(\epsilon_1, \ldots, \epsilon_n)$，$\epsilon_i \in \{-1, +1\}$）把 outlier 能量分散到所有坐标：

$$\mathbf{Y} = \mathbf{X}\mathbf{W}^\top = (\mathbf{X}\mathbf{H}_n')(\mathbf{W}\mathbf{H}_n')^\top$$

旋转后，tensor 变得"扁平"——大多数值集中在中等幅值，极端 outlier 消失。这时候：
- E2M1 的宽动态范围成了"浪费"
- E2M1 的非均匀间距把数据推到最不对称的 bin
- E1M2 的均匀间距恰好匹配这种扁平分布

### 6.3 为什么 SR 只在 dY 上？

- bwd_dw 的量化误差是 leaf-gradient，直接被 optimizer 消费
- fwd_y 和 bwd_dx 的误差会级联到后续层，累积 Shrinkage Bias
- SR 在 dY 上保留了梯度期望，同时避免了在所有 operand 上做 SR 的开销和方差

### 6.4 为什么 forward RHT 比 backward RHT 更有用？

从 ablation 看，fwd_y + bwd_dw（$\Delta = -0.00644$）比 bwd_dx + bwd_dw（$\Delta = -0.00290$）收益更大。这可能因为：
- Forward path 在每个 token 上被执行一次，影响所有后续层
- Backward path 的 bwd_dx 误差虽然也级联，但梯度信号本身已经经过 ReLU/SiLU 等非线性，分布更"驯服"

---

## 7. 与相关工作的关系

### 7.1 格式设计轴
- **MXFP4 / NVFP4**: 基于 E2M1，细粒度 block scaling
- **HiFloat4**（Taghian et al., 2026）: 采用均匀 S1P2 data element，在 Ascend NPU 上原生支持，是 UFP4 的天然候选平台
- **MixFP4**（Zou et al., 2026）: 每 block 自适应选择 E2M1 或 E1M2
- **Four Over Six**（Cook et al., 2026）: 通过 alternative block scales 改进 NVFP4 对接近最大值的表示

### 7.2 Quantizer-side 方法
- **Microsoft FP4**（Wang et al., 2025）: 可微量化估计器 + outlier clamping
- **Quartet II**（Panferov et al., 2026）: microscaling EDEN routine 改进无偏梯度估计
- **FAAR**（Li et al., 2026）: 显式考虑非均匀 E2M1 网格的 format-aware adaptive rounding
- **TetraJet-v2**（Chen et al., 2026）: backward alignment + stochastic rounding 改进 + 振荡抑制

这些方法与 UFP4 互补——可以在均匀网格上叠加改进的 estimator、adaptive rounding 或 stabilizer。

### 7.3 Tensor-side 预处理
- **QuaRot**（Ashkboos et al., 2024）: outlier-free 4-bit inference
- **SpinQuant**（Liu et al., 2025）: learned rotations
- **FlatQuant**（Sun et al., 2025）: flatness matters
- **SmoothQuant**（Xiao et al., 2023）: outlier-channel separation
- **SVDQuant**（Li et al.）: low-rank component 吸收 outlier

论文指出：在 E2M1 下，这些 tensor-side 方法可能仍受 Shrinkage Bias 限制；换到 E1M2/INT4 后，同样预处理能更好地转化为量化质量增益。

---

## 8. 对未来硬件的启示

论文的推荐很明确也很节制：

1. **E2M1 应保留**用于 raw outlier-heavy tensor 和 inference workload
2. **未来训练加速器应原生支持 E1M2/INT4-style 均匀 4-bit 网格**作为 first-class training data element
3. 这样 UFP4 这类 recipe 能结合 post-RHT 数值稳定性与原生 4-bit matrix throughput

特别提到：
- **NVIDIA Blackwell/Rubin-class** 和 **AMD MI350-series** 目前都围绕 E2M1
- **Huawei Ascend 960**（Xu, 2025）+ HiFloat4 可能是原生实现 UFP4 的有前景平台

---

## 9. 我的思考与联想

### 9.1 与 FP8 历史的呼应
FP8 时代 E4M3 vs E5M2 的选择也是 dynamic range vs precision 的权衡。FP4 的 E2M1 vs E1M2 是类似但更极端的版本——只有 16 个值，每个 bit 的几何影响都被放大。

### 9.2 与 outlier 文献的连接
SwiGLU（linear_fc2）的 outlier-amplifying 行为（Jiang et al., 2026, POWLU）在这篇论文中是 RHT 反转格式排名的关键触发点。这暗示激活函数设计与量化格式之间存在深层耦合。

### 9.3 与信息论的连接
$B_{\text{eff}}$ 本质上是 bucket 熵的指数化。RHT 提升 bucket 熵意味着更接近 maximum entropy distribution。在均匀网格下，maximum entropy 直接转化为最低量化误差；在非均匀网格下，maximum entropy 可能落入高偏差 bin。

### 9.4 与 Random Projection 的连接
RHT 本质是 Johnson-Lindenstrauss-style 的 random projection。论文显示，random projection 的"均匀化"效果与均匀量化网格是**协同的**，与非均匀网格是**对抗的**。这是一个深层的几何-信息论洞察。

### 9.5 与 Lottery Ticket / Pruning 的联想
Shrinkage Bias 的乘性累积让我联想到 pruning 中 magnitude-based pruning 对小权重系统性偏好。两者都是"几何不对称导致的系统性偏差"。

### 9.6 与 MoE 的特殊关联
论文在 MoE 7.9B 和 MoE 124B 上验证，MoE 的 expert routing 可能加剧 tensor 分布的异质性。Shrinkage Bias 在 MoE 中可能更严重，因为不同 expert 的激活分布差异大，RHT 的"均匀化"作用更关键。

### 9.7 与 Gradient Noise 的对比
SGD 理论中，gradient noise 如果是 zero-mean，会帮助 escape saddle points；如果是 biased，会导致 systematic drift。Shrinkage Bias 是 quantization 引入的 systematic drift，类似于 biased gradient——它会推动模型朝某个"收缩"方向收敛。

---

## 10. 参考链接

- **论文本身**: arXiv 链接（论文日期 June 19, 2026）
- **NVFP4 paper**: https://arxiv.org/abs/2509.25149
- **MXFP4 (Microscaling)**: https://arxiv.org/abs/2310.10537
- **QuaRot**: NeurIPS 2024, https://arxiv.org/abs/2404.00356
- **SpinQuant**: ICLR 2025, https://openreview.net/forum?id=ogO6DGE6FZ
- **FlatQuant**: ICML 2025
- **SmoothQuant**: https://arxiv.org/abs/2211.10438, ICML 2023
- **HiFloat4**: https://arxiv.org/abs/2604.08826
- **Microsoft FP4 (Metis)**: ICLR 2026, https://openreview.net/forum?id=I2ZrCi5O84
- **Quartet II**: https://arxiv.org/abs/2601.22813
- **TetraJet-v2**: https://arxiv.org/abs/2510.27527
- **Four Over Six**: https://arxiv.org/abs/2512.02010
- **MixFP4**: https://arxiv.org/abs/2605.31035
- **FAAR**: https://arxiv.org/abs/2603.22370
- **DeepSeek-V3 Technical Report**: https://arxiv.org/abs/2412.19437
- **Ling Team scaling law**: ICLR 2026, https://openreview.net/forum?id=7r2lkhDGUj
- **Ling Team 1T model**: https://api.semanticscholar.org/CorpusID:282390137
- **Huawei Ascend SuperPoD**: https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech
- **FP8 formats (Micikevicius et al.)**: https://arxiv.org/abs/2209.05433
- **Deep Learning with Limited Numerical Precision (Gupta et al., 2015)**: https://proceedings.mlr.press/v37/gupta15.html
- **Chinchilla scaling law (Hoffmann et al., 2022)**: NeurIPS 2022
- **POWLU activation function**: https://api.semanticscholar.org/CorpusID:288669934
- **SVDQuant**: ICLR 2025
- **INT vs FP comparison (Chen et al., 2025)**: https://arxiv.org/abs/2510.25602

---

## 总结

这篇论文做了一件非常优雅的事：它把"FP4 训练不稳定"这个看似复杂的问题，归约到了一个非常简单的几何事实——**非均匀量化网格的 RTNE bin 不对称会导致系统性负偏差**。

这个偏差在单次 GEMM 中可能很小，但通过乘性累积在深度网络中指数放大。RHT 这个本来用来改善量化的工具，在 E2M1 下反而推波助澜，因为它把数据推到最不对称的 bin。

解决方案 UFP4 很简洁：换用均匀网格（E1M2/INT4），就能安全地把 RHT 扩展到所有三个 GEMM 路径，在 124B MoE 长训练上将 BF16-relative loss error 从 1.73% 降到 1.39%。

这对硬件设计有直接影响：未来的 training accelerator 应该把 E1M2/INT4-style 均匀网格作为 first-class primitive，与 E2M1 并列。Huawei Ascend 960 + HiFloat4 可能是一个有前景的原生实现路径。

整篇论文的思维方式——从数据格式的微观几何出发，推导到宏观训练稳定性——非常符合你一直强调的"build intuition from first principles"的理念。
