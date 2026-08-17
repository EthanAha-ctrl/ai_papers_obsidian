---
source_pdf: PRETRAINING LARGE LANGUAGE MODELS WITH MXFP4 ON NATIVE FP4 HARDWARE.pdf
paper_sha256: c8142d8145fd44ba7ffed57493e22b30b49ac9b539b4e1c09eb3403f77b27c48
processed_at: '2026-08-06T05:57:34-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 Paper

Andrej，好，我把前面的技术堆叠放下，用大白话把这篇 paper 讲清楚。

---

## 这群人到底在干嘛

一句话：**能不能用 4-bit 精度训练大模型，而且还能比 8-bit 更快？**

你知道现在训练大模型基本都是 FP8 或 BF16。4-bit (FP4) 如果能跑通，理论上算力翻倍、显存减半，巨大的 win。但问题是从 FP8 降到 FP4，training 基本就崩了——loss 飞走、不收敛、或者要多喂好多 token 才能训到同样水平。

这群 AMD + Penn State 的人想搞清楚一个很具体的问题：**FP4 训练到底是哪一步崩的，为什么崩，能不能修？**

---

## 他们怎么做的实验

非常 clean 的 ablation 思路。Transformer 里每次算 GEMM（矩阵乘法），其实有三条路：

- **Fprop**：前向，$Y = XW^T$，输入乘权重得到输出
- **Dgrad**：反向算对输入的梯度，$\nabla X = \nabla Y \cdot W$
- **Wgrad**：反向算对权重的梯度，$\nabla W = (\nabla Y)^T X$

他们一条一条地把这三条路从 FP8 换成 MXFP4，看每多换一条，训到 perplexity 3.3 要多花多少 token。跑的是 Llama 3.1-8B，C4 数据集，MI355X 上原生 FP4 硬件。

结果很简单很震撼：

| 开了哪些 path | Token overhead |
|--------------|---------------|
| 只 Fprop | 8-9% |
| Fprop + Dgrad | 10-11% |
| Fprop + Dgrad + Wgrad | **26-27%** |

看到没？只开前向几乎没成本。加上 Dgrad 几乎没成本。一旦把 Wgrad 也量化，直接跳到 27%。**罪魁祸首就是 Wgrad。**

---

## 为什么 Wgrad 这么难搞

让我用人话讲讲背后的直觉。

Fprop 是 $XW^T$，其中一个 $W$ 是权重，是相对稳定的，更新很慢。Dgrad 是 $\nabla Y \cdot W$，同样有个稳定的 $W$ 在里面。这两条路都有一个 "anchor"——那个缓慢变化的权重矩阵，所以 outlier 的 pattern 是稳定的，microscaling 的 block quantization 能 handle。

Wgrad 是 $(\nabla Y)^T X$。$\nabla Y$ 是从上面传回来的梯度，每步都在变，而且经过 chain rule 累积后 distribution 很 skewed。$X$ 是当前层的输入 activation，也每步都在变。**两个 operand 都是动态的，都带 outliers，相乘后 outlier 会被放大**。

更要命的是 Wgrad 的 noise 会 **沉积** 到权重里。Fprop 和 Dgrad 的 quantization error 是 transient 的——下一步输入变了，error 的影响就过去了。但 Wgrad 的 error 会通过 optimizer 更新永久写进 $W$，形成 systematic drift。就像你记账时偶尔算错一次没事，但如果每次都往同一个方向算错，年底账就全错了。

还有个 microscaling 的结构问题。MXFP4 是每 32 个 element 共享一个 exponent。如果 block 里有一个 magnitude 1000 的 outlier，整个 block 的 scale 被拉到 1000，其他 magnitude 1 的 element 量化完基本是 0。Wgrad 两个动态 operand 相乘后，这种 structured outlier 非常严重，block quantization 就崩了。

---

## 他们试了什么，什么管用什么不管用

### 试了 Stochastic Rounding，不管用

Stochastic rounding 就是量化时随机往上或往下 round，期望是 unbiased 的。在 FP16/BF16 训练里经常用，因为多次 step 后 noise 会 average out。

但在 Wgrad 上 **直接不收敛**。

直觉解释：Adam 的二阶矩 $\hat{v}_t = \beta_2 \hat{v}_{t-1} + (1-\beta_2)\nabla W_t^2$ 会把 stochastic noise 直接 inflate 进去。本来 Wgrad signal 就弱，再灌 noise 进去，Adam 的分母变大，有效学习率变小，训不动。Unbiased 不等于 harmless——在 extreme quantization 下，noise variance 进入二阶矩是致命的。

### 试了 Randomized Hadamard，也不管用

Randomized Hadamard = 固定的 Hadamard matrix 乘一个 random ±1 diagonal，每步随机。数学上还是 orthogonal，理论上还是 neutral。

但在 Wgrad 上也 **不收敛**。

这说明每步变化的 random sign 会破坏 microscaling block 的 spatial coherence。量化误差的 pattern 每 step 都不一样，optimizer 学不到一个稳定的补偿。在 4-bit 这种极端量化下，randomness 不是你的朋友。

### Deterministic Hadamard，管用！

就用一个固定的 Hadamard matrix $H_{16}$（16×16），不加任何 randomness。结果 Wgrad 的 overhead 从 26-27% 直接降到 8-9%，回到 Fprop-only 的水平。

---

## Hadamard 到底干了啥

用人话说：**它把一个集中的大数打散成很多个中等大的数。**

你有个 vector $x = [1000, 1, 1, 1, ...]$，一个 outlier 压倒一切。乘上 Hadamard matrix $H$ 之后，得到 $\tilde{x} = xH$。因为 $H$ 的每行都是 ±1，$\tilde{x}$ 的每个 element 都是 $x$ 所有 element 的 ±1 加权和。那个 1000 的 energy 被 spread 到所有 dimension，结果 $\tilde{x}$ 的 dynamic range 远远小于 $x$。

这样进 microscaling block 时，block 内的 element magnitude 接近，shared exponent 不会 outlier 拉爆，量化效率大幅提升。

关键妙处：$H$ 是 orthogonal 的，$HH^T = I$。所以你在 GEMM 前对 $X$ 和 $W$ 都乘 $H$，GEMM 时 $HH^T = I$ 自己消掉了，数学结果完全不变。**它不改变计算，只改变数据进入 quantizer 之前的分布**。这是一个完美的 probe——你 isolate 出来"是 quantization error 的问题，不是 computation 的问题"。

Forward pass 的数学：

$$Y_{out} = (XH)(WH)^T = XHH^TW^T = XW^T$$

变量说明：$X$ 是 input activation，$W$ 是 weight，$H$ 是 Hadamard matrix，$HH^T=I$ 让整个 transform 在 GEMM 中 cancel 掉。

Wgrad 的数学（最关键）：

$$\nabla W = ((\nabla Y)^TH)(X^TH)^T = (\nabla Y)^THH^TX = (\nabla Y)^TX$$

$\nabla Y$ 和 $X$ 两个 operand 都被 Hadamard spread 了 outliers，但 GEMM 时 $H$ 自己消掉。量化发生在 spread 之后，所以 error 大幅降低。这就是它能救 Wgrad 的原理。

---

## 最反直觉的发现

这篇 paper 最让我"哦"的一下是：**deterministic 比 stochastic 好**。

这违反了量化领域的传统直觉。传统认为 stochastic rounding 好，因为 unbiased。传统认为 randomized Hadamard 好，因为更 random 更 robust。但这篇 paper 证明：**在 4-bit 这种极端量化下，optimizer 需要的是 structured, predictable error，不是 unbiased random noise**。

直觉上，Adam 这种 adaptive optimizer 其实是在 implicitly 学习量化误差的 correction。如果 error pattern 每 step 都一样（deterministic），optimizer 能 learn 一个稳定的 correction。如果 error pattern 每 step 都变（stochastic），optimizer 学不到，noise 进入二阶矩反而压制 signal。

这暗示一个更深的原理：**extreme quantization 下，structured error 比 unstructured error 好**，因为 structured error 是可学习的，unstructured error 是不可学习的。

---

## 实际能省多少

Table 2 给的数字：

- Step throughput：+20%（FP4 GEMM 比 FP8 算力密度高，带宽减半）
- Token overhead：+8-9%（收敛慢一点）
- Net end-to-end speedup：+9-10%（$1.20/1.085 \approx 1.106$）

关键 insight：**FP4 的 speedup 是 stability-gated 的**。如果没 Hadamard，Wgrad 不稳定，token overhead 是 27%，net speedup 是 $1.20/1.27 \approx 0.94$，**反而更慢**。所以 naive FP4 training 看起来硬件更快，实际跑完更慢，因为多花的 token 把硬件优势吃掉了。

Hadamard 把 token overhead 压到 8-9%，net 才变正。**stabilizer 不是锦上添花，是必要条件**。

---

## 这篇 paper 的方法论价值

我觉得这篇 paper 最 elegant 的地方不是结论，是方法论：

1. **用 controlled ladder 隔离问题**：stage-wise enablement 让你能 pinpoint 是哪条 path 出问题。不做这个 ablation，你只会看到"FP4 训练崩了"，不知道是哪崩的。

2. **用 mathematically neutral probe 探索**：Hadamard 是 orthogonal 的，不改变 computation，只改变 quantization 的 input distribution。这让你能 isolate "是 quantization error 在起作用" vs "是别的东西在起作用"。比试一堆 heuristic 强太多。

3. **stochastic vs deterministic 的对照**：把 stochastic rounding、randomized Hadamard、deterministic Hadamard 放一起比，才能得出 "structured > stochastic" 这个反直觉结论。

这种 "ablation ladder + neutral probe + 机制对照" 的方法，是好的 systems research 的标志。你以前在 Tesla 做 AI 时强调过 first principles thinking，这篇 paper 就是 first principles approach to quantization debugging——不理解每条 path 的 sensitivity，你只会盲目 patch；理解了，你才能精准下刀。

---

## 我的几个直觉联想

### Hadamard 和 FFT 是亲戚

Hadamard transform 本质是 Walsh-Hadamard Transform，是 Fourier transform 在 binary group 上的 analog。Basis function 是 square wave (±1) 不是 sinusoid。计算 cost $O(n\log n)$ 类似 FFT，但常数极小（只有加法没乘法）。所以在 hardware 上 fuse 进 GEMM kernel 几乎免费。这也是为什么这个方法 practical——如果用 FFT 那种 sinusoid transform，硬件实现成本就大了。

### 和 Outlier Suppression 的关系

之前 Wei et al. 2022 的工作用 shift + scale 抑制 outlier。Hadamard 是更 principled 的版本——直接用 orthogonal transform 把 outlier energy spread 出去，不引入 learnable parameter，不改变 computation。这种 "用线性代数结构解决问题" 比 "用 learnable parameter 补偿" 更 robust，泛化更好。

### 和 QK-Norm 的哲学

QK-norm 在 attention 里加 LayerNorm 抑制 outlier。Hadamard 是一个 linear alternative，不引入 nonlinearity。两者哲学类似——都是 normalize 掉 outlier 让下游计算更稳定。但 Hadamard 更 transparent，数学上完全 neutral。

### 能不能 push 到 FP3

Hadamard 能 spread outliers 但不能 eliminate。FP4 有 16 个 representable level，FP3 只有 8 个。直觉上 Hadamard 的 benefit 会 marginal 递减——spread 完的 distribution 仍然可能超出 FP3 的 representable range。这是个 open question，可能需要结合 learnable scale factor 或 non-uniform quantization。

### Finetuning 时的行为

Pretraining 时 Wgrad 是 dominant cost，Hadamard 很关键。Finetuning 时 gradient 更 sparse 更小，Wgrad 的 outlier pattern 可能不同，Hadamard 的 benefit 可能 marginal。Paper 老实承认了这一点，recipe 不是 universal。

### 和 SAM 的潜在联系

SAM (Sharpness-Aware Minimization) 通过 perturbation 找 flat minima。Hadamard 让 gradient 更 isotropic，可能间接鼓励 flat minima。有个更深的猜想：**structured orthogonal transform 让 optimization landscape 更各向同性**，这可能是 stability 提升的深层原因。需要理论分析。

---

## 一句话总结

**FP4 训练崩在 Wgrad，因为两个动态 operand 相乘放大 outlier，量化误差沉积进权重形成 systematic drift；用一个 deterministic Hadamard transform 把 outlier 在 GEMM 前 spread 掉，数学上完全 neutral 但让 microscaling block 量化效率大幅提升，从而把 token overhead 从 27% 压到 9%，让 FP4 的硬件优势真正兑现成 end-to-end speedup；stochastic 方法反而有害，因为在 extreme quantization 下 Adam 的二阶矩会把 noise inflate 掉 signal。**

核心 insight：在 extreme quantization 下，structured error 比 unbiased noise 好，因为 optimizer 能 learn 到一个稳定的 correction。

---

## 参考

- AMD MI355X: https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
- MX format 原始 paper (Rouhani et al. 2023): https://arxiv.org/abs/2310.10537
- Llama 3: https://arxiv.org/abs/2407.21783
- Outlier Suppression (Wei et al. 2022): https://proceedings.neurips.cc/paper_files/paper/2022/file/0727a803fa6c556c70a6701c9b9105cd-Paper-Conference.pdf
- NVFP4 Pretraining (Abecassis et al. 2025): https://arxiv.org/abs/2509.25149
- FP4 All the Way (Chmiel et al. 2025): https://arxiv.org/abs/2505.19115
- Towards Efficient Pre-training (Zhou et al. 2025): https://arxiv.org/abs/2502.11458
- LLM-FP4 (Liu et al. 2023): https://aclanthology.org/2023.emnlp-main.40/
- NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- MLPerf benchmark: https://mlcommons.org/benchmarks/training/

Andrej，人话版就到这里。核心就三件事：Wgrad 是 bottleneck，Hadamard 救它，deterministic 比 stochastic 好。剩下的都是 build intuition 的细节。这篇 paper 最值得学的其实是方法论——用 ablation ladder + neutral probe 把一个复杂问题拆清楚，而不是盲目堆 heuristic。这种思维方式你在 micrograd 里教过，就是"理解每一步在干嘛"，只是 scale 到了 quantization 层面。

---

# MXFP4 Native FP4 Pretraining 深度解读

Andrej, 这篇 paper 非常对你的胃口——它用一个 controlled experiment 去隔离 FP4 training 失效的 root cause, 而且是在真正的 native FP4 hardware (AMD MI355X) 上跑的, 没有 software emulation 的污染。让我一层层 build intuition。

---

## 1. 核心问题的 framing

这篇 paper 的精髓在于一个**渐进式 ablation ladder**: 把 transformer 里的 GEMM 操作分成三条 path, 逐步开启 MXFP4:

| Path | 数学表达 | 含义 |
|------|---------|------|
| **Fprop** | $Y = XW^T$ | 前向计算, X 是 activation, W 是 weight |
| **Dgrad** | $\nabla X = \nabla Y \cdot W$ | 对 input 的 gradient, 用于传给上一层 |
| **Wgrad** | $\nabla W = (\nabla Y)^T X$ | 对 weight 的 gradient, 用于 optimizer 更新 |

然后测量每种组合到达 perplexity 3.3 所需的 token 数。这种 stage-wise 启用方式让你能精确定位 quantization noise 在哪个 path 上最致命。

**关键发现**: Wgrad 量化单独就贡献了从 10-11% 跳到 26-27% 的 overhead, Fprop 和 Dgrad 加起来只增加 ~2%。

---

## 2. 为什么 Wgrad 是 bottleneck? Intuition building

让我深挖一下为什么 Wgrad 特别敏感。这涉及到几个数学和结构上的原因:

### 2.1 Wgrad 的数值结构

Wgrad 的计算是:
$$\nabla W = (\nabla Y)^T X$$

这里 $\nabla Y$ 是上游传回来的 activation gradient, $X$ 是该层的 input activation。两个 operands 都是 **dynamic 的**, 都包含 outliers, 而且在 backprop 时 $\nabla Y$ 的 magnitude distribution 经过多层 chain rule 累积后变得非常 skewed。

相比之下:
- **Fprop**: $Y = XW^T$, $W$ 是相对 static 的 (缓慢更新), 可以用 EMA 之类的方法平滑
- **Dgrad**: $\nabla X = \nabla Y \cdot W$, 同样有一个相对 static 的 $W$
- **Wgrad**: 两个都是 dynamic, outlier 都很严重

### 2.2 累积效应

Wgrad 直接进入 optimizer, 每个 step 都累加到 weight 上:
$$W_{t+1} = W_t - \eta \cdot \text{Adam}(\nabla W_t)$$

Fprop 和 Dgrad 的 quantization noise 是 "transient" 的——它们影响当前 step 的 output, 但下一步 X 会变。而 Wgrad 的 noise 会 **永久性地沉积** 到 weight 中, 形成系统性 bias。这就像 random walk 里的 systematic drift, 比 transient noise 危险得多。

### 2.3 Microscaling 的 block 结构与 Wgrad 的耦合

MXFP4 的核心是 microscaling: 每 32 个 element 共享一个 E8M0 的 shared exponent。公式:

$$x_{MX} = Q_{FP4}\left(\frac{x_B}{2^{E_{\text{shared}}}}\right) \times 2^{E_{\text{shared}}}$$

其中:
- $x_B$ 是一个 32-element block
- $E_{\text{shared}} = \max_{i \in B} \text{exponent}(x_i)$ 是该 block 内的最大 exponent
- $Q_{FP4}$ 是 round-to-nearest 到 4-bit float (E2M1 格式, 只有 ±0.5, ±1, ±2, ±3, ±4, ±6 等 16 个值)

**问题**: 如果 block 内有一个 outlier (比如一个 magnitude 1000 的 element), 整个 block 的 scale 就被这个 outlier 拉到 1000, 其他 magnitude ~1 的 element 量化后基本是 0。这就是 "outlier dominates quantization range" 的经典问题。

Wgrad 中, $\nabla Y$ 和 $X$ 都有这种 structured outliers (尤其是 attention output 的某些 channel), 它们相乘后的 product $\nabla W$ 在某些 row/column 上有极端大的值, 破坏 block quantization 的 efficiency。

---

## 3. Hadamard Transform: 数学原理与直觉

这是 paper 最 elegant 的部分。Hadamard matrix $H_n$ 是 $n \times n$ 的 orthogonal matrix, 元素都是 $\pm 1/\sqrt{n}$ (paper 里省略了 normalization, 但不影响 orthogonality)。$H_{16}$ 和 $H_{32}$ 是常用的尺寸。

### 3.1 关键性质: $HH^T = H^TH = I$

这意味着如果你对 input 做 $H$ rotation, 对 weight 也做 $H$ rotation, 那么 GEMM 结果不变:

$$Y_{out} = (XH)(WH)^T = XHH^TW^T = XW^T$$

所以 Hadamard transform 是一个 **数学上完全 neutral 的操作**——它不改变 computation, 只改变 data 进入 quantizer 之前的 distribution。这是一个绝妙的 "probe": 你能 isolate 出 "是 quantization error 在起作用" vs "是 computation 在起作用"。

### 3.2 为什么 Hadamard 能缓解 outliers?

Hadamard transform 本质是一个 **energy compaction / spreading** 操作。考虑一个 vector $x = [1000, 1, 1, 1, ...]$ (有一个 outlier)。乘以 $H$ 后, energy 被 spread 到所有 dimension:

$$\tilde{x} = xH$$

每个 $\tilde{x}_i$ 都是 $x$ 的所有 element 的 ±1 加权和 (除以 $\sqrt{n}$), 所以 outlier 的 energy 被 "dilute" 到所有 output dimension。结果是 $\tilde{x}$ 的 dynamic range 远小于 $x$, block quantization 的 efficiency 大幅提升。

数学上, Hadamard 是一个 **normalized tight frame**, 它的 condition number 是 1, 是 isometry, 保持 L2 norm。但它改变了 L∞ norm (max element), 这正是 quantization 关心的。

### 3.3 Forward pass 的完整推导 (Appendix C.1)

标准: $Y = XW^T$
变换后:
$$Y_{out} = \tilde{X}\tilde{W}^T = (XH)(WH)^T = XH \cdot H^TW^T = X(HH^T)W^T = XW^T$$

变量:
- $X \in \mathbb{R}^{B \times d}$: input activation (B=batch×seq, d=hidden dim)
- $W \in \mathbb{R}^{d_{out} \times d}$: weight matrix
- $H \in \mathbb{R}^{d \times d}$: Hadamard matrix, $d$ 必须是 16 或 32 的倍数 (tiled)
- $\tilde{X} = XH$: rotated activation
- $\tilde{W} = WH$: rotated weight

### 3.4 Dgrad 推导 (Appendix C.2)

标准: $\nabla X = \nabla Y \cdot W$
设 $\tilde{G} = \nabla Y \cdot H$, $\tilde{W}_{rot} = W^TH$:
$$\nabla X = \tilde{G} \cdot \tilde{W}_{rot}^T = (\nabla Y \cdot H)(W^TH)^T = \nabla Y \cdot H \cdot H^T \cdot (W^T)^T = \nabla Y \cdot W$$

### 3.5 Wgrad 推导 (Appendix C.3) — 最关键的部分

标准: $\nabla W = (\nabla Y)^T X$
设 $\tilde{G}_T = (\nabla Y)^TH$, $\tilde{X}_T = X^TH$:
$$\nabla W = \tilde{G}_T \tilde{X}_T^T = ((\nabla Y)^TH)(X^TH)^T = (\nabla Y)^T H H^T X = (\nabla Y)^T X$$

这里 Hadamard 同时作用在 $\nabla Y$ 和 $X$ 上, 两个 operands 的 outliers 都被 spread 开, 然后 GEMM 时 $HH^T = I$ 让数学结果不变, 但 quantization 在 spread 之后进行, 误差大幅降低。**这正是 paper 要解决 Wgrad 问题的核心机制。**

---

## 4. 为什么 Stochastic 方法失败? Intuition

这是 paper 里最反直觉的发现, 也是最有启发性的。

### 4.1 Stochastic Rounding 的机制

Stochastic rounding 是 quantization 时以概率保留 round up 或 down:
$$Q_{stoch}(x) = \begin{cases} \lfloor x \rfloor & \text{with prob } 1-(x-\lfloor x \rfloor) \\ \lceil x \rceil & \text{with prob } x-\lfloor x \rfloor \end{cases}$$

它的好处是 **unbiased estimator**: $E[Q_{stoch}(x)] = x$。在 FP16/FBF16 训练中, stochastic rounding 经典地用于补偿 precision loss, 因为 noise 在多次 step 后 average out。

### 4.2 为什么在 Wgrad 上失效?

关键 insight: **unbiased noise 对 Adam 这类 adaptive optimizer 是有害的**。

Adam 的 update: $W_{t+1} = W_t - \eta \cdot \hat{m}_t / \sqrt{\hat{v}_t}$

其中 $\hat{v}_t$ 是 gradient 的二阶矩 running average:
$$\hat{v}_t = \beta_2 \hat{v}_{t-1} + (1-\beta_2) \nabla W_t^2$$

如果 $\nabla W_t = \nabla W_{true} + \epsilon_t$, 其中 $\epsilon_t$ 是 stochastic rounding 的 zero-mean noise, 那么:
$$E[\hat{v}_t] = \beta_2 \hat{v}_{t-1} + (1-\beta_2)(\nabla W_{true}^2 + \text{Var}(\epsilon_t))$$

**Stochastic noise 直接 inflate 了 $\hat{v}_t$**, 压制了真实 gradient 的 signal。在 Wgrad 这种 sensitive path 上, signal-to-noise ratio 已经很低, 再加 noise 直接 break optimization。

### 4.3 Randomized Hadamard 失败的原因

Randomized Hadamard = $H \cdot D$, 其中 $D$ 是 random ±1 diagonal matrix (random sign flips)。数学上仍然 orthogonal, 仍然 $HD(HD)^T = I$, 所以理论上应该和 deterministic Hadamard 一样 neutral。

但 paper 的实验显示 randomized 版本在 Wgrad 上 **does not converge**。可能的解释:

1. **Random sign flips 引入 per-step variation**: 每个 step 的 D 不同, 虽然 $E[\cdot]$ 是 isometry, 但单步来看 quantization 的 block 分配会变化, 引入额外的 noise
2. **破坏了 microscaling 的 spatial coherence**: deterministic H 让 outlier 始终 spread 到相同的 dimension, quantization error 是 structured 的, 可以被 optimizer 学习补偿。Randomized 让 error pattern 每 step 都变, optimizer 无法 learn 到一个稳定的 correction

这暗示一个深刻原理: **在 extreme quantization 下, structured error > unstructured error**, 因为 optimizer 可以适应 structured error, 但 random error 进入二阶矩后是 "不可补偿" 的。

---

## 5. 架构图解析 (Figure 3)

Paper 的 Figure 3 展示了 Hadamard-transformed MXFP4 architecture。让我解读:

```
Forward Pass:
X ──[H]──> X̃ ──────────────────┐
                                 ├──> MXFP4 GEMM ──> Y_out
W ──[H]──> W̃ ──────────────────┘
                  (H H^T = I cancels)

Backward Pass (Dgrad):
∇Y ─[H]──> G̃ ───────────────────┐
                                 ├──> MXFP4 GEMM ──> ∇X
W ──[H]──> W̃_rot ───────────────┘

Backward Pass (Wgrad):
∇Y ─[H]^T──> G̃_T ──────────────┐
                                 ├──> MXFP4 GEMM ──> ∇W
X ──[H]^T──> X̃_T ───────────────┘
```

关键点:
1. Hadamard transform 在 **GEMM 之前** 应用, 所以 quantization 看到的是 rotated input
2. 由于 orthogonality, 数学结果完全等价于不 rotate
3. 实现上, Hadamard 可以 **fuse 到 GEMM kernel** 里 (AMD ROCm Transformer Engine), 所以 overhead 很小
4. $H_{16}$ 比 $H_{32}$ 快 8% (1.08x vs 1.00x), 因为更小的 tile 更适合 MI355X 的 vector unit

---

## 6. 实验数据表深度解析

### 6.1 Table 1: Stage-wise Enablement

让我重新整理这个表, 突出 pattern:

| Stabilizer | Fprop | +Dgrad | +Wgrad | 观察 |
|-----------|-------|--------|--------|------|
| None | 8-9% | 10-11% | 26-27% | Wgrad 是 jump point |
| Stochastic Rounding | 8-9% | 10-11% | **DNC** | Stochastic 在 Wgrad 上有害 |
| Randomized Hadamard (H16) | 8-9% | 10-11% | **DNC** | Randomness 在 Wgrad 上有害 |
| Deterministic Hadamard (H32) | 8-9% | 10-11% | **8-9%** | 恢复到 Fprop-only 水平 |
| Deterministic Hadamard (H16) | 8-9% | 10-11% | **8-9%** | 同 H32 但更快 |

几个值得注意的 pattern:

1. **Fprop 量化本身 overhead 很小 (8-9%)**: 这与我之前认为 "FP4 量化 activation 很危险" 的 intuition 相反。原因是 microscaling 的 per-32-element shared exponent 已经足够 fine-grained 来 handle activation outliers, 只要 weight 不被量化。

2. **Dgrad 几乎免费 (10-11% vs 8-9%)**: 只增加 2%。因为 Dgrad $\nabla X = \nabla Y W$ 中, $W$ 是相对 static 的, outlier pattern 稳定, microscaling 能 well handle。

3. **Wgrad 是 phase transition (26-27%)**: 这就是 paper 的核心发现。两个 dynamic operands 相乘, outlier amplification 严重。

4. **Hadamard 让 Wgrad 的 overhead 回到 Fprop 水平 (8-9%)**: 这意味着 Hadamard 完全消除了 Wgrad 量化的额外 cost。Wgrad 量化的"困难"几乎 100% 来自 outlier structure, 被 Hadamard spread 掉后就和 Fprop 一样容易了。

### 6.2 Table 2: End-to-end Efficiency

| Metric | MXFP4 + H16 vs FP8 |
|--------|-------------------|
| Train step throughput | +20% |
| Token overhead | +8-9% |
| End-to-end speedup | +9-10% |

直觉检查:
- Step throughput 提升 20%: 因为 FP4 GEMM 比 FP8 GEMM 算力密度高 ~2x, memory bandwidth 减半。但实际只提升 20%, 说明 **GEMM 不是唯一 bottleneck**, 还有 communication, optimizer step 等
- Token overhead 8-9%: 这是 convergence 速度的 cost
- Net speedup 9-10%: $1.20 / 1.085 \approx 1.106$, 大致 10.6%, 与 reported 9-10% 吻合

**重要 insight**: FP4 的 speedup 是 **stability-gated**。如果 Wgrad 不稳定, token overhead 是 26-27%, 那么 end-to-end 是 $1.20 / 1.27 \approx 0.94$, 反而 **变慢**。这就是为什么 naive FP4 training 在实际中经常 "看起来有理但实际更慢"——不稳定吃掉了硬件效率。

---

## 7. 1D vs 2D Quantization (Appendix D)

这部分的 intuition 很重要, 涉及 microscaling 的 block shape 选择:

### 7.1 2D Block (32×32)
- 一个 $32 \times 32$ 的 region 共享一个 E8M0 scale
- **Transpose-invariant**: $W$ 和 $W^T$ 用同一个 scale, 这对 Fprop (用 $W^T$) 和 Wgrad (用 $W$) 都需要的情况非常有利
- 适合 weight, 因为 weight 在多个 path 中被使用

### 7.2 1D Row-wise (1×32)
- 每 32 个连续 element (沿 row) 共享 scale
- **Per-token granularity**: 对 activation 自然, 因为 transformer 的 token 是基本单位
- 适合 activation, 因为不同 token 的 magnitude 差异大, per-token scale 能 capture

### 7.3 1D Column-wise (32×1)
- 每 32 个 element (沿 column) 共享 scale
- **Per-channel granularity**: 对 weight 自然, 因为不同 output channel 的 magnitude 可能差异大
- 但 **memory access pattern 不友好** (transpose 需要先做), 且 GEMM 时需要 gather, 慢

Paper 暗示实际使用的是 hybrid: activation 用 1D row-wise, weight 用 2D (因为 transpose-friendly)。这是一个 hardware-software co-design 的好例子。

---

## 8. 与相关工作的联系

### 8.1 Outlier Suppression (Wei et al., 2022)
https://proceedings.neurips.cc/paper_files/paper/2022/file/0727a803fa6c556c70a6701c9b9105cd-Paper-Conference.pdf

这篇是 outlier 处理的经典工作, 思路是 **shift** activation 让 outlier 不那么突出。Hadamard 是更 principled 的方法——直接用 orthogonal transform 把 outlier energy spread 出去, 不需要 learnable shift。

### 8.2 FP4 All the Way (Chmiel et al., 2025)
https://arxiv.org/abs/2505.19115

这个工作用 learnable scaling factor 和 GPTQ-like post-processing 来 enable FP4。与本文的 contrast: 本文 **不 learn 任何东西**, 只用一个 fixed deterministic Hadamard。说明 outlier 问题是 **structural** 的, 不需要 learned correction, geometric transform 就够了。

### 8.3 NVFP4 Pretraining (Abecassis et al., 2025)
https://arxiv.org/abs/2509.25149

NVIDIA 的对应工作, 用 NVFP4 格式 (类似 MXFP4 但 block size 不同, 32-element block vs NVIDIA 的 16-element)。两者结论一致: Wgrad 是最难的, 但 NVIDIA 用的是 module-wise mixed precision + staged schedule, 而本文用 Hadamard。AMD 的方案更 elegant——一个 transform 解决所有问题。

### 8.4 Microscaling Formats (Rouhani et al., 2023)
https://arxiv.org/abs/2310.10537

这是 MX format 的原始 paper, 定义了 OCP standard。MXFP4 = E2M1 mantissa + E8M0 shared exponent per 32 elements。这个标准是 AMD/NVIDIA/Intel 共同推的, 所以 hardware 是互通的。

### 8.5 Llama 3 (Grattafiori et al., 2024)
https://arxiv.org/abs/2407.21783

Baseline model。Llama 3.1-8B 是个 good testbed 因为它足够大能 reflect real training dynamics, 又足够小能在合理时间跑 ablation。

### 8.6 Towards Efficient Pre-training (Zhou et al., 2025)
https://arxiv.org/abs/2502.11458

这个工作也探索 FP4 pretraining, 用 module-wise mixed precision。与本文 contrast: 本文发现 **不需要 mixed precision**, 全 pipeline FP4 + Hadamard 就行, 更简单。

### 8.7 LLM-FP4 (Liu et al., 2023)
https://aclanthology.org/2023.emnlp-main.40/

早期的 FP4 inference 工作, 识别了 outlier 问题。本文的 Hadamard 是对这个问题更 elegant 的 solution。

---

## 9. 联想与延伸

### 9.1 Hadamard 与 Fourier 的关系

Hadamard transform 本质是 **Walsh-Hadamard Transform (WHT)**, 是 Fourier transform 在 binary group 上的 analog。它的 basis function 是 square wave (±1), 不是 sinusoid。这意味着:
- 计算 cost 是 $O(n \log n)$ (类似 FFT), 但常数极小 (只有 ±1 加法)
- 适合 hardware (无乘法)
- 在 spread outliers 上和 Fourier 类似, 但更 cheap

### 9.2 与 QK-Normalization 的联系

近期一些工作 (e.g., QK-norm in ViT) 用 LayerNorm 来抑制 attention 中的 outliers。Hadamard 是一个 **linear** 的 alternative——不引入 nonlinearity, 数学上 neutral, 更适合作为 "transparent" stabilizer。

### 9.3 与 Mixup / CutMix 的哲学对比

Mixup 在 data augmentation 里用 linear interpolation 来 regularize。Hadamard 在 feature space 里做 linear combination 来 spread outliers。两者都是 **linear transform 改变 distribution 但保留信息** 的哲学。

### 9.4 是否可以 extend 到 FP3 或更低?

Hadamard 能 spread outliers, 但不能消除。FP4 有 16 个 level, FP3 只有 8 个 level, dynamic range 更窄。直觉上, Hadamard 的 benefit 会 marginal 递减——当 quantization 太粗时, 即使 spread 后的 distribution 也超出 representable range。这可能是一个 future work 方向。

### 9.5 与 Sharpness-Aware Minimization (SAM) 的联系

SAM 通过 perturbation find flat minima。Hadamard 让 weight update 更 stable, 间接鼓励 flat minima (因为 noise 被 spread, gradient direction 更 stable)。可能存在 deeper connection: **structured transforms 使 optimization landscape 更 isotropic**。

### 9.6 MI355X Hardware 的意义

https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html

MI355X 是 AMD 的最新 Instinct GPU, 支持 native FP4 tensor core。这意味着:
- FP4 GEMM 是 **硬件加速的**, 不是 emulation
- Throughput 数据是真实的, 不是 estimate
- 这让 paper 的 efficiency claim 可信

NVIDIA Blackwell 也有 FP4 support (NVFP4), 所以这是一个 industry-wide trend。

### 9.7 Transformer Engine 的实现

Paper 提到用 AMD ROCm Transformer Engine。NVIDIA 有对应的 Transformer Engine (https://github.com/NVIDIA/TransformerEngine)。这些 library 的作用是 fuse quantization + GEMM + dequantization 到一个 kernel 里, 避免 memory bandwidth overhead。Hadamard 也可以 fuse 进去, 所以 overhead 极小。

### 9.8 为什么 H16 比 H32 快?

直觉上, 更小的 tile 意味着:
- 更少的 register pressure
- 更好的 vectorization (MI355X 的 SIMD width 可能 favor 16)
- 更适合 L1 cache 的 tile size

8% 的 speedup 虽然不大, 但在 scale 上有意义。

### 9.9 Generalization 的担忧

Paper 在 Discussion 里诚实地说: "a stabilization that works for full pretraining (MLPerf C4 dataset, Llama 3.1–8B) may not generalize to other models or finetuning methods"。

这个 caveat 很重要。Finetuning 时 gradient distribution 与 pretraining 不同 (更小, 更 sparse), 可能 Hadamard 的 benefit 变小甚至消失。需要更多实验确认。

### 9.10 与 Curriculum Learning 的关系

Wgrad quantization 的 26-27% overhead 可能可以通过 curriculum 来 mitigate——早期用 FP8, 后期切换到 FP4。Paper 没探索这个, 但 Zhou et al. (2025) 的 staged schedule 是这个思路。Hadamard + staged schedule 可能是 combinable 的。

---

## 10. 可能的 Future Direction

基于这篇 paper 的 framework, 我能想到几个有价值的 extension:

1. **Hadamard 的 learnable variant**: 让 network learn 一个 orthogonal transform, 可能比 fixed Hadamard 更好 (但开销可能抵消 benefit)
2. **Per-layer Hadamard size**: 不同层的 outlier pattern 不同, 用不同大小的 H
3. **Hadamard + LoRA**: finetuning 时 Wgrad 只对 adapter 计算, Hadamard 的作用可能不同
4. **Hadamard 在 attention 里的 special handling**: attention 的 softmax 已经是 normalized, outlier pattern 与 FFN 不同
5. **与 8-bit optimizer 结合**: Adam 的 state 用 8-bit, Wgrad 用 FP4, 可能 compound benefit
6. **Theory of structured quantization error**: 为什么 deterministic > stochastic? 需要一个理论 framework 来解释 optimizer 对 structured vs unstructured error 的 response

---

## 11. 总结: 三个 Key Understanding

Paper 自己总结了三个, 让我用更技术性的语言 reframe:

### KU1: Wgrad 是 FP4 training 的 critical bottleneck
不是因为 Wgrad 的 computation 更难, 而是因为 Wgrad 的 error 会 **persistent** (沉积到 weight 里), 而 Fprop/Dgrad 的 error 是 transient。两个 dynamic operands (X 和 ∇Y) 相乘时 outlier amplification 最严重。

### KU2: FP4 speedup 是 stability-gated
Hardware 给你 20% step speedup, 但如果 convergence 需要 27% 更多 token, net 是负的。Stabilizer (Hadamard) 把 token overhead 压到 8-9%, 让 net speedup 变正 (9-10%)。这是 FP4 实用化的 **必要条件**。

### KU3: Structured > Stochastic 在 extreme quantization
Stochastic rounding 和 randomized Hadamard 都失败, deterministic Hadamard 成功。这挑战了 "unbiased noise is good" 的经典直觉。在 4-bit 这种 extreme quantization 下, optimizer (尤其 Adam 的二阶矩) 对 noise 的 sensitive 超过了 unbiased 的 benefit。Structured, predictable error 让 optimizer 能 learn 一个稳定的 correction。

---

## 参考链接汇总

- AMD MI355X: https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
- Microscaling Formats (MXFP4 原始 paper): https://arxiv.org/abs/2310.10537
- Llama 3: https://arxiv.org/abs/2407.21783
- Outlier Suppression: https://proceedings.neurips.cc/paper_files/paper/2022/file/0727a803fa6c556c70a6701c9b9105cd-Paper-Conference.pdf
- NVFP4 Pretraining: https://arxiv.org/abs/2509.25149
- FP4 All the Way: https://arxiv.org/abs/2505.19115
- Towards Efficient Pre-training (FP4): https://arxiv.org/abs/2502.11458
- LLM-FP4: https://aclanthology.org/2023.emnlp-main.40/
- NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- MLPerf: https://mlcommons.org/benchmarks/training/

Andrej, 这篇 paper 的 elegance 在于它用一个 **mathematically neutral probe (Hadamard)** 来 isolate 问题, 而不是用一堆 heuristic 去 patch。这种 "controlled experiment + principled intervention" 的方法论正是好的 systems research 的标志。Wgrad 是 bottleneck 这个发现, 加上 deterministic > stochastic 这个反直觉结论, 让这篇 paper 既有工程价值也有科学价值。你之前在 micrograd/llm.c 里强调过理解 backprop 的每一步, 这篇 paper 正是把这个原则 push 到了 quantization level——理解每个 GEMM path 的 quantization sensitivity, 才能 diagnose 问题在哪。
