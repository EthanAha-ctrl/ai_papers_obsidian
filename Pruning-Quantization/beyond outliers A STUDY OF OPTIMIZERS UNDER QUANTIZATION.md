---
source_pdf: beyond outliers A STUDY OF OPTIMIZERS UNDER QUANTIZATION.pdf
paper_sha256: 59681c529cf7e4bf6d82574f3bfddd53adb76ff08d34b8d1a50ddb03857db3e2
processed_at: '2026-07-18T17:03:58-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Beyond Outliers: Optimizer × Quantization 系统性研究详解

## 1. 核心动机与问题定位

这篇paper来自 ETH Zurich (Saleh Ashkboos, Torsten Hoefler, Georgios Vlassis) 和 ISTA (Dan Alistarh, Alexandra Volkova)，研究的核心问题是：

**给定相同的validation loss，用不同optimizer训练出来的模型，在quantization（PTQ/QAT）下的表现是否相同？如果不同，为什么？**

这个问题在2025年这个时间点特别timely，因为：
- 新一代optimizer（Muon, SOAP, Scion, PSGD等）在2024-2025年大量涌现
- LLM quantization（W4A4）已经成为deployment标准
- 但是optimizer和quantization两个community几乎没有交集

Paper最大的insight在于：**传统用来预测quantization难度的metric（MMR、Kurtosis）其实是错的**，因为它们只看isolated layer的outlier，完全忽略了error propagation。Paper提出新的ABC decomposition framework，证明quantization error主要由"accumulated error propagation"决定，而不是local outlier。

参考链接：
- arXiv: https://arxiv.org/abs/2502.07178 (类似工作)
- Muon: https://kellerjordan.github.io/posts/muon/
- QuEST: https://openreview.net/forum?id=I0Ux2nAN6u
- OLMo2: https://arxiv.org/abs/2501.00656

---

## 2. 实验设置细节

### 2.1 Model Architecture
基于 **OLMo2**，加上几个修改：
- No biases
- Rotary positional embeddings (RoPE)
- RMSNorm (instead of LayerNorm)
- Reordered pre-normalization
- **QKNorm** (Henry et al., 2020) — 对query/key做normalization
- Weight tying (input embedding和output embedding共享)
- **ReLU² activation** (So et al., 2022) — ReLU后平方， Primer论文里发现的好东西

Model sizes: 50M / 125M / 350M / 500M / 760M / 1.5B

### 2.2 六个Optimizer
- **AdamW**：baseline，1st/2nd moment
- **Muon** (Keller Jordan 2024)：对hidden layer用Newton-Schulz迭代近似orthogonalize gradient，本质上是把update约束到modular norm ball里
- **PSGD** (Li 2015)：Preconditioned SGD，维护左右preconditioner matrices P_L, P_R
- **Shampoo** (Gupta 2018)：full-matrix preconditioner，用gradient的左右Gram矩阵的逆矩阵幂来precondition
- **SOAP** (Vyas 2025)：Shampoo + Adam，在eigenspace里做Adam，定期recompute eigenbasis
- **Scion** (Pethick 2025)：Scion用spectral LMO（类似Muon但更principled），把update约束在norm ball里

### 2.3 数据和训练
- **ClimbMix**：400B tokens高质量mix
- **Chinchilla-optimal**：20× tokens/parameter ratio
- **Common Loss (CL)**：所有optimizer能达到的最低validation loss，用于PTQ时确保所有model起点相同
- 评测：PIQA, HellaSwag, ARC-Easy（zero-shot）

### 2.4 Quantization设置
- **PTQ**：row-wise symmetric AbsMax quantization，W4A4，所有linear layer都量化
- **QAT**：用 **QuEST** (Panferov 2025)，forward pass用Hadamard transform + optimal clipping + STE backward

---

## 3. Full-Precision结果：Muon的统治

Table 2的核心结果：

| Model | AdamW | Muon | Shampoo |
|-------|-------|------|---------|
| 50M | 43.75 | 45.03 | 44.81 |
| 125M | 48.64 | 49.62 | 49.53 |
| 350M | 56.58 | **58.08** | 56.51 |
| 760M | 63.90 | **64.63** | 63.05 |
| 1.5B | 67.93 | **69.19** | 68.16 |

**Key observation**：Muon在大模型上consistently最好，gap随model size增大（350M只赢0.01%，1.5B赢1.03%）。这与Wen et al. 2025 ("Fantastic pretraining optimizers") 的发现一致。

### Learning Rate vs MMR的诡异现象
Figure 2展示了一个非常反直觉的现象：**所有optimizer都满足 learning rate ↑ → MMR ↑**。

这里 MMR (Max-to-Median Ratio) 定义为：
$$\text{MMR} = \frac{\max_i |h_i|}{\text{median}_i |h_i|}$$

其中 $h_i$ 是row-wise的activation值。

**Intuition**：更大的learning rate让weight走得更远，导致activation distribution更"尖"（peaky），outlier更明显。这说明outlier feature某种程度上是**优化轨迹的副产物**，而不是模型本身inherent的属性。

Muon在所有optimizer中MMR最低——这暗示Muon的update方向（orthogonalized）让activation distribution更平滑。当时大家的直觉是：Muon低MMR → Muon应该PTQ友好。

**但下面PTQ结果完全颠覆了这个intuition。**

---

## 4. PTQ结果：传统wisdom的崩塌

### 4.1 反直觉的核心实验

Table 3：所有model训练到相同Common Loss，再做PTQ。结果：

| Model | AdamW | Muon | PSGD | Scion | **Shampoo** | SOAP |
|-------|-------|------|------|-------|---------|------|
| 350M | 49.23 | 47.42 | 50.09 | 49.80 | **53.93** | 49.08 |
| 760M | 59.22 | 50.00 | 52.11 | 53.74 | **59.26** | 46.22 |
| 1.5B | 62.51 | 47.75 | N/A | N/A | **63.88** | N/A |

**Shampoo是PTQ最robust的optimizer**，但它有**最高的MMR**！

而**Muon的MMR最低**，但**PTQ degradation最严重**（760M上从64.63掉到50.00）。

这就是paper标题"Beyond Outliers"的含义：**outlier metric不仅不predictive，甚至是反predictive的**。

Figure 1里的scatter plot显示：
- MMR vs PTQ accuracy的Spearman correlation ρ ≈ 0（不相关）
- Kurtosis vs PTQ accuracy的ρ ≈ 0（不相关）
- 新提出的 $R_L$ vs PTQ accuracy的ρ很高（强相关）

---

### 4.2 ABC Decomposition的数学推导

这是paper的核心理论贡献。让我详细推一遍。

#### Setup
考虑一个L层网络，第ℓ层的activation：
$$h_\ell = f_\ell(h_{\ell-1}), \quad \ell \in \{1, \ldots, L\}$$

Quantize后：
$$h_\ell^q = f_\ell^q(h_{\ell-1}^q)$$

注意这里 $f_\ell$ 可以是任意transformation（linear, attention, norm, activation等），非常general。

#### 关键观察
$\Delta h_\ell = h_\ell^q - h_\ell$ 同时受到两个变化的影响：
1. **Input变化**：$h_{\ell-1} \to h_{\ell-1}^q$（来自前ℓ-1层的propagation）
2. **Function变化**：$f_\ell \to f_\ell^q$（当前层的quantization noise）

#### Shapley-style分解
如果先perturb function再perturb input：
$$\Delta h_\ell = \underbrace{(f_\ell^q(h_{\ell-1}^q) - f_\ell^q(h_{\ell-1}))}_{\text{input change under } f_\ell^q} + \underbrace{(f_\ell^q(h_{\ell-1}) - f_\ell(h_{\ell-1}))}_{\text{function change under } h_{\ell-1}}$$

如果先perturb input再perturb function：
$$\Delta h_\ell = \underbrace{(f_\ell^q(h_{\ell-1}^q) - f_\ell(h_{\ell-1}^q))}_{\text{function change under } h_{\ell-1}^q} + \underbrace{(f_\ell(h_{\ell-1}^q) - f_\ell(h_{\ell-1}))}_{\text{input change under } f_\ell}$$

两种分解都是valid的，但代表不同的attribution顺序。为了避免attribution bias，paper借鉴**Shapley value**的思想，对两种分解取平均：

$$\Delta h_\ell = a_\ell + b_\ell$$

其中：
$$a_\ell := \frac{(f_\ell^q(h_{\ell-1}^q) - f_\ell^q(h_{\ell-1})) + (f_\ell(h_{\ell-1}^q) - f_\ell(h_{\ell-1}))}{2}$$
$$b_\ell := \frac{(f_\ell^q(h_{\ell-1}^q) - f_\ell(h_{\ell-1}^q)) + (f_\ell^q(h_{\ell-1}) - f_\ell(h_{\ell-1}))}{2}$$

**直觉解释**：
- $a_\ell$ 完全捕捉 **input变化** 的效果（averaging两种function下的input effect）
- $b_\ell$ 完全捕捉 **function变化** 的效果（averaging两种input下的function effect）

#### 从vector到scalar
$\Delta h_\ell, a_\ell, b_\ell$ 都是向量。为了得到interpretable number，取relative L2 norm：
$$r_\ell := \frac{\|\Delta h_\ell\|}{\|h_\ell\|}$$

这里normalize by $\|h_\ell\|$ 是因为绝对error大小没意义，要看相对规模。

为了能用Law of Cosines，取平方：
$$R_\ell := r_\ell^2 = \frac{\|a_\ell + b_\ell\|^2}{\|h_\ell\|^2}$$

#### ABC分解
用Law of Cosines展开：
$$R_\ell = \underbrace{\left(\frac{\|a_\ell\|}{\|h_\ell\|}\right)^2}_{A_\ell} + \underbrace{\left(\frac{\|b_\ell\|}{\|h_\ell\|}\right)^2}_{B_\ell} + \underbrace{\frac{2\langle a_\ell, b_\ell\rangle}{\|h_\ell\|^2}}_{C_\ell}$$

**各项含义**：
- $A_\ell$：**accumulated error** from previous ℓ-1 layers（通过input变化传导过来的）
- $B_\ell$：**local error** introduced at layer ℓ（function变化）
- $C_\ell$：**interaction term**（两个变化的cross effect）

这就是**ABC decomposition**名字的由来。

#### 实验发现
Figure 3显示：**几乎所有layer都是 $A_\ell \gg B_\ell, C_\ell$**，即 $R_\ell$ 主要由accumulated error主导。

**这彻底解释了为什么MMR没用**：MMR只衡量 $B_\ell$（local outlier），但实际 $R_\ell$ 由 $A_\ell$（accumulated propagation）决定。即使每层local error小，如果层层amplify，最终 $R_L$ 也很大。

---

### 4.3 Gain的进一步分解

定义**gain**：how much a layer amplifies previous error
$$G_\ell := \frac{A_\ell}{R_{\ell-1}}$$

这是衡量"layer ℓ把前ℓ-1层的error放大多少倍"的量。

#### Linear layer的closed-form分解

对于linear layer $h_\ell = W_\ell h_{\ell-1}$，joint quantization后：
$$h_\ell^q = (W_\ell + \varepsilon_\ell^W) h_{\ell-1}^q + \varepsilon_\ell^h$$

其中：
- $\varepsilon_\ell^W$：weight quantization noise
- $\varepsilon_\ell^h$：activation quantization noise（rounding error）

经过代数运算（详见Appendix A.4），可以得到：
$$G_\ell = G_{1,\ell} \cdot G_{2,\ell}$$

其中：

**Spectral ratio**：
$$G_{1,\ell} := \left(\frac{\|W_\ell + \frac{1}{2}\varepsilon_\ell^W\|_*}{\|W_\ell\|_*}\right)^2$$

$\|\cdot\|_*$ 是spectral norm（最大奇异值）。这个比值衡量 **quantization对weight spectral norm的影响**。如果quantization不改变spectral norm（即 $\varepsilon_\ell^W$ 的spectral norm相对 $W_\ell$ 很小），则 $G_{1,\ell} \approx 1$。

注意 $\frac{1}{2}$ 来自Shapley averaging。

**Alignment ratio**：
$$G_{2,\ell} := \left(\frac{\cos \phi_\ell}{\cos \psi_\ell}\right)^2$$

其中：
- $\phi_\ell$：$\Delta h_{\ell-1}$（error direction）和 $(W_\ell + \frac{1}{2}\varepsilon_\ell^W)$（quantized weight）之间的angle
- $\psi_\ell$：$h_{\ell-1}$（original activation）和 $W_\ell$（original weight）之间的angle

**直觉**：
- $\cos \psi_\ell$ 衡量"正常activation和weight有多aligned"——如果activation恰好沿weight的最大奇异向量方向，那么 $\cos \psi_\ell = 1$，layer把这个activation最大化放大
- $\cos \phi_\ell$ 衡量"error direction和quantized weight有多aligned"——如果error恰好沿weight的最大奇异向量方向，error会被最大化放大
- $G_{2,\ell} = (\cos \phi_\ell / \cos \psi_\ell)^2$ 是**相对alignment ratio**：如果error比activation更aligned with weight的主方向，则gain > 1，error被amplify

#### 实验观察
Figure 4显示：
1. **$G_{1,\ell} \approx 1$ for all optimizers**：quantization几乎不改变spectral norm，所以spectral ratio不是主要因素
2. **$G_{2,\ell}$ 是主要差异来源**：alignment决定了gain
3. **Muon的 $G_\ell$ 最高**（linear layer）：解释了为什么Muon虽然MMR低，但PTQ degradation严重——它的layer把error越传越大
4. **AdamW和Shampoo的 $G_\ell$ 最低**：error不被amplify
5. 对于 $\cos \phi_\ell$，Shampoo和AdamW的error direction和weight alignment最低——这是好事情，说明error不在weight的"高增益方向"上

**为什么Shampoo的alignment好？** 一个可能的解释：Shampoo的preconditioner本质上让gradient的different directions按相同的"曲率尺度"更新，weight matrix的singular directions之间会更"均衡"，没有特别dominant的方向让error被selectively amplify。

---

## 5. QAT结果：FP表现不能predict QAT

### 5.1 实验结果
Table 4：4-bit QAT via QuEST，括号里是相对FP的degradation。

| Model | AdamW | Muon | **Shampoo** |
|-------|-------|------|---------|
| 350M | 54.64 (-3.43) | 55.19 (-4.98) | 55.02 **(-2.64)** |
| 500M | 60.07 (-0.53) | 61.05 (-0.73) | 60.80 **(-0.38)** |
| 760M | 62.22 (-2.63) | 62.32 (-3.57) | 62.76 **(-0.46)** |
| 1.5B | 66.82 (-1.63) | 67.08 (-2.11) | 67.34 **(-1.20)** |

**Shampoo在几乎所有size都是degradation最小的optimizer**，但它在FP下表现平平。

Muon虽然在FP下最强，但QAT下degradation也最大（760M上掉3.57%）。

### 5.2 Scaling Law

Paper拟合每个optimizer的scaling law（基于Kumar et al. 2024的precision-aware scaling law）：

$$L = \frac{A}{(N \cdot \rho)^\alpha} + \frac{B}{D^\beta} + E$$

由于实验固定 $D/N = 20$，实际拟合iso-compute version：
$$L = \frac{A'}{(N \cdot \rho)^\alpha} + E$$

**变量解释**：
- $L$：final test loss
- $N$：parameter count
- $D$：training tokens
- $\rho$：**parameter efficiency** — 4-bit QAT下的有效参数比例
- $\alpha$：power law curvature
- $E$：irreducible error
- $A, A', B$：scale constants

**直觉**：$\rho_{4bit} = 0.879$ 意味着4-bit QAT训练的size $N$ 模型，等价于FP训练的size $0.879N$ 模型。

Table 10的关键结果：

| Optimizer | $\rho_{4bit}$ | $E$ |
|-----------|---------------|-----|
| AdamW | 0.863 | 1.40 |
| Muon | 0.852 | 1.85 |
| PSGD | 0.739 | 1.39 |
| Scion | 0.856 | 1.75 |
| **Shampoo** | **0.879** | 1.72 |
| SOAP | 0.822 | 2.22 |

**Shampoo的 $\rho_{4bit} = 0.879$ 最高**，意味着它最efficient地利用parameter capacity under quantization。

注意 $E$（irreducible error）的差异也很有意思：Muon和SOAP的 $E$ 高，说明即使无限scale，它们的asymptote也更差——这可能与optimizer导致的representation structure有关。

---

## 6. 我的intuition构建与发散思考

### 6.1 为什么Shampoo在quantization下最robust？

让我尝试构建一个完整的intuition链：

1. **Shampoo的本质**：维护 $L = \mathbb{E}[g g^T]$ 和 $R = \mathbb{E}[g^T g]$ 两个preconditioner，update形式为 $L^{-1/4} g R^{-1/4}$。这相当于把gradient投影到一个"曲率均衡"的basis里。

2. **为什么这导致weight matrix的结构差异？** Shampoo的update在每个direction上按inverse 4th root of curvature scale，意味着high-curvature directions更新慢，low-curvature directions更新快。结果是weight matrix的singular values分布更**uniform**。

3. **uniform singular values如何帮助quantization？** 回到 $G_{2,\ell} = (\cos \phi_\ell / \cos \psi_\ell)^2$：如果singular values uniform，那么weight matrix接近一个scaled orthogonal matrix。对于这种matrix：
   - 没有特别"high-gain direction"
   - error $\Delta h_{\ell-1}$ 无论在哪个方向，都不会被selectively放大
   - 即 $\cos \phi_\ell$ 不会特别大

4. **为什么Muon反而差？** Muon是把gradient orthogonalize，相当于每一步update都强制在steepest direction。这可能导致weight matrix发展出更anisotropic的结构——某些方向dominant。一旦error恰好align到这些方向，就会被amplify。

### 6.2 与QuIP/QuarOT等rotation方法的关系

Paper里提到的Hadamard transform（QuEST里用的）本质上也是把weight变得"更orthogonal"。所以**Shampoo和Hadamard rotation在某种意义上是殊途同归**：都在追求更isotropic的effective weight structure，从而降低 $G_{2,\ell}$。

这暗示一个更深层的问题：**optimizer本身就能"内置"quantization-robustness**，而不需要后期rotation。这对未来的research有指导意义。

参考：QuarOT (Ashkboos 2024): https://arxiv.org/abs/2404.00456

### 6.3 与Muon/Yang et al. feature learning theory的关系

Yang et al. 2023 ("A spectral condition for feature learning") 提出，Muon-like update（在modular norm下scale invariant）能实现更好的feature learning。这解释了为什么Muon在FP下强。

但**feature learning强 ≠ quantization robust**。Feature learning意味着weight的singular structure anisotropic（少数direction承载feature），这正好是quantization的weak point。这是一个非常深刻的trade-off：

$$\text{Feature learning ability} \quad \leftrightarrow \quad \text{Anisotropic weight structure} \quad \leftrightarrow \quad \text{Quantization vulnerability}$$

参考：https://arxiv.org/abs/2310.17813

### 6.4 与Kurtail、Outlier Suppression等工作的对比

Kurtail (Akhondzadeh 2025) 和 Outlier Suppression (Wei 2022) 都是通过activation regularization或architectural change来reduce kurtosis。Paper的结论对这类方法提出质疑：

**如果kurtosis/MMR不predictive，那么单纯降低这些metrics可能没用**。真正要做的应该是降低 $R_L$，即控制error propagation，这可以通过控制每层的 $G_\ell$ 实现。

实际上，paper的ABC framework给出了一个**principled design objective**：设计regularizer或architecture让 $G_{2,\ell} = (\cos \phi_\ell / \cos \psi_\ell)^2 \leq 1$，即让error alignment小于activation alignment。

参考：Kurtail: https://arxiv.org/abs/2503.01483

### 6.5 关于scaling law的 $\rho$

$\rho_{4bit}$ 量化了optimizer在quantization下的"parameter efficiency"。这个metric可以扩展到：

- 不同bitwidth (2-bit, 3-bit, 8-bit) 的 $\rho_b$
- Different QAT schemes (Straight-through, GPTQ-style, etc.)
- 与activation quantization vs weight-only quantization的对比

Paper提到future work会研究microscaling format (MXFP4)的error propagation，这是H100/H200硬件原生支持的格式。

### 6.6 关于Common Loss (CL)的实验设计巧思

Paper的一个实验设计细节很精妙：定义CL为"所有optimizer都能达到的最低loss"。这确保了**所有model在pre-quantization时的downstream performance相同**，所以post-quantization的差异**纯粹**来自quantization robustness，而不是FP training的差异。

这是一个非常好的controlled experiment设计。但潜在limitation是：CL是suboptimal的（ strongest optimizer被handicapped），如果换成"每个optimizer各自最优checkpoint"做PTQ，结论可能不同。Paper选择CL是为了isolate quantization effect。

### 6.7 关于Newton-Schulz迭代的computational cost

Appendix A.5详细列了所有optimizer的complexity。Muon/Scion的Newton-Schulz迭代：
$$X' = aX + b(XX^T)X + c(XX^T)^2X$$

每次迭代是 $O(2m^2n + m^3)$（假设 $m \leq n$），T次迭代总complexity $O(T(2m^2n + m^3))$。Table 9显示Muon和AdamW的wall-clock time几乎一样（Muon稍慢），这说明Newton-Schulz在GPU上overhead很小（因为 $m$ 通常远小于 $n$，比如 $m=768, n=3072$）。

Shampoo则慢得多（760M上8.38s vs AdamW 6.28s，慢33%），因为需要维护两个preconditioner并做eigendecomposition。

---

## 7. Limitations和可能的extension

1. **只测了W4A4**：8-bit、6-bit、MXFP4未测。不同bitwidth下optimizer ranking可能变化（比如2-bit可能完全不同）。
2. **CL setup的limitation**：如上所述，handicaps strong optimizer。
3. **只分析了linear layer的gain**：Attention module的gain分解还没做，paper说留作future work。Attention的softmax会让error propagation更复杂（非线性amplify）。
4. **没有理论解释为什么Shampoo产生low-gain weights**：只有实验观察。一个理论分析可以是：Shampoo的stationary point性质（什么时候 $L^{-1/4} g R^{-1/4} = 0$）与weight的singular spectrum shape的关系。
5. **Scale up**：1.5B还不够大。Moonshot级别的model（100B+）可能涌现不同的pattern。
6. **没测attention quantization**：W4A4 quantize了linear layer，但attention softmax是FP的。如果quantize attention（比如K/V cache quantization），Muon的high gain可能更显灾难。

---

## 8. 对实践者的建议

1. **如果要deploy quantized model，用Shampoo训练**（虽然FP稍弱，但PTQ/QAT loss小）
2. **如果纯FP inference，用Muon**
3. **传统outlier metrics（MMR/Kurtosis）不要再看**，改用 $R_L$ 作为diagnostic
4. **设计quantization-friendly architecture**时，关注如何让 $G_{2,\ell} = (\cos \phi / \cos \psi)^2 \leq 1$
5. **Pretraining compute overhead** Shampoo比AdamW多33%，但deployment节省更多——trade-off合理

参考相关论文：
- Optimizer benchmarking: https://arxiv.org/abs/2509.02046
- Scaling laws for precision: https://arxiv.org/abs/2411.04330
- Scion: https://openreview.net/forum?id=2Oqm2IzTy9
- SOAP: https://openreview.net/forum?id=IDxZhXrpNf

---

## 9. 总结

这篇paper的beauty在于：**用一个简单的Shapley-style decomposition，揭示了整个quantization community追了多年的"outlier"其实是错的方向**。真正的driver是error propagation，由weight matrix的alignment structure决定。Optimizer通过塑造weight的singular structure，间接决定了quantization robustness。

Shampoo的胜利，从深层看，是"曲率均衡"哲学对"steepest descent"哲学的胜利——在quantization约束下，平衡比激进更稳健。

如果让我bet，**未来quantization-friendly optimizer的设计会explicitly optimize $G_{2,\ell}$**，可能通过某种spectral regularization或structural constraint实现。这paper为这条路打下了theoretical foundation。
