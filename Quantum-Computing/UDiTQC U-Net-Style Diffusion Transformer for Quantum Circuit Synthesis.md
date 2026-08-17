---
source_pdf: UDiTQC U-Net-Style Diffusion Transformer for Quantum Circuit Synthesis.pdf
paper_sha256: ff9db2cb484d5288107bfdd92d148b9f55216e43e8503296edc6c4ca9bf15fec
processed_at: '2026-08-12T18:53:27-07:00'
target_folder: Quantum-Computing
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 UDiTQC

Andrej 你让我用"人话"讲，那我就想象咱俩在 NeurConfig 白板前聊这篇 paper，不带学术八股，讲讲这帮人到底在搞什么、为什么这么搞、搞得好不好。

---

## 一句话总结

这帮人想用 diffusion model 自动生成 quantum circuit，之前的 [GenQC](https://www.nature.com/articles/s42256-024-00920-8) 用的是 U-Net 当 backbone，效果一般般。他们就把 U-Net 的骨架留着，但把每个 block 换成 [DiT (Diffusion Transformer)](https://arxiv.org/abs/2212.09748) 的 adaLN-Zero block，再做一些工程小 trick（residual 替代 skip、decoder 加厚、中间层加宽），结果在 entanglement generation 和 unitary compilation 上都比 GenQC 涨了几个点。架构本身没太多新东西，主要是工程整合 + quantum 应用层的 encoding 设计。

---

## 这事为什么有意思

先说 quantum circuit synthesis 是个什么活儿。

你有一堆 qubit，你想让它们做完一番操作之后达到某个目标状态——比如产生某种特定的 entanglement pattern，或者实现某个目标 unitary matrix。你得告诉 quantum computer：第一步放什么 gate、第二步放什么 gate、第三步放什么 gate……一套 sequence 跑下来，才能得到你要的东西。

这个事情传统上是靠人手设计 + [Solovay-Kitaev](https://arxiv.org/abs/quant-ph/0505030) 这种算法逼近，或者靠 [reinforcement learning](https://www.nature.com/articles/s42005-021-00633-5) 慢慢学。麻烦在于：gate 种类很多、qubit 之间的 entanglement 关系是高度非局部的、远距离 qubit 想连还得靠 SWAP 链。这本质上是个**结构化、离散、带物理约束的组合生成问题**。

[GenQC (Furrutter et al. 2024)](https://www.nature.com/articles/s42256-024-00920-8) 这个工作的 insight 是：诶，diffusion model 现在不是 image generation SOTA 吗？我把 quantum circuit 编码成 2D tensor（一个维度是 qubit index，一个维度是 time step），然后让 DDPM 来生成这种 tensor 不就完事了？给个 condition（"我要 SRV = [2,2,1]"），模型就吐出对应的 circuit。这思路挺巧，相当于把 quantum circuit 当 image 来做。

结果证明确实 work，但是 GenQC 用的 backbone 是 U-Net，就那套 [Ronneberger 2015](https://arxiv.org/abs/1505.04597) 的老古董——conv → downsample → conv → bottleneck → upsample → conv → skip connection。问题在：

1. U-Net 的 attention 只在 bottleneck 处，token 数多了之后 attention 计算爆炸，而且 attention 只能看 highly-compressed 的 feature，long-range 建模其实不彻底
2. U-Net 的 conv 归纳偏置偏局部，但是 quantum entanglement 是一个 CNOT 的 control 和 target 隔老远都能纠缠上的事情，conv 的 locality 假设不匹配
3. 数据稀疏——一个 5-qubit、28-gate 的电路 tensor 是 5×28 的矩阵，大部分位置是 padding/background，conv 在上面浪费 capacity

那么 [DiT (Peebles & Xie 2023)](https://arxiv.org/abs/2212.09748) 呢？DiT 在 image generation 上把 U-Net 干翻了，纯 Transformer + adaLN-Zero + full attention，[ImageNet 256 FID SOTA](https://arxiv.org/abs/2212.09748)。直接套到 quantum 上行不行？

行，但不够。原因：DiT 是 single-scale 的，所有 token 同一 resolution 走到底。对 image 这没问题，因为 image pixel 是 dense 的。但是 quantum circuit 是稀疏的，纯 DiT 把大量 attention capacity 花在 padding 上，对有效 gate 之间的建模反而被稀释。

所以这帮人就想：**那我把 U-Net 的金字塔骨架留着，把里面的 conv block 全换成 DiT block，不就 multi-scale + global attention 两手抓了吗？** 这就是 UDiT 的核心 idea。

---

## UDiT 架构到底怎么搭的

### 大骨架：U-Net 风格

你想象一个 5-stage 的 U 字形：
- Encoder：input sequence → DiT layer → downsample → DiT layer → downsample
- Bottleneck：中间一个 DiT layer
- Decoder：DiT layer → upsample → DiT layer → upsample → DiT layer → output

每一段 DiT layer 里是 N 个 DiT block 串联。Downsample 用 conv（stride=2 的卷积），upsample 用 interpolation + conv 还原。

这是传统 U-Net 的拓扑——多尺度、压缩再展开、保留中间 bottleneck。但内部 block 全换成 Transformer。

### 小 block：DiT 的 adaLN-Zero

每个 DiT block 长这样：

```
input
  ↓
adaLN（用 timestep + condition 回归出 γ, β）
  ↓
Multi-Head Self-Attention
  ↓
× α  (α 初始化为 0)
  ↓
+ input  (residual)
  ↓
adaLN（再用一次 γ', β'）
  ↓
FFN
  ↓
× α'  (α' 也初始化为 0)
  ↓
+ 上一段输出  (residual)
  ↓
output
```

这里的 adaLN-Zero 是 [DiT 原论文](https://arxiv.org/abs/2212.09748) 那个 trick。γ, β, α 三个参数都是用一个 MLP 从 (timestep embedding + class embedding) 回归出来的，关键是 α 初始化为 0，意味着训练刚启动的时候整个 block 是 identity——输入啥输出啥，啥都不干。

为什么要这样？Karpathy 你肯定秒懂：Transformer 训练初期如果 block 直接就开始 attention + FFN，gradient flow 容易乱。让 block 初始化成 identity，相当于训练开始时整个网络是个浅网络，gradient 顺顺当当流到底，loss 下降稳定；训练中后期 α 逐渐 learn 出非零值，block 开始真正干活，网络"深度"逐渐展开。这就是 implicit depth curriculum。

直觉上就是：让网络先学会"什么都不做"是 OK 的，再慢慢学会"做点修改"，再学会"做大修改"。比一开始就强迫它做完整 transformation 友好得多。

公式上：

$$h_{out} = \alpha \cdot \text{Attn}(\gamma \cdot \text{LN}(h_{in}) + \beta) + h_{in}$$

- $h_{in}$: 输入
- $\gamma, \beta, \alpha$: 由 MLP 从 $(t_{emb} + c_{emb})$ 回归出来
- $\gamma, \beta$: LayerNorm 的 scale 和 shift
- $\alpha$: 残差路径的 gate，初始为 0
- $\text{LN}$: LayerNorm
- $\text{Attn}$: multi-head self-attention

conditioning 信息就这样通过 adaLN 注入到每个 block 里——timestep 和 class label 的 embedding 加起来，过 MLP，输出 6 个参数（attention 路径 3 个 + FFN 路径 3 个），各管各的。

### 关键创新点：用 residual 替代 long skip

传统 U-Net 你熟的：encoder 第 $i$ 层的输出直接通过 long skip connection 跨过 bottleneck，concat 或 add 到 decoder 第 $i$ 层的输入。这是 U-Net 多尺度信息融合的核心机制。

UDiT 不这么干。它说：long skip 让信息流变得不规整，破坏了 Transformer 的纯 residual stack 美感。LLM 训出来都是纯残差，没有 long skip，照样 work。所以我用 residual connection 重写这个 U 形。

公式 4 是这样：

$$f(x) = f_u\left(U\left(f_m(D(h)) - D(h)\right) + h\right)$$

变量含义：
- $h = f_d(x)$: encoder 输出
- $f_d, f_m, f_u$: encoder / middle / decoder 三段
- $D$: downsample 操作（conv）
- $U$: upsample 操作（interpolation + conv）
- $f_m(D(h)) - D(h)$: middle 段学到的"差异"

意思是：middle 段不是直接给你一个 output，而是给你一个"对 $D(h)$ 的修改量"。然后把这个修改量 upsample 回去，加回 $h$ 上，再过 decoder。

这是个挺巧的写法。本质上 long skip 还在，但形式上变成了纯残差——整个 U 形相当于一个 residual function：$f(x) - x$ 就是这个 U 形学到的 transformation。

训练初期 $f_m(D(h)) \approx D(h)$（因为 adaLN-Zero 让 block 都是 identity），所以 $f_m(D(h)) - D(h) \approx 0$，整个网络近似浅残差网络。训练后期 middle 段学到东西，"差异"逐渐变大，网络深度逐渐显形。这和 adaLN-Zero 的 implicit depth curriculum 是协同的，两层 curriculum 叠加。

参考 [U-ViT (Bao et al. 2023)](https://arxiv.org/abs/2301.11093) 和 [Simple Diffusion (Hoogeboom 2023)](https://arxiv.org/abs/2301.11093) 是一脉相承的——这帮欧洲组搞 image diffusion 时也观察到 long skip 退化成 residual 反而更好。UDiT 在 quantum 上验证了同样的事情。

### Decoder 加厚

UDiT 是 asymmetric 的：encoder（downsampling 段）放少量 DiT block，decoder（upsampling 段）放更多。

为什么？因为 encoder 干的活是"压缩、抽象"——这个相对容易，几层 attention 就能把 sequence 关键信息抓出来。decoder 干的活是"展开、具体化"——它得从压缩表示还原出完整的 circuit tensor，这个难度高得多，需要更多 capacity。

这和 GPT 这种 decoder-only LLM 比 BERT encoder-only 强是一回事：生成任务上 decoder 需要更多算力。论文里引用了 [Vaswani 2017](https://arxiv.org/abs/1706.03762) 和 [Hoogeboom 2023](https://arxiv.org/abs/2301.11093) 支持这个说法。

### 中间层加宽

每个 stage 的 hidden feature dim 和 attention head 数量是可以分别设的。UDiT 在最压缩的中间层用更高 feature dim + 更少 attention heads。直觉上：sequence length 在 bottleneck 已经很短了，每个 token 需要承载更多信息（因为压缩过），所以 channel 加宽；同时 attention head 数减少，每个 head 维度变大，更适合捕获粗粒度全局关系。

这个 trick 在 [Table 1](https://arxiv.org/abs/2410.19324) ablation 里很神奇：之前几个改动让训练速度从 10.56 慢慢降到 8.56 steps/s，结果一加 "+Fixed Hidden Feature"，速度反跳到 15.5 steps/s（最快）。说明 attention 的计算瓶颈其实不在 sequence length，而在 hidden dim × heads 的 channel 复杂度。固定 dim 并减 heads 把 attention 成本压下去，整个网络就快了。

这给我一个 intuition：在 diffusion transformer 里，attention 的 channel 复杂度比 spatial 复杂度更主导。所以光搞 [FlashAttention](https://arxiv.org/abs/2205.14135) 这种 spatial 优化还不够，channel 维度 budgeting 也很重要。

---

## 怎么把 quantum circuit 塞进神经网络

这部分是 quantum application layer 的工程，我觉得挺有意思。

### 电路 → 2D tensor

一个 quantum circuit 长这样：横轴是 qubit，纵轴是 time step，每个位置要么是个 gate 要么是空（idle）。Paper 把它编成 2D tensor，shape `[n_qubits, n_timesteps]`，每个时间步**只允许放一个 gate**（强约束）。

每个 gate 用一个**正交高维连续 embedding** 表示。比如你有 H、CX、Z、X、CCX、SWAP 这 6 个 gate，那就有 6 个正交向量 $v_1, \dots, v_6 \in \mathbb{R}^d$，$d = N + 2$（+2 留给 padding 和 background）。

技巧在 multi-qubit gate：CNOT 有 control 和 target 两个节点，怎么办？用**同一个 embedding，符号相反**。所以 CNOT 的 control 节点位置放 $+v_{CNOT}$，target 节点位置放 $-v_{CNOT}$。这样从 sign 就能区分 control/target。

### Decoding：从连续向量找回 gate

模型 inference 完输出一个连续 tensor，得把它映射回 gate token。方法是 cosine similarity：

$$\tilde{k} = \arg\max_k |S_C(v_k, v_{gen})|$$

先取**绝对值**最大的——找最匹配的 gate 种类。然后：

$$k = \tilde{k} \cdot \text{sign}\, S_C(v_{\tilde{k}}, v_{gen})$$

用 sign 决定是 control 还是 target（正还是负）。

这就是把一个连续向量"翻译"回离散的 gate + role。如果模型把两个 gate 放同一时间步，或者 control/target 配错，就记为 error circuit。

### 一个有趣的现象：vector space "overload"

Paper Appendix A 提到一个 surprising observation：即使 gate 种类 $N$ 比 embedding dimension $d$ 还多（$N > d$），模型仍能准确 match embedding。

按理论说，$d$ 维空间里最多有 $d$ 个正交向量，$N > d$ 就不可能正交了。但模型还是能区分——这意味着模型没真用 orthogonal 几何，而是学到了一个非线性 decoder，把 $v_{gen}$ 投影到最近的 $v_k$ 上。

这其实有点像 [VQ-VAE codebook](https://arxiv.org/abs/1711.00937) 的反问题：codebook collapse 是多个 code 退化成一个，overload 是一个 code 表示多个 mode（这里是从 sign 区分）。Paper 没深挖这个现象，但我直觉这背后可能有更深的几何——比如模型在 $\mathbb{R}^d$ 上学到了一个非线性的 manifold embedding，gate 在这个 manifold 上是可分的，即使不严格正交。

### Patchify

2D tensor 搞定后，还得 sequence 化才能喂给 Transformer。用的是 ViT/DiT 的 patchify 思路：把 `[n_qubits, n_timesteps]` 切成 patch，flatten 成长度 $K = \max_{qubits} \times \max_{gates}$ 的 fixed-length sequence。每个 token 是一个 patch embedding。

加 sinusoidal 2D positional embedding（sine-cosine 那套），让 transformer 同时知道每个 token 在 qubit 维和时间维的位置。

Paper §2.3 提了 [FiT (Lu 2024)](https://arxiv.org/abs/2402.12376) 和 [VisionLLaMA (Chu 2024)](https://arxiv.org/abs/2403.00522) 用的 RoPE2D，但最终选了 sinusoidal。我猜是因为 quantum circuit 的长度范围窄（最多 8 qubits × 52 gates），RoPE 的 extrapolation 优势用不上，sinusoidal 更简单稳定。

---

## Conditioning 怎么做

UDiTQC 有两种 condition：

### 1. Class label（SRV 或 gate set subset）

对 entanglement generation 任务，condition 是 Schmidt Rank Vector (SRV)——一个向量，每个 subsystem 是 1（separable）或 2（entangled），描述这个 circuit 产生的 entanglement 结构。fixed qubits 下 SRV 种类有限，可以直接编号成 class label，比如 3-qubit 有 5 种 SRV，5-qubit 有 27 种，8-qubit 有 121 种。

对 unitary compilation 任务，condition 是 gate set subset——比如 "用 H + CX 编译" 还是 "用 H + CX + T + CCX 编译"。每种 subset 是一个 class label。

Class label 通过一个 `LabelEmbedder` 嵌入成向量。训练时用 **label dropout**：随机把 label 替换成一个 learnable null token ∅，让模型同时学 conditional 和 unconditional 分布。这是 [Classifier-Free Guidance (Ho & Salimans 2022)](https://arxiv.org/abs/2207.12598) 的标准做法。

### 2. Unitary embedding（U-enc）

对 unitary compilation，光知道 gate set 还不够，还得知道**要编译哪个 unitary**。给一个 $2^n \times 2^n$ 的复数矩阵 $U$，模型得生成实现这个 $U$ 的 circuit。

怎么把 unitary 嵌入？用一个小 Transformer encoder（叫 U-enc）：
1. 把复数矩阵拆成 real part 和 imag part 两个 channel
2. Conv 提取局部特征
3. 加 2D positional encoding（编码矩阵元素的绝对位置）
4. Transformer encoder（self-attention，因为 unitary 矩阵任意元素都和其它所有元素耦合，信息是非局部的，必须用全局 attention）
5. 2×2 downsample + conv 扩展到 hidden dim

U-enc 和 UDiT 一起 jointly train。为了防止 overfitting（因为 unitary 空间很大但每个 unitary 在训练集里对应的 circuit 实现不多），U-enc 内部用 dropout。

最后 U-enc 输出和 time embedding + label embedding **concat** 起来，过 linear 投到统一维度，作为整个 condition vector 送进每个 DiT block 的 adaLN。

---

## 训练和推理细节

### 训练

标准 DDPM ε-prediction loss：

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon_t}\left[\|\epsilon_t - \epsilon_\theta(x_t, t, c)\|_2^2\right]$$

- $t$: 均匀采样的 timestep，$t \sim \mathcal{U}[0, T]$
- $x_0$: clean circuit tensor
- $\epsilon_t \sim \mathcal{N}(0, I)$: 加的 Gaussian 噪声
- $\epsilon_\theta(x_t, t, c)$: UDiT 预测的噪声，以 $t$ 和 condition $c$ 为条件
- 训练目标是让模型预测的噪声 $\epsilon_\theta$ 逼近真实噪声 $\epsilon_t$

Variance schedule 用 squared cosine（[Improved DDPM, Nichol & Dhariwal 2021](https://arxiv.org/abs/2102.09772)），$T = 1000$ 步。

Optimizer 是 [AdamW (Loshchilov 2017)](https://arxiv.org/abs/1711.05101)，learning rate 用 one-cycle policy（[Smith & Topin 2019](https://doi.org/10.1117/12.2520589)），初始 $3 \times 10^{-4}$，训 300 epoch。

### 推理

从 $\mathbf{x}_T \sim \mathcal{N}(0, I)$ 开始，跑 DDPM 反向过程。但用 [DDIM (Song et al. 2020)](https://arxiv.org/abs/2010.02502) 加速：实际只需要 50-100 步 denoise 就能生成质量 OK 的 sample。

用 **rescaled classifier-free guidance**：

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

- $\epsilon_\theta(x_t, \emptyset)$: null condition 输出
- $\epsilon_\theta(x_t, c)$: 真实 condition 输出
- $s$: guidance scale, paper 用 $s = 7.5$
- $s = 1$ 退化成标准 conditional sampling
- $s > 1$ 强化 condition 信号，但太大易 mode collapse

"Rescaled" 指 [Ho & Salimans 2022](https://arxiv.org/abs/2207.12598) 里说的高 guidance 时要 rescale variance，避免 under-exposed samples。

对每个 condition，生成 1024 个 sample circuits，统计有多少是 valid 且满足目标 SRV/unitary 的。

---

## 实验结果怎么样

### Entanglement Generation

数据集：H + CX 两个 gate，3-8 qubits，gate 数 16 到 52 不等。SRV 作为 condition。

[Figure 3](https://arxiv.org/abs/2410.19324) 的结果：UDiTQC 在 3-8 qubits 上一致超过 GenQC。差距在 2-5 个百分点，qubits 多了之后差距更大——这印证了 UDiT 的多尺度+全局建模在长 circuit 上更优势。

[Figure 4](https://arxiv.org/abs/2410.19324) 是 5-qubit 的 confusion matrix，行是 input SRV、列是 generated SRV。对角线很强，主误差在"少 entangled → 多 entangled"方向——这符合直觉，因为生成高 entanglement 需要更多 gate，模型容易漏 gate 导致 entanglement 不够。

模型还能 generate 训练集外的 novel circuit——泛化性 OK，不是纯 memorization。

### Masking（Figure 5a）

这是 quantum-specific 的应用：量子芯片物理上 qubit 连接有限，远距离 qubit 想做两-qubit gate 得靠 SWAP 链。Masking 就是把 input tensor 的某些位置强制为 0（白区），模型 inference 时就不会在那里放 gate。

听起来很美，但 paper Appendix C.2 自己爆了个雷：**masking 任务的 error rate 从 <1% 飙到 ~80%**。也就是说 80% 生成的 circuit 是 invalid——要么两个 gate 撞同一时间步，要么 control/target 配错。

这是 diffusion model 处理 hard constraint 的根本问题。DDPM 没有显式 constraint enforcement 机制，全靠 learned distribution 来 honor constraint。遇到 OOD constraint（比如训练时没见过的 mask pattern），marginal 分布就偏离了 valid circuit manifold。

[Repaint (Lugmayr 2022)](https://arxiv.org/abs/2201.09865) 试图解决这个但没根治。我直觉这个问题用 [Discrete Diffusion (D3PM, Austin 2021)](https://arxiv.org/abs/2107.03006) 或者 [MaskGIT (Chang 2022)](https://arxiv.org/abs/2202.04200) 这种直接在 token 空间做的会更合适，因为它们天然能 enforce 结构约束。

### Editing（Figure 5b, 6）

固定 circuit 前几 gate（作为"初始量子态"），让模型在后续位置 generate gate 来达到目标 SRV。

Figure 6 是个 input SRV → target SRV 的转换矩阵：
- 从低 entanglement → 高 entanglement（对角线上方）容易，>85% 成功
- 从高 → 低 entanglement（对角线下方）难

为啥高→低难？因为模型得先"untangle"已纠缠的 qubit，再"re-tangle"成目标 SRV，gate sequence 复杂得多。这就像把一个凌乱的房间整理成另一个凌乱的房间，比把空房间布置成凌乱的房间难。

整体 85.2% 成功率，比 masking 友好得多。

### Unitary Compilation（Figure 7）

这是 paper 最有 practical impact 的实验。

Gate pool: $\{H, CX, Z, X, CCX, SWAP\}$，3 qubits，gate 数 2-12。Condition 是 gate set subset + 目标 unitary matrix（通过 U-enc）。

测试集是 5000+ 个 unseen unitaries，每个生成 1024 个 candidate circuits。Metric 是：
- Compilation accuracy: 电路是否真的实现了目标 unitary
- Frobenius norm: $\frac{1}{2}\|U_t - U_g\|_F^2$ 衡量距离
  - $U_t$: 目标 unitary matrix
  - $U_g$: 生成 circuit 对应的 unitary

结果：
- **UDiTQC: 94.9%** accuracy
- **GenQC: 92.6%** accuracy
- 大部分 unitary 编译出 Frobenius norm = 0（完美匹配）
- 即使 nonzero 的 norm 也远低于 random baseline
- 每个 unitary 通常能生成多个 distinct circuit——给用户选择空间（gate depth vs fidelity tradeoff）

这非常有 practical 价值。传统 quantum compilation 用 [Solovay-Kitaev](https://arxiv.org/abs/quant-ph/0505030) 或 [RL (Moro 2021)](https://www.nature.com/articles/s42005-021-00633-5)，UDiTQC 提供了第三条路：**diffusion model 当 compiler**，inference 时 batch 出多个候选。

我直觉这条路大有可为，因为 diffusion model 天然是"一对多"生成——给一个 unitary，能 batch 出 N 个不同的 circuit 实现，用户可以根据自己的 hardware 约束（gate depth、fidelity、connectivity）选最合适的。RL 这种 sequential 方法很难 batch 出多个候选。

### Ablation Study（Table 1）

3-qubit entanglement generation 上 8 个 variant：

| Model | Speed (steps/s) | Avg Acc (%) | Entangled Acc (%) |
|---|---|---|---|
| DiT (base) | 10.56 | 64.08 | 44.1 |
| DiTsq (sequence embedding) | 15.1 | 77.52 | 64.75 |
| U-Net-Style DiTsq | 10.74 | 79.18 | 69.2 |
| U-Net-Style DiT | 9.24 | 84.14 | 56.06 |
| + Asymmetric | 9.87 | 85.1 | 60.54 |
| + Residual Connections | 8.56 | 86.2 | 61.97 |
| + Hidden Feature Expansion | 9.9 | 82.4 | 59.14 |
| + Fixed Hidden Feature (final) | **15.5** | **89.12** | 65.72 |

几个有意思的观察：

1. **纯 DiT 在 quantum task 上弱（64%）**——说明 multi-scale 真的有用，单 scale 全 attention 不够
2. **DiTsq（sequence gate embedding）大幅提升到 77.5%**——encoding 方式很重要，每个时间步的 gate 用 sequence 表示比 single embedding 强
3. **U-Net-Style + Residual + Asymmetric 一路涨**，但 speed 从 10.56 降到 8.56——架构复杂化是有 cost 的
4. **+Fixed Hidden Feature 后 speed 反弹到 15.5（最快）且 accuracy 涨到 89.12（最高）**——这是 killer trick，固定 feature dim + 减 attention head 数既快又准
5. **Entangled Accuracy（最难 case，full entanglement）不是单调提升的**：DiTsq 系列高（69.2），但 final UDiT 只有 65.72。说明 final 设计对 avg accuracy 友好但对极端 case 可能 over-regularized

Karpathy 你看这个 ablation 表会想到啥？我直觉是：每个 trick 单独加都有用，但 trick 之间有 interference。U-Net-style + sequence embedding 可能有冗余（都做 multi-scale），所以 U-Net-Style DiTsq 反而比 U-Net-Style DiT avg accuracy 低（79 vs 84）。这种 interference 没在 paper 里深挖，留给后续工作了。

---

## 数据集怎么造

[Table 2](https://arxiv.org/abs/2410.19324) 列了所有训练集参数：

| Qubits | Gate Pool | Min/Max Gates | # Labels | Total Circuits |
|---|---|---|---|---|
| 3 | H, CX | 2/16 | 5 | 200K |
| 4 | H, CX | 3/20 | 12 | 300K |
| 5 | H, CX | 4/28 | 27 | 459K |
| 6 | H, CX | 5/40 | 58 | 470K |
| 7, 8 | H, CX | 6/52 | 121 | 484K |
| 3 (unitary) | H,CX,Z,X,CCX,SWAP | 2/12 | 63 | 925K |

数据集生成流程：
1. 随机采样 gate 数（uniform 分布）
2. 随机采样 gate 类型（from gate pool）
3. 用 [Qiskit transpile](https://docs.quantum.ibm.com/api/qiskit/qiskit.compiler.transpile) 优化（合并 redundant gate、删除 redundant qubit）
4. 去重
5. 对 SRV 类别做 class balancing（随机生成时全 1 或全 2 占多，得平衡）

# Labels 公式 $num = 2^{q-1}$ 在 $q=3$ 给 5（实际是 SRV 可能的 rank vector 种类数，跟 [Schmidt decomposition](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.030501) 有关）。

---

## 我读出来的几个 limitation

1. **Masking 80% error rate 是大问题**。Diffusion 在 hard constraint 下不稳定，paper §C.2 自己承认。需要后续工作解决，可能要用 [constrained diffusion](https://arxiv.org/abs/2402.13172) 或 [structure-aware sampling](https://arxiv.org/abs/2305.13274)。

2. **Max 8 qubits 离实用规模（50-100 qubits）还远**。Sequence length $K = \max_{qubits} \times \max_{gates}$ 会爆炸，paper 没讨论 scaling 到大规模的可行性。

3. **Gate set 太小**。Entanglement generation 只用 H + CX，没测 T gate 或 arbitrary RZ。Unitary compilation 用 6 种 gate 但仍然小 gate set，离实用 universal gate set 还远。

4. **没和 non-ML compiler 比**。只比了 GenQC，没和 [Qiskit transpiler](https://docs.quantum.ibm.com/api/qiskit/qiskit.compiler.transpile) 或 [t|ket⟩](https://github.com/CQCL/tket) 这种传统编译器比，不知道 ML 方法到底比传统强多少。

5. **"Overload" 现象没理论解释**。$N > d$ 仍 work 是 empirical surprise，paper 没深挖。这背后可能有更深的几何，比如模型学到了一个非线性 manifold embedding。

6. **Frobenius norm 不直接等价 unitary fidelity**。$\|U_t - U_g\|_F$ 不归一化，应该用 [process fidelity](https://en.wikipedia.org/wiki/Quantum_fidelity) $\frac{1}{d}|\text{tr}(U_t^\dagger U_g)|$ 更标准。Paper 选 Frobenius 可能是为了计算方便，但严格 quantum 信息意义上不是最佳 metric。

---

## Karpathy-style 几个直觉思考

### 1. Quantum + Diffusion 的根本张力

Diffusion 假设数据分布是连续 Gaussian noise 上的 manifold。但 quantum circuit 是**离散 + 严格结构约束**的对象：gate 种类有限、gate 之间有 control/target 配对关系、每个 timestep 只能放一个 gate。

Paper 用 orthogonal continuous embedding 做了 hack，把离散 structure "假装成" continuous。这能让 DDPM 跑起来，但本质上是不匹配的。

更彻底的方案：
- **Discrete diffusion** ([D3PM, Austin 2021](https://arxiv.org/abs/2107.03006)) 直接在 token 空间做
- **Masked generative** ([MaskGIT, Chang 2022](https://arxiv.org/abs/2202.04200), [VQ-Diffusion, Gu 2022](https://arxiv.org/abs/2111.14822)) 用 mask-and-predict
- **Equivariant diffusion** ([EDM, Hoogeboom 2022](https://arxiv.org/abs/2202.03026)) 把 circuit 的对称性（gate permutation、qubit relabeling）编进去

UDiTQC 选了 continuous diffusion 这条路，简单 work，但 masking 80% error 暴露了根本张力。

### 2. Residual 替代 Skip 的本质

公式 4 的 $f_m(D(h)) - D(h)$ 是个 residual function：把"中间层学到的 transformation"显式表达成"对输入的修改"。

这是 LLM 残差风格的延伸——所有 transformation 都是 identity + delta。训练初期 delta≈0，模型是浅的，gradient flow 好；训练后期 delta 增长，模型变深，capacity 释放。这种 implicit depth curriculum 比 U-Net 的 hard skip concat 更平滑。

我直觉这个 trick 在大模型上会更显著。LLM 训练经验告诉我们：纯残差 stack 的 scalability 比带 skip 的网络好得多——GPT 这种 thousand-layer 网络能训起来，U-Net 这种带 long skip 的网络很难堆深。UDiT 把 U-Net 拓扑用 residual 写一遍，相当于让 U-Net 也获得 LLM 的 scalability。

### 3. Attention 的 channel vs spatial 复杂度

Ablation 表里 +Residual 阶段 speed 降到 8.56，+Fixed Hidden Feature 一加反跳到 15.5。这给我一个重要 intuition：**attention 的计算成本主要来自 channel 维度（hidden dim × heads），不是 spatial 维度（sequence length）**。

[FlashAttention (Dao 2022)](https://arxiv.org/abs/2205.14135) 优化的是 spatial 复杂度（IO-aware），但 channel 复杂度还是 $\mathcal{O}(d^2)$（$d$ 是 hidden dim）。UDiT 的 +Fixed Hidden Feature 实际上是 channel budgeting——固定 dim，减 heads，每个 head dim 变大但总 channel 复杂度可控。

这暗示了 diffusion transformer 加速的另一条路径：除了 spatial 优化（FlashAttention、sparse attention、linear attention），channel 维度的 budgeting 也很重要。比如 [Mixture-of-Depths (MoD, Raposo 2024)](https://arxiv.org/abs/2404.02258) 这种 per-token routing 也算 channel budgeting 的一种。

### 4. Quantum Application 的特殊性

我读这篇 paper 最大的收获其实不是架构创新（UDiT 本身不算 radical），而是看到 diffusion model 怎么 apply 到一个完全不同 domain 的工程艺术。

Quantum circuit 作为数据有特殊性：
- **稀疏**：大部分位置是 padding
- **结构化**：control/target 配对、timestep 约束
- **非局部**：entanglement 是高度全局属性
- **离散**：gate 种类有限
- **对称性**：qubit relabeling、gate commutation

Paper 用 orthogonal embedding 解决离散，用 sign 解决 control/target 配对，用 multi-scale U-Net 解决稀疏，用 full attention 解决非局部。但**没解决对称性**——qubit relabeling 应该不影响 circuit 功能，但 UDiTQC 没显式 encode 这个。这可能是未来工作的大方向：[Equivariant Diffusion (Hoogeboom 2022)](https://arxiv.org/abs/2202.03026) 那套思想直接套到 quantum circuit 上。

### 5. "Overload" vector space 现象

Paper §A 提到：即使 $N > d$（gate 数比 embedding dim 多），模型仍能准确 match embedding。这是个 fascinating 的现象。

按线性代数说，$d$ 维空间最多 $d$ 个正交向量，$N > d$ 不可能正交。但模型还是能区分——说明它学到了一个非线性 decoder，把 $v_{gen}$ 投影到最近 $v_k$。这有点像 [VQ-VAE codebook](https://arxiv.org/abs/1711.00937) 的逆问题：codebook collapse 是多个 code 退化成一个，overload 是一个 code 表示多个 mode。

我直觉这背后是模型在 $\mathbb{R}^d$ 上学到了一个非线性 manifold embedding，gate 在这个 manifold 上是可分的，即使不严格正交。类似 [Word2Vec](https://arxiv.org/abs/1301.3781) 里 word vector 不是正交但语义可分。这个现象值得理论分析，paper 没做。

### 6. Diffusion 作为 Compiler 的范式

Unitary compilation 实验让我觉得最有 practical 价值。传统 quantum compilation 是 sequential 算法（Solovay-Kitaev、RL），一次只能出一个 circuit。Diffusion model 天然"一对多"——给一个 unitary，能 batch 出 N 个不同 circuit 实现，用户可以根据 hardware 约束选最合适的。

这就像 LLM 生成代码：传统 compiler 是 deterministic，一次出一个 binary；LLM 可以 sample 出多个候选程序，用户选最合适的。Diffusion model 把这个范式带到 quantum compilation 上，挺 elegant。

未来如果加上 hardware-aware conditioning（chip topology、gate fidelity、coherence time），就是一个 end-to-end hardware-aware quantum compiler。Paper §5 提了 future work 要做 measurement-based quantum computing (MBQC)，那个方向也很美——MBQC 用 graph state + measurement pattern，天然是 discrete structure，diffusion 可以直接生成 graph。

---

## 总结：值不值得读

值得，特别是如果你对 diffusion model 架构和 quantum application 都感兴趣。架构创新不算 radical（U-ViT 系列已经做过类似的事），但工程整合很 clean，quantum application 层的 encoding 设计有巧思，unitary compilation 实验很有 practical impact。

Masking 80% error rate 暴露了 continuous diffusion 在 hard constraint 上的根本问题，这是后续工作的金矿。Scaling 到 50+ qubits 也是 open challenge。Equivariant diffusion on circuit DAG 是我直觉最有希望的方向。

如果你只读一篇 quantum + diffusion 的 paper，这篇和 [GenQC (Furrutter 2024)](https://www.nature.com/articles/s42256-024-00920-8) 是必读组合——一个 baseline 一个改进，能快速建立 quantum circuit synthesis 的全景。

---

### 主要 Reference

- [UDiTQC Paper](https://arxiv.org/abs/2410.19324) (这篇)
- [DiT (Peebles & Xie 2023)](https://arxiv.org/abs/2212.09748)
- [GenQC (Furrutter 2024, Nature MI)](https://www.nature.com/articles/s42256-024-00920-8)
- [DDPM (Ho 2020)](https://arxiv.org/abs/2006.11239)
- [U-ViT / Simple Diffusion (Hoogeboom 2023)](https://arxiv.org/abs/2301.11093)
- [SiD2 (Hoogeboom 2024)](https://arxiv.org/abs/2410.19324)
- [Classifier-Free Guidance (Ho & Salimans 2022)](https://arxiv.org/abs/2207.12598)
- [DDIM (Song 2020)](https://arxiv.org/abs/2010.02502)
- [Equivariant Diffusion (Hoogeboom 2022)](https://arxiv.org/abs/2202.03026)
- [D3PM (Austin 2021)](https://arxiv.org/abs/2107.03006)
- [MaskGIT (Chang 2022)](https://arxiv.org/abs/2202.04200)
- [VQ-VAE (Oord 2017)](https://arxiv.org/abs/1711.00937)
- [Repaint (Lugmayr 2022)](https://arxiv.org/abs/2201.09865)
- [RL for Quantum Compiling (Moro 2021)](https://www.nature.com/articles/s42005-021-00633-5)
- [Schmidt Rank Vector (Huber 2013)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.030501)
- [Qiskit transpile](https://docs.quantum.ibm.com/api/qiskit/qiskit.compiler.transpile)
- [FlashAttention (Dao 2022)](https://arxiv.org/abs/2205.14135)
- [ViT (Dosovitskiy 2020)](https://arxiv.org/abs/2010.11929)
- [FiT (Lu 2024)](https://arxiv.org/abs/2402.12376)
- [VisionLLaMA (Chu 2024)](https://arxiv.org/abs/2403.00522)
- [Improved DDPM cosine schedule (Nichol 2021)](https://arxiv.org/abs/2102.09772)
- [AdamW (Loshchilov 2017)](https://arxiv.org/abs/1711.05101)

---

# UDiTQC: U-Net-Style Diffusion Transformer for Quantum Circuit Synthesis 深度解析

Andrej 你好，这篇 paper 我从头到尾啃了一遍，挺有意思——它本质上是把 **DiT** (Diffusion Transformer, [Peebles & Xie 2023](https://arxiv.org/abs/2212.09748)) 的 backbone 重新摁回 **U-Net** ([Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597)) 的多尺度 encoder-decoder 骨架里，然后用它来生成 quantum circuits。整个故事链是：GenQC ([Furrutter et al. 2024, Nature MI](https://www.nature.com/articles/s42256-024-00920-8)) 用 U-Net DDPM 生成量子电路 → U-Net 在长程序列上不够强 → DiT 解决长程但缺 multi-scale → **UDiT** 想要鱼与熊掌兼得 → UDiTQC 加上 quantum-specific encoding & conditioning。

---

## 1. Motivation: 为什么 GenQC 不够，为什么纯 DiT 也不够

GenQC 是第一个把 conditional DDPM ([Ho et al. 2020](https://arxiv.org/abs/2006.11239)) 应用到 quantum circuit synthesis 上的工作。它把电路表示成 2D tensor `[qubits × time_steps]`，用 U-Net 当 denoiser，以 SRV (Schmidt Rank Vector) 或 gate set 当 class condition。问题在 three-fold:

1. **Computational efficiency**: U-Net 的 attention 在 bottleneck 处计算量随 token 数二次膨胀，qubits 多了之后很贵;
2. **Data distribution sensitivity**: 量子电路作为稀疏结构数据（很多 padding），U-Net 卷积归纳偏置过强容易过拟合到训练集分布;
3. **Global context 缺失**: U-Net 的 inductive bias 偏局部，但 quantum entanglement 是高度非局部的属性，一个 CNOT 的 control 和 target 可以隔很远。

DiT 在 image generation ([ImageNet 256×256, SOTA FID](https://arxiv.org/abs/2212.09748)) 上证明 full attention + adaLN-Zero + scale/shift 比 U-Net 强，scalability 也更好。但 DiT 是 single-scale 的——所有 token 一个 resolution 走到底，没有 U-Net 的 multi-scale 上下文聚合。对量子电路这种"结构稀疏但全局耦合"的数据，两个优势都需要：multi-scale 处理稀疏布局 + global attention 处理 entanglement。

UDiT 的核心 insight 就是：**保留 U-Net 的金字塔拓扑，但把每个 stage 的 block 换成 DiT block，并用 residual connection 替代传统 skip connection**。这思路和同期 [U-ViT (Hoogeboom et al. 2023, Simple Diffusion)](https://arxiv.org/abs/2301.11093) 和 [SiD2](https://arxiv.org/abs/2410.19324) 是一脉相承的——把 long skip connection 当作 residual 看待，让信息流更像 LLM 中的纯 residual stack。

---

## 2. DDPM 基础回顾 (公式逐项解析)

Forward process 是一个 Markov chain，把 clean sample $x_0$ 在 $T$ 步内逐步加噪:

$$q(x_T | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})$$

- $x_0$: clean sample (这里是一个 tokenized quantum circuit tensor)
- $x_t$: noisy sample at timestep $t$
- $T$: 总 diffusion 步数 (paper 中 $T=1000$)

闭式 marginal:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

- $\bar{\alpha}_t = \prod_{i=0}^{t} \alpha_i = \prod_{i=0}^{t}(1 - \beta_i)$: cumulative signal preservation
- $\beta_i$: 第 $i$ 步加的噪声 variance (paper 用 squared cosine schedule, [Nichol & Dhariwal 2021](https://arxiv.org/abs/2102.09772))
- $I$: 单位矩阵，对应各维度独立 isotropic 噪声

Reverse process:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t), \sigma_\theta(x_t))$$

- $\mu_\theta, \sigma_\theta$: 由 UDiT 网络回归得到
- $\theta$: 所有网络参数 (UDiT blocks + embedding + decoder)

训练目标 (ε-prediction reparameterization):

$$\mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}[0,T],\, x_0 \sim q(x_0),\, \epsilon_t \sim \mathcal{N}(0, I)}\left[\|\epsilon_t - \epsilon_\theta(x_t, t, c)\|_2^2\right]$$

- $t$: 均匀采样的 timestep
- $c$: condition vector (SRV label embedding 或 unitary embedding)
- $\epsilon_\theta(x_t, t, c)$: UDiT 预测的噪声
- 这里不是 predict $x_0$ 而是 predict noise ε，是 [DDPM](https://arxiv.org/abs/2006.11239) 的标准 reparameterization，numerically 更稳定

---

## 3. UDiT 架构详解

### 3.1 整体拓扑 (Figure 1a)

UDiT 保留了 U-Net 的 5-stage encoder-decoder 结构:
- Encoder: 2 次 downsampling 把 sequence length 缩短 (类似 image U-Net 的 spatial pooling，但这里是对 sequence 维度做 conv stride=2)
- Bottleneck: middle stage
- Decoder: 2 次 upsampling (interpolation + conv) 还原到原 sequence length

每个 stage 内部是 **N 个 DiT block 串联**。这是关键——传统 U-Net 每个 stage 是 ResNet block + (optional) attention，这里全部替换为 DiT block。

### 3.2 DiT Block (Figure 1c) - adaLN-Zero

每个 DiT block 是:

```
input → adaLN(γ, β) → Multi-Head Self-Attention → ×α → residual → adaLN(γ, β) → FFN → ×α → residual → output
```

- **Conditioning fusion**: timestep embedding $t_{emb}$ + class embedding $c_{emb}$ 相加，过一个 MLP 回归出 6 组参数: $\gamma_1, \beta_1, \alpha_1$ (attention path) 和 $\gamma_2, \beta_2, \alpha_2$ (FFN path)
- **adaLN-Zero**: $\alpha$ 初始化为 zero，意味着训练初始时整个 block 是 identity——这是 DiT 训练稳定性的关键 trick。直觉上是让网络从"什么都不做"开始，逐步 learn 怎么用 attention 和 FFN。参考 [DiT paper §3.2](https://arxiv.org/abs/2212.09748)
- 数学形式:
  
  $$h_{out} = \alpha \cdot \text{Attn}(\gamma \cdot \text{LN}(h_{in}) + \beta) + h_{in}$$
  
  其中 $\gamma, \beta, \alpha = \text{MLP}(t_{emb} + c_{emb})$，初始 $\alpha = 0$

### 3.3 Residual Connection 替代 Skip Connection (Figure 1d, 公式 4)

这是 UDiT 最有 idea 的一点。传统 U-Net 用的是 long skip connection: encoder 的 stage $i$ 输出直接 concat/add 到 decoder 对应 stage。UDiT 拒绝这种设计，理由是 LLM/ViT 训练实践证明 pure residual stack 更 scalable，long skip 会让信息流变得不规整。

UDiT 把 downsampling→middle→upsampling 的整个 U 形抽象成一个**两段 residual**:

$$f(x) = f_u\left(U\left(f_m(D(h)) - D(h)\right) + h\right)$$

变量含义:
- $h = f_d(x)$: encoder 阶段输出
- $f_d, f_m, f_u$: downsampling stage / middle stage / upsampling stage
- $D$: downsampling operation (paper 用 conv kernel)
- $U$: upsampling operation (interpolation + conv)
- 关键: $f_m(D(h)) - D(h)$ 是"中间部分相对输入做了多少改变"，upsample 这个 diff 后加回 $h$，相当于在 residual 路径上做 U 形 transform

递归性: 中间 stage $f_m$ 本身可以再递归地包含一个 sub-UDiT，所以 multi-stage 就像 fractal 的 residual。这和 [U-ViT (Bao et al.)](https://arxiv.org/abs/2301.11093) 和 [Hoogeboom Simple Diffusion](https://arxiv.org/abs/2301.11093) 的设计高度类似——long skip 退化成"在 residual 函数里 downsample-then-upsample"。

直觉上: 训练初期整个 $f_m(D(h)) \approx D(h)$，所以 $f(x) \approx f_u(h) = f_u(f_d(x))$，相当于一个浅的纯残差网络。随着训练，$f_m$ 学到的差异逐渐贡献到输出。这种"先做浅后做深"的 implicit curriculum，比传统 U-Net 的 hard skip concat 更友好。

### 3.4 Asymmetric Design

Paper 把更多 DiT blocks 堆在 **decoder (upsampling)** 半边，encoder (downsampling) 半边少。这借鉴自 [Vaswani 2017](https://arxiv.org/abs/1706.03762) 和 [Hoogeboom 2023](https://arxiv.org/abs/2301.11093) 关于 decoder-heavy 架构在 generative 任务上更强的观察。

直觉解释: encoder 只需"压缩/抽象"，decoder 需要"展开/具体化"，后者难度更高，需要更多 capacity。这也呼应 GPT 系列 (decoder-only) 比 BERT (encoder-only) 在生成上更受青睐的现象。

### 3.5 Hidden Feature 维度策略

Intermediate stages 用 **更高 feature dim + 更少 attention heads**，对应 table 1 中 "+Hidden Feature Expansion" 和 "+Fixed Hidden Feature" 两步。直觉是: 在最压缩的中间层，token 数已经很少，每个 token 需要承载更多信息，所以加宽 channel; attention head 数减少但每个 head dim 更大，方便捕获更粗粒度的全局关系。

---

## 4. UDiTQC: Quantum Circuit 适配层

### 4.1 Circuit Encoding (Appendix A, Figure 8)

量子电路被编码为 2D tensor，shape `[n_qubits, n_timesteps]`，每个时间步只放一个 gate (强约束)。每个 gate 用 **orthogonal high-dim continuous embedding**:

- 单 qubit gate: embedding 是某个向量 $v_k$
- 多 qubit gate (如 CNOT): control 和 target 用 **同一个 embedding，符号相反**——这样模型从 sign 就能区分 control/target

Embedding dimension: $d = N + 2$，其中 $N$ 是 gate 种类数，$+2$ 留给 padding token 和 background token (没有 gate 的位置)。

Decoding 通过 **cosine similarity**:
$$\tilde{k} = \arg\max_k |S_C(v_k, v_{gen})|$$
$$k = \tilde{k} \cdot \text{sign}\, S_C(v_{\tilde{k}}, v_{gen})$$

- $v_k$: 第 $k$ 个 predefined gate embedding
- $v_{gen}$: UDiT 输出的连续向量
- $S_C$: cosine similarity
- 绝对值找最近 gate，sign 决定 control 还是 target
- 如果模型把两个 gate 放同一 timestep 或 control/target 配对错，就记为 "error circuit"

Paper 提到有趣现象: 即使 $N > d$ (gate 数比 embedding 维度高)，模型仍然能学到 match embedding，paper 称为 "overload" vector space——这其实是在做低维空间里的非线性 code，类似于 [VQ-VAE codebook collapse 的逆问题](https://arxiv.org/abs/1711.00937)。

### 4.2 Patchify 到 sequence

类似 ViT/DiT 的 patchify: 把 `[n_qubits, n_timesteps]` 的 2D tensor 切成 patch，flatten 成长度 $K = \max_{qubits} \times \max_{gates}$ 的 fixed-length sequence，每个 token 是一个 patch embedding。

加 **sinusoidal 2D positional embedding** (sine-cosine)，让 transformer 同时建模 temporal 和 spatial 位置。这是 [ViT](https://arxiv.org/abs/2010.11929) 风格而非 RoPE2D (FiT/VisionLLaMA 用的)，paper 在 §2.3 提了 RoPE2D 但最终选 sinusoidal——可能是因为 quantum circuit 长度变化范围小，RoPE 的 extrapolation 优势用不上。

### 4.3 Conditioning (Figure 2b, Appendix B)

**两种 condition**:

1. **Class label (SRV 或 gate set subset)**: 通过 `LabelEmbedder` 嵌入。SRV 是个 vector，每个 subsystem 是 1 (separable) 或 2 (entangled)，所以 fixed qubits 下 SRV 种类有限，可以直接编号成 class label。训练时用 **label dropout** (随机替换为 learnable null token ∅) 实现 classifier-free guidance ([Ho & Salimans 2022](https://arxiv.org/abs/2207.12598))。

2. **Unitary embedding (U-enc)**: 对 unitary compilation 任务，给定一个 $2^n \times 2^n$ 复数矩阵 $U$:
   - 拆 real/imag 为 2 channels
   - Conv → 2D positional encoding → Transformer encoder (self-attention，捕获非局部信息——unitary 任意元素都和其它所有元素耦合，这是必须的) → 2×2 downsample → conv 扩展到 hidden dim
   - 与 time+label embedding **concat** 后过 linear 投回统一维度
   - U-enc 内部 dropout 防止 overfitting (因为 unitary 空间大，但训练集每个 unitary 只有少量 circuit 实现，容易记死)

### 4.4 Inference 用的 Rescaled CFG

公式 8:
$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

- $s$: guidance scale, paper 中 $s = 7.5$
- $\emptyset$: null condition (训练时 dropout 学到的)
- $s=1$ 退化成标准 conditional sampling
- $s>1$ 强化 condition 信号，但太大易 mode collapse

"Rescaled" 指的是 [Ho & Salimans 2022](https://arxiv.org/abs/2207.12598) 里说的对 CFG 输出做 variance rescale，避免 high guidance 时 under-exposed samples。

---

## 5. Experiments 一览

### 5.1 Entanglement Generation (Figure 3, 4)

数据集用 **H, CX** 两个 gate (Clifford subset 的一部分)，3-8 qubits，gate 数从 16 到 52 不等。SRV 作为条件。

**结果**:
- UDiTQC 在 3-8 qubits 上一致超过 GenQC
- 5-qubit confusion matrix (Figure 4) 显示对角线强，主误差来自"少 entangled → 多 entangled"方向 (符合直觉，需要更多 gate 才能产生复杂纠缠)
- 模型还能 generate 训练集外的 novel circuit (泛化性)

### 5.2 Masking (Figure 5a)

量子芯片物理限制: 远距离 qubit 之间需要 SWAP 网络，不能直接两-qubit gate。Masking 把 input tensor 的某些位置置零 (white area)，模型在 inference 时不会在那里放 gate。

**结果**: 满足物理约束，accuracy 略降。但 paper §C.2 提到一个**致命问题**: masking 任务的 error rate 从 <1% 飙到 ~80%——意思是 80% 的 generated circuit 是 invalid (gate 冲突或 control-target 配对错)。这暗示 inference 时 masking 这种 hard constraint 让 DDPM 的 marginal 分布偏离了 valid circuit manifold。

直觉上，DDPM 没有 explicit constraint handling 机制，靠 learned distribution 来 honor constraint，遇到 OOD constraint 就退化。这是 [Repaint (Lugmayr et al.)](https://arxiv.org/abs/2201.09865) 试图解决但没根本解决的问题。或许 discrete diffusion ([D3PM, Austin et al.](https://arxiv.org/abs/2107.03006)) 或者 [AR with structured head](https://arxiv.org/abs/2305.13274) 更适合 quantum 这种严格结构约束场景。

### 5.3 Editing (Figure 5b, 6)

固定 circuit 前几 gate (作为"初始量子态")，让模型在后续位置 generate gate 来达到目标 SRV。

**结果**: 
- 从低 entanglement → 高 entanglement (Figure 6 对角线上方) 容易，>85% 成功
- 从高 → 低 entanglement 难，因为要先"untangle"再 edit，gate sequence 复杂
- 整体 85.2% 成功率，比 masking 友好得多

### 5.4 Unitary Compilation (Figure 7, §4.3)

Gate pool: $\{H, CX, Z, X, CCX, SWAP\}$，3 qubits，gate 数 2-12。Label 是 gate subset (248 种 subset，但训练用 63 种)。

**Metric**: 编译正确率 + Frobenius norm $\frac{1}{2}\|U_t - U_g\|_F^2$

- $U_t$: 目标 unitary
- $U_g$: generated circuit 对应的 unitary
- Frobenius norm = 0 意味着 circuit 完全实现 target unitary

**结果**:
- UDiTQC: **94.9%** compilation accuracy (5000+ unseen unitaries, 每个 1024 sample circuits)
- GenQC: 92.6%
- 大部分 unitary 编译出 Frobenius norm = 0，少数 nonzero 的 norm 也远低于 random baseline
- 每个 unitary 通常能生成多个 distinct circuit——给用户选择空间 (gate depth vs fidelity tradeoff)

这是 paper 最有 practical impact 的结果。Quantum compilation 传统上用 [solovay-kitaev](https://arxiv.org/abs/quant-ph/0505030) 或 [reinforcement learning (Moro et al.)](https://www.nature.com/articles/s42005-021-00633-5)，UDiTQC 提供了第三条路: **diffusion model 当 compiler**，且 inference 时 batch 出多个候选。

### 5.5 Ablation Study (Table 1)

3-qubit entanglement generation 上 8 个 variant:

| Model | Speed (steps/s) | Avg Acc (%) | Entangled Acc (%) |
|---|---|---|---|
| DiT (base) | 10.56 | 64.08 | 44.1 |
| DiTsq (sequence embedding) | 15.1 | 77.52 | 64.75 |
| U-Net-Style DiTsq | 10.74 | 79.18 | 69.2 |
| U-Net-Style DiT | 9.24 | 84.14 | 56.06 |
| + Asymmetric | 9.87 | 85.1 | 60.54 |
| + Residual Connections | 8.56 | 86.2 | 61.97 |
| + Hidden Feature Expansion | 9.9 | 82.4 | 59.14 |
| + Fixed Hidden Feature (final) | **15.5** | **89.12** | 65.72 |

**关键观察**:
1. 纯 DiT 在 quantum task 上弱 (64% avg)——说明 multi-scale 真的有用
2. DiTsq (sequence gate embedding) 比 vanilla DiT 强很多 (77.5% vs 64%)，但加 U-Net-style 之后提升变小——意味着 U-Net 风格和 sequence embedding 有冗余
3. **Asymmetric + Residual + Fixed Hidden Dim** 组合时，speed 反而从 8.56 跳到 15.5——feature dim 固定避免了 attention 复杂度的额外开销，attention heads 减少进一步加速
4. "Entangled Accuracy" (最难 case，full entanglement) 不是单调提升的: DiTsq 系列高，但 final UDiT (65.72) 不如 U-Net-Style DiTsq (69.2)。说明 final 设计对 avg accuracy 友好但对极端 case 可能 over-regularized

---

## 6. 数据集 (Table 2)

- **Entanglement generation**: H + CX gate pool, 3-8 qubits, max gate 从 16 增到 52
- **Unitary compilation**: 6 种 gate (含 CCX, SWAP), 3 qubits, 925K circuits
- 用 [Qiskit transpile](https://docs.quantum.ibm.com/api/qiskit/qiskit.compiler.transpile) 优化 (合并 redundant gate, 删除 redundant qubit), 然后去重
- SRV 分布严重不均衡 (全 1 或全 2 占多)，做 class balancing

| Qubits | Gate Pool | Min/Max Gates | # Labels | Total Circuits |
|---|---|---|---|---|
| 3 | H, CX | 2/16 | 5 | 200K |
| 4 | H, CX | 3/20 | 12 | 300K |
| 5 | H, CX | 4/28 | 27 | 459K |
| 6 | H, CX | 5/40 | 58 | 470K |
| 7,8 | H, CX | 6/52 | 121 | 484K |
| 3 (unitary) | H,CX,Z,X,CCX,SWAP | 2/12 | 63 | 925K |

# labels 公式: $2^{2^q - 1}$? 实际看数据 $q=3 \to 5$ labels (公式 $2^{q-1}$? $2^2=4$ 不对，paper 说 $num = 2^{q-1}$? 但 $q=3$ 给 5)，似乎是 SRV 的所有 valid 排列数，公式原文 $num = 2^{2^q-1}$ 实际是 SRV 模空间。

---

## 7. Intuition & 一些 Karpathy-style 思考

### 7.1 为什么 pure DiT 在 quantum 上不如 U-Net-style DiT

纯 DiT 把所有 token 投到同一 resolution 的 attention pool，对所有 token 一视同仁。但 quantum circuit tensor 高度稀疏——大部分位置是 padding/background。Full attention 在稀疏 token 上浪费 capacity，把计算花在 padding 上。U-Net 的 downsampling 把 padding 区域压缩，effective attention 聚焦到 active gate——这和 [Hierarchical Attention](https://arxiv.org/abs/2106.10554) 的思路类似。

### 7.2 Residual 替代 Skip 的本质

公式 4 的 $f_m(D(h)) - D(h)$ 是个 **residual function**：把"中间层学到的 transformation"显式表达成"对输入的修改"。这是 LLM 残差风格的延伸——所有 transformation 都是 identity + delta。训练初期 delta≈0，模型是浅的，gradient flow 好; 训练后期 delta 增长，模型变深，capacity 释放。这种 implicit depth curriculum 比 U-Net 的 hard skip concat 更平滑。

### 7.3 Quantum + Diffusion 的根本张力

Diffusion 假设数据分布是连续 Gaussian noise 上的 manifold。但 quantum circuit 是 **离散 + 严格结构约束** 的对象——gate 种类有限，gate 之间有 control/target 配对关系，每个 timestep 只能放一个 gate。Paper 用 orthogonal continuous embedding 做了 hack，但本质上还是把 discrete structure "假装成" continuous。

更彻底的方案可能是:
- **Discrete diffusion / Masked generative** ([D3PM](https://arxiv.org/abs/2107.03006), [MaskGIT](https://arxiv.org/abs/2202.04200), [VQ-Diffusion](https://arxiv.org/abs/2111.14822)) 直接在 token 空间做
- **ARS diff. on graphs** (circuit 本质是 DAG)
- **Equivariant diffusion** ([Hoogeboom 2022](https://arxiv.org/abs/2202.03026)) 把 circuit 的对称性编进去

### 7.4 "Overload" vector space 的现象

Paper §A 提到: 即使 gate 数 $N > d$ (embedding dim)，模型仍能准确 match embedding。这意味着模型实际上没真正用 orthogonal embedding 的几何，而是 **学到了一个非线性 decoder** 把 $v_{gen}$ 投影回最近 $v_k$。这和 [VQ-VAE codebook](https://arxiv.org/abs/1711.00937) 的现象有点像，但反过来——codebook collapse 是多个 code 退化成一个，overload 是一个 code 表示多个 mode。

### 7.5 Asymmetric + Residual 速度提升的反常

Ablation 表里 "+Residual Connections" 阶段 speed 降到 8.56 (最低)，但 "+Fixed Hidden Feature" 一加，speed 反弹到 15.5 (最高)。说明 attention 计算成本主要来自 **hidden dim × heads** 的复杂度，而不是 sequence length。固定 dim 并减 heads 把 attention cost 压下去，UDiT 的 U 形结构带来的额外 conv 计算相对可忽略。

这给我们的 intuition: 在 diffusion transformer 里，**attention 的 channel 复杂度比 spatial 复杂度更主导**，所以 [FlashAttention](https://arxiv.org/abs/2205.14135) 之外的另一条加速路径是 attention 内部的 channel/head 维度 budgeting。

---

## 8. 与同期工作对比

- **[U-ViT (Bao et al. 2023)](https://arxiv.org/abs/2301.11093)**: 几乎同期 work，把 U-Net long skip 加到 ViT 上，做 image generation。UDiT 更进一步用 **residual connection 替代 long skip**，并且做 asymmetric + dim schedule
- **[Simple Diffusion (Hoogeboom 2023)](https://arxiv.org/abs/2301.11093)**: 高分辨率 image diffusion，强调 multi-scale 处理 + U-ViT-style skip。UDiT 直接引用其设计哲学
- **[FiT (Lu 2024)](https://arxiv.org/abs/2402.12376)**: Flexible ViT for diffusion, 用 RoPE2D 支持任意分辨率。UDiT 选 sinusoidal 而非 RoPE2D，可能因为 quantum circuit 分辨率范围窄
- **[SiD2 (Hoogeboom 2024)](https://arxiv.org/abs/2410.19324)**: 1.5 FID on ImageNet 512 with pixel-space diffusion, 用类似 U-ViT 结构。说明这条 architecture line 在 image 上也是 SOTA
- **[GenQC (Furrutter 2024)](https://www.nature.com/articles/s42256-024-00920-8)**: 直接 baseline，UDiTQC 一致超过它
- **[Rietsch 2024 RL for Clifford+T](https://arxiv.org/abs/2404.14865)**: 用 RL 做 unitary synthesis，UDiTQC 的 unitary compilation 是 diffusion 替代 RL 路线
- **[Moro 2021](https://www.nature.com/articles/s42005-021-00633-5)**: 早期 RL quantum compiling

---

## 9. Limitations 我读出来的

1. **Masking 80% error rate**: paper §C.2 自己承认。这说明 diffusion 在 hard constraint 下不稳定，需要后续工作解决 (可能要用 [constrained diffusion](https://arxiv.org/abs/2402.13172) 或 [structure-aware sampling](https://arxiv.org/abs/2305.13274))
2. **Max 8 qubits**: 离实用规模 (50-100 qubits) 还远。Sequence length $K = \max_{qubits} \times \max_{gates}$ 会爆炸
3. **Gate set 小**: entanglement generation 只用 H, CX (Clifford 子集)，没测 T gate 或 arbitrary RZ。Unitary compilation 用 6 种 gate 但仍然小 gate set
4. **No comparison to non-ML compiler**: 没和 [Qiskit transpiler](https://docs.quantum.ibm.com/api/qiskit/qiskit.compiler.transpile) 或 [t|ket⟩](https://github.com/CQCL/tket) 比，只比了 GenQC
5. **"Overload" 现象没理论解释**: $N > d$ 仍 work 是 empirical surprise，paper 没深挖
6. **Frobenius norm 不直接等价 unitary fidelity**: $\|U_t - U_g\|_F$ 不归一化，应该用 [process fidelity](https://en.wikipedia.org/wiki/Quantum_fidelity) $\frac{1}{d}|\text{tr}(U_t^\dagger U_g)|$ 更标准

---

## 10. 未来方向 (paper 提到 + 我的联想)

Paper §5 提到要扩展到 **measurement-based quantum computing (MBQC)**——这非常合理，因为 MBQC 用 graph state + measurement pattern，天然是 discrete structure，diffusion model 可以直接生成 graph。参考 [MBQC 综述](https://arxiv.org/abs/quant-ph/0602190)。

我自己的联想:
- **Equivariant diffusion on circuit DAG**: gate 之间 permutation 和 qubit relabeling 是对称性，可以编进 model ([EDM, Hoogeboom 2022](https://arxiv.org/abs/2202.03026))
- **Discrete diffusion for quantum gate**: 用 [D3PM](https://arxiv.org/abs/2107.03006) 直接在 gate token 上做 diffusion，跳过 orthogonal embedding 的 hack
- **Transformer predictor + diffusion sampler hybrid**: 类似 [DiffuLLaMA](https://arxiv.org/abs/2402.05130) 在 LLM 上的尝试，diffusion 做整体 planning，AR head 做 gate-by-gate refinement
- **Hardware-aware compilation**: 把 chip topology (qubit connectivity graph) 作为额外 condition embedding，直接生成 hardware-native circuit (现在 paper 的 masking 是简单版本)
- **Quantum error correction code synthesis**: 用 diffusion 生成 [[n,k,d]] stabilizer code 是个 open problem，UDiTQC 框架可以直接套上

---

## 11. 总结 (一句话)

UDiTQC 把 DiT 的 adaLN-Zero + full attention 塞进 U-Net 的多尺度 encoder-decoder 骨架，用 residual connection 替代 long skip，做 asymmetric (decoder-heavy) + scheduled hidden dim，应用层用 orthogonal gate embedding + sinusoidal 2D pos + U-enc (unitary Transformer encoder) 把 quantum circuit 翻译成 diffusion model 能消化的高维序列，在 entanglement generation 和 unitary compilation 上超过 GenQC baseline 2-3 个百分点，且支持 masking/editing 这种 inference-time constraint。

架构创新不算 radical，但工程上把 U-ViT 系列的几个 trick (residual skip, asymmetric, dim schedule) 整合得很 clean，quantum application 也很 natural。下一步如果做大规模 quantum circuit (50+ qubits) 或真正 hardware-aware compilation，会更有 impact。

---

### Reference Links

- [DiT - Scalable Diffusion Models with Transformers (Peebles & Xie 2023)](https://arxiv.org/abs/2212.09748)
- [GenQC - Quantum circuit synthesis with diffusion models (Furrutter et al. 2024, Nature MI)](https://www.nature.com/articles/s42256-024-00920-8)
- [DDPM - Denoising Diffusion Probabilistic Models (Ho et al. 2020)](https://arxiv.org/abs/2006.11239)
- [DDIM (Song et al. 2020)](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Guidance (Ho & Salimans 2022)](https://arxiv.org/abs/2207.12598)
- [U-ViT / Simple Diffusion (Hoogeboom et al. 2023)](https://arxiv.org/abs/2301.11093)
- [SiD2 - Simpler Diffusion (Hoogeboom et al. 2024)](https://arxiv.org/abs/2410.19324)
- [ViT - An Image is Worth 16x16 Words (Dosovitskiy 2020)](https://arxiv.org/abs/2010.11929)
- [Attention is All You Need (Vaswani 2017)](https://arxiv.org/abs/1706.03762)
- [Latent Diffusion (Rombach 2022)](https://arxiv.org/abs/2112.10752)
- [EDM - Elucidating Design Space of Diffusion (Karras 2022)](https://arxiv.org/abs/2206.00364)
- [FiT - Flexible Vision Transformer for Diffusion (Lu 2024)](https://arxiv.org/abs/2402.12376)
- [VisionLLaMA (Chu 2024)](https://arxiv.org/abs/2403.00522)
- [Repaint - Inpainting with DDPM (Lugmayr 2022)](https://arxiv.org/abs/2201.09865)
- [D3PM - Discrete Diffusion (Austin 2021)](https://arxiv.org/abs/2107.03006)
- [Equivariant Diffusion for Molecules (Hoogeboom 2022)](https://arxiv.org/abs/2202.03026)
- [VQ-VAE (Oord 2017)](https://arxiv.org/abs/1711.00937)
- [FlashAttention (Dao 2022)](https://arxiv.org/abs/2205.14135)
- [RL for Quantum Compiling (Moro 2021)](https://www.nature.com/articles/s42005-021-00633-5)
- [RL for Clifford+T (Rietsch 2024)](https://arxiv.org/abs/2404.14865)
- [Qiskit Documentation](https://docs.quantum.ibm.com/api/qiskit/qiskit.compiler.transpile)
- [Schmidt Rank Vector (Huber & de Vicente 2013)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.030501)
- [Improved DDPM - Cosine schedule (Nichol & Dhariwal 2021)](https://arxiv.org/abs/2102.09772)
- [MaskGIT (Chang 2022)](https://arxiv.org/abs/2202.04200)
- [VQ-Diffusion (Gu 2022)](https://arxiv.org/abs/2111.14822)
