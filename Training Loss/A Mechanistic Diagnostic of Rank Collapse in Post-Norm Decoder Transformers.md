---
source_pdf: A Mechanistic Diagnostic of Rank Collapse in Post-Norm Decoder Transformers.pdf
paper_sha256: 0e55642e6bb0f0519f2afa99a9b919c900c597bb9ffda9ee9b49ea88cef4a020
processed_at: '2026-08-17T23:23:55-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
Transformer 训练塌掉的根因是: 前面每一层 attention 都在偷偷让 token 变得越来越像,一旦像到一定程度,反向梯度就会被 RMSNorm 不断压缩、根本传不到浅层去修复,最后网络只能输出"最常见的词",然后就卡死在那儿爬不出来了。

你有一句话 "the cat sat on",经过 Transformer 后,4 个 token 在高维空间里本应该是 4 个不同的点。但训练着训练着,这 4 个点开始往一个位置挤,最后挤成一个点 —— 所有 token 的 hidden state 变得一模一样。

这就叫 **rank collapse**。

为什么坏?因为下游 LM head 拿到的输入是同一个向量重复 $n$ 次,它不管你是第 1 个 token 还是第 50 个 token,输出都是同一个分布。这就跟"模型对位置完全不敏感"一样,等价于退化成一个 unigram 模型 —— 它只会预测"哪个词在训练集出现得多"。

paper 用一个 scalar $t_{\text{sim}}$ 来量化这件事:$t_{\text{sim}} = 1$ 就是完全塌, $t_{\text{sim}} = 1/n$ 就是 token 之间完全独立。整个故事就是追这个 scalar 怎么从 $1/n$ 爬到 $1$。

---

## 2. 前向:为什么 token 会自动变得越来越像

### 直觉

causal attention 的本质是:每个 token 去看它前面的 token,然后做加权和。在随机初始化时,attention logits 是零均值高斯,softmax 出来近似均匀分布。换句话说,**每个 token 几乎是在把它前面所有 token 取平均,加到自己身上**。

paper 把这个近似抽象成一个矩阵 $\mathbf{C}_n$,叫 **prefix-averaging matrix**:
- 第 1 个 token:不动(前面只有自己)
- 第 2 个 token:取前 2 个的平均
- 第 $i$ 个 token:取前 $i$ 个的平均

注意一件事:**取平均会消灭差异**。如果你有 4 个数 [3, 5, 7, 9],你把它们取平均得到 6,然后再用 [3,5,7,9] 加上 [6,6,6,6]/某个 scale,新的数就变成 [接近 6, 接近 6, 接近 6, 接近 6]。差异被压扁了。

这就是 attention 在初始化时干的事 —— 它本质上是一个 **低通滤波器**,消灭了 token 之间的高频差异,只留下低频的"共享成分"。所以每过一层 attention,$t_{\text{sim}}$ 就往上爬一点。

### 公式什么样

paper 给出了 closed form (Theorem 3.2):

$$
\Delta_{\text{attn}} = \frac{s \cdot f_1(t_1)}{1 + s \cdot f_2(t_1)} > 0
$$

关键变量是 $s = n d^2 \sigma_W^2 / \|\mathbf{X}\|_F^2$,叫 "amount of attention"。**这个 $s$ 越大,similarity 爬得越快**。

- 在 Post-Norm 里,每一层 RMSNorm 都把 $\|\mathbf{X}\|_F^2$ 归一化回 $nd$,所以 $s$ 在每一层都是常数 $d \sigma_W^2$ —— 每一层都同等强度地推 similarity 上去。
- 在 Pre-Norm 里,residual stream 不重新归一化,$\|\mathbf{X}\|_F^2$ 随层数线性增长,所以 $s \propto 1/l$ —— 越深的层,推 similarity 的力气越小。

**这就是 Post-Norm 比 Pre-Norm 危险的第一个原因:它的 attention 放大效应不随层数衰减,48 层堆起来,similarity 一路爬到天花板。**

### SwiGLU FFN 呢?

paper 也算了 FFN 的贡献 (Theorem 3.4),发现它是 **轻微负的** —— SwiGLU 想把 similarity 拉回来一点,但量级只有 0.01 左右,跟 attention 的贡献比就是杯水车薪。

paper 还做了一个漂亮的 intervention 实验 (Corollary 3.3):把 attention 里的 prefix-averaging 部分减掉,具体是 $\mathbf{P} \to \mathbf{P} - \alpha \mathbf{C}_n$。随着 $\alpha$ 从 0 加到 1,similarity 的增长被显著压低。这就直接证明:**罪魁祸首就是 prefix averaging 这一坨**。

---

## 3. 反向:为什么塌了之后修不回来

到这里你可能会想:就算初始化时 similarity 高了一点,训练时只要梯度能传回来,模型自己会学着把 token 推回去啊。

**问题就在这:梯度传不回来。**

### RMSNorm backward 的两个坑

RMSNorm 的反向 Jacobian 长这样:

$$
\mathbf{J}_{\text{RMS}}(\mathbf{y}) = \frac{\sqrt{d}}{\|\mathbf{y}\|_2} \left( \mathbf{I} - \frac{\mathbf{y} \mathbf{y}^\top}{\|\mathbf{y}\|_2^2} \right)
$$

两个东西在搞你:

**第一个:前面的系数 $\sqrt{d}/\|\mathbf{y}\|_2$**

如果 residual stream 的 norm $\|\mathbf{y}\|$ 很大,这个系数就很小,梯度被压扁。问题是 —— paper 的 Proposition D.2 证明 —— 在 collapse 状态下,residual norm 会**自动增长**。

为什么?因为 RMSNorm 的 Jacobian 里有一个 $\mathbf{I} - \text{Proj}(\mathbf{y})$,这个东西把梯度里跟 $\mathbf{y}$ 平行的部分删掉了,只剩垂直部分。所以梯度永远垂直于当前 residual stream,梯度下降一步相当于在垂直方向加了一个分量,根据勾股定理:

$$
\|\mathbf{Y}_{\text{new}}\|^2 = \|\mathbf{Y}_{\text{old}}\|^2 + \eta^2 \|\mathbf{U}\|^2
$$

norm 只会涨不会降。**这是 RMSNorm 自己挖的坑,然后自己掉进去**:norm 涨 → 系数 $\sqrt{d}/\|\mathbf{y}\|$ 变小 → 梯度变小 → 修不动 → norm 继续涨 → 系数更小 → 梯度更小...

**第二个:位置问题**

在 Post-Norm 里,RMSNorm 放在 residual addition **之后**。所以反向传播时,梯度要从 $\mathbf{X}_{k+1}$ 流回 $\mathbf{X}_k$,必须穿过 RMSNorm 的 Jacobian —— 整个梯度(包括 skip-connection 那一支)都被这个 Jacobian 乘一遍。

$$
\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{X}_k}\right\| \leq \alpha_k \cdot \frac{\sqrt{d}}{\|\mathbf{y}_k\|} \cdot \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{X}_{k+1}}\right\| := c(\mathbf{y}_k, \alpha_k) \cdot \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{X}_{k+1}}\right\|
$$

当 $c < 1$,这一层就是 contractive 的。叠 48 层,梯度按 $c^{48}$ 衰减 —— 如果 $c = 0.9$,衰减到 0.6%,如果 $c = 0.8$,衰减到万分之一。**浅层基本收不到梯度,什么也修不了。**

而在 Pre-Norm 里,RMSNorm 放在 residual addition **之前**。反向时,skip-connection 那一支的梯度**不穿过** RMSNorm 的 Jacobian,只有 sublayer 那一支被压缩。所以即使 residual norm 涨了,主路径梯度还是好好的。**这就是 Pre-Norm 稳定的核心。**

### 实验长啥样

paper 在 48 层 Post-Norm 上做实验,LR=8e-4 时大概 step 2644 开始塌,这叫 transition window。观察到的现象:

- 各层的 $\alpha_k$(sublayer 自己的 gradient contribution)确实在涨,但最多到 2.5
- 各层的 $\sqrt{d}/\|\mathbf{y}_k\|$(RMSNorm 压缩因子)急剧下降,远低于 1
- 两者的乘积 $c$ 大部分层掉到 1 以下
- 最后 early layer 的 gradient norm 比 last layer 小好几个数量级

这就是 Figure 4-5 展示的画面:塌掉那一刻,浅层梯度直接被腰斩。

---

## 4. 塌了之后会停在哪儿

最后一个问题:网络塌了之后,loss 会停在哪里?为什么它就停在那儿不动了?

### 频率分布

如果所有 token 的 hidden state 都一样,那 LM head 拿到的输入就是同一个向量重复 $n$ 次,输出就是同一个概率分布 $\hat{\mathbf{p}}$ 重复 $n$ 次。

cross-entropy loss 是:

$$
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \log \hat{p}_{y_i}
$$

要最小化它,就要选 $\hat{p}_i \propto c_i$,其中 $c_i$ 是 label $i$ 在 batch 里出现的次数。**这就是"频率分布"** —— 每个词的概率等于它在数据里的频率。

最优 loss 等于这个分布的 entropy,paper 叫它 **frequency loss** $\mathcal{L}_{\text{freq}}$。在 C4 上,这个值大概就是 unigram entropy,是个相当高的 loss floor。

### 为什么停在频率分布不动

Theorem 3.8 part (ii) 证明:一旦网络输出恰好等于频率分布,**所有 collapse 层的 parameter gradient 都是 0**。

直觉是这样的:frequency distribution 是 $\hat{\mathbf{p}}$,one-hot label matrix 是 $\mathbf{E}(\mathbf{y})$。如果 $\hat{\mathbf{p}}$ 等于 $\mathbf{y}$ 的经验分布,那 $\mathbb{1}_n^\top (\hat{\mathbf{P}} - \mathbf{E}(\mathbf{y})) = 0$,也就是 **gradient 的列和为零**。

paper 用四个 lemma 串起来证明这个"列和为零"性质会从 LM head 一路传到上一层:RMSNorm 保持它,FFN 保持它,attention 保持它。所以每一层 collapse 的 parameter matrix,gradient 全部消失。

再加上前面说的浅层梯度本来就被压缩到几乎为零,**整个网络就停在一个 near-stationary point**,既不能往好的方向走(梯度为零),也回不到好的状态(浅层梯度被腰斩)。这就是为什么 collapse 是 absorbing state —— 进去了就出不来。

### 实验验证

Figure 6 直接画了 collapse 后的 training loss 和 frequency loss 的对比:两者贴得很紧。无论用 500M tokens 算 frequency,还是用单 batch 算,training loss collapse 后都接近对应的 frequency loss。

---

## 5. 整个故事串起来:一个比喻

想象你在一个 48 层的办公楼里办公,每层都有一个"协调员"(attention sublayer)和一个"执行员"(FFN sublayer)。

**前向阶段(初始化):** 每层的协调员都会做一件事:把当前层所有人的意见跟前几层的意见取平均,然后传给下一层。越往下走,意见越趋同 —— 因为平均了太多次。到了 48 层,所有人想法已经一模一样了。

**反向阶段(训练):** 大楼顶层发现问题,想发通知让底层改变做法。但通知每下一层都要经过"门卫"(RMSNorm),门卫会:
1. 把通知里"跟当前主流意见一致"的部分删掉(只留垂直部分)
2. 如果当前主流意见很强势(norm 大),把整个通知的音量调小

更糟的是,门卫每次删通知都会让"主流意见"变得更强(勾股定理),下一层的门卫删得更狠。48 层下来,底层收到的通知已经听不清了。

**塌掉之后:** 大楼里所有人意见都一样了,只会喊"最常听到的词" —— 这就是频率分布。而且因为通知的列和恰好为零,门卫觉得"大家没什么意见要改的",彻底躺平。大楼就这么卡住了。

**为什么 Pre-Norm 大楼没事:** 在 Pre-Norm 大楼里,门卫站在协调员和执行员的办公室门口,不在主走廊上。主走廊的通知从顶楼直接走到一楼,一路畅通无阻。只有协调员/执行员内部的小通知会被门卫管,但主路径没事。所以即使 48 层,通知也能传到一楼。

---

## 6. 这件事为什么重要

这篇 paper 的价值不在于提出新的 stabilization 方法 —— 它没提。它的价值是**把一个工程师早就知道的现象(深层 Post-Norm 难训)给出了完整的 mechanical explanation**:

1. 前向:为什么 Post-Norm 在初始化时就接近 collapse(prefix averaging + $s$ 不衰减),Pre-Norm 没事($s$ 衰减)
2. 反向:为什么一旦接近 collapse 就爬不出来(RMSNorm Jacobian contractive + residual norm 自增长),Pre-Norm 没事(skip-connection 绕过 Jacobian)
3. 终态:为什么 collapse 后 loss 停在 frequency loss 附近(频率分布是 Bayes-optimal under no-information + gradient vanishing)

读完之后,你再看 Llama、GPT-NeoX、T5 全部用 Pre-Norm + RMSNorm,就会觉得"显然如此" —— 不是 follow fashion,是这两个机制上的差异太根本了。

Karpathy 你之前讲 "build intuition" 的精神,这篇 paper 正是这种风格的体现:不堆 trick,不卖弄 bound,而是把一个复杂现象用一个 scalar $t_{\text{sim}}$ 串起来,前向有 closed form,反向有 contractive factor,终态有 frequency loss,三个 stage 实验上全部对得上。这种 clarity 在 transformer theory 圈里相当罕见。

主要参考链接:

- Paper 本身(arxiv 版本尚未公开,作者主页待追踪)
- Xiong et al. 2020 ICML "On Layer Normalization in the Transformer Architecture" https://arxiv.org/abs/2002.04745
- Dong et al. 2021 ICML "Attention is Not All You Need" https://arxiv.org/abs/2103.07448
- Noci et al. 2022 NeurIPS "Signal Propagation in Transformers" https://arxiv.org/abs/2206.02202
- Yu & Zhang 2026 JMLR "Why Classic Transformers Are Shallow" https://jmlr.org/papers/volume27/64
- Zhang & Sennrich 2019 NeurIPS "Root Mean Square Layer Normalization" https://arxiv.org/abs/1910.07467
- Shazeer 2020 "GLU Variants Improve Transformer" https://arxiv.org/abs/2002.05202
- He et al. 2016 CVPR "Deep Residual Learning" https://arxiv.org/abs/1512.03385

---

# 《Why Post-Norm Transformers Collapse: Attention Amplification and Gradient Repair Failure》深度技术讲解

## 0. 文章定位与核心论点

这篇paper由香港中文大学(深圳)的Wang、Han、Luo、Zhang组完成,直接目标是给**深层Post-Norm decoder-only Transformer的rank collapse**做一个完整的"两阶段机械论解释":为什么 Post-Norm 在深层会塌,为什么 warmup/LR 稍微一大就翻车,为什么一旦进入collapse就爬不出来。作者把整个动力学压缩到一个 scalar state variable —— **token similarity** $\mathrm{t}_{\mathrm{sim}}(\mathbf{X})$ —— 上,然后用 closed-form 公式追踪它如何随depth放大、随training step收缩 backward signal。

核心论点可以用一句话总结:**Post-Norm的collapse源于前向"causal attention做prefix averaging"持续放大相似度 + 反向"RMSNorm Jacobian随residual norm增长而contractive"持续消灭修复梯度的耦合机制**。这两个机制在Pre-Norm中要么被减弱,要么被结构性地避免,这就是Pre-Norm稳定的原因。

- Paper arxiv 版本未公开,但作者和 referenced Yu & Zhang (2026) JMLR 的 lineage 一致: https://jmlr.org/papers/volume27/64
- 关于 Post-Norm vs Pre-Norm 的经典分析见 Xiong et al. 2020 ICML: https://arxiv.org/abs/2002.04745
- 关于rank collapse的pure attention 谱分析见 Dong et al. 2021 ICML: https://arxiv.org/abs/2103.07448

---

## 1. 为什么这件事重要:背景与历史脉络

### 1.1 Post-Norm vs Pre-Norm 的结构差异

在 Vaswani et al. 2017 原始 Transformer 中,LayerNorm 放在 residual addition **之后**(Post-Norm):

$$
\mathbf{Y} = \mathrm{LN}(\mathbf{X} + \mathrm{Sublayer}(\mathbf{X}))
$$

而 Pre-Norm (Xiong et al. 2020, Liu et al. 2020 EMNLP: https://aclanthology.org/2020.emnlp-main.46/) 把 LN 移到 **之前**:

$$
\mathbf{Y} = \mathbf{X} + \mathrm{Sublayer}(\mathrm{LN}(\mathbf{X}))
$$

看起来只是位置变化,但对训练动力学影响巨大。Llama/T5/GPT-NeoX 几乎全部采用 Pre-Norm + RMSNorm (Zhang & Sennrich 2019: https://arxiv.org/abs/1910.07467),原因正是 Post-Norm 深层训练极度敏感。

### 1.2 已知现象,但解释缺失

在本文之前,学界已经识别了与 Post-Norm 失败相关的若干现象:

- **梯度消失**:Xiong et al. 2020、Emadi 2026 (https://arxiv.org/abs/2602.18849)、Chen & Wei 2026 都指出 Post-LN 在初始化时各层 gradient magnitude 不平衡。
- **Rank collapse**:Dong et al. 2021 证明纯attention栈中token表示double-exponentially收敛到rank-1;Noci et al. 2022 (https://arxiv.org/abs/2206.02202) 把collapse和signal propagation联系起来;Saada et al. 2024 (https://arxiv.org/abs/2410.07799) 做了谱分析;Yu & Zhang 2026 给出了 Post-Norm encoder 的quantitative分析。
- **Stabilization methods**:Fixup (Zhang 2019b https://arxiv.org/abs/1901.09321)、ReZero (Bachlechner 2021)、DeepNet (Wang 2024 https://arxiv.org/abs/2203.00555)、B2T (Takase 2023 https://aclanthology.org/2023.findings-acl.194)、NormFormer (Shleifer 2021 https://arxiv.org/abs/2110.09456)、Spectrum Control (Wang 2020 https://openreview.net/forum?id=H1xuzdEcYB)、attention entropy regularization (Zhai et al. 2023 https://arxiv.org/abs/2305.10337) 等。

但这些都只是 practical fix,**为什么 causal decoder 在 training 中一旦接近collapse就爬不出来**这个核心问题没有完整机制论解释。本paper填补这个空白。

---

## 2. Preliminaries:符号系统与架构定义

### 2.1 核心scalar: Token Similarity

整个论文的状态变量是一个scalar:

$$
\mathrm{t}_{\mathrm{sim}}(\mathbf{X}) := \frac{\|\boldsymbol{\Pi}_1 \mathbf{X}\|_F^2}{\|\mathbf{X}\|_F^2}, \qquad \boldsymbol{\Pi}_1 := \frac{1}{n}\mathbb{1}_n \mathbb{1}_n^\top
$$

**变量含义**:

- $n$ = sequence length (paper实验中 $n=2048$)
- $\mathbb{1}_n \in \mathbb{R}^{n \times 1}$ = all-one column vector
- $\boldsymbol{\Pi}_1 \in \mathbb{R}^{n \times n}$ = mean projection matrix,把每个row替换为所有row的均值
- $\mathbf{X} \in \mathbb{R}^{n \times d}$ = hidden representation matrix, $d$ = model dimension ($d=512$)
- $\|\cdot\|_F$ = Frobenius norm

**几何直觉**:

- $\boldsymbol{\Pi}_1 \mathbf{X}$ 是"共享component":每一行都等于 $\mathbf{X}$ 的行均值
- $\mathrm{t}_{\mathrm{sim}} = 1$ ⟺ $\mathbf{X} = \mathbb{1}_n \mathbf{x}^\top$ (所有row完全相同) ⟺ rank collapse
- $\mathrm{t}_{\mathrm{sim}} = 1/n$ ⟺ 所有row相互正交 (full rank,完全diverse)

这是个**标量化**的rank collapse指标,可类比成matrix的"rank-1 energy fraction"。它比直接看rank或者singular value更可解析,因为这个scalar的演化可以写成 closed form。

### 2.2 Post-Norm Block Forward Pass

完整的一个 Post-Norm Transformer block (Equation 1):

$$
\mathbf{X}_1^l = \mathrm{RMS}(\mathbf{Y}_0^l), \quad \mathbf{Y}_1^l = \mathbf{X}_1^l + \mathrm{Attn}(\mathbf{X}_1^l), \quad \mathbf{X}_2^l = \mathrm{RMS}(\mathbf{Y}_1^l), \quad \mathbf{Y}_2^l = \mathbf{X}_2^l + \mathrm{FFN}(\mathbf{X}_2^l)
$$

**上下标含义**:

- 上标 $l$ = layer index ($l \in \{1, \dots, L_{\mathrm{depth}}\}$,实验中 $L_{\mathrm{depth}}=48$)
- 下标 $k \in \{1,2\}$ = sublayer index: $k=1$ 表 attention sublayer,$k=2$ 表 FFN sublayer
- $\mathbf{X}_k^l$ = 进入sublayer $k$ 之前的 normalized 输入
- $\mathbf{Y}_k^l$ = sublayer $k$ residual addition 之后的输出 (未归一化)

简化版 RMSNorm (no learnable gain, no $\epsilon$):

$$
\mathrm{RMS}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\|\mathbf{x}\|^2/d}}
$$

逐 row 施加,保证每行 L2 norm = $\sqrt{d}$。

### 2.3 Sublayer 定义

**Attention branch** (single-head,Equation 2):

$$
\mathrm{Attn}(\mathbf{X}) = \mathbf{P} \mathbf{X} \mathbf{W}, \qquad \mathbf{W} := \mathbf{W}_V \mathbf{W}_O
$$

$$
\mathbf{P} = \mathrm{Softmax}_{\mathrm{row}}\!\left( \frac{(\mathbf{X}\mathbf{W}_Q)(\mathbf{X}\mathbf{W}_K)^\top}{\sqrt{d}} + \mathbf{M}_{\mathrm{causal}} \right)
$$

- $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O \in \mathbb{R}^{d \times d}$
- $\mathbf{W}$ = folded value-output projection,Gaussian $\sigma_{\mathbf{W}}^2 = d \sigma_{\mathbf{W}_V}^2 \sigma_{\mathbf{W}_O}^2$
- $\mathbf{M}_{\mathrm{causal}} \in \mathbb{R}^{n \times n}$:$(\mathbf{M}_{\mathrm{causal}})_{i,j} = 0$ if $i \geq j$, $-\infty$ otherwise

**FFN branch (SwiGLU, Shazeer 2020 https://arxiv.org/abs/2002.05202)**:

$$
\mathrm{FFN}(\mathbf{X}) = \big( \mathrm{SiLU}(\mathbf{X}\mathbf{W}_1) \odot \mathbf{X}\mathbf{W}_3 \big) \mathbf{W}_2
$$

- $\mathrm{SiLU}(z) = z / (1 + \exp(-z))$ (Sigmoid Linear Unit / Swish)
- $\odot$ = Hadamard product
- $\mathbf{W}_1, \mathbf{W}_3 \in \mathbb{R}^{d \times d_{\mathrm{ff}}}$ ($d_{\mathrm{ff}}=1536$ in exp)
- $\mathbf{W}_2 \in \mathbb{R}^{d_{\mathrm{ff}} \times d}$

### 2.4 标准初始化

| 矩阵 | shape | variance |
|---|---|---|
| $\mathbf{W}_V, \mathbf{W}_O, \mathbf{W}_Q, \mathbf{W}_K$ | $\mathbb{R}^{d \times d}$ | $1/(3d)$ |
| $\mathbf{W}_1, \mathbf{W}_3$ | $\mathbb{R}^{d \times d_{\mathrm{ff}}}$ | $1/(3d)$ |
| $\mathbf{W}_2$ | $\mathbb{R}^{d_{\mathrm{ff}} \times d}$ | $1/(3d_{\mathrm{ff}})$ |

注意 SwiGLU 比 ReLU FFN 多一个 gate matrix,所以 variance 分母里有 3 不是 2。

---

## 3. Stage I: Forward Similarity Amplification

整个第一阶段回答一个问题:**初始化时,t_sim 怎么变?**

### 3.1 三个 approximation

要让 closed-form 公式可解,作者做了三个 simplification:

#### Approximation 1: Prefix-Averaging (Assumption C.2)

把随机 attention matrix $\mathbf{P}$ 替换为 **causal prefix-averaging matrix** $\mathbf{C}_n$:

$$
(\mathbf{C}_n)_{i,j} = \begin{cases} 1/i & \text{if } i \geq j \\ 0 & \text{otherwise} \end{cases}
$$

- 第 $i$ 行: 前 $i$ 个位置均匀权重 $1/i$,后面 0
- 这正是"每个 token attend 到它前面所有 token 的均匀平均"
- Appendix F.2 实测 $\|\mathbb{E}[\mathbf{P}^l] - \mathbf{C}_n\|_F / \|\mathbf{C}_n\|_F$ 大约 2.1%,验证了 approximation

**直觉**:在 random init,attention logits $\mathbf{X}\mathbf{W}_Q \mathbf{W}_K^\top \mathbf{X}^\top / \sqrt{d}$ 是 zero-mean Gaussian,softmax 输出近似均匀分布;causal mask 让每个 token 只看 prefix,prefix 内 softmax≈uniform,即 prefix mean。

#### Approximation 2: Ratio-of-Expectations

$$
\mathbb{E}[\mathrm{t}_{\mathrm{sim}}(\mathbf{Y})] = \mathbb{E}\!\left[\frac{\|\boldsymbol{\Pi}_1 \mathbf{Y}\|_F^2}{\|\mathbf{Y}\|_F^2}\right] \approx \frac{\mathbb{E}[\|\boldsymbol{\Pi}_1 \mathbf{Y}\|_F^2]}{\mathbb{E}[\|\mathbf{Y}\|_F^2]}
$$

- 期望的 ratio 没有闭式,改用 ratio of expectations
- Appendix C.7 用 Hanson-Wright + Bernstein + sub-Gaussian concentration 证明这个surrogate在 width $d$ 大时concentrate
- Lemma C.11: conditioned on $(\mathbf{X}, \mathbf{P})$, $\xi_2 := \|\mathbf{Y}\|_F^2 / \|\mathbf{X}\|_F^2$ 是 sub-exponential,concentration scale 是 $\sigma_W^2 \|\mathbf{P}\mathbf{X}\|_F^2 / \|\mathbf{X}\|_F^2$,在初始化时是 $O(1)$

#### Approximation 3: Equal-Correlation Closure (Assumption 3.1 / C.1(iii))

把 Gram matrix $\mathbf{X}_1 \mathbf{X}_1^\top$ 替换为只由 $t_1 := \mathrm{t}_{\mathrm{sim}}(\mathbf{X}_1)$ 决定的 surrogate:

$$
\mathbf{X}_1 \mathbf{X}_1^\top \approx \mathbf{R}_{\mathrm{eq}}(t_1) := d \big[ (1-\rho(t_1)) \mathbf{I} + n \rho(t_1) \boldsymbol{\Pi}_1 \big]
$$

$$
\rho(t_1) = \frac{n t_1 - 1}{n - 1}
$$

**变量与约束**:

- $\rho(t_1)$ = pairwise row correlation coefficient (off-diagonal / diagonal ratio)
- 约束1: 对角元素 $= d$ (因为每行 RMS-normalized,$\|(\mathbf{X}_1)_{i,:}\|_2^2 = d$)
- 约束2: $\mathrm{tr}(\boldsymbol{\Pi}_1 \mathbf{R}_{\mathrm{eq}}) / \mathrm{tr}(\mathbf{R}_{\mathrm{eq}}) = t_1$ (定义性约束)

**直觉**: 整个 Gram matrix 由 $n^2$ 个 entry 描述,但 $t_1$ 这个 scalar 只能约束 1 个自由度(平均 off-diagonal / 平均 diagonal)。所以 surrogate 假设所有 off-diagonal 取相同值 $\rho d$,所有 diagonal 取 $d$。这是把"任意相似度分布"塌缩成"等相似度分布"的 mean-field 风格 closure。

### 3.2 Theorem 3.2: Attention 单步 closed-form

经过这三个 approximation,得到 attention 单步 similarity 变化 (Equation 5):

$$
\boxed{\Delta_{\mathrm{attn}}(s, t_1) := \mathbb{E}_{\mathbf{W}}[\mathrm{t}_{\mathrm{sim}}(\mathbf{Y}_1)] - \mathrm{t}_{\mathrm{sim}}(\mathbf{X}_1) = \frac{s \, f_1(t_1)}{1 + s \, f_2(t_1)}}
$$

**变量**:

- $s := \frac{n d^2 \sigma_{\mathbf{W}}^2}{\|\mathbf{X}_1\|_F^2}$ = "amount of attention" scalar
  - Post-Norm 中 $\|\mathbf{X}_1^l\|_F^2 = n d$ 对所有 $l$,所以 $s = d \sigma_{\mathbf{W}}^2$ = constant across layers
  - Pre-Norm 中 residual stream 不重新归一化,$\|\mathbf{X}_1^l\|_F$ 随 depth 线性增长,$s \sim O(d\sigma_{\mathbf{W}}^2/l)$
- $f_1(t_1) = \frac{1}{n(n-1)} (1-t_1)(n - H_n)(n t_1 + 1)$
- $f_2(t_1) = \frac{1}{n-1} \big[ (n - H_n) t_1 + H_n - 1 \big]$
- $H_n = \sum_{k=1}^n 1/k$ harmonic number, $H_n \approx \log n + \gamma_{\mathrm{euler}} \approx 0.5772$

### 3.3 Theorem 3.2 的关键性质

**性质1: $\Delta_{\mathrm{attn}} > 0$ for $t_1 \in [1/n, 1)$**

每项 $f_1, f_2$ 中的所有 factor 都严格正,所以 attention 总是increase similarity。这是崩溃的 forward 推力。

**性质2: 单调性**

$$
\frac{\partial \Delta_{\mathrm{attn}}}{\partial s} = \frac{f_1(t_1)}{(1 + s f_2(t_1))^2} > 0
$$

- amount of attention 越大,similarity 增长越多
- 解释了为什么深层 Post-Norm (各层 $s$ 恒定) 比 Pre-Norm (深层 $s$ 衰减) 增长快

**性质3: $f_1$ 在 $t_1 = (n-1)/(2n) \approx 1/2$ 处取最大**

- similarity 还没到崩溃时,attention amplification 最强
- 一旦 $t_1$ 接近 1,边际 amplification 反而减小 (因为已经接近天花板)

### 3.4 物理直觉:为什么 C_n 放大 similarity

考虑前向传播 $\mathbf{Y}_1 = \mathbf{X}_1 + \mathbf{C}_n \mathbf{X}_1 \mathbf{W}$。

- $\mathbf{C}_n \mathbf{X}_1$ 把第 $i$ 个 token 替换成前 $i$ 个 token 的均值
- $\boldsymbol{\Pi}_1 \mathbf{C}_n \mathbf{X}_1$ 提取这个 prefix-averaged 序列的全局均值
- 关键洞察: **prefix averaging 是一个低通滤波器**,它会把序列中每个 token 拉向其前缀均值,而前缀均值已经包含了全局均值的成分。如果 tokens 已经相似,prefix mean 几乎等于全局 mean,prefix averaging 就等于 mean projection,similarity 不变;如果 tokens 还有差异,prefix averaging 会**消灭这些差异的高频部分**。

更精确地看 $\|\boldsymbol{\Pi}_1 \mathbf{C}_n\|_F^2$ (Lemma in Appendix C.2 推导):

$$
\|\boldsymbol{\Pi}_1 \mathbf{C}_n\|_F^2 = 2 - \frac{H_n}{n}
$$

- 这个值接近 2 (远大于 $\|\boldsymbol{\Pi}_1\|_F^2 = 1$)
- 意味着 prefix averaging 之后,signal energy 中 shared component 的比例放大了约 2 倍

而 $\|\mathbf{C}_n\|_F^2 = H_n \approx \log n$ (Frobenius norm 是调和级数),所以 overall signal 的 scaling 也增大。两者比例决定了 similarity 增量。

### 3.5 Corollary 3.3: Removing Prefix-Averaging 作为干预实验

如果把 $\mathbf{P}$ 替换为 $\mathbf{P} - \alpha \mathbf{C}_n$,那 similarity 增量变为:

$$
\Delta_{\mathrm{de}}(s, t_1) = \frac{(1-\alpha)^2 \, s \, f_1(t_1)}{1 + (1-\alpha)^2 \, s \, f_2(t_1)}
$$

- 关于 $\alpha$ 单调递减
- $\alpha = 1$ 时,attention 对 similarity 的贡献完全消失

这是 paper 的 key intervention 实验 (Figure 2)。当 $\alpha$ 从 0 加到 1,t_sim across depth 显著被压低。这是"因果归因"—— 把 similarity 增长直接归到 prefix averaging 这一项上。

### 3.6 Theorem 3.4: SwiGLU FFN 的 damping

FFN 的 similarity 变化 (Equation 6):

$$
\Delta_{\mathrm{FFN}}(\xi, t_2) = \frac{\xi (t_2 - 1/n)(m(\rho(t_2)) - m(1))}{1 + \xi m(1)}
$$

**变量**:

- $\xi := d \, d_{\mathrm{ff}} \, \sigma_{\mathbf{W}_2}^2 \sigma_{\mathbf{W}_3}^2$ = FFN 强度系数
- $\rho(t_2) = (n t_2 - 1)/(n-1)$ 同 attention
- $m(\rho)$ = **pairwise SwiGLU moment**:
$$
m(\rho) := \mathbb{E}[\mathrm{SiLU}(u) \, \mathrm{SiLU}(v)], \quad (u,v) \sim \mathcal{N}\!\left(0, \sigma_{\mathbf{W}_1}^2 d \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right)
$$

**数值**:在标准 init 下 $m(\rho)$ 单调增 (Appendix C Figure 7):
- $m(0) \approx 0.006$
- $m(1) \approx 0.100$

所以 $m(\rho(t_2)) - m(1) < 0$ 当 $t_2 < 1$,即 $\Delta_{\mathrm{FFN}} < 0$。

**SwiGLU 提供 small damping**:
$$
|\Delta_{\mathrm{FFN}}| \leq \xi \, m(1) \approx \frac{1}{9} \cdot 0.10 \approx 0.011
$$

对比 attention 的 $O(s)$ 量级,$\xi m(1)$ 几乎可忽略。这就是为什么作者说 forward 阶段 **attention dominates, FFN only damps mildly**。

### 3.7 Post-Norm vs Pre-Norm: $s$ 的演化

Theorem 3.2 对两者形式相同,差异在 $s^l$:

| | Post-Norm | Pre-Norm |
|---|---|---|
| $\|\mathbf{X}_1^l\|_F^2$ | $n d$ (常数,因 RMSNorm 每层后归一化) | $\propto l$ (线性增长,Xiong et al. 2020) |
| $s^l$ | $d \sigma_{\mathbf{W}}^2$ (常数) | $O(d\sigma_{\mathbf{W}}^2/l)$ (衰减) |
| $\Delta_{\mathrm{attn}}(s^l, t)$ | 每层相同 | 深层 $\to 0$ |

这就是 Pre-Norm 在深层稳定的核心:深层 attention contribution scale 衰减,相似度增长被天然抑制。

### 3.8 实验:Figure 1 验证

48-layer stack,sublayer init variance sweep ($\times 1, \times 2, \times 4$):

- x 轴: layer index (1-48)
- y 轴: one-step $\Delta t_{\mathrm{sim}}$
- 左列: attention 增量,右列: SwiGLU 增量
- 上排 Post-Norm,下排 Pre-Norm

观察:

1. Post-Norm attention 增量在浅层很大,$t_{\mathrm{sim}}$ 迅速饱和到 1,后续层增量降到近 0
2. Pre-Norm 增量随 depth 缓慢衰减,不会饱和
3. SwiGLU 增量很小且为负
4. 理论曲线 (dashed) 与实测 (solid) 符合 sign 和 overall scale

---

## 4. Stage II: Backward Repair Incapacity

第一阶段告诉我们初始化时 similarity 已经被推高,但 training 完全可以repair。第二阶段回答:**为什么 repair 不发生?**

### 4.1 RMSNorm Backward Jacobian

反向传播时,RMSNorm 的 Jacobian 是 (Section 3.2):

$$
\mathbf{J}_{\mathrm{RMS}}(\mathbf{y}) := \frac{1}{\sqrt{\|\mathbf{y}\|^2/d}} \big( \mathbf{I} - \mathrm{Proj}(\mathbf{y}^\top) \big), \quad \mathrm{Proj}(\mathbf{y}) := \frac{\mathbf{y} \mathbf{y}^\top}{\|\mathbf{y}\|^2}
$$

**两个 contractive term**:

1. **Pre-factor** $1/\sqrt{\|\mathbf{y}\|^2/d} = \sqrt{d}/\|\mathbf{y}\|_2$:
   - residual norm $\|\mathbf{y}\|$ 增长 → 这个因子减小
2. **Projection** $\mathbf{I} - \mathrm{Proj}(\mathbf{y})$:
   - 删除与 $\mathbf{y}$ 对齐的 component
   - 这一项 spectral norm = 1,本身不缩,但限制了 gradient 的方向

paper 只分析 first term (pre-factor),因为它在 residual norm 增长时主导 contraction。

### 4.2 Sublayer Gradient Contribution Factor (Definition 3.5)

定义:

$$
\alpha_k^l := \frac{\|\partial \mathcal{L}/\partial \mathbf{X}_k^l\|_F}{\|\partial \mathcal{L}/\partial \mathbf{Y}_k^l\|_F}
$$

- 衡量 sublayer $k$ 把 backward gradient 放大或缩小多少
- 包含 skip-connection 和 sublayer branch 两部分
- 在 Post-Norm 中,两者都经过 RMSNorm 的 Jacobian

### 4.3 Theorem 3.6: Exact-Collapse Gradient Contraction

在 exact collapse ($\mathrm{t}_{\mathrm{sim}}(\mathbf{Y}_k^l) = \mathrm{t}_{\mathrm{sim}}(\mathbf{X}_k^l) = 1$) 假设下,$\mathbf{Y}_k^l = \mathbb{1}_n (\mathbf{y}_k^l)^\top$,可以推出:

$$
\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{X}_k^l}\right\|_F \leq c(\mathbf{y}_k^l, \alpha_k^l) \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{X}_{k+1}^l}\right\|_F
$$

$$
\boxed{c(\mathbf{y}, \alpha) := \alpha \cdot \frac{\sqrt{d}}{\|\mathbf{y}\|_2}}
$$

**关键 condition**: 当 $\|\mathbf{y}\|_2^2 / d > \alpha^2$,即 $\|\mathbf{y}\|_2 > \sqrt{d} \, \alpha$ 时,$c < 1$,**gradient contracts**。

合并两个 sublayer ($k=1,2$):

$$
\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{X}_1^l}\right\|_F \leq c(\mathbf{y}_1^l, \alpha_1^l) \, c(\mathbf{y}_2^l, \alpha_2^l) \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{X}_1^{l+1}}\right\|_F
$$

如果对每个 collapsed layer 都有 $c < \gamma < 1$,那 gradient reaching earlier layers 是 exponentially小:

$$
\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{X}_1^1}\right\|_F \leq \gamma^{L_{\mathrm{depth}}} \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{X}_1^{L_{\mathrm{depth}}}}\right\|_F
$$

对 $L_{\mathrm{depth}}=48$, $\gamma = 0.9$,这个衰减到 $0.9^{48} \approx 0.006$,实际上早期层 gradient 比末层小 3 个数量级。

### 4.4 Pre-Norm Comparison: J_RMS 位置差异

Pre-Norm backward:

$$
\left(\frac{\partial \mathcal{L}}{\partial \mathbf{X}_k}\right)_{i,:} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{X}_{k+1}}\right)_{i,:} + \left(\mathrm{Grad}_S\!\left(\mathrm{RMS}(\mathbf{X}_k), \frac{\partial \mathcal{L}}{\partial \mathbf{X}_{k+1}}\right)\right)_{i,:} \mathbf{J}_{\mathrm{RMS}}((\mathbf{X}_k)_{i,:})
$$

**关键区别**:

- **Pre-Norm**: $\mathbf{J}_{\mathrm{RMS}}$ 只乘以 sublayer branch 的 gradient,skip-connection gradient 不变
- **Post-Norm**: $\mathbf{J}_{\mathrm{RMS}}$ 乘以整个 (sublayer + skip) gradient

所以当 residual norm 增长时:
- Post-Norm: 整个 backward signal 都被 shrink
- Pre-Norm: 只 shrink sublayer 那部分,主路径保持

这就是 Theorem 3.6 的 contraction mechanism 在 Pre-Norm 中失效的原因。

### 4.5 Proposition D.2: Residual Norm Growth at Exact Collapse

要使 Theorem 3.6 的 contraction 持续发生,需要 residual norm $\|\mathbf{y}\|$ 持续增长。Proposition D.2 证明了这个增长在 exact collapse 下是 monotonically driven by gradient step:

**Attention case**: 一次 $\mathbf{W}_O$ 的 gradient step:
$$
\mathbf{Y}_{1, W_O}(\eta) = \mathbf{Y}_1 - \eta \mathbf{U}_1, \quad \mathbf{U}_1 := \mathbf{H}_1 \mathbf{H}_1^\top \mathbf{G}_1, \quad \mathbf{G}_1 := \partial \mathcal{L}/\partial \mathbf{Y}_1
$$

$$
\|\mathbf{Y}_{1, W_O}(\eta)\|_F^2 = \|\mathbf{Y}_1\|_F^2 + \eta^2 \|\mathbf{U}_1\|_F^2
$$

**关键**: cross term $-2\eta \langle \mathbf{Y}_1, \mathbf{U}_1 \rangle = 0$ 因为 **RMSNorm Jacobian 让 $\mathbf{G}_1$ 与 $\mathbf{Y}_1$ 正交** (Lemma: $\partial \mathcal{L}/\partial \mathbf{Y} \cdot \mathbf{Y}^\top = 0$)。

**直觉**: RMSNorm 的 backward gradient 永远垂直于 residual stream 自身(因为 RMSNorm 删除了 $\mathbf{y}$ 方向的 component)。所以 gradient step 推 $\mathbf{Y}_1$ 朝垂直方向走,$\|\mathbf{Y}_1\|^2$ 严格增加(勾股定理)。

FFN case 类似:
$$
\|\mathbf{Y}_{2, W_2}(\eta)\|_F^2 = \|\mathbf{Y}_2\|_F^2 + \eta^2 \|\mathbf{U}_2\|_F^2, \quad \mathbf{U}_2 := \mathbf{R}_s \mathbf{R}_s^\top \mathbf{G}_2
$$

这就是 **residual norm 自增强 growth**: gradient step 让 norm 增大 → 下一步 RMSNorm factor $\sqrt{d}/\|\mathbf{y}\|$ 更小 → backward signal 更弱 → repair 更难。

### 4.6 Proposition D.1: Near-Collapse Extension

exact collapse 是 idealization,真实 training 中 $t_{\mathrm{sim}}$ 不会精确等于 1。Proposition D.1 把 contraction 推广到 near-collapse:

设 $\mathrm{t}_{\mathrm{sim}}(\mathbf{Y}_k) = 1 - \delta$ with $\delta \leq 1/(4n+1)$,分解 $\mathbf{Y}_k = \mathbb{1}_n \bar{\mathbf{y}}_k^\top + \mathbf{F}$,则:

$$
\frac{\sqrt{d}}{\|\mathbf{y}_{k,i}\|_2} \leq c_0 \left(1 + O(\sqrt{n\delta})\right), \quad c_0 := \frac{\sqrt{d}}{\|\bar{\mathbf{y}}_k\|_2}
$$

所以 contraction 在 near-collapse neighborhood 也成立,只要 $c_0 \alpha_k < 1 - \eta$ for some $\eta > 0$。

这避免了"理论只在 measure-zero 的 exact-collapse point 成立"的批评。

### 4.7 Experiments: Figure 3-5 验证

实验 setup: LR = 8e-4, transition window at step ≈ 2644。

**Figure 3**: $\alpha_k^l$ over full training
- (a) training loss (sharp rise at step 2644)
- (b) last-layer t_sim mean (sharp rise)
- (c) attention $\alpha_1^l$ across layers 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48
- (d) FFN $\alpha_2^l$ across same layers

观察:
- $\alpha_k^l$ 在 transition window 附近都有 jump
- 但 $\alpha < 2.5$ 始终保持 bounded
- 说明 sublayer 自身 contribution 不大,contraction 主要来自 RMSNorm factor

**Figure 4**: transition window zoom (step 2640-2650)
- (a) attention $\sqrt{d}/\|\mathbf{y}_1^l\|_2$: 高 similarity 层在 transition 急剧下降
- (b) attention $c(\mathbf{y}_1^l, \alpha_1^l)$: 匹配下降,大部分层 $c < 1$
- (c) FFN $\sqrt{d}/\|\mathbf{y}_2^l\|_2$ 同样下降
- (d) FFN $c(\mathbf{y}_2^l, \alpha_2^l)$ 跨越 1
- 例外: attention layer 1,$\alpha_1^1$ 上升过快, $c$ 没下降

**Figure 5**: per-layer gradient norm at step 2640-2650
- 早期层 gradient norm 下降多个数量级
- 晚期层变化不大
- geometric compounding of per-block contraction 直接可视化

**Appendix F.4 LR ablation**: LR = 1.2e-3, 1.5e-3, 1.8e-3 都collapse, transition 在不同 step,但 pattern 一致。Figure 12-20。
**Appendix F.5 non-collapse control**: LR = 6e-4 不collapse, 所有 backward 量保持稳定。

---

## 5. Stage III: Properties of Collapsed Network

第三阶段回答:**collapse 后网络的 loss 会停在哪里?为什么 stationary?**

### 5.1 Frequency Distribution (Definition 3.7)

给定 label sequence $\mathbf{y} \in (\mathbb{Z}^+)^n$,令 $c_i = \sum_j \mathbb{I}[y_j = i]$ 为 label $i$ 在 $\mathbf{y}$ 中的 count:

$$
\mathbf{p}_{\mathrm{freq}}(\mathbf{y}) = \left[ \frac{c_i}{n} \right]_{i=1}^v, \qquad \mathcal{L}_{\mathrm{freq}}(\mathbf{y}) = \mathcal{H}(\mathbf{p}_{\mathrm{freq}}(\mathbf{y})) = -\sum_{i=1}^v (\mathbf{p}_{\mathrm{freq}})_i \log (\mathbf{p}_{\mathrm{freq}})_i
$$

**直觉**: 如果每个 position 输出同一个 distribution $\hat{\mathbf{p}}$,那 cross-entropy 是 $-\sum_i c_i \log \hat{p}_i / n$,最优解是 $\hat{p}_i \propto c_i$,即 frequency distribution。最优 loss 是这个分布的 entropy。

### 5.2 Theorem 3.8: Collapsed Network Properties

#### Part (i): Lower Bound

设 $\mathbf{X}^H \in \mathbb{R}^{n \times d}$ 是 LM head 前的 hidden,$\mathbf{W}_{\mathrm{lm}} \in \mathbb{R}^{d \times v}$ 是 LM head。若 $\|\mathbf{X}^H - \boldsymbol{\Pi}_1 \mathbf{X}^H\|_F \leq \epsilon$,则:

$$
\mathcal{L}_{\mathrm{CE}}(F_\theta(\mathbf{t}), \mathbf{y}) \geq \mathcal{L}_{\mathrm{freq}}(\mathbf{y}) - \frac{2}{\sqrt{n}} \|\mathbf{W}_{\mathrm{lm}}\|_2 \, \epsilon
$$

**证明 sketch** (Appendix E.2):

1. Cross-entropy 在 logit $\mathbf{Z} = \mathbf{X} \mathbf{W}_{\mathrm{lm}}$ 上 convex (log-sum-exp 是 convex)
2. $\mathbf{Z}$ 关于 $\mathbf{X}$ (固定 $\mathbf{W}_{\mathrm{lm}}$) 是 linear,所以 loss 关于 $\mathbf{X}$ convex
3. $\partial \mathcal{L}/\partial \mathbf{X} = (\mathrm{Softmax}(\mathbf{X}\mathbf{W}_{\mathrm{lm}}) - \mathbf{E}(\mathbf{y})) \mathbf{W}_{\mathrm{lm}}^\top / n$
4. $\|\mathrm{Softmax}\|_F \leq \sqrt{n}$ (每行 probability vector, L2 norm ≤ 1)
5. $\|\mathbf{E}(\mathbf{y})\|_F = \sqrt{n}$ (n 个 one-hot)
6. $\|\partial \mathcal{L}/\partial \mathbf{X}\|_F \leq \frac{2}{\sqrt{n}} \|\mathbf{W}_{\mathrm{lm}}\|_2$
7. Subgradient inequality: $\mathcal{L}(\mathbf{X}) \geq \mathcal{L}(\bar{\mathbf{X}}) + \langle \nabla, \mathbf{X} - \bar{\mathbf{X}} \rangle \geq \mathcal{L}(\bar{\mathbf{X}}) - \|\nabla\|_F \cdot \epsilon$
8. $\bar{\mathbf{X}} = \boldsymbol{\Pi}_1 \mathbf{X}$ 是 row-constant,最优 loss 即 frequency loss

**直觉**: 当 hidden state 接近 collapse,$\mathbf{X}^H$ 几乎是 row-constant。最优 output 是 frequency distribution,任何偏离都被 LM head 放大 bounded by $\|\mathbf{W}_{\mathrm{lm}}\|_2$。所以 collapse 后的 loss floor 是 frequency loss 减去一个小项。

#### Part (ii): Gradient Vanishing

如果 output = $\mathbf{p}_{\mathrm{freq}}(\mathbf{y})$ at every position,且 $\mathrm{t}_{\mathrm{sim}}(\mathbf{X}_k^l) = 1$ for collapsed layers,则 collapsed layers 中所有 parameter matrix 的 gradient = 0。

**证明 structure** (Appendix E.3):

通过四个 lemmas 串起来:

**Lemma E.2** (Base case at LM head):
- 若 $\hat{\mathbf{P}} = \mathbb{1}_n \hat{\mathbf{p}}^\top$ (rank-1 output),则 $\mathbb{1}_n^\top (\hat{\mathbf{P}} - \mathbf{E}(\mathbf{y})) = 0$ (zero column sum)
- 因为 $\mathbb{1}_n^\top \hat{\mathbf{P}} = n \hat{\mathbf{p}}^\top$,而 $\hat{\mathbf{p}} = \mathbf{p}_{\mathrm{freq}}$ 时 $\mathbb{1}_n^\top \mathbf{E}(\mathbf{y}) = [c_i]_i = n \hat{\mathbf{p}}^\top$
- $\partial \mathcal{L}/\partial \mathbf{W}_{\mathrm{lm}} = \mathbf{X}^\top (\hat{\mathbf{P}} - \mathbf{E}(\mathbf{y})) = \mathbf{x} (\mathbb{1}_n^\top (\hat{\mathbf{P}} - \mathbf{E}(\mathbf{y}))) = 0$
- $\mathbb{1}_n^\top \partial \mathcal{L}/\partial \mathbf{X}^H = 0$

**Lemma E.1** (Forward closure):
- row-constant input → row-constant output (for Attn, FFN, RMSNorm)
- collapse 是 absorbing state

**Theorem E.3(a)** (FFN gradient vanishing):
- 假设 $\mathbb{1}_n^\top \partial \mathcal{L}/\partial \mathbf{Y}_2^l = 0$ 且 $\mathbf{X}_2^l = \mathbb{1}_n (\mathbf{x}_2^l)^\top$
- SwiGLU 的所有中间量 $\mathbf{A}, \mathbf{G}, \mathbf{U}, \mathbf{H}$ 都是 row-constant
- 利用 $\mathbf{M} \odot (\mathbb{1}_n \mathbf{v}^\top) = \mathbf{M} \mathrm{Diag}(\mathbf{v})$ 性质
- 证明 $\partial \mathcal{L}/\partial \mathbf{W}_1 = \partial \mathcal{L}/\partial \mathbf{W}_2 = \partial \mathcal{L}/\partial \mathbf{W}_3 = 0$
- 且 $\mathbb{1}_n^\top \partial \mathcal{L}/\partial \mathbf{X}_2^l = 0$ (零 column sum 性质传递)

**Theorem E.3(b)** (Attention gradient vanishing):
- 假设 $\mathbb{1}_n^\top \partial \mathcal{L}/\partial \mathbf{Y}_1^l = 0$ 且 $\mathbf{X}_1^l = \mathbb{1}_n (\mathbf{x}_1^l)^\top$
- 关键性质: $\mathbf{P}_h \mathbb{1}_n = \mathbb{1}_n$ (softmax row-stochastic)
- 所以 $\mathbf{O}_h = \mathbf{P}_h \mathbf{V}_h = \mathbb{1}_n \mathbf{v}_h^\top$ (row-constant)
- $\partial \mathcal{L}/\partial \mathbf{W}_O = \mathbf{O}^\top \partial \mathcal{L}/\partial \mathbf{Y} = [\mathbf{v}_h \mathbb{1}_n^\top \partial \mathcal{L}/\partial \mathbf{Y}] = 0$
- softmax Jacobian: $\partial \mathcal{L}/\partial \mathbf{S}_h \mathbb{1}_n = 0$ (softmax row-derivative 的零 column sum 性质)
- 推出 $\partial \mathcal{L}/\partial \mathbf{W}_Q = \partial \mathcal{L}/\partial \mathbf{W}_K = \partial \mathcal{L}/\partial \mathbf{W}_V = 0$

**Lemma E.4** (RMSNorm preserves zero column sum):
- 若 $\mathbf{Y} = \mathbb{1}_n \mathbf{y}^\top$,RMSNorm 的 row-wise Jacobian 都相同
- $\partial \mathcal{L}/\partial \mathbf{Y} = \partial \mathcal{L}/\partial \mathbf{X} \cdot \frac{\sqrt{d}}{\|\mathbf{y}\|} (\mathbf{I} - \mathrm{Proj}(\mathbf{y}))$
- $\mathbb{1}_n^\top \partial \mathcal{L}/\partial \mathbf{Y} = (\mathbb{1}_n^\top \partial \mathcal{L}/\partial \mathbf{X}) \cdot (\dots) = 0$

把这些串起来: 从 LM head 出发,zero column sum property 通过 RMSNorm → FFN → RMSNorm → Attention 一路传到上一层的输出。所有 collapsed layer 的 parameter gradient = 0。

**整体 intuition**:

collapse 后,网络输出 frequency distribution,这个分布让所有 collapsed layer 的 parameter gradient 消失。再加上 backward contraction 让早期层 gradient 也消失,网络接近 stationary point。这就是为什么 collapse 是 absorbing state,training 爬不出来。

### 5.3 Experiments: Figure 6

**Left panel**: no-warmup,三种 label scope (full training 500M tokens / single batch 0.5M / single sequence 2048)
- 各自的 training loss 都 approach 对应的 frequency loss

**Right panel**: warmup Post-Norm runs at LR = 1.2e-3, 1.5e-3, 1.8e-3
- transition 在不同 step,但 collapse 后 loss 都 stay near frequency loss

**这验证了 Theorem 3.8 part (i)**: collapse 后 loss floor = frequency loss。

---

## 6. 整体架构图与流程总结

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage I: Forward Amplification (Initialization)                  │
│                                                                   │
│  Causal Attn ≈ prefix averaging C_n                               │
│       ↓                                                           │
│  Y_1 = X_1 + C_n X_1 W                                            │
│       ↓                                                           │
│  t_sim(Y_1) = t_sim(X_1) + Δ_attn(s, t_1),  Δ_attn > 0           │
│       ↓                                                           │
│  Per layer in Post-Norm: s = d σ_W² constant                      │
│  → similarity accumulates linearly with depth                      │
│  → starts training already near collapse                          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  Stage II: Backward Repair Incapacity (Training)                  │
│                                                                   │
│  Residual norm grows under exact collapse                         │
│  (Proposition D.2: ||Y||² grows as η² ||U||²                      │
│   because ∂L/∂Y ⊥ Y due to RMSNorm Jacobian)                      │
│       ↓                                                           │
│  RMSNorm pre-factor √d/||y|| shrinks                             │
│       ↓                                                           │
│  c(y, α) = α √d / ||y|| < 1                                       │
│       ↓                                                           │
│  ||∂L/∂X^l|| ≤ c(y, α)^L · ||∂L/∂X^{L}||                          │
│  → early-layer gradients decay exponentially                      │
│  → repair impossible                                              │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  Stage III: Collapsed Network Properties (Stuck State)             │
│                                                                   │
│  Hidden ≈ rank-1 (all rows identical)                             │
│       ↓                                                           │
│  Optimal output = frequency distribution                          │
│  (Theorem 3.8 (i): loss floor = L_freq)                           │
│       ↓                                                           │
│  Zero column sum property propagates backward                     │
│  (Lemmas E.2-E.4)                                                 │
│       ↓                                                           │
│  All collapsed-layer parameter gradients vanish                   │
│  (Theorem 3.8 (ii))                                               │
│       ↓                                                           │
│  Network is near-stationary → stuck                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键 Technical 附件细节

### 7.1 Appendix C.6: RMSNorm Forward Correction Bound

定理 C.6 给 RMSNorm 的相似度扰动一个 deterministic bound:

设 $\mathbf{T}$ 是 diagonal matrix with positive entries (RMSNorm 的 scaling),$\mu = \mathrm{tr}(\mathbf{T})/n$ 是 mean factor,$\eta = \|\mathbf{T} - \mu \mathbf{I}\|_2 / \mu$ 是相对 spread。若 $\eta < 1$:

$$
\frac{(\max\{\sqrt{t} - \eta, 0\})^2}{(1+\eta)^2} \leq \mathrm{t}_{\mathrm{sim}}(\mathbf{T}\mathbf{Y}) \leq \frac{(\sqrt{t} + \eta)^2}{(1-\eta)^2}
$$

其中 $t = \mathrm{t}_{\mathrm{sim}}(\mathbf{Y})$。

**直觉**: RMSNorm 是 row-wise scaling,各 row 的 scaling factor 不同。如果 spread $\eta$ 小,RMSNorm 几乎不改变 similarity;如果 spread 大 (各 row norm 差异大),similarity 可能漂移。这个 bound 把漂移控制在 $\eta$ 量级。

**Lemma C.5**: Attention branch 的 expected row norm 是 position-dependent:

$$
\mathbb{E}[\|\mathbf{Y}_{i,:}\|_2^2] = \alpha + \beta/i, \quad \alpha = d + d^2 \rho \sigma_W^2, \quad \beta = d^2 \sigma_W^2 (1 - \rho)
$$

- 后面位置的 row norm 更大 (因为 attend to 更多 token)
- spread $\eta$ 在 initialization 是 $O(1/\sqrt{n})$ 量级,所以 RMSNorm forward correction 很小

**Lemma C.4**: FFN branch 的 expected row norm 是 position-independent:
$$
\mathbb{E}[\|\mathbf{Y}_{i,:}\|_2^2] = d + d_{\mathrm{ff}} d^2 \sigma_{\mathbf{W}_3}^2 \sigma_{\mathbf{W}_2}^2 m(1)
$$

### 7.2 Appendix C.7: 浓度不等式证明 ratio-of-expectations

Lemma C.7-C.12 用三个经典 concentration inequality:

- **Hoeffding** (Lemma C.8): sub-Gaussian sum
- **Bernstein** (Lemma C.9): sub-exponential sum
- **Hanson-Wright** (Lemma C.10): quadratic form in sub-Gaussian

具体应用到 $\xi_2 = \|\mathbf{Y}\|_F^2 / \|\mathbf{X}\|_F^2$ (Lemma C.11):
- $\xi_2 = 1 + L_2 + Q_2$
- $L_2 = 2 \langle \mathbf{X}, \mathbf{PXW} \rangle / \|\mathbf{X}\|_F^2$ 是 1D Gaussian,concentration 是 $\exp(-c \epsilon^2 \|\mathbf{X}\|_F^2 / (\sigma_W^2 \|\mathbf{PX}\|_F^2))$
- $Q_2 = \|\mathbf{PXW}\|_F^2 / \|\mathbf{X}\|_F^2$ 是 sum of $d$ independent Gaussian quadratic forms,用 Hanson-Wright + Bernstein
- combined tail bound 是 sub-exponential

在初始化时 $\|\mathbf{PX}\|_F^2 / \|\mathbf{X}\|_F^2 = O(1)$,所以 concentration scale 是 $O(1)$,ratio-of-expectations 在 $d$ 大时 tightly concentrate。

参考: Hanson-Wright 原始论文 https://arxiv.org/abs/1301.3403, Rudelson-Vershynin 浓度综述 https://arxiv.org/abs/1301.3407

---

## 8. Limitations 与 Open Problems

paper 自承:

1. **没解释 transition time**:为什么 LR=8e-4 时 transition 在 step 2644 而不是其他时候?这是 future work。
2. **Equal-correlation closure 是 approximation**:实际 Gram matrix 不是等相关的,但 paper 的目标是 mechanism 而非 tight bound。
3. **Prefix-averaging 是 approximation**:实测 ~2% 误差,但 suffices for theory。
4. **Single-head theory, multi-head exp**:theory 简化为 single-head,实验是 multi-head,作者声称直接 extend。
5. **没考虑 RMSNorm 的 learnable gain 和 $\epsilon$**:实际实现有 gain $\gamma$ 和 $\epsilon$,theory 忽略。

可能的 extension 方向:

- 解释 transition window 触发条件 (与 LR, warmup length, init variance 的关系)
- Tighter bound: 把 equal-correlation 替换为更精细的 closure
- Multi-head theory 直接推导
- Fixup/ReZero/DeepNet 等 stabilization method 的统一理论解释
- 与 attention entropy collapse (Zhai et al. 2023) 的关系

---

## 9. 与 Karpathy 直觉的连接

Karpathy 你在多个 talk 中强调 build intuition,这里给你几个 mental model:

### 9.1 "Post-Norm 是把 ResNet 的 identity path 给打破"

在 He et al. 2015 ResNet (https://arxiv.org/abs/1512.03385) 中,identity skip 让 deep network 训练成为可能,因为 gradient 可以无损流过 identity path。Pre-Norm 保持了这个 identity path (skip-connection gradient 不经过 normalization Jacobian)。Post-Norm 把 LN 放在 residual addition 之后,等价于在 identity path 上插了一个 normalization layer,这个层的 Jacobian 在 residual norm 增长时 contractive。这就是 Theorem 3.6 的几何解释。

### 9.2 "Prefix averaging 是 low-pass filter"

$\mathbf{C}_n$ 矩阵在频域是什么?把 $\mathbf{C}_n$ 看作一个 linear operator,它把 sequence 的每个 position 替换为前缀均值。这是 non-uniform low-pass filter (越往后 position,window 越长,smoothing 越强)。low-pass filter 消灭 high-frequency component,即消灭 position-specific information。这就是为什么 similarity 增长—— position-specific 的部分被 filter out,shared 的部分保留。

### 9.3 "Frequency loss 是 Bayes-optimal under no-information"

如果模型对每个 position 输出同一个 distribution (因为 collapse),那 best prediction 是哪个 distribution?是 marginal distribution of labels。在 C4 上这个 marginal 大约是 unigram distribution,它的 entropy 就是 frequency loss。这是个 information-theoretic lower bound on collapsed network performance。这跟 categorical cross-entropy 的 Bayes risk 概念一致。

### 9.4 "Concentration of measure 救了 approximation"

paper 的三个 approximation (prefix-avg, equal-correlation, ratio-of-expectations) 听起来很激进,但 width $d=512$ 足够大时,concentration of measure (Hanson-Wright 等) 让随机量 tightly 围绕 mean。所以 even rough closure 在大 $d$ 下 works,这就是为什么 Figure 1 实测曲线与 theory dashed line 在 overall scale 上吻合。

### 9.5 "Gradient perpendicular to residual stream"

Proposition D.2 的核心: RMSNorm backward $\partial \mathcal{L}/\partial \mathbf{Y} \perp \mathbf{Y}$。这是因为 RMSNorm 的 Jacobian $\mathbf{I} - \mathrm{Proj}(\mathbf{y})$ 删除了 $\mathbf{y}$ 方向的 component。gradient step 推 $\mathbf{Y}$ 沿垂直方向,导致 $\|\mathbf{Y}\|$ 几何增长 (勾股定理)。这是个非常 elegant 的几何机制: RMSNorm 的"删除自己方向"性质让 residual stream 持续膨胀,膨胀又让 RMSNorm 自身 contractive,形成 self-reinforcing collapse。

---

## 10. 实验数据表汇总

### 10.1 主实验配置

| Parameter | Value |
|---|---|
| Architecture | Llama-2 style decoder-only Post-Norm |
| Layers | 48 |
| $d$ (model dim) | 512 |
| $d_{\mathrm{ff}}$ (FFN hidden) | 1536 |
| Heads | 4 |
| Total params | ~180M |
| Dataset | C4 |
| Seq length $n$ | 2048 |
| Micro-batch | 4 |
| Grad accum | 64 |
| Effective batch | 256 |
| Optimizer | AdamW $\beta_1=0.9, \beta_2=0.95$ |
| Weight decay | 0.1 |
| LR schedule | Cosine, 2000 warmup, 40000 total |
| Grad clip | 1.0 |
| LR sweep | 6e-4 to 1.8e-3 |

### 10.2 Collapse behavior vs LR

| LR | Transition step | Behavior |
|---|---|---|
| 6e-4 | — | Stable, no collapse |
| 8e-4 | ≈ 2644 | Collapse |
| 1.2e-3 | ≈ 3123 (from F.4) | Collapse |
| 1.5e-3 | ≈ 1699 | Collapse |
| 1.8e-3 | ≈ 1754 | Collapse |

### 10.3 Approximation 误差

| Approximation | 测量 | 误差 |
|---|---|---|
| $\mathbb{E}[\mathbf{P}] \approx \mathbf{C}_n$ | relative Frobenius | ~2.1% mean over layers (2.7% layer 1, 1.2% layer 48) |
| $\mathbb{E}[\mathbf{PXX}^\top \mathbf{P}^\top] \approx \mathbb{E}[\mathbf{P}] \mathbb{E}[\mathbf{XX}^\top] \mathbb{E}[\mathbf{P}^\top]$ | unconditional proxy | small throughout stack (Figure 8) |
| SwiGLU moment $m(1)$ | numerical | $\approx 0.100$ |
| SwiGLU damping $|\Delta_{\mathrm{FFN}}|$ | upper bound | $\leq 0.011$ |

### 10.4 Backward quantities at transition (LR=8e-4, step ≈ 2644)

| Quantity | Pre-transition | Post-transition |
|---|---|---|
| $\alpha_k^l$ (sublayer contribution) | $< 1$ | jump to $< 2.5$ |
| $\sqrt{d}/\|\mathbf{y}_k^l\|_2$ (RMS factor) | ~1 | sharp drop, well below 1 for high-sim layers |
| $c(\mathbf{y}_k^l, \alpha_k^l)$ (total contraction) | ~1 | drop below 1 for most layers |
| Early-layer gradient norm | moderate | drops by orders of magnitude |

---

## 11. 对 LLM 训练实践的含义

### 11.1 为什么 Llama/T5/GPT-NeoX 全部用 Pre-Norm

本 paper 提供了 mechanical explanation:

1. Pre-Norm 让 $s^l$ 随 depth 衰减 → forward similarity 增长慢
2. Pre-Norm 让 skip-connection gradient 不经过 RMSNorm Jacobian → backward 不 contract
3. Pre-Norm 的 RMSNorm Jacobian 只乘 sublayer branch → 即使 residual norm 增长也只影响 sublayer 那部分

### 11.2 Warmup 的作用

paper 没直接分析 warmup,但暗示:warmup 让早期 training step 的 LR 小,给一个 grace period 让 network 远离 collapse regime。一旦进入 collapse regime,gradient 已经无法 repair (Theorem 3.6 + 3.8)。

参考 Xiong et al. 2020 ICML "On Layer Normalization in the Transformer Architecture" https://arxiv.org/abs/2002.04745 对 warmup 的分析。

### 11.3 LR sensitivity 的解释

实验显示 LR > 6e-4 时 collapse,LR = 6e-4 时 stable。paper 解释为:LR 越大,gradient step 越大,达到 collapse regime 的速度越快,backward contraction 一旦启动就 self-reinforcing。

### 11.4 Stabilization methods 重新审视

- **Fixup/ReZero/DeepNet**: 通过 rescale residual 让 $s$ 在初始化时更小,减缓 forward amplification
- **NormFormer**: 在 sublayer 内部加 LN,改变 effective $s$
- **B2T**: 把 residual 分成 two-stream,绕过 Post-Norm 的 contraction
- **Spectrum Control**: 直接干预 singular value distribution,抑制 collapse

这些都可以放在 paper 的两阶段框架内理解。

---

## 12. 与其他 Rank Collapse 工作的关系

### 12.1 Dong et al. 2021 (Pure Attention Doubly Exponential)

Dong et al. 证明纯 attention stack (无 residual, 无 FFN) 中 similarity doubly exponentially 收敛到 1。本 paper 的 setting 不同:

- 有 residual connection (He et al. 2016)
- 有 FFN branch (SwiGLU)
- 有 RMSNorm
- 是 causal decoder 不是 encoder

Dong 的结果是 pure attention 的极端 case,本 paper 显示 residual + norm 不足以拯救 Post-Norm。

### 12.2 Noci et al. 2022 (Signal Propagation)

Noci et al. 用 signal propagation 分析初始化时的 rank collapse,把 attention 视为 mean-field operator。本 paper 的 prefix-averaging closure 是更精细的版本,且 paper 扩展到 training dynamics。

### 12.3 Saada et al. 2024 (Spectral Analysis)

Saada et al. 从 spectral 视角分析,区分 depth collapse 和 width collapse。本 paper 用 scalar t_sim 替代 spectrum,在 closed form 上更 tractable。

### 12.4 Yu & Zhang 2026 (Encoder Transformer)

Yu & Zhang 2026 给了 Post-Norm encoder 的 quantitative 分析,token similarity 作为指标。本 paper 是这个工作的 decoder 扩展,且增加了 backward analysis 和 collapsed network characterization。

---

## 13. 总结

这篇 paper 的核心贡献是用一个 scalar state variable $t_{\mathrm{sim}}$ 把 Post-Norm Transformer collapse 的完整动力学串起来:

1. **Forward (initialization)**: prefix-averaging 让 attention 持续放大 similarity,SwiGLU 只提供 small damping;Post-Norm 中 $s^l$ 恒定而 Pre-Norm 中衰减,这是 Post-Norm 不稳定的根源
2. **Backward (training)**: 一旦 similarity 高,RMSNorm Jacobian 因 residual norm 增长而 contractive,gradient to early layers 几何衰减,repair 失败
3. **Collapsed state**: hidden state rank-1 时最优输出是 frequency distribution,gradient vanishing 让 collapse 是 absorbing stationary point

理论配合三个 stage 的实验验证 (Figure 1-6),approximation 误差在 Appendix F 系统检验 (约 2%),多个 LR 和 seed 重复 (Figure 9-22)。

整个 framework 把以前碎片化的现象 (rank collapse, gradient vanishing, warmup sensitivity, frequency loss floor) 统一在一个一致的机械论下。对于想 train very deep Post-Norm Transformer 的人,这个 paper 提供了"哪里会出问题,为什么出问题,以及为什么 Pre-Norm 工作"的清晰图景。

---

## 主要参考文献 (with web links)

- **Vaswani et al. 2017** "Attention Is All You Need" https://arxiv.org/abs/1706.03762
- **Xiong et al. 2020 ICML** "On Layer Normalization in the Transformer Architecture" https://arxiv.org/abs/2002.04745
- **Liu et al. 2020 EMNLP** "Understanding the Difficulty of Training Transformers" https://arxiv.org/abs/2009.08798
- **Zhang & Sennrich 2019 NeurIPS** "Root Mean Square Layer Normalization" https://arxiv.org/abs/1910.07467
- **Shazeer 2020** "GLU Variants Improve Transformer" https://arxiv.org/abs/2002.05202
- **He et al. 2016** "Deep Residual Learning" https://arxiv.org/abs/1512.03385
- **Ba et al. 2016** "Layer Normalization" https://arxiv.org/abs/1607.06450
- **Dong et al. 2021 ICML** "Attention is Not All You Need" https://arxiv.org/abs/2103.07448
- **Noci et al. 2022 NeurIPS** "Signal Propagation in Transformers" https://arxiv.org/abs/2206.02202
- **Saada et al. 2024** "Mind the Gap: Spectral Analysis of Rank Collapse" https://arxiv.org/abs/2410.07799
- **Yu & Zhang 2026 JMLR** "Why Classic Transformers Are Shallow" https://jmlr.org/papers/volume27/64
- **Touvron et al. 2023** "Llama 2" https://arxiv.org/abs/2307.09288
- **Raffel et al. 2020 JMLR** "Exploring the Limits of Transfer Learning: T5" https://arxiv.org/abs/1910.10683
- **Loshchilov & Hutter 2019** "Decoupled Weight Decay Regularization: AdamW" https://arxiv.org/abs/1711.05101
- **Zhang et al. 2019b** "Fixup Initialization" https://arxiv.org/abs/1901.09321
- **Bachlechner et al. 2021** "ReZero is All You Need" https://arxiv.org/abs/2003.04887
- **Wang et al. 2024** "DeepNet: Scaling Transformers to 1000 Layers" https://arxiv.org/abs/2203.00555
- **Takase et al. 2023 ACL Findings** "B2T Connection" https://aclanthology.org/2023.findings-acl.194
- **Shleifer et al. 2021** "NormFormer" https://arxiv.org/abs/2110.09456
- **Zhai et al. 2023 ICML** "Stabilizing Transformer Training by Preventing Attention Entropy Collapse" https://arxiv.org/abs/2305.10337
- **Wang et al. 2020 ICLR** "Improving Neural Language Generation with Spectrum Control" https://openreview.net/forum?id=H1xuzdEcYB
- **Pennington et al. 2017** "Resurrecting the Sigmoid in Deep Learning through Dynamical Isometry" https://arxiv.org/abs/1711.04735
- **Schoenholz et al. 2016** "Deep Information Propagation" https://arxiv.org/abs/1611.01232
- **Poole et al. 2016 NeurIPS** "Exponential Expressivity in Deep Neural Networks through Transient Chaos" https://papers.nips.cc/paper/6322
- **Lee et al. 2018** "Deep Neural Networks as Gaussian Processes" https://arxiv.org/abs/1711.00165
- **Hanson-Wright 原始** https://arxiv.org/abs/1301.3403
- **Vershynin 浓度综述** "Introduction to the non-asymptotic analysis of random matrices" https://arxiv.org/abs/1011.3027
- **Chen & Luo 2026 NeurIPS** "From Condensation to Rank Collapse" https://arxiv.org/abs/2402.18818
- **Emadi 2026** "Exact Attention Sensitivity and the Geometry of Transformer Stability" https://arxiv.org/abs/2602.18849
- **Chen & Wei 2026** "Post-LayerNorm is Back: Stable, Expressive, and Deep" https://arxiv.org/abs/2601.19895
- **Gao et al. 2019** "Representation Degeneration Problem in Training Natural Language Generation Models" https://arxiv.org/abs/1907.12009
- **Yan et al. 2022 UAI** "Addressing Token Uniformity in Transformers via Singular Value Transformation" https://arxiv.org/abs/2210.06112
- **Wu et al. 2024 NeurIPS** "On the Role of Attention Masks and LayerNorm in Transformers" https://arxiv.org/abs/2407.15532
- **Nguyen & Salazar 2019** "Transformers without Tears" https://arxiv.org/abs/1910.05895
- **Kaplan et al. 2020** "Scaling Laws for Neural Language Models" https://arxiv.org/abs/2001.08361
- **Hoffmann et al. 2022** "Training Compute-Optimal Large Language Models (Chinchilla)" https://arxiv.org/abs/2203.15556
- **Brown et al. 2020 NeurIPS** "GPT-3: Language Models are Few-Shot Learners" https://arxiv.org/abs/2005.14165
- **Dosovitskiy et al. 2021 ICLR** "An Image is Worth 16x16 Words: ViT" https://arxiv.org/abs/2010.11929
- **Devlin et al. 2019 NAACL** "BERT" https://arxiv.org/abs/1810.04805
- **He et al. 2015** "Delving Deep into Rectifiers" https://arxiv.org/abs/1502.01852
- **Glorot & Bengio 2010 AISTATS** "Understanding the Difficulty of Training Deep Feedforward Neural Networks" http://proceedings.mlr.press/v9/glorot10a.html
- **Neal 2012** "Bayesian Learning for Neural Networks" https://link.springer.com/book/10.1007/978-1-4612-0745-0
- **Matthews et al. 2018** "Gaussian Process Behaviour in Wide Deep Neural Networks" https://arxiv.org/abs/1804.11271
- **Xiao et al. 2018** "Dynamical Isometry and a Mean Field Theory of CNNs" https://arxiv.org/abs/1806.05393
- **Chen et al. 2018** "Dynamical Isometry and RNNs" https://arxiv.org/abs/1806.05394
- **Touvron et al. 2021 ICCV** "Going Deeper with Image Transformers" https://arxiv.org/abs/2103.17239
- **He et al. 2020** "RealFormer: Transformer Likes Residual Attention" https://arxiv.org/abs/2012.11747
- **Xie et al. 2023** "ResiDual: Transformer with Dual Residual Connections" https://arxiv.org/abs/2304.14802
- **Wang et al. 2019 ACL** "Learning Deep Transformer Models for Machine Translation" https://arxiv.org/abs/1906.08455
- **Zhang et al. 2019a EMNLP** "Improving Deep Transformer with Depth-Scaled Initialization" https://arxiv.org/abs/1908.11365

整篇 paper 的核心 takeaway 是: **Post-Norm 的失败是 forward amplification 与 backward contraction 这两个 mechanism 耦合的必然结果**。把 LN 放在 residual 之前 (Pre-Norm) 同时减弱了 forward $s^l$ 和避开了 backward 的 J_RMS 乘 skip-connection,这就是为什么 Pre-Norm 是 LLM 的 de-facto 标准。
