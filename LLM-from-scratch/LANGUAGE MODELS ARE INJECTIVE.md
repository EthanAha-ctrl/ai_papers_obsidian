---
source_pdf: LANGUAGE MODELS ARE INJECTIVE.pdf
paper_sha256: 2c6af6b08253c17e2bed0f2239351892ca4d78aab9bf28c19fdc809113907975
processed_at: '2026-08-05T11:44:49-07:00'
target_folder: LLM-from-scratch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

Andrej，我用最plain的方式重新讲一遍，把math details先放一边，讲story和intuition。

---

## 一句话总结

大家一直觉得Transformer把input信息"压缩"丢了，这篇paper说：**没丢，而且能精确还原**。

---

## 为什么大家之前觉得会丢信息

我们一直有这样的folk belief：

- LayerNorm会把某些方向的information collapse掉
- Attention rank会随depth doubly-exponentially衰减（https://proceedings.mlr.press/v139/dong21a.html）
- Softmax bottleneck限制输出分布的rank（https://arxiv.org/abs/1711.03953）
- Residual connection可以互相cancel

这些都是真的。所以community的intuition一直是："hidden state是input的有损压缩"。

但这篇paper指出一个关键区别：上面这些analysis都是在讨论 $\mathbb{R}^d \to \mathbb{R}^d$ 这种continuous到continuous的映射。而实际我们关心的是**prompt到hidden state的映射**，这里的prompt来自一个finite set $\mathcal{V}^{\leq K}$（vocabulary有限、context length有限）。

从finite set到 $\mathbb{R}^d$，要保证injective，只需要在这些finite个点上不发生collision。这个要求比continuous injectivity弱得多。

---

## 数学直觉：real-analytic函数的"二选一"

整个证明靠一个数学fact：

**如果一个函数能写成收敛的power series（叫real-analytic），那它的零点要么是"到处都是"（函数恒为零），要么"几乎到处都不是"（零集measure zero）。没有中间地带。**

这是跟普通smooth函数最大的区别。Smooth函数的零集可以是任意closed set（用bump function构造），但analytic函数不行——它的零点必须structure很特殊，要么全是零，要么几乎没零点。

具体参考Mityagin 2015: https://arxiv.org/abs/1512.07276

---

## 把这个数学fact用到Transformer

**第一步：Transformer是real-analytic的**

每个building block都保持real-analyticity：
- Embedding lookup: 选行操作，polynomial
- Positional encoding: 加法
- Attention里的softmax: $\exp$ 和除法的组合，分母恒正
- LayerNorm: 关键是有 $\epsilon > 0$ 在分母里，所以 $\sqrt{\sigma^2 + \epsilon}$ 永远positive，可以展开成binomial series
- GELU/tanh: 都是analytic的
- 这些东西加、乘、除、复合起来还是analytic

所以整个 $h(\theta) = \|r(s;\theta) - r(s';\theta)\|_2^2$ 是 $\theta$ 的real-analytic函数。

**第二步：证明 $h$ 不恒为零**

这是关键一步。要证明"存在某个参数设置下，两个不同prompt的表示不同"。

**Case A**：两个序列在最后token或长度上不同。把所有weights设为0，网络退化成identity。让对应位置的embedding不同就行。trivial。

**Case B**：两个序列在中间位置不同，长度和最后token相同。这个更subtle。paper的witness构造很精妙：

设 $i^\star$ 是第一个不同的position。选三个互相正交的向量 $e, p, q$，让token embedding用 $e$，让position $i^\star$ 的positional encoding用 $p$，最后position用 $q$。

然后让**一个attention head**：query用最后position的 $e$ 方向，key用 $i^\star$ 的 $p$ 方向，value用 $i^\star$ 的 $e$ 方向。这样最后位置会attend到 $i^\star$，把那个位置的token信息copy过来。

由于序列s在 $i^\star$ 是某个token（embedding = $e$），序列t是另一个token（embedding ≠ $e$），所以copy过来的信息不同，最终输出不同。

这个构造告诉你：**只要有一个attention head能"看"到第一个不同的位置，injectivity就保住了**。而standard Transformer每个block有多个head，这个条件trivially满足。

**第三步：套用dichotomy**

既然 $h$ real-analytic且不恒为零，那 $h(\theta) = 0$ 的参数集合 $\mathcal{C}$ measure zero。

**第四步：标准初始化避开measure zero set**

Gaussian、uniform、Xavier这些初始化都有density，绝对连续w.r.t. Lebesgue measure。绝对连续分布给measure zero set的概率正好是0。所以初始化后，a.s.没有collision。

---

## 训练为什么不破坏这个性质

GD update是 $\phi(\theta) = \theta - \eta \nabla L(\theta)$。这个 $\phi$ 也是real-analytic（因为loss是real-analytic，gradient也是analytic）。

要担心的是：GD会不会把原本"分散"的参数映射到collision set $\mathcal{C}$ 上？

paper的论证是：
1. $\phi$ 的Jacobian determinant $\det D\phi(\theta)$ 是real-analytic
2. 在某个witness点 $\theta_\star = 0$ 计算Hessian，证明 $\det D\phi(\theta_\star) > 0$
3. 所以 $\det D\phi$ 不恒为零，其零集measure zero
4. 在非零集上，Inverse Function Theorem说 $\phi$ 是local diffeomorphism
5. Diffeomorphism保持绝对连续性（通过change of variables formula）

Hessian在 $\theta_\star = 0$ 的计算是paper最technical的地方。设LN的 $\gamma = 0$，logits退化成 $z = U\beta$（bilinear form），在 $(u, \beta) = (0, 0)$ 处计算二阶导，得到eigenvalues是 $\pm\|w\|$ 各 $d$ 个，加上 $p - 2d$ 个零，其中 $w = \text{softmax}(0) - p = \frac{1}{n}\mathbf{1} - p$。

然后 $D\phi(\theta_\star) = I - \eta H$ 的eigenvalues是 $1 \pm \eta\|w\|$ 各 $d$ 个，加上 $p - 2d$ 个 $1$。所以 $\det D\phi(\theta_\star) = (1 - \eta^2\|w\|^2)^d > 0$（假设 $\eta\|w\| < 1$，这在 $\eta \in (0,1)$ 且 $w$ 有限时成立）。

这样GD一步保持绝对连续，迭代有限步也保持。结论：训练后参数仍然绝对连续，仍然避开collision set。

---

## 把injectivity变成实用算法：SIPIT

Injectivity告诉你不同prompt有不同hidden state，但没说怎么反推。SIPIT算法做这个事。

**核心idea**：Causal masking让position $t$ 的hidden state只依赖prefix $\pi = s_{1:t-1}$ 和当前token $s_t$。所以可以逐位反推。

**One-step map**：固定prefix $\pi$，定义
$$F(v; \pi, t) = h_t(\pi \oplus v)$$
这是"给定前缀、试一个token、看position $t$ 的hidden state"。

**Algorithm**：从 $t=1$ 开始：
1. 已知当前prefix $\pi$
2. 对vocabulary里每个候选token $v$，计算 $F(v; \pi, t)$
3. 找到使 $F(v; \pi, t)$ 等于观察到的 $\hat{h}_t$ 的那个 $v$
4. 那个 $v$ 就是 $s_t$，append到prefix，继续

由于one-step map a.s. injective，这个 $v$ 是唯一的。Worst case每个position遍历整个vocabulary，所以是 $O(T \cdot |\mathcal{V}|)$，linear time。

**Policy优化**：naive遍历vocabulary太慢。paper的gradient-guided policy用一个continuous proxy vector $e$，做gradient descent让 $F(e; \pi, t)$ 接近 $\hat{h}_t$，然后找最近的token embedding。这把runtime从~4000秒降到~28秒，超过100x加速。

---

## 实验讲了什么

### Collision search
- 100k prompts from wikipedia-en + C4 + Pile + python-code
- 6个SOTA模型：GPT-2, Gemma3, Llama-3.1-8B, Mistral-7B, Phi-4-mini, TinyStories-33M
- 5 billion pairwise comparisons
- **Zero collisions observed**

所有模型的minimum pairwise distance在所有layer都远超collision threshold $10^{-6}$。

一些pattern：
- 浅层就有清晰separation（layer 1的min distance ~ $10^{-3}$）
- 深度增加separation（layer L比layer 1大几百倍）
- 更小的model有时separation更大（TinyStories的layer 1 min distance = 0.029 vs Llama的0.001）

### Exhaustive test
对10个最相近的prompt，append每个vocab token，343 billion pairs per model。仍然no collision。这个实验的cost非常惊人。

### SIPIT结果
- GPT-2 Small上100% exact recovery
- 20 token的prompt，28秒（gradient-guided policy）
- vs HARDPROMPTS: 0% accuracy（这个方法本来是为CLIP设计的，没verifier机制）
- vs BRUTEFORCE: 100% accuracy但慢200倍

---

## 这意味着什么

### 对interpretability

如果你能access到last-layer hidden state，你能exact reconstruct输入。所以hidden state不是"压缩的representation"，而是input的另一个encoding。

这跟mechanistic interpretability的circuits-level analysis是complementary视角：circuits看的是feature层面的computation，injectivity看的是end-to-end的信息保持。

### 对privacy

Hamburg Data Protection Commissioner之前claim说weights不算personal data因为training examples不能trivially reconstruct（https://datenschutz-hamburg.de/fileadmin/user_upload/HmbBfDI/Datenschutz/Informationen/240715_Discussion_Paper_Hamburg_DPA_KI_Models.pdf）。

但这个paper说：**inference time的user input可以从hidden state完全recover**。任何存储或传输hidden states的系统，本质上就是在处理plaintext。这对RAG、embedding database、caching系统都有直接implication。

### 对model design的启示

Paper的failure cases很诚实：
- 两个vocab item有完全相同embedding → collision
- 两个positional embedding完全相同 → collision
- Quantization → 不再analytic
- Weight tying → 不再analytic
- ReLU activation → 在0点不解析（GELU是解析的）

这些都是"deliberately engineer才会出问题"的情况。Standard training pipeline下，a.s.不会出现。但如果你做quantization-aware training或者weight tying，理论上injectivity不再保证。

---

## 你的intuition应该怎么update

1. **不要把"feature层面的compression"和"input层面的loss"混淆**。Superposition（https://transformer-circuits.pub/2022/toy_model/index.html）在feature层面，injectivity在input层面，两者不矛盾。

2. **Real-analyticity是一个underexplored的数学工具**。"某性质是generic还是exceptional"的问题，只要能表示成analytic函数的零集，就能套用这个dichotomy。这对ML理论应该有更多应用。

3. **Witness构造是核心数学技巧**。要证明analytic函数不恒为零，只需要构造一个让它非零的点。Transformer的witness构造揭示了一个minimal structure：一个attention head + orthogonal embeddings + 三个正交方向就够。

4. **Discrete input是关键**。从finite set到 $\mathbb{R}^d$ 的injectivity要求比continuous到continuous弱得多。这就是为什么很多folk theorem的"lossy"结论在这里不适用。

5. **Invertibility是first-class property**。Hidden states是input的encoding，这个encoding可以被invert。这对privacy、interpretability、probing analysis都有直接含义。

---

## 一些我自己想到的open question

- **Multimodal extension**：vision transformer的patch embedding是continuous的（pixel values），不再是discrete input。这个framework可能需要扩展。
- **Approximate inversion under quantization noise**：理论保证 $\epsilon < \Delta/2$，但实际margin可能很小。需要研究quantization noise下的robust inversion bound。
- **Connection to compression theory**：如果hidden state是input的exact encoding，那information content应该等于input entropy。这跟"language model压缩互联网到weights"的narrative有什么关系？
- **Implication for context distillation**：如果hidden state可以invert回input，那"context distillation"（把long context压缩进fewer tokens）在理论上存在什么bottleneck？

---

**关键参考**：
- 原paper: 附件
- Mityagin zero set: https://arxiv.org/abs/1512.07276
- Sutter et al. pointwise injectivity: https://arxiv.org/abs/2507.08802
- Jiang & Haghtalab surjectivity: https://arxiv.org/abs/2508.19445
- Dong attention rank collapse: https://proceedings.mlr.press/v139/dong21a.html
- Yang softmax bottleneck: https://arxiv.org/abs/1711.03953
- Wen et al. Hard Prompts: https://arxiv.org/abs/2302.03668
- Morris et al. LM inversion: https://arxiv.org/abs/2311.13647
- Hamburg DPA: https://datenschutz-hamburg.de/fileadmin/user_upload/HmbBfDI/Datenschutz/Informationen/240715_Discussion_Paper_Hamburg_DPA_KI_Models.pdf
- Anthropic Superposition: https://transformer-circuits.pub/2022/toy_model/index.html

---

# LANGUAGE MODELS ARE INJECTIVE - 深度解读

Andrej，这篇paper相当有意思，它从real analysis的角度重新审视了我们一直以为很清楚的Transformer性质。让我把核心直觉和数学细节讲透。

## 1. 核心claim的重新表述

很多年以来，community的intuition是：Transformer的hidden states是"lossy"的压缩，因为LayerNorm会丢掉per-example statistics，attention rank会doubly-exponentially collapse（Dong et al. 2021, https://proceedings.mlr.press/v139/dong21a.html），softmax bottleneck限制了输出分布（Yang et al. 2018, https://arxiv.org/abs/1711.03953），residual connections可以cancel。

但这篇paper说：当你把Transformer看作从**离散prompt空间** $\mathcal{V}^{\leq K}$ 到 **连续表示空间** $\mathbb{R}^d$ 的映射时，在标准初始化和训练下，这个映射几乎必然是injective的。

关键的区别在于：之前人们讨论的是 $\mathbb{R}^d \to \mathbb{R}^d$ 这种continuous到continuous的map的non-injectivity；但这篇讨论的是discrete到continuous的map，而discrete domain的injectivity只需要在finite点集上不发生collision。

## 2. 真正的核心技术：Real-Analyticity

整个证明的核心是一个数学trick：**real-analytic函数的零集dichotomy**。

**Theorem A.1** (Mityagin 2015, https://arxiv.org/abs/1512.07276)：如果 $f \in C^\omega(\mathcal{U}; \mathbb{R}^n)$ 且 $f \not\equiv 0$，那么它的零集 $Z(f) = f^{-1}(\{0\})$ 的Lebesgue测度为零。

这里的直觉是：real-analytic函数（局部有收敛幂级数展开的函数）的零点不能任意聚集。任何零点都必须是isolated的（在一维情况下），或者在一个低维的analytic variety上（在多维情况下）。这跟 $C^\infty$ smooth函数完全不同——smooth函数的零集可以是任何closed set（通过bump function构造）。

**为什么这个对Transformer重要**：paper证明Transformer的每个组件都是real-analytic的：
- Polynomials (embeddings, projections): trivially real-analytic
- $\exp$ and softmax: real-analytic (Proposition A.5, A.7)
- $t \mapsto t^{-1/2}$ on $(0,\infty)$: real-analytic via binomial series
- LayerNorm with $\epsilon > 0$: $\sqrt{\sigma_x^2 + \epsilon}$ 的分母永远正，所以是real-analytic
- GELU, tanh activations: real-analytic
- 这些东西的composition、sum、product、quotient (分母非零) 都保持real-analytic

所以整个 $h(\theta) = \|r(s; \theta) - r(s'; \theta)\|_2^2$ 是real-analytic。

**Witness construction**: 为了证明 $h \not\equiv 0$，paper构造了一个特定的参数配置 $\theta_\star$，使得 $h(\theta_\star) > 0$。这里有两种case：

### Case A: 序列在最后位置或长度上不同
设置所有Transformer weights为0，网络退化成identity。然后：
- 如果 $s_{T_s} \neq t_{T_t}$: 设 $E_{s_{T_s}} = e_1$, $E_{t_{T_t}} = e_2 \neq e_1$, 其他embedding为0，$P = 0$。则 $r(s;\theta_\star) = e_1$, $r(t;\theta_\star) = e_2$, $h(\theta_\star) = \|e_1 - e_2\|^2 > 0$.
- 如果长度不同 $T_s \neq T_t$: 用positional embedding区分。

### Case B: 序列在中间位置不同，但长度和最后token相同
设 $i^\star$ 是第一个不同的位置。这个case更tricky，需要利用attention。

**关键构造**：选择三个正交向量 $e, p, q \in \mathbb{R}^d$，且都正交于 $\mathbf{1}_d$（这样LayerNorm的均值是0）。注意这要求 $d \geq 4$（这就是Assumption C.1的来源）。

Embedding设置：
- $E_v = e$ 当 $v \in \{s_{i^\star}, s_T\}$
- $P_j = p$ 当 $j = i^\star$, $P_j = q$ 当 $j = T$, 其他为0

这样position $i^\star$ 的输入是 $e + p$（对s）vs $p$（对t，因为t的token embedding是0），而最后位置的输入都是 $e + q$。

设attention head的参数 $Q = \alpha e e_1^\top$, $K = \beta p e_1^\top$, $V = e e_1^\top$。这样：
- query在最后位置是 $\alpha c_{ep} e_1$（两序列相同）
- key在 $i^\star$ 对s是 $\beta c_{ep} e_1$，对t是 $\beta c_e e_1$
- value在 $i^\star$ 对s是 $c_{ep} e_1$，对t是 $0$

通过设置 $\alpha\beta = \sqrt{d_\eta} L / c_{ep}^2$ 其中 $L = \log((1-\delta)(T-1)/\delta)$，让最后位置的attention几乎完全集中在 $i^\star$。

最终：
$$\langle y_T^{(s)} - y_T^{(t)}, e_1 \rangle \geq (1-\delta) c_{ep} - 2\delta c_e$$

选 $\delta < c_{ep}/(c_{ep} + 2c_e)$ 让这个严格正。这样head output在 $e_1$ 方向有差异，经过 $W^O$ 后传递到最后输出。

## 3. 为什么梯度下降保持injectivity

这是paper最technical的部分。直觉是：GD的update map $\phi(\theta) = \theta - \eta \nabla L(\theta)$ 是real-analytic的，它的Jacobian $D\phi(\theta) = I_p - \eta \nabla^2 L(\theta)$ 的determinant也是real-analytic的。

**关键一步：在witness点 $\theta_\star = 0$ 计算Hessian**

设 $\gamma = 0$（LN的scale），则LN输出恒等于 $\beta$，logits变成 $z = U\beta$，这是一个bilinear形式。

在 $(u, \beta) = (0, 0)$，Hessian是：
$$\nabla^2_{(u,\beta)} \mathcal{L}(\theta_\star) = \begin{pmatrix} 0 & I_d \otimes w \\ I_d \otimes w^\top & 0 \end{pmatrix}$$

其中 $w = \frac{1}{n}\mathbf{1}_n - p$ 是softmax(0)与target的差。

**Lemma C.3的谱计算**：通过 $H^2$ 的eigenvalues（注意 $H$ 对称），得到 $\text{spec}(\nabla^2 \mathcal{L}) = \{\pm\|w\|_2 \text{ each mult. } d, 0 \text{ mult. } p-2d\}$。

所以 $D\phi(\theta_\star) = I_p - \eta \nabla^2 \mathcal{L}(\theta_\star)$ 的eigenvalues是 $1 \pm \eta\|w\|$ 各 $d$ 个，加上 $p-2d$ 个 $1$。

$$\det D\phi(\theta_\star) = (1 - \eta^2 \|w\|^2)^d > 0$$

（这里需要 $\eta \|w\| < 1$，但paper假设 $\eta \in (0,1)$，并且这个inequality在witness点自动满足，因为w是softmax-1和target-p的差，其norm有限）

**为什么这重要**：$\det D\phi(\theta_\star) > 0$ 保证了 $\det D\phi$ 不是identically zero的real-analytic函数。因此它的零集 $\mathcal{C}$ 是measure zero（Theorem A.1）。

**Inverse Function Theorem的应用**：在 $\mathbb{R}^p \setminus \mathcal{C}$ 的每一点，$\phi$ 是local $C^1$ diffeomorphism。但这是uncountable cover，需要利用 $\mathbb{R}^p$ 的second-countability（Proposition A.15）抽取countable subcover（Lemma C.5）。

**Change of variables**：在每个chart上，$\psi_k$ 把null set映成null set（Theorem C.4, https://www.wiley.com/en-us/Real+Analysis-p-9780471317166）。加上critical set $\mathcal{C}$ 本身是null set，所以 $\phi^{-1}$ 把任何null set映成null set（Lemma C.6）。

**结论**：如果 $\mu \ll \text{Leb}_p$，则 $\phi_\# \mu \ll \text{Leb}_p$（Theorem C.5）。迭代有限步（Corollary C.5.1），参数分布保持绝对连续。

## 4. SIPIT算法

**核心idea**：Causal masking让position $t$ 的hidden state只依赖于prefix $\pi = s_{1:t-1}$ 和当前token $s_t$。所以可以顺序inversion。

**One-step map** (Definition D.1)：
$$F(v; \pi, t) := h_t(\pi \oplus v) \in \mathbb{R}^d$$

**Theorem D.1**：在Assumption D.2下（参数来自绝对连续分布），$F$ 几乎必然injective。

**Lemma D.1**：margin $\Delta_{\pi,t} = \min_{v \neq v'} \|F(v) - F(v')\|_2 > 0$ a.s.

**Algorithm 1 (SIPIT)**：对每个位置 $t$，遍历vocabulary，找到唯一使 $\hat{h}_t \in \mathcal{A}_{\pi,t}(v; \epsilon)$ 的token。

**Theorem D.2**：在 $\epsilon < \frac{1}{2}\Delta_{\pi_t, t}$ 条件下，SIPIT a.s.精确恢复原序列。

**Proposition D.4**：worst case总verifier tests数为 $T|\mathcal{V}|$，linear time。

## 5. 实验数据深度解读

### Collision search (Section 4.1)

数据集混合：wikipedia-en, C4, The Pile, python-github-code，100k prompts。

**Table 1 - Minimum pairwise L2 distances**:

| Model | Layer 1 | Layer L/2 | Layer L |
|-------|---------|-----------|---------|
| Llama-3.1-8B | 0.001 | 0.129 | 0.620 |
| Mistral-7B-v0.1 | 0.002 | 0.187 | 1.274 |
| Phi-4-mini-ins | 0.014 | 1.336 | 9.020 |
| TinyStories-33M | 0.029 | 1.434 | 2.793 |

几个值得注意的pattern：
1. **浅层就有清晰separation**：即使在Layer 1，最小距离 $10^{-3}$ 级别，远超collision threshold $10^{-6}$。
2. **深度增加separation**：Layer L的distance通常比Layer 1大几百倍。这与理论一致——deeper composition of analytic maps在generic parameter下tends to separate points further。
3. **更小的模型（TinyStories-33M）反而有更大的layer-1 distance**：0.029 vs Llama的0.001。这可能是因为小模型vocabulary小、representation空间使用更"满"。

### Exhaustive collision test

对10个最相近的prompt，append每个vocab token，做343 billion pairs的exhaustive search。仍然no collisions。这个实验的cost是惊人的——343B pairs $\times$ 4个模型。

### SIPIT vs baselines

**Table 2**:

| Method | Mean Time (s) | Accuracy |
|--------|---------------|----------|
| HARDPROMPTS | 6132.59 ± 104.61 | 0.00 |
| BRUTEFORCE | 3889.61 ± 691.17 | 1.00 |
| SIPIT (ours) | **28.01 ± 35.87** | **1.00** |

- HARDPROMPTS (Wen et al. 2023, https://arxiv.org/abs/2302.03668): 0% accuracy。这个方法本来是为CLIP设计的，迁移到LM上没有verifier机制。
- BRUTEFORCE: 100% accuracy但慢200倍。说明verifier本身是对的，问题是policy效率。
- SIPIT: 100% accuracy，gradient-guided policy让时间从~4000s降到~28s，超过100x加速。

**Figure 6的细节**：Inversion time vs layer depth。从layer 1到layer L，time只增加很温和。这是因为虽然deeper layer的forward cost更大，但deeper layer的separation更好，需要更少的candidate proposal。

## 6. 给你的intuition

Andrej，我觉得这篇paper对你build intuition最重要的几点：

### (1) Discrete vs Continuous injectivity
不要混淆两个不同的injectivity问题：
- **Continuous $\to$ Continuous**：LayerNorm $x \mapsto \gamma \cdot (x - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$ 确实是non-injective（沿 $\mathbf{1}_d$ 方向被collapse）。
- **Discrete $\to$ Continuous**：在finite vocabulary和finite context length下，输入空间是finite set。Injectivity只需要在这个finite set上没有collision，这比continuous的injectivity要求弱得多。

Paper证明的是后者。Real-analyticity + measure-zero argument给出generic property。

### (2) Real-analyticity的power
关键数学fact：real-analytic函数的零集要么是整个空间，要么是measure zero。这是一个**dichotomy**——没有中间地带。

要利用它，只需要：
- 证明你的函数是real-analytic（Transformer是）
- 构造一个witness使得函数不在那点为零（证明就完成）

这个pattern在ML理论里其实underexplored。它对任何"集合是generic还是exceptional"的问题都适用，只要集合可以表示为某个analytic函数的零集。

### (3) Witness construction的精妙
Case B的构造特别值得品味。它需要：
- 三个正交向量（要求 $d \geq 4$，所以Assumption C.1）
- 让attention几乎完全聚焦在第一个不同的position
- 通过 $V$ 矩阵把那个位置的token信息copy到output

这是"什么样的minimal structure能让两个序列区分开"的clean illustration。本质上说：**只要有一个attention head能attend到第一个不同的position，并且通过value matrix把那个token的信息传到最后，injectivity就保证了**。

### (4) Privacy含义的严肃性
这paper的Discussion部分提到Hamburg Data Protection Commissioner的claim（https://datenschutz-hamburg.de/fileadmin/user_upload/HmbBfDI/Datenschutz/Informationen/240715_Discussion_Paper_Hamburg_DPA_KI_Models.pdf）说weights不算personal data因为training examples不能trivially reconstruct。

但这个paper说：**inference time的user input可以从hidden states完全recover**。所以任何存储或传输hidden states的系统，本质上就是在处理plaintext。这对RAG系统、embedding database、caching系统都有直接implication。

### (5) Limitations要诚实面对

Paper的failure cases部分写得很谨慎：
- 两个vocab item有完全相同的embedding → collision
- 两个position embedding完全相同 + 其他weights suppress positional signal → collision
- Quantization → 不再real-analytic
- Weight tying → 不再real-analytic（embedding与unembedding tied）
- ReLU activation → 在0点不解析（虽然GELU是解析的）

这些都是"如果deliberately engineer才会出问题"的情况。在standard training pipeline下，a.s.不会出现。

## 7. 一些可能的extension方向

- **Multimodal Transformers**：vision transformer的patch embedding是continuous的，不再是discrete input。这个framework可能需要扩展。
- **Approximate inversion under noise**：理论保证 $\epsilon < \Delta/2$，但实际margin可能很小。需要研究quantization noise下的robust inversion。
- **Connection to mechanistic interpretability**：如果last-token state faithfully encodes输入，那么superposition hypothesis（https://transformer-circuits.pub/2022/toy_model/index.html）和injectivity的关系是什么？Superposition是在feature层面，injectivity是在input层面，两者并不矛盾。

## 8. 总结

这篇paper的核心贡献是用real-analyticity这个工具把"Transformer是lossy的"这个folk belief替换成了一个rigorous theorem。证明的关键ingredient：

1. Transformer是real-analytic（每个component保持解析性）
2. Collision set是analytic函数的零集，要么满要么measure zero
3. Witness构造证明零集不是满 → measure zero
4. 连续初始化避开measure zero set
5. GD的Jacobian determinant非trivial → a.s. nondegenerate → 保持绝对连续
6. Causal masking + one-step injectivity → SIPIT算法

实验在6个SOTA模型上做了billions of comparisons，no collisions observed。SIPIT在GPT-2 Small上100% exact recovery。

这个工作对interpretability的意义是：**如果你能access到last-layer hidden state，你能exact reconstruct输入**。所以hidden states不是"压缩"的representation，而是input的另一个encoding。这跟 mechanistic interpretability的circuits-level analysis是complementary的视角。

---

**关键参考链接**：
- Paper PDF: 附件提供
- Mityagin zero set theorem: https://arxiv.org/abs/1512.07276
- Sutter et al. (pointwise injectivity at init): https://arxiv.org/abs/2507.08802
- Jiang & Haghtalab (surjectivity): https://arxiv.org/abs/2508.19445
- Dong et al. (attention rank collapse): https://proceedings.mlr.press/v139/dong21a.html
- Yang et al. (softmax bottleneck): https://arxiv.org/abs/1711.03953
- Wen et al. (Hard Prompts): https://arxiv.org/abs/2302.03668
- Morris et al. (LM inversion): https://arxiv.org/abs/2311.13647
- Hamburg DPA discussion: https://datenschutz-hamburg.de/fileadmin/user_upload/HmbBfDI/Datenschutz/Informationen/240715_Discussion_Paper_Hamburg_DPA_KI_Models.pdf
- Anthropic Toy Models of Superposition: https://transformer-circuits.pub/2022/toy_model/index.html
