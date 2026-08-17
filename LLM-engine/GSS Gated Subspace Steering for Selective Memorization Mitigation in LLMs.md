---
source_pdf: GSS Gated Subspace Steering for Selective Memorization Mitigation in LLMs.pdf
paper_sha256: 7bac61af5f13126103ec6d43695bc6097517d30733d5dfc17bdf0842360c35da
processed_at: '2026-08-04T23:11:03-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GSS 用人话讲：一个"探测器+纠偏器"的故事

Karpathy，我把这篇paper的story重新用更直觉的方式讲一遍。核心就一个idea：**memorization是稀疏的、突发的，so you need a smart detector that only fires when needed**。

---

## The Story: 为什么要搞这个？

LLM fine-tune之后有个尴尬的problem——它会背training data。verbatim reproduction。这带来privacy leak、copyright问题、还hurting generalization。

现有approaches都太暴力。举几个例子：

**Unlearning methods (NPO, GA)**: 你告诉模型"忘记这些data"，它通过gradient descent改参数。problem是——它uniformly地damage everything。Table 2里NPO在TinyMem math上把memorization降到0.56%，but accuracy也从97%暴跌到27.26%。Catastrophic forgetting。

**Knowledge Editing (ROME, MEMIT)**: 你locate某个fact在哪个neuron，然后改那个neuron的weights。problem是——memorization根本就不是localized到几个neuron的fact，它是token-level的behavior。

**Activation Steering (CAA)**: 你找一个"memorization direction"，然后从hidden state里减掉这个direction。这更像"surgical"。but prior work用**同一个direction同时做detection和correction**——对每个token都减同样的vector。问题是大部分token根本没在memorize，你减它们干嘛？

GSS的insight其实很简单：**detect和correct应该是两回事**。你先看这个token是不是在背答案，如果是，再correct它。如果不是，leave it alone。

---

## The Key Observation: Memorization是Sparse的

作者fine-tune Qwen3 on GSM8K，然后measure每个token的"memorization signal":

$$\omega_t = \log p_\theta(x_t | x_{<t}) - \log p_{\text{ref}}(x_t | x_{<t})$$

人话翻译：fine-tuned model说这个token的概率是$p_\theta$，reference model说是$p_{\text{ref}}$，取log之后相减。

如果$\omega_t > 0$，说明fine-tuned model比reference model更"确信"这个token——这是memorization的signal。reference model通常是pre-trained base model，它没见过fine-tune data，so如果fine-tuned model突然变得super confident，很可能是因为它"背过"了。

$\omega_t$的distribution揭示了两个事实：

**Fact 1: Heavy-tailed distribution**。只有18.7%的tokens的$\omega_t$超过mean。memorization集中在少数high-magnitude tokens上。大部分tokens其实是在generalize。

**Fact 2: Burst length很短**。mean burst length只有1.64 tokens，60.8%的bursts是single token。即使在一个verbatim memorized sequence里，model也是在memorize和generalize之间频繁交替——背几个token，然后自己generate几个token，再背几个。

This is the "aha moment"——memorization不是sequence-level的property，是token-level的sparse event。你用uniform intervention就像用散弹枪打蚊子——collateral damage巨大。

---

## The Framework: Probe + Steer, Decoupled

### 4.1 The Basic Idea

Standard activation steering是:
$$h' = h - \alpha \nu$$

每个token都减同样的vector $\nu$。stupid。

GSS加了个gate:
$$h' = h - \mathcal{G}(|u^\top h| > \epsilon) \cdot \nu$$

人话：先算hidden state $h$在direction $u$上的projection $u^\top h$。如果这个projection的绝对值超过threshold $\epsilon$，才减$\nu$。otherwise，$h$不变。

这里两个direction是decoupled的：
- $u$是**probe**——探测器，检测"这个token是不是在memorize"
- $\nu$是**steer**——纠偏器，把memorization component移除

为什么必须decouple？因为detection和correction的optimal direction可能完全不同。detection需要high signal-to-noise ratio on memorization tokens，correction需要maximally reduce memorization loss。如果你force它们一样，就suboptimal了。

### 4.2 Multi-Rank Extension

一个direction可能不够，so扩展到rank-K:
$$\Delta h = -\sum_{k=1}^K (u_k^\top h) \nu_k$$

每个$(u_k, \nu_k)$ pair是一个independent的detector-corrector mode。

---

## The Math: 为什么是SVD?

This is where it gets interesting。作者把"找最优probe-steer pair"formulate成一个constrained optimization。

### 4.3 Memorization Alignment Objective

第一个requirement：intervention在memorized tokens上要maximally effective。

对memorized token set $\mathcal{D}_{\text{mem}}$，给activation $h$加perturbation $\Delta h$，memorization loss的一阶Taylor展开:
$$\Delta \mathcal{L}_{\text{mem}} \approx g^\top \Delta h$$

其中$g = \nabla_h \mathcal{L}_{\text{mem}}$是loss对activation的gradient。

考虑到probe触发intervention的条件$|u^\top h| > \epsilon$，expected loss reduction是:
$$\mathbb{E}_{h \sim \mathcal{D}_{\text{mem}}}[(u^\top h)(g^\top \nu)] = u^\top \mathbf{M} \nu$$

这里$\mathbf{M} = \mathbb{E}_{h \sim \mathcal{D}_{\text{mem}}}[h g^\top]$是**memorization matrix**。它captures"activation perturbation怎么translate到loss reduction"。

人话：memorization matrix告诉你，如果你push activation在direction $h$，loss会怎么变。$g$是gradient direction，$h$是activation direction，$hg^\top$是outer product——它是一个$d \times d$的matrix，captures所有"activation perturbation → loss change"的二阶interaction。

### 4.4 Generalization Safety Constraint

第二个requirement：probe在generalized tokens上要mostly inactive。

用covariance $\Sigma_{\text{gen}} = \mathbb{E}_{h \sim \mathcal{D}_{\text{gen}}}[(h-\mu)(h-\mu)^\top]$描述generalization manifold的geometry。

Constraint: $u^\top \Sigma_{\text{gen}} u \leq \delta$

这是$u$在$\Sigma_{\text{gen}}$下的Mahalanobis norm。通过Chebyshev's inequality，这保证probe最多在$\delta$-fraction的generalized tokens上activate。

人话：如果你在"正常"tokens的distribution里投影$u$，variance不能太大。不然你会在正常tokens上误触发。

### 4.5 The Optimization Problem

$$\max_{\{u_k, \nu_k\}} \sum_{k=1}^K u_k^\top \mathbf{M} \nu_k$$
$$\text{s.t.} \quad u_k^\top \Sigma_{\text{gen}} u_k \leq \delta, \quad \|\nu_k\|_2 = 1$$

### 4.6 The Magic: Whitening + SVD

这是paper最beautiful的部分。Constraint是ellipsoidal的（$u^\top \Sigma_{\text{gen}} u \leq \delta$），不好处理。但是用Cholesky decomposition $\Sigma_{\text{gen}} = LL^\top$，做一个whitening transform:
$$\tilde{u} = L^\top u \iff u = L^{-\top} \tilde{u}$$

Constraint变成:
$$u^\top \Sigma_{\text{gen}} u = \tilde{u}^\top \tilde{u} = \|\tilde{u}\|_2^2 \leq \delta$$

漂亮！ellipsoidal constraint变成isotropic ball constraint。

Objective变成:
$$\tilde{u}^\top (L^{-1}\mathbf{M}) \nu = \tilde{u}^\top \mathbf{M}_{\text{op}} \nu$$

定义**whitened memorization matrix** $\mathbf{M}_{\text{op}} = L^{-1}\mathbf{M}$。

这就是标准的bilinear form maximization，solution是SVD:
$$u^* = \sqrt{\delta} \cdot L^{-\top} \tilde{u}_1, \quad \nu^* = \tilde{\nu}_1$$

其中$\tilde{u}_1, \tilde{\nu}_1$是$\mathbf{M}_{\text{op}}$的leading left和right singular vectors。

### 4.7 Geometric Intuition

为什么需要whitening？因为LLM的hidden state distribution是anisotropic的——"narrow cone"现象。少数principal components占绝大部分variance，而memorization signal往往aligns with low-variance axes。

如果直接在original space用hard threshold $|u^\top h| > \epsilon$，threshold会predominantly trigger在high-variance axes的noise上——false positive rate极高。

Whitening by $L^{-1}$把covariance equalize成$\approx I$。所有dimensions的energy被"拉平"。然后SVD在whitened space找到的方向$U^*$确保threshold对应的是Mahalanobis distance，each dimension被公平对待。

Figure 7的visualization很说明问题：whitening前，latent cloud高度elongated，gating region被high-variance noise淹没；whitening后变成spherical geometry，high-memorization tokens清晰align到extracted axis上。

---

## Adaptive Coefficient: 每个token的steering strength不同

Steering strength $\alpha$也是adaptive的。要求intervention后probe response不超过safe margin $\epsilon$:
$$\langle h', u \rangle \leq \epsilon$$

代入$h' = h - \alpha \nu$:
$$\langle h, u \rangle - \alpha \langle \nu, u \rangle \leq \epsilon$$

Instead of取boundary solution，用Tikhonov regularization求稳定解:
$$\alpha^* = \arg\min_\alpha (\langle h, u \rangle - \alpha \langle \nu, u \rangle - \epsilon)^2 + \delta \alpha^2$$

求导置零:
$$\alpha_k = \frac{\langle u_k, \nu_k \rangle}{\langle u_k, \nu_k \rangle^2 + \delta}$$

$\delta > 0$防止$\langle \nu, u \rangle$很小时的不稳定。

最终inference-time formula:
$$h' = h - \alpha_k \sum_{k=1}^K \mathbb{I}(|u_k^{*\top} h| > \epsilon_k) \nu_k^*$$

$\epsilon_k$设为validation set上$|u_k^\top h|$的95th percentile。

---

## 与LoRA的Connection: Dynamic LoRA

Without gating:
$$h' = h - (h^\top \nu) u = (I - u\nu^\top) h$$

如果在weight matrix $W$前apply:
$$Wh' = (W - Wu\nu^\top)h = (W + \Delta W)h$$

其中$\Delta W = -Wu\nu^\top$是rank-1 update。这就是LoRA的形式！

With gating:
$$\Delta W_t = -\mathcal{G}(\langle h_t, \nu \rangle) \cdot Wu\nu^\top$$

This is **dynamic, context-dependent rank-1 LoRA**——每个token的effective weight update不同。当$\langle h_t, \nu \rangle$大（memorization signal）时，update强activate；小时deactivate。

Static LoRA无法区分memorized和generalized tokens——uniformly apply同一个modification。GSS实现selective intervention without degrading normal inference。

---

## Experimental Results: 数据说话

### TinyMem (Table 2)

| Setting | Method | %Mem ↓ | Acc ↑ | Time ↓ |
|---------|--------|--------|-------|--------|
| Math-Noise | GSS | 0.00 | 96.98 | 0.001s |
| Math-Backdoor | GSS | 0.00 | 96.82 | 0.001s |
| Lang-Noise | GSS | 0.00 | 63.13 | 0.002s |
| Lang-Backdoor | GSS | 0.00 | 63.17 | 0.003s |
| Math-Noise | NPO | 0.56 | 27.26 | 0.57s |
| Math-Noise | HC | 0.00 | 74.97 | 0.24s |
| Math-Noise | AlphaSteer | 24.30 | 96.98 | 6.35s |

GSS是唯一同时实现zero memorization + near-baseline accuracy的方法。而且runtime是0.001-0.003s级别，比其他方法快100-1000倍。

### Pythia Scaling (Table 3)

| Model | Method | %Mem ↓ | PPL ↓ | Time ↓ |
|-------|--------|--------|-------|--------|
| Pythia 2.8B | Baseline | 52.87 | 21.75 | - |
| Pythia 2.8B | GSS | 6.93 | 28.26 | 0.16s |
| Pythia 6.9B | Baseline | 89.31 | 19.46 | - |
| Pythia 6.9B | GSS | 6.96 | 29.15 | 0.21s |
| Pythia 6.9B | Durable-agg | 10.98 | 34.40 | 320.60s |
| Pythia 6.9B | BalancedSub | 86.73 | 17.15 | 233.42s |

从89.31%降到6.96% memorization，PPL trade-off合理（19.46→29.15）。Time overhead 0.21s vs其他方法的300s级别。

### Pareto Frontier (Figure 4)

在GSM8K reasoning task上，GSS在memorization-utility trade-off上dominate所有baselines。更surprising的是，在moderate steering strength时观察到**utility boost (> 1.0)**——memorization在reasoning task上相当于overfitting noise，projecting it out反而recover了latent generalization capability。

### Rank Ablation (Figure 6)

Lower ranks ($K \leq 5$)在high-utility regime明显outperform higher ranks。Confirms memorization是low-dimensional的——增加$K$会capture spurious directions反而degrade generalization。

### Reference Model Sensitivity (Appendix H)

Clean pre-trained reference始终把memorization suppress到0。Partially trained reference表现non-monotonic——initially mitigation减弱，later partial recovery。Reference capacity matters：4-layer reference在late epochs出现degenerate cases，8-layer reference更stable。Clean accuracy不变across all configurations。

---

## Computational Complexity

### Offline Calibration
- Statistics estimation: $O(Nd^2)$——一次性的
- Cholesky + SVD: $O(d^3)$——$d=4096$时只需几秒

### Online Inference
- Projection: $O(Kd)$——$K$个inner products
- Gated accumulation: $O(Kd)$——$K$个scalar-vector multiplications

相对于Transformer layer的$O(d^2)$，overhead是$O(K/d) \approx 1.2\%$（$K=50, d=4096$）。Memory footprint小于1MB VRAM。

---

## 我的Intuition总结

GSS的elegance在于把复杂问题reduce成三个clean operations:

1. **Detection**: $\omega_t$告诉你哪些tokens在memorize
2. **Geometry**: Whitening by $\Sigma_{\text{gen}}^{-1/2}$处理anisotropic geometry
3. **Correction**: SVD of whitened memorization matrix给出optimal probe-steer pairs

The core insight: **memorization在activation space中是low-dimensional, sparse, token-conditional subspace**。GSS用gating只在memorization signal超过threshold时activate correction，避免static方法的collateral damage。

This is like sparse coding——找到数据中"少数但重要"的components，在inference时conditional处理。GSS把这个idea apply到LLM activation space，用information-theoretic memorization signal作为sparsity prior。

The decoupling principle is the key takeaway: **detect when**和**decide how**应该是两个independent optimizations。这个pattern在safety filtering、hallucination detection、甚至future MoE routing里都有potential。

---

## Reference Links

- [GSS Paper](https://arxiv.org/abs/2507.05578)
- [Extracting Training Data from LLMs (Carlini et al., 2021)](https://arxiv.org/abs/2012.07805)
- [Contrastive Activation Addition (Rimsky et al., 2024)](https://aclanthology.org/2024.acl-long.857/)
- [TinyMem Benchmark (Sakarvadia et al., 2024)](https://arxiv.org/abs/2410.02159)
- [NPO (Zhang et al., 2024)](https://arxiv.org/abs/2404.05868)
- [ROME (Meng et al., 2022)](https://arxiv.org/abs/2202.05262)
- [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685)
- [AlphaSteer (Sheng et al., 2025)](https://arxiv.org/abs/2506.07022)
- [LUNAR (Shen et al., 2025)](https://arxiv.org/abs/2410.11334)
- [Pythia Suite](https://arxiv.org/abs/2304.01373)
- [The Geometry of Truth (Marks & Tegmark)](https://arxiv.org/abs/2310.06824)

---

# GSS: Gated Subspace Steering 深度解析

Karpathy, 这篇paper的核心intuition其实非常优雅——它把memorization mitigation问题重新formulate成一个**token-level的sparse detection + conditional correction**问题，并通过一个principled的optimization framework给出了closed-form solution。让我一层层拆解。

---

## 1. Motivation: Memorization的Geometric性质

### 1.1 Token-level Memorization Signal

作者首先定义了一个token-level的memorization signal:

$$\omega_t \triangleq \log p_\theta(x_t \mid x_{<t}) - \log p_{\text{ref}}(x_t \mid x_{<t})$$

变量含义：
- $\omega_t$: 在token position $t$ 的memorization signal
- $p_\theta$: fine-tuned model的conditional probability distribution
- $p_{\text{ref}}$: reference model（通常是pre-trained base model）的conditional probability
- $x_t$: sequence中的第 $t$ 个token
- $x_{<t}$: position $t$ 之前的context tokens

这个定义的intuition来自information theory。原始的memorization measure是mutual information:
$$\text{Mem}(X) = I(X; \Theta \mid \Theta_{\text{ref}}) = H(X \mid \Theta_{\text{ref}}) - H(X \mid \Theta)$$

通过Shannon source coding theorem的interpretation，$-\log p(x)$ 是model-based code length。因此 $\omega_t$ 本质上就是**excess code length**——fine-tuned model比reference model在某个token上多消耗了多少bits。如果 $\omega_t > 0$，说明fine-tuned model比reference对这个token更"确信"，这通常是memorization的signal。

### 1.2 Empirical Observations: Sparsity & Intermittency

在Qwen3 fine-tuned on GSM8K上的实验揭示了两个key observations:

**Observation 1 (Temporal Sparsity)**: $\omega_t$ 的分布是heavy-tailed的，只有18.7%的tokens超过均值。这告诉我们memorization集中在少数high-magnitude tokens上。

**Observation 2 (Token-level Intermittency)**: Memorization bursts的平均长度只有1.64 tokens，60.8%的bursts是single token。即使在verbatim memorized sequences中，model也在memorization和generalization之间频繁交替。

这两个observations直接motivate了作者的设计选择：**intervention必须是token-conditional的**，uniformly的static intervention会产生大量collateral damage。

---

## 2. GSS Framework: Decouple Probe & Steer

### 2.1 核心架构

标准activation steering的形式:
$$h' = h - \alpha \nu$$

这uniformly地对所有token施加同一个intervention。GSS的核心创新是引入gating:

$$h' = h - \mathcal{G}(|u^\top h| > \epsilon) \cdot \nu$$

变量含义：
- $h \in \mathbb{R}^d$: 某layer的hidden state activation
- $h'$: intervention后的hidden state
- $u \in \mathbb{R}^d$: **probe direction**，用于检测memorization-relevant activations
- $\nu \in \mathbb{R}^d$: **steer direction**，correction的方向
- $\epsilon$: gating threshold，calibration后决定何时触发intervention
- $\mathcal{G}(\cdot)$: indicator gating function

**关键insight**: probe和steer是decoupled的！这与prior steering方法（如Contrastive Activation Addition, CAA）形成对比——prior work用同一个direction同时做detection和correction，这在memorization sparse的情况下会导致over-correction。

### 2.2 Rank-K扩展

对于多方向的correction:
$$\Delta h = -\sum_{k=1}^{K} (u_k^\top h) \nu_k, \quad k = 1, \ldots, K$$

每个 $(u_k, \nu_k)$ pair是independent的mode-wise detector-corrector。Scalar $u_k^\top h$ 作为第 $k$ 个mode的detection signal。

---

## 3. Principled Optimization: 为什么是SVD?

### 3.1 Memorization Alignment Objective

第一个requirement: intervention在memorization-prone tokens上要maximally effective。

给定memorized token set $\mathcal{D}_{\text{mem}}$，对activation $h$ 加一个perturbation $\Delta h$，memorization loss的一阶变化是:
$$\Delta \mathcal{L}_{\text{mem}} \approx g^\top \Delta h$$

其中 $g = \nabla_h \mathcal{L}_{\text{mem}}$ 是memorization loss的gradient。

Accounting for probe magnitude触发intervention的条件，objective变成:
$$\mathbb{E}_{h \sim \mathcal{D}_{\text{mem}}}[(u^\top h)(g^\top \nu)] = u^\top \mathbf{M} \nu$$

这里 $\mathbf{M} = \mathbb{E}_{h \sim \mathcal{D}_{\text{mem}}}[h g^\top]$ 是**memorization matrix**——它captures activation perturbations如何translate到memorization loss reduction。

### 3.2 Generalization Safety Constraint

第二个requirement: probe在generalized tokens上应该mostly inactive。

Model generalized activations的geometry通过covariance:
$$\Sigma_{\text{gen}} = \mathbb{E}_{h \sim \mathcal{D}_{\text{gen}}}[(h - \mu)(h - \mu)^\top]$$

Constraint: $u^\top \Sigma_{\text{gen}} u \leq \delta$

这通过Chebyshev's inequality保证probe最多在 $\delta$-fraction的generalized tokens上activate。

### 3.3 Constrained Optimization Problem

完整的rank-K optimization:
$$\max_{\{u_k, \nu_k\}_{k=1}^K} \sum_{k=1}^K u_k^\top \mathbf{M} \nu_k$$
$$\text{s.t.} \quad u_k^\top \Sigma_{\text{gen}} u_k \leq \delta, \quad \|\nu_k\|_2 = 1, \quad k = 1, \ldots, K$$

### 3.4 Theorem 4.1: Whitening + SVD的Closed-Form Solution

这是paper最beautiful的部分。通过Cholesky decomposition $\Sigma_{\text{gen}} = LL^\top$（$L$ 是lower triangular matrix），做变量替换:
$$\tilde{u} = L^\top u \iff u = L^{-\top} \tilde{u}$$

Constraint变成isotropic:
$$u^\top \Sigma_{\text{gen}} u = \tilde{u}^\top \tilde{u} = \|\tilde{u}\|_2^2 \leq \delta$$

Objective变成:
$$\tilde{u}^\top \underbrace{(L^{-1}\mathbf{M})}_{\mathbf{M}_{\text{op}}} \nu$$

定义**whitened memorization matrix**: $\mathbf{M}_{\text{op}} = L^{-1}\mathbf{M}$

这是标准的bilinear form maximization，solution是SVD:
$$u^* = \sqrt{\delta} \cdot L^{-\top} \tilde{u}_1, \quad \nu^* = \tilde{\nu}_1$$

其中 $\tilde{u}_1, \tilde{\nu}_1$ 是 $\mathbf{M}_{\text{op}}$ 的leading left和right singular vectors。

**Geometric intuition**: 
- $u^*$ 并非简单是 $\mathbf{M}$ 的top singular vector
- 而是经过 $\Sigma_{\text{gen}}$ whitening后，在memorization-relevant direction上per unit generalization variance的最佳sensitivity
- Whitening by $L^{-1}$ 把anisotropic的generalization manifold"拉直"成isotropic sphere
- SVD在这个whitened space找到memorization最concentrated的方向

---

## 4. Adaptive Gating Coefficient

### 4.1 Tikhonov-Regularized Derivation

Steering strength $\alpha$ 不是固定的，而是adaptively derived。要求intervention后probe response不超过safe margin $\epsilon$:
$$\langle h', u \rangle \leq \epsilon$$

代入 $h' = h - \alpha \nu$:
$$\langle h, u \rangle - \alpha \langle \nu, u \rangle \leq \epsilon$$

不是直接取boundary solution，而是minimize Tikhonov-regularized residual:
$$\alpha^* = \arg\min_{\alpha} (\langle h, u \rangle - \alpha \langle \nu, u \rangle - \epsilon)^2 + \delta \alpha^2$$

求导置零:
$$\alpha_k = \frac{\langle u_k^*, \nu_k^* \rangle}{\langle u_k^*, \nu_k^* \rangle^2 + \delta}$$

这里 $\delta > 0$ 防止 $\langle \nu, u \rangle$ 很小时的不稳定。

### 4.2 最终Inference-Time Formula

$$h' = h - \alpha_k \sum_{k=1}^K \mathbb{I}(|u_k^{*\top} h| > \epsilon_k) \nu_k^*$$

这里 $\epsilon_k$ 设为validation set上 $|u_k^\top h|$ 的95th percentile。

---

## 5. 与LoRA的Connection

### 5.1 Static Version = Rank-1 LoRA

不加gating的话:
$$h' = h - (h^\top \nu) u = (I - u\nu^\top) h$$

如果在weight matrix $W$ 之前apply:
$$Wh' = W(I - u\nu^\top)h = (W - Wu\nu^\top)h = (W + \Delta W)h$$

其中 $\Delta W = -Wu\nu^\top$ 是rank-1 update。

### 5.2 Dynamic LoRA

加上gating后:
$$\Delta W_t = -\mathcal{G}(\langle h_t, \nu \rangle) \cdot Wu\nu^\top$$

这是**context-dependent rank-1 update**，每个token的effective $\Delta W$ 都不同。当 $\langle h_t, \nu \rangle$ 大（memorization signal）时，update强烈activate；小时deactivate。

这比static LoRA好在哪？Static LoRA无法区分memorized和generalized tokens——uniformly apply同一个modification。GSS实现selective intervention without degrading normal inference。

---

## 6. Geometric Intuition: 为什么需要Whitening?

### 6.1 Anisotropic Latent Geometry

LLMs的hidden state分布呈现"narrow cone"特性——少数principal components占据绝大部分variance。Memorization signal往往aligns with low-variance axes（minor components）。

如果直接在original space用hard threshold $|u^\top h| > \epsilon$，SNR是:
$$\text{SNR}_{\text{raw}} = \frac{u^\top \mathbf{M} u}{u^\top \Sigma_{\text{gen}} u}$$

在anisotropic setting（$\Sigma_{\text{gen}}$ 的eigenvalues跨越几个order of magnitude），hard threshold会predominantly trigger在high-variance axes的noise上，false positive率极高。

### 6.2 Whitening = SNR Optimization

通过 $L^{-1}$ 变换，$h_{\text{white}} = L^{-1} h$，covariance变成 $\approx I$。所有dimensions的energy被"equalize"。

SVD在whitened space提取的steering basis $U^*$ 确保gating threshold对应的是Mahalanobis distance而非raw Euclidean magnitude。Figure 7的visualization很说明问题：unwhitened时latent cloud高度elongated，gating region被high-variance axis的noise淹没；whitening后变成spherical geometry，high-$\omega$ points清晰align到extracted steering axis上。

---

## 7. 实验数据深度解析

### 7.1 TinyMem Results (Table 2)

| Setting | Method | %Mem ↓ | Acc ↑ | Time ↓ |
|---------|--------|--------|-------|--------|
| Math-Noise | GSS | 0.00 | 96.98 | 0.001s |
| Math-Backdoor | GSS | 0.00 | 96.82 | 0.001s |
| Lang-Noise | GSS | 0.00 | 63.13 | 0.002s |
| Lang-Backdoor | GSS | 0.00 | 63.17 | 0.003s |

对比baselines:
- **NPO**: 0.56% mem但accuracy只有27.26%——catastrophic forgetting
- **HC/Slim/Act**: 能reduce memorization但accuracy严重drop（60-75%）
- **AlphaSteer/Lunar**: 24.3% mem还在，但accuracy保持

GSS的**独特优势**: 完全eliminate memorization同时保持near-baseline accuracy，且inference overhead极小（0.001-0.003s vs 其他方法的seconds级别）。

### 7.2 Pythia Scaling (Table 3)

| Model | Method | %Mem ↓ | PPL ↓ | Time ↓ |
|-------|--------|--------|-------|--------|
| Pythia 2.8B | Baseline | 52.87 | 21.75 | - |
| Pythia 2.8B | GSS | 6.93 | 28.26 | 0.16s |
| Pythia 6.9B | Baseline | 89.31 | 19.46 | - |
| Pythia 6.9B | GSS | 6.96 | 29.15 | 0.21s |

从89.31%降到6.96%的memorization，PPL从19.46升到29.15——trade-off合理。Time overhead只有0.21s，相对于其他方法（Durable-agg 320.60s, BalancedSub 233.42s）快了1000倍量级。

### 7.3 Pareto Frontier Analysis (Figure 4)

在GSM8K reasoning task上，GSS不仅dominate frontier，还在moderate steering strength时观察到**utility boost (> 1.0)**。这说明memorization在reasoning task上相当于overfitting noise，通过gated subspace projection反而recover了latent generalization capability。

### 7.4 Rank Ablation (Figure 6)

Lower ranks ($K \leq 5$) 在high-utility regime明显outperform higher ranks ($K \geq 5$)。这confirms memorization是low-dimensional的——增加 $K$ 会capture spurious directions反而degrade generalization。

### 7.5 Deviation Score (Tables 7-8)

作者定义了composite metric:
$$\text{DS}_\lambda = 100\sqrt{m^2 + \lambda \cdot \text{PPLDeg}^2}$$

其中 $m = \% \text{Mem}/100$，$\text{PPLDeg} = \max(0, \text{PPL}/\text{PPL}_{\text{base}} - 1)$。

在Pythia 6.9B上，GSS的 $\text{DS}_{0.05} = 13.13$，是所有方法中最低的（BalancedSub 86.73, Durable-agg 20.38, Greedy 17.42）。

---

## 8. Computational Complexity

### 8.1 Offline Calibration

- **Statistics estimation**: $O(Nd^2)$ —— $N$ 是calibration corpus tokens数，$d$ 是hidden dimension
- **Cholesky decomposition**: $O(d^3)$
- **SVD of whitened matrix**: $O(d^3)$

对于 $d = 4096$ 的标准LLM，这些 $O(d^3)$ 操作只需几秒。

### 8.2 Online Inference

每个token的overhead:
- **Projection**: $O(Kd)$ —— $K$ 个inner products
- **Gated accumulation**: $O(Kd)$ —— $K$ 个scalar-vector multiplications

相对于Transformer layer的 $O(d^2)$:
$$\Gamma = \frac{O(Kd)}{O(d^2)} = O(K/d)$$

实际 $K = 50, d = 4096$ 时，$\Gamma \approx 1.2\%$。Memory footprint只有 $O(2Kd)$，小于1MB VRAM。

---

## 9. Reference Model Sensitivity (Appendix H)

$\omega_t$ 依赖reference model的选择。实验发现:

1. **Clean pre-trained reference** 始终能把memorization suppress到0
2. **Partially trained reference** 表现non-monotonic——初始mitigation减弱（memorization signals变diffuse），后期partial recovery（remaining signal变concentrated）
3. **Reference capacity matters**: 4-layer reference在late epochs出现degenerate cases，8-layer reference更stable
4. **Clean accuracy不变**: 96.98% across all configurations

这告诉我们reference model的作用是**anchor memorization signal**——它需要足够expressive来区分"generalizable structure"和"memorized content"。

---

## 10. Broader Context & Related Work

### 10.1 vs. Machine Unlearning
- Unlearning (NPO, GA): parameter-space, sequence-level, requires forget set
- Knowledge Editing (ROME, MEMIT): fact-level, localized parameter modification  
- GSS: **token-level, no forget set required, inference-time only**

### 10.2 vs. Activation Steering
- CAA (Contrastive Activation Addition): same direction for detection AND correction
- GSS: **decoupled probe and steer**, with principled optimization

### 10.3 vs. LoRA
- LoRA: static rank-r weight update
- GSS: **dynamic, context-dependent rank-r update** via gating

---

## 11. 我的Intuition总结

GSS的elegance在于它把一个seemingly complex的问题reduce成三个clean的mathematical operations:

1. **Detection**: 通过memorization signal $\omega_t$ 确定哪些tokens是memorized
2. **Geometry**: 通过whitening by $\Sigma_{\text{gen}}^{-1/2}$ 处理anisotropic geometry
3. **Correction**: 通过SVD of whitened memorization matrix找到optimal probe-steer pairs

最key的intuition是: **memorization在activation space中是一个low-dimensional, sparse, token-conditional的subspace**。GSS通过gating只在memorization signal exceed threshold时activate correction，避免了static方法的collateral damage。

这让我联想到sparse coding的思想——找到数据中"少数但重要"的components，并在inference时conditional地处理它们。GSS把这个思想apply到LLM的activation space，用information-theoretic的memorization signal作为sparsity prior。

---

## Reference Links

- **Paper**: [GSS: Gated Subspace Steering for Selective Memorization Mitigation in LLMs](https://arxiv.org/abs/2507.05578) (Xiong et al., 2025)
- **Memorization in LLMs**: [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) (Carlini et al., 2021)
- **Activation Steering**: [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.857/) (Rimsky et al., 2024)
- **TinyMem Benchmark**: [Mitigating Memorization in Language Models](https://arxiv.org/abs/2410.02159) (Sakarvadia et al., 2024)
- **NPO**: [Negative Preference Optimization](https://arxiv.org/abs/2404.05868) (Zhang et al., 2024)
- **ROME**: [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (Meng et al., 2022)
- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2022)
- **AlphaSteer**: [Learning Refusal Steering with Principled Null-Space Constraint](https://arxiv.org/abs/2506.07022) (Sheng et al., 2025)
- **LUNAR**: [LLM Unlearning via Neural Activation Redirection](https://arxiv.org/abs/2410.11334) (Shen et al., 2025)
- **Pythia**: [Pythia: A Suite for Analyzing LMs across Training and Scaling](https://arxiv.org/abs/2304.01373)
- **The Geometry of Truth**: [Marks & Tegmark](https://arxiv.org/abs/2310.06824)

这篇paper我觉得最值得借鉴的是它的**decoupling principle**——把"detect when"和"decide how"分成两个independent的optimization problems。这个pattern在很多场景下都有潜力：safety filtering, hallucination detection, 甚至可能是future的MoE routing机制。
