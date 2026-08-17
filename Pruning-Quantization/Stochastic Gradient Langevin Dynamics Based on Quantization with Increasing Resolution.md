---
source_pdf: Stochastic Gradient Langevin Dynamics Based on Quantization with Increasing
  Resolution.pdf
paper_sha256: 5952814f64de9df9229dfb1c6772b026e68a0cdd809e84486bcc10abab4ece48
processed_at: '2026-08-12T11:15:34-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 QSGLD

Andrej，咱们抛开公式，用大白话把这个 paper 的 story 拆开。

---

## 一句话版本

> 你给 gradient update 做 quantization（取整），quantization 带来的 round-off error **本身就是 Langevin dynamics 需要的 noise**——不用再额外 inject Gaussian noise 了。

---

## 为什么这事 cool？先看传统做法有多别扭

### SGLD 的尴尬

[Welling & Teh 2011](https://www.stats.ox.ac.uk/~teh/research/npbg/WelTeh2011.pdf) 的 SGLD 长这样：

$$
X_{\tau+1} = X_\tau - \lambda \nabla f(X_\tau) + \sqrt{2\lambda T} \cdot z_\tau
$$

那个 $z_\tau \sim \mathcal{N}(0, I_d)$ 是你 **每一步都要 sample 的 Gaussian noise**，维度跟 weight 一样大。ResNet-50 有 25M 参数，你每 step 要 sample 25M 个 Gaussian random number。这 sounds stupid，但大家一直就这么干。

### LSR (Linear Scaling Rule) 也不优雅

[Krizhevsky 2014](https://arxiv.org/abs/1404.5997) 和 [Goyal et al. 2018](https://arxiv.org/abs/1706.02677) 发现：你把 batch size 撑大，SGD 自己的 sampling noise 就变小了，所以 learning rate 要跟着 linearly scale up。这本质上是用 **data sampling 的方差** 当 noise source。

问题是：
- Large batch 在 distributed/federated 场景 [Lin et al. 2020](https://openreview.net/forum?id=B1eyO1BFPr) 根本玩不转
- 需要 warm-up，不然爆掉
- Noise 的 structure 跟 data 分布强耦合，你没法 independent 控制

---

## 这篇 paper 的 "啊哈" 时刻

### Insight #1: Quantization error 是 free noise

你想，你做 $x^Q = \frac{1}{Q_p} \lfloor Q_p x + 0.5 \rfloor$，这个 rounding 操作天然有误差：

$$
\epsilon^q = x^Q - x \in \left[-\frac{1}{2Q_p}, \frac{1}{2Q_p}\right)
$$

这个 $\epsilon^q$ 是 **uniform 分布的**，zero-mean，variance 是 $\frac{1}{12 Q_p^2}$。

传统 signal processing 里，这个 error 是 **要消灭的敌人**——大家花几十年研究怎么 reduce quantization noise [Gray & Neuhoff 1998](https://ieeexplore.ieee.org/document/1455556)。

这篇 paper 说：**慢着，这玩意儿本身就是一个 zero-mean noise source，正好可以当 Langevin 的 Brownian motion 用**。

### Insight #2: Uniform + CLT = Gaussian

Langevin SDE 需要 **Gaussian** noise。但 quantization error 是 **uniform**。怎么办？

这里是 paper 最聪明的地方：**sum over mini-batches，让 CLT 干活**。

你一个 epoch 有 $B$ 个 mini-batch，每个 mini-batch update 都产生一个 uniform quantization error。把一个 epoch 内的所有 update 加起来：

$$
\sum_{\tau=0}^{B-1} \epsilon_\tau^q \xrightarrow{B \to \infty} \mathcal{N}(0, \cdot)
$$

by [Lindeberg-Lévy CLT](https://en.wikipedia.org/wiki/Central_limit_theorem)。

**你啥 RNG 都不用！** Quantization error 自己 sum 起来就变 Gaussian。实验里 $B \geq 50$ 就够了。

这就像你在路边捡了一堆 uniform 的石头，堆在一起，按 CLT 的魔法，整体看起来就是 Gaussian 分布的 hill。

### Insight #3: 增大 resolution = cooling = annealing

Quantization parameter $Q_p$ 控制 grid 精细度：
- $Q_p$ 小 → grid 粗 → quantization error 大 → noise 大
- $Q_p$ 大 → grid 细 → quantization error 小 → noise 小

Paper 让 $Q_p$ 随时间 **缓慢增大**（log-log 增长），noise 就缓慢减小。

这正好对应 **simulated annealing 的 cooling schedule**！[Geman & Hwang 1986](https://epubs.siam.org/doi/10.1137/0324062) 证明过，只要 cooling 是 logarithmic 的（$\sigma^2(t) \sim 1/\log t$），annealing 就收敛到 global optimum。

这里 $Q_p(t) \sim \sqrt{\log t}$，所以 $\sigma^2 \sim 1/Q_p^2 \sim 1/\log t$，**完美 match**。

你完全不用设计 annealing schedule——quantization resolution 增长 **物理上就是** cooling。

### Insight #4: Early paralysis 与 dithering

这里有个坑：训练初期 gradient 很小（尤其是 KL-divergence loss），如果 $|\lambda h| < \frac{1}{2Q_p}$，量化后直接变 0——**network 卡死不动了**。

Signal processing 里有个老 trick 叫 **dithering** [Marco & Neuhoff 2005](https://ieeexplore.ieee.org/document/1430531)：在 quantize 之前加一点 uniform noise，就能 break 掉 quantization error 跟 input 的 correlation。

Paper 的 compensation function $r(\tau, X)$ 干的就是这个——在 quantize 之前，沿着 gradient 方向加一个 sigmoid-shaped 的 perturbation：

$$
r(\tau, X) = \lambda \cdot \text{sigmoid}(-\kappa(\tau - \tau_0)) \cdot \frac{h(X)}{\|h(X)\|}
$$

早期 $\tau \ll \tau_0$ 时 $r \approx \lambda \cdot \text{sgn}(h)$，保证有东西可以 quantize；后期 $\tau \gg \tau_0$ 时 $r \to 0$，自动消失，不影响 final convergence。

这就是 dithering 的 directional 版本——不但 break correlation，还沿着有用方向推一把。

---

## 整个 story 连起来

```
传统 view:                        This paper's view:
                                  
quantization = 压缩工具            quantization = noise source
quantization error = 敌人          quantization error = free Brownian motion
                                  
SGLD:                             QSGLD:
  sample Gaussian z                quantize update
  inject z                         quantization error = noise
  schedule temperature             increase Q_p = cooling
  ↳ 手动设计                       ↳ 物理自然涌现
```

---

## 数学上为什么 converge？

[Theorem 3.3](https://arxiv.org/abs/2410.14272) 的核心链路（大白话版）：

1. **Discrete → Continuous**: 你的 quantized update equation weakly converge 到 Langevin SDE（用 [Li et al. 2019](http://jmlr.org/papers/v20/17-526.html) 的 framework）

2. **Langevin SDE = Markov process**: 这个 SDE 有 transition probability $p(t, x, t+1, y)$

3. **用 Girsanov theorem**: 把"有 drift 的 Langevin"跟"纯 Brownian"的 probability measure 联系起来。Radon-Nikodym derivative 给你一个 bound：

$$
\frac{dP_X}{dQ_{\bar{X}}} \geq \exp\left(-\frac{C_3}{\sigma^2(t)}\right)
$$

4. **Bound transition probability**: 从这个 bound 得到 $\delta_t \geq \exp(-C_5/\sigma^2(t))$

5. **Divergence check**: 
$$
\sum_{k=0}^\infty \delta_{t+k} \geq \sum_{k=0}^\infty \frac{1}{t+2+k} = \infty
$$
因为 harmonic series 发散，所以 Cauchy convergence 成立。

关键就一句：**logarithmic cooling 保证 Markov chain ergodic**。这是 [Geman & Hwang 1986](https://epubs.siam.org/doi/10.1137/0324062) 的老结果，这里恰好通过 $Q_p(t) \sim \sqrt{\log t}$ 满足。

---

## 实验结果有多 impressive？

| 对比 | 提升 |
|------|------|
| CIFAR-10 / ResNet-50: QSGD vs SGD | **+10.5%** (73.8 vs 63.3) |
| CIFAR-100 / ResNet-50: QSGD vs SGD | **+11.9%** (37.8 vs 25.9) |
| CIFAR-10: QADAM vs ADAM | +3.0% |
| CIFAR-100: QADAM vs ADAM | +3.3% |

CIFAR-100 上 SGD 只到 25.9%，QSGD 到 37.8%——这 gap 巨大。而且实现就 Algorithm 1 那 6 行代码。

---

## 我觉得最 elegant 的地方

1. **发现 quantization 的 dual role**: compression tool → noise source。这种 "reframing" 是好研究的标志。

2. **CLT 让 uniform 自动变 Gaussian**: 你不用维护 RNG，noise distribution 从 quantization 结构中 **物理涌现**。这比 SGLD 的 explicit Gaussian injection 自然得多。

3. **Cooling schedule 物理化**: $Q_p(t)$ 增长 ↔ noise 减小 ↔ annealing cooling。一个 knob 同时控制 quantization 精度和 noise temperature。

4. **Dithering 的 directional 版本**: compensation function 不只是 break correlation，还沿着 gradient 方向，是 "useful dithering"。

5. **三个领域缝合**: signal processing (quantization/WNH) + stochastic process (Langevin/Girsanov) + optimization (SGD/annealing)。这三个 community 平时不说话，这里串起来了。

---

## 我的吐槽

1. **只在 vision 上试**: 没有 transformer / LM 实验。LLM pre-training 上能不能 work 完全不知道。

2. **CIFAR-100 QSGD 只到 37.8%**: 虽然比 SGD 强，但绝对值低。说明 quantization noise 的 structure 可能不适合特别难的任务。

3. **Compensation function 太 heuristic**: sigmoid 形状、$\tau_0$ 比例都是 hand-tuned。Table 4 显示 $\tau_0$ 敏感性其实不小。

4. **没跟 SAM 比**: [Foret et al. 2020](https://arxiv.org/abs/2010.01412) 的 SAM 也是找 flat minima，思路相近（perturb weight 看 worst case），但 mechanism 完全不同。应该 head-to-head。

5. **$\eta^2 = 2^{19}$ 这个数从哪来的**: 论文没解释。这种 magic number 让人 nervous。

6. **WNH 是 asymptotic 的**: $Q_p \to \infty$ 才严格成立，实际 $Q_p$ 有限。Paper 没量化 finite-$Q_p$ 的 deviation。

---

## 如果你只记一件事

> Quantization error 是 zero-mean uniform noise，sum 起来 by CLT 变 Gaussian，正好当 Langevin 的 Brownian motion。Increase quantization resolution = cooling schedule。不需要 RNG，noise 从 quantization 物理涌现。

这是 [Seok & Cho 2024](https://arxiv.org/abs/2410.14272) 整篇 paper 的精华。

如果你想在 NanoGPT 上试一下，实现就 6 行，我可以帮你写 PyTorch code。这玩意儿如果在 LLM pre-training 上 work，那 story 就完整了。

---

# QSGLD: Quantization-Based Stochastic Gradient Langevin Dynamics 深度解析

Andrej，这篇 paper 来自 ETRI (Electronics and Telecommunications Research Institute) 的 Jinwuk Seok 和 Changsik Cho，core idea 非常 elegant——用 **quantization error 本身来产生 Langevin SDE 需要的 Gaussian noise**，从而避免传统 SGLD 中的 explicit random number generator。我帮你 build 一下 intuition，从 high-level motivation 一直 drill down 到 theorem 和实验。

---

## 1. Motivation: 为什么需要"又一种"noisy SGD？

### 1.1 背景脉络

Stochastic analysis of SGD 的两条主路线：
- **Linear Scaling Rule (LSR)** [Krizhevsky 2014, Goyal et al. 2018]: 控制 noise variance 的方式是改变 mini-batch size $B$。batch 越大，noise 越小，需要 linearly scale learning rate $\lambda \propto B$。
- **Stochastic Gradient Langevin Dynamics (SGLD)** [Welling & Teh 2011]: 直接 inject isotropic Gaussian noise $\sqrt{2\lambda T} \cdot \mathcal{N}(0, I_d)$。

LSR-based SGD 和 SGLD 各有痛点：
- LSR 要求 **warm-up** [Goyal et al. 2018]，在分布式/federated learning 环境下 large batch 不现实 [Lin et al. 2020]。
- SGLD 需要 **i.i.d. Gaussian random number generator**——这不是免费的，每一步都要 sample $z \sim \mathcal{N}(0, I_d)$，维度和 weight vector 相同。
- 还有 SVAG (Stochastic Variance Amplified Gradient) [Li et al. 2021, Malladi et al. 2022] 这种 extra computation 的方案。

Reference: 
- [Welling & Teh 2011 SGLD](https://www.stats.ox.ac.uk/~teh/research/npbg/WelTeh2011.pdf)
- [Goyal et al. 2018 Large Batch Training](https://arxiv.org/abs/1706.02677)
- [Lin et al. 2020 Local SGD](https://openreview.net/forum?id=B1eyO1BFPr)

### 1.2 这篇 paper 的 key insight

> Quantization error 本身在 WNH (White Noise Hypothesis) 下是 I.I.D. white noise。我们不需要额外 inject noise——只需要 **quantize** gradient update step，让 quantization error **充当** Langevin SDE 中的 Brownian motion driver。

这非常 elegant：传统 quantization 是为了 **reduce computational burden**（model compression, 1-bit SGD [Seide et al. 2014] 等），这里 quantization 反过来作为 **noise source for optimization**。

而且 quantization resolution $Q_p(t)$ **随时间 increasing**，对应 noise variance 随时间 decreasing——这就是 annealing，和 SGLD 中需要手动 schedule temperature 完全一致，但物理来源更自然。

---

## 2. Quantization 的数学 setup

### 2.1 Scalar quantization (Eq 5)

$$
x^Q \triangleq \frac{1}{Q_p} \lfloor Q_p \cdot (x + 0.5 \cdot Q_p^{-1}) \rfloor = x + \varepsilon^q Q_p^{-1}
$$

变量解析：
- $x \in \mathbb{R}$：输入实数（例如 weight 或 gradient 的一个分量）
- $Q_p \in \mathbb{Q}^+$：quantization parameter，本质是 $\Delta^{-1}$，即 quantization step 的倒数
- $\lfloor \cdot \rfloor$：floor function
- $\varepsilon^q \in \mathbb{R}[-1/2, 1/2)$：**scalar factor for quantization**，由 floor operation 产生的"误差归一化值"

Intuition: $Q_p$ 越大，grid 越精细，量化误差 $\epsilon^q = \varepsilon^q / Q_p$ 越小。$Q_p \to \infty$ 时退化为 identity。

### 2.2 Vector quantization (Eq 6)

$$
\boldsymbol{x}^Q \triangleq \sum_{i=1}^d (\boldsymbol{x} \cdot \boldsymbol{e}^{(i)})^Q \boldsymbol{e}^{(i)} \implies \boldsymbol{\epsilon}^q = \boldsymbol{x}^Q - \boldsymbol{x} = Q_p^{-1} \boldsymbol{\varepsilon}^q
$$

每个 component 独立量化，关键分离：
- $\boldsymbol{\varepsilon}^q \in \mathbb{R}^d[-1/2, 1/2)^d$：**unit-scale factor**，分布固定
- $\boldsymbol{\epsilon}^q \in \mathbb{R}^d[-\frac{Q_p^{-1}}{2}, \frac{Q_p^{-1}}{2})^d$：**actual quantization error**，scale 由 $Q_p$ 决定

### 2.3 Time-varying quantization parameter (Eq 7)

$$
Q_p(\varepsilon^q, t) = \eta(\varepsilon^q) \cdot b^{\bar{p}(t)}
$$

- $\eta: \mathbb{R}^d \mapsto \mathbb{Q}^{++}$：auxiliary function，可以是 constant，也可以用来做 distribution transformation
- $b \in \mathbb{Z}^+$：base（实际实验用 $b = 2$）
- $\bar{p}: \mathbb{R}^{++} \mapsto \mathbb{Z}^+$：power function，monotone increasing $\bar{p}(t) \uparrow \infty$ as $t \to \infty$

这就是 "increasing resolution" 的来源——$Q_p(t)$ 随 $t$ 增大，noise variance $c_0 Q_p^{-2}(t)$ 减小，对应 simulated annealing 的 cooling schedule。

### 2.4 Statistical properties (Eq 8)

在 Assumption 2 (uniform distribution of quantization error) 下：

$$
\mathbb{E}_{\varepsilon^q} \epsilon^q = 0, \quad \mathbb{E}_{\varepsilon^q} (\epsilon^q)^2 = \frac{1}{12} Q_p^{-2} = c_0 Q_p^{-2}
$$

其中 $c_0 = 1/12$。这是 uniform distribution $[-Q_p^{-1}/2, Q_p^{-1}/2)$ 的标准方差。

### 2.5 WNH (White Noise Hypothesis) [Jiménez et al. 2007]

> 当 sample 数足够多且 $Q_p$ 足够大时，quantization error 是 I.I.D. white noise。

这就是让整个 SDE analysis 成立的关键假设。但有一个 caveat：**早期阶段** $Q_p$ 不够大时，quantization error 与 input $x$ 是 correlated 的（Eq 9）：

$$
\mathbb{E}_{\varepsilon^q}[X \epsilon^q | X^Q = k Q_p^{-1}] = c_0 Q_p^{-2} \neq 0
$$

这个 correlation 会导致 **early paralysis**——如果 $|h|$ 很小（远小于 $Q_p^{-1}$），量化后 $h^Q = 0$，learning 就停了。这就是为什么需要 compensation function。

Reference: [Jiménez et al. 2007 WNH](https://epubs.siam.org/doi/10.1137/060653809), [Gray & Neuhoff 1998 Quantization](https://ieeexplore.ieee.org/document/1455556)

---

## 3. 从 quantization 到 Langevin SDE

### 3.1 Fundamental learning equation (Eq 10-13)

普通 SGD update:
$$
\boldsymbol{X}_{\tau+1} = \boldsymbol{X}_\tau + \lambda h(\boldsymbol{X}_\tau)
$$

Quantized version (Eq 11):
$$
\boldsymbol{X}_{\tau+1}^Q = \boldsymbol{X}_\tau^Q + [\lambda h(\boldsymbol{X}_\tau^Q)]^Q
$$

展开 (Eq 12-13):
$$
\boldsymbol{X}_{\tau+1}^Q = \boldsymbol{X}_\tau^Q + \lambda h(\boldsymbol{X}_\tau^Q) + Q_p^{-1}(\boldsymbol{\varepsilon}_\tau^q, \tau) \boldsymbol{\varepsilon}_\tau^q
$$

对比标准 discrete Langevin equation:
$$
\boldsymbol{X}_{\tau+1} = \boldsymbol{X}_\tau - \lambda \nabla f(\boldsymbol{X}_\tau) + \sqrt{2\lambda T} \boldsymbol{z}_\tau, \quad \boldsymbol{z}_\tau \sim \mathcal{N}(0, I_d)
$$

**Structural correspondence**:
- $\lambda h(\boldsymbol{X}_\tau^Q)$：drift term（gradient or momentum direction）
- $Q_p^{-1}(\tau) \boldsymbol{\varepsilon}_\tau^q$：diffusion term，**uniformly distributed** noise with variance $c_0 Q_p^{-2}(\tau) I_d$

### 3.2 两种 transform 路径

#### Path A: Direct Gaussian transform (Eq 14)

用 Box-Muller / Ziggurat / inverse transform sampling，把 uniform $\varepsilon^q$ 变成 Gaussian $z_\tau$：

$$
\boldsymbol{X}_{\tau+1}^Q = \boldsymbol{X}_\tau^Q - \lambda \nabla \tilde{f}_\tau(\boldsymbol{X}_\tau^Q) + \sqrt{\lambda} \cdot b^{-\bar{p}(\tau)} \boldsymbol{z}_\tau
$$

但作者明确说 **这条路径在 implementation 上没优势**——你还是要做 transformation，等于你还是要算 Gaussian random number。所以这条路径只是 theoretical validation。

#### Path B: CLT-based approach (Eq 15) ← 实际用的方法

这是论文真正的 implementation path。把一个 epoch 内的 mini-batch updates 求和：

$$
\boldsymbol{X}_{t_e+1}^Q = \boldsymbol{X}_{t_e}^Q - \lambda \sum_{\tau=0}^{B-1} \nabla \tilde{f}_\tau(\boldsymbol{X}_{t_e+\tau/B}^Q) + b^{-\bar{p}(t_e)} \lambda \sqrt{\frac{C_q}{c_0}} \sum_{\tau=0}^{B-1} \boldsymbol{\varepsilon}_{t_e+\tau/B}^q
$$

关键点：$Q_p^{-1}(t_e) = \lambda \sqrt{C_q/c_0} b^{-\bar{p}(t_e)}$ 是 **每个 epoch 内 constant** 的。

由 **Lindeberg-Lévy CLT**：
$$
\sqrt{\frac{\lambda}{c_0}} \sum_{\tau=0}^{B-1} \boldsymbol{\varepsilon}_{t_e+\tau/B}^q \xrightarrow{B \uparrow \infty} \boldsymbol{z}_{t_e} \sim \mathcal{N}(0, I_d)
$$

即 uniform 的 quantization error 在一个 epoch 内 sum 起来，asymptotically Gaussian。

**Intuition**: 你不需要任何额外的 RNG，只要 mini-batch 数 $B$ 够大（实验中 ≥50 就够），sum of uniform quantization errors 自己就变成 Gaussian。这是非常聪明的。

参考 Eq 61-62 的 derivation：
$$
\text{Cov}(\boldsymbol{S}_B) = \frac{B c_0}{B^2} I_d = \lambda c_0 I_d
$$
所以 $\boldsymbol{S}_B / \sqrt{\lambda c_0}$ 经过 normalization 后是 standard normal。

### 3.3 Continuous-time limit (Lemma 3.2)

当 $\lambda \downarrow 0$，得到 Langevin SDE (Eq 21):
$$
d\boldsymbol{X}_t = -\nabla f(\boldsymbol{X}_t) dt + \sqrt{C_q} \cdot \sigma(t) d\boldsymbol{B}_t, \quad \sigma(t) \triangleq b^{-\bar{p}(t)}
$$

- $\boldsymbol{B}_t$：standard Brownian motion in $\mathbb{R}^d$
- $\sigma(t) = b^{-\bar{p}(t)}$：time-varying diffusion coefficient，monotone decreasing（因为 $\bar{p}$ increasing）
- $C_q$：常数，用 normalizing uniform variance 与 Gaussian variance

这是 **time-inhomogeneous Langevin SDE**。Diffusion coefficient 随时间衰减等价于 cooling，对应 Gibbs sampling 中 temperature 降低。在 stationary distribution 上，这最终 concentrates 在 mode 附近。

**Order-1 weak approximation** (Definition 3, Li et al. 2019 / Malladi et al. 2022):
$$
\max_{\tau \in \mathbb{Z}[0, \lfloor T/\lambda \rfloor]} |\mathbb{E} g(\boldsymbol{X}_t) - \mathbb{E} g(\boldsymbol{X}_{\lfloor \tau/B \rfloor}^Q)| \leq C \lambda^2
$$

对于任意 polynomial growth test function $g$。证明需要 verification of bounded moments (Eq 76-87)。

Reference:
- [Li et al. 2019 SME](http://jmlr.org/papers/v20/17-526.html)
- [Malladi et al. 2022 SDE for adaptive methods](https://proceedings.neurips.cc/paper_files/paper/2022/file/32ac710102f0620d0f28d5d05a44fe08-Paper.pdf)

---

## 4. Early Paralysis 问题与 Compensation Function

### 4.1 问题 statement (Eq 16)

如果初期 $\max \|\lambda h\| < 0.5 Q_p^{-1}(\tau) - \delta$，那么：
$$
\frac{1}{Q_p} \| \lfloor Q_p(\lambda h + 0.5 Q_p^{-1}) \rfloor \| \leq \frac{1}{Q_p} \lfloor 1 - \delta Q_p \rfloor = 0
$$

即 **整个 update 直接消失**！这在 KL-divergence-based loss (cross-entropy) 训练初期特别危险，因为 initial gradient magnitude 很小。

### 4.2 Compensation function (Eq 18 / 40)

$$
r(\tau, \boldsymbol{X}_\tau) = \lambda \cdot \left( \frac{\exp(-\varkappa (\tau - \tau_0))}{1 + \exp(-\varkappa (\tau - \tau_0))} \cdot \frac{h(\boldsymbol{X}_\tau^Q)}{\|h(\boldsymbol{X}_\tau^Q)\|} \right)
$$

变量解析：
- $\varkappa > 0$：控制 sigmoid 衰减速度的 shape parameter
- $\tau_0 \in \mathbb{Z}^{++}$：half-time，表示 compensation 衰减到 0.5 的时间点
- $\frac{h(\boldsymbol{X}_\tau^Q)}{\|h(\boldsymbol{X}_\tau^Q)\|}$：normalized direction（unit vector）

当 $\tau \ll \tau_0$，sigmoid 趋近 1，$r \approx \lambda \cdot \text{sgn}(h)$；当 $\tau \gg \tau_0$，$r \to 0$。

### 4.3 Theorem 3.1 的核心

$$
h^Q(\boldsymbol{X}_\tau^Q) \triangleq \frac{1}{Q_p} \lfloor Q_p \cdot (\lambda h(\boldsymbol{X}_\tau^Q) + r(\tau, \boldsymbol{X}_\tau^Q)) + 0.5 \rfloor
$$

则 quantization input $h(\boldsymbol{X}_\tau^Q)$ 与 quantization error $\epsilon_\tau^q$ **uncorrelated**：
$$
\mathbb{E}_{\epsilon_\tau^q}[h(\boldsymbol{X}_\tau^Q) \epsilon_\tau^q | h^Q = k Q_p^{-1}] = 0
$$

**为什么这个 important**: WNH 要求 quantization error 独立于 input，但原始 quantization 不满足这个（Eq 9）。Compensation function 通过 "add dither-like signal" 把 correlation 破坏掉，类似 signal processing 中的 **dithering** [Marco & Neuhoff 2005, Gray & Neuhoff 2006]。

证明思路（Eq 48）：
$$
\mathbb{E}[h \epsilon^q | h^Q = k Q_p^{-1}] = \mathbb{E}_r[r] + \mathbb{E}_{\epsilon^q}[\epsilon^q] = 0 + 0 = 0
$$

因为 $\mathbb{E}_r[r] = 0$（compensation function 是对称 Bernoulli-like）和 $\mathbb{E}_{\epsilon^q}[\epsilon^q] = 0$（zero-mean uniform）。

### 4.4 Dithering 类比

这部分（Section B.2）非常 informative。在 signal processing 中，dithering 是 **添加均匀分布的 noise 到 signal 中以破坏 quantization error 与 signal 的 correlation**。这里 compensation function 起同样作用，但 **directional**——沿着 $h / \|h\|$ 方向，所以还提供了有用的探索方向。

参考 Eq 37:
$$
\mathbb{E}_{X, z}[X(X - (X+z)^Q) | X^Q = k Q_p^{-1}] = 0
$$

加入 $z \sim \text{Uniform}[-Q_p^{-1}/2, Q_p^{-1}/2)$ 后，correlation 立即归零。

Reference:
- [Marco & Neuhoff 2005 Additive Noise Model](https://ieeexplore.ieee.org/document/1430531)
- [Zamir & Feder 1996 Lattice Quantization](https://ieeexplore.ieee.org/document/508838)

---

## 5. 收敛性分析

### 5.1 Theorem 3.3 (Weak convergence without convexity)

Quantization parameter bound (Eq 22):
$$
\sup_{t \geq 0} Q_p(t) = \sqrt{\frac{1}{C} \log(t+2)}, \quad C \in \mathbb{R}^{++}
$$

实际实现 (Eq 24):
$$
Q_p = \left\lfloor \sqrt{\frac{1}{C} \log(t_e + 2)} \right\rfloor
$$

收敛性 (Eq 23):
$$
\varlimsup_{\bar{\tau} \to \infty} \sup_{X_t, \bar{X}_t} \| p(t, \bar{X}_t, t+\bar{\tau}, x^*) - p(t, X_t, t+\bar{\tau}, x^*) \| \leq \tilde{C} \cdot \exp\left(-\sum_{\bar{\tau}=0}^\infty \delta_{t+\bar{\tau}}\right)
$$

变量：
- $p(t, X_t, t+\bar{\tau}, x^*)$：从 $X_t$ 出发，在 $t+\bar{\tau}$ 时刻到达 optimal point $x^*$ 的 transition probability
- $\delta_t = \inf_{x, y} p(t, x, t+1, y)$：单位时间内的最小 transition probability
- $\sum \delta_{t+\bar{\tau}} = \infty$ 是 convergence 条件

**Proof sketch (基于 Geman & Hwang 1986)**：
1. 上下界 transition probability 用 $\prod (1 - \delta_{t+k})$
2. 用 exponential approximation: $\prod (1-\delta_k) \leq \exp(-\sum \delta_k)$
3. 用 **Girsanov theorem** (Eq 100) 计算 Radon-Nikodym derivative $\frac{dP_X}{dQ_{\bar{X}}}$，对比有 drift 的 Langevin 与纯 Brownian
4. 用 Lipschitz continuity (Assumption 1) bound $\|\nabla f\| \leq L_1 \rho = C_0$
5. 用 Itô formula 展开 $df(X_s)$ (Eq 104)
6. 得到 $\delta_t \geq \exp(-C_5 / \sigma^2(t))$ (Eq 115)
7. 取 $\sigma^2(t) \geq C_5 / \log(t+2)$，则 $\sum \delta_{t+k} \geq \sum \frac{1}{t+2+k} = \infty$（harmonic series diverges）

**Intuition**: 这就是把 Langevin SDE 当 Markov process，证明其 ergodic。Key 是 cooling schedule 不能太快——$\sigma^2(t) \sim 1/\log(t)$ 是 logarithmic cooling，这是 simulated annealing 中的经典结果 [Geman & Hwang 1986]。

Reference: [Geman & Hwang 1986 Simulated Annealing](https://epubs.siam.org/doi/10.1137/0324062), [Øksendal SDE](https://link.springer.com/book/10.1007/978-3-662-03620-4)

### 5.2 Theorem 3.4 (Local convergence under convexity)

假设 optimal point 附近 Hessian $H(f)$ 是 positive definite (Assumption 4)，则:
$$
\mathbb{E}_{\epsilon^q_\tau} f(X_{\tau+k}^Q) - f(x^*) \leq \exp(-C_0 \cdot k) \cdot L_0 \rho
$$

证明 (Eq 118-129) 用 standard convex analysis：
1. Taylor 展开 $f(X_{\tau+1}^Q) - f(X_\tau^Q)$
2. 用 Hessian 的 eigenvalue bound $m_{\min}, M_{\max}$
3. 学习率 $\lambda < \min\{1, 2/M\}$ 保证收敛
4. 用 Lipschitz continuity bound initial distance $\|X_0 - x^*\| \leq \rho$

---

## 6. 算法实现

### 6.1 Algorithm 1 总结

```
Initialize τ=0, X_0 ∈ Q^d
repeat:
    h(X_τ^Q) = -∇f(X_τ^Q)  # or ADAM direction
    Q_p(τ) = η * b^(p̄(τ))
    r(τ, X_τ^Q) = λ * sigmoid(-κ(τ - τ_0)) * h(X_τ^Q) / ||h(X_τ^Q)||
    h^Q_τ = (1/Q_p) * floor(Q_p * (λ h(X_τ^Q) + r(τ, X_τ^Q) + 0.5*Q_p^{-1}))
    X_{τ+1}^Q = X_τ^Q + h^Q_τ
    τ += 1
until convergence
```

### 6.2 关键 hyperparameters (Eq 134-137)

$$
\bar{p}(t_e)|_{t_e = \tau/B} = \lfloor 0.5 \cdot \log_b \log(\tau + 2) \rfloor
$$

$$
Q_p = \eta \cdot b^{\bar{p}(t_e)}
$$

实验推荐设置：
- $\eta^2 \in 2^{19} \approx 0.5 \times 10^6$（即 $\eta \approx 724$）
- $C = 1/\eta^2$
- $b = 2$
- $\kappa = 2.0$ 或 $4.0$
- $\tau_0 = 5.2\%$ 至 $20\%$ 的 total epochs（数据集 dependent）

### 6.3 为什么 $\bar{p} = \lfloor 0.5 \log_b \log(\tau + 2) \rfloor$？

由 Eq 22 的 bound $\sqrt{(1/C) \log(t_e+2)}$ 和 $Q_p = \eta \cdot b^{\bar{p}}$，求 $\bar{p}$：

$$
\eta^2 b^{2\bar{p}} \leq \frac{1}{C} \log(t_e + 2) \implies \bar{p} \leq 0.5 \log_b \left(\frac{1}{\eta^2 C} \log(\tau+2)\right)
$$

取 $C = 1/\eta^2$，化简为 $\bar{p} = 0.5 \log_b \log(\tau+2)$，再加 floor 保证 $Q_p \in \mathbb{Z}$。

**Intuition**: $\bar{p}$ 是 log-log 增长——非常缓慢的 cooling。早期 $\bar{p} = 0$，$Q_p = \eta$（很小），noise 大；后期 $\bar{p}$ 缓慢增加，noise 缓慢减小。

---

## 7. 实验结果详细分析

### 7.1 实验配置

| 项目 | 配置 |
|------|------|
| Framework | PyTorch 1.13.1, Python 3.10 |
| GPU | NVIDIA GTX 1080Ti / RTX 3050 |
| Models | Vanilla CNN (8 layers) for FashionMNIST; ResNet-50 (56 layers) for CIFAR |
| Datasets | FashionMNIST, CIFAR-10, CIFAR-100 |
| Learning rate | 0.01 (fixed) |
| Epochs | 200 |
| Batch sizes | 100 (FashionMNIST), 128 (CIFAR-10), 100 (CIFAR-100) |

### 7.2 主结果表 (Table 1 / Table 3)

| Dataset | Model | Algorithm | Training Acc | Testing Acc | Training Error |
|---------|-------|-----------|--------------|-------------|----------------|
| FashionMNIST | CNN-8 | **QSGD** | 97.10 | **91.59** | 0.085 |
| FashionMNIST | CNN-8 | QADAM | 98.43 | 89.29 | 0.060 |
| FashionMNIST | CNN-8 | SGD | 95.59 | 91.47 | 0.133 |
| FashionMNIST | CNN-8 | ADAM | 92.45 | 87.12 | 0.176 |
| CIFAR-10 | ResNet-50 | **QSGD** | 99.90 | **73.80** | 0.009 |
| CIFAR-10 | ResNet-50 | QADAM | 99.99 | **85.09** | 0.011 |
| CIFAR-10 | ResNet-50 | SGD | 99.99 | 63.31 | 0.001 |
| CIFAR-10 | ResNet-50 | ADAM | 99.75 | 82.08 | 0.012 |
| CIFAR-100 | ResNet-50 | **QSGD** | 99.04 | 37.77 | 0.030 |
| CIFAR-100 | ResNet-50 | QADAM | 98.62 | **49.60** | 0.038 |
| CIFAR-100 | ResNet-50 | SGD | 98.24 | 25.90 | 0.005 |
| CIFAR-100 | ResNet-50 | ADAM | 98.85 | 46.32 | 0.039 |

### 7.3 结果解读

**CIFAR-10 / ResNet-50**:
- QSGD vs SGD: +10.5% test accuracy (73.80 vs 63.31) — **巨大提升**
- QADAM vs ADAM: +3.0% test accuracy (85.09 vs 82.08)
- QADAM vs NADAM: +2.6%
- QADAM vs RADAM: +2.8%

**CIFAR-100 / ResNet-50**:
- QSGD vs SGD: +11.9% test accuracy (37.77 vs 25.90) — **巨大提升**
- QADAM vs ADAM: +3.3% test accuracy (49.60 vs 46.32)

**FashionMNIST / CNN-8**:
- 提升较小，因为 FashionMNIST 已经接近 ceiling，且 objective 较 convex

### 7.4 Hyperparameter 敏感性 (Table 4)

Compensation function application period $\tau_0$ 的 effect：

| Period Ratio (%) | CIFAR-10 Testing | CIFAR-100 Testing |
|------------------|------------------|-------------------|
| 0.5 | 83.61 | 44.51 |
| 5.0 | 83.39 | **49.60** |
| 10.0 | 84.18 | 44.59 |
| 20.0 | **85.08** | 46.78 |
| 50.0 | 84.04 | 48.49 |
| 100.0 | 83.39 | 46.76 |

**Intuition**: Compensation 起作用有一个 sweet spot。
- 太短：early paralysis 没完全解决
- 太长：后期 noise 太大，干扰 fine-tuning
- CIFAR-100 难度高，需要更长的 exploration（5%）
- CIFAR-10 适中（20%）

---

## 8. 与相关工作的关系

### 8.1 与 SGLD 的对比

| 维度 | SGLD | QSGLD |
|------|------|-------|
| Noise source | Explicit Gaussian RNG | Quantization error (free) |
| Noise distribution | Gaussian by design | Uniform per step, Gaussian by CLT |
| Noise variance control | Temperature $T$ | Quantization resolution $Q_p$ |
| Mini-batch dependency | Independent | Needs $B \geq 50$ for CLT |
| Distribution shift | None | Need compensation for early stage |

### 8.2 与 SVAG [Li et al. 2021, Malladi et al. 2022] 对比

SVAG 通过 **scaling gradient noise** 来 match SDE，需要 extra computation。QSGLD 通过 quantization 实现同样效果，**without extra computation**——quantization 本身就引入了 noise。

### 8.3 与 Quantization in Deep Learning 对比

| 工作 | 目的 | 是否利用 quantization noise |
|------|------|----------------------------|
| Han et al. 2015 (Deep Compression) | Model compression | 否 |
| Seide et al. 2014 (1-bit SGD) | Communication efficiency | 否 |
| Jung et al. 2019 (LSQ) | Learnable quantization | 否 |
| Zhang et al. 2022 (Low-precision SGLD) | SGLD with low precision | 部分 |
| **This paper** | **Optimization noise source** | **核心利用** |

[Zhang et al. 2022 Low-precision SGLD](https://proceedings.mlr.press/v162/zhang22ag.html) 是最相近的，但它是把 low precision 作为 SGLD 的 implementation trick，本 paper 把 quantization 作为 SGLD 的 **noise source 本身**。

### 8.4 与 Simulated Annealing 的关系

Eq 21 中的 $\sigma(t) = b^{-\bar{p}(t)}$ 对应 annealing schedule。Theorem 3.3 中的 $\sigma^2(t) \geq C_5 / \log(t+2)$ 对应 **logarithmic cooling**，是 Geman & Hwang 1986 证明 SA 收敛的标准 condition。

这意味着 QSGLD 本质上是 **continuous-time simulated annealing** 的一种实现，且 annealing schedule 通过 quantization resolution 物理实现，不依赖外部 RNG。

Reference:
- [Raginsky et al. 2017 SGLD non-convex](http://proceedings.mlr.press/v65/raginsky17a.html)
- [Xu et al. 2018 Langevin non-convex](https://proceedings.neurips.cc/paper/2018/file/9c19a2aa1d84e04b0bd4bc888792bd1e-Paper.pdf)

---

## 9. 关键 Intuition 总结

### 9.1 Three-layer intuition

**Layer 1 (surface)**: Quantize gradient updates，让 quantization error 当 noise。

**Layer 2 (mechanism)**:
- Uniform per-step quantization error
- Sum over mini-batches → Gaussian via CLT
- $Q_p$ 随 epoch 增大 → noise variance 减小 → cooling schedule
- Compensation function 破坏 input-error correlation，避免 early paralysis

**Layer 3 (optimization theory)**:
- Limit SDE 是 time-inhomogeneous Langevin
- $\sigma(t) \sim 1/\sqrt{\log t}$ 满足 logarithmic cooling condition
- Girsanov theorem 给出 transition probability 的 Radon-Nikodym bound
- Sum of transition probabilities diverges (harmonic series) → convergence

### 9.2 为什么这个方法 work？

1. **Quantization error 是 "free noise"**——不需要 RNG，不需要额外计算
2. **CLT 提供 Gaussian 收敛**——只要 mini-batch 数够，uniform sum 自动 Gaussian
3. **Increasing resolution 提供 annealing**——log-log 增长满足 SA 的收敛 condition
4. **Compensation function 解决 cold-start**——dithering 思想

### 9.3 与 SGD noise 的区别

SGD noise 来自 **mini-batch sampling variance**：
$$
\text{Var}(\nabla f_B) \approx \frac{1}{B} \text{Var}(\nabla f_{single})
$$

QSGLD noise 来自 **quantization error**：
$$
\text{Var}(\epsilon^q) = \frac{c_0}{Q_p^2}
$$

两者**独立**，可以叠加！这意味着 QSGLD 实际上同时利用了：
- SGD noise（来自 sampling）
- Quantization noise（来自 rounding）

且 quantization noise 是 **isotropic uniform**，与 SGD noise 的结构不同，可能提供 exploration 的额外维度。

### 9.4 Theoretical beauty

这套理论把三个看似不相关的领域串起来：
1. **Signal processing** 的 quantization + dithering + WNH
2. **Stochastic process** 的 Langevin SDE + Girsanov theorem
3. **Optimization** 的 SGD + annealing + non-convex convergence

而且实现极其简单——Algorithm 1 只有 6 行。

---

## 10. Limitations 和 Open Questions

### 10.1 论文 self-acknowledged limitations

1. **QSLD（非 gradient-based）的收敛分析缺失**——只证明了 QSGLD（用 gradient direction）
2. **Compensation function 的设计是 heuristic**——sigmoid 形状、$\tau_0$ 选择缺乏理论指导
3. **Generalized objective function domain quantization** 未探索
4. **CIFAR-100 中 QSGD 性能低**——large-batch SGD 在 CIFAR-100 上表现差（25.90%），quantization 没完全救回来

### 10.2 我看到的其他 limitations

1. **Only tested on vision tasks**——没有 NLP/Transformer 实验，对 large-scale LM 训练适用性未知
2. **No comparison with SAM (Sharpness-Aware Minimization)** 或其他 flat-minima 寻找方法
3. **No comparison with KFAC / second-order methods**
4. **WNH 在 finite sample 下的 deviation 没量化**——理论上 CLT 是 $B \to \infty$，实际 $B = 50$ 是否够？
5. **Quantization parameter $\eta$ 的选择 heuristic**——$\eta^2 = 2^{19}$ 的依据是什么？
6. **Compensation function 可能干扰 final convergence**——论文 Section B.2 末尾承认这一点

### 10.3 可能的 extensions

1. **Per-layer quantization resolution**：不同层用不同 $Q_p$，类似 LSQ [Jung et al. 2019]
2. **Learnable $\bar{p}(t)$**：把 schedule parameterize 让网络学
3. **结合 SAM**：quantization noise + sharpness penalty 可能有 synergy
4. **Federated learning 应用**：作者提到 large batch 不适合 federated，但 quantization-based noise 在 small batch 场景下应该有天然优势
5. **MCMC perspective**：可以作为 efficient sampler，不只是 optimizer
6. **Connection to Lottery Ticket Hypothesis**：quantization 找 sparse subnetwork？

---

## 11. PyTorch 实现伪代码

基于 Algorithm 1 和 Eq 136：

```python
import torch
import math

class QSGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, eta=724.0, b=2.0, 
                 kappa=2.0, tau0_ratio=0.1, total_epochs=200):
        defaults = dict(lr=lr, eta=eta, b=b, kappa=kappa,
                       tau0_ratio=tau0_ratio, total_epochs=total_epochs)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        
        for group in self.param_groups:
            lr = group['lr']
            eta = group['eta']
            b = group['b']
            kappa = group['kappa']
            tau0 = group['tau0_ratio'] * group['total_epochs'] * self._batch_count
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                # Compute quantization parameter
                tau = self._global_step
                p_bar = math.floor(0.5 * math.log(math.log(tau + 2), b))
                Q_p = eta * (b ** p_bar)
                
                # Compute compensation function
                if tau < 2 * tau0:
                    sigmoid_val = 1.0 / (1.0 + math.exp(kappa * (tau - tau0)))
                    grad_norm = grad.norm() + 1e-8
                    r = lr * sigmoid_val * grad / grad_norm
                else:
                    r = torch.zeros_like(grad)
                
                # Quantized update
                scaled = Q_p * (lr * grad + r) + 0.5 / Q_p
                quantized = torch.floor(scaled) / Q_p
                
                p.data.add_(quantized)
        
        self._global_step += 1
        return loss
```

注意几个细节：
- `tau0` 是用 mini-batch count 计算，不是 epoch
- `p_bar` 用 log-log schedule
- 0.5 / Q_p 是 rounding offset（对应 Eq 5 的 0.5 * Q_p^{-1}）
- 当 tau < 2*tau0 才用 compensation（实际 implementation 简化）

---

## 12. Final Thoughts

Andrej，这个工作我觉得最 elegant 的地方在于 **发现了 quantization 在 optimization context 下的 dual role**：

> Quantization 一直被看作"压缩"工具——为了节省 storage/communication 而引入误差。但这里 quantization error 本身变成了 Langevin dynamics 需要的 Brownian motion source。

这有点像 **dropout 的 dual role**——本来是 regularization，但其实也提供了 Bayesian approximation [Gal & Ghahramani 2016]。

数学上最深的 connection 是 **CLT 把 uniform quantization error 自动变 Gaussian**，这让你免于维护 RNG，且 noise distribution 从物理结构中自然 emerge。Theoretical convergence 借用 Geman & Hwang 1986 的 SA framework，cooling schedule 与 quantization resolution 通过 log-log 增长自然匹配。

**实用的下个 step** 我觉得是：
1. 在 LLM pre-training 上 test（PyTorch 实现 6 行，应该容易 plug-in）
2. 跟 SAM、AdamW、Shampoo 等 SOTA optimizer 在大规模 benchmark 上 head-to-head
3. 与 low-precision training (FP8, INT8) 结合——如果 weight 已经 quantize 了，QSGLD 可能"免费"

如果你有 specific aspect 想深入 drill-down（例如 Girsanov theorem 部分，或 CLT 速率，或与 Levy process 的关系），告诉我，我可以更 detail 展开。

Reference 整理：
1. [Welling & Teh 2011 - SGLD](https://www.stats.ox.ac.uk/~teh/research/npbg/WelTeh2011.pdf)
2. [Goyal et al. 2018 - Large Batch Training](https://arxiv.org/abs/1706.02677)
3. [Li et al. 2019 - Stochastic Modified Equations](http://jmlr.org/papers/v20/17-526.html)
4. [Malladi et al. 2022 - SDEs for Adaptive Methods](https://proceedings.neurips.cc/paper_files/paper/2022/file/32ac710102f0620d0f28d5d05a44fe08-Paper.pdf)
5. [Geman & Hwang 1986 - Diffusions for Global Optimization](https://epubs.siam.org/doi/10.1137/0324062)
6. [Jiménez et al. 2007 - WNH](https://epubs.siam.org/doi/10.1137/060653809)
7. [Gray & Neuhoff 1998 - Quantization IEEE Survey](https://ieeexplore.ieee.org/document/1455556)
8. [Zhang et al. 2022 - Low-precision SGLD](https://proceedings.mlr.press/v162/zhang22ag.html)
9. [Raginsky et al. 2017 - Non-convex SGLD](http://proceedings.mlr.press/v65/raginsky17a.html)
10. [Øksendal - Stochastic Differential Equations](https://link.springer.com/book/10.1007/978-3-662-03620-4)
