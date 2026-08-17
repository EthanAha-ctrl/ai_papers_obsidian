---
source_pdf: Estimator Meets Equilibrium Perspective.pdf
paper_sha256: 2e02b657edf5216e47e2405b9420195c95cb8ae9640d7eaf092f638b995ea0c7
processed_at: '2026-08-04T05:10:47-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 ReSTE 这篇 paper

## 1. 先说说 BNNs 到底在搞什么

你知道 neural networks 现在又大又贵，动不动几百亿参数，训练一次烧掉几百万美金。BNNs (Binary Neural Networks) 想干的事情很简单粗暴——把所有的 weights 和 activations 从 32-bit float 砍到 1-bit，就是 +1 和 -1 两个值。

好处显而易见：model size 直接缩小 32 倍，inference 可以用 bitwise operations (XNOR + popcount) 替代 float multiplication，速度能快几十倍。坏处也明显——精度损失严重，一个 32-bit 的 weight 你硬要它变成 ±1，信息量丢了一大半。

BNNs 这个 field 的关键 question 就是：**怎么在保持 speed advantage 的前提下，把 accuracy 尽量拉回来**。

## 2. STE 这个 "dirty trick"

BNNs training 的 fundamental challenge 是一个 chicken-and-egg problem：

Forward pass 用 `sign(z)` 把 full-precision 的 `z` 变成 ±1，这没问题。但 backward pass 要算 gradient，`sign` function 的 gradient 几乎处处为 0（原点处无穷大），你 backprop 一下 gradient 全没了，network 根本学不动。

BinaryConnect (Courbariaux et al., 2015) 提出的解决方案特别 "hacky"——直接装作 `sign` function 的 gradient 是 1，把 downstream 的 gradient 原封不动传回去：

```
Forward:  z_b = sign(z)
Backward: ∂L/∂z = ∂L/∂z_b   (STE: just copy the gradient)
```

这就是 **Straight Through Estimator (STE)**。说实话这个 trick 非常 ad-hoc，forward 和 backward 用了完全不同的 function，数学上很不 elegant。但它 work，而且 surprisingly well，所以后面所有 BNNs work 基本都 build on 这个 paradigm 之上。

你可以想象 STE 就像一个 "lie"——forward 告诉你 "我是 sign function"，backward 又告诉你 "我的 gradient 是 identity"。这种 inconsistency 就是 BNNs training 的原罪。

## 3. 后人的改进方向——以及他们都掉进的那个坑

后续一堆工作意识到 STE 的 inconsistency problem，想 design better estimators 来 reduce 这个 gap。代表性的有：

- **BNN+** (Darabi et al., 2018): SignSwish function
- **Bi-Real Net** (Liu et al., ECCV 2018): piece-wise polynomial  
- **DSQ** (Gong et al., ICCV 2019): tanh-based differentiable soft quantization
- **IR-Net** (Qin et al., CVPR 2020): EDE function, `k·tanh(t·z)`
- **FDA** (Xu et al., NeurIPS 2021): Fourier series approximation
- **RBNN** (Lin et al., NeurIPS 2020): polynomial function with rotation

这些方法的共同思路：**让 estimator 在 backward pass 中更接近 sign function 的形状**，从而 reduce estimating error。

听起来 reasonable 对吧？更接近目标 function 应该更好。但这篇 paper 指出了一个所有人忽略的 fact：

> **当你 reduce estimating error 的时候，gradient stability 会 concomitantly 下降。**

Figure 2 里可视化了这个现象：IR-Net 的 gradient distribution 是 highly divergent 的——有些 gradient 特别大，有些特别小，spread 很广。相比之下 STE 的 gradient 就是 constant 1，分布非常 concentrated。

Divergent gradients 意味着什么？意味着不同 parameters 收到完全不同 magnitude 的 update，有些 parameter 被 giant step 暴力更新，有些几乎不动。这会导致：
- Training loss 剧烈 oscillate
- 部分 gradients 直接 vanish (太小)
- 部分 gradients 直接 explode (太大)
- 最终 model 学不好

**所以纯粹的 "reduce estimating error" 是一个 trap**。你把 estimator 拉得越靠近 sign function，estimating error 确实小了，但 gradients 越来越 divergent，training 越来越 unstable。这就是为什么 Figure 6 里 `o_end=10` 的时候，training loss 在 600-700 epochs 直接暴走，accuracy 从 86.75 掉到 82.86。

## 4. Equilibrium Perspective——这篇 paper 的核心 insight

这篇 paper 的核心 contribution 不是一个新的 estimator，而是一个 **perspective shift**：

> **BNNs training 本质上是 estimating error 和 gradient stability 之间的 equilibrium。**

就像一个跷跷板。你 push down 一头，另一头就翘起来。你不能无限地 reduce estimating error，因为 gradient stability 会崩。你也不能无限追求 stability（比如一直用 STE），因为 estimating error 太大，model 学不到 accurate 的东西。

为了 quantitatively demonstrate 这个 phenomenon，作者 design 了两个 indicators：

### Estimating Error Indicator

$$e = D(\text{sign}(\mathbf{z}), \mathbf{f}(\mathbf{z}))$$

变量含义：
- $e$: estimating error 的 scalar 值
- $D(\cdot)$: distance metric，论文里用 L2-norm
- $\text{sign}(\mathbf{z})$: forward pass 真实使用的 sign function 输出（"ground truth"）
- $\mathbf{f}(\mathbf{z})$: backward pass 用的 estimator function 的输出

直觉：estimator 输出和 sign function 输出离得越远，estimating error 越大。这就是之前所有 work 想优化的 target。

### Gradient Instability Indicator

$$s = \text{var}(|\mathbf{g}|)$$

变量含义：
- $s$: gradient instability 的 scalar 值
- $\mathbf{g}$: 一次 iteration 中所有 parameters 的 gradients（一个 vector）
- $|\cdot|$: element-wise absolute value
- $\text{var}(\cdot)$: variance

为什么用 absolute？因为只关心 gradient magnitude，direction 不影响 stability。variance 大意味着 gradients 发散——有些大有些小，不 uniform。

这两个 indicator 就像是跷跷板两端的 "rulers"，让你能 measure 当前的 state。Figure 4 的实验结果非常 striking：随着 $o_{\text{end}}$ 增大，estimating error 单调下降，gradient instability 单调上升，accuracy 呈倒 U 形——先升后降。倒 U 的 peak 就是 equilibrium 的 sweet spot。

## 5. ReSTE——一个 elegant 的解决方案

### 5.1 核心 idea

既然 BNNs training 是 estimating error vs. gradient stability 的 equilibrium，那我们需要一个 estimator，能 **flexibly control 这个 equilibrium 的 degree**。

作者 design 了两个 property 一个好的 estimator 应该满足：

**Property 1 (Rational):** estimator 的 estimating error 应该 always ≤ STE 的 estimating error。这很 rational——如果某个 estimator 在某些 range 比 STE 的 error 还大，那在这些 range 不如直接用 STE（更 stable 且 error 更小）。formally: $D(\text{sign}(\mathbf{z}), \mathbf{f}(\mathbf{z})) - D(\text{sign}(\mathbf{z}), \mathbf{z}) \leq 0$

**Property 2 (Flexible):** estimator 能从 STE 渐变到 sign function，且这种变化是 gradual 的（每个 point 每次只移动 small step）。

### 5.2 Power Function——为什么 work

作者的关键 observation：**STE (identity function) 是 power function 的一个 special case**。

考虑这个 function：

$$\mathbf{f}(\mathbf{z}) = \text{sign}(\mathbf{z}) \cdot |\mathbf{z}|^{\frac{1}{o}}, \quad o \geq 1$$

变量和指数解析：
- $\mathbf{f}(\mathbf{z})$: ReSTE estimator 的输出
- $\text{sign}(\mathbf{z})$: 保留 input $\mathbf{z}$ 的符号（+1 或 -1）
- $|\mathbf{z}|$: input 的 absolute value
- $\frac{1}{o}$: exponent，$o$ 是 hyperparameter
- $o$: 控制 "rectified degree"，也即 equilibrium 的 degree

Behavior 分析：
- 当 $o = 1$: $|\mathbf{z}|^{1/1} = |\mathbf{z}|$，所以 $\mathbf{f}(\mathbf{z}) = \text{sign}(\mathbf{z}) \cdot |\mathbf{z}| = \mathbf{z}$，**这就是 STE (identity function)**！
- 当 $o \to \infty$: $|\mathbf{z}|^{1/\infty} = |\mathbf{z}|^0 = 1$，所以 $\mathbf{f}(\mathbf{z}) = \text{sign}(\mathbf{z}) \cdot 1 = \text{sign}(\mathbf{z})$，**这就是 sign function**！
- $o$ 在 $[1, \infty)$ 之间连续变化，提供了一个 smooth interpolation

这非常 elegant。一个 single hyperparameter $o$ 就 control 了 estimator 从 STE 到 sign function 的 entire spectrum。

### 5.3 Gradient 推导

对 ReSTE 求导：

$$\mathbf{f}'(\mathbf{z}) = \frac{1}{o} \cdot |\mathbf{z}|^{\frac{1-o}{o}}$$

指数和变量解析：
- $\frac{1}{o}$: 前面的 scaling factor，$o$ 增大时这个 factor 减小（整体 gradient magnitude 减小）
- $\frac{1-o}{o}$: $|\mathbf{z}|$ 的 exponent
  - 当 $o = 1$: exponent $= \frac{1-1}{1} = 0$，所以 $\mathbf{f}'(\mathbf{z}) = 1 \cdot |\mathbf{z}|^0 = 1$，**constant gradient，这就是 STE**
  - 当 $o > 1$: exponent $= \frac{1-o}{o} < 0$，即 **negative exponent**
  - Negative exponent 意味着 $|\mathbf{z}|$ 越小，gradient 越大；$|\mathbf{z}|$ 越大，gradient 越小

Gradient behavior 直觉：
- 当 $|\mathbf{z}| \to 0$ (input 接近原点): $|\mathbf{z}|^{(1-o)/o} \to \infty$，gradient explode——这 mimic 了 sign function 在原点处 gradient 无穷大的特性
- 当 $|\mathbf{z}| \to \infty$ (input 很大): $|\mathbf{z}|^{(1-o)/o} \to 0$，gradient vanish——这 mimic 了 sign function 在其他处 gradient 为 0 的特性

所以 ReSTE 的 gradient 在 shape 上和 sign function 的 gradient 是 similar 的，只是 smoothed out 了。$o$ 越大，越 sharp，越接近 sign function 的真实 gradient。

### 5.4 两个 Property 的证明

**Rational property 的证明** 基于 Lemma 3.1: 如果 $o_1 \geq o_2$，则 $D(\text{sign}(\mathbf{z}), \mathbf{f}(\mathbf{z}, o_1)) \leq D(\text{sign}(\mathbf{z}), \mathbf{f}(\mathbf{z}, o_2))$

证明的 key step 是展开 L2 distance：

$$D(\text{sign}(\mathbf{z}), \mathbf{f}(\mathbf{z}, o)) = \sum_{i=1}^{d} (\text{sign}(z_i) - \text{sign}(z_i)|z_i|^{1/o})^2 = \sum_{i=1}^{d} (1 - |z_i|^{1/o})^2$$

（把 $\text{sign}(z_i)$ 提取出来，因为 $\text{sign}(z_i)^2 = 1$）

然后分两种情况：
- $|z_i| \leq 1$: $|z_i|^{1/o_1} \geq |z_i|^{1/o_2}$（因为 $o_1 \geq o_2$，exponent $1/o_1 \leq 1/o_2$，对于 $|z_i| \leq 1$ 的 base，exponent 越小值越大）
  
  Wait 让我重新想一下。For $0 < x < 1$, $x^a$ 对于 $a > 0$ 是递减的 in $a$。So $x^{1/o_1}$ vs $x^{1/o_2}$ where $o_1 \geq o_2$ means $1/o_1 \leq 1/o_2$, so $x^{1/o_1} \geq x^{1/o_2}$ (because smaller exponent for base < 1 gives larger value).
  
  So $|1 - |z_i|^{1/o_1}| = 1 - |z_i|^{1/o_1} \leq 1 - |z_i|^{1/o_2} = |1 - |z_i|^{1/o_2}|$
  
  ✅ 这是对的
  
- $|z_i| \geq 1$: $|z_i|^{1/o_1} \leq |z_i|^{1/o_2}$（因为 exponent $1/o_1 \leq 1/o_2$，对于 base > 1，exponent 越小值越小）
  
  So $|1 - |z_i|^{1/o_1}| = |z_i|^{1/o_1} - 1 \leq |z_i|^{1/o_2} - 1 = |1 - |z_i|^{1/o_2}|$
  
  ✅ 这也是对的

所以 $o$ 越大，estimating error 单调递减。因为 STE = $\mathbf{f}(\mathbf{z}, 1)$ 且 ReSTE 要求 $o \geq 1$，所以 ReSTE 的 estimating error always ≤ STE 的 estimating error。**Rational property 得证**。

**Flexible property** 也得证：$o$ 从 1 到 $\infty$ 连续变化，每增加一点，每个 $z_i$ 都 only move a small step toward sign function。这是 gradual 的。

### 5.5 Practical Tricks

为了让 training 更 stable，作者加了两个 gradient truncation tricks：

**Trick 1 - Saturation clipping:**
当 $|z| > t$ (设 $t = 1.5$) 时，把对应 gradient clip 为 0。这模拟了 BNNs 中 input 太大时 sign function 完全 saturate 的现象——input 再大，output 依然是 ±1，gradient 应该是 0。

**Trick 2 - Numerical approximation near zero:**
在 $(0, m)$ 和 $(-m, 0)$ 区间（设 $m = 0.1$），用 numerical method 替代解析 gradient：
$$\text{gradient} = \frac{f(m) - f(0)}{m} \quad \text{or} \quad \frac{f(0) - f(-m)}{m}$$

这是因为 $|\mathbf{z}| \to 0$ 时 gradient $\frac{1}{o}|\mathbf{z}|^{(1-o)/o} \to \infty$，数值上会 explode。用 secant line 的 slope 来 approximate，避免了这个 singularity。

## 6. 实验结果——为什么 ReSTE work

### 6.1 和 SOTA 的比较

Table 1 (CIFAR-10) 和 Table 2 (ImageNet) 的结果都很 impressive。我来 highlight 几个 key numbers：

**CIFAR-10, ResNet-20, 1W/1A setting:**
- RBNN (with rotation module): 86.50%
- ReSTE (no auxiliary): **86.75%** (+0.25%)

**CIFAR-10, ResNet-18, 1W/1A:**
- RBNN (with module): 92.20%
- ReSTE (no auxiliary): **92.63%** (+0.43%)

**ImageNet, ResNet-18, 1W/1A:**
- FDA (with noise adaptation module): 60.20%
- ReSTE (no auxiliary): **60.88%** (+0.68%)

**ImageNet, ResNet-34, 1W/1A:**
- LCR-BNN (with Lipschitz loss): 63.50%
- ReSTE (no auxiliary): **65.05%** (+1.55%)

Key takeaway：ReSTE 在 **没有任何 auxiliary module 或 auxiliary loss** 的情况下 surpass 所有 SOTA。其他方法都 rely on 额外的 module (rotation, noise adaptation) 或 loss (Lipschitz regularization, knowledge distillation) 来 boost performance。ReSTE 纯粹通过 design a better estimator 就 beat 它们。

### 6.2 Estimator-only 公平比较

Table 3 更 apples-to-apples——只用 estimator 本身，去掉所有 auxiliary：

| Estimator | Type | Rational | Flexible | Acc(%) |
|-----------|------|----------|----------|--------|
| STE | Identity | ✓ | ✗ | 84.44 |
| DSQ | Tanh-alike | ✗ | Little | 84.11 |
| EDE | Tanh-alike | ✗ | Little | 85.20 |
| FDA | Fourier | ✗ | Little | 85.80 |
| RBNN | Polynomial | ✗ | Little | 85.87 |
| **ReSTE** | **Power** | **✓** | **✓** | **86.75** |

ReSTE 是唯一同时满足 rational 和 flexible property 的 estimator，比第二名 RBNN 高 0.88%。

这个结果证明了两件事：
1. Rational property 很重要——比 STE error 更小是必要的
2. Flexible property 很重要——能 control equilibrium degree 让你找到 sweet spot

### 6.3 Equilibrium Phenomenon 的 demonstration

Figure 4 是这篇 paper 最 beautiful 的实验。作者在 ResNet-20, ResNet-18, VGG-small 三个 backbone 上 adjust $o_{\text{end}}$ 从小到大，同时 measure：

- Estimating error indicator $e$
- Gradient instability indicator $s$  
- Top-1 accuracy

结果：
- $e$ 随 $o_{\text{end}}$ 增大单调下降 ✅ (estimator 越来越接近 sign function)
- $s$ 随 $o_{\text{end}}$ 增大单调上升 ✅ (gradient 越来越 divergent)
- Accuracy 呈 **倒 U 形**，peak 在 $o_{\text{end}} = 3$ 附近 ✅ (equilibrium sweet spot)

而且这个 sweet spot $o_{\text{end}} = 3$ 在三个不同 backbone 上是一致的，说明 ReSTE 的 optimal configuration 是 robust 的，不需要 extensive hyperparameter tuning。

### 6.4 Divergent Gradient 的危害

Figure 6 是一个 dramatic 的 cautionary tale。当 $o_{\text{end}} = 10$ 时（太接近 sign function），training loss 在 600-700 epochs 直接剧烈 oscillate，accuracy 从 86.75% 暴跌至 82.86%。当 $o_{\text{end}}$ 进一步增大，training 直接 irreversible failure。

这 quantitative 地证明了：**纯粹追求 small estimating error 是有害的**。Gradient stability 必须被 consider。

## 7. 更 broad 的联想和 intuition building

### 7.1 和 Gumbel-Softmax 的 connection

ReSTE 的 power function trick 让我强烈联想到 Gumbel-Softmax (Jang et al., ICLR 2017) 中的 temperature parameter $\tau$：

$$p_i = \frac{\exp((g_i + \log \pi_i) / \tau)}{\sum_j \exp((g_j + \log \pi_j) / \tau)}$$

- $\tau \to 0$: distribution 变成 one-hot (discrete)
- $\tau \to \infty$: distribution 变成 uniform (continuous)
- $\tau$ 在中间: smooth interpolation

ReSTE 的 $o$ 和这里的 $\tau$ 是 **spiritually identical** 的——都是 control discrete vs. continuous 的 trade-off knob。两者都 recognize 了：training discrete structures 需要 annealing from continuous to discrete。

这种 "temperature-like" parameter 在很多地方都出现：
- **Concrete distribution** (Maddison et al., 2017): 类似 Gumbel-Softmax
- **Variational quantization** (Shayar et al., 2019): 用 temperature 控制 quantization sharpness
- **Self-supervised contrastive learning** (Chen et al., SimCLR): temperature $\tau$ 控制 distribution sharpness

ReSTE 的 $o$ 是这个 family 在 BNNs context 下的 instantiation。

### 7.2 和 Progressive Training / Curriculum Learning 的 connection

论文用 progressive strategy: $o$ 从 1 线性增到 $o_{\text{end}} = 3$。这本质上是 curriculum learning (Bengio et al., 2009) 的一个 instance：

- **Early training**: $o \approx 1$，estimator ≈ STE，gradient stable，model 先学 coarse structure
- **Late training**: $o \approx 3$，estimator 更接近 sign function，estimating error 小，model fine-tune 细节

这种 "start easy, gradually increase difficulty" 的 strategy 在 deep learning 里处处可见：
- **Residual learning**: 先学 identity mapping (easy)，再学 residual (hard)
- **ResNet stages**: early layers 学 low-level features，late layers 学 high-level
- **Warmup**: learning rate 先 small 再 large
- **Quantization-aware training** (Jacob et al., 2018): 逐渐 introduce quantization noise

ReSTE 的 progressive $o$ strategy 是这个 universal principle 在 BNNs 中的 specific application。

### 7.3 Power Function 为什么这么 natural

Power function $\text{sign}(z) \cdot |z|^{1/o}$ 其实是一个非常 natural 的 parameterization，因为它 directly controls 了 function 的 "sharpness"。

考虑 $|z| \in [0, 1]$ (input 落在 [-1, 1] 之间，这是 BNNs 中 input 的 typical range):
- $o = 1$: $|z|^1 = |z|$，linear
- $o = 2$: $|z|^{0.5} = \sqrt{|z|}$，sublinear（更 "bowed"）
- $o = 3$: $|z|^{1/3}$，更 bowed
- $o \to \infty$: $|z|^0 = 1$，step function

这就像是一个 "morphing"——linear function 逐渐 "squash" 成 step function。Power function 是 control 这种 morphing 最 natural 的 way，因为：
1. 它 preserves sign (因为 $\text{sign}(z)$ 单独提取)
2. 它 monotonic (power function 对 positive base 是 monotonic)
3. 它 smoothly connects STE ($o=1$) 和 sign ($o=\infty$)
4. 它 single-parameter (只有一个 $o$ 要 tune)

这种 elegance 让我想到 Occam's Razor——最 simple 的 parameterization 往往是最 powerful 的。

### 7.4 和 Lipschitz Continuity 的 tension

LCR-BNN (Shang et al., ECCV 2022) 通过 enforce Lipschitz continuity 来 regularize BNNs training。一个 function $f$ 是 Lipschitz continuous 如果 $|f(x_1) - f(x_2)| \leq L \cdot |x_1 - x_2|$ for some constant $L$。

ReSTE 的 gradient $|\mathbf{f}'(\mathbf{z})| = \frac{1}{o}|\mathbf{z}|^{(1-o)/o}$ 在 $|z| \to 0$ 时趋向无穷，所以 ReSTE **not** Lipschitz continuous。但作者通过 truncation tricks (clip + numerical approximation) 实现了 effectively bounded gradient。

这 suggests 一个 deep 的 tension：在 BNNs 中，完全的 Lipschitz continuity 可能太 restrictive（因为 sign function 本身不是 Lipschitz 的），但完全不 constrain gradient stability 又会 divergent。ReSTE 通过 "approximately Lipschitz" (truncate the singularity) 找到了 middle ground。

### 7.5 为什么 $o_{\text{end}} = 3$ 是 universal sweet spot

实验发现 $o_{\text{end}} = 3$ 在 ResNet-20, ResNet-18, VGG-small 三个不同 architecture 上都是 optimal。这很 striking——为什么 3 是 magic number？

我的 hypothesis: $o = 3$ 对应 exponent $1/o = 1/3$，即 cube root。Cube root function 有一些 nice properties：
- $f(0) = 0$, $f(1) = 1$, $f(-1) = -1$ (和 sign function 在关键 points 重合)
- Derivative at $z=1$: $f'(1) = \frac{1}{3} \cdot 1^{(1-3)/3} = \frac{1}{3}$ (reasonable gradient magnitude)
- 相比 $o=2$ (square root, gradient at $z=1$ is $1/2$) 和 $o=\infty$ (sign, gradient is 0)，$o=3$ 给了一个 "moderate" 的 gradient sharpness

当然这只是 retroactive rationalization，真正的 reason 可能需要更多 theoretical analysis。但 universality of $o=3$ 强烈 suggests BNNs training 有某种 universal 的 optimal equilibrium point，independent of architecture。

### 7.6 能不能 learn $o$ 而非 fix it?

论文用 fixed schedule ($o$ 从 1 到 3)，但一个 natural extension 是: **let $o$ be learnable**。

比如可以 define $o$ 为一个 parameter，让 backprop 自动 learn 它的 optimal value。或者更 sophisticated——让 $o$ 是 input-dependent 的，即 different neurons 在 different layers 用 different $o$。

这和 **adaptive temperature** 的 idea 相关，在 knowledge distillation (e.g., Ye et al., 2021) 和 contrastive learning (e.g., Wang et al., 2021) 中都有应用。BNNs 中 explore adaptive $o$ 是一个 promising future direction。

### 7.7 和 Binary Concrete / Hard Concrete Distribution 的 connection

最近 BNNs 有一个 trend 是用 **concrete distribution** (Maddison et al., 2017; Shayar et al., 2019) 来 model binarization，把 binary variables 看成从 Bernoulli distribution sample 出来的，用 concrete relaxation 让它 differentiable。

Concrete distribution 的 CDF 形式：
$$F(z) = \sigma\left(\frac{\log \alpha + z}{\tau}\right)$$

其中 $\tau$ 是 temperature，$\alpha$ 是 location parameter。$\tau \to 0$ 时变成 step function (binary)，$\tau \to \infty$ 时变成 smooth。

这和 ReSTE 的 $o$ 在 spirit 上是 similar 的——都是 temperature-like parameter control discreteness。但 ReSTE 更 simple——直接用 power function 的 single exponent，不需要 sample，deterministic。

## 8. Take-aways

用一句话总结这篇 paper：**BNNs training 是 estimating error 和 gradient stability 之间的 equilibrium，ReSTE 用一个 power function 的 single hyperparameter $o$ elegant 地 control 了这个 equilibrium，在没有任何 auxiliary 的情况下 surpass SOTA**。

更 deep 的 lessons：

1. **不要只 optimize 一个 metric**。之前所有 work 都只盯着 estimating error，忽略了 gradient stability。很多时候一个 metric 的 improvement 是以另一个 metric 的 degradation 为代价的。这种 trade-off 需要 be explicitly modeled。

2. **Simple parameterization 往往 best**。ReSTE 只有一个 hyperparameter $o$，比 IR-Net 的 multi-parameter tanh function、FDA 的 Fourier series 都 simple，但效果更好。Elegant math 优于 complex hacks。

3. **STE 作为 special case 的 insight 很 powerful**。把 STE 看成 power function family 的一个 member ($o=1$)，sign function 看成另一个 extreme ($o=\infty$)，整个 family 就 natural 出来了。这种 "generalize a special case" 的思路在 ML 中很 useful。

4. **Equilibrium / trade-off perspective 是 universal**。在 deep learning 里，太多 examples 都是 trade-off：
   - Bias vs. variance
   - Underfitting vs. overfitting  
   - Exploration vs. exploitation (RL)
   - Width vs. depth (network design)
   - Estimating error vs. gradient stability (this paper)
   
   Identifying and quantifying the trade-off 是 design better algorithms 的 prerequisite。

## 9. Reference Links

- **ReSTE paper (ACM MM 2023)**: [https://dl.acm.org/doi/10.1145/3503161.3547834](https://dl.acm.org/doi/10.1145/3503161.3547834)
- **ReSTE code (GitHub)**: [https://github.com/DravenALG/ReSTE](https://github.com/DravenALG/ReSTE)
- **BinaryConnect** (Courbariaux et al., NeurIPS 2015): [https://papers.nips.cc/paper/2015/hash/e3345be1b3e8a8c5a8e2e5f9a0e3f9a8-Abstract.html](https://papers.nips.cc/paper/2015/hash/e3345be1b3e8a8c5a8e2e5f9a0e3f9a8-Abstract.html)
- **BNN** (Hubara et al., NeurIPS 2016): [https://arxiv.org/abs/1602.02830](https://arxiv.org/abs/1602.02830)
- **XNOR-Net** (Rastegari et al., ECCV 2016): [https://arxiv.org/abs/1603.05279](https://arxiv.org/abs/1603.05279)
- **Bi-Real Net** (Liu et al., ECCV 2018): [https://arxiv.org/abs/1808.00278](https://arxiv.org/abs/1808.00278)
- **IR-Net** (Qin et al., CVPR 2020): [https://openaccess.thecvf.com/content_CVPR_2020/papers/Qin_Forward_and_Backward_Information_Retention_for_Accurate_Binary_Neural_CVPR_2020_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qin_Forward_and_Backward_Information_Retention_for_Accurate_Binary_Neural_CVPR_2020_paper.pdf)
- **DSQ** (Gong et al., ICCV 2019): [https://arxiv.org/abs/1908.05033](https://arxiv.org/abs/1908.05033)
- **RBNN** (Lin et al., NeurIPS 2020): [https://proceedings.neurips.cc/paper/2020/hash/7e426603cc5818d4e9c8f4375cc21f82-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/7e426603cc5818d4e9c8f4375cc21f82-Abstract.html)
- **FDA** (Xu et al., NeurIPS 2021): [https://proceedings.neurips.cc/paper/2021/hash/2e9ed4ae92ad4c58eec91d46ec243501-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2e9ed4ae92ad4c58eec91d46ec243501-Abstract.html)
- **LCR-BNN** (Shang et al., ECCV 2022): [https://arxiv.org/abs/2207.05751](https://arxiv.org/abs/2207.05751)
- **DoReFa-Net** (Zhou et al., 2016): [https://arxiv.org/abs/1606.06160](https://arxiv.org/abs/1606.06160)
- **Gumbel-Softmax** (Jang et al., ICLR 2017): [https://arxiv.org/abs/1611.01144](https://arxiv.org/abs/1611.01144)
- **Concrete Distribution** (Maddison et al., 2017): [https://arxiv.org/abs/1611.00712](https://arxiv.org/abs/1611.00712)
- **Curriculum Learning** (Bengio et al., ICML 2009): [https://dl.acm.org/doi/10.1145/1553374.1553380](https://dl.acm.org/doi/10.1145/1553374.1553380)
- **LSQ** (Learned Step Size Quantization, Esser et al., 2020): [https://arxiv.org/abs/1902.08153](https://arxiv.org/abs/1902.08153)
- **Quantization-Aware Training** (Jacob et al., CVPR 2018): [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)
- **BNN Survey** (Qin et al., 2022): [https://arxiv.org/abs/2202.07216](https://arxiv.org/abs/2202.07216)

---

简单总结一句：ReSTE 这篇 paper 的 beauty 在于它用一个 power function 的 single hyperparameter，elegant 地 bridges 了 STE 和 sign function 两个 extreme，并通过 equilibrium perspective 给出了一个 principled 的 tuning knob。它提醒我们，在 deep learning 中，sometimes 最 simple 的 parameterization 蕴含着最 deep 的 insight。Equilibrium / trade-off 的 perspective 是理解 many ML algorithms 的 universal lens。

---

# ReSTE: Rectified Straight Through Estimator for BNNs Training

## 1. 论文背景与动机

这篇paper来自中山大学Wu等人,发表于ACM MM 2023。它关注Binary Neural Networks (BNNs)训练中一个fundamental的问题:forward和backward过程的inconsistency。

### 1.1 BNNs的基本范式

BinaryConnect建立了BNNs的基本paradigm:forward用sign function二值化,backward用Straight Through Estimator (STE)近似梯度:

$$\text{Forward: } \mathbf{z}_b = \text{sign}(\mathbf{z})$$
$$\text{Backward: } \frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_b}$$

其中:
- $\mathbf{z}$: full-precision input (weight或activation)
- $\mathbf{z}_b$: binarized output (1-bit, 取值+1或-1)
- $\mathcal{L}$: loss function
- sign: element-wise sign function

STE的本质是identity function——它把binarized output的梯度原封不动地传给full-precision input。这种做法completely忽略了sign function本身的梯度特性(sign函数几乎处处梯度为0,在原点处梯度无穷大),相当于一个"暴力近似"。

### 1.2 Inconsistency Problem

后续工作如IR-Net的EDE、DSQ的tanh-based、FDA的Fourier series等,都试图design better estimators来减少estimating error——即让estimator更接近sign function。但这些work忽略了一个crucial fact: **当estimating error减少时,gradient stability会concomitantly下降**。

Figure 2可视化了这个问题:IR-Net虽然声称减少了estimating error,但其gradient distribution高度发散,相比之下STE的gradient分布要稳定得多。这种divergent gradient会:
- Harm model training
- Increase risk of gradient vanishing
- Increase risk of gradient exploding

## 2. 核心贡献:Equilibrium Perspective

### 2.1 关键insight

论文的核心insight是: **BNNs training本质上是estimating error和gradient stability之间的equilibrium**。

这就像一个跷跷板:
- 一端是estimating error (estimator与sign function的差距)
- 另一端是gradient stability (梯度的发散程度)
- 当你压低一端(estimating error减小),另一端就会升高(gradient instability增大)

Figure 1的intuition:当estimator越来越接近sign function,gradients变得highly divergent,最终harm training。sign function本身是gradient stability的极端——梯度要么为0(vanishing),要么无穷大(exploding)。STE则是另一个极端——梯度完全constant,最稳定但estimating error最大。

### 2.2 两个定量指标

为了quantify这个equilibrium现象,论文设计了两个indicators。

**Estimating Error Indicator:**

$$e = D(\text{sign}(\mathbf{z}), \mathbf{f}(\mathbf{z}))$$

其中:
- $D(\cdot)$: distance metric,论文用L2-norm
- $\mathbf{f}(\cdot)$: estimator function
- $\text{sign}(\mathbf{z})$: 目标(前向的真实输出)
- $\mathbf{f}(\mathbf{z})$: estimator的输出

这个indicator衡量estimator输出和sign function输出的距离——距离越小,estimating error越小。

**Gradient Instability Indicator:**

$$s = \text{var}(|\mathbf{g}|)$$

其中:
- $\mathbf{g}$: 一次iteration中所有parameters的gradients
- $|\cdot|$: element-wise absolute operation
- $\text{var}(\cdot)$: variance operator

这里用absolute value是因为只关心gradient magnitude(更新方向与stability无关)。$s$越大表示gradient越不稳定。

## 3. ReSTE: Rectified Straight Through Estimator

### 3.1 设计动机

基于equilibrium perspective,论文提出一个好的estimator应该满足两个properties:

**Property 1 - Rational Property:**

estimator的estimating error应该始终 ≤ STE的estimating error:

$$D(\text{sign}(\mathbf{z}), \mathbf{f}(\mathbf{z})) - D(\text{sign}(\mathbf{z}), \mathbf{z}) \leq 0$$

intuition: 如果某个estimator在某些range比STE的estimating error还大,那在这些range直接用STE更合理(更稳定且error更小)。

**Property 2 - Flexible Property:**

estimator能够:
1. 从STE渐变到sign function
2. 这种变化是gradual的(每个点每次只移动small step)

### 3.2 ReSTE函数

核心公式:

$$\mathbf{f}(\mathbf{z}) = \text{sign}(\mathbf{z}) |\mathbf{z}|^{\frac{1}{o}}, \quad s.t. \quad o \geq 1$$

变量解析:
- $\mathbf{f}(\mathbf{z})$: ReSTE estimator
- $\text{sign}(\mathbf{z})$: 保留input的符号(+1或-1)
- $|\mathbf{z}|^{\frac{1}{o}}$: input magnitude的power transformation
- $o$: hyperparameter,控制power,也即equilibrium的程度

关键性质:
- 当 $o=1$: $\mathbf{f}(\mathbf{z}) = \text{sign}(\mathbf{z})|\mathbf{z}| = \mathbf{z}$,这就是STE(identity function)!
- 当 $o \to \infty$: $|\mathbf{z}|^{1/o} \to 1$,所以 $\mathbf{f}(\mathbf{z}) \to \text{sign}(\mathbf{z})$,这就是sign function!
- $o$在$[1, \infty)$之间连续变化,提供了从STE到sign function的smooth transition

### 3.3 梯度推导

对ReSTE求导:

$$\mathbf{f}'(\mathbf{z}) = \frac{1}{o} |\mathbf{z}|^{\frac{1-o}{o}}$$

变量和指数解析:
- $\frac{1}{o}$: scaling factor,$o$增大时这个factor减小
- $\frac{1-o}{o}$: 这是$|\mathbf{z}|$的exponent
  - 当$o=1$: exponent = 0,所以$\mathbf{f}'(\mathbf{z}) = 1$,constant gradient(STE行为)
  - 当$o>1$: exponent = $\frac{1-o}{o} < 0$,即负指数
  - 负指数意味着$|\mathbf{z}|$越小,梯度越大; $|\mathbf{z}|$越大,梯度越小

梯度行为分析:
- 当$|\mathbf{z}| \to 0$: $|\mathbf{z}|^{(1-o)/o} \to \infty$,梯度exploding risk
- 当$|\mathbf{z}| \to \infty$: $|\mathbf{z}|^{(1-o)/o} \to 0$,梯度vanishing(saturation effect)
- 这种行为在某种程度上mimics了sign function的梯度特性(原点处无穷大,其他处为0),但是是smooth version

### 3.4 Lemma 3.1证明直觉

**Lemma 3.1:** 如果 $o_1 \geq o_2$,则 $D(\text{sign}(\mathbf{z}), \mathbf{f}(\mathbf{z}, o_1)) \leq D(\text{sign}(\mathbf{z}), \mathbf{f}(\mathbf{z}, o_2))$

证明的核心:

$$D(\text{sign}(\mathbf{z}), \mathbf{f}(\mathbf{z}, o)) = \sum_{i=1}^{d} |\text{sign}(z_i) - \text{sign}(z_i)|z_i|^{1/o}|^2 = \sum_{i=1}^{d} |1 - |z_i|^{1/o}|^2$$

关键observation: 对任意$z_i$,当$o$增大时:
- 若$|z_i| \leq 1$: $|z_i|^{1/o}$随着$o$增大而增大(趋近于1),所以$|1-|z_i|^{1/o}|$减小
- 若$|z_i| \geq 1$: $|z_i|^{1/o}$随着$o$增大而减小(趋近于1),所以$|1-|z_i|^{1/o}|$减小

因此$o$增大 ⟹ estimating error单调递减。这个lemma保证了:
1. **Rational property**: 因为$o \geq 1$且STE = $\mathbf{f}(\mathbf{z}, 1)$,所以ReSTE的error ≤ STE的error
2. **Flexible property**: $o$从1到$\infty$连续变化,error单调递减,是gradual的

### 3.5 梯度截断Tricks

为增强训练稳定性,论文加了两个truncation tricks:

**Trick 1 - Saturation clipping:**
当$|z| > t$ (threshold,设$t=1.5$)时,将对应梯度clip为0。这模拟了BNNs中的saturation现象。

**Trick 2 - Numerical approximation near zero:**
在$(0, m)$和$(-m, 0)$区间($m=0.1$),用数值方法替代解析梯度:
$$\frac{f(m) - f(0)}{m} \quad \text{和} \quad \frac{f(0) - f(-m)}{m}$$

这是因为$|\mathbf{z}| \to 0$时梯度趋向无穷大,数值方法可以avoid这个singularity。

## 4. 实验结果分析

### 4.1 CIFAR-10结果

Table 1展示了CIFAR-10上的结果:

| Backbone | Method | W/A | Auxiliary | Acc(%) |
|----------|--------|-----|-----------|--------|
| ResNet-20 | RBNN | 1/1 | Module | 86.50 |
| ResNet-20 | **ReSTE** | 1/1 | - | **86.75** |
| ResNet-20 (Bi-Real) | RBNN | 1/1 | Module | 87.50 |
| ResNet-20 (Bi-Real) | **ReSTE** | 1/1 | - | **87.92** |
| ResNet-18 | RBNN | 1/1 | Module | 92.20 |
| ResNet-18 | **ReSTE** | 1/1 | - | **92.63** |

ReSTE在无任何auxiliary的情况下超越所有SOTA,包括使用了额外module的RBNN和额外loss的LCR-BNN。

### 4.2 ImageNet结果

Table 2展示ImageNet结果:

| Backbone | Method | W/A | Top-1(%) |
|----------|--------|-----|----------|
| ResNet-18 | FDA | 1/1 | 60.20 |
| ResNet-18 | **ReSTE** | 1/1 | **60.88** |
| ResNet-34 | LCR-BNN | 1/1 | 63.50 |
| ResNet-34 | **ReSTE** | 1/1 | **65.05** |

ReSTE在ResNet-34上达到65.05% Top-1 accuracy,相比LCR-BNN提升1.55%。

### 4.3 Estimator公平比较

Table 3在相同设置下比较不同estimators:

| Estimator | Type | Rational | Flexible | Acc(%) |
|-----------|------|----------|----------|--------|
| STE | Identity | ✓ | ✗ | 84.44 |
| DSQ | Tanh-alike | ✗ | Little | 84.11 |
| EDE | Tanh-alike | ✗ | Little | 85.20 |
| FDA | Fourier | ✗ | Little | 85.80 |
| RBNN | Polynomial | ✗ | Little | 85.87 |
| **ReSTE** | **Power** | **✓** | **✓** | **86.75** |

ReSTE是唯一同时满足rational和flexible property的estimator,且性能最优。

## 5. Equilibrium Analysis

Figure 4通过调整$o_{\text{end}}$来可视化equilibrium现象:

- **Estimating error**: 随$o_{\text{end}}$增大而单调递减
- **Gradient instability**: 随$o_{\text{end}}$增大而单调递增
- **Accuracy**: 先增后减,呈倒U形

这个倒U形曲线完美诠释了equilibrium——太小$o$(接近STE)error太大,太大$o$(接近sign)gradient太divergent。最优$o_{\text{end}}=3$在所有backbone上一致,显示robustness。

Figure 6展示了一个dramatic example:当$o_{\text{end}}=10$时,training loss在600-700 epochs剧烈波动,accuracy从86.75%暴跌至82.86%,验证了divergent gradient的危害。

## 6. 更广泛的联系与思考

### 6.1 与Gumbel-Softmax的联系

ReSTE的power function trick让我联想到Gumbel-Softmax中的temperature parameter $\tau$。Gumbel-Softmax:

$$p_i = \frac{\exp((g_i + \log \pi_i)/\tau)}{\sum_j \exp((g_j + \log \pi_j)/\tau)}$$

$\tau \to 0$时逼近离散分布,$\tau \to \infty$时逼近uniform。这和ReSTE中$o$控制discrete-continuous的trade-off是spiritually similar的。

### 6.2 与Lipschitz连续性的关系

LCR-BNN通过Lipschitz continuity来约束训练。ReSTE的梯度:

$$|\mathbf{f}'(\mathbf{z})| = \frac{1}{o} |\mathbf{z}|^{(1-o)/o}$$

在$|z| \to 0$时无界,所以ReSTE不是Lipschitz continuous的。但通过truncation tricks(clip和numerical approximation),实际实现了bounded gradient。这暗示了一个更deep的issue:在BNNs中完全的Lipschitz continuity可能太restrictive。

### 6.3 Progressive Adjusting Strategy

论文采用progressive strategy:训练过程中$o$从1线性增加到$o_{\text{end}}=3$。这和curriculum learning的思想相通——先学简单(STE-like, stable),再逐渐增加难度(sign-like, accurate)。

这种策略在quantization-aware training中很常见,比如LSQ( Learned Step Size Quantization)也用类似的warmup。

### 6.4 对BNNs未来的启示

ReSTE的成功暗示了几个direction:
1. **Estimator design**: 不应该只追求逼近sign function,要考虑gradient dynamics
2. **Adaptive $o$**: 是否可以learn $o$而非用固定schedule?
3. **Connection to straight-through Gumbel**: 能否用更sophisticated的relaxation?

## 7. 参考资源

- **论文PDF**: [ACM MM 2023](https://dl.acm.org/doi/10.1145/3503161.3547834)
- **代码**: [GitHub - ReSTE](https://github.com/DravenALG/ReSTE)
- **BinaryConnect** (Bengio et al., NeurIPS 2015): [arXiv](https://arxiv.org/abs/1511.00363)
- **IR-Net** (Qin et al., CVPR 2020): [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qin_Forward_and_Backward_Information_Retention_for_Accurate_Binary_Neural_CVPR_2020_paper.pdf)
- **RBNN** (Lin et al., NeurIPS 2020): [paper](https://proceedings.neurips.cc/paper/2020/hash/7e426603cc5818d4e9c8f4375cc21f82-Abstract.html)
- **FDA** (Xu et al., NeurIPS 2021): [paper](https://proceedings.neurips.cc/paper/2021/hash/2e9ed4ae92ad4c58eec91d46ec243501-Abstract.html)
- **LCR-BNN** (Shang et al., ECCV 2022): [paper](https://link.springer.com/chapter/10.1007/978-3-031-19797-0_10)
- **DoReFa-Net** (Zhou et al., 2016): [arXiv](https://arxiv.org/abs/1606.06160)
- **Bi-Real Net** (Liu et al., ECCV 2018): [arXiv](https://arxiv.org/abs/1808.00278)
- **BNN survey** (Hubara et al.): [JMLR](http://jmlr.org/papers/v18/16-456.html)
- **Gumbel-Softmax** (Jang et al., ICLR 2017): [arXiv](https://arxiv.org/abs/1611.01144)
- **LSQ** (Learned Step Size Quantization): [paper](https://arxiv.org/abs/1902.08153)

ReSTE的beauty在于它的simplicity——用一个power function的single hyperparameter $o$就elegant地bridges了STE和sign function两个极端,并通过equilibrium perspective给出了一个principled的tuning knob。这提醒我们,在deep learning中,sometimes最simple的parameterization蕴含着最deep的insight。
