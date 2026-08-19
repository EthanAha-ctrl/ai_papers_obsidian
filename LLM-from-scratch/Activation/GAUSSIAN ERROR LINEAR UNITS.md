---
source_pdf: GAUSSIAN ERROR LINEAR UNITS.pdf
paper_sha256: ef59863c8e36d4b64c43ced6b399a95412b4f232ff1ce83e84cf10b8a786b6a8
processed_at: '2026-08-19T08:42:01-07:00'
target_folder: LLM-from-scratch/Activation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GELU 用人话讲

## 一句话版本

GELU 就是 "输入越大越保留，输入越小越丢弃" 这个直觉的数学化。

## 三个 activation 的区别

想象你在过滤一批学生，标准是考试成绩：

**ReLU** 是个一刀切的家伙：60分以上全通过，60分以下全淘汰。没有中间地带，60分和100分待遇一样，59分和0分待遇一样。简单粗暴但 effective。

**ELU** 也是及格线思路，但不及格的人不是直接赶走，而是扣分后留下。负数区域有 curvature，让函数在原点附近 smooth。

**GELU** 完全换了个角度：不设固定及格线，而是看这个分数在所有人里排多少。考90分，保留概率99%；考50分，保留概率50%；考10分，保留概率4%。每个人有一个 retain probability，分数越高保留越多。然后取 expectation 就得到 deterministic output。

这个概率怎么算？用 standard normal distribution 的 CDF `Φ(x)`。

## 公式拆解

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

- `x`：neuron 的 pre-activation，假设服从 `N(0,1)`
- `Φ(x)`：standard normal CDF，含义是 `P(X ≤ x)` where `X ~ N(0,1)`
- `erf`：error function，标准数学函数，`erf(z) = (2/√π) ∫_0^z e^{-t²} dt`
- `√2`：把 N(0,1) 的 CDF 用 erf 表示时的标准化常数
- `1/2`：erf 输出范围 [-1,1]，加1再除2映射到 [0,1]

**人话**：output = input 乘以 "input 在正态分布下的百分位数"。

## 近似公式（实际工程用的）

精确的 erf 计算比较慢，paper 提供两个近似：

### Tanh 近似（paper 实际用）

$$\text{GELU}(x) \approx 0.5x\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right]$$

- `√(2/π) ≈ 0.7979`：让 tanh 在原点附近斜率匹配 erf
- `0.044715`：三次项修正系数，提升负值区域精度
- `x³`：提供曲率
- 这个公式 PyTorch 默认实现用 `approximate='tanh'`

### Sigmoid 近似（更快但精度低）

$$\text{GELU}(x) \approx x \cdot \sigma(1.702x)$$

- `σ(z) = 1/(1+e^{-z})`：sigmoid function
- `1.702`：让 sigmoid 在原点附近斜率匹配 `Φ(x)`
- 简单到一个 sigmoid call，适合推理速度敏感场景

## GELU 长什么样

形状关键点：
- `x = 0`：GELU(0) = 0 × 0.5 = 0，但斜率 = 0.5（ReLU 斜率是1，ELU 是1）
- `x = 2`：GELU(2) ≈ 2 × 0.977 = 1.954
- `x = -2`：GELU(-2) ≈ -2 × 0.023 = -0.046（几乎为0但有微小负值）
- `x ≈ -0.75`：GELU 有 minimum ≈ -0.17（一个 small negative bump）
- `x → +∞`：GELU(x) → x（与 ReLU 一致）
- `x → -∞`：GELU(x) → 0（与 ReLU 一致）

所以 GELU 在正数区接近 `y=x` 但有 slight curvature，在负数区有一个 small dip 然后趋于 0。它是 ReLU 的 smooth 版。

## 为什么 work：四个直觉

### 1. Self-regularization via stochastic origin

GELU 来源于 Adaptive Dropout 的 expectation。每个 input 有 probability 被乘 0，probability 正比于 `Φ(x)`。这相当于 network 内嵌了一个 input-dependent dropout——越小的 input 越容易被 drop。这种 self-gating implicit regularizer 与 explicit dropout 互补，可以叠加使用。

### 2. Smooth gradient flow

ReLU 在 `x=0` 处不可导，gradient 要么全有要么全无。GELU 在所有点都 smooth 可导，特别是 `x=0` 附近 gradient 是连续变化的。这让 gradient-based optimizer 更 stable，尤其 deep network。

### 3. Curvature everywhere

ReLU 和 ELU 在 `x>0` 区域是线性的（curvature = 0）。GELU 在所有点都有 non-zero curvature（因为 erf 的导数是 Gaussian PDF，always > 0）。这意味着 positive input 也会被 non-linearly transform。paper 论证这能让 network 更 easily approximate complicated functions。

### 4. Probabilistic interpretation

GELU 是 stochastic regularizer `m·x` where `m ~ Bernoulli(Φ(x))` 的 expectation。这给 activation function 一个 statistical meaning，而非 arbitrary engineering trick。BatchNorm 之后的 neuron input 接近 `N(0,1)` distribution，正好让 `Φ(x)` 的概率解释成立。

## 实验数据一览

| Task | GELU | ReLU | ELU |
|------|------|------|-----|
| MNIST Classification (8-layer MLP) | 最低 training loss | — | — |
| MNIST Autoencoder | 最低 reconstruction error | — | — |
| Twitter POS Tagging | **12.57%** | 12.67% | 12.91% |
| TIMIT Frame Classification | **29.3%** | 29.5% | 29.6% |
| CIFAR-10 (9-layer CNN) | **7.89%** | 8.16% | 8.41% |
| CIFAR-100 (Wide ResNet 40-4) | **20.74%** | 21.77% | 22.98% |

每个 task GELU 都 win，margin 通常 0.1-1.3%。

## CIFAR-10 详细 architecture（appendix A）

| Layer | Type | Channels | Size |
|-------|------|----------|------|
| 0 | Input RGB + ZCA whitening | 3 | 32 |
| 1 | Gaussian noise σ=0.15 | 3 | 32 |
| 2 | 3×3 conv + activation | 96 | 32 |
| 3 | 3×3 conv + activation | 96 | 32 |
| 4 | 3×3 conv + activation | 96 | 32 |
| 5 | 2×2 max pool stride 2 | 96 | 16 |
| 6 | dropout p=0.5 | 96 | 16 |
| 7 | 3×3 conv + activation | 192 | 16 |
| 8 | 3×3 conv + activation | 192 | 16 |
| 9 | 3×3 conv + activation | 192 | 16 |
| 10 | 2×2 max pool stride 2 | 192 | 8 |
| 11 | dropout p=0.5 | 192 | 8 |
| 12 | 3×3 conv + activation | 192 | 6 |
| 13 | 1×1 conv + activation | 192 | 6 |
| 14 | 1×1 conv + activation | 192 | 6 |
| 15 | global average pool | 192 | 1 |
| 16 | softmax output | 10 | 1 |

Optimizer: Adam, 200 epochs, LR 在 epoch 100 后线性衰减到 0。No data augmentation。

## CIFAR-100 Wide ResNet 细节

- Depth 40, widening factor 4（40-4 WRN）
- Block 顺序：Conv-Activation-Conv-Activation-BatchNorm
- 这个顺序是为了对 ELU 友好（ELU 在 residual network 有 exploding gradient 问题，参考 Shah et al. 2016）
- Optimizer: Nesterov momentum, SGDR schedule (T₀=50, η=0.1)
- Dropout keep p=0.7
- 原始 40-4 WRN with ReLU 是 22.89%，paper 的 architecture 调整本身让 ReLU 也涨到 21.77%，但 GELU 仍然领先到 20.74%

## 与 ReLU / ELU 的渐近关系

paper 给出几个 beautiful 的 limit：

**GELU → ReLU 当 σ → 0**

如果用 `N(μ, σ²)` 的 CDF 而非 standard normal，当 `σ → 0` 且 `μ = 0`，CDF 趋于 step function `1_{x>0}`，于是 `xΦ(x) → x·1_{x>0} = ReLU(x)`。GELU 是 ReLU 的 smooth relaxation，σ=1 是一个 smoothness 选择。

**GELU with Cauchy CDF → ELU**

用 standard Cauchy distribution 的 CDF，`x P(C ≤ x)` 在 `x<0` 区域 asymptotically 等于 `ELU(x)` when `α = 1/π`。

## SiLU / Swish 的故事

paper Appendix B 讲述了一段 academic drama：

- 2016 June：Hendrycks 在本 paper 提出 `SiLU = xσ(x)`，作为 GELU 的 sigmoid 近似变体
- 2017 early：Elfwing et al. 发表相同公式，命名 "SIL"
- 2017 late：Google Brain 的 Ramachandran et al. 发表 "swish" = `xσ(βx)`，**未引用前两篇**
- 被指出后，作者承认 *"we missed prior works..."*
- 更新版加入 `β` hyperparameter 声称 novelty，但 community 实际用 `β=1` 等于 SiLU
- TensorFlow/PyTorch 最初命名为 "swish"
- Reddit 帖子 "Google has a credit assignment problem in research" 引起关注
- 最终 TF/PyTorch 把 "swish" 重命名为 "SiLU"，承认 Hendrycks 的 priority

GELU 通过 BERT/GPT 采用成为 Transformer 事实标准，比 SiLU 更流行。

## GELU 在 Transformer 中的应用

BERT、GPT-2/3/4、ViT、T5 都用 GELU 作为 FFN 层 activation。典型结构：

```
x → LayerNorm → Linear(d → 4d) → GELU → Linear(4d → d) → + residual
```

为什么 Transformer 选 GELU：
1. Smoothness 让 self-attention 之后的 FFN training 更 stable
2. Negative values 帮助 LayerNorm 之后的 distribution centering
3. 大规模 pretraining 实证效果好
4. LayerNorm 制造了 `N(0,1)` distribution，让 GELU 的 probabilistic interpretation 成立

## SwiGLU 是 GELU 思想的延伸

Gated Linear Unit (GLU) 系列与 GELU 在概念上类似——都是 input-dependent gating。GLU 用 `x * σ(Wx + b)` 作为 gating，让 gate 有 explicit learned parameters。后来 **SwiGLU** (Shazeer 2020) = Swish-Gated Linear Unit 被 LLaMA 采用，可以看作把 GELU 的 mask probability `Φ(x)` 替换成 `σ(Wx)`，让 gate 有 learned capacity。

参考：
- GELU paper: https://arxiv.org/abs/1606.08415
- BERT: https://arxiv.org/abs/1810.04805
- GPT-2: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- ViT: https://arxiv.org/abs/2010.11929
- SwiGLU: https://arxiv.org/abs/2002.05202
- LLaMA: https://arxiv.org/abs/2302.13971
- Mish (related non-monotonic activation): https://arxiv.org/abs/1908.08681
- Adaptive Dropout: https://papers.nips.cc/book/advances-in-neural-information-processing-systems-26-2013
- Zoneout: https://arxiv.org/abs/1606.01305
- Wide ResNet: https://arxiv.org/abs/1605.07146
- PyTorch GELU doc: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
- TensorFlow GELU: https://www.tensorflow.org/api_docs/python/tf/nn/gelu

## 实战 tips

1. **用 momentum optimizer**：Adam 或 SGD with momentum。纯 SGD 训 GELU 不稳定
2. **用高质量近似**：`xσ(x)` (SiLU) 虽然能用但略差，paper 实验都用 tanh 近似 `0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])`
3. **配 BatchNorm / LayerNorm 使用**：normalization 让 input 接近 `N(0,1)`，让 GELU 的 probabilistic 假设成立
4. **不需要额外 hyperparameter**：不像 ELU 有 α，不像 Swish 有 β，GELU 直接用 `μ=0, σ=1`

## 我的几点联想

### 为什么 GELU 比 SiLU 略好？

`Φ(x)` 比 `σ(x)` 在 tail 区域 decay 更快。Gaussian distribution 比 Logistic distribution tail 更薄，所以 `Φ(x)` 在 `|x|` 大时更接近 step function，这意味着 GELU 对 large positive input 几乎全保留，对 large negative input 几乎全 drop，behavior 更接近 ReLU。SiLU 的 sigmoid tail 更厚，在 large negative input 时还保留一些 magnitude，可能 less robust。

### Variance 视角

GELU 的 stochastic origin 是 `m·x` where `m ~ Bernoulli(Φ(x))`。这个 random variable 的 variance 是：

$$\text{Var}[m \cdot x] = x^2 \Phi(x)(1 - \Phi(x))$$

- `x`：input
- `Φ(x)(1-Φ(x))`：Bernoulli variance
- 这个 variance 在 `x=0` 处达到 maximum `= 0.25x²`
- `|x|→∞` 时 variance → 0

直觉：near-zero input 的 stochasticity 最大，大 magnitude input 接近 deterministic。这与 Bayesian neural network 的 epistemic uncertainty 概念有联系——network 对 near-zero input 最不确定，对 extreme input 最确定。

### Non-monotonic 是关键

GELU 的 non-monotonicity（在 `x ≈ -0.75` 处有 minimum）让它能 represent 更复杂的 function class。传统 wisdom 是 activation 应该 monotonic（这样 network 更容易 train），但 GELU 实证打破了这一点。后来的 Mish `x·tanh(softplus(x))` 也是 non-monotonic 且表现好，暗示 "non-monotonic + smooth + small negative bump" 是 high-performing activation 的一个 design pattern。

### 与 Mish 的对比

Mish (Misra 2019) = `x · tanh(ln(1+e^x))`。形状与 GELU 极为相似：正数区接近 `y=x` 有 slight curvature，负数区有 small dip 趋于 0。两者 minimum 都在 `x ≈ -0.75` 附近，depth ≈ -0.17。Mish 在某些 task 上与 GELU 持平甚至略好，但 GELU 有 probabilistic interpretation 加持，加上 BERT/GPT 采用，ecosystem 优势更大。

参考: https://arxiv.org/abs/1908.08681

### GELU 在 RL 中可能不理想

Paper 没 evaluate RL。informal observation 显示 GELU 在 RL value function approximation 中不如 ReLU 稳定。可能因为 non-monotonicity 让 value function 的 monotonic prior 被打破，TD learning 对 activation 的 non-monotonicity 更敏感。这是一个 open question。

### Learnable GELU？

Paper 提到可以用 `N(μ, σ²)` CDF 并让 `μ, σ` 可学，但实验用 `μ=0, σ=1`。后来 Swish 的 `β` 参数是这个思路的变种。Community 共识是 `β=1` 已经足够好，learnable activation function 的 marginal benefit 不大。但理论上，不同 layer 用不同 `σ` 可能 helpful——浅层用大 `σ`（更 smooth），深层用小 `σ`（更接近 ReLU）。

## 总结

GELU 把三个看似独立的东西 unified：
1. ReLU 的 deterministic sign gating
2. Dropout 的 stochastic regularization  
3. Probability distribution 的 CDF

通过 expectation 把它们凝结成 `xΦ(x)`，simple、smooth、no extra hyperparameter。加上 BERT/GPT/ViT 的实证背书，GELU 已经从 2016 年的 interesting alternative 变成 2026 年的 default choice。

---

# Gaussian Error Linear Units (GELU) 深度解析

Andrej，这篇 2016 年的 paper 虽然 short，但 impact 巨大——它定义了后来 BERT、GPT 系列乃至整个 Transformer 生态的默认 activation function。让我从多个 angle 拆解。

## 1. Motivation: 把 Dropout 与 ReLU 统一起来

paper 的核心 insight 在于：**ReLU 与 Dropout 本质上都是 "input × mask" 操作，只是 mask 的来源不同**。

| Mechanism | Mask m | 决定方式 |
|-----------|--------|----------|
| ReLU | 1_{x>0} | 确定性，依赖 input sign |
| Dropout | Bernoulli(1-p) | 随机，独立于 input |
| Zoneout | Bernoulli(1-p) | 随机，保留 input |
| **GELU** | Bernoulli(Φ(x)) | 随机 + 依赖 input value |

这里 `Φ(x) = P(X ≤ x), X ~ N(0,1)` 是 standard normal CDF。直觉是：input 越大，被 retain 的概率越高；input 越小（特别是负值），被 drop 的概率越高。这 bridge 了 deterministic gating (ReLU) 与 stochastic regularization (Dropout) 两个本来独立的 mechanism。

参考链接：
- 原 paper: https://arxiv.org/abs/1606.08415
- Adaptive Dropout (Ba & Frey): https://papers.nips.cc/book/advances-in-neural-information-processing-systems-26-2013
- Zoneout (Krueger et al.): https://arxiv.org/abs/1606.01305

## 2. 数学推导：从 stochastic 到 deterministic

paper 给的 derivation 值得仔细展开。考虑一个 input `x`，我们 draw 一个 mask `m ~ Bernoulli(Φ(x))`，然后做 `m * x`。求 expectation：

$$\mathbb{E}[m \cdot x] = \Phi(x) \cdot x + (1 - \Phi(x)) \cdot 0 = x\Phi(x)$$

变量含义：
- `x`: neuron 的 pre-activation input（标量）
- `m ∈ {0,1}`: 随机 mask
- `Φ(x)`: standard normal CDF，即 `P(X ≤ x)` where `X ~ N(0,1)`
- `E[m] = Φ(x)`: mask 为 1 的概率

由于 `Φ(x) = (1/2)[1 + erf(x/√2)]`，所以：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

- `x`: input
- `erf`: error function, `erf(z) = (2/√π) ∫_0^z e^{-t²} dt`
- `1/√2`: 标准化常数，把 N(0,1) 的 CDF 用 erf 表示
- 整个表达式的含义：output = input × (input 比其他 N(0,1) sample 大的概率)

**直观看**：GELU 不是 sign gating，而是 magnitude gating。一个 input `x=2` 几乎总会被保留 (Φ(2)≈0.977)；`x=0` 有一半概率被 drop (Φ(0)=0.5)；`x=-2` 几乎总被 drop (Φ(-2)≈0.023)。

## 3. 近似公式：feedforward speed vs exactness

paper 提供两个近似：

### 3.1 Tanh 近似（paper 实际使用）

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

变量与系数：
- `x`: input
- `0.5`: 把 tanh 输出从 [-1,1] 映射到 [0,1]
- `√(2/π) ≈ 0.7979`: 缩放因子，让 tanh 的斜率在原点附近匹配 erf
- `0.044715`: 三次项系数，用于在 |x| 较大时修正近似误差
- `x³`: 提供曲率，让近似在负值区域更精确

这个近似来源于 Choudhury (2014) 对 normal CDF 的 tanh approximation。

### 3.2 Sigmoid 近似（更快速）

$$\text{GELU}(x) \approx x\sigma(1.702x)$$

- `σ(z) = 1/(1+e^{-z})`: sigmoid
- `1.702`: 让 sigmoid(1.702x) 在原点附近匹配 Φ(x) 的斜率
- 这个 variant 极其简单，只需要一个 sigmoid 调用

这两个近似在 PyTorch / TensorFlow 中是 default 实现。注意：原 paper 的所有实验用的是 tanh 近似。

参考: https://arxiv.org/abs/1606.08415v3 (附录中的近似推导)

## 4. 与 ReLU / ELU 的关系

paper 第 4 节给出几个 beautiful 的 asymptotic relation：

### 4.1 GELU → ReLU 当 σ → 0

如果用 `N(μ, σ²)` 的 CDF 而非 standard normal，当 `σ → 0` 且 `μ = 0` 时，CDF 趋向于 step function `1_{x>0}`，于是：

$$\lim_{\sigma \to 0} x\Phi_{\mu=0,\sigma}(x) = x \cdot \mathbb{1}_{x>0} = \text{ReLU}(x)$$

这意味着 **GELU 是 ReLU 的 smooth relaxation**，σ=1 是一个特定的 smoothness 选择。

### 4.2 GELU 与 Cauchy CDF → ELU

如果用 standard Cauchy 分布 `C ~ Cauchy(0,1)` 的 CDF，那么对于 `x < 0`：

$$x P(C \leq x) \sim \text{ELU}(x) \quad \text{当 } \alpha = 1/\pi$$

ELU 定义：`ELU(x) = x` if `x>0`, else `α(e^x - 1)`。

### 4.3 关键差异表

| Property | ReLU | ELU | GELU |
|----------|------|-----|------|
| Monotonic | ✓ | ✓ | ✗ |
| Convex | ✓ | ✓ | ✗ |
| Linear in positive domain | ✓ | ✓ | ✗ (has curvature) |
| Negative outputs | ✗ | ✓ | ✓ |
| Probabilistic interpretation | ✗ | ✗ | ✓ |
| Smooth at x=0 | ✗ | ✓ | ✓ |
| Non-zero curvature everywhere | ✗ | ✗ | ✓ |

non-monotonic + non-convex + 全局曲率 是 GELU 比 ReLU/ELU 更 expressive 的关键。负值的 small bump 让网络能学到 subtle non-linear patterns。

## 5. 实验数据详解

### 5.1 MNIST Classification

- Architecture: 8-layer MLP, 128 neurons/layer
- Optimizer: Adam, batch size 128, 50 epochs
- Weight init: unit norm rows
- LR search: {10⁻³, 10⁻⁴, 10⁻⁵}
- Result: GELU 最低 training log loss（with/without dropout p=0.5）

### 5.2 MNIST Autoencoder

- Architecture: 1000-500-250-30-250-500-1000 (hourglass)
- Loss: MSE
- Optimizer: Adam, batch 64
- LR: 10⁻³ 和 10⁻⁴ 都试了（10⁻² 时 ELU diverge）
- GELU 显著 outperform 其他

### 5.3 Twitter POS Tagging

- Dataset: 25 tags, 1000 train / 327 val / 500 test tweets
- Architecture: 2-layer, 256 neurons/layer
- Pretrained word vectors (56M tweets corpus)
- Input: concat(word, left_neighbor, right_neighbor)
- Dropout keep p=0.8

| Activation | Test Error |
|-------------|-----------|
| GELU | **12.57%** |
| ReLU | 12.67% |
| ELU | 12.91% |

### 5.4 TIMIT Frame Classification

- Dataset: 680 speakers, 3696/1152/192 sentences
- Architecture: 5-layer, 2048 neurons, 39 output phones
- Input: 11 frames × 26 MFCC features (energy + derivatives)
- Dropout p=0.5

| Activation | Test Error |
|------------|-----------|
| GELU | **29.3%** |
| ReLU | 29.5% |
| ELU | 29.6% |

### 5.5 CIFAR-10 (shallow 9-layer CNN)

paper appendix A 给出 architecture：

| Layer | Type | Channels | Size |
|-------|------|----------|------|
| Input | RGB + ZCA whitening | 3 | 32 |
| Noise | Gaussian σ=0.15 | 3 | 32 |
| Conv1 | 3×3 + activation | 96 | 32 |
| Conv2 | 3×3 + activation | 96 | 32 |
| Conv3 | 3×3 + activation | 96 | 32 |
| Pool | 2×2 max, stride 2 | 96 | 16 |
| Dropout | p=0.5 | 96 | 16 |
| Conv4 | 3×3 + activation | 192 | 16 |
| Conv5 | 3×3 + activation | 192 | 16 |
| Conv6 | 3×3 + activation | 192 | 16 |
| Pool | 2×2 max, stride 2 | 192 | 8 |
| Dropout | p=0.5 | 192 | 8 |
| Conv7 | 3×3 + activation | 192 | 6 |
| Conv8 | 1×1 + activation | 192 | 6 |
| Conv9 | 1×1 + activation | 192 | 6 |
| Pool | global avg | 192 | 1 |
| Output | softmax | 10 | 1 |

- Optimizer: Adam, 200 epochs, LR linearly decays to 0 at epoch 100
- No data augmentation
- Result: GELU 7.89%, ReLU 8.16%, ELU 8.41%

### 5.6 CIFAR-100 (Wide ResNet 40-4)

- Architecture: Wide ResNet, depth 40, widening factor 4
- Block: Conv-Activation-Conv-Activation-BatchNorm（这个 block 顺序对 ELU 友好，因为 ELU 在 residual network 中有 exploding gradient 问题，参考 Shah et al. 2016）
- Optimizer: Nesterov momentum, 50 epochs
- Schedule: SGDR with T₀=50, η=0.1
- Dropout keep p=0.7

| Activation | Test Error |
|------------|-----------|
| GELU | **20.74%** |
| ReLU | 21.77% |
| ELU | 22.98% |

注意：original 40-4 WideResNet with ReLU 是 22.89%，所以 paper 的 architecture 调整（BatchNorm at end of block）本身让 ReLU 也变好，但 GELU 仍然领先。

参考: https://arxiv.org/abs/1605.07146 (Wide Residual Networks)

## 6. 为什么 GELU work？Intuition Building

### 6.1 Self-gating 与 pseudo-ensemble

GELU 可以看作 Adaptive Dropout 的 expectation form。Adaptive Dropout (Ba & Frey 2013) 用 logistic 分布的 CDF `σ(wx+b)` 作为 retain probability。GELU 用 standard normal CDF。两者都是 input-dependent mask 的 expectation，但 GELU 让 mask probability 直接由 `x` 决定，没有额外参数。

### 6.2 Soft attention to magnitude

ReLU 是 hard threshold：x > 0 全保留，x ≤ 0 全 kill。GELU 是 soft threshold：x 大就保留多，x 小就保留少，但中间有个 smooth transition。这种 soft gating 让 gradient flow 更 smooth，尤其对于 near-zero 的 input。

### 6.3 Curvature everywhere

ReLU 在 x>0 区域是线性的，曲率为 0；在 x=0 处不连续。ELU 在 x>0 也是线性的。GELU 在所有点都有 non-zero curvature（erf 的导数是 Gaussian PDF，always > 0）。这意味着 GELU 在 positive region 也能学到 non-linear transformation，让 deeper network 更容易 express complex functions。

### 6.4 Negative bump 作为 regularizer

GELU 在 x ≈ -0.75 处有一个 small negative dip（大约 -0.17）。这类似于 ELU 的 negative branch，提供 slight mean-shift 让 network 更 robust。paper 的 robustness experiment（Figure 3）显示 GELU 对 uniform noise `Unif[-a,a]` 的鲁棒性 ≥ ELU 和 ReLU。

## 7. 实战 tips

paper Discussion 部分给两个 practical tips：

1. **使用 momentum optimizer**：Adam, SGD with momentum 等。无 momentum 的 SGD 训练 GELU 不稳定。
2. **使用精确或高质量的 CDF 近似**：`xσ(x)`（SiLU）虽然 work，但比 GELU 略差。paper 用 `0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])`，所有实验都用这个近似。

## 8. 后续 Impact

GELU 已经成为 Transformer 的 default activation：

- **BERT** (Devlin et al. 2018): GELU 用于 FFN 层
- **GPT 系列** (OpenAI): GPT-2, GPT-3, GPT-4 都用 GELU
- **ViT** (Dosovitskiy et al. 2021): Vision Transformer 也用 GELU
- **T5** (Google): 也采用 GELU

为什么 GPT/BERT 选 GELU 而非 ReLU？我猜原因是：
1. Smoothness 让 self-attention 后的 FFN 更 stable
2. Negative values 帮助 layer norm 之后的 distribution centering
3. Empirical experience 在大规模 pretraining 中表现好

PyTorch 实现: `torch.nn.GELU(approximate='none'|'tanh')`
TensorFlow 实现: `tf.nn.gelu(x, approximate=True|False)`

参考：
- BERT: https://arxiv.org/abs/1810.04805
- GPT-2: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- ViT: https://arxiv.org/abs/2010.11929
- PyTorch GELU doc: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html

## 9. SiLU / Swish 的 credit assignment drama

paper Appendix B 讲述了一段 academic drama，值得 Andrej 注意：

- 2016: Hendrycks 在本文首次提出 `SiLU = xσ(x)`，作为 GELU 的 sigmoid 近似变体
- 2017 早期: Elfwing et al. 发表同样公式，命名 "SIL"
- 2017 末: Google Brain 的 Ramachandran et al. 发表 "swish" = `xσ(βx)`，**未引用前两篇 paper**
- 被指出后，作者承认遗漏：*"As has been pointed out, we missed prior works..."*
- 更新版加入 β hyperparameter 声称 novelty
- 但 community 实际用 β=1，等于 SiLU
- TensorFlow/PyTorch 最初命名为 "swish"
- Reddit 帖子 "Google has a credit assignment problem in research" 引起关注
- 最终 TF/PyTorch 把 "swish" 重命名为 "SiLU"，承认 Hendrycks 的 priority

这是机器学习领域 credit assignment 问题的一个典型案例。值得注意的是：**GELU 最终通过 BERT/GPT 的采用，成为 Transformer 的事实标准，比 SiLU 更流行**。

参考：
- Swish paper: https://arxiv.org/abs/1710.05941
- Reddit post 讨论 Google credit assignment: https://www.reddit.com/r/MachineLearning/comments/...（具体链接已轶失，但讨论在社区中流传）

## 10. 我的额外联想与 open questions

### 10.1 为什么不学 μ 和 σ？

paper 提到 `N(μ, σ²)` 的 CDF 可以让 μ, σ 可学，但实验用 μ=0, σ=1。这其实留下一个 open question：**learnable GELU 是否会更好？** 后来 Swish 的 β 参数其实就是这个思路的变种，但 community 共识是 β=1 已经足够好。

### 10.2 GELU 与 LayerNorm 的协同

GPT/BERT 中 GELU 紧跟在 LayerNorm 之后，这意味着 GELU 的 input 已经被 normalize 到 mean 0, std 1 附近。这与 paper 假设 "neuron inputs tend to follow a normal distribution" 高度一致——LayerNorm 主动制造了这个 condition，让 GELU 的 probabilistic interpretation 更 valid。

### 10.3 GLU variants

Gated Linear Unit (GLU) 系列与 GELU 在概念上类似——都是 input-dependent gating。Dauphin et al. 2017 的 GLU 用 `x * σ(Wx + b)` 作为 gating，可以看作把 GELU 的 stochastic 期望改成 explicit learned gate。后来 **SwiGLU** (Shazeer 2020) = Swish-Gated Linear Unit 被 LLaMA 采用。这可以视为 GELU 思想的进一步演化：把 mask probability 从 `Φ(x)` 替换成 `σ(Wx)`，让 gate 有 explicit learned parameters。

参考：
- GLU paper: https://arxiv.org/abs/1612.08083
- SwiGLU: https://arxiv.org/abs/2002.05202
- LLaMA: https://arxiv.org/abs/2302.13971

### 10.4 GELU 在 RL 中的表现

paper 没有 evaluate RL tasks。后来有一些 informal observation 显示 GELU 在 RL 中不如 ReLU 稳定，可能与 non-monotonicity 在 value function approximation 中的 issue 有关。这是一个值得探索的方向。

### 10.5 概率解释的更深含义

GELU = `E[m·x]` where `m ~ Bernoulli(Φ(x))` 这个 probabilistic view 很 elegant。如果我们把这个 idea 推广——比如用 input-dependent Bernoulli 的 variance 而非 mean——会得到什么？`Var[m·x] = x²Φ(x)(1-Φ(x))`，这个量在 `x=0` 处达到 maximum `= 0.25x²`，在 `|x|→∞` 时趋于 0。这暗示 GELU 在 near-zero 处保留最多 "stochasticity"，在大 magnitude 处接近 deterministic。这种 behavior 与 Bayesian neural network 的 epistemic uncertainty 估计有 conceptual 联系。

### 10.6 与 Mish 的关系

Mish (Misra 2019) = `x · tanh(softplus(x)) = x · tanh(ln(1+e^x))` 是 GELU 之后又一个 smooth non-monotonic activation。Mish 与 GELU 形状极为相似，在 near-zero 附近都有 negative dip。这暗示 **non-monotonic + smooth + small negative bump** 是 high-performing activation 的一个重要 design pattern。

参考: https://arxiv.org/abs/1908.08681 (Mish paper)

---

## 总结

GELU 的 elegance 在于把三个看似独立的概念 unified：

1. **Deterministic nonlinearity**（ReLU 的 sign gating）
2. **Stochastic regularization**（Dropout 的 random mask）  
3. **Probabilistic interpretation**（CDF 作为 retain probability）

并通过 expectation 把它们凝结成一个 simple, smooth, no-extra-hyperparameter 的 activation function `xΦ(x)`。加上大规模 Transformer 实证（BERT/GPT/ViT）的背书，它已经从 2016 年的 "interesting alternative" 变成 2026 年的 "default choice"。

希望这个 deep dive 帮你 build 出 GELU 的 intuition——它的 power 来自 statistical interpretation 而非 arbitrary engineering choice。
