---
source_pdf: Quantum-Optimized Selective State Space Model.pdf
paper_sha256: 518d23b58dcd534b610d71e974fbc0971270a33217baf277a31a0012edb37642
processed_at: '2026-08-06T07:56:16-07:00'
target_folder: Quantum-Computing
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Q-SSM

## 一句话说清楚

作者把 Mamba 这种 state space model 里的 sigmoid gate 换成了一个量子电路的输出，发现训练更稳了，长程预测比 Transformer 好一截，比 S-Mamba 好一点点。

---

## 问题出在哪

时序预测模型的核心痛点就一句话：**长 horizon 上容易崩**。

Transformer 系（Autoformer、Informer、Reformer）用 attention 去捕捉长程依赖，但 attention 是 $O(W^2)$ 的，序列一长就爆显存，而且 horizon 越长效果越差。Mamba 用 selective state space 搞线性时间递归，解决了效率问题，但训练不稳定，对初始化敏感，multivariate 上经常抽风。

根本原因在于 **gating 机制**。Mamba 这类 SSM 的 hidden state 更新靠一个 gate $g$：

$$h_t = (1-g) h_{t-1} + g \cdot u_t$$

$g$ 大就多接收新信息，$g$ 小就多保留旧记忆。这个 gate 传统上用 sigmoid 生成：$g = \sigma(w^\top x + b)$。

Sigmoid 有两个病：

**病一：饱和**。当 $w^\top x$ 的绝对值一大，sigmoid 就卡在 0 或 1 附近不动了，gradient 直接趋零。模型一旦进入饱和区就僵住了，该忘的忘不掉，该记的记不进。

**病二：线性 pre-activation**。$w^\top x + b$ 是个线性函数，表达力有限。面对非平稳序列的 regime shift，线性 gate 根本反应不过来。

---

## 量子门干了啥

作者的 idea 很直接：**把 sigmoid 前面那个线性 pre-activation 换成量子电路的 expectation value**。

具体长这样：

```
|0⟩ ── RY(θ) ── RX(φ) ── 测 Z ──→ z = cos(θ)·cos(φ) ∈ [-1, 1]
```

一个 qubit，初始化在 $|0\rangle$，先绕 Y 轴转 $\theta$，再绕 X 轴转 $\phi$，然后测量 Z 方向的期望值。数学上这个 expectation 就是 $\cos\theta \cdot \cos\phi$。

然后两个这样的 circuit 并联，linear combination 一下，过 sigmoid，clip 到 $[0.05, 0.95]$：

$$g = \text{clip}(\sigma(w_1 \cos\theta_1 \cos\phi_1 + w_2 \cos\theta_2 \cos\phi_2 + b_g), 0.05, 0.95)$$

---

## 为啥这玩意儿有用

**关键在于 $\cos\theta \cos\phi$ 是振荡的，不是单调的**。

Sigmoid 的 pre-activation $w^\top x + b$ 是单调线性，一旦 drift 到饱和区就死在那。量子 pre-activation 是 trigonometric 的，在 $(\theta, \phi)$ 空间里波浪起伏。即使 sigmoid 在某个 $s$ 值上饱和了，gradient descent 会推动 $\theta$ 或 $\phi$ 移动到下一个 non-saturated 区域，因为 cos 天然有多个"出口"。

数学上更漂亮：

- $\partial z / \partial \theta = -\sin\theta \cos\phi$，绝对值 $\leq 1$，所以 $z$ 对参数是 1-Lipschitz 的
- $\partial g / \partial \theta \leq |w|/4$，bounded，不会爆炸
- $\partial h_t / \partial h_{t-1} = (1-g) I$，因为 $g \in [0.05, 0.95]$，所以 $|1-g| < 1$，这是个 contraction

Contraction 意味着 hidden state 长期不会爆炸也不会消失，信息既不会被无限放大也不会彻底遗忘。这对 long-horizon 是命根子。

---

## 用一个 mental model 理解

想象 gate 是个水龙头，控制"新水流进来 vs 旧水留住"的比例。

- Classical sigmoid gate：水龙头把手是线性弹簧推的，推到极限就卡死，要么全开要么全关，中间状态保不住
- Quantum gate：水龙头把手是两个旋转的偏心轮带动的，cosine 波浪式运动，永远在变化，永远不会卡死

时序数据本身就有 trend（慢变）+ seasonality（周期）+ noise（快变），量子门这种 oscillatory 的 inductive bias 天然跟 seasonality 对得上。

---

## 架构长啥样

整体就是三块：

**输入端**：原始 multivariate 序列 + calendar features（hour/day 的 sin/cos encoding）→ linear projection 到 hidden space → LayerNorm

**Backbone**：单方向的 selective state space 递归，gate 用量子电路输出，$h_t = (1-g) h_{t-1} + g \cdot u_t$，每步更新

**Decoder**：MLP + Dropout + Linear 把最后的 hidden state 投影成 $H \times F$ 的预测，然后加回 last observation 做 residual

这里有几个设计选择值得注意：

- **单向 backbone**（不是 S-Mamba 的 bidirectional），降低复杂度
- **2 个独立 single-qubit circuit**（不是 entangled multi-qubit），计算 overhead 几乎为零
- **Residual decoder**：预测的是相对 last observation 的 delta，不是 absolute value。这避免 long-horizon 上的 drift，N-BEATS、DLinear 都用这招

---

## 实验结果怎么读

作者在 ETT（4个变体）、Traffic、Exchange Rate 上测，horizon 从 96 到 720。

**对比 Transformer 系**：Q-SSM 完胜。MSE 降低 30-57% 是常态。这主要归功于线性时间 backbone 本身，attention 在长序列上确实是结构性劣势。

**对比 S-Mamba**：Q-SSM 赢 32/36 个配置，但大多数领先不到 1%。在 ETTh2、Traffic、Exchange 的 $H=720$ 上 Q-SSM 反而输给 S-Mamba。

这说明啥？**量子门在 short-to-medium horizon 上确实稳定训练，但 long horizon 上 S-Mamba 的 bidirectional + FFN 信息通路更宽**。单方向 backbone 的信息 bottleneck 在超长 horizon 上是硬伤。

---

## 几个要 honest 说的事

**第一，这里没有真量子加速**。作者用 PennyLane 的 statevector simulator 跑，实际就是算 $\cos\theta \cos\phi$，在 classical GPU 上 trivial。如果上真量子硬件，每个 expectation 要 1000+ shots 采样，parameter-shift 每个 gradient 要跑 8 次 circuit，加上 NISQ noise，训练稳定性可能反而崩。这篇 paper 的 "quantum" 更准确说是 "quantum-inspired trigonometric activation"。

**第二，提升 marginal**。相比 S-Mamba 大多 <1%，没有 error bar，没有多 seed std。这个量级的提升到底是量子门的功劳还是 hyperparameter 调出来的，存疑。

**第三，缺 ablation**。如果直接用 classical $\cos(w^\top x + b)$ 当 pre-activation，效果是不是一样？如果只用 1 个量子电路而不是 2 个？如果 clip range 换成 $[0.1, 0.9]$？这些都没测。如果 classical trig 能达到同样效果，那 "quantum" 这个 framing 就站不住。

**第四，Table I 里 baseline 数字复用了 Autoformer paper**。ETTh1/ETTh2/ETTm1/ETTm2 在 $H=96$ 上 Informer、LogTrans、Reformer 等的数字完全一样，说明没重新跑 baseline，公平性打折扣。

---

## 我的直觉

Q-SSM 的真正 contribution 在于：**用 oscillatory、Lipschitz-bounded、参数极少的非线性替换 sigmoid 的线性 pre-activation，给长程 SSM 的 gating 提供稳定性 inductive bias**。

它跟 SIREN 用 sinusoidal activation、跟 Fourier feature map 用 trigonometric embedding 是一类思路。Quantum 只是个 framing，本质是 bounded oscillatory nonlinearity。Contractivity + oscillatory escape 共同解决了 long-horizon 上的 gradient pathology。

如果 Andrej 你想 build 更深的 intuition，我建议做三件事：

1. **直接 ablate**：把 $\cos\theta \cos\phi$ 换成 $\cos(w^\top x + b)$ 的 classical 版本，看效果掉不掉。如果不掉，"quantum" 就是 narrative 不是 substance
2. **看 gradient norm 分布**：训练过程中 quantum gate 的 gradient 是否真的比 sigmoid gate 更 healthy、更不饱和
3. **在真量子硬件上 reproduce**：哪怕就跑 ETTm1 $H=96$，看 NISQ noise 下 contractivity property 还保不保得住

这篇 paper 的价值在 "提出 quantum gate 作为 SSM 的 inductive bias" 这条 path 的开创性，不在实验数字的绝对优势。理解这一点能 calibrate 对这类 hybrid quantum-classical 工作的期望。

---

# Q-SSM: Quantum-Optimized Selective State Space Model 深度解析

## 1. Paper 的核心 thesis 与定位

这篇 paper 来自 Politehnica University Timișoara (罗马尼亚), 发表于 2024 末/2025 初。它属于一个相对小众但 growing 的方向: **hybrid quantum-classical 架构用于 sequence modeling**。作者团队之前还有 *Quantum Machine Learning* 相关工作 ([Udrescu 的 quantum ML 论文](https://arxiv.org/abs/2501.04464))。

核心 thesis 一句话总结: **用 parametrized quantum circuit 的 expectation value 替换 classical sigmoid gate 的 linear pre-activation, 提供 Lipschitz-bounded、non-vanishing gradient 的 gating dynamics, 从而稳定 long-horizon SSM 训练。**

这篇 paper 在 Mamba 系之后 ([Mamba 原文](https://arxiv.org/abs/2312.00752))、S-Mamba ([S-Mamba](https://arxiv.org/abs/2403.11144)) 提出 time-series 适配之后, 探索 "用 quantum inductive bias 替换 S-Mamba 的 gating" 这条路径。

---

## 2. Background: 从 classical SSM 到 Q-SSM 的演化路径

### 2.1 连续时间 SSM (Eq. 1)

$$\frac{d}{dt} h(t) = A h(t) + B x(t), \quad y(t) = C h(t) + D x(t)$$

变量含义:
- $h(t) \in \mathbb{R}^N$: hidden state vector, 存储过去信息
- $x(t) \in \mathbb{R}$: scalar input at time $t$
- $y(t) \in \mathbb{R}$: scalar output
- $A \in \mathbb{R}^{N \times N}$: state transition matrix, 决定 dynamics 的固有 decay rate (eigenvalues $\lambda_i(A)$ 控制每个 mode 的衰减)
- $B \in \mathbb{R}^{N \times 1}$: input coupling vector, 决定 input 如何 inject 到 state
- $C \in \mathbb{R}^{1 \times N}$: observation/readout vector
- $D \in \mathbb{R}$: feed-through (direct skip connection, 通常设为 0)

**Intuition**: $A$ 的 eigenvalues 在左半平面决定衰减, 但不同 eigenvalue 对应不同时间尺度。S4 ([S4 paper](https://arxiv.org/abs/2111.00396)) 用 HiPPO matrix 给出连续的 polynomial basis projection, 让 state 自带"natural" 多尺度 representation。

### 2.2 Discretization → Linear recurrence

$$h_{t+1} = \bar{A} h_t + \bar{B} x_t, \quad y_t = \bar{C} h_t + \bar{D} x_t$$

其中 $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B$ (zero-order hold)。这是 S4 的核心 trick: 通过结构化 $A$ matrix, 可以用 FFT 在 $O(N \log N)$ 而非 $O(N^2)$ 计算 convolution。

### 2.3 Mamba 的 selective SSM ([Mamba paper](https://arxiv.org/abs/2312.00752))

Mamba 的关键 idea: 把 $\bar{B}$, $\bar{C}$, $\Delta$ **变成 input-dependent** (selective)。这打破 S4 的 LTI (linear time-invariant) 约束, 允许 model "选择" 何时记住何时遗忘。公式上 Mamba 用类似以下形式:

$$h_t = \exp(-\Delta_t \cdot \text{softplus}(A)) \cdot h_{t-1} + \Delta_t \cdot B(x_t) \cdot x_t$$

input-dependent gating 通过 sigmoid 实现, 这就是 Q-SSM 改进的地方。

### 2.4 S-Mamba 的 time-series 适配

S-Mamba ([arXiv](https://arxiv.org/abs/2403.11144)) 把 Mamba 用 bidirectional SSM block 处理 multivariate time series, 配合 FFN 和 linear projection。但 paper 指出 S-Mamba 的痛点:
- hyperparameter initialization 敏感
- bidirectional block 引入额外训练成本
- multivariate 鲁棒性不足

Q-SSM 想用更简单的 single-directional backbone + quantum gate 来获得更稳定的 training dynamics。

---

## 3. Q-SSM 架构深度解析

### 3.1 整体架构 (对应 Figure 2, 3)

```
Input X ∈ R^{B×T×F}
    │
    ├─ Calendar features augmentation (sin/cos of hour, day)
    │
    ▼
[Linear Projection P: R^F → R^k]
    │
    ▼
[Linear Transform W: R^k → R^d] + bias b + α·c (calendar scalar)
    │
    ▼
[Layer Normalization] → u_t
    │
    ├─── Quantum Gate g = clip(σ(w1·z1 + w2·z2 + b_g), 0.05, 0.95)
    │        ↑ z_i = ⟨ψ(θ_i, φ_i)|Z|ψ(θ_i, φ_i)⟩ = cos(θ_i)·cos(φ_i)
    │        ↑ R_Y(θ) R_X(φ) |0⟩
    │
    ▼
h_t = (1-g)·h_{t-1} + g·u_t  (recurrence, Eq. 9/10)
    │
    ▼ (final h_T)
[Decoder: MLP + Dropout + Linear Projection]
    │
    ▼
Ŷ_base ∈ R^{H×F}
    │
    ▼
+ 1_H · x_T^T   (residual, broadcast last observation)
    │
    ▼
Ŷ ∈ R^{H×F}
```

**关键设计决策**:
1. **Single-directional recurrence**, 不像 S-Mamba 用 bidirectional — 这降低复杂度但保留 causal 性
2. **Quantum gate 用 2 个独立 single-qubit circuits**, 不是 multi-qubit entangled circuit — 计算 overhead 极小
3. **Residual decoder** 把预测变成"相对 last observation 的 deviation" — 这是 Long-term Forecasting 的常见 trick (参见 [N-BEATS](https://arxiv.org/abs/1905.10437), [N-HiTS](https://arxiv.org/abs/2201.12886))

### 3.2 Backbone recurrence (Eq. 9) 详细拆解

$$h_t = (1-g) h_{t-1} + g \cdot \text{LN}(W(P(x_t)) + b + \alpha \cdot c)$$

各项含义:
- $P: \mathbb{R}^F \to \mathbb{R}^k$: 输入 linear projection, $k=128$ (paper 设定)
- $W: \mathbb{R}^k \to \mathbb{R}^d$: 中间 transform, $d=128$
- $b \in \mathbb{R}^d$: bias
- $\alpha \in \mathbb{R}$: learnable scalar, controls calendar influence
- $c$: aggregated calendar signal (calendar features 在 input window 上的 mean)
- $\text{LN}$: Layer Normalization, 稳定 internal covariate shift
- $g \in [0.05, 0.95]$: quantum gate output

**为什么有 $\alpha \cdot c$?** 这是一种 "global temporal context injection"。Calendar signal 在整个 window 内 mean 一下, 给每个 timestep 提供 seasonal prior。这相当于 S-Mamba 中 FFN 处理 cross-time 信息的简化版本。

**为什么 LN 在 gate 之后而不是之前?** LN 在 backbone 中起 "input stabilization" 作用: 即使 $W(P(x_t))$ 数值范围漂移 (non-stationary), LN 也能让 hidden state update 保持 bounded scale。

### 3.3 Quantum Gate (核心创新, Eq. 12-15)

这是 paper 的核心。让我详细拆解。

#### 3.3.1 Quantum circuit 设计 (Figure 1)

```
Quantum circuit (per timestep, per batch):

|0⟩ ──[R_Y(θ)]──[R_X(φ)]── Measure Z ──→ z = cos(θ)·cos(φ) ∈ [-1, 1]
```

这是一个 **2-parameter ansatz**, 单 qubit, 两个 rotation gate 串联。Qiskit 自动添加 classical register 存 measurement result, 但实际不用 0/1 discrete outcome, 而是用 **expectation value** $\langle Z \rangle$。

为什么选 RY-RX 而不是 RX-RY 或 RZ-rotation? 因为 RY-RX 在 Bloch sphere 上产生 non-commuting rotation, 给出 trigonometric 乘积 $\cos\theta \cos\phi$ (Eq. 13)。如果用 commuting gates (e.g., RY-RY) 会得到 single-variable 函数, expressivity 不够。

#### 3.3.2 Expectation value 计算 (Eq. 13)

$$z(\theta, \phi) = \langle \psi(\theta, \phi) | Z | \psi(\theta, \phi) \rangle = \cos\theta \cos\phi$$

推导:
- $R_Y(\theta) |0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$ (rotation around Y axis)
- $R_X(\phi) R_Y(\theta) |0\rangle$: 在 Bloch sphere 上, 这个 state 的 Bloch vector 是 $(\sin\theta \sin\phi, -\sin\theta \cos\phi, \cos\theta \cos\phi)$ 的某个变换, $Z$-expectation 是 $z$-component
- 因此 $\langle Z \rangle = \cos\theta \cos\phi$

**Intuition**: 这个函数在 $(\theta, \phi)$ 平面是 oscillatory 的, 而不是 monotonic 的 (像 sigmoid)。它有两个 "frequency direction", 能在 training 过程中 escape local minima, 因为 gradient 不只是单调衰减。

#### 3.3.3 Two-circuit linear combination (Eq. 14)

$$s = w_1 z_1 + w_2 z_2 + b_g$$

为什么 2 个 circuits? 单个 circuit 的 $z = \cos\theta \cos\phi$ 是 symmetric 的, 在 $[0, \pi]$ 范围内 single-mode。两个 circuits 给出 linear combination, 表达能力类似 "two-layer trigonometric basis"。增加 circuits 数量会单调增加 expressivity, 但 paper 选 2 个作为 expressivity-vs-cost 的 sweet spot。

#### 3.3.4 Final gate (Eq. 15)

$$g = \text{clip}(\sigma(s), g_{\min}, g_{\max})$$

with $g_{\min} = 0.05$, $g_{\max} = 0.95$.

**关键设计点**: 为什么仍用 $\sigma(\cdot)$? Paper Section III-C-4 解释: sigmoid 在这里**只作为 normalization** (mapping to (0,1)), 不是 nonlinearity 来源。nonlinearity 来自 quantum expectation $\cos\theta\cos\phi$。这避免了 trivial saturation (因为 quantum pre-activation 是 oscillatory, 不会让 sigmoid 长期停在 saturated region)。

为什么 clip 到 $[0.05, 0.95]$? 保证:
- 不会 $g=0$ → 完全 no update, 死掉
- 不会 $g=1$ → 完全 overwrite, 失去 memory

这是 recurrent model 的常见技巧, 类似 Highway Networks 的 $g \in [0.1, 0.9]$ 启发式。

### 3.4 数学保证: Lipschitz continuity 与 contractivity

#### 3.4.1 Quantum expectation gradients (Eq. 16-17)

$$\frac{\partial z}{\partial \theta} = -\sin\theta \cos\phi, \quad \frac{\partial z}{\partial \phi} = -\cos\theta \sin\phi$$

$$\left|\frac{\partial z}{\partial \theta}\right| \leq 1, \quad \left|\frac{\partial z}{\partial \phi}\right| \leq 1$$

这给出 **1-Lipschitz** w.r.t. each parameter。这是一个 beautiful property: 量子 parameter 的 perturbation 永远不会让 output 爆炸。

#### 3.4.2 Gate gradient bound (Eq. 18-19)

$$\frac{\partial g}{\partial \theta_i} = \sigma'(s) \cdot w_i \cdot \frac{\partial z_i}{\partial \theta_i}$$

$$\left|\frac{\partial g}{\partial \theta_i}\right| \leq \frac{|w_i|}{4}$$

因为 $|\sigma'(s)| \leq 1/4$ (sigmoid 的 max gradient) 且 $|\partial z_i / \partial \theta_i| \leq 1$。

**Intuition**: 与 classical sigmoid gate 相比, classical gate 的 gradient w.r.t. $w$ 是 $\sigma'(w^\top x + b) \cdot x$, 当 $|w^\top x|$ 大时 gradient → 0 (vanishing)。Quantum gate 由于 pre-activation 是 oscillatory, 不会出现长期 saturation。

#### 3.4.3 Contractivity (Eq. 20-21)

$$\frac{\partial h_t}{\partial h_{t-1}} = (1-g) I$$

$$\left\|\frac{\partial h_t}{\partial h_{t-1}}\right\|_2 = |1-g| < 1$$

由于 $g \in [0.05, 0.95]$, $1-g \in [0.05, 0.95]$, 所以 $\|J\|_2 < 1$。

这给出 **contraction mapping**, 隐含 Banach fixed-point theorem: 长期 state 会 converge, 不会爆炸或完全消失。这与 GRU 的 $1 - r_t$ gate 类似, 但 GRU 不强制 bounded range。

**这个 contractivity 与 S4 的 $\bar{A}$ 矩阵 condition 数对比**: S4 的 HiPPO matrix 设计保证 eigenvalues 在左半平面, 但 discretized $\bar{A} = \exp(\Delta A)$ 的 norm 不一定 < 1 (取决于 $\Delta$)。Q-SSM 通过显式 gate clipping 保证严格 contractivity。

### 3.5 Decoder (Eq. 22-Section III-D)

四步:
1. **MLP nonlinear projection**: $z = \text{ReLU}(W_1 h + b_1)$, $W_1 \in \mathbb{R}^{d \times d}$
2. **Dropout**: $z' = \text{Dropout}(z; p=0.1)$
3. **Linear projection**: $\hat{y}_{\text{flat}} = W_2 z' + b_2$, $W_2 \in \mathbb{R}^{(H \cdot F) \times d}$
4. **Residual**: $\hat{Y} = \hat{Y}_{\text{base}} + \mathbf{1}_H x_T^\top$

**Residual 的 intuition**: 这是 long-horizon forecasting 的关键 trick。如果直接预测 absolute value, non-stationary series 会让 decoder 在 long horizon 上 drift。预测相对 last observation 的 delta, 让 decoder 只需要学习 "trend continuation" + "deviation patterns", 不需要学习 absolute level。这跟 N-BEATS 的 basis projection 和 DLinear 的 trend-seasonal decomposition 思路一致 ([DLinear](https://arxiv.org/abs/2205.13504))。

### 3.6 Complexity (Section III-E)

- Backbone: $\mathcal{O}(W(Fk + kd))$ per sequence, $W$=window length
- Decoder: $\mathcal{O}(H \cdot d \cdot F)$
- 总计: $\mathcal{O}(W(Fk+kd) + HdF)$

对比:
- Autoformer: $\mathcal{O}(W \log W)$ (auto-correlation 用 FFT)
- Mamba: $\mathcal{O}(Wd^2)$ (quadratic in hidden dim, 因为 selective scan 在 $d$ 维 state 上并行)
- Transformer: $\mathcal{O}(W^2 d)$ (quadratic in sequence length)

Q-SSM 在 $W$ 和 $H$ 上都是 linear, 在 $d$ 上是 linear, 这是相比 Mamba 的实际优势 (Mamba 在 $d$ 上 quadratic, 因为 hardware-efficient scan 需要 $d$ 维 state)。但 paper 中 $d=128$, 差异不大。

Quantum gate overhead: 每 batch 2 个 single-qubit expectation values, 用 statevector simulator 跑是 constant time, 几乎可忽略。但真实 quantum hardware 上需要多次 measurement shot 来 estimate expectation, 这是 scaling 到 quantum hardware 的潜在瓶颈。

---

## 4. 实验深度分析

### 4.1 Datasets

| Dataset | Frequency | #Features | Total Length | Periodicity |
|---------|-----------|-----------|--------------|-------------|
| ETTh1/ETTh2 | Hourly | 7+4 calendar | 17,420 | Strong (yearly) |
| ETTm1/ETTm2 | 15-min | 7+4 calendar | 69,680 | Strong (yearly) |
| Traffic | Hourly | 862+4 calendar | 17,544 | Strong (weekly) |
| Exchange | Daily | 8 | 7,588 | Weak (stochastic) |

**为什么 ETT 加 calendar 但 Exchange 不加?** Paper 在 Section IV-B 给出 explanation: Exchange 是 highly stochastic, calendar encoding 不会 improve performance。这是 dataset-specific inductive bias。

### 4.2 主结果 (Table I 拆解)

让我重新整理最有意义的对比 (Q-SSM vs S-Mamba, vs Autoformer):

| Dataset | H | Q-SSM MSE | S-Mamba MSE | Δ vs S-Mamba | Δ vs Autoformer |
|---------|---|-----------|-------------|--------------|-----------------|
| ETTm1 | 96 | 0.330 | 0.333 | -0.9% | -34.7% |
| ETTm1 | 720 | 0.472 | 0.475 | -0.6% | -29.6% |
| ETTm2 | 96 | 0.172 | 0.179 | -3.9% | -32.5% |
| ETTm2 | 720 | 0.407 | 0.411 | -1.0% | -3.6% (vs Autoformer 0.422) |
| ETTh1 | 96 | 0.384 | 0.386 | -0.5% | -14.5% |
| ETTh2 | 720 | 0.429 | 0.426 | +0.7% (loses!) | -16.7% |
| Traffic | 96 | 0.380 | 0.382 | -0.5% | -38.0% |
| Traffic | 720 | 0.472 | 0.460 | +2.6% (loses!) | -28.5% |
| Exchange | 96 | 0.084 | - | - | -57.4% |
| Exchange | 720 | 0.875 | 0.867 | +0.9% (loses!) | -39.5% |

**关键观察**:
1. **Q-SSM 相比 Transformer-based (Autoformer) 的提升巨大且稳定**: 30-40% MSE reduction 是 significant 的
2. **Q-SSM 相比 S-Mamba 的提升 marginal**: 大多 <1%, 在 long horizon 上 (ETTh2 H=720, Traffic H=720, Exchange H=720) Q-SSM **输给** S-Mamba
3. **Q-SSM 在 short horizon 上更优, S-Mamba 在 long horizon 上更优**: 这可能是因为 Q-SSM 的 single-directional backbone 在 long horizon 上信息 bottleneck 比 S-Mamba 的 bidirectional + FFN 严重

### 4.3 计算 setup 的 critical 细节

> "The quantum module evaluates only two expectation values of a single simulated qubit, so its runtime overhead is negligible"

**这是一个需要 critical 思考的点**: 论文用 **statevector simulator** (PennyLane) 跑 quantum circuit。这意味着:
- Quantum gate 实际是 **classical computation** (sin/cos)
- 在真实 quantum hardware 上, 每个 expectation value 需要多次 measurement shot (~1000+ shots per gradient eval), 这会显著增加 latency
- Parameter-shift rule (Eq. 6) 需要每个 parameter 跑 2 次 circuit, 4 个 parameter 共 8 次 circuit evals per gradient step

所以这个 paper 实际上是 "用 quantum-inspired nonlinearity 替换 sigmoid", 而不是真 quantum advantage。这点 paper 没有明确 disclaimer, 但应该提示读者。

### 4.4 实验结果审视

仔细看 Table I, 我发现一个 **可疑的模式**: ETTh1, ETTh2, ETTm1, ETTm2 在 $H=96$ 上, Informer, LogTrans, Reformer, LSTNet, LSTM, TCN 的数值完全一样 (0.365/0.453, 0.768/0.642, 0.658/0.619, 3.142/1.365, 2.041/1.073, 3.041/1.330)。

这意味着 paper 复用了 [Autoformer paper](https://arxiv.org/abs/2106.13008) 的 baseline numbers, 没有重新 run。这是 time series forecasting 领域的常见做法, 但要小心: baseline implementation 细节可能不一致, comparison 公平性存疑。

---

## 5. Intuition-building: 为什么 quantum gate 工作?

### 5.1 Sigmoid gate 的 pathology

Classical gate: $g = \sigma(w^\top x + b)$

**问题 1 (vanishing gradient)**: $\sigma'(x) \leq 1/4$, 当 $|x|$ 大时 $\sigma'(x) \to 0$。在 long-horizon 训练中, $w^\top x$ 容易 drift 到 saturated region, gate "卡死" 在 0 或 1。

**问题 2 (linear pre-activation)**: $w^\top x + b$ 是 input 的 linear function, 限制了 gate 对 input pattern 的 adaptability。Non-stationary series 中, linear gate 无法捕捉 nonlinear regime shift。

### 5.2 Quantum gate 的 fix

**Fix 1 (oscillatory pre-activation)**: $s = w_1 \cos\theta_1 \cos\phi_1 + w_2 \cos\theta_2 \cos\phi_2 + b_g$。这个 pre-activation 在 $(\theta, \phi)$ 空间 oscillate, 不会单调 drift 到 saturated region。即使 $\sigma$ 在某个 $s$ 上 saturated, gradient descent 会推 $\theta$ 或 $\phi$ 移动到 non-saturated region (因为 $\cos$ 的 oscillatory nature)。

**Fix 2 (bounded Lipschitz)**: $|\partial g / \partial \theta| \leq |w|/4$。bounded gradient 保证 Adam 不会 overshoot, 训练 trajectory 平滑。

**Fix 3 (contractivity)**: $|1-g| < 1$ 保证 hidden state 不会 explode。

### 5.3 与其他 stable recurrent 设计的对比

- **Highway Networks** ([Srivastava et al.](https://arxiv.org/abs/1505.00387)): $h_t = g \cdot H(x_t) + (1-g) \cdot x_t$, 但 $g$ 是 sigmoid, 同样有 saturation 问题
- **GRU** ([Cho et al.](https://arxiv.org/abs/1406.1078)): $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$, $h_t = (1-z_t) h_{t-1} + z_t \tilde{h}_t$。和 Q-SSM 结构几乎一样, 但 GRU 的 gate 也是 sigmoid
- **Mogrifier LSTM** ([Gloeckle et al.](https://arxiv.org/abs/1909.01792)): 用 iterative multiplication 替代 sigmoid gate
- **Q-SSM**: 用 quantum expectation 替代 sigmoid pre-activation, 保留 sigmoid 作为 normalizer

可以理解为: **Q-SSM 把 quantum expectation 当作 "richer nonlinearity" 注入到 gate 的 pre-activation 中**。这跟用 Fourier feature 替代 linear projection ([Tancik et al.](https://arxiv.org/abs/2006.10739)) 或用 Bessel function 作为 activation 的思路类似。

### 5.4 从 optimization landscape 角度看

Classical sigmoid gate 的 loss surface 在 saturation region 是 flat (vanishing gradient), 优化器容易 stuck。Quantum gate 由于 oscillatory pre-activation, loss surface 是 "wavy" 的, 优化器可以在 wavy landscape 中 escape saddle points。

这是 paper Section III-C-4 没有明说但暗含的 intuition: **quantum gate = oscillatory inductive bias, 利于 escape bad local minima**。

### 5.5 从 inductive bias 角度看

Quantum gate 提供的 inductive bias:
1. **Periodic/oscillatory gating** (cos product): 适合 time series 中的 seasonality
2. **Bounded smoothness** (Lipschitz): 适合 long-horizon stability
3. **Parameter-efficient**: 4 个 quantum parameter + 3 个 classical ($w_1, w_2, b_g$) = 7 个 gate parameter 总共

Classical sigmoid gate 的 inductive bias:
1. **Monotonic gating**: 假设 "more input → more update"
2. **Saturation**: 假设 gate 有 clear on/off states

对 time series, **periodic inductive bias 更合理** (seasonality 是 time series 的关键结构)。

---

## 6. 批判性思考与潜在问题

### 6.1 实验提升 marginal

Q-SSM 相比 S-Mamba 在多数 dataset/horizon 组合上提升 <1%。这 marginal improvement 是否真来自 quantum gate, 还是 hyperparameter tuning 的随机性? Paper 没有给 error bars 或 multiple seeds 的 std dev。

### 6.2 "Quantum" 实际是 classical simulation

Statevector simulator 跑的 RY-RX ansatz, 实际就是计算 $\cos\theta \cos\phi$。这在 classical hardware 上 trivial 实现。**这不是真 quantum advantage**, 而是 "quantum-inspired activation function"。

如果用真 quantum hardware:
- 每个 expectation 需 ~1000 shots 来 estimate (PennyLane 默认 shots=1000)
- Parameter-shift 需要 2x circuit evals per gradient
- 4 个 quantum parameter × 2 (parameter shift) × 1000 shots = 8000 measurement per training step

在 NISQ (noisy intermediate-scale quantum) hardware 上, noise 会让 expectation value 偏离理想 $\cos\theta\cos\phi$, 训练可能不稳定。

### 6.3 2 个 single-qubit circuit 太简单

Paper 提到 "single-qubit rotations, no multi-qubit circuitry required"。但 single-qubit circuit 没有 entanglement, 没有 quantum parallelism 的真优势。两个 single-qubit circuits 等价于两个 classical trigonometric functions 的 linear combination, **expressivity 不比 RBF kernel 或 Fourier feature 强**。

Paper 在 Section VI 也承认: "scaling Q-SSM to larger qubit ansatze may enable richer non-linear dynamics, provided efficient hardware implementations become available."

### 6.4 Long-horizon 输给 S-Mamba

Q-SSM 在 ETTm1/ETTh1 长 horizon 上 marginal 输给 S-Mamba, 在 ETTh2/Traffic/Exchange 长 horizon 上明显输。这说明:
- 单向 backbone 在 long-horizon 上信息 bottleneck 严重
- S-Mamba 的 bidirectional + FFN 在 long-horizon 上有真优势
- Quantum gate 的 "稳定 long-horizon" claim 在实验上不完全成立

### 6.5 缺少 ablation

Paper 没有给:
- 用 classical trigonometric function (直接 $\cos(w^\top x + b)$) 替换 quantum 的 ablation
- 用 single quantum circuit 替换 two circuits 的 ablation
- 不同 $g_{\min}, g_{\max}$ 的 ablation
- 不同 circuit depth 的 ablation

这些 ablation 是关键, 因为如果 classical trigonometric function 能达到同样效果, "quantum" 的 framing 就 weak。

### 6.6 没有 quantum hardware 实验

全部实验在 simulator 上跑。真正有意义的是在 IBM Quantum / IonQ / Rigetti 等 hardware 上 reproduce, 但 paper 没做。

---

## 7. 与其他相关工作的联系

### 7.1 Quantum ML 在 sequence modeling 的其他探索

- **Quantum LSTM** ([Chen et al. 2020](https://arxiv.org/abs/2009.01783)): 用 variational circuit 替换 LSTM cell 的 gate
- **Quantum attention** ([Zhou et al. 2022](https://arxiv.org/abs/2206.00680)): 用 quantum self-attention 替代 classical attention
- **Q-SSM**: 是 SSM 与 quantum 的第一次结合, 与 quantum LSTM 的思路一脉相承但选择 SSM 作为 backbone

### 7.2 Time series forecasting 的 inductive bias 讨论

- [DLinear](https://arxiv.org/abs/2205.13504): 简单 linear + decomposition 就能 beat Transformer, 说明 inductive bias 比 architecture 重要
- [Are Transformers Effective for Time Series Forecasting? (Zeng et al.)](https://arxiv.org/abs/2205.13504): One-shot linear model 打败复杂 Transformer
- Q-SSM 也支持这个 trend: 简单 backbone + good inductive bias (quantum gate) > 复杂 attention

### 7.3 Mamba 在 time series 的后续

- S-Mamba ([arxiv](https://arxiv.org/abs/2403.11144)): time series 专用 Mamba
- TimeMachine ([arxiv](https://arxiv.org/abs/2403.09898)): Mamba 的 bidirectional variant
- MambaTS ([arxiv](https://arxiv.org/abs/2405.00040)): Mamba for time series with rearrangement
- Q-SSM 是这条线上的一个 quantum-augmented 探索

### 7.4 Lipschitz-bounded activation 的其他工作

- **SIREN** ([Sitzmann et al.](https://arxiv.org/abs/2006.09661)): 用 sinusoidal activation, 类似 quantum gate 的 oscillatory inductive bias
- **Lipschitz Networks** ([Liu et al.](https://arxiv.org/abs/2002.08571)): 显式 Lipschitz bound
- **Neural ODEs** ([Chen et al.](https://arxiv.org/abs/1806.07366)): 用 ODE solver 训练 continuous-depth network, Q-SSM 的 contractivity 与此相关

---

## 8. 实用 takeaways

1. **如果你做 long-horizon time series**: Q-SSM 是一个 reasonable baseline, 但 S-Mamba 在 long horizon 上可能更优。可以考虑 hybrid (Q-SSM 短 horizon + S-Mamba 长 horizon)
2. **如果你做 quantum ML**: 这是 "quantum-inspired activation" 范例, 可以借鉴 quantum expectation 作为 differentiable nonlinearity 的思路
3. **如果你研究 stable recurrence**: contractivity + Lipschitz bound 是重要 property, quantum gate 是实现方式之一, 但用 classical bounded activation (e.g., tanh × sin) 也可能达到类似效果
4. **复现陷阱**: statevector simulator 的 "quantum" 在 classical hardware 上是 deterministic trigonometric function; 真量子 hardware 需要 noise-aware training

---

## 9. Web references

- Paper GitHub: [https://github.com/stephanjura27/quantum_ssm](https://github.com/stephanjura27/quantum_ssm)
- Mamba 原文: [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)
- S4: [https://arxiv.org/abs/2111.00396](https://arxiv.org/abs/2111.00396)
- S-Mamba: [https://arxiv.org/abs/2403.11144](https://arxiv.org/abs/2403.11144)
- Autoformer: [https://arxiv.org/abs/2106.13008](https://arxiv.org/abs/2106.13008)
- Informer: [https://arxiv.org/abs/2012.07436](https://arxiv.org/abs/2012.07436)
- Reformer: [https://arxiv.org/abs/2001.04451](https://arxiv.org/abs/2001.04451)
- DLinear "Are Transformers Effective for TS": [https://arxiv.org/abs/2205.13504](https://arxiv.org/abs/2205.13504)
- PennyLane (quantum ML framework): [https://pennylane.ai/](https://pennylane.ai/)
- Parameter-shift rule: [https://arxiv.org/abs/1811.11184](https://arxiv.org/abs/1811.11184)
- Highway Networks: [https://arxiv.org/abs/1505.00387](https://arxiv.org/abs/1505.00387)
- SIREN (sinusoidal activations): [https://arxiv.org/abs/2006.09661](https://arxiv.org/abs/2006.09661)
- Quantum LSTM: [https://arxiv.org/abs/2009.01783](https://arxiv.org/abs/2009.01783)
- ETT Dataset: [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
- Lipschitz Networks: [https://arxiv.org/abs/2002.08571](https://arxiv.org/abs/2002.08571)
- Neural ODEs: [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)
- Variational Quantum Algorithms review: [https://www.nature.com/articles/s42254-021-00348-9](https://www.nature.com/articles/s42254-021-00348-9)

---

## 10. Final intuition

Q-SSM 的本质: **用 oscillatory, Lipschitz-bounded, parameter-efficient nonlinearity (来自 quantum expectation) 替换 sigmoid 的 linear pre-activation, 以稳定 long-horizon SSM 的 gating dynamics**。

它不是 "quantum speedup", 而是 "quantum-inspired inductive bias for stable gating"。Long-horizon forecasting 的关键 challenge 是 stability across non-stationary regimes, 而 quantum gate 通过 contractivity + oscillatory escape 提供这个 stability。但这个 inductive bias 是否真需要 quantum framing (vs. classical trigonometric activation) 是一个未解 question, 需要更 thorough ablation。

如果你 (Andrej) 想 build 更深的 intuition, 我建议:
1. 复现 Q-SSM 并 ablate quantum vs classical trigonometric (直接用 $\cos(w_1^\top x + b_1) \cos(w_2^\top x + b_2)$ 替换 quantum)
2. 检查 gradient norm 在 training 过程中的 distribution, 看 quantum gate 是否真避免 saturation
3. 在 IBM Quantum hardware 上 reproduce, 看 NISQ noise 是否 destroy stability property
4. Scale up qubit 数量 (e.g., 2-qubit entangled ansatz), 看 expressivity 是否真提升

这个 paper 的 contribution 在 "提出 quantum gate 作为 SSM inductive bias" 这条线, 而非 "quantum hardware advantage"。理解这点能帮助 calibrate expectations。
