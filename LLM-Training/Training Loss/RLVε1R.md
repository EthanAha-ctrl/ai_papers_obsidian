---
source_pdf: RLVε1R.pdf
paper_sha256: efa344487c614f64b006b977aa3179d69201a02d901c33c29d9f1ba7bc4ad00a
processed_at: '2026-08-12T00:10:24-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 RLV<sup>ε</sup>R

## 一句话版本

**你的 verifier 只要比瞎猜稍微强一点点 (J>0), RL 就能学; 只要 J 一过零, 系统就崩。**

---

## 故事开场

想象你是一个老师, 要批改学生的编程作业。但你手头只有几个 test cases, 学生写的代码又很复杂。你会犯错:
- 有时候对的代码你给判错了 (false negative)
- 有时候错的代码你给判对了 (false positive)

问题来了: 如果你的批改是"脏"的, 学生还能学吗? 还是会被你带偏?

这篇 paper 就是回答这个问题的。

---

## 核心发现: 一个数字决定生死

### Youden's J 是什么

你的批改质量可以用一个数字概括:

$$J = TPR - FPR$$

- TPR = 正确代码被你正确表扬的概率
- FPR = 错误代码被你错误表扬的概率
- J = 你比"瞎猜"强多少

三种情况:
- **J > 0**: 你比瞎猜强, 虽然有 noise, 但信号还在
- **J = 0**: 你跟瞎猜一样, 信号完全是 noise
- **J < 0**: 你比瞎猜还差, 系统性误导学生

### 最关键的结论

**J = 0 是一条 sharp phase transition 线。**

| J 值 | 发生什么 |
|---|---|
| J > 0 | 学生在学, 只是变慢了, 最终能学到一样的水平 |
| J = 0 | 学生在原地打转, 毫无进展 |
| J < 0 | 学生在"反学习", 越学越差 |

---

## 为什么 GRPO 像自然选择

### 把 LLM 想成一个生态系统

每个 prompt 对 LLM 来说, 都有几种"思路" (reasoning modes)。比如解一道数学题, 可以:
- 思路 A: 正确解法 1
- 思路 B: 正确解法 2  
- 思路 C: 错误解法 1
- 思路 D: 错误解法 2

GRPO 每次采样一组 completions, 比较它们的 reward, 好的加强, 坏的削弱。

**这本质就是 natural selection**: 适者生存。

### 数学上就是 replicator dynamics

$$\dot{p}_i = p_i (f_i - \bar{f})$$

翻译成人话: 第 i 种思路的概率变化 = 它的概率 × (它的 fitness - 平均 fitness)

如果某种思路的 reward 比平均高, 它的概率就增长; 比平均低, 就衰减。

---

## "Rate, not Fate" 是什么意思

### Fate (命运) 不变

只要 J > 0, 不管你的 verifier 有多 noisy, 最终系统都会收敛到 **同一个 basin of attraction** —— 也就是正确解法 dominate, 错误解法消亡。

**Noise 改变的是到达终点的时间, 不改变终点本身。**

### Rate (速度) 变了

$$\frac{\text{noisy 速度}}{\text{perfect 速度}} \propto \frac{1}{J}$$

举个例子:
- J = 1 (完美 verifier): 1x 速度
- J = 0.5: 大约 2x 时间到达同样的终点
- J = 0.1: 大约 10x 时间

**更多的 compute 能 compensate imperfect reward。**

---

## 三种 noise regime 的直觉

### Regime 1: J > 0 (信号还在)

Verifier 虽然有 noise, 但 net direction 是对的。系统在学习, 只是慢。

类比: 你导航偶尔说错一句话, 但大方向对, 你最终能到家, 只是绕了点路。

### Regime 2: J = 0 (纯 noise)

Verifier 给的信息跟 random guessing 一样, 信号互相抵消, 系统原地打转。

类比: 你导航每次都随机指方向, 你在原地转圈。

### Regime 3: J < 0 (反信号)

Verifier 系统性地把对的判错, 错的判对。系统在**反学习**。

类比: 你导航地图是镜像的, 你越走离家越远。

---

## 实验: 把理论扔进现实看

### Setup

- 模型: Qwen2.5-3B
- 任务: Python 代码生成 (OpenR1)
- 用人工注入 noise: 按不同 (TPR, FPR) 翻转 oracle 的判断

### 结果

| J 值 | (FPR, FNR) | pass@1 | 相对提升 |
|---|---|---|---|
| -0.1 | (0.60, 0.50) | 0.16% | **-12.6% (退化!)** |
| 0.0 | (0.50, 0.50) | 13.40% | +0.6% (原地打转) |
| 0.3 | (0.00, 0.70) | 16.00% | +3.2% |
| 0.7 | (0.20, 0.10) | 18.6% | +5.8% |
| 1.0 | (0.00, 0.00) | 20.8% | +8.0% |

**理论预测的 phase transition 在实验中精确复现。**

### 一个细节: False Positive 比 False Negative 更危险

固定 J = 0.3:
- (FPR=0, FNR=0.70): 16.00% ← 只有 false negative
- (FPR=0.70, FNR=0): 14.60% ← 只有 false positive

为什么? 因为 false positive 会**奖励错误的解法**, 让它增长, 跟正确解法竞争。而 false negative 只是漏掉了正确的, 不会主动放大错误。

---

## 几何视角: 为什么 good arm 会 winner-take-all

### Good Arms: 多样性崩塌

正确解法之间会竞争。论文证明, 最终只有一个正确解法存活, 其他都消亡。

数学上:
$$\dot{y}_j = \kappa y_j (y_j - \|y\|_2^2)$$

如果 $y_j > \|y\|_2^2$ (高于"平均的平方"), 它 super-linearly 增长。

**这是 GRPO 的内置行为**: 即使有多种正确解法, 最终也会坍缩到一种。

### Bad Arms: 趋向均匀分布

错误解法之间, 系统反而推向 uniform distribution (max entropy)。

$$\dot{z}_m = -\kappa z_m (z_m - \|z\|_2^2)$$

注意那个**负号**: 这跟 good arms 的行为相反。

直觉: 系统在削弱 bad arms 整体的 mass, 但 bad arms 内部在"摊平", 不会出现某个错误解法 dominate。

---

## Learnability: 什么难度的 prompt 最值得训

### 理论预测

在 noiseless 情况下, 单步 progress:
$$|\Delta p| \propto [p(1-p)]^{3/2}$$

在 $p = 1/2$ 时最大。

翻译: **模型在某个 prompt 上 50-50 对错时, 训练收益最大。**

太简单的 (p ≈ 0): 模型已经会了, 学不到什么。
太难的 (p ≈ 1): 模型全错, 学不到什么。
中间难度的最 learnable。

### 这跟之前的 observation 吻合

- Bae et al. (2025): progress bounds ∝ $p(1-p)$
- Foster et al. (2025): learnability 与 reward variance 关联

**论文给出了一个动力学解释**: 同样的 non-saturation 现象, 既给出高 information content, 又控制 bad mass 的消除速度。

---

## 加入 KL 正则化: 从 phase transition 到 smooth equilibrium

### 没有 KL 时的行为

- J > 0: p → 0 (perfect)
- J < 0: p → 1 (collapse)
- J = 0: p 停在初始点

**这是 sharp phase transition**, 跟物理中的相变类似。

### 加入 KL 后

引入一个 reference bad mass $p_{\text{ref}}$, 加一个 penalty:
$$\dot{p}\big|_{KL} = -\beta p(1-p) \left(\log\frac{p}{1-p} - \log\frac{p_{\text{ref}}}{1-p_{\text{ref}}}\right)$$

**任何 $\beta > 0$ 都会把 sharp boundary 变成 smooth interior equilibrium。**

- J > 0: $p^* \in (0, p_{\text{ref}})$, 接近 0 但不到 0
- J = 0: $p^* = p_{\text{ref}}$
- J < 0: $p^* \in (p_{\text{ref}}, 1)$, 接近 1 但不到 1

**KL 是 stability 工具, 不是 signal quality 的 substitute。** 即使 J < 0, KL 也能 prevent total collapse, 但不能把一个误导的 verifier 变成学习信号。

---

## 几何细节: Shahshahani metric

### 为什么 simplex 有特殊几何

概率 simplex $\Delta^{K+M-1}$ 不是 Euclidean 的。把 mass 从一个 rare arm 移走, 比从 common arm 移走"代价更高"。

Shahshahani metric 形式化了这个直觉:
$$\langle u, v \rangle_{\text{Shah}} = \sum_i \frac{u_i v_i}{p_i}$$

### Softmax Jacobian 就是这个 metric 的 inverse

$$\mathfrak{I}(\mathbf{p}) = \text{Diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$$

这个矩阵:
1. 是 softmax 的 Jacobian
2. 把 gradient 投影到 simplex tangent space
3. 是 Shahshahani metric 的 inverse

**所以 GRPO 的 update 自然就是在 Shahshahani geometry 下的 natural gradient flow。**

---

## 和 Mirror Descent 的联系

### Entropic Mirror Ascent

$$\mathbf{p}^+ = \arg\max_{\mathbf{q}} \left\{\langle \mathbf{A}, \mathbf{q}\rangle - \frac{1}{\eta} D_{KL}(\mathbf{q} \| \mathbf{p})\right\}$$

解出来就是 multiplicative weights:
$$p_i^+ = \frac{p_i \exp(\eta A_i)}{\sum_j p_j \exp(\eta A_j)}$$

### First-Order 展开

$$\mathbf{p}^+ - \mathbf{p} = \eta \mathfrak{I}(\mathbf{p})\mathbf{A} + \mathcal{O}(\eta^2)$$

**这正是 replicator dynamics 的 Euler step。**

**结论**: Mirror descent 是 replicator flow 的离散化, 两者是同一个东西。

---

## "支持壁垒": RLVR 不能无中生有

### 关键观察

在 $J = 1$ (完美 verifier) 下, 闭合解:
$$p(t) \sim \frac{4}{\eta^2 t^2} \to 0$$

但前提是 $p_0 \neq 0, 1$。

**Boundary states $p_0 \in \{0, 1\}$ 是 absorbing 的。**

### 含义

如果 prompt 完全超出 base model 能力 ($p_0 = 1$, 模型完全不会), RL 永远学不起来, 因为它采样不到正确解法, 没有什么可以 amplify。

**RLVR 主要在 sharpen 和 reweight 已有的 reasoning paths, 不能 expand capability beyond initial support。**

这解释了为什么 RLVR 提升 pass@1, 但 large-k coverage 可能 shrink——因为多样性在 collapse。

---

## 实际建议

1. **尽早测量 J**
   - 估算你的 verifier 的 TPR 和 FPR
   - 如果 $J \leq 0$, 不要 scale compute, 先 fix verifier

2. **如果 J > 0, 耐心**
   - noisy-but-informative rewards 只是慢, 最终能到同样的终点
   - More rollouts / steps 能加速

3. **特别小心 false positives**
   - FP 会主动 reward 错误解法, 让它跟正确解法竞争
   - FN 只是漏掉正确解法, 不会主动放大错误
   - 固定 J 下, 高 FPR 比 high FNR 更 harmful

4. **KL 正则化是 stability 工具**
   - 能 prevent extreme collapse
   - 把 sharp phase transition 变成 smooth interior equilibrium
   - 但不能 turn J < 0 into learning signal

5. **Medium-difficulty prompts 最值得训**
   - 模型 50-50 对错的 prompt 收益最大
   - 太简单或太难的收益小

---

## 这篇 paper 的 beauty

它把一个 messy practical problem (noisy reward in RLVR) 用 elegant 理论框架分析:

1. **Multi-armed bandit abstraction**: 把 sequence-level RLVR 变成 tractable 的 bandit 问题
2. **Replicator dynamics**: 自然出现, 跟 evolutionary game theory 连接
3. **Single scalar J**: 决定 phase, 把复杂的 noise structure 压缩成一个数字
4. **Closed-form dynamics**: 给出 explicit trajectories 和 tail rates
5. **Phase transition**: 类比物理中的 critical phenomena
6. **Information geometry**: Shahshahani metric 给出 underlying geometry

**理论是 predictive 的, 不只是 descriptive**: 实验精确复现了 J=0 的 phase transition, FP vs FN 的 asymmetry, 以及 medium-difficulty 的 learnability peak。

---

## 你应该记住的 5 件事

1. **J = TPR - FPR 是唯一关键数字**
2. **J > 0: rate 慢但 fate 不变; J ≤ 0: 系统崩**
3. **False positives 比 false negatives 更危险**
4. **Medium-difficulty prompts (p ≈ 0.5) 最 learnable**
5. **RLVR sharpen 已有 capability, 不能无中生有**

这篇 paper 让你能用一把尺子 (J) 去诊断你的 RLVR pipeline 是否健康, 并给出具体的 intervention 策略。**这就是 good theory 的力量。**

---

# RLV<sup>ε</sup>R: Reinforcement Learning with Verifiable Noisy Rewards 深度解析

Andrej, 这篇论文很有意思——它把 RLVR 的 noisy reward 问题转化为一个 analytically tractable 的 multi-armed bandit / replicator dynamics 问题, 并发现了一个 sharp phase transition, 堪称是 RLVR 的"统计力学"。让我详细讲讲。

---

## 1. Core Question: Rate or Fate?

论文的核心 question 非常 Karpathy-style:

> **当 verifier 是 sloppy 的, noise 是 merely slow down learning (rate), 还是 flip the outcome (fate)?**

这个 question 在 practical RLVR 中极其重要, 因为 real-world 的 verifier 几乎从来不干净:
- Unit tests 只覆盖有限 corner cases
- Human/synthetic labels 有 error
- LLM-as-Judge (RLAIF) 可以被 exploit
- Coding tasks 越难, test coverage 越差

论文的 central answer 是 **"Rate, not Fate"**: 只要 Youden's index $J > 0$, noisy reward 只改变 convergence speed, 不改变 asymptotic basin of attraction。

---

## 2. The Key Quantity: Youden's Index J

### 2.1 Noise Model

定义两个 error rates:

$$\delta_{FN} = Pr(r=0 \mid \text{good}), \quad \delta_{FP} = Pr(r=1 \mid \text{bad})$$

- $\delta_{FN}$: false negative rate (correct solution 被误判为 0)
- $\delta_{FP}$: false positive rate (incorrect solution 被误判为 1)
- $r \in \{0,1\}$: observed noisy binary reward

### 2.2 Youden's Index

$$J := 1 - \delta_{FN} - \delta_{FP} = TPR - FPR \in [-1, 1]$$

变量解释:
- $TPR = 1 - \delta_{FN}$: true positive rate (good solution 被 correctly reward 的概率)
- $FPR = \delta_{FP}$: false positive rate (bad solution 被 incorrectly reward 的概率)
- $J$: Youden's index, 来自 statistical decision theory (Youden 1950)

几何解释: $J$ 是 ROC curve 与 diagonal "random line" 之间的垂直距离, 衡量 verifier 偏离 random guessing 的程度。

三种 regime:
- $J = 1$: perfect rewarder (TPR=1, FPR=0)
- $J = 0$: chance-level, uninformative
- $J < 0$: anti-informative (比 random 还差, 系统性误导)

参考: [Youden's J statistic on Wikipedia](https://en.wikipedia.org/wiki/Youden%27s_J_statistic)

---

## 3. Mean-Field Dynamics: GRPO as Natural Selection

### 3.1 Binary Setup: Good vs Bad

最简单的 setup: LLM 生成 "good" 或 "bad" solution。设 $p = Pr(\text{bad})$, controlled by logit $z$:
$$p = \sigma(z) = \frac{1}{1+e^{-z}}$$

REINFORCE-style update 给出:
$$\Delta z \propto E[\hat{A} \nabla_z \log \pi(a)]$$

Score function:
- $\nabla_z \log \pi(\text{bad}) = 1 - p$
- $\nabla_z \log \pi(\text{good}) = -p$

定义 conditional expected advantages:
$$f(\text{bad}) = E[\hat{A} \mid \text{bad}], \quad f(\text{good}) = E[\hat{A} \mid \text{good}]$$

Full expectation:
$$E[\hat{A} \nabla_z \log \pi(a)] = p(1-p)(f(\text{bad}) - f(\text{good}))$$

转 continuous time ($\dot{p} = p(1-p)\dot{z}$):

### 3.2 GRPO Dynamics (Binary)

$$\boxed{\dot{p}(t) = -\eta [p(t)(1-p(t))]^2 (f(\text{good}) - f(\text{bad}))}$$

变量解释:
- $\dot{p}$: bad mass 的变化率
- $\eta$: learning rate
- $p(t)$: 时刻 t 的 bad mass
- $f(\text{good}), f(\text{bad})$: conditional expected normalized advantages

### 3.3 Noisy Reward 的关键 Effect

在 noisy reward 下, advantage gap 变成纯几何形式:
$$E[\hat{A} \mid \text{good}] - E[\hat{A} \mid \text{bad}] = \frac{J}{\sigma(p)}$$

变量:
- $\sigma(p) = \sqrt{q(p)(1-q(p))}$: reward standard deviation
- $q(p) = (1-\delta_{FN}) - Jp$: expected reward given bad mass p

代入得到 **核心 ODE**:
$$\boxed{\dot{p} = -\eta \frac{J}{\sigma(p)} p^2 (1-p)^2, \quad J = TPR - FPR}$$

这就是 **"Law of Motion"**: $J$ 作为 learning 的 signed friction coefficient。

---

## 4. Three Learning Regimes: Phase Transition

### 4.1 三种 regime

$$\begin{cases}
J > 0 &\Rightarrow \dot{p} < 0 \Rightarrow \text{Learning: bad mass shrinks} \\
J = 0 &\Rightarrow \dot{p} = 0 \Rightarrow \text{Neutral: pure drift} \\
J < 0 &\Rightarrow \dot{p} > 0 \Rightarrow \text{Anti-learning: bad mass grows}
\end{cases}$$

这是一个 **sharp phase transition** at $J = 0$ (TPR = FPR), 类似于物理系统中的相变。

### 4.2 Bifurcation Analysis

Fixed points: $p^* = 0$ (all good) 和 $p^* = 1$ (all bad)

- **J > 0**: $p = 0$ globally attracting, $p = 1$ repels
- **J < 0**: basin 反转, $p = 1$ attracts, $p = 0$ repels
- **J = 0**: knife edge, 整个 [0,1] 是 continuum of neutrally stable fixed points

### 4.3 Noise-Free Closed Form (J = 1)

$$p(t) = \frac{1}{2} + \frac{1}{2} \frac{\varphi(p_0) - \frac{\eta}{2}t}{\sqrt{4 + (\varphi(p_0) - \frac{\eta}{2}t)^2}}$$

其中 $\varphi(p) = \frac{2p-1}{\sqrt{p(1-p)}}$

Late-time asymptotic:
$$p(t) \sim \frac{4}{\eta^2 t^2} \to 0, \quad t \to \infty$$

**universal $t^{-2}$ tail**: accuracy 以 polynomial rate 趋向 1。

### 4.4 Asymptotic Tails under Noise

定义 boundary variances:
$$\sigma_0 = \sqrt{(1-\delta_{FN})\delta_{FN}}, \quad \sigma_1 = \sqrt{\delta_{FP}(1-\delta_{FP})}$$

**Case (i)**: $J > 0$, attractor at $p = 0$
- Nondegenerate noise ($\delta_{FN} > 0$): $p(t) \sim \frac{\sigma_0}{\eta J} \cdot \frac{1}{t}$ (**$t^{-1}$ tail**)
- Variance-degenerate ($\delta_{FN} = 0$): $p(t) \sim \frac{4}{\eta^2 J} \cdot \frac{1}{t^2}$ (**$t^{-2}$ tail**)

**Case (ii)**: $J < 0$, attractor at $p = 1$
$$u(t) = 1 - p(t) \sim \frac{\sigma_1}{\eta|J|} \cdot \frac{1}{t}$$

关键 insight: **reward variance 在 attractor 处是否 vanish 决定了 tail rate**。

---

## 5. Rate, Not Fate

### 5.1 Time Rescaling

比较 noisy 和 noise-free dynamics:
$$\frac{\dot{p}_{\text{noisy}}}{\dot{p}_{\text{perfect}}} \propto \frac{1}{J}$$

含义: $J = 0.5$ 时, noisy system 需要约 2x compute 来 trace 同样的 trajectory。

**Additional compute 可以 compensate imperfect reward signal**。

### 5.2 Maximal Learnability at Intermediate Difficulty

在 noiseless regime ($J=1$):
$$|\Delta p| \propto [p(1-p)]^{3/2}$$

maximized at $p^* = 1/2$。

**"Medium-difficulty" prompts (model 50-50 between good/bad) 最 learnable**。

这与 recent observations 一致:
- Bae et al. (2025): progress bounds ∝ $p(1-p)$, empirically $p(x) ≈ 0.5$ 最 learnable
- Foster et al. (2025): learnability 与 reward variance 关联, Bernoulli $Var(r) = q(1-q)$ 在 $q=1/2$ 最大

参考: [Online difficulty filtering for reasoning RL (Bae et al.)](https://arxiv.org/abs/2504.03380)

---

## 6. LLM as Multi-Armed Bandit

### 6.1 Coarse-Graining into Modes

Fix prompt $x$, sample completion $y \sim \pi_\omega(\cdot | x)$。定义 coarse-graining map:
$$\phi: \mathcal{V}_{\leq L_{\max}} \to \mathcal{H} = \{h_1, \ldots, h_{K+M}\}$$

每个 mode $h_i$ 是一个 "arm" in the bandit。

### 6.2 Mode Policy

$$\pi_\theta(h_i | x) = \text{softmax}(\theta)_i$$

- $\theta = (\theta_1, \ldots, \theta_{K+M})$: effective logits
- $K$ good modes + $M$ bad modes

### 6.3 Decomposition

Partition: $\mathcal{H} = \mathcal{H}^+ \cup \mathcal{H}^-$, $|\mathcal{H}^+| = K$, $|\mathcal{H}^-| = M$

Aggregate masses:
$$\alpha = \sum_{h \in \mathcal{H}^+} \pi_\theta(h|x), \quad p = \sum_{h \in \mathcal{H}^-} \pi_\theta(h|x) = 1 - \alpha$$

Within-block distributions:
$$y_i = \frac{\pi_\theta(h_i|x)}{\alpha} \in \Delta^{K-1}, \quad z_j = \frac{\pi_\theta(h_j|x)}{p} \in \Delta^{M-1}$$

Full distribution: $\mathbf{p} = (\alpha y_1, \ldots, \alpha y_K, p z_1, \ldots, p z_M)$

---

## 7. Geometric Flow on the Probability Simplex

### 7.1 Shahshahani Metric

Simplex $\Delta^{K+M-1}$ 上的 Riemannian metric:
$$\langle u, v \rangle_{\text{Shah}, \mathbf{p}} = \sum_{i=1}^{K+M} \frac{u_i v_i}{p_i}$$

这正是 **Fisher information metric** of the categorical family。

### 7.2 Softmax Jacobian as Inverse Metric

$$\mathfrak{I}(\mathbf{p}) = \text{Diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$$

性质:
1. Softmax 的 Jacobian
2. Projects 到 simplex tangent space
3. Shahshahani metric 的 inverse

### 7.3 GRPO Dynamics on Simplex

$$\dot{\mathbf{p}} = \eta \mathfrak{I}(\mathbf{p})^2 \mathbf{A}$$

等价的 replicator form:
$$\dot{\mathbf{p}} = \eta \mathbf{p} \odot [\mathfrak{I}(\mathbf{p})\mathbf{A} - \langle \mathbf{p}, \mathfrak{I}(\mathbf{p})\mathbf{A}\rangle \mathbf{1}]$$

两个关键性质:
1. **Multiplicativity**: simplex faces 不变
2. **Relative Performance**: mass 严格按 relative advantage 流动

---

## 8. Decoupling: Shape vs Mass

### 8.1 Decoupled ODE System

应用 $\mathbf{p} = ((1-p)y, pz)$ 分解:

$$\boxed{\dot{y} = +\kappa(p) y \odot (y - \|y\|_2^2 \mathbf{1})} \quad \text{(Diversity Collapse)}$$

$$\boxed{\dot{z} = -\kappa(p) z \odot (z - \|z\|_2^2 \mathbf{1})} \quad \text{(Entropy Increase)}$$

$$\boxed{\dot{p} = -\eta \frac{J}{\sigma(p)} [p(1-p)]^2 (\|y\|_2^2 + \|z\|_2^2)} \quad \text{(Bad Mass Evolution)}$$

其中 $\kappa(p) = \eta \frac{J}{\sigma(p)} p(1-p)$

### 8.2 三种几何力

1. **Diversity Collapse in Good Arms** ($\dot{y}$):
   - $y_i > \|y\|_2^2$ 的 arms super-linearly grow
   - 收敛到单一 dominant good arm
   - $y_i(t) \to 0$ for $i \notin S^*$

2. **Entropy Increase in Bad Arms** ($\dot{z}$):
   - Negative feedback loop, 推向 uniform distribution
   - $z(t) \to \frac{1}{M}\mathbf{1}$ (maximum entropy)

3. **Bad Mass Evolution** ($\dot{p}$):
   - $J > 0$ 时单调下降
   - Rate 被 $\|y\|_2^2 + \|z\|_2^2$ 调制

### 8.3 Winner-Take-All

在 $J > 0$ regime:
- Good arms: 最初概率最高的 arm 最终 dominate (symmetry breaking)
- Bad arms: 趋向 uniform

Figure 2 展示了这个 striking structural property。

---

## 9. Theorem 6.1: Full Multi-Bad-Arm GRPO ODE

### 9.1 Main Theorem

**Theorem 6.1** (Bad-mass ODE, internal-time logit form):

$$\dot{p}(t) = -\eta \frac{J}{\sigma(p(t))} [p(t)(1-p(t))]^2 C_{\text{geo}}(t) + \mathcal{O}(\eta^2)$$

其中 geometry factor:
$$C_{\text{geo}}(t) := s_2(t) + t_2(t) \in \left[\frac{1}{K} + \frac{1}{M}, 2\right]$$

- $s_2(t) = \|y(t)\|_2^2 \in [1/K, 1]$
- $t_2(t) = \|z(t)\|_2^2 \in [1/M, 1]$

### 9.2 Internal-Time Logit Form

定义 logit $L(t) = \log\frac{p(t)}{1-p(t)}$ 和 internal time:
$$\tau(t) = \int_0^t \eta \frac{|J|}{\sigma(p(u))} p(u)(1-p(u)) du$$

则:
$$\frac{dL}{d\tau} = -\text{sign}(J) C_{\text{geo}}(\tau) = -\text{sign}(J)(s_2(\tau) + t_2(\tau))$$

### 9.3 PPO Clipping 和 Importance Sampling

**关键结论**: 在 small-step regime ($\eta \ll \varepsilon, \varepsilon'$), clipping和 IS 只影响 $\mathcal{O}(\eta^2)$ 项, 不改变 leading-order drift。

---

## 10. KL Regularization: Smoothing the Phase Transition

### 10.1 KL-Regularized ODE

加入 KL penalty toward $p_{\text{ref}}$:
$$\dot{p} = -\eta \frac{J}{\sigma(p)} [p(1-p)]^2 C(y,z) - \beta p(1-p)\left(\log\frac{p}{1-p} - \log\frac{p_{\text{ref}}}{1-p_{\text{ref}}}\right)$$

变量:
- $\beta$: KL penalty strength
- $p_{\text{ref}}$: reference bad mass

### 10.2 Interior Equilibrium

平衡条件:
$$\beta(L(p^*) - L(p_{\text{ref}})) = -\eta \frac{J}{\sigma(p^*)} p^*(1-p^*) C(y,z)$$

**Unique interior equilibrium** $p^* \in (0,1)$ for any $\beta > 0$:
- $J > 0$: $0 < p^* < p_{\text{ref}}$
- $J = 0$: $p^* = p_{\text{ref}}$
- $J < 0$: $p_{\text{ref}} < p^* < 1$

### 10.3 Asymptotic Regimes

**Strong KL** ($\beta \to \infty$):
$$p^* \approx p_{\text{ref}} - \frac{\eta J}{\beta} \frac{[p_{\text{ref}}(1-p_{\text{ref}})]^2}{\sigma(p_{\text{ref}})} C(y,z)$$

**Weak KL** ($\beta \downarrow 0$, $J < 0$):
$$1 - p^* \sim \frac{\beta}{c} \log\frac{c}{\beta}, \quad c = \frac{-\eta J}{\sigma(1)} C(y,z) > 0$$

即使 infinitesimal $\beta > 0$ 也能 prevent total collapse to $p^* = 1$。

---

## 11. Experiments

### 11.1 Setup

| Component | Detail |
|---|---|
| Base Model | Qwen2.5-3B |
| Data | OpenR1 Python coding (10,239 train / 594 val) |
| Algorithm | GRPO (VeRL library) |
| Group Size G | 8 rollouts/prompt |
| Steps | 1410 (2 epochs) |
| KL coeff β | 0.0 |
| Clipping | (0.2, 0.2) |
| LR | 10⁻⁶ |
| Evaluation | E[pass@1], 5 seeds |

### 11.2 Synthetic Noise Injection

```python
def NoisyCheck(program):
    z = Oracle(program)  # ground truth
    if z == 1:
        r = Bernoulli(TPR)
    else:
        r = Bernoulli(FPR)
    return r
```

### 11.3 Results Table

| J | (FPR, FNR) | E[Pass@1] | Improvement |
|---|---|---|---|
| -0.1 | (0.60, 0.50) | 0.16% | -12.6% |
| 0.0 | (0.50, 0.50) | 13.40% | +0.6% |
| 0.3 | (0.00, 0.70) | 16.00% | +3.2% |
| 0.3 | (0.70, 0.00) | 14.60% | +1.8% |
| 0.7 | (0.20, 0.10) | 18.6% | +5.8% |
| 1.0 | (0.00, 0.00) | 20.8% | +8.0% |

### 11.4 Key Observations

1. **Phase transition confirmed at J = 0** ($\mathcal{H}_1$)
2. **J < 0**: accuracy actively degrades (-12.6%), anti-learning
3. **J = 0**: minimal improvement (+0.6%), neutral drift
4. **J > 0**: monotonic improvement, stronger J → faster convergence
5. **FN 比 FP 更可容忍**: at fixed J=0.3, (FPR=0, FNR=0.70) achieves 16.00% vs (FPR=0.70, FNR=0) achieves 14.60%

这与理论一致: $\delta_{FN} = 0$ 给 $t^{-2}$ tail, $\delta_{FN} > 0$ 给 $t^{-1}$ tail (slower)。

参考: [VeRL framework](https://arxiv.org/abs/2409.19256), [OpenR1](https://github.com/huggingface/open-r1)

---

## 12. Connection to Evolutionary Game Theory

### 12.1 Replicator Dynamics

$$\dot{p}_i(t) = p_i(t)(f_i(\mathbf{p}(t)) - \bar{f}(\mathbf{p}(t)))$$

- $f_i$: fitness of type $i$
- $\bar{f} = \sum_j p_j f_j$: average fitness
- Above-average fitness 的 type 增长, below-average 的 衰减

**GRPO 就像 natural selection**。

### 12.2 Wright-Fisher Diffusion (Genetic Drift)

Finite sampling ($G$ rollouts) 引入 stochasticity:
$$dp = \dot{p} dt + \frac{\eta\sqrt{\nu}}{\sqrt{G}} \Sigma(p)^{1/2} dW_t$$

- $\Sigma(p) = \text{Diag}(p) - pp^\top$
- Fluctuations $\sim \mathcal{O}(G^{-1/2})$

参考: [Replicator dynamics](https://en.wikipedia.org/wiki/Replicator_equation), [Wright-Fisher model](https://en.wikipedia.org/wiki/Genetic_drift)

---

## 13. Lyapunov Analysis (Appendix E)

### 13.1 Potential Function

定义 scalar potential:
$$F(\mathbf{p}) = F(s(\mathbf{p})), \quad F'(s) = \Delta(s)$$

其中 $s(\mathbf{p}) = \sum_{j \leq K} p_j$ (good mass), $\Delta(s) = J/\sigma(s)$。

### 13.2 Lyapunov Identity

$$\frac{d}{dt}F(\mathbf{p}(t)) = \eta \|\mathfrak{I}(\mathbf{p}(t))\mathbf{A}(\mathbf{p}(t))\|_2^2 \geq 0$$

### 13.3 Sign of $\dot{s}$

$$\dot{s} = \eta \Delta(s) [s(1-s)]^2 (\|y\|_2^2 + \|z\|_2^2)$$

**Sign of $\dot{s}$ 完全由 sign of J 决定**。

### 13.4 Quantitative Tail Bound

For $J > 0$, after finite transient $T_{1/2}$:
$$P_{\text{bad}}(t) \leq \left(\frac{1}{P_{\text{bad}}(T_{1/2})} + \frac{\eta J}{4\sigma_{\max}}\left(\frac{1}{K}+\frac{1}{M}\right)(t - T_{1/2})\right)^{-1}$$

这给出 $\mathcal{O}(1/t)$ 收敛保证。

---

## 14. Inner Dynamics of Good Arms (Appendix I)

### 14.1 Within-Good ODE

$$\dot{y} = \kappa(p)(y \odot y - \|y\|_2^2 y)$$

在 internal time $\tau$:
$$\frac{dy_j}{d\tau} = y_j(y_j - s_2), \quad s_2 = \sum_i y_i^2$$

### 14.2 Closed-Form Solution

令 $u_j = 1/y_j$, 则:
$$y_j(\tau) = \frac{q_j / (1 - I(\tau)q_j)}{\sum_\ell q_\ell / (1 - I(\tau)q_\ell)}$$

其中 $q = y(0)$, $I(\tau)$ 是 strictly increasing scalar。

### 14.3 Order Preservation

$$\delta_{ij}' = \delta_{ij}(y_i + y_j - s_2)$$

**Sign of $\delta_{ij}$ 不变**: 最初最大的 arm 永远是 maximizer。

### 14.4 Stability Summary (Table 2)

| Equilibrium | J condition | Stability |
|---|---|---|
| Uniform $y_j = 1/K$ | $J > 0$ | Unstable (diversity collapse) |
| Uniform $y_j = 1/K$ | $J < 0$ | Stable (diversity preserved) |
| Vertex $y = e_j$ | $J > 0$ | Stable (specialization) |
| Vertex $y = e_j$ | $J < 0$ | Unstable (reverts to mixture) |

---

## 15. Inner Dynamics of Bad Arms (Appendix J)

Bad block 是 good block 的 sign-reversed analogue:

$$\frac{dz}{d\tau} = -(z \odot z - \|z\|_2^2 z)$$

### Stability (Table 3)

| Equilibrium | J condition | Stability |
|---|---|---|
| Uniform $z_m = 1/M$ | $J > 0$ | Stable (bad mass diffuses) |
| Uniform $z_m = 1/M$ | $J < 0$ | Unstable |
| Vertex $z = e_m$ | $J > 0$ | Unstable |
| Vertex $z = e_m$ | $J < 0$ | Stable (bad-mode collapse) |

**J > 0 时 bad mass 趋向 uniform (max entropy), good mass 趋向 vertex (concentration)**。

---

## 16. Physical-Time Asymptotics (Theorem I.19)

Assume $\sigma(p) \sim \sigma_0 p^\gamma$ as $p \downarrow 0$, with $a = 1 + 1/M$:

| Regime | $p(t)$ | $1 - y_m(t)$ |
|---|---|---|
| $\gamma < 1$ | $\asymp t^{-1/(1-\gamma)}$ | $\asymp t^{-1/[a(1-\gamma)]}$ |
| $\gamma = 1$ | $\asymp e^{-(aJ/\sigma_0)t}$ | $\asymp e^{-(J/\sigma_0)t}$ |
| $\gamma > 1$ | $\asymp (t_\infty - t)^{1/(\gamma-1)}$ | $\asymp (t_\infty - t)^{1/[a(\gamma-1)]}$ |

**Power-law exponents for $p(t)$ universal (independent of K, M)**。

---

## 17. Mirror Descent Connection (Appendix H)

### 17.1 Entropic Mirror Ascent

$$\mathbf{p}^+ = \arg\max_{\mathbf{q} \in \Delta} \left\{\langle \mathbf{A}, \mathbf{q}\rangle - \frac{1}{\eta}D_{KL}(\mathbf{q} \| \mathbf{p})\right\}$$

Closed form (multiplicative weights):
$$p_i^+ = \frac{p_i \exp(\eta A_i)}{\sum_j p_j \exp(\eta A_j)}$$

### 17.2 First-Order Expansion

$$\mathbf{p}^+ - \mathbf{p} = \eta \mathfrak{I}(\mathbf{p})\mathbf{A} + \mathcal{O}(\eta^2)$$

这正是 replicator flow 的 Euler step。

**Mirror descent 是 Shahshahani natural gradient flow 的离散化**。

参考: [Mirror descent](https://en.wikipedia.org/wiki/Mirror_descent), [Natural gradient](https://en.wikipedia.org/wiki/Natural_gradient)

---

## 18. Limitations and Open Questions

### 18.1 Stated Limitations

1. **Oracle imperfection**: finite test suite 引入 systematic bias in (TPR, FPR) estimation
2. **Context length effects**: $J < 0$ 时长 response 超出 token limit, VeRL 给 reward 0 + high clipping → systematic FN → shifts effective J downward
3. **Generalization**: 仅测试 Python coding + Qwen2.5-3B
4. **Time-dependent noise**: 实验用 fixed noise rates, 实际中 TPR(t), FPR(t) 可能 drift

### 18.2 Open Directions

1. Math reasoning, creative writing with LLM-as-Judge
2. Larger models
3. Co-evolutionary dynamics (policy + reward model)
4. Asymmetric noise 的更精细分析

---

## 19. Practical Takeaways for RLVR

### 19.1 Diagnostic Checklist

1. **Early measure J = TPR - FPR**
2. **If $J \leq 0$**: scaling RL compute 不会 fix 问题, 会 stagnate 或 degrade
3. **If $J > 0$**: compute helps (mostly) by buying time
4. **False positives 尤其 dangerous**: holding J fixed, high FPR 比 high FNR 更 damaging
5. **Use KL regularization for stability**, 不是 signal quality 的 substitute

### 19.2 Key Insight: RLVR Cannot Create New Capability

Equation 8 暴露 **support barrier**:
- Boundary states $p_0 \in \{0, 1\}$ 是 absorbing
- 如果 $1 - p_0 = 0$ (prompt 超出 base model capability), dynamics trapped at $p(t) = 1$, learning never takes off

**RLVR 主要 sharpen/reweight 已有 reasoning paths, 不能 expand capability beyond initial support**。

这与 Yue et al. (2025) 的发现一致: RLVR boosts pass@1 while large-k coverage can shrink。

参考: [Does RL really incentivize reasoning capacity beyond base model?](https://arxiv.org/abs/2504.13837)

---

## 20. Broader Connections

### 20.1 Information Geometry
- Shahshahani metric = Fisher-Rao metric on categorical family
- Natural gradient = replicator dynamics
- KL divergence 是 Bregman divergence for this geometry

### 20.2 Evolutionary Dynamics
- Replicator dynamics from evolutionary game theory
- Wright-Fisher diffusion from population genetics
- Genetic drift = finite sampling noise

### 20.3 Statistical Decision Theory
- Youden's J from ROC analysis
- Signal detection theory
- Diagnostic test evaluation

### 20.4 Optimization Theory
- Mirror descent / multiplicative weights
- Natural gradient methods
- Trust region methods (PPO clipping)

### 20.5 Related RLVR Papers
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [RLVR implicitly incentivizes correct reasoning](https://arxiv.org/abs/2506.14245)
- [Crossing the reward bridge](https://arxiv.org/abs/2503.23829)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Let's verify step by step (PRM)](https://arxiv.org/abs/2305.20050)
- [RLAIF](https://arxiv.org/abs/2309.00267)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)

---

## 21. Intuitive Summary

这篇论文的核心 intuition:

**GRPO 是 natural selection 的 analogue**:
- 每个 reasoning mode 是一个 species
- Reward 是 fitness
- Group normalization 是 relative fitness
- Noisy reward 是 noisy environment

**Youden's J 是信号的"方向"**:
- $J > 0$: 信号指向正确方向, natural selection 工作, 只是变慢
- $J = 0$: 信号是 noise, 漂移
- $J < 0$: 信号指向错误方向, natural selection 反向工作

**Phase transition at J = 0** 类似于:
- 物理中的 phase transition
- Critical phenomena
- Bifurcation in dynamical systems

**Rate, not Fate** 类似于:
- Simulated annealing 中的 temperature
- Gradient descent 中的 learning rate
- 只影响速度, 不影响 destination (if basin is correct)

---

## 22. Final Thoughts

这篇论文的 beauty 在于:
1. **Minimal abstraction**: multi-armed bandit view 抓住 essence
2. **Sharp predictions**: single scalar $J$ 决定 phase
3. **Closed-form dynamics**: replicator flow 给出 explicit trajectories
4. **Practical relevance**: 直接指导 RLVR pipeline design

对 Karpathy 来说, 这论文展示了 **how theoretical analysis can yield clean, operational insights for practical RLVR**——measure J early, if $J \leq 0$ don't waste compute, if $J > 0$ be patient, watch out for false positives, use KL for stability.

---

### Key References

1. [Paper GitHub: cognichip/Noisy-RL](https://github.com/cognichip/Noisy-RL)
2. [Youden (1950) - Index for rating diagnostic tests](https://doi.org/10.3322/canclinrev.3.1.32)
3. [Shahshahani (1979) - A new mathematical framework for linkage and selection](https://www.ams.org/books/memo/0211)
4. [Amari et al. (2019) - Fisher information and natural gradient](https://arxiv.org/abs/1808.07172)
5. [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300)
6. [PPO](https://arxiv.org/abs/1707.06347)
7. [VeRL framework](https://arxiv.org/abs/2409.19256)
8. [OpenR1](https://github.com/huggingface/open-r1)
9. [Replicator equation](https://en.wikipedia.org/wiki/Replicator_equation)
10. [Wright-Fisher model](https://en.wikipedia.org/wiki/Genetic_drift)
11. [Mirror descent](https://en.wikipedia.org/wiki/Mirror_descent)
12. [Does RL really incentivize reasoning?](https://arxiv.org/abs/2504.13837)
13. [Online difficulty filtering](https://arxiv.org/abs/2504.03380)
14. [RLAIF](https://arxiv.org/abs/2309.00267)
15. [Constitutional AI](https://arxiv.org/abs/2212.08073)

这篇 paper 真的是把 RLVR 的 noisy reward 问题用 elegant 的 dynamical systems / information geometry 框架给分析透了, 是近年来 RLVR theory 中少见的 clean 且 predictive 的工作。
