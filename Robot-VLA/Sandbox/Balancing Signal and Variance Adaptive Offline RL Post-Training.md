---
source_pdf: Balancing Signal and Variance Adaptive Offline RL Post-Training.pdf
paper_sha256: a8f1737992a0e449415bd93fbe78b9fb31bb8d4c2cc9b9368349e0d95f973758
processed_at: '2026-08-18T02:15:37-07:00'
target_folder: Robot-VLA/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ARFM 用人话讲

## 一句话总结

**给π₀这种flow model做fine-tune的时候，用RL advantage给数据"加权"，但权重容易爆掉，所以搞了个自适应算法动态调temperature，让信号够强但训练不崩。**

---

## 背景为什么需要这个

π₀是个flow matching的VLA model，本质上它学的是一个 **动作轨迹的概率分布**。原来的fine-tune就是behavior cloning——你给我什么数据我就模仿什么，不管数据好坏。

问题来了：offline dataset里有些trajectory明显比另一些好（比如更接近成功、更平滑、更快完成任务）。纯imitation learning看不出来这个区别，把好数据和烂数据一视同仁地学。

RL的优势恰恰是能判断"这条轨迹比平均水平好多少"——这就是advantage。于是自然想法：**用advantage给每个样本打个分，好样本权重大一点，烂样本权重小一点。**

---

## 核心问题：权重怎么设

最直觉的做法是softmax weighting：

$$w_i = \frac{\exp(\alpha \cdot R_i^*)}{\sum_j \exp(\alpha \cdot R_j^*)}$$

- $R_i^*$：第 $i$ 个样本的advantage（好多少）
- $\alpha$：temperature，控制权重"尖锐程度"

**$\alpha$ 太小**：所有权重都差不多等于 $1/B$，等于没加权，白搞
**$\alpha$ 太大**：softmax变成one-hot，只盯着最好的那一个样本学，gradient variance爆炸，训练直接崩

这就是经典的bias-variance tradeoff，但没人给你告诉你该选多大的 $\alpha$。

---

## ARFM干了什么

**核心：每个batch动态算出最优的 $\alpha$。**

怎么算？构造一个目标函数：

$$J(\alpha) = \underbrace{\text{Var}(\nabla_\theta L)}_{\text{要小，防爆炸}} - \lambda \underbrace{S(\alpha)}_{\text{要大，保信号}}$$

- 第一项：gradient的方差，越小训练越稳
- 第二项：RL信号强度，越大说明advantage被保留得越好
- $\lambda$：两者之间的tradeoff knob

然后做了三个假设（advantage是Gaussian、loss是Gaussian、batch够大），在Gaussian的MGF下推出了closed form：

$$J(\alpha) = \sigma_L^2 \left[e^{2\alpha^2 \sigma_R^2} - e^{\alpha^2 \sigma_R^2}\right] - \lambda \alpha \sigma_R^2$$

- $\sigma_R^2$：当前batch的advantage方差
- $\sigma_L^2$：当前batch的flow loss方差

对 $J(\alpha)$ 求导=0，得到一个非线性方程，没有闭式解但是单调，所以用 **bisection二分法** 20次迭代就能解出来。

**直觉**：
- 当advantage方差 $\sigma_R$ 大的时候（数据质量差异大），$\alpha^*$ 自动变小——因为差异大时softmax容易爆炸
- 当loss方差 $\sigma_L$ 大的时候，也会影响 $\alpha$
- 完全是data-driven的自适应

---

## 为什么ReinboT不行

ReinboT的做法是把return-to-go当作condition喂给模型，类似Decision Transformer那种"预测未来return"的思路。

但π₀是flow model，生成的是 **整个轨迹的vector field**。return-to-go只能间接影响生成方向，相当于在vector field外面套了层壳，效率低。

ARFM直接在loss层面reweight，相当于 **直接改变target distribution**：

$$q(A|o) \propto p(A|o) \cdot \exp(\alpha R^*)$$

这比"条件生成"更直接，因为flow matching本来就是distribution learning。

---

## 实验效果

几个关键数字：

**Multi-task**：92.1% vs π₀的88.1%，+4.5%
- Long suite提升最大（74.2→82.6），因为长horizon任务advantage信号价值最高

**加noise鲁棒性**：+11.4%
- energy weighting让policy更集中在高质量区域，对扰动更稳

**Few-shot 10-shot**：27.7% vs π₀的22.1%，+25%
- 数据少的时候，每条数据都要榨干价值，adaptive α最大化利用

**Continual learning遗忘**：NBT降38%
- 高return的旧知识有更大的"权重惯性"，不容易被新task覆盖

---

## 真正的insight

1. **Flow matching + energy weighting 天然配对**：flow matching学distribution density，energy weighting直接重塑density，数学上比autoregressive model上加RL更自然

2. **Adaptive temperature是关键**：固定α要么太弱要么太强，自适应是唯一出路

3. **Gaussian假设其实是合理的**：post-training阶段loss已经收敛到低方差，advantage标准化后也接近Gaussian，这两个假设没有很离谱

4. **Bisection很轻量**：每个batch多算20次函数求值，开销几乎可以忽略

---

## 局限

- 13个reward component要手工设计权重，这块还是人工
- Gaussian假设在多模态reward下会失效
- 只适用于flow model，autoregressive VLA要另外想办法
- batch size 16其实不大，统计量估计可能不够准

---

## 类比

跟LLM的DPO有点像：DPO也是直接用preference信号重塑distribution，绕开reward model和PPO的复杂性。ARFM同理——直接用advantage重塑action distribution，绕开critic和policy gradient的复杂性。只不过ARFM是在flow matching的连续空间里做，DPO是在discrete token空间里做。

---

# ARFM: VLA Flow Model的自适应Offline RL Post-Training深度解析

## 1. Paper核心定位与动机

这篇paper针对 **VLA (Vision-Language-Action) flow model** 的post-training问题，核心对象是 **π₀ model** 这类基于trajectory vector field的flow matching policy。作者的核心insight在于：**flow matching本质上是distribution-level的学习**，单纯的imitation learning (behavior cloning) 无法挖掘offline dataset中的quality distribution structure，而RL advantage signal恰好能弥补这一点。

但直接把RL信号塞进flow loss会出问题——这是paper的关键观察：
- α (scaling factor) 太小 → RL advantage被淹没，等于没加
- α 太大 → softmax weight指数爆炸 → gradient variance爆炸 → training崩溃

ARFM的核心贡献是构造了一个 **bias-variance tradeoff objective**，并推导出一个 **bisection iteration** 来自适应求解每个batch的最优α。

**Web reference:**
- π₀ paper: https://arxiv.org/abs/2410.24164
- Energy-Weighted Flow Matching: https://arxiv.org/abs/2503.04975
- ReinboT (baseline): https://arxiv.org/abs/2505.07395
- Flow Matching original: https://arxiv.org/abs/2210.02747

---

## 2. 基础架构解析：Energy-Weighted Flow Matching

### 2.1 Flow Matching的连续性方程

paper从flow matching的基础定义出发。概率密度路径 $p: [0,1] \times \mathbb{R}^d \to \mathbb{R}_{\geq 0}$，向量场 $\mathbf{v}_t: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ 满足 **连续性方程 (continuity equation)**：

$$\frac{d}{dt}p_t(\mathbf{x}) + \text{div}\big(\mathbf{v}_t(\mathbf{x}) p_t(\mathbf{x})\big) = 0$$

- $p_t(\mathbf{x})$: 时刻 $t$ 的概率密度
- $\mathbf{v}_t(\mathbf{x})$: 生成该密度的向量场
- $\text{div}$: 散度算子
- $t \in [0,1]$: flow的时间参数，$t=0$ 是噪声，$t=1$ 是数据

**Intuition**: 这个方程描述了概率"流体"如何随时间演化，向量场 $\mathbf{v}_t$ 就像是流体的速度场。

### 2.2 Energy-Guided Distribution

定义energy-guided分布：
$$q_0(\mathbf{x}_0) \propto p_0(\mathbf{x}_0) \exp(-\beta \mathcal{E}(\mathbf{x}_0))$$

- $p_0(\mathbf{x}_0)$: 原始数据分布（behavior policy的分布）
- $\mathcal{E}(\cdot)$: energy function（在ARFM中就是RL advantage的负值）
- $\beta$: inverse temperature，控制energy影响的强度
- $q_0$: energy-guided distribution，是我们要学习的目标分布

**Intuition**: 这就是Boltzmann分布的形式，低energy（高return）的样本被放大。

### 2.3 Theorem 1：Marginal Vector Field的闭式解

$$\hat{\mathbf{u}}_t(\mathbf{x}) = \int_{\mathbf{x}_0} p_{0t}(\mathbf{x}_0|\mathbf{x}) \mathbf{u}_{t0}(\mathbf{x}|\mathbf{x}_0) \frac{\exp(-\beta \mathcal{E}(\mathbf{x}_0))}{\exp(-\mathcal{E}_t(\mathbf{x}))} d\mathbf{x}_0$$

其中 intermediate energy:
$$\mathcal{E}_t(\mathbf{x}) = -\log \mathbb{E}_{p_{0t}(\mathbf{x}_0|\mathbf{x})}[\exp(-\beta \mathcal{E}(\mathbf{x}_0))]$$

- $p_{0t}(\mathbf{x}_0|\mathbf{x})$: backward conditional，给定当前 $\mathbf{x}$ 推断原始 $\mathbf{x}_0$
- $\mathbf{u}_{t0}(\mathbf{x}|\mathbf{x}_0)$: conditional vector field（已知是Optimal Transport形式）
- $\mathcal{E}_t(\mathbf{x})$: 时间 $t$ 处的"effective energy"，是原始energy在噪声扰动下的期望

**问题**: 这个闭式解不可计算，因为 $\mathcal{E}_t(\mathbf{x})$ 未知。

### 2.4 Theorem 2：Loss等价性（关键简化）

paper证明两个loss的梯度相等：
$$\nabla_\theta \mathcal{L}_{EFM}(\theta) = \nabla_\theta \mathcal{L}_{CEFM}(\theta)$$

**EFM loss** (marginal):
$$\mathcal{L}_{EFM}(\theta) = \mathbb{E}_{\mathbf{x},t}\left[\frac{\exp(-\mathcal{E}_t(\mathbf{x}))}{\mathbb{E}_{\tilde{\mathbf{x}} \sim p_t}[\exp(-\mathcal{E}_t(\tilde{\mathbf{x}}))]} \|\mathbf{v}_\theta(\mathbf{x}) - \hat{\mathbf{u}}_t(\mathbf{x})\|^2\right]$$

**CEFM loss** (conditional, 可计算):
$$\mathcal{L}_{CEFM}(\theta) = \mathbb{E}_{\mathbf{x}_0,\mathbf{x},t}\left[\frac{\exp(-\beta \mathcal{E}(\mathbf{x}_0))}{\mathbb{E}_{\tilde{\mathbf{x}}_0 \sim p_0}[\exp(-\beta \mathcal{E}(\tilde{\mathbf{x}}_0))]} \|\mathbf{v}_\theta(\mathbf{x}) - \mathbf{u}_{t0}(\mathbf{x}|\mathbf{x}_0)\|^2\right]$$

**Intuition**: 这个等价性让我们绕开了不可计算的 $\mathcal{E}_t$，直接用conditional vector field + energy weight来训练。这是整个方法可行的数学基础。

---

## 3. RL Advantage Signal: Leave-One-Out

paper采用 **REINFORCE Leave-One-Out (RLOO)** 来估计advantage：

$$R^*(c, x_k) = \frac{K}{K-1}\left(R(c, x_k) - \frac{1}{K}\sum_{i=1}^K R(c, x_i)\right)$$

- $c$: context (observation $o_t$)
- $x_k$: 第 $k$ 个采样 (action chunk $A_t$)
- $R(c, x_k)$: return (这里用return-to-go)
- $K$: 采样数量
- $K/(K-1)$: 无偏性校正系数

**Intuition**: 用其他 $K-1$ 个样本的均值作为baseline，这是无偏且低方差的advantage估计。在VLA多任务设置下，paper按task type做standardization，让不同任务的advantage可比较。

**Web reference**: 
- RLOO原始paper: https://arxiv.org/abs/1905.10606
- Decision Transformer (return-to-go思想来源): https://arxiv.org/abs/2106.01345

---

## 4. π₀ VLA Flow Model架构

### 4.1 输入输出结构

观察输入：
$$\mathbf{o}_t = [I_1^t, \ldots, I_n^t, \ell^t, q^t]$$

- $I_i^t$: 第 $i$ 个RGB图像（multi-view）
- $\ell^t$: language token序列
- $q^t$: 关节角度向量 (proprioception)

Action chunk:
$$\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+H-1}]$$

- $H$: action horizon (paper中为50，见Table 7)
- 每个 $\mathbf{a}_t$ 是7维（关节角度）

### 4.2 Flow Matching Loss

$$L_{FM}(\theta) = \mathbb{E}_{p(\mathbf{A}_t|\mathbf{o}_t), q(\mathbf{A}_\tau^t|\mathbf{A}_t)}\left[\|\mathbf{v}_\theta(\mathbf{A}_\tau^t, \mathbf{o}_t) - \mathbf{u}(\mathbf{A}_\tau^t|\mathbf{A}_t)\|^2\right]$$

- $\mathbf{v}_\theta$: 神经网络预测的vector field
- $\mathbf{u}(\mathbf{A}_\tau^t|\mathbf{A}_t)$: ground truth conditional vector field
- $\mathbf{A}_\tau^t = \tau \mathbf{A}_t + (1-\tau)\boldsymbol{\epsilon}$: Optimal Transport加噪
- $\tau \in [0,1]$: flow时间
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: 标准Gaussian噪声

**Conditional vector field** (Optimal Transport):
$$\mathbf{u}(\mathbf{A}_\tau^t|\mathbf{A}_t) = \boldsymbol{\epsilon} - \mathbf{A}_t$$

### 4.3 推理过程

从 $\tau=0$ 积分到 $\tau=1$：
$$\mathbf{A}_{\tau+\delta}^t = \mathbf{A}_\tau^t + \delta \mathbf{v}_\theta(\mathbf{A}_\tau^t, \mathbf{o}_t)$$

- $\delta$: 积分步长
- 起始点 $\mathbf{A}_0^t \sim \mathcal{N}(0, \mathbf{I})$

---

## 5. ARFM核心方法：自适应α的推导

### 5.1 Energy-Weighted VLA Flow Loss

目标分布：
$$\pi(\mathbf{A}_t|\mathbf{o}_t) \propto p(\mathbf{A}_t|\mathbf{o}_t) \exp(\alpha R^*(\mathbf{o}_t, \mathbf{A}_t))$$

注意符号：这里 $\alpha$ 对应 EWFM 中的 $-\beta$（因为advantage越大越好，energy越小越好）。

practical loss（batch内实现）：
$$L_1^\tau(\theta) = \sum_{i=1}^B w_i(\alpha) \|\mathbf{v}_\theta(\{\mathbf{A}_t^i\}^\tau, \mathbf{o}_t) - \mathbf{u}(\{\mathbf{A}_t^i\}^\tau|\mathbf{A}_t^i)\|^2$$

$$w_i(\alpha) = \frac{\exp(\alpha R^*(\mathbf{A}_t^i, \mathbf{o}_t))}{\sum_{j=1}^B \exp(\alpha R^*(\mathbf{A}_t^j, \mathbf{o}_t))}$$

- $B$: batch size (paper中为16)
- $w_i(\alpha)$: softmax-normalized energy weight
- $R^*$: standardized RL advantage

### 5.2 优化目标：Bias-Variance Tradeoff

$$J(\alpha) = \text{Var}(\hat{g}(\alpha)) - \lambda S(\alpha)$$

- $\hat{g}(\alpha) = \nabla_\theta L_1^\tau(\theta)$: 损失梯度
- $\text{Var}(\hat{g}(\alpha))$: 梯度方差（要最小化）
- $S(\alpha) = \sum_i \hat{w}_i(\alpha) R^*(\mathbf{A}_t^i, \mathbf{o}_t) / \sum_i \hat{w}_i(\alpha)$: RL signal score（要最大化）
- $\lambda$: balance hyperparameter (paper中为 5.0e-4)

**Intuition**: 
- 第一项防止gradient explosion（α太大时softmax会one-hot化，方差爆炸）
- 第二项保证RL信号被保留（α太小时所有weight均匀，信号被稀释）

### 5.3 三个关键假设

**Assumption 1**: $R^*(\mathbf{A}_t, \mathbf{o}_t) \sim \mathcal{N}(0, \sigma_R^2)$
- 标准化后的advantage近似零均值Gaussian

**Assumption 2**: $L_{CFM}^i \sim \mathcal{N}(\mu_L, \sigma_L^2)$
- flow matching loss值近似Gaussian
- paper论证：post-training阶段loss值会快速收敛到低方差状态

**Assumption 3**: 大batch size下，样本统计量近似真实统计量

### 5.4 Corollary 1推导详解

定义矩生成函数：
$$m_1(\alpha) = \mathbb{E}[\exp(\alpha R^*)]$$
$$m_2(\alpha) = \mathbb{E}[\exp(2\alpha R^*)]$$

利用Gaussian的MGF公式 $\mathbb{E}[\exp(tX)] = \exp(t\mu + \frac{1}{2}t^2\sigma^2)$，当 $\mu=0$：

$$m_1(\alpha) = \exp\left(\frac{1}{2}\sigma_R^2 \alpha^2\right)$$
$$m_2(\alpha) = \exp\left(2\sigma_R^2 \alpha^2\right)$$

经过推导（详见paper Appendix），得到：

$$J(\alpha) = \sigma_L^2 \left[e^{2\alpha^2 \sigma_R^2} - e^{\alpha^2 \sigma_R^2}\right] - \lambda \alpha \sigma_R^2$$

**各项含义**：
- $\sigma_L^2 e^{2\alpha^2 \sigma_R^2}$: 来自 $m_2(\alpha)$，是二阶矩贡献
- $-\sigma_L^2 e^{\alpha^2 \sigma_R^2}$: 来自 $m_1(\alpha)^2$，是一阶矩平方贡献
- $-\lambda \alpha \sigma_R^2$: RL signal项，线性增长

### 5.5 Corollary 2：求解α*

对 $J(\alpha)$ 求导并令其为0：

$$J'(\alpha) = \sigma_L^2 \left[4\alpha \sigma_R^2 e^{2\alpha^2 \sigma_R^2} - 2\alpha \sigma_R^2 e^{\alpha^2 \sigma_R^2}\right] - \lambda \sigma_R^2 = 0$$

化简：
$$4\alpha^2 e^{2\alpha^2 \sigma_R^2} - 2\alpha^2 e^{\alpha^2 \sigma_R^2} - \frac{\lambda}{\sigma_L^2} = 0$$

令 $x = \alpha^2 \sigma_R^2$：
$$4\sqrt{x} e^{2x} - 2\sqrt{x} e^x - \frac{\lambda \sigma_R}{\sigma_L^2} = 0$$

$$\alpha^* = \frac{\sqrt{x^*}}{\sigma_R}$$

**关键insight**: 
- 最优 $\alpha^*$ 与 $\sigma_R$ 成反比 → advantage方差大时，α自动减小
- 最优 $\alpha^*$ 与 $\sigma_L$ 相关 → flow loss方差大时，α也调整
- 这个方程没有闭式解，但有单调性，可用 **bisection iteration** 高效求解

### 5.6 Algorithm 1: Bisection Iteration

```
Input: R_i*, L_FM^i, B, λ, [α_min, α_max], M, ε
1: 计算 σ_R², μ_L, σ_L², x_low = σ_A²·α_min, x_high = σ_A²·α_max
2: 定义 F(x) = 4√x·e^(2x) - 2√x·e^x - λ·σ_R/σ_L²
3-8: bisection循环M次
9: α* = √(0.5(x_low + x_high)) / σ_A
10: return clip(α*, α_max, α_min)
```

- $M$: iteration数 (paper中20)
- $\epsilon$: tolerance (1.0e-5)
- 整个过程计算量极小，每batch只增加常数开销

---

## 6. 完整ARFM算法流程

**Algorithm 2** 的核心步骤：

```
for each batch {A_t^i, o_t^i}:
    for i in [B]:
        1. 采样 ε_i ~ N(0,I), τ ~ Uniform(0,1)
        2. {A_t^i}^τ = τ·A_t^i + (1-τ)·ε_i  (Optimal Transport加噪)
        3. 计算 R^i = R*(A_t^i, o_t^i)  (RLOO advantage)
        4. 计算 L_FM^i = ||v_θ({A_t^i}^τ, o_t) - (ε_i - A_t^i)||²
    5. 用Algorithm 1求最优α*
    6. 计算 w_i(α*) = exp(α*·R^i) / Σ_j exp(α*·R^j)
    7. L_1^τ(θ) = Σ_i w_i(α*)·L_FM^i
    8. 梯度下降一步
```

---

## 7. 实验数据深度分析

### 7.1 Multi-task Learning (Table 1)

| Model Type | Models | Goal | Spatial | Object | Long | Average |
|---|---|---|---|---|---|---|
| Non-Flow | Octo | 84.6 | 78.9 | 85.7 | 51.1 | 75.1 |
| Non-Flow | OpenVLA | 79.2 | 84.7 | 88.4 | 53.7 | 76.5 |
| Non-Flow | Dita | 85.4 | 84.2 | 96.3 | 63.8 | 82.4 |
| Non-Flow | QueST | 80.8 | 87.4 | 93.6 | 68.8 | 82.7 |
| Flow | π₀ | 93.8 | 91.2 | 93.2 | 74.2 | 88.1 |
| Flow | ReinboT | 94.0 | 95.6 | 93.8 | 81.4 | 91.2 (+3.5%) |
| Flow | RWR | 94.4 | 94.0 | 94.3 | 80.4 | 90.8 (+3.1%) |
| Flow | **ARFM** | **94.9** | **95.8** | **95.0** | **82.6** | **92.1 (+4.5%)** |

**Key observations**:
1. Flow matching类型整体 > non-flow，验证trajectory-level建模优势
2. ARFM在所有4个suite都达到SOTA
3. **Long suite提升最显著** (74.2→82.6, +8.4%)，因为Long任务需要long-horizon规划，RL advantage signal帮助最大

### 7.2 Action Perturbation Robustness (Table 2)

加入Gaussian noise (0.1, 0.15, 0.2, 0.25, 0.3):

| Models | Goal | Spatial | Object | Long | Avg. |
|---|---|---|---|---|---|
| π₀ | 47.5 | 50.6 | 44.9 | 30.0 | 43.3 |
| ReinboT | 51.4 | 59.6 | 44.8 | 29.3 | 46.3 (+6.9%) |
| RWR | 49.5 | 60.1 | 46.9 | 29.1 | 46.4 (+7.2%) |
| **ARFM** | **49.7** | **61.1** | **48.9** | **33.0** | **48.2 (+11.4%)** |

**Intuition**: ARFM通过energy weighting学到的policy更"集中"在高return区域，所以对noise更鲁棒。这印证了bias-variance tradeoff的设计目的——稳定的gradient带来稳定的policy。

### 7.3 Few-shot Learning (Table 3, LIBERO-Long)

| Models | 30-shot | 20-shot | 10-shot | Avg. |
|---|---|---|---|---|
| π₀ | 41.7 | 33.8 | 22.1 | 32.5 |
| ReinboT | 39.5 | 37.5 | 24.6 | 33.9 (+4.1%) |
| RWR | 39.5 | 37.7 | 26.7 | 34.6 (+6.5%) |
| **ARFM** | **42.9** | **38.9** | **27.7** | **36.5 (+12.2%)** |

**Intuition**: 少样本下数据效率至关重要。ARFM的adaptive α能根据当前batch的advantage分布动态调整，最大化利用有限数据。

### 7.4 Continual Learning (Table 4)

训练序列: Long(30) → Long(15)+Goal(15) → Long(2)+Goal(2)+Object(2)

| Models | Avg. NBT ↓ | Avg. SR ↑ |
|---|---|---|
| π₀ | 7.5 | 55.2 |
| ReinboT | 6.6 (-12.0%) | 55.9 (+1.2%) |
| RWR | 7.3 (-2.3%) | 55.3 (+0.2%) |
| **ARFM** | **4.7 (-38.0%)** | **61.0 (+10.5%)** |

**NBT (Negative Backward Transfer)**: 
$$\text{NBT} = \frac{1}{T-1}\sum_i^{T-1} \max(0, (SR)_i - (SR)_i^T)$$

- $(SR)_i$: 学完task $i$ 后的success rate
- $(SR)_i^T$: 学完所有task后task $i$ 的success rate
- NBT越小越好，0表示无遗忘

**Intuition**: ARFM的NBT降低38%，说明energy weighting帮助模型保留高return的旧task知识。RL advantage相当于给数据打了"重要性标签"，新task训练时不会过度覆盖高质量旧知识。

### 7.5 Ablation Study (Figure 4)

**λ (balance hyperparameter)**: 性能不敏感，说明ARFM的自适应能力robust
**M (bisection iterations)**: $M \geq 10$ 即稳定，说明bisection收敛快

---

## 8. Dense Reward Design (Table 8)

paper用了13个reward component，分4类：

**Sub-goal Achievement** (7项):
- Image MSE / SSIM / ORB similarity
- Gripper Image MSE / SSIM / ORB
- Joint Position MSE

**Task Progress**: Sub-goal Division $n(s_t)/|\{s^*\}|$

**Behavior Smoothness**:
- Joint Velocity: $-|\dot{\mathbf{q}}|^2$
- Joint Acceleration: $-|\ddot{\mathbf{q}}|^2$
- Action Velocity: $-|\mathbf{a}_{t-1} - \mathbf{a}_t|^2$
- Action Acceleration: $-|\mathbf{a}_{t-2} - 2\mathbf{a}_{t-1} + \mathbf{a}_t|^2$

**Task Completion**: 0/1 success indicator

**Intuition**: 多维reward让advantage信号更丰富，能区分"接近成功但失败"和"完全没接近"的轨迹，这对flow model学习细微的distribution shift很有帮助。

---

## 9. 与ReinboT的关键差异

paper在Introduction提到ReinboT在flow model上效果有限，原因是：

**ReinboT方法**: 用return-to-go作为generation condition
- 模型输入: $[o_t, RTG_{target}]$
- 生成: 通过conditioned vector field生成action
- 问题: RTG只能间接控制trajectory vector field的生成方向，无法直接塑造action distribution

**ARFM方法**: 用advantage作为energy weight
- 直接在loss层面reweight样本
- 数学上等价于学习 $q_0 \propto p_0 \cdot \exp(\alpha R^*)$
- 直接改变target distribution

**Intuition**: ReinboT是"条件生成"，ARFM是"分布重塑"。对于flow model这种density-based方法，后者更自然。

---

## 10. Hyperparameter配置 (Table 7)

| Parameter | Value |
|---|---|
| Post-Training Steps | 4e4 (LIBERO) / 6e4 (UR5) |
| Batch Size | 16 |
| Action Horizon | 50 |
| λ | 5.0e-4 |
| M | 20 |
| α range | [0.01, 5] |
| ε (tolerance) | 1.0e-5 |
| LR (AdamW) | 1.0e-4 |
| Betas | (0.9, 0.95) |
| Gradient Clip | 10 |
| Warmup Steps | 1e3 |
| Peak LR | 2.5e-5 |
| Decay LR | 2.5e-6 |

**Training cost**: 2×A100 GPU, 11小时(LIBERO) / 16小时(UR5)

---

## 11. 直觉总结与批判性思考

### 11.1 为什么ARFM有效？

**从distribution角度**: 
- Vanilla flow matching学 $p(\mathbf{A}_t|\mathbf{o}_t)$，是behavior cloning
- ARFM学 $q(\mathbf{A}_t|\mathbf{o}_t) \propto p \cdot \exp(\alpha R^*)$，是energy-reshaped distribution
- 这相当于在action空间做importance sampling，但通过flow matching平滑地学习

**从optimization角度**:
- 固定α相当于固定temperature的Boltzmann distribution
- Adaptive α相当于自适应temperature，避免训练早期gradient爆炸、训练后期信号不足

**从RL角度**:
- 这是critic-free的offline RL，不需要训练value function
- RLOO advantage已经是unbiased estimator
- Energy weighting比直接的policy gradient更稳定

### 11.2 局限性与未来方向

1. **Gaussian假设的局限**: Assumption 1, 2假设advantage和loss是Gaussian，实际中可能不是。特别是多模态reward分布下，σ_R会很大，α会被压得很小，可能丢失信号。

2. **Batch size依赖**: Assumption 3要求大batch，paper用B=16其实不算大，可能影响统计量估计精度。

3. **Reward design**: 13个reward component需要手工设计权重，这本身是个open problem。

4. **只针对flow model**: 方法依赖Theorem 2的梯度等价性，对autoregressive VLA不直接适用。

5. **Online RL extension**: paper在Conclusion提到online RL post-training是future work，这需要处理exploration和distribution shift问题。

### 11.3 与LLM RLHF的类比

ARFM的思路与LLM的RLHF/DPO有有趣的类比：
- **RLHF (PPO)**: 训练reward model + policy optimization，复杂
- **DPO**: 直接用preference对重构reward，简化
- **ARFM**: 直接用advantage作为energy weight，类似DPO的思路，但用flow matching而非logistic regression

**Web reference**:
- DPO: https://arxiv.org/abs/2305.18290
- RLHF survey: https://arxiv.org/abs/2307.15217

### 11.4 与Diffusion RL的关系

paper的Related Work提到diffusion-based offline RL：
- **Diffusion-QL** (Wang et al. 2022): 在diffusion上做Q-learning
- **CEM energy-guided diffusion** (Lu et al. 2023): 用energy引导diffusion sampling
- **ARFM**: energy直接进入flow loss，不需要单独的energy model

ARFM的优势在于：energy直接作用于训练loss，而非inference时的guidance，所以训练完inference无额外开销。

**Web reference**:
- Diffusion-QL: https://arxiv.org/abs/2204.13649
- Energy-guided diffusion: https://arxiv.org/abs/2304.12824

---

## 12. 实现细节联想

基于paper的Code Appendix，ARFM实现基于 **huggingface/lerobot**:
- 仓库: https://github.com/huggingface/lerobot
- 基础模型: π₀ 的Gemma Expert backbone
- ReinboT复现: 在Gemma Expert上拼接learnable tokens学return-to-go，用gated fusion融合

**关键实现trick**:
1. Advantage standardization按task type分组，避免跨任务比较偏差
2. α的clip范围 [0.01, 5] 防止数值异常
3. Image augmentation (brightness/contrast/saturation/hue/sharpness) 提升鲁棒性
4. Cosine decay with warmup的LR schedule

---

## 13. 总结

ARFM的核心贡献是把 **adaptive temperature scaling** 引入VLA flow model的offline RL post-training，通过：
1. 构造bias-variance tradeoff objective $J(\alpha) = \text{Var}(\hat{g}) - \lambda S$
2. 在Gaussian假设下推导出closed-form近似
3. 用bisection iteration实时求解

方法优雅，理论扎实，实验全面（multi-task/perturbation/few-shot/continual/real-world）。对于build intuition，核心记住：**flow matching的density learning特性让energy weighting天然适用，而adaptive α解决了RL signal强度与training stability的根本矛盾**。

**Future work方向**:
1. Online RL post-training (paper已提及)
2. 多模态reward (去掉Gaussian假设)
3. Autoregressive VLA的类比方法
4. Hierarchical advantage (trajectory-level + step-level)
5. 与world model结合的model-based RL post-training
