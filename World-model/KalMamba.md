---
source_pdf: KalMamba.pdf
paper_sha256: 400eb7adf22988658d8e5d630123179568f274fec701cfdf48ebb39bc9ce2630
processed_at: '2026-08-05T11:04:20-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# KalMamba 用人话说

## 这 paper 到底在搞啥

想象你在玩一个游戏，屏幕画面很糊，有时候还会黑屏几帧。你要在这种垃圾视野下做决策。

传统方法两条路：
- **Probabilistic SSM**（像 RSSM, VRKN）：能处理 uncertainty，知道"我现在不太确定自己在哪"，但计算慢，像老牛拉破车
- **Deterministic SSM**（像 Mamba）：快得飞起，能并行处理整条序列，但不知道自己不确定，像个过度自信的傻子

KalMamba 的 idea 就是：**让 Mamba 当前锋探路，让 Kalman Filter 当军师算概率**。前锋跑得快，军师算得准，两人配合。

---

## 核心 trick 是什么

这里有个 math trick 非常 elegant。

Kalman Filter 你可以看成一条 chain：每个 timestep 产生一个 "filtering element"，包含 transition matrix、observation、noise 这些东西。关键是，**这些 element 可以像数字加法一样 combine**，满足 associative law。

啥意思？就像 $(a + b) + c = a + (b + c)$。一旦你有 associativity，就能用 parallel scan：把 sequence 拆成两半，各自算，再 merge，logarithmic time 搞定。

[这篇 paper](https://arxiv.org/abs/1905.13002) 早就证明了 Kalman smoother 也能这么搞。但一直没人真的把它塞进 deep learning 框架里用。KalMamba 就是干这个的。

---

## 为什么之前没人这么做

因为有个 chicken-and-egg 问题。

Kalman Filter 需要 dynamics parameters $(A_t, b_t, \Sigma_t)$。在经典控制里这些是固定的。在 deep learning 里你想让 neural network 来学这些参数。但问题来了：

- 如果参数依赖 current observation，那还行，可以并行
- 但如果参数依赖 **past filtered belief**（像 VRKN 那样，在 filtered mean 处 linearize），就形成 circular dependency：算 parameter 要先算 filter，算 filter 要先有 parameter。这只能 sequential 跑

VRKN 就卡在这。它用 Kalman filter + smoother 拿到了 tight ELBO 和好的 uncertainty estimate，但 dynamics parameter 依赖 filtered mean $\mu_t^+$，整个东西变成 autoregressive，没法并行。

KalMamba 的解法：**让 Mamba 来预测 dynamics parameter**，而不是依赖 filtered mean。Mamba 本身就是 parallel 的 sequence model，它看完所有历史 observation 和 action，直接 output $(A_t, b_t, \Sigma_t)$。

这样 dependency chain 变成：
1. Mamba 吃 $(o_{\le t}, a_{\le t})$ → 吐出 $(A_t, b_t, \Sigma_t)$ （parallel）
2. Kalman filter 用这些参数算 belief （parallel scan）
3. Kalman smoother 再来一遍 backward pass （parallel scan）

全链路 parallelizable。

---

## 那 Mamba 凭什么能预测 dynamics parameter

这里有个 subtle 的设计。

Mamba 看完历史，输出一个中间表示 $\mathbf{m}_t$。这个 $\mathbf{m}_t$ 理论上应该包含和 filtered belief $q(z_t | o_{\le t}, a_{\le t-1})$ 一样的信息。然后 $\mathbf{m}_t$ 和 current action $a_t$ 一起，喂给一个小 NN，输出 dynamics parameters。

但问题：如果没有约束，Mamba 可能偷懒，直接把 observation 信息编码进 $A_t, b_t$，绕过 latent state $z_t$。这样 Kalman filter 就变成摆设了。

解法是 Mahalanobis regularizer：

$$R = \sum_t (\mathbf{m}_t - \mu_t^+)^T (\Sigma_t^+)^{-1} (\mathbf{m}_t - \mu_t^+)$$

强制 $\mathbf{m}_t$ 和 filtered mean $\mu_t^+$ 对齐。但不是硬对齐，是用 inverse covariance 加权。Covariance 大的维度允许偏离，covariance 小的维度严格匹配。这很 reasonable：不确定的维度你随便飘，确定的维度你得对上。

这个 regularizer mirror 了经典 EKF 的做法：EKF 在 filtered mean 处 linearize。KalMamba 也 incentivize Mamba 输出和 filtered mean 对齐的 representation。

---

## 为什么 training 用 smoothing，acting 用 filtering

这是另一个 clever 的设计。

**Smoothing**（条件于 future observation）给你更好的 posterior estimate，ELBO tight，uncertainty estimate 准。但 acting 时你哪来 future observation？所以 acting 只能用 filtering。

问题：如果你 train 时 optimize smoothed belief，但 act 时用 filtered belief，两者可能有 gap。

KalMamba 的 trick：**smoother 不引入任何新 parameter**。Smoother 只是 Kalman filter 的 backward pass，closed-form computation，没有 trainable weight。

这意味着 smoothing objective 的 gradient 会自动 backpropagate 通过 filter 的参数，间接拉高 filtered belief 的质量。你 optimize smoothing，filtering 跟着变好。Free lunch。

---

## Diagonal matrix 的取舍

所有 matrix（$A_t$, $\Sigma_t^{\text{dyn}}$, $\Sigma_t^{\text{w}}$）都是 diagonal。这牺牲了 expressiveness：你没法建模 cross-dimension correlation。

但换来巨大效率提升：
- Matrix multiply 变 element-wise multiply
- Matrix inverse 变 element-wise inverse
- $O(n^3)$ 变 $O(n)$

在 RL 的 latent dimension（30-256）规模下，这个 trade-off 划算。而且 Mamba 的 input-dependent parameterization 能部分补偿 expressiveness loss。

$A_t$ 的 eigenvalue 还被 constrain 在 [0.4, 0.99]，保证 dynamics 稳定（不会 blow up 也不会衰减太快）。

---

## Monte-Carlo Dropout 为什么重要

Smoothing inference 用 future information 推 past state。如果 model 对未来 trajectory 的 epistemic uncertainty 没建模，smoother 会过度自信，觉得"我看到了 future，所以 past 一定就是这样"。

[MC Dropout](https://arxiv.org/abs/1506.02142) 在 Mamba backbone 里加 dropout，每次 forward pass 随机 drop 一些 neuron，相当于 approximate Bayesian inference。这让 smoother 意识到"我对 future 的 prediction 也不确定"，从而保持合理的 uncertainty estimate。

Ablation study 显示，去掉 MC Dropout 在最难的 quadruped_walk 任务上掉分最多。说明 epistemic uncertainty modeling 在复杂任务中确实关键。

---

## 实验结果说了啥

### Image-based DMC tasks

KalMamba 和 RSSM+SAC、VRKN+SAC 差不多，略差一点点但 competitive。所有 SSM+SAC 组合都 match 或 beat DreamerV3。

这说明啥？**用 SAC 直接在 belief 上训 Q-function，比 latent imagination rollout 更高效**。DreamerV3 那套在 latent space 做 rollout 的方法，在 DMC 这种 task 上没有优势。

### Noisy state-based tasks

加了 observation noise $\mathcal{N}(0, 0.3)$ 和 dynamics noise $\mathcal{N}(0, 0.3)$。

- Naive SAC 直接崩
- RSSM 有改善但不够好
- VRKN 和 KalMamba 都不错

这验证了 smoothing inference 的价值。RSSM 的 filtering-only inference 在这个 noise level 下不够 robust。

### Runtime scaling - 最亮的结果

Sequence length 从 32 涨到 256：
- RSSM、VRKN：时间线性增长，VRKN 在 256 时直接超时（24h 只跑 650k steps）
- KalMamba：时间几乎不变，logarithmic scaling

更 amazing 的是 **KalMamba 在长序列上性能反而更好**。seq=256 比 seq=32 表现更好。这说明 Mamba backbone 确实在利用 long-term context，而 baseline SSMs 在长序列上 dynamics model 会 degrade。

---

## 直觉总结

KalMamba 的核心 insight 其实很简单：**把 probabilistic inference 拆成两部分，parameter prediction 用 Mamba（快），belief inference 用 Kalman（准），两者用 linear Gaussian SSM 作为 interface**。

就像一家公司：
- Mamba 是销售，跑得快，收集市场信息（observation history），转化成 dynamics parameters
- Kalman 是财务，算得准，拿着这些 parameters 做 probabilistic inference
- Linear Gaussian SSM 是两者之间的合同，规定好信息怎么传递

合同设计得好的话，两个部门各自发挥优势，公司运转高效。

之前的方法要么销售太慢（VRKN 的 sequential linearization），要么财务不专业（RSSM 的 loose ELBO），要么只有一个人干活（Mamba 自己不知道 uncertainty）。

---

## 这 paper 的意义

这 paper 给了一个 template：**如何把 classical probabilistic inference (Kalman) 塞进 modern efficient architecture (Mamba) 里**。

这个 template 可以 extend：
- Complex-valued latent state（paper 提到的 future work）
- Block-diagonal matrices（trade off expressiveness 和 efficiency）
- 更复杂的 inference（particle filter 的 parallel formulation？）
- 其他 domain：time series forecasting, video prediction, robotics...

核心就是找到 classical algorithm 的 associative formulation，然后用 parallel scan 加速。这个 idea 不局限于 Kalman filter。

[Parallel scan primitives for GPU](https://graphics.cs.uni-saarland.de/?page_id=890) 是 GPU computing 的经典技术，现在 deep learning 越来越多用到它。Mamba 用它，S5 用它，KalMamba 也用它。未来会有更多 algorithm 被重新审视："hey, 这个能 parallel scan 吗？"

---

## 一句话版本

Mamba 负责快，Kalman 负责准，linear Gaussian SSM 当桥梁，parallel scan 让两者都飞起来。

---

# KalMamba 深度解析

这篇 paper 解决的问题非常 precise：**如何同时获得 probabilistic SSM 的 uncertainty modeling 能力和 deterministic SSM (Mamba) 的 time-parallelizable 效率**。下面我一层层 build up intuition。

---

## 1. 核心问题动机

Reinforcement Learning 中处理 partial observability 和 noise 需要 probabilistic latent state space models。现有方法存在一个 fundamental trade-off：

| 方法 | Inference | Smoothing | Parallelizable |
|------|-----------|-----------|----------------|
| RSSM (Hafner 2019) | $q(\mathbf{z}_t \| \mathbf{h}_t, \mathbf{o}_t)$ | ✗ | ✗ |
| R2I (Samsami 2024) | $q(\mathbf{z}_t \| \mathbf{o}_t)$ | ✗ | ✓ |
| VRKN (Becker 2022) | $q(\mathbf{z}_t \| \mathbf{o}_{\le T}, \mathbf{a}_{\le T})$ | ✓ | ✗ |
| **KalMamba** | $q(\mathbf{z}_t \| \mathbf{o}_{\le T}, \mathbf{a}_{\le T})$ | ✓ | ✓ |

关键张力在于：
- **Smoothing inference**（条件于未来观测）给出 tight variational lower bound，能正确估计 uncertainty
- 但 smoothing 通常要求 sequential forward-backward pass，无法时间并行
- **Filtering-only** 方法（如 RSSM, R2I）可以并行化，但 lower bound 松散，uncertainty 估计差

KalMamba 的核心 insight：**Kalman filtering 和 smoothing 可以表述为 associative operations**（[Särkkä & García-Fernández 2020](https://arxiv.org/abs/1905.13002)），因此可以用 parallel associative scan 实现 logarithmic time complexity。

---

## 2. Background: SSM 家族谱系

### 2.1 Deterministic SSMs (S4 → S5 → Mamba)

这类 model 基于 continuous-time linear ODE 的 discretization：

$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)$$

通过 zero-order hold 离散化得到：

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$

其中 $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \Delta B$。

- **S4** ([Gu et al. 2021](https://arxiv.org/abs/2111.00396))：使用 structured parameterization（HIPPO matrix）+ convolutional formulation
- **S5** ([Smith et al. 2022](https://arxiv.org/abs/2208.04933))：用 parallel associative scan 替代卷积
- **Mamba** ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752))：**selective** SSM，参数 $B_t, C_t, \Delta_t$ 依赖输入，加 input-dependent gating

Mamba 的关键创新是 **selection mechanism**：让 model 能够 selectively propagate 或 forget information，类似 transformer 的 attention 但保持 linear time complexity。

### 2.2 Probabilistic SSMs for RL

**RSSM** (Recurrent State Space Model, [Hafner et al. 2019](https://arxiv.org/abs/1911.01670))：
- 拆分 latent state $\mathbf{z}_t = (\mathbf{s}_t, \mathbf{h}_t)$
- Deterministic path: $\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{a}_{t-1}, \mathbf{s}_{t-1})$，用 GRU 实现
- Stochastic path: $\mathbf{s}_t \sim p(\mathbf{s}_t | \mathbf{h}_t)$
- Inference: $q(\mathbf{s}_t | \mathbf{h}_t, \mathbf{o}_t)$，filtering only

**问题**：GRU 是 autoregressive，无法并行；filtering 的 ELBO 松散。

**VRKN** (Variational Recurrent Kalman Network, [Becker & Neumann 2022](https://arxiv.org/abs/2202.10650))：
- 在 latent space 用 linear Gaussian SSM
- Kalman filter + smoother 实现 closed-form inference
- Tight ELBO，uncertainty 估计好
- **问题**：dynamics 参数依赖于 filtered mean $\mu_t^+$，形成 nonlinear autoregressive dependency → 无法并行

**R2I** ([Samsami et al. 2024](https://arxiv.org/abs/2310.04407))：
- 用 S4 替代 GRU
- 为了并行化，简化 inference 为 $q(\mathbf{z}_t | \mathbf{o}_t)$，丢弃所有历史信息
- 在 noisy/partial observation 下表现差

---

## 3. KalMamba 架构详解

### 3.1 整体数据流

```
Observation o_t → Encoder φ → (w_t, Σ_t^w)
                                      ↓
(w_t, a_{t-1}) → NN_pre → Mamba → m_t → (m_t, a_t) → NN_post → (A_t, b_t, Σ_t^dyn)
                                                                      ↓
                                              Parallel Kalman Filter → q(z_t | o_≤t, a_≤t-1)
                                                                      ↓
                                              Parallel Kalman Smoother → q(z_t | o_≤T, a_≤T)
                                                                      ↓
                                                          Decoder → p(o_t | z_t)
                                                          Reward head → p(r_t | z_t)
                                                                      ↓
                                                              SAC policy
```

### 3.2 Observation Model 设计

引入 auxiliary observation $\mathbf{w}_t$：

$$q(\mathbf{w}_t | \mathbf{z}_t) = \mathcal{N}(\mathbf{w}_t | \mathbf{z}_t, \Sigma_t^{\mathbf{w}})$$

通过 encoder $\phi(\mathbf{o}_t)$ 提取 $(\mathbf{w}_t, \Sigma_t^{\mathbf{w}})$。

**为什么这样设计？** 将复杂的 $\mathbf{o}_t \to \mathbf{z}_t$ 关系分解为：
1. Encoder 学习 $\mathbf{o}_t \to \mathbf{w}_t$（高维到低维的 nonlinear compression）
2. Latent space 中用 simple Gaussian observation model

这样做 vs 直接在 latent space 做 observation model（如 [Fraccaro et al. 2017](https://arxiv.org/abs/1710.05841)）的优势：
- $\mathbf{w}_t$ 当作 observable，减少 latent variables 数量
- Encoder 的 observation uncertainty $\Sigma_t^{\mathbf{w}}$ 可以直接 propagate 到 state belief
- Inference 更简单且可并行

### 3.3 Dynamics Model

$$p(\mathbf{z}_{t+1} | \mathbf{z}_t, \mathbf{a}_t) = \mathcal{N}\left(\mathbf{z}_{t+1} \mid \mathbf{A}_t(\mathbf{o}_{\le t}, \mathbf{a}_{\le t}) \mathbf{z}_t + \mathbf{b}_t(\mathbf{o}_{\le t}, \mathbf{a}_{\le t}), \Sigma_t^{\text{dyn}}(\mathbf{o}_{\le t}, \mathbf{a}_{\le t})\right) \tag{1}$$

关键设计决策：
- **$\mathbf{A}_t$ 和 $\Sigma_t^{\text{dyn}}$ 都是 diagonal matrices**：这让 Kalman filter/smoother 中的矩阵运算变成 point-wise 操作，计算复杂度从 $O(n^3)$ 降到 $O(n)$
- **$\mathbf{A}_t$ 的特征值约束在 [0.4, 0.99]**：保证 dynamics 稳定（eigenvalue < 1 是 stability 的必要条件，下界 0.4 防止信息衰减太快）
- **参数依赖所有历史 $(\mathbf{o}_{\le t}, \mathbf{a}_{\le t})$**：通过 Mamba backbone 实现，而非仅依赖当前 $(\mathbf{o}_t, \mathbf{a}_t)$

**重要 subtlety**：dynamics 在 $\mathbf{z}_t$ 上是线性的，但参数 $\mathbf{A}_t, \mathbf{b}_t$ 通过神经网络 nonlinearly 依赖于历史观测。这相当于在每一步做 **input-dependent linearization**，类似 extended Kalman filter 但 linearization point 由 Mamba 选择。

### 3.4 Mamba Backbone 的信息流

参考 Figure 2 的架构：

1. **Input fusion**: $\text{concat}(\mathbf{w}_t, \mathbf{a}_{t-1}) \to \text{NN}_{\text{pre}}$
2. **Temporal aggregation**: Mamba blocks 处理整个序列，输出 $\mathbf{m}_t = \text{Mamba}(\mathbf{o}_{\le t}, \mathbf{a}_{\le t-1})$
3. **Dynamics parameterization**: $\text{concat}(\mathbf{m}_t, \mathbf{a}_t) \to \text{NN}_{\text{post}} \to (\mathbf{A}_t, \mathbf{b}_t, \Sigma_t^{\text{dyn}})$

**$\mathbf{m}_t$ 的作用**：
- 作为信息瓶颈，积累与 filtered belief $q(\mathbf{z}_t | \mathbf{o}_{\le t}, \mathbf{a}_{\le t-1})$ 相同的信息
- 允许 regularization（见下文）
- Monte-Carlo Dropout 在此引入 epistemic uncertainty

**为什么需要 Monte-Carlo Dropout？** Smoothing inference 使用 future information 来推断 past states。如果 model 对未来 trajectory 的 epistemic uncertainty 没有建模，smoother 会过度自信，导致 overfitting。MC Dropout ([Gal & Ghahramani 2016](https://arxiv.org/abs/1506.02142)) 提供 Bayesian approximation 来量化这种不确定性。

---

## 4. 并行 Kalman Filter/Smoother 的数学

这是 paper 最 technical 的核心。基于 [Särkkä & García-Fernández 2020](https://arxiv.org/abs/1905.13002) 的工作。

### 4.1 标准 Kalman Filter 回顾

**Prediction step**:
$$\mu_{t|t-1} = A_t \mu_{t-1|t-1} + b_t$$
$$\Sigma_{t|t-1} = A_t \Sigma_{t-1|t-1} A_t^T + \Sigma_t^{\text{dyn}}$$

**Update step** (with observation $\mathbf{w}_t$):
$$K_t = \Sigma_{t|t-1} (\Sigma_{t|t-1} + \Sigma_t^{\mathbf{w}})^{-1}$$
$$\mu_{t|t} = \mu_{t|t-1} + K_t (\mathbf{w}_t - \mu_{t|t-1})$$
$$\Sigma_{t|t} = (I - K_t) \Sigma_{t|t-1}$$

（这里 observation model 是 $H = I$ 因为 $q(\mathbf{w}_t | \mathbf{z}_t) = \mathcal{N}(\mathbf{w}_t | \mathbf{z}_t, \Sigma_t^{\mathbf{w}})$）

### 4.2 Associative Formulation

关键 insight：将 filter 的 prediction-update 表示为两个 "filtering elements" 的 associative combination。

定义 filtering element at time $t$:
$$e_t = (A_t, b_t, \Sigma_t^{\text{dyn}}, \mathbf{w}_t, \Sigma_t^{\mathbf{w}})$$

定义 binary operator $\otimes$ that combines $e_{i:j}$ (filter from $i$ to $j$) with $e_{j:k}$:

$$e_{i:k} = e_{i:j} \otimes e_{j:k}$$

这个 operator 满足 **associativity**：$(e_1 \otimes e_2) \otimes e_3 = e_1 \otimes (e_2 \otimes e_3)$。

具体的 combination rules（简化版，假设 diagonal）：
$$A_{i:k} = A_{j:k} \cdot A_{i:j}$$
$$b_{i:k} = A_{j:k} \cdot b_{i:j} + b_{j:k}$$
$$\Sigma_{i:k} = A_{j:k}^2 \cdot \Sigma_{i:j} + \Sigma_{j:k}$$

由于 associativity，可以用 **parallel prefix sum** (parallel scan) 算法在 $O(\log T)$ 时间内计算所有前缀组合。

### 4.3 Kalman Smoother 的并行化

Rauch-Tung-Striebel (RTS) smoother 也可以类似地 associative formulation。Smoother 需要一个 backward pass：

$$\mu_{t|T} = \mu_{t|t} + J_t (\mu_{t+1|T} - \mu_{t+1|t})$$
$$\Sigma_{t|T} = \Sigma_{t|t} + J_t (\Sigma_{t+1|T} - \Sigma_{t+1|t}) J_t^T$$

其中 $J_t = \Sigma_{t|t} A_t^T (\Sigma_{t+1|t})^{-1}$ 是 smoothing gain。

同样可以定义 backward associative operator，用 parallel scan 实现。

### 4.4 Diagonal Structure 的作用

所有 covariance matrices 都是 diagonal：
- $\Sigma_t^{\text{dyn}}$: diagonal（设计选择）
- $\Sigma_t^{\mathbf{w}}$: diagonal（encoder 输出）
- $\Sigma_0$: diagonal（learned）
- $\mathbf{A}_t$: diagonal（设计选择）

这意味着 filtered/smoothed belief 的 covariance 也是 diagonal（通过 induction）。所有 matrix operations 变成 element-wise operations：
- Matrix multiply → element-wise multiply
- Matrix inverse → element-wise inverse
- Memory: $O(n)$ vs $O(n^2)$

这在 RL 中 latent dimension 通常 30-256 的规模下，带来巨大的 speedup。

---

## 5. Training Objective 详解

### 5.1 Variational Lower Bound

$$\mathcal{L}_{\text{ssm}} = \sum_{t=1}^{T} \bigg(\underbrace{\mathbb{E}_{q(\mathbf{z}_t | \mathbf{o}_{\le T}, \mathbf{a}_{\le T})}[\log p(\mathbf{o}_t | \mathbf{z}_t)]}_{\text{reconstruction}} - \underbrace{\mathbb{E}_{q(\mathbf{z}_{t-1} | \mathbf{o}_{\le T}, \mathbf{a}_{\le T})}[\text{KL}[q(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{a}_{\ge t-1}, \mathbf{o}_{\ge t}) \| p(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{a}_{t-1})]]}_{\text{KL regularization}}\bigg)$$

变量解释：
- $q(\mathbf{z}_t | \mathbf{o}_{\le T}, \mathbf{a}_{\le T})$: **smoothed belief**，条件于所有观测和动作（past + future）
- $q(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{a}_{\ge t-1}, \mathbf{o}_{\ge t})$: **smoothed dynamics**，给定 $\mathbf{z}_{t-1}$ 和 future information 的 posterior
- $p(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{a}_{t-1})$: **prior dynamics**，model 的 prediction

**为什么这个 bound 是 tight 的？** 因为 smoothing posterior $q(\mathbf{z}_t | \mathbf{o}_{\le T}, \mathbf{a}_{\le T})$ 比 filtering posterior $q(\mathbf{z}_t | \mathbf{o}_{\le t}, \mathbf{a}_{\le t})$ 更接近 true posterior $p(\mathbf{z}_t | \mathbf{o}_{\le T})$。在 filtering 中，KL term 会有一个额外的 "filtering gap" 因为 posterior 只用了 partial information。

### 5.2 Mahalanobis Regularizer

$$R = \sum_{t=1}^{T} (\mathbf{m}_t - \mu_t^+)^T (\Sigma_t^+)^{-1} (\mathbf{m}_t - \mu_t^+) \tag{2}$$

变量：
- $\mathbf{m}_t$: Mamba backbone 的中间表示，shape $(d_z,)$
- $\mu_t^+$: filtered belief $q(\mathbf{z}_t | \mathbf{o}_{\le t}, \mathbf{a}_{\le t-1})$ 的 mean
- $\Sigma_t^+$: filtered belief 的 covariance（diagonal）

**Intuition**：这是 weighted L2 distance，权重是 inverse covariance。它在 covariance 大的维度上允许 $\mathbf{m}_t$ 偏离 $\mu_t^+$，在 covariance 小的维度上严格要求匹配。

**为什么需要这个？** 如果没有它，Mamba backbone 可能学习直接把 $\mathbf{o}_{\le t}$ 的信息编码进 $\mathbf{A}_t, \mathbf{b}_t$，bypass 掉 latent state $\mathbf{z}_t$ 的信息流。这样 Kalman filter 变成空壳，无法提供有意义的不确定性估计。Regularizer 强制 information 必须流经 SSM。

这 mirror 了经典 EKF 的设计：EKF 在 filtered mean 处 linearize dynamics，KalMamba 也 incentivize backbone 输出与 filtered mean 对齐。

### 5.3 完整 Objective

$$\mathcal{L}_{\text{KalMamba}} = \mathcal{L}_{\text{ssm}} + \mathbb{E}_{q(\mathbf{z}_t | \mathbf{o}_{\le T}, \mathbf{a}_{\le T})}[\log p(r_t | \mathbf{z}_t)] - \alpha R$$

- Reward model $p(r_t | \mathbf{z}_t)$: 帮助 latent state 编码 task-relevant 信息
- $\alpha = 1$: regularizer 权重

---

## 6. RL 集成：Filtered vs Smoothed Belief

关键设计：**训练时用 smoothed belief，acting 时用 filtered belief**。

- Smoothed belief $q(\mathbf{z}_t | \mathbf{o}_{\le T}, \mathbf{a}_{\le T})$: 用于 model training（tight ELBO）
- Filtered belief $q(\mathbf{z}_t | \mathbf{o}_{\le t}, \mathbf{a}_{\le t-1})$: 用于 SAC actor/critic 输入

**为什么这能 work？** Smoothing pass 不引入额外 trainable parameters（只是 Kalman smoother 的 closed-form computation）。因此 smoothing 的 gradient 会 backpropagate 通过 filter 的 parameters，间接训练 filtered belief 的质量。这是一种 **inductive bias**：smoothing objective 拉着 filtered belief 一起变好。

与 latent imagination methods（Dreamer 系列）的区别：KalMamba 直接用 Q-function on beliefs，避免 autoregressive rollout 在 latent space，进一步减少 sequential computation。

---

## 7. 实验结果分析

### 7.1 Image-based DMC Suite (Figure 3)

Tasks: cartpole_swingup, quadruped_walk, walker_walk, walker_run

- KalMamba 略差于 RSSM+SAC 和 VRKN+SAC，但整体 competitive
- 所有 SSM+SAC 组合都 match 或超过 DreamerV3
- **Ablation 关键发现**：
  - **No Mamba**: 性能大幅下降。说明用 single SSM layer linearize around current $(\mathbf{o}_t, \mathbf{a}_t)$ 不够，需要 Mamba 的 stacked layers 和 temporal context
  - **No Regularization**: 训练不稳定，性能下降
  - **No MC Dropout**: 性能下降，尤其在最难的任务 quadruped_walk

### 7.2 Noisy State-based Tasks (Figure 4)

- Observation noise: $\mathcal{N}(0, 0.3)$ 加到 observation
- Dynamics noise: $\mathcal{N}(0, 0.3)$ 加到 action（在 environment 内部，invisible to model）

结果：
- **Naive SAC**: 完全失败（噪声太大）
- **RSSM**: 有改善但显著差于 VRKN 和 KalMamba
- **VRKN**: 最好
- **KalMamba**: 收敛稍慢但最终接近 VRKN

这验证了 smoothing inference 在 uncertainty 下的关键作用。RSSM 的 filtering-only inference 无法正确处理这种 noise level。

### 7.3 Runtime Scaling (Figure 5, 6) - 最关键的结果

在 state-based noisy walker-walk 上测试不同 sequence length：

| Seq Length | RSSM | VRKN | KalMamba |
|------------|------|------|----------|
| 32 | ~1M steps/24h | ~1M steps/24h | ~1M steps/24h |
| 256 | 慢，性能下降 | **超时** (650k steps/24h) | **几乎无额外开销，性能提升** |

**Forward pass 时间** (Figure 6 left)：
- RSSM, VRKN: linear scaling in sequence length
- KalMamba: near-logarithmic scaling

**Training batch 时间** (Figure 6 right)：
- 同样的 scaling pattern

这直接展示了 associative scan 的优势。在 seq=256 时，KalMamba 比 VRKN 快约 4x（从超时到完成 1M steps）。

更 interesting 的是 **性能也随 sequence length 提升**：KalMamba 在 seq=256 比 seq=32 表现更好。这表明 Mamba backbone 能有效利用 long-term context，而 baseline SSMs 的 dynamics model 在长序列上 degrade。

---

## 8. Intuition Building: 为什么这个设计 work？

### 8.1 解耦的智慧

KalMamba 的核心是 **分离 concerns**：
- Mamba 负责：高维 observation history → low-dimensional dynamics parameters
- Kalman 负责：latent space 中的 probabilistic inference

Mamba 不直接做 probabilistic inference（它不擅长），Kalman 不直接处理高维输入（它不能）。两者各司其职。

### 8.2 Linear Gaussian 作为 Interface

Linear Gaussian SSM 是 Mamba 和 Kalman 之间的 perfect interface：
- Mamba 可以 output $(A_t, b_t, \Sigma_t)$ 任意 nonlinear function of history
- Kalman 可以 closed-form infer 给定这些参数
- 这让两个 module 的 gradient 可以 clean 地 backpropagate

### 8.3 Diagonal Constraint 的 trade-off

Diagonal matrices 牺牲 expressiveness 换取效率：
- Full matrix: $O(n^3)$ matrix operations, $O(n^2)$ memory
- Diagonal: $O(n)$ point-wise operations, $O(n)$ memory

在 latent dim ~30 (stochastic) + ~200 (deterministic for RSSM) 的规模下，这个 trade-off 划算。而且 Mamba backbone 可以补偿 expressiveness 的损失，通过 input-dependent parameterization。

### 8.4 Smoothing-Filtering Coupling

这是最 subtle 的设计。传统做法要么：
- 只用 filtering（RSSM, R2I）：parallelizable 但 loose bound
- 只用 smoothing for training, filtering for acting（VRKN）：tight bound 但无法并行

KalMamba 的 trick：smoothing 不引入新参数，所以 smoothing objective 的 gradient 自动 improve filtering quality。这创造了一个 **free lunch**：得到 smoothing 的好处（tight bound, good uncertainty）without 额外参数，且两者都可并行。

---

## 9. 与相关工作的联系

### 9.1 Latent S4 (LS4, [Zhou et al. 2023](https://arxiv.org/abs/2305.10521))

LS4 也尝试给 deterministic SSM 加 probabilistic interpretation，但 latent states 不是 Markovian（依赖整个 history via S4 的 recurrent state），难以用于 control。KalMamba 保持 Markov property 通过 linear Gaussian SSM。

### 9.2 Deep Variational Bayes Filters (DVBF, [Karl et al. 2016](https://arxiv.org/abs/1605.06432))

DVBF 也在 latent space 用 linear Gaussian SSM + Kalman inference，但 dynamics 参数是 fixed 或简单 parameterized，无法处理复杂 observation。KalMamba 用 Mamba 解决这个 limitation。

### 9.3 Switching Linear Dynamics ([Becker-Ehmck et al. 2019](https://arxiv.org/abs/1906.10115))

用 mixture of linear dynamics models，通过 discrete latent variable switch。KalMamba 用 continuous time-varying parameters via Mamba，更 flexible。

### 9.4 Backprop KF ([Haarnoja et al. 2016](https://arxiv.org/abs/1605.07470))

Learn discriminative state estimator via backprop through Kalman filter。KalMamba 用 generative model + variational inference，更 principled 且能 sample。

---

## 10. 局限性和未来方向

Paper 自己提到的：
1. 只在 small-scale DMC tasks 上验证，需要更 realistic tasks
2. 没有和 R2I 直接比较（不同 inference scheme 的 trade-off）
3. Complex-valued latent state 可以 expand learnable dynamics range
4. Better regularization for Mamba backbone
5. Policy learning 可以利用 filtered belief 的 uncertainty（目前 SAC 只用 mean）

我会补充的：
1. **Diagonal constraint 的 expressiveness loss**：在需要 cross-dimension correlation 的任务（如 multi-object interaction）可能受限。Future work 可以探索 block-diagonal 或 low-rank approximations
2. **Mamba backbone 的 gradient flow**：smoothing pass 的 gradient 通过 closed-form Kalman 回传到 Mamba，可能存在 vanishing/exploding gradient 问题（虽然 paper 没报告）
3. **与 Transformer world models 的比较**：如 [TransDreamer](https://arxiv.org/abs/2202.09481) 的 attention-based approach，在超长 context 上可能更有优势
4. **CUDA custom kernel**：Paper 提到 PyTorch 实现已经很快，但 custom CUDA（像 Mamba 的 mamba_ssm 包）可能再加速 2-5x

---

## 11. 关键 Take-aways

1. **Probabilistic + Efficient 不矛盾**：通过 careful architectural design，可以同时获得 uncertainty modeling 和 time-parallelizable efficiency
2. **Associative scan 是万能工具**：不仅用于 deterministic SSMs，也能加速 probabilistic inference（Kalman filter/smoother）
3. **Interface design 很重要**：Linear Gaussian SSM 作为 Mamba 和 Kalman 之间的 interface，让两个 module 各自发挥优势
4. **Smoothing-filtering coupling**：通过不引入额外参数的 smoothing pass，free-ride 提升 filtering quality
5. **Diagonal structure 是 practical 的**：在 RL 的 latent dimension 规模下，expressiveness loss 被 efficiency gain 弥补

---

## References

- [KalMamba paper](https://arxiv.org/abs/2410.14682) (paper 本身)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [S4: Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
- [S5: Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933)
- [RSSM / PlaNet: Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1911.01670)
- [DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- [VRKN: On Uncertainty in Deep State Space Models](https://arxiv.org/abs/2202.10650)
- [R2I: Mastering Memory Tasks with World Models](https://arxiv.org/abs/2310.04407)
- [Temporal Parallelization of Bayesian Smoothers](https://arxiv.org/abs/1905.13002)
- [Monte-Carlo Dropout](https://arxiv.org/abs/1506.02142)
- [SAC: Soft Actor-Critic](https://arxiv.org/abs/1812.05905)
- [TransDreamer](https://arxiv.org/abs/2202.09481)
- [LS4: Deep Latent State Space Models for Time-Series Generation](https://arxiv.org/abs/2305.10521)
- [Scan Primitives for GPU Computing](https://graphics.cs.uni-saarland.de/?page_id=890)
- [Backprop KF](https://arxiv.org/abs/1605.07470)
- [DeepMind Control Suite](https://arxiv.org/abs/1801.00690)
