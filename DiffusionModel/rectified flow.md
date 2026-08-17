---
source_pdf: rectified flow.pdf
paper_sha256: d81790cb6988ef480002adcc84000d4f44855b11fb868001fd53aad133bc3a74
processed_at: '2026-08-11T21:56:03-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Rectified Flow 人话版

## 一句话说清楚

**把一堆点从 A 搬到 B，最笨的办法是走直线，但直线会打架。Rectified flow 让你学一个 ODE，尽量走直线，但 ODE 天生不会打架。多走几次（reflow），路就越走越直了。**

---

## 问题是什么

你有一堆 noise（$\pi_0$），想变成一堆 image（$\pi_1$）。

或者你有一堆 cat face（$\pi_0$），想变成一堆 human face（$\pi_1$）。

本质都是：**把分布 A 变成分布 B**。

---

## 最 naive 的想法

随便从 A 抽一个 $X_0$，从 B 抽一个 $X_1$，连一条直线：

$$X_t = tX_1 + (1-t)X_0$$

$t=0$ 时在 $X_0$，$t=1$ 时在 $X_1$。沿着直线走过去不就行了？

**问题**：直线会交叉。Figure 2a 里你能看到一堆线乱穿。

为什么交叉是坏事？因为你要的是一个 **causal** 的过程——给定当前位置，你能决定下一步往哪走。但如果两条直线在中间交叉了，说明站在交叉点上，你不知道该沿哪条线走（一条去这个 $X_1$，另一条去那个 $X_1$）。你需要"看见未来"才能决定方向，这不 causal。

---

## ODE 天生不交叉

ODE $dZ_t = v(Z_t, t) dt$ 的解是唯一的。同一个点 $(z, t)$ 只有一个速度 $v(z, t)$，所以两条 trajectory 不可能在同一个时空点交叉。一旦交叉，解就不唯一了，矛盾。

**这就是 rectified flow 的核心 trick**：用 ODE 来"模仿"直线，但 ODE 的 non-crossing 性质会自动帮你把交叉的路"拨开"。

---

## 怎么"模仿"直线？

训练一个神经网络 $v(x, t)$，目标是：

$$\min_v \mathbb{E}\left[\|(X_1 - X_0) - v(X_t, t)\|^2\right]$$

翻译成人话：
- $X_t$ 是直线上 $t$ 时刻的位置
- $X_1 - X_0$ 是直线的方向（constant velocity）
- 你让网络 $v$ 在位置 $X_t$、时刻 $t$ 时，预测的方向尽量接近 $X_1 - X_0$

**最优解**是条件期望：
$$v^*(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x]$$

意思是：站在 $x$，看所有经过 $x$ 的直线，取它们方向的**平均**。

这就是"拨开交叉"的数学实现——交叉点处有多个方向，你取平均，得到一个确定的方向，交叉就消失了。

---

## 为什么"拨开后"还是对的？

**Theorem 3.3 (Marginal preserving)**: 虽然你拨开了交叉，但每一时刻 $t$ 的 marginal distribution 不变。

直觉：拨开交叉是在"重新分配路径"，但每个时刻每个位置进出多少 mass 是不变的。就像交通管制，你重新规划了路线，但每个时刻每个路口的车流量不变。

所以 $Z_0 \sim \pi_0$ 沿 ODE 走到 $Z_1$，$Z_1$ 还是服从 $\pi_1$。**你的 ODE 确实把 $\pi_0$ 变成了 $\pi_1$**。

---

## 为什么 transport cost 不会增加？

**Theorem 3.5**: $\mathbb{E}[c(Z_1 - Z_0)] \leq \mathbb{E}[c(X_1 - X_0)]$ 对所有凸 $c$ 成立。

直觉（用 $c = \|\cdot\|$ 即路径长度）：
- 原来：每条直线长度 $\|X_1 - X_0\|$
- 现在：ODE 走的路径长度 $\int \|\dot{Z}_t\| dt$
- ODE 是直线的"rewiring"，走的路更"合并"、更"共享"
- 合并后的路总长度更短（三角不等式）

类比：原本 10 个人各走各的直线去 10 个目的地，总路程 = 10 条直线之和。现在让他们先合并走一段，再分开，总路程更短（因为共享的部分只算一次）。

---

## Reflow：让路越走越直

第一次 rectified flow 之后，ODE 走的不是完美直线（因为取了平均，方向会变）。

**Reflow 的 idea**：
1. 用第一版 ODE 生成一堆 $(Z_0, Z_1)$ 对
2. 这堆对的 transport cost 已经比原来低了
3. 在这堆新对上再训一次 rectified flow

每次 reflow，路径都更直一点。

**Theorem 3.7**: straightness $S(Z) \to 0$ at rate $O(1/K)$。

实践上：**reflow 1-2 次就够了**，路径已经很直。

---

## 路径直了有什么用？

**直线 ODE 可以用 1 步 Euler 精确求解**：
$$Z_1 = Z_0 + v(Z_0, 0) \times 1$$

如果路径完美直，$v$ 沿 path 是常数，一步就到终点。

这就是从 "1000 步 diffusion" 到 "1 步 generation" 的飞跃。

---

## 和 DDPM/DDIM 的关系

DDPM 的 derivation 是：
```
Forward SDE (OU process) → Time reversal → Backward SDE → 转 ODE (PF-ODE)
```

绕了一大圈，最后用 ODE。而且因为 OU process 的历史包袱，PF-ODE 的 $\alpha_t$ 是指数形式（Equation 7），导致：
1. 路径是弯的（Figure 4）
2. 速度不均匀（前期慢后期快，Figure 5）
3. Reflow 也直化不了（因为不是 linear interpolation）

**Rectified flow 的观点**：既然目标是 ODE，为什么从 SDE 绕？直接学 ODE，用 linear interpolation，路径直、速度快、能 reflow。

**Proposition 3.11**: VP ODE / DDIM / sub-VP ODE 都是 nonlinear rectified flow 的特例，只是 $\alpha_t, \beta_t$ 选得不好。

---

## 为什么 diffusion noise 不是必需的？

DDPM 成功的真正原因：
1. **简单的 regression loss**（不是 GAN 的 minimax）
2. **稳定的训练**（不 collapse）
3. **可扩展到大模型**

Noise 本身不是关键。Rectified flow 用纯 ODE + 同样的 regression loss，达到了类似甚至更好的效果。Noise 的主要作用是在 SDE 的 derivation 里，但如果你直接学 ODE，noise 可以完全去掉。

---

## Image-to-Image translation 怎么做？

**关键 insight**：把 $\pi_0$ 设成 source domain（不是 Gaussian）就行了。

同一个算法，既做 generation（$\pi_0 = $ Gaussian），又做 transfer（$\pi_0 = $ cat face, $\pi_1 = $ human face）。

CycleGAN 需要 adversarial + cycle consistency，rectified flow 只需要 ODE 的 reversibility（本来就可逆）。

唯一的小修改（Equation 20）：加一个 saliency weight $\nabla h(x)$，保证转移时保留 identity。

---

## 整个 pipeline

```
1. Train: 在 (X_0, X_1) 上训 v(x,t)，target = X_1 - X_0
   ↓
2. Reflow (1-2次): 用 v 生成新 coupling，再训 v
   ↓ 路径变直
3. Distill: 把直的 flow 蒸馏成 one-step model
   ↓
4. Inference: z_1 = z_0 + v(z_0, 0)  # 一步出图
```

---

## 最核心的 intuition

1. **Linear interpolation 是 Euclidean geodesic**：最短、最直、constant speed。选它没坏处。
2. **ODE non-crossing = 自动 deconvolution**：取平均消除多解性，得到确定 coupling。
3. **Reflow = self-distillation**：每次让 flow 更可预测，路径更短更直。
4. **Straight = one-step**：路径直了，一步 Euler 就解，不用 1000 步。
5. **不绕 SDE**：直接学 ODE 更简单、更自由、效果不差。

---

## 一句话再总结

**Rectified flow = 用 ODE 模仿直线 + reflow 把路拉直 + distill 成一步。**

简单到你觉得"这也能发 ICLR？"，但就是 work，而且比绕了一大圈的 diffusion 还好。这就是好 taste。

---

# Rectified Flow 深度技术讲解

这是 Liu et al. 2022 的工作，提出了一个看似简单但极其强大的框架，统一了 generative modeling 和 domain transfer。核心 idea 是**学习 ODE 跟踪连接两个分布样本的直线路径**。

---

## 1. 核心问题与 Motivation

### 1.1 The Transport Mapping Problem

给定两个经验分布 $X_0 \sim \pi_0$, $X_1 \sim \pi_1$ on $\mathbb{R}^d$，寻找 transport map $T: \mathbb{R}^d \to \mathbb{R}^d$，使得当 $Z_0 \sim \pi_0$ 时，$T(Z_0) \sim \pi_1$。

这涵盖了：
- **Generative modeling**: $\pi_0 = \mathcal{N}(0, I)$, $\pi_1$ = data distribution
- **Domain transfer**: $\pi_0, \pi_1$ 都是 arbitrary empirical distributions (image-to-image translation, domain adaptation)
- **Latent representation**: 将 data 映射到 simple distribution

### 1.2 现有方法的痛点

| 方法 | 问题 |
|------|------|
| **GAN** | minimax 不稳定，mode collapse，需要大量 tuning |
| **VAE** | likelihood 不可解，需要变分近似 |
| **Normalizing Flow** | 需要 invertible 架构，Jacobian 计算昂贵 |
| **DDPM/SDE** | inference 慢（需 1000+ steps），design space 复杂，依赖 stochastic calculus |
| **Neural ODE (MLE)** | 需要 backprop through time，likelihood 计算昂贵 |

Rectified flow 的 motivation 是：**既然我们要学 ODE，为什么要从 SDE 绕一圈？为什么不直接学 ODE 跟踪直线路径？**

---

## 2. 算法核心

### 2.1 Algorithm 1: Rectified Flow

**输入**: 从 $\pi_0, \pi_1$ 的样本对 $(X_0, X_1)$ (可以 independent coupling)

**训练**：求解 least squares 问题

$$\min_v \int_0^1 \mathbb{E}\left[\left\|(X_1 - X_0) - v(X_t, t)\right\|^2\right] dt, \quad X_t = tX_1 + (1-t)X_0$$

**变量解释**：
- $X_0 \in \mathbb{R}^d$: 从 $\pi_0$ 抽取的样本（e.g., Gaussian noise）
- $X_1 \in \mathbb{R}^d$: 从 $\pi_1$ 抽取的样本（e.g., real image）
- $t \in [0,1]$: 时间参数
- $X_t = tX_1 + (1-t)X_0$: **线性插值**，从 $X_0$ 到 $X_1$ 的直线
- $v: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$: 神经网络参数化的速度场
- $(X_1 - X_0)$: 直线的方向（constant velocity）

**Intuition**: 把 $X_t$ 想象成一条从 $X_0$ 到 $X_1$ 的直线公路，$v(X_t, t)$ 是 ODE 在位置 $X_t$ 时刻 $t$ 的速度。我们希望 ODE 的速度尽可能匹配直线方向 $(X_1 - X_0)$。

**采样**：从 $Z_0 \sim \pi_0$ 开始，解 ODE
$$dZ_t = v_{\hat{\theta}}(Z_t, t) dt$$
得到 $Z_1 \sim \pi_1$。

### 2.2 为什么这个 loss 是对的？

最优解 (Equation 2):
$$v^X(x, t) = \mathbb{E}[X_1 - X_0 | X_t = x]$$

这是**条件期望**：给定当前位置 $x$ 和时间 $t$，返回所有通过 $x$ 的直线的平均方向。

**关键性质**:
1. **Marginal preserving** (Theorem 3.3): $\text{Law}(Z_t) = \text{Law}(X_t), \forall t$
   - 直觉：通过 mass conservation，流入流出每个体积元素的质量相同
2. **Transport cost 非增** (Theorem 3.5): 对所有凸函数 $c$，
   $$\mathbb{E}[c(Z_1 - Z_0)] \leq \mathbb{E}[c(X_1 - X_0)]$$

### 2.3 Flows 避免交叉

这是理解 rectified flow 的关键。

**线性插值 $X_t$ 的问题**: 不同样本的直线可能交叉（Figure 2a），导致 non-causal——你要知道终点 $X_1$ 才能知道当前方向。

**ODE 的性质**: 良定义的 ODE 解唯一，不同 trajectory 不能交叉。否则在交叉点 ODE 有多个解。

**Rectified flow 的作用**: "rewire" 在交叉点的轨迹，避免交叉，同时保持 marginal distributions 不变（Figure 2b）。

**物理直觉**: 把 $X_t$ 看作建好的公路网，rectified flow 是在公路上行驶的粒子流，遵守"不交叉"规则，结果是更确定的配对 $(Z_0, Z_1)$。

---

## 3. Reflow: 直化路径

### 3.1 算法

```
Z^1 = RectFlow((X_0, X_1))         # 第一次 rectification
Z^2 = RectFlow((Z_0^1, Z_1^1))     # reflow
Z^3 = RectFlow((Z_0^2, Z_1^2))     # 再 reflow
...
```

每次 reflow:
1. 用当前 flow 生成耦合 $(Z_0^k, Z_1^k)$
2. 在这个新耦合上训练新的 rectified flow

### 3.2 直度度量

$$S(Z) = \int_0^1 \mathbb{E}\left[\left\|(Z_1 - Z_0) - \dot{Z}_t\right\|^2\right] dt$$

- $S(Z) = 0$: 完美直线
- $\dot{Z}_t$: ODE 在时刻 $t$ 的实际速度
- $(Z_1 - Z_0)$: 直线方向

### 3.3 Theorem 3.7: $O(1/K)$ 收敛

$$\min_{k \in \{0, \ldots, K\}} S(Z^k) \leq \frac{\mathbb{E}[\|X_1 - X_0\|^2]}{K}$$

**Proof sketch** (telescoping sum):
对 $c(x) = \|x\|^2$，每次 rectification:
$$\mathbb{E}[\|Z_1^k - Z_0^k\|^2] - \mathbb{E}[\|Z_1^{k+1} - Z_0^{k+1}\|^2] = S(Z^{k+1}) + V((Z_0^k, Z_1^k))$$

对所有 $k$ 求和得 telescoping，bound 为初始 $\mathbb{E}[\|X_1 - X_0\|^2]$。

**Intuition**: 每次 reflow 都在"削平"路径的弯曲部分。1/K 速率说明只需 1-2 次 reflow 就能获得显著直化效果。

### 3.4 为什么直线路径好？

**Burgers 方程视角**: 若 flow 是直的 ($v(Z_t, t) = Z_1 - Z_0 = \text{const}$)，则 $v$ 满足 inviscid Burgers' equation:
$$\partial_t v + (\partial_z v) v = 0$$

这是因为 $\frac{d}{dt}v(Z_t, t) = \partial_z v \cdot \dot{Z}_t + \partial_t v = \partial_z v \cdot v + \partial_t v = 0$（速度沿 path 不变）。

**计算优势**: 直线 flow 可以用 single Euler step 精确求解:
$$Z_1 = Z_0 + v(Z_0, 0) \cdot 1$$

这等价于 **one-step model**！

---

## 4. 非线性扩展: 与 Probability Flow ODE 的关系

### 4.1 Generalized Rectified Flow

用任意时间可微曲线替代线性插值：
$$\min_v \int_0^1 \mathbb{E}\left[w_t \|v(X_t, t) - \dot{X}_t\|^2\right] dt$$

其中 $\dot{X}_t$ 是 $X_t$ 的时间导数。

**重要**: 只有线性插值保证 transport cost 非增 + reflow 直化。非线性插值仍保持 marginal preserving，但失去这两个性质。

### 4.2 PF-ODE 和 DDIM 是特例

**VP ODE / DDIM** (Equation 7, 8):
$$\alpha_t = \exp\left(-\frac{1}{4}a(1-t)^2 - \frac{1}{2}b(1-t)\right), \quad \beta_t = \sqrt{1 - \alpha_t^2}$$
默认 $a = 19.9, b = 0.1$

**sub-VP ODE**:
$$\beta_t = 1 - \alpha_t^2$$

这里 $X_t = \alpha_t X_1 + \beta_t \xi$, $\xi \sim \mathcal{N}(0, I)$。

**VE ODE**:
$$\alpha_t = 1, \quad \beta_t = \sigma_{\min}\sqrt{r^{2(1-t)} - 1}$$

**Proposition 3.11**: 所有 PF-ODE 都是 nonlinear rectified flow 的特例，用 $X_t = \alpha_t X_1 + \beta_t \xi$。

### 4.3 为什么 VP/sub-VP ODE 不好？

**问题 1: Non-straight paths** (Figure 4, 5)
- $\beta_t \neq 1 - \alpha_t$ 导致曲线轨迹
- Reflow 无法直化（因为不是 linear interpolation）

**问题 2: Non-uniform speed** (Figure 5, 6)
- 指数 $\alpha_t$ 在 $t \approx 0.5$ 才快速变化
- 大部分更新集中在后期
- 即使用单步也效果差

**对比**: Rectified flow 用 $\alpha_t = t, \beta_t = 1-t$，路径直线，速度均匀。

**Key insight**: PF-ODE 的设计约束来自 SDE-to-ODE 的 derivation（OU process），但这个约束对 ODE 本身没有意义。Rectified flow 直接学 ODE，摆脱了这个约束。

---

## 5. 实验结果分析

### 5.1 CIFAR10 (Table 1a)

| Method | NFE | FID↓ | Recall↑ |
|--------|-----|------|---------|
| 1-Rectified Flow (RK45) | 127 | **2.58** | **0.57** |
| 2-Rectified Flow + Distill (N=1) | 1 | **4.85** | 0.50 |
| 3-Rectified Flow + Distill (N=1) | 1 | 5.21 | **0.51** |
| VP ODE (RK45) | 140 | 3.93 | 0.51 |
| sub-VP ODE (RK45) | 146 | 3.16 | 0.55 |
| VP SDE (N=2000) | 2000 | 2.55 | 0.58 |

**观察**:
1. 1-Rectified Flow 全精度超过所有 ODE 方法，接近 SDE
2. 2-Rectified Flow + Distill one-step 达到 **FID 4.85**，超过所有 one-step U-Net GAN
3. Recall (diversity) 0.51 超过 StyleGAN2+ADA 的 0.49

### 5.2 直化效果 (Figure 9)

通过测量 $\hat{z}_1^t = z_t + (1-t)v(z_t, t)$ 对 $t$ 的依赖性：
- 完美直线 flow: $\hat{z}_1^t$ 应与 $t$ 无关
- 2-Rectified Flow: $\hat{z}_1^t$ 几乎不变 → 几乎完美直线
- 1-Rectified Flow: 早期 ($t \approx 0.1$) 就能 extrapolate 出清晰图像
- sub-VP ODE: 需要 $t \approx 0.6$ 才能获得清晰 extrapolation

### 5.3 Image-to-Image Translation

**创新点**: 同一个算法，只需把 $\pi_0$ 设为 source domain（不再是 Gaussian）。

**关键修改** (Equation 20):
$$\min_v \int_0^1 \mathbb{E}\left[\left\|\nabla h(X_t)^\top (X_1 - X_0 - v(X_t, t))\right\|_2^2\right] dt$$

- $h(x)$: domain classifier 的 feature map
- $\nabla h(x)^\top$: saliency weighting，重点惩罚改变 identity 的误差

**结果** (Figure 13, 14):
- Cat ↔ Wild Animals, MetFace ↔ CelebA
- 2-Rectified Flow 用 single Euler step 就能获得高质量转换
- 无需 CycleGAN 的 adversarial training 和 cycle consistency

### 5.4 Domain Adaptation (Table 2)

| Method | OfficeHome | DomainNet |
|--------|-----------|-----------|
| ERM | 66.5 | 40.9 |
| CORAL | 68.7 | 41.5 |
| **Ours** | **69.2** | 41.4 |

在 latent space 上构造 rectified flow，将 test domain 映射到 training domain。

---

## 6. 与 DDPM/Diffusion 的深层对比

### 6.1 DDPM 的 derivation 路径

```
Forward SDE (OU process) → Time reversal → Backward SDE → PF-ODE
```

**问题**:
1. 需要 stochastic calculus 工具
2. Forward process 限定为 OU（为可逆性）
3. $\alpha_t, \beta_t$ 由 OU process 决定，而非 ODE 优化
4. 引入 noise 不必要（若目标是 ODE）

### 6.2 Rectified flow 的 derivation

```
直接指定 X_t = tX_1 + (1-t)X_0 (linear interpolation)
   → 最小二乘拟合 v
   → ODE
```

**优势**:
- 概念简单
- 路径可任意指定
- $\pi_0$ 与 $X_t$ 解耦
- Linear interpolation 是默认推荐（geodesic）

### 6.3 Diffusion noise 的角色重新审视

Paper 的重要 claim: **Diffusion noise 不是 DDPM 成功的关键**。成功主要来自：
- 简单稳定的优化（避免 GAN 的 minimax）
- 可扩展性
- 无需 case-by-case tuning

Rectified flow 用纯 ODE 达到可比性能，证明 noise 非必需。

---

## 7. 理论细节深挖

### 7.1 Theorem 3.5 证明直觉

对 $c(\cdot) = \|\cdot\|$ (L1 cost):
$$\mathbb{E}[\|Z_0 - Z_1\|] = \text{Length}(Z_t \text{ path}) \stackrel{(*)}{\leq} \text{Length}(\text{rewired straight paths}) \stackrel{(**)}{=} \text{Length}(X_t \text{ paths}) = \mathbb{E}[\|X_0 - X_1\|]$$

- $(*)$: 三角不等式（curve 比 rewired straight 短不可能，因为 rewired 路径不交叉）
- $(**)$: Rectified flow 是 $X_t$ 的 rewiring，保持 marginal

对一般凸 $c$: 用 Jensen 不等式两次。

### 7.2 Theorem 3.6: 直线的等价刻画

以下等价：
1. 存在严格凸 $c$ 使得 $\mathbb{E}[c(Z_1-Z_0)] = \mathbb{E}[c(X_1-X_0)]$
2. $(X_0, X_1) = (Z_0, Z_1)$ (fixed point)
3. $X = Z$ (linear interpolation = rectified flow)
4. $V((X_0, X_1)) = 0$ (路径不交叉)

**Intuition**: 直线 = 不交叉 = fixed point of Rectify = 严格凸 cost 的 equality case。

### 7.3 1D 特殊情况 (Theorem 3.10)

在 $\mathbb{R}$ 上:
- 直线耦合 ⟺ monotonic deterministic coupling
- Monotonic coupling 同时最优所有凸 cost
- **唯一** straight coupling

**Proof idea** (Lemma 3.9): 
- 直线 ⟹ ODE 解唯一 ⟹ 不交叉 ⟹ monotonic
- Monotonic ⟹ 不交叉 ⟹ linear interpolation 自身是 ODE 解 ⟹ straight

### 7.4 多维: Straight ≠ Optimal

在 $\mathbb{R}^d, d \geq 2$:
- 不同 $c$ 有不同 optimal coupling
- Straight coupling 不优化特定 $c$
- Straight 是 $c$-optimal 的必要非充分条件

**改进方向** (Section 3.4, ref [42]): 限制 $v$ 为 gradient field $v = \nabla f$，移除 rotational component，可达 quadratic optimal coupling。

---

## 8. 代码实现细节

### 8.1 训练 (Algorithm 2)

```python
def train(data):
    model = init_velocity_network()
    for x0, x1 in data:  # x0~π_0, x1~π_1
        optimizer.zero_grad()
        t = torch.rand(batch_size)  # t ~ Uniform[0,1]
        x_t = t * x1 + (1-t) * x0  # 线性插值
        target = x1 - x0  # 直线方向
        pred = model(x_t, t)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        optimizer.step()
    return model
```

### 8.2 采样 (Algorithm 3)

```python
def sample(model, x0):
    # Forward: 解 ODE 从 t=0 到 t=1
    z = x0
    for t in linspace(0, 1, N):
        z = z + model(z, t) * (1/N)  # Euler step
    return z
```

### 8.3 Reflow (Algorithm 4)

```python
def reflow(data, K):
    coupling = data
    for k in range(K):
        model = train(coupling)
        # 生成新 coupling
        new_coupling = []
        for x0 in data_pi0:
            z1 = sample(model, x0)
            new_coupling.append((x0, z1))
        coupling = new_coupling
    return coupling
```

### 8.4 Distillation

```python
def distill(k_rectified_flow):
    # 学一个 network 直接预测 z1 from z0
    # loss: E[||(Z_1^k - Z_0^k) - v(Z_0^k, 0)||^2]
    # 即 (1) 在 t=0 的情况
    pass
```

---

## 9. 与后续工作的联系

### 9.1 Stable Diffusion 3 / Flow Matching

Rectified flow 是 **Flow Matching** 框架的基础：
- Stable Diffusion 3 (Esser et al. 2024) 采用 rectified flow 作为核心
- Meta Flow Matching (Tenenbaum et al.)
- 改进: 不需要 reflow，直接用更精细的 coupling (OT-based, minibatch OT)

### 9.2 Consistency Models

OpenAI 的 Consistency Models (Song et al. 2023) 可以看作 rectified flow + distillation 的另一路径：
- 直接学习 $T(z_0) = z_1$ 的 consistency function
- 无需显式 reflow

### 9.3 WaveFlow, Voicebox

音频生成中:
- Voicebox (Le et al. 2024) 用 flow matching
- WaveFlow 类似架构

### 9.4 关键 references

- [Original paper](https://arxiv.org/abs/2209.03003)
- [Flow Matching for Generative Modeling (Lipman et al.)](https://arxiv.org/abs/2210.02747) - ICLR 2023 best paper
- [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3)
- [Consistency Models](https://openai.com/research/consistency-models)
- [On Rectified Flow and Optimal Coupling (Liu 2022)](https://www.cs.utexas.edu/~lqiang/rectflow/files/opt-rectflow.pdf)
- [SD3 Technical Report](https://arxiv.org/abs/2403.03206)

---

## 10. 我的 Intuition 总结

1. **Linear interpolation 是关键**: 它是 Euclidean geodesic，保证 transport cost 非增和 reflow 直化。任何偏离都会破坏这些性质。

2. **Reflow 是 self-distillation**: 每次 reflow 让 flow 更直，本质上是把 ODE 的 long-horizon behavior 蒸馏到 short-horizon 可预测性。

3. **ODE > SDE for generative modeling**: 当目标是学 transport map，noise 没必要。SDE 的优势在 modeling time-correlations，不是 marginal distributions。

4. **Straight ≠ Optimal Transport**: 但 straight 足够好（计算上等价 one-step），且更容易达到。

5. **统一框架的力量**: 同一算法解 generative + transfer + adaptation，只需改 $\pi_0$ 的选择。

6. **$\pi_0$ 可以是任意分布**: 这是相比 DDPM 的关键解放——不再需要 $\pi_0$ 是 Gaussian，可以是任何 source domain。

7. **Distillation 是最后一步**: Reflow 直化 + distillation 单步化 = 完整 pipeline。

8. **$O(1/K)$ rate 的实践意义**: 1-2 次 reflow 就足够，不需大量 iteration。

9. **VP ODE 的指数 $\alpha_t$ 是历史包袱**: 来自 OU process derivation，对 ODE 无意义。Linear $\alpha_t = t$ 更好。

10. **Non-crossing 是核心**: ODE 解的唯一性强制 non-crossing，这自动给出更确定的 coupling，减少 transport cost。

---

## 11. 思考与延伸

### 11.1 为什么 linear interpolation 特殊？

它满足:
- Geodesic in Euclidean space
- Constant speed
- $\dot{X}_t = X_1 - X_0$ (constant direction)
- Time-symmetric

任何其他 interpolation 都会破坏至少一个性质。例如:
- Sigmoid interpolation: 非 constant speed
- Spline interpolation: 非 straight
- VP ODE 的 exponential: 非 uniform speed

### 11.2 Reflow 的信息论视角

每次 reflow:
- 输入: $(Z_0^k, Z_1^k)$ 的 joint distribution
- 输出: $(Z_0^{k+1}, Z_1^{k+1})$ 的 joint distribution

$Z_t^{k+1}$ 的 marginal 与 $Z_t^k$ 相同，但 joint 更"确定"（更 deterministic coupling）。

可以看作 entropy reduction 过程:
- $H(Z_1 | Z_0)$ 减少
- $H(Z_1)$ 不变（marginal preserving）

### 11.3 与 Schrödinger Bridge 的关系

Schrödinger Bridge 是 entropy-regularized OT:
- Find coupling $(X_0, X_1)$ minimizing $\mathbb{E}[c(X_1 - X_0)] + \epsilon H(X_1 | X_0)$

Rectified flow:
- 无 explicit regularization
- 但通过 ODE 的 deterministic nature 隐式 minimize $H(Z_1 | Z_0)$
- Reflow 进一步推动到 deterministic limit

### 11.4 与 Optimal Transport 的差距

在 $\mathbb{R}^d, d > 1$:
- Rectified flow 不优化特定 $c$
- 但所有 straight couplings 都计算上等价（one-step）
- 所以从 inference speed 角度，straight 就够了

若要 specific $c$-optimality:
- 限制 $v = \nabla f$ (gradient field)
- 参考 [Liu 2022] "On Rectified Flow and Optimal Coupling"

### 11.5 高维 OT 的 curse

Theorem 3.5 保证 cost 非增，但收敛到 optimal coupling 不保证。这是 high-dim OT 的本质困难。Rectified flow 的聪明之处: 不追求 OT optimality，只追求 straightness，而 straightness 足够实用。

---

## 12. 实践建议

### 12.1 何时用 Rectified Flow?

- 需要快速 inference（one-step 或 few-step）
- 需要 domain transfer（不只 generation）
- 不想 tune diffusion hyperparameters
- 想要 deterministic latent space

### 12.2 何时仍用 DDPM?

- 需要 stochastic sampling（如 SDEdit 的 noise-based editing）
- 需要丰富的 time-correlation（video, music）
- 已有 well-tuned diffusion pipeline

### 12.3 超参数选择

- $\pi_0$: 任意，推荐 $\mathcal{N}(0, I)$ for generation
- $X_t = tX_1 + (1-t)X_0$ (linear interpolation)
- Reflow steps: 1-2 足够
- Distill: 最后一步用 LPIPS loss for $k=1$

### 12.4 架构

- U-Net (DDPM++): 已验证
- Transformer (DiT): Stable Diffusion 3 用此
- 关键: 时间 embedding 和 conditional architecture

---

## 13. 公式速查

| 公式 | 作用 |
|------|------|
| $\min_v \int_0^1 \mathbb{E}[\|(X_1-X_0) - v(X_t,t)\|^2] dt$ | 训练目标 |
| $v^X(x,t) = \mathbb{E}[X_1-X_0 \| X_t=x]$ | 最优解 |
| $S(Z) = \int_0^1 \mathbb{E}[\|(Z_1-Z_0) - \dot{Z}_t\|^2] dt$ | 直度度量 |
| $Z^{k+1} = \text{RectFlow}((Z_0^k, Z_1^k))$ | Reflow |
| $\min_k S(Z^k) \leq \mathbb{E}[\|X_1-X_0\|^2]/K$ | 收敛率 |
| $\text{Law}(Z_t) = \text{Law}(X_t), \forall t$ | Marginal preserving |
| $\mathbb{E}[c(Z_1-Z_0)] \leq \mathbb{E}[c(X_1-X_0)]$ | Cost 非增 |

---

## 14. 开放问题

1. **更好的 coupling**: 能否直接构造 near-optimal coupling 而不用 reflow?
2. **Non-Euclidean settings**: Riemannian manifold, discrete spaces
3. **Conditional generation**: class-conditional, text-conditional 的最优 integration
4. **Multi-modal couplings**: 超过两个分布的 chain
5. **理论 tighter bound**: $O(1/K)$ 是否 optimal? 能否 $O(1/K^2)$?
6. **与 score matching 的统一**: 是否本质等价?
7. **Architectural innovations**: 专门为 flow 设计的架构

---

## 15. 进一步阅读

- [Flow Matching paper](https://arxiv.org/abs/2210.02747)
- [Stable Diffusion 3 paper](https://arxiv.org/abs/2403.03206)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [DDPM original](https://arxiv.org/abs/2006.11239)
- [Score-based SDE](https://arxiv.org/abs/2011.13456)
- [DDIM](https://arxiv.org/abs/2010.02502)
- [Neural ODE](https://arxiv.org/abs/1806.07366)
- [Original Rectified Flow code](https://github.com/gnobitab/RectifiedFlow)
- [Qiang Liu's group page](https://www.cs.utexas.edu/~lqiang/)

---

这篇 paper 的 elegance 在于: 它把 diffusion 的成功去神秘化，指出核心是"稳定的 regression-based 训练"+"ODE formulation"，而不是 stochastic noise。Rectified flow 用最简单的 linear interpolation + least squares，达到了 SDE 方法的性能，同时获得 one-step inference 的能力。这种**简化并改进**的思路，是优秀 research 的典范。
