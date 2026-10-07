---
source_pdf: Flow Straight and Fast.pdf
paper_sha256: d81790cb6988ef480002adcc84000d4f44855b11fb868001fd53aad133bc3a74
processed_at: '2026-08-04T09:39:04-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Rectified Flow 人话版

Andrej，咱们换个画风，用最直觉的方式把这篇 paper 捏碎。

---

## 一句话总结

**找一条最直的路，从 noise 走到 data。直到一步到位，不用像 diffusion 那样磨叽 1000 步。**

---

## 1. 这 paper 到底在干啥

想象你在北京（noise），要去上海（data）。

- **Diffusion model** 的走法：先从上海慢慢往北京走，记录一路上每个路口往哪拐（学 reverse process）。生成时反过来，从北京出发，照着记录的拐弯方向，一步一步挪到上海。1000 步，累死。

- **Rectified Flow** 的走法：直接画条直线。直线最短，一步蹦过去。

问题是——你怎么知道这条直线对不对？你怎么学？

---

## 2. 核心招式：Linear Interpolation + Regression

### 2.1 画路

给定 $x_0 \sim \pi_0$（noise）和 $x_1 \sim \pi_1$（real image），画一条直线连起来：

$$X_t = t \cdot x_1 + (1-t) \cdot x_0, \quad t \in [0,1]$$

- $t=0$：你在北京（noise $x_0$）
- $t=1$：你在上海（data $x_1$）
- $t=0.5$：你在济南（中点）

这条直线的"速度"是恒定的：$\dot{X}_t = x_1 - x_0$（始终指向上海）。

### 2.2 问题来了

这个 $\dot{X}_t = x_1 - x_0$ 是 **作弊**——它需要提前知道终点 $x_1$。现实中你站在济南，不知道上海在哪。

### 2.3 解决：学一个 navigator

训一个 neural network $v_\theta$，给它当前位置和时间，让它预测"往哪走"：

$$\min_\theta \mathbb{E}\left[\|v_\theta(X_t, t) - (x_1 - x_0)\|^2\right]$$

- **Input**：当前位置 $X_t$、当前时间 $t$
- **Target**：真实方向 $(x_1 - x_0)$
- **Loss**：MSE，plain regression

就这么简单。**没有 GAN 的 minimax，没有 ELBO，没有 score matching 的弯弯绕**。就是一个 regression。

### 2.4 最优解是啥

$$v^*(x, t) = \mathbb{E}[x_1 - x_0 \mid X_t = x]$$

当前位置 $x$ 可能被很多条直线穿过（因为 $x_0, x_1$ 是随机配对的），取这些方向的**平均**。就像在十字路口，看到四面八方的路标，取个平均方向继续走。

---

## 3. 为什么"直"这么重要

### 3.1 直线 = 一步到位

如果 trajectory 是完美的直线，velocity $v$ 沿路径恒定。Euler 一步：

$$z_1 = z_0 + v(z_0, 0) \cdot 1$$

直接到终点。**1 NFE**。

### 3.2 弯路 = 多步

Diffusion / PF-ODE 的路径是弯的（Figure 4, 5）。弯曲的原因：
- VP ODE 用 exponential schedule $\alpha_t = \exp(-\frac{1}{4}a(1-t)^2 - \frac{1}{2}b(1-t))$
- 前半段 $t < 0.5$ 几乎不动，后半段突然加速
- 弯曲 + non-uniform speed = 大 step size 时炸掉

### 3.3 实验对比

CIFAR10，one-step generation：

| Method | FID |
|--------|-----|
| VP ODE + distill | 16.23 |
| sub-VP ODE + distill | 14.32 |
| **2-Rectified Flow + distill** | **4.85** |

差 3 倍。这就是"直"的力量。

---

## 4. Reflow：怎么让路径变直

### 4.1 第一次的路径还不够直

第一次训练（1-Rectified Flow），用的是 $(x_0, x_1) \sim \pi_0 \times \pi_1$——**随机配对**。

随机配对意味着：同一个 noise $x_0$ 可能对应很多不同的 $x_1$，路径会交叉（Figure 2a）。交叉 = 弯路。

### 4.2 Reflow 的魔法

Step 1：用 1-RF 生成一堆 pairs $(z_0, z_1)$——从 noise 出发，走 ODE 到 data。

Step 2：用这些 $(z_0, z_1)$ 重新训练——这次配对不再是随机的，而是 **ODE 确定出来的**。

为什么变直了？因为 ODE 是 deterministic 的——给定 $z_0$，$z_1$ 唯一确定。确定性 $\Rightarrow$ 路径不交叉（non-crossing property）$\Rightarrow$ 更直。

### 4.3 数学保证

Straightness measure：

$$S(Z) = \int_0^1 \mathbb{E}\left[\|(Z_1 - Z_0) - \dot{Z}_t\|^2\right] dt$$

- $Z_1 - Z_0$：理想方向（直线方向）
- $\dot{Z}_t$：实际 velocity
- $S=0$：完美直线

Theorem 3.7：

$$\min_{k \leq K} S(Z^k) \leq \frac{\mathbb{E}[\|x_1 - x_0\|^2]}{K}$$

$O(1/K)$ 收敛。实践上 **1 次 reflow 就够了**（2-RF），再多会累积估计误差。

### 4.4 Burgers' equation 的彩蛋

如果 flow 完美 straight，velocity 满足 **inviscid Burgers' equation**：

$$\partial_t v + (\partial_z v) v = 0$$

- $\partial_t v$：velocity 对时间的偏导
- $\partial_z v$：velocity 对空间的 Jacobian
- $v$：velocity field 本身

这是流体力学经典方程。物理直觉：直线路径要求 velocity 沿轨迹不变，即 $\frac{d}{dt}v(Z_t, t) = 0$，展开就是 Burgers。

---

## 5. 关键理论：三个保证

### 5.1 Marginal Preserving（Theorem 3.3）

$$\text{Law}(Z_t) = \text{Law}(X_t), \quad \forall t$$

意思：ODE 走出来的 $Z_t$，在每一个时刻 $t$，分布和 linear interpolation $X_t$ 一样。

**直觉**：velocity field $v^X$ 的定义保证了"流入流出质量平衡"。$X_t$ 和 $Z_t$ 满足同一个 continuity equation：

$$\dot{\pi}_t + \nabla \cdot (v_t \pi_t) = 0$$

- $\pi_t$：$t$ 时刻的 density
- $\dot{\pi}_t$：density 随时间变化
- $\nabla \cdot (v_t \pi_t)$：flux 的 divergence

同一个方程，同一个初始条件 $\to$ 同一个解。

但注意：**marginal 相同，joint 不同**。$X_t$ 是 stochastic、non-causal 的；$Z_t$ 是 deterministic、causal 的。Rectify 做了 causalize + derandomize。

### 5.2 Cost Reduction（Theorem 3.5）

$$\mathbb{E}[c(Z_1 - Z_0)] \leq \mathbb{E}[c(X_1 - X_0)], \quad \forall \text{ convex } c$$

**直觉**：直线段连线长度 = $\|x_1 - x_0\|$。Rectified flow 是这些直线的"rewiring"——在交叉点把路径重新接，避免绕路。三角形不等式保证总长度不增。

对任意 convex cost $c$（不只是 L2），都成立。这是 **Pareto descent**——不针对特定 $c$，但所有 convex cost 都不增。

### 5.3 Straightening（Theorem 3.7）

前面讲过了，$O(1/K)$。

---

## 6. 和 Diffusion 的关系：原来是亲戚

### 6.1 PF-ODE 是 Rectified Flow 的特例

PF-ODE（Song et al. 2021）用：

$$X_t = \alpha_t X_1 + \beta_t \xi, \quad \xi \sim \mathcal{N}(0, I)$$

- $\alpha_t = \exp(-\frac{1}{4}a(1-t)^2 - \frac{1}{2}b(1-t))$：VP/sub-VP 的 schedule
- $\beta_t$：见 equation 8

Rectified Flow 框架说：这就是 nonlinear rectified flow 的一个特例，用了特定的 $\alpha_t, \beta_t$。

**但这个选择不好**：
1. $\beta_t \neq 1 - \alpha_t$ $\to$ 路径弯曲，reflow 没用
2. Exponential schedule $\to$ non-uniform speed，前慢后快

### 6.2 Rectified Flow 的选择

直接用 $\alpha_t = t$, $\beta_t = 1 - t$：

$$X_t = t X_1 + (1-t) X_0$$

- Straight paths
- Uniform speed
- $\pi_0$ 任意（不强制 Gaussian）

### 6.3 Diffusion noise 是多余的吗

Paper 末尾有个大胆论断：**diffusion model 成功的关键是 stable optimization procedure，不是 diffusion noise 本身**。

证据：
- Rectified Flow 去掉 noise，效果一样好甚至更好
- SDE 的 training loss 和 ODE 的只差一个 reparameterization
- SDE 的 marginal 可以被 ODE 完美复现（probability flow ODE）

Diffusion noise 的"功劳"被高估了。真正 work 的是 **regression-based training** 的稳定性。

---

## 7. Image-to-Image Translation：同一个招式

### 7.1 设定

$\pi_0$ = human faces，$\pi_1$ = cat faces。没有配对数据。

### 7.2 做法

完全一样的算法！只是 $\pi_0$ 从 Gaussian 换成 human faces。

### 7.3 Style-aware loss

为了保留 identity 只改 style，用 classifier feature 加权：

$$\min_v \int_0^1 \mathbb{E}\left[\|\nabla h(X_t)^\top (x_1 - x_0 - v(X_t, t))\|^2\right] dt$$

- $h(x)$：classifier 的 latent（区分两个 domain）
- $\nabla h(X_t)$：saliency map，哪些像素/特征对 style 重要
- 效果：loss 聚焦在 style-relevant 的方向上

### 7.4 结果

2-RF 用 **single Euler step** 就能做人脸 $\to$ 猫脸（Figure 14）。CycleGAN 需要训练两个 GAN + cycle consistency loss，Rectified Flow 一个 ODE 搞定，还有 reversibility 自带 cycle consistency。

---

## 8. 实操 cheat sheet

### 8.1 训练

```python
for x0, x1 in dataloader:  # x0~π₀, x1~π₁
    t = torch.rand(B)
    x_t = t * x1 + (1-t) * x0
    loss = MSE(model(x_t, t), x1 - x0)
    loss.backward()
    optimizer.step()
```

### 8.2 Reflow

```python
# 1. 用 1-RF 生成 4M pairs
pairs = []
for x0 in data:
    z = odeint(model, x0, [0, 1])  # forward
    pairs.append((x0, z))

# 2. 用 pairs fine-tune → 2-RF
for z0, z1 in pairs:
    t = torch.rand(B)
    z_t = t * z1 + (1-t) * z0
    loss = MSE(model(z_t, t), z1 - z0)
    loss.backward()
    optimizer.step()
```

### 8.3 Distillation（one-step）

```python
# Fine-tune for N=1
for z0, z1 in pairs:
    loss = LPIPS(z0 + model(z0, 0), z1)  # 用 LPIPS 比 L2 好
    loss.backward()
```

### 8.4 超参数（CIFAR10）

- Architecture: DDPM++ U-Net
- Optimizer: Adam, lr=2e-4, dropout=0.15
- EMA: 0.999999
- Reflow: 4M pairs, 300K steps
- Distillation: LPIPS loss

---

## 9. Intuition 深挖

### 9.1 为什么 reflow 能 straighten——另一种理解

第一次：$(x_0, x_1)$ 随机配对。想象 1000 个北京人随机去 1000 个上海地址，路径乱七八糟，到处交叉。

Rectify 后：ODE 把交叉的路径"理顺"了（non-crossing property）。现在 $(z_0, z_1)$ 的配对更有规律——附近的 noise 去附近的 data。

第二次 reflow：用这个更有规律的配对重新画直线，交叉更少，更直。

类比：**梳头**。第一次梳，头发从乱到有点顺。第二次梳，更顺。第三次，几乎完美直。

### 9.2 为什么 non-crossing = lower cost

两条直线交叉，意味着"绕路"。A 从左上到右下，B 从右上到左下，如果交叉，A 和 B 都走了多余的路。

Rectify 在交叉点"rewire"：A 直接走左上到右下的直线，B 直接走右上到左下。总路程变短。

数学上就是 triangle inequality / Jensen's inequality。

### 9.3 ODE 的 deterministic 性质是关键

$(z_0, z_1)$ 是 deterministic coupling——给定 $z_0$，$z_1$ 唯一。

对比：
- Independent coupling $(x_0, x_1) \sim \pi_0 \times \pi_1$：一个 $x_0$ 可能对应很多 $x_1$
- Rectified coupling：一个 $z_0$ 对应一个 $z_1$

确定性 = 更少的路径交叉 = 更直 = 更快。

### 9.4 和 Optimal Transport 的区别

OT 找特定 cost $c$ 的最优 coupling。Rectified Flow 找 **straight** coupling。

- 1D：straight = monotone = OT optimal（所有 convex cost 同时最优）
- 高维：straight 是 c-optimal 的必要条件，但不充分

Straight coupling 不唯一（可以旋转），OT optimal 是其中一个。但对 fast inference 来说，**所有 straight coupling 都一样好**——都能一步模拟。

---

## 10. 代码实现要点

### 10.1 核心模型

```python
class RectifiedFlow(nn.Module):
    def __init__(self, velocity_net):
        self.net = velocity_net  # U-Net, (x, t) -> v
    
    def velocity(self, x, t):
        return self.net(x, t)
    
    def sample(self, x0, N=1):
        dt = 1.0 / N
        z = x0
        for i in range(N):
            t = i * dt
            z = z + self.velocity(z, t) * dt  # Euler
        return z
```

### 10.2 Training loss

```python
def rectified_flow_loss(model, x0, x1):
    B = x0.shape[0]
    t = torch.rand(B, 1, 1, 1)  # per-sample time
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = model(x_t, t.squeeze())
    return F.mse_loss(pred, target)
```

### 10.3 Euler sampling

```python
def euler_sample(model, x0, N=1):
    z = x0
    dt = 1.0 / N
    for i in range(N):
        t = torch.full((B,), i * dt)
        z = z + model(z, t) * dt
    return z
```

---

## 11. 后续影响

这篇 paper 是 **Flow Matching** 的奠基工作之一（concurrent with Meta 的 Flow Matching paper）。

后续：
- **Stable Diffusion 3**：用了 Rectified Flow 的 schedule
- **Flux**（Black Forest Labs）：基于 flow matching
- **Consistency Models**（OpenAI）：受 reflow + distillation 启发
- **SDXL Turbo**：distillation 思路类似

**参考链接**：
- 原文 arXiv: https://arxiv.org/abs/2209.03003
- Qiang Liu on Rectified Flow & OT: https://www.cs.utexas.edu/~lqiang/
- Meta Flow Matching: https://arxiv.org/abs/2210.02747
- Stable Diffusion 3 Technical Report: https://arxiv.org/abs/2403.03206
- OpenAI Consistency Models: https://openai.com/research/consistency-models
- SDXL Turbo: https://stability.ai/news/stability-ai-sdxl-turbo

---

## 12. 一图胜千言

```
Diffusion / PF-ODE:
noise ----→ curve ----→ curve ----→ data
(1000 steps, 弯弯曲曲)

1-Rectified Flow:
noise ----→ straight-ish ----→ data
(2-5 steps, 基本直)

2-Rectified Flow + distill:
noise -----------------------→ data
(1 step, 完美直线)
```

核心 message：**直 = 快。Reflow = 梳直。Distill = 锁定一步到位。**

---

## 13. 为什么这篇 paper great

1. **Simplicity**：一个 MSE regression，统一 generative + transfer
2. **Theory**：三个定理（marginal preserving, cost reduction, straightening）都有 clean proof
3. **Insight**：揭示了 diffusion noise 是多余的，真正 work 的是 stable optimization
4. **Practical**：one-step generation SOTA（CIFAR10 FID 4.85）
5. **Generality**：同一个算法做 generation、translation、domain adaptation

这是那种"**看完觉得 obvious，但只有作者做出来**"的 paper。Great research 就是把复杂问题简化到本质。

---

# Rectified Flow: Flow Straight and Fast 深度解析

这篇 paper 是 Qiang Liu 团队（UT Austin）的工作，提出了 **Rectified Flow** —— 一种极其简洁的 ODE-based 方法，统一了 generative modeling 和 domain transfer。核心 insight 极其优雅：**学习一个 ODE 去尽可能沿着连接 π₀ 和 π₁ 样本点的直线运动**。

---

## 1. 核心动机：为什么"直"很重要

### 1.1 Transport Mapping Problem 的统一视角

给定两个经验分布 $X_0 \sim \pi_0$, $X_1 \sim \pi_1$ on $\mathbb{R}^d$，寻找 transport map $T: \mathbb{R}^d \to \mathbb{R}^d$，使得 $Z_1 = T(Z_0) \sim \pi_1$ when $Z_0 \sim \pi_0$。

- **Generative modeling**：$\pi_0 = \mathcal{N}(0, I)$（Gaussian noise），$\pi_1 = $ data distribution
- **Domain transfer**：$\pi_0, \pi_1$ 都是 empirical data distributions（如 human faces → cat faces）

这两类问题在 Rectified Flow 框架下被完全统一——只需切换 $\pi_0$ 的设定。

### 1.2 现有方法的痛点

| Method | Pain Point |
|--------|-----------|
| GAN | minimax instability, mode collapse, heavy tuning |
| VAE / Normalizing Flow | intractable likelihood, architecture constraints (invertibility) |
| DDPM / Score SDE | inference 极慢（需要 1000+ NFE），design space 复杂 |
| Neural ODE (MLE) | 需要 backprop through time, gradient vanishing/exploding |
| PF-ODE / DDIM | 继承了 SDE 推导的复杂性，路径是 curved 的，non-uniform speed |

Rectified Flow 的突破口：**直接学 ODE，跳过 SDE，让路径变直**。直线路径意味着单个 Euler step 就能精确模拟，inference cost 大幅下降。

---

## 2. 方法详解

### 2.1 Linear Interpolation 与 Causalization

**Step 1：构造 linear interpolation**

$$X_t = t X_1 + (1 - t) X_0, \quad t \in [0, 1]$$

- $X_0 \sim \pi_0$：起点
- $X_1 \sim \pi_1$：终点
- $t$：时间参数，从 0 到 1
- $X_t$：在 $X_0$ 和 $X_1$ 之间的线性插值

这个 interpolation 满足 ODE：$\mathrm{d}X_t = (X_1 - X_0)\mathrm{d}t$，但这是 **non-causal**（anticipating）的——因为更新 $X_t$ 需要知道终点 $X_1$ 的信息。在实际 inference 时，我们看不到未来。

**Step 2：Causalize —— 学一个 velocity field $v$ 去近似 $(X_1 - X_0)$**

$$\min_v \int_0^1 \mathbb{E}\left[\left\|(X_1 - X_0) - v(X_t, t)\right\|^2\right] \mathrm{d}t \tag{1}$$

- $v: \mathbb{R}^d \to \mathbb{R}^d$：待学习的 velocity field（通常用 neural network 参数化为 $v_\theta$）
- $X_t = tX_1 + (1-t)X_0$：covariate
- 目标：让 $v(X_t, t)$ 尽量接近 "true velocity" $(X_1 - X_0)$

**最优解**（Theorem 的核心）：

$$v^X(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x] \tag{2}$$

这是所有穿过 $x$（在时刻 $t$）的直线方向的**条件期望**。直觉上：在位置 $x$ 和时间 $t$，有多条 linear interpolation 路径穿过，我们取它们方向的平均。

### 2.2 训练算法（Algorithm 1, 2）

PyTorch-style 伪代码：

```python
# Algorithm 2: Train
for x0, x1 in DataLoader:  # x0 ~ π₀, x1 ~ π₁
    t = torch.rand(batch_size)  # t ~ Uniform[0,1]
    x_t = t * x1 + (1 - t) * x0  # linear interpolation
    v_pred = model(x_t, t)  # neural net forward
    loss = (v_pred - (x1 - x0)).pow(2).mean()  # MSE
    loss.backward()
    optimizer.step()
```

极其简单——就是一个 regression，target 是 $(x_1 - x_0)$。**没有 minimax，没有 variational bound，没有 score matching 的复杂性**。

### 2.3 Sampling

训练好 $v_{\hat\theta}$ 后，求解 ODE：

$$\mathrm{d}Z_t = v_{\hat\theta}(Z_t, t) \mathrm{d}t, \quad Z_0 \sim \pi_0$$

- **Forward**：$Z_0 \sim \pi_0 \to$ 求解 ODE 到 $t=1$ $\to$ 得到 $Z_1 \sim \pi_1$
- **Backward**：从 $Z_1 \sim \pi_1$ 出发，求解 $\mathrm{d}\tilde{X}_t = -v(\tilde{X}_t, t)\mathrm{d}t$，然后 $X_t = \tilde{X}_{1-t}$

时间对称性：objective (1) 在交换 $X_0 \leftrightarrow X_1$ 并 flip $v$ 的 sign 后等价，所以 forward/backward 同等 favored。

---

## 3. 核心理论性质

### 3.1 Marginal Preserving（Theorem 3.3）

$$\text{Law}(Z_t) = \text{Law}(X_t), \quad \forall t \in [0, 1]$$

**直觉**：$v^X$ 的定义保证了在每一个 location 和 time，进入和离开的 "mass" 在 $X_t$ 和 $Z_t$ 动力学下相等。用 continuity equation 来看：

$$\dot{\pi}_t + \nabla \cdot (v_t^X \pi_t) = 0 \tag{11}$$

- $\pi_t = \text{Law}(X_t)$：$X_t$ 的 marginal density
- $\dot{\pi}_t$：$\pi_t$ 对 $t$ 的偏导
- $\nabla \cdot$：divergence operator
- $v_t^X = v^X(\cdot, t)$：velocity field

$X_t$ 和 $Z_t$ 都满足同一个 continuity equation，同一个初始条件，所以 marginal 相同（需要解的唯一性，即 $v^X$ Lipschitz）。

**关键区分**：marginal 相同，但 **joint distribution 不同**。$X_t$ 是 non-causal、non-Markov 的 stochastic process；$Z_t$ 是 causal、Markov、deterministic 的。Rectified flow 做的是 **causalize + Markovianize + derandomize**，同时保持所有 marginal。

### 3.2 Non-crossing Property

ODE 的解唯一 $\Rightarrow$ 不同 trajectory 不能在 $t \in [0,1)$ 时刻交叉。如果在 location $z$、time $t$ 有两条 path 沿不同方向穿过，ODE 解就不唯一了。

但 linear interpolation $X_t$ 的路径**可以交叉**（Figure 2a）。Rectified flow 在交叉点"rewire"路径（Figure 2b），避免交叉，同时保持相同的 density map。

**这个 rewire 正是 transport cost 降低的来源**。

### 3.3 减少凸传输成本（Theorem 3.5）

$$\mathbb{E}[c(Z_1 - Z_0)] \leq \mathbb{E}[c(X_1 - X_0)], \quad \forall \text{ convex } c: \mathbb{R}^d \to \mathbb{R}$$

- $c$：任意 convex cost function（如 $c(\cdot) = \|\cdot\|^\alpha$, $\alpha \geq 1$）
- $(Z_0, Z_1) = \text{Rectify}((X_0, X_1))$：rectified coupling

**证明直觉**（以 $c(\cdot) = \|\cdot\|$ 为例）：

$$\mathbb{E}[\|Z_0 - Z_1\|] = \text{Length}(\text{trajectory of } Z_t) \leq \text{Length}(\text{trajectory of } X_t) = \mathbb{E}[\|X_0 - X_1\|]$$

- 第一个 $=$：$Z_t$ 是 deterministic ODE，路径长度 $= \|Z_1 - Z_0\|$
- $\leq$：triangle inequality（$Z_t$ 的路径是 $X_t$ 路径的 rewiring，去掉了交叉导致的"绕路"）
- 第二个 $=$：$X_t$ 是直线，长度正好 $= \|X_1 - X_0\|$

对一般 convex $c$，用 **Jensen's inequality** 两次：

1. $c(\mathbb{E}[\cdot]) \leq \mathbb{E}[c(\cdot)]$（conditional expectation）
2. 凸性 + 积分

**重要意义**：Rectify 是对**所有** convex cost 的 Pareto descent，不针对任何特定 $c$。这区别于 traditional optimal transport（显式优化某个 $c$）。

### 3.4 Straightening via Reflow（Theorem 3.7）

定义 straightness measure：

$$S(Z) = \int_0^1 \mathbb{E}\left[\left\|(Z_1 - Z_0) - \dot{Z}_t\right\|^2\right] \mathrm{d}t \tag{3}$$

- $Z_1 - Z_0$：端点连线方向（理想 velocity）
- $\dot{Z}_t$：实际 velocity at time $t$
- $S(Z) = 0$：perfectly straight（每条 path 上 velocity 恒定 = $Z_1 - Z_0$）

**Reflow 操作**：

$$Z^{k+1} = \text{RectFlow}((Z_0^k, Z_1^k)), \quad (Z_0^0, Z_1^0) = (X_0, X_1)$$

递归应用 rectification，路径越来越直。

**收敛速率**（Theorem 3.7）：

$$\min_{k \in \{0, \ldots, K\}} S(Z^k) \leq \frac{\mathbb{E}[\|X_1 - X_0\|^2]}{K}$$

- $K$：reflow 次数
- 收敛速率 $O(1/K)$

**证明 telescoping**：取 $c(x) = \|x\|^2$，每次 rectification 满足：

$$\mathbb{E}[\|Z_1^k - Z_0^k\|^2] - \mathbb{E}[\|Z_1^{k+1} - Z_0^{k+1}\|^2] = S(Z^{k+1}) + V((Z_0^k, Z_1^k)) \tag{13}$$

- 左边：transport cost 的减少量
- $V((Z_0^k, Z_1^k))$：crossing measure（见 (12)），衡量路径交叉程度

$$V((X_0, X_1)) = \int_0^1 \mathbb{E}\left[\|X_1 - X_0 - \mathbb{E}[X_1 - X_0 \mid X_t]\|^2\right] \mathrm{d}t$$

$V = 0$ 意味着穿过每个 $X_t$ 的直线唯一，即无交叉。

telescoping sum $k=0,\ldots,K$：

$$\sum_{k=0}^K \left[S(Z^{k+1}) + V((Z_0^k, Z_1^k))\right] = \mathbb{E}[\|X_1 - X_0\|^2] - \mathbb{E}[\|Z_1^{K+1} - Z_0^{K+1}\|^2] \leq \mathbb{E}[\|X_1 - X_0\|^2]$$

所以 $\min_k S(Z^k) \leq \mathbb{E}[\|X_1 - X_0\|^2]/K$。

**实操建议**：paper 中实践显示 1 次 reflow 已经足够（2-rectified flow），更多 reflow 会累积 $v^X$ 的估计误差。

### 3.5 Straight vs. Optimal Coupling（Section 3.4）

- **Straight coupling**：fixed point of Rectify(·)，路径不交叉
- **c-optimal coupling**：minimize $\mathbb{E}[c(Z_1 - Z_0)]$

**Theorem 3.8**：c-optimal（strictly convex $c$）$\Rightarrow$ straight。但 reverse 不成立（除非 $d=1$）。

**1D case（Theorem 3.10）**：在 $\mathbb{R}$ 上，straight = monotonic = deterministic，且**同时**对所有 convex $c$ optimal。这是 1D optimal transport 的经典结果（monotone coupling）。

**多维 case**：straight coupling 存在但不唯一，也不一定 c-optimal。要找 c-optimal，需要 restrict $v$ 为 gradient field $v(x,t) = \nabla f(x,t)$（见 [42]）。这去掉了 $v^X$ 的 rotational component。

---

## 4. 与 PF-ODEs / DDIM 的统一（Section 2.3, 3.5）

### 4.1 Nonlinear Extension

将 linear interpolation 推广为任意 time-differentiable curve：

$$X_t = \alpha_t X_1 + \beta_t X_0$$

- $\alpha_t, \beta_t$：可微序列，满足 $\alpha_1 = \beta_0 = 1$, $\alpha_0 = \beta_1 = 0$（边界条件）
- $\dot{X}_t = \dot{\alpha}_t X_1 + \dot{\beta}_t X_0$

训练目标：

$$\min_v \int_0^1 \mathbb{E}\left[w_t \|v(X_t, t) - \dot{X}_t\|^2\right] \mathrm{d}t \tag{6}$$

- $w_t$：positive weighting（default $w_t = 1$）

**Marginal preserving 仍然成立**（Theorem 3.3 对任意 interpolation 都成立）。但：
- Transport cost 不再保证下降
- Reflow 不再 straighten

### 4.2 PF-ODEs 是特例（Proposition 3.11）

**VP ODE / sub-VP ODE**（Song et al. 2021）使用：

$$\alpha_t = \exp\left(-\frac{1}{4}a(1-t)^2 - \frac{1}{2}b(1-t)\right), \quad \text{default: } a=19.9, b=0.1 \tag{7}$$

$$\text{VP ODE: } \beta_t = \sqrt{1 - \alpha_t^2}; \quad \text{sub-VP ODE: } \beta_t = 1 - \alpha_t^2 \tag{8}$$

**VE ODE**：$\alpha_t = 1$, $\beta_t = \sigma_{\min}\sqrt{r^{2(1-t)} - 1}$

这些都可写为 $X_t = \alpha_t X_1 + \beta_t \xi$（$\xi \sim \mathcal{N}(0, I)$）的形式。

**PF-ODEs 的问题**（Figure 4, 5, 6）：

1. **Non-straight paths**：$\beta_t \neq 1 - \alpha_t$，所以路径是 curved 的，reflow 无法 straighten
2. **Non-uniform speed**：$\alpha_t$ 是 exponential 形式，early phase ($t \lesssim 0.5$) 变化慢，late phase 集中更新。这导致大 step size 时表现差
3. **不必要的 SDE 假设**：$\pi_0$ 必须是 spherical Gaussian，$\xi$ 必须是 Gaussian——这些是 SDE 推导的产物，ODE 视角下完全没必要

**Rectified Flow 的改进**：直接用 linear interpolation $X_t = tX_1 + (1-t)X_0$，uniform speed，straight paths，$\pi_0$ 任意。

### 4.3 DDPM 训练 loss 的等价性

DDPM / score SDE 的训练 loss（equation 15）：

$$\min_v \int_0^1 \mathbb{E}\left[w_t \|v(V_t, t) - Y_t\|_2^2\right] \mathrm{d}t, \quad V_t = \alpha_t X_1 + \beta_t \xi_t, \quad Y_t = -\eta_t V_t - \frac{\sigma_t^2}{\beta_t}\xi_t \tag{15}$$

PF-ODE 的 loss（equation 18）只差一个 $1/2$ factor：

$$\tilde{Y}_t = -\eta_t V_t - \frac{\sigma_t^2}{2\beta_t}\xi_t$$

**Proposition 3.11 证明**：利用 (16) 的关系 $\eta_t = -\dot{\alpha}_t/\alpha_t$ 和 $\sigma_t^2 = 2\beta_t^2(\dot{\alpha}_t/\alpha_t - \dot{\beta}_t/\beta_t)$，可以验证 $\tilde{Y}_t = \dot{X}_t$。所以 PF-ODE loss = nonlinear rectified flow loss with $X_t = \alpha_t X_1 + \beta_t \xi$。

---

## 5. Velocity Field 的解析形式（Section 2.2）

### 5.1 条件期望表示（equation 4）

如果 $X_0 \mid X_1 = x_1$ 有 conditional density $\rho(x_0 \mid x_1)$：

$$v^X(z, t) = \mathbb{E}\left[\frac{X_1 - z}{1 - t} \eta_t(X_1, z)\right]$$

$$\eta_t(X_1, z) = \frac{\rho\left(\frac{z - tX_1}{1 - t} \mid X_1\right)}{\mathbb{E}\left[\rho\left(\frac{z - tX_1}{1 - t} \mid X_1\right)\right]}$$

- $\eta_t(X_1, z)$：posterior weight，给定 $X_t = z$ 时 $X_1$ 的后验
- $\frac{z - tX_1}{1 - t}$：从 $X_t = z$ 反推 $X_0$（因为 $z = tX_1 + (1-t)X_0$）
- $\frac{X_1 - z}{1 - t}$：从 $z$ 到 $X_1$ 的方向（scaled by $1/(1-t)$）

**梯度**：

$$\nabla_z v^X(z, t) = \frac{1}{1 - t}\mathbb{E}\left[((X_1 - z)\nabla_z \log \eta_t(X_1, z) - 1)\eta_t(X_1, z)\right]$$

如果 $\nabla_z \log \eta_t$ 连续，$v^X$ Lipschitz $\Rightarrow$ ODE 解唯一。

### 5.2 Non-parametric estimator（equation 5）

低维情况下用 Nadaraya-Watson style estimator：

$$v^{X,h}(z, t) = \mathbb{E}\left[\frac{X_1 - z}{1 - t} \omega_h(X_t, z)\right]$$

$$\omega_h(X_t, z) = \frac{\kappa_h(X_t, z)}{\mathbb{E}[\kappa_h(X_t, z)]}$$

- $\kappa_h(x, z) = \exp(-\|x - z\|^2 / 2h^2)$：Gaussian RBF kernel
- $h$：bandwidth，$h \to 0^+$ 时收敛到 $v^X$

实践中用 kNN 近似：

$$v^{X,h}(z, t) \approx \sum_{i \in \text{knn}(z, m)} \frac{x_1^{(i)} - z}{1 - t} \omega_h(x_t^{(i)}, z)$$

---

## 6. 实验结果详解

### 6.1 CIFAR10 Unconditioned Generation（Table 1a）

| Method | NFE | IS ↑ | FID ↓ | Recall ↑ |
|--------|-----|------|-------|----------|
| 1-Rectified Flow (+Distill) | 1 | 9.08 | 6.18 | 0.45 |
| 2-Rectified Flow (+Distill) | 1 | 9.01 | **4.85** | 0.50 |
| 3-Rectified Flow (+Distill) | 1 | 8.79 | 5.21 | **0.51** |
| VP ODE (+Distill) | 1 | 8.73 | 16.23 | 0.29 |
| sub-VP ODE (+Distill) | 1 | 8.80 | 14.32 | 0.35 |
| 1-Rectified Flow (RK45) | 127 | 9.60 | **2.58** | **0.57** |
| VP ODE (RK45) | 140 | 9.37 | 3.93 | 0.51 |
| sub-VP ODE (RK45) | 146 | 9.46 | 3.16 | 0.55 |
| VP SDE (N=2000) | 2000 | 9.58 | 2.55 | 0.58 |

**关键观察**：

1. **One-step generation**：2-Rectified Flow + Distill 达到 FID 4.85，远超 VP ODE distill 的 16.23。Recall 0.50 也超过 StyleGAN2+ADA 的 0.49
2. **Full simulation**：1-Rectified Flow 用 127 NFE 达到 FID 2.58，比 VP SDE 的 2000 NFE 还略好（FID 2.55），但 NFE 减少 15×
3. **Reflow 的 trade-off**：full simulation 时 1-RF > 2-RF > 3-RF（误差累积），但 one-step 时 2-RF > 1-RF（straightening 效果主导）

### 6.2 Few-step regime（Figure 8a）

在 $N \leq 80$ 的 small step regime，reflow 大幅改善 FID 和 recall。但 $N > 80$ 时，1-RF 反而更好——因为 reflow 引入了 $v^X$ 的估计误差。

### 6.3 Straightening 可视化（Figure 9, 10）

定义 extrapolation：

$$\hat{z}_1^t = z_t + (1 - t) v(z_t, t)$$

- 如果 trajectory 是直线，$\hat{z}_1^t$ 应该与 $t$ 无关（恒等于 $Z_1$）
- 1-RF：$\hat{z}_1^t$ 随 $t$ 变化，但 $t \approx 0.1$ 时已经可识别
- 2-RF：$\hat{z}_1^t$ 几乎不随 $t$ 变化——**几乎完美直线**
- sub-VP ODE：需要 $t \approx 0.6$ 才能识别

### 6.4 Image-to-Image Translation（Section 5.3）

**Loss 设计**（equation 20）——style-aware variant：

$$\min_v \int_0^1 \mathbb{E}\left[\left\|\nabla h(X_t)^\top (X_1 - X_0 - v(X_t, t))\right\|_2^2\right] \mathrm{d}t$$

- $h(x)$：classifier 的 latent representation（区分两个 domain）
- $\nabla h(X_t)$：saliency weight，re-weight coordinates 使 loss 聚焦于 style-relevant 的变化
- 这样 transfer 时保留 identity，只改 style

实验在 AFHQ（cat/wild/dog）、MetFace、CelebA-HQ 上进行。2-RF 用 single Euler step ($N=1$) 就能得到高质量结果（Figure 14）。

### 6.5 Domain Adaptation（Table 2）

| Method | OfficeHome | DomainNet |
|--------|-----------|-----------|
| ERM | 66.5 ± 0.3 | 40.9 ± 0.1 |
| IRM | 64.3 ± 2.2 | 33.9 ± 2.8 |
| CORAL | 68.7 ± 0.3 | 41.5 ± 0.2 |
| **Ours** | **69.2 ± 0.5** | 41.4 ± 0.1 |

在 latent representation 上构建 rectified flow，将 test domain 转移到 train domain。

---

## 7. Distillation（Section 2.2）

Reflow 后得到 k-RF，进一步 distill 为 one-step model：

$$\hat{T}(z_0) = z_0 + v(z_0, 0)$$

**Distillation loss**：

$$\mathbb{E}\left[\left\|(Z_1^k - Z_0^k) - v(Z_0^k, 0)\right\|^2\right]$$

这就是 (1) 在 $t = 0$ 时的项。

**Distillation vs. Rectification 的区别**：
- **Distillation**：faithfully approximate $(Z_0^k, Z_1^k)$，不改变 coupling
- **Rectification**：yield 新的 coupling $(Z_0^{k+1}, Z_1^{k+1})$，更低 transport cost，更直

Distillation 只在最后阶段用，用于 fine-tune fast one-step inference。

---

## 8. Reflow 算法（Algorithm 4）

```python
# Algorithm 4: Reflow
Coupling = Data  # {(x0, x1)}
for k in range(K):
    Model = Train(Coupling)  # 训练 k-RF
    Coupling = Sample(Model, Data)  # 生成新的 (z0, z1) pairs
return Coupling
```

实践中（CIFAR10）：
- 每次 reflow 生成 4M pairs
- fine-tune 300K steps
- 用 EMA (ratio 0.999999) 平滑
- Distillation 时用 LPIPS loss 代替 L2（empirically 更好）

---

## 9. ODE vs. SDE 的深度对比（Section 4）

Paper 末尾有一个非常 insightful 的讨论，argue **diffusion noise 不是 diffusion model 成功的关键**：

| 维度 | ODE | SDE |
|------|-----|-----|
| Conceptual simplicity | ✓ 简单 | ✗ 涉及 stochastic calculus |
| Numerical speed | ✓ 快 | ✗ 慢 |
| Time reversibility | ✓ forward/backward 对称 | ✗ 复杂 |
| Latent space | ✓ deterministic, low transport cost | ✗ stochastic, useless for latent |
| Training difficulty | ✓ 简单 regression | 类似（loss 差一个 reparameterization） |
| Expressive power | 等价（SDE → ODE via probability flow） | 等价 |
| Manifold data | ✓ 自然 smooth | ✗ 需要 anneal noise |

**核心论点**：DDPM/score SDE 的成功主要归功于 **stable optimization-based training procedure**（避免了 GAN 的 minimax），而非 diffusion noise 本身。Rectified Flow 证明了：去掉 noise，直接学 ODE，效果一样好甚至更好。

---

## 10. Build Intuition 的几个关键点

### 10.1 为什么 linear interpolation 是"对的"

Linear interpolation $X_t = tX_1 + (1-t)X_0$ 是 Euclidean space 的 **geodesic**。它是最短路径，constant speed。在 ODE 视角下，这是最 natural 的"road"。

PF-ODEs 用 exponential $\alpha_t$ 是因为它们继承了 Ornstein-Uhlenbeck process 的结构——这是 SDE 推导的副产物，ODE 视角下毫无必要。

### 10.2 Reflow 为什么能 straighten

第一次 rectification 从 independent coupling $(X_0, X_1) \sim \pi_0 \times \pi_1$ 出发。这个 coupling 是**随机的**，路径会交叉（Figure 2a）。

Rectify 后得到 $(Z_0, Z_1)$，这是一个 **deterministic** coupling（ODE 的解是 deterministic 的）。确定性意味着：给定 $Z_0$，$Z_1$ 唯一确定。这大大减少了路径交叉的可能。

第二次 rectify 从 $(Z_0, Z_1)$ 出发，这个 coupling 已经比 independent 更"有序"，linear interpolation 路径交叉更少，rectify 后更接近直线。

数学上，Theorem 3.7 的 $O(1/K)$ 收敛率来自 transport cost 的 telescoping——每次 rectify 减少 $\mathbb{E}[\|Z_1 - Z_0\|^2]$，减少量 = $S(Z) + V(\text{coupling})$。

### 10.3 为什么 straight = fast

Straight path 意味着 $v(Z_t, t) = Z_1 - Z_0 = \text{const}$（沿每条 path）。Euler method：

$$Z_{t + \Delta t} = Z_t + v(Z_t, t) \Delta t$$

如果 $v$ 恒定，**一步**就到：$Z_1 = Z_0 + v(Z_0, 0) \cdot 1$。没有 discretization error。

Curved path 需要 multiple steps 来近似曲线。PF-ODEs 的 curved + non-uniform speed 导致即使 $N=10$ 也表现不好（Figure 8a）。

### 10.4 Burgers' equation 的联系

如果 flow 是 straight，$v$ 必须满足 **inviscid Burgers' equation**：

$$\partial_t v + (\partial_z v) v = 0$$

这是从 $\frac{\mathrm{d}}{\mathrm{d}t}v(Z_t, t) = 0$（velocity 沿 path 恒定）推出的。Burgers' equation 是流体力学中的经典方程，这里出现在 straight flow 的必要条件中——很美的联系。

---

## 11. 后续影响与相关工作

这篇 paper 是 **Flow Matching** 系列工作的奠基之一（与 Meta 的 Concurrent work 一起）。后续发展：

- **Stable Diffusion 3** 使用了 Rectified Flow 的变体
- **Optimal Transport 条件**：[42] 中 Qiang Liu 进一步探讨 restrict $v$ 为 gradient field 来找 c-optimal coupling
- **Diffusion Bridges**：[44] 扩展到 bridge processes
- **SDXL / Flux** 等大规模 model 采用 flow matching 框架

**参考链接**：
- arXiv: https://arxiv.org/abs/2209.03003
- Qiang Liu 个人主页: https://www.cs.utexas.edu/~lqiang/
- OpenAI Consistency Models (受此启发): https://openai.com/research/consistency-models
- Meta Flow Matching: https://arxiv.org/abs/2210.02747
- Stable Diffusion 3 (使用 Rectified Flow): https://stability.ai/news/stable-diffusion-3-research-paper

---

## 12. 实操建议总结

1. **Default choice**：用 linear interpolation $X_t = tX_1 + (1-t)X_0$，$\pi_0 = \mathcal{N}(0, I)$
2. **训练**：简单 MSE regression，target = $(x_1 - x_0)$，Adam optimizer
3. **Reflow**：1 次足够（2-RF），生成 4M pairs，fine-tune
4. **Distillation**：最后阶段用 LPIPS loss，得到 one-step model
5. **Sampling**：few-step 用 Euler，full simulation 用 RK45
6. **Image translation**：用 style-aware loss (equation 20)，$\pi_0, \pi_1$ 都是 data distributions

---

## 13. 开放问题

1. **多维 c-optimal**：如何 efficiently restrict $v$ 为 gradient field？
2. **Non-Euclidean geometry**：如果 data 在 manifold 上，linear interpolation 可能不合适。如何设计 geodesic interpolation？
3. **Reflow 的误差累积**：理论上 $O(1/K)$，实践中 2-3 次最优。能否设计更 stable 的 reflow？
4. **Conditional generation**：如何 incorporate text/image conditioning？
5. **Video / 3D**：temporal consistency 如何保证？

---

这篇 paper 的优雅之处在于：**用一个极其简单的 regression objective，统一了 generative modeling 和 domain transfer，同时提供了 strong theoretical guarantees（marginal preserving, cost reduction, straightening）**。它揭示了 diffusion model 成功的真正原因——stable optimization procedure——而 diffusion noise 本身是多余的。这种 "less is more" 的 insight 正是 great research 的标志。
