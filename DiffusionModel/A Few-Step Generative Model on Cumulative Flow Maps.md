---
source_pdf: A Few-Step Generative Model on Cumulative Flow Maps.pdf
paper_sha256: d70dd7fe2dd9fdc12e8efff5ae77a83e5dcd869fbad171fe82b66a09e6d392d5
processed_at: '2026-08-17T23:16:36-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 CFM

## 一句话版

现在的 diffusion model 采样要 1000 步，太慢。已有方法要么靠 distillation（要训练 teacher），要么像 Mean Flow 只能在 flow matching 的 velocity 形式上 work。这篇 paper 的核心 trick 是：**把所有生成模型抽象成同一个函数形式，然后让模型直接学一个"长程映射"（从起点一步跳到终点），用 PDE 把它和瞬时场绑起来引入监督**，不动 architecture、不做 distillation、只改 loss 和 time embedding，就把 DDIM / EDM / Flow Matching 四种主流形式都加速到 1-10 步。

参考：https://arxiv.org/abs/2505.13447 （Mean Flow，CFM 的特例）

---

## 为什么慢？

Diffusion 训练时学的是"瞬时动力学"——给定当前状态 $x$ 和时间 $t$，预测一小步 $h$ 之后应该往哪里走（velocity / x0-prediction / denoiser，看你用哪种 parameterization）。

采样时反复调用这个瞬时场，每次走一小步，从 $t=0$ 走到 $t=1$ 需要 1000 步。每一步都要过一次神经网络，所以慢。

想法很自然：**能不能让模型直接学一个"从 $t$ 跳到任意 $r$"的长程映射？** 这样一两步就能到终点。

---

## 为什么直接学长程映射难？

### 难点 1：没法直接监督

多步模型训练时，监督信号来自 conditional：给定数据点 $x_1$，从噪声 $x_0$ 出发的中间状态 $x_t = t x_1 + (1-t) x_0$ 是已知的（直线路径），所以可以直接告诉模型"这一步的目标是 $x_1 - x_0$"。

但长程映射 $m_{t\to r}(x)$ 没法解析算出来——它是无数条 conditional 直线"平均"出来的弯曲 marginal 路径，闭式解不存在。所以**没法直接告诉模型"长程目标是什么"**。

### 难点 2：Mean Flow 只对一种形式 work

Mean Flow 把这个想法在 $u$-prediction Flow Matching 上实现了，通过"Mean Flow identity"对时间求导推出一个自洽方程。但这个推导**严重依赖 $u$-FM 的特殊结构**——$F$ 是线性的、$f_1$ 直接是速度。对 DDIM（带 $\sqrt{\bar\alpha}$ 系数）、EDM（除以 $t$）、$x_1$-FM（除以 $1-t$）都不工作。而 graphics 应用（geometry / SDF）恰恰常用 EDM 和 DDIM，不是 FM。

---

## CFM 的三个核心 insight

### Insight 1：统一抽象

把所有形式写成同一个模板：

$$\psi_{t\to t+h}(x) = F[m_t(x),\, x,\, t,\, t+h]$$

- $F$ 是个抽象函数（四种形式对应四种 $F$）；
- $m_t(x)$ 是模型要学的量（velocity / x0-pred / denoiser / x1-pred）；
- $x$ 是当前状态；$t, t+h$ 是起止时间。

然后定义**累积场** $m_{t\to r}(x)$ 满足 $\psi_{t\to r}(x) = F[m_{t\to r}(x), x, t, r]$——**同一个 $F$，把 $m_t$ 换成 $m_{t\to r}$，把 $t+h$ 换成 $r$**。

这就把"瞬时"和"累积"统一到同一个框架里了。

### Insight 2：PDE reformulation 引入监督

$m_{t\to r}$ 没法直接监督，但它满足一个 PDE（特征线方程 / 反向 transport equation）：

$$\partial_t \psi_{t\to r}(x) = -(\partial_x \psi_{t\to r}(x))[\partial_\tau \psi_{t\to \tau}(x)]|_{\tau=t}$$

物理直觉：累积映射对"起点时间"求导，等于 Jacobi 矩阵乘以"瞬时方向"。这是 Lie 群 / ODE flow 的标准性质。

把这个 PDE 改写成 $m_{t\to r}$ 的方程，会得到：

$$m_{t\to r}(x) = G(t,r)\,m_t(x) + H(t,r)\,E[\,m_{t\to r}(x),\, \partial_t m_{t\to r}(x),\, \partial_x m_{t\to r}(x),\, m_t(x)\,]$$

翻译成人话：**累积场 = 系数 × 瞬时场 + 另一个系数 × (累积场自己 + 它的两个导数 + 瞬时场)**。

关键点：
- 瞬时场 $m_t(x)$ 那一项**可以用 conditional $m_t(x|x_1)$ 监督**，把数据信号引进来；
- 累积场自己那一项用 stop-gradient 当 target，避免梯度循环；
- 导数用 JVP 或一步前向差分估计。

### Insight 3：$r=t$ 退化回多步

当 $r \to t$ 时，$m_{t\to r} \to m_t$，loss 退化回原始多步 loss。训练时**混一半 $r=t$ 的样本**，等于给模型一个"安全锚点"，防止 cumulative 场在 self-consistent 训练中飘走。Table 1b 显示关掉这个 mixing → FID 直接 572 崩掉。

---

## 实现上有多简单？

只改两样东西：

1. **Time embedding**：原来的 $t$-embedder 之外加一个 $r$-embedder，输入用 $(\text{emb}_t + \text{emb}_r)/2$。当 $r=t$ 时退化回原始 embedding，所以可以从已训练的多步 checkpoint 直接 warm-start（叫 self-distillation mode）。

2. **Loss function**：换成 Eq. 7 的 surrogate loss，根据你用的形式（u-FM / x1-FM / DDIM / EDM）代入对应 $F$。

**不动 architecture、不做 distillation、batch size 不变**。训练成本 2-3×（因为要算导数，需要额外前向）。

---

## 实验亮点

### Image generation（CelebA-HQ 256, DiT-B/2）

| Method | 128-step | 4-step | 1-step |
|---|---|---|---|
| DDIM baseline | 23.0 | 123.4 | 132.2 |
| Consistency Distillation | 59.5 | 39.6 | 38.2 |
| **CFM-DDIM** | **19.2** | **17.5** | **24.9** |

128-step 反而比 baseline 更好（19.2 vs 23.0）—— cumulative 训练"附带"提升了多步质量。

### Geometry Distribution（GeoDist, EDM）

| Method | 60-step | 6-step | 3-step |
|---|---|---|---|
| GeoDist baseline | 0.017 | 0.119 | 0.153 |
| CFM $u$-FM | 0.630 | 0.629 | 0.628 |
| **CFM-EDM** | **0.017** | **0.018** | **0.018** |

重要发现：**EDM 在几何任务上完胜 flow matching**。$u$-FM（Mean Flow 等价形式）在这个任务上完全不行（0.63）。CFM 的统一框架让你能根据任务**挑对 parameterization**，这是 paper 强调的 graphics 价值。

### PDT（joint 位置预测，CFM-DDIM）

200× 加速（5 步 vs 1000 步），IoU / Precision / Recall 全面提升。原 PDT 把步数降到 50 就崩了（CD-J2J 26.6%），CFM 把曲线整个抬起来。

### Sketch / SDF

50× / 6-16× 加速，质量持平或更好。

---

## 为什么 work？三个 intuition

### Intuition A：解算子 vs ODE 右端

- $m_t(x)$ 像 ODE $\dot{x} = v_t(x)$ 的右端，局部信息；
- $m_{t\to r}(x)$ 像 ODE 的解算子 $\Phi_{t,r}$，全局信息。
学解算子更难（非线性、长 horizon），但推理时一步到位。

### Intuition B：PDE 当 anchor

累积场满足一个 PDE，这个 PDE 把"累积场自己 + 它的导数 + 瞬时场"绑在一起。**学瞬时场就等于给 PDE 钉了一个边界条件**，整个 PDE 解（即 cumulative field）就被数据驱动地学出来。Stop-gradient 让模型用"上一版估计"当 target，避免梯度走回头路，类似于 EMA teacher 的作用但不需要额外网络。

### Intuition C：为什么 EDM 在 geometry 上比 FM 强？

EDM 的 conditional path 是 $x = x_1 + t\epsilon$（噪声直接加在数据上），对几何结构友好；FM 是 $x = tx_1 + (1-t)x_0$（差值插值），对点云稀疏区域容易塌陷。CFM 的抽象让你能根据任务挑工具，不被迫用 FM。

---

## 与已有方法的精确关系

- **Mean Flow** = CFM 在 $u$-prediction FM 下的特例。把 $F$ 取成 $f_1(f_4-f_3)+f_2$ 代入 Theorem 3，推出的 loss 和 Mean Flow 一模一样。但 Mean Flow 的推导绑死了 $u$-FM 的特殊结构，CFM 通过抽象 $F$ + Assumption 1（特别是 affine 结构假设）generalize 到四种形式。
- **Consistency Model**：纯 self-consistency，不需要 conditional 监督；CFM 仍然用 conditional $m_t(x|x_1)$ 当 anchor，靠 PDE reformulation 引入 self-consistent target，所以训练更稳、能扩展到 DDIM/EDM/x1-FM。
- **Distillation 系列**（DMD、Progressive、Consistency Distillation）：要训练 teacher；CFM 不需要。

---

## 局限

作者自己承认：
- 只验证了 5 个 task、4 种 formulation，ImageNet 大规模未做；
- 训练成本 2-3×；
- DDIM 形式对 learning rate 敏感（1e-5 → FID 84.9；1e-4 → 566 崩；要降到 1e-6 才稳），$u$-FM 鲁棒得多。

后续方向：扩展到 video / text-to-image（SD3、Flux）；跟 distillation / GAN loss 结合；把 $F$ 推广到更一般形式（去掉 affine 假设）覆盖 rectified flow 高阶插值。

---

## 一句话总结

CFM 把"瞬时动力学预测 → 数值积分"这套多步范式抽象成统一函数 $F[m_t(x), x, t, t+h]$，再通过 PDE reformulation 让模型直接学长程累积场 $m_{t\to r}(x)$，**只改 loss 和 time embedding**，就把 DDIM/EDM/u-FM/x1-FM 四种主流生成范式都加速到 1-10 步推理，在 graphics 多种任务上实现 10×-200× 加速且常常质量更好。Mean Flow 是其在 u-FM 下的特例。

参考链接：
- Mean Flow: https://arxiv.org/abs/2505.13447
- Flow Matching: https://arxiv.org/abs/2210.02747
- EDM: https://arxiv.org/abs/2206.00364
- DDIM: https://arxiv.org/abs/2010.02502
- Consistency Models: https://arxiv.org/abs/2303.01409
- Shortcut Models: https://arxiv.org/abs/2410.18557
- Improved Mean Flows: https://arxiv.org/abs/2512.02012

---

# Cumulative Flow Maps (CFM): 从瞬时到累积的统一框架

## 1. 问题背景与动机

生成模型（diffusion、flow matching 等）的本质，可以理解为**学习一个概率空间上的 transport**：把简单分布 $p_0$（高斯）的样本 $x_0$ 推送到复杂数据分布 $p_{\text{data}}$ 的样本 $x_1$。这种 transport 在流体力学里就叫做 **flow map** $\psi_{t\to r}(x)$ —— 给定 $t$ 时刻位置 $x$，告诉你在 $r$ 时刻它会到哪里。

现有方法的痛点：
- **多步生成**（DDIM、EDM、FM）：学习 *瞬时* dynamics $m_t(x)$（例如 velocity $u_t(x)$、x0-prediction $\tilde{x}_t(x)$、denoiser $D_t(x)$），然后用小步长 $h$ 反复积分。1000 步很常见。
- **Few-step 方法**：distillation 系列（Consistency Distillation、DMD、Progressive Distillation）需要 teacher–student；Mean Flow 通过学"平均速度"实现了 one-step，但**只绑定 $u$-prediction Flow Matching**，对 DDIM、EDM、$x_1$-FM 都不适用。

CFM 想做的：**直接学一个 long-range 的累积映射 $\psi_{t\to r}$**，从 $t$ 一步跳到任意 $r$，并且这个框架要能"插件式"地套到 DDIM / EDM / $u$-FM / $x_1$-FM 上，只改 time embedding 和 loss，不动 architecture，不做 distillation。

参考：
- Mean Flow: https://arxiv.org/abs/2505.13447
- Flow Matching: https://arxiv.org/abs/2210.02747
- EDM: https://arxiv.org/abs/2206.00364
- DDIM: https://arxiv.org/abs/2010.02502
- Consistency Models: https://arxiv.org/abs/2303.01409
- Shortcut Models: https://arxiv.org/abs/2410.18557

---

## 2. 瞬时流图（Instantaneous Flow Map）的统一表示

### 2.1 连续时间 Markov 过程的视角

把 DDIM / EDM / FM 全部放进 CTMP 框架：状态过程 $\{X_t\}_{t\in I}$，从 $t_0$ 到 $t_1$，由 transition kernel $p_{t+h|t}(A|x)$ 描述。模型学一个时间依赖函数 $m_t(x)$，parameterize 这个 kernel 的局部行为：

$$p_{t+h|t}(A\mid x) = p_{t+h|t}(A\mid x; m_t(x)) + O(h).$$

直接 loss $\mathcal{L}(\theta) = \mathbb{E}_{t,x\sim P_t}\|m_t^\theta(x) - m_t(x)\|^2$ 不能用，因为 marginal $P_t$ 和 reference $m_t(x)$ 都不知道。经典技巧：构造 conditional $p_t(x\mid X_{t_1})$（这里 $X_{t_1}\sim p_{\text{data}}$），用 surrogate loss
$$\mathcal{L}_c(\theta) = \mathbb{E}_{t, x\sim P_t(\cdot|x_1), x_1\sim p_{\text{data}}} \|m_t^\theta(x) - m_t(x\mid x_1)\|^2,$$
满足 $\nabla_\theta \mathcal{L}_c = \nabla_\theta \mathcal{L}$。

### 2.2 确定性瞬时流图

对小 $h$，存在确定性 $\psi_{t\to t+h}: S\to S$ 使 $p_{t+h|t}(\delta_{\psi(x)}|x)=1$，同时保持分布路径 $(P_t)$。把所有形式写成统一形式：

$$\boxed{\psi_{t\to t+h}(x) = F[m_t(x),\, x,\, t,\, t+h] + O(h)}$$

其中 $F[f_1, f_2, f_3, f_4]$ 是一个抽象函数，$f_1=m$（要学的量）、$f_2=x$（当前状态）、$f_3=t$（起始时间）、$f_4=t+h$（终止时间）。四个具体实例：

| 形式 | $F[f_1,f_2,f_3,f_4]$ | $f_1 = m_t(x)$ 是什么 |
|---|---|---|
| $u$-FM | $f_1(f_4 - f_3) + f_2$ | 速度 $u_t(x)$ |
| $x_1$-FM | $\dfrac{f_1 - f_2}{1 - f_3}(f_4 - f_3) + f_2$ | 终点预测 $x_t^1(x)$ |
| DDIM | $\sqrt{\bar\alpha_{f_4}}\,f_1 + \sqrt{1-\bar\alpha_{f_4}}\dfrac{f_2 - \sqrt{\bar\alpha_{f_3}} f_1}{\sqrt{1-\bar\alpha_{f_3}}}$ | $\tilde{x}_t(x)$（$x_0$-prediction） |
| EDM | $(f_4 - f_3)\dfrac{f_2 - f_1}{f_3} + f_2$ | denoiser $D_t(x)$ |

**Assumption 1** 对 $F$ 加四条约束：
1. **可微性**：对每个 $f_i$ 光滑；
2. **可逆性**：$\partial^2 F/\partial f_1\partial f_4$ 和 $\partial^2 F/\partial f_1\partial f_3$ a.e. 非退化；
3. **单位性**：$f_3 = f_4 \Rightarrow F = f_2$（"零步"映射到自身）；
4. **仿射结构**：$F[f_1,f_2,f_3,f_4] = P[f_3,f_4]\,f_1 + Q[f_3,f_4]\,f_2$。

第 4 条非常关键 —— 它把 $F$ 写成 $f_1$（要学的量）和 $f_2$（当前状态）的仿射组合，系数只依赖时间。所有常见 parameterization 都满足这个仿射结构，使得后面 Theorem 3 的 reformulation 才可能存在。

---

## 3. 累积流图（Cumulative Flow Maps）

### 3.1 Definition 2：把无穷小映射"拼"成长程映射

把瞬时映射 $\psi_{t\to t+h}$ 沿着一个分割 $\{t_i\}$ 无穷细分地复合起来，取极限就是累积映射：

$$\psi_{t\to r}(x) = \lim_{\max_i\{t_i-t_{i-1}\}\to 0}\, \psi_{t_{n-1}\to r}\circ\psi_{t_{n-2}\to t_{n-1}}\circ\cdots\circ\psi_{t\to t_1}(x). \tag{1}$$

它自然满足 **半群性质**：$\psi_{t\to r} = \psi_{s\to r}\circ\psi_{t\to s}$。这正是 ODE 解算子（solution operator）的特征。

### 3.2 累积参数化场

类比瞬时情形，定义一个累积场 $m_{t\to r}(x)$ 使得

$$\boxed{\psi_{t\to r}(x) = F[m_{t\to r}(x),\, x,\, t,\, r]} \tag{2}$$

也就是说，**同一个抽象函数 $F$ 既描述瞬时也描述累积**，只是把 $m_t$ 换成 $m_{t\to r}$，把 $t+h$ 换成 $r$。这是 CFM 的核心统一性所在。

### 3.3 Theorem 1（一致性）

$$\lim_{r\to t} m_{t\to r}(x) = m_t(x). \tag{3}$$

intuition：累积场在区间长度趋于零时退化为瞬时场。这有两个用处：
- 训练时让 $r=t$ 的样本变成普通的 multi-step 监督，稳定训练；
- 可以从已训练的 multi-step 模型 warm-start。

---

## 4. 训练的核心挑战与解决

### 4.1 Challenge 1：条件累积场不存在

直接用 $\|m_{t\to r}^\theta(x) - m_{t\to r}(x)\|^2$ 训练不可能，因为 $m_{t\to r}(x)$ 解析不出来。在 multi-step 模型里我们靠 conditional $m_t(x\mid x_1)$ 引入监督，但这里不行：**不存在 conditional cumulative field $m_{t\to r}(x\mid X_{t_1})$** 同时满足 (i) 与 conditional path transition 一致，(ii) marginal 等于 conditional 期望。

为什么？因为 conditional path 是直线（$x = tx_1 + (1-t)x_0$ 等），把 conditional transition 复合起来仍是一条直线，无法表达"真正的 marginal transport"，后者是无数 conditional 的平均，路径是弯的。所以**不能像 multi-step 那样直接 condition 化**。

### 4.2 关键 reformulation：Lemma 2 + Theorem 3

**Lemma 2**（初始时间导数）：
$$\partial_t \psi_{t\to r}(x) = -\big(\partial_x \psi_{t\to r}(x)\big)\,\big[\partial_\tau \psi_{t\to \tau}(x)\big]\Big|_{\tau=t}. \tag{4}$$

intuition：把累积映射对初始时刻求导，等于"Jacobi 矩阵"乘以瞬时方向场。这是 Lie 群 / ODE flow 的标准性质（特征线方程的反向 transport equation）。

**Theorem 3**：存在抽象函数 $E$（对最后一个 argument 仿射）、标量函数 $G, H$（满足 $G(t,t)=1, H(t,t)=0$），使得

$$\boxed{m_{t\to r}(x) = G(t,r)\,m_{t\to t}(x) + H(t,r)\,E\big[m_{t\to r}(x),\,x,\,t,\,r,\,\partial_t m_{t\to r}(x),\,\partial_x m_{t\to r}(x),\,m_{t\to t}(x)\big]} \tag{5}$$

并且 $E$ 内部对 $\partial_t m_{t\to r}$ 和 $\partial_x m_{t\to r}$ 的依赖**只通过组合项**

$$\partial_t m_{t\to r}(x) + \partial_4 F[m_{t\to t}(x), x, t, t]\,\partial_x m_{t\to r}(x) \tag{6}$$

出现。这点很重要 —— 它意味着两个导数必须以这个特定组合出现，使得离散化只需要"一步前向差分"就能估计（见 §5）。

### 4.3 用 reformulation 引入监督

把 (5) 代入 $\mathcal{L}^{CFM}$，右侧出现两项带数据监督潜力的量：
- $m_{t\to t}(x) = m_t(x)$：可以换成 conditional $m_t(x\mid x_1)$ 引入数据监督；
- $m_{t\to r}^\theta(x)$ 自己：用 stop-gradient 当作"当前最佳估计"（self-distillation 风格）。

于是得到 surrogate loss：

$$\boxed{\mathcal{L}_c^{CFM}(\theta) = \mathbb{E}_{t, r, x\sim P_t(\cdot|x_1), x_1\sim p_{\text{data}}}\Big\|m_{t\to r}^\theta(x) - \text{sg}\big(G[t,r]m_t(x|x_1) + H[t,r]E[\cdots]\big)\Big\|^2} \tag{7}$$

其中 $\text{sg}(\cdot)$ 是 stop-gradient。**Theorem 4** 保证 $\mathcal{L}_c^{CFM} = \mathcal{L}^{CFM} + C$（$C$ 与 $\theta$ 无关），所以这个 surrogate 是 asymptotically unbiased 的。

intuition 串起来：
- 我们要学 long-range 算子 $m_{t\to r}$；
- 它没法直接监督；
- 但它满足一个 PDE/特征线方程，能写成"自己 + 瞬时场 + 自己的导数"的组合；
- "瞬时场"那一项可以 condition 化来引入数据；
- "自己 + 导数"那一项用 stop-gradient 让它做 self-consistent target，避免循环依赖。

### 4.4 四个具体实例

把 (7) 代入到每个 $F$ 的具体形式：

**(1) $u$-FM**（与 Mean Flow 数学等价）：
$$\mathcal{L}_c^{FM} = \mathbb{E}\Big\|u_{t\to r}(x) - \text{sg}\Big((r-t)\big(\partial_t u_{t\to r}(x) + (x_1-x_0)\partial_x u_{t\to r}(x)\big) + (x_1-x_0)\Big)\Big\|^2.$$

变量：$x = tx_1 + (1-t)x_0$；$x_0\sim p_0$、$x_1\sim p_1$。

**(2) $x_1$-FM**：
$$\mathcal{L}_c^{FM} = \mathbb{E}\Big\|x_{t\to r}^1(x) - \text{sg}\Big(\tfrac{r-t}{1-r}\big((1-t)\partial_t u_{t\to r}(x) + (x_1-x)\partial_x u_{t\to r}(x)\big) + x_1\Big)\Big\|^2.$$

**(3) DDIM**（含 $\bar\alpha_t = \prod_{s\le t}\alpha_s$、$\beta_t$ 调度）：
$$\mathcal{L}_c^{FM} = \mathbb{E}\Big\|\tilde{x}_{t\to r}(x) - \text{sg}\Big(\Big(\tfrac{\sqrt{1-\bar\alpha_t}\sqrt{\bar\alpha_r}}{\sqrt{1-\bar\alpha_r}\sqrt{\bar\alpha_t}} - 1\Big)\Big((\sqrt{\bar\alpha_t}x_1 - \bar\alpha_t x)\partial_x x_{0,t\to r}(x) - \tfrac{2(1-\bar\alpha_t)(1-\beta_t)}{\beta_t}\partial_t x_{0,t\to r}(x)\Big) + x_1\Big)\Big\|^2.$$

**(4) EDM**（噪声尺度 $t\in[\sigma_{\max}, 0]$）：
$$\mathcal{L}_c^{FM} = \mathbb{E}\Big\|D_{t\to r}(x) - \text{sg}\Big(x_0 + \tfrac{r-t}{r}\big(t\,\partial_t D_{t\to r}(x) + \partial_x D_{t\to r}(x)\,(x-x_0)\big)\Big)\Big\|^2,$$
其中 $x = x_0 + t x_{\sigma_{\max}}$。

---

## 5. 算法与实现细节

### 5.1 时间采样器 $\mathcal{T}$

- 独立从 $\mathcal{U}[0,1]$ 采 $t, r$，若 $r$ 更接近 $t_0$ 就 swap（保持 $t$ 靠近起点）；
- 以比例 $\alpha = 0.5$ **混入 $r = t$ 的样本**，对应训练原始 instantaneous $m_t^\theta$。这点极其关键，Table 1b 显示关掉 mixing → FID 572（崩）。

### 5.2 模型修改（极小）

只改 time embedding：原 $t$-embedder 之外再加一个 $r$-embedder，输入用 $(\text{emb}_t + \text{emb}_r)/2$。
- $r = t$ 时：$(\text{emb}_t + \text{emb}_t)/2 = \text{emb}_t$，模型与原始 multi-step 完全一致；
- $r \ne t$ 时：模型能感知"从哪到哪"。
- 因此可以从已训练的 multi-step checkpoint 直接 warm-start（self-distillation mode），加速收敛。

### 5.3 导数计算：JVP vs 有限差分

由 Theorem 3，导数以组合形式 $\partial_t m + \partial_4 F[\,\cdots\,]\,\partial_x m$ 出现。两种实现：

**JVP 方式**（PyTorch/JAX 自动微分）：用 forward-mode Jacobian-vector product 一次算出组合项。

**有限差分方式**（适合不支持 JVP 的网络）：
$$\partial_t m_{t\to r}(x) + \partial_4 F\,\partial_x m_{t\to r}(x) \approx \frac{m_{s\to r}\big(\psi_{t\to t+h}(x)\big) - m_{t\to r}(x)}{h},\quad s = t+h.$$

intuition：沿瞬时映射走一步，看累积场的变化率，恰好捕获 (6) 这个组合。这非常优雅，因为 $\psi_{t\to t+h}(x) = F[m_t(x), x, t, t+h]$ 已知，所以只需额外一次前向。

### 5.4 训练 cost

JVP 或 finite-difference 都引入额外前向，训练成本约为 multi-step baseline 的 **2×–3×**。image 任务 2×，geometry distribution 任务 3×。

---

## 6. 实验

### 6.1 Image Generation（CelebA-HQ 256，DiT-B/2 + sd-vae-ft-mse）

| Method | 128-step | 4-step | 1-step |
|---|---|---|---|
| DDIM (baseline) | 23.0 | 123.4 | 132.2 |
| Consistency Distillation | 59.5 | 39.6 | 38.2 |
| Consistency Training | 53.7 | 19.0 | 33.2 |
| **CFM-DDIM (ours)** | **19.2** | **17.5** | **24.9** |

观察：
- 128-step 反而比 baseline 更好（19.2 vs 23.0）—— cumulative 训练"附带"提升了多步质量；
- 1-step FID 24.9，4-step 17.5，几乎无质量损失。

训练策略 ablation（100K step）：
- Scratch → 1-step FID 46.9；
- Self-distillation（先 50K 多步，再切 CFM）→ 42.7；
- 不混 instantaneous（$\alpha=0$）→ 572.3 全崩。

### 6.2 Geometry Distribution（GeoDist, EDM 形式）

$n = 2^{25}$ 个表面点，MLP 输入输出都是 3D 坐标，metric 是 Chamfer Distance：

| Method | 60-step | 6-step | 3-step |
|---|---|---|---|
| GeoDist (baseline) | 0.017 | 0.119 | 0.153 |
| CFM $x_1$-FM | 0.017 | 0.031 | 0.064 |
| CFM $u$-FM | 0.630 | 0.629 | 0.628 |
| **CFM-EDM** | **0.017** | **0.018** | **0.018** |

重要发现：**EDM 在几何分布任务上明显胜过 flow matching**。$u$-FM（即 Mean Flow 等价形式）在这个任务上完全不行，CFM 的统一框架让你能挑对的工具。

### 6.3 PDT（joint 位置预测，RigNet，CFM-DDIM）

| Method | CD-J2J↓ | IoU↑ | Prec↑ | Rec↑ |
|---|---|---|---|---|
| PDT (DDPM 1000-step) | 6.4% | 57.4% | 53.6% | 64.5% |
| PDT (DDPM 50-step) | 26.6% | 1.0% | 0.5% | 42.7% |
| PDT (DDPM 10-step) | 27.8% | 0.8% | 0.4% | 36.3% |
| CFM-DDIM 50-step | 5.4% | 66.9% | 60.8% | 77.7% |
| CFM-DDIM 10-step | 5.2% | 66.3% | 60.8% | 76.9% |
| CFM-DDIM 5-step | 6.2% | 54.3% | 47.3% | 67.8% |

**200× 加速**（5 步 vs 1000 步）的同时 IoU/Precision/Recall 全面提升。原 PDT 把步数降到 50 就崩了（CD-J2J 26.6%），CFM 把曲线整个抬起来。

### 6.4 Image-to-Sketch Generation（ControlSketch）

SwiftSketch 原本 50 步；CFM 1 步 / 4 步匹配：

| Setup | MS-SSIM↑ (seen/unseen) | DreamSim↓ (seen/unseen) |
|---|---|---|
| Cat SwiftSketch 50-step | 0.619 / 0.614 | 0.577 / 0.577 |
| Cat CFM 4-step | 0.618 / 0.612 | 0.578 / 0.577 |
| Cat CFM 1-step | 0.617 / 0.611 | 0.579 / 0.576 |

50× 加速无显著降质。

### 6.5 SDF Generation（Functional Diffusion，64 个表面点 sparse 条件）

| Method | Chamfer↓ | F1↑ | Boundary↓ |
|---|---|---|---|
| CFM 4-step | 0.048 | 0.659 | 0.011 |
| CFM 10-step | 0.048 | 0.660 | 0.011 |
| FuncDiff 64-step | 0.101 | 0.707 | 0.012 |

Chamfer 反而更好（0.048 vs 0.101），F1 略低，6–16× 加速。

### 6.6 Learning-rate sensitivity（CelebA-HQ, 1-step FID-50K）

- **DDIM** 对 LR 高度敏感：LR=1e-5 → FID 84.94；LR=1e-4 → 566.11；继续降到 1e-6 再训 → 65.89。
- **$u$-FM** 在 1e-4 / 3e-5 / 1e-5 三档都收敛到相近 FID。

实操含义：用 CFM-DDIM 时需要更小心调 LR，$u$-FM 更鲁棒。

---

## 7. 与 Mean Flow 的精确关系

Mean Flow 学的是平均速度 $u_{t\to r}(x) = (\psi_{t\to r}(x) - x)/(r-t)$，它通过 Mean Flow identity（$\psi_{t\to r}(x) = x + \int_t^r u_\tau(\psi_{t\to \tau}(x))\,d\tau$）两边对 $t$ 求导，推出一个自洽方程。把 $F$ 取成 $u$-FM 的形式 $F[f_1,f_2,f_3,f_4] = f_1(f_4-f_3)+f_2$ 代入 (5)，可以验证得到的就是 Mean Flow 的损失。所以 **Mean Flow = CFM 在 $u$-prediction FM 下的特例**。

但 Mean Flow 的推导**强烈依赖 $u$-prediction 的特殊结构**（$F$ 是线性的、$f_1$ 直接是速度），对 DDIM（带 $\sqrt{\bar\alpha}$ 复杂系数）、EDM（除以 $t$ 的形式）、$x_1$-FM（除以 $1-t$）都不工作。CFM 通过抽象函数 $F$ + Assumption 1（特别是 affine 结构假设）把这件事 generalize 到了所有这四种形式，并且每个都给出 closed-form loss。

---

## 8. 建立 Intuition 的几个关键点

### Intuition A：瞬时场 vs 累积场 = 微分方程右端 vs 解算子
- $m_t(x)$ 像是 ODE $\dot{x} = v_t(x)$ 的右端 $v_t$，局部信息；
- $m_{t\to r}(x)$ 像是 ODE 的解算子 $\Phi_{t,r}$，全局信息。
学解算子更难（非线性、长 horizon），但推理时一步到位。

### Intuition B：为什么 conditional cumulative 不存在？
conditional path 在 $u$-FM 下是直线 $x = tx_1 + (1-t)x_0$。直线复合无数次还是同一条直线，给出 $m_{t\to r}(x\mid x_1) = (x_1 - x_0)$ 这样的常量。但 marginal path 是无数直线的"包络"，弯曲。直线无法表示弯曲线，所以 conditional cumulative field 在 Definition 2 意义下不存在。Challenge 1 的本质就是这个。

### Intuition C：Theorem 3 的 PDE 视角
$m_{t\to r}(x)$ 满足一个反向 transport equation（对 $t$）。这个 PDE 把 $m_{t\to r}$ 自己、它的 $\partial_t$ 和 $\partial_x$、以及瞬时场 $m_t$ 都卷到一起。所以"监督瞬时场"等价于"在 PDE 里钉了一个边界条件"，从而让整个 PDE 解（即 cumulative field）被数据驱动地学出来。**Stop-gradient 的作用**：把 PDE 右端出现的 $m_{t\to r}$ 自身当成"上一版估计"，避免梯度走回头路。

### Intuition D：为什么混 $\alpha=0.5$ 的 instantaneous training？
$r=t$ 时 (5) 退化成 $m_{t\to t} = m_t$，loss 变回原始 multi-step loss。这给模型一个"安全锚点"，防止 cumulative field 在 self-consistent 训练中飘走。可类比 consistency model 的 "two-step loss + one-step loss" 混合训练。

### Intuition E：CFM 与 Consistency Model 的区别
Consistency Model 直接 enforce $\psi_{t\to 1} = \psi_{t'\to 1}$ 这种"端点一致性"，loss 是 $\|f_\theta(x_t, t) - f_\theta(x_{t'}, t')\|^2$；它**不需要任何 conditional 监督**，纯 self-consistency。CFM 不一样：它**仍然用 conditional $m_t(x\mid x_1)**当 anchor**，靠 PDE reformulation 引入 self-consistent target，所以训练更稳、能扩展到 DDIM/EDM/$x_1$-FM。

### Intuition F：为什么 EDM 在 geometry 上比 FM 强？
EDM 的 conditional path 是 $x = x_1 + t\epsilon$（噪声直接加在数据上），对几何结构更友好；FM 是 $x = tx_1 + (1-t)x_0$（差值插值），对点云稀疏区域容易"塌陷"。CFM 的统一抽象让你能根据任务**选对 parameterization**，这是 paper 强调的图形学应用价值。

---

## 9. 局限与未来工作

作者承认：
- 只验证了 5 个 task、4 种 formulation；ImageNet 大规模未做；
- 训练成本 2–3×；
- DDIM 形式对 LR 敏感。

自然的后续方向：
- 扩展到 video diffusion、text-to-image latent diffusion（SD3、Flux）；
- 跟 distillation 方法结合（CFM + GAN loss + DMD）；
- 把 abstract function $F$ 推广到更一般形式（去掉 affine 假设），覆盖 rectified flow 的高阶 polynomial 插值等。

---

## 10. 一句话总结

CFM 把"瞬时动力学预测 → 数值积分"这一套 multi-step 范式抽象成统一函数 $F[m_t(x),x,t,t+h]$，再通过 PDE reformulation 让模型直接学长程累积场 $m_{t\to r}(x)$，**不动 architecture、不做 distillation、只改 time embedding 和 loss**，就把 DDIM/EDM/$u$-FM/$x_1$-FM 四种主流生成范式都加速到 1–10 步推理，在图形学多种任务上实现 10×–200× 加速且常常**质量更好**。Mean Flow 是其在 $u$-FM 下的特例。

参考链接：
- Paper (arxiv when available): https://arxiv.org/abs/2505.13447 (Mean Flow，作为 CFM 的特例)
- Flow Matching: https://arxiv.org/abs/2210.02747
- EDM: https://arxiv.org/abs/2206.00364
- DDIM: https://arxiv.org/abs/2010.02502
- Consistency Models: https://arxiv.org/abs/2303.01409
- Shortcut Models: https://arxiv.org/abs/2410.18557
- Functional Diffusion: https://arxiv.org/abs/2406.17879
- Improved Mean Flows: https://arxiv.org/abs/2512.02012
