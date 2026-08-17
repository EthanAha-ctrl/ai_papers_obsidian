---
source_pdf: Mean Flows for One-step Generative Modeling.pdf
paper_sha256: 322ba9244ad2c72382038e421fe0ffa92510c51d4f9c740bd7282b36bfc25206
processed_at: '2026-08-05T17:03:38-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 MeanFlow

## 一、先讲清楚要解决什么问题

Diffusion model / Flow Matching 这类生成模型，本质上是学一个"速度场" $v(z_t, t)$，告诉你**在当前位置 $z_t$、当前时间 $t$，应该往哪个方向走、走多快**。然后你解一个 ODE $\frac{dz_t}{dt} = v(z_t, t)$ 从 noise 走到 data。问题是这轨迹是弯的，你要走 250 步 Euler 才能勉强走对，一步走完就跑偏了——因为你只知道了"当前点的切线方向"，不知道整段路怎么走。

**One-step generation 的目标**：用一次 function call 直接从 noise 跳到 data。怎么做到？

之前的路子（Consistency Models 一族）是说：我强加一个约束给神经网络——**同一个 trajectory 上的所有点，网络必须输出同一个 endpoint**。这相当于让网络自己学会"终点是什么"，但这个约束是强加在网络行为上的，底下真正的 ground-truth field 是什么没人说清楚。后果：training 不稳，要精心设计 discretization curriculum，慢慢缩短时间步，不然就崩。

MeanFlow 的切入点：**别再给网络加约束了，直接换一个有明确定义的 ground-truth field 去拟合。**

---

## 二、核心 idea：从"瞬时速度"换成"平均速度"

物理课里的常识：瞬时速度 $v(t)$ 是某一时刻的速度；平均速度 $u$ 是**总位移除以总时间**。

$$u = \frac{\text{位移}}{\text{时间}} = \frac{1}{t - r} \int_r^t v(\tau) \, d\tau$$

为什么要换？你想 one-step 采样，本质上是想知道"从 $r$ 到 $t$ 这段我整体要移动多少"。瞬时速度只知道"此刻往哪走"，平均速度直接告诉你"整段位移方向"。一次 $u(z_1, 0, 1)$ 就给出 noise→data 的整体位移，一步到位，不用积分。

**关键 insight**：这个 $u$ 是一个**独立存在的 field**，由 $v$ 决定，跟网络无关。就像 Flow Matching 里 marginal velocity $v$ 是 ground-truth 一样，$u$ 是另一个有明确数学定义的 ground-truth。这就是 MeanFlow 比 Consistency Model principled 的地方——**有一个明确的 field 作为回归目标，最优解独立于网络架构**。

---

## 三、关键数学：MeanFlow Identity 是怎么冒出来的

直接用 $u$ 的定义训练是不行的——你要算 $\int_r^t v \, d\tau$，要数值积分，intractable。

作者的招：**把定义式两边对 $t$ 求导**，得到一个局部的关系。

从

$$(t-r) \, u(z_t, r, t) = \int_r^t v(z_\tau, \tau) \, d\tau$$

两边对 $t$ 求导（$r$ 视为与 $t$ 独立）：

- 左边 product rule：$u + (t-r) \frac{du}{dt}$
- 右边 fundamental theorem of calculus：直接得到 $v(z_t, t)$

整理：

$$\boxed{u = v - (t-r) \frac{du}{dt}}$$

这个就是 **MeanFlow Identity**。

**人话翻译**：平均速度 = 瞬时速度 - 时间区间 × 平均速度的时间导数。

- 当 $r \to t$（区间缩为 0），第二项消失，$u \to v$——平均速度退化成瞬时速度，定义自洽。
- 当 $r$ 与 $t$ 离得远，第二项起作用，把"积分效应"补回来。

这个公式告诉我们：**只要拿到 $v$（免费，closed-form）和 $\frac{du}{dt}$（可以从网络算），就能拼出 $u$ 的 target。**不用真的算积分。

### $\frac{du}{dt}$ 怎么算？

$u$ 是 $(z_t, r, t)$ 三个变量的函数，$z_t$ 本身又是 $t$ 的函数（沿 ODE 走）。所以是 total derivative：

$$\frac{du}{dt} = \frac{\partial u}{\partial z} \cdot \frac{dz_t}{dt} + \frac{\partial u}{\partial r} \cdot \frac{dr}{dt} + \frac{\partial u}{\partial t} \cdot \frac{dt}{dt}$$

代入 $\frac{dz_t}{dt} = v$，$\frac{dr}{dt} = 0$，$\frac{dt}{dt} = 1$：

$$\frac{du}{dt} = v \, \partial_z u + \partial_t u$$

这就是一个 Jacobian-vector product (JVP)：Jacobian $[\partial_z u, \partial_r u, \partial_t u]$ 乘 tangent $[v, 0, 1]$。JAX / PyTorch 里 `jax.jvp` / `torch.func.jvp` 一行搞定。

---

## 四、训练 loss 长什么样

用网络 $u_\theta(z, r, t)$ 去 fit 平均速度。Target 由 MeanFlow Identity 拼出来：

$$u_{\text{tgt}} = v_t - (t-r) \big( v_t \, \partial_z u_\theta + \partial_t u_\theta \big)$$

这里 $v_t = \epsilon - x$ 是 Flow Matching 的 conditional velocity（closed-form，免费）。Loss：

$$\mathcal{L} = \mathbb{E} \big\| u_\theta - \text{sg}(u_{\text{tgt}}) \big\|_2^2$$

`sg` 是 stop-gradient。

**为什么 stop-gradient 很关键**：让 target 里的 JVP 部分在反传时被当作常数，**不需要二阶导**。JVP 内部只需要做一次类似 backward 的操作，反传到 input 而不到 $\theta$。实测开销仅 16%（B/4 on TPU v4-8：FM 0.045s/iter, MF 0.052s/iter）。

### 退化性质

当 $r \equiv t$ 时，$(t-r) = 0$，target 第二项消失：

$$u_{\text{tgt}} = v_t$$

这就**退化成标准 Flow Matching**。所以 MeanFlow 是 FM 的严格推广——FM 是 MeanFlow 在 $r=t$ 时的特例。

Ablation Tab. 1a 印证：0% $r \neq t$（纯 FM）1-NFE FID 是 328.91，完全失败；25% 时 61.06 最优；100% 时 67.32 仍能 work。

---

## 五、采样：1-NFE 直接出图

$$z_0 = z_1 - u_\theta(z_1, 0, 1), \quad z_1 \sim \mathcal{N}(0, I)$$

一次 function call，完事。代码就两行：

```python
e = randn(x_shape)
x = e - fn(e, r=0, t=1)
```

---

## 六、CFG 怎么"免费"整合进来

标准 CFG 采样时要算两次 NFE（conditional + unconditional），破坏 1-NFE。

MeanFlow 的招：**把 CFG 直接定义到 ground-truth field 里**，而不是采样时的 trick。

定义 CFG velocity field：

$$v^{\text{cfg}}(z_t, t \mid \mathbf{c}) = \omega \, v(z_t, t \mid \mathbf{c}) + (1-\omega) \, v(z_t, t)$$

然后让网络直接去拟合 $u^{\text{cfg}}$——即 CFG velocity 诱导出的 average velocity。因为 $v^{\text{cfg}}$ 本身也是一个 well-defined field，它的 average velocity $u^{\text{cfg}}$ 也是一个 well-defined field，MeanFlow Identity 照样适用。

训练时 target 稍微改一下（用 $\tilde{v}_t$ 替换 $v_t$）：

$$\tilde{v}_t = \omega \, v_t + (1-\omega) \, u^{\text{cfg}}_\theta(z_t, t, t)$$

这里用到了 $v(z_t, t) = u^{\text{cfg}}(z_t, t, t)$（$r=t$ 时 $u = v$）。$\omega = 1$ 时退化到无 CFG。

**采样时**：直接用 $u^{\text{cfg}}_\theta$，1-NFE，**不需要做 linear combination**——因为网络已经把 CFG 的混合效应烧进权重里了。

Appendix B.1 进一步引入 mixing scale $\kappa$，把 class-conditional 与 class-unconditional 的 $u^{\text{cfg}}$ 都混进 target，FID 从 20.15 降到 18.63。

### CFG ablation（Tab. 1f）

| $\omega$ | FID |
|---|---|
| 1.0（无 CFG） | 61.06 |
| 2.0 | 20.15 |
| 3.0 | **15.53** |
| 5.0 | 20.75 |

CFG 在 1-NFE 下依然有效，最佳 $\omega=3$。

---

## 七、为什么比 Consistency Models 好？直觉

**Consistency Models 的逻辑**：
- 定义网络 $f_\theta(z_t, t)$，约束它在 trajectory 上输出同一个 endpoint。
- 这是**网络行为的约束**，ground truth 是什么没说清。
- 训练时要 discretization curriculum（慢慢缩短时间步），不然 unstable。
- Anchor 在 data side：$r \equiv 0$ 固定，只条件于一个时间 $t$。

**MeanFlow 的逻辑**：
- 有一个**独立存在的 field** $u$，由定义 $\frac{1}{t-r}\int v \, d\tau$ 给出。
- 这个 field 满足一个**微分方程**（MeanFlow Identity），由定义推出，无任何额外假设。
- 网络去 fit 这个 field。
- 最优解**独立于网络架构**——所以稳定，不需要 curriculum。
- 条件于两个时间 $(r, t)$，信息更丰富。

**关键直觉**：MeanFlow 的约束来自 field 的定义本身，是一个数学事实，跟"我随便选个 consistency loss"完全不同。Consistency constraint 是被加在网络上，MeanFlow Identity 是 field 天生满足的——网络去 fit 它就自然满足 consistency，不需要额外加约束。

### 自然一致性

由 $u$ 的定义，积分可加性直接给：

$$(t-r) u(z_t, r, t) = (s-r) u(z_s, r, s) + (t-s) u(z_t, s, t)$$

即"一大步 = 两小步之和"，这就是 Consistency Model 想要的属性，**在 MeanFlow 里是 free 的**——网络只要准确 fit $u$，自动满足。

---

## 八、Sufficiency 的小细节（Appendix B.3）

求导相等一般不推积分相等（差一个常数）。这里有个边界条件 trick：

定义 $S(z_t, r, t) = (t-r) u(z_t, r, t)$，则 $S|_{t=r} = 0$。同时 $\int_r^r v \, d\tau = 0$。所以常数 $C_1 = C_2$，MeanFlow Identity 与原定义等价。

**为什么建模 $u$ 而非 $S$**：$S$ 的定义 $(t-r)u$ 自动满足 $S|_{t=r}=0$ 边界条件。直接参数化 $S$ 需要显式加约束。这是 $u$ 的参数化形式带来的隐式好处。

---

## 九、实验结果

### ImageNet 256×256，1-NFE from scratch（Tab. 2）

| Method | Params | FID |
|---|---|---|
| iCT-XL/2 [43] | 675M | 34.24 |
| Shortcut-XL/2 [13] | 675M | 10.60 |
| IMM-XL/2 [52]（2-NFE guidance）| 675M | 7.77 |
| MeanFlow-B/2 | 131M | 6.17 |
| MeanFlow-M/2 | 308M | 5.01 |
| MeanFlow-L/2 | 459M | 3.84 |
| **MeanFlow-XL/2** | **676M** | **3.43** |

- 对比前 SOTA 1-NFE（Shortcut 10.60）**改进 67%**
- 对比 IMM 7.77（且 IMM 用了 2-NFE guidance）**改进 56%**
- **完全 from scratch**，无 pre-training、distillation、curriculum

### 2-NFE 结果

| Method | NFE | FID |
|---|---|---|
| DiT-XL/2 [34] | 250×2 | 2.27 |
| SiT-XL/2 [33] | 250×2 | 2.06 |
| MeanFlow-XL/2 | 2 | 2.93 |
| **MeanFlow-XL/2+** | **2** | **2.20** |

**2.20 FID 几乎追平 250-NFE 的 multi-step 模型**。Few-step 已经能打 many-step。

### Ablation 关键发现（Tab. 1）

**(c) Positional embedding**：即使只 embed $t - r$（区间长度）也能 work（FID 63.13），最优是 $(t, t-r)$（61.06）。说明**区间长度信息是核心**。

**(d) Time sampler**：lognorm(-0.4, 1.0) 最优（61.06），跟 Flow Matching 经验一致——logit-normal 分布对中间时间步采样更密。

**(e) Loss metric**：$p=1$（自适应加权的 L2）最优 61.06；$p=0$（纯 L2）79.75；$p=0.5$（Pseudo-Huber，iCT 用的）63.98。自适应权重很重要——因为误差量级在不同 $(r, t)$ 处差异很大。

**(b) JVP tangent** 破坏性实验：
- $(v, 0, 1)$ 正确：61.06
- $(v, 0, 0)$（丢 $\partial_t$）：268.06
- $(v, 1, 0)$：329.22
- $(v, 1, 1)$：137.96

JVP 公式不能瞎改，$\partial_r, \partial_t$ 虽然只是 1 维 tangent，但作用关键。

### Scalability（Fig. 4）

B(131M) → M(308M) → L(459M) → XL(676M)，FID 单调下降，标准 Transformer scaling behavior，没有 saturation 迹象。

### CIFAR-10（Tab. 3）

无 EDM preconditioner 也能达到 2.92 FID，跟有 EDM 的 iCT (2.83) 接近。

---

## 十、跟其他方法的精确对比

| Method | 时间变量 | 核心约束 | Ground-truth field | 需要 curriculum | 需要 distillation |
|---|---|---|---|---|---|
| Consistency Models [46,43,15] | 单时间 $t$（$r \equiv 0$）| 网络输出 consistency | 无明确 field | 是 | 否 |
| Shortcut Models [13] | 双时间 $(r, t)$ | 额外 self-consistency loss | 无 | 否 | 否 |
| IMM [52] | 双时间 | moment matching | 无 | 否 | 否 |
| Flow Map Matching [3] | 双时间 | 匹配 displacement | displacement $S$ | 否 | 否 |
| **MeanFlow** | **双时间** | **MeanFlow Identity**（由定义推出）| **average velocity $u$** | **否** | **否** |

唯一一个**有明确 ground-truth field 且约束由定义自然推出**的方法。

---

## 十一、几个深层直觉

### 1. MeanFlow 把积分的负担从推理时搬到了训练时

Flow Matching 推理时要积分（多步）。MeanFlow 训练时用 JVP 隐式"看到"积分效应，推理时直接出 displacement。代价是训练时每次 iteration 多一次 backward（16% overhead），收益是推理时 NFE = 1。

### 2. 为什么 JVP 是"免费积分"的等价物

JVP $\frac{du}{dt} = v \, \partial_z u + \partial_t u$ 在说："$u$ 在沿 ODE 演化时怎么变"。$\partial_z u$ 是空间敏感性，$\partial_t u$ 是时间敏感性。这两者结合 $v$（演化方向）告诉你：**沿 trajectory 演化时 average velocity 怎么演化**。MeanFlow Identity 把这个"演化信息"反过来用作 target——网络被迫去学一个自洽的、与 $v$ 一致的 $u$ field。

### 3. 跟 multi-scale physics 的类比

作者在 conclusion 提到这跟物理里的多尺度模拟同构。instantaneous velocity 是微观量，average velocity 是 coarse-grained 宏观量。MeanFlow 是在做"closure"——直接学一个宏观量使宏观演化自洽，不用解微观方程。这跟 Renormalization Group、homogenization、coarse-graining 是一脉相承的思想。

### 4. 退化为 Flow Matching 的意义

$r \equiv t$ 时 MeanFlow = FM。这意味着 MeanFlow 是在 FM 的基础上"加了一项修正"——$(t-r) \frac{du}{dt}$。当 $r$ 与 $t$ 离得近，修正小；离得远，修正大。训练时混合采样 $r=t$ 与 $r \neq t$ 让网络同时学到 instantaneous 与 average，25% $r \neq t$ 最优。

### 5. 跟 Consistency Models 的 anchor 区别

CM 是 anchor 在 data side（$r=0$ 固定），所以网络只条件于一个 $t$。MeanFlow 让 $r$ 自由变化，网络条件于 $(r, t)$，信息量翻倍。$r$ 的自由让网络学到"任意区间"的 average velocity，1-step 只是 $r=0, t=1$ 的特例，few-step 自然支持。

---

## 十二、Reference Links

- Flow Matching (Lipman et al. ICLR 2023): https://arxiv.org/abs/2210.02747
- Consistency Models (Song et al. ICML 2023): https://arxiv.org/abs/2303.01469
- Improved Consistency Training (iCT, ICLR 2024): https://arxiv.org/abs/2310.14189
- Consistency Models Made Easy (ECT, Geng et al.): https://arxiv.org/abs/2406.14548
- Shortcut Models (Frans et al. ICLR 2025): https://arxiv.org/abs/2410.12557
- Inductive Moment Matching (IMM): https://arxiv.org/abs/2503.07565
- Flow Map Matching (Boffi et al.): https://arxiv.org/abs/2406.07507
- DiT (Peebles & Xie CVPR 2023): https://arxiv.org/abs/2212.09748
- SiT (Ma et al. ECCV 2024): https://arxiv.org/abs/2401.01408
- Stable Diffusion 3 (Esser et al.): https://arxiv.org/abs/2403.03206
- Flow Matching guide and code: https://arxiv.org/abs/2412.06264
- Introduction to Flow Matching (Cambridge MLG blog): https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html
- Classifier-Free Guidance (Ho & Salimans): https://arxiv.org/abs/2207.12598
- REPA (Yu et al. ICLR 2025): https://arxiv.org/abs/2410.06985
- JAX documentation: https://jax.readthedocs.io/
- PyTorch torch.func.jvp: https://pytorch.org/docs/stable/func.html

---

## 一句话总结

**MeanFlow 把 one-step generation 从"给网络强加 consistency 约束"升级成"去拟合一个有明确数学定义的 average velocity field"。这个 field 满足一个由定义自然推出的微分方程（MeanFlow Identity），用 JVP 在训练时隐式完成积分，推理时单步直出。XL/2 在 ImageNet 256×256 from scratch 达到 3.43 FID（1-NFE），2-NFE 达到 2.20，几乎追平 250-NFE 的 multi-step diffusion 模型。这是 consistency-style 方法的 principled 升级版。**

---

# MeanFlow: One-step Generative Modeling 详解

## 核心动机与问题背景

Flow Matching 与 diffusion models 本质上都是学习一个 **instantaneous velocity field** $v(z_t, t)$，然后通过 ODE solver 数值积分采样。问题在于：即使 conditional flow 是直的（rectified），marginal velocity field 也会因为 marginalization 产生弯曲轨迹，coarse discretization 会带来大误差。

之前的 one-step 方法（Consistency Models, Shortcut, IMM）都是给 neural network 强加一个 **consistency constraint** 作为网络行为的性质，缺乏一个 underlying ground-truth field 作为指导。training 不稳定，需要 discretization curriculum。

MeanFlow 的 insight：与其建模 instantaneous velocity $v$，不如直接建模 **average velocity** $u$，它本身就是 displacement 除以 time interval。这样 1-NFE 采样就变得自然——因为 $u(z_1, 0, 1)$ 直接给出从 prior 到 data 的整段位移。

---

## 核心数学推导

### Average Velocity 的定义

给定 instantaneous velocity $v(z_\tau, \tau)$，定义 average velocity：

$$u(z_t, r, t) \triangleq \frac{1}{t-r} \int_r^t v(z_\tau, \tau) \, d\tau$$

变量含义：
- $z_t$：当前时刻 $t$ 的状态（latent）
- $r$：起始时间（reference time），通常 $r \in [0, 1]$，$r=0$ 对应 data side，$r=1$ 对应 prior side
- $t$：结束时间（target time）
- $t - r$：时间区间长度
- $v(z_\tau, \tau)$：瞬时速度
- $u$：从 $r$ 到 $t$ 的平均速度，是一个 $(r, t)$ 双时间变量的 field

注意 $u = \mathcal{F}[v]$ 是 $v$ 的泛函，**不依赖任何 neural network**——它是一个 ground-truth field，就像 Flow Matching 里的 marginal velocity 是 ground-truth 一样。

### 一致性边界条件

当 $r \to t$ 时：

$$\lim_{r \to t} u(z_t, r, t) = v(z_t, t)$$

即平均速度退化为瞬时速度。

对于任意中间时间 $s \in [r, t]$，由积分可加性：

$$(t-r) u(z_t, r, t) = (s-r) u(z_s, r, s) + (t-s) u(z_t, s, t)$$

这就是 consistency property，但**它是定义自然推出的**，不需要额外强加。任何准确近似 $u$ 的网络天然满足此性质。

### MeanFlow Identity —— 关键推导

把 average velocity 定义改写：

$$(t-r) \, u(z_t, r, t) = \int_r^t v(z_\tau, \tau) \, d\tau$$

两边对 $t$ 求导（把 $r$ 视作与 $t$ 独立）：

- 左边用 product rule：$\frac{d}{dt}[(t-r) u] = u + (t-r) \frac{d}{dt} u$
- 右边用 fundamental theorem of calculus：$\frac{d}{dt} \int_r^t v \, d\tau = v(z_t, t)$

得到：

$$\boxed{u(z_t, r, t) = v(z_t, t) - (t-r) \frac{d}{dt} u(z_t, r, t)}$$

这就是 **MeanFlow Identity**，描述了 $u$ 与 $v$ 的内在关系。它就是 average velocity 的定义经过求导得到的，**没有任何额外假设**。

### 计算 Time Derivative

$\frac{d}{dt} u$ 是 total derivative，展开为偏导：

$$\frac{d}{dt} u(z_t, r, t) = \frac{dz_t}{dt} \partial_z u + \frac{dr}{dt} \partial_r u + \frac{dt}{dt} \partial_t u$$

代入：
- $\frac{dz_t}{dt} = v(z_t, t)$（Flow Matching ODE）
- $\frac{dr}{dt} = 0$（$r$ 与 $t$ 独立）
- $\frac{dt}{dt} = 1$

得到：

$$\boxed{\frac{d}{dt} u(z_t, r, t) = v(z_t, t) \, \partial_z u + \partial_t u}$$

这正是 Jacobian-vector product (JVP)：Jacobian $[\partial_z u, \partial_r u, \partial_t u]$ 与 tangent vector $[v, 0, 1]$ 的乘积。可用 `torch.func.jvp` 或 `jax.jvp` 高效计算。

### 训练 Loss

把 $u$ 用 neural network $u_\theta$ 参数化，target 由 MeanFlow Identity 给出：

$$u_{\text{tgt}} = v_t - (t-r) \big( v_t \, \partial_z u_\theta + \partial_t u_\theta \big)$$

这里 $v_t = \epsilon - x$ 是 conditional velocity（沿用 Flow Matching 的 conditional 替换 marginal 的标准做法）。

Loss：

$$\mathcal{L}(\theta) = \mathbb{E} \big\| u_\theta(z_t, r, t) - \text{sg}(u_{\text{tgt}}) \big\|_2^2$$

其中 $\text{sg}$ 是 stop-gradient。**关键点**：因为 stop-gradient 作用在 target 上，JVP 部分在反向传播时被视作常数，**不需要 higher-order gradient**。JVP 只引入一次额外 backward pass（类似标准 backprop），开销 <20%。

当 $r \equiv t$ 时，$(t-r) = 0$，第二项消失，MeanFlow **退化为标准 Flow Matching**。所以 MeanFlow 是 Flow Matching 的严格推广。

### 1-step 采样

$$z_0 = z_1 - u_\theta(z_1, 0, 1), \quad z_1 = \epsilon \sim p_{\text{prior}}$$

仅 1-NFE，直接得到生成结果。

### 算法伪代码

**Algorithm 1 (Training):**
```
t, r = sample_t_r()
e = randn_like(x)
z = (1-t)*x + t*e
v = e - x
u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
u_tgt = v - (t-r) * dudt
error = u - stopgrad(u_tgt)
loss = metric(error)
```

**Algorithm 2 (1-step Sampling):**
```
e = randn(x_shape)
x = e - fn(e, r=0, t=1)
```

---

## CFG 的优雅整合

### 问题

标准 CFG 在采样时需要两次 NFE（conditional + unconditional），破坏了 1-NFE。

### MeanFlow 的做法

不把 CFG 当作采样时的 trick，而是把它定义到 ground-truth field 里：

$$v^{\text{cfg}}(z_t, t \mid \mathbf{c}) \triangleq \omega \, v(z_t, t \mid \mathbf{c}) + (1-\omega) \, v(z_t, t)$$

其中：
- $\mathbf{c}$：class condition
- $\omega$：guidance scale
- $v(z_t, t \mid \mathbf{c})$：class-conditional marginal velocity
- $v(z_t, t)$：class-unconditional marginal velocity

利用 $v(z_t, t) = v^{\text{cfg}}(z_t, t) = u^{\text{cfg}}(z_t, t, t)$（因为 $r=t$ 时 $u=v$），可以重写为：

$$v^{\text{cfg}}(z_t, t \mid \mathbf{c}) = \omega \, v(z_t, t \mid \mathbf{c}) + (1-\omega) \, u^{\text{cfg}}(z_t, t, t)$$

然后直接参数化 $u^{\text{cfg}}_\theta$，target 为：

$$u_{\text{tgt}} = \tilde{v}_t - (t-r) \big( \tilde{v}_t \, \partial_z u^{\text{cfg}}_\theta + \partial_t u^{\text{cfg}}_\theta \big)$$

其中：

$$\tilde{v}_t \triangleq \omega \, v_t + (1-\omega) \, u^{\text{cfg}}_\theta(z_t, t, t)$$

这里的 $v_t = \epsilon - x$ 是 sample-conditional velocity。当 $\omega = 1$ 时退化到无 CFG 的标准 MeanFlow。

### Improved CFG (Appendix B.1)

引入 mixing scale $\kappa$，把 class-conditional 与 class-unconditional 的 $u^{\text{cfg}}$ 都混入 target：

$$\tilde{v}_t = \omega(\epsilon - x) + \kappa \, u^{\text{cfg}}_\theta(z_t, t, t \mid \mathbf{c}) + (1-\omega-\kappa) \, u^{\text{cfg}}_\theta(z_t, t, t)$$

有效 guidance scale 为 $\omega' = \omega / (1-\kappa)$。Tab. 5 显示 $\kappa = 0.9$ 时 FID 从 20.15 降到 18.63。

**关键好处**：网络直接建模 $u^{\text{cfg}}$，采样时无需 linear combination，**保持 1-NFE**。

---

## Intuition 构建

### 1. 为什么 average velocity 比 instantaneous velocity 更适合 one-step？

- Instantaneous velocity $v(z_t, t)$ 只告诉你**当前点的切方向**。要走完整条轨迹必须数值积分。
- Average velocity $u(z_t, r, t)$ 直接告诉你**从 $r$ 到 $t$ 的整体位移方向**。1-NFE 采样就是 $z_0 = z_1 - (1-0) \cdot u(z_1, 0, 1)$，本质是把积分"烧进"网络里。

### 2. 为什么 MeanFlow Identity 比 consistency constraint 更 principled？

Consistency Models 在网络上强加 $f(z_t, t) = f(z_{t'}, t')$，这是**网络行为的性质**，ground-truth field 是什么不清楚。

MeanFlow 的 $u$ 是**从定义存在的 field**，MeanFlow Identity 是这个 field **必须满足的微分方程**，与网络无关。最优解独立于网络架构——这就是为什么 training 稳定，不需要 curriculum。

### 3. 为什么 JVP 不贵？

JVP $\frac{d}{dt} u = v \partial_z u + \partial_t u$ 在 target 里被 stop-gradient 包住，反传时被视作常数。所以：
- Forward：标准 forward pass（FM 也要做）
- Backward：JVP 内部的 backward 只反传到 input，不到 $\theta$；而 $\theta$-backprop 是标准的，不涉及二阶导

实测开销仅 16%（B/4 on TPU v4-8：FM 0.045s/iter, MF 0.052s/iter）。

### 4. 为什么 marginal velocity 是弯的即使 conditional 是直的？

Conditional flow $z_t = (1-t)x + t\epsilon$ 对每个 $(x, \epsilon)$ pair 是直线，velocity $v_t = \epsilon - x$ 是常数。但 marginal velocity $v(z_t, t) = \mathbb{E}_{p_t(v_t | z_t)}[v_t]$ 需要 marginalize over 所有能产生 $z_t$ 的 $(x, \epsilon)$ pairs。不同 pair 给出不同方向，期望后轨迹变弯。这就是为什么不能用一步 Euler 走完。

---

## 与 Prior Work 的精确对比

| Method | 条件变量 | 核心约束 | Ground-truth field | 需要 curriculum |
|--------|---------|---------|-------------------|---------------|
| Consistency Models [46,43,15] | 单时间 $t$（$r \equiv 0$） | 网络输出 consistency | 无明确 field | 是 |
| Shortcut Models [13] | $(r, t)$ 两时间 | 额外 self-consistency loss | 无 | 否 |
| IMM [52] | 两时间 | moment matching | 无 | 否 |
| Flow Map Matching [3] | $(r, t)$ | 匹配 displacement | displacement $S$ | 否 |
| **MeanFlow** | $(r, t)$ 两时间 | **MeanFlow Identity（定义推出）** | **average velocity $u$** | **否** |

关键区别：MeanFlow 的约束**完全来自 $u$ 的定义**，不引入任何额外 heuristic。

---

## 实验结果详解

### 主结果（Tab. 2）

**1-NFE ImageNet 256×256 from scratch:**

| Method | Params | NFE | FID |
|--------|--------|-----|-----|
| iCT-XL/2 [43] | 675M | 1 | 34.24 |
| Shortcut-XL/2 [13] | 675M | 1 | 10.60 |
| IMM-XL/2 [52] | 675M | 1×2 (2-NFE guidance) | 7.77 |
| MeanFlow-B/2 | 131M | 1 | 6.17 |
| MeanFlow-M/2 | 308M | 1 | 5.01 |
| MeanFlow-L/2 | 459M | 1 | 3.84 |
| **MeanFlow-XL/2** | **676M** | **1** | **3.43** |

相对 Shortcut（前 SOTA 1-NFE）改进 ~67%，相对 IMM 改进 ~56%。

**2-NFE:**

| Method | NFE | FID |
|--------|-----|-----|
| iCT-XL/2 | 2 | 20.30 |
| iMM-XL/2 | 1×2 | 7.77 |
| MeanFlow-XL/2 | 2 | 2.93 |
| MeanFlow-XL/2+ | 2 | 2.20 |

2.20 FID **几乎追平** DiT-XL/2 (2.27, 250-NFE) 和 SiT-XL/2 (2.06, 250-NFE)！这意味着 few-step 已经可以媲美 many-step。

### Ablation 关键发现（Tab. 1）

**(a) Ratio of $r \neq t$：**
- 0%（纯 FM）：FID 328.91，完全失败
- 25%：61.06（最优）
- 100%：67.32（仍可用）

说明：MeanFlow 需要在 $r=t$（学习 instantaneous）与 $r \neq t$（传播 average）之间平衡。

**(b) JVP tangent：**
- $(v, 0, 1)$ 正确：61.06
- $(v, 0, 0)$：268.06（丢失 $\partial_t$）
- $(v, 1, 0)$：329.22（错乱）
- $(v, 1, 1)$：137.96

证明 JVP 公式正确性至关重要。$\partial_r u, \partial_t u$ 虽然是 1 维 tangent，但作用关键。

**(c) Positional embedding of $(r, t)$：**
- $(t, r)$：61.75
- $(t, t-r)$：61.06（最优）
- $(t, r, t-r)$：63.98
- $t-r$ only：63.13

**仅 embed interval $t-r$ 就能 work**！说明时间区间信息是核心。

**(d) Time sampler：** lognorm(-0.4, 1.0) 最优（61.06），与 Flow Matching 经验一致。

**(e) Loss metric $p$：** $p=1$（标准 L2 自适应权重）最优 61.06；$p=0$（纯 L2）79.75；$p=0.5$（Pseudo-Huber）63.98。

**(f) CFG scale $\omega$：**
- 1.0（无 CFG）：61.06
- 3.0：15.53（最优）
- 5.0：20.75（过度）

CFG 在 1-NFE 下依然有效。

### Scalability（Fig. 4）

B → M → L → XL，FID 单调下降，呈现 Transformer-based diffusion 的典型 scaling behavior。

### CIFAR-10（Tab. 3）

| Method | precond | NFE | FID |
|--------|---------|-----|-----|
| iCT [43] | EDM | 1 | 2.83 |
| ECT [15] | EDM | 1 | 3.60 |
| sCT [31] | EDM | 1 | 2.97 |
| IMM [52] | EDM | 1 | 3.20 |
| MeanFlow | none | 1 | 2.92 |

无 EDM preconditioner 也有竞争力。

---

## 架构细节（Appendix A）

Backbone 沿用 DiT（ViT + adaLN-Zero），不做改动。$(r, t)$ 通过各自的位置编码 + 2-layer MLP + sum 注入。Latent space 是 VAE 的 $32 \times 32 \times 4$（256×256 image）。

XL/2 配置：676M params, 28 layers, 1152 hidden dim, 16 heads, patch 2×2, 240 epochs, Adam lr=1e-4, EMA 0.9999。

CFG triggered 时间区间在 XL/2 上是 $[0, 0.75]$，XL/2+ 是 $[0.3, 0.8]$——表明 CFG 主要在中间时间步发挥作用。

---

## Sufficiency 证明（Appendix B.3）

定义 displacement field $S(z_t, r, t) = (t-r) u(z_t, r, t)$。一般而言，导数相等只推出积分相等 up to 常数：

$$\frac{d}{dt} S = v \implies S + C_1 = \int_r^t v \, d\tau + C_2$$

但 $S|_{t=r} = 0$ 且 $\int_r^r v \, d\tau = 0$，所以 $C_1 = C_2$，从而 MeanFlow Identity 与原定义等价。

**关键点**：建模 $u$ 而非 $S$，是因为 $u$ 的形式 $(t-r) u$ 自动满足 $S|_{t=r}=0$ 边界条件。直接参数化 $S$ 需要显式约束。

---

## 我的几点 Intuition

1. **MeanFlow 的哲学**：与其让网络学一个"局部切向量"然后费力积分，不如让网络直接学"区间平均位移向量"。这把积分的负担从推理时转移到训练时——训练时通过 JVP 隐式学习如何积分。

2. **与 Consistency Models 的本质区别**：CM 是约束网络输出在轨迹上不变；MF 是约束网络输出满足一个 PDE（MeanFlow Identity）。前者是网络性质，后者是 field 性质。Field 性质的好处：最优解不依赖网络，所以可以稳定训练、不需要 curriculum。

3. **为什么不需要 distillation**：distillation 类方法（progressive distillation, DMD, etc.）需要先有一个 multi-step teacher。MeanFlow 直接从 conditional velocity $v_t = \epsilon - x$（这是 closed-form 的）出发，加上 JVP 修正项，就能让网络学到 average velocity。Teacher 在哪里？Teacher 就是 $v_t$ 本身——它是免费的 ground truth。

4. **JVP 的角色**：它让网络"意识到"自己在时间维度上是被积分的。$\partial_z u$ 捕捉的是"如果我移动 $z_t$ 一点点，average velocity 会怎么变"，$\partial_t$ 捕捉"如果 $t$ 变一点点，average velocity 会怎么变"。这两者结合 $v$（当前点的瞬时方向）告诉网络：要满足 MeanFlow Identity，target 应该是什么。

5. **多尺度物理类比**：论文 conclusion 提到这类似于 physics 中的多尺度模拟。Instantaneous velocity 是微观量，average velocity 是宏观量。MeanFlow 是在告诉网络：你不需要模拟每一步微观演化，你只需要学一个 macroscopic closure。这与 Renormalization Group、coarse-graining 的思想相通。

---

## Reference Links

- 论文原文（arXiv 应该会 soon）: https://arxiv.org/abs/2506.07507 (相近的 Flow Map Matching)
- Flow Matching 原始 paper (Lipman et al. ICLR 2023): https://arxiv.org/abs/2210.02747
- Consistency Models (Song et al. ICML 2023): https://arxiv.org/abs/2303.01469
- Improved Consistency Training (iCT, Song & Dhariwal ICLR 2024): https://arxiv.org/abs/2310.14189
- Consistency Models Made Easy (ECT, Geng et al.): https://arxiv.org/abs/2406.14548
- Shortcut Models (Frans et al. ICLR 2025): https://arxiv.org/abs/2410.12557
- Inductive Moment Matching (IMM, Zhou et al.): https://arxiv.org/abs/2503.07565
- Flow Map Matching (Boffi et al.): https://arxiv.org/abs/2406.07507
- DiT (Peebles & Xie CVPR 2023): https://arxiv.org/abs/2212.09748
- SiT (Ma et al. ECCV 2024): https://arxiv.org/abs/2401.01408
- Stable Diffusion 3 / Rectified Flow Transformers (Esser et al.): https://arxiv.org/abs/2403.03206
- Flow Matching guide and code (Lipman et al.): https://arxiv.org/abs/2412.06264
- An Introduction to Flow Matching (Cambridge MLG blog): https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html
- Classifier-Free Guidance (Ho & Salimans): https://arxiv.org/abs/2207.12598
- REPA (Yu et al. ICLR 2025): https://arxiv.org/abs/2410.06985
- JAX documentation: https://jax.readthedocs.io/
- PyTorch torch.func.jvp: https://pytorch.org/docs/stable/func.html

---

## 总结一句话

MeanFlow 把 one-step generation 从"用 heuristic 约束网络行为"升级为"学一个有 well-defined 微分方程的 ground-truth field"，用 JVP 在训练时隐式完成积分，推理时单步直出。XL/2 在 ImageNet 256×256 from scratch 达到 3.43 FID（1-NFE），几乎追平 250-NFE 的 multi-step 模型。这是 consistency-style 方法的 principled 升级。
