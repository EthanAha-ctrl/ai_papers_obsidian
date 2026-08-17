---
source_pdf: SHAPE OPTIMIZATION AND HOMOGENIZATION.pdf
paper_sha256: 6e61e20b4ce0ab4cc5318338e244b84e67104083991bf805beb109976181a3fc
processed_at: '2026-08-12T05:34:13-07:00'
target_folder: Math
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 - Shape Optimization 的故事

好，Karpathy，我换个人话版本，但该有的数学我还是要给你。

## 故事的开头

想象你是个造桥工程师。你有一块铁，体积有限，要造一个最"硬"的结构——在外力下变形最少。这个问题看起来简单：不就是找个好形状嘛。

数学上写成：

$$\min_\Omega J(\Omega), \quad J(\Omega) = \int_{\Gamma_N} \mathbf{g} \cdot \mathbf{u}\, dx$$

- Ω: 形状（你造的那个铁块的几何）
- $\mathbf{g}$: 外力
- $\mathbf{u}$: 变形（PDE的解）
- $J$ 越小越硬

**这叫 compliance minimization**，是结构优化的圣杯问题。

## 第一个反转：最优形状不存在

你跑算法，让它找最优形状。算法说："好，我挖个洞试试。" 然后又挖一个，又挖一个... 孔越挖越多，越挖越小。每次挖完都更硬一点。然后算法就疯了——它想要无限多个无限小的孔。

数学上，minimizing sequence $\Omega_n$ 满足 $J(\Omega_n) > J(\Omega_{n+1})$，但是这个序列**不收敛到任何 classical 形状**。极限是个"无限细分"的东西，没法用普通的集合描述。

这叫 **ill-posedness**。Murat 1972 年第一个看出这个问题。

参考: Murat 1972 论文 https://www.sciencedirect.com/science/article/pii/0022247X72901863

## 为什么会这样？一个直觉

看 compliance 的对偶形式：

$$c(\chi) = \min_{\text{div}\,\boldsymbol{\tau}=0} \int_D (\chi A)^{-1}\boldsymbol{\tau}\cdot\boldsymbol{\tau}\, dx$$

如果你交换两个 min（对 $\chi$ 和对 $\boldsymbol{\tau}$），得到一个能量密度：

$$f(\boldsymbol{\tau}) = \begin{cases} A^{-1}\boldsymbol{\tau}\cdot\boldsymbol{\tau} + \ell & \boldsymbol{\tau} \neq 0 \\ 0 & \boldsymbol{\tau} = 0 \end{cases}$$

这个 $f$ 在 0 处有跳跃。**非凸且非 quasi-convex**。在 calculus of variations 里，这是 minimizer 不存在的标志（Morrey 准则）。

直觉解释：能量在 0 和非 0 之间有"gap"，最小化序列会**疯狂振荡**，在 0 处取"无材料"，在非 0 处取"有材料"，averaging 出中间值。这个 averaged 状态比任何 classical 解都好，但它本身不是 classical 解。

## 救援方案 A：Hadamard 方法

老的思路（1907年）：用边界扰动参数化形状。

$$\Omega = (\text{Id} + \boldsymbol{\theta})\Omega_0$$

形状导数（怎么变形最降低 J）：

$$J'(\Omega)(\boldsymbol{\theta}) = -\int_\Gamma A e(\mathbf{u})\cdot e(\mathbf{u})\, \boldsymbol{\theta}\cdot\mathbf{n}\, ds$$

- $A e(\mathbf{u})\cdot e(\mathbf{u})$: 局部应变能密度（哪里"应力大"）
- $\boldsymbol{\theta}\cdot\mathbf{n}$: 法向位移

**下降方向**：$\boldsymbol{\theta} = (A e(\mathbf{u})\cdot e(\mathbf{u}) - \ell)\mathbf{n}$

**问题**：
1. 拓扑锁死——孔的个数永远不变
2. 3D 要 remesh，贵死
3. 数值上陷入 local minima

直觉：Hadamard 是"修边界"的工具，但你想要的是"长出新孔"，它做不到。

参考: Sokolowski-Zolesio 1992 https://link.springer.com/book/10.1007/978-3-642-87652-6

## 救援方案 B：Homogenization（Allaire 主推）

核心 insight：**承认"复合材料"作为合法的形状**。

每点不止是"有材料"或"没材料"，可以是"50% 密度的层状复合材料"。这样 admissible set 变得足够大，minimizing sequence 就收敛了。

数学上，把 characteristic function $\chi \in \{0,1\}$ 放松成 density $\theta \in [0,1]$，并附上一个 homogenized tensor $A^*$：

$$\chi_\varepsilon \rightharpoonup^* \theta, \quad (1-\chi_\varepsilon)A^0 + \chi_\varepsilon A^1 \xrightarrow{H} A^*$$

$A^*$ 是"等效宏观弹性张量"——一个 composite 的 effective 行为。

**Relaxation 的精髓**：
1. 新问题有 minimizer
2. 原 minimizer = minimizing sequence 的极限
3. infimum 值不变

所以你没改变问题，你只是允许问题有解。

## 复合材料的麻烦：G_θ 是什么？

定义 $G_\theta$ = 两相 $A^0, A^1$ 按 $(1-\theta), \theta$ 比例混合能产生的**所有** homogenized tensor 的集合。

- 在 2D conductivity 下，$G_\theta$ 完全已知（Hashin-Shtrikman bounds）
- 在 elasticity 下，**至今未知**——open problem！

边界是：

$$\lambda_\theta^- = \left(\frac{\theta}{A^1} + \frac{1-\theta}{A^0}\right)^{-1} \leq \lambda_i(A^*) \leq \lambda_\theta^+ = \theta A^1 + (1-\theta)A^0$$

- $\lambda_i$: $A^*$ 的特征值
- $\lambda_\theta^-$: 串联（harmonic mean）
- $\lambda_\theta^+$: 并联（arithmetic mean）

## Compliance 的"奇迹"

为什么 Allaire 这套在 elasticity 里能 work？因为 compliance 是 special 的。它有双 min 结构：

$$J = \min_{A^*\in G_\theta} \min_{\text{div}\,\boldsymbol{\tau}=\mathbf{f}} \int_D (A^*)^{-1}\boldsymbol{\tau}\cdot\boldsymbol{\tau}\, dx$$

这个双 min 让我们能用 **sequential laminates**（一种特殊的简单复合结构）替代整个 $G_\theta$。Sequential laminates 的 $A^*$ 有显式公式！

直觉：compliance 是个"对偶 friendly"的目标，它不 care 复合材料的内部细节，只 care 等效刚度。所以最简单的微结构（rank-1 或 rank-2 laminate）就够了。

**数值算法**：

```
1. Init (θ_0, A_0*)
2. For k:
   a. Solve elasticity with current design → stress τ_k
   b. Update (θ, A*) from τ_k via explicit optimality
3. Penalize: θ_pen = (1 - cos(πθ))/2  ← 拉回 {0,1}
```

惩罚公式 $\theta_{pen} = \frac{1-\cos(\pi\theta)}{2}$ 是个把 [0,1] 映到 [0,1] 但挤压中间值的 map——θ=0→0, θ=1→1, θ=0.5→0.5（不动点）。

**关键结果**：数值上找到的都是 **global minima**（对 affine boundary conditions 严格证明）。

参考 Allaire 拓扑优化图集: http://www.cmap.polytechnique.fr/~optopo/homog_en.html

## Homogenization 何时失效？

如果 objective 依赖 gradient（比如最大应力）：

$$J(\chi) = \int_D j(\nabla u)\, dx$$

这类泛函在 $H^1$ 弱拓扑下不连续。Homogenization 不够用——**需要 correctors**（描述微观尺度修正的量）。

这是 paper 里点出的 open problem。工程上极重要（防止塑性变形、断裂），但理论上几乎没人做出来。

参考: Grabovsky 2013 https://epubs.siam.org/doi/book/10.1137/1.9781611971973

## Small Amplitude 救场

Allaire 的 hack：假设两相性质接近：

$$A^1 = A^0(1+\eta), \quad |\eta| \ll 1$$

物理意义：你只是在基体材料里"轻轻掺杂"一点另一种材料。低对比度场景在工程上常见（比如轻微合金化）。

于是 $A(\mathbf{x}) = A^0(1+\eta\chi)$。位移做 Taylor 展开：

$$u = u^0 + \eta u^1 + \eta^2 u^2 + \mathcal{O}(\eta^3)$$

代入 PDE 逐阶求解：

- $u^0$: 与 $\chi$ 无关，基体响应
- $u^1$: 线性于 $\chi$
- $u^2$: 二次于 $\chi$

目标函数：

$$J_{sa}(\chi) = \int j(\nabla u^0) + \eta\int j'(\nabla u^0)\cdot\nabla u^1 + \eta^2\int\left[j'(\nabla u^0)\cdot\nabla u^2 + \frac{1}{2}j''(\nabla u^0)\nabla u^1\cdot\nabla u^1\right]$$

**仍然 ill-posed！** 但现在只到二阶，可以用更轻量的工具处理。

## H-measures - 小而美的工具

Gérard 和 Tartar 发明的 H-measure 是一种 "defect measure"，捕捉 weakly convergent 序列"为什么没强收敛"。

直觉：你有个序列 $u_\varepsilon \rightharpoonup 0$ in $L^2$，但它没强收敛——它在振荡。H-measure $\mu(\mathbf{x}, \boldsymbol{\xi})$ 告诉你"在位置 $\mathbf{x}$ 处，沿方向 $\boldsymbol{\xi}$ 振荡了多少"。

数学上：

$$\lim_{\varepsilon\to 0}\int q(u_\varepsilon)\cdot\bar{u}_\varepsilon\, dx = \iint q_{ij}(\mathbf{x},\boldsymbol{\xi})\mu_{ij}(d\mathbf{x},d\boldsymbol{\xi})$$

- $q$: degree-0 pseudo-differential operator
- $\boldsymbol{\xi} \in \mathbb{S}^{N-1}$: 振荡方向（单位球面上的点）
- $\mu_{ij}$: Hermitian 非负矩阵测度

**对 characteristic function 的关键 lemma**（Kohn-Tartar）：

$$\mu(\mathbf{x}, \boldsymbol{\xi}) = \theta(\mathbf{x})(1-\theta(\mathbf{x}))\nu(\mathbf{x}, \boldsymbol{\xi})$$

- $\theta(\mathbf{x})$: 局部密度
- $\theta(1-\theta)$: "二项方差"，捕捉摆动幅度
- $\nu(\mathbf{x}, \boldsymbol{\xi})$: 单位球面上的概率测度，捕捉振荡方向分布

直觉：在每点，振荡的"幅度"由 $\theta(1-\theta)$ 决定（密度 0.5 时最大振荡，密度 0 或 1 时无振荡），"方向"由 $\nu$ 决定。

参考: Gérard 1991 https://link.springer.com/article/10.1007/BF01696476

## 放松后的问题

Relaxed state equations 变成：

$$-\text{div}(A^0\nabla u^2) = \text{div}(\theta A^0\nabla u^1) - \text{div}(\theta(1-\theta)A^0 M A^0\nabla u^0)$$

最后一项 $\theta(1-\theta)A^0 M A^0$ 是 **homogenization 的痕迹**！

其中：

$$M(\mathbf{x}) = \int_{\mathbb{S}^{N-1}} \frac{\boldsymbol{\xi}\otimes\boldsymbol{\xi}}{A^0\boldsymbol{\xi}\cdot\boldsymbol{\xi}}\nu(\mathbf{x}, d\boldsymbol{\xi})$$

- $\boldsymbol{\xi}\otimes\boldsymbol{\xi}$: 方向投影矩阵
- 分母 $A^0\boldsymbol{\xi}\cdot\boldsymbol{\xi}$: 沿 $\boldsymbol{\xi}$ 的方向刚度

Relaxed objective 多出一项：

$$\eta^2\int \theta(1-\theta) A^0 N A^0\nabla u^0\cdot\nabla u^0\, dx$$

其中 $N$ 包含 $j''(\nabla u^0)$——这就是 **correctors 的痕迹**！

新设计变量：$(\theta, \nu)$——密度 + 振荡方向分布。

## 主定理：Rank-One Laminate 总是最优

**Theorem**: Relaxed problem 的 minimizer 中，$\nu$ 是 Dirac mass。即 $\nu = \delta_{\boldsymbol{\xi}^*}$，所有振荡集中在**单一方向**。

**证明的绝妙之处**：

$J_{sa}^*$ 对 $\nu$ 是 **affine** 的（因为 $u^2$ 和 $N$ 都 affine 依赖 $\nu$）。

凸集 $\mathcal{P}(\Omega, \mathbb{S}^{N-1})$ 上，affine 函数在 **extremal points** 取最小。这个凸集的 extremal points 就是 **Dirac masses**！

直觉：在 small amplitude regime 下，复杂微结构（rank-2, rank-3 laminates）不会比简单的 rank-1（单向层状）更好。低对比度让最优解退化到最简单的几何。

**这给计算带来巨大好处**：
- 微结构方向 $\boldsymbol{\xi}^*$ 只需算一次（与 $\theta$ 无关）
- 刚度矩阵 $A^0$ 是常数 → Cholesky 分解只做一次
- 后续迭代只需 back-substitution → 极快

## 跟你熟悉领域的联想

Karpathy，这跟你做的几个东西有非平凡的联系：

### 1. NAS + DARTS

DARTS 用 softmax over candidate operations 做"连续 relaxation"，最后 argmax 回离散。这跟 homogenization relaxation + penalization 是**同一个 trick**：

$$\text{DARTS: } \alpha_i \in \mathbb{R}^K \to \text{softmax} \to \text{argmax}$$
$$\text{Homog: } \chi \in \{0,1\} \to \theta \in [0,1] \to \theta_{pen}$$

Allaire 的 $\theta_{pen} = (1-\cos(\pi\theta))/2$ 是一种"温度退火"——把中间值挤到边界。DARTS 也做类似的事（L1 regularization 或温度退火）。

参考 DARTS: https://arxiv.org/abs/1806.09055

### 2. Lottery Ticket + Pruning

"Lottery ticket hypothesis" 说稠密网络里有稀疏子网络能 train 到同样性能。这跟 shape optimization 的"找最小材料实现最大刚度"在哲学上同源。

更深层：magnitude pruning 的 minimizing sequence（越来越稀疏）和 shape optimization 的 minimizing sequence（越来越多小孔）在数学上**可能可以同构**——都是 quasi-convexity 缺失导致的 ill-posedness。

参考 Lottery Ticket: https://arxiv.org/abs/1803.03635

### 3. Differentiable Physics + Taichi/JAX-MD

这些工具把 PDE solver 嵌进 gradient-based optimizer，本质是 Hadamard 方法的现代 GPU 实现。它们继承了同样的缺陷：**拓扑锁死**。如果你用 Taichi 做 fluid-shape co-design，初始 mesh 决定了能 reach 哪些拓扑。

一个有趣的 open direction：用 **implicit neural representations**（SIREN, NeRF）替代 mesh 做 shape optimization。连续场表示可以绕过拓扑锁死，但理论性质还没人研究过。

参考 Taichi: https://github.com/taichi-dev/taichi

### 4. Score Matching + Generative Models

H-measure 的"defect measure 捕捉振荡方向"思想，跟 generative model 的 latent space 结构有种结构上的呼应。如果你把 minimization sequence $\chi_\varepsilon$ 看成"从粗到细的生成过程"，H-measure 就是在描述这个过程的"模式分布"。

我没看到有人 explicitly 做这个 connection，但直觉上感觉有东西在那。

### 5. Bayesian NN 的 Weight Posterior Geometry

$G_\theta$（所有合法的 homogenized tensor 的集合）的刻画难题，跟 BNN 里"刻画 weight posterior 的几何结构"有种结构上的类似——都是"刻画某个函数空间里 valid 对象的集合"。

### 6. Sub-grid Scale Modeling (LES, Turbulence)

H-measure 在位置-方向空间上分布，LES 的 sub-grid stress 也在描述"未解析尺度的能量分布"。两者都是"用低维 measure 描述高维未解析动力学"的尝试。可能有人做过 H-measure 在 turbulence 上的应用，但我没见到。

## 总结 - 一句话讲清楚

**Shape optimization 的病在 quasi-convexity 缺失；homogenization 把解空间扩大到复合材料来治这个病；compliance 目标让最优复合材料简化为 sequential laminates；低对比度假设让 H-measure 上场，rank-1 laminate 总是 extremal，计算变得极快。**

更深层的智慧：**当一个问题 ill-posed 时，正确的反应不是加约束（Hadamard 方案），而是承认"广义解"（homogenization 方案）**。这种"扩大问题让解存在"的思路，是 20 世纪 PDE 理论的精髓（ distributions, weak solutions, homogenization 都是这一脉），也应该是 ML 理论的未来方向。

参考 Allaire 专著: https://link.springer.com/book/10.1007/978-1-4612-0193-5

希望这个版本能 build 起你的 intuition。如果你想 drill down 任何部分（sequential laminates 的显式公式、H-measure 的 Fourier 分析推导、跟 ML 的具体 cross-pollination idea），告诉我就好。

---

# Grégoire Allaire - Shape Optimization and Homogenization 深度讲解

你好 Karpathy! 这篇 paper 是 Allaire 2010 年在 IHES Schlumberger Chaire 的讲座，浓缩了他 2001 年那本经典专著 *Shape Optimization by the Homogenization Method* 的核心思想，外加他和 Gutierrez、Kelly 关于 small amplitude regime 的新工作。这份材料把 PDE、calculus of variations、optimization、optimal control 和 scientific computing 编织在一起，给你展现一个完整的 shape optimization 宇宙。

## 1. Big Picture: 这篇 paper 到底在做什么

核心动机来自结构力学：给定一个"工作区域" D ⊂ ℝ^d 和体积预算 V₀，找一个最"刚性"的子集 Ω ⊂ D（在给定 load 下）。刚性用 **compliance** 度量：

$$J(\Omega) = \int_{\Gamma_N} \mathbf{g} \cdot \mathbf{u}\, dx$$

- $\Gamma_N$: 施加 traction load $\mathbf{g}$ 的边界部分（固定）
- $\mathbf{u}$: 位移场，是 elasticity PDE 的解
- $J$ 越小 → 结构越刚硬（在外力下变形越少）

这个看似简单的问题有个深刻的"病"：**一般不存在最优解**。Allaire 整篇 paper 就在解释为什么不存在，以及如何通过 **homogenization relaxation** 恢复 well-posedness，最后讨论 small amplitude 这种特殊 regime 下 H-measures 怎么把整个问题变得更 tractable。

Reference: Allaire 2001 book https://link.springer.com/book/10.1007/978-1-4612-0193-5

## 2. 模型问题 - Linearized Elasticity

### 状态方程

形状 Ω ⊂ ℝ^d，边界被分成三块：

$$\partial\Omega = \Gamma \cup \Gamma_N \cup \Gamma_D$$

- $\Gamma_D$: Dirichlet 边界（固定，u=0）
- $\Gamma_N$: Neumann 边界（施加 load g，固定）
- $\Gamma$: free boundary，**这是唯一要优化的部分**

位移场 $\mathbf{u}: \Omega \to \mathbb{R}^d$ 满足：

$$\begin{cases} -\text{div}(A e(\mathbf{u})) = 0 & \text{in } \Omega \\ \mathbf{u} = 0 & \text{on } \Gamma_D \\ (A e(\mathbf{u}))\mathbf{n} = \mathbf{g} & \text{on } \Gamma_N \\ (A e(\mathbf{u}))\mathbf{n} = 0 & \text{on } \Gamma \end{cases}$$

各变量含义：
- $e(\mathbf{u}) = \frac{1}{2}(\nabla\mathbf{u} + \nabla\mathbf{u}^t)$: 应变张量，上标 $t$ 表示转置
- $\sigma = A e(\mathbf{u})$: 应力张量
- $A$: 各向同性弹性张量（4阶），由 Lamé 参数 $\lambda, \mu$ 决定：$A_{ijkl} = \lambda\delta_{ij}\delta_{kl} + \mu(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})$
- $\mathbf{n}$: 单位外法向

### Admissible set

$$\mathcal{U}_{ad} = \{\Omega \subset D \text{ open} : \Gamma_D \cup \Gamma_N \subset \partial\Omega, \int_\Omega dx = V_0\}$$

V₀ 是体积预算。问题就是 $\inf_{\Omega \in \mathcal{U}_{ad}} J(\Omega)$。

## 3. 为什么问题 ill-posed - 直觉与数学

### Murat 反例的直觉

考虑一块板，要它最刚又最轻。算法自然演化出"挖越来越多、越来越小的孔"的序列（minimizing sequence $\Omega_n$，$J(\Omega_n) \to J(\Omega_{n+1})$ 递减）。极限状态下，材料在每一点都被无限细分，这种"广义形状"不是任何 classical Ω 能描述的——它在每点都有一个微观结构。

参考 Murat 1972 原始论文和 Murat-Tartar 的 H-convergence: https://link.springer.com/chapter/10.1007/978-3-642-61284-3_4

### Calculus of variations 视角 - 真正的根源

用 characteristic function $\chi$ 参数化形状：

$$A(\mathbf{x}) = (1-\chi(\mathbf{x}))A^0 + \chi(\mathbf{x})A^1$$

引入 Lagrange 乘子 $\ell$ 处理体积约束：

$$\inf_{\chi \in L^\infty(D;\{0,1\})} \left(J(\chi) = c(\chi) + \ell\int_D \chi\, d\mathbf{x}\right)$$

**关键步骤**：compliance 有对偶能量表示：

$$c(\chi) = \min_{\substack{\text{div}\,\boldsymbol{\tau} = 0 \text{ in } D \\ \boldsymbol{\tau}\mathbf{n} = \mathbf{g} \text{ on } \partial D}} \int_D (\chi(\mathbf{x})A)^{-1} \boldsymbol{\tau}\cdot\boldsymbol{\tau}\, d\mathbf{x}$$

这里 $\boldsymbol{\tau}$ 是满足平衡方程的 stress field。注意 $\boldsymbol{\tau} \equiv 0$ 在 Ω 外（material 外没应力）。

**交换两个 min**（这是 Tartar 的绝招）：

$$\inf_{\text{div}\,\boldsymbol{\tau}=0 \text{ in } D} \int_D f(\boldsymbol{\tau})\, d\mathbf{x}$$

其中：

$$f(\boldsymbol{\tau}) = \begin{cases} A^{-1}\boldsymbol{\tau}\cdot\boldsymbol{\tau} + \ell & \text{if } \boldsymbol{\tau} \neq 0 \\ 0 & \text{if } \boldsymbol{\tau} = 0 \end{cases}$$

这个 $f$ 既不是 convex，也不是 Morrey 意义下的 quasi-convex。**Quasi-convexity 是 vectorial calculus of variations 中存在 minimizer 的充要条件**（Morrey 1952，Dacorogna 的书有详细讨论 https://link.springer.com/book/10.1007/978-3-642-11840-1）。

直觉上：$f$ 在 0 和非 0 之间跳跃，制造一种 "微细振荡更便宜" 的现象。最小化序列会在每点交替取 0 和非 0，averaging 到一个介于两者之间的值，而这个 averaged 行为用单一 $\chi$ 实现不了。

## 4. 两种补救方案

### 方案 A: Hadamard 方法（局部 shape derivative）

参数化形状：

$$\Omega = (\text{Id} + \boldsymbol{\theta})\Omega_0, \quad \boldsymbol{\theta} \in W^{1,\infty}(\mathbb{R}^d;\mathbb{R}^d)$$

要求 $\|\boldsymbol{\theta}\|_{C^1} < 1$ 保证是 diffeomorphism → **拓扑不变**（孔的数量固定）。

**Shape derivative of compliance**：

$$J'(\Omega)(\boldsymbol{\theta}) = -\int_\Gamma A e(\mathbf{u})\cdot e(\mathbf{u})\, \boldsymbol{\theta}\cdot\mathbf{n}\, ds$$

- $A e(\mathbf{u})\cdot e(\mathbf{u}) = \sigma : e(\mathbf{u})$: 局部应变能密度
- $\boldsymbol{\theta}\cdot\mathbf{n}$: 法向位移幅度

**Steepest descent 方向**：

$$\boldsymbol{\theta} = (A e(\mathbf{u})\cdot e(\mathbf{u}) - \ell)\mathbf{n}$$

$\ell$ 是 Lagrange 乘子对应体积约束。

### Hadamard 方法的局限

1. **拓扑不变** — 孔的数量由初始 design 决定，永远改不了
2. **3D 大变形需要 remesh** — 代价巨大
3. **数值上常陷入 local minima** — 是 ill-posedness 的数值症状
4. 对 initial design 敏感，对 mesh size 敏感

参考 Sokolowski-Zolesio 1992 https://link.springer.com/book/10.1007/978-3-642-87652-6

### 方案 B: Homogenization relaxation（Allaire 主推）

**核心思想**：扩大 admissible set 到 "generalized shapes" = composite materials。在每点引入一个 microscopic 微结构，由 density $\theta(\mathbf{x}) \in [0,1]$ 和 homogenized tensor $A^*(\mathbf{x})$ 描述。

关键 trick：把 holes 用弱材料 $A^0$ 填充，把实体用强材料 $A^1$ 填充，转化为 two-phase optimal design 问题。

$$A(\mathbf{x}) = (1-\chi(\mathbf{x}))A^0 + \chi(\mathbf{x})A^1, \quad \chi(\mathbf{x}) \in \{0, 1\}$$

## 5. Homogenization 的细节

### H-convergence (Murat-Tartar)

考虑 minimizing sequence $\chi_\varepsilon$，$\varepsilon \to 0$。存在子序列使：

$$\chi_\varepsilon \rightharpoonup^* \theta \text{ in } L^\infty(D;[0,1])$$
$$(1-\chi_\varepsilon)A^0 + \chi_\varepsilon A^1 \xrightarrow{H} A^*$$

H-convergence 的含义：对任意 $f \in L^2(D)$，状态方程的解 $u_\varepsilon \rightharpoonup u$ in $H^1(D)$，其中 $u$ 满足 homogenized 方程：

$$\begin{cases} -\text{div}(A^*\nabla u) = f & \text{in } D \\ u = 0 & \text{on } \Gamma_D \\ A^*\nabla u \cdot \mathbf{n} = 0 & \text{on } \Gamma_N \end{cases}$$

### G_θ - 一个未完全解决的问题

**$G_\theta$ = 所有由两相 $A^0, A^1$ 按比例 $(1-\theta), \theta$ 混合而成的 composite 的有效张量集合**。

在 conductivity 设定下（2D），$G_\theta$ 完全被 Hashin-Shtrikman bounds 刻画：

$$\lambda_\theta^- = \left(\frac{\theta}{A^1} + \frac{1-\theta}{A^0}\right)^{-1}, \quad \lambda_\theta^+ = \theta A^1 + (1-\theta)A^0$$

其中 $\lambda_1, \lambda_2$ 是 $A^*$ 的特征值，且需满足 $\lambda_\theta^- \leq \lambda_i \leq \lambda_\theta^+$（还要更细的优化条件）。

在 elasticity 下，$G_\theta$ 至今 **未完全已知**！这是 open problem。

### Relaxed formulation

$$\min_{(\theta, A^*) \in \mathcal{U}_{ad}^*} J(\theta, A^*) = \int_D j(u)\, d\mathbf{x}$$

$$\mathcal{U}_{ad}^* = \{(\theta, A^*) : 0 \leq \theta \leq 1, A^* \in G_\theta, \int_D \theta\, d\mathbf{x} = V_0\}$$

**Relaxation 的真意**（三个性质）：
1. 存在 minimizer $(\theta, A^*)$
2. 任何 minimizer 是 minimizing sequence $\chi_\varepsilon$ 的极限
3. 任何 minimizing sequence 收敛到某个 minimizer

注意 relaxation **没有改变 infimum 值** — 同一个问题的"延拓版本"。

### Compliance 的 miracle

$$J(\theta, A^*) = \int_D \mathbf{f}\cdot\mathbf{u}\, d\mathbf{x} = \min_{\text{-div}\,\boldsymbol{\tau}=\mathbf{f}} \int_D (A^*)^{-1}\boldsymbol{\tau}\cdot\boldsymbol{\tau}\, d\mathbf{x}$$

这是个 **双重 minimization**（对 design 参数 $A^*$ 和 stress field $\boldsymbol{\tau}$）。在 compliance 这种特殊结构下，可以用 $L_\theta$（sequential laminates）替代整个 $G_\theta$，且 sequential laminates 的 homogenized properties 是 **显式** 的！

Sequential laminates: 逐层堆叠两相，每层方向由单位向量 $\mathbf{e}_i$ 决定，整体比例 $\theta$ vs $(1-\theta)$。优化条件直接从 stress $\boldsymbol{\tau}$ 算出最优 lamination 方向和层数。

## 6. Numerical Algorithm

```
1. Initialize (θ_0, A_0*)
2. For k = 0, 1, 2, ...:
   a. Solve linear elasticity with (θ_k, A_k*) to get stress τ_k
   b. Update (θ_{k+1}, A_{k+1}*) using explicit optimality conditions based on τ_k
3. Penalize: θ_pen = (1 - cos(π θ_opt)) / 2
```

Penalization 公式 $\theta_{pen} = \frac{1-\cos(\pi\theta_{opt})}{2}$ 把 [0,1] density 拉回 {0,1}：θ=0→0, θ=1→1, θ=0.5→0.5（不动点）。cos 函数让中间密度被"挤出"。

**关键性质**：数值上只找到 **全局 minima**（对 affine boundary conditions 这已被严格证明）。shape-capturing — 同时找最优形状和拓扑。

参考 Allaire 的 online gallery: http://www.cmap.polytechnique.fr/~optopo/homog_en.html

## 7. Homogenization 方法的边界

### 它 work 的情况

- **Conductivity**: 任意 $J(\chi) = \int_D j(u)\, d\mathbf{x}$
- **Elasticity**: 仅 compliance $J = \int_D \mathbf{f}\cdot\mathbf{u}\, d\mathbf{x}$ 及相关（特征频率、robust compliance）

### 它不 work 的情况 - Open Problem

当 objective 依赖 gradient 时：

$$J(\chi) = \int_D j(\nabla u)\, d\mathbf{x}$$

这类 functional 在 $H^1$ 弱拓扑下 **不连续**。工程上极其重要 — stress limits（在塑性或断裂前）就是这种。

**问题根源**：relaxation 似乎需要的不只是 homogenized tensors，还要 **correctors**（描述微观尺度修正的量）。

只有零星结果：Grabovsky、Lipton、Pedregal、Tartar 的部分工作。

参考 Grabovsky 的书: https://epubs.siam.org/doi/book/10.1137/1.9781611971973

## 8. Small Amplitude Regime - 论文后半的精华

### 假设与展开

假设两相低对比度：

$$A^1 = A^0(1+\eta), \quad |\eta| \ll 1$$

于是：

$$A(\mathbf{x}) = A^0(1+\eta\chi(\mathbf{x}))$$

位移场做二阶展开：

$$u = u^0 + \eta u^1 + \eta^2 u^2 + \mathcal{O}(\eta^3)$$

代入状态方程，逐阶匹配：

**Order 0** (与 χ 无关):
$$-\text{div}(A^0 \nabla u^0) = f, \quad u^0|_{\Gamma_D} = 0, \quad A^0\nabla u^0\cdot\mathbf{n}|_{\Gamma_N} = g$$

**Order 1** (线性于 χ):
$$-\text{div}(A^0\nabla u^1) = \text{div}(\chi A^0 \nabla u^0), \quad u^1|_{\Gamma_D} = 0, \quad A^0\nabla u^1\cdot\mathbf{n}|_{\Gamma_N} = -\chi A^0\nabla u^0\cdot\mathbf{n}$$

**Order 2** (二次于 χ):
$$-\text{div}(A^0\nabla u^2) = \text{div}(\chi A^0 \nabla u^1)$$

边界条件类似。

### 目标函数展开

$$J(\chi) = J_{sa}(\chi) + \mathcal{O}(\eta^3)$$

其中：

$$J_{sa}(\chi) = \int_\Omega j(\nabla u^0)\, d\mathbf{x} + \eta\int_\Omega j'(\nabla u^0)\cdot\nabla u^1\, d\mathbf{x} + \eta^2\int_\Omega\left[j'(\nabla u^0)\cdot\nabla u^2 + \frac{1}{2}j''(\nabla u^0)\nabla u^1\cdot\nabla u^1\right] d\mathbf{x}$$

- $j'$: $j$ 对其 argument 的一阶导
- $j''$: Hessian
- 技术假设: $|j(\lambda)| \leq C(|\lambda|^2+1)$, $|j'(\lambda)| \leq C(|\lambda|+1)$, $|j''(\lambda)| \leq C$

**注意** $J_{sa}$ 仍然是 χ 的二次泛函，**仍然 ill-posed**！

## 9. H-measures (Gérard, Tartar) - 神奇工具

### 直觉

H-measure 是一种 **defect measure**，量化一个 weakly convergent sequence 在 $L^2$ 中"为什么没强收敛"的部分。它捕捉了振荡的"方向分布"。

设 $u_\varepsilon \rightharpoonup 0$ in $L^2(\mathbb{R}^N)^p$。H-measure 是个矩阵值测度 $\mu = (\mu_{ij})_{1 \leq i,j \leq p}$，定义在 $\mathbb{R}^N \times \mathbb{S}^{N-1}$ 上，使得对任何 degree-0 pseudo-differential operator $q$ with symbol $q_{ij}(\mathbf{x}, \boldsymbol{\xi})$：

$$\lim_{\varepsilon\to 0} \int_{\mathbb{R}^N} q(u_\varepsilon)\cdot\bar{u}_\varepsilon\, d\mathbf{x} = \int_{\mathbb{R}^N}\int_{\mathbb{S}^{N-1}} \sum_{i,j} q_{ij}(\mathbf{x},\boldsymbol{\xi})\, \mu_{ij}(d\mathbf{x}, d\boldsymbol{\xi})$$

- $\mathbb{S}^{N-1}$: 单位球面，捕获振荡方向
- $\mu_{ij} = \bar{\mu}_{ji}$ (Hermitian)
- $\sum \lambda_i\bar{\lambda}_j \mu_{ij} \geq 0$ (non-negative)

详细参考: Gérard 1991 original paper https://www.semanticscholar.org/paper/Microlocal-defect-measures-G%C3%A9rard/91bd57d68af75b7d1e21a16c04545e27f67a73f1

### Characteristic function 的 H-measure (Kohn-Tartar Lemma)

若 $\chi_\varepsilon \rightharpoonup^* \theta$ in $L^\infty(\Omega;[0,1])$，则 $(\chi_\varepsilon - \theta)$ 的 H-measure 形式必然是：

$$\mu(\mathbf{x}, \boldsymbol{\xi}) = \theta(\mathbf{x})(1-\theta(\mathbf{x}))\, \nu(\mathbf{x}, \boldsymbol{\xi})$$

其中 $\nu(\mathbf{x}, \boldsymbol{\xi})$ 是 $\mathbb{S}^{N-1}$ 上的 probability measure（每个 $\mathbf{x}$ 处）。反过来，任何这种形式的 $\mu$ 都可由某个 $\chi_\varepsilon$ 序列实现。

**直觉**：$\theta(1-\theta)$ 是 "二项分布的方差"，捕捉平均密度的"摆动幅度"。$\nu(\boldsymbol{\xi})$ 捕捉"oscillation 在哪些方向上发生"。

### Relaxed state equations

对 limit $(\theta, \nu)$：

$$\begin{cases}
-\text{div}(A^0\nabla u^0) = f \\
-\text{div}(A^0\nabla u^1) = \text{div}(\theta A^0\nabla u^0) \\
-\text{div}(A^0\nabla u^2) = \text{div}(\theta A^0\nabla u^1) - \text{div}(\theta(1-\theta)A^0 M A^0\nabla u^0)
\end{cases}$$

关键新量：

$$M(\mathbf{x}) = \int_{\mathbb{S}^{N-1}} \frac{\boldsymbol{\xi}\otimes\boldsymbol{\xi}}{A^0\boldsymbol{\xi}\cdot\boldsymbol{\xi}}\, \nu(\mathbf{x}, d\boldsymbol{\xi})$$

- $\boldsymbol{\xi}\otimes\boldsymbol{\xi}$: 并矢积，是个矩阵
- 分母 $A^0\boldsymbol{\xi}\cdot\boldsymbol{\xi}$: 沿方向 $\boldsymbol{\xi}$ 的"directional stiffness"
- $M$ 是"微结构方向偏置"的二阶矩

这个 $\theta(1-\theta)A^0 M A^0$ 项就是 **homogenization 的痕迹**！

### Relaxed objective

$$J_{sa}^*(\theta, \nu) = J_{sa}^*(\theta, \nu)\big|_{\text{without last term}} + \eta^2\int_\Omega \theta(1-\theta) A^0 N A^0 \nabla u^0 \cdot \nabla u^0\, d\mathbf{x}$$

其中：

$$N(\mathbf{x}) = \frac{1}{2}\int_{\mathbb{S}^{N-1}} \frac{j''(\nabla u^0)\boldsymbol{\xi}\cdot\boldsymbol{\xi}}{(A^0\boldsymbol{\xi}\cdot\boldsymbol{\xi})^2}\, \boldsymbol{\xi}\otimes\boldsymbol{\xi}\, \nu(\mathbf{x}, d\boldsymbol{\xi})$$

这就是 **correctors 的痕迹**！正是 homogenization 之外需要的新信息。

### Relaxed problem

$$\min_{(\theta, \nu) \in \mathcal{U}_{ad}^*} J_{sa}^*(\theta, \nu)$$

$$\mathcal{U}_{ad}^* = \{(\theta, \nu) \in L^\infty(\Omega;[0,1]) \times \mathcal{P}(\Omega, \mathbb{S}^{N-1}) : \int_\Omega \theta\, d\mathbf{x} = \Theta|\Omega|\}$$

$\mathcal{P}(\Omega, \mathbb{S}^{N-1})$: $\Omega \times \mathbb{S}^{N-1}$ 上的 Radon measure，每个 $\mathbf{x}$ 处是概率测度。

**Relaxation 的三个性质再次成立**（well-posed！）。

## 10. 主定理 - Rank-one laminates 总是最优

**Theorem**: relaxed problem 存在 minimizer 使 $\nu$ 是 Dirac mass（即 $\nu = \delta_{\boldsymbol{\xi}^*}$）。最优 Dirac 不依赖 $\theta$。

Design parameters 退化为：density $\theta(\mathbf{x})$ + 单个 lamination direction $\boldsymbol{\xi}^*(\mathbf{x})$。

### 证明的精髓

$$J_{sa}^*(\theta, \nu) \text{ 是 } \nu \text{ 的 affine 函数}$$

因为 $u^2$ 和 $N$ 都 affine 依赖 $\nu$。

凸集 $\mathcal{P}(\Omega, \mathbb{S}^{N-1})$ 上的 affine 函数在 **extremal points** 取最小，而 $\mathcal{P}$ 的 extremal points 就是 Dirac masses！

$\Rightarrow \nu = \delta_{\boldsymbol{\xi}^*}$。

**直觉**：在 small amplitude 下，最优微结构总是"rank-one laminate"（单一方向的层状结构），不需要复杂的多方向 rank-2 或更高 rank laminates。这是 small amplitude 假设带来的极大简化。

### 推广

对 elasticity、multiple loads 等同样成立。**Relaxation 和 small-amplitude ansatz 交换**。

## 11. Numerical Algorithm for Small Amplitude

```
1. Initialize θ_0
2. Compute optimal lamination direction ξ* via adjoint (once, independent of θ)
3. For k = 0, 1, ...:
   a. Compute ∇_θ J_sa^*(θ_k) via another adjoint
   b. Update θ_{k+1} by steepest descent
```

**计算优势**：
- 微结构方向 $\boldsymbol{\xi}^*$ 只算一次！
- 刚度矩阵 $A^0$ 是常数 → **Cholesky 分解只做一次**！
- 后续迭代只需 back-substitution，极其快速

FEM solver: FreeFem++ (Hecht-Pironneau)，$P_2/P_0$ 单元 https://freefem.org/

### 实验数据示例

从 paper 的图来看，对比实验：
- **Composite design** (left): 灰度图，density $\theta \in [0,1]$
- **Penalized design** (right): 黑白图，density ∈ {0,1}

测试场景：
1. Compliance minimization（标准 benchmark）
2. Strain minimization（square fixed at bottom, vertically loaded at top, η=-0.1, volume=50%）
3. Stress minimization（同上 setup）
4. Minimal dissipation of a wheel (elastodynamics)

**关键发现**：small amplitude 方法成功处理 stress/strain minimization — 这些是经典 homogenization 方法做不了的（因为依赖 gradient 的 $j(\nabla u)$）。

## 12. 综合 intuition 与开放问题

### 三种方法的对比

| Method | Topology change | Cost | 何时 work | Limit |
|--------|-----------------|------|-----------|-------|
| Hadamard | ❌ | 低（2D），高（3D remesh） | 局部 shape refinement | 锁死在初始拓扑 |
| Homogenization (full) | ✅ | 中 | compliance-type only | $G_\theta$ 在 elasticity 未知 |
| Small amplitude + H-measures | ✅ | 低（一次 Cholesky） | stress/strain + compliance | 仅低对比度 |

### 论文中最深的洞察

1. **Quasi-convexity 缺失** 是 shape optimization ill-posedness 的真正根源（Murat 1972）
2. **Compliance 的特殊结构**（双 min + 对偶能量）让它能用 sequential laminates 完全 relax — 这是"miracle"
3. **依赖 gradient 的 objective** 不能只靠 homogenized tensors relax，需要 correctors 信息
4. **Small amplitude regime** 让 H-measures 介入，正确处理了 correctors，并且 rank-one 总是最优
5. **Affine-on-convex-set 极值在 extreme points** 这个简单观察 → Dirac mass → rank-one laminates

### Open Problems

- elasticity 下完整 $G_\theta$ 的刻画
- 高对比度 + gradient-dependent objective 的 relaxation（需 correctors 理论）
- 高 porosity regime (Bourdin-Kohn 工作)
- Mixed geometry/topology optimization (with P. Frey)
- 与 level set method、topological gradient 的统一框架

### 推荐深入阅读

1. Allaire 2001 专著: https://link.springer.com/book/10.1007/978-1-4612-0193-5
2. Bendsøe-Sigmund 2003 拓扑优化圣经: https://link.springer.com/book/10.1007/978-3-662-05086-6
3. Cherkaev 2000: https://link.springer.com/book/10.1007/978-1-4757-4383-2
4. Tartar 2000 homogenization 一般理论: https://link.springer.com/book/10.1007/978-3-642-02944-3
5. Gérard H-measures 原始论文: https://link.springer.com/article/10.1007/BF01696476
6. Murat-Tartar H-convergence: https://link.springer.com/chapter/10.1007/978-3-642-61284-3_4
7. Allaire 拓扑优化 gallery: http://www.cmap.polytechnique.fr/~optopo/
8. FreeFem++: https://freefem.org/

### 给你（Karpathy）的一些联想

这篇 paper 跟你熟悉的领域有几个有趣的 connection：

1. **Neural network pruning / lottery ticket** 跟 topology optimization 在精神上同源 — 都是"找最优的稀疏结构"。Sokół et al. 的工作把 lottery ticket hypothesis 跟 magnitude-based pruning 的关系做了理论分析，跟 shape optimization 的"minimizing sequence 制造越来越多小孔"如出一辙。

2. **Neural architecture Search (NAS)** 中的 DARTS 用连续 relaxation（每个 operation 是 softmax over candidates）+ 概率 → 离散选择，这跟 homogenization relaxation + penalization 惊人地相似。Allaire 的 $\theta_{pen} = (1-\cos(\pi\theta))/2$ penalty 思路跟 NAS 的 annealing 是同一个 trick。

3. **Diffusion models 中的 score matching** 在数学结构上跟 Hadamard shape derivative 有点像：都是"沿某方向 perturb 后看 objective 变化"。

4. **Differentiable physics simulators**（如 Taichi, JAX-MD）把 PDE solver 嵌入 gradient-based optimizer，这正是 Hadamard 方法在 GPU 时代的现代实现，但它继承了同样的 topology-locking 缺陷。

5. **Implicit Neural Representations (SIREN, NeRF)** 用连续场替代离散 mesh，某种程度上绕过了 Hadamard 方法的 topology 问题，但还没人系统研究过这种 representation 在 shape optimization 上的理论性质 — 这可能是个好方向！

6. H-measures 的"defect measure 捕捉 oscillation 方向"思想，在 turbulence modeling（如 LES 的 sub-grid stress）和 generative modeling（如 latent space 的 microstructure）中应该有 analog，但据我所知还没人 explicit 建立联系。

7. **G_θ 的计算复杂度** 和 **Bayesian neural network 中 weight posterior 的几何刻画** 有种结构上的类似 — 都是"刻画某个 set of valid tensors/distributions"。

希望这个深入讲解帮你 build 起对 shape optimization 和 homogenization 的 intuition。如果你想把任何部分再展开（比如 sequential laminates 的显式公式、H-measures 的 Fourier 分析细节、或者跟 ML 的 cross-pollination），请告诉我！
