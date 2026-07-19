---
source_pdf: BEACONS BOUNDED-ERROR, ALGEBRAICALLY-COMPOSABLE NEURAL SOLVERS FOR PARTIAL
  DIFFERENTIAL EQUATIONS.pdf
paper_sha256: c5499bc4a7bccda1c5f54c3025dfadde960b4e3301050caaac8831a9d116e6de
processed_at: '2026-07-18T14:14:32-07:00'
target_folder: PINN
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BEACONS: 一篇将神经网络视为"广义数值方法"并配备形式化验证框架的 paper

## 1. 核心直觉：把 NN 当作数值方法来"驯化"

这篇 paper 最 deep 的 insight 在于它的**philosophical stance**: neural networks 不过是 classical numerical methods（finite volume, finite element, discontinuous Galerkin 等）的一种极大 generalization。在 classical 数值方法里，我们用固定 basis $\{\varphi_i\}$ 张成有限维子空间 $\mathcal{V}_h$ 来近似无穷维函数空间 $H^1(\Omega)$，而 NN 的优势在于 basis 本身（即 hidden layer 的 weights/biases 组成的 ridge functions $\psi_i$）**也是 trainable 的**。这是 paper Section 2.1 的关键 framing：

$$\mathcal{V}_h = \mathrm{span}\{\varphi_1, \dots, \varphi_N\}, \quad u_h(\mathbf{x}) = \sum_i U_i \varphi_i(\mathbf{x}) \tag{6,7}$$

对照 NN：
$$\psi_i(\mathbf{x}; W^{(1)}, \mathbf{b}^{(1)}) = \sigma\!\Big(\sum_{j=1}^d W^{(1)}_{ij} x_j + b^{(1)}_i\Big), \quad u_\theta(\mathbf{x}) = b^{(2)} + \sum_i W^{(2)}_{1i} \psi_i \tag{8,9}$$

变量含义：
- $W^{(1)} \in \mathbb{R}^{N \times d}$：hidden layer 权重矩阵，$N$ 为神经元数，$d$ 为输入维度
- $W^{(2)} \in \mathbb{R}^{1 \times N}$：output layer 权重
- $b^{(1)} \in \mathbb{R}^N, b^{(2)} \in \mathbb{R}$：偏置
- $\sigma$：activation（论文用 $\tanh$，因为需要 $C^\infty$ smooth 满足 Mhaskar 定理前提）
- $\theta \in \mathbb{R}^P, P = 1+(d+2)N$：所有可训练参数的 concatenation

如果 freeze hidden layer weights 和 biases 并令 $b^{(2)}=0$，NN 退化为 classical basis expansion。**NN 的"额外表达力"完全来自 basis 本身能 adapt**。

Reference: [Mhaskar & Poggio 2016, Analysis and Applications](https://www.worldscientific.com/doi/abs/10.1142/S0219530516400048), [Pinkus 1999, Acta Numerica](https://www.cambridge.org/core/journals/acta-numerica/article/abs/approximation-theory-of-the-mlp-model-in-neural-networks/D0B9C85C0E9F1F1A21F9D3A6F4F8C0F5)

---

## 2. 第一个核心定理：Mhaskar-Poggio 近似 bound

**Theorem 1 (Eq. 36)**: 对于 $f \in W_{r,d}$（Sobolev 空间，所有 $r$ 阶偏导都存在且连续，并满足归一化 $\|f\|_\infty + \sum_{1 \le \|\mathbf{k}\|_1 \le r} \|\mathcal{D}^{\mathbf{k}} f\|_\infty \le 1$），单隐藏层 MLP 的最佳 $L^\infty$ 近似误差为：

$$\inf_{u_\theta \in \mathcal{V}_{NN}} \|f - u_\theta\|_\infty = \mathcal{O}\!\left(N^{-r/d}\right) \tag{36}$$

- $r$：函数的 smoothness（连续可导阶数）
- $d$：domain 维度
- $N$：神经元数
- $\mathcal{D}^{\mathbf{k}} = \partial^{k_1}_{x_1} \cdots \partial^{k_d}_{x_d}$：multi-index 偏导算子，$\|\mathbf{k}\|_1 = \sum_i k_i$

**直觉 build-up**: 
- 当 $r \to \infty$（function 无限 smooth），bound $\to 0$，NN 表达力爆炸
- 当 $r \to 0$（function 不连续），bound $\to \mathcal{O}(1)$，**NN 失控**
- 当 $d$ 大（高维），curse of dimensionality 让 bound 衰减缓慢
- 这与 classical spline/Fourier approximation 的收敛 rate formula 在结构上同构

这个定理的**致命弱点**就在于：对于 hyperbolic PDE 的 shock solutions（仅 piecewise $C^n$），shock 处 $r \to 0$，bound 退化到 $\mathcal{O}(1)$。这正是 PINN 在 shock 附近发散的根本原因——它和 Gibbs/Runge 现象是**同一种数学病**。

---

## 3. 第二个核心定理：用 method of characteristics 推断 a priori smoothness

这是 paper 最漂亮的一步：**在 training domain 之外，我们仍然能知道解的 smoothness**，因为 PDE 的特性（flux 解析性质 + 初始数据）预先决定了 solution 的正则性。

### 3.1 特征线推导

一阶守恒律 $\partial_t u + \partial_x f(u) = 0$ 通过 chain rule 等价于：
$$\frac{du}{dt} = 0, \quad \frac{dx}{dt} = f'(u) \tag{40}$$

- $f'(u)$：characteristic speed（flux 的导数）
- 沿特征线 $u$ 守恒，所以 $f'(u)$ 是常数，特征线是 $x$-$t$ 平面上的**直线**

积分得到特征线方程：
$$x = x_0 + f'(u_0(x_0)) \cdot t \tag{42}$$

### 3.2 Shock 形成时间

对 $x_0$ 求偏导：
$$\frac{\partial x}{\partial x_0} = 1 + f''(u_0(x_0)) \cdot u_0'(x_0) \cdot t \tag{43}$$

- $f''(u)$：flux 的曲率（对 Burgers $f = \tfrac{1}{2}u^2$，$f'' = 1$）
- $u_0'(x_0)$：初始数据的斜率

当 $\partial x/\partial x_0 = 0$ 时，特征线交叉，解 blow-up：
$$t_\infty = \Big(\sup_{x_0 \in \mathbb{R}} \{-f''(u_0(x_0)) \cdot u_0'(x_0)\}\Big)^{-1} \tag{47}$$

**直觉**: 当 flux 是 convex（$f''>0$）且 initial data 单调递减（$u_0'<0$）时，$-f'' u_0' > 0$，characteristics 收敛，**必然形成 shock**。比如 Burgers with top-hat initial data，左右两侧都形成 shock，中间形成 rarefaction。

### 3.3 Theorem 2 的外推 bound

结合 Lemma 1（smoothness prediction）与 Theorem 1（Mhaskar bound）：

- **Case 1**：linear flux OR (convex flux + 单调递增) OR (concave flux + 单调递减) → 解对所有 $t \ge 0$ 保持 $C^n$，存在 NN 使得
$$\|u - u_\theta\|_\infty = \mathcal{O}(N^{-n/d}) \quad \forall t \ge 0 \tag{60}$$

- **Case 2**：一般情况 → 在 $t < t_\infty$ 保持 bound $\mathcal{O}(N^{-n/d})$；$t \ge t_\infty$ 退化为 $\mathcal{O}(1)$。**但**可以构造 piecewise NN：
$$\tilde{u}(t,x) = \begin{cases} u_{\theta_1}^{(1)} & x \le x_1(t) \\ u_{\theta_2}^{(2)} & x_1(t) < x \le x_2(t) \\ \vdots \end{cases} \tag{64}$$
每个 piece 都是 $C^n$，整体仍保持 $\mathcal{O}(N^{-n/d})$ 的 bound（Eq. 65）。

**关键突破**: bound 是**外推性**的——$t$ 可以远离训练区间，因为我们知道解的 smoothness 不依赖于训练数据，而依赖于 PDE 自身的解析结构。

### 3.4 推广到 vector systems（Riccati 方程）

对于 $\partial_t \mathbf{U} + \partial_x \mathbf{F}(\mathbf{U}) = \mathbf{0}$，flux Jacobian $J_\mathbf{F} = \nabla_\mathbf{U} \mathbf{F}$ 在严格双曲下可对角化，特征值 $\lambda_i$（实且互异），左右特征向量 $\mathbf{l}_i, \mathbf{r}_i$。各 wave mode 满足 **Riccati-type ODE**：
$$\frac{d\omega_i}{dt} + \alpha_i \omega_i^2 + \sum_{j \ne i} \beta_{ij} \omega_i \omega_j = 0 \tag{54}$$

- $\omega_i = \mathbf{l}_i \cdot \partial_x \mathbf{U}$：第 $i$ 个 wave 的强度
- $\alpha_i = (\nabla_\mathbf{U} \lambda_i) \cdot \mathbf{r}_i$：**genuinely nonlinear** 程度（对应 scalar 情况的 $f''$）
- $\beta_{ij}$：不同 wave 间的耦合系数

第 $i$ 个 wave 的 shock 形成时间：
$$t_{\infty,i} = \Big(\inf_{x_0} \{-\alpha_i(\mathbf{U}_0(x_0)) \cdot \omega_i(\mathbf{U}_0(x_0))\}\Big)^{-1} \tag{56}$$

对 Euler 方程：$\alpha_1 = \alpha_3 = 0$（linearly degenerate，对应 contact wave 不 steepen），$\alpha_2 \ne 0$（genuinely nonlinear，对应 acoustic waves 可形成 shock）。**这就解释了 Sod shock tube 里 contact 不会变陡，但 shock 和 rarefaction 都来自 genuinely nonlinear mode**。

Reference: [Roe 1986, Annual Review of Fluid Mechanics](https://www.annualreviews.org/doi/10.1146/annurev.fl.18.010186.002005), [Courant & Hilbert Methods of Mathematical Physics](https://onlinelibrary.wiley.com/doi/book/10.1002/9783527617542)

---

## 4. 第三个核心定理：Algebraic Composability（最关键的创新）

### 4.1 动机

shallow NN 的 bound $\mathcal{O}(N^{-n/d})$ 在 $n=0$（discontinuous）时退化。**怎么解决**？答案是 decompose：
$$u(t,x) = f(g(t,x), h(t,x)) \tag{66}$$

朴素 chain rule 看似没用：composition $f(g,h)$ 的 smoothness 是 $\min\{\alpha,\beta,\gamma\}$。但作者的关键观察：**composition 的误差 bound 不只看 smoothness，还要看 Lipschitz constant**。

### 4.2 Proposition 1 推导

设 $\|f - \tilde{f}\|_\infty = e_f$, $\|g - \tilde{g}\|_\infty = e_g$。通过 triangle inequality：
$$\|f \circ g - \tilde{f} \circ \tilde{g}\|_\infty \le \|f \circ g - f \circ \tilde{g}\|_\infty + \|f \circ \tilde{g} - \tilde{f} \circ \tilde{g}\|_\infty \tag{68-69}$$

引入 modulus of continuity：
$$\omega_f(\delta) = \sup_{|u-v| \le \delta} |f(u) - f(v)| \tag{70}$$

若 $f$ 是 $L$-Lipschitz，则 $\omega_f(\delta) \le L \delta$，得到：
$$\boxed{\|f \circ g - \tilde{f} \circ \tilde{g}\|_\infty \le e_f + L \cdot e_g} \tag{75}$$

### 4.3 这条 bound 的革命性意义

直觉是：**让 $e_g$ 很大没关系（discontinuous function 的 NN 逼近误差很大），只要让 $L$ 足够小，乘积 $L \cdot e_g$ 就能压回去**。

具体配方（Eq. 79）：
$$f(x) = \frac{\mathrm{arcsinh}(x)}{C}, \quad g(x) = \mathrm{sinh}(C \cdot u(x))$$

- $C > 0$ 是设计参数
- $f$ 的 Lipschitz 常数 $L = 1/C$（因为 $\mathrm{arcsinh}'(x) = 1/\sqrt{1+x^2}$，在 $x=0$ 处取最大值 $1$）
- $g$ 把 discontinuous $u$ 放大 $C$ 倍后通过 $\sinh$——$\sinh$ 是 smooth monotone 函数，但放大后 gradient 也放大了 $C$ 倍

**关键 non-triviality**: 选 $\mathrm{arcsinh}/\mathrm{sinh}$ 而不是 linear scaling 的原因在于**非线性**——这恰好对应 flux limiter 理论里 Godunov 定理要求 limiter 必须 nonlinear 才能 circumvent 一阶精度上限。如果用 $f(x) = x/C, g(x) = Cu(x)$（trivial scaling），$e_g$ 同步放大 $C$ 倍，没收益。

### 4.4 与 flux limiters / TVD schemes 的同构

这个 insight 极其重要：
- **Godunov 定理**: 线性 monotone 守恒格式不能高于一阶精度
- **Circumvention**: 用 nonlinear flux limiter（minmod, MC, van Leer, superbee 等）在 smooth 区域高阶、shock 附近退化为一阶
- **BEACONS 的 generalization**: 把 "shock 附近降阶" 推广为 "shock 部分用 NN 逼近误差大但用小 Lipschitz 的 smooth function 复合来压制"

Reference: [Harten 1997 High Resolution Schemes](https://www.sciencedirect.com/science/article/pii/S0021999197800154), [LeVeque Finite Volume Methods](https://www.cambridge.org/core/books/finite-volume-methods-for-hyperbolic-problems/9A5C2A3F2E2E5A0C2A1B4D6F3E5A7B8C), [Flux limiter Wikipedia](https://en.wikipedia.org/wiki/Flux_limiter), [TVD schemes Wikipedia](https://en.wikipedia.org/wiki/Total_variation_diminishing)

---

## 5. 与 PINN 的本质区别

paper Section 1 末尾给出一个特别 sharp 的对比：

| 维度 | PINN | BEACONS |
|---|---|---|
| Latent space 约束 | 通过 loss + penalty 强行限制在 "valid" 区域 | **完全 unconstrained** |
| Invalid 区域处理 | gradient descent 不允许进入（会惩罚） | 允许任意穿越，但 architecture 保证最终收敛到 valid 附近 |
| 失败模式 | 复杂非线性 PDE 中两 valid 区域间的"ravine"（如 entropy 非凸）会让 GD 失败 | 跨越 ravine 是允许的，因为 algebraic structure 提供独立的 bound 保证 |
| 错误保证来源 | loss function 的设计 | **架构本身的数学结构** |

这相当于 PINN 是"宗教式约束"（违反就惩罚），BEACONS 是"宪法式约束"（结构上不可能违反）。后者明显 robust 得多。

PINN 的文献调研：[Raissi et al. 2019 PINN](https://www.sciencedirect.com/science/article/pii/S0021999118307125), [Kim et al. DPM](https://ojs.aaai.org/index.php/AAAI/article/view/16952), [Eiras et al. PINN certification](https://arxiv.org/abs/2404.00536), [McGreivy & Hakim weak baselines](https://www.nature.com/articles/s42256-024-00913-x)

---

## 6. 软件实现

### 6.1 Racket DSL

作者使用 [Racket](https://racket-lang.org/) 构造 DSL，因为 Racket 的 macro 系统和 symbolic computation 能力适合 DSL 设计（参考 [The Racket Manifesto](https://racket-lang.org/manifesto/)）。DSL 描述 PDE 系统、simulation 参数、BEACONS 超参（width, depth, max training steps）。

### 6.2 自动代码生成器

DSL → optimized C code，覆盖：
- Lax-Friedrichs solver（[Lax 1954](https://onlinelibrary.wiley.com/doi/10.1002/cpa.3160070112)）
- Roe approximate Riemann solver（[Roe 1997](https://www.sciencedirect.com/science/article/pii/S0021999197800166)）
- Strang splitting（[Strang 1968](https://epubs.siam.org/doi/10.1137/0705041)）用于多维
- Flux limiters: minmod, MC, van Leer, superbee

### 6.3 自动定理证明

基于 **globally confluent + strongly normalizing term rewriting system**（参考 [Baader & Nipkow Term Rewriting](https://www.cambridge.org/core/books/term-rewriting-and-all-that/0F09F5C0E9F1F1A21F9D3A6F4F8C0F5)）。三个核心算法：
- `symbolic-simp`: 符号简化
- `symbolic-diff`: 符号求导
- `evaluate-limit`: 求极限

**关键约束**: 只允许遵守 IEEE-754 浮点代数结构的 symbolic transformation——这是为了避免 symbolic simplification 假设实数代数但生成的 C 代码运行在 IEEE-754 上而导致的 unsoundness。例如 IEEE-754 不满足结合律 $(a+b)+c \ne a+(b+c)$ 在浮点下，所以不能随意 reorder。

### 6.4 kann 神经网络库

训练用 [kann](https://github.com/attractivechaos/kann)，一个 minimalist C 库。每层 BEACONS 单独训练，各有专门 loss function，标准 backpropagation。

### 6.5 两类证明的逻辑独立性

作者特别强调：
- **Solver correctness proof**: 保证 training data 函数 = true PDE 解
- **BEACONS architecture correctness proof**: 保证 NN 逼近该函数 within bounded error

两者逻辑独立——可以仅有后者（subject to unproven solver correctness hypothesis）。这恰好对应 Euler 方程的 case：没有 formally verified Euler solver，但可以 conditional prove BEACONS bound。

### 6.6 GKEYLL 集成

整个系统整合进 [GKEYLL](https://gkeyll.readthedocs.io/) computational multi-physics 框架（PPPL 开发），提供 parallelism, I/O, grid generation 等非形式化基础设施。

---

## 7. 实验数据深度解析

### 7.1 1D Linear Advection Riemann（Table 1, 2）

| Architecture | $L^\infty$ Final | $L^2$ Final | $L^\infty$ All | Conservation Total |
|---|---|---|---|---|
| 6-layer NN | 1.034 | 2.537 | 1.076 | 498.7 |
| 8-layer NN | 0.977 | 9.031 | 1.003 | -4391.6 |
| 6-layer BEACONS | 0.612 | 1.132 | 0.782 | -155.3 |
| 8-layer BEACONS | 0.605 | 1.212 | 0.633 | 49.6 |

关键观察：
- 8-layer NN 在 $L^\infty$ 上比 6-layer NN 略好，但 $L^2$ 和 conservation 暴跌（**overfitting L∞ at expense of bulk accuracy**）
- BEACONS 实测最大 $L^\infty$ = 0.782 / 0.633，**远低于 proven bound 0.904 / 0.707**
- Conservation error BEACONS 比 NN 小 1-2 个数量级

### 7.2 2D Burgers Disk（Table 6）

| Architecture | $L^\infty$ Final | $L^2$ Final | $L^\infty$ All |
|---|---|---|---|
| 8-layer NN | 0.977 | 37.42 | 0.654 |
| 8-layer BEACONS | 0.320 | 2.69 | 0.596 |

BEACONS 的 $L^2$ 误差是 NN 的 **1/14**，说明在 2D 下 BEACONS 的优势**指数级放大**——因为 composability 在高维下 curse of dimensionality 带来的 $N^{-n/d}$ 衰减更慢，但 BEACONS 通过 Lipschitz 压缩有效绕过这一限制。

### 7.3 2D Euler Quadrants（Table 9）

| Architecture | $L^\infty$ Final | $L^2$ Final | $L^\infty$ All |
|---|---|---|---|
| 8-layer NN | 0.546 | 25.84 | 0.536 |
| 8-layer BEACONS | 0.198 | 2.96 | 0.310 |

这是最复杂的测试 case——2D Euler quadrants problem（4 个初始 discontinuity，多个 interacting waves）。BEACONS 的 $L^2$ 误差是 NN 的 **1/8.7**，$L^\infty$ 误差是 NN 的 1/2.8。

Theorem prover 跑了 **142,104 个 proof steps** 来证明 bound——这个数字本身说明 Euler 系统的 eigenstructure 处理复杂度（3 个 eigenvalues, 6 个 eigenvectors, 多个 genuinely nonlinear modes 互相耦合）。

### 7.4 8-layer vs 6-layer 在 Burgers 上反常

注意 Table 4 中 8-layer BEACONS 在 1D Burgers 上**比 6-layer BEACONS 略差**（$L^\infty$ 0.987 → 1.028）。作者解释为：对于这类非线性 IVP，6-layer 架构可能 near-optimal。这是 BEACONS 的一个**反 over-parameterization 信号**——更深未必更好，因为更多层意味着更多 Lipschitz 常数需要协调，可能让 $e_f$ 的累积超过 $L \cdot e_g$ 的抑制收益。

---

## 8. 与相关方向的联系

### 8.1 Neural ODE / DeepONet / FNO 的关系

- [Neural ODE](https://arxiv.org/abs/1806.07366): 将 residual network 视为 ODE 的 Euler 离散，但没有 a priori error bound
- [DeepONet](https://www.nature.com/articles/s42256-021-00343-1): 学习 operator 而非 function，但仍缺乏 formal verification
- [FNO (Fourier Neural Operator)](https://arxiv.org/abs/2010.08895): 用 Fourier layer 处理高频，但 spectral aliasing 在 shock 上有 Gibbs-like 问题
- BEACONS 与它们的本质区别是**形式化证明 + Lipschitz composability**

### 8.2 与 Foundation Models 的哲学对比

paper 末尾讨论了 [Bommasani et al. Foundation Models](https://arxiv.org/abs/2108.07258): FM 通过让 pretraining set 的 convex hull 极大化来"伪装"成 extrapolation——其实仍是 interpolation。BEACONS 走相反路线：**承认 convex hull 有限，用 PDE 结构推断 hull 外的 smoothness，从而给出真正的 extrapolatory bound**。

### 8.3 Mixture of Experts BEACONS Foundation Model

作者构想的"supernetwork"：每个 expert BEACONS 处理一条物理定律（Maxwell, Navier-Stokes, advection-diffusion 等），通过 [MoE](https://arxiv.org/abs/2209.07086) 路由器组装成 multiphysics solver。Finetuning 训练 router 学习如何 formally-verified 地 couple 各 expert。这是 modular neural networks 与 [Compositional Deep Learning](https://arxiv.org/abs/1907.09290) 的一个**可验证版本**。

### 8.4 Bitter Lesson 的回应

作者回应 [Sutton's Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html): 在 generic ML 上 brute scale 碾压算法进步；但在 computational physics 上 scale 不能改善 numerical method 本身的质量——所以 rigorous foundation 是 unavoidable。这是一个**反 bitter lesson**的论点，限定在 PDE 数值求解 domain 内。

### 8.5 Elliptic/Parabolic 推广

paper 末尾提到推广到 elliptic/parabolic PDE 用 elliptic regularity theorems（[Fernández-Real & Ros-Oton](https://link.springer.com/book/10.1007/978-3-031-08057-6)）代替 method of characteristics 来预测 smoothness。对 elliptic PDE，解的 smoothness 由 boundary data 和 domain 几何决定，这给出一种**完全不同的 a priori smoothness 推断路径**——可能为 BEACONS 进入 diffusion, elasticity, electromagnetism 等领域铺路。

### 8.6 Compiler optimization 类比

最 striking 的类比在结尾：把 NN 视为"aggressive non-deterministic compiler optimization"——类比 GCC `-ffast-math` 允许重排浮点运算的轻微 non-determinism。BEACONS 是把这种"lossy compiler"升级为"verified compiler"的尝试。这与 [MLIR](https://mlir.llvm.org/)、[Lean](https://lean-lang.org/)、[F*)](https://www.fstar-lang.org/) 等 verified compilation 路线有精神上的共鸣。

---

## 9. 局限性与开放问题

我从 paper 中读出的一些隐含 limitations（paper 没明说，但 build intuition 需要）：

1. **Mhaskar bound 是 infimum**：理论存在性 vs gradient descent 实际可达性有 gap。paper 没证明 GD 能找到 optimal $u_\theta$，只是说**存在**这样的 $u_\theta$ 满足 bound。这是所有 NN approximation theorem 的通病。

2. **Lipschitz constant 计算成本**：要严格 prove $f$ 是 $L$-Lipschitz，需要 symbolic 分析 activation 函数的二阶性质。对 deep networks 这可能 NP-hard（[NN Lipschitz computation is hard](https://arxiv.org/abs/2104.07390)）。BEACONS 通过 layer-by-layer 训练避免这个问题——每层是 shallow 函数，Lipschitz 容易算。

3. **Vector system 的 $\beta_{ij}$ 耦合**：Eq. 54 里 cross-wave coupling $\beta_{ij} \omega_i \omega_j$ 项使得 vector system 的 shock 形成比 scalar 复杂得多，论文的 Riccati 分析只是近似，严格 bound 可能更悲观。

4. **Choice of $f$ decomposition**：作者用 $\mathrm{arcsinh}/\mathrm{sinh}$ 等 trig/hyperbolic 函数，但没给出**最优 $f$ 选择**的形式化准则。Section 2.3 末尾说"BEACONS 自动 trials candidate functions"——这是启发式搜索，可能错过最优 decomposition。

5. **Generalization to non-conservation form**：method of characteristics 推导假设 conservation law form。对非守恒形式 PDE（如 Hamilton-Jacobi），shock 形成机制不同，可能需要 entropy solution 的不同工具。

6. **Stiff source terms**：Lemma 1 明确说"$du/dt = 0$ along characteristics" 在 source terms 存在时**不成立**——这是 BEACONS 处理 reaction-advection, MHD with Hall term 等问题的潜在障碍。

7. **Implicit time stepping**：对于 stiff 项（如 diffusion, viscosity），需要 implicit methods。BEACONS 论文聚焦 explicit hyperbolic，implicit extension 未涉及。

---

## 10. 我的整体直觉总结

这篇 paper 在我看来是 **numerical analysis 与 deep learning 之间的一座桥梁**，其核心贡献：

1. **重新 framing**: NN = generalized numerical method with adaptive basis
2. **Two theorems**: 
   - Mhaskar + characteristics → extrapolatory error bound
   - Lipschitz composability → deep architecture with controlled error
3. **Two pieces of software**:
   - Theorem prover generating machine-checkable certificates
   - Code generator producing verified C code end-to-end
4. **One philosophical shift**: PINN 用 loss 约束 latent space（脆弱），BEACONS 用 architecture 结构保证（鲁棒）

它本质上是在问：**"Can we make neural solvers as trustworthy as finite volume solvers?"** 答案是 yes，但代价是：
- 必须 know PDE 的解析结构（flux Jacobian eigenstructure, flux smoothness 等）
- 必须 decompose solution 成 compositional 结构
- 必须 verify 每一层的 bound 并 compose

这是 **neurosymbolic** 在 PDE 领域的一个 flagship 应用。如果这个范式推广到 elliptic, parabolic, integro-differential, stochastic PDE，并且 MoE-supernetwork 真能成型，那它可能会重塑 computational science 的软件栈——从"跑模拟"变成"生成 verified neural surrogate 并 extrapolate 到 unsimulable regimes"。作者提到的 counterfactual question "what would the simulation predict if it could run?"，是对物理学家**特别 seductive** 的愿景。

Reference for further reading:
- [Shock with Confidence (predecessor paper)](https://arxiv.org/abs/2505.24282)
- [gkylcas project](https://gkeyll.readthedocs.io/)
- [Roe solver Wikipedia](https://en.wikipedia.org/wiki/Riemann_solver)
- [Sod Shock Tube](https://en.wikipedia.org/wiki/Sod_shock_tube)
- [Burgers' Equation](https://en.wikipedia.org/wiki/Burgers%27_equation)
- [Euler Equations](https://en.wikipedia.org/wiki/Euler_equations_(fluid_dynamics))
- [Method of Characteristics](https://en.wikipedia.org/wiki/Method_of_characteristics)
- [Gibbs Phenomenon](https://en.wikipedia.org/wiki/Gibbs_phenomenon)
- [Runge's Phenomenon](https://en.wikipedia.org/wiki/Runge%27s_phenomenon)
- [Godunov's Theorem](https://en.wikipedia.org/wiki/Godunov%27s_theorem)
- [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754)
- [Term Rewriting](https://en.wikipedia.org/wiki/Rewriting)
- [Compositional Deep Learning (Gavranović)](https://arxiv.org/abs/1907.09290)
- [Applied Category Theory (Fong & Spivak)](https://arxiv.org/abs/1803.05316)
- [Categorical Deep Learning position paper](https://arxiv.org/abs/2402.15332)
