---
source_pdf: Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds.pdf
paper_sha256: 51b530c916d2f2c62af63589373f0fb55cbd29e5d3157fcbefc99cfa52619cc2
processed_at: '2026-08-04T21:24:28-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲一遍

## 一句话总结

在弯曲的configuration space里做sampling-based planning，传统方法用Euclidean distance相当于在地球表面用直线距离规划路径，几何上不对。这篇paper用一个很漂亮的trick——**在midpoint处做对称差分**——把geodesic distance的近似精度从二阶提升到三阶，代价只是一次额外的retraction计算。

---

## 问题是什么

想象你在山地上找最短路径。如果你用flat地图上的直线距离来规划，路线肯定偏——山地本身是弯的。

机器人configuration space就是这种弯曲山地。具体例子：7-DoF Franka机械臂。joint space表面上看是$\mathbb{R}^7$，平直的。但每个joint的有效惯量不同——base joint很重，wrist joint很轻。所以"在joint space里走一条直线"实际消耗的能量并不最小，真正能量最优的motion在joint space里是弯曲的。

这种"弯曲"用什么描述？Riemannian metric。具体说，kinetic energy metric $G(q) = M(q)$就是mass-inertia matrix，随configuration $q$变化。在这个metric下，最短路径叫geodesic，对应minimum kinetic energy motion。

---

## 为什么传统方法不行

**Sampling-based planner**（RRT、RRT*）scale好，高维能用。但它们用Euclidean distance做nearest neighbor查询，用直线插值做local planning。这在弯曲空间里几何上inconsistent，相当于把直线当测地线。

那为什么不直接解geodesic ODE？因为geodesic满足
$$
\ddot q^k + \Gamma^k_{ij} \dot q^i \dot q^j = 0
$$
要解boundary value problem。在7-DoF上，每次nearest neighbor查询都解一次BVP？完全不现实。

**Variational方法**（minimize energy functional $E = \frac{1}{2}\int \dot\pi^\top G \dot\pi \, dt$）在高维+obstacle场景下，landscape高度non-convex，对initial guess极度敏感。论文corridor场景variational solver只有8%成功率，几乎完全失败。

所以gap很清楚：sampling-based scalable但几何不对，geodesic solver几何对但不scalable且不robust。

---

## 核心insight：central finite difference的manifold版本

这是paper最漂亮的地方。直觉完全来自numerical analysis里central finite difference vs forward finite difference。

- Forward difference: $\frac{f(x+h) - f(x)}{h}$，误差$O(h)$
- Central difference: $\frac{f(x+h) - f(x-h)}{2h}$，误差$O(h^2)$

为什么central difference精度高一阶？Taylor展开里偶数阶项在对称差分下自动cancel。这个机制极其朴素且普适。

paper把同样的trick用在Riemannian distance近似上。

要算$q_x$到$q_y$的geodesic distance，**不要在endpoint处算retraction distance**（这是forward difference类比，误差$O(d^2)$），**而是在midpoint处算对称差分**：

$$
\hat d_{\mathcal{M}}(q_x, q_y) = \left\| \mathcal{R}_{\hat q_{\text{mid}}}^{-1}(q_y) - \mathcal{R}_{\hat q_{\text{mid}}}^{-1}(q_x) \right\|_{\hat q_{\text{mid}}}
$$

midpoint用retraction构造：
$$
\hat q_{\text{mid}} = \mathcal{R}_{q_x}\left(\frac{1}{2}\mathcal{R}_{q_x}^{-1}(q_y)\right)
$$

$\mathcal{R}$是retraction，exponential map的一阶近似，便宜很多。

---

## 为什么是三次精度

证明骨架（Appendix A）极其干净，我重新走一遍intuition：

**Step 1**: 在真正geodesic midpoint $q_{\text{mid}}$处用Riemann normal coordinates。这坐标系下metric tensor在原点等于identity，一阶偏导为零。设$u = \log_{q_{\text{mid}}}(q_y)$，则$\|u\| = h/2$，$h$是true distance。$q_x$和$q_y$的坐标分别是$-u$和$u$。

**Step 2**: retraction midpoint $\hat q_{\text{mid}}$偏离真正midpoint的量是$\delta = O(\|u\|^2)$。这步用retraction的一阶Taylor展开：
$$
\mathcal{R}_q(v) = q + v + O(\|v\|^2)
$$
代入$q_x = -u$，算$v = \mathcal{R}_{q_x}^{-1}(q_y) = 2u + O(\|u\|^2)$，再算$\hat q_{\text{mid}} = \mathcal{R}_{q_x}(\frac{1}{2}v) = -u + u + O(\|u\|^2) = O(\|u\|^2)$。

直觉：midpoint的偏移是二阶小量，因为retraction的一阶项精确线性，二阶项才引入误差。

**Step 3**: 在$\hat q_{\text{mid}}$处对两个endpoint做inverse retraction，得到$w_y$和$w_x$。它们的差：
$$
w_y - w_x = 2u + O(\|u\|^3)
$$

关键步骤：retraction在$z = \delta$附近做二阶Taylor展开
$$
\mathcal{R}(z, \zeta) = z + \zeta + \mathcal{Q}(\zeta, \zeta) + \mathcal{B}(z, \zeta) + O(\|(z,\zeta)\|^3)
$$
其中$\mathcal{Q}$是$\zeta$的二次型，$\mathcal{B}$是$(z, \zeta)$的双线性形式。

设$w_y = u + e_y$，$w_x = -u + e_x$。求解误差项：
$$
e_y = -\delta - \mathcal{Q}(u,u) - \mathcal{B}(\delta, u) + O(\|u\|^3)
$$
$$
e_x = -\delta - \mathcal{Q}(-u,-u) - \mathcal{B}(\delta, -u) + O(\|u\|^3)
$$

**核心magic**：用对称性$\mathcal{Q}(-u,-u) = \mathcal{Q}(u,u)$（二次型）和双线性$\mathcal{B}(\delta,-u) = -\mathcal{B}(\delta,u)$，相减：
$$
e_y - e_x = -2\mathcal{B}(\delta, u) + O(\|u\|^3)
$$

偶数阶项$-\delta$和$-\mathcal{Q}(u,u)$完全cancel！剩下的只有bilinear项$\mathcal{B}(\delta, u)$，而$\|\delta\| = O(\|u\|^2)$，所以这项是$O(\|u\|^3)$。

**Step 4**: 取norm。normal coordinates下$G(\delta) = I + O(\|u\|^2)$，于是
$$
\hat d_{\mathcal{M}} = \|2u + O(\|u\|^3)\|_{G(\delta)} = 2\|u\| + O(\|u\|^3) = h + O(h^3)
$$

误差$O(h^3)$，cubic accuracy。endpoint-based只能给$O(h^2)$。

**整个证明的灵魂**：Taylor展开的even-order项在symmetric difference下自动cancel。跟central finite difference完全同构。

---

## Local planner：natural gradient + retraction

有了distance近似，还要能"插值"出一条geodesic。Euclidean planner用直线插值，本文用discrete geodesic tracing。

把potential定义为平方distance：
$$
\phi(q) = \frac{1}{2}\hat d_{\mathcal{M}}(q, q^\dagger)^2
$$
$q^\dagger$是target（比如$q_{\text{rand}}$）。

Riemannian gradient（也就是**natural gradient**）：
$$
\text{grad}\,\phi(q) = G(q)^{-1}\nabla_u (\phi \circ \mathcal{R}_q)(0)
$$

这里$G(q)^{-1}$起的作用就是metric-aware的preconditioning——把Euclidean gradient用metric inverse重新normalize。这跟Amari在neural network training里提出的natural gradient完全一样，只是$G$从Fisher information matrix换成kinetic energy metric。

Update rule：
$$
q_{k+1} = \mathcal{R}_{q_k}(-s_k \hat v_k), \quad \hat v_k = \frac{v_k}{\|v_k\|_{q_k}}
$$

$\hat v_k$是归一化后的natural gradient方向，$s_k$是step size，带backtracking。retraction保证每一步都仍然在manifold上。

**Local planner的本质**：在distance field上做natural gradient descent，用retraction保证manifold约束，用backtracking处理curvature强的区域。整条路径是离散化的geodesic。

Algorithm 1里两个stopping criterion的直觉：
- Line 6-10 backtracking：如果retraction一步的实际displacement $\hat d(q, q_{\text{next}})$超过$\lambda s$（线性超步长阈值），说明curvature太强或metric变化太大，把$s$减半。这是Armijo line search的几何版本。
- Line 11-13 cumulative distance$d > d_{\max}$时停止，避免在强曲率区域无限扩展。

---

## 实验直觉

### 7-DoF Franka场景

环境来自MotionBenchMaker的table pick场景。50 trials，每trial 10 runs。

| Method | Length ↓ | Energy ↓ | Success ↑ |
|---|---|---|---|
| Variational | 2.5 ± 0.6 | 3.1 ± 1.5 | 96% |
| Sampling (Euclidean) | 2.6 ± 0.5 | 3.5 ± 1.5 | 85% |
| **Sampling (Ours)** | **2.1 ± 0.2** | **2.3 ± 0.4** | 90% |

读这张表的关键观察：
- **Euclidean sampling**能找到feasible path，但忽略configuration-dependent inertia，heavy base joint被不必要excite，能量高。
- **Variational** explicit minimize energy，但barrier function重塑metric后landscape超non-convex，方差大（±1.5），对参数敏感。
- **Ours**在length上降16%，energy上降26%，方差小很多（±0.4 vs ±1.5）。低方差说明采样exploration比贪心优化robust——这是sampling-based相对于variational的本质优势：**随机化restart天然explore多个homotopy class**，避开non-convex陷阱。

### SE(2) anisotropic场景

metric设成$G = \text{diag}(w_x, w_y, w_\theta)$，让$w_y \gg w_x$，惩罚lateral translation。这相当于**soft nonholonomic constraint**——把car-like车的"不能横移"编码进metric，不需要硬kinodynamic约束。

Doorway场景：
| Method | Length ↓ | Energy ↓ | Success ↑ |
|---|---|---|---|
| Variational | 24.9 ± 0.6 | 310.4 ± 14.0 | 86% |
| Sampling (Euclidean) | 43.7 ± 3.7 | 954.4 ± 174.7 | 100% |
| **Sampling (Ours)** | **23.2 ± 0.5** | **269.1 ± 11.7** | 100% |

Corridor窄通道场景：
| Method | Length ↓ | Energy ↓ | Success ↑ |
|---|---|---|---|
| Variational | ∞ | ∞ | 8% |
| Sampling (Euclidean) | 95.6 ± 9.2 | 4571.2 ± 873.0 | 100% |
| **Sampling (Ours)** | **43.0 ± 0.5** | **925.9 ± 22.5** | 100% |

**Corridor场景戏剧性**：variational几乎完全失败（8% success）。窄通道里barrier function重塑的landscape陷阱密集，variational solver卡在local minima。Euclidean能feasible但length比Ours长一倍多，能量高5倍——因为它忽略metric anisotropy，产生"skidding motion"（侧向平移）。

**Soft constraint via metric design**这个idea本身就很美：把硬约束（kinodynamic）软化成metric各向异性，planner自然就aligned到car-like行为。这比显式constraint handling优雅得多。

---

## 更大的picture与连接

### "对称性cancel误差"是个普适机制

这篇paper的核心机制在numerical analysis里无处不在：

- **Central finite difference**比forward高一阶：偶数阶Taylor项cancel
- **Symplectic integrator**（velocity Verlet）保能量：midpoint evaluation让even-order误差项消失，Hamiltonian structure保持
- **Gauss-Legendre quadrature**在中点取值，精度比Newton-Cotes高
- **Midpoint rule**在ODE积分里能量漂移最小

paper把这个机制正确移植到Riemannian manifold的retraction上，通过normal coordinates + Taylor展开clean地证明cubic accuracy。数学上很优雅，因为**几何对称性提升数值精度**这个pattern本身就有普适性。

### 与natural gradient / K-FAC的连接

公式8的Riemannian gradient $G(q)^{-1}\nabla\phi$就是Amari的natural gradient。在deep learning里$G$是Fisher information matrix，natural gradient让SGD在parameter space的Riemannian结构上正确前进，理论上比standard SGD更高效。实际Fisher计算太贵，有K-FAC、Shampoo等Kronecker-factored近似。

本文的midpoint retraction某种意义上也是"midpoint approximation of metric"——在midpoint处evaluate metric而不是endpoint，类似K-FAC在midpoint处evaluate Kronecker factors。两者共享同一个数学灵魂：**用对称性减少approximation error**。

### 与heat method的对比

[Crane et al. 2013](https://dl.acm.org/doi/10.1145/2516971.2516978)的heat method通过解heat equation $\partial_t u = \Delta u$并积分gradient flow近似geodesic distance，在mesh上global consistent但需要discretize manifold。

本文midpoint retraction是**local pairwise**近似，用于sampling-based planner的nearest neighbor查询，不需要全局discretize，scalable到高维。两者是不同尺度的geodesic distance近似：heat method是全局的、本文是局部的。

### 与RMPflow的对比

[RMPflow (Cheng et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9035283)把多个motion policy在Riemannian结构上组合，用pullback metric把task space metric拉回configuration space。哲学一致但操作相反：**RMPflow design metric来encode行为，本文respect given intrinsic metric并保证在它下面最优**。

类似工作还有[Region-Avoiding Metrics (Klein et al., 2023)](https://ieeexplore.ieee.org/abstract/document/10342301)用barrier function重塑metric避免collision。本文把collision avoidance交给sampling-based planner的free space sampling，metric本身不变，更modular。

### Information geometry视角

如果把configuration space看作probability manifold，kinetic energy metric $M(q)$在某种意义上是Fisher metric的一种"physical instantiation"——它measure configuration扰动对应的动能cost。从information geometry视角，**所有natural gradient类型算法都是Riemannian gradient descent on a chosen metric**。

本文midpoint retraction trick原则上能移植到任何Riemannian optimization场景：matrix manifold optimization、probability manifold上的VI、score-based generative model在manifold数据上的sampling。

---

## 可能的extension

1. **Bidirectional RRT**：当前single-tree导致Franka 90% success，用[RRT-Connect](https://ieeexplore.ieee.org/document/844730)的双向树能显著提升coverage。

2. **Heuristic design on manifold**：用$\hat d_{\mathcal{M}}$作为RRT*的cost-to-go heuristic估计，但midpoint approximation可能underestimate或overestimate，需要admissibility分析。正曲率空间（sphere）下可能需要inflate，负曲率（hyperbolic）可能需要deflate。

3. **Adaptive step size**：Algorithm 1的backtracking是geometric减半，可以用Barzilai-Borwein step或trust region策略，理论上convergence更快。

4. **Learned metric**：把$G(q)$用neural network参数化，从demonstration或RL学习cost-to-go function作为metric，midpoint retraction作为可微分operator。这就把Riemannian motion planning与imitation learning接通了。

5. **Constrained manifold**：现在paper限定$\mathcal{Q} = \mathcal{M}$（unconstrained case），但implicit manifold上的constrained motion planning [Jaillet & Porta, 2012](https://ieeexplore.ieee.org/document/6297116)需要projection operator。**Projection operator itself就是个retraction**，所以本文framework能自然扩展到constrained case。

6. **Sample complexity vs curvature**：Theorem 1只保证local cubic accuracy，但RRT*的asymptotic optimality依赖metric的global consistency。一个值得研究的open question：在curved space下，RRT*的asymptotic convergence rate如何随sectional curvature变化？我的直觉是正曲率（如sphere）下volume growth慢，sample efficiency可能更高；负曲率（hyperbolic）下volume growth快，sample效率更低——这跟hyperbolic neural network的representation power直觉一致。

7. **Differentiable planning**：把整个planner做成differentiable，metric $G(q)$通过end-effector task cost或human demonstration学习。midpoint retraction作为可微分operator，整个pipeline可end-to-end训练。这就把model-based planning与model-free learning的边界又往前推一步。

---

## 最终intuition

这篇paper的精髓可以浓缩成一句话：**Sampling-based planner的subroutine如果用Euclidean metric，相当于在curved space上把直线当作geodesic——几何上错；如果直接解geodesic BVP，在高维上太慢——计算上错。midpoint retraction distance通过对称差分用三次精度近似geodesic distance，既保留sampling-based scalability又恢复geometric fidelity。**

更深层的intuition：**几何对称性cancel数值误差**这个机制在numerical analysis里是核心套路，从central difference到symplectic integrator到Gauss-Legendre quadrature都靠它。本文把这个机制正确移植到Riemannian manifold的retraction上，数学证明clean，实验效果显著。这种"几何对称性提升数值精度"的pattern在robotic learning、differentiable physics、geometric deep learning里都有广泛应用空间。

参考链接合集：
- Pinocchio library: [stack-of-tasks/pinocchio](https://github.com/stack-of-tasks/pinocchio)
- OMPL: [ompl/ompl](https://github.com/ompl/ompl)
- StochMan: [MachineLearningLifeScience/stochman](https://github.com/MachineLearningLifeScience/stochman)
- MotionBenchMaker: [KavrakiLab/motion_bench_maker](https://github.com/KavrakiLab/motion_bench_maker)
- RRT*: [Karaman & Frazzoli 2011](https://journals.sagepub.com/doi/10.1177/0278364911406761)
- Natural gradient: [Amari 1998](https://www.mitpressjournals.org/doi/10.1162/089976698300017746)
- Heat method: [Crane et al. 2013](https://dl.acm.org/doi/10.1145/2516971.2516978)
- RMPflow: [Cheng et al. 2021](https://ieeexplore.ieee.org/abstract/document/9035283)
- Boumal textbook: [nicolasboumal.net/book](https://www.nicolasboumal.net/book/index.html)
- Bullo & Lewis: [Springer link](https://link.springer.com/book/10.1007/978-1-4612-0435-1)
- K-FAC: [Martens & Grosse](https://www.cs.toronto.edu/~jmartens/docs/KFAC.pdf)

---

# Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds — 深度解析

这篇paper来自UTIAS的Phone Thiha Kyaw与Jonathan Kelly，核心命题非常优雅：**把sampling-based motion planning从Euclidean ambient space搬到Riemannian manifold上，用midpoint retraction做cubic-accurate的geodesic distance近似**。我想从几何直觉、数学证明骨架、算法实现、实验数据几个层面把这篇paper拆开讲，并把它和更广的natural gradient / information geometry / symplectic integration做关联。

---

## 1. 为什么Euclidean metric在robotics里是错的

机器人configuration space $\mathcal{M}$本质上不是一个flat Euclidean space。典型的几种几何结构：

- **Lie group结构**: rigid body pose在SE(2)/SE(3)上；articulated manipulator在torus $T^n = S^1 \times \cdots \times S^1$上；
- **Implicit manifold**: closed-chain约束$f(q)=0$诱导的低维子流形；
- **Configuration-dependent metric**: 即使topologically是$\mathbb{R}^n$，但kinetic energy metric $G(q) = M(q)$（mass-inertia matrix）随$q$变化，所以measurement of length本身是弯曲的。

一个straight line in joint space对应一个end-effector path的弯弯曲曲运动，并不对应minimum kinetic energy。Riemannian view告诉我们：**真正"自然"的motion是geodesic**，满足

$$
\ddot q^k(t) + \Gamma^k_{ij}(q(t))\dot q^i(t) \dot q^j(t) = 0, \quad i,j,k \in \{1,\ldots,n\}
$$

变量含义：$q^k(t)$是第$k$个joint的局部坐标，$\Gamma^k_{ij}$是Christoffel symbol of the second kind，描述坐标basis在manifold上的变化率。它的表达式：

$$
\Gamma^k_{ij} = \frac{1}{2} G^{kl}\left(\frac{\partial G_{il}}{\partial q^j} + \frac{\partial G_{jl}}{\partial q^i} - \frac{\partial G_{ij}}{\partial q^l}\right)
$$

上标$k$是contravariant index（通过$G^{kl}$即metric inverse提升），下标$ij$是covariant indices；$G_{il}$是metric tensor component；$\partial/\partial q^j$是对第$j$个坐标的偏导。

**直接解这个ODE需要解BVP（boundary value problem），在高维manipulator上完全不可用**。Variational solver通过minimize energy functional $E(\pi) = \frac{1}{2}\int \dot\pi^\top G_{\pi}\dot\pi \, dt$也面对非凸landscape。Sampling-based planner（RRT/RRT*）scalable但用Euclidean distance，几何上inconsistent。

这就是paper要桥接的gap。

参考：[Bullo & Lewis, Geometric Control of Mechanical Systems](https://link.springer.com/book/10.1007/978-1-4612-0435-1)；[Karaman & Frazzoli, RRT*](https://journals.sagepub.com/doi/10.1177/0278364911406761)。

---

## 2. 核心insight：midpoint symmetry → cubic accuracy

这是paper的技术heart，我觉得非常漂亮，思路跟numerical analysis里central finite difference vs. forward difference是同构的。

### 2.1 精确恒等式（Lemma 1）

设$q_x, q_y \in \mathcal{M}$在一个geodesically convex邻域内。geodesic midpoint定义为

$$
q_{\text{mid}} = \exp_{q_x}\left(\frac{1}{2}\log_{q_x}(q_y)\right)
$$

直觉：从$q_x$出发沿测地线走到$q_y$的中途。

**Lemma 1 (exact identity)**:

$$
d_{\mathcal{M}}(q_x, q_y) = \left\| \log_{q_{\text{mid}}}(q_y) - \log_{q_{\text{mid}}}(q_x) \right\|_{q_{\text{mid}}}
$$

证明思路：设$\pi:[0,1]\to\mathcal{M}$是从$q_x$到$q_y$的唯一minimizing geodesic。在midpoint处，向$q_y$的segment为$\pi_y(t) = \pi(1/2 + t/2)$，于是$\log_{q_{\text{mid}}}(q_y) = \dot\pi_y(0) = \frac{1}{2}\dot\pi(1/2)$；同理$\log_{q_{\text{mid}}}(q_x) = -\frac{1}{2}\dot\pi(1/2)$。两者相减得$\dot\pi(1/2)$，取norm（因为geodesic has constant speed）就等于$\int_0^1 \|\dot\pi(t)\|dt = d_{\mathcal{M}}(q_x,q_y)$。

**关键symmetry**: $\log_{q_{\text{mid}}}(q_y)$与$\log_{q_{\text{mid}}}(q_x)$在midpoint tangent space里恰好反向对称。这个对称性是后续error cancellation的根源。

### 2.2 Retraction近似（公式7）

但$\exp$与$\log$在大维度上仍然昂贵。Retraction $\mathcal{R}_q: T_q\mathcal{M} \to \mathcal{M}$是exponential map的一阶近似，满足$\mathcal{R}_q(0)=q$和$D\mathcal{R}_q(0) = I$（单位微分）。Retraction可以是很便宜的操作，比如cannonical retraction on Lie group。

把$\exp$/$\log$换成$\mathcal{R}$/$\mathcal{R}^{-1}$：

$$
\hat d_{\mathcal{M}}(q_x, q_y) = \left\| \mathcal{R}_{\hat q_{\text{mid}}}^{-1}(q_y) - \mathcal{R}_{\hat q_{\text{mid}}}^{-1}(q_x) \right\|_{\hat q_{\text{mid}}}
$$

其中retraction midpoint

$$
\hat q_{\text{mid}} = \mathcal{R}_{q_x}\left(\frac{1}{2}\mathcal{R}_{q_x}^{-1}(q_y)\right)
$$

### 2.3 Theorem 1: cubic accuracy

$$
\left| \hat d_{\mathcal{M}}(q_x, q_y) - d_{\mathcal{M}}(q_x, q_y) \right| = \mathcal{O}\left(d_{\mathcal{M}}(q_x, q_y)^3\right)
$$

这是核心定理。直觉如下：

- 如果用endpoint-based retraction distance $\|\mathcal{R}_{q_x}^{-1}(q_y)\|_{q_x}$，因为$\mathcal{R}$只是exponential的一阶近似，误差是$\mathcal{O}(d^2)$（second order）。
- 但**在midpoint处用对称差分，even-order distortion项相互cancel**，所以二阶项消失，误差降到$\mathcal{O}(d^3)$。
- 完全类比central finite difference $\frac{f(x+h)-f(x-h)}{2h}$比forward difference $\frac{f(x+h)-f(x)}{h}$高一阶精度。

### 2.4 证明骨架（Appendix A）

设$h = d_{\mathcal{M}}(q_x,q_y)$，$u = \log_{q_{\text{mid}}}(q_y)$，则$\|u\| = h/2$。在$q_{\text{mid}}$处用Riemann normal coordinates。

**Lemma 2**: retraction midpoint $\hat q_{\text{mid}}$ 在normal coordinates下偏离原点的量 $\delta = \mathcal{O}(\|u\|^2)$。

证明：因为$\mathcal{R}_q(0)=q$, $D\mathcal{R}_q(0) = I$，所以
$$
\mathcal{R}_q(v) = q + v + \mathcal{O}(\|v\|^2)
$$
$$
\mathcal{R}_q^{-1}(p) = (p-q) + \mathcal{O}(\|p-q\|^2)
$$

把$q_x = -u$, $q_y = u$代入：
$$
v = \mathcal{R}_{q_x}^{-1}(q_y) = (q_y - q_x) + \mathcal{O}(\|u\|^2) = 2u + \mathcal{O}(\|u\|^2)
$$

于是
$$
\hat q_{\text{mid}} = \mathcal{R}_{q_x}\left(\tfrac{1}{2}v\right) = -u + u + \mathcal{O}(\|u\|^2) = \mathcal{O}(\|u\|^2)
$$

所以$\|\delta\| = \mathcal{O}(\|u\|^2) = \mathcal{O}(h^2)$。Midpoint的偏移是二阶小量。

**Lemma 3**: 
$$
w_y - w_x := \mathcal{R}_{\hat q_{\text{mid}}}^{-1}(q_y) - \mathcal{R}_{\hat q_{\text{mid}}}^{-1}(q_x) = 2u + \mathcal{O}(\|u\|^3)
$$

证明：把retraction在$z=\delta$附近做二阶Taylor展开
$$
\mathcal{R}(z, \zeta) = z + \zeta + \mathcal{Q}(\zeta,\zeta) + \mathcal{B}(z,\zeta) + \mathcal{O}(\|(z,\zeta)\|^3)
$$

其中$\mathcal{Q}$是$\zeta$的二次型，$\mathcal{B}$是$(z,\zeta)$的双线性形式。

设$w_y = u + e_y$, $w_x = -u + e_x$，从$\mathcal{R}(\delta, w_y) = u$与$\mathcal{R}(\delta, w_x) = -u$求解误差项：

$$
e_y = -\delta - \mathcal{Q}(u,u) - \mathcal{B}(\delta, u) + \mathcal{O}(\|u\|^3)
$$
$$
e_x = -\delta - \mathcal{Q}(-u,-u) - \mathcal{B}(\delta, -u) + \mathcal{O}(\|u\|^3)
$$

用对称性$\mathcal{Q}(-u,-u) = \mathcal{Q}(u,u)$与双线性$\mathcal{B}(\delta,-u) = -\mathcal{B}(\delta,u)$，相减：

$$
e_y - e_x = -2\mathcal{B}(\delta, u) + \mathcal{O}(\|u\|^3)
$$

由Lemma 2，$\|\delta\| = \mathcal{O}(\|u\|^2)$，所以$\mathcal{B}(\delta, u) = \mathcal{O}(\|\delta\|\|u\|) = \mathcal{O}(\|u\|^3)$，最终：

$$
w_y - w_x = 2u + \mathcal{O}(\|u\|^3)
$$

**Theorem 1 finalize**: 在normal coordinates下$G(\delta) = I + \mathcal{O}(\|u\|^2)$，于是

$$
\hat d_{\mathcal{M}} = \|2u + r\|_{G(\delta)} = \|2u\|_{G(0)}\cdot(1+\mathcal{O}(\|u\|^2)) + \mathcal{O}(\|u\|^3) = 2\|u\| + \mathcal{O}(\|u\|^3)
$$

因为$d_{\mathcal{M}}(q_x,q_y) = 2\|u\| = h$，所以误差是$\mathcal{O}(h^3)$。$\square$

这个证明极其干净，关键在于**Taylor展开的even-order项在symmetric difference下自动cancel**。

参考：[Absil, Mahony, Sepulchre, Optimization Algorithms on Matrix Manifolds](https://press.princeton.edu/absil)；[Boumal, Intro to Optimization on Smooth Manifolds](https://www.nicolasboumal.net/book/index.html)。

---

## 3. Local planner：retraction + Riemannian natural gradient

### 3.1 Riemannian gradient的定义

设$\phi:\mathcal{M}\to\mathbb{R}$，Riemannian gradient $\text{grad}\,\phi$由Riesz表示定理唯一确定：

$$
D\phi(q)[v] = \langle v, \text{grad}\,\phi(q)\rangle_q, \quad \forall v \in T_q\mathcal{M}
$$

在局部坐标下，Riemannian gradient与Euclidean gradient的关系是

$$
\text{grad}\,\phi(q) = G(q)^{-1}\nabla_u (\phi \circ \mathcal{R}_q)(0)
$$

下标$u$表示在tangent space坐标下的Euclidean gradient；$G(q)^{-1}$是metric tensor的逆。**这就是natural gradient的formula**——Amari在information geometry里最早用于神经网络优化。

参考：[Amari, Natural Gradient Works Efficiently in Learning](https://www.mitpressjournals.org/doi/10.1162/089976698300017746)。

### 3.2 拟geodesic tracing（Algorithm 1）

把potential取为平方距离函数（公式9）：

$$
\phi(q) = \frac{1}{2}\hat d_{\mathcal{M}}(q, q^\dagger)^2
$$

其中$q^\dagger$是固定target（比如$q_{\text{rand}}$）。

Update rule（沿着$-\text{grad}\,\phi$走一步retraction）：

$$
q_{k+1} = \mathcal{R}_{q_k}(-s_k \hat v_k), \quad \hat v_k = \frac{v_k}{\|v_k\|_{q_k}}
$$

下标$k$是iteration index，$s_k > 0$是step length（通过backtracking调节），$\hat v_k$是归一化Riemannian gradient方向。

**Algorithm 1的关键tricks**:
- Line 6–10: backtracking。如果retraction step的实际displacement $\hat d_{\mathcal{M}}(q, q_{\text{next}})$超过$\lambda s$（线性超步长阈值），说明curvature太强或metric变化太大，把$s$减半。这是经典Armijo line search的几何版本。
- Line 11–13: cumulative distance $d > d_{\max}$时停止，避免在强曲率区域无限扩展。
- Line 15: 当$\hat d_{\mathcal{M}}(q, q_{\text{rand}}) \leq s$（已接近target）停止。

**为什么这是离散geodesic**: 我们在minimize squared distance potential，natural gradient方向就是Riemannian意义下的最速下降方向，retraction保证每一步仍然在manifold上。所以这其实是**discretized geodesic equation by steepest descent on distance field**——和heat method、fast marching思路有亲缘关系，但用的是first-order retraction而非discretized mesh。

参考：[Crane, Weischedel, Wardetzky, Geodesics in Heat](https://dl.acm.org/doi/10.1145/2516971.2516978)；[Sethian, Fast Marching Method](https://www.pnas.org/doi/10.1073/pnas.93.4.1591)。

---

## 4. 实验数据深度解析

实验在三个场景上做：(i) 2-link planar arm (kinetic-energy metric)，(ii) 7-DoF Franka (kinetic-energy metric + obstacles)，(iii) SE(2) rigid body (anisotropic metric模拟soft nonholonomic constraint)。

### 4.1 2-link planar arm

- Link length 1.0 m, mass 1.0 kg each
- $q_{\text{start}} = [-\pi/4, -\pi/4]^\top$, $q_{\text{goal}} = [3\pi/4, 3\pi/4]^\top$
- Metric = mass-inertia matrix $M(q)$ via Pinocchio's Composite Rigid Body Algorithm

**观察**: Euclidean baseline给出joint space的直线，但实际effective inertia在base joint处远高于elbow joint，所以这条"直线"路径消耗的能量并非最小。Variational solver与BVP都对initial guess敏感，经常收敛到local minima。本文方法直接找到globally optimal geodesic——**采样树天然explore多个homotopy class**，避开优化方法的non-convex陷阱。

这一点非常重要：**variational optimization在非凸Riemannian energy landscape上的初始敏感性是个本质缺陷**，sampling-based方法通过随机化的多重restart克服了它。

### 4.2 7-DoF Franka

环境来自[MotionBenchMaker](https://motionbenchmaker.cs.cornell.edu/)的table pick场景。50 trials，每trial 10 runs。

| Method | Length ↓ | Energy ↓ | Success ↑ |
|---|---|---|---|
| Variational | 2.5 ± 0.6 | 3.1 ± 1.5 | 96% |
| Sampling (Euclidean) | 2.6 ± 0.5 | 3.5 ± 1.5 | 85% |
| **Sampling (Ours)** | **2.1 ± 0.2** | **2.3 ± 0.4** | 90% |

解读：
- Euclidean sampling能找到可行path但**不计入configuration-dependent inertia**，导致heavy base joint被不必要excite，能量高。
- Variational explicit minimize energy，但barrier function重塑metric后landscape变得高度non-convex，参数难调，方差大（±1.5）。
- Ours在length上**降低16%**（2.1 vs 2.5），在energy上**降低26%**（2.3 vs 3.1），方差显著更小（±0.4 vs ±1.5）。这个低方差说明采样-based的解空间exploration比贪心优化更鲁棒。
- Success rate 90%略低于Variational 96%，因为single-tree RRT，作者建议用bidirectional RRT-Connect改善。

### 4.3 SE(2) anisotropic planning（Willow Garage map）

在SE(2)上用left-invariant metric $G = \text{diag}(w_x, w_y, w_\theta)$。设$w_y \gg w_x$，惩罚lateral translation，相当于**soft nonholonomic constraint**——把"只能前进不能横移"这个car-like约束编码进metric，而不是显式作为kinodynamic约束。

Doorway场景：

| Method | Length ↓ | Energy ↓ | Success ↑ |
|---|---|---|---|
| Variational | 24.9 ± 0.6 | 310.4 ± 14.0 | 86% |
| Sampling (Euclidean) | 43.7 ± 3.7 | 954.4 ± 174.7 | 100% |
| **Sampling (Ours)** | **23.2 ± 0.5** | **269.1 ± 11.7** | 100% |

Corridor场景（更窄通道）：

| Method | Length ↓ | Energy ↓ | Success ↑ |
|---|---|---|---|
| Variational | ∞ | ∞ | 8% |
| Sampling (Euclidean) | 95.6 ± 9.2 | 4571.2 ± 873.0 | 100% |
| **Sampling (Ours)** | **43.0 ± 0.5** | **925.9 ± 22.5** | 100% |

**Corridor场景戏剧性**: Variational几乎完全失败（8% success），因为窄通道里barrier function重塑的landscape陷阱密集。Euclidean能feasible但length比Ours长一倍多，能量高5倍（4571 vs 926）。

**Insight**: Euclidean规划器忽略metric anisotropy，产生"skidding或screw motion"——刚体侧向平移。Geometry-aware方法自然align朝向与运动方向，与car-like车的实际约束一致。这就是**soft constraint via metric design**的力量，比硬约束更flexible。

---

## 5. 与更大context的连接

### 5.1 Natural gradient与deep learning

公式8的Riemannian gradient $\text{grad}\,\phi = G(q)^{-1}\nabla\phi$正是Amari的natural gradient在neural network training上的同构形式。在那里$G$是Fisher information matrix，natural gradient让SGD在参数空间的Riemannian结构上正确前进。

Karpathy你肯定熟悉这个：standard SGD用Euclidean metric implicitly假设参数空间flat，但实际loss landscape有强各向异性（不同layer的scale、不同方向curvature差异巨大）。Natural gradient通过Fisher metric或Hessian inverse重新normalize，理论上更优，但Fisher计算代价高，所以实际有K-FAC、Shampoo等近似。

本文思路完全一致：用Riemannian metric描述各向异性，但用**midpoint-based retraction**做几何上correct的更新，类似于K-FAC用midpoint Kronecker factored approximation。

参考：[Martens & Grosse, Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://www.cs.toronto.edu/~jmartens/docs/KFAC.pdf)。

### 5.2 Symplectic integration与midpoint methods

Midpoint-based思想在数值积分里无处不在。Velocity Verlet integrator在Hamiltonian dynamics里能量长期conserved，而forward Euler会能量漂移，本质是因为**midpoint evaluation让even-order误差项消失**，相当于在symplectic structure上保持几何一致性。

本文Lemma 2–3的Taylor展开里$\mathcal{Q}(-u,-u) = \mathcal{Q}(u,u)$这个对称性cancel even-order error，与symplectic integrator保能量是同一个数学机制：**几何对称性 → 误差阶提升**。

参考：[Hairer, Lubich, Wanner, Geometric Numerical Integration](https://link.springer.com/book/10.1007/3-540-30666-8)。

### 5.3 RMPflow与Riemannian motion policies

[RMPflow (Cheng et al., 2021)](https://ieeexplore.ieee.org/abstract/document/9035283)把多个motion policy在Riemannian结构上组合，用pullback metric把task space metric拉回configuration space。本文与之哲学一致但操作相反：**RMPflow reshape metric来encode行为，本文work with given intrinsic metric并保证在它下面最优**。

类似工作还有[Region-Avoiding Metrics (Klein et al., 2023)](https://ieeexplore.ieee.org/abstract/document/10342301)用barrier function重塑metric避免collision，但本文把collision avoidance交给sampling-based planner的free space sampling，metric本身不变，更modular。

### 5.4 Heat method与geodesic distance

[Crane et al. 2013](https://dl.acm.org/doi/10.1145/2516971.2516978)的heat method通过解heat equation $\partial_t u = \Delta u$并积分gradient flow来近似geodesic distance，在mesh上有global consistency但需要discretize manifold。

本文midpoint retraction则是**local pairwise**近似，用于sampling-based planner的nearest neighbor查询，不需要全局discretize，scalable到高维。两者是不同尺度的geodesic distance近似：heat method是全局的、本文是局部的。

### 5.5 StochMan与variational geodesic

paper的variational baseline用了[StochMan library](https://github.com/MachineLearningLifeScience/stochman)，用cubic spline参数化geodesic并optimize energy functional。这种variational approach在[Hauberg group的若干工作](https://www2.compute.dtu.dk/~sohau/)里很流行，但在高维manipulator + obstacle场景下严重受困于non-convex landscape。

### 5.6 Information geometry的更广视角

如果我们把configuration space看作probability manifold，kinetic energy metric $M(q)$在某种意义上是Fisher metric的一种"physical instantiation"——它measure configuration扰动对应的动能cost。从information geometry的视角，**所有natural gradient类型算法都是Riemannian gradient descent on a chosen metric**。

本文给出的midpoint retraction trick原则上能移植到任何Riemannian optimization场景：matrix manifold optimization、probability manifold上的VI、score-based generative model在manifold数据上的sampling。

参考：[Amari, Information Geometry and Its Applications](https://link.springer.com/book/10.1007/978-4-431-55978-8)。

---

## 6. 我对paper的评价与可能的extension

**Strengths**:
1. Theorem 1的cubic accuracy证明是clean的mathematical contribution，把central finite difference的直觉正确移植到Riemannian manifold上。
2. Retraction-based local planner让整个framework能在任何有retraction定义的manifold上工作，包括SE(3)、SO(n)、Stiefel manifold等。
3. Soft nonholonomic constraint via anisotropic metric的演示非常有启发性——metric design可以替代硬约束，对kinodynamic planning是个优雅替代。

**可能的extension与open questions**:

1. **Heuristic design on manifold**: paper conclusion里提到要设计curved-space admissible heuristic。一个自然的方向是用$\hat d_{\mathcal{M}}$作为RRT*的heuristic cost-to-go估计，需要确保admissibility（不超过true distance），但midpoint approximation可能underestimate也可能overestimate，需要bias correction。

2. **Bidirectional RRT**: 当前single-tree导致Franka场景90% success，用[RRT-Connect (Kuffner & LaValle)](https://ieeexplore.ieee.org/document/844730)的双向树能显著提升coverage。

3. **Adaptive step size**: Algorithm 1的backtracking是geometric减半，可以用Barzilai-Borwein step或trust region策略，理论上convergence更快。

4. **与CHOMP/TRAC-IK的对比**: CHOMP用Euclidean gradient + obstacle potential，本文用Riemannian natural gradient + sampling-based collision avoidance，能否做一个统一benchmark比较值得做。

5. **Manifold constraint projection**: 现在paper限定$\mathcal{Q} = \mathcal{M}$（unconstrained case），但implicit manifold上的constrained motion planning [Jaillet & Porta, 2012](https://ieeexplore.ieee.org/document/6297116)需要projection operator。Midpoint retraction可以自然扩展到constrained case——projection operator itself就是个retraction。

6. **Information geometric extension**: 如果把configuration space视为probability manifold，$G(q)$取Fisher metric，本文framework变成natural gradient sampling on probabilistic models——可能对Bayesian inference、VI、normalizing flow training有意义。

7. **Sample complexity**: Theorem 1只保证local cubic accuracy，但RRT*的asymptotic optimality依赖metric的global consistency。一个值得研究的问题：在curved space下，RRT*的asymptotic convergence rate如何随sectional curvature变化？我的直觉是正曲率（如sphere）下sample efficiency可能更高，因为volume growth更慢；负曲率（如hyperbolic space）下更慢。

---

## 7. 总结性intuition

整篇paper的精髓可以用一句话总结：**Sampling-based planner的nearest-neighbor与local-planning subroutine若用Euclidean metric，相当于在curved space上把直线当作geodesic——错；若直接解geodesic BVP，则在高维度上太慢——也错。midpoint retraction distance用三次精度近似geodesic distance，既保留sampling-based scalability又恢复geometric fidelity。**

更深层的intuition：**对称性cancel误差**这个机制是numerical analysis的核心套路之一，从central difference到symplectic integrator到Gauss-Legendre quadrature都靠它。本文把这个机制正确地移植到Riemannian manifold的retraction上，并通过normal coordinates + Taylor expansion clean地证明cubic accuracy。这种"几何对称性提升数值精度"的pattern在robotic learning、differentiable physics、geometric deep learning里都有广泛应用空间。

下一步如果让我extend，我会想到的是把这个midpoint trick用于**learned metric**（neural network-parameterized Riemannian metric）——比如RL学一个cost-to-go function作为metric，然后用本文的geometry-aware planner在它上面规划。这把model-free RL与model-based planning的边界又往前推一步。

或者更激进一点：把整个planner做成differentiable，metric $G(q)$通过end-effector task cost或human demonstration学习，midpoint retraction作为可微分operator——这就把Riemannian motion planning与imitation learning接通了。

参考链接合集：
- Paper PDF: [arxiv version](https://arxiv.org/abs/2507.01697)（注：实际arxiv ID需查证，可能为[此链接](https://arxiv.org/abs/2412.05197)中相关工作的引用）
- Pinocchio library: [stack-of-tasks/pinocchio](https://github.com/stack-of-tasks/pinocchio)
- OMPL: [ompl/ompl](https://github.com/ompl/ompl)
- StochMan: [MachineLearningLifeScience/stochman](https://github.com/MachineLearningLifeScience/stochman)
- MotionBenchMaker: [KavrakiLab/motion_bench_maker](https://github.com/KavrakiLab/motion_bench_maker)
- RRT*: [Karaman & Frazzoli 2011](https://journals.sagepub.com/doi/10.1177/0278364911406761)
- Natural gradient: [Amari 1998](https://www.mitpressjournals.org/doi/10.1162/089976698300017746)
- Heat method: [Crane et al. 2013](https://dl.acm.org/doi/10.1145/2516971.2516978)
- RMPflow: [Cheng et al. 2021](https://ieeexplore.ieee.org/abstract/document/9035283)
- Boumal textbook: [nicolasboumal.net/book](https://www.nicolasboumal.net/book/index.html)
- Bullo & Lewis: [Springer link](https://link.springer.com/book/10.1007/978-1-4612-0435-1)
