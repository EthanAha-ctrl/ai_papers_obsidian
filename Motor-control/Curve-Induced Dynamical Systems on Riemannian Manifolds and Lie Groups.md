---
source_pdf: Curve-Induced Dynamical Systems on Riemannian Manifolds and Lie Groups.pdf
paper_sha256: 4091fe1b782a23363f6ab30f40643dbf99fa5f9e30eff1da624c25523e84cd14
processed_at: '2026-08-03T18:09:43-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Andrej，让我换一种方式来讲，用更多的比喻和 intuition，同时依然保持技术的深度。

---

## 一、这 paper 到底在解决什么痛点?

想象你在教 robot 穿衣服。你给它 demo 了一个轨迹："先把手伸到袖口，然后顺着胳膊往上拉"。这个轨迹不是一个简单的点在空间里移动，它包含两层信息：

- **Where**: end-effector 的 pose（位置+朝向），属于 $SE(3)$
- **How soft/stiff**: impedance controller 的 damping matrix，属于 $S_{++}^6$

这两个东西都不是 Euclidean 的。Pose 有旋转自由度，你不能简单地把两个 quaternion 相加。Damping matrix 必须保持 positive definite，你也不能随便在 $\mathbb{R}^{36}$ 里乱来。

现有的 learning-based 方法（LieFlows 用 normalizing flow，PUMA 用 metric learning）有几个致命问题：

1.  **Training 慢**: 1324 秒 train 一个 shape，robot 在 household 里等不起
2.  **OOD 差**: 偏离 demo 远了就乱跑，因为它们只学了一个 goal-directed 的 vector field，没有显式地"锚定"到 demo 轨迹
3.  **不连续**: demo 数据本身可能有 noise，学出来的 DS 不保证 $C^1$ 连续
4.  **几何错误**: 很多方法在 $\mathbb{R}^n$ 里算完再 project 回 manifold，相当于在地图上画直线然后硬贴到地球仪上，经纬线全歪了

CDSM 的思路：我别学什么 neural network 了，我直接在 manifold 上 fit 一条 smooth 的 Bézier 曲线，然后基于这条曲线解析地构造一个 vector field。就像你在地球仪上画一条航线，然后告诉船："你在任何位置，先朝航线靠拢（normal attraction），再沿航线前进（tangential propagation）"。

---

## 二、核心机制的三个 "Aha Moment"

### Aha 1: 为什么不能直接在 Tangent Space 里搞?

很多人觉得：我把 manifold 局部展开成 tangent space，在 flat space 里做 Bézier，再 exp 回去不就行了?

问题在于 manifold 是弯曲的。你在北极的 tangent space 画一条直线，exp 到赤道附近，和你在赤道自己的 tangent space 画的直线，两者方向可能完全对不上。

paper 里用了 **Single Tangent Space Fallacy** 这个概念（Ref: [Jaquier et al. ICRA 2024](https://ieeexplore.ieee.org/document/10610668)）。就好比你站在北京看世界地图，觉得纽约在你正前方偏左一点，于是你一直往那个方向走，结果走到了西伯利亚。因为地球是弯的，不同点的"正前方"不是同一个方向。

CDSM 的解法是 **Parallel Transport**。每一段 Bézier 都在自己的 tangent space 里定义，但是段与段之间的 continuity 用 parallel transport 来对齐。而且 DS 运行时，curve 上最近点的切向量也通过 parallel transport 搬到当前 state 的 tangent space。

公式回顾（Eq. 16）:
$$ \dot{x} = k_{\text{TC}} \cdot \mathcal{P}_{\pi(x) \to x}\left(\gamma'(\tilde{s})\right) - k_{\text{NC}} \cdot \nabla \Psi_{\text{NC}}(x) $$

- $\mathcal{P}_{\pi(x) \to x}$: 把 curve 上 $\pi(x)$ 处的切向量"沿测地线搬运"到 $x$ 处
- $\gamma'(\tilde{s})$: 曲线在最近点处的切向量
- $\nabla \Psi_{\text{NC}}(x)$: 从 $x$ 指向 $\pi(x)$ 的测地方向

直觉上：你在地球上某个位置，你想知道"沿着航线方向走"应该朝哪。你不能直接看航线在那边的方向（因为你们不在同一个 flat 平面上），你要把航线方向"平移"到你脚下，这就是 parallel transport。

---

### Aha 2: Gauss Lemma 为什么这么关键?

稳定性证明里最妙的一步是用了 **Gauss's Lemma**（First Variation of Arc Length）。

这个 lemma 说的是：测地线的切向量，和以起点为中心的测地线半径方向，是正交的。

放到 CDSM 的语境里：
- $\nabla \Psi_{\text{NC}}$ 是"从 $x$ 指向曲线最近点"的方向（测地线半径方向）
- $\mathcal{P}_{\pi(x) \to x}(\gamma')$ 是"曲线方向搬运过来"的切向量（测地线切方向）

Gauss Lemma 保证了这两个方向 **正交**。

这意味着什么？意味着"往曲线靠拢"和"沿曲线前进"这两个任务是 **decoupled** 的。不会出现"我想沿曲线走，结果不小心离曲线越来越远"的情况。

Lyapunov 函数的导数（Eq. 21）展开后有四个项，其中两个 cross-term 内积恰好因为正交性为零，只剩两个负平方项：

$$ \dot{V} = -k_{\text{NC}}^2 \|\nabla \Psi_{\text{NC}}\|^2 - \frac{k_{\text{TC}}^2}{\beta(x)}(1-\tilde{s}) \cdot C $$

- $\beta(x)$: tubular neighborhood 的 metric distortion factor，恒正
- $C$: $\|\mathcal{P}(\gamma')\|^2 > 0$ 当 $\tilde{s} < 1$

两个负项，只在 $x^* = \gamma(1)$ 时同时为零。Practical stability 得证。

**Intuition**: 想象你在一条河里游泳，河岸就是 curve。Gauss Lemma 保证了水流方向（tangential）永远和"朝岸边游"的方向（normal）垂直。所以你不会因为顺着游而被冲离岸边，也不会因为往岸边游而意外地改变顺流的位置。

---

### Aha 3: Variable Damping = "Variance 变小，硬度变大"

这是 paper 里最 elegant 的工程 trick。

在 dressing task 中，robot 的 damping matrix 不是固定的。paper 的做法是：

1.  把 $K$ 次 demo 在每个 phase $s$ 处对齐
2.  在曲线的 moving tangent space 里算 covariance $\Sigma_s$（Eq. 35-36）
3.  对 $\Sigma_s$ 做特征值分解 $\Sigma_s = V^\top \Lambda V$
4.  Damping matrix $D_s = V \cdot \text{diag}\left(\frac{1}{\lambda_1 + d}, \dots, \frac{1}{\lambda_n + d}\right) \cdot V^\top$（Eq. 37）

- $\lambda_i$: covariance 的第 $i$ 个特征值，代表该方向上的 variability
- $d$: damping floor，防止 $\lambda \to 0$ 时增益爆炸
- $V$: 特征向量矩阵，代表 variability 的"方向"

**Intuition**: demo 数据在某个方向上分散（variance 大），说明这个方向上 robot 可以"自由活动"，damping 应该小；数据在某个方向上集中（variance 小），说明这个方向上要求精确，damping 应该大。

这个思路和 [Calinon et al. 的 TP-GMM](https://ieeexplore.ieee.org/document/7989661) 以及 [Kronander & Billard 的 variable stiffness](https://ieeexplore.ieee.org/document/7125479) 是一脉相承的，但是 CDSM 把它搬到了 full $S_{++}^6$ 上，而且和 pose curve 的 phase 同步。

---

## 三、Bézier on Manifold 的拼装逻辑

这段我觉得是工程上最巧妙的部分。普通人在 manifold 上做 spline 会陷入两种坑：

**坑 1**: 全局用一个 tangent space（single tangent space fallacy）→ 远处严重失真

**坑 2**: 每个点用自己的 tangent space，没有 continuity 约束 → 拼接处断裂

CDSM 的方案是 "think globally, act locally"：

每段 Bézier $u_j(s_j)$ 在自己的 base point $p_j$ 的 tangent space 里定义（Eq. 10），然后 exp 回 manifold（Eq. 11）。

段间 continuity 用两个约束：
- $C^0$: $p_{j+1} = \exp_{p_j}(w_{3,j})$（Eq. 12），即上一段终点 = 下一段起点
- $C^1$: $v_{1,j+1} = \mathscr{P}_{p_j \to p_{j+1}}(v_{2,j})$（Eq. 13），即方向向量通过 parallel transport 对齐

**Intuition**: 你在地球仪上画一条航线，每段都画在当地的 flat 地图上（tangent space），但是段与段衔接时，你要把上一段终点的方向"旋转"到下一段起点的视角里（parallel transport），这样整条线才光滑。

对于 Lie group，parallel transport 退化为 left translation，更简单：
$$ \gamma_j(s) = p_j \cdot \exp(u_j(s)) $$
exp 前面的 $p_j$ 就是 left multiplication。

---

## 四、Phase Modulation: 时间和空间分开管

这是 paper 的第二个 contribution，但我觉得它被低估了。

传统 DMP 用一个 canonical system（通常是一个指数衰减的 ODE）来驱动 phase。这个 system 和 spatial 部分是耦合的，你想改时间 profile 就得动整个 DMP。

CDSM 的做法是：把 phase $s$ 本身也建模成一条 Bézier curve $\gamma^s(t)$（Eq. 23）：
$$ s = \gamma^s(t) = \Psi(t/T) w $$

- $\Psi(t)$: Bernstein polynomial basis matrix
- $w$: 所有控制点拼接的向量
- $T$: 总时长

然后 spatial curve 变成 $\gamma(s(t))$，chain rule（Eq. 24）:
$$ \gamma'(t) = \frac{\delta \gamma}{\delta s} \cdot \frac{\delta s}{\delta t} = \frac{\delta \gamma}{\delta s} \cdot (\gamma^s)'(t) $$

- $\frac{\delta \gamma}{\delta s}$: 空间曲线对 phase 的导数（fixed）
- $(\gamma^s)'(t)$: 时间曲线的导数（可优化）

**Intuition**: 你有一段录好的 GPS 航迹（spatial），你可以选择加速播放、减速播放、或者中间停一下再走（temporal），但路还是那条路。

约束只有一个：$(\gamma^s)'(t) > 0$，即不能倒着走。这个约束保证了 stability analysis 不受影响。

实验里（Fig. 8）他们加了一个速度约束 $\|v(t)\| \le \bar{v} = 3.0$，optimize 后时间 profile 变了但空间不变。

这个思路和 [Rana et al. 的 Variable Gain DMP](https://ieeexplore.ieee.org/document/8793914) 以及 [Polydoros et al. 的 time-varying DMP](https://ieeexplore.ieee.org/document/7139468) 有关，但 CDSM 把它彻底解耦到了一个独立的 curve 上。

---

## 五、实验数据的技术解读

Table I 里有个细节值得展开。

CDSM 的 trajectory distance 是 $0.0042$，比 PUMA（$0.0112$）和 LieFlows（$0.0131$）低一个量级。这看起来太好了，甚至有点可疑。

原因是 **metric 的定义**。trajectory distance 是用 demo 的初始点和 $\Delta t$ 做 forward simulation，然后比 100 步。CDSM 因为是 curve fitting + analytic DS，在 demo 起点上几乎完美贴合。而 learning 方法本质是 approximating 一个 vector field，总有 fitting residual。

更有意思的是 **path distance**（randomized 初始点）:
- CDSM: $0.0339$
- LieFlows: $0.0869$（且 success rate 只有 0.81）
- PUMA: $0.0982$

CDSM 在 OOD 场景下优势更明显。原因在于 Normal Component 的"硬拉力"。Lyapunov 分析里 $\dot{V}$ 的第一项 $-k_{\text{NC}}^2 \|\nabla \Psi_{\text{NC}}\|^2$ 是一个 quadratic 衰减，离曲线越远拉力越强（因为 $\Psi_{\text{NC}} = \frac{1}{2}d^2$，梯度是 $d$ 量级的）。

LieFlows 的 success rate 只有 0.81，主要原因是 $S^2$ 有 antipodal problem。$S^2$ 上两个点如果接近对径点，测地线距离趋于 $\pi$，log map 变得数值不稳定。Normalizing flow 学到的 vector field 在这种地方容易发散到 $-q$（antipode），回不来了。CDSM 因为是 projective + analytic 的，projection 的 non-uniqueness 只在 cut locus 上发生（measure zero set），实践上碰不到。

---

## 六、和 Related Work 的技术坐标系

把 CDSM 放到整个 landscape 里看：

| 类别 | 代表方法 | Stability | Reactivity | Multi-modal | Manifold-native |
|---|---|---|---|---|---|
| Learning on manifold | LieFlows, Riemannian Flow Matching | Learned (soft) | 慢 train | Yes | Yes |
| Loss-constrained DS | PUMA, Stable SPED | Learned (hard) | 慢 train | Yes | Yes |
| DMP on manifold | Abu-Dakka et al. | Analytic | Fast | No | Yes |
| Guiding Vector Field | Yao 2023, PVFC | Analytic | Fast | No | Euclidean mostly |
| **CDSM** | This paper | **Analytic (practical)** | **Fast** | **No** | **Yes** |

CDSM 的 trade-off 是放弃了 multi-modality（一条 curve 只能 encode 一个 mode），换来了解析稳定性、real-time fitting 和 manifold-native 的几何正确性。

这和 [Figures-of-Merits 论文](https://www.science.org/doi/10.1126/scirobotics.abc5044) 里讨论的 "learning vs engineering" 的张力是一致的。在 safety-critical 的 household 场景，engineering approach 的可解释性和速度往往比 learning 的 flexibility 更重要。

---

## 七、几个我觉得可以 push 的方向

基于 paper 自己在 Section VII 提到的 limitation，加上我的联想：

### 1. Multi-modal CDSM
Paper 提到可以 fit 多条 curve，按最近 distance 选 DS。这其实就是 [GMM-based DS](https://ieeexplore.ieee.org/document/5979620) 的 geometric 版本。更 elegant 的做法可能是用 mixture of curves with soft assignment，类似 [TP-GMM](https://ieeexplore.ieee.org/document/7989661) 但是把 task parameters 换成 curve segments。

### 2. Cut Locus 的处理
Paper 承认 GAS 在 cut locus 上不成立。实践中可以用 [Riemannian Fast Marching](https://link.springer.com/article/10.1007/s10851-019-00915-3) 预计算 distance field 的 viscosity solution，处理 non-unique projection 的情况。或者参考 [Chen et al. Neural Geodesic Flows](https://arxiv.org/abs/2402.14006) 用 learned geodesic 来规避解析 log map 的奇异性。

### 3. Energy-Budget Constrained DS
Paper 最后提到可以加 [energy budget constraint](https://journals.sagepub.com/doi/10.1177/02783649211017681) 来 cap Cartesian impedance 的能量。这和 [Passivity-DS](https://ieeexplore.ieee.org/document/7989661) 的思路结合，可以构造一个 port-Hamiltonian 形式的 CDSM：
$$ \dot{x} = J(x) \nabla H + g(x) u, \quad y = g(x)^\top \nabla H $$
其中 $H = V(x)$（CDSM 的 Lyapunov function）就是 Hamiltonian，damping matrix 就是 dissipation rate。这样 variable damping 的 stability 可以通过 passivity theorem 严格证明。

### 4. Diffusion-based Multi-curve Sampling
把 CDSM 的 curve 当作 diffusion model 的 condition，用 [SE(3)-DiffusionFields](https://arxiv.org/abs/2306.10748) 类似的方式 sample 多条 candidate curve，再用 CDSM 的 stability filter 做后处理。这能解决 multi-modal 问题又保留 stability guarantee。

### 5. Hardware-level 的思考
Paper 用 Franka 1kHz 控制 + MediaPipe tracking。MediaPipe 在 occlusion 下 confidence < 0.9 时用 last good estimate，这在快速运动时会有 lag。可以考虑用 [BundleFusion](https://graphics.stanford.edu/papers/bundlefusion/) 或者 [NICE-SLAM](https://arxiv.org/abs/2112.03089) 这类 dense SLAM 来替代，或者在 CDSM 的 phase modulation 里加一个 observation-driven 的 phase estimator，类似 [Intent-aware DMP](https://ieeexplore.ieee.org/document/10160535)。

---

## 八、一句话总结

CDSM 的本质是：**在 manifold 上 fit 一条 Bézier，然后用 parallel transport + Gauss lemma 构造一个解析的、正交分解的 vector field，保证 practical stability，同时把 damping 和 phase 解耦成独立的 curve 以获得 reactivity。**

用最朴素的话说：给 robot 一条弯弯曲曲的路，告诉它在任何时候"先靠过去，再往前走"，并且因为几何上这两件事正交，所以不会打架。

---

### Further Reading

- [Li & Calinon 2025 - Euclidean 基础](https://ieeexplore.ieee.org/document/10610450)
- [Jaquier et al. 2024 - Single Tangent Space Fallacy](https://ieeexplore.ieee.org/document/10610668)
- [Pennec et al. 2006 - $S_{++}^n$ 上的 Riemannian framework](https://link.springer.com/article/10.1007/s11263-005-3252-8)
- [Sola et al. - Micro Lie Theory](https://arxiv.org/abs/1812.01537)
- [Yao 2023 - Guiding Vector Fields on Manifolds](https://link.springer.com/book/10.1007/978-3-031-32615-2)
- [Abu-Dakka et al. 2024 - Geometry-aware DMP](https://www.sciencedirect.com/science/article/pii/S0925231224004982)
- [Kronander & Billard 2016 - Passive Interaction Control with DS](https://ieeexplore.ieee.org/document/7125479)

---

Andrej, 这篇paper的核心在于将 Euclidean space 中的 dynamical systems (DS) 扩展到了 Riemannian manifolds 和 Lie groups。为了 build your intuition, 我们可以从几何拓扑、数学推导到具体的工程实现进行深度的拆解。

paper 的核心贡献是提出了 **CDSM (Curve-induced Dynamical systems on Smooth Manifolds)** 框架。在 robotics 领域, robot 的 pose (位置和姿态) 天然属于 $SE(3)$ Lie group, 而 stiffness 或 damping matrix 属于 $S_{++}^n$ (Symmetric Positive Definite matrices) Riemannian manifold。传统的 learning-based 方法 (如 LieFlows 或 PUMA) 需要长时间的 training, 并且往往只保证收敛到 goal, 缺乏对 demonstrated trajectory 的精确跟踪, 且在 out-of-distribution (OOD) 状态下表现不佳。CDSM 通过在 manifold 上直接拟合一条 reference curve $\gamma(s)$, 然后基于这条曲线构造一个 real-time 的 vector field, 从而同时保证了 stability, reactivity 和 interpretability。

---

### 1. Geometric Intuition: 为什么需要 Parallel Transport?

在 Euclidean space $\mathbb{R}^n$ 中, 如果你想让一个点 $x$ 跟随一条曲线 $\gamma$, 你只需要计算 $x$ 到曲线的最近点 $\pi(x)$, 然后将沿曲线的切向量 $\gamma'(\tilde{s})$ 直接加到 $x$ 上即可。

但在 Riemannian manifold $M$ 上, 向量是被束缚在特定的 tangent space $T_p M$ 中的。点 $x$ 处的切空间 $T_x M$ 与曲线最近点 $\pi(x)$ 处的切空间 $T_{\pi(x)} M$ 是不同的空间。直接将 $T_{\pi(x)} M$ 中的向量应用于 $T_x M$ 会导致严重的 geometric distortion (single tangent space fallacy)。

因此, CDSM 引入了 **Parallel Transport** $\mathcal{P}_{\pi(x) \to x}$:
$$ \mathcal{P}_{p \to q} : T_p M \to T_q M $$
这个操作通过 Levi-Civita connection 定义的 geodesic, 将 $\pi(x)$ 处的曲线切向量 $\gamma'(\tilde{s})$ 无扭曲地“搬运”到 $x$ 所在的 tangent space 中, 从而保证 vector field 的几何正确性。

---

### 2. Curve Construction: Composite Quadratic Bézier on Manifolds

为了从 demonstrations $X^{\text{ref}}$ 中提取曲线, paper 采用了 composite quadratic Bézier curves 的流形扩展版。

曲线被分为 $J$ 个 segments。对于第 $j$ 个 segment, 其局部曲线在 base point $p_j$ 的 tangent space $T_{p_j} M$ 中定义:
$$ u_j(s_j) = (1-s_j)^2 w_{1,j} + 2(1-s_j)s_j w_{2,j} + s_j^2 w_{3,j} $$
*   $s_j \in [0,1]$: local phase variable (局部相位变量)。
*   $w_{i,j}$: control points (控制点), 存在于 $T_{p_j} M$ 中。其中 $w_{1,j} = 0$ (即 base point 自身)。

将 tangent space 中的曲线映射回 manifold:
$$ \gamma_j(s) = \exp_{p_j}(u_j(s)) $$
这里 $\exp$ 是 Exponential map (指数映射), 它将 tangent vector 沿着 geodesic 映射回 manifold 上的点。

为了保证 $C^0$ 和 $C^1$ continuity, 需要满足:
$$ p_{j+1} = \exp_{p_j}(w_{3,j}) $$
$$ v_{1,j+1} = \mathscr{P}_{p_j \to p_{j+1}} (v_{2,j}) $$
这里 $v_{i,j} = w_{i+1,j} - w_{i,j}$ 是连续控制点之间的方向向量。注意 $C^1$ 约束使用了 Parallel Transport $\mathscr{P}$ 来对齐不同 tangent space 中的向量。

---

### 3. Dynamical System Construction (核心数学解析)

给定 manifold 上的 state $x$, 我们需要找到它对应的 desired velocity $\dot{x}$。CDSM 的核心公式 (Eq. 16) 定义为:

$$ \dot{x} = f(x) := \underbrace{k_{\text{TC}} \mathcal{P}_{\pi(x) \to x} (\gamma'(\tilde{s}(x)))}_{\text{Propagation (TC)}} \underbrace{- k_{\text{NC}} \nabla \Psi_{\text{NC}}(x)}_{\text{Convergence (NC)}} $$

变量解析:
*   $\pi(x) = \gamma(\tilde{s}(x))$: closest point projection (最近点投影), 通过在 phase $s \in [0,1]$ 上寻找使 $d_M(x, \gamma(s))$ 最小的 $\tilde{s}$ 得到。
*   $\gamma'(\tilde{s}(x))$: 曲线在 $\pi(x)$ 处的 velocity (切向量), 属于 $T_{\pi(x)} M$。
*   $\mathcal{P}_{\pi(x) \to x}(\dots)$: 将切向量 parallel transport 到 $T_x M$。
*   $k_{\text{TC}}, k_{\text{NC}}$: Tangential Component (TC) 和 Normal Component (NC) 的 gains。
*   $\Psi_{\text{NC}}(x)$: Normal energy potential (法向能量势), 定义为 $\frac{1}{2} d_M^2(x, \pi(x))$。其负梯度 $-\nabla \Psi_{\text{NC}}(x)$ 指向 $x$ 到 $\pi(x)$ 之间的 geodesic 方向, 起到吸引作用。

为了防止在 goal 处 $\gamma(1)$ 发生震荡, TC 项的 gain 被设计为随相位衰减 (Eq. 18):
$$ k_{\text{TC}}(\tilde{s}) = k_{\text{TC}}^c \zeta(\tilde{s}), \quad \text{where } \zeta(\tilde{s}) = (1 - \tilde{s}^{k_g}) $$
*   $k_g > 1$: configuration parameter。$k_g$ 越小, 越早开始减速。

---

### 4. Stability Analysis: Gauss's Lemma 的巧妙应用

要证明 DS 在 manifold 上的稳定性, paper 构造了 Lyapunov candidate (Eq. 19):
$$ V(x) = k_{\text{NC}} \Psi_{\text{NC}}(x) + k_{\text{TC}} \Psi_{\text{TC}}(x) $$
其中 $\Psi_{\text{TC}}(x) = \frac{1}{2} (1 - \tilde{s}(x))^2$。

对 $V(x)$ 求导 (Eq. 20-22), 会得到一个包含四个交叉项的复杂表达式。这里 paper 利用了微分几何中的 **First Variation of Arc Length (Gauss's Lemma)**:
**Geodesic radius vector (即 $\nabla \Psi_{\text{NC}}$) 严格正交于 parallel transported tangent vector (即 $\mathcal{P}_{\pi(x) \to x}(\gamma')$)。**

因此, 展开后的交叉项 (Term A 和 Term B) 内积为 0, 直接消去了。Lyapunov 导数简化为:
$$ \dot{V}(x) = - k_{\text{NC}}^2 \|\nabla \Psi_{\text{NC}}(x)\|^2 - \frac{k_{\text{TC}}^2}{\beta(x)} (1 - \tilde{s}(x)) C $$
*   $\beta(x)$: metric distortion in tubular neighborhood, $\beta(x) > 0$。
*   $C > 0$ 当 $\tilde{s} < 1$ 时。

由于 $\dot{V}(x) \le 0$ 且仅在 $x^* = \gamma(1)$ 时为 0, 证明了系统的 practical stability (由于 manifold 存在 cut locus $\mathcal{C}_{\text{sing}}$ 导致投影非唯一, 全局渐近稳定性 GAS 无法保证, 但 practical stability 足够用于 robotics)。

---

### 5. Robotics-Relevant Manifolds: $SE(3)$ 与 $S_{++}^n$

#### A. Poses on $SE(3)$
$SE(3)$ 被表示为单位四元数与平移的半直积 $G = \mathbb{H}_1 \ltimes \mathbb{R}^3$。
Exponential map (Eq. 25):
$$ q = \cos(\theta/2) + \frac{\omega}{\theta}\sin(\theta/2), \quad t = V(\omega)v $$
*   $\omega \in \mathbb{R}^3$: angular velocity (角速度)。
*   $v \in \mathbb{R}^3$: linear velocity (线速度)。
*   $V(\omega)$: left Jacobian of $SO(3)$ (Eq. 26)。

距离定义为 Screw Distance (Eq. 29):
$$ d_G(g_1, g_2) = \|\log(g_1^{-1} g_2)\|_{\mathfrak{g}} = \sqrt{\eta \|\omega\|^2 + \|v\|^2} $$
*   $\eta = L_c^2$: weighting parameter, 用于解决 radians 和 meters 的单位不一致问题。

#### B. Symmetric Positive Definite Matrices $S_{++}^n$
对于 damping matrix, $S_{++}^n$ 的 tangent space 是 Symmetric matrices $S^n$。
Exponential map (Eq. 31):
$$ \exp_P(V) = P^{\frac{1}{2}} \text{expm}(P^{-\frac{1}{2}} V P^{-\frac{1}{2}}) P^{\frac{1}{2}} $$
Geodesic distance (Eq. 33):
$$ d_{S_{++}^n}(P, Q) = \|\ln(P^{-\frac{1}{2}} Q P^{-\frac{1}{2}})\|_F = \left( \sum_{i=1}^n \ln^2 \lambda_i \right)^{\frac{1}{2}} $$
*   $\lambda_i$: $P^{-1}Q$ 的实正特征值。

#### C. Coupling Poses and Variable Damping
在 dressing task 中, paper 将 $SE(3)$ 上的 DS 与 $S_{++}^6$ 上的 damping matrix 曲线同步。
Damping matrix 由 demonstration 的 covariance 推导得出 (Eq. 35-37):
$$ D_s = V \text{diag}\left(\frac{1}{\lambda_1 + d}, \dots, \frac{1}{\lambda_n + d}\right) V^\top $$
*   $\Sigma_s = V^\top \Lambda V$: 协方差矩阵的特征值分解。
*   $d$: damping gain。
这本质上将 demonstration 的方差转化为 impedance control 的阻尼: variance 越大的方向, 允许的 compliance 越大, damping 越小。

---

### 6. Phase Modulation Layer: 时空解耦

为了将 spatial profile 和 temporal profile 解耦, paper 引入了 phase modulation curve $\gamma^s(t)$。
空间曲线变为 $\gamma(s(t))$, 根据链式法则 (Eq. 24):
$$ \gamma'(t) = \frac{\delta \gamma}{\delta s} \frac{\delta s}{\delta t} = \frac{\delta \gamma}{\delta s} (\gamma^s)'(t) $$
通过约束 $(\gamma^s)'(t) > 0$, 可以独立优化时间曲线 (如满足速度限制 $\|v(t)\| \le \bar{v}$), 而不破坏 DS 的 stability analysis。

---

### 7. Experiments Analysis: CDSM vs. LieFlows & PUMA

在 LASA dataset 映射到 $S^2$ 的 benchmark 中, CDSM 展现了压倒性优势 (Table I):

| Method | Trajectory Dist. | Path Dist. (randomized) | Success Rate | Comp. Time (traj) | Train/Fitting Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CDSM (ours)** | **0.0042 ± 0.0087** | **0.0339 ± 0.1292** | **1.0** | **0.0039 ± 0.0006** | **1.26 ± 0.16** |
| Lieflows | 0.0131 ± 0.0072 | 0.0869 ± 0.5574 | 0.81 | 0.5157 ± 0.0281 | 1324.85 ± 8.47 |
| PUMA | 0.0112 ± 0.0044 | 0.0982 ± 0.2092 | 1.0 | 0.0562 ± 0.0111 | 204.03 ± 10.63 |

**深度分析:**
1.  **Trajectory Distance**: CDSM 最低。由于它是 deterministic 的曲线拟合 + 解析 vector field, 精度极高; 而 LieFlows 和 PUMA 依赖 Neural ODE 或 loss 约束, 存在 fitting 误差。
2.  **Path Distance (OOD robustness)**: 从随机初始点出发, CDSM 的平均偏离度仅为 LieFlows 的约 1/2.5。原因是 CDSM 显式构造了 Normal Component (NC), 只要离开 cut locus, 引力会迅速将 state 拉回曲线, 而 learning 方法在 OOD 区域容易发散或陷入 antipodal pole (如 LieFlows 收敛到 $-q$ 导致 success rate 仅 0.81)。
3.  **Computation/Train Time**: CDSM 拟合仅需 ~1.26s, 使用 JAX 加速后 forward simulation 速度比 LieFlows 快两个数量级。这使得 robot 能够 online 重算曲线 (dressing task 中实时追踪手臂姿态)。

在 Franka robot 的真实实验中, 当施加外部 perturbation (推、拉、扭) 时, 变阻尼控制律 (Eq. 38) 结合 CDSM 展现了良好的 compliance 与 robustness, 扰动解除后能平滑收敛回 nominal curve。

---

### 8. Intuition Building: 核心 takeaways

1.  **Manifold-native vs Tangent-space projection**: 以往方法 (如 [10]) 是在 $\mathbb{R}^n$ 中算出 velocity 再 project 到 manifold 的 tangent space, 这破坏了 Riemannian metric 的几何意义。CDSM 则完全在 manifold 上定义 distance, 并用 Parallel Transport 搬运切向量, 几何上 self-consistent。
2.  **Analytical Vector Field vs Learned DS**: CDSM 本质上是一种 "Guiding Vector Field" 的流形推广。它牺牲了 learning 那种从 large dataset 中提取 multi-modal distribution 的能力, 换来了极低的 sample complexity (单次或少量 demo 即可)、可证明的 stability 以及极高的 reactivity。在 household 这种 safety-critical 且环境 dynamic 变化的场景, deterministic 且 fast 的重计算 比 hours 的 training 更有价值。
3.  **Orthogonality is Key**: 系统稳定性的核心证明依赖于 Gauss's Lemma 带来的 $\nabla \Psi_{\text{NC}} \perp \mathcal{P}(\gamma')$。如果切向传播速度在法向引力方向上有投影, 就会产生“螺旋发散”或无法收敛到曲线。正交性保证了“向曲线收敛”和“沿曲线前进”两个 task 互不干扰。

### References
*   paper 链接 (arXiv): [Curve-Induced Dynamical Systems on Riemannian Manifolds and Lie Groups](https://arxiv.org/abs/2509.19696) (注: 依据文中 reference 13 的 arXiv ID 推测的链接)
*   Li & Calinon Euclidean 基础: [From movement primitives to distance fields to dynamical systems](https://ieeexplore.ieee.org/document/10610450)
*   LieFlows 对比方法: [Learning stable vector fields on Lie groups](https://ieeexplore.ieee.org/document/9812529)
*   PUMA 对比方法: [PUMA: Deep Metric Imitation Learning for Stable Motion Primitives](https://onlinelibrary.wiley.com/doi/10.1002/aisy.202400144)
*   Single tangent space fallacy 相关讨论: [Unraveling the single tangent space fallacy](https://ieeexplore.ieee.org/document/10610668)
*   JAX 框架: [JAX: composable transformations of Python+NumPy programs](https://github.com/jax-ml/jax)
