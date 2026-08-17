---
source_pdf: Geometrical Formulation of Quantum Mechanics.pdf
paper_sha256: 0a8c1f6ac37098c71629447c0743bb4da88b66e6a5f8500b1483e3951da1ab16
processed_at: '2026-08-04T21:20:24-07:00'
target_folder: Quantum-Computing
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 Paper

## 一句话总结

Textbook 里讲 quantum mechanics 总是强调 "线性" —— state 是 vector, observable 是 matrix, Schrödinger equation 是线性方程。**这其实是误导**。物理上真正有意义的是 projective Hilbert space $\mathcal{P}$, 它是个 nonlinear 的弯曲 manifold, 上面有 symplectic structure (像 classical mechanics) 和 Riemannian metric (quantum 独有)。整个 QM 完全可以用 pure geometry 讲, 一行线性代数都不需要。

---

## 为什么 Textbook 的讲法有问题

打开任何一本 QM textbook, 第一页就告诉你: state 是 Hilbert space $\mathcal{H}$ 里的 vector $\Psi$, observable 是 self-adjoint operator $\hat{A}$, measurement 给 eigenvalue, probability 是 $|\langle\phi|\psi\rangle|^2$, evolution 是 $\hat{H}\Psi = i\hbar \dot{\Psi}$。

全是 linear algebra。

但你仔细想想有个 bug: $\Psi$ 和 $e^{i\theta}\Psi$ 是**同一个物理 state**。你做任何 measurement, 算任何 expectation value, 它们给的结果完全一样。所以 Hilbert space 里有大量 "gauge redundancy" —— 同一个物理 state 对应 infinitely many vectors, 只差一个 phase。

那真正的 state space 是什么? 是 rays, 也就是 $\mathcal{P} = \mathcal{H}/\mathbb{C}^\times$ (把所有只差一个 complex scalar 的 vector 粘成一点)。这个 $\mathcal{P}$ 是 **nonlinear** 的 manifold。比如 $\mathcal{H} = \mathbb{C}^{n+1}$, 那 $\mathcal{P} = \mathbb{CP}^n$, 复射影空间, 它是弯曲的。

所以 textbook 把你锁在一个 "方便但 redundant" 的 linear space 里, 真正的 physics 发生在一个 nonlinear manifold 上。Ashtekar 说: 这就像 special relativity 里 Einstein 一开始用 inertial frames 和 coordinates, Minkowski 后来用 pure geometry reformulate, 才发现 line element 和 hyperbolic geometry 是 essential 的, coordinates 只是 convenience。这里也是一样: Hilbert space 的 linear structure 是 convenience, Kähler geometry on $\mathcal{P}$ 才是 essential。

---

## $\mathcal{P}$ 上有什么 structure

### Symplectic structure $\omega$ —— "classical 的部分"

Hermitian inner product $\langle\Phi, \Psi\rangle$ 有 real part 和 imaginary part:

$$\langle \Phi, \Psi \rangle = \frac{1}{2\hbar} G(\Phi, \Psi) + \frac{i}{2\hbar} \Omega(\Phi, \Psi)$$

- $G$: real, symmetric, positive definite → Riemannian metric
- $\Omega$: real, skew-symmetric, non-degenerate → symplectic form
- 它俩通过 complex structure $J$ (乘以 $i$) 联系: $G(\Phi, \Psi) = \Omega(\Phi, J\Psi)$

这个 triple $(J, G, \Omega)$ 就是 **Kähler structure**。

Symplectic form $\Omega$ (descend 到 $\mathcal{P}$ 上变成 $\omega$) 干什么用呢? 它 define Poisson bracket, define Hamiltonian vector field, define dynamics。**和 classical mechanics 的 phase space 上的 symplectic form 完全一样的角色。**

具体地, 对每个 observable $\hat{F}$, define expectation value function:

$$F(\Psi) = \langle\Psi, \hat{F}\Psi\rangle$$

这个 $F$ 是 $\mathcal{H}$ 上的 real function。因为 $F(e^{i\theta}\Psi) = F(\Psi)$, 它 descend 到 $\mathcal{P}$ 上变成 $f: \mathcal{P} \to \mathbb{R}$。

然后神奇的事情发生: Schrödinger equation $\dot{\Psi} = -(i/\hbar)\hat{H}\Psi$ 在 $\mathcal{P}$ 上看, 就是 $f = h$ (energy expectation value) 的 **Hamiltonian flow**:

$$\dot{p} = X_h(p)$$

完全和 classical mechanics 的 Hamilton's equation 一样的形式! Schrödinger equation 就是 Hamilton's equation **in disguise**。

两个 observables 的 Poisson bracket 也正好对应 commutator:

$$\{f, k\}_\omega = \left\langle \frac{1}{i\hbar}[\hat{F}, \hat{K}] \right\rangle$$

所以 commutator Lie algebra 就是 Poisson bracket Lie algebra, symplectic structure 完全 capture 了 classical-like 的部分。

### Riemannian metric $g$ —— "quantum 的部分"

那 quantum 和 classical 的差别在哪? 在 Riemannian metric $g$ (从 $G$ descend 来的)。

Classical phase space 只有 symplectic form, 没有 Riemannian metric。Quantum phase space $\mathcal{P}$ 两者都有。所有 quantum-specific 的 features 都来自 metric:

**1. Uncertainty**:
$$(\Delta \hat{F})^2 = \{F, F\}_+ - F^2 = \frac{\hbar}{2} G(X_F, X_F) - F^2$$

其中 $\{F, F\}_+$ 是 Riemann bracket, 由 metric 定义。Variance 就是 Hamiltonian vector field 的 "长度" (minus 平方的 expectation)。没有 metric 就没有 uncertainty 的概念。

**2. Transition probability**:
$$|\langle\Psi_0, \Psi\rangle|^2 = \cos^2\left(\frac{\sigma(p_0, p)}{\sqrt{2\hbar}}\right)$$

其中 $\sigma(p_0, p)$ 是 $\mathcal{P}$ 上的 geodesic distance (用 metric $g$ 算的)。**Measurement probability 完全由 geodesic distance 决定**。两个 state 在 $\mathcal{P}$ 上越 "近", transition probability 越大。这就是 Fubini-Study metric 的 deep 含义。

**3. State reduction (collapse)**:
测量 observable $f$ 得 eigenvalue $\lambda$ 后, state collapse 到 eigenmanifold $\mathcal{E}_\lambda$ 上。Collapse 到**哪个点**? **离 $p_0$ 最近的那个点** (geodesically)!

$$P_\lambda(p_0) = \arg\min_{q \in \mathcal{E}_\lambda} \sigma(p_0, q)$$

Collapse probability:
$$\Pr(\lambda) = \cos^2\left(\frac{\sigma(p_0, \mathcal{E}_\lambda)}{\sqrt{2\hbar}}\right)$$

**State reduction 就是 geometric projection 到最近的 eigenstate**。这是一个非常 elegant 的 picture。

**4. Anandan-Aharonov interpretation**:
$$(\Delta h)^2 = \frac{\hbar}{2} g(X_h, X_h)$$

Energy uncertainty 就是 "state 在 $\mathcal{P}$ 上运动的速度" 的平方。Energy eigenstate ($\Delta h = 0$) 不运动 (stationary), energy 不确定的 state 在 $\mathcal{P}$ 上 "跑" 得快。

---

## Observables 的 intrinsic characterization

Textbook 说: observable 是 self-adjoint operator。但这 reference 了 Hilbert space。

Geometric formulation 要 intrinsic: 怎么在 $\mathcal{P}$ 上直接 characterize observables?

Key observation: Schrödinger vector field $Y_{\hat{F}}$ 生成 unitary transformations, unitary transformations preserve Hermitian inner product, 所以 preserve $G$ 和 $\Omega$。因此 $X_f$ (descend 到 $\mathcal{P}$ 上) 是 **Killing vector field** (preserve metric $g$)。

**Corollary 1**: Smooth function $f: \mathcal{P} \to \mathbb{R}$ 是 observable **iff** 它的 Hamiltonian vector field $X_f$ 是 Killing vector field。

对比 classical mechanics: observable 是 arbitrary smooth function (Hamiltonian vector field 只需要 preserve $\omega$, 这是自动的)。Quantum mechanics 多了 "preserve $g$" 的 requirement, 所以 observable space 小很多。

这就解释了为什么 finite-dim Hilbert space ($\dim = n$) 有 $n^2$ 个 independent observables (Hermitian matrices), 但 smooth functions on $\mathcal{P} = \mathbb{CP}^{n-1}$ 是 infinite-dimensional。Killing condition 把 infinite-dim function space cut down 到 finite-dim。

---

## 重新表述 Postulates

整个 QM 可以用 pure geometric language 写出来:

**(P)** Physical states 是 Kähler manifold $\mathcal{P}$ 上的 points (projective Hilbert space)。

**(K)** Evolution 是 $\mathcal{P}$ 上 preserve Kähler structure 的 flow。

**(O)** Observables 是 $\mathcal{P}$ 上的 smooth real functions $f$, 其 Hamiltonian vector field $X_f$ preserve Kähler structure (即 Killing + symplectic-preserving)。

**(I)** 测量 $f$ 得到 $f \in \Lambda$ 的 probability:

$$\Pr = \cos^2\left(\frac{\sigma(p, P_{f,\Lambda}(p))}{\sqrt{2\hbar}}\right)$$

其中 $\sigma$ 是 geodesic distance, $P_{f,\Lambda}(p)$ 是 $p$ 到 eigenmanifold $\mathcal{E}_{f,\Lambda}$ 的最近点。

**(R)** Measurement 后 state collapse 到这个最近点 $P_{f,\Lambda}(p)$。

**一行 Hilbert space 都没提!** 全是 geometry。

---

## 这有什么用? —— Generalizations

如果 QM 只是 reformulate, 那就是好看的 mathematical exercise。但这 reformulate 的 power 在于它**suggest generalizations**。

### Generalized dynamics (保守路线)

Standard QM: Hamiltonian flow 必须 preserve 整个 Kähler structure ($\omega$ + $g$)。

Generalized dynamics: 只要求 preserve $\omega$ (像 classical mechanics), 不要求 preserve $g$。

这在 geometric language 里是最 natural 的 generalization: 允许任意 smooth function $f: \mathcal{P} \to \mathbb{R}$ 作为 Hamiltonian, 不要求 $X_f$ 是 Killing。

Ashtekar 指出, 文献里那些 "non-linear Schrödinger equation" 其实就是这个:

**Non-linear Schrödinger equation** (Birula-Mycielski):
$$i\hbar \dot{\Psi} = \hat{H}_0 \Psi + \epsilon |\Psi|^2 \Psi$$

**Logarithmic equation** (Birula-Mycielski):
$$i\hbar \dot{\Psi} = \hat{H}_0 \Psi - b\ln(|\Psi|^2) \Psi$$

**Weinberg's framework**: Weinberg 1989 年提出一个 general framework for non-linear QM, 要求 Hamiltonian function on $\mathcal{H}$ 是 homogeneous degree 2 + phase-invariant。Ashtekar 指出, 这恰好等价于 "smooth function on $\mathcal{P}$"。

所以这些 "不同的" proposals 其实是**同一个东西**的不同 presentation:
- Geometric formulation: Hamiltonian function on $\mathcal{P}$
- Weinberg: Homogeneous degree-2 phase-invariant function on $\mathcal{H}^\times$
- Non-linear Schrödinger: 在 $\mathcal{H}$ 上写 non-linear equation, 但 project 到 $\mathcal{P}$ 后是 Hamiltonian flow

Weinberg 说 non-linear Schrödinger equation 的结果 "of no use" 到他的 framework, 这是 **misconception**。从 $\mathcal{P}$ perspective, 它们 induce 相同的 flow on $\mathcal{P}$, 只是在 $\mathcal{H}$ 上的 lift 不同。物理上等价。

### Generalized kinematics (激进路线)

更激进: 换 state space 本身。用 arbitrary Kähler manifold $(\mathcal{M}, g, \omega)$ 作为 quantum phase space。

那 standard QM 被什么条件挑出来? Ashtekar 给出 **reconstruction theorem**:

Standard QM (finite-dim) 等价于: $\mathcal{M}$ 是 complete, simply-connected Kähler manifold, observable algebra 是 maximal, 且 closed under symmetric (Jordan) bracket。

"Maximal" 的意思: 在每一点 $p$, 任何 "symmetry data" $(\lambda, X, K)$ 都能 integrate 到 global observable。也就是 phase space "admit as many observables as possible"。

关键结果: maximality + Jordan closure ⇒ Riemann tensor 是 **constant holomorphic sectional curvature (CHSC) $= 2/\hbar$**:

$$R_{\alpha\beta\gamma\delta} = \frac{C}{2}\left[g_{\gamma[\alpha}g_{\beta]\delta} + \omega_{\alpha\beta}\omega_{\delta\gamma} - \omega_{\gamma[\alpha}\omega_{\beta]\delta}\right]$$

with $C = \hbar/2$。

Finite-dim case: complete + simply-connected + CHSC $= 2/\hbar$ 的 Kähler manifold 一定是 $\mathbb{CP}^n$ (或 non-compact dual)。所以这些条件完全 characterize standard QM。

Infinite-dim case: **open problem**! 可能存在其他 infinite-dim Kähler manifolds with CHSC $= 2/\hbar$ 不 isomorphic 到 projective Hilbert space。如果有, 它们就是 non-trivial kinematical generalizations of QM, 而且可能还能有 consistent measurement theory。这是 paper 最 exciting 的 open direction。

这个 characterization 就像 Riemannian geometry 里 "maximally symmetric spaces are constant curvature spaces" 一样。Standard QM 是 "maximally symmetric" 的 Kähler manifold, 就像 sphere/flat space 是 maximally symmetric Riemannian manifolds。General relativity 就是 generalize 到 non-constant curvature。或许 future quantum gravity 也需要 generalize 到 non-CHSC Kähler?

---

## Semi-Classical Picture: $\mathcal{P}$ 作为 Bundle over $\Gamma$

这个 picture 我觉得最 elegant。

设经典 phase space $\Gamma$ (finite-dim, $2n$-dim), 量子 phase space $\mathcal{P}$ (infinite-dim)。

Define map: $\rho: \mathcal{P} \to \Gamma$, 把 quantum state $x$ map 到 $(\langle\hat{Q}_i\rangle, \langle\hat{P}_j\rangle)$, 即 elementary observables 的 expectation values。

$\rho$ 是 bundle projection! $\mathcal{P}$ 是 bundle over $\Gamma$。

- **Vertical directions**: 沿着 fiber, elementary observables 不变
- **Horizontal directions**: $\omega$-orthogonal to vertical, 沿着这些方向 expectation values change

Classical symplectic structure 就是 quantum symplectic structure 的 "horizontal part":
$$\alpha(\xi, \zeta) = \omega(\tilde{\xi}, \tilde{\zeta})$$

其中 $\tilde{\xi}, \tilde{\zeta}$ 是 horizontal lifts。

**Horizontal sections 是什么?** 恰好是 generalized coherent states (Perelomov 的 construction)!

Coherent states 是 "most classical" 的 quantum states —— 在这些 sections 上, 量子 uncertainty 是 constant 的, 量子 dynamics reduce 到 classical dynamics (对 harmonic oscillator)。

对 harmonic oscillator 具体: quantum Hamiltonian 拆成 classical part $h_0 = p^2/(2m) + m\omega^2 q^2/2$ 和 uncertainty part $h_\Delta = (\Delta p)^2/(2m) + m\omega^2(\Delta q)^2/2$。

$X_{h_\Delta}$ 是 vertical (在 coherent state section 上为零)。所以 coherent state section 被 evolution preserve。这个 section 对应 oscillator ground state 生成的标准 coherent states。

**WKB approximation 也是 generalized dynamics!** WKB equation (drop quantum potential) 对应一个 Hamiltonian flow on $\mathcal{P}$, preserve $\omega$ 但不 preserve $g$ in general。所以 WKB 就是 Weinberg-type non-linear dynamics 的一个 example。

---

## 一些 Intuition 上的 Takeaways

### 1. Linear structure 是 emergent 的 convenience

Hilbert space 的 linear structure 不是 fundamental, 是我们为了方便 "embed" $\mathcal{P}$ 到一个 linear space 里。就像 Riemannian geometry 里我们可以把球面 embed 到 $\mathbb{R}^3$ 里做计算, 但球面的 intrinsic geometry 不依赖 embedding。

### 2. Quantum 的 "extra ingredient" 是 metric

Classical phase space 有 symplectic structure, 没 metric。Quantum phase space 多了 Riemannian metric, 所有 quantum-specific 的东西 (uncertainty, probability, collapse) 都来自 metric。

### 3. Schrödinger = Hamilton

Schrödinger equation 不是什么新的 quantum 东西, 它就是 Hamilton's equation, 只是 Hamiltonian function 是 energy expectation value, phase space 是 $\mathcal{P}$。

### 4. Probability = Geometry

Quantum probability 不是 "fundamental randomness", 它是 geometry: 两个 state 在 $\mathcal{P}$ 上的 geodesic distance 决定 transition probability。远的 state 不容易 collapse 过去。

### 5. Measurement = Projection

State reduction 就是 geometric projection 到最近的 eigenstate。不是神秘的 "wave function collapse", 是 Riemannian geometry 里的 nearest-point projection。

### 6. Generalizations 的方向

要 generalize QM:
- **保守**: 换 dynamics, 允许 non-Killing Hamiltonian (Weinberg, non-linear Schrödinger)
- **激进**: 换 state space, 用其他 Kähler manifold (open problem, 可能用于 quantum gravity)

### 7. 和 ML 的 loose analogy

给 Karpathy 的 loose analogy:

- $\mathcal{P}$ 像 "data manifold", $\Gamma$ 像 "effective low-dim representation"
- Coherent states 像 "learned features" — 把 high-dim quantum state project 到 low-dim classical phase space 还能 preserve dynamics
- Horizontal section 像 "optimal submanifold" — 在这上面 dynamics 最好地 approximate classical
- WKB approximation 像 "low-rank approximation" — 丢掉 high-order terms 还能保持 main dynamics

Reference:
- [Projective Hilbert space](https://en.wikipedia.org/wiki/Projective_Hilbert_space)
- [Fubini-Study metric](https://en.wikipedia.org/wiki/Fubini%E2%80%93Study_metric)
- [Kähler manifold](https://en.wikipedia.org/wiki/K%C3%A4hler_manifold)
- [Geometric quantization](https://en.wikipedia.org/wiki/Geometric_quantization)
- [Coherent states](https://en.wikipedia.org/wiki/Coherent_states)
- [Weinberg's non-linear QM paper](https://doi.org/10.1016/0003-4916(89)90276-5)
- [Ashtekar-Schilling thesis](https://apps.dtic.mil/sti/pdfs/ADA315190.pdf)
- [Anandan-Aharonov: Geometry of quantum evolution](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.65.1697)
- [Kibble 1979: Geometrization of QM](https://link.springer.com/article/10.1007/BF01197539)

---

# Geometrical Formulation of Quantum Mechanics — 详细讲解

这篇 paper 是 Abhay Ashtekar (loop quantum gravity 的创始人之一) 和 Troy Schilling 在 1997 年写的。它的核心 thesis 非常 elegant: standard QM 在 textbook presentation 中强调 Hilbert space 的 linear structure, 但是物理上真正 meaningful 的 state space 是 projective Hilbert space $\mathcal{P}$, 它是一个 nonlinear 的 Kähler manifold。所以 QM 完全可以 reformulate 成 geometric 语言, 类似于 classical mechanics 的 symplectic formulation。这个 reformulation 揭示了 quantum-classical correspondence 的本质, 同时为 generalizations of QM 提供了 natural framework。

让我从 big picture intuition 开始, 逐步深入到技术细节。

---

## 1. Big Picture Intuition

### 1.1 两种 mechanics 的结构对比

| 结构 | Classical Mechanics | Quantum Mechanics (textbook) | Geometric QM |
|------|---------------------|------------------------------|--------------|
| State space | Symplectic manifold $(\Gamma, \alpha)$ | Hilbert space $\mathcal{H}$ (linear) | Projective Hilbert space $\mathcal{P}$ (Kähler, nonlinear) |
| Observables | Smooth functions $f: \Gamma \to \mathbb{R}$ | Self-adjoint operators $\hat{F}$ | Special functions $f: \mathcal{P} \to \mathbb{R}$ (Killing) |
| Algebra | Commutative product + Poisson bracket | Jordan product + commutator | Symmetric bracket + Poisson bracket |
| Dynamics | Hamiltonian flow of $H$ | Schrödinger equation $\hat{H}$ | Hamiltonian flow of $h$ on $\mathcal{P}$ |
| Measurement | $f(p)$ with certainty | Eigenvalues + probabilities | Critical values + geodesic probabilities |
| Uncertainty | Absent | Heisenberg relation | Via Riemannian metric $g$ |

### 1.2 Key insight: Symplectic vs Riemannian 的分工

这是整个 paper 的 conceptual core:

- **Symplectic structure $\omega$** (from $\text{Im}\langle\cdot,\cdot\rangle$): 承担 "classical-like" 的部分 — dynamics, commutator Lie algebra, Poisson bracket
- **Riemannian metric $g$** (from $\text{Re}\langle\cdot,\cdot\rangle$): 承担 "quantum-specific" 的部分 — uncertainty, probabilistic interpretation, state reduction

经典 phase space 只有 symplectic structure, 没有 Riemannian metric, 所以没有 uncertainty, 没有 probabilistic measurement outcome, 没有 state reduction。Quantum phase space 多了一个 Riemannian metric, 这就是 quantum 的 "extra ingredient"。

参考: [Ashtekar 的 Penn State 主页](https://www.phys.psu.edu/~ashtekar/), [Kähler manifold - Wikipedia](https://en.wikipedia.org/wiki/K%C3%A4hler_manifold)

---

## 2. Hilbert Space 作为 Kähler Space

### 2.1 Decomposition of Hermitian inner product

考虑一个 complex Hilbert space $\mathcal{H}$。把它看作 real vector space, 配备 complex structure $J$ (multiplication by $i$, 满足 $J^2 = -I$)。

Hermitian inner product 分解:

$$\langle \Phi, \Psi \rangle = \frac{1}{2\hbar} G(\Phi, \Psi) + \frac{i}{2\hbar} \Omega(\Phi, \Psi) \tag{2.1}$$

变量解释:
- $\Phi, \Psi \in \mathcal{H}$: state vectors
- $G(\Phi, \Psi) = \text{Re}(2\hbar\langle\Phi,\Psi\rangle)$: real, positive definite, symmetric bilinear form (Riemannian metric)
- $\Omega(\Phi, \Psi) = \text{Im}(2\hbar\langle\Phi,\Psi\rangle)$: real, skew-symmetric, non-degenerate 2-form (symplectic form)
- $1/(2\hbar)$: normalization factor, 选这个是为了后面 expectation value function 直接对应物理量

关键 compatibility relation:

$$G(\Phi, \Psi) = \Omega(\Phi, J\Psi) \tag{2.2}$$

这意味着 triple $(J, G, \Omega)$ 构成 **Kähler structure**。任何 Hilbert space 自然就是一个 Kähler space。

### 2.2 Schrödinger vector field = Hamiltonian vector field

定义 Schrödinger vector field:

$$Y_{\hat{F}}(\Psi) := -\frac{1}{\hbar} J \hat{F} \Psi \tag{2.3}$$

变量解释:
- $Y_{\hat{F}}$: vector field on $\mathcal{H}$ generated by observable $\hat{F}$
- $J$: complex structure (multiplication by $i$)
- $\hat{F}$: self-adjoint operator
- 负号: convention, 使 Schrödinger equation 形式标准

Schrödinger equation 在这个 notation 下变成: $\dot{\Psi} = -(1/\hbar) J \hat{H} \Psi = Y_{\hat{H}}(\Psi)$。

定义 expectation value function:

$$F(\Psi) := \langle \Psi, \hat{F}\Psi\rangle = \frac{1}{2\hbar} G(\Psi, \hat{F}\Psi) \tag{2.4}$$

关键计算 (Eq. 2.5): 对任意 tangent vector $\eta$ at $\Psi$,

$$(dF)(\eta) = \langle\Psi, \hat{F}\eta\rangle + \langle\eta, \hat{F}\Psi\rangle = \frac{1}{\hbar} G(\hat{F}\Psi, \eta) = \Omega(Y_{\hat{F}}, \eta) = (i_{Y_{\hat{F}}}\Omega)(\eta)$$

这里用到了 self-adjointness of $\hat{F}$, Eq. (2.2), 和 Eq. (2.3)。

**结论**: Schrödinger vector field $Y_{\hat{F}}$ 恰好就是 expectation value function $F$ 的 Hamiltonian vector field $X_F$。

**Schrödinger equation 就是 Hamilton's equation in disguise!** Hamiltonian function 就是 energy expectation value $h(\Psi) = \langle\Psi, \hat{H}\Psi\rangle$。

### 2.3 Poisson bracket 对应 commutator

两个 expectation value functions $F, K$ 的 Poisson bracket:

$$\{F, K\}_\Omega = \Omega(X_F, X_K) = \left\langle \frac{1}{i\hbar}[\hat{F}, \hat{K}] \right\rangle \tag{2.6}$$

变量解释:
- $\{F, K\}_\Omega = \Omega_{ab} X_F^a X_K^b$: 由 $\Omega$ 定义的 Poisson bracket
- $[\hat{F}, \hat{K}] = \hat{F}\hat{K} - \hat{K}\hat{F}$: operator commutator
- $\langle\cdot\rangle$: expectation value

注意: 这里的 Poisson bracket 是 "quantum" Poisson bracket, 由 Hermitian inner product 的 imaginary part 定义。它 **不是** Dirac correspondence principle 中的 classical Poisson bracket。这两者形式上一样, 但定义在不同 space 上。

参考: [Symplectic geometry - Wikipedia](https://en.wikipedia.org/wiki/Symplectic_geometry), [Hamiltonian vector field - Wikipedia](https://en.wikipedia.org/wiki/Hamiltonian_vector_field)

---

## 3. Riemannian Metric 和 Uncertainty

### 3.1 Riemann bracket 对应 Jordan product

定义 Riemann bracket:

$$\{F, K\}_+ := \frac{\hbar}{2} G(X_F, X_K) = \left\langle \frac{1}{2}[\hat{F}, \hat{K}]_+ \right\rangle \tag{2.7}$$

变量解释:
- $\{F, K\}_+$: Riemann bracket
- $[\hat{F}, \hat{K}]_+ = \hat{F}\hat{K} + \hat{K}\hat{F}$: Jordan product (anti-commutator)
- $G$: Riemannian metric
- $\hbar/2$: normalization 使之 match Jordan product

Classical phase space 没有 Riemannian metric, 所以 Riemann bracket 在 classical mechanics 中没有 analogue。这就是为什么 Jordan product 是 quantum-specific 的。

### 3.2 Uncertainty 的几何 expression

Uncertainty:

$$(\Delta \hat{F})^2 = \langle \hat{F}^2\rangle - \langle\hat{F}\rangle^2 = \{F, F\}_+ - F^2 \tag{2.8}$$

变量解释:
- $(\Delta\hat{F})^2$: variance of $\hat{F}$ in state $\Psi$ (unit norm)
- $\{F,F\}_+$: Riemann bracket of $F$ with itself
- $F^2 = \langle\hat{F}\rangle^2$: square of expectation value

Heisenberg uncertainty relation 的 geometric form:

$$(\Delta\hat{F})^2 (\Delta\hat{K})^2 \geq \left(\frac{\hbar}{2}\{F,K\}_\Omega\right)^2 + (\{F,K\}_+ - FK)^2 \tag{2.10}$$

变量解释:
- LHS: product of variances
- 第一项: 由 symplectic structure (commutator) 贡献, 对应 classical-like 的 non-commutativity
- 第二项: 由 Riemannian metric (Jordan product) 贡献, 对应 quantum covariance

经典力学没有第二项 (没有 Riemannian metric), 所以 classical observables 可以 simultaneous measure 到 arbitrary precision。

### 3.3 Anandan-Aharonov interpretation

Energy uncertainty 作为 "速度":

$$(\Delta h)^2 = \frac{\hbar}{2} g(X_h, X_h)$$

变量解释:
- $g(X_h, X_h)$: Hamiltonian vector field $X_h$ 的 length squared
- 所以 energy uncertainty ∝ speed of evolution through $\mathcal{P}$

Intuition: 系统在 $\mathcal{P}$ 中 "走" 的速度。Energy uncertainty 大的 state "走" 得快, energy uncertainty 小的 state (比如 energy eigenstate) "走" 得慢 (实际上不动, 只 acquire phase)。

参考: [Anandan-Aharonov paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.65.1697)

---

## 4. Projective Hilbert Space 作为 Quantum Phase Space

### 4.1 为什么要 quotient?

Hilbert space $\mathcal{H}$ 有 gauge redundancy: $\Psi$ 和 $e^{i\theta}\Psi$ 代表同一物理 state。Textbook 通常 normalize 到 unit sphere $S$, 但还有 phase freedom。

真正的 physical state space 是 projective Hilbert space $\mathcal{P} = \mathcal{H}/\mathbb{C}^\times$ (rays)。

### 4.2 Gauge reduction (Bergmann-Dirac constrained systems)

Ashtekar 用了一个 elegant 的 construction: 把 normalization + phase 看作 constraint + gauge freedom。

Constraint:
$$C(\Psi) := \langle\Psi, \Psi\rangle - 1 = \frac{1}{2\hbar} G(\Psi, \Psi) - 1 \tag{2.11}$$

变量解释:
- $C$: constraint function
- $C = 0$ 定义 unit sphere $S$
- Constraint surface $S$ 是 "physically accessible" 部分

$C$ 是 first-class constraint (i.e., $\{C, C\} = 0$), 所以 generate gauge transformations。Gauge generator:

$$\mathcal{T}^a := \hbar X_C^a\big|_S = -J^a{}_b \Psi^b\big|_S \tag{2.13}$$

变量解释:
- $\mathcal{T}$: phase rotation generator ($\Psi \to e^{i\theta}\Psi$)
- $J$: complex structure
- $\Psi$: state vector

所以 **phase freedom 就是 gauge freedom**, projective Hilbert space 就是 reduced phase space:

$$\mathcal{P} = S / \text{gauge}$$

这是一个非常 clean 的 interpretation: 我们在用 Dirac 的 constrained Hamiltonian system 理论来 "derive" projective Hilbert space。

### 4.3 Reduced Kähler structure

$\mathcal{P}$ 继承 Kähler structure from $S$:

**Symplectic form $\omega$ on $\mathcal{P}$**: 通过 standard reduced phase space construction, $\pi^* \omega = i^* \Omega$, 其中 $\pi: S \to \mathcal{P}$, $i: S \to \mathcal{H}$。

**Riemannian metric $g$ on $\mathcal{P}$**: 因为 $i^* G$ 在 gauge direction 不 degenerate, 需要减掉 vertical component:

$$\tilde{g} := \left[G - \frac{1}{2\hbar}(\Psi \otimes \Psi + \mathcal{T} \otimes \mathcal{T})\right]\Bigg|_S \tag{2.16}$$

变量解释:
- $\Psi \otimes \Psi$: radial direction component (会 vanish 因为 restrict 到 $S$, 但形式上保留)
- $\mathcal{T} \otimes \mathcal{T}$: phase direction component
- 减去这两项得到 horizontal metric $\tilde{g}$, 它 descend 到 $\mathcal{P}$ 成为 $g$

$(\mathcal{P}, g, \omega, j)$ 构成 Kähler manifold, 其中 $j$ 是 induced complex structure。

参考: [Projective Hilbert space - Wikipedia](https://en.wikipedia.org/wiki/Projective_Hilbert_space), [Fubini-Study metric - Wikipedia](https://en.wikipedia.org/wiki/Fubini%E2%80%93Study_metric), [Dirac constrained Hamiltonian systems](https://en.wikipedia.org/wiki/Dirac_bracket)

---

## 5. Observables 的几何 Characterization

### 5.1 Observable functions

Definition II.1: $f: \mathcal{P} \to \mathbb{R}$ 是 observable function iff 存在 bounded self-adjoint operator $\hat{F}$ 使得 $\pi^* f = \langle\hat{F}\rangle|_S$。

从 $\hat{F}$ 到 $f$:
- $F(\Psi) = \langle\Psi, \hat{F}\Psi\rangle$ on $\mathcal{H}$
- Restrict 到 $S$: $i^* F$
- 因为 $F$ 是 phase-invariant, $i^* F$ descend 到 $f$ on $\mathcal{P}$

### 5.2 Killing characterization

Key observation: Schrödinger vector field $Y_{\hat{F}}$ 既 preserve $\Omega$ 又 preserve $G$ (因为 unitary transformations preserve Hermitian inner product)。所以 $X_F$ 是 Killing vector field on $(\mathcal{H}, G)$。

Push-forward 到 $\mathcal{P}$, $X_f$ 仍然是 Killing vector field on $(\mathcal{P}, g)$。

**Corollary 1**: Smooth function $f: \mathcal{P} \to \mathbb{R}$ 是 observable iff $X_f$ 是 Killing vector field on $(\mathcal{P}, g)$。

这是一个完全 intrinsic 的 characterization! 不再需要 reference Hilbert space。

Intuition: 
- Classical mechanics: observables 是 arbitrary smooth functions (Hamiltonian vector field 只需 preserve $\omega$)
- Quantum mechanics: observables 是那些 Hamiltonian vector field **同时** preserve $g$ 的 functions

这就是为什么 quantum observable algebra 比 classical observable algebra 小很多。Finite-dim case: smooth functions on $\mathcal{P}$ 是 infinite-dimensional, 但 self-adjoint operators 是 finite-dimensional ($n^2$ for $n$-dim Hilbert space)。

### 5.3 Symmetry data $S_p$ 和 maximality

在任意点 $p \in \mathcal{P}$, Killing vector field 由其值和 first covariant derivative at $p$ 完全决定 (standard Riemannian geometry fact, 见 [Wald's GR textbook](https://press.uchicago.edu/ucp/books/book/chicago/G/bo3684005.html))。

对 observable function $f$, 我们有:
- $f(p) \in \mathbb{R}$ (value)
- $(X_f)_\alpha|_p \in T^*_p\mathcal{P}$ (Hamiltonian vector field)
- $\nabla_\alpha (X_f)_\beta|_p = K_{\alpha\beta}$ (covariant derivative, skew-symmetric because Killing)

加上 integrability condition:
$$\omega_\alpha{}^\gamma K_{\gamma\beta} = \omega_\beta{}^\gamma K_{\gamma\alpha} \text{ (symmetric)}$$

定义 **symmetry data**:
$$\mathcal{S}_p = \{(\lambda, X_\alpha, K_{\alpha\beta}) | \omega_\alpha{}^\gamma K_{\gamma\beta} \text{ symmetric}\}$$

Theorem II.1: 对任意 $(\lambda, X, K) \in \mathcal{S}_p$, 存在 observable function $f$ 使得 $f(p) = \lambda$, $X_f = X$, $\nabla X_f = K$ at $p$。

所以 standard QM 中, observable algebra is isomorphic to $\mathcal{S}_p$ at any point $p$。这叫做 **maximality** (Definition III.1): phase space admits "as many observables as possible"。

参考: [Killing vector field - Wikipedia](https://en.wikipedia.org/wiki/Killing_vector_field)

---

## 6. Measurement 的几何描述

### 6.1 Transition probability via geodesic distance

Theorem II.2: 对任意 $p_0, p \in \mathcal{P}$, 存在 closed geodesic 通过它们, 且

$$\delta_{p_0}(p) = \cos^2\left(\frac{\sigma(p_0, p)}{\sqrt{2\hbar}}\right) \tag{2.29}$$

变量解释:
- $\delta_{p_0}(p) = |\langle\Psi_0, \Psi\rangle|^2$: quantum transition probability
- $\sigma(p_0, p)$: minimal geodesic distance on $(\mathcal{P}, g)$ (Fubini-Study metric)
- $\sqrt{2\hbar}$: normalization constant

这是非常 elegant 的结果! Quantum probability 纯粹由 Riemannian geometry (geodesic distance) 决定。System 更可能 collapse 到 nearby state, less likely 到 distant state。

### 6.2 Eigenstates 作为 critical points

Definition II.3: $f: \mathcal{P} \to \mathbb{R}$ 的 critical points (即 $X_f = 0$) 称为 eigenstates, 对应的 critical values 称为 eigenvalues。

Intuition: 在 eigenstate, Hamiltonian vector field 消失, 意味着 evolution 是 trivial (只 acquire phase, 但 phase 是 gauge, 所以物理上 stationary)。

### 6.3 State reduction 的几何 picture

测量前 system 在 state $p_0$。测量 $f$ 得到 eigenvalue $\lambda$ 后, system collapse 到 eigenmanifold $\mathcal{E}_\lambda$ 上的某个点。

**关键事实**: collapse 到的点 $P_\lambda(p_0) \in \mathcal{E}_\lambda$ 是 $\mathcal{E}_\lambda$ 中 **geodesically closest** to $p_0$ 的点!

$$P_\lambda(p_0) = \arg\min_{q \in \mathcal{E}_\lambda} \sigma(p_0, q)$$

Collapse probability:
$$\Pr(\lambda) = \cos^2\left(\frac{\sigma(p_0, \mathcal{E}_\lambda)}{\sqrt{2\hbar}}\right)$$

这里 $\sigma(p_0, \mathcal{E}_\lambda) = \sigma(p_0, P_\lambda(p_0))$ 是 minimal geodesic distance from $p_0$ 到 eigenmanifold。

### 6.4 Continuous spectrum

对 continuous spectrum, 我们用 spectral projection $P_{f,\Lambda}$ 对应 closed subset $\Lambda \subset \text{sp}(f)$。

Eigenmanifold:
$$\mathcal{E}_{f,\Lambda} = \{q \in \mathcal{P} | \underbrace{\{f, \{f, \cdots \{f, f\}_+\cdots\}_+\}_+}_{n \text{ factors}}(q) \in \Lambda^n, \forall n > 0\} \tag{2.33}$$

变量解释:
- $n$-fold symmetric bracket: 对应 $\hat{F}^n$ 的 expectation value
- $\Lambda^n = \{\lambda^n | \lambda \in \Lambda\}$: image of $\Lambda$ under $n$-th power map
- Condition: 在 $q$ 处, 所有 moments of $f$ 都 consistent with support 在 $\Lambda$ 上

Measurement: 询问 "$f \in \Lambda$?", 系统被 drive 到 $P_{f,\Lambda}(p_0)$ (yes) 或 $P_{f,\Lambda^c}(p_0)$ (no), 其中 $\Lambda^c$ 是 complement。

参考: [Fubini-Study metric](https://en.wikipedia.org/wiki/Fubini%E2%80%93Study_metric), [Geometric quantum mechanics - Scholarpedia](http://www.scholarpedia.org/article/Geometric_quantum_mechanics)

---

## 7. Postulates of QM 的几何 Formulation

完整几何 postulates:

**(P) Physical states**: Points of Kähler manifold $\mathcal{P}$ (projective Hilbert space).

**(K) Kähler evolution**: Evolution 是 preserve Kähler structure 的 flow, generated by densely defined vector field on $\mathcal{P}$.

**(O) Observables**: Real-valued smooth functions $f$ on $\mathcal{P}$ whose Hamiltonian vector fields $X_f$ preserve Kähler structure (即 $X_f$ 是 Killing).

**(I) Probabilistic interpretation**: 对 closed subset $\Lambda \subset \text{sp}(f)$, state $p \in \mathcal{P}$ 测量得 $f \in \Lambda$ 的 probability:

$$\delta_p(\Lambda) = \cos^2\left(\frac{\sigma(p, P_{f,\Lambda}(p))}{\sqrt{2\hbar}}\right) \tag{2.34}$$

**($\mathcal{R}_D$) Reduction, discrete**: 测量 $f$ 得 eigenvalue $\lambda$, state collapse 到 $P_{f,\lambda}(p)$.

**($\mathcal{R}_C$) Reduction, continuous**: 对 closed $\Lambda \subset \text{sp}(f)$, state collapse 到 $P_{f,\Lambda}(p)$ 或 $P_{f,\Lambda^c}(p)$, 由 measurement result 决定。

**注意**: 整个 formulation 完全没有 reference to Hilbert space 或 linear structure! Hilbert space 只是 technical convenience, 就像 Riemannian geometry 中经常 embed manifold 到 $\mathbb{R}^n$ 一样, embedding 不是 essential 的。

Ashtekar 自己 draw 的 analogy: Minkowski 把 Einstein 的 special relativity reformulate 成 geometric language, paving the way 到 general relativity。或许 geometric QM 也 paving the way 到某种 "generalized quantum mechanics"。

---

## 8. Generalizations of Quantum Mechanics

### 8.1 Generalized dynamics (conservative approach)

保守 approach: 保留 $\mathcal{P}$ 作为 state space, 只 generalize dynamics。

Standard QM: Hamiltonian flow 必须 preserve 整个 Kähler structure (既 $\omega$ 又 $g$)。

Generalized dynamics: 只要求 preserve $\omega$ (像 classical mechanics), 不要求 preserve $g$。

允许的 Hamiltonian functions 类:
$$\mathcal{C}_H = \{f: \mathcal{P} \to \mathbb{R} | f \text{ smooth, generates global flow}\}$$

这个 class 比 observable functions 大很多 (finite-dim case: $\mathcal{O}$ finite-dim, $\mathcal{C}_H$ infinite-dim)。

### 8.2 Weinberg functions

从 $\mathcal{P}$ lift 到 $\mathcal{H}^\times = \mathcal{H} - \{0\}$ 的 preferred extension:

$$F_{ext}(\Psi) := \|\Psi\|^2 F(\Psi/\|\Psi\|) \tag{3.1}$$

性质:
- Homogeneous of degree 2 (degree 1 in $\Psi$, degree 1 in $\bar{\Psi}$)
- Phase-invariant
- Generates flow that preserve norm

**Weinberg functions** $\mathcal{O}_W$: $\mathcal{H}^\times$ 上 homogeneous degree 2, phase-invariant functions。

**One-to-one correspondence**: $\mathcal{C}_H$ (smooth on $\mathcal{P}$) ↔ $\mathcal{O}_W$ (Weinberg functions on $\mathcal{H}^\times$)。

Weinberg 在 [他的 1989 paper](https://www.sciencedirect.com/science/article/pii/0003491689902765) 中提出这个 framework, 但他从 Hilbert space perspective 出发, 不太清楚和 projective space 的关系。Geometric formulation clarify 了这个 correspondence。

### 8.3 Non-linear Schrödinger equation

具体例子 1: Non-linear Schrödinger equation:

$$i\hbar \frac{\partial\Psi}{\partial t}(x,t) = (\hat{H}_0 \Psi)(x,t) + \epsilon |\Psi(x,t)|^2 \Psi(x,t) \tag{3.4}$$

变量解释:
- $\hat{H}_0 = \hat{P}^2/(2m) + \hat{V}$: standard Hamiltonian
- $\epsilon$: non-linearity strength
- $|\Psi(x,t)|^2$: local probability density (不是 state vector norm!)

对应的 Hamiltonian function:
$$H_\epsilon(\Psi) = \frac{\epsilon}{2} \int d^n x [\Psi^*(x,t)\Psi(x,t)]^2 \tag{3.5}$$

Note: $H_\epsilon$ **不** 是 Weinberg function (homogeneity degree 4 而不是 2)。但是它在 unit sphere $S$ 上的 restriction 是 well-defined, descend 到 $\mathcal{P}$ 上。所以它 induce 的 flow on $\mathcal{P}$ 仍然可以用 Weinberg function 描述, 只需要用 Eq. (3.1) 重新 extend:

$$H_\epsilon'(\Psi) = \|\Psi\|^2 H_\epsilon|_S(\Psi/\|\Psi\|)$$

**Weinberg's misconception**: Weinberg 说 non-linear Schrödinger equation 的 results "of no use" to his framework。Ashtekar 指出这是 misleading 的: 从 $\mathcal{P}$ perspective, non-linear Schrödinger equation **就是** Weinberg-type generalized dynamics, 只是对应的 Hamiltonian function on $\mathcal{H}^\times$ 不满足 homogeneity, 但可以 re-extend 使其满足。

### 8.4 Logarithmic equation (Bialynicki-Birula & Mycielski)

具体例子 2: Logarithmic equation:

$$i\hbar \frac{\partial\Psi}{\partial t} = \hat{H}_0 \Psi + \alpha(|\Psi|^2)\Psi \tag{3.8}$$

其中 $\alpha(\rho) = -b\ln(a^n \rho)$。

变量解释:
- $\alpha(\rho)$: non-linear term, depends on local density $\rho = |\Psi|^2$
- $a, b$: constants

Motivation: 要求 non-interacting subsystems 不通过 $\alpha$ term 产生 interaction, severely restrict $\alpha$ 的形式, 只能是 logarithmic。

对应 Hamiltonian:
$$H_1(\Psi) = b \int d^n x \Psi^*(x)\Psi(x)[1 - \ln(\Psi^*(x)\Psi(x))] \tag{3.10}$$

Re-extended as Weinberg function:
$$H_1'(\Psi) = H_1(\Psi) + b\|\Psi\|^2 \ln(\|\Psi\|^2) \tag{3.11}$$

所以 logarithmic equation 也是 Weinberg-type generalized dynamics。

参考: [Weinberg's paper](https://doi.org/10.1016/0003-4916(89)90276-5), [Bialynicki-Birula & Mycielski paper](https://doi.org/10.1016/0003-4916(76)90002-1)

---

## 9. Characterization of Standard QM Kinematics

### 9.1 Maximal observable algebra

考虑 arbitrary Kähler manifold $(\mathcal{M}, g, \omega)$ 作为 generalized quantum phase space。

Observables: $\mathcal{O} = \{f: \mathcal{M} \to \mathbb{R} | \mathcal{L}_{X_f} g = 0\}$ (Killing Hamiltonian vector fields)。

Definition III.1: $\mathcal{O}$ 是 maximal iff 对每个 $p \in \mathcal{M}$, 每个 symmetry data $(\lambda, X, K) \in \mathcal{S}_p$ 都 integrable 到 global observable。

Standard QM 的 observable algebra 是 maximal。Torus 上的 Kähler structure 没有 non-trivial observable (只有 constants), 不是 maximal。

### 9.2 Constant holomorphic sectional curvature

Riemann tensor of projective Hilbert space:

$$R_{\alpha\beta\gamma\delta} = \frac{C}{2}[g_{\gamma[\alpha}g_{\beta]\delta} + \omega_{\alpha\beta}\omega_{\delta\gamma} - \omega_{\gamma[\alpha}\omega_{\beta]\delta}] \tag{3.13}$$

变量解释:
- $R_{\alpha\beta\gamma\delta}$: Riemann curvature tensor
- $g_{\alpha\beta}$: Kähler metric
- $\omega_{\alpha\beta}$: Kähler form
- $C = \hbar/2$ (注意: 这是 constant, 不是 CHSC value, CHSC = $2/\hbar$)
- $[\alpha\beta] = \frac{1}{2}(\alpha\beta - \beta\alpha)$: skew-symmetrization

满足这个 form 叫做 **constant holomorphic sectional curvature (CHSC)**。

### 9.3 Lie algebra structure on $\mathcal{S}_p$

定义 $\mathcal{S}_p$ 上的 bracket (mirror Poisson bracket on $\mathcal{O}$):

$$[(f_1, X_1, K_1), (f_2, X_2, K_2)]_p := \left(\omega(X_1, X_2), X_2^\beta K_{1\beta\alpha} - X_1^\beta K_{2\beta\alpha}, K_{2\alpha}{}^\gamma K_{1\gamma\beta} - K_{1\alpha}{}^\gamma K_{2\gamma\beta} + X_{1\mu}X_{2\nu}R_{\alpha\beta}{}^{\mu\nu}\right) \tag{3.16}$$

**Lemma III.1**: $[\cdot, \cdot]_p$ 是 Lie bracket on $\mathcal{S}_p$ iff Riemann tensor at $p$ 是 CHSC。

### 9.4 Symmetric bracket closure

定义 $\mathcal{S}_p$ 上的 symmetric bracket (mirror Jordan product):

$$((f_1, X_1, K_1), (f_2, X_2, K_2))_p := \left(f_1 f_2 + \frac{\hbar}{2}g(X_1, X_2), \ldots, \ldots\right) \tag{3.19}$$

(完整表达式见 Eq. 3.19, 涉及 $K_1, K_2, R_{\alpha\beta\gamma\delta}$ 等)

**Lemma III.2**: $\mathcal{S}_p$ closed under $(\cdot, \cdot)_p$ iff Riemann tensor at $p$ 是 CHSC $= 2/\hbar$。

注意: Lemma III.1 只要求 CHSC (some constant), Lemma III.2 further fix 了 constant 的 value 为 $2/\hbar$。

### 9.5 Reconstruction theorem

**Theorem III.1**: 如果 $\mathcal{O}$ 是 maximal 且 $\mathcal{S}_p$ 在某一点 closed under symmetric bracket, 则 $\mathcal{M}$ 是 CHSC $= 2/\hbar$。

**Finite-dimensional case**: Complete, simply-connected Kähler manifold of CHSC $= 2/\hbar$ is isomorphic to projective Hilbert space $CP^n$ (or its non-compact dual)。所以:

> Standard QM (finite-dim) is characterized by: complete, simply-connected Kähler + maximal $\mathcal{O}$ + closed under $\{\cdot, \cdot\}_+$.

**Infinite-dimensional case**: Open problem! 可能存在其他 infinite-dim Kähler manifolds with CHSC $= 2/\hbar$, 它们就是 non-trivial kinematical generalizations of QM。

这是 paper 的最 profound 的 result 之一: 给出了 systematic 的 search direction for genuine generalizations of QM。要 generalize, 必须 violate 至少一个条件:
- 不是 complete
- 不是 simply-connected
- Observable algebra 不是 maximal
- 不 closed under symmetric bracket

参考: [Ashtekar-Schilling thesis](https://apps.dtic.mil/sti/pdfs/ADA315190.pdf), [Complex projective space](https://en.wikipedia.org/wiki/Complex_projective_space)

---

## 10. Semi-Classical Considerations

### 10.1 Bundle structure: $\mathcal{P}$ over $\Gamma$

设经典 phase space $(\Gamma, \alpha)$, $\dim\Gamma = 2n$。Quantum phase space $\mathcal{P}$ 是 infinite-dimensional。

Map: $\rho: \mathcal{P} \to \Gamma$, $\rho(x) = (q_i(x), p_j(x))$ (expectation values of elementary operators)。

Equivalence relation: $x_1 \sim x_2 \Leftrightarrow f_r(x_1) = f_r(x_2) \forall r$, where $f_r \in \{q_i, p_i\}$。

**$\mathcal{P}$ 是 bundle over $\Gamma$** (而且 bundle 是 trivial 的)。

Vertical space:
$$\mathcal{V}_x = \{v \in T_x\mathcal{P} | v(f_r) = 0 \forall r\} = \{v | \omega(X_{f_r}|_x, v) = 0 \forall r\} \tag{4.2}$$

Horizontal space $\mathcal{V}_x^\perp$: $\omega$-orthogonal complement。

Classical symplectic structure:
$$\alpha(\xi, \zeta) := \omega(\tilde{\xi}, \tilde{\zeta}) \tag{4.3}$$

其中 $\tilde{\xi}, \tilde{\zeta}$ 是 horizontal lifts。这就是 "horizontal part" of quantum symplectic structure。

### 10.2 Generalized coherent states = horizontal sections

Perelomov coherent states:
$$\Psi_{(q', p')} := \exp\left[-\frac{i}{\hbar}\sum_i (q_i' \hat{P}_i - p_i' \hat{Q}_i)\right] \Psi_0$$

变量解释:
- $\Psi_0$: fiducial state
- $(q', p')$: parameters labeling coherent states
- Exponential: Heisenberg-Weyl group element

Key facts:
1. $q_i(x_{(q',p')}) = q_i(x_0) + q_i'$, $p_i(x_{(q',p')}) = p_i(x_0) + p_i'$ (Eq. 4.4)
2. Uncertainties $\Delta q_i, \Delta p_j$ 在 coherent state space 上 constant
3. $\frac{\partial}{\partial q'}\Psi_{(q',p')} = \frac{1}{i\hbar}\hat{P}\Psi_{(q',p')}$, $\frac{\partial}{\partial p'}\Psi_{(q',p')} = -\frac{1}{i\hbar}\hat{Q}\Psi_{(q',p')}$ (Eqs. 4.5, 4.6)

Property 3 表明 coherent state spaces 是 **horizontal** 的!

**Theorem**: Horizontal sections of $\mathcal{P} \to \Gamma$ 恰好是 generalized coherent state spaces。

这是非常 deep 的 connection。Coherent states 不只是 convenient basis, 它们在 geometric formulation 中有 intrinsic meaning: 它们是 "most classical" 的 quantum states, 沿着这些 sections, 量子 symplectic structure 直接 reduce 到 classical symplectic structure。

### 10.3 Harmonic oscillator: preferred horizontal section

Quantum Hamiltonian for 1D harmonic oscillator:

$$h = \frac{1}{2m}p^2 + \frac{m\omega^2}{2}q^2 + \underbrace{\frac{1}{2m}(\Delta p)^2 + \frac{m\omega^2}{2}(\Delta q)^2}_{h_\Delta} \tag{4.9}$$

变量解释:
- $h_0 = \frac{p^2}{2m} + \frac{m\omega^2 q^2}{2}$: classical form
- $h_\Delta$: uncertainty term, **不能** 用 $q, p$ 单独 express (因为 $\Delta q, \Delta p$ 是额外自由度)

分解: $X_h = X_{h_0} + X_{h_\Delta}$
- $X_{h_0}$: horizontal component (drives classical-like evolution)
- $X_{h_\Delta}$: vertical component (only affects quantum-internal state)

**问题**: 是否有 horizontal section 被 evolution preserve?

要求: $X_{h_\Delta} = 0$ on the section, 即 $h_\Delta$ 在该 section 取 extremum。

计算:
$$\frac{1}{\omega\hbar} h_\Delta = \left[\Delta q \sqrt{\frac{m\omega}{2\hbar}} - \Delta p \sqrt{\frac{1}{2m\omega\hbar}}\right]^2 + \frac{1}{\hbar}\Delta p \Delta q \geq \frac{1}{2} \tag{4.10}$$

变量解释:
- 不等式: Heisenberg uncertainty relation $\Delta q \Delta p \geq \hbar/2$
- Equality 当 $\Delta q \Delta p = \hbar/2$ 且 bracket 内项 = 0

Extremum 在:
- $(\Delta q)^2 = \hbar/(2m\omega)$
- $(\Delta p)^2 = m\omega\hbar/2$
- $h_\Delta = \omega\hbar/2$ (zero-point energy)

这个 section 就是 **standard coherent state space** (generated by oscillator ground state)!

**Intuition**: Standard coherent states 是 "most classical" 的 quantum states, 它们是唯一 horizontal section 被 harmonic oscillator evolution preserve 的 section。

### 10.4 WKB approximation as generalized dynamics

Schrödinger equation with $\Psi = \sqrt{\rho} \exp(iS/\hbar)$:

$$\frac{\partial S}{\partial t} + \frac{1}{2m}(\vec{\partial}S)^2 + V(x) = \frac{\hbar^2}{2m}\frac{\Delta\sqrt{\rho}}{\sqrt{\rho}} \tag{4.16}$$

$$m\frac{\partial\rho}{\partial t} + \vec{\partial}\cdot(\rho\vec{\partial}S) = 0 \tag{4.17}$$

变量解释:
- $S$: Hamilton-Jacobi action function
- $\rho = |\Psi|^2$: probability density
- $\vec{\partial}S$: classical momentum field
- RHS of Eq. (4.16): "quantum potential" term

WKB approximation: drop quantum potential term。

Hamiltonian in terms of $(\rho, S)$:

$$H(\rho, S) = \int d^n x \left[\underbrace{\frac{\hbar^2}{8m\rho}(\vec{\partial}\rho)^2}_{H_\hbar} + \underbrace{\frac{1}{2m}\rho(\vec{\partial}S)^2 + \rho V}_{H_{WKB}}\right] \tag{4.18}$$

WKB evolution: $X_{WKB}$ generated by $H_{WKB} = H - H_\hbar$。

**Key insight**: WKB evolution actually preserves unit sphere! 因为:
- $\{H_\hbar, C\} = 0$ ($H_\hbar$ independent of $S$)
- $\{H, C\} = 0$ (standard Schrödinger preserves norm)
- 所以 $\{H_{WKB}, C\} = 0$

因此 **WKB dynamics 是 Weinberg-type generalized dynamics**! 它 induce 一个 Hamiltonian flow on $\mathcal{P}$, 只 preserve $\omega$, 不 preserve $g$ in general。

Validity condition of WKB:
$$\frac{1}{2m\rho^2}\left|(m\vec{J})^2 - \rho K\right| \ll |\hat{H}\Psi|$$

其中:
- $\vec{J} = \frac{1}{2}(\phi\vec{\partial}\pi - \pi\vec{\partial}\phi)$: momentum density
- $K = \frac{\hbar}{2}(\phi\Delta\phi + \pi\Delta\pi)$: squared-momentum density

Intuition: WKB valid 当 "density of squared-momentum" (weighted by $1/\rho$) comparable to "squared density of momentum"。

参考: [Coherent states - Wikipedia](https://en.wikipedia.org/wiki/Coherent_states), [WKB approximation - Wikipedia](https://en.wikipedia.org/wiki/WKB_approximation)

---

## 11. 一些进一步的 Intuition 和联想

### 11.1 Linear structure 的 "emergence"

Textbook QM 强调 Hilbert space 的 linear structure。Geometric formulation 揭示:
- Physical state space $\mathcal{P}$ 是 nonlinear 的
- Linear structure 是 technical convenience, 让我们 embed $\mathcal{P}$ 到 $\mathcal{H}$
- 真正 essential 的 structures (symplectic, Riemannian) 不依赖 linearity

这让人想起 Minkowski reformulation of special relativity: Einstein 用 inertial frames 的 linear structure, Minkowski 用 geometric language, paving the way 到 general relativity。Ashtekar 在 paper 中 explicit draw 这个 analogy: 或许 geometric QM 也 paving the way 到 generalized QM, 比如用于 quantum gravity。

### 11.2 Quantum-classical correspondence 的层次

Geometric formulation 揭示 quantum-classical correspondence 有 multiple levels:

1. **Algebraic level**: Poisson bracket (commutator) correspondence — Dirac's original insight
2. **Symplectic level**: $\mathcal{P}$ 是 symplectic manifold, classical phase space 也是
3. **Bundle level**: $\mathcal{P} \to \Gamma$ 是 bundle, classical 是 base
4. **Coherent state level**: Horizontal sections 是 coherent states, "most classical" quantum states
5. **Dynamical level**: 在 coherent state section 上, quantum dynamics reduce 到 classical dynamics (for harmonic oscillator)

这种 multi-level 结构 suggest: classical 是 quantum 的某种 "limit" 或 "projection", 但具体是哪种 limit 很 nuanced。

### 11.3 与其他 geometric approaches 的关系

- **Geometric quantization** (Kostant, Souriau): 从 symplectic manifold 构造 Hilbert space, 是反方向 work。Geometric QM 可以看作 "geometric de-quantization"。
  
- **Deformation quantization**: 用 star product 在 classical phase space 上 deform Poisson bracket 到 Moyal bracket, 实现 QM 的 algebraic structure。

- **Berry phase**: 当 parameter 缓慢变化, quantum state 在 $\mathcal{P}$ 中沿 closed loop, 获得的 phase 是 holonomy of Fubini-Study connection。

- **Quantum information geometry**: Fubini-Study metric 是 quantum analogue of Fisher information metric。Quantum Fisher information 和 Bures metric 都与 $\mathcal{P}$ 的 geometry 相关。

- **Second quantization**: Ashtekar 提到 second quantization 就是$^2$: 因为 $\mathcal{P}$ 本身是 Kähler, 可以 geometric quantize 它, 得到 second-quantized theory。

参考: [Geometric quantization](https://en.wikipedia.org/wiki/Geometric_quantization), [Berry phase](https://en.wikipedia.org/wiki/Berry_phase), [Quantum Fisher information](https://en.wikipedia.org/wiki/Quantum_Fisher_information), [Bures metric](https://en.wikipedia.org/wiki/Bures_metric)

### 11.4 给 Karpathy 的 possible connections

考虑到 Karpathy 在 deep learning 和 AI 方向的工作, 这里有一些可能 interesting 的 connections:

1. **Information geometry of neural networks**: Neural network parameter space 配备 Fisher information metric, 是 Riemannian manifold。Natural gradient descent 利用这个 geometry。Quantum state space $\mathcal{P}$ 也配备 Fubini-Study metric, 它们之间有 formal analogy。

2. **Manifold hypothesis**: ML 中 data 假设在 high-dimensional space 的 low-dimensional manifold 上。类似地, classical phase space $\Gamma$ 是 low-dim, embedded 在 high-dim quantum phase space $\mathcal{P}$ 作为 horizontal section。两者都是 "high-dim ambient space + low-dim effective structure" 的 pattern。

3. **Neural network quantum states**: Carleo 和 Troyer 用 restricted Boltzmann machine represent quantum states。Geometric formulation 可能为这种 approach 提供 natural language — 不必 commit 到 specific Hilbert space basis。

4. **Hamiltonian Monte Carlo**: 用 Hamiltonian dynamics 在 parameter space 上 sample, 与 quantum dynamics 的 geometric formulation 有共同点 (都是 symplectic flow)。

5. **Diffusion models 和 Schrödinger bridge**: Diffusion models 学习 stochastic process 从 noise 到 data distribution。Schrödinger bridge problem 是 optimal transport 的一种, 涉及 probability flows。Quantum state evolution 在 $\mathcal{P}$ 上也是 deterministic flow, 加 noise 后变成 stochastic。

6. **Tensor networks 和 geometry**: Tensor networks (MPS, PEPS, MERA) 有 natural holographic interpretation, 与 geometry 相关。或许 quantum phase space $\mathcal{P}$ 的 geometric structure 可以用 tensor networks 来 discretize 或 approximate。

### 11.5 Open problems in the paper

Ashtekar 列出的 open problems:

1. **Spin-statistics postulate** 的 geometric formulation 还没有。
2. **Generalized dynamics 的 measurement theory**: 如果 Hamiltonian function 不是 observable function (即 $X_f$ 不是 Killing), 还能不能有 consistent measurement theory?
3. **Infinite-dimensional characterization**: 是否只有 projective Hilbert space 有 CHSC $= 2/\hbar$? 或者存在其他 infinite-dim Kähler manifolds?
4. **Direct quantization procedure**: 能否直接从 classical phase space 构造 $(\mathcal{P}, \omega, g)$, 不经过 Hilbert space?

### 11.6 一些关键公式的进一步 intuition

**Eq. (2.5)** $dF = i_{Y_{\hat{F}}}\Omega$: 这是说 expectation value function $F$ 的 differential 等于 $\Omega$ 和 Schrödinger vector field 的 interior product。Geometric meaning: $F$ 增长最快的方向就是 $Y_{\hat{F}}$ 方向 (up to $\Omega$ 旋转)。

**Eq. (2.13)** $\mathcal{T} = -J\Psi|_S$: Gauge generator 是 "multiplication by $i$" (即 phase rotation)。这把 phase freedom 自然地 interpret 为 gauge freedom, 是 geometric picture 的一个 elegant 之点。

**Eq. (2.29)** $\delta_{p_0}(p) = \cos^2(\sigma/\sqrt{2\hbar})$: Transition probability 完全由 geodesic distance 决定。这是 Fubini-Study metric 的 deep property, 也说明 Riemannian metric "encodes" quantum probability。

**Eq. (3.13)** $R_{\alpha\beta\gamma\delta}$ 的 special form: 这是 Kähler analogue of constant sectional curvature (real Riemannian case)。CHSC $= 2/\hbar$ 这个 specific value 是由 symmetric bracket closure condition (Lemma III.2) fix 的, 体现了 Jordan product structure 的强 constraint。

**Eq. (4.10)** $h_\Delta \geq \omega\hbar/2$: Zero-point energy 就是 uncertainty term 的 minimum。这给了 zero-point energy 一个 geometric interpretation: 它是 $\mathcal{P}$ 的 geometry 强加 的 minimum uncertainty cost。

---

## 12. Summary

这篇 paper 的 main contributions:

1. **Equivalent geometric formulation**: QM 可以完全 geometric 地 formulate, 不 reference Hilbert space。
2. **Conceptual clarity**: Symplectic = classical-like, Riemannian = quantum-specific。这把 quantum mechanics 的 "special features" 都 trace 到 Riemannian metric。
3. **Unified framework for generalizations**: Weinberg, non-linear Schrödinger, logarithmic equation 都在同一 framework 内, corrections of Weinberg's misconception。
4. **Reconstruction theorem**: Standard QM 由 (complete + simply-connected + maximal observable + closed under Jordan product) characterization, 对应 CHSC $= 2/\hbar$。
5. **Semi-classical insights**: $\mathcal{P} \to \Gamma$ bundle structure, coherent states = horizontal sections, WKB = Weinberg-type generalized dynamics。

**Biggest philosophical takeaway**: Hilbert space 的 linear structure 是 convenience, 物理 essential 的是 $\mathcal{P}$ 上的 Kähler geometry。这 suggest 未来的 generalized QM (比如用于 quantum gravity) 可能需要放弃 linear structure, 但保留某种 Kähler-type geometric structure。

对 Karpathy 这样的 researcher, 或许最 interesting 的 directions 是:
- Information geometry 的 quantum analogue
- Quantum-classical correspondence 和 ML 中 "high-dim ambient + low-dim effective" pattern 的 connection
- 用 geometric language 重新看 quantum algorithms 和 quantum ML
- Neural network quantum states 和 projective Hilbert space geometry 的 connection

---

## Web Links for Reference

- [Ashtekar's Penn State page](https://www.phys.psu.edu/~ashtekar/)
- [Ashtekar-Schilling thesis (Schilling's PhD)](https://apps.dtic.mil/sti/pdfs/ADA315190.pdf)
- [Kähler manifold - Wikipedia](https://en.wikipedia.org/wiki/K%C3%A4hler_manifold)
- [Symplectic geometry - Wikipedia](https://en.wikipedia.org/wiki/Symplectic_geometry)
- [Projective Hilbert space - Wikipedia](https://en.wikipedia.org/wiki/Projective_Hilbert_space)
- [Fubini-Study metric - Wikipedia](https://en.wikipedia.org/wiki/Fubini%E2%80%93Study_metric)
- [Killing vector field - Wikipedia](https://en.wikipedia.org/wiki/Killing_vector_field)
- [Geometric quantization - Wikipedia](https://en.wikipedia.org/wiki/Geometric_quantization)
- [Berry phase - Wikipedia](https://en.wikipedia.org/wiki/Berry_phase)
- [Quantum Fisher information - Wikipedia](https://en.wikipedia.org/wiki/Quantum_Fisher_information)
- [Coherent states - Wikipedia](https://en.wikipedia.org/wiki/Coherent_states)
- [WKB approximation - Wikipedia](https://en.wikipedia.org/wiki/WKB_approximation)
- [Weinberg's "Testing quantum mechanics" paper](https://doi.org/10.1016/0003-4916(89)90276-5)
- [Bialynicki-Birula & Mycielski paper](https://doi.org/10.1016/0003-4916(76)90002-1)
- [Anandan-Aharonov paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.65.1697)
- [Kibble's 1979 paper](https://link.springer.com/article/10.1007/BF01197539)
- [Hughston's 1996 paper on stochastic state reduction](https://royalsocietypublishing.org/doi/abs/10.1098/rspa.1996.0048)
- [Cirelli, Manià, Pizzocchero papers (1990)](https://doi.org/10.1063/1.528949)
- [Scholarpedia: Geometric quantum mechanics](http://www.scholarpedia.org/article/Geometric_quantum_mechanics)
- [Geometric QM lecture notes by Brody](https://www.imperial.ac.uk/~dcbrody/GeoQM/GeoQM.pdf)

如果你想 dive deeper into 某个 specific aspect, 比如 infinite-dimensional characterization 的 open problem, 或者 coherent states 和 horizontal sections 的 connection 的更多 details, 或者 generalizations 的具体 examples, 我可以 elaborate more。
