---
source_pdf: M-ABD Scalable, Efficient, and Robust Multi-Affine-Body Dynamics.pdf
paper_sha256: db08a31efaae18ef99170ee8af033afc78ab82f580aa29b60003901adc6df66e
processed_at: '2026-08-05T15:57:46-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# M-ABD 用人话讲

Andrej，我换个讲法，用更直觉的方式从头讲一遍这篇 paper 的 story。

---

## 一、这 paper 到底在解决什么痛？

想象你做 robotics 或者 graphics，要 simulate 一堆刚体连起来的 system——比如一个 robot arm、一串 pulley、一棵树、一件衣服（很多小 rigid piece 拼起来的）。

经典做法是 **Rigid Body Dynamics (RBD)**：每个 body 6 个自由度（3 translation + 3 rotation），rotation 用 quaternion 或者 Euler angle 表示。听起来挺合理对吧？

但 RBD 有个隐藏的坑：**rotation 这个东西天生非线性**。你用 quaternion $q$ 表示 rotation，那 spatial position $x_i = R(q)\bar{x}_i + t$ 这个 mapping 从 $q$ 到 $x$ 是 non-linear 的。这导致一个连锁反应：

1. 你要 enforce joint constraints（比如 hinge joint 只允许绕一个轴转）
2. Constraint 的 gradient $\nabla C$ 依赖当前 configuration
3. 所以 system matrix 是 time-varying 的
4. Implicit integration 下，每个 time step 都得 re-assemble + re-factorize matrix
5. System 一大（比如几十万 links），这就爆了

主流 simulator 的 workaround 是：用 explicit integration + penalty methods（joint 不严格 enforce，用 spring 拉），或者 stabilized velocity constraints（Baumgarte stabilization）。结果就是 **constraint drift**——joint 会慢慢拉开，simulation 不稳定，大 time step 下直接 crash。

参考：
- MuJoCo: https://doi.org/10.1109/IROS.2012.6386109
- Bullet: https://github.com/bulletphysics/bullet3
- PhysX: https://github.com/NVIDIA-Omniverse/PhysX
- Baumgarte stabilization: https://doi.org/10.1016/0045-7825(72)90018-7

---

## 二、Affine Body Dynamics (ABD) 是什么？

2022 年 Lan et al. 提出了 **Affine Body Dynamics**：不要用 6-DOF rigid body，用 12-DOF affine body。

$$x_i = A \bar{x}_i + t \tag{1}$$

这里 $A \in \mathbb{R}^{3\times 3}$ 是一个 general 的 3×3 matrix（9 个自由 entry），$t \in \mathbb{R}^3$ 是 translation。$A$ 不再强制是 rotation matrix，它是任意的 linear transformation。

那 rigidity 怎么保证？用 elastic energy 软约束：

$$\Psi = k_A \|AA^\top - I_3\|_F^2 \tag{3}$$

当 $A$ 是真正的 rotation matrix 时 $AA^\top = I_3$，energy 为零。$A$ 偏离 rotation 越多，energy 越大。$k_A$ 是 stiffness，调大就让 body 越接近 rigid。

**关键好处**：现在 $x$ 和 generalized coordinate $q = [\text{vec}^\top(A), t^\top]^\top \in \mathbb{R}^{12}$ 之间是**线性关系**：

$$x_i = J_i q, \quad J_i = [\bar{x}_i^\top \otimes I_3, I_3] \tag{4}$$

$J_i$ 是 constant matrix（rest shape 给定就固定）。这意味着 constraint Jacobian 可以是 constant，KKT 系统的 structure 更 friendly。

**代价**：12 DOF vs 6 DOF，system matrix 翻倍。原始 ABD 每个 Newton iteration 要 assemble + factorize 一个 12×12 matrix，比 RBD 慢很多。

原始 ABD paper: https://doi.org/10.1145/3528223.3530064

---

## 三、这 paper 的核心 insight：Co-rotation

作者观察到：在 multibody system 里，body 都是 **high stiffness** 的，意味着 actual deformation $A - R$ 很小（$R$ 是 $A$ 的 rotation part）。

这种 small deformation 的 prior 带来一个重要推论：**不同 material model 在 rest shape 附近的 stress-strain 关系几乎一样**。换句话说，你用 St. Venant-Kirchhoff、neo-Hookean、还是 linear elasticity，在 small deformation 下都差不多。

数学上：

$$K(x)\delta x \approx \text{diag}_N(R) \bar{K} \big(\text{diag}_N(R^\top) \delta x\big) \tag{8}$$

变量解释：
- $K(x) \in \mathbb{R}^{3N \times 3N}$：full-space tangent stiffness（通常很贵）
- $\bar{K}$：rest-shape stiffness（**constant，pre-computable**）
- $\text{diag}_N(R) \in \mathbb{R}^{3N \times 3N}$：$N$ 个 $R$ 的 block-diagonal（$N$ 是 mesh node 数）
- $R(q) = A(A^\top A)^{-1/2}$：从 affine coordinate 提取 rotation（polar decomposition）

**直觉**：small deformation 下，stiffness 就是 "rest stiffness rotate 一下"。rotation 部分可以单独提出来处理。

### Co-rotation property

这里有个很漂亮的代数性质。对任意 rotation $R$：

$$\text{diag}_N(R) J = J \text{diag}_4(R) \tag{10}$$

意思是：在 full space rotate 一下（$\text{diag}_N(R)$），等价于在 generalized space rotate 一下（$\text{diag}_4(R)$）。$\text{diag}_4(R)$ 是 4 个 $R$ 的 block-diagonal（因为 $q$ 由 $A$ 的 3 列 + $t$ 组成，共 4 个 3-vector）。

利用这个，reduced stiffness 变成：

$$K_A = J^\top \tilde{K} J = \text{diag}_4(R) \underbrace{J^\top \bar{K} J}_{\bar{K}_A} \text{diag}_4(R^\top) \tag{11}$$

$\bar{K}_A$ 是 **constant**！同理 mass matrix：

$$M_A = \text{diag}_4(R) \bar{M}_A \text{diag}_4(R^\top) \tag{12}$$

### 最终 single-body system

$$\text{diag}_4(R) \underbrace{\left(\frac{1}{h^2}\bar{M}_A + \bar{K}_A\right)}_{\bar{H}_A} \text{diag}_4(R^\top) \delta q = -J^\top g \tag{13}$$

**$\bar{H}_A \in \mathbb{R}^{12 \times 12}$ 是 constant，可以 pre-factorize！**

每个 Newton iteration 只需要：
1. 算 r.h.s. force $f_A$
2. 用 $\text{diag}_4(R^\top)$ rotate 回 local frame
3. 用 pre-factorized $\bar{H}_A$ solve
4. 用 $\text{diag}_4(R)$ rotate 回 world frame

**全是大 matrix 的 re-factorize 都没了。**

---

## 四、Skipping polar decomposition —— 再砍一刀

Eq. (9) 的 polar decomposition $R = A(A^\top A)^{-1/2}$ 在 high stiffness 下成了 computation bottleneck。

作者提出：既然 $A \approx R$，我们其实不需要精确的 $R$。对于 rigid rotation，我们只关心它 **preserve vector length**。所以对任意向量 $a$：

$$Ra \approx \frac{\|a\|}{\|Aa\|} Aa \tag{17}$$

**直觉**：$Aa$ 方向已经接近 $Ra$，只是长度可能差一点。scale 一下让长度对齐就行。这是个 $O(1)$ 的 normalization，比 polar decomposition 便宜多了。

### Benchmark 对比

| 方法 | 10K steps 总时间 |
|---|---|
| Vanilla implicit ABD [Lan 2022] | 161 ms |
| Implicit RBD | 44 ms |
| Explicit RBD | 32 ms |
| Co-rotated ABD + polar decomp | 34 ms |
| **Co-rotated ABD skip polar** | **27 ms** |

**ABD 居然比 explicit RBD 还快 20%**——这完全反直觉，因为 ABD 多 6 个 DOF。但 pre-factorization 的威力太大了，把每步的 assembly + factorize 成本全部摊到 preprocessing 里了。

---

## 五、Multi-body：用 Control Points 定义 joint

为了方便定义 joint constraint，引入 **Control Points (CP)**：选 4 个 rest-shape 点 $\bar{y}_1, \bar{y}_2, \bar{y}_3, \bar{y}_4$ 形成一个 tet（control tetrahedron），deformed 后的位置 $y_1, y_2, y_3, y_4$ 就是 CP coordinate。

$$y = T q, \quad T = [Y^\top \otimes I_3, \mathbf{1}_{4\times 1} \otimes I_3] \tag{18}$$

$Y = [\bar{y}_1, \bar{y}_2, \bar{y}_3, \bar{y}_4]$ 是 rest-shape CP matrix，$T$ 是 constant transformation。

**直觉**：CP 就是 affine body 的 "handle"。Joint constraint 在 CP 坐标下定义非常直观：

### Ball joint (3 DOF, linear)
让两个 body 的某个 CP 重合：
$$S_B^\alpha y^\alpha - S_B^\beta y^\beta = 0 \tag{19}$$
$S_B \in \mathbb{R}^{3 \times 12}$ 选出某个 CP 的 3 个坐标。**Linear constraint**，gradient constant。

### Hinge joint (5 DOF compact, nonlinear)
6-DOF 版本是 linear（两个 edge 对齐），但作者选了 5-DOF compact 版本：

先算一个 rotation $R_H$ 把 hinge axis 对齐到 local $y$ 轴：
$$R_H = I_3 + [v]_\times + \frac{[v]_\times^2}{1 + a_y} \tag{20}$$
$v = a \times e_y$，$a$ 是 hinge axis，$e_y = [0,1,0]^\top$。这是 Rodrigues formula 变体。

转到 local frame 后，5 个约束：1 个 ball（3 DOF）+ 2 个让另一个 CP 只能沿 axis 移动。

### Universal joint (4 DOF compact, nonlinear)
类似地，4 个约束。

### Prismatic joint (5 DOF, nonlinear)
只允许沿一个轴平移。

### Linear vs Nonlinear 的 trade-off

作者选 **compact + nonlinear** 版本。为什么？因为后面要 solve dual space，rank 越小 dual matrix 越小。Nonlinear 的 gradient 计算麻烦点，但省的求解成本更大。

---

## 六、Dual-space KKT —— Scalability 的核心

### 全局 KKT 系统

$M$ 个 bodies + $K$ 个 joints，每个 Newton step 解：

$$\begin{bmatrix} \tilde{H} & \nabla^\top \tilde{C} \\ \nabla \tilde{C} & 0 \end{bmatrix} \begin{bmatrix} \delta\tilde{q} \\ \delta\tilde{\lambda} \end{bmatrix} = \begin{bmatrix} \tilde{f}_A \\ 0 \end{bmatrix} \tag{26}$$

变量：
- $\tilde{H} = \text{diag}_M(H_A^j)$：block-diagonal，每个 $12 \times 12$ block 是 single body 的 Hessian（**pre-factorized**）
- $\nabla \tilde{C}$：constraint gradient，shape 是 $(\sum C_k) \times 12M$
- $\delta\tilde{q} \in \mathbb{R}^{12M}$：global primal unknown
- $\delta\tilde{\lambda} \in \mathbb{R}^{\sum C_k}$：Lagrange multipliers

### Schur complement 消元

从第一行解出 $\delta\tilde{q}$：

$$\delta\tilde{q} = \tilde{H}^{-1}(\tilde{f}_A - \nabla^\top\tilde{C}\delta\tilde{\lambda}) \tag{27}$$

代入第二行：

$$\underbrace{(\nabla\tilde{C}\tilde{H}^{-1}\nabla^\top\tilde{C})}_{\text{dual matrix}} \delta\tilde{\lambda} = \nabla\tilde{C}\tilde{H}^{-1}\tilde{f}_A \tag{28}$$

**关键**：
1. Dual matrix size 是 $\sum C_k \times \sum C_k$，远小于 $12M \times 12M$
2. $\tilde{H}^{-1}$ 通过 pre-factorized $\bar{H}_A$ 高效计算（每个 body 一个 12×12 solve）
3. Dual matrix 是 **block-sparse**：off-diagonal block 非零当且仅当两个 joints 共享一个 body

### 为什么 dual 比 primal 好？

直觉：joint constraint 的 rank $C_k$ 通常很小（ball=3, hinge=5, universal=4, prismatic=5），所以 $\sum C_k \ll 12M$。比如 1M links 的 pulley system，$12M = 12M$，但 $\sum C_k \approx 5M$（每个 hinge 5 DOF），dual matrix 还是小 2.4 倍。更重要的是 block-sparse structure 让你能用 specialized solver。

---

## 七、四种 topology 的 specialized solvers

### Case I: Joint chain (block-tridiagonal)

Chain 结构（$M - K = 1$，每个 joint 连两个相邻 body），dual matrix 是 **block-tridiagonal**：

$$\begin{bmatrix} D^1 & B^1 & & \\ B^{1^\top} & D^2 & \ddots & \\ & \ddots & \ddots & B^{K-1} \\ & & B^{K-1}^\top & D^K \end{bmatrix} \begin{bmatrix} \delta\lambda^1 \\ \vdots \\ \delta\lambda^K \end{bmatrix} = \begin{bmatrix} b^1 \\ \vdots \\ b^K \end{bmatrix} \tag{39}$$

用 **block Thomas algorithm** [Press 2007] 可以 $O(K)$ 解决。这是为什么 1M-link pulley system 能跑 904 ms/step 的关键。

Numerical Recipes: https://www.amazon.com/Numerical-Recipes-3rd-Scientific-Computing/dp/0521880688

### Case II: Joint tree (ABD-ABA)

Featherstone 的 **Articulated Body Algorithm (ABA)** 推广到 ABD。

**Spatial twist mapping**：

$$G(A^j) = \begin{bmatrix} \frac{1}{2}[q_1^j]_\times & \frac{1}{2}[q_2^j]_\times & \frac{1}{2}[q_3^j]_\times & 0 \\ 0 & 0 & 0 & I_3 \end{bmatrix} \in \mathbb{R}^{6 \times 12} \tag{42}$$

$$V^j = G(A^j)\dot{q}^j = [\omega^{j^\top}, v^{j^\top}]^\top$$

**直觉**：angular velocity $\omega$ 由 $A$ 的 3 列 $q_1, q_2, q_3$ 与对应 time derivatives 的 cross product 之和的一半给出。当 $A \approx R$ 时，$\dot{R} = [\omega]_\times R$，所以 $\omega = \frac{1}{2}(q_1 \times \dot{q}_1 + q_2 \times \dot{q}_2 + q_3 \times \dot{q}_3)$；linear velocity $v = \dot{t}$。

**Gyroscopic term 自动消失**：传统 RBD 有 $V \times^* (IV)$ 这种 gyroscopic term，在 ABD 里被 linear kinematics 自动吸收。数学上：

$$G^\top I^j \dot{G}\dot{q} + G^\top(V \times^* IV) = 0 \tag{49}$$

这是因为 ABD 的 mass matrix $M_A$ 是 constant，kinetic energy $T = \frac{1}{2}\dot{q}^\top M_A \dot{q}$ 不依赖 $q$，所以 Euler-Lagrange 里的 $\dot{M}_A\dot{q} - \partial T/\partial q = 0$。

**Upward condensation (leaf → root)**：

$$U^j = \hat{H}_A^j S_{abd}^j, \quad D^j = S_{abd}^{j^\top} U^j \tag{52}$$

$$\Delta H_A^j = \hat{H}_A^j - U^j(D^j)^{-1}U^{j^\top} \tag{53}$$

$$\Delta f_A^j = \hat{f}_A^j - U^j\alpha^j$$

通过 relative rotation $R_{\text{rel}} = A^{p^\top}A^j$ rotate 到 parent frame 并 accumulate。

**Downward pass (root → leaf)**：给定 parent increment $\delta q^p$，对每个 child solve local KKT：

$$\begin{bmatrix} \hat{H}_A^j & \nabla C_j^{j^\top} \\ \nabla C_j^j & 0 \end{bmatrix} \begin{bmatrix} \delta q^j \\ \delta\lambda^j \end{bmatrix} = \begin{bmatrix} \hat{f}_A^j \\ r^j \end{bmatrix} \tag{57}$$

Global nonlinear solve 被分解为一系列 lightweight joint-size local solves。

Featherstone's book: https://link.springer.com/book/10.1007/978-3-540-73931-5

### Case III: Joint loop

Loop = chain 头尾相连。把某个 body 暂时"移除"，partition KKT：

$$\begin{bmatrix} \mathcal{A} & C^\top \\ C & \mathcal{D} \end{bmatrix} \begin{bmatrix} w_\mathcal{A} \\ w_\mathcal{D} \end{bmatrix} = \begin{bmatrix} b_\mathcal{A} \\ b_\mathcal{D} \end{bmatrix} \tag{58}$$

$\mathcal{A}$ 是剩余 chain（block-tridiagonal，Thomas 解决），用 **Schur complement** $S = \mathcal{D} - C\mathcal{A}^{-1}C^\top$ solve low-rank system。

### Case IV: Joint graph

最 general 情况用 **multi-directional block Gauss-Seidel**：把系统看作多个 joint chains 的组合，沿预先定义的 chains 在 dual space 中逐 joint relax。

---

## 八、Experimental results 亮点

### Single-body validation

| Benchmark | 验证内容 | 关键观察 |
|---|---|---|
| Spinning box | Linear/angular momentum conservation | ABD 与 implicit RBD 匹配，linear momentum 完全保持 |
| T-handle | Intermediate-axis theorem (Dzhanibekov effect) | ABD 准确重现周期性翻转，比 implicit RBD 快 30%+ |
| Heavy top | Precession + nutation | ABD 对 $h$ 更不敏感，coarse step 下 distortion 更小 |
| Physical pendulum | Analytic elliptic-integral reference | ABD 比 implicit RBD 更贴近解析解 |

### Multi-body scalability

| Scene | # Link | # Cons. | h | Sim./step |
|---|---|---|---|---|
| Joint net 100×100 | 30K | 120K | 10 ms | 84 ms |
| Huge pulley | **1M** | **3M** | 10 ms | **904 ms** |
| Willow tree | 21K | 63K | 10 ms | 18 ms |
| Net cloak | 12K | 48K | 10 ms | 33 ms |

**1 million links 在 single CPU thread 上 904 ms/step**。MuJoCo、Bullet、PhysX 在这些例子上全部 fail（即使 $h = 10^{-4}$ s）。

VQ [Maloisel 2025] 也 exact enforce constraints，但在 10×10 net 上 27 ms/step vs ABD <1 ms/step，scale 不上去。

VQ paper: https://doi.org/10.1145/3730872

---

## 九、用人话总结核心 story

1. **RBD 的痛**：rotation 非线性 → constraint Jacobian time-varying → 每个 time step 都要 re-factorize matrix → 大 system 跑不动

2. **ABD 的 idea**：用 12-DOF affine 代替 6-DOF rigid，让 kinematic mapping 变线性。但原来 ABD 还是要每步 re-assemble stiffness，所以反而更慢。

3. **Co-rotation 的魔法**：high stiffness 下 small deformation，stiffness 就是 "rest stiffness rotate 一下"。把 rotation 提出来后，reduced stiffness $\bar{K}_A$ 是 constant，可以 pre-factorize。**ABD 从 "比 RBD 慢" 变成 "比 explicit RBD 还快"**。

4. **Dual-space KKT**：把 primal 的 $12M$ 维问题消元成 dual 的 $\sum C_k$ 维问题。Dual matrix block-sparse，size 小很多。

5. **Topology-aware solvers**：chain 用 block Thomas ($O(K)$)，tree 用 ABD-ABA (Featherstone 推广)，loop 用 Schur complement，graph 用 multi-directional GS。

6. **结果**：1M links, single CPU thread, 904 ms/step, 1 iteration。所有 competitors 在这个 scale 下都 crash。

---

## 十、我的整体直觉

这篇 paper 的 elegance 在于它把几个"反直觉"的 idea 串起来：

**反直觉 1**：用更多 DOF（12 vs 6）反而更快。因为 linear kinematic mapping 允许 pre-factorization，而 RBD 的 nonlinear rotation 阻止了 pre-factorization。

**反直觉 2**：nonlinear joint（compact 版本）反而比 linear joint 好。因为 dual space solve 时 rank 小更重要，nonlinear gradient 的额外计算成本被 pre-factorized Hessian absorb 了。

**反直觉 3**：ABD 居然不需要 gyroscopic term。传统 RBD 的 $V \times^* (IV)$ 在 ABD 的 linear kinematics 下自动 cancel out。这是 ABD 坐标的 "free lunch"。

**反直觉 4**：implicit integration 可以比 explicit 还快。因为 pre-factorization 把每步的 matrix 操作成本摊到 preprocessing 里了，implicit 的 stability 还让你能用大 time step。

**潜在影响**：
- **Embodied AI training**：parallel 环境每个 instance 资源有限，1-iteration ABD 是 ideal
- **Robotics real-time control**：fast + robust constraint enforcement
- **Structural biology**：protein backbone 模拟（Fig. 22 的 SARS-CoV-2 例子）
- **Graphics large-scale scenes**：forests, pulleys, garments

**值得思考的 open questions**：
1. Linear elasticity 在 soft material（soft robotics）下误差如何？
2. Contact handling（IPC）结合时如何保持 pre-factorization？
3. Differentiable simulation：ABD 的 linear kinematic mapping 对 backward pass 友好，可能比 SO(3) parameterization 更适合 differentiable physics
4. GPU parallelization：dual solver 在大规模 joint network 下应该能 GPU 加速很多

参考资源：
- ABD original paper: https://doi.org/10.1145/3528223.3530064
- IPC: https://doi.org/10.1145/3386569.3392425
- Featherstone book: https://link.springer.com/book/10.1007/978-3-540-73931-5
- Chen et al. 2022 unified Newton barrier: https://doi.org/10.1145/3528223.3530076
- RedMax: https://doi.org/10.1145/3306346.3322952

希望这个版本更直觉！如果你想 drill down 某个具体点（比如 co-rotation 的几何意义、ABD-ABA 与 Featherstone ABA 的具体差异、或者 constraint gradient 的 skew-symmetrize trick），我可以再展开。

---

# M-ABD 深度技术解读

Andrej，这篇 paper 我读下来感觉作者团队（Utah/MIT/UCSD/UCLA/UBC/ByteDance）做了一件相当 elegant 的事情：把 **Affine Body Dynamics** 从一个"看起来 DOF 更多所以更慢"的方法，通过 co-rotation + dual-space KKT 的组合拳，变成在 large-scale articulated system 上 **单线程 CPU 都能跑百万 links** 的 framework。核心 insight 是分离 geometric nonlinearity 与 material nonlinearity，让 system matrix 在 implicit integration 下也能 pre-factorize。

---

## 1. 核心动机：RBD 的根本痛点

经典 **Rigid Body Dynamics (RBD)** [Baraff 1997] 用 6 DOF 表示一个 body（3 translation + 3 rotation），rotation 通常用 quaternion 或 Euler angles 参数化。问题在于：

$$x_i = R \bar{x}_i + t$$

这里 $x_i$ 是 spatial position，$\bar{x}_i$ 是 rest-shape position，$R \in SO(3)$ 是 rotation matrix，$t$ 是 translation。**kinematic map 从 RBD 坐标到 spatial 坐标是非线性的**（因为 $R$ 的参数化）。

这在 single body 上问题不大，但在 multibody system 中要 exact enforce joint constraints via **KKT (Karush-Kuhn-Tucker)** [Boyd & Vandenberghe 2004] 时，constraint Jacobian $\nabla C$ 会随时间变化，导致 system matrix time-varying，必须每步 re-assemble + re-factorize。这就是为什么 fully implicit RBD multibody simulation 很少见，主流 simulator（MuJoCo [Todorov et al. 2012]、Bullet [Coumans 2015]、PhysX）多采用 explicit integration + penalty methods 或 stabilized velocity constraints，会有 constraint drift。

参考资料：
- Baraff 1997 SIGGRAPH course notes: https://www.cs.cmu.edu/~baraff/sigcourse/
- MuJoCo paper: https://doi.org/10.1109/IROS.2012.6386109
- Boyd & Vandenberghe Convex Optimization: https://web.stanford.edu/~boyd/cvxbook/

---

## 2. ABD 的 kinematic linearization

**Affine Body Dynamics (ABD)** [Lan et al. 2022] 把 rigid body 从 6 DOF 扩展到 12 DOF：

$$x_i = A(t) \bar{x}_i + t(t) \tag{1}$$

其中 $A \in \mathbb{R}^{3\times 3}$ 是 general affine matrix（不再强制 $A \in SO(3)$），$t \in \mathbb{R}^3$ 是 translation。rigidity 通过 elastic potential energy 软约束：

$$E_A = \int_\Omega \Psi \, d\Omega \tag{2}$$

原始 ABD 用一个简单的 polynomial energy：

$$\Psi = k_A \|AA^\top - I_3\|_F^2 \tag{3}$$

其中 $\|\cdot\|_F^2$ 是 Frobenius norm，$I_3$ 是 $3\times 3$ identity，$k_A$ 是 affine stiffness。当 $A \to R \in SO(3)$ 时 $AA^\top = I_3$，energy 为零。

**关键 linearization**：nodal positions $x$ 与 generalized coordinate $q$ 之间是 **线性关系**：

$$x_i = J_i q = [\bar{x}_i^\top \otimes I_3, I_3] q \tag{4}$$

其中 $q = [\text{vec}^\top(A), t^\top]^\top \in \mathbb{R}^{12}$（把 $A$ 的 9 个 entry 按 column-major stack，再接 $t$ 的 3 个 entry），$J_i$ 是 constant coordinate Jacobian（rest-shape 给定后就固定）。

ABD 原始 paper: https://doi.org/10.1145/3528223.3530064

### 直觉：
- RBD：$R$ 是非线性参数化（quaternion unit norm constraint、Euler singularities），$x$ 与 $q$ 非线性
- ABD：$A$ 是 9 个自由 entry，$x$ 与 $q$ 线性，constraint Jacobian 可以是 constant
- 代价：12 DOF vs 6 DOF，系统矩阵大一倍，且需要 elasticity 来约束 rigidity

---

## 3. Co-rotated formulation —— 本文的"魔法"

这是整篇 paper 最聪明的部分。问题在于：虽然 $J$ 是 constant，但 elasticity energy $\Psi$ 仍然 nonlinear w.r.t. $A$（rotation-invariant 的本质就是 nonlinear）。每个 Newton iteration 还是要 re-assemble tangent stiffness $K(x) = \partial^2 \Psi / \partial x^2$ 并 project 到 affine subspace $K_A = J^\top K(x) J$。

**关键观察**：multibody system 里都是 high-stiffness objects，actual deformation $A - R$ 很小。这意味着 **strain-stress relations 在 material space 几乎和 rest shape 一样**，于是：

$$K(x)\delta x \approx \text{diag}_N(R) \bar{K} \big(\text{diag}_N(R^\top) \delta x\big) = \underbrace{\big(\text{diag}_N(R) \bar{K} \text{diag}_N(R^\top)\big)}_{\tilde{K}} \delta x \tag{8}$$

变量含义：
- $\text{diag}_N(R) \in \mathbb{R}^{3N \times 3N}$：block-diagonal matrix，$N$ 个 $R$ 副本沿对角
- $R(q) \in \mathbb{R}^{3\times 3}$：当前 body 的 rotation matrix，通过 **polar decomposition** 从 affine coordinate 提取：

$$R(q) = A(A^\top A)^{-1/2}, \quad A(q) = \text{vec}^{-1}([I_9, 0_{9\times 3}]q) \tag{9}$$

- $\bar{K}$：rest-shape tangent stiffness（constant，pre-computable）
- $\tilde{K}$：rotated 版本，近似真实 $K(x)$，当 $A \to R$ 时误差趋于零

### Co-rotation property：
对任意 $R$，left-multiply Eq. (1) 两边：

$$Rx_i = (RA)\bar{x}_i + Rt$$

即 $\text{diag}_N(R) x = \text{diag}_N(R) J q = J \text{diag}_4(R) q$，对任意 $q$ 成立，所以：

$$\text{diag}_N(R) J = J \text{diag}_4(R) \tag{10}$$

这里 $\text{diag}_4(R) \in \mathbb{R}^{12 \times 12}$：4 个 $R$ 副本 block-diagonal（因为 $q$ 由 4 个 3-vector 组成：$A$ 的 3 列 + $t$）。

### 推导 reduced stiffness：

$$K_A = J^\top \tilde{K} J = J^\top \text{diag}_N(R) \bar{K} \text{diag}_N(R^\top) J = \text{diag}_4(R) \underbrace{J^\top \bar{K} J}_{\bar{K}_A} \text{diag}_4(R^\top) \tag{11}$$

同理 mass matrix 也 co-rotate：

$$M_A = \text{diag}_4(R) \bar{M}_A \text{diag}_4(R^\top) \tag{12}$$

### 最终 single-body ABD 系统：

$$\text{diag}_4(R) \underbrace{\left(\frac{1}{h^2} \bar{M}_A + \bar{K}_A\right)}_{\bar{H}_A} \text{diag}_4(R^\top) \delta q = -J^\top g \tag{13}$$

**核心：$\bar{H}_A \in \mathbb{R}^{12\times 12}$ 是 constant matrix，可以 pre-factorize！** 即使使用 implicit integration，每个 Newton iteration 只需要：
1. 计算 r.h.s. $f_A$（aggregate spatial force）
2. Apply co-rotation $\text{diag}_4(R^\top)$ 到 $f_A$
3. 用 pre-factorized $\bar{H}_A$ solve 一个 12×12 linear system
4. Apply co-rotation $\text{diag}_4(R)$ 到结果

### Linear elasticity 选择：
本文用 linear elasticity 而不是原始的 polynomial energy：

$$\frac{\partial \Psi}{\partial A} = \mu(A + A^\top - 2I_3) + \lambda \text{tr}(A - I_3) I_3 \tag{15}$$

变量：$\mu, \lambda$ 是 **Lamé parameters**（与 Young's modulus $E$ 和 Poisson ratio $\nu$ 关系：$\mu = E/(2(1+\nu))$，$\lambda = E\nu/((1-2\nu)(1+\nu))$）。

**为什么 linear elasticity 在这里够用**：因为 Newton iteration 本身已经 linearize 了 equation of motion，material simplification 的误差被 Newton linearization 的误差 hide 掉，convergence rate 不受影响。

---

## 4. Skipping polar decomposition —— 进一步加速

Eq. (9) 的 polar decomposition 在 high-stiffness 场景下成了 major hurdle。因为 $A \approx R$ 已经接近 rotation matrix，可以用一个 cheap normalization：

$$Ra \approx \frac{\|a\|}{\|Aa\|} A a \tag{17}$$

直觉：对任意向量 $a$，我们只需要 $Ra$ 保留 $a$ 的长度（rigid rotation 的核心性质）。$A a$ 方向已经接近 $R a$，只需 scale 一下让它长度对齐。

这样整个 ABD integration 只用 **BLAS level 1/2 operations**（向量/矩阵-向量运算），没有 level 3（matrix-matrix）。

### Algorithm 1 pseudo code 解析：

```
Input: M_A, K̄_A, q, J̄, f, h
1. H_A ← (1/h²) M_A + K̄_A            # constant, pre-factorized
2. f_A ← J̄^T f                        # aggregate spatial force to affine
3. A ← [q_1, q_2, q_3]                # q_1,2,3 是 A 的 3 列
4-8. for k=1..4:                      # 替代 polar decomposition
       l_k² ← f_A,k · f_A,k           # 记录原长度
       f_A,k ← A^T f_A,k              # rotate back to local frame
       f_A,k ← √(l_k² / (f_A,k · f_A,k)) · f_A,k   # renormalize
9. Solve δp via H_A δp = f_A          # 12×12 solve, pre-factorized
10-14. 同样对 δp 做 inverse co-rotation
```

### Fig. 2 benchmark 数据：
| 方法 | 总时间（10K steps） |
|---|---|
| Vanilla implicit ABD [Lan 2022] | 161 ms |
| Implicit RBD | 44 ms |
| Explicit RBD | 32 ms |
| Co-rotated ABD + polar decomp | 34 ms |
| Co-rotated ABD skip polar | **27 ms** |

**Co-rotated ABD 比 explicit RBD 还快 20%**，这是反直觉的结果——ABD 本来因为多 6 DOF 应该更慢，但通过 pre-factorization 反而超越了 RBD。

---

## 5. Control Points 重新参数化

为了方便定义 joint constraints，引入 **Control Points (CP)** 和 **Control Tetrahedron (CT)**。

选 4 个 rest-shape positions $\bar{y}_1, \bar{y}_2, \bar{y}_3, \bar{y}_4$ 形成一个 non-degenerate tet，deformed 后的位置：

$$y = [Y^\top \otimes I_3, \mathbf{1}_{4\times 1} \otimes I_3] q = T q \tag{18}$$

变量：
- $Y = [\bar{y}_1, \bar{y}_2, \bar{y}_3, \bar{y}_4] \in \mathbb{R}^{3\times 4}$：rest-shape CP matrix
- $T \in \mathbb{R}^{12 \times 12}$：constant transformation
- $y \in \mathbb{R}^{12}$：CP coordinate（4 个 3D points stack 起来）

由于 CT 是 non-degenerate 的，$T$ 可逆，$q = T^{-1} y$，Jacobian $\partial q / \partial y = T^{-1}$ 是 constant。

**直觉**：CP 是 affine body 的"handle"，joint constraints 在 CP 坐标下定义非常自然——ball joint 就是让两个 body 的某个 CP 重合，hinge joint 就是让某条 edge 对齐等。

---

## 6. Joint constraints 详解

### 6.1 Ball joint (3 DOF)
让两个 body $\alpha, \beta$ 的某个 CP 重合：

$$S_B^\alpha y^\alpha - S_B^\beta y^\beta = 0 \tag{19}$$

$S_B \in \mathbb{R}^{3 \times 12}$：selection matrix 选出某个 CP 的 3 个 coordinates。两个 ball-joint-connected bodies 总 DOF = 12 + 12 - 3 = 21。**Linear constraint**（在 CP 坐标下 gradient 是 constant）。

### 6.2 Hinge joint (5 DOF compact 版本)
先找一个 rotation $R_H$ 把 hinge axis $a = [a_x, a_y, a_z]^\top$ 对齐到 local $y$ 轴：

$$R_H = I_3 + [v]_\times + \frac{[v]_\times^2}{1 + a_y} \tag{20}$$

其中 $v = a \times e_y$，$[v]_\times$ 是 skew-symmetric matrix，$e_y = [0,1,0]^\top$。这是 **Rodrigues rotation formula** 的变体。

转到 local frame $\tilde{y} = \text{diag}_4(R_H) y$ 后，constraint 是 5 维：
- 1 个 ball constraint（固定两个 body 在 axis 上的某个 CP 重合）→ 3 DOF
- 2 个额外约束：限制另一个 CP 只能沿 axis 移动（local $x$ 和 $z$ 坐标相等）→ 2 DOF

$$S_H^\alpha \text{diag}_4(R_H) T^\alpha q^\alpha - S_H^\beta \text{diag}_R T^\beta q^\beta = 0 \tag{21}$$

$S_H \in \mathbb{R}^{5 \times 12}$。**Nonlinear constraint**（因为 $R_H$ 依赖 axis，但 axis 是 fixed 的——其实这里 $R_H$ 本身是 constant，但 $R_H$ 作用在 $y$ 上后，相对 body 的 rotation gradient w.r.t. $q$ 是 nonlinear 的，详见 Section 5.1）。

### 6.3 Universal joint (4 DOF compact 版本)
两个 orthogonal axes $a_1 \perp a_2$，构建：

$$R_U = [a_1 \times a_2, a_1, a_2]^\top \tag{22}$$

让 $R_U a_1$ 对齐 local $x$ 轴，$R_U a_2$ 对齐 local $z$ 轴。Constraint 是 4 维：1 个 ball + 1 个 equality 让第二个 CP 的 local $x$ coordinate 与第一个 CP 相同（这强制第二个 CP 只能在 local $y$-$z$ plane 移动，与 $a_1$ 正交）。

### 6.4 Prismatic joint (5 DOF)
类似 hinge joint 的 $R_P$，但约束 CP 的不同 local coordinates。5 个约束方程保证只沿一个 axis 平移。

### Linear vs Nonlinear 总结：
- Ball joint: **linear**（在 CP 坐标下）
- 6-DOF hinge (用 edge-edge): **linear**，但需要 12 DOF virtual body
- 5-DOF hinge (compact): **nonlinear**，但 rank 最小
- Universal, prismatic 类似

本文选 **compact + nonlinear** 版本，因为后续要在 dual space 解决，rank 越小 dual matrix 越小。

### 关于 inequality constraints（joint limits）：
用 **strain-limiting** [Provot 1995] + explicit penalty，clamp 超出范围的 DOF 到边界，然后在 r.h.s. 加 penalty force $k(\theta - \hat{\theta})$。避免 implicit inequality handling 是为了 fully exploit pre-factorized per-body Hessian。

---

## 7. Dual-space KKT —— 真正的 scalability 关键

### 7.1 全局 KKT 系统

对于 $M$ bodies + $K$ joints 的系统，每个 Newton step 要解：

$$\begin{bmatrix} \tilde{H} & \nabla^\top \tilde{C} \\ \nabla \tilde{C} & 0 \end{bmatrix} \begin{bmatrix} \delta \tilde{q} \\ \delta \tilde{\lambda} \end{bmatrix} = \begin{bmatrix} \tilde{f}_A \\ 0 \end{bmatrix} \tag{26}$$

变量：
- $\tilde{H} = \text{diag}_M(H_A^j)$：block-diagonal，每个 $12 \times 12$ block 是单个 body 的 reduced Hessian
- $\nabla \tilde{C}$：global constraint gradient，shape 是 $\sum_{k=1}^K C_k \times 12M$，$C_k$ 是第 $k$ 个 joint 的 rank
- $\delta \tilde{q} \in \mathbb{R}^{12M}$：global primal unknown
- $\delta \tilde{\lambda} \in \mathbb{R}^{\sum C_k}$：Lagrange multipliers (dual DOFs)

### 7.2 Schur complement 消元

从第一行解出：

$$\delta \tilde{q} = \tilde{H}^{-1}(\tilde{f}_A - \nabla^\top \tilde{C} \delta \tilde{\lambda}) \tag{27}$$

代入第二行：

$$\underbrace{(\nabla \tilde{C} \tilde{H}^{-1} \nabla^\top \tilde{C})}_{\text{dual matrix}} \delta \tilde{\lambda} = \nabla \tilde{C} \tilde{H}^{-1} \tilde{f}_A \tag{28}$$

**关键**：
1. Dual matrix 的 size 是 $\sum C_k \times \sum C_k$，远小于 $12M \times 12M$
2. $\tilde{H}^{-1}$ 通过 pre-factorized $\bar{H}_A$ 高效计算
3. Dual matrix 是 **block-sparse**（每个 joint 只影响其 incident bodies，off-diagonal block 非零当且仅当两个 joints 共享一个 body）

### 7.3 Constraint gradient 计算

对于 nonlinear joints (hinge, universal, prismatic)，统一形式：

$$C^k(R_{\text{Joint}}(q^\alpha, q^\beta), q^\alpha, q^\beta) = 0 \tag{29}$$

Gradient 推导（以 hinge 为例）：

$$\nabla C^k = \frac{\partial C^k}{\partial R_{\text{Joint}}} : \frac{\partial R_{\text{Joint}}}{\partial q^\alpha} + \frac{\partial C^k}{\partial R_{\text{Joint}}} : \frac{\partial R_{\text{Joint}}}{\partial q^\beta} + \frac{\partial C^k}{\partial q^\alpha} + \frac{\partial C^k}{\partial q^\beta} \tag{30}$$

其中 ":" 是 double contraction。

$\partial C^k / \partial q^\alpha$ 把 $R_{\text{Joint}}$ 当 constant 容易算：

$$\frac{\partial C^k}{\partial q^\alpha} = S_H^\alpha \text{diag}_4(R_H) T^\alpha \tag{31}$$

$\partial R_H / \partial q$ 难算。但因为 $R \approx A$ 且 $q$ 是 $A$ 的 linear function：

$$\frac{\partial \text{vec}(\Delta R_H R^\top)}{\partial q} \approx (I_3 \otimes \Delta R_H) \tilde{I}_{9 \times 12} \tag{34}$$

$\tilde{I}_{9 \times 12}$ 是 constant permutation matrix（re-index $\text{vec}(A^\top)$ 到 $q$，translation DOF 处 zero-pad）。

为了消除 $R \approx A$ 的近似误差，利用 **skew-symmetry**：$\partial \text{vec}(\Delta R_H R^\top) / \partial q$ 是 12 个 vectorized skew-symmetric matrices 的 stack，每个有 3 个 independent entry。强行 "skew-symmetrize"：

$$s_{\ell,1} \gets 0, s_{\ell,5} \gets 0, s_{\ell,9} \gets 0$$

$$(s_{\ell,2}, s_{\ell,4}) \gets \pm \frac{1}{2}(|s_{\ell,2}| + |s_{\ell,4}|) \tag{35}$$

（其余 pair 类似）。这样保证 gradient 是真正的 skew-symmetric matrix，对应 rigid body rotation 的 tangent space。

---

## 8. 四种 topology 的 specialized solvers

### 8.1 Case I: Joint chain (block-tridiagonal)

对 chain（$M - K = 1$，每个 joint 连两个相邻 body），dual matrix 是 **block-tridiagonal**：

$$\begin{bmatrix} D^1 & B^1 & & \\ B^{1^\top} & D^2 & \ddots & \\ & \ddots & \ddots & B^{K-1} \\ & & B^{K-1}^\top & D^K \end{bmatrix} \begin{bmatrix} \delta\lambda^1 \\ \vdots \\ \delta\lambda^K \end{bmatrix} = \begin{bmatrix} b^1 \\ \vdots \\ b^K \end{bmatrix} \tag{39}$$

其中：

$$D^j = \nabla C_j^j (H_A^j)^{-1} \nabla^\top C_j^j + \nabla C_{j+1}^j (H_A^{j+1})^{-1} \nabla^\top C_{j+1}^j \in \mathbb{R}^{C_j \times C_j} \tag{37}$$

$$B^j = \nabla C_{j+1}^j (H_A^{j+1})^{-1} \nabla^\top C_{j+1}^{j+1} \in \mathbb{R}^{C_j \times C_{j+1}}$$

用 **block Thomas algorithm** [Press 2007] 可以 $O(K)$ 解决。这是为什么 1M-link pulley system 能跑 904 ms/step 的关键。

### 8.2 Case II: Joint tree (ABD-ABA)

Featherstone 的 **Articulated Body Algorithm (ABA)** [Featherstone 2008] 推广到 ABD。

**Spatial twist** mapping：

$$G(A^j) = \begin{bmatrix} \frac{1}{2}[q_1^j]_\times & \frac{1}{2}[q_2^j]_\times & \frac{1}{2}[q_3^j]_\times & 0 \\ 0 & 0 & 0 & I_3 \end{bmatrix} \in \mathbb{R}^{6 \times 12} \tag{42}$$

$$V^j = G(A^j) \dot{q}^j = [\omega^{j^\top}, v^{j^\top}]^\top$$

**直觉**：angular velocity $\omega$ 由 $A$ 的 3 列 $q_1, q_2, q_3$ 与对应 time derivatives 的 cross product 之和的一半给出（这是因为 $A$ 接近 $R$ 时，$\dot{R} = [\omega]_\times R$ 给出 $\omega = \frac{1}{2}(q_1 \times \dot{q}_1 + q_2 \times \dot{q}_2 + q_3 \times \dot{q}_3)$）；linear velocity $v = \dot{t}$。

**Rigid-motion embedding**（reverse map）：

$$E(A^j) = \begin{bmatrix} -[q_1^j]_\times & 0 \\ -[q_2^j]_\times & 0 \\ -[q_3^j]_\times & 0 \\ 0 & I_3 \end{bmatrix} \in \mathbb{R}^{12 \times 6} \tag{50}$$

满足 $G(A) E(A) = I_6$（rigid motion 子空间内互逆）。

**ABD joint subspace**：

$$S_{abd}^j = E(A^j) S^j \in \mathbb{R}^{12 \times m_j} \tag{51}$$

$S^j \in \mathbb{R}^{6 \times m_j}$ 是标准 Featherstone 的 joint subspace matrix，$m_j = 6 - C_j$ 是 joint DOF count。

**Gyroscopic term 自动消失**：从 RBD 的方程 $W_{\text{dyn}} = I\dot{V} + V \times^* (IV)$，project 到 ABD：

$$f_{A,\text{dyn}}^j = \underbrace{G^\top I^j G}_{M_A^j} \ddot{q}^j + (G^\top I^j \dot{G} \dot{q} + G^\top(V \times^* IV)) \tag{46}$$

而 Euler-Lagrange 给：

$$f_{A,\text{dyn}}^j = M_A^j \ddot{q}^j + (\dot{M}_A^j \dot{q} - \partial T / \partial q) \tag{47}$$

在 ABD 中 $M_A$ 是 constant（pre-computed），$T = \frac{1}{2}\dot{q}^\top M_A \dot{q}$ 不依赖 $q$，所以两项都为零：

$$G^\top I^j \dot{G} \dot{q} + G^\top(V \times^* IV) = 0 \tag{49}$$

**这就是 ABD 不需要 gyroscopic term 的原因**——它被 ABD 坐标的线性性质自动吸收。

**Upward condensation (leaf → root)**：

$$U^j = \hat{H}_A^j S_{abd}^j, \quad D^j = S_{abd}^{j^\top} U^j \tag{52}$$

Solve $D^j \alpha^j = S_{abd}^{j^\top} \hat{f}_A^j$，然后 condensed contributions：

$$\Delta H_A^j = \hat{H}_A^j - U^j (D^j)^{-1} U^{j^\top} \tag{53}$$

$$\Delta f_A^j = \hat{f}_A^j - U^j \alpha^j$$

通过 relative rotation $R_{\text{rel}} = A^{p^\top} A^j$ 旋转到 parent frame 并 accumulate：

$$\hat{H}_A^p \mathrel{+}= X^{j^\top} \Delta H_A^j X^j, \quad \hat{f}_A^p \mathrel{+}= X^{j^\top} \Delta f_A^j \tag{54}$$

$X^j = \text{diag}_4(R_{\text{rel}}^\top)$。

**Downward pass (root → leaf)**：给定 parent increment $\delta q^p$，对每个 child solve local KKT：

$$\begin{bmatrix} \hat{H}_A^j & \nabla C_j^{j^\top} \\ \nabla C_j^j & 0 \end{bmatrix} \begin{bmatrix} \delta q^j \\ \delta \lambda^j \end{bmatrix} = \begin{bmatrix} \hat{f}_A^j \\ r^j \end{bmatrix} \tag{57}$$

其中 $r^j = -\nabla C_p^j \delta q^p$。Global nonlinear solve 被分解为一系列 lightweight joint-size local solves。

Featherstone's book: https://link.springer.com/book/10.1007/978-3-540-73931-5

### 8.3 Case III: Joint loop

Loop = chain 头尾相连。把某个 body $j$ 暂时"移除"，partition KKT：

$$\begin{bmatrix} \mathcal{A} & C^\top \\ C & \mathcal{D} \end{bmatrix} \begin{bmatrix} w_\mathcal{A} \\ w_\mathcal{D} \end{bmatrix} = \begin{bmatrix} b_\mathcal{A} \\ b_\mathcal{D} \end{bmatrix} \tag{58}$$

$\mathcal{A}$ 是剩余 chain 系统（block-tridiagonal，Thomas 解决），用 **Schur complement**：

$$S = \mathcal{D} - C \mathcal{A}^{-1} C^\top$$

是 low-rank system，solve 后 back-substitute 得 $w_\mathcal{A}$。

### 8.4 Case IV: Joint graph

最 general 情况用 **multi-directional block Gauss-Seidel**：把系统看作多个 joint chains 的组合，沿预先定义的 chains 在 dual space 中逐 joint relax。

---

## 9. Experimental results 深度分析

### 9.1 Single-body 验证（Figs. 7-10）

| Benchmark | 验证内容 | 关键观察 |
|---|---|---|
| **Spinning box** (Fig. 7) | Linear/angular momentum conservation | $p_0 = [100,0,0]^\top$ kg·m/s, $L_0 = [0,100,0]^\top$ kg·m²/s。Linear momentum 完全保持，angular momentum 有 implicit integration 的 numerical damping（$h$ 越小越好），ABD 与 implicit RBD 匹配 |
| **T-handle** (Fig. 8) | Intermediate-axis theorem (Dzhanibekov effect) | $\omega_0 = 3$ rad/s 沿 intermediate principal axis。Reference 用 RK4 at $h = 10^{-4}$ s。ABD 准确重现周期性翻转，比 implicit RBD 快 30%+ |
| **Heavy top** (Fig. 9) | Precession + nutation | $5^\circ$ tilt, $\omega = 10$ rad/s。ABD 比 implicit RBD 对 $h$ 更不敏感，coarse step 下 waveform distortion 更小 |
| **Physical pendulum** (Fig. 10) | Analytic elliptic-integral reference | $\theta(t) = \pi/2 - 2\arcsin(\kappa \text{sn}(K(\kappa) - \omega_{\text{lin}} t, \kappa))$，$K(\kappa)$ 是第一类完全 elliptic integral，$\text{sn}(\cdot, \kappa)$ 是 Jacobi elliptic sine。ABD 比 implicit RBD 更贴近解析解 |

### 9.2 Multi-body 比较（Figs. 11-14）

**Fig. 11 chain with heavy end mass**：$h = 10^{-3}$ s 下 MuJoCo equality constraint 有 joint drift，articulated model NaN。Bullet 和 PhysX crash。ABD 即使 $h = 10^{-2}$ s 也稳定，1 iteration 就够，还能捕捉 chain 伸直后的 elastic vibration。

**Fig. 12-13 ball-joint net**：10×10 net, 280 links, $h = 1/30$ s。ABD 1 iteration，PhysX 10 iterations, MuJoCo/Bullet 30 iterations，都明显有 joint gap。VQ [Maloisel 2025] 也 exact 但 <1 ms vs 27 ms per step。

**Fig. 14 scalability**：20×20, 50×50, 100×100 net，$h = 10^{-2}$ s, 1 iteration。ABD 全部稳定，competitors 全部 fail。

VQ paper: https://doi.org/10.1145/3730872

### 9.3 大规模 system（Table 1）

| Scene | # Link | # Cons. | h | E | Sim./step |
|---|---|---|---|---|---|
| Joint net 100×100 | 30K | 120K | 10 ms | 1E9 | 84 ms |
| Pulley (small) | 1.5K | 4.5K | 10 ms | 1E8 | 2 ms |
| **Huge pulley (Fig. 1)** | **1M** | **3M** | **10 ms** | 1E8 | **904 ms** |
| Willow tree | 21K | 63K | 10 ms | 1E8 | 18 ms |
| Pear tree | 29K | 87K | 10 ms | 1E8 | 23 ms |
| Net cloak | 12K | 48K | 10 ms | 1E9 | 33 ms |
| Armadillo | 2.7K | 10.8K | 10 ms | 1E6 | 116 ms (7 iter) |
| Ragdolls | 1.5K | 6K | 5 ms | 1E6 | 17 ms |
| Falling joints | 1.4K | 2.9K | 10 ms | 1E9 | 54 ms |
| Protein | 14K | 56K | - | 1E6 | 14 ms |

**1 million links 在 single CPU thread 上 904 ms/step**——这是 RBD-based 方法完全达不到的 scale。

### 9.4 Cross-domain applications

- **Embodied AI** (Fig. 21)：Franka Panda pick-and-place，1 iteration/step 保持 joint integrity
- **Protein unfolding** (Fig. 22)：SARS-CoV-2 Spike glycoprotein (PDB: 6VXX) backbone 作为 articulated chain，IK keyframe + ABD dynamics rollout 填补 sparse observations 之间的 motion

PDB 6VXX: https://www.rcsb.org/structure/6VXX
Walls et al. 2020: https://doi.org/10.1016/j.cell.2020.02.058

---

## 10. 与相关工作的 intuition 关系

### 10.1 与 IPC 的关系
ABD 12 DOF 能 store elastic energy，与 **Incremental Potential Contact (IPC)** [Li et al. 2020] 天然兼容。本文 Eq. (3) 的 polynomial energy 就是 IPC-friendly 的。但本文专注于 **articulated rigid** 而非 deformable body contact，contact 处理沿用了之前 unified Newton barrier [Chen et al. 2022] 的思路。

IPC paper: https://doi.org/10.1145/3386569.3392425
Chen et al. 2022: https://doi.org/10.1145/3528223.3530076

### 10.2 与经典 Featherstone ABA 的关系
传统 ABA 在 **minimal coordinate** 下递归传播 articulated inertia，避免显式 constraint forces。本文 ABD-ABA 把它移植到 ABD 坐标：
- 用 $G(A)$ 把 ABD velocity map 到 spatial twist
- 用 $E(A)$ 把 spatial wrench map 回 ABD force
- **Gyroscopic term 自动消失**——这是 ABD linear kinematics 的"福利"

### 10.3 与 co-rotational FEM 的关系
Co-rotational formulation [Müller & Gross 2004] 在 graphics FEM 里是经典 trick——把 rotation 提取出来后 small deformation 假设下 stiffness 是 constant。本文把这个 idea 用在 ABD 上，但因为 ABD 只有一个"element"（control tet），stiffness 就是 $12 \times 12$ 而不是 large sparse matrix，pre-factorization 成本 trivial。

### 10.4 与 reduced coordinate methods 的关系
RedMax [Wang et al. 2019] 在 maximal/hybrid coordinates 下做 articulated dynamics。本文的 dual-space KKT 其实本质上是 **Schur complement 消元后做 reduced solve**，但 reduced 的对象是 joint multipliers 而非 joint angles，所以可以保留 pre-factorized per-body Hessian 的优势。

RedMax: https://doi.org/10.1145/3306346.3322952

---

## 11. Limitations & Future work

- **Inequality constraints (contact)**：barrier methods 会破坏 per-body Hessian 的 block-diagonal 结构，pre-factorization 不再适用。Future: 主动 DOF set 识别
- **GPU parallelization**：dual solver 在大规模 joint network 下可以 GPU 加速（molecular simulation 场景）
- **Non-manifold topologies**：当前 ABD-ABA 假设 tree，loop 用 Schur 但有限

---

## 12. 我的整体评价

这篇 paper 是 **multi-body simulation 的 free lunch**——通过 co-rotation 把 ABD 的 12-DOF "包袱" 转化为优势（pre-factorization + linear kinematic mapping），又通过 dual-space KKT 把 scalability 从 $O(M)$ per-body cost 转化为 $O(\sum C_k)$ joint cost，最后用 topology-aware solvers 把常数变得很小。**1M links on single CPU thread** 是 striking 的数字。

潜在影响：
- **Embodied AI training**：parallel 环境中每个 instance 资源有限，1-iteration ABD 是 ideal
- **Robotics**：real-time control 需要 fast + robust constraint enforcement
- **Structural biology**：protein backbone 模拟
- **Graphics**：large-scale articulated scenes (forests, pulleys, garments)

值得思考的几个点：
1. **Linear elasticity 的极限**：当 material 不是 so stiff 时（e.g. soft robotics），co-rotated approximation 误差如何？
2. **Contact handling**：与 IPC 结合时如何保持 pre-factorization？
3. **Differentiable simulation**：ABD 的 linear kinematic mapping 对 backward pass 友好，可能比 RBD 的 SO(3) parameterization 更适合 differentiable physics

希望这个解读对你 build intuition 有帮助！如果你想 drill down 某个部分（比如 co-rotation property 的几何意义、ABD-ABA 与 Featherstone ABA 的具体差异、或 constraint gradient 的 skew-symmetrize trick），我可以再展开。
