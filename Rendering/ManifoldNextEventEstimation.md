---
source_pdf: ManifoldNextEventEstimation.pdf
paper_sha256: 9e6cd3f0df24aefc6321c0cba6ab37580484acda6ebd1ac52fcdb545471bfa2c
processed_at: '2026-08-05T16:17:42-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MNEE 人话版

## 一句话总结

> **光线穿玻璃打到下面再出来这种 path 特别难采样，MNEE 的做法是先画一条直线，然后像捏橡皮泥一样把它掰成符合折射定律的形状。**

---

## 问题到底有多烦

想象你在渲染一个场景：Harry Potter 脸上有一滴汗珠，汗珠下面是皮肤（diffuse），光要先进汗珠（refraction），打到皮肤，再出来。

这种 path 叫 **SDS path**：Specular（折射）→ Diffuse（皮肤）→ Specular（折射出去）。

传统 **Next Event Estimation (NEE)** 的工作方式是这样的：我现在在皮肤这个 shading point 上，我直接 sample 光源上一个点，连一条直线过去，看看能不能看到光源。如果能，就算 contribution。

问题在于：从皮肤到光源的直线会穿过汗珠，但真实光线走的是折线（满足 Snell's law），所以这条直线**物理上不可能存在**。NEE 直接返回零贡献。

**BDPT (Bidirectional Path Tracing)** 也救不了，它本质还是做直线连接，同样穿不过 specular interface。

你可能会想：那就用 random walk 慢慢碰运气呗？可以，但 SDS path 在 path space 里占的比例极小，random walk 基本永远碰不到。这就像在太平洋里捞一根针。

---

## 以前的解法都有什么毛病

### Photon Mapping

往场景里打几千万个光子，存在一个 cache 里，渲染时去查。问题是：
- Memory 巨大（几千万光子）
- 大部分光子浪费了（只有 caustic 区域需要，但光子到处都是）
- Animation 时 cache 每帧不同，temporal flicker

### MLT (Metropolis Light Transport)

用 Markov chain 探索 path space：找到一个好 path 之后，在它附近 perturb，找更多好 path。

听起来很美，实际用起来：
- Markov chain 有 "memory"，相邻 sample 相关性强
- 产生 **low-frequency noise**（大的色块 blotch）
- Animation 中每帧 noise pattern 不一样，画面 "boiling"
- Bidirectional mutation 的 acceptance rate 低于 1%，探索效率极差

对于电影渲染，temporal stability 是命根子。观众看到画面忽明忽暗的闪烁会疯掉。所以 MLT 在 production 基本没人用。

---

## MNEE 的核心 insight

Paper 原话很妙：让 NEE 变得 **"stubborn"**（固执）。

普通 NEE 发现直线连接违反物理定律就放弃了。MNEE 说：别放弃，这条直线虽然不对，但它**接近**某条正确的 path，我们把它当初始猜测，用 **Newton iteration** 慢慢修正，直到它满足所有 half vector constraints。

类比：你写代码有个 bug，编译器报错。普通 NEE 直接关掉 IDE。MNEE 说：这个 bug 不大，我 local search 修一下，几轮迭代就好。

---

## 具体怎么"掰弯"的

### Step 1: 构造 seed path

从 shading point $\mathbf{x}_b$ 到光源点 $\mathbf{x}_c$ 画一条直线，记录这条直线穿过的所有 transmissive interface 的交点。这就是 seed path $\mathbf{Y}$。

此时 seed path 不满足 Snell's law，是"错的"。

### Step 2: Sample target half vectors

对每个 transmissive vertex，按 BSDF 的分布 sample 一个 half vector $\mathbf{h}_i$。这是我们的"目标"——最终 path 要满足这些 half vector constraints。

### Step 3: Newton iteration（manifold walk）

这是核心。我们有一组 constraints：每个 vertex 的 half vector 要等于 target。当前 seed path 的 half vector 跟 target 有差距 $\Delta \mathbf{H}$。

我们需要找到 position 调整量 $\Delta \mathbf{X}$，使得：

$$M \cdot \Delta \mathbf{X} = \Delta \mathbf{H}$$

其中 $M$ 是 **constraint derivative matrix**（block-tridiagonal），描述 "如果我移动 vertex 位置，half vector 会怎么变"。

这个 $M$ 来自 **differential geometry**：你需要知道 surface normal 怎么随 position 变化（curvature），才能算 half vector 对 position 的导数。

解这个线性系统（block-tridiagonal，$O(k)$ 复杂度），得到 $\Delta \mathbf{X}$，移动 vertices，project 回 surface，重复直到收敛。

### Step 4: 算 contribution

收敛后的 path 满足所有物理约束，算它的 measurement contribution 和 PDF，塞进 MIS 框架。

---

## 为什么 Newton iteration 能 work

关键数学 fact：所有满足 Fermat's principle 的 paths 在 path space 里构成一个 **manifold**（流形）。

Seed path（直线连接）不在 manifold 上，但它"靠近" manifold 上的某个点。Newton iteration 就是把 seed path **投影** 到 manifold 上。

这就像：你在 3D 空间里有个点，你想找它到某个曲面上的最近点。你用梯度下降（这里是 Newton，因为有 analytic Jacobian）一步步走过去。

Block-tridiagonal 结构来自 path 的 **locality**：每个 vertex 的 half vector 只依赖它自己和左右邻居的位置。这是 Markov property，允许 exact $O(k)$ 求解，不用求 $O(k^3)$ 的逆矩阵。

---

## 为什么比 MLT 好（关键区别）

这是 paper 最 important 的 insight。

**MLT** 用 Markov chain：当前 path → perturb → 新 path → accept/reject → 下一个 path。相邻 samples 之间有强相关性。这个 "memory" 导致：
- Low-frequency noise（色块）
- Temporal instability（每帧 noise pattern 不同）

**MNEE** 是 **single-step perturbation**：
- 每个 sample 独立
- Seed path 永远由 $(\mathbf{x}_b, \mathbf{x}_c)$ deterministically 定义（直线）
- Half vector sampling 是独立的 random event
- 没有 Markov chain，没有 "memory"

所以 MNEE 的 noise 是 **high-frequency white noise**，人眼和 denoiser 都很容易处理。Animation 中每帧的 noise 是 i.i.d. 的，temporal coherence 完全来自 scene 的 spatial coherence，不会有额外的 boiling。

直觉对比：
- MLT 像一个醉汉在 path space 里随机游走，走得很慢，留下一串 correlated footprints
- MNEE 像一个狙击手，每次独立瞄准目标，弹道用 Newton 修正，命中率有限但每次独立

---

## Marginalised distribution 的巧妙处理

这是 paper 里最 math-heavy 的部分，但 intuition 很简单。

在标准 Monte Carlo 里，你要知道 sample 的 PDF $p(\mathbf{X})$ 才能算 unbiased estimator。MNEE 的 sampling 过程是：先 sample seed path $\mathbf{Y}$，再 perturb 成 $\mathbf{X}$。所以：

$$p(\mathbf{X}) = \int p(\mathbf{X}|\mathbf{Y}) \cdot p(\mathbf{Y}) \, d\mathbf{Y}$$

这个 integral 很难算（要对所有可能的 seed paths 积分）。

MLT 的 trick 是用 detailed balance 绕过这个 integral：只要 acceptance probability 满足 detailed balance，stationary distribution 就对了。

MNEE 的 trick 更简单粗暴：**假设给定 half vectors $\mathbf{H}$，seed path $\mathbf{Y}$ 和 admissible path $\mathbf{X}$ 之间存在 bijection**。

这意味着对于给定的 $\mathbf{X}$，只有一个 $\mathbf{Y}$ 能 perturb 出它，所以 $p(\mathbf{X}|\mathbf{Y}) = \delta(\mathbf{Y})$，integral 退化成单点：

$$p(\mathbf{X}) = p(\mathbf{Y}^*) \cdot p_{d\mathbf{H}}(\mathbf{H}) \cdot \left| \frac{d\mathbf{H}}{d\mathbf{X}} \right|$$

这个 bijection 假设在什么情况下成立？当 half vectors 唯一确定 path 的 inner vertices 时。对于**单个 transmissive layer** 这个假设基本成立。对于复杂情况（multiple admissible paths，见 Fig 7 的 caustic folding），假设破坏，Newton 只找到一个 path，引入 bias。

---

## Production 数据

Paper 在 Weta Digital 的 **Manuka** renderer 里实现（就是渲染 Hobbit 那个）。

Fig 6 是 Gandalf，脸上 sweaty，用 MNEE 渲染。数据：
- MNEE 比 NEE 慢 45%（Newton iteration 的 overhead）
- 但对于 caustic paths，效率提升 **thousands of times**
- Net win：caustic 区域占画面比例小但视觉关键，值得

Table 1 的 Newton iteration 性能：
- 5 iterations：33% 成功率，avg 3.55 iterations
- 15 iterations（推荐上限）：~45% 成功
- 50 iterations：46-59% 成功（over-relaxation 帮助有限）
- 500 iterations：额外 gain < 1%

**Practical takeaway**：max 15 iterations，超过不值得。

---

## 局限性（paper 很诚实地说了）

1. **Reflected caustics 不处理**：MNEE 只在 $\mathbf{x}_b$ 和 $\mathbf{x}_c$ 之间有 occlusion（需要穿 interface）时构造 seed path。Reflected caustics（光在物体外面反射聚焦）不在这个范围。

2. **Multimodal paths**：一个 seed path 可能对应多条 admissible paths（caustic 自己折叠到自己身上，Fig 7）。Newton 只找到一个，其他 path 采样不到。Paper 建议未来用 multimodal optimization。

3. **Geometry 质量**：Differential geometry 需要法向连续可导（$C^2$）。如果 geometry 有 displacement 或 poor tessellation（Fig 9, 10），normal derivative 不可靠，MNEE 和 HSLT 都会 fail。

4. **Roughness 太大不值得**：$\alpha$ 很大时（Fig 8），BSDF lobe 很宽，普通 NEE 直接连就行，MNEE 的 overhead 反而拖慢。

5. **不是 silver bullet**：MNEE 是 specialized technique，必须嵌入 MIS 框架，和 PT、NEE、BDPT 配合用。单独用 MNEE 是 biased 的（覆盖不了所有 path types）。

---

## Deep Learning 类比（给 Andrej 的）

从 ML 角度看，MNEE 的结构很像 **amortized inference + iterative refinement**：

- **Encoder**：$(\mathbf{x}_b, \mathbf{x}_c)$ → seed path $\mathbf{Y}$（deterministic，直线连接）。类似 encoder 把 input 编码成 latent。
- **Latent sampling**：sample half vectors $\mathbf{H}$（按 BSDF 分布）。类似在 latent space 里 sample。
- **Decoder**：Newton iteration 把 $\mathbf{H}$ decode 成 path $\mathbf{X}$。类似 decoder 把 latent 解码成 output。
- **Jacobian**：analytic differential geometry 提供 exact Jacobian，类似用 autograd 但 closed-form，所以快。

Newton iteration 本身类似 **implicit function solver**：给定 constraints $\mathbf{H}(\mathbf{X}) = \mathbf{H}_{target}$，求解 $\mathbf{X}$。和 NeRF 的 optimization、GAN inversion 类似，但用 analytic Jacobian 而非 autograd，所以快几个量级。

Block-tridiagonal 结构 = path 的 **Markov property**，类似 chain CRF 的 exact inference，$O(k)$ 而非 $O(k^3)$。

和 normalizing flows 的对比：MNEE 的 change of variables $\mathbf{H} \to \mathbf{X}$ 有 exact Jacobian determinant（在 LU 分解中免费计算），类似 normalizing flow 的 log-determinant trick，但这里是 analytic 而非 learned。

---

## 相关链接

- [MNEE 原文 PDF](https://jo.dreggn.org/home/2015_mnee.pdf) - Johannes Hanika 主页
- [Manifold Exploration (Jakob 2013)](https://www.cs.cornell.edu/~wenzel/ManifoldExploration/) - MNEE 的理论基础
- [Half Vector Space Light Transport (Kaplanyan 2014)](https://research.nvidia.com/publication/2014-08_Natural-constraint-representation) - HSLT, MNEE 直接 build on 此
- [Metropolis Light Transport (Veach 1997)](https://www.keenbeantech.com/mlt.pdf) - 对比方法
- [MIS (Veach & Guibas 1995)](https://www.cs.cornell.edu/~srm/papers/SIGGRAPH1995-mis.pdf) - MIS 框架
- [Weta Digital Manuka](https://www.wetafx.co.nz/) - production 实现
- [PN triangles](https://dl.acm.org/doi/10.1145/364238.364266) - 解决 non-$C^2$ geometry 的可能方案
- [Discrete Exterior Calculus](https://arxiv.org/abs/math/0508341) - 另一种 geometry 表示的 future direction

---

# Manifold Next Event Estimation (MNEE) 详解

## 1. 核心问题: SDS paths 与 refractive caustics

这篇 paper 解决的核心痛点是 **SDS (specular-diffuse-specular) paths** 的采样问题. 想象一个场景: 皮肤上有汗珠, 汗珠下方是 diffuse 皮肤, 光线要先穿过空气-水界面 (specular refraction), 打到 diffuse 皮肤, 再反射回来穿过水-空气界面 (specular refraction), 最后到达 camera. 

传统 **Next Event Estimation (NEE)** 的逻辑是: 从 shading point $\mathbf{x}_b$ 直接 sample 光源上一点 $\mathbf{x}_c$, 然后连接它们. 但如果 $\mathbf{x}_b$ 到 $\mathbf{x}_c$ 之间有 refractive interface, 这条直线连接违反 **Fermat's principle** (实际光路是折线, 需要满足 Snell's law). 所以 NEE 直接给出 zero contribution.

**Bidirectional Path Tracing (BDPT)** 也救不了, 因为它本质上还是做 deterministic connection, 同样无法穿过 specular interface.

MNEE 的 insight: 让 NEE 变得 "stubborn" — 不要因为直线连接违反约束就放弃, 而是把这条 seed path 当作初始猜测, 用 **Newton iteration** 把它 "bend" 成一条满足所有 half vector 约束的 admissible path.

---

## 2. Path space 与 Half vector space 的重新参数化

### 2.1 Path space formulation

Path space $\Omega$ 中的路径 $\mathbf{X} = (\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_k)$, 其中:
- $\mathbf{x}_0$: camera/eye vertex
- $\mathbf{x}_a$ (with $a=0$): 起点
- $\mathbf{x}_b$: NEE 被调用的 shading point (path tracing 最后一个通过 BSDF sampling 创建的 vertex)
- $\mathbf{x}_c$ (with $c=k$): 光源上被 sample 的点

Pixel measurement (Eq.1):

$$I_j = \int_\Omega f(\mathbf{X}) \, d\mathbf{X}$$

其中 $d\mathbf{X} = \prod_{i=0}^{k} d\mathbf{x}_i$ 是 **product vertex area measure**.

Measurement contribution function (Eq.2, 在 projected solid angle measure $d\mathbf{O}^\perp$ 下):

$$f_{d\mathbf{o}^\perp}(\mathbf{X}) = W(\mathbf{x}_0) \, L_e(\mathbf{x}_k) \, \prod_{i=1}^{k-1} f_r(\mathbf{i}_i, \mathbf{x}_i, \mathbf{o}_i)$$

变量含义:
- $W(\mathbf{x}_0)$: eye responsivity (pixel filter × sensor response)
- $L_e(\mathbf{x}_k)$: emitted radiance at light source vertex
- $f_r(\mathbf{i}_i, \mathbf{x}_i, \mathbf{o}_i)$: BSDF at vertex $\mathbf{x}_i$, with incoming direction $\mathbf{i}_i$ (from eye side) and outgoing direction $\mathbf{o}_i$ (toward light side)
- 下标 $i$: vertex index, 上标 $\perp$: projected (onto surface normal)

### 2.2 Half vector 参数化

每个 vertex $\mathbf{x}_i$ 有一个 half vector $\mathbf{h}_i$:

$$\mathbf{h}_i = \text{normalize}(\mathbf{i}_i + \mathbf{o}_i) \quad \text{(reflection)}$$

或对于 refraction (Snell's law form):

$$\eta_i (\mathbf{i}_i \cdot \mathbf{n}_i) - \eta_{i+1} (\mathbf{o}_i \cdot \mathbf{n}_i) = \text{...满足 half vector constraint}$$

关键 insight 来自 [Jak13] 和 [KHD14a]: 给定 path 的两个端点和所有 inner vertices 的 half vectors, 整条 path 可以被唯一确定 (up to local constraints). 这给出了一个 **bijection** between path positions $\mathbf{X}_{b+1,c-1}$ 和 half vectors $\mathbf{H}_{b+1,c-1}$.

---

## 3. Differential geometry: Constraint derivative matrix

这是整个方法的核心数学工具. 对每个 vertex $\mathbf{x}_i$, 我们需要知道 half vector $\mathbf{h}_i$ 如何随相邻 vertices 变化:

$$A_i = d\mathbf{h}_i / d\mathbf{x}_{i-1}, \quad B_i = d\mathbf{h}_i / d\mathbf{x}_i, \quad C_i = d\mathbf{h}_i / d\mathbf{x}_{i+1} \quad \text{(Eq.4)}$$

这些是 $2 \times 3$ 矩阵 (half vector 是 2D, position 是 3D), 通过 **differential geometry** [dC76] 解析计算, 涉及:
- Surface normal $\mathbf{n}_i$
- Normal derivative (curvature tensor): $d\mathbf{n}_i / d\mathbf{x}_i$
- 方向与法向的点积

整个 path 的约束方程 $\mathbf{H} - \mathbf{H}_{target} = 0$ 对应一个 **block-tridiagonal system**:

$$\begin{pmatrix} B_{b+1} & C_{b+1} & 0 & \cdots \\ A_{b+2} & B_{b+2} & C_{b+2} & \cdots \\ 0 & A_{b+3} & B_{b+3} & \cdots \\ \vdots & & & \ddots \end{pmatrix} \begin{pmatrix} \Delta\mathbf{x}_{b+1} \\ \Delta\mathbf{x}_{b+2} \\ \Delta\mathbf{x}_{b+3} \\ \vdots \end{pmatrix} = \begin{pmatrix} \Delta\mathbf{h}_{b+1} \\ \Delta\mathbf{h}_{b+2} \\ \Delta\mathbf{h}_{b+3} \\ \vdots \end{pmatrix}$$

这个 block-tridiagonal 结构可以用 **Thomas algorithm** (tridiagonal 的 block 版本) 或 LU decomposition 高效求逆, 复杂度 $O(k)$ 而非 $O(k^3)$ [Sal06].

---

## 4. 为什么不用 Markov chain: Correlated sampling 的关键区别

### 4.1 MLT 的问题

**Metropolis Light Transport (MLT)** [VG97] 用 Markov chain 探索 path space. Acceptance probability (Eq.3):

$$a = \min\left\{1, \frac{f(\mathbf{X}^t) / T(\mathbf{X}^i \to \mathbf{X}^t)}{f(\mathbf{X}^i) / T(\mathbf{X}^t \to \mathbf{X}^i)}\right\}$$

变量:
- $\mathbf{X}^i$: current state
- $\mathbf{X}^t$: tentative proposal
- $T(\cdot \to \cdot)$: transition probability
- $a$: acceptance probability (detailed balance)

MLT 的问题:
1. **Low-frequency noise**: Markov chain 的 mixing rate 慢, 相邻 pixels 之间相关性高, 产生 blotchy artifacts
2. **Temporal instability**: animation 中每帧的 noise pattern 不同, 产生 boiling/flickering
3. **Bidirectional mutation acceptance rate < 1%**: 探索效率低

### 4.2 MNEE 的 correlated sampling

MNEE 不用 Markov chain. 它是 **single-step perturbation**: seed path $\mathbf{Y}$ 总是由 $(\mathbf{x}_b, \mathbf{x}_c)$ deterministically 定义 (直线连接). Marginalised distribution (Eq.5):

$$p(\mathbf{X}) = \int_{\mathbf{Y} \in \Omega} p(\mathbf{X}|\mathbf{Y}) \cdot p(\mathbf{Y}) \, d\mathbf{Y}$$

关键假设: 给定 half vectors $\mathbf{H}_{b+1,c-1}$, seed path $\mathbf{Y}$ 和 admissible path $\mathbf{X}$ 之间存在 **bijection**. 所以 $p(\mathbf{X}|\mathbf{Y}) = \delta(\mathbf{Y})$, integral 退化:

$$p(\mathbf{X}) = p(\mathbf{Y}^*) \cdot p_{d\mathbf{H}}(\mathbf{H}) \cdot \left| \frac{d\mathbf{H}}{d\mathbf{X}} \right|$$

其中 $\mathbf{Y}^*$ 是唯一对应的 seed path. 这避免了 MLT 中需要显式计算 marginal distribution 的问题, 同时保持了 **frame-to-frame independence** (每个 sample 独立), 所以 temporal stability 好.

直觉: MLT 像随机游走探索整个 path space, MNEE 像 "guided missile" — 每次从 shading point 出发, 精确瞄准光源, 中间用 Newton iteration 修正弹道.

---

## 5. 算法流程详解

### 5.1 Sampling (Algorithm 1)

```
Input: path prefix X_{a,b} (从 eye 到 shading point)
Output: full path X_{a,c} with contribution

1. Sample light source point x_c with PDF p_dx(x_c)  // 标准 NEE
2. Construct seed path Y_{b+1,c-1}: 直线 x_b → x_c 上所有 transmissive vertices
3. Sample half vectors H_{b+1,c-1} with PDF p_dH(H)  // 用 BSDF 的 sampling function
4. Newton iteration: X_{b+1,c-1} = h_to_positions(H, Y)
5. Return f_r(x_b) · f_dH(X_{b,c}) · L_e(x_c) / (p_dH(H) · p_dx(x_c))
```

### 5.2 PDF evaluation (Algorithm 2)

关键: PDF 评估需要验证给定 admissible path $\mathbf{X}$, 是否能从某个 seed path 通过 sampling 重构出它.

```
1. p_dx(x_c) ← PDF of light point
2. Construct seed path Y: 直线 (x_b, x_c)
3. p_dH(H) ← PDF of half vectors on INPUT path (不是 converged path!)
4. X' = h_to_positions(H, Y)  // 重新跑 Newton
5. if |X - X'| > ε: return 0  // 找到不同的 path, 失败
6. return p_dX(X_{b+1,c-1}) · p_dx(x_c)
```

为什么 step 3 用 input path 的 half vectors 而非 converged path 的? 因为 Newton solver 有 finite epsilon tolerance, converged path 可能略有偏差. 用 input path 的 half vectors 保证 MIS weights 一致.

---

## 6. Half vector space measurement: 消除 geometry terms

Eq.6 是数值稳定性的关键. 在 half vector space 计算 measurement:

$$f_{d\mathbf{H}}(\mathbf{X}_{b,c}) = \left| \frac{d\mathbf{o}_b}{d\mathbf{x}_c} \right| \prod_{i=b+1}^{c-1} f_r(\mathbf{x}_i) \left| \frac{d\mathbf{o}_i}{d\mathbf{h}_i} \right| \left| \frac{\langle \mathbf{o}_i, \mathbf{n}_i \rangle}{\langle \mathbf{h}_i, \mathbf{n}_i \rangle} \right|$$

变量解释:
- $|d\mathbf{o}_b / d\mathbf{x}_c|$: Jacobian from light point area to outgoing direction at $\mathbf{x}_b$, 等于 $G(\mathbf{x}_b, \mathbf{x}_{b+1}) |T_{b+1}|$, 其中 $T_{b+1}$ 是 transfer matrix
- $|d\mathbf{o}_i / d\mathbf{h}_i|$: microfacet Jacobian, reflection 时为 $4 |\langle \mathbf{o}_i, \mathbf{h}_i \rangle|$
- $\langle \mathbf{o}_i, \mathbf{n}_i \rangle / \langle \mathbf{h}_i, \mathbf{n}_i \rangle$: projected solid angle correction
- 下标 $i \in [b+1, c-1]$: 所有 inner transmissive vertices

好处: sub-path $\mathbf{X}_{b+1,c}$ 不再有 geometry terms (visibility / distance squared), 避免数值爆炸.

### Specular case (Eq.9)

对于纯 specular vertex, half vector PDF 是 Dirac delta:

$$f_{r, d\mathbf{h}} = f_r(\mathbf{x}) \left| \frac{d\mathbf{o}}{d\mathbf{h}} \right| \langle \mathbf{o}, \mathbf{n} \rangle = \kappa \cdot \delta_{d\mathbf{h}}(\mathbf{h})$$

其中 $\kappa$ 是 Fresnel term ($R$ for reflection). Delta function 在 numerator 和 denominator 中 cancel, 永远不需要显式 evaluate. 这是 half vector space formulation 的优雅之处.

---

## 7. Jacobian determinant 计算 (Eq.8)

PDF conversion (Eq.7):

$$p_{d\mathbf{X}}(\mathbf{X}) = p_{d\mathbf{H}}(\mathbf{H}) \left| \frac{d\mathbf{H}}{d\mathbf{X}} \right|$$

Jacobian determinant 在 LU decomposition 中 "免费" 计算 (Eq.8):

$$\left| \frac{d(\mathbf{h}_{b+1} \ldots \mathbf{h}_{c-1})}{d(\mathbf{x}_{b+1} \ldots \mathbf{x}_{c-1})} \right| = \prod_{i=b}^{c-1} |\Lambda_i|$$

其中 $\Lambda_i$ 是 LU 分解中 diagonal blocks 的 determinant. 注意 index range 是 $[b, c-1]$ 而非整条 path $[0, k-1]$, 因为只对 sub-path 求导.

**Numerical warning**: single precision 不够, 必须 double precision. 因为 $|d\mathbf{o}_b / d\mathbf{x}_c|$ 涉及 transfer matrix determinant, 在 grazing angle 时数值极小.

---

## 8. Newton solver / Predictor-Corrector 细节

### 8.1 基本迭代

给定 seed path $\mathbf{Y}$ 和 target half vectors $\mathbf{H}_{target}$:

```
Iteration i:
  1. Compute current half vectors H^i from X^i
  2. ΔH = H_target - H^i
  3. Solve: M · ΔX = ΔH  (M = block-tridiagonal constraint matrix)
  4. X^{i+1}_temp = X^i + ΔX
  5. Project X^{i+1}_temp back to surface: X^{i+1} = P(X^{i+1}_temp)
```

### 8.2 Projection: Closest point search vs Ray tracing

Paper 的一个实用发现: 用 **closest point search** 而非 ray tracing 做 projection. 原因:
- Ray tracing 在 surface 接近 parallel 时容易 miss
- Closest point search 可以限制在半径 $|\Delta\mathbf{x}_i|$ 内, 用 BVH 遍历, 对复杂 geometry 更快

### 8.3 Successive over-relaxation

当 seed path $\mathbf{Y}$ 离 admissible path $\mathbf{X}$ 太远时 (比如 caustic 的 dim region), Newton 步长可能陷入 local minimum. 解决方案: 允许 step size > 1, 允许 error 临时增加.

Table 1 数据:
- 5 iterations: 33% success, 3.55 avg iterations
- 15 iterations (推荐): ~45% success
- 50 iterations: 46% success (A) / 59% success (B, with over-relaxation)
- 500 iterations: <1% 额外 gain

**Practical recommendation**: max 15 iterations. 超过 50 收益递减.

---

## 9. Multiple Importance Sampling (MIS) 集成

MNEE 是一个 **specialised technique**: 它只覆盖 transmissive suffix paths. 对于:
- Reflected caustics
- Direct visibility (no interface)
- Paths MNEE 无法 converge 的

需要其他 techniques (PT, NEE, BDPT) 通过 MIS [VG95] 补充.

MIS weight:

$$w_{MNEE}(\mathbf{X}) = \frac{p_{MNEE}(\mathbf{X})}{\sum_j p_j(\mathbf{X})}$$

当 MNEE PDF 为 0 (convergence failure 或 non-transmissive path), $w_{MNEE} = 0$, 完全交给其他 techniques. 这保证了 **unbiasedness** (modulo Sec 3.5).

### Outlier removal (Sec 3.5, biased variant)

如果 MNEE 应该高效 (transmissive suffix 存在) 但失败 (PDF = 0), 可以 discard 其他 techniques 的 contribution too, 设置它们 MIS weight = 0. 这引入 bias 但去除 outlier noise, 对 production rendering 有用.

---

## 10. Limitations 与 future work

1. **Reflected caustics**: 当 $\mathbf{x}_b$ 和 $\mathbf{x}_c$ 直接可见 (无 occlusion), MNEE 不构造 seed path, 所以不处理 reflected caustics outside shadow region

2. **Multimodal paths** (Fig 7): 一个 seed path 可能对应多个 admissible paths (caustic folding), Newton 只找到一个. 需要 multimodal optimization

3. **Displacement & non-$C^2$ geometry** (Fig 9, 10): Differential geometry 需要连续的法向导数. PN triangles [VPBM01] 或 Discrete Exterior Calculus [Hir03, DHLM05] 可能是解决方案

4. **Rough interfaces**: roughness $\alpha$ 很大时, NEE 本身就够用, MNEE 的 overhead 不值得 (Fig 8)

---

## 11. 与相关工作的关系

| Method | Seed path | Exploration | Temporal stability | Memory |
|--------|-----------|-------------|-------------------|--------|
| NEE | Direct connection | None | Good | None |
| BDPT | Two random walks | Deterministic connection | Good | None |
| Photon mapping | Light path cache | Kernel search | Medium | High |
| MLT [VG97] | Current path | Markov chain | Poor | None |
| ERPT [CTE05] | Markov chain + redistribution | Markov chain | Medium | None |
| HSLT [KHD14a] | Current path in H-space | Markov chain in H-space | Medium | None |
| **MNEE** | Straight line (deterministic) | Single-step correlated perturbation | **Good** | **None** |

---

## 12. Intuition building: 为什么这个方法 work

**Key insight 1: Path space manifold**

所有满足 Fermat's principle (half vector constraints) 的 paths 构成 path space 中的一个 **manifold** $\mathcal{M} \subset \Omega$. 在这个 manifold 上, 我们可以用 half vectors 作为 local coordinates.

**Key insight 2: Seed path 是 manifold 的 "投影"**

直线连接 $\mathbf{x}_b \to \mathbf{x}_c$ 违反 constraints, 但它 "接近" manifold 上的某个点. Newton iteration 就是把 seed path "投影" 到 manifold 上.

**Key insight 3: 为什么 single-step 就够**

因为 seed path 是 deterministically defined (由 $\mathbf{x}_b, \mathbf{x}_c$ 唯一确定), 且 half vector sampling 是 independent 的. 没有 Markov chain 的 mixing problem. 每个 sample 都是 i.i.d., 所以 temporal coherence 来自 spatial coherence of the scene, 而非 sampling correlation.

**Key insight 4: 为什么比 MLT 好 for animation**

MLT 的 Markov chain 有 "memory": 当前 sample 影响下一个 sample. 这导致:
- 相邻 frames 的 noise pattern 不同 (不同 starting points)
- Low-frequency noise 难以 average out

MNEE 的每个 sample 独立, noise 是 high-frequency, 人眼和 denoiser 都更容易处理.

---

## 13. Production 实用性

Paper 在 Weta Digital 的 Manuka renderer 中实现 (Fig 6, Gandalf). 数据:
- MNEE 比 NEE 慢 45% (overhead 来自 Newton iteration)
- 但对于 caustic paths, 效率提升 thousands of times
- Net win: production 场景中 caustics 占比小但视觉重要

适用场景:
- 水滴在皮肤上
- 玻璃球内的物体
- 汗水、眼泪
- 任何 refractive interface 后的 caustic

不适用场景:
- 大面积 rough surface (NEE 够用)
- Reflected caustics (需要 future work)
- 高频 displacement geometry (differential geometry 不可靠)

---

## References (web links)

- [Original paper (EGSR 2015)](https://jo.dreggn.org/home/2015_mnee.pdf)
- [Jakob 2013 - Light transport on path-space manifolds](https://www.cs.cornell.edu/~wenzel/ManifoldExploration/) - MNEE 的理论基础
- [Kaplanyan et al. 2014 - Half vector space light transport](https://research.nvidia.com/publication/2014-08_Natural-constraint-representation) - HSLT, MNEE 直接 build on 此
- [Veach 1997 - Metropolis Light Transport](https://www.keenbeantech.com/mlt.pdf) - 对比方法
- [Veach & Guibas 1995 - MIS](https://www.cs.cornell.edu/~srm/papers/SIGGRAPH1995-mis.pdf) - MIS 框架
- [Hanika & Droske - Manuka renderer](https://www.wetafx.co.nz/) - production 实现
- [Heitz & d'Eon 2014 - Visible normals sampling](https://eheitzresearch.wordpress.com/) - half vector sampling 策略
- [PN triangles](https://dl.acm.org/doi/10.1145/364238.364266) - 解决 non-$C^2$ geometry 的可能方案
- [Discrete Exterior Calculus](https://arxiv.org/abs/math/0508341) - 另一种 geometry 表示

---

## 14. 给 Andrej 的额外 intuition

从 deep learning 的角度类比: MNEE 像是一个 **amortized inference** 方法. 

- MLT 是 MCMC: 慢慢探索 posterior, 有 mixing time 问题
- MNEE 是 variational / direct sampling: 用一个 deterministic "encoder" (seed path = 直线连接) 加一个 "decoder" (Newton iteration = iterative refinement) 来 generate samples

Newton iteration 本质上是一个 **implicit function solver**: 给定 half vector constraints $\mathbf{H}(\mathbf{X}) = \mathbf{H}_{target}$, 求解 $\mathbf{X}$. 这和 GAN inversion, NeRF optimization 中的 gradient descent 类似, 但用的是 **analytic Jacobian** (来自 differential geometry) 而非 automatic differentiation, 所以快很多.

Block-tridiagonal 结构是 path 的 **Markov property**: 每个 vertex 只依赖邻居, 类似 chain CRF. 这允许 $O(k)$ 的 exact inference 而非 $O(k^3)$.

这个 paper 的 elegance 在于: 它识别到了 NEE 的失败模式 (直线连接违反 Fermat), 然后用 manifold projection 修复, 同时避免了 MLT 的 temporal instability. 是一个 **minimal intervention** 的设计 — 只在最需要的地方加 complexity.
