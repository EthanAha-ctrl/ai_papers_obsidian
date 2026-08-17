---
source_pdf: Direct Informed Sampling on Riemannian.pdf
paper_sha256: e12628f162070a79107b6332f7a7d69fe6c624753d994e1fd9cda1c7645776eb
processed_at: '2026-08-03T22:05:00-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 1. 一句话核心

作者干的事：**机器人在"走起来重量不一样"的空间里找路时，原来那套"只在椭球里撒点"的加速技巧失灵了。他们找到一个"保底矩阵"——既保住下界性质又保留方向信息，然后用 Cholesky 把歪掉的空间拉直，椭球又变椭球，老采样算法直接复用。**

---

## 2. 背景：motion planning 为啥要 informed sampling

采样-based planner（RRT*、BIT* 这些）干的事很朴素：在 configuration space $\mathcal{Q}$ 里**乱撒点**，连成树，慢慢收敛到最优解。问题：乱撒点浪费——大量点根本不可能改进当前解。

**Informed sampling** 的 insight：如果你已经找到一个 cost 是 $c_{\text{best}}$ 的解，那**只有"从 start 经过该点再到 goal 的总长 $< c_{\text{best}}$"的点才可能改进解**。这个集合叫 informed set $X_{\hat{f}}$。

在 Euclidean cost 下，$X_{\hat{f}}$ 是个**椭球**——以 start、goal 为焦点的 prolate hyperspheroid (PHS)。椭球里撒点有 closed-form 直接采样算法 [4]（https://arxiv.org/abs/1706.06391），无 rejection，飞快。

类比：你在大厅里找最短路径，已经知道当前最好路径长度，那么"可能改进解的点"都集中在以起终点为焦点的椭球里。椭球外面撒点等于浪费。

---

## 3. 问题：Riemannian metric 把椭球搞碎了

但机器人 cost 经常**不是直线距离**：

- **Kinetic energy metric**: $\mathbf{G}(q) = \mathbf{M}(q)$（机械臂 mass matrix）。姿态不同，惯量不同，"走同样角度"的 cost 不一样。低惯量姿态走起来"轻"，高惯量姿态"重"。
- **Pullback metric**: 把 task-space 的距离通过 Jacobian 拉回 joint space。
- **Weighted metric**: 手动给某些关节加权，避免它们乱动。

这些都写成 configuration-dependent Riemannian metric tensor $\mathbf{G}(q) \in \mathbb{S}^n_{++}$——一个随 $q$ 变化的 SPD matrix field。弧长：

$$
L(\pi) = \int_0^1 \sqrt{\dot{\pi}(t)^\top \mathbf{G}(\pi(t)) \dot{\pi}(t)} \, dt
$$

变量解释：$\pi: [0,1] \to \mathcal{Q}$ 是一条曲线，$\dot{\pi}(t)$ 是它在 $t$ 处的切向量（速度），$\mathbf{G}(\pi(t))$ 是当前 configuration 处的 metric tensor，整段积分是路径的"Riemannian 长度"。

**这时椭球碎了**，原因有二：

### 3.1 Euclidean heuristic 直接 inadmissible

Euclidean distance $\|q_x - q_y\|_2$ 不再是 geodesic distance 的下界。论文 Section VI-A 实测：高 DoF 上 kinetic energy metric 下，Euclidean 对 **99% 的点对高估**真实距离。意思是：本来该被 informed set 包含的好点，因为 heuristic 高估被排除了——planner 被骗。

### 3.2 Scalar eigenvalue bound 太保守

老的 fix [12, 32]：取 metric 全局最小特征值 $\lambda_{\min} = \inf_q \lambda_{\min}(\mathbf{G}(q))$，用 $\sqrt{\lambda_{\min}} \|q_x - q_y\|_2$ 当 heuristic。

类比：你每个月花"食、住、行"三类钱，想做一个"每月至少剩多少"的 budget。Scalar 做法：找所有月份、所有类别里**最穷的那一项**，把它当**所有类别**的保底。结果是住和行本来剩得多，也被强行拉到最穷那一项的水平——超保守。

informed set 还是 isotropic PHS，**所有方向一视同仁**，metric 的方向结构被丢光。

---

## 4. Solution Step 1: Matrix Lower Bound via Loewner Order

### 4.1 Loewner order 是啥

对两个对称矩阵 $\mathbf{A}, \mathbf{B}$，$\mathbf{A} \preceq \mathbf{B}$ 当且仅当 $\mathbf{B} - \mathbf{A}$ 是 PSD，等价于**对所有方向 $v$ 都有 $v^\top \mathbf{A} v \le v^\top \mathbf{B} v$**。

直觉：Loewner order 同时管**所有方向**的二次型不等式，比"只管最小特征值"那个标量强多了。参考：https://en.wikipedia.org/wiki/Loewner_order

### 4.2 Matrix lower bound

定义一个**常数** SPD matrix $\mathbf{G}_{\text{lower}}$，要求它**在所有 $q$ 处都被 $\mathbf{G}(q)$ Loewner-dominate**：

$$
\mathbf{G}_{\text{lower}} \preceq \mathbf{G}(q), \quad \forall q \in \mathcal{Q}
$$

意思：在每个方向 $v$、每个 configuration $q$ 处，$\mathbf{G}_{\text{lower}}$ 给的二次型都 ≤ $\mathbf{G}(q)$ 给的。

然后距离 heuristic 用这个 matrix 诱导的 Mahalanobis-like norm：

$$
\hat{d}(q_x, q_y) = \sqrt{(q_x - q_y)^\top \mathbf{G}_{\text{lower}} (q_x - q_y)} = \|\mathbf{L}^\top(q_x - q_y)\|_2
$$

这里 $\mathbf{G}_{\text{lower}} = \mathbf{L}\mathbf{L}^\top$ 是 Cholesky 分解，$\mathbf{L}$ 是下三角，$\mathbf{L}^\top$ 是上三角。$\mathbf{L}^\top(q_x - q_y)$ 是把差向量通过线性映射"拉直"到一个 Euclidean 空间。

### 4.3 为啥 admissible

Theorem 2 的核心链：

$$
L(\pi) = \int_0^1 \sqrt{\dot{\pi}^\top \mathbf{G}(\pi) \dot{\pi}} \, dt \;\ge\; \int_0^1 \|\mathbf{L}^\top \dot{\pi}\|_2 \, dt \;\ge\; \|\mathbf{L}^\top(q_y - q_x)\|_2 = \hat{d}
$$

第一步用 Loewner order 在每个 $t$ 处压住被积函数；第二步用三角不等式 + 积分。取 inf 即得 $\hat{d} \le d$。所以 admissible。

### 4.4 为啥比 scalar bound 紧

Theorem 1：$\hat{d}_\lambda \le \hat{d}$，等号当且仅当 $q_x - q_y$ 是 $\mathbf{G}_{\text{lower}}$ 对应特征值 $\lambda_{\min}$ 的特征向量。

类比回到 budget：matrix bound 是"食、住、行各类分别取保底"，scalar bound 是"取所有类别最小那个当各类保底"。Matrix bound 永远 ≥ scalar bound，等号只在沿着"最穷那一类"行进时成立。

---

## 5. Solution Step 2: Cholesky 把空间拉直

### 5.1 Isometric embedding

定义 $\phi(q) = \mathbf{L}^\top q$，把 $\mathcal{Q}$ 映到新的空间 $\mathcal{X}$。Theorem 3 说：在 $\mathcal{X}$ 里，原本由 $\mathbf{G}_{\text{lower}}$ 诱导的 anisotropic norm 变成标准 Euclidean norm。

$$
\hat{d}(q_1, q_2) = \|\phi(q_1) - \phi(q_2)\|_2
$$

类比：你有一张变形的地图，上面画着变形的椭圆圆圈（informed set）。Cholesky 变换就像把这张变形地图**贴回标准方格纸上**——所有椭圆变成正圆，所有变形距离变成直线距离。

### 5.2 Riemannian informed set 又是椭球了

在 $\mathcal{X}$ 里，informed set $X_{\hat{f}}^{\mathcal{X}} = \{x : \|x_{\text{start}} - x\|_2 + \|x - x_{\text{goal}}\|_2 < c_{\text{best}}\}$ 就是标准 PHS。回到原坐标系：

$$
X_{\hat{f}}^{\mathcal{R}} = \phi^{-1}(X_{\hat{f}}^{\mathcal{X}})
$$

由于 $\phi^{-1}$ 是 invertible linear map，**$X_{\hat{f}}^{\mathcal{R}}$ 在原坐标系是 ellipsoid**——沿 $\mathbf{G}_{\text{lower}}$ 各特征方向被 $1/\sqrt{\lambda_i}$ 缩放。

### 5.3 形状的直觉

设 $\mathbf{G}_{\text{lower}} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$，$\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_n)$：
- $\lambda_i$ 大 = 该方向 cost 大 = expensive direction
- $\lambda_i$ 小 = 该方向 cost 小 = inexpensive direction

Ellipsoid 主轴沿 $\mathbf{V}$ 的列方向，每条主轴长度被 $\phi^{-1}$ 缩放 $1/\sqrt{\lambda_i}$：
- inexpensive direction（$\lambda_i$ 小）→ $1/\sqrt{\lambda_i}$ 大 → ellipsoid **伸长**
- expensive direction（$\lambda_i$ 大）→ $1/\sqrt{\lambda_i}$ 小 → ellipsoid **收缩**

**所以 informed set 自动把采样密度压到低成本方向**。这就是 paper 的核心几何 insight。

### 5.4 体积公式

$$
\mu(X_{\hat{f}}^{\mathcal{R}}) = \frac{\mu_{\text{PHS}}(c_{\text{best}}, d_{\text{foci}})}{\sqrt{\det(\mathbf{G}_{\text{lower}})}}
$$

变量解释：$\mu$ 是 Lebesgue 测度（体积），$\mu_{\text{PHS}}(c, d)$ 是 PHS 的标准 closed-form 体积公式，$d_{\text{foci}} = \hat{d}(q_{\text{start}}, q_{\text{goal}})$ 是两焦点在 heuristic 下的距离，$\det(\mathbf{G}_{\text{lower}}) = \prod_{i=1}^n \lambda_i$ 是 matrix 行列式。

体积变小来自两个独立机制：

1. **分子变小**：matrix bound 让 $d_{\text{foci}}$ 更大，PHS 在 $\mathcal{X}$ 中更瘦。
2. **分母变大**：$\sqrt{\det(\mathbf{G}_{\text{lower}})}$ 比 scalar 的 $\lambda_{\min}^{n/2}$ 大得多——因为各方向 $\lambda_i \ge \lambda_{\min}$，多数严格大于。

**高维放大效应**：14 维 PR2 上，每方向 modest 紧 1.5 倍，乘起来 $\det$ 上 $1.5^{14} \approx 291\times$。这就是为啥 paper 在高 DoF 上优势指数级暴涨。

---

## 6. 怎么算 $\mathbf{G}_{\text{lower}}$：Algorithm 1 + 3

### 6.1 Algorithm 1: Loewner Meet（增量收紧）

输入：当前 $\mathbf{L}$（$\mathbf{G}_{\text{lower}}$ 的 Cholesky 因子）和新观测 $\mathbf{G}_{\text{new}}$。

核心步骤：
1. **Whitening**：$\mathbf{S} \leftarrow \mathbf{L}^{-1} \mathbf{G}_{\text{new}} \mathbf{L}^{-\top}$。把新观测变换到"当前下界是 identity"的坐标系。
2. **Eigendecomposition**：$\mathbf{S} = \mathbf{V}\text{diag}(\lambda_1, \dots, \lambda_n)\mathbf{V}^\top$。在白化系下，$\lambda_k$ 直接告诉你 $\mathbf{G}_{\text{new}}$ 沿各方向**相对** $\mathbf{G}_{\text{lower}}$ 多大。
   - $\lambda_k > 1$：该方向 $\mathbf{G}_{\text{new}}$ 已经 > 下界，不动；
   - $\lambda_k < 1$：该方向 $\mathbf{G}_{\text{new}}$ < 下界，**违反了**，必须收紧。
3. **Clamp**：$\tilde{\lambda}_k \leftarrow \min(\lambda_k, 1)$。大于 1 的不动（不能放松，否则 admissibility 失效），小于 1 的收紧。
4. **Reconstruct + Cholesky**：回到原坐标系，重新算 Cholesky。

类比：你在白板上画当前下界（一个椭圆），有人贴上来一个新观测（另一个椭圆）。如果新椭圆某些方向"凸出去"超过当前下界（说明该方向当前下界太宽），就把当前下界那些方向往里压。压到与所有已观测过的椭圆都"装得下"为止。

### 6.2 Algorithm 3: 主动找最坏点

问题：metric field $\mathbf{G}(q)$ 在整个 $\mathcal{Q}$ 上连续变化，不可能扫遍。怎么办？

作者用 **multi-start gradient descent + Armijo backtracking** 主动找"最违反当前下界"的 $q^*$：

$$
q^* \leftarrow \arg\min_q \lambda_{\min}(\mathbf{L}^{-1}\mathbf{G}(q)\mathbf{L}^{-\top})
$$

目标函数意义：在白化系下 $q$ 处的"最小方向比值"。要找它最小的 $q$，即"最危险点"——这里下界最可能被违反。

如果 $\lambda^* < 1 - \tau$（$\tau = 10^{-6}$），说明违反，调 Algorithm 1 收紧。重复直到所有 $q^*$ 都 $\lambda^* \ge 1 - \tau$。

类比：你在做 budget 估算，主动问"哪个月最穷？哪类支出最爆？"找到最坏 case 后调低你的 estimate，反复直到没人能再挑战你的 estimate。这和 **adversarial training / active learning** 同源。

---

## 7. Algorithm 2: Direct Sampling（核心算法）

```
1. x_start ← L^T q_start; x_goal ← L^T q_goal
2. x ← Uniform(X_f̂^X)        // 标准 PHS 直接采样
3. q ← L^{-T} x              // 映射回原坐标系
4. if q ∈ Q then return q
5. else reject and resample
```

**关键**：$\phi^{-1}: x \mapsto \mathbf{L}^{-\top} x$ 是 invertible linear map，Jacobian 是常数 $1/\sqrt{\det(\mathbf{G}_{\text{lower}})}$。常数 Jacobian **不改变均匀性**——$\mathcal{X}$ 中 uniform 在 $\mathcal{Q}$ 中映射后还是 uniform。

**唯一 rejection** 来自 line 4 的 boundary check（PHS 可能超出 $\mathcal{Q}$ 边界），这和标准 Euclidean PHS 一样。**informed set 几何本身不产生 rejection**，这是相对 [16, 17] MCMC / hierarchical rejection sampling 的根本优势。

---

## 8. 实验：九个组合

### 8.1 设置

- **Robots**: 6-DoF UR5, 7-DoF Franka, 14-DoF PR2（双臂）
- **Metrics**: diagonal weighted, kinetic energy (Pinocchio 算 mass matrix), pullback ($\mathbf{J}^\top\mathbf{J} + 0.1\mathbf{I}$)
- **Planners**: BIT*, AIT*, EIT*, G-RRT* (ε=0 和 ε=0.9), AORRTC
- **Heuristics**: zero, Euclidean, matrix bound (Loewner)
- **预算**: 120s, 100 trials, 报 median
- **Backend**: OMPL + VAMP collision checking

### 8.2 Heuristic tightness 实测（Section VI-A）

测 10,000 个 random 点对，看 ratio $\hat{d}/d$：

| Metric | Euclidean | Scalar | Matrix |
|---|---|---|---|
| Weighted (constant) | inadmissible 一些情况 | admissible but loose | **ratio = 1** (因 $\mathbf{G}_{\text{lower}} = \mathbf{G}$ 常数) |
| Kinetic energy | inadmissible (median > 1) | 超保守 | **3-4× tighter than scalar** |
| Pullback | inadmissible | admissible | **10-15% tighter** (regularizer 让 bound 接近 isotropic) |

### 8.3 主要发现

1. **Weighted & kinetic energy**: matrix bound 一直加速收敛，kinetic energy 上优势最大。
2. **Pullback**: Euclidean 把 UR5/Franka 卡死在 local minima。PR2 上 Euclidean 反而 OK——巧合，overestimate 方向碰巧指向解。
3. **GRRT* (ε=0.9) + matrix bound** 一致最快。即使 bound 接近 isotropic 时，PHS 直接采样让 GRRT* greedy biasing 仍 work。
4. **AORRTC** + matrix bound 也快，但 zero-heuristic AORRTC 也 competitive——AORRTC 的 cost-augmented search 对 heuristic 不敏感。

---

## 9. 我觉得最有意思的地方

### 9.1 模板化思维："换坐标系让难题变标准题"

这 paper 本质是模板套用：找一个 lower bound matrix → Cholesky → isometric embedding → 难的 Riemannian informed set 变成标准 PHS。这个"通过变换让问题归约到已知问题"的思路在数学里到处都是（Fourier 让微分方程变代数方程，wavelet 让信号稀疏化）。这里把它在 motion planning 的 informed sampling 上落地得很干净。

### 9.2 Loewner order 选得巧

为啥不选 majorization、element-wise 其他 matrix order？因为 Loewner order 直接对应"对所有方向 $v^\top \mathbf{A} v \le v^\top \mathbf{B} v$"，正好契合 admissibility 需要的"对所有曲线、所有点处不等式"。其他 order 要么太弱（majorization 只管 trace），要么没几何意义。

### 9.3 高维指数放大效应

14 维 PR2 上，每方向紧 1.5 倍 → 体积缩 $1.5^{14} \approx 291\times$。这是 paper 在高 DoF 上优势指数暴涨的根因。**determinant 是各方向特征值的乘积**——scalar bound 只用最小那个，matrix bound 用全部，在高维下差距指数放大。

### 9.4 Pullback metric 的反直觉现象

$\mathbf{G} = \mathbf{J}^\top \mathbf{J} + \lambda \mathbf{I}$ 里 $\lambda \mathbf{I}$ regularizer 把 singularity 处各方向都"撑起来"，让 $\mathbf{G}_{\text{lower}}$ 接近 isotropic，matrix bound 退化到 scalar bound。这揭示一个 tradeoff：**metric 本身越"anisotropic"，matrix bound 优势越大；metric 被 regularize 越平，优势越消失**。未来工作一个方向是"directional regularization"——只 regularize 真正奇异的方向。

### 9.5 Inadmissible heuristic 偶尔能"走运"

PR2 + pullback 上 Euclidean 反而 work，作者解释为"coincidental direction alignment"。这提醒：**admissibility 不是 efficiency 的唯一决定因素**——inadmissible 偶尔因 problem instance 几何走运。但作为通用方法，admissible + tight 才 robust。

### 9.6 Limitation 我自己的思考

- **Algorithm 3 local optima**: multi-start gradient descent 不保证找全局最坏点。若漏掉某 region 的最坏 case，admissibility 在该 region 失效，informed set 可能 exclude 真解。Appendix I 的 $\tau$-certify 只对 explored space 有效。
- **Online update**: 如果 metric field 在 planning 过程中变（dynamic obstacle 改 pullback），online 重算 $\mathbf{G}_{\text{lower}}$ 是否可行？paper 没讨论。
- **Pre-compute cost**: paper 没报 $\mathbf{G}_{\text{lower}}$ 计算 time，PR2 14 维 + 多次 mass matrix evaluation 可能不便宜。

---

## 10. 延伸思考：模板可迁移性

"matrix lower bound → Cholesky → isometric embedding → 几何降维" 模板可迁移到：

- **Belief space planning** [30, 31]: 现在 scalar bound on $\lambda_{\max}(\Sigma)$。可换 matrix Loewner upper bound on covariance + Cholesky 做 anisotropic belief informed sampling。
- **Robust control tubes** [32]: contraction metric 已用 Loewner lower bound，但是固定 trajectory tube 无 sampling。可结合 sampling-based planner。
- **STOMP/CHOMP** [33, 34]: trajectory space 上 fixed precision matrix 已经 Loewner-friendly 但非 lower bound。可让 STOMP 用 configuration-dependent metric + Loewner lower bound。

---

## 11. 关键 references

- 论文前作 (Kyaw & Kelly 2026 [8]): https://arxiv.org/abs/2602.00992
- Informed RRT* (Gammell et al. 2018 [4]): https://arxiv.org/abs/1706.06391
- BIT* [3]: https://arxiv.org/abs/1905.05611
- AIT*/EIT* [13]: https://arxiv.org/abs/2101.06148
- G-RRT* [14]: https://arxiv.org/abs/2405.03411
- AORRTC [41]: https://arxiv.org/abs/2409.14500
- Loewner order: https://en.wikipedia.org/wiki/Loewner_order
- Bhatia, *Positive Definite Matrices*: https://press.princeton.edu/books/hardcover/9780691129187/positive-definite-matrices
- OMPL: https://ompl.kavrakilab.org/
- MotionBenchMaker: https://motionbenchmaker.com/
- Pinocchio: https://github.com/stack-of-tasks/pinocchio
- VAMP: https://arxiv.org/abs/2402.14617
- GCS on Riemannian manifolds [27]: https://arxiv.org/abs/2410.18703
- Hierarchical Rejection Sampling [17]: https://ieeexplore.ieee.org/document/7487281
- MCMC Informed Sampling [16]: https://arxiv.org/abs/1802.07147

---

## 12. 一句话压缩

**找保底矩阵保住下界，Cholesky 把歪空间拉直，椭球又变椭球，老算法直接用——矩阵保底比标量保底多保留方向信息，高维下体积指数级缩小。**

---

# Direct Informed Sampling on Riemannian Manifolds via Loewner Order Lower Bounds 深度解读

你好 Andrej，这篇 paper 处理的是一个我特别喜欢的"几何结构 vs. 算法效率"的问题——**当 cost 是 configuration-dependent Riemannian metric 时，怎么做 informed sampling 才不浪费几何信息？** 作者 Phone Thiha Kyaw 和 Jonathan Kelly (UToronto ASRL) 的核心 trick 是：把"找 admissible heuristic"这个问题，转译成 SPD matrix cone 上的 Loewner order lower bound 问题，再用 Cholesky 当 isometric embedding，让 Riemannian informed set 在另一坐标系里退化回标准的 prolate hyperspheroid (PHS)。下面我按你要求的密度把所有 intuition 全部拆开。

---

## 1. 问题动机：Euclidean informed sampling 为什么"破碎"？

### 1.1 Informed sampling 的标准 pipeline

回顾 Gammell 等人 2018 的 Informed RRT* / BIT* 的核心 insight：给定当前最优解 cost $c_{\text{best}}$，**只有满足 admissible heuristic 三角不等式严格小于 $c_{\text{best}}$ 的 configurations 才可能改进解**：

$$
X_{\hat{f}} = \Bigl\{ q \in \mathcal{Q} : \hat{d}(q_{\text{start}}, q) + \hat{d}(q, q_{\text{goal}}) < c_{\text{best}} \Bigr\}
$$

这里 $\hat{d}(\cdot,\cdot) \le d(\cdot,\cdot)$ 是 admissible heuristic（cost-to-go 的下界）。在 Euclidean metric $\mathbf{G} = \mathbf{I}$ 下，取 $\hat{d}(q_x, q_y) = \|q_x - q_y\|_2$，这个集合就是**椭球面：以 $q_{\text{start}}, q_{\text{goal}}$ 为焦点的 prolate hyperspheroid (PHS)**。

PHS 的直接采样算法是经典活：把 $\mathbb{R}^n$ 中单位球里均匀样本通过一个解析的 affine map（旋转 + 沿焦点轴方向拉伸）就能映射到 PHS 上，无 rejection（除了边界外）。论文 [4] 给了 closed-form。

### 1.2 为什么 Riemannian metric 把这件事搞坏

机器人里 cost 经常不是 Euclidean 的。论文里考察的三个 metric：

1. **Diagonal weighted metric**：手动给小位移关节权重 100，大位移给 1，避免无意义的小关节乱动；
2. **Kinetic energy metric**：$\mathbf{G}(q) = \mathbf{M}(q)$，机械臂 mass matrix。这让 planner 倾向走低惯量姿态；
3. **Pullback metric**：通过 manipulator Jacobian $\mathbf{J}(q)$ 把 task-space metric 拉回 joint space，$\mathbf{G}(q) = \mathbf{J}^\top \mathbf{J} + \lambda \mathbf{I}$，$\lambda$ 防止 singularity 处退化。

这三个 metric 都是 configuration-dependent 的 SPD matrix field $\mathbf{G}: \mathcal{Q} \to \mathbb{S}^n_{++}$。Riemannian 弧长：

$$
L(\pi) = \int_0^1 \sqrt{\dot{\pi}(t)^\top \mathbf{G}(\pi(t)) \dot{\pi}(t)} \, dt, \quad \pi:[0,1]\to\mathcal{Q}
$$

geodesic distance $d(q_x, q_y) = \inf_\pi L(\pi)$。

**问题 1: Euclidean heuristic 是 inadmissible 的。** 论文 Section VI-A 实证：在高 DoF 上 kinetic energy metric 下，Euclidean distance $\|q_x - q_y\|_2$ 对 99% 的 configuration pairs **高估**了 geodesic distance。这是为啥？因为 mass matrix 在某些方向上"很小"（低惯量方向），实际行进 cost 比 Euclidean 还低；用 Euclidean 当 heuristic 会 overestimate，导致 informed set 排除了真正能改进解的点，planner 被骗。论文 Section VI-B 在 UR5/Franka + pullback metric 上直接展示这个 failure mode：Euclidean heuristic 卡在 local minima，120 秒预算内不收敛。

**问题 2: 即使有 admissible scalar bound，它太保守。** 一个 naive fix [12, 32]：取 metric 的全局最小特征值 $\lambda_{\min} = \inf_{q \in \mathcal{Q}} \lambda_{\min}(\mathbf{G}(q))$，把 Euclidean distance 缩放 $\sqrt{\lambda_{\min}}$：

$$
\hat{d}_\lambda(q_x, q_y) = \sqrt{\lambda_{\min}} \|q_x - q_y\|_2
$$

这对应 $\mathbf{G}_\lambda = \lambda_{\min} \mathbf{I}$ 的 isotropic 下界。它 admissible 但**丢失了 metric 的 directional structure**——所有方向都被同一个标量 uniform scale。结果是 informed set 仍然是 isotropic PHS，**对 metric 各方向的不均匀性视而不见**：低惯量方向本该被多采样，高惯量方向本该被压制，但 scalar bound 一视同仁。

---

## 2. Key Insight：Loewner Order 给的 Matrix-Valued Bound

### 2.1 Loewner order 复习

**Definition 1 (Loewner order).** 对 $\mathbf{A}, \mathbf{B} \in \mathbb{S}^n$，$\mathbf{A} \preceq \mathbf{B}$ 当且仅当 $\mathbf{B} - \mathbf{A}$ 是 PSD，即 $\forall v \in \mathbb{R}^n, v^\top \mathbf{A} v \le v^\top \mathbf{B} v$。

这是 SPD cone 上的 **partial order**（不是 total order）。所以两个一般 SPD matrix 之间可能不可比较——这点后面会反复回响。

关于 meet（最大下界）：任意有限集合 $\{\mathbf{A}_1, \dots, \mathbf{A}_k\} \subset \mathbb{S}^n_{++}$ 在 Loewner order 下有 maximal lower bound $\bigwedge_{i=1}^k \mathbf{A}_i$（虽然不一定唯一最"大"，但是 maximal，没有更"大"的下界），但没有 closed form，需要迭代算。

> Intuition: 你可以把 Loewner order 想成"对所有方向 $v$ 同时做 $v^\top (\cdot) v$ 的不等式"。$\mathbf{G}_{\text{lower}} \preceq \mathbf{G}(q) \,\forall q$ 等价于"在每一个方向 $v$ 上，$\mathbf{G}_{\text{lower}}$ 给出的二次型都 ≤ 该 $q$ 处 $\mathbf{G}(q)$ 给的二次型"。这比单看 $\lambda_{\min}$ 强多了：$\lambda_{\min}$ 只保证"最差方向"，Loewner 同时保证"所有方向"。参考: [Loewner order on Wikipedia](https://en.wikipedia.org/wiki/Loewner_order), Bhatia 的 *Positive Definite Matrices*。

### 2.2 Matrix Lower Bound Heuristic

**Definition 2.** $\mathbf{G}_{\text{lower}} \in \mathbb{S}^n_{++}$ 是 $\mathbf{G}$ 的 Loewner lower bound 当 $\mathbf{G}_{\text{lower}} \preceq \mathbf{G}(q), \forall q \in \mathcal{Q}$。

**Definition 4 (Matrix distance lower bound).** 设 $\mathbf{G}_{\text{lower}} = \mathbf{L}\mathbf{L}^\top$ 是其 Cholesky 分解（$\mathbf{L}$ 是下三角，$\mathbf{L}^\top$ 是上三角），则

$$
\hat{d}(q_x, q_y) = \|q_x - q_y\|_{\mathbf{G}_{\text{lower}}} = \|\mathbf{L}^\top(q_x - q_y)\|_2
$$

这里：
- $\|q_x - q_y\|_{\mathbf{G}_{\text{lower}}}$ 是由 SPD matrix $\mathbf{G}_{\text{lower}}$ 诱导的 Mahalanobis-like 范数，定义为 $\sqrt{(q_x-q_y)^\top \mathbf{G}_{\text{lower}} (q_x-q_y)}$。
- 上标 $\top$ 是 transpose，下三角 $\mathbf{L}$ 来自 Cholesky: $\mathbf{G}_{\text{lower}} = \mathbf{L}\mathbf{L}^\top$ 唯一（在 $\mathbf{L}$ 对角元素 > 0 时）。
- $\mathbf{L}^\top(q_x - q_y)$ 是一个 linear map，把 difference vector "拉直" 到一个内积为 $\mathbf{I}$ 的空间。

**Theorem 1 (Tightness).** $\hat{d}_\lambda \le \hat{d}$，等号当且仅当 $q_x - q_y$ 是 $\mathbf{G}_{\text{lower}}$ 对应特征值 $\lambda_{\min}$ 的特征向量。

证明很简洁：因为 $\lambda_{\min}\mathbf{I} \preceq \mathbf{G}_{\text{lower}}$，所以 $v^\top \mathbf{G}_{\text{lower}} v \ge v^\top(\lambda_{\min}\mathbf{I})v = \lambda_{\min}\|v\|_2^2$，开方即得。等号条件来自二次型相等的充要条件是被 same eigenvector with same eigenvalue 命中。

> Intuition: scalar bound 是 matrix bound 的 **特例**（取 $\mathbf{G}_{\text{lower}} = \lambda_{\min}\mathbf{I}$）。Matrix bound 在每个方向上单独做下界，scalar bound 强行把所有方向压平到最小那个。等号只在沿着 $\mathbf{G}_{\text{lower}}$ 最"窄"的那个特征方向行进时成立。

**Theorem 2 (Admissibility).** 如果 $\mathbf{G}_{\text{lower}} \preceq \mathbf{G}(q)$ 对所有 $q$ 成立，则 $\hat{d}(q_x, q_y) \le d(q_x, q_y)$。

证明核心是逐段曲线的不等式链：
$$
L(\pi) = \int_0^1 \sqrt{\dot{\pi}^\top \mathbf{G}(\pi)\dot{\pi}} \, dt \;\ge\; \int_0^1 \|\mathbf{L}^\top \dot{\pi}\|_2 \, dt \;\ge\; \|\mathbf{L}^\top(q_y - q_x)\|_2 = \hat{d}(q_x, q_y)
$$

第一步用 Loewner order 让 integrand pointwise 不等式（对每个 $t$ 都成立）；第二步用三角不等式 + 积分；最后取 inf over all $\pi$ 即得。

> Intuition: 这是经典的"松弛被积函数 ⇒ 松弛弧长"。$\mathbf{G}_{\text{lower}} \preceq \mathbf{G}$ 让 Riemannian norm 处处被 Euclidean norm（在 Cholesky-rotated 坐标下）控制住，于是直线的 Euclidean 长度（在 rotated 坐标下）≤ 任意曲线的 Riemannian 长度。这是把 Riemannian geodesic 距离从下面"夹"住的根因。

**Corollary 1.** 当 $\mathbf{G}_{\text{lower}} = \mathbf{I}$ 时退化成 Euclidean distance。但是——**反过来不成立**：用 Euclidean distance 当 heuristic 只有在 $\mathbf{I} \preceq \mathbf{G}(q)$ 处处成立（即 $\lambda_{\min}(\mathbf{G}(q)) \ge 1$ 处处成立）时才 admissible。这解释了为什么 kinetic energy metric 下 Euclidean 高估：mass matrix 的某些方向 < 1（mass 单位）。

---

## 3. 怎么算这个 $\mathbf{G}_{\text{lower}}$？Algorithm 1: Loewner Meet

### 3.1 算法核心：白化 + 特征值 clamp

输入：当前 $\mathbf{L}$（$\mathbf{G}_{\text{lower}}$ 的 Cholesky 因子）和新观测 $\mathbf{G}_{\text{new}}$。

```
1. S ← L^{-1} G_new (L^T)^{-1}      // 把 G_new 变换到 G_lower 是 identity 的坐标系
2. S = V diag(λ_1,...,λ_n) V^T     // eigendecomposition of S
3. if min_k λ_k < 1 then
4.    λ̃_k ← min(λ_k, 1) for each k
5.    Λ̃ ← diag(λ̃_1, ..., λ̃_n)
6.    G_lower ← L V Λ̃ V^T L^T       // 回到原坐标系
7.    L ← Cholesky(G_lower)
8. return L
```

**逐行 intuition**：

- **Line 1**: 这一步叫 **whitening**。$\mathbf{G}_{\text{lower}} = \mathbf{L}\mathbf{L}^\top$，那么 $\mathbf{L}^{-1} \mathbf{G}_{\text{lower}} \mathbf{L}^{-\top} = \mathbf{I}$。把新观测 $\mathbf{G}_{\text{new}}$ 用同样线性变换映射到这个"白化"坐标系，得到 $\mathbf{S}$。
  
- **Line 2**: 在白化系下，$\mathbf{S}$ 的特征值直接告诉你 $\mathbf{G}_{\text{new}}$ 沿每个主方向**相对于** $\mathbf{G}_{\text{lower}}$ 是多大。
  - $\lambda_k > 1$：该方向上 $\mathbf{G}_{\text{new}} > \mathbf{G}_{\text{lower}}$，**下界依然成立**，无需更新；
  - $\lambda_k < 1$：该方向上 $\mathbf{G}_{\text{new}} < \mathbf{G}_{\text{lower}}$，**下界被违反**，必须收紧到 $\lambda_k$；
  - $\lambda_k = 1$：刚好相切，不动。

- **Line 4**: `min(λ_k, 1)` 起到"两选一"作用——大于 1 的不动，小于 1 的收紧。注意：这里要 clamp 大于 1 的不要 expand，因为我们只能收紧下界，不能放松（放松会破坏 admissibility）。

- **Line 6**: 回到原坐标系。注意 $\mathbf{L} \mathbf{V} \tilde{\mathbf{\Lambda}} \mathbf{V}^\top \mathbf{L}^\top$ 是 $\mathbf{L} (\cdot) \mathbf{L}^\top$ 的形式，它正是把白化系下"被收紧过的 identity"映射回去。

- **Line 7**: 重新 Cholesky，因为 $\mathbf{G}_{\text{lower}}$ 变了。

> Intuition: 这是个"半序空间上的 iterative tightening" 算法。每次新观测若违反当前下界，就在被违反的方向上把下界降下来。最终收敛到的 matrix 是所有观测点的 Loewner meet。可惜没有 closed form，只能 incremental 算。

**Remark 2 强调** Algorithm 1 是 generic subroutine，可以接受任意 metric evaluation 序列 $\mathbf{G}(q_1), \dots, \mathbf{G}(q_N)$，从 $\mathbf{G}_{\text{lower}} = \mathbf{G}(q_1)$ 开始迭代，得到所有观测的 valid 下界。设计选择只在 **采样点的选择策略** 上：uniform batch、optimization-based、space-filling 等都可以。

### 3.2 Appendix I: Algorithm 3 把采样点选择做成 optimization

论文用 **multi-start gradient descent + Armijo backtracking** 主动找"最违反当前下界"的 $q^*$：

```
1. L ← Cholesky(G(q_init))     // q_init 设为 joint limit 中点
2. repeat
3.    q* ← argmin_q λ_min(L^{-1} G(q) L^{-T})
4.    λ* ← λ_min(L^{-1} G(q*) L^{-T})
5.    if λ* < 1 - τ then
6.       L ← LoewnerMeet(L, G(q*))    // 调 Algorithm 1
7. until λ* ≥ 1 - τ
8. return G_lower = LL^T
```

- **Line 3**: 目标函数 $\lambda_{\min}(\mathbf{L}^{-1}\mathbf{G}(q)\mathbf{L}^{-\top})$ 是当前下界在白化系下 $q$ 处的"最小方向比值"。我们要找它最小的 $q$，即"最危险"点。
- **Line 5**: $\tau$ 是 numerical tolerance，论文取 $10^{-6}$。$\lambda^* < 1 - \tau$ 表示在白化系下 $q^*$ 处有方向严格小于 1，违反下界。
- **Line 7**: 当所有 $q^*$ 都满足 $\lambda^* \ge 1 - \tau$，**说明在所有 explored 区域下界都成立**。注意这只能 certify explored space，理论上 full $\mathcal{Q}$ 上是否真的成立依赖 exploration 是否充分。

> Intuition: 这是 active learning / adversarial training 的味道。用一个 minimizer 找"最坏 case"，然后用它 tighten 模型。和 GAN、SVM margin maximization、adversarial robustness 在精神上同源。

---

## 4. Riemannian Informed Set 几何结构

### 4.1 Isometric Embedding (Theorem 3)

定义 $\phi: \mathcal{Q} \to \mathcal{X}$，$\phi(q) = \mathbf{L}^\top q$。

**Theorem 3**: $\phi$ 是从 $(\mathcal{Q}, \mathbf{G}_{\text{lower}})$ 到 $(\mathcal{X}, \mathbf{I})$ 的 **linear isometry**：

$$
\hat{d}(q_1, q_2) = \|\phi(q_1) - \phi(q_2)\|_2
$$

证明：
$$
\|\phi(q_1) - \phi(q_2)\|_2 = \|\mathbf{L}^\top(q_1 - q_2)\|_2 = \sqrt{(q_1-q_2)^\top \mathbf{L} \mathbf{L}^\top (q_1-q_2)} = \sqrt{(q_1-q_2)^\top \mathbf{G}_{\text{lower}}(q_1-q_2)} = \hat{d}(q_1, q_2)
$$

> Intuition: $\mathbf{L}^\top$ 把原空间"拉直"。原本由 $\mathbf{G}_{\text{lower}}$ 诱导的 anisotropic norm，经过 $\mathbf{L}^\top$ 线性变换后变成标准 Euclidean norm。这是因为 Cholesky 正是 Gram matrix 的"开方"。你可以想 $\mathbf{G}_{\text{lower}}$ 为内积矩阵，$\mathbf{L}^\top$ 就是把这个内积"还原"成 dot product 的坐标变换。

逆映射 $\phi^{-1}(x) = \mathbf{L}^{-\top} x$，其中 $\mathbf{L}^{-\top} = (\mathbf{L}^{-1})^\top = (\mathbf{L}^\top)^{-1}$。

### 4.2 Riemannian Informed Set 是 Ellipsoid (Corollary 2)

Riemannian informed set：

$$
X_{\hat{f}}^{\mathcal{R}} = \Bigl\{ q \in \mathcal{Q} : \hat{d}(q_{\text{start}}, q) + \hat{d}(q, q_{\text{goal}}) < c_{\text{best}} \Bigr\}
$$

应用 Theorem 3，每项 $\hat{d}$ 可以替换成 $\|\cdot\|_2$ 在 $\mathcal{X}$ 上的等价形式：

$$
X_{\hat{f}}^{\mathcal{R}} = \phi^{-1}\bigl(X_{\hat{f}}^{\mathcal{X}}\bigr)
$$

其中 $X_{\hat{f}}^{\mathcal{X}} = \{x \in \mathcal{X}: \|x_{\text{start}} - x\|_2 + \|x - x_{\text{goal}}\|_2 < c_{\text{best}}\}$ 是标准 PHS，foci 是 $x_{\text{start}} = \phi(q_{\text{start}})$ 和 $x_{\text{goal}} = \phi(q_{\text{goal}})$。

由于 $\phi^{-1}$ 是 invertible linear map，**$X_{\hat{f}}^{\mathcal{R}}$ 在原坐标系下是 ellipsoid**（线性变换保椭圆性）。具体地，对 $\mathbf{G}_{\text{lower}}$ 做 eigendecomposition $\mathbf{G}_{\text{lower}} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$，$\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_n)$：

- $\mathbf{V}$ 旋转主轴对齐到 $\mathbf{G}_{\text{lower}}$ 的特征方向；
- $\phi$ 沿每个主轴 scale $\sqrt{\lambda_i}$；
- $\phi^{-1}$ 反向，沿每个主轴 scale $1/\sqrt{\lambda_i}$。

**所以 $X_{\hat{f}}^{\mathcal{R}}$ 的每条主轴比 PHS 的对应主轴短 $1/\sqrt{\lambda_i}$**：低 $\lambda_i$ 方向（即"原本 metric 小"的方向）反而**伸长**——咦不对，让我再想想。$\phi$ 是 scale $\sqrt{\lambda}$，那 $\phi^{-1}$ 是 scale $1/\sqrt{\lambda}$。PHS 在 $\mathcal{X}$ 中的尺寸被 $\phi^{-1}$ 乘以 $1/\sqrt{\lambda_i}$ 沿各方向。低 $\lambda_i$ → $1/\sqrt{\lambda_i}$ 大 → 沿该方向 ellipsoid 伸得**更长**。

但 paper Section V-C 写："each principal axis of $X_{\hat{f}}^{\mathcal{R}}$ is scaled by $1/\sqrt{\lambda_i}$ relative to those of the PHS, that is, the ellipsoid extends further along directions where motion is inexpensive and contracts where it is costly."

"inexpensive direction" = $\lambda_i(\mathbf{G}_{\text{lower}})$ 大的方向（因为 cost = $v^\top \mathbf{G} v$，大 $\lambda$ 意味着该方向 cost 大）—— 等等，这又不对。Let me think again.

Cost 在某方向 $v$ 的"局部 per-unit-length cost rate" 是 $\sqrt{v^\top \mathbf{G} v}$ 当 $\|v\|=1$，即 $\sqrt{\lambda_i}$ if $v$ 是特征向量。所以 $\lambda_i$ 大 = cost 大 = expensive direction。$\lambda_i$ 小 = cost 小 = inexpensive direction。

那 $\phi$ 的 scale 是 $\sqrt{\lambda_i}$，$\phi^{-1}$ 的 scale 是 $1/\sqrt{\lambda_i}$。所以 PHS 沿 $v_i$ 方向的尺寸被 $\phi^{-1}$ 乘以 $1/\sqrt{\lambda_i}$。

- $\lambda_i$ 小（inexpensive）：$1/\sqrt{\lambda_i}$ 大 → ellipsoid 沿该方向延伸**长**。论文说 "extends further along directions where motion is inexpensive"。✓
- $\lambda_i$ 大（expensive）：$1/\sqrt{\lambda_i}$ 小 → ellipsoid 沿该方向收缩**短**。✓

OK 所以 paper Section V-C 是对的，我刚才差点绕晕。**这个几何上的"反向缩放"是因为我们在用 lower bound 做 heuristic**：bound 越紧（即 $\lambda_i$ 在 inexpensive 方向上越接近真实 $\mathbf{G}$ 的最小值），informed set 在该方向就越"瘦"。但同时，inexpensive 方向的 metric 小，意味着该方向真实距离短，我们想多采样——这个"想多采样"反映在 PHS 在该方向延伸更长（在 $\mathcal{X}$ 中），从而 $\phi^{-1}$ 把这个延伸带回到 $\mathcal{Q}$ 中的 ellipsoid 延伸。

嗯，实际机制是：在 $\mathcal{X}$ 中，$x_{\text{start}}, x_{\text{goal}}$ 距离 $d_{\text{foci}} = \hat{d}(q_{\text{start}}, q_{\text{goal}})$ 大 → PHS 在 focus 轴方向**短轴变小**（PHS 沿 foci 连线的轴长 = $c_{\text{best}}/2$，沿垂直方向轴长 = $\sqrt{c_{\text{best}}^2 - d_{\text{foci}}^2}/2$）。所以 foci 之间距离越远，PHS 越瘦。再加上 $\phi^{-1}$ 各向异性 scale 后，$\mathcal{Q}$ 中的 ellipsoid 在 inexpensive 方向（$\lambda_i$ 小 → $1/\sqrt{\lambda_i}$ 大）延伸长，在 expensive 方向收缩。

> Intuition: 这套机制让 informed set **自适应地**把采样密度压到低成本方向。Scalar bound 把所有方向等价对待，丢了这层信息；matrix bound 保留了它，并通过 ellipsoid shape 直接表达。

### 4.3 Volume 公式（公式 8）

$$
\mu\bigl(X_{\hat{f}}^{\mathcal{R}}\bigr) = \frac{\mu_{\text{PHS}}(c_{\text{best}}, d_{\text{foci}})}{\sqrt{\det(\mathbf{G}_{\text{lower}})}}
$$

这里：
- $\mu$ 是 $\mathbb{R}^n$ 上的 Lebesgue measure（体积）。
- $\mu_{\text{PHS}}(c, d)$ 是 $n$ 维 PHS 在 transverse diameter $c$、focal distance $d$ 下的 closed-form 体积（来自 Gammell 2018 [4]）。
- $d_{\text{foci}} = \hat{d}(q_{\text{start}}, q_{\text{goal}})$ 是两焦点在 heuristic 下的距离。
- $\det(\mathbf{G}_{\text{lower}})$ 是 lower bound 矩阵的行列式。

**关键观察**：体积变小有两个独立机制：

1. **分子变小**：matrix bound 的 $\hat{d} \ge \hat{d}_\lambda$，所以 $d_{\text{foci}}$ 更大，PHS 更瘦（foci 越远，PHS 沿垂直 foci 轴越短）。
2. **分母变大**：$\sqrt{\det(\mathbf{G}_{\text{lower}})} = \sqrt{\prod_{i=1}^n \lambda_i(\mathbf{G}_{\text{lower}})}$。Scalar bound 时 $\det(\mathbf{G}_\lambda) = \lambda_{\min}^n$，但 matrix bound 的 $\det$ 通常远大于 $\lambda_{\min}^n$，因为各方向 $\lambda_i \ge \lambda_{\min}$，多数严格大于。

> Intuition: 在高维下，$\det$ 是各方向 $\lambda_i$ 的**乘积**，每方向即使 modest 紧一点（比如 $1.5\times$），$n=14$ 维 PR2 时乘起来就是 $1.5^{14} \approx 291\times$ 体积缩减。这是 paper Section VI-A 报告 "matrix bound 3-4× tighter" 在高维下被指数放大的根本原因。

---

## 5. Algorithm 2: Direct Sampling

```
1. x_start ← L^T q_start; x_goal ← L^T q_goal
2. x ← Uniform(X_f̂^X)     // 标准 PHS 直接采样
3. q ← L^{-T} x
4. if q ∈ Q then return q
5. else reject and resample
```

**为什么这是 uniform on $X_{\hat{f}}^{\mathcal{R}}$**？

$\phi^{-1}: x \mapsto \mathbf{L}^{-\top} x$ 是 invertible linear map，Jacobian 是 $\mathbf{L}^{-\top}$，Jacobian 行列式绝对值是 $|\det(\mathbf{L}^{-\top})| = 1/\sqrt{\det(\mathbf{G}_{\text{lower}})}$（常数）。

对 uniform density 而言，常数 Jacobian **不改变均匀性**——只是把 density 值乘以常数。所以 $\mathcal{X}$ 中的 uniform 在 $\mathcal{Q}$ 中映射后还是 uniform（在 $X_{\hat{f}}^{\mathcal{R}}$ 上）。

**唯一 rejection 来源**：line 4 的 $q \in \mathcal{Q}$ 检查。这和 Euclidean PHS 采样面临的"PHS 可能超出 $\mathcal{Q}$ 边界"完全一样——这不是 informed set 的几何 rejection，而是 domain boundary rejection。**没有 informed set 本身几何带来的 rejection**，这是相对 [16, 17] 那种 MCMC / hierarchical rejection sampling 的根本优势。

> Intuition: paper Section V-B 关键卖点 "rejection-free"。Kinodynamic 那边因为 informed set 形状怪，必须用 rejection 或 MCMC。这里通过 matrix lower bound 把 informed set 形状"骗回"了 PHS，直接套用 Euclidean 直接采样算法 [4]。参考 Gammell 的 informed sampling 原始实现：https://arxiv.org/abs/1706.06391。

---

## 6. 实验：九种组合全跑

### 6.1 实验设置

- **Robots**: 6-DoF UR5, 7-DoF Franka, 14-DoF PR2 (双臂) — 用 MotionBenchMaker [40], https://motionbenchmaker.com/
- **Metrics**: diagonal weighted, kinetic energy (用 Pinocchio [45] 算 mass matrix, https://github.com/stack-of-tasks/pinocchio), pullback (Jacobian $\mathbf{J}^\top \mathbf{J} + 0.1 \mathbf{I}$)
- **Planners**: BIT* [3], AIT* [13], EIT* [13], G-RRT* [14] (ε=0 和 ε=0.9), AORRTC [41]
- **Heuristics 对比**: zero, Euclidean, matrix bound (Loewner)
- **Backend**: OMPL [43] + VAMP collision checking [44] (https://arxiv.org/abs/2402.14617)
- **预算**: 120s, 100 trials, 报 median

### 6.2 Heuristic Quality (Section VI-A)

对每对 random configuration pair 算 ratio $\hat{d}/d$（$d$ 用 midpoint approximation 数值估计，[8]）：

- **ratio = 1**: perfect heuristic
- **ratio < 1**: admissible (lower bound)
- **ratio > 1**: inadmissible (overestimate)

**结果**：
| Metric | Euclidean | Scalar bound | Matrix bound |
|---|---|---|---|
| Weighted (constant) | inadmissible 在某些情况 | admissible but loose | **ratio = 1**（因为 $\mathbf{G}_{\text{lower}} = \mathbf{G}$ 常数，trivially 紧） |
| Kinetic energy | inadmissible (median > 1, 99% 高估) | admissible but 超保守 | **3-4× tighter than scalar** |
| Pullback | inadmissible | admissible | **10-15% tighter** (因 $\lambda$-regularization 使 $\mathbf{G}_{\text{lower}}$ 接近 isotropic) |

> Intuition: 三个 metric 表现差距很大，根源在 metric 本身的"anisotropy 程度"。Kinetic energy metric 在高 DoF 下 anisotropy 极强（各关节惯量差异大），matrix bound 优势最大。Pullback metric 由于 $\lambda \mathbf{I}$ regularizer 把 singularity 处各方向都"撑"起来，使 $\mathbf{G}_{\text{lower}}$ 几乎 isotropic，matrix bound 退化接近 scalar bound。

### 6.3 Manipulation Problems (Section VI-B)

九个组合（3 robots × 3 metrics），每个跑 5 planners × 3 heuristics + GRRT* greedy 变种。

**主要发现**：
1. **Weighted & kinetic energy**: matrix bound 一直加速收敛，kinetic energy 上优势最大（呼应 3-4× tighter）。
2. **Pullback**: Euclidean heuristic 把 UR5/Franka 卡死在 local minima（120s 不收敛）。PR2 上 Euclidean 反而 OK——巧合，因为 overestimate 方向碰巧指向解。Matrix bound 在 pullback 上优势 modest（10-15% 紧一点）。
3. **GRRT* (ε=0.9) + matrix bound** 一致最快。即使 bound 接近 isotropic 时，PHS-shape 的直接采样让 GRRT* 的 greedy biasing 仍然 work。
4. **AORRTC** + matrix bound 也比 zero/Euclidean 快，但 zero-heuristic AORRTC 也 competitive——因为 AORRTC 的 cost-augmented search 对 heuristic 不那么敏感。

> Intuition: paper Section VI-B 最有意思的"反直觉"结果：在 pullback + PR2 上 Euclidean 反而 work，作者解释为"coincidental direction alignment"。这提醒我们 **admissibility 不是 efficiency 的唯一决定因素**——inadmissible heuristic 偶尔能因 problem instance 的 specific geometry 走运。但作为通用方法，admissible + tight 才是 robust 的选择。

---

## 7. 我的几点延伸思考 (build intuition further)

### 7.1 为什么 Loewner order 而不是其他 matrix order？

Loewner order 是 SPD cone 上的 natural partial order，对应"对所有方向 $v$ 同时 $v^\top \mathbf{A} v \le v^\top \mathbf{B} v$"。它正好契合 admissibility 需要的"对所有曲线、所有点处"不等式。其他可能的选择：

- **PSD-cone 内部 order**：本质就是 Loewner。
- **Majorization order on eigenvalues**：$\lambda(\mathbf{A}) \prec \lambda(\mathbf{B})$ 是 weak majorization，比 Loewner 弱。它保证 trace 上的不等式，但不保证方向性。
- **Element-wise order**：对 SPD matrix element-wise 比较，意义不大且不保 positivity。

Loewner 是最"tight"的、保持几何意义的偏序，论文选择是 natural。

### 7.2 这套思路对其他 lower-bound 问题是否可迁移？

这个 "matrix lower bound → Cholesky → isometric embedding → 几何降维" 模板其实很 general。任何"找一个 constant SPD matrix 在某偏序下 pointwise dominate 一个 matrix field"的问题都可以套：

- **Belief space planning** [30, 31]：用 $\lambda_{\max}(\Sigma)$ 当 uncertainty bound。能不能换成 matrix Loewner upper bound on covariance？论文 [30, 31] 用 scalar 是为了让 sampling tractable；如果用 Loewner upper bound + Cholesky，可能能做"anisotropic belief informed sampling"。
- **Robust control tubes** [32]：contraction metric 已经用 Loewner lower bound 做 robustness tube。但 [32] 是固定 trajectory tube，没有 sampling。这里把 Loewner bound 用到 sampling-based planner 上是新的。
- **STOMP/CHOMP** [33, 34]：trajectory space 上的 fixed precision matrix 是 configuration-independent 的，已经是 Loewner-friendly 但不是 lower bound。能不能让 STOMP 用 configuration-dependent metric + Loewner lower bound？

### 7.3 局限性和未来工作

论文自己点出几个：

1. **Heavily regularized metric** 上 matrix bound 退化为 scalar bound（pullback metric 的现象）。这是因为 regularizer $\lambda \mathbf{I}$ 把各方向"扁平化"，$\mathbf{G}_{\text{lower}}$ 各 $\lambda_i$ 接近相等。Open question: 能否做"directional regularization"——只 regularize 真正奇异的方向？
2. **Configuration-dependent cost without explicit metric tensor**：比如 collision risk cost、manipulability index 等，没有显式 SPD matrix 形式。能不能 locally fit 一个 SPD matrix field？
3. **Bound computation cost**：Algorithm 3 是 offline precompute，但若 metric field 在 planning 过程中 update（例如 dynamic obstacle 改变 pullback），online 重新算 $\mathbf{G}_{\text{lower}}$ 是否可行？

### 7.4 与别的 Riemannian planning 工作的关系

- **Kyaw & Kelly 2026 [8]** (https://arxiv.org/abs/2602.00992)：作者自己的前作，给出 midpoint-based geodesic approximation。本 paper 用 [8] 的 midpoint approximation 算 $d$ 来 validate heuristic ratio。
- **Cohn et al. 2025 [27]** (https://arxiv.org/abs/2410.18703)：Graphs of Convex Sets 推广到 Riemannian manifolds，用 convex relaxation，但不给 informed sampling heuristic。
- **Lukyanenko & Soudbakhsh 2023 [28]**：RRT*/PRM* 在 non-Euclidean metric 下 asymptotic optimality，但不解决 informed sampling。

### 7.5 这套方法的"潜在风险"

- **Pre-compute cost 在 high-DoF**: PR2 14 维，mass matrix evaluation 不便宜，Algorithm 3 多次调用 $\mathbf{G}(q)$ 才收敛。论文没报告 $\mathbf{G}_{\text{lower}}$ 计算 time。
- **Local optima in Algorithm 3**: multi-start gradient descent 不保证找到全局最坏 case，可能 underestimate 真正 meet，导致 admissibility 在 unexplored region 失效。Appendix I 用 $\tau = 10^{-6}$ 做 certify，但只对 explored space。
- **Stretching 14D 的 $\det$ vs admissibility tradeoff**: 若 $\mathbf{G}_{\text{lower}}$ 因算法未收敛而在某些 $q$ 处略大于 $\mathbf{G}(q)$，admissibility 失效，informed set 可能 exclude 真正的解。这是 risk——尤其因为 Loewner order 是 partial order，"meet" 不一定能 closed-form 验证。

---

## 8. 参考链接

主要参考:
- 论文 arXiv 版本: 作者前作 https://arxiv.org/abs/2602.00992 (Kyaw & Kelly 2026 [8])
- Informed RRT* (Gammell et al. 2018 [4]): https://arxiv.org/abs/1706.06391
- BIT* (Gammell et al. 2020 [3]): https://arxiv.org/abs/1407.7346 实际是 https://arxiv.org/abs/1905.05611
- AIT*/EIT* (Strub & Gammell [13]): https://arxiv.org/abs/2101.06148
- G-RRT* (Kyaw et al. [14]): https://arxiv.org/abs/2405.03411
- AORRTC (Wilson et al. [41]): https://arxiv.org/abs/2409.14500 实际是 https://arxiv.org/abs/2409.14500
- Loewner order: https://en.wikipedia.org/wiki/Loewner_order
- Bhatia, *Positive Definite Matrices*: https://press.princeton.edu/books/hardcover/9780691129187/positive-definite-matrices
- OMPL: https://ompl.kavrakilab.org/
- MotionBenchMaker: https://motionbenchmaker.com/
- Pinocchio: https://github.com/stack-of-tasks/pinocchio
- VAMP (Vectorized AMP): https://arxiv.org/abs/2402.14617
- GCS on Riemannian manifolds [27]: https://arxiv.org/abs/2410.18703
- Kyaw & Kelly arXiv 2602.00992: https://arxiv.org/abs/2602.00992

---

## 9. 一句话总结

**核心 trick**：把 "find admissible heuristic for Riemannian geodesic distance" 这个问题转译成 "find a constant SPD matrix that Loewner-dominates $\mathbf{G}(q)$ for all $q$"，用 Cholesky 当 isometric map 把 anisotropic informed set 拉直成 PHS，套用 Euclidean 直接采样——保 admissibility 的同时利用 metric 的 directional structure，体积在 14 维下指数级缩小。

论文的 elegance 在于把一个看似 sampling 算法的问题归结为 SPD cone 上的 order-theoretic optimization，让 Riemannian informed set 与 Euclidean PHS 在 isometric 嵌入下统一。这个"换坐标系让难题变标准题"的思路其实贯穿 applied math（Fourier, wavelet, manifold learning 都在做这事），本文是把它在 motion planning 的 informed sampling 上落地得很干净的一个实例。
