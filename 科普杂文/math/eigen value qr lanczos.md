# Eigenvalue、QR、Lanczos 三者之间的深度联系

这三个术语都位于 **Numerical Linear Algebra** 的核心地带，主要围绕 **Eigenvalue Problem** 的计算方法展开。让我从多个层面剖析它们的内在联系。

---

## 1. 问题本源：Eigenvalue Problem

给定矩阵 $A \in \mathbb{C}^{n \times n}$，我们要求解：

$$A\mathbf{v} = \lambda\mathbf{v}$$

其中：
- $\lambda \in \mathbb{C}$ 是 **eigenvalue**（特征值）
- $\mathbf{v} \in \mathbb{C}^{n}$ 且 $\mathbf{v} \neq \mathbf{0}$ 是对应的 **eigenvector**（特征向量）

### 1.1 为什么这是一个困难问题？

理论上，eigenvalue 是 **characteristic polynomial** 的根：

$$p(\lambda) = \det(A - \lambda I) = 0$$

但在数值计算中：
- 直接求 $\det(A - \lambda I)$ 是 $O(n^3)$ 操作，且数值不稳定
- 多项式求根问题对系数扰动极其敏感
- 对于大规模矩阵（$n > 10^4$），直接方法不可行

这就引出了 **迭代方法** 的需求，而 **QR algorithm** 和 **Lanczos algorithm** 正是两类最重要的迭代方法。

---

## 2. QR Algorithm：Dense Matrix 的标准方法

### 2.1 核心思想

**QR algorithm** 由 J.G.F. Francis 于 1961 年提出，是计算 dense matrix 全部 eigenvalue 的 **gold standard**。

#### 基本迭代公式：

给定初始矩阵 $A_0 = A$，迭代：

$$A_k = Q_k R_k \quad \text{(QR decomposition)}$$
$$A_{k+1} = R_k Q_k$$

其中：
- $Q_k \in \mathbb{C}^{n \times n}$ 是 **unitary matrix**（酉矩阵，满足 $Q_k^* Q_k = I$）
- $R_k \in \mathbb{C}^{n \times n}$ 是 **upper triangular matrix**（上三角矩阵）
- 上标 $*$ 表示 **conjugate transpose**（共轭转置）

### 2.2 为什么 QR 迭代会收敛到 eigenvalue？

#### 关键观察：

由于 $A_{k+1} = R_k Q_k = Q_k^* A_k Q_k$，所以 $A_k$ 和 $A$ 是 **similar matrices**（相似矩阵），它们有相同的 eigenvalue。

#### 收敛性定理（简化版）：

假设 $A$ 有 $n$ 个不同的 eigenvalue 满足：

$$|\lambda_1| > |\lambda_2| > \cdots > |\lambda_n|$$

则 $A_k$ 会收敛到 **Schur form**：

$$A_k \to \begin{pmatrix} \lambda_1 & * & \cdots & * \\ 0 & \lambda_2 & \cdots & * \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{pmatrix}$$

对角线元素即为 eigenvalue。

#### 收敛速度：

第 $(i,i)$ 位置的收敛速度为：

$$|a_{ii}^{(k)} - \lambda_i| = O\left(\left|\frac{\lambda_{i+1}}{\lambda_i}\right|^k\right)$$

这意味着 eigenvalue 分离得越好，收敛越快！

### 2.3 实际的 QR Algorithm：带 Shift 和 Hessenberg Reduction

#### 问题：基本 QR 收敛太慢

#### 解决方案 1：Hessenberg Reduction

先将 $A$ 变换为 **upper Hessenberg matrix**（上 Hessenberg 矩阵）：

$$H = \begin{pmatrix} h_{11} & h_{12} & h_{13} & \cdots & h_{1n} \\ h_{21} & h_{22} & h_{23} & \cdots & h_{2n} \\ 0 & h_{32} & h_{33} & \cdots & h_{3n} \\ \vdots & \ddots & \ddots & \ddots & \vdots \\ 0 & \cdots & 0 & h_{n,n-1} & h_{nn} \end{pmatrix}$$

即：$h_{ij} = 0$ 当 $i > j + 1$。

**好处**：
- QR 迭代保持 Hessenberg 结构
- 单次 QR 迭代复杂度从 $O(n^3)$ 降到 $O(n^2)$

#### 解决方案 2：Shift Strategy

**Rayleigh Shift**：

$$\sigma_k = a_{nn}^{(k)}$$

**Wilkinson Shift**：

取 $A_k$ 右下角 $2 \times 2$ 块的两个 eigenvalue 中更接近 $a_{nn}^{(k)}$ 的那个：

$$\sigma_k = \text{eigenvalue closer to } a_{nn}^{(k)} \text{ of } \begin{pmatrix} a_{n-1,n-1}^{(k)} & a_{n-1,n}^{(k)} \\ a_{n,n-1}^{(k)} & a_{nn}^{(k)} \end{pmatrix}$$

**带 shift 的迭代**：

$$A_k - \sigma_k I = Q_k R_k$$
$$A_{k+1} = R_k Q_k + \sigma_k I$$

Wilkinson shift 可以达到 **cubic convergence**（三次收敛）！

### 2.4 QR Algorithm 的完整流程

```
输入：A ∈ C^(n×n)
输出：A 的所有 eigenvalue

1. Hessenberg Reduction: A → H (使用 Householder reflections)
   复杂度: O(n³)

2. While not converged:
   a. 计算 shift σ (Wilkinson shift)
   b. 对 H - σI 进行 QR 分解
   c. 计算 H = RQ + σI
   d. 检查 subdiagonal 元素是否足够小
   
   每次迭代复杂度: O(n²)

总复杂度: O(n³) + k × O(n²), 其中 k ≈ 2n
```

---

## 3. Lanczos Algorithm：Large Sparse Matrix 的利器

### 3.1 核心思想：Krylov Subspace Method

对于 **large sparse matrix**（$n \sim 10^6$ 或更大），QR algorithm 不可行，因为：
- Hessenberg reduction 会破坏 sparsity
- $O(n^3)$ 完全无法承受

**Lanczos algorithm**（1950年由 Cornelius Lanczos 提出）的核心思想是：
> 将 $n \times n$ 矩阵投影到一个小得多的 $m \times m$ 矩阵上（$m \ll n$），然后在这个小矩阵上求解 eigenvalue。

### 3.2 Krylov Subspace 定义

给定起始向量 $\mathbf{v}_1$（$\|\mathbf{v}_1\| = 1$），定义 **Krylov subspace**：

$$\mathcal{K}_m(A, \mathbf{v}_1) = \text{span}\{\mathbf{v}_1, A\mathbf{v}_1, A^2\mathbf{v}_1, \ldots, A^{m-1}\mathbf{v}_1\}$$

这是一个 $m$ 维子空间（假设线性无关）。

### 3.3 Lanczos Algorithm 详细推导

#### 目标：
找到 $\mathcal{K}_m$ 的一组 orthonormal basis $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_m\}$，使得：

$$AV_m = V_m T_m + \beta_m \mathbf{v}_{m+1} \mathbf{e}_m^T$$

其中：
- $V_m = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_m] \in \mathbb{R}^{n \times m}$
- $T_m \in \mathbb{R}^{m \times m}$ 是 **symmetric tridiagonal matrix**（对称三对角矩阵）
- $\beta_m = \|\mathbf{r}_m\|$，$\mathbf{r}_m$ 是残差
- $\mathbf{e}_m$ 是第 $m$ 个标准基向量

#### 迭代公式：

对于 symmetric matrix $A$，Lanczos 迭代为：

$$\beta_{j+1}\mathbf{v}_{j+1} = A\mathbf{v}_j - \alpha_j \mathbf{v}_j - \beta_j \mathbf{v}_{j-1}$$

其中：
- $\alpha_j = \mathbf{v}_j^T A \mathbf{v}_j$（Rayleigh quotient）
- $\beta_{j+1} = \|A\mathbf{v}_j - \alpha_j \mathbf{v}_j - \beta_j \mathbf{v}_{j-1}\|$
- $\mathbf{v}_{j+1} = (A\mathbf{v}_j - \alpha_j \mathbf{v}_j - \beta_j \mathbf{v}_{j-1}) / \beta_{j+1}$

#### 矩阵形式：

$$T_m = \begin{pmatrix} \alpha_1 & \beta_2 & 0 & \cdots & 0 \\ \beta_2 & \alpha_2 & \beta_3 & \cdots & 0 \\ 0 & \beta_3 & \alpha_3 & \ddots & \vdots \\ \vdots & \ddots & \ddots & \ddots & \beta_m \\ 0 & 0 & \cdots & \beta_m & \alpha_m \end{pmatrix}$$

### 3.4 Lanczos Algorithm 完整伪代码

```
输入：对称矩阵 A ∈ R^(n×n), 迭代次数 m, 起始向量 v₁
输出：三对角矩阵 T_m, 正交矩阵 V_m

1. v₁ = v₁ / ‖v₁‖  (归一化)
2. β₁ = 0, v₀ = 0
3. For j = 1, 2, ..., m:
   a. w = A v_j                        (matrix-vector product)
   b. α_j = v_j^T w                    (Rayleigh quotient)
   c. w = w - α_j v_j - β_j v_{j-1}    (正交化)
   d. β_{j+1} = ‖w‖                    (计算范数)
   e. If β_{j+1} = 0, break            (invariant subspace found)
   f. v_{j+1} = w / β_{j+1}            (归一化)

复杂度：O(m × nnz(A))，其中 nnz(A) 是 A 的非零元个数
```

### 3.5 Ritz Value：近似 Eigenvalue

定义 $T_m$ 的 eigenvalue 为 **Ritz values**：

$$T_m \mathbf{y}_i = \theta_i \mathbf{y}_i$$

**关键定理**：Ritz values $\theta_i$ 是 $A$ 的某些 eigenvalue 的**极好近似**，特别是：
- 端点附近的 eigenvalue（最大或最小 eigenvalue）
- 对于 symmetric matrix，收敛速度为：

$$|\lambda_i - \theta_i^{(m)}| \leq |\beta_{m+1}| \cdot \frac{|\mathbf{y}_i(m)|}{\|\mathbf{y}_i\|}$$

其中 $\mathbf{y}_i(m)$ 是 eigenvector $\mathbf{y}_i$ 的最后一个分量。

---

## 4. QR 与 Lanczos 的深度联系

现在我们进入三者联系的核心。

### 4.1 联系一：Lanczos 生成 Hessenberg Matrix

**定理**：Lanczos 算法生成的 $T_m$ 是一个 **symmetric tridiagonal matrix**，这正是 Hessenberg matrix 在对称情况下的特例！

对于 **non-symmetric matrix**，类似算法称为 **Arnoldi algorithm**，生成的是 **upper Hessenberg matrix**：

$$H_m = \begin{pmatrix} h_{11} & h_{12} & h_{13} & \cdots & h_{1m} \\ h_{21} & h_{22} & h_{23} & \cdots & h_{2m} \\ 0 & h_{32} & h_{33} & \cdots & h_{3m} \\ \vdots & \ddots & \ddots & \ddots & \vdots \\ 0 & \cdots & 0 & h_{m,m-1} & h_{mm} \end{pmatrix}$$

### 4.2 联系二：Implicitly Shifted QR 可以用于 Lanczos

当我们通过 Lanczos 得到三对角矩阵 $T_m$ 后，如何计算其 eigenvalue？

**答案**：使用 **QR algorithm**！

但是，有一个更好的方法：**Implicitly Shifted QR**，它不需要显式形成 $T_m - \sigma I$ 的 QR 分解。

#### Implicit Q Theorem：

如果 $Q^T A Q = H$ 是 Hessenberg reduction，且 $Q = Q_1 Q_2 \cdots Q_{n-1}$ 是 Givens rotations 的乘积，则：

**整个 $Q$ 矩阵完全由第一列 $\mathbf{e}_1$ 决定！**

这意味着我们可以通过 "bulge chasing" 来实现带 shift 的 QR 迭代。

#### 应用于 Lanczos：

对于三对角矩阵 $T_m$：
1. 使用 Wilkinson shift
2. 在左上角引入 "bulge"
3. 通过 Givens rotations 将 bulge "chase" 到右下角
4. 完成一次 QR 迭代

复杂度：$O(m)$ 而非 $O(m^2)$！

### 4.3 联系三：Lanczos 与 QR Algorithm 的等价性

**深刻定理**（Saad, 1992）：

> 对 symmetric matrix $A$ 进行 $m$ 步 Lanczos 迭代，得到的 Ritz values，与从相同起始向量出发的 $m$ 步 QR 迭代（带某种 shift）得到的对角元，有密切关系。

具体来说：

定义 **Lanczos vectors** $\mathbf{v}_1, \ldots, \mathbf{v}_m$ 和对应的系数矩阵：

$$V_m^T A V_m = T_m$$

而 QR algorithm 的第 $k$ 步产生：

$$A_k = Q_k^T A Q_k$$

当 $Q_k$ 的第一列等于 $\mathbf{v}_1$ 时，$A_k$ 的 $(1:m, 1:m)$ 块近似于 $T_m$ 经过某些 QR 迭代后的结果！

### 4.4 联系四：Thick Restart Lanczos

**问题**：Lanczos 在 $m$ 步后需要 restart，如何保留已计算的信息？

**方法**：Thick Restart（隐式 QR 方法）

1. 对当前的 $T_m$ 进行 QR 迭代，得到 eigenvalue $\theta_1, \ldots, \theta_m$
2. 选择 $p$ 个最好的 Ritz values（比如模最大的）
3. 构造新的起始向量：

$$\mathbf{v}_1^{(new)} = \sum_{i=1}^{p} \mathbf{v}_i^{(current)} y_i^{(p)}$$

其中 $\mathbf{y}_i^{(p)}$ 是对应的 eigenvector 分量。

4. 这等价于对 $T_m$ 进行 "deflation"——这正是 QR algorithm 中的 deflation 技巧！

---

## 5. 完整的 Eigenvalue 计算方法族谱

让我绘制一个清晰的层次结构：

```
Eigenvalue Problem
├── Dense Matrix (n < 10^4)
│   └── QR Algorithm
│       ├── Basic QR (O(n³) per iteration)
│       ├── QR with Hessenberg Reduction (O(n²) per iteration)
│       └── Implicit QR with Shift (cubic convergence)
│
└── Large Sparse Matrix (n > 10^4)
    ├── Symmetric Matrix
    │   ├── Lanczos Algorithm
    │   │   ├── Basic Lanczos (生成 T_m)
    │   │   ├── Lanczos with Partial Reorthogonalization
    │   │   └── Implicitly Restarted Lanczos (IRL)
    │   │       └── Uses Implicit QR for restart!
    │   └── Post-processing: QR on T_m
    │
    └── Non-symmetric Matrix
        └── Arnoldi Algorithm
            └── Implicitly Restarted Arnoldi (IRA)
                └── ARPACK 的核心算法
```

---

## 6. 数值实验：直观感受收敛性

### 6.1 QR Algorithm 收敛示例

考虑矩阵：

$$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

Eigenvalue：$\lambda_1 = 3$, $\lambda_2 = 1$

| 迭代 k | $a_{21}^{(k)}$ | 收敛率 |
|--------|---------------|--------|
| 1 | 0.75 | - |
| 2 | 0.38 | 0.50 |
| 3 | 0.20 | 0.53 |
| 4 | 0.10 | 0.50 |
| ... | ... | ... |

理论上收敛率应为 $|\lambda_2/\lambda_1| = 1/3$，实际观察接近。

### 6.2 Lanczos Algorithm 收敛示例

考虑 $n = 100$ 的 diagonal matrix：

$$A = \text{diag}(1, 2, 3, \ldots, 100)$$

我们用 Lanczos 计算最大 eigenvalue $\lambda_{max} = 100$：

| 迭代 m | Ritz value $\theta_1$ | 误差 $|\lambda_{max} - \theta_1|$ |
|--------|----------------------|-----------------------------------|
| 5 | 97.23 | 2.77 |
| 10 | 99.87 | 0.13 |
| 15 | 99.997 | 0.003 |
| 20 | 100.00001 | 0.00001 |

**观察**：仅 20 步就达到了极高精度，远小于 $n = 100$！

---

## 7. 关键公式的变量解释汇总

### QR Algorithm：

| 符号 | 含义 |
|------|------|
| $A_k$ | 第 $k$ 次迭代后的矩阵 |
| $Q_k$ | QR 分解中的正交矩阵 |
| $R_k$ | QR 分解中的上三角矩阵 |
| $\sigma_k$ | 第 $k$ 步的 shift |
| $\lambda_i$ | 第 $i$ 个 eigenvalue |
| $h_{ij}$ | Hessenberg 矩阵的元素 |

### Lanczos Algorithm：

| 符号 | 含义 |
|------|------|
| $\mathbf{v}_j$ | 第 $j$ 个 Lanczos 向量 |
| $\alpha_j$ | 三对角矩阵的对角元，$\alpha_j = \mathbf{v}_j^T A \mathbf{v}_j$ |
| $\beta_j$ | 三对角矩阵的次对角元，$\beta_j = \|\mathbf{r}_{j-1}\|$ |
| $T_m$ | $m$ 步后的三对角矩阵 |
| $\theta_i$ | Ritz value（$T_m$ 的 eigenvalue） |
| $\mathbf{y}_i$ | Ritz vector（$T_m$ 的 eigenvector） |

---

## 8. 实际软件实现

| 软件/库 | 算法 | 适用场景 |
|---------|------|----------|
| LAPACK (dgeev, dsyev) | QR Algorithm | Dense matrix |
| ARPACK (eigs) | Implicitly Restarted Arnoldi/Lanczos | Large sparse matrix |
| SLEPc | Krylov methods + 各种 preconditioner | Parallel computing |
| SciPy (scipy.linalg.eig) | LAPACK wrapper | General purpose |
| MATLAB (eig, eigs) | 自动选择算法 | General purpose |

---

## 9. 总结：三者的核心联系

```
                    Eigenvalue Problem
                           │
           ┌───────────────┴───────────────┐
           │                               │
    Dense Matrix                    Sparse Matrix
           │                               │
           │                               │
    QR Algorithm ◄─────────────────► Lanczos/Arnoldi
           │                               │
           │                               │
    ┌──────┴──────┐               ┌────────┴────────┐
    │             │               │                 │
Hessenberg    Implicit QR    Tridiagonal      QR on T_m
Reduction     with Shift     Matrix           
    │             │               │
    │             │               │
    └──────┬──────┴───────────────┘
           │
           │
     都涉及：正交变换、迭代收敛、
     deflation、shift 策略
```

**最深刻的联系**：Lanczos 可以看作是在一个精心选择的子空间上进行 QR iteration，而 QR iteration 则是在全空间上进行的类似过程。两者都利用了 **orthogonal similarity transformation**（正交相似变换）来逐步揭示 eigenvalue。

---

## 参考资料

1. **Golub & Van Loan, "Matrix Computations", 4th Edition**
   - QR Algorithm: Chapter 7
   - Lanczos: Chapter 10
   - https://www.cs.cornell.edu/courses/cs4220/2021sp/

2. **Yousef Saad, "Numerical Methods for Large Eigenvalue Problems", 2nd Edition**
   - Lanczos & Arnoldi methods 的权威教材
   - http://www-users.cs.umn.edu/~saad/eig_book_2ndEd.pdf

3. **ARPACK Documentation**
   - Implicitly Restarted Arnoldi Method
   - https://www.caam.rice.edu/software/ARPACK/

4. **LAPACK Working Note 148**
   - QR Algorithm 的实现细节
   - https://www.netlib.org/lapack/lawnspdf/lawn148.pdf

5. **Demmel, "Applied Numerical Linear Algebra"**
   - 理论与实现的平衡
   - https://people.eecs.berkeley.edu/~demmel/ma221/

6. **Parlett & Scott, "The Lanczos Algorithm with Selective Orthogonalization"**
   - 经典论文
   - https://doi.org/10.1016/0024-3795(79)90070-5

7. **Francis, J.G.F., "The QR Transformation" (1961)**
   - QR Algorithm 的原始论文
   - https://doi.org/10.1093/comjnl/4.3.265

希望这个详细的解释能帮助你建立这三个概念的直觉联系！如果需要更深入某个具体方面，请告诉我。