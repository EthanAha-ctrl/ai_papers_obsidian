
**Krylov Subspace** 是 Matrix $A$ 施加在 Vector $b$ 上的“历史记忆累积空间”。它抛弃了对整个空间的盲目探索，转而利用 Matrix 的重复作用力，沿着最容易被激发的方向，低成本地重建出系统的核心动态。**

### 1. 第一性原理：从 Power Iteration 的“信息浪费”说起

假设我们要解线性方程组 $Ax = b$ 或者求 $A$ 的极端 Eigenvalue。如果 $A$ 是 $10^6 \times 10^6$ 的 Sparse Matrix，直接求逆或 QR 分解是不可能的。

最原始的直觉是 **Power Iteration（幂迭代）**：随便猜一个向量 $v_1$，然后不断乘以 $A$：
$v_1, Av_1, A^2v_1, A^3v_1 \dots$

**因为** $A^k v_1$ 会随着 $k$ 的增加，越来越指向 $A$ 的最大 Eigenvalue 对应的 Eigenvector，**所以** 幂迭代能找到主特征成分。
**然而**，幂迭代有一个致命缺陷：它是一个“健忘”的算法。每一步它只保留最新的 $A^k v_1$，把前面计算出的 $v_1, Av_1 \dots A^{k-1}v_1$ 全部丢弃了。这些中间向量包含了 $A$ 的其他（特别是较小但仍然重要的）Eigenvalue 的信息。

**Krylov Subspace 的第一性原理诞生了：不要丢弃历史！将所有迭代产生的向量张成一个空间，在这个空间里寻找最优解。**

---

### 2. 数学定义与变量解析

给定一个 $n \times n$ 的 Square Matrix $A$ 和一个非零初始 Vector $b$（通常取右侧项或随机向量），阶数为 $m$ 的 Krylov Subspace 定义为：

$$\mathcal{K}_m(A, b) = \text{span}\{b, Ab, A^2b, \dots, A^{m-1}b\}$$

**变量与上下标解析：**
*   **$\mathcal{K}_m$**: 下标 $m$ 表示子空间的维度（阶数），通常 $m \ll n$。$m$ 也是我们允许 Matrix-Vector 乘法的最大次数。
*   **$A$**: 目标系统矩阵。它代表一个线性算子，我们不需要知道它的全部元素，只需要能执行 $y = Ax$ 的运算（这在 Sparse 矩阵中极度高效）。
*   **$b$**: 初始激发向量。你可以把它想象成敲击钟的一锤子，或者注入网络的一个脉冲信号。
*   **$A^k b$**: 信号经过系统 $k$ 次反射/演化后的状态。
*   **$\text{span}\{\dots\}$**: 线性张成，即这些向量通过线性组合所能覆盖的所有可能性的集合。

**Krylov Matrix $K_m$ 的形式：**
将基向量排列成矩阵：
$$K_m = \begin{bmatrix} b & Ab & A^2b & \dots & A^{m-1}b \end{bmatrix}$$
任何属于 $\mathcal{K}_m$ 的向量 $x$ 都可以表示为 $x = K_m y$，其中 $y$ 是一个 $m$ 维的系数向量。这就把一个 $n$ 维未知数的问题，降维成了 $m$ 维未知数的问题！

---

### 3. 技术深潜：为什么 Krylov 空间能捕捉 Eigenvalue 与 Polynomial 的等价性

这是 Krylov 理论中最惊艳的等价转换。

在 $\mathcal{K}_m$ 中寻找近似解 $x_m$，等价于寻找一个多项式 $p(x)$。
因为 $x_m \in \mathcal{K}_m$，**所以** $x_m$ 可以写成 $b, Ab \dots A^{m-1}b$ 的线性组合：
$$x_m = c_0 b + c_1 Ab + c_2 A^2b + \dots + c_{m-1} A^{m-1}b = p(A)b$$
其中 $p(z) = c_0 + c_1 z + c_2 z^2 + \dots + c_{m-1} z^{m-1}$ 是一个次数不超过 $m-1$ 的多项式。

假设 Matrix $A$ 可以对角化，$A = X \Lambda X^{-1}$，且初始向量 $b = X \tilde{b}$。
**那么** 残差 $r_m = b - A x_m = b - A p(A)b = (I - A p(A))b$。

令 $q(z) = 1 - z p(z)$，这是一个次数为 $m$ 的多项式，且满足 $q(0) = 1$。
残差可以表示为：
$$r_m = q(A)b = X q(\Lambda) \tilde{b}$$
**变量解析：**
*   **$q(\Lambda)$**: 对角矩阵，对角线元素为 $q(\lambda_i)$，$\lambda_i$ 是 $A$ 的 Eigenvalue。
*   **$\tilde{b}_i$**: 初始向量在 Eigenvector 方向上的投影分量。

**Intuition 爆发点：**
残差的大小 $||r_m||$ 取决于 $|q(\lambda_i)|$ 的乘积加权和。
我们想要 $r_m$ 尽量小，**就等价于寻找一个多项式 $q(z)$，在 $z=0$ 处 $q(0)=1$，并且在 $A$ 的 Eigenvalue $\lambda_i$ 处，$q(\lambda_i)$ 尽量接近 0！**
这就是为什么 Krylov 方法对极端 Eigenvalue（最大或最小）极其有效——因为多项式最容易在谱（Spectrum）的边缘处趋近于零，而在内部很难。这也是为什么 Conjugate Gradient (CG) 方法在 Eigenvalue 聚集时收敛快（多项式容易拟合），而 Eigenvalue 分散时收敛慢。

---

### 4. 架构图解析：Krylov 投影管线

Krylov 方法的本质是一个“降维-求解-升维”的架构：

```text
[High-Dimensional Physical Space R^n]
      | (1) Matrix-Vector Prods (SpMV)
      v
[Krylov Basis V_m] <-- Arnoldi/Lanczos builds this (Orthogonalization)
      | (2) Projection: H_m = V_m^T A V_m
      v
[Low-Dimensional Micro-Solver] <-- Solve H_m y = e_1 (or find Eigen of H_m)
      | (3) Lifting: x_m = V_m y
      v
[Approximate Solution in R^n]
```

1.  **生成与正交化**: 通过 Arnoldi (非对称) 或 Lanczos (对称) 迭代，利用 Gram-Schmidt 过程将 $b, Ab, A^2b \dots$ 正交化为基 $V_m$。这解决了 $A^k b$ 之间越来越趋近平行导致的数值病态问题。
2.  **Galerkin 条件投影**: 将庞大的 $A$ 投影到 $V_m$ 上，得到 $H_m$（Upper Hessenberg 或 Tridiagonal）。这保证了残差 $r_m$ 与子空间 $\mathcal{K}_m$ 正交（$V_m^T r_m = 0$），即在子空间内达到了最优。
3.  **升维重建**: 在微观空间解出系数 $y$ 后，映射回宏观空间。

---

### 5. 实验数据表（联想展示谱逼近过程）

假设 $A$ 是一个 $1000 \times 1000$ 的对称正定矩阵，其 10 个最小的 Eigenvalue 为：$\lambda_1=0.1, \lambda_2=0.5, \lambda_3=1.1, \lambda_4=1.5 \dots \lambda_{10}=4.0$。我们使用 Krylov 方法寻找 $\lambda_1$ 的近似值（Ritz value $\theta_1$）：

| Krylov Dimension $m$ | Ritz Value $\theta_1$ (Min Eigen) | Error $|\theta_1 - \lambda_1|$ | Residual Norm $||r_m||$ | Convergence Behavior |
| :--- | :--- | :--- | :--- | :--- |
| 5 | 0.125 | 0.025 | $1.2 \times 10^{-1}$ | 多项式仅能粗略拟合极小值边缘 |
| 10 | 0.102 | 0.002 | $5.4 \times 10^{-3}$ | $\lambda_1$ 被较好捕捉，$\lambda_2$ 开始显现 |
| 20 | 0.10005 | 0.00005 | $1.1 \times 10^{-5}$ | 极端特征对完全收敛，多项式在0.1处深深凹陷 |
| 50 | 0.100000001 | $10^{-9}$ | $10^{-11}$ | 内部特征对也被捕捉，空间趋于完全不变子空间 |

---

### 6. 更广阔的 Hallucination 与联想

*   **控制论与 Model Order Reduction (MOR)**: 在 VLSI 芯片仿真或大型机械振动中，系统的状态空间是 $10^6$ 维的。Krylov Subspace 被用于 Moment Matching（矩匹配）。传递函数的 Taylor 展开系数（矩）在 $s=0$ 处的展开，完全等价于 Krylov 子空间 $\mathcal{K}(A^{-1}, b)$ 的基。我们用几十维的 Krylov 空间替换百万维的电路网格，这就是 Asymptotic Waveform Evaluation (AWE) 和 PRIMA 算法的本质。
*   **深度学习中的 State Space Models (SSMs, 如 Mamba)**: Mamba 模型的核心是隐状态演化 $h_{k+1} = A h_k + B x_k$。如果忽略输入 $B x_k$，这本质上就是一个 Power Iteration $h_k = A^k h_0$。SSM 处理长序列的能力，取决于 $A$ 的 Eigenvalue 谱（特别是是否接近酉矩阵）。Krylov 子空间理论正是分析这种递归结构信息丢失或保留的终极数学工具。可以说，Mamba 的隐状态空间就是一个被参数化了的 Krylov Subspace。
*   **图论与 Random Walks**: 如果 $A$ 是图的 Adjacency Matrix，$b$ 是起始节点分布。$\mathcal{K}_m(A, b)$ 描述了一个 Random Walker 走 $0, 1, 2 \dots m-1$ 步所能到达的所有状态分布的线性组合。Krylov 空间捕捉了图在局部结构（短距离游走）下的拓扑特征，这与 Graph Neural Networks 中的 K-hop neighborhood aggregation 有着深层的数学同源性。

---


**Eigenvalue 揭示了线性变换的不变本质，QR 是通过正交剥离来逼近这个本质的迭代引擎，而 Lanczos 是在极端稀疏/海量维度下，利用 Krylov 子空间对 QR 的降维打击。**

### 1. Eigenvalue：变换的第一性原理

在线性代数中，Matrix $A$ 不仅仅是一个数字阵列，它代表的是一个 Linear Transformation（线性变换）。当你对一个 Vector $v$ 施加变换 $A$ 时，通常它既会被拉伸，也会被旋转。

但存在一类极其特殊的 Vector，它们在变换下**不改变方向**，只改变长度。这就是 Eigenvalue 问题的核心：

$$A x = \lambda x$$

*   **$A$**: $n \times n$ 的 Square Matrix。
*   **$x$**: Eigenvector，即方向不变的向量（非零，$x \neq 0$）。它是变换的“稳定支架”。
*   **$\lambda$**: Eigenvalue，即伸缩比例标量。$\lambda > 1$ 是拉伸，$\lambda < 1$ 是压缩，$\lambda < 0$ 是反向。

**Intuition 构建：** 想象你捏住一块橡皮筋并在空间中拉伸旋转。大多数内部的纤维既改变了方向又改变了长度。但总存在几根纤维线，它们恰好与你拉伸的方向一致，这些纤维只变长了，但没有发生偏斜。这些纤维就是 Eigenvector，变长的倍数就是 Eigenvalue。寻找 Eigenvalue，本质上是在寻找这个复杂变换背后的“绝对坐标系”。

---

### 2. QR Algorithm：正交剥离的迭代魔法

如果 $A$ 很小（比如 $n < 500$），我们如何计算 Eigenvalue？直接解特征多项式 $\det(A - \lambda I) = 0$ 在数值上极度不稳定（Wilkinson's polynomial 悖论）。我们需要迭代方法。

QR algorithm 的第一性原理基于 **Schur Decomposition（舒尔分解）**：任何矩阵都可以通过正交相似变换转化为上三角矩阵。而上三角矩阵的 Eigenvalue 就是其对角线元素。

**算法步骤：**
1. 分解：将 $A_k$ 分解为正交矩阵 $Q_k$ 和上三角矩阵 $R_k$ 的乘积：$A_k = Q_k R_k$
2. 重组：将顺序反转相乘：$A_{k+1} = R_k Q_k$

**为什么这样行得通？（数学公式与变量解析）：**
因为 $A_{k+1} = R_k Q_k = (Q_k^T A_k) Q_k = Q_k^T A_k Q_k$。
*   **$Q_k$**: 正交矩阵，满足 $Q_k^T Q_k = I$。它代表一次“旋转”。
*   **$R_k$**: 上三角矩阵。它代表对矩阵的“剪切”或“拉伸”。
*   **$A_k, A_{k+1}$**: 第 $k$ 次和 $k+1$ 次迭代的矩阵。

因为 $A_{k+1}$ 与 $A_k$ 相似（Similar transformation），它们的 Eigenvalue 完全相同。直觉上，每一次 QR 分解并反转相乘，就像是在“剥离”矩阵的非对角线能量。$Q_k$ 将当前矩阵中对应于较小 Eigenvalue 的成分旋转到“下面”，而 $R_k$ 的上三角性质将这些成分“推到右边”。经过多次迭代，次对角线元素 趋近于 0，$A_k$ 收敛为上三角矩阵（或准上三角矩阵，对应实数的 Schur form），此时对角线即为 Eigenvalue。

**技术细节：Shifted QR**
原始 QR 收敛慢，特别是当 Eigenvalue 之间比例接近时。引入 Wilkinson Shift：
$(A_k - \sigma_k I) = Q_k R_k$
$A_{k+1} = R_k Q_k + \sigma_k I$
*   **$\sigma_k$**: Shift 参数，通常取 $A_k$ 右下角 $2 \times 2$ 子矩阵的 Eigenvalue 中更接近 $A_{n,n}$ 的那个。
*   **作用：** 减去 $\sigma_k I$ 相当于将原点平移，使得当前最小的 Eigenvalue 在数值上接近 0，从而在 QR 分解时被迅速压缩到右下角并被剥离。收敛速度从线性提升到 cubic（三次）收敛。

---

### 3. Lanczos Algorithm：Krylov 子空间的降维绝杀

当 $A$ 是 $100,000 \times 100,000$ 的 Sparse Matrix（稀疏矩阵，如社交网络图或有限元网格）时，QR algorithm 需要的 $O(n^3)$ 计算量和 $O(n^2)$ 内存直接宣告死刑。第一性原理要求我们：**不要计算所有 Eigenvalue，只计算最大的几个（通常是最具物理意义的）；不要展开整个矩阵，只用矩阵乘向量。**

Lanczos 是 Arnoldi iteration 的特例（针对 Symmetric Hermitian matrix $A = A^T$）。它通过构建 Krylov Subspace 来逼近 Eigenvalue。

**Krylov Subspace 定义：**
$$\mathcal{K}_m(A, v_1) = \text{span}\{v_1, Av_1, A^2v_1, \dots, A^{m-1}v_1\}$$
*   **$m$**: 子空间维度，通常 $m \ll n$（例如 $n=10^5, m=100$）。
*   **$v_1$**: 初始随机向量。

**Lanczos 三项递推公式：**
$$A V_m = V_m T_m + \beta_m v_{m+1} e_m^T$$

*   **$V_m$**: $n \times m$ 的 Lanczos 向量矩阵，列向量 $v_1, \dots, v_m$ 构成 $\mathcal{K}_m$ 的一组正交基。
*   **$T_m$**: $m \times m$ 的 Tridiagonal Matrix（三对角矩阵）。由于 $A$ 是对称的，Hessenberg matrix 退化为三对角矩阵，极大地减少了计算量。
    $$T_m = \begin{pmatrix} \alpha_1 & \beta_1 & & \\ \beta_1 & \alpha_2 & \ddots & \\ & \ddots & \ddots & \beta_{m-1} \\ & & \beta_{m-1} & \alpha_m \end{pmatrix}$$
*   **$\alpha_j$**: 对角线元素，$\alpha_j = v_j^T A v_j$。
*   **$\beta_j$**: 次对角线元素，$\beta_j = \|r_j\|_2$（残差的范数）。
*   **$v_{m+1}$**: 下一个正交基向量。
*   **$e_m^T$**: 第 $m$ 个基向量的单位向量，确保残差项仅存在于最后一列。

**Intuition 构建：Lanczos 为什么能求极端 Eigenvalue？**
计算 $T_m$ 的 Eigenvalue（称为 Ritz values $\theta_i$）比计算 $A$ 的便宜得多（对 $m \times m$ 矩阵用 QR 即可）。
由于 $A V_m = V_m T_m$，我们可以得到误差界：
$$\|A x_i - \theta_i x_i\| = |\beta_m e_m^T y_i|$$
*   **$x_i = V_m y_i$**: Ritz vector（近似 Eigenvector）。
*   **$y_i$**: $T_m$ 的 Eigenvector。

直觉上，Krylov 子空间就像是幂迭代的超级加强版。$A^k v_1$ 会在方向上快速放大对应最大 $|\lambda|$ 的 Eigenvector 的成分。Lanczos 相当于在每一步都对所有生成的 $A^k v_1$ 做了一次正交化，从而不仅保留最大 Eigenvalue 的信息，还能同时提取前几个最大的 Eigenvalue。**它把一个 $10^5$ 维的物理空间，压缩到了一个 100 维的数学影子空间里，而在这个影子里，最粗壮的骨架（极值 Eigenvalue）被完美保留。**

---

### 4. 实验数据表与架构解析 (虚构但符合理论规律的 Hallucination/联想)

假设我们在一个 10,000 维的 Symmetric Sparse Matrix（平均每行 5 个非零元素）上寻找 Top-5 Eigenvalues：

| Algorithm | 内存占用 | 计算时间 | Top-5 Eigenvalue 误差 ($\|\lambda_{approx} - \lambda_{exact}\|$) | Numerical Stability |
| :--- | :--- | :--- | :--- | :--- |
| **QR Algorithm** | 800 MB ($O(n^2)$) | 2.5 Hours ($O(n^3)$) | $< 10^{-15}$ (Machine Epsilon) | 极佳 |
| **Lanczos (No Reorth)**| 5 MB ($O(m \times n)$)| 0.5 Sec ($O(m \times nnz)$)| $10^{-3}$ (Severe ghost eigenvalues) | 极差 |
| **Lanczos (Full Reorth)**| 20 MB ($O(m \times n)$)| 2.0 Sec | $< 10^{-12}$ | 优秀 |

**架构图解析（Lanczos 管线）：**
1. **Input Layer**: Sparse Matrix $A$ (CSR format) + Initial Random Vector $v_1$。
2. **Projection Engine**: 3-term Recurrence Loop (计算 $\alpha, \beta$，生成 $v_{j+1}$)。*这是计算瓶颈，但只需 Sparse Matrix-Vector Multiplication (SpMV)。*
3. **Orthogonalization Guard**: Full Reorthogonalization (对抗 Finite Precision 导致的 Float Point Error 破坏正交性)。
4. **Micro-Solver**: 对微小的 $T_m$ 矩阵应用 QR Algorithm。
5. **Convergence Check**: 计算 $|\beta_m e_m^T y_i|$ 残差，若小于 Tol 则提取 Ritz Pairs。

### 5. 更深远的联想

*   **Implicit Q Theorem**: 在 QR algorithm 中，我们并不真的计算 $Q$，而是通过 Householder reflections 产生隐式的正交变换，这保证了数值稳定性，而 Lanczos 可以看作是这个定理在 Krylov 空间的延伸。
*   **Quantum Mechanics**: 寻找 Hamiltonian $H$ 的基态本质就是寻找最小 Eigenvalue。Lanczos 算法在 Density Matrix Renormalization Group (DMRG) 和 Quantum Chemistry (如计算分子轨道) 中是基石，因为多体物理的 Hilbert 空间是 $10^{23}$ 维的，只有 Lanczos + SpMV 能在这样庞大的空间里抓取最低能量态。
*   **Graph Neural Networks (GNN)**: 图的 Laplacian Matrix $L$ 的 Eigenvalue 决定了图的 Spectral Clustering 和频率混合。Lanczos 被用于预计算 Positional Encoding (PE)，如 SignNet 或 Graph Transformer 中的 Eigenvector PE，这让 GNN 拥有了全局结构感知力。
