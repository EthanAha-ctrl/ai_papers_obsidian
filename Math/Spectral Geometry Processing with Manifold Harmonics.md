---
source_pdf: Spectral Geometry Processing with Manifold Harmonics.pdf
paper_sha256: 860adb5999e9ebd8b0d15377cceae7c6cd0b6a3c951371896c0b68883c93abbb
processed_at: '2026-08-12T09:50:51-07:00'
target_folder: Math
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

好，我换个口吻。想象咱俩在咖啡馆，你在白板上画图，我跟你聊这篇 paper 到底干了啥。

---

## 一句话版本

他们让 mesh 上也能做 Fourier Transform，你能像 Photoshop 调图像频率那样调 mesh 的"频率"，而且能跑到一百万 vertices。

---

## 这事为啥难（intuition 层）

你想，普通图像为啥能做 FFT？因为 pixel grid 是均匀的，sine wave 在上面就是 $\sin(kx)$，非常规整。

但 mesh 是啥？是一堆三角形乱连的 graph，vertex 分布不均匀，还有任意 topology（球、torus、dragon 都行）。你直接拿 graph Laplacian 的 eigenvectors 当 Fourier basis，会出 bug。

具体看 Figure 2 那张图，他们 filter 一个左半密、右半疏的 mesh。结果：
- (A) combinatorial Laplacian：右腿比左腿粗，因为 basis 只看 connectivity 不看几何
- (B) unweighted cotangent：还是有形变，因为没考虑 sampling density
- (C) weighted cotangent $(\cot\beta + \cot\beta')/A_i$：直接崩了，重建不出来，因为这矩阵**不对称**，eigenvectors 不正交
- (D) 经验 symmetrize $(\cot\beta + \cot\beta')/(A_i + A_j)$：好一点但仍有形变
- (E) 他们的 $(\cot\beta + \cot\beta')/\sqrt{A_i A_j}$：终于对了

直觉上发生了啥？你看 weight 的分母从 $A_i$ 变成 $\sqrt{A_i A_j}$。这是个几何平均数。它把"两个 vertex 各自的 sampling density"以对称方式平均掉，同时把 geometry 信息保留。这就像 normalized graph Laplacian $D^{-1/2} L D^{-1}^{-1/2}$ 那种 trick，把 degree matrix 从 operator 里"挤"到 inner product 里去。参考 Chung 的 Spectral Graph Theory (https://www.cs.yale.edu/homes/spielman/561/2009/chung.pdf) 讲的就是这个思路在 graph 上的版本。

---

## DEC 这一套到底在干嘛

Discrete Exterior Calculus 听着吓人，其实就是个"怎么在 mesh 上严格定义 Laplacian"的框架。

你看三个核心 object：

1. **0-form**：vertex 上的标量值，就是"函数值"。
2. **1-form**：edge 上的标量值，就是"差分"。
3. **Hodge star $\star$**：告诉你怎么把 0-form 和 2-form、1-form 和 1-form 互相对偶。在 surface 上，$\star_0$ 是 diagonal matrix，每个 vertex 存它的 dual cell area（Voronoi area）；$\star_1$ 也是 diagonal，每个 edge 存 $\cot\beta + \cot\beta'$。

Laplacian 在这套语言里就是 $\Delta = -\star_0^{-1} d^T \star_1 d$。展开后你会发现，哎这不就是 cotangent Laplacian 嘛！

但 DEC 的好处是它严格区分了"operator"和"inner product"。inner product 是 $\langle f, g\rangle = f^T \star_0 g$。当你想让 eigenvectors 在这个 inner product 下 orthonormal，你就要把 operator 对称化，得到 $\bar{\Delta} = \star_0^{-1/2} \Delta \star_0^{-1/2}$。这就是 Figure 2 (E) 那个 weight 的来源。

为啥对称化能让 eigenvectors 正交？因为对称矩阵的 spectral theorem 保证 eigenvectors 互相正交。就这么简单。非对称矩阵没这个 guarantee，Figure 2 (C) 就翻车了。

参考 Desbrun et al. 2005 DEC course notes (https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf) 和 Hirani 2003 thesis (https://thesis.library.caltech.edu/2943/1/thesis_hirani.pdf)。

---

## MHB 和 MHT 是什么

**MHB（Manifold Harmonics Basis）** = Laplacian 的 eigenvectors。第 $k$ 个 eigenvector $H^k$ 是一个长度 $n$ 的 vector，每个分量 $H^k_i$ 是 vertex $i$ 上的值。Figure 3 画了几个，你看那些花花绿绿的图，长得就跟 2D sine wave 似的，只是被 wrap 到 surface 上了。

**MHT（Manifold Harmonic Transform）** = 把几何投影到这个 basis。

forward：
$$\tilde{x}_k = x^T \star_0 H^k = \sum_i x_i (\star_0)_{ii} H^k_i$$

变量解释：
- $x_i$：vertex $i$ 的 $x$ 坐标
- $(\star_0)_{ii} = A_i$：vertex $i$ 的 dual area
- $H^k_i$：第 $k$ 个 eigenfunction 在 vertex $i$ 的值
- $\tilde{x}_k$：第 $k$ 个 frequency coefficient

注意那个 $A_i$ —— 它是 sampling density correction。你 mesh 在某个区域密一点，那个区域的 vertex 权重就小一点（因为面积小），这是 Riemann sum 的精神。

inverse：
$$x_i = \sum_k \tilde{x}_k H^k_i$$

Figure 4 那张图特别好：用前几个 coefficient 重建 dragon，是个模糊的 dragon silhouette；加到几百个，细节慢慢出来。这跟 JPEG 里你保留多少 DCT coefficient 一个道理。

---

## Filter 怎么用

每个 $\tilde{x}_k$ 对应一个频率 $\omega_k = \sqrt{\lambda_k}$（注意是 sqrt，因为 $\lambda$ 是 $\omega^2$，跟 continuous 情况 $-\partial^2 \sin(\omega x) = \omega^2 \sin(\omega x)$ 对应）。

你想 low-pass，就定义 $F(\omega) = 1$ if $\omega < \omega_c$ else $0$。然后 filtered 几何是：

$$x^F_i = \sum_k F(\omega_k) \tilde{x}_k H^k_i$$

但有个细节：你只算了前 $m$ 个 eigenpairs，剩下高频丢了。论文里把这部分 residual 存下来，叫 $x^{hf}$：

$$x^{hf}_i = x_i - \sum_{k=1}^m \tilde{x}_k H^k_i$$

最后 filtered 结果（式 8）：
$$x^F_i = \sum_{k=1}^m F(\omega_k) \tilde{x}_k H^k_i + f^{hf} x^{hf}_i$$

其中 $f^{hf}$ 是 filter 在 $[\omega_m, \omega_M]$ 上的平均。这一项挺 hacky 的，相当于把整个高频段当一个 wave packet 整体处理，不再细分。Nyquist 频率 $\omega_M$ 是平均 edge length inverse 的一半。

为啥 interactive？因为 MHB 和 MHT 都不依赖 $F(\omega)$，precompute 完之后改 filter 只需重跑 inverse MHT，是个 $O(nm)$ 的 streaming 操作。Figure 5 那个 slider 你拖一下就更新。

---

## Numerical 怎么搞到 1M vertices

这是这篇 paper 的真正硬核贡献。

### 难在哪

你想算几千个 eigenpairs。两个 bottleneck：

**Bottleneck 1**：iterative eigensolver（ARPACK 用的 Arnoldi）擅长 spectrum 的高端（large eigenvalue），但你要的是低端（small eigenvalue，低频）。低端 condition number 差，收敛慢。

直觉：低频 eigenvector 对应 strong smoothing 的 fixed point，smoothing kernel 是 $e^{-t\Delta}$ 的高次幂，谱 gap 小的话收敛就慢。

**Bottleneck 2**：算的多 eigenvectors 时间 superlinear，而且 1M vertices × 1000 eigenvectors 存不下。

### Shift-Invert 是啥

技巧是同时做两件事：

**Shift**：把 $\bar{\Delta}$ 平移成 $\Delta_S = \bar{\Delta} - \lambda_S I$，把你要的频段移到 0 附近。

**Invert**：解 $\Delta_S^{-1} \bar{H}^k = \mu_k \bar{H}^k$，把"靠近 0 的 eigenvalue"放大成"很大的 inverse eigenvalue"。

数学验证：如果原来 $\bar{\Delta} \bar{H}^k = \lambda_k \bar{H}^k$，那么 $\Delta_S \bar{H}^k = (\lambda_k - \lambda_S)\bar{H}^k$，所以 $\Delta_S^{-1} \bar{H}^k = \frac{1}{\lambda_k - \lambda_S}\bar{H}^k$。反推：

$$\lambda_k = \lambda_S + \frac{1}{\mu_k}$$

变量含义：
- $\lambda_S$：你 shift 到的中心
- $\mu_k$：变换后的 eigenvalue（Arnoldi 求这个）
- $\lambda_k$：原 Laplacian 的 eigenvalue

Arnoldi 善于找 large $\mu$，正好对应原 spectrum 中靠近 $\lambda_S$ 的 $\lambda$。把弱项变强项。

### 实际求解

不需要显式算 $\Delta_S^{-1}$。Arnoldi 每次 query "给定向量 $\vec{\nu}$，求 $\Delta_S^{-1}\vec{\nu}$"。你预先做一次 sparse Cholesky factorization $\Delta_S = LDL^T$，然后每次 query 就是 back-substitution，$O(\text{nnz})$ 快得很。

对 1M vertices，Cholesky factor 也装不下内存，用 out-of-core factorization（Meshar-Irony-Toledo 2006, https://dl.acm.org/doi/10.1145/1149951.1149956）。这是 Sivan Toledo 给他们的 TAUCS future release。

### Band-by-band

不能一次算几千个 eigenpairs。所以切成 band，每个 band 算 50 个。算法长这样：

```
λ_S = 0, λ_last = 0
while λ_last < ω_m²:
    factorize Δ_S = Δ̄ - λ_S·I     # 一次 Cholesky
    compute 50 largest μ eigenpairs of Δ_S⁻¹  # Arnoldi
    for each:
        λ_k = λ_S + 1/μ_k
        if λ_k > λ_last: stream to disk
    λ_S ← max(λ_k) + 0.4·bandwidth    # overlap 40%
    λ_last ← max(λ_k)
```

40% overlap 是防止 band 边界漏掉 eigenvalue。如果发现漏了就 recompute 更大的 band。

关键：每个 band 时间 = 一次 factorization + 一次 Arnoldi（找 50 个）。总时间 linear in #eigenpairs。破解了 superlinear 瓶颈。

### Limited-memory Filtering

如果只想要最终 filtered mesh，根本不需要存整个 MHB。用个代数 trick：

$$x^F = \sum_k F(\omega_k) \langle x, H^k\rangle H^k$$

而 $x = \sum_k \langle x, H^k\rangle H^k$，所以：

$$x^F = x + \sum_k (F(\omega_k) - 1)\langle x, H^k\rangle H^k$$

伪代码：
```
x^F ← x
for each (H^k, ω_k):
    x^F ← x^F + (F(ω_k) - 1) · (x^T ⋆_0 H^k) · H^k
```

每次只 load 一个 eigenvector，用完丢掉。内存 $O(n)$ 而不是 $O(nm)$。再配合 shift-invert，只算 filter 真正修改的频段，省更多。

---

## 实验数据再说一遍

| Model | n | m | MHB | MHT | MHT$^{-1}$ | LM-filt |
|-------|---|---|-----|-----|------------|---------|
| dino | 56K | 447 | 77s | 0.34s | 0.53s | 18s |
| drago | 150K | 315 | 160s | 0.65s | 1.02s | 41s |
| drago1 | 244K | 667 | 9m | 18s | 4s | 135s |
| drago2 | 500K | 800 | 2h21m | 32s | 48s | 28m |
| drago3 | 1M | 1331 | 6h | 76s | 85s | 1h |

读这张表的 intuition：
- MHB 时间随 $n$ 大致 $n^{1.5}$ 量级（sparse Cholesky 的典型 scaling）
- MHT / inverse MHT 是 $O(nm)$ streaming，很快
- LM-filt 当 filter 只覆盖 1/4 频谱时，比完整 MHB+MHT+MHT$^{-1}$ 还快，因为只算需要的 band
- 1M vertices 6h MHB 是真瓶颈，论文 future work 提 multiresolution

---

## 这套东西在更大 context 里

### 在 geometry processing 里

- Taubin 1995 (https://dl.acm.org/doi/10.1145/218380.218484)：用 graph Laplacian 做"近似 low-pass filter"，但只把 Fourier 当理论工具，没真正算 eigenvectors。这篇 paper 是"真算出来"的版本。
- HKS (Sun et al. 2009, https://dl.acm.org/doi/10.1145/1531326.1531336)：用 heat kernel $K_t(x,x) = \sum_k e^{-\lambda_k t} (H^k(x))^2$ 做 shape descriptor。MHB 是它的 precompute。
- WKS (Aubry et al. 2011, https://dl.acm.org/doi/10.1145/2010324.1966442)：wave kernel signature，MHB 的另一种用法。
- Functional Maps (Ovsjanikov et al. 2012, https://dl.acm.org/doi/10.1145/2185520.2185526)：shape correspondence 在 MHB-reduced 空间里做，basis 就是这篇的输出。

### 在 ML 里

- Bruna et al. 2013 spectral CNN (https://arxiv.org/abs/1312.6203)：graph 上用 Laplacian eigenvectors 做 convolution，跟这篇 paper 思路同源，只是 graph 而非 mesh。scalability 同样是问题。
- Defferrard et al. 2016 ChebNet (https://arxiv.org/abs/1606.09375)：用 Chebyshev polynomial 逼近 $F(\lambda)$ 避免 explicit eigenvectors。ML 界后来都走这条路，因为 filter 是 learned 不是 designed，polynomial 便于 backprop。
- 这篇 paper 走的是"explicit eigenvector computation"路线的极致优化，跟 ChebNet 是两条平行解法。

### 在 diffusion models 里

Heat kernel $K_t = \sum_k e^{-\lambda_k t} H^k \otimes H^k$ 是 diffusion 的 Green function。Diffusion model 在 manifold 上跑 Langevin dynamics $dx_t = -\nabla\log p\,dt + \sqrt{2}dB_t$，背后的 generator 就是 Laplacian。MHB 就是它的谱分解。参考 score-based on implicit manifolds (https://arxiv.org/abs/2202.03013)。

---

## 这篇 paper 自己承认的 limitation

1. **MHB 不 spatially localized**：每个 eigenvector 是 global sine wave，重建 geometry 要很多系数。Figure 4 你看到要几百个 coefficient 才能认出 dragon。JPEG 用 small block 避免 global DCT 这个问题，paper future work 提到 partitioning 但会破坏 continuity。
2. **Future work: Manifold Wavelets** (Grinspun et al. 2002 CHARMS, https://dl.acm.org/doi/10.1145/566570.566594)：要 frequency 和 spatial 同时 localized。
3. **Creases 处理差**：Fourier 类方法在 sharp edges 上 Gibbs ringing。Anisotropic Laplacian 可能改善。
4. **1M+ vertices 6h 太慢**：要 multiresolution。
5. **不能直接用于 mesh compression**：MHB 依赖几何，Karni-Gotsman (https://dl.acm.org/doi/10.1145/344779.344934) 的 spectral compression 要 combinatorial Laplacian（connectivity-invariant）。

---

## 总体直觉

如果让我用三句话总结这篇 paper 的"啊哈"：

1. **Manifold 上 Fourier = Laplacian eigenvectors**，前提是你得严格处理 symmetry（DEC + $\star_0^{-1/2}$ trick）。
2. **Shift-invert 把 iterative solver 的弱点变成强点**：找 small $\lambda$ 难，但找 large $\mu = 1/(\lambda - \lambda_S)$ 容易。
3. **Band-by-band + streaming**：把"一次算太多"切成"每次算一点"，时间内存都 linear。

剩下的就是 engineering 细节：Cholesky factorization、out-of-core、40% overlap 保险，这些是把数学 idea 落到 1M vertices 上的必要工作。

希望这次更"人话"的版本好读一些。如果你想 drill down 哪个具体点（比如 indefinite Cholesky 怎么处理负 eigenvalue、为什么是 40% overlap 不是别的、HKS 怎么用 MHB），我可以再展开。

---

# Spectral Geometry Processing with Manifold Harmonics — 深度讲解

Andrej, 这篇 2008 SGP 的 paper 是 Bruno Vallet 和 Bruno Lévy 在 INRIA ALICE group 的工作，它把 signal processing 在 surface 上的完整 pipeline（不是只用作理论分析工具）真正 explicit 地实现出来，并且通过 numerical tricks 把可处理 size 从几千 vertices 推到了 1M vertices。下面我从 Fourier intuition 开始逐层 build up。

---

## 1. 从 Fourier 到 Manifold：核心 intuition

### 1.1 Fourier 的"真实身份"

经典 Fourier Transform 表面看是 sine/cosine 展开，但本质上它是 **Laplacian operator 的 eigenfunction 展开**。在 [0,1] 闭曲线上：

$$-\frac{\partial^2}{\partial x^2} H^{2k}(x) = k^2 \sin(kx) = k^2 H^{2k}(x)$$

这里 $H^{2k}(x) = \sin(kx)$ 是 eigenfunction，$k^2$ 是 eigenvalue。频率 $\omega_k = k = \sqrt{\lambda_k}$。Paper 中式 (1)–(2) 把这写成：

$$f(x) = \sum_{k=1}^{\infty} \tilde{f}_k H^k(x), \quad \tilde{f}_k = \langle f, H^k \rangle = \int_0^1 f(x) H^k(x)\, dx$$

变量含义：
- $H^k$：第 $k$ 个 basis function（circle harmonics，sin 或 cos）
- $\tilde{f}_k$：第 $k$ 个 Fourier coefficient（标量）
- $\langle \cdot, \cdot \rangle$：$L^2$ inner product

**关键 insight**：一旦你接受"basis = Laplacian 的 eigenfunctions"这件事，推广到任意 manifold 就变得自然 —— 只要把 $\partial^2/\partial x^2$ 换成 Laplace-Beltrami operator $\Delta_g$，把 inner product 换成 manifold 上的 $L^2$ inner product。这就是为什么这种东西叫 "shape harmonics" 或 "manifold harmonics"。

参考这个 intuition 在 ML 上的化身：graph spectral GCN（Bruna et al. 2013, https://arxiv.org/abs/1312.6203）做的就是把同一套搬到 graph Laplacian 上，每个 eigenvector 是 graph 上的"频率"。ChebNet (Defferrard et al. 2016, https://arxiv.org/abs/1606.09375) 再用 Chebyshev polynomial 避免 explicit 计算 eigenvectors，本质上和本文的 shift-invert 是两种不同的数值策略。

---

## 2. DEC 框架：为什么需要它

直接把 cotangent Laplacian 拿来做 spectral analysis 会出 bug。Figure 2 展示了 4 种不同 Laplacian 在一个左密右疏 mesh 上的 filter 结果，只有 (E) 正确。问题核心在 **symmetry**。

### 2.1 DEC 的基本对象

Discrete Exterior Calculus 把 smooth differential forms 离散化（参考 Desbrun et al. 2005 course notes: https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf 以及 Hirani 2003 thesis）。

- **k-simplex** $s_k$：k+1 个点的凸包。0-simplex = vertex, 1-simplex = edge, 2-simplex = triangle。
- **discrete k-form** $\omega^k$：在每个 oriented k-simplex 上给一个 real value。$\Omega^k(S)$ 是个 $n_k$ 维 vector space（$n_k$ 是 k-simplex 数量）。
- **exterior derivative** $d_k: \Omega^k \to \Omega^{k+1}$：signed adjacency matrix，$(d_k)_{s_k, s_{k+1}} = \pm 1$ 视 orientation 而定。
- **Hodge star** $\star_k$：diagonal matrix，元素 $|s_k^*|/|s_k|$，其中 $s_k^*$ 是 circumcentric dual。

对 surface mesh 具体值：

$$(\star_0)_{\nu\nu} = |\nu^*| \quad \text{(dual cell 面积)}$$
$$(\star_1)_{ee} = \frac{|e^*|}{|e|} = \cot\beta_e + \cot\beta'_e$$

这里 $\beta_e, \beta'_e$ 是 edge $e$ 两侧对面的两个角。注意 $\star_1$ 的几何意义就是 **edge 的 dual / primal 长度比**，恰好等于 cotangent 之和。

### 2.2 Laplace-de Rham 离散形式

Laplace-de Rham operator 在 0-forms 上定义为：

$$\Delta = -\star_0^{-1} d_1^T \star_1 d_0$$

把它写成显式 entries（论文 Section 2.2 的式子）：

$$\Delta_{ij} = -\frac{\cot(\beta_{ij}) + \cot(\beta'_{ij})}{|\nu_i^*|}, \quad \Delta_{ii} = -\sum_j \Delta_{ij}$$

变量：
- $\beta_{ij}, \beta'_{ij}$：edge $ij$ 两侧的对角
- $|\nu_i^*|$：vertex $i$ 的 Voronoi dual cell 面积

注意这个形式和 FEM 中 cotangent Laplacian **几乎一样**，差别在于 mass matrix：FEM 的 mass matrix 是 one-ring area 加总（非对角），DEC 的 $\star_0$ 是 dual cell area（diagonal）。这是 FEM "lumped mass approximation" 的天然产物。

### 2.3 对称化：为什么是关键

直接看上式，$\Delta_{ij} \neq \Delta_{ji}$，因为分母 $|\nu_i^*| \neq |\nu_j^*|$。这是灾难性的：非对称矩阵的 eigenvectors 不一定正交，不能做"Fourier transform"。

**Symmetrization trick**：canonical basis $\{\phi_i\}$ 在 $\star_0$-weighted inner product 下不正交（$|\nu_i^*|$ 不同），先 orthonormalize：

$$\bar{\phi}_i = \star_0^{-1/2} \phi_i$$

在新的 orthonormal basis 下，Laplacian 变成：

$$\boxed{\bar{\Delta} = \star_0^{-1/2}\, \Delta\, \star_0^{-1/2}, \quad \bar{\Delta}_{ij} = -\frac{\cot\beta_{ij} + \cot\beta'_{ij}}{\sqrt{|\nu_i^*|\,|\nu_j^*|}}}$$

这就是 Figure 2-E 用的 weight。它对称、半正定、eigenvectors 正交。

**为什么这个 symmetrization 在 intuition 上是对的**：你做的本质是把"unweighted Euclidean inner product 下的非对称算子"换到"weighted inner product 下的对称算子"。Mass matrix $\star_0$ 编码了 sampling density，把它从 operator 里 "exponent $\pm 1/2$" 出来，相当于把"非均匀采样"的几何信息均匀化。这个 trick 在 spectral graph theory 里也常见：normalized Laplacian $D^{-1/2} L D^{-1/2}$（Chung 1997）就是同一招。

参考 Wardetzky et al. 2007 "Discrete Laplace operators: No free lunch" (https://dl.acm.org/doi/10.2312/SGP/SGP07/061-070) 证明不存在同时满足所有"好性质"的 discrete Laplacian，所以这里是做了 trade-off：选 symmetry + geometry-aware，代价是没了 graph-Laplacian 的某些 combinatorial 性质。

---

## 3. Manifold Harmonics Basis (MHB) 与变换

### 3.1 构造流程

1. 组装对称 $\bar{\Delta}$
2. 解 eigenvalue problem：$\bar{\Delta} \bar{H}^k = \lambda_k \bar{H}^k$
3. 把 eigenvectors 从 orthonormal basis 映回 canonical basis：$H^k = \star_0^{-1/2} \bar{H}^k$

最终 $\{H^k\}$ 在 $\star_0$-weighted inner product 下 orthonormal。Figure 3 显示了几个 $H^k$，看起来确实很像 DCT 的 sine products，但它们是 surface-adaptive 的。

### 3.2 Manifold Harmonic Transform (MHT)

几何 $x$ 是 piecewise linear function，在 canonical basis 下展开（式 5）：

$$x = \sum_i x_i \phi^i$$

在 MHB 下展开（式 6）：

$$x = \sum_k \tilde{x}_k H^k$$

利用 orthonormality $\langle H^k, H^{k'}\rangle = \delta_{kk'}$，左右两边 inner product 上 $H^{k'}$，得到 forward transform（式 7）：

$$\boxed{\tilde{x}_k = \langle x, H^k \rangle = x^T \star_0 H^k = \sum_i x_i (\star_0)_{ii} H^k_i}$$

变量含义：
- $x_i$：vertex $i$ 的 x 坐标
- $(\star_0)_{ii} = |\nu_i^*|$：vertex $i$ 的 dual area（per-vertex mass weight）
- $H^k_i$：MHB 第 $k$ 个 basis function 在 vertex $i$ 的取值

注意 $(\star_0)_{ii}$ 出现这件事 —— 它就是 sampling density correction。在均匀 grid 上 $\star_0 = I$，退化成普通 dot product。

Inverse MHT 就是式 (6)：

$$x_i = \sum_k \tilde{x}_k H^k_i$$

Figure 4 展示了用前 $m$ 个 MHB 系数重建 dragon，前几十个就抓住 global shape，后续才补 details。

---

## 4. Filtering 与频率

### 4.1 频率的物理意义

对 closed curve，$\omega = k$，$\lambda = k^2 = \omega^2$。所以推广到 manifold：

$$\omega_k = \sqrt{\lambda_k}$$

Donnelly-Fefferman 1988 (https://link.springer.com/article/10.1007/BF01389312) 证明 nodal set（eigenfunction 零点集）长度 $\sim \sqrt{\lambda}$。所以 $\omega_k$ 的量纲是 inverse length —— 就是真正的 spatial frequency。

### 4.2 Filter 公式

每个 MHT coefficient $\tilde{x}_k$ 对应频率 $\omega_k$，filter 是函数 $F(\omega)$。直接做点积（论文式子在 3.2 节）：

$$x^F_i = \sum_{k=1}^m F(\omega_k)\, \tilde{x}_k\, H^k_i$$

### 4.3 High-frequency 残差处理

实际只计算到 cutoff $\omega_m$（论文里取 10× average edge length 的 inverse），高频信息丢了。但可以保存 residual：

$$x^{hf}_i = x_i - \sum_{k=1}^m \tilde{x}_k H^k_i$$

最终 filtered 结果（式 8）：

$$x^F_i = \sum_{k=1}^m F(\omega_k)\, \tilde{x}_k\, H^k_i + f^{hf}\, x^{hf}_i$$

其中 $f^{hf}$ 是 filter 在 $[\omega_m, \omega_M]$ 上的均值，$\omega_M$ 是 Nyquist 频率（half inverted edge length）。这一项当作"高频 wave packet" 整体处理，是个 pragmatic hack。

### 4.4 交互性来源

观察：MHB 和 MHT 只依赖 mesh 与几何，不依赖 filter。一旦 precompute 完，改变 $F(\omega)$ 只需重跑 inverse MHT，是 $O(nm)$ 而且可流式做。这就是 Figure 5 interactive slider 的来源。

---

## 5. Numerical Solution：让 1M vertices 可解的核心

### 5.1 直接求解的两个瓶颈

1. **Lower-end spectrum 难算**：iterative eigensolver（ARPACK/Arnoldi）擅长 spectrum 的"另一端"（large magnitude）。我们要 low $\lambda$（low frequency），是难的一端。直觉：low frequency = 高次 smoothing kernel 的 fixed point，condition number 很差。
2. **Eigenvector 数量爆炸**：要算几千个 eigenvectors，时间 superlinear in count，而且 1M vertices × 1000 eigenvectors = 1B floats ≈ 4GB，RAM 装不下。

### 5.2 Shift-Invert Spectral Transform

把 spectrum 做两个变换：

**Shift**：$\Delta_S = \bar{\Delta} - \lambda_S I$

**Invert**：解新 eigenproblem（式 10）：

$$\Delta_S^{-1} \bar{H}^k = \mu_k \bar{H}^k$$

数学上验证：如果原来 $\bar{\Delta} \bar{H}^k = \lambda_k \bar{H}^k$，那么

$$\Delta_S \bar{H}^k = (\lambda_k - \lambda_S)\bar{H}^k \implies \Delta_S^{-1} \bar{H}^k = \frac{1}{\lambda_k - \lambda_S}\bar{H}^k = \mu_k \bar{H}^k$$

关系：

$$\lambda_k = \lambda_S + \frac{1}{\mu_k}$$

变量：
- $\lambda_S$：shift 参数（中心频率 squared）
- $\mu_k$：shifted/inverted eigenvalue
- $\lambda_k$：原 Laplacian 的 eigenvalue

**妙处**：原 spectrum 中靠近 $\lambda_S$ 的 $\lambda_k$ → 在 $\Delta_S^{-1}$ 中 $\mu_k$ 很大 → Arnoldi 立刻收敛（找 large $\mu$ 是它的强项）。把 "找 small $\lambda$" 转化成 "找 large $\mu$"，正好把 iterative solver 的弱点变成强点。

### 5.3 实际求解：Cholesky Factorization

$\Delta_S$ 可能 indefinite（因为 shift 后部分 eigenvalue 变负）。所以用 **indefinite Cholesky**（LDL^T 类）做 sparse factorization：

$$\Delta_S = L D L^T$$

iterative solver 每次 query "给定向量 $\vec{\nu}$ 求 $\Delta_S^{-1}\vec{\nu}$"，就等价于解 $\Delta_S \vec{x} = \vec{\nu}$，用 back-substitution $O(\text{nnz})$ 完成。比起反复算 $\Delta_S^{-1}$ 显式形式快得多。

对 1M vertices 用 out-of-core symmetric indefinite factorization (Meshar-Irony-Toledo 2006, https://dl.acm.org/doi/10.1145/1149951.1149956)，即论文中提到 Sivan Toledo 提供的 TAUCS 未来版本。

### 5.4 Band-by-band 算法（论文 Algorithm 1 伪代码）

```
1. λ_S ← 0, λ_last ← 0
2. while (λ_last < ω_m^2)
3.   factorize Δ_S = Δ̄ - λ_S·Id        (OOC Cholesky)
4.   compute 50 first eigenpairs (H̄^k, μ_k) of Δ_S^{-1}   (Arnoldi)
5.   for k=1 to 50
6.     λ_k ← λ_S + 1/μ_k
7.     if λ_k > λ_last: write (H̄^k, λ_k) to disk
8.   end
9.   λ_S ← max(λ_k) + 0.4·(max(λ_k) - min(λ_k))   (overlap 40%)
10.  λ_last ← max(λ_k)
11. end
```

**关键设计**：
- 每个 band 算 50 个 eigenpairs，band 间 overlap 40%，保证不漏
- 写入 disk（streaming），不占内存
- 每 band 时间 = factorization（一次，可 OOC）+ Arnoldi 几十次
- 总时间 **linear in number of eigenpairs**，破解了 superlinear 瓶颈

### 5.5 Limited-memory Filtering（不存 MHB）

观察一个代数恒等式：

$$x^F = \sum_k F(\omega_k)\, \tilde{x}_k H^k = \sum_k F(\omega_k)\, \langle x, H^k \rangle H^k$$

而

$$x = \sum_k \langle x, H^k \rangle H^k$$

所以

$$x^F = x + \sum_k (F(\omega_k) - 1)\, \langle x, H^k \rangle H^k$$

伪代码：

```
1. x^F ← x
2. for each eigenpair (H^k, ω_k):
3.   x^F ← x^F + (F(ω_k) - 1) · (x^T ⋆_0 H^k) · H^k
4. end
```

**精妙处**：一次只 load 一个 eigenvector 到内存，用完丢掉。内存 $O(n)$，不是 $O(nm)$。结合 shift-invert，只算 filter 实际修改的 band，不碰其它 band。

---

## 6. 实验数据表（Table 1 重排）

| Model | n (vertices) | m (eigenpairs) | MHB | MHT | MHT$^{-1}$ | LM-filt |
|-------|-------|------|-----|------|-----------|----------|
| dinoFig.2 | 56K | 447 | 77s | 0.34s | 0.53s | 18s |
| dragoFig.1 | 150K | 315 | 160s | 0.65s | 1.02s | 41s |
| drago1* (OOC MHT) | 244K | 667 | 9m | 18s | 4s | 135s |
| drago2** (OOC both) | 500K | 800 | 2h21m | 32s | 48s | 28m |
| drago3** (Fig.6) | 1M | 1331 | 6h | 76s | 85s | 1h |

读这张表的几个直觉：

1. **MHB 计算时间 ~ $n^{1.5}$ 左右**：从 150K 到 1M（6.7×），时间从 160s 到 6h（135×），约 $n^{1.4\sim1.5}$，符合 sparse Cholesky factorization scaling。
2. **MHT/MHT$^{-1}$ ~ $O(nm)$**：1M × 1331 ≈ 1.3B 乘加，76s 是合理的 throughput。
3. **LM-filt 在 filter 只覆盖 1/4 spectrum 时，比完整 MHB+MHT+MHT$^{-1}$ 快**：因为只算需要的 band。
4. **1M vertices 6h MHB 是主要瓶颈**，论文最后也承认了，future work 提到 multiresolution。

---

## 7. 与相关工作的 positioning

### 7.1 对比 Taubin 1995 (https://dl.acm.org/doi/10.1145/218380.218484)

Taubin 把 graph Laplacian 的 eigenvectors 当 DCT 类比，但只是用作 **理论分析工具** 设计 filter approximations（$\sum c_k L^k$ 这种 polynomial filter）。本文真正把 eigenbasis 算出来用。

### 7.2 对比 Kim & Rossignac GeoFilter 2005 (https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2005.00863.x)

GeoFilter 用 implicit + explicit scheme 组合，每个 frequency band 用不同方法。优点：快。缺点：filter 类别受限；低频段需要 $\bar{\Delta}^{-k}$，inverse 不 sparse，性能差。本文在低频段比它好。

### 7.3 对比 Spherical Harmonics 方法（Zhou et al. 2004, Mousa et al. 2006）

Spherical harmonics 要 resample 到球面 / star-shape，且只对 genus 0。本文任意 topology 都行，无 resampling。代价是 MHB 要 precompute。

### 7.4 对比 Sorkine et al. Geometry-aware bases 2005 (https://ieeexplore.ieee.org/document/1543431)

他们用 least-squares 定义的 "geometry-aware bases"，本质上不是 Laplacian 的 eigenvectors，所以频率物理意义不直接。本文 eigenvectors of symmetric Laplacian，频率 = $\sqrt{\lambda_k}$，对应 nodal set length。

### 7.5 与 ML 中 spectral GCN 的关系

- Bruna et al. 2013 (https://arxiv.org/abs/1312.6203)：spectral CNN，直接用 graph Laplacian eigenvectors 做 filter，但因为 eigenvectors 计算 cost + non-locality 限制，scalability 差。
- Defferrard et al. 2016 ChebNet (https://arxiv.org/abs/1606.09375)：用 Chebyshev polynomial 在 $\lambda$ 上逼近 $F(\lambda)$，避免 explicit eigenvectors，scalable。
- 本文正好是"explicit eigenvector computation"路线在 geometry processing 上的极致优化（band-by-band + OOC），属于 Bruna 风格。但 ML 界后来倾向 ChebNet 风格，因为 ML 要端到端训练，filter 是 learned 不是 designed，所以 polynomial approximation 更方便 backprop。

### 7.6 与 Diffusion Models / Heat Kernel 的关系

Heat kernel $K_t(x, y)$ 的谱分解：

$$K_t(x, y) = \sum_k e^{-\lambda_k t} H^k(x) H^k(y)$$

所以 diffusion filtering 就是用 $F(\omega_k) = e^{-\lambda_k t} = e^{-\omega_k^2 t}$ 做低通。这正是 paper Figure 5 low-pass 的连续版本。

更广义地，**所有 manifold 上的 PDE-based smoothing、diffusion、wave propagation 都可以在 MHB 下对角化**。Score-based generative models 在 manifold 上的 Langevin dynamics，$dx_t = -\nabla \log p(x_t) dt + \sqrt{2} dB_t$，背后的 generator 就是 Laplacian，谱分解正好用 MHB。参考 Song et al. 2021 score-based (https://arxiv.org/abs/2011.13456) 在 Euclidean 上的形式，manifold 推广（如 score-based generative models on implicit manifolds, https://arxiv.org/abs/2202.03013）天然用到这套 basis。

### 7.7 Shape Descriptors：HKS, WKS

- **Heat Kernel Signature (HKS)** Sun et al. 2009 (https://dl.acm.org/doi/10.1145/1531326.1531336)：用 $K_t(x, x) = \sum_k e^{-\lambda_k t} (H^k(x))^2$ 作为 per-vertex feature。
- **Wave Kernel Signature (WKS)** Aubry et al. 2011 (https://dl.acm.org/doi/10.1145/2010324.1966442)：用 wave packet 替代 heat kernel。

本文的 MHB 是这些 descriptor 的基础。

### 7.8 Functional Maps

Ovsjanikov et al. 2012 (https://dl.acm.org/doi/10.1145/2185520.2185526) 的 functional map 在 reduced basis 上做 shape correspondence，basis 选的就是 Laplacian eigenvectors。本文提供了 scalable 的 MHB 计算，正好是 functional map pipeline 的 precompute 步骤。

---

## 8. Limitations 与 paper 自己的 honest 承认

1. **MHB 不 spatially localized**：每个 eigenvector 是 global sine wave。重建 geometry 需要很多 coefficients（Figure 4）。JPEG 用 small block 避免 global DCT 这个问题，本文也提到 partitioning 能部分解决，但损失 continuity。
2. **Future work: Manifold Wavelets**（Grinspun et al. 2002 CHARMS, https://dl.acm.org/doi/10.1145/566570.566594）：需要在 frequency 和 spatial 上同时 localized。这预言了后来 Lifting scheme 在 manifold 上的工作。
3. **Creases 处理**：Fourier-like 方法在 sharp edges 上 Gibbs ringing。Anisotropic Laplacian 可能改善 frequency localization。
4. **1M+ vertices 6h 计算太慢**，future work 提 multiresolution。
5. **Mesh compression 不好用**：MHB geometry-dependent，而 Karni-Gotsman compression 需要 combinatorial Laplacian 保证 connectivity-invariance。本文方法要换 combinatorial Laplacian 才能用于 compression。

---

## 9. 总体直觉总结（build intuition 视角）

1. **Manifold Harmonics = surface-adaptive sine/cosine**，是 Laplace-Beltrami 的 eigenfunctions。频率 = $\sqrt{\lambda_k}$，对应 nodal set 长度的 inverse。
2. **DEC + $\star_0^{-1/2}$ symmetrization** 解决了 cotangent Laplacian 的 symmetry 问题，本质是 "把 mass matrix 从 operator 里挤出去到 inner product 里"。
3. **Shift-invert spectral transform** 把"找 small eigenvalue"转化为"找 large inverse-shifted eigenvalue"，绕开 Arnoldi 的弱点。
4. **Band-by-band + Cholesky factorization + OOC** 把 superlinear 难题切成 linear 的 band 序列，scalability 推到 1M vertices。
5. **Limited-memory filtering** 用代数恒等式 $x^F = x + \sum_k (F-1)\langle x, H^k\rangle H^k$ 把内存从 $O(nm)$ 压到 $O(n)$。
6. **交互性来源**：MHB 和 MHT 不依赖 filter，precompute 后改 filter 只重做 inverse MHT，是 streaming $O(nm)$。

---

## 参考链接汇总

- **Paper PDF**: https://dl.acm.org/doi/10.2312/SGP/SGP08/251-259 (SGP 2008)
- **Bruno Lévy 工作主页 (Manifold Harmonics)**: https://alice.loria.fr/index.php/software/4-manifold-harmonics/20-manifold-harmonics.html
- **DEC course notes (Desbrun et al.)**: https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf
- **Discrete Laplacian no free lunch (Wardetzky et al.)**: https://dl.acm.org/doi/10.2312/SGP/SGP07/061-070
- **Taubin 1995**: https://dl.acm.org/doi/10.1145/218380.218484
- **Karni-Gotsman Spectral Compression 2000**: https://dl.acm.org/doi/10.1145/344779.344934
- **HKS (Sun et al. 2009)**: https://dl.acm.org/doi/10.1145/1531326.1531336
- **WKS (Aubry et al. 2011)**: https://dl.acm.org/doi/10.1145/2010324.1966442
- **Functional Maps (Ovsjanikov et al. 2012)**: https://dl.acm.org/doi/10.1145/2185520.2185526
- **Spectral CNN (Bruna et al. 2013)**: https://arxiv.org/abs/1312.6203
- **ChebNet (Defferrard et al. 2016)**: https://arxiv.org/abs/1606.09375
- **Score-based Diffusion (Song et al. 2021)**: https://arxiv.org/abs/2011.13456
- **Diffusion on implicit manifolds**: https://arxiv.org/abs/2202.03013
- **Donnelly-Fefferman Nodal Sets 1988**: https://link.springer.com/article/10.1007/BF01389312
- **OOC Cholesky (Meshar-Irony-Toledo 2006)**: https://dl.acm.org/doi/10.1145/1149951.1149956
- **CHARMS (Grinspun et al. 2002)**: https://dl.acm.org/doi/10.1145/566570.566594
- **Geometry-aware bases (Sorkine et al.)**: https://ieeexplore.ieee.org/document/1543431
- **Spectral surface quadrangulation (Dong et al. 2006)**: https://dl.acm.org/doi/10.1145/1179352.1141966

---

## 10. 后续值得 follow 的方向

1. **Multiresolution MHB**：论文 future work 提到。对应后来 Hodge decomposition on multiresolution meshes 的工作。
2. **Manifold Wavelets**：同时 localized in frequency and spatial。可以看 Huguet et al. 2023 "Manifold Harmonics for Networks" 这类现代延伸。
3. **Anisotropic MHB**：用 anisotropic Laplacian $\nabla \cdot (G \nabla)$，$G$ 是 tensor field。能保留 creases。Andrle & Crane 2024 等近期工作。
4. **神经 manifold operator**：把 MHB 计算替换成 neural operator（FNO 风格），在 fixed mesh family 上 amortize。可以看 Li et al. 2020 FNO (https://arxiv.org/abs/2010.08895) 和它在 surface PDE 上的衍生。
5. **Diffusion models + MHB**：把 score model 训练成 MHB coefficients 的 denoiser，比 vertex-space score 更 sparse。早期 symbol: Latent Diffusion (https://arxiv.org/abs/2112.10752) 在 pixel latent 上做，类似思路搬到 manifold spectral latent。

希望这个 walk-through 帮你 build up intuition。这篇 paper 的 elegant 之处在于它把 DEC、spectral transform、numerical linear algebra 三个看似独立的工具用一个目标（在 manifold 上做 Fourier-like filtering）串起来，每个 trick 都对应一个具体的物理 / 数学 obstruction，特别适合用来理解 "geometry processing 中的 signal processing" 这整套思维框架。
