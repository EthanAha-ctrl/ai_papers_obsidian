---
source_pdf: GEOMETRIC LAPLACE NEURAL OPERATOR.pdf
paper_sha256: ef28d3edccfba57940a59adeff8e129c7249ee5cf0033fc0751a2661350e90e4
processed_at: '2026-08-04T21:17:44-07:00'
target_folder: PINN
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GLNO 用人话说一遍

Andrej，我把这篇 paper 拆成"给一个懂数学但没读过 operator learning 的人"能听懂的版本。技术细节都保留，但换成 intuition 优先的讲法。

---

## 一、这篇 paper 到底干了什么

一句话：**教 neural network 怎么在"弯弯曲曲的几何表面"上做卷积，同时还能处理"会衰减、会消散"的物理信号**。

稍微展开一点：传统 neural operator 比如 FNO 做的事，是"把函数变到 Fourier 域做乘法再变回来"。这有两个前提：(1) 信号是周期的，(2) 定义在规整网格上。这篇 paper 把这两个前提都拆掉了。它构造了一个新的 basis，既能表达指数衰减，又能直接定义在任意 manifold 的 eigenfunction 上。

类比一下：

- **FNO** 像是在一维时间轴上放一堆永不停歇的三角波 $\cos(\omega t), \sin(\omega t)$，然后说"任何信号都能用这些拼出来"。
- **LNO** 说："物理信号很多会衰减啊，我加一堆 $e^{-\sigma t}\cos(\omega t)$ 这种阻尼振荡，能更好地拟合 transient。"
- **GLNO** 说："但 LNO 还得用 FFT，必须规整网格。我把 $e^{-i\omega t}$ 换成 manifold 上的 LBO eigenfunction $\phi_\omega(x)$，把 $e^{-\sigma t}$ 换成沿某个几何坐标 $\mathcal{P}(x)$ 的指数衰减，这样就能在任意曲面上做同样的事。"

理解这条演化路径，paper 的全部内容就顺了。

---

## 二、为什么 FNO 不够用：两个痛点用大白话讲

### 痛点 1：Fourier basis 假设周期性

Fourier basis 是 $\{e^{-i\omega t}\}$，即 $\{\cos(\omega t), \sin(\omega t)\}$。这些函数永远振荡，永远不衰减，永远不增长。

现实物理信号大量是 transient（瞬态）的：
- 阻尼摆：$\sin(\omega t) e^{-ct}$，振幅随时间指数衰减
- 扩散方程：$e^{-k t}$，单调衰减到稳态
- 热传导：初始 hot spot 平滑化
- 冲击波：localized spike

这些信号要 Fourier 近似得好，需要非常多高频项去 cancel 掉不衰减的振荡，制造出"看起来衰减"的效果。系数衰减慢，效率低。

打个比方：你要画一条"从 1 平滑降到 0 的曲线"，用一堆永不停歇的正弦波叠加，数学上能做（Fourier 收敛），但要叠加很多项。但你用 $e^{-t}$ 一个函数就搞定了。

### 痛点 2：FFT 要求规整网格

FFT 需要函数定义在等距 grid 上。所以 FNO 处理 PDE 时，domain 必须是 $[0,L]^d$ 这样的矩形 grid。

后续工作想处理曲面：
- **Spherical FNO** 用 spherical harmonics 替代 Fourier，能处理球面（地球大气）
- **Geo-FNO** 把不规则 mesh "warp" 回规整 grid，再做 FFT
- **GINO / Graph Neural Operator** 用 message passing 在 mesh 上跑，但 cost 高、interpretability 差

但所有这些都受限于"surface 是 structured 的"或"得 deform 回 grid"。对于真正的 arbitrary topology（比如 RNA 分子表面、人体表面、汽车外形），上面这些方法都不直接适用。

GLNO 想做到：给一个 mesh（任意拓扑、任意顶点数），直接在它上面跑 spectral operator，连 resampling 都不需要。

---

## 三、LNO 的核心 trick 用大白话讲

LNO 的 idea 是：把 kernel 直接在 Laplace domain 参数化成 pole-residue 形式。

什么是 pole-residue？你把一个有理函数写成：

$$K_\theta(s) = \sum_{n=1}^N \frac{\beta_n}{s - \mu_n}$$

变量解释：
- $s \in \mathbb{C}$：Laplace domain 的复频率变量
- $n$：第 $n$ 个 pole 的 index，$n = 1, \ldots, N$
- $\mu_n \in \mathbb{C}$：第 $n$ 个 learnable pole，复数，$\text{Re}(\mu_n)$ 控制 decay/growth rate，$\text{Im}(\mu_n)$ 控制振荡频率
- $\beta_n \in \mathbb{C}$：第 $n$ 个 learnable residue，决定该 mode 的振幅

为什么这个 trick 牛？因为：

1. **Inverse Laplace 有解析解**：$\mathcal{L}^{-1}\{\frac{\beta_n}{s-\mu_n}\} = \beta_n e^{\mu_n t}$。所以 kernel 在时间域直接是 $\sum_n \beta_n e^{\mu_n t}$，一组指数/阻尼振荡函数，天生就能表达 transient。

2. **Convolution theorem**：时域卷积 = Laplace 域乘法。所以 $(\kappa * f)(t)$ 变成 $K_\theta(s) \cdot F(s)$，pure algebra。

3. **Pole 决定 mode，residue 决定 amplitude**：网络只需学 $N$ 个复数 pole + $N$ 个复数 residue，就能表达任意 $N$ 阶 LTI 系统的 impulse response。极简，极物理。

LNO 的 limitation 在哪里？输入分解 $f(t) = \sum_\omega \alpha_\omega e^{i\omega t}$ 还是用 FFT，所以输入端仍是周期假设 + 规整网格。它只是把 kernel 端做灵活了，input 端没动。

GLNO 把这两端都做了改造，是 paper 的核心创新。

---

## 四、GLNO 的核心 idea：把"时间轴"搬到 manifold 上

这是 paper 最关键的 intuition。

Euclidean Laplace basis 是：

$$\varepsilon_z(t) = e^{-zt} = e^{-\sigma t} \cdot e^{-i\omega t}$$

两个因子：
- $e^{-i\omega t}$：周期振荡，把"时间轴 $t$"映射到"频率 $\omega$"
- $e^{-\sigma t}$：指数衰减，沿"时间轴 $t$"的阻尼

GLNO 做的替换：

$$\varepsilon_z(x) = e^{-\sigma \mathcal{P}(x)} \cdot \phi_\omega(x)$$

- $e^{-i\omega t} \to \phi_\omega(x)$：把"沿时间轴的振荡"换成"沿 manifold 的 LBO eigenfunction 振荡"
- $e^{-\sigma t} \to e^{-\sigma \mathcal{P}(x)}$：把"沿时间轴衰减"换成"沿 manifold 上某个几何坐标 $\mathcal{P}$ 的衰减"

这里 $\mathcal{P}: \mathcal{M} \to \mathbb{R}$ 是 paper 说的"intrinsic geometric properties"，比如 curvature、boundary distance。它扮演了"time-like coordinate"的角色。

打个比方：原来 Laplace 变换是"信号沿时间轴衰减，沿时间轴振荡"。GLNO 把"时间轴"换成 manifold 上一个有意义的几何方向（比如沿曲率的方向），然后"信号沿这个几何方向衰减，沿 LBO eigenfunction 振荡"。Laplace 变换的代数结构完全保留，只是物理含义从"时间"换成了"几何"。

这是为什么 paper 把这个方法叫"Geometric Laplace"——Laplace 还是那个 Laplace，只是定义域从时间轴变成了 manifold 上的几何坐标。

---

## 五、为什么 Laplace-Beltrami eigenfunction 是"manifold 上的 Fourier"

LBO 是 manifold 上的 Laplacian 算子。它的特征值问题：

$$-\Delta_\mu \phi_{\omega_k} = \lambda_k \phi_{\omega_k} = \omega_k^2 \phi_{\omega_k}$$

变量：
- $\Delta_\mu$：manifold $(\mathcal{M}, \mu)$ 上的 Laplace-Beltrami operator
- $\phi_{\omega_k}$：第 $k$ 个 eigenfunction，是 manifold 上的标量函数
- $\lambda_k \geq 0$：第 $k$ 个 eigenvalue
- $\omega_k = \sqrt{\lambda_k}$：eigenfrequency
- $k$：mode index，从 1 开始，$0 = \omega_1 < \omega_2 \leq \omega_3 \leq \cdots$

为什么 LBO eigenfunction 是"manifold 上的 Fourier mode"？看几个例子就懂了：

- 在 flat line $\mathbb{R}$ 上，LBO 就是 $\frac{d^2}{dt^2}$，eigenfunction 是 $e^{i\omega t}$，就是经典 Fourier
- 在 flat torus $\mathbb{T}^2$ 上，eigenfunction 是 $e^{i(k_1 x + k_2 y)}$，二维 Fourier
- 在 sphere $S^2$ 上，eigenfunction 是 spherical harmonics $Y_l^m$，所以 Spherical FNO 用 $Y_l^m$ 做 spectral basis
- 在任意 mesh 上，LBO eigenfunction 没有 closed form，但可以数值求（用 cotangent Laplacian 离散化后做 eigendecomposition）

关键性质：**LBO eigenfunction 完全由 manifold 内蕴几何决定，与具体 mesh discretization 无关**。同一个 manifold，你用 1000 个顶点 mesh 它，还是用 5000 个顶点 mesh 它，eigenfunction（作为 manifold 上的函数）是同一个，只是离散表示不同。

这就是 GLNO 的 "grid invariance" 的根源。你的训练 mesh 和测试 mesh 可以顶点数完全不同、连接关系完全不同，只要它们 discretize 的是"同一个 manifold"，operator 学到的就是 manifold 上的 operator，不是 mesh 上的 operator。

参考：
- [Laplace-Beltrami operator 基础](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator)
- [The Heat Method for shape analysis (Crane 2013)](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)
- [DiffusionNet (Sharp 2022)](https://dnaralab.github.io/DiffusionNet/)

---

## 六、最微妙的 trick：Inverse Laplace 的 Gaussian projection

这是 paper 最聪明、也最"engineering"的地方。

正向走：input $f(x) \to$ Laplace domain $F(s) \to$ 乘 kernel 得 $G(s) \to$ 逆变换回 $g(x)$。

逆向走时遇到一个问题：kernel 的 pole $\mu_n \in \mathbb{C}$ 是网络 learnable 的连续值，但 manifold 上 LBO 谱是离散的 $\{\omega_k\}_{k=1}^\infty$。当 $\mu_n = 0.3 + 2.7i$ 时，$2.7$ 这个频率不会恰好等于某个 $\omega_k$。

如果硬要把 $\mu_n$ 的 imaginary part snap 到最近的 $\omega_k$，会损失信息。如果完全保留连续 pole，又没法在离散 LBO basis 上表达。

GLNO 的解法：用 Gaussian filter 做软分配。

$$\mathcal{L}^{-1}\left\{\frac{1}{s - \mu_n}\right\} = \exp(\text{Re}(\mu_n) \mathcal{P}(x)) \cdot \sum_\omega \text{Gauss}(\text{Im}(\mu_n) - \omega) \phi_\omega(x)$$

其中：

$$\text{Gauss}(x) = \frac{1}{\sqrt{2\pi}\Sigma} \exp\left(-\frac{x^2}{2\Sigma^2}\right)$$

变量：
- $\mu_n$：第 $n$ 个 learnable kernel pole
- $\text{Re}(\mu_n)$：pole 的实部，控制 decay/growth rate，直接作用在 $\mathcal{P}(x)$ 上
- $\text{Im}(\mu_n)$：pole 的虚部，是想要的目标频率
- $\omega$：离散 LBO eigenfrequency，遍历所有 mode
- $\phi_\omega(x)$：对应的 eigenfunction
- $\Sigma$：Gaussian 的 bandwidth

直觉：
- 实部 $\text{Re}(\mu_n)$ 完全保留，直接作用在 $\mathcal{P}$ 上做指数衰减
- 虚部 $\text{Im}(\mu_n)$ 用 Gaussian 加权分配到附近的几个 $\omega$ mode 上
- 如果 $\text{Im}(\mu_n)$ 恰好等于某个 $\omega_k$，Gaussian 集中在该 mode，hard assignment
- 如果 $\text{Im}(\mu_n)$ 介于两个 $\omega$ 之间，两个 mode 都被激活，soft interpolation

这相当于在离散 LBO 谱上做 RBF interpolation。$\Sigma$ 控制"软硬程度"：$\Sigma$ 大就糊一片（多个 mode 激活），$\Sigma$ 小就聚焦到一个 mode。

类比：你有一个连续频率 2.7，离散频率是 $\{2, 3, 4, \ldots\}$。硬 snap 就是 round 到 3，损失信息。Soft assignment 就是给 2 分配权重 $\text{Gauss}(0.7)$，给 3 分配权重 $\text{Gauss}(-0.3)$，给 4 分配权重 $\text{Gauss}(-1.3)$，等等。如果 $\Sigma = 0.5$，权重主要集中在 3 和 2，4 几乎为 0。

这个 trick 让网络保留了 continuous spectral learning 的表达力（任意复频率都能学），同时遵守了 manifold 离散谱的物理约束（必须用 LBO eigenfunction 表达）。Paper 没明确说 $\Sigma$ 怎么选，我猜是 hyperparameter，可能跟 mesh resolution 有关。

---

## 七、整体网络架构（GLNONet）

```
Input f: M → R^{d_in}        # 任意 mesh M 上的函数
   ↓
Geometric features           # 用 SciPy 算 LBO eigenpairs, curvature, boundary distance
   ↓
Encoder MLP P                # R^{d_in} → R^{d_latent}, 升维
   ↓
GLNO Block 1 → Block 2 → ... → Block L    # 核心 operator layer 堆叠
   ↓
Decoder MLP Q                # R^{d_latent} → R^{d_out}, 降维
   ↓
Output u: M → R^{d_out}
```

每个 GLNO Block 内部的流程：

1. **Decomposition**：用 learnable $\{\sigma_i\}_{i=1}^M$ 和 LBO eigenfunction $\{\phi_{\omega_i}\}_{i=1}^M$ 把 input 分解成 generalized Laplace coefficients $\{\alpha_i\}$
2. **Forward Laplace**：把 $\alpha_i$ 送进 Laplace domain，得到 $F(s) = \sum_i \frac{\alpha_i}{s + z_i}$
3. **Pole-residue multiplication**：乘 kernel $K_\theta(s) = \sum_n \frac{\beta_n}{s - \mu_n}$，做 partial fraction 拆成 steady + transient
4. **Inverse Laplace with Gaussian**：用上面那个 Gaussian projection 把 continuous pole 软映射回 LBO basis
5. **Feature fusion**：把 spectral output 和原始 geometric feature（curvature 等）用 lightweight MLP 融合 + skip connection

参数 hyperparameter 在 Table 5：

| Task | Blocks | Channels | Poles | Sigma | LR | Epoch |
|---|---|---|---|---|---|---|
| Poisson | 4 | 64 | 1 | 1 | 1e-3 | 300 |
| Shape-Net Car | 4 | 64 | 1 | 4 | 5e-4 | 300 |
| SHREC-11 | 4 | 64 | 4 | 4 | 5e-4 | 200 |
| RNA | 4 | 64 | 1 | 1 | 1e-3 | 300 |
| Human Body | 4 | 64 | 1 | 1 | 1e-3 | 300 |

总参数 0.13M-0.21M，远小于 LSM 的 21.68M。Paper 在 Table 4 详细列了 efficiency。

---

## 八、实验结果用人话读

### 8.1 1D ODE/PDE (Table 2)

挑几个有代表性的：

**Driven Pendulum（带阻尼摆）**：方程 $\ddot{x} + c\dot{x} + \sin(x) = f(t)$
- $c = 0$（无阻尼，纯振荡）：FNO 0.367，LNO 0.154，GLNO **0.088**
- $c = 0.5$（带阻尼，transient 衰减）：FNO 0.609（炸了），LNO 0.172，GLNO **0.097**

观察：阻尼越大，FNO 越惨。因为 Fourier basis 拟合 $e^{-0.5t}\sin(t)$ 需要非常多高频项。LNO 已经好很多，GLNO 又进一步。

**Lorenz system（混沌系统）**：
- $\rho = 5$：FNO 0.019，LNO 0.506（惨），CNO 0.005（最好），GLNO 0.036
- $\rho = 10$：FNO 0.482，LNO 0.583（惨），CNO 0.492，GLNO 0.248

观察：Lorenz 是 chaotic + localized dynamics，CNO（local 卷积）反而最好。GLNO 不是 SOTA 但还算能打。说明 spectral method 的 global bias 对高度 localized chaos 不占便宜。Paper 也很诚实地承认了这一点。

**Diffusion equation（扩散方程）**：
- FNO 0.0064，LNO 0.0011，GLNO **0.0006**

Diffusion 是纯衰减动力学，最对 GLNO 胃口，比 LNO 又好了一倍。

### 8.2 Unstructured mesh Poisson (Table 3)

$$-\Delta u = f \text{ in } \Omega, \quad u|_{\partial\Omega} = 0$$

不规则 mesh 上的 Poisson 方程。GLNO relative error 0.0044，Geo-FNO 0.0049（FNO 的 unstructured 变种），GINO 0.1623（graph method，惨）。

结论：LBO-based 架构在 unstructured grid 上确实有 intrinsic 优势，不需要 warp 回规整 grid。

### 8.3 Real-world classification/segmentation

**SHREC-11（30 类 shape 分类，每类 10 样本）**：
- GLNO 99.7%，DiffusionNet 99.4%，GNOT 35.3%，Transolver 43.8%
- 其他所有 neural operator 都很差（30-60%），因为它们没真正利用 intrinsic geometry

**RNA surface segmentation**：
- GLNO 90.1%，AMG 88.5%，DiffusionNet 85.6%，Transolver 86.0%

**Human body segmentation**：
- GLNO 91.0%，DiffusionNet 90.3%，AMG 62.5%，GNOT 66.2%

GLNO 在 RNA 和 Human 上都 best，而且高曲率区域（手指、脚趾、面部）尤其好，因为 non-periodic basis $e^{-\sigma \mathcal{P}}$ 能捕捉这些 sharp feature。

### 8.4 Shape-Net Car（CFD 模拟）

输入车表面流体量，预测下一时刻流体量。
- Pressure prediction：GLNO 0.0960 best，AMG 0.0978 次优
- Velocity prediction：GLNO 0.1037，AMG 0.0919 best

GLNO 在 velocity 稍逊 AMG。Paper 解释是 Shape-Net Car 的几何变化相对单一（都是车），AMG 这种专门方法能 fit 得更好。但 GLNO 作为 general-purpose 方法接近 SOTA 已经够说明问题。

### 8.5 Ablation：去掉 $\sigma$ 会怎样

| Variant | Poisson | SHREC-11 | RNA | Human |
|---|---|---|---|---|
| GLNO | 0.0044 | 99.7% | 90.1% | 91.0% |
| GLNO w/o $\sigma$ | 0.0087 | 95.5% | 82.2% | 88.5% |

去掉 $\sigma$（即去掉 exponential decay basis，只剩 LBO eigenfunction）后，所有 task 都退化。最严重是 RNA（-7.9%）。这证明 non-periodic basis 是关键 component，光靠 LBO 不够。

这个 ablation 其实很 informative：它告诉我们 GLNO 的成功不是"用 LBO 替代 FFT"这么简单，而是"LBO + exponential damping"的组合才 work。光用 LBO 就退化成类似 DiffusionNet 的东西，确实能达到不错 baseline，但 transient 表达力不够。

---

## 九、理论分析用人话讲

### 9.1 Computational complexity (Appendix B.5)

总 cost：

$$C_{\text{forward}} = O\Big(D(MK\log K + MN + (M+N)K)\Big) \approx O(DMK\log K)$$

变量：
- $K$：discretization 点数（mesh 顶点数或时间步数）
- $M$：generalized Laplace basis 数（input 端 mode 数）
- $N$：kernel pole 数（kernel 端 learnable mode 数）
- $D$：channel dimension

主导项是 $O(DMK\log K)$，来自 decomposition 阶段的 FFT（每个 $\sigma_i$ 一次 FFT）。和 FNO 的 $O(DK\log K)$ 同阶，多了 $M$ 因子。$M$ 通常不大（几个到几十个），所以 cost 增加可接受。

### 9.2 Approximation bound (Appendix B.6)

误差拆成两部分：

$$\|\mathcal{T}_* - \mathcal{T}_\theta\| \leq \varepsilon_{\text{basis}}(M,N) + \varepsilon_{\text{par}}(d_\theta)$$

- $\varepsilon_{\text{basis}}(M,N)$：用 $M+N$ 个 exponential function 逼近真实 kernel 的 best error
- $\varepsilon_{\text{par}}(d_\theta)$：有限参数化的逼近 error

如果真实 kernel $k_* \in H^\alpha$（Sobolev 正则性 $\alpha$），经典 exponential approximation 给：

$$\varepsilon_{\text{basis}}(M,N) \leq C_1 (M+N)^{-\alpha}$$

这是 Sobolev-type rate。对 transient system 来说，比 Fourier 的 polynomial decay 更适合（因为 Fourier basis 表达衰减需要很多项 cancel）。

### 9.3 Sample complexity (Appendix B.7)

Rademacher complexity：

$$\Re_N(\mathcal{H}_{\text{GLNO}}) = O\left(\sqrt{\frac{M + d_\theta}{N}}\right)$$

达到精度 $\epsilon$ 所需样本：

$$N_{\min}^{\text{GLNO}} = O\left(\frac{M + d_\theta}{\epsilon^2}\right)$$

线性 scaling，与经典 operator learning 一致。$M$ 是 basis 数，$d_\theta$ 是 parameter 数。这意味着 basis 数 $M$ 越大，需要的样本越多，trade-off 很 clear。

---

## 十、几个我自己想到的问题和联想

### 10.1 $\mathcal{P}$ 的选择有多 robust？

Paper 用 curvature、boundary distance 作为 $\mathcal{P}$，但定理 B.4 要求 $A(\mathcal{P})$（$\mathcal{P}$ 的 level set 面积）变化 sub-exponential。这是个隐含约束。

举个例子：如果 manifold 是一个细长的"管子"，$\mathcal{P}$ 取沿长轴的弧长坐标，那 level set 面积基本是常数（截面圆面积），定理 B.4 严格成立。

如果 manifold 是个 funnel 形状，$\mathcal{P}$ 是沿轴向距离，那 level set 面积随 $\mathcal{P}$ 指数变化，定理 B.4 失效，Laplace transform $\to \frac{1}{s+z}$ 的 mapping 不再精确。

Paper 没系统研究 $\mathcal{P}$ 选择对精度的影响。这是个明显的 limitation。理论上能 learn $\mathcal{P}$ 吗？比如把 $\mathcal{P}$ 本身参数化为某个 MLP on manifold？这可能是 future work 的方向。

### 10.2 LBO eigendecomposition 的 cost

LBO eigendecomposition 在大 mesh 上 expensive。直接求 $O(n^3)$，iterative 求 $O(n^2)$ per mode。Paper 用 SciPy，没具体说 mesh 大到多少就不可行。

实际 mesh 比如 10000 顶点，求前 100 个 eigenpair 是可接受的（几秒到几十秒）。但 100k+ 顶点就麻烦了。这是 spectral method 的通病。

解决思路：
- 用 [Heat Method (Crane 2013)](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/) 近似 eigenfunction，不求精确解
- 用 randomized SVD 求前几个低频 mode
- 用 multiresolution 层级，coarse mesh 求低频，fine mesh 求高频

### 10.3 Gaussian projection 的 $\Sigma$ 怎么选

Paper 没明确说 $\Sigma$ 是 learnable 还是 hyperparameter。从 description 看应该是 fixed hyperparameter。

直觉上 $\Sigma$ 应该跟 LBO 谱间距有关。如果 $\omega_k$ 间隔大约是 $\Delta\omega$，那 $\Sigma \approx \Delta\omega$ 让一个 pole 影响附近 2-3 个 mode，是个合理选择。

如果 $\Sigma$ learnable，可能能自适应不同 manifold 的谱密度。这是个 potential improvement。

### 10.4 与 DiffusionNet 的对比

DiffusionNet (Sharp 2022) 也用 LBO eigenfunction 做几何深度学习，在 SHREC-11 上已经 99.4%。GLNO 是 99.7%，差距很小。

但 DiffusionNet 只做 classification/segmentation，不做 simulation（PDE solving）。GLNO 既能做 simulation（Poisson、Shape-Net Car CFD），又能做 segmentation/classification，更 general purpose。

DiffusionNet 的核心是 diffusion operator 做特征传播，没有 explicit Laplace domain。GLNO 把 LBO eigenfunction 当 oscillatory basis，再叠加 exponential damping，explicitly 引入 transient 表达。

哲学上：DiffusionNet 是"用 LBO 做特征平滑"，GLNO 是"用 LBO 做 spectral transform"。前者偏 spatial，后者偏 spectral。两者其实是互补的。

### 10.5 Spectral vs spatial 的经典 tradeoff

GLNO 是 spectral approach，GNO/graph method 是 spatial approach。经典 tradeoff：

- **Spectral**：global mode，frequency-domain 乘法 cheap，但 basis 受限于 manifold 几何（LBO 高频收敛慢）
- **Spatial**：local message passing，basis 灵活，但 cost $O(n^2)$ for attention 或 $O(n \cdot k)$ for graph conv

GLNO 在 efficiency 上赢了（0.13M params, 2-8s/epoch），表达上受限于 LBO 低频 bias。对低频主导的 physical system（Poisson、扩散、shape classification）特别合适，对高频主导的（turbulence、shock）可能不够。

### 10.6 Pole-residue kernel 的物理意义

Pole-residue form $K_\theta(s) = \sum_n \frac{\beta_n}{s - \mu_n}$ 在控制论里就是 transfer function 的 partial fraction decomposition，对应一阶 LTI 系统的并联。

每个 pole $\mu_n$ 对应一个 time-domain mode $e^{\mu_n t}$：
- $\text{Re}(\mu_n) < 0$：衰减 mode（stable）
- $\text{Re}(\mu_n) > 0$：增长 mode（unstable，物理上少见）
- $\text{Im}(\mu_n) \neq 0$：振荡
- $\text{Im}(\mu_n) = 0$：纯 exponential

学到的 pole 的物理可解释性很强：可以直接读出"系统有哪些固有 mode"。这是 graph/transformer method 没有的好处。

类比：物理学家用 normal mode 分析振动系统，每个 mode 对应一个 eigenfrequency。GLNO 学到的 pole 就是 operator 的 "learned normal mode"。

参考：
- [Transfer function 和 pole-residue (控制论基础)](https://en.wikipedia.org/wiki/Transfer_function)
- [Partial fraction decomposition](https://en.wikipedia.org/wiki/Partial_fraction_decomposition)

### 10.7 与 Transformer operator 的对比

Transolver (Wu 2024)、GNOT (Hao 2023) 用 transformer 做 operator learning。它们的优势是 spatial flexibility + attention 的 universal approximation，劣势是 $O(n^2)$ cost 和差的 interpretability。

Table 3 中：
- SHREC-11 上 GNOT 35.3%（惨），Transolver 43.8%（惨）
- RNA 上 GNOT 84.3%，Transolver 86.0%（还行）
- Human 上 GNOT 66.2%，Transolver 61.9%（惨）

Transformer 在 shape classification 上不行，因为没 intrinsic geometric bias。GLNO 用 LBO 把几何结构编码进 basis，自然得好。

但 transformer 在 PDE 上（Fluid、CFD）通常不错，因为 attention 能学 arbitrary spatial coupling。GLNO 在 Shape-Net Car 上跟 AMG/Transolver 差不多，没明显优势。这印证了"geometry-rich task GLNO 强，geometry-poor task 大家差不多"。

---

## 十一、可能的 future work 和我自己的猜想

1. **Learn $\mathcal{P}$**：现在 $\mathcal{P}$ 是手工选的 curvature/distance。能否用 network learn 一个 optimal $\mathcal{P}$ for given task？比如 $\mathcal{P}_\theta(x) = \text{MLP}(\text{geometric features})$。理论上这是更 general 的形式，但 theorem B.4 的条件需要重新检查。

2. **Learnable Gaussian $\Sigma$**：让 $\Sigma$ 随 training 学，可能让网络自适应不同 manifold 的谱密度。

3. **Multi-scale LBO**：LBO 高频收敛慢是 known issue。能否用多分辨率 LBO（coarse-to-fine）同时捕捉低频 global mode 和高频 local feature？类似 multigrid 方法。

4. **Combination with attention**：GLNO 的 spectral branch + attention 的 spatial branch 做混合，可能兼得效率与表达力。类似 spectral-spatial hybrid。

5. **Dynamic manifold**：现在 manifold 是静态的。如果 manifold 本身随时间演化（比如 fluid-structure interaction 中的变形 boundary），能否扩展 GLNO 到 time-varying manifold？这需要 LBO eigenfunction 也随时间变化，技术上很难。

6. **Approximate LBO eigenpair**：用 Heat Method 或 randomized SVD 求近似 LBO eigenpair，可能大幅加速大 mesh 上的 preprocessing。

7. **Application to weather/climate**：Spherical FNO 已经在 weather prediction 上有应用。GLNO 理论上能处理非球面地形（比如带地形的地球表面），可能是 climate simulation 的新工具。

---

## 十二、最终一句话总结

GLNO 是 Laplace 变换在 manifold 上的几何化扩展。它把"时间轴上的指数衰减 + 振荡"换成"manifold 上沿几何坐标 $\mathcal{P}$ 的衰减 + LBO eigenfunction 振荡"，同时保留 pole-residue calculus 的代数结构。通过 Gaussian projection 软映射连续 learned pole 到离散 LBO 谱，它实现了 grid-invariant + transient-aware 的 operator learning。

**它适合的任务**：定义在任意 mesh 上的 PDE，带 transient/decaying 动力学的系统，shape classification/segmentation。

**它不适合的任务**：高度 localized chaos（Lorenz），高频主导的 turbulence/shock，超大规模 mesh（LBO eigendecomposition cost 高）。

**核心 trick**：$e^{-i\omega t} \to \phi_\omega(x)$（替换 oscillatory basis）+ $e^{-\sigma t} \to e^{-\sigma \mathcal{P}(x)}$（替换 damping axis）+ Gaussian projection（连续到离散的软映射）。

如果只记一句话：**GLNO 是把 Laplace 变换的"时间轴"换成 manifold 上的"几何坐标轴"，其他都顺理成章**。

---

参考链接汇总：
- [GLNO Paper (本文)](https://arxiv.org/) — 假设最终发 arXiv
- [LNO (Cao 2024, Nature MI)](https://www.nature.com/articles/s42256-024-00835-6)
- [FNO (Li 2020)](https://arxiv.org/abs/2010.08895)
- [Geo-FNO (Li 2023)](https://jmlr.org/papers/v24/23-0065.html)
- [Spherical FNO (Bonev 2023)](https://proceedings.mlr.press/v202/bonev23a.html)
- [DiffusionNet (Sharp 2022)](https://dnaralab.github.io/DiffusionNet/)
- [Heat Method (Crane 2013)](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)
- [Laplace-Beltrami operator](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator)
- [Transfer function & pole-residue](https://en.wikipedia.org/wiki/Transfer_function)
- [GNOT (Hao 2023)](https://proceedings.mlr.press/v202/hao23b.html)
- [Transolver (Wu 2024)](https://arxiv.org/abs/2402.02366)
- [Neural Operator review (Kovachki 2023)](https://jmlr.org/papers/v24/21-0806.html)

如果你想 build 更深的 intuition，建议把 LNO 原文先读一遍，再回头看 GLNO 就会觉得"哦，原来就是把 $t$ 换成 $\mathcal{P}$，把 $e^{-i\omega t}$ 换成 $\phi_\omega$"，整个 paper 的骨架就清楚了。剩下的 engineering 细节（Gaussian projection、网络架构）都是围绕这个核心 idea 的实现。

---

# GLNO: Geometric Laplace Neural Operator 深度解析

Andrej, 这篇paper试图解决一个很有意思的问题：如何把 Laplace 变换的 operator learning 框架扩展到 arbitrary Riemannian manifold，同时还能处理 transient/decaying dynamics。下面我尽量把 intuition 和 technical details 都拆开讲。

---

## 1. Motivation: 现有 Neural Operator 的两个痛点

**痛点一：Periodicity assumption**
FNO 用 Fourier basis $\{e^{-i\omega t}\}$，这天然假设信号是周期的。但真实物理系统里大量是 aperiodic + decaying 的 transient response，比如阻尼振荡 $\sin(\omega t) e^{-\sigma t}$、heat diffusion 的指数衰减、PDE 中的 shock wave。Fourier basis 在这种情况下需要非常多高频项去近似不连续/衰减，效率极低。

**痛点二：Euclidean grid dependence**
FNO 依赖 FFT，需要 uniform grid。即使后续的 Geo-FNO、Spherical FNO 扩展到了 sphere、torus，本质上还是要求 manifold 上有 well-defined 的 spectral transform 和 symmetry。对于 anatomical surface、RNA 分子表面、汽车几何这种 arbitrary topology + irregular mesh，FFT 完全失效。

LNO (Cao et al. 2024) 部分解决了痛点一，用 pole-residue formulation 引入 transient component，但它 still 用 FFT 来算 Laplace transform 的系数，所以 still 受限于 grid。

GLNO 的核心 insight：**把 Laplace 变换从 Euclidean 时间轴搬到 manifold 上，关键在于构造一个 "time-like" 的 scalar function $\mathcal{P}(x)$，同时用 Laplace-Beltrami eigenfunction $\phi_\omega$ 替代 $e^{-i\omega t}$ 作为 oscillatory basis**。这样就同时摆脱了周期性假设和 grid 依赖。

---

## 2. Preliminary 1: LNO 的 pole-residue 架构

先理解 LNO 的 Euclidean 版本，因为 GLNO 是在它基础上做的几何扩展。

**输入分解：** 输入函数 $f(t)$ 先做 Fourier 分解 $f(t) = \sum_\omega \alpha_\omega e^{i\omega t}$，然后对每个 Fourier mode 做 Laplace transform：

$$F(s) = \sum_\omega \frac{\alpha_\omega}{s - i\omega}$$

这里 $s \in \mathbb{C}$ 是 Laplace domain 变量，$i\omega$ 是 input pole（来自 Fourier basis 的 Laplace 变换 $\mathcal{L}\{e^{i\omega t}\} = \frac{1}{s-i\omega}$）。

**Kernel 参数化：** LNO 把 operator kernel 直接在 Laplace domain 参数化为 pole-residue 形式：

$$K_\theta(s) = \sum_{n=1}^N \frac{\beta_n}{s - \mu_n}, \quad \mu_n \in \mathbb{C}, \beta_n \in \mathbb{C}$$

其中 $\{\mu_n\}_{n=1}^N$ 是 learnable pole（决定 transient mode 的复频率，包含 decay rate $\text{Re}(\mu_n)$ 和振荡频率 $\text{Im}(\mu_n)$），$\{\beta_n\}_{n=1}^N$ 是 learnable residue（决定每个 transient mode 的振幅）。

**Spectral multiplication + residue calculus：** 卷积在 Laplace domain 变成乘法：

$$G(s) = F(s) \cdot K_\theta(s) = \left(\sum_\omega \frac{\alpha_\omega}{s-i\omega}\right)\left(\sum_{n=1}^N \frac{\beta_n}{s-\mu_n}\right)$$

通过 partial fraction decomposition 拆成两组 pole：

$$G(s) = \underbrace{\sum_\omega \frac{\hat{a}_\omega^{\text{steady}}}{s-i\omega}}_{\text{稳态：input pole 继承}} + \underbrace{\sum_{n=1}^N \frac{\hat{a}_n^{\text{transient}}}{s-\mu_n}}_{\text{瞬态：kernel pole 注入}}$$

residue 通过留数定理算：

$$\hat{a}_n^{\text{transient}} = \lim_{s\to\mu_n}(s-\mu_n)G(s) = \beta_n F(\mu_n) = \beta_n \sum_\omega \frac{\alpha_\omega}{\mu_n - i\omega}$$

$$\hat{a}_\omega^{\text{steady}} = \lim_{s\to i\omega}(s-i\omega)G(s) = \alpha_\omega K_\theta(i\omega) = \alpha_\omega \sum_{n=1}^N \frac{\beta_n}{i\omega - \mu_n}$$

**逆变换重建：** 逆 Laplace 变换有解析形式：

$$g(t) = \mathcal{L}^{-1}\{G(s)\} = \sum_{n=1}^N \hat{a}_n^{\text{transient}} e^{\mu_n t} + \sum_\omega \hat{a}_\omega^{\text{steady}} e^{i\omega t}$$

第一项是 transient response（含 $e^{\text{Re}(\mu_n)t}$ 衰减/增长因子），第二项是 steady-state oscillation。这正是 LNO 比 FNO 优越之处：它能自然表达衰减动力学。

**LNO 的局限：** input decomposition 仍用 FFT，所以仍需 uniform grid，仍假设 input 是周期的。

---

## 3. Preliminary 2: Laplace-Beltrami Operator

在 Riemannian manifold $(\mathcal{M}, \mu)$ 上，LBO $\Delta_\mu$ 是 Euclidean Laplacian 的几何推广。它的特征值问题：

$$-\Delta_\mu \phi_{\omega_k} = \lambda_k \phi_{\omega_k} = \omega_k^2 \phi_{\omega_k}$$

- $\lambda_k \geq 0$：第 $k$ 个特征值，物理上是"低频振动模态的能量"
- $\omega_k = \sqrt{\lambda_k}$：eigenfrequency，对应 oscillatory 频率
- $\phi_{\omega_k}$：第 $k$ 个特征函数，manifold 上的 "Fourier mode"
- 排序 $0 = \omega_1 < \omega_2 \leq \omega_3 \leq \cdots$，$\omega_1=0$ 对应常数特征函数
- 正交归一 $\langle \phi_\omega, \phi_{\omega'} \rangle_\mu = \delta_{\omega\omega'}$

直觉：在 sphere 上 LBO eigenfunction 就是 spherical harmonics $Y_l^m$；在 flat torus 上就是平面波 $e^{i\mathbf{k}\cdot\mathbf{x}}$；在 arbitrary mesh 上 LBO eigenfunction 提供 coordinate-free 的 spectral basis。LBO 完全由 manifold 内蕴几何决定，与具体 discretization 无关——这就是 grid invariance 的来源。

参考文献：
- [Laplace-Beltrami operator in geometry processing (Crane et al., 2013)](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)
- [DiffusionNet (Sharp et al., 2022)](https://dnaralab.github.io/DiffusionNet/)

---

## 4. 核心方法：Generalized Laplace Basis

GLNO 的第一步是推广 Laplace basis，去掉周期性假设。定义：

$$\varepsilon_z(x) = e^{-zx} = \underbrace{e^{-\sigma x}}_{\text{non-periodic}} \cdot \underbrace{e^{-i\omega x}}_{\text{periodic}}, \quad z = \sigma + i\omega \in \mathbb{C}$$

- $\sigma = \text{Re}(z)$：控制 exponential decay（$\sigma > 0$）/ growth（$\sigma < 0$）
- $\omega = \text{Im}(z)$：控制 oscillation 频率
- 当 $\sigma = 0$ 时退化回 Fourier basis $\{e^{-i\omega x}\}$，所以 generalized Laplace basis 是 Fourier basis 的严格超集

**Spectral decomposition：** 对任意 $f(x)$，定义分解

$$\mathcal{D}\{f(x)\}(z) = \left(\int_0^\infty f(x) e^{-\sigma x} e^{i\omega x} dx\right) \varepsilon_z(x) = \langle e^{-\sigma x} f, e^{-i\omega x}\rangle \cdot \varepsilon_z(x)$$

关键 trick：把 $f(x)$ 先乘上 $e^{-\sigma x}$（这是 learnable 的 damping），再做 FFT 算系数 $\langle e^{-\sigma x} f, e^{-i\omega x}\rangle$。这样既保留了 FFT 的计算效率，又获得了 exponential basis 的表达能力。

**Laplace transform：** 

$$\mathcal{L}\{\varepsilon_z(t)\} = \int_0^\infty e^{-zt} e^{-st} dt = \frac{1}{s+z}$$

注意符号：相比 LNO 的 $\frac{1}{s-i\omega}$，这里是 $\frac{1}{s+z}$，因为 pole 从 $i\omega$ 推广到了 $-z = -\sigma - i\omega$。

**Learnable approximation：** 实际实现时只保留 $M$ 个 learnable basis function：

$$\mathcal{D}\{f(t)\} = \sum_{i=1}^M \alpha_i \varepsilon_{z_i}(t), \quad \alpha_i = \langle f, \varepsilon_{z_i}\rangle, \quad z_i = \sigma_i + i\omega_i$$

**输出重建：** 走完 pole-residue 计算，输出为

$$g(t) = \sum_{n=1}^N \hat{a}_n^{\text{transient}} e^{\mu_n t} + \sum_{i=1}^M \hat{a}_i^{\text{steady}} e^{-z_i t}$$

第一项是 kernel 引入的 transient mode，第二项是 input basis 继承的 steady mode，但现在是 exponential 而非纯振荡——**input decomposition 也获得了 transient 表达能力**，这是 LNO 没有的。

---

## 5. 核心方法：Geometric Laplace Neural Operator

把 generalized Laplace basis 搬到 manifold 上，关键替换：

$$\varepsilon_z(x) = \underbrace{\exp(-\sigma \mathcal{P}(x))}_{\text{non-periodic}} \cdot \underbrace{\phi_\omega(x)}_{\text{periodic}}, \quad z = \sigma + i\omega_k \in D \subset \mathbb{C}$$

这里做了两个替换：

1. $e^{-i\omega x}$（Euclidean plane wave）$\to \phi_\omega(x)$（LBO eigenfunction）。这是 manifold 上的"Fourier mode"，coordinate-free。

2. $e^{-\sigma x}$（时间轴上的 exponential decay）$\to \exp(-\sigma \mathcal{P}(x))$，其中 $\mathcal{P}: \mathcal{M} \to \mathbb{R}$ 是 intrinsic geometric scalar function。Paper 里 $\mathcal{P}$ 可以取 curvature、boundary distance 等。这是把 "time axis" 替换成 "geometric coordinate axis"——$\mathcal{P}$ 扮演 "time-like coordinate" 的角色。

$D \subset \mathbb{C}$ 是离散 spectral domain，由 manifold 几何决定（因为 $\omega_k$ 来自 LBO 离散谱，是离散的；而 $\sigma$ 是 learnable 的连续实数）。

**Manifold 上的分解：**

$$f(x) = \sum_{i=1}^M \alpha_i \varepsilon_{z_i}(x) = \sum_{i=1}^M \langle e^{-\sigma_i \mathcal{P}} f, \phi_{\omega_i}\rangle_\mu \cdot \varepsilon_{z_i}(x)$$

系数通过 manifold 上的 inner product 计算：先乘 $e^{-\sigma_i \mathcal{P}(x)}$ 再投影到 LBO eigenfunction。

**Geometric Laplace transform (核心定理 B.4)：** 

$$\mathcal{L}\{\varepsilon_z(x)\} \mapsto \frac{1}{s+z}, \quad \forall z \in D$$

证明 sketch 比较微妙：用 coordinate chart 把 manifold 拆成 $\mathcal{P}$-radial 坐标 + angular 坐标。volume element $d\mu_g = J(\mathcal{P},\theta) d\mathcal{P} d\theta$。当 $A(\mathcal{P}) = \int_{\mathcal{P}^{-1}(\mathcal{P})} J d\theta$ 接近常数（即 level set 面积不随 $\mathcal{P}$ 指数增长）时，几何 Laplace transform 近似退化为经典 Laplace transform，得到 $\frac{1}{s+z}$。这是一个 perturbation argument，本质上要求 $\mathcal{P}$ 是 "radially well-behaved" 的。

**Operator action：** Kernel 仍用 pole-residue $K_\theta(s) = \sum_{n=1}^N \frac{\beta_n}{s-\mu_n}$，在 spectral domain 做乘法：

$$G(s) = F(s) K_\theta(s) = \left(\sum_{i=1}^M \frac{\alpha_i}{s+z_i}\right)\left(\sum_{n=1}^N \frac{\beta_n}{s-\mu_n}\right)$$

partial fraction 拆成 steady + transient：

$$G(s) = \sum_{i=1}^M \frac{\hat{a}_i^{\text{steady}}}{s+z_i} + \sum_{n=1}^N \frac{\hat{a}_n^{\text{transient}}}{s-\mu_n}$$

residue 同样用留数定理：

$$\hat{a}_n^{\text{transient}} = \beta_n F(\mu_n) = \beta_n \sum_{i=1}^M \frac{\alpha_i}{\mu_n + z_i}$$

$$\hat{a}_i^{\text{steady}} = \alpha_i K_\theta(-z_i) = \alpha_i \sum_{n=1}^N \frac{\beta_n}{-z_i - \mu_n}$$

---

## 6. 关键 trick: Inverse Geometric Laplace Transform

最微妙的地方在于逆变换。Steady term 容易：$\mathcal{L}^{-1}\{\frac{1}{s+z_i}\} = e^{-z_i t} \to \varepsilon_{z_i}(x)$，直接用 geometric basis。

Transient term 难：pole $\mu_n \in \mathbb{C}$ 是 learnable 的连续值，但 manifold 上 LBO 谱是离散的 $\{\omega_k\}$。learned pole $\mu_n$ 不会恰好落在离散 grid 上。

GLNO 的解法是 **Gaussian filter smoothing**：

$$\mathcal{L}^{-1}\left\{\frac{1}{s-\mu_n}\right\} = \exp(\text{Re}(\mu_n) \mathcal{P}) \cdot \sum_\omega \text{Gauss}(\text{Im}(\mu_n) - \omega) \phi_\omega(x)$$

其中

$$\text{Gauss}(x) = \frac{1}{\sqrt{2\pi}\Sigma} \exp\left(-\frac{x^2}{2\Sigma^2}\right)$$

这里 $\Sigma$ 是 Gaussian bandwidth。直觉：
- $\text{Re}(\mu_n)$ 控制整体 decay/growth，直接作用在 $\mathcal{P}$（geometric coordinate）上
- $\text{Im}(\mu_n)$ 控制振荡频率，但因为是连续值，需要"分配"到离散 LBO eigenfunction 上
- Gaussian kernel 把连续频率 soft-assign 到附近的离散 eigenfrequency $\omega$，类似 RBF interpolation
- 当 $\text{Im}(\mu_n)$ 恰好等于某个 $\omega_k$ 时，Gaussian 集中在该 mode；介于两个 $\omega$ 之间时，两个 mode 都被激活

这是整个架构的"软"技巧——既保留了 continuous spectral learning 的表达力，又遵守了 manifold 离散谱的物理约束。

最终重建：

$$g(x) = \sum_{i=1}^M \hat{a}_i^{\text{steady}} \varepsilon_{z_i}(x) + \sum_{n=1}^N \hat{a}_n^{\text{transient}} \cdot \mathcal{L}^{-1}\left\{\frac{1}{s-\mu_n}\right\}$$

---

## 7. 网络架构 GLNONet

整体 pipeline：

```
Input f: M → R^{d_in}
   ↓
[Geometric features: curvature, boundary distance, LBO eigenpairs] (SciPy 计算)
   ↓
Encoder MLP P: R^{d_in} → R^{d_latent}
   ↓
[GLNO Block 1] → [GLNO Block 2] → ... → [GLNO Block L]
   ↓
Decoder MLP Q: R^{d_latent} → R^{d_out}
   ↓
Output u: M → R^{d_out}
```

每个 GLNO Block 内部：
1. 计算 learnable decomposition（用 $\sigma_i$ 和 LBO eigenfunction）
2. Laplace transform 到 spectral domain
3. Pole-residue multiplication with kernel $K_\theta$
4. Inverse transform 用 Gaussian projection
5. 与原始 geometric feature 通过 lightweight MLP 融合 + skip connection

**Grid invariance 的来源：** LBO eigenfunction 是 coordinate-free 的，同一个 manifold 不管用什么 mesh 离散，eigenfunction（作为 manifold 上的函数）是同一个。换 mesh 只改变离散表示，不改变 intrinsic spectral structure。所以 train on one mesh, test on another mesh 直接可行，不需要 resampling 或插值。

对比 Geo-FNO 需要把不规则 mesh deform 到 regular grid 再做 FFT，本质上还是依赖 grid；GLNO 完全跳出 grid 框架。

参考：
- [FNO 原论文](https://arxiv.org/abs/2010.08895)
- [LNO 原论文 (Cao et al., 2024)](https://www.nature.com/articles/s42256-024-00835-6)
- [Geo-FNO](https://jmlr.org/papers/v24/23-0065.html)

---

## 8. 实验结果分析

### 8.1 ODEs/PDEs (固定 grid)

Table 2 上的关键观察：

| Task | FNO | LNO | GLNO |
|---|---|---|---|
| Pendulum c=0 | 0.3668 | 0.1540 | **0.0875** |
| Pendulum c=0.5 | 0.6090 | 0.1718 | **0.0965** |
| Duffing c=0 | 0.4681 | 0.1362 | **0.0725** |
| Beam | 0.0034 | 0.0083 | **0.0026** |
| Diffusion | 0.0064 | 0.0011 | **0.0006** |

GLNO 几乎在所有 task 上都是最好或并列最好。值得注意的是：
- Pendulum/Duffing 带阻尼时（$c > 0$），FNO 退化严重（c=0.5 时 error 0.6+），因为它强行用周期基去拟合衰减信号
- LNO 已经比 FNO 好很多，但 GLNO 再进一步——这说明 geometric basis 引入的 $\mathcal{P}$ 信息有附加价值
- Lorenz system 是个反例：CNO（local 方法）最好，因为 Lorenz 是高度 chaotic、localized dynamics，spectral 方法的 global bias 反而吃亏

### 8.2 Unstructured mesh: Poisson 方程

$$-\Delta u = f, \quad u|_{\partial\Omega} = 0$$

GLNO 达到 0.0044 relative error，Geo-FNO 是 0.0049，验证了 LBO-based 架构在 unstructured grid 上的优越性。重要的是 GLNO 在 transient region 表现尤其好。

### 8.3 Real-world 任务

| Task | GLNO | 次优 | DiffusionNet |
|---|---|---|---|
| SHREC-11 (30-way classification) | **99.7%** | GNOT 35.3% | 99.4% |
| RNA segmentation | **90.1%** | AMG 88.5% | 85.6% |
| Human segmentation | **91.0%** | AMG 62.5% | 90.3% |
| Shape-Net Car (pressure) | **0.0960** | AMG 0.0978 | - |
| Shape-Net Car (velocity) | 0.1037 | AMG 0.0919 | - |

最 striking 的结果是 SHREC-11：其他 neural operator 都很差（30-60%），因为它们没有真正用 intrinsic geometry；DiffusionNet（专门的几何深度学习方法）能达到 99.4%；GLNO 99.7%，说明 GLNO 的 geometric spectral basis 在 shape analysis 上有 intrinsic advantage。

参数效率极好：GLNO 0.13M-0.21M params，远小于 LSM 的 21.68M、GINO 的 4.74M。原因是 spectral approach 避免了 graph convolution 的 message passing 复杂度或 attention 的 $O(n^2)$ cost。

### 8.4 Ablation: 去掉 $\sigma$

| Variant | Poisson | SHREC-11 | RNA | Human |
|---|---|---|---|---|
| GLNO | 0.0044 | 99.7% | 90.1% | 91.0% |
| GLNO w/o σ | 0.0087 | 95.5% | 82.2% | 88.5% |

去掉 $\sigma$（non-periodic basis）后所有 task 都退化，证实 generalized Laplace basis 中的 exponential 部分确实是关键。退化最严重的是 RNA（-7.9%），因为 RNA 分子表面高曲率区域的 transient 特征需要 non-periodic basis 捕捉。

---

## 9. 理论分析

### 9.1 Computational Complexity (Appendix B.5)

记 $K$ = 离散点数，$M$ = generalized basis 数，$N$ = kernel pole 数，$D$ = channel dimension。

- **Decomposition：** 每个 $\sigma_i$ 一次 FFT，cost $O(MK \log K \cdot D)$
- **Kernel multiplication：** $M \times N$ 交互，cost $O(MN \cdot D)$
- **Reconstruction：** 在 $K$ 个点上 evaluate $(M+N)$ 个 exponential，cost $O((M+N)K \cdot D)$

总复杂度：

$$C_{\text{forward}} = O\Big(D(MK\log K + MN + (M+N)K)\Big) \approx O(DMK\log K)$$

当 $K$ 大、$M, N$ 中等时，decomposition 的 FFT 是主导项。这和 FNO 的 $O(K\log K \cdot D)$ 同阶，但 $M$ 因子（basis 数）是额外开销——这是为表达能力付出的代价。

### 9.2 Approximation Bound (Appendix B.6)

把总误差拆两部分：

$$\|\mathcal{T}_* - \mathcal{T}_\theta\|_{L^2\to L^2} \leq \varepsilon_{\text{basis}}(M,N) + \varepsilon_{\text{par}}(d_\theta)$$

- $\varepsilon_{\text{basis}}(M,N)$：用 $M+N$ 个 exponential 逼近真实 kernel $k_*$ 的 best approximation error
- $\varepsilon_{\text{par}}(d_\theta)$：有限参数化误差

如果 $k_* \in H^\alpha(0,T)$（Sobolev regularity $\alpha$），经典 exponential approximation 给出：

$$\varepsilon_{\text{basis}}(M,N) \leq C_1 (M+N)^{-\alpha}$$

注意这是 Sobolev-type rate，即使 $H_*(s)$ 不是有理函数（即 kernel 不是有限阶 LTI 系统）也成立。这比 FNO 的 polynomial decay in mode number 更适合 transient system。

有限参数化部分，假设 $\theta \mapsto k_\theta$ 是 $L$-Lipschitz：

$$\varepsilon_{\text{par}}(d_\theta) \leq C_2 d_\theta^{-p}$$

其中 $p$ 依赖网络架构。最终：

$$\|\mathcal{T}_* - \mathcal{T}_\theta\| \leq C\Big((M+N)^{-\alpha} + d_\theta^{-p}\Big)$$

### 9.3 Sample Complexity (Appendix B.7)

Hypothesis class $\mathcal{H}_{\text{GLNO}}$ 的 Rademacher complexity：

$$\Re_N(\mathcal{H}_{\text{GLNO}}) = O\left(\sqrt{\frac{M + d_\theta}{N}}\right)$$

其中 $N$ 是 sample 数，$M$ 是 basis 数，$d_\theta$ 是 parameter 数。Generalization bound：

$$\mathcal{R}(\theta) \leq \hat{\mathcal{R}}(\theta) + C\sqrt{\frac{M + d_\theta + \log(1/\delta)}{N}}$$

达到精度 $\epsilon$ 所需样本：

$$N_{\min}^{\text{GLNO}} = O\left(\frac{M + d_\theta}{\epsilon^2}\right)$$

线性 scaling，与经典 operator learning 一致。

---

## 10. Intuition 总结与批判性思考

**Intuition 1：为什么这个框架 work？**
关键是把 Laplace domain 的 pole-residue calculus 这个分析工具完整搬到了 manifold 上。Laplace 变换的代数结构（卷积→乘法、pole 决定 mode、residue 决定振幅）是纯 algebraic 的，不依赖 domain 的具体几何——只要你能定义一个"basis function $\to$ $\frac{1}{s+z}$" 的映射。GLNO 通过巧妙选择 $\varepsilon_z(x) = e^{-\sigma\mathcal{P}} \phi_\omega$ 实现了这个映射。

**Intuition 2：$\mathcal{P}$ 的选择**
Paper 里 $\mathcal{P}$ 可以是 curvature、boundary distance 等。但严格来说，定理 B.4 要求 $A(\mathcal{P})$（$\mathcal{P}$ 的 level set 面积）变化 sub-exponential。这是个隐含约束——如果 manifold 是狭长的、$\mathcal{P}$ 是沿长轴的坐标，level set 面积可能急剧变化，定理就近似失效。Paper 没有系统研究 $\mathcal{P}$ 选择对精度的影响，这是个 limitation。

**Intuition 3：Gaussian projection 的"软"性质**
Gaussian projection 实际上是一个 bandwidth-limited 的 spectral filter。当 $\Sigma$ 大时，连续 pole 被平滑到多个 mode，类似 low-pass；$\Sigma$ 小时集中到单个 mode，类似 hard assignment。Paper 没明确说 $\Sigma$ 是 learnable 还是 fixed，但暗示是 hyperparameter。这个选择可能影响 manifold 谱的"分辨率"——小 $\Sigma$ 容易过拟合，大 $\Sigma$ 容易欠拟合。

**Intuition 4：与 spectral geometry 方法的联系**
DiffusionNet 已经用 LBO eigenfunction 做几何深度学习，效果很好。GLNO 与之的区别：
- DiffusionNet 用 diffusion operator 做特征传播，没有 explicit Laplace domain
- GLNO 把 LBO eigenfunction 当 oscillatory basis，再叠加 exponential damping $\mathcal{P}$，explicitly 引入 transient 表达

Table 3 中 DiffusionNet 在 SHREC-11 已经 99.4%，GLNO 99.7%，差距不大。但 DiffusionNet 只做 classification，GLNO 还能做 simulation（Poisson、Shape-Net Car）和 segmentation（Human、RNA），更 general purpose。

**Intuition 5：未解决的问题**
- LBO eigen-decomposition 本身在大 mesh 上 expensive（$O(n^3)$ 直接求，$O(n^2)$ iterative）。Paper 用 SciPy，没具体说 mesh 大到多少就不可行。
- $\mathcal{P}$ 必须预先定义或从 geometric feature 里选——这是个手工 design choice，理论上能否 learn $\mathcal{P}$ 本身？
- Pole-residue kernel $K_\theta$ 的参数 $\{\mu_n, \beta_n\}$ 是 layer-wise 共享还是每个 spatial location 不同？Paper 没完全说清楚。
- 与 GNO (graph neural operator) 对比：GLNO 用 spectral basis，GNO 用 spatial message passing，本质上 spectral vs spatial 的经典 tradeoff。GLNO 在 efficiency 上赢了，但表达可能受限于 LBO 的低频 bias（LBO eigenfunction 高频收敛慢）。

---

## 11. 相关工作与延伸阅读

如果你想 build 更深的 intuition，推荐：

1. **FNO family：**
   - [FNO (Li et al., 2020)](https://arxiv.org/abs/2010.08895)
   - [Geo-FNO (Li et al., 2023)](https://jmlr.org/papers/v24/23-0065.html)
   - [Spherical FNO (Bonev et al., 2023)](https://proceedings.mlr.press/v202/bonev23a.html)
   - [FourCastNet (Pathak et al., 2022)](https://arxiv.org/abs/2202.11214)

2. **Laplace 变换方法：**
   - [LNO (Cao et al., 2024, Nature Machine Intelligence)](https://www.nature.com/articles/s42256-024-00835-6)
   - [Wavelet Neural Operator (Navaneeth et al., 2024)](https://www.sciencedirect.com/science/article/pii/S0045782523007522)

3. **LBO spectral geometry：**
   - [DiffusionNet (Sharp et al., 2022)](https://dnaralab.github.io/DiffusionNet/)
   - [The Heat Method (Crane et al., 2013)](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)
   - [HodgeNet (Smirnov & Solomon, 2021)](https://ranlvl.github.io/hodgenet/)
   - [DeltaConv (Wiersma et al., 2022)](https://rudinelp.github.io/deltaconv/)

4. **Transformer/graph neural operators：**
   - [GNOT (Hao et al., 2023)](https://proceedings.mlr.press/v202/hao23b.html)
   - [Transolver (Wu et al., 2024)](https://arxiv.org/abs/2402.02366)
   - [GKNO / AMG (Li et al., 2025)](https://dl.acm.org/doi/10.1145/3691334)

5. **Operator learning 综述：**
   - [Kovachki et al., 2023, Neural Operator review](https://jmlr.org/papers/v24/21-0806.html)
   - [Operator Learning: Algorithms and Analysis (Kovachki et al., 2024)](https://arxiv.org/abs/2402.15715)

6. **Laplace 变换与 PDE：**
   - [Classical Laplace transform for PDEs (Wikipedia)](https://en.wikipedia.org/wiki/Laplace_transform)
   - [Pole-residue decomposition (Signal processing)](https://en.wikipedia.org/wiki/Partial_fraction_decomposition)

---

## 12. 一句话总结

GLNO = (generalized Laplace basis $e^{-\sigma x} e^{-i\omega x}$) 替换为 (manifold 版本 $e^{-\sigma\mathcal{P}(x)} \phi_\omega(x)$) + (Laplace domain pole-residue kernel) + (Gaussian projection 把 continuous learned pole 软映射到 discrete LBO spectrum)。它继承了 LNO 处理 transient 的能力，同时通过 LBO 获得了 grid invariance，在 irregular mesh PDE 和 geometric shape analysis 上都取得 SOTA，且参数效率高。

如果你想真正 build intuition，建议从两个角度切入：一是把 LBO eigenfunction 想成"manifold 上的 Fourier mode"，二是把 $\mathcal{P}$ 想成"manifold 上的时间轴"——这样从 FNO → LNO → GLNO 的演化路径就很自然了。数学上最微妙的依然是定理 B.4 的 perturbation argument 和 Gaussian projection 的连续-离散转换，这两处是 paper 的核心贡献，也是 future work 最容易改进的地方。
