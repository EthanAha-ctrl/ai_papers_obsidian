---
source_pdf: Hessian Geometry of Latent Space in Generative Models.pdf
paper_sha256: 5b2d4769070ecdd47326aa9341e790c187bee2ac008da1295b4ed5f7f06c543b
processed_at: '2026-08-19T11:01:26-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们抛开那些复杂的 math derivation，用最直白的人话来过一遍这篇 paper 到底在说什么。核心就是：**Diffusion model 的 latent space 是一块充满“断层线”的破碎大地，作者找到了一种方法把这些断层线精确地画了出来。**

### 1. 核心问题：为什么 Diffusion 走着走着就突然“变异”了？

你玩 Stable Diffusion 的时候肯定遇到过：在 latent space 里做线性插值，从一张图变到另一张图，大部分时候很平滑，但偶尔会突然“闪现”——上一秒还是一座山，下一秒突然变成了一头狮子。

这说明了什么？说明 latent space 绝对不是一个平坦、均匀的几何空间。空间里存在某种“不可逾越的鸿沟”或者“断层”。

### 2. 核心类比：把 Diffusion 当成统计物理系统

作者的 insight 非常漂亮：**把 Diffusion model 完全当成一个统计物理系统来看待。**

在统计物理（比如 Ising model）里：
*   参数 $t$ （比如温度 $T$、磁场 $H$）控制系统的宏观状态。
*   样本 $x$ （比如自旋网格的排列）是微观状态。
*   存在一个神秘的函数 $\log Z(t)$（log-partition function，物理上叫 Free energy 自由能）。
*   **关键性质**：这个 Free energy 的曲率（二阶导数 $\nabla^2 \log Z$）正好等于 **Fisher metric**。Fisher metric 衡量的是“参数稍微动一下，样本分布会变化多剧烈”。

作者说：好，那我们在 Diffusion 里对号入座：
*   Latent code $z$ 就是参数 $t$。
*   生成的 image 就是 microstate $x$。
*   只要想办法把 Diffusion 的 $\log Z(z)$ 算出来，我们不就能知道 latent space 每一点的“崎岖程度”了吗？那些极度崎岖、梯度爆炸的地方，就是“相变”发生的地方！

### 3. 怎么算出 $\log Z(z)$？——盲人摸象的反向工程

难点在于：Diffusion model 的公式是个黑盒，我们根本不知道 $\log Z(z)$ 的解析式长什么样。

作者证明了两个 Theorem，给出了一个极其聪明的“反向工程”方法（Theorem 3.1 & 3.2）：

**直觉解释**：假设真实参数是 $t'$，我从这里抽了一批样本。现在我在整个空间里盲猜参数 $t$，然后问：这批样本在 $t$ 下的似然度是多少？
猜得越准（$t$ 越靠近 $t'$），似然度越高。当你取极限（样本数 $N \to \infty$），似然度的形状会坍缩成一个极端锐利的尖峰，这个尖峰的轮廓形状，**完全由 $\log Z(t)$ 决定**。

所以，方法就出来了：
1.  找一堆生成的 image，用 CLIP 提取特征，算出它们之间的距离，近似估计出那个“似然度形状”（也就是 posterior distribution）。
2.  训练一个小小的 MLP（5层，512维），让它输出一个曲面 $\log Z_\theta(t)$，使得这个曲面算出来的形状，和刚才用 CLIP 估出来的形状匹配。
3.  匹配上了？恭喜你，这个 MLP 输出的就是 latent space 的 Free energy！它的 Hessian 矩阵就是 Fisher metric。

### 4. 实验发现 1：物理模型上完美复原

先拿有标准答案的物理模型试手。在 Ising model 和 TASEP（一维交通模型）上，他们用这个方法恢复出的 Free energy，和物理学家用严谨公式算出来的 ground truth 几乎完美重合。

| Metric | 他们的方法 (RMSE) | 传统 VAE 方法 (RMSE) |
| :--- | :--- | :--- |
| TASEP Free Energy | **0.0112** | 0.0524 |
| TASEP 导数 | **0.1165** | 0.3837 |

*变量解释*：RMSE 是均方根误差，越低越好。导数代表 susceptibility（敏感度），这是物理学家用来精确定位相变的关键指标。你看，他们在关键的相变边界上，误差比传统方法低了整整一个数量级。

### 5. 实验发现 2：Diffusion 的“分形相界”

把方法用到 Stable Diffusion 上，炸出了这篇 paper 最 striking 的发现。

在 2D 的 latent slice 上，他们画出 $\log Z$ 的曲面，发现这个曲面不是平滑的，上面布满了陡峭的“悬崖”。这些悬崖就是 image 内容突变的地方——phase boundary。

但最可怕的是它的**分形结构**。
通常物理里的相变边界是一条清晰的线。但在 Diffusion 里，这个边界是**分形**的！你把 latent code 改变 $10^{-5}$（十万分之一），图片从山变成了狮子；你继续在这个边界上放大，改变 $10^{-6}$，它又变了；再放大，一直到 float16 精度的极限 $10^{-8}$，这种“在山和狮子之间反复横跳”的现象依然存在。

**Intuition**: 为什么会这样？因为 diffusion 把一个连续的高斯分布，硬生生映射到了 image space 里几个完全不连通的低维 manifold 上（参考 Brown et al. Union of manifolds 假设）。这种“撕裂和重组”的操作，必然在边界处产生数学上的奇点，导致无穷无尽的细节，就像 Mandelbrot set 一样。

### 6. 实验发现 3：Geodesic 是如何绕开陷阱的

既然有了 metric（地图），我们就能走最短路径了。

在单一的一个 phase 内部，Fisher metric 很平坦，所以最短路径几乎就是直线。这就解释了为什么有些前人论文说“latent interpolation 接近线性”。

但是！一旦插值路径跨越了 phase boundary，线性插值就会直接撞上“悬崖”，导致画面突变。作者用他们的 metric 计算了真正的 Riemannian geodesic，发现 geodesic 会**自动弯曲绕开**那些 metric 巨大的禁区，从而实现真正平滑的过渡。

### 7. 为什么会爆炸？一个极简的数学解释

作者给了一个 Proposition 4.1 来解释 Lipschitz constant 为什么会发散。

**思想实验**：假设目标数据只有两堆，一堆在 $x=-1$，一堆在 $x=+1$，中间什么都没有。Latent space 是一个标准正态分布，集中在 $x=0$。
Diffusion 的 reverse process 要把 $x=0$ 的点，硬推向两边。在 $x=0$ 这个分水岭上，任何一点点微小的扰动，都会决定样本被推向左边还是右边。

公式推导出的 Lyapunov exponent $\lambda$（衡量轨迹发散速度的指标）：
$$ \lambda = \frac{\beta}{2} \left( 1 + \frac{1-\sigma^2}{\sigma^4} \right) $$
*   $\beta$：noise schedule 的扩散率。
*   $\sigma$：目标数据那两堆 Gaussians 的宽度。
*   当 $\sigma \to 0$（数据模式非常尖锐、完全不连通时），$\lambda \to \infty$。

**Intuition 总结**：**数据的多模态性导致了 Diffusion 在 latent space 上的 Lipschitz singularity。** 只要你要把一个连着的 latent 空间映射到断开的 image 空间，中间就一定会产生撕裂，这个撕裂处就是 phase boundary。

### 8. 总结

如果用一句话向 Andrej 总结这篇 paper：

**“我们通过反推 Fisher information metric 发现，Diffusion model 的 latent space 并非一个光滑的球面，它像是一个被地下断层切割过的星球。断层线（phase boundary）呈现分形结构，在断层线上模型对输入极度敏感（Lipschitz 发散）。利用我们重建出的这个几何地图，可以计算出绕开断层的平滑路径，实现无突变的 image interpolation。”**

参考链接：
*   Paper 代码: https://github.com/alobashev/hessian-geometry-of-diffusion-models
*   Brown et al. 2022 Union of manifolds 假设: https://arxiv.org/abs/2207.02862
*   Yang et al. 2023 扩散模型时间维度的 Lipschitz 奇点: https://arxiv.org/abs/2306.11838

---

# Hessian Geometry of Latent Space in Generative Models - 深入讲解

Andrej, 这篇 paper 把 statistical physics 的工具搬到了 diffusion model 的 latent space 上，做了一件相当 elegant 的事：从生成的样本反推出 latent space 上的 Riemannian metric，并发现这个 metric 上面有 fractal 结构的 phase transition boundaries。下面我尽量把 intuition 和技术细节都讲清楚。

## 1. 核心问题与 motivation

Diffusion model 在 latent interpolation 时经常出现 abrupt transition —— 比如从一只狮子突然变成一座山，而不是平滑渐变。这意味着 latent space 不是均匀光滑的几何体。先前工作有两类思路：

- **Riemannian geometry 路线** (Shao et al. 2018, Park et al. 2023, Arvanitidis et al. 2017)：通过 generator Jacobian pullback Euclidean metric 来构造 latent space 上的 metric。但这些都是 **deterministic** generators。
- **Phase transition learning 路线** (Carrasquilla & Melko 2017, Van Nieuwenburg et al. 2017, Walker et al. 2020)：用 VAE/ML 在 Ising model 上找 phase boundary。

这篇 paper 的核心 insight 是：**把 stochastic generative model 当作一个 statistical physics 系统**，latent code $t$ 当作温度/磁场等热力学参数，生成样本 $x$ 当作 microstate，于是 latent space 上天然有 Fisher information metric。然后用 information geometry 工具研究它。

参考链接：
- Shao et al. 2018 "Riemannian Geometry of Deep Generative Models": https://arxiv.org/abs/1710.11379 (注: 实际是 CVPR Workshop 2018 paper)
- Park et al. 2023 "Understanding the latent space of diffusion models through the lens of riemannian geometry" NeurIPS: https://arxiv.org/abs/2310.06678
- Arvanitidis et al. 2017 "Latent Space Oddity": https://arxiv.org/abs/1710.11379
- Carrasquilla & Melko 2017 "Machine learning phases of matter" Nature Physics: https://www.nature.com/articles/nphys4035
- Brown et al. 2022 "Union of manifolds hypothesis": https://arxiv.org/abs/2207.02862

## 2. Information Geometry 背景

### 2.1 Fisher metric

给定参数化分布 $p(x|t)$，其中 $t \in S \subset \mathbb{R}^n$ 是参数，Fisher metric 定义为：

$$g_F(t) = \int_{\mathcal{X}} p(x|t) \, \nabla_t \log p(x|t) \, (\nabla_t \log p(x|t))^T dx \tag{4}$$

变量解释：
- $t$ 是 latent parameter (比如 diffusion 中的 noise tensor，或 Ising 中的 $(T,H)$)
- $\nabla_t \log p(x|t)$ 是 score function w.r.t. 参数
- 积分是 expectation over $p(x|t)$

Intuition: Fisher metric 度量了"参数 $t$ 变化一点点，KL divergence 增加多少"。$D_{KL}(p(\cdot|t) \| p(\cdot|t+dt)) \approx \frac{1}{2} g_F(t)[dt, dt]$。所以 Fisher metric 大的地方，参数稍微动一动分布就剧烈改变 —— 这就是 phase transition 应该发生的地方。

### 2.2 Exponential Family 与 Hessian metric

Exponential family 形式：

$$p(x|t) = e^{\langle f(x), t \rangle - \log Z(t)} \tag{6}$$

其中：
- $f(x)$ 是 sufficient statistic (统计物理里叫 Hamiltonian 的相关量)
- $Z(t) = \int_\mathcal{X} e^{\langle f(x), t \rangle} dx$ 是 partition function
- $t$ 是 natural parameters

关键性质：**Fisher metric 等于 log-partition function 的 Hessian**：

$$g_F(t) = \nabla^2 \log Z(t) \tag{8}$$

这是 exponential family 的 standard 结果 (Amari 的经典工作)。变量解释：$\nabla^2$ 表示对 $t$ 的二阶偏导矩阵，即 $\frac{\partial^2 \log Z}{\partial t^i \partial t^j}$。

Intuition: $\log Z(t)$ 就是 free energy (差一个 $-1/\beta$ 因子)。它的二阶导数是 susceptibility (磁化率、热容等) —— 物理上这正是 phase transition 处发散的量。

### 2.3 Hessianizability 与 Bryant-Amari-Armstrong 定理

Theorem 2.1 (Bryant-Amari-Armstrong): **任何 2D 解析 Riemannian metric 都局部允许 Hessian 表示**。

这意味着即使 generative model 不是 exponential family (比如 diffusion model)，只要我们取 latent space 的 2D slice，就 **总能** 找到一个 $\log Z(t)$ 使得 $\nabla^2 \log Z$ 等于该 slice 上的 Fisher metric。这是 paper 把方法应用到 diffusion model 的理论基础。

参考:
- Bryant 2013 MathOverflow 回答: http://mathoverflow.net/questions/122308/
- Bryant 2024 "Hessianizability of surface metrics": https://arxiv.org/abs/2405.06998
- Amari & Armstrong 2014 "Curvature of Hessian manifolds": https://www.sciencedirect.com/science/article/pii/S0926225114000160

## 3. 核心理论：从 posterior 反推 partition function

### 3.1 Theorem 3.1 - Bayesian 集中度

设 $x_1, \ldots, x_N \sim p(x|t')$ i.i.d. 从某个真实参数 $t'$ 采样。Posterior:

$$p(t|x_1,\ldots,x_N) = \frac{p(x_1,\ldots,x_N|t) \, p(t)}{\int_S p(x_1,\ldots,x_N|s) p(s) ds}$$

定理说：当 $N \to \infty$，

$$\lim_{N\to\infty} \big(p(t|x_1,\ldots,x_N)\big)^{1/N} \stackrel{a.s.}{=} e^{-D_{\log Z(t)}(t, t')} \tag{11}$$

其中 Bregman divergence:

$$D_{\log Z(t)}(t,t') = \log Z(t) - \log Z(t') - \langle \nabla_{t'} \log Z(t'), t - t' \rangle \tag{12}$$

而且对 exponential family，这个 Bregman divergence 等于 KL divergence $D_{KL}(p(x|t') \| p(x|t))$。

**变量与上标下标解释**：
- $t'$ 是 "true" 参数（数据来源）
- $t$ 是被评估的参数
- $D_{\log Z(t)}(t,t')$ 表示以 $\log Z(t)$ 作为 potential function 的 Bregman divergence
- $\nabla_{t'}$ 表示在 $t'$ 处求梯度

**Proof intuition (我会帮你 build intuition)**：

1. Posterior 的 $N$-th root 把 likelihood $\prod p(x_i|t)$ 变成 $p(x|t)^N$ 形式，再用 $N$-th root 拿掉 $N$。
2. Strong law of large numbers: $\frac{1}{N}\sum f(x_i) \to \mathbb{E}_{x \sim p(\cdot|t')}[f(x)] = \nabla_{t'} \log Z(t')$ (exponential family 性质)。
3. Laplace principle / Varadhan's lemma: $\frac{1}{N} \log \int e^{N\phi(s)} ds \to \max_s \phi(s)$。这里 $\phi$ 是 concave 的 (关于 $s$)，所以 max 在 $s = t'$ 处取到。
4. 把分子分母都化简后，得到 $\exp(-D_B(t,t'))$ 形式。

直观图示 (Fig.2)：posterior 在大 $N$ 下高度集中在 $t'$，尾部按 $e^{-N \cdot D_{KL}}$ 衰减。如果取 $N$-th root，$N$ 这个指数被抵消掉，剩下的"形状"就是 $e^{-D_{KL}}$，**这个形状由 $\log Z$ 完全决定**。

**为什么这个 theorem 重要**：它告诉我们，**只需观察 posterior 的形状**（一个可以直接 fit 的分布），就能反推出 $\log Z(t)$。

### 3.2 Theorem 3.2 - 反推的稳定性

如果两个候选 $\log Z_1, \log Z_2$ 产生的 $e^{-D}$ 在 $L^2$ 意义下接近：

$$\int_S \int_S \big| e^{-D_{\log Z_1}(t,t')} - e^{-D_{\log Z_2}(t,t')} \big|^2 dt \, dt' \to 0 \tag{13}$$

那么它们的 Hessian uniformly 接近：

$$\|\nabla^2 \log Z_1(t) - \nabla^2 \log Z_2(t)\| \to 0 \tag{14}$$

**Proof intuition**：$e^{-D}$ 相等 $\Rightarrow$ $D$ 相等 $\Rightarrow$ $\log Z$ 至多差一个 affine 项 $\langle c, t\rangle + b$ $\Rightarrow$ Hessian 完全相同（affine 项 Hessian 为零）。

这意味着 partition function 只能恢复到 affine 变换 —— 但 paper 关心的是 Fisher metric (即 Hessian)，所以这个 ambiguity 不影响最终目标。

## 4. Method 实操

### Step 1: Approximate posterior $p(t|x_1,\ldots,x_N)$

两条路线：

**(A) 直接训练 mapping (统计物理模型用)**

对 Ising / TASEP 这种 stochastic microstate，没有靠谱的 feature extractor。用 **U²-Net** (Qin et al. 2020) 直接把 sample bundle $\{x_1, \ldots, x_K\}$ 映射成 $S$ 上的 categorical 分布。Loss 是 negative log likelihood of true $t'$:

$$\mathcal{L}(\theta) = -\sum_i \log p_\theta(t_i' | x_{i,1}, \ldots, x_{i,K})$$

Target 分布可以做成 smoothed delta (Gaussian bump on $t'$)，smoothing $\sigma$ 控制精度。

**(B) Feature extractor (diffusion 用)**

由 Theorem 3.1，posterior 形状 $\sim e^{-N \cdot D_{KL}}$。对 exponential family, $D_{KL} \approx d(\mathcal{E}(x_1), \mathcal{E}(x_2))$ 当 $\mathcal{E}$ 是 approximate sufficient statistic (Lemma 18 推导：如果 features 是 Gaussian $\mathcal{N}(\mu_i, I)$，则 $D_{KL} = \frac{1}{2}\|\mu_1 - \mu_2\|^2$)。

Paper 选 CLIP 作为 $\mathcal{E}$。所以 posterior 近似为:

$$p(t|x_1,\ldots,x_N) \approx e^{-\frac{N}{2}\|\mathcal{E}(x) - \mathcal{E}(x')\|^2} \tag{19}$$

变量解释：$x \sim p(x|t)$ 是候选参数对应的 image，$x' \sim p(x|t')$ 是观测到的 image。

**Intuition**: CLIP embedding 距离大 $\Leftrightarrow$ 语义上 KL 大 $\Leftrightarrow$ posterior 在那里取小值。这把 Theorem 3.1 的抽象公式变成了可计算的 image-space 距离。

参考:
- U²-Net: https://github.com/NathanUA/U-2-Net
- CLIP: https://github.com/openai/CLIP

### Step 2: 训练 $\log Z_\theta(t)$

理论 loss (13) 是 MSE between $e^{-D}$，但 Lemma A.6 证明这会 **vanishing gradient** (梯度以 $e^{-C}$ 速度衰减，$C$ 是 Bregman divergence 上界)。原因：posterior 大部分区域近乎零，loss 几乎处处 flat。

Fix: 把 $e^{-N D}$ 归一化为概率分布，用 **Jensen-Shannon divergence** 作为 loss (Eq. 22)：

$$\mathcal{L}_1(\theta) = \int_S D_{JS}\big(p(t|x_1,\ldots,x_N), \, p_{\log Z_\theta}(t|t')\big) dt' \tag{22}$$

其中归一化分布 (Lemma 3.3):

$$p_{\log Z}(t|t') = \frac{\exp\big(-\langle t, \nabla_{t'} \log Z(t')\rangle + \log Z(t)\big)}{\int_S \exp\big(-\langle s, \nabla_{t'} \log Z(t')\rangle + \log Z(s)\big) ds} \tag{21}$$

JSD 定义 (Eq. 23):
$$D_{JS}(P,Q) = \frac{1}{2}[D_{KL}(P\|M) + D_{KL}(Q\|M)], \quad M = (P+Q)/2$$

JSD 是 bounded (≤ $\log 2$) 且处处有梯度，比 KL 更友好。

$\log Z_\theta$ 用 5 层 MLP，hidden size 512，ReLU 激活。**注意：不强制 convexity**，期望训练让它自然收敛到 convex 函数 (因为目标 partition function 是 convex 的)。

### Step 3: Geodesic approximation

有了 $g_F(t) = \nabla^2 \log Z_{\theta^*}(t)$，geodesic 是 Riemannian 长度泛函的极小化曲线：

$$L[\gamma] = \int_0^1 \sqrt{\dot\gamma(t)^T g_F(\gamma(t)) \dot\gamma(t)} \, dt \tag{25}$$

变量解释：
- $\gamma(t)$: 参数空间中的曲线，$\gamma(0)=t_{start}$, $\gamma(1)=t_{end}$
- $\dot\gamma(t) = d\gamma/dt$: 切向量
- $g_F(\gamma(t))$: 在 $\gamma(t)$ 处的 metric 张量
- 平方根里的是 Riemannian 内积 $\langle \dot\gamma, \dot\gamma \rangle_{g_F}$

离散化为 $\{\gamma_0, \gamma_1, \ldots, \gamma_N\}$，用 Adam 优化中间点最小化离散 length。这把 geodesic 计算变成简单的可微优化问题。

## 5. 实验 1：在 Ising 和 TASEP 上的 ground truth 验证

### 5.1 模型

**2D Ising model** (Eq. 1): 微观态 $x = \{s_i = \pm 1\}_{L\times L}$, 参数 $t = (T, H)$, 分布:

$$p(x|T,H) = \frac{1}{Z(T,H)} \exp\left(-\frac{1}{T}\sum_{\langle i,j\rangle} s_i s_j - \frac{H}{T}\sum_i s_i\right)$$

$T_{cr} \approx 2.27$ 处有 phase transition。

**TASEP** (Eq. 2-3): 一维粒子输运模型，参数 $(\alpha, \beta)$ 是边界注入/移除概率。Free energy 解析式:

$$F_{TASEP}(\alpha, \beta) = \begin{cases} 1/4 & \alpha > 1/2, \beta > 1/2 \\ \alpha(1-\alpha) & \alpha < \beta, \alpha < 1/2 \\ \beta(1-\beta) & \beta < \alpha, \beta < 1/2 \end{cases}$$

三个 phase: low-density, high-density, maximal current。

### 5.2 结果 (Table 1)

| ISING | F RMSE | $\frac{dF}{dT}$ RMSE | $\frac{dF}{dH}$ RMSE |
|---|---|---|---|
| **Convex (Ours)** | **0.0883** | **0.1106** | **0.1237** |
| Mean-as-Stat | 0.0981 | 0.4766 | 1.0936 |
| PCA-VAE | 0.1669 | 0.7428 | 0.7988 |

| TASEP | F RMSE | $\frac{dF}{d\alpha}$ RMSE | $\frac{dF}{d\beta}$ RMSE |
|---|---|---|---|
| **Convex (Ours)** | **0.0112** | **0.1165** | **0.1135** |
| Mean-as-Stat | 0.0529 | 0.3832 | 0.3833 |
| PCA-VAE | 0.0524 | 0.3837 | 0.3872 |

**关键观察**: Paper 的方法在 **导数** 上的优势比 free energy 本身大一个数量级。导数对应 heat capacity / magnetization / current —— 这些是检测 phase transition 需要的量。Baseline 在 transition 附近导数重建都很差，因为它们没有利用 exponential family 的几何结构。

参考:
- Onsager 1944 Ising 解析解: https://journals.aps.org/pr/abstract/10.1103/PhysRev.65.117
- TASEP exact solution (Derrida et al. 1993): https://iopscience.iop.org/article/10.1088/0305-4470/26/7/011

## 6. 实验 2：Diffusion model 上的发现

### 6.1 Setup

用 Stable Diffusion 1.5 (Dreamshaper 8 checkpoint), DDIM scheduler, 50 步, CFG=5。构造 2D latent slice:

$$\mathbf{z} = z_0 + \alpha(z_1 - z_0) + \beta(z_2 - z_0)$$

其中 $z_0, z_1, z_2$ 是三个 random latent，$\alpha, \beta \in [0,1]$ uniform。每个 triplet 生成 60000 张图。Latent vector 归一化保持 Gaussian hypersphere 上。

参考:
- Dreamshaper 8: https://huggingface.co/Lykon/dreamshaper-8
- DDIM (Song et al. 2020): https://arxiv.org/abs/2010.02502

### 6.2 主要发现

**(1) Free energy landscape 有 sharp 结构 (Fig. 5, Fig. 7)**

$\log Z(\alpha, \beta)$ 不是平滑的，有 abrupt derivative changes。这些 jump 对应 image 内容的突然切换 (e.g., 山 → 狮子)。Fig. 7C 显示一阶导数有 plateaus (单个 phase 内) 和 jumps (boundary)。

**(2) Fractal phase boundary (Fig. 6) - 最 striking 的发现**

Phase boundary **不是光滑曲线**，而是 **fractal**！放大到 $10^{-5}$ 尺度还能看到 self-similar 的 phase 切换。具体说，latent vector 变化 $10^{-5}$ 就能让输出从山变狮子。这种自相似性一直延伸到 float16 精度极限 $10^{-8}$。

**Intuition**: Diffusion 把 unimodal Gaussian latent 映射到 union of low-dimensional image manifolds (Brown et al. 2022)。这种"压缩并撕裂"的映射必然在边界处产生 Lipschitz singularity —— 你越接近 boundary，越能在任意小尺度上看到 phase 切换。

**(3) Geodesic 在 phase 内线性，在 boundary 非线性 (Fig. 7A, 7B)**

在 single phase 内：geodesic 几乎是直线。这印证了 Shao et al. 2018 关于 latent interpolation 接近 geodesic 的观察 —— 但 paper 指出 **该结论只在 phase 内成立**。

在 phase boundary 上：Fisher metric 不连续，geodesic 必须弯曲绕过 metric 大的地方。Linear interpolation 在 boundary 会"跳过" metric 巨大的区域，造成 image 内容突变。

**(4) Lipschitz constant 在 boundary 处 divergent**

定量观察：在 phase boundary 附近，DDIM 输出对 latent 的敏感度没有上界。这与 Yang et al. 2023 关于 diffusion 在 time 维度上 Lipschitz constant divergent 的结果互补 —— paper 强调的是 **latent 维度**上的发散。

参考:
- Yang et al. 2023 "Lipschitz singularities in diffusion models" ICLR: https://arxiv.org/abs/2306.11838
- Sclocchi et al. 2025 PNAS "Phase transition in diffusion models reveals hierarchical nature of data": https://www.pnas.org/doi/10.1073/pnas.2408799121
- Biroli et al. 2024 Nature Comm "Dynamical regimes of diffusion models": https://www.nature.com/articles/s41467-024-54473-y

## 7. Proposition 4.1 - 理论解释 phase transition 处的发散

### 7.1 Setup

Toy model: target data 是 bimodal Gaussian mixture:

$$p_0(x) = \frac{1}{2}\mathcal{N}(x|-1, \sigma^2) + \frac{1}{2}\mathcal{N}(x|1, \sigma^2) \tag{26}$$

Latent 是 $\mathcal{N}(0,1)$。VP-SDE:

$$dX_t = -\frac{1}{2}\beta X_t dt + \sqrt{\beta} dW_t \tag{27}$$

变量解释：
- $\beta$: diffusion coefficient (noising rate)
- $W_t$: Wiener process
- VP-SDE 保持 variance 不变

### 7.2 结论

Reverse-time ODE 在 $x=0$ (phase boundary 的对应点) 处的 **Lyapunov exponent**:

$$\lambda = \frac{\beta}{2}\left(1 + \frac{1-\sigma^2}{\sigma^4}\right) \tag{28}$$

当 $\sigma \to 0$ (data modes 越来越 sharp，越 disjoint)，$\lambda \to \infty$。

### 7.3 Proof sketch

1. **Reverse-time probability flow ODE** (Eq. 108):
   $$\frac{dX_s}{ds} = -f(X_s, t) + \frac{g(t)^2}{2}\nabla_x \log p_t(X_s)$$
   对 VP-SDE: $f = -\frac{1}{2}\beta x$, $g = \sqrt{\beta}$。

2. **Lyapunov exponent** = 在不动点处 drift 的一阶导数:
   $$\lambda = v'(0) = -f'(0,t) + \frac{g^2}{2} \cdot \frac{d^2}{dx^2}\log p_t(x)\big|_{x=0} \tag{110}$$

3. **Forward noised distribution** (Eq. 112-114):
   - $p_t(x) = \frac{1}{2\sqrt{2\pi}\sigma_1(t)} A(x)$
   - $A(x) = \exp\left(-\frac{(x-\mu(t))^2}{2\sigma_1^2}\right) + \exp\left(-\frac{(x+\mu(t))^2}{2\sigma_1^2}\right)$
   - $\sigma_1^2(t) = e^{-\beta t}\sigma^2 + (1 - e^{-\beta t})$
   - $\mu(t) = e^{-\beta t / 2}$

   变量解释：$\sigma_1(t)$ 是 $t$ 时刻每个 mode 的展宽，$\mu(t)$ 是 mode 中心位置 (随时间向 0 衰减)。

4. **求二阶导** 在 $x=0$ 处 (对称性使一阶导为零)，得到 Eq. 115:
   $$\lambda = \frac{\beta}{2} + \frac{\beta}{2} \cdot \frac{e^{-\beta t} - \sigma_1^2(t)}{\sigma_1^4(t)}$$

5. **取 $t \to 0$**: $\sigma_1(t) \to \sigma$, 得到 Eq. 28。

**Intuition**: $\sigma$ 小意味着 data modes 间距远大于 mode 宽度，分布严重 bimodal。在两个 mode 之间的 valley ($x=0$)，$\log p$ 的曲率正比于 $1/\sigma^2$ 量级 (因为 $\log p$ 在 valley 处有一个 "barrier")，所以 ODE 在 valley 处指数级排斥附近的 trajectory，导致 Lyapunov 发散。

这给了 **diffusion model phase boundary 的数学解释**：data 分布的 multimodality → reverse ODE 在 mode 之间产生 Lipschitz singularity → 对应 latent space 中 Fisher metric 不连续 / 发散的 phase boundary。

## 8. 与相关工作的关系与不同

### 8.1 vs. Shao et al. 2018 / Wang et al. 2021 (pullback metric)

| 维度 | 本 paper | Shao / Wang |
|---|---|---|
| 模型类型 | stochastic (diffusion, Ising) | deterministic (GAN, VAE) |
| Metric 来源 | Fisher (= Hessian of $\log Z$) | Pullback of Euclidean / LPIPS |
| 计算方式 | 训练 $\log Z_\theta$ | Jacobian finite-difference |
| 适用范围 | Exponential family 或 2D slice (Bryant-Amari-Armstrong) | 任意 deterministic |

Table 2 显示，在 deterministic diffusion ($\eta=0$) 下，paper 的方法与 baselines **路径长度相当**，但 **curvature 显著更低** (0.367 vs 1.33)，因为 baselines 用 finite-difference 引入 high-frequency 噪声。Paper 的 metric 天然平滑 (来自 MLP 输出)。

### 8.2 vs. Yang et al. 2023 (Lipschitz singularity in time)

Yang et al. 关注 **time 维度** ($t$ 在 DDIM steps 中) 的 Lipschitz singularity。本 paper 关注 **latent 维度** ($z$) 的 Lipschitz singularity。两者互补：分别在 forward process 时间轴和 reverse process 起点两处揭示 diffusion 的非光滑性。

### 8.3 vs. Sclocchi et al. 2025 / Biroli et al. 2024 (diffusion phase transitions)

这些工作也谈 diffusion 的 phase transitions，但角度不同：
- Sclocchi / Biroli: 在 **time-step** 维度上分析 hierarchy of phases
- 本 paper: 把 latent space 当 parameter space, 分析 latent 上的 phase boundaries

## 9. 我自己的 critical thoughts 和延伸

### 9.1 强项

1. **理论 grounded**: 不像很多 latent space interpretability 工作 (e.g., 寻找"方向")，这里 metric 来自严密的 information geometry，对 exponential family 有 provable convergence。
2. **统一视角**: 把 diffusion model 当 statistical physics 系统，瞬间获得 free energy, susceptibility, phase transition 等强大工具。
3. **Fractal boundary 的发现**是令人兴奋的实证发现，与 Brown et al. union of manifolds 假设高度自洽。
4. **Toy model (Prop 4.1)** 给出了清晰的 mechanism 解释，不是纯 phenomenological。

### 9.2 局限与 open questions

1. **只覆盖 2D slice**: Bryant-Amari-Armstrong theorem 只在 2D 上保证 Hessianizability，所以 method 只能取 2 维切片。整个 latent space (SD 1.5 是 4×64×64 ≈ 16384 维) 上 **不**保证有 Hessian 表示。这意味着只能看局部窗口。

2. **依赖 feature extractor**: 在 diffusion 上靠 CLIP 来近似 KL divergence。Lemma 18 假设 feature 分布 Gaussian — 这个假设多严格？如果 $\mathcal{E}$ 不是 sufficient statistic，方法会引入 systematic bias。U²-Net 直接 mapping 的版本给出更"pixel-level"的结果，但是 deterministic sampler 才行 ($\eta=0$)，加 noise ($\eta=0.1$) 后 free energy 平滑掉了 sharp boundary (Fig. 9)。

3. **Smoothing parameter $\sigma$**: U²-Net 训练时 target 是 smoothed Gaussian with $\sigma = 1/50$ (Ising) or $1/150$ (TASEP)。这决定了 phase boundary 的 resolution。可能存在不同 $\sigma$ 下看到不同 fractal level 的 multi-resolution structure。

4. **MLP 不强制 convex**: 虽然目标 partition function 是 convex 的，但训练用普通 MLP (ReLU)，没有 barrier function 保证 convexity。在数据稀疏区域可能 fit 出非 convex 的 $\log Z$。

5. **Phase 概念稍 loose**: 在统计物理中 phase 有严格定义 (singular free energy derivative)，paper 在 diffusion 上用的是 image content 的 abrupt change。这两者通过 Fisher metric 的 singularity 联系，但严格性需要更细致论证 (例如 finite dataset 上能否区分 numerical noise 与真实 singularity)。

### 9.3 与我熟悉的 ReLU network / KAN 的潜在连接

Andrej 你对 network 训练动力学熟。这个 paper 的发现让我联想到：

1. **Phase boundary = manifold "撕裂处"**: Diffusion 把 Gaussian noise 压缩到 image manifold 上，但 image manifold 是 disconnected union (Brown et al.)。Compression 必然在 mode 之间产生 Lipschitz blow-up (Prop 4.1 formalize 了这一点)。这本质上是 **dimensionality reduction 的代价**。

2. **Fractal boundary 与神经网络 loss landscape**: 这种 self-similarity 让我想到 grokking / phase transition in training dynamics 的研究 (e.g., Nakkiran et al.)。Latent space 上出现 fractal 是否暗示训练时 loss landscape 上也有类似结构？

3. **与 SDE 在 reverse process 上的 chaos 现象**: Lyapunov 发散意味着 reverse ODE 在 boundary 处 chaotic —— 小扰动会指数放大。这可能解释了为什么 diffusion sampling 对 random seed 和 tiny perturbation 极度敏感。

### 9.4 可能的 extensions

1. **3D+ slice**: 把 Bryant-Amari-Armstrong 推广到更高维是 open math problem。如果可能，能看到完整的 phase diagram 而不是 2D 投影。
2. **Time-dependent Fisher metric**: 不只看 final latent $z_0$，看整个 reverse trajectory 上的 metric 演化 (这就接上了 Yang et al. 和 Sclocchi et al.)。
3. **Controllable phase transition**: 既然知道了 boundary 在哪，能否设计 latent perturbation 来 **可控地** 跨越 phase (生成 interpolation)? Fig. 7A 的 geodesic 已 hint 了这点。
4. **分类 phase**: Paper 没有尝试给 phase **命名/分类** (像物理里 ferromagnetic / paramagnetic)。能否用 unsupervised clustering + learned Fisher metric 自动给 diffusion phase 命名?
5. **Class-conditional models**: Class label 是否对应于一个 macroscopic phase? Class boundary 在 latent 上是否对应 Fisher metric singularity?

参考链接 (一些可能有用的延伸阅读):
- Nakkiran et al. "Deep Double Descent": https://openai.com/research/deep-double-descent
- Liu et al. 2021 "Smoothing disentangled latent style space": https://arxiv.org/abs/2103.06321
- Guo et al. 2024 "Smooth diffusion: Crafting smooth latent spaces in diffusion models": https://arxiv.org/abs/2312.01110
- Li et al. 2024 "Self-discovering interpretable diffusion latent directions": https://arxiv.org/abs/2311.18114

## 10. 总结

Paper 的核心 narrative:

> 1. 把 stochastic generative model 看作 statistical physics system → latent space 上自然有 Fisher metric = $\nabla^2 \log Z$
> 2. 用 Bayes posterior 在大样本下的极限形状 (Theorem 3.1) 反推 $\log Z(t)$ (Theorem 3.2 保证 stability)
> 3. 在 Ising / TASEP 上 ground-truth 验证，超越 baseline 一个数量级 (尤其导数)
> 4. 应用到 diffusion model 2D slice：发现 **fractal phase boundary**，Lipschitz 在 boundary divergent
> 5. 用 bimodal Gaussian toy model (Prop 4.1) 给出 mechanism: data 多模态 → reverse ODE 在 mode 间 valley 处 Lyapunov 发散

最大的 takeaway：**diffusion model 的 latent space 不是均匀光滑的几何体，而是由"phase region"组成的镶嵌结构，phase 之间用 fractal boundary 分隔，boundary 上发生 Lipschitz singularity**。这给了 latent interpolation 不平滑这一长期观察一个 clean theoretical explanation，并提供了具体可计算的 metric 来检测 / 绕开这些 singularity。

代码: https://github.com/alobashev/hessian-geometry-of-diffusion-models

希望这些讲解帮到你 build intuition, Andrej！如果想深入任何一个细节 (比如 Laplace principle 在 Theorem 3.1 里的作用，或者 Prop 4.1 的 ODE derivation)，告诉我，我可以再展开。
