---
source_pdf: LeJEPA.pdf
paper_sha256: fe06b5488907f56aa9682e2e2febc552d85fe761fda43f2727e65d60fbeb6485
processed_at: '2026-08-05T14:24:38-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LeJEPA 用人话说

## 一句话总结

让 encoder 吐出来的 embedding 服从 isotropic Gaussian，这种分布对任何下游任务都最友好，而 enforce 这个分布只需要一个简单的 trick：把 embedding 随机投影到一堆一维方向上，每个方向检查是否像 standard normal，就这么简单。

---

## 为什么 embedding 应该是 isotropic Gaussian？用比喻讲

想象 embedding 空间是一块橡皮泥，每个样本就是橡皮泥上的一点。下游任务就是有人闭着眼睛拿刀切这块橡皮泥，切到什么形状就根据那块来预测标签。

**Anisotropic embedding** 就是橡皮泥有的方向被压扁了（像一张薄饼），有的方向很厚（像一块面包）。压扁的方向切出来的样本很少，而下游任务在这个方向上的信息就不够。同时，因为压扁方向上样本稀疏，你每次重新采样橡皮泥（新的训练集），切出来的分布就差别很大——high variance。

**Isotropic embedding** 就是橡皮泥被揉成完美的球，每个方向厚度一样，切哪里都均匀，切多少次结果都稳定。

但光球形还不够，还要 **Gaussian** 的密度。为什么？因为 Gaussian 是所有给定 covariance 的分布里，Fisher information 最小的——也就是说它最"平坦"，gradient $\nabla \log p$ 最小。k-NN 这种 local 方法在 density 陡峭的地方（density 快速变化）会有大 bias，因为附近邻居的标签可能跟 query 差别很大。Gaussian 让 density 变化最温和，bias 最小。

数学上：

$$\text{ISB}_{k\text{-NN}} = \frac{r_0^4}{(K+2)^2} \tau_g^2 J(p) + O(r_0^4)$$

- $r_0$：k-NN 的半径
- $K$：embedding 维度
- $\tau_g^2$：target function 梯度的 isotropic prior 强度
- $J(p) = \int \|\nabla \log p(x)\|^2 p(x) dx$：Fisher information functional

$J(p)$ 越小，bias 越小。Cramér-Rao bound 告诉我们 $J(p) \geq \text{tr}(\Sigma^{-1})$，等号当且仅当 $p$ 是 Gaussian。然后在固定 trace/det/Frobenius 约束下，$\text{tr}(\Sigma^{-1})$ 在 $\Sigma = sI_d$ 时最小，即 isotropic Gaussian。

所以：**isotropic Gaussian = 球形橡皮泥 + 最平坦密度**，对 k-NN、kernel regression、linear probe 都最优。

---

## 传统 SSL 为什么脆？用打地鼠比喻

老式 SSL 像 Whac-A-Mole（打地鼠）游戏：

- **Stop-gradient**：上头按住一个洞（防止 collapse），地鼠从另一个洞冒出来
- **Teacher-student + EMA**：放个假地鼠在洞口骗它（让 student 追 teacher）
- **Negative samples**：旁边放个假地鼠让它别靠近（contrastive）
- **Feature whitening**：把洞口形状规范化（VICReg）
- **Register tokens**：给地鼠搭个假洞让它钻进去

每加一个 trick 都能压住一种 collapse mode，但每种 trick 都 under-specified：你 minimize 了它的 loss，但 embedding 还是可能 degenerate。所以大家要小心翼翼调 hyperparameter、改 architecture、调 EMA schedule、改 view 数量。一旦数据集、模型、batch size 变了，整套 recipe 又得重新调。

更糟的是，这些 trick 互相耦合：去掉 stop-gradient 训练就崩，去掉 EMA 性能就掉，去掉 register tokens attention 就 artifact。整个 SSL pipeline 变成一个精心调好的"配方"，没人知道为什么这么配，只知道少了某个料就不好吃。

LeJEPA 的态度：**别再打地鼠了，从源头把洞填平**。理论告诉你应该 isotropic Gaussian，那就直接 enforce 它，所有 collapse mode 都不可能发生。

---

## SIGReg 怎么 enforce？再一个比喻

直接在高维空间匹配分布，像在三维空间里找一个看不见的物体——你想量它的体积，要遍历所有方向，curse of dimensionality 把你搞死。

**SIGReg 的 trick**：从球面上随机撒一把方向 $a_1, a_2, \ldots, a_M$，每个方向把 embedding 投影上去 $a^\top z$，得到一维 sample。然后每个一维 sample 跟 standard normal 比，看像不像。

$$\text{SIGReg}_T(\mathbb{A}, \{z_n\}) = \frac{1}{|\mathbb{A}|} \sum_{a \in \mathbb{A}} T(\{a^\top z_n\}_{n=1}^N)$$

- $\mathbb{A}$：采样的方向集合，每个 $a \in S^{K-1}$ 是单位向量
- $T$：一维 normality test
- $z_n = f_\theta(x_n)$：encoder 输出
- $a^\top z_n$：投影到一维

为什么这个 trick 有效？Cramér-Wold theorem 说：两个高维分布相等，iff 它们在所有方向上的投影都相等。所以你只要在足够多方向上检查一维分布像不像 Gaussian，就能推断高维分布是不是 isotropic Gaussian。

为什么不需要太多方向？Theorem 5：

$$\mathbb{E}_a\left[\int |\varphi_a(t) - \varphi_N(t)|^2 dt\right] \leq C(K, \alpha) |\mathbb{A}|^{-2\alpha/(K-1)}$$

- $\alpha$：embedding density 的 Sobolev smoothness
- $K$：embedding 维度
- $|\mathbb{A}|$：方向数

深度网络的输出天然 smooth（ReLU + weight decay 让 $\alpha$ 大），所以 $|\mathbb{A}| = O(K)$ 就够。实验上 $|\mathbb{A}| = 16$ 都能跑，因为 SGD 每步 resample 方向，累积 coverage 线性增长。

---

## 为什么 Epps-Pulley 检验最合适？三选一的故事

论文比较了三种一维 normality test：

### 1. Moment-based (Jarque-Bera)

检查 skewness 和 kurtosis 是否匹配 Gaussian。问题：你只检查 3、4 阶矩，存在无数个分布共享这些矩但完全不是 Gaussian。要 identifiability 就得检查更多矩，但高阶矩的梯度是 $O(k)$ 增长，方差是 $O(k^2 m_{2(k-1)})$，训练直接爆炸。**Identifiability 和 stability 不可兼得**。

### 2. CDF-based (Cramér-von Mises, Anderson-Darling)

比较 empirical CDF 和 theoretical CDF。问题：CDF 需要排序，$O(N \log N)$，多 GPU 同步代价高，排序不可微，要 relax 又引入一堆 hyperparameter。

### 3. CF-based (Epps-Pulley) - 胜出

比较 empirical characteristic function $\hat{\phi}_X(t) = \frac{1}{N}\sum_j e^{itX_j}$ 和 standard normal 的 $\phi(t) = e^{-t^2/2}$：

$$EP = N \int_{-\infty}^{\infty} |\hat{\phi}_X(t) - \phi(t)|^2 w(t) dt$$

- $\hat{\phi}_X(t)$：empirical characteristic function
- $\phi(t) = e^{-t^2/2}$：standard normal 的 CF
- $w(t) = e^{-t^2/\sigma^2}$：Gaussian weight

为什么 CF-based 优秀：

1. **梯度天然有界**：$|e^{itX}| = 1$ 始终，所以 $|\hat{\phi}| \leq 1$，$|\phi| \leq 1$，所有梯度都有界
   $$\left|\frac{\partial EP}{\partial z_i}\right| \leq \frac{4\sigma^2}{N}$$
   
2. **天然可微**：$e^{itX}$ 是 $X$ 的解析函数，无需 relaxation
3. **DDP-friendly**：$\hat{\phi} = \frac{1}{N}\sum_j e^{itX_j}$ 跨 GPU 只需 `all_reduce` 平均
4. **线性复杂度**：$O(N)$ 时间和内存

实操：积分用 17 个点的 trapezoidal quadrature 就够（Figure 20 显示比二次收敛还快），因为 Gaussian integrand 本身光滑。

---

## LeJEPA 的 prediction loss 怎么设计？

采用 DINO 的 multi-view setup：$V_g$ 个 global views + $V_l$ 个 local views。所有 views 都预测 global views 的均值：

$$\mathcal{L}_{\text{pred}} = \frac{1}{V} \sum_{v'=1}^V \|\mu_n - z_{n,v'}\|_2^2$$

其中 $\mu_n = \frac{1}{V_g} \sum_{v=1}^{V_g} z_{n,v}$ 是 global views 的均值。

推导（Section B.6）：原始形式是 all-to-all pairwise distance，展开后 cross-terms cancel，剩下到均值点的距离。简洁。

总 loss：

$$\mathcal{L}_{\text{LeJEPA}} = \frac{\lambda}{V} \sum_{v=1}^V \text{SIGReg}(\{z_{n,v}\}_{n=1}^B) + \frac{1-\lambda}{B} \sum_{n=1}^B \mathcal{L}_{\text{pred}}(\{z_{n,v}\}_{v=1}^V)$$

- $\lambda$：唯一超参数，推荐 0.05
- $B$：batch size
- $V$：view 数量

就这么简单。没有 stop-gradient，没有 EMA，没有 predictor，没有 register tokens。50 行代码。

---

## 实验亮点

### 1. 训练 loss 直接预测下游性能

以前 SSL 训练 loss 毫无意义，必须 labeled downstream task 监控。LeJEPA 训练 loss 和下游 accuracy Spearman 相关 ~85%，加个 scaling $\alpha=0.4$ 到 99%：

$$C^{(\alpha)} = \rho_s\left(\frac{\text{train\_loss}}{\lambda^\alpha}, \text{test\_accuracy}\right)$$

这意味着可以做 label-free model selection、cross-validation。第一个这么有用的 SSL 训练 loss。

### 2. In-domain 击败 frontier transfer

Galaxy10 数据集（11000 galaxy images）实验，21M 参数 ResNet-34 in-domain LeJEPA 击败 630M 参数 DINOv3 transfer learning：

| Setting | LeJEPA ResNet-34 (21M) | DINOv3 ViT-S/16 | DINOv2 Small |
|---------|--------------------------|------------------|---------------|
| 1-shot | 24.27% | 24.71% | 21.05% |
| 10-shot | 53.95% | 44.71% | 36.23% |
| Full (frozen) | 78.17% | 71.38% | 67.62% |
| Full (FT) | 83.28% | 81.60% | 78.34% |

**这是 paradigm shift**：天文、医疗、遥感这些 specialized domain，与其用 frontier model transfer，不如直接 in-domain LeJEPA pretrain。

### 3. 跨架构稳定

50 个 timm models（ResNet, ViT, ConvNeXt, Swin, MaxViT, EfficientNet, LeViT, MobileViT）全部跑通，linear probe 91.5%-95%。无需 architecture-specific tuning。

### 4. 大规模

- ViT-Large/14: ImageNet-1K linear probe 79%
- ConvNeXt-V2 Huge (660M): 78.5%
- ViT-gigantic (1.8B): 训练曲线稳定

### 5. 涌现语义

PCA 可视化自发产生 object-background 分离；self-attention 阈值化自发产生 video object segmentation，无需任何 segmentation label。

---

## 为什么 VICReg 是 SIGReg 的退化？

VICReg 的 loss 是：

$$\mathcal{L}_{\text{VICReg}} = \text{sim}(z, z') + \lambda \cdot [\text{var}(z) + \text{var}(z')] + \mu \cdot [\text{cov}(z) + \text{cov}(z')]$$

var term enforce 每个维度方差为 1，cov term enforce 协方差矩阵非对角元为 0。

如果用 SIGReg 的 degenerate test：

$$T(\{x_n\}) = \text{mean}(\{x_n\})^2 + (\text{std}(\{x_n\}) - 1)^2$$

在 $|\mathbb{A}| \to \infty$ 极限下，SIGReg enforce：
- $\mathbb{E}[Z] = 0$（通过反证：如果 $\mu \neq 0$，取 $a = \mu/\|\mu\|$ 则投影均值非零）
- $\text{Cov}(Z) = I_d$（通过投影方差恒为 1 推出所有对角元为 1，通过 $a = (e_i + e_j)/\sqrt{2}$ 推出非对角元为 0）

**所以 VICReg = SIGReg 只匹配前两阶矩**。但 Theorem 3 说前两阶矩不足以 identifiability，VICReg 仍会 collapse。SIGReg 用 characteristic function 匹配整个分布，理论更鲁棒。这是为什么 LeJEPA 不用 VICReg 的 trick——VICReg 在某些 collapse mode 下 loss 仍是 0。

---

## 对 SSL 哲学的意义

LeCun 一直说：world model 应该在 latent space 预测，不在 pixel space 重建。这是 JEPA 的核心 vision。但 JEPA 之前没有理论 foundation，全靠 heuristic 拼凑，大家将信将疑。

LeJEPA 给出 closed-form answer：
1. **理论**：embedding 应该是 isotropic Gaussian（对 worst-case downstream risk 最优）
2. **算法**：用 sketching + CF matching 高效 enforce
3. **工程**：1 个超参数，50 行代码，无 heuristics

这把 SSL 从 alchemy 拉回到 mathematics。以后做 SSL 研究不再问"加什么 trick 能 prevent collapse"，而是问"我的目标分布对不对，enforce 得够不够 efficient"。

---

## 开放联想

1. **Multimodal CLIP-style**：image encoder 和 text encoder 各自 SIGReg 到 isotropic Gaussian，prediction loss 用 contrastive 或 cosine。理论应该 generalize。

2. **Video / time-series**：JEPA 的原始 motivation 是预测下一帧。LeJEPA 在 video 上用 temporal views，SIGReg 应该直接适用。

3. **Language models**：BERT-style masked prediction 是 JEPA 的 text 版本。是否可以用 SIGReg 替换 layer norm 或其他 normalization？理论上有意思。

4. **Reinforcement learning**：robotics 的 world model 学习 latent dynamics。LeJEPA 的 isotropic Gaussian prior 对 state representation 应该有 regularization 作用。

5. **Generative models**：score matching 和 diffusion models 的 connection。SIGReg 的 sketching 思想可能简化 score matching 的高维估计。

6. **理论深化**：Sobolev smoothness $\alpha$ 对 ReLU network 的精确估计？LeJEPA 的 generalization bound？

7. **Optimal $\lambda$**：论文用 $\lambda = 0.05$ 一刀切，是否可以 task-adaptive？

8. **Online learning**：SIGReg 在 streaming data 下是否稳定？$O(1/N)$ bias 在小 batch 下是否 accumulate？

---

## Web Links

- LeJEPA paper (推断 arXiv): https://arxiv.org/abs/2510.05949
- GitHub repo (推测): https://github.com/randbalestriero/LeJEPA 或 https://github.com/facebookresearch/le-jepa
- JEPA 原始 essay (LeCun 2022): https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- VICReg: https://arxiv.org/abs/2105.04906
- Barlow Twins: https://arxiv.org/abs/2103.03230
- SimSiam: https://arxiv.org/abs/2011.10566
- BYOL: https://arxiv.org/abs/2006.07733
- MoCo v3: https://arxiv.org/abs/2104.02057
- SimCLR: https://arxiv.org/abs/2002.05709
- Epps-Pulley test (original 1983): https://academic.oup.com/biomet/article-abstract/70/3/723/258788
- Cramér-Wold theorem: https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Wold_theorem
- Sliced Score Matching: https://arxiv.org/abs/1905.07088
- Sliced Wasserstein Distance: https://arxiv.org/abs/1506.02438
- Neural Collapse (Papyan et al.): https://www.pnas.org/doi/10.1073/pnas.2015509117
- Register Tokens (Darcet et al.): https://arxiv.org/abs/2309.16588
- timm library: https://github.com/huggingface/pytorch-image-models
- PyTorch DDP: https://pytorch.org/docs/stable/distributed.html
- Randall Balestriero homepage: https://randbalestriero.github.io/
- Yann LeCun homepage: https://yann.lecun.com/
- Meta FAIR: https://ai.meta.com/
- I-JEPA + STOP (Bar et al.): https://arxiv.org/abs/2308.00566
- Sobolev spaces (Adams & Fournier): https://www.elsevier.com/books/sobolev-spaces/adams/978-0-12-044143-3
- Trapezoidal rule convergence: https://en.wikipedia.org/wiki/Trapezoidal_rule
- Hyperspherical harmonics & MZ inequalities: https://www.sciencedirect.com/science/article/pii/S0377042705001568
- Helmholtz "Handbook of Physiological Optics" 1867
- Tolman cognitive maps: https://psycnet.apa.org/record/1949-04205-001
- Friston Free Energy Principle: https://www.nature.com/articles/nrn2787
- LeCun "A Path Towards Autonomous Machine Intelligence": https://openreview.net/forum?id=BZ5a1r-kVsf
- DINOv2 official code: https://github.com/facebookresearch/dinov2
- VICReg official code: https://github.com/facebookresearch/vicreg
- Characteristic function in probability: https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)
- Cramér-Rao bound: https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound
- Kernel MMD: https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
- Entropy estimation (Székely-Rizzo): https://www.sciencedirect.com/science/article/pii/S0047259X04000535
- Sobol sequence / QMC: https://en.wikipedia.org/wiki/Sobol_sequence
- Henze-Zirkler test (multivariate normality): https://www.sciencedirect.com/science/article/pii/S0167947300001013
- Jarque-Bera test: https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test
- Anderson-Darling test: https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test
- Watson test: https://en.wikipedia.org/wiki/Watson%27s_U%C2%B2_test
- Shapiro-Wilk test: https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
- Bootstrap and resampling in SSL: https://arxiv.org/abs/2304.12210 (Balestriero et al. SSL cookbook)
- Gaussian embeddings (Balestriero et al. 2025): https://arxiv.org/abs/2510.05949

---

## 最后再给个 take-away

LeJEPA 让 SSL 变成"**让 embedding 是个球**"这么简单一件事。以前大家觉得 SSL 是 alchemy，需要上百个 trick 组合，需要 EMA、stop-gradient、register tokens、negative samples、careful hyperparameter tuning。LeJEPA 说：去掉所有这些，直接 enforce embedding 服从 isotropic Gaussian，用 random projections + characteristic function matching 高效实现，1 个超参数，50 行代码。

理论告诉你应该这么做，算法告诉你怎么高效做，实验告诉你确实 work。这就是 first principles 的力量。

对 Andrej 这种喜欢从底层重新思考的人来说，这篇 paper 应该很有共鸣——你不需要继续在老 framework 上 patch，而是问"什么才是 fundamental 的"，然后从那里重新构造。这正是 LeJEPA 做的事。

---

# LeJEPA: 从第一性原理推导 JEPA 的最优形式

这篇论文是 Randall Balestriero 和 Yann LeCun 的合作,核心 contribution 极其优雅:从第一性原理出发,严格证明了 JEPA 的 embedding 应该服从 **isotropic Gaussian** 分布,并且设计了一个 clean 的 regularization (SIGReg) 来 enforce 这个分布,从而完全消除传统 JEPA 的 collapse 问题和各种 heuristics。

---

## 1. 核心问题:JEPA 的 "anti-collapse" 困境

JEPA (Joint-Embedding Predictive Architecture) 的目标非常直观:让 encoder $f_\theta$ 产生的 embeddings 在 semantically related views 之间互相 predictable。形式化:

$$\text{JEPA} \iff \text{Enc}(x_{n,t+1,\cdot}) \text{ is predictable from } \text{Enc}(x_{n,t,\cdot}), \forall n,t$$

同时要求 $\text{Enc}$ 是 non-degenerate 的。

但问题在于:**predictability 单独 admits collapse**。如果所有 input 都被 map 到同一个点,prediction loss 自然为零,但 representation 毫无用处。这是 JEPA 的根本 pathology。

传统解决方法都是 heuristics 堆叠:
- **stop-gradient** (SimSiam, BYOL)
- **teacher-student + EMA** (DINO, MoCo, I-JEPA)
- **asymmetric views** (DINO, I-JEPA)
- **negative samples** (SimCLR, MoCo)
- **feature whitening** (W-MSE, VICReg)
- **batch covariance regularization** (Barlow Twins, VICReg)
- **register tokens** (DINOv2, DINOv3)

这些 heuristics 全部 under-specified:它们能 minimize 各自的 loss 同时让 embedding 处于 degenerate configuration。而且 hyperparameter sensitive, architecture specific, 跨 domain 需要重新调参。

LeJEPA 的核心 insight:与其继续 patch 这些 heuristics,不如从 first principles 出发,问一个更基本的问题:**encoder 的 embedding 应该服从什么分布,才能 minimize 所有可能下游任务的 expected risk?**

---

## 2. 核心理论:Isotropic Gaussian 是最优 embedding 分布

### 2.1 Linear Probing 分析

考虑 standard linear probe (ridge regression):

$$\hat{\beta} = \arg\min_{\beta \in \mathbb{R}^K} \|y - Z\beta\|_2^2 + \lambda \|\beta\|_2^2$$

变量含义:
- $Z \in \mathbb{R}^{N \times K}$: $N$ 个样本的 embedding 矩阵,每个 embedding 是 $K$ 维
- $y \in \mathbb{R}^N$: 标签向量(假设 univariate,multivariate 类似)
- $\beta \in \mathbb{R}^K$: 线性 probe 参数
- $\lambda \geq 0$: Tikhonov (ridge) 正则化强度
- $\hat{\beta}$: 估计的 probe 参数

closed-form 解:
$$\hat{\beta} = (Z^T Z + \lambda I)^{-1} Z^T y$$

假设 ground truth $y = Z \beta_{\text{true}} + \varepsilon$,其中 $\mathbb{E}[\varepsilon] = 0$。

#### Lemma 1: 各向异性放大 bias

Bias 为:
$$\text{Bias}(\hat{\beta}) = \mathbb{E}[\hat{\beta}] - \beta_{\text{true}} = -\lambda (Z^T Z + \lambda I)^{-1} \beta_{\text{true}}$$

设 $Z^T Z = Q \Lambda Q^T$ (eigendecomposition),$\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_K)$ 是特征值。

考虑两种 embedding:
- $Z_{\text{iso}}$: 所有特征值 $\lambda_k = \bar{\lambda} = \frac{1}{K}\sum_k \lambda_k$ (各向同性)
- $Z_{\text{aniso}}$: 原始特征值 (各向异性)

两者 trace 相同 (能量相同),column span 相同,但几何不同。

构造一个 adversarial 任务:让 $\beta_{\text{true}} = \kappa \cdot \mathbf{q}_p$,其中 $\mathbf{q}_p$ 是最小特征值 $\lambda_p$ 对应的 eigenvector。则:

$$\|\text{Bias}(\hat{\beta})\|_{\text{aniso}} = \frac{\lambda}{\lambda_p + \lambda} \|\beta_{\text{true}}\|$$
$$\|\text{Bias}(\hat{\beta})\|_{\text{iso}} = \frac{\lambda}{\bar{\lambda} + \lambda} \|\beta_{\text{true}}\|$$

由于 $\lambda_p < \bar{\lambda}$(算术平均大于最小值,严格不等 when anisotropic):
$$\frac{\lambda}{\lambda_p + \lambda} > \frac{\lambda}{\bar{\lambda} + \lambda}$$

所以 anisotropic 总存在下游任务产生更大 bias。**Anisotropy amplifies bias**.

#### Lemma 2: 各向异性放大 variance

当 $\lambda = 0$ 时 (OLS),estimator 是 unbiased:
$$\hat{\beta} = (Z^T Z)^{-1} Z^T y$$

方差:
$$\text{Var}(\hat{\beta} | Z) = \sigma^2 (Z^T Z)^{-1} = \sigma^2 Q \Lambda^{-1} Q^T$$

总方差:
$$\text{tr}(\text{Var}(\hat{\beta})) = \sigma^2 \sum_{j=1}^K \frac{1}{\lambda_j}$$

由 Jensen 不等式 ($1/x$ 是严格凸函数 on $(0, \infty)$):
$$\frac{1}{K} \sum_k \frac{1}{\lambda_k} \geq \frac{1}{\frac{1}{K}\sum_k \lambda_k} = \frac{K}{\sum_k \lambda_k}$$

所以:
$$\text{tr}(\text{Var}(\hat{\beta}))_{\text{aniso}} \geq \text{tr}(\text{Var}(\hat{\beta}))_{\text{iso}}$$

严格不等当各向异性时。**Anisotropy amplifies variance**.

**直觉构建**: 各向异性意味着 embedding 在某些方向"瘦",其他方向"胖"。瘦的方向上,数据稀疏,任何下游任务在那个方向上都不容易学。同时方差巨大,因为 $1/\lambda_p$ 主导。Isotropic 让所有方向"平等胖瘦",平衡 bias 和 variance。

### 2.2 Nonlinear Probing 分析

论文进一步分析 k-NN 和 Nadaraya-Watson kernel regression。这部分数学更复杂,但 conclusion 一致。

#### k-NN 的 bias (Lemma 4)

半径 $r_0$ 内 k-NN 估计:
$$\hat{y}(q) = \frac{1}{|N_{r_0}(q)|} \sum_{n \in N_{r_0}(q)} y_n$$

其中 $N_{r_0}(q) = \{n : \|z_n - q\| \leq r_0\}$。

通过 Taylor 展开和 ball integrals 计算,bias 为:
$$\text{Bias}(q) = \frac{r_0^2}{d+2} \left( \nabla \eta(q)^\top \nabla \log p_z(q) + \frac{1}{2} \Delta \eta(q) \right) + o(r_0^2)$$

变量含义:
- $\eta: \mathbb{R}^K \to \mathbb{R}$: target function,$\eta(z) = \mathbb{P}(Y=1|z)$ for classification
- $p_z$: embedding 的密度
- $\nabla \log p_z$: score function
- $\Delta \eta$: target function 的 Laplacian

#### Theorem 1: Isotropic Gaussian 是 k-NN 的唯一最优

对 bias 平方积分 (假设 isotropic gradient prior $\mathbb{E}[\nabla \eta \nabla \eta^\top] = \tau_g^2 I_d$):

$$\mathbb{E}_z[\text{Bias}(z)^2] = \frac{r_0^4}{(K+2)^2} \tau_g^2 J(p) + O(r_0^4)$$

其中 $J(p)$ 是 **Fisher information functional**:
$$J(p) = \int_{\mathbb{R}^d} \|\nabla \log p(x)\|^2 p(x) dx$$

**关键步骤**: 通过 Cramér-Rao bound 证明 (Lemma 6):
$$J(p) \geq \text{tr}(\Sigma^{-1})$$

等号 iff $p = \mathcal{N}(0, \Sigma)$ (Gaussian)。

然后在标量约束下优化 $\text{tr}(\Sigma^{-1})$:
- **Trace constraint** $\text{tr}(\Sigma) = t$: 用 Cauchy-Schwarz,$\Sigma = \frac{t}{d} I_d$ 最优
- **Det constraint** $\det(\Sigma) = \delta$: 用 AM-GM,$\Sigma = \delta^{1/d} I_d$ 最优
- **Frobenius** $\|\Sigma\|_F = c$: 用 Lagrangian,$\Sigma = \frac{c}{\sqrt{d}} I_d$ 最优
- **Spectral radius** $\rho(\Sigma) \leq r$: $\Sigma = r I_d$ 最优

**所有情况都是 isotropic Gaussian 唯一最优!**

**直觉构建**: Fisher information $J(p)$ 衡量 density 的"陡峭程度"。给定 covariance budget,最不陡的 density 就是 Gaussian,而最"均匀"的 Gaussian 就是 isotropic。这等价于让 score function $\nabla \log p$ 在所有方向上同等强度,与 isotropic gradient prior 匹配。Anisotropic Gaussian 在瘦方向上 score 太陡,k-NN 在那里采样不足,bias 爆炸。

### 2.3 Geometric Insight

Figure 3 和 Figure 17 的可视化非常 informative:
- Isotropic embedding 下,OLS $\hat{\beta}$ 在不同训练样本间变化小
- Anisotropic embedding 下,$\hat{\beta}$ 分布像香蕉形,巨大方差
- 这种差异在 logistic regression 上也明显

Figure 18 显示 cosine similarity between $\hat{\beta}$ 和 $\beta_{\text{true}}$:
- Isotropic: 接近 1,与 $\lambda$ 无关
- Anisotropic: 随 regularization 强度急剧下降

---

## 3. SIGReg: 可扩展的 Isotropic Gaussian Regularization

理论告诉我们 target 是 isotropic Gaussian,问题是如何 enforce。直接用 KL divergence 或 kernel MMD 在高维下都 quadratic complexity,curse of dimensionality 严重。

### 3.1 Hypothesis Testing Framework

核心 idea:把分布匹配转化为 hypothesis test:
$$H_0: P_\theta = Q \quad \text{vs.} \quad H_1: P_\theta \neq Q$$

其中 $Q = \mathcal{N}(0, I_K)$ 是目标 isotropic Gaussian。

### 3.2 Hyperspherical Cramér-Wold Theorem (Lemma 3)

经典 Cramér-Wold: $X \stackrel{d}{=} Y \iff \langle X, a \rangle \stackrel{d}{=} \langle Y, a \rangle, \forall a \in \mathbb{R}^D$。

LeJEPA 改成 unit-norm 方向:
$$X \stackrel{d}{=} Y \iff \langle X, a \rangle \stackrel{d}{=} \langle Y, a \rangle, \forall a \in S^{K-1}$$

证明通过 characteristic function 唯一性:任意 $t \in \mathbb{R}^K$ 可写为 $t = s \cdot u$,$s = \|t\|, u = t/\|t\| \in S^{K-1}$,所以 $\varphi_X(t) = \mathbb{E}[e^{is\langle u, X\rangle}]$ 完全由单位球面上的投影决定。

### 3.3 SIGReg 定义

$$\text{SIGReg}_T(\mathbb{A}, \{f_\theta(x_n)\}) = \frac{1}{|\mathbb{A}|} \sum_{a \in \mathbb{A}} T(\{a^\top f_\theta(x_n)\}_{n=1}^N)$$

变量:
- $\mathbb{A} = \{a_1, \ldots, a_M\}$: 采样的方向集合,每个 $a_m \in S^{K-1}$
- $T$: 一维 statistical test(目标是检验 univariate standard normal)
- $f_\theta(x_n)$: encoder 输出
- $a^\top f_\theta(x_n) \in \mathbb{R}$: 投影到方向 $a$ 的一维 sample

用 average 代替 max (Theorem 2 中的) 是为了避免 sparse gradient。

### 3.4 为什么用 Epps-Pulley 检验?

论文比较了三类 candidate:

#### 3.4.1 Moment-based (Jarque-Bera, Extended JB)

Extended Jarque-Bera:
$$\text{EJB}(u) = \frac{N \hat{\mu}(u)^2}{\hat{\sigma}(u)^2} + \frac{(N-1)(\hat{\sigma}(u)^2 - 1)^2}{2} + \frac{N}{6}\left(\widehat{\text{skew}}(u)^2 + \left(\frac{\widehat{\text{kurt}}(u) - 3}{2}\right)^2\right)$$

问题 (Theorem 3): K moments 不足以识别分布。存在不同分布共享前 K 个 moments。

要 identifiability 就需要 large K,但梯度增长 $\|\nabla m_k\| = O(k)$,Monte Carlo 方差 $O(k^2 m_{2(k-1)})$,训练不稳。**Stability 和 identifiability 矛盾**。

#### 3.4.2 CDF-based (Cramér-von Mises, Anderson-Darling, Watson)

$$T_w = N \int_{-\infty}^{\infty} (F_N(x) - F(x))^2 w(x) dF(x)$$

问题:
- 需要排序,$O(N \log N)$
- 多 GPU 同步代价高
- 不可微 (排序是 non-differentiable),需要 relaxation

#### 3.4.3 CF-based (Epps-Pulley) - 最终选择

$$EP = N \int_{-\infty}^{\infty} |\hat{\phi}_X(t) - \phi(t)|^2 w(t) dt$$

变量:
- $\hat{\phi}_X(t) = \frac{1}{N} \sum_{j=1}^N e^{itX_j}$: empirical characteristic function (ECF)
- $\phi(t) = e^{-t^2/2}$: standard normal 的 CF
- $w(t) = e^{-t^2/\sigma^2}$: Gaussian weight (常用 $\sigma = 1$)

**为什么 Epps-Pulley 优秀**:

1. **有界梯度** (Theorem 4):
$$\left|\frac{\partial EP}{\partial z_i}\right| \leq \frac{4\sigma^2}{N}, \quad \left|\frac{\partial^2 EP}{\partial z_i^2}\right| \leq \frac{C\sqrt{\pi}\sigma^3}{2N}$$

因为 $|e^{itX}| = 1$ 始终成立,而 $|\phi(t)| \leq 1$,所有项都有界。这与 moments 的 $O(k)$ 增长形成鲜明对比。

2. **天然可微**: $e^{itX}$ 是 $X$ 的解析函数
3. **DDP-friendly**: ECF 是 $\frac{1}{N}\sum_j e^{itX_j}$,在 GPU 间只需 `all_reduce` 平均
4. **线性复杂度**: $O(N)$ 时间和内存

### 3.5 Beat Curse of Dimensionality (Theorem 5)

对 Sobolev regularity $\alpha$ 的密度 $p_\theta$:

$$\mathbb{E}_a\left[\int_{\mathbb{R}} |\varphi_a(t) - \varphi_N(t)|^2 dt\right] \leq C(K, \alpha) |\mathbb{A}|^{-2\alpha/(K-1)}$$

其中 $C(K, \alpha) = \frac{2^{2\alpha} \pi^{(K-1)/2} \Gamma(\alpha + \frac{K-1}{2})}{(K-1) \Gamma(\alpha) \Gamma(\frac{K-1}{2})}$。

变量:
- $\alpha$: Sobolev smoothness,$p_\theta \in H^\alpha(\mathbb{R}^K)$
- $|\mathbb{A}|$: 采样方向数
- $K$: embedding 维度
- $\Gamma$: Gamma 函数

**关键 insight**: 深度网络产生的 embedding 因为 ReLU/GELU + weight decay 的 implicit smoothness,$\alpha$ 较大。所以 $|\mathbb{A}| = O(K)$ 就够 $\epsilon$-approximation。理论上 $|\mathbb{A}|^{2\alpha/(K-1)}$ 衰减意味着只要 $|\mathbb{A}|$ 与 $K$ 线性即可。

**SGD 复合效应**: 即便每步 $|\mathbb{A}|$ 小(几百),由于每 step resample 方向,累积 coverage 线性增长。Figure 7 显示 resample 比固定方向好很多。

### 3.6 Synthetic Validation (Figure 6)

论文做了 controlled experiment:在 $D=512$ 维 isotropic Gaussian 中,把前两维换成 adversarial "X" 形分布。即使 $M=16$ directions,SIGReg 能 detect 那 2 个 degenerate 维度并 restore 回 Gaussian。这证明 sketching 在高维下真的有效。

---

## 4. LeJEPA 实现

### 4.1 Prediction Loss

采用 DINO 的 multi-view setup: $V_g$ global views + $V_l$ local views,总 $V = V_g + V_l$ views。

$$\mathcal{L}_{\text{pred}}(\{z_{n,v}\}_{v=1}^V) = \frac{1}{V} \sum_{v'=1}^V \|\mu_n - z_{n,v'}\|_2^2$$

其中 $\mu_n = \frac{1}{V_g} \sum_{v=1}^{V_g} z_{n,v}$ 是 global views 的均值。

推导 (Section B.6):
$$\frac{1}{V_g V} \sum_{v=1}^{V_g} \sum_{v'=1}^V \|z_{n,v} - z_{n,v'}\|^2 = \frac{1}{V} \sum_{v'=1}^V \|\bar{z} - z_{n,v'}\|^2$$

利用 $\|\bar{z}\|^2 = \frac{1}{V_g} \sum_v \|z_{n,v}\|^2$ (因为 $V_g^{-2} \sum_{v,v''} z_v^\top z_{v''} = V_g^{-1} \sum_v \|z_v\|^2$ 在 cross terms cancel)。

### 4.2 LeJEPA 总损失

$$\mathcal{L}_{\text{LeJEPA}} = \frac{\lambda}{V} \sum_{v=1}^V \text{SIGReg}(\{z_{n,v}\}_{n=1}^B) + \frac{1-\lambda}{B} \sum_{n=1}^B \mathcal{L}_{\text{pred}}(\{z_{n,v}\}_{v=1}^V)$$

**只有一个超参数 $\lambda$**,推荐 $\lambda = 0.05$。

### 4.3 PyTorch 实现 (~50 lines)

Algorithm 1 给出 SIGReg:
```python
def SIGReg(x, global_step, num_slices=256):
    # Synced random projection directions
    g = torch.Generator(device=x.device)
    g.manual_seed(global_step)
    A = torch.randn((x.size(1), num_slices), generator=g, device=x.device)
    A /= A.norm(p=2, dim=0)
    
    # Epps-Pulley integration grid
    t = torch.linspace(-5, 5, 17, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)  # target CF for N(0,1)
    
    # Empirical CF
    x_t = (x @ A).unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    
    # Weighted L2 distance
    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = x.size(0) * world_size
    T = torch.trapz(err, t, dim=1) * N
    return T
```

整个实现只有几个 dozen lines。无需 stop-gradient,无需 EMA teacher,无需 predictor,无需 register tokens。

### 4.4 Mini-batch bias (Theorem 6)

$$\mathbb{E}[\hat{\phi}_N(t)] = \phi_\theta(t) + \frac{1}{N}\phi_\theta(t)(1 - |\phi_\theta(t)|^2) \cdot dt$$

所以 Epps-Pulley 的 bias 是 $O(1/N)$,对 $N=16$ minibatch 已经很小。

---

## 5. 关键理论联系

### 5.1 VICReg 是 SIGReg 的退化 (Lemma 9 / Section B.14)

如果用 degenerate test:
$$T(\{x_n\}) = \text{mean}(\{x_n\})^2 + (\text{std}(\{x_n\}) - 1)^2$$

那么 SIGReg 在 $|\mathbb{A}| \to \infty$ 极限下 enforce:
- $\mathbb{E}[Z] = 0$ (通过 $a = \mu/\|\mu\|$ 反证)
- $\text{Cov}(Z) = I_d$ (通过 unit vector 的所有投影方差为 1)

**这就是 VICReg**!所以 VICReg 是 SIGReg 只匹配前两阶矩的退化版。但 Theorem 3 说前两阶矩不足以 identifiability,VICReg 容易 collapse。SIGReg 通过 characteristic function 匹配整个分布,理论更鲁棒。

### 5.2 与 Sliced Score Matching 的相似性

Song et al. 2020 的 sliced score matching 也用 random projections 来 estimate score in generative models。思想类似:高维问题投影到一维。但目标不同——score matching 是 generative,LeJEPA 是 representation。

### 5.3 与 Sliced Wasserstein Distance 的相似性

Bonneel et al. 2015 用 slicing 简化 optimal transport。同样是把高维问题降到一维 Wasserstein 距离,然后平均。

### 5.4 与 Kernel MMD 的关系

Epps-Pulley 用 exact integral 时等价于 kernel MMD(Gaussian kernel),但 MMD 是 $O(N^2)$ 复杂度,SIGReg sketching 后是 $O(N)$。

### 5.5 与 Neural Collapse 的关联

Papyan et al. 2020 发现 supervised learning 末期 embedding 也变成 isotropic (within-class variance 球形化)。这是 supervised 的"自然终点",LeJEPA 通过显式 regularizer 让 SSL 也能达到这个状态。

### 5.6 与 Free Energy Principle

论文 intro 提到 Helmholtz, Tolman, Friston。这些都是 predictive coding / active inference 的先驱。JEPA 可以看作 free energy principle 的工程化实例:brain 在 latent space 预测,而不是 pixel space 重建(后者对应 autoencoder)。

### 5.7 与 I-JEPA / DINOv2 的对比

I-JEPA (Assran et al. 2023):
- 需要 teacher-student + EMA
- 需要 stop-gradient
- 需要 masked image modeling predictor
- 主要针对 ViT

DINOv2 (Oquab et al. 2023):
- 需要 register tokens 防止 attention artifacts
- 需要 careful EMA schedule
- 复杂 hyperparameter tuning

LeJEPA 全部移除这些,并且 architecture-agnostic(50+ timm models 都能跑)。

### 5.8 与 EMA/SWA 的关系

论文发现 SWA (Stochastic Weight Averaging) 对 ViT 有 small boost,但不是必需。SWA 等价于 implicit ensemble,与 isotropic Gaussian 目标在 expectation 上一致。对 ResNet 几乎没影响,因为 ResNet 本身就 smoother。

### 5.9 Register Tokens 的真正原因

Darcet et al. 2023 发现 ViT 需要 register tokens 来吸收 attention sink。LeJEPA 证明这是 training objective conditioning 不好的副产品。SIGReg 让 objective landscape smooth,attention 自然不 collapse,不需要 register tokens。

---

## 6. 实验亮点

### 6.1 训练 Loss 预测下游性能

Figure 10 显示在 (SIGReg, prediction loss) 2D 平面上,downstream accuracy 等高线指向左下角。Spearman 相关 ~85%。

通过简单 scaling:
$$C^{(\alpha)} = \rho_s\left(\frac{\text{train\_loss}}{\lambda^\alpha}, \text{test\_accuracy}\right)$$

$\alpha \approx 0.4$ 时相关 ~99% (Figure 11)!

**这是巨大的实用价值**: SSL 第一次可以做 label-free model selection 和 cross-validation。以前需要 labeled downstream task 监控,现在训练 loss 直接告诉你模型好不好。

### 6.2 In-Domain Pretraining 击败 Frontier Models

Galaxy10 数据集(11000 galaxy images,10 类)实验(Figure 12, Table 3):

| Setting | LeJEPA (ResNet-34, 21M) | DINOv3 ViT-S/16 | DINOv2 Small |
|---------|--------------------------|------------------|---------------|
| 1-shot | 24.27% | 24.71% | 21.05% |
| 10-shot | 53.95% | 44.71% | 36.23% |
| Full (frozen) | 78.17% | 71.38% | 67.62% |
| Full (FT) | 83.28% | 81.60% | 78.34% |

21M 参数的 ResNet-34 in-domain LeJEPA 击败 630M 参数的 DINOv3 transfer!

**这是 paradigm shift**:证明了 principled SSL 能让 in-domain pretraining 在小数据集上复活。Frontier models 在 specialized domain (天文、医疗、遥感) 不再有绝对优势。

### 6.3 跨架构稳定性

Figure 9: 50 个 timm models,8 个 architecture families,< 20M params,在 ImageNet-10 上 LeJEPA 全部跑通,linear probe 91.5%-95% top-1。

包括 ResNet, ViT, ConvNeXt, Swin, MaxViT, EfficientNet, LeViT, MobileViT 等。无需 architecture-specific tuning。

### 6.4 大规模训练

- ViT-Large/14: ImageNet-1K linear probe 79%
- ConvNeXt-V2 Huge (660M): 78.5%
- ViT-gigantic (1.8B): 稳定训练曲线

### 6.5 涌现语义

Figure 14: PCA 可视化显示 LeJEPA 自发产生 object-background 分离(暖色前景,冷色背景)。

Figure 13: ViT self-attention 阈值化自发产生 video object segmentation,无需任何 segmentation label。

### 6.6 Few-shot Transfer (Table 2)

LeJEPA 在 DTD, flowers102, food101 等 fine-grained 任务上击败 I-JEPA,训练只用 100 epochs vs I-JEPA 的 300 epochs。

---

## 7. 限制与开放问题

虽然论文没明说,但可以推断:

1. **Epps-Pulley quadrature**: 17 个 integration points 看起来很少,虽然 trapezoid rule 对 Gaussian integrand 收敛快,但严格的 error bound 需要更多分析。Figure 20 显示更快 than quadratic 收敛,但只是经验。

2. **Sobolev smoothness 假设**: Theorem 5 假设 $p_\theta \in H^\alpha$。ReLU network 的输出是 piecewise linear,$\alpha$ 可能不大。但 paper 实验显示 $|\mathbb{A}| = 16$ 就够,说明实际 $\alpha$ 很大。这需要更深入的理论分析。

3. **Mini-batch bias**: $O(1/N)$ bias 可能在大 batch size 下重要?但 paper 说 $N=16$ 都 OK。

4. **理论 vs 实际最优**: Isotropic Gaussian 是 worst-case risk 最优,但 specific downstream task 可能 prefer 其他分布(如 supervised labels 自然产生 class-conditional clusters)。这是 worst-case vs average-case 的 trade-off。

5. **Generalization to multimodal**: 论文聚焦 vision,但 theory 应该 generalize to language, video, audio。Code 已经在 timm 上跑,但 text encoder 还没 test。

6. **CLIP-style multimodal**: 是否可以用 SIGReg 来 regularize image-text dual encoder?这是 open question。

---

## 8. 我的直觉构建

让我用一个比喻总结 LeJEPA 的核心 insight:

想象 embedding 空间是物理空间。Anisotropic embedding 就像在某些方向被压扁的气球,另一些方向过度膨胀。下游任务就像光线从不同方向照射:
- 光线沿压扁方向照射:看不清(高 bias)
- 光线沿膨胀方向照射:晃动剧烈(高 variance)
- 球形气球:光线均匀,清晰稳定

**Isotropic Gaussian = 球形气球 + Gaussian 密度分布**。

而 SIGReg 的 sketching 技巧,就像从 100 个随机角度给气球拍照,看每个角度的轮廓是否匹配 Gaussian 轮廓。这比直接看 3D 体积简单得多,而且因为气球表面是 smooth 的(Theorem 5),少量角度就够 reconstruct 整体形状。

更深的哲学意义:**LeJEPA 把 SSL 从"heuristic 拼凑"变成了"principled optimization"**。以前 SSL 研究是 trial-and-error,加 stop-gradient 看是否 collapse,加 EMA 看是否稳定。现在变成:你的 embedding 服从什么分布?你能不能 efficiently enforce 这个分布?LeJEPA 给出 closed-form answer。

这也回应了 LeCun 长期倡导的 JEPA vision:world model 在 latent space 预测,不在 pixel space 重建。LeJEPA 证明这条路有理论 ground,并且工程上简单。

---

## 9. References / Web Links

- 论文 arXiv 链接(通过 paper key 推断): https://arxiv.org/abs/2510.05949 (Balestriero et al. 2025 "Gaussian embeddings: How JEPAs secretly learn your data density" 是相关作者前作)
- GitHub repo(论文中提到 "GitHub repo" 但未在 markdown 中给出 URL,可能在正式版本会有): 推测在 https://github.com/facebookresearch/le-jepa 或 https://github.com/randbalestriero/LeJEPA
- JEPA 原始概念: LeCun 2022 "A Path Towards Autonomous Machine Intelligence" https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- VICReg: https://arxiv.org/abs/2105.04906
- Barlow Twins: https://arxiv.org/abs/2103.03230
- SimCLR: https://arxiv.org/abs/2002.05709
- SimSiam: https://arxiv.org/abs/2011.10566
- BYOL: https://arxiv.org/abs/2006.07733
- MoCo: https://arxiv.org/abs/1911.05722
- Cramér-Wold 定理: https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Wold_theorem
- Epps-Pulley test: https://academic.oup.com/biomet/article-abstract/70/3/723/258788
- Sliced Score Matching: https://arxiv.org/abs/1905.07088
- Sliced Wasserstein Distance: https://arxiv.org/abs/1506.02438
- Neural Collapse: https://www.pnas.org/doi/10.1073/pnas.2015509117
- Register Tokens (Darcet et al.): https://arxiv.org/abs/2309.16588
- Sobolev spaces: Adams & Fournier book https://www.elsevier.com/books/sobolev-spaces/adams/978-0-12-044143-3
- Randall Balestriero 个人主页: https://randbalestriero.github.io/
- Yann LeCun NYU 主页: https://yann.lecun.com/
- Meta AI / FAIR: https://ai.meta.com/
- Bar et al. I-JEPA + STOP: https://arxiv.org/abs/2308.00566
- Hyperspherical Cramér-Wold theorem 相关工作: https://www.jstor.org/stable/2332819
- VICReg 原作者 Bardes GitHub: https://github.com/facebookresearch/vicreg
- LeCun free energy / world model essay: https://openreview.net/forum?id=BZ5a1r-kVsf
- DINOv2 official code: https://github.com/facebookresearch/dinov2
- timm library (50 architectures tested): https://github.com/huggingface/pytorch-image-models
- PyTorch DDP all_reduce docs: https://pytorch.org/docs/stable/distributed.html
- Trapezoidal quadrature convergence analysis: https://en.wikipedia.org/wiki/Trapezoidal_rule
- Helmholtz on perception: "Handbook of Physiological Optics" 1867
- Tolman cognitive maps: https://psycnet.apa.org/record/1949-04205-001
- Friston Free Energy Principle: https://www.nature.com/articles/nrn2787

---

## 10. 总结

LeJEPA 是 SSL 领域的 landmark paper,因为它做到了:

1. **理论 rigor**: 严格证明 isotropic Gaussian 是 JEPA embedding 的唯一最优分布(对 linear, k-NN, kernel probe 都成立)
2. **算法 elegance**: SIGReg 用 random projections + characteristic function matching,线性复杂度,有界梯度
3. **工程 simplicity**: ~50 行代码,1 个超参数,无 heuristics
4. **实验 breadth**: 50+ architectures, 10+ datasets, scaling to 1.8B params
5. **实用 impact**: 训练 loss 可做 label-free model selection,in-domain pretraining 击败 frontier transfer

对 Andrej 这种喜欢 first principles 的人来说,这篇 paper 应该很合口味——它把 SSL 从 alchemy 拉回到 mathematics。 JEPA 这个 LeCun 长期倡导的 vision 终于有了理论 foundation 和实际简单 implementation。后续 multimodal extension (image-text, video-audio), reinforcement learning (robotics latent dynamics), language models 都可能基于 LeJEPA 框架重新思考。
