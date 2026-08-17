---
source_pdf: VISReg Variance-Invariance-Sketching Regularization for JEPA training.pdf
paper_sha256: 0f0b3e3782eeac9ecc7231d25d12b4f65807639028db7950bb5226200cfc0baa
processed_at: '2026-08-13T01:56:34-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VISReg 用人话讲

## 1. 这 paper 在干啥

SSL (self-supervised learning) 有个老大难问题: **embedding collapse**。模型要是没约束,会把所有输入都映射到同一个点,啥也没学到。

防 collapse 现在主流两条路:

**第一条路 (heuristics)**: DINO, BYOL, SimSiam 这帮人。用 EMA teacher、stop-gradient、centering、sharpening 这些工程 trick 硬防。管用,但理论不清晰,超参敏感,换个 dataset 可能直接崩。

**第二条路 (explicit regularization)**: VICReg, Barlow Twins 这帮人。直接在 embedding space 上加约束,比如要求每个维度方差别太小、维度之间别太相关。理论清晰,但约束太弱——只管二阶统计量。

LeCun 团队 2025 年搞了个 LeJEPA / SI-GReg,想法更激进: 直接把 embedding 分布"拉"到 isotropic Gaussian。基于 Cramér-Wold theorem,理论上比 covariance 强很多。但有两个毛病: (a) collapse 时 gradient 消失(最需要救的时候没信号);(b) scale 和 shape 耦在一起,调不动。

**VISReg 就是把 VICReg 和 SI-GReg 的优点合并**: scale 用 VICReg 的 variance term(shape loss 不碰它),shape 用 Sliced-Wasserstein Distance(比 covariance 严格,比 Epps-Pulley 有 gradient)。

## 2. 核心直觉: 为什么要 decouple scale 和 shape

想象 embedding 分布是一个气球。气球有两个独立属性:

- **Scale (大小)**: 气球吹多大。对应每个维度的 std。
- **Shape (形状)**: 气球是圆的、扁的、还是长条形。对应分布的几何形状。

VICReg 的问题: covariance 同时管 scale 和 shape,搅在一起。你调一个超参,两个都变。

SI-GReg 的问题: 直接对齐到 isotropic Gaussian,但 Gaussian 的 std=1 是固定的。模型要是想学一个"大一点的"embedding 空间,没办法。

VISReg 的解法:
- **Scale loss**: 每个维度 std 逼近 1。collapse 时 (std→0) gradient = 常数,永远有信号。
- **Shape loss**: 先 normalize 掉 scale (用 stop-gradient),再用 SWD 对齐分布形状到 Gaussian。只管形状,不管大小。

这样你可以在 low-quality dataset 上调高 shape loss 权重(多约束形状),在 high-quality dataset 上用默认权重(让模型自由学)。Table 12 显示这个 decoupling 在 ImageNet-LT 上 +3.2%, 非常显著。

## 3. Sliced-Wasserstein Distance 到底在干啥

高维分布对齐很难(需要估计密度,或者做高维 OT,都贵)。Cramér-Wold theorem 说: 两个高维分布相等 ⟺ 它们所有 1D 投影的分布相等。

这就像: 你看一个 3D 物体,从所有角度拍 2D 阴影。所有阴影一样,物体就一样。

所以 VISReg 的做法:
1. 随机采 K 个方向 $w_k \in \mathbb{R}^D$
2. 把 batch embedding 投影到这些方向: $P_k = \tilde{\mathbf{Z}} w_k \in \mathbb{R}^N$
3. 对每个投影排序,跟标准 Gaussian 的分位数对齐

**1D Wasserstein 的妙处**: 在 1D 情况下,Wasserstein 距离有 closed-form。两个经验分布(各 N 个样本),排序后一一对应,距离平方和就是 W2²。这比通用 OT 算法(需要解 linear programming)快太多了。

公式:
$$\mathcal{L}_{\mathrm{shape}} = \frac{1}{K} \sum_{k=1}^{K} \left\| \mathrm{sort}(\tilde{\mathbf{Z}} w_k) - \mathbf{q}_N \right\|_2^2$$

- $K$: 随机方向数,典型 4096
- $w_k$: 第 k 个随机方向(高斯采样后归一化)
- $\tilde{\mathbf{Z}}$: normalize 后的 embedding (scale 被 stop-gradient 剥离)
- $\mathrm{sort}(\cdot)$: 沿 batch 维度排序
- $\mathbf{q}_N$: 标准 Gaussian 的 N 个分位数, $q_N^{(i)} = \Phi^{-1}(\frac{i}{N+1})$

**人话**: 把 batch 沿一个方向投影成 1D 数轴上的点。理想情况这些点应该按 Gaussian 分布排开(中间密、两头疏)。排序后跟 Gaussian 分位数一一对应,距离越近越好。K 个方向都这么干,就约束了整个高维分布的形状。

这比 covariance 严格在哪? Covariance 只要求"维度间不相关",但单维度可以是任意分布(双峰、重尾等)。SWD 要求每个 1D 投影都是 Gaussian,联合起来就是 isotropic Gaussian。约束强得多,所以学到的 representation 更"通用",OOD 更好。

## 4. 为什么 collapse 时 VISReg 有 gradient 但 SI-GReg 没有

Figure 2 的实验: 模拟 embedding norm r 从大到小(模拟 collapse 过程),看不同方法 gradient 的 norm。

- **Barlow Twins**: gradient 随 r 减小而增大,接近 collapse 时 gradient 最强。好。
- **VISReg**: 类似 Barlow Twins,collapse 时 gradient 强。好。
- **SI-GReg**: gradient 随 r 减小而减小,collapse 时趋近 0。坏——最需要救的时候没信号了。

**直觉解释**:
- VISReg 的 scale loss: $(1 - \sigma)^2$。当 $\sigma \to 0$ (collapse),$L \to 1$,gradient $= -2(1-\sigma) \to -2$,常数。
- SI-GReg 用 Epps-Pulley test(频域特征函数匹配),它的信号依赖于"分布的非高斯性"。Collapse 时分布变成一个 delta function,频域特征变成常数,跟 Gaussian 的差异... 信号反而弱了。

所以 VISReg 在"快要崩"时能给模型一记重拳把它拉回来,SI-GReg 就软绵绵的。

## 5. 复杂度的猫腻

表面看 VISReg 复杂度 $O(NDK)$,N batch, D dim, K slices,三个都线性,比 VICReg 的 $O(ND^2)$ 好。

但实际实验发现: **K 必须大于 D 一个因子 C > 1** 才能达到最优精度(Figure 4)。为什么? Cramér-Wold 说要"所有方向",实践中 K 个随机方向采样。D 维球面上,K 太稀疏覆盖不够,分布对齐就不准。

所以真实复杂度是 $O(ND \cdot CD) = O(CD^2 N)$,跟 VICReg 差不多? 论文的 workaround: **分布式 GPU 各采独立的 K/M 个 slices**。Figure 6 显示 8 个 GPU 各 128 slices ≈ 单 GPU 1024 slices 的效果。所以单 GPU 上 K 保持常数,靠加 GPU 拉总 slices,复杂度真的线性。

这是个实用的 scaling 观察: 你 scaling 模型时,不用按 D² 增加 K,加 GPU 就行。

## 6. 实验里最 striking 的结论

### 6.1 OOD 性能 (Table 6) —— 真正的卖点

6 个 OOD dataset: DTD, Galaxy10, AID, ChestXRay, RetinaMNIST, OrganAMNIST,涵盖医学、天文、遥感。

| 方法 | Backbone | 训练数据 | OOD avg |
|------|----------|---------|--------|
| DINOv2 | ViT-L/14 | LVD-142M (142M 图) | 72.93% |
| VISReg-Inet22K | ViT-L/14 | ImageNet22K (14.2M 图) | **72.94%** |

VISReg 用 **1/10 数据**追平 DINOv2。这是 paper 最强的结论。

**为什么 VISReg OOD 这么好**: SWD 强制 embedding 分布匹配 isotropic Gaussian,约束极强。模型不能 overfit ImageNet 的 in-domain 偏置(比如靠某些 spurious correlation 线性可分),必须学更"通用"的特征。Covariance 只管二阶矩,模型有大量自由度去 overfit。

### 6.2 Low-quality dataset 鲁棒性 (Table 1, 2)

ImageNet-LT (long-tailed) ViT-S/8:
- DINO: **5.13%** (直接 collapse!)
- VICReg: 33.08%
- VISReg* (shape weight 加重): **35.14%**

DINO 的 EMA + centering + sharpening 在长尾分布上失灵了。VISReg 的 explicit regularization 不依赖这些 heuristic,更 robust。

而且 Table 12 显示在 low-quality dataset 上调高 shape loss 权重(2:1 或 4:1)显著提升性能,high-quality dataset 用默认 1:1:1 最好。**这正是 decouple scale/shape 的价值**——给你一个调节旋钮。

### 6.3 Transfer Learning (Table 7)

ViT-B/16 fine-tuning:
- ImageNet1K: VISReg 83.0% vs DINO 82.8%
- Flowers: VISReg 99.0% vs DINO 98.8%
- Galaxy10: VISReg 87.0% vs DINO 86.6%

注意 VISReg 在 in-domain linear probe 上比 DINO 低 3%,但 fine-tuning 反超。这说明 VISReg 学的特征更"可塑"——不是 linear separable 但 brittle,而是需要 fine-tuning 才发挥,但跨任务更通用。

### 6.4 Dense Prediction (Table 8) —— 局限

ADE20K linear segmentation mIoU:
- MoCoV3: 31.69 (best)
- DINO: 29.40
- VISReg: 30.16
- MAE: 23.60

VISReg 不及 MoCoV3 和 iBOT。作者承认这是 future work。**可能原因**: covariance / InfoNCE 类目标天然鼓励 spatial structure(因为正负样本对比 patch),VISReg 的 distributional target 只管全局分布形状,不直接优化 patch-level 结构。

### 6.5 Image Generation Guidance (Table 9)

用 VISReg features 指导 SiT-B/2 生成:
- DINO guidance: gFID 41.15
- VISReg guidance: gFID 40.36

VISReg 更好。与 2025 年 "Representation alignment for generation" 的方向一致——好的 SSL representation 能加速 diffusion 训练。

## 7. 整体 pipeline 一图概括

```
Image
  │
  ▼
[Multi-crop: 4 global + 6 local views]  (DINO-style)
  │
  ▼
[ViT Backbone — 共享权重,NO EMA, NO teacher-student]
  │
  ▼
[CLS tokens from last 2 layers, concat]
  │
  ▼
[3-layer MLP projector: 2048→2048→d_p, BN+GELU]
  │
  ▼
Embedding Z ∈ R^(N·V × D)
  │
  ├───────────────────────┬───────────────────────┐
  ▼                       ▼                       ▼
L_pred (invariance)     L_scale (variance)     L_shape (SWD)
mean(global views) vs   每维 std → 1           sort(Z·w_k) → q_N
each view               collapse 时有 gradient   K 个随机方向
                                               stop-grad on std
  │                       │                       │
  └───────────────────────┼───────────────────────┘
                          ▼
                      L_center = ||μ||²
                          │
                          ▼
      L = (1-λ)·L_pred + λ·(λ_s·L_scale + λ_h·L_shape + λ_c·L_center)
```

**设计哲学**: VICReg 三个 loss 都搅在 Z 上,互相干扰。VISReg 用 stop-gradient 把 scale 和 shape 完全隔离,各管各的。这从 "engineering trick" 升级到 "principled decomposition"。

## 8. 我觉得最 clever 的几个点

**8.1 Stop-gradient 从 trick 变成 principle**: BYOL/SimSiam 用 stop-gradient 是"不这样做就 collapse",理论含糊。VISReg 用 stop-gradient 是"我要把 scale 和 shape 优化解耦",有明确数学含义。这种从 empirical 到 principled 的进化,是 SSL 走向成熟的方向。

**8.2 1D Wasserstein 的 closed-form**: 1D 情况下 Wasserstein 距离 = 排序后样本的 Lp 距离。这把高维 OT 问题(贵)变成排序 + L2(便宜)。配合 Cramér-Wold 用 K 个随机方向覆盖,既严格又高效。这是 SWD 在 generative modeling (SWGAN, SWAE) 成熟后第一次系统引入 SSL regularization。

**8.3 K vs D 的分布式 workaround**: K 必须 > D 是 Cramér-Wold 近似的内在限制。作者发现各 GPU 独立采样 slices 可以累积,变相让 K 保持常数。这是工程上的优雅处理——理论限制用分布式架构绕过。

**8.4 OOD 性能 vs in-domain 性能的 trade-off**: VISReg 在 in-domain linear probe 上不如 DINO/iBOT,但 OOD 和 transfer 更好。这揭示一个深层问题: in-domain linear separability 可能是 overfit 的表现。DINO 的 heuristics (EMA, centering) 可能在 in-domain 上 overfit ImageNet 的偏置,导致 OOD 弱。VISReg 的强 distributional constraint 防止这种 overfit。

## 9. Limitations 和 future work

**9.1 Dense prediction 弱**: ADE20K 上不及 MoCoV3。shape loss 没有空间结构约束,可能需要加 patch-level SWD 或 spatial-aware variant。

**9.2 in-domain linear probe 有 gap**: ViT-L 上 VISReg 77.0%,iBOT-L 81.0%。如果 OOD 优势能保持,in-domain 小 gap 可以接受,但实际应用中 in-domain 也很重要。

**9.3 Target distribution 是固定 Gaussian**: 如果用 heavy-tailed (student-t) 或 mixture target,在 low-quality data 上可能更鲁棒。paper 没探索。

**9.4 K 的理论最优值**: 论文经验上 K > D 的因子 C,但 C 怎么随 D 变化没给理论分析。Cramér-Wold 的 finite sample bound 可能能推。

## 10. 与 LeCun 路线图的关系

LeCun 2022 的 "A Path Towards Autonomous Machine Intelligence" 提出 JEPA 架构。演进路线:

```
I-JEPA (2023) → V-JEPA → DINOv2 (2024) → LeJEPA/SI-GReg (2025) → VISReg (2025) → DINOv3 (2025)
```

I-JEPA 用 EMA teacher(有 heuristic)。DINOv2 大规模数据 + EMA + centering + sharpening。LeJEPA 证明 sketching to Gaussian 可以去 heuristic,但 collapse 时 gradient 弱。VISReg 解决了 gradient 问题,同时保留 distributional rigor。

这条路线的核心信念: **heuristic-free SSL 是可能的**,而且可能比 heuristic-based 更 data-efficient、更 robust、更 OOD-friendly。VISReg 用 ImageNet22K 追平 DINOv2 的 LVD-142M 是支持这个信念的最强证据。

## 11. Reference Web Links

**核心论文与代码**:
- VISReg 项目与代码: https://haiyuwu.github.io/visreg
- VICReg: https://arxiv.org/abs/2105.04906
- VICReg 代码: https://github.com/facebookresearch/vicreg
- LeJEPA / SI-GReg: https://arxiv.org/abs/2511.08544
- I-JEPA: https://arxiv.org/abs/2301.08243
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- DINO: https://arxiv.org/abs/2104.14294

**Sliced-Wasserstein**:
- SWD Barycenters (Bonneel et al., 2015): https://link.springer.com/article/10.1007/s10851-014-0505-0
- Generative Modeling with SWD (Deshpande et al., 2018): https://arxiv.org/abs/1803.11188
- Computational Optimal Transport (Peyré & Cuturi): https://arxiv.org/abs/1803.00567

**理论基础**:
- Cramér-Wold 原始论文: https://londmathsoc.onlinelibrary.wiley.com/doi/abs/10.1112/jlms/s1-11.4.290
- Epps-Pulley test: https://academic.oup.com/biomet/article-abstract/70/3/723/242936
- LeCun JEPA 路线图: https://openreview.net/pdf?id=BZ5a1r-kVsf

**对照方法**:
- Barlow Twins: https://arxiv.org/abs/2103.03230
- W-MSE: https://arxiv.org/abs/2007.06346
- MAE: https://arxiv.org/abs/2111.06377
- SimSiam: https://arxiv.org/abs/2011.10566
- BYOL: https://arxiv.org/abs/2006.07733
- MoCo v3: https://arxiv.org/abs/2104.02057
- iBOT: https://arxiv.org/abs/2111.07832
- iREPA: https://arxiv.org/abs/2512.10794
- Representation alignment for generation: https://arxiv.org/abs/2410.06985

**Sibling 工作**:
- KerJEPA: https://arxiv.org/abs/2512.19605
- LpJEPA / RDMReg: https://arxiv.org/abs/2602.01456

**Datasets**:
- ImageNet: https://www.image-net.org/
- ImageNet-LT: https://arxiv.org/abs/1904.05116
- Galaxy10: https://astronn.readthedocs.io/en/latest/galaxy10.html
- ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K/
- DTD: https://www.robots.ox.ac.uk/~vgg/data/dtd/

**工具**:
- timm: https://github.com/huggingface/pytorch-image-models
- ViT 原论文: https://arxiv.org/abs/2010.11929

---

**一句话总结**: VISReg 把 VICReg 的 scale 约束 + Sliced-Wasserstein 的 shape 约束用 stop-gradient 解耦,得到一个 heuristic-free、linear complexity、collapse-robust 的 SSL 方法,用 1/10 数据在 OOD 上追平 DINOv2。这是 LeCun 团队 "principled SSL" 路线的关键一步。

---

# VISReg: Variance-Invariance-Sketching Regularization 深度解析

## 1. 背景: SSL regularization 范式的演进

SSL (self-supervised learning) 防止 embedding collapse 的方法大致分两类: **heuristics-based** (EMA, teacher-student, stop-gradient 等隐式机制) 和 **explicit regularization** (显式约束 embedding space 的统计性质)。VISReg 属于后者,在 VICReg 和 SI-GReg 的基础上做了一次精妙的 "merge"。

VICReg (Bardes et al., 2022) 把目标拆成 **Variance + Invariance + Covariance**,工程负担小,但 covariance 只捕获二阶统计量。两个分布可以有相同的 mean 和 covariance,但 shape 可以完全不同(比如高斯 vs 多峰)。

LeJEPA / SI-GReg (Balestriero & LeCun, 2025) 直接 sketching embedding 分布到 isotropic Gaussian,基于 Epps-Pulley test 和 Cramér-Wold theorem,理论上更严格,但有两个问题: (a) collapse 时 gradient 消失(图 2 显示当 feature norm r 变小时,SI-GReg 的 $||\nabla L||$ 趋近 0);(b) scale 和 shape 耦合在一起,缺乏灵活性。

VISReg 的核心 insight: **把 scale (variance) 和 shape (distributional shape) 解耦**,前者用 VICReg 的 variance term,后者用 Sliced-Wasserstein Distance 替代 covariance。这样既保留了 VICReg 的可解释性,又获得了 sketching 方法的 distributional rigor,并且在 collapse 时仍提供强 gradient。

## 2. 方法核心公式详解

### 2.1 Scale Regularization (控制尺度)

给定 centered embedding $\hat{\mathbf{Z}} \in \mathbb{R}^{N \times D}$(N 是 batch size,D 是 projection dimension):

$$\mathcal{L}_{\mathrm{scale}} = \frac{1}{D} \sum_{j=1}^{D} (1 - \sigma_j(\hat{\mathbf{Z}}))^2$$

变量解释:
- $D$: projection dimension (输出 embedding 的维度)
- $j$: dimension index,遍历 $1$ 到 $D$
- $\sigma_j(\hat{\mathbf{Z}})$: 第 $j$ 维的标准差
- 目标:让每个维度的 std 接近 1,防止某一维或整体 collapse 到 0

**直觉**: 直接 KL 散度到 isotropic Gaussian 需要 $O(D^3)$ 复杂度,通过 marginal factorization 降到 $O(D)$。当 embedding collapse (std → 0) 时,$(1 - 0)^2 = 1$,gradient 是常数,提供可靠 corrective signal。这正是 SI-GReg 缺失的特性。

### 2.2 Shape Regularization (控制分布形状)

第一步,**normalize** 把 scale 从 shape 中剥离:

$$\tilde{\mathbf{Z}} = \frac{\hat{\mathbf{Z}}}{sg(\sigma) + \epsilon}$$

- $sg(\cdot)$: stop-gradient 操作,前向传播时数值照常,反向传播时 gradient 不流回 $\sigma$
- $\epsilon$: 防止除零的小常数
- **关键作用**: 让 shape loss 的 gradient 不会"顺手"调节 std,从而干扰 scale loss 的优化。BYOL/SimSiam 也用 stop-gradient 作为 collapse 防御 heuristic,但 VISReg 这里是 **objective decomposition 的原则性使用**,概念完全不同。

第二步,基于 **Cramér-Wold theorem** 把高维分布对齐问题转化为 1D 投影对齐:

**Lemma 3.1 (Cramér-Wold)**: 两个 $\mathbb{R}^d$ 上的概率测度 $\mu, \nu$ 相等,当且仅当它们沿所有方向 $\theta \in \mathbb{S}^{d-1}$ 的 Radon transform 相等:
$$\mu = \nu \iff \mathcal{R}\mu(\theta, \cdot) = \mathcal{R}\nu(\theta, \cdot), \quad \forall \theta$$

Radon transform $\mathcal{R}\mu(\theta, t) := \int_{\mathbb{R}^d} \delta(t - \langle x, \theta \rangle) d\mu(x)$ 实际就是沿方向 $\theta$ 投影后的 1D 分布。

**直觉**: 高维分布的"形状"完全由它所有 1D 投影的集合决定。这就像你看一个 3D 物体,从所有角度拍 2D 阴影照片,这些照片集合能唯一确定物体本身。

第三步,对每个随机投影方向 $w_k \in \mathbb{R}^D$,计算 1D 投影 $P_k = \tilde{\mathbf{Z}} w_k$,然后用 **1D Wasserstein-2 距离**对齐到标准高斯分位数:

**Lemma 3.2 (1D Wasserstein Closed-Form)**:
$$\mathcal{W}_p^p(\hat{\mu}, \hat{\nu}) = \frac{1}{N} \sum_{i=1}^{N} \|x_{(i)} - y_{(i)}\|^p$$

- $x_{(i)}, y_{(i)}$: 第 $i$ 个 order statistic(即排序后第 $i$ 位的值)
- $p=2$ 时:经验分布间的 Wasserstein 距离 = 排序后样本的 $L_2$ 距离

最终 shape loss:
$$\mathcal{L}_{\mathrm{shape}} = \frac{1}{K} \sum_{k=1}^{K} \left\| \mathrm{sort}(\tilde{\mathbf{Z}} w_k) - \mathbf{q}_N \right\|_2^2$$

- $K$: 随机投影方向的个数(slices)
- $w_k$: 第 $k$ 个随机方向,通常从 $\mathbb{S}^{D-1}$ 上采样(高斯采样后归一化)
- $\mathrm{sort}(\cdot)$: 沿 batch 维度排序
- $\mathbf{q}_N \in \mathbb{R}^N$: 标准 Gaussian 的固定分位数,即 $q_N^{(i)} = \Phi^{-1}\left(\frac{i}{N+1}\right)$,其中 $\Phi^{-1}$ 是标准正态的 inverse CDF (PPF)

**直觉**: 把 batch 投影到一个 1D 数轴,排序后,理想情况应该匹配标准正态的 quantile。这等价于在 1D 上做"optimal transport"到 Gaussian——sorted values 一一对应,距离之和就是 Wasserstein 距离。比 covariance 强很多:covariance 只要求 decorrelation,这要求 marginal 在每个投影方向都是 Gaussian,严格更强。

### 2.3 Center Regularization

$$\mathcal{L}_{\mathrm{center}} = \|\mu\|_2^2$$

$\mu$ 是 batch mean,目标让 embedding 中心在原点。论文 Table 10 ablation 显示这个 term 只贡献 0.41% 准确度但加速收敛,所以加上。

### 2.4 Invariance (预测) Loss

$$\mathcal{L}_{\mathrm{pred}} = \frac{1}{V} \sum_{i=1}^{V} \|\mu_g - z_i\|_2^2$$

- $V$: augmentation 数量(views)
- $\mu_g$: global views 的 mean embedding(类似 BYOL 的 target)
- $z_i$: 包括 global 和 local view 的 embedding

完整目标:
$$\mathcal{L}_{\mathrm{VISReg}} = (1-\lambda)\mathcal{L}_{\mathrm{pred}} + \lambda\mathcal{L}_{\mathrm{Reg}}$$

其中 $\mathcal{L}_{\mathrm{Reg}} = \lambda_{\mathrm{scale}}\mathcal{L}_{\mathrm{scale}} + \lambda_{\mathrm{shape}}\mathcal{L}_{\mathrm{shape}} + \lambda_{\mathrm{center}}\mathcal{L}_{\mathrm{center}}$。$\lambda$ 控制 invariance 和 regularization 的相对权重。

## 3. Algorithm 1 代码解析

```python
def visreg(z, K=64):
    # 1. Center loss
    mu = z.mean(dim=0)                    # (D,) batch mean
    L_center = (mu).pow(2).mean()
    
    # 2. Scale loss
    z_cent = z - mu                        # centering
    std = z_cent.std(dim=0, unbiased=False)  # (D,) per-dim std
    L_scale = (1.0 - std).pow(2).mean()
    
    # 3. Shape loss: SWD
    z_norm = z_cent / (std.detach())       # stop-grad on std!
    W = torch.randn(D, K)                  # random projections
    W /= W.norm(p=2, dim=0)                # unit-norm columns
    p = z_norm @ W                         # (N, K) projections
    p_sorted = torch.sort(p, dim=0).values # sort along batch
    u = torch.arange(1, N+1) / (N+1)       # quantile positions
    target = Normal(0, 1).icdf(u)          # Gaussian quantiles
    L_shape = (p_sorted - target).pow(2).mean()
    
    return L_scale + L_shape + L_center
```

注意 `std.detach()` 这一行,这是 stop-gradient 的实现,是整个方法的"灵魂"之一。另外 `Normal(0, 1).icdf(u)` 是预计算的,不参与 gradient。

## 4. 复杂度与 Scaling 分析

定义 $\mathbf{Z} \in \mathbb{R}^{N \times D}$,slices 数 $K$:

$$\mathcal{C}_{\mathrm{Reg}} = \underbrace{O(NDK)}_{\text{projection}} + \underbrace{O(KN\log N)}_{\text{sorting}}$$

当 $\log N \ll D$ 时(大规模训练常态),简化为 $O(NDK)$,对 N, D, K 都是线性的。对比 VICReg 的 $O(ND^2)$(协方差矩阵计算),这非常友好。

**关键 scaling 发现 (Section 3.2)**: 论文实验(Figure 4, 5, 6)揭示了一个微妙的事实——$K$ 必须大于 $D$ 一个因子 $C > 1$ 才能达到最优精度。这表面上看把复杂度变成 $O(ND \cdot CD) = O(CD^2 N)$,似乎不线性了。

但作者发现:**K 个 random slices 在 M 个 GPU 间独立采样,每个 GPU 取 $K/M$ 个**。Figure 6 显示 8 个 GPU 各 128 slices 的效果接近单 GPU 1024 slices。因此 $K$ 可以在单 GPU 上保持常数(比如 1024),通过分布式 GPU 拉总 slices,保留 $O(NDK)$ 复杂度。这是一个非常实用的工程观察。

Figure 3 显示在单 H100 上,batch size 50K,projection dim 10K,slices 2.5K,views 8 的设置下,VISReg 比 SI-GReg 快 13.7% 内存——因为 Epps-Pulley test 需要 17-knot sampling,开销大。

## 5. 实验结果深度解读

### 5.1 ImageNet-1K Linear Probe (Table 5)

- VISReg ViT-B/16, 400 epochs: **75.7%** (w/o heuristics group 内最佳)
- VISReg ViT-L/14, 100 epochs: 75.6%,400 epochs: 77.0%
- DINO ViT-B/16, 400 epochs: 78.2%(heuristics group)
- iBOT ViT-L/16, 250 epochs: 81.0%

in-domain 上 VISReg 落后于 heuristics-based 方法,但 DTD (OOD) 上 VISReg-B/16 (75.7%) 甚至超过 DINO-L 和 I-JEPA-H。

### 5.2 OOD 性能 (Table 6) —— VISReg 的核心亮点

6 个 OOD dataset: DTD, Galaxy10, AID, ChestXRay, RetinaMNIST, OrganAMNIST。

- VISReg ViT-L/14 (ImageNet1K): avg **70.63%**
- DINOv2 ViT-L/14 (LVD-142M, 10x 数据): avg 72.93%
- **VISReg-Inet22K ViT-L/14: avg 72.94%** ← 用 1/10 数据追平 DINOv2

这是 paper 最 striking 的结论。DINOv2 用 142M 图像训练,VISReg 只用 ImageNet22K (14.2M),达到几乎一样的 OOD avg。

**Intuition**: Sliced-Wasserstein enforces 完整的 marginal 分布,比 covariance decorrelation 强。这意味着学到的 representation 更"通用",不 overfit ImageNet 的 in-domain 偏置,因此 OOD 表现好。Covariance 只能保证 dims 之间不相关,但单 dim 可以是任意分布(重尾、多峰等),这种 distributional 自由度容易 overfit。

### 5.3 Transfer Learning (Table 7)

ViT-B/16 fine-tuning:
- CIFAR10: VISReg 99.2% vs DINO 99.1% vs Sup 99.0%
- CIFAR100: VISReg 91.8% vs DINO 91.7%
- Flowers: VISReg 99.0% vs DINO 98.8%
- ImageNet1K: VISReg 83.0% vs DINO 82.8%
- Galaxy10: VISReg 87.0% vs DINO 86.6%

VISReg 在 in-domain linear probe 落后 DINO 3%,但 fine-tuning 反超。这暗示 VISReg 学到的特征更"通用 / 可塑",而非 in-domain linear separable 但 brittle 的特征。

### 5.4 Dense Prediction (Table 8)

ADE20K linear segmentation mIoU:
- MoCoV3: 31.69 (best)
- DINO: 29.40
- VISReg: 30.16
- MAE: 23.60

VISReg 不及 MoCoV3,作者承认这是 future work 方向。可能因为 covariance / InfoNCE 类目标天然鼓励 spatial structure,VISReg 的 distributional target 不直接优化 patch-level 结构。

### 5.5 Image Generation Guidance (Table 9)

用 iREPA (Singh et al., 2025) 框架,把 SSL features 作为 SiT-B/2 的 guidance,训练 100K steps:
- DINO: IS 33.47, gFID 41.15, Precision 50.51, Recall 60.70
- VISReg: IS 33.48, gFID 40.36, Precision 51.38, Recall 61.26

VISReg 全面胜出。这与 Yu et al. (2025) "Representation alignment for generation" 的方向一致。

### 5.6 Low-Quality Dataset 鲁棒性 (Table 1, 2)

ImageNet-LT (long-tailed) ViT-S/8, 400 epochs:
- VICReg: 33.08%
- SIGReg: 32.00%
- VISReg (default): 32.11%
- VISReg* (shape loss 加重): **35.14%**
- DINO: 5.13% (collapse!)

Galaxy10 (low-rank, 17K images, 10 classes, 大量黑色像素):
- SWD/SIGReg/VISReg: ~80.5%
- VICReg: 79.93%
- DINO: 73.49%

**关键 insight (Table 12)**: 在 low-quality dataset 上,提高 shape loss 权重 (shape 4:1 → 2.0) 显著提升性能:ImageNet-LT +3.2%, Galaxy10 +1.3%。但 high-quality (ImageNette) 上 default 最好。这验证了 decouple scale/shape 的实际价值——可以根据数据 regime 调节。

DINO 在 ImageNet-LT 直接 collapse 到 5.13%,这是 heuristic-based 方法的脆弱性。EMA + centering + sharpening 在分布偏移时失灵,而 explicit regularization 的 robust gradient 信号更可靠。

## 6. 架构图与机制直觉

把整个 pipeline 画出来:

```
Image x
   │
   ▼
[Augmentation: 4 global + 6 local views]  (DINO-style multi-crop)
   │
   ▼
[ViT Backbone (shared weights, NO EMA, NO teacher-student)]
   │
   ▼
CLS tokens from last 2 layers, concatenated
   │
   ▼
[3-layer MLP projector: 2048→2048→d_p, BN+GELU]
   │
   ▼
Embedding Z ∈ R^(N·V × D)
   │
   ├──────────────────┬──────────────────┐
   ▼                  ▼                  ▼
L_pred              L_scale            L_shape
(mean of global     (1-std)^2          (sort(Z·w_k) - q_N)^2
views vs each view)  per-dim std        K random 1D projections
                    target std=1       target = Gaussian quantiles
                                      stop-grad on std for normalization
   │                  │                  │
   └──────────────────┼──────────────────┘
                      ▼
              L_center = ||μ||^2
                      │
                      ▼
      L = (1-λ)·L_pred + λ·(λ_s·L_scale + λ_h·L_shape + λ_c·L_center)
```

**核心设计哲学**: VICReg 三个 loss 都同时优化 Z 的统计性质,导致 scale 和 shape 耦合。VISReg 通过 stop-gradient + normalization 把 shape loss 限制在"shape only",scale loss 限制在"scale only",这两个正交的子问题各自独立优化,符合 "divide and conquer"。

**为什么 SWD 比 covariance 严格更强**:
- Covariance: 只约束 $E[ZZ^T] = I$,即维度间不相关
- SWD with K→∞ slices: 约束每个 1D 投影分布都是 Gaussian,等价于联合分布是 isotropic Gaussian
- 由 Cramér-Wold: 如果所有 1D 投影都是 Gaussian,联合分布必然是 Gaussian

所以 SWD 是"逐 marginal"地逼近联合 Gaussian,covariance 只保证二阶矩匹配。这就是为什么 VISReg OOD 好——分布形状被严格约束。

## 7. 与 Related Work 的关联

### 7.1 VICReg 家族
- VICReg (Bardes et al., 2022): V+I+C, $O(ND^2)$
- Barlow Twins (Zbontar et al., 2021): 冗余消除,cross-correlation matrix
- W-MSE (Ermolov et al., 2021): 球面投影 + whitening

### 7.2 Sketching 家族
- LeJEPA / SI-GReg (Balestriero & LeCun, 2025): Epps-Pulley test,Cramér-Wold 启发
- KerJEPA (Zimmermann et al., 2025): MMD,但 $O(N^2)$ 复杂度
- LpJEPA / RDMReg (Kuang et al., 2026): sparse + max-entropy

### 7.3 JEPA 家族 (Yann LeCun 路线)
- I-JEPA (Assran et al., 2023): masked image prediction in embedding space
- V-JEPA (video 版本)
- DINOv2 (Oquab et al., 2024): 大规模 + heuristics
- DINOv3 (Siméoni et al., 2025): 最新版本

VISReg 在 LeCun 团队内部推进,从 I-JEPA → LeJEPA → VISReg,逐步去除 heuristics 同时加强 distributional rigor。

### 7.4 Optimal Transport in SSL
- SWD (Bonneel et al., 2015; Deshpande et al., 2018): 在 generative modeling (SWAE, SWGAN) 中已经广泛使用
- VISReg 把 SWD 第一次系统引入 SSL embedding regularization

## 8. 个人 Intuition 与思考

**8.1 SWD 的"分位数匹配"非常优雅**: $\mathrm{sort}(\tilde{\mathbf{Z}} w_k) \to \mathbf{q}_N$ 就是让 batch 的 1D 投影按 quantile 严格对齐 Gaussian。这等价于一个 "soft ranking loss",强迫样本均匀分布在 Gaussian 分位数上。这种 uniform coverage 在 embedding space 中很有信息量——防止 collapse 的同时保证多样性。

**8.2 Stop-gradient 的原则性角色**: 在 BYOL/SimSiam 里 stop-gradient 是"防 collapse heuristic",理论不清晰。在 VISReg 里它有明确含义——分解目标的子问题。这种从 "engineering trick" 到 "principled decomposition" 的进化正是 SSL 走向成熟的方向。

**8.3 K vs D 的关系**: Cramér-Wold 要求 "all directions",实践中用 K 个随机方向近似。Figure 4-5 显示 K 必须超过 D,因为 D 维球面上随机 K 个方向要足够密集才能"覆盖"所有方向。这有点像 numerical integration——维度越高,采样方向需要越多。但论文发现分布式 GPU 可以缓解,这是个实用 hack。

**8.4 ImageNet22K vs LVD-142M**: DINOv2 的核心优势是数据量。VISReg 用 1/10 数据追平 OOD 性能,说明 distributional regularization 比 EMA + centering + sharpening 更"data-efficient"。这对 AI for Science(数据稀缺领域)非常重要。

**8.5 与 DINOv3 的关系**: Siméoni et al., 2025 的 DINOv3 在 reference 中出现,暗示这领域在快速演化。VISReg 的 OOD 优势若能保持 scaling,可能成为未来 foundation model 的新 baseline。

**8.6 Limitations 作者承认**: dense prediction (ADE20K) 上 VISReg 不及 MoCoV3。这可能因为 covariance / InfoNCE 类目标天然鼓励 spatial structure,VISReg 的 distributional target 不直接优化 patch-level 结构。未来的工作可能在 shape loss 中加入 spatial structure 约束。

**8.7 一个可能的拓展**: SWD 对齐的是 isotropic Gaussian。如果 target 是其他分布(如 student-t, mixture),会有什么效果?特别是 low-quality data 上,heavy-tailed target 可能更鲁棒。

## 9. Reference Web Links

**核心论文与代码**:
- VISReg 项目与代码: https://haiyuwu.github.io/visreg
- VICReg 论文: https://arxiv.org/abs/2105.04906
- VICReg 代码: https://github.com/facebookresearch/vicreg
- LeJEPA / SI-GReg (Balestriero & LeCun, 2025): https://arxiv.org/abs/2511.08544
- I-JEPA (Assran et al., 2023): https://arxiv.org/abs/2301.08243
- DINOv2 (Oquab et al., 2024): https://arxiv.org/abs/2304.07193
- DINOv3 (Siméoni et al., 2025): https://arxiv.org/abs/2508.10104
- DINO (Caron et al., 2021): https://arxiv.org/abs/2104.14294

**Sliced-Wasserstein 相关**:
- Sliced Wasserstein Barycenters (Bonneel et al., 2015): https://link.springer.com/article/10.1007/s10851-014-0505-0
- Generative Modeling with SWD (Deshpande et al., 2018): https://arxiv.org/abs/1803.11188
- Computational Optimal Transport (Peyré & Cuturi): https://arxiv.org/abs/1803.00567

**Cramér-Wold 与 Epps-Pulley**:
- Cramér-Wold 原始论文: https://londmathsoc.onlinelibrary.wiley.com/doi/abs/10.1112/jlms/s1-11.4.290
- Epps-Pulley test: https://academic.oup.com/biomet/article-abstract/70/3/723/242936

**Barlow Twins / W-MSE**:
- Barlow Twins: https://arxiv.org/abs/2103.03230
- W-MSE: https://arxiv.org/abs/2007.06346

**其他对照方法**:
- MAE (He et al., 2022): https://arxiv.org/abs/2111.06377
- SimSiam (Chen & He, 2021): https://arxiv.org/abs/2011.10566
- BYOL (Grill et al., 2020): https://arxiv.org/abs/2006.07733
- MoCo v3 (Chen et al., 2021): https://arxiv.org/abs/2104.02057
- iBOT (Zhou et al., 2021): https://arxiv.org/abs/2111.07832
- iREPA (Singh et al., 2025): https://arxiv.org/abs/2512.10794
- Representation alignment for generation (Yu et al., 2025): https://arxiv.org/abs/2410.06985

**Sibling 工作**:
- KerJEPA (Zimmermann et al., 2025): https://arxiv.org/abs/2512.19605
- LpJEPA / RDMReg (Kuang et al., 2026): https://arxiv.org/abs/2602.01456

**Datasets**:
- ImageNet: https://www.image-net.org/
- ImageNet-LT (Liu et al., 2019): https://arxiv.org/abs/1904.05116
- Galaxy10: https://astronn.readthedocs.io/en/latest/galaxy10.html
- ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K/
- DTD: https://www.robots.ox.ac.uk/~vgg/data/dtd/

**ViT 与 timm**:
- ViT (Dosovitskiy et al., 2021): https://arxiv.org/abs/2010.11929
- timm library: https://github.com/huggingface/pytorch-image-models

**LeCun 的 JEPA 路线图**:
- A Path Towards Autonomous Machine Intelligence (LeCun, 2022): https://openreview.net/pdf?id=BZ5a1r-kVsf

## 10. 总结

VISReg 是 SSL explicit regularization 的一次漂亮综合:**VICReg 的解耦哲学 + Sliced-Wasserstein 的 distributional rigor + Stop-gradient 的原则性分解**。它用 $O(NDK)$ 复杂度做到了 covariance 做不到的事——把 embedding 分布的完整形状约束到 isotropic Gaussian。

最值得关注的实验结论: ImageNet22K 训练的 ViT-L/14 在 OOD 上追平 DINOv2 (LVD-142M, 10x 数据),这意味着 distributional regularization 比 heuristics 更"data-efficient"。同时 ImageNet-LT 上 DINO collapse 到 5%,VISReg 鲁棒——这对真实世界低质量数据的应用非常有价值。

未来的发展方向自然包括: (a) 加入 spatial structure 约束改进 dense prediction;(b) 探索 non-Gaussian target distribution;(c) scaling 到更大模型 + 更多数据;(d) 跨模态 (audio, video, multi-modal) 的 VISReg 变体。这条 LeCun 主导的 "heuristic-free SSL" 路线非常值得跟踪。
