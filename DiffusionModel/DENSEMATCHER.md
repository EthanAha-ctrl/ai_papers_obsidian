---
source_pdf: DENSEMATCHER.pdf
paper_sha256: 6a75deb15fb469d14bf0f8726ffe93a40a7c32378d108708776abd670b547dd7
processed_at: '2026-08-03T19:53:47-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用大白话再讲一遍。

## 这篇 paper 在干嘛

让机器人看一段人类操作物体的视频，就能去操作一个完全没见过的物体。比如你给它看人剥香蕉，它能去剥另一根形状大小不同的香蕉，甚至能去剥茄子。这件事的核心是：在两个 3D mesh 之间，建立每一个 vertex 对应到哪个 vertex 的 dense mapping。有了这个 mapping，人类 demo 里的 contact point（手抓哪里）就可以 transfer 到新物体上。

## 为什么 hard

3D correspondence 这事学界搞了很多年，但都有问题：

**第一，老方法只看 geometry**。经典做法是用 HKS、WKS 这种 intrinsic shape descriptor，本质就是 "热扩散"、"量子粒子振动" 这种 purely geometric 的 signal。这在人、动物这种有 unique local geometry 的 shape 上还行，但在 cup、banana、chair 这种 daily object 上歇菜 —— 因为它们的局部几何太相似，光看几何区分不出 "这是 banana 的 stem 还是 tip"。

**第二，2D foundation model semantic 强但缺 3D 一致性**。DINOv2、Stable Diffusion 这种 2D 大模型 feature 很强，能识别出语义。但要从 2D feature 拿到 3D vertex feature，只能把多个视角的 2D feature 平均到 vertex 上。问题是不同视角下 vertex 的 pixel coordinate 不稳定，平均出来的 feature noisy，而且缺乏 "这个 vertex 在 3D 上跟周围 vertex 应该是什么关系" 的概念。Diff3F 就是这么做的，在 cross-category 上直接崩溃（held-out AUC 只有 0.423）。

**第三，dataset 缺**。之前的 3D matching benchmark 要么只有人、动物这种 synthetic shape，要么没 texture。模型见都没见过 textured daily object，自然 generalize 不出去。

## DenseMatcher 的核心 trick

一句话：**把 2D foundation model 的 semantic prior 当 appearance feature，再用一个小的 trainable 3D 网络在 mesh surface 上 "organize" 这些 feature，让它们 spatially consistent，最后用 functional map 做 matching**。

具体几个关键设计：

### (1) 2D feature 提取

渲染 5 个视角的 RGB（不需要 Diff3F 那种 100 个），用 SD-DINO 提 2D feature map（SD-DINO 是 DINOv2 + Stable Diffusion 的融合，既有 DINOv2 的 semantic prior 又有 diffusion feature 的 fine-grained localization）。用 FeatUp 把 16×16 feature 上采样到 512×512 避免模糊。每个 vertex 投影到 2D 取 feature，多视角平均，得到 768 维 feature。这步叫 $f_{\text{multiview}}$。

### (2) 3D refiner（关键创新）

光有 $f_{\text{multiview}}$ 还不行，noisy 且缺 spatial consistency。所以接一个 DiffusionNet（4 blocks, 5M 参数，唯一 trainable 的部分）。输入是 $f_{\text{multiview}}$ 拼上 HKS 几何 descriptor 拼上 XYZ 的 sinusoidal positional encoding。DiffusionNet 的核心是 diffusion-style propagation layer，相当于让每个 vertex feature 跟它的 mesh 邻域互相 influence —— 在 surface 上做一次 "heat diffusion" 一样的事。效果就是 feature 在 3D surface 上 spatially smooth 了，2D backbone 给的 semantic prior 被 "reorganize" 成 3D-aware feature。

输出 512 维，L2 normalize，记为 $f(v_i)$。

### (3) Semantic distance loss（最巧妙的设计）

定义两个 vertex 之间的 semantic distance：把它们的 semantic group 之间做 bipartite matching，算 geodesic 距离的平均。比如 banana 上 "stem" group 与另一个 banana 上 "stem" group，geodesic 距离小；"stem" 与 "tip" 距离大。

Loss 强制 feature L2 距离与 semantic distance 线性正相关，用 negative cosine similarity 形式：

$$
L_{\text{semantic}} = -\frac{\sum_{i,j} \|f(v_i) - f(v_j)\|_2 \cdot D_{\text{semantic}}(v_i, v_j)}{\sqrt{\sum \|f(v_i) - f(v_j)\|_2^2} \sqrt{\sum D_{\text{semantic}}(v_i, v_j)^2}}
$$

这个 loss 的妙处：让 feature metric 直接 encode semantic 距离。Paper Appendix 里证明，这个 loss minimize 完之后，functional map 的 feature consistency 项 $\|CF - G\|_2^2$ 等价于最小化所有 matched pair 的总 semantic distance。换句话说，feature 空间被塑造成 semantic 距离的 mirror。

### (4) Preservation loss

光有 semantic distance loss 不够，因为 metric learning 会让 feature space collapse —— 只保留 semantic 距离信息，把 object identity、材质、texture 等都丢了。所以再加一个 reconstruction loss，训一个 linear matrix $W$ 把 $f_{\text{output}}$ 反投回 $f_{\text{multiview}}$。这个相当于 autoencoder bottleneck，强制 $f_{\text{output}}$ 保留 2D backbone 学到的所有 rich information。

Ablation 显示：去掉这个 loss，AUC 从 0.845 掉到 0.568，几乎回到 URSSM baseline 水平。这是个 striking 结果 —— 说明光有 metric supervision 会让 model 忘掉 2D prior。

### (5) Functional map 做 matching

两个 mesh 的 vertex feature 拿到之后，要找 dense correspondence。直接优化 $n_N \times n_M$ 的 point-to-point map 是 combinatorial 难的。Functional map 的 trick：把 map 表示成 $\Pi \approx \Phi_N C \Phi_M^+$，其中 $\Phi$ 是 Laplace-Beltrami eigenfunctions（mesh 上的 "sine waves"），$C$ 是 $k \times k$ 的小矩阵（$k=10$，所以只有 100 个参数）。

直觉：1D 上 sine wave 是 Laplacian 的 eigenfunction；manifold 上 Laplace-Beltrami operator 的 eigenfunction 就是 "surface 上的 sine wave"。低 eigenvalue 对应 smooth global 模式，高 eigenvalue 对应 fine local 模式。把 function 投到这些 eigenfunction 上，相当于 Fourier transform 的 manifold 版本。

优化 $C$ 时同时加三个约束：
- feature consistency: $\|CF - G\|_2^2$
- isometry: $C$ 与 Laplacian 可换，即 $\Lambda_N C = C \Lambda_M$
- descriptor commutativity: $C$ 与 feature multiplication operator 可换

Paper 还加了两个新的：sparsity 的 entropy penalty，强制 $\Pi$ 接近 one-hot；soft assignment constraint，强制 row sum=1、column sum 比例正确。这两个对 daily object 上的 dense correspondence continuity帮助大。

## 结果

**Benchmark 上**：AUC 从 URSSM 的 0.589 提到 0.845，提升 43.5%。在 held-out category（训练样本 0）上 AUC 0.775，比 Diff3F 的 0.423 高 35 个百分点。这个 gap 说明 2D + 3D 的两阶段架构是 cross-category generalization 的关键。

**真机机器人**：6 个 task 平均 76.7% 成功率，比 Robo-ABC 的 50% 高 26 个百分点。包括剥香蕉、放鞋子、装饰圣诞树、拔胡萝卜、用笔指物体部位等等，都跨 instance + 跨 category，部分还跨 material（plush carrot ↔ real carrot）。

**Color transfer**：dense correspondence 的副产品，把一个 mesh 的 vertex color 直接 copy 到对应 vertex，零样本把 banana 的颜色涂到 eggplant 上，外观迁移。

## 几个关键 takeaway

1. **为什么 2D + 3D 比纯 2D 强这么多**：2D foundation model feature 在 pixel space，多视角 average 没法 enforce 3D 一致性。DiffusionNet 通过 diffusion propagation 让 feature 在 mesh surface 上 "spatially organize"，相当于在 2D semantic prior 之上加一层 3D spatial reasoning。

2. **为什么 functional map 比 Hungarian / nearest neighbor 强**：Hungarian 只 match point-wise feature，不考虑 spatial consistency，结果出现 speckled mismatches。Functional map 在 spectral domain 表示 map，天然带 spatial smoothness prior，输出 smooth mapping。这对 manipulation 至关重要，因为 contact point 转移需要局部连续性。

3. **为什么 semantic distance loss 比 contrastive loss 适合 correspondence**：contrastive 只给 binary 信号（相似 / 不相似），semantic distance 给 continuous 距离谱。这让 feature 空间 encode 不只是 "是不是同一部位" 而是 "差几个语义等级"。

4. **为什么 5 个训练样本就能 generalize**：trainable 的只有 5M 参数的 DiffusionNet，2D backbone frozen。它学的不是 "记住这个 category 长什么样"，而是 "如何 transform 2D feature 让它 3D-aware"。这是一种 generic transformation，少量样本就能 fit。结合 2D backbone 的强大 prior，whole pipeline 表现出 strong few-shot 能力。

5. **为什么 preservation loss 重要**：metric learning 会让 feature collapse 到只 encode distance，丢掉 identity / texture。Preservation loss 强制保留 2D backbone 的所有信息，类似 SimSiam / BYOL 的 invariance-vs-variance trade-off。

## 局限

- Remesh 到 500-2500 vertex 损失 fine-grained geometry，对 sub-millimeter manipulation 不够。
- Inference 8-12 秒对 fast control 慢，closed-loop visual servoing 还做不到。
- Mirror-symmetric mesh（intrinsic ambiguity）会 collapse，XYZ positional encoding 只是 mitigation。
- Topology 差异（4 腿椅 vs 3 腿椅）靠人工 semantic group 标注，没有自动 mechanism。
- 2D backbone frozen，没法端到端 finetune 让 SD-DINO 对 3D-aware objective 友好。
- Real robot 实验是 template-based manipulation（waypoint hand-specified），不是 learning-based control policy。

整体看，这是把 graphics 领域的 functional map 工具第一次 carefully 接入 robotics manipulation pipeline，用 2D foundation + 3D refiner 的两阶段架构获取 semantic-aware 3D feature，并 release 第一个 textured 3D matching dataset。下游机器人实验展示了 cross-category / cross-material / long-horizon / multi-keypoint 的综合能力，是 correspondence-based manipulation paradigm 的强 baseline。

---

# DenseMatcher: 一个 bridge 2D 与 3D semantic correspondence 的 framework

你好 Karpathy！这篇 paper 是 Tsinghua IIIS / Shanghai Qi Zhi Institute 等合作的工作（第一作者 Junzhe Zhu 与 Yuanchen Ju），项目页面：https://tea-lab.github.io/DenseMatcher/ 。核心 motivation 是：当前 3D dense correspondence 领域存在严重的 texture/color 缺失以及 category 单一问题，主流方法要么只用 geometry（FAUST、SMAL 上训练的 deep functional map），要么 naive average 多视角 2D feature（如 Diff3F https://arxiv.org/abs/2311.17024），二者各有短板。DenseMatcher 把 SD-DINO 这种 2D foundation model 的 semantic prior 与 DiffusionNet 的 3D geometric reasoning 拼在一起，配合一个改进的 functional map solver，在 cross-instance、cross-category 甚至是 zero-shot（held-out categories 训练样本为 0）的设置下都能 work。

---

## 1. 为什么 3D dense semantic correspondence 是值得做的 task

Paper 第 1 节把 correspondence 沿 "density × dimensionality" 二维 grid 分成四类：3D dense / 3D sparse / 2D dense / 2D sparse（Figure 2）。

- 2D dense correspondence 的问题：viewpoint 变化、occlusion、perspective distortion导致semantic ambiguity。比如一只杯子从侧视到正视，把手相对杯口的pixel位置完全不同。
- 3D sparse correspondence（keypoint matching）的问题：无法保证 surface 上的连续 mapping。机器人 manipulation 需要 "multi-point contact"，需要每个 contact 之间的相对位置也被 transferred，sparse keypoints 不够用。
- 3D dense correspondence 在 manifold 上提供了一个 smooth、bijective 的 mapping，是 manipulation、affordance transfer、color transfer 的天然 substrate。

而 semantic vs. shape correspondence 的差别也很关键：shape correspondence 关注 "geometrically similar" 的部位（比如大象腿 ↔ 犀牛腿），semantic correspondence 关注 "functionally / structurally similar"（比如大象鼻 ↔ 犀牛角，如果按 function 分；或大象鼻 ↔ 犀牛鼻，如果按 location 分）。语义对应更 subjective，但对 manipulation 更有用 —— 你抓 banana 的 stem 而不是 middle 是有功能意义的。

---

## 2. Dataset: DenseCorr3D

这是 paper 第一个 main contribution：之前 3D matching 的 benchmarks（FAUST https://arxiv.org/abs/1505.06748 , TOSCA, SHREC https://api.semanticscholar.org/CorpusID:13810094, SMAL https://arxiv.org/abs/1705.10343）几乎都是 untextured synthetic shapes，category 单一。DenseCorr3D 从 Objaverse-XL https://arxiv.org/abs/2307.05663 和 OmniObject3D https://arxiv.org/abs/2306.07753 中筛 589 个 instances / 23 个 categories，全部带 texture。

### 2.1 Semantic groups 的形式化定义

对每个 mesh vertex $v_i$，定义 group index $n(v_i)$，semantic group $\mathbb{G}_{v_i} := \{v_j \mid n(v_j) = n(v_i)\}$。同 category 的 mesh 共享相同的 semantic groups 标注 schema。比如 apple 上 "stem top" / "stem side" / "top ring" / "middle ring" / "bottom ring" 是几个 groups；banana 上 "stem" / "tip" / "left peel" / "right peel" / "body" 等。Figure 5 给了一个 hand 的两种 partitioning 方案，体现 "correspondence 的主观性" —— 圆对称的 strip 可以是 single group，但 cat ears 这种镜像对称但可区分的部件要分成不同 groups。

### 2.2 Annotation pipeline

- Fruits / vegetables：用 StrayRobots 的 3D annotation tool 标 sparse landmarks，再用基于 graph 的 shortest path interpolation 拿到 dense vertex groups。一个 mesh 大约 10 秒。
- 复杂日常物品（chairs、tools 等）：用 Blender Vertex Brush 直接 paint，每个 mesh 5 分钟左右。
- 后处理：normalize scale，长边乘 0.3，center at origin；保留 largest connected component；isotropic explicit remeshing，目标 vertex 数 500–2500 随机。

### 2.3 Category split（Table 4）

主类别有 train/val/test 三分；样本少的 held-out 类别（celery、cucumber、eggplant）train=0、val=0，全部用来测 zero-shot 泛化。Animals / Tools / Vehicles / Backpacks / Toiletries / Chairs 这些非食品类都只给 5–6 个 train instances，做 few-shot。

---

## 3. Functional Map 数学回顾（build intuition 的关键部分）

这是 paper 的方法骨架。强烈推荐配合 Ovsjanikov et al. 2012 的原论文 https://www.lix.polytechnique.fr/~maks/papers/fmaps_siggraph_2012.pdf 和 Nogneng & Ovsjanikov 2017 https://inria.hal.science/hal-01401897/document 一起读。

### 3.1 Laplace-Beltrami eigenfunctions 的 intuition

在 1D 上，Laplacian $\Delta = d^2/dx^2$ 的 eigenfunctions 是 $\sin(k\pi x / L)$ —— 一系列 frequency 递增的 sine waves。在 manifold 上，Laplace-Beltrami operator $\Delta$ 的 eigenfunctions $\Phi_j$（满足 $W \Phi_j = \lambda_j A \Phi_j$）就是 manifold 上的 "sine waves"。低 eigenvalue 对应 low-frequency、smooth、global 的振动模式；高 eigenvalue 对应 high-frequency、local 的细节。

- $W \in \mathbb{R}^{n \times n}$: cotangent weight matrix（Meyer et al. 2003 离散化）
- $A \in \mathbb{R}^{n \times n}$: diagonal vertex area matrix
- $\Phi_M \in \mathbb{R}^{n_M \times k}$: source mesh 的前 k 个 eigenfunctions（columns 是 eigenfunctions）
- $\Phi_N \in \mathbb{R}^{n_N \times k}$: target mesh 的前 k 个 eigenfunctions
- $\Lambda_M, \Lambda_N \in \mathbb{R}^{k \times k}$: diagonal eigenvalue matrices
- $\Phi^+ := \Phi^T A$ (pseudo-inverse，利用 $A$-weighted inner product)

任意 vertex 上的 scalar function $x \in \mathbb{R}^n$ 可以投影到 spectral domain: $X = \Phi^+ x \in \mathbb{R}^k$，反之 $x \approx \Phi X$。k 通常取 10–30，所以一个 $n=2000$ vertex 的 mesh 上的 function 被 10 个数 compactly 编码。

### 3.2 Vertex-to-vertex map 的 low-rank 表示

真实想要的 $\Pi \in \mathbb{R}^{n_N \times n_M}$ 是一个 sparse binary matrix，对每 column 只有一个 1：$\Pi_{ij} = 1 \iff i = \text{match}(j)$。直接优化 $\Pi$ 是 combinatorial 难的。

Functional map 的核心 trick 是：$\Pi \approx \Phi_N C \Phi_M^+$，其中 $C \in \mathbb{R}^{k \times k}$ 只有 $k^2$ 个参数（典型 $k=10$ → 100 个参数）。这是一个**spectral bottleneck**：先 $\Phi_M^+$ 把 source function 压到 spectral domain（10 维），用 $C$ 做线性变换，再 $\Phi_N$ lift 回 target mesh。

### 3.3 特征一致性约束

给定 vertex features $f \in \mathbb{R}^{n_M \times d_\text{feat}}$（source）和 $g \in \mathbb{R}^{n_N \times d_\text{feat}}$（target），对应顶点应有相同 feature。即 $g \approx \Pi f = \Phi_N C \Phi_M^+ f$。两边左乘 $\Phi_N^+$ 得到 spectral-domain 约束：

$$
\underbrace{\Phi_N^+ g}_{G \in \mathbb{R}^{k \times d_\text{feat}}} \approx C \underbrace{\Phi_M^+ f}_{F \in \mathbb{R}^{k \times d_\text{feat}}}
$$

这就是 $\|CF - G\|_2^2$ 这一项。

### 3.4 Isometry 约束

如果 $\Pi$ 是 isometric（保持 Riemannian distance），则 $\Pi$ 与 Laplacian 可换：$\Pi \Delta = \Delta \Pi$。在 spectral domain 等价于 $\Lambda_N C = C \Lambda_M$，即 $\|\Lambda_N C - C \Lambda_M\|_2^2$ 最小。Paper Appendix A.4.3 给了完整推导，关键步骤：

$$
\Pi \Delta(x) = \Delta(\Pi x) \implies \Phi_N C \Phi_M^+ \Phi_M \Lambda_M X = \Phi_N \Lambda_N \Phi_N^+ \Phi_N C X \implies C \Lambda_M X = \Lambda_N C X
$$

对所有 $x$ 成立 → $C \Lambda_M = \Lambda_N C$。

### 3.5 Descriptor commutativity 约束

Nogneng & Ovsjanikov 2017 提出：要让 $C$ 真正对应 point-to-point map，必须与每个 feature channel 的 point-wise multiplication operator 可换。设 $X^{(p)} = \Phi_M^+ \text{Diag}(f^{(p)}) \Phi_M$，$Y^{(p)} = \Phi_N^+ \text{Diag}(g^{(p)}) \Phi_N$，则约束 $\|CX^{(p)} - Y^{(p)}C\|_2^2$。直觉：feature multiplication 在 spectral domain 是什么样子？$\text{Diag}(f^{(p)}) \Phi_M$ 表示 "把 $p$-th feature 乘到每个 eigenfunction 上"，然后投影回 spectral domain，得到一个 $k \times k$ 的 multiplication operator 表达。如果 $C$ 是真的 point-to-point map，这种 multiplication 应当与 map 可换。

### 3.6 完整 objective（Eq. 1）

$$
C_\text{opt} = \arg\min_C \underbrace{\|CF - G\|_2^2}_\text{feature consistency} + \underbrace{\alpha \|\Lambda_N C - C \Lambda_M\|_2^2}_\text{isometry} + \underbrace{\beta \sum_{p=1}^{d_\text{feat}} \|CX^{(p)} - Y^{(p)}C\|_2^2}_\text{descriptor commutativity}
$$

$\alpha = 10^{-2}, \beta = 10^{-4}$。$k=10$ eigenvectors，$C$ zero-init，L-BFGS 求解。

---

## 4. DenseMatcher 架构

Figure 6 给的 pipeline 是 frozen 2D backbone → trainable 3D neck → functional map solver。

### 4.1 Multi-view foundation feature 提取

- 渲染多视角 RGB images。Diff3F 渲染 100 个 views，每个 mesh 要跑 100 次昂贵的 2D backbone forward，~5 分钟。DenseMatcher 因为有 3D refiner，只需 5 个 views（3 lateral + 1 top + 1 bottom），训练和推理都用同样的设置。
- 用 **SD-DINO**（Zhang et al. 2023 https://arxiv.org/abs/2306.01761）提取 2D feature map。SD-DINO 融合 DINOv2（https://arxiv.org/abs/2304.07193，self-supervised ViT，semantic prior 强）和 Stable Diffusion（https://arxiv.org/abs/2112.10752，diffusion U-Net feature，对 fine-grained localization 友好）。
- **FeatUp**（https://arxiv.org/abs/2403.10552）上采样 feature map 从 $16 \times 16$ 到 $512 \times 512$，避免 bilinear interpolation 把 high-frequency semantic boundary 模糊掉。
- 对每个 vertex $v_i$，project 到 2D image coordinate，bilinear interpolation 取 feature。多个 visible views 的 feature 平均，不可见则 zero vector。得到 $f_\text{multiview}(v_i) \in \mathbb{R}^{768}$。

这一步与 Diff3F 的关键区别：Diff3F 没有任何后续 3D 处理，直接拿这个 average feature 做 functional map。结果就是 noise 很重，因为不同视角的 pixel coordinate 不稳定，feature aggregation 缺乏 global 3D consistency。

### 4.2 DiffusionNet refiner

输入特征 concat 三部分：
1. $f_\text{multiview} \in \mathbb{R}^{768}$：semantic appearance
2. **HKS** (Heat Kernel Signature, https://www.lix.polytechnique.fr/~maks/papers/HKS_SGP_2009.pdf)：geometry descriptor。直觉：把热扩散方程在 vertex $v_i$ 处的解在不同时间 scale 上采样，得到一个 intrinsic、isometry-invariant 的 descriptor。它对 mesh 的 bending 鲁棒，对 mesh 的 topology 也鲁棒。
3. Sinusoidal positional encoding of XYZ（NeRF https://arxiv.org/abs/2003.08934 风格）：让模型知道 vertex 在 canonical pose 下的位置，但训练时 random rotation 让它不依赖 canonical pose。

输入送 **DiffusionNet**（Sharp et al. 2022, https://arxiv.org/abs/2012.03497）：alternating MLP layers 与 diffusion-style propagation layer。DiffusionNet 的核心思想是把卷积在 surface 上的 feature propagation 实现成 PDE diffusion 的形式 —— 用 heat diffusion operator 在 mesh 上传播 feature，等价于在频域上 attenuate high-frequency modes。这样 discretization-agnostic，可以在不同 mesh resolution 上训练并迁移。

输出 $f_\text{output}(v_i) \in \mathbb{R}^{512}$，L2 unit-normalize 得到 $f(v_i)$。这是 DenseMatcher 唯一 trainable 的部分，4 blocks / 512 channels / ~5M 参数。

### 4.3 Loss function

$$
L = L_\text{semantic} + L_\text{preservation}
$$

#### 4.3.1 Semantic distance loss $L_\text{semantic}$

定义两个 vertices $v_i, v_j$ 的 semantic distance $D_\text{semantic}(v_i, v_j)$：
- 对同一 mesh：bipartite matching 它们的 semantic groups 之间的 pairwise geodesic distance matrix，取 matched pairs 平均。
- 对跨 mesh：先找到 source group 在 target mesh 上对应的 group（基于 group index 一致性），再算。

形式化：给定 group $\mathbb{G}_{v_i}$ ($m$ vertices) 和 $\mathbb{G}_{v_j}$ ($n$ vertices)，定义

$$
D_\text{semantic}(v_i, v_j) = \frac{1}{\min(m, n)} \min_{\pi_1 \in S_m, \pi_2 \in S_n} \sum_{k=1}^{\min(m, n)} D_\text{geodesic}(\mathbb{G}_{v_i}(\pi_1(k)), \mathbb{G}_{v_j}(\pi_2(k)))
$$

其中 $\pi_1, \pi_2$ 是 bipartite matching 的 permutation，$D_\text{geodesic}$ 用 heat method (Crane et al. 2013, https://arxiv.org/abs/1204.6216) 计算。

Loss 的形式是 negative cosine similarity between $\|f(v_i) - f(v_j)\|_2$ 和 $D_\text{semantic}(v_i, v_j)$，希望它们 linearly proportional：

$$
L_\text{semantic} = -\cos(\theta) = -\frac{\sum_{i,j} \|f(v_i) - f(v_j)\|_2 D_\text{semantic}(v_i, v_j)}{\sqrt{\sum_{i,j} \|f(v_i) - f(v_j)\|_2^2} \sqrt{\sum_{i,j} D_\text{semantic}(v_i, v_j)^2}}
$$

这是 Pearson correlation 的负值。直觉：feature 空间里的距离（L2）与 semantic 空间里的距离（geodesic on semantic group matching）应当线性正相关。

附录 A.4.2 证明：把这个 loss 完全 minimize 后，functional map 的 feature consistency term $\|CF - G\|_2^2$ 等价于最小化所有 matched vertex pair 的 total $D_\text{semantic}$。核心推导：

$$
\frac{1}{s} \sum_j D_\text{semantic}(v_{\text{match}(j)}, v_j) = \|\Pi f - g\|_2 = \|\Phi_N C \Phi_M^+ f - g\|_2 \approx \|CF - G\|_2
$$

中间用 $g \approx \Phi_N \Phi_N^+ g$ 把 $g$ 也投影到 spectral domain。$s$ 是 $\|f_i - g_j\|_2 = s \cdot D_\text{semantic}(v_i, v_j)$ 的线性常数。

这个 loss 的妙处：它让 feature metric 直接 encode semantic distance，而不是单纯 mirror 2D feature 的 cosine similarity。这就是 DenseMatcher 能在 cross-category 上 generalize 的根本原因。

#### 4.3.2 Feature preservation loss $L_\text{preservation}$

$$
L_\text{preservation} = \sum_{i}^{|V|} \|f_\text{multiview}(v_i) - W f_\text{output}(v_i)\|
$$

其中 $W \in \mathbb{R}^{768 \times 512}$ 是可学习的 back-projection matrix。直觉：DiffusionNet 是 nonlinear operator，单纯靠 $L_\text{semantic}$ 训练会让 feature space collapse 到只 encode semantic distance —— 物体类型、材质、纹理等信息会丢失。所以训一个 linear reconstructor 让 $f_\text{output}$ 还能 reconstruct $f_\text{multiview}$，类似 autoencoder bottleneck 的 reconstruction loss，保住 2D foundation model 学到的 rich semantic prior。

### 4.4 Improved functional map solver

Paper 提出两个新的 regularization term：

#### (1) Sparsity via entropy

$\Pi = \Phi_N C \Phi_M^+$ 是 continuous relaxation，理论上对应 point-to-point map 时每 row 应只有一个 1。但 dense objects 上几何相似度高，$\Pi$ 容易变 diffuse。所以 clamp $\Pi$ 到 $[0, 1]$：$\tilde{\Pi}_{ij} = \max(0, \min(1, \Pi_{ij}))$，然后 penalize 它的 entropy：

$$
-\sum_{i=1}^{n_N} \sum_{j=1}^{n_M} \tilde{\Pi}_{ij} \log \tilde{\Pi}_{ij}
$$

注意这是 **负** entropy（加在 cost 上）—— 优化最小化 cost 等价于最大化 entropy？这里需要仔细看一下符号。在 paper 的 wording 是 "penalize its entropy to promote sparsity"，也就是希望 entropy 小，那应该是 $+\sum \tilde{\Pi}_{ij} \log \tilde{\Pi}_{ij}$ 才对（因为 entropy $H = -\sum p \log p$，最小化 $-H$ 即最大化 $H$，不是 sparsity）。这里有歧义，或者作者定义的 "entropy term" 直接就是 $\sum p \log p$ 不带负号。从实际效果推断：cost 应当包含 $\sum \tilde{\Pi}_{ij} \log \tilde{\Pi}_{ij}$（penalize peakiness 的反义，即鼓励 peakedness/sparsity）。我会按 "鼓励 sparsity" 的语义理解：cost 中加的是 $\sum \tilde{\Pi}_{ij} \log \tilde{\Pi}_{ij}$，当 $\Pi$ 是 one-hot 时该项为 0，当 $\Pi$ 是 uniform 时该项最大（最负）→ 不对，uniform 的 $\sum p \log p = -\log n$ 是负数，加到 cost 里反而鼓励 uniform。

这里 paper 的公式确实写得有 sign ambiguity。从 "promote sparsity" 的字面理解，应当是 minimize $-\sum \tilde{\Pi}_{ij} \log \tilde{\Pi}_{ij}$（即 entropy 本身），那加到 cost 里应该是 $+\sum \tilde{\Pi}_{ij} \log \tilde{\Pi}_{ij}$？不对，entropy 是 $-\sum p \log p$，最小化 entropy 等价于在 cost 中加 $+\sum p \log p$。但 $p \log p$ 在 $p \to 0$ 时为 0，在 $p=1/e$ 时为 $-1/e$，在 $p=1$ 时为 0；它对 $p \in (0,1)$ 全是负的。Cost 加上 $\sum p \log p$ 会鼓励所有 $p$ 都集中到 0 或 1（边界），符合 sparsity 鼓励。

最终我倾向：cost 加 $+\sum_{i,j} \tilde{\Pi}_{ij} \log \tilde{\Pi}_{ij}$（或等价地 $-\sum \tilde{\Pi}_{ij} \log \tilde{\Pi}_{ij}$ 作为负 entropy 项），鼓励 sparse assignment。Paper 公式写成 $-\sum \tilde{\Pi}_{ij} \log \tilde{\Pi}_{ij}$ 但文字说 "penalize entropy"，这里 paper 写法与字面解释矛盾，按 "sparsity 目的" 来理解应当实际加的是 $+\sum \tilde{\Pi}_{ij} \log \tilde{\Pi}_{ij}$。实测 weight $10^{-5}$。

#### (2) Soft assignment constraint

强制 $\Pi$ 的 row sum = 1（每个 target vertex 应恰好对应一个 source vertex），column sum = $n_N / n_M$（source vertex 在 target 上的 "soft 出现次数" 与面积比例一致）：

$$
\sum_{i=1}^{n_N} \Big(\sum_{j=1}^{n_M} \Pi_{ij} - 1\Big)^2 + \sum_{j=1}^{n_M} \Big(\sum_{i=1}^{n_N} \Pi_{ij} - \frac{n_N}{n_M}\Big)^2
$$

这两个 term 加上 weight $10^{-5}$ 和 $10^{-3}$ 进 Eq. 1 的 cost。

### 4.5 部分遮挡的处理（Appendix A.5）

- 两边都 partial：functional map 直接 work（Fig. 11 的真实机器人实验就证明）。
- Source partial / target full：用 partial functional correspondence (Rodola et al. 2017, https://arxiv.org/abs/1509.05739) 的 formulation，额外学习一个 mask $\eta \in \mathbb{R}^{n_N}$ 表示每个 target vertex 是否被 matched；加 Mumford-Shah functional、area preservation、$\eta$ 的 entropy penalization。

---

## 5. 实验

### 5.1 3D Dense Matching benchmark (Table 1)

| Method | All AUC↑ | All Err↓ | Held-out AUC↑ | Held-out Err↓ |
|---|---|---|---|---|
| ConsistFMap (FAUST) | 0.537 | 7.86 | 0.497 | 8.39 |
| ConsistFMap (DenseCorr3D) | 0.541 | 7.23 | 0.502 | 7.92 |
| URSSM (FAUST) | 0.568 | 6.37 | 0.532 | 7.07 |
| URSSM (DenseCorr3D) | 0.589 | 6.08 | 0.539 | 6.87 |
| Diff3F | 0.522 | 5.96 | 0.423 | 8.53 |
| **DenseMatcher (Ours)** | **0.845** | **1.74** | **0.775** | **2.82** |
| w/o DiffusionNet | 0.672 | 4.74 | 0.662 | 5.53 |
| w/o Preservation Loss | 0.568 | 5.11 | 0.509 | 6.92 |
| w/o FeatUp | 0.741 | 3.48 | 0.638 | 5.78 |
| w/o Constraint for FMap | 0.824 | 1.98 | 0.735 | 3.32 |

关键观察：
- Diff3F 在 Held-out set 上暴跌（AUC 0.423），说明纯 2D feature average 没有跨 category 泛化能力。DenseMatcher held-out 0.775 比 Diff3F held-out 高 35 个百分点。
- w/o DiffusionNet 比完整模型掉 17 个百分点 AUC —— 3D refiner 是 cross-category generalization 的核心。
- w/o Preservation Loss 几乎回到 URSSM 水平（AUC 0.568）—— 这是个 striking 结果，说明光有 semantic distance loss 不够，必须保住 2D foundation 的 rich information，否则 feature space collapse 后连 in-distribution 类别都做不好。
- w/o FeatUp 掉 10 个百分点 —— 高分辨率 feature map 对 vertex-level correspondence 重要。
- w/o FMap constraint 只掉 2 个百分点 —— 但 qualitative Fig. 10(b) 显示 constraint 对 continuity 影响很大。

### 5.2 Real-world robotic manipulation (Tables 2 & 3)

6 个 task：peel a banana / flower arrangement / place shoes / decorate Christmas tree / pull out the carrot / point object parts with pen。

对比 Robo-ABC（https://arxiv.org/abs/2401.07487）的两个 variant：
- Robo-ABC† (原始 memory)：30% 整体成功率
- Robo-ABC* (只用本 paper 提供的 demo 当 memory)：50%
- **DenseMatcher：76.7%**

DenseMatcher 在 Christmas tree（5/5）、place shoes（4/5）、peel banana（4/5）、pull carrot（4/5）上表现突出，所有 task 都跨 instance + 跨 category + 多种额外 difficulty（multi-keypoint / long-horizon / cross-material / cluttered viewpoint / multiple objects）。

Workflow（Fig. 7）：
1. RGB-D 视频记录 human demo
2. Hand-object detector（Shan et al. 2020 https://arxiv.org/abs/2005.05227）检测 hand-object contact frame
3. 采样 hand bbox 与 object bbox 的 overlap 作为 contact point，trace back 到第一帧避 occlusion
4. 得到 template mesh + template keypoints
5. DenseMatcher 计算 template 与 target mesh 的 dense correspondence
6. Keypoints 通过 mapping transfer 到 target
7. AnyGrasp（https://arxiv.org/abs/2301.07756）从 keypoint 推断 grasp pose
8. MoveIt! (https://arxiv.org/abs/1404.3785) 规划 trajectory

### 5.3 Color transfer (Fig. 9)

利用 dense correspondence 把 source mesh 的 vertex color 直接 copy 到 target mesh 对应 vertex。例如 banana ↔ eggplant、tomato ↔ kabocha squash、wine bottle 之间、手套之间。这是把 dense correspondence 的 "bijective map" 性质利用起来 —— 不需要额外 supervision，就能 zero-shot transfer appearance。Ofri-Amar et al. 2023 在 2D 上做过类似 https://arxiv.org/abs/2305.19027 ，3D 上据 paper 说没见过先前工作。

### 5.4 Ablation: feature vs. matching algorithm (Fig. 10)

- Fig. 10(a)：同样用 functional map，feature 换成 HKS 或 WKS（pure geometric descriptors）。结果斑驳错位。说明 pure geometry feature 在 daily objects（缺少 unique local geometric signature）上不行。
- Fig. 10(b)：同样用 DenseMatcher feature，matching 算法换成 Hungarian 或 nearest neighbor。结果出现 "speckled mismatches" —— 因为这两个方法只 match point-wise feature，不考虑 spatial consistency。Functional map 通过 spectral representation 同时保持 point-wise feature consistency 和 neighborhood structure consistency，得到 smooth mapping。

### 5.5 Runtime (Table 5, Appendix A.3.3)

| Method | 500-vertex | 2000-vertex |
|---|---|---|
| Functional Map (DenseMatcher) | 0.8s | 2.2s |
| SpiderMatch (CVPR 2024, https://arxiv.org/abs/2404.19114) | ~10s | >200s |
| Hungarian Matching | 0.01-0.4s | 0.5-2.5s |

端到端一个 mesh pair 8.4-12.4s on single A100，足以 support 实时机器人 planning。

---

## 6. 几点 build-intuition 的总结

1. **为什么是 spectral bottleneck（functional map）而不是直接 learn attention？** Functional map 把 $O(n^2)$ 的 point-to-point map 压到 $O(k^2)$，并且通过 Laplacian eigenfunction 自然引入 spatial smoothness prior。Hungarian / NN matching 不带这个 prior，会出现 speckled noise。同时 spectral domain 的 optimization 是 differentiable 的，可以 back-prop 到 vertex feature。

2. **为什么 2D + 3D 比纯 2D 强这么多？** 2D foundation model（DINOv2、SD）的 feature 在 pixel space，多视角 average 没法 enforce geometric consistency。DiffusionNet 通过 diffusion-style propagation 让 feature 在 mesh surface 上 "spatially organize"，相当于在 2D semantic prior 之上加一层 3D spatial reasoning。

3. **为什么 semantic distance loss 比 contrastive loss 更适合 correspondence？** 普通 contrastive loss 只有 binary "相似 / 不相似" 信号；semantic distance loss 把 feature L2 distance 与 semantic group 间的 geodesic distance linearly couple，给出 "相似程度的连续谱"。这使得 functional map 的 feature consistency term 自然 encode semantic 距离（Appendix A.4.2 证明）。

4. **为什么 preservation loss 重要？** Semantic distance loss 是 "metric learning" 风格，会 collapse 掉与 metric 无关的信息（材质、object identity、texture detail）。Preservation loss 通过 linear reconstructor 强制 $f_\text{output}$ 保留 $f_\text{multiview}$ 的全部信息。这是一个 "metric + identity" 双目标，类似 SimSiam / BYOL 的 invariance-vs-variance trade-off，但在 vertex feature 上做。

5. **为什么 few-shot（5 个训练样本）就能 generalize？** 因为真正学的是 DiffusionNet refiner，2D backbone frozen。5M 参数的 DiffusionNet 在 5 个样本上只学 "如何 transform 2D feature 以匹配 3D structure" 这种 generic transformation，而非 "记住这个 category 的 appearance"。结合预训练 2D backbone 的强大 prior，whole pipeline 表现出 strong few-shot ability。

---

## 7. 与我熟悉的相关工作串联

- **Florence et al. 2018 DenseObjectNets** (https://arxiv.org/abs/1806.08756) 是 dense correspondence 用于 manipulation 的开山作，但用 autoencoder 在 single object 上训练，无跨 category 能力。
- **VRB (Bahl et al. 2023)** https://arxiv.org/abs/2304.04499 从 human video 提取 contact point，纯 2D。
- **Robo-ABC** https://arxiv.org/abs/2401.07487 把 contact affordance 用 2D semantic correspondence transfer，跨 category 但有 retrieval bottleneck。
- **RAM (Kuang et al. 2024)** https://arxiv.org/abs/2407.04689 3D 版的 affordance retrieval transfer。
- **Diff3F** https://arxiv.org/abs/2311.17024 是 DenseMatcher 最直接的 baseline，做 untextured shape 的 2D feature projection。
- **ConsistFMap (Cao & Bernard 2022)** https://arxiv.org/abs/2207.04300 和 **URSSM (Cao et al. 2023)** https://arxiv.org/abs/2304.14419 是 SOTA deep functional map 方法，但都用 pure geometric feature。
- **DiffusionNet** https://arxiv.org/abs/2012.03497 是 DenseMatcher 的 trainable neck，key property 是 discretization-agnostic。
- **Diff3F / SD-DINO / Diffusion Hyperfeatures (Luo et al. 2023)** https://arxiv.org/abs/2308.06743 都属于 "用 diffusion feature 做 correspondence" 这个 lineage。
- **SpiderMatch (Roetzer & Bernard 2024)** https://arxiv.org/abs/2404.19114 是 functional map 的全局最优替代方案，但慢两个量级。
- **Objaverse-XL** https://arxiv.org/abs/2307.05663 和 **OmniObject3D** https://arxiv.org/abs/2306.07753 是 dataset 来源。

---

## 8. 可能的局限和未来方向

paper 自己没明说但可以推测：
1. **Remesh 到 500-2500 vertex** 损失了 fine-grained geometry。对于需要 sub-millimeter 精度的 manipulation（比如插钥匙），需要 dense mesh 上的 correspondence。
2. **Laplace-Beltrami eigenfunctions 是 intrinsic 的**，对 mirror-symmetric 的 mesh（无特征左右对称）会 collapse。paper 用 HKS + XYZ positional encoding 来 mitigate，但 fundamental intrinsic ambiguity 还在。Zhang et al. 2024 CVPR "Telling Left from Right" https://arxiv.org/abs/2310.13028 处理类似问题。
3. **Topological differences**：chair 有 4 条腿 vs. 3 条腿 vs. 1 条 central leg，semantic group 划分本身就棘手。Paper 把这归为 "subjective correspondence" 交给人工 annotation，但没有自动 mechanism 处理 cross-topology。
4. **Inference 8-12s** 对 fast control 还是太慢。要做 closed-loop visual servoing 还需要 lightweight 版本。
5. **2D backbone 完全 frozen** 是优点也是缺点 —— 没法端到端 finetune SD-DINO 让它对 3D-aware objective 友好。
6. **Color transfer 实验** 没有 quantitative metric，只有 qualitative figure。
7. **Real-world manipulation 用 MoveIt! + AnyGrasp** 是把 correspondence 输出 plug 进现有 stack，没有 learning-based control policy，long-horizon 任务靠 waypoint hand-specification。

整体上，这篇 paper 的贡献在于：把 functional map 这个 graphics 领域成熟工具第一次 carefully 接入 robotics manipulation pipeline，用 2D foundation model + 3D refiner 的两阶段架构获取 semantic-aware 3D feature，并 release 了第一个 textured 3D matching dataset。下游机器人实验虽然是 template-based manipulation（不是 policy learning），但展示了 cross-category / cross-material / long-horizon / multi-keypoint 的综合能力，是 correspondence-based manipulation 这个 paradigm 的强 baseline。

Reference links：
- Project: https://tea-lab.github.io/DenseMatcher/
- Functional Maps original: https://www.lix.polytechnique.fr/~maks/papers/fmaps_siggraph_2012.pdf
- Nogneng & Ovsjanikov 2017: https://inria.hal.science/hal-01401897/document
- DiffusionNet: https://arxiv.org/abs/2012.03497
- DINOv2: https://arxiv.org/abs/2304.07193
- Stable Diffusion: https://arxiv.org/abs/2112.10752
- SD-DINO: https://arxiv.org/abs/2306.01761
- FeatUp: https://arxiv.org/abs/2403.10552
- Diff3F: https://arxiv.org/abs/2311.17024
- URSSM: https://arxiv.org/abs/2304.14419
- ConsistFMap: https://arxiv.org/abs/2207.04300
- Heat Kernel Signature: https://www.lix.polytechnique.fr/~maks/papers/HKS_SGP_2009.pdf
- Wave Kernel Signature: https://arxiv.org/abs/1110.4016
- Geodesics in Heat: https://arxiv.org/abs/1204.6216
- Objaverse-XL: https://arxiv.org/abs/2307.05663
- OmniObject3D: https://arxiv.org/abs/2306.07753
- Robo-ABC: https://arxiv.org/abs/2401.07487
- VRB: https://arxiv.org/abs/2304.04499
- AnyGrasp: https://arxiv.org/abs/2301.07756
- MoveIt!: https://arxiv.org/abs/1404.3785
- SpiderMatch: https://arxiv.org/abs/2404.19114
- Partial Functional Correspondence: https://arxiv.org/abs/1509.05739
- Neural Congealing (color transfer 2D analog): https://arxiv.org/abs/2305.19027
- Telling Left from Right (geometry-aware semantic corr): https://arxiv.org/abs/2310.13028
- DenseObjectNets: https://arxiv.org/abs/1806.08756
- Diffusion Hyperfeatures: https://arxiv.org/abs/2308.06743
- Hand-object contact (Shan et al. 2020): https://arxiv.org/abs/2005.05227
