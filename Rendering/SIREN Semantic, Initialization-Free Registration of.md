---
source_pdf: SIREN Semantic, Initialization-Free Registration of.pdf
paper_sha256: f46b9404d405759874f28ea26e42fc61675ef586639e8b5b46f632854b2e8b10
processed_at: '2026-08-12T07:02:52-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SIREN 用人话讲

## 一句话总结

多个机器人各自拍了一圈，各自 train 了一个 GSplat map，SIREN 帮你把这几个 map 拼成一个完整的，不需要 camera pose、不需要原始图片、不需要知道机器人在哪、不需要任何初始化。

---

## 为什么这事难

想象两个人从不同角度拍同一个房间，各自建了个 3D 模型。你要把两个模型对齐。

经典做法是 ICP：找一个点，在另一个模型里找最近的点，当 correspondence，然后算 transform。但问题在于——你根本不知道哪个点对应哪个点。两个模型还没对齐的时候，"最近的点"基本是瞎配。

更惨的是真实机器人场景：quadruped 趴在地上拍，drone 在天上飞，manipulator 在桌上转。viewpoint 完全不一样，overlap 可能只有 10%，local feature descriptor（SIFT、FPFH）全废。Prior work（PhotoReg、GaussReg）在这些场景下 rotation error 动辄 40-50 度，基本等于没配。

---

## SIREN 的核心 idea

**用语义当 anchor**。

房间里有个 chair。Quadruped 拍到的 chair 和 drone 拍到的 chair，长得不一样，但 CLIP 知道这俩都是 "chair"，semantic embedding 的 cosine similarity 会很高。

所以 SIREN 不去匹配所有点，而是：
1. 先问 "chair 在哪" "table 在哪" "couch 在哪"
2. 把每个 map 里这些 semantic-rich 的 Gaussian 挑出来
3. 在这些挑出来的 Gaussian 之间找 correspondence
4. 用 closed-form 算 transform

**关键 insight**：registration 问题的真正瓶颈不是优化本身（给你准确 correspondence，Umeyama 1991 一行公式就解了），而是怎么找到准确的 correspondence。Semantics 提供了一个跨 viewpoint、跨 scale、跨 illumination 都 robust 的 correspondence prior。

---

## 三步走

### Step 1: 给每个 Gaussian 打上 semantic 标签

训 GSplat 的时候，同时训一个 semantic field $\psi$，输入 3D 坐标，输出 semantic embedding。用 multi-resolution hash grid 参数化，跟 GSplat 一起 train。

训完之后，每个 Gaussian 不光有 position、color、covariance，还有一个 semantic embedding。你可以用 text query "chair" 去 localize 所有跟 chair 相关的 Gaussian。

这步比 prior work（LangSplat、Feature 3DGS）简单——它们额外训一个 autoencoder 把 CLIP feature 降维。SIREN 直接用原始 CLIP feature，因为 GSplat 自带高质量 depth，可以直接 back-project 像素到 3D，不需要 NeRF 那套 proposal network 的 memory 开销。

### Step 2: Coarse alignment，closed-form 求解

在两个 map 的 semantic-rich Gaussian 之间找 correspondence，然后解一个 optimization：

$$\min \sum w_{ij} \left( \|s R p_i + t - q_j\|^2 + \|s R H_{p_i} \Lambda_{p_i} - H_{q_j} \Lambda_{q_j}\|_F^2 \right)$$

第一项是 position 对齐（point-to-point），第二项是**形状和朝向对齐**——这是 SIREN 相对经典 Umeyama 的扩展。每个 Gaussian 不光有 position，还有 covariance（描述椭球的形状和朝向），这等于每个 correspondence 提供了 9 个 scalar constraint 而非 3 个。

这个 optimization 有 closed-form 解：先求 translation（对齐 weighted centroid），再把问题转成 trace minimization，用 SVD 解 rotation（经典 Orthogonal Procrustes），最后求 scale。整个过程一行 linear algebra 搞定，不需要 iterative optimization。

$w_{ij}$ 是 semantic cosine similarity，用来 down-weight false correspondence。加 RANSAC 进一步去 outlier。

### Step 3: Fine alignment，用 photometric refinement

Coarse 步骤只用了几何信息，没用 GSplat 最强的视觉信息。Fine 步骤用 novel-view synthesis 渲染图像，做 image-to-image registration。

具体流程：
1. 用 coarse transform 把 source 的 camera pose 变到 target frame，两边同时渲染
2. 用 CLIP filter 掉相似度低的 image pair
3. SuperPoint 提 local feature，SuperGlue match
4. SfM + Bundle Adjustment 求 camera pose
5. 再解一个 SE(3) registration 把所有 pose 对齐

这步把 rotation error 从 tens of degrees 推到 sub-degree。

---

## 为什么 work 得这么好

几个原因叠加：

**Semantics 是极强的 correspondence prior**。CLIP 在 billion-scale data 上 pretrain，对 "chairness" 这种抽象概念有跨 domain 的 robustness。两个不同机器人拍的同一把椅子，geometric descriptor 可能完全不一样，但 CLIP embedding 很 close。

**Gaussian covariance 提供额外 anchor**。普通 point cloud registration 每个点只给 3 个 scalar，SIREN 每个 Gaussian 给 9 个（3 position + 6 covariance）。correspondence 数量少的时候这个增益巨大。

**Coarse-to-fine 的 information hierarchy 用对了**。Semantic 稀疏但强，用于 global initialization；photometric dense 但需要大致对齐，用于 local refinement。SIREN 严格按这个 hierarchy 走，prior work 跳过 semantic 直接上 photometric，所以在 zero-init 下崩。

---

## 实验结果有多夸张

Mobile robot 场景（quadruped mapping kitchen）：
- SIREN: rotation error 0.25°, translation error 3.17
- GaussReg（最强 baseline）: rotation error 40.89°, translation error 1477

改进 **160x rotation, 465x translation, 488x scale**。

Workshop 场景更夸张，SIREN-R vs GaussReg 是 **415x rotation, 1287x translation, 2962x scale**。

Abstract 里的 "90x / 300x / 44x" 是 SIREN-R vs 最强 baseline 的 worst-case。

Mip-NeRF360 标准数据集上 SIREN 也是全面最优，除了 Room scene 的 rotation error 被 PhotoReg 拿了——但 PhotoReg 的 translation error 巨大，说明它 rotation 对了但 translation 跑飞了，registration 是 SE(3) joint problem，不能单独看一个 metric。

---

## 一个有意思的坑

Truck scene 里 RANSAC-GR 表面 PSNR/SSIM 最高，但仔细看 std 是 SIREN 的 2-3x。Figure 3 揭示原因：RANSAC-GR 实际把 truck 左半部分完全丢了，右半部分 render 看着 OK。Mean photometric metric 被 "能 render 的那一半" 拉高了，掩盖了 catastrophic registration failure。

**Takeaway**：评估 registration 不能只看 mean photometric metric，必须配合 geometric error + qualitative inspection。

---

## Finetuning trick

Fused map 会继承 submap 里的 floaters（GSplat 在 low-supervision region 的经典 artifact）。SIREN 不预处理。

Trick：用 fused map 自己 render 的 images 做 finetune，完全不需要 real-world data。70-90 秒就能把 PSNR 从 29.1 提到 30.8，缩小跟 ground-truth (36.3) 的 gap 约 20-40%。

---

## Limitation

依赖 semantic。如果 submap 没训 semantic 或 scene 没有语义结构（比如纯几何的 factory floor），SIREN 退化。Mitigation：可以 post-train semantic 进已有 GSplat，或者直接用 2D vision foundation model 在 rendered image 上 back-project。

假设 global rigid transform。如果两个 submap 在不同时间拍（场景变化），需要 non-rigid registration，SIREN 没做。

Computation time ~40% 花在 semantic extraction。FastSplat 可以加速。

---

## 我的 takeaway

SIREN 的真正贡献不是某个具体公式，而是**把 semantics 作为 registration 的 first-class citizen**，从 correspondence finding 到 outlier rejection 到 image filtering 全链路贯穿。在 minimal-overlap、cross-embodiment、zero-init 的真实机器人场景，semantic prior 是必须的。

但也要注意：baseline 在 robot 场景崩得这么惨，部分原因是 baseline 的设计假设（overlap 大、viewpoint 相近）被违反。SIREN 的 100-1000x 改进不是 "fair comparison 下的碾压"，而是 "在 baseline 不适用的场景下 SIREN 依然 work"。

后续可以 watch：(1) DINOv2 替代 CLIP 做 semantic field，dense feature 更强；(2) object-level non-rigid registration；(3) online multi-robot GSplat-SLAM with SIREN-style loop closure；(4) 3D foundation model（如 3D-LLM）直接 ground 在 GSplat 上做 correspondence。

---

# SIREN: 多机器人 Gaussian Splatting 地图的 Semantic Registration 详解

## 1. 核心 Problem 与 Motivation

**Problem setup**：多个机器人（quadruped、drone、manipulator）各自部署一次，独立 train 一个 GSplat submap，需要 fuse 成一个 global map。Prior work (Nerf2nerf, DReg-NeRF, LoopSplat, PhotoReg, GaussReg) 都假设有 access 到 camera poses、原始 images，或者有 inter-map relative pose 的良好 initialization。SIREN 完全消除这些假设。

**Key insight**：在 registration 问题中，真正的瓶颈是**correspondence identification**。如果给你准确的 point-to-point correspondences，registration 在 closed-form 下可解 (Umeyama 1991)。SIREN 用 semantics 来 anchor correspondence 搜索，因为 semantics 对 viewpoint、illumination、scale 鲁棒——同一个 "chair" 在两个 submap 里 semantic embedding 应该 close。

**为什么用 semantics 而非 geometry**：
- 几何 descriptors (FPFH, local surface patches) 对 noise、density variation、scale 敏感
- Texture-based methods (SIFT、SuperPoint) 在跨 submap 时 viewpoint 差异大，重复性差
- Semantic features 来自 vision-language foundation model (CLIP / DINO)，被 pretrained 在 billion-scale data 上，cross-domain robustness 显著强

参考：[Umeyama 1991 Least-squares estimation of transformation parameters](https://ieeexplore.ieee.org/document/88473)、[CLIP](https://openai.com/research/clip)、[LERF](https://lerf.io/)

---

## 2. SIREN Pipeline Overview

```
Local GSplat Maps (G1, G2)
        ↓
[Step 1] Semantic Feature Extraction & Matching
        ↓ Correspondence set E = {(i,j)}
[Step 2] Coarse Gaussian-to-Gaussian Registration (closed-form)
        ↓ R_c*, s_c*, t_c*
[Step 3] Fine Photometric Registration (SfM + Bundle Adjustment)
        ↓ R_f*, s_f*, t_f*
        ↓
Fused GSplat Map G_f
        ↓ (optional)
Finetuning on rendered images
```

三步走的核心 motivation：**coarse-to-fine**。Coarse 用稀疏 semantic anchors 在 SE(3) + scale 上做 closed-form global alignment，把 submaps 从 "完全不知道对方在哪" 拉到 "大致重合"；fine 用 photometric loss 在 dense feature 上 refine，恢复 GSplat 的 photorealism。

---

## 3. Step 1: Semantic Gaussian Splatting 与 Feature Matching

### 3.1 Semantic Field 训练

Prior semantic GSplat 工作 (LangSplat、Feature 3DGS) 通常训一个 autoencoder / CNN 把 CLIP feature (d=512) 降维到 ~3D 再 distill 进 Gaussian。SIREN 跳过这个 compression，直接 train 一个 semantic field：

$$\psi: \mathbb{R}^3 \mapsto \mathbb{R}^d$$

参数化用 **multi-resolution neural hash grid**（来自 Instant-NGP 的设计思想），与 GSplat 联合训练：

$$\mathcal{L} = \mathcal{L}_{\mathrm{gs}} + \gamma \sum_{\mathcal{I} \in \mathcal{D}} \|\mathcal{I}_f - \hat{\mathcal{I}}_f\|_F^2 - \beta \sum_{\mathcal{I} \in \mathcal{D}} \phi(\mathcal{I}_f, \hat{\mathcal{I}}_f)$$

变量解释：
- $\mathcal{L}_{\mathrm{gs}}$：原始 GSplat rendering loss（L1 + D-SSIM）
- $\mathcal{I}_f \in \mathbb{R}^{W \times H \times d}$：ground-truth semantic feature map（来自 CLIP image encoder 在 training image 上 forward 得到）
- $\hat{\mathcal{I}}_f$：predicted semantic feature map
- $\gamma, \beta$：relative weights
- $\phi$：cosine similarity
- 第二项是 L2 reconstruction on feature map，第三项是 cosine similarity maximization（对比学习味道）

**Key trick**：GSplat 自带高质量 depth（这是 GSplat 相对 NeRF 的隐藏优势之一，2D GSplat 论文也强调了这点）。所以可以**直接 back-project** 像素到 3D world，query $\psi$ 得到 semantic feature，无需像 NeRF 那样 train proposal network 来 sample ray termination points。这避开了 Mip-NeRF / Zip-NeRF 那套 coarse-to-fine proposal sampling 的 memory 开销。

参考：[2D Gaussian Splatting](https://buaacyw.github.io/2d-gaussian-splatting/)、[Instant-NGP](https://nvlabs.github.io/instant-ngp/)、[LangSplat](https://langsplat.github.io/)

### 3.2 Feature Extraction via Semantic Localization

Train 完后，每个 Gaussian $G_i$ 增加一个 semantic attribute，就是 query $\psi$ at Gaussian mean $\mu_i$。

然后做 **open-vocabulary localization**（LERF 风格的 relevancy score）：

$$\mathrm{rel}(G_i, \text{query}) = \frac{\exp(\cos(f_i, e_\text{query}))}{\exp(\cos(f_i, e_\text{query})) + \exp(\cos(f_i, e_\text{null}))}$$

其中 $f_i$ 是 $G_i$ 的 semantic feature，$e_\text{query}$ 是 CLIP text encoder 对 "chair" 之类的 query 输出，$e_\text{null}$ 是 generic object 的 text embedding（denominator trick 防止 relevancy 总是高）。

通过一组 query (e.g., {"chair", "table", "vehicle", ...})，把每个 submap 中**feature-rich region** 抽出来。这是 SIREN 比 ICP 类方法强的根本原因之一：ICP 把所有点都参与 alignment，被 featureless regions（地板、墙面）淹没；SIREN 只在 informative regions 上做 correspondence。

### 3.3 Feature Matching

给定 source Gaussians $\{p_i\}$ 和 target Gaussians $\{q_j\}$：

1. 对每个 source Gaussian $p_i$，在 target map 里找 within distance $r$ 的 candidate Gaussians（KD-tree 加速）
2. 从 candidate 中 sample $M$ 个，可 uniform 或按 cosine-similarity 加权采样
3. 计算 cosine similarity，构成 correspondence set $\mathcal{E} = \{(i, j)\}$
4. 可以混合 FPFH geometric descriptor 作为 secondary signal

注意这里没有强制 1-to-1 匹配，一个 source Gaussian 可能对应多个 target candidates。后面用 RANSAC + robust weight 来处理 outlier。

---

## 4. Step 2: Coarse Gaussian-to-Gaussian Registration（公式深度解析）

### 4.1 原始 Formulation

$$\underset{s_c, R, t}{\text{minimize}} \quad \frac{1}{2} \sum_{(i,j) \in \mathcal{E}} w_{ij} \left( \|s_c R p_i + t - q_j\|_2^2 + \|s_c^2 R \Sigma_{p_i} R^\top - \Sigma_{q_j}\|_F^2 \right)$$

变量逐个解释：
- $s_c \in \mathbb{R}_{++}$: scale（标量，必须正），因为两个 submap 可能用不同 unit 训练，或 measurement scale 不一致
- $R \in \mathrm{SO}(3)$: rotation（3D 旋转矩阵）
- $t \in \mathbb{R}^3$: translation
- $p_i, q_j \in \mathbb{R}^3$: source / target Gaussian 的 mean（位置）
- $\Sigma_{p_i}, \Sigma_{q_j} \in \mathbb{R}^{3 \times 3}$: Gaussian 的 covariance（描述 Gaussian 椭球的形状和朝向）
- $w_{ij}$: weight，proportional to cosine similarity of semantic embeddings，**用来 down-weight spurious correspondences**
- $\|\cdot\|_F$: Frobenius norm

第一项是 position alignment（经典 point-to-point），第二项是 **shape/orientation alignment**（这是 SIREN 相对 Umeyama 1991 的扩展——利用 Gaussian 不仅有 position 还有 covariance 这个丰富信息）。

### 4.2 简化到 Closed-Form Solvable Form

利用 GSplat 的 covariance 分解：
$$\Sigma_{p_i} = H_{p_i} \Lambda_{p_i} \Lambda_{p_i}^\top H_{p_i}^\top$$

其中：
- $H_{p_i} \in \mathrm{SO}(3)$: Gaussian 的 rotation（orientation）
- $\Lambda_{p_i} = \mathrm{diag}(\lambda_1, \lambda_2, \lambda_3)$: Gaussian 的 scaling（沿 3 个主轴的半轴长）

把这个分解代进 covariance 项：

$$s_c^2 R \Sigma_{p_i} R^\top = s_c^2 R H_{p_i} \Lambda_{p_i} \Lambda_{p_i}^\top H_{p_i}^\top R^\top$$

要等于 $\Sigma_{q_j} = H_{q_j} \Lambda_{q_j} \Lambda_{q_j}^\top H_{q_j}^\top$。

如果我们 take square root 形式（不严格，但是关键 reformulation），等价于：

$$s_c R H_{p_i} \Lambda_{p_i} \approx H_{q_j} \Lambda_{q_j}$$

定义 $\check{H}_{p_i} = H_{p_i} \Lambda_{p_i}$, $\check{H}_{q_j} = H_{q_j} \Lambda_{q_j}$（即 covariance 的 "square root"），reformulation 变成：

$$\underset{s_c, R, t}{\text{minimize}} \quad \frac{1}{2} \sum_{(i,j)} w_{ij} \left( \|s_c R p_i + t - q_j\|_2^2 + \|s_c R \check{H}_{p_i} - \check{H}_{q_j}\|_F^2 \right) \quad (\text{公式 3})$$

这是一个 **weighted Umeyama with shape anchors** 问题。

### 4.3 Closed-Form Solution 推导

**Step A: Optimal Translation**

对 $t$ 求一阶导 = 0：

$$\nabla_t J = \sum_{(i,j)} w_{ij}(s_c R p_i + t - q_j) = 0$$

得：

$$t_c^* = \tilde{\mu}_Q - s_c^* R^* \tilde{\mu}_P$$

其中 $\tilde{\mu}_P, \tilde{\mu}_Q$ 是 weighted mean：
$$\tilde{\mu}_P = \frac{\sum w_{ij} p_i}{\sum w_{ij}}, \quad \tilde{\mu}_Q = \frac{\sum w_{ij} q_j}{\sum w_{ij}}$$

直觉：最优平移就是把两个 weighted centroid 对齐。

**Step B: Substitution + Centering**

代入 $t^*$，定义 zero-centered 矩阵 $\check{P} \in \mathbb{R}^{3 \times N}$，第 $i$ 列为 $p_i - \tilde{\mu}_P$，类似 $\check{Q}$。问题变成：

$$\underset{s_c, R}{\text{minimize}} \quad \frac{1}{2}\|(s_c R \check{P} - \check{Q}) W^{1/2}\|_F^2 + \frac{1}{2}\sum w_{ij} \|s_c R \check{H}_{p_i} - \check{H}_{q_j}\|_F^2$$

利用 $\|A\|_F^2 = \mathrm{trace}(A^\top A)$ 转 trace minimization，这是关键 trick，因为 trace 可以分离 $R$ 和 $s_c$：

$$\underset{R \in \mathrm{SO}(3)}{\text{minimize}} \quad -\mathrm{trace}\left(R^\top \underbrace{\left(\check{Q} W \check{P}^\top + \sum w_{ij} \check{H}_{q_j} \check{H}_{p_i}^\top\right)}_{M}\right)$$

这是经典的 **Orthogonal Procrustes** 问题（[Gower & Dijksterhuis 2004](https://link.springer.com/article/10.1007/s10182-004-0153-0)）。设 $M = U_c \Sigma_c V_c^\top$ (SVD)，则：

$$R_c^* = U_c \Theta_c V_c^\top, \quad \Theta_c = \mathrm{diag}(1, 1, \det(U_c V_c^\top))$$

最后的 $\det$ 项保证 $R \in \mathrm{SO}(3)$（避免 reflection）。

**Step C: Optimal Scale**

对 $s_c$ 求一阶导 = 0：

$$s_c^* = \frac{\mathrm{trace}(\Theta_c \Sigma_c)}{\mathrm{trace}\left(W \check{P}^\top \check{P} + \sum w_{ij} \check{H}_{p_i}^\top \check{H}_{p_i}\right)}$$

分子是 cross-correlation 的强度，分母是 source side 的 norm（包含 position + shape contribution）。形式上就是 weighted Umeyama 的 scale，但分母多了 $\check{H}_{p_i}^\top \check{H}_{p_i}$ 项——这是 covariance anchor 的贡献。

### 4.4 RANSAC Wrapper

虽然 closed-form 对 fixed correspondence 是最优的，但 $\mathcal{E}$ 里可能有 false matches。SIREN-R 用 RANSAC 反复 sample subset → solve (公式 4-6) → 用 inlier 集重 solve，提升鲁棒性。SIREN-NR 直接用全部 correspondence 一次性 solve。

---

## 5. Step 3: Fine Photometric Registration

Coarse 步骤只用 Gaussian 的几何属性（mean + covariance），丢了 GSplat 最强大的 visual fidelity。Fine 步骤用 novel-view synthesis 渲染图像，做 image-level registration。

### 5.1 Image Generation

利用 coarse 结果，把 source map 的 camera pose 变换到 target frame，在两个 map 里渲染 "对应视角" 的图像。关键 challenge：**没访问 original camera poses**，怎么选 pose？SIREN 在 semantic submap 区域内随机采样 pose，然后两 map 同时 render。

**Filter via semantics**：不是所有 rendered pair 都 informative。用 CLIP 计算 image embedding 的 cosine similarity，过滤掉低相似度 pair。

### 5.2 Feature Extraction & Matching

- **NetVLAD**：global image descriptor（weakly supervised place recognition 经典方法）
- **SuperPoint**：local feature detector (corners, edges, blobs)
- **SuperGlue**：graph neural network based feature matcher

相比 SIFT 更鲁棒，尤其在 cross-view、illumination variation 场景。

### 5.3 Image Registration + Triangulation

从 matched features 估 camera relative pose + 3D feature points 位置（standard SfM pipeline，[COLMAP](https://colmap.github.io/)）。

### 5.4 Bundle Adjustment

Joint optimize camera poses 和 3D feature points via nonlinear least squares，用 Levenberg-Marquardt。得到每个 image 在一个 common frame $\mathcal{A}$ 下的 pose。

### 5.5 Final SE(3) Registration

现在有两套 camera poses：在 source/target frame 下的（通过 GSplat 渲染时已知）和在 $\mathcal{A}$ 下的（通过 SfM+BA 估出来）。求 $\mathcal{A}$ 到 source / target 的 transform：

$$\underset{s_f, R, t}{\text{minimize}} \quad \frac{1}{2}\sum_{(i,j) \in \mathcal{V}} \left(\|s_f R a_i + t - b_j\|_2^2 + \beta_{ij} \|R R_{a_i} - R_{b_j}\|_F^2\right) \quad (\text{公式 7})$$

变量：
- $a_i \in \mathbb{R}^3$: camera origin in frame $\mathcal{A}$
- $b_j \in \mathbb{R}^3$: 对应 camera origin in $B_s$ 或 $B_t$
- $R_{a_i}, R_{b_j}$: camera orientations
- $\beta_{ij}$: rotation error 的 weight

**Asymptotic closed-form**：当 $\beta_{ij} \to 0$，问题退化成经典 weighted Umeyama：

$$R_f^* \to U_f \Theta_f V_f^\top, \quad U_f \Sigma_f V_f^\top = \check{B}\check{A}^\top$$
$$s_f^* \to \frac{\mathrm{trace}(\Theta_f \Sigma_f)}{\mathrm{trace}(\check{A}^\top \check{A})}$$
$$t_f^* \to \mu_B - s_f^* R_f^* \mu_A$$

实际中 $\beta$ 小但非零，用 iterative optimization（Riemannian / sequential convex programming）refine。

### 5.6 Map Fusion

由 $\mathcal{A} \to B_s$ 和 $\mathcal{A} \to B_t$ compose 得到 $B_s \to B_t$ 的 transform，把 source map 的所有 Gaussian transform 到 target frame，merge 两个 Gaussian set，得到 fused GSplat。

---

## 6. Experiments 详解

### 6.1 Setup

- **Datasets**: Mip-NeRF360 (Playroom, Truck, Room) + mobile robot data (Kitchen/Workshop by quadruped, Apartment by drone) + tabletop manipulator
- **Hardware**: Unitree Go1 quadruped, Modal AI drone, Franka Panda manipulator (wrist camera)
- **Training**: Nerfstudio 用于 SIREN, 原 GSplat repo 用于 baseline
- **Compute**: RTX 3090 24GB (SIREN) vs H20 (baselines)
- **GSplat iterations**: 30000

### 6.2 Baselines

| Method | Type | 需要 camera poses? | 需要 images? | 需要 init? |
|---|---|---|---|---|
| PhotoReg | photometric GSplat reg | Yes | Yes | No |
| GaussReg | geometric transformer + 2D CNN | No | No | No (but needs pre-trained CNN) |
| RANSAC-GR | classical global point cloud reg | No | No | No |
| FGR | Fast Global Registration | No | No | No |
| ICP / Colored-ICP | local refinement | No | No | Yes |
| **SIREN** | **semantic + closed-form + SfM** | **No** | **No** | **No** |

### 6.3 Mip-NeRF360 Geometric Results (Table I)

以 Playroom 为例，rotation error (RE)：
- SIREN-R: **0.170** deg
- SIREN-NR: 0.348 deg
- Colored-ICP: 0.194 deg
- GaussReg: 0.766 deg
- PhotoReg: 6.036 deg

Translation error (TE) 上 SIREN-R 达到 1.933，比 PhotoReg (18806) 好 ~10000x。

注意 Room scene 中 PhotoReg 居然拿到最低 RE (0.161)，但 TE 和 SE 巨大，说明 rotation 对了但 translation 完全跑飞——registration 是 SE(3) joint problem，单独看一个 metric 容易误导。

### 6.4 Mobile Robot Results (Table II) — 最 dramatic 的结果

**Kitchen (quadruped)**:
- SIREN-NR: RE=0.253, TE=3.173, SE=0.352
- GaussReg: RE=40.89, TE=1477, SE=171.8
- 改进倍数：~160x RE, ~465x TE, ~488x SE

**Workshop (quadruped)**:
- SIREN-R: RE=0.134, TE=7.400, SE=10.88
- GaussReg: RE=55.66, TE=9531, SE=4305
- 改进倍数：~415x RE, ~1287x TE, ~2962x SE (与 PhotoReg 比是 90x / 300x / 44x)

Abstract 里的 "90x rotation, 300x translation, 44x scale" 就来自这里——SIREN-R vs 最强 baseline (GaussReg) 的 worst-case 改进。

**为什么 baseline 在 robot 数据上崩**：robot submaps 的 overlap 极小（Kitchen/Workshop 都 minimal overlap），且 robot viewpoint 高度 idiosyncratic（quadruped 低视角、drone 俯视）。Photometric / geometric matching 在 minimal overlap 下找不到足够 inlier。

### 6.5 Photometric Results (Table III, IV)

- Playroom: SIREN-R PSNR=28.3, SSIM=0.90, LPIPS=0.15，全面最优
- Room: SIREN-NR PSNR=24.8 最好
- Truck: RANSAC-GR 表面 PSNR/SSIM 高，但**仔细看 std**：std 是 SIREN 的 2-3x。Figure 3 揭示原因——RANSAC-GR 实际把 truck 左半部分丢了，右半部分 render 看着 OK，但左半部分完全 missing。Mean PSNR 被 "renderable half" 拉高了。

这个观察非常 methodologically 重要：**mean photometric metric 会掩盖 catastrophic registration failure**，需要配合 geometric error + qualitative inspection。

### 6.6 Ablation (Table V, VI)

四个 variant：
- **SIREN-CNR**: coarse no RANSAC + no fine → 表现极差 (Playroom RE=22.72)
- **SIREN-CR**: coarse RANSAC + no fine → 中等
- **SIREN-NR**: coarse no RANSAC + fine → 好
- **SIREN-R**: coarse RANSAC + fine → 最好

**关键结论**：
1. Fine registration 是必须的——coarse-only 即使有 RANSAC，error 还在 tens of degrees
2. Fine registration 把 rotation error 推到 sub-degree，translation 改进可达 100x
3. RANSAC 在 coarse 步骤提供更 robust 的 initialization，让 fine 步骤能找到足够 inlier

### 6.7 Finetuning (Table VII)

Fused map 里有 floaters（GSplat 在 low-supervision region 的经典 artifact）。SIREN 不预处理 submap，所以 floater 会被继承。

**Trick**：用 fused map 自身 render 的 images 做 finetune，**完全不需要 real-world data**。流程：
1. 选 camera pose (in source 或 target local frame)
2. 从原 submap 渲染 image
3. Transform pose 到 fused frame
4. 用 构成 finetune dataset
5. Finetune fused GSplat ~70-90 sec

结果（Playroom）：
- Pre-finetune: PSNR=29.1
- Post-finetune: PSNR=30.8 (+1.7)
- Ground-truth: PSNR=36.3

Gap 缩小约 20-40%。这是非常实用的 trick——no real-world interaction needed to clean up floaters。

---

## 7. Limitations

1. **依赖 semantic**：如果 submap 没训 semantic 或 scene 没有语义结构，SIREN 退化。Mitigation: 可 post-train semantic 进已有 GSplat，或直接用 2D vision foundation model 在 rendered image 上 back-project。
2. **Floaters**：fused map 继承 submap floaters。Mitigation: finetune on rendered images。
3. **Computation time**: ~40% 时间花在 semantic extraction。FastSplat (Shorinwa et al. 2024) 可以加速。
4. **Beta 的选择**：公式 (7) 中 $\beta_{ij}$ 的设定没充分讨论，作者用 asymptotic closed-form 近似。
5. **Non-rigid assumption**: SIREN 假设两个 submap 间是 single similarity transform (scale + rotation + translation)，没考虑 submap 内部 deformation（e.g., GSplat 训练 noise 导致局部 distortion）。

---

## 8. Intuition Building: 为什么 SIREN Work

### 8.1 Registration 问题的本质困难

Point cloud registration 表面看是 SE(3) optimization，**实际困难全在 correspondence**。给定准确 correspondence，Umeyama 1991 给 closed-form 解。所有现代方法（ICP、RANSAC-GR、FGR、GeoTransformer、GaussReg）本质上都在 attack correspondence 问题：
- ICP: 用 nearest neighbor 当 correspondence，迭代
- RANSAC-GR: 用 FPFH descriptor 匹配 + RANSAC outlier rejection
- GeoTransformer: 学习 geometric transformer descriptor
- GaussReg: 用 pretrained CNN + geometric transformer

这些方法在 cross-robot、cross-deployment 场景失败，因为：
1. Viewpoint 差异巨大 → local descriptor 重复性低
2. Density / scale 不一致 → geometric descriptor 失真
3. Overlap 极小 → inlier 比例低，RANSAC 失效

### 8.2 Semantic 是 correspondence 的"高维 prior"

CLIP feature 把 image patch embed 到 512D space，其中 "chairness"、"tableness" 这些抽象语义被 cluster。两个不同机器人拍的 "同一把椅子"，即使视角、光照、scale 完全不同，CLIP embedding 仍 cosine similarity > 0.8。这给了 correspondence 一个**极强的 prior**，远超 FPFH 之类的 geometric descriptor 能力。

### 8.3 Gaussian Covariance 作为额外 anchor

经典 point cloud registration 只用 point position。GSplat 里每个 Gaussian 还有 covariance $\Sigma$，描述形状和朝向。SIREN 把 covariance 作为额外 alignment signal（公式 3 第二项）相当于**双倍 correspondence**——每个 anchor 点提供 9 个 scalar constraint（3 position + 6 covariance），而非 3 个。这让 closed-form 解更 robust，尤其在 correspondence 数量少时。

### 8.4 Coarse-to-Fine 的 Information Hierarchy

- **Semantic**：稀疏、强、跨 viewpoint robust → 用于 global initialization
- **Photometric + learned feature**：dense、强、但需要大致对齐 → 用于 local refinement
- **Bundle Adjustment**：高精度、joint optimize → 收尾

SIREN 严格按这个 hierarchy 设计。Prior work（PhotoReg、GaussReg）跳过了 semantic 这一层，直接从 photometric / geometric 开始，所以在 zero-initialization 下崩。

---

## 9. 相关联想与 Open Problems

### 9.1 与 NeRF Registration 的对比
- Nerf2nerf (Goli et al. 2023): 需要人 manual keypoint annotation
- DReg-NeRF (Chen & Lee 2023): 用 learned 3D feature descriptor + RANSAC，没利用 semantics

### 9.2 与 SLAM 的关系
SIREN 解决的是 "map fusion" 而非 "online SLAM"。但思路可以直接 plug into GSplat-SLAM（如 SplaTAM、MonoGS）做 multi-robot loop closure，替代传统 pose graph optimization。

### 9.3 与 Foundation Model 趋势
SIREN 用 CLIP，但 DINOv2、SAM、Depth-Anything 都可以 plug in。特别是 DINOv2 的 dense feature 在 cross-view matching 上比 CLIP 强，可作为 SIREN 的 semantic field 替代。

### 9.4 与 Multi-Agent RL 的 connection
Multi-robot mapping 本质上是 decentralized collaborative perception。SIREN 是 offline version，online 版本需要考虑 bandwidth、async communication、partial observability——这是 multi-agent RL + GSplat 的开放方向。

### 9.5 Extension: Non-Rigid Registration
公式 (2) 假设 global rigid transform。如果两个 submap 在不同时间拍摄（场景变化），需要 non-rigid / Gaussian-level deformation field。这跟 NeRFEditing、GS deformation work（如 SC-GS、Deformable-3DGS）思路接近。

### 9.6 Open: Object-Level Registration
Semantic 已经 localize 到 object，可以做 object-level registration（per-object SE(3) + global consistency），更精细但需要 object discovery + multi-object joint optimization。

---

## 10. 关键 Links

- [Gaussian Splatting 原始项目页](https://repo1.dso.org/inria/gaussian-splatting)
- [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/)
- [Nerfstudio](https://docs.nerf.studio/)
- [LERF: Language Embedded Radiance Fields](https://lerf.io/)
- [LangSplat: 3D Language Gaussian Splatting](https://langsplat.github.io/)
- [Feature 3DGS](https://feature-3dgs.github.io/)
- [CLIP (OpenAI)](https://openai.com/research/clip)
- [DINOv2](https://dinov2.metaproject.dev/)
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [NetVLAD](https://github.com/Relja/netvlad)
- [COLMAP (SfM)](https://colmap.github.io/)
- [Umeyama 1991 (Stanford CS273 reading)](https://web.stanford.edu/class/cs273/reading/umeyama.pdf)
- [Fast Global Registration (FGR)](https://github.com/opencv/opencv_contrib/tree/master/modules/reg)
- [Geometric Transformer (GeoTransformer)](https://github.com/qinzheng93/GeoTransformer)
- [GaussReg](https://github.com/aigcz/GaussReg)（如开源）
- [PhotoReg arXiv](https://arxiv.org/abs/2410.05044)
- [LoopSplat arXiv](https://arxiv.org/abs/2408.10154)
- [2D Gaussian Splatting](https://buaacyw.github.io/2d-gaussian-splatting/)
- [SplaTAM (GSplat-SLAM)](https://github.com/naver/splatam)
- [FastSplat (Shorinwa et al.)](https://arxiv.org/abs/2411.13753)
- [Splat-Mover](https://arxiv.org/abs/2410.02602)
- [Distilled Feature Fields (F3RM)](https://f3rm.github.io/)

---

## 11. 总结

SIREN 的核心贡献是把 **semantics 作为 registration 的 first-class citizen**，从 feature extraction、correspondence matching、outlier rejection 到 image filtering 全链路贯穿。技术上最 elegant 的部分是公式 (3) 的 closed-form derivation——把 Gaussian covariance 拆解成 $H\Lambda$ 后，Umeyama 的 Procrustes 框架自然 extend 到 "position + shape" joint alignment。实验上 mobile-robot 场景的 100-1000x 改进非常 compelling，但也要注意：这些场景 baseline 崩得很惨，部分原因是 baseline 设计的假设（overlap 大、viewpoint 相近）被违反，而非 SIREN 在 "fair comparison" 下碾压。真正的 takeaway 是：**在 minimal-overlap、cross-embodiment、zero-init 的真实机器人场景，semantic prior 是必须的**。

后续可以 watch 的方向：(1) DINOv2 / SAM 替代 CLIP 做 semantic field；(2) object-level non-rigid registration；(3) online multi-robot GSplat-SLAM with SIREN-style loop closure；(4) foundation model 的 3D native extension（如 LLaVA-3D、3D-LLM）直接 ground 在 GSplat 上做 correspondence。
