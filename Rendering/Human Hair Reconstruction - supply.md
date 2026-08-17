---
source_pdf: Human Hair Reconstruction - supply.pdf
paper_sha256: c13cfb90a585fc3fbe880145770a694ceeb9f63f5c95c667d34a03e88c4a2eef
processed_at: '2026-08-05T07:39:53-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Gaussian Haircut

## 一句话版本

你想把一个人的头发从照片"3D 扫描"出来,得到的不是一堆好看但没用的零散 points,而是真正的 hair strands——一根一根的、可以塞进 Unreal Engine 里跑物理模拟的那种。这篇 paper 就做了这事,而且比之前快 10 倍。

Project page: https://eth-ait.github.io/GaussianHaircut

---

## 为什么这事难

头发这东西在 vision 里特别烦,几个矛盾凑一起:

**矛盾一:可见的太少**。你拍 100 张照片,只能拍到头发最外层那一圈"壳"。里面 90% 的 hair volume 你永远看不到。可你要 simulate,就得知道里面长啥样——否则头一动,内层头发会互相穿透,物理完全不对。

**矛盾二:表示方式打架**。Computer graphics 里 simulate 头发的"金标准"是 **strands**——3D polylines,每根线从头皮出发一直到发梢。可以 attach 到 scalp、可以跑 collision、可以保持长度。问题是这种 1D 结构特别难从 2D images 反推——你怎么知道某个 pixel 对应 strand 的哪个点?

另一条路是 **3D Gaussian Splatting**(3DGS,2023 年爆火的那个)。它能 photorealistic 地 reconstruct 任意 scene,渲染又快又好看。可惜输出就是一堆**零散的 Gaussians**,没有"哪根连哪根"的拓扑信息。拿这玩意去 simulate 头发,就像拿一把沙子去算水怎么流。

**矛盾三:supervision signal 噪声大**。传统方法从 2D images 里提取头发方向,用的是 Gabor filter banks——基本就是一堆 oriented edge detectors。出来的叫 **2D orientation maps**。问题是这东西特别 noisy,尤其是在头发密、光照差的地方。你拿这噪声去 supervise 几万根 strands,很容易拟合到 garbage。

Neural Haircut(Sklyarova et al. ICCV 2023,https://saicv.github.io/NeuralHaircut/)是之前 SOTA,也是这篇 paper 的主要 baseline。它用 NeuS 重建一个 hair surface mesh,再用 strands 去 fit 这个 surface。问题是 mesh rasterizer 没法 propagate 高频细节,photometric loss 基本无效,最后还是靠 3D 几何约束硬拉 strands。慢,且细节不够。

---

## 核心思路:dual representation

作者的核心 insight 就一句话:

> 既然 3DGS 渲染好但不 simulate,strands 能 simulate 但渲染差,那就**把每根 strand 的每一段 line segment 绑一个 Gaussian**。

这样你有两层东西叠在一起:

- **Strand 那层**是你的 ground truth——3D polylines,FLAME head mesh 上发根,有拓扑,有长度,可以进 physics engine;
- **Gaussian 那层**是"渲染代理"——每个 line segment 上挂一个又细又长的 anisotropic Gaussian,只负责把 strand 渲染成 image,接收 photometric gradient,再把这个 gradient 反传回 strand 的 control points。

一个 strand polyline 是一串 3D points $\{p_1, p_2, \dots, p_L\}$。每相邻两点 $p_l, p_{l+1}$ 形成 line segment,绑一个 Gaussian。Gaussian 的 scale 参数是这样设的:

$$s_l = \left\{\frac{1}{2}\|p_{l+1} - p_l\|_2,\ \epsilon,\ \epsilon\right\}$$

变量解释:
- $s_l$ 是 Gaussian 的 3D scale 向量,三个分量对应三个 principal axes;
- 第一个分量 $\frac{1}{2}\|p_{l+1} - p_l\|_2$ 沿 strand 切线方向,等于 segment 长度的一半(这样 Gaussian 刚好"覆盖"整段线);
- 第二、三个分量 $\epsilon$ 是小常数(比如 0.01),沿切线垂直方向——这让 Gaussian 极扁,像一个"胶囊"或"小棒";
- $p_{l+1}, p_l$ 是 strand 上相邻两个 control points 的 3D 坐标;
- $\|\cdot\|_2$ 是 L2 范数,就是欧氏距离。

Gaussian 的 rotation 把 x-axis 对齐到 segment direction。这样渲染出来,Gaussian 看起来就是一根细头发丝的形状,相邻 strands 的 Gaussians 不会互相 bleed。Opacity 全设 1(完全不透明),Color 用 spherical harmonics 学。

这个 design 让 gradient flow 路径清晰:image pixel → Gaussian α-blending → Gaussian position/rotation/scale → strand control points。**可微渲染的"最后一公里"被这个 dual representation 打通了**。

---

## 两阶段 pipeline

整篇 paper 的 pipeline 可以拆成两个 stage:

### Stage 1: 用 unstructured Gaussians 做 3D lifting

这一阶段**根本没用 strands**,就是普通的 3DGS,只是加了几个 modifications。

输入:multi-view images + COLMAP 的 cameras + Gabor filter 算的 noisy orientation maps。

每个 Gaussian 多了两个 learnable parameters:
- **hair segmentation label** $l$ —— 这是不是 hair;
- **orientation confidence** $\tau$ —— 这个 Gaussian 的 strand 方向有多可信。

Gaussian 的 covariance matrix $\Sigma = R S S^T R^T$ 里,$R$ 是从 quaternion $q$ 来的旋转矩阵,$S = \text{diag}(s)$。**covariance 的最大 eigenvector 就是 hair strand 的 3D 切线方向**——这是 3DGS 的 emergent property,加 supervision 让它更 explicit。

渲染 orientation 时,把每个 Gaussian 的最大方差方向 $\beta_i$ 按 α-blending 渲染到 pixel $p$:

$$\beta_p = \sum_{i=1}^{N} T_p^i \alpha_p^i \beta_i$$

变量解释:
- $\beta_p$ 是 pixel $p$ 渲染出来的方向(2D vector);
- $N$ 是 pixel $p$ 覆盖的 Gaussians 数量;
- $T_p^i = \prod_{j=1}^{i-1}(1 - \alpha_p^j)$ 是 transmittance,表示"前面 Gaussian 们剩多少没挡住";
- $\alpha_p^i$ 是 Gaussian $i$ 在 pixel $p$ 的 opacity(标准 Gaussian splatting 公式);
- $\beta_i$ 是 Gaussian $i$ 的最大方差方向(投影到 screen space)。

**Orientation loss** 长这样:

$$\mathcal{L}_{\text{dir}} = \sum_p \tau_p \min\{d(\beta_p, \hat{\beta}_p),\ d(\beta_p, \hat{\beta}_p) \pm \pi\} - \log \tau_p$$

变量解释:
- $\tau_p$ 是 pixel $p$ 渲染的 orientation confidence;
- $d(\cdot, \cdot)$ 是绝对角度差;
- $\hat{\beta}_p$ 是 Gabor filter 算的 ground truth orientation;
- $\min\{d, d \pm \pi\}$ 是因为 orientation 是 undirected line,差 π 和差 0 是一回事;
- 第一项 $\tau_p \cdot d$ 是 confidence-weighted angular error;
- 第二项 $-\log \tau_p$ 是 entropy regularization,**防止网络把 $\tau_p$ 全压成 0 来 trivially minimize loss**(这是 uncertainty-aware loss 的经典 trick,见 Kendall & Gal 2017)。

还有一个关键 trick:**camera refinement**。SfM 在头发场景里 camera localization 不准(头发 texture 重复、有 specular reflection),作者把 BARF [Lin et al. ICCV 2021, https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/] 的 6-DoF learnable camera parameterization 叠加在 COLMAP 初始估计上,前 15k steps 跟 Gaussians 一起 optimize,之后 freeze。

为什么这重要?头发单根直径 < 1mm,5cm 的 camera error 在 1m 距离上对应 ~3° 角度差,就是 ~5mm 位置 error——比单根 strand 直径大一个数量级。Camera 不准,后面 strand fitting 的 photometric loss 完全失效。

训练 30,000 steps,得到:(a) 优化过的 cameras;(b) 一组 unstructured Gaussians,每个都对齐了某根 strand 的方向。然后把这些 Gaussians 渲染出来,得到 clean 的 multi-view RGB + masks + orientation maps,作为 stage 2 的 supervision。

**Ablation 数据**:相比直接用 Gabor filter 算的 2D orientation maps(angular error 8°),3D line lifting 把 error 降到 7°。看起来提升小,但**关键在 3D consistency**——multi-view 的 orientation 互相一致了,不像 Gabor 那样每个 view 各算各的,view-inconsistent 噪声进入 stage 2 会搞乱 strand fitting。

### Stage 2: 优化 strand-based hairstyle

先把 FLAME head model [Li et al. SIGGRAPH Asia 2017, https://flame.is.tue.mpg.de/] fit 到 multi-view data(用 facial keypoints),作为头皮 surface。

头发表示成 **hair map** $H$——一张 scalp region 的 texture map,每个 texel 存一根 3D strand polyline $S^k = \{p_1^k, p_2^k, \dots, p_L^k\}$。

但 $H$ 自由度太高:30,000 strands × 100 control points × 3 coords = 9,000,000 DOF。直接 optimize 会 overfit。所以引入 **latent hair map** $Z$:

$$H = \mathcal{G}(Z), \quad Z = \mathcal{E}(H)$$

- $\mathcal{G}$ 是预训练的 strand decoder(在 synthetic hair collection 上训练,frozen);
- $\mathcal{E}$ 是预训练的 encoder;
- $Z$ 是低维 latent texture map(类似 VAE 的 latent space)。

**Coarse-to-fine 两步**:

#### Coarse step:优化 latent $Z$

每 step 从 $Z$ decode 出 1,000 根 guiding strands $H'$(memory 限制,不能一次 decode 全部 30,000 根),然后 **interpolate 到 10,000 根 dense strands** $\hat{H}$ 再 rasterize。

Interpolation 的细节:用 K-nearest neighbors 在 3D coordinate 空间做(不是在 latent space 做,这是跟 HAAR [Sklyarova et al. 2023, https://arxiv.org/abs/2312.14066] 的区别)。每根 strand 在自己 originating scalp face 的 TBN (tangent-bitangent-normal) basis 里定义,然后 inverse-distance weighting blend。

Trainable params:
- latent map $Z$;
- appearance decoder $\mathcal{G}_a$(架构同 $\mathcal{G}$,但**from scratch** 训练,负责预测每段 Gaussian 的 SH coefficients)。

#### Fine step:优化 explicit $H$

Latent decoder 直接 decode 全部 30,000 根 strands,**直接 optimize 它们的 3D coordinates** $H$ 和 SH coefficients $f_l^k$。

这一步能搞高频细节,因为 latent space 有 bottleneck,没法 express 出每根 strand 的精细 adjustment。Fine step 让你直接 touch strands 本身。

#### Diffusion-based prior(SDS)

内层头发看不见,得用 prior 补。沿用 Neural Haircut 的思路,用一个预训练的 latent diffusion model(hair-style prior),通过 **Score Distillation Sampling** [Poole et al. ICLR 2023, https://dreamfusionpaper.github.io/] 加约束。

Coarse 阶段:直接对 $Z$ 的 subsampled version $Z'$ 施加 SDS loss,跟 Neural Haircut 一样。

Fine 阶段 trick:diffusion model 只认 latent space,但 fine 阶段优化的是 explicit $H$。所以每 step:
1. 从 $H$ 随机抽 1,000 strands;
2. KNN interpolate 到 regular grid $H'$;
3. 用 encoder $\mathcal{E}$ encode 成 $Z'$;
4. 算 SDS loss,$\nabla \mathcal{L}_{\text{sds}}$ 通过 encoder 反传回 strands。

这里有个 stochastic regularization 的 trick:**每次只有 1,000 根 strands 接受 SDS gradient**,剩下 29,000 根这一 step 没 prior。但 10k steps 下来,所有 strands 都被 sample 过很多次,prior 累积 effective。

#### Final loss

$$\mathcal{L}_{\text{strand}} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{seg}}\mathcal{L}_{\text{seg}} + \lambda_{\text{dir}}\mathcal{L}_{\text{dir}} + \lambda_{\text{sds}}\mathcal{L}_{\text{sds}}$$

权重:$\lambda_{\text{seg}} = \lambda_{\text{dir}} = 0.1$,$\lambda_{\text{sds}} = 0.01$。SDS 权重小一点,避免 prior 过强 dominate photometric signal。

**总耗时**:~6 hours on RTX 4090,比 Neural Haircut 快 10× 以上。

---

## 为什么这个 design work

几个关键直觉:

### 1. Unstructured Gaussians 当"denoiser + 3D integrator"

Gabor filter 给的 2D orientation map 是 view-inconsistent 的——每个 view 各算各的,在 3D 里不一定一致。Unstructured Gaussians 因为 3DGS 的 multi-view photometric loss,会**被迫把不同 view 的 orientation 集成成 3D consistent field**。相当于一个免费的 multi-view stereo 集成。

而且 3DGS 的 emergent property 让 Gaussians 倾向 align 到 thin structures——所以即使你只用 photometric loss,Gaussians 也会自动沿头发丝方向变长椭球。再加 orientation loss,就 explicit 强化了这个 alignment。

### 2. Strand-aligned Gaussians 是 differentiable rasterization 的"hack"

Strand 本身是个 1D polyline,直接 rasterize 到 2D image 就是几个 pixel-wide 的细线,gradient signal 极弱。绑上 anisotropic Gaussians 后,每段 line segment 在 image space 有"footprint"——细但可微,gradient 可以 propagate。这是把 1D structure 用 3DGS 的成熟 rasterizer 做可微渲染的优雅方案。

### 3. Coarse-to-fine 让 latent 和 explicit 各发挥所长

Latent space 有 prior,但 bottleneck 限制 expressiveness;explicit space 表达力强,但 high DOF 容易 overfit。两阶段:
- Coarse:用 latent space,prior regularization 容易施加(直接 SDS on $Z$),得到大致形状;
- Fine:切到 explicit,加细节,SDS 用 round-trip trick 仍能施加。

---

## Experiments 数据

### Synthetic eval(用 Hair Meshes 数据集 [Yuksel et al. 2009])

2D orientation map angular error:
| 方法 | Angular Error |
|---|---|
| Gabor filter [Paris et al. 2004] | 8° |
| Gaussian Haircut 3D lifting | **7°** |

70 个 test views 平均。提升只有 1°,但**3D consistency** 是关键——stage 2 拿到的是 multi-view consistent supervision。

### Real-world eval

10× speedup vs Neural Haircut(6 hours vs 60+ hours)。

Ablation study(Fig. 6, 13, 14):
| 移除组件 | 效果 |
|---|---|
| w/o fine optimization | 内层结构不真实 |
| w/o synthetic orientation renders | dense strands 区域几何错误 |
| w/o strand upsampling | **fine fitting 不能收敛**(coarse 渲染 holes) |
| w/o $\mathcal{L}_{\text{dir}}$ | strand 方向乱 |
| w/o $\mathcal{L}_{\text{rgb}}$ | 颜色和几何都次优 |
| w/o $\mathcal{L}_{\text{sds}}$ | 内层 hair volume 不真实 |

### Applications

Reconstructed strands 可以:
- **秒级渲染** photorealistic images(Fig. 4);
- **直接进 Unreal Engine 做物理仿真**(Fig. 5),因为 strands + FLAME attachment + realistic internal structure 都满足。

---

## Limitations

- **Curly hairstyles**:diffusion prior 是 root-to-tip 设计,对 curly hair 表现差(Fig. 12);
- **Braids / complicated internal structures**:prior 训练数据没见过;
- 依赖预训练的 strand decoder / diffusion prior,这些 prior 偏 straight + wavy。

---

## 这篇 paper 的 takeaway

如果你做 3D vision / graphics,这篇 paper 的核心 lesson 是:**representation engineering 比单纯 algorithm 重要**。

3DGS、strands、SDS、BARF——这些都是 existing components。作者没发明新 framework,而是把它们重新组合,**用一个 dual representation 把"渲染友好"和"模拟友好"两个矛盾目标 unified**。

这种把 neural rendering output 真正"deploy 到 production graphics pipeline"的方向,可能是 vision-for-graphics 的下一个增长点。不只是头发——fur、feathers、cloth folds,凡是 thin-structure + 需要 physics simulation 的,都可以套这个 dual representation 模板。

---

## Reference 链接汇总

- **Gaussian Haircut project**: https://eth-ait.github.io/GaussianHaircut
- **arXiv**: https://arxiv.org/abs/2409.12978
- **3DGS (基础)**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Neural Haircut (baseline)**: https://saicv.github.io/NeuralHaircut/
- **Neural Strands (prior 来源)**: https://ait.ethz.ch/projects/2022/neural-strands/
- **HAAR (generative prior)**: https://arxiv.org/abs/2312.14066
- **BARF (camera 优化)**: https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/
- **DreamFusion (SDS)**: https://dreamfusionpaper.github.io/
- **FLAME (head model)**: https://flame.is.tue.mpg.de/
- **Hair Meshes (synthetic benchmark)**: https://www.cs.ubc.ca/labs/imager/tr/2009/hair_meshes/
- **Paris et al. 2004 (orientation maps)**: https://www-sop.inria.fr/reves/Basilic/2004/PSD04/
- **GaussianHair (concurrent)**: https://github.com/ProjectGaussianHair/GaussianHair
- **HairStep**: https://github.com/zhengyuf/HairStep
- **COLMAP**: https://colmap.github.io/
- **Unreal Engine**: https://www.unrealengine.com/
- **Blender**: https://www.blender.org/

---

# Gaussian Haircut: Strand-Aligned 3D Gaussians for Human Hair Reconstruction

## 1. Paper 概览

这篇 paper 来自 ETH Zürich + Max Planck Institute + Meta + TU Darmstadt 团队，一作是 Egor Zakharov (也是 Neural Haircut 的作者)，发表于 ECCV 2024。核心目标是解决一个 long-standing 的 vision problem：**从 multi-view images 重建 strand-based 3D 头发**，让结果可以**直接 plug 进 modern computer graphics engines (比如 Unreal Engine)** 用于 simulation/rendering/animation。

Project page: https://eth-ait.github.io/GaussianHaircut
arXiv: https://arxiv.org/abs/2409.12978
Supplementary code: https://github.com/EGois/GaussianHaircut (作者 release 的 reference 实现)

---

## 2. 核心问题与 motivation 的 intuition

要从多个视角的 2D images 重建头发之所以困难，根本原因是**头发几何严重 underconstrained**：

- 头发由 ~10⁴–10⁵ strands 组成，每根 strand 是一个 3D polyline；
- 但 multi-view capture 只能看到**外表面**的头发（visible hair surface），**内层 hair volume 完全被 occluded**；
- 即使可见的部分，由于头发是 quasi-dense thin structures，传统 mesh/volume 表示要么太重，要么不能直接做 physics simulation。

之前工作的思路可以分成两派：

**(A) Unstructured representations**（3DGS [Kerbl et al. 2023]、NeRF、volumetric primitives）：能 photorealistically render，但**输出的是一堆零散的 Gaussians / density fields**，**没有 strand 拓扑**，物理引擎无法直接 simulate（没法跑 collision、没法 attach 到 scalp、没法保持 length）。

**(B) Strand-based representations**（Neural Haircut [Sklyarova et al. ICCV 2023]、Neural Strands [Rosu et al. ECCV 2022]）：输出真正的 3D polylines，可以 simulate，但渲染 pipeline 不强，几何 supervisory signal 又受限于 noisy 的 2D orientation maps（用 Gabor filter 算出来的）。

Gaussian Haircut 的核心 insight 是：**把这两派 unified 起来**，用一个 **dual representation**——每根 strand 的每条 line segment 都绑一个 anisotropic 3D Gaussian。这样：

- Strand 结构作为**骨架**（simulation 需要）；
- Gaussian 作为**渲染/可微渲染**的载体（photometric supervision 需要）；
- Unstructured Gaussians 作为**第一阶段的 denoising + 3D lifting 工具**，生成 clean 的 multi-view pseudo-ground-truth，再去 supervise 第二阶段的 structured strands。

这个 idea 很 elegant，因为 3DGS 训练出的 Gaussians 在没有显式监督的情况下，**本身就倾向于 align 到 thin structures**（hair strands、grass blades 等），作者把这一 emergent property 当成免费的 3D orientation field。

---

## 3. 两阶段 pipeline 详解

### Stage 1: 3D Line Lifting with Unstructured Gaussians

**目标**：从 multi-view images + noisy 2D orientation maps，重建一个 unstructured Gaussian 场，使得每个 Gaussian 的 principal axis（covariance 最大方差方向）≈ 该位置 hair strand 的 3D 切线方向。

**输入**：multi-view images $\{I_v\}$、COLMAP SfM 出来的初始 cameras、segmentation masks、用 Gabor filter bank [Paris et al. SIGGRAPH 2004] 算出的 2D orientation maps $\hat{\beta}_p$（每像素一个 hair 切线方向 + confidence）。

**修改的 3DGS**：每个 Gaussian 的参数除了原始的 $\mu, s, q, o$ 之外，额外加了：

- 球谐系数 $f$（view-dependent color）；
- hair segmentation label $l$；
- **3D orientation confidence $\tau$**（这个是关键新增）；
- covariance matrix $\Sigma = R S S^T R^T$，其中 $R$ 是 quaternion $q$ 的旋转矩阵，$S = \text{diag}(s)$。

**渲染公式**（Eq. 1–4 in paper）：

颜色公式（标准 α-blending）：
$$C_p = \sum_{i=1}^{N} T_p^i \alpha_p^i c_i$$

其中：
- $p$ 是像素 index；
- $i$ 是按深度排序后的第 $i$ 个 Gaussian；
- $\alpha_p^i$ 是 Gaussian $i$ 在 pixel $p$ 处的 opacity（Eq. 2 的 exp 项）；
- $T_p^i = \prod_{j=1}^{i-1}(1-\alpha_p^j)$ 是 transmittance（前面所有 Gaussian 的"剩余透过率"）；
- $T_p^1 = 1$ 是 initial transmittance；
- $c_i$ 是 Gaussian $i$ 的 SH-rendered color。

α 公式（标准 2D Gaussian splatting）：
$$\alpha_p^i = o_i \exp\left(-\frac{1}{2}(p - \mu_i')^T \Sigma_i'^{-1} (p - \mu_i')\right)$$

- $o_i$：Gaussian $i$ 的 learnable opacity；
- $\mu_i'$、$\Sigma_i'$：Gaussian $i$ 的 mean 和 covariance 投影到 screen space 后的版本。

**Segmentation / confidence 渲染**（Eq. 3）：
$$l_p = \sum_i T_p^i \alpha_p^i l_i, \quad \tau_p = \sum_i T_p^i \alpha_p^i \tau_i, \quad s_p = \sum_i T_p^i \alpha_p^i$$

$s_p$ 是 rendered silhouette（mask）。

**Orientation 渲染**（Eq. 4）：
$$\beta_p = \sum_i T_p^i \alpha_p^i \beta_i$$

其中 $\beta_i$ 是 Gaussian $i$ 的**最大方差方向**（即 covariance matrix 的最大 eigenvalue 对应的 eigenvector）。这里有个 subtle point：direction 是 undirected line（没方向，只是 orientation，所以可能 ±π 等价）。这就跟 strand 不一样——strand 是 root-to-tip 的有向 polyline。

**Loss**（Eq. 5–6）：
$$\mathcal{L}_{\text{dir}} = \sum_p \tau_p \min\{d(\beta_p, \hat{\beta}_p),\, d(\beta_p, \hat{\beta}_p) \pm \pi\} - \log \tau_p$$

- $d(\cdot, \cdot)$：绝对角度差；
- $\hat{\beta}_p$：Gabor filter 算出的 ground truth orientation（注意是 undirected）；
- 第一项乘 $\tau_p$：confident 的像素权重大；
- 第二项 $-\log \tau_p$：entropy-like regularization，**避免网络把所有 $\tau_p$ 都压成 0 来 trivially minimize loss**——这是一个很重要的 trick（类似于 uncertainty-aware loss [Kendall & Gal 2017]）。

**Bundle adjustment**：作者把 BARF [Lin et al. ICCV 2021] 的 6-DoF learnable camera parameterization 叠加在 COLMAP 的初始估计上，作为 residual 一起 optimize。这是因为他们发现 SfM 在 hair-centric 场景里 camera localization 不够精确，会导致后续 strand fitting 时 photometric loss 失效。Camera 优化前 15k steps 做，之后 freeze。

Total loss：
$$\mathcal{L}_{\text{gaussian}} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{dir}} \mathcal{L}_{\text{dir}}$$

$\mathcal{L}_{\text{rgb}}$ 是 L1 + SSIM，$\mathcal{L}_{\text{seg}}$ 是 L1，$\lambda_{\text{seg}} = \lambda_{\text{dir}} = 0.1$。

**产出**：30,000 步训练后，得到 (a) 优化过的 cameras，(b) 一组 unstructured Gaussians，每个都对齐了一根 hair strand 的方向。把这些 Gaussians 渲染出来，得到**denoised 的 multi-view orientation maps + RGB + masks**，作为下一阶段的 supervision。

**Ablation 数据**：相比 Gabor filters 直接算的 2D orientation maps（average angular error 8°），3D line lifting 把 error 降到 7°。看似提升不大，但**关键在于 3D lifting**：unstructured Gaussians 已经把 2D noisy orientations 集成为 3D consistent field，下一步 strand fitting 不用再处理 view-inconsistency 问题。

---

### Stage 2: 3D Hair Strands Reconstruction (Coarse-to-Fine)

**Head model alignment**：先把 FLAME head model [Li et al. SIGGRAPH Asia 2017] fit 到 multi-view data，用 facial keypoints 做 multi-view optimization（这部分沿用 Neural Haircut 的 pipeline）。

**Strand parameterization**：头发表示为 **hair map** $H$，是 scalp region 的 texture map，每个 texel 存一根 3D strand polyline $S^k = \{p_l^k\}$。但 $H$ 自由度太高（10⁴ strands × ~100 control points × 3 coords = 几百万 DOF），直接 optimize 会 overfit，所以引入 **latent hair map** $Z$：

$$H = \mathcal{G}(Z), \quad Z = \mathcal{E}(H) \quad \text{(Eq. 7)}$$

- $\mathcal{G}$：预训练的 strand decoder（frozen），在 synthetic hair collection 上训练（沿用 Neural Strands [Rosu et al. 2022] 的设计）；
- $\mathcal{E}$：预训练的 encoder；
- $Z$ 是低维 latent texture map。

#### Coarse fitting: 优化 latent map $Z$

每 step 从 $Z$ decode 出 1,000 根 guiding strands $H'$（memory 限制），然后 interpolate 到 10,000 根 dense hair map $\hat{H}$ 再 rasterize。

**Strand-aligned Gaussians**（核心创新）：每根 strand 的每段 line segment $\{p_l^k, p_{l+1}^k\}$ 都绑一个 anisotropic Gaussian：
- Scale $s_l^k = \{\frac{1}{2}\|p_{l+1}^k - p_l^k\|_2, \epsilon, \epsilon\}$：**只有沿 strand 方向的 scale 可变**，另外两个固定为小常数 $\epsilon$，这样 Gaussian 看起来像一根细长的"胶囊"；
- Rotation：quaternion 把 x-axis 对齐到 strand 方向；
- Opacity $o_l^k = 1$（全 opaque）；
- Orientation confidence $\tau_l^k = 1$；
- Color：spherical harmonics $f_l^k$，由 trainable appearance decoder $\mathcal{G}_a$ 从 latent map 预测（架构同 $\mathcal{G}$ 但 **from scratch** 训练）。

**Trainable params**: $Z$ 和 $\mathcal{G}_a$。

**Strand direction extraction**：因为 stage 2 已经有 root-to-tip 方向，所以 $\beta_i$ 直接用 segment direction $v_l^k = p_{l+1}^k - p_l^k$，而不是 covariance 的 eigenvector。这是一个重要 simplification，让 supervision 更直接。

**Strand upsampling**：从 1,000 guiding strands 到 10,000 dense strands 的 interpolation，**在 3D coordinate 空间做**（不像 HAAR [Sklyarova et al. 2023] 在 latent space 做）。用 K-nearest neighbors（KNN）+ inverse-distance weighting。每根 strand 在自己 originating scalp face 的 TBN (tangent-bitangent-normal) basis 里定义，再做 blending。**没有这一步，rendered geometry 会有 holes**，photometric loss 失效（ablation Fig. 6 第四列验证）。

#### Fine fitting: 优化 explicit hair map $H$

不再用 latent space，**直接 optimize 30,000 strands 的 3D coordinates** $H$。SH coefficients $f_l^k$ 也直接 optimize。

#### Diffusion-based prior (SDS)

为了补**不可见的内层头发**，沿用 Neural Haircut 的思路：用预训练的 latent diffusion model [Sklyarova et al. HAAR / Neural Haircut prior]，通过 **Score Distillation Sampling** [Poole et al. ICLR 2023] 加约束。

Coarse 阶段：直接对 $Z'$（$Z$ 的 subsampled version）施加 SDS loss。

Fine 阶段：从 $H$ 随机抽 1,000 strands → interpolate 到 regular grid → 用 $\mathcal{E}$ encode 成 latent $Z'$ → 算 SDS loss。**这个 trick 让 fine 阶段也能享受 prior regularization**，因为 diffusion model 只在 latent space 训练。

#### Final objective (Eq. 8)

$$\mathcal{L}_{\text{strand}} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{seg}}\mathcal{L}_{\text{seg}} + \lambda_{\text{dir}}\mathcal{L}_{\text{dir}} + \lambda_{\text{sds}}\mathcal{L}_{\text{sds}}$$

- $\lambda_{\text{seg}} = \lambda_{\text{dir}} = 0.1$
- $\lambda_{\text{sds}} = 0.01$（小一点，避免 prior 过度 dominate photometric signal）

**总耗时**：~6 hours on RTX 4090，**比 Neural Haircut 快 10× 以上**（Neural Haircut 因为 mesh-based rasterization + NeuS layer 慢得多）。

---

## 4. 与相关工作的对比

### vs. 3D Gaussian Splatting [Kerbl et al. 2023]
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

3DGS 是 unstructured，输出**只是 visible surface 的 Gaussians**，没有内层 volume，没有 strand connectivity。Gaussian Haircut 借用了 3DGS 的可微渲染和 α-blending framework，但**每个 Gaussian 都被绑到 strand 上**，多了 strand 这一 layer of structure。

### vs. Neural Haircut [Sklyarova et al. ICCV 2023]
https://saicv.github.io/NeuralHaircut/

主要 baseline。Neural Haircut 也用 strand prior + 2D orientation maps，但 supervision 是 **mesh-based NeuS hair surface**，用 Chamfer distance 把 strands 拉到 surface 上，再做**有限的** photometric refinement（mesh rasterizer 无法 propagate 高频细节）。Gaussian Haircut：
1. 用 unstructured Gaussians 做 3D lifting（替代 NeuS surface）；
2. 用 strand-aligned Gaussians 做 differentiable rasterization（替代 mesh rasterizer）；
3. Camera 优化（Neural Haircut 没做）；
4. 10× speedup + 更好的 quality。

### vs. GaussianHair [Luo et al. 2024]
https://github.com/ProjectGaussianHair/GaussianHair

Concurrent work。也用 Gaussians 重建头发，但**需要 studio capture (uniform lighting)**，而 Gaussian Haircut 在 unconstrained lighting 下工作。GaussianHair 用 3D constraints 优化 strands（跟 Neural Haircut 类似），Gaussian Haircut 完全依赖新的 differentiable rendering scheme。

### vs. Relightable Gaussian Codec Avatars [Saito et al. 2023]
https://shunsukesaito.github.io/rgca/

Meta 的 codec avatars 用 3DGS 但**不把头发作为单独 layer**，整个 head 一起 Gaussians，没法单独 simulate 头发。

### vs. HAAR [Sklyarova et al. 2023]
https://arxiv.org/abs/2312.14066

Text-conditioned generative model of strand-based hairstyles。是本 paper 的 prior 来源之一。本 paper 借鉴了它的 strand upsampling 方法，但搬到 3D coordinate 空间。

### vs. HairStep [Zheng et al. CVPR 2023]
https://github.com/zhengyuf/HairStep

One-shot frontal image 输入。比较时 HairStep 因为只有一个 view，几何精度差很多（Fig. 11）。

---

## 5. Experiments 关键数据

### Synthetic evaluation
- 数据集：straight + curly hairstyles from Hair Meshes [Yuksel et al. 2009]
- 2D orientation map angular error：**Gaussian Haircut 7° vs Gabor 8°**（70 个 test views 平均）

### Real-world evaluation
- 数据：MPI-IS Capture Team 提供，3 FPS 抽帧，HyperIQA 选 best frame per 1/3s，Matte-Anything 出 segmentation，最终 128 训练 views
- 10× speedup vs Neural Haircut
- 渲染 quality 看起来 photorealistic（Fig. 4 test views）

### Ablation study (Fig. 6, 13, 14)
- **w/o fine optimization**：coarse only 几何粗糙，内层结构不真实；
- **w/o synthetic renders**（直接用 Gabor orientation maps）：poor illumination / dense strands 区域几何错误；
- **w/o strands upsampling**：fine fitting **不能收敛**（coarse 阶段渲染 holes 导致 photometric loss 无意义）；
- **w/o $\mathcal{L}_{\text{dir}}$**：orientation supervision 缺失，strands 方向乱；
- **w/o $\mathcal{L}_{\text{rgb}}$**：颜色不 match，geometry 也次优；
- **w/o $\mathcal{L}_{\text{sds}}$**：内层 hair prior 缺失，内部 volume 不真实。

### Applications
- Unreal Engine simulation (Fig. 5)：reconstructed strands 直接进 UE 做物理仿真，因为 strands 结构 + FLAME 头皮 attachment + realistic 内层；
- Photorealistic rendering in seconds。

---

## 6. Limitations

1. **Curly hairstyles**：跟 Neural Haircut 一样，diffusion prior 是 root-to-tip 设计，对 curly hair 表现差（Fig. 12）；
2. **Braids & complicated internal structures**：prior 没见过这些；
3. 还是依赖预训练的 strand decoder / diffusion prior，prior 训练数据集偏 straight + wavy。

---

## 7. 深入的 intuition building

### 为什么 dual representation 是 key insight？

传统的 strand reconstruction 困境：
- 如果**只用 strands**，可微渲染困难——polylines 是 1D objects，光栅化后只是几个 pixel-wide 的细线，gradient signal 难以 propagate 到 3D coordinates，特别是 internal strands 完全 invisible；
- 如果**只用 unstructured Gaussians**，渲染容易但没法做 physics simulation（no strand connectivity、no length constraint、没法 attach to scalp）。

**Dual representation 把"渲染友好"和"模拟友好"解耦**：每个 line segment 的 Gaussian 是"渲染代理"，但它的位置/旋转/scale 直接由 polyline segment 决定，所以 gradient 从 Gaussian 渲染端反向 flow 回 polyline vertices 时，路径是 well-defined 的。

公式细节：$s_l^k = \{\frac{1}{2}\|p_{l+1}^k - p_l^k\|_2, \epsilon, \epsilon\}$ 这个 scale 的设计很巧妙——**沿 strand 方向的 scale 跟着 segment length 走**，所以长 segments 自然有更大的 Gaussian footprint，short segments 细小。另外两个 scale 固定为 $\epsilon$（小值，比如 0.01），让 Gaussian 在 orthogonal 方向上极细，**避免相邻 strands 的 Gaussians 互相 bleed**，render 出来的"hairline"sharp。

### 为什么 camera refinement 重要？

Hair-centric capture 场景里 SfM 失效的原因：
1. 头发是 quasi-repetitive texture，feature matching 不稳定；
2. COLMAP 假设 static scene + Lambertian，但头发有强 view-dependent specular，会污染 triangulation；
3. 即使微小 camera error（几度），在 strand 级别 photometric loss 上会产生大 misalignment——因为单根 strand 直径 < 1mm，camera offset 5cm 对应 1° 在 1m 距离上就是 ~17mm 误差，远超 strand 直径。

BARF 的 6-DoF camera parameterization 让 camera 在前 15k steps 跟 Gaussians 一起 optimize，之后 freeze，避免后期 camera drift 把 strands 拉歪。

### SDS prior 在 fine stage 的 trick

Diffusion model 只在 latent space 训练（输入是 latent hair map $Z$）。Fine stage 优化的是 explicit hair map $H$，要应用 SDS 必须 round-trip：

$$H \xrightarrow{\mathcal{E}} Z \xrightarrow{\text{SDS}} \nabla \mathcal{L}_{\text{sds}}$$

每 step 随机抽 1,000 strands → KNN interpolate 到 regular grid $H'$ → $\mathcal{E}$ encode → $Z'$ → 算 SDS gradient → backprop 通过 encoder 回到 strands。

这里有个 subtle point：**只对 1,000 sampled strands backprop**，剩下 29,000 strands 这一 step 没有 SDS gradient。但因为 step 数足够多（10k steps fine），所有 strands 最终都接受过 prior regularization。这是 stochastic regularization 的常见 trick。

### Strand direction 的 representation 细节

Stage 1 用 covariance 的最大 eigenvector $\beta_i$（undirected line），Stage 2 用 segment vector $v_l^k$（directed root-to-tip）。这种 switch 很关键：

- Stage 1 的 ground truth 是 Gabor filter 算的 orientation，Gabor filter **本身没方向**（$\theta \equiv \theta + \pi$），所以 covariance-based undirected representation 自然 match；
- Stage 2 已经有 strand topology，知道 root 在哪里 tip 在哪里，用 directed vector 更 informative，loss 函数可以区分"反向 hair"。

公式 $\min\{d(\beta_p, \hat{\beta}_p), d(\beta_p, \hat{\beta}_p) \pm \pi\}$ 中的 $\pm \pi$ 处理就是 undirected case 的 wrap-around。

---

## 8. 可能的延伸方向 (个人联想)

1. **Hair dynamics capture**：现在 strands 是 static 的，如果能 capture video with hair motion，每 frame 重建 strand + temporal regularization（acceleration smoothness），就能得到 dynamic hairstyle；
2. **Differentiable physics layer**：把 strand dynamics simulation（比如 Fast LDL method [Daviet 2023]）嵌入 fine fitting，让 strands 跟 subject head motion 一起 solve 静态 equilibrium，更 physically plausible；
3. **Better prior**：当前 prior 对 curly/braid 差，可以用更多样化的 synthetic dataset 训练 prior，或者用 transformer-based strand prior 替代 diffusion prior（更易控制）；
4. **Grooming editing**：dual representation 让 edit 很自然——artist 拉一根 strand，所有绑定的 Gaussians 跟着动，render 立刻 feedback；
5. **Relightable hair**：现在 SH coefficients 是 per-scene trained，可以学一个 universal hair BRDF（类似 [Sadeghi et al. 2010] 的 hair shading model），让 hair relightable；
6. **Cross-subject prior transfer**：fine-tune strand decoder 到不同 hair type（Asian / African / Caucasian）的 specific prior；
7. **Real-time capture**：当前 6h on RTX 4090，可以用 SLAM-style incremental 3DGS + early strand fitting 逐步 reduce 时间；
8. **Multi-modal priors**：text-conditioned prior (HAAR) + image-conditioned prior (HairStep) 融合，做单 image 到 strand 的 reconstruction；
9. **Hair-body interaction**：扩展到 body hair、facial hair、animal fur——同样的 dual representation 应该 work；
10. **GAN-style discriminator prior**：替代 SDS，用 strand-GAN discriminator 做 realism constraint，可能比 SDS 更 stable。

---

## 9. Reference 链接汇总

- Project page: https://eth-ait.github.io/GaussianHaircut
- arXiv: https://arxiv.org/abs/2409.12978
- ECCV 2024 proceedings: https://link.springer.com/chapter/10.1007/978-3-031-72913-9_22
- 3DGS (基础): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Neural Haircut (baseline): https://saicv.github.io/NeuralHaircut/
- Neural Strands (prior 来源): https://ait.ethz.ch/projects/2022/neural-strands/
- HAAR (generative prior): https://arxiv.org/abs/2312.14066
- BARF (camera 优化): https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/
- GaussianHair (concurrent): https://github.com/ProjectGaussianHair/GaussianHair
- HairStep: https://github.com/zhengyuf/HairStep
- FLAME (head model): https://flame.is.tue.mpg.de/
- DreamFusion (SDS): https://dreamfusionpaper.github.io/
- Paris et al. 2004 (orientation maps): https://www-sop.inria.fr/reves/Basilic/2004/PSD04/
- Hair Meshes (synthetic benchmark): https://www.cs.ubc.ca/labs/imager/tr/2009/hair_meshes/
- COLMAP: https://colmap.github.io/
- Matte-Anything: https://github.com/SHI-Labs/Matte-Anything
- Unreal Engine: https://www.unrealengine.com/
- Blender (用于 synthetic rendering eval): https://www.blender.org/
- HyperIQA (frame selection): https://github.com/SSL92/HyperIQA

---

## 10. 总结

Gaussian Haircut 是一个 **representation engineering** 的精彩例子——不发明新 framework，但通过**重新组合 existing primitives**（3DGS + strand prior + diffusion SDS + bundle adjustment），把"渲染友好"和"模拟友好"这两个看似矛盾的目标 unified 在一个 dual representation 里。技术贡献清晰，三个核心：

1. **3D line lifting**：用修改版 3DGS 把 2D noisy orientation maps 提升到 3D consistent field，顺便 refine cameras；
2. **Strand-aligned Gaussians**：每段 line segment 绑一个 anisotropic Gaussian，实现 strand 的 differentiable rasterization；
3. **Coarse-to-fine strand fitting**：latent → explicit 两阶段优化，配合 SDS prior + photometric + geometric losses。

工程实现上很多细节（uncertainty-aware orientation loss、KNN interpolation in 3D、fine-stage SDS round-trip、camera freeze schedule）都是 ablation 验证过的关键 trick。

最终结果：state-of-the-art strand-based hair reconstruction + 10× speedup + directly simulatable in UE。这是把 neural rendering 的 research output 真正"deploy 到 production pipeline"的重要一步。
