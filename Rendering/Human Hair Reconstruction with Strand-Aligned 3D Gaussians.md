---
source_pdf: Human Hair Reconstruction with Strand-Aligned 3D Gaussians.pdf
paper_sha256: c13cfb90a585fc3fbe880145770a694ceeb9f63f5c95c667d34a03e88c4a2eef
processed_at: '2026-08-05T07:41:04-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Gaussian Haircut

## 一句话说清楚

这篇paper的核心想法其实很质朴：**先用3DGS把乱七八糟的2D方向信号"擦干净"提升到3D，再用这干净信号教一堆3D曲线怎么长**。

## 为什么需要这么折腾？

你想 reconstruct一个真实人的发型，最后要拿到Unreal Engine里能甩来甩去的physics simulation。那只能用 **strand**——就是一条条3D曲线，从scalp出发延伸到发梢，每条线有几百个顶点。3DGS重建出来的是一堆零散的椭球，覆盖在头发表面像一层壳，你想让它像真头发一样飘起来？没门，它没有内部结构，不是"一缕一缕"的，是"一片"的。

但问题是：直接从照片拟合strand很难。你照片里看到的只是一堆发丝的2D投影，发根发梢在3D空间哪边走完全看不见。更糟的是，传统方法用Gabor filter从照片里提取发丝2D方向——这个filter本身就noisy，经常提错方向，尤其在光照差或头发乱成一团的地方。

## 3DGS是个意外的好帮手

作者发现一个现象：你跑3DGS重建头的时候，那些Gaussian椭球会 **自动拉长对齐发丝方向**。因为发丝就是thin structure，3DGS要尽量少地用primitive拟合它，最经济的办法就是把Gaussian拉长沿发丝方向覆盖。

这就给了作者一个insight：3DGS的covariance matrix其实编码了发丝方向信息。最大variance方向就是发丝tangent方向。把2D的noisy Gabor方向图丢给3DGS让它拟合，3DGS会"自作主张"把它denoise——因为它得让多视角的方向都consistent，那些Gabor乱提的方向就被多视角约束给smooth掉了。

所以Stage 1本质上把3DGS当 **geometric filter** 用，把2D噪声提升成3D干净方向场。

## Strand-Aligned Gaussians 是真正的核心 trick

但Stage 1做完你拿到的还是unstructured Gaussians，一层壳，不能simulate。Stage 2要把这堆Gaussian的information"蒸馏"到真正的strand polylines里。

难点：你怎么让rendering loss的gradient流到strand顶点？strand本身是几何线段，直接rasterize它没法differentiable（mesh rasterization是hard assignment）。

作者的解法特别巧妙——**给每段线绑一个Gaussian，让Gaussian的scale"锁死"在线段长度上**。

具体说，strand上每两个相邻顶点 $p_l, p_{l+1}$ 之间，绑一个Gaussian：
- 这个Gaussian沿segment方向的scale = 线段长度的一半
- 另外两个方向scale = 极小值 $\epsilon$（让Gaussian变得像根细面条）
- 旋转让x轴对齐线段方向
- opacity强制为1

这样Gaussian就是"穿着线段外衣"的可微proxy。你render这个Gaussian得到image，算photometric loss，gradient顺着 $\partial s / \partial p$ 流回线段顶点。3DGS的可微rendering引擎被"借用"来supervise strand几何——你不用自己写differentiable strand rasterizer，借用现成的就行。

这是整个paper最聪明的地方。它告诉你：**3DGS不只能做重建，它的rendering engine可以嫁接到任何structured representation上，只要你能让Gaussian参数成为那个representation的函数**。

## Coarse-to-Fine 为什么是两段

Coarse阶段：优化latent hair map $Z$（低维压缩表示），用pre-trained decoder $\mathcal{G}$ 解出guiding strands。好处是维度低、prior容易施加（直接在latent space跑diffusion SDS）。坏处是decoder太贵，每iter只能decode 1000条guiding strand，rendering出来有"窟窿"，所以做了KNN插值补到10000条。

Fine阶段：直接优化explicit hair map $H$ 的3D坐标，30000条strand全部explicit。这时候没有decoder bottleneck，可以fully dense render，photometric loss能传fine detail。但prior怎么施加？每iter随机sample 1000条explicit strand，用encoder $\mathcal{E}$ 编回latent，跑SDS。这trick让你在explicit空间优化还能借latent prior的力。

两阶段本质上是 **先在压缩空间warm start拿到大致shape，再在原始空间精修细节**。这种coarse-to-fine pattern在neural fields里到处都是（NeRF的coarse-fine sampling，Instant-NGP的hash resolution progressive），这里换个形式应用在strand上。

## 几个值得品味的细节

**Orientation loss里的 $\pm\pi$ symmetry**：发丝是没有"箭头"的，一条线你可以从根到梢也可以从梢到根。Gabor filter给你的是无方向线，3DGS给你的也是。所以loss要允许 $\beta$ 和 $\hat{\beta}$ 反向。这个细节看似trivial，但absent的话整个pipeline就崩了。

**Confidence $\tau$ 的设计**：作者给每个Gaussian额外学一个confidence值，rendering时按 $\tau$ 加权orientation loss，再加个 $-\log\tau$ entropy regularizer防止 $\tau$ 退化到0逃避loss。这相当于让网络自己学"这个区域我对方向估计有多自信"，自信的地方多supervise，不自信的地方少supervise。这种learned per-pixel weighting在robust loss里很常见（比如student-t noise model），这里用在direction supervision上很合适。

**BARF camera refinement**：COLMAP在hair-centric scene上camera localization不准（hair是ambiguous texture）。作者用BARF的6-DoF residual跟3DGS joint优化15000步。这一步是real-world泛化的关键，没有它unconstrained capture场景基本跑不通。

**KNN interpolation in 3D space而不是latent space**：HAAR原版在latent space插值，但latent space不一定smooth。这里改成在3D坐标空间插值，保证几何连续性。这是个小改动但避开了latent space的non-linearity坑。

## 整个故事的高层insight

Representation matters more than algorithm。你用unstructured 3DGS，再牛的优化也只得到一层shell；你用strands from 2D supervision，再好的prior也救不了noisy 2D信号。作者的核心contribution是发现 **3DGS可以当representation bridge**——它既能做differentiable rendering（output端），又能infer geometric field（intermediate端），还能把这两端通过dual representation耦合到structured target上。

类似思路你能在别的地方看到：
- NeRF用density field作为SDF的可微proxy（NeuS）
- PIFu用pixel-aligned implicit function替代直接mesh fitting
- DIB-R用differentiable rasterizer把mesh参数wrap到可微渲染

这篇paper把这个pattern用在了hair这个特定domain上，效果惊艳。本质上是在differentiable rendering和structured geometry之间找到了一个"sweet spot representation"——strand-aligned Gaussians。它既是Gaussian（可微rendering友好）又是strand（physics simulation友好），是两种representation的intersection。

## 实际跑下来什么样

- 输入：128张multi-view照片（从视频里挑quality最好的帧）
- Stage 1：30000步3DGS训练，前15000步joint优化camera
- Stage 2 coarse：20000步，优化latent map
- Stage 2 fine：10000步，优化explicit strands
- 总耗时：~6小时 on RTX 4090
- 比Neural Haircut快10倍以上，quality还更好

## 失败case和局限

- Curly hair：prior是root-to-tip设计，curls的topology不匹配，效果差
- 编发辫这种internal topology：完全没法model
- 6小时还是偏慢，离real-time还远
- 只做static，没extend到dynamic sequence

总之这篇paper给我最大的启发是：**有时候最好的differentiable algorithm是找一个proxy representation，让它的参数成为你target representation的可微函数**。3DGS在hair这个任务上恰好是个perfect proxy——covariance天然编码方向，rendering天然可微，scale能rigidly耦合到几何长度。这种proxy-finding的思路在很多问题里都值得试。

参考资料：
- [Gaussian Haircut Project Page](https://eth-ait.github.io/GaussianHaircut)
- [3D Gaussian Splatting原始paper](https://repo.sammons.io/blogs/3dgs/3d-gaussian-splatting-for-real-time-radiance-field-rendering)
- [Neural Haircut基线](https://arxiv.org/abs/2312.06604)
- [Neural Strands prior](https://justusthies.github.io/projects/neural_strands/)
- [BARF camera refinement](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/)

---

# Gaussian Haircut: Strand-Aligned 3D Gaussians for Human Hair Reconstruction

## 1. Motivation与核心Insight

Andrej你好，这篇paper解决一个长期存在的tension：在digital humans领域，3DGS ([Kerbl et al. 2023](https://research.nvidia.com/labs/innsbruck/3dgaussian-splatting/)) 这类unstructured primitives能实现photorealistic rendering，但是physical simulation需要 **strand-based** representation（即3D polylines连接scalp到tip）。传统method如Neural Haircut ([Sklyarova et al. ICCV 2023](https://arxiv.org/abs/2312.06604)) 依赖Gabor filter banks从2D orientation maps拟合strands，但2D orientation maps本身带噪且无法约束occluded interior geometry。

Gaussian Haircut的关键insight是：**3DGS天然会在thin structures上自对齐**——观察3DGS重建后的Gaussian covariance，最大variance方向tend to align with hair strand tangent。这给了我们一个denoising机会：用3DGS做 "geometric filter" 把noisy 2D Gabor orientations提升到3D space，再用这cleaned 3D orientation field去supervise strand fitting。

## 2. 整体架构：Two-Stage Pipeline

### Stage 1 — 3D Line Lifting with Unstructured Gaussians
Input：multi-view RGB images + COLMAP初始camera + Matte-Anything segmentation masks + Gabor 2D orientation maps。

训练一个修改版3DGS（30,000 steps，前15,000 steps joint优化camera via BARF residual）。

### Stage 2 — Coarse-to-Fine Strand Fitting  
用Stage 1得到的Gaussian renders作为pseudo-ground-truth，监督explicit hair strands。
- Coarse step（20,000 steps）：优化latent hair map $Z$
- Fine step（10,000 steps）：优化explicit hair map $H$（30,000 strands）

总耗时 ~6 hours on RTX 4090，相比Neural Haircut的~60+ hours有10×+ speedup。

## 3. Stage 1 技术细节

### 3.1 Modified Gaussian参数化
每个primitive包含：
- Mean $\mu \in \mathbb{R}^3$
- Scaling $s \in \mathbb{R}^3$
- Rotation quaternion $q$
- Opacity $o$
- Spherical harmonics coefficients $f$ (view-dependent color)
- Hair segmentation label $l \in \{0,1\}$
- 3D orientation confidence $\tau > 0$ (新引入)

Covariance standard:
$$\Sigma = R S S^T R^T$$
其中 $R$ 是 $q$ 的matrix form，$S = \text{diag}(s)$。

### 3.2 关键trick：从covariance提取strand方向
对每个Gaussian，取**最大eigenvalue对应的eigenvector**作为strand tangent $\beta_i \in \mathbb{R}^3$（无方向性unit vector，因此存在 $\pm$ ambiguity）。Orthogonal directions的variance衡量不确定性——若Gaussian是anisotropic且well-aligned with strand，orthogonal variance小，confidence高。

### 3.3 Differentiable Rendering Equations

**Eq 1 (alpha-blending):**
$$C_p = \sum_{i=1}^{N} T_p^i \alpha_p^i c_i, \quad T_p^i = \prod_{j=1}^{i-1}(1-\alpha_p^j), \quad T_p^1 = 1$$

- $p$: pixel index
- $i$: sorted Gaussian index (按depth排序)
- $N$: 有效Gaussian数
- $C_p$: rendered feature (color / label / confidence / direction)
- $T_p^i$: **transmittance**，前 $i-1$ 个Gaussian累积透射率（=未遮挡概率）
- $\alpha_p^i$: i-th Gaussian在pixel p处的alpha值
- $c_i$: i-th Gaussian的feature

**Eq 2 (per-Gaussian alpha):**
$$\alpha_p^i = o_i \exp\left(-\frac{1}{2}(p-\mu_i')^T \Sigma_i'^{-1} (p-\mu_i')\right)$$

- $\mu_i'$, $\Sigma_i'$: 投影到screen space的mean和covariance
- 注意这里2D Gaussian是各向异性的，long axis沿strand方向

**Eq 3 (extra channels):**
$$l_p = \sum_i T_p^i \alpha_p^i l_i, \quad \tau_p = \sum_i T_p^i \alpha_p^i \tau_i, \quad s_p = \sum_i T_p^i \alpha_p^i$$

这里 $s_p$ 是rendered silhouette（用于seg loss），$l_p$ 是hair/face label，$\tau_p$ 是rendered orientation confidence。

**Eq 4 (direction rendering):**
$$\beta_p = \sum_{i=1}^{N} T_p^i \alpha_p^i \beta_i$$

注意：直接对unit vectors做alpha-blending有 $\pm$ ambiguity问题——如果两个well-aligned strands方向相反会"cancel out"。这是Eq 5中引入 $\pm\pi$ symmetry的原因。

### 3.4 Orientation Loss (Eq 5)

$$\mathcal{L}_{dir} = \sum_p \tau_p \min\{d(\beta_p, \hat{\beta}_p), d(\beta_p, \hat{\beta}_p) \pm \pi\} - \log \tau_p$$

- $d(\cdot, \cdot)$: absolute angular difference
- $\hat{\beta}_p$: ground-truth Gabor orientation at pixel $p$
- $\tau_p$: rendered confidence (acts as learned per-pixel weight)
- $\min\{\cdot, \cdot \pm \pi\}$: handles strand direction ambiguity（strand线方向无箭头）
- $-\log \tau_p$: regularizer防止 $\tau$ 学成0以逃避loss；类似unobserved categorical latent的entropy bonus

这个confidence-weighted formulation非常聪明：在Gaussian还没well-align的区域，$\tau$低，少supervise；aligned后$\tau$自然升高。

### 3.5 Total Objective (Eq 6)

$$\mathcal{L}_{gaussian} = \mathcal{L}_{rgb} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{dir}\mathcal{L}_{dir}$$

- $\mathcal{L}_{rgb}$: L1 + SSIM
- $\mathcal{L}_{seg}$: L1 between rendered silhouette/label和Matte-Anything GT mask
- $\lambda_{seg} = \lambda_{dir} = 0.1$

### 3.6 BARF Camera Refinement
直接用COLMAP的camera参数在hair-centric scenes不够accurate。引入6-DoF camera residual $\Delta\xi = (\Delta R, \Delta t)$ from [BARF (Lin et al. ICCV 2021)](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/) 与3DGS joint优化15,000 steps。Learning rate schedule跟Gaussian mean和rotation一致。这是个简单但critical的细节——本方法的real-world capture泛化性部分来自这里。

## 4. Stage 2 技术细节：Strand-Aligned Dual Representation

### 4.1 Hair Map Parameterization

 hairstyles parameterized on FLAME ([Li et al. SIGGRAPH Asia 2017](https://flame.is.tue.mpg.de/)) scalp区域。Hair map $H$ 的每个texel存储一条3D polyline：
$$S^k = \{p_l^k\}_{l=1}^{L}$$

- $k$: strand index
- $l$: vertex index along strand
- $p_l^k \in \mathbb{R}^3$: 第 $k$ 条strand的第 $l$ 个3D点（定义在scalp face的TBN basis下）

Latent representation:
$$H = \mathcal{G}(Z), \quad Z = \mathcal{E}(H) \quad \text{(Eq 7)}$$

- $\mathcal{G}$: pre-trained strand decoder ([Rosu et al. ECCV 2022](https://justusthies.github.io/projects/neural_strands/))，frozen
- $\mathcal{E}$: pre-trained encoder
- $Z$: latent hair map（低维，可优化）

### 4.2 Strand-Aligned Gaussians (核心contribution)

对每个segment $\{p_l^k, p_{l+1}^k\}$附加一个3D Gaussian，参数：
- **Scale**: $s_l^k = \left\{\frac{1}{2}\|p_{l+1}^k - p_l^k\|_2, \epsilon, \epsilon\right\}$
  - 第一维（沿segment方向）：segment长度的一半，所以Gaussian正好"包"住segment端点
  - 其他两维：固定small $\epsilon$，approximates infinitely thin strand
- **Rotation**: x-axis对齐segment方向 $v_l^k = p_{l+1}^k - p_l^k$
- **Opacity**: $o_l^k = 1$ (强制opaque)
- **Confidence**: $\tau_l^k = 1$ (这里geometry已经explicit，不需要learned confidence)
- **Color**: Spherical Harmonics coefficients $f_l^k$，可训练

注意关键点：**scaling不再独立可学**，它完全由polyline几何决定。这意味着rendering loss的gradient可以通过$\partial s / \partial p$ 直接传到strand顶点。这是dual representation的核心好处——3DGS的可微rendering引擎被"借用"来supervise strand geometry。

### 4.3 Coarse Stage的Guiding Strands + Upsampling

由于 $\mathcal{G}$ decoder昂贵，每batch只能decode 1,000 guiding strands $H'$，但dense rendering需要 ~10,000 strands。解决方案：**KNN interpolation in 3D coordinate space**（不是latent space，区别于HAAR [Sklyarova et al. 2023](https://arxiv.org/abs/2312.06604)）。

具体：
1. 对每个待插值strand origin，找到最近的4个guiding strands
2. 权重 $w \propto 1/d$ 其中 $d$ 是origin在texture coordinate空间的距离
3. 在每个guiding strand的TBN basis下做weighted blend
4. Final strand：在query origin的TBN下重建3D points

Why 3D space而不是latent space? 因为 $\mathcal{G}$ decoder产生的latent可能non-smooth，直接interpolate可能产生非realistic strands。3D coordinates直接interpolate保证几何连续性。

### 4.4 Diffusion-based Prior (SDS)

借用Neural Haircut的pre-trained diffusion model ([Sklyarova et al.](https://arxiv.org/abs/2312.06604))，用Score Distillation Sampling ([Poole et al. ICLR 2023](https://dreamfusion3d.github.io/)) regularize latent space：

$$\mathcal{L}_{sds} = \text{SDS}(Z', \epsilon_\phi)$$

- $Z'$: subsampled latent hair map（1,000 strands encoded from random subset of H）
- $\epsilon_\phi$: pre-trained diffusion U-Net预测的score
- $\lambda_{sds} = 10^{-2}$

Coarse step：直接对 $Z$ subsample得到 $Z'$。
Fine step：每iter随机sample 1,000 strands from $H$，用 $\mathcal{E}$ encode回latent，KNN interpolate到regular grid作为 $Z'$。这trick保证fine stage explicit optimization仍能受prior约束。

### 4.5 Final Objective (Eq 8)

$$\mathcal{L}_{strand} = \mathcal{L}_{rgb} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{dir}\mathcal{L}_{dir} + \lambda_{sds}\mathcal{L}_{sds}$$

注意 $\mathcal{L}_{dir}$ 在Stage 2中：
- $\beta_p$ 用 **directed** segment $v_l^k = p_{l+1}^k - p_l^k$（root-to-tip有方向）
- 仍保留 $\pm\pi$ symmetry因为strands本身无明确"前进方向"概念

### 4.6 Appearance Decoder $\mathcal{G}_a$
Coarse step中SH coefficients通过另一个decoder $\mathcal{G}_a$（架构同 $\mathcal{G}$ 但 **从scratch训练**，scene-specific）预测。Why不直接学 $f$？因为latent $Z$ dim有限，$f$ 是per-segment-per-vertex的，量大；用decoder可以共享statistical priors across strands。Fine step中直接optimize per-segment $f_l^k$（fully explicit）。

## 5. Postprocessing

训练后strands可能与FLAME mesh相交：
1. 计算每strand vertex到head mesh的signed distance
2. 找intersection segments，prune掉
3. 将每个pruned strand的"开始"连到最近scalp vertex

这步保证 **simulatable**——可在Unreal Engine中跑physics。

## 6. Experiments & Ablations

### 6.1 Quantitative on Synthetic
在两个synthetic hairstyles ([Yuksel et al. 2009 Hair Meshes](https://www.cs.utah.edu/~fishma/hairmeshes/))上：

| Method | Avg Angular Error (70 views) |
|---|---|
| Gabor filter (baseline) | 8° |
| **Ours (3D line lifting)** | **7°** |

1° improvement看起来小，但累计到strand fitting阶段对几何quality影响大。

### 6.2 Real-world Comparison vs Neural Haircut
- Reconstruction time: **>10× speedup**（6h vs 60h+）
- Quality: Fig 3-10显示更sharp的strand geometry，特别是illumination poor区域和high-density entangled区域
- Failure case: curly hairstyles（Fig 12）—— prior的root-to-tip设计inherently不适合curls

### 6.3 Ablation Study (Fig 6, 13, 14)
逐步remove components，观察quality degradation：

| Configuration | Effect |
|---|---|
| w/o fine optimization | coarse latent only，可见区域 geometry粗，细节missing |
| w/o synthetic renders (orientation maps from Stage 1) | high-density + poor lighting区域失败——证明3D line lifting的核心价值 |
| w/o strands upsampling | coarse stage渲染有holes，fine stage无法收敛 |
| w/o $\mathcal{L}_{dir}$ | strand orientation乱，偏离Gabor GT |
| w/o $\mathcal{L}_{rgb}$ | 颜色不对，几何也略损（photometric loss确实传gradient到geometry） |
| w/o $\mathcal{L}_{sds}$ | interior structure不realistic，prior无效 |

### 6.4 Applications
- 直接import到Unreal Engine ([Epic Games](https://www.unrealengine.com)) 做physics simulation（Fig 5）
- Test-view rendering (Fig 4) photorealistic，无manual lighting adjustment需要

## 7. 与Concurrent Work的对比

Luo et al.的 [GaussianHair (arXiv 2024)](https://arxiv.org/abs/2402.10483)是concurrent work，也用3D Gaussians做hair strands，但：
- GaussianHair需要studio capture with uniform lighting
- 本方法支持unconstrained capture（through BARF camera refinement）
- GaussianHair用3D constraints优化strands（类似Neural Haircut），本方法fully依赖differentiable rendering

## 8. 我的Intuition总结

如果你（Andrej）从ML的视角看，这paper的deep insight是 **representation alignment via differentiable proxy**：

1. **Target representation**（strand polylines）不能直接differentiably render到2D with高quality
2. **Proxy representation**（3DGS）能differentiably render，且能infer the needed geometric field（orientations）
3. **Bridge**：让3DGS的covariance matrix编码strand tangent direction，再用这个cleaned 3D field作为GT去supervise strand fitting
4. **Dual representation trick**：让3DGS的scaling "rigidly coupled"到polyline segment length，借用3DGS rendering gradient反传到polyline vertices——这是把non-differentiable mesh rasterization的"功能"嫁接到可微3DGS上

更深层看，这是 **representation bottleneck matters**: 如果直接学unstructured 3DGS，得到的"hair"只是一层shell，不能simulate；如果直接学strands from 2D supervision，没3D denoising过程，优化landscape极差。两阶段decoupling让每个stage都解决一个明确的subproblem。

类似思路在其它领域也出现：
- PIFu ([Saito et al. ICCV 2019](https://shunsukesaito.github.io/PIFu/)) 用pixel-aligned implicit function代替direct mesh fitting
- NeuS ([Wang et al. NeurIPS 2021](https://arxiv.org/abs/2106.10689)) 用density field作为SDF的可微proxy

## 9. Limitations & Future Directions

- Curly hair prior weak（root-to-tip design不合curls的topology）
- Braids等复杂internal topology无法model
- Single subject，没showcase multi-identity priors
- 6h仍偏慢，可能用更efficient decoder (e.g., MLP-based G instead of transformer-based) 或distillation能加速
- 没combine dynamic hair sequences（仅static reconstruction）；可结合 [NeuWigs (Wang et al. CVPR 2023)](https://neuwigs.github.io/) 的dynamic priors

## Reference Links

- [Gaussian Haircut Project Page (ETH AIT)](https://eth-ait.github.io/GaussianHaircut)
- [3D Gaussian Splatting (Kerbl et al. SIGGRAPH 2023)](https://repo.sammons.io/blogs/3dgs/3d-gaussian-splatting-for-real-time-radiance-field-rendering)
- [Neural Haircut (Sklyarova et al. ICCV 2023)](https://arxiv.org/abs/2312.06604)
- [Neural Strands (Rosu et al. ECCV 2022)](https://justusthies.github.io/projects/neural_strands/)
- [HAAR (Sklyarova et al. 2023)](https://arxiv.org/abs/2312.06604)
- [BARF (Lin et al. ICCV 2021)](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/)
- [FLAME head model](https://flame.is.tue.mpg.de/)
- [DreamFusion / SDS (Poole et al.)](https://dreamfusion3d.github.io/)
- [GaussianHair concurrent work](https://arxiv.org/abs/2402.10483)
- [Hair Meshes dataset (Yuksel et al.)](https://www.cs.utah.edu/~fishma/hairmeshes/)
- [Matte Anything segmentation](https://arxiv.org/abs/2306.04121)
- [COLMAP SfM](https://colmap.github.io/)
- [Unreal Engine](https://www.unrealengine.com)

---

Summary一句话：Gaussian Haircut把3DGS从"渲染surface"重新定位为"几何先验蒸馏器"——先用它把noisy 2D orientations提升到clean 3D，再用其differentiable rendering engine通过strand-aligned dual representation监督explicit strand polylines。是representation design + differentiable simulation的优雅例子。
