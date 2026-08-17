---
source_pdf: SKYFALL-GS.pdf
paper_sha256: d6b7807158244bfebdf790e173117a174de3c5bb043a342ee826f217c95b9333
processed_at: '2026-08-12T07:40:52-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：Skyfall-GS

## 一句话概括

**拿几张卫星照片，变出一个能飞进去逛的3D城市。**

## 为什么这事难

你想想，卫星照片是从天上往下看的，对吧？那建筑物长什么样，卫星只能看到屋顶，看不到墙面。就像你只看到一堆盒子的顶部，侧面全是黑的。

而且卫星虽然能拍好几张照片，但拍摄角度差不了太多——毕竟都在天上，离地面几百公里。这点parallax（视差）根本不够重建出像样的3D。

所以你直接拿卫星照片做3D重建，结果就是：屋顶还行，一到低角度想看building facade，全是糊的、扭曲的、飘着的artifact。

## 那别人怎么做的

有一类方法比如CityDreamer、GaussianCity，它们走的是另一条路——给一个semantic map（哪里是楼、哪里是路）加上height field（每个地方多高），然后train一个generative model生成城市。

问题是这些方法：
- 需要你提前准备好semantic map和height map，麻烦
- 只在特定数据集上train，换个城市可能就拉了
- 生成的楼太简化，长得都差不多，不realistic

## Skyfall-GS的核心idea

关键insight：**卫星照片已经给了你真实的coarse geometry和texture，只是缺了facades和低角度细节。那缺的部分让diffusion model去"脑补"就好了。**

Diffusion model（这里用的是FLUX）见过几十亿张自然照片，它知道建筑墙面长什么样、阴影该怎么打、texture该多丰富。把它当做一个超级强大的"图像修复师"。

## 具体怎么干

整个pipeline就两步：

### 第一步：先把卫星照片拧成一个3DGS

你有多张不同日期拍的卫星照片，先重建一个initial 3D Gaussian Splatting scene。

这里有几个坑要处理：

**坑1：不同日期照片光照差很多。** 今天拍的晴空万里，上周拍的阴天，下个月拍的雪景。你得让模型知道"这些光照变化不是scene本身的变化"，否则它会被搞晕。解决办法是给每张照片一个embedding，告诉模型"这是第j天的光照条件"，然后学一个color transform把光照normalize掉。

**坑2：很多"浮游Gaussian"飘在空中。** 因为视差不够，很多Gaussian没法被约束到正确surface上，就飘着，opacity很低。解决办法是加一个entropy loss，逼着每个Gaussian要么"明确存在"（opacity→1），要么"明确消失"（opacity→0）。这样floater就死掉了，geometry更干净。

**坑3：geometry还是不够完整。** 比如屋顶和路面这些flat区域，Gaussian容易乱长。解决办法是在地面附近放一些"虚拟相机"，从这些角度render一下，用monocular depth estimator（MoGe）估个depth，然后约束3DGS的depth跟这个估计的depth结构一致。注意是Pearson correlation——因为monocular depth只有相对结构，绝对尺度没意义。

这一步出来，你有一个粗糙但基本对的3D城市，但低角度render还是糊的。

### 第二步：从天上往地面"课程学习"式refine

这是最clever的部分。

**Observation**：你从satellite-trained 3DGS render不同elevation的角度会发现——高角度（接近top-down）render还行，因为satellite照片本来就是高角度拍的，这是"分布内"。但elevation越低（越接近水平），render越烂，因为这是"分布外"。

**Curriculum idea**：那就别一上来就挑战低角度。先从高elevation开始refine，这时候3DGS render还行，diffusion model只要稍微修一下artifact。然后慢慢降低elevation，让diffusion model逐步"补"出越来越多的facade细节。就像教小孩，先做简单题再做难题。

**Refine用什么**：FlowEdit + FLUX.1。FlowEdit是个inversion-free的editing方法，你给它一个source prompt描述"这张图有blur和warping artifact"，再给一个target prompt说"我要sharp building、smooth edge、natural lighting"，它就把图给refine了。

**但有个问题**：每个view独立做diffusion edit，各view之间会inconsistent。因为2D diffusion不知道3D consistency，每个view它都"自由发挥"一下，结果拼起来几何对不上。

**解决方案**：每个view不要只sample一次，sample N_s=2次。然后训练3DGS的时候，photometric loss会隐式地average over这些samples。这样3DGS就找到一个"共识"——既贴合每个sample的style，又保持geometric coherence。有点像让多个人投票，取平均意见，比单一个人的意见robust。

**Iterative循环**：
1. 当前3DGS → render一些views
2. 每个view用FlowEdit refine一下
3. 把refined images加进训练集
4. 重新train 3DGS
5. 下一个episode，elevation降一点
6. 重复5个episode

**关键细节**：训练时75%用refined images，25%用原始satellite images。这保证了最后的结果既漂亮又忠实于输入——不会跑偏成另一个城市。

## 效果怎么样

**FID_CLIP**（衡量生成质量的核心指标）：
- DFC2019数据集：Sat-NeRF 88 → ours 27，**3倍提升**
- GoogleEarth数据集：CityDreamer 36 → ours 10，**3.6倍提升**

**User study**：89个人盲测，我们赢率97%（vs Sat-NeRF的3%）

**速度**：rendering时11 FPS on T4（便宜卡），CityDreamer 0.18 FPS on A100（贵卡还慢50倍）。因为3DGS是rasterization，render时不需要跑neural network。

## 为什么这工作有意思

我觉得它好在几个地方：

1. **Input太便宜了**：就几张satellite照片，Google Earth全世界都能下载。不需要semantic map、不需要height field、不需要3D scan、不需要street-level照片。

2. **Leverage foundation model**：用pretrained FLUX的prior，避免了domain-specific training的数据瓶颈。FLUX见过几亿张图，它知道building该长什么样。

3. **每个design choice都解决一个具体问题**：
   - 光照变化 → appearance modeling
   - Floater → entropy opacity reg
   - Geometry不完整 → pseudo depth supervision
   - 低角度degenerate → curriculum
   - Multi-view不一致 → multiple samples
   - 跑偏 → 25% original satellite混合

4. **Curriculum idea很elegant**：不是简单粗暴地sample所有角度，而是利用了"satellite view天然是高elevation"这个prior，从easy到hard逐步过渡。

## Limitations

Paper自己说：
- 跑refinement要6小时，有点慢
- 极端street-level（真的贴地平线）还是太smooth
- 没做dynamic scene（车、人）

我自己想到的：
- Diffusion refine每张图是独立的，如果能用3D-aware diffusion（比如MVDream那种multi-view consistent的）会更好
- FlowEdit的prompt是人写的，如果能自动从satellite image分析出该refine什么会更好
- 目前只测了两个数据集，不知道在复杂地形（山区、城中村）表现如何

## 一句话总结intuition

**卫星照片给你"真"的coarse shell，diffusion model给你"假"但plausible的细节，curriculum让两者优雅地merge起来。**

这就像你有个毛坯房（satellite reconstruction），请了个室内设计师（diffusion model）来装修，但你规定他必须保留原始结构（25% original satellite混合），而且分阶段装修——先装修容易的房间再装修难的（curriculum），最后得到一个又像原房又漂亮的成品。

---

# Skyfall-GS：从卫星图像合成沉浸式3D城市场景

这篇paper解决一个非常有意思的问题：**如何仅从多视角satellite imagery合成可自由飞行的、photorealistic的3D城市场景**。让我从intuition开始，逐层拆解。

## 1. 核心问题与动机

### 1.1 为什么satellite imagery是好的输入？

Satellite imagery有独特优势：覆盖广、自动化采集、高分辨率。比如Maxar WorldView-3每天能capture约680,000 km²的图像，分辨率达到31cm/pixel。这意味着我们有一个realistic的coarse geometry和texture source，且scalable。

### 1.2 为什么直接用3DGS重建satellite imagery不够？

关键问题在于**parallax不足**。Satellite imagery虽然有multi-view，但viewpoint差异有限（视角小、距离远），导致：
- Building facades几乎invisible
- 重建出来的geometry充满floating artifacts
- 低视角渲染时严重模糊和distortion

Paper中Figure 2(a)展示了Sat-NeRF和naive 3DGS的失败case。

### 1.3 为什么现有city generation方法也不够？

像CityDreamer (Xie et al., 2024)和GaussianCity (Xie et al., 2025b)这类方法：
- 依赖semantic maps和height fields作为输入（强假设）
- 在small-scale、domain-specific datasets上overfit
- 输出的building geometry过于简化，texture不真实
- 无法处理tunnels、bridges、multi-level architecture

Paper的insight：**用open-domain diffusion model作为external information source**，利用其zero-shot generalization和diversity来补充missing信息。

## 2. 方法Pipeline概述

整个pipeline分两个stage，如Figure 3所示：

### Stage 1: Reconstruction Stage
从multi-view satellite imagery重建initial 3DGS，包含：
- Appearance modeling处理multi-date illumination变化
- Opacity regularization消除floaters
- Pseudo-camera depth supervision增强geometry

### Stage 2: Synthesis Stage  
用curriculum-based Iterative Dataset Update (IDU)：
- 从高elevation到低elevation逐步refine views
- 每个view用FlowEdit + FLUX.1 [dev] diffusion model refine
- Multiple samples per view避免single trajectory偏差

## 3. 3DGS Preliminary

3D Gaussian Splatting (3DGS) (Kerbl et al., 2023)用一组Gaussians表示scene。每个Gaussian参数化为：

- **μ_i**: Gaussian center（3D位置）
- **Σ_i**: covariance matrix（形状和方向）
- **α_i**: opacity（不透明度）
- **view-dependent color**: 通常用spherical harmonics (SH)表示

### 投影公式

每个Gaussian投影到image plane的covariance：

$$\Sigma_i'^{\bot} = J W \Sigma_i W^T J^T$$

这里：
- **W**: viewing transformation（世界坐标到相机坐标）
- **J**: affine-projection Jacobian（相机坐标到image plane的局部线性化）
- **Σ_i**: 3D空间中的covariance
- **Σ_i'^⊥**: 2D image plane上的covariance
- **上标⊥**: 表示projected（投影后）

### Alpha compositing

Pixel的颜色由front-to-back的alpha compositing得到：

$$C = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j<i} (1 - \alpha_j)$$

### 基础loss

$$\mathcal{L}_{\mathrm{color}} = \lambda_{\mathrm{D-SSIM}} \mathrm{DSSIM}(\hat{C}, C) + (1 - \lambda_{\mathrm{D-SSIM}}) \|\hat{C} - C\|_1$$

变量：
- **λ_D-SSIM**: DSSIM项的权重（paper中=0.2）
- **Ĉ**: rendered color
- **C**: ground truth color
- **DSSIM**: structural dissimilarity（SSIM的补）
- **||·||_1**: L1 norm

Reference: [3D Gaussian Splatting paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## 4. Stage 1: Initial 3DGS Reconstruction

### 4.1 相机参数近似

Satellite imagery通常用Rational Polynomial Camera (RPC) model，直接mapping image coordinates到geographic coordinates。但3DGS pipeline需要perspective camera parameters。

解决方案：用SatelliteSfM (Zhang et al., 2019)从RPC approximate出extrinsic和intrinsic，并生成sparse SfM points作为initial 3DGS points。

Reference: [SatelliteSfM code](https://github.com/zhangkai410/SatelliteSfM)

### 4.2 Appearance Modeling

**Intuition**: Multi-date satellite imagery会有巨大的illumination变化——不同日期、不同季节、不同时间，甚至transient objects（车、云等）。如果直接训练，3DGS会被这些变化confuse。

**方法**（来自WildGaussians (Kulhanek et al., 2024)的思路）：

1. **Per-image embeddings {e_j}_{j=1}^N**: 每张训练图像有一个可训练的embedding，捕获global illumination条件
2. **Per-Gaussian embeddings g_i**: 每个Gaussian有一个embedding，捕获localized appearance变化（如特定区域的阴影）
3. **Appearance MLP f**: 一个轻量MLP，输入三个量，输出affine color transformation

公式：

$$(\beta, \gamma) = f(e_j, g_i, \bar{c}_i)$$

变量：
- **e_j**: 第j张图像的embedding（dimension=32）
- **g_i**: 第i个Gaussian的embedding（dimension=24）  
- **c̄_i**: 第i个Gaussian的0-th order SH（即base color）
- **f**: 2层hidden layer、每层128 neurons、ReLU的MLP
- **β**: color shift（bias）
- **γ**: color scale

变换后的color：

$$\tilde{c}_i(\mathbf{r}) = \gamma \cdot \hat{c}_i(\mathbf{r}) + \beta$$

- **ĉ_i(r)**: 原始view-dependent color（r是viewing direction）
- **c̃_i(r)**: appearance-transformed color

**关键设计**: SH限制为0阶和1阶，**防止把appearance变化建模为view-dependent effect**。这个设计很subtle——如果SH阶数太高，模型会把illumination变化编码到view-dependent color里，导致rendering时geometry和appearance混淆。

Reference: [WildGaussians](https://arxiv.org/abs/2406.12273)

### 4.3 Opacity Regularization

**Intuition**: 从sparse satellite views重建的3DGS会有很多floater——那些低opacity的Gaussian漂浮在scene中，因为没有足够的view constraint把它们push到正确surface上。

**方法**: Entropy-based opacity regularization，push opacity分布向binary（0或1）：

$$\mathcal{L}_{\mathrm{op}} = -\sum_i \alpha_i \log(\alpha_i) + (1 - \alpha_i) \log(1 - \alpha_i)$$

变量：
- **α_i**: 第i个Gaussian的opacity
- **-α_i log(α_i)**: entropy term，鼓励opacity接近1（visible）
- **(1-α_i) log(1-α_i)**: 鼓励opacity接近0（invisible）

这个loss在α_i=0.5时最大（最penalized），在α_i=0或1时为0。效果是：
- 高opacity的Gaussian保持visible
- 低opacity的Gaussian被push到0，在densification时被prune
- 几何更adhere到实际surface

Paper中λ_op=10，权重很大。

### 4.4 Pseudo Camera Depth Supervision

**Intuition**: Satellite view的parallax太小，即使有opacity regularization，geometry仍然不完整。Paper的key insight：在更接近地面的位置放置pseudo-cameras，但这些cameras没有真实GT——怎么办？

**方法**：
1. 在地面附近sample pseudo-cameras
2. 从这些cameras render RGB image I_RGB和alpha-blended depth D̂_GS
3. 用off-the-shelf monocular depth estimator (MoGe, Wang et al., 2024a)从I_RGB预测scale-invariant depth D̂_est  
4. 用Pearson correlation作为监督信号

**为什么用Pearson correlation而不是L1/L2?** 因为monocular depth estimation是scale-invariant和shift-invariant的，绝对值没有意义，只有相对深度结构有意义。Pearson correlation恰好capture这种结构相似性。

公式：

$$\mathcal{L}_{\mathrm{depth}} = \|\mathrm{PCorr}(\hat{D}_{\mathrm{GS}}, \hat{D}_{\mathrm{est}})\|_1$$

$$\mathrm{PCorr}(\hat{D}_{\mathrm{GS}}, \hat{D}_{\mathrm{est}}) = \frac{\mathrm{Cov}(\hat{D}_{\mathrm{GS}}, \hat{D}_{\mathrm{est}})}{\sqrt{\mathrm{Var}(\hat{D}_{\mathrm{GS}}) \mathrm{Var}(\hat{D}_{\mathrm{est}})}}}$$

变量：
- **D̂_GS**: 3DGS rasterized depth
- **D̂_est**: MoGe预测的depth
- **Cov(·,·)**: covariance
- **Var(·)**: variance
- **PCorr**: Pearson correlation coefficient，范围[-1, 1]
- **||·||_1**: 取绝对值（因为深度结构相关即可，方向不重要）

Paper中取绝对值是因为可能需要负相关也接受？实际上depth map的符号取决于near/far表示，所以abs让loss更robust。

**Sampling策略**（Section A.1）：
- 24 views每10 iterations
- Look-at points: (x, y, z), 其中 x, y ~ N(0, 128), z=0
- Azimuth: 均匀sample [0, 2π]
- Elevation: 从80°降到45°
- Radius: 从300降到250 units

Reference: [MoGe](https://arxiv.org/abs/2410.19115)

### 4.5 Stage 1总loss

$$\mathcal{L}_{\mathrm{sat}}(G, C) = \mathcal{L}_{\mathrm{color}} + \lambda_{\mathrm{op}} \mathcal{L}_{\mathrm{op}} + \lambda_{\mathrm{depth}} \mathcal{L}_{\mathrm{depth}}$$

- **G**: 3DGS representation
- **C**: set of ground-truth satellite images
- **λ_op=10**, **λ_depth=0.5**

Training: 30,000 iterations，densification在1,000-21,000，约1小时on RTX A6000。

## 5. Stage 2: Curriculum-based IDU

这是paper的核心创新点。

### 5.1 Motivation: Curriculum Strategy

**Key observation**（Figure 4）：从satellite-trained 3DGS render时：
- 高elevation角度的render质量较好
- 低elevation角度的render严重degenerate

这个observation非常自然——satellite views本身就是高elevation，所以3DGS在分布内（高elevation）render好，分布外（低elevation）render差。

**Curriculum insight**: 既然如此，refinement应该从easy（高elevation）开始，逐步过渡到hard（低elevation）。这就像curriculum learning——先学简单的，再学复杂的。

### 5.2 Iterative Dataset Update (IDU)

IDU (来自Instruct-NeRF2NeRF (Haque et al., 2023)和IM-3D (Melas-Kyriazi et al., 2024))的核心循环：
1. Render views
2. Edit/refine renders with diffusion model
3. Update dataset with refined renders
4. Re-train 3DGS
5. 重复

**Skyfall-GS的改进**: 之前的方法从original training views或simple orbits sample camera poses，而Skyfall-GS用curriculum schedule——elevation从高到低。

Reference: [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/), [IM-3D](https://arxiv.org/abs/2403.02113)

### 5.3 Render Refinement with FlowEdit

**Intuition**: 初始3DGS的renders有blurry texture和artifacts。把这些renders当作diffusion process的noisy intermediate results，然后用diffusion model"denoise"它们——但这里的denoise其实是refine/enhance。

**方法**: Prompt-to-prompt editing (Hertz et al., 2022) + FlowEdit (Kulikov et al., 2024) + FLUX.1 [dev] (Black Forest Labs, 2024)

FlowEdit的核心idea：不需要DDIM inversion，直接用flow model从source image走到target image。

**Prompts设计**（Section A.1）：
- **Source prompt**: "Satellite image of an urban area with modern and older buildings, roads, green spaces. Some areas appear distorted, with blurring and warping artifacts."
- **Target prompt**: "Clear satellite image of an urban area with sharp buildings, smooth edges, natural lighting, and well-defined textures."

**Noise parameters**: n_min=4, n_max=10，balance artifact removal vs detail preservation。

Reference: [FlowEdit](https://github.com/UCSC-VLAA/FlowEdit), [FLUX.1](https://blackforestlabs.ai/)

### 5.4 Multiple Diffusion Samples

**问题**: 如果每个view只做一次diffusion refine，各view之间inconsistent——因为independent 2D denoising不preserve 3D consistency。

**数学视角**（paper的精彩论述）：
- 理想的optimal denoising分布：所有views同步，保持3D appearance
- 独立2D denoising：每个view的trajectory独立sample
- 结果：denoising trajectory分布是optimal trajectories的superset
- 从这个expanded分布中选单一trajectory，得到3D-consistent optimal结果的probability可忽略不计

**解决方案**: 对每个view生成N_s个独立refined samples（N_s=2 in paper）。Training时，photometric loss L_color隐式地average over这些samples。

**Intuition**: 这就像multi-view consensus——不是commit到单一可能的refinement，而是让3DGS optimization找到一个consensus representation，平衡fidelity to individual samples和geometric coherence across views。

这个想法让我联想到ProlificDreamer (Wang et al., 2023)的Variational Score Distillation——也是用分布而不是point estimate来避免mode collapse。

### 5.5 Algorithm 1 详解

```
Input: N_e episodes, N_v views/point, N_s samples/view, N_p look-at points
Input: {P_i} look-at points, {R_i} decreasing radius, {E_i} decreasing elevation
Input: T_src, T_tgt prompts, n_min, n_max FlowEdit params
Input: G initial 3DGS

G' ← G
for i = 1 to N_e:
    radius ← R_i
    elevation ← E_i
    cam_views ← OrbitViews({P_i}, radius, elevation, N_v)
    render_views ← Render(G', cam_views)
    refine_views ← FlowEditRefine(render_views, T_src, T_tgt, n_min, n_max, N_s)
    G' ← Train(G', refine_views)
return G'
```

**参数细节**:
- DFC2019: N_e=5, N_p=9 (3×3 grid), N_v=6, N_s=2
- Elevation: 85° → 45°
- Radius: 300 → 250 units (DFC2019), 600 fixed (GoogleEarth)
- 每episode 10,000 iterations，densification到9,000
- 75% refined images + 25% original satellite images（保持fidelity to input）

### 5.6 Stage 2 Loss

$$\mathcal{L}_{\mathrm{IDU}}(G_{i-1}, \tilde{C}_i) = \mathcal{L}_{\mathrm{color}} + \lambda_{\mathrm{depth}} \mathcal{L}_{\mathrm{depth}}$$

- **G_{i-1}**: previous episode的3DGS
- **C̃_i**: current refined images
- **注意**: 没有opacity regularization——curriculum本身通过multi-view consistency mitigate floaters，且variable opacity对semi-transparent structures（如玻璃幕墙）有益

Training: 约6小时on RTX A6000。

## 6. 实验结果分析

### 6.1 Datasets

1. **DFC2019** (Le Saux et al., 2019): WorldView-3, Jacksonville Florida, 2048×2048, 35cm/pixel
   - 4个AOI: JAX_004, JAX_068, JAX_214, JAX_260
   - 训练图像数: 9-21张（非常sparse！）

2. **GoogleEarth** (Xie et al., 2024): NYC scenes
   - 4个AOI: 004, 010, 219, 336
   - 每AOI 60张训练图像，80° elevation模拟satellite condition

### 6.2 Baselines

两类对比：
- **Satellite reconstruction**: Sat-NeRF, EOGS, Mip-Splatting, CoR-GS
- **City generation**: CityDreamer, GaussianCity

### 6.3 Metrics

**Distribution-based**（primary）:
- **FID_CLIP** (Kynkäänniemi et al., 2023): 用CLIP backbone的FID，比传统InceptionV3-based FID更适合modern generative models
- **CMMD** (Jayasumana et al., 2024): CLIP-based Maximum Mean Discrepancy

**Pixel-level**（secondary）:
- PSNR, SSIM, LPIPS

### 6.4 Table 1 & 2 分析

**DFC2019 (Table 1)**:
| Method | FID_CLIP↓ | CMMD↓ | PSNR↑ |
|--------|-----------|-------|-------|
| Sat-NeRF | 88.36 | 4.868 | 10.05 |
| EOGS | 87.74 | 5.286 | 7.26 |
| Mip-Splatting | 87.19 | 5.405 | 11.89 |
| CoR-GS | 89.03 | 5.241 | 11.55 |
| **Ours** | **27.35** | **2.086** | 12.38 |

FID_CLIP从~88降到27.35，**3倍以上提升**！这是huge gap。

**GoogleEarth (Table 2)**:
| Method | FID_CLIP↓ | CMMD↓ | PSNR↑ |
|--------|-----------|-------|-------|
| CityDreamer | 36.52 | 4.152 | 12.58 |
| GaussianCity | 28.73 | 2.917 | 13.41 |
| CoR-GS | 27.32 | 3.752 | 12.85 |
| **Ours** | **9.91** | **2.009** | 14.28 |

同样huge improvement。注意pixel-level metrics提升不大——因为所有方法在pixel-level都差不多（generative task本身不适合pixel metrics），但distribution-based metrics显示巨大差距。

### 6.5 User Study

89 participants，三个维度：geometric accuracy, spatial alignment, overall quality。

- DFC2019: Ours ~97% winrate vs Sat-NeRF ~3%
- GoogleEarth: Ours ~90% winrate vs CityDreamer ~4%

### 6.6 Rendering Efficiency

| Method | FPS | GPU |
|--------|-----|-----|
| CityDreamer | 0.18 | A100 |
| GaussianCity | 10.72 | A100 |
| **Ours** | **11** | T4 |
| **Ours** | **40** | MacBook Air M2 |

**Insight**: CityDreamer需要A100还只有0.18 FPS，而Skyfall-GS在便宜得多的T4上就达到11 FPS，在MacBook上达到40 FPS。因为3DGS是rasterization-based，不需要ray marching或neural network inference during rendering。

Reference: [3DGS rendering efficiency analysis](https://research.nvidia.com/labs/lpr/3dgaussian-splatting/)

## 7. Ablation Studies

### 7.1 Reconstruction Stage Ablation (Table 3)

| App. Modeling | Opacity Reg. | Depth Sup. | FID_CLIP↓ | CMMD↓ |
|---------------|-------------|------------|-----------|-------|
| ✗ | ✗ | ✗ | Failed | Failed |
| ✓ | ✗ | ✗ | 41.90 | 2.45 |
| ✓ | ✓ | ✗ | 39.95 | 2.40 |
| ✓ | ✓ | ✓ | 38.01 | 2.31 |

**关键发现**: 没有appearance modeling，训练直接failed！因为multi-date illumination变化太大，无法converge。这验证了satellite imagery的特殊挑战。

### 7.2 Synthesis Stage Ablation (Table 4)

| Multiple Samples | Curriculum | FID_CLIP↓ | CMMD↓ |
|-----------------|------------|-----------|-------|
| ✗ | ✗ | 34.11 | 3.19 |
| ✓ | ✗ | 33.79 | 3.36 |
| ✓ | ✓ | **28.35** | **2.88** |

**Interesting**: Multiple samples alone提升不大（甚至CMMD略升），但加上curriculum后大幅提升。说明curriculum是关键——它让3DGS逐步learn occluded regions，而不是一次性面对所有困难views。

## 8. 与相关工作的联系

### 8.1 SDS vs. IDU

DreamFusion (Poole et al., 2022)用Score Distillation Sampling (SDS)把2D diffusion prior注入3D optimization。Skyfall-GS用IDU——一个更直接的iterative approach。

**对比**:
- **SDS**: 在gradient space注入prior，需要backprop through diffusion U-Net，memory intensive
- **IDU**: 在data space注入prior，render → edit → retrain，更简单且memory efficient

Paper选择IDU是合理的——对于large-scale satellite scene，SDS的memory开销会prohibitive。

Reference: [DreamFusion](https://dreamfusion3d.github.io/)

### 8.2 与Sat-NeRF, EOGS的关系

Sat-NeRF (Marí et al., 2022)和EOGS (Savant Aira et al., 2025)是satellite-specific NeRF方法，处理RPC camera和transient objects。但它们都是**reconstruction方法**，不generate missing regions。

Skyfall-GS的positioning很独特：**reconstruction + generation hybrid**。先用reconstruction得到coarse structure，再用generation填补occluded regions。

Reference: [Sat-NeRF](https://github.com/centreborelli/sat-nerf), [EOGS](https://github.com/centreborelli/EOGS)

### 8.3 与CityDreamer, GaussianCity的关系

CityDreamer (Xie et al., 2024)用BEV neural fields + height fields。GaussianCity (Xie et al., 2025b)用BEV-Point splats。

**关键区别**: 这些方法需要semantic maps和height fields作为input，且在specific dataset上训练。Skyfall-GS直接用satellite imagery + pretrained diffusion model，**zero-shot generalization**。

这让我联想到foundation model vs. task-specific model的tension——Skyfall-GS leverage foundation model (FLUX)的prior knowledge，避免了domain-specific training的数据瓶颈。

Reference: [CityDreamer](https://github.com/hzxie/CityDreamer), [GaussianCity](https://github.com/hzxie/GaussianCity)

## 9. Limitations & Future Directions

Paper自己提到：
1. **计算资源**: refinement process耗时（6小时on A6000）
2. **Street-level detail**: 极端street-level视角texture过于smooth
3. **Scaling**: 更大环境、dynamic scenes

**我的speculation**: 
- 可以用3D-aware diffusion model (如MVDream (Shi et al., 2023))替代2D FlowEdit，但需要finetune for satellite domain
- 可以用progressive Gaussian densification (DreamGaussian (Tang et al., 2023))加速refinement
- 可以结合video diffusion model处理temporal consistency for dynamic scenes

## 10. Technical Insights Summary

### 10.1 Key Design Choices

1. **Appearance modeling with limited SH**: 防止view-dependent color absorbing appearance变化
2. **Entropy-based opacity regularization**: push floaters to 0 or 1，binary-like geometry
3. **Pearson correlation depth loss**: 处理monocular depth的scale-ambiguity
4. **Curriculum from sky to ground**: 利用satellite view的分布特性，progressive refinement
5. **Multiple diffusion samples**: 避免commit to single suboptimal denoising trajectory
6. **75% refined + 25% original**: 保持fidelity to input satellite imagery

### 10.2 Why this works so well

**Intuition**: Satellite imagery给我们**真实的coarse geometry和texture**，但missing facades和低视角detail。Pretrained diffusion model (FLUX)有**rich natural image prior**，知道buildings应该长什么样。Curriculum IDU让两者synergize——从easy高elevation views开始，diffusion model refine minor artifacts；逐步到hard低elevation views，diffusion model hallucinate missing facades。

整个pipeline的beauty在于：**每个component都解决了satellite imagery的特定failure mode**：
- Sparse views → appearance modeling + depth supervision
- Floaters → opacity regularization  
- Missing facades → diffusion-based refinement
- Multi-view inconsistency → multiple samples
- Hard low-elevation views → curriculum learning

### 10.3 Broader Impact

这个工作对urban simulation、robotics training、gaming都有implications。想象一下：给定任意城市的satellite imagery（Google Earth到处都有），就能生成free-flight navigable 3D city——这对embodied AI training data generation、digital twins、VR/AR应用都有巨大价值。

**Potential extensions**:
- 用不同diffusion model control style（day/night, season, weather）
- 用LLM generate text prompts from satellite image analysis（自动prompt engineering）
- 结合OpenStreetMap metadata增强semantic understanding
- temporal extension：multi-temporal satellite imagery → 4D city (construction, demolition)

## References

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Kerbl et al., 2023
- [WildGaussians](https://arxiv.org/abs/2406.12273) - Kulhanek et al., 2024
- [MoGe: Monocular Geometry Estimation](https://arxiv.org/abs/2410.19115) - Wang et al., 2024
- [FlowEdit](https://github.com/UCSC-VLAA/FlowEdit) - Kulikov et al., 2024
- [FLUX.1](https://blackforestlabs.ai/) - Black Forest Labs
- [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/) - Haque et al., 2023
- [IM-3D](https://arxiv.org/abs/2403.02113) - Melas-Kyriazi et al., 2024
- [DreamFusion (SDS)](https://dreamfusion3d.github.io/) - Poole et al., 2022
- [ProlificDreamer (VSD)](https://arxiv.org/abs/2305.13913) - Wang et al., 2023
- [Sat-NeRF](https://github.com/centreborelli/sat-nerf) - Marí et al., 2022
- [CityDreamer](https://github.com/hzxie/CityDreamer) - Xie et al., 2024
- [GaussianCity](https://github.com/hzxie/GaussianCity) - Xie et al., 2025
- [DFC2019 Dataset](https://www.grss-ieee.org/2019-ieee-grss-data-fusion-contest/) - Le Saux et al., 2019
- [Mip-Splatting](https://github.com/autonomousvision/mip-splatting) - Yu et al., 2024
- [CoR-GS](https://github.com/Jiawei-Zhang-AI/CoR-GS) - Zhang et al., 2024
- [SatelliteSfM](https://github.com/zhangkai410/SatelliteSfM) - Zhang et al., 2019
- [Google Earth Studio](https://earth.google.com/studio/)
- [FID_CLIP](https://github.com/microsoft/qlora-paper-2023) - Kynkäänniemi et al., 2023
- [CMMD](https://github.com/google-research/google-research/tree/master/cmmd) - Jayasumana et al., 2024
- [Skyfall-GS Project Page](https://skyfall-gs.jayinnn.dev/)

这篇paper的精彩之处在于**将satellite reconstruction和generative refinement优雅结合**，用curriculum learning解决distribution shift问题，用multiple samples处理multi-view consistency。整个pipeline设计principled且each component都有明确justification。值得细读。
