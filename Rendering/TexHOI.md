---
source_pdf: TexHOI.pdf
paper_sha256: 84e4a9a971c11a817bdbd27a3b8446805c9a621801334869fc3c1dd04e5eba40
processed_at: '2026-08-12T13:45:11-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 TexHOI

## 一句话版本

你拍一个人抓着瓶子的视频，想把瓶子的3D model和干净texture重建出来。问题是手一直挡着瓶子、还在瓶子上投阴影、皮肤反光也映在瓶子上。之前的工作要么假装手不存在，要么直接把手影"烤"进瓶子texture里（换光线就穿帮）。TexHOI是**第一个认真把手当dynamic occluder建模进inverse rendering pipeline**的工作。

---

## 1. Problem到底是什么——一个日常场景

想象你用手机拍朋友抓可乐罐的视频，绕一圈。你想用这个视频重建可乐罐的3D模型，texture要干净——就是罐子上印的"Coca-Cola"字样、红色、铝质感，**不包含**任何"手挡住的反光、手投下的阴影、皮肤黄颜色反射到罐子上"这些transient stuff。

之前方法的尴尬:

- **HOLD** (2024 CVPR best paper candidate): 重建geometry很准，texture也能看，但它本质是NeRF，把手影、皮肤反射、environment highlight统统bake进object texture。你换一束光打这个重建的罐子，texture上还残留着原来手影的位置，穿帮。
- **PhySG / InvRender / NeRFactor**: 这些physics-based inverse rendering方法能disentangle albedo和lighting，但只处理**static single object**。你让它处理一只手在物体前面动来动去的视频，它不会算手投的shadow——要么假设没有occluder，要么用Monte-Carlo ray-tracing (NeRFactor)，每frame 30分钟，手每帧都变，完全跑不动。

所以problem: **怎么在hand-occluded monocular video里，把object的真实albedo拎出来，同时把手投的shadow和skin reflection当作physical phenomenon显式建模掉？**

---

## 2. Solution的直觉——两步走

TexHOI的设计哲学: **pose和material是两个不同性质的优化问题，混在一起会互相干扰，分开做**。

### Stage 1: 先把pose搞准 (Compositional NeRF)

第一阶段的逻辑很简单:
- 你给hand估了个pose (用MANO model)，给object估了个pose
- 这两个初始pose都是不准的 (off-the-shelf estimator误差大)
- 用三个NeRF分别表示hand、object、background，composite渲染出来和ground truth image对比
- 通过RGB loss、segmentation loss、contact loss ( fingertip贴着object surface ) 反传，同时refine hand pose、object pose和三个NeRF的geometry

这一步用的就是HOLD的思路，没什么新东西。但有个聪明的小trick:

**Hand-Object visible mask** $M_{ho}$: 想知道object哪些部分被手挡住、哪些能看见？很简单，在volumetric rendering里**把object的color强制设为1，hand的color强制设为0**，再render一遍。被手挡的object ray会被手absorb，没被挡的object ray透出来color=1。这个mask免费得到，给stage 2用。

### Stage 2: 在已知pose下做physics-based分解

第一阶段输出: refined hand pose、object pose、object的coarse geometry、object mask、visible object mask $M_{ho}$。

第二阶段: **不用NeRF volumetric rendering了，改成surface rendering**。surface rendering的优势是可以精确控制每个surface point的material参数——roughness、specular、albedo——直接对应PBR equation里的BRDF term。

PBR equation就一句话: **一个pixel看到的color = 这个surface point收到的light × 这个surface的BRDF × 几何投影因子**。

$$c(\omega_o; x) = \int_{\Omega^+} L(\omega_i, x) \cdot \phi(x; \omega_o, \omega_i) \cdot (\omega_i \cdot n) \, d\omega_i$$

三个term:
- $L(\omega_i, x)$: 从方向$\omega_i$打到点$x$的光
- $\phi$: BRDF，surface怎么反射光
- $\omega_i \cdot n$: Lambert cosine law，光斜射时单位面积能量减少

**Stage 2的目标就是分解这三个term**，让bake进$L$和$\phi$里的hand effect显式出来。

---

## 3. 三个term怎么建模——核心技术

### 3.1 Environment Light: 128个Spherical Gaussian

environment light是360度来的光。怎么参数化？用128个SG (Spherical Gaussian)。

SG是个什么玩意: 想象球面上一个"光斑"，中心方向$\xi$最亮，离中心越远越暗，衰减速度由sharpness $\eta$控制。数学上:

$$G(v; \xi, \mu, \eta) = \mu \cdot e^{\eta(v \cdot \xi - 1)}$$

- $v$: 球面上某个方向
- $\xi$: 光斑的中心方向
- $\mu$: 中心峰值亮度
- $\eta$: 光斑的"胖瘦"，越大越窄像激光，越小越宽像阴天

128个SG加起来近似整个environment的light map。**object在动**——每帧object自己旋转，所以从object的坐标系看，environment是在反向旋转——所以每帧根据object pose反向rotate这128个SG的lobe directions $\psi_j$。

这一步来自PhySG。SG的好处: SG × SG 还是SG，SG的hemisphere积分有closed-form，整个PBR integral可以解析求解。

### 3.2 BRDF: Diffuse + Specular

每个surface point $x$:
- **Albedo** $a(x)$: 物体本身的颜色 (RGB)，比如可乐罐的红色——这是TexHOI最想求的东西
- **Roughness** $r(x)$: 表面粗糙度，0=镜面，1=磨砂
- **Specular reflectance** $s$: 金属感，0=塑料，1=金属

Diffuse BRDF (Lambertian): $\phi_d = a(x)/\pi$，光均匀散射到各个方向。

Specular BRDF (Cook-Torrance microfacet): $\phi_s = \frac{F \cdot g}{4(\omega_o \cdot n)(\omega_i \cdot n)}$

直觉: roughness小→specular聚成一个亮highlight；roughness大→specular散开成大模糊高光。

### 3.3 Hand Occlusion: 108个Sphere ⭐ 核心创新

**这是paper最clever的部分**。

挑战: 要算hand投在object上的shadow，传统方法是ray-tracing——对每个object surface point，对每个incoming light direction，发射一根ray看是否被hand挡住。Hand每帧deform，ray-tracing每帧重做。NeRFactor用Monte-Carlo ray-tracing每frame 30分钟，dynamic hand完全不可行。

TexHOI的方案: **把MANO hand用108个sphere表示，然后用closed-form公式算occlusion**。

#### Step 1: 108个sphere怎么放在canonical hand上

在canonical hand (手掌平展的标准姿势) 上手动放108个sphere。每个sphere有center $p_i$ 和 radius $r_i$。

怎么决定每个sphere"管"哪些hand surface vertices？用**power diagram**——比Voronoi diagram多一个权重，让大sphere管的区域更大。每个vertex分配给power distance最近的sphere。

Power distance: $d_{pow}(x, p_i, r_i) = ||x - p_i||^2 - r_i^2$

#### Step 2: Hand deform时sphere怎么跟着动

当MANO hand因为pose参数deform (抓握、伸展)，每个vertex移到新位置$\hat{v}_j$。那么:

- New sphere center $\hat{p}_i$ = 这个sphere cell里所有deformed vertices的centroid
- New sphere radius $\hat{r}_i$ = 每个vertex沿其normal方向到sphere center的投影距离的平均

为什么用normal projection而不是直线距离: **让sphere tangent于local surface**，sphere不会oversize超出hand表面，能贴合真实hand geometry。如果用直线距离，sphere可能鼓出hand表面太多。

#### Step 3: Occlusion怎么closed-form算

对一个object surface point $x$，它的incoming light来自整个hemisphere (上半球)。把SG lobe (照亮这个点的环境光分布) 离散化成$64 \times 64 = 4096$个patches，每个patch代表一小块incoming direction。

108个sphere投影到这个hemisphere上，覆盖某些patches。被覆盖的patches代表被hand挡住的incoming direction。

**关键公式**: 一个patch内SG的积分可以用normalized ISG (Integral Spherical Gaussian) closed-form算出:

$$\hat{S}(\theta, \phi, \eta) \approx \frac{1}{1 + e^{-g(\eta)(\theta - \pi/2)}} \cdot \frac{1}{1 + e^{-h(\eta)(\phi - \pi/2)}}$$

这是logistic sigmoid的product，常数$g(\eta)$和$h(\eta)$是sharpness $\eta$的四次多项式 (paper里给了具体数值)。

对每个patch用四个corner的$\hat{S}$值通过inclusion-exclusion原理算patch内积分:

$$\mathcal{F} = \hat{S}(\theta_1, \phi_1) - \hat{S}(\theta_1, \phi_0) - \hat{S}(\theta_0, \phi_1) + \hat{S}(\theta_0, \phi_0)$$

所有被108个sphere覆盖的patches的fractional contributions加起来，就是hand对这个点的occlusion fraction $\mathcal{F}(x)$。

**然后整个被occlude部分的SG积分 = $\mathcal{F}(x) \times$ 全hemisphere的SG积分** (因为全hemisphere的SG积分有解析解)。

这就是paper的核心magic: **用sphere-to-patch的投影 + normalized ISG closed-form，完全避免ray-tracing**。

108个sphere的投影是简单几何操作，patch的fractional integral是closed-form sigmoid，整个occlusion计算是matrix operation，对GPU友好。这比Monte-Carlo ray-tracing快几个数量级。

---

## 4. Hand Reflection也要建模

Hand不仅挡光，还反射skin color到object上。比如你抓白杯子，手指附近的杯子表面会染上一点黄皮肤色——这就是indirect illumination。

paper用一个constant RGB $L_i$ 表示这个skin reflection，初始值是网上查的generic skin color (#E0AC69, 大约RGB (224, 172, 105))。

最终 illumination:

$$L(\omega_i, x) = L_d(\omega_i) \cdot (1 - O(x, \omega_i)) + L_i \cdot O(x, \omega_i)$$

直觉: 在occluded方向，environment light进不来，但hand反射的skin color替代进去。这是简化模型——真实情况是multi-bounce light transport，但用single constant RGB抓住了dominant effect (skin reflection的色调)。

这个$L_i$在训练中被优化，从generic skin color出发，让网络微调到符合具体场景的skin颜色。

---

## 5. 一个巧妙的小trick: Cosine term也用SG近似

PBR equation里有个$\omega_i \cdot n$的cosine term (Lambert's law)。为了让整个integrand都是SG (从而closed-form积分):

$$\omega_i \cdot n \approx G(\omega_i; 0.0315, 32.7080, n) - 31.7003$$

一个以normal为lobe direction的SG，加上offset，近似cosine function。常数0.0315、32.7080、-31.7003是PhySG paper里拟合得到的。

这样: SG (illumination) × SG (cosine) × SG (BRDF specular) 全是SG，integral closed-form，整个PBR equation可微，可以end-to-end backprop。

---

## 6. 实验结果——重点在哪里

**Table 1 (Average over 6 HO3D objects)**:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| InvRender | 19.23 | 0.9301 | 0.154 |
| HOLD | 20.61 | 0.9354 | 0.101 |
| **TexHOI** | **20.96** | **0.9387** | **0.095** |

**关键观察**:

- PSNR只比HOLD高0.35dB，看起来不多。但paper解释: HOLD把shadow bake进texture，pixel-wise反而和ground truth近 (因为ground truth本来就是带shadow的image)。TexHOI的albedo是干净的，relight到原环境才和ground truth匹配。0.35dB PSNR + 0.006 SSIM + **0.006 LPIPS**的提升在inverse rendering领域是显著的。
- **LPIPS提升更convincing** (0.101 → 0.095): LPIPS是perceptual metric，衡量人眼感受相似度。TexHOI的texture在perceptual quality上明显更好。

**Drilling Machine的有趣case**:
TexHOI (17.64 PSNR) < HOLD (18.12 PSNR)。Drilling machine是黑色金属表面，specular强。HOLD把high-light和shadow都bake进texture，pixel-wise反而更近。但TexHOI的LPIPS (0.139) < HOLD (0.152)，perceptual更好——说明TexHOI把metal highlight建模为specular BRDF而不是bake进albedo，relight时更合理。

**Ablation (Table 2)**:

| Config | PSNR | SSIM | LPIPS |
|--------|------|------|-------|
| w/o indirect illum | 20.87 | 0.9380 | 0.112 |
| w/o occlusion | 20.85 | 0.9351 | 0.113 |
| Full | 20.96 | 0.9387 | 0.095 |

- 去掉indirect illumination: hand grasp区域偏暗 (因为没有skin reflection补偿)
- 去掉occlusion handling: hand grasp区域出现shadow artifact被bake进albedo
- Full model在LPIPS上明显领先 (0.095 vs 0.112/0.113)，证明两个components都对perceptual quality有贡献

---

## 7. Limitation——为什么这个方法还不够好

### 7.1 Static Hand问题 (最严重)

如果你抓一个瓶子，手一直不动抓同一个位置，整段视频手都不换位置——那么TexHOI会把手的shape和shadow **bake进geometry和texture**。

为什么: 整个pipeline依赖**hand的视角变化和移动**来区分hand contribution和object intrinsic property。如果hand对object的相对关系不变，optimization无法判断"这片暗的区域是object本身的dark albedo"还是"hand投的shadow"。需要temporal diversity来disambiguate。

这个limitation挺本质的: inverse rendering本质上是个under-constrained problem，需要多视角/多光照变化来约束。Hand不动就缺少了"hand变化的视角"这一组约束。

### 7.2 Constant Skin Reflection $L_i$

用single RGB表示整个hand的skin reflection过于简化。实际:
- 手心、手背颜色不同
- 指甲、关节褶皱颜色不同
- 不同光照下skin反射的色调变化 (subsurface scattering)

更准确的模型应该是: spatially-varying skin BRDF，可能用diffusion prior (像IntrinsicAnything那样)。

### 7.3 Sphere Inter-penetration

108 sphere的中心在hand surface外，但sphere半径可能让sphere边界侵入object内部。尤其在fingertip附近，sphere贴着object，容易overlap。造成occlusion计算小误差。

未来可以加collision detection或differentiable physics simulation (DiffTaichi) 来约束sphere不overlap object。

### 7.4 Two-Stage不是End-to-End

Stage 1训练14小时，stage 2再14小时，共28小时。如果end-to-end可能更efficient，但gradient scale不一致是challenge。

---

## 8. 几个Intuitive的类比帮助理解

### 8.1 108 sphere像什么

想象你要给一只glove (代表hand) 内部填充108个小气球。每个气球贴着glove的一部分内壁。气球的大小根据它负责的那块glove面积决定。当glove被手撑开变形 (MANO pose变化)，每个气球跟着移动、变大变小，但始终贴合globe内壁。

这108个气球是hand的proxy。算shadow时，不用真的检查hand mesh的每个triangle，只需检查这108个气球挡了哪些方向。

### 8.2 SG像什么

想象environment light是天空中的光分布。128个SG就像128个"探照灯"分布在天空各方向，每个有自己的方向、亮度、聚焦程度。加起来就是整个天空的light map。物体转动时，从物体看天空在反向转动，所以128个探照灯的方向需要反向旋转。

### 8.3 Power Diagram像什么

想象108个speaker放在房间里，每个speaker音量不同。Power diagram就是"每个位置的人听哪个speaker最响"的partition——音量大的speaker管的区域更大。Hand surface的每个vertex被分配给"听得最清楚的speaker"对应的sphere。

### 8.4 Normalized ISG像什么

想象SG lobe是一个圆形光斑。要算这个光斑被一块patch (一个方形区域) 遮挡了多少比例。直接积分很难，但Iwasaki 2012发现这个normalized integral可以用两个sigmoid相乘近似——sigmoid就是logistic函数那种S形曲线。所以patch的fractional contribution变成两个S形曲线相乘，closed-form算出。

---

## 9. 大局观: 这个工作在field里的位置

Inverse rendering从2018 NeRF出现后有几波浪潮:

**Wave 1 (2020-2021)**: NeRF本身，能重建geometry和view-dependent color，但不分解material和lighting
- NeRF: https://arxiv.org/abs/2003.08934

**Wave 2 (2021-2022)**: Physics-based inverse rendering，用SG或SH分解material/lighting
- PhySG: https://arxiv.org/abs/2104.01431
- NeRFactor: https://arxiv.org/abs/2106.09820
- InvRender: https://arxiv.org/abs/2212.05038
- NeRD: https://arxiv.org/abs/2012.03918

这些方法都假设**static single rigid object**。Dynamic occluder完全没考虑。

**Wave 3 (2023-2024)**: Compositional NeRF for multi-object
- Object-NeRF: https://arxiv.org/abs/2105.13591
- HOLD: https://arxiv.org/abs/2404.01804
- BundleSDF: https://arxiv.org/abs/2303.14130
- DiffHOI: https://arxiv.org/abs/2307.11889

这些处理dynamic multi-object，但只重建geometry，不处理physics-based texture。

**TexHOI的位置**: 介于Wave 2和Wave 3之间——把Wave 2的physics-based inverse rendering**扩展到**Wave 3的dynamic hand-object interaction场景。第一个认真把dynamic hand作为occluder加进PBR pipeline。

**Wave 4 (2024-)**: 3DGS-based inverse rendering
- Relightable 3DGS: https://arxiv.org/abs/2311.16043
- GS-IR: https://arxiv.org/abs/2311.16473
- GaussianShader: https://arxiv.org/abs/2311.17977

这些用3D Gaussian Splatting替代NeRF，更快。但还没人做3DGS-based hand-object interaction inverse rendering——TexHOI future work明确提到这个方向。

---

## 10. 如果Karpathy想extend这个工作

几个interesting方向:

### 10.1 Diffusion Prior for Occluded Albedo

被hand挡住的object region，albedo完全无法从video观测到。当前TexHOI靠MLP在canonical space插值。如果加Stable Diffusion prior (像IntrinsicAnything https://arxiv.org/abs/2404.11593那样用diffusion约束albedo的naturalness)，可能更好。比如被挡住的瓶子label部分，diffusion prior能生成合理的label图案。

### 10.2 Video Diffusion作为supervisor

Text-to-video diffusion model (如Stable Video Diffusion https://arxiv.org/abs/2311.15127) 对natural video有强prior。把TexHOI的rendering结果和video diffusion的output distribution对齐，可能improve temporal consistency和occluded region inpainting。

### 10.3 从YouTube大规模scrape训练albedo predictor

TexHOI是per-scene optimization (28小时)。可以scrape millions个YouTube unboxing video (hand-object interaction丰富)，用TexHOI生成大量 (video, albedo) pairs，训练一个feed-forward albedo predictor network。类似DINO (https://arxiv.org/abs/2104.14294) 之于image understanding，但这里做texture understanding。

### 10.4 Robot Imitation Learning的real-to-sim

High-fidelity object texture对robot simulator (如Isaac Gym https://developer.nvidia.com/isaac-gym) 很重要。TexHOI能从real video重建object + albedo，直接放进simulator训练manipulation policy，缩小real-to-sim gap。

### 10.5 3DGS-based End-to-End HOI Inverse Rendering

TexHOI未来工作明确提到这个方向。3DGS (https://arxiv.org/abs/2302.07681) 比NeRF快100倍训练，可微rendering天然支持。挑战: 3DGS的离散Gaussian对occlusion计算需要新的formulation (sphere-to-splat投影?)，可能比NeRF的volumetric formulation更复杂或更简单。

### 10.6 Differentiable Physics for Hand-Object Contact

当前contact loss (Eq. 12) 只是fingertip到object surface的minimum distance。真实的hand-object interaction有friction、deformation、force closure。用differentiable physics simulator (DiffTaichi https://github.com/taichi-dev/difftaichi, Brax https://github.com/google/brax) 能给更physical的prior。

---

## 11. 最后的take-away

TexHOI的核心contribution可以浓缩成一句:

> **把hand用108个可参数化sphere表示，把SG lobe discretize成64×64 patches，用normalized ISG closed-form算每个patch的occlusion fraction，从而在inverse rendering pipeline里显式建模dynamic hand的shadow和skin reflection，避免albedo被baked。**

整个paper的beauty在于: problem是practical的 (hand-object interaction场景到处都是)，solution是elegant的 (closed-form occlusion避免ray-tracing)，engineering是sound的 (two-stage decoupling稳定训练)。

虽然PSNR提升不大，但**首次formulate了这个问题并给出了可用的solution**——这是first-mover work的典型特征。后续工作会在这个framework上扩展: 3DGS加速、diffusion prior inpaint、end-to-end optimization、multi-hand extension。

对Karpathy这样对system building有taste的人来说，TexHOI的appeal可能在于: 它是**careful engineering meeting physical insight**的产物——108 sphere这个数字、power diagram的partition方式、normalized ISG的sigmoid近似，每个design choice都有physical motivation，不是随便堆trick。

**参考链接汇总**:
- TexHOI project: https://alakhag.github.io/TexHOI-website
- HOLD baseline: https://arxiv.org/abs/2404.01804
- PhySG foundation: https://arxiv.org/abs/2104.01431
- InvRender baseline: https://arxiv.org/abs/2212.05038
- Iwasaki ISG 2012: https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2012.03092.x
- MANO model: https://arxiv.org/abs/2201.02610
- HO3D dataset: http://ho3d.stevenhampali.com/
- NeRF: https://arxiv.org/abs/2003.08934
- NeRFactor ray-tracing: https://arxiv.org/abs/2106.09820
- 3DGS: https://arxiv.org/abs/2302.07681
- Relightable 3DGS: https://arxiv.org/abs/2311.16043
- IntrinsicAnything diffusion prior: https://arxiv.org/abs/2404.11593
- Mitsuba renderer: https://mitsuba-renderer.org/
- DiffTaichi: https://github.com/taichi-dev/difftaichi
- YCB objects: https://www.ycbbenchmarks.com/
- DexYCB: https://github.com/dexycb/dex-ycb-toolkit
- SAM-Track: https://arxiv.org/abs/2305.06558
- Kajiya rendering equation 1986: https://dl.acm.org/doi/10.1145/15922.15902

---

# TexHOI: Reconstructing Textures of 3D Unknown Objects in Monocular Hand-Object Interaction Scenes

## 1. Paper核心问题与动机

TexHOI 由 Alakh Aggarwal, Ningna Wang, Xiaohu Guo 提出，处理一个被忽视得令人惊讶的问题：当 hand 持续 grasp 一个 rigid object 时，hand 会在 object surface 上 cast shadows、reflect skin color、block environment light，这些 transient effects会被之前的 inverse rendering 方法 **baked into** albedo texture，导致 texture 在不同 lighting 下 generalize 失败。

**核心 motivation**:
- HO3D、DexYCB、Contact-Pose 等 dataset 提供了丰富的 hand-object interaction 数据，但利用这些数据重建 high-fidelity texture 时，hand 引起的 shadows 和 reflections 严重污染 texture prediction
- 现有方法分两类，各有缺陷：
  - Geometry-focused (BundleSDF、DiffHOI、HOLD)：重建 geometry，不处理 texture 中的 lighting contamination
  - Physics-based inverse rendering (PhySG、InvRender、RefNeRF、NeRFactor)：用 Spherical Harmonics 或 Spherical Gaussians 建模 environment illumination，但只处理 **static single rigid object**，对 dynamic occluder (hand) 失效
- Monte-Carlo ray-tracing 在 single GPU 上每 frame 需要 30 分钟 (NeRFactor)，对于 dynamic hand 这种每帧变化的 occluder，cost 难以承受

**Project page**: https://alakhag.github.io/TexHOI-website

---

## 2. Two-Stage Pipeline Architecture Overview

整篇 paper 的 architecture 是一个 **decoupled two-stage** 设计，其 intuition 是：pose optimization 和 material/light decomposition 是两个本质不同的优化问题，混在一起会导致 local minima。

### Stage 1: Compositional Volumetric Rendering
- **目的**: refine hand pose + object pose，得到 coarse geometry 和 low-fidelity texture
- **技术**: 三个 NeRF (hand, object, background) 在 observation space composite rendering
- **借鉴**: HOLD 的 compositional NeRF 思路
- **输出**: refined poses, object implicit geometry network, object mask $M_{obj}$, hand-object mask $M_{ho}$

### Stage 2: Physics-based Surface Rendering with Spherical Gaussians
- **目的**: disentangle albedo from environment illumination 和 hand occlusion effects
- **技术**: surface rendering + PBR equation + 108 sphere hand representation + SG-based occlusion
- **借鉴**: PhySG 的 SG-based inverse rendering + Iwasaki et al. 的 Integral Spherical Gaussian (ISG) for real-time occlusion
- **关键创新**: 用 power diagram 划分的 108 spheres 表示 MANO hand，将 occlusion 计算从 ray-tracing 退化到 closed-form fractional integral

---

## 3. Stage 1: Compositional NeRF 的技术细节

### 3.1 Object NeRF — Rigid Body Canonical Space

对于 object 这种 rigid body，将 observation space点 $x'$ 逆变换到 object canonical space:

$$x_{obj} = R^{-1}(x' - t) \quad \text{(Eq. 4)}$$

变量含义:
- $x' \in \mathbb{R}^3$: observation space point (camera coordinate)
- $R \in SO(3)$: object pose 的 rotation matrix
- $t \in \mathbb{R}^3$: object pose 的 translation
- $x_{obj}$: 在 object 自己的 canonical space 中的 point

**Intuition**: object 是 rigid 的，所以 object 内部任何点的相对位置不变；我们将 network 的输入从 camera space 转到 object canonical space，使得 network 只需要学习 object 本身的 radiance field，不需要关心 object 在 camera 哪里。

### 3.2 Hand NeRF — Linear Blend Skinning 的 Inverse

Hand 是 articulated 的，使用 MANO model 的 skinning weights。canonical pose 是 stretched hand (手掌平展)。逆变换用 linear blend skinning 的 inverse:

$$x_{hand} = \left(\sum_{i=1}^{n_{jnt}} w_i(x') T_i \right)^{-1} x' \quad \text{(Eq. 5)}$$

变量含义:
- $n_{jnt}$: hand 关节数量 (MANO 默认 15 joints + root)
- $T_i \in SE(3)$: 第 $i^{th}$ joint 的 transformation matrix
- $w_i(x') \in [0,1]$: 点 $x'$ 对 joint $i$ 的 skinning weight，$\sum_i w_i = 1$
- $x_{hand}$: canonical hand space 中的 point

**Intuition**: 一个 observation space点受到多个 joints 影响，每个 joint 给一个 weight，weighted sum of $T_i$ 给出该点的 effective transformation。这个 weighted sum 是 $3 \times 3$ 矩阵 + translation 部分，取 inverse 就得到从 observation 到 canonical 的逆变换。这就是 MANO forward LBS 公式 $x' = \sum_i w_i T_i x_{hand}$ 的 inversion。

### 3.3 Compositional Volumetric Rendering

三个 NeRF (hand, object, background) 各自输出 color 和 density。沿一条 ray，对 hand 和 object 各 sample $n$ 个点，共 $2n$ 个点，按到 camera 的距离 sort 后做 alpha compositing:

$$C_{fg}(r) = \sum_{i=1}^{2n} \tau_i c_i \quad \text{(Eq. 7)}$$

其中:
$$\tau_i = \exp\left(-\sum_{j<i} \sigma_j \delta_j\right) \left(1 - \exp(-\sigma_i \delta_i)\right)$$

- $\sigma_i$: 第 $i^{th}$ sample point 的 density
- $\delta_i = t_{i+1} - t_i$: 相邻 sample points 的距离
- $\tau_i$: 这是 NeRF 的 standard alpha-compositing weight，第一项是 accumulated transmittance (前面所有点的 occlusion)，第二项是当前点的 absorption probability

最终 composite color 加入 background:

$$C(r) = C_{fg}(r) + (1 - M_{fg}(r)) \cdot C_{bg}(r) \quad \text{(Eq. 6)}$$

- $M_{fg}(r) = \sum_i \tau_i$: foreground mask，可以理解为 foreground 沿 ray 的 "presence"

**Segmentation 通过 argmax**:
$$M(r) = \arg\max\{M_{hand}(r), M_{obj}(r), M_{bg}(r)\} \quad \text{(Eq. 9)}$$

这是 one-hot label，用于和 SAM-Track 得到的 ground truth segmentation 比较。

### 3.4 Hand-Object Visible Mask $M_{ho}$ — 重要 trick

为了 stage 2 只在 visible object region 上 optimize albedo，需要剔除被 hand occlude 的 object region。trick 是: **强制 object sample 点 color=1，hand sample points color=0**，再做一遍 volumetric rendering。这样 hand 区域完全 absorb，只有透过 hand 看得到的 object region 才有 transmittance。

$$M_{ho}(r) = \text{composite rendering with } c^{obj}_i=1, c^{hand}_i=0$$

这是 Eq. 7 和 8 的特殊化使用。

### 3.5 Stage 1 Loss Function

$$\mathcal{L}_{s1} = \mathcal{L}_{rgb1} + \lambda_{seg1}\mathcal{L}_{seg1} + \lambda_{eikonal1}\mathcal{L}_{eikonal1} + \lambda_{hand-sdf1}\mathcal{L}_{hand-sdf1} + \lambda_{contact1}\mathcal{L}_{contact1} \quad \text{(Eq. 13)}$$

**Contact Loss** 是关键，鼓励 fingertip vertices $V_{tip}$ 接近 object vertices $V_o$:

$$\mathcal{L}_{contact1} = \sum_i \min_j ||V_{tip}^i - V_o^j|| \quad \text{(Eq. 12)}$$

这是 minimum distance，对每个 fingertip vertex 找最近 object vertex，pulling them together。这是 hand-object interaction 重建中的 standard physical prior，避免 hand 漂浮在 object 之外。

**Hyperparameters**:
- $\lambda_{seg1}$: 1.1 → 0.1 (progressively decrease over 30,000 iterations)
- $\lambda_{eikonal1}$: 1.0
- $\lambda_{hand-sdf1}$: 5.0
- $\lambda_{contact1}$: 0 → 1.0 (progressively increase)

$\lambda$ 的 schedule 很 intuitive: 开始时让 segmentation 和 RGB 主导，等网络有基本 shape 后逐步打开 contact prior 防止一开始就 stuck 在 local minimum。

---

## 4. Stage 2: Physics-based Inverse Rendering — 核心创新

### 4.1 PBR Equation

$$c(\omega_o; x) = \int_{\omega_i \in \Omega^+(n)} L(\omega_i, x) \, \phi(x; \omega_o, \omega_i) \, (\omega_i \cdot n) \, d\omega_i \quad \text{(Eq. 1)}$$

变量含义:
- $x$: object surface point
- $\omega_o \in \mathbb{S}^2$: outgoing view direction (从 surface 到 camera)
- $\omega_i \in \Omega^+(n)$: incoming light direction，$\Omega^+(n)$ 是 surface normal $n$ 定义的 hemisphere
- $L(\omega_i, x)$: incident radiance 从方向 $\omega_i$ 来到点 $x$
- $\phi(x; \omega_o, \omega_i)$: BRDF (Bidirectional Reflectance Distribution Function)
- $n$: surface normal at $x$
- $\omega_i \cdot n$: cosine term (Lambert's law, geometry factor)

**关键 insight**: 这个 integral over hemisphere 是 PBR 的 rendering equation (Kajiya 1986)，所有 inverse rendering 工作都在解这个 equation 的各个 components。

### 4.2 Spherical Gaussian (SG) 表示

$$G(v; \xi, \mu, \eta) = \mu \, e^{\eta(v \cdot \xi - 1)} \quad \text{(Eq. 2)}$$

变量含义:
- $v \in \mathbb{S}^2$: 球面上一个方向
- $\xi \in \mathbb{S}^2$: lobe direction (SG 的对称轴方向)
- $\mu \in \mathbb{R}^+$: lobe intensity (peak value)
- $\eta \in \mathbb{R}^+$: lobe sharpness (越大越窄，越像 dirac-delta，越小越宽，越像 diffuse)

**Intuition**: SG 是球面上的 Gaussian bump。当 $v = \xi$ 时 $G = \mu e^0 = \mu$；当 $v$ 远离 $\xi$ 时 $v \cdot \xi$ 减小，exponent 变负，G 衰减。$\eta$ 控制衰减速度。

### 4.3 Environment Illumination — 128 SGs

$$L_d(\omega_i) = \sum_{j=1}^{128} G(\omega_i; \psi_j, \mu_j, \eta_j) \quad \text{(Eq. 14)}$$

- 128 个 SGs 近似 environment illumination
- $\psi_j$: 第 $j^{th}$ SG 的 lobe direction
- $\mu_j, \eta_j$: 对应 intensity 和 sharpness
- 因为 object 在每帧有 rotation，environment 在 object canonical space 中是 counter-rotating 的，所以 $\psi_j$ 需要根据 object pose 反向 rotate

**Intuition**: environment light 在 world 中是固定的，但 object 自己的坐标系在旋转，所以从 object 看出去 environment 也在反向旋转。这就是为什么需要 rotate SG lobe directions。

### 4.4 BRDF — Diffuse + Specular

$$\phi(x; \omega_o, \omega_i) = \phi_d(x) + \phi_s(x; \omega_o, \omega_i) \quad \text{(Eq. 15)}$$

Diffuse BRDF (Lambertian):
$$\phi_d(x) = \frac{a(x)}{\pi}$$

- $a(x) \in [0,1]^3$: surface point 的 albedo color (RGB)，由 MLP 预测
- $\pi$ 是 normalization factor 确保 energy conservation

Specular BRDF (microfacet model，使用 Cook-Torrance form):
$$\phi_s(x; \omega_o, \omega_i) = \frac{F(s, r(x)) \, g(r(x))}{4 (\omega_o \cdot n) (\omega_i \cdot n)} \quad \text{(Eq. 16)}$$

- $s$: specular reflectance (F0，Fresnel at normal incidence)，控制 metalness
- $r(x) \in [0,1]$: roughness，由 MLP 预测
- $F(s, r)$: Fresnel term (Schlick approximation usually)
- $g(r)$: geometric shadowing / masking term (Smith GGX usually)
- $4 (\omega_o \cdot n)(\omega_i \cdot n)$: microfacet BRDF 的 standard normalization

**Intuition**: diffuse 给 base color，specular 给 highlights。roughness 大 → specular 散开成大 highlight；roughness 小 → specular 集中成小亮点。specular reflectance $s$ 控制金属感。

### 4.5 Hand Occlusion — 108 Parameterizable Spheres ⭐ 关键创新

#### 4.5.1 Sphere Placement via Power Diagram

paper 用 108 spheres 划分 canonical MANO hand volume $\Psi$。每个 sphere $(p_i, r_i)$ 通过 **power diagram** 划分 hand 表面 vertices:

$$\Psi_i^{pow}: \{x \in \Psi \mid d_{pow}(x, p_i, r_i) \leq d_{pow}(x, p_j, r_j), \forall j \neq i\} \quad \text{(Eq. 17)}$$

power distance:
$$d_{pow}(x, p_i, r_i) = ||x - p_i||^2 - r_i^2$$

- $p_i$: sphere center
- $r_i$: sphere radius
- $\Psi_i^{pow}$: power cell (比 Voronoi cell 多了 $r_i$ 影响)

**Intuition**: power diagram 是 weighted Voronoi diagram，加 $-r_i^2$ 让大 sphere 的 cell 更大。每个 surface vertex 被分配给最近的 sphere (按 power distance)。

#### 4.5.2 Sphere Parameters Update via MANO Deformation

当 MANO hand 因为 pose 和 shape 参数 deform 时，每个 sphere 的 center 和 radius 通过分配给它的 vertices 计算:

$$\hat{p}_i = \frac{1}{|\mathcal{V}_i|} \sum_{v_j \in \mathcal{V}_i} \hat{v}_j \quad \text{(Eq. 18 上)}$$

$$\hat{r}_i = \frac{1}{|\mathcal{V}_i|} \sum_{v_j \in \mathcal{V}_i} ||\hat{n}_j \cdot (\hat{v}_j - \hat{p}_i)|| \quad \text{(Eq. 18 下)}$$

变量:
- $\mathcal{V}_i$: power cell $i$ 中的所有 vertices set
- $\hat{v}_j$: deformed vertex position
- $\hat{n}_j$: deformed vertex normal
- $\hat{p}_i$: new sphere center (cell 中所有 vertices 的 centroid)
- $\hat{r}_i$: new sphere radius (每个 vertex 到 sphere center 沿 vertex normal 方向的投影绝对值的平均)

**为什么用 normal projection 而非直接距离**: 这个 radius 计算很妙。sphere 是用来 occlude light 的，sphere 应该贴着 hand surface。沿 vertex normal 方向投影，相当于让 sphere tangent 于 local surface。这样 sphere 不会 oversize 出去，能 reasonable 表示 hand 局部 surface。

#### 4.5.3 Occlusion Computation — SG Lobe 的 64×64 Patching

对于每个 object surface point $x$，将 SG 的 hemispherical lobe 分成 $64 \times 64$ patches。每个 sphere 投影到这个 lobe 上，覆盖某些 patches (记为 $\tilde{\Omega}$)。

**Integral Spherical Gaussian (ISG)**:
$$S(\theta, \phi, \eta) = \int_0^{\theta} \int_0^{\phi} e^{\eta(\cos\phi' - 1)} \, d\phi \, d\theta \quad \text{(Eq. 19)}$$

变量:
- $\theta, \phi$: polar angle, azimuthal angle on hemisphere
- $\eta$: SG sharpness
- $\phi'$: angle between SG lobe direction 和某 direction $\omega_i$
- $S$: accumulated SG value over a patch

**Normalized ISG**:
$$\hat{S}(\theta, \phi, \eta) = \frac{S(\theta, \phi, \eta)}{S(\pi, \pi, \eta)} \quad \text{(Eq. 20)}$$

normalized 到 $[0, 1]$，使 fractional contribution 可计算。

**Approximation** (Iwasaki et al.):
$$\hat{S}(\theta, \phi, \eta) \approx \frac{1}{1 + e^{-g(\eta)(\theta - \pi/2)}} \cdot \frac{1}{1 + e^{-h(\eta)(\phi - \pi/2)}} \quad \text{(Eq. 21)}$$

这是 logistic sigmoid 的 product，对 $\theta$ 和 $\phi$ 分别归一化。$g(\eta), h(\eta)$ 是 $\eta$ 的 4-th degree polynomial (Eq. 22)。

**Patch 的 fractional contribution**:
$$\mathcal{F}(x, \theta_0, \theta_1, \phi_0, \phi_1) = \hat{S}(\theta_1, \phi_1, \eta) - \hat{S}(\theta_1, \phi_0, \eta) - \hat{S}(\theta_0, \phi_1, \eta) + \hat{S}(\theta_0, \phi_0, \eta) \quad \text{(Eq. 23)}$$

这是 2D inclusion-exclusion 原理，用四个 corner 的 $\hat{S}$ 值组合出 patch 内的 fractional integral。

**Key Approximation**:
$$\int_{\Omega^+} G(\omega_i) O(x, \omega_i) \, d\omega_i = \int_{\tilde{\Omega}} G(\omega_i) \, d\omega_i \approx \mathcal{F}(x) \cdot \int_{\Omega^+} G(\omega_i) \, d\omega_i \quad \text{(Eq. 24)}$$

- $O(x, \omega_i) \in \{0, 1\}$: occlusion binary function (1 if hand blocks direction $\omega_i$ from point $x$)
- $\tilde{\Omega}$: 被遮挡的 patches
- $\mathcal{F}(x) = \sum_{\text{occluded patches}} \mathcal{F}(x, \cdot)$: 总 occluded fraction
- 右边是 closed-form: SG 的全 hemisphere 积分有解析解，乘以 occluded fraction 即可

**这是 paper 的 key insight**: 不做 Monte-Carlo ray-tracing，把 occlusion 离散化到 SG patch 上，每个 patch 的 fractional contribution 由 normalized ISG closed-form 给出。108 spheres 投影到 SG lobe 上，找出哪些 patches 被覆盖，求和 fractional contributions。

### 4.6 Final Illumination with Hand Indirect Lighting

hand 不仅 occlude light，还会 reflect skin color 到 object surface (indirect illumination)。paper 用一个 constant $L_i$ 表示:

$$L(\omega_i, x) = L_d(\omega_i) \cdot (1 - O(x, \omega_i)) + L_i \cdot O(x, \omega_i) \quad \text{(Eq. 25)}$$

- $L_d(\omega_i)$: environment 的 direct illumination (sum of 128 SGs)
- $L_i$: hand 间接照明，初始值是 generic skin color #E0AC69 (RGB ~ (224, 172, 105))
- $O(x, \omega_i)$: occlusion binary

**Intuition**: 在 occluded 方向，environment light 不能直接 reach，但 hand 反射的 skin color 替代。这是 simplified 二选一 model，不是真正的 multi-bounce rendering，但可以 capture 皮肤反射这个 dominant effect。

### 4.7 Cosine Term Approximation

为了把 cosine term $\omega_i \cdot n$ 也用 SG 表示 (这样整个 PBR integral 可以 closed-form):

$$\omega_i \cdot n \approx G(\omega_i; 0.0315, 32.7080, n) - 31.7003 \quad \text{(Eq. 26)}$$

- 这是一个以 $n$ 为 lobe direction 的 SG，peak value $0.0315$，sharpness $32.7080$
- 偏移 $-31.7003$ 使其 zero-crossing 正确

这种 trick 来自 PhySG，使整个 integrand 都是 SG，从而 PBR integral 可以 closed-form 求出。

### 4.8 Stage 2 Loss

$$\mathcal{L}_{rgb2} = \frac{1}{N_{ho}} \sum_{i=1}^{N_{ho}} ||c_i^{ho} - \hat{c}_i|| \quad \text{(Eq. 27)}$$

- $N_{ho}$: pixels in intersection of predicted object mask 和 ground truth $M_{ho}$ mask
- $c_i^{ho}$: predicted color via PBR
- $\hat{c}_i$: ground truth RGB

只对 visible object pixels (不被 hand occlude 的) 计算 RGB loss，避免 hand 区域干扰。

$$\mathcal{L}_{mask2} = \frac{1}{N_{no}} \sum_{j=1}^{N_{no}} \frac{\ln(1 + e^{-50 \cdot S_j^{no}})}{50} \quad \text{(Eq. 28)}$$

- $N_{no}$: pixels **outside** intersection of predicted object mask 和 ground truth $M_{obj}$
- $S_j^{no}$: SDF value at these pixels (positive outside object)
- 这是 IDR 风格的 soft L1 log loss for SDF

$$\mathcal{L}_{s2} = \mathcal{L}_{rgb2} + \lambda_{mask2}\mathcal{L}_{mask2} + \lambda_{eikonal2}\mathcal{L}_{eikonal2} \quad \text{(Eq. 29)}$$

- $\lambda_{mask2} = 100.0$
- $\lambda_{eikonal2} = 0.1$

---

## 5. Implementation Details

- 12 GB Nvidia GeForce RTX 2080 Ti GPU
- Stage 1: ~14 hours
- Stage 2: ~14 hours
- 总计 ~28 hours per scenario

**Stage 1 inputs**:
- Hand-object pixel-wise segmentation (SAM-Track, Grounding DINO, AOT, Decoupling Features 等)
- MANO pose estimation (End-to-End Human Pose Transformer)
- Object pose estimation (SuperGlue, HF-Net coarse-to-fine)

**Stage 2 inputs**:
- Stage 1 输出的 refined poses, object geometry, object mask, hand-object mask
- Object agnostic stage 1 但 stage 2 依赖 stage 1，所以整 pipeline 不是 object agnostic

---

## 6. Experiments

### 6.1 Datasets

**HO3D** (6 textured objects):
- Sugar box ("ShSu")
- Cracker box ("MC")
- Bleach cleanser ("ABF")
- Meat can ("GPMF")
- Mustard bottle ("SM")
- Drilling machine ("MDF")

Skip scissors, banana, mug 因为缺乏 texture detail for comparison。

**In-the-wild** (3 objects from HOLD paper):
- Bottle
- Kettle
- Rubik's cube

### 6.2 Quantitative Results

Table 1 的数据 (Average across 6 HO3D objects):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---------|--------|--------|---------|
| InvRender | 19.23 | 0.9301 | 0.154 |
| HOLD | 20.61 | 0.9354 | 0.101 |
| **TexHOI** | **20.96** | **0.9387** | **0.095** |

**Per-object analysis**:

| Object | TexHOI PSNR | HOLD PSNR | InvRender PSNR |
|--------|-------------|-----------|----------------|
| Sugar Box | 23.55 | 22.20 | 22.36 |
| Bleach Cleanser | 20.78 | 19.62 | 18.76 |
| Drilling Machine | 17.64 | 18.12 | 14.51 |
| Cracker Box | 19.88 | 19.52 | 17.64 |
| Mustard Bottle | 21.93 | 22.08 | 21.86 |
| Meat Can | 22.02 | 22.16 | 20.26 |

**注意 Drilling Machine** TexHOI (17.64) 比 HOLD (18.12) 还低！Paper 解释这是因为 drilling machine 的 surface 大部分是黑色 metal，specular 强烈，材质复杂。HOLD 通过直接 bake shadow 得到 pixel-wise 更近的 image，但 albedo 不真实。LPIPS 上 TexHOI (0.139) 比 HOLD (0.152) 好。

### 6.3 Ablation Studies

Table 2:

| Configuration | PSNR | SSIM | LPIPS |
|---------------|------|------|-------|
| w/o indirect illum | 20.87 | 0.9380 | 0.112 |
| w/o occlusion | 20.85 | 0.9351 | 0.113 |
| Full | 20.96 | 0.9387 | 0.095 |

**关键观察**: LPIPS 上 full model 显著最优 (0.095 vs 0.112/0.113)，说明 perceptual quality 显著提升。PSNR 差异小 (0.1dB)，因为 PSNR 是 pixel-wise metric，baked shadow 反而 pixel close to ground truth 但 perceptual 差。

**Ablation insight**:
- Without indirect illumination: hand shadow 区域变暗，因为 hand reflection 不被 modeling
- Without hand occlusion: hand grasp 区域出现 artifact，shadow 被 bake 进 albedo

---

## 7. Limitations

**Static hand relative to object**: 如果 hand 不动一直 grasp object 同一位置，hand 的 impression 会 bake 进 geometry 和 texture。原因是 pipeline 依赖 hand 的 perspective 变化和 movement 来 distinguish object 和 hand 的 contribution。

**Sphere inter-penetration**: 108 spheres 的 center 在 object surface 外，但 sphere 半径可能让 sphere 末端 overlap 进 object，尤其在 fingertip 附近，造成 occlusion 计算的小误差。

**Constant $L_i$**: 用一个 global RGB 常数代表 skin reflection，没有考虑 hand 不同部位 (palm vs back of hand) 和不同光照下 skin 反射的 variation。

---

## 8. Connections to Related Work — 更宽视角

### 8.1 Inverse Rendering 历史

Inverse rendering 三大子问题 (Marschner 1998):
1. **Inverse lighting**: predict environment light
2. **Inverse reflectometry**: predict BRDF / texture
3. **Shape reconstruction**: predict geometry

Traditional methods 只解其中一个，assumed 其他已知。Neural representation (NeRF, NeuS, DVGO, Instant-NGP, 3DGS) 开始 unify这三个。

**Web references**:
- NeRF: https://arxiv.org/abs/2003.08934
- NeuS: https://arxiv.org/abs/2106.10689
- DVGO: https://arxiv.org/abs/2206.05085
- Instant-NGP: https://arxiv.org/abs/2201.05989
- 3DGS: https://arxiv.org/abs/2302.07681

### 8.2 Physics-based Inverse Rendering

PBR equation (Eq. 1) 是 Kajiya 1986 rendering equation 的 simplified form (assuming no emission, no subsurface scattering)。

Recent works 通过 SG 或 Spherical Harmonics (SH) 近似 illumination 和 BRDF:
- **PhySG**: https://arxiv.org/abs/2104.01431 — SG-based surface rendering
- **InvRender**: https://arxiv.org/abs/2212.05038 — MLP visibility estimation
- **NeRFactor**: https://arxiv.org/abs/2106.09820 — Monte-Carlo ray-tracing visibility
- **NeRD**: https://arxiv.org/abs/2012.03918 — NeRF-based reflectance decomposition
- **RefNeRF**: https://arxiv.org/abs/2112.11625 — reflected radiance parametrization
- **NVDiffRec**: https://nvlabs.github.io/nvdiffrrec/
- **NeRO**: https://arxiv.org/abs/2305.17130 — glossy object reconstruction
- **NeRFactor**: https://arxiv.org/abs/2106.09820
- **NeILF**: https://arxiv.org/abs/2203.07622
- **Neural-PIL**: https://arxiv.org/abs/2110.03888
- **PS-NeRF**: https://arxiv.org/abs/2207.07006
- **SAMU-RAI**: https://arxiv.org/abs/2210.13589
- **L-Tracing**: https://arxiv.org/abs/2207.04036 — sphere tracing visibility
- **Neural-PBIR**: https://arxiv.org/abs/2305.15979
- **TensoIR**: https://arxiv.org/abs/2210.17527
- **TensoSDF**: https://arxiv.org/abs/2402.02771
- **MIRReS**: https://arxiv.org/abs/2406.16360
- **IRON**: https://arxiv.org/abs/2204.01010
- **Relightable 3D Gaussian**: https://arxiv.org/abs/2311.16043
- **IntrinsicAnything**: https://arxiv.org/abs/2404.11593

### 8.3 Hand-Object Interaction

**Compositional NeRF** for multi-object:
- Object-NeRF: https://arxiv.org/abs/2105.13591
- Compositional NeRF (Yang et al. ICCV 2021): https://arxiv.org/abs/2105.13591

**Hand-Object specific**:
- HOLD: https://arxiv.org/abs/2404.01804 — compositional NeRF for hand+object+background
- BundleSDF: https://arxiv.org/abs/2303.14130 — neural 6-DoF tracking
- DiffHOI: https://arxiv.org/abs/2307.11889 — diffusion-guided reconstruction
- Hand-NeRF (Guo et al.): https://arxiv.org/abs/2303.10807 — animatable interacting hands
- Hand-NeRF (Choi et al. ICRA 2024): https://arxiv.org/abs/2301.02595 — single RGB image reconstruction

### 8.4 Datasets

- **HO3D**: https://arxiv.org/abs/2007.02418 / http://ho3d.stevenhampali.com/
- **DexYCB**: https://arxiv.org/abs/2104.04784
- **ContactPose**: https://arxiv.org/abs/2003.02736
- **YCB**: https://arxiv.org/abs/1711.00199

### 8.5 Foundations

- **MANO model** (Romero et al.): https://arxiv.org/abs/2201.02610 — embodied hand model
- **Spherical Gaussians in graphics**: Iwasaki et al. 2012 (CGF) — ISG approximation
- **Power Diagram** for medial axis: Wang et al. SIGGRAPH Asia 2022 https://arxiv.org/abs/2211.16825
- **Mitsuba Renderer**: https://mitsuba-renderer.org/
- **SAM-Track**: https://arxiv.org/abs/2305.06558
- **SAM**: https://arxiv.org/abs/2304.02643
- **Grounding DINO**: https://arxiv.org/abs/2303.05499
- **AOT**: https://arxiv.org/abs/2104.04304
- **Decoupling Features** (Yang et al. NeurIPS 2022): https://arxiv.org/abs/2207.10160
- **End-to-End Human Pose Transformer** (Lin et al.): https://arxiv.org/abs/2012.04775
- **SuperGlue**: https://arxiv.org/abs/2002.08738
- **HF-Net / coarse-to-fine localization** (Sarlin et al.): https://arxiv.org/abs/1812.03586

---

## 9. Intuition Building — 关键 Insights

### 9.1 为什么 decouple into two stages?

如果同时优化 pose 和 material，会出现 **chicken-and-egg** 问题:
- Material prediction 需要 accurate pose (pixel alignment)
- Pose refinement 需要 material knowledge (texture features visible)
- 同时优化陷入 local minimum

Stage 1 用 volumetric NeRF (相对 robust to pose error) 求得粗略 geometry 和 accurate pose，stage 2 在已知 pose 下做 surface rendering (sensitive to pose) fine-tune material。

### 9.2 为什么 SG 而不是 SH (Spherical Harmonics)?

SG 的优势:
- SG 的 product 还是 SG (closed-form)
- SG 的 integral over hemisphere 有 closed-form
- Cosine term 可以被 SG 近似 (Eq. 26)
- ISG approximation (Iwasaki 2012) 让 patch-based occlusion 可计算

SH 的 disadvantage:
- SH product 是 triple tensor product，复杂
- SH 不易处理局部 occlusion
- Occlusion in SH basis 需要重新投影，每个 occlusion event 都需要 transform

### 9.3 为什么 108 spheres 而不是 mesh?

Mesh-based occlusion 需要 ray-tracing，对每个 object surface point 和每个 incoming direction，检查是否被 hand mesh intersect。108 spheres 把 occlusion 问题 discretize:
- Sphere 投影到 SG lobe 上是 closed-form geometric operation
- Patch-based fractional integration 用 normalized ISG closed-form
- 整个 occlusion computation 没有 ray-tracing，只需 matrix 操作

Sphere count 108 的选择: trade-off between accuracy (more spheres) 和 compute (fewer spheres)。MANO hand 15 joints + 5 fingers × ~3 phalanges 大概 ~20 segments，每 segment 几个 spheres 加起来 108 是合理选择。

### 9.4 为什么用 power diagram 而非简单 sphere packing?

Power diagram 给出 **deterministic** partition，每个 vertex 唯一属于一个 sphere 的 cell。当 MANO deform 时:
- vertex-to-sphere assignment 不变 (因为 cell 是按 canonical space 计算的)
- 但 sphere center 和 radius 通过 deformed vertices recompute

这是 **differentiable** 的: vertices 通过 MANO pose parameters 微分，sphere parameters 也随之微分，gradients 通过 sphere → SG occlusion → PBR loss 反传回 MANO pose。

### 9.5 Hand-Object Mask $M_{ho}$ 的妙用

$M_{ho}$ 通过强制 object color=1, hand color=0 在 volumetric rendering 中:
- 被手遮挡的 object region 中，hand 区域 absorb 所有 light (color=0 → density 起作用)，$M_{ho}=0$
- 没被遮挡的 object region: object color=1 propagate 出来，$M_{ho}=1$
- 因此 $M_{ho}$ 是 visible object mask，stage 2 只在这些 pixels 上做 RGB loss

这个 trick 是 **reusing the volumetric renderer** to get semantic mask without training separate network。

### 9.6 Cosine Term SG Approximation 的 magic

$$\omega_i \cdot n \approx G(\omega_i; 0.0315, 32.7080, n) - 31.7003$$

常数 0.0315, 32.7080, -31.7003 是通过拟合得到的。直觉: cosine function $\omega_i \cdot n$ 在 hemisphere (where $\omega_i \cdot n > 0$) 上是一个以 $n$ 为 peak 的 lobe，类似 SG。但 cosine peak 是 1 而 SG peak 是 $\mu$，所以用 scaled SG 加 offset 近似 cosine function shape。

这个 trick 让 PBR integral 中:
- Illumination $L$: sum of 128 SGs
- BRDF: product of constants + SG-like terms
- Cosine: SG

SG × SG × SG 的 hemisphere integral closed-form，于是整个 PBR equation closed-form 可微。

---

## 10. Future Work 的几个 interesting direction

paper 提到的 future work:

1. **Sophisticated skin reflection model**: 不用 constant $L_i$，而用 spatially-varying skin BRDF (考虑 melanin、hemoglobin 含量，subsurface scattering)

2. **3D Gaussian Splatting-based inverse rendering for HOI**: 替换 NeRF 表示为 3DGS，可能大幅提升 speed。Reference: Relightable 3D Gaussian https://arxiv.org/abs/2311.16043

3. **End-to-end 优化**: 消除 two-stage 设计，所有 parameters 同时 optimize。挑战是 stage 1 和 stage 2 的 gradient scale 不一致，需要 careful scheduling。

4. **Collision detection** for hand-object inter-penetration: 当前 sphere 表示假设 sphere 不 overlap with object，fingertip 附近会 violate。可以引入 differentiable physics simulator (e.g., DiffTaichi https://github.com/taichi-dev/difftaichi)

5. **Multiple hands interaction**: 当前 framework 设计为 single right hand。扩展到 two-hand interaction 需要处理 hand-hand occlusion in addition to hand-object occlusion。

6. **Temporal consistency**: 当前每 frame 独立做 PBR，video 中可能有 temporal flicker。加入 temporal smoothing regularizer。

---

## 11. Critical Analysis

### 11.1 论文 strengths

- First to explicitly model hand-induced shadow/reflection in object texture reconstruction
- 108 spheres + ISG 的 closed-form occlusion computation 是 elegant engineering，避免 ray-tracing 同时保持 accuracy
- Two-stage decoupling 是 well-motivated design choice
- Quantitative gains over HOLD 和 InvRender

### 11.2 论文 weaknesses (potential critique)

- **Two-stage 28 hours total** 比 NeRFactor (30 min/frame × N frames) 在 wall-clock 上未必快很多
- **Constant $L_i$ skin reflection**: 复杂场景下 skin 反射 varies，single RGB 无法 capture
- **108 spheres 固定 placement**: 不同 hand shape 下 sphere distribution 可能 sub-optimal
- **Drilling Machine case underperforms HOLD on PSNR**: 表明对 specular/dark surface 还是有 issue
- **Static hand limitation**: 如果 hand 一直 grasp 同一位置，baking effect 还在。这 reveal 了 framework 依赖 temporal diversity 而非真正的 hand-object disentanglement
- **Lacks comparison with recent 3DGS-based inverse rendering**: 2024 年很多 3DGS inverse rendering 出现，未对比

### 11.3 Possible extensions Karpathy might think about

考虑到 Karpathy 的背景，他可能对以下方向感兴趣:

- **Diffusion prior integration**: 用 Stable Diffusion 或 video diffusion 作 prior，对 occluded region inpaint albedo (类似 IntrinsicAnything 但 for HOI)
- **Self-supervised scaling**: 从 YouTube hand-object videos 大规模 scrape，用 TexHOI pipeline 训练 albedo predictor network (类似 DINO for texture)
- **Generative refinement**: train a generator on TexHOI outputs，让其作为 differentiable critic
- **Robot imitation learning**: high-fidelity object texture 可用于 simulator training，real-to-sim gap reduction

---

## 12. Summary

TexHOI 是 hand-object interaction texture reconstruction 的 **first-mover** 工作。其核心 contribution 不是 algorithmic breakthrough，而是 **problem formulation + engineering elegance**: 把 dynamic hand 这个长期被忽略的 occluder explicitly 模型化进 PBR pipeline。108 spheres + ISG closed-form 是基于 Iwasaki 2012 工作 的巧妙 reuse，避免 ray-tracing 的同时达到 reasonable accuracy。

Two-stage decoupling (NeRF for pose → PBR for material) 是 conservative 但 stable 的设计。量化 results 在 LPIPS 上显著提升 (0.095 vs 0.101 HOLD, 0.154 InvRender) 验证 perceptual quality gain。

未来工作方向清晰: 3DGS-based HOI inverse rendering, end-to-end optimization, sophisticated skin BRDF, multi-hand interaction, temporal consistency。这个领域还有大量空间。

**最终 reference 链接**:
- Project: https://alakhag.github.io/TexHOI-website
- HOLD (main baseline): https://arxiv.org/abs/2404.01804
- PhySG (SG-based PBR foundation): https://arxiv.org/abs/2104.01431
- InvRender (baseline + indirect illum): https://arxiv.org/abs/2212.05038
- Iwasaki 2012 (ISG approximation): https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2012.03092.x
- MANO: https://arxiv.org/abs/2201.02610
- HO3D: http://ho3d.stevenhampali.com/
- Mitsuba: https://mitsuba-renderer.org/
- NeRF: https://arxiv.org/abs/2003.08934
- IDR: https://arxiv.org/abs/2012.03907
- NeRFactor: https://arxiv.org/abs/2106.09820
- BundleSDF: https://arxiv.org/abs/2303.14130
- DiffHOI: https://arxiv.org/abs/2307.11889
- YCB Object: https://www.ycbbenchmarks.com/
- DexYCB: https://github.com/dexycb/dex-ycb-toolkit
- ContactPose: https://arxiv.org/abs/2003.02736
- SAM-Track: https://arxiv.org/abs/2305.06558
- Relightable 3DGS: https://arxiv.org/abs/2311.16043
- Kajiya Rendering Equation 1986: https://dl.acm.org/doi/10.1145/15922.15902
- IntrinsicAnything: https://arxiv.org/abs/2404.11593
