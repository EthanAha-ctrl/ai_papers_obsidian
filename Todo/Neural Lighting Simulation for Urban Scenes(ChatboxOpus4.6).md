## 一、核心问题与动机 (First Principles)

从**第一性原理**出发，self-driving vehicle (SDV) 的 camera-based perception 系统面临一个根本问题：

> **照明条件 (illumination) 是 outdoor scene appearance 的最大变量之一，但训练数据几乎不可能覆盖所有 lighting 组合。**

直觉上，同一条街道在正午阳光直射、阴天散射光、日落侧光下，camera 看到的 pixel intensity、shadow pattern、color temperature 完全不同。如果 perception model 只在晴天训练，它在阴天就可能把 shadow edge 误检为 obstacle，或者在低光照下漏检 vehicle。

**现有解决方案的不足**:
- **Graphics-based simulator** (如 CARLA [18], AirSim [68]): 手工建 asset，domain gap 大，diversity 有限
- **Data-driven simulator** (如 UniSim [89], Neural Scene Graphs [57]): 将 lighting "bake" 进 neural radiance field，无法改光照
- **Inverse rendering methods** (如 NeRF-OSR [64], FEGR [83]): 尝试分解 geometry/material/lighting，但 ill-posed，大规模 urban scene 下 decomposition 不准确
- **Image-based relighting** (如 style transfer): 缺乏 physical 的 3D consistency，没有正确的 shadow casting

**LightSim 的核心思路**: 把 **physically-based rendering (PBR)** 和 **learned neural deferred rendering** 结合起来——用 PBR 提供 physically-correct 的 lighting cue (shadow map, normal, AO 等)，再用 neural network 弥补 geometry/material decomposition 的不完美，产生 photorealistic 的 relit image。

---

## 二、系统架构总览

整个 pipeline 分为两大阶段：

```
Stage 1: Building Lighting-Aware Digital Twins
  ├── Neural Scene Reconstruction (Sec 3.1)
  │     → SDF-based neural field → textured mesh 提取
  ├── Neural Lighting Estimation (Sec 3.2)
  │     → Multi-camera panorama stitching → inpainting → LDR-to-HDR sky dome
  └── Learning (Sec 3.3)
        → Joint optimization of geometry/appearance + training inpainting/sky dome networks

Stage 2: Neural Lighting Simulation (Sec 4)
  ├── Scene editing (actor insertion/removal/modification)
  ├── Physically-based rendering → render buffers + shadow maps
  └── Neural Deferred Rendering → final relit image
```

---

## 三、Stage 1: Building Lighting-Aware Digital Twins

### 3.1 Neural Scene Reconstruction

**核心**: 使用 modified UniSim [89] 学习 scene 的 geometry 和 **view-independent diffuse color**。

Neural field 定义为:
$$\mathcal{F}: \mathbf{x} \mapsto (s, \mathbf{k}_d)$$

其中：
- $\mathbf{x} \in \mathbb{R}^3$: 3D spatial location
- $s \in \mathbb{R}$: **signed distance** (SDF value)，$s=0$ 定义 surface
- $\mathbf{k}_d \in \mathbb{R}^3$: **view-independent diffuse color** (RGB)

**为什么用 view-independent color？** 这是关键设计——传统 NeRF 学 view-dependent radiance（包含 specular highlight + lighting），但这会把 lighting "bake" 进去。LightSim 故意只学 diffuse base color，这样后面可以在 PBR 中用不同的 environment map 重新 shade。

**Scene decomposition**: 将 driving scene 分为:
- Static background $\mathcal{B}$: 用一个 neural field
- Dynamic actors $\{\mathcal{A}_i\}_{i=1}^{M}$: 用另一个 neural field

两者共享相同的 MLP 架构但参数独立，使用 **multi-resolution hash grid** [54] (Instant-NGP) 作为 spatial feature。这种 compositional representation 允许 **actor insertion/removal/modification**。

**Mesh extraction**: 训练完成后，用 **Marching Cubes** [47] 从 learned SDF 提取 mesh，再用 **Quadric Mesh Decimation** [21] 简化。Vertex color 通过 query appearance MLP 获得。Material 使用 Blender 的 **Principled BSDF** [10] (Disney BRDF model)，简化为 fixed base material。

**Training loss** (Eq. 3):
$$\mathcal{L}_{\text{scene}} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{lidar}} \mathcal{L}_{\text{lidar}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

$$\mathcal{L}_{\text{rgb}} = \frac{1}{|\mathcal{R}_{\text{img}}|} \sum_{r \in \mathcal{R}_{\text{img}}} \| C(r) - \hat{C}(r) \|^2$$

$$\mathcal{L}_{\text{lidar}} = \frac{1}{|\mathcal{R}_{\text{lidar}}|} \sum_{r \in \mathcal{R}_{\text{lidar}}} \| D(r) - \hat{D}(r) \|^2$$

其中:
- $\mathcal{R}_{\text{img}}$: camera ray 集合
- $\mathcal{R}_{\text{lidar}}$: LiDAR ray 集合
- $C(r)$/$\hat{C}(r)$: observed / predicted color for ray $r$
- $D(r)$/$\hat{D}(r)$: observed / predicted depth for ray $r$
- $\mathcal{L}_{\text{reg}}$: 包含 **Eikonal regularization** ($\|\nabla s\| = 1$，强制 SDF gradient 为 unit norm) 和 free space loss

**直觉**: LiDAR loss 提供 strong geometry supervision，camera loss 提供 texture 信息。两者互补——camera alone 在 outdoor unbounded scene 中很难得到 accurate depth，LiDAR 给出 precise depth but sparse。

### 3.2 Neural Lighting Estimation

这是整个系统最关键的模块之一。目标是从 SDV 的 multi-camera + GPS 数据中恢复 **HDR panoramic sky dome** $\mathcal{E}$。

#### Step 1: Panorama Reconstruction (Eq. 1)

$$I_{\text{pano}} = \Theta(I, D, P) = E(\pi^{-1}(I, D, P))$$

其中:
- $I = \{I_i\}_{i=1}^{K}$: $K$ 个 camera 的 images
- $D = \{D_i\}_{i=1}^{K}$: 从 extracted mesh $\mathcal{M}$ render 出的 depth maps, $D_i = \psi(\mathcal{M}, P_i)$
- $P = \{P_i\}_{i=1}^{K}$: camera projection matrices, $P_i \in \mathbb{R}^{3 \times 4}$
- $\pi^{-1}$: inverse projection (pixel → 3D world coordinate using depth)
- $E$: equirectangular projection (3D → panorama pixel)
- $\Theta$: 整体的 pixel-wise transformation

**过程**: 对每个 camera pixel $(u', v')$，用 depth 反投影到 3D，再投影到 equirectangular panorama 的 $(u, v)$。Sky region 的 depth 设为 infinity。Overlap 区域取平均。

SDV 的 multi-camera setup (如 PandaSet 的 6 个 camera) 通常覆盖 360° horizontal FoV，但 vertical FoV 有限（无法完全 cover sky）。

#### Step 2: Panorama Inpainting

使用 **DeepFillv2** [93] inpainting network 填补缺失的 sky region，得到完整的 360°×180° LDR panorama。

训练数据: **Holicity** [100] dataset (6k panorama images)。Training 时用 SDV camera intrinsics 生成 visibility mask 来 simulate incomplete panorama。

#### Step 3: LDR → HDR Sky Dome (Eq. 2)

$$\mathcal{E} = \text{HDRdecoder}(z_{\text{sky}}, [f_{\text{int}}, f_{\text{dir}}])$$
$$z_{\text{sky}}, f_{\text{int}}, f_{\text{dir}} = \text{LDRencoder}(L)$$

其中:
- $L$: completed LDR panorama
- $z_{\text{sky}} \in \mathbb{R}^{64}$: **sky appearance latent** (编码 sky color, cloud pattern 等)
- $f_{\text{int}}$: **peak sun intensity** (scalar, 编码 sun 的 HDR 强度)
- $f_{\text{dir}}$: **sun direction** (编码 sun 在 sky dome 上的 azimuth/elevation)

**为什么显式分离 sun intensity 和 direction？** 两个原因:
1. **Controllability**: 用户可以直接修改 sun direction 或 intensity 来 simulate 不同时间的光照
2. **Accuracy**: 当有 GPS + time of day 信息时，可以用 astronomical calculation 替换 encoder-predicted direction，得到更精确的 sun position

**Dual-encoder architecture**: 论文发现对于 cloudy sky，sun 不可见时 direction estimation 很难。因此使用两个 encoder——一个只在 sun visible 的 HDR 上训练（更准的 direction），另一个在所有 HDR 上训练（更 robust 的 intensity + sky latent）。

**训练数据**: 400 HDR sky images from **HDRMaps** [24]。Random distortion (exposure scaling, rotation, flipping) + tone-mapping 产生 LDR-HDR pairs。Loss 包含 L1 angular loss, L1 peak intensity, L2 HDR reconstruction loss (在 log space)。

---

## 四、Stage 2: Neural Lighting Simulation

### 4.1 Scene Editing (Augmented Reality Representation)

由于 scene representation 是 **compositional** 的（background + individual actors），可以:
- **Remove** actors: 只 render background neural field
- **Insert** actors: 添加 reconstructed actor 或 CAD model
- **Modify** trajectories: 改变 actor 的 pose sequence
- **Change** SDV viewpoint: 修改 camera extrinsics

这些 edit 产生 augmented reality representation $\{\mathcal{M}', \mathcal{E}_{\text{src}}, I'_{\text{src}}\}$。

### 4.2 Physically-Based Rendering

将 textured mesh $\mathcal{M}$ 放入 **Blender Cycles** [7] (path-tracing renderer)，在 target environment map $\mathcal{E}_{\text{tgt}}$ 下 render:

- **Render buffers** $I_{\text{buffer}} \in \mathbb{R}^{h \times w \times 8}$: position (3ch), depth (1ch), normal (3ch), ambient occlusion (1ch)
- **Rendered image** $I_{\text{render}}|_{\mathcal{E}}$: full PBR rendering result
- **Shadow ratio map**: $S = I_{\text{render}} / \tilde{I}_{\text{render}}$，其中 $\tilde{I}_{\text{render}}$ 是关闭 shadow visibility ray 后的 rendering

Shadow ratio map $S$ 编码了 **shadow 的空间分布和强度**。$S$ 接近 1 的地方没有 shadow，接近 0 的地方是 deep shadow。这是一个非常巧妙的 representation——通过提供 source shadow map $S_{\text{src}}$ 和 target shadow map $S_{\text{tgt}}$，network 知道哪些 shadow 需要 remove (source有target没有) 和哪些需要 add (source没有target有)。

### 4.3 Neural Deferred Rendering (核心创新)

**Deferred rendering** [16, 65] 是 real-time graphics 中的经典技术：先 render geometry info (G-buffer)，再在 screen-space 做 lighting。LightSim 借鉴这个思想但引入 **learned component**。

**RelitNet** (Eq. 4):
$$I_{\text{tgt}} = \text{RelitNet}([I_{\text{src}}, I_{\text{buffer}}, S_{\text{src}}, S_{\text{tgt}}], [\mathcal{E}_{\text{src}}, \mathcal{E}_{\text{tgt}}])$$

**架构** (Fig. A11): 基于 **U-Net** [63] 的 image-to-image network，包含四个组件:

1. **Image Encoder**: 输入 = source RGB $I_{\text{src}}$ + render buffers $I_{\text{buffer}}$ (AO, normal, depth, position) + shadow ratio maps $\{S_{\text{src}}, S_{\text{tgt}}\}$。总共约 $(3 + 8 + 3 + 3) = 17$ channels。
2. **Lighting Encoder**: 分别 encode source 和 target HDR sky dome 为 latent vectors
3. **Latent Fuser**: Image feature → Conv2D → Linear → concat with lighting latents → upsample
4. **Rendering Decoder**: 从 fused latent 逐步 upsample 产生 final relit image $\hat{I}_{\text{tgt}} \in \mathbb{R}^{H \times W \times 3}$

**直觉**: 这个 network 本质上在做 **image translation conditioned on physical rendering cues**。Source image 提供 high-frequency texture 和 real-world appearance；render buffers 提供 geometric context (哪里是什么角度的 surface)；shadow maps 告诉 network shadow 应该怎么变；HDR sky domes 提供 global lighting context。Network 学会 "根据 physical cues 来 modify source image 的 lighting appearance"。

### 4.4 Training Strategy (关键设计)

训练面临一个经典鸡生蛋问题：我们没有同一场景在不同 lighting 下的 real image pairs！LightSim 用 **混合训练** 解决:

**Data Pair 1 (Sim-Sim)**: $I_{\text{render}}|_{\mathcal{E}_{\text{src}}} \to I_{\text{render}}|_{\mathcal{E}_{\text{tgt}}}$
- 用 PBR renderer 在同一个 digital twin 上 render 两个不同 lighting 的 synthetic image
- 教 network 学习 **lighting transfer 的物理规律**

**Data Pair 2 (Sim-Real)**: $I_{\text{render}}|_{\mathcal{E}_{\text{src}}} \to I_{\text{real}}$
- 将任意 lighting 下的 synthetic image 映射回 original real image (用 estimated source lighting)
- 教 network 学习 **bridging sim-to-real gap**，消除 mesh artifact、不完美 material 等

**Data Pair 3 (Identity)**: 当 $\mathcal{E}_{\text{src}} = \mathcal{E}_{\text{tgt}}$ 时，output 应等于 input
- 教 network **self-consistency**，避免在 same lighting 下引入 artifact

### 4.5 Loss Function (Eq. 5)

$$\mathcal{L}_{\text{relight}} = \frac{1}{N} \sum_{i=1}^{N} \left( \underbrace{\|I_{\text{tgt}}^i - \hat{I}_{\text{tgt}}^i\|^2}_{\mathcal{L}_{\text{color}}} + \lambda_{\text{lpips}} \underbrace{\sum_{j=1}^{M} \|V_j(I_{\text{tgt}}^i) - V_j(\hat{I}_{\text{tgt}}^i)\|^2}_{\mathcal{L}_{\text{lpips}}} + \lambda_{\text{edge}} \underbrace{\|\nabla I_{\text{tgt}}^i - \nabla \hat{I}_{\text{tgt}}^i\|^2}_{\mathcal{L}_{\text{edge}}} \right)$$

其中:
- $N$: training images 数量
- $I_{\text{tgt}}^i$ / $\hat{I}_{\text{tgt}}^i$: ground truth / predicted target image
- $V_j(\cdot)$: pre-trained **VGG** network [97] 的第 $j$ 层 feature extraction（perceptual loss 的标准做法）
- $\nabla I$: image gradient，用 **Sobel-Feldman operator** [70] 近似
- $\lambda_{\text{lpips}} = 1$, $\lambda_{\text{edge}} = 400$

**三个 loss 的作用**:
- $\mathcal{L}_{\text{color}}$: pixel-level accuracy
- $\mathcal{L}_{\text{lpips}}$: perceptual similarity，防止 blurry output（VGG feature 空间中的匹配比 pixel-level L2 更符合人类视觉）
- $\mathcal{L}_{\text{edge}}$: **content-preserving loss**，保留 source image 的 high-frequency edge detail（防止 relighting 过程中 fine structure 被 smooth 掉）

---

## 五、实验结果

### Dataset
- **PandaSet** [87]: 103 urban scenes, San Francisco, 8s each, 80 frames @10Hz, 6 cameras (1920×1080), 64-beam LiDAR
- **nuScenes** [11]: 1000 scenes, Boston/Singapore, ~40 frames @2Hz, 6 cameras, 32-beam LiDAR

### 5.1 Perceptual Quality (Table 1)

| Method | FID↓ | KID (×10³)↓ |
|--------|------|-------------|
| Self-OSR [94] | 124.8 | 107.1±4.3 |
| NeRF-OSR [64] | 143.9 | 94.0±7.5 |
| Color Transfer [60] | **85.4** | 29.5±4.3 |
| EPE [61] | 93.0 | 56.0±5.0 |
| **LightSim (Ours)** | 87.1 | **30.4±4.0** |

**分析**: Color Transfer 的 FID 最好，但它只调 global color histogram，不能产生 physically-correct shadows。LightSim 的 FID/KID 接近 Color Transfer 但提供了真正的 3D-aware relighting (shadow casting, highlight changes)。

在 47-sequence 大规模评估中 (Table A3)，LightSim FID=52.3，KID=16.0，和 Color Transfer (50.1, 18.0) 非常接近。

### 5.2 Downstream Perception Training (Table 2)

这是论文最 impactful 的实验——用 **BEVFormer** [44] 做 camera-based 3D object detection:

| Model | mAP (%) |
|-------|---------|
| Real only | 32.1 |
| Real + Color aug. | 33.8 (+1.7) |
| Real + Sim (Self-OSR) | 30.3 (−1.8) |
| Real + Sim (EPE) | 32.5 (+0.4) |
| Real + Sim (Color Transfer) | 35.1 (+3.0) |
| **Real + Sim (LightSim)** | **36.6 (+4.5)** |

**关键 insight**: 
- Self-OSR 的 artifact 太严重，反而 **harm** 了 perception performance (−1.8 mAP)
- LightSim 带来了 **+4.5 mAP** 的显著提升，证明 realistic lighting simulation 对 perception robustness 有实际价值
- 训练集 (68 city snippets) 和验证集 (35 suburban snippets) 有不同 lighting distribution，LightSim 帮助 bridge 这个 gap

### 5.3 Lighting Estimation (Table A6)

| Method | Input | Angular Error↓ |
|--------|-------|----------------|
| SOLDNet [74] | Limited-FoV | 69.98° |
| NLFE [82] | Limited-FoV | 78.29° |
| NLFE* [82] | Panorama | 47.39° |
| **Ours (no GPS)** | Panorama | **20.01°** |
| Ours (with GPS) | Panorama | **3.78°** |

**直觉**: 从 single limited-FoV image 估计 sun direction 几乎不可能（FoV 可能根本看不到 sun），但用 multi-camera panorama + GPS 就能极大提升精度。

### 5.4 Ablation Study (Table A4)

关键发现:
- 去掉 **rendering buffers**: FID 从 55.4 升到 109.8（network 无法 synthesize correct lighting effects without physical cues）
- 去掉 **sim-real training pairs**: FID 从 55.4 升到 60.9（realism 下降）
- 去掉 **identity pairs**: FID 从 55.4 升到 62.5（self-consistency 变差）
- 去掉 **edge loss**: 出现 mesh-like artifact（high-frequency detail 丢失）

---

## 六、Limitations 与 Failure Cases

1. **Baked shadows**: View-independent reconstruction 会把 shadows bake 到 diffuse color 中。在 strong directional sunlight 下，neural deferred rendering 无法完全 remove baked shadows（Fig. A26）
2. **No nighttime support**: HDR sky dome 只建模 sun + sky，无法处理 street lights、vehicle headlights 等 local light sources
3. **Fixed materials**: 没有学习 per-surface material properties (roughness, metalness)，导致 specular reflection 不准
4. **Rendering speed**: 依赖 Blender Cycles (path-tracing)，生成 render buffers 很慢
5. **Lens effects**: 无法处理 lens flare、vignetting 等 camera-specific optical effects

---

## 七、与相关工作的关键区别

| 方面 | UniSim [89] / SUDS [77] | FEGR [83] | NeRF-OSR [64] | **LightSim** |
|------|------------------------|-----------|---------------|-------------|
| Lighting decomposition | ✗ (baked) | ✓ (per-scene optimization) | ✓ (per-scene optimization) | ✓ (feed-forward network) |
| Dynamic scenes | ✓ | ✗ (static only) | ✗ (static only) | ✓ |
| Scalable (多 scene) | ✓ | ✗ | ✗ | ✓ |
| Neural deferred rendering | ✗ | ✗ | ✗ | ✓ |
| Shadow editing | ✗ | ✓ | ✗ | ✓ |
| Actor insertion | ✗ (no lighting) | ✓ | ✗ | ✓ (lighting-aware) |

---

## 八、核心 Intuition 总结

1. **Perfect decomposition is too hard, but approximate decomposition + learned refinement is effective.** LightSim 不追求完美的 intrinsic decomposition (geometry/material/lighting)，而是用 PBR 提供 approximately-correct lighting cues，再让 neural network 学会 "修正" imperfection。这是一种 **hybrid physics-learning** 的哲学。

2. **The key to realistic relighting is providing the right physical context.** Neural deferred rendering 之所以有效，是因为 render buffers (normal, AO, shadow maps) 告诉 network "physics 认为这个 pixel 应该怎么变"，network 只需要学习 "如何在 real image 上 apply 这些 changes while maintaining realism"。

3. **Mixed sim-real training breaks the chicken-and-egg problem.** 我们没有 real paired relighting data，但我们有 (a) 从 digital twin 生成的 synthetic pairs (正确的 lighting physics but imperfect appearance) 和 (b) synthetic-to-real pairs (bridge the domain gap)。两者结合使 network 既学会 physical relighting 又保持 photorealistic。

4. **Multi-sensor fusion is crucial for outdoor lighting estimation.** Single limited-FoV camera 几乎不可能准确估计 HDR sun intensity 和 direction。但 multi-camera panorama + GPS + time of day 的组合可以将 angular error 从 ~70° 降到 ~4°。

---

## 九、相关参考资源

- **UniSim**: [https://waabi.ai/unisim/](https://waabi.ai/unisim/) — LightSim 的 neural rendering backbone
- **Instant-NGP** (multi-resolution hash grid): [https://nvlabs.github.io/instant-ngp/](https://nvlabs.github.io/instant-ngp/)
- **NeRF-OSR**: [https://github.com/r00tman/NeRF-OSR](https://github.com/r00tman/NeRF-OSR) — outdoor relighting baseline
- **FEGR**: Wang et al., CVPR 2023 — concurrent inverse rendering work for urban scenes
- **UrbanIR**: Lin et al., 2023 — another concurrent urban inverse rendering work
- **BEVFormer**: [https://github.com/fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer) — downstream detection model
- **PandaSet**: [https://scale.com/open-av-datasets/pandaset](https://scale.com/open-av-datasets/pandaset)
- **DeepFillv2**: [https://github.com/zhaoyuzhi/deepfillv2](https://github.com/zhaoyuzhi/deepfillv2) — inpainting network
- **Disney BRDF/Principled BSDF**: Burley, SIGGRAPH 2012 — [https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf)
- **Blender Cycles**: [https://www.blender.org/](https://www.blender.org/) — path-tracing engine used for PBR
- **HDRMaps**: [https://hdrmaps.com/](https://hdrmaps.com/) — HDR sky dome training data
- **EPE (Enhancing Photorealism Enhancement)**: [https://github.com/isl-org/PhotorealismEnhancement](https://github.com/isl-org/PhotorealismEnhancement)

---

这篇论文的价值在于它展示了一个 **end-to-end practical system** 的完整设计——从 sensor data 到 lighting-aware digital twin 到 controllable relighting 到 downstream perception improvement。它不在任何单一技术点上 claim SOTA（lighting estimation、inverse rendering、neural rendering 各自都有 concurrent work 更好），但它把这些模块 **系统性地整合** 在一起，并通过 neural deferred rendering 优雅地处理了各模块 imperfection 的 accumulation，在实际 downstream task 上取得了显著 gain (+4.5 mAP)。这正是 Waabi 这类 autonomous driving 公司的典型研究风格——**engineering-driven, system-level, practically impactful**。