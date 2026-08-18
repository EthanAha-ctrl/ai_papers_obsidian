---
source_pdf: ABot-Earth 0.5 Generative 3D Earth Model.pdf
paper_sha256: b84d9f5ad231e381893d327a334536ad7637f41a53d84e0a0757783ef602c60e
processed_at: '2026-08-17T23:41:16-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，好的，我换个更直觉的方式跟你聊，就像我们在白板前讨论一样。

---

## 一句话总结

**输入一张 satellite image，输出一片可以飞进去逛的 3D 城市，每平方公里不到 10 分钟。**

就这么简单。剩下的全是工程细节和算法巧思。

---

## 为什么这件事难

你想生成一只猫的 3D 模型，现在有很多方法能做得很好了，TRELLIS, Hunyuan3D, Seed3D 这些都在 object level 上表现不错。但你把同样的思路搬到 Earth scale，立刻撞墙：

**第一个墙：Scale。** 一只猫是 $1 \text{ meter}$ 量级，一个城市是 $10 \text{ km}$ 量级，整个地球的 built-up area 是 $800{,}000 \text{ km}^2$。数据量差了十几个数量级。你不可能用一个 monolithic network 一次性生成整个地球，必须 tile-based 分块处理，但分块就会产生 seam。

**第二个墙：Representation。** Object generator 基本都输出 mesh。但 outdoor scene 里有树、有水、有半透明的东西，mesh 天生不擅长表达这些 non-manifold 结构。3DGS 天生擅长，但 3DGS 是 unstructured 的，百万个 Gaussian primitives 散在空间里，你怎么 learn 一个 generative model 直接输出这堆 primitives？这跟生成一个 structured mesh 完全不是一个难度。

**第三个墙：Conditional input 的 domain gap。** 你训练的时候有 aerial view 的重建数据，但推理的时候只有 satellite top-down image。Satellite image 的 resolution, sensor, 大气条件都不一样。你训练的 conditional distribution 和推理时的 conditional distribution 是 misaligned 的。

**第四个墙：Deployment。** 就算你算法上解决了，$3.2 \text{ trillion}$ 个 Gaussian primitives，怎么在 web 端 real-time 渲染？Google Earth 用的是 photogrammetry mesh + LOD，你这是 trillion-scale 的 Gaussians，没有任何现成引擎能直接吃下。

这四个墙就是 ABot-Earth 0.5 要推倒的东西。

---

## 他们的核心思路

### 思路 1：直接在 3DGS space 里做 generation

不绕弯子，不先生成 mesh 再转 3DGS，不先生成 NeRF 再转 3DGS。直接学一个 model，输入 satellite image，输出 Gaussian primitives。

为什么必须这样？因为你最终要的是 3DGS 的 rendering quality 和 speed。如果你中间经过 mesh，你就会 lose 掉那些 foliage, water surface, specular facade 的细节。这些细节恰恰是 outdoor scene 真实感的核心。

他们用的策略叫 **compression-generation paradigm**。先把百万级别的 primitives 压缩到一个 compact latent space，然后在 latent space 里做 diffusion 或 flow matching 生成。这跟 Latent Diffusion 的思路一样，只不过 latent 对应的不是 2D image patches，而是 3D Gaussian primitives 的 structured representation。

### 思路 2：Multi-LOD 直接在 decoder 里生成

传统的 LOD 是后处理：先生成高精度模型，再 downsample 出低精度版本。但 downsample 会 lose quality。

ABot-Earth 的做法是让 decoder **直接输出 hierarchical structure**。高精度层（zoom 17-19）由 model 原生生成，低精度层（zoom 14-16）通过 statistical decimation 从 zoom-17 数据中导出。

这里有个很优雅的细节：低精度层用的是 **Bhattacharyya distance** 来决定哪些 Gaussians 可以合并或丢弃。Bhattacharyya distance 衡量两个 Gaussian distribution 的相似度，同时考虑了 spatial position 和 shape (covariance)。如果两个 Gaussians 离得近、shape 也像，就说明它们在视觉上是冗余的，可以合并成一个。这个操作是纯解析的，不需要跑 network，所以可以直接扔给 CPU 并行处理，GPU 继续 inference，完全不增加 latency。

$$D_B(p, q) = \frac{1}{8} (\mu_p - \mu_q)^T \Sigma^{-1} (\mu_p - \mu_q) + \frac{1}{2} \ln \left( \frac{|\Sigma|}{\sqrt{|\Sigma_p| |\Sigma_q|}} \right)$$

- $\mu_p, \mu_q$：两个 Gaussian 的 3D 位置
- $\Sigma_p, \Sigma_q$：两个 Gaussian 的 covariance（由 scale 和 rotation 决定）
- $\Sigma = \frac{\Sigma_p + \Sigma_q}{2}$：平均 covariance

第一项衡量位置差异，第二项衡量 shape 差异。这比简单的 Euclidean distance 聪明得多，因为两个相距较远但 shape 很大的 Gaussian 可能在视觉上是重叠的，应该合并。

### 思路 3：Sliding-window inference 解决 seam 问题

单次 inference 在 A100 上能处理 $1.6\text{km} \times 1.6\text{km}$ 的区域。但地球上有 $312{,}500$ 个这样的 tile，你需要把它们拼起来。

naive 的拼接会在 boundary 产生 visible seam——几何不连续、纹理不匹配。他们的 sliding-window strategy 在 generation 阶段就考虑了 overlap region，让相邻 tile 在 overlap zone 互相 influence。这可能是在 latent space 做 blending，也可能是在 Gaussian space 做 weighted merging。具体实现 paper 没有完全展开，但核心思想是：seamlessness 是 generation 时就要考虑的，不是 post-processing 能修好的。

### 思路 4：VLM-based harness 解决 domain gap

训练时，conditional input 是从 3DGS scene 渲染出来的 simulated satellite view——干净、consistent、resolution 可控。

推理时，conditional input 是 real-world satellite image——可能有云、有 atmospheric haze、resolution 不统一、sensor 不一样。

他们引入了一个 VLM-based harness 来 bridge this gap。我的猜测是：VLM 分析 real satellite image，提取出 semantic context（这是城市还是自然地形？什么气候带？什么季节？），然后把这些信息转化为 conditional signal 注入 generative model。这样即使 input image 质量差，model 也能 generate 出符合该区域地理特征的 plausible 3D scene。

---

## Data Pipeline：垃圾进垃圾出，所以他们先造高质量数据

这是整个 system 的地基。没有高质量 training data，再好的 generative model 也白搭。

他们的 data engine 叫 **ABot-3DGS**，输入 multi-source imagery，输出 city-scale 3DGS scenes。

**Data sources：**
- **Multi-stereo satellite**：DFC 2019 等公开数据 + proprietary 数据。用 FromOrbit2Ground 模块，先 Z-Monotonic SDF 恢复 geometry，再 diffusion 补 facade texture。
- **Aerial**：high-res oblique imagery，core training source。可以选配 LiDAR 和 photogrammetric mesh 作为 auxiliary geometric prior。
- **Urban**：drone + street-view，补充 low-altitude facade 细节。

**Cross-view fusion：** Aerial 提供 broad coverage，urban 提供 fine-grained facade detail，satellite 提供 global scalability。三种数据 cross-view matched 之后联合重建。

**Quality assessment 三层过滤：**
1. **Tile-level**：PSNR/SSIM/LPIPS + VLM perceptual score + geometric accuracy + spatial completeness。不合格的回炉或丢弃。
2. **View-level**：先滤掉 low accumulated opacity 的 view（void region），再用 VLM 评估 texture sharpness 和 artifact。
3. **Dataset-level**：stratified sampling 平衡 scene diversity（防止全是高楼城市），embedding space clustering 做 semantic deduplication（防止 mode collapse）。

这个 data pipeline 的直觉是：**3DGS reconstruction 本身就不完美，floaters, artifacts, under-observed regions 都会污染 training data。如果你不过滤就直接训，model 会学到这些 artifact，生成出来的 scene 也会满是 floaters。** 所以 VLM 在这里充当了一个 scalable 的 automated human annotator，帮你把 garbage filter 掉。

---

## Deployment：从 algorithm 到 planetary service

这部分是 paper 里最 hard-core 的工程内容。

### Production pipeline

- **Hardware**：1000 张 A100 GPU
- **Single tile inference**：~25 分钟
- **Total tiles**：~312,500 个
- **Total production time**：<10 天
- **Output**：~3.2 trillion Gaussian primitives

**Georeferencing 的关键细节：** Web Mercator (EPSG:3857) 在高纬度有 areal distortion。如果你直接把 Web Mercator tile 喂给 model，高纬度地区的 GSD (Ground Sampling Distance) 会和训练时不一致，导致 geometry 和 texture 都走样。他们的解法是先把 Web Mercator tile mosaic 成连续 geographic image，再做 isotropic resampling 到统一 GSD。这个细节很关键，否则你在赤道和北纬 60 度生成的 scene quality 会有肉眼可见的差异。

### Rendering pipeline (EarthScape)

3.2 trillion primitives，consumer GPU 根本渲染不动。三步走：

**Step 1：Geographic Alignment。** 每个 block 用 affine transformation 恢复到 EPSG:3857，然后建立 ENU (East-North-Up) local tangent plane 坐标系。所有 Gaussian 的 position, rotation quaternion, scaling 统一变换到 ENU frame。这样每个 block 内部是 meter-scale 的局部坐标系，适合渲染引擎，同时通过 ENU origin 锚定全球坐标。

**Step 2：LOD Reorganization。** 重新 partition 到标准 map tile hierarchy (zoom 14-19)。高精度层 model 原生生成，低精度层 Bhattacharyya decimation。构建两级 spatial index：explicit `tileset.json` (OGC 3D Tiles 标准) + implicit `{zoom}/{x}/{y}` path convention 方便 CDN caching。

**Step 3：Rendering Scheduling。** 集成到 Amap Yunjing 渲染引擎。每帧根据 camera viewport 动态计算需要的 tile set 和 precision level。近处 zoom 17-19，远处 zoom 14-15，level 之间 smooth fade transition。复用引擎的 frustum culling 和 async streaming infrastructure。

---

## Evaluation：FID 从 69.5 降到 16.1

| Method | FID ↓ | KID ↓ |
|---|---|---|
| CityDreamer | 97.3 | 0.096 |
| GaussianCity | 86.9 | 0.090 |
| EarthCrafter | 69.5 | 0.061 |
| **ABot-Earth 0.5** | **16.1** | **0.006** |

FID 从 69.5 到 16.1，这是 4x 的提升。而且他们的 GT 是 real-world 3DGS reconstruction，不是 synthetic dataset，建模难度更高。

跟 Google Earth 比：
- **Coverage**：Google Earth 只覆盖 scanned region，ABot-Earth 只要有 satellite image 就能生成，infinite coverage。
- **Efficiency**：Google Earth 更新需要数月到数年，ABot-Earth 每平方公里 <10 分钟。
- **Geometry/Texture fidelity**：Google Earth 仍然更强，因为有多年的 photogrammetry 优化 + Manhattan-world assumption + manual post-processing。但 ABot-Earth 在 overall aesthetics 上得分更高，因为 generative model 天生擅长 lighting harmony。

他们用了一个很到位的类比：**现在的 ABot-Earth 就像第一代的 LRM 或 CLAY，跟手工建模比还有差距，但这个差距不是 fundamental 的，会随着 scaling 逐步缩小。**

---

## Hybrid landmark enhancement：最聪明的 product decision

生成一个普通城市 block，只要 plausible 就够了。但生成 Eiffel Tower 或 Colosseum，用户脑子里有一个非常具体的 mental image，差一点都不行。

ABot-Earth 的解法：**生成 context，reconstruct landmark。** 用 COLMAP 从 crowd-sourced imagery 重建 high-fidelity landmark 3DGS model，然后 geo-register 并 composite 到 generative environment 中。

这个思路的直觉是：**Generative model 的优势是 scalability 和 plausibility，reconstruction 的优势是 accuracy。把两者的长处结合，用 generation 填充 vast context，用 reconstruction 锚定 iconic landmark。** 这也是 native 3DGS representation 的一个 bonus——因为生成的和重建的都是 3DGS format，天然 composable，不需要 format conversion。

---

## 对 Embodied AI 的意义

这是我觉得最 exciting 的部分。

现在的 UAV simulation（AirSim, FlightGoggles）都用 synthetic assets，sim-to-real gap 很大。UAV 在 synthetic world 里训练的 policy，迁移到 real world 会 fail，因为 texture distribution, lighting, geometry detail 都不一样。

ABot-Earth 提供了一个 **photorealistic, 3D structured, infinite-scale world model**：
- **Photorealistic**：从 real-world 3DGS reconstruction 学到的 prior，生成的 scene 在 visual distribution 上接近 real world。
- **3D structured**：不是 2D video，是真 3D Gaussians，UAV 可以在 3D space 中 fly around，获取任意 viewpoint 的 observation，做 occlusion reasoning 和 3D path planning。
- **Infinite-scale**：只要有 satellite image 的地方都能生成，包括 unmapped developing countries 和 natural terrain。
- **Closed-loop ready**：UAV action → environment state → sensor observation，完整的 closed-loop simulation loop。

这本质上是一个 **低成本、高覆盖、photorealistic 的 3D simulation sandbox**，对于 drone navigation, obstacle avoidance, exploration 的 research 有直接价值。

---

## Scaling law 的猜想

Paper 在 conclusion 提到要系统性验证 outdoor 3D scene generation 的 scaling laws。几个维度值得期待：

1. **Data scale**：训练数据从 300 cities 扩展到 1000+ cities，FID 会不会按 power law 下降？
2. **Model size**：从当前的 model size 扩大 10x, 100x，generation quality 如何变化？
3. **Input resolution**：从 4K satellite image 扩展到 8K, 16K，单次 inference 覆盖面积从 $2.56\text{km}^2$ 扩大到 $10\text{km}^2$，sliding-window 的 seam 问题会不会自然减轻？
4. **Output fidelity**：从当前接近 Google Earth 到超越 Google Earth，需要多大的 compute？

这些问题的答案会决定 generative 3D earth modeling 这个方向的天花板在哪里。

---

## 相关联想

- **Sora as world model vs. 3DGS as world model**：Sora 是 2D pixel space + temporal coherence，缺乏 3D structure。3DGS world model 有真 3D structure，agent 可以在里面做 spatial reasoning。两者最终可能 converge。
- **NeRF vs. 3DGS in large-scale generation**：NeRF 的 volume rendering 太慢，不支持 real-time。3DGS 的 rasterizer 天生 fast。在 generative setting 下，3DGS 还有一个巨大优势：primitives 是 explicit 的，可以直接 edit, composite, merge。NeRF 的 implicit field 很难做这些操作。
- **Google Earth 的 photogrammetry pipeline vs. generative pipeline**：Photogrammetry 是 "capture everything then reconstruct"，cost 跟 area 成正比。Generative 是 "learn prior then imagine"，cost 跟 area 几乎无关（inference cost 很低）。这是 paradigm shift。
- **Street-level extension**：Paper 提到下一步要从 aerial 3D 下沉到 street-view level。参考他们之前的工作 Sat3DGen [22]，已经证明了从 single satellite image 生成 street-level 3D scene 的可行性。如果 ABot-Earth 能做到 aerial + street-level 的 unified generation，那就是真正的 full-stack 3D earth model。

---

## 参考链接

- [3D Gaussian Splatting 原始论文](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [CityGaussian - Large-scale 3DGS rendering](https://arxiv.org/abs/2403.14555)
- [CLoD-GS - Continuous LOD for 3DGS](https://arxiv.org/abs/2510.09997)
- [TRELLIS - Structured 3D latents](https://arxiv.org/abs/2412.01501)
- [BlockFusion - Sliding-window latent extrapolation](https://arxiv.org/abs/2404.00918)
- [OGC 3D Tiles Specification](https://www.ogc.org/standard/3dtiles/)
- [AirSim - UAV simulation](https://arxiv.org/abs/1705.05065)
- [Sat3DGen - Street-level 3D from satellite](https://openreview.net/forum?id=E7JzkZCofa)
- [EarthCrafter - Dual-sparse latent diffusion for earth generation](https://arxiv.org/abs/2403.12465)
- [GaussianCity - Generative 3DGS for city generation](https://arxiv.org/abs/2406.06526)

---

总结一句人话：**他们把"给一张卫星图，还你一个 3D 世界"这件事做成了，而且做到了 planet scale，还能在浏览器里实时逛。** 算法上直接在 3DGS space 生成，工程上搞定了 trillion-scale 的 LOD 和 streaming，产品上用 hybrid 策略把 landmark fidelity 也保住了。这对 embodied AI simulation 是 infra-level 的突破。

---

Andrej, 非常高兴能与你深入探讨这篇来自 AMAP CV Lab 的工作。ABot-Earth 0.5 是一个极具野心的 system, 它试图将 Earth-scale 的 3D generation 从 expensive 的 photogrammetry pipeline 转化为基于 generative model 的 real-time inference pipeline。这篇 paper 的核心价值在于它不仅提出了算法, 还展示了一个完整的、能够部署在 planetary scale 的工程系统。

以下我将从 representation, data pipeline, method, system engineering 到 sim-to-real 的 intuition 为你进行深度拆解。

---

### 1. Representation: Native 3DGS Generative Framework

在当前 3D generation 领域, 大多数 SOTA 方法 (如 TRELLIS, Hunyuan3D) 都依赖于 mesh 或 structured latents (如 tri-plane, sparse voxel)。然而, ABot-Earth 0.5 选择直接在 native 3D Gaussian Splatting (3DGS) space 中进行 generation。

**Intuition:** 为什么选 3DGS 而不是 NeRF 或 Mesh? 因为 outdoor scene (尤其是 vegetation, water surface, complex facades) 包含大量的 non-manifold topologies。Mesh 难以表达这些 semi-transparent 且拓扑复杂的结构; NeRF 虽然能表达, 但是体积渲染的计算成本极高, 无法支持 real-time, web-based 的 trillion-scale 探索。3DGS 采用 explicit 的 Gaussian primitives, 结合 tile-based rasterizer, 既保留了 photometric fidelity, 又具备了极好的渲染吞吐量。

**技术细节: 3DGS 渲染公式解析**

3DGS 的核心 rendering equation 基于 alpha blending。对于图像平面上的像素 $\mathbf{u}$, 其颜色 $C(\mathbf{u})$ 由所有覆盖该像素的 Gaussians 按深度排序叠加而成:

$$C(\mathbf{u}) = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

*   $C(\mathbf{u})$: 像素 $\mathbf{u}$ 最终的 RGB 颜色。
*   $\mathcal{N}$: 覆盖像素 $\mathbf{u}$ 的 Gaussian primitives 集合, 且按深度排序。
*   $\mathbf{c}_i$: 第 $i$ 个 Gaussian 的 color, 通常由 Spherical Harmonics (SH) 表示, 以模拟 view-dependent 的外观。
*   $\alpha_i$: 第 $i$ 个 Gaussian 的 2D opacity (透明度), 它是由 3D space 的 opacity $\sigma_i$ 和 2D projected Gaussian 共同决定的。$\alpha_i = \sigma_i \exp\left(-\frac{1}{2} \Delta^T \Sigma_i^{-1} \Delta\right)$, 其中 $\Sigma_i$ 是 covariance matrix, $\Delta$ 是像素中心到 projected Gaussian 中心的距离。
*   $\prod_{j=1}^{i-1}(1-\alpha_j)$: 透射率, 表示光线穿过前 $i-1$ 个 Gaussians 后剩余的光线比例。

在 ABot-Earth 的 generative framework 中, model 需要直接预测每个 Gaussian 的 position (mean $\mu$), covariance (由 scaling $s$ 和 rotation $q$ 导出), opacity $\alpha$, 以及 SH coefficients。这种 compression-generation paradigm 类似于 Latent Diffusion Models (LDM), 将百万级别的 unstructured primitives 压缩到一个 compact latent space 中, 然后通过 diffusion 或 flow matching 生成。

---

### 2. Data Pipeline: 高质量 3DGS 的生产与质检

模型的 ceiling 由 training data 决定。由于 internet 上不存在大规模的 outdoor 3DGS 数据, AMAP 团队构建了 ABot-3DGS 引擎来从多源数据重建 city-scale 场景。

**Data Sources 与 Cross-view fusion:**
Paper 提到了三类数据: Multi-Stereo Satellite Imagery, Aerial Data, Urban (Drone/Street-view) Data。
*   **Satellite:** 使用了 FromOrbit2Ground 模块。由于 satellite 视角极端且缺少 facade 信息, 他们采用了 Z-Monotonic SDF 恢复 watertight geometry, 并用 diffusion 补全 facade texture。
*   **Aerial:** 高分辨率 oblique imagery, 提供 core geometry 和 texture。
*   **Urban:** 弥补 aerial 在 low-altitude 时的 facade 细节缺失。

**Quality Assessment (Multi-granularity):**
这是极其工程化的一环。为了防止 "garbage in, garbage out", 团队引入了 VLM (Vision-Language Model) 进行自动质量评估。
1.  **Tile-level:** 评估 PSNR/SSIM/LPIPS, geometric accuracy, VLM perceptual score, spatial completeness。
2.  **View-level:** 剔除 low accumulated opacity 的 view (避免 void regions), 并利用 VLM 评估 texture sharpness 和 artifact。
3.  **Dataset-level:** 通过 stratified sampling 平衡 spatial diversity (避免单一城市形态主导), 并在 embedding space 进行 semantic deduplication (防止 mode collapse)。

**Intuition:** 将 VLM 引入 data curation pipeline 是一种非常 scalable 的数据清洗策略。在 building foundation models 时, 数据质量的重要性远超模型结构的微调。VLM 充当了一个自动化的人工标注员, 能够 filter 掉 floaters, artifacts 以及由于 multi-source appearance variation 导致的失败的 reconstruction。

---

### 3. Method Innovations: Scale, Continuity, and Robustness

要将 object-level 的 generator 扩展到 Earth-scale, 必须解决几个核心 challenges。Paper 提出了四个主要创新点。

#### 3.1 Inherent Multi-LOD Decoding
Earth-scale 的数据量是 trillion-scale 的 primitives, 必须依赖 Level-of-Detail (LOD) 才能实时渲染。传统的做法是生成高精度模型后再进行 downsampling。ABot-Earth 的 decoder 在生成阶段直接输出 hierarchical 3DGS structure。

**技术联想:** 这类似于在生成 3D asset 时, 直接生成一个 Octree 结构, 每一层 node 包含对应 LOD 的 Gaussians。结合 paper 中提到的 continuous level-of-detail (CLoD) 技术 (如 CLoD-GS [40]), 可能是通过预测每个 Gaussian 的 importance score 或 size, 渲染引擎根据 viewport 距离动态选择渲染哪些 primitives。

#### 3.2 Seamless Sliding-Window Inference
单次 inference 在 A100 上只能处理 1.6km x 1.6km 的 4K satellite image。为了生成连续的 3D 世界, 需要在 overlapping tiles 之间进行平滑过渡。

**Intuition:** 单纯的 tile stitching 会产生 visible seams (接缝)。Sliding-window strategy 在 generation phase 就考虑了 overlap region 的影响。这可能借鉴了 BlockFusion [42] 的思想, 在 latent space 对 overlapping region 进行 blending 或 extrapolation, 确保 transition zone 的 geometry 和 texture 连续。

#### 3.3 Cross-Domain Adaptation via VLM Harness
训练时使用的是 simulated satellite-view renderings (从 3DGS scene 渲染出的 top-down 视角)。但推理时输入是 real-world satellite image (如 Google Maps, Bing Maps), 存在巨大的 domain gap (sensor 差异, 大气干扰, resolution 差异)。

为了弥合这个 gap, ABot-Earth 引入了一个 VLM-based harness。我猜测它的运作机制是: VLM 分析 real-world satellite image, 提取 semantic context, seasonal information, 或 regional style, 然后将这些信息作为 conditional prompt 或 latent embedding 注入到 generative model 中, 从而 guide 生成符合该区域真实地理特征的 3D content。

---

### 4. Deployment: Engineering at Planetary Scale

这部分是本文最 core 的价值之一。从算法到 deploy 一个 3.2 trillion primitives 的 web-based 地图引擎, 工程挑战巨大。

#### 4.1 Global-Scale 3DGS Production Pipeline
*   **Scale:** Global built-up area 约 800,000 $km^2$, 划分为约 312,500 个 tiles。
*   **Throughput:** 1,000 张 A100 GPU, 每块 tile 推理约 25 分钟, 总计 under 10 days。每 $km^2$ 耗时 under 10 minutes。
*   **Georeferencing:** 必须处理 Web Mercator (EPSG:3857) 在高纬度地区的 areal distortion。通过将 Web Mercator tiles mosaic 后进行 isotropic resampling, 保证 uniform GSD (Ground Sampling Distance), 确保模型输入的 scale 一致性。

#### 4.2 EarthScape: Scalable Rendering Pipeline
3.2 trillion primitives 如何在 web 端实时渲染?

**I. Geographic Alignment:**
每个 block 被还原到 EPSG:3857 投影坐标系, 然后建立 ENU (East-North-Up) local tangent plane coordinate system。所有 Gaussian 的 position, rotation quaternion, scaling 都变换到这个 ENU frame。ENU 是一种以局部切平面为基准的坐标系, 适合局部高精度渲染, 同时通过 origin 锚定全球坐标。

**II. LOD Data Reorganization:**
*   **Re-partitioning:** 将所有 Gaussians 重新分配到标准地图层级 (zoom level 14 to 19)。
*   **Multi-level LOD Generation:** 
    *   高精度层级 (zoom 17-19) 由 inference model 原生输出, 避免 downsampling 导致的质量损失。
    *   低精度层级 (zoom 14-16) 通过 statistical decimation 生成, 核心算法基于 **Bhattacharyya distance**。

**技术细节: Bhattacharyya Distance 用于 Gaussian Decimation**
Bhattacharyya distance $D_B$ 用于衡量两个概率分布的相似性。对于两个 multivariate Gaussian distributions $p = \mathcal{N}(\mu_p, \Sigma_p)$ 和 $q = \mathcal{N}(\mu_q, \Sigma_q)$, 其 Bhattacharyya distance 为:

$$D_B(p, q) = \frac{1}{8} (\mu_p - \mu_q)^T \Sigma^{-1} (\mu_p - \mu_q) + \frac{1}{2} \ln \left( \frac{|\Sigma|}{\sqrt{|\Sigma_p| |\Sigma_q|}} \right)$$

*   $\mu_p, \mu_q$: 两个 Gaussian distributions 的 mean vectors (即 3D positions)。
*   $\Sigma_p, \Sigma_q$: 两个 Gaussian distributions 的 covariance matrices (由 scale 和 rotation 决定)。
*   $\Sigma$: 两个 covariance matrices 的平均值, $\Sigma = \frac{\Sigma_p + \Sigma_q}{2}$。
*   $|\cdot|$: 行列式。
*   $T$: 矩阵转置。

**Intuition:** 这个距离度量同时考虑了 spatial distance ($\mu_p - \mu_q$) 和 shape difference ($\Sigma_p, \Sigma_q$)。在 LOD decimation 时, 算法可以计算相邻 Gaussians 之间的 $D_B$, 将距离小的 Gaussians 合并或剔除, 从而在保持 statistical shape 分布近似的前提下, 大幅减少 primitive 数量。由于这个操作是解析的, 可以在 CPU 上并行处理, 不占用 GPU 资源, 大幅降低 end-to-end latency。

**III. Rendering Scheduling:**
集成到 Amap Yunjing 渲染引擎。利用 frustum culling 和 viewport-dependent tile scheduling, 近处加载 zoom 17-19, 远处加载 zoom 14-15。异步流式加载实现 trillion-scale 的实时帧率。

---

### 5. Evaluation: Beyond Pixel-level Fidelity

#### 5.1 Generative Fidelity
Paper 中的 Table 2 显示了压倒性的优势。

| Method | FID $\downarrow$ | KID $\downarrow$ |
| :--- | :--- | :--- |
| CityDreamer [15] | 97.3 | 0.096 |
| GaussianCity [24] | 86.9 | 0.090 |
| EarthCrafter [14] | 69.5 | 0.061 |
| **Ours** | **16.1** | **0.006** |

FID (Fréchet Inception Distance) 衡量生成图像与真实图像在 Inception 网络特征空间中的分布距离。从 69.5 到 16.1 是质的飞跃。且 ABot-Earth 的 GT 是高度复杂的 real-world 3DGS reconstruction, 建模难度远高于合成数据集。

#### 5.2 System-Level Applicability
与 Google Earth 和 Marble 对比:
*   **Coverage:** Google Earth 是 sparse (仅限 scanned region), ABot-Earth 是 infinite (只要有 satellite image 就能生成)。
*   **Efficiency:** Google Earth 需要数月到数年更新, ABot-Earth 生成 1 $km^2$ 耗时 under 10 minutes。
*   **Visual Quality:** Google Earth 在 geometric/textural fidelity 上更优 (由于有 manual post-processing 和 Manhattan-world assumption), 但 ABot-Earth 在 overall aesthetics 上得分更高, 因为其 generative model 能产生更 harmonious 的色彩和光照。

#### 5.3 Landmark Enhancement
这是一个非常聪明的 "hybrid" 策略。Generative model 擅长 producing plausible context, 而 reconstruction (如 COLMAP) 擅长 producing accurate geometry for specific targets。将高精度 reconstruct 的 Eiffel Tower 或 Colosseum 3DGS 模型 compositing 到 generative background 中, 达到了 visual fidelity 和 scale 的最佳平衡。这也证明了 native 3DGS 的 composability。

---

### 6. Build Intuition: World Models & Sim-to-Real

对于 Embodied AI (如 UAV navigation) 而言, 传统的 simulation (如 AirSim) 基于 synthetic assets, 存在巨大的 sim-to-real gap。UAV 在 synthetic world 中训练的 policy 很难直接迁移到 real world。

ABot-Earth 0.5 实际上提供了一个 **Photorealistic, Infinite-scale, 3D Structured World Model**。

1.  **Density of Information:** 相比于 Sora 等 video generation world models (2D pixel space + temporal coherence), 3DGS 提供了真实的 3D structure。UAV 可以在 3D space 中获取 multi-view observation, 甚至进行 occlusion reasoning 和 3D path planning。
2.  **Closed-loop Simulation:** 因为生成的是 native 3DGS, UAV 可以在这个环境中 flying around, rendering 出 new viewpoints。这构成了一个闭环: Agent action -> Environment state update -> Sensor observation rendering。
3.  **Cost-efficiency:** 传统获取真实世界多视角 drone 数据成本极高。通过 generative model, 我们可以在 satellite image 覆盖的任何区域 (即使是未勘探的野外) 快速搭建一个高保真 sandbox, 极大降低了 drone algorithm 验证的门槛。

**Scaling Laws Speculation:** Paper 在 conclusion 中提到将系统性地验证 outdoor 3D scene generation 的 scaling laws。我们可以联想 LLM 中的 scaling laws, 随着 satellite image resolution 提升, training data 覆盖面积增加, 以及 model parameters 扩大, 生成质量是否会呈现 power-law 的提升? 目前 1.6km x 1.6km 的单次 inference 是受限于 A100 VRAM, 未来在 H100 或 B200 上, 可能实现单次生成 4km x 4km 甚至更大的无缝区域, 这将进一步减少 sliding-window 带来的 boundary artifacts。

---

### 7. Web Links for Reference

为了方便你深入挖掘, 这里提供相关技术的参考链接:

1.  **3D Gaussian Splatting (Original Paper):** https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
2.  **CityGaussian (Large-scale 3DGS rendering):** https://arxiv.org/abs/2403.14555
3.  **CLoD-GS (Continuous Level-of-Detail for 3DGS):** https://arxiv.org/abs/2510.09997
4.  **TRELLIS (Structured 3D Latents, related to generative 3D):** https://arxiv.org/abs/2412.01501
5.  **BlockFusion (Related to sliding-window/latent extrapolation):** https://arxiv.org/abs/2404.00918
6.  **OGC 3D Tiles Specification (Deployment format):** https://www.ogc.org/standard/3dtiles/
7.  **AirSim (UAV Simulation baseline):** https://arxiv.org/abs/1705.05065
8.  **Sat3DGen (Related previous work by same authors on street-level 3D generation):** https://openreview.net/forum?id=E7JzkZCofa

总结而言, ABot-Earth 0.5 是一次 representation, generative algorithm, 和 system engineering 的完美结合。它避开了传统 photogrammetry 的物理限制, 通过 generative prior 和 native 3DGS 实现了 Earth-scale 3D simulation 的民主化。对于未来的 spatial computing 和 robotics 而言, 这种 infrastructural level 的突破是至关重要的。
