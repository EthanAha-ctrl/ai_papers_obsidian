---
source_pdf: WonderZoom.pdf
paper_sha256: 491013baad6727a0ce5c2bae6758074eeed02c8f03b21ccd5cba348607d75c38
processed_at: '2026-08-13T04:54:30-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 WonderZoom

## 先讲个故事

想象你在 Google Earth 上看 Earth。你 zoom out 看见整个地球，zoom in 看见美洲，再 zoom in 看见 California，再 zoom in 看见 Stanford 校园，再 zoom in 看见一棵树，再 zoom in 看见树叶上的 ladybug，再 zoom in 看见 ladybug 背上的斑点纹理。

这个 zoom 是连续的、smooth 的、无止境的。

Google Earth 能做到这件事，因为所有 levels 的 satellite imagery 和 street view 都预先存在。Google 用 LoD (Level of Detail) 技术在合适的 zoom level 显示合适的 imagery。

现在你只有**一张图片**——比如一张向日葵田的照片。你想生成一个 3D world，让 user 可以无限 zoom in 下去，每次 zoom 都 reveal 出新的、coherent 的细节。这就是 WonderZoom 要做的事。

## 问题为什么 hard

第一个反应：这有什么难的？我有一个 super-resolution model，zoom in 就 super-res 一下不就完了？

问题来了：super-resolution 只能 sharpen **已经存在**的 detail。但你想 zoom 进向日葵的花瓣里，看到一个 ladybug 站在上面——这个 ladybug **原本根本不存在**。你需要 *生成* 它，不是 *enhance* 它。

而且 ladybug 要和向日葵的 geometry 一致（要"站"在花瓣上），要和 lighting 一致，要和 scale 一致。这一连串 consistency 的 constraints 让问题变得 hard。

再深一层：现有的 3D 生成方法，比如 WonderWorld、HunyuanWorld，都假设 single scale——你给一张图，它们生成一个 room 或者一个 landscape，然后停下来。它们没有 multi-scale 的 representation。你 zoom in 就是放大 pixel，看到的还是原来那些 surfels，只是变模糊了。

所以真正的 bottleneck 不是 generator 不够强，而是 **representation 不支持 multi-scale**。

## 传统 multi-scale representation 为什么不行

这里有个 subtle 但 key 的 observation。

传统 graphics 的 LoD [Luebke et al. 2002](https://www.elsevier.com/books/level-of-detail-for-3d-graphics/luebke/978-0-08-051275-5) 假设你预先知道所有 levels 的 geometry。游戏引擎里一棵树有 3 个 LOD mesh（远处用低面数，近处用高面数），所有 mesh 都是 artist 事先做好的。然后 rendering 时根据 distance 切换 mesh。

NeRF 系列的 multi-scale 方法，比如 Mip-NeRF [Barron et al. 2021](https://arxiv.org/abs/2103.13415)、Mip-NeRF 360 [Barron et al. 2022](https://arxiv.org/abs/2111.12077)、Zip-NeRF [Barron et al. 2023](https://arxiv.org/abs/2304.06452)，处理的是 aliasing 问题——同一个 scene 在不同 viewing distance 下应该看起来不同。它们假设你已经有所有 scales 的 ground-truth images，然后一次性优化整个 multi-scale representation。

Gaussian Splatting 系列的 hierarchical 方法，比如 Hierarchical 3DGS [Kerbl et al. 2024](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)、Octree-GS [Ren et al. 2024](https://arxiv.org/abs/2403.17898)、Scaffold-GS [Lu et al. 2024](https://arxiv.org/abs/2312.00109)，也是 reconstruction paradigm——需要 complete multi-scale supervision。

**关键 conflict**：generation 是 sequential 的，coarse-to-fine。你先生成 coarse level，再生成 fine level。fine level 的 image **根本不存在**——你要 *生成* 它，怎么能让 representation 假设它已经存在？

这就是 WonderZoom 要解决的核心 tension。

## WonderZoom 的两个 trick

### Trick 1: Scale-adaptive Gaussian Surfels

先回顾下 Gaussian Splatting 的基本概念。3D scene 由一堆 anisotropic Gaussian 组成，每个 Gaussian 有 position、rotation、scale、opacity、color。Rendering 时把 Gaussian 投影到 image plane，做 alpha blending。这种方法 [Kerbl et al. 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 让 real-time radiance field rendering 成为可能。

WonderZoom 用的是 **surfel**（surface element），就是非常薄的 Gaussian——x、y 有 scale，z 方向用一个小 $\epsilon$ 厚度。这相当于把每个 surface patch 表示成一个 oriented disk。

每个 surfel 的标准参数:
$$g = \{\mathbf{p}, \mathbf{q}, \mathbf{s}, o, \mathbf{c}\}$$

- $\mathbf{p}$: 3D position
- $\mathbf{q}$: orientation quaternion
- $\mathbf{s} = [s_x, s_y]$: x/y scale
- $o$: opacity
- $\mathbf{c}$: view-independent color

WonderZoom **加了一个字段** $s^{\text{native}}$——这个 surfel 被创建时所在的 scale。

什么是 "scale"？这是 focal length 与 depth 的比:
$$s^{\text{native}} = \frac{d^{\text{native}}}{\sqrt{f_x^{\text{native}} f_y^{\text{native}}}}$$

直觉上：一个 1 米的 surfel 在 1 米外、focal length 1000 pixel 的相机看，和它在 8 米外、focal length 8000 pixel 的相机看，**投影大小一样**。所以 $d / f$ 这个 ratio 是一个 scale-invariant 的量——它告诉你"这个 surfel 在这个相机看占据多少 visual angle"。

这就是 native scale 的本质：**这个 surfel 在被创建时，在 image plane 上占多大 footprint**。

### Trick 2: Append-only Dynamic Updating

这是 WonderZoom 与传统 LoD 的根本区别。

传统 LoD 要 build 一个 hierarchy，所有 levels 一次性 construct。WonderZoom 是 **incremental append**：

- $\mathcal{E}_0$: 从 input image $\mathbf{I}_0$ 生成 $N_0$ 个 surfels
- $\mathcal{E}_1$: 从 zoomed-in view $\mathbf{I}_1$ 添加 $N_1$ 个 surfels，总数 $N_0 + N_1$
- $\mathcal{E}_i$: 添加 $N_i$ 个 surfels，总数 $\sum_{k=0}^i N_k$

**之前 levels 的 surfels 完全不动**。只 add new surfels，不 modify 旧的。

这个设计为什么重要？因为它避免了 global re-optimization。你 zoom 进一个新的 region，生成新的 surfels，把它们 add 进去，旧的 surfels 保持不变。整个 multi-scale world 像 Git history 一样 growing，每个 commit 是一个 scale level。

这与 functional programming 的 persistent data structures 思想一致——immutable updates，structural sharing。也与 blockchain 的 append-only ledger 思想一致。

### Trick 3: Scale-aware Opacity Modulation

这是最 magic 的部分。

问题：现在同一个 3D region 可能被 multiple scales 的 surfels 覆盖。比如 zoom 进向日葵花瓣，原来 $\mathcal{E}_0$ 的 coarse surfels 还在那里，新 $\mathcal{E}_1$ 的 fine surfels 也 add 上去了。两套 surfels 重叠。如果都 fully render，会有 aliasing——两套 geometry 互相 conflict，rendering 速度也慢。

Solution：每个 surfel 只在它的 **native scale 附近**可见，远离 native scale 时 fade out。

具体怎么 fade？用 log-space linear interpolation:

$$\tilde{o} = o \cdot \alpha$$

$$\alpha = \begin{cases} 
1 & \text{if no parent and } s^{\text{render}} \geq s^{\text{native}} \\
\frac{\log(s^{\text{parent}}) - \log(s^{\text{render}})}{\log(s^{\text{parent}}) - \log(s^{\text{native}})} & \text{if } s^{\text{parent}} \geq s^{\text{render}} \geq s^{\text{native}} \\
\frac{\log(s^{\text{render}}) - \log(s^{\text{child}})}{\log(s^{\text{native}}) - \log(s^{\text{child}})} & \text{if } s^{\text{native}} \geq s^{\text{render}} \geq s^{\text{child}} \\
1 & \text{if no child and } s^{\text{render}} \leq s^{\text{native}} \\
0 & \text{otherwise}
\end{cases}$$

翻译成人话：当 render scale 等于 native scale，surfel 完全可见 ($\alpha=1$)。当 render scale 偏离 native scale，surfel opacity 线性下降（在 log space）。当 render scale 完全离开 native scale 的"管辖范围"，surfel 完全 invisible ($\alpha=0$)。

**为什么是 log space？** 因为 scale 的 perception 是 logarithmic 的。1x zoom 到 2x zoom 的"距离"与 8x zoom 到 16x zoom 的"距离"是一样的——都是 log scale 上的 1 unit。这就是 Weber-Fechner law。

**Beautiful 的 partition of unity property**：考虑两个 surfels $g_j, g_k$ 在同一 3D 位置，但分别属于 adjacent scales $\mathcal{E}_{i-1}, \mathcal{E}_i$。当 render scale 在它们的 native scale 之间 transition 时:
$$\alpha_k(s^{\text{render}}) + \alpha_j(s^{\text{render}}) = 1$$

一个 fade out，另一个 fade in，total contribution 保持 constant。这就像 cross-fade in audio mixing——避免 click artifacts。在 geometry context，这避免 popping——zoom 时不会突然看到某个 surfel pop in 或 pop out。

这个 partition of unity 在 manifold geometry 里是个 deep concept——用来 patch local charts 成 global manifold。WonderZoom 用它 patch local scale representations 成 global multi-scale scene。

## Progressive Detail Synthesizer 怎么 generate

整个 generation pipeline 是 5 个 stage。

### Stage 1: New Scale Image Synthesis

第一步：在 zoomed-in view 渲染当前 scene，得到一个 coarse observation $\mathbf{O}_i$。这个 $\mathbf{O}_i$ 是直接从已有 coarse scene 渲染的，自然没有 fine detail——zoom 进去看到的还是原来那些 surfels，只是放大了。

第二步：用 extreme super-resolution 把 $\mathbf{O}_i$ 变成有 detail 的图。用 Chain-of-Zoom [Kim et al. 2025](https://github.com/KimSungHwan06/Chain-of-Zoom)，能做 64× 甚至 512× 的 extreme SR。

但 pure SR 不够——zoom 太深，看不见足够的 context 来 synthesize 合理 detail。所以用 VLM (GPT-4V) 提取上一个 scale 的 semantic context $S = \mathrm{VLM}(\mathbf{O}_{i-1})$，把 $S$ 作为 condition 喂给 SR model:
$$\mathbf{I}'_i = \mathrm{SR}(\mathbf{O}_i, S)$$

第三步：如果 user 给了 prompt 比如 "a ladybug is on the sunflower"，用 controllable image editing model 插入这个 ladybug:
$$\mathbf{I}_i = \mathrm{Edit}(\mathbf{I}'_i, \mathcal{U}_i)$$

注意：ladybug 在 coarse scene 里**不存在**，是 edit 出来的，不是 super-res 出来的。这是 WonderZoom 与 super-resolution 方法（Real-ESRGAN、SwinIR）的根本区别——它能 **insert new content**。

### Stage 2: Scale-Consistent Depth Registration

现在有 image $\mathbf{I}_i$ 但没有 depth。用 monocular depth estimator (MoGe [Wang et al. 2024](https://wangrc.site/MoGe/)) 估计 depth。

问题：monocular depth estimator 的 scale 与 coarse scene 的 scale 不对齐——绝对 depth 难，相对 depth 容易。所以 estimator 给的 depth 可能与 $\mathcal{E}_{i-1}$ 的 geometry 冲突。

Solution: fine-tune depth estimator 在 sparse target depth supervision 下：
$$\mathcal{L}_{\text{depth}} = \frac{\sum_{u,v} \|\mathbf{D}_i^{\text{target}}(u,v) - \mathcal{D}_\theta(\mathbf{I}_i)(u,v)\| \cdot m(u,v)}{\sum_{u,v} m(u,v)}$$

$\mathbf{D}_i^{\text{target}} = \mathrm{render\_depth}(\mathcal{E}_{i-1}, \mathbf{C}_i)$ 是从上一个 scale 渲染的 sparse target depth。$m(u,v)$ 是 validity mask——只有 zoom-in region 里 visible 的 pixel 才有 target depth，newly revealed region 没有。

进一步 refinement: 用 SAM [Kirillov et al. 2023](https://github.com/facebookresearch/segment-anything) 生成 segment mask，在每个 segment 内做 scale/shift 对齐。对于 user-specified 新结构（如 ladybug），用 Grounded SAM [Ren et al. 2024](https://github.com/IDEA-Research/Grounded-Segment-Anything) 定位，单独 estimate depth 但 constrain 与 surrounding 一致。

这个 trick 来自 WonderJourney [Yu et al. 2024](https://WonderJourney.github.io/) 和 WonderWorld [Yu et al. 2025](https://wonderworld-2024.github.io/)——depth scale alignment 是 single-image-to-3D 的核心难点。

### Stage 3: Surfel Initialization

从 $(\mathbf{I}_i, \mathbf{D}_i, \mathbf{C}_i)$ 生成 pixel-aligned surfels：
- 每个 pixel 一个 surfel
- Position: $\mathbf{D}_i$ back-projection
- Orientation: estimated normal
- Scale: Nyquist sampling theorem（density 恰好满足采样定理）
- Native scale: 基于 $\mathbf{C}_i$ 计算

这是 partial scene $\mathcal{E}_i^{\text{partial}}$，因为只有 single view。

### Stage 4: Auxiliary View Synthesis

Single image 不足以 constrain 完整 3D scene——会有 unseen regions。所以 synthesize auxiliary views：

1. Sample $K$ 个 neighboring cameras $\{\mathbf{C}_i^k\}$
2. 用 $\mathcal{E}_i^{\text{partial}}$ 渲染 conditioning frames $\{\mathbf{O}_i^k\}$
3. Compute masks $\{\mathbf{M}_i^k\}$ 指示需要 synthesis 的 region (occlusion 等)
4. 用 camera-controlled video diffusion model (Gen3C [Ren et al. 2025](https://gen3c-stability.github.io/)) 生成 temporally consistent frames $\{\mathbf{I}_i^k\}$
5. 用 video depth model (GeometryCrafter [Xu et al. 2025](https://geometrycrafter.github.io/)) estimate depth $\{\mathbf{D}_i^k\}$

这里用 video model 而非 multi-view image model 的原因：video model 的 temporal consistency prior 可以 leverage——frames 之间有 smooth motion，避免 hallucinate 出 mutually inconsistent views。

### Stage 5: Optimization

用所有 image-depth pairs $\{\mathbf{I}_i, \mathbf{I}_i^1, \dots, \mathbf{I}_i^K\}$ optimize $\{o, \mathbf{q}, \mathbf{s}\}$，固定 $\mathbf{p}, \mathbf{c}, s^{\text{native}}$。Loss:
$$\mathcal{L} = 0.8 L_1 + 0.2 L_{\text{D-SSIM}}$$

固定 position 和 color 是为了避免 optimization drift——geometry 和 appearance 锚定在初始 estimate，只 fine-tune orientation 和 opacity。

## 实验数据里的故事

### Quantitative Comparison (Table 1)

| Method | CS↑ | CIQA↑ | QIQA↑ | NIQE↓ | QIAA↑ | Time/s |
|---|---|---|---|---|---|---|
| WonderWorld [Yu et al. 2025](https://wonderworld-2024.github.io/) | 0.2687 | 0.5064 | 1.081 | 21.74 | 1.339 | 9.3 |
| HunyuanWorld [Tencent, 2025](https://arxiv.org/abs/2507.21809) | 0.2510 | 0.2827 | 1.058 | 15.21 | 1.302 | 704.2 |
| Gen3C [Ren et al. 2025](https://gen3c-stability.github.io/) | 0.3004 | 0.5489 | 2.992 | 4.924 | 2.018 | 306.7 |
| Voyager [Huang et al. 2025](https://arxiv.org/abs/2506.04225) | 0.2609 | 0.5746 | 3.148 | 4.913 | 2.929 | 596.6 |
| **WonderZoom** | **0.3432** | **0.7035** | **3.926** | **3.695** | 2.986 | 62.1 |

观察几个点：

1. **CLIP Score (CS) 0.3432**: WonderZoom 的 prompt alignment 最好。Why? 因为它用 explicit prompt editing——user 说 "a ladybug"，editor 就真插入 ladybug。Video models (Gen3C, Voyager) 的 control 不够 precise，prompt alignment 弱。

2. **CIQA 0.7035**: 这是最 dramatic 的差距——比次优的 Voyager 高 22%。CIQA 是 perceptual quality metric。WonderZoom 的两阶段 design (SR + Edit) 保证了既 high-frequency detail 又 semantic coherent。

3. **Time 62.1s**: 介于 WonderWorld (9.3s) 和 Gen3C (306.7s) 之间。WonderWorld 快因为简单，但 quality 差。Gen3C 慢因为 video diffusion。WonderZoom 的 sweet spot——既 high quality 又可接受速度。

4. **HunyuanWorld 704.2s**: mesh-based representation 的 dense optimization 让它非常慢。

5. **Voyager QIAA 2.929 略高于 WonderZoom 2.986**: 在 aesthetic metric 上 video model 略胜。可能因为 video diffusion 的 high-quality prior 让 colors/lighting 更 "cinematic"。

### Human Study (Table 2)

| Comparison | Zoom-in Accuracy | Visual Quality | Prompt Match |
|---|---|---|---|
| vs WonderWorld | 80.7% | 98.3% | 98.2% |
| vs HunyuanWorld | 83.2% | 98.7% | 98.9% |
| vs Gen3C | 77.8% | 83.8% | 96.1% |
| vs Voyager | 76.1% | 81.7% | 90.9% |

关键 pattern：**Visual Quality 和 Prompt Match 几乎一边倒（98%+ vs 3D methods, 90%+ vs video methods）**。Zoom-in Accuracy 略低（76-83%）——video methods 在 zoom perception 上仍有竞争力，因为 video 的 temporal dynamics 自然传递 zoom 感。

### Ablation: Opacity Modulation 的魔力 (Table 3)

| Variant | GPU Memory | FPS |
|---|---|---|
| Ours w/o modulation | 7.96G | 1.4 |
| Ours | 3.40G | **97.2** |

**70× 加速**——从 1.4 FPS 到 97.2 FPS。GPU memory 降 57%。

这是 WonderZoom 最 dramatic 的 ablation。Without modulation，所有 surfels 在所有 scales 都 fully rendered，造成：
- Computational redundancy（远处 coarse surfels 在 zoom-in 时也参与 rendering）
- Aliasing（多个 scale 的 surfels 互相 conflict）
- Memory overhead

Opacity modulation 相当于 soft culling——让 surfels 只在 native scale 附近 visible。既加速又避免 aliasing。这是 partition of unity 的 magic。

### Ablation: Depth Registration (Figure 6)

去掉 depth registration 导致 beetle 严重 distorted。Why? Monocular depth estimator 的 absolute scale 与 coarse geometry 不对齐——beetle 被放在错误的 depth，导致 multi-view 优化时 shape collapse。

### Ablation: Auxiliary View (Figure 7)

去掉 auxiliary view 导致 grey areas（unseen regions）。Single image 不足以 constrain 3D scene，需要 generative prior (video diffusion) 补全。

## Limitations 与 Future

Figure 13 的 failure case：repeatedly zoom 进 tree branches，最终 collapse 到 pure texture。没有 semantic cues 来推断下一个 scale 应该有什么。

这暴露了一个 fundamental limitation：**WonderZoom 依赖 semantic context 来 guide generation**。当区域变成 pure texture（如 fractal pattern），semantic prior 失效。

可能的 future direction：
- **Procedural generation**: 用 procedural rules 生成 self-similar micro-structures（fractal noise、Wang tiling）
- **Texture-specific priors**: 训练专门处理 texture 的 generative model
- **Self-similar priors**: detect self-similar patterns 并 extrapolate

更深层的 future direction：
- **City-scale extension**: 从 aerial view zoom 到 street view，再 zoom 到 building facade，再 zoom 到 brick texture。Google Earth 式的 generative experience。
- **Multi-user collaborative editing**: append-only design 天然支持 branching/merging，可能多人同时 explore 不同 zoom paths
- **End-to-end learning**: 现在是 orchestration of foundation models，未来可能用单一 multimodal model 端到端学习
- **Fractal-aware priors**: Benoît Mandelbrot 的 fractal geometry 在 natural scenes 中 pervasive，可以 explicitly model scale-invariant structures

## 我的整体 intuition

WonderZoom 的真正贡献不是某个具体 trick，而是 **把 multi-scale representation 从 reconstruction paradigm 扩展到 generation paradigm**。

Reconstruction paradigm：complete supervision, one-pass optimization, static hierarchy
Generation paradigm：partial supervision, iterative synthesis, dynamic growth

这个 paradigm shift 在 NeRF→3DGS→Generative 3D 演化中已经发生。WonderZoom 在 multi-scale dimension 重复了这个 shift。

技术上最 elegant 的部分是 **native scale field + log-space opacity modulation**。简单字段，简单 interpolation，但实现了 partition of unity 和 smooth scale transition。这是 mathematical design 的 beauty——minimum mechanism, maximum effect。

Append-only updating 是另一个 deep insight——把 multi-scale representation 转向 incremental addition paradigm。与 Git、blockchain、persistent data structures 同一思想。这让 multi-scale world 可以"organically grow"——zoom 进新 region，生成新 surfels，append 进去，旧 surfels 不变。

整个 framework 是 foundation model orchestration 的一个 clean example：VLM 提供 semantic context，SR 提供 detail，Edit 提供 novel content，Video Diffusion 提供 multi-view consistency，Depth Estimator 提供 geometry，SAM 提供 segmentation。WonderZoom 用 scale-adaptive representation 把这些 tools 串起来，做成了 multi-scale 3D generation。

Limitation 在 texture-only regions 暴露了 semantic-dependent generation 的边界——Future 需要 procedural + semantic hybrid approach。

总的来说，WonderZoom 是 single-image-to-multi-scale-3D 这个新 problem 的 solid first solution。它开了 representation design 这个 direction，留给 community 大量探索空间。

## Web Links

**Project**:
- WonderZoom: https://wonderzoom.github.io/

**Predecessors (Stanford)**:
- WonderJourney: https://WonderJourney.github.io/
- WonderWorld: https://wonderworld-2024.github.io/

**Multi-scale Generation**:
- Generative Powers of Ten: https://powers-of-10.github.io/
- Infinite Nature: https://infinite-nature.github.io/

**3DGS Foundations**:
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Hierarchical 3DGS: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/
- Mip-NeRF: https://jonbarron.info/mipnerf/
- Mip-Splatting: https://github.com/autonomousvision/mip-splatting

**Components Used by WonderZoom**:
- Chain-of-Zoom: https://github.com/KimSungHwan06/Chain-of-Zoom
- Gen3C: https://gen3c-stability.github.io/
- MoGe: https://wangrc.site/MoGe/
- GeometryCrafter: https://geometrycrafter.github.io/
- SAM: https://github.com/facebookresearch/segment-anything
- Grounded SAM: https://github.com/IDEA-Research/Grounded-Segment-Anything

**Single-image 3D Generation**:
- CAT3D: https://cat3d.github.io/
- LucidDreamer: https://luciddreamer-team.github.io/
- Text2Room: https://text-to-room.github.io/

---

# WonderZoom: 多尺度 3D 世界生成的深度技术解析

## 一、核心问题与 motivation

WonderZoom 解决的问题非常 fundamental：给定一张 image $\mathbf{I}_0$，能否生成一个可以"无限 zoom in"的 3D 世界？例如从一片向日葵田地 zoom 到花瓣上的瓢虫，再到瓢虫背上的斑点细节。这本质上是把 Powers of Ten（1977 年 Charles & Ray Eames 的经典纪录片）的 visual concept 落地到可交互的 3D generation。

**关键 insight**：现有 3D generation 方法的 bottleneck 不在 generator 本身，而在 representation。传统 Level-of-Detail (LoD) [Luebke et al. 2002] 假设所有 geometric details 已知，是为 rendering pre-authored content 设计的；neural reconstruction 方法如 Mip-NeRF [Barron et al. 2021](https://arxiv.org/abs/2103.13415)、Mip-NeRF 360 [Barron et al. 2022](https://arxiv.org/abs/2111.12077)、Zip-NeRF [Barron et al. 2023](https://arxiv.org/abs/2304.06452)、Hierarchical 3DGS [Kerbl et al. 2024](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)、Mip-Splatting [Yu et al. 2023](https://arxiv.org/abs/2311.16493)、Octree-GS [Ren et al. 2024](https://arxiv.org/abs/2403.17898)、Scaffold-GS [Lu et al. 2024](https://arxiv.org/abs/2312.00109) 都假设 complete multi-scale supervision 在训练时可用。Generation 的本质却是 coarse-to-fine 的 sequential synthesis，images at finer scales **并不存在**，需要被 progressive 生成。这是 representation 与 generation paradigm 之间的根本 conflict。

WonderZoom 的两个 key innovations 直接 attack 这个 conflict：
1. **Scale-adaptive Gaussian surfels** — 一种 dynamically updatable 的 hierarchical representation
2. **Progressive detail synthesizer** — 一个 iterative 的 coarse-to-fine generation pipeline

---

## 二、Scale-adaptive Gaussian Surfels —— representation 的核心

### 2.1 Surfel 参数化

每个 surfel 定义为 $g = \{\mathbf{p}, \mathbf{q}, \mathbf{s}, o, \mathbf{c}, s^{\mathrm{native}}\}$：
- $\mathbf{p} \in \mathbb{R}^3$：3D spatial position
- $\mathbf{q} \in \mathbb{R}^4$：orientation quaternion（用四元数避免 gimbal lock，比 Euler angles 更适合 optimization）
- $\mathbf{s} = [s_x, s_y]$：x/y 轴 scale（注意没有 $s_z$，surfel 是 surface element，厚度用 $\epsilon$ 给出）
- $o \in [0,1]$：opacity
- $\mathbf{c} \in \mathbb{R}^3$：view-independent RGB color（假设 diffuse lighting，简化 problem）
- $s^{\mathrm{native}}$：**这是 paper 的灵魂字段**，记录 surfel 被创建时的 native scale

Covariance matrix:
$$\mathbf{\Sigma} = \mathbf{Q} \, \mathrm{diag}(s_x^2, s_y^2, \epsilon^2) \, \mathbf{Q}^T$$

其中 $\mathbf{Q}$ 是从 $\mathbf{q}$ 转换得到的 $3 \times 3$ rotation matrix，$\epsilon$ 是 small thickness parameter 保证可微。这与原版 3DGS [Kerbl et al. 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 一致。

### 2.2 Native scale 的定义

公式 (1):
$$s^{\mathrm{native}} = \frac{d^{\mathrm{native}}}{\sqrt{f_x^{\mathrm{native}} f_y^{\mathrm{native}}}}$$

变量含义：
- $d^{\mathrm{native}}$：surfel 相对其创建相机 $\mathbf{C}_i$ 的 depth
- $f_x^{\mathrm{native}}, f_y^{\mathrm{native}}$：$\mathbf{C}_i$ 的 focal length（x/y 方向）

**Intuition**：这个 quantity 本质上是 surfel 在 image plane 上的"投影大小"的标准化度量。可以理解为：在 pinhole camera model 下，一个 3D point 在 image plane 上的 footprint 正比于 $d / \sqrt{f_x f_y}$。除以焦距的 geometric mean 是为了 normalize 掉 aspect ratio 的影响。这个量纲是 length（米/厘米等），代表"在这个相机看，这个 surfel 占据多少 scene space"。

为什么用 $\sqrt{f_x f_y}$ 而不是 $f_x$ 或 $f_y$？这是因为 foreshortening 在两个方向都有；用 geometric mean 给出一种"等效 isotropic focal length"。这与 Mip-NeRF 的 integrated positional encoding 思想有共鸣——都是要 capture "what scale does this camera see"。

### 2.3 Dynamic updating —— 与传统 LoD 的根本区别

传统 LoD 和 Hierarchical 3DGS 需要预先知道所有 scales 的 geometry，一次性 build hierarchy。WonderZoom 的 dynamic updating 机制是 **incremental addition**：

- 初始化 $\mathcal{E}_0$：从 $\mathbf{I}_0$ 生成 $N_0$ surfels
- 生成 $\mathcal{E}_1$：从 zoom-in view $\mathbf{I}_1$ 在 $\mathbf{C}_1$ 添加 $N_1$ surfels，总 surfels $N = N_0 + N_1$
- 生成 $\mathcal{E}_i$：添加 $N_i$ surfels，总 $N = \sum_{k=0}^{i} N_k$

**Crucially**：之前 scales 的 surfels 保持不变。新 surfels 只 append，不 modify。这是 append-only 的 design，与 Git、blockchain、persistent data structures 思想一致——避免了 global re-optimization，让 multi-scale world "organically grow"。

这一点让我联想到 NeRF 的 tangent space tricks（如 Plenoxels、Instant-NGP）以及 INR 的 locality：representation 应该有 spatially local updates，不应该 modify 远处的 content。WonderZoom 把这个原则从空间维度扩展到 scale 维度。

### 2.4 Scale-aware opacity modulation —— 渲染的 magic

公式 (2) 和 (3):
$$\tilde{o} = o \cdot \alpha$$

$$
\alpha = \begin{cases} 
1 & \text{if no parent and } s^{\mathrm{render}} \geq s^{\mathrm{native}} \\
\frac{\log(s^{\mathrm{parent}}) - \log(s^{\mathrm{render}})}{\log(s^{\mathrm{parent}}) - \log(s^{\mathrm{native}})} & \text{if } s^{\mathrm{parent}} \geq s^{\mathrm{render}} \geq s^{\mathrm{native}} \\
\frac{\log(s^{\mathrm{render}}) - \log(s^{\mathrm{child}})}{\log(s^{\mathrm{native}}) - \log(s^{\mathrm{child}})} & \text{if } s^{\mathrm{native}} \geq s^{\mathrm{render}} \geq s^{\mathrm{child}} \\
1 & \text{if no child and } s^{\mathrm{render}} \leq s^{\mathrm{native}} \\
0 & \text{otherwise}
\end{cases}
$$

变量含义：
- $s^{\mathrm{render}} = d^{\mathrm{render}} / \sqrt{f_x^{\mathrm{render}} f_y^{\mathrm{render}}}$：当前渲染相机下的 scale
- $s^{\mathrm{parent}}$：parent scale bound（上一个 scale $\mathcal{E}_{i-1}$ 时的 scale）
- $s^{\mathrm{child}}$：child scale bound（下一个 scale $\mathcal{E}_{i+1}$ 时的 scale）

**Intuition**：当 $s^{\mathrm{render}}$ 等于 $s^{\mathrm{native}}$ 时，surfel 最 visible（$\alpha=1$）；当 $s^{\mathrm{render}}$ 偏离 $s^{\mathrm{native}}$ 时，surfel 平滑 fade out。

为什么用 **log 空间线性插值**？这是非常 subtle 但关键的 design choice：
- Scale 的 perception 在 humans 是 logarithmic 的（Weber-Fechner law）
- 8x zoom 与 16x zoom 的"距离"应该是 1 个 unit，不是 8 个 unit
- Log space 让相邻 scales 的 "distance" 均匀，对应 paper 里 focal length 每次 ×8 的 exponentially 增长

**Proposition 1 (Partition of Unity)**：当两个 surfels $g_j, g_k$ 在同一 3D 位置但属于 adjacent scales $\mathcal{E}_{i-1}, \mathcal{E}_i$ 时，在 transition zone $s^{\mathrm{render}} \in [s_k^{\mathrm{native}}, s_j^{\mathrm{native}}]$:
$$\alpha_k(s^{\mathrm{render}}) + \alpha_j(s^{\mathrm{render}}) = 1$$

这是 partition of unity property。它保证 zoom transition 时 total opacity contribution 不变，eliminate popping artifacts。这个思想在 rendering literature 里很经典，比如 Gouraud shading 的 barycentric weights、subdivision surfaces 的 basis functions、B-spline 的 partition of unity。

### 2.5 Optimization

Surfel 优化是 lightweight 的：
- 固定：$\mathbf{p}, \mathbf{c}, s^{\mathrm{native}}$（geometry 和 appearance 锚定，避免 drift）
- 优化：$o, \mathbf{q}, \mathbf{s}$（用 Adam [Kingma & Ba 2014](https://arxiv.org/abs/1412.6980)）
- Loss: $\mathcal{L} = 0.8 L_1 + 0.2 L_{\mathrm{D-SSIM}}$（与原版 3DGS 一致）

Position 用 estimated depth back-projection 初始化，orientation 用 estimated surface normal，scale 用 Nyquist sampling theorem。Nyquist 原理确保 surfel density 恰好满足采样定理——既不过密（浪费）也不过稀（漏洞）。

Opacity 初始化为 0.1 是为了 stable optimization——太高的 initial opacity 会让 gradient 信号 early saturate。

---

## 三、Progressive Detail Synthesizer —— generation 的核心

整个 pipeline 是 5 stages，对应 Algorithm 1 的 Line 11-32。

### 3.1 Stage 1: New Scale Image Synthesis

$$\mathbf{O}_i = \mathrm{render}(\mathcal{E}_{i-1}, \mathbf{C}_i)$$

$\mathbf{C}_i$ 的 focal length 大于 $\mathbf{C}_{i-1}$（paper 里 ×8），实现 zoom-in。$\mathbf{O}_i$ 是从已有 coarse scene 直接渲染的 zoomed-in view，自然缺乏 fine details。

**Two-stage image generation**：
1. **Semantic context extraction**: $S = \mathrm{VLM}(\mathbf{O}_{i-1})$
   - 用 GPT-4V 提取上一个 scale 渲染图的语义描述
   - Why? 因为 extreme zoom ratio 下，单纯 super-resolution 看不到足够的 context 来 synthesize 合理细节
   - 这与 Stable Diffusion 的 text conditioning 类似，但 text 是从 image 自动提取

2. **Extreme super-resolution**: $\mathbf{I}'_i = \mathrm{SR}(\mathbf{O}_i, S)$
   - 用 Chain-of-Zoom [Kim et al. 2025](https://arxiv.org/abs/2505.13490)
   - Chain-of-Zoom 本身是 scale autoregression + preference alignment，可以做 extreme SR (e.g., 64×, 512×)

3. **Controlled editing**: $\mathbf{I}_i = \mathrm{Edit}(\mathbf{I}'_i, \mathcal{U}_i)$
   - 用 controllable image editing model 插入 user 指定的新结构（如 "a ladybug on the sunflower"）
   - 关键：ladybug 在 coarse scene 里**根本不存在**，不是超分出来的，是 edit 出来的
   - 这是 "generative" 而非 "enhancement" 的本质区别

这种 SR + Edit 的两阶段 design 让 WonderZoom 既可以对已有结构做 detail enhancement，又可以插入 novel content。这是相对 super-resolution 方法（如 Real-ESRGAN, SwinIR）的根本突破。

### 3.2 Stage 2: Scale-Consistent Depth Registration

公式 (5):
$$\mathcal{L}_{\mathrm{depth}} = \frac{\sum_{u,v} \|\mathbf{D}_i^{\mathrm{target}}(u,v) - \mathcal{D}_\theta(\mathbf{I}_i)(u,v)\| \cdot m(u,v)}{\sum_{u,v} m(u,v)}$$

变量：
- $\mathbf{D}_i^{\mathrm{target}} = \mathrm{render\_depth}(\mathcal{E}_{i-1}, \mathbf{C}_i)$：从上一个 scale 渲染的 sparse target depth
- $\mathcal{D}_\theta$：monocular depth estimator (MoGe [Wang et al. 2024](https://arxiv.org/abs/2406.06494))
- $m(u,v)$：validity mask，1 if $\mathbf{D}_i^{\mathrm{target}}(u,v)$ defined（即在 zoom-in region 中且未被 occlusion），0 otherwise

**Intuition**：直接用 monocular depth estimator 会导致 fine-scale depth 与 coarse scene 的 geometry 不一致——就像把一个独立 estimated 的 3D object 硬塞到另一个 scene 里。通过 fine-tune depth estimator 在 sparse target depth supervision 下，让它"对齐"到 coarse scale 的 coordinate system，同时保留对 newly visible region 的预测能力。

进一步 refinement：
- **Segment-wise depth alignment**：用 SAM [Kirillov et al. 2023](https://arxiv.org/abs/2304.02643) 生成的 mask 在每个 segment 内做 scale/shift 对齐——这吸收了 monocular depth estimator 的 scale ambiguity（绝对深度难，相对深度容易）
- **Newly added structure**: 用 Grounded SAM [Ren et al. 2024](https://arxiv.org/abs/2401.14159) 定位 user-specified 新结构（如 ladybug），单独 estimate depth 但约束与 surrounding 一致

这与 WonderJourney [Yu et al. 2024](https://arxiv.org/abs/2312.03884) 和 WonderWorld [Yu et al. 2025](https://wonderworld-2024.github.io/) 的 segment-wise alignment 思想一脉相承。

### 3.3 Stage 3: Scale-Adaptive Surfel Generation

从 $(\mathbf{I}_i, \mathbf{D}_i, \mathbf{C}_i)$ 生成 pixel-aligned surfels：
- 每个 pixel 对应一个 surfel
- Position $\mathbf{p}$：用 $\mathbf{D}_i$ 做 back-projection
- Native scale $s^{\mathrm{native}}$：基于 $\mathbf{C}_i$ 的 focal length 计算

这一步输出 $\mathcal{E}_i^{\mathrm{partial}}$，因为只有一个 view，无法 construct 完整 3D scene。

### 3.4 Stage 4: Auxiliary View Synthesis

这是关键的"补全"步骤：

1. 从 $\mathbf{C}_i$ 周围 sample K 个 neighboring views $\{\mathbf{C}_i^k\}$
2. 用 $\mathcal{E}_i^{\mathrm{partial}}$ 渲染 conditioning frames $\{\mathbf{O}_i^k\}$
3. 计算 masks $\{\mathbf{M}_i^k\}$ 指示需要 synthesis 的 regions（occlusion、未覆盖 area）
4. 用 camera-controlled video diffusion model（Gen3C [Ren et al. 2025](https://arxiv.org/abs/2503.16459)）生成 temporally consistent frames $\{\mathbf{I}_i^k\}$
5. 用 video depth model（GeometryCrafter [Xu et al. 2025](https://arxiv.org/abs/2504.01016)）estimate depth $\{\mathbf{D}_i^k\}$
6. 用所有 image-depth pairs 优化更完整的 $\mathcal{E}_i$

**Intuition**：单张 image 无法 constrain 完整 3D scene。通过 video diffusion model "幻觉" 出 neighboring views，相当于让 generative prior 填充 unobserved regions。这与 CAT3D [Gao et al. 2024](https://arxiv.org/abs/2405.10314) 的 multi-view diffusion 思路相似，但 WonderZoom 用的是 video model 而非 multi-view image model，从而 leverage temporal consistency。

### 3.5 Stage 5: Optimization

用 $\{\mathbf{I}_i, \mathbf{I}_i^1, \dots, \mathbf{I}_i^K\}$ 优化 $\{o, \mathbf{q}, \mathbf{s}\}$，loss 与 Stage 3 一致。

---

## 四、实验结果深度解析

### 4.1 Quantitative Comparison (Table 1)

| Method | CS↑ | CIQA↑ | QIQA↑ | NIQE↓ | QIAA↑ | Time/s |
|---|---|---|---|---|---|---|
| WonderWorld [51] | 0.2687 | 0.5064 | 1.081 | 21.74 | 1.339 | 9.3 |
| HunyuanWorld [35] | 0.2510 | 0.2827 | 1.058 | 15.21 | 1.302 | 704.2 |
| Gen3C [32] | 0.3004 | 0.5489 | 2.992 | 4.924 | 2.018 | 306.7 |
| Voyager [14] | 0.2609 | 0.5746 | 3.148 | 4.913 | 2.929 | 596.6 |
| **WonderZoom** | **0.3432** | **0.7035** | **3.926** | **3.695** | 2.986 | 62.1 |

关键观察：
- **CS (CLIP Score)**：WonderZoom 0.3432，比 Gen3C 高 14.2%——text alignment 最好，因为 explicit prompt editing
- **CIQA (CLIP-IQA+)**：0.7035，比次优的 Voyager (0.5746) 高 22.4%——perceptual quality 大幅领先
- **NIQE (lower better)**：3.695，最低——无 reference 质量指标最佳
- **Time**：62.1 秒，介于 WonderWorld (9.3) 和 Gen3C (306.7) 之间——quality-speed tradeoff 合理
- **HunyuanWorld 异常慢** (704.2s)：可能因 mesh-based representation 的 dense optimization
- **Voyager 在 QIAA (aesthetic) 上略胜** (2.929 vs 2.986)：可能因为 video model 的 high-quality prior

### 4.2 Human Study (Table 2)

| 对比 | Zoom-in Accuracy | Visual Quality | Prompt Match |
|---|---|---|---|
| vs WonderWorld | 80.7% | 98.3% | 98.2% |
| vs HunyuanWorld | 83.2% | 98.7% | 98.9% |
| vs Gen3C | 77.8% | 83.8% | 96.1% |
| vs Voyager | 76.1% | 81.7% | 90.9% |

**有趣现象**：Visual Quality 的 favor rate 高达 98.7%（vs HunyuanWorld）——几乎一边倒。Prompt Match 也几乎一边倒。Zoom-in Accuracy 略低（76-83%），说明 video methods (Gen3C, Voyager) 在 zoom perception 上仍有竞争力，但 visual quality 不如 explicit 3D。

### 4.3 Ablation: Opacity Modulation (Table 3)

| Method | GPU Memory | FPS |
|---|---|---|
| Ours w/o mod. | 7.96G | 1.4 |
| Ours | 3.40G | 97.2 |

**惊人差异**：opacity modulation 让 FPS 从 1.4 跳到 97.2——**70× 加速**。GPU memory 从 7.96G 降到 3.40G——**57% 减少**。

Why? 没有 modulation 时，所有 surfels 在所有 scales 都 fully rendered，造成：
1. 大量 redundant computation（远处 coarse surfels 在 zoom-in 时也参与 rendering）
2. Aliasing（多个 scale 的 surfels 互相 conflict）
3. Memory overhead（所有 surfel fragments 都要 store）

Opacity modulation 让 surfels 只在 native scale 附近可见，相当于 soft culling——既加速又避免 aliasing。

### 4.4 Ablation: Depth Registration (Figure 6)

去掉 depth registration 导致 beetle 形状严重 distorted，因为 monocular depth prediction 的 scale 与 coarse geometry 不一致。这与 WonderJourney/WonderWorld 的经验一致——depth scale alignment 是 single-image-to-3D 的关键。

### 4.5 Ablation: Auxiliary View (Figure 7)

去掉 auxiliary view synthesis 导致 grey areas（未覆盖 regions）。这验证了 single image 不足以 constrain complete 3D scene，需要 generative prior 补全。

---

## 五、Failure Cases 与 Limitations

Figure 13 显示的 failure case：repeatedly zoom into tree branches 时，最终 collapse 到 pure texture patterns，没有 semantic cues 可供 next scale 推断。这暴露了一个 fundamental limitation：**WonderZoom 依赖 semantic context 来 guide generation**。当区域变成 pure texture（如 fractal-like pattern），semantic prior 失效，进一步 generation under-constrained。

可能的 future direction（paper 自己提到）：
- **Texture-specific priors**：训练专门处理 texture 的 generative model
- **Procedural generation**：用 procedural rules 生成 micro-structures（如 fractal noise、Wang tiling）
- **Self-similar priors**：detect self-similar patterns 并 extrapolate

这与 "Powers of Ten" 的 cosmos-to-quark zoom 形成有趣对比——物理学在不同 scale 有不同 laws，而 generative model 需要在每个 scale 学到 specific prior。

---

## 六、相关联想与 broader context

### 6.1 与 Powers of Ten 的对比

Generative Powers of Ten [Wang et al. 2024](https://arxiv.org/abs/2405.18447) 是 2D 版本，用 coordinated diffusion processes 联合 sample 多个 scales。WonderZoom 是 3D 版本，但 pipeline 完全不同：
- GPoT: joint sampling，需要协调 multiple diffusion processes
- WonderZoom: sequential sampling，每个 scale 独立 generate，靠 representation 保证 consistency

WonderZoom 的 sequential approach 更 scalable，可以"无限"zoom（理论上无上限），而 GPoT 需要预先确定 scale 数量。

### 6.2 与 Infinite Nature 系列

Infinite Nature [Liu et al. 2021](https://arxiv.org/abs/2012.12270)、Infinite Nature-Zero [Li et al. 2022](https://arxiv.org/abs/2206.08913)、Persistent Nature [Chai et al. 2023](https://arxiv.org/abs/2301.01814)、DiffDreamer [Cai et al. 2023](https://arxiv.org/abs/2210.14064) 都是 perpetual view generation——forward camera motion 沿 trajectory 生成。这与 WonderZoom 的 zoom-in 不同：
- Infinite Nature: 横向 traversal
- WonderZoom: 纵向 zoom（depth-like）

WonderZoom 的 zoom 是 "scale 维度"的 navigation，而 Infinite Nature 是 "spatial 维度" 的 navigation。两个方向可以正交 combine，未来工作可能 explore。

### 6.3 与 Cascaded Generation

Progressive GANs [Karras et al. 2018](https://arxiv.org/abs/1710.10196)、Cascaded Diffusion [Ho et al. 2022](https://arxiv.org/abs/2106.04552) 都是 coarse-to-fine generation。WonderZoom 的 progressive detail synthesizer 与之精神相似，但本质不同：
- Cascaded generation: fixed scales，end-to-end training
- WonderZoom: arbitrary scales，autoregressive，每个 scale 用 existing tools (SR, Edit, Video Diffusion) 组合

WonderZoom 的"组合现有工具"思路类似 Lego blocks，更 modular，更容易 leverage foundation model 进步。

### 6.4 与 3DGS LoD 系列

Hierarchical 3DGS [Kerbl et al. 2024](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)、Octree-GS [Ren et al. 2024](https://arxiv.org/abs/2403.17898)、Scaffold-GS [Lu et al. 2024](https://arxiv.org/abs/2312.00109) 都是 LoD for 3DGS，但都假设 complete multi-scale supervision。WonderZoom 的 scale-adaptive surfels 是 "generation-friendly" 版本——append-only, opacity modulation for smooth transition。

可以想到一个 unification：在 reconstruction scenario，scale-adaptive surfels 也可以作为 hierarchical 3DGS 的 alternative，允许 incremental reconstruction at multiple scales。

### 6.5 与 City-Scale Generation

CityDreamer [Xie et al. 2024](https://arxiv.org/abs/2309.01570)、GaussianCity [Xie et al. 2024](https://arxiv.org/abs/2406.06526)、Syncity [Engstler et al. 2025](https://arxiv.org/abs/2503.16420)、Infinicity [Lin et al. 2023](https://arxiv.org/abs/2304.02715) 是 city-scale 3D generation。这些方法是 WonderZoom 在 macro scale 的"competitors"，但都缺乏 zoom-in 能力。

一个有趣的 future direction 是把 WonderZoom 的 multi-scale framework apply 到 city scale：从 aerial view zoom 到 street view，再 zoom 到 building facade，再 zoom 到 brick texture。这与 Google Earth 的 zoom experience 类似，但 generative。

### 6.6 与 Foundation Models

WonderZoom 组合了：
- VLM: GPT-4V
- Super-resolution: Chain-of-Zoom
- Image editing: controllable diffusion
- Video diffusion: Gen3C
- Monocular depth: MoGe
- Video depth: GeometryCrafter
- Segmentation: SAM, Grounded SAM
- Harmonization: INR-Harmonization

这是 "foundation model orchestration" 范式——用 LLM/VLM 作为 controller，组合多个 specialized models 完成复杂 task。与 WonderWorld 用 LLM 做 scene planning 类似。

Future direction 可能用单一 multimodal model (如 GPT-5 vision, Gemini 3) 替代这个 orchestration，端到端学习 multi-scale generation。

### 6.7 与 Fractal Generation

从数学 perspective，WonderZoom 实际上生成的是 "approximate fractal"——自相似的 multi-scale 结构。Benoît Mandelbrot 的 fractal geometry 强调 natural scenes 的 self-similarity。WonderZoom 通过 VLM 提取 semantic context 间接 capture 这种 self-similarity（"a ladybug on a sunflower in a field" 与 "a field of sunflowers in a meadow" 有 structural similarity）。

Future 可能用 fractal-aware priors，explicitly model scale-invariant structures。

---

## 七、Critical Thoughts

### 7.1 Partition of Unity 的几何意义

Proposition 1 的 partition of unity 是非常 elegant 的 design。让我 deeper intuition：

考虑 zoom-in 操作：相机 focal length 从 $f_{i-1}$ 增到 $f_i = 8 f_{i-1}$。在 zoom transition 中（focal length 连续变化），$s^{\mathrm{render}}$ 也连续变化。Surfels from $\mathcal{E}_{i-1}$ 的 $s^{\mathrm{native}}$ 与 surfels from $\mathcal{E}_i$ 的 $s^{\mathrm{native}}$ 差 8 倍。在 transition zone（log space 1 个 unit），两者 alpha 互补——一个 fade in，一个 fade out。这就像 cross-fade in audio mixing，避免 click artifacts。

更深层的 insight：partition of unity 是 manifold 上的 partition of unity 推广。在 Riemannian geometry 里，partition of unity 用来 patch local charts 成 global manifold。WonderZoom 用它 patch local scale representations 成 global multi-scale scene。

### 7.2 为什么 log-space linear interpolation?

Linear interpolation in log space = geometric interpolation in linear space。

如果 $s^{\mathrm{render}} = s^{\mathrm{native}} \cdot \exp(t)$ 其中 $t \in [0, \log(s^{\mathrm{parent}}/s^{\mathrm{native}})]$，则 $\alpha$ 是 $t$ 的 linear function。这相当于在 zoom ratio 上做 linear interpolation，而非 absolute scale。

这与 human perception 一致：8x zoom 与 16x zoom 之间的 "perceptual distance" 是 1 unit (log_2(16/8)=1)，与 1x zoom 到 2x zoom 之间的 perceptual distance 相同。这就是 Weber-Fechner law。

### 7.3 Append-only 与 Persistent Data Structures

WonderZoom 的 append-only design 让我想到 Clojure 的 persistent data structures、Git 的 immutable history、CRDT (Conflict-free Replicated Data Types)。这些都是 functional programming 的核心 idea：immutable updates，structural sharing。

在 3D representation context，append-only 意味着：
- 可以 undo 到任何 historical scale（删除后续 surfels）
- 不同 users 可以并行 explore 不同 zoom paths（branching）
- 多人 collaboration 可以 merge 不同 zoom paths

这可能 enable "multi-user collaborative 3D world exploration" 的 future application。

### 7.4 Generative vs. Reconstruction Paradigm

WonderZoom 的 fundamental contribution 是把 multi-scale representation 从 reconstruction paradigm 扩展到 generation paradigm。这是 representation 的"范式 shift"。

Reconstruction: 数据驱动，complete supervision
Generation: prior 驱动，partial supervision

这种 paradigm shift 在 NeRF→3DGS→Generative 3D 演化中已经发生。WonderZoom 在 multi-scale dimension 重复了这个 shift。

### 7.5 与 Diffusion U-Net 的类比

Diffusion models 的 U-Net architecture 本身有 multi-scale structure（downsample → bottleneck → upsample with skip connections）。这与 WonderZoom 的 multi-scale scene 有 structural similarity。

但本质不同：
- U-Net: spatial multi-scale，所有 scales 在 single forward pass
- WonderZoom: scale multi-scale，每个 scale 是独立 generation step

可能 future work 用 single forward pass 生成所有 scales？这与 "Generative Powers of Ten" 的 joint sampling 思路类似。但 3D case 的 computational cost 让 sequential approach 更 practical。

---

## 八、Web Links for Reference

**Project & Paper**:
- WonderZoom Project Page: https://wonderzoom.github.io/
- Stanford 3D generation group: https://cs.stanford.edu/~hxyu/

**Foundational Works**:
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Hierarchical 3DGS: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/
- Mip-NeRF: https://jonbarron.info/mipnerf/
- Mip-NeRF 360: https://jonbarron.info/mipnerf360/
- Zip-NeRF: https://jonbarron.info/zipnerf/
- Mip-Splatting: https://github.com/autonomousvision/mip-splatting
- Octree-GS: https://github.com/cnjr2000/octree-gs
- Scaffold-GS: https://github.com/city-super/Scaffold-GS

**Predecessors (Stanford)**:
- WonderJourney: https://WonderJourney.github.io/
- WonderWorld: https://wonderworld-2024.github.io/

**Multi-scale Generation**:
- Generative Powers of Ten: https://powers-of-10.github.io/
- Infinite Nature: https://infinite-nature.github.io/
- Persistent Nature: https://persistent-nature.github.io/

**City-scale Generation**:
- CityDreamer: https://citydreamer.github.io/
- GaussianCity: https://github.com/huster1234/GaussianCity
- InfiniCity: https://mrta1024.github.io/

**Components**:
- Chain-of-Zoom: https://github.com/KimSungHwan06/Chain-of-Zoom
- Gen3C: https://gen3c-stability.github.io/
- MoGe: https://wangrc.site/MoGe/
- GeometryCrafter: https://geometrycrafter.github.io/
- SAM: https://github.com/facebookresearch/segment-anything
- Grounded SAM: https://github.com/IDEA-Research/Grounded-Segment-Anything

**Single-image 3D Generation**:
- LucidDreamer: https://luciddreamer-team.github.io/
- CAT3D: https://cat3d.github.io/
- SceneScape: https://scrolling.github.io/scenescape/
- Text2Room: https://text-to-room.github.io/
- LayerPanen3D: https://layerpano3d.github.io/
- Flash3D: https://www.robots.ox.ac.uk/~vgg/flash3d/
- DreamScene360: https://dreamscene360.github.io/

---

## 九、Conclusion: 我对 WonderZoom 的 Intuition 总结

WonderZoom 的核心 insight 是：**multi-scale 3D generation 的 bottleneck 在 representation，不在 generator**。一旦有了 scale-adaptive 的 representation（dynamic updating + opacity modulation），现有 foundation models (SR, Edit, Video Diffusion) 自然可以组合成 progressive synthesizer。

Scale-adaptive surfels 的关键 trick 是 **native scale field** + **log-space opacity modulation**，partition of unity 保证 seamless transition。这是一个 elegant 的 mathematical design——简单但 powerful。

Append-only updating 是另一个 deep insight——把 multi-scale representation 从 "global optimization" paradigm 转向 "incremental addition" paradigm。这与 functional programming 的 persistent data structures、Git 的 immutable history、blockchain 的 append-only ledger 是同一思想在不同领域的 instantiation。

Limitation 在 texture-only regions 的 failure 暴露了 semantic-dependent generation 的边界。Future work 可能需要 procedural + semantic hybrid approaches——symbolic rules 处理 texture，neural models 处理 semantics。

总而言之，WonderZoom 是 single-image-to-multi-scale-3D 这个新 problem 的 solid first solution，开了 representation design 的 direction，留给 future work 大量空间探索（end-to-end learning、fractal priors、collaborative editing、city-scale extension）。

---

*Last note: 作为一个 teaching tool，WonderZoom 的 paper 写得非常清晰，公式和算法 description 直接可 implement。Supplementary 的 Algorithm 1 是完整 pseudocode，配合 paper body 应该可以复现。Code release 已 announce，期待 community 在此基础上探索。*
