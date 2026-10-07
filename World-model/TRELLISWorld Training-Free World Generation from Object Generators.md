---
source_pdf: TRELLISWorld Training-Free World Generation from Object Generators.pdf
paper_sha256: 7f9fca8909ba5cd8d454c59df9bd7868b9015c7a7a46eaa8324e4cad5e2956bd
processed_at: '2026-08-12T18:16:25-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TRELLISWorld 用人话讲

## 这篇paper到底在干嘛

一句话: **拿一个能生成单个3D物体的模型，拼出一整个3D世界，不用重新训练**。

打个比方。你有一个烤箱，只能烤6寸蛋糕。但你想要一个3尺大的婚礼蛋糕。怎么办？你可以烤很多个6寸小蛋糕，然后拼在一起。问题是怎么拼得看不出接缝。这篇paper就是解决这个拼接问题的。

参考链接:
- TRELLIS原文: https://arxiv.org/abs/2412.01506
- 本文TRELLISWorld: 从attachment看

---

## 为什么这个问题难

### 3D场景生成的困境

现在3D生成领域有个尴尬局面:

**物体生成**很成熟了。TRELLIS、Hunyuan3D、Clay这些模型，给一句话就能生成一个像样的3D物体。Objaverse数据集有1000万个3D物体可以训练。

**场景生成**却很拉胯。原因:
1. 没有3D场景数据集。最大的FurniScene只有10万个房间，89类物体。跟Objaverse的1000万差了100倍
2. 场景有long-range dependencies。一个城市的左边和右边要风格统一，这个约束很难学
3. 场景太大了。物体生成模型训练时只见过小的tensor（比如16³），直接喂个大的就out of distribution了

现有的场景生成方法都有硬伤:
- **Domain-specific方法**（CityDreamer4D、BlockFusion、MIDI）只能在特定领域用，训练数据很窄
- **2D方法**（SynCity、WonderJourney）先在2D图像上生成，再reconstruct成3D。2D生成有error，这个error会propagate到3D，越拼越歪

参考:
- SynCity: https://arxiv.org/abs/2503.16420
- CityDreamer4D: https://arxiv.org/abs/2406.06526
- BlockFusion: https://arxiv.org/abs/2401.17053
- MIDI: https://arxiv.org/abs/2412.03558

### 为什么不直接train一个scene generator

因为没数据。Objaverse是1000万级别的物体数据集。3D场景数据集呢？FurniScene才10万房间。要train一个general的scene generator，数据量差太远了。

而且场景的diversity远超物体。一个物体就是一把椅子、一栋楼。一个场景可以是城市、森林、海底、太空站。标注和curate这些数据的成本极高。

所以TRELLISWorld的核心思路: **不训练场景模型，直接拿物体模型来拼**。物体模型已经见过1000万个物体了，它的prior足够丰富。只要解决"怎么拼"这个问题就行。

---

## 核心Idea: Tiled Diffusion

### Intuition

Diffusion model生成东西的过程是这样的: 从一个全噪声开始，一步一步denoise，最后变成一个干净的sample。

TRELLISWorld说: 既然如此，我搞一个**大噪声tensor**（比如整个世界的size），分成很多**重叠的小块tiles**，每个tile单独denoise一步，然后在重叠区域做**加权平均**，合成回大tensor。重复这个process直到完全denoise。

这个idea的精妙之处在于:

1. **每个tile独立denoise**，所以每个tile只占小内存，可以并行
2. **重叠区域加权平均**，自然消除了接缝
3. **整个过程在latent space做**，不在pixel/voxel space做，更高效更smooth
4. **不需要重新训练**，直接用物体生成模型

### 为什么重叠能消除接缝

想象两个tile，A和B，它们在边界处有overlap。

如果不overlap（紧贴），A的右边缘和B的左边缘是独立生成的，大概率对不上，产生硬接缝。

如果overlap，在overlap区域，A和B都给出了预测。我们用一个weight function，让A的中心权重大、边缘权重小，B也是中心权重大、边缘权重小。在overlap区域，两个预测加权平均，自然smooth过渡。

这就像Photoshop里拼两张照片，边缘做feather羽化。cosine weighting就是这个feather的数学形式。

### 跟MultiDiffusion的关系

这个idea在2D图像生成里已经有先例。MultiDiffusion（2023年）做了类似的事情: 把大图分成重叠的小块，每块独立denoise，重叠区域smooth blending。

TRELLISWorld是这个idea的3D版本。但3D有额外挑战:
1. 3D表示更复杂（occupancy grid、sparse tensor、Gaussian Splatting）
2. 多stage diffusion（TRELLIS有两个stage）
3. Decoder也要tiled

参考:
- MultiDiffusion: https://arxiv.org/abs/2302.08113
- SyncDiffusion: https://arxiv.org/abs/2306.05490

---

## 具体怎么做: 数学细节

### Cosine Weighting Mask (公式1)

$$\beta(x', y', z') = \prod_{d \in \{x', y', z'\}} \cos\left(\pi\left(\frac{d+1}{S+1} - \frac{1}{2}\right)\right)$$

讲讲变量:
- $x', y', z'$ 是tile内的local坐标，从0到$S-1$。$S$是tile的size，TRELLIS中是16
- $d$是遍历三个维度$\{x', y', z'\}$
- 三个维度的cosine相乘，得到3D weight

我们分析一下这个函数的行为:

**中心位置** $d = (S-1)/2 = 7.5$:
$$\cos\left(\pi\left(\frac{8.5}{17} - \frac{1}{2}\right)\right) = \cos(0) = 1$$

中心权重最大，等于1。

**边缘位置** $d = 0$ 或 $d = S-1 = 15$:
$$\cos\left(\pi\left(\frac{1}{17} - \frac{1}{2}\right)\right) = \cos(-15\pi/34) \approx 0.187$$

边缘权重最小，约0.19。

**三维相乘后**: 中心是$1 \times 1 \times 1 = 1$，边缘是$0.19 \times 0.19 \times 0.19 \approx 0.0068$。边缘权重极小，几乎为0。

这个设计很巧妙:
- 中心完全保留tile自己的内容
- 边缘几乎不贡献，所以相邻tile在边界处smooth handoff
- 不需要显式定义"哪里是边界"，cosine自然taper off

**为什么不直接用linear ramp或Gaussian?** cosine在边界处更平滑，导数更小，产生的blending更自然。Gaussian需要选sigma，linear在边界处有kink（一阶导数不连续）。cosine是一个parameter-free的选择。

### Update Rule (公式2)

$$v_{t-\Delta t}^{(x,y,z)} = \frac{\sum_{\{w_i^{(t)}\}} \beta(f_{w_i^{(t)}}(x,y,z)) \cdot \left[w_i^{(t)} - \Delta t \cdot \theta(w_i^{(t)}, t) + \mathcal{O}(\Delta t^2)\right]_{f_{w_i^{(t)}}(x,y,z)}}{\sum_{\{w_i^{(t)}\}} \beta(f_{w_i^{(t)}}(x,y,z))}$$

逐个拆解:

- $v_{t-\Delta t}^{(x,y,z)}$: 在global position $(x,y,z)$ 处，下一time step的voxel值
- $w_i^{(t)}$: 第$i$个tile在time step $t$的state
- $f_{w_i^{(t)}}(x,y,z)$: 把global坐标映射到tile $w_i$的local坐标
- $\beta(\cdot)$: 上面那个cosine权重
- $\theta(w_i^{(t)}, t)$: diffusion model（velocity field）在tile $w_i$、time $t$的output
- $\Delta t$: time step size
- $\mathcal{O}(\Delta t^2)$: Euler discretization的truncation error
- $[\cdot]_{f_{w_i^{(t)}}(x,y,z)}$: 在local tile space里取对应位置的值

**用大白话讲这个公式在干什么**:

对每个global voxel $(x,y,z)$:
1. 找到所有覆盖它的tiles（一个voxel可能被多个tile覆盖，因为tiles overlap）
2. 每个tile独立做一步Euler denoising: 当前值减去 $\Delta t \times$ velocity field的output
3. 用cosine weight对每个tile的denoising结果加权
4. 归一化（除以总weight）

**深层intuition**: 这相当于在做**local denoising的ensemble**。在overlap区域，多个tiles给出不同的predictions。如果它们agree，averaging后结果更robust。如果它们disagree，averaging会smooth out差异，起到agreement constraint的作用。

从数学角度看，这modifies了原本的ODE/SDE的drift term。原本是 $\theta(w, t)$，现在变成了 $\sum_i \beta_i \cdot \theta(w_i, t) / \sum_i \beta_i$。这可以看作是一个spatially-smoothed velocity field。

参考flow matching理论: https://arxiv.org/abs/2210.02747

---

## 在TRELLIS上实现

TRELLIS是个multi-stage的latent diffusion model。它的pipeline:

```
Text Prompt
    ↓
[θ₁: Structure Diffusion Transformer] 
    → 16³ dense latent (压缩表示)
    ↓
[Sparse Structure Decoder] 
    → 64³ occupancy grid (SS，标记哪里有物体)
    ↓
[Convert to sparse tensor, keep voxels where occupancy > 0]
    ↓
[Apply noise to sparse regions]
    ↓
[θ₂: Structure Latent Diffusion Transformer] 
    → sparse latent (SLAT)
    ↓
[Structure Latent Gaussian Decoder] 
    → Gaussian Splatting (最终3D表示)
```

TRELLISWorld要在每个stage都做tiled:

### Stage 1: $\theta_1$ (dense latent)

输入是16³的dense latent。tiles在这个latent space里切分。Masks要down-sample 4×，因为latent比voxel space小4倍。

### Stage 2: $\theta_2$ (sparse latent)

输入是sparse latent。同样tiled处理。Blending在latent space做。

**为什么在latent space做blending?**
1. 计算效率: latent是16³，voxel是64³，快64倍
2. Latent space更smooth，blending效果更好
3. Diffusion model本身就在latent space operate，保持consistency

### Decoder也要Tiled

这是个nontrivial的细节。TRELLIS的structure latent Gaussian decoder不是probabilistic model。所以set stride $s = S$，即disable blending，每个tile独立decode。

Figure 4显示，如果直接把整个大latent喂给decoder（不tiled），会有severe artifacts。因为decoder训练时只见16³输入，直接喂大的就OOD了。

**为什么decoder不需要blending?** 因为decoder是deterministic mapping（不是probabilistic），给定latent，输出是确定的。只要latent在overlap区域已经blended好，decode出来的结果自然coherent。

---

## 实验结果

### 跟SynCity对比

SynCity是之前SOTA的training-free方法。它的pipeline:
1. 2D inpainting生成下一chunk的image
2. 用image-to-3D模型生成3D chunk
3. 用3D inpainting修复seams

这个pipeline的问题: 2D inpainting有error，error propagate到3D。而且autoregressive，慢。

TRELLISWorld的优势:

| 指标 | SynCity | TRELLISWorld |
|------|---------|--------------|
| 速度 | 452 sec/chunk | 78 sec/chunk (5.8× faster) |
| CLIP score | 0.260 | 0.265 |
| CannyAvg (seam visibility) | 7.82 | 5.61 (less seams) |
| 内存 | RTX 4090 48GB才能跑 | RTX 4080 16GB能跑 |

### Ablation Studies

**1. Tiled Diffusion vs Autoregressive** (Figure 3)
Autoregressive方法在chunk edges有less coherent generation。TRELLISWorld在所有themes（desert、cyberpunk、forest）都更smooth。

**2. Tiled Decoder** (Figure 4)
不用tiled decoder会有severe artifacts。因为decoder没见过大输入。

**3. Blending方法** (Figure 5)
Average blending会在tile borders产生visible walls和colored edges。Cosine blending让borders几乎不可见。

### Stride Analysis (Figure 8)

- $s = 16$ (no overlap): seams明显
- $s = 8$ (50% overlap): 最优trade-off
- $s < 8$: quality提升不显著，computation linearly增加

**Intuition**: stride越小，overlap越大，blending越smooth，但tiles越多，计算越慢。$s = S/2$ 是个sweet spot。

---

## 为什么这个方法Work

### 1. Locality of Diffusion Denoising

Diffusion model的velocity field $\theta(w, t)$主要依赖local neighborhood。虽然attention理论上能capture long-range dependencies，但实际训练后，denoising step的效果主要是local的。所以独立tile denoising是合理的approximation。

### 2. Overlap Region as Agreement Constraint

在overlap region，多个tiles同时denoise。如果它们disagree，cosine weighting会average out差异。这相当于implicit agreement constraint，鼓励tiles在overlap region produce consistent predictions。

### 3. Latent Space Smoothness

TRELLIS的latent space是learned的，比raw voxel space更smooth。Blending在latent space做，decode出来自然更coherent。

### 4. Multi-Scale Signal Structure of 3D Environments

3D环境有multi-scale structure。一座城市，local看是建筑细节，global看是街区layout。Object generator的prior已经encode了这个multi-scale structure。Tiled generation只是把local prior应用到global scale。

---

## Image-Conditioned Model为什么会失败 (Section A.5)

这是paper里一个深刻的分析。

他们试了把method应用到image-conditioned的TRELLIS模型。结果: floor level都对不齐。

为什么? 论文用数学解释:

**Text-conditioned和image-conditioned model有相同的marginalized distribution**:
$$q_{\theta_{img}}(x_0) = \int q_{\theta_{img}}(x_0|c_{img}) p(c_{img}) dc_{img} = \int q_{\theta_{text}}(x_0|c_{text}) p(c_{text}) dc_{text} = q_{\theta_{text}}(x_0)$$

**但conditional distribution不同**:
$$q_{\theta_{text}}(x_0|c_{text}) \neq q_{\theta_{img}}(x_0|c_{img})$$

即使text和image对应同一个物体。

**Key insight**: text prompt有inherent ambiguity。一句"a chair"可以是任何椅子。模型学到broad distribution。而image condition更precise，模型学到narrow distribution。

在tiled generation中，overlap region的不同tiles给出slightly different conditions。Text-conditioned model因为distribution diffuse，对这种perturbation robust。Image-conditioned model因为distribution sharp，slightly different condition导致significantly different output。

**General principle**: diffuse conditional distribution更适合compositional generation。这对未来设计scene generation model有指导意义。可能text-conditioned model天生更适合做这种tiled composition。

---

## Applications

### 1. Area-Specific Prompting (Figure 12)

每个tile可以有不同prompt。Prompt organized as3D tensor。比如:

```lua
prompt = {
  {"Spring forest... blooming flowers"},
  {"Summer meadow..."},
  {"Autumn woods..."},
  {"Winter ice lake... skating marks"}
}
```

Sample nearest prompt for each tile。这实现了smooth semantic transitions across scene。比如从春到冬，从森林到冰湖。

### 2. Scene Expansion (Figure 11)

用RePaint with Gaussian-blurred mask:
- Initialize noise with parts of ground truth
- Blurred mask preserves edges
- Encourages smoother transitions

给一个1×1×1 chunk，能extend成3×3×1 scene。

### 3. 3D Tiling (Figure 13)

不限于2D surface，可以3D tiling:
- Group of fish (2×2×2 chunk)
- Castle in the Sky (1×1×2 chunk)
- 用area-specific prompting做3D blending

这很impressive。地球上大多数macro-structure是2D surface，但3D tiling能生成空中城堡、鱼群这种真正3D的东西。

---

## Limitations

1. **Dependence on base model**: TRELLIS的能力直接限制scene quality
2. **Object-level separation**: 无法disentangle individual objects post-generation。因为所有tiles在一个batch里生成，生成完之后没法分离单个物体

---

## 我的理解: 这篇paper的insight

### Insight 1: Composition > End-to-End Training

这篇paper是composition philosophy的胜利。与其train一个end-to-end的scene generator（需要海量数据、海量compute），不如用composition把object generator拼成scene generator。

这跟LLM里的modular reasoning、tool use有异曲同工之妙。大模型不是万能的，composition和tool use能解决scaling问题。

### Insight 2: Overlap > Hard Boundary

Overlap region + weighted blending是消除artifact的关键。这个idea在2D image generation（MultiDiffusion）、panorama stitching、texture synthesis里都有出现。本质上是soft constraint比hard constraint更robust。

### Insight 3: Latent Space > Pixel Space

Blending在latent space做比在pixel/voxel space做好太多。Latent space是learned的，更smooth，更semantic。这跟VAE、diffusion autoencoder的philosophy一致。

### Insight 4: Diffuse Distribution适合Composition

Text-conditioned model的diffuse conditional distribution让它对compositional generation robust。这是个deep insight。未来设计compositional system时，应该考虑condition的diffuseness。

### Insight 5: Simplicity Wins

这个method的核心idea一句话能讲清。没有复杂的architecture，没有精巧的loss，没有海量数据。就是cosine blending + tiled diffusion。这跟Occam's razor一致。

---

## 相关联想

### 1. 跟LLM Context Window的类比

LLM用sliding window attention处理long context。Tiled diffusion是3D generation的analog:
- Sliding window = overlapping tiles
- Attention blending = cosine weighted blending
- KV cache = tile state cache

LLM里也有类似insight: long context不需要global attention，local attention + overlap就够了。

### 2. 跟Texture Synthesis的connection

Computer graphics里有classic problem: texture synthesis。给一小块texture，合成大块texture。方法就是patch-based: 切小块，overlap，blend。Efros & Leung 1999, Image Quilting 2001都是这个思路。

TRELLISWorld是3D generation领域的texture synthesis。数学结构几乎一样:
- 小patch → 大texture
- 小tile → 大scene
- Overlap + blend = seamless transition

参考:
- Texture Synthesis by Non-parametric Sampling: https://www.eecs.yorku.ca/~kosta/CompVis_NP_files/efros-iccv99.pdf
- Image Quilting: https://www.eecs.yorku.ca/~kosta/CompVis_NP_files/efros-siggraph01.pdf

### 3. 跟Mesh Generation的future

TRELLISWorld输出的是Gaussian Splatting。Gaussian Splatting适合rendering，但不适合edit。如果能输出mesh，会更实用。

可能的方向:
1. 用marching cubes把Gaussian Splatting转成mesh
2. 直接train一个mesh decoder（像BRepGen、MeshGPT那样）
3. Hybrid representation: Gaussian for rendering, mesh for editing

参考:
- MeshGPT: https://arxiv.org/abs/2311.11275
- BRepGen: https://arxiv.org/abs/2401.15063

### 4. 跟Sora/Video Generation的类比

Sora生成视频，也是分块的。Sora用spatiotemporal patches，把video切成3D patches（2D space + 1D time）。TRELLISWorld的3D tiling是空间维度的，如果加上time维度，就是4D tiling。

未来可能: tiled video generation。Video太长时，分成overlapping的spatiotemporal chunks，每个chunk独立denoise，overlap区域blend。这能解决long video generation的memory问题。

参考:
- Sora technical report: https://openai.com/sora
- VideoLDM: https://arxiv.org/abs/2304.08818

### 5. 跟Procedural Generation的connection

Game开发里procedural generation已经用了decades。Minecraft用Perlin noise生成terrain，用tile-based方法生成dungeon。

TRELLISWorld是neural版的procedural generation。传统方法用hand-crafted rules，TRELLISWorld用learned priors。这可能是future of game development: artist提供prompt，AI生成世界，artist再edit细节。

参考:
- Procedural Content Generation: https://en.wikipedia.org/wiki/Procedural_generation

### 6. 跟World Model的connection

Yann LeCun一直提倡world model的概念。World model需要能predict环境的dynamics。TRELLISWorld生成的是static world，但加上time维度，就能生成dynamic world。

可能的extension:
1. Tiled 4D generation (3D space + time)
2. 生成world后，用video model添加dynamics
3. Interactive world: user click某个区域，world model predict下一帧

### 7. 跟Robotics Simulation的connection

Robotics需要大量simulation environment来train policy。现实世界data collection很贵，simulation是关键。TRELLISWorld能快速生成diverse的3D环境，这对robotics simulation很有价值。

可能的application:
1. 生成diverse的室内场景，train navigation policy
2. 生成outdoor terrain，train locomotion policy
3. 生成object-rich environment，train manipulation policy

参考:
- Habitat: https://arxiv.org/abs/1907.08440
- iGibson: https://arxiv.org/abs/2012.02924

---

## Future Directions

### 1. Multi-Scale Tiling

当前所有tiles同size。可以multi-scale:
- Coarse tiles (大stride, 大S): capture global layout
- Fine tiles (小stride, 小S): add local details

类似wavelet multi-resolution analysis。不同scale capture不同structure。

### 2. Adaptive Tiling

Complex region（city center）用small stride，simple region（ocean）用large stride。需要scene complexity estimator。

### 3. Hierarchical Generation

1. 先generate low-res global layout (4×3×1 at 16³)
2. 再super-resolve到64³
3. 再add details with fine tiles

类似coarse-to-fine generation in image synthesis。

### 4. Physics-Aware Blending

当前blending纯geometric (cosine weight)。可以add physics constraints:
- Gravity (objects rest on ground)
- Collision (objects don't overlap)
- Light consistency (shadows align)

参考GALA3D: https://arxiv.org/abs/2402.07207

### 5. Interactive Scene Editing

结合area-specific prompting，可以build interactive system:
- User click region, type prompt
- System re-generate only that region
- Blend with surrounding context

这跟Inpaint-anything、Segment-anything的philosophy一致。

### 6. Theory Analysis

Tiled diffusion本质上modifies了SDE/ODE的drift term。理论分析:
- Convergence guarantee?
- Error bound vs stride s?
- 与rectified flow的关系?

这需要stochastic calculus和ODE theory。

参考flow matching: https://arxiv.org/abs/2210.02747

### 7. 4D World Generation

加入time dimension:
- 4D tiles (3D space + time)
- Generate dynamic scenes
- Characters moving, weather changing

---

## 总结

TRELLISWorld是个simple yet powerful的方法。Core idea: **把object generator当tile generator，用cosine blending拼成scene**。Training-free, scalable, editable。

Key insights:
1. Composition比end-to-end training更data-efficient
2. Overlap + weighted blending消除seams
3. Latent space blending比pixel space好
4. Diffuse conditional distribution适合compositional generation
5. Simplicity wins

这个work opens up很多follow-up: multi-scale tiling, adaptive tiling, 4D generation, physics-aware blending, interactive editing, theory analysis。它是一个simple yet extensible foundation for general-purpose 3D world generation。

参考相关工作的landscape:
- Object Generation: TRELLIS (https://arxiv.org/abs/2412.01506), Hunyuan3D 2.5 (https://arxiv.org/abs/2506.16504)
- 2D Scene: WonderJourney (https://arxiv.org/abs/2312.03884), WonderWorld (https://arxiv.org/abs/2406.09394)
- 3D Scene-Native: CityDreamer4D (https://arxiv.org/abs/2406.06526), BlockFusion (https://arxiv.org/abs/2401.17053), MIDI (https://arxiv.org/abs/2412.03558), HunyuanWorld (https://arxiv.org/abs/2507.21809)
- Object-Based: SynCity (https://arxiv.org/abs/2503.16420), GALA3D (https://arxiv.org/abs/2402.07207)

希望这个讲解helps build intuition! 这个paper的beauty在于它的simplicity和generality。它证明了composition philosophy的power，给3D world generation opens up了一条新路。

---

# TRELLISWorld 深度解析

## 1. Big Picture: 这篇论文在解决什么问题

3D scene generation 一直是个 hard problem。现有方法面临几个 fundamental constraints:

- **Object-level generators** (TRELLIS, Hunyuan3D, Clay 等) 只能生成单个 object，无法 scale 到 scene
- **Scene-native methods** (CityDreamer4D, BlockFusion, MIDI) 需要 domain-specific training data，而 3D scene dataset 极其稀缺（最大的 FurniScene 只有 100k rooms，而 Objaverse 有 10M+ objects）
- **2D-based methods** (SynCity, WonderJourney) 依赖 2D inpainting 作为 intermediate，errors 会从 image domain propagate 到 3D

TRELLISWorld 的 key insight 非常 elegant: **把 global scene synthesis reformulate 成 multi-tile denoising problem**。直接 reuse object-level diffusion model 作为 modular tile generator，通过 overlapping regions 和 cosine-weighted blending 实现 coherence。这是 training-free 的，general 的，scalable 的。

参考链接:
- TRELLIS: https://arxiv.org/abs/2412.01506
- SynCity: https://arxiv.org/abs/2503.16420
- Objaverse: https://arxiv.org/abs/2212.08051

---

## 2. Core Method: Tiled Diffusion 的数学详解

### 2.1 Problem Setup

给定一个 text-conditioned 3D generative diffusion model θ（是一个 velocity field，基于 flow matching framework）。这个模型能从 text prompt p 生成 size $S^3$ 的 3D structure。目标是生成 size $(X \times Y \times Z) \gg S^3$ 的大 scale world。

为了简化，论文假设 θ 是 pixel-diffusion model（直接 operate on values，not latents）。每个 object sample 是 $\mathbb{R}^{S^3}$ 中的 tensor。

### 2.2 Tiling Strategy

初始化整个 world $W$ of size $(X, Y, Z)$ with Gaussian noise $W \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

把 world 分成 overlapping cubic tiles $\{w_i\}$，每个 size $(S, S, S)$，stride $(s, s, s)$，其中 $s < S$ 确保 overlap。

**关键 intuition**: stride $s$ 控制 overlap 程度。$s = S$ 表示 no overlap（硬拼接），$s = S/2$ 表示 50% overlap。实验发现 $s = S/2 = 8$ 是 quality 和 computation 的 sweet spot。

### 2.3 Cosine Weighting Mask (公式 1)

$$\beta(x', y', z') = \prod_{d \in \{x', y', z'\}} \cos\left(\pi\left(\frac{d+1}{S+1} - \frac{1}{2}\right)\right)$$

**变量解析**:
- $x', y', z'$: local position within tile, range $[0, S-1]$
- $d$: 遍历 $\{x', y', z'\}$ 三个维度
- $S$: tile size (TRELLIS 中 latent 是 16)

**为什么这样设计?** 让我们分析边界行为:

当 $d = 0$（tile 边缘）:
$$\cos\left(\pi\left(\frac{1}{S+1} - \frac{1}{2}\right)\right) = \cos\left(\pi \cdot \frac{1 - S}{2(S+1)}\right)$$
当 $S = 16$: $\cos(\pi \cdot (-15/34)) = \cos(-1.384) \approx 0.187$

当 $d = (S-1)/2 = 7.5$（tile 中心附近）:
$$\cos\left(\pi\left(\frac{8.5}{17} - \frac{1}{2}\right)\right) = \cos(0) = 1$$

当 $d = S-1 = 15$（另一边缘）:
$$\cos\left(\pi\left(\frac{16}{17} - \frac{1}{2}\right)\right) = \cos(\pi \cdot 15/34) \approx 0.187$$

所以这个 mask 在 tile 中心 weight=1，向边缘 taper off 到 ~0.19，三个维度相乘后边缘 weight 很小。这确保了 overlap region 主要由中心 tiles 贡献，transition 自然 smooth。

**Comparison with average blending**: Figure 5 显示 average blending 会在 tile borders 产生 visible walls（"room" example）和 colored edges（"lego tile" example）。Cosine weighting 通过强调 center, 抑制 edge，自然地 hide seams。

### 2.4 Update Rule (公式 2)

$$v_{t-\Delta t}^{(x,y,z)} = \frac{\sum_{\{w_i^{(t)}\}} \beta(f_{w_i^{(t)}}(x,y,z)) \cdot \left[w_i^{(t)} - \Delta t \cdot \theta(w_i^{(t)}, t) + \mathcal{O}(\Delta t^2)\right]_{f_{w_i^{(t)}}(x,y,z)}}{\sum_{\{w_i^{(t)}\}} \beta(f_{w_i^{(t)}}(x,y,z))}$$

**逐项解析**:

- $v_{t-\Delta t}^{(x,y,z)}$: global position $(x,y,z)$ 处的 voxel 在 time step $t - \Delta t$ 的值（即下一步的 state）
- $w_i^{(t)}$: 第 $i$ 个 tile 在 time step $t$ 的 state
- $f_{w_i^{(t)}}: \mathbb{N}^3 \to \mathbb{Z}^3$: 把 global position 映射到 tile $w_i$ 的 local position。如果 result 落在 $\{0, ..., S-1\}^3$，说明这个 tile 覆盖该 voxel
- $\beta(f_{w_i^{(t)}}(x,y,z))$: cosine weight，根据 local position 计算
- $\theta(w_i^{(t)}, t)$: velocity field (diffusion model) 在 tile $w_i$ 和 time $t$ 的 output
- $\Delta t$: time step size（Euler discretization）
- $\mathcal{O}(\Delta t^2)$: Euler discretization 的 truncation error
- $[\cdot]_{f_{w_i^{(t)}}(x,y,z)}$: 在 local tile space 中取对应位置的值

**这个公式的含义**:
1. 对每个 global voxel $(x,y,z)$，找到所有覆盖它的 tiles
2. 每个 tile 独立做一步 Euler denoising: $w_i^{(t)} - \Delta t \cdot \theta(w_i^{(t)}, t)$
3. 用 cosine weight 加权 average 所有 tiles 的 denoising 结果
4. 归一化（除以 weight sum）

**深层 intuition**: 这其实是在做 **ensemble of local denoising steps**。在 overlap region，多个 tiles 给出不同的 denoising predictions，weighted averaging 相当于 reduce variance，类似 ensemble effect。因为所有 tiles 都基于相同的 global structure（只是 local context 不同），它们在 overlap region 应该 agree，averaging 让 result 更 robust。

参考 flow matching: https://arxiv.org/abs/2210.02747

---

## 3. Implementation on TRELLIS

### 3.1 TRELLIS Architecture Overview

TRELLIS 是 multi-stage latent diffusion model:

```
Text Prompt
    ↓
[θ₁: Structure Diffusion Transformer] (operates on 16³ dense latent)
    ↓
[Sparse Structure Decoder] → 64³ occupancy grid (SS)
    ↓
[Convert to sparse tensor, keep voxels where occupancy > 0]
    ↓
[Apply noise to sparse regions]
    ↓
[θ₂: Structure Latent Diffusion Transformer] (operates on sparse latent)
    ↓
[SLAT: Structured Latent Representation]
    ↓
[Structure Latent Gaussian Decoder] → Gaussian Splatting
```

### 3.2 Tiled Diffusion 适配

TRELLIS 有两个 diffusion stage，都要 tiled:

**Stage 1 (θ₁, dense)**:
- Input: 16³ dense latent
- Tiles 在 encoded latent space
- Masks down-sampled 4× (因为 latent 比 voxel space 小 4×)

**Stage 2 (θ₂, sparse)**:
- Input: sparse latent
- 同样 tiled 处理
- Blending 在 latent space 做

**为什么在 latent space 做 blending?**
1. 计算效率：latent 是 16³，voxel 是 64³，计算量减少 64×
2. Latent space 通常更 smooth，blending 效果更好
3. Diffusion model 本身 operate on latent，在 latent space blend 保持 consistency

### 3.3 Tiled Decoder

Decoder 也要 tiled，但策略不同:

- **Structure latent Gaussian decoder 不是 probabilistic model**
- 所以 set stride $s = S$（disable blending）
- 即每个 tile 独立 decode，不 overlap

Figure 4 显示如果不 tiled decode，会有 severe artifacts。这是因为 decoder 训练时只见 16³ 输入，直接喂大的 64³ 输入会 OOD。

---

## 4. Experiments 详解

### 4.1 Setup

- cfg = 7.5 (classifier-free guidance)
- stride $s = S/2 = 8$
- 25 diffusion steps
- Euler sampler

### 4.2 Ablation Studies

**Tiled Diffusion vs Autoregressive** (Figure 3):
- Autoregressive 方法基于 inpainting，chunk edges 有 less coherent generation
- TRELLISWorld 显示 better blending across tiles

**Tiled Decoder** (Figure 4):
- Without tiled decoder: severe artifacts in Gaussian Splatting
- With tiled decoder: clean output

**Blending methods** (Figure 5):
- Average blending: visible walls at borders, colored edges
- Cosine blending: borders become unnoticeable

### 4.3 Quantitative Results (Table 1)

| Method | CLIP Mean ↑ | CannyAvg Mean ↓ |
|--------|-------------|-----------------|
| SynCity | 0.26020 | 7.81725 |
| inpaint baseline | 0.26419 | 7.71797 |
| avg. blending | 0.26203 | 6.55861 |
| TRELLISWorld | **0.26520** | **5.61331** |

**CannyAvg metric 解释**:
- Orthographic top-down rendering at 1536×2048
- Canny edge detection with threshold [200, 400]
- Average pixel intensity of binary edge map
- Lower = fewer seams = better

为什么 Canny 比 Sobel/Laplacian 好（Figure 15）: Canny 最 align human perception of seams，其他方法 highlight internal content。

### 4.4 Computational Efficiency

- **TRELLISWorld**: 77.96 sec/chunk
- **SynCity**: 452.04 sec/chunk
- **5.80× speedup**

**Memory**:
- SynCity 无法在 RTX 4080 (16GB) 上运行
- 需要 RTX 4090 (48GB)
- TRELLISWorld 内存需求显著更低

**Scaling behavior** (Figure 9, 10):
- Runtime 随 chunk 数 linear 增长
- Per-tile time 保持 27.21 sec constant
- 因为 not autoregressive，可以 multi-GPU parallelize

### 4.5 Stride Analysis (Figure 8)

- $s = 16$ (no overlap): CannyAvg 高，seams 明显
- $s = 8$: 最优 trade-off
- $s < 8$: quality 提升不显著，computation cost 线性增加

---

## 5. 为什么这个方法 Work: Intuition Building

### 5.1 Locality of Diffusion Denoising

Diffusion model 的 velocity field $\theta(w, t)$ 主要依赖 local neighborhood。虽然 attention 理论上可以 capture long-range dependencies，但实际训练后，denoising step 的 effect 主要是 local。这就是为什么 independent tile denoising 可行。

### 5.2 Overlap Region as Agreement Constraint

在 overlap region，多个 tiles 同时 denoise。如果它们 disagree，cosine weighting 会 average out differences。这相当于 implicit agreement constraint，鼓励 tiles 在 overlap region produce consistent predictions。

### 5.3 Latent Space Smoothness

TRELLIS 的 latent space 是 learned 的，比 raw voxel space 更 smooth。Blending 在 latent space 做，decoded 出来自然更 coherent。这是为什么 latent diffusion models 比 pixel diffusion models 更适合 tiled generation。

### 5.4 与 MultiDiffusion 的 Connection

这个 idea 在 2D image generation 中已有先例:
- MultiDiffusion: https://arxiv.org/abs/2302.08113
- SyncDiffusion: https://arxiv.org/abs/2306.05490

TRELLISWorld 是 3D 版本，但 3D 有额外挑战:
1. 3D structure 更复杂（Gaussian Splatting, occupancy grid）
2. 需要处理 sparse representation
3. Decoder 也要 tiled

---

## 6. Applications

### 6.1 Area-Specific Prompting (Figure 12)

每个 tile 可以有不同 prompt。Prompt organized as 3D tensor:
```lua
prompt = {
  {"Spring forest... blooming flowers"},
  {"Summer meadow..."},
  {"Autumn woods..."},
  {"Winter ice lake... skating marks"}
}
```
Sample nearest prompt for each tile. 这 enables smooth semantic transitions across scene.

### 6.2 Scene Expansion (Figure 11)

用 RePaint with Gaussian-blurred mask:
- Initialize noise with parts of ground truth
- Blurred mask preserves edges
- Encourages smoother transitions

### 6.3 3D Tiling (Figure 13)

不限于 2D surface，可以 3D tiling:
- Group of fish (2×2×2 chunk)
- Castle in the Sky (1×1×2 chunk)
- 用 area-specific prompting 做 3D blending

---

## 7. Limitations 和为什么 Image-Conditioned Model 失败

### 7.1 Limitations

1. **Dependence on base model**: TRELLIS 的能力直接限制 scene quality
2. **Object-level separation**: 无法 disentangle individual objects post-generation

### 7.2 Image-Conditioned Model 的失败 (Section A.5)

这是论文一个深刻分析。Text-conditioned 和 image-conditioned model 有相同 marginalized distribution:
$$q_{\theta_{img}}(x_0) = \int q_{\theta_{img}}(x_0|c_{img}) p(c_{img}) dc_{img} = \int q_{\theta_{text}}(x_0|c_{text}) p(c_{text}) dc_{text} = q_{\theta_{text}}(x_0)$$

但 conditional distribution 不同:
$$q_{\theta_{text}}(x_0|c_{text} = \text{specific-text}) \neq q_{\theta_{img}}(x_0|c_{img} = \text{specific-img})$$

**Key insight**: $q_{\theta_{text}}(x_0|c_{text})$ 更 diffuse，$q_{\theta_{img}}(x_0|c_{img})$ 更 sharp。Text prompt 有 inherent ambiguity，模型学到 broad distribution。Image condition 更 precise，模型学到 narrow distribution。

在 tiled generation 中，overlap region 的不同 tiles 给出 slightly different conditions。Text-conditioned model 因为 distribution diffuse，对这种 perturbation robust。Image-conditioned model 因为 distribution sharp，slightly different condition 导致 significantly different output，floor level 都对不齐。

这给未来研究一个 insight: **diffuse conditional distribution 更适合 tiled generation**。

---

## 8. 更广的联想和 Future Directions

### 8.1 Multi-Scale Tiling

当前所有 tiles 同 size。可以 multi-scale:
- Coarse tiles (大 stride, 大 S): capture global layout
- Fine tiles (小 stride, 小 S): add local details
类似 wavelet multi-resolution analysis。

### 8.2 Adaptive Tiling

Complex region（如 city center）用 small stride，simple region（如 ocean）用 large stride。需要 scene complexity estimator。

### 8.3 Hierarchical Generation

1. 先 generate low-res global layout (4×3×1 at 16³)
2. 再 super-resolve 到 64³
3. 再 add details with fine tiles

### 8.4 Video Generation 的类比

这个 idea 可以 extend 到 video:
- Video 有 spatial + temporal dimensions
- Tiled in space-time
- Overlap in both dimensions
- 解决 long video generation 的 memory issue

参考相关 video generation:
- Sora-style: https://openai.com/sora
- Video diffusion models: https://arxiv.org/abs/2204.03458

### 8.5 与 LLM Context Window 的类比

LLM 用 sliding window attention 处理 long context。Tiled diffusion 是 3D generation 的 analog:
- Sliding window = overlapping tiles
- Attention blending = cosine weighted blending
- KV cache = tile state cache

### 8.6 Physics-Aware Blending

当前 blending 纯 geometric (cosine weight)。可以 add physics constraints:
- Gravity (objects rest on ground)
- Collision (objects don't overlap)
- Light consistency (shadows align)

参考 GALA3D: https://arxiv.org/abs/2402.07207

### 8.7 与 NeRF/Gaussian Splatting 的结合

TRELLISWorld 输出 Gaussian Splatting。可以:
- Train NeRF on generated Gaussians for refinement
- Use generated scene as NeRF initialization
- Edit NeRF with language prompts

### 8.8 Interactive Scene Editing

结合 area-specific prompting，可以 build interactive system:
- User click region, type prompt
- System re-generate only that region
- Blend with surrounding context

### 8.9 4D World Generation

加入 time dimension:
- 4D tiles (3D space + time)
- Generate dynamic scenes
- Characters moving, weather changing

### 8.10 与 Diffusion Model Theory 的 Connection

Tiled diffusion 本质上 modifies the SDE/ODE 的 drift term。理论分析:
- Convergence guarantee?
- Error bound vs stride s?
- 与 rectified flow 的关系?

参考 flow matching: https://arxiv.org/abs/2210.02747

---

## 9. 相关工作 Landscape

### 9.1 Object Generation Foundation

- NeRF: https://arxiv.org/abs/2003.08934
- SDS (DreamFusion): https://arxiv.org/abs/2209.14988
- ProlificDreamer: https://arxiv.org/abs/2305.16213
- MVDream: https://arxiv.org/abs/2308.16512
- LRM: https://arxiv.org/abs/2311.04400
- TRELLIS: https://arxiv.org/abs/2412.01506
- Hunyuan3D 2.5: https://arxiv.org/abs/2506.16504

### 9.2 2D-Based Scene Generation

- Infinite Nature: https://arxiv.org/abs/2012.09855
- SceneScape: https://arxiv.org/abs/2302.01133
- WonderJourney: https://arxiv.org/abs/2312.03884
- WonderWorld: https://arxiv.org/abs/2406.09394
- LucidDreamer: https://arxiv.org/abs/2311.13384

### 9.3 3D Scene-Native Generation

- InfiniCity: https://arxiv.org/abs/2301.09637
- CityDreamer4D: https://arxiv.org/abs/2406.06526
- BlockFusion: https://arxiv.org/abs/2401.17053
- MIDI: https://arxiv.org/abs/2412.03558
- SemCity: https://arxiv.org/abs/2403.07773
- HunyuanWorld: https://arxiv.org/abs/2507.21809

### 9.4 Object Generator-Based Scene Generation

- GALA3D: https://arxiv.org/abs/2402.07207
- CAST: https://arxiv.org/abs/2502.12894
- LayoutGPT: https://arxiv.org/abs/2305.15393
- SynCity: https://arxiv.org/abs/2503.16420

---

## 10. 个人 Thoughts

这个 paper 的 beauty 在于 simplicity。Core idea 一句话能讲清楚: **把 object generator 当 tile generator，用 cosine blending 拼成 scene**。

但 implementation 有很多 subtle details:
1. Latent space blending (not pixel/voxel space)
2. Tiled decoder (non-trivial, 因为 decoder 不是 probabilistic)
3. Multi-stage diffusion (θ₁ dense, θ₂ sparse)
4. Area-specific prompting 实现
5. RePaint for expansion

Section A.5 关于 image-conditioned model 失败的分析特别 insightful。这揭示了一个 general principle: **diffuse conditional distribution 更适合 compositional generation**。这对未来设计 scene generation model 有指导意义。

未来 research direction 我最看好的:
1. **Multi-scale tiling**: 不同 scale capture 不同 structure
2. **Physics-aware blending**: add physical constraints to blending
3. **4D extension**: temporal tiling for dynamic scenes
4. **Theory analysis**: 严格分析 tiled diffusion 的 convergence 和 error bound

这个 method 是 general-purpose, training-free 3D scene generation 的 simple yet powerful foundation。它 opens up 很多 follow-up 工作。
