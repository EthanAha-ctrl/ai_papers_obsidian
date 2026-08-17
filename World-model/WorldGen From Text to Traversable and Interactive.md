---
source_pdf: WorldGen From Text to Traversable and Interactive.pdf
paper_sha256: 58d81822fdb51f6ebd445dea58dd3e49c058c9eaf8c7e851c47f931dc5f40aa8
processed_at: '2026-08-13T05:42:41-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# WorldGen 大白话版

好，Andrej，我换个方式讲。假设我们坐在咖啡馆里，你问我这篇 paper 在干嘛，我跟你聊天的语气说一下。

---

## 它到底在解决什么问题？

想象你跟一个 AI 说："给我生成一个 medieval village。"

现在的 AI 能生成一张**图片**，看起来很漂亮。但问题来了——

你能在里面**走**吗？能撞到墙吗？地上有坑会掉下去吗？桌子周围有椅子吗？路走得通吗？

**答案是：不知道。** 因为图片只有 2D 信息，没有 3D 结构，更没有"可走性"这个概念。

WorldGen 想做的事情就是：你说一句话，它吐出来一个**真正能玩的 3D 世界**——有地、有房子、有树、有路，角色能在里面跑、跳、撞东西、不会卡住，而且直接能塞进 Unreal Engine 或者 Unity 用。

---

## 他们怎么想的？

这里有个很聪明的 insight。

游戏行业里，artist 做 3D 场景其实有一套固定流程：

1. **先搭白模**（叫 blockout）——用方块摆出大致结构，哪里是房子、哪里是空地、路怎么走通
2. **再贴图加细节**——把方块变成真正的房子、树、石头

WorldGen 就是把这套流程**自动化**了，而且每个环节用了不同的 AI 技术。

关键在于：他们发现了一个很好的分工方式——

**procedural generation（规则生成）负责"骨架"**，保证场景 functional、路走得通、结构合理。

**diffusion model（扩散模型）负责"血肉"**，保证场景好看、有细节、风格统一、语义合理。

一个管"能不能用"，一个管"好不好看"。

---

## 四个 Stage 用人话讲

### Stage 1：画蓝图

你输入 "medieval village"。

**第一步**，LLM 把你这句话翻译成一堆参数。比如：
- 地形：起伏的丘陵
- 密度：中等
- 垂直变化：低
- 布局风格：有机的、不规则的

**第二步**，一个 procedural generator 拿着这些参数，用规则去摆方块。先摆地标（大建筑），再摆中等物体（树、墙），最后摆小装饰。地面用 Perlin noise 生成起伏。

**关键点**：这些方块**没有语义**。一个方块不知道自己是树还是房子——它就是一个占位符。语义后面由 image generator 决定。

**第三步**，从这些方块里提取 **navmesh**——就是"角色能踩的地面"。用 Recast 这个工具，自动算出哪些表面可走，哪些是室内不算。

**第四步**，把方块场景渲染成 depth map，送进 image generator，生成一张参考图。这张图就有 medieval village 的样子了——石头房子、茅草屋顶、泥巴路。

还有一个 trick：depth map 里加一点 Gaussian noise，免得生成出来的东西全是方块感。

**输出**：blockout（方块布局）+ navmesh（可走地面）+ reference image（参考图）。

---

### Stage 2：生成 3D 场景

现在拿着参考图和 navmesh，要生成真正的 3D mesh。

他们用的 base model 叫 **AssetGen2**，是 Meta 之前做 object-level 3D 生成的工作。核心是用 VecSet 表示——把 3D shape 压成一组 latent vector，然后在 latent space 上做 diffusion。

**核心改进**：加 navmesh conditioning。

为什么需要这个？因为参考图是单视角的，有遮挡——你看不到房子背面，generator 可能乱搞背面。navmesh 是 3D 的完整信息，它告诉 generator："这块地方必须能走通，你不能在这里放障碍物。"

实现上：把 navmesh 也编码成一组 token，通过 cross-attention 注入 diffusion transformer。

**训练 trick**：
- End-to-end fine-tune 整个网络，不能只训新加的 conditioning 层
- Data normalization 把 navmesh 和 mesh 一起缩放到 $[-1,1]^3$ 立方体里，ground plane 对齐到原点

**结果**：生成一个完整的 3D scene mesh，几何上跟 navmesh 对齐，外观上跟参考图一致。然后再用 TRELLIS 的 volumetric texture generator 给它上粗纹理。

为什么用 volumetric 不用 multi-view？因为场景有很多 self-occlusion（房子挡房子），multi-view 方法很多面看不到，volumetric 直接在 3D 里生成 texture，不怕遮挡。

---

### Stage 3：拆成单独物体

现在有一个完整的 scene mesh，但它是**一坨**——所有东西焊在一起了。这不行，game engine 不好处理，也没法编辑。

所以要把这一坨拆成单独的 objects：地面、建筑、树、道具……

他们用之前做的 **AutoPartGen**，但有两个问题：
1. **慢**——autoregressive 一个一个生成，10 分钟
2. **不适应 scene**——原来训练在 object level

**加速 trick**：不再按固定顺序生成 parts，而是按"连接度"排序——先生成跟最多其他 part 接触的 pivot part（比如地面）。地面一旦出来，剩下的就好分了。

更聪明的：训练一个 "remainder mode"，用一个 flag token 让模型一次性吐出所有剩余 geometry，然后后处理用 connected-component analysis 拆开。

**5-step schedule**：4 个 pivot parts + 1 个 remainder → 1 分钟搞定。

**数据问题**：没有现成的 scene-level part annotation dataset。他们自己造——用 VLM 从 asset 库里挖 scene-like 的东西，然后用 heuristic pipeline 处理（检测 connected components、识别地面、merge 小 parts、过滤质量）。

---

### Stage 4：精修每个物体

现在有了低分辨率的单独 objects，但质量不够。要逐个提升。

**三步走**：

**Step 1：Image enhancement**

对每个 object，渲染一个低质量图，加上整个场景的 top-down view（目标 object 用红色标出来），加上 global reference image，送进 LLM-VLM。

LLM-VLM 看了 top-down view 就知道这个 object 在场景里的位置和角色，然后生成一张高质量的该 object 的图。

为什么需要 top-down view？做过 ablation——没有它，LLM-VLM 不知道 object 在场景里的 context，生成出来的东西风格跟场景不搭。

还有 verification loop：算 enhanced image 跟 coarse render 的 IoU，太低就重做。防止 LLM-VLM 乱改视角或 hallucinate 新 geometry。

**Step 2：Mesh enhancement**

拿着 coarse mesh 和 enhanced image，用 Mesh Refinement Model 生成高质量 mesh。

架构上：把 coarse mesh 的 latent 跟 noise latent concatenate 一起送进 diffusion。coarse latent 只作 conditioning，不作输出——它是"锚"，防止 refined mesh 偏离太远。

训练数据怎么造？把高质量 object 拼成 grid（2×2 或 3×3），渲染成 image，送进 AssetGen2 重建——这样模拟 Stage II 产生的 degradation。然后从中提取 degraded object 作为训练输入。

**Step 3：Texture enhancement**

先 delighting——把 enhanced image 里的 baked lighting 去掉。fine-tune 一个 diffusion model 做 delighting。

然后生成 10 张 multi-view images：8 张 side view（每 45° 一张）+ top + bottom。Sequential generation，前面的 view condition 后面的。

Disentangled attention 是这里的关键架构创新——self-attention 拆成三种：in-plane（各 view 内部）、reference（所有 view attend 到 reference image）、multi-view（views 互相 attend）。

最后 backproject 到 UV space，inpainting 补洞。

---

## 跟 Marble (World Labs) 的区别

Marble 是 view-based 的，从单视角往外长场景，用 Gaussian Splats 表示。

**Marble 的问题**：
- 相机移动 3-5 米外画质就崩
- Gaussian splats 在 game engine 里不原生支持
- 不能 per-object 编辑
- 在 mobile 上慢

**WorldGen 的优势**：
- 50×50 米的场景全程一致
- 输出 textured mesh，game engine 直接用
- 每个 object 独立可编辑
- Mobile 友好

**Marble 的优势**：
- Photorealism 更高（radiance field baking 天生擅长）
- 从单视角生成，更灵活

本质上 WorldGen 牺牲了一些 photorealism，换来了 **实用性**——能落地到现有 game engine 工作流。

---

## 我觉得有意思的几个点

**1. PG 和 Diffusion 的分工真的很聪明**

PG 保证 functional（路走得通、结构合理），diffusion 保证 aesthetic（好看、有细节、风格统一）。两者各做擅长的。

而且 PG 只出 anonymous blocks，语义留给 image generator 决定——这让系统既有 controllability 又有 diversity。

**2. Navmesh conditioning 是核心 technical contribution**

它解决了一个 fundamental 问题：单视角 image 不足以约束 3D scene 的 navigability。navmesh 提供 3D structural supervision，image 提供 appearance/semantics。

更 cool 的是它支持 layout editing——改 navmesh，scene 自动 adapt，不用改 image。

**3. Remainder token 是个聪明的工程 trick**

AutoPartGen 加速那段，5-step schedule 的 remainder token 本质上是 hybrid autoregressive + non-autoregressive。scene 里大量 part 是 trivial 的，一旦 pivot 确定剩下就能 connected-component 搞定。训练一个 "remainder mode" 一次 forward 输出剩余 geometry，再后处理拆分。

**4. Modular pipeline 的 trade-off**

好处：每个 stage 可独立改进，可 inject human control，失败可局部重跑。

代价：error 会 propagate，coordination 复杂。

但目前 end-to-end 单 model 解决 scene generation 还不现实，modular 是务实选择。

**5. Limitations 很诚实**

Paper 自己承认：
- 单 reference view 限制 scene scale
- 不能做多 floor 或 interior/exterior transition
- Objects 没 reuse，超 large scene 渲染效率有问题

我觉得还有个潜在问题：PG 部分依赖 LLM 把 prompt 翻译成 JSON 参数，如果用户给的 scene type 是 PG 没见过的（比如 "cyberpunk floating market"），PG 可能搞不定。

---

## 一句话总结

**WorldGen 让 PG 出 functional skeleton，让 diffusion 出 semantic flesh，通过四阶段 pipeline 把 text 变成 game-engine-ready 的 traversable 3D world。**

核心 insight 是承认 image generator 强但不 functional，承认 PG functional 但没细节，然后让两者各做擅长的。最终输出是 textured mesh 而非 Gaussian splats，这是它能落地到现有 game engine 的关键。

---

参考资料：
- WorldGen Blog: https://www.meta.com/blog/worldgen-3d-world-generation-reality-labs-generative-ai-research/
- AssetGen2: https://developers.meta.com/horizon/blog/worlds-AssetGen2/
- AutoPartGen: https://arxiv.org/abs/2411.04459 (NeurIPS 2025)
- VecSet: https://arxiv.org/abs/2309.07966
- TRELLIS: https://arxiv.org/abs/2412.01506
- PartPacker: https://arxiv.org/abs/2506.09980
- Meta 3D TextureGen: https://arxiv.org/abs/2503.23490
- Recast Navigation: https://github.com/recastnavigation/recastnavigation
- Perlin Noise: https://dl.acm.org/doi/10.1145/325165.325247

---

# WorldGen: 从 Text 到 Traversable 3D Worlds

Andrej，这篇来自 Meta Reality Labs 的 WorldGen paper 我读完之后觉得挺有意思的——它把 procedural generation、diffusion-based 3D generation、autoregressive decomposition、compositional enhancement 这几个东西串成了一条 pipeline，本质上是在处理 "如何让生成出来的 3D 世界真的可以玩" 这个问题。

让我从 intuition 的角度展开讲讲。

---

## 1. 核心问题的定位

WorldGen 解决的问题不是 "如何生成一个 3D object"，而是 **如何生成一个 coherent、navigable、interactive 的 3D scene**。

这里有一个关键的 distinction：

- **Object generation**（比如 AssetGen2, TRELLIS, Tripo）已经解决了——单 object 从 text/image 生成 mesh。
- **Scene generation** 的难点在于 objects 之间必须 **thematic / stylistic / structural / functional** 四个维度都 consistent。一个 medieval village 里不该出现 Oxford chair；一个 dining table 周围必须有 chairs；scene 必须 traversable，不能让角色卡住。

WorldGen 的核心 intuition 是：**承认 image generator 很强但不 functional**，**承认 procedural generation 很 functional 但没细节**，然后让 PG 出 layout 和 functional guarantee，让 image generator 出 details and semantics。

这其实是把计算机图形学里 artist workflow 中的 "blockout → detail pass" 这个 pipeline 给 generative 化了。

参考资料：
- Meta Blog: https://www.meta.com/blog/worldgen-3d-world-generation-reality-labs-generative-ai-research/
- AssetGen2: https://developers.meta.com/horizon/blog/worlds-AssetGen2/
- Recast Navigation (navmesh tool): https://github.com/recastnavigation/recastnavigation

---

## 2. 四阶段 Pipeline 总览

整个 pipeline 数学上是这样定义的：

给定 user prompt $y$（比如 "medieval village"），输出 scene：

$$\mathcal{X} = (\{(x_i, g_i)\}_{i=1}^N, S)$$

其中：
- $x_i$ 是第 $i$ 个 object 的 3D shape + UV texture
- $g_i \in SE(3)$ 是 rigid pose（special Euclidean group in 3D，包含 rotation + translation）
- $S$ 是 navmesh（walkable surface）
- $N$ 是 object 数量，由模型自动决定

输出是从条件分布 $p(\mathcal{X} | y)$ 采样得到的。

四个 stage：

| Stage | 输入 | 输出 | 关键模型 |
|---|---|---|---|
| I. Scene Planning | text prompt $y$ | $\mathcal{L} = (B, \mathbf{R}, S)$ | LLM + PG + Depth-conditioned diffusion |
| II. Scene Reconstruction | $\mathcal{L}$ | holistic textured mesh $M$ | AssetGen2 + Navmesh conditioning |
| III. Scene Decomposition | $M$ | $\hat{\mathcal{X}} = \{(\hat{x}_i, g_i)\}$ | AutoPartGen (加速版) |
| IV. Scene Enhancement | $\hat{\mathcal{X}}$ + $\mathbf{R}$ | final $\mathcal{X}$ | LLM-VLM + Mesh Refinement + TextureGen |

整个 pipeline 大约 5 分钟跑完（并行 GPU）。

---

## 3. Stage I: Scene Planning —— 让 PG 听懂人话

### 3.1 LLM 解析 prompt 到 JSON

传统 procedural generation（PG）的痛点是：**可控但只能用参数控制**，无法接受自然语言。

WorldGen 的做法是让 LLM 当翻译官：

$$y \xrightarrow{\text{LLM}} \text{JSON}(\text{terrain type, density, verticality, regularity, ...})$$

这个 JSON 参数驱动一个 modular PG pipeline。

直觉上这是把 "medieval village" 翻译成 `{"terrain": "rolling", "density": "medium", "verticality": "low", "layout": "organic"}` 这种结构化 spec，然后 PG 拿着 spec 去生成 blockout。

### 3.2 PG 内部三步走

**Step 1: Terrain Generation**

用 Perlin noise generator（Ken Perlin 1985 那套）或者 rule-based height map。JSON 参数控制：
- terrain type（flat / steep / rolling）
- surface roughness
- elevation range

**Step 2: Spatial Partitioning**

针对不同环境用不同算法：

| 场景类型 | 算法 |
|---|---|
| structured (urban, grid village) | Binary Space Partitioning (BSP), uniform grids, k-d trees |
| organic (archipelago, jungle) | Voronoi diagrams, noise-based partitions, Drunkard's Walk |

参考资料：
- BSP: Fuchs et al. 1980
- k-d trees: Bentley 1975 (https://dl.acm.org/doi/10.1145/361002.361007)
- Drunkard's Walk: Pearson 1905 (random walk 的鼻祖)

**Step 3: Hierarchical Asset Placement**

三遍放置：
1. **Hero assets**（landmark buildings）—— 先放，确立 structure
2. **Medium-scale**（trees, walls, bridges）—— 相对 hero asset 放置
3. **Small decorative** —— 填补 residual spaces

最后加 terrain smoothing 防 collision。

**关键设计点**：PG 只生成 **anonymous blocks**，不决定语义。一个 box 可能是 tree、rock、building——具体是什么由后面 image generator 决定。这个 abstraction 是 WorldGen 能产生 detail + diversity 的关键。

### 3.3 Navmesh 提取

用 Recast 算法（Mononen et al. 2016-2026）从 blockout $B$ 提取 navmesh $S$。Recast 只识别 exterior traversable surfaces，排除 indoor areas。

### 3.4 Reference Image 生成

Render blockout $B$ 到 isometric depth map，相机 elevation 约 45° 最大化 coverage。

然后关键 trick：对 non-terrain depth 加 Gaussian perturbation，scale 与 depth 成比例：

$$d'_{\text{non-terrain}} = d_{\text{non-terrain}} + \mathcal{N}(0, \sigma \cdot d)$$

直觉：PG 产生的 box 太 rectilinear 了，直接 condition 会让 image 生成的世界全是方块。加 Gaussian noise 让 outline 更自然，但保留整体 structure。

这个 depth map 作为 condition 送给 depth-conditioned diffusion 生成 reference image $\mathbf{R}$。

Figure 4 是个 ablation grid：行是 verticality range，列是 density，能看到 PG 控制力很强——从 sparse 平地到 dense 复杂环境全覆盖。

---

## 4. Stage II: Scene Reconstruction —— AssetGen2 + Navmesh Conditioning

这是 paper 里数学最密集的部分。

### 4.1 VecSet Representation 复习

AssetGen2 用 VecSet (Zhang et al. 2023a, https://arxiv.org/abs/2309.07966) 作为 3D latent representation。

给定 3D object 的 point cloud：

$$\mathcal{P} = \{(p_i, n_i)\}_{i=1}^M, \quad p_i \in \mathbb{R}^3, \quad n_i \in \mathbb{S}^2$$

- $p_i$：point 位置
- $n_i$：normal direction（单位球面 $\mathbb{S}^2$ 上的点）
- $M$：原始 point cloud 大小

Encoder $E$ 先用 **Farthest Point Sampling (FPS)** 降采样到 $K$ 个点：

$$\hat{\mathcal{P}} = \text{FPS}(\mathcal{P} | K) = \{\hat{p}_1, \ldots, \hat{p}_K\}$$

FPS 的 intuition：贪心选最远的点，保证 spatial coverage 均匀。

然后通过 sinusoidal spatial encoding + cross-attention + transformer layers，把 $\mathcal{P}$ 的信息压缩到 $K$ 个 sparse points 上，得到 latent：

$$z = E(\mathcal{P}) \in \mathbb{R}^{K \times D}$$

- $K$：token 数量（比如 2048）
- $D$：每个 token 的 feature 维度（比如 256 或 512）
- 这个 $z$ 是 **permutation-invariant** 的（set representation，不是 sequence）

Decoder $D$ 给定 query point $q \in \mathbb{R}^3$ 输出 SDF value：

$$\text{SDF}(q) = \mathcal{D}(q | z) \in \mathbb{R}$$

最后用 **Marching Cubes** (Lorensen & Cline 1987) 从 SDF 提取 watertight mesh。

### 4.2 Image-to-3D Diffusion

AssetGen2 学一个条件分布：

$$p(z | \mathbf{I}; \Phi)$$

其中 $\mathbf{I}$ 是输入 image，$\Phi$ 是 transformer 参数。Diffusion 在 latent space 上做 denoising，类似 latent diffusion 的思路。

### 4.3 Navmesh Conditioning 的核心改进

WorldGen 要 sample 的是：

$$p(z | \mathbf{R}, S; \Phi)$$

而不是 $p(z | \mathbf{R}; \Phi)$。**为什么需要 navmesh？** 因为 $\mathbf{R}$ 是单视角，有 self-occlusion——navmesh 没在 image 里露出的部分，generator 可能乱搞。

**Navmesh encoder $\mathcal{E}'$**：

1. 从 navmesh surface 随机采样点云 $\mathcal{P} \in \mathbb{R}^{M \times 3}$（无 normals！）
2. FPS 降采样：$\hat{\mathcal{P}} = \text{FPS}(\mathcal{P} | K) \in \mathbb{R}^{K \times 3}$
3. 两套点都用 coordinate positional encoder 映射到 $\mathbb{R}^D$
4. Sparse points 通过 cross-attention attend 到 dense points
5. **特别**：sparse point 的 positional encoding 加回 cross-attention 输出（强化 location 信息）

跟 VecSet encoder 的差异：
- **不用 normals**（navmesh 是 2D manifold in 3D，normal 意义不大）
- **不加额外 transformer layers**（省内存，navmesh 信息相对简单）
- **residual positional encoding**（防 cross-attention 把 location 信息洗掉）

得到 navmesh embedding $\mathcal{E}'(S)$，通过额外 cross-attention layers 注入 diffusion transformer。

### 4.4 Training Strategy

两个关键发现：

**1. End-to-end fine-tuning 比 freezing pre-trained weights + 只训新 cross-attention 层效果好。**

直觉：scene-level alignment 需要整个网络 adapt，单纯加 conditioning 层不够。验证 loss 比较 low。

**2. Data normalization trick**：

AssetGen2 在 $[-1, 1]^3$ cube 里操作。WorldGen 用 scene mesh 的 scale 因子 rescale navmesh，然后 jointly translate 让 navmesh ground plane 中心在 $(0,0,0)$。

公式上：

$$S_{\text{normalized}} = \frac{S - \mathbf{t}}{s}, \quad M_{\text{normalized}} = \frac{M - \mathbf{t}}{s}$$

其中 $s$ 是 scale，$\mathbf{t}$ 是把 navmesh ground plane 移到原点的 translation。

Inference 时（没 GT mesh）用 blockout $B$ 的 scale 来 normalize。

### 4.5 量化结果（Table 1）

| Model | NavMesh CD |
|---|---|
| Top Image-to-3D Model A | 0.038 |
| Baseline (AssetGen2) | 0.042 |
| Baseline* (scene triplets fine-tuned) | 0.038 |
| **Ours (navmesh-conditioned)** | **0.022** |

CD = Chamfer Distance，越低越好。Ours 比 baseline 低 40-50%。

benchmark 细节：50 个 procedural scenes，每个含 moderate verticality terrain + 10-30 个 objects。所有 geometry 归一化到 $[-1,1]^3$，用 **ICP (Iterative Closest Point)** 对齐 GT navmesh 和 generated navmesh，然后算 CD。

### 4.6 TRELLIS Texture

scene mesh 生成后还是 textureless。但 multi-view texture 方法在 scene 这种 packed geometry 上有 self-occlusion 问题——很多面看不到，没法 backproject。

WorldGen 用 **TRELLIS (Xiang et al. 2025b)** 的 volumetric texture generator，因为它直接在 3D 里产生 texture，对 occlusion 鲁棒。Meta 在 in-house dataset 上重训了 TRELLIS（包含 object + scene level 数据）。

这个粗 texture 主要为了给 Stage IV 的 per-object enhancement 提供初始 guidance。

---

## 5. Stage III: Scene Decomposition —— AutoPartGen 加速版

### 5.1 原版 AutoPartGen 的问题

AutoPartGen (Chen et al. 2025a) 是 autoregressive 的：一个一个 part 生成，每个 part conditioned on holistic mesh + 之前生成的 parts。

两个问题：
1. **慢**：autoregressive 10 分钟
2. **不能泛化到 scene**：训练在 object-level data

### 5.2 PartPacker 启发的 Connectivity-Degree Ordering

灵感来自 PartPacker (Tang et al. 2025a)。

原版 AutoPartGen 用固定 lexicographical order (z-x-y) 生成 parts。WorldGen 改成按 **connectivity degree** 排序：每个 part 与多少其他 part 碰撞，按 degree 降序生成。

**Intuition**：先生成 pivot parts（结构锚点），比如 outdoor scene 里的 ground。Ground 几乎和所有东西碰撞，degree 最高。一旦 ground 提取出来，剩下 objects 通过 connected-component analysis 就容易分了。

### 5.3 Remainder Geometry 的特殊 Token

引入 **binary flag token**：激活时模型一次性生成所有剩余 geometry。

Inference 用 5-step schedule：
1. 生成 4 个 pivot parts
2. 生成 1 个 remainder part
3. 对 remainder 做 connected-component analysis 进一步分解

时间从 10 分钟降到 1 分钟。

### 5.4 Scene Decomposition Data Curation

难点：没有现成的 scene-level part annotation dataset。

Meta 自己造数据：

1. **VLM mining**：用 vision-language model 扫 internal 3D asset repository，识别 "scene-like" 资产（多 object、plausible layout、可见 ground）
2. **Heuristic processing pipeline** 四步：
   - (a) Vertex welding 后检测 connected components 作为 minimal parts
   - (b) 检测 ground，把 thin overlays（比如 traffic lines）merge 到 ground 作为独立 part
   - (c) 去重 + 迭代 merge 小 parts 到最近邻，但保持 ground 独立
   - (d) 按 part count / part imbalance / ground confidence 过滤

### 5.5 量化结果（Table 2）

| Model | Chamfer↓ | F@0.01↑ | F@0.02↑ | F@0.03↑ | F@0.05↑ | Time |
|---|---|---|---|---|---|---|
| Top PartGen A | 0.171 | 0.090 | 0.215 | 0.307 | 0.443 | 1 min |
| Top PartGen B | 0.136 | 0.155 | 0.357 | 0.481 | 0.633 | 3 min |
| AutoPartGen | 0.144 | 0.281 | 0.526 | 0.613 | 0.683 | 10 min |
| **Ours** | **0.061** | **0.322** | **0.644** | **0.761** | **0.853** | **1 min** |

Ours 在所有 metric 上都是 SOTA，同时保持最快速度。F-score@0.05 达到 0.853 意味着 85.3% 的 GT part 在 0.05 距离内能找到 prediction match。

---

## 6. Stage IV: Scene Enhancement —— 三步精修

这个 stage 是 WorldGen 视觉质量的关键。

### 6.1 Per-Object Image Enhancement (LLM-VLM)

输入：
- Global reference image $\mathbf{R}$
- 整个 scene $M$ 的 **top-down view**，target object 用 **红色 highlight**
- Per-object coarse render $\hat{\mathbf{I}}_i$（从 $\hat{x}_i$ 渲染）

LLM-VLM 看到 top-down view 就知道 object 在 scene 里的位置、role、周围 context。然后 generate 高质量 image $\mathbf{I}_i$。

**Ablation**（Figure 11）：拿掉 top-down view，只用 global reference + coarse render，LLM-VLM 就生成不出 style-consistent 的 object image。这证明了 top-down view 提供的 spatial + semantic context 是关键。

**Verification loop**：

计算 enhanced image 和 coarse render 的 foreground **IoU**：

$$\text{IoU} = \frac{|M_{\text{enhanced}} \cap M_{\text{coarse}}|}{|M_{\text{enhanced}} \cup M_{\text{coarse}}|}$$

只有 IoU > threshold 才接受。否则反馈给 LLM-VLM 重新生成。

这个 step 防 LLM-VLM 产生 geometric drift（旋转视角、扭曲形状、幻觉新 geometry）。

### 6.2 Per-Object Mesh Enhancement

输入：coarse mesh $\hat{x}_i$ + enhanced image $\mathbf{I}_i$。输出：refined mesh $x_i$。

**架构**（Figure 13）：

1. 用 AssetGen2 的 VAE encode coarse mesh 得到 latent $\hat{z}_i$
2. 加 positional embedding + zero-initialized linear projection（保留 pre-trained prior）
3. 与 diffusion noise latent 沿 **sequence dimension** concatenate
4. 送入 diffusion transformer denoise
5. Denoise 后 $\hat{z}_i$ 被丢弃（只作 conditioning，不作输出）

**关键直觉**：coarse mesh 的 latent 不参与生成输出，只作 "anchor"——防止 refined mesh 偏离太远。

**训练数据 curation**：

需要 triplets $\{\hat{x}_i, x_i, \mathbf{I}_i\}$。

问题：$\hat{x}_i$ 应该模拟 Stage II 的 degradation。怎么模拟？

方法：
1. 把 high-quality objects $x_i$ 按 $2 \times 2$ 或 $3 \times 3$ grid 排成合成 "scene"
2. Grid size 控制 degradation 程度（越大越 degraded）
3. 渲染 scene image → AssetGen2 重建 → 提取 degraded $\hat{x}_i$
4. $\mathbf{I}_i$ 从不同视角渲染 GT object $x_i$

**Augmentation**：加 floaters、masked-out regions、broken surfaces、color jitter、random backgrounds、random blur——提升 robustness。

**Scale restoration**：

Mesh refinement model 在 normalized space 工作，输出后要 rescale 回 scene。关键：**refinement model 保留 coarse mesh 的 orientation**，所以只需要 axis-wise scale + centroid position：

$$x_i^{\text{world}} = s_{\text{axis}} \odot x_i^{\text{normalized}} + \mathbf{c}_i$$

其中 $s_{\text{axis}}$ 是 per-axis scale，$\mathbf{c}_i$ 是 centroid。

### 6.3 Per-Object Texture Enhancement

基于 Meta 3D TextureGen (Bensadoun et al. 2024)。

**Step 1: Delighting**

Enhanced image $\mathbf{I}_i$ 里有 baked lighting / shadows / specular highlights。Fine-tune 一个 text-to-image latent diffusion model 做 delighting——shaded image 的 latent 作为 in-context conditioning。

**Step 2: Multi-View Generation**

生成 10 张 orthographic multi-view images：
- 8 张 side views，每隔 $45^\circ$ 一张，elevation $0^\circ$
- 1 张 top view
- 1 张 bottom view

**Sequential generation strategy**：先 frontal，再 side（conditioned on frontal），最后 top/bottom（conditioned on 之前所有 views）。

**Disentangled Multi-View Attention** 是这里的 architectural 创新：

Self-attention 拆成三块：

| Attention 类型 | 作用 |
|---|---|
| **In-plane self-attention** | 每个 view 独立 attend 自己 spatial features，保留 local detail |
| **Reference attention** | 生成 views (1~N-1) 通过 cross-attention attend 到 reference view (view 0)，保证与 $\mathbf{I}_i$ 一致 |
| **Multi-view attention** | 生成 views 互相 attend，促进 3D global consistency |

这种 disentanglement 让 feature 交互更结构化。

**Step 3: Texture Post-processing**

Back-project 10 views 到 UV space 初始化 texture。然后 UV-space inpainting 填补 gaps 和 unobserved areas。

---

## 7. 跟 Marble (World Labs) 的对比

Paper 里专门讨论了 World Labs 的 Marble 系统（view-based monolithic generation，用 Gaussian Splats）。

**核心差异**：

| 维度 | WorldGen | Marble |
|---|---|---|
| Conditioning | Global reference image + full layout | Single viewpoint |
| Representation | Textured meshes | Millions of Gaussian splats |
| Extent | ~50×50m fully textured | 视觉 fidelity 在 camera 移动 3-5m 后 degrade |
| Engine compatibility | Native (Unreal/Unity) | 需要特殊 rendering pipeline |
| Editability | Compositional, per-object | Monolithic |
| Mobile rendering | 高效 | 比 mesh 慢 orders of magnitude |
| Photorealism | 中等 | 高（radiance field baking） |

Gaussian splats 的优势是 photorealism（容易 bake radiance field），劣势是 **game engine 不原生支持、artist 工具不兼容、mobile 慢**。

WorldGen 的 textured mesh 输出能直接 drop 进任何 game engine，这是 paper 强调的 practical 价值。

---

## 8. 我的 Intuition 和观察

### 8.1 为什么 Pipeline 这么分？

这四个 stage 对应 artist workflow 的：

1. **Blockout** → Stage I
2. **Layout pass** → Stage II（holistic reconstruction）
3. **Object separation** → Stage III
4. **Detail pass** → Stage IV

把 generative AI 拆成这种 modular pipeline 有几个好处：
- 每个 stage 可以独立改进
- 可以 inject human control 在任何 stage
- 失败时可以局部重跑

但代价是：pipeline 整体 coordination 复杂，error 会 propagate。

### 8.2 Navmesh Conditioning 是核心贡献

我觉得 paper 最 techincal 的 contribution 是 navmesh conditioning。它解决了一个 fundamental 问题：**单视角 image 不足以约束 3D scene 的 navigability**。

Navmesh 提供 3D structural supervision，而 image 提供 appearance / semantics。两者通过 cross-attention 融合。

而且这个 conditioning 支持 layout editing（Figure 7）——用户改 navmesh，生成的 scene 自动 adapt，不用改 image。这比改 2D image 容易多了。

### 8.3 Remainder Token 是个聪明 trick

AutoPartGen 加速那段，5-step schedule（4 pivot + 1 remainder）的 remainder token 是个很好的工程 trick。

直觉：autoregressive 慢是因为要生成很多 part。但 scene 里大量 part 是 "trivial"——一旦 pivot 确定了，剩下的通过 connected-component analysis 就能搞。所以训练一个 "remainder mode" 一次 forward 输出所有剩余 geometry，再 post-process 拆分。

这其实是 **hybrid autoregressive + non-autoregressive** 的思路。

### 8.4 Limitations

Paper 自己承认：
- 单 reference view 限制 scene scale（不能做 km 级 open world）
- 不能做多 floor dungeons 或 seamless interior/exterior
- Objects 没 reuse，超 large scene 渲染效率有问题

我觉得还有一个潜在 limitation：**PG 部分 hard-coded 了某些 scene type**。比如 "medieval village" 这个 prompt → JSON 参数映射，依赖 LLM 理解。如果用户给一个 PG 没见过的 scene type（比如 "cyberpunk floating market"），PG 可能搞不定。

---

## 9. 进一步探索的相关工作

如果你想深入，我推荐这几个方向：

**Image-to-3D 基础**：
- VecSet: https://arxiv.org/abs/2309.07966
- AssetGen2: https://developers.meta.com/horizon/blog/worlds-AssetGen2/
- TRELLIS: https://arxiv.org/abs/2412.01506

**3D Reconstruction Feed-Forward**：
- DUSt3R: https://arxiv.org/abs/2312.14132
- VGGT: https://arxiv.org/abs/2503.11651
- π³: https://arxiv.org/abs/2507.13347

**Compositional Scene Generation**：
- SceneWiz3D: https://arxiv.org/abs/2312.08875
- GALA3D: https://arxiv.org/abs/2402.07284
- MIDI: https://arxiv.org/abs/2412.03530

**Procedural + LLM**：
- Infinigen: https://arxiv.org/abs/2306.09310
- SceneCraft: https://arxiv.org/abs/2403.01248

**Part Decomposition**：
- AutoPartGen (NeurIPS 2025): Chen et al.
- PartPacker: https://arxiv.org/abs/2506.09980

**View-based Scene Generation**（Marble 类）：
- Text2Room: https://arxiv.org/abs/2303.11989
- WonderJourney: https://arxiv.org/abs/2312.03284
- LucidDreamer: https://arxiv.org/abs/2410.15299

**Gaussian Splatting 基础**：
- 3DGS: https://arxiv.org/abs/2308.14737

---

## 10. 总结性 Intuition

WorldGen 的 **philosophy** 是：**让 PG 负责 functional skeleton，让 diffusion 负责 semantic flesh**。

这个 division of labor 让 WorldGen 同时拥有：
- PG 的 **guaranteed navigability** 和 **controllable layout**
- Diffusion 的 **rich detail**、**stylistic diversity**、**hallucinated small objects**

四个 stage 对应四种不同的 generative paradigm：
- Stage I: **LLM + PG**（symbolic reasoning + rule-based）
- Stage II: **Latent diffusion + cross-modal conditioning**（continuous + structural）
- Stage III: **Autoregressive + non-autoregressive hybrid**（compositional discovery）
- Stage IV: **Multi-modal refinement + iterative verification**（detail enhancement）

这种 modular design 让每个 stage 都可以用更专门的 model，而不是 end-to-end 训一个 giant model。

而且最终输出是 **game-engine-ready textured meshes**，不是 Gaussian splats 或 NeRF——这是它能落地到现有 game engine 工作流的关键。

我觉得这是当前 3D scene generation 一个很务实的方向。end-to-end 单 model 解决所有问题的 dream 还很远，modular pipeline + 各 stage SOTA model 组合是更可行的路径。
