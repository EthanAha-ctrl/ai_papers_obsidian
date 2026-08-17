---
source_pdf: GSWT__Gaussian_Splatting_Wang_Tiles.pdf
paper_sha256: 55eebd6a831d524d1cff763dc32e0515d562a6c0c17a30206b22197212d124e5
processed_at: '2026-08-04T23:13:03-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 GSWT

## 一句话版本

你拍一小块草地，reconstruct成3DGS，然后把它切成一堆能像拼图一样无缝拼起来的"Gaussian tile"，拼到哪算到哪，理论上能铺满整个地球表面，还跑60fps。

---

## 这事儿为什么难

3DGS很猛，但你训练完它就fixed了——它只认识你拍过的那块地方。你想让草地"无限延伸"，3DGS本身没办法。传统procedural terrain（Perlin noise那种）倒是能无限生成，但出来的东西一看就是假的，没有photorealism。

所以这paper的question就是：**能不能把一小块captured的photorealistic 3DGS，像texture一样tile到无限大的terrain上？**

texture synthesis领域早在2003年就用Wang Tiles解决了类似的2D问题 [Cohen et al. 2003](https://doi.org/10.1145/882262.882265)。Wang Tile就是方块四边带color，拼接时要求相邻边color matching，这样拼出来的pattern是non-periodic但seamless的。这paper的核心move就是把tile的内容从"2D像素"换成"3D Gaussian field"。

---

## Tile长什么样

一个tile是XY-plane上一个正方形区域，里面装了一坨3D Gaussians。结构上参考 [Zhang and Kim 2008](https://doi.org/10.1016/j.gmod.2007.10.002) 的strict Wang Tile设计：

```
        ◇
      ┌─┼─┐
   ◇  │ C │  ◇
      └─┼─┘
        ◇
```

- 四条边上各放一个**diamond-shaped edge patch**（菱形），这四个edge patch定义了四条边的color
- 中间挖一个**center patch**（方形）盖住四个edge patch交会的中心区域，消除seam

每个方向用2种color，所以一个完整tile set有 $2^4 = 16$ 个tiles（覆盖所有edge color组合）。做6个LOD level，总共96个tiles。

**为什么用diamond shape？** 原始Wang Tiles（方形edge）在四角会有corner artifact——四块tile的角凑在一起，很难保证seamless。Diamond shape让四条edge在中心收敛到一点，corner问题自然消失。

---

## 怎么造tile——Graph Cut是关键

### 核心难点

要从exemplar 3DGS里"切"出patch来组装tile，切的过程必须保证：
1. 切口处颜色连续（视觉seamless）
2. 不要把石头、花这种object切成两半

3DGS是unstructured的point set，Gaussian之间没有像素那种grid connectivity，直接在上面做graph cut没法定义。Paper的解法很elegant：**render成2D image再cut**。

### 具体流程

1. 把edge patches和center patch分别沿Z轴orthographic render成两张图：$I_e$（edge）和 $I_c$（center）
2. 在这两张2D image上做graph cut，找一条"切线"
3. 把cut mask lift回3D，retain mask内的Gaussian

### Connectivity weight公式（Eq.1）

$$W(s,t) = \frac{D_I}{G_I} = \frac{\|I_e(s) - I_c(s)\| + \|I_e(t) - I_c(t)\|}{\|G_{I_e}^d(s)\| + \|G_{I_e}^d(t)\| + \|G_{I_c}^d(s)\| + \|G_{I_c}^d(t)\|}$$

人话翻译：
- $s, t$ 是相邻两个像素
- 分子 $D_I$：两个像素在两张图里的颜色差。颜色差大 → $W$ 大 → graph cut倾向于切这里（因为切在颜色差大的地方，视觉上不显眼）
- 分母 $G_I$：两张图在 $s, t$ 处的gradient magnitude之和。gradient大 → $W$ 小 → graph cut避免切这里（因为切在有gradient的地方会留下可见的seam）

直觉上：**切在"颜色差异大但gradient小"的地方最不显眼**——颜色差异大说明切完两边颜色本来就不同（patch本来就不一样），gradient小说明切线本身是平滑的edge。

### Semantic-aware改进（Eq.2）

纯color graph cut经常把石头切成两半。Paper用SAM 2 [Ravi et al. 2024](https://arxiv.org/abs/2408.00714) 给image做semantic segmentation，把每个segment填成average color，background填黑。然后：

$$W(s,t) = \frac{\gamma D_I + (1-\gamma) D_S}{\gamma G_I + (1-\gamma) G_S}$$

- $D_S, G_S$：对semantic image $S_e, S_c$ 做相同计算
- $\gamma = 0.6$：color权重0.6，semantic权重0.4

**Intuition**：同一SAM segment内的像素在semantic image里颜色一样，所以 $D_S \approx 0$，$W$ 小，graph cut会avoid切过object。Background区域 $D_S$ 大，graph cut会prefer。

### Extended Cut Area

有时候即使有semantic guidance，graph cut还是会切穿object（因为没有更优解）。Paper利用3DGS的特性——不像2D texture每个像素只能来自一个patch，3DGS可以在overlap区域**同时保留两个patch的Gaussian**。

做法：对每个SAM mask，算它和cut area的overlap比例，超过50%就把整个mask扩展进cut area。这样object被完整保留在两个patch的overlap里，切完两边都有完整的object。

---

## Rendering——怎么real-time跑起来

### 多线程架构

- **Main thread**：跟WebGL交互，每帧render
- **Worker thread**：管Wang Tile的tiling、LOD update、sorting（这些活儿重但不那么latency-sensitive）
- 之间用channel通信

### Procedural Tiling on the Fly

XY-plane分成tile-sized grid，维护一个2D active tile map。Camera移动时，map整体shift到camera为中心，掉出去的tile丢弃，进来的新区域spawn tile。

**怎么决定新tile的edge color？** 看已经放好的neighbor：有neighbor的方向必须match neighbor的edge color，没neighbor的方向随机选。因为tile set包含所有color组合，必然能找到valid tile。

Planar surface用97×97的map，sphere用200×80。最远的horizon用mesh proxy（height field textured quad）当最coarsest LOD。

### Pre-sorting——避免runtime sort

3DGS每帧sort Gaussian按depth back-to-front是性能瓶颈。Paper借鉴 [Chen et al. 2012](https://dl.acm.org/doi/10.1145/2366145.2366152) 的Depth-Presorted Triangle Lists思想：

**每个tile预先从9个viewpoint sort好9个index buffer**：
- 4个XY-plane上观察
- 4个45° elevation观察  
- 1个top-down

Runtime根据camera position选最近的pre-sort view的index buffer，直接用，不sort。

### Tile-based Rendering（不merge）

借鉴 [Wei 2004](https://dl.acm.org/doi/10.1145/1015436.1015445) 的tile-based texture mapping：
- 初始化时所有tile的Gaussian + 9个pre-sorted index buffer一次性upload到GPU
- 每个active tile：bind对应index buffer → set position offset → draw call
- 完全没有runtime buffer upload
- 加frustum culling跳过视野外的tile

### Tile-Level Topological Sorting

Tile之间不能纯按screen-space depth sort。Key insight：两个相邻tile $T_1, T_2$ 的rendering order由它们shared 3D boundary决定——如果camera从 $T_1$ 侧朝boundary看，$T_1$ 应该后render（back-to-front）。

实现：算每个tile boundary的normal → 构建partial order graph → topological sort得到complete order。

### Selective Tile Merging（解决boundary artifact）

Pre-sort + 顺序render tile，boundary处会有artifact，特别是boundary aligned view direction时（理想情况应该interleave两tile的Gaussian）。

**触发条件**：对每条tile boundary，算boundary normal和unnormalized camera-to-edge vector的absolute dot product，低于threshold就merge这对tile。Unnormalized vector保证近的boundary优先merge。

**Merge怎么做**：pre-sort时除了sorted order，还存每个Gaussian在9个view下的projected depth。Merge时用selected pre-sort view的depth来interleave两tile的Gaussian。如果两tile用不同pre-sort view（仅当tile离camera太近），就runtime sort整个merged tile。Merged tile cached复用。

---

## LOD System

### Construction

6个LOD level，每个level独立reconstruct：
- Image downsample $2^i$
- Gaussian count cap $N_i = N_0 \cdot 4^{-i}$
- 重复整个tile construction pipeline

### LOD Selection

基于tile center到camera的Euclidean distance $d$。Goal是让不同距离的Gaussian在screen space size大致一致。

$$D_i = D_{\max} \cdot (S_i / S_{n-1})$$

- $S_i$：LOD $i$ 的average Gaussian scale
- $D_{\max} = 384$：最远距离
- $S_{n-1}$：coarsest LOD的scale（归一化基准）

从Table 2看，scale大致每级翻倍（Desert: 0.00456 → 0.012 → 0.0319 → 0.0716 → 0.147 → 0.289，ratio ≈ 2×），所以 $D_i$ 也等比递增。

### Per-Gaussian Opacity Blending

借鉴 [Sander and Mitchell 2005](https://dl.acm.org/doi/10.2312/SGP/SGP05/001-018) 的progressive geomorphing和 [Cesium 2022](https://cesium.com/blog/2022/10/20/smoother-lod-transitions-in-cesium-for-unreal/) 的dithered transitions。

当tile到camera距离 $d$ 接近transition threshold $D_i$ 时，同时render $L_i$ 和 $L_{i+1}$ 的Gaussian，opacity在vertex shader里算：

$$\alpha(d) = \begin{cases} 0.5 - \frac{d - D_i}{2\Delta} & \text{if } D_i - \Delta \leq d < D_i + \Delta \\ 0 & \text{otherwise} \end{cases}$$

- $d$：该Gaussian的camera distance
- $D_i$：transition threshold
- $\Delta = 0.05 \cdot D_i$：transition bandwidth

$L_i$ 的Gaussian乘 $\alpha$，$L_{i+1}$ 的Gaussian乘 $1-\alpha$。

**为什么per-Gaussian而不是per-tile？** Per-tile blending会让整个tile一起淡入淡出，bandwidth很宽，ghosting明显。Per-Gaussian可以精确控制只有距离在 $D_i \pm \Delta$ 范围内的Gaussian参与blend，bandwidth极窄，overdraw minimal，而且 $\alpha = 0$ 的Gaussian可以在shader里直接discard。

---

## Tiling到任意surface

### Open surface

定义parameterization $f: \mathbb{R}^3 \to \mathbb{R}^2$（orthographic projection via height field），2D Wang tiling在parameter domain做，再map回3D。

### Closed surface（sphere）

用quad-based icosahedron mapping [Fu and Leung 2005](https://www.cg.tuwien.ac.at/research/publications/2005/FU_2005/)：把icosahedron细分成quad faces近似sphere，每face内部做Wang tiling。

### Gaussian Transform

给定parameterization $f^{-1}: \mathbb{R}^2 \to \mathbb{R}^3$，对每个Gaussian center $\mathbf{p_0} = (x,y,z)$：

1. Project到parameter domain: $\mathbf{c} = (x,y)$
2. Map到3D surface: $\mathbf{p} = f^{-1}(\mathbf{c})$
3. Finite difference估Jacobian：

$$\mathbf{a} = \mathcal{T}_p(x), \quad \mathbf{b} = \mathcal{T}_p(y)$$

$\mathcal{T}_p$ 是关于点 $\mathbf{p}$ 的Jacobian，$\mathbf{a}, \mathbf{b}$ 是surface在 $\mathbf{p}$ 处的两个tangent vector。

4. 构造transform matrix $M = [\mathbf{a} \ \mathbf{b} \ \mathbf{c}]$，其中 $\mathbf{c} = (\mathbf{a} \times \mathbf{b}) / \|\mathbf{a} \times \mathbf{b}\|$ 是normal

5. Gaussian新center：

$$\mathbf{p_0'} = \mathbf{p} + M \cdot (\mathbf{p_0} - \mathbf{c})$$

6. Covariance transform: $\Sigma' = M \Sigma M^T$

**Intuition**：$M$ 是从parameter domain tangent space到3D world space的local linear map。$\mathbf{p_0} - \mathbf{c}$ 是Gaussian在tile local coordinate里的位置（相对于tile center），乘 $M$ 后就rotate/scale到了surface上正确的位置和方向。

---

## 实验数据解读

### 性能（Table 1）

| Scene | Splat count | Render time | Sort time (% frames triggered) | Update time (% frames triggered) |
|-------|-------------|-------------|--------------------------------|----------------------------------|
| Desert | 7.9M | 7.39±0.77ms | 6.72±1.75ms (91.87%) | 5.93±0.43ms (10.71%) |
| Meadow | 32.2M | 21.29±5.16ms | 22.08±8.64ms (82.98%) | 5.26±0.42ms (27.66%) |
| Flowers | 25.5M | 17.51±4.57ms | 13.52±5.01ms (96.94%) | 5.04±0.43ms (23.58%) |
| Planet | 16.7M | 12.69±2.87ms | 10.80±4.75ms (91.13%) | 5.01±0.60ms (17.59%) |

**怎么看这个表**：
- Render time全在7-21ms之间，全部 >60fps，interactive没问题
- Sort time占比最大，但只在view change时trigger（82-98% frames）。注意sort在worker thread上跑，不block main thread的rendering
- Update time ~5ms，只在camera position变化时trigger（10-28% frames）

### LOD分布（Table 2，以Desert为例）

| LOD | Scale | Gaussians/tile |
|-----|-------|----------------|
| 0 | 0.00456 | 91.7K |
| 1 | 0.012 | 24.7K |
| 2 | 0.0319 | 6.32K |
| 3 | 0.0716 | 1.60K |
| 4 | 0.147 | 398 |
| 5 | 0.289 | 86.5 |

Scale ratio LOD5/LOD0 ≈ 63.4×（接近 $2^6 = 64$），验证scale每级翻倍。Gaussian count ratio每级恰好4×，符合 $N_i = N_0 \cdot 4^{-i}$。

### Full-res vs LOD（Fig.7）

- Full resolution: 96.1ms, 286.7M splats
- LOD system: 16.8ms, 22.3M splats
- **5.7× speedup, 12.8× splat reduction**

---

## 我的intuition总结

1. **Wang Tiles + 3DGS是natural fit**：2D texture synthesis的Wang Tiles数学结构（edge color matching → seamless tiling）在3D Gaussian field上同样成立，因为Gaussian splatting是spatially local operation——每个Gaussian只影响有限screen footprint，所以Gaussian field可以像texture一样切tile。

2. **Graph cut on rendered image是工程智慧**：3DGS的unstructured特性让直接在Gaussian space做graph cut很难定义。Paper绕道render成2D image，用成熟的image graph cut + SAM semantic guidance，再把cut mask lift回3D。这个"投影到2D处理再lift回3D"的paradigm在3DGS领域很有用，可能可以extend到其他task（比如Gaussian segmentation、editing）。

3. **Pre-sort + selective merge是rendering性能关键**：3DGS的sorting bottleneck是众所周知的问题。Paper的解法是把global sort decompose成(1) tile-level topological sort (cheap) + (2) intra-tile pre-sort for 9 views + (3) selective merge for view-aligned boundaries。这个idea可能可以extend到其他tile-based或chunk-based 3DGS rendering系统。

4. **Per-Gaussian opacity blending比per-tile/layer好很多**：因为Gaussian的position是continuous的，per-Gaussian计算让transition band可以精确到只有真正在transition distance附近的Gaussian参与blend，而不是整个tile或整个layer一起淡入淡出。这比传统mesh LOD的geomorphing更精细。

5. **Limitation指向future work**：当前只能2D manifold surface tiling，不能生成true 3D volumetric structure（比如有overhang的terrain）。Wang Cubes [Culik and Kari 1996](https://link.springer.com/chapter/10.1007/3-540-60922-9_11) 是6面color-coding的3D版本，理论上可以解决，但计算复杂度和rendering都会大幅上升。

参考链接：
- [3DGS original paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting)
- [Gauzilla renderer](https://github.com/BladeTransformerLLC/gauzilla)
- [SAM 2](https://arxiv.org/abs/2408.00714)
- [Wang Tiles original](https://doi.org/10.1002/j.1538-7305.1961.tb03975.x)
- [Cohen et al. 2003](https://doi.org/10.1145/882262.882265)
- [Zhang and Kim 2008](https://doi.org/10.1016/j.gmod.2007.10.002)
- [Chen et al. 2012 Depth-Presorted Triangle Lists](https://dl.acm.org/doi/10.1145/2366145.2366152)
- [gsplat library](https://github.com/hbb1/gsplat)
- [FLoD](https://arxiv.org/abs/2408.12894)

---

# GSWT: Gaussian Splatting Wang Tiles 论文详解

## 1. 核心动机与High-level Idea

这篇paper试图解决一个非常具体的问题：**给定一小块captured的3DGS exemplar场景，如何procedurally生成无限的、photorealistic的terrain？**

3DGS本身在reconstruction上很强，但有几个根本限制：
- 它只能reconstruct被input images覆盖的bounded region
- 无法synthesize超出training data footprint的新区域
- 传统procedural terrain generation（如Perlin noise height fields）缺乏photorealism

GSWT的key insight：把3D Gaussian field本身作为Wang Tile的内容。Wang Tiles [Wang 1961](https://doi.org/10.1002/j.1538-7305.1961.tb03975.x)通过color-coded edges的matching constraint实现non-periodic但seamless的tessellation，原本用于2D texture synthesis [Cohen et al. 2003](https://doi.org/10.1145/882262.882265)。GSWT把它lift到3D Gaussian field——每个tile内部存的是一组3D Gaussians，相邻tile的Gaussian field在边界处必须满足continuity constraint。

直觉上，这相当于把一块terrain exemplar"切片"成可以无缝拼接的Gaussian patch集合，然后像拼贴画一样infinite extend。

---

## 2. Tile Definition

采用 [Zhang and Kim 2008](https://doi.org/10.1016/j.gmod.2007.10.002) 的strict Wang Tile设计，每个tile由：
- **4个diamond-shaped edge patches**：沿tile四边放置，定义edge color constraint
- **1个center patch**：从square patch中cut出来，覆盖中心区域以消除edge patches之间的seams

每条edge用2种color（W/E用2种，N/S用2种），所以一个完整tile set有 2×2×2×2 = **16个tiles**。每LOD level一套，6个LOD就是96个tiles。

这个设计的关键 intuition：diamond-shaped edge patches的形状让edge constraint在corner处自然converge到一点，避免了原始Wang Tiles的corner artifact问题。

---

## 3. Tile Construction Pipeline

### 3.1 Scene Reconstruction

Standard流程：
- COLMAP [Schönberger and Frahm 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html) 估camera和point cloud
- GSplat library [Ye et al. 2025](https://jmlr.org/papers/v26/24-0377.html) 做reconstruction
- 用MCMC strategy [Kheradmand et al. 2024](https://papers.nips.cc/paper_files/paper/2024) 更新Gaussian count

Multi-LOD reconstruction：对每个LOD level $L_i$：
- Training image下采样 $2^i$ 倍（anti-aliased bilinear）
- Gaussian count cap: $N_i = N_0 \cdot 4^{-i}$
- 实验中 $N_k = 4^{10-k}$，即LOD0最多 $4^{10} \approx 10^6$ Gaussians per exemplar

### 3.2 Patch Sampling

在XY-plane上随机sample区域，然后收集所有projected position落在该区域的Gaussians。**关键细节：忽略Z-coordinate**——只要XY projection inside sample region，就纳入patch。这样patch相当于一个Gaussian column（沿Z轴的Gaussian集合），便于后续tiling on 2D surface。

### 3.3 Graph Cut on 3DGS

直接在3DGS上做graph cut很困难，因为Gaussian之间没有structured connectivity。Paper的解法：

1. 把edge patches和center patch分别render成orthographic image $I_e$ 和 $I_c$（沿Z轴）
2. 在这些2D images上做graph cut
3. 把cut mask lift回3D，retain对应Gaussian

**Connectivity weight公式（Eq.1）**：

$$W(s,t) = \frac{D_I}{G_I} = \frac{\|I_e(s) - I_c(s)\| + \|I_e(t) - I_c(t)\|}{\|G_{I_e}^d(s)\| + \|G_{I_e}^d(t)\| + \|G_{I_c}^d(s)\| + \|G_{I_c}^d(t)\|}$$

变量含义：
- $s, t$：neighboring pixels
- $I_e(\cdot), I_c(\cdot)$：edge patch image和center patch image在该pixel的color
- $G_{I_e}^d(s)$：image $I_e$ 在direction $d = \overrightarrow{st}$ 上的gradient at pixel $s$
- 分子 $D_I$：两个pixel在两个image中的color dissimilarity，越大越鼓励cut（因为cut通过color差异大的地方视觉不显眼）
- 分母 $G_I$：gradient magnitude sum，越小越鼓励cut（gradient小的地方切下去更连续）

Intuition：graph cut希望沿着"颜色一致但gradient低"的边界切——颜色一致意味着切完视觉seamless，gradient低意味着切线本身不显眼。

### 3.4 Semantic-Aware Graph Cut（核心创新）

纯color graph cut经常把objects切成两半。Paper引入SAM 2 [Ravi et al. 2024](https://arxiv.org/abs/2408.00714) 做semantic guidance：

把SAM的segmentation mask转成semantic image $S_e, S_c$（每个region用average color，background黑色；丢弃面积>50%的mask防止SAM segment整个background）。

**Modified connectivity weight（Eq.2）**：

$$W(s,t) = \frac{\gamma D_I + (1-\gamma) D_S}{\gamma G_I + (1-\gamma) G_S}$$

其中：
- $D_S, G_S$：和 $D_I, G_I$ 相同计算，只是image换成 $S_e, S_c$
- $\gamma = 0.6$：color vs semantic的balancing parameter

Semantic term的intuition：同一SAM segment内的pixel在 $S_e, S_c$ 中color一致（因为填的average color），所以cut经过object内部时 $D_S$ 大、$W$ 大，graph cut会avoid。Background区域 $D_S$ 小，graph cut会prefer。

### 3.5 Extended Cut Area

即使有semantic guidance，有时仍会切到object（因为没有更优解）。GSWT利用3DGS的结构特性——和texture graph cut不同（每个pixel只能来自一个patch），3DGS可以**同时保留两个patch的Gaussian**在overlap区域。

算法：对每个SAM mask，计算它和cut area的overlap比例，如果 > 0.5，则把整个mask扩展到cut area。这样object被完整保留在两个patch的overlap区域。

### 3.6 Center Patch Selection

随机选center patch不一定能和edge patches良好融合。Paper采样多个候选（实验用8个），选graph-cut score（即connectivity weight sum）最低的。Trade-off：
- 候选太多 → tiles趋同，diversity下降
- semantic guidance → bias向大background区域的patch

---

## 4. Real-Time Rendering System

### 4.1 Multi-thread Architecture

- **Main thread**：WebGL backend交互，rendering
- **Worker thread**：Wang Tile管理（tiling, LOD update, sorting）
- 通信通过channels，worker异步prepare数据

### 4.2 Procedural Tiling on the Fly

XY-plane分成tile-sized grid，维护**2D active tile map**：
- 每个grid cell分配一个tile
- 新tile的edge color由已放置的neighbor决定（matching constraint）
- 如果某方向无neighbor，random选color
- 由于tile set包含所有edge color组合，必然存在valid tile

Camera移动时，tile map整体shift到camera为中心，out-of-boundary tiles丢弃，新区域spawn tiles。这保证**constant memory usage** across infinite traversal。

最远horizon区域用**mesh grid proxy**（height field textured quad per tile）作为最coarsest LOD，远距离和Gaussian representation seamless blend。

Map size: 97×97 for planar, 200×80 for sphere。

### 4.3 Tile Pre-Sorting（关键技术）

3DGS的sorting是性能瓶颈（back-to-front for alpha blending）。GSWT借鉴 [Chen et al. 2012](https://dl.acm.org/doi/10.1145/2366145.2366152) 的Depth-Presorted Triangle Lists思想：

**每个tile pre-compute 9个sorted index buffers**，对应9个pre-sort views：
- 4个XY-plane上观察
- 4个45° elevation观察
- 1个top-down

Runtime时根据camera position选最近的pre-sort view的index buffer，避免runtime sorting。

**Tile-based rendering without merging**（借鉴 [Wei 2004](https://dl.acm.org/doi/10.1145/1015436.1015445) 的tile-based texture mapping）：
- 所有Gaussian和pre-sorted index buffers初始化时一次性upload
- 每个active tile：bind对应index buffer + set position offset + draw call
- Tile-level frustum culling
- 完全avoid dynamic buffer upload

### 4.4 Tile-Level Topological Sorting

Tiles之间不能纯按screen-space depth sort。Key observation：两个相邻tile $T_1, T_2$ 的partial order由它们shared 3D tile boundary决定——如果camera从 $T_1$ 侧朝boundary看，则 $T_1$ 应在back-to-front blending中后render（即晚于 $T_2$）。

实现：
1. 计算每个tile boundary的surface normal
2. 构建partial order graph
3. Topological sort得到complete order

### 4.5 Selective Tile Merging（解决boundary artifact）

Pre-sort和sequential tile rendering导致boundary处artifact，特别是boundary aligned view direction时（理想sort应该interleave两tile的Gaussian）。

**触发条件**：对每条tile boundary，计算boundary plane normal和unnormalized camera-to-edge vector的absolute dot product，低于threshold就merge。Unnormalized vector保证近boundary优先merge（closer boundary → smaller dot product）。

**Merging implementation**：
- Pre-sort时同时存每个Gaussian在9个pre-sort view下的projected depth
- Merge时用selected pre-sort view的projected depth来merge Gaussians
- 假设merged tiles用相同pre-sort view（多数情况成立）
- 如果不同（仅tile离camera太近时），runtime sort整个merged tile
- Merged tile cached and reused当viewpoint在同 vicinity内

---

## 5. Level of Detail System

### 5.1 LOD Construction

每个level独立reconstruct（参考 [Seo et al. 2024](https://arxiv.org/abs/2408.12894) FLoD）：
- Image downsample $2^i$
- $N_i = N_0 \cdot 4^{-i}$
- 重复整个GSWT construction

### 5.2 LOD Selection

基于Euclidean distance $d$ between tile center and camera。Goal：不同距离的Gaussian在screen space上size大致一致。

**Transition distance formula**：

$$D_i = D_{\max} \cdot (S_i / S_{n-1})$$

变量：
- $S_i$：LOD $i$ 的average Gaussian scale
- $D_{\max}$：maximum distance（实验384）
- $S_{n-1}$：coarsest LOD的scale（归一化基准）

Intuition：scale大一倍的Gaussian可以在远一倍的距离看起来同样大，所以transition distance正比于scale ratio。从Table 2可验证，scale大致每级翻倍（如Desert: 0.00456 → 0.012 → 0.0319 → ...），$D_i$ 也大致等比递增。

### 5.3 Per-Gaussian Opacity Blending（亮点）

借鉴 [Sander and Mitchell 2005](https://dl.acm.org/doi/10.2312/SGP/SGP05/001-018) 的progressive geomorphing和 [Cesium 2022](https://cesium.com/blog/2022/10/20/smoother-lod-transitions-in-cesium-for-unreal/) 的dithered transitions。

当tile bounding box到camera距离 $d$ 接近 $D_i$ 时，同时render $L_i$ 和 $L_{i+1}$ 的Gaussian，opacity按公式3 blending：

$$\alpha(d) = \begin{cases} 0.5 - \frac{d - D_i}{2\Delta} & \text{if } D_i - \Delta \leq d < D_i + \Delta \\ 0 & \text{otherwise} \end{cases}$$

变量：
- $d$：该Gaussian的camera distance
- $D_i$：LOD transition threshold
- $\Delta$：transition bandwidth（5% of $D_i$）

$L_i$ 的Gaussian乘 $\alpha$，$L_{i+1}$ 的Gaussian乘 $1-\alpha$。

**关键优势**：
- Per-Gaussian计算 → transition band可以非常窄
- 早期discard $\alpha = 0$ 的Gaussian，节省overdraw
- 避免full tile或full layer blending的ghosting artifact
- Spawn/despawn tiles也可视为与empty state的transition

---

## 6. Tiling on Arbitrary Surfaces

### 6.1 Open Surfaces

定义parameterization $f: \mathbb{R}^3 \to \mathbb{R}^2$（如orthographic projection via height field texture）。2D Wang tiling在parameter domain做，然后map回3D surface。

### 6.2 Closed Surfaces (genus-0)

用quad-based icosahedron mapping [Fu and Leung 2005](https://www.cg.tuwien.ac.at/research/publications/2005/FU_2005/)：把icosahedron subdivision成quadrilateral faces来approximate sphere，每face内部做Wang tiling，face boundary处特殊处理。

### 6.3 Gaussian Transform

给定surface parameterization $f^{-1}: \mathbb{R}^2 \to \mathbb{R}^3$，对每个Gaussian center $\mathbf{p_0} = (x,y,z)$：

1. Project到parameter domain: $\mathbf{c} = (x,y)$
2. Map到3D surface: $\mathbf{p} = f^{-1}(\mathbf{c})$
3. Finite difference估计Jacobian（Eq.4）：

$$\mathbf{a} = \mathcal{T}_p(x), \quad \mathbf{b} = \mathcal{T}_p(y)$$

其中 $\mathcal{T}_p$ 是关于点 $\mathbf{p}$ 的Jacobian，$\mathbf{a}, \mathbf{b}$ 是surface在 $\mathbf{p}$ 处的tangent vectors。

4. 构造transform matrix $M = [\mathbf{a} \ \mathbf{b} \ \mathbf{c}]$，其中 $\mathbf{c} = (\mathbf{a} \times \mathbf{b}) / \|\mathbf{a} \times \mathbf{b}\|$ 是normal方向

5. Gaussian新center（Eq.5）：

$$\mathbf{p_0'} = \mathbf{p} + M \cdot (\mathbf{p_0} - \mathbf{c})$$

6. Covariance transform: $\Sigma' = M \Sigma M^T$

Intuition：$M$ 是从parameter domain tangent space到3D world space的local linear transform，把tile-local的Gaussian position和shape rotate/scale到surface上对应位置。

---

## 7. Results & Statistics

### 7.1 Platform

- Python 3.12.7 + PyTorch 2.5.1 (tile construction)
- Rust 1.87.0-nightly + WebGL ES 3.0 (renderer, 基于 [Gauzilla](https://github.com/BladeTransformerLLC/gauzilla))
- Windows 11, Intel i9-13900K, RTX 4090D

### 7.2 Datasets

- 5个synthetic scenes (Blender, 4096×4096, 100 views: 36个45° elevation circle + 64个uniform sphere)
- 5个real-world scenes (drone video, 200 views)
- 6 LOD levels ($L_0, ..., L_5$)
- SH degree 1, scale reg weight 0.5
- 30000 iterations reconstruction

### 7.3 Performance (Table 1)

帧时间分析（重点场景）：

| Scene | Splat count | Render time | Sort time | Update time |
|-------|-------------|-------------|-----------|-------------|
| Desert | 7.9M | 7.39±0.77ms | 6.72±1.75ms (91.87% frames) | 5.93±0.43ms (10.71% frames) |
| Meadow | 32.2M | 21.29±5.16ms | 22.08±8.64ms (82.98%) | 5.26±0.42ms (27.66%) |
| Flowers | 25.5M | 17.51±4.57ms | 13.52±5.01ms (96.94%) | 5.04±0.43ms (23.58%) |
| Planet | 16.7M | 12.69±2.87ms | 10.80±4.75ms (91.13%) | 5.01±0.60ms (17.59%) |

关键观察：
- 帧时间7.39-21.29ms，全部interactive (>60fps)
- Sort time占比最高，但只在view change时trigger（82-98% frames）
- Update time ~5ms且仅在camera position变化时trigger

### 7.4 LOD Statistics (Table 2)

以Desert为例：
- LOD0: 91.7K Gaussians/tile, scale 0.00456
- LOD5: 86.5 Gaussians/tile, scale 0.289
- Scale ratio LOD5/LOD0 ≈ 63.4×（接近 $2^6 = 64$，验证scale每级翻倍）

Total memory overhead: hierarchical reduction保证完整LOD结构不超过base representation 33%额外内存。

### 7.5 Full-res vs LOD (Fig.7)

- Full resolution: 96.1ms, 286.7M splats
- LOD system: 16.8ms, 22.3M splats → **5.7× speedup, 12.8× splat reduction**

---

## 8. Limitations & Future Work

1. **Surface-bound paradigm**：只能tiling在2D manifold surface上，不能生成true 3D volumetric structure。可能解法：Wang Cubes [Culik and Kari 1996](https://link.springer.com/chapter/10.1007/3-540-60922-9_11)——3D版本Wang Tiles，每个unit cell有6个color-coded faces。

2. **Single-class terrain**：当前只支持单类terrain synthesis。Multi-class可能通过hybrid tiling（water body / vegetation cluster用不同tile set + 空间分布约束）。

3. **Tile boundary artifact**：selective tile merging减少但未完全消除extreme viewing angle下的artifact。可能改进：temporal coherence / neural reprojection。

---

## 9. Intuition Building Summary

GSWT的关键贡献是把2D texture synthesis领域的Wang Tiles机制lift到3D Gaussian field。核心intuition可以总结为：

1. **Gaussian field作为tile内容**：因为3DGS的rendering是spatially local的（每个Gaussian只影响有限的screen footprint），所以可以把Gaussian field切分成tiles，每个tile独立rendering。这与2D texture的tile-based synthesis在数学结构上同构。

2. **Edge color constraint等价于boundary Gaussian sharing**：传统Wang Tile的edge color matching本质上是boundary function的一致性约束。在Gaussian field里实现这个约束，需要让相邻tile的edge patches是同一组Gaussian（共享edge patch）。Paper的tile set设计——4个edge patches先放，再cut center patch——正是基于这个insight。

3. **Semantic-aware graph cut解决3DGS unstructured connectivity**：直接在Gaussian space定义graph cut的connectivity很难。Paper通过render成2D image + SAM semantic segmentation，把3D问题投影到2D像素级的graph cut，是elegant的工程解法。

4. **Pre-sort + Selective merge解决sorting bottleneck**：把global per-frame sorting decompose为(1) tile-level topological sort (cheap) + (2) pre-computed intra-tile sort for common views + (3) selective merge for view-aligned boundaries。这是把 [Chen et al. 2012](https://dl.acm.org/doi/10.1145/2366145.2366152) 的static scene idea extend到dynamic tiling场景。

5. **Per-Gaussian opacity blending解决LOD popping**：把mesh LOD的geomorphing思想应用到Gaussian splatting的opacity上，narrow bandwidth使overdraw minimal。

参考链接：
- Project page (假设): https://gswt.github.io
- 3DGS原paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting
- Gauzilla (renderer基础): https://github.com/BladeTransformerLLC/gauzilla
- SAM 2: https://arxiv.org/abs/2408.00714
- gsplat library: https://github.com/hbb1/gsplat
- Wang Tiles原始: https://doi.org/10.1002/j.1538-7305.1961.tb03975.x
- Cohen et al. 2003: https://doi.org/10.1145/882262.882265
- Zhang and Kim 2008: https://doi.org/10.1016/j.gmod.2007.10.002
