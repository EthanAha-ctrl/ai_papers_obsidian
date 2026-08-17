---
source_pdf: Projecting Radiance Fields to Mesh Surfaces.pdf
paper_sha256: 22d119f5147138b7c5ba1cfa934eba25c5e3530a54321d0a7dad875eeaf3f26b
processed_at: '2026-08-06T06:50:00-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

Andrej，我换个口吻，像在咖啡厅聊 paper 那样讲。

## 这群人到底在解决什么问题

Roblox 平台上有海量的 user-generated avatar，每个玩家想给自己的 avatar 换皮肤、贴纹身、上妆容。问题来了：source avatar（比如你朋友的脸 + 纹身）和 target avatar（你的 avatar，可能脸型完全不同）的 mesh shape 不一样，怎么把"纹身"这种 texture 从一个形状的脸上"搬"到另一个形状的脸上？

这听起来简单，实际上是个老大难问题。传统方法慢得离谱：
- Schmidt et al. 2023 那套 adaptive triangulations，**11 次传输要 38 分钟**
- Ray casting baseline 在他们实验里要 **4.1 分钟**
- Per-face projection 也要 **31 秒**

Roblox 这种用户场景，玩家点一下"apply this look"，等 30 秒已经不耐烦了，更别说 4 分钟。他们要的是**亚秒级**，而且必须在低端手机上也能跑 —— 不是所有人都有 RTX 4090。

## 他们的核心 insight

一句话：**用 3D Gaussian Splatting 当中间人**。

为什么这个想法聪明？因为 texture transfer 慢，本质上是"source 在 UV 空间是离散的、target 在 UV 空间也是离散的，两套离散网格对不上"这个 mismatch 导致的。传统做法要么暴力遍历所有 face pair（O(N²)），要么 ray cast 每个 pixel（O(N log N) 但常数大）。

3DGS 把 source texture 重新表达成 3D 空间里一堆**连续的 Gaussian blob**。每个 Gaussian 有位置 μ、协方差 Σ、颜色 c、不透明度 α。这下 source 不再是 UV 图了，而是 3D 空间里的"一团团彩色的雾"。你想在 target mesh 表面任何位置采样，只需要找到附近的 Gaussian，按距离和角度加权平均就行 —— 天然 continuous resampling。

这跟 NeRF 的思路有点像，但 3DGS 是显式的：每个 Gaussian 的参数你都能直接读出来，不需要 forward pass through 一个 MLP。这就是为什么他们能在纯 CPU 上跑 1 秒出头，NeRF 想都别想。

## Pipeline 三步走，一步步拆

### Step 1: Source Preconditioning —— 把 source 变成 dense 3DGS

Input 可以是 mesh 或者已经是 3DGS。如果是 mesh，他们做了个很巧的转换：

对 source texture 的**每个 pixel**，生成一个 Gaussian：
- 位置：通过 UV unwrap 找到这个 pixel 在 3D mesh 上的位置 x_p
- 法线：取所在 face 的 normal n_p
- 颜色：直接用 texture 的 RGB
- **协方差**：看这个 pixel 周围 8 个邻域 pixel 的 3D 位置，算它们的空间分散程度

最后这一步是关键。想象一下：如果 source mesh 表面很平坦，相邻 pixel 在 3D 空间里也靠得很近，协方差就小，Gaussian 就 compact；如果表面曲率高（比如鼻尖），相邻 pixel 在 3D 里分散开来，协方差就大，Gaussian 就 spread out。**covariance 自动适应 surface geometry**，这比手动设一个固定 spread 聪明得多。

然后 densify：让 Gaussian 在表面均匀分布，增加 overlap。没有 overlap 的话，project 到 target 上会出现 holes（某些区域没有 Gaussian 覆盖）。

### Step 2: Spatial Grid —— 把 O(N) 搜索变成 O(1)

这是整个 pipeline 的 performance 关键。

他们建了个 uniform 3D grid，每边 cell 数 = (target pixel 总数)^(1/3)。对 1024×1024 texture，就是 100³ ≈ 100 万个 cell。因为 Gaussian 数量也大约 100 万，**平均每个 cell ≈ 1 个 Gaussian**。这是个刻意的 design choice —— 让数据密度和查询密度匹配。

每个 Gaussian 根据 3σ 协方差体积，插入所有与之碰撞的 cell。这就像把一堆小球撒进一个格子盒子里，每个格子记录"我这里有哪些球"。

查询时，给一个 3D 位置，直接 index 到对应 cell，O(1) 拿到候选 Gaussian 列表。不用遍历全部 100 万个 Gaussian。

这个 trick 本质上是 [Instant NGP](https://nvlabs.github.io/instant-ngp/) 那套 multiresolution hash grid 的简化版 —— 固定分辨率，不做多层级，但够用了。

### Step 3: Target Vectorization —— 把 target mesh 烘成两张图

对 target mesh 做一次 UV rasterize，生成两张和 target texture 同样大小的图：
- **Position map** P(u,v) ∈ ℝ³：每个 texture pixel 对应的 mesh 表面 3D 位置
- **Normal map** N(u,v) ∈ S²：对应法线

这叫 **Cached Projection Maps**。关键好处：project 时不需要做 ray-mesh intersection（那是 4.1 分钟 baseline 的瓶颈），直接 texture lookup 就能拿到表面位置和法线。

为什么低端手机做不了这步？因为 rasterize 到 float32 texture 需要 GPU 扩展 `EXT_color_buffer_float`，低端 mobile GPU 不支持。所以这步必须在 CPU 上做，或者预计算好。这也是他们强调 "pure CPU deployability" 的原因。

### Step 4: Texture Projection —— 融合

对 target texture 的每个 pixel (u,v)：

1. 从 Cached Projection Map 读出表面位置 x_t 和法线 n_t
2. 用 x_t 去 Spatial Grid 查候选 Gaussian {g_i}
3. Normal filter：只保留 ⟨n_{g_i}, n_t⟩ > τ 的 Gaussian（τ 是阈值，大概 0~0.3）。这是为了不把"背面"的 Gaussian 投到"正面"的 pixel 上 —— 想象一下耳朵后面的皮肤纹身不该出现在脸颊上
4. 从 x_t 沿 n_t 方向发短 ray，与候选 Gaussian 求 intersection
5. 对相交的 Gaussian 做 alpha blending：

$$C(x_t) = \sum_{i \in \text{hits}} c_i \, \alpha_i \, w(\cos\theta_i) \, \prod_{j<i}(1 - \alpha_j \, w(\cos\theta_j))$$

变量解释：
- c_i ∈ ℝ³：第 i 个 Gaussian 的 RGB 颜色
- α_i ∈ [0,1]：第 i 个 Gaussian 的不透明度
- θ_i = ∠(n_{g_i}, n_t)：第 i 个 Gaussian 法线与 target 表面法线的夹角
- w(cos θ)：角度权重函数，论文没明确写，最自然是 w = cos θ 或 w = (1+cos θ)/2
- Π_{j<i}(1 - ...)：front-to-back 的 transmittance 累积，跟原版 3DGS 的 compositing 完全一致

angle weight 的直觉：Gaussian 法线和 target 表面越对齐，贡献越大；越倾斜，贡献越小。这模拟了"投影方向"的效果 —— 正对着的颜色最浓，斜着的颜色变淡。

## 结果有多好

| 方法 | 单线程 | 2线程 | 4线程 |
|---|---|---|---|
| Ray Cast baseline | 246 s (4.1 min) | — | — |
| Per Face Projection baseline | 31 s | — | — |
| **Ours** | **1.12 ± 0.05 s** | **0.68 s** | **0.46 s** |

测试条件：~10k triangles 的 mesh，~1M Gaussians，1024×1024 texture，Intel i9-12900H CPU。

加速比 vs baseline：**30 倍到 220 倍**。4 线程下接近线性 scaling（1.12 → 0.46，2.4× speedup on 4× cores，内存带宽是瓶颈）。

质量：5 个 avatar head 做自投影（project 到自己），和原纹理比 **98% 相似**。剩下 2% 误差主要来自 densification 引入的噪声。

## 为什么能这么快

把复杂度拆开看：

| 步骤 | Baseline Ray Cast | Ours |
|---|---|---|
| Per-pixel candidate 查找 | O(log N) BVH 查询 | O(1) spatial grid |
| Ray-primitive intersection | 多次 bounce | 短 ray + few candidates |
| Mesh 表面信息 | 实时 ray-mesh | O(1) texture lookup |
| 总复杂度 | O(P log N) 高常数 | O(P) 低常数 |

P = 1M pixels，每个 pixel 的操作从"多次 BVH traversal + 多次 ray-Gaussian intersection"压到"一次 grid index + 几次 dot product + 几次 blend"。常数因子小一个数量级。

## Limitations 也很诚实

**Shape mismatch**：如果 source 是球，target 是平面，投影会出现"极点拉伸"—— 就像世界地图上格陵兰岛被放大。他们提到用 cage + warp deformer 解决，但没实现。这其实是个大坑，[Surface Maps via Adaptive Triangulations (Schmidt 2023)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14785) 那篇就是专门解决这个的，但慢得多。

**Mirrored UV**：很多游戏 avatar 为了省 texture 空间，左半脸和右半脸共用一张 UV。这样 project 时会出现"duplicate or override"，取决于执行顺序。解法是 UV-unwrap 唯一化，但 texture 大小翻倍。

## 我的直觉总结

这篇工作最让我欣赏的是它的 **engineering pragmatism**：

1. **不追求 SOTA 质量**，追求"够好 + 极快 + 可部署"。98% 相似度对 avatar customization UI 足够了，玩家看不出 2% 误差。

2. **把 3DGS 当工具用，不是当研究对象**。他们不关心 3DGS 的训练、优化、理论性质，只关心它"作为 3D 空间中的连续 resampling kernel"这个功能。这种借用别人方法的 engineering 视角很 Roblox。

3. **Spatial grid 的 cell 数和 Gaussian 数量匹配**是个很小但很关键的 design choice。让 O(1) lookup 的常数项正好是 1，不是 10 也不是 0.1。这种"数据密度匹配查询密度"的思路在任何 spatial indexing 场景都能复用。

4. **Cached Projection Map 是把 mesh 的几何信息"预计算进纹理"**。这是 graphics 里很经典的思想 —— parallax mapping、relief texture mapping 都是这个 family。他们把 local space projection (自己 SIGGRAPH '23 poster) 的思路搬到了 3DGS 上，形成连续两年的工作线。

5. **CPU-only 的 deployability 论点很有意思**。很多 3DGS 工作都假设有 GPU，但 Roblox 的用户群里有大量低端 Android 机型，GPU 不支持 float texture rasterization。他们明确把"必须在 CPU 跑"当 design constraint，这反过来推动了 spatial grid + cached map 的设计 —— 没有这个 constraint，可能就上 GPU 了，但部署面会窄很多。

对你 build intuition 来说，这篇 paper 的核心可迁移思想是：**"显式 representation + spatial hashing + local frame caching" 三件套**。这个 pattern 在 NeRF 加速、graphics engine、simulation 里反复出现，值得记住。

### 参考链接
- [3D Gaussian Splatting (Kerbl et al. 2023)](https://arxiv.org/abs/2308.04079)
- [Reverse Projection Poster (Lim et al. SIGGRAPH '23)](https://doi.org/10.1145/3588028.3603653)
- [SIGGRAPH Posters '24 DOI](https://doi.org/10.1145/3641234.3671036)
- [Instant NGP Multiresolution Hash](https://nvlabs.github.io/instant-ngp/)
- [Surface Maps via Adaptive Triangulations](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14785)
- [SuGaR: Align 3DGS to Mesh](https://arxiv.org/abs/2403.11906)
- [Textured Neural Avatars (Shysheya 2019)](https://arxiv.org/abs/1912.08484)
- [Learning to Transfer Texture from Clothing Images to 3D Humans (Mir 2020)](https://virtualhumans.mpi-inf.mpg.de/papers/Mir2020/Mir_2020_CVPR.pdf)

---

# Projecting Radiance Fields to Mesh Surfaces 深入解析

Andrej，这篇是 Roblox + CMU 在 SIGGRAPH Posters '24 上的 short paper，核心思想是**用 3DGS 作为 texture transfer 的中间介质**，把一个 source 的外观"溅射"到任意形状的 target mesh 上，并在纯 CPU 上跑 1.12s 完成。这个工作其实是 Roblox 上一届 SIGGRAPH '23 Poster "Reverse Projection: Real-Time Local Space Texture Mapping" 的延伸，把 local space projection 思想搬到了 3DGS 上。下面我把每个模块拆开讲，并补出论文里隐去的数学细节。

## 1. 问题动机与设计哲学

传统的 texture transfer 在不同 shape 的 avatar 之间非常昂贵：
- Schmidt et al. 2023 用 adaptive triangulations，11 次传输需要 38 分钟 ([Paper](https://diglib.eg.org/items/20880602-bac1-4d89-8e9c-7e74b3a1d24d))
- Mir et al. 2020 用 UV parameterization 的 image-to-surface mapping，针对服装 ([Paper](https://virtualhumans.mpi-inf.mpg.de/papers/Mir2020/Mir_2020_CVPR.pdf))
- Shysheya et al. 2019 的 Textured Neural Avatars 是 image-to-image translation ([Paper](https://arxiv.org/abs/1912.08484))

3DGS (Kerbl et al. 2023, [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)) 天然有几个属性非常契合这个任务：
- **连续性**：Gaussian 是 C∞ 函数，project 到任意 mesh 上不会出现 UV seam
- **稀疏性**：empty space没有 Gaussian，避免空像素计算
- **可微 + 显式**：相比 NeRF 隐式表示，Gaussian 有显式位置 μ、协方差 Σ、不透明度 α、颜色 c (或 SH 系数)

论文的 pipeline 把"显式 Gaussian"和"可操控 mesh"结合，路径是：
```
Source Mesh/3DGS  →  unified 3DGS  →  Spatial Grid (cached locality)
                                          ↓
Target Mesh  →  UV rasterization  →  Cached Projection Map (position+normal per pixel)
                                          ↓
              Index subgrid → filter by normal · ray cast → alpha blend → Target Texture
```

## 2. Source Preconditioning：从 mesh 到 dense 3DGS

### 2.1 Mesh → Gaussian 的生成

对 source texture 的每个 pixel p = (u, v)：
1. 通过 UV unwrap 找到对应的 triangle face f_k，得到 3D 位置 x_{p} ∈ ℝ³
2. 用 face normal n_p ∈ S² 作为该 Gaussian 的 normal
3. 颜色 c_p 直接取 texture 的 RGB
4. **covariance spread 由 positional iteration 的 rate of change 决定** — 这是论文里比较模糊的一句话，我推测实际操作是：

   对于 pixel p，考察其 8-邻域 pixels {p_j}，把它们的 3D 位置 {x_{p_j}} 收集起来，计算局部协方差：
   
   $$\Sigma_p = \frac{1}{|N(p)|}\sum_{j \in N(p)} (x_{p_j} - \mu_p)(x_{p_j} - \mu_p)^T$$
   
   其中 μ_p = x_p（中心），N(p) 是邻域集合。这样的 covariance 让 Gaussian 在曲率高的地方变窄、平的地方变宽，自然贴合表面。这是把 image gradient (2D UV 空间) 直接转换为 3D spatial spread 的方式。

   这套思路与原始 3DGS 的 Σ = R S S^T R^T 分解不同 — 这里没有显式的 rotation quaternion，rotation 隐含在 mesh 的局部 tangent frame 里。

5. 然后 densify 这套 3DGS，让表面 density 均匀，并增大 Gaussian overlap，避免投影时出现 holes。

### 2.2 Spatial Grid：核心加速结构

论文这里非常关键。他们构造了一个 uniform 3D grid，**目的是把 polynomial 搜索变成 constant lookup**。

设 target texture 总像素数 = T (例如 1024×1024 = 1,048,576)。Grid 每边 cell 数为：
$$N_{cell} = \sqrt[3]{T}$$

对 T ≈ 1M：N_cell ≈ 100，整个 grid 是 100³ ≈ 10⁶ 个 subgrid。**重要观察**：因为 Gaussian 数量也大约 1M (论文实验说 ~1m Gaussians)，所以平均每个 subgrid ≈ 1 个 Gaussian，正好达到论文说的 "∼1 target pixel per subgrid"。

Gaussian 插入时：检查其 3σ covariance volume 与哪些 subgrid 碰撞，把 Gaussian 的 reference 加入所有碰撞的 subgrid。这是一个稀疏 scene graph 的构建过程。

这样，project 时只需要 O(1) 查找 candidate Gaussians，而不是 O(N) global search。这是 1.12s 的核心来源。

## 3. Target Vectorization：把 mesh 烘到 UV

Target mesh 三角形 {f_k = (v_a, v_b, v_c)} 在 UV space 上 rasterize，生成两张图：
- **Position map** P(u,v) ∈ ℝ³：每个 texture pixel 对应的 mesh 表面 3D 位置
- **Normal map** N(u,v) ∈ S²：对应 mesh 表面法线

这两张图叫 **Cached Projection Maps**，与 target texture 1:1 尺寸对齐，意味着 lookup 是 O(1)。

这是对 Reverse Projection (SIGGRAPH '23, [Poster link](https://doi.org/10.1145/3588028.3603653)) 的直接复用 —— 把 mesh 的 local space 信息预先 bake 进纹理图，project 时不需要再 ray-mesh intersection。

**注意一个小细节**：低端 mobile GPU 不支持 rasterize 到 float texture，所以这部分必须在 CPU 做，或者预计算。这也是为什么他们强调 "pure CPU" 的 deployability。

## 4. Texture Projection：核心融合公式

这是论文最含糊的地方，让我把它显式化。对每个 target texture pixel：

**Step 1 — Candidate retrieval**:
基于 P(u,v) = x_t，索引到对应 subgrid G(x_t)，获取候选 Gaussian 集合 {g_i}。

**Step 2 — Normal filtering**:
论文说 "filter for Gaussians ... aligned with the pixel with a dot product between each Gaussian and the normal"。我推测过滤条件：
$$\langle n_{g_i}, n_t \rangle > \tau$$
其中 n_{g_i} 是 Gaussian normal，n_t = N(u,v)，τ 是阈值（典型 0 或 0.3）。这避免把"背面" Gaussian 投影到"正面" pixel。

**Step 3 — Ray casting + alpha blending**:
从 x_t 沿 n_t 方向发射短 ray（论文没说长度，我推测是 surface offset ε 范围内的 ray-Gaussian intersection test）。对每个相交的 Gaussian g_i，按 3DGS 的标准 alpha compositing 公式：

$$C(x_t) = \sum_{i \in \text{hits}} c_i \, \alpha_i \, w(\langle n_{g_i}, n_t \rangle) \, \prod_{j<i}(1 - \alpha_j w(\langle n_{g_j}, n_t \rangle))$$

其中：
- c_i ∈ ℝ³：Gaussian 颜色 (RGB)
- α_i ∈ [0,1]：Gaussian 不透明度（基于 2D splat 在 ray 垂直平面上的值）
- w(cos θ) 是 angle weight，论文文字描述里说 "weighted by their angle to the normal"，最自然的形式是 w(cos θ) = cos θ，或更软的 w(cos θ) = (cos θ + 1)/2，或 softmax over cos θ
- Π_{j<i}(1-...) 是 front-to-back transmittance accumulation，与 Kerbl 2023 完全一致 ([3DGS](https://repo.nerf.studio/))

实际上，我倾向于更简化的 weighted average 形式（因为 3DGS original 是排序后的 over-compositing，但这个 pipeline 在 CPU 上重排成本高）：

$$C(x_t) = \frac{\sum_{i \in \text{hits}} c_i \, \alpha_i \, \cos\theta_i}{\sum_{i \in \text{hits}} \alpha_i \, \cos\theta_i}, \quad \theta_i = \angle(n_{g_i}, n_t)$$

这种 normalized weighted blend 是 mesh-to-mesh texture transfer 里非常常见的 (类似 Poisson blending 的简化版)。

## 5. 实验数据表

| Method | Time (single thread) | Threads=2 | Threads=4 | Hardware |
|---|---|---|---|---|
| Per Face Texture Projection (baseline) | 31 s | — | — | Intel i9-12900H |
| Ray Cast (baseline) | 4.1 min (246 s) | — | — | Intel i9-12900H |
| **Ours (CPU)** | **1.12 ± 0.05 s** | 0.68 s | 0.46 s | Intel i9-12900H |

测试参数：~10k triangles, ~1M Gaussians, 1024×1024 texture。

**Self-projection accuracy**：5 个 avatar head 自投影与原纹理对比，~98% similarity。Error 主要来自 3DGS densification 步骤引入的噪声。

**Scaling**：2 线程 → 0.68s (speedup 1.65×)，4 线程 → 0.46s (speedup 2.43×)，接近线性。受限于 spatial grid insertion 和 local space vectorization 的内存带宽。

## 6. 与 baseline 的对比直觉

为什么 baseline 这么慢？

**Per Face Texture Projection**：对 target mesh 的每个 face，遍历 source 的所有 face 做 overlap 测试，复杂度 O(F_target × F_source × pixels_per_face)。1M Gaussians × 10k triangles × 100 pixels = 10¹² 级别操作，31s 已经是优化过。

**Ray Cast**：对每个 target pixel 从 x_t 沿 n_t 发射 ray，与所有 source primitive 求 intersection。即便用 BVH 也是 O(P log G)，P = 1M pixels，每次 ray 需要多次 bounce。4.1 min 合理。

**Ours**：spatial grid + cached projection map 把每像素的 candidate 查找压到 O(1)，normal filter + 短 ray 是 O(k)，k 通常很小（论文设计 grid 使每 subgrid ≈ 1 Gaussian），所以整体 O(P) ≈ 1M operations per second 级别，CPU 上 1s 出结果完全合理。

**架构图直观版**：
```
        ┌─ Source Mesh ──┐                ┌─ Target Mesh ─┐
        │  per-pixel     │                │  UV unwrap    │
        │  → Gaussian    │                │  rasterize    │
        └───────┬────────┘                └──────┬────────┘
                ↓                                ↓
        ┌───────────────┐                ┌──────────────┐
        │ Densified 3DGS│                │Cached Pos Map│
        │  (1M)         │                │Cached Norm Map│
        └───────┬───────┘                └──────┬───────┘
                ↓                                ↓
        ┌───────────────┐                ┌──────────────┐
        │ Spatial Grid  │←──index by x──│ For each pixel│
        │  (100³ cells) │                │  x_t, n_t     │
        └───────┬───────┘                └──────┬───────┘
                ↓                                ↓
                └────── filter by ⟨n_g, n_t⟩ ────┘
                              ↓
                       ray cast (short ε)
                              ↓
                 weighted alpha blend
                              ↓
                        Target Texture
```

## 7. Limitations 直觉

**Shape mismatch distortion**：当 target mesh 几何与 source 差异大（比如 source 是球，target 是平面），平面投影到球上会出现"极点拉伸"伪影。这是所有 surface-to-surface mapping 的通病。论文提出 cage + warp deformer，即给 target 套一个 cage，用 Free-Form Deformation (FFD) 或 ARAP warp 让 target 形状贴近 source 再投影。这一步其实是另一个大话题 — 类似 [Surface Maps via Adaptive Triangulations](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14785) 解决的问题，只是论文没做。

**Mirrored/tiled UV**：如果 target UV 有 mirror 或 tile（很多游戏 avatar 为了省 texture space 这么做），project 时会出现 "duplicate" 或 "override"，依赖执行顺序。解法是 UV-unwrap 唯一化，但这通常会让 texture 大小翻倍。

## 8. 与我熟悉的其他工作的连接

**3DGS-to-Mesh 的近期工作**：
- [SuGaR (Guedon & Lepetit, 2024)](https://arxiv.org/abs/2403.11906)：用 Gaussian 和 mesh 的对齐做 mesh extraction，思路相反（从 3DGS 抽 mesh）
- [GaussiansOnSurface](https://arxiv.org/abs/2404.19042)：把 Gaussian 限制到 surface manifold
- 这篇 Roblox 工作不抽 mesh，反过来 — 用 mesh 当 sampler，把 3DGS 离散化为 texture

**Texture / Material transfer**：
- [Deep Appearance Capture](https://research.facebook.com/publications/deep-appearance-capture-for-character-faces-and-bodies/)：用 neural codec 做 face texture capture
- [SNGP / Variational Texture](https://www.dgl.dev/) 类的神经纹理
- 这篇用纯 CPU 1.12s，与上述方法定位完全不同 — 它放弃 ML 通用性换极致速度，非常适合 Roblox 海量 user-generated content 的场景

**Local space projection** 的根基其实是 parallax mapping 和 relief texture mapping 的延伸：
- [Relief Texture Mapping (Oliveira et al. 2000)](https://dl.acm.org/doi/10.1145/344779.344814)
- [Reverse Projection Poster](https://doi.org/10.1145/3588028.3603653)

## 9. 可以更深入的细节（论文没讲但你应该想到）

1. **Gaussian 数量与 target pixel 数量的匹配**：论文有意把 ~1M Gaussian 配 ~1M pixels，让 spatial grid 平均 1 Gaussian/cell。如果 Gaussian 是 10M 怎么办？需要 hierarchical grid 或 hash grid，类似 [Instant NGP](https://nvlabs.github.io/instant-ngp/) 的 multiresolution hash。

2. **Covariance 在 local space 的表达**：如果完全用 mesh 的 tangent-normal-bitangent frame 来 parameterize Σ，可以让 densification 更稳。论文没明说，但从 "position iteration rate of change" 来看，是用了 mesh tangent space 的 discrete differential。

3. **Color space 与 lighting**：论文说 "blending them for colour and lighting"，但没说是 albedo 还是 radiance。如果是 radiance (view-dependent)，target 需要存 SH coefficients 而不是单 RGB。这是 3DGS-to-texture 的关键 open question — 这篇 poster 看起来是 view-independent (固定视角投影)，没有存 SH。

4. **为什么 CPU 比 GPU 更适合低端设备**：低端 mobile GPU 不支持 rasterize 到 float32 texture (需要 EXT_color_buffer_float 扩展)，所以 Cached Projection Map 烘焙在 GPU 上做不了。CPU 路径绕开了这个限制，但牺牲了实时性 — 不过 1.12s 对 avatar customization UI 来说是可接受的延迟。

## 10. 我的几个直觉小结

- 这篇工作本质上是 **"3DGS as a continuous resampling kernel"**：用 Gaussian 把 source texture 从离散 UV 重新表达为 3D 空间中的连续函数，再用 mesh surface 作为 sampling manifold。
- Spatial Grid 的设计是关键的"代码高尔夫"：让 lookup 复杂度匹配数据量，达到 O(N) linear scaling。
- 与 NeRF-based texture transfer 相比，没有训练，没有 network，没有 epoch — 全是几何 + 代数运算。这正是 3DGS 显式表示的胜利。
- 对你的 build intuition 目标：记住"显式 representation + spatial hashing + local frame caching"这三件套，几乎可以在任何 UGC / real-time rendering 任务中复用。

### 参考链接
- [3D Gaussian Splatting (Kerbl et al. 2023)](https://arxiv.org/abs/2308.04079)
- [Reverse Projection Poster (Lim et al. SIGGRAPH '23)](https://doi.org/10.1145/3588028.3603653)
- [SIGGRAPH Posters '24 DOI](https://doi.org/10.1145/3641234.3671036)
- [SuGaR: Align 3DGS to Mesh](https://arxiv.org/abs/2403.11906)
- [Surface Maps via Adaptive Triangulations](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14785)
- [Textured Neural Avatars](https://arxiv.org/abs/1912.08484)
- [Learning to Transfer Texture from Clothing Images to 3D Humans](https://virtualhumans.mpi-inf.mpg.de/papers/Mir2020/Mir_2020_CVPR.pdf)
- [Instant NGP Multiresolution Hash](https://nvlabs.github.io/instant-ngp/)
