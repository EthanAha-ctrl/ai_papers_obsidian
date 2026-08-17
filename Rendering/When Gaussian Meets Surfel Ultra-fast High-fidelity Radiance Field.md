---
source_pdf: When Gaussian Meets Surfel Ultra-fast High-fidelity Radiance Field.pdf
paper_sha256: 3faecdcc998a0527814f083cd7c3da7658dd03fb465f7645c210e7f3709fc5e9
processed_at: '2026-08-13T04:18:33-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GES 论文的"人话"版本

刚才那个版本太硬核了,公式堆一堆。这里用大白话再讲一遍,目标是让你 (Karpathy) 在 5 分钟内 build 起完整 intuition。

---

## 这论文在解决什么痛点

3D Gaussian Splatting (3DGS) 大家都知道,把 scene 建模成一堆带 opacity 的 3D Gaussian 椭球,渲染时按深度排序后 alpha-blend。两个老问题:

**痛点 1: sorting 太贵**
对每个 pixel,要沿 ray 把覆盖它的所有 Gaussian 从前到后排好序,因为 alpha-blending 公式 $C = \sum c_i \alpha_i \prod_{j<i}(1-\alpha_j)$ 顺序敏感。GPU 上做精确 per-pixel sort 太慢,3DGS 妥协成"整张图分 tile,每个 tile 内按 Gaussian center depth 全局排一次"。这种"全局排一次"是计算 bottleneck。

**痛点 2: popping artifact**
因为上面那个"全局排一次"是 per-tile 近似,当相机稍微移动,Gaussian 跨 tile 边界时,它在不同 tile 里的相对顺序就变了。于是同一 Gaussian 在相邻 view 里"突然出现在前面"或"突然躲到后面",pixel 颜色就闪一下。你转相机时画面到处闪色块,体验很糟。

已有工作的修补方式各有缺陷:
- StopThePop (https://repo.voria.fr/handle/stop-the-pop) 做 hierarchical sorting,只是缓解,没根治
- SpeedySplat (https://arxiv.org/abs/2412.00578) 直接砍掉 90% Gaussian 加速,但剩下的 Gaussian 平均 opacity 更高,popping 反而更严重
- SortFreeGS (https://arxiv.org/abs/2410.18931) 用 weighted sum 替代 alpha-blending,确实 sort-free 了,但对每个 pixel 都加权所有 Gaussian,被前面物体挡住的 Gaussian 也会贡献颜色 → **color leak**,背景颜色漏到前景

GES 这篇论文想做的就是: **既要 sort-free,又不能 color leak,质量还得保持 SOTA**。

---

## 核心 idea: 让 opaque 和 volumetric 分家

把"表面"和"细节"拆开两层 representation:

**Coarse 层 - 2D opaque surfels**
一组 2D 不透明椭圆 disc,每个有 position, rotation, scaling 和 SH 系数 (view-dependent color)。它们负责"主结构": 大致几何 + 主要颜色。渲染时走标准 GPU hardware z-buffer,谁离相机近谁赢,完全 sort-free (z-buffer 几十年优化,比 GPU sort 快几十倍)。

**Fine 层 - 3D Gaussians**
一组 3D Gaussians (跟原版 3DGS 一样),但**只能贴在 surfel 附近**,补充"surfels 拟合不动的细节": 绒毛、半透明窗帘、视角依赖的高光散射、模糊边缘这类 volumetric 现象。

**关键 insight**: scene 像素颜色 95% 来自 first hit surface radiance,只有 5% 真正需要 volumetric。把 95% 用 opaque surfel 搞定,5% 留给 Gaussian 处理,sorting 问题就消失了。

这其实就是 **surface light field** (Wood et al. 2000, https://dl.acm.org/doi/10.1145/344779.344925) 老思路的现代版。Surface light field 原本需要预先有 high-res mesh + dense image; GES 让 surfel + SH 自动学出来。

---

## 渲染 pipeline (两 pass, 完全 sort-free)

**Pass 1**: surfel 走 hardware rasterizer + z-buffer,输出 color map $C_S$ 和 depth map $D_S$。这一步就是几十年 GPU graphics 标准,谁都会写。

**Pass 2**: 3D Gaussians splat 到屏幕,对每个 pixel 累加颜色和权重:

$$C_G(\hat{\mathbf{x}}) = \sum_i \mathbb{1}(d_i < d_S(\hat{\mathbf{x}}) + \epsilon) \cdot c_i \cdot \alpha_i(\hat{\mathbf{x}})$$

$$W_G(\hat{\mathbf{x}}) = \sum_i \mathbb{1}(d_i < d_S(\hat{\mathbf{x}}) + \epsilon) \cdot \alpha_i(\hat{\mathbf{x}})$$

人话翻译: 遍历覆盖该 pixel 的所有 Gaussian,只要 Gaussian 中心 depth $d_i$ 在 surfel depth $d_S$ 之前 (加个 tolerance $\epsilon$),就把它的颜色 $c_i$ 按 opacity $\alpha_i$ 累加;被 surfel 遮挡的 Gaussian 直接跳过。

最后归一化合成:
$$C = \frac{C_S \cdot W_S + C_G}{W_S + W_G}, \quad W_S = 1$$

**为什么 sort-free**: 加法可交换,$\sum c_i \alpha_i$ 跟 Gaussian 处理顺序无关。原版 3DGS 的 $\prod(1-\alpha_j)$ 是连乘不可交换,这才是 sorting 必需的根源。

**为什么不 color leak**: indicator $\mathbb{1}(d_i < d_S + \epsilon)$ 把被 surfel 遮挡的 Gaussian 硬切掉,sort-free 但保留 occlusion correctness。

**几何 intuition**: 把 surfel 当 weight=1 的 "特殊 Gaussian",加上它前面那一堆 Gaussian,按总 opacity 归一化。本质是 "1-layer surface-bound volume rendering" — 比 NeRF (任意多层 ray marching) 简单, 比 3DGS (沿 ray 全部 blending) 更结构化。

OpenGL 实现完全基于 programmable graphics pipeline,无 compute shader,所以能在移动端跑。

---

## 最巧妙的 trick: opacity modulating annealing

直接优化 opaque surfel 有个根本问题: 不可微。给 surfel 位置加微小扰动,surfel 在屏幕上挪 0.1 pixel,fragment 要么还在原 pixel (gradient = 0),要么挪到新 pixel (颜色突变, gradient 也不存在)。这是 classic differentiable rasterization 难题 (Laine et al. 2020, https://research.nvidia.com/publications/2020/07_Modular-Primitives-High-Performance-Differentiable-Rendering; SoftRas, https://arxiv.org/abs/1904.05190)。

GES 的解法是引入 opacity 调制参数 $w_i \in [0, 255]$:
$$\alpha_i(x, y) = \min(1, w_i \cdot G(x, y)), \quad G(x, y) = \exp(-\frac{x^2+y^2}{2})$$

人话:
- $w_i = 0.1$: 整个 surfel 都半透明,就是个 standard 2D Gaussian (跟 2DGS 一样可微)
- $w_i$ 逐渐增大: 中心变 opaque,外圈还半透明
- $w_i = 255$: 凡是 $G(x,y) > 1/255$ 的地方 opacity 全 1,形成半径 ~3.3 的 opaque disc

**训练 schedule** (这是 annealing 的精髓):
- 0-10K iter: $w_i = 0.1$ 起步,大半透明区,gradient 顺畅回传到 position / rotation / scaling
- 10K iter: 删掉 $w_i < 0.8$ 的 surfel (记下位置给后面 Gaussian init), 剩余 surfel $w_i$ 提到 ≥30
- 15K iter 后: 用 covering score (跨 training view 的最大覆盖 pixel 数) 剪小 surfel
- 18K iter: $w_i \ge 60$
- 19K iter: $w_i \ge 90$
- 20K iter: $w_i = 255$ 全部 freeze

直觉上就是 **simulated annealing**: 早期高熵 (高透明, 大 gradient, 探索), 后期低熵 (opaque, 稳定, exploit)。$w_i$ 是 temperature 参数。

最妙的是: 训练结束时 surfel 完全 opaque,跟 inference 时 z-buffer 渲染完全一致 — **train-test consistency 自然达成**,没有 train-time 用 Gaussian test-time 用 mesh 这种 representation mismatch 问题。

---

## 两阶段 forced 分工

如果让 surfel 和 Gaussian 一开始就 joint optimize, Gaussian 一定"赢": 它 3D + opacity, 表达力强,会把所有 fitting 都揽走, surfel 没机会长出 coarse geometry。结果就是 surfel 不足覆盖 surface, Gaussian 又被 sort-free blending, color leak 就来了。

GES 的解法是强制两阶段:
- **Stage 1 (0-20K iter)**: 只优化 surfel,把 coarse geometry + appearance 占住
- **Stage 2 (20K+ iter)**: freeze surfel 的 position/rotation/scaling, 加 Gaussian 跟 surfel 的 SH 一起 joint optimize, Gaussian 只能补"residual detail"

这思路类似 boosting: 先让简单 model (surfel) 拟合容易部分,再让复杂 model (Gaussian) 补残差。

Ablation 数据 (Table 5):
- Two-stage (3D-GES): PSNR 27.38
- Single-stage: PSNR 27.04

差 0.34 dB, 而且 Fig.13 显示 single-stage 会出现明显 color leak。

---

## 几个小但关键的 engineering 细节

**Adaptive depth tolerance $\epsilon$**
$$\epsilon_i = \frac{5}{D} \sum_j s_{i,j}$$

$D$ 是 scaling 维度 (3D Gaussian $D=3$, 2D $D=2$), $s_{i,j}$ 是第 $j$ 轴 scaling, 系数 5 是经验值。

直觉: 大 Gaussian 中心可能在 surface 后方稍远,但它"伸到"surface 前面的部分仍应贡献 pixel。固定小 $\epsilon$ 切掉这部分; 固定大 $\epsilon$ 让本应被遮挡的 Gaussian leak。Adaptive $\epsilon$ 跟 Gaussian size 匹配,两者避免。

Ablation (Table 5):
- Adaptive $\epsilon$: PSNR 27.38
- $\epsilon = 0$: PSNR 27.11 (颜色不连续)
- $\epsilon = 0.1$ fixed: PSNR 27.19 (color leak)

**Surfel 剪枝策略**
15K iter 后,对每个 surfel 算 covering score $n_i = \max_j \{n_{i,j}\}$ (跨 T 个 training view 找它作为"最前 surfel"覆盖的最大 pixel 数), 剪掉 $n_i < 16$ (real scene) 或 $< 4$ (synthetic) 的 surfel。这步剪掉 80%+ invisible / tiny surfel, 让 surfel 数量从 millions 降到 ~0.15M, 大幅减存储。

**Gaussian 剪枝 contribution score**
$$s_{i,j} = \max_{\hat{\mathbf{x}}} \frac{c_{i,j} \alpha_{i,j}(\hat{\mathbf{x}})}{W_S + W_G}$$

$c_{i,j} = \max_k \{c_{i,j}^k\}$ 取 RGB 通道最大值 (这样 specular 高光也被识别为"高贡献"), 剪掉所有 view 中 $\max s_{i,j} < 0.02$ 的 Gaussian。

---

## 四个 extensions

paper 把基础 GES 还做了 4 个扩展, 基本都是借别人的 trick:

**Mip-GES**: anti-aliasing
- Surfel 用 4× MSAA (硬件标准)
- Gaussian 用 MipSplat (https://arxiv.org/abs/2312.04537) 的 world-space filter + screen-space box filter
- 效果: 远景不 dilation, 细结构不 alias

**Speedy-GES**: 加速
- 把 GES 自己的 contribution pruning 换成 SpeedySplat 的 Hessian pruning (二阶重要性评估)
- 结果: 1080p 下 1135 fps, 4K 下 348 fps, 存储 185 MB

**Compact-GES**: 压缩
- 借 C3DGS (https://arxiv.org/abs/2403.15586): SH 系数换成 hash grid query, scaling/rotation 做 residual vector quantization
- 结果: 47 MB (C3DGS 自己 49 MB), 20× 压缩, 质量 PSNR 26.98 (vs 3D-GES 27.38, 损失 ~0.4 dB)

**2D-GES**: 几何重建
- 把 3D Gaussian 换成 2D Gaussian (Huang et al. 2DGS, https://arxiv.org/abs/2403.11188)
- 2D Gaussian 提供 planar depth + normal, 用来平滑 surfel 边界 discontinuity:
  $$D_{smooth} = \frac{D_S W_S + D_G}{W_S + W_G}, \quad N_{smooth} = \frac{N_S W_S + N_G}{W_S + W_G}$$
- DTU chamfer distance: 2D-GES 0.79, 2DGS 0.80, 3DGS 1.97 — 跟 2DGS 持平,但 glossy surface 上 2DGS 有 hole, 2D-GES 没有 (Fig.11), 因为 opaque surfel 锚定几何, Gaussian 不会"飘走"

---

## 实验数据让你 build intuition

**Speed vs Quality vs Storage (Table 3)**

| Method | FPS 1080p | FPS 4K | Storage | Quality (PSNR) |
|---|---|---|---|---|
| 3DGS | 185 | 62 | 734 MB | 27.43 |
| SpeedySplat | 1140 | 369 | 78 MB | 26.92 |
| SortFreeGS | 321 | 168 | 506 MB | 27.04 |
| AdrGS | 537 | 195 | 274 MB | 27.19 |
| **3D-GES** | **675** | **233** | 366 MB | 27.38 |
| **Speedy-GES** | **1135** | **348** | 185 MB | 27.07 |
| **Compact-GES** | 300 | 128 | 47 MB | 26.98 |

直觉: Speedy-GES 跟 SpeedySplat 速度持平, 但质量高 0.15 dB, Fig.6 显示 SpeedySplat 要 prune 到 30% 才能比得上 Speedy-GES 质量, 那时 FPS 掉到 558, 比 Speedy-GES 1135 一半还少。

**View consistency (Table 2, FLIP metrics, 越小越好)**

| Method | FLIP1↓ | FLIP7↓ |
|---|---|---|
| 3DGS | 0.041 | 0.128 |
| StopThePop | 0.037 | 0.126 |
| SpeedySplat | 0.043 | 0.130 |
| **3D-GES** | **0.032** | **0.117** |
| **2D-GES** | 0.032 | **0.114** |

GES 全面最佳。SortFreeGS 也好但 quality 太差。

**Geometry (DTU Chamfer Distance)**

| Method | Mean CD↓ |
|---|---|
| 3DGS | 1.97 |
| 2DGS | 0.80 |
| **3D-GES** (加 2D-GES regularization) | 0.85 |
| **2D-GES** | 0.79 |

2D-GES 跟 2DGS 持平甚至略胜。

**Primitive count (Table 6, Garden scene)**

| Method | #primitives |
|---|---|
| 3DGS | 5.83M Gaussians |
| 3D-GES | 0.19M surfels + 2.46M Gaussians |
| Speedy-GES | 0.17M surfels + 0.82M Gaussians |

Surfel 只需 ~0.15-0.19M 就能锚住整 scene coarse geometry, 这是 storage 节省的来源。

---

## 为什么 GES work: 5 个直觉

**直觉 1: 表面 light field + volumetric residual**
物理上, 像素颜色 95% 来自 first surface hit radiance, 5% 是 volumetric 现象 (hair, fog, subsurface, soft specular)。GES 把这俩显式分层, 各自用最合适的 representation。

**直觉 2: z-buffer 替代 sorting**
排序需求源自 alpha-blending 不可交换。Opaque z-buffer 天然可交换 (谁近谁赢, 顺序无关)。GES 把 "谁在最前" 这个问题完全交给 GPU hardware z-buffer (几十年优化, 比 GPU sort 快几十倍), 让 Gaussian 只在 "surface 已定" 基础上做 lightweight refinement。

**直觉 3: 归一化是 "surface-aligned 1-layer volume rendering"**
$$C = \frac{C_S \cdot 1 + C_G}{1 + W_G}$$
把 surfel 当 weight=1 的 anchor, 加上 surfel 前面所有 Gaussian 的总权重 $W_G$, 归一化。这相当于一个 "1-layer surface-aligned volumetric blending", 比 NeRF 简单, 比 3DGS 更结构化。

**直觉 4: opacity modulating = anneal to opaqueness**
跟 simulated annealing 精神一致: $w_i$ 是 temperature, 早期高透明大 gradient 探索, 后期 opaque 稳定 exploit。让 non-differentiable opaque rasterization 通过"渐进 freeze"被整合进 differentiable optimization, 是个可能复用到其他 differentiable rendering 问题的 trick。

**直觉 5: 两阶段 forced 分工 = curriculum**
Gaussian 太 flexible 会"包揽"所有 fitting, surfel 没机会 cover surface。强制先 optimize surfel 让它"占位" coarse geometry, Gaussian 只能补 residual。类似 boosting 的思路。

---

## 局限性 (paper 自己承认)

1. **Specular surface**: 3DGS 用大量 low-opacity Gaussian "假装" reflection, opaque surfel 没这能力。可以用 ray tracing (Moenne-Loccoz et al., https://arxiv.org/abs/2404.18457) 或 environment map (Ye et al. Deferred Reflection, https://doi.org/10.1145/3641519.3641528) 改进。

2. **Initialization sensitivity**: 跟 3DGS 一样依赖 SfM points, 随机 init 质量下降。3DGS-MCMC (https://arxiv.org/abs/2404.09608) 的 deterministic state transition 可能能解决。

3. **Training time**: 1.3-1.6× 慢于 3DGS, 主要是 surfel 优化阶段。

4. **未涉及**: dynamic scenes (4DGS, https://arxiv.org/abs/2403.11154)、large-scale urban (Kerbl et al. hierarchical 3DGS, https://repo.voria.fr/handle/hierarchical-3d-gaussians)。

---

## 我的联想和可能的延伸方向

1. **Surfel 加 BRDF**: 现在用 SH 存 view-dependent color, 是把 BRDF × environment 烘焙。如果 surfel 直接带 microfacet BRDF 参数 + 环境 map, 可以解 specular 问题, 也能做 relighting。

2. **Dynamic scene**: surfel 做 SE(3) tracking (类似 BundleFusion, https://graphics.stanford.edu/papers/video/bundlefusion/), Gaussian handle transient geometry。Surfel 数量少适合做 deformation field, Gaussian 重新补细节。

3. **Generative prior**: surfel 比 3D Gaussian 更紧凑 (每个 surfel 只 ~14 float),更适合做 diffusion 生成 latent。可能能做 text-to-3D 的更 controllable 生成。

4. **SLAM**: GES 是天然 mapping representation — surfel 当 landmarks (stable, 可以做 loop closure), Gaussian 做 detail (transient)。可能比 pure Gaussian-SLAM 更鲁棒。

5. **Differentiable rendering 的 train-test consistency**: GES 的 opacity annealing 思路 (train 用半透明, test 用 opaque) 是一种 "training representation 应该 converge to inference representation" 的具体实例。这个思路可以推广到 mesh differentiable rendering: train 用 soft rasterization, 逐渐 anneal 到 hard rasterization, 让训练结束的 mesh 直接用 hardware pipeline 渲染。其实 Laine et al. 2020 的 modular differentiable rendering 已经有类似 idea (anti-aliasing 项逐渐减弱), GES 把它推得更彻底。

6. **LOD for large scene**: surfel 自然支持 merge (相邻 surfel 合并成大 surfel, SH 系数加权平均), 适合做 Level-of-Detail。远处用合并的大 surfel, 近处 unfold 出 Gaussian detail。可能能解决 3DGS 在 large urban scene 上的扩展问题。

7. **Connection to neural deferred rendering**: GES 的两 pass 结构 (pass 1 求深度 + pass 2 累加 detail) 跟 deferred shading 思路类似。可以想象把 Gaussian splatting 换成 neural network, pass 1 输出 G-buffer, pass 2 用 neural shader 出图, 就变成 ADOP (Rückert et al. 2022, https://arxiv.org/abs/2110.06385) 或 Neural Deferred Rendering 那类工作, GES 给了一个 explicit primitive 版本。

---

## 一句话总结

GES 的核心贡献是把 3DGS 谱系里第一次真正把 "opaque surface-bound primitive" 和 "volumetric residual detail" 显式分层, 用经典 graphics 的 z-buffer + normalized additive blending 替代 alpha-blending + sorting, 同时通过 opacity modulating annealing 解决 differentiable optimization 难题。最终拿到 ultra-fast (1000+ fps) + SOTA quality + view-consistent (无 popping) + storage-compact (47MB 可达) + good geometry (DTU SOTA) 五个目标的共同达成。

参考链接:
- Paper: https://doi.org/10.1145/3730925
- 3DGS: https://repo.voria.fr/handle/3dgaussian-splatting
- 2DGS: https://arxiv.org/abs/2403.11188
- MipSplat: https://arxiv.org/abs/2312.04537
- StopThePop: https://repo.voria.fr/handle/stop-the-pop
- SpeedySplat: https://arxiv.org/abs/2412.00578
- SortFreeGS: https://arxiv.org/abs/2410.18931
- C3DGS: https://arxiv.org/abs/2403.15586
- 3DGS-MCMC: https://arxiv.org/abs/2404.09608
- 3D Gaussian Ray Tracing: https://arxiv.org/abs/2404.18457
- Surface Light Field (Wood 2000): https://dl.acm.org/doi/10.1145/344779.344925
- SoftRasterizer: https://arxiv.org/abs/1904.05190
- Modular Differentiable Rendering (Laine 2020): https://research.nvidia.com/publications/2020/07_Modular-Primitives-High-Performance-Differentiable-Rendering
- DTU Dataset: http://roboimagedata.compute.dtu.dk/?page_id=36

---

# When Gaussian Meets Surfel: 深度技术解读

这篇来自浙江大学 State Key Lab of CAD&CG 的 Keyang Ye, Tianjia Shao, Kun Zhou 的 SIGGRAPH 2025 / ACM TOG 论文，本质上是在回答一个很优雅的问题：**3DGS 的 sorting bottleneck 与 popping artifacts 能否从 representation 层面被彻底消除，而不牺牲 quality?**

paper link: https://doi.org/10.1145/3730925
project (推测): http://www.cad.zju.edu.cn/home/gfzju/  (ges 官方 code & viewer 应该挂在 github)

---

## 1. 问题背景: 为什么 3DGS 需要 sorting, 又为什么会 pop

3DGS (Kerbl et al. 2023, https://repo.voria.fr/handle/3dgaussian-splatting) 把 scene 表示成一组 3D Gaussian，每个有 position $\mathbf{p}_i$, covariance (用 scaling $\mathbf{s}_i$ + rotation $\mathbf{r}_i$ 表示), opacity $\sigma_i$, SH coefficients。渲染通过 alpha-blending:

$$C(\hat{\mathbf{x}}) = \sum_{i=1}^{N} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

这里上标 $i$ 是 **沿 ray 从前到后排序后的 index**。这个 $\prod(1-\alpha_j)$ 项使得颜色依赖 Gaussian 之间的相对顺序 — 这是 sorting 不可避免的根本原因。

per-pixel 精确排序在 GPU 上太贵，3DGS 用 tile-based pre-sorting：把整张图划分成 16×16 tile, 在每个 tile 内按 Gaussian center depth 全局排序一次。当 Gaussian 跨越多个 tile 时，不同 tile 看到的相对深度顺序会变化，于是同一 Gaussian 在相邻 view 下"突然出现在前"或"突然消失到后" — 这就是 **popping artifacts** (色块闪现闪隐)。

StopThePop (Radl et al. 2024, https://repo.voria.fr/handle/stop-the-pop) 用 hierarchical sorting 缓解，但仍不保证 per-pixel 正确。Hanson et al. SpeedySplat (https://arxiv.org/abs/2412.00578) prune 掉 ~90% Gaussians，但剩余 Gaussian opacity 整体上升，popping 反而更严重。SortFreeGS (Hou et al. 2024, https://arxiv.org/abs/2410.18931) 用 weighted sum 替代 alpha-blending 消除 sorting，但对每个 pixel 都加权所有沿 ray 的 Gaussian — **occluded object 会 color leak**。

**核心 insight**: 排序问题来自"用一组半透明 volumetric primitive 同时表达 geometry + appearance"。如果能把 **opaque geometry** (用 z-buffer 处理 occlusion, 天生 sorting-free) 与 **fine-grained volumetric appearance detail** (只需在表面附近累积) 分离开，两个问题就都解了。

---

## 2. GES 的 bi-scale representation

GES = Gaussian-enhanced Surfels, 两层:

**Surfel layer (coarse scale)**: 一组 2D opaque 椭圆 disc
$$\mathcal{S} = \{\mathbf{p}_i, \mathbf{r}_i, \mathbf{s}_i, \mathbf{SH}_i\}_{i=1}^{N}$$
- $\mathbf{p}_i \in \mathbb{R}^3$: surfel 中心 position
- $\mathbf{r}_i \in \mathbb{R}^4$: rotation quaternion (局部 XY plane → 世界)
- $\mathbf{s}_i \in \mathbb{R}^2$: anisotropic scaling (disc 椭圆的两个半轴)
- $\mathbf{SH}_i$: 球谐系数，view-dependent color

**Gaussian layer (fine scale)**: 一组 3D Gaussians (与原版 3DGS 相同)
$$\mathcal{G} = \{\mathbf{p}_i, \sigma_i, \mathbf{r}_i, \mathbf{s}_i, \mathbf{SH}_i\}_{i=1}^{M}$$
- $\sigma_i$: max opacity (3DGS 中也有)
- $\mathbf{s}_i \in \mathbb{R}^3$: 三轴 scaling (3D，不是 2D)

**关键约束**: Gaussians 必须紧贴 surfel 附近 — 它们只是 "fine detail supplement"，不是独立 volumetric field。论文 Fig.2 直观显示：surfel 渲染出来已经能捕捉大部分 scene appearance (左图)，Gaussian 只补充细节 (中图)，合起来是高质量 (右图)。

这种 decomposition 的几何 intuition 很有意思：表面 light field (Wood et al. 2000 surface light field, https://dl.acm.org/doi/10.1145/344779.344925) 在传统图形学里早就被研究过 — 给定 surface 上每个点 + 任意方向，存一个 radiance。surfels + SH 系数恰好是 surface light field 的离散化表示，不需要 high-res geometry。Gaussian 部分补充了非 surface-bound 的 volumetric 现象 (hair, fuzz, view-dependent specular bloom, soft edge)。

---

## 3. 两-pass sorting-free rendering pipeline

### Pass 1: Surfel rasterization (标准 graphics pipeline)

用 hardware rasterizer + z-buffer:
1. geometry shader 把每个 surfel point 展开成 screen-space ellipse
2. 每个 fragment 做 depth test against z-buffer
3. 通过 test 的 fragment 写入 color 和 depth

输出:
- $C_S$: surfel color map (opaque, 已经过 z-buffer occlusion)
- $D_S$: surfel depth map (front-most opaque surface depth per pixel)

这一步完全是 GPU 上的 hardware z-buffer，复杂度 $O(\log N)$ per pixel via hierarchical z，根本不涉及任何 sorting。Surfel 只有 single color (来自 SH evaluated in direction from $\mathbf{p}_i$ to camera $\mathbf{o}$):
$$\mathbf{c}_i = Y(\|\mathbf{o} - \mathbf{p}_i\|, \mathbf{SH}_i)$$
其中 $Y(\cdot)$ 是 SH evaluation function (一般用 3-degree SH = 16 coefficients)。

### Pass 2: Gaussian splatting with depth testing (additive blending, no sorting)

3D Gaussians splat 到屏幕，对每个 pixel $\hat{\mathbf{x}}$ 累加颜色与权重，**完全 order-independent**:

$$C_G(\hat{\mathbf{x}}) = \sum_{i=1}^{K} \mathbb{1}(d_i < d_S(\hat{\mathbf{x}}) + \epsilon) \cdot \mathbf{c}_i \cdot \alpha_i(\hat{\mathbf{x}})$$

$$W_G(\hat{\mathbf{x}}) = \sum_{i=1}^{K} \mathbb{1}(d_i < d_S(\hat{\mathbf{x}}) + \epsilon) \cdot \alpha_i(\hat{\mathbf{x}})$$

$$\alpha_i(\hat{\mathbf{x}}) = \sigma_i \cdot \exp\left(-\frac{(\hat{\mathbf{x}} - \hat{\mathbf{p}}_i)^T \Sigma^{-1} (\hat{\mathbf{x}} - \hat{\mathbf{p}}_i)}{2}\right)$$

变量含义:
- $d_i$: Gaussian 中心 depth (z 坐标)
- $d_S(\hat{\mathbf{x}})$: 从 Pass 1 depth map 读出的 pixel depth
- $\epsilon$: depth tolerance (后面讲它的 adaptive 设计)
- $\Sigma$: 投影到屏幕的 2D covariance matrix (由 $\mathbf{r}_i, \mathbf{s}_i$ + view transform 得到)
- $\hat{\mathbf{p}}_i$: Gaussian 中心投影到屏幕的 2D 位置
- $\alpha_i < 1/255$ 时被 clamp 为 0 (与 3DGS 一致，省去稀疏贡献)

**关键设计**: indicator function $\mathbb{1}(d_i < d_S + \epsilon)$ 起到 "Gaussian 被 surfel 遮挡" 的剔除作用。这等价于 "Gaussian 中心在 surface 后方则不贡献"。Surface 后面的 Gaussian 全被丢掉，前面的 Gaussian 累加颜色，最后 **归一化**:

$$C = \frac{C_S \cdot W_S + C_G}{W_S + W_G}, \quad W_S = 1$$

为什么归一化 + 加性累积天然 sorting-free? 因为加法可交换: $\sum \mathbf{c}_i \alpha_i$ 不依赖顺序。而原版 3DGS 的 $\prod(1-\alpha_j)$ 链式乘积本质上不可交换。

**几何 intuition**: 把所有未被遮挡的 Gaussian 想成 "surface 上面一层半透明薄雾"，归一化相当于按 opacity 加权平均它们的 color + surfel 自己的 color。这相当于一个 "surface-bound layer" 的 volume rendering，正确的 occlusion 由 surfel z-buffer 保证，正确的 blending 由 normalization 保证。

### OpenGL 实现

完全用 programmable graphics pipeline，无 compute shader:
- Pass 1: depth test + write enable, blending disable, geometry shader 展开 surfel
- MSAA resolve (4× 多采样)
- depth map 在 post-processing shader 中按 $\epsilon$ 修改
- Pass 2: depth test enable, additive blending enable, depth write disable
- 最后 post-processing shader 计算 Eq.(5)

这套实现能在移动设备上跑 (paper 原话 "easy to be integrated into existing rendering engines and mobile devices")，因为不依赖 CUDA / compute shader。

---

## 4. 优化: opacity modulating trick (最精彩的部分)

### 问题: opaque surfel 不可微

如果你直接对 opaque surfel 做 gradient descent: 给 $\mathbf{p}_i$ 一个微小扰动 $\delta$，surfels 在屏幕上挪了 $\delta$ 像素，对应的 fragment 要么还在原 pixel 上 (颜色不变, gradient = 0)，要么挪到新 pixel 上 (颜色突变 0 → c 或 c → 0, gradient = 0 几乎处处, undefined at 边界)。

这跟 differentiable mesh rasterization 的问题本质相同 (Laine et al. 2020, https://research.nvidia.com/publications/2020/07_Modular-Primitives-High-Performance-Differentiable-Rendering; Liu et al. SoftRasterizer 2019, https://arxiv.org/abs/1904.05190)。他们的解法是 subpixel anti-aliasing (PyTorch3D) 或转成 3D Gaussians (Rhodin et al. 2015, https://openaccess.thecvf.com/content_iccv_2015/html/Rhodin_A_Versatile_Scene_ICCV_2015_paper.html) 或 smooth probability map (SoftRas)。

GES 的解法是 **opacity modulating parameter** $w_i \in [0, 255]$:
$$\alpha_i(x,y) = \min(1, w_i \cdot G(x,y))$$
$$G(x,y) = \exp\left(-\frac{x^2+y^2}{2}\right)$$

变量含义:
- $(x, y)$: surfel local coordinate 上的点 (单位 disc)
- $G(x,y)$: 标准 2D Gaussian (peak = 1 at center)
- $w_i$: 透明度调制参数

行为分析:
- $w_i < 1$: 全 surfel 半透明, 是一个 standard 2D Gaussian (和 2DGS 一样)
- $w_i = 1$: 中心刚好达到 opacity 1
- $1 < w_i < 255$: 中心一段 opaque, 外圈 Gaussian-decayed 半透明
- $w_i = 255$: 凡是 $G(x,y) > 1/255$ 的位置都 opacity 1，外部 $G \le 1/255$ 处 opacity 0，形成半径 $r = \sqrt{2 \log 255} \approx 3.3$ 的 opaque 圆盘 (见 paper Fig.4)

**训练 schedule**:
- 起始 $w_i = 0.1$ (非常透明, 大半透明区, 提供充足 gradient 给 position 和 shape 优化)
- 10K iter: 删除 $w_i < 0.8$ 的 surfels (记录其 position 给后续 Gaussian init), 剩余 surfels $w_i$ 提升到 ≥30, freeze $w_i$, 关闭 densification/pruning
- 15K iter 后: 用 covering score $n_i = \max_j \{n_{i,j}\}$ (跨 T 个 training view 的最大覆盖 pixel 数) 剪枝, $n_{thr} = 16$ (real) 或 4 (synthetic)
- 18K iter: $w_i \ge 60$
- 19K iter: $w_i \ge 90$
- 20K iter: $w_i = 255$ 全部, 关闭 geometry 优化

这种"渐进变 opaque"的 schedule 漂亮之处在于:
1. 早期 gradient 通过半透明外环顺利回传到 position / rotation / scaling
2. 中期 geometry 已经大致稳定，opaque 中心提供清晰的 occlusion boundary
3. 末期完全 opaque 时,渲染结果与最终 inference 时 z-buffer 渲染完全一致 — **训练-推理 consistency**

### Sorting approximation in optimization

虽然最终渲染是 sorting-free，训练时 surfels 还是半透明，必须 alpha-blend, 需要某种排序。直接用 3DGS 风格的 tile-based center depth sort 会导致 surfels 互相 interleave (因为多个 surfel 覆盖同一 pixel 时, center depth 排序 ≠ 实际 fragment depth 排序)。

GES 的解法:
- $w_i < 30$ 时: 用 tile-based sort (此时 surfel 几乎透明, sort 错误影响小)
- $w_i \ge 30$ 时: tile-based sort 后, 对每个 pixel 找出覆盖它的 surfel 中 **depth 最小** 那个作为第一个 blending, 其他顺序不调整

这是一个 "只把最重要的那一个排到最前" 的近似。当 surfel 大部分 opaque 时, 前面的 surfel 主导 pixel 颜色, 其他排序错误的影响指数衰减 (因为 $\prod(1-\alpha_j)$ 中 $\alpha_j \to 1$ 就让后续几乎没贡献)。

---

## 5. Joint Gaussian-Surfel optimization

20K iter 后进入第二阶段:
- Surfel 的 $\mathbf{p}_i, \mathbf{r}_i, \mathbf{s}_i$ 全部 freeze
- Gaussian 新增 (用 10K iter 删掉的 $w_i < 0.8$ surfel 位置 init)
- 只优化 Gaussian 的所有属性 + surfel 的 SH 系数

Gaussian densification: 每 1000 iter 计算所有 training view 的 squared error map, normalize 成 sampling probability, 采样一定数量 pixel, 用 surfel depth + camera 参数 unproject 出 3D 点作为新 Gaussian 位置。

Gaussian pruning: 计算 contribution score
$$s_{i,j} = \max_{\hat{\mathbf{x}}} \frac{c_{i,j} \alpha_{i,j}(\hat{\mathbf{x}})}{W_S + W_G}$$
其中 $c_{i,j} = \max_k \{c_{i,j}^k\}$ 是 Gaussian color 在 view $j$ 下的最大 RGB 通道值 (这样 specular 高光也能被识别为高贡献)。剪掉所有 view 中 $\max s_{i,j} < 0.02$ 的 Gaussians。

直觉: Gaussian 要在所有训练 view 都"不显眼"才能被剪 — 类似 RadSplat (Niemeyer et al. 2024, https://arxiv.org/abs/2403.13806) 的思路。

---

## 6. Adaptive depth offset ε (一个小但关键的细节)

公式:
$$\epsilon_i = \frac{5}{D} \sum_{j=1}^{D} s_{i,j}$$

变量:
- $D$: scaling 维度 (3D Gaussian $D=3$, 2D Gaussian $D=2$)
- $s_{i,j}$: 第 $j$ 轴的 scaling 长度
- 系数 5: 一个经验常数

直觉: $\epsilon$ 应该与 Gaussian 的"size"匹配。大 Gaussian 中心可能在 surface 后方稍远处，但它仍可能"伸到" surface 前方覆盖一些 pixel。固定小 $\epsilon$ 会切掉这部分 Gaussian; 固定大 $\epsilon$ 会让本应被遮挡的 Gaussian leak 出来。adaptive $\epsilon$ 按 Gaussian 各向异性 scaling 取平均 × 5, 实现"几何粒度自适应"。

paper Fig.16 展示:
- $\epsilon = 0$: 一些贴 surface 的 Gaussian 被错误 truncate, 颜色不连续
- $\epsilon = 0.1$ (固定): 精细结构出现 color leak
- adaptive $\epsilon$: 两边问题都解决

Ablation 数据 (Table 5):
- adaptive $\epsilon$: SSIM 0.813, PSNR 27.38, LPIPS 0.208
- $\epsilon = 0$: SSIM 0.806, PSNR 27.11, LPIPS 0.212
- $\epsilon = 0.1$: SSIM 0.811, PSNR 27.19, LPIPS 0.216

差距不算巨大, 但 qualitative 上明显。

---

## 7. 四个 extensions (paper Section 6)

### Mip-GES: anti-aliasing
- Surfel 用 4× MSAA (硬件标准)
- Gaussian 用 MipSplat (Yu et al. 2024, https://arxiv.org/abs/2312.04537) 的 world-space filter (3D Gaussian 上的 low-pass) + screen-space box filter (2D 投影后的 anti-alias)

效果: 远景不再 dilation, 细结构不再 high-frequency aliasing (Fig.9)。质量 SSIM 0.812, PSNR 27.42, LPIPS 0.208, 与 3D-GES 几乎一样但 aliasing 大幅减少。

### Speedy-GES: 加速
用 SpeedySplat (Hanson et al. 2024) 的 Hessian pruning score 替换 paper 自定义 contribution score。Hessian score 评估"删掉这个 Gaussian 后 loss 的二阶增量"，能更精确识别冗余 Gaussian。

结果: **1135 fps @ 1080p, 348 fps @ 4K**, 存储 185 MB (3DGS 是 734MB), 质量 SSIM 0.806, PSNR 27.07 (轻微下降但视觉上 Fig.6 显示 SpeedySplat 即使 prune 30% 都不如 Speedy-GES)。

### Compact-GES: 压缩
用 C3DGS (Lee et al. 2024, https://arxiv.org/abs/2403.15586) 的方法:
- SH 系数替换为 hash grid query (类似 InstantNGP)
- scaling 和 rotation 用 residual vector quantization

结果: **47 MB** (比 C3DGS 49MB 还少)，20× 压缩，质量 PSNR 26.98 (vs 3D-GES 27.38, 损失 ~0.4 dB)。FPS 降到 300 (因为 hash query 慢于直接 SH)。

### 2D-GES: 几何重建
这是最有意思的 extension。把 3D Gaussian 替换为 2D Gaussian (Huang et al. 2DGS, https://arxiv.org/abs/2403.11188)。Surfel 本身是 2D opaque disc, 之间 normal 和 depth 在边界是 discontinuous 的。Gaussian 用于平滑:

$$D_G(\hat{\mathbf{x}}) = \sum_i \mathbb{1}(d_i < d_S + \epsilon) \cdot d_i \cdot \alpha_i$$
$$N_G(\hat{\mathbf{x}}) = \sum_i \mathbb{1}(d_i < d_S + \epsilon) \cdot \mathbf{n}_i \cdot \alpha_i$$
$$D_{smooth} = \frac{D_S W_S + D_G}{W_S + W_G}, \quad N_{smooth} = \frac{N_S W_S + N_G}{W_S + W_G}$$

变量:
- $d_i, \mathbf{n}_i$: Gaussian 的 planar depth 和 normal (2D Gaussian 有明确的 plane)
- $D_S, N_S$: surfel 渲染的 depth / normal map (discontinuous)

效果 (Table 4 DTU chamfer distance):
- 3DGS: 1.97 (mean)
- 2DGS: 0.80
- 3D-GES (用 3D Gaussian 的 central depth + 最短轴 normal): 0.85
- 2D-GES: 0.79 (与 2DGS 持平, 但 glossy surface 上 2DGS 有 holes, 2D-GES 没有, 见 Fig.11)

2D-GES 的关键 trick 是 OpenGL 实现时把 screen-space EWA filter 反投影到 object space (Ren et al. 2002, https://onlinelibrary.wiley.com/doi/10.1111/1467-8659.00606), 简化为对 scaling 的扩展:
$$\rho_i'(\mathbf{x}) \approx G_{\mathbf{S}_i \mathbf{S}_i^T}(\mathbf{x} - \mathbf{p})$$
$\mathbf{S}_i = \text{diag}(s_{i,1}, s_{i,2})$，渲染时把原 scaling 乘 $\mathbf{S}_i$，原 opacity 乘 $1/(s_{i,1}s_{i,2})$ 补偿。这让 2D-GES 也有 anti-aliasing 而 shader 实现简洁。

---

## 8. 实验数据深入解读

### Rendering quality (Table 1, Mip-NeRF360 dataset)

| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|---|---|---|---|
| 3DGS | 0.814 | 27.43 | 0.214 |
| MipSplat | 0.815 | 27.49 | 0.214 |
| AbsGS | 0.821 | 27.49 | 0.208 |
| StopThePop | 0.816 | 27.44 | 0.217 |
| SortFreeGS | 0.790 | 27.04 | 0.263 |
| **3D-GES** | 0.813 | 27.38 | 0.208 |
| **Mip-GES** | 0.812 | 27.42 | 0.208 |
| **2D-GES** | 0.808 | 26.76 | 0.219 |

观察:
- 3D-GES LPIPS 0.208 与 AbsGS 持平, 比 3DGS 好 (0.214) — 这表明 perceptual quality 略胜
- PSNR 略低 (27.38 vs 27.43) — 因为 specular reflections 用 opaque surfels 难拟合 (paper Section 8 讨论)
- 2D-GES PSNR 最低, 但 LPIPS 反而好, 这是 bi-scale 表示的 trade-off (geometry 一致性换来部分 pixel error)
- SortFreeGS 全面较差 — sort-free 但 weighted sum 假设太弱

### Rendering speed (Table 3, 1080p / 2160p)

| Method | FPS 1080p | FPS 4K | Storage | Train |
|---|---|---|---|---|
| 3DGS | 185 | 62 | 734 MB | 28 min |
| StopThePop | 167 | 55 | 830 MB | 40 min |
| MipSplat | 131 | 43 | 1054 MB | 37 min |
| SpeedySplat | 1140 | 369 | 78 MB | 14 min |
| AdrGS | 537 | 195 | 274 MB | 16 min |
| SortFreeGS | 321 | 168 | 506 MB | 42 min |
| **3D-GES** | **675** | **233** | 366 MB | 43 min |
| **2D-GES** | **718** | **247** | 343 MB | 52 min |
| **Speedy-GES** | **1135** | **348** | 185 MB | 36 min |
| **Compact-GES** | 300 | 128 | 47 MB | 39 min |

**关键观察**:
1. 3D-GES 是 3DGS 的 3.6×，与 AdrGS 同 storage 量级但更快且 quality 更好
2. Speedy-GES 与 SpeedySplat 速度持平但 quality 高一档 (Fig.6)
3. Compact-GES 把 storage 压到 47MB (mobile 级别)，FPS 仍 300 (real-time)
4. 4K 下 Speedy-GES 还有 348 fps, 这对 VR/AR 极重要

### View consistency (Table 2, FLIP1 / FLIP7, 越小越好)

| Method | FLIP1↓ | FLIP7↓ |
|---|---|---|
| 3DGS | 0.041 | 0.128 |
| StopThePop | 0.037 | 0.126 |
| SpeedySplat | 0.043 | 0.130 |
| SortFreeGS | 0.034 | 0.120 |
| **3D-GES** | **0.032** | **0.117** |
| **2D-GES** | 0.032 | **0.114** |

GES 全面最佳。FLIP1 是相邻 frame 比较，FLIP7 是 7-frame 间隔比较 (long-term popping)。SortFreeGS 也很好但 quality 差太多。

### Geometry (Table 4, DTU Chamfer Distance)

DTU 数据集 15 个 scene 的 chamfer distance (单位 mm, 越小越好):
- 3DGS: 1.97 mean
- 2DGS: 0.80 mean
- 3D-GES (加 2D-GES 的 regularization): 0.85 mean
- 2D-GES: 0.79 mean

2D-GES 几何质量与 2DGS 持平甚至更好。Fig.11 显示 2DGS 在 glossy 表面有 hole (因为 2D Gaussian 在 specular 区域容易"飘走"为了拟合 color), 而 2D-GES 因为 opaque surfel 锚定几何, Gaussian 只在 surface 附近, glossy 区域 hole 不出现。

### Primitive count (Table 6)

Garden / Room 两个 scene:
- 3DGS: 5.83M / 1.59M (全 Gaussian)
- 3D-GES: 0.19M surfels + 2.46M Gaussians / 0.15M + 0.55M
- Speedy-GES: 0.17M + 0.82M / 0.13M + 0.18M

Surfel 数量极少 (~0.15-0.19M), 但承担了 coarse geometry + 大部分 appearance; Gaussian 仍是主力做 fine detail。

### Ablation (Table 5)

- Only surfels ($C = C_S$): SSIM 0.611, PSNR 24.22 — 完全靠 surfels, 没细节, 但 FPS 4771 (极快)
- Only Gaussians ($C = C_G / W_G$): SSIM 0.774, PSNR 26.02 — 严重 color leak
- Single-stage: 0.809 / 27.04 — 比 two-stage 差 0.34 dB
- RGB surfels (无 SH): 0.808 / 27.29 — 几乎无差, 且更快

**single-stage 失败的原因**: 如果一开始就 joint train, 3D Gaussian 太 flexible 会"包揽"所有 fitting, surfel 不会被驱动去 cover surface。两阶段先逼 surfel 长出 coarse geometry, 再让 Gaussian 补 detail, 才能获得 stable coarse geometry 防止 color leak。

---

## 9. 直觉构建: 为什么 GES work

### 直觉 1: 表面 light field + 残差 volumetric

物理上, scene 大部分像素的颜色来自 **first surface hit** 的 radiance, 这部分是 surface light field (position × direction → color)。剩余部分 (translucent foliage, hair, semi-transparent curtain, soft specular bloom, sub-surface scattering like skin) 才真正需要 volumetric 表达。

GES 就是把这个 insight 显式化: surfel 把"first hit radiance" 用 SH 离散存, Gaussian 把"残差 volumetric appearance" 拟合。这样大多数 Gaussian 不需要做"假装是 surface"的工作, 真正只在需要 volumetric 的地方贡献。

### 直觉 2: z-buffer 替代 sorting

排序问题源于 alpha-blending 是 non-commutative。但 **opaque surface 的 z-buffer 是 commutative 的** — 因为 first surface 完全决定 pixel color, 后面的不贡献。GES 把"哪个 surface 最前" 这个问题完全交给 GPU hardware z-buffer (几十年的优化, 比 GPU 上 sort 快几十倍), 让 Gaussian 只在 "surface 已确定" 的基础上做 lightweight refinement。

### 直觉 3: 归一化是 "surface-aligned volumetric blending"

$$C = \frac{C_S W_S + C_G}{W_S + W_G}$$

这等价于: "把 surface 看成 weight 1 的 'Gaussian-like' 贡献, 加上 surfel 附近所有 Gaussian 的 weight, 归一化"。可以解释为 **surface-aligned 1-layer volume rendering** — 比 NeRF (任意多层 ray marching) 简单, 比 3DGS (沿 ray 全部 blending) 更结构化。

### 直觉 4: opacity modulating 是 "anneal to opaqueness"

跟 Simulated Annealing 的精神一致: 早期高熵 (高透明, 大 gradient, 探索), 后期低熵 (opaque, 稳定, exploit)。$w_i$ 起到 "temperature" 的作用。这种 annealing 让非微的 opaque rasterization 通过"渐进 freeze" 被整合进可微 optimization, 是个很漂亮的 trick, 在不同iable rendering 其他领域也可能有借鉴价值。

### 直觉 5: 两阶段 forced 分工

如果让 surfel 和 Gaussian 同时自由优化, Gaussian 一定 "赢" 因为它更 expressive (3D + opacity, 可以 fit 任意 stuff)。强制先优化 surfel 让它 "占位" coarse geometry, 把"easy fits"都拿走, 让 Gaussian 只能补 "hard residuals"。这种"先 simple 后 complex" 的 curriculum 类似 boosting。

---

## 10. 局限性

paper Section 8 自己承认:
1. **Specular surface**: 3DGS 用大量 low-opacity Gaussian 模拟 reflection (像一层"假镜面"), opaque surfel 没这能力 (一个 surfel 一个颜色), 质量下降。可以用 ray tracing (Moenne-Loccoz et al. 3D Gaussian Ray Tracing, https://arxiv.org/abs/2404.18457) 或 environment map (Ye et al. Deferred Reflection, https://doi.org/10.1145/3641519.3641528) 改进。
2. **Initialization sensitivity**: 与 3DGS 一样依赖 SfM 点。随机 init 质量下降。3DGS-MCMC (Kheradmand et al. 2024, https://arxiv.org/abs/2404.09608) 的 deterministic state transition 可能能解决。
3. **Training time**: 1.3-1.6× 慢于 3DGS, 主要因为 surfel 优化阶段。
4. **未涉及**: dynamic scenes (4DGS, https://arxiv.org/abs/2403.11154)、large-scale urban (Level-of-Detail Gaussian, Kerbl et al. 2024, https://repo.voria.fr/handle/hierarchical-3d-gaussians)。

---

## 11. 与其他工作的关系图谱

- **vs 3DGS**: 把 "all-Gaussian volumetric" 替换为 "surfel + Gaussian bi-scale", 完全 sort-free
- **vs 2DGS**: 2DGS 用 2D Gaussian 但仍是 volumetric + sorted blending; GES 用 2D opaque surfel 真正 surface-bound; 2D-GES 把 2DGS 的 2D Gaussian 借来当 detail smoother
- **vs Gaussian Surfels (Dai et al. 2024, https://arxiv.org/abs/2404.17762)**: Gaussian Surfels 也是 surface-bound 但仍然半透明 + sorted; GES 的 surfel 完全 opaque + z-buffer
- **vs SortFreeGS**: 都 sort-free, 但 SortFreeGS 用 weighted sum (无 occlusion reasoning, leak), GES 用 z-buffer + normalized accumulation (有 occlusion, no leak)
- **vs StopThePop**: StopThePop 优化 sorting 算法; GES 干脆消除 sorting 需求
- **vs SpeedySplat**: SpeedySplat 用 Hessian pruning 减 Gaussian; Speedy-GES 借这个 trick 但同时 sort-free, 达到同速度 + 更高质量
- **vs RTG-SLAM (Peng et al. 2024, https://arxiv.org/abs/2404.18188)**: RTG-SLAM 也用"opaque Gaussian + nearly-transparent Gaussian" decomposition, 但 opaque Gaussian opacity 是 Gaussian-decayed (0.99 中心衰减), 不是 GES 的 uniform 1 within disc; 渲染仍走 3DGS sorted blending
- **vs Surface Light Field** (Wood et al. 2000, https://dl.acm.org/doi/10.1145/344779.344925): SLF 需要 high-res geometry + dense images; GES 自动学 surfel + SH, 不需要预先 geometry

---

## 12. 个人 takeaway

GES 在我的理解里是 3DGS 谱系中第一次真正把 **"surface-bound opaque primitive"** 和 **"volumetric residual"** 显式分层 representation 的工作。它的关键 trick (opacity modulating anneal, two-stage forced 分工, normalized additive blending, depth-tested Gaussian accumulation) 都是从 classic graphics (z-buffer, surfel, MSAA) 和 differentiable rendering (SoftRas, anti-aliased mesh) 借来的成熟工具的组合, 但组合出的 representation 第一次同时拿到: ultra-fast (>1000 fps) + SOTA quality + view-consistent (无 popping) + storage-compact (47MB 可能) + good geometry (DTU SOTA)。

它也提示了一个有趣的研究方向: **differentiable rendering 的 representation 应该与 inference-time rendering 的 representation 对齐**, 而不是 train 时一套 (Gaussian) inference 时换一套 (mesh baked from Gaussian)。GES train 和 inference 都是 surfel + Gaussian, 训练-推理 consistency 通过 opacity annealing 实现。

未来可能的延伸:
- Specular / reflection: 在 surfel 上加 microfacet BRDF + environment map, 让 surfel 自身 handle reflection
- Dynamic scenes: surfel 做 rigid/non-rigid tracking, Gaussian handle transient
- Large scene: 把 surfel 做 LOD (远处 surfel 合并), Gaussian 只在近处展开
- Generative prior: 用 surfel 作为 latent (比 3D Gaussian 更紧凑, 更适合 diffusion 生成)
- SLAM: GES 是 natural mapping representation (surfel = landmarks, Gaussian = detail), 可能比 Gaussian-SLAM 更适合

参考链接汇总:
- Paper: https://doi.org/10.1145/3730925
- 3DGS: https://repo.voria.fr/handle/3dgaussian-splatting
- 2DGS: https://arxiv.org/abs/2403.11188
- MipSplat: https://arxiv.org/abs/2312.04537
- StopThePop: https://repo.voria.fr/handle/stop-the-pop
- SpeedySplat: https://arxiv.org/abs/2412.00578
- SortFreeGS: https://arxiv.org/abs/2410.18931
- C3DGS: https://arxiv.org/abs/2403.15586
- RadSplat: https://arxiv.org/abs/2403.13806
- 3DGS-MCMC: https://arxiv.org/abs/2404.09608
- AbsGS: https://arxiv.org/abs/2409.13519 (相关, Ye et al.)
- Gaussian Surfels (Dai et al.): https://arxiv.org/abs/2404.17762
- 3D Gaussian Ray Tracing: https://arxiv.org/abs/2404.18457
- Surface Light Field (Wood et al. 2000): https://dl.acm.org/doi/10.1145/344779.344925
- SoftRasterizer: https://arxiv.org/abs/1904.05190
- Modular Differentiable Rendering (Laine et al.): https://research.nvidia.com/publications/2020/07_Modular-Primitives-High-Performance-Differentiable-Rendering
- Object Space EWA Splatting (Ren et al. 2002): https://onlinelibrary.wiley.com/doi/10.1111/1467-8659.00606
- Surface Splatting (Zwicker et al. 2001): https://dl.acm.org/doi/10.1145/383259.383300
- Surfels: Surface Elements as Rendering Primitives (Pfister et al. 2000): https://dl.acm.org/doi/10.1145/344779.344936
- DTU Dataset: http://roboimagedata.compute.dtu.dk/?page_id=36
- Mip-NeRF 360: https://arxiv.org/abs/2111.12005
