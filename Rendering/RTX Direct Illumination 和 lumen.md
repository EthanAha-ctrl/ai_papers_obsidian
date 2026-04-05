


好的，我来详细解释 RTX Direct Illumination (RTXDI) 和 Lumen 这两个实时渲染技术。

## RTX Direct Illumination (RTXDI)

**RTXDI** 是 NVIDIA 推出的一个框架，专门用于实现**高效的直接光照采样**，能够在实时渲染中处理**数百万个动态光源**。

### 核心思想传统渲染中，每个光源都需要单独计算贡献，光源数量受限。RTXDI 使用 **Spatiotemporal Importance Resampling (STIR)** 技术，通过重要性采样和时空缓存，用有限的 ray budget（如每像素 1-2 个 ray）高效地聚合海量光源的贡献。

### 关键技术细节

#### 1. Spatiotemporal Resampling 流程
RTXDI 的工作流程遵循以下循环：
- **Temporal stage**: 利用上一帧的缓存（称为 reservoir）来估计当前帧的光照贡献。
- **Spatial stage**: 在当前帧内，从相邻像素的 reservoir 进行重采样。

每一帧执行多次（如 4 次）迭代以逐步提升质量。

#### 2. Reservoir 和 BRDF 采样
RTXDI 基于 ReSTIR (Reservoir-based Spatiotemporal Importance Resampling) 框架。核心数据结构是 **reservoir**，它存储了：
- 光源采样位置（light sample position）
- 权重（weight）
- 累计贡献累计值（sum of contributions）

公式上，**目标 PDF** 为：
```
p̂_target(x, ω_o) = Σ_{i=1}^{N_lights} p_i(x, ω_o) * (L_i * f_r * cosθ) / E[contribution]
```
其中：
- p_i: 第 i 个光源的采样概率
- L_i: 光源的辐射亮度（radiance）
- f_r: BRDF 值
- cosθ: 几何项中的余弦
- E[contribution]: 当前估计的期望贡献

**实际采样** 使用 **BRDF 导向的采样** + **灯光导向的采样** 的组合，并通过 reservoir 存储最优光源。

#### 3. Direct Light Sampling
对于直接光照，RTXDI 使用 **light tree** 或 **BVH** 来加速对大量灯光的查询。灯光可以动态添加/删除，BVH 更新开销很低。

#### 4. 递归 GI 支持
RTXDI 可以扩展到 **间接光照**，通过对每个 reservoir 中的命中点继续进行 ray tracing，将光照传递到其他表面，形成一次或多次反弹。

#### 性能数据
NVIDIA 的 demo 显示，在 **单张 RTX GPU** 上，可以实现：
- **百万级动态 point/spot lights**
- 每帧仅需 **~1-2 rays/pixel**（对于直接光照）
- 延迟在 **10-15 ms**（1080p，不包括 ray generation 时间）

参考链接：
- https://github.com/NVIDIA-RTX/RTXDI
- https://developer.nvidia.com/blog/lighting-scenes-with-millions-of-lights-using-rtx-direct-illumination/

---

## Lumen

**Lumen** 是 Unreal Engine 5 的**全动态全局光照（GI）和反射系统**，专为次世代主机和 PC 设计。它完全基于 **hardware ray tracing** 和 **软件渲染 Fallback**，提供实时的间接光照、反射、阴影和颜色渗色（color bleeding）。

### 核心架构

#### 1. 两个主要 Pipeline
Lumen 使用两套系统：
- **Ray Tracing Pipeline**（推荐，需要 RTX GPU）
- **Software Ray Tracing Pipeline**（Fallback，用于非-RTX GPU）

两者的输出最终融合到同一个全局光照贴图（Surface Cache）。

#### 2. Surface Cache
Lumen 不使用传统的 irradiance cache 或 voxelization，而是使用一个**屏幕空间+世界空间混合的缓存**，称为 **Surface Cache**。这个缓存记录场景表面在各个位置的信息：
- 位置（position）
- 法线（normal）
- BRDF 参数（粗糙度、金属度）
- 上次的间接光照（previous GI）

Surface Cache 以 **voxel 网格**（world-space）组织，但只在有几何体的区域存储，密度可变。

#### 3. Lighting Pass 流程
每一帧 Lumen 的 GI 计算步骤：

**(a) Tracing shadow rays**  
从相机或 surface cache 位置向**灯光**发射 shadow ray，判断直接光照是否被遮挡。

**(b) Indirect ray tracing**  
从表面向**环境发射 ray**（通常是 hemisphere 方向），寻找其他表面作为间接光源。这里有多种追踪模式：
- **Distance Field Ray Marching**（软件模式，使用 SDF）
- **Hardware Ray Tracing**（RT模式，使用 BVH）

Ray tracing 返回击中点（hit point）的颜色（radiance）和距离，然后用**几何衰减**和**BRDF**计算贡献。

**公式**：
```
L_o(p, ω_o) = L_direct + ∫_{Ω} f_r(p, ω_i, ω_o) * L_i(p, ω_i) * cosθ_i * V(p, ω_i) dω_i
```
- L_direct: 直接光照（来自光源）
- ∫: 半球积分
- f_r: BRDF（通常使用 Split-Sum 近似或材质响应表）
- L_i: 间接入射 radiance（来自 traced ray）
- cosθ_i: 几何项
- V: visibility（shadowing）

**(c) Temporal Reprojection**  
利用前一帧的 Surface Cache 和运动矢量，将历史信息 reproject 到当前帧，并进行**temporal accumulation**，降噪并减少闪烁。

**(d) Reflection Pass**  
类似 GI，但反射通常使用更高的 ray count和更精确的采样，因为反射需要锐利细节。

#### 4. Lumen 的 HPG 2020 论文要点
UE5 的 Lumen 基于以下论文（SIGGRAPH 2019/HPG 2020）：
- 使用 **Voxel Global Illumination** 的变种，但 voxel 只在有 geometry 的地方生成（sparse voxelization）
- **Double-buffered** world-space grid，每几帧更新一次，适应动态场景
- **Distance field ray marching** 用于软件模式，加速 ray tracing（无硬件 RT 时）
- **Two-bounce approximation** 用于间接光照：first bounce 用了 ray tracing，second bounce 用的 irradiance cache性能参考（来自 Unreal Engine 5早期 demo）：
- 在 PlayStation 5 上，GI 延时约 **8-12 ms**（含 ray tracing）
- 分辨率1440p，动态场景，保持 60 FPS

参考链接：
- https://www.unrealengine.com/en-US/tech-blog/unreal-engine-5-goes-all-in-on-dynamic-global-illumination-with-lumen
- https://cghero.com/glossary/what-is-lumen
- https://radicalgalaxy.com/advancing-architectural-realism-with-lumen-and-nanite-in-ue5/

---

## RTXDI vs Lumen 对比

| 特性 | RTXDI | Lumen |
|------|--------|--------|
| **发光体类型** | 直接光源（point/spot/area lights） | 全动态 GI，包括间接光照 |
| **主要用途** | 渲染数百万灯光 | 实时全局光照 + 反射 |
| **技术基础** | ReSTIR (reservoir-based resampling) | 硬件/软件 ray tracing + surface cache |
| **速度** | 极快，低 ray budget | 中等，需要多次 bounce |
| **硬件需求** | 需要 RTX GPU（RT core） | 需要 RTX GPU 以获得最佳效果，软件 fallback 可用 |
| **兼容引擎** | 独立框架，可集成 | UE5 专属 |
| **动态效果** | 支持灯光完全动态 | 场景几何和材质完全动态 |

### 互补性
实际上，RTXDI 可以用于 Lumen 的**直接光照阶段**（将海量灯光采样集成到 Lumen 管线中），而 Lumen 提供间接光照。两者在技术上不冲突，而是可以结合使用。

---

## 深入公式与 Reservoir 机制（RTXDI）

RTXDI 的核心是 **reservoir R(x)**，对于像素位置 x：
```
R(x) = { (L_i, w_i) | i = 1...M } // M samples
```
每个 sample 包含一个灯光 L_i 和权重 w_i。

**更新规则**（当一个新 candidate sample s 到来时）：
```
p = rand(0, 1)
if p < w_s / (w_current + w_s):
    replace one current sample with s
```
这类似于 **sequential reservoir sampling**。

最终，光照估算为：
```
L̂(x) = (1 / M) * Σ_{i=1}^{M} W_i * L_i_contribution
```
其中：
- W_i: 每个样本的权重（基于 BRDF 和光源强度）
- L_i_contribution: 该灯光对 x 的直接光照贡献（使用 shadow ray 验证可见性）

### 时间重用
 Temporal reuse 通过将上一帧的 reservoir 投影到当前帧，并用 motion vector 修正位置。然后进行 spatial resampling 混合相邻像素。

---

## Lumen 的 Surface Cache 结构

Surface Cache 是一个**稀疏体素网格**（sparse voxel grid），每个 voxel 存储：
- Position: p_v- Normal: n_v
- Material properties: roughness, metallic
- Irradiance: E_v (累积的间接光照)
- Radius: r_v (表示该 voxel 的“影响范围”，用于插值)

每次 frame 会更新部分 voxel（通过 ray tracing 获得新的间接光照），并进行 **temporal filter**：
```
E_v(t) = mix(E_v(t-1 reprojected), E_new, α)
```
其中 α 取决于 velocity 和 confidence。

---

## 总结

**RTXDI** 和 **Lumen** 代表了两种不同的实时 GI 思路：

- **RTXDI** 是“**解决光源采样效率**”的问题，用极少的 ray 聚合海量灯光，适合灯光密度极高的场景（如城市夜景、节日灯光）。

- **Lumen** 是“**解决动态场景全局光照**”的问题，提供完整的 indirect bounces，适合开放世界、室内室外混合场景。

在现代引擎中，两者可以共存：用 Lumen 计算一次和二次 bounce的 GI，再用 RTXDI 处理千万盏动态灯光的直接贡献，最终叠加得到完整的实时光照。

---

## 扩展阅读与参考

- NVIDIA RTXDI 官方实现：https://github.com/NVIDIA-RTX/RTXDI
- ReSTIR 论文（SIGGRAPH 2020）：https://research.nvidia.com/publication/2020/07/27/restir-path-reservoir-based-spatiotemporal-resampling-ray-tracing
- Lumen 技术博客（UE5）：https://www.unrealengine.com/en-US/tech-blog/unreal-engine-5-goes-all-in-on-dynamic-global-illumination-with-lumen
- HPG 2020 Lumen 论文：https://cg.ivd.kit.edu/publications/2020/lumen/
- Wccftech 对 RTXDI 的介绍：https://wccftech.com/rtx-direct-illumination-shows-games-will-soon-feature-up-to-millions-of-dynamic-lights/
- NVIDIA blog 英文版：https://developer.nvidia.com/blog/render-millions-of-direct-lights-in-real-time-with-rtx-direct-illumination-rtxdi/

如果需要更多关于 BRDF 方程、reservoir 概率推导、或 Lumen voxel 更新算法的细节，我可以继续展开。