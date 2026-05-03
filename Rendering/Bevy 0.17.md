

好的！这是一篇非常精彩的技术博客，介绍了 **Bevy 0.17** 中名为 **Solari** 的实时光线追踪系统。让我为你全面深入地解析这篇文章。

---

# 📖 Solari: Bevy 0.17 实时光线追踪系统详解

## 一、背景与动机

### 1.1 项目历史

作者在 **2023年** 开始开发 Solari 项目，经历了多次技术迭代：

| 阶段 | 技术 | 结果 |
|------|------|------|
| 第一版 (2023) | Lumen-style screen space probes | 质量难以控制，probe 复用困难 |
| 第二版 | Radiance Cascades | 带来 artifacts 和性能开销 |
| 当时 | Wgpu raytracing 只是一个未合并的 PR | 维护 fork 成本太高，项目搁置 |
| **现在** | **ReSTIR + DLSS-RR** | 成功集成到 Bevy 0.17！ |

### 1.2 为什么 Indie 游戏引擎需要光线追踪？

作者提出了一个非常实际的观点：

> **即使玩家没有支持光线追踪的硬件，开发者也需要一个"ground truth"（真实参考）**

核心问题：
- 独立开发者常常不知道为什么他们的光照看起来"不对劲"
- 缺少 **indirect light (间接光)**、**proper reflections (正确反射)**、**accurate shadows (精确阴影)**
- 不知道需要 **bake lighting (烘焙光照)**

**光线追踪的价值**：
1. 提供 **参考实现** — 让开发者看到光照"应该"是什么样子
2. **向前看** — 3-4年后光线追踪会更普及
3. **吸引 AAA 开发者** — 《DOOM: The Dark Ages》、《Cyberpunk 2077》都重度依赖 RT
4. 因为很酷！

---

## 二、Frame Breakdown — 渲染管线详解

Solari 支持的渲染特性：
- **Raytraced Diffuse Direct Lighting (DI)** — 直接光照
- **Raytraced Indirect Lighting (GI)** — 间接光照/全局光照
- 光源类型：**Emissive triangle meshes** 或 **Analytic directional lights**
- **完全实时、动态、无需烘焙**

核心技术栈：
```
Direct Lighting  → ReSTIR DI
Indirect Lighting → ReSTIR GI + World-Space Irradiance Cache
Denoising        → DLSS Ray Reconstruction
```

---

## 三、GBuffer Raster Pass

### 3.1 为什么用 Raster 而不是 Ray Tracing for Primary Visibility？

作者选择用 **光栅化** 来处理 primary visibility，而非光线追踪。原因：

| Raster | Ray Tracing (Primary Visibility) |
|--------|----------------------------------|
| 可以使用 **低分辨率代理 mesh** 在 RT 加速结构中 | 必须全分辨率 |
| 主视图保持高质量 mesh 和 texture | 代价高昂 |
| 与 Bevy 的 **Virtual Geometry** 兼容更好 | 不兼容 |

### 3.2 GBuffer Attachments 结构

Bevy 的 GBuffer 使用了紧凑的打包方式：

**主 Attachment: `Rgba32Uint`**

| Channel | 内容 | 编码方式 |
|---------|------|----------|
| **R** | sRGB base color + perceptual roughness | 4x8 unorm |
| **G** | Emissive color | pre-exposed **Rgb9e5** |
| **B** | Reflectance + metallic + baked diffuse occlusion | 4x8 unorm |
| **A** | World-space normal (24 bits) + flags (8 bits) | **Octahedral encoding** |

**其他 Attachments**：
- `Rg16Float` — Motion vectors
- Depth attachment

### 3.3 Drawing 优化技术

- **Multi-draw indirect** — 批量绘制多个 mesh
- **Sub-allocated buffers** — 减少 buffer 切换
- **GPU culling** — Two-pass occlusion culling against **Hierarchical Depth Buffer (HZB)**
- **Bindless textures** — 无需绑定 texture 到 shader slot
- **最小化 pipeline permutations**

---

## 四、ReSTIR DI (Direct Illumination)

### 4.1 问题背景

计算直接光照的朴素方法：
```
For each pixel:
    For each light:
        For each point on the light:
            Calculate contribution
            Test visibility (raytrace)
```
**代价极其高昂！**

解决方案：**Importance Sampling** — 选择"好"的样本，用少量样本逼近真实结果。

### 4.2 ReSTIR 原理概述

**ReSTIR = Reservoir-based Spatiotemporal Importance Resampling**

核心理念：
> 不只是随机采样，而是通过 **重采样** 在像素间 **共享样本**，选择对当前像素贡献最大的光。

**信号放大器**：
- 输入一般样本 → 输出好样本
- 输入好样本 → 输出优秀样本

**关键数据结构: Reservoir**

```wgsl
struct Reservoir {
    sample: LightSample,           // 选中的光样本
    confidence_weight: f32,        // 置信权重 M
    unbiased_contribution_weight: f32,  // 无偏贡献权重 W (作为 PDF)
}
```

### 4.3 ReSTIR DI Pipeline

分 **两个 Compute Dispatch**：

#### **Pass 1: Initial + Temporal Resampling**

**Initial Resampling**:
```
1. 从 Light Tile 中采样 32 个候选光样本
2. 使用 RIS (Resampling Importance Sampling) 选择最亮的
3. 追踪 visibility ray
4. 如果遮挡，设置 unbiased_contribution_weight = 0
```

**为什么 32 个样本？**
- 场景光少时 overkill
- 但这是 Solari 最昂贵的部分之一
- 未来计划让用户控制这个数量

**Temporal Resampling**:
```
1. 通过 motion vector 获取上一帧的像素
2. 使用 pixel_dissimilar heuristic 验证重投影
3. 检查时序光样本是否仍然存在（光未被 despawn）
4. 复用上一帧的 visibility（省一次 raytrace，代价是阴影延迟1帧）
```

**pixel_dissimilar Heuristic**:
```wgsl
fn pixel_dissimilar(depth: f32, world_position: vec3<f32>, 
                    other_world_position: vec3<f32>, 
                    normal: vec3<f32>, other_normal: vec3<f32>) -> bool {
    // Reject if tangent plane difference > 0.3% or angle between normals > 25 degrees
    let tangent_plane_distance = abs(dot(normal, other_world_position - world_position));
    let view_z = -depth_ndc_to_view_z(depth);
    
    return tangent_plane_distance / view_z > 0.003 || dot(normal, other_normal) < 0.906;
}
```

**变量解释**：
- `tangent_plane_distance` — 两点在切平面上的距离
- `view_z` — 视空间深度
- `dot(normal, other_normal)` — 法线夹角余弦值，`0.906 ≈ cos(25°)`

**Merging**: 使用 **constant MIS weights**（比 balance heuristic 便宜很多）

#### **Pass 2: Spatial Resampling + Shading**

**Spatial Resampling**:
```
1. 在 30 像素半径的圆盘内随机选择一个邻居像素
2. 借用其 reservoir
3. 用 pixel_dissimilar 验证
4. 必须追踪 visibility ray（邻居像素的光样本不一定可见）
```

**为什么只用 1 个 spatial sample？**
- 更多样本不改善质量，但增加开销
- 但不能跳过！— spatial sample 是防止 temporal artifacts 的关键

**作者尝试过的其他方案**：
| 方案 | 结果 |
|------|------|
| Subgroup-level resampling | 便宜但有 tiling artifacts，不便于移植 |
| Workgroup-level resampling | 质量好但贵2倍，correlations 破坏 denoiser |

**Shading**:
```wgsl
final_color = shade_with_reservoir(final_reservoir)
```

**总 Raytrace 次数**: **2 rays/pixel** (1 initial + 1 spatial)

---

## 五、ReSTIR GI (Global Illumination)

### 5.1 问题背景

间接光照比直接光照更昂贵：
- 需要追踪 **multiple bounces** 才能得到正确的光照路径
- 一次 bounce 就很贵

### 5.2 GI Reservoir 结构

```wgsl
struct Reservoir {
    radiance: vec3<f32>,                    // 从采样点反射回来的 radiance
    sample_point_world_position: vec3<f32>, // 采样点世界坐标
    sample_point_world_normal: vec3<f32>,   // 采样点法线
    confidence_weight: f32,                  // M
    unbiased_contribution_weight: f32,       // W
}
```

### 5.3 GI Initial Sampling

**只生成 1 个样本**（因为太贵了！）

```wgsl
fn generate_initial_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, 
                               rng: ptr<function, u32>) -> Reservoir {
    var reservoir = empty_reservoir();

    // 1. 从 uniform hemisphere 随机选一个方向
    let ray_direction = sample_uniform_hemisphere(world_normal, rng);
    
    // 2. 追踪 ray
    let ray_hit = trace_ray(world_position, ray_direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);

    if ray_hit.kind == RAY_QUERY_INTERSECTION_NONE {
        return reservoir;  // 没打中东西
    }

    let sample_point = resolve_ray_hit_full(ray_hit);

    // 3. 跳过 emissive mesh（直接光由 ReSTIR DI 处理）
    if all(sample_point.material.emissive != vec3(0.0)) {
        return reservoir;
    }

    // 4. 设置 reservoir 属性
    reservoir.unbiased_contribution_weight = uniform_hemisphere_inverse_pdf();
    reservoir.sample_point_world_position = sample_point.world_position;
    reservoir.sample_point_world_normal = sample_point.world_normal;
    reservoir.confidence_weight = 1.0;

    // 5. 从 World Cache 查询 irradiance
    reservoir.radiance = query_world_cache(sample_point.world_position, 
                                            sample_point.geometric_world_normal, 
                                            view.world_position);

    // 6. 乘以 BRDF
    let sample_point_diffuse_brdf = sample_point.material.base_color / PI;
    reservoir.radiance *= sample_point_diffuse_brdf;

    return reservoir;
}
```

**关键点**：
- 使用 **Uniform Hemisphere Sampling**（未来想尝试 Spatiotemporal Blue Noise）
- 在 hit point 查询 **World Cache** 获取 irradiance
- 乘以 diffuse BRDF: `BRDF = base_color / π`

### 5.4 Permutation Sampling for Temporal

GI 的 temporal resampling 有个特殊技巧：

```
1. 通过 motion vector 重投影
2. 在重投影位置附近 jitter 几个像素
3. 这相当于在 temporal resampling 中加入 spatial component
```

**作用**: 打破 temporal correlations，否则 denoiser 会产生 **blotchy noise**

### 5.5 GI Jacobian — 最难的部分

由于 temporal 和 spatial resampling 都使用邻居像素，需要加入 **Jacobian determinant** 来处理采样域的变化：

```
J = |∂x_neighbor / ∂x_current|
```

**问题**：
1. Jacobian 在角落处增加噪声
2. Jacobian 可能导致数值爆炸 → **inf overflow**
3. 即使长时间运行在静态场景，最终也会 overflow

**解决方案**：
```wgsl
// 当 Jacobian > 2 时拒绝邻居样本
if jacobian > 2.0 {
    reject_neighbor_sample();
}
```

使用 **balance heuristic** 来缓解 Jacobian 引入的噪声。

---

## 六、Light Tile Presampling

### 6.1 问题：Divergent Memory Access

朴素的 ReSTIR DI 实现：
```
每个像素独立：
1. 生成随机光样本
2. 解析光样本
3. 计算贡献
```

**GPU 最讨厌的东西**：
- Divergent branches（分支发散）
- Incoherent memory access（不连贯内存访问）

### 6.2 Light Tile 方案

**Idea**: 把光采样过程分离出来，预计算并存储。

**Light Sampling API**:

```wgsl
struct LightSample {
    light_id: u16,      // 光源在全局列表中的 ID
    triangle_id: u16,   // 三角形 ID（用于 emissive mesh）
    seed: u32,          // 用于初始化 RNG
}

struct ResolvedLightSample {
    world_position: vec4<f32>,  // w=0 表示 directional light, w=1 表示 emissive mesh
    world_normal: vec3<f32>,
    emitted_radiance: vec3<f32>,
    inverse_pdf: f32,
}
```

**Light Tile Generation**:
```
1. 预生成 128 个 tiles
2. 每个 tile 包含 1024 个 ResolvedLightSamplePacked
3. 样本完全随机，不基于场景信息
```

**打包格式**：
```wgsl
struct ResolvedLightSamplePacked {
    world_position_x: f32,
    world_position_y: f32,
    world_position_z: f32,
    packed_normal: u32,        // octahedral encoded
    packed_radiance: u32,      // Rgb9e5
    inverse_pdf: f32,          // 负值表示 directional light
}
```

**效果**: 大幅提高 cache hit rate，**光采样是 Solari 最大的性能瓶颈**！

---

## 七、World Cache (世界空间辐照度缓存)

### 7.1 目的

- ReSTIR GI 只能用 1 个初始样本
- GI 对精度要求比 DI 低 — "mostly correct" 就够了
- 可以在多个像素间 **共享工作**

### 7.2 空间哈希 实现

```wgsl
fn query_world_cache(world_position: vec3<f32>, world_normal: vec3<f32>, 
                     view_position: vec3<f32>) -> vec3<f32> {
    let cell_size = get_cell_size(world_position, view_position);  // LOD

    let world_position_quantized = quantize_position(world_position, cell_size);
    let world_normal_quantized = quantize_normal(world_normal);

    var key = compute_key(world_position_quantized, world_normal_quantized);
    let checksum = compute_checksum(world_position_quantized, world_normal_quantized);

    // 线性探测解决哈希冲突
    for (var i = 0u; i < WORLD_CACHE_MAX_SEARCH_STEPS; i++) {
        let existing_checksum = atomicCompareExchangeWeak(
            &world_cache_checksums[key], 
            WORLD_CACHE_EMPTY_CELL, 
            checksum
        ).old_value;
        
        if existing_checksum == checksum {
            // 已存在 — 返回 irradiance，重置 lifetime
            atomicStore(&world_cache_life[key], WORLD_CACHE_CELL_LIFETIME);
            return world_cache_irradiance[key].rgb;
        } else if existing_checksum == WORLD_CACHE_EMPTY_CELL {
            // 空格子 — 初始化
            atomicStore(&world_cache_life[key], WORLD_CACHE_CELL_LIFETIME);
            world_cache_geometry_data[key].world_position = world_position;
            world_cache_geometry_data[key].world_normal = world_normal;
            return vec3(0.0);
        } else {
            // 冲突 — 跳到下一个位置
            key = wrap_key(pcg_hash(key));
        }
    }

    return vec3(0.0);
}
```

**Key 组成**：
- `world_position` — 量化后的世界坐标
- `world_normal` — 量化后的几何法线（非着色法线）
- LOD factor — 远距离使用更大的 cell

### 7.3 World Cache 更新流程

```
┌─────────────────────────────────────────────────────────┐
│  Frame N                                                │
├─────────────────────────────────────────────────────────┤
│  1. Decay Cells    — 每个 entry 的 life 减 1             │
│  2. Compaction     — 统计活跃 entry，生成密集索引数组     │
│  3. Sample Lighting — 每个 voxel 追踪 2 条 rays:         │
│     - Direct light sample (RIS)                         │
│     - Indirect light sample (cosine hemisphere + cache) │
│  4. Blend New Samples — 与历史值做 temporal blend        │
└─────────────────────────────────────────────────────────┘
```

**关键洞察**: Cache sample from itself!

```
Frame 5: Cell A samples light source
Frame 6: Cell B samples Cell A
Frame 7: Cell C samples Cell B
→ Full path: light → A → B → C → primary surface → camera
```

这实现了 **multi-bounce path tracing**，只不过分布在多个帧上。

**Temporal Blend**:
```wgsl
let sample_count = min(old_irradiance.a + 1.0, WORLD_CACHE_MAX_TEMPORAL_SAMPLES);
let blended_irradiance = mix(old_irradiance.rgb, new_irradiance, 1.0 / sample_count);
```

---

## 八、DLSS Ray Reconstruction

### 8.1 作用

DLSS-RR 同时完成三件事：
1. **Upscaling** — 从 1600x900 → 3200x1800
2. **Anti-aliasing**
3. **Denoising**

### 8.2 为什么需要额外 Copy Pass？

DLSS-RR 需要从特定格式的纹理读取，但 Bevy GBuffer 格式不同，所以需要先 copy 到 standalone textures。

---

## 九、性能分析

### 9.1 测试环境

- **GPU**: RTX 3080
- **分辨率**: 1600x900 → 3200x1800 (DLSS-RR Performance Mode)

### 9.2 各 Pass 耗时

| Pass | PICA PICA (ms) | Bistro (ms) | Cornell Box (ms) |
|------|----------------|-------------|------------------|
| Presample Light Tiles | 0.03 | 0.08 | 0.02 |
| World Cache: Decay | 0.02 | 0.02 | 0.01 |
| World Cache: Compaction P1 | 0.04 | 0.04 | 0.04 |
| World Cache: Compaction P2 | 0.01 | 0.01 | 0.01 |
| World Cache: Write Active Cells | 0.01 | 0.02 | 0.01 |
| **World Cache: Sample Lighting** | 0.06 | **2.09** | 0.05 |
| World Cache: Blend | 0.01 | 0.07 | 0.01 |
| **ReSTIR DI: Initial + Temporal** | **1.25** | **1.85** | **1.28** |
| ReSTIR DI: Spatial + Shade | 0.19 | 0.66 | 0.18 |
| ReSTIR GI: Initial + Temporal | 0.37 | **2.75** | 0.33 |
| ReSTIR GI: Spatial + Shade | 0.44 | 0.60 | 0.46 |
| DLSS-RR: Copy | 0.04 | 0.07 | 0.04 |
| **DLSS-RR** | **5.75** | **6.29** | **5.82** |
| **Total** | **8.22** | **14.55** | **8.25** |

**观察**：
- Bistro 场景最贵（世界更大，更多 cache entries）
- DLSS-RR 占用大量时间，但比不做 upscaling 总开销低

### 9.3 NSight Trace 分析

**瓶颈分析**：

| Pass | 瓶颈类型 | 原因 |
|------|----------|------|
| ReSTIR DI Initial + Temporal | **Memory bound** | 从 light tiles 加载 ResolvedLightSamplePacked |
| ReSTIR DI/GI 其他 passes | **Raytracing throughput** | RT core 繁忙 |

**Occupancy 问题**:
- 当前: 32/48 warps occupied
- 限制因素: **Registers per thread**
- 主要消耗: `resolve_triangle_data_full()` 函数

---

## 十、未来工作

### 10.1 Feature Parity

| 功能 | 状态 |
|------|------|
| Specular materials | 开发中 |
| Transparent materials | 待开发 |
| Alpha-masked materials | 待开发 |
| Custom materials | 被 wgpu RT pipeline 支持阻塞 |
| Skinned meshes | 需要 Bevy GPU-driven skinning |
| Point/Spot lights | 待开发 |
| Image-based lighting | 待开发 |

### 10.2 Light Sampling 改进

当前问题: 光采样完全随机

潜在方案:
1. **Spherical Gaussian Light Trees** — 另一位 Bevy 开发者在探索
2. **MegaLights Visible Light Lists** — 世界空间光可见性列表

### 10.3 Chromatic ReSTIR

问题: 重叠的 RGB 不同颜色的光难以处理 — ReSTIR 只能选一个样本

解决方案: **Ratio Control Variates (RCV)**

```wgsl
// 对每个通道应用不同权重
weight = (sample_contribution / total_scene_light)
```

需要在 reservoir 中跟踪每通道的总光量。

### 10.4 GI 质量问题

**World Cache 问题**:
1. Voxel 位置/法线固定 — 初始化选不好就永远不好
2. **Energy loss** — 体素化导致能量丢失
3. **Temporal instability** — bright outliers 被 ReSTIR 放大
4. **Slow reaction to scene changes** — 移动光源留下 trail

**潜在方案**:
- Reproject last frame 替代 always 用 world cache
- Visible light list in world space 作为反馈机制

---

## 十一、总结

### 关键技术公式

**1. Reservoir Merging (Weighted Reservoir Sampling)**:

$$W = \frac{w_1 M_1 + w_2 M_2}{M}$$

其中：
- $W$ — 合并后的 unbiased contribution weight
- $w_i$ — 各 reservoir 的 weight
- $M_i$ — 各 reservoir 的 confidence weight
- $M = M_1 + M_2$

**2. Light Contribution**:

$$L_{received} = L_{emitted} \cdot \frac{\cos\theta_o \cdot \cos\theta_l}{d^2}$$

其中：
- $\theta_o$ — 着色点法线与光方向的夹角
- $\theta_l$ — 光源法线与光方向的夹角  
- $d$ — 光源到着色点的距离

**3. Diffuse BRDF**:

$$f_r = \frac{\rho}{\pi}$$

其中 $\rho$ 是 base color / albedo

**4. Temporal Blend**:

$$L_{new} = (1 - \alpha) L_{old} + \alpha L_{sample}$$

$$\alpha = \frac{1}{\min(N+1, N_{max})}$$

### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Solari Rendering Pipeline                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐   │
│  │ GBuffer     │     │ ReSTIR DI        │     │ ReSTIR GI       │   │
│  │ Raster      │────▶│ Pass 1: Init+Temp│────▶│ Pass 1: Init+Temp│  │
│  │             │     │ Pass 2: Spatial  │     │ Pass 2: Spatial │   │
│  └─────────────┘     └──────────────────┘     └─────────────────┘   │
│         │                    │                        │             │
│         │                    ▼                        ▼             │
│         │           ┌──────────────────────────────────────┐       │
│         │           │           Light Tiles                │       │
│         │           │   (128 tiles × 1024 samples each)    │       │
│         │           └──────────────────────────────────────┘       │
│         │                                           │               │
│         │                                           ▼               │
│         │                                  ┌─────────────────┐      │
│         │                                  │   World Cache   │      │
│         │                                  │ (Spatial Hash)  │      │
│         │                                  └─────────────────┘      │
│         │                                           │               │
│         └──────────────────┬────────────────────────┘               │
│                          ▼                                          │
│                 ┌─────────────────┐                                 │
│                 │    DLSS-RR      │                                 │
│                 │ (Upscale + Denoise)                               │
│                 └─────────────────┘                                 │
│                          │                                          │
│                          ▼                                          │
│                 ┌─────────────────┐                                 │
│                 │   Final Image   │                                 │
│                 └─────────────────┘                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 十二、参考资源

### ReSTIR 相关
- [A Gentle Introduction to ReSTIR: Path Reuse in Real-time](https://research.nvidia.com/labs/rtr/neural-rendering/reSTIR/)
- [A gentler introduction to ReSTIR](https://agraphicsguy.wordpress.com/2021/03/31/a-gentler-introduction-to-restir/)
- [Rearchitecting Spatiotemporal Resampling for Production](https://research.nvidia.com/labs/rtr/neural-rendering/reSTIR/)

### DLSS-RR
- [NVIDIA DLSS Ray Reconstruction](https://www.nvidia.com/en-us/geforce/technologies/dlss/)
- [dlss_wgpu GitHub](https://github.com/bevyengine/dlss_wgpu)

### Bevy
- [Bevy Engine](https://bevyengine.org/)
- [Bevy GitHub](https://github.com/bevyengine/bevy)

### 光线追踪基础
- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
- [PBR Book](https://www.pbr-book.org/)

### World Cache / Spatial Hash
- [Sparse Voxel Octree GI](https://research.nvidia.com/sites/default/files/pubs/2011-09_Sparse-Voxel-Octree/svo_gi.pdf)
- [Spatial Hashing for Real-Time GI](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-19-generic-adaptive-mesh-refinement)

---

这篇文章是作者三年业余工作的结晶，展示了如何将前沿的学术研究（ReSTIR）应用到实际的游戏引擎中。如果你对光线追踪、实时渲染感兴趣，强烈建议阅读原文并尝试运行 Solari example！