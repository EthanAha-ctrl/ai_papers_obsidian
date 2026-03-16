# 关于 SDF 与 GPU Texture Compression 的技术讨论解析

这个讨论围绕 **Signed Distance Fields (SDF)** 在 GPU 上的实现，特别是 **3D texture compression formats** 的现状与局限展开。让我深入解析各个技术要点。

---

## 1. 讨论背景：SDF 与 GPU 硬件的关系

### 什么是 SDF？

**Signed Distance Field (SDF)** 是一个标量场函数 $f: \mathbb{R}^3 \rightarrow \mathbb{R}$，定义为：

$$
f(\mathbf{p}) = \begin{cases}
+d & \text{if } \mathbf{p} \text{ is outside the object} \\
-d & \text{if } \mathbf{p} \text{ is inside the object} \\
0 & \text{if } \mathbf{p} \text{ is on the surface}
\end{cases}
$$

其中 $d = \min_{\mathbf{s} \in S} \|\mathbf{p} - \mathbf{s}\|$ 表示点到最近表面的距离，$S$ 是物体表面点集。

### SDF 的核心优势

| 特性 | 说明 |
|------|------|
| **Level-set representation** | 表面是 $f(\mathbf{p}) = 0$ 的等值面 |
| **Ray marching 友好** | 可用 sphere tracing 算法高效求交 |
| **布尔运算简单** | Union: $\min(f_1, f_2)$, Intersection: $\max(f_1, f_2)$, Subtraction: $\max(f_1, -f_2)$ |
| **自适应细节** | 配合 LOD/Mipmap 可动态调整精度 |

---

## 2. Texture Compression Formats 深度解析

### 2.1 BC (Block Compression) 系列格式

**BC 格式**（又称 **DXT/S3TC**）是桌面 GPU 的主流纹理压缩标准。核心思想是 **fixed-rate block-based compression**：

```
┌─────────────────────────────────────────────────────┐
│  4×4 texel block (16 pixels)                        │
│  ┌───┬───┬───┬───┐                                  │
│  │   │   │   │   │                                  │
│  ├───┼───┼───┼───┤   Compressed to:                 │
│  │   │   │   │   │   • 2 endpoint colors (c0, c1)   │
│  ├───┼───┼───┼───┤   • 16 × 2-bit indices           │
│  │   │   │   │ │ │                                   │
│  └───┴───┴───┴───┘                                  │
│                                                      │
│  Reconstruction: color = lerp(c0, c1, index/3)      │
└─────────────────────────────────────────────────────┘
```

#### BC4 单通道格式详解

**BC4** 是专为单通道数据设计的格式（常用于 height maps, shadow maps, SDF）：

- **压缩率**: 0.5 bytes/texel（原始 1 byte → 压缩后 0.5 byte）
- **Block size**: 8 bytes 存储 4×4 = 16 texels
- **两种 encoding modes**:
  - **Mode 0**: 6 interpolation levels
  - **Mode 1**: 4 interpolation levels + 2 explicit values

```
BC4 Block Layout (64 bits):
┌────────────────────────────────────────────────────┐
│ Bit 0:      Mode select (0 or 1)                   │
│ Bits 1-16:  Endpoint A (red_0, 16-bit unorm)       │
│ Bits 17-32: Endpoint B (red_1, 16-bit unorm)       │
│ Bits 33-64: 16 × 2-bit indices                     │
└────────────────────────────────────────────────────┘

Interpolation (Mode 0):
  index=0 → color = red_0
  index=1 → color = red_1  
  index=2 → color = (2/3)*red_0 + (1/3)*red_1
  index=3 → color = (1/3)*red_0 + (2/3)*red_1
  index=4 → color = (1/2)*red_0 + (1/2)*red_1
  index=5 → color = 0  (or special value)
```

**TinkersW 指出的问题**：BC4 的 Mode 1 对 SDF 来说几乎是浪费，因为 SDF 需要的是 **均匀分布的距离值**，而非特殊编码模式。

### 2.2 ASTC (Adaptive Scalable Texture Compression)

**ASTC** 是更先进的压缩格式，由 ARM 开发：

```
ASTC 特性：
┌────────────────────────────────────────────────────────┐
│  • 可变 block size: 4×4 到 12×12 (2D), 3×3×3 到 6×6×6 (3D) │
│  • 支持 3D blocks (原生平面压缩!)                        │
│  • 支持多达 4 channels                                  │
│  • Bitrate: 0.89 - 8.0 bpp (bits per pixel)            │
│  • LDR/HDR 支持                                        │
│  • 高质量，但编解码复杂                                  │
└────────────────────────────────────────────────────────┘
```

**ASTC 3D Block 示例**：
```
3D Block: 4×4×4 = 64 voxels
压缩到: 128 bits (2 bits/voxel)
支持: 3D 插值, 3D filtering
```

**关键问题**：ASTC 的 **桌面 GPU 支持极其有限**：

| GPU Vendor | ASTC 支持 | BC 支持 |
|------------|----------|---------|
| NVIDIA (Desktop) | ❌ 不支持 | ✅ 全系列 |
| AMD (Desktop) | ❌ 不支持 | ✅ 全系列 |
| Intel (Desktop) | ⚠️ 部分支持 | ✅ 支持 |
| Apple M-series | ✅ 完整支持 | ✅ 支持 |
| Mobile (ARM) | ✅ 标配 | ⚠️ 部分支持 |
| NVIDIA Tegra | ✅ 支持 | ✅ 支持 |

**参考**: [Khronos ASTC Wiki](https://www.khronos.org/opengl/wiki/ASTC_Texture_Compression), [Wikipedia ASTC](https://en.wikipedia.org/wiki/Adaptive_scalable_texture_compression)

---

## 3. 3D Texture 与 SDF 存储的挑战

### 3.1 当前 PC 平台的局限

**TinkersW 的核心抱怨**：

> "An actual 3D block based format would be useful, currently on PC at least the only formats are for 2D textures."

这意味着：

```
当前 PC GPU 纹理压缩现状：
┌─────────────────────────────────────────────────────────┐
│                                                          │
│   3D Texture = 多层 2D slices 拼接                       │
│                                                          │
│   ┌───────┐  ┌───────┐  ┌───────┐                       │
│   │ 2D    │  │ 2D    │  │ 2D    │  ... (N slices)       │
│   │ Slice │  │ Slice │  │ Slice │                       │
│   │ 0     │  │ 1     │  │ 2     │                       │
│   └───────┘  └───────┘  └───────┘                       │
│       ↓          ↓          ↓                           │
│   每层独立 2D 压缩 (BC format)                           │
│                                                          │
│   问题:                                                  │
│   • Z 方向没有压缩/相关性利用                            │
│   • Z 方向 interpolation 精度低                         │
│   • 内存带宽浪费                                         │
│                                                          │
└─────────────────────────────────────────────────────────┘

理想情况 (ASTC 3D blocks):
┌─────────────────────────────────────────────────────────┐
│   ┌─────────────┐                                       │
│   │ 4×4×4 Block │  ← 真正的 3D 压缩                     │
│   │   (3D)      │    Z 方向也有相关性编码               │
│   └─────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

### 3.2 为什么 Z 方向压缩很重要？

对于 SDF，3D 空间中的值具有 **强空间相关性**：

$$
|f(\mathbf{p}) - f(\mathbf{p} + \Delta)| \leq \|\Delta\|
$$

这是 SDF 的 **Lipschitz 连续性**。意味着相邻 voxel 的值差异有限，理想的 3D 压缩应利用这个性质。

---

## 4. TMU (Texture Mapping Unit) 与插值精度

### 4.1 什么是 TMU？

**Texture Mapping Unit (TMU)** 是 GPU 中专门处理纹理采样的固定功能单元：

```
GPU Pipeline 中的 TMU：
┌────────────────────────────────────────────────────────┐
│                                                         │
│   Shader Core ──→ TMU ──→ Texture Cache ──→ Memory     │
│                   │                                     │
│                   ▼                                     │
│            ┌──────────────┐                            │
│            │ Address      │  计算 texel 地址           │
│            │ Calculation  │  (u,v,w → texel coords)    │
│            └──────────────┘                            │
│                   │                                     │
│                   ▼                                     │
│            ┌──────────────┐                            │
│            │ Filtering    │  Bilinear/Trilinear/       │
│            │ Unit         │  Anisotropic filtering     │
│            └──────────────┘                            │
│                   │                                     │
│                   ▼                                     │
│            ┌──────────────┐                            │
│            │ Format       │  解压缩 BC/ASTC            │
│            │ Decode       │  格式转换                   │
│            └──────────────┘                            │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### 4.2 插值精度讨论

**dogma1138 提到的关键点**：

> "Aren't the TMUs on AMD GPUs still cap at FP16 at least for advertised throughput?"

这涉及到 TMU 的 **内部计算精度**：

#### 纹理过滤公式

**Bilinear interpolation** (2D):
$$
T(u,v) = \sum_{i=0}^{1}\sum_{j=0}^{1} w_{ij} \cdot T_{ij}
$$

其中权重：
$$
\begin{aligned}
w_{00} &= (1-\alpha)(1-\beta) \\
w_{10} &= \alpha(1-\beta) \\
w_{01} &= (1-\alpha)\beta \\
w_{11} &= \alpha\beta
\end{aligned}
$$

$\alpha, \beta \in [0,1]$ 是 fractional coordinates。

**Trilinear interpolation** (3D):
$$
T(u,v,w) = (1-\gamma)T_{low}(u,v) + \gamma \cdot T_{high}(u,v)
$$

其中 $\gamma$ 是 mip level 间的插值因子。

#### AMD vs NVIDIA TMU 精度

| 方面 | NVIDIA | AMD (GCN/RDNA) |
|------|--------|----------------|
| **Filtering 精度** | 高精度 (接近 FP32) | 优化为 FP16/FP24 |
| **吞吐量** | ~2× AMD | 标准吞吐 |
| **Sub-pixel 精度** | 更好的 edge cases | 某些情况用 shader 模拟 |
| **FP16 模式** | 支持但不限于此 | FP16 是"甜点"精度 |

**关键洞察**：TMU 设计是 **带宽-精度-面积** 的 trade-off：

$$
\text{TMU Throughput} \propto \frac{1}{\text{Precision} \times \text{Filter Taps}}
$$

对于 SDF ray marching，精度影响：

```
SDF Ray Marching 算法:
────────────────────────────────────────────────
d = texture3D(sdf, p)  // TMU 插值

if (d < threshold):
    // 需要高精度判断 surface intersection
    // TMU 精度直接影响 rendering artifacts

step = d  // conservative stepping
p += ray_dir * step
────────────────────────────────────────────────
```

**参考**: [NVIDIA Texture Filtering](https://developer.nvidia.com/gpugems/GPUGems2/gpugems2_chapter20.html), [AMD RDNA Architecture](https://www.amd.com/system/files/documents/rdna-whitepaper.pdf)

---

## 5. OpenVDB 与 NanoVDB

### 5.1 OpenVDB 简介

**OpenVDB** 是 DreamWorks 开发的开源 C++ 库，专为 **稀疏体积数据** 设计：

```
OpenVDB 数据结构：
┌────────────────────────────────────────────────────────┐
│                                                         │
│  Root Node (root)                                       │
│       │                                                 │
│       ├── Internal Node (level 2)                       │
│       │       ├── Internal Node (level 1)               │
│       │       │       ├── Leaf Node (level 0)           │
│       │       │       │   ├── 8×8×8 voxels (dense)      │
│       │       │       │   └── Value mask (active/inactive) │
│       │       │       └── ...                           │
│       │       └── ...                                   │
│       └── ...                                           │
│                                                         │
│  特性：                                                  │
│  • Hierarchical sparse storage                         │
│  • O(log n) access time                                │
│  • 空区域不占用内存                                      │
│  • VFX 行业标准                                         │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### 5.2 NanoVDB: GPU-Optimized 版本

**NanoVDB** 是 OpenVDB 的轻量级 GPU 友好版本：

```cpp
// NanoVDB 特点
struct NanoVDB {
    // 1. Flat, self-contained memory layout
    // 2. No external dependencies
    // 3. GPU-friendly access pattern
    
    // Random access on GPU:
    float readVoxel(Vec3f p) {
        // Tree traversal on GPU
        // Cache-friendly for ray marching
    }
};
```

**但关键问题是**：OpenVDB/NanoVDB 是 **软件数据结构**，不是 **硬件压缩格式**：

```
对比：
┌────────────────┬─────────────────────┬──────────────────┐
│                │ OpenVDB/NanoVDB     │ BC/ASTC          │
├────────────────┼─────────────────────┼──────────────────┤
│ 硬件支持       │ ❌ 无               │ ✅ TMU 直接解码  │
│ 带宽优化       │ ⚠️ Sparse, 但需计算 │ ✅ 固定压缩率   │
│ 随机访问       │ ✅ 支持             │ ✅ 支持          │
│ 实时编码       │ ❌ 复杂             │ ✅ 简单          │
│ 压缩率         │ ⚠️ 依赖稀疏性       │ ✅ 固定/可预测   │
└────────────────┴─────────────────────┴──────────────────┘
```

**参考**: [OpenVDB Official](https://www.openvdb.org/), [NanoVDB GitHub](https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb)

---

## 6. Apple Metal 与 3D Textures

**jb1991 提到**：

> "Metal offers 3D textures, and they are nice for SDFs."

**Metal** 的 3D texture 支持：

```swift
// Metal 3D Texture 创建
let descriptor = MTLTextureDescriptor()
descriptor.textureType = .type3D
descriptor.width = 256
descriptor.height = 256
descriptor.depth = 256
descriptor.pixelFormat = .r32Float  // 32-bit float per voxel
descriptor.storageMode = .private   // GPU-only memory

let texture3D = device.makeTexture(descriptor: descriptor)
```

**Metal 的优势**：
- 原生 3D texture 支持
- 支持 ASTC 压缩（Apple Silicon）
- 高效的 trilinear filtering
- 与 SDF ray marching 配合良好

---

## 7. 讨论中的关键技术争议

### 7.1 "Separating block min/max" 的设计批评

**TinkersW 批评原文讨论的方法**：

> "separating the block min/max into a separate memory location from the indices doesn't make much sense"

这涉及 **block compression 的数据布局优化**：

```
传统 BC 格式:
┌─────────────────────────────────────┐
│ Block N:                             │
│   [min][max][indices...]             │  ← 所有数据连续存储
└─────────────────────────────────────┘

提议的分离布局:
┌─────────────────────────────────────┐
│ Separate Region A:                   │
│   [min₀][max₀][min₁][max₁]...       │  ← 所有 min/max 连续
├─────────────────────────────────────┤
│ Separate Region B:                   │
│   [indices₀][indices₁]...           │  ← 所有 indices 连续
└─────────────────────────────────────┘
```

**问题在于**：
- 如果目的是降低精度，直接用 lower mip level 即可
- 分离存储破坏了 cache locality
- 增加了 memory indirection

### 7.2 NVIDIA 与 SDF 的关系

**cpgxiii 的观点**：

> "Nvidia GPUs can perform interpolation on 3D textures, which is specifically useful for performing finer distance queries on a 3D SDF."

**atq2119 的反驳**：

> "Interpolation of 3D textures has been a standard feature in graphics APIs for a very long time."
> "the author is correct in that this isn't specifically geared towards SDFs."

**真相在中间**：

```
3D Texture Interpolation 的历史：
┌────────────────────────────────────────────────────────┐
│ 1990s: SGI RealityEngine ─ 首次硬件 3D texture        │
│ 2000s: NVIDIA/AMD 普遍支持 3D texture                 │
│ 2010s: SDF 成为实时渲染研究热点                        │
│                                                         │
│ NVIDIA 的特殊之处：                                     │
│ • 内部研究大量使用 SDF (Neural SDF, Instant-NGP 等)   │
│ • TMU 精度设计对 SDF 更友好                            │
│ • 但 TMU 并非 "为 SDF 设计"                            │
└────────────────────────────────────────────────────────┘
```

**cpgxiii 后续澄清**：

> "Nvidia does specifically use their 3D texture units for SDF interpolation in their simulation work."

这指 NVIDIA 内部的 **simulation work**（可能是 Omniverse 或物理模拟项目）大量使用 SDF，所以 TMU 设计考虑了这个用例。

---

## 8. 技术总结与未来方向

### 8.1 当前 SDF 存储的最佳实践

```
SDF 存储选择矩阵：
┌─────────────────────────────────────────────────────────┐
│                    │ Desktop GPU    │ Mobile/Apple     │
├────────────────────┼────────────────┼──────────────────┤
│ Dense SDF          │ BC4 / R8       │ ASTC 3D          │
│ Sparse SDF         │ NanoVDB        │ NanoVDB          │
│ High precision     │ R32F (uncomp)  │ R32F             │
│ Real-time encoding │ R8/BC4         │ ASTC (如果支持)  │
└────────────────────┴────────────────┴──────────────────┘
```

### 8.2 行业需求

讨论揭示的核心需求：

1. **真正的 3D block compression** for desktop GPUs
2. **Simple, fast encoding** for real-time SDF generation
3. **1 byte per entry** 单通道格式（比 BC4 更简单）
4. **Better Z-axis compression** for volumetric data

### 8.3 可能的解决方案

```
潜在技术方向：
┌─────────────────────────────────────────────────────────┐
│                                                          │
│ 1. 扩展 BC 格式:                                         │
│    • BC7 的单通道变体                                    │
│    • 支持 3D blocks                                      │
│                                                          │
│ 2. GPU 厂商支持 ASTC:                                    │
│    • Desktop GPUs 理论上可以                             │
│    • 但商业动力不足（移动优先的格式）                     │
│                                                          │
│ 3. Custom software compression:                          │
│    • 类似 RTM (Ray Tracing Minification)                │
│    • 配合 compute shader 解码                            │
│                                                          │
│ 4. Neural compression:                                   │
│    • 如 NVIDIA 的 Neural Texture Compression            │
│    • SDF-specific 网络                                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 9. 参考资源

- [ASTC Texture Compression - Khronos Wiki](https://www.khronos.org/opengl/wiki/ASTC_Texture_Compression)
- [BCn Block Compression - Microsoft Docs](https://docs.microsoft.com/en-us/windows/win32/direct3d11/texture-block-compression)
- [OpenVDB Official Site](https://www.openvdb.org/)
- [NanoVDB GitHub](https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb)
- [NVIDIA TMU Architecture](https://developer.nvidia.com/content/gpu-texture-filtering)
- [AMD RDNA Whitepaper](https://www.amd.com/system/files/documents/rdna-whitepaper.pdf)
- [Metal 3D Textures - Apple Developer](https://developer.apple.com/documentation/metal/mtltexturetype/type3d)
- [SDF Ray Marching - Inigo Quilez](https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm)

---

这个讨论深刻反映了 **GPU 纹理压缩标准滞后于现代渲染需求** 的现状，特别是 SDF 这类新兴应用场景缺乏专门优化的硬件支持。