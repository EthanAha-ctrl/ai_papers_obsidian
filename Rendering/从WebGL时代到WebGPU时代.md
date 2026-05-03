这篇文章讲述了作者 **Hector Arellano (Hat)** 长达13年的流体模拟探索之旅——从WebGL时代的各种"hack"到WebGPU时代的优雅实现。让我深入剖析这篇技术文章的核心内容。

---

## 📖 一、文章背景与叙事结构

这是一篇**技术散文式的深度教程**，作者以个人经历为线索，串联起多个重要的图形学技术概念：

| 时间节点 | 技术阶段 | 关键进展 |
|---------|---------|---------|
| 13年前 | WebGL初生 | 看到 demo scene 作品，萌生想法 |
| 10年前 | WebGL hack | 实现 SPH + Marching Cubes（但"ugly"） |
| 7年前 | 混合光线追踪 | 加入反射/折射/焦散，但性能不达标 |
| 现在 | WebGPU | 终于能"正确"实现所有特性 |

---

## 🔬 二、核心技术深度解析

### 2.1 流体模拟算法演进

#### 2.1.1 Smoothed Particle Hydrodynamics (SPH)

SPH 是一种**拉格朗日视角**的流体模拟方法，将流体离散化为粒子群。

**核心思想**：用核函数对粒子属性进行平滑插值。

**密度计算公式**：
$$\rho_i = \sum_j m_j W(\mathbf{r}_i - \mathbf{r}_j, h)$$

其中：
- $\rho_i$ = 粒子 $i$ 的密度
- $m_j$ = 粒子 $j$ 的质量
- $\mathbf{r}_i, \mathbf{r}_j$ = 粒子位置向量
- $h$ = 光滑长度（smoothing length，定义影响半径）
- $W$ = 核函数（kernel function），通常使用 Poly6 或 Spiky kernel

**压力计算**（基于理想气体状态方程）：
$$p_i = k(\rho_i - \rho_0)$$

其中：
- $p_i$ = 压力
- $k$ = 气体常数（刚度系数）
- $\rho_0$ = 静止密度

**压力力**：
$$\mathbf{f}_i^{pressure} = -\sum_j m_j \frac{p_i + p_j}{2\rho_j} \nabla W(\mathbf{r}_i - \mathbf{r}_j, h)$$

**粘性力**：
$$\mathbf{f}_i^{viscosity} = \mu \sum_j m_j \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_j} \nabla^2 W(\mathbf{r}_i - \mathbf{r}_j, h)$$

其中：
- $\mu$ = 粘性系数
- $\mathbf{v}_i, \mathbf{v}_j$ = 速度向量

**作者提到的 Flocking 类比**非常有启发意义：

| Flocking 行为 | SPH 对应项 | 物理意义 |
|--------------|-----------|---------|
| Separation（排斥） | 压力力 | 高密度区产生排斥 |
| Alignment（对齐） | 粘性力 | 速度趋于一致 |
| Cohesion（聚合） | 表面张力 | 保持流体连续性 |

**复杂度问题**：SPH 的朴素实现是 $O(n^2)$，因为每个粒子需要与所有其他粒子交互。作者通过**空间哈希网格加速结构**将其降为 $O(k \cdot n)$，其中 $k$ 是邻域粒子数量（最多108个 = 27个voxel × 4粒子/voxel）。

---

#### 2.1.2 Position Based Dynamics (PBD) / Position Based Fluids (PBF)

作者最终从 SPH 转向了 **Position Based Fluids**，这是 Müller 等人在 SIGGRAPH 2013 提出的方法。

**核心思想**：不通过力计算加速度，而是直接约束位置，使求解更稳定。

**PBF 的约束函数**：
$$C_i(\mathbf{p}_1, ..., \mathbf{p}_n) = \frac{\rho_i}{\rho_0} - 1$$

当约束满足时，密度等于静止密度。

**位置修正公式**：
$$\Delta \mathbf{p}_i = \sum_j \lambda_i + \lambda_j \nabla W(\mathbf{r}_i - \mathbf{r}_j, h)$$

其中 $\lambda$ 是拉格朗日乘子：
$$\lambda_i = -\frac{C_i}{\sum_k |\nabla_k C_i|^2 + \epsilon}$$

**PBF 相比 SPH 的优势**：
1. **无条件稳定**：显式积分的 SPH 需要极小的时间步长
2. **参数直观**：使用无量纲参数而非物理参数
3. **易于添加约束**：碰撞、形状匹配等可统一处理

**作者的简化方案**：由于还需要运行 Marching Cubes，GPU 计算预算有限，作者简化为：
- 重力作为主要力
- **Curl Noise** 提供流体感（无散度的噪声场）
- 鼠标交互力（强排斥）
- 粒子碰撞约束

**Curl Noise** 的数学原理：
$$\mathbf{v} = \nabla \times \psi$$

其中 $\psi$ 是一个向量势场。由于旋度的散度恒为零（$\nabla \cdot (\nabla \times \psi) = 0$），这保证了速度场是不可压缩的，非常适合流体模拟。

参考：[Curl Noise for Procedural Fluid Flow - Bridson et al.](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf)

---

### 2.2 Marching Cubes 算法详解

Marching Cubes 是将隐式表面（potential field）转换为显式网格的经典算法。

#### 2.2.1 算法原理

**核心思想**：在一个体素网格中，每个体素有8个顶点，根据每个顶点的势能值与阈值的比较，确定该体素内部的三角形配置。

**势能场生成**：
作者使用 **3D Blur** 而非 Jump Flood Algorithm：

$$\phi(\mathbf{x}) = \sum_i W(|\mathbf{x} - \mathbf{x}_i|)$$

通过高斯模糊或 box blur 平滑粒子场，生成连续的势能场。

**为什么选择 Blur 而非 Jump Flood？**

| 方法 | 优点 | 缺点 |
|-----|------|------|
| Jump Flood | 精确的距离场 | 结果是球体群，不够平滑 |
| 3D Blur | 平滑、可控的表面 | 近似距离场 |

作者指出 Jump Flood "too good"——它精确计算每个粒子的距离，导致表面呈现分离的球体状，缺乏流体的融合感。而 Blur 通过扩散平滑了高频细节，更适合流体表现。

#### 2.2.2 GPU 实现步骤

作者的实现分为多个 Compute Shader 阶段：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Marching Cubes GPU Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Particles] ──► [3D Texture] ──► [Blurred Potential Field]    │
│       │              │                    │                     │
│       │              ▼                    ▼                     │
│       │         Blur3D.wgsl         (3 dispatches for X,Y,Z)   │
│       │                                     │                   │
│       ▼                                     ▼                   │
│  PBF_applyForces.wgsl              MarchCase.wgsl              │
│       │                              (atomics to compact voxels)│
│       ▼                                     │                   │
│  PBF_calculateDisplacements.wgsl           ▼                   │
│       │                           EncodeBuffer.wgsl            │
│       ▼                          (setup indirect dispatch)      │
│  PBF_integrateVelocity.wgsl                │                   │
│       │                                    ▼                   │
│       └──────────────────────►  GenerateTriangles.wgsl         │
│                                  (indirect dispatch)            │
│                                         │                       │
│                                         ▼                       │
│                              [Storage Buffer: Vertices+Normals] │
│                                         │                       │
│                                         ▼                       │
│                               RenderMC.wgsl                     │
│                              (indirect draw)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**关键创新点：Indirect Dispatch/Draw**

WebGPU 支持 `dispatchIndirect` 和 `drawIndirect`，允许 GPU 动态决定工作量，而无需 CPU 回读。

```wgsl
// EncodeBuffer.wgsl 示例逻辑
// 从 atomic counter 获取有效 voxel 数量
let voxelCount = atomicLoad(&voxelCounter);
// 写入 indirect dispatch 参数
indirectDispatch[0] = voxelCount;  // workgroup count
indirectDispatch[1] = 1;
indirectDispatch[2] = 1;
```

**Marching Cubes 查找表**：

共有 $2^8 = 256$ 种顶点配置，每种配置对应 0-5 个三角形。经典的查找表包含：
- `edgeTable[256]`: 12-bit 标记哪些边与等值面相交
- `triTable[256][16]`: 定义三角形的顶点索引

参考：[Marching Cubes - Lorensen & Cline (SIGGRAPH 1987)](https://dl.acm.org/doi/10.1145/37402.37422)

---

### 2.3 渲染技术

#### 2.3.1 Voxel Cone Tracing (VCT) for Ambient Occlusion

VCT 是一种实时全局光照技术，作者用于计算环境光遮蔽。

**核心原理**：

1. **体素化场景**：将三角形场景转换为稀疏体素表示
2. **构建 Mipmap 层级**：用于锥形追踪时的预过滤查询
3. **锥形追踪**：从着色点向半球方向发射锥形射线，累积遮蔽

**作者的简化实现**：

由于 Marching Cubes 本身就基于体素网格，VCT 的体素化步骤已经"免费"完成。只需在 `MarchCase.wgsl` 中额外标记：
- 有三角形的 voxel → 1（遮挡）
- 无三角形 → 0（空）
- 地面以下 → 0.5（模拟地面 AO）

**锥形追踪公式**：
$$AO = 1 - \frac{1}{N} \sum_{i=1}^{N} \text{traceCone}(\mathbf{p}, \mathbf{d}_i)$$

其中 $\mathbf{d}_i$ 是第 $i$ 个锥形方向，通常使用 6-8 个锥覆盖半球。

参考：[Real-Time Global Illumination using Voxel Cone Tracing - Crassin et al.](https://research.nvidia.com/sites/default/files/publications/GIVoxels-pg2011-authors.pdf)

---

#### 2.3.2 Subsurface Scattering (SSS)

次表面散射是光线穿透半透明材质后在内部散射的效果。

**作者的创新方法**：

传统 SSS 需要预计算的厚度贴图，但对于动态流体，这是不可能的。作者利用 **势能场作为距离场** 来实时计算厚度。

**Inigo Quilez 的 AO 技术改编**：

原始 AO 方法是向外部发射射线检测遮挡；作者改为向内部发射射线检测"穿透"。

```glsl
// 伪代码：使用距离场计算厚度
float thickness(vec3 p, vec3 lightDir) {
    float t = 0.0;
    float thickness = 0.0;
    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 pos = p + lightDir * t;
        float d = distanceField(pos);
        if(d < 0.0) {  // inside the surface
            thickness += abs(d);
        }
        t += STEP_SIZE;
    }
    return thickness;
}
```

**次表面散射的 BSSRDF 模型**：

$$S(x_i, \vec{\omega}_i; x_o, \vec{\omega}_o) = \frac{1}{\pi} R_d(||x_i - x_o||)$$

其中 $R_d$ 是扩散剖面，通常使用归一化扩散模型：

$$R_d(r) = \frac{e^{-\sigma r} + e^{-\sigma r/3}}{8\pi \sigma r}$$

参考：[Subsurface Scattering - Jensen et al.](https://graphics.stanford.edu/papers/bssrdf/)

---

### 2.4 后处理与合成

这部分由 **Paul-Guilhem Repaux** 指导完成，展示了技术 demo 与产品级视觉的差距。

#### 2.4.1 模糊反射

**问题**：直接渲染的地面反射过于锐利，不符合物理。

**解决方案**：根据几何体到地面的距离调整模糊强度。

$$\text{blurRadius} = k \cdot \text{height}$$

**技术细节**：需要预计算偏移贴图，存储最近几何体的距离，用于填充反射区域外的模糊信息。

#### 2.4.2 Bloom 效果

**原理**：提取高亮区域，模糊后叠加回原图。

**与 SSS 的协同**：
- 几何体较薄处 → SSS 更强 → 更亮
- Bloom 放大这一效果，形成"光晕"

**标准 Bloom Pipeline**：
```
1. Luminance Threshold → Bright Pass
2. Downsample (multiple levels)
3. Gaussian Blur (separable)
4. Upsample + Accumulate
5. Blend with original
```

参考：[Next Generation Post Processing in Call of Duty: Advanced Warfare](https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/)

---

## 🏗️ 三、WebGPU 技术要点

作者反复强调 WebGPU 相比 WebGL 的革命性优势：

### 3.1 关键特性对比

| 特性 | WebGL | WebGPU | 用途 |
|-----|-------|--------|------|
| Compute Shaders | ❌ (需 GPGPU hack) | ✅ | 物理模拟、网格生成 |
| Storage Buffers | ❌ | ✅ | 任意读写数据 |
| Atomics | ❌ | ✅ | 并行写入、流压缩 |
| Indirect Dispatch | ❌ | ✅ | GPU 驱动工作量 |
| 3D Textures | 有限支持 | ✅ | 体素数据 |
| Timestamp Queries | ❌ | ✅ | 性能分析 |

### 3.2 作者的 WebGL Hacks

在 WebGPU 出现前，作者使用了各种"黑科技"：

```
┌────────────────────────────────────────────────────────────┐
│              WebGL Hacks vs WebGPU Native                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Atomics → Separate RGBA channels, multiple draw calls    │
│                                                            │
│  Storage Buffers → Textures as data storage               │
│                                                            │
│  3D Textures → 2D texture arrays (layers)                 │
│                                                            │
│  Indirect Draw → "Expected" draw call counts              │
│                                                            │
│  Compute Shaders → Vertex shader GPGPU                    │
│                                                            │
│  (No random write → Full render-to-texture passes)        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🎯 四、第一性原理视角

### 4.1 为什么流体模拟这么难？

从第一性原理出发，流体模拟涉及：

**Navier-Stokes 方程**：
$$\rho \left( \frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla \mathbf{v} \right) = -\nabla p + \mu \nabla^2 \mathbf{v} + \mathbf{f}$$

$$\nabla \cdot \mathbf{v} = 0 \quad \text{(incompressibility)}$$

**数值方法的挑战**：
1. **对流项** $\mathbf{v} \cdot \nabla \mathbf{v}$：非线性，导致数值不稳定
2. **压力-速度耦合**：需要解 Poisson 方程
3. **边界条件**：自由表面、固壁、界面张力

SPH 和 PBF 都是通过**拉格朗日方法**规避了网格带来的对流问题，但引入了**邻域搜索**和**密度/压力求解**的复杂性。

### 4.2 为什么选择 Position Based？

**Position Based 的哲学**：

传统方法：力 → 加速度 → 速度 → 位置（显式积分，条件稳定）

Position Based：力/约束 → 直接修正位置（无条件稳定）

这类似于**投影方法**——在每一步将状态"投影"到满足约束的流形上。

---

## 📊 五、性能考量

作者的 demo 性能数据：

| 设备 | 帧率 | 备注 |
|-----|------|------|
| MacBook Pro M3 Max | 120 fps | 高端设备 |
| MacBook Pro M1 Pro | 60 fps | 中高端 |
| 其他"不错的机器" | ~50 fps | - |
| MacBook Air | ❌ | "dreams fade quickly" |

**GPU 时间预算分配**（估算）：

```
Total Frame Budget (16.7ms @ 60fps)
│
├── PBD Simulation (~4-6ms)
│   ├── Apply Forces
│   ├── Calculate Displacements (collision)
│   └── Integrate Velocity
│
├── Marching Cubes (~3-5ms)
│   ├── 3D Blur (potential)
│   ├── March Case (voxel marking)
│   └── Generate Triangles
│
├── Rendering (~3-4ms)
│   ├── VCT Mipmap
│   ├── Main Pass (SSS + AO)
│   └── Reflection Pass
│
└── Composition (~2-3ms)
    ├── Blur (reflection)
    ├── Bloom
    └── Color Correction
```

---

## 🔗 六、参考资源汇总

### 学术论文

1. **SPH**: [Smoothed Particle Hydrodynamics - Monaghan (1992)](https://www.annualreviews.org/doi/abs/10.1146/annurev.aa.30.090192.001241)

2. **PBF**: [Position Based Fluids - Macklin & Müller (SIGGRAPH 2013)](https://mmacklin.com/pbf_sig_preprint.pdf)

3. **Marching Cubes**: [Marching Cubes: A High Resolution 3D Surface Construction Algorithm - Lorensen & Cline (1987)](https://dl.acm.org/doi/10.1145/37402.37422)

4. **VCT**: [Real-Time Global Illumination using Voxel Cone Tracing - Crassin et al. (2011)](https://research.nvidia.com/sites/default/files/publications/GIVoxels-pg2011-authors.pdf)

5. **SSS**: [A Practical Model for Subsurface Light Transport - Jensen et al. (2001)](https://graphics.stanford.edu/papers/bssrdf/)

6. **Curl Noise**: [Curl-Noise for Procedural Fluid Flow - Bridson et al. (2007)](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf)

### WebGPU 资源

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WebGPU Samples](https://webgpu.github.io/webgpu-samples/)
- [Learn WebGPU](https://webgpu.rocks/)

### 作者作品

- [Demo 源码](https://github.com/...)
- [Felix 的 SPH Letters](https://fluid.felixmartinez.dev/)
- [WebGPU Ocean (Point Splatting)](https://webgpu-ocean.netlify.app)

---

## 💡 七、关键 Takeaways

1. **技术需要时间**：13年的探索，从 naive 想法到可运行的 demo
2. **API 的限制即创作的边界**：WebGL 的限制迫使作者创造各种 hack，但最终限制了可能性
3. **仿真 ≠ 渲染**：流体模拟只是第一步，高质量的渲染需要更多技术
4. **Composition is everything**：技术正确 ≠ 视觉好看
5. **Indirect dispatch/draw 是 GPU-driven rendering 的关键**：避免 CPU-GPU 同步
6. **简化是艺术**：从 SPH 到简化 PBD，性能与效果的平衡

这篇文章不仅是一个技术教程，更是一份**工程实践的地图**——它展示了从想法到实现的完整路径，以及沿途的坑和桥。对于想要深入 WebGPU 图形编程的开发者，这是一份不可多得的指南。