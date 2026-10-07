 **PIC/FLIP**（Particle-in-Cell/Fluid Implicit Particle）流体模拟器的 **meshing 和 rendering pipeline**。
 作者两年前在使用 **semi-Lagrangian** 流体模拟器时遇到了严重问题：

### 1.1 旧方法的核心问题

```
问题层级结构：
├── Meshing 问题
│   ├── 使用 standard marching cubes
│   ├── 直接从 simulation level set 构建 mesh
│   └── 大量 flickering（闪烁）现象
└── Liquid-solid interface 问题
    ├── 液体无法正确适应固体边界
    └── interface 无法精确对齐
```

### 1.2 渲染问题

旧渲染方法使用 **Vray** 的 **irradiance map + light cache** 方案：
- **Irradiance map**：预计算间接光照缓存
- **Light cache**：基于光子的近似全局光照方案
- **致命缺陷**：对于高运动、大量折射流体的场景，时间上缺乏连贯性

## 二、核心技术架构

### 2.1 PIC/FLIP 的解耦特性

这是新方法的关键洞察：

```
Grid resolution (128x64x64)
    ↕
Projection step（投影步骤）
    ↕
Particle count（粒子数量，决定实际分辨率）
    ↕
Mesh resolution（可独立于 grid resolution）
```

**数学表达**：

在 PIC/FLIP 方法中，流体由 Lagrangian particles（拉格朗日粒子）表示：

$$\mathbf{x}_p^{n+1} = \mathbf{x}_p^n + \Delta t \cdot \mathbf{u}_p^n$$

其中：
- $\mathbf{x}_p^n$：粒子 $p$ 在时间步 $n$ 的位置
- $\Delta t$：时间步长
- $\mathbf{u}_p^n$：粒子速度

Grid 仅用于 **pressure projection（压力投影）**：

$$\nabla \cdot \mathbf{u}^{n+1} = 0$$

因此，**mesh 分辨率可以完全解耦**。

### 2.2 OpenVDB Toolkit 架构

OpenVDB 提供了三个关键组件：

| 组件 | 功能 | 技术细节 |
|------|------|----------|
| **Level set construction** | 从 particles 构建 level set | 每个粒子被视为半径 $r_p$ 的球体：$\phi(\mathbf{x}) = \min_p \|\mathbf{x} - \mathbf{x}_p\| - r_p$ |
| **Adaptive meshing** | 自适应分辨率 meshing | 基于曲率和几何特征调整三角面密度 |
| **Level set operators** | 艺术化调优工具 | Dilation（膨胀）、Erosion（腐蚀）、Smoothing（平滑） |

**Level set 数学定义**：

Level set 函数 $\phi(\mathbf{x})$ 定义为：
$$
\phi(\mathbf{x}) = 
\begin{cases}
< 0 & \text{如果 } \mathbf{x} \text{ 在流体内部} \\
= 0 & \text{如果 } \mathbf{x} \text{ 在流体表面} \\
> 0 & \text{如果 } \mathbf{x} \text{ 在流体外部}
\end{cases}
$$

## 三、关键技术问题与解决方案

### 3.1 Adaptive Meshing 的时间连贯性问题

**问题描述**：

OpenVDB 的 adaptive meshing 算法优化每帧的 adaptivity：

$$\mathcal{A}_n = \text{Adaptivity}(\text{Mesh}_n)$$

这导致：
$$\text{Resolution}(\text{Mesh}_n) \neq \text{Resolution}(\text{Mesh}_{n+1})$$

**影响链条**：

```
Per-frame adaptive optimization
    ↓
Mesh resolution varies temporally
    ↓
Normal reconstruction becomes unstable
    ↓
Flickering in final render
```

**实验观察**：

| Adaptivity 值 | Poly count | Detail loss | Flickering 风险 |
|--------------|-----------|-------------|-----------------|
| 0.0（无自适应） | 高 | 无 | 低 |
| 0.5 | 中等 | 明显 | 高 |

**解决方案**：

对于小规模模拟：**禁用 adaptivity**
对于大规模模拟：**启用 adaptivity**（因为渲染尺度使 normal 问题变得不那么明显）

### 3.2 Liquid-Solid Interface 对齐问题

#### 3.2.1 初始方法：Level set 差分

**算法思想**：

$$\phi_{\text{liquid}}^{\text{final}} = (\phi_{\text{liquid}} \oplus r) - \phi_{\text{solid}}$$

其中：
- $\oplus$：Dilation operator（膨胀算子）
- $r$：dilation radius
- $-$：Difference operator（差分算子）

**实现流程**：

```
1. Dilate liquid level set
2. Construct high-res solid level set
3. Apply difference operator
4. Mesh the result
```

**致命缺陷**：

1. **Resolution explosion**：
   - 为了捕获固体边界细节（如尖角），需要极高分辨率的 solid level set
   - OpenVDB difference operator 要求输出 resolution 匹配最高输入 resolution
   - 结果：liquid level set 也被强制极高分辨率

2. **Performance catastrophe**：
   ```
   Complexity ∝ N_grid × N_operations
   ```
   其中 $N_{\text{grid}} = 1024 \times 512 \times 512$ 时，操作数量爆炸。

#### 3.2.2 最终方法：Mesh-based 对齐

**核心洞察**：Mesh 比其对应的 level set 表示的数据量小得多。

**算法流程**：

```
For each vertex v in liquid mesh:
    1. Find closest point p on solid boundary
       p = argmin_{x ∈ ∂Ω_solid} ||v - x||
    
    2. If ||v - p|| < ε:
           Move v to p
```

**技术细节**：

**Nearest point search 方法**：

1. **Stochastic approach**（随机方法）：
   ```python
   def find_closest_stochastic(vertex, solid_mesh, num_samples=100):
       best_dist = ∞
       best_point = None
       for _ in range(num_samples):
           sample = random_point_on_solid(solid_mesh)
           dist = ||vertex - sample||
           if dist < best_dist:
               best_dist = dist
               best_point = sample
       return best_point
   ```

2. **Level set guided approach**（level set 引导方法）：
   ```python
   def find_closest_guided(vertex, solid_level_set):
       # Use gradient as initial search direction
       ∇φ = gradient(solid_level_set, vertex)
       initial_dir = -∇φ / ||∇φ||  # Point towards surface
       
       # Raycast in gradient direction
       ray = Ray(origin=vertex, direction=initial_dir)
       intersection = raycast_to_solid(ray)
       return intersection.point
   ```

**参数说明**：

- $\varepsilon$：Tolerance threshold（容差阈值），通常为 grid cell size 的分数
- $v$：Liquid mesh vertex position（液体网格顶点位置）
- $\partial\Omega_{\text{solid}}$：Solid boundary surface（固体边界表面）

**优势分析**：

| 指标 | Level set 方法 | Mesh-based 方法 |
|------|----------------|-----------------|
| Memory usage | $O(N^3)$ | $O(N^2)$ |
| Computational cost | 极高 | 中等 |
| Implementation complexity | 高 | 低 |
| Flexibility | 低 | 高（可直接在 Houdini 中原型化） |

## 四、渲染 Pipeline

### 4.1 数据流程

```
Simulation (Particles)
    ↓
Level set construction (OpenVDB)
    ↓
Meshing (OpenVDB)
    ↓
Liquid-solid interface cleanup (Custom)
    ↓
PLY export
    ↓
VRMesh conversion (Vray ply2mesh)
    ↓
Rendering (Vray brute force pathtracing)
```

### 4.2 Brute Force Pathtracing

**为什么放弃 irradiance cache**：

Irradiance cache 的问题：

$$I_c(\mathbf{x}) \approx \int_{\Omega} L(\mathbf{x}, \omega) f_r(\mathbf{x}, \omega, \omega_o) \cos\theta d\omega$$

当场景快速变化时，cache points 无法保持时间连贯性。

**Brute force pathtracing** 优势：

```cpp
for each pixel:
    for each sample:
        ray = camera_ray(pixel, sample)
        Li = trace_ray(ray, depth)
        accumulate(Li)
```

**代价**：
- 计算：$O(N_{\text{samples}} \times N_{\text{rays}})$
- 但结果无时间相关 artifacts

### 4.3 VRMesh 格式

**VRMesh** 是 Vray 的优化网格格式，提供：
- **Spatial acceleration structure**（空间加速结构）
- **LOD（Level of Detail）** support
- **Efficient streaming** for large meshes

## 五、技术深度解析

### 5.1 OpenVDB Adaptive Meshing 算法

**核心原理**：

OpenVDB 使用 **Octree-based adaptive voxelization**：

```
Level 0: 1 voxel covering entire domain
Level 1: 8 voxels (2³)
Level 2: 64 voxels (4³)
...
Level N: (2^N)³ voxels
```

**Adaptivity metric**：

$$\mathcal{A}(\text{cell}) = w_1 \cdot \kappa + w_2 \cdot \|\nabla \phi\| + w_3 \cdot \sigma$$

其中：
- $\kappa$：曲率（curvature）
- $\|\nabla \phi\|$：Level set gradient magnitude
- $\sigma$：Noise metric（噪声度量）
- $w_i$：权重系数

**Marching cubes on adaptive grid**：

对于每个活跃 cell，应用 standard marching cubes lookup table（256 种配置）：

```
cube_config = (φ[v0] < 0) << 0 | 
              (φ[v1] < 0) << 1 |
              ... |
              (φ[v7] < 0) << 7

triangles = MC_TABLE[cube_config]
```

### 5.2 Partio Library 集成

**Partio** 由 Walt Disney Animation Studios 开发，提供：

```python
import partio

# Write particles
p = partio.create()
p.addAttribute("position", partio.VECTOR, 3)
p.addAttribute("velocity", partio.VECTOR, 3)

for i in range(num_particles):
    p.addParticle()
    p.set("position", i, positions[i])
    p.set("velocity", i, velocities[i])

partio.write("particles.bgeo", p)
```

**支持的格式**：
- **PDB**（Particle Database）
- **BGEO**（Houdini binary）
- **PTC**（RenderMan point cloud）
- **OBJ**（Wavefront）

## 六、实验结果与参数配置

### 6.1 Simulation 参数

```
Simulation grid: 128 x 64 x 64
Particle count: ~1,000,000
Level set resolution: 1024 x 512 x 512
Mesh adaptivity: 0.0 (disabled for stability)
```

### 6.2 Visual 改进

**Before cleanup**：
- Liquid-solid interface 间隙：~1-2 grid cells
- Flickering：明显
- Poly count：不稳定

**After cleanup**：
- Interface alignment：< 0.1 grid cells
- Flickering：最小化
- Poly count：稳定

## 七、相关技术资源

**核心库**：
- OpenVDB: https://www.openvdb.org/
- Partio: https://github.com/wdas/partio

**学术论文**：
- Zhu and Bridson, "Animating Sand as a Fluid", SIGGRAPH 2005
- M. Nielsen et al., "Generalizing the Level Set Method", SIGGRAPH 2004

**教程**：
- SIGGRAPH 2013 OpenVDB Course Notes
- DreamWorks OpenVDB Presentation

这个 blog 展示了如何通过合理的架构设计和工程实践，将理论算法转化为可用的 production pipeline，特别是在处理时间连贯性和性能权衡方面的决策很有参考价值。