# Dreamcast Microsoft Direct3D 性能优化深度解析

## 一、文章概览与历史背景

这篇文章由 **Kalisto Entertainment** 的 **Sebastian Wloch** 于 **1999年3月**撰写，是关于在 **Sega Dreamcast** 游戏机上优化基于 **Windows CE** 和 **Microsoft Direct3D** 的游戏性能的经典技术文档。

**历史背景**：Dreamcast 是 Sega 于1998年推出的第六代游戏机，它采用了当时先进的 **PowerVR Series 2** GPU，其核心特征是 **Tile-Based Deferred Rendering (TBDR)** 架构。这与当时主流的 **Immediate Mode Rendering** 架构完全不同。

文章的核心发现：**PC上的Direct3D优化技巧不能直接应用于Dreamcast**，因为Direct3D被专门针对Dreamcast硬件进行了优化。

---

## 二、Dreamcast 3D硬件架构详解

### 2.1 Tile-Based Rendering 架构

Dreamcast 的 **PowerVR2 (CLX2)** 芯片采用了一种革命性的渲染方式：

```
传统IMR渲染流程:
Triangles → Vertex Processing → Setup → Rasterization (scanline) → Pixel Processing → Framebuffer
                           ↓
                    每个像素可能被多次渲染（当多个三角形重叠时）

Dreamcast TBDR渲染流程:
Triangles → Vertex Processing → Setup → Geometry Buffer存储 →
                           ↓
Tile Processing (32×32 pixels):
  1. 对每个Tile: 确定哪些三角形与Tile相交
  2. 深度排序（Z-sort，但不是传统Z-buffer）
  3. 只渲染最接近相机的三角形像素
  4. 输出到Framebuffer
```

### 2.2 关键技术公式

#### 2.2.1 Tile 分割

屏幕被划分为规则tiles：

```
Let ScreenWidth = 640, ScreenHeight = 480 (典型Dreamcast分辨率)
TileSize = 32 × 32 pixels

Number of Tiles in X dimension:
  NTiles_X = ⌈ScreenWidth / TileSize⌉ = ⌈640 / 32⌉ = 20

Number of Tiles in Y dimension:
  NTiles_Y = ⌈ScreenHeight / TileSize⌉ = ⌈480 / 32⌉ = 15

Total Tiles = NTiles_X × NTiles_Y = 20 × 15 = 300
```

#### 2.2.2 像素覆盖测试

对于每个tile中的每个pixel (x, y)，硬件需要确定哪个三角形是最接近相机的：

```
For each tile T at position (i, j):
  For each pixel p = (x, y) in T:
    Let S_T = {triangles T₁, T₂, ..., Tₙ} that intersect T
    
    For each triangle Tₖ ∈ S_T:
      Compute depth value zₖ at pixel p through interpolation:
        zₖ = (1 - u - v)·z₀ + u·z₁ + v·z₂
        其中:
          (z₀, z₁, z₂) 是三角形三个顶点的z值
          (u, v) 是像素p在三角形内的重心坐标
          满足: u ≥ 0, v ≥ 0, u + v ≤ 1
    
    Find triangle T* with minimum z*:
      z* = min{zₖ | Tₖ ∈ S_T}
    
    Render pixel p using T* (only once!)
```

### 2.3 架构优势分析

#### 2.3.1 Fill Rate 独立性

**公式说明**：
```
传统架构: 
  Total Pixel Operations = Σ N_triangles × Average_Pixels_Per_Triangle
  当多个三角形重叠时，每个像素会被渲染多次

Dreamcast架构:
  Total Pixel Operations = ScreenWidth × ScreenHeight
  每个像素只渲染一次（不透明物体）
```

这意味着：
- **Overdraw 不再是瓶颈**
- 对于高密度三角形场景，Dreamcast比传统GPU有显著优势
- 不需要传统的 Z-buffer（节省显存带宽）

#### 2.3.2 无需View Frustum Clipping

```
传统Clipping测试:
  For each vertex v = (x, y, z, w):
    Test if -w ≤ x ≤ w and -w ≤ y ≤ w and 0 ≤ z ≤ w
    If outside: Clip triangle to new vertices

Dreamcast:
  Triangles stored as-is in geometry buffer
  During tile processing, triangles are implicitly clipped
  No explicit clipping calculations needed
```

#### 2.3.3 透明度排序

对于透明三角形：
```
Pass 1: Render all opaque triangles (single pass)
Pass 2: Sort transparent triangles by depth
Pass 3+: Render transparent triangles in back-to-front order

Hardware automatically performs per-pixel sorting:
  For each pixel, render transparent triangles from farthest to closest
```

---

## 三、几何剔除技术详解

### 3.1 三层剔除策略

文章提出了三层几何剔除金字塔：

```
                    ┌─────────────────────────┐
                    │  Application Level      │
                    │  (高级场景组织)          │
                    │  消除10-50%的三角形      │
                    └─────────────┬───────────┘
                                  │
                    ┌─────────────▼───────────┐
                    │  Backface Culling       │
                    │  背面剔除               │
                    │  消除10-50%的三角形      │
                    └─────────────┬───────────┘
                                  │
                    ┌─────────────▼───────────┐
                    │  View Frustum Culling   │
                    │  视锥剔除               │
                    │  消除大量三角形          │
                    └─────────────────────────┘
```

### 3.2 View Frustum Culling 视锥剔除

#### 3.2.1 数学基础

**View Frustum 定义**：

View frustum 是一个由6个平面定义的金字塔：
```
Frustum = { P ∈ ℝ³ | Left(P) ≥ 0 ∧ Right(P) ≥ 0 ∧ Top(P) ≥ 0 ∧ 
                      Bottom(P) ≥ 0 ∧ Near(P) ≥ 0 ∧ Far(P) ≥ 0 }

其中每个平面方程为: Plane(P) = n·P + d = 0
```

**测试三角形与Frustum的关系**：

对于三角形的每个顶点，计算其到6个平面的距离：
```
对于顶点 v = (x, y, z, 1):
  Distance_to_plane_i = nᵢ·v + dᵢ

分类规则:
  完全在Frustum内: 所有顶点对所有平面的距离 ≥ 0
  完全在Frustum外: 存在一个平面，所有顶点对该平面的距离 < 0
  与Frustum相交: 其他情况
```

#### 3.2.2 Bounding Sphere 优化

**Bounding Sphere 构建**：
```
对于包含N个顶点的对象:
  Center (C) = (1/N) × Σ vᵢ  (barycentrum/质心)
  Radius (R) = max{ ||vᵢ - C|| }  (到中心的最大距离)

其中 ||·|| 表示欧几里得范数:
  ||vᵢ - C|| = √[(xᵢ - Cₓ)² + (yᵢ - Cᵧ)² + (zᵢ - Cz)²]
```

**Frustum-Sphere 相交测试**：
```
对于每个平面 Pᵢ: nᵢ·P + dᵢ = 0

计算球心到平面的有向距离:
  distance = nᵢ·C + dᵢ
  = nₓ·Cₓ + nᵧ·Cᵧ + n_z·C_z + dᵢ

判定规则:
  If distance < -R:
    Sphere完全在平面外侧 → 剔除整个对象
  If distance > R:
    Sphere完全在平面内侧 → 继续测试下一个平面
  Else:
    Sphere与平面相交 → 可能部分可见，继续测试
```

**性能优势分析**：
```
原始方法: 每个三角形需要 6×3 = 18次平面测试
Bounding Sphere: 每个对象需要 6次平面测试

若对象包含1000个三角形:
  加速比 = (18 × 1000) / 6 = 3000倍
```

#### 3.2.3 层次化剔除 (Hierarchy)

对于大型对象，使用 **Octree** 或 **SEAD** (Spatially Enclosed Area Delimiter):

```
Octree结构:
  Level 0: Root Bounding Sphere (整个场景)
  Level 1: 8个子节点 (每个覆盖1/8空间)
  Level 2: 每个子节点继续细分
  ...
  Level N: 叶节点包含少量三角形

递归剔除算法:
  Function CullNode(node):
    If not TestFrustum(node.bounding_sphere):
      return CULLED  // 整个分支不可见
    Else If node.is_leaf:
      return VISIBLE  // 叶节点可见
    Else:
      children_visible = []
      For each child in node.children:
        result = CullNode(child)
        If result == VISIBLE:
          children_visible.append(child)
      return children_visible
```

**Octree 细分公式**：
```
对于边界框 [xmin, xmax] × [ymin, ymax] × [zmin, zmax]:
  
  子节点0: [xmin, mid] × [ymin, mid] × [zmin, mid]
  子节点1: [mid, xmax] × [ymin, mid] × [zmin, mid]
  子节点2: [xmin, mid] × [mid, ymax] × [zmin, mid]
  子节点3: [mid, xmax] × [mid, ymax] × [zmin, mid]
  子节点4: [xmin, mid] × [ymin, mid] × [mid, zmax]
  子节点5: [mid, xmax] × [ymin, mid] × [mid, zmax]
  子节点6: [xmin, mid] × [mid, ymax] × [mid, zmax]
  子节点7: [mid, xmax] × [mid, ymax] × [mid, zmax]

其中 mid = (min + max) / 2
```

### 3.3 Backface Culling 背面剔除

#### 3.3.1 数学原理

**背面判定条件**：
```
给定三角形的三个顶点 v₀, v₁, v₂ (按逆时针顺序):

计算两个边向量:
  e₁ = v₁ - v₀ = (x₁ - x₀, y₁ - y₀, z₁ - z₀)
  e₂ = v₂ - v₀ = (x₂ - x₀, y₂ - y₀, z₂ - z₀)

计算法向量 (叉积):
  N = e₁ × e₂ = | i   j   k |
                | e₁ₓ e₁ᵧ e₁z |
                | e₂ₓ e₂ᵧ e₂z |
  
  = (e₁ᵧ·e₂z - e₁z·e₂ᵧ,
     e₁z·e₂ₓ - e₁ₓ·e₂z,
     e₁ₓ·e₂ᵧ - e₁ᵧ·e₂ₓ)

法向量归一化:
  n̂ = N / ||N||

给定视角向量 V (从三角形到相机):
  V = CameraPosition - v₀

计算点积:
  dot = n̂ · V / ||V||

判定:
  If dot < 0:
    三角形背面 → 剔除
  Else:
    三角形正面 → 保留
```

#### 3.3.2 Triangle Strip 批量优化

对于共面或近似共面的三角形条带：
```
Strip包含N个三角形 {T₀, T₁, ..., Tₙ₋₁}

计算平均法向量:
  N_avg = (1/N) × Σ Nᵢ
  n̂_avg = N_avg / ||N_avg||

对于整个strip使用一次背面测试:
  dot = n̂_avg · V / ||V||

If |dot| < threshold:
  // strip边缘情况，需要更精确测试
  For each triangle in strip:
    Test individually
Else If dot < 0:
  // 整个strip背面
  Cull entire strip
Else:
  // 整个strip正面
  Render entire strip

性能提升:
  原始: N次三角形测试
  优化: 1次strip测试 (在共面情况下)
```

---

## 四、几何数据结构与传输优化

### 4.1 Triangle Lists vs Triangle Strips

#### 4.1.1 数据量对比

**Triangle List**：
```
每个三角形独立存储，不共享顶点:

对于N个三角形:
  顶点数量 = 3N
  
  数据传输量 = 3N × VertexSize
  其中 VertexSize = sizeof(x,y,z) + sizeof(nx,ny,nz) + 
                   sizeof(u,v) + sizeof(r,g,b) + ...

例如: VertexSize = 12 + 12 + 8 + 4 = 36 bytes
     N = 1000 triangles
     数据量 = 3000 × 36 = 108,000 bytes
```

**Triangle Strip**：
```
连续三角形共享顶点:

对于N个三角形:
  顶点数量 = N + 2 (第一个三角形3顶点，之后每三角形+1顶点)
  
  数据传输量 = (N + 2) × VertexSize

同样例子: N = 1000 triangles
           数据量 = 1002 × 36 = 36,072 bytes
           
节省比例 = (3N - (N+2)) / (3N) = (2N - 2) / (3N) ≈ 66.7%
```

#### 4.1.2 Strip 连接约束

```
两个相邻顶点 vᵢ 和 vⱼ 必须满足:

条件1: 几何相同
  vᵢ.xyz = vⱼ.xyz

条件2: 法向量相同
  vᵢ.nxny = vⱼ.nxny
  vᵢ.nz = vⱼ.nz

条件3: 纹理坐标相同
  vᵢ.uv = vⱼ.uv

条件4: 颜色相同
  vᵢ.rgb = vⱼ.rgb

若任何条件不满足 → Strip必须中断 → 新的Strip开始
```

#### 4.1.3 Strip 生成算法

```
贪婪算法 (Greedy Strip Generation):

Input: 三角形网格 M = {T₀, T₁, ..., Tₙ₋₁}
Output: Strip列表 S = {S₀, S₁, ..., Sₖ₋₁}

Function GenerateStrips(M):
  visited = {}  // 已访问的三角形集合
  strips = []
  
  While there exists unvisited triangle T:
    current_strip = [T]
    visited.add(T)
    last_triangle = T
    
    While exists unvisited triangle T' adjacent to last_triangle:
      查找与 last_triangle 共享两个顶点的未访问三角形 T'
      
      If strip continuity test passes (约束条件):
        Add T' to current_strip
        visited.add(T')
        last_triangle = T'
      Else:
        Break
    
    strips.append(current_strip)
  
  Return strips
```

### 4.2 DrawPrimitive vs DrawIndexedPrimitive

#### 4.2.1 DrawPrimitive

```
函数签名:
  HRESULT DrawPrimitive(
    D3DPRIMITIVETYPE dptPrimitiveType,
    DWORD dwVertexTypeDesc,
    LPVOID lpvVertices,
    DWORD dwVertexCount,
    DWORD dwFlags
  )

使用模式1: Triangle List
  DrawPrimitive(D3DPT_TRIANGLELIST, vertexType, vertices, 3*N, 0)
  
使用模式2: Triangle Strip (推荐)
  DrawPrimitive(D3DPT_TRIANGLESTRIP, vertexType, vertices, N+2, 0)

优势: 直接顶点访问，无需索引
劣势: 重复顶点占用更多内存和带宽
```

#### 4.2.2 DrawIndexedPrimitive

```
函数签名:
  HRESULT DrawIndexedPrimitive(
    D3DPRIMITIVETYPE dptPrimitiveType,
    DWORD dwVertexTypeDesc,
    LPVOID lpvVertices,
    DWORD dwVertexCount,
    LPWORD lpwIndices,
    DWORD dwIndexCount,
    DWORD dwFlags
  )

使用模式:
  Vertices: 包含所有唯一顶点
  Indices: 三角形顶点索引序列
  
示例:
  三角形: [v0, v1, v2], [v1, v3, v2], [v2, v3, v4]
  
  Vertices = [v0, v1, v2, v3, v4]  (5个唯一顶点)
  Indices  = [0, 1, 2, 1, 3, 2, 2, 3, 4]
  
  DrawIndexedPrimitive(D3DPT_TRIANGLELIST, type, Vertices, 5, 
                       Indices, 9, 0)

优势:
  1. 顶点共享，减少数据量
  2. Direct3D自动将indices转换为strips
  
自动Strip生成:
  分析indices序列，检测共享边的连续三角形
  若 [i, j, k] 和 [j, k, l] 连续 → 自动合并为strip
```

### 4.3 Vertex Types 深度解析

#### 4.3.1 D3D_VERTEX (最灵活)

```
typedef struct _D3DVERTEX {
    union {
        float x;
        float dvX;
    };
    union {
        float y;
        float dvY;
    };
    union {
        float z;
        float dvZ;
    };
    DWORD dwReserved;
    union {
        float nx;
        float dvNX;
    };
    union {
        float ny;
        float dvNY;
    };
    union {
        float nz;
        float dvNZ;
    };
    DWORD dwReserved;
    union {
        float tu;
        float dvTU;
    };
    union {
        float tv;
        float dvTV;
    };
} D3DVERTEX, *LPD3DVERTEX;

特点:
  - 未变换的顶点 (model space)
  - 未光照 (需要计算)
  - Direct3D负责变换和光照
  
变换流程:
  ModelSpace → WorldSpace (World矩阵)
    P_world = M_world × P_model
  
  WorldSpace → ViewSpace (View矩阵)
    P_view = M_view × P_world = M_view × M_world × P_model
  
  ViewSpace → ProjectionSpace (Projection矩阵)
    P_proj = M_proj × P_view = M_proj × M_view × M_world × P_model

光照计算:
  N_world = M_world_normal × N_model  (法向量变换)
  L = LightPosition - P_world          // 光照方向
  diffuse = max(0, N_world · L̂)       // Lambertian
  final_color = ambient + diffuse × material_color
```

#### 4.3.2 D3D_LVERTEX (预光照)

```
typedef struct _D3DLVERTEX {
    union {
        float x;
        float dvX;
    };
    union {
        float y;
        float dvY;
    };
    union {
        float z;
        float dvZ;
    };
    DWORD dwReserved;
    union {
        D3DCOLOR color;
        unsigned long dvColor;
    };
    union {
        D3DCOLOR specular;
        unsigned long dvSpecular;
    };
    union {
        float tu;
        float dvTU;
    };
    union {
        float tv;
        float dvTV;
    };
} D3DLVERTEX, *LPD3DLVERTEX;

特点:
  - 未变换的顶点 (model/world/view space)
  - 已光照 (color字段包含计算结果)
  - Direct3D只负责变换
  
光照公式 (应用层计算):
  对于定向光:
    N = (nx, ny, nz)  // 顶点法向量
    L = LightDirection  // 光方向
    diffuse = max(0, N · L)  // 点积
    
    color.r = material.r × (ambient.r + diffuse.r)
    color.g = material.g × (ambient.g + diffuse.g)
    color.b = material.b × (ambient.b + diffuse.b)
    color.a = material.a
  
  对于点光源:
    L = LightPosition - VertexPosition
    distance = ||L||
    L = L / distance  // 归一化
    attenuation = 1 / (kc + kl×distance + kq×distance²)
    diffuse = max(0, N · L) × attenuation

性能权衡:
  + 避免Direct3D光照计算
  - 需要CPU计算光照
  - 静态光照效果好
  - 动态物体光照需要每帧更新
```

#### 4.3.3 D3D_TLVERTEX (已变换)

```
typedef struct _D3DTLVERTEX {
    union {
        float sx;
        float dvSX;
    };
    union {
        float sy;
        float dvSY;
    };
    union {
        float sz;
        float dvSZ;
    };
    union {
        float rhw;
        float dvRHW;
    };
    union {
        D3DCOLOR color;
        unsigned long dvColor;
    };
    union {
        D3DCOLOR specular;
        unsigned long dvSpecular;
    };
    union {
        float tu;
        float dvTU;
    };
    union {
        float tv;
        float dvTV;
    };
} D3DTLVERTEX, *LPD3DTLVERTEX;

特点:
  - 已变换到屏幕空间
  - 已光照
  - Direct3D只负责光栅化

屏幕空间坐标:
  sx, sy: 屏幕像素坐标
  sz: 深度值 (0-1范围，用于Z-sort)
  rhw: 1/w (w是齐次坐标的w分量)
       用于透视正确的插值:
         value_at_pixel = (value/w) / (1/w) = value / w

适用场景:
  1. OSD (On-Screen Display): 2D UI元素
  2. 屏幕空间生成的几何体 (如粒子系统)
  3. Bezier patches: 曲面细分后直接在屏幕空间渲染

性能特点:
  + 完全跳过顶点变换和光照
  + 直接光栅化，最快
  - 需要应用处理所有变换
  - 适合静态或特殊几何体
```

### 4.4 DONOTCLIP 标志优化

```
标志定义:
  D3DDP_DONOTCLIP = 0x00000020L

传统clipping流程 (flag未设置):
  For each triangle:
    For each vertex:
      Check if vertex inside view frustum
      If outside: Clip to create new vertices
    Check if triangle near plane (z < epsilon)
    If too close: Clip

DONOTCLIP 流程 (flag设置):
  Skip all clipping calculations
  Hardware implicitly clips during tile rendering

优化策略:
  Function RenderObject(object):
    If object.intersects_near_plane:
      // 不能使用DONOTCLIP
      DrawPrimitive(..., 0)
    Else:
      // 安全使用DONOTCLIP
      DrawPrimitive(..., D3DDP_DONOTCLIP)

近平面相交测试:
  对于边界球:
    near_plane: z = znear (正数)
    If (center.z - radius) < znear:
      intersects = true
    Else:
      intersects = false

性能提升:
  避免复杂的平面相交计算
  减少顶点分裂和重新生成
```

### 4.5 内存对齐与缓存优化

#### 4.5.1 32字节对齐

```
对齐要求: 顶点数据必须32字节对齐

正确分配:
  vertices = aligned_alloc(32, num_vertices * vertex_size)
  
  验证对齐:
    assert((uintptr_t)vertices % 32 == 0)

错误分配:
  vertices = malloc(num_vertices * vertex_size)
  // malloc只保证4字节对齐
  // Direct3D需要复制到对齐缓冲区

对齐检查公式:
  is_aligned = (address & (alignment - 1)) == 0
  对于32字节: is_aligned = (address & 0x1F) == 0

对齐后的内存布局:
  Address: 0x...0  [vertex 0] (32 bytes)
           0x...20 [vertex 1] (32 bytes)
           0x...40 [vertex 2] (32 bytes)
           ...
           每个顶点在32字节边界开始
```

#### 4.5.2 缓存局部性

```
缓存行大小: Dreamcast = 32 bytes

良好布局 (按渲染顺序):
  Mesh {
    Vertex vertices[N]  // 连续存储
    {
      vertices[0] = {..., texture=0}
      vertices[1] = {..., texture=0}
      ...
      vertices[100] = {..., texture=0}  // 同一texture
      vertices[101] = {..., texture=1}
      ...
    }
  }
  
渲染时:
  渲染texture=0的顶点 → 缓存命中率高
  渲染texture=1的顶点 → 缓存命中率高

糟糕布局 (随机顺序):
  Mesh {
    Vertex vertices[N]
    {
      vertices[0] = {..., texture=0}
      vertices[1] = {..., texture=5}
      vertices[2] = {..., texture=2}
      vertices[3] = {..., texture=0}
      vertices[4] = {..., texture=3}
      ...
    }
  }
  
渲染时:
  频繁切换texture → 缓存miss → 性能下降

分组优化:
  Function OrganizeForRendering(mesh):
    For each texture T:
      group_T = [vertices with texture T]
      Sort groups by rendering order
      
    Return organized_mesh

缓存miss公式:
  CacheMissRate = MissCount / TotalAccess
  
优化目标: Minimize CacheMissRate
```

---

## 五、SH4 处理器 intrinsics 优化

### 5.1 SH4 浮点单元特性

**SH4 处理器架构**：
```
Hitachi SuperH SH-4 FPU 特性:
  - IEEE 754 单精度浮点运算
  - SIMD (Single Instruction Multiple Data) 指令
  - 硬件平方根倒数 (1/√x)
  - 四指令发射宽度

关键流水线阶段:
  Fetch → Decode → Execute → Memory → Writeback
  |       |        |        |        |
  1 cycle 1 cycle 1 cycle 1 cycle 1 cycle
```

### 5.2 Dot Product 内置函数

```
函数原型:
  float fipr(float v1[4], float v2[4]);

操作:
  result = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3]

汇编实现 (简化):
  fmov    fr0, @(v1, r0)    // 加载v1[0]
  fmov    fr1, @(v1+4, r0)  // 加载v1[1]
  fmov    fr2, @(v1+8, r0)  // 加载v1[2]
  fmov    fr3, @(v1+12, r0) // 加载v1[3]
  fmov    fr4, @(v2, r1)    // 加载v2[0]
  fmul    fr0, fr4          // v1[0]*v2[0]
  fmov    fr5, @(v2+4, r1)  // 加载v2[1]
  fmul    fr1, fr5          // v1[1]*v2[1]
  fmov    fr6, @(v2+8, r1)  // 加载v2[2]
  fmul    fr2, fr6          // v1[2]*v2[2]
  fmov    fr7, @(v2+12, r1) // 加载v2[3]
  fmul    fr3, fr7          // v1[3]*v2[3]
  fadd    fr0, fr1          // 部分和
  fadd    fr2, fr3          // 部分和
  fadd    fr0, fr2          // 最终结果

应用: 向量点积
  // 计算法向量与光照方向的点积
  float N[4] = {nx, ny, nz, 0};
  float L[4] = {lx, ly, lz, 0};
  float dot = fipr(N, L);
  float intensity = max(0, dot);  // Lambertian shading

性能对比:
  标准C: 4次乘法 + 3次加法 = 7次操作
  fipr:  1条指令
  加速: ~7倍
```

### 5.3 Reciprocal Square Root (1/√x)

```
函数原型:
  float fsrra(float x);

操作:
  result = 1 / sqrt(x)

应用1: 向量归一化
  float v[4] = {vx, vy, vz, 0};
  float dot = fipr(v, v);        // ||v||²
  float inv_length = fsrra(dot); // 1/||v||
  
  v[0] *= inv_length;  // 归一化
  v[1] *= inv_length;
  v[2] *= inv_length;

应用2: 光照衰减计算
  float distance = sqrt(dx*dx + dy*dy + dz*dz);
  float inv_distance = fsrra(distance * distance);
  float attenuation = inv_distance;  // 1/distance²

性能对比:
  标准C:  sqrt() + 1.0/x = ~50周期
  fsrra:  1条指令 = ~5周期
  加速: ~10倍
```

### 5.4 Sine/Cosine 内置函数

```
函数原型:
  void fsca(float angle, float* sin_result, float* cos_result);

操作:
  同时计算 sin(angle) 和 cos(angle)

应用1: 相机旋转
  float angle = rotation_speed * time;
  float sin_theta, cos_theta;
  fsca(angle, &sin_theta, &cos_theta);
  
  // 旋转矩阵
  float rotation_matrix[16] = {
    cos_theta,  0, sin_theta, 0,
    0,          1, 0,         0,
    -sin_theta, 0, cos_theta, 0,
    0,          0, 0,         1
  };

应用2: 角色骨骼动画
  For each joint:
    Calculate rotation angles from animation data
    fsca(pitch, &sin_pitch, &cos_pitch)
    fsca(yaw, &sin_yaw, &cos_yaw)
    fsca(roll, &sin_roll, &cos_roll)
    Build rotation matrix
    Apply to vertices

性能优势:
  同时计算sin和cos → 1条指令
  单独计算: sin() + cos() = 2条指令
  节省50%的三角函数计算
```

---

## 六、性能分析工具深度解读

### 6.1 Windows CE Performance Viewer 架构

```
工具界面布局:
┌─────────────────────────────────────────────────┐
│  Frame Rate Bar (紫色)                          │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│  ↑                                            │
│  长度 ∝ 1/FPS                                  │
├─────────────────────────────────────────────────┤
│  Direct3D Bar (灰色 + 彩色线)                  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│  ═════════════════════════════════════════════  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│  ↑            ↑                                │
│  灰色         彩色线                           │
│  应用层       Direct3D DrawPrimitive调用        │
├─────────────────────────────────────────────────┤
│  Hardware Render Bar (浅蓝)                     │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│  ↑                                            │
│  长度 ∝ 硬件渲染时间                            │
└─────────────────────────────────────────────────┘

时间关系:
  Total_Frame_Time = App_Time + D3D_Time + HW_Time
  FPS = 1 / Total_Frame_Time
```

### 6.2 性能指标数学分析

```
性能优化目标函数:
  Minimize: Total_Frame_Time
  Subject to: Visual_Quality ≥ threshold

时间分解:
  T_total = T_app + T_D3D + T_hw
  
  T_app = T_culling + T_animation + T_AI + T_physics
  T_D3D = T_setup + T_state_changes + N_calls × T_call_overhead
  T_hw = N_triangles × T_triangle_avg

优化策略对各项的影响:
  
  1. 视锥剔除:
    ΔT_culling = + (检测时间)
    ΔT_D3D = - (减少的三角形) × T_triangle_avg
    条件: ΔT_D3D > ΔT_culling → 有益
    
  2. Strips:
    ΔT_D3D = - (减少的顶点数) × T_vertex_processing
    ΔT_hw = - (减少的顶点数) × T_vertex_transfer
    无额外开销 → 总是有益
    
  3. DONOTCLIP:
    ΔT_D3D = - (减少的clipping计算)
    无额外开销 → 总是有益
    
  4. 数据对齐:
    ΔT_D3D = - (减少的内存拷贝)
    ΔT_cache = + (更好的缓存利用率)
    总是有益

性能公式:
  Speedup = T_before / T_after
  
  例如: 文章案例
    T_before = 100ms (FPS ≈ 10)
    T_after = 16.67ms (FPS = 60)
    Speedup = 6倍
```

### 6.3 实验数据分析

```
优化过程数据表 (从文章推断):

阶段        App Bar    D3D Bar     HW Bar    FPS
────────────────────────────────────────────────
初始状态    ████████   ████████   ████████   <10
            ████████   ════════════════════   (零散调用)
            ████████   ████████

32字节对齐  ████████   ████████   ████████   ~15
            ████████   ════════════════════   (连续)
            ████████   ████████

状态分组    ███████   ███████    ████████    ~30
            ███████   ═════════              (更少调用)
            ███████   ███████

Strip生成   ██████    ██████     ████████    ~45
            ██████    ════════              (单次调用)
            ██████    ██████

完全优化    ████      ████       ████████    ~60
            ████      ════                  (最小数据)
            ████      ████

关键观察:
  1. D3D Bar的"灰色"部分增加 → 应用层做更多工作(culling)
  2. D3D Bar的"彩色"减少 → DrawPrimitive调用减少
  3. HW Bar相对稳定 → 瓶颈转移到应用层
```

---

## 七、深度技术细节与扩展

### 7.1 VQ纹理压缩

文章提到的 **VQ (Vector Quantization)** 压缩技术：

```
VQ压缩原理:
  
  原始纹理: W × H × C bytes (C = bytes per pixel)
  
  压缩步骤:
    1. 将纹理划分为 4×4 blocks (16 pixels per block)
    2. 为每个block计算代表性颜色向量 (code vector)
    3. 将code vectors存储在codebook中
    4. 每个block存储一个codebook index (2 bytes)
  
  Codebook大小:
    N_code_vectors × 4 (RGBA) × 2 bytes = 8×N_code_vectors bytes
    通常N_code_vectors = 256
    Codebook = 2048 bytes = 2 KB (与文章一致)
  
  压缩比计算:
    Block count = (W×H) / 16
    Compressed size = 2×Block_count + 2048
    
    例如: 256×256 RGBA纹理
      Block count = 65536 / 16 = 4096
      Compressed size = 8192 + 2048 = 10240 bytes
      Original size = 65536 × 4 = 262144 bytes
      压缩比 = 262144 / 10240 ≈ 25.6:1
    
    文章声称8:1 → 可能是针对较小纹理或不同参数设置

VQ解压缩 (硬件加速):
  For each block at index i:
    code = texture_data[i]
    color = codebook[code]
    For each pixel in 4×4 block:
      pixel = color

性能影响:
  + 显存带宽减少8倍
  + 缓存利用率提高
  - 颜色精度损失 (每个block只有一种颜色)
```

### 7.2 Bump Mapping 实现

```
凹凸贴图原理:
  
  对于每个像素:
    1. 读取高度图 H(u, v) ∈ [0, 1]
    2. 计算高度梯度:
         ∂H/∂u ≈ (H(u+1, v) - H(u-1, v)) / 2
         ∂H/∂v ≈ (H(u, v+1) - H(u, v-1)) / 2
    
    3. 计算扰动法向量:
         N' = N + (∂H/∂u) × T + (∂H/∂v) × B
         其中T和B是切线空间的基向量
         归一化: N'' = N' / ||N'||
    
    4. 使用N''进行光照计算:
         diffuse = max(0, N'' · L)
         specular = pow(max(0, R · V), shininess)
         R = reflect(L, N'')
         V = view direction vector

Dreamcast硬件优化:
  硬件内联高度梯度计算
  硬件法向量扰动
  避免应用层循环遍历每个像素
  单周期完成整个操作
```

### 7.3 Volume Testing 技术

```
Volume Testing原理:
  
  定义测试体积 V (sphere, box, cylinder等):
    V = { P ∈ ℝ³ | Condition(P) = true }
  
  对于每个屏幕像素 p = (x, y):
    检测像素是否在体积V的投影内
    
    If pixel in V:
      应用特定操作:
        - Color modification: pixel_color *= factor
        - Transparency: pixel_alpha = new_alpha
        - Texture replacement: use_texture(T_special)
        - Shadow: darken pixel
        - Spotlight: brighten pixel

数学表达:
  Let V be a sphere with center C and radius R:
    Condition(P) = ||P - C|| ≤ R
    
  For pixel p at screen position (sx, sy):
    通过逆投影获得世界空间位置 P_world
    If ||P_world - C|| ≤ R:
      Apply operation to pixel p

应用场景:
  1. 体积光/雾
  2. 投射阴影
  3. 聚光灯效果
  4. 区域性后期处理
  5. 动态环境光遮蔽

优势:
  + 不需要修改几何体
  + 硬件加速测试
  + 精确到像素级别
  + 保持几何流水线完整性
```

### 7.4 网格生成最佳实践

```
艺术家与程序员协作指南:

1. 纹理映射策略:
   - 使用少量大纹理而非多张小纹理
   - 确保相邻顶点共享UV坐标
   - 避免纹理接缝
   
   纹理坐标共享条件:
     For vertices v₁ and v₂:
       If v₁ and v₂ are geometrically adjacent:
         Then UV(v₁) should = UV(v₂)
  
2. 拓扑结构:
   - 优先使用三角形strip友好的拓扑
   - 减少三角形面数
   - 确保网格是流形 (manifold)
   
   流形条件:
     - 每条边恰好属于1或2个三角形
     - 无孤立顶点
     - 无自相交
   
3. 法向量连续性:
   - 硬边需要独立顶点
   - 平滑面共享法向量
   
   法向量计算:
     For vertex v with N adjacent triangles {T₁, ..., Tₖ}:
       N_v = (1/k) × Σ N(Tᵢ)
       N_v = N_v / ||N_v||
  
4. LOD (Level of Detail) 考虑:
   - 为不同距离创建不同面数的模型
   - 确保LOD切换连续
   
   LOD距离公式:
     For object at distance D:
       If D < D_near: use LOD_0 (highest detail)
       Elif D < D_mid: use LOD_1
       Else: use LOD_2 (lowest detail)
```

---

## 八、与其他平台的对比分析

### 8.1 Dreamcast vs PS2 (PlayStation 2)

| 特性 | Dreamcast (PowerVR2) | PS2 (GS + EE) |
|------|---------------------|---------------|
| **渲染架构** | Tile-Based Deferred | Immediate Mode |
| **Z-Buffer** | 不需要 | 需要 (24-bit) |
| **显存带宽** | ~800 MB/s (共享) | ~48 GB/s (eDRAM) |
| **像素填充率** | ~100 MPixels/s | ~2.4 GPixels/s |
| **几何处理** | SH-4 CPU + PVR T&L | VU0/VU1 矢量单元 |
| **Overdraw** | 免费 | 昂贵 |
| **透明度** | 自动排序 | 手动排序 |

**性能公式对比**：

```
传统IMR (PS2):
  RenderTime = N_triangles × (T_setup + T_raster) × (1 + Overdraw)
  
  对于高overdraw场景:
    Overdraw = 3-5倍
    代价巨大

Dreamcast TBDR:
  RenderTime = N_pixels × T_raster
  
  Overdraw对渲染时间无影响
  
  等效公式:
    T_IMR = T_TBDR × (1 + Overdraw × (T_setup/T_raster))
```

### 8.2 Dreamcast vs PC (1999年)

```
1999年典型PC配置:
  CPU: Pentium III 500-700 MHz
  GPU: TNT2, Voodoo3, GeForce 256
  
PC Direct3D vs Dreamcast Direct3D:

1. API调用开销:
   PC: AGP总线延迟高
   Dreamcast: 集成内存，低延迟
   
2. 内存架构:
   PC: CPU内存 (system RAM) + GPU内存 (VRAM)
   Dreamcast: 统一内存架构 (UMA)
   
3. 驱动层:
   PC: 通用驱动，优化少
   Dreamcast: 专用驱动，硬件优化
   
4. 几何处理:
   PC: CPU或GPU T&L (早期只有CPU)
   Dreamcast: 专用T&L硬件
```

---

## 九、现代技术演进与关联

### 9.1 Tile-Based Rendering 的现代继承者

```
TBDR技术树:

PowerVR Series 2 (Dreamcast, 1998)
    ↓
PowerVR Series 3-5 (Mobile GPUs, 2000s)
    ↓
PowerVR SGX (iPhone 3G, 2008)
    ↓
PowerVR Rogue (iPhone 5S, 2013)
    ↓
PowerVR Series 7/8 (iPhone X, 2017+)
    ↓
Apple A系列GPU (定制设计, 继承TBDR)

其他TBDR实现:
  - ARM Mali (Midgard/Valhall)
  - Qualcomm Adreno (部分特性)
  - Apple M系列GPU
```

### 9.2 现代API中的对应概念

```
文章中的概念 → 现代API对应:

1. DrawPrimitive/DrawIndexedPrimitive
   → Modern: vkCmdDrawIndexed (Vulkan), glDrawElements (OpenGL)
   → Metal: drawIndexedPrimitives

2. DONOTCLIP
   → Modern: 硬件自动clipping，无需标志
   → 优化重点: Early-Z testing

3. VQ纹理压缩
   → Modern: ASTC, ETC2, BPTC
   → 压缩比: 6:1 (ASTC LDR) 到 12:1 (ASTC HDR)

4. SH4 Intrinsics
   → Modern: SIMD指令集 (SSE, AVX, NEON)
   → 着色器语言: intrinsic functions

5. Performance Viewer
   → Modern: GPU Profiler工具
     • NVIDIA Nsight
     • AMD Radeon GPU Profiler
     • Apple Instruments
     • Snapdragon Profiler
```

### 9.3 现代优化技术对比

```
文章技术 (1999)           现代对应 (2020s)
─────────────────────────────────────────────
Bounding Sphere剔除      → BVH (Bounding Volume Hierarchy) 
                        → Bounding Volume Hierarchy in RTX

Triangle Strips          → Index Buffer + Cache optimization
                        → Mesh Shaders

VQ纹理压缩              → ASTC / BC7 / ETC2

32字节对齐              → Cache line alignment (64 bytes)
                        → SoA (Structure of Arrays) layout

View Frustum Culling     → Hierarchical Z-buffer
                        → GPU-driven culling (Amplification Shaders)
```

---

## 十、关键公式汇总

### 10.1 几何测试公式

```
【点到平面距离】
  d = n·P + d₀
  其中 n = (nₓ, nᵧ, n_z) 是单位法向量
      d₀ 是平面常数
      P = (x, y, z) 是点坐标

【球与Frustum相交测试】
  For each plane i (i = 0..5):
    dᵢ = nᵢ·C + dᵢ₀
    If dᵢ < -R:
      Return OUTSIDE
    Elif dᵢ > R:
      Continue
    Else:
      Return INTERSECT
  Return INSIDE

【背面剔除】
  N = (v₁ - v₀) × (v₂ - v₀)
  V = CameraPosition - v₀
  If N·V < 0:
    Cull triangle

【向量归一化】
  v̂ = v / ||v||
  ||v|| = sqrt(vₓ² + vᵧ² + v_z²)
  使用fsrra优化:
    inv_length = fsrra(vₓ² + vᵧ² + v_z²)
    v̂ = v × inv_length
```

### 10.2 性能公式

```
【帧率计算】
  FPS = 1 / T_frame
  T_frame = T_app + T_D3D + T_hw

【优化加速比】
  Speedup = T_before / T_after

【Strip数据节省】
  Ratio = (3N) / (N + 2) ≈ 3 (对于大N)

【VQ压缩比】
  CompressionRatio = (W×H×C) / (2×(W×H)/16 + 2048)
  
  理想极限 (大纹理):
    Limit = (C×16) / 2 = 8C
    对于 RGBA (C=4): Limit = 32:1
    文章声称: 8:1 (保守估计)
```

---

## 十一、参考资料与扩展阅读

### 核心论文与技术文档

1. **Tile-Based Deferred Rendering**:
   - [PowerVR Architecture Overview](https://www.imgtec.com/technology/powervr-architecture/)
   - [Tile-Based Graphics](https://developer.qualcomm.com/software/adreno-gpu-sdk/gpu-programs)

2. **Dreamcast Hardware**:
   - [Dreamcast Technical Documentation](https://github.com/Kazade/dcdev)
   - [SH-4 Processor Manual](https://www.renesas.com/us/en/doc/products/mpumcu/doc/sh7750e_hw.pdf)

3. **Direct3D Optimization**:
   - [Direct3D 9 Optimization](https://docs.microsoft.com/en-us/windows/win32/direct3d9/d3d-optimization)
   - [GPU Gems: Programming Techniques](https://developer.nvidia.com/gpugems/gpugems)

4. **现代GPU架构**:
   - [ARM Mali Architecture](https://developer.arm.com/ip-products/graphics/mali-gpu)
   - [Apple GPU Architecture](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

### 相关技术关键词

- **Tile-Based Deferred Rendering (TBDR)**
- **Hierarchical Z-Buffering**
- **Early-Z Culling**
- **Bounding Volume Hierarchy (BVH)**
- **Triangle Strip Generation Algorithms**
- **Vertex Cache Optimization**
- **SIMD Vector Processing**
- **Texture Compression (VQ, ASTC, BC7)**
- **GPU Profiling Tools**
- **Draw Call Batching**
- **State Sorting**

---

## 总结

这篇文章是游戏开发史上关于 **专用硬件优化** 的经典文档。其核心贡献在于：

1. **揭示了TBDR架构的独特优势**，特别是对overdraw的免费处理
2. **提供了完整的几何优化流水线**，从视锥剔除到strip生成
3. **强调了数据局部性和内存对齐**的重要性
4. **展示了性能分析工具**在优化过程中的关键作用

文章中的许多概念（如tile-based渲染、向量量化压缩、intrinsics优化）在今天依然相关，并且在现代移动GPU（如Apple A系列、Qualcomm Adreno、ARM Mali）中得到延续和发展。

---

*如需更深入了解某个具体技术点，请告知，我可以提供更详细的公式推导、代码示例或架构图解。*