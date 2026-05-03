
# OpenVDB 与 NanoVDB：从第一性原理深入理解

---

## 一、核心问题：为什么需要 VDB？

想象你要在 3D 空间中存储一团**烟雾**。烟雾是"稀疏的"——大部分空间是空的，只有少数区域有密度值。

### 朴素方案的问题

| 方案 | 存储量 | 问题 |
|------|--------|------|
| 均匀 3D Grid (128³) | 2M cells | **96%+ 是空的**，浪费巨大 |
| Octree | 自适应 | **随机访问慢**，cache 不友好 |
| Hash Map | 只存非空 | **邻域遍历极慢**，无法做插值 |

**第一性原理思考**：我们需要一种数据结构，它同时满足：
1. **空间效率**：只存有数据的区域（sparse）
2. **随机访问 O(1)**：给定 $(x, y, z)$，快速找到值
3. **局部性**：相邻体素在内存中也相邻
4. **动态增删**：能随时添加/删除活跃区域

这就是 **VDB** 要解决的问题。

---

## 二、OpenVDB 数据结构详解

### 2.1 核心思想：Hierarchical Tree of Fixed-Branching-Factor

OpenVDB 使用一棵 **5 层树**，每层分支因子固定为 $2^{b_i}$（$b_i$ 为该层的 bit 宽度），形成一个**隐式的空间索引**。

```
Root (层4) → 可变数量的子节点
  │
  ├─ InternalNode (层3): 覆盖 32³ 体素
  │    │
  │    ├─ InternalNode (层2): 覆盖 16³ 体素
  │    │    │
  │    │    └─ LeafNode (层1): 覆盖 8³ = 512 体素 ← 实际数据
  │    │
  │    └─ ...
  └─ ...
```

### 2.2 各层参数表

| 层级 | 类型 | 维度 bit | 覆盖范围 | 子节点数 | 子节点类型 |
|------|------|----------|----------|----------|-----------|
| 4 | RootNode | 可变 | 全局 | 可变 | InternalNode2 |
| 3 | InternalNode2 | 32²×16 | 32×32×4096 体素 | $32×32×256 = 262144$ | InternalNode1 |
| 2 | InternalNode1 | 16³ | 16³ = 4096 体素 | $16^3 = 4096$ | LeafNode |
| 1 | LeafNode | 8³ | 8³ = 512 体素 | — | **实际数据** |

> 总覆盖：$32 \times 16 \times 8 = 4096$ 体素/维度方向，即覆盖 $4096^3$ 的 3D 空间。

### 2.3 关键数据结构：LeafNode

```
┌─────────────────────────────────────────┐
│ LeafNode (8³ = 512 voxels)              │
│ ┌─────────────────────────────────────┐ │
│ │ bitmask: 512 bits (64 bytes)        │ │ ← 哪些 voxel 有值
│ │ value mask: 512 bits (64 bytes)     │ │ ← 哪些 voxel ≠ background
│ │ values: ValueType[512]              │ │ ← 实际数据 (或 tile value)
│ │ bbox: CoordBBox                     │ │ ← 局部 bounding box
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Tile 优化**：如果一个 LeafNode 的 512 个体素值**全部相同**（比如都是 0 或都是 0.5），它不会存储 512 个浮点数，而是只存**一个 tile value** + 一个标志位。这带来了巨大的压缩比。

### 2.4 访问算法：如何从 $(x,y,z)$ 找到值？

```
给定世界坐标 (x, y, z):

1. 计算 Root 的子节点 key:
   key = (x >> 27, y >> 27, z >> 27)  // 右移 27 bit = 9+9+9

2. 在 Root 的 hash map 中查找 key → InternalNode2*

3. 在 InternalNode2 中:
   child = (x >> 12, y >> 12, z >> 12) & 0x1F  // 取 5 bit
   → InternalNode1*

4. 在 InternalNode1 中:
   child = (x >> 8, y >> 8, z >> 8) & 0xF      // 取 4 bit
   → LeafNode*

5. 在 LeafNode 中:
   offset = ((x & 7) << 6) | ((y & 7) << 3) | (z & 7)  // 3+3+3 bit
   → values[offset]
```

**复杂度**：$O(\text{tree depth}) = O(4)$，本质上是**常数时间**。

每个坐标 $(x,y,z)$ 的 **bit 分解**：

```
坐标 x (假设 32-bit int):
├─ bit 27-31: Root key (5 bit，实际可变)
├─ bit 12-26: InternalNode2 index (15 bit = 5+5+5)
├─ bit 8-11:  InternalNode1 index (4 bit)
├─ bit 0-7:   LeafNode offset (8 bit = 3+3+2，但VDB用8³=9 bit)
```

### 2.5 内存效率示例

假设一帧烟雾体积为 $1000^3$ voxel，但只有 5% 有值：

| 表示方法 | 内存 |
|----------|------|
| Dense Grid (float) | $1000^3 \times 4\text{B} \approx 4\text{GB}$ |
| OpenVDB (sparse) | $1000^3 \times 0.05 \times 4\text{B} + \text{tree overhead} \approx 200\text{MB} + \sim20\text{MB}$ |
| **压缩比** | **~20x** |

---

## 三、OpenVDB 的局限：为什么需要 NanoVDB？

### 3.1 OpenVDB 的 CPU 中心设计

OpenVDB 是 2012 年由 DreamWorks Animation 开发的，**为 CPU 设计**：

- 使用了大量的 **C++ 虚函数** 和继承
- 使用了 **`std::shared_ptr`** 和引用计数
- 内部节点使用 **`std::map`** 存储 child pointers
- 内存布局是**指针追逐型**（pointer-chasing），极度不 GPU 友好

### 3.2 GPU 的根本限制

| CPU 友好的 | GPU 友好的 |
|-----------|-----------|
| 指针 + 链表 | 连续内存 (Array of Structures / SOA) |
| 虚函数 + 多态 | 固定布局 + switch/compute |
| 深层嵌套 tree 遍历 | 扁平化、可预测的访问模式 |
| 动态分配 | 静态或 semi-static 分配 |

**核心矛盾**：OpenVDB 的树遍历过程中，每一步都是**间接寻址**（dereference pointer），这在 GPU 上意味着：
- **无法在 CUDA kernel 中直接遍历**
- **warp divergence**：不同线程走不同路径
- **cache miss 率极高**：GPU L1 cache 小，pointer chasing 是灾难

---

## 四、NanoVDB：GPU 友好的重新设计

### 4.1 设计哲学

NanoVDB 的核心思想是：**把 OpenVDB 的动态树结构"烘焙"（bake）成一个 GPU 可以直接遍历的扁平化、自包含的内存块。**

```
OpenVDB (CPU, dynamic)          NanoVDB (GPU, static/baked)
┌──────────────────┐           ┌──────────────────────────┐
│  shared_ptr<Node> │  ──Bake──→│  Flat buffer: void*      │
│  virtual funcs   │           │  No pointers, offsets only│
│  std::map        │           │  POD types only           │
│  mutable         │           │  Read-only on GPU         │
└──────────────────┘           └──────────────────────────┘
```

### 4.2 NanoVDB 内存布局

NanoVDB 将整棵 VDB 树打包成**一块连续的内存**，由以下部分组成：

```
┌──────────────────────────────────────────────────────────┐
│ NanoVDB Grid Buffer (一整块连续内存)                      │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Grid Metadata (固定大小头部)                        │  │
│  │   - grid type (float, vec3, etc.)                  │  │
│  │   - index-to-world transform (4×4 matrix)         │  │
│  │   - world-to-index transform (4×4 matrix)         │  │
│  │   - voxel size, bbox                              │  │
│  │   - offset to tree                                │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Tree Header                                        │  │
│  │   - node counts per level                         │  │
│  │   - offsets to root, internal, leaf arrays         │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Root Table (array of RootNode entries)             │  │
│  │   每个 entry: {coord_min, child_offset}            │  │
│  │   child_offset 是相对于 buffer 起点的 byte 偏移    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Upper Internal Nodes (Level 2) - 连续数组          │  │
│  │   每个: {bbox, child_mask, value_mask,             │  │
│  │          table[4096] = {tile_value | child_offset}}│  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Lower Internal Nodes (Level 1) - 连续数组          │  │
│  │   同上结构                                         │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Leaf Nodes (Level 0) - 连续数组 ★关键★             │  │
│  │   每个: {bbox, value_mask, values[512], min, max}  │  │
│  │   values 就是 512 个实际 voxel 值                   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 4.3 关键设计决策

#### (1) 用 **byte offset** 替代 **pointer**

```cpp
// OpenVDB: 指针
ChildNodeType* childPtr = node->child(i);

// NanoVDB: byte offset
uint64_t childOffset = node->childOffset(i);
// 实际地址 = buffer_base + childOffset
```

这样整个 buffer 可以被 `cudaMemcpy` 到 GPU，所有偏移量仍然有效！

#### (2) 所有节点都是 **POD (Plain Old Data)**

```cpp
// NanoVDB LeafNode (简化版)
struct LeafNode {
    CoordBBox  bbox;          // 24 bytes
    uint64_t   valueMask;     // 8 bytes (8×64=512 bits)
    ValueType  values[512];   // 2048 bytes (float)
    ValueType  minimum;       // 4 bytes
    ValueType  maximum;       // 4 bytes
    // 总计: ~2088 bytes
};
```

- **没有虚函数表**
- **没有 `std::` 容器**
- **可以直接 memcpy**

#### (3) 同级节点连续存储

所有 LeafNode 存放在一个连续数组中。这意味着：
- 当一个 warp (32 threads) 访问相邻 voxel 时，它们大概率命中同一个 LeafNode
- **内存访问 coalescing** 极佳

### 4.4 NanoVDB 的遍历算法 (GPU Kernel)

```cuda
__device__ float sampleNanoVDB(const NanoGrid* grid, float wx, float wy, float wz) {
    // Step 1: World → Index transform
    float ix = wx * grid->worldToIndex[0][0] + grid->worldToIndex[0][3];
    float iy = wy * grid->worldToIndex[1][1] + grid->worldToIndex[1][3];
    float iz = wz * grid->worldToIndex[2][2] + grid->worldToIndex[2][3];
    
    // Step 2: Trilinear interpolation → 需要 8 个相邻 voxel
    int x0 = floorf(ix), y0 = floorf(iy), z0 = floorf(iz);
    
    // Step 3: 遍历 VDB tree 找 LeafNode (对 8 个 voxel 分别做)
    // Root lookup
    const RootNode* root = grid->tree().root();
    // ... InternalNode traversal via offsets ...
    // ... LeafNode access via offset ...
    
    // Step 4: Trilinear blend
    //   f = (1-dx)(1-dy)(1-dz) * v000
    //     + dx*(1-dy)*(1-dz)    * v100
    //     + ...
    //     + dx*dy*dz            * v111
    return f;
}
```

### 4.5 性能对比

来自 Pixar 的实测反馈（来源：NVIDIA GTC 演讲）：

| 场景 | CPU OpenVDB (RenderMan) | GPU NanoVDB (原型) | 加速比 |
|------|------------------------|-------------------|--------|
| 烟雾体积 ray march | ~30 sec/frame | ~3 sec/frame | **~10x** |
| 碰撞检测 | CPU bound | GPU accelerated | **5-20x** |

SideFX (Houdini) 的实测：
- Vellum Solver 的 static collision → GPU
- Pyro Solver 的 sourcing → GPU
- 交互式体验从"卡顿"变为"流畅"

---

## 五、OpenVDB ↔ NanoVDB 的工作流

```
                  CPU                               GPU
          ┌──────────────┐                  ┌──────────────┐
  创建 →  │  OpenVDB     │                  │              │
  编辑 →  │  (动态,可修改) │  ──createNanoVDB()──→  │  NanoVDB     │
  模拟 →  │              │                  │  (静态,只读)  │
          │              │  ←─toOpenVDB()────  │              │
          └──────────────┘                  └──────────────┘
                                               ↓
                                          Ray Marching
                                          Collision
                                          Filtering
                                          Simulation
```

关键 API：

```cpp
// OpenVDB → NanoVDB
openvdb::FloatGrid::Ptr vdbGrid = ...;
nanovdb::GridHandle<> nanoHandle = nanovdb::createNanoGrid(*vdbGrid);

// 传输到 GPU
void* d_buffer = nanoHandle.devicePtr();  // CUDA managed memory

// GPU 上使用
const nanovdb::NanoGrid<float>* deviceGrid = 
    reinterpret_cast<const nanovdb::NanoGrid<float>*>(d_buffer);

// NanoVDB → OpenVDB (如果需要回 CPU 修改)
openvdb::FloatGrid::Ptr backToVDB = nanovdb::toOpenVDB(nanoHandle);
```

---

## 六、架构图解析

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Houdini │  │  Blender │  │  Maya    │  │ Unreal/Unity│  │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘  │
│       │             │             │                │         │
├───────┼─────────────┼─────────────┼────────────────┼─────────┤
│       ▼             ▼             ▼                ▼         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              OpenVDB Library (CPU)                    │   │
│  │  - Sparse volume editing / simulation                │   │
│  │  - Dynamic tree management                          │   │
│  │  - Level set / fog volume tools                     │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │ createNanoGrid()                   │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │              NanoVDB Library (GPU/CPU portable)       │   │
│  │  - Flat buffer representation                        │   │
│  │  - CUDA / HIP / CPU traversal kernels                │   │
│  │  - Read-only, cache-friendly access                  │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │              fVDB (Deep Learning on VDB)              │   │
│  │  - PyTorch integration                               │   │
│  │  - Neural radiance fields (NeRF) on sparse volumes   │   │
│  │  - AI-based volume generation / completion           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 七、数学基础：VDB 的空间映射

### 7.1 Index Space → World Space Transform

VDB 定义了一个 **仿射变换**：

$$\mathbf{x}_{\text{world}} = \mathbf{M} \cdot \mathbf{x}_{\text{index}} + \mathbf{t}$$

其中：
- $\mathbf{x}_{\text{index}} = (i, j, k)^T$ 是整数索引坐标
- $\mathbf{M} \in \mathbb{R}^{3\times3}$ 是缩放/旋转矩阵
- $\mathbf{t} \in \mathbb{R}^3$ 是平移向量
- $\mathbf{x}_{\text{world}}$ 是世界坐标

通常简化为：$\mathbf{M} = \text{diag}(\Delta x, \Delta y, \Delta z)$，其中 $\Delta x, \Delta y, \Delta z$ 是 voxel 尺寸。

### 7.2 坐标到 Tree 路径的映射

给定索引坐标 $(i, j, k)$，VDB 使用 **位操作** 将其分解为各层索引：

$$\text{root\_key} = \left( \lfloor i / 2^{27} \rfloor, \lfloor j / 2^{27} \rfloor, \lfloor k / 2^{27} \rfloor \right)$$

$$\text{internal2\_key} = \left( (i \gg 12) \,\&\, 0x1F,\; (j \gg 12) \,\&\, 0x1F,\; (k \gg 12) \,\&\, 0x7F \right)$$

$$\text{internal1\_key} = \left( (i \gg 8) \,\&\, 0xF,\; (j \gg 8) \,\&\, 0xF,\; (k \gg 8) \,\&\, 0xF \right)$$

$$\text{leaf\_offset} = (i \,\&\, 7) \times 64 + (j \,\&\, 7) \times 8 + (k \,\&\, 7)$$

其中：
- $\gg$ 表示算术右移
- $\&$ 表示按位与
- $0x1F = 31$, $0xF = 15$, $7 = 0b111$
- 上标 $27, 12, 8$ 分别是各层覆盖的 bit 宽度之和

### 7.3 Trilinear 插值公式

当采样点 $(i_f, j_f, k_f)$ 不在整数格点时，需要三线性插值：

$$f(i_f, j_f, k_f) = \sum_{a=0}^{1} \sum_{b=0}^{1} \sum_{c=0}^{1} w_x^a \cdot w_y^b \cdot w_z^c \cdot V_{i_0+a, j_0+b, k_0+c}$$

其中：
- $i_0 = \lfloor i_f \rfloor$, $j_0 = \lfloor j_f \rfloor$, $k_0 = \lfloor k_f \rfloor$
- $w_x^0 = 1 - (i_f - i_0)$, $w_x^1 = i_f - i_0$（即分数部分）
- $w_y, w_z$ 同理
- $V_{i,j,k}$ 是整数格点上的 voxel 值
- 需要 **8 次 VDB 树遍历** 获取 8 个邻域值

---

## 八、NanoVDB 的 GPU 优化细节

### 8.1 Warp-Cooperative Traversal

在 GPU ray marching 中，一个 warp 内的 32 个线程通常访问**空间相邻**的 voxel。NanoVDB 利用这一点：

1. **Root/Internal 层级**：同一 warp 很可能走同一条路径 → **极低 divergence**
2. **Leaf 层级**：同一 warp 命中相同 LeafNode 概率高 → **L1 cache 命中率高**

### 8.2 内存节省

NanoVDB 使用 **16-bit half** 或甚至 **8-bit quantized** 格式存储值：

| 格式 | 每个 voxel | 精度 | 适用场景 |
|------|-----------|------|---------|
| float32 | 4 B | 全精度 | Level set, CSG |
| float16 | 2 B | ~3 位十进制 | 渲染 |
| uint8 (quantized) | 1 B | 256 级 | 可视化预览 |

相比 OpenVDB 的 `float` (4B)，NanoVDB 可以将 LeafNode 从 2088B 缩减到 ~1050B (half) 或 ~530B (uint8)。

### 8.3 CUDA Managed Memory 支持

NanoVDB 支持 `cudaMallocManaged`，使得同一个 buffer 可以在 CPU 和 GPU 之间**零拷贝共享**（在支持 unified memory 的架构上）：

```cpp
nanovdb::GridHandle<> handle = nanovdb::createNanoGrid(*vdbGrid, nanovdb::StatsMode::All, 
                                                         nanovdb::ChecksumMode::Full,
                                                         /*bufferType=*/nanovdb::GridClass::LevelSet,
                                                         /*allocator=*/nanovdb::Allocator::Device);
```

---

## 九、生态与应用场景

| 应用 | 工具/引擎 | 用途 |
|------|----------|------|
| 电影 VFX | Houdini, Maya, RenderMan | 烟/火/水/爆炸体积渲染 |
| 实时渲染 | Blender Cycles, Omniverse | GPU ray tracing 体积 |
| 碰撞检测 | Houdini Vellum | GPU sparse collision |
| AI/深度学习 | fVDB | NeRF, 3D reconstruction |
| 游戏开发 | Unreal Engine (插件) | 实时体积云/雾 |
| 医学影像 | 自研 pipeline | CT/MRI 体素处理 |

---

## 十、总结对比

| 维度 | OpenVDB | NanoVDB |
|------|---------|---------|
| **设计目标** | CPU 编辑/模拟 | GPU 渲染/查询 |
| **可变性** | 可动态增删节点 | 只读 |
| **内存布局** | 指针追逐, 分散 | 连续 buffer, 偏移寻址 |
| **数据类型** | C++ 模板 + 虚函数 | POD, 可 memcpy |
| **GPU 支持** | 无原生支持 | CUDA/HIP/CPU 原生 kernel |
| **转换成本** | — | `createNanoGrid()` 一次性开销 |
| **维护者** | ASWF (Academy Software Foundation) | ASWF + NVIDIA |
| **许可证** | Apache 2.0 | Apache 2.0 |
| **代码量** | ~200K+ 行 C++ | ~30K 行 CUDA/C++ |

**一句话总结**：
> **OpenVDB 是稀疏体积数据的"编辑器"，NanoVDB 是它的"GPU 加速只读副本"。两者配合：CPU 创建/修改 → 烘焙到 GPU → 实时渲染/查询 → (可选) 回传 CPU 再修改。**

---

## 参考链接

- OpenVDB 官方仓库: [https://github.com/AcademySoftwareFoundation/openvdb](https://github.com/AcademySoftwareFoundation/openvdb)
- NanoVDB 在 OpenVDB 内的文档: [https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb](https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb)
- NVIDIA NanoVDB 页面: [https://developer.nvidia.com/nanovdb](https://developer.nvidia.com/nanovdb)
- fVDB (NanoVDB 上的深度学习框架): [https://developer.nvidia.com/fvdb](https://developer.nvidia.com/fvdb)
- OpenVDB 原始论文 (Museth, 2013): *VDB: High-Resolution Sparse Volumes with Dynamic Topology*, ACM TOG [https://doi.org/10.1145/2487228.2487235](https://doi.org/10.1145/2487228.2487235)
- NanoVDB 论文 (Museth, 2021): *NanoVDB: A GPU-Friendly and Portable VDB Data Structure For Real-Time Rendering And Simulation*, HPG [https://dl.acm.org/doi/10.1145/3459029.3460114](https://dl.acm.org/doi/10.1145/3459029.3460114)
- GTC 演讲: *NanoVDB: Real-Time GPU Rendering of Volumes* [https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41583/](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41583/)
- ASWF OpenVDB 项目: [https://openvdb.org/](https://openvdb.org/)