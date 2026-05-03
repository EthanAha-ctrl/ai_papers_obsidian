# Wavefront Path Tracing 详解

## 1. 传统 Path Tracing 算法与 GPU 执行问题

### 传统 Path Tracing 算法

传统 Path Tracing 算法的伪代码非常简洁：

```cpp
vec3 Trace(vec3 O, vec3 D)
    IntersectionData i = Scene::Intersect(O, D)
    if (i == NoHit) return vec3(0)          // ray left the scene
    if (i == Light) return i.material.color // lights do not reflect
    vec3 R, pdf = RandomDirectionOnHemisphere(i.normal), 1 / 2PI
    return Trace(i.position, R) * i.BRDF * dot(i.normal, R) / pdf
```

这个递归算法模拟了光子从光源到camera的路径。输入是一个从camera通过屏幕像素的主ray。

### GPU 执行模型：SIMT (Single Instruction Multiple Thread)

GPU 采用 **SIMT** 执行模型。以 NVIDIA Pascal 架构为例，32 个 threads 组成一个 **warp**，这些 threads 共享同一个 **program counter**，以 **lock-step** 方式执行：

```
┌─────────────────────────────────────────────────────────┐
│                    WARP (32 threads)                    │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐     │
│  │ T0 │ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ T7 │... │T31 │     │
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘     │
│                     ↓                                    │
│              Single Program Counter                       │
│                     ↓                                    │
│         All execute same instruction                      │
└─────────────────────────────────────────────────────────┘
```

### 核心问题：Divergence（分支分歧）

当 warp 内的 threads 遇到条件分支时，会产生 **branch divergence**：

```
Thread Divergence 问题示意：

                    ┌─────────────────┐
                    │   if (condition)│
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
      ┌───────▼───────┐             ┌───────▼───────┐
      │   THEN path   │             │   ELSE path  │
      │  (threads A)  │             │  (threads B) │
      └───────┬───────┘             └───────┬───────┘
              │                             │
              │   Threads B are idle       │
              │   while A executes         │
              │                             │
      ┌───────▼───────┐             ┌───────▼───────┐
      │   THEN done   │             │   ELSE done  │
      └───────┬───────┘             └───────┬───────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌───────▼───────┐
                    │  Merge back   │
                    └───────────────┘

Effective Utilization: ~50% (if branches are balanced)
```

**关键问题**：
- 一个主 ray 可能立即找到 light（直接命中）
- 也可能在一次 bounce 后找到 light
- 也可能需要 50 次 bounces 才能找到 light

这导致：
```
Path Length Distribution:
┌────────────────────────────────────────────────────────┐
│ Iteration 0: ████████████████████████████████ (100%)   │
│ Iteration 1: ███████████████████████░░░░░░░░░ (75%)    │
│ Iteration 2: ███████████████░░░░░░░░░░░░░░░░░ (50%)    │
│ Iteration 3: ████████░░░░░░░░░░░░░░░░░░░░░░░░ (25%)    │
│ ...                                                     │
│ Iteration N: █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (5%)     │
└────────────────────────────────────────────────────────┘
█ = Active threads, ░ = Inactive/Waiting threads
```

---

## 2. Wavefront Path Tracing 核心思想

### 第一性原理分析

从第一性原理出发，我们需要解决的核心矛盾是：

```
┌─────────────────────────────────────────────────────────────┐
│                     矛盾分析                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Traditional Approach:                                      │
│   ┌────────────────────────────────────────────────────┐    │
│   │  Thread = Path (一个thread处理一条完整路径)          │    │
│   │                                                     │    │
│   │  Problem:                                          │    │
│   │  - Paths have varying lengths                      │    │
│   │  - Divergence at every conditional                │    │
│   │  - Low hardware utilization                        │    │
│   └────────────────────────────────────────────────────┘    │
│                                                              │
│   Wavefront Solution:                                        │
│   ┌────────────────────────────────────────────────────┐    │
│   │  Thread = Operation at a specific path length       │    │
│   │  (一个thread只处理某一特定bounce的操作)             │    │
│   │                                                     │    │
│   │  Benefits:                                         │    │
│   │  - All threads do the same operation               │    │
│   │  - No divergence within a kernel                   │    │
│   │  - High hardware utilization                       │    │
│   └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Stream Processing 模式

Wavefront Path Tracing 将传统递归算法转换为 **dataflow pipeline**：

```
传统递归模式:
┌─────────────────────────────────────────────────────────────┐
│                    Single Kernel (Megakernel)               │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │Generate │───▶│ Extend  │───▶│  Shade  │───▶│Connect  │──┼──┐
│  │ (ray)   │    │(intersect)   │ (BSDF)  │    │(shadow) │  │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │  │
│       ▲                                                        │ │
│       │                    Recursion                          │ │
│       └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

Wavefront 模式:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌─────────┐         ┌─────────┐         ┌─────────┐        │
│  │ Kernel 1│         │ Kernel 2│         │ Kernel 3│        │
│  │Generate │────────▶│ Extend  │────────▶│  Shade  │        │
│  │(once)   │         │         │         │         │        │
│  └─────────┘         └────┬────┘         └────┬────┘        │
│                           │                   │              │
│                           │                   │              │
│                           ▼                   ▼              │
│                     ┌─────────┐         ┌─────────┐        │
│                     │  Rays   │         │New Rays │        │
│                     │ Buffer  │         │ Buffer  │        │
│                     └─────────┘         └────┬────┘        │
│                                              │              │
│                              ┌───────────────┘              │
│                              │                              │
│                              ▼                              │
│                       ┌─────────┐         ┌─────────┐      │
│                       │ Kernel 4│         │         │      │
│                       │Connect  │────────▶│ Loop    │      │
│                       │(shadow)  │         │ back to │      │
│                       └─────────┘         │ Kernel 2│      │
│                                           └─────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 四个阶段详解

### Phase 1: Generate (生成主 rays)

```cpp
// Kernel 1: Generate Primary Rays
__global__ void generateEyeRays(
    Ray* rayBuffer,          // Output: ray buffer
    int* counter,            // Output: ray count
    Camera camera,           // Input: camera parameters
    int width, int height    // Input: screen dimensions
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Calculate ray direction using pinhole camera model
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;
        
        vec3 D = normalize(
            camera.topLeft + 
            u * camera.right + 
            v * camera.down
        );
        
        rayBuffer[idx].origin = camera.position;
        rayBuffer[idx].direction = D;
    }
}
```

**关键点**：
- 所有 threads **无条件** 执行相同操作
- 输出是连续的 ray buffer
- Ray 数量 = width × height（固定已知）

### Phase 2: Extend (Ray-Scene Intersection)

```
Extension Kernel 数据流:
┌─────────────────────────────────────────────────────────────┐
│                        Input Buffer                         │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐    │
│  │Ray 0│Ray 1│Ray 2│Ray 3│Ray 4│Ray 5│... │Ray N│     │    │
│  │(O,D)│(O,D)│(O,D)│(O,D)│(O,D)│(O,D)│    │(O,D)│     │    │
│  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴─────┴──┬──┴─────┘    │
│     │     │     │     │     │     │          │              │
│     ▼     ▼     ▼     ▼     ▼     ▼          ▼              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            BVH Traversal (all threads active)        │    │
│  │                                                       │    │
│  │   for each ray:                                       │    │
│  │       find nearest intersection                      │    │
│  │                                                       │    │
│  └─────────────────────────────────────────────────────┘    │
│     │     │     │     │     │     │          │              │
│     ▼     ▼     ▼     ▼     ▼     ▼          ▼              │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐          │
│  │Hit 0│Hit 1│Hit 2│Hit 3│Hit 4│Hit 5│... │Hit N│          │
│  │(t,u,v,triID)                                       │          │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘          │
│                        Output Buffer                        │
└─────────────────────────────────────────────────────────────┘
```

**Intersection 数据结构**：
```
struct IntersectionData {
    float t;           // Distance along ray
    float u, v;        // Barycentric coordinates
    uint primitiveID;  // Triangle index
    uint materialID;   // Material index
};
```

### Phase 3: Shade (BSDF 评估与新 Ray 生成)

这是最复杂的阶段：

```cpp
// Kernel 3: Shading
__global__ void shade(
    IntersectionData* hits,        // Input: intersection results
    Ray* extensionRays,            // Output: new path extension rays
    Ray* shadowRays,               // Output: shadow rays
    int* extCounter,              // Atomic counter for extensions
    int* shadowCounter,           // Atomic counter for shadow rays
    PathState* pathStates         // Per-path state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < activePathCount) {
        IntersectionData hit = hits[idx];
        PathState state = pathStates[idx];
        
        if (hit.t == INF) {
            // Ray escaped to environment
            state.contribution = envMap(hit.direction);
            state.active = false;
        }
        else if (hit.material->isEmissive) {
            // Hit light source
            state.contribution *= hit.material->emission;
            state.active = false;
        }
        else {
            // Evaluate BSDF and sample
            
            // Next Event Estimation (NEE)
            vec3 lightDir = sampleLightSource();
            Ray shadowRay = makeShadowRay(hit.position, lightDir);
            
            // Write shadow ray
            int shadowIdx = atomicAdd(shadowCounter, 1);
            shadowRays[shadowIdx] = shadowRay;
            
            // Continue path extension
            vec3 bounceDir = sampleBSDF(hit.material);
            Ray extRay = makeRay(hit.position, bounceDir);
            
            // Write extension ray
            int extIdx = atomicAdd(extCounter, 1);
            extensionRays[extIdx] = extRay;
            
            // Update path state
            state.throughput *= evaluateBSDF(hit.material);
            state.bounceCount++;
        }
    }
}
```

**Shading 阶段的复杂性**：

```
┌─────────────────────────────────────────────────────────────┐
│                   Shade Kernel 流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input: Intersection Results                               │
│          ┌───┬───┬───┬───┬───┬───┐                          │
│          │H0 │H1 │H2 │H3 │H4 │...│                          │
│          └─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┘                          │
│            │   │   │   │   │   │                            │
│            ▼   ▼   ▼   ▼   ▼   ▼                            │
│          ┌─────────────────────┐                             │
│          │   Classification    │                             │
│          │   (no divergence)   │                             │
│          └─────────┬───────────┘                             │
│                    │                                         │
│         ┌──────────┼──────────┐                              │
│         ▼          ▼          ▼                              │
│    ┌────────┐ ┌────────┐ ┌────────┐                         │
│    │ No Hit │ │ Light  │ │ Surface│                         │
│    │        │ │ Source │ │        │                         │
│    └────┬───┘ └────┬───┘ └────┬───┘                         │
│         │          │          │                              │
│         ▼          ▼          ▼                              │
│    Terminate  Terminate   Continue Path                      │
│    Path       Path        - Shadow Ray                       │
│                          - Extension Ray                     │
│                                                                │
│   Output Buffers:                                            │
│   ┌─────────────────┐  ┌─────────────────┐                   │
│   │ Extension Rays  │  │  Shadow Rays    │                   │
│   │ (compacted)     │  │  (compacted)    │                   │
│   └─────────────────┘  └─────────────────┘                   │
│                                                                │
└─────────────────────────────────────────────────────────────┘
```

### Phase 4: Connect (Shadow Ray 处理)

Shadow rays 只需要判断是否被遮挡，不需要找最近交点：

```cpp
// Kernel 4: Connect (Shadow Ray Intersection)
__global__ void connect(
    Ray* shadowRays,           // Input
    int* shadowResults,        // Output: visibility
    int shadowRayCount         // Input: number of shadow rays
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < shadowRayCount) {
        Ray ray = shadowRays[idx];
        
        // Any-hit test (faster than closest-hit)
        bool occluded = sceneAnyHit(ray);
        
        shadowResults[idx] = occluded ? 0 : 1;
    }
}
```

---

## 4. 完整 Pipeline 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Wavefront Path Tracing Pipeline                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                         FRAME START                               │  │
│   └──────────────────────────────┬───────────────────────────────────┘  │
│                                  │                                       │
│                                  ▼                                       │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  PHASE 1: GENERATE (Kernel: generateEyeRays)                     │  │
│   │  ─────────────────────────────────────────────                    │  │
│   │  Input:  Camera parameters, Screen dimensions                     │  │
│   │  Output: Primary ray buffer (N rays)                             │  │
│   │  Memory: RayBuffer[width × height]                               │  │
│   │  Divergence: NONE (all threads do same work)                     │  │
│   └──────────────────────────────┬───────────────────────────────────┘  │
│                                  │                                       │
│                                  ▼                                       │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                      ▀▀▀ ITERATION LOOP ▀▀▀                       │  │
│   │  ┌──────────────────────────────────────────────────────────┐    │  │
│   │  │  PHASE 2: EXTEND (Kernel: intersectRays)                  │    │  │
│   │  │  ───────────────────────────────────────                  │    │  │
│   │  │  Input:  Ray buffer, Scene BVH                           │    │  │
│   │  │  Output: Intersection buffer                             │    │  │
│   │  │  Operation: Closest-hit ray-scene intersection           │    │  │
│   │  │  Divergence: NONE (all rays processed uniformly)        │    │  │
│   │  └──────────────────────────────────────────────────────────┘    │  │
│   │                              │                                     │  │
│   │                              ▼                                     │  │
│   │  ┌──────────────────────────────────────────────────────────┐    │  │
│   │  │  PHASE 3: SHADE (Kernel: shade)                          │    │  │
│   │  │  ─────────────────────────────────                        │    │  │
│   │  │  Input:  Intersection buffer, Material data              │    │  │
│   │  │  Output: Extension rays buffer, Shadow rays buffer       │    │  │
│   │  │  Operations:                                            │    │  │
│   │  │    - Evaluate material BSDF                              │    │  │
│   │  │    - Sample next direction                               │    │  │
│   │  │    - Next Event Estimation (NEE)                         │    │  │
│   │  │    - Russian Roulette termination                        │    │  │
│   │  │  Atomic counters: extCounter, shadowCounter             │    │  │
│   │  └──────────────────────────────────────────────────────────┘    │  │
│   │                              │                                     │  │
│   │                ┌─────────────┴─────────────┐                     │  │
│   │                │                           │                     │  │
│   │                ▼                           ▼                     │  │
│   │  ┌─────────────────────────┐  ┌─────────────────────────┐       │  │
│   │  │ PHASE 4: CONNECT        │  │ Shadow Results Buffer   │       │  │
│   │  │ (Kernel: traceShadow)   │  │ (visibility for NEE)    │       │  │
│   │  │ ──────────────────────  │  └─────────────────────────┘       │  │
│   │  │ Input: Shadow rays      │                                     │  │
│   │  │ Output: Visibility      │                                     │  │
│   │  │ Operation: Any-hit test │                                     │  │
│   │  └─────────────────────────┘                                     │  │
│   │                              │                                     │  │
│   │                              ▼                                     │  │
│   │  ┌──────────────────────────────────────────────────────────┐    │  │
│   │  │  BUFFER SWAP                                             │    │  │
│   │  │  ───────────────────────────────────                     │    │  │
│   │  │  Extension rays become input for next iteration          │    │  │
│   │  │  extCounter → new active path count                      │    │  │
│   │  │  Reset counters for next iteration                       │    │  │
│   │  └──────────────────────────────────────────────────────────┘    │  │
│   │                              │                                     │  │
│   │                              ▼                                     │  │
│   │                    ┌─────────────────┐                            │  │
│   │                    │ More rays?      │                            │  │
│   │                    │ extCounter > 0? │                            │  │
│   │                    └────────┬────────┘                            │  │
│   │                             │                                      │  │
│   │              ┌──────────────┴──────────────┐                      │  │
│   │              │ YES                    NO  │                      │  │
│   │              ▼                            ▼                      │  │
│   │         Back to Phase 2           Exit Loop                      │  │
│   │                                                                    │  │
│   │  └──────────────────────────────────────────────────────────────┘  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│                                  ▼                                       │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  FINALIZE & ACCUMULATE                                            │  │
│   │  ───────────────────────                                          │  │
│   │  - Accumulate contributions to framebuffer                        │  │
│   │  - Temporal accumulation for progressive rendering               │  │
│   │  - Tone mapping & output                                          │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                         FRAME END                                  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 数学公式与渲染方程

### Path Tracing 数学基础

**Rendering Equation (渲染方程)**:

$$L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o) + \int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o) L_i(\mathbf{x}, \omega_i) (\mathbf{n} \cdot \omega_i) d\omega_i$$

其中：
- $L_o(\mathbf{x}, \omega_o)$ = 从点 $\mathbf{x}$ 沿方向 $\omega_o$ 的 outgoing radiance
- $L_e(\mathbf{x}, \omega_o)$ = 自发光项
- $f_r(\mathbf{x}, \omega_i, \omega_o)$ = BRDF (Bidirectional Reflectance Distribution Function)
- $L_i(\mathbf{x}, \omega_i)$ = 从方向 $\omega_i$ 来的 incoming radiance
- $\mathbf{n} \cdot \omega_i$ = cosine term，表示几何衰减
- $\Omega$ = 上半球积分域

### Monte Carlo Estimator

**直接光照估计**:

$$\hat{L}_d = \frac{f_r(\mathbf{x}, \omega_i, \omega_o) L_e(\mathbf{y}, \omega_i') (\mathbf{n} \cdot \omega_i)}{p(\omega_i)}$$

**Path Contribution 公式**:

$$L = L_e + \sum_{i=1}^{N} \left( \prod_{j=1}^{i} \frac{f_r(\mathbf{x}_j, \omega_{j-1}, \omega_j) (\mathbf{n}_j \cdot \omega_j)}{p(\omega_j)} \right) L_e(\mathbf{x}_{i+1})$$

其中：
- $N$ = path length (bounce count)
- $\mathbf{x}_j$ = $j$-th intersection point
- $\omega_j$ = direction from $\mathbf{x}_j$ to $\mathbf{x}_{j+1}$
- $p(\omega_j)$ = sampling PDF

### Russian Roulette 终止概率

为了无偏地终止无限长路径：

$$P_{continue} = \min\left(1, \frac{L(\mathbf{x})}{\tau}\right)$$

其中：
- $P_{continue}$ = 继续路径的概率
- $L(\mathbf{x})$ = 当前 throughput
- $\tau$ = 阈值参数

路径权重调整：

$$w_{new} = \frac{w_{old}}{P_{continue}}$$

---

## 6. 内存布局与 Buffer 结构

### 详细 Buffer 规格

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Wavefront Path Tracing Buffers                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Buffer 1: Primary/Extension Rays                                       │
│  ──────────────────────────────────────                                 │
│  struct Ray {                                                            │
│      float origin[3];    // 12 bytes                                   │
│      float padding;      // 4 bytes (alignment)                         │
│      float direction[3]; // 12 bytes                                   │
│      float padding;      // 4 bytes (alignment)                         │
│  };  // Total: 32 bytes per ray                                         │
│                                                                          │
│  Capacity: width × height × max_bounces                                  │
│  Example: 1920 × 1080 × 32 bytes = ~66 MB                               │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ R0 │ R1 │ R2 │ R3 │ R4 │ ... │ R(N-1) │ unused │ unused │     │    │
│  └────────────────────────────────────────────────────────────────┘    │
│       Ray Buffer (compacted, atomic append)                             │
│                                                                          │
│  Buffer 2: Intersection Results                                        │
│  ──────────────────────────────────────                                 │
│  struct Intersection {                                                   │
│      float t;            // 4 bytes (ray parameter)                     │
│      float u, v;         // 8 bytes (barycentric)                      │
│      uint primID;        // 4 bytes (triangle index)                   │
│      uint matID;         // 4 bytes (material index)                   │
│  };  // Total: 20 bytes → padded to 32 bytes for alignment             │
│                                                                          │
│  Buffer 3: Shadow Rays                                                  │
│  ──────────────────────────────────────                                 │
│  struct ShadowRay {                                                      │
│      float origin[3];    // 12 bytes                                   │
│      float tMax;         // 4 bytes (max distance)                     │
│      float direction[3]; // 12 bytes                                   │
│      uint pixelIdx;      // 4 bytes (contribution target)              │
│  };  // Total: 32 bytes                                                 │
│                                                                          │
│  Buffer 4: Path State (persistent across bounces)                      │
│  ──────────────────────────────────────                                 │
│  struct PathState {                                                      │
│      vec3 throughput;    // 12 bytes (accumulated BSDF product)       │
│      vec3 contribution;  // 12 bytes (final contribution)              │
│      uint pixelIdx;      // 4 bytes (target pixel)                     │
│      uint bounceCount;   // 4 bytes (path length)                      │
│      uint flags;         // 4 bytes (active, specular, etc.)           │
│  };  // Total: 48 bytes                                                 │
│                                                                          │
│  Buffer 5: Counters (atomic)                                            │
│  ──────────────────────────────────────                                 │
│  int activePathCount;    // Current iteration input count              │
│  int extensionCount;     // Output: new extension rays                │
│  int shadowRayCount;     // Output: new shadow rays                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 内存带宽计算

对于 1920×1080 分辨率：

```
Per iteration data transfer:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Read:                                                                   │
│  ──────────────────────────────────────────────────────────────         │
│  - Ray buffer:        32 bytes × N_rays                                │
│  - Intersection:      32 bytes × N_rays                                │
│  - Path states:       48 bytes × N_rays                                │
│  - Materials:         variable                                          │
│                                                                          │
│  Write:                                                                  │
│  ──────────────────────────────────────────────────────────────         │
│  - Intersection:      32 bytes × N_rays                                │
│  - Extension rays:    32 bytes × N_ext (≤ N_rays)                      │
│  - Shadow rays:       32 bytes × N_shadow (≤ N_rays × lights)          │
│  - Path states:       48 bytes × N_rays                                │
│                                                                          │
│  Total per iteration: ~200-300 bytes per active path                    │
│                                                                          │
│  For 1920×1080, first bounce:                                           │
│  2,073,600 rays × 256 bytes ≈ 531 MB                                    │
│                                                                          │
│  With 5 bounces average:                                                 │
│  ~2-3 GB/s bandwidth typical                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Wavefront vs Megakernel 对比

### 性能对比分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 Wavefront vs Megakernel 性能对比                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                     Megakernel              Wavefront                   │
│                  ┌─────────────────┐     ┌─────────────────┐           │
│  Divergence      │ HIGH             │     │ LOW              │           │
│                  │ (every branch)   │     │ (per phase)      │           │
│                  └─────────────────┘     └─────────────────┘           │
│                                                                          │
│  Kernel          │ Single kernel    │     │ Multiple kernels│           │
│  Launches        │ × 1 per frame    │     │ × 4 per bounce  │           │
│                  └─────────────────┘     └─────────────────┘           │
│                                                                          │
│  Register        │ Fixed for all    │     │ Optimized per   │           │
│  Pressure        │ code paths       │     │ phase           │           │
│                  └─────────────────┘     └─────────────────┘           │
│                                                                          │
│  Memory I/O      │ Low              │     │ High             │           │
│                  │ (local storage)  │     │ (global buffers) │           │
│                  └─────────────────┘     └─────────────────┘           │
│                                                                          │
│  Atomic Ops      │ None             │     │ Moderate        │           │
│                  └─────────────────┘     └─────────────────┘           │
│                                                                          │
│  Occupancy       │ Degrades with    │     │ Consistent      │           │
│                  │ path divergence  │     │ 100%            │           │
│                  └─────────────────┘     └─────────────────┘           │
│                                                                          │
│  Scalability     │ Poor for complex│     │ Good for complex│           │
│                  │ materials        │     │ scenes          │           │
│                  └─────────────────┘     └─────────────────┘           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Occupancy 公式详解

**NVIDIA Occupancy 定义**:

$$\text{Occupancy} = \frac{\text{Active Warps}}{\text{Maximum Warps}}$$

其中：
- $\text{Active Warps}$ = 当前 SM 上运行的 warp 数量
- $\text{Maximum Warps}$ = SM 支持的最大 warp 数量

**Warp 占用率计算**:

$$\text{Warp Count} = \left\lfloor \frac{\text{Registers per SM}}{\text{Registers per Thread} \times 32} \right\rfloor \times \text{SM Count}$$

$$\text{Warp Count} = \min\left( \left\lfloor \frac{\text{Shared Memory}}{\text{Shared per Block}} \right\rfloor \times \text{Blocks per SM}, \text{Max Warps} \right)$$

**Divergence 效率损失**:

$$\text{Efficiency} = \frac{\sum_{w \in Warps} \text{Active Threads in } w}{32 \times |Warps|}$$

对于完全随机分支：
$$\text{Expected Efficiency} = \frac{1}{2} = 50\%$$

---

## 8. 实现细节：Atomic Counter 优化

### 问题：Buffer Compaction

Shade 阶段会生成新 rays，但数量不确定。如何高效写入 buffer？

```
传统方案（有 gaps）:
┌─────────────────────────────────────────────────────────────────────────┐
│  Thread 0:  writes to buffer[0]     → Active                           │
│  Thread 1:  inactive                → Gap (wasted space)              │
│  Thread 2:  writes to buffer[1]     → Active                           │
│  Thread 3:  inactive                → Gap                              │
│  Thread 4:  writes to buffer[2]     → Active                           │
│  Thread 5:  writes to buffer[3]     → Active                           │
│  ...                                                                     │
│  Result: Sparse buffer, poor cache utilization                          │
└─────────────────────────────────────────────────────────────────────────┘

Wavefront 方案（compact, 使用 atomic）:
┌─────────────────────────────────────────────────────────────────────────┐
│  Thread 0:  idx = atomicAdd(&counter, 1)  → writes to buffer[0]         │
│  Thread 1:  inactive                                                   │
│  Thread 2:  idx = atomicAdd(&counter, 1)  → writes to buffer[1]         │
│  Thread 3:  inactive                                                   │
│  Thread 4:  idx = atomicAdd(&counter, 1)  → writes to buffer[2]         │
│  Thread 5:  idx = atomicAdd(&counter, 1)  → writes to buffer[3]         │
│  ...                                                                     │
│  Result: Compacted buffer, no gaps, better memory access patterns       │
└─────────────────────────────────────────────────────────────────────────┘
```

### GPU Atomic 操作性能

根据文章：

> "An atomic write is as expensive as an un-cached write to global memory."

NVIDIA GPU 上的 atomic 实现：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GPU Atomic 操作流水线                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Memory Hierarchy:                                                       │
│                                                                          │
│  ┌─────────────┐                                                        │
│  │   Thread    │─────┐                                                  │
│  │   Register  │     │                                                  │
│  └─────────────┘     │                                                  │
│                      ▼                                                  │
│              ┌─────────────┐                                            │
│              │   L1 Cache   │                                            │
│              │   (per SM)  │                                            │
│              └──────┬──────┘                                            │
│                     │                                                    │
│                     ▼                                                   │
│              ┌─────────────┐     Atomic Unit                            │
│              │   L2 Cache   │◄────────────────┐                         │
│              │   (shared)  │                 │                         │
│              └──────┬──────┘                 │                         │
│                     │                         │                         │
│                     ▼                         │                         │
│              ┌─────────────┐                 │                         │
│              │   Global     │─────────────────┘                         │
│              │   Memory     │                                           │
│              └─────────────┘                                            │
│                                                                          │
│  Atomic Cost Analysis:                                                  │
│  ─────────────────────                                                  │
│  - Best case:  L1 cache hit         → ~20-40 cycles                   │
│  - Typical case: L2 cache access    → ~100-200 cycles                 │
│  - Worst case:   Global memory      → ~400-600 cycles                │
│                                                                          │
│  Contention Impact:                                                     │
│  ─────────────────────                                                  │
│  - No contention:  1 atomic per thread → linear scaling               │
│  - High contention: serialization     → O(N) latency                  │
│                                                                          │
│  For wavefront path tracing:                                            │
│  - Contention is spread across many threads                            │
│  - Latency hidden by parallel execution                                 │
│  - Effective cost similar to global memory write                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 寄存器压力与 Kernel 分离优势

### Megakernel 寄存器问题

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Register Pressure 分析                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Megakernel (单一大 kernel):                                            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                                                                 │    │
│  │   Registers needed for ALL code paths:                         │    │
│  │   ────────────────────────────────────                         │    │
│  │   - BVH traversal variables                                     │    │
│  │   - BSDF evaluation variables                                   │    │
│  │   - Light sampling variables                                    │    │
│  │   - Path state variables                                        │    │
│  │   - Temporary variables for ALL branches                        │    │
│  │                                                                 │    │
│  │   Total: ~100-200 registers per thread                         │    │
│  │                                                                 │    │
│  │   Consequence:                                                  │    │
│  │   - Fewer threads can run concurrently                          │    │
│  │   - Lower occupancy                                            │    │
│  │   - Poor latency hiding                                        │    │
│  │                                                                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Wavefront (分离 kernel):                                                │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                                                                 │    │
│  │   Kernel 1 (Generate):                                          │    │
│  │   - Camera ray generation only                                  │    │
│  │   - ~20-30 registers                                            │    │
│  │   - Maximum threads per SM                                      │    │
│  │                                                                 │    │
│  │   Kernel 2 (Extend):                                            │    │
│  │   - BVH traversal only                                          │    │
│  │   - ~40-60 registers                                            │    │
│  │   - High occupancy                                              │    │
│  │                                                                 │    │
│  │   Kernel 3 (Shade):                                             │    │
│  │   - Material evaluation                                         │    │
│  │   - ~50-80 registers                                            │    │
│  │   - Variable based on material complexity                       │    │
│  │                                                                 │    │
│  │   Kernel 4 (Connect):                                           │    │
│  │   - Shadow ray test                                             │    │
│  │   - ~20-30 registers                                            │    │
│  │   - Maximum threads                                             │    │
│  │                                                                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Occupancy Comparison:                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Scenario          │ Megakernel │ Wavefront │ Improvement       │   │
│  ├────────────────────┼────────────┼───────────┼──────────────────┤   │
│  │  Simple scene      │    60%     │    95%    │     +35%         │   │
│  │  Complex material  │    30%     │    80%    │     +50%         │   │
│  │  High bounce count │    10%     │    95%    │     +85%         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. 代码架构示例 (Lighthouse 2 风格)

```cpp
// ============== 主渲染循环 ==============
void WavefrontPathTracer::render() {
    // Phase 1: Generate primary rays
    int pixelCount = width * height;
    generateEyeRays<<<blocks, threads>>>(
        rayBuffer, 
        camera, 
        width, 
        height
    );
    
    int activeRayCount = pixelCount;
    
    // Iteration loop
    for (int bounce = 0; bounce < maxBounces && activeRayCount > 0; bounce++) {
        
        // Phase 2: Extend - ray intersection
        extendRays<<<blocks, threads>>>(
            rayBuffer,
            intersectionBuffer,
            bvh,
            activeRayCount
        );
        
        // Reset counters for output
        extensionCounter = 0;
        shadowCounter = 0;
        
        // Phase 3: Shade - material evaluation
        shade<<<blocks, threads>>>(
            intersectionBuffer,
            rayBuffer,
            extensionRayBuffer,
            shadowRayBuffer,
            extensionCounter,
            shadowCounter,
            pathStateBuffer,
            materialBuffer,
            activeRayCount
        );
        
        // Phase 4: Connect - shadow rays
        if (shadowCounter > 0) {
            connectShadowRays<<<blocks, threads>>>(
                shadowRayBuffer,
                shadowResultBuffer,
                bvh,
                shadowCounter
            );
            
            // Apply shadow results to contributions
            applyShadowResults<<<blocks, threads>>>(
                shadowResultBuffer,
                pathStateBuffer,
                shadowCounter
            );
        }
        
        // Swap buffers for next iteration
        std::swap(rayBuffer, extensionRayBuffer);
        
        // Get new active ray count
        cudaMemcpyFromSymbol(&activeRayCount, extensionCounter, sizeof(int));
    }
    
    // Finalize: accumulate to framebuffer
    accumulateResults<<<blocks, threads>>>(
        pathStateBuffer,
        framebuffer,
        pixelCount
    );
}

// ============== CUDA Kernels ==============
__global__ void generateEyeRays(
    Ray* rays,
    Camera camera,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Pinhole camera model
    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;
    
    vec3 pixelPos = camera.topLeft 
                  + u * camera.right * camera.aspectRatio
                  + v * camera.down;
    
    vec3 direction = normalize(pixelPos - camera.position);
    
    rays[idx].origin = camera.position;
    rays[idx].direction = direction;
}

__global__ void extendRays(
    Ray* rays,
    Intersection* hits,
    BVHNode* bvh,
    int rayCount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rayCount) return;
    
    Ray ray = rays[idx];
    
    // BVH traversal (simplified)
    Intersection hit;
    hit.t = INFINITY;
    
    traverseBVH(ray, bvh, hit);
    
    hits[idx] = hit;
}

__global__ void shade(
    Intersection* hits,
    Ray* rays,
    Ray* extensionRays,
    Ray* shadowRays,
    int* extCounter,
    int* shadowCounter,
    PathState* states,
    Material* materials,
    int rayCount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rayCount) return;
    
    Intersection hit = hits[idx];
    PathState state = states[idx];
    
    if (hit.t == INFINITY) {
        // Escaped ray - add environment contribution
        state.contribution += state.throughput * envMapSample(ray.direction);
        state.active = false;
    }
    else if (materials[hit.matID].isEmissive) {
        // Hit light source
        state.contribution += state.throughput * materials[hit.matID].emission;
        state.active = false;
    }
    else {
        // Surface hit - continue path
        
        // Get material properties
        Material mat = materials[hit.matID];
        
        // Next Event Estimation (NEE)
        vec3 lightDir;
        float lightPdf;
        vec3 lightContrib = sampleLight(hit.position, lightDir, lightPdf);
        
        // Shadow ray
        Ray shadowRay;
        shadowRay.origin = hit.position + hit.normal * EPSILON;
        shadowRay.direction = lightDir;
        shadowRay.tMax = length(lightDir);
        
        int shadowIdx = atomicAdd(shadowCounter, 1);
        shadowRays[shadowIdx] = shadowRay;
        
        // BSDF sampling for path extension
        vec3 bounceDir;
        float bsdfPdf;
        vec3 f = sampleBSDF(mat, hit.normal, ray.direction, bounceDir, bsdfPdf);
        
        // Update throughput
        state.throughput *= f * dot(hit.normal, bounceDir) / bsdfPdf;
        
        // Russian roulette
        if (state.bounceCount > 3) {
            float continueProb = min(1.0f, luminance(state.throughput));
            if (random() > continueProb) {
                state.active = false;
            } else {
                state.throughput /= continueProb;
            }
        }
        
        // Extension ray
        if (state.active) {
            Ray extRay;
            extRay.origin = hit.position + hit.normal * EPSILON;
            extRay.direction = bounceDir;
            
            int extIdx = atomicAdd(extCounter, 1);
            extensionRays