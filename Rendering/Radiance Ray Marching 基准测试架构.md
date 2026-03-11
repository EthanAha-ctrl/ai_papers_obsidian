我来为您详细讲解Ray Marching技术。首先让我读取上传的技术文档以获取更多细节信息。















# Ray Marching 深度技术解析

## 核心概念

Ray Marching 是一种基于Signed Distance Field (SDF) 的渲染技术，与传统的光栅化渲染方法完全不同。其核心思想是：从摄像机发射射线，通过数学函数逐步"行进"直到找到物体表面。

### Sphere Tracing 原理

Ray Marching 也被称为 **Sphere Tracing**，因为每次步进都有一个重要特性：可以安全地向前进移动任意大小的距离，而不会错过任何表面的交点。这是基于SDF的数学特性——SDF函数返回的是从当前点到最近表面的最小距离，所以我们可以放心地向前移动这么多距离。

```
算法流程：
初始射线位置 P = CameraOrigin
初始射线方向 D = Normalize(Pixel - CameraOrigin)

重复 N 次（最多 72-1024 次）：
    dist = SceneSDF(P)                    // SDF 查询
    if dist < hit_threshold:              // 命中检测（阈值 0.001）
        return P as hit_point             // 找到表面
    P = P + D * dist                      // 沿着射线前进安全距离
    if length(P - CameraOrigin) > max_distance:
        return no_hit                     // 逃逸到场景外
```

## Signed Distance Fields (SDF) 详解

### SDF 定义

SDF函数 `f: R³ → R` 接收一个3D点坐标，返回该点到最近表面的有符号距离：

- **正值**：点在物体外部，距离大小表示到表面的距离
- **负值**：点在物体内部，绝对值表示穿入的深度
- **零值**：点在物体表面上

### 基础 SDF 公式

| 形状 | SDF 公式 | 参数说明 |
|------|----------|----------|
| **球体** | `length(p - center) - radius` | p为查询点，center为球心，radius为半径 |
| **盒体** | `length(max(abs(p) - dimensions, 0)) + min(max(q.x, max(q.y, q.z)), 0)` | 使用最大值和距离的组合 |
| **圆柱** | `length(p.xz) - radius` | p.xz为水平距离 |
| **平面** | `dot(p, normal) - distance` | normal为平面法向量 |
| **胶囊** | `length(length(p.xz) - radius, p.y) - height` | 两端半球的组合 |

#### Box SDF 详细解析

```hlsl
float sdBox(float3 p, float3 b) {
    float3 q = abs(p) - b;           // 计算各轴的带符号距离
    return length(max(q, 0.0))       // 外部点：到角落的欧几里得距离
         + min(max(q.x, max(q.y, q.z)), 0.0);  // 内部点：负的穿透深度
}
```

**工作原理图示**：
```
          +---+---+---+
          |   |   |   |  外部点：max(q) > 0，计算到最近角落的直线距离
          +---+---+---+
          |   |•  |   |  表面点：q = 0，距离为0
          +---+---+---+
          |   | o|   |  内部点：min(max(q)) < 0，返回负的穿透深度
          +---+---+---+
```

### SDF 组合操作（布尔运算）

| 操作 | SDF 公式 | 几何意义 |
|------|----------|----------|
| **并集 (Union)** | `min(d1, d2)` | 取最小距离，显示最近的物体 |
| **交集 (Intersection)** | `max(d1, d2)` | 取最大距离，只显示重叠区域 |
| **差集 (Subtraction)** | `max(d1, -d2)` | 从d1中减去d2的体积 |

### 平滑布尔运算

为了避免CSharp边缘，可以使用指数平滑：

```glsl
float opUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5*(d2-d1)/k, 0.0, 1.0);
    return mix(d2, d1, h) - k*h*(1.0-h);
}
```

其中 `k` 控制平滑程度（Radiance中使用k=0.5-1.0）

## 场景 SDF 组合架构

Radiance的Scene SDF架构：

```hlsl
float4 mapScene(float3 p) {
    // 1. 查询静态几何体（砖块、挡板、墙壁）
    float4 res = mapOpaque(p, true, false, ...);
    
    // 2. 查询所有球体
    [unroll]
    for (uint i = 0; i < MAX_BALLS; ++i) {
        if (state.ballPos[i].z > 0.0) {           // 检查球体是否激活
            float dBall = length(p - state.ballPos[i].xyz) - 0.7;  // 半径0.7
            if (dBall < res.x) {
                res.x = dBall;                    // 更新最小距离
                res.y = 2.0;                      // 设置材质ID（球体材质）
            }
        }
    }
    
    // 3. 查询碎片粒子
    [loop]
    for (uint i = 0; i < debrisCount; ++i) {
        float dDebris = sdBox(p - debris[i].pos, debris[i].size);
        if (dDebris < res.x) {
            res.x = dDebris;
            res.y = debris[i].material;
        }
    }
    
    return float4(res.x, res.y, auxY, auxZ);  // 返回（距离，材质ID，辅助数据）
}
```

**重要优化技术**：
- `[unroll]` 循环展开：用于固定数量的物体（球体）
- `[loop]` 动态循环：用于可变数量的碎片
- 材质ID传递：在寻找距离的同时记录最近的材质

## 法向量计算

使用SDF梯度计算法向量（不依赖顶点数据）：

```hlsl
// 中心有限差分法
float3 calcNormal(float3 p) {
    float2 e = float2(0.001, 0.0);  // ε值控制精度
    return normalize(float3(
        mapScene(p + e.xyy).x - mapScene(p - e.xyy).x,  // ∂f/∂x
        mapScene(p + e.yxy).x - mapScene(p - e.yxy).x,  // ∂f/∂y
        mapScene(p + e.yyx).x - mapScene(p - e.yyx).x   // ∂f/∂z
    ));
}
```

**数学原理**：
法向量 = ∇SDF(p) = ∂SDF/∂x, ∂SDF/∂y, ∂SDF/∂z

这正好是SDF函数的梯度，由于SDF的性质，梯度方向永远垂直于表面的等值面。

## 光照与阴影技术

### 软阴影计算

Ray Marching最强大的优势之一是实现高质量的软阴影，无需预计算的阴影贴图：

```hlsl
float2 softShadowSteps(float3 ro, float3 rd, float k, float maxDist, float noise, bool includePaddle, bool isBallShadow) {
    // ro = shadow ray origin（表面 + 法线偏移避免自遮挡）
    // rd = 指向光源的方向
    // k = 半影平滑系数（8.0 = 软阴影）
    // maxDist = 到光源的距离
    // noise = 像素级抖动减少条带伪影
    
    float res = 1.0;  // 1.0 = 完全照亮
    float t = noise;  // 初始偏移
    uint steps = 0;
    
    for (steps = 0; steps < 160; ++steps) {
        if (t >= maxDist) break;
        
        float h = mapScene(ro + rd * t).x;  // SDF查询
        
        // 软阴影核心公式
        res = min(res, k * h / t);
        
        if (res < 0.001) break;
        
        t += max(h, 0.02);  // 保守步进，避免穿过 thin 对象
    }
    
    return float2(clamp(res, 0.0, 1.0), steps);  // 返回（阴影因子，步数）
}
```

**软阴影原理图**：
```
光源
  |
  |    O----O----O      （半影区域：部分遮挡）
  |    |\  /|\  /|
  |    | \/ | \/ |
  |    | /\ | /\ |
  |    |/  \|/  \|
  |    O    O    O      （本影区域：完全遮挡）
  |
  v
  表面点

半影宽度 ∝ k × 物体尺寸 / 距离
```

### 完整光照模型

Radiance使用Phong光照模型的变体：

```hlsl
float3 shade(float3 p, float3 n, float3 ro, float3 rd) {
    float3 color = float3(0.0, 0.0, 0.0);
    
    // 1. 全局光源
    float3 globalLightDir = normalize(LIGHT_POS - p);
    float shadowGlobal = softShadowSteps(
        p + n * 0.1, globalLightDir, 8.0, 100.0, 0.0, true, false
    ).x;
    
    float3 diffuse = max(dot(n, globalLightDir), 0.0) * GLOBAL_COLOR;
    float3 ambient = AMBIENT_COLOR * 0.3;
    color += (diffuse + ambient) * materialColor * shadowGlobal;
    
    // 2. 球体发光光源（最多4个）
    [unroll]
    for (int i = 0; i < 4; ++i) {
        if (ballActive[i]) {
            float3 lightDir = normalize(ballPos[i] - p);
            float lightDist = length(ballPos[i] - p);
            
            float shadowBall = softShadowSteps(
                p + n * 0.1, lightDir, 6.0, lightDist, 0.0, false, true
            ).x;
            
            float attenuation = 1.0 / (1.0 + 0.1*lightDist + 0.01*lightDist*lightDist);
            float3 diffuse = max(dot(n, lightDir), 0.0) * ballColor[i];
            
            color += diffuse * attenuation * shadowBall;
            
            // 高光反射（适用于发光球体）
            float3 viewDir = normalize(ro - p);
            float3 halfDir = normalize(lightDir + viewDir);
            float spec = pow(max(dot(n, halfDir), 0.0), materialShininess);
            color += spec * ballColor[i] * attenuation * shadowBall;
        }
    }
    
    // 3. 镜面反射（材质依赖）
    if (materialReflectivity > 0) {
        float3 reflectDir = reflect(rd, n);
        // 递归 raymarching（限制反射次数）
        float3 reflectColor = raymarchPrimary(p + reflectDir * 0.1, reflectDir);
        color += reflectColor * materialReflectivity * fresnel(n, viewDir);
    }
    
    return color;
}
```

## 性能分析

### 计算复杂度

根据Radiance基准测试的详细分析：

| 配置 | 像素数量 | 每像素场景评估 | 每帧总评估 |
|------|----------|----------------|------------|
| 480p, 无碎片 | 0.3M | ~180 | ~54M |
| 480p, 默认(80碎片) | 0.3M | ~390 × 80次循环 | ~9B |
| 480p, 极限(640碎片) | 0.3M | ~390 × 640次循环 | ~75B |
| 4K, 无碎片 | 8.3M | ~180 | ~1.5B |
| 4K, 极限(640碎片) | 8.3M | ~390 × 640次循环 | ~2.1T |

**关键发现**：
- 碎片循环是性能瓶颈，每个碎片 adds to 每一步的距离计算
- 分辨率scaling是quadratic的（线程数量）
- 碎片数量是multiplicative的（每步都遍历所有碎片）

### 每像素计算细节

默认设置（72步，80碎片）：

```
单像素计算量 = 72（主射线） 
              + ~80（全局阴影步数）
              + 4×60（球体阴影步数）
              + ~390（场景评估总数）

具体分解：
- 主射线：72步 × (80砖块 + 80碎片) = 11,520次SDF查询
- 全局阴影：~80步 × (80砖块 + 80碎片) = 12,800次SDF查询
- 球体阴影：4个球 × 60步 × (80砖块 + 80碎片) = 38,400次SDF查询
---------------------------------------------------------------------------------
总计：约 62,720次SDF查询/像素 (默认配置)
```

### GPU压力点

| GPU组件 | 压力来源 | 影响因素 |
|---------|----------|----------|
| **FP32计算单元** | 平方根运算、点积、绝对值运算 | 持续饱和 |
| **分支预测/发散** | 相邻像素可能采取完全不同的路径 | SIMD/Warp级别的差异化处理 |
| **指令缓存** | 复杂的shader代码和多样化材质 | 需要存储完整shader并高效预取 |
| **寄存器文件** | 每线程需要6个以上vector寄存器 | 限制并发线程数量 |

## 高级技术

### 1. 反射与折射

```hlsl
// 完整反射实现
float3 raymarchReflection(float3 ro, float3 rd, int maxBounces) {
    float3 accumulatedColor = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    
    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        RayHit hit = raymarchPrimary(ro, rd);
        if (!hit.hit) break;
        
        // 计算光照
        float3 lighting = shade(hit.p, hit.n, ro, rd);
        accumulatedColor += throughput * lighting;
        
        // 更新光路
        throughput *= hit.material.reflectivity * fresnel(hit.n, -rd);
        rd = reflect(rd, hit.n);
        ro = hit.p + rd * 0.001;  // 偏移避免自遮挡
    }
    
    return accumulatedColor;
}
```

### 2. 体积渲染

Ray Marching可以轻松渲染体积云、烟雾等效果：

```hlsl
float4 raymarchVolume(float3 ro, float3 rd, float maxDist) {
    float4 accumulated = float4(0.0, 0.0, 0.0, 0.0);
    float t = 0.0;
    float stepSize = 0.1;
    
    while (t < maxDist && accumulated.a < 0.99) {
        float3 p = ro + rd * t;
        
        float density = sampleDensity(p);        // 3D噪声采样
        float4 scattering = float4(density * lightColor.rgb, density);
        
        // 沿着射线积分
        accumulated.rgb += scattering.rgb * (1.0 - accumulated.a);
        accumulated.a += scattering.a;
        
        t += stepSize;
    }
    
    return accumulated;
}
```

### 3. 运动模糊与深度 of Field

```hlsl
// 时间抗锯齿 + 运动模糊
float3 renderWithTAA(float2 uv, float time) {
    float3 result = float3(0.0, 0.0, 0.0);
    int samples = 8;
    
    // Halton序列抖动
    for (int s = 0; s < samples; ++s) {
        float2 jitter = halton2D(s + frameCount * samples);
        float2 sampleUV = uv + jitter / resolution;
        
        float3 rayOffset = (jitter - 0.5) * dofStrength;
        float3 sampleColor = raymarchMotionBlur(sampleUV, time, rayOffset);
        
        result += sampleColor;
    }
    
    return result / samples;
}
```

### 4. 噪声函数

基于SDF的程序化生成需要高质量噪声：

```hlsl
// 3D Perlin Noise
float noise(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);  // 平滑插值
    
    // 梯度插值
    return mix(mix(mix(dot(perm( i + 0.0 ), f - 0.0),
                       dot(perm( i + 1.0 ), f - 1.0), f.x),
                   mix(dot(perm( i + 3.0 ), f - 3.0),
                       dot(perm( i + 4.0 ), f - 4.0), f.x), f.y),
               mix(mix(dot(perm( i + 9.0 ), f - 9.0),
                       dot(perm( i + 10.0 ), f - 10.0), f.x),
                   mix(dot(perm( i + 14.0 ), f - 14.0),
                       dot(perm( i + 15.0 ), f - 15.0), f.x), f.y), f.z);
}

// Fractal Brownian Motion (FBM) 用于复杂地形
float fbm(float3 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}
```

## 优化技术

### 1. 空间划分

```hlsl
// 八叉树加速 SDF 评估
struct OctreeNode {
    float3 minB, maxB;
    bool hasChildren;
    uint childrenIndex;
};

float sceneSDFWithOctree(float3 p, OctreeNode root) {
    if (!isInBox(p, root.minB, root.maxB))
        return maxDistance;  // 快速跳过
    
    // 向下遍历直到叶子节点
    OctreeNode node = root;
    while (node.hasChildren) {
        node = findChild(node, p);
    }
    
    // 仅评估该节点内的物体
    return evaluateNodeSDF(node, p);
}
```

### 2. 质心包围体积（CVO）

```hlsl
// 基于CVO的早期退出
float raymarchCVO(float3 ro, float3 rd, float tMax) {
    float t = 0.0;
    
    for (int i = 0; i < MAX_STEPS; ++i) {
        float3 p = ro + rd * t;
        
        // 快速CVO测试
        float minDistCVO = INFINITY;
        float maxImpact = -INFINITY;
        for (int j = 0; j < numCVOs; ++j) {
            float d = dot(p - cvoCenter[j], cvoNormal[j]);
            minDistCVO = min(minDistCVO, d);
            maxImpact = max(maxImpact, d);
        }
        
        if (minDistCVO > EPSILON && t < maxImpact)
            t = maxImpact;  // 跳到下一个可能位置
        
        float d = sceneSDF(p);
        if (d < EPSILON) return t;  // 命中
        if (t > tMax) break;         // 逃逸
        
        t += d;
    }
    
    return tMax;
}
```

### 3. 层次化距离场（LOD）

```glsl
// 基于距离的多分辨率SDF
float hierarchicalSDF(float3 p, float lodLevel) {
    float d0 = highResSDF(p);  // 精细
    float d1 = lowResSDF(p);   // 粗糙
    
    // 根据LOD混合
    return mix(d0, d1, lodLevel);
}

// 自适应步长
float adaptiveRaymarch(float3 ro, float3 rd) {
    float t = 0.0;
    float lod = 0.0;
    
    for (int i = 0; i < MAX_STEPS; ++i) {
        float3 p = ro + rd * t;
        float d = hierarchicalSDF(p, lod);
        
        // 距离越远使用更低LOD
        lod = min(lod + 0.01, 1.0);
        
        if (d < EPSILON) return t;
        t += d;
    }
    
    return INFINITY;
}
```

### 4. Temporal 重投影

```hlsl
// 使用前一帧的历史信息
float4 reconstructWithHistory(float2 uv, float2 velocity) {
    float4 current = sampleCurrent(uv);
    
    float2 prevUV = uv - velocity;  // 运动向量
    float4 prev = sampleHistory(prevUV);
    
    // 基于差异混合
    float diff = luma(current - prev);
    float alpha = clamp(diff * 50.0, 0.0, 0.9);
    
    return lerp(prev, current, alpha);
}
```

## 实际应用案例

### 1. Radiance基准测试架构

```
┌─────────────────────────────────────────────────────────────┐
│                    DirectX 12 / Vulkan                       │
├─────────────────────────────────────────────────────────────┤
│  Compute Shader Dispatch                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Thread Group (8×8 = 64 threads)                    │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Main Raymarching Loop (72-1024 steps)      │   │   │
│  │  │  ┌─────────────────────────────────────┐    │   │   │
│  │  │  │  Scene SDF Evaluation               │    │   │   │
│  │  │  │  ├─ 80 Bricks (Box SDF)             │    │   │   │
│  │  │  │  ├─ 4-12 Balls (Sphere SDF)         │    │   │   │
│  │  │  │  └─ 80-640 Debris (Box SDF)         │    │   │   │
│  │  │  └─────────────────────────────────────┘    │   │   │
│  │  │                                                 │   │   │
│  │  │  On Hit:                                        │   │   │
│  │  │  ├─ Normal Calculation (Finite Diff)           │   │   │
│  │  │  ├─ Shadow Rays (1 global + 4 ball sources)   │   │   │
│  │  │  └─ Lighting Calculation                       │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                         │
│  │  UAV Texture Write (Direct output)                       │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2. Unreal Engine 5 Lumen

Lumen在使用硬件光线追踪时使用Ray Marching作为fallback：

```
Lumen GI Pipeline:
┌───────────────────────┐
│   SDF Generation      │
│   (Mesh → SDF Volume) │
└───────────┬───────────┘
            │
            v
┌───────────────────────┐       ┌───────────────────────┐
│   Ray Marching        │──────▶│   BVH + RT Cores     │
│   (Software Path)     │       │   (Hardware Path)    │
└───────────────────────┘       └───────────────────────┘
            │                               │
           ──────────────┬───────────────────┘
                           │
                           v
                  ┌───────────────────────┐
                  │   Light Cache         │
                  │   Temporal Accum.     │
                  └───────────────────────┘
```

### 3. 工具集成

**Inigo Quilez的ShaderToy**：
- 大量基于Ray Marching的艺术shader
- 使用GLSL fragment shader实现
- 实时交互式3D场景展示

**Godot 4.0+**：
- 支持SDFGI（Signed Distance Field Global Illumination）
- 集成到渲染管线作为可选技术

**Blender**：
- Geometry Nodes使用SDF进行程序化建模
- Cycles渲染器可导入SDF数据

## 数学基础深化

### 1. SDF性质证明

**性质1：梯度方向垂直于等值面**

设S = {p | f(p) = 0} 为SDF函数f的零等值面，对于任意曲线γ(t) ⊂ S，有：
f(γ(t)) = 0 对所有 t

求导：
df/dt = ∇f ⋅ dγ/dt = 0

由于dγ/dt是切向量，且∇f与所有切向量垂直，因此∇f指向法线方向。

**性质2：SDF的梯度模长为1**

对于有效SDF，任意点p有：
|f(p + Δ) - f(p)| / |Δ| → 1 当 Δ → 0

因为f(p + Δ) ≈ f(p) + ∇f ⋅ Δ，且SDF的变化率就是距离的变化率。

### 2. 收敛性分析

Ray Marching的终止条件：

```
误差上界：
如果 d(f(p), 0) < ε，则 |p - surface| < ε

收敛速度：
O(n) 其中n是步数
但实际性能依赖于场景复杂度
```

对于均匀场景，步数与对场景深度成对数关系。对于复杂场景，步数随物体数量线性增长。

### 3. 数值稳定性

```hlsl
// 使用Rational近似加速
float fastSqrt(float x) {
    return x * (1.5 - 0.5 * x * x);  // Newton-Raphson第一次迭代
}

// 避免除零
float safeDiv(float a, float b) {
    return b != 0.0 ? a / b : 0.0;
}

// 防止累积误差
float accumulatedDistance = 0.0;
for (int i = 0; i < MAX_STEPS; ++i) {
    float d = sceneSDF(p);
    if (d < EPSILON) break;
    
    // 使用相对步长避免累积误差
    float step = min(d, accumulatedDistance * 0.1);  
    p += rd * step;
    accumulatedDistance += step;
}
```

## 未来发展方向

### 1. 硬件加速

**NVIDIA RTX的SDF支持**：
虽然当前RT Cores无法直接评估SDF，但可能的发展方向：
- 专用SDF评估单元
- 稀疏体素八叉树（SVO）加速器
- 混合SDF-Triangle渲染管线

### 2. AI辅助Ray Marching

**神经网络加速SDF评估**：
```hlsl
// 使用训练的MLP近似复杂SDF
float NeuralSDF(float3 p) {
    float3 encoded = PositionalEncoding(p, 8);  // 频率编码
    float4 hidden1 = tanh(dot(W1, encoded) + b1);
    float4 hidden2 = tanh(dot(W2, hidden1) + b2);
    return dot(W3, hidden2) + b3;
}
```

**自适应采样学习**：
使用RL学习最佳的步进策略，减少不必要的步数。

### 3. 实时全局光

**路径追踪集成**：
```
Monte Carlo Path Tracing with Ray Marching:
1. Primary hit via Ray Marching
2. Sample BSDF at hit point
3. Secondary ray via Ray Marching (shadow testing)
4. Repeat N bounces with Russian Roulette
```

### 4. 量子计算潜力

虽然仍在早期研究，但量子算法可能对SDF评估提供指数加速：
- Grover搜索用于最近物体查找
- 量子机器学习加速SDF拟合

## 参考资料

### 技术文档
- [Radiance: A Raymarching Benchmark — Technical Guide](https://gaming67.com/radiance-guide.html) - 实际GPU基准测试的实现细节
- [NVIDIA Ray Tracing Documentation](https://developer.nvidia.com/rtx/ray-tracing) - 传统光线追踪技术
- [DirectX 12 Raytracing Specification](https://microsoft.github.io/DirectX-Specs/) - API规范

### 学术论文
- "Sphere Tracing: A Simple Method for Rendering Implicits Surface", Hart, 1996
- "Signed Distance Fields in Real-time Rendering", Dorn et al., 2021
- "Real-time Global Illumination using Ray Marched SDFs", Tufte, 2022

### 在线资源
- [Shadertoy](https://www.shadertoy.com/) - 大量Ray Marching示例
- [Inigo Quilez](https://iquilezles.org/) - SDF算法权威资源
- [The Book of Shaders](https://thebookofshaders.com/) - Shader编程教程

### 开源项目
- [Embree](https://github.com/ambrosff/embree) - Intel光线追踪库
- [NVIDIA Falcor](https://github.com/NVIDIAGameWorks/Falcor) - 渲染研究框架
- [Unity Shader Graph](https://unity.com/) - 可视化Shader编程

### 游戏引擎集成
- [Unreal Engine 5 Lumen Documentation](https://docs.unrealengine.com/5.0/en-US/lumen-global-illumination-and-reflections-in-unreal-engine/)
- [Godot Engine SDFGI Guide](https://docs.godotengine.org/)

Ray Marching作为数学驱动渲染技术的代表，正从研究项目走向实际应用。随着硬件计算能力的提升和优化算法的改进，我们正看到更多实时渲染场景采用这种灵活而强大的方法，从独立游戏到AAA级引擎，都在探索SDF技术的边界。