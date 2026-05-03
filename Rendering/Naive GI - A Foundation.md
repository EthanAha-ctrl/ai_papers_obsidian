# Naive GI: A Foundation - 深度技术解析

这篇文章是关于**实时全局光照** 的基础教程，为后续更高级的 **Radiance Cascades** 算法铺路。让我从第一性原理出发，深入解析每个技术模块。

---

## 1. 全局光照的核心概念

### 1.1 什么是全局光照？

**直接光照**: 光源直接照射到表面
**全局光照**: 光线在场景中多次反弹，间接照亮其他表面

当我们打开房间的灯，光线会：
1. 直接照亮墙壁、地板、天花板
2. 从这些表面反弹，间接照亮其他物体
3. 产生柔和的阴影和色彩溢出

### 1.2 传统方法的局限性

| 方法 | 优点 | 缺点 |
|------|------|------|
| Ray Tracing | 物理准确 | 计算量大，无法实时 |
| Path Tracing | 高质量GI | 需要大量样本，噪声严重 |
| 光照贴图 | 运行时高效 | 只能静态，预计算耗时 |

**Radiance Cascades** 的突破在于：
- ✅ 实时运行
- ✅ 无噪声
- ✅ 消费级硬件可运行

---

## 2. 可绘制表面

### 2.1 SDF (Signed Distance Function) 基础

**核心思想**: 用距离函数描述几何形状

对于任意像素点 `p`，计算其到形状表面的**有向距离**：
- 距离 > 0：在形状外部
- 距离 < 0：在形状内部
- 距离 = 0：正好在表面上

### 2.2 SDF 线段绘制

文章使用的核心函数：

```glsl
float sdfLineSquared(vec2 p, vec2 from, vec2 to) {
  vec2 toStart = p - from;
  vec2 line = to - from;
  float lineLengthSquared = dot(line, line);
  float t = clamp(dot(toStart, line) / lineLengthSquared, 0.0, 1.0);
  vec2 closestVector = toStart - line * t;
  return dot(closestVector, closestVector);
}
```

**数学原理详解**：

```
        to
         •
        /
       /
      •----• p (当前像素)
     /
    /
   •
 from
```

给定线段 $\vec{AB}$ (从 `from` 到 `to`) 和点 $P$：

1. **向量表示**：
   - $\vec{AB} = \text{to} - \text{from}$
   - $\vec{AP} = p - \text{from}$

2. **投影参数 $t$**：
   $$t = \frac{\vec{AP} \cdot \vec{AB}}{\vec{AB} \cdot \vec{AB}}$$
   
   这是点 $P$ 在线段 $AB$ 上的**参数化位置**

3. **Clamp 到 [0, 1]**：
   - $t = 0$：最近点是 `from`
   - $t = 1$：最近点是 `to`
   - $t \in (0,1)$：最近点在线段中间

4. **最近点**：
   $$\text{closest} = \text{from} + t \cdot \vec{AB}$$

5. **距离平方**：
   $$d^2 = \|P - \text{closest}\|^2$$

### 2.3 优化技巧：避免 sqrt

传统计算：
```glsl
float dist = distance(p, closest); // 内部调用 sqrt
if (dist < radius) { ... }
```

优化版本：
```glsl
float distSquared = dot(closest, closest); // 无 sqrt
if (distSquared < radiusSquared) { ... }
```

**为什么有效？**
- $\text{distance}(a, b) = \sqrt{\text{dot}(a-b, a-b)}$
- 比较操作保持单调性：$a < b \Leftrightarrow a^2 < b^2$ (当 $a, b \geq 0$)

**性能提升**: `sqrt` 是昂贵的GPU操作，避免它可以显著提速。

---

## 3. Raymarching (光线步进)

### 3.1 基本原理

**核心思想**: 从每个像素出发，向多个方向发射"光线"，沿途收集光照信息。

```
        ↖  ↑  ↗
          \ |
           \|/
     ← ----- • ----- →
           /|\
          / |
        ↙  ↓  ↘
```

### 3.2 算法伪代码

```glsl
vec4 raymarch() {
    vec4 radiance = vec4(0.0);
    float oneOverRayCount = 1.0 / float(rayCount);
    float tauOverRayCount = TAU * oneOverRayCount; // 角度间隔
    
    float noise = rand(vUv); // 随机偏移，避免规律性伪影
    
    for(int i = 0; i < rayCount; i++) {
        float angle = tauOverRayCount * (float(i) + noise);
        vec2 direction = vec2(cos(angle), -sin(angle));
        
        vec2 sampleUv = vUv;
        
        for(int step = 0; step < maxSteps; step++) {
            sampleUv += direction / size;
            
            if (outOfBounds(sampleUv)) break;
            
            vec4 sampleLight = texture(sceneTexture, sampleUv);
            if (sampleLight.a > 0.1) {
                radiance += sampleLight;
                break;
            }
        }
    }
    
    return radiance * oneOverRayCount; // 平均
}
```

### 3.3 数学细节

**角度计算**：
- $\tau = 2\pi$ (一周)
- $\Delta\theta = \frac{2\pi}{N}$ (N条光线，均匀分布)
- 第 $i$ 条光线角度：$\theta_i = i \cdot \Delta\theta + \text{noise}$

**方向向量**：
$$\vec{d}_i = (\cos\theta_i, -\sin\theta_i)$$
(负号是因为屏幕坐标系Y轴向下)

**步进公式**：
$$\vec{uv}_{t+1} = \vec{uv}_t + \frac{\vec{d}_i}{\text{size}}$$

### 3.4 噪声的作用

**问题**: 固定角度导致规律性伪影

```
* * * * *
 * * * *
* * * *    <- 规律性条纹
 * * * *
* * * * *
```

**解决方案**: 每像素添加随机偏移

```glsl
float rand(vec2 uv) {
    return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
}
```

噪声打破规律性，视觉上更自然。

### 3.5 算法复杂度分析

设：
- $P$ = 像素数量
- $R$ = 每像素光线数
- $S$ = 最大步进数

**朴素Raymarching复杂度**：
$$O(P \cdot R \cdot S)$$

对于 512×512 画布：
- $P = 262,144$
- $R = 8$
- $S = 256$
- 总计：约 **5.37 亿次** 操作/帧

这就是为什么朴素方法在较大画布上性能堪忧。

---

## 4. Jump Flood Algorithm (JFA)

### 4.1 问题定义

我们需要计算**距离场**：
- 对每个像素，找到最近的"填充"像素
- 存储距离值

**直接方法**: 对每个像素，扫描所有填充像素 → $O(N^2)$

**JFA方法**: $O(N \log N)$

### 4.2 算法核心思想

**核心直觉**: 不要从每个像素寻找最近点，而是让填充点的信息"向外传播"。

```
Pass 1 (步长=4):
○ ○ ○ ○ ○ ○ ○ ○      ● ○ ○ ○ ○ ○ ○ ●
○ ○ ○ ○ ○ ○ ○ ○      
○ ○ ○ ○ ○ ○ ○ ○      信息传播到远处
○ ○ ○ ○ ● ○ ○ ○      
○ ○ ○ ○ ○ ○ ○ ○      
○ ○ ○ ○ ○ ○ ○ ○      
○ ○ ○ ○ ○ ○ ○ ○      
● ○ ○ ○ ○ ○ ○ ○     

Pass 2 (步长=2):      
逐步精细化传播        
        
Pass 3 (步长=1):      
最终收敛到精确距离
```

### 4.3 详细算法步骤

**Step 1: 种子纹理**

将填充像素的UV坐标编码到颜色通道：

```glsl
float alpha = texture(surfaceTexture, vUv).a;
gl_FragColor = vec4(vUv * alpha, 0.0, 1.0);
// R = uv.x (如果有填充)
// G = uv.y (如果有填充)
// A = 1.0
```

**Step 2: 多Pass传播**

```javascript
const passes = Math.ceil(Math.log2(Math.max(width, height)));

for (let i = 0; i < passes; i++) {
    // 步长从大到小：N/2 → N/4 → ... → 1
    let offset = Math.pow(2, passes - i - 1);
    
    // 执行JFA pass
    // ...
}
```

**Step 3: JFA Shader**

```glsl
vec4 nearestSeed = vec4(-2.0); // 初始化为远点
float nearestDist = 999999.9;

for (float y = -1.0; y <= 1.0; y += 1.0) {
    for (float x = -1.0; x <= 1.0; x += 1.0) {
        // 采样 3x3 邻域（偏移量由步长控制）
        vec2 sampleUV = vUv + vec2(x, y) * uOffset / resolution;
        
        if (outOfBounds(sampleUV)) continue;
        
        vec4 sampleValue = texture(inputTexture, sampleUV);
        vec2 sampleSeed = sampleValue.xy; // 存储的最近点UV
        
        if (sampleSeed.x != 0.0 || sampleSeed.y != 0.0) {
            vec2 diff = sampleSeed - vUv;
            float dist = dot(diff, diff); // 距离平方
            
            if (dist < nearestDist) {
                nearestDist = dist;
                nearestSeed = sampleValue;
            }
        }
    }
}

gl_FragColor = nearestSeed; // 输出最近点的UV
```

### 4.4 数学证明：为什么 log(N) 次足够？

**关键观察**：每次Pass后，传播半径翻倍。

设纹理尺寸为 $N$，初始步长为 $N/2$：

| Pass | 步长 | 累计传播距离 |
|------|------|-------------|
| 0 | $N/2$ | $N/2$ |
| 1 | $N/4$ | $N/2 + N/4 = 3N/4$ |
| 2 | $N/8$ | $3N/4 + N/8 = 7N/8$ |
| ... | ... | ... |
| $k$ | $N/2^{k+1}$ | $\approx N(1 - 1/2^{k+1})$ |

经过 $\log_2(N)$ 次Pass，覆盖整个纹理。

**几何直觉**：
```
Pass k=0:  ●----------●----------●  (步长=N/2)
           ↓          ↓          ↓
Pass k=1:  ●----●----●----●----●  (步长=N/4)
           ↓    ↓    ↓    ↓    ↓
Pass k=2:  ●--●--●--●--●--●--●  (步长=N/8)
```

每个点的信息以指数速度传播。

### 4.5 Ping-Pong Buffer 技术

**问题**: 并行计算中，不能同时读写同一纹理

**解决**: 双缓冲交替

```javascript
let [renderA, renderB] = this.jfaRenderTargets;
let currentInput = inputTexture;
let currentOutput = renderA;

for (let i = 0; i < passes; i++) {
    // 从 currentInput 读取，写入 currentOutput
    render(currentInput, currentOutput);
    
    // 交换
    currentInput = currentOutput.texture;
    currentOutput = (currentOutput === renderA) ? renderB : renderA;
}
```

```
Frame 1: Input → A → B → A → B → Output
         读A    写B  读B  写A
         
(A和B交替作为输入输出)
```

---

## 5. Distance Field (距离场)

### 5.1 从JFA到距离场

JFA输出：每个像素存储**最近填充点的UV**

距离场转换：
```glsl
vec2 nearestSeed = texture(jfaTexture, vUv).xy;
float dist = clamp(distance(vUv, nearestSeed), 0.0, 1.0);
gl_FragColor = vec4(vec3(dist), 1.0);
```

### 5.2 距离场的威力

**关键性质**: 如果我知道到最近障碍物的距离 $d$，我可以安全地跳 $d$ 步而不碰撞！

```
当前位置: ○
最近障碍: ●
距离: d = 5

安全跳跃: ○----→○----→○----→●
         跳5步   新位置   继续
```

---

## 6. 优化的 Raymarching

### 6.1 使用距离场的Raymarching

```glsl
for (int step = 1; step < maxSteps; step++) {
    // 查询到最近物体的距离
    float dist = texture(distanceTexture, sampleUv).r;
    
    // 安全跳跃
    sampleUv += rayDirection * dist;
    
    if (outOfBounds(sampleUv)) break;
    
    // 命中检测
    if (dist < EPS) { // EPS ≈ 0.001
        radDelta += texture(sceneTexture, sampleUv);
        break;
    }
}
```

### 6.2 性能对比

| 方法 | 每光线步数 | 画布512²性能 |
|------|-----------|-------------|
| 朴素Raymarch | 256 固定步 | ~5 FPS |
| 距离场Raymarch | ~32 平均步 | ~30+ FPS |

**加速原理**:
- 朴素方法：固定小步长，大量无效计算
- 距离场方法：自适应大步长，跳过空白区域

```
朴素方法:
○→○→○→○→○→○→○→○→●
(8步)

距离场方法:
○--------→○---→●
(2步，利用距离场跳过大片空白)
```

---

## 7. 高级特性

### 7.1 天空与太阳光照

```glsl
const vec3 skyColor = vec3(0.02, 0.08, 0.2);  // 淡蓝色天空
const vec3 sunColor = vec3(0.95, 0.95, 0.9);   // 接近白色的太阳

vec3 sunAndSky(float angle) {
    float angleToSun = mod(rayAngle - sunAngle, TAU);
    float sunIntensity = smoothstep(1.0, 0.0, angleToSun);
    return sunColor * sunIntensity + skyColor;
}
```

**物理意义**：
- 天空：半球形的环境光，贡献均匀但较弱
- 太阳：方向性强光源，只在一个方向上有强贡献

**smoothstep 函数**：
$$\text{smoothstep}(1.0, 0.0, x) = \begin{cases} 1 & x \leq 0 \\ 3x^2 - 2x^3 & 0 < x < 1 \\ 0 & x \geq 1 \end{cases}$$

产生平滑的过渡而非硬边缘。

### 7.2 时间累积

**问题**: 低光线数导致噪声

**解决方案**: 累积多帧结果

```glsl
vec4 prevRadiance = texture(lastFrameTexture, vUv);
gl_FragColor = mix(finalRadiance, prevRadiance, 0.9);
```

**混合公式**：
$$\text{output} = \alpha \cdot \text{current} + (1-\alpha) \cdot \text{previous}$$

$\alpha = 0.1$ 时，当前帧贡献10%，历史帧贡献90%。

**收敛性**: 经过 $n$ 帧，历史帧权重为 $0.9^n$，约23帧后衰减到 $<10\%$。

### 7.3 砂粒效果

通过在角度计算中添加噪声偏移：

```glsl
float rayAngleStepSize = angleStepSize + offset * TAU;
float angle = rayAngle * (float(i) + offset) + sunAngle;
```

某些像素会"双重捕获"太阳光，产生闪烁的砂粒效果。

---

## 8. 从朴素GI到Radiance Cascades

### 8.1 朴素方法的瓶颈

1. **冗余计算**: 相邻像素发射的光线大量重叠
   ```
   像素A: →→→→→→→→●
   像素B: →→→→→→→→●
   (大量重复采样)
   ```

2. **低效的光线利用**: 每条光线只贡献一个方向的radiance

### 8.2 Radiance Cascades的核心改进

**思想**: 分层缓存radiance信息，复用计算结果

```
层级结构:
Level 0: 高分辨率，近距离
Level 1: 中分辨率，中距离
Level 2: 低分辨率，远距离
```

远距离信息用低分辨率表示（因为变化缓慢），大幅减少计算量。

---

## 9. 总结：技术栈全景图

```
┌─────────────────────────────────────────────────────────────┐
│                     Naive GI Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   用户绘制 ──→ Surface Texture (填充像素的位置和颜色)      │
│                          │                                  │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │ Jump Flood Algorithm  │                      │
│              │   log(N) passes       │                      │
│              └───────────────────────┘                      │
│                          │                                  │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │   JFA Seed Texture    │                      │
│              │  (最近填充点的UV)      │                      │
│              └───────────────────────┘                      │
│                          │                                  │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │   Distance Field      │                      │
│              │ (每像素到最近点距离)   │                      │
│              └───────────────────────┘                      │
│                          │                                  │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │   Raymarching Shader   │                      │
│              │  - 自适应步进          │                      │
│              │  - 天空/太阳光照       │                      │
│              │  - 时间累积(可选)      │                      │
│              └───────────────────────┘                      │
│                          │                                  │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │   Final GI Render     │                      │
│              └───────────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. 关键公式汇总

| 概念 | 公式 | 说明 |
|------|------|------|
| 线段SDF | $t = \text{clamp}\left(\frac{(p-A)\cdot(B-A)}{(B-A)\cdot(B-A)}, 0, 1\right)$ | 投影参数 |
| 最近点 | $C = A + t(B-A)$ | 线上最近点 |
| 角度间隔 | $\Delta\theta = \frac{2\pi}{N}$ | N条光线 |
| JFA步长 | $s_k = N/2^{k+1}$ | 第k pass |
| Radiance平均 | $L = \frac{1}{N}\sum_{i=1}^{N} L_i$ | N条光线结果 |
| 时间混合 | $L_{final} = \alpha L_{new} + (1-\alpha)L_{old}$ | 累积 |

---

## 11. 参考资源

1. **Inigo Quilez SDF教程**: https://iquilezles.org/articles/distfunctions2d/
2. **Book of Shaders**: https://thebookofshaders.com/
3. **Jump Flood Algorithm原论文**: Rong and Tan, "Jump Flooding in GPU with Applications to Voronoi Diagram and Distance Transform" (2006)
4. **Radiance Cascades (下一篇)**: 这是系列第二部分，介绍核心算法

---

这篇文章的价值在于**从零构建直觉**：不是直接给出最优解，而是展示朴素方法的问题，然后一步步优化，最终引出Radiance Cascades。这种教学方式非常适合理解GI的本质原理。