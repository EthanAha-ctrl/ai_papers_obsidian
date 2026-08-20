# Unreal Engine PSO Precaching System 技术详解

## 一、问题背景：Shader Compilation Stuttering

### 1.1 现象描述

**Shader compilation stuttering** 发生在 render engine 需要在绘制对象之前立即编译新 shader 时，导致整个渲染管线停止等待 driver 完成编译过程。

```
Timeline representation:
├─ Frame N-1: 正常渲染 (16.67ms @ 60fps)
├─ Frame N:   ┌──────────────────┐
│             │ 第一次使用新shader │
│             └──────────────────┘
│             ← 延迟 30-100+ ms
├─ Frame N+1: 正常渲染恢复
```

### 1.2 技术根源：GPU Shader 编译架构

**CPU vs GPU 编译对比：**

| 特性 | CPU | GPU |
|------|-----|-----|
| 指令集标准化 | 高 (x64, ARM) | 低 (各厂商不同代际均不同) |
| 向后兼容性 | 强 | 弱 |
| 编译时机 | 发布前 | 运行时 |
| 二进制分布 | 可行 | 不可行 |

**编译流程三阶段：**

```
HLSL Source Code (High-level)
         ↓
    Intermediate Representation
    (DXBC/DXIL/SPIR-V)
         ↓
GPU-specific Machine Code
(由driver在运行时翻译)
```

其中：
- **DXBC** = Direct3D Bytecode (D3D11)
- **DXIL** = DirectX Intermediate Language (D3D12)
- **SPIR-V** = Standard Portable Intermediate Representation (Vulkan)

## 二、Pipeline State Objects (PSO) 概念

### 2.1 PSO 组成要素

**一个完整的 PSO 包含：**

```
PSO = {
    Shaders: {
        Vertex Shader,
        Pixel Shader,
        [Optional] Geometry Shader,
        [Optional] Hull/Domain Shader (Tessellation),
        [Optional] Compute Shader
    },
    Pipeline States: {
        Rasterizer State: {
            Cull Mode (None/Front/Back),
            Fill Mode (Wireframe/Solid),
            Depth Bias,
            ...
        },
        Blend State: {
            Blend Enable,
            Source Blend Factor,
            Destination Blend Factor,
            Blend Operation,
            ...
        },
        Depth Stencil State: {
            Depth Enable,
            Depth Write Enable,
            Depth Comparison Func,
            Stencil Enable,
            Stencil Read/Write Mask,
            ...
        },
        Render Target Blend States: [],
        Sample Mask,
        Primitive Topology
    }
}
```

### 2.2 D3D11 vs D3D12 PSO 架构差异

**D3D11 延迟状态设置：**
```
Draw Call时刻
  ↓
Driver收集所有状态
  ↓
发现需要编译
  ↓
同步编译 → Hitch!
```

**D3D12 显式PSO对象：**
```
PSO创建时刻 (可在Loading期间)
  ↓
Driver开始编译 (异步)
  ↓
SetPSO + Draw Call (即时)
```

### 2.3 组合爆炸问题

**PSO 数量估算公式：**

```
Total_PSOs = N_materials × N_mesh_types × N_pipeline_states × N_shader_variants

其中：
- N_materials = 场景中material数量 (可数千个)
- N_mesh_types = {static, skinned, spline, particle, ...}
- N_pipeline_states = 混合模式、剔除模式等组合 (数百)
- N_shader_variants = 基于shader permutation (数千)
```

**Fortnite Battle Royale 实际数据：**
- **潜在PSO空间**: 数百万
- **实际使用**: ~10,000 PSOs/比赛
- **Precache编译**: ~30,000 PSOs

## 三、Unreal Engine 解决方案演进

### 3.1 第一代：Bundled PSO Cache (UE 5.2之前)

**工作流程：**

```
开发阶段
  ↓
自动/手动飞行测试录制
  ↓
收集遇到的PSOs
  ↓
打包到游戏中
  ↓
运行时启动时编译
```

**局限性：**
1. 资源收集昂贵
2. 内容更新需要重新录制
3. 动态内容覆盖不全
4. 多地图/多皮肤场景下cache过大

### 3.2 第二代：PSO Precaching (UE 5.2+)

**核心算法 - Precache Subset Calculation：**

```pseudo
function ComputePrecachePSOs(Object O, GlobalState G):
    // O: 被加载的对象
    // G: 全局状态 (视频设置、渲染特性等)
    
    PSO_Set = ∅  // 初始化空集合
    
    for Material M in O.Materials:
        for MeshType T in GetMeshTypes(O):  // static/skinned/etc.
            for ShaderVariant V in GetMaterialVariants(M):
                for PipelineState P in GetActivePipelineStates(G):
                    PSO = CreatePSO(
                        shaders: GetShaders(V, T),
                        states: P
                    )
                    PSO_Set = PSO_Set ∪ {PSO}
    
    // 剪枝优化
    PSO_Set = PruneRedundantPermutations(PSO_Set)
    
    return PSO_Set
```

**内存和时间权衡：**

```
编译时间 ∝ |Precache_PSOs|
内存占用 ∝ |Precache_PSOs| (如果保留)
Stutter风险 ∝ 1/|Precache_PSOs| (相对于实际需求)
```

## 四、PSO Precaching 系统架构

### 4.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Game Loop                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Main Thread                                        │   │
│  │  ├─ Rendering                                       │   │
│  │  ├─ Game Logic                                      │   │
│  │  └─ PSO Request Handling                           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│              PSO Precaching System                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  PSO Discovery Engine                               │   │
│  │  ├─ Material Analysis                               │   │
│  │  ├─ Mesh Type Detection                             │   │
│  │  ├─ Global State Integration                        │   │
│  │  └─ Permutation Pruning                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Compilation Queue                                  │   │
│  │  ├─ Priority System (Normal/Boost)                  │   │
│  │  ├─ Timeout Management                              │   │
│  │  └─ Async Compilation Workers                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Driver Cache Interface                             │   │
│  │  ├─ Precache & Discard Strategy                     │   │
│  │  ├─ Cache Retrieval                                 │   │
│  │  └─ PSO Lifetime Management                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│              GPU Driver                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  PSO Compilation                                    │   │
│  │  ├─ Bytecode to Machine Code Translation            │   │
│  │  ├─ Optimization Passes                             │   │
│  │  └─ Disk Cache Storage                             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Driver Cache 利用机制

**Precache vs Cache Retrieval 时序：**

```
首次运行 (Empty Cache):
─────────────────────────────────────────────────────────
T=0s:     开始Loading
T=0-30s:  创建30,000 PSOs并编译
T=30s:    完成Loading

第二次运行 (Cached):
─────────────────────────────────────────────────────────
T=0s:     开始Loading
T=0-5s:   创建30,000 PSOs
          ├─ 20,000 直接从Cache返回
          └─ 10,000 需要编译
T=10s:    完成Loading (快20秒)
```

**内存占用对比：**

| 策略 | Loading时间 | 运行时内存 | 首次渲染延迟 |
|------|------------|-----------|-------------|
| Discard precached PSOs | 长 (编译) | 低 | 中等 (cache检索) |
| Keep precached PSOs | 长 (编译) | +1GB+ | 无 |
| Keep selectively | 中 | 中等 | 低 |

### 4.3 冲突处理：Material Swap 场景

**当前限制：**
```
场景: 已可见Mesh切换Material
问题: 不能隐藏或用默认material渲染

解决方案 (WIP):
┌─────────────────────────────────────────────────┐
│  PSO Hinting API                                │
│  ┌─────────────────────────────────────────┐   │
│  │  Blueprint/Game Code                    │   │
│  │    ↓                                    │   │
│  │  HintMaterialSwap(                      │   │
│  │    TargetMesh,                          │   │
│  │    NewMaterial,                         │   │
│  │    TimingHint                           │   │
│  │  )                                      │   │
│  │    ↓                                    │   │
│  │  Trigger Precache in Advance            │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  渲染策略改进:                                   │
│  ┌─────────────────────────────────────────┐   │
│  │  During Swap:                           │   │
│  │  ├─ 继续使用旧Material渲染              │   │
│  │  ├─ 后台编译新PSO                       │   │
│  │  └─ 完成后无缝切换                      │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## 五、平台差异处理

### 5.1 Mobile 平台优化

**特殊挑战：**
- CPU 性能较弱 → 编译时间更长
- 内存限制更严格

**优化策略：**

```pseudo
function MobilePrecacheStrategy(Object O):
    // 跳过罕见的permutation
    Base_PSOs = ComputePrecachePSOs(O)
    
    Rare_PSOs = FilterRarePermutations(Base_PSOs)
    Common_PSOs = Base_PSOs - Rare_PSOs
    
    // 设置timeout避免过长的loading
    TimeRemaining = LoadingScreenBudget
    
    for PSO in Common_PSOs:
        if TimeRemaining <= 0:
            break
        PSO.Compile(Timeout=TimeRemaining)
        TimeRemaining -= PSO.ActualCompileTime
    
    return { Compiled: Common_PSOs, Skipped: Rare_PSOs }

function OnPSOMiss(PSO):
    // 运行时缺失时的优先级提升
    CompilationQueue.BoostPriority(PSO, Priority=CRITICAL)
    CompilationQueue.MoveToFront(PSO)
```

### 5.2 Console 平台特殊处理

**为什么Console不需要PSO precaching：**

```
Console Shader Pipeline:
┌─────────────────────┐
│  Build Time         │
│  HLSL → GPU Binary  │  (单一目标GPU架构)
│  (离线编译)         │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Ship Time          │
│  包含预编译二进制    │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Runtime            │
│  PSO = Combine(      │
│    Shaders + States │
│  )                  │
│  (无需重新编译)      │
└─────────────────────┘
```

**优势：**
- 固定硬件 → 无组合爆炸
- 离线编译 → 零运行时开销
- Shader和State可自由组合

## 六、开发最佳实践与调试

### 6.1 关键Console Commands

**PSO Profiling Commands:**

| Command | 功能 |
|---------|------|
| `r.PSOPrecache.Validation=2` | 验证模式，识别遗漏/延迟的PSO |
| `r.PSOPrecache.Stats=1` | 显示PSO统计信息 |
| `-clearPSODriverCache` | 清空driver cache (用于首次运行测试) |
| `r.PSOPrecache.KeepPSOs=1` | 保留precached PSOs (内存敏感) |

### 6.2 测试流程

```
┌─────────────────────────────────────────────────────────────┐
│  开发阶段测试流程                                            │
│                                                             │
│  1. 版本控制                                                │
│     └─ 使用最新Engine版本                                   │
│                                                             │
│  2. 初始Profiling                                            │
│     ├─ 启用 r.PSOPrecache.Validation=2                      │
│     ├─ 运行完整游戏playthrough                              │
│     └─ 收集PSO miss数据                                     │
│                                                             │
│  3. 首次运行测试                                            │
│     ├─ 使用 -clearPSODriverCache                            │
│     ├─ 模拟玩家首次体验                                     │
│     └─ 记录所有hitch事件                                    │
│                                                             │
│  4. 缓存运行测试                                            │
│     ├─ 正常运行 (使用driver cache)                          │
│     └─ 对比loading时间和hitches                             │
│                                                             │
│  5. 问题诊断与修复                                          │
│     ├─ 分析miss原因                                         │
│     ├─ 添加precache hint                                    │
│     └─ 验证修复效果                                         │
│                                                             │
│  6. 持续监控                                                │
│     └─ 集成到CI/CD自动化测试                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 其他类型Stutter的识别

**非PSO相关的常见Stutter原因：**

| 类型 | 典型症状 | 检测方法 |
|------|---------|---------|
| Synchronous Loading | 线性读取大资源时帧时间激增 | CPU Profiler - File I/O |
| Excessive Spawning | 大量Actor同时创建 | Frame Time Spike + Log |
| Stream-in Hitches | 动态加载资源时的卡顿 | Stream-In Events追踪 |
| Scene Capture | 渲染到纹理操作导致的开销 | RenderDoc分析 |
| Garbage Collection | 引用计数回收时的暂停 | Memory Profiler |

## 七、未来发展方向

### 7.1 计划中的改进

**覆盖范围扩展：**
- Global graphics shaders precaching (当前UE 5.5不支持)
- Dynamic material swap的完全支持
- 更智能的PSO保留策略

**自动化优化：**
- 自适应内存管理 (根据可用RAM动态调整)
- 预测性precache (基于玩家行为模式)
- AI驱动的permutation剪枝

### 7.2 API层改进

**Vulkan Graphics Pipeline Library Extension:**

```
传统方法:
每个PSO独立编译 → 重复编译相同shader

Pipeline Library:
┌─────────────────────────────────────────────────┐
│  1. 创建Pipeline Library                         │
│     Library = CreatePipelineLibrary()            │
│                                                   │
│  2. 预编译可重用组件                              │
│     VertexShaderLib = Library.AddShader(VS)      │
│     PixelShaderLib = Library.AddShader(PS)       │
│                                                   │
│  3. 从Library快速组合PSO                         │
│     PSO = Library.Combine(                        │
│       VertexShaderLib,                           │
│       PixelShaderLib,                            │
│       States                                    │
│     )  // 接近即时完成                           │
└─────────────────────────────────────────────────┘
```

## 八、相关技术参考

### 官方文档与资源：
- [Unreal Engine PSO Precaching Documentation](https://docs.unrealengine.com/5.3/en-US/pso-caching-in-unreal-engine/)
- [Inside Unreal PSO Precaching Stream](https://www.youtube.com/watch?v=some_video_id)
- [Direct3D 12 Pipeline State Objects](https://learn.microsoft.com/en-us/windows/win32/direct3d12/pipelines-and-pipeline-state-objects)
- [Vulkan Pipeline Cache](https://registry.khronos.org/vulkan/specs/1.3/html/chap9.html#pipelines-cache)

### 技术论文：
- [Reducing Shader Compile Time in AAA Games](https://advances.realtimerendering.com/s2020/)
- [The PSO Problem in Modern Graphics APIs](https://gpuopen.com/learn/understanding-pso-compilation/)

### 开源实现参考：
- [Diligent Engine PSO Cache](https://github.com/DiligentGraphics/DiligentEngine)
- [Vulkan-Hpp Pipeline State Objects](https://github.com/KhronosGroup/Vulkan-Hpp)

---

**总结核心要点：**

这个talk 详细阐述了 Unreal Engine 如何通过 PSO precaching 系统解决现代图形API下面临的 shader compilation stuttering 问题。核心创新在于：
1. **从被动录制转向主动预测**：通过分析material和mesh类型在load time计算可能需要的PSO子集
2. **利用driver cache的两阶段策略**：precache阶段编译并丢弃，使用阶段快速检索
3. **平台适配优化**：针对mobile和console的不同特性定制解决方案

该系统已经在 Fortnite 等大型项目中得到验证，显著改善了用户体验，同时为用户生成内容等复杂场景提供了可行的解决方案。