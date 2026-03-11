# Vulkan Roadmap 深度解析

## 概述架构

Vulkan Roadmap由Khronos Group推出的渐进式功能保障计划，为开发者提供了明确的硬件能力预期。整体架构如下：

```
Vulkan 1.3 → Roadmap 2022 
               ↓
Vulkan 1.3/1.4 → Roadmap 2024
                   ↓
Vulkan 1.4 → Roadmap 2026
```

---

## Roadmap 2022 里程碑详解

### 核心要求
**API版本要求**: Vulkan 1.3

**关键Feature矩阵**:

| Vulkan版本 | Feature | 用途 |
|-----------|---------|------|
| 1.0 | `descriptorIndexing` | 支持动态索引descriptor |
| 1.0 | `shaderSampledImageArrayDynamicIndexing` | Sampled image数组动态索引 |
| 1.1 | `samplerYcbcrConversion` | YCbCr颜色空间转换 |
| 1.2 | `samplerMirrorClampToEdge` | 镜像包装模式 |
| 1.2 | `runtimeDescriptorArray` | 运行时descriptor数组 |

### 关键限制提升对比表

| 参数 | Vulkan Core | Roadmap 2022 | 提升幅度 |
|------|-------------|--------------|----------|
| `maxImageDimension2D` | 4096 | 8192 | 2x |
| `maxPerStageDescriptorSamplers` | 16 | 64 | 4x |
| `maxPerStageDescriptorSampledImages` | 16 | 200 | 12.5x |
| `maxUniformBufferRange` | 16KB | 64KB | 4x |
| `maxComputeWorkGroupInvocations` | 128 | 256 | 2x |

### 必需扩展
- `VK_KHR_global_priority`: 提供线程优先级控制

---

## Roadmap 2024 里程碑详解

### 依赖关系
```
Roadmap 2022 → Roadmap 2024
```

核心目标：让开发者可以依赖重要的rasterization和shader功能。

### Shader能力增强

**小型类型支持**:
```glsl
// 16-bit integer
int16_t value16 = int16_t(some_value);

// 8-bit integer  
int8_t color8 = int8_t(color_component);

// 16-bit float
float16_t pos16 = float16_t(position);
```

**Subgroup Reconvergence保证**:
- `VK_KHR_shader_maximal_reconvergence`
- `VK_KHR_shader_quad_control`

**浮点控制增强**:
- `VK_KHR_shader_float_controls2`: 提供32位和16位浮点的round-to-nearest-even模式

### Rasterization功能

| Feature | 描述 | 影响 |
|---------|------|------|
| `multiDrawIndirect` | 单次indirect draw调⽤处理多个draw | 减少CPU开销 |
| `shaderDrawParameters` | Shader内访问draw index | 简化instance管理 |
| `8-bit indices` | 使用uint8作为索引类型 | 降低内存带宽 |
| Dynamic Rendering Local Read | 动态渲染中本地读取 | 提升多pass效率 |

### 关键限制表

| 参数 | Roadmap 2022 | Roadmap 2024 | 变化 |
|------|-------------|--------------|------|
| `maxBoundDescriptorSets` | 4 | 7 | +75% |
| `maxColorAttachments` | 7 | 8 | +14% |
| `timestampComputeAndGraphics` | FALSE | TRUE | 新增 |

### 必需扩展列表

| 扩展 | 功能领域 |
|------|---------|
| `VK_KHR_dynamic_rendering_local_read` | 渲染管线优化 |
| `VK_KHR_shader_float_controls2` | 精度控制 |
| `VK_KHR_line_rasterization` | 线条渲染 |
| `VK_KHR_vertex_attribute_divisor` | 实例渲染 |
| `VK_KHR_push_descriptor` | 快速descriptor更新 |
| `VK_KHR_maintenance5` | 维护更新 |

---

## Roadmap 2026 里程碑详解（最新）

### 架构层级
```
Vulkan 1.4
    ↓
Roadmap 2024
    ↓
Roadmap 2026 + VK_EXT_descriptor_heap（试验性）
```

### 核心创新点

**1. Descriptor Heap Revolution**:
```
传统Descriptor Set:
┌──────────────────────────┐
│ VkDescriptorSetLayout     │
│  ├─ binding 0: Sampler    │
│  └─ binding 1: Uniform    │
└──────────────────────────┘

Descriptor Heap (VK_EXT_descriptor_heap):
┌──────────────────────────┐
│ Direct Memory Access      │
│  └─ Raw descriptor layout │
└──────────────────────────┘
```

**关键优势**:
- 直接访问descriptor内存
- 保持与legacy descriptor sets兼容
- 降低API调用开销
- 最终将替代现有descriptor set机制

**2. Shader高级特性**

| 扩展 | 能力 | 应用场景 |
|------|------|---------|
| `VK_KHR_shader_clock` | Shader内访问高精度时钟 | 性能分析、timed rendering |
| `VK_KHR_compute_shader_derivatives` | Compute shader中使用dFdx/dFdy | Image processing、filtering |
| `VK_KHR_cooperative_matrix` | 硬件加速矩阵运算 | AI推理、ML工作负载 |

**3. Variable Rate Shading (VRS)**:

```cpp
// VRS配置示例
VkFragmentShadingRateKHR fragmentShadingRate = {
    .fragmentSize = {2, 2},  // 2x2 fragment shading
    .combinerOps = {
        VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
        VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR
    }
};
```

性能提升可达30-40%（视场景而定）

### 显著限制提升

| 类别 | 参数 | Core/2024 | 2026 | 提升倍数 |
|------|------|-----------|------|---------|
| Uniform Buffers | `maxPerStageDescriptorUniformBuffers` | 15 | 200 | 13.3x |
| Storage Buffers | `maxPerStageDescriptorStorageBuffers` | 4 | 200 | 50x |
| Input Attachments | `maxPerStageDescriptorInputAttachments` | 4 | 8 | 2x |
| Vertex Output | `maxVertexOutputComponents` | 64 | 124 | 1.94x |
| Compute Memory | `maxComputeSharedMemorySize` | 16KB | 32KB | 2x |
| Framebuffer | `maxFramebufferWidth/Height` | 7680 | 8192 | 1.07x |

### Swapchain保证

| 扩展 | 功能 | 解决问题 |
|------|------|---------|
| `VK_KHR_present_mode_fifo_latest_ready` | FIFO最新可用呈现 | 减少latency |
| `VK_KHR_present_id2` | 改进的present ID | 更好的帧同步 |
| `VK_KHR_present_wait2` | 改进的present等待 | 降低功耗 |
| `VK_KHR_swapchain_maintenance1` | Swapchain维护 | 更灵活的管理 |

### Host Image Copy

```cpp
// 直接从host内存复制到image，无需staging buffer
void cmdCopyMemoryToImage(
    VkCommandBuffer commandBuffer,
    const VkCopyMemoryToImageInfo* pInfo
);
```

**性能优势**:
- 消除staging buffer复制开销
- 减少内存占用
- 简化代码逻辑

### 算力支持改进

**Robustness 2.0**:
```cpp
VkPhysicalDeviceRobustness2FeaturesEXT robustness2 = {
    .robustBufferAccess2 = VK_TRUE,        // 改进的越界访问保护
    .robustImageAccess2 = VK_TRUE,         // 改进的image访问保护  
    .nullDescriptor = VK_TRUE              // 支持null descriptor
};
```

### 完整扩展列表

**Render核心**:
- `VK_KHR_fragment_shading_rate`
- `VK_KHR_shader_clock`
- `VK_KHR_compute_shader_derivatives`
- `VK_KHR_host_image_copy`

**内存与访问**:
- `VK_KHR_copy_memory_indirect`
- `VK_KHR_workgroup_memory_explicit_layout`
- `VK_KHR_shader_untyped_pointers`

**维护与兼容**:
- `VK_KHR_maintenance7` / `8` / `9`
- `VK_KHR_pipeline_binary`

**显示与交换**:
- `VK_KHR_surface`
- `VK_KHR_swapchain`  
- `VK_KHR_present_mode_fifo_latest_ready`
- `VK_KHR_present_id2`
- `VK_KHR_present_wait2`
- `VK_KHR_surface_maintenance1`
- `VK_KHR_swapchain_maintenance1`

---

## 时间线与生态系统

### 开发时间表

```
2026年1月: Roadmap 2026公告
2026年2月: Vulkanised 2026会议讨论
2026年Q1: Vulkan SDK发布初始支持
2026年底: 大多数Vulkan adopter提供兼容设备
```

### 社区反馈渠道

- **GitHub**: Vulkan-Docs repository
- **Discord**: Vulkan官方频道
- **会议**: Vulkanised 2026 (San Diego, Feb 9-11)

### 硬件支持范围

Roadmap适用于:
- ✅ Mid-to-high-end smartphones
- ✅ Tablets
- ✅ Laptops
- ✅ Consoles  
- ✅ Desktop devices

---

## 技术意义总结

| 方面 | 影响 |
|------|------|
| **开发效率** | 减少可选功能检测代码，统一baseline |
| **性能潜力** | VRS、Cooperative Matrix带来30%+提升 |
| **资源管理** | Descriptor Heap简化内存模型 |
| **跨平台** | 统一各平台最小功能集 |
| **未来演进** | 为AI/ML工作负载提供硬件加速支持 |

---

## 参考资源

1. **Vulkan Roadmap 2026 Blog**: https://www.khronos.org/blog/vulkan-introduces-roadmap-2026-and-new-descriptor-heap-extension
2. **Official Roadmap Documentation**: https://docs.vulkan.org/spec/latest/appendices/roadmap.html
3. **Vulkan Registry**: https://registry.khronos.org/vulkan/
4. **Vulkanised Conference**: https://www.vulkanised.com/
5. **Vulkan Discord**: https://discord.gg/vulkan

这些Roadmap确保了Vulkan生态系统的持续演进，同时为开发者提供了可预测的硬件能力承诺，大大简化了跨平台图形开发的复杂度。