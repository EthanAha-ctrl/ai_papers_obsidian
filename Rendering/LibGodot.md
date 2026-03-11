## 一、当前 Godot 4.6 LibGodot 的实际架构

```
libgodot.so (或者 libgodot_core.so)
├── RenderingServer (作为 Singleton Object 内部实现)
├── PhysicsServer (作为 Singleton Object 内部实现)
├── AudioServer (作为 Singleton Object 内部实现)
├── RenderingServer3D (后端实现：ForwardPlus/Mobile)
├── PhysicsServer3D (后端实现：Jolt/Godot)
└── AudioServer (后端实现：OpenAL/WASAPI 等)
```

---

## 二、Server 层的真实架构

### 当前实现：Singleton Objects（单例对象）

```
┌─────────────────────────────────────────────────────────────────────┐
│                      libgodot.so (单一共享库)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │       Level 3: Engine Core Layer (引擎核心层)                  │ │
│  │  ┌─────────────────────────────────────────────────────────┐  │ │
│  │  │    Global Singletons (全局单例)                           │  │ │
│  │  │                                                          │  │ │
│  │  │    RenderingServer::singleton                           │  │ │
│  │  │    ┌──────────────────────────────────────────┐         │  │ │
│  │  │    │ RenderingServer 类实例                    │         │  │ │
│  │  │    │ ├── mesh_create()                        │         │  │ │
│  │  │    │ ├── texture_2d_create()                  │         │  │ │
│  │  │    │ ├── camera_create()                      │         │  │ │
│  │  │    │ └── draw()                               │         │  │ │
│  │  │    └──────────────────────────────────────────┘         │  │ │
│  │  │                                                          │  │ │
│  │  │    PhysicsServer::singleton                             │  │ │
│  │  │    ┌──────────────────────────────────────────┐         │  │ │
│  │  │    │ PhysicsServer 类实例                       │         │  │ │
│  │  │    │ ├── space_create()                       │         │  │ │
│  │  │    │ ├── body_create()                        │         │  │ │
│  │  │    │ ├── step()                               │         │  │ │
│  │  │    │ └── space_get_direct_state()             │         │  │ │
│  │  │    └──────────────────────────────────────────┘         │  │ │
│  │  │                                                          │  │ │
│  │  │    AudioServer::singleton                               │  │ │
│  │  │    ┌──────────────────────────────────────────┐         │  │ │
│  │  │    │ AudioServer 类实例                         │         │  │ │
│  │  │    │ ├── bus_create()                         │         │  │ │
│  │  │    │ ├── stream_play()                        │         │  │ │
│  │  │    │ └── update()                             │         │  │ │
│  │  │    └──────────────────────────────────────────┘         │  │ │
│  │  └─────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │       Level 2: Server Backends (Server 后端实现)              │ │
│  │  ┌─────────────────────────────────────────────────────────┐  │ │
│  │  │    RenderingServerBackend (渲染后端)                      │  │ │
│  │  │    ├── ForwardPlusRenderer                               │  │ │
│  │  │    │   ├── Clustered Lighting                            │  │ │
│  │  │  │   │   ├── Depth Pre-pass                             │  │ │
│  │  │    │   └── Opaque/Transparent Passes                     │  │ │
│  │  │    │                                                       │  │ │
│  │  │    └── MobileRenderer                                     │  │ │
│  │  │        ├── Single-pass Lighting                          │  │ │
│  │  │        └── Tile-based Rendering                          │  │ │
│  │  │                                                           │  │ │
│  │  │    PhysicsServerBackend (物理后端)                        │  │ │
│  │  │    ├── JoltPhysicsServer3D                                │  │ │
│  │  │    │   ├── JoltPhysics 初始化                            │  │ │
│  │  │    │   ├── Collision Detection (GJK/EPA)                 │  │ │
│  │  │    │   ├── Constraint Solver (Jacobian)                  │  │ │
│  │  │    │   └── Integration (Semi-implicit Euler)             │  │ │
│  │  │    │                                                       │  │ │
│  │  │    └── GodotPhysicsServer3D (备用)                         │  │ │
│  │  │        ├── Godot 原生物理引擎                              │  │ │
│  │  │        ├── Collision Detection                            │  │ │
│  │  │        └── Constraint Solver                              │  │ │
│  │  │                                                           │  │ │
│  │  │    AudioServerBackend (音频后端)                          │  │ │
│  │  │    ├── AudioDriverOpenAL                                  │  │ │
│  │  │    ├── AudioDriverWASAPI (Windows)                        │  │ │
│  │  │    ├── AudioDriverPulseAudio (Linux)                      │  │ │
│  │  │    └── AudioDriverALSA                                   │  │ │
│  │  └─────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │       Level 1: Driver/I/O Layer (驱动层)                      │ │
│  │  ┌─────────────────────────────────────────────────────────┐  │ │
│  │  │    RenderingDevice (GPU 统一抽象)                        │  │ │
│  │  │    ├── RenderingDeviceVulkan                            │  │ │
│  │  │    ├── RenderingDeviceD3D12                             │  │ │
│  │  │    └── RenderingDeviceMetal                             │  │ │
│  │  │                                                           │  │ │
│  │  │    VideoDriver                                           │  │ │
│  │  │    ├── VulkanDriver                                      │  │ │
│  │  │    ├── D3D12Driver                                       │  │ │
│  │  │    └── MetalDriver                                       │  │ │
│  │  │                                                           │  │ │
│  │  │    AudioDriver                                           │  │ │
│  │  │    ├── AudioDriverOpenAL                                  │  │ │
│  │  │    ├── AudioDriverWASAPI                                 │  │ │
│  │  │    └── AudioDriverPulseAudio                             │  │ │
│  │  └─────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

所有这些编译到单一的 libgodot.so（或 libgodot_core.so）中！
```

---

## 三、Server 初始化流程（真实实现）

### 单例注册和初始化

```cpp
// core/os/os.cpp
void OS::initialize_servers() {
    // ┌────────────────────────────────────────────────────────────┐
    // │ Server 单例注册和初始化                                     │
    // ├────────────────────────────────────────────────────────────┤
    // │                                                            │
    // │ 1. RenderingServer 初始化                                   │
    // │ ┌──────────────────────────────────────────────────────┐  │
    // │ │ RenderingServer::singleton = memnew(RenderingServer());│  │
    // │ │                                                         │  │
    // │ │ // 通过 ClassDB 注册引擎类                              │  │
    // │ │ ClassDB::register_class<RenderingServer>();            │  │
    // │ │                                                         │  │
    // │ │ // 创建渲染后端                                         │  │
    // │ │ RenderingServerBackend* backend = nullptr;                │  │
    // │ │                                                         │  │
    // │ │ if (get_rendering_driver() == "vulkan") {                │  │
    // │ │     backend = memnew(ForwardPlusRenderer());             │  │
    // │ │ } else if (get_rendering_driver() == "forward_mobile") {│  │
    // │ │     backend = memnew(MobileRenderer());                  │  │
    // │ │ }                                                        │  │
    // │ │                                                         │  │
    // │ │ RenderingServer::singleton->set_backend(backend);       │  │
    // │ └──────────────────────────────────────────────────────┘  │
    // │                                                            │
    // │ 2. PhysicsServer 初始化                                    │
    // │ ┌──────────────────────────────────────────────────────┐  │
    // │ │ PhysicsServer::singleton = memnew(PhysicsServer());    │  │
    // │ │                                                         │  │
    // │ │ // 通过 ClassDB 注册引擎类                              │  │
    // │ │ ClassDB::register_class<PhysicsServer>();              │  │
    // │ │                                                         │  │
    // │ │ // 创建物理后端                                         │  │
    // │ │ if (get_physics_engine() == "jolt") {                   │  │
    // │ │     PhysicsServerBackend* jolt =                        │  │
    // │ │         memnew(JoltPhysicsServer3D());                  │  │
    // │ │     PhysicsServer::singleton->set_active_backend(jolt); │  │
    // │ │ } else {                                                 │  │
    // │ │     PhysicsServerBackend* godot =                       │  │
    // │ │         memnew(GodotPhysicsServer3D());                 │  │
    // │ │     PhysicsServer::singleton->set_active_backend(godot);│  │
    // │ │ }                                                        │  │
    // │ └──────────────────────────────────────────────────────┘  │
    // │                                                            │
    // │ 3. AudioServer 初始化                                      │
    // │ ┌──────────────────────────────────────────────────────┐  │
    // │ │ AudioServer::singleton = memnew(AudioServer());        │  │
    // │ │                                                         │  │
    // │ │ // 通过 ClassDB 注册引擎类                              │  │
    // │ │ ClassDB::register_class<AudioServer>();                │  │
    // │ │                                                         │  │
    // │ │ // 初始化音频驱动                                       │  │
    // │ │ AudioDriver* driver = audio_driver_create();           │  │
    // │ │ driver->initialize();                                  │  │
    // │ │                                                         │  │
    // │ │ // 设置输出总线                                         │  │
    // │ │ AudioServer::singleton->bus_create();                  │  │
    // │ │ AudioServer::singleton->bus_set_name(0, "Master");     │  │
    // │ └──────────────────────────────────────────────────────┘  │
    // └────────────────────────────────────────────────────────────┘
}
```

### Server 单例访问

```cpp
// 通过 Godot 类系统访问 Server
RenderingServer* rs = RenderingServer::get_singleton();
PhysicsServer* ps = PhysicsServer::get_singleton();
AudioServer* as = AudioServer::get_singleton();

// 这不是独立的库，而是通过 ClassDB 查找的内部对象
// 实现类似：
namespace godot {
    RenderingServer* RenderingServer::get_singleton() {
        return static_cast<RenderingServer*>(
            ClassDB::get_singleton("RenderingServer")->get_static_instance()
        );
    }
}
```

---

## 四、Godot 模块系统 vs 独立共享库

### Godot 的模块化构建系统（编译时模块化）

```
Godot 源码树结构：
godot/
├── modules/
│   ├── xr/                    # XR 模块（可选）
│   ├── mono/                  # C# 支持（可选）
│   ├── jsonrpc/               # JSON-RPC 支持（可选）
│   ├── mobilevr/              # 移动端 VR（可选）
│   └── ...
│
├── drivers/
│   ├── vulkan/                # Vulkan 驱动
│   ├── d3d12/                 # DirectX 12 驱动
│   ├── metal/                 # Metal 驱动
│   ├── gles3/                 # OpenGL ES 3 驱动
│   ├── unix/                  # Unix 音频驱动
│   ├── windows/               # Windows 音频驱动
│   └── ...
│
├── servers/
│   ├── audio/                 # AudioServer 实现
│   ├── physics/               # PhysicsServer 实现
│   ├── rendering/             # RenderingServer 实现
│   ├── visuals/               # VisualServer 实现
│   └── ...
│
└── ...
```

**编译时模块化**：
```bash
# 编译引擎时，可以选择包含/排除模块
# 但最终仍然编译到单一的二进制文件中

# 构建 LibGodot（包含所有模块）
scons platform=linuxbsd library_type=shared_library module_xr_enabled=yes module_mono_enabled=no

# 结果：
├── libgodot.so (单一文件，包含所有模块的代码)
```

---

## 五、为什么当前实现是单例而不是独立库？

### 技术原因

#### 1. 依赖关系复杂

```
Server 之间的复杂的相互依赖：

RenderingServer
├── 依赖 PhysicsServer (物理约束对渲染的影响，如 Cloth)
├── 依赖 AudioServer (音频可视化)
├── 依赖 NavigationServer (导航网格可视化)
├── 依赖 AnimationServer (动画播放影响变换)
└── 依赖 ResourceManager (资源访问)

PhysicsServer
├── 依赖 RenderingServer (碰撞形状可视化，debug)
├── 依赖 NavigationServer (导航代理)
└── 依赖 ResourceManager

AudioServer
├── 依赖 RenderingServer (音频可视化)
├── 依赖 PhysicsServer (3D 空间音效)
└── 依赖 ResourceManager

如果拆分成独立的库，需要处理：
- 循环依赖
- 版本兼容
- ABI 稳定性
- 二进制兼容性
```

#### 2. ClassDB 系统的限制

```cpp
// Godot 的 ClassDB 系统期望所有类在同一二进制中
class ClassDB {
private:
    HashMap<StringName, ClassInfo> classes;
    
public:
    static void register_class(const StringName& p_class) {
        // 注册到 ClassDB 时，期望类已经在当前二进制中定义
        classes[p_class] = {};
    }
    
    static Object* get_static_instance(const StringName& p_class) {
        // 直接在当前二进制中查找对象
        return classes[p_class].static_instance;
    }
};

// 如果 RenderingServer 在独立库中：
// libgodot_rendering.so 会注册自己类
// libgodot_core.so 访问时，需要运行时动态查找
// 增加复杂性和性能开销
```

#### 3. Godot 对象系统的设计

```cpp
// Godot 使用 ID 和引用计数管理对象 RID (Resource ID)
class RID {
    uint64_t id;  // 全局唯一的资源 ID
                   // 在单一二进制中，ID 管理简单
                   // 如果拆分成独立库，需要跨库的 ID 管理
                   // 增加同步复杂度
};

// RenderingServer 返回 RID
RenderingServer* rs = RenderingServer::get_singleton();
RID mesh_rid = rs->mesh_create();

// PhysicsServer 也使用同一 RID 系统
PhysicsServer* ps = PhysicsServer::get_singleton();
RID shape_rid = ps->sphere_shape_create();

// 如果在独立库中：
// - 无法轻易共享 RID 系统
// - 需要跨库的数据传递
// - 增加同步开销
```

---

## 六、LibGodot 的实际模块化能力

### 当前 LibGodot 能做的

```
┌─────────────────────────────────────────────────────────────────────┐
│           LibGodot 的实际模块化能力（基于构建系统）                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 编译时模块选择（SCons 构建选项）                                 │
│     ├── module_3d_enabled=yes/no         # 3D 渲染模块              │
│     ├── module_bsp_enabled=yes/no        # BSP 工具模块             │
│     ├── module_csg_enabled=yes/no        # CSG（Constructive Solid） │
│     ├── module_editor_enabled=yes/no     # 编辑器模块（纯运行时可禁用）│
│     ├── module_gridmap_enabled=yes/no    # GridMap 模块              │
│     ├── module_navigation_enabled=yes/no # 导航模块                  │
│     ├── module_openxr_enabled=yes/no     # OpenXR 模块             │
│     └── module_xr_enabled=yes/no         # XR 模块                   │
│                                                                     │
│  2. 驱动选择                                                         │
│     ├── rendering_driver=vulkan/d3d12/metal/opengl3                │
│     ├── physics_engine=jolt/godot                                  │
│     └── build_with_lto=yes/no           # Link Time Optimization   │
│                                                                     │
│  3. 平台选择                                                         │
│     ├── platform=linuxbsd/windows/macos/android/ios/web             │
│     └── target=editor/template_release/template_debug               │
│                                                                     │
│  结果：通过构建选项减少二进制大小，但仍然是单一库                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 实际构建命令示例

```bash
# 最小化 LibGodot（仅核心功能，用于嵌入）
scons platform=linuxbsd \
     library_type=shared_library \
     module_3d_enabled=no \
     module_csg_enabled=no \
     module_bsp_enabled=no \
     module_navigation_enabled=no \
     module_openxr_enabled=no \
     module_xr_enabled=no \
     module_mono_enabled=no \
     target=template_release

# 结果：libgodot.so (~5-10MB) - 极简版本
# 仍然包含：
# - RenderingServer（基本 2D 渲染）
# - PhysicsServer（基本 2D 物理）
# - AudioServer（基本音频）
# - 核心引擎系统
# - ClassDB
# - Resource System
```

---

## 七、社区项目：尝试拆分 LibGodot

### migueldeicaza/libgodot 项目

```
GitHub: https://github.com/migueldeicaza/libgodot

这个项目尝试将 Godot 编译为独立库，但不是拆分成多个库，而是：
┌─────────────────────────────────────────────────────────────────┐
│ migueldeicaza/libgodot 架构                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  目标：将 Godot 编译为共享库，并通过 GDExtension API 暴露控制      │
│                                                                  │
│  ┌──────────────┐                                               │
│  │ libgodot.so  │                                               │
│  │              │   ← 单一共享库                                 │
│  │ [Godot       │                                               │
│  │  Engine]     │                                               │
│  │              │                                               │
│  │ Exposed API: │                                               │
│  │ - godot_initialize()                                          │
│  │ - godot_step()                                                │
│  │ - godot_shutdown()                                            │
│  │ - godot_get_scene_tree()                                      │
│  │ - godot_get_singleton() [RenderingServer, etc.]              │
│  └──────┬───────┘                                               │
│         │                                                       │
│         │ GDExtension API                                        │
│         ├─────────────────────────────────────────────────────┐  │
│         │                                                     │  │
│  ┌──────▼──────┐   ┌──────────────┐   ┌──────────────────┐  │  │
│  │ Swift Apps  │   │ Rust Apps    │   │ Python Apps      │  │  │
│  │             │   │              │   │                  │  │  │
│  │ SwiftGodot  │   │ godot-rust   │   │ py-godot         │  │  │
│  │ (bindings)  │   │ (bindings)   │   │ (bindings)       │  │  │
│  └─────────────┘   └──────────────┘   └──────────────────┘  │  │
│                                                             │  │
│  但这仍然不是拆分 servers 为独立库！                          │  │
└─────────────────────────────────────────────────────────────┘
```

### V-Sekai/libgodot-project 项目

```
GitHub: https://github.com/V-Sekai/libgodot-project

这个项目也尝试类似的方案：
┌─────────────────────────────────────────────────────────────────┐
│ V-Sekai/libgodot-project 架构                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  目标：编译 Godot 为静态/共享库 + 动态加载模块                     │
│                                                                  │
│  ┌──────────────────┐                                            │
│  │ libgodot_core.so │                                            │
│  │                  │                                            │
│  │ [Core Engine]    │                                            │
│  │ - OS             │                                            │
│  │ - MainLoop       │                                            │
│  │ - ClassDB        │                                            │
│  │ - Resource System│                                            │
│  │ - Script Engine  │                                            │
│  └──────┬───────────┘                                            │
│         │                                                        │
│         └──────────────────────────────┐                          │
│         ▼                              ▼                          │
│  ┌──────────────────┐    ┌──────────────────────┐               │
│  │libgodot_servers.so│  │ (可选)                  │               │
│  │                  │    │ libgodot_module_xr.so │               │
│  │ [Servers]        │    │ libgodot_module_mono.so│               │
│  │ - Rendering      │    └──────────────────────┘               │
│  │ - Physics        │                                          │
│  │ - Audio          │                                          │
│  │ - Navigation     │                                          │
│  └──────────────────┘                                          │
│                                                                  │
│  但这仍然是实验性方案！                                          │
│  并不是 Godot 4.6 官方 LibGodot 的默认行为！                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 八、架构对比：当前 vs 未来可能

### 当前 LibGodot (Godot 4.6)

```
┌─────────────────────────────────────────────────────────────────┐
│              当前 LibGodot (单一库，编译时模块化)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ libgodot.so / libgodot_core.so (单一文件)                 │   │
│  │                                                           │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ RenderingServer (Singleton Object)                 │   │   │
│  │ │ PhysicsServer (Singleton Object)                  │   │   │
│  │ │ AudioServer (Singleton Object)                    │   │   │
│  │ │                                                    │   │   │
│  │ │ RenderingServerBackend::ForwardPlusRenderer        │   │   │
│  │ │ PhysicsServerBackend::JoltPhysicsServer3D          │   │   │
│  │ │ AudioServerBackend::AudioDriverOpenAL              │   │   │
│  │ │                                                    │   │   │
│  │ │ RenderingDeviceVulkan/D3D12/Metal                  │   │   │
│  │ │ VideoDriver::Vulkan/D3D12/Metal                   │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Access:                                                         │
│  通过 ClassDB::get_singleton("RenderingServer")                  │
│  不是独立的共享库！                                              │
│                                                                  │
│  优势：                                                           │
│  ✅ 简单的构建系统                                               │
│  ✅ 零性能开销（无跨库调用）                                     │
│  ✅ ID 系统统一                                                 │
│  ✅ 单例管理简单                                                 │
│                                                                  │
│  限制：                                                           │
│  ❌ 无法选择性加载单独的 server                                  │
│  ❌ 整个二进制必须一起发布                                       │
│  ❌ 无法独立更新 server                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 未来可能的架构（社区提案）

```
┌─────────────────────────────────────────────────────────────────┐
│            未来可能的 LibGodot (运行时模块化)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────┐                                     │
│  │ libgodot_core.so         │                                     │
│  │                          │                                     │
│  │ [Core Engine]            │                                     │
│  │ - OS                     │                                     │
│  │ - MainLoop               │                                     │
│  │ - ClassDB                │                                     │
│  │ - Resource System        │                                     │
│  │ - Script Engine          │                                     │
│  │ - Memory Manager         │                                     │
│  └───────────┬──────────────┘                                     │
│              │                                                   │
│              │ Plugin System                                    │
│              │ ┌──────┐ ┌──────┐ ┌──────┐                        │
│              ├─┤ dlopen│ │dlopen│ │dlopen│                        │
│              │ └──────┘ └──────┘ └──────┘                        │
│              │            │         │                            │
│      ┌───────▼──────────▼─────────▼────────┐                      │
│      │                                        │                      │
│  ┌───▼────┐  ┌──────────────┐  ┌──────────┐  │                      │
│  │Render- │  │  Physics     │  │  Audio   │  │                      │
│  │ingServer│  │  Server      │  │  Server  │  │                      │
│  │.so      │  │  .so         │  │  .so     │  │                      │
│  │         │  │              │  │          │  │                      │
│  │ - Forward│  │ - Jolt       │  │ - OpenAL │  │                      │
│  │   Plus   │  │ - Bullet     │  │ - WASAPI  │  │                      │
│  │ - Mobile │  │ - ODE        │  │ - Pulse   │  │                      │
│  │          │  │              │  │ - ALSA    │  │                      │
│  └─────────┘  └──────────────┘  └──────────┘  │                      │
│                                            │                        │
│      ┌─────────────────────────────────────┼──────────────────┐     │
│      │                                     ▼                  │     │
│      │                         ┌────────────────────┐        │     │
│      │                         │ libgodot_driver.so │        │     │
│      │                         │                    │        │     │
│      │                         │ - Vulkan Driver    │        │     │
│      │                         │ - D3D12 Driver     │        │     │
│      │                         │ - Metal Driver     │        │     │
│      │                         │ - OpenGL Driver    │        │     │
│      │                         └────────────────────┘        │     │
│      └───────────────────────────────────────────────────────┘     │
│                                                                  │
│  优势：                                                           │
│  ✅ 选择性加载所需模块                                            │
│  ✅ 独立更新 modules                                             │
│  ✅ 更小的发布包（最小化二进制）                                  │
│  ✅ 可扩展的插件系统                                             │
│                                                                  │
│  挑战：                                                           │
│  ❌ 跨库调用的性能开销                                            │
│  ❌ ABI 稳定性要求                                               │
│  ❌ 依赖关系管理                                                 │
│  ❌ 调试难度增加                                                 │
│  ❌ Godot 内部 ID 系统需要重新设计                                │
│                                                                  │
│  状态：                                                           │
│  📄 社区提案和讨论中，不是 Godot 4.6 的官方实现！                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 十、LibGodot 4.6 实际调用流程（修正版）

```cpp
// ┌─────────────────────────────────────────────────────────────────┐
// │  正确的 LibGodot 调用流程（Server 层是内部单例）                   │
// └─────────────────────────────────────────────────────────────────┘

// 1. 加载 LibGodot（单一库）
void* libgodot = dlopen("libgodot.so", RTLD_LAZY);

// 2. 获取引擎创建函数
typedef godot::Engine* (*create_engine_func)();
create_engine_func create_engine = (create_engine_func)dlsym(libgodot, "create_engine");

// 3. 创建引擎（内部初始化所有 singleton servers）
godot::Engine* engine = create_engine();
godot::EngineConfig config;
config.project_path = "res://";
config.rendering_driver = "vulkan";
config.physics_driver = "jolt";
engine->initialize(config);

// 4. 访问 Server（通过 ClassDB 单例，不是独立的库！）
godot::RenderingServer* rs = godot::RenderingServer::get_singleton();
godot::PhysicsServer* ps = godot::PhysicsServer::get_singleton();
godot::AudioServer* as = godot::AudioServer::get_singleton();

// 5. 调用 Server API（直接调用，无跨库开销）
RID mesh = rs->mesh_create();  // 直接函数调用
RID space = ps->space_create(); // 直接函数调用
int bus = as->bus_create();     // 直接函数调用

// 6. 主循环
while (running) {
    engine->process_frame();  // 内部调用所有 servers
    // 等效于：
    // - rs->draw();
    // - ps->step();
    // - as->update();
    // ...
}

// 7. 清理
engine->shutdown();
dlclose(libgodot);
```

---

## 十一、总结：澄清 Key Points

### ✅ 正确的理解

1. **LibGodot 是单一共享库（或少数几个库）**
   - 不是拆分成多个独立的 server 库
   - Servers 是内部 Singleton Objects

2. **Server 层是逻辑分层，不是物理分离的库**
   - RenderingServer、PhysicsServer、AudioServer 是设计概念
   - 实现为单例类，通过 ClassDB 管理访问

3. **编译时可以选择模块**
   - 通过 SCons 构建选项
   - 减少二进制大小
   - 但仍然是单一库

4. **零跨库调用开销**
   - 所有 servers 在同一二进制中
   - 直接函数调用
   - 无 IPC 或 dlookup 开销

### ❌ 误解的澄清

1. **RenderingServer 不是独立的 libgodot_rendering.so**
   - 这是社区提案，不是 Godot 4.6 实现

2. **LibGodot 不支持运行时动态加载单独的 servers**
   - 这是未来的可能方向

3. **Server 不是插件系统**
   - Godot 的 Backend 接口是编译时选择的

### 🔍 社区项目与官方 LibGodot 的区别

| 项目 | 类型 | 是否拆分 servers？ | 支持平台 |
|------|------|------------------|---------|
| **Godot 4.6 官方 LibGodot** | 官方 | ❌ 否（单一库） | Linux/Windows/macOS |
| **migueldeicaza/libgodot** | 社区 | ❌ 否（单一库） | 跨平台 |
| **V-Sekai/libgodot-project** | 社区 | 🔄 部分尝试（实验性） | 有限 |
| **SwiftGodotKit** | 社区 | ❌ 否（调用单一库） | macOS/iOS |

---

## 参考资源

• [Godot 4.6 - Godot's Architecture Overview - 官方文档](https://docs.godotengine.org/en/4.6/engine_details/architecture/godot_architecture_diagram.html)

• [Godot Architecture Diagram - 官方文档](https://docs.godotengine.org/en/stable/engine_details/architecture/godot_architecture_diagram.html)

• [Optimization using Servers - Godot Docs](https://docs.godotengine.org/en/stable/tutorials/performance/using_servers.html)

• [Dev snapshot: Godot 4.6 dev 2 - Reddit 讨论](https://www.reddit.com/r/godot/comments/1obv796/dev_snapshot_godot_46_dev_2/)

• [Could anyone explain LibGodot? - Reddit 讨论](https://www.reddit.com/r/godot/comments/1pl6np3/could_anyone_explain_libgodot/)

• [Building, compiling, modules and libraries - Reddit 讨论](https://www.reddit.com/r/godot/comments/mbr5az/building_compiling_modules_and_libraries/)

• [V-Sekai/libgodot-project - GitHub](https://github.com/V-Sekai/libgodot-project)

• [migueldeicaza/libgodot - GitHub](https://github.com/migueldeicaza/libgodot)

---

**总结**：您的观察完全正确！Godot 4.6 LibGodot 的 **RenderingServer、PhysicsServer、AudioServer 等servers 目前还不是独立的 libraries**。它们是打包在 `libgodot.so`（或 `libgodot_core.so`）**内部的 Singleton Objects**，通过 Godot 的 **ClassDB 系统注册和管理**。LibGodot 的模块化是**编译时模块化**（通过 SCons 构建选项），而不是运行时加载独立的共享库。感谢您的认真观察和提问！💪🎯


---

## 二、Layers（分层详解）

### Layer 1: Driver/I/O Layer（底层驱动抽象）

这是最底层，提供硬件和系统服务的统一抽象：

#### Video Drivers（视频驱动）
```cpp
// video_driver.h - 核心接口
class VideoDriver {
public:
    virtual Error initialize() = 0;
    virtual void swap_buffers() = 0;
    virtual RenderingDevice* get_rendering_device() = 0;
};

// Vulkan 实现
class VulkanVideoDriver : public VideoDriver {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice logical_device;
    VkSwapchainKHR swapchain;
    RenderingDeviceVulkan* rendering_device;  // Godot 的 GPU 抽象
    
public:
    Error initialize()override {
        // 创建 Vulkan 实例
        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Godot Embedded";
        app_info.apiVersion = VK_API_VERSION_1_0;
        
        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        
        vkCreateInstance(&create_info, nullptr, &instance);
        
        // 选择物理设备
        physical_device = select_physical_device();
        
        // 创建逻辑设备
        logical_device = create_logical_device();
        
        // 创建交换链
        swapchain = create_swapchain(native_window);
        
        // 创建 RenderingDevice
        rendering_device = new RenderingDeviceVulkan(logical_device);
        
        return OK;
    }
    
    void swap_buffers() override {
        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain;
        
        vkQueuePresentKHR(present_queue, &present_info);
    }
};
```

#### RenderingDevice（GPU 统一抽象）
```
RenderingDevice 是 Godot 对 Vulkan/Direct3D12/Metal 的统一抽象层：

┌─────────────────────────────────────────────────────────┐
│              RenderingDevice Abstraction                 │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────┐  │
│  │  Buffer Management                                │  │
│  │  ├── vertex_buffer_create(size, usage)           │  │
│  │  ├── index_buffer_create(size, usage)            │  │
│  │  └── uniform_buffer_create(size, usage)          │  │
│  ├───────────────────────────────────────────────────┤  │
│  │  Texture Management                               │  │
│  │  ├── texture_2d_create(width, height, format)     │  │
│  │  ├── texture_cube_create(size, format)           │  │
│  │  └── texture_sampler_create(...)                 │  │
│  ├───────────────────────────────────────────────────┤  │
│  │  Rendering Pipeline                               │  │
│  │  ├── render_pipeline_create(shader_rid, state)    │  │
│  │  ├── compute_pipeline_create(shader_rid)         │  │
│  │  └── command_buffer_pool_create()                │  │
│  ├───────────────────────────────────────────────────┤  │
│  │  Draw Commands                                    │  │
│  │  ├── draw_list_begin()                           │  │
│  │  ├── draw_list_bind_pipeline(pipeline_rid)       │  │
│  │  ├── draw_list_bind_uniform_set(set, index)      │  │
│  │  ├── draw_list_draw(vertex_count, instance_count)│  │
│  │  └── draw_list_end()                             │  │
│  └───────────────────────────────────────────────────┘  │
└───────────┬─────────────────────────────────────────────┘
            │
            └── 到后端实现：
                ├── RenderingDeviceVulkan (Vulkan)
                ├── RenderingDeviceD3D12 (Direct3D 12)
                └── RenderingDeviceMetal (Metal)
```

**RenderingDevice 统一接口示例**：
```cpp
// 通过 RenderingDevice 操作 GPU（跨平台）
RenderingDevice* rd = RenderingServer::get_singleton()->get_rendering_device();

// 创建顶点缓冲区
RID vertex_buffer = rd->vertex_buffer_create(
    sizeof(vertices),  // 大小
    vertices,          // 数据
    RenderingDevice::BufferUsageFlags::USAGE_VERTEX_BUFFER_BIT |
    RenderingDevice::BufferUsageFlags::USAGE_TRANSFER_DST_BIT
);

// 创建纹理
RID texture = rd->texture_2d_create(
    1024,                           // 宽度
    1024,                           // 高度
    RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM,
    RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
    RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT
);

// 更新纹理数据（零拷贝，直接 GPU 上传）
rd->texture_update(texture, 0, texture_data);

// 提交绘制
rd->draw_list_bind_uniform_set(uniform_set, 0);
rd->draw_list_draw(4, 1);  // 4个顶点，1个实例
```

---

### Layer 2: Server Layer（服务器层）

这是引擎的核心服务层，提供无状态的服务接口：

#### RenderingServer 架构
```cpp
// rendering_server.h
class RenderingServer : public Object {
    SINGLETON(RenderingServer);
    
public:
    // CanvasItem API (2D 渲染)
    RID canvas_create();
    RID canvas_item_create();
    void canvas_item_add_mesh(RID ci, RID mesh, Transform2D xform, Color modulate);
    void canvas_item_set_transform(RID ci, Transform2D xform);
    void canvas_item_set_visible(RID ci, bool visible);
    
    // World API (3D 渲染)
    RID world_2d_create();
    RID world_3d_create();
    void世界_add_camera(RID world, RID camera);
    void世界_add_light(RID world, RID light);
    void世界_add_mesh(RID world, RID mesh, Transform3D xform);
    
    // Mesh API
    RID mesh_create();
    void mesh_add_surface(RID mesh, Mesh::PrimitiveType primitive, 
                          Array arrays, Array blend_shapes, 
                          Mesh::CompressionMode compression);
    
    // Material/Shader API
    RID shader_create();
    void shader_set_code(RID shader, String code);
    RID material_create();
    void material_set_shader(RID material, RID shader);
    void material_set_param(RID material, StringName param, Variant value);
    
    // Camera API
    RID camera_create();
    void camera_set_perspective(RID camera, float fov_degrees, 
                                 float z_near, float z_far);
    void camera_set_transform(RID camera, Transform3D transform);
};
```

#### RenderingServer 数据流
```
应用代码
    │
    ↓ 调用 RenderingServer API
    │
┌──▼──────────────────────────────────────────────────────┐
│  RenderingServer (前端接口层)                           │
│  - 提供 RID（Resource ID）访问资源                      │
│  - 统一的 API 签名                                     │
│  - 参数验证和错误处理                                   │
└──┬──────────────────────────────────────────────────────┘
   │
   ↓ 分发命令
   │
┌──▼──────────────────────────────────────────────────────┐
│  RenderingServerBackend (后端实现层)                    │
│  ┌───────────────────────────────────────────────┐    │
│  │ ForwardPlusRenderer                            │    │
│  │  - Clustered Lighting                         │    │
│  │  - Depth Pre-pass                             │    │
│  │  - Opaque Pass                                │    │
│  │  - Transparent Pass                           │    │
│  │  - Post-processing                            │    │
│  └───────────────────────────────────────────────┘    │
│  ┌───────────────────────────────────────────────┐    │
│  │ MobileRenderer                                │    │
│  │  - Single-pass lighting                       │    │
│  │  - Tile-based rendering optimization          │    │
│  │  - Sub-pass merging                           │    │
│  └───────────────────────────────────────────────┘    │
└──┬──────────────────────────────────────────────────────┘
   │
   ↓ 调用 RenderingDevice
   │
┌──▼──────────────────────────────────────────────────────┐
│  RenderingDevice (GPU 统一抽象)                         │
│  - 生成命令缓冲区                                       │
│  - 管理同步原语                                         │
│  - 资源生命周期管理                                     │
└──┬──────────────────────────────────────────────────────┘
   │
   ↓ 提交到 GPU 驱动
   │
  Vulkan/D3D12/Metal 驱动 → GPU 硬件
```

#### RenderingServer 完整调用示例
```cpp
// 创建渲染服务器实例
RenderingServer* rs = RenderingServer::get_singleton();

// 1. 创建 3D 世界
RID world_3d = rs->world_3d_create();
RID camera_3d = rs->camera_create();
rs->camera_set_perspective(camera_3d, 75.0, 0.1, 1000.0);

// 2. 创建网格 (立方体)
RID mesh = rs->mesh_create();

// 立方体顶点数据
Vector3 vertices[8] = {
    Vector3(-1, -1, -1), Vector3(1, -1, -1),
    Vector3(1, 1, -1), Vector3(-1, 1, -1),
    Vector3(-1, -1, 1), Vector3(1, -1, 1),
    Vector3(1, 1, 1), Vector3(-1, 1, 1)
};

int indices[36] = {  // 索引缓冲区
    0, 1, 2, 2, 3, 0,  // Front face
    5, 4, 7, 7, 6, 5,  // Back face
    // ... 其他面
};

// 添加网格表面
Array arrays;
arrays.resize(Mesh::ARRAY_MAX);
arrays[Mesh::ARRAY_VERTEX] = vertices;
arrays[Mesh::ARRAY_INDEX] = indices;

rs->mesh_add_surface(
    mesh,
    Mesh::PRIMITIVE_TRIANGLES,
    arrays,      // 顶点数组
    Array(),     // 混合形状
    Mesh::COMPRESSION_LEVEL_2  // 压缩级别
);

// 3. 创建材质和着色器
RID shader = rs->shader_create();
rs->shader_set_code(shader, R"(
    shader_type spatial;
    render_mode unshaded;
    
    void vertex() {
        VERTEX = VERTEX;  // 简单传递顶点
    }
    
    void fragment() {
        ALBEDO = vec3(0.2, 0.5, 0.8);  // 蓝色
    }
)");

RID material = rs->material_create();
rs->material_set_shader(material, shader);
rs->mesh_surface_set_material(mesh, 0, material);

// 4. 添加到世界
rs->world_add_mesh(world_3d, mesh, Transform3D());
rs->world_add_camera(world_3d, camera_3d);

// 5. 渲染循环
while (running) {
    rs->force_draw();
}
```

#### PhysicsServer（物理服务器）
```cpp
// physics_server.h
class PhysicsServer : public Object {
    SINGLETON(PhysicsServer);
    
public:
    // Space API
    RID space_create();
    void space_set_active(RID space, bool active);
    void space_set_param(RID space, SpaceParameter param, Variant value);
    PhysicsDirectBodyState* space_get_direct_state(RID space);
    
    // Body API
    RID body_create();
    void body_set_space(RID body, RID space);
    void body_set_mode(RID body, BodyMode mode);
    void body_set_transform(RID body, Transform3D transform);
    void body_add_shape(RID body, RID shape, Transform3D transform);
    void body_set_linear_velocity(RID body, Vector3 velocity);
    void body_set_angular_velocity(RID body, Vector3 velocity);
    void body_apply_central_impulse(RID body, Vector3 impulse);
    
    // Shape API
    RID sphere_shape_create();
    void shape_set_radius(RID shape, float radius);
    
    RID box_shape_create();
    void shape_set_size(RID shape, Vector3 size);
    
    // Physics Step
    void step(float delta);
};
```

**PhysicsServer 调用流程**：
```
应用代码
    │
    ↓ body_apply_central_impulse(body, Vector3(0, 100, 0))
    │
┌──▼──────────────────────────────────────────────────────┐
│  PhysicsServer (前端接口)                               │
│  - 验证参数                                               │
│  - 将命令加入队列                                         │
└──┬──────────────────────────────────────────────────────┘
   │
   ↓ 分发到后端
   │
┌──▼──────────────────────────────────────────────────────┐
│  PhysicsServer3D (3D 物理后端)                          │
│  ┌─────────────────────────────────────────────┐       │
│  │ JoltPhysicsServer3D (默认，Godot 4.6)       │       │
│  │  - 使用 Jolt Physics 库                    │       │
│  │  - 高性能刚体动力学                         │       │
│  │  - 支持软体和布料                           │       │
│  └─────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────┐       │
│  │ GodotPhysicsServer3D (旧实现，可选)         │       │
│  │  - 原生实现的物理引擎                       │       │
│  └─────────────────────────────────────────────┘       │
└──┬──────────────────────────────────────────────────────┘
   │
   ↓ 执行物理模拟
   │
┌──▼──────────────────────────────────────────────────────┐
│  Physics Simulation (物理循环)                          │
│  ───────────────────────────────────────────────────── │
│  1. Collision Detection (碰撞检测)                     │
│     ├── Broad Phase:  BVH/AABB Tree                    │
│     ├── Narrow Phase: GJK/EPA Algorithm                 │
│     └── Spatial Partition: Grid/Tree                    │
│  ───────────────────────────────────────────────────── │
│  2. Constraint Solver (约束求解)                       │
│     ├── Sequential Impulse (SI)                         │
│     ├── Jacobian-based Solver                          │
│     └── Iterative (默认 8-16 次迭代)                   │
│  ───────────────────────────────────────────────────── │
│  3. Integration (位置积分)                             │
│     ├── Semi-implicit Euler (默认)                     │
│     └── Velocity Verlet (可选)                         │
│  ───────────────────────────────────────────────────── │
│  4. Generate Collisions (生成碰撞信息)                  │
│     ├── Collision Shape: Contact points                │
│     ├── Collision Normal: 法线方向                      │
│     ├── Penetration Depth: 穿透深度                     │
│     └── Impulse Magnitude: 冲量大小                     │
└──┬──────────────────────────────────────────────────────┘
   │
   ↓ 回调通知
   │
│ 应用代码: _physics_process(delta)
```

**PhysicsServer 完整示例**：
```cpp
// 创建物理服务器
PhysicsServer* ps = PhysicsServer::get_singleton();

// 1. 创建物理空间
RID space = ps->space_create();
ps->space_set_active(space, true);
ps->space_set_param(space, PhysicsServer::SPACE_PARAM_GRAVITY, 
                    Vector3(0, -9.81, 0));

// 2. 创建刚体
RID body = ps->body_create();
ps->body_set_space(body, space);
ps->body_set_mode(body, PhysicsServer::BODY_MODE_RIGID);

// 3. 创建形状 (球体)
RID shape = ps->sphere_shape_create();
ps->shape_set_radius(shape, 1.0);
ps->body_add_shape(body, shape, Transform3D());

// 4. 设置物理属性
ps->body_set_mass(body, 10.0);
ps->body_set_friction(body, 0.7);
ps->body_set_bounce(body, 0.5);
ps->body_set_linear_velocity(body, Vector3(10, 0, 0));

// 5. 应用外力
ps->body_apply_central_impulse(body, Vector3(0, 50, 0));

// 6. 物理步进
for (int i = 0; i < substeps; i++) {
    ps->step(1.0 / 60.0 / substeps);  // 固定时间步长
}

// 7. 获取物理状态
PhysicsDirectBodyState* state = ps->space_get_direct_state(space);
Vector3 velocity = state->get_linear_velocity();
Transform3D transform = state->get_transform();
```

#### AudioServer（音频服务器）
```cpp
// audio_server.h
class AudioServer : public Object {
    SINGLETON(AudioServer);
    
public:
    // Bus API
    int bus_create();
    void bus_set_name(int bus, String name);
    void bus_set_volume_db(int bus, float volume_db);
    void bus_set_send(int bus, int send_bus);
    
    // Stream API
    RID audio_stream_create();
    void stream_set_sample(RID stream, RID sample);
    void stream_play(RID stream);
    void stream_stop(RID stream);
    void stream_set_volume(RID stream, float volume);
    void stream_set_pitch_scale(RID stream, float pitch);
    
    // Sample API
    RID sample_create(AudioSample::Format format, bool stereo, 
                      int mix_rate, int data_len);
    void sample_set_data(RID sample, const PackedByteArray& data);
};
```

**AudioServer 架构**：
```
┌─────────────────────────────────────────────────────────┐
│  AudioServer (音频服务器)                                │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────┐    │
│  │ AudioBusSystem (混音总线系统)                │    │
│  │ ┌─────────┐  ┌─────────┐  ┌─────────┐       │    │
│  │ │ Master  │←─│ SFX_Bus │←─│ UI_Bus  │       │    │
│  │ │ (Output)│  │         │  │         │       │    │
│  │ └────┬────┘  └─────────┘  └─────────┘       │    │
│  │      ↓                                    │    │
│  │  ┌─────────────────────────────────────┐  │    │
│  │  │   Audio Mixer (48kHz, stereo)       │  │    │
│  │  │   - 64-bit float samples            │  │    │
│  │  │   - 3D Spatial Audio (HRTF)         │  │    │
│  │  │   - Reverb/Echo Effects             │  │    │
│  │  └─────────────────────────────────────┘  │    │
│  └───────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────┐    │
│  │ AudioStreamPlayer (音频流播放器)             │    │
│  │ ┌───────┐ ┌───────┐ ┌───────┐               │    │
│  │ │Stream1│ │Stream2│ │Stream3│               │    │
│  │ │ (OGG) │ │ (WAV) │ │ (MP3) │               │    │
│  │ └───┬───┘ └───┬───┘ └───┬───┘               │    │
│  │     │         │         │                   │    │
│  └─────┼─────────┼─────────┼───────────────────┘    │
└───────┼─────────┼─────────┼───────────────────────────┘
        ↓         ↓         ↓
    ┌────────────────────────────────────────────────────┐
    │  Audio Driver (音频驱动)                           │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐              │
    │  │ OpenAL  │ │ALSA     │ │WASAPI   │              │
    │  │ ┌─────┐ │ │ ┌─────┐ │ │ ┌─────┐ │              │
    │  │ │HRTF │ │ │ │Pulse│ │ │ │Audio│ │              │
    │  │ │3D   │ │ │ │Audio│ │ │ │Thru │ │              │
    │  │ └─────┘ │ │ └─────┘ │ │ └─────┘ │              │
    │  └─────────┘ └─────────┘ └─────────┘              │
    └────────────────────────────────────────────────────┘
```

---

### Layer 3: Engine Core Layer（引擎核心层）

这是引擎的核心抽象层，提供主循环和全局服务：

#### OS（操作系统抽象）
```cpp
// os.h
class OS : public Object {
    SINGLETON(OS);
    
public:
    // Window Management
    void initialize(int video_driver, int audio_driver);
    void finalize();
    
    void set_window_size(Size2i size);
    void set_window_position(Vector2i position);
    void set_window_fullscreen(bool fullscreen);
    void set_window_resizable(bool resizable);
    
    void set_native_window(void* handle);  // LibGodot 关键！
    
    // Rendering Loop
    void main_loop_begin();
    bool main_loop_iterate();
    void main_loop_end();
    void run_main_loop();
    
    // Time Management
    uint64_t get_ticks_usec();
    uint64_t get_ticks_msec();
    double get_unix_time();
    
    // File I/O
    Error file_open(const String& path, int flags);
    Error file_write(const uint8_t* data, int length);
    Error file_read(uint8_t* buffer, int length);
    String get_data_dir();
    String get_user_data_dir();
};
```

#### MainLoop（主循环抽象）
```cpp
// main_loop.h
class MainLoop : public Object {
    GDCLASS(MainLoop, Object);
    
public:
    virtual void initialize() {}
    virtual void finalize() {}
    
    // 每帧调用
    virtual bool iteration(float delta) { return true; }
    
    // 空闲时调用
    virtual bool idle(float delta) { return true; }
    
    // 物理更新
    virtual void physics_process(float delta) {}
    
    // 输入事件
    virtual void input_event(const Ref<InputEvent>& p_event) {}
};

// SceneTree 实现
class SceneTree : public MainLoop {
    Node* root = nullptr;           // 场景根节点
    Ref<ViewportLayer> viewport;
    bool paused = false;
    float physics_ticks_per_second = 60.0;
    
public:
    void initialize() override;
    bool iteration(float delta) override;
    void physics_process(float delta) override;
    
    // 场景管理
    void change_scene_to_file(const String& path);
    void add_child(Node* p_child, bool force_readable_name = false);
    Node* get_root() const { return root; }
};
```

#### SceneTree 主循环实现
```cpp
// scene_tree.cpp
bool SceneTree::iteration(float delta) {
    // ┌─────────────────────────────────────────────────────────┐
    // │ Frame Start                                             │
    // └─────────────────────────────────────────────────────────┘
    frame_counter++;
    
    // ┌─────────────────────────────────────────────────────────┐
    // │ Input Processing                                        │
    // └─────────────────────────────────────────────────────────┘
    Input* input = Input::get_singleton();
    input->flush_buffered_events();
    
    for (const InputEvent& event : input->get_events()) {
        input_event(event);  // 分发到 SceneGraph
    }
    
    // ┌─────────────────────────────────────────────────────────┐
    // │ Animation Update                                        │
    // └─────────────────────────────────────────────────────────┘
    AnimationPlayer* animation_player = root->get_node<AnimationPlayer>();
    if (animation_player) {
        animation_player->advance(delta);
    }
    
    // ┌─────────────────────────────────────────────────────────┐
    // │ Process Update (_process)                               │
    // └─────────────────────────────────────────────────────────┘
    process_signal_queue();
    
    // 通知所有节点（前序遍历）
    void SceneTree::process_nodes(Node* node, float delta) {
        // 子节点先处理
        for (int i = 0; i < node->get_child_count(); i++) {
            process_nodes(node->get_child(i), delta);
        }
        
        // 处理当前节点
        if (node->is_processing()) {
            node->_process(delta);
        }
    }
    
    process_nodes(root, delta);
    
    // ┌─────────────────────────────────────────────────────────┐
    // │ Physics Process (_physics_process)                         │
    // └─────────────────────────────────────────────────────────┘
    // 计算需要多少个子步进
    float physics_step = 1.0 / physics_ticks_per_second;
    float steps_frame = delta / physics_step;
    int steps = (int)steps_frame;
    
    PhysicsServer* physics = PhysicsServer::get_singleton();
    
    for (int i = 0; i < steps; i++) {
        // 通知节点
        void SceneTree::physics_process_nodes(Node* node) {
            for (int i = 0; i < node->get_child_count(); i++) {
                physics_process_nodes(node->get_child(i));
            }
            
            if (node->is_physics_processing()) {
                node->_physics_process(physics_step);
            }
        }
        
        physics_process_nodes(root);
        
        // 执行物理步进
        physics->step(physics_step);
    }
    
    // ┌─────────────────────────────────────────────────────────┐
    // │ Render                                                  │
    // └─────────────────────────────────────────────────────────┘
    RenderingServer* rendering = RenderingServer::get_singleton();
    
    // 更新相机
    update_viewports();
    
    // 提交绘制命令
    rendering->frame_pre_draw();   // 渲染前准备
    
    if (viewport.is_valid()) {
        viewport->get_viewport_rid();  // 渲染到视口
    }
    
    rendering->draw();           // 实际绘制
    rendering->frame_post_draw();  // 渲染后处理
    
    // ┌─────────────────────────────────────────────────────────┐
    // │ Audio Update                                            │
    // └─────────────────────────────────────────────────────────┘
    AudioServer* audio = AudioServer::get_singleton();
    audio->update();
    
    // ┌─────────────────────────────────────────────────────────┐
    // │ Garbage Collection                                      │
    // └─────────────────────────────────────────────────────────┘
    CoreStringName::increment_frame_index();
    
    // ┌─────────────────────────────────────────────────────────┐
    // │ End Frame                                                │
    // └─────────────────────────────────────────────────────────┘
    return true;
}
```

#### Resource System（资源系统）
```cpp
// resource_loader.h
class ResourceLoader : public Object {
    SINGLETON(ResourceLoader);
    
public:
    // 资源加载接口
    Ref<Resource> load(const String& path, 
                       const String& type_hint = "",
                       CacheMode cache_mode = CACHE_MODE_REUSE);
    
    void set_abort_on_missing_resources(bool abort);
    
    // 支持的格式
    static const char* get_imported_file_format();  // .import
    static const char* get_resource_format();       // .res, .scn, .tscn
    static const char* get_packed_format();          // .pck
};

// Resource 缓存系统
class ResourceCache {
private:
    HashMap<String, Ref<Resource>> cache;
    HashMap<String, uint64_t> last_used;
    
public:
    void add(const String& path, Ref<Resource> resource) {
        cache[path] = resource;
        last_used[path] = OS::get_singleton()->get_ticks_usec();
    }
    
    Ref<Resource> get(const String& path) {
        if (cache.has(path)) {
            last_used[path] = OS::get_singleton()->get_ticks_usec();
            return cache[path];
        }
        return nullptr;
    }
    
    void cleanup_old_resources(uint64_t max_age_usec) {
        uint64_t current_time = OS::get_singleton()->get_ticks_usec();
        
        for (auto& entry : cache) {
            if (current_time - last_used[entry.key] > max_age_usec) {
                unload(entry.key);
            }
        }
    }
    
    void unload(const String& path) {
        if (cache.has(path)) {
            cache.erase(path);
            last_used.erase(path);
        }
    }
};
```

---

### Layer 4: API Binding Layer（API 绑定层）

这是最上层，提供各种编程语言的绑定：

#### GDExtension API（C 接口）
```c
// gdextension_interface.h (C API)
struct GDExtensionInterface {
    // === Core API ===
    GDExtensionInterfaceVariantGetPtrConstructorFunc variant_get_ptr_constructor;
    GDExtensionInterfaceVariantGetPtrBuiltinMethodFunc variant_get_ptr_builtin_method;
    GDExtensionInterfaceVariantGetPtrOperatorEvaluatorFunc variant_get_ptr_operator_evaluator;
    GDExtensionInterfaceVariantGetPtrSetterFunc variant_get_ptr_setter;
    GDExtensionInterfaceVariantGetPtrGetterFunc variant_get_ptr_getter;
    
    // === Object API ===
    GDExtensionInterfaceObjectMethodBindCallFunc object_method_bind_call;
    GDExtensionInterfaceObjectMethodBindPtrcallFunc object_method_bind_ptrcall;
    GDExtensionInterfaceObjectDestroyFunc object_destroy;
    GDExtensionInterfaceGlobalDef godot_global_get_singleton;
    
    // === String API ===
    GDExtensionInterfaceStringNewWithUtf8CharsFunc string_new_with_utf8_chars;
    GDExtensionInterfaceStringToUtf8CharsFunc string_to_utf8_chars;
    GDExtensionInterfaceStringLengthFunc string_length;
    
    // === Memory API ===
    GDExtensionInterfaceMemAllocFunc mem_alloc;
    GDExtensionInterfaceMemReallocFunc mem_realloc;
    GDExtensionInterfaceMemFreeFunc mem_free;
    
    // === Print API ===
    GDExtensionInterfacePrintErrorFunc print_error;
    GDExtensionInterfacePrintWarningFunc print_warning;
    
    // === ClassDB API ===
    GDExtensionInterfaceClassdbRegisterClassFunc classdb_register_class;
    GDExtensionInterfaceClassdbRegisterMethodFunc classdb_register_method;
    GDExtensionInterfaceClassdbRegisterPropertyFunc classdb_register_property;
    
    // === Thread API ===
    GDExtensionInterfaceThreadCreateFunc thread_create;
    GDExtensionInterfaceThreadJoinFunc thread_join;
    
    // === Mutex API ===
    GDExtensionInterfaceMutexCreateFunc mutex_create;
    GDExtensionInterfaceMutexLockFunc mutex_lock;
    GDExtensionInterfaceMutexUnlockFunc mutex_unlock;
    
    // === RefCounted API ===
    GDExtensionInterfaceRefIncreaseReferenceFunc ref_increase_reference;
    GDExtensionInterfaceRefDecreaseReferenceFunc ref_decrease_reference;
};

// C API 使用示例（通过 GDExtension 创建自定义类）
GDExtensionClassLibraryPtr* library = nullptr;

// 自定义类信息
const GDExtensionClassCreationInfo my_class_info = {
    .is_virtual = false,
    .set_func = nullptr,
    .get_func = nullptr,
    .get_property_list_func = nullptr,
    .free_property_list_func = nullptr,
    .property_can_revert_func = nullptr,
    .property_get_revert_func = nullptr,
    .notification_func = &my_class_notification,  // 通知回调
    .to_string_func = nullptr,
    .reference_func = nullptr,
    .unreference_func = nullptr,
    .get_virtual_func = nullptr,
    .class_userdata = nullptr,
};

// 注册类
void GDExtensionInit(GDExtensionInterfaceGetProcAddress p_get_proc_address,
                     GDExtensionClassLibraryPtr p_library,
                     GDExtensionInitialization* r_initialization)
{
    library = p_library;
    
    // 获取 GDExtension 接口
    struct GDExtensionInterface* godot_interface = (struct GDExtensionInterface*)p_get_proc_address("GodotExtensionInterface");
    
    // 初始化回调函数指针
    godot_interface->variant_get_ptr_constructor = (GDExtensionInterfaceVariantGetPtrConstructorFunc)p_get_proc_address("godot_variant_get_ptr_constructor");
    // ... 更多初始化
    
    // 注册自定义类
    godot_interface->classdb_register_class(
        library,
        "MyCustomClass",              // 类名
        "RefCounted",                 // 父类
        &my_class_info                // 类信息
    );
    
    // 填充初始化结构
    r_initialization->initialize = &initialize_my_module;
    r_initialization->deinitialize = &deinitialize_my_module;
    r_initialization->minimum_initialization_level = GDEXTENSION_INITIALIZATION_SCENE;
}
```

----

## 三、Callflow（调用流程）

### 完整的 LibGodot 生命周期调用流

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                    Phase 1: Library Loading (库加载)                              │
└───────────────────────────────────────────────────────────────────────────────────┘

Host Application (C++/Swift/Python/etc.)
    │
    │ dlopen("libgodot.so") / LoadLibrary("godot.dll")
    │
    ↓
┌───────────────────────────────────────────────────────────────────────────────────┐
│  libgodot.so (Shared Library)                                                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │ Static Initialization (静态初始化)                                         │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │ 1. Core Libraries 初始化                                                      │  │
│  │ ┌──────────────────────────────────────────────────────────────┐          │  │
│  │ │ Memory Allocator: godot_memory_init()                         │          │  │
│  │ │    ├── PoolAllocator (小对象 < 4KB)                           │          │  │
│  │ │    ├── LargeAllocator (大对象 > 4KB)                          │          │  │
│  │ │    └── RefCounted Pool                                        │          │  │
│  │ ├──────────────────────────────────────────────────────────────┤          │  │
│  │ │ Thread System: thread_global_init()                           │          │  │
│  │ │    ├── Main Thread ID 记录                                     │          │  │
│  │ │    └── TLS (Thread Local Storage) 初始化                      │          │  │
│  │ ├──────────────────────────────────────────────────────────────┤          │  │
│  │ │ ClassDB Initialization: classdb_init()                        │          │  │
│  │ │    ├── Register all engine classes                            │          │  │
│  │ │    ├── Node, MeshInstance3D, Camera3D, etc.                   │          │  │
│  │ │    └── Method/Property metadata setup                         │          │  │
│  │ └──────────────────────────────────────────────────────────────┘          │  │
│  │                                                                             │  │
│  │ 2. StringName Internment                                                  │  │
│  │ ┌─────────────────────────────────────────────────────────────────────┐   │  │
│  │ │ StringName::global_internment_table = HashMap<String, uint32_t>       │   │  │
│  │ │ ┌──────────┬───────────┬─────────┐                                 │   │  │
│  │ │ │ "transform"│ "rotation"│ "scale" │  ← 预定义常用字符串            │   │  │
│  │ │ └──────────┴───────────┴─────────┘                                 │   │  │
│  │ └─────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │ 3. Memory Pool Initialization                                              │  │
│  │ ┌────────────────────────────────────────────────────────────────────┐    │  │
│  │ │ Variant::TypePool (16MB)      - Variant 类型池                       │    │  │
│  │ │ String::CharPool (8MB)       - String 字符池                        │    │  │
│  │ │ Array::ElementPool (4MB)     - Array 元素池                         │    │  │
│  │ │ Dictionary::EntryPool (2MB)  - Dictionary 条目池                    │    │  │
│  │ └────────────────────────────────────────────────────────────────────┘    │  │
│  │                                                                             │  │
│  │ 4. Global Singleton Initialization                                         │  │
│  │ ┌────────────────────────────────────────────────────────────────────┐    │  │
│  │ │ ProjectSettings::singleton   ← 项目配置单例                         │    │  │
│  │ │ OS::singleton                  ← OS 抽象单例                         │    │  │
│  │ │ Input::singleton              ← 输入系统单例                       │    │  │
│  │ │ FileAccess::singleton         ← 文件访问单例                       │    │  │
│  │ │ DisplayServer::singleton      ← 显示服务器单例                     │    │  │
│  │ │ ThemeDB::singleton            ← 主题数据库单例                     │    │  │
│  │ │ TranslationServer::singleton ← 翻译服务单例                       │    │  │
│  │ │ AudioServer::singleton        ← 音频服务器单例                     │    │  │
│  │ │ RenderingServer::singleton    ← 渲染服务器单例                     │    │  │
│  │ │ PhysicsServer::singleton     ← 物理服务器单例                     │    │  │
│  │ └────────────────────────────────────────────────────────────────────┘    │  │
│  │                                                                             │  │
│  │ 完成时间: ~2-5ms                                                            │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────┘
    │
    │ libgodot.so 完成静态初始化
    │
    ↓
Host Application
    │
    │ engine = godot::Engine::create()
    │
    ↓
┌───────────────────────────────────────────────────────────────────────────────────┐
│                    Phase 2: Engine Creation (引擎创建)                           │
└───────────────────────────────────────────────────────────────────────────────────┘

Engine* godot::Engine::create()
    │
    ↓
┌───────────────────────────────────────────────────────────────────────────────────┐
│  Engine Instance Construction                                                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │ Engine::Engine()                                                              │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │ 1. Allocate Engine Instance                                                   │  │
│  │    engine_instance = memnew_custom(Engine, sizeof(Engine));                  │  │
│  │                                                                            │  │
│  │ 2. Initialize Engine State                                                   │  │
│  │    engine_state = EngineState::UNINITIALIZED                                │  │
│  │    main_loop = nullptr                                                       │  │
│  │    scene_tree = nullptr                                                      │  │
│  │                                                                            │  │
│  │ 3. Setup Thread Safety                                                       │  │
│  │    main_thread_id = Thread::get_current_id();                               │  │
│  │    lock = Mutex::create();                                                   │  │
│  │                                                                            │  │
│  │ 4. Prepare for Dynamic Initialization                                        │  │
│  │    config = EngineConfig::default();                                         │  │
│  │                                                                            │  │
│  │ 完成时间: < 0.1ms                                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────┘
    │
    │ 返回 Engine* 实例指针
    │
    ↓
Host Application
    │
    │ engine->initialize(config)
    │
    ↓
┌───────────────────────────────────────────────────────────────────────────────────┐
│              Phase 3: Engine Initialization (引擎初始化)                        │
└───────────────────────────────────────────────────────────────────────────────────┘

void Engine::initialize(const EngineConfig& config)
    │
    ↓
┌───────────────────────────────────────────────────────────────────────────────────┐
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: Validate Configuration (配置验证)                                   │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  检查配置参数:                                                                 │  │
│  │  - project_path 是否有效                                                    │  │
│  │  - rendering_driver 是否支持 (Vulkan/D3D12/Metal/OpenGL)                   │  │
│  │  - physics_driver 是否支持 (Jolt/Godot)                                   │  │
│  │  - window_mode 是否有效 (Embedded/Fullscreen/Windowed)                    │  │
│  │                                                                             │  │
│  │  失败: 返回 Error::ERR_INVALID_PARAMETER                                    │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ Step 2: Load Project Settings (加载项目配置)                                │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  ProjectSettings* ps = ProjectSettings::get_singleton();                    │  │
│  │                                                                             │  │
│  │  // 加载 project.godot 文件                                                   │  │
│  │  Error err = ps->load_resource(config.project_path + "/project.godot");     │  │
│  │                                                                             │  │
│  │  [application]                                                             │  │
│  │  config/name="MyGame"                                                      │  │
│  │  config/description="Awesome game"                                         │  │
│  │  config/features=PackedStringArray("4.6", "Embedded")                      │  │
│  │  run/main_scene="res://scenes/main.tscn"                                  │  │
│  │  config/icon="res://icon.png"                                             │  │
│  │                                                                             │  │
│  │  [display]                                                                 │  │
│  │  window/size/viewport_width=1920                                          │  │
│  │  window/size/viewport_height=1080                                         │  │
│  │  window/stretch/mode="viewport"                                           │  │
│  │                                                                             │  │
│  │  [rendering]                                                               │  │
│  │  rendering_driver/opengl3                                                 │  │
│  │  renderer/rendering_method="forward_plus"                                 │  │
│  │                                                                             │  │
│  │  [physics]                                                                 │  │
│  │  physics_engine="jolt"                                                     │  │
│  │  3d/default_gravity=Vector3(0, -9.8, 0)                                   │  │
│  │                                                                             │  │
│  │ 完成时间: ~5-10ms (取决于项目文件大小)                                       │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ Step 3: Initialize OS Layer (初始化 OS 层)                                 │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  OS* os = OS::get_singleton();                                               │  │
│  │                                                                             │  │
│  │  // 3.1 Initialize Video Driver                                              │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ VideoDriver* video_driver = OS::create_video_driver(                 │   │  │
│  │  │     config.rendering_driver                                          │   │  │
│  │  │ );                                                                    │   │  │
│  │  │                                                                       │   │  │
│  │  │ if (video_driver == nullptr) {                                       │   │  │
│  │  │     ERR_PRINT("Failed to create video driver");                      │   │  │
│  │  │     return Error::ERR_CANT_CREATE;                                   │   │  │
│  │  │ }                                                                     │   │  │
│  │  │                                                                       │   │  │
│  │  │ Error video_err = video_driver->initialize();                         │   │  │
│  │  │ if (video_err != OK) {                                                │   │  │
│  │  │     return video_err;                                                 │   │  │
│  │  │ }                                                                     │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 3.2 Initialize Audio Driver                                             │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ AudioDriver* audio_driver = OS::create_audio_driver(                 │   │  │
│  │  │     config.audio_driver                                              │   │  │
│  │  │ );                                                                    │   │  │
│  │  │                                                                       │   │  │
│  │  │ Error audio_err = audio_driver->initialize(&audio_data);              │   │  │
│  │  │                                                                       │   │  │
│  │  │ // Audio 默认 48kHz, stereo                                                │  │
│  │  │ audio_data.mix_rate = 48000;                                           │   │  │
│  │  │ audio_data.speaker_mode = SPEAKER_MODE_STEREO;                         │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 3.3 Initialize Window (LibGodot 关键！)                                  │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ if (config.window_mode == EngineConfig::WINDOW_MODE_EMBEDDED) {      │   │  │
│  │  │     // 嵌入模式：使用原生窗口句柄                                         │   │  │
│  │  │     os->set_native_window(config.native_window_handle);                │   │  │
│  │  │     os->set_window_title(config.window_title);                         │   │  │
│  │  │     os->set_window_size(config.window_size);                           │   │  │
│  │  │     os->set_window_position(config.window_position);                   │   │  │
│  │  │ }                                                                       │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~10-20ms (驱动初始化)                                              │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ Step 4: Initialize Resource System (初始化资源系统)                        │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  ResourceLoader* rl = ResourceLoader::get_singleton();                      │  │
│  │  ResourceSaver* rs = ResourceSaver::get_singleton();                        │  │
│  │                                                                             │  │
│  │  // 4.1 Register Importers (注册导入器)                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ ResourceFormatImporter* importer = ResourceFormatImporter::create();│   │  │
│  │  │                                                                       │   │  │
│  │  │ // Image Importers                                                    │   │  │
│  │  │ importer->add_importer(new ResourceImporterTexture());              │   │  │
│  │  │ // .png, .jpg, .webp, .tga, .bmp, .hdr, .exr                         │   │  │
│  │  │                                                                       │   │  │
│  │  │ // Mesh Importers                                                     │   │  │
│  │  │ importer->add_importer(new ResourceImporterMesh());                 │   │  │
│  │  │ // .glb, .gltf, .fbx, .obj, .blend                                   │   │  │
│  │  │                                                                       │   │  │
│  │  │ // Audio Importers                                                    │   │  │
│  │  │ importer->add_importer(new ResourceImporterAudio());                │   │  │
│  │  │ // .wav, .ogg, .mp3                                                  │   │  │
│  │  │                                                                       │   │  │
│  │  │ // Shader Importers                                                   │   │  │
│  │  │ importer->add_importer(new ResourceImporterShader());               │   │  │
│  │  │ // .gdshader                                                         │   │  │
│  │  │                                                                       │   │  │
│  │  │ // Scene Importers                                                    │   │  │
│  │  │ importer->add_importer(new ResourceImporterScene());                │   │  │
│  │  │ // .tscn, .scn                                                       │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 4.2 Setup Resource Cache (设置资源缓存)                                  │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ ResourceCache* cache = ResourceCache::get_singleton();              │   │  │
│  │  │                                                                       │   │  │
│  │  │ cache->set_max_size(256 * 1024 * 1024);  // 256MB                    │   │
│  │  │ cache->set_cleanup_interval(5.0);          // 每 5 秒清理一次          │   │  │
│  │  │ cache->set_auto_cache(true);              // 自动缓存                  │   │  │
│  │  │                                                                       │   │  │
│  │  │ // 预加载常用资源 (可选)                                                  │   │
│  │  │ if (config.preload_common_resources) {                               │   │  │
│  │  │     Ref<ShaderMaterial> default_material =                            │   │  │
│  │  │         rl->load("res://default_material.tres", "ShaderMaterial");    │   │  │
│  │  │ }                                                                     │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │ 完成时间: ~5-10ms                                                            │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ Step 5: Initialize Rendering Server (初始化渲染服务器)                      │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  RenderingServer* rs = RenderingServer::get_singleton();                    │  │
│  │                                                                             │  │
│  │  // 5.1 Create Rendering Context                                            │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ RenderingContext* context = video_driver->get_rendering_context();   │   │  │
│  │  │                                                                       │   │  │
│  │  │ // Vulkan Context                                                     │   │  │
│  │  │ if (context->is_vulkan()) {                                          │   │  │
│  │  │     RenderingDeviceVulkan* device = context->get_device();          │   │  │
│  │  │                                                                       │   │  │
│  │  │     // 创建默认渲染管线                                                  │   │  │
│  │  │     RID pipeline = device->screen_triangle_pipeline_create();        │   │  │
│  │  │ }                                                                     │   │  │
│  │  │                                                                       │   │  │
│  │  │ // Direct3D 12 Context                                                │   │  │
│  │  │ else if (context->is_d3d12()) {                                      │   │  │
│  │  │     RenderingDeviceD3D12* device = context->get_device();            │   │  │
│  │  │ }                                                                     │   │  │
│  │  │                                                                       │   │  │
│  │  │ // Metal Context                                                      │   │  │
│  │  │ else if (context->is_metal()) {                                       │   │  │
│  │  │     RenderingDeviceMetal* device = context->get_device();            │   │  │
│  │  │ }                                                                     │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 5.2 Create Default Resources (创建默认资源)                              │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ // 黑色纹理 (默认背景)                                                 │   │  │
│  │  │ RID black_texture = rs->texture_2d_create(                           │   │  │
│  │  │     4, 4,                                                          │   │  │
│  │  │     RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM                      │   │  │
│  │  │ );                                                                  │   │  │
│  │  │ rs->texture_update(black_texture, 0, {0, 0, 0, 0});                 │   │  │
│  │  │                                                                       │   │  │
│  │  │ // 白色纹理 (默认材质)                                                 │   │  │
│  │  │ RID white_texture = rs->texture_2d_create(                           │   │  │
│  │  │     4, 4,                                                          │   │  │
│  │  │     RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM                      │   │  │
│  │  │ );                                                                  │   │  │
│  │  │ rs->texture_update(white_texture, 0, {255, 255, 255, 255});         │   │  │
│  │  │                                                                       │   │  │
│  │  │ // 默认着色器                                                          │   │  │
│  │  │ RID default_shader = rs->shader_create();                            │   │  │
│  │  │ rs->shader_set_code(default_shader, DEFAULT_SHADER_CODE);            │   │  │
│  │  │                                                                       │   │  │
│  │  │ // 默认材质                                                            │   │  │
│  │  │ RID default_material = rs->material_create();                        │   │  │
│  │  │ rs->material_set_shader(default_material, default_shader);           │   │  │
│  │  │ rs->material_set_param(default_material, "albedo", Color(1, 1, 1)); │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~20-50ms (GPU 初始化)                                              │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ Step 6: Initialize Physics Server (初始化物理服务器)                        │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  PhysicsServer* ps = PhysicsServer::get_singleton();                        │  │
│  │                                                                             │  │
│  │  // 6.1 Create Physics Engine Instance                                     │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ if (config.physics_driver == "jolt") {                              │   │  │
│  │  │     JoltPhysicsServer3D* jolt = new JoltPhysicsServer3D();           │   │  │
│  │  │     jolt->initialize();                                             │   │  │
│  │  │     ps->set_active_backend(jolt);                                   │   │  │
│  │  │ }                                                                     │   │  │
│  │  │ else if (config.physics_driver == "godot") {                         │   │  │
│  │  │     GodotPhysicsServer3D* godot_phys = new GodotPhysicsServer3D();    │   │  │
│  │  │     godot_phys->initialize();                                       │   │  │
│  │  │     ps->set_active_backend(godot_phys);                              │   │  │
│  │  │ }                                                                     │   │  │
│  │  │                                                                       │   │  │
│  │  │ // 配置物理参数                                                         │   │  │
│  │  │ ps->space_set_param(                                                  │   │  │
│  │  │     default_physics_space,                                            │   │  │
│  │  │     PhysicsServer::SPACE_PARAM_GRAVITY,                              │   │  │
│  │  │     Vector3(0, -9.81, 0)                                             │   │  │
│  │  │ );                                                                    │   │  │
│  │  │                                                                       │   │  │
│  │  │ ps->space_set_param(                                                  │   │  │
│  │  │     default_physics_space,                                            │   │  │
│  │  │     PhysicsServer::SPACE_PARAM_LINEAR_DAMPING,                       │   │  │
│  │  │     0.1                                                             │   │  │
│  │  │ );                                                                    │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~5-10ms                                                           │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ Step 7: Initialize Audio Server (初始化音频服务器)                          │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  AudioServer* as = AudioServer::get_singleton();                            │  │
│  │                                                                             │  │
│  │  // 7.1 Setup Audio Bus System                                              │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ // 创建 "Master" 总线 (输出)                                           │   │  │
│  │  │ int master_bus = as->bus_create();                                  │   │  │
│  │  │ as->bus_set_name(master_bus, "Master");                             │   │  │
│  │  │ as->bus_set_volume_db(master_bus, 0.0);                              │   │  │
│  │  │                                                                       │   │  │
│  │  │ // 创建 "SFX" 总线 (音效)                                              │   │  │
│  │  │ int sfx_bus = as->bus_create();                                      │   │  │
│  │  │ as->bus_set_name(sfx_bus, "SFX");                                    │   │  │
│  │  │ as->bus_set_volume_db(sfx_bus, 0.0);                                 │   │  │
│  │  │ as->bus_set_send(sfx_bus, master_bus);                               │   │  │
│  │  │                                                                       │   │  │
│  │  │ // 创建 "Music" 总线 (音乐)                                            │   │  │
│  │  │ int music_bus = as->bus_create();                                    │   │  │
│  │  │ as->bus_set_name(music_bus, "Music");                                │   │  │
│  │  │ as->bus_set_volume_db(music_bus, -3.0);  // -3dB (略低于主音量)       │   │  │
│  │  │ as->bus_set_send(music_bus, master_bus);                             │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~3-5ms                                                            │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ Step 8: Initialize Scene System (初始化场景系统)                             │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  // 8.1 Create SceneTree                                                   │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ scene_tree = memnew(SceneTree());                                │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 设置状态                                                         │   │  │
│  │  │ scene_tree->set_physics_process_mode(SceneTree::PHYSICS_PROCESS_IDLE);│  │
│  │  │ scene_tree->set_auto_accept_quit(false);  // 嵌入模式不自动退出       │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 设置物理参数                                                     │   │  │
│  │  │ scene_tree->set_physics_ticks_per_second(60);                     │   │  │
│  │  │ scene_tree->set_max_physics_steps_per_frame(8);                   │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 8.2 Create Root Node                                                  │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ root = memnew(Node());                                             │   │  │
│  │  │ root->set_name("root");                                            │   │  │
│  │  │ scene_tree->add_child(root);                                       │   │  │
│  │  │ scene_tree->set_root_scene(root);                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 根节点进入场景树                                                    │   │  │
│  │  │ root->_enter_tree();                                               │   │  │
│  │  │ root->_ready();      // 触发 _ready() 回调                          │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 8.3 Setup Viewport (如果需要)                                          │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ if (config.create_default_viewport) {                              │   │  │
│  │  │     Viewport* viewport = memnew(Viewport());                       │   │  │
│  │  │     viewport->set_path("/root/Viewport");                          │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 创建相机                                                        │   │  │
│  │  │     Camera3D* camera = memnew(Camera3D());                         │   │  │
│  │  │     camera->set_path("/root/Viewport/Camera3D");                   │   │  │
│  │  │     camera->set_perspective(75.0, 0.1, 1000.0);                   │   │  │
│  │  │     viewport->add_child(camera);                                   │   │  │
│  │  │                                                                    │   │  │
│  │  │     root->add_child(viewport);                                     │   │  │
│  │  │ }                                                                  │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~2-5ms                                                            │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ Step 9: Connect MainLoop (连接主循环)                                       │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  OS* os = OS::get_singleton();                                               │  │
│  │  os->set_main_loop(scene_tree);                                              │  │
│  │                                                                             │  │
│  │  engine_state = EngineState::INITIALIZED;                                   │  │
│  │                                                                             │  │
│  │  完成时间: < 0.1ms                                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────┘
    │
    │ 初始化完成
    │ 总时间: ~50-120ms (取决于配置和硬件)
    │
    ↓
Host Application
    │
    │ engine->process_frame()
    │   (每一帧调用)
    │
    ↓
┌───────────────────────────────────────────────────────────────────────────────────┐
│            Phase 4: Main Loop Iteration (主循环帧迭代 - 每帧 ~16.7ms)            │
└───────────────────────────────────────────────────────────────────────────────────┘

void Engine::process_frame()
    │
    ↓
┌───────────────────────────────────────────────────────────────────────────────────┐
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Time Management (时间管理)                                              │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  OS* os = OS::get_singleton();                                               │  │
│  │                                                                             │  │
│  │  // 获取当前时间戳                                                           │  │
│  │  uint64_t current_time = os->get_ticks_usec();                              │  │
│  │                                                                             │  │
│  │  // 计算 delta 时间                                                         │  │
│  │  float delta = (current_time - last_frame_time) / 1000000.0;               │  │
│  │  last_frame_time = current_time;                                           │  │
│  │                                                                             │  │
│  │  // 限制 delta 避免时间跳变                                                  │  │
│  │  if (delta > 0.1) {  // > 100ms 视为跳帧                                     │  │
│  │      delta = 0.0167;  // 使用 60fps 默认值                                   │  │
│  │  }                                                                         │  │
│  │                                                                             │  │
│  │  // 更新全局时间                                                             │  │
│  │  Engine::singleton->set_frames_per_second(1.0 / delta);                     │  │
│  │  Engine::singleton->set_physics_jitter_fix(0);                             │  │
│  │                                                                             │  │
│  │  完成时间: < 0.01ms                                                           │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ 2. Input Processing (输入处理)                                             │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  Input* input = Input::get_singleton();                                      │  │
│  │  DisplayServer* display = DisplayServer::get_singleton();                  │  │
│  │                                                                             │  │
│  │  // 2.1 Process Platform Events                                           │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ while (display->has_event()) {                                    │   │  │
│  │  │     DisplayServer::Event event = display->pop_event();            │   │  │
│  │  │                                                                    │   │  │
│  │  │     switch (event.type) {                                         │   │  │
│  │  │         case DisplayServer::EVENT_WINDOW_CLOSE_REQUEST:          │   │  │
│  │  │             // 嵌入模式：通知宿主应用而非直接退出                       │   │  │
│  │  │             on_window_close_requested();                           │   │  │
│  │  │             break;                                                 │   │  │
│  │  │                                                                    │   │  │
│  │  │         case DisplayServer::EVENT_MOUSE_MOTION:                   │   │  │
│  │  │             input->set_mouse_position(                              │   │  │
│  │  │                 event.mouse_motion.position                        │   │  │
│  │  │             );                                                     │   │  │
│  │  │             break;                                                 │   │  │
│  │  │                                                                    │   │  │
│  │  │         case DisplayServer::EVENT_MOUSE_BUTTON:                   │   │  │
│  │  │             Ref<InputEventMouseButton> mb = memnew(InputEventMouseButton);│ │  │
│  │  │             mb->set_button_index(event.mouse_button.button_index); │   │  │
│  │  │             mb->set_pressed(event.mouse_button.pressed);           │   │  │
│  │  │             mb->set_position(event.mouse_button.position);         │   │  │
│  │  │             input->parse_input_event(mb);                          │   │  │
│  │  │             break;                                                 │   │  │
│  │  │                                                                    │   │  │
│  │  │         case DisplayServer::EVENT_KEY:                            │   │  │
│  │  │             Ref<InputEventKey> key = memnew(InputEventKey);        │   │  │
│  │  │             key->set_keycode(event.key.keycode);                   │   │  │
│  │  │             key->set_pressed(event.key.pressed);                   │   │  │
│  │  │             input->parse_input_event(key);                          │   │  │
│  │  │             break;                                                 │   │  │
│  │  │     }                                                              │   │  │
│  │  │ }                                                                │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 2.2 Action Map Processing                                            │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ // 检查输入动作映射                                                     │   │  │
│  │  │ if (input->is_action_just_pressed("ui_up")) {                       │   │  │
│  │  │     // 创建动作事件                                                   │   │  │
│  │  │     Ref<InputEventAction> action = memnew(InputEventAction);        │   │  │
│  │  │     action->set_action("ui_up");                                    │   │  │
│  │  │     action->set_pressed(true);                                      │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 分发到场景树                                                   │   │  │
│  │  │     scene_tree->input_event(action);                               │   │  │
│  │  │ }                                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 处理游戏手柄输入                                                   │   │  │
│  │  │ if (input->is_action_just_pressed("jump")) {                        │   │  │
│  │  │     scene_tree->input_event(...);                                   │   │  │
│  │  │ }                                                                  │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~0.1-0.5ms (取决于输入事件数量)                                     │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ 3. Process Signal Queue (信号队列处理)                                      │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  // 处理延迟的信号 (Deferred Signals)                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ while (!signal_queue.is_empty()) {                                 │   │  │
│  │  │     SignalCall call = signal_queue.pop_front();                    │   │  │
│  │  │                                                                    │   │  │
│  │  │     Object* caller = call.caller;                                  │   │  │
│  │  │     StringName signal = call.signal_name;                          │   │  │
│  │  │     Object* target = call.target;                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 调用信号回调                                                   │   │  │
│  │  │     target->emit_signal(signal, call.args);                        │   │  │
│  │  │ }                                                                  │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~0.01-0.1ms (取决于信号数量)                                      │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ 4. Animation Update (_animation_update)                                    │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  // 更新所有 AnimationPlayer                                              │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ void SceneTree::_update_animations(float delta) {                  │   │  │
│  │  │     AnimationPlayer* player = AnimationPlayer::singleton;          │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 播放速度                                                       │   │  │
│  │  │     float speed_scale = player->get_speed_scale();                 │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 更新动画位置                                                   │   │  │
│  │  │     player->advance(delta * speed_scale);                          │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 应用动画变换到目标节点                                           │   │  │
│  │  │     for (AnimationTrack& track : player->get_tracks()) {          │   │  │
│  │  │         Node* target = track.get_target();                         │   │  │
│  │  │                                                                    │   │  │
│  │  │         if (track.type == Animation::TYPE_POSITION_3D) {          │   │  │
│  │  │             Vector3 position = track.interpolate(                   │   │  │
│  │  │                 player->get_position(), track.key_times            │   │  │
│  │  │             );                                                     │   │  │
│  │  │             target->set_position(position);                         │   │  │
│  │  │         }                                                          │   │  │
│  │  │         else if (track.type == Animation::TYPE_ROTATION_3D) {     │   │  │
│  │  │             Quaternion rotation = track.interpolate(                │   │  │
│  │  │                 player->get_position(), track.key_times            │   │  │
│  │  │             );                                                     │   │  │
│  │  │             target->set_quaternion(rotation);                       │   │  │
│  │  │         }                                                          │   │  │
│  │  │         // ... 其他属性 (scale, blend_shape, etc.)                   │   │  │
│  │  │     }                                                              │   │  │
│  │  │ }                                                                  │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~0.5-2.0ms (取决于动画复杂度)                                       │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ 5. Process Update (_process)                                                │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  // 每帧调用所有启用的节点的 _process 方法                                   │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ void SceneTree::process_nodes(Node* node, float delta) {           │   │  │
│  │  │     // 后序遍历：子节点先处理，父节点后处理 (自下而上)                   │   │  │
│  │  │     for (int i = 0; i < node->get_child_count(); i++) {            │   │  │
│  │  │         process_nodes(node->get_child(i), delta);                  │   │  │
│  │  │     }                                                              │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 处理当前节点                                                   │   │  │
│  │  │     if (node->is_processing()) {                                  │   │  │
│  │  │         // 检查暂停状态                                               │   │  │
│  │  │         if (!node->is_physics_processing() &&                     │   │  │
│  │  │             !node->get_tree()->is_paused()) {                      │   │  │
│  │  │             node->_process(delta);  // 调用 _process()              │   │  │
│  │  │         }                                                          │   │  │
│  │  │     }                                                              │   │  │
│  │  │ }                                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ process_nodes(root, delta);  // 从根节点开始遍历                      │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  示例 GDScript _process:                                                     │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ extends Area3D                                                      │   │  │
│  │  │                                                                    │   │  │
│  │  │ func _process(delta):                                              │   │  │
│  │  │     # 移动物体                                                       │   │  │
│  │  │     position.x += speed * delta                                    │   │  │
│  │  │                                                                    │   │  │
│  │  │     # 更新动画                                                       │   │  │
│  │  │     if is_moving:                                                   │   │  │
│  │  │         play_animation("walk")                                     │   │  │
│  │  │     else:                                                           │   │  │
│  │  │         play_animation("idle")                                     │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~1.0-5.0ms (取决于节点数量 _process 逻辑)                            │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ 6. Physics Processing (_physics_process)                                  │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  // 固定步长的物理模拟                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ float physics_step = 1.0 / scene_tree->get_physics_ticks_per_second();│  │  │
│  │  │ float steps_frame = delta / physics_step;                          │   │  │
│  │  │ int steps = (int)steps_frame;                                      │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 累积误差处理                                                         │   │  │
│  │  │ frame_time_accumulator += (steps_frame - steps) * physics_step;    │   │  │
│  │  │ if (frame_time_accumulator > physics_step) {                       │   │  │
│  │  │     steps++;                                                       │   │  │
│  │  │     frame_time_accumulator -= physics_step;                        │   │  │
│  │  │ }                                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 执行物理子步进                                                       │   │  │
│  │  │ PhysicsServer* ps = PhysicsServer::get_singleton();                │   │  │
│  │  │ for (int i = 0; i < steps; i++) {                                  │   │  │
│  │  │     _physics_process_step(physics_step);                           │   │  │
│  │  │ }                                                                  │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 物理单步进                                                              │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ void SceneTree::_physics_process_step(float step) {               │   │  │
│  │  │     // 通知节点 _physics_process                                      │   │  │
│  │  │     void SceneTree::physics_process_nodes(Node* node, float step) {│  │  │
│  │  │         for (int i = 0; i < node->get_child_count(); i++) {        │   │  │
│  │  │             physics_process_nodes(node->get_child(i), step);       │   │  │
│  │  │         }                                                          │   │  │
│  │  │                                                                    │   │  │
│  │  │         if (node->is_physics_processing()) {                       │   │  │
│  │  │             node->_physics_process(step);                          │   │  │
│  │  │         }                                                          │   │  │
│  │  │     }                                                              │   │  │
│  │  │                                                                    │   │  │
│  │  │     physics_process_nodes(root, step);                            │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 执行物理模拟                                                     │   │  │
│  │  │     PhysicsServer* ps = PhysicsServer::get_singleton();            │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 6.1 Collision Detection (碰撞检测)                         │   │  │
│  │  │     ps->collision_detection_step();                                │   │  │
│  │  │     // ├── Broad Phase: AABB Tree / BVH                           │   │  │
│  │  │     // ├── Narrow Phase: GJK / EPA Algorithm                       │   │  │
│  │  │     // └── Optimization: Spatial hashing                           │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 6.2 Constraint Solving (约束求解)                           │   │  │
│  │  │     ps->constraint_solve_step();                                   │   │  │
│  │  │     // ├── Jacobian-based Solver                                  │   │  │
│  │  │     // ├── Iterative (8-16 iterations)                            │   │  │
│  │  │     // └── Warm-starting                                           │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 6.3 Integration (位置积分)                                   │   │  │
│  │  │     ps->integrate_forces_step(step);                               │   │  │
│  │  │     // ├── Semi-implicit Euler (默认)                              │   │  │
│  │  │     // └── Velocity = Velocity + Acceleration * step               │   │  │
│  │  │     //     Position = Position + Velocity * step                   │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 6.4 Generate Collision Events (生成碰撞事件)                  │   │  │
│  │  │     ps->generate_collision_events();                               │   │  │
│  │  │     // ├── _on_body_entered() / _on_body_exited()                 │   │  │
│  │  │     // ├── _on_area_entered() / _on_area_exited()                 │   │  │
│  │  │     // └── Update physics queries (RayCast3D, etc.)              │   │  │
│  │  │ }                                                                  │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  示例 GDScript _physics_process:                                            │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ extends RigidBody3D                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ func _physics_process(delta):                                      │   │  │
│  │  │     # 物理力的应用                                                     │   │  │
│  │  │     if is_jumping:                                                   │   │  │
│  │  │         apply_central_impulse(Vector3(0, jump_force, 0))           │   │  │
│  │  │                                                                    │   │  │
│  │  │     # 移动物体                                                         │   │  │
│  │  │     linear_velocity.x = move_speed                                 │   │  │
│  │  │                                                                    │   │  │
│  │  │     # 旋转物体                                                         │   │  │
│  │  │     angular_velocity.y = rotation_speed                             │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~1.0-10.0ms (取决于物理步进次数和物理对象数量)                          │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ 7. Transform Update (变换更新)                                            │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  // 更新场景树中的所有变换 (World Transform)                                 │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ void Node::update_world_transform() {                             │   │  │
│  │  │     // 计算全局变换                                                     │   │  │
│  │  │     if (parent != nullptr) {                                       │   │  │
│  │  │         world_transform = parent->world_transform * local_transform; │  │  │
│  │  │     } else {                                                        │   │  │
│  │  │         world_transform = local_transform;                         │   │  │
│  │  │     }                                                              │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 通知子节点更新                                                   │   │  │
│  │  │     for (int i = 0; i < get_child_count(); i++) {                  │   │  │
│  │  │         get_child(i)->update_world_transform();                    │   │  │
│  │  │     }                                                              │   │  │
│  │  │ }                                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 递归更新所有节点                                                   │   │  │
│  │  │ root->update_world_transform();                                   │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  变换计算公式：                                                                │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ Local Transform (局部变换):                                        │   │  │
│  │  │   T_local = R_local × P_local × S_local                            │   │  │
│  │  │   (Rotation × Position × Scale)                                    │   │  │
│  │  │                                                                    │   │  │
│  │  │ World Transform (全局变换):                                         │   │  │
│  │  │   T_world = T_parent × T_child                                     │   │  │
│  │  │                                                                    │   │  │
│  │  │ Matrix4x4 组成:                                                      │   │  │
│  │  │   [ R00 R01 R02 Tx ]                                               │   │  │
│  │  │   [ R10 R11 R12 Ty ]                                               │   │  │
│  │  │   [ R20 R21 R22 Tz ]                                               │   │  │
│  │  │   [ 0   0   0   1  ]                                               │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~0.5-2.0ms (取决于节点数量)                                         │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ 8. Rendering Draw (渲染绘制)                                                │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  RenderingServer* rs = RenderingServer::get_singleton();                    │  │
│  │                                                                             │  │
│  │  // 8.1 Frame Pre-Draw (渲染前准备)                                         │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ rs->frame_pre_draw();                                              │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 8.1.1 Culling (视锥剔除)                                           │   │  │
│  │  │ Vector<Camera3D*> cameras = scene_tree->get_cameras();              │   │  │
│  │  │ for (Camera3D* camera : cameras) {                                 │   │  │
│  │  │     Frustum frustum = camera->get_frustum();                       │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 剔除不在视锥内的物体                                              │   │  │
│  │  │     void cull_node(Node* node, Frustum frustum,                    │   │  │
│  │  │                    List<RID>& visible_meshes) {                     │   │  │
│  │  │         if (!frustum.contains(node->get_aabb())) {                  │   │  │
│  │  │             return;  // 剔除此节点                                    │   │  │
│  │  │         }                                                          │   │  │
│  │  │                                                                    │   │  │
│  │  │         if (node->is_class("MeshInstance3D")) {                    │   │  │
│  │  │             visible_meshes.push_back(                              │   │  │
│  │  │                 node->cast_to<MeshInstance3D>()->get_mesh_rid()      │   │  │
│  │  │             );                                                     │   │  │
│  │  │         }                                                          │   │  │
│  │  │                                                                    │   │  │
│  │  │         for (int i = 0; i < node->get_child_count(); i++) {        │   │  │
│  │  │             cull_node(node->get_child(i), frustum, visible_meshes);│   │  │
│  │  │         }                                                          │   │  │
│  │  │     }                                                              │   │  │
│  │  │                                                                    │   │  │
│  │  │     visible_meshes.clear();                                         │   │  │
│  │  │     cull_node(root, frustum, visible_meshes);                     │   │  │
│  │  │ }                                                                  │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 8.1.2 Sorting (深度排序)                                             │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ // 透明物体的深度排序 (从后往前)                                        │   │  │
│  │  │ void sort_transparent_objects(List<RID>& meshes, Camera3D* camera) {│   │  │
│  │  │     meshes.sort([&](RID a, RID b) {                                 │   │  │
│  │  │         float dist_a = camera->get_distance_to(a.get_position());    │   │  │
│  │  │         float dist_b = camera->get_distance_to(b.get_position());    │   │  │
│  │  │         return dist_a > dist_b;  // 远的先绘制                        │   │  │
│  │  │     });                                                             │   │  │
│  │  │ }                                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ sort_transparent_objects(transparent_objects, camera);             │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 8.2 Draw Opaque Pass (不透明物体渲染)                                    │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ for (RID mesh_id : visible_opaque_meshes) {                       │   │  │
│  │  │     RenderData* mesh_data = rs->get_mesh_data(mesh_id);            │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 获取着色器和材质                                                  │   │  │
│  │  │     RID shader = mesh_data->material->get_shader();                │   │  │
│  │  │     RID pipeline = rs->get_pipeline(shader);                       │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 设置 Uniform Set (着色器参数)                                 │   │  │
│  │  │     RID uniform_set = rs->create_uniform_set(                      │   │  │
│  │  │         uniform_data, shader, 0                                    │   │  │
│  │  │     );                                                             │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 计算模型-视图-投影 (MVP) 矩阵                                   │   │  │
│  │  │     Transform3D model = mesh_data->transform;                      │   │  │
│  │  │     Transform3D view = camera->get_transform().inverse();           │   │  │
│  │  │     Transform3D proj = camera->get_projection();                   │   │  │
│  │  │     ProjectionMatrix mvp = proj * view * model;                    │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 设置 Uniform 变量                                              │   │  │
│  │  │     rs->set_uniform(uniform_set, 0, mvp);   // MVP 矩阵             │   │  │
│  │  │     rs->set_uniform(uniform_set, 1, mesh_data->material_params);  // 材质参数 │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 绑定 Uniform Set                                              │   │  │
│  │  │     rs->draw_list_bind_uniform_set(uniform_set, 0);                │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 绘制网格                                                        │   │  │
│  │  │     rs->draw_list_bind_pipeline(pipeline);                         │   │  │
│  │  │     rs->draw_list_bind_vertex_buffer(mesh_data->vertex_buffer);    │   │  │
│  │  │     rs->draw_list_bind_index_buffer(mesh_data->index_buffer);      │   │  │
│  │  │     rs->draw_list_draw(                                            │   │  │
│  │  │         mesh_data->index_count,                                     │   │  │
│  │  │         mesh_data->instance_count                                   │   │  │
│  │  │     );                                                             │   │  │
│  │  │ }                                                                  │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 8.3 Draw Transparent Pass (透明物体渲染)                                  │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ for (RID mesh_id : sorted_transparent_meshes) {                    │   │  │
│  │  │     // ... (类似不透明物体渲染，但需要启用 Alpha Blending)            │   │  │
│  │  │     rs->draw_list_enable_alpha_blending(true);                     │   │  │
│  │  │ }                                                                  │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 8.4 Post-Processing (后期处理)                                         │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ // 将屏幕内容绘制到 Framebuffer                                        │   │  │
│  │  │ RID framebuffer = rs->get_framebuffer();                           │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 应用后期处理效果                                                     │   │  │
│  │  │ if (bloom_enabled) {                                               │   │  │
│  │  │     apply_bloom_effect(framebuffer, bloom_threshold, bloom_intensity);│  │  │
│  │  │ }                                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ if (ssr_enabled) {                                                 │   │  │
│  │  │     apply_screen_space_reflections(framebuffer);                   │   │  │
│  │  │ }                                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ if (motion_blur_enabled) {                                        │   │  │
│  │  │     apply_motion_blur(framebuffer, previous_framebuffer);           │   │  │
│  │  │ }                                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ // Tone Mapping (色调映射)                                           │   │  │
│  │  │ apply_tone_mapping(framebuffer, tone_mapping_mode);                │   │  │
│  │  │                                                                    │   │  │
│  │  │ // Gamma Correction (伽马校正)                                        │   │  │
│  │  │ apply_gamma_correction(framebuffer, 2.2);                          │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  // 8.5 Frame Post-Draw (渲染后)                                           │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ rs->frame_post_draw();                                             │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 提交绘制到 GPU                                                     │   │  │
│  │  │ RenderingDevice* rd = rs->get_rendering_device();                  │   │  │
│  │  │ rd->submit_queue();                                                │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 等待 GPU 渲染完成                                                   │   │  │
│  │  │ rd->draw_list_end();                                               │   │  │
│  │  │ rd->present();  // 交换缓冲区                                         │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~2.0-10.0ms (取决于场景复杂度、光照数量、后期处理效果)                    │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ 9. Audio Update (音频更新)                                                 │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  AudioServer* as = AudioServer::get_singleton();                            │  │
│  │                                                                             │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ // 更新音频混音器                                                      │   │  │
│  │  │ as->get_audio_driver()->mix_audio_streams();                         │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 更新所有播放的音频流                                                  │   │  │
│  │  │ for (AudioStreamPlayer* player : active_audio_players) {           │   │  │
│  │  │     if (player->is_playing()) {                                    │   │  │
│  │  │         player->advance_samples(delta);                            │   │  │
│  │  │         player->apply_pitch_scale();                               │   │  │
│  │  │         player->apply_volume();                                    │   │  │
│  │  │     }                                                              │   │  │
│  │  │ }                                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 应用音频效果 (混响、回声等)                                          │   │  │
│  │  │ as->apply_audio_effects();                                         │   │  │
│  │  │                                                                    │   │  │
│  │  │ // 处理音频事件                                                        │   │  │
│  │  │ process_audio_events();                                            │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~0.1-1.0ms (取决于音频流数量和效果)                                   │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ 10. Garbage Collection (垃圾回收)                                          │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  // 清理不再使用的资源 (引用计数为 0)                                         │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ void CoreStringName::increment_frame_index() {                    │   │  │
│  │  │     frame_index++;                                                 │   │  │
│  │  │                                                                    │   │  │
│  │  │     // 每 100 帧执行一次清理                                           │   │  │
│  │  │     if (frame_index % 100 == 0) {                                   │   │  │
│  │  │         cleanup_unused_resources();                                │   │  │
│  │  │         cleanup_unused_string_names();                             │   │  │
│  │  │     }                                                              │   │  │
│  │  │ }                                                                  │   │  │
│  │  │                                                                    │   │  │
│  │  │ void cleanup_unused_resources() {                                 │   │  │
│  │  │     ResourceCache* cache = ResourceCache::get_singleton();        │   │  │
│  │  │     uint64_t max_age = 5.0 * 1000000;  // 5 秒                      │   │  │
│  │  │     cache->cleanup_old_resources(max_age);                         │   │  │
│  │  │ }                                                                  │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  │  完成时间: ~0.01-0.1ms (取决于清理频率和资源数量)                               │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │ 11. End Frame (帧结束)                                                      │  │
│  │ ───────────────────────────────────────────────────────────────────────── │  │
│  │  frame_counter++;                                                          │  │
│  │                                                                             │  │
│  │  // 计算帧率                                                                 │  │
│  │  Engine::singleton->set_frames_per_second(1.0 / delta);                     │  │
│  │                                                                             │  │
│  │  总帧时间: ~5-25ms (60FPS @ 16.7ms 目标)                                     │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、架构对比：Standalone vs LibGodot 嵌入

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Standalone Godot (传统模式)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────┐            ┌──────────────────┐
 │  Host Application│            │  Godot.exe       │
 │     (C++/C#)     │            │  (Standalone)    │
 └────────┬─────────┘            └────────┬─────────┘
          │                                │
          │ Process::spawn()              │
          │ ─────────────────────────────→│
          │                                │
          │ OS Fork / CreateProcess       │
          │                                │
          │         ┌──────────────────────┼─────────────────────┐
          │         │                      │                     │
          ↓         ↓                      ↓                     ↓
     ┌─────────────────────────────────────────────────────────────┐
     │                    Separate Processes                       │
     │  ┌──────────────────┐    ┌──────────────────┐              │
     │  │  Process A       │    │  Process B       │              │
     │  │  ┌────────────┐  │    │  ┌────────────┐  │              │
     │  │  │ App Logic │  │    │  │   Engine   │  │              │
     │  │  │ ┌────────┐ │  │    │  │ ┌────────┐ │  │              │
     │  │  │ │ Memory │ │  │    │  │ │ Memory │ │  │              │
     │  │  │ │ Pool A  │ │  │    │  │ │ Pool B  │ │  │              │
     │  │  │ └────────┘ │  │    │  │ └────────┘ │  │              │
     │  │  └────────────┘  │    │  └────────────┘  │              │
     │  │                  │    │ ┌────────────┐   │              │
     │  │                  │    │ │ OS Abstr. │   │              │
     │  │                  │    │ │ - Video   │   │              │
     │  │                  │    │ │ - Audio   │   │              │
     │  │                  │    │ │ - Window  │   │              │
     │  │                  │    │ └────────────┘   │              │
     │  └──────────────────┘    └──────────────────┘              │
     └─────────────────────────────────────────────────────────────┘
                                   │
                                   │ Inter-Process Communication
                                   │ ───────────────────────────┐
                                   │                             │
          ┌────────────────────────┴────────┐  ┌──────────────────┴──────────┐
          │                                 │  │                            │
          ↓                                 ↓  ↓                            ↓
   ┌─────────────┐                ┌─────────────────────┐      ┌─────────────────────┐
   │  Pipe/SHM   │                │  Windows Messages   │      │   HTTP/WebSocket   │
   │  (Data)     │                │  (Window Control)   │      │  (Control/Sync)    │
   └─────────────┘                └─────────────────────┘      └─────────────────────┘
          │                                 │                            │
          │ +0.001 - 0.010s overhead       │ +0.001 - 0.005s overhead  │ +0.005 - 0.020s overhead
          │ (Serialization)               │ (Message Queue)           │ (Network Stack)
          │                                 │                            │

通信开销分析：
┌──────────────────┬────────────────────────┬──────────────────────────┐
│  Operation       │  Standalone Mode      │  LibGodot Embedded       │
├──────────────────┼────────────────────────┼──────────────────────────┤
│ Function Call    │ ~1-10ms (IPC)         │ ~0-0.01ms (Direct)      │
│ Memory Access    │ ~0.1-1ms (Copy)       │ ~0ms (Shared)           │
│ Data Transfer    │ ~0.5-5ms (Serialize)  │ ~0ms (Zero-Copy)        │
│ Resource Upload  │ ~1-10ms (IPC+Copy)    │ ~0.1-1ms (Direct)       │
│ Input Latency    │ ~3-10ms (Queue)       │ ~0.1-1ms (Direct)       │
└──────────────────┴────────────────────────┴──────────────────────────┘
```

---

## 五、LibGodot 嵌入式架构优势

### 1. **Zero-Copy 内存访问**

```cpp
// 传统模式（IPC 需要序列化）
class StandaloneCommunication {
public:
    void send_texture_data(uint8_t* data, int size) {
        // 需要序列化到共享内存
        serialize_to_shared_memory(data, size);
        // 发送通知
        send_ipc_notification("texture_ready");
        // 等待接收确认
        wait_for_acknowledgment();
        // 总时间: ~1-10ms
    }
};

// LibGodot 嵌入模式（直接指针访问）
class LibGodotEmbedded {
public:
    void update_texture_directly(RID texture, uint8_t* data, int size) {
        RenderingServer* rs = RenderingServer::get_singleton();
        
        // 直接上传到 GPU (零拷贝)
        rs->texture_update(texture, 0, data);
        // 总时间: ~0.1-1ms
    }
};
```

### 2. **统一上下文共享**

```cpp
// 场景：宿主应用需要直接访问 Godot 资源
class EmbeddedScenario {
public:
    void integrate_native_ui_with_godot() {
        // 创建原生 UI (Qt/Win32/etc.)
        QPushButton* button = new QPushButton();
        
        // 在同一个内存空间内直接访问 Godot 资源
        RenderingServer* rs = RenderingServer::get_singleton();
        SceneTree* st = Engine::get_singleton()->get_scene_tree();
        
        // 获取 Godot 场景中的节点
        Node* player_node = st->get_root()->get_node("Player");
        CharacterBody3D* player = Object::cast_to<CharacterBody3D>(player_node);
        
        // 原生 UI 按钮直接调用 Godot 方法
        QObject::connect(button, SIGNAL(clicked()), [=]() {
            // 零延迟调用
            player->call("jump");  // 直接函数调用，无 IPC
        });
        
        // Godot 节点直接通知原生 UI
        player->connect("score_changed", [=](int new_score) {
            // 无序列化，直接更新
            score_label->setText(QString::number(new_score));
        });
    }
};
```

### 3. **扩展性示例（Rust 嵌入）**

```rust
// Rust 通过 GDExtension 嵌入 LibGodot
use godot::prelude::*;
use godot::classes::{Engine, Node, RenderingServer};

#[derive(GodotClass)]
#[class(base = Node)]
struct RustEmbeddedNode {
    #[base]
    base: Base<Node>,
}

#[godot_api]
impl RustEmbeddedNode {
    #[func]
    fn init(&mut self) {
        // 从 Rust 访问 LibGodot API
        let engine = Engine::singleton();
        let rendering_server = RenderingServer::singleton();
        
        // 创建 Godot 资源（从 Rust）
        let mesh = rendering_server.mesh_create();
        // ... 配置 mesh
        
        godot_print!("Rust node initialized with LibGodot!");
    }
    
    #[func]
    fn process(&mut self, delta: f64) {
        // 处理帧逻辑（从 Rust）
        godot_print!("Processing frame from Rust: {} seconds", delta);
    }
}

fn init_godot_embedded_rust() {
    // 初始化 LibGodot
    Engine::singleton()
        .set_max_fps(60)
        .set_physics_ticks_per_second(60);
}
```

---

## 六、参考资源

• [Godot 4.6 - Engine Architecture Overview - 官方文档](https://docs.godotengine.org/en/4.6/engine_details/architecture/godot_architecture_diagram.html)

• [Godot 4.6 - Internal Rendering Architecture - 官方文档](https://docs.godotengine.org/en/stable/engine_details/architecture/internal_rendering_architecture.html)

• [LibGodot - Embed Godot Engine Everywhere - Gergely Kis (GodotCon Boston 2025)](https://talks.godotengine.org/godotcon-us-2025/talk/XBJFYV/)

• [Godot Modified to be Embeddable - GitHub (migueldeicaza/libgodot)](https://github.com/migueldeicaza/libgodot)

• [Build Godot Engine as a shared library - GitHub Issue](https://github.com/godotengine/godot-proposals/issues/4773)

• [SwiftGodotKit: Embed Godot into Swift apps - GitHub (migueldeicaza/SwiftGodotKit)](https://github.com/migueldeicaza/SwiftGodotKit)

• [LibGodot Project Reference - GitHub (migeran/libgodot)](https://github.com/migeran/libgodot)

• [RenderingDevice Godot Docs](https://swiftpackageindex.com/migueldeicaza/SwiftGodot/main/documentation/swiftgodot/renderingserver)

---

**总结**：LibGodot 通过其分层架构（Driver/IO层、Server层、Engine Core层、API Binding层）和清晰的调用流（从库加载到主循环帧迭代），将 Godot 引擎从一个**独立游戏引擎**转变为一个**通用 3D 引擎库**，使得任何本地应用（C++、Swift、Rust、Python 等）都能够以零拷贝、低延迟的方式深度集成 Godot 的渲染、物理、音频和脚本功能。