












好的，我来从第一性原理出发，深入剖析 **rive-app/rive-runtime** 这个 project 到底在干什么。

---

## 一句话总结

**rive-app/rive-runtime** 是 Rive 的 **底层 C++ runtime 与 GPU vector renderer**，它负责在各个平台（Web, iOS, Android, Flutter, Desktop, 甚至 IoT）上 **解码 `.riv` 二进制格式 → 驱动 state machine → 渲染 2D vector graphics**，是整个 Rive 生态的 "engine room"。

> GitHub repo: [https://github.com/rive-app/rive-runtime](https://github.com/rive-app/rive-runtime)

---

## 1. 从第一性原理理解 Rive Runtime 存在的 reason

### 1.1 问题空间

传统 animation pipeline 是这样的：

```
Designer (After Effects) → JSON/Lottie → Runtime (Lottie-web/Android/iOS) → Canvas/SVG rendering
```

这 pipeline 有几个 fundamental 问题：

| 问题 | 根因 |
|---|---|
| **Linear animation only** | Lottie 本质是 keyframe interpolation，没有 state 概念 |
| **JSON overhead** | Lottie JSON 动辄几百 KB，复杂 animation 上 MB |
| **No interactivity** | Animation 是 "play and forget"，无法响应 user input |
| **Rendering quality** | 依赖各平台 Canvas/SVG 实现，不一致且慢 |
| **No real-time** | 不可能 60fps 复杂 vector animation on low-end device |

### 1.2 Rive 的 first-principle 重新设计

Rive 从根本重新思考：**如果 animation 是 interactive、stateful 的 first-class citizen，整个 pipeline 应该怎样？**

推导出的核心 architecture：

```
Rive Editor (design tool)
     ↓ export
   .riv (binary, stateful format)
     ↓ load
   Rive Runtime (C++ core)
     ├── .riv decoder / deserializer
     ├── State Machine engine
     ├── Animation engine (linear interpolation, bones, IK)
     └── Renderer (GPU-accelerated vector path renderer)
     ↓ platform bindings
   iOS / Android / Flutter / Web / C# / React / ...
```

**rive-runtime 就是上述 pipeline 中 "C++ core" 那一整块**。

---

## 2. Project 结构深度解析

根据 GitHub repo 的 directory layout，rive-runtime 包含以下关键 submodule：

```
rive-runtime/
├── include/rive/          # Public C++ API headers
│   ├── artboard.hpp       # Artboard — the "scene" container
│   ├── animation/         # LinearAnimation, Interpolator
│   ├── state_machine/     # StateMachine, State, Transition, Input
│   ├── shape/             # Shape, Path, Vertex
│   ├── text/              # Text run, Font subsystem
│   ├── bones/             # Bone, IK, Skinnable
│   ├── data_bind/         # Data binding (new feature)
│   └── math/              # Mat2D, AABB, etc.
├── src/                   # Implementation (.cpp)
├── renderer/              # ★ THE GPU RENDERER ★
│   ├── src/
│   │   ├── draw/          # Draw path abstraction
│   │   ├── tess/          # Tesselation (CPU-side path → triangle mesh)
│   │   ├── gpu/          # GPU backend: Metal / Vulkan / OpenGL / D3D
│   │   └── text/         # Font rendering via raw bezier
│   └── include/rive/renderer/
├── rivinfo/               # CLI tool: inspect .riv files
├── test/                  # Unit & integration tests
└── skia/                  # Optional Skia renderer backend
```

### 2.1 核心数据模型 (Object Model)

Rive 的 object model 是一个严格的 DAG (Directed Acyclic Graph)：

```
File
 └── Artboard₁, Artboard₂, ...
      ├── Shape
      │    ├── Path (cubic bezier commands)
      │    ├── Paint (fill / stroke)
      │    └── Effect (blur, shadow, etc.)
      ├── Bone
      │    └── Skinnable → weights on vertices
      ├── TextValueRun
      ├── StateMachine
      │    ├── State (any state, animation state, entry state)
      │    ├── Transition
      │    │    ├── Condition (input compare, event)
      │    │    └── Trigger (fire event, set input)
      │    └── Input (boolean, number, trigger)
      └── LinearAnimation
           ├── KeyedProperty
           └── Interpolator (cubic, ease, elastic...)
```

**关键公式 — Animation interpolation:**

$$v(t) = v_0 \cdot (1 - s(t)) + v_1 \cdot s(t)$$

其中：
- $v(t)$ 是 property 在时间 $t$ 的值
- $v_0, v_1$ 是两个 keyframe 的值
- $s(t) \in [0, 1]$ 是由 interpolator (cubic bezier) 计算的 normalized time

$$s(t) = B_{\text{cubic}}\left(\frac{t - t_0}{t_1 - t_0}; P_1, P_2\right)$$

其中 $P_1 = (x_1, y_1), P_2 = (x_2, y_2)$ 是 cubic bezier 的两个 control point，定义 easing curve。

---

## 3. State Machine Engine — Rive 的 "大脑"

这是 Rive 区别于 Lottie 的最核心 feature。State Machine 是一个 **finite state automaton**：

```
              [Input: isHover = true]
    Idle ────────────────────────────────► Hover
     │                                       │
     │ [Input: click = true]                 │ [Input: isHover = false]
     ▼                                       ▼
   Active ──────────────────────────────── Idle
```

### 3.1 State Machine 的 formal model

一个 State Machine $\mathcal{M}$ 是一个 tuple：

$$\mathcal{M} = (S, s_0, I, O, T, E)$$

| Symbol | Meaning | Rive 对应 |
|---|---|---|
| $S$ | Finite set of states | `StateInstance` (any state, animation state) |
| $s_0$ | Initial state | Entry State |
| $I$ | Input alphabet | `SMIInput` (BooleanInput, NumberInput, TriggerInput) |
| $O$ | Output alphabet | Events (`rive::Event`) |
| $T$ | Transition function $T: S \times I \rightarrow S$ | `Transition` with `Condition` |
| $E$ | Set of events | `rive::Event` fired on transition |

### 3.2 Transition evaluation per frame

每一 frame，runtime 执行：

```python
for state in active_states:
    for transition in state.outgoing_transitions:
        if evaluate_conditions(transition.conditions, inputs):
            fire_transition(transition)
            apply_animation(transition.animation)
            emit_events(transition.events)
```

**Condition evaluation 的具体逻辑：**

| Condition Type | Formula |
|---|---|
| `InputBooleanCondition` | `input.value == expected` |
| `InputNumberCondition` | `input.value < bound` or `input.value > bound` |
| `EventCondition` | `event.occurred_in_frame == true` |

---

## 4. The Renderer — Rive 的 "心脏"

这是 rive-runtime 中最技术 dense 的部分。Rive 的 renderer 是一个 **from-scratch, GPU-accelerated 2D vector path renderer**，不走 Skia 那种 "heavyweight library" 的路线。

### 4.1 Rendering pipeline 总览

```
.riv binary
   ↓ deserialize
Artboard (scene graph)
   ↓ advance (animation + state machine)
Dirty shapes/paths
   ↓
[CPU Side: Tessellation]
   Path (cubic bezier commands)
   → triangulate into mesh
   → compute stroke geometry
   → generate gradient fills
   ↓
[GPU Side]
   Vertex Buffer + Index Buffer
   → Draw calls (Metal / Vulkan / GL / D3D)
   → Framebuffer output
```

### 4.2 Path Tessellation 算法

Rive 的 renderer 使用一种 **iterative flattening + fan triangulation** 的方法：

**Step 1: Cubic Bezier → Flatten to line segments**

$$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3$$

其中 $P_0, P_1, P_2, P_3$ 是 cubic bezier 的 4 个 control point。

通过 recursive subdivision 直到 chord distance 小于 tolerance $\epsilon$：

$$\|B(t_{mid}) - \frac{B(t_0) + B(t_1)}{2}\| < \epsilon$$

**Step 2: Fan triangulation for fill**

给定一个 closed polygon $V = [v_0, v_1, ..., v_n]$，triangulate 为 fan：

$$\text{Triangles} = \{(v_0, v_i, v_{i+1}) \mid i = 1, ..., n-1\}$$

**Step 3: Stroke expansion**

对每个 line segment，向法线方向 offset $d$ 生成 quad：

$$v_{i}^{+} = v_i + d \cdot \hat{n}_i, \quad v_{i}^{-} = v_i - d \cdot \hat{n}_i$$

在 join 和 cap 处使用 round join (arc → tessellated triangles) 或 miter join。

### 4.3 GPU Backend Abstraction

Rive renderer 的 GPU 抽象层：

```cpp
// Platform-agnostic render context
class RenderContext {
    virtual void drawPath(const RenderPath* path, const RenderPaint* paint) = 0;
};

// Metal backend
class RenderContextMetal : public RenderContext { ... };

// Vulkan backend  
class RenderContextVulkan : public RenderContext { ... };

// OpenGL backend
class RenderContextGL : public RenderContext { ... };

// Direct3D backend
class RenderContextD3D : public RenderContext { ... };
```

每个 backend 负责：
- Buffer allocation (`VertexBuffer`, `IndexBuffer`)
- Shader compilation & binding
- Draw call submission
- Blend mode management

### 4.4 Vector Feathering — Rive 的独创

传统 2D renderer 处理 "soft edge" 的方法是 **raster blur** (对 path 做 gaussian blur)。Rive 的 Chris Dalton 团队从 scratch 发明了 **Vector Feathering**：

核心 idea：**不 blur raster，而是在 vector domain 对 path 做膨胀/收缩，形成 inner/outer contour，然后在中间区域做 gradient fill。**

```
Original path (hard edge)
     ↓ outer offset by r (feather radius)
  Outer contour
     ↓ inner offset by -r
  Inner contour
     ↓ fill region between inner/outer with opacity gradient
  Soft edge result
```

公式：feathered opacity at distance $d$ from original path：

$$\alpha(d) = \text{clamp}\left(\frac{r - d}{2r} + 0.5, 0, 1\right)$$

其中：
- $r$ = feather radius
- $d$ = signed distance from original path (positive = outside)
- 结果在 inner contour ($d = -r$) 处 $\alpha = 0$，outer contour ($d = +r$) 处 $\alpha = 1$

这在 GPU 上的实现是 **per-fragment** 的，利用 distance field 或 contour mesh 的 attribute interpolation。

> 参考: [Reinventing feathering for the Vectorian era](https://rive.app/blog/how-rive-reinvented-feathering-for-the-vectorian-era)

---

## 5. `.riv` Binary Format

### 5.1 为什么不用 JSON？

| | Lottie (JSON) | Rive (.riv) |
|---|---|---|
| Size | 100–500 KB typical | 10–50 KB typical |
| Parse speed | `JSON.parse()` → slow | Binary deserialize → fast |
| State machine | ❌ No | ✅ Built-in |
| Interactivity | ❌ No | ✅ Input + Event |

### 5.2 `.riv` 的结构

`.riv` 文件基于 Rive 自研的 **binary serialization format**（类似 FlatBuffers 的思路，但更 lightweight）：

```
.riv file layout:
┌────────────────────────┐
│ Header                 │
│  - magic number "RIVE" │
│  - version              │
├────────────────────────┤
│ Property table          │
│  - offsets to objects   │
├────────────────────────┤
│ Object definitions      │
│  - Artboard             │
│  - Shape / Path / Paint │
│  - Bone / Skin          │
│  - StateMachine         │
│  - LinearAnimation      │
│  - Font / TextRun       │
├────────────────────────┤
│ Keyframe data           │
│  - binary interpolated  │
│  - float arrays         │
└────────────────────────┘
```

Deserialize 过程是 **zero-copy** 的：pointer 直接指向 buffer 中的 data，不需要额外 allocation。这极大减少了 memory footprint 和 load time。

---

## 6. Runtime 的核心 API

### 6.1 Loading & Playing

```cpp
#include "rive/file.hpp"
#include "rive/artboard.hpp"
#include "rive/animation/state_machine_instance.hpp"

// 1. Load .riv file
auto file = rive::File::import(rivData, rivDataLength, nullptr);

// 2. Get artboard (scene)
auto artboard = file->artboard("Main");

// 3. Create state machine instance
auto stateMachine = artboard->stateMachine("State Machine");

// 4. Per frame:
stateMachine->advance(elapsedSeconds);
artboard->advance(elapsedSeconds);

// 5. Render
renderer->save();
artboard->draw(renderer);
renderer->restore();
```

### 6.2 Input & Event

```cpp
// Set boolean input
auto hoverInput = stateMachine->getBool("isHover");
hoverInput->value = true;

// Set number input  
auto progressInput = stateMachine->getNumber("progress");
progressInput->value = 0.5;

// Fire trigger input
auto clickTrigger = stateMachine->getTrigger("click");
clickTrigger->fire();

// Listen for events
stateMachine->addEventListener([](const rive::Event* event) {
    printf("Event fired: %s\n", event->name().c_str());
});
```

---

## 7. 平台 Bindings 如何使用 rive-runtime

rive-runtime 是 **pure C++ core**，各平台 binding 通过 submodule 引入：

| Platform Repo | Binding 方式 |
|---|---|
| [rive-android](https://github.com/rive-app/rive-android) | JNI → C++ |
| [rive-ios](https://github.com/rive-app/rive-ios) | ObjC++ wrapper → C++ |
| [rive-flutter](https://github.com/rive-app/rive-flutter) | FFI (dart:ffi) → C++ |
| [rive-js (Web)](https://github.com/rive-app/rive-wasm) | Emscripten → WASM → C++ |
| [rive-unity](https://github.com/rive-app/rive-unity) | P/Invoke → C++ |
| [rive-react](https://github.com/rive-app/rive-react) | 包装 rive-wasm |

所有 platform runtime 都 **共享同一个 C++ core**，即 rive-runtime 中的代码。这就是为什么它叫 "Low-level C++ Rive runtime and renderer"。

---

## 8. Rendering 性能对比

来自社区 benchmark 和 Hacker News 讨论：

| Renderer | Approach | 复杂 animation (100+ paths) |
|---|---|---|
| Lottie + Canvas2D | CPU rasterization | ~30fps on mobile |
| Lottie + SVG | DOM manipulation | ~15-20fps |
| Rive + Skia backend | GPU via Skia | ~60fps |
| **Rive + Native renderer** | **GPU tessellation, no Skia** | **60-120fps** |

Rive native renderer 之所以快，是因为：

1. **No intermediate raster** — 直接 GPU triangle mesh
2. **Animation-first design** — 只 re-tessellate dirty paths（spatial caching）
3. **Minimal draw calls** — batch by blend mode
4. **Zero-copy .riv deserialize** — binary direct map

---

## 9. 与 Lottie 的根本区别

| 维度 | Lottie | Rive Runtime |
|---|---|---|
| **Format** | JSON (text) | .riv (binary) |
| **State** | Linear timeline | State Machine (FSM) |
| **Interactivity** | ❌ | ✅ Input/Event/Transition |
| **Rendering** | Platform Canvas/SVG | Custom GPU renderer |
| **File size** | Large | ~5-10x smaller |
| **CPU cost** | JSON parse + DOM | Binary decode + GPU tessellation |
| **Use case** | Linear decorative animation | **Interactive UI components** |

Rive 的本质 insight：**animation 不应该只是 "decorative overlay"，而应该是 "interactive UI component"**。一个 button hover 效果、loading animation with progress、character responding to click — 这些都是 "stateful + interactive" 的，需要 State Machine 而非 linear timeline。

---

## 10. 总结

从 first-principle 看，rive-runtime 解决的问题是：

> **如何在任何平台上，以最小 memory/CPU cost，运行 stateful、interactive、high-quality 2D vector animation？**

它的 answer 是三层 stack：

1. **Binary format** (.riv) → 最小化 storage + parse cost
2. **State Machine engine** → 让 animation 逻辑内嵌在文件中，code 只需 set input / listen event
3. **Custom GPU renderer** → bypass Skia/Canvas，直接 tessellation → GPU triangle mesh，实现 60fps+ on any device

这就是为什么 Spotify、Duolingo、Google、Disney 等 product team 都在用 Rive — 它把 animation 从 "装饰品" 提升为 "first-class interactive component"。

---

### 参考链接

| 资源 | URL |
|---|---|
| rive-runtime GitHub | [https://github.com/rive-app/rive-runtime](https://github.com/rive-app/rive-runtime) |
| Rive Runtimes 官方页 | [https://rive.app/runtimes](https://rive.app/runtimes) |
| Rive Renderer 开源公告 | [https://rive.app/blog/rive-renderer-now-open-source-and-available-on-all-platforms](https://rive.app/blog/rive-renderer-now-open-source-and-available-on-all-platforms) |
| Vector Feathering 技术深潜 | [https://rive.app/blog/how-rive-reinvented-feathering-for-the-vectorian-era](https://rive.app/blog/how-rive-reinvented-feathering-for-the-vectorian-era) |
| State Machine 入门 | [https://rive.app/blog/how-state-machines-work-in-rive](https://rive.app/blog/how-state-machines-work-in-rive) |
| Rive vs Lottie 对比 | [https://rive.app/blog/rive-as-a-lottie-alternative](https://rive.app/blog/rive-as-a-lottie-alternative) |
| Rive Format 详解 | [https://www.learnrive.com/running-animations/rive-runtime-and-format](https://www.learnrive.com/running-animations/rive-runtime-and-format) |
| Hacker News: Rive Renderer 讨论 | [https://news.ycombinator.com/item?id=39768515](https://news.ycombinator.com/item?id=39768515) |
| GPU Path Rendering (NVIDIA StC) | [https://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/opengl/gpupathrender.pdf](https://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/opengl/gpupathrender.pdf) |
| Rive iOS Runtime | [https://github.com/rive-app/rive-ios](https://github.com/rive-app/rive-ios) |
| Rive Android Runtime | [https://github.com/rive-app/rive-android](https://github.com/rive-app/rive-android) |
| Rive Flutter Runtime | [https://github.com/rive-app/rive-flutter](https://github.com/rive-app/rive-flutter) |