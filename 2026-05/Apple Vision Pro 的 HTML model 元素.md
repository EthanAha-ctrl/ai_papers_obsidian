# 深度解析：Apple Vision Pro 的 HTML `<model>` 元素 — Spatial Web 的关键一步

## 📌 文章概要

这篇文章由 **Brandel Zachernuk** 撰写（2025年6月26日），宣布了 **visionOS 26** 中一个里程碑式的更新：**HTML `<model>` 元素默认启用**，并带来了完整的 JavaScript API。这标志着 Web 从「平面文档」迈向「空间计算」的实质性一步。

---

## 🧠 第一性原理：为什么需要 `<model>` 元素？

从第一性原理出发，我们要问：**Web 展示3D内容的根本需求是什么？**

```
传统 Web:  HTML → 浏览器渲染 → 2D 平面输出
空间 Web:  HTML → 浏览器渲染 → 3D 空间输出（立体渲染）
```

现有的 3D Web 方案存在根本性问题：

| 方案 | 问题 |
|------|------|
| **WebXR** | 需要 JavaScript 深度介入，门槛高，无法与普通网页内容并排展示 |
| **AR Quick Look** | 脱离浏览器上下文，全屏沉浸，无法嵌入网页 |
| **JS 库（Three.js 等）** | 依赖 JavaScript 运行时，无原生 accessibility/隐私保护 |
| **`<model>` 元素** ✅ | **声明式 HTML**，浏览器原生处理渲染、accessibility、隐私，可与普通内容并排 |

核心洞察：正如 `<video>` 让视频成为 Web 的一等公民，`<model>` 让 3D 内容也成为 Web 的一等公民——**声明式、可访问、可组合**。

---

## 🏗️ API 架构详解：Lights, Camera, Action

文章用 **"Lights, Camera, Action"** 这个电影工业隐喻来组织 API 的三大核心模块：

### 1️⃣ Action — 基础交互

#### `ready` Promise
```html
<model id="teapot">
  <source src="teapot.usdz" type="model/vnd.usdz+zip">
</model>
```
```javascript
const teapot = document.querySelector('#teapot');
await teapot.ready;
```

- **为什么是 Promise 而非事件？** Promise 代表一次性异步操作完成，语义比 `addEventListener('load')` 更精确。3D 资产需要：下载 → 解压 → 解析 → 构建 scene graph → 上传 GPU → 就绪。这是一个多阶段管线（pipeline）：

$$T_{ready} = T_{download} + T_{decompress} + T_{parse} + T_{GPU\_upload}$$

- **`<source>` 元素**：与 `<video>`/`<audio>` 一致的模式，支持多源回退（fallback），`type` 属性使用 MIME type `model/vnd.usdz+zip`。

#### Orbit Mode（轨道模式）

```html
<model stagemode="orbit">
    <source src="teapot.usdz" type="model/vnd.usdz+zip">
    <img alt="a teapot for interacting with" src="fallback/teapot-orbit.jpg">
</model>
```

- `stagemode="orbit"` 的交互映射：
  - **水平 pinch-and-drag** → 绕 Y 轴旋转（heading/yaw）
  - **垂直 pinch-and-drag** → 绕 X 轴旋转（pitch）

这本质上是 **球坐标相机控制**（spherical camera control）：

$$\theta_{yaw} = \theta_0 + k_h \cdot \Delta x$$
$$\phi_{pitch} = \phi_0 + k_v \cdot \Delta y$$

其中 $k_h, k_v$ 为灵敏度系数，$\Delta x, \Delta y$ 为手势位移。

- `<img>` 作为 **fallback**：在不支持 `<model>` 的浏览器中优雅降级，这是 Web 平台 **渐进增强**（progressive enhancement）理念的体现。

### 2️⃣ Camera — EntityTransform

这是整个 API 中**最精妙的设计**，也是理解空间 Web 与传统 3D 渲染差异的关键。

#### 核心问题：为什么不用传统 Camera？

在传统 3D 引擎（Three.js, Unity）中，场景通过**虚拟相机**渲染：

```
虚拟相机 → 投影矩阵 → 2D 视口
```

但在 Apple Vision Pro 中，**用户的双眼就是相机**！浏览器无法控制用户眼睛的焦距（FoV）、瞳距（IPD）等——这些由 visionOS 的渲染管线自动处理。

因此，API 不提供 "camera" 控制，而是提供 **entityTransform**——即**变换物体本身**而非相机：

```javascript
teapot.entityTransform = new DOMMatrix().translate(0, 0, -0.5).rotate(0, 90, 0);
```

#### DOMMatrix 深度解析

`DOMMatrix` 是 CSS Transforms 规范中的类型，表示一个 **4×4 齐次变换矩阵**（homogeneous transformation matrix）：

$$M = \begin{bmatrix} m_{11} & m_{12} & m_{13} & m_{14} \\ m_{21} & m_{22} & m_{23} & m_{24} \\ m_{31} & m_{32} & m_{33} & m_{34} \\ m_{41} & m_{42} & m_{43} & m_{44} \end{bmatrix} = \begin{bmatrix} R_{3\times3} & T_{3\times1} \\ 0 & 1 \end{bmatrix}$$

其中：
- $R_{3\times3}$：旋转+缩放的 3×3 子矩阵
- $T_{3\times1} = [m_{41}, m_{42}, m_{43}]^T$：平移向量（注意 CSS 中平移在最后一行）
- $m_{14}, m_{24}, m_{34}$：透视变换参数（通常为 0）
- $m_{44}$：齐次坐标缩放因子（通常为 1）

**变换顺序的关键性**：矩阵乘法不可交换！

$$M = T \cdot R \neq R \cdot T$$

`new DOMMatrix().translate(0, 0, -0.5).rotate(0, 90, 0)` 的计算过程：

1. 初始 $M_0 = I_{4\times4}$
2. `.translate(0, 0, -0.5)` → $M_1 = M_0 \cdot T(0, 0, -0.5)$
3. `.rotate(0, 90, 0)` → $M_2 = M_1 \cdot R_Y(90°)$

最终效果：**先平移再旋转** → 物体沿 Z 轴后退 0.5 米，然后绕 Y 轴旋转 90°。

> ⚠️ 注意：这里的坐标单位是**米**（visionOS 的标准单位），而非 CSS 的像素。Z 轴负方向为「远离用户」。

#### boundingBoxCenter & boundingBoxExtents

```javascript
// 加载后可访问
model.boundingBoxCenter  // DOMPointReadOnly {x, y, z, w}
model.boundingBoxExtents  // DOMPointReadOnly {x, y, z, w}
```

- **boundingBoxCenter**：模型轴对齐包围盒（AABB）的质心，记为 $\mathbf{c} = (c_x, c_y, c_z)$
- **boundingBoxExtents**：从质心到各轴边界的半宽度，记为 $\mathbf{e} = (e_x, e_y, e_z)$

完整包围盒为：$[\mathbf{c} - \mathbf{e}, \quad \mathbf{c} + \mathbf{e}]$

这两个值的实际用途：
1. **自动居中**：浏览器默认设置 entityTransform 使模型居中显示
2. **手动校准**：如果你知道模型脚在原点，可以用 `boundingBoxCenter` 计算偏移量使模型「站在地面」
3. **碰撞检测前端**：可用于粗略的空间布局计算

### 3️⃣ Lights — Environment Map

#### 为什么 3D 模型需要 Environment Map？

**基于图像的照明**（Image-Based Lighting, IBL）的物理基础：

真实世界中，物体表面的入射光来自**所有方向**：

$$L_{in}(\omega_i) = \int_{\Omega} L_{env}(\omega_i) \cdot f_r(\omega_i, \omega_o) \cdot \cos\theta_i \, d\omega_i$$

其中：
- $L_{env}(\omega_i)$：环境光在方向 $\omega_i$ 的辐射亮度
- $f_r(\omega_i, \omega_o)$：BRDF（双向反射分布函数）
- $\cos\theta_i$：Lambert 余弦定律

**Environment Map 就是对 $L_{env}$ 的离散化表示**——它是一个球面函数的采样。

#### HDR 的必要性

人眼可感知的亮度范围约 $10^{-3}$ ~ $10^{6}$ $cd/m^2$（12 个数量级）。而标准 8-bit sRGB 仅能表示 0-255（约 2.4 个数量级）。

日光下：
- 天空亮度：~$10^{4}$ $cd/m^2$
- 直射太阳：~$10^{6}$ $cd/m^2$
- 阴影区域：~$10^{2}$ $cd/m^2$

HDR 格式（EXR / Radiance HDR）使用浮点存储，保留完整动态范围。

文章支持的格式：
- **OpenEXR**（`.exr`）：工业标准，半精度/单精度浮点，支持多通道
- **Radiance HDR**（`.hdr`）：每像素 32-bit（RGBE 编码）

#### 等距柱状投影

```
球面坐标 (θ, φ) → 像素坐标 (u, v)

u = φ / (2π)
v = θ / π
```

其中 $\theta \in [0, \pi]$ 为极角（zenith angle），$\phi \in [0, 2\pi)$ 为方位角（azimuth）。

文章特别指出投影带来的**极地畸变**问题——在极点附近，单位球面面积对应更少的像素，导致高频细节丢失。这也是为什么 UV unwarp 后极点附近纹理拉伸。

```html
<model environmentmap="studio.exr">
```

```javascript
await model.environmentMapReady;  // 确保 HDR 已下载并上传 GPU
```

`environmentMapReady` 是独立的 Promise，因为 HDR 文件通常很大（数 MB 到数十 MB），需要异步加载管线：

$$T_{env} = T_{download} + T_{decode} + T_{mipmap} + T_{GPU\_upload}$$

其中 mipmap 生成是 IBL 的关键预处理——用于 specular 反射的 **预过滤**（prefiltering）：

$$L_{specular}(\omega_o, r) \approx \text{sampleCubemap}(M_{r-roughness}, \omega_r)$$

---

## 🎬 Animation API

动画控制完全对齐 `<video>`/`<audio>` 的播放 API：

```javascript
model.play();           // 播放动画
model.pause();          // 暂停动画
model.currentTime = 2.5;  // 跳转到 2.5 秒
model.playbackRate = 1.5; // 1.5 倍速
model.duration;         // 动画总时长（秒）
```

声明式属性：
```html
<model autoplay loop>
    <source src="watch.usdz" type="model/vnd.usdz+zip">
</model>
```

- `autoplay`：页面加载后自动播放
- `loop`：循环播放

**为什么复用 `<video>` API 而非新建？**

这是 **API 最小惊讶原则**（Principle of Least Astonishment）：开发者已熟悉 `play()/pause()/currentTime/duration`，复用降低学习成本。时间线模型统一为：

$$\text{frame}(t) = \text{skeletal\_animation}(\text{USDZ\_timeline}, t \cdot \text{playbackRate})$$

---

## 📦 USDZ 格式深度解析

### USD 的前世今生

- **USD**（Universal Scene Description）：Pixar 开发的 3D 场景描述框架
- 2016 年开源 → **OpenUSD**
- **USDZ**：USD 的 **零压缩打包格式**（Z = Zero-compression），即所有数据在单一文件中，无外部引用

### USDZ 的技术特点

| 特性 | 说明 |
|------|------|
| **文件结构** | 单一 ZIP-like 宁器，无需解压 |
| **几何表示** | 支持网格、细分曲面、NURBS |
| **材质** | UsdPreviewSurface（PBR 金属粗糙度工作流） |
| **动画** | 基于 timeSamples 的骨骼动画 |
| **空间单位** | 厘米（1 unit = 1 cm） |
| **坐标系** | 右手坐标系，Y-up |

USDZ 的 **UsdPreviewSurface** 材质模型核心参数：

$$C_{final} = (1 - metallic) \cdot C_{diffuse} + metallic \cdot C_{specular}$$

$$F_{Schlick}(\cos\theta, f_0) = f_0 + (1 - f_0)(1 - \cos\theta)^5$$

其中：
- $C_{diffuse}$：漫反射颜色
- $metallic$：金属度 [0, 1]
- $f_0$：法线入射方向的菲涅尔反射率
- $\cos\theta$：视线与法线的夹角余弦

---

## 🌐 标准化进程与生态

文章指出 `<model>` 元素正在 **W3C** 和 **WHATWG** 审议中：

| 组织 | 角色 |
|------|------|
| **W3C** | Model Element API 规范提案 |
| **WHATWG** | HTML 规范管理 |
| **AOUSD**（Alliance for OpenUSD） | USD 格式标准化 |

这与当年 `<video>` 元素的标准推进路径相似：
1. 浏览器厂商实现（Apple率先）→ 2. 社区反馈 → 3. W3C 标准化 → 4. 其他浏览器跟进

---

## 🔮 延伸思考与直觉构建

### 为什么这比 WebXR 更重要？

```
WebXR 范式:  全屏沉浸 → 用户离开网页 → 独立体验
<model> 范式: 嵌入式 3D → 用户留在网页 → 混合体验
```

`<model>` 的哲学是 **2D 与 3D 的共处**（co-existence），而非替代。这更符合空间计算设备的真实使用模式：你不会一直在全沉浸模式，你会在空间中排布多个窗口和物体。

### 与现有生态的对比

| 特性 | `<model>` | `<canvas>` + Three.js | `<iframe>` + Unity |
|------|-----------|----------------------|-------------------|
| 声明式 | ✅ | ❌ | ❌ |
| 原生 accessibility | ✅ | ❌ | ❌ |
| 隐私沙箱 | ✅ | ⚠️ 部分 | ❌ |
| 与页面 DOM 交互 | ✅ | ⚠️ 有限 | ❌ |
| 自定义着色器 | ❌ | ✅ | ✅ |
| 物理引擎 | ❌ | ✅ | ✅ |
| GPU 粒子系统 | ❌ | ✅ | ✅ |

### 当前局限

1. **仅支持 USDZ**：不支持 glTF/GLB（Web 3D 的事实标准格式），生态兼容性受限
2. **无自定义着色器**：无法实现自定义材质效果
3. **无交互事件**：缺少 `onmodelclick`、射线拾取等 3D 交互 API
4. **仅限 visionOS**：其他平台（Meta Quest、Pico 等）尚未实现
5. **无物理/碰撞**：纯渲染，无物理模拟

### 可能的演进方向

```
Phase 1 (Now):  <model src="xxx.usdz"> → 静态/动画展示
Phase 2:        <model> + JS Events → 交互式 3D widget
Phase 3:        <scene> + <model> → 多模型空间布局
Phase 4:        <model> + glTF/glb → 跨平台标准
Phase 5:        Spatial DOM → 完整的空间 Web 标准体系
```

---

## 📚 参考资料

- [WebKit Blog - HTML Model Element](https://webkit.org/blog/16662/a-step-into-the-spatial-web-the-html-model-element-in-apple-vision-pro/)（本文原始出处）
- [W3C Model Element Proposal](https://github.com/WebKit/standards-positions/issues/32)
- [WHATWG HTML Specification](https://html.spec.whatwg.org/)
- [OpenUSD Documentation](https://openusd.org/release/index.html)
- [Alliance for OpenUSD (AOUSD)](https://aousd.org/)
- [Polyhaven - 免费 HDR Environment Maps](https://polyhaven.com/hdris)
- [Apple Reality Composer Pro](https://developer.apple.com/documentation/reality-composer/)
- [CSS DOMMatrix Specification](https://drafts.fwg.org/css-transforms-1/#dommatrix-interface)
- [USDZ Specification - Apple](https://developer.apple.com/augmented-reality/quick-look/)
- [Pixar USD Technical Documentation](https://graphics.pixar.com/usd/release/index.html)

---

**总结**：`<model>` 元素是 Web 从 2D 文档模型向 3D 空间模型演进的关键基础设施。它用**声明式 HTML** 的方式解决了 3D 内容的**可访问性、隐私保护、与现有 Web 内容的融合**三大问题。虽然当前实现（仅 USDZ、仅 visionOS、有限交互）还处于早期阶段，但其设计哲学——**浏览器承担渲染复杂度，开发者享受简单声明**——正是 Web 平台成功的核心密码。