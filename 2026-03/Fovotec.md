







现在我已经收集了足够的信息，可以给出详细的回答。让我综合所有搜索结果来解释 Fovotec 是做什么的。

---

## Fovotec 是做什么的？

**Fovotec**（https://www.fovotec.com/）是一家总部位于英国 Cardiff 的软件公司，成立于 2017 年，由 **Robert Pepperell**（艺术家兼科学家，曾在 Slade School of Art 学习）联合创立。公司的核心使命是：**"To emulate the true experience of natural human vision in real-time computer graphics"**（在实时计算机图形中复现真实人类视觉体验）。公司名 "Fovotec" = **F**ield **o**f **V**iew **o**pened + **tec**hnology，即"打开视野的技术"。

---

### 核心产品：Dynamic Projection

Fovotec 的旗舰产品叫 **Dynamic Projection**，是一款面向 **3D 渲染引擎的实时插件**（rendering plug-in），目前已支持 **Unreal Engine** 和 **Unity** 两大主流游戏引擎。

#### 它要解决什么问题？

传统 3D 渲染使用**线性透视投影**（Linear Perspective Projection），这是自文艺复兴以来沿用至今的方法。线性透视的数学模型如下：

$$x_{proj} = \frac{f \cdot X_{cam}}{Z_{cam}}, \quad y_{proj} = \frac{f \cdot Y_{cam}}{Z_{cam}}$$

其中：
- $(X_{cam}, Y_{cam}, Z_{cam})$ 是相机空间中的 3D 点坐标
- $f$ 是焦距（focal length）
- $(x_{proj}, y_{proj})$ 是投影到屏幕上的 2D 坐标

**核心问题**：线性透视投影在 **FOV（Field of View）增大时**，画面边缘会产生严重的 **透视畸变**（perspective distortion），导致：
1. 边缘物体被极度拉伸（stretching）
2. 直线在边缘弯曲
3. 人类观看超宽画面时感觉"不自然"，因为这不是人眼实际感知世界的方式

人眼的工作方式本质上是**非线性**的——中央凹（fovea）区域分辨率极高，而外围视觉（peripheral vision）分辨率低但感知范围广。人类双眼视野合计可达约 **200° 水平方向**，且大脑会自动"矫正"畸变。

#### Dynamic Projection 的原理

Dynamic Projection 的核心思路是：**用非线性投影替代线性投影**，使得渲染结果更接近人类自然视觉体验。其数学本质是：

$$x_{dp} = g\left(\frac{X_{cam}}{Z_{cam}}\right), \quad y_{dp} = g\left(\frac{Y_{cam}}{Z_{cam}}\right)$$

其中 $g(\cdot)$ 是一个**非线性映射函数**（Dynamic Projection 的核心 IP），而非简单的线性除法。这个函数的设计使得：

| 特性 | 线性透视 | Dynamic Projection |
|------|----------|-------------------|
| 宽 FOV 边缘畸变 | 严重拉伸 | 大幅减少 |
| 视觉自然感 | 边缘不自然 | 接近人眼感知 |
| 透视收缩 | 近大远小很剧烈 | 更平缓的深度压缩 |
| 沉浸感 | 有限 | 增强超广角沉浸 |

具体来说，Dynamic Projection 可以实现以下效果：

1. **扩展视口空间**（extend viewport space）：在相同屏幕上显示更多场景内容，而不引入线性透视的畸变
2. **减少/消除边缘畸变**（reduce or eliminate peripheral distortion）
3. **增强沉浸感**（increase levels of immersion）
4. **支持超广角视觉**（ultra-wide visuals）——如同人眼自然看到的那样

#### 技术实现架构

```
┌─────────────────────────────────────────────┐
│           3D Engine (Unreal / Unity)         │
│  ┌─────────────┐     ┌──────────────────┐   │
│  │  Scene Graph │────▶│ Vertex Shader    │   │
│  │  (meshes,    │     │ (standard MVP    │   │
│  │   lights)    │     │  transform)      │   │
│  └─────────────┘     └───────┬──────────┘   │
│                              │               │
│                    ┌─────────▼──────────┐    │
│                    │ Dynamic Projection │    │
│                    │ Plugin (post-      │    │
│                    │ projection warp)   │    │
│                    │                    │    │
│                    │  P' = g(P_proj)    │    │
│                    │  nonlinear mapping │    │
│                    └─────────┬──────────┘    │
│                              │               │
│                    ┌─────────▼──────────┐    │
│                    │ Rasterizer +       │    │
│                    │ Pixel Shader       │    │
│                    └─────────┬──────────┘    │
│                              │               │
│                    ┌─────────▼──────────┐    │
│                    │ Frame Buffer       │    │
│                    │ (output to screen) │    │
│                    └────────────────────┘    │
└─────────────────────────────────────────────┘
```

关键实现路径：Dynamic Projection 作为 **rendering plug-in** 插入渲染管线，在顶点经过标准 MVP（Model-View-Projection）变换后，对投影结果进行**非线性重映射**（nonlinear remapping）。这个重映射步骤将线性透视坐标 $(x_{proj}, y_{proj})$ 转换为 Dynamic Projection 坐标 $(x_{dp}, y_{dp})$。

#### 与 Foveated Rendering 的区别

| 概念 | Foveated Rendering | Dynamic Projection |
|------|-------------------|-------------------|
| 核心思想 | 根据视线方向降低外围分辨率 | 改变投影几何映射方式 |
| 主要目标 | 节省 GPU 算力 | 消除畸变、增强沉浸感 |
| 需要眼动追踪 | 是（或固定中心区域） | 否 |
| 改变的层级 | 像素级（分辨率） | 几何级（投影变换） |
| 技术互补性 | — | 可以与 Dynamic Projection 结合使用 |

---

### 应用场景

1. **游戏开发（Gaming）**：让玩家在超宽 FOV 下获得自然视觉体验，不产生边缘畸变
2. **企业 3D 可视化（Enterprise 3D Visualization）**：用于建筑、工业设计等需要大视野查看的专业场景
3. **VR/AR**：超广角沉浸式体验
4. **离线渲染（Offline Rendering）**：不仅限于实时，也可用于电影预览等

---

### 商业模式 & 状态

- 产品形态：**渲染插件**（plug-in），兼容 Unreal Engine 和 Unity
- 目前处于 **Early Access**（早期访问）阶段，面向游戏开发者开放
- 公司尚未获得外部融资（unfunded），自举运营
- Robert Pepperell 本人既是艺术家也是研究者，长期关注人类视觉感知与计算机图形学的交叉

---

### 关键直觉总结

**从第一性原理看**：人类视觉系统本质上是一个**非线性传感器**——中央凹高分辨率 + 外围低分辨率 + 大脑的自动畸变矫正。传统 3D 渲染使用**线性透视**，这是一个数学上优雅但在大 FOV 下与人类感知不匹配的模型。Fovotec 的 Dynamic Projection 本质上就是**让渲染投影函数 $g(\cdot)$ 更接近人类视觉系统的非线性特性**，从而在大视角下减少畸变、增强沉浸感。这是对自 Brunelleschi 以来 500 多年线性透视范式的一个挑战与修正。

---

### 参考链接

- [Fovotec 官网](https://www.fovotec.com/)
- [Fovotec - Dynamic Projection 发布公告](https://www.fovotec.com/dynamic-projection-launch)
- [Fovotec Gaming 页面](https://www.fovotec.com/gaming)
- [Fovotec Media 页面](https://www.fovotec.com/media)
- [Crunchbase 公司简介](https://www.crunchbase.com/organization/fovotec)
- [GamesBeat: FOVOTEC opens early access](https://gamesbeat.com/fovotec-opens-early-access-to-dynamic-projection-bringing-human-like-vision-to-game-graphics/)
- [AWN: FOVOTEC Opens Dynamic Projection Technology Early Access](https://www.awn.com/news/fovotec-opens-dynamic-projection-technology-early-access)
- [Fovotec Mission & Backstory PDF](https://irp.cdn-website.com/1415f791/files/uploaded/Fovotec_Mission%20%26%20Backstory_White%20paper%20v1.1.pdf)
- [Robert Pepperell LinkedIn](https://uk.linkedin.com/in/robertpepperell)
- [Fovotec Instagram](https://www.instagram.com/fovotec/)
- [Fovotec YouTube](https://www.youtube.com/@dynamicprojection)