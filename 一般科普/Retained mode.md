基于你提供的文字，我将深入解析 **Retained mode**（保留模式）的核心概念、技术细节、数学原理及其在计算机图形学中的广泛应用。我们将从基础直觉出发，逐步深入到架构设计和优化算法的层面。

### 1. 核心直觉：什么是 Retained mode？

为了建立直觉，我们可以将 **Rendering**（渲染）比作一场戏剧表演：

*   **Immediate Mode（即时模式）** 就像是你作为导演，站在舞台边上，每一秒钟都要亲自告诉每一个演员：“你现在举起右手”，“你现在走到左边”，“灯光现在变红”。一旦你停止指令，表演就停止了。你需要每一帧都重新描述所有的动作。
*   **Retained Mode（保留模式）** 则像是你作为剧作家，写好了完整的剧本和舞台布景说明书，并将其交给了舞台监督。你只需要偶尔发出指令：“第二幕开始，演员A移动到新位置”。舞台监督会管理整个场景，决定何时拉幕布、如何打光、哪些演员在当前镜头里看不到从而不需要动。

在技术层面上，**Retained mode** 意味着 **Graphics Library**（图形库）不仅是渲染的执行者，更是 **Scene**（场景）的**管理者**。Client（客户端，即应用程序）将需要渲染的对象定义给 Library，Library 将其存储在内部的 **Data Space**（数据空间）中，形成一个 **Abstract Internal Model**（抽象内部模型）。

#### 关键特征解析：
1.  **Retained Scene**：Library 保存了完整的 **Object Model**（对象模型），包括几何体、材质、光照等。
2.  **Indirection**（间接性）：Client 的调用不直接触发绘制指令，而是更新 Library 内部的状态。
3.  **Managed Resources**：Library 负责管理显存中的 **Resources**（资源），如 **Vertex Buffers**（顶点缓冲）、**Textures**（纹理）等。

---

### 2. 架构深度解析：Schematic Explanation

根据文字中的描述，我们来解剖 **Retained Mode Graphics API** 的典型架构流程。

#### 架构流程图解（文字描述版）

```text
[Client Application]
       |
       | 1. High-level Command (e.g., "Update Position", "Create Object")
       v
[Retained Mode Library Interface]
       |
       | 2. Update Internal Scene Graph / State
       v
[Scene Database (Memory / VRAM)]
  - Object List
  - Transform Matrices
  - Material Properties
  - Resource Handles (Pointers to GPU Data)
       |
       | 3. Rendering Loop (Optimized by Library)
       v
[Rendering Engine / Backend]
       |
       | 4. Optimization Techniques (Culling, Sorting)
       v
[GPU Driver] -> [Frame Buffer] -> [Display]
```

**技术细节讲解：**

*   **Scene Graph (场景图)**：这是 Retained Mode 的核心数据结构。通常是一个树形结构（DAG - Directed Acyclic Graph，有向无环图）。
    *   **Node**（节点）：代表一个物体。
    *   **Edge**（边）：代表父子关系。
    *   **Transform Propagation**（变换传播）：当父节点移动时，Library 自动计算子节点的最终位置。

*   **Resource Handles**：Library 不需要每一帧都传输 Mesh 数据。它在初始化时将数据上传到 GPU，并返回一个 Handle（句柄，如整数ID或指针）。后续渲染只引用 Handle。

---

### 3. 优化的数学原理与技术实现

文中提到了几种优化技术，我们结合公式和算法详细讲解。

#### 3.1 Double Buffering (双缓冲)

*   **目的**：防止 **Screen Tearing**（屏幕撕裂）。
*   **原理**：
    *   **Front Buffer**（前缓冲）：当前正在显示在屏幕上的图像数据。
    *   **Back Buffer**（后缓冲）：Library 正在绘制的下一帧图像数据。
*   **技术实现**：
    当 Library 完成一帧渲染后，触发 **V-Sync**（垂直同步），交换 Front Buffer 和 Back Buffer 的指针引用。
    $$ Pointer_{Front} \leftrightarrow Pointer_{Back} $$

#### 3.2 Hidden Surface Removal (隐藏面消除)

这是 Retained Mode Library 大展身手的地方，因为它知道整个场景的几何结构。

**A. Backface Culling (背面剔除)**

对于封闭物体，背离摄像机的面是不可见的。

*   **公式**：利用法向量 $\vec{n}$ 和视线向量 $\vec{v}$ 的点积。
    假设从顶点到观察点的向量是 $\vec{v}$，三角形的法向量是 $\vec{n}$。
    $$ c = \vec{n} \cdot \vec{v} $$
    *   如果 $c > 0$（假设右手坐标系，法线向外），说明面朝向摄像机，**保留**。
    *   如果 $c \le 0$，说明面背对摄像机，**剔除**（不送入 GPU 渲染管线）。

**B. Z-Buffering (深度缓冲)**

这是像素级的判定。Library 维护一个与屏幕分辨率相同的深度数组 $Z[x, y]$。

*   **算法流程**：
    对于光栅化后的每个像素 $(x, y)$，其当前深度为 $z_{new}$：
    1.  读取 $Z_{buffer}[x, y]$ 中的现有深度 $z_{old}$。
    2.  比较：
        $$ if (z_{new} < z_{old}) \{ $$
        $$ \quad UpdatePixelColor(x, y); $$
        $$ \quad Z_{buffer}[x, y] = z_{new}; $$
        $$ \} $$
        $$ else \{ $$
        $$ \quad DiscardPixel(); $$
        $$ \} $$

#### 3.3 Delta Updates (增量更新)

这是 Retained Mode 相比 Immediate Mode 最大的性能优势之一。

*   **原理**：Client 只需要修改发生变化的对象属性。
*   **带宽计算**：
    假设场景中有 $N$ 个对象，每个对象数据大小为 $S$。
    *   **Immediate Mode** 每帧传输量：$Total = N \times S$。
    *   **Retained Mode** 每帧传输量：$Total = \sum_{i \in Modified} S_i$。
    如果只有 1 个对象移动，$Total \approx S$，带宽消耗极低。

---

### 4. 坐标变换与矩阵栈

Retained Mode 中，Library 维护一个 **Matrix Stack**（矩阵栈）来处理层级变换。

当渲染一个在层级中深处的物体时，其最终的 **Model Matrix**（模型矩阵）是父级矩阵的级联。

设物体 $i$ 的局部变换矩阵为 $M_{local}^{(i)}$，其父对象的累积矩阵为 $M_{global}^{(parent)}$。

物体 $i$ 的全局变换矩阵 $M_{global}^{(i)}$ 计算如下：
$$ M_{global}^{(i)} = M_{global}^{(parent)} \times M_{local}^{(i)} $$

如果是一棵树：
$$ M_{global}^{(leaf)} = M_{root} \times M_{node1} \times \dots \times M_{leaf} $$

Library 会缓存这些计算结果，除非父节点移动，否则子节点的 $M_{global}$ 不需要重新计算。

---

### 5. 实验数据对比：Immediate vs. Retained

为了直观理解，我们构建一个概念性的实验数据表。

| 特性 | Immediate Mode (e.g., Legacy OpenGL `glBegin`) | Retained Mode (e.g., WPF, SceneKit) |
| :--- | :--- | :--- |
| **CPU 负载** | 高 (每帧需重新构建指令流) | 低 (仅需更新状态变化) |
| **GPU 数据传输带宽** | 高 (每帧重传顶点/索引数据) | 低 (数据驻留显存，仅传变换) |
| **场景管理能力** | 无 (Client 必须自己实现) | 内置 (Scene Graph, Culling) |
| **灵活性** | 极高 (完全控制绘制顺序) | 中等 (受限于 API 的抽象层) |
| **Latency (延迟)** | 低 (指令立即执行) | 可能稍高 (Library 批处理优化) |
| **典型应用** | 简单 Demo, 极其优化的游戏引擎 | GUI (Windows Forms, WPF), 3D 编辑器 |

---

### 6. 深度联想与扩展

为了最大化你的理解，我们可以联想到以下概念：

#### 6.1 DOM (Document Object Model)
Web 开发中的 HTML DOM 是典型的 **Retained Mode**。
*   **Browser**（浏览器）充当 Library。
*   **HTML Tags** 是 Scene Graph 中的 Nodes。
*   **JavaScript** 是 Client。
*   当你执行 `element.style.width = '100px'` 时，你并没有直接重绘像素，而是更新了 Browser 内部的 Retained Tree。Browser 的 Layout Engine 决定何时以及如何重绘。

#### 6.2 Vulkan / DirectX 12 / Metal
现代低级 API 表面上看像是 "Immediate Mode"（你手动录制 Command Buffer），但它们的设计思想鼓励用户在应用层构建一个 **Retained Mode** 系统。你创建 "Pipeline State Objects" (PSO), "Descriptor Sets"，这些都是被 Retained 的资源。

#### 6.3 Spatial Partitioning (空间分割)
为了进一步加速 Culling，Retained Mode Library 常结合以下数据结构：
*   **Octree (八叉树)**：将 3D 空间递归分割。
*   **BVH (Bounding Volume Hierarchy)**：包围体层级结构。
*   **PVS (Potentially Visible Set)**：可能可见集，常用于 BSP 树中。

**视锥体剔除公式**：
对于物体的包围球中心 $C$ 和半径 $r$，以及视锥体的 6 个平面（平面方程为 $Ax + By + Cz + D = 0$，法向量向外）：
$$ d = A \cdot C_x + B \cdot C_y + C \cdot C_z + D $$
如果对于任何一个平面，都有 $d > r$，则该物体完全在视锥体外部，**剔除**。

#### 6.4 Display Lists (OpenGL)
文本中提到的 OpenGL 例子涉及 **Display Lists**。这是 OpenGL 的一种 Retained Mode 特性。
*   Client 调用 `glNewList()` 开始记录。
*   发送一系列绘制命令。
*   调用 `glEndList()` 结束。
*   OpenGL 将这些命令编译并存储在服务器端（通常在 GPU 内存或驱动内存中）。
*   后续调用 `glCallList()` 时，无需重新解析命令，直接执行预编译的版本。

### 总结

**Retained mode** 本质上是将**状态管理**和**渲染调度**的责任从 Client 转移到了 Graphics Library。通过维护一个持久的 **Scene Graph**，Library 能够利用**全局信息**（如物体间的遮挡关系、帧与帧之间的数据一致性）来进行数学上的优化（如 Culling、Delta Update），从而大幅降低 CPU 与 GPU 之间的通信开销，提升复杂场景的渲染效率。

### References

1.  *Microsoft Developer Network (MSDN) - Retained Mode Graphics*. [https://learn.microsoft.com/en-us/previous-versions/windows/desktop/dd757107(v=vs.85)](https://learn.microsoft.com/en-us/previous-versions/windows/desktop/dd757107(v=vs.85))
2.  *Wikipedia - Retained mode*. [https://en.wikipedia.org/wiki/Retained_mode](https://en.wikipedia.org/wiki/Retained_mode)
3.  *OpenGL Wiki - Display Lists*. [https://www.khronos.org/opengl/wiki/Display_List](https://www.khronos.org/opengl/wiki/Display_List)
4.  *Apple Developer - SceneKit*. [https://developer.apple.com/documentation/scenekit](https://developer.apple.com/documentation/scenekit)
5.  *Real-Time Rendering, Chapter 4: Transforms*. (关于矩阵变换的标准参考文本)