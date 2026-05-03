









**Scratchapixel 是一个专注于从第一性原理教授计算机图形学编程的教育网站**，特别是通过手动实现渲染算法（如 Ray Tracing 和 Rasterization）来帮助学习者深入理解图形学底层原理，而不依赖 OpenGL、DirectX 或 Vulkan 等现成 API。

---

### 核心架构与教学内容

网站内容按**数学基础 → 2D 图形 → 3D 渲染 → 高级渲染技术**的路径组织：

1. **数学物理基础**  
   - 强调 **Linear Algebra**（向量、矩阵、坐标变换）、**Analytic Geometry**（点/向量/法线的区别）、**Calculus**（积分在光照模型中的应用）。  
   - 例如，讲解矩阵乘法时，会拆解 $M_{ij}$ 的意义，并解释为何用 $4 \times 4$ 矩阵表示 **affine transformation**（包含平移的线性变换）。

2. **基础渲染管线**  
   - **Rasterization**：从定义 **Triangle** 的顶点坐标开始，讲解 **Edge Function** 判断像素是否在三角形内、**Barycentric Coordinates** 插值法向量和纹理坐标。  
   - **Ray Tracing**：从 **Primary Ray Generation**（根据视场角 FOV、近裁剪面计算射线方向）到 **Ray-Primitive Intersection**（三角形、球体求交公式：$t = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$）。  

3. **高级主题**  
   - **Monte Carlo Integration**：用随机采样近似求解渲染方程 $L_o(p, \omega_o) = L_e(p, \omega_o) + \int_{\Omega} f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) (\omega_i \cdot n) d\omega_i$。  
   - **Global Illumination**：讲解 **Path Tracing** 与 **Photon Mapping** 的差异，包括 **Radiance** 和 **Flux** 的物理定义。  
   - **Real-time Rendering** 部分还会涉及 **BRDF**（双向反射分布函数）、**Importance Sampling** 等。

---

### 教学特色与价值

1. **代码即教程**：每节课提供完整 C++ 示例，从 `main` 函数开始一步步构建渲染器，避免“黑盒调用”。  
2. **第一性原理推导**：例如在讲解 **Pinhole Camera** 模型时，会从针孔成像几何出发，推导像素坐标到摄像机坐标的变换矩阵。  
3. **系统性覆盖**：不仅包含离线渲染（Offline Rendering），还延伸到 **Digital Imaging**（图像文件格式、色彩空间）、**Geometry**（BVH 加速结构）、**Simulation**（流体、粒子系统）等领域。

---

### 与同类资源的区别

- 相比 **Scratchapixel** 更注重理论基础，**LearnOpenGL** 侧重 API 实战，而 **Peter Shirley《Ray Tracing in One Weekend》** 更偏向快速原型。  
- 对比 **University of Utah's CS 5600** 等课程，Scratchapixel 以自包含 Web 教程形式，适合自学且无需前置知识。  
- 它提供的深度数学推导（如推导 **Point-Light vs Area-Light** 的积分形式）在多数教程中难以见到。

---

### 典型学习路径示例

若你学习 **Ray Tracing 渲染阴影**：  
1. 先完成数学模块的 **Vectors & Matrices**（理解点积 $\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}||\mathbf{b}|\cos\theta$ 在光照计算中的用途）。  
2. 进入 **Ray-Primitive Intersection** 实现 **Ray-Sphere Intersection**。  
3. 在 **Light 和阴影** 课中，实现 **Shadow Ray**：从表面点向光源发射射线，若 $t_{min} < t < t_{max}$ 有交点，则该点在阴影中。

---

### 参考链接

- 官方网站与课程主页：  
  https://www.scratchapixel.com/  
  https://www.scratchapixel.com/lessons/3d-basic-rendering/

- Reddit 社区讨论其教学价值：  
  https://www.reddit.com/r/GraphicsProgramming/comments/15mg0po/is_scratchapixelcom_good_should_i_continue_with_it/

- Hacker News 对其书籍的介绍：  
  https://news.ycombinator.com/item?id=11744763

- 数学基础课节选：  
  https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/points-vectors-and-normals.html

---

> Scratchapixel 的独特之处在于它不教你“如何调库”，而是让你从零推导每个数学公式和算法，最终拥有能力自己编写任意渲染器。这种训练方式能建立对图形学底层机制的直觉，是成为高级图形工程师的必经之路。