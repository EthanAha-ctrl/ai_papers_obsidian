### 1. 核心架构：Stack-based Interpreter (基于栈的解释器)

PostScript 的语言设计深受 Forth 语言的影响。它采用 Reverse Polish Notation (RPN，逆波兰表示法)。它不需要复杂的语法解析器，执行效率极高。
#### Stack Operations (栈操作)
PostScript 虚拟机维护一个 Operand Stack (操作数栈)。所有的运算符都直接从栈中取值，计算后放回栈中。

*   **基本逻辑：**
    如果我们要计算 $ (1 + 2) \times 3 $，在 infix notation (中缀表达式) 中是 `1 + 2 * 3`。
    在 PostScript 中，代码是：
    ```text
    1 2 add 3 mul
    ```
    **执行流程解析：**
    1.  `1` pushed onto Stack $\rightarrow [1]$
    2.  `2` pushed onto Stack $\rightarrow [1, 2]$
    3.  `add` operator pops top two values ($1, 2$), sums them, pushes result $\rightarrow [3]$
    4.  `3` pushed onto Stack $\rightarrow [3, 3]$
    5.  `mul` operator pops top two values ($3, 3$), multiplies them, pushes result $\rightarrow [9]$

*   **公式化描述栈变换：**
    设 Stack 的状态为 $S$。
    对于二元操作符 $\oplus$ (如 `add`, `sub`, `mul`, `div`)：
    $$ S_{new} = S_{old} \setminus \{v_1, v_2\} \cup \{v_1 \oplus v_2\} $$
    其中，$v_1$ 是 Stack Top，$v_2$ 是 Stack Top-1。

#### Graphics State Stack (图形状态栈)
除了数据栈，PostScript 还维护一个 Graphics State Stack。`gsave` 和 `grestore` 指令允许你保存当前的绘图环境（坐标系、颜色、线宽等），进行修改和绘制，然后“弹栈”恢复原状。这对于绘制复杂的嵌套图形（如旋转的文字框内的图形）至关重要。

### 2. 坐标系统与变换矩阵

PostScript 使用笛卡尔坐标系。最核心的技术概念是 **Current Transformation Matrix (CTM)**。所有用户空间的坐标在转化为设备空间（打印机的 dpi 点阵）之前，都必须乘以这个矩阵。

#### 数学公式与矩阵解析
二维坐标变换通常使用 $3 \times 3$ 的齐次坐标矩阵来表示仿射变换（平移、缩放、旋转、剪切）。

CTM 矩阵 $M$ 定义如下：
$$
M = \begin{bmatrix}
a & b & 0 \\
c & d & 0 \\
e & f & 1
\end{bmatrix}
$$

**变量含义解析：**
*   $a, d$: 控制 Scaling (缩放)。$a$ 是 x 轴缩放因子，$d$ 是 y 轴缩放因子。
*   $b, c$: 控制 Shearing (剪切) 和 Rotation (旋转)。
*   $e, f$: 控制 Translation (平移)。$e$ 是 x 轴位移量，$f$ 是 y 轴位移量。

**坐标变换公式：**
假设用户空间的一个点坐标为 $(x, y)$，转换后的设备空间坐标 $(x', y')$ 计算如下：
$$
\begin{bmatrix} x' & y' & 1 \end{bmatrix} = \begin{bmatrix} x & y & 1 \end{bmatrix} \times \begin{bmatrix} a & b & 0 \\ c & d & 0 \\ e & f & 1 \end{bmatrix}
$$
展开后得到：
$$
x' = a \cdot x + c \cdot y + e
$$
$$
y' = b \cdot x + d \cdot y + f
$$

**建立 Intuition：** 当你在 PostScript 中写 `rotate 45` 时，解释器实际上是在修改 CTM 中的 $a, b, c, d$ 值，使得后续所有传入的坐标 $(x, y)$ 都自动被旋转了 45 度。这就是为什么 PostScript 画旋转文字极其高效的原因。

### 3. 曲线渲染：Bézier Curves (贝塞尔曲线)

PostScript 不使用直线段逼近曲线（除非必要），而是直接使用三次 Bézier 曲线作为图元。这使得它无论放大多少倍边缘总是平滑的。

#### 三次 Bézier 曲线公式
给定四个点：起点 $P_0$，控制点 $P_1$，控制点 $P_2$，终点 $P_3$。曲线上的点 $B(t)$ 由参数 $t \in [0, 1]$ 决定：

$$
B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3
$$

**技术细节解析：**
*   $P_0, P_3$: 曲线必须经过的锚点。
*   $P_1, P_2$: 控制点，曲线的切线方向由 $P_0 \rightarrow P_1$ 和 $P_2 \rightarrow P_3$ 决定。这叫做“橡皮筋效应”。
*   PostScript 指令 `curveto` (即 `x1 y1 x2 y2 x3 y3 curveto`) 直接对应这后三个点的输入 ($P_1, P_2, P_3$)，当前路径的当前位置即为 $P_0$。

### 4. Font Technology (字体技术)：Type 1 与 Type 3

PostScript 最初革命性地将字体视为图形指令。

*   **Type 1 Fonts:** 使用了 **Cubic Bézier Curves** 和 **Hinting** (网格拟合) 指令。Hinting 是为了解决低分辨率打印机（如 300 dpi）下线条对齐网格的问题，防止字体模糊。
*   **Type 3 Fonts:** 允许在字体轮廓中包含任意 PostScript 代码（甚至位图），非常灵活，但不可缩放且处理速度慢。
*   **CharString:** 这是一个特殊的加密的字节码，用于描述字符轮廓，它实际上是一个精简版的 PostScript 子集。

**技术对比表：**

| Feature | PostScript Type 1 | TrueType (竞品) |
| :--- | :--- | :--- |
| **Curve Math** | Cubic Bézier (3次) | Quadratic Bézier (2次) |
| **Control Points** | 2 control points (off-curve) | 1 control point (off-curve) per segment |
| **Hinting** | Complex, flexible instructions | Simpler instructions |
| **Encoding** | Binary (encrypted/proprietary originally) | Table-based |

### 5. 打印流程：The Raster Image Processor (RIP)

PostScript 文件发送到打印机后，必须经过 RIP 处理。这是从“Vector”到“Raster”的转换过程。

1.  **Interpretation:** 解释器读取 PS 代码，建立 Display List (显示列表)。
2.  **Scan Conversion:** 将矢量图形根据分辨率（例如 600 dpi 或 2400 dpi）转化为 Bitmap。
    *   公式概念：计算每个 Pixel 的中心点 $(x_c, y_c)$ 是否在几何图形的 Fill Rule (如 Non-Zero Winding Rule 或 Even-Odd Rule) 定义区域内。
3.  **Halftoning (半调网屏)：** 对于彩色打印，将连续的色调（0-255）转换为微小的 Halftone Cell (如 AM Screening 或 FM Screening)。
    *   **Dot Gain Compensation:** RIP 会计算墨粉在纸张扩散的数学模型，以提前减少输出墨量。

### 6. 封装与版本演进

*   **Encapsulated PostScript (EPS):** 这是一种单页的 PostScript 格式，包含一个低分辨率的 Preview (通常是 PICT 或 TIFF)，用于在 Layout Software (如 PageMaker, QuarkXPress) 中显示缩略图。
    *   **Magic Number:** EPS 文件通常以 `%!PS-Adobe-3.0 EPSF-3.0` 开头。
*   **Level 1, 2, 3:**
    *   **Level 1:** 黑白为主，基本的矢量。
    *   **Level 2:** 引入了 Composite Color (合成色彩)，Binary Data Compression (如 Run-Length Encoding, LZW, JPEG)，极大减小了文件体积。
    *   **Level 3:** 引入了渐进式渲染，更好的 Masking 和 Shading（Mesh Gradients）。

### 8. 扩展联想
*   **Ghostscript:** 这是一个非常重要的 Open Source Interpreter，它可以将 PostScript 转换为 PDF，或者转换为 Raster formats (如 PNG, TIFF)。许多 Linux printing system (CUPS) 底层都依赖它。
*   **Display PostScript:** NeXT Computer (Steve Jobs 的公司) 曾经尝试将 PostScript 作为 Windowing System (NeWS) 的屏幕渲染语言，这意味着屏幕上的窗口和打印机上的纸张使用完全相同的语言描述。
*   **Interpress:** Xerox PARC 开发的 PostScript 的前身，John Warnock (Adobe 创始人) 参与了开发但认为它太复杂，于是离开后写了更简洁的 PostScript。
*   **PCL (Printer Command Language):** HP 的竞争协议。PCL 是一种 Command-based protocol，更接近于控制打印机的寄存器，虽然快但不如 PS 精确且不具备设备无关性。
*   **Variable Data Printing (VPP):** 利用 PS 的编程能力，每一页的内容可以根据数据库动态变化（例如：`<Name> <Address>` 每页都不同），这被称为 Personalized Printing。
