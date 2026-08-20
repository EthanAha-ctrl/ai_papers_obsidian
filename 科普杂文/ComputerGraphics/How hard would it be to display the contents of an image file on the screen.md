
终端图像查看器 vv

### 1.3 GPU纹理压缩格式的被遗忘

这是一个很有趣的角度——**Block Compression (BC)** 和 **Ericsson Texture Compression (ETC)**：
但几乎没有图像查看器能打开这些格式！作者之前的 etcpak 项目就受限于 PNG 解码速度。

## 🖱️ 二、鼠标光标：第一个Side Adventure

### 2.1 Xcursor 格式

作者发现 Xcursor 文件格式意外地简单，**20行代码就能解析**：

```cpp
struct XcursorHdr {
    uint32_t magic;    // 魔数
    uint32_t header;   // 头部大小
    uint32_t version;  // 版本
    uint32_t ntoc;     // 目录条目数量
};

struct XcursorToc {
    uint32_t type;      // 类型
    uint32_t subtype;   // 子类型（通常是尺寸）
    uint32_t pos;       // 文件位置
};

struct XcursorImage {
    uint32_t width, height;  // 尺寸
    uint32_t xhot, yhot;     // 热点坐标
    uint32_t delay;          // 动画帧延迟
};
```

### 2.2 光标主题的混乱

作者发现 KDE 会因光标主题自引用而崩溃，wlroots 也有路径处理问题。更荒谬的是 **光标类型的标准化**：

> 当前 Wayland 的光标类型集合，是基于 **Web开发需求** 确定的。

文章展示了一组古老的 X11 光标，包括：
- 船、海盗旗、航天飞机
- 星际迷航企业号（版权问题？）
- Cryengine logo（？？）

不同桌面环境对同一个概念用不同的名字：
- "帮助"光标：GNOME 用 `help`，KDE 用 `whats_this`

### 2.3 Windows 光标格式 (.cur/.ani)

Windows 光标本质上是修改过的 ICO 格式，包含 BMP payload。但有几个坑：

1. **BMP 高度是实际高度的两倍**（因为同时存储颜色和 alpha mask）
2. **颜色和 alpha 使用不同的 bit depth**
3. **缩放时会出现"发光像素"**：

![](https://wolf.nereid.pl/img/cursor-glow.png)

原因是 Windows 95 的默认箭头光标是从其他光标类型"改造"来的，被 mask 掉的像素在缩放滤波时会"渗出"。

### 2.4 RIFF 容器的教训

Windows 光标使用 RIFF 容器（1985年发明）。作者的评价很犀利：

> "这在当时可能是绝妙的设计，但实践中是个 **misfeature**。"

问题：
- "万能容器"在写加载器时反而是负担
- 文件格式是不可变的，"可扩展"只是空想
- 需要建立目录、按正确顺序加载，一点也不简单

---

## 📚 三、图像加载库评述

作者给出了一个主观但很有参考价值的评分：

| 等级 | 库 | 评价 |
|------|-----|------|
| **stb_image tier** | libwebp | 调一个函数，搞定 |
| **Good tier** | libheif | README有简洁示例 |
| **Okay tier** | libpng, libjpeg | 需要阅读"导游指南"，混杂Amiga时代的低质量解码选项 |
| **Bad tier** | libjxl | Doxygen文档让人不知所措，返回码文档不全，输出浮点通道 |

关于 libjxl 的吐槽尤其精彩：

> "检查文件是否可能是 JPEG XL 时，文档说需要提供'文件开头'，但从未说明需要多少字节。阅读源码才知道是 **12 bytes**。"

---

## 🖥️ 四、终端图像显示技术

### 4.1 Unicode Block Elements

最基础的方案：使用 Unicode 的半块字符 `▄` 或 `▀`。

原理：
- 终端字体通常是 **宽度:高度 ≈ 1:2**
- 设置前景色和背景色，一个字符就变成两个"像素"

```
▄ → 上半部分 = 前景色，下半部分 = 背景色
▀ → 上半部分 = 背景色，下半部分 = 前景色
```

### 4.2 ANSI Escape Sequences 深度解析

这部分是很好的教程材料。基本结构：

```
ESC [ Pm m
```

其中：
- `ESC` = `\x1b` (ASCII 27)
- `[` = Control Sequence Introducer (CSI)
- `Pm` = 参数列表，用 `;` 分隔

**颜色演进**：

| 模式 | 格式 | 示例 |
|------|------|------|
| 16色 | `ESC[3Xm` | `ESC[31m` = 红色前景 |
| 256色 | `ESC[38;5;Xm` | `ESC[38;5;196m` |
| True Color | `ESC[38;2;R;G;Bm` | `ESC[38;2;255;128;0m` |

> 作者态度："True color 在1994年就标准化了，如果你的终端不支持，我不在乎。"

### 4.3 Sixel

1982年 DECwriter IV 引入的古老技术，最近被重新发现。

- 支持有限的颜色数量，需要 dithering
- 作者用 `libsixel` 作为 fallback，但文档匮乏

### 4.4 Kitty Graphics Protocol

这是 **vv 的主力输出方式**：

**特点**：
- 真彩色 + Alpha 通道
- 文档完善
- 被 Kitty、Konsole、Ghostty 等终端支持

**传输流程**：
```
RGBA像素数据 → deflate压缩 → base64编码 → 分割为4KB chunks → 发送
```

**动画支持**：Kitty 支持原生动画循环，发送帧序列后动画会持续播放，即使程序退出。

### 4.5 终端响应读取

终端不是单向的！发送 `CSI c` (Send Device Attributes) 后终端会响应，例如 `CSI ? 1 ; 2 c` 表示 "VT100 with Advanced Video Option"。

作者提供了完整的终端文件描述符获取代码，包括：
- 使用 `tcgetattr`/`tcsetattr` 设置非规范模式
- 禁用 echo
- 处理 poll 和 timeout（考虑慢速 SSH 连接）

---

## 🌈 五、HDR 图像处理

### 5.1 问题引入

OpenEXR 等 HDR 格式的数据无法直接在 SDR 显示器上显示：

- 整体太暗
- 高光区域被裁切，产生难看的饱和色块
- 暗部细节丢失

### 5.2 Tone Mapping

作者选择了 **PBR Neutral** tone mapping operator，原因：
- 只需要简单的数学实现
- 不需要大型 lookup table

其他选项：Tony McMapface, AgX

### 5.3 Linear vs sRGB

**关键概念**：EXR 存储的是 **线性颜色值**，对应光子数量的物理测量。

人眼对光的响应是指数曲线，所以需要 gamma correction。

**sRGB 转换公式**（正确的版本，不是简单的 1/2.2 power）：

$$L' = \begin{cases} 12.92 \times L & \text{if } L < 0.0031308 \\ 1.055 \times L^{1/2.4} - 0.055 & \text{if } L \geq 0.0031308 \end{cases}$$

变量说明：
- $L$ = 线性空间的亮度值 (0-1)
- $L'$ = sRGB 空间的亮度值 (0-1)
- 0.0031308 是线性段和指数段的分界点

这个线性段的设计是为了 **减少暗部的 color banding 和噪声可见度**。

---

## 🎨 六、色彩管理

### 6.1 为什么需要色彩管理

RGB 的 0-1 范围本身没有意义，必须在 **色彩空间** 的上下文中解释。

**Chromaticity Diagram** 展示了：
- 人眼可见的所有颜色（彩色区域）
- 不同色彩空间的 gamut（三角形）
- sRGB 的 gamut 相当有限，无法表现鲜艳的日落等

### 6.2 ICC Profile 与 Little CMS

图像中的色彩空间信息存储在 **ICC color profile** 中。

作者使用 Little CMS 库：

```cpp
// 创建输入 profile（从图像嵌入的 ICC）
cmsHPROFILE inputProfile = cmsOpenProfileFromMem(iccData, iccSize);

// 创建输出 profile（sRGB 用于显示器）
cmsHPROFILE outputProfile = cmsCreate_sRGBProfile();

// 创建转换
cmsHTRANSFORM transform = cmsCreateTransform(
    inputProfile, TYPE_RGBA_8,
    outputProfile, TYPE_RGBA_8,
    INTENT_PERCEPTUAL, 0
);

// 应用转换
cmsDoTransform(transform, inputBuffer, outputBuffer, pixelCount);
```

---

## 📺 七、YCbCr 与电视信号

### 7.1 Luminance vs Luma

**Luminance (Y)**：物理亮度测量
$$Y = 0.2126 \times R + 0.7152 \times G + 0.0722 \times B$$

**Luma (Y')**：gamma 压缩后的加权值

### 7.2 YUV/YCbCr 编码

电视信号为了向后兼容黑白电视：

$$U = B' - Y'$$
$$V = R' - Y'$$

**YCbCr** 类似，用于数字视频：

$$Y = 0.299 \times R + 0.587 \times G + 0.114 \times B$$
$$Cb = 0.564 \times (B - Y)$$
$$Cr = 0.713 \times (R - Y)$$

### 7.3 Chroma Subsampling

人眼对亮度变化更敏感，所以可以降低色差通道的分辨率：

| 格式 | 描述 |
|------|------|
| 4:4:4 | 无 subsampling |
| 4:2:2 | 水平方向色度减半 |
| 4:2:0 | 水平和垂直都减半 |

**Full Range vs Limited Range**：

- Limited range: Y 在 16-235，UV 在 16-240
- Full range: 使用完整的 0-255

---

## 🔬 八、HDR Profile 与 NCLX

### 8.1 HEIF/AVIF 的色彩信息

libheif 可以返回两种 profile：
1. **ICC profile**：完整的色彩空间描述
2. **nclx profile**：编码-independent 的代码点（CICP）

```cpp
struct heif_color_profile_nclx {
    enum heif_color_primaries color_primaries;           // 色原
    enum heif_transfer_characteristics transfer_characteristics;  // 传输特性
    enum heif_matrix_coefficients matrix_coefficients;   // 矩阵系数
    uint8_t full_range_flag;                             // 全范围标志
};
```

### 8.2 Matrix Coefficients 转换

YCbCr → RGB 转换：

```cpp
switch(matrix) {
case Conversion::BT601:
    a = 1.402f;  b = -0.344136f;  c = -0.714136f;  d = 1.772f;
    break;
case Conversion::BT709:
    a = 1.5748f; b = -0.1873f;   c = -0.4681f;   d = 1.8556f;
    break;
case Conversion::BT2020:
    a = 1.4746f; b = -0.164553f; c = -0.571353f; d = 1.8814f;
    break;
}

for(pixel : image) {
    R = Y + a * Cr;
    G = Y + b * Cb + c * Cr;
    B = Y + d * Cb;
}
```

### 8.3 PQ Transfer Function

**Perceptual Quantizer (PQ)** 用于 HDR，将 10-12 bit 整数映射到 0.0001-10000 nits：

```cpp
float Pq(float N) {
    constexpr float m1 = 0.1593017578125f;
    constexpr float m1inv = 1.f / m1;
    constexpr float m2 = 78.84375f;
    constexpr float m2inv = 1.f / m2;
    constexpr float c1 = 0.8359375f;
    constexpr float c2 = 18.8515625f;
    constexpr float c3 = 18.6875f;

    const auto Nm2 = std::pow(std::max(N, 0.f), m2inv);
    return 10000.f * std::pow(std::max(0.f, Nm2 - c1) / (c2 - c3 * Nm2), m1inv) / 255.f;
}
```

参数说明：
- $m_1, m_2$：曲线形状参数
- $c_1, c_2, c_3$：偏移和缩放参数
- $N$：归一化的整数编码值
- 返回值：以 nits 为单位的线性亮度

### 8.4 完整的 HDR 处理 Pipeline

```
加载图像
    ↓
将整数值转换为浮点（考虑 limited/full range）
    ↓
YCbCr → RGB 转换（使用 matrix coefficients）
    ↓
色彩管理（ICC 或 nclx）
    ↓
Linearize（应用 transfer function）
    ↓
Tone mapping
    ↓
Linear → sRGB
```

---

## ⚡ 九、性能优化

### 9.1 问题

一张 9504×6336 的 HDR 图像加载需要 **8秒**，不可接受。

### 9.2 多线程优化

最初的做法：**每个步骤并行化**

问题：每步都要读写 918 MB（9504×6336×4×4 bytes），内存带宽成为瓶颈。

**改进方案**：**顶层并行**，每个线程处理一个 chunk，完成所有步骤：

```cpp
// 每个 job
for(chunk : image) {
    load YCbCr chunk;
    convert to float;
    YCbCr → RGB;
    color management;
    transfer function;
    tone mapping;
    write to output;
}
```

chunk 小到可以放入 CPU cache，内存带宽问题解决。

### 9.3 SIMD 优化

#### Power Function

`std::pow()` 是 PQ 函数的瓶颈。

**数学基础**：
$$x^y = e^{y \ln x} = 2^{y \log_2 x}$$

使用 base-2 的原因：**IEEE 754 浮点数天然就是 base-2 表示**：

```
float = (-1)^S × 1.M × 2^(E-127)
```

所以：
$$\log_2(1.M \times 2^E) = \log_2(1.M) + E$$

mantissa 的对数可以用多项式近似（因为范围固定在 1-2）。

#### 性能结果

| 实现 | 时间 |
|------|------|
| Scalar | 1.56 s |
| SSE4.1 + FMA | 528 ms (3×) |
| AVX2 | 276 ms (5.6×) |
| AVX512 | 145 ms (10.8×) |
| AVX512 + 多线程 | **31 ms (50×)** |

**总计**：从 8 秒优化到 **0.8 秒**！

---

## ⚖️ 十、GPL 许可证争议

作者使用了一个"狡猾"的方法链接 GPL 许可的 poppler 库：

```cpp
typedef void*(*LoadPdf_t)(int, const char*, GError**);
// ... 其他函数指针

auto lib = dlopen("libpoppler-glib.so", RTLD_LAZY);
LoadPdf_t LoadPdf = (LoadPdf_t)dlsym(lib, "poppler_document_new_from_fd");
```

**作者的论点**：
1. GPL 说的是 "modify a work"，但链接不涉及复制或修改
2. 函数原型和库名是"事实陈述"，不受版权保护
3. 美国最高法院在 Google vs Oracle 案中裁定 API 使用是 fair use
4. 欧盟指令 2009/24/EC 对互操作性有保护

> "你甚至不需要安装 poppler 就能编译程序。解释一下怎么能被一个你不使用的库的许可证约束。"

---

## 📊 十一、最终结果

处理 9504×6336 HDR 图像的时间分解：

| 步骤 | 时间 |
|------|------|
| libheif 解码 YCbCr | 248 ms |
| YCbCr → RGB + color management + PQ + tone mapping | 355 ms |
| 图像缩放 | 111 ms |
| zlib 压缩 | 19 ms |
| 写入终端 | 127 ms |
| **总计** | **~860 ms** |

---

## 💡 十二、核心洞察与 Intuition Building

### 12.1 图像处理的"隐藏冰山"

表面上"加载并显示一张图片"实际涉及：

| 层面 | 问题 |
|------|------|
| **文件格式** | PNG/JPEG/AVIF/HEIC/EXR... |
| **色彩编码** | RGB/YCbCr/CMYK... |
| **色彩空间** | sRGB/Rec.2020/DCI-P3/CIE XYZ... |
| **Gamma/Transfer** | sRGB/PQ/HLG/Linear... |
| **动态范围** | SDR/HDR |
| **容器格式** | RIFF/HEIF/QuickTime... |

### 12.2 "简单API陷阱"

```cpp
heif_decode_image(hnd, &img, heif_colorspace_RGB, ...);
```

这看起来简单，但对于 HDR 图像会返回 **错误的结果**（8-bit，色彩错误）。

**教训**：简单 API 在高级场景下是陷阱。

### 12.3 标准化的碎片化

- 光标类型：X11 的混乱 → Web 标准
- 色彩空间：ITU-R BT.xxx 系列
- Transfer function：PQ/HLG/Linear/...

每个标准都在自己的领域定义了一套 enum 值，互不统一。

### 12.4 SIMD 的威力

**数据并行 + cache-friendly 的访问模式** = 巨大的性能提升

关键 insight：
- 浮点数的二进制表示直接编码了 exponent
- $\log_2$ 可以通过提取 exponent + mantissa 多项式近似来实现

---

## 🔗 参考链接

- [原文](https://wolf.nereid.pl/posts/viewing-images/)
- [vv 项目](https://github.com/wolfpld/vv)
- [ITU-T H.273 (CICP)](https://www.itu.int/rec/T-REC-H.273)
- [Little CMS](https://www.littlecms.com/)
- [Kitty Graphics Protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/)
- [PBR Neutral Tone Mapping](https://modelviewer.vn/sample-audio/)
- [Game Engine Black Book: Wolfenstein 3D](https://fabiensanglard.net/gebbwolfenstein/)
- [Chroma Subsampling Explained](https://www.rtings.com/tv/learn/chroma-subsampling)

---

## 📝 总结

这篇博客不仅是一个开发日志，更是一次对现代图像处理技术栈的全面审视。作者从"为什么不能在终端看图片"这个简单问题出发，一路深入到：

- 文件格式与容器
- 色彩科学与色彩管理
- HDR 与 tone mapping
- YCbCr 编码与电视信号历史
- SIMD 优化与浮点数表示
- 开源许可证的法律边界

**核心启示**：计算机图形学的"简单"问题往往隐藏着跨学科的复杂性，而"能用"和"正确"之间的差距，正是工程师价值的体现。