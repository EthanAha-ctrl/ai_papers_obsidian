---
source_pdf: Spatially-Varying Autofocus.pdf
paper_sha256: 102fdadc94dd22f04ef761519a0a76fb097fd084de1bd999999c4e44e9d24f28
processed_at: '2026-08-12T09:41:05-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话先抓住核心

你拍一张照片，lens 只能把**一个平面**对焦清楚，前面后面都糊。这篇 paper 说：我让**每个像素自己挑对焦距离**，于是整张图所有东西都清楚，而且是**光学上**就清楚了，不用后期 deconvolution。

就这么个事。听起来简单，但实现很巧。

---

## 问题是什么

传统相机有个根本矛盾：

- **想全清楚** → 缩小光圈（小 f/#）→ DoF 变大
- **缩光圈的代价** → (1) 进光少，暗；(2) 衍射 blur 变大，因为 Airy disk 大小 ∝ λ · f/#。你光圈缩到 f/22 以后，其实 sharpness 反而下降，这就是为什么手机 f/36 拍出来糊成一团。

那怎么办？现有的几条路都有问题：

1. **Focal stack**：拍 50 张不同对焦的图，每块选最 sharp 的拼起来。问题：慢，dynamic scene 没法用。
2. **Wavefront coding**（cubic phase plate）：故意把所有 depth 都拍成一样模糊的 PSF，然后 deconvolution 还原。问题：你一开始就丢了 sharpness，SNR 差。
3. **Light field camera**（Lytro 那种）：用 microlens array 记录 4D 光场，后期数字 refocus。问题：spatial resolution 大幅下降，因为 sensor 被切成 angular × spatial 两半。
4. **Coded aperture**：把光圈开个奇怪的洞，让 PSF 跟 depth 强相关，再解 inverse problem。问题：还是 inverse problem，要 deconvolution。

这篇 paper 走第五条路：**先估 depth，再让 lens 光学地把每个像素对焦到它该在的 depth**。这是 forward problem，不是 inverse problem。

---

## 核心 trick：Lohmann lens

要实现"每个像素自己挑焦距"，硬件上得有一个**焦距可变、且 spatially 可变**的 lens。这玩意儿怎么造？

关键 idea 来自 1970 年 Lohmann 发明的一个东西，叫 **Lohmann lens**（也叫 Alvarez lens，两人同年独立发明）。

### 直觉解释

想象两片**厚度随 $x^3$ 变化**的玻璃板，但方向相反：

- 第一片：$h_1(x) = \kappa x^3$，左边薄右边厚
- 第二片：$h_2(x) = -\kappa x^3$，左边厚右边薄

把它们叠在一起，**完全对齐**：相位互相抵消，$h_1 + h_2 = 0$，就是两片普通玻璃，没任何效果。

但如果把它们**横向错位** $\Delta$：

$$
h_1(x+\Delta) + h_2(x-\Delta) = \kappa\left[(x+\Delta)^3 - (x-\Delta)^3\right]
$$

你把这个展开，会得到：

$$
= \kappa(6\Delta x^2 + 2\Delta^3)
$$

**重点来了**：展开后剩下一个 $x^2$ 项，系数是 $6\kappa\Delta$。

而 $x^2$ 相位是什么？**就是 lens**。薄透镜的相位 profile 就是 $\phi(x) = -\frac{k}{2f}x^2$，$k = 2\pi/\lambda$。所以 $x^2$ 项的系数正比于 $1/f$。

所以：**错位量 $\Delta$ 直接控制焦距**。$\Delta$ 大 → $1/f$ 大 → $f$ 小 → 对焦近。$\Delta = 0$ → $f = \infty$ → 不对焦。

### 为什么 cubic 错位会产生 quadratic？背后的数学直觉

奇函数错位相减会留下偶次项。$(x+\Delta)^3 - (x-\Delta)^3$ 展开后 $x^3$ 项抵消（奇），$x^2$ 项保留（偶，因为两个 $\Delta$ 相乘），$x$ 项抵消（奇），常数项保留。所以剩下 $6\Delta x^2 + 2\Delta^3$。

更一般地，任何奇函数 $f(x)$ 错位相减 $f(x+\Delta) - f(x-\Delta)$，Taylor 展开里只留下偶次项。cubic 的特殊之处在于：**第一个非零偶次项就是 $x^2$**，所以直接得到 lens 相位。如果你用 $x^5$，错位后会有 $x^4$ 项，那个不是 lens，是更复杂的 aberration。

所以 Lohmann 选 cubic 不是随便选的，是数学上唯一最简单的选择。

---

## 把机械位移换掉：Split-Lohmann

Lohmann lens 需要物理平移玻璃板。慢、笨重、做不到 per-pixel。

作者 2023 年 SIGGRAPH 那篇 display paper（[Split-Lohmann multifocal displays](https://imaging.cs.cmu.edu/split-lohmann/)）搞了个 trick：**用 SLM 上的 phase ramp 模拟机械位移**。

### 原理

4f relay 系统里，中间有个 Fourier plane。Fourier optics 告诉我们：

- **Fourier plane 上放一个线性 phase ramp** $\exp(j \cdot 2\pi v \cdot u)$（$u$ 是 Fourier plane 坐标，$v$ 是 ramp slope）
- **等价于 output plane 上一个空间平移**，平移量正比于 $v$

所以你在 Fourier plane 写一个 ramp，光"虚拟地"在 cubic plate 上错位了。slope $v$ 取代了机械位移 $\Delta$。

slope 可程序化、可 per-pixel、可瞬时切换。这就是 SLM 替代机械的核心。

### 4f relay 的直觉

为什么 4f 能让 Fourier plane 上的 ramp 变成 image shift？因为 4f 系统中间那个 plane 是输入 image 的 Fourier transform。Fourier 域乘一个线性相位 = 空间域平移（Fourier shift theorem）。这是 undergrad signals 课的内容，$\mathcal{F}\{f(x-x_0)\} = F(u) e^{-j2\pi u x_0}$。

所以 Fourier plane 的 ramp slope = image shift 量 = cubic plate 的等效 $\Delta$ = 焦距。三层映射，全是线性关系，很干净。

---

## per-pixel 控制：sensor 和 SLM 光学共置

到这步还只是"全局可变焦"。要 per-pixel 可变焦，得让 sensor 上每个区域对应 SLM 上一个独立的 patch。

paper 用**第二个 4f relay** 把 sensor 和 SLM 光学共置（collocate）。意思是：sensor 上某个 pixel 看到的光，正好打过 SLM 上对应的那个小区域。于是 SLM 上写一个 **spatially-varying 的 phase ramp pattern**（不同区域不同 slope），sensor 上对应区域就各自对焦到不同 depth。

打个比方：传统 lens 像一个**全公司统一调空调温度**（一个 focal plane）。这套系统像**每个工位有自己的温控**（per-pixel focus）。SLM 就是那个"per-pixel 温控面板"。

架构图（paper Fig. 2b）的核心就是这步 collocation。

---

## 算法：怎么知道每个像素该对焦到哪

光有硬件没用，你得告诉 SLM 写什么 pattern。这就要估 depth map。paper 提了两套算法。

### 方法 A：CDAF（Contrast Detection Autofocus）的空间化版本

传统 CDAF：对着一个 patch，调 lens 直到 contrast 最大。**这里对每个 superpixel 都同时这么做**。

**为什么 superpixel 而不是 per-pixel？** 因为 contrast 是 local statistic，单个像素谈 contrast 没意义，得有一个 patch。但 patch 内部 depth 要尽量一致，否则一个 focus 给不了所有像素 sharp。

**Superpixel 怎么选？** 用 SLIC superpixel（[Achanta 2012](https://ieeexplore.ieee.org/document/6205760)）基于 texture 分割。因为 depth edge 通常 align with texture edge，所以 texture superpixel 内部 depth 大致一致。每轮重新 segment，因为图像越来越清楚，边界越来越准。

**怎么 search？** contrast 关于 diopter（$d = 1/z$，焦距倒数）的函数是 unimodal（一个 peak）。所以用 **ternary search**：

设工作范围 $[0, W]$ diopters。每轮拍 3 张图，对焦在 $\frac{W}{4}, \frac{W}{2}, \frac{3W}{4}$。哪张 contrast 最大，下一轮 search range 缩到它附近那一半。每轮 range 减半，log 收敛。

直觉：这就是在一维 unimodal 函数上做二分搜索的变体。3 张图评估 → range 减半 → 3 张图评估 → range 再减半。$K$ 轮 = $3K$ 张图 = 把 range 压缩 $2^K$ 倍。

paper 用 3 轮 10 张图就收敛（其实 9 张，加一张初始）。

### 方法 B：PDAF（Phase Detection Autofocus）的空间化版本

CDAF 还是要 search，要好几张图。PDAF 更猛：**一张图就告诉你 focus 该往哪调**。

**Dual-pixel sensor** 是什么？每个 microlens 下面有两个 photodiode，分别接 aperture 左半和右半的光。这相当于 aperture 内部的一个 mini-stereo，基线 ≈ aperture 直径。

- scene point 对焦到 sensor 上：两个 photodiode 收到同一根光 → disparity = 0
- point 在 focal plane 前：disparity > 0（一个方向）
- point 在 focal plane 后：disparity < 0

disparity 的 **sign 和 magnitude** 直接告诉你 focus 要调多少。这是 PDAF 比 CDAF 快的本质：**先验方向已知，不用 blind search**。

### PDAF 算法的细节

paper 用 [Ce Liu 的 optical flow solver](https://people.csail.mit.edu/celiu/) 算两路 dual-pixel 图像之间的 disparity。

但有个坑：disparity 估计主要靠 **vertical gradient** 强的像素驱动（因为 disparity 是水平方向的 stereo offset）。但 depth boundary 两侧，光靠 local 信息没法判断 disparity 该 assign 给哪边。

paper 的解法：用 [Segment Anything (SAM)](https://segment-anything.com/) 先把图像分成语义 layer（不同物体），**每个 layer 内部独立算 optical flow**，再合起来。

直觉：SAM 给你一个"这是狮子、这是背景"的 prior，让你知道哪些像素属于同一 depth 层。每个 layer 内部 depth 一致，flow 平滑，不会在边界附近被对面的像素污染。

这是把 **high-level semantic segmentation 当作 low-level 几何估计的 regularizer**，挺巧的。

### 收敛速度

PDAF 3 步、4 张图就达到 dense focal stack 69 张图的 PSNR。这个效率提升很显著。

---

## 实验：到底有多好

### 跟其他 AIF 方法对比（Table 1）

paper 把所有 AIF 方法列了个表，关键 4 个维度：

| 方法 | optical sharpness | 需要几张图 | AIF 怎么生成 | 出 depth 吗 |
|---|---|---|---|---|
| Small aperture | 低（衍射） | 1 | optical | 否 |
| Cubic phase plate | 低 | 1 | deconvolution | 否 |
| Focal sweep | 低 | 1 | deconvolution | 否 |
| Focal stack | 高 | 多（69） | contrast metric | 是 |
| Coded aperture | 低 | 1 | depth-dep deconv | 是 |
| Light field | 低（spatial res 损失） | 1 | contrast metric | 是 |
| Dual-pixel deblur | 低 | 1 | hard inverse | 是 |
| **Ours** | **高** | **2 起** | **optical** | **是** |

**Ours 是唯一一个"optical sharpness 高 + AIF 是 optical 生成 + 出 depth"的方法**。这是关键区分点。

### PSNR/SSIM 关于 #photos 的曲线（Fig. 11）

- Phase-based (4 photos) > Contrast-based (10 photos) > Focal stack (20 photos) > Focal sweep (1 photo)

4 张图胜过 focal stack 20 张。这就是"先估 depth 再 optical 对焦"的效率。

### MTF 对比（Fig. 12）

USAF target 三 depth 测 MTF：
- Ours（phase & contrast）≈ Focal stack（optical & computational）
- Focal sweep 明显差
- Small aperture (f/36) 高频衰减最快（衍射）

→ 我们在 spatial resolution 上跟 dense focal stack 持平，没牺牲。

### Freeform DoF 的玩法（Fig. 7, 8）

这套系统能做的不止 AIF。知道 depth 后，SLM 可以写任意 focal surface：

- **Tilt-shift without Scheimpflug**：直接在 SLM 上写一个倾斜的 focal plane，不用物理 tilt lens
- **Selective focus**：用户指定几个区域对焦，其余 defocus
- **Thin structure removal**（Fig. 8）：前面有 wire mesh，把 mesh 对应的 sensor 区域对焦到背景 depth，mesh 的 PSF 变成大 blur，几乎消失。这是个光学"occluder removal"trick

---

## 硬件限制

### 光效率 1/8

最大的痛点。原因：

- Phase-only SLM 用 **polarization-based phase modulation**，光要先过 polarizer，损失 1/2
- **Beamsplitter** 把光分到 SLM 再分回 sensor，50/50 两次，损失 1/4
- 总共 $1/2 \times 1/4 = 1/8$ 光到 sensor

rest state f/6.8，extreme focus 时 light throughput 降到 76%（因为 phase ramp 倾斜 cubic plate，effective aperture 缩小）。

改进方向：
- Reflective phase SLM（如 [TI DLP](https://www.ti.com/dlp)）替代 polarization-based LCoS
- 去掉 beamsplitter，用 off-axis geometry
- 大 diameter cubic plate 提升 throughput

### Prototype 成本

- Holoeye GAEA-2 SLM：$50k+ 量级
- Canon R10 dual-pixel camera
- Custom cubic phase plate（laser etching）
- 3× Samyang 85mm f/1.4 relay lens
- 40mm Macro 主镜头

不是普通 lab 能随手搭的。

---

## 我的直觉与联想

### 1. 这是 forward problem，不是 inverse problem

很多 computational photography 方法本质都是 inverse problem：capture 一个 degraded image，反推 sharp image。deconvolution 是 ill-posed，要 prior，要 regularization，对噪声敏感。

这篇 paper 把它变成 forward problem：先估 depth（这是个 well-posed 几何问题），再让 optics 直接 render sharp image。**computation 移到 capture 之前，不是 capture 之后**。

这跟 deep optics（[Wetzstein lab](https://computationalimaging.org/)）的哲学有点像，但他们还是 capture-then-deconvolve。这里更彻底，capture 之前就解决了。

### 2. 跟昆虫复眼的类比

蜻蜓的眼睛是 compound eye，每个 ommatidium 独立对焦自己的小视野。这套系统某种意义上是 **digital compound eye**——sensor 上每块区域像一只小 ommatidium，对焦自己负责的 depth。

但比昆虫强：昆虫每个 ommatidium 是独立 lens，这里用一个大 SLM + cubic plate 实现等价功能，更紧凑。

### 3. 跟 phased array radar 的类比

phased array radar 通过**每个 antenna 元素的相位** electronic 控制 beam 方向，不用机械转天线。

这里 SLM 是 optical phased array 的 2D 版本：每个像素的 phase 控制 local focal length。**radar 用 phased array 控制 beam direction，这里用 phase ramp 控制 focal surface**。本质上都是用 phase 编程控制光的传播方向。

### 4. 跟 SIMD vs MIMD 的类比

传统 lens 是 SIMD：所有像素共享一个 focal setting。
这套系统是 MIMD：每个像素有自己的 focal setting。

GPU 从 SIMD 演化到 MIMD（warp divergence 等）是为了更灵活。camera 从 global focus 演化到 per-pixel focus 也是为了更灵活。这是 hardware 朝着更细粒度控制演化的普遍规律。

### 5. 跟 NeRF 的对偶关系

NeRF 是 **inverse rendering**：从 2D images 反推 3D radiance field。
这套系统是 **forward rendering with known geometry**：知道 depth，optics 直接 render AIF。

两个对偶。有意思的是，如果我们的相机输出 AIF + depth 给 NeRF 当 ground truth，可能加速 NeRF 训练（特别是 [depth-supervised NeRF](https://arxiv.org/abs/2207.05294)）。

反过来，如果用 NeRF/Gaussian Splatting 在线估 depth，feed 给 SLM，可以做 closed-loop capture。

### 6. 跟 microscope 的关系

显微镜 EDOF（extended depth of field）一直是大问题。现在的方法：

- Z-stack + fusion：慢，对 live cell 不友好
- Wavefront coding：SNR 差
- [Double-helix PSF](https://www.pnas.org/doi/10.1073/pnas.0802910106)：复杂

这套系统直接 plug-in 显微镜，理论上能做 single-shot AIF microscopy。对 live cell imaging 这种不能长时间曝光的样品是杀手级应用。只是 SLM 帧率（~60Hz）限制高速场景。

### 7. 跟 Alvarez 眼镜的镜像

[Adaptive eyeglasses](https://en.wikipedia.org/wiki/Adaptive_lens) 用 Alvarez lens 给老花眼患者可调焦眼镜。一副眼镜，用户自己调焦距看远看近。

这套系统相当于**每个视网膜细胞戴一副独立的 Alvarez 眼镜**。如果未来 metasurface 能做到 per-pixel phase 控制，可能真的能做成超薄眼镜，每像素独立矫正（连散光、像差都能 per-pixel 补）。

### 8. 跟 holography 的关系

SLM 写一个 phase pattern 让光会聚到特定点 = **computer-generated hologram (CGH)** 的核心。这里每个 SLM patch 写一个 quadratic phase（= 一个 local lens），整个 SLM = 一组 local lenslet array，但 lenslet 焦距可程序化。

跟 holographic display 的区别：holographic display 写复杂 diffraction pattern 重构整个 light field；这里只写 piecewise-quadratic phase，每个 patch 一个 lens，更简单但足够做 AIF。

可以理解为 **holography 的特例**——只用 quadratic phase 项，不用高阶。

### 9. 跟 display ↔ camera duality

作者 2023 年做了 Split-Lohmann **display**（VR multifocal），2025 年这篇是 Split-Lohmann **camera**。光学对偶：

- Display 端：每像素 emit 出去的光，perceived depth 可控
- Camera 端：每像素 receive 进来的光，source depth 可控

很多 optical architecture 都有这种 duality（light field、integral imaging、computer-generated hologram）。这个思路可以推到别的 display tech：哪些 display trick 反过来能做 camera？这是个 fruitful 的 research direction。

### 10. 跟 LIDAR fusion

如果配一个 [LIDAR](https://en.wikipedia.org/wiki/Lidar) 给稀疏 depth prior，可以跳过 CDAF/PDAF 的 search，直接用 LIDAR depth 驱动 SLM。这是 hardware-level sensor fusion，比 software fusion 更直接。

### 11. 跟 event camera 结合

[DAVIS 346](https://inivation.com/dvs/) 这种 event+frame camera，event 触发的位置就是 contrast 变化的地方。如果用 event 信号驱动 autofocus，可以做 microsecond 级响应的 spatially-varying AF。高速运动场景很有用。

### 12. 跟 SPAD / ToF 的结合

[SPAD camera](https://www.nature.com/articles/s41928-024-01196-8) 可以 single-photon ToF 测 depth。如果 sensor 换成 SPAD，depth 直接来自 ToF，SLM pattern 直接由 ToF 算出。理论 2 次曝光就能 AIF，第一次 ToF 测深，第二次 SLM 写 pattern 对焦。

### 13. Chromatic aberration 问题

paper 用 broadband white light 测 PSF，但 cubic phase plate 有 dispersion（不同色光折射率不同），所以不同色光 focal length 不同，会有 chromatic artifact。

解决办法：
- Apochromatic design（refractive + diffractive hybrid 消色差）
- [Yang et al. SIGGRAPH Asia 2024](https://dl.acm.org/doi/10.1145/3680528.3687657) 的 differentiable ray-wave model 优化 hybrid phase plate

这是后续工作要解决的。

### 14. End-to-end differentiable optics 的潜力

cubic phase plate 现在是 fixed 的，SLM pattern 是 search 出来的。如果用 [Sitzmann et al. SIGGRAPH 2018](https://dl.acm.org/doi/10.1145/3197517.3201333) 的 end-to-end differentiable optics 框架，可以 jointly optimize：

- Cubic plate 的 profile（不再是纯 cubic，可能是任意奇函数）
- SLM pattern（不再纯 phase ramp，可能是任意相位）
- Downstream task（不是只 maximize contrast，可能 maximize detection accuracy）

这是把整个 optical system 当 neural network 训练，gradient 流过光学元件。meta-optics 的 software analog。

### 15. 跟 metasurface 的结合

cubic phase plate 现在是 bulk glass，laser-etched。如果换成 [metasurface cubic phase](https://www.nature.com/articles/ncomms5113)，可以做到 sub-wavelength thickness、flat、lightweight。整个系统可以做成手机厚度的 thin camera。这是 portable 化的关键。

### 16. Potential killer apps

我觉得真正能落地的几个方向：

- **显微镜**：live cell AIF imaging，不用 z-stack
- **半导体检测**：wafer inspection 需要宽 DoF 高分辨率
- **长距离 surveillance**：远距离 depth 变化小，SLM pattern 平缓，hardware 要求低
- **AR/VR pass-through**：跟作者 2023 display 工作闭环，capture + display 同一套 optics
- **天文 imaging**：wide FOV 望远镜保持 sharpness

---

## 总结直觉

我看完这篇 paper 后脑子里留下的几条核心 takeaway：

1. **cubic 错位 = lens**：两个相反 cubic plate 错位后，奇次项相消、偶次项保留，第一个非零偶次项就是 $x^2$ = lens。这个数学 trick 是整个系统的基石。

2. **Fourier plane 的 ramp = image shift**：这是 SLM 替代机械位移的物理基础。Fourier shift theorem 的硬件实现。

3. **sensor-SLM 光学共置 = per-pixel focus**：用 4f relay 把 sensor 和 SLM collocate，让每个 sensor 区域对应一个 SLM patch，于是 spatially-varying focus 成为可能。

4. **forward problem 优于 inverse problem**：估 depth + optical render 比 capture-then-deconvolve 在 SNR、resolution、speed 上都更好。这是个 fundamental 哲学转变。

5. **display ↔ camera duality**：optics 上的对偶关系是个金矿，很多 display trick 反过来能做 camera，反之亦然。

6. **光效率是瓶颈**：1/8 throughput 是真实问题，限制了 low-light 应用。reflective SLM + off-axis geometry 是改进方向。

7. **per-pixel control 是新维度**：从全局 focus 到 per-pixel focus，就像从 SIMD 到 MIMD，是 hardware granularity 的一次跃迁。这个 idea 在 optics 里还很新。

8. **闭环 vision**：capture → estimate depth → program optics → capture sharp。这是个 closed-loop active sensing 系统，跟 active vision、sensorimotor loop 在精神上相通。

希望这次"人话版"能帮你 build intuition。如果你想再 deep dive 哪个点（比如 4f relay 的 Fourier optics 推导、PDAF disparity 到 diopter 的精确映射、cubic plate 制造工艺），告诉我，我可以把数学和物理细节再展开。

Reference:
- [Paper project page](https://imaging.cs.cmu.edu/svaf)
- [Split-Lohmann display (前置工作)](https://imaging.cs.cmu.edu/split-lohmann/)
- [Lohmann 1970 original](https://opg.optica.org/ao/abstract.cfm?uri=ao-9-7-1669)
- [Alvarez lens patent](https://patents.google.com/patent/US3507565A)
- [Goodman, Introduction to Fourier Optics](https://www.elsevier.com/books/introduction-to-fourier-optics/goodman/978-0-12-408134-1)
- [SLIC superpixels](https://ieeexplore.ieee.org/document/6205760)
- [Segment Anything](https://segment-anything.com/)
- [Ce Liu optical flow](https://people.csail.mit.edu/celiu/)
- [Wetzstein computational imaging lab](https://computationalimaging.org/)
- [Double-helix PSF](https://www.pnas.org/doi/10.1073/pnas.0802910106)
- [Yang et al. hybrid refractive-diffractive](https://dl.acm.org/doi/10.1145/3680528.3687657)
- [Sitzmann end-to-end optics](https://dl.acm.org/doi/10.1145/3197517.3201333)
- [TI DLP](https://www.ti.com/dlp)
- [DAVIS event camera](https://inivation.com/dvs/)
- [Depth-supervised NeRF](https://arxiv.org/abs/2207.05294)

---

# Spatially-Varying Autofocus — 深度技术解读

Andrej 你好，这篇 paper 是 CMU 的 Yingsi Qin、Aswin Sankaranarayanan、Matthew O'Toole 三人 2025 年的工作。它的核心 insight 非常优雅：**把一个原本只对焦到一个 plane 的 lens，变成一个可以 per-pixel 程序化对焦到任意 depth surface 的 computational lens，从而在 sensor 上光学地（optically）一次性拍出 all-in-focus (AIF) 图像，零后处理**。下面我尽量把所有技术细节、公式、架构、related work、以及我自己脑中产生的联想都倒出来。

---

## 1. Big Picture & Core Insight

传统 lens 的物理约束：一个 lens 把光会聚到一个 plane（focal plane），所有不在该 plane 的 scene point 都会 defocus。要增大 DoF 就要缩小 aperture，但代价是 (i) light throughput 下降；(ii) diffraction blur 增大（Airy disk ∝ λ·f/#）。

这篇 paper 的核心想法：**DoF 形状本身应该是可编程的**。与其让所有像素共享一个 focal plane，不如让每个像素自己挑 focal length，这样 depth map 就直接编码进了 lens 上，sensor 收到的就是一张 optically AIF 的图。

项目主页：https://imaging.cs.cmu.edu/svaf

这个 idea 其实是把作者 2023 年 SIGGRAPH 的 Split-Lohmann **display** 工作 [Qin et al., ACM ToG 2023](https://dl.acm.org/doi/10.1145/3592098) **反过来用**：display 端是 "每像素选择 emit 出去的光的 perceived depth"，camera 端就是 "每像素选择接收到的光来自哪个 depth"。这种 display↔camera 对偶在 optics / computational photography 里很常见（light field、integral imaging、caustic design 都有类似对偶）。

---

## 2. 光学架构详解 (Split-Lohmann Computational Lens)

### 2.1 Lohmann / Alvarez lens 的物理基础

Lohmann lens（也叫 Alvarez lens）由两片 **cubic phase plate** 叠加而成。设：

- $x$：横向坐标（1D 推导，2D 可分离变量）
- $\kappa$：curvature-related parameter，控制 cubic profile 的强度，与折射率、plate 厚度梯度有关
- $\Delta$：两片 plate 之间的横向相对位移（标量，2D 推广为向量 $\boldsymbol{\Delta}=(\Delta_x, \Delta_y)$）

两片 plate 的 optical path profile：

$$
h_1(x) = \kappa x^3, \qquad h_2(x) = -\kappa x^3
$$

当它们以相对位移 $\Delta$ 叠加时，总 optical path 为：

$$
h_1(x+\Delta) + h_2(x-\Delta) = \kappa\left[(x+\Delta)^3 - (x-\Delta)^3\right]
$$

展开：

$$
(x+\Delta)^3 - (x-\Delta)^3 = 6\Delta x^2 + 2\Delta^3
$$

所以：

$$
h_{\text{tot}}(x) = \kappa\left(6\Delta x^2 + 2\Delta^3\right) \quad (1)
$$

**关键直觉**：
- $x^2$ 项是一个 **lens 的 quadratic phase**（薄透镜相位 $\phi(x) = -\frac{\pi}{\lambda f}x^2$，所以 $x^2$ 项对应一个焦距）
- 系数 $6\kappa\Delta$ 决定 focal length：$\frac{1}{f} \propto \Delta$
- 常数项 $2\kappa\Delta^3$ 是全局 phase 偏移，对成像无影响

因此，**通过移动 $\Delta$，可以连续变焦**。Lohmann 1970 原始 paper：[Applied Optics 9(7):1669](https://opg.optica.org/ao/abstract.cfm?uri=ao-9-7-1669)。

### 2.2 Split-Lohmann：去掉机械运动

原始 Lohmann 需要 **物理平移** 两片 plate，速度慢、难做成 per-pixel。Split-Lohmann 的关键 trick：

1. 把两片 cubic plate **光学共置**（用 4f relay，实际只用一片 cubic plate 即可，因为光通过它两次，相当于两个 cubic 作用叠加，等价于 Lohmann 对）
2. 在 4f 系统的 **Fourier plane** 放一个 **phase ramp** $\phi(u) = 2\pi v \cdot u$（$u$ 是 Fourier plane 坐标，$v$ 是 spatial frequency，即 ramp slope）
3. 这个 phase ramp 在 image plane 上等价于一个 image shift，而 shift 量正比于 $v$
4. 由于 cubic plate 是奇函数相位，shift cubic 一次等价于移动它，shift 两次后总相位 = Lohmann 的 cubic pair 相位

→ **slope $v$ 取代了机械位移 $\Delta$**，focal length 由 SLM 上 phase ramp 的 slope 直接控制。

这里有个 Fourier optics 的小细节：在 4f 系统中，Fourier plane 上的一个 **线性 phase ramp** $\exp(j\cdot 2\pi v u)$ 在 output plane 上对应一个 **平移** $\exp(j\cdot 2\pi v x_{\text{img}})$。所以 Fourier plane 上的 ramp ⟺ image plane 上的 shift。这就是 Lohmann 对的 mechanical translation 被替代的物理原理。可参考 [Goodman, Introduction to Fourier Optics](https://www.elsevier.com/books/introduction-to-fourier-optics/goodman/978-0-12-408134-1)。

### 2.3 Sensor-SLM 共置：per-pixel focus 控制的核心

paper 用 **第二个 4f relay** 把 sensor 和 SLM 在光学上共置（collocate）。这意味着 sensor 上每个 pixel 都在 SLM 上有一个对应的 patch（放大比取决于 relay lens 的焦距比）。

由于 SLM 是 phase-only 的、可以 **per-pixel 写入任意 phase pattern**，所以可以在 SLM 上同时显示 **spatially-varying 的 phase ramp**：每个 sensor 区域对应一个 slope $v(x,y)$，每个 slope 给出一个 focal length $f(x,y)$，每个 focal length 对焦到 depth $z(x,y) = 1/f$（按 Gaussian lens formula 在 thin lens 近似下）。

**这就是 spatially-varying autofocus 的硬件基础**。架构图（Fig. 2b）的关键：sensor 看到的不是单一 focal plane，而是一个 **focal surface**（depth 的空间分布）。

### 2.4 整体光路

光路大致如下（参考 Fig. 5 的 prototype photo）：

```
Scene
  → Objective lens (AF-S DX Micro NIKKOR 40mm f/2.8G)
  → Cubic phase plate (custom laser-etched)
  → 4f relay #1 (3x Samyang 85mm f/1.4)
      ↳ Fourier plane: Holoeye GAEA-2 phase-only SLM (4160×2464, 3.74μm pitch)
  → 4f relay #2
      ↳ Image plane: Canon R10 dual-pixel sensor (6000×4000, 3.72μm pitch)
```

PSF 形状随 SLM phase ramp slope 变化，见 Fig. 6：dot 在不同 depth 下，slope 不同时 PSF 收紧成一个点。

### 2.5 光效率问题

paper 提到的最大 limitation：
- Phase-only SLM 用 **polarization-based phase modulation**，光要先经过 polarizer（损失 1/2）
- 用 **beamsplitter** 把光分到 SLM 又分回 sensor，50/50 两次（损失 1/4）
- 总计最多 1/2 × 1/4 = 1/8 light 到达 sensor

rest state f/# = 6.8，extreme focus 时 light throughput 降到 76%。后续可以用 reflective SLM（如 [TI DLP](https://www.ti.com/dlp)）或 liquid crystal on silicon (LCoS) 优化。

---

## 3. Spatially-Varying CDAF (Contrast Detection Autofocus)

### 3.1 核心 idea

传统 CDAF：对一个 patch，调 lens 直到 local contrast 最大。**这里对每个 patch 都做这件事，但所有 patch 并行**，因为 SLM 可以同时给每个 patch 不同的 focus。

contrast 关于 depth（用 diopter $d = 1/z$ 度量）的函数 $C(d)$ 通常是 **smooth 且 unimodal**（只有一个 peak）。这是这个算法能 logarithmic search 的前提。

### 3.2 Logarithmic Search 算法

设 total working range = $[0, W]$ diopters。

**第 K 步**：
- capture 三张图，对应 focus 设置 $\frac{W}{4}$、$\frac{W}{2}$、$\frac{3W}{4}$ diopters
- 对每个 superpixel，计算三张图的 contrast（比如 local variance 或 gradient magnitude）
- 选最大 contrast 对应的 diopter，把 search range 缩小到该 diopter 周围的半区间

**例如**：若 $\frac{W}{4}$ 处 contrast 最大，下一轮 search range = $[0, \frac{W}{2}]$。

每轮 search range 减半，所以经过 $K$ 轮后，search range = $W / 2^K$。每轮 3 张图，总共 $3K$ 张图，达到 **linear search** 的精度，但 photos 数量 logarithmic。

直觉：这就像 ternary search on unimodal function，每轮 O(1) 评估缩小一半 range。

### 3.3 Superpixel Patching Strategy

关键设计：每个 "patch" 的定义不能随便。如果 patch 内部 depth 不一致，单个 focus 设定无法让所有像素清晰。

解决方法：用 **SLIC superpixel** [Achanta et al., TPAMI 2012](https://ieeexplore.ieee.org/document/6205760) 基于 **texture** 做分割。因为 depth edges 通常 align with texture edges，所以 texture superpixel 内部一般 depth 单调。

每轮 autofocus 后，更新 superpixel segmentation（因为图像更清晰，texture 边界更准）。新 superpixel 跨越多个旧 depth 时，选 contrast 最大的那个 depth。

### 3.4 动态场景

只搜索一次后，可以 incremental refine（因为物体不会瞬移），适合 dynamic scenes。这是 video autofocus 的关键优势。

---

## 4. Spatially-Varying PDAF (Phase Detection Autofocus)

### 4.1 Dual-Pixel Sensor 原理

Dual-pixel sensor：每个 microlens 下有 **两个 photodiode**，分别接收 aperture 左半和右半的光，相当于 **aperture plane 上的 stereo**。基线 ≈ aperture diameter。

- 当 scene point 在 focus：两个 photodiode 收到同一光线 → disparity = 0
- 当 point 在 focal plane 前（defocus near）：disparity > 0（一个方向偏移）
- 当 point 在 focal plane 后（defocus far）：disparity < 0

disparity 的 **sign 和 magnitude** 直接告诉你 focus 该往哪调多少。这是 PDAF 比 CDAF 快的本质原因：**one-shot 给出修正方向**。

参考 [Wadhwa et al., SIGGRAPH 2018](https://research.google/pubs/synthetic-depth-of-field-with-a-single-camera-mobile-phone/) 和 [Abuolaim et al., ICCV 2021](https://research.google/pubs/learning-to-reduce-defocus-blur-by-realistically-modeling-dual-pixel-data/)。

### 4.2 算法步骤

paper 用 [Ce Liu's optical flow](https://people.csail.mit.edu/celiu/) 的 conjugate gradient solver 计算 disparity：

1. Capture 一张 DP image（两个 photodiode 各自一张）
2. 用 [Segment Anything (SAM)](https://segment-anything.com/) 把图像分成语义 layers
3. 对 **每个 layer 独立** 计算 masked optical flow → 每个 layer 的 disparity map
4. 把所有 layer 的 disparity 加起来 → 全图 disparity map
5. disparity → focus 修正量（signed diopter shift）
6. 用新的 SLM pattern 重新 capture
7. 重复直到 convergence（paper 中 3 步即可）

### 4.3 Depth Boundary 问题

disparity 是水平方向的 stereo offset，所以主要是 **vertical gradient 强的 pixel** 在驱动 optical flow 估计。但 depth boundary 的两侧哪个像素该 assign 哪个 disparity？光看 local context 无法判断。

→ SAM 把图像分成语义区域（如不同物体），同一 layer 内部 depth 大致一致，**layer-wise optical flow 给出平滑的 disparity**，避免了边界附近的估计被两侧污染。

这是一个把 high-level semantic segmentation 当 prior 来 regularize low-level 几何估计的例子，类似 [depth in the wild](https://cs.cornell.edu/~snavely/projects/depthinthewild/) 的思路。

### 4.4 优势

- **single-shot 给方向**：1 张图就能 estimate 大致 disparity
- **不会陷入 local minimum**：CDAF 在 contrast 是多峰时可能 stuck
- **dynamic scene 友好**：每帧 capture 都更新 focus

### 4.5 缺点

- 需要 dual-pixel sensor（Canon R10 这里刚好有）
- large working range 时 disparity 大 → defocus 大 → optical flow 估计变难，但只要 sign 对，几步内就能 converge

---

## 5. 实验

### 5.1 对比方法

paper Table 1 给了一个很清晰的对比：

| 方法 | optical sharpness | # images | AIF generation | outputs depth |
|------|---|---|---|---|
| Small aperture | low (diffraction) | 1 | optical | no |
| Cubic phase plate [Dowski & Cathey 1995](https://opg.optica.org/ao/abstract.cfm?uri=ao-34-11-1859) | low | 1 | deconvolution | no |
| Focal sweep [Nagahara 2008](https://link.springer.com/chapter/10.1007/978-3-540-88690-7_5) | low | 1 | deconvolution | no |
| Focal stack [Nayar & Nakagawa 1994](https://ieeexplore.ieee.org/document/295916) | high | many | contrast metric | yes |
| Coded aperture [Levin et al. 2007](https://dl.acm.org/doi/10.1145/1276377.1276444) | low | 1 | depth-dependent deconv | yes |
| Light field cameras [Ng 2005](https://dl.acm.org/doi/10.1145/1186822.1073259) | low (sacrifice spatial res) | 1 | contrast metric | yes |
| Dual-pixel deblur [Xin et al. ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Xin_Defocus_Map_Estimation_and_Deblurring_From_a_Single_Dual-Pixel_ICCV_2021_paper.pdf) | low | 1 | hard inverse problem | yes |
| **Ours (SVA)** | **high** | **2 (min)** | **optical** | **yes** |

**关键差异**：
- optical sharpness: 大多 prior 工作（small aperture、cubic phase、coded aperture、focal sweep）都 **intentionally blur** 来获得 depth-invariant PSF，然后 deconvolution 还原。我们的方法是 optical 直接对焦，**没有 deconvolution**。
- spatial resolution: light field 用 microlens array 牺牲 spatial resolution 换 angular resolution。我们 **保持 full sensor resolution**。
- dynamic scene: focal stack 太慢；我们 2 张图就够。

### 5.2 定量结果 (Fig. 11)

PSNR/SSIM 关于 # photos 的曲线：
- **Phase-based** (4 photos at step 3) > contrast-based (10 photos) > focal stack (20 photos) > focal sweep (1 photo)

3 步 PDAF（4 张图）就达到 focal stack 69 张图的 PSNR。这个效率惊人。

### 5.3 MTF 对比 (Fig. 12)

USAF target 在三个 depth 上测 MTF：
- Phase-based ≈ Contrast-based ≈ Focal stack (optical) ≈ Focal stack (computational)
- Focal sweep 明显差
- Small aperture (f/36) 因 diffraction 在 high frequency 衰减最快

→ 我们的方法在保持 spatial resolution 的同时，sharpness 跟 dense focal stack 持平。

### 5.4 Freeform DoF 应用

- **Tilt-shift without Scheimpflug**（Fig. 7）：直接在 SLM 上写一个倾斜的 focal surface
- **Selective focus**（Fig. 7 右下）：用户指定几个区域对焦，其余 defocus
- **Thin structure removal**（Fig. 8）：场景前面有 wire mesh，通过把 wire mesh 对应的 sensor 区域对焦到背景 depth，让 mesh 的 PSF 变成 large blur，几乎消失。这是一个光学 " occluder removal " trick，跟 [code aperture separation](https://www.cs.cmu.edu/~ILIM/projects/07/00481/index.html) 有点像。

### 5.5 PSF 测试 (Fig. 6)

PSF 在不同 SLM slope 下：slope 越大，cubic plate 的 effective shift 越大，focal length 越短（depth 越近），PSF 从 large blur 收紧到 dot。多个 dot 同时清晰（图 6 右下），说明 spatially-varying focus 工作正常。

---

## 6. Discussion 与 Future Work

### 6.1 Aperture trade-off

rest state f/6.8，extreme focus 时 light throughput = 76%。原因：phase ramp 倾斜 cubic plate，相当于 aperture 缩小。可以用 **大 diameter cubic plate** 改进 throughput。

### 6.2 Aberration correction

两个方向：
- **End-to-end differentiable optics** [Sitzmann et al., SIGGRAPH 2018](https://dl.acm.org/doi/10.1145/3197517.3201333) 和 [Sun et al., SIGGRAPH 2021](https://dl.acm.org/doi/10.1145/3450626.3459764)：jointly optimize lens + SLM pattern 修 chromatic focal shift、depth boundary artifacts
- **Hybrid refractive-diffractive** [Yang et al., SIGGRAPH Asia 2024](https://dl.acm.org/doi/10.1145/3680528.3687657)：用 differentiable ray-wave model 优化 cubic phase plate profile 本身

### 6.3 Light efficiency

- 用 reflective phase SLM（如 [TI DLP](https://www.ti.com/dlp)）替代 polarization-based LCoS
- 去掉 beamsplitter，用 off-axis geometry
- 用 [Polarization-independent phase modulation](https://www.nature.com/articles/s41598-021-83813-0)

---

## 7. Related Work & Intuition Building

### 7.1 这个 idea 在 optics 历史中的位置

Lohmann 1970 提出可变焦 lens via cubic plates。Alvarez 同年（US Patent 3,507,565）独立提出，[Wikipedia: Alvarez lens](https://en.wikipedia.org/wiki/Alvarez_lens)。这是 **mechanical varifocal** 的经典设计，目前仍用于 phoropter（验光仪）等。

把 mechanical motion 替换为 SLM 上的 phase ramp 是 Split-Lohmann [Qin 2023](https://imaging.cs.cmu.edu/split-lohmann/) 的贡献，但只用于 VR display。这篇 paper 是把同一个 optical architecture 反过来用做 imaging。

### 7.2 跟其他 extended DoF 方法的对比直觉

| 类别 | 直觉 |
|------|------|
| **Small aperture** | 用物理 DoF 增加，但 diffraction 限制 spatial frequency cutoff = D/λ。我们用大 aperture + per-pixel focus 绕过 |
| **Wavefront coding (cubic phase)** | 把 PSF 变成 depth-invariant，然后一次性 deconvolution。问题：所有 depth 都模糊，SNR 差 |
| **Focal sweep** | 曝光时间内扫描 focal plane，PSF depth-invariant。同样 deconvolution 还原，且需要 moving lens |
| **Focal stack** | 多次 capture，每个 depth 选最 sharp 的合成。sharpness 好，但慢 |
| **Light field** | 用 angular 维度做 digital refocus，spatial resolution 大幅下降 |
| **Coded aperture** | 改 aperture shape 让 PSF depth-discriminative，再 inverse problem 解 depth。其实是个 blind deconvolution |
| **Dual-pixel** | aperture 内的 stereo，相对 disparity 给 defocus 方向。depth 估计 easy，但 deblur 还是 inverse problem |
| **Ours** | 把 depth 估计和 AIF 成像分开，AIF 是 **optical** 的，所以不需要 deconvolution |

直觉上，这个工作 **把 inverse problem 消解为 forward problem**：传统方法 capture blurred image → 反推 sharp image；我们 estimate depth → programmable lens 把 sharp image 直接投在 sensor 上。

### 7.3 跟 NeRF / neural rendering 的连接

很有趣的是，这篇 paper 的 AIF image 在某种意义上是 **scene depth-aware** 的 optical rendering。NeRF 是 **从 2D images 反推 3D radiance field**，然后用 volume rendering 合成新视角。这里则是 **从 depth map 直接 optical render AIF image**，类似 NeRF 的反过程。

如果我们的系统能 streaming 给一个 NeRF 模型 [Mildenhall et al. 2020](https://arxiv.org/abs/2003.08934) 高质量 AIF + depth ground truth，可能加速 NeRF 训练（特别是 depth-supervised NeRF [Deng et al. 2022](https://arxiv.org/abs/2207.05294)）。

### 7.4 跟 microscopy 的联系

显微镜的 **extended depth of field (EDOF)** 一直是个重要问题。传统方法：
- Z-stack + fusion
- Wavefront coding (cubic phase)
- [Double-helix PSF](https://www.pnas.org/doi/10.1073/pnas.0802910106) [Pavani et al., PNAS 2009]

我们的方法可以直接用于显微镜，把 slice by slice 的 z-stack 替换为 single-shot AIF。对 live cell imaging 这种不能长时间曝光的样品特别有用。

### 7.5 跟 computational displays 的双向 flow

camera 和 display 在 optics 上是对偶的。Split-Lohmann 已经在 display 端做 [multifocal VR display](https://imaging.cs.cmu.edu/split-lohmann/)。这篇 paper 把它反过来。未来的 vision 可能：
- **Bidirectional display+camera**：同一套 SLM + cubic plate optics，既能 capture AIF，又能 render multifocal display
- 对 telepresence、holography 很有意义

### 7.6 跟 differentiable optics / deep optics 的连接

[Gordon Wetzstein 的 DeepOpt lab](https://computationalimaging.org/) 多年来用 deep learning + differentiable optics 设计 [coded aperture](https://web.stanford.edu/class/ee367/Reading%20List/Levin_etal_SIGGRAPH2007.pdf)、[phase masks](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_PhaseCam3D_Learning_Phase_Masks_for_Passive_Single_View_Depth_ICCV_2019_paper.pdf)、[lens](https://computationalimaging.org/publications/)。

我们这个系统天然是 **programmable optical element**，所以可以在线学习 SLM pattern 来 optimize 某 task（比如 maximize downstream detection accuracy）。这是 [meta-optics](https://www.nature.com/articles/s41586-022-05166-0) 的 software-equivalent。

### 7.7 跟 phase mask optimization 的关系

cubic phase plate 本身是固定的，但 SLM 上的 pattern 可以任意程序化。一个可能的扩展：**用 neural network 输出 SLM pattern**，jointly optimize with downstream task。这跟 [metasurface cameras](https://www.nature.com/articles/s41928-024-01196-8) 的精神一致。

### 7.8 跟 holographic display 的关系

phase-only SLM 是 holographic display 的核心元件。COVE / Looking Glass 等都在做 holographic display。我们这里是把同样的硬件用于 capture。本质上 SLM 写一个 **computer-generated hologram (CGH)**，每个像素对应一个 local lens（quadratic phase），所以可以看作是一个 **piecewise lenslet array**，但 lenslet 的 focal length 是 per-pixel 可编程的。

这是 [holographic photography](https://en.wikipedia.org/wiki/Holography) 的一个简化形式。

---

## 8. 一些可能的延伸联想

### 8.1 跟事件相机 / DAVIS 结合

如果 sensor 换成 [DAVIS 346](https://inivation.com/dvs/) 这类 event camera，可以做 **high-speed spatially-varying autofocus**。event 触发的位置就是 contrast 变化的地方，可以作为 autofocus 的 cue。结合 [event-based AF](https://arxiv.org/abs/2003.02590) 可能在 fast motion 场景下很有用。

### 8.2 跟 SPAD / single-photon imaging 结合

[SPAD cameras](https://www.jstor.org/stable/26945329) (single-photon avalanche diode) 可以做 time-of-flight depth。如果把 SPAD 作为 sensor，depth 估计直接来自 ToF，SLM pattern 可以直接由 ToF depth 算出，理论上 AIF 在 2 次曝光内完成。

### 8.3 跟 metamaterials 的关系

[cubic phase plate](https://www.edmundoptics.com/p/150mm-x-150mm-cubic-phase-plate/15691/) 在 metamaterial 实现下可以做 flat optics。把 cubic plate 替换为 [metasurface cubic phase](https://www.nature.com/articles/ncomms5113)，整个系统可以做到 thin、flat、low weight，更适合 portable / drone 应用。

### 8.4 跟 neural fields 的结合

Depth map 在 paper 里是 dense focal stack ground truth 训练出的。如果用 [Instant-NGP](https://nvlabs.github.io/instant-ngp/) 或 [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 表征 scene，可以 robustly recover depth，再用我们的 lens 去 optically render AIF。

### 8.5 暗光场景

光效率 1/8 是个硬伤。如果用 [photon-counting](https://www.nature.com/articles/nphoton.2014.198) 或 [single-photon imaging](https://www.nature.com/articles/s41467-019-13712-6) 补偿光损失，或者把 SLM 换成更高效的 deformable mirror，可以在低光下用。

### 8.6 Spectral domain extension

paper 只测了 white light 下 PSF。cubic phase plate 的 dispersion 关系不同色光 focal length 不同（chromatic aberration），所以 AIF 在 broadband 下会有 chromatic artifacts。可以借鉴 [apochromatic design](https://en.wikipedia.org/wiki/Apochromat) 用 diffractive-refractive hybrid 修色差。

### 8.7 Large-scale / outdoor 场景

室外远距离 surveillance 是 paper 提到的应用之一。远距离时 depth 变化小，所以 SLM pattern 变化平缓，可以低分辨率 SLM 就足够。这降低 hardware 要求。

### 8.8 Astronomy

望远镜的 [Schiefspiegler](https://en.wikipedia.org/wiki/Schiefspiegler) 设计有类似 freeform focal surface 的需求。可能用于大视场成像、astrophotography 中保持 sharpness across wide FOV。

### 8.9 ToF / Lidar fusion

[Lidar](https://en.wikipedia.org/wiki/Lidar) 已经给出稀疏 depth。我们可以用 sparse lidar depth 作为 SLM pattern 的 initialization，加速 convergence。这是 [multimodal sensor fusion](https://arxiv.org/abs/2105.06789) 在 hardware 层面的体现。

### 8.10 跟 plenoptic camera 的本质对比

[Lytro](https://en.wikipedia.org/wiki/Lytro) 用 microlens array 把 sensor 切成 4D light field（spatial × angular），spatial resolution 损失严重。我们这里**不在 sensor 上做 trade-off**，而是把 angular 维度移到 SLM 上做程序化选择。某种意义上，SLM 是一个 **active light field selector**。

### 8.11 Coded exposure 与 motion blur

如果 SLM pattern 在曝光内变化（per-frame focus sweep），可以同时解决 motion blur + defocus blur。这跟 [Coded exposure photography](https://dl.acm.org/doi/10.1145/1276377.1276444) 的 idea 类似。

### 8.12 应用在 robotics / machine vision

paper 提到 machine vision。一个直接应用：[defocus microscopy for industrial inspection](https://www.olympus-lifescience.com/en/microscope-resource/primer/techniques/extendeddepthoffield)。半导体 defect detection 经常需要 wide DoF 高分辨率，我们的方法显然有优势。

### 8.13 3D printing 中的 cubic phase plate

paper 用 subtractive laser etching 做 cubic phase plate。未来可以用 [two-photon polymerization](https://www.nature.com/articles/s41598-020-6974-5) 或 [gray-tone lithography](https://www.thorlabs.com/) 做更平滑 profile，减少 scatter / aberration。

### 8.14 跟 [ Alvarez lens ] 在眼镜中的应用

[Adaptive eyeglasses](https://en.wikipedia.org/wiki/Adaptive_lens) 用 Alvarez lens 给老花眼患者可调焦眼镜。我们这里相当于每个像素都戴一副 Alvarez 眼镜，对于 **personalized vision correction** 的 hardware analog 是个有趣的镜像。

---

## 9. 复现 / 代码 / 数据

项目页：https://imaging.cs.cmu.edu/svaf
作者在 paper 中承诺 code、datasets、real-time video demos 都会开源。

要复现需要：
- Holoeye GAEA-2 SLM（约 $50k+）
- Canon R10 dual-pixel camera
- Custom cubic phase plate（需要 fab）
- 4f relay lenses（3× Samyang 85mm f/1.4）
- 控制软件：gphoto2 + LibRaw

CDAF 算法部分可以用普通 monitor + phase mask 软件模拟。PDAF 部分需要 dual-pixel raw 数据，可以用 [LibRaw](https://www.libraw.org/) 从 Canon CR3 raw 提取。

---

## 10. 总结我的 Intuition

这篇 paper 让我想到几条线索：

1. **Optical computation vs digital computation** 的张力：很多 computational photography 方法（coded aperture、wavefront coding、light field）都选择"capture less sharp image + deconvolution"。我们的方法选择"capture sharp image directly by programming the optics"。这是个 fundamental shift：**把 inverse problem 变成 forward problem**。

2. **Per-pixel control** 是关键：之前的 varifocal lens 都是全局控制。SLM 让我们可以 spatially-vary，从而把 focal **plane** 升级为 focal **surface**。

3. **Hardware cost vs algorithmic flexibility**：SLM 是 slow device（Hz ~ 60），不能做超快 autofocus，但够用。phase-only SLM 的光效率低（polarization 损失）是真实问题。

4. **从 display 反过来做 camera** 是个很好的思考框架：很多 display optics（multifocal、light field、holographic）都有对应的 camera 版本，反之亦然。这是一种 **optical duality**。

5. **Depth-first imaging**：future cameras 可能先 estimate depth（用任何 modality：ToF、stereo、PDAF、defocus），再用 programmable optics直接 optical render the desired image（AIF、tilt-shift、selective focus）。这是 **a different kind of computational camera** —— 把 computation 放在 optics 里，而不是 post-processing。

6. **跟 inverse rendering 的关系**：neural rendering 在 software 里做 inverse rendering，这里是 hardware 做 forward rendering with known scene geometry。两者对偶，可以互相 ground truth。

7. **Possible killer apps**：显微术（live cell imaging）、机器视觉（高速检测）、AR/VR capture-pass-through、长距离 surveillance、天文 imaging。

8. **What's next**：
   - 跟 metasurface 结合做 thin camera
   - 跟 neural network 结合 jointly optimize SLM pattern
   - 跟 SPAD / event camera 结合做 high-speed
   - chromatic aberration 修复
   - 光效率从 1/8 提升到 1/2+
   - 推广到 video AF + tracking + refocus 联合 pipeline

---

## Reference Links

- Paper project page: https://imaging.cs.cmu.edu/svaf
- Split-Lohmann display (前置工作): https://imaging.cs.cmu.edu/split-lohmann/
- Lohmann 1970 original: https://opg.optica.org/ao/abstract.cfm?uri=ao-9-7-1669
- Alvarez lens patent: https://patents.google.com/patent/US3507565A
- Wavefront coding (Dowski & Cathey): https://opg.optica.org/ao/abstract.cfm?uri=ao-34-11-1859
- Focal sweep: https://link.springer.com/chapter/10.1007/978-3-540-88690-7_5
- Shape from focus (Nayar): https://ieeexplore.ieee.org/document/295916
- Coded aperture (Levin): https://dl.acm.org/doi/10.1145/1276377.1276444
- Light field camera (Ng): https://dl.acm.org/doi/10.1145/1186822.1073259
- Dual-pixel defocus (Wadhwa): https://research.google/pubs/synthetic-depth-of-field-with-a-single-camera-mobile-phone/
- SLIC superpixels: https://ieeexplore.ieee.org/document/6205760
- Segment Anything: https://segment-anything.com/
- Optical flow (Ce Liu): https://people.csail.mit.edu/celiu/
- End-to-end optics (Sitzmann): https://dl.acm.org/doi/10.1145/3197517.3201333
- Differentiable ray tracing (Sun et al.): https://dl.acm.org/doi/10.1145/3450626.3459764
- Hybrid refractive-diffractive (Yang): https://dl.acm.org/doi/10.1145/3680528.3687657
- Double-helix PSF: https://www.pnas.org/doi/10.1073/pnas.0802910106
- NeRF: https://arxiv.org/abs/2003.08934
- Instant-NGP: https://nvlabs.github.io/instant-ngp/
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Goodman Fourier Optics: https://www.elsevier.com/books/introduction-to-fourier-optics/goodman/978-0-12-408134-1
- TI DLP: https://www.ti.com/dlp
- Lytro: https://en.wikipedia.org/wiki/Lytro
- Holography: https://en.wikipedia.org/wiki/Holography

希望这个详尽讲解能帮你 build intuition。如果你对某个具体公式（比如 4f relay 的 Fourier optics 推导，或者 PDAF disparity 到 diopter 的映射）想再 deep dive，告诉我，我可以把数学细节再展开。
