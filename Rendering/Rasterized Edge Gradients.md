---
source_pdf: Rasterized Edge Gradients.pdf
paper_sha256: 71288fea84cdfd86f9c4da9093db6a5babe7102cb05ff99c78f4f7faf5965e53
processed_at: '2026-08-11T20:56:00-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲这篇 Paper

参考：[ECCV 2024 Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6502_ECCV_2024_paper.php) | [Nvdifrast](https://nvlabs.github.io/nvdiffrast/) | [作者的 CVPR talk](https://www.youtube.com/watch?v=jNDBs6B6rSE)

---

## 想象一个场景

你想让电脑"看着一堆照片，反推出一个 3D 模型"。比如拍个人脸 100 张照片，让电脑猜出这个人的 3D 脸长啥样。

电脑的工作流是：
1. 猜一个 3D 模型
2. 用它"画"一张图
3. 跟真实照片比，算差距
4. 根据差距微调 3D 模型
5. 回到第 2 步，循环

第 4 步就是难点。电脑要回答："如果我把 3D 模型的某个点稍微往左挪 0.001 毫米，画出来的图会怎么变？" 这个问题叫 **gradient**。

知道 gradient，电脑就知道该往哪个方向调整模型。

---

## 为什么 Rasterization 的 Gradient 难算

Rasterization 是"把 3D 三角形画到 2D 像素网格上"的过程。每个像素就是一个小方块，要么属于某个三角形，要么不属于——没有中间状态。

想象一个红色三角形的边正好从某个像素中间穿过：

- 这个像素 70% 是红色，30% 是背景黑色
- 但 rasterization 的做法是：看像素中心点在不在三角形里。如果在，整个像素涂红；不在，整个像素涂黑
- 这是离散的"全有或全无"判定

问题来了：如果三角形往右挪 0.001 像素，像素中心可能突然从"在三角形里"变成"不在三角形里"，这个像素从红变黑。这是**跳变**，不是平滑变化。

数学上，"跳变"的地方没法算导数。就像 |x| 在 x=0 处有个尖角，左右导数不一样。

**这就是 differentiable rasterization 的核心难题**：怎么在"跳变"处算出有用的 gradient。

---

## 现有方法的笨办法

**Nvdifrast**（Nvidia 的方法，业界标杆）：把锐利的边"模糊"一下。像素中心虽然不在线上，但只要靠近，就给点部分红色。这样跳变变成平滑斜坡，可以求导了。

问题：
- 模糊会改变图片本身。你想算 depth map（深度图）的 gradient，模糊会把不同表面的深度混在一起，根本不对
- 模糊需要复杂的几何数据结构来追踪每个三角形覆盖了哪些像素
- 对 self-intersection（三角形互相穿插）完全无能为力

**Soft Rasterizer**：模糊得更狠。所有三角形按深度排序，每个像素是所有三角形的加权平均。gradient 好算了，但图片糊得没法看。

**光线追踪方法**（Mitsuba、Redner）：理论最对，但要在每条边上采样很多点算积分，慢得没法用。

---

## 这篇 Paper 的妙招：Micro-edge

核心 idea 一句话：**假装所有边都是阶梯状的小折线，而且这些小折线永远精确落在像素之间的缝隙里**。

什么叫"阶梯状小折线"？想象一条斜线（比如 45°）从左下到右上。它本来是直的。现在你把它替换成一串小台阶：先往右走一格，再往上走一格，再往右一格，再往上一格……整体方向还是 45°，但局部看全是水平或垂直的小段。

这篇 paper 说：我们就假装所有边都是这种阶梯，而且每个小台阶正好卡在两个像素中间的"缝隙"里。

**这个假装带来三个好处**：

1. **图片完全不变**。因为台阶永远在像素缝隙里，每个像素要么完全被覆盖，要么完全不被覆盖。rasterization 画出来的图跟原来一模一样。

2. **每条边都是局部问题**。原本斜线穿过一个像素，要算这个像素的 gradient，得考虑整条斜线。现在台阶是水平或垂直的，夹在某两个像素之间，只跟这两个像素有关。

3. **可以假装台阶能滑动**。虽然台阶位置固定在像素缝隙，但算 gradient 时假装它可以左右滑动一点点。滑动的效果是：左边像素失去一小条、右边像素获得一小条。

---

## 核心公式：用大白话

两个相邻像素 A 和 B 之间有一条边。算"边滑动一点点时 loss 怎么变"的公式是：

$$\frac{\partial L}{\partial p_{AB}} = \frac{1}{2}\left(\frac{\partial L}{\partial I_A} + \frac{\partial L}{\partial I_B}\right)(I_A - I_B) + \Omega$$

翻译成人话：

**边的 gradient = (A 像素的 loss gradient + B 像素的 loss gradient) 的平均 × (A 像素的值 − B 像素的值) + 平滑部分的贡献**

变量解释：
- $L$：你定义的 loss function，衡量画出来的图跟真实图差多少
- $p_{AB}$：A 和 B 之间那条边的位置
- $I_A, I_B$：A、B 两个像素的值（颜色、深度、或 mask 值）
- $\frac{\partial L}{\partial I_A}$：loss 对 A 像素值的导数，从神经网络反传回来的"loss 想让 A 变大还是变小"的信号
- $\Omega$：边没移动时，像素内部因为 shading、texture 等变化产生的 gradient，这部分 standard AD 能算，这篇 paper 不管它

**为什么这个公式对**：

- 如果 A 和 B 颜色一样（$I_A = I_B$），边在它们中间怎么滑动都不影响图，gradient 应该是 0。公式确实给 0 ✓
- 如果 loss 对 A 和 B 的 gradient 互相抵消（$\frac{\partial L}{\partial I_A} + \frac{\partial L}{\partial I_B} = 0$），边滑动让 A 变多一点 B 变少一点，对 loss 的净影响是 0。公式确实给 0 ✓
- 否则 gradient 正比于"颜色跳变大小"乘以"loss 敏感度"。颜色跳得越厉害、loss 越在意这两个像素，边的 gradient 越大 ✓

这就是个**点积**：边的重要程度 = 颜色差 × loss 关注度。

---

## 怎么把这个公式接到 3D 模型上

公式给的是"loss 对边位置的 gradient"。但我们要的是"loss 对 3D 模型顶点位置的 gradient"。中间差一步：**顶点移动时，边跟着怎么移动**？

这个映射分四种情况：

**情况 1：没有边**。两个相邻像素属于同一个三角形，或者都属于背景。顶点移动不影响这条边的位置（因为这条边不存在）。gradient 直接为 0，啥也不用做。

**情况 2：相邻三角形**。两个像素分别属于两个三角形，但这俩三角形共享一条边。比如人脸网格上，每个小三角面片跟邻居共享边。这种边在 mesh 内部，不是 visibility boundary。移动共享边同时影响两个三角形，pixel membership 不变。gradient 为 0，不散射。

**情况 3：一个三角形盖住另一个**。前景三角形遮挡背景。前景的顶点动，边跟着动（系数 = 1）；背景的顶点动，边不动（系数 = 0）。道理很简单：你遮住的东西动了，遮挡边界不变。

**情况 4：两个三角形互相穿插**。这是最难的情况，也是这篇 paper 相比 prior art 的最大优势。

---

## 自相交为什么难，这篇怎么解决

想象两个三角形在 3D 空间中互相穿过，像两张纸交叉。它们有一条交线，这条交线投影到图像上就是一条边。

这条边不是任何三角形的边！它是**视图相关**的：换个角度，交线投影就变了。

更麻烦的是，当你移动一个三角形的顶点时，交线可能：
- 跟着同方向移动（普通情况）
- 反方向移动
- 几乎不动（三角形接近平行时）
- 大幅跳动（三角形接近平行时，交线在远处）

Nvdifrast、Soft Rasterizer、甚至 Mitsuba 和 Redner 在这里都跪了。

这篇 paper 的解法用到 micro-edge 的简化：因为所有边都是水平或垂直的，交线投影到图像上也是水平或垂直的阶梯。所以可以在 2D 平面（x-z 或 y-z）里分析，不用考虑完整 3D。

核心公式（公式 12）：
$$\frac{\partial p}{\partial r} = -n_z^f \left[\mathbf{n}^{fT} \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \mathbf{n}^v\right]^{-1}$$

人话翻译：

**边在图像上移动的速度 = − 固定三角形法向的 z 分量 ÷ (固定三角形的切向 ⋅ 移动三角形的法向)**

变量解释：
- $p$：边在图像平面上的位置（x 坐标）
- $r$：移动的那个三角形的 fragment 位置
- $n_z^f$：固定三角形法向量的 z 分量，衡量它有多"正对镜头"
- $\mathbf{n}^f$：固定三角形的法向（2D，在 x-z 平面投影后）
- $\mathbf{n}^v$：移动三角形的法向
- $\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$：90° 旋转矩阵，把法向转成切向
- 分母整体：固定三角形的切向跟移动三角形法向的"对齐程度"

**直觉**：
- 两个三角形越接近平行，分母越接近 0，gradient 越爆炸。这有物理意义：平行平面的交线在无穷远，一点点动就跑很远
- 固定三角形越正对镜头（$n_z^f$ 大），边越敏感
- 负号是因为旋转方向约定的

paper 对两个三角形各算一次（先固定 A 算 B 移动的影响，再固定 B 算 A），用 product rule 加起来。

---

## 工程实现的小聪明

paper 把整个 differentiable renderer 拆成 5 个模块，最后一个是 EdgeGrad。最妙的设计是：

**EdgeGrad 模块的 forward pass 什么都不做，就是个 identity**。

它接收 rasterized image，原样输出。但它在 PyTorch 的计算图里注册了一个 custom backward。当 loss 反传到 EdgeGrad 时，它接收 $\frac{\partial L}{\partial I}$（loss 对每个像素的 gradient），然后用前面那些公式算出 $\frac{\partial L}{\partial \text{vertex}}$（loss 对每个顶点的 gradient），传给前面的模块。

为什么 forward 是 identity 这么重要？

因为 rasterization 可以输出各种东西：RGB 图、segmentation mask、depth map、normal map。如果像 Nvdifrast 那样在 forward 改图，mask 和 depth 就被模糊污染了——mask 模糊成 0.7 这种值根本没意义，depth 模糊把两个表面的深度混成平均值也是错的。

EdgeGrad 不动 forward，所以 mask 还是 0/1，depth 还是真实深度。只在 backward 时补上 discontinuity 处的 gradient。这是个很干净的 design。

---

## 实验结果的故事

**Forward gradient 对比**：在 4 个 test scene 上对比 finite differences（参考答案）、Mitsuba、Redner、Nvdifrast 和 EdgeGrad。Mitsuba 和 Redner 在普通 scene 上很准，但一遇到 self-intersection 就跪。Nvdifrast 因为假设"silhouette 边的三角形一定覆盖某个像素中心"，经常漏掉 gradient，结果很 noisy。**EdgeGrad 是唯一在所有 case 都能匹配 finite differences 的方法**。

**速度对比**：EdgeGrad 比 Mitsuba 快一个数量级，比 Nvdifrast 也快。原因是 EdgeGrad 只需要处理 boundary 像素（数量跟 $\sqrt{\text{像素数}}$ 成正比），不需要 edge sampling，不需要 connectivity 数据结构。

**Blender 数据集重建**：在 8 个 NeRF synthetic scene 上做 mesh 重建。EdgeGrad 比 Nvdifrast PSNR 高 1-2 dB。去掉 intersection handling 的 ablation 版本 PSNR 几乎一样，但 LPIPS（感知指标）明显差——说明 intersection artifact 虽然小但很影响视觉质量。

**Dynamic Head 重建**：这是 paper 最 impressive 的应用。用 mesh-based avatar 拟合多视角 lightstage 捕获的人脸表演。重点攻嘴巴内部：牙齿、舌头、嘴唇。

挑战：嘴巴内部有频繁的 self-intersection（舌头滑过牙齿、牙齿咬合、舌头推嘴唇），observability 差（很多角度看不到内部），topology 复杂。

以前的方法要么需要艺术家手工对齐，要么用物理仿真防止 self-intersection，要么用专门的嘴内采集设备。

paper 用 EdgeGrad + 2D segmentation mask 监督（用 HRNet 从相机图像预测 teeth/tongue/lips 的 mask 作为监督信号），全自动拟合出复杂的接触几何。Figure 9 和 10 展示了舌头滑过牙齿、牙齿咬合、舌头推嘴唇等场景，质量惊人。

---

## 这篇 Paper 在大图里的位置

Differentiable rendering 这个领域有两个流派：

**Volume-based**（NeRF、Gaussian Splatting）：把场景表示成密度场或一堆高斯点，每个像素是 ray marching 或 alpha blending 的结果，自然可微，没有 discontinuity 问题。但表达不了 mesh topology，不能直接做 animation、physics、registration。

**Surface-based**（mesh rasterization）：用三角形 mesh，渲染快、能编辑、能注册到 template。但 rasterization 的离散性让 gradient 难算。这就是这篇 paper 攻克的难点。

最近趋势是两者融合：mesh + neural texture，或者 neural SDF + rasterization。这篇 paper 让 mesh-based 方法的 gradient 质量追上了 volume-based 方法，让 mesh 路线在更多场景可行。

特别是嘴部重建这种"需要 mesh topology、需要处理 self-intersection、需要快速迭代优化"的场景，EdgeGrad 让 mesh 路线第一次成为实际选项。

---

## 局限性（paper 自己承认的）

- 不能处理透明（rasterization 固有的）
- 没有真正的 antialiasing（forward image 是 aliased 的，只有 backward gradient 修正）
- 不做 physically-based rendering，不处理光照材质
- 对 sub-pixel triangle 处理不好（finely tessellated mesh 中相邻三角形检测可能失效）

---

## 我觉得这个工作美在哪里

**数学上的简洁**：最终公式 $\frac{1}{2}(\frac{\partial L}{\partial I_A} + \frac{\partial L}{\partial I_B})(I_A - I_B)$ 美得像 finite difference。一个本来看起来很复杂的问题（discontinuity gradient、Dirac delta、线积分），经过 micro-edge 近似后塌缩成一个小学算术。

**工程上的干净**：forward identity + custom backward 这个设计，把"不破坏 forward image"和"补上 backward gradient"两件事干净分开。可以套到任何 rasterized output 上。

**理论上的统一**：paper 把 prior art 都装进一个 framework。Nvdifrast 是"修改 forward 让它连续"，Soft Rasterizer 也是，Li et al. 是"边采样但慢"，OpenDR 是"Sobel filter 后处理"。EdgeGrad 选择"不修改 forward，但用 micro-edge 近似让 backward 简化"——这是个之前没人探索的 design point。

**应用上的突破**：嘴部重建这种"刚需 self-intersection 处理"的场景，第一次用纯 mesh-based 方法做到全自动。这对 facial avatar、talking head、AR/VR 表情驱动有直接价值。

---

如果让我一句话总结这篇 paper 给你听：

> 它把"3D 模型边缘移动时图像怎么变"这个本该很复杂的问题，通过一个"假装所有边都是像素缝隙里的小阶梯"的近似，简化成"相邻两个像素的值差乘以 loss 对它们的关注度"——一个小学算术公式。这个近似不破坏图像本身，却能算出准确的 gradient，连三角形互相穿插这种复杂情况都能处理。

这就是用对了 viewpoint 之后，复杂问题塌缩成简单问题的典范。

---

# Rasterized Edge Gradients: Handling Discontinuities Differentiably 深度解析

你好 Karpathy！这篇 paper 是 Meta Reality Labs 的工作，解决的是 differentiable rasterization 中 visibility discontinuity 处的 gradient 计算问题。我会从直觉出发，深入到公式细节，build up 你的理解。

参考链接：
- Paper PDF (ECCV 2024): https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6502_ECCV_2024_paper.php
- Nvdifrast (对照基线): https://nvlabs.github.io/nvdiffrast/
- Mitsuba 3 (differentiable ray tracing): https://mitsuba-renderer.org/
- Redner (Li et al. edge sampling): https://github.com/BachiLi/redner
- Li et al. 2018 (理论源头): https://rgl.epfl.ch/publications/Li2018Differentiable
- Nicolet et al. 2021 (Laplacian preconditioning): https://rgl.epfl.ch/publications/Nicolet2021Large

---

## 1. 问题本质：为什么 rasterization 难以微分

Rasterization 的 forward pass 本质上做了三件事：
1. 把 triangle 投影到 image plane
2. 用 z-buffer 决定每个 pixel 看到哪个 triangle
3. 在 triangle 内部做 barycentric interpolation 得到 pixel 属性

步骤 3 在 triangle 内部是 smooth 的，可以用 standard AD 处理。步骤 2 是问题所在：pixel 的 "triangle membership" 是离散的，pixel 中心要么在 triangle 内要么在外。

考虑两类 pixel：
- **Interior pixel**：完全位于一个 triangle 内部。infinitesimal mesh motion 不会改变其 membership，所以 $\frac{\partial I}{\partial \phi}$ 中只有 smooth component（$\Omega$）贡献。
- **Boundary pixel**：位于 triangle 边界、occlusion 边界或 intersection 处。infinitesimal mesh motion 会改变其 membership，这部分 gradient 来自 visibility discontinuity。

**核心难点**：在 boundary pixel 处，pixel value $I$ 关于 vertex position 是 piece-wise constant（或 piece-wise linear）的，在 $p_{AB}=0$ 处有 **C0 或 C1 discontinuity**，standard AD 算不出来。

现有方法的缺陷：
- **Nvdifrast** [Laine 2020]：用 analytic antialiasing 把 sharp edge 变成 smooth transition。问题：(1) 假设 silhouette edge 的 triangle 一定覆盖某个 pixel center（不总是成立），(2) 修改了 forward image，(3) 复杂的 connectivity 数据结构，(4) 无法处理 triangle intersection。
- **Soft Rasterizer** [Liu 2019]：模糊边界 + 平均 depth。问题：smoothing 把不同 surface 的 normals/depth 混在一起，破坏 geometry 完整性。
- **Ray tracing 方法** [Li 2018, Zhang 2019]：理论 principled，但需要 silhouette edge sampling，计算昂贵。

这篇 paper 的核心 insight 用一句话概括：**假设 rasterized image 是某个连续过程的输出，而这个连续过程"恰好"与离散 rasterization 对齐**。这样 forward image 完全不变，只在 backward 时计算 gradient。

---

## 2. 理论基础：从 Li et al. 2018 出发

### 2.1 Pixel value 的积分形式

公式 (1)：
$$I = \iint_D k(x, y) R(x, y) \, dx \, dy$$

变量含义：
- $I$：单个 pixel 的 intensity（RGB 通道之一，或 scalar mask value）
- $D$：pixel footprint，通常是以 pixel center 为中心的单位正方形
- $k(x, y)$：pixel filter kernel，例如 box filter（$k=1$ in $D$）或 Gaussian
- $R(x, y)$：radiance function，描述场景在 image plane 上每点的辐射

记 $f(x,y) = k(x,y) R(x,y)$ 简化记号。

### 2.2 目标：对 scene parameter 求 gradient

公式 (3)：
$$\frac{\partial I}{\partial \phi} = \frac{\partial}{\partial \phi} \iint_D f(x, y) \, dx \, dy$$

- $\phi$：scene parameter，通常是 mesh vertex 的 3D position（也可以是 texture、camera 参数等）
- 我们想知道 mesh 顶点稍微移动一点，pixel value 怎么变

### 2.3 把 $f$ 分成两个半空间

公式 (4)：
$$f(x, y) = \theta(\alpha(x, y)) f_a(x, y) + \theta(-\alpha(x, y)) f_b(x, y)$$

变量含义：
- $\alpha(x, y) = 0$：定义一条 edge（直线），把 pixel footprint 分成两个区域
- $\theta(\cdot)$：Heaviside step function，$\theta(z) = 1$ if $z>0$，$\theta(z)=0$ if $z<0$
- $f_a(x, y)$：edge 一侧（$\alpha > 0$）的 smooth radiance（来自 triangle A 的 shading）
- $f_b(x, y)$：edge 另一侧（$\alpha < 0$）的 smooth radiance（来自 triangle B 或 background）

**直觉**：pixel 被 edge 切成两块，每块有自己的 smooth shading function。$f$ 在 $\alpha=0$ 处有 jump。

### 2.4 对积分求导：product rule 展开

公式 (5)：对 $f_a$ 部分应用 product rule
$$\frac{\partial}{\partial \phi} \iint_D \theta(\alpha) f_a \, dx\,dy = \iint_D \delta(\alpha) \frac{\partial \alpha}{\partial \phi} f_a \, dx\,dy + \iint_D \theta(\alpha) \frac{\partial f_a}{\partial \phi} \, dx\,dy$$

两项含义：
- **第一项（Dirac delta 项）**：edge 移动导致 intensity 变化。$\delta(\alpha)$ 是 Dirac delta，集中在 edge 上。$\frac{\partial \alpha}{\partial \phi}$ 是 edge equation 对 scene parameter 的 derivative（edge 怎么随 vertex 移动）。
- **第二项 $\Omega$**：smooth 区域的 derivative。$f_a$ 内部（例如 texture、barycentric interpolation）对 $\phi$ 的 sensitivity。standard AD 能算。

paper 把第二项记为 $\Omega$ 并暂时忽略，专注于第一项。

### 2.5 Dirac delta 转为线积分

公式 (6)：
$$\iint_D \delta(\alpha(x, y)) \frac{\partial \alpha}{\partial \phi} f_a \, dx\,dy = \int_{\alpha(x, y) = 0} \frac{\partial \alpha}{\partial \phi} \|\nabla_{x,y} \alpha(x, y)\|^{-1} f_a \, dt$$

变量含义：
- $\|\nabla_{x,y} \alpha(x, y)\|$：edge equation 在 image plane 上的 gradient 的 $L^2$ norm，$\sqrt{(\partial \alpha/\partial x)^2 + (\partial \alpha/\partial y)^2}$
- $dt$：沿 edge 的弧长参数

**为什么有 $\|\nabla \alpha\|^{-1}$**：这是 Dirac delta 函数的变量替换 Jacobian。如果 $\alpha(x,y) = ax+by+c$，那么 $\delta(\alpha) = \delta(ax+by+c)$ 沿 edge 的线积分需要除以 $\sqrt{a^2+b^2}$ 才能正确归一化。直觉上：edge 越倾斜，单位 $\alpha$ 变化对应的 image plane 距离越小，需要补偿。

这一步是 Li et al. 2018 的核心贡献：把"visibility 边界移动"的 gradient 转换为沿 boundary 的线积分。但 ray tracing 中实现这个需要 edge sampling，昂贵。

---

## 3. Micro-edge：核心构造

### 3.1 关键 insight

paper 的核心 idea：

> 假设 rasterized image 是某个连续过程（带 antialiasing filter）的输出，这个连续过程"恰好"与离散 rasterization 在像素级对齐。

具体构造为 **micro-edge**：
- 任意方向的 edge（triangle 边、intersection 线）都被替换为一串 micro-edges
- 每个 micro-edge 要么严格水平，要么严格垂直
- micro-edge 总是精确位于两个像素之间

这样得到四个性质：
- (a) Boundaries 永远不与像素内部相交，只在像素之间
- (b) 像素永远完全被覆盖或完全不被覆盖（保持 rasterization 的离散性）
- (c) 相邻像素之间最多一条 boundary
- (d) 不需要访问 source geometry 来定位 edge

**关键 trick**：antialiasing filter 在这种构造下变成 identity operation（因为 boundary 永远在像素之间，filter 不会跨 boundary 平均），所以 forward image 完全等于 rasterized image。这避免了 Nvdifrast 和 Soft Rasterizer 修改 forward image 的问题——对 normals、depth、segmentation masks 至关重要。

### 3.2 为什么这个近似合理

直觉上的担忧：把斜边替换成阶梯状 micro-edges，perimeter 都变了，gradient 不会出错吗？

paper 用 **divergence theorem** 解释：

公式 (13)：
$$\oint_C \mathbf{f} \cdot \mathbf{n} \, dt = \iint_D (\nabla_\phi \cdot \mathbf{f}) \, dx\,dy$$

变量含义：
- $C$：某条闭合 boundary
- $\mathbf{n}$：boundary 的外法线
- $\mathbf{f}$：vector field（这里 $\mathbf{f}$ 的具体形式来自 gradient 推导）
- $\nabla_\phi \cdot \mathbf{f}$：divergence

公式 (6) 中的 $\frac{\partial \alpha}{\partial \phi}$ 项本质上是 boundary 的法线方向（gradient of edge equation 就是法线方向），所以左边的 boundary integral 可以转换为右边的 area integral。

**关键观察**：micro-edge 与原 edge 之间的差异区域面积 $\to 0$ as pixel size $\to 0$。所以在 area integral 视角下，micro-edge 构造在极限下收敛到原 formulation 的解。

这是个巧妙的论证：作者不直接说 micro-edge 准确，而是说"在像素无限小的极限下准确"。对于实际像素大小，误差是 $O(\text{pixel size})$ 量级。

---

## 4. Pixel Pair：核心公式推导

### 4.1 设定

考虑两个相邻 pixel A、B，被一条 edge 分开。设为 horizontal pair（vertical edge 在它们之间）。

公式 (7)：
$$I_A = \iint_{D_A} \theta(\alpha) f_a + \theta(-\alpha) f_b \, dx\,dy$$
$$I_B = \iint_{D_B} \theta(-\alpha) f_a + \theta(\alpha) f_b \, dx\,dy$$

变量含义：
- $D_A, D_B$：pixel A、B 的 footprint（互不相交的单位正方形）
- $\alpha(x, y) = p_{AB} - x$：edge equation，$p_{AB}$ 是 edge 的 x 坐标
- 当 $p_{AB}=0$ 时 edge 正好在 pixel A、B 之间

注意 $I_A$ 和 $I_B$ 的表达式中 $f_a, f_b$ 的位置互换：在 pixel A 一侧 $f_a$ 占主导（$\alpha>0$），在 pixel B 一侧 $f_b$ 占主导。

### 4.2 Loss 对 edge 位置的 gradient

公式 (8)：
$$\frac{\partial L}{\partial p_{AB}} = \frac{\partial L}{\partial I_A} \frac{\partial I_A}{\partial p_{AB}} + \frac{\partial L}{\partial I_B} \frac{\partial I_B}{\partial p_{AB}}$$

变量含义：
- $L$：scalar loss function
- $\frac{\partial L}{\partial I_A}, \frac{\partial L}{\partial I_B}$：从 loss 反传回来的 incoming gradient（AD 计算）
- $\frac{\partial I_A}{\partial p_{AB}}, \frac{\partial I_B}{\partial p_{AB}}$：未知量，需要推导
- $p_{AB}$：edge 位置（暂时把它当 scene parameter）

### 4.3 C1 discontinuity 问题

直接对 $I_A$ 关于 $p_{AB}$ 求导遇到 C1 discontinuity：

公式 (9)（one-sided limits）：
$$\frac{\partial I_B}{\partial p_{AB}}^- = \frac{\partial I_A}{\partial p_{AB}}^+ = 0$$
$$\frac{\partial I_B}{\partial p_{AB}}^+ = \frac{\partial I_A}{\partial p_{AB}}^- = \int_{x=0} [f_a(x, y) - f_b(x, y)] \, dy$$

**直觉**：edge 在 $p_{AB}=0$ 处。
- 如果 $p_{AB}$ 从 0 增到 $\epsilon>0$（edge 向右移），pixel B 失去一小条 $f_a$ 区域、获得一小条 $f_b$ 区域，所以 $\frac{\partial I_B}{\partial p_{AB}}^+ = \int [f_a - f_b] dy$。同时 pixel A 不变（edge 还没进入 A 内部），所以 $\frac{\partial I_A}{\partial p_{AB}}^+ = 0$。
- 如果 $p_{AB}$ 从 0 减到 $-\epsilon<0$（edge 向左移），对称地 $\frac{\partial I_A}{\partial p_{AB}}^- = \int [f_a - f_b] dy$，$\frac{\partial I_B}{\partial p_{AB}}^- = 0$。

注意 $\|\nabla_{x,y}\alpha\| = 1$（因为 $\alpha = p_{AB} - x$，gradient 是 $(-1, 0)$），所以公式 (6) 中的 Jacobian 项为 1。

### 4.4 平均 one-sided limits

公式 (10)：
$$\frac{\partial I_B}{\partial p_{AB}} = \frac{\partial I_A}{\partial p_{AB}} = \frac{1}{2} \int_{x=0} [f_a(x, y) - f_b(x, y)] \, dy$$

**直觉**：C1 discontinuity 意味着 sub-derivative 不唯一，作者取左右极限的平均。这等价于把 Dirac delta 在边界处"半属于 $D_A$，半属于 $D_B$"。

### 4.5 利用 micro-edge 性质简化

micro-edge 假设：像素内值恒定（"pixels are always either fully covered or not covered at all"）。所以 $f_a, f_b$ 在 $D_A, D_B$ 内分别是常数，等于 $I_A, I_B$。

公式 (11)（**核心结果**）：
$$\boxed{\frac{\partial L}{\partial p_{AB}} = \frac{1}{2} \left(\frac{\partial L}{\partial I_A} + \frac{\partial L}{\partial I_B}\right) (I_A - I_B) + \Omega}$$

变量含义：
- $\frac{\partial L}{\partial p_{AB}}$：loss 对 edge 位置的 gradient
- $\frac{1}{2}\left(\frac{\partial L}{\partial I_A} + \frac{\partial L}{\partial I_B}\right)$：两个相邻 pixel 的 incoming gradient 的平均
- $(I_A - I_B)$：两个相邻 pixel 的 intensity 差
- $\Omega$：smooth component（之前暂时忽略的第二项，由 AD 处理）

**这个公式的 intuition 极其优美**：
- 如果 $I_A = I_B$（两个像素看起来一样），edge 移动对 loss 没影响，gradient = 0。合理！
- 如果 $\frac{\partial L}{\partial I_A} + \frac{\partial L}{\partial I_B} = 0$（两个像素的 incoming gradient 互相抵消），edge 移动也没影响。合理！
- gradient 大小正比于 intensity jump 和 incoming gradient 的乘积。

这是个**点积形式**：edge gradient = (average incoming gradient) · (intensity difference)。

### 4.6 等价的 alternative derivation

Supplementary material §A.0 给了另一种推导：把 pixel pair 视为整体，平均 intensity $I_{AB} = \frac{I_A + I_B}{2}$。这样 $I_{AB}$ 关于 $p_{AB}$ 是连续可微的，没有 C1 discontinuity。

公式 (25)：
$$\frac{\partial I_{AB}}{\partial p} = \frac{1}{2}(I_B - I_A)$$

然后近似 $\frac{\partial I_A}{\partial p_{AB}} \approx \frac{\partial I_B}{\partial p_{AB}} \approx \frac{\partial I_{AB}}{\partial p_{AB}}$，得到相同结果。

这个推导更简洁，但物理含义稍弱：它假设两个 pixel 的 derivative 相等，是个近似。

---

## 5. Edge Classification 和 Gradient Scattering

### 5.1 从 edge gradient 到 fragment gradient

公式 (11) 给的是 $\frac{\partial L}{\partial p_{AB}}$（对 edge 位置的 gradient）。但实际目标是 $\frac{\partial L}{\partial r}$（对 fragment 3D 位置的 gradient），然后通过 AD 反传到 vertex。

转换：
$$\frac{\partial L}{\partial r} = \frac{\partial L}{\partial p_{AB}} \cdot \frac{\partial p_{AB}}{\partial r}$$

- $r$：fragment 在 3D clip space 的位置
- $\frac{\partial p_{AB}}{\partial r}$：fragment 移动单位距离时 edge 移动多少

$\frac{\partial p_{AB}}{\partial r}$ 依赖于 edge 类型。

### 5.2 四种 edge 类型

paper 通过比较相邻 pixel 的 triangle ID 来分类：

| 类型 | 检测方法 | $\frac{\partial p}{\partial r}$ |
|------|---------|-------------------------------|
| No edge | triangle ID 相同 | 0（不散射） |
| Adjacent primitives | 两个 pixel center 互不在对方的 triangle 内 | 0（不散射，因为 shared edge 不构成 visibility discontinuity） |
| Overhanging | 一个 pixel center 在对方 triangle 内，另一个不在 | 前景 = 1，被覆盖 = 0 |
| Intersecting | 两个 pixel center 都在对方 triangle 内 | 见下节公式 |

**Overhanging 的 intuition**：前景 triangle 直接覆盖背景。前景 fragment 移动 1 像素，edge 也移动 1 像素（$\frac{\partial p}{\partial r} = 1$）。背景 fragment 移动不影响 edge（$\frac{\partial p}{\partial r} = 0$）。

**Adjacent 的 intuition**：两个 triangle 共享一条边，这条边在 mesh 内部，不是 visibility boundary。移动 shared edge 同时改变两个 triangle，不会改变 pixel membership（理论上互相抵消）。所以不散射 gradient。但 $\Omega$ 项仍然存在。

### 5.3 检测算法

paper 用 simple test：
1. 取相邻 pixel A、B
2. 比较 triangle ID
3. ID 相同 → no edge
4. ID 不同 → 进一步测试：
   - 把 A 的 pixel center 测试是否在 B 的 triangle 内
   - 把 B 的 pixel center 测试是否在 A 的 triangle 内
5. 都在 → intersecting；一个在 → overhanging；都不在 → adjacent

这个检测避免了 Nvdifrast 需要的 hash map connectivity 数据结构。

---

## 6. Geometry Intersections：复杂情况

### 6.1 问题

两个 triangle 在 3D 中相交，intersection 线投影到 image plane 形成 edge。这种 edge 不是任何 triangle 的边，是**视图相关的**。

当 fragment 移动时，intersection edge 移动的方式复杂：可能同向、反向、甚至不动（取决于两个 triangle 的相对方向）。

### 6.2 简化到 2D

由于 micro-edge 假设，所有 edge 都是水平或垂直。所以：
- Vertical edge → 在 x-z 平面分析
- Horizontal edge → 在 y-z 平面分析

z 轴垂直于 image plane。

### 6.3 推导

记两个 primitive：
- **Fixed primitive**：normal $\mathbf{n}^f$，暂时固定
- **Varying primitive**：normal $\mathbf{n}^v$，沿其 normal 移动 $\partial r$

观察：
- Varying primitive 沿其平面内的 translation 不动 edge
- Varying primitive 绕 edge 的 rotation 也不动 edge
- 只有沿 normal 的 translation 会动 edge

公式 (27)：fixed primitive 在 x-z 平面内的单位方向向量
$$\mathbf{b} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \mathbf{n}^f$$

变量含义：
- $\mathbf{b}$：fixed primitive 在 x-z 平面内的切向单位向量（normal 旋转 90°）
- $\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$：90° 逆时针旋转矩阵

公式 (28)：varying primitive 的位移
$$\partial \mathbf{r} = \partial r \, \mathbf{n}^v$$

公式 (29)：edge 位移 $\mathbf{s}$ 在 $\mathbf{n}^v$ 方向的投影等于 $\partial r$
$$\partial \mathbf{r} = (\mathbf{s} \cdot \mathbf{n}^v) \mathbf{n}^v$$

**关键约束**：edge 只能沿 fixed primitive 的切向 $\mathbf{b}$ 移动（因为 fixed primitive 没动，intersection 线必须在 fixed primitive 的平面上）。

公式 (30)：
$$\mathbf{s} = \frac{\mathbf{b}}{\mathbf{b} \cdot \mathbf{n}^v} \partial r$$

公式 (31)：edge 在 image plane 上的位移 = $\mathbf{s}$ 在 x 轴上的投影
$$\frac{\partial p}{\partial r} = \frac{\mathbf{b}}{\mathbf{b} \cdot \mathbf{n}^v} \cdot \mathbf{e}_x$$

代入公式 (27) 得到公式 (12)（**intersection case 的核心结果**）：
$$\frac{\partial p}{\partial r} = -n_z^f \left[\mathbf{n}^{fT} \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \mathbf{n}^v\right]^{-1}$$

变量含义：
- $p$：edge 在 image plane 上的位置（x 坐标）
- $r$：varying fragment 的 3D 位置
- $n_z^f$：fixed primitive normal 的 z 分量
- $\mathbf{n}^f, \mathbf{n}^v$：fixed、varying primitive 的 normal（在 x-z 平面投影后 2D 向量）
- $\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$：90° 旋转矩阵
- 分母 $\mathbf{n}^{fT} \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \mathbf{n}^v$：fixed primitive 切向与 varying primitive normal 的点积，衡量两个 primitive 的"非平行度"

**符号解释**：
- 负号来自旋转矩阵的方向约定
- 当两个 primitive 接近平行时（分母 $\to 0$），gradient 发散——这有物理意义：平行平面相交线在无穷远，small motion 引起 large edge shift
- $n_z^f$ 表示 fixed primitive 有多"正面朝向 camera"，越正面 edge 越敏感

### 6.4 对称处理

paper 对两个 fragment 都做这个计算：先固定 A 变化 B，再固定 B 变化 A，然后用 product rule 求和。这给出两个 fragment 各自的 $\frac{\partial p}{\partial r}$。

---

## 7. Forward Pass 实现：模块化

paper 把实现拆成 5 个 CUDA 模块：

1. **Vertex transform**：把 vertex 从 model space 变到 camera space 再投影到 image plane。Standard AD 处理 backward。
2. **Rasterization**：输出 index image（每 pixel 的 triangle ID）+ depth image。**没有 backward**（这部分由 EdgeGrad 接管）。
3. **Barycentrics**：根据 index image 计算每 pixel 的 barycentric coordinates。Standard AD。
4. **Interpolation**：根据 vertex attributes、barycentrics、index image 插值出 per-pixel attributes。Standard AD。
5. **EdgeGrad**：核心模块。Forward 是 identity（什么也不做），backward 接收 $\frac{\partial L}{\partial I_{i,j}}$ 并计算 vertex position gradient。

**精妙之处**：EdgeGrad 的 forward 是 identity，所以它**完全不修改 rasterized image**。这是与 Nvdifrast 的关键区别——Nvdifrast 的 antialiasing 模块会修改 forward image。

这个设计使得 EdgeGrad 可以应用于：
- RGB image
- Segmentation mask（binary）
- Depth map
- Normal map

而不会因为 smoothing 混淆不同 surface 的属性。

---

## 8. 实验分析

### 8.1 Forward Gradient 比较

Figure 5 比较 6 个方法在 4 个 test scene 上的 forward gradient：
- (a) Scene + parameter arrows
- (b) Finite differences（参考 ground truth）
- (c) Mitsuba 3（256 samples per pixel）
- (d) Redner
- (e) Nvdifrast
- (f) EdgeGrad (本文)

观察：
- Mitsuba 和 Redner 通常准确，但在 **geometry intersection** 处失败
- Nvdifrast noisy，因为它假设 silhouette edge 的 triangle 总覆盖 pixel center（常常不成立）
- **EdgeGrad 是唯一在 intersection case 也能匹配 finite differences 的方法**

### 8.2 Backward Gradient 误差

Table 1 显示三个 scene 的 relative error：
- Scene 1（无 intersection）：所有方法都还行
- Scene 2, 3（有 intersection）：EdgeGrad 显著优于其他

### 8.3 Runtime vs Image Size

Figure 6：
- (a) Runtime vs image size：EdgeGrad 在所有尺寸都比 Mitsuba 和 Nvdifrast 快
- (b) Runtime vs triangle count：EdgeGrad 几乎不随 triangle count 增长（因为只处理 boundary pixel）
- (c) Gradient error vs image size（无 intersection）：EdgeGrad 与 Mitsuba 相当
- (d) Gradient error vs image size（有 intersection）：EdgeGrad 显著更好

**为什么 EdgeGrad 快**：
- 只处理 boundary pixel（数量 $\propto \sqrt{\text{pixel count}}$）
- 不需要 edge sampling
- 不需要 connectivity 数据结构
- Forward pass 完全等价于标准 rasterization

### 8.4 Blender Dataset 重建

Table 2 显示 8 个 scene 的 PSNR/SSIM/LPIPS：

| Method | Lego PSNR | Chair PSNR | Mic PSNR |
|--------|-----------|------------|----------|
| Continuous only | 16.34 | 28.15 | 21.80 |
| Nvdifrast | 29.44 | 29.80 | 29.64 |
| EdgeGrad -intersect | 29.57 | 33.08 | 31.22 |
| EdgeGrad (full) | 29.67 | 32.98 | 31.35 |

观察：
- "Continuous only" baseline 完全失败（说明 discontinuity gradient 必须处理）
- EdgeGrad 比 Nvdifrast 全面提升
- Intersection handling 主要提升 LPIPS（perceptual metric）而非 PSNR——因为 intersection artifact 通常小但对感知影响大

### 8.5 Dynamic Head Reconstruction

这是 paper 最 impress 的应用。用 mesh-based avatar [Ma 2021, Pixel Codec Avatars] 拟合多视角 lightstage 捕获的 facial performance。

挑战：mouth 内部（牙齿、舌头、嘴唇）有：
- 频繁 self-intersection
- 接触、滑动
- 强 occlusion
- poor observability

之前方法需要：
- 艺术家手动 alignment
- 物理仿真防止 self-intersection [Ichim 2017]
- 专门的 mouth内部 capture [Wu 2016, Medina 2022]

paper 证明：用 EdgeGrad + 2D segmentation mask 监督，可以**全自动**拟合复杂 mouth 内部几何。

实现细节：
- 基础 avatar：encoder-decoder VAE，输出 mesh displacement + neural texture
- 加 segmentation texture（one-hot 编码 teeth/tongue/lips labels）
- 用 HRNet [Wang 2020] 从 camera image 预测 2D segmentation mask 作为监督
- L2 loss between rendered segmentation 和 predicted segmentation

Figure 9, 10 显示：能正确重建 tongue 滑过 teeth、teeth-teeth contact、tongue 推 lips 等复杂接触几何。

---

## 9. 局限性

paper 自己列出：
- **无法处理 transparency**：rasterization 固有限制
- **缺乏精确 antialiasing**：没有 multisampling，forward image 是 aliased 的
- **没有 physically-based rendering**：只处理 visibility，不处理光照/材质
- **没有 global illumination**
- **Sub-pixel triangle 问题**：finely tessellated mesh 中 adjacent primitive 检测可能失效

---

## 10. 个人评论：这个工作的优美之处

### 10.1 数学上的优雅

公式 (11) 的形式极其简洁：
$$\frac{\partial L}{\partial p_{AB}} \propto \langle \text{avg grad} \rangle \cdot \langle \text{intensity jump} \rangle$$

这让人想起 finite difference 和 image gradient 的本质——edge 的 gradient 就是 intensity 差乘以 loss sensitivity。paper 用一套严谨的推导（Heaviside、Dirac delta、divergence theorem）把这个直觉 grounding 在数学上。

### 10.2 工程上的精妙

**Forward pass 是 identity** 这个设计太聪明了：
- 不修改 forward image
- 可以应用于任何 rasterized output（RGB、mask、depth、normal）
- 模块化，可以替换 Nvdifrast 的 antialiasing 模块

### 10.3 与 prior art 的关系

paper 把 prior art 统一在一个 framework 下：
- Nvdifrast [Laine 2020]：用 analytic antialiasing 修改 forward
- Soft Rasterizer [Liu 2019]：用 fuzzy edge 修改 forward
- Li et al. 2018：用 edge sampling 但昂贵
- Loper et al. 2014 (OpenDR)：用 Sobel filter 检测 discontinuity

EdgeGrad 的位置：**不修改 forward，但用 micro-edge 近似使得 backward gradient 计算简化为 local pixel-pair 操作**。

### 10.4 与 3D Gaussian Splatting 的对比

3DGS [Kerbl 2023] 用 alpha blending，本质是 differentiable 的（每 pixel 是 sorted splat 的加权平均），没有 visibility discontinuity 问题。但 3DGS 不能直接表达 mesh topology，不适合需要 mesh 的应用（animation、physics、registration）。

EdgeGrad 保留了 mesh 表达，同时解决了 differentiability 问题——这是不同的 design point。

### 10.5 可能的扩展

- **Multi-sample antialiasing (MSAA)**：如果 forward 用 MSAA，每个 sub-pixel sample 有自己的 triangle ID。EdgeGrad 可以扩展到 sub-pixel level，可能提升小 triangle 的处理。
- **Transparent geometry**：需要扩展到 order-independent transparency，可能用 weighted blended OIT + EdgeGrad。
- **Animation parameters**：除了 vertex position，还可以微分 blend shapes、skeleton joints。
- **Neural rendering**：可以作为 differentiable rasterization backbone 接入 NeRF-like pipeline，做 hybrid mesh + neural texture。

### 10.6 与你的工作的潜在联系

Karpathy，你在 nanoGPT、micrograd 等工作中强调过理解 backprop 的本质。这篇 paper 本质上是在做"**手工 backprop**"——对于 standard AD 无法处理的 discontinuity，手工推导 gradient 公式并实现为 custom backward。

这种"识别 AD 的边界、手工补丁"的思路，和 micrograd 中你展示的 reverse-mode AD 的本质一脉相承。AD 不是魔法，它依赖于计算图的 local differentiability；当 local differentiability 失效（discontinuity、discrete choice、stochastic sample），需要手工注入 gradient。

EdgeGrad 的 micro-edge trick 类似于 **reparameterization trick** (VAE 中用 $\epsilon \sim \mathcal{N}(0,1)$, $z = \mu + \sigma \epsilon$ 把 stochastic gradient 变 deterministic)：通过引入一个虚构的连续过程（micro-edge），把不可微的离散操作"包装"成可微的。

---

## 11. 总结

这篇 paper 的核心贡献：

1. **Micro-edge 构造**：把任意方向 edge 近似为水平/垂直 micro-edges，位于像素之间。Forward image 不变。
2. **核心公式 (11)**：$\frac{\partial L}{\partial p_{AB}} = \frac{1}{2}(\frac{\partial L}{\partial I_A} + \frac{\partial L}{\partial I_B})(I_A - I_B) + \Omega$，简洁的 local pixel-pair gradient。
3. **Edge classification**：四种类型（no/adjacent/overhanging/intersecting），简单测试分类。
4. **Geometry intersection handling**：公式 (12) 给出 intersection edge 的 $\frac{\partial p}{\partial r}$，是首个能处理 self-intersection 的 rasterization-based differentiable renderer。
5. **应用**：dynamic head reconstruction 中 mouth 内部几何拟合，复杂接触场景。

希望这个讲解 build up 了你的 intuition！如果想深入某个部分（比如 intersection 推导的具体几何图，或者 EdgeGrad module 的 CUDA 实现细节），可以继续聊。
