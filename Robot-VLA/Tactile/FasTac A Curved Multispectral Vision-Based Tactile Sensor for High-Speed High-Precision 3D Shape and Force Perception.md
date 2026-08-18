---
source_pdf: FasTac A Curved Multispectral Vision-Based Tactile Sensor for High-Speed
  High-Precision 3D Shape and Force Perception.pdf
paper_sha256: 2d440ccef1406c2e3a81d800365c6e3dff293dfa3e6313f9b8f3336c74886e0e
processed_at: '2026-08-18T12:48:53-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话拆解 FasTac

## I. 它到底想解决什么问题？

想象你闭着眼睛用手抓东西。你的手指肚是弯的，当你碰到一个草莓或者滑动的杯子时，你脑子立刻知道三件事：草莓表面有个小坑（**精细 3D 形状**）、杯子往下掉产生了摩擦力（**切向与法向力区分**）、杯子在手里快速震了一下（**瞬态信号**）。

FasTac 就是一个想给机器人仿造这种“曲面指尖肚”的传感器。以前的 tactile sensor 通常是平的，哪怕做成弯的，也会遇到三个让人头疼的大麻烦：

1. **灯照不匀**：弯曲的 gel 表面会让某些地方有阴影，传统的纯 RGB photometric stereo 在这些地方直接失效，算出来的 3D 形状会飘。
2. **肉厚不均**：曲面 gel 在不同位置的厚度、拉伸程度都不一样。同样的力，按在指尖正中央跟按在侧面，变形规律完全不同。用传统的固定公式（fixed CNN kernel）去算力，误差极大。
3. **反应太慢**：如果摄像头拍了照，传给旁边的电脑（CPU/GPU）去算，等算出来抓取指令，杯子早掉地上了，更别提抓百赫兹的微小震动了。

## II. 它的三板斧是怎么破局的？

FasTac 之所以牛，是因为它针对这三个痛点，搞了三套对应的工程 trick，并且揉进了一个极小的指尖壳里。这就是典型的 system-level co-design（系统级协同设计）。

### 第一板斧：加一个红外眼 (RGB-NIR Multispectral)
为了解决曲面灯光照不匀的问题，常规思路是加更多摄像头或者搞个三棱镜分光，但那样指尖就做不小了。
FasTac 用了 OmniVision 出的一种特殊 sensor（OV2736），它的像素阵列是 RGB 和 NIR 交错的。一次拍照，RGB 和 NIR 全有了，而且像素级天然对齐，不需要配准。
**直觉**：这就好比本来只有红绿蓝三盏灯打光，遇到曲面死角这三盏灯的几何约束不够用了（数学上叫 rank-deficient）。加了 NIR 这第四盏灯，多了一个独立观察通道，哪怕有一盏灯被曲面挡住，剩下三盏灯依然能把 3D 法向量算出来，鲁棒性直接拉满。

### 第二板斧：带图纸算 Depth (Boundary-Prior Poisson)
算出法向量后，得把它们拼成完整的 depth map（深度图）。传统做法假设传感器边缘是平的（depth 为 0），这在平面的 GelSight 上没问题，但在弯指尖上，边缘本来就有高度！强行当 0 算的话，误差会从边缘一路累积到中心，整个图就彻底飘了。
FasTac 的做法很直接：把传感器出厂的 CAD 模型拿出来，把边缘的真实高度当作“已知答案”（Boundary Prior）塞进 Poisson 方程的右边。这就好比修路前先把路基的边缘标高定死，中间怎么填土都不会跑偏。这一招把 depth 的 MAE 从 0.2730 mm 直接干到 0.0415 mm。

### 第三板斧：见风使舵的卷积 (HyperForce Dynamic Convolution)
算力的时候，因为曲面指尖各处的“软硬程度”不同，固定不变的 CNN kernel 根本行不通。
FasTac 搞了个叫 HyperForce 的网络，参考了 HyperNetworks 的思想。它不直接学力，而是学“怎么根据像素位置生成对应的 convolution kernel”。
**直觉**：就像 FEM（有限元分析）里把弹性体切成网格，每个网格的刚度矩阵都不一样。HyperForce 相当于让网络明白：“哦，这个像素在指尖侧面，厚度薄边界紧，我给它配一套硬一点的 kernel；那个像素在正中间，肉厚软乎，我给它配一套软一点的 kernel”。而且训练完之后，这些位置相关的 kernel 直接固化成 ROM 查表，极大降低了计算量。

### 第四板斧：在指尖里长个“小脑” (FPGA Edge Deployment)
这是最硬核的一步。图像数据哪怕传给离得最近的 GPU，经过 MIPI 传输、OS 调度、显存读写，端到端延迟也有 3.26ms，而且有抖动。
FasTac 直接在指尖背面挂了一块 Xilinx Zynq UltraScale+ FPGA。FPGA 的好处是纯流水线、确定性极强。像素一个个流进去，算 normal、算 Poisson、算 Force，像流水线工厂一样直通到底，根本不用等攒齐一整张图。
**结果**：延迟干到 1.09ms。测 100Hz 的振动，CPU 和 GPU 全因为采样率不够发生 aliasing（频谱混叠）了，只有 FPGA 稳稳抓出了正确的频谱。这就相当于在手指肚里直接长了一段脊髓反射神经，不过大脑直接出反应。

## III. 总结一下 Intuition

FasTac 的核心逻辑就是“顺着物理规律走”。
曲面光照不均，就加 NIR 补足秩；
曲面边缘不平，就老老实实把 CAD 图纸塞进 Poisson 积分当 boundary prior；
曲面各处刚度不同，就抛弃 fixed CNN，用 dynamic kernel 去做 spatially adaptive 映射；
端到端延迟大，就用 FPGA 流水线把数据按死在本地算。
这种每一个 trick 都精准对应一个物理痛点的设计，才是 robotics 最漂亮的地方。

## IV. Reference Links

- [OmniVision OV2736 RGB-IR Sensor 介绍](https://www.omnivision.com/products/sensor/ov2736)
- [HyperNetworks 论文](https://arxiv.org/abs/1609.09106)
- [Fast Poisson Solver 原理参考](https://www.cs.cmu.edu/~kmcrane/Projects/PoissonStressTest/paper.pdf)
- [Xilinx Zynq UltraScale+ MPSoC 产品页](https://www.xilinx.com/products/silicon-devices/soc/zynq-ultrascale-mpsoc.html)
- [GelSight 原始论文](https://www.mdpi.com/1424-8220/17/12/2762)
- [DenseTact 2.0 论文](https://ieeexplore.ieee.org/document/10161336)

---

# FasTac Paper 深度解析

## I. Overall Positioning 与 Motivation

FasTac 要解决的核心 problem：**curved fingertip** 上同时做到 high-precision 3D shape reconstruction + three-axis force estimation + low-latency processing。这是 dexterous manipulation 在真实场景下的三重需求叠加 —— fine geometry（小凸起/边缘）、normal vs shear load 的 decoupling、transient signal 的 capture。

Curved geometry 带来的 fundamental difficulty 有三层：
1. **Photometric stereo 的 rank-deficient 问题**：RGB 三个光源在 compact curved fingertip 内部无法均匀覆盖整面，会出现局部 shadow 或某一通道亮度被 distance attenuation 吃掉，导致 $\mathbf{L}_{obs}$ 的秩不足 3，surface normal 退化。
2. **Curved elastomer 的 mechanical nonuniformity**：同一 displacement 在 fingertip 顶部（薄、boundary-free）和侧边（厚、boundary-constrained、curvature 大）对应的 force 完全不同，translation-invariant 的 CNN kernel 无法建模这种 spatially varying stiffness。
3. **Latency vs bandwidth 的 tradeoff**：high-resolution tactile image stream 给到 host CPU/GPU 跑的 image-to-force pipeline，会经过 MIPI transfer + USB buffer + OS scheduling + kernel launch，end-to-end latency 和 jitter 都很大，无法稳定 capture 100Hz vibration。

FasTac 的三个 contributions 精准对应这三层 difficulty：RGB-NIR multispectral imaging（解决 rank-deficient）、HyperForce position-aware dynamic convolution（解决 mechanical nonuniformity）、FPGA edge deployment（解决 latency）。

参考 [GelSight 原始 paper](https://ieeexplore.ieee.org/document/8216565)、[GelSight360](https://ieeexplore.ieee.org/document/10102858)、[DenseTact 2.0](https://ieeexplore.ieee.org/document/10161336) 可以看到 prior work 在这三个方向各自的局限。

---

## II. Sensor Hardware 设计细节

### A. Mechanical Layout

FasTac 的结构（Fig. 2）分三层：
- **Light-guiding fingertip skeleton**（photosensitive resin 3D-printed）：housing camera + 提供 diffuse backlight scattering。关键设计 trick：把 LED 放在 skeleton 后方，让 discrete LED point source 通过 resin 的 translucency 转成 soft diffuse field，这避免了 specular highlight 同时增加 illumination uniformity。这跟 [GelSight Mini](https://www.frontiersin.org/articles/10.3389/frobt.2022.899230/full) 用 photodiode + 内壁 paint 的方案思路相反 —— FasTac 用 material 本身的 subsurface scattering 做 diffuser。
- **Multispectral LED FPC**（flexible printed circuit）：R/G/B/NIR 四色 LED 在 FPC 四角，其中 R/G 垂直于 B/NIR，maximally increase angular diversity of illumination vectors。这个设计直接服务 photometric stereo 的 numerical stability：四个 $\mathbf{l}$ vectors 要尽可能 span R^3，正交排列保证 rank=3 即使有一个 channel 失效。
- **Gel skin**：Smooth-On Solaris silicone + Slacker softener（1:1:3 重量比），高 Slacker 比例给出 high compliance，sensitively capture 微小 contact deformation。表面涂银粉 reflective paint（800 mesh + silicone solvent + thinner），之后 laser ablation 标记 marker dots 并用 black silicone ink 填充。

### B. RGB-NIR Single-Sensor Imaging 的关键 insight

这里用 **OmniVision OV2736 RGB-IR CMOS sensor**，4×4 RGB-IR color filter array（CFA）。这个设计的核心 motivation：用 single optical path 同时获取 RGB 和 NIR，避免了 [GelSplitter3D](https://ieeexplore.ieee.org/document/10629487) 用 beam-splitter 多 camera 方案带来的 spatial registration 问题。同一个 pixel 的 R/G/B/NIR 信号来自同一 optical center，intrinsic pixel-level alignment，zero parallax。

参考 [RGB-IR CFA 的原理](https://www.omnivision.com/products/sensor/ov2736) 和 [NIR 在 tactile 中的应用](https://link.springer.com/chapter/10.1007/978-3-031-43987-6_2)，NIR 对 silicone elastomer 的穿透深度比 visible 深，对 surface inclination 的响应在 high-curvature 区域更线性，因此是 curved surface 上第四个独立 observation channel 的天然候选。

---

## III. Multispectral Photometric Stereo 的 3D Reconstruction

### A. 为什么需要 four-source photometric stereo

Standard photometric stereo 的 image formation model：
$$I = \rho \mathbf{l}^T \mathbf{n}$$

其中 $\rho$ 是 albedo，$\mathbf{l} \in \mathbb{R}^3$ 是 illumination direction（单位向量），$\mathbf{n} \in \mathbb{R}^3$ 是 surface normal（单位向量），$I$ 是观测 intensity。

对三个光源 R/G/B，stacked 系统 $\mathbf{L}_{obs} = [\mathbf{l}_R, \mathbf{l}_G, \mathbf{l}_B]^T \in \mathbb{R}^{3\times3}$，当 $\text{rank}(\mathbf{L}_{obs}) = 3$ 时 surface normal 可解：
$$\mathbf{n} = \frac{1}{\rho} \mathbf{L}_{obs}^{-1} \mathbf{I}$$

Curved fingertip 的 problem：在 compact volume 内，某些 surface region 离某个 LED 太远、被自身遮挡、或 illumination vector 跟其他 source 接近 collinear，会让 $\mathbf{L}_{obs}$ 在该 pixel 上 rank-deficient，surface normal 欠定。

FasTac 引入 NIR 作为第四 channel，构造 over-determined system：
$$\mathbf{I} = \rho \mathbf{L} \mathbf{n}, \quad \mathbf{L} = [\mathbf{l}_R, \mathbf{l}_G, \mathbf{l}_B, \mathbf{l}_{NIR}]^T \in \mathbb{R}^{4\times3}$$

Over-determined system 用 least-squares 解，即使一个 channel occluded/degenerate，剩余三个仍能 span $\mathbb{R}^3$。这是 robustness 的核心来源。Intuition：把 photometric stereo 的 redundancy 从 0（rank=3 刚好可解）提到 1，相当于在 curved geometry 上加保险。

### B. Spectral Demultiplexing 的 preprocessing pipeline

4×4 RGB-IR CFA 把同一 sensor 上不同 pixel 分配给 R/G/B/IR。phase 定义：
$$\phi(u,v) := (v \bmod 4, u \bmod 4)$$

$(u,v)$ 是 pixel coordinate，$\bmod 4$ 把像素分到 16 类 phase，对应 4×4 CFA pattern 的 16 个位置。

NIR channel 采样：取 odd row + odd column 的 IR pixel，记作 $I_{NIR}^s(m,n) = I_{blc}(2m+1, 2n+1)$，然后 upsample 成 dense image $I_{NIR}$。

IR crosstalk subtraction 的公式（Eq. 1）：
$$\tilde{I}(u,v) = \text{clip}\left(I_{blc}(u,v) - \alpha \min\{I_{NIR}(u,v), I_{max}/\beta\}, 0, I_{max}\right)$$

变量解释：
- $I_{blc}(u,v)$: black-level-corrected raw image at pixel $(u,v)$。Black level 是 sensor 在无光时的暗电流 offset，必须先减掉否则 NIR subtraction 会被 baseline 污染。
- $I_{NIR}(u,v)$: upsampled 后的 NIR intensity estimate at 该 pixel。
- $\alpha$: IR subtraction strength，控制减多少 NIR。α 越大 visible channel 越 pure 但 risk over-subtraction。
- $\beta$: 防止 overcorrection，把 NIR estimate 上限钳到 $I_{max}/\beta$。Intuition：如果 NIR 估计本身被 noise 或 specular 推到很高，直接减会把 visible 信号也干掉，所以用 min 操作设上限。
- $I_{max}$: RAW10 saturation value（10-bit = 1023）。
- $\text{clip}(\cdot, 0, I_{max})$: 截断到 valid range。
- $\tilde{I}(u,v)$: IR-corrected visible-channel intensity。

这个减法本质上是在做 **RGB-IR CFA 的 crosstalk compensation**：visible pixel 的 R/G/B filter 会让一部分 NIR 漏过去（特别是 R channel，red filter 对 NIR 几乎透明），所以 visible reading 里混了 NIR contamination。用 dedicated IR pixel 读到的 NIR 作为 estimate 减掉，恢复了 visible channel 的真实 visible-only intensity。这是 RGB-IR sensor 的 standard processing，参考 [OmniVision RGB-IR whitepaper](https://www.omnivision.com/technologies/rgb-ir)。

### C. Position-Aware MLP for Surface Normal Estimation

关键 insight：在 compact curved fingertip 内部，LED 是 near-field，illumination vector $\mathbf{L}(p)$ 是 pixel position $p=(u,v)$ 的函数，不再 global constant。Image formation 变成：
$$I_{u,v} = \mathcal{R}(\mathbf{n}_{u,v}, \mathbf{L}(p), \rho)$$

这里 $\mathcal{R}(\cdot)$ 是 local image-formation function（包含 distance attenuation、incident angle、subsurface scattering 等）。

如果 MLP 只拿 4-channel intensity 作 input，无法区分两种 case：
- (a) pixel 远离光源 → intensity 低
- (b) pixel 表面倾斜背光 → intensity 低

这两种 case 几何上完全不同但 intensity 表现一样。FasTac 把 normalized 坐标 $p_{u,v}$ 作为 positional encoding concatenate 进 input：
$$\hat{\mathbf{n}}_{u,v} \approx \mathcal{F}(\mathbf{I}_{u,v} \oplus \mathbf{p}_{u,v})$$

这里 $\oplus$ 是 feature concatenation，input dim = 4 (spectral) + 2 (position) = 6，output 是 $\hat{\mathbf{n}} = [n_x, n_y, n_z]$。MLP 是 pixel-wise 的，no neighborhood convolution —— 因为这个 mapping 由 single-point intensity + position 决定，不需要 texture context。

Loss（Eq. 2）：
$$\mathcal{L}_n = \frac{1}{|\Omega|} \sum_{p \in \Omega} \sum_{\alpha \in \{x,y,z\}} |\hat{n}_\alpha(p) - n_\alpha^*(p)|$$

变量解释：
- $\Omega$: valid pixel domain
- $\alpha$: surface normal component index ∈ {x, y, z}
- $\hat{n}_\alpha(p)$: predicted unit-normalized surface normal 的 α 分量
- $n_\alpha^*(p)$: ground-truth unit-normalized surface normal 的 α 分量

Component-wise L1 loss（对 x/y/z 三分量等权求和）。预测和真值都先 unit-normalize 到 length=1，因为 normal 的方向信息比 magnitude 重要。L1 比 L2 在 outlier 上更 robust，并且 component-wise 而不是 cos-angle loss 让 optimization landscape 更 smooth。

这个设计的 intuition 跟 [NeRF 的 positional encoding](https://arxiv.org/abs/2003.08934) 思路一致 —— high-frequency spatial variation 需要 explicit position 信息才能由 MLP 拟合。这里 position encoding 不是 high-frequency Fourier feature 而是 normalized coordinate，因为 illumination field 在 cm-scale 上 spatial smooth，不需要捕捉高频。

### D. Boundary-Prior Fast Poisson Depth Reconstruction

Photometric stereo 给的是 surface normal $\mathbf{n} = [n_x, n_y, n_z]$，要转成 depth map $D(u,v)$ 需要 integration。Standard relation：
$$p = -\frac{n_x}{n_z}, \quad q = -\frac{n_y}{n_z}$$

是 surface 的 gradient（这里 sign convention 是 z 轴向内为正）。Poisson reconstruction 解：
$$\nabla^2 D = \frac{\partial p}{\partial u} + \frac{\partial q}{\partial v}$$

Discretized Poisson equation 在 2D grid 上：
$$D_{i+1,j} + D_{i-1,j} + D_{i,j+1} + D_{i,j-1} - 4D_{i,j} = f_{i,j}$$

这里 $f_{i,j}$ 是 source term（discrete divergence of gradient field）。

Standard fast Poisson solver 用 DST-I（Type-I Discrete Sine Transform）默认 zero Dirichlet boundary condition，即边界 $D=0$。对 planar GelSight 传感器，sensor frame 边界确实 depth 接近 0，所以 zero boundary 合理。

Curved fingertip 的 problem：boundary 的真实 depth 是 non-zero 的，对应 fingertip 的 CAD 几何 shape。如果强行 zero boundary，integration 会从边界开始累积 drift，导致内部 depth 全部偏移。

FasTac 引入 boundary prior $D_{prior}$（从 CAD model 渲染），modified source term（Eq. 3）：
$$\tilde{f}_{i,j} = f_{i,j} - \sum_{(m,n) \in \mathcal{N}(i,j) \cap \partial\Omega} D_{prior}(m,n)$$

变量解释：
- $f_{i,j}$: original Poisson source term at pixel $(i,j)$，即 discrete divergence
- $\tilde{f}_{i,j}$: boundary-corrected source term
- $\mathcal{N}(i,j)$: four-connected neighborhood of pixel $(i,j)$
- $\partial\Omega$: boundary of valid reconstruction domain
- $D_{prior}(m,n)$: known boundary depth from CAD
- 求和项：把所有"邻居是 boundary pixel"的位置上的 known depth 移到 RHS

数学上等价于：把已知 boundary value 移到 linear system 的 RHS，interior unknowns 仍满足 zero Dirichlet after 平移。这样 DST-I solver 仍能用，但解出来的是相对 boundary 的 interior depth，避免了 drift。

这个 trick 在 numerical PDE 里叫 **lifting** —— 用 known boundary data 修正 source，然后 homogeneous boundary 求解。参考 [Fast Poisson Solver with boundary conditions](https://www.cs.cmu.edu/~kmcrane/Projects/PoissonStressTest/paper.pdf)。

从 Table III 看到 boundary prior 的巨大 impact：no prior 时 MAE ~0.27mm，有 prior 时 MAE ~0.04mm，差 6.5 倍。这是 paper 里最重要的 ablation 之一。

---

## IV. HyperForce: Position-Aware Dynamic Convolution for Force Estimation

这是 paper 最创新的部分。理解它要先看 FEM-based vision-based tactile force estimation 的背景。

### A. FEM-based Force Reconstruction 的 physics background

参考 [Dense Tactile Force Estimation with GelSlim and inverse FEM](https://ieeexplore.ieee.org/document/8793619) 和 [iFEM2.0](https://ieeexplore.ieee.org/document/10610368)：把 elastomer discretize 成 finite element mesh，每个 node 有 displacement，整个 mesh 的 global displacement vector $\mathbf{U}$ 和 global nodal force vector $\mathbf{F}$ 满足：
$$\mathbf{F} = \mathbf{K}\mathbf{U}$$

这里 $\mathbf{K}$ 是 global stiffness matrix（由 material Young's modulus、Poisson ratio、element geometry 组装出来）。$\mathbf{K}$ 是 sparse block matrix —— 每个 node 的 force 只由其 neighborhood 内的 displacement 决定，远处 node 几乎无 contribution。

### B. FEM-as-Convolution 的 approximation

Eq. (7) 是把 FEM 的 local multiplication 写成 convolution-like 形式：
$$\mathbf{f}(p) \approx \sum_{q \in \mathcal{N}} \mathbf{G}_{p,q} \mathbf{u}(p+q)$$

变量解释：
- $p$: current pixel / node
- $q$: relative offset in neighborhood $\mathcal{N}$
- $\mathbf{u}(p+q) = [d_x, d_y, d_z]^T$: local 3D displacement at neighbor pixel
- $\mathbf{f}(p) = [f_x, f_y, f_z]^T$: local 3D force response at pixel $p$
- $\mathbf{G}_{p,q}$: local stiffness-like coupling weight，4D tensor（pixel × offset × 3 input × 3 output）

如果 elastomer 是 homogeneous + planar + regular mesh，$\mathbf{G}_{p,q}$ 退化成 $\mathbf{G}_q$（translation invariant），这就是 standard CNN 的 shared kernel。但 FasTac 是 curved fingertip，local thickness / curvature / boundary constraint / normal direction 都随 $p$ 变，所以 $\mathbf{G}$ 必须依赖 $p$。

### C. Tangential Displacement 提取（Marker-based）

Sparse marker displacement $\mathbf{d}_k$ 用 thresholding + blob detection 提取 M 个 marker centers，之后用 RBF interpolate 成 dense tangential field（Eq. 4）：
$$\mathbf{u}_{tan}(p) = \sum_{k=1}^M \mathbf{w}_k \phi(|p - c_k|), \quad \phi(r) = \frac{1}{\sqrt{r^2 + \varepsilon^2}}$$

变量解释：
- $c_k$: detected marker center (k-th marker)
- $\mathbf{w}_k$: RBF weight vector，由 sparse marker constraints solve 出来
- $r = |p - c_k|$: pixel-domain Euclidean distance from interpolation point $p$ to marker $c_k$
- $\varepsilon$: kernel shape parameter，控制 RBF 的 sharpness。ε 大 → 平滑、远距离影响；ε 小 → sharp、近距离主导。
- $\phi(r) = 1/\sqrt{r^2 + \varepsilon^2}$: inverse multiquadric (IMQ) kernel

为什么选 IMQ kernel？因为半无限弹性体受 point load 时的 surface displacement field 在 Boussinesq 解里大致是 $1/r$ 量级（vertical point load 作用下 surface 垂直位移 $\propto 1/r$），IMQ 在 $r \gg \varepsilon$ 时 $\phi \to 1/r$，跟物理响应 decay 形式一致。所以 RBF interpolation 在物理上 well-motivated，参考 [Boussinesq solution](https://en.wikipedia.org/wiki/Boussinesq%27s_problem)。

Normal displacement $d_z$ 由 depth map 差值得到（Eq. 5）：
$$d_z(p) = D_0(p) - D_t(p)$$

$D_0(p)$ 是 background depth（无接触时的 reference depth map），$D_t(p)$ 是 contact 时的 current reconstructed depth。两者都来自 boundary-prior Poisson reconstruction。

### D. HyperForce 的 dynamic convolution 架构

HyperForce 用 **hypernetwork**（参考 [Ha et al., HyperNetworks](https://arxiv.org/abs/1609.09106)）生成 position-dependent kernel。

对每个 pixel $p$，先取 $K \times K$ 的 displacement patch（K 是 kernel side length），vectorize 成 $\mathbf{v}(p) \in \mathbb{R}^{3K^2}$。3 是 displacement 通道数，$K^2$ 是 patch 内 pixel 数。

Hypernetwork 输入是 normalized 3D coordinate encoding $c(p)$（应该包含 pixel 的 normalized $(u,v)$ + 可能的 CAD-derived local normal/curvature），输出是三个 dynamic kernel（对应 $F_x, F_y, F_z$ 三个 force component）（Eq. 8）：
$$\mathbf{w}_\alpha(p) = \mathcal{H}_\alpha(c(p)), \quad \mathbf{w}_\alpha(p) \in \mathbb{R}^{3K^2}, \quad \alpha \in \{x,y,z\}$$

变量解释：
- $\mathcal{H}_\alpha$: hypernetwork（用 $1\times1$ conv 实现的 lightweight network）for force component α
- $c(p)$: normalized 3D coordinate encoding of pixel $p$
- $\mathbf{w}_\alpha(p)$: position-dependent dynamic kernel，size 跟 patch vector 一致 ($3K^2$)
- $\alpha$: force component index ∈ {x, y, z}

Pixel-wise force component（Eq. 9）：
$$\hat{f}_\alpha(p) = \mathbf{w}_\alpha(p)^T \mathbf{v}(p)$$

这就是 inner product between dynamic kernel 和 patch vector。本质上是一个 position-conditioned linear layer。

Resultant three-axis force（Eq. 10）：
$$\hat{\mathbf{F}} = \sum_{p \in \Omega} \hat{\mathbf{f}}(p)$$

把所有 pixel 的 force response 求和成 total force。注意 supervision 是 global resultant force（一个三维向量 per sample），不是 dense force map，因为 ground truth 是 ATI Nano17 给的 single resultant wrench，不是 dense distributed force。

Loss（Eq. 11）：
$$\mathcal{L}_F = \sum_{\alpha \in \{x,y,z\}} |\hat{F}_\alpha - F_\alpha^*|$$

等权 L1 loss on three-axis force component。$\hat{F}_\alpha$ 是 predicted resultant，$F_\alpha^*$ 是 ATI reference。

### E. Ablation 数据的深度解读

Table IV 的 ablation：

| Kernel | Input | $F_x$ MAE | $F_y$ MAE | $F_z$ MAE |
|---|---|---|---|---|
| Dynamic | $d_x+d_y$ | 0.0386 | 0.0279 | 0.1570 |
| Dynamic | $d_z$ | 0.0235 | 0.0325 | 0.0589 |
| Dynamic | $d_x+d_y+d_z$ | **0.0235** | **0.0246** | **0.0545** |
| Fixed | $d_x+d_y$ | 0.0545 | 0.0318 | 0.5123 |
| Fixed | $d_z$ | 0.1028 | 0.0637 | 0.4980 |
| Fixed | $d_x+d_y+d_z$ | 0.0516 | 0.0344 | 0.4905 |

关键 insights：

1. **Dynamic vs Fixed kernel 的 $F_z$ 差异巨大**（0.0545 vs 0.4905 N）：fixed kernel 的 $F_z$ MAE 是 dynamic 的 9 倍。这说明 curved elastomer 的 spatially varying stiffness 严重 break translation invariance，fixed CNN kernel 完全无法 capture position-dependent mechanical response。

2. **Normal force estimation 单看 $d_z$ 比 $d_x+d_y$ 好**（dynamic 下 0.0589 vs 0.1570 N）：这符合直觉 —— normal force 主要由 normal indentation 驱动，tangential marker motion 是次要 cue。

3. **Full 3D displacement 最佳**：$d_x+d_y+d_z$ 配 dynamic kernel 给出全 best。说明 tangential marker displacement 提供 complementary information，特别是 boundary 附近 tangential displacement encode 了 Poisson effect（material 在受 normal load 时 lateral expansion 受 boundary 约束产生 tangential flow）。

Fixed kernel 的 $F_z$ 在所有 input setting 下都 ~0.5 N MAE，说明 curved geometry 下 fixed kernel 几乎无法学到 useful mapping。这跟 [DenseTact 2.0](https://ieeexplore.ieee.org/document/10161336) 用 DenseNet 全图 shared kernel 的方案对比鲜明 —— 在 planar sensor 上 shared kernel 够用，curved 必须用 position-aware 方法。

---

## V. FPGA Edge Deployment

### A. 为什么 FPGA 比 CPU/GPU 适合 tactile

Tactile feedback 的 latency budget 在 dexterous manipulation 中极紧 —— 典型 contact event 持续 10-100ms，要在 1-5ms 内完成 sensing → processing → motor command。CPU/GPU 的问题：
- **Image transfer overhead**：tactile image 从 sensor 经过 MIPI/USB 到 host 的 transfer latency 几 ms
- **OS scheduling jitter**：CPU 上其他 process 抢占，导致 1-10ms jitter
- **GPU kernel launch overhead**：CUDA kernel launch ~10-50μs 但 memory copy 和 sync 累积起来 1-3ms
- **Power consumption**：GPU 几瓦到几十瓦，对 multi-finger hand 不实用

FPGA 的优势：
- **Deterministic timing**：fully pipelined streaming，每 pixel 处理 cycle 数固定
- **In-sensor computation**：image stream 不出 board，省 transfer
- **Low power**：mW 级
- **High frame rate**：可以用更小 readout window 提升 camera fps

参考 [Hundhausen et al.](https://ieeexplore.ieee.org/document/9636861) 之前已经在 in-hand FPGA 上做 CNN，但他们没做 full tactile pipeline。

### B. Heterogeneous PS + PL Architecture

FasTac 用 **Xilinx Zynq UltraScale+ MPSoC (XCZU19EG)**，分 Processing System (PS) 和 Programmable Logic (PL)：
- PS (ARM Cortex): frame-level orchestration, image sync, background metadata, AXI-Lite control, result readout
- PL (FPGA fabric): latency-critical streaming datapath, 三个 computation module 之间用 internal FIFO

### C. Surface Normal MLP 的 quantization

MLP 在 FPGA 上要 quantize 到定点。这里用 16×16 MVM tile（16 input lanes × 16 output neurons，concurrent dot products）。Activation quantization（Eq. 12）：
$$q_i^{(l+1)} = \text{clip}_{[0,127]} \left( \left\lfloor \frac{M_l}{2^{r_l}} \max(0, a_i^{(l)}) \right\rfloor \right), \quad l = 0,1,2$$

变量解释：
- $q_i^{(l+1)}$: neuron $i$ 在 layer $l+1$ 的 quantized activation
- $a_i^{(l)}$: neuron $i$ 在 layer $l$ 的 biased MVM output（pre-activation）
- $M_l$: integer multiplier（fixed-point scale，用整数乘法近似浮点 scale）
- $r_l$: right shift amount（相当于除以 $2^{r_l}$，用 bit-shift 高效实现）
- $\lfloor \cdot \rfloor$: floor operation（round to integer）
- $\text{clip}_{[0,127]}(\cdot)$: saturate to unsigned 7-bit range [0, 127]
- $\max(0, \cdot)$: ReLU activation

整个表达式等价于：fixed-point ReLU + requantization to 7-bit unsigned。FC3 是 output layer，只加 bias 不做 ReLU / requantization，因为 output 要送进 gradient ratio computation（$p = -\hat{n}_x/\hat{n}_z$, $q = -\hat{n}_y/\hat{n}_z$），需要 wider accumulator 保留精度。

Streaming pixel-wise MLP：不需要等 full frame，pixel 到就 compute，pipeline throughput 匹配 camera pixel rate。这是 FPGA 相对 GPU 的核心优势 —— GPU 的 batch-oriented execution 模式不适合 single-pixel streaming。

### D. Boundary-Prior Poisson Solver on FPGA

RHS generation（Eq. 13）：
$$b_{i,j} = (p_{i,j} - p_{i,j-1}) + (q_{i,j} - q_{i-1,j})$$

变量解释：
- $b_{i,j}$: RHS value at inner pixel $(i,j)$
- $p_{i,j}$: x-gradient at pixel $(i,j)$（即 $-\hat{n}_x/\hat{n}_z$）
- $q_{i,j}$: y-gradient at pixel $(i,j)$
- Differences: discrete divergence of the gradient field

Boundary-corrected 版本减去 boundary prior，跟 Eq. 3 一致。

DST-I Poisson solver 的核心：Poisson equation 在 zero Dirichlet BC 下用 DST-I 对角化 Laplacian 算子。2D DST = 1D row transform → matrix transpose → 1D row transform，类似 FFT 算法。FasTac 用 ping-pong buffer overlap write/compute/read 三 phase，hide memory bandwidth behind compute。Forward 和 inverse transform 用独立 executor，可以 separately scheduled。

### E. Streaming $F_z$ Estimation

这里 FPGA 上只 deploy $F_z$ branch（normal force），不做 $F_x, F_y$。理由：
- $F_z$ 是 grasp stability 和 contact detection 的最直接变量
- Normal contact dominant 时 tangential displacement 小，$F_z$ 可由 $d_z$ 单独 inference
- $F_z$ pipeline 完全 streaming，无 marker tracking，适合 FPGA

Quantized normal displacement（Eq. 14）：
$$q_{\Delta z}(p) = Q_z\left(\max(Z_0(p) - \hat{Z}_t(p), 0)\right)$$

变量解释：
- $q_{\Delta z}$: quantized normal displacement
- $Q_z(\cdot)$: displacement quantization function
- $Z_0(p)$: background depth
- $\hat{Z}_t(p)$: current depth
- $\max(\cdot, 0)$: 只保留 positive indentation（gel 被压入），negative（gel surface 外凸）截断为 0

Patch generation 和 kernel lookup（Eq. 15）：
$$\mathbf{v}_p = \{q_{\Delta z}(p + \delta_m)\}_{m=1}^{K^2}, \quad k_p = \text{ROM}(u,v)$$

变量解释：
- $K$: displacement window side length
- $\delta_m$: m-th offset within K×K window
- $\mathbf{v}_p$: displacement patch vector
- $k_p$: position-aware kernel vector from ROM
- $\text{ROM}(u,v)$: 把 pre-computed dynamic kernel 存在 ROM，按 pixel address $(u,v)$ 查找

这里有个 clever 的 optimization：dynamic kernel 在训练后是 deterministic function of position $p$，所以可以 offline pre-compute 所有 pixel 的 kernel vector 存 ROM，runtime 只查表，避免 hypernetwork 在 FPGA 上 forward。这把 dynamic convolution 简化成 **position-indexed look-up table + dot product**，硬件成本大幅下降。

Force computation（Eq. 16）：
$$\hat{f}_z(p) = \sum_{m=1}^{K^2} k_{p,m} v_{p,m}$$
$$A_z = \sum_{p \in \Omega} \hat{f}_z(p)$$
$$\hat{F}_z = s_F A_z, \quad s_F = \frac{1}{s_k s_d}$$

变量解释：
- $\hat{f}_z(p)$: pixel-wise integer force response（integer dot product）
- $k_{p,m}$: m-th element of position-aware quantized kernel
- $v_{p,m}$: m-th element of quantized displacement patch
- $A_z$: raw full-frame integer accumulator
- $s_k$: kernel quantization scale（kernel value / quantized kernel value）
- $s_d$: displacement quantization scale（displacement / quantized displacement）
- $s_F = 1/(s_k s_d)$: combined scale factor，把 integer accumulator 转回 Newton 单位

Intuition：quantized kernel 和 quantized displacement 相乘累加，结果是 integer，但物理意义是 (kernel / s_k) × (displacement / s_d) × scale，所以乘回 $1/(s_k s_d)$ 还原成 Newton。

Streaming 实现：patch generator 只缓存最近 K 行 buffer，新 pixel 来时 shift window 取 patch，avoid full-frame caching。

---

## VI. Experiment 深度分析

### A. RGB-NIR vs RGB-only 的 reconstruction 对比

Table II: gradient error（即 surface normal-derived gradient $p, q$ 的 MAE）

| Component | RGB-only | RGB-NIR | Improvement |
|---|---|---|---|
| $G_x$ (MAE) | 0.0047 | 0.0028 | ↓40% |
| $G_y$ (MAE) | 0.0051 | 0.0040 | ↓22% |
| Total | 0.0098 | 0.0068 | ↓31% |

$x$ 方向（横向）改善大于 $y$ 方向（纵向）。推测是因为 fingertip 的纵向 curvature 比 横向 更陡，self-shadowing 在 $x$ 方向更严重，NIR 对该方向 illumination 缺失的补全作用更明显。

Table III: depth MAE under 不同 illumination 和 boundary condition

| Boundary | RGB-only | RGB-NIR |
|---|---|---|
| With depth prior | 0.0618 | **0.0415** |
| Without depth prior | 0.2730 | 0.2686 |

两个 main effects：
1. **Boundary prior 的巨大作用**：no prior 时 MAE ~0.27mm，有 prior 时降到 0.04-0.06mm。证实 curved fingertip reconstruction 必须用 CAD boundary prior。
2. **NIR 在有 boundary prior 时进一步降低 MAE 32.8%**：从 0.0618 → 0.0415mm。NIR 单独不能 fix drift（无 prior 时 RGB-NIR 0.2686 vs RGB 0.2730 几乎一样），但跟 prior 互补 —— prior 解决低频 drift，NIR 解决局部 normal accuracy。

### B. Force Estimation 的 Performance

Fig. 8 显示三轴 force 跟 ATI ground truth 高度 linear 相关：
- $F_x$: NMAE 2.37%, NRMSE 3.15%, $R^2$ 0.9921
- $F_y$: NMAE 2.41%, NRMSE 3.33%, $R^2$ 0.9873
- $F_z$: NMAE 2.74%, NRMSE 3.72%, $R^2$ 0.9690

对比 Table I 中其他 curved sensors：
- [DenseTact 2.0](https://ieeexplore.ieee.org/document/10161336): NMAE 2.93%
- [Insight](https://www.nature.com/articles/s42256-022-00469-8): NMAE 4.00%
- [SoftBubble FEM](https://ieeexplore.ieee.org/document/10610632): NMAE 7.75%

FasTac 的 2.74% 是 SOTA，且 force output format 是 distributed three-axis map（其他多为 resultant）。

### C. Friction-coefficient Feedback Grasping

Eq. (17):
$$F_s = \sqrt{F_x^2 + F_y^2}, \quad F_n = F_z, \quad \mu = \frac{F_s}{F_n}$$

变量解释：
- $F_s$: shear force magnitude（切向 force 的模长）
- $F_n$: normal force（这里直接取 $F_z$，因为 fingertip 局部 normal 接近 z 方向）
- $\mu$: measured friction coefficient（实际利用的 friction ratio）

Controller logic：当 $\mu > \mu_s + d$ 时（$\mu_s$ 实验 stable 摩擦系数，$d$ safety margin），reduce inter-finger angle 让两 distal joint 夹紧，增加 $F_n$ 把 $\mu$ 拉回 stable region。

Fig. 9 对比 no-feedback vs with-feedback：无 feedback 时 repeated downward disturbance 持续推高 $\mu$，最后在某点 $a_3$ 发生 slip（force drop）。有 feedback 时每次 $\mu$ 越界触发 grasp tightening，$\mu$ 被拉回。这 validated real-time closed-loop tactile feedback 在 dynamic disturbance 下的 effectiveness。

### D. CPU/GPU/FPGA Pipeline 对比

Table V: end-to-end image-to-$F_z$

| Platform | Latency (ms) | Force MAE (N) | Energy (mJ/frame) |
|---|---|---|---|
| CPU (AMD Ryzen 7 8845H) | 6.82 | 0.0669 | 238.05 |
| GPU (RTX 4060 Laptop) | 3.26 | 0.0667 | 33.60 |
| **FPGA (Zynq UltraScale+)** | **1.09** | 0.0679 | **8.41** |

Latency: FPGA 6.3× faster than CPU, 3× faster than GPU。
Energy: FPGA 28× less than CPU, 4× less than GPU。
Accuracy: 三者几乎一样（0.0669 / 0.0667 / 0.0679 N），quantization 损失很小。

### E. Vibration Frequency Response

Fig. 10 测试 50/70/100 Hz 三种频率振动：
- 50 Hz: CPU/GPU/FPGA 都能正确恢复
- 70 Hz: CPU 出现 aliasing 到 31.58 Hz（CPU 输出 ~104 Hz，Nyquist ~52 Hz），GPU 和 FPGA 仍正确
- 100 Hz: CPU 和 GPU 都 alias（CPU → 1.10 Hz，GPU → 49.39 Hz），只有 FPGA 测到 100.04 Hz

Nyquist limit：$f_{max} \approx \min(f_s, f_{out})/2$。CPU 输出 104 Hz → Nyquist 52 Hz，GPU 156 Hz → 78 Hz，FPGA 240 Hz → 120 Hz。FPGA 在 100 Hz 仍 < Nyquist，所以能正确测。

这个实验的重要 message：**high-frequency tactile sensing 对 bandwidth 要求极严，FPGA 的 deterministic timing 让它成为唯一能 capture >100Hz 振动的 platform**。GPU 虽然 raw compute 强但 scheduling jitter + image transfer 让 effective output rate 上不去。

---

## VII. Intuition Building 和 Open Questions

### A. Key Takeaways for Building Intuition

1. **Photometric stereo 在 curved surface 上的 rank-deficiency 是真实 problem**。Single-source 的 near-field illumination 在 compact volume 内无法均匀覆盖，需要 redundant spectral channel。NIR 不只是 visible 的补充，它跟 visible 在 silicone 内的 scattering 路径不同，给出真正 independent observation。

2. **Boundary prior 对 curved reconstruction 是 critical**。Planar sensor 的 zero-boundary Poisson 假设 invalid 在 curved fingertip 上，必须用 CAD-aware Dirichlet boundary。这个 trick 可推广到任何 curved vision-based tactile sensor。

3. **Curved elastomer 的 mechanical response 是 spatially nonuniform**，translation-invariant CNN kernel 根本 wrong。HyperForce 的 dynamic convolution 用 hypernetwork 生成 position-dependent kernel，物理上对应 FEM 的 spatially varying stiffness matrix。

4. **FPGA 的 value 不止是 low latency，是 deterministic latency**。CPU/GPU 的 jitter 在 tactile closed-loop control 里是 killer，FPGA 的 fully pipelined streaming 让 latency variance 接近 0。

5. **Pre-compute dynamic kernel 存 ROM** 是 brilliant 的工程 optimization。训练完的 hypernetwork $\mathcal{H}_\alpha(c(p))$ 是 deterministic function of $p$，所以 offline 跑一遍存 LUT，runtime 只做 lookup + dot product。这把 dynamic convolution 的 inference cost 从 hypernetwork forward 简化成 memory access。可以推广到所有 position-conditioned 但 time-invariant 的 layer。

### B. 可能的 Extension 和联想

1. **Event camera integration**：[EveTac](https://ieeexplore.ieee.org/document/10610368) 用 event camera 大幅 reduce readout bandwidth。FasTac 跟 event camera 结合可以在 transient contact 上进一步降 latency 到 sub-ms。

2. **Differentiable FEM layer**：HyperForce 是 FEM-inspired 但仍是 black-box CNN。如果用 [DiffHand](https://differentiablerobotics.github.io/) 类的可微 FEM，能 incorporate material nonlinearity 和 hysteresis，可能在 out-of-distribution load 上 generalize 更好。

3. **Multi-finger hand scale-up**：当前 FPGA 在 single fingertip 上跑 full pipeline。Multi-finger hand 需要 5-10 个 fingertip，可以 explore shared FPGA fabric + time-division multiplexing，每个 fingertip share same DSP slices。

4. **Active illumination control**：当前 LED intensity 是 fixed。可以根据 local curvature dynamic adjust LED brightness 优化 SNR。这需要 FPGA 闭环控制 LED driver。

5. **Self-supervised adaptation**：surface normal 和 depth ground truth 来自 CAD-rendered synthetic，domain gap 到 real object 大。可以用 [NeRF-based tactile rendering](https://arxiv.org/abs/2305.05498) 或 self-supervised photometric consistency 在 real data 上 fine-tune MLP。

6. **Marker design optimization**：当前 marker dot 是固定 laser pattern。可以 optimize marker distribution（density, pattern, color）使得 tangential displacement extraction 在 high-curvature region 更 robust。

7. **Hypernetwork capacity vs generalization**：当前 hypernetwork 是 lightweight $1\times1$ conv，可能 underfitting 复杂 stiffness variation。可以试 [NeRF-like Fourier feature](https://arxiv.org/abs/2006.10739) encoding 让 hypernetwork capture high-frequency spatial variation。

8. **Tactile sim-to-real**：[TACTO simulator](https://arxiv.org/abs/2012.08456) 可以 render synthetic tactile image，跟 FasTac 的 CAD-based ground truth 生成结合，可以做 large-scale pretraining。

9. **Force vector field 输出**：当前只输出 resultant 三轴 force。如果输出 dense per-pixel force map，可以用于 incipient slip detection —— local shear force gradient 超过 friction limit 时预示 slip。

10. **Vibration spectroscopy for material identification**：100Hz+ 的 vibration sensing 可以用来 identify contact material（不同 material 的 resonance frequency 不同）。参考 [BioTac 的 vibration sensing](https://ieeexplore.ieee.org/document/5651366)。

### C. 一些 critical 角度和潜在 limitations

1. **Calibration 的 robustness**：boundary prior 来自 CAD model，但实际 gel 浇铸会有 deviation。CAD-to-gel 的 alignment 误差会引入 systematic depth bias。Paper 没有 quantify 这个 sensitivity。

2. **Material aging**：silicone elastomer 在反复 contact 后会 fatigue，reflective paint 会磨损。Stiffness $\mathbf{K}$ 和 albedo $\rho$ 都会 drift，影响 long-term accuracy。需要 online recalibration mechanism。

3. **Temperature dependence**：silicone 的 Young's modulus 随温度变化（5-10% per 10°C），Force estimation model 没 incorporate temperature。Hand 在不同环境温度下会有 bias。

4. **Out-of-distribution contact**：训练 data 是 spherical indenter + controlled normal/tangential load。Real object 有 sharp edge、large area contact、multi-point contact，分布外 generalization 没 validate。

5. **FPGA resource utilization**：paper 没报 LUT / DSP / BRAM utilization。128×128 input 上 fit 在 Zynq UltraScale+，但 resource 是否还能支持更大 input 或 full 3D force (Fx, Fy, Fz) 不清楚。

---

## VIII. Reference Web Links

- [GelSight original paper (Yuan, Dong, Adelson 2017)](https://www.mdpi.com/1424-8220/17/12/2762)
- [GelSight360 (Tippur, Adelson 2023)](https://ieeexplore.ieee.org/document/10102858)
- [DenseTact 2.0](https://ieeexplore.ieee.org/document/10161336)
- [Insight (Sun, Kuchenbecker, Martius 2022)](https://www.nature.com/articles/s42256-022-00469-8)
- [AllSight](https://ieeexplore.ieee.org/document/10341489)
- [GelSplitter3D (Lin et al. 2025)](https://ieeexplore.ieee.org/document/10629487)
- [GelStereo BioTip (Cui et al. 2024)](https://ieeexplore.ieee.org/document/10341489)
- [MinSight (Andrussow et al. 2023)](https://onlinelibrary.wiley.com/doi/10.1002/aisy.202300042)
- [HiVTac](https://www.mdpi.com/1424-8220/22/11/4196)
- [EveTac (Funk et al. 2024)](https://ieeexplore.ieee.org/document/10610368)
- [Dense Tactile Force Estimation with GelSlim and inverse FEM (Ma, Donlon, Dong, Rodriguez 2019)](https://ieeexplore.ieee.org/document/8793619)
- [iFEM2.0 (Zhao, Liu, Ma 2025)](https://ieeexplore.ieee.org/document/10610368)
- [HyperNetworks (Ha, Dai, Le 2016)](https://arxiv.org/abs/1609.09106)
- [Photometric Stereo (Woodham 1980)](https://ieeexplore.ieee.org/document/4767268)
- [OmniVision OV2736 RGB-IR sensor](https://www.omnivision.com/products/sensor/ov2736)
- [Zynq UltraScale+ MPSoC](https://www.xilinx.com/products/silicon-devices/soc/zynq-ultrascale-mpsoc.html)
- [TACTO simulator](https://arxiv.org/abs/2012.08456)
- [Fast Poisson Solver reference](https://www.cs.cmu.edu/~kmcrane/Projects/PoissonStressTest/paper.pdf)
- [Boussinesq's problem (Wikipedia)](https://en.wikipedia.org/wiki/Boussinesq%27s_problem)
- [FPGA-accelerated tactile (Hundhausen et al. 2021)](https://ieeexplore.ieee.org/document/9636861)
- [FPGA tactile suite (Oballe-Peinado et al. 2017)](https://ieeexplore.ieee.org/document/8028987)

---

## IX. Summary

FasTac 是 vision-based tactile sensing 在 curved fingertip 上的一次重要工程整合：用 RGB-NIR single-sensor multispectral imaging 解决 photometric stereo 的 rank-deficiency；用 boundary-prior Poisson 解决 curved surface 上的 depth drift；用 HyperForce position-aware dynamic convolution 解决 curved elastomer 的 spatially nonuniform stiffness；用 FPGA streaming pipeline 解决 latency 和 jitter。四个 contributions 之间互相 enable —— 没有 NIR 就 rank-deficient，没有 boundary prior 就 drift，没有 dynamic kernel 就 force MAE 爆炸，没有 FPGA 就无法 capture 100Hz 振动。这种 system-level co-design 是 robotics hardware paper 的标杆，跟 [BioTac](https://ieeexplore.ieee.org/document/5651366)、[DIGIT](https://ieeexplore.ieee.org/document/9196700)、[GelSight](https://www.mdpi.com/1424-8220/17/12/2762) 一起构成了 vision-based tactile 的演进谱系。后续 work 应该关注 long-term stability、out-of-distribution contact、和 multi-finger hand 上的 scale-up。
