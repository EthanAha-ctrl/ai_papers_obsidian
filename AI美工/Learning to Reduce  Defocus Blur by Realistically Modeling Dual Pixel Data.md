---
source_pdf: Learning to Reduce  Defocus Blur by Realistically Modeling Dual Pixel
  Data.pdf
paper_sha256: 6d56c72b53ff17a32a15ddbb5bac69695510ad21fd67b78ab4b0418f65d60974
processed_at: '2026-08-05T13:56:41-07:00'
target_folder: AI美工
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版本

好，前面技术细节都讲完了，现在我们坐下来用大白话把这篇 paper 的核心 idea 捋一遍。

---

## 这篇 paper 到底在干嘛

一句话：**手机拍照常常"主体清楚、背景糊"，这篇 paper 想把背景糊掉的部分给恢复清楚**。

这种糊叫 defocus blur (散焦模糊)。你拍人像开大光圈，背景那个圆乎乎的糊团就是它。有时候这是艺术效果，有时候就是烦——比如你拍文件、拍屏幕、拍远处标识，糊了就废。

问题是这个糊跟 motion blur (运动模糊) 完全不一样。Motion blur 是"东西动了，整个东西拉成一条线"，方向感很强。Defocus blur 是"东西没动，但镜头没法把它聚焦到 sensor 上，光散成一个圆圈"——这个圆圈叫 **circle of confusion (CoC)**。圆圈越大，糊得越狠。圆圈的大小取决于这个物体离对焦面有多远。

恢复 defocus blur 的难点在于：**这个圆圈在图上每个像素位置都不一样大**。因为你画面里有的东西近、有的东西远，每个位置的 CoC 半径都不同。所以你不能用一个固定的"去模糊滤波器"全场跑一遍，得知道每个像素位置 blur 有多严重。传统做法先估一张 "defocus map" (整张图每个像素的模糊程度)，再做 deconvolution。问题是 defocus map 估不准，后面 deconvolution 全崩。

---

## Dual-Pixel Sensor 是什么宝贝

DP sensor 是这篇 paper 的主角。它原本是给相机 **autofocus (自动对焦)** 用的。

普通 sensor 一个 pixel 就一个 photodiode (光敏二极管)，收光、给一个数。DP sensor 一个 pixel 位置塞了 **两个** photodiode，左半边收光、右半边收光，分别读出来。你可以理解为 sensor 把每个像素劈成两半，左半和右半各拍一张图，所以你一次曝光能拿到 **两张略有差异的图** (left view 和 right view)。

这两张图的差异在哪？就在 defocus blur 上。

想象一个点光源正好在对焦面上：光线会聚成一个点，左右两半 sensor 各收到一半光，左右图都看到同一个点，**没有差异**。

现在这个点光源跑到了对焦面前面：光线还没会聚成点就打到 sensor 上了，形成一个圆圈 (CoC)。左半 sensor 看到的圆圈和右半 sensor 看到的圆圈，会有一个 **水平的相对偏移**——因为左半 sensor 在物理上偏左，它看到的圆圈中心也偏左一点；右半 sensor 偏右，看到的圆圈中心也偏右一点。所以 left view 和 right view 里这个点的位置会**水平错开**。

点跑到对焦面后面同理，只是错开方向反过来。

**这个错开量 (disparity) 直接告诉你 CoC 有多大，也就告诉你这个像素糊得有多狠**。

这是 DP sensor 对 defocus deblurring 的巨大优势：你不需要先估 defocus map，sensor 直接把"模糊信息"编码在两张图的差异里给你了。这就像 sensor 自带一个"模糊探测器"。

Google Pixel 手机和 Canon 5D Mark IV 都有 DP sensor。但只有 Canon 5D Mark IV 允许你读出 raw DP data，而且允许你调光圈。

---

## 前作 DPDNet 的痛点

2020 年 Abuolaim 和 Brown 发了一篇 ECCV，叫 DPDNet，第一次用 DP sensor 做 defocus deblurring。他们用 Canon 5D Mark IV 拍了个数据集：每张场景拍两张，一张大光圈 (模糊)，一张小光圈 (清晰当 ground truth)，同时读出 DP 的 left 和 right view。然后用一个 U-Net 之类的 CNN 学：输入 left + right，输出清晰的 deblur 图。结果非常好，吊打之前所有 defocus map 估计的方法。

但 DPDNet 有几个让人抓狂的问题：

**第一，数据采集极其麻烦**。你想，要拍一对训练样本，你得同一个场景、同一个 pose，先大光圈拍一张，再小光圈拍一张。光圈切换瞬间，相机可能微动，光线可能微变。Paper 里 Fig. 3 专门画了这个 misalignment：同一个 in-focus 区域，从两次拍摄的 patch 抠出来做 2D cross-correlation，峰值不在中心，说明两张图有局部错位。你的 ground truth 是脏的。

**第二，只有 Canon 5D Mark IV 能用**。它是市面上唯一同时满足"给 raw DP access"+"可调光圈"两个条件的相机。你想换别的相机采数据？没门。这导致整个研究方向被一台相机绑架。

**第三，智能手机根本采不到 GT**。手机的光圈是固定的 (fixed aperture)，你拍不了小光圈清晰图当 GT。所以 Pixel 4 虽然有 DP sensor，能读出 DP data，但永远没 GT，训练不了 DPDNet。手机端 defocus deblurring 这条路被堵死了。

**第四，video 完全没戏**。没有任何相机允许你同步录 DP video。你想要 video defocus deblurring？连数据都没有，怎么训网络。

---

## 这篇 paper 的核心 idea

既然真实数据采不到，那我自己**合成**不就完了？

合成数据听起来简单，做起来很难。你得让合成的 DP 图跟真实 DP sensor 拍出来的看起来一样，否则网络在合成数据上训完，到真实数据上跑就崩 (domain gap)。

这篇 paper 的核心贡献就是一条**完整的 DP 成像链路模拟器**，把真实 sensor 从光到图的全过程一段段建模出来：

### 第一步：Thin Lens Model 算 CoC

用最简单的薄透镜模型，给定焦距、光圈、对焦距离、场景深度，算出每个像素位置的 CoC 半径。公式 1。

CoC 半径的**符号**有物理意义：正的代表"物体在对焦面后面" (front focus，背景糊)，负的代表"物体在对焦面前面" (back focus，前景糊)。这个符号后面决定 DP 的 left/right PSF 谁往哪边偏。

### 第二步：模拟 DP PSF — 这是关键

CoC 是个理想圆盘。但真实镜头拍出来的 PSF 长什么样？你如果去拿 Canon 实测一下，会发现 PSF **不是均匀圆盘**，而是个**甜甜圈**形状——中间凹下去，周围一圈亮，边缘渐变 fall-off。

为什么？光学像差、microlens、sensor well 深度，一堆物理因素合在一起的结果。前作 [36] (Punnappurath, ICCP 2020) 提了一个单参数模型，能 capture left/right 的对称性，但匹配不上这个甜甜圈形状。

这篇 paper 的招：拿一个 2D Butterworth filter 调制圆盘。Butterworth filter 本身是个低通滤波器的形状，中间高、边缘低。用它调制圆盘，再 rescale 让中心有个最小残留 $\beta$ (保证中心是凹陷但不是零)，就做出来了甜甜圈。然后用一个小 Gaussian 平滑边缘，让它不那么生硬。

接着把 combined PSF 拆成 left 和 right：用一个 2D ramp mask (斜坡 mask)，往一边逐渐 fall-off，乘到 combined PSF 上得到 left PSF。Ramp 的方向由 CoC 半径符号决定——前面就往左 fall-off，后面就往右 fall-off。然后 left PSF 水平翻转就是 right PSF。

这招挺 hack 的，但有效。实测 PSF 跟他们模型生成的 PSF 做 2D cross-correlation，相似度比前作 [36] 高很多 (Fig. 5)。

他们一共搜出来 48 组 PSF 参数组合当 PSF bank，训练时随机采样。

### 第三步：Radial distortion

镜头有桶形/枕形畸变，直线在边缘会弯。他们用 division model 拿 5 组 focal length 各自标定一组径向畸变系数，合成时按焦距套用。

### 第四步：Signal-dependent noise

CG 渲染的图是干净的，真实 sensor 有噪声。噪声强度跟像素亮度成正比 (亮的地方噪声方差大)，这叫 signal-dependent Gaussian noise。Left view 和 right view 各自独立采样噪声，但共享同一个 $\sigma$。

### 第五步：合成 DP views

数据源用 SYNTHIA dataset，一个 CG 渲染的虚拟城市街景数据集，自带 depth map。按 per-pixel depth 把图拆成最多 500 层，每层用对应深度的 PSF 卷积 (image 和 mask 都卷)，然后 back-to-front alpha blend。再套 radial distortion，再套 noise。公式 6。

每个 SYNTHIA 序列生成 5 个 blurred 版本 (5 套相机参数)，总共 2023 训练 + 201 测试。

**这套合成数据的关键性质**：in-focus 区域 left/right 无 disparity，out-of-focus 区域有 disparity——跟真实 DP sensor 一模一样。

---

## RDPD 网络架构

有了合成数据，现在训网络。作者提出 RDPD (Recurrent Dual-Pixel Deblurring)。

### 主体：Encoder-Decoder + convLSTM

骨架是 U-Net 风格的 encoder-decoder，带 skip connections。跟 DPDNet 一样的套路。但 bottleneck 处塞了个 **convLSTM**。

为什么要 convLSTM？因为作者想同时支持**单图**和**视频**。单图就跑一帧，视频就连续跑多帧，convLSTM 在时间维度上传递信息。

ConvLSTM 跟普通 LSTM 的区别是：普通 LSTM 用 dot product，丢失空间信息；convLSTM 用卷积，保留空间结构。这对 defocus deblurring 是必须的——因为 PSF 在空间上是 varying 的，网络得知道"这是图的哪个位置"。

但有个问题：没有任何相机录 DP video，作者怎么训 video 模式？答：用合成数据。SYNTHIA 本身是视频序列，合成 DP views 时保留时间维度，就能训 video deblur。真实数据只能单图，合成数据可以多帧。

训练时迭代：一个 batch 真实单图、一个 batch 合成多帧序列交替喂。convLSTM 的好处是它不在乎 sequence length，单图就当 length=1 的序列跑。

### Trick 1: Radial Distance Patch

Patch-wise training (抠 512×512 patch 训练) 是 deblur 任务的标准操作，省内存、增广数据。但问题来了：CNN 看到的只是这个 patch，它不知道这个 patch 来自全图的哪个位置。可是 PSF 因为 radial distortion 和 lens aberration，在径向方向上 spatially varying——离图像中心越远，PSF 形状越歪。

作者的招：再喂一个 1-channel 的 patch，每个像素值代表"这个像素离图像中心的相对径向距离"。相当于告诉网络"你现在看的是图中心还是边缘"。

简单粗暴，+0.4 dB。

### Trick 2: Multi-Scale Edge Loss

Defocus blur 把边缘糊掉。Sobel 算子能提边缘。那就让网络输出的 Sobel edge map 跟 GT 的 Sobel edge map 做 MSE，强迫网络恢复边缘。

但单一尺度 Sobel (3×3) 只抓一种粗细的 edge。Defocus blur 的 CoC 大小跟深度相关，有大有小，对应不同尺度的 edge 模糊。所以用 3 个尺度 (3×3, 7×7, 11×11) 的 Sobel。

另外，DP sensor 的 disparity 在水平方向，所以垂直方向上的 edge 更难恢复 (left/right view 里垂直 edge 位置一致，水平 edge 位置会错开)。因此 x 方向和 y 方向的 edge loss 给不同权重 ($\lambda_x = 0.03, \lambda_y = 0.02$)，y 方向 (水平 edge) 给更低权重，让网络更专注于垂直 edge 的恢复。

这个设计有物理动机，不是瞎调。+0.3 dB。

### 其他改动

- 每个 block 节点数减半，模型更轻
- 最后一层用 linear + [0,1] clamping，不用 sigmoid
- convLSTM 512 units，dropout 0.4 防 overfit

---

## 实验结果说人话

### Table 1: Canon 测试集单图

RDPD+ (合成数据+Canon 数据混合训) 比 DPDNet+ 高 0.27 dB overall，indoor 高 0.45 dB，速度快 40% (0.3s vs 0.5s)，参数量更少。

Outdoor 略低 0.08 dB，作者的解释很有意思：**outdoor 的 GT 本身有 misalignment (前面 Fig. 3 说的那个)，DPDNet 在 overfit 这个 noise**。RDPD+ 因为加了干净的合成数据，反而"debias"了，不会去拟合 GT 里的瑕疵。所以 PSNR 略低但其实 deblur 质量更高。NIQE (无参考图像质量指标) 上 RDPD+ 是 3.19，DPDNet+ 是 3.73，RDPD+ 更好，印证了这一点。

**这个观察挺重要的**：当 GT 不干净时，PSNR 高未必是好事，可能是在学 GT 的瑕疵。

### Table 2: 合成数据上 video vs single frame

RDPD+ (多帧训练) 31.09 dB，sRDPD+ (单帧训练) 30.26 dB，差 0.83 dB。ConvLSTM 在时间维度上学到了东西。

Fig. 8 用 Canon 模拟 4 帧 sequence (小相机 motion)，多帧训练 +0.4 dB。

### Ablation 几个关键数

- DP dual view vs single view: +1.15 dB ← 最大头，disparity 信息是核心
- 我们的 PSF 模型 vs [36] 的 PSF 模型: +0.7 dB ← 甜甜圈建模有效
- Radial distortion 加 vs 不加: +0.2 dB
- Radial distance patch: +0.4 dB
- Multi-scale edge loss vs no edge loss: +0.33 dB

每个组件都贡献明确。

### Cross-camera generalization

**RDPD baseline 只用合成数据训练**，直接拿到 Canon 和 Pixel 4 上跑，能输出合理的 deblur 结果。Pixel 4 没有 GT，但视觉上看起来对。这说明合成 pipeline 真的逼真，sim2real 跨过来了。

这是 paper 最强的 claim：你可以在合成数据上训，部署到任何有 DP sensor 的相机上，包括手机。

---

## 我的直觉理解

1. **DP sensor 的本质是一台 2-sample light field camera**。它只采样了 light field 的水平方向两个位置，但这两个位置足够你提取 defocus 信息。它不给你完整的 4D light field，但给你一个"模糊探测器"，足够 deblur 用。

2. **Donut-shaped PSF 的物理来源是 optical aberration**。理想镜头的 PSF 是均匀圆盘，但真实镜头有球差、彗差、场曲、像散，加上 microlens 和 sensor well 深度，合在一起把均匀圆盘"啃"成甜甜圈。这个甜甜圈形状在 DP sensor 上尤其明显，因为 DP 的两个 view 本身就是半个 aperture 的采样，aberration 在半个 aperture 上更不对称。

3. **Synthetic data 的核心 trick 是 domain randomization**。作者没有试图 hardcode 一台 Canon 5D Mark IV，而是用 5 套相机参数 × 48 组 PSF × 多个 noise level 撒一个分布。网络在这么广的分布上训，到任何一台真实相机上都是分布内推理。这跟 sim2real 文献的思路一致 ([31] Maximov CVPR 2020, [41] Shrivastava CVPR 2017)。**你不需要匹配真实，你需要覆盖真实**。

4. **Patch-wise training 丢失位置信息这件事**在 deblur 任务里被很多人忽略。CNN 是 translation invariant 的，它不知道 patch 来自哪里。但 defocus PSF 因为 radial distortion 在径向上变化，位置信息很关键。Radial distance patch 是个简单的 fix，但思想值得推广——任何 spatially-varying 任务的 patch-wise training 都该考虑喂位置信息。

5. **Edge loss 的方向性分解**是个有物理直觉的设计。DP disparity 在水平方向，所以水平 edge 在 left/right view 间位置不变，容易恢复；垂直 edge 在 left/right view 间位置错开，更难恢复。给不同权重让网络集中火力在难方向上。

---

## 这篇 paper 的真正价值

技术上，PSF 模型那块挺 hack 的——拿 Butterworth filter 调制圆盘，brute-force 搜参数，48 组合构成 bank。谈不上优雅，但 work。

真正的价值在**解锁了几个之前做不到的应用**：

1. **智能手机 defocus deblurring**：手机 fixed aperture 采不到 GT，现在用合成数据训一次就能部署。
2. **Video defocus deblurring**：没有相机录 DP video，合成数据让 convLSTM 有东西可学。
3. **跨 camera 通用**：不用每台相机采一遍数据，合成 pipeline 调一下参数就能生成新相机的 DP 数据。

这是工程意义上的解锁。学术界很多人不喜欢 synthetic data paper，觉得"不就是仿真嘛"。但仿真做到能 sim2real 跨设备 generalize，且超过在真实数据上训的 baseline，这就不只是仿真了，是对成像物理的深度理解。

References:
- Paper PDF: https://arxiv.org/abs/2112.01997
- GitHub code: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel
- DPDNet 前作: https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel
- Abuolaim 主页: https://www.eecs.yorku.ca/~abuolaim/
- DP depth (Garg et al. ICCV 2019): https://arxiv.org/abs/1904.08039
- DP PSF modeling (Punnappurath ICCP 2020): https://arxiv.org/abs/2003.12719
- SYNTHIA dataset: http://synthia-dataset.net/
- NTIRE 2021 DP deblur challenge: https://openaccess.thecvf.com/content/CVPR2021/papers/Abuolaim_NTIRE_2021_Challenge_for_Defocus_Deblurring_Using_Dual-Pixel_Images_CVPR_2021_paper.pdf

---

# Paper 讲解：Learning to Reduce Defocus Blur by Realistically Modeling Dual-Pixel Data

## 1. 这篇 paper 解决什么问题

这篇 ICCV 2021 paper 由 Abdullah Abuolaim (York University) + Google Research 团队提出，本质上是一个 **数据生成 + 网络架构联合设计** 的工作，对应解决前作 DPDNet (ECCV 2020, [1]) 的几个痛点。

Defocus blur (散焦模糊) 的物理来源是 scene point 落在 camera's depth of field (DoF) 之外，其 PSF 在空间上是 spatially-varying 的——不仅依赖 scene depth，还依赖 aperture、focal length、focus distance、radial distortion、optical aberrations 等多个量。传统 pipeline ([21][24][40]) 是先估 defocus map，再做 non-blind deconvolution，结果被 defocus map 的精度卡死。

Abuolaim & Brown 在 ECCV 2020 第一次提出用 **dual-pixel (DP) sensor** 来做 defocus deblurring。DP sensor 在每个 pixel 位置有两个 photodiode，相当于一台简化版的两视角 light-field camera，原本是给 autofocus 用的，但 [1] 发现两个 view 之间相对的 phase shift 直接对应 CoC 大小。问题在于：

1. **数据采集瓶颈**：要把一张 wide-aperture 模糊图和一张 narrow-aperture 清晰图配对作为 GT，必须 adjustable aperture；同时只有 Canon 5D Mark IV 既给 raw DP access 又能调光圈。
2. **GT 不干净**：因为 wide/narrow aperture 必须依次拍，所以 Fig. 3 里的 indoor/outdoor dataset 都存在 local misalignment、small motion、illumination drift。
3. **智能手机无法采 GT**：Pixel 4 这类手机是 fixed aperture，根本拍不到 narrow-aperture GT。
4. **没有 video DP data**：没有任何相机允许同步录 DP video，所以 video defocus deblurring 这条路之前根本走不通。

这篇 paper 用 **synthetic DP data generation pipeline** + **recurrent CNN (RDPD)** 一起解决上面四条。

GitHub: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel

---

## 2. 核心 idea：模拟 DP 成像链路

### 2.1 Thin Lens Model (公式 1)

先用 thin lens model 把 CoC (circle of confusion) 半径算出来。给焦距 $f$、对焦距离 $s$、f-stop $F$、场景点距离 $d$：

$$s' = \frac{f s}{s - f}, \quad q = \frac{f}{F}$$

- $s'$: lens 到 sensor 的像距
- $q$: aperture 直径
- $d$: 场景点到 lens 的物距
- $s$: 对焦平面到 lens 的距离

CoC 半径：

$$r = \frac{q}{2} \cdot \frac{s'}{s} \cdot \frac{d - s}{d} \tag{1}$$

注意 $r$ 的符号有物理意义：$r>0$ 表示 front focus (对焦面后面的点模糊)，$r<0$ 表示 back focus (对焦面前的点模糊)。这一点 DP 的 left/right PSF 之间会"翻面"，这是后文 Eq. 4 中用 ramp mask 方向的核心依据。

### 2.2 DP PSF 模型 (公式 2-4) — 这是 paper 最关键的创新

前作 [36] (Punnappurath et al., ICCP 2020) 用单参数模型近似 DP PSF，能用 horizontal symmetry 关系 $H_r = H_l^f$ (水平翻转)，但匹配不上真实 PSF 的 **donut-shaped depletion** (中心凹陷，环形凸起)，原因是 optical aberrations。

这篇 paper 用一个 2D Butterworth filter 作为"调制内核"：

$$\mathbf{B}(x,y) = \left(1 + \left(\frac{D_o}{\sqrt{(x-x_o)^2 + (y-y_o)^2}}\right)^{2n}\right)^{-1} \tag{2}$$

- $n$: filter order，控制 roll-off 平滑度
- $D_o$: 3 dB cutoff 位置
- $(x_o, y_o)$: kernel 中心

然后定义：

$$\mathbf{H} = \mathbf{B} \circ \mathbf{C}(x_o, y_o) \tag{3}$$

- $\mathbf{C}$: 半径为 $r$ 的圆盘 (CoC 半径决定边界)
- $\circ$: Hadamard product
- $D_o$ 是 $r$ 的函数，通过参数 $\alpha$ 调节
- $\mathbf{B}$ 的值 rescale 到 $[\beta, 1]$，$\beta > 0$ 控制中心最小 depletion (中心残留是正的，跟 measured PSF 一致)
- 用 Gaussian kernel (std = $\kappa r$, $0 < \kappa \ll 1$) 平滑 $\mathbf{H}$，使边缘 fall-off 更接近真实

Combined DP PSF: $\mathbf{H} = \mathbf{H}_l + \mathbf{H}_r$，约束 $\mathbf{H}_r = \mathbf{H}_l^f$ (水平翻转)：

$$\mathbf{H}_l = \mathbf{H} \circ \mathbf{M}, \quad \text{s.t. } \mathbf{H}_l \geq 0, \quad \sum \mathbf{H}_l = \frac{1}{2} \tag{4}$$

- $\mathbf{M}$: 2D ramp mask with constant decay，方向由 $r$ 的符号决定 (front/back focus)，给 left PSF 一个"向右逐渐 fall-off"的渐变

5 个参数 $\{n, \alpha, \beta, \kappa\}$ 加上 mask 方向。Fig. 5 显示用这个模型生成的 PSF 跟 Canon 5D Mark IV 实测 PSF 的 2D cross-correlation 比 [36] 高很多。

**参数搜索** (Sec. S2) 用 brute-force 在以下离散空间搜：

$$n \in \{1, \ldots, 15\}, \quad \alpha, \beta \in \{0.1, \ldots, 1.0\}, \quad \kappa \in \{0.14, 0.21, \ldots, 0.42\}$$

最优解收敛到 $n \in \{3,6,9\}$, $\alpha \in \{0.4, 0.6, 0.8, 1.0\}$, $\beta \in \{0.1, 0.2, 0.3, 0.4\}$, $\kappa = 0.14$，一共 48 组合，构成 PSF bank。

### 2.3 Radial distortion + signal-dependent noise

Radial distortion 用 division model (Fitzgibbon, CVPR 2001 [8])：

$$(x_d, y_d) = (x_o, y_o) + \frac{(x_u - x_o, y_u - y_o)}{1 + c_1 R^2 + c_2 R^4 + \cdots} \tag{5}$$

- $(x_u, y_u)$: undistorted 坐标
- $(x_d, y_d)$: distorted 坐标
- $c_i$: 第 $i$ 个 radial distortion coefficient
- $R$: 像素到 image center $(x_o, y_o)$ 的径向距离

5 个焦距对应的 5 组系数在 Sec. S3 里给出 (覆盖 barrel 和 pincushion 两类)。

Noise 用 signal-dependent Gaussian (Foi et al. TIP 2008 [9], Liu et al. TPAMI 2007 [27])：

$$\mathbf{I}_{\text{noise}} = \mathbf{I} + \mathbf{I} \circ \mathbf{N}, \quad \mathbf{N} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{Id})$$

- $\mathbf{I}$: noiseless image
- $\mathbf{N}$: zero-mean Gaussian noise layer
- $\sigma$: 控制噪声强度，跟像素强度成正比
- left/right view 各自独立采样 $\mathbf{N}_l, \mathbf{N}_r$，但用同一 $\sigma$

### 2.4 DP view 合成 (公式 6)

数据源是 SYNTHIA dataset [16] (虚拟城市 street view，CG 渲染 photorealistic 序列，自带 depth-buffer + segmentation)。

合成步骤：
1. 按 per-pixel depth 把 image 拆成 ≤500 层
2. 每层用对应的 PSF 卷积 (image 和 mask 都卷)
3. back-to-front alpha blend，用 blurred mask 当 alpha
4. Radial distortion 应用到 $\mathbf{I}_l, \mathbf{I}_r, \mathbf{I}_s$
5. 加 signal-dependent noise $\mathbf{N}_l, \mathbf{N}_r$ (独立采样)

公式：

$$\mathbf{I}_l = \mathbf{I}_s * \mathbf{H}_l, \quad \mathbf{I}_r = \mathbf{I}_s * \mathbf{H}_r \tag{6}$$

最终 blurred image: $\mathbf{I}_b = \mathbf{I}_l + \mathbf{I}_r$。

**Intuition 这里**：合成出来的 DP view 在 in-focus 区域没有 disparity，out-of-focus 区域有 disparity，这个性质跟真实 DP sensor 完全一致。

5 个 camera parameter set:
- $\{4, 5, 6\}$, $\{5, 8, 6\}$, $\{7, 5, 8\}$, $\{10, 13, 12\}$, $\{22, 10, 30\}$ (focal length, aperture, focus distance)

每个 SYNTHIA 序列生成 5 个 blurred 版本，总共 2023 train + 201 test。

---

## 3. RDPD (Recurrent Dual-Pixel Deblurring) 架构

### 3.1 整体结构

Encoder-decoder CNN + skip connections (U-Net 风格 [38], Mao et al. NeurIPS 2016 [30])，bottleneck 处插入 convLSTM (Shi et al. NIPS 2015 [48]) 处理 sequence。

跟 DPDNet 的差异：
1. bottleneck 加 convLSTM units (size 512)
2. 加 radial distance patch 输入
3. multi-scale edge loss
4. 每个 block 节点数减半 → 更轻量
5. 最后一层换成 linear layer with $[0,1]$ clamping (跟 [11] 一致)

输入 7 channel (6 RGB DP view + 1 radial distance)；输出 3 channel (sharp RGB)。

### 3.2 convLSTM 公式 (Eq. 7-11)

给定时间 $t$ 的 encoded feature $\mathbf{X}_t$：

$$\mathbf{i}_t = \Sigma(W_i^X * \mathbf{X}_t + W_i^\mathcal{H} * \mathcal{H}_{t-1} + W_i^\mathcal{C} \circ \mathcal{C}_{t-1} + b_i) \tag{7}$$
$$\mathcal{F}_t = \Sigma(W_\mathcal{F}^X * \mathbf{X}_t + W_\mathcal{F}^\mathcal{H} * \mathcal{H}_{t-1} + W_\mathcal{F}^\mathcal{C} \circ \mathcal{C}_{t-1} + b_\mathcal{F}) \tag{8}$$
$$\mathbf{o}_t = \Sigma(W_o^X * \mathbf{X}_t + W_o^\mathcal{H} * \mathcal{H}_{t-1} + W_o^\mathcal{C} \circ \mathcal{C}_{t-1} + b_o) \tag{9}$$
$$\mathcal{C}_t = \mathcal{F}_t \circ \mathcal{C}_{t-1} + \mathbf{i}_t \circ \tau(W_\mathcal{C}^X * \mathbf{X}_t + W_\mathcal{C}^\mathcal{H} * \mathcal{H}_{t-1} + b_\mathcal{C}) \tag{10}$$
$$\mathcal{H}_t = \mathbf{o}_t \circ \tau(\mathcal{C}_t) \tag{11}$$

变量含义：
- $\mathbf{i}_t, \mathcal{F}_t, \mathbf{o}_t$: input/forget/output gate (sigmoid $\Sigma$)
- $\mathcal{C}_t$: memory cell
- $\mathcal{H}_t$: hidden state
- $W$: 各 gate 的卷积权重 (上标指明对 $\mathbf{X}, \mathcal{H}, \mathcal{C}$ 的卷积)
- $b$: bias
- $\tau$: tanh

输出：
$$\mathbf{I}_d(t) = \text{CNN-Decoder}(\mathbf{o}_t) \tag{12}$$

convLSTM 用卷积替代 vanilla LSTM 的 dot product，保留 spatial information，这对 spatially-varying PSF 估计是必须的。

### 3.3 Radial distance patch

Patch-wise training 把 patch 从全图中抠出来独立喂给网络，丢失了空间位置信息。但 PSF 在径向上 spatially varying (radial distortion + lens aberration)，所以喂一个额外 1-channel patch 表示"距 image center 的相对径向距离"。这一支给 +0.4 dB (Table S4)。

### 3.4 Multi-scale edge loss (Eq. 13-16)

$$G_d^x = \mathbf{I}_d * S_{m \times m}^x, \quad G_d^y = \mathbf{I}_d * S_{m \times m}^y \tag{13}$$
$$G_s^x = \mathbf{I}_s * S_{m \times m}^x, \quad G_s^y = \mathbf{I}_s * S_{m \times m}^y \tag{14}$$

- $S_{m \times m}^x, S_{m \times m}^y$: 尺寸 $m$ 的 vertical/horizontal Sobel 算子
- $\mathbf{I}_d$: 网络输出 deblur 图
- $\mathbf{I}_s$: GT sharp 图

$$\mathcal{L}_\text{edge}^{\{x,y\}} = \mathbb{E}[\text{MSE}(G_s^{\{x,y\}}, G_d^{\{x,y\}})] \tag{15}$$

$$\mathcal{L} = \mathcal{L}_\text{MSE} + \lambda_x \mathcal{L}_\text{edge}^x + \lambda_y \mathcal{L}_\text{edge}^y \tag{16}$$

- $m \in \{3, 7, 11\}$: 3 个 scale
- $\lambda_x = 0.03, \lambda_y = 0.02$ (方向上不等权重，因为 sensor 水平方向上有 DP disparity，要更多关注垂直方向 edge)

跟 [28] 的 single-scale $3\times3$ Sobel loss 比，多尺度 + 方向分解 +0.3 dB (Table S5)。

---

## 4. 实验数据

### 4.1 主表 (Table 1) Canon DP test set

| Method | Indoor PSNR | Outdoor PSNR | Overall PSNR | Overall SSIM | Time (s) |
|---|---|---|---|---|---|
| EBDB [21] | 25.77 | 21.25 | 23.45 | 0.683 | 929.7 |
| DMENet [24] | 25.70 | 21.51 | 23.55 | 0.720 | 613.7 |
| JNB [40] | 26.73 | 21.10 | 23.84 | 0.715 | 843.1 |
| DPDNet [1] | 27.48 | 22.90 | 25.13 | 0.786 | 0.5 |
| DPDNet+ [1] | 27.65 | 22.72 | 25.12 | 0.784 | 0.5 |
| **RDPD+** (ours) | **28.10** | 22.82 | **25.39** | 0.772 | **0.3** |

注意点：
- RDPD+ 比 DPDNet+ 在 indoor +0.45 dB，overall +0.27 dB，参数少很多，速度快 40%
- Outdoor 上 RDPD+ 略输 0.08 dB，作者的解释 (Sec. 5): outdoor GT 有 misalignment/illumination 不一致，DPDNet 在 overfit 这个 noise，而 RDPD+ 因为加了 synthetic 干净数据"debias"了。这一点挺有意思—— PSNR 在不干净 GT 上反而不能反映真实性能
- RDPD baseline (synthetic only, 不见真实数据) 能 generalize 到 Canon 和 Pixel 4，说明合成 pipeline 真的逼真

### 4.2 Sequence 实验 (Table 2)

| Method | PSNR | SSIM | MAE |
|---|---|---|---|
| DPDNet [1] | 26.38 | 0.782 | 0.034 |
| DPDNet+ [1] | 29.84 | 0.828 | 0.025 |
| sRDPD+ (single frame) | 30.26 | 0.849 | 0.020 |
| **RDPD+** (multi-frame) | **31.09** | **0.861** | **0.016** |

Video setting 比 single-frame 高 0.83 dB，证明 convLSTM 在 temporal dependency 上的作用。Fig. 8 在 Canon sequence 上做 4 帧模拟 capture (小相机 motion)，多帧训练后平均 PSNR +0.4 dB。

### 4.3 Ablation study (Sec. S4)

在 Canon combined test 上：

| Component | Δ PSNR |
|---|---|
| Our PSF vs [36] PSF | +0.7 |
| Radial distortion on vs off | +0.2 |
| Dual view vs single view (RSPD+) | +1.15 |
| Radial distance patch | +0.4 |
| Multi-scale edge loss vs no edge | +0.33 |
| Multi-scale vs single-scale Sobel | +0.28 |
| Both (radial+edge) vs none | +0.49 |

每个组件都贡献明确。最大的是 dual view (+1.15 dB)，这跟 [1] 的结论一致：DP disparity 信息是核心信号，不能省。

### 4.4 Cross-camera generalization (Fig. 7, Fig. S8)

RDPD baseline (synthetic only) 直接拿到 Pixel 4 上跑，能输出合理 deblur 结果，而 Pixel 4 没有 ground truth (因为 fixed aperture)。这是 paper 最强 claim 之一：**synthetic pipeline 跨 camera generalization**，且支持了之前无法做的 video DP 应用。

---

## 5. 我觉得的几个关键 intuition

1. **Donut-shaped PSF 的建模**：之前 [36] 的单参数模型就是没法 capture 中心凹陷。But 这个凹陷本质是 optical aberration，跟 anti-aliasing filter + microlens + sensor well depth 一起作用的结果。用 Butterworth 调制的做法很 hack 但有效——本质上是用一组有"中心凹陷、边缘平滑 fall-off"的函数族去 fit 实测 PSF。

2. **Disparity 跟 defocus 的耦合**：DP sensor 给的两个 view 不是单纯的"两张图"，它们的 disparity 直接编码了 CoC 大小，所以 left/right 谁更模糊一目了然。前作 [36] 用这个性质做 depth from defocus-disparity，这里 [1]+本作 用它做 deblur。两者是同一物理现象的两个 view。

3. **Patch-wise training 丢失的位置信息**：CNN 看不见 patch 在全图中的位置，但 radial distortion + aberration 让 PSF 沿径向变化。补一个 radial distance channel 是简单粗暴有效的 trick，等价于告诉网络"你现在看的是 image 的哪个部位"。

4. **Multi-scale Sobel loss 为什么有用**：defocus blur 会把 edge 模糊掉，single-scale Sobel 只抓一个尺度的 edge；defocus PSF 的尺寸跟 CoC 半径成正比 (跟 depth 相关)，所以多尺度 Sobel 能覆盖各种 blur 等级的 edge。分 x/y 方向是因为 DP sensor 的 left/right 视差在水平方向，让网络在垂直方向上恢复更狠，这一支设计是有物理动机的。

5. **Synthetic domain gap 通过"分布式参数"弥合**：作者没有 hardcode 一个 camera，而是用 5 个 camera × 48 个 PSF × 多个 σ noise level 撒一个分布。这跟 sim2real literature ([31] Maximov et al. CVPR 2020, [41] Shrivastava et al. CVPR 2017) 的思路一致——**domain randomization 比 domain matching 更鲁棒**。

6. **Outdoor 数据集"脏"导致 PSNR 失真**：这点提醒在 vision 任务里很重要。GT 不完美时，模型 PSNR 高未必 deblur 好，可能是在 overfit noise。RDPD+ 在 outdoor 略输 0.08 dB，作者解释是它没 overfit 到 noise 上，这个 claim 在 NIQE 上有支撑 (RDPD+ NIQE 3.19 vs DPDNet+ 3.73)。

---

## 6. 跟其他工作的关联

- **[1] Abuolaim & Brown, ECCV 2020** Defocus deblurring using dual-pixel data — 这篇的直接前作，建立了 DP defocus deblur 任务和 Canon 数据集。https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel
- **[36] Punnappurath et al., ICCP 2020** Modeling defocus-disparity in dual-pixel sensors — DP PSF 的第一个参数化模型，本作 PSF model 直接对标它。https://arxiv.org/abs/2003.12719
- **[31] Maximov et al., CVPR 2020** Focus on defocus: bridging the synthetic to real domain gap for depth estimation — 跟本作同一思路但做 depth。
- **[10] Garg et al., ICCV 2019** Learning single camera depth estimation using dual pixels — Google 用 DP 做单目 depth 的开山之作。https://arxiv.org/abs/1904.08039
- **[46] Wadhwa et al., SIGGRAPH 2018** Synthetic depth-of-field with a single-camera mobile phone — Pixel 手机上 bokeh 的工程实现。https://research.google/pubs/synthetic-depth-of-field-with-a-single-camera-mobile-phone/
- **[24] Lee et al., CVPR 2019** DMENet — defocus map estimation 的 SOTA baseline。
- **[4] NTIRE 2021 Challenge** Abuolaim et al. (CVPR Workshop) — 跟这篇同期的一个 DP deblur challenge 报告。https://openaccess.thecvf.com/content/CVPR2021/papers/Abuolaim_NTIRE_2021_Challenge_for_Defocus_Deblurring_Using_Dual-Pixel_Images_CVPR_2021_paper.pdf
- **[25] Lee et al., CVPR 2021** Iterative Filter Adaptive Network — 单图 defocus deblur 的另一条路。
- **[32] Pan et al., CVPR 2021** Dual pixel exploration — 同时做 depth estimation 和 image restoration。
- **[45] Vo, CVPR Workshop 2021** Attention! stay focus! — DP deblur 用 attention。

---

## 7. 我的几点 critical 观察

1. **PSF bank 是离散的，不是连续的**：48 组合的 PSF bank 在训练时随机采样，但真实 PSF 是连续变化的 (depth 连续、aperture 连续)。这是 brute-force 搜索的代价。后续如果用 MLP 或者 implicit function 把 PSF 参数化为 depth/aperture 的函数，会更优雅。

2. **SYNTHIA 数据集的局限**：CG 渲染的 city street scene，texture 和 real world 还是有 gap。Table 1 的 RDPD (synthetic only) 比 DPDNet 略低，说明 sim-to-real gap 还在。如果用 Unreal Engine 之类更高保真度的渲染器，结果可能更好。

3. **没测真正的 video**：因为没有任何相机录 DP video，作者用 Canon DSLR 模拟"4 张 capture 带 small camera motion"。这跟真实 video 的运动模式 (object motion + camera motion + rolling shutter) 还差很多。convLSTM 的潜力在真正 video 上才能完全体现。

4. **Edge loss 的设计偏启发式**：方向上不等权 (λ_x vs λ_y) 在 sensor 水平方向有 disparity 的物理直觉是对的，但数值 (0.03/0.02) 是手调的，没给 ablation on 这两个权重的具体值。

5. **没考虑 DP sensor 的 chromatic aberration**：实测 PSF 在 RGB channel 上不一样 (色散)，但本作没建模这一点。Paper Fig. S2 看到的 estimated PSF 是 luminance 上的。

6. **RDPD 比 DPDNet 参数少一半还更快**：Table 1 时间 0.3s vs 0.5s，但是这是 single image。Video setting 上 RDPD 因为 convLSTM 需要顺序处理，frame rate 会受 sequence length 影响，paper 没给 video inference time。

7. **Outdoor PSNR 反例**：作者承认 outdoor GT 有 misalignment 导致 DPDNet overfit 反而 PSNR 更高，但 paper Table 1 把 DPDNet+ 标成 "yellow second best"。这一段诚实地暴露出来，比直接无视要好，但还是说明 metric 的局限。建议未来 work 应该重新采集一个干净的 indoor+outdoor DP dataset。

---

## 8. Summary

这篇 paper 的核心贡献其实是一个 **sim2real 的工程化案例**：把 DP sensor 的成像链路 (thin lens → CoC → donut-shaped PSF → radial distortion → signal-dependent noise) 一段一段建模，用 CG 渲染的 SYNTHIA 当作 GT 源，合成出"看起来真"的 DP 数据。再用一个 encoder-decoder + convLSTM 的网络 (RDPD) 同时处理单图和 video，配 radial distance patch 和 multi-scale edge loss 两个 trick 把性能往上推。

价值最大的是：**让 smartphone 这种 fixed aperture 的设备也能用 DP defocus deblur**——以前因为没 GT 这条路堵死了，现在用 synthetic 数据训练一次就行。这是工程意义上的解锁。

References:
- Paper: https://arxiv.org/abs/2112.01997
- Project page: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel
- DPDNet 前作: https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel
- Abuolaim 主页: https://www.eecs.yorku.ca/~abuolaim/
