---
source_pdf: Dual Pixel Exploration.pdf
paper_sha256: d38dc9a1c523b39a5181572f6258ebe3f5ea84e539e57c5f23cdc19c1cf8b311
processed_at: '2026-08-04T00:27:38-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Dual Pixel Exploration

## 一句话概括

Dual-pixel sensor 一次拍照同时给出两张图(左眼+右眼),这两张图里既藏着 stereo disparity 又藏着 defocus blur。以前的 paper 要么只看 disparity 要么只看 blur,这篇 paper 说:**这俩其实是同一个物理量在数学上的两个 face,应该 joint model**,而且 blur 里 carry 的 depth 信息不该当噪声扔掉。

---

## 先讲清楚 dual-pixel 是个啥

你的单反相机或者手机摄像头,每个像素其实被劈成了两半——左半个 photodiode 收集从 aperture 左半边过来的光,右半个收集右半边过来的光。一次拍照就能同时拿到两张图 I_L 和 I_R,这就是 dual-pixel sensor 的物理基础。

原厂设计这玩意儿是为了 autofocus:两张图之间有微小 shift(视差),shift 大小告诉你"焦对准了没",然后镜头自动挪一下。所以 DP 这个传感器在所有 DSLR 和大部分手机里都内置了,只是大多数用户感知不到它的存在。

**关键点**:这两张图不是普通 stereo pair,它们有两个特别之处:

1. **Baseline 极小**:左右半孔径之间大概 1mm 不到,远小于传统 stereo camera。好处是 occlusion 几乎不存在;坏处是 disparity 太小,精度受限。

2. **同时有 defocus blur**:不在 focal plane 上的点,在两张图里都会糊掉。而且 blur 大小跟 depth 直接相关。

这就引出 paper 的核心 motivation:

- 之前 Garg et al. ICCV 2019、Wadhwa et al. SIGGRAPH 2018 这些工作把 DP 当 stereo pair 直接做 matching,忽略了 blur 信号;
- Abuolaim & Brown CVPR 2020 (DPDNet) 把 blur 当噪声先 deblur,扔掉了 blur 里的 depth 信息;
- Punnappurath et al. ICCP 2020 用 PSF model blur 但假设 PSF 对称,只对 constant depth 区域成立,而且分三步非 end-to-end。

这篇 paper 说:**别这么割裂地看,blur 和 disparity 在数学上是同一个量**。

---

## 最 elegant 的数学观察

paper Section 3.1 那个 Observation,我觉得是整篇 paper 最聪明的一步。

设 lens 在 X=0 平面,中心在原点;world 在 X<0;camera 在 X>0。薄透镜把 world 点 X = (X, Y, Z) 成像成 virtual world W' 里的点 X' = (X', Y', Z'),由薄透镜方程:

$$\mathbf{X}' = \frac{f}{f+X}(X, Y, Z) \tag{1}$$

变量解释:
- $X, Y, Z$ — world 点 3D 坐标(X 是沿光轴的深度方向,负值)
- $X', Y', Z'$ — virtual world 中的像点坐标
- $f$ — lens 焦距(标量)

paper 的观察:**通过 aperture 左半 A_L 上某点 C_L 看到的图 I_L,等价于以 C_L 为中心、以 F 为焦距的 pinhole 相机对 virtual world W' 的成像。**

这个观察的 magic 在于:它把"half-aperture + thin-lens + defocus blur"这个复杂的物理过程,化简成了"针孔相机看 virtual world"的标准几何模型。

**直觉上发生啥了**:world 点 X 被 lens 折射,会聚到 virtual 像点 X',然后再发散开。从 A_L 中任意一点 C_L 看过去,光线走 X → C_L → X' → focal plane 上某点 Y。反过来看,这条路径恰好就是从 C_L 出发穿过 X' 打到 focal plane 的射线,也就是 pinhole 投影。所以 I_L 就是"站在 C_L 这个针孔位置看 virtual world"的图。

A_L 要是无穷小的一个点,I_L 就是 sharp 的 pinhole 图。但 A_L 是 aperture 的左半边,有面积,所以是所有 C_L ∈ A_L 形成的 pinhole 图叠加,自然就糊了。**blur 的本质就是 aperture region 上所有 pinhole 投影的 superposition**。

---

## disparity 和 blur 怎么统一

paper Eq. (4) 是另一个关键公式:

$$T(\mathbf{X}, \mathbf{C}) = \frac{d'-F}{d'}\mathbf{C} + \frac{F}{d'}\mathbf{X}' \tag{4}$$

变量解释:
- $T(\mathbf{X}, \mathbf{C})$ — world 点 X 通过 lens 上点 C 成像到 focal plane 上的位置
- $d'$ — virtual 像点 X' 的 X 坐标,由 $d' = fd/(d-f)$ 给出
- $F$ — focal plane 到 lens 的距离(注意不是焦距 f)
- $\mathbf{C} = (0, Y_0, Z_0)$ — lens 上某点
- $\mathbf{X}'$ — virtual 像点坐标

这个公式可以拆成两项看:

**第二项 $\frac{F}{d'}\mathbf{X}'$**:跟 C 无关,就是 virtual 像点投影到 focal plane 的位置。这是"sharp"那个位置——C 取 aperture 中心时的成像位置。

**第一项 $\frac{d'-F}{d'}\mathbf{C}$**:对每个不同的 C(lens 上不同点),给一个不同的 offset。所有 C ∈ A_L 累加起来,就形成一个 blur disk。blur disk 的形状就是 A_L 的相似形,缩放比例是 $(d'-F)/d'$,位移方向由 C 决定。

**关键 moment**:看这个缩放因子 $s = \frac{d'-F}{d'}$ 的物理含义:

- $|s|$ 的大小 = blur kernel 的半径 → **defocus blur amount**
- $s$ 的符号 = 物点比焦平面远还是近 → kernel 正立或倒立
- 不同的 C 乘以同一个 s,产生不同的位移 → 不同 C 形成的 sharp 图之间错开 → **disparity**

**所以 disparity 和 blur radius 就是同一个量 s 的两种表现。**这就是 paper 把 stereo cue 和 defocus cue unify 起来的数学根源。你不能把 blur 去掉再做 stereo,因为你扔掉的是同一个 depth 信号的一半。

---

## Reblur loss: 把 forward model 编进 loss

paper 的另一个核心 contribution 是 reblur loss。直觉上是这样:

网络输出一个 sharp image $\hat{\mathbf{I}}$ 和一个 inverse depth map $\hat{\mathbf{D}}$。把这俩喂给 paper 自己写的 DP simulator(基于 Eq. (5) 那个 forward model),重新 blur 出一对 DP pair $\hat{\mathbf{B}}_{L,R}$。要求 $\hat{\mathbf{B}}_{L,R}$ 跟真实输入 $\mathbf{B}_{L,R}$ 对得上:

$$\mathcal{L}_{reb} = \frac{1}{N}\sum_{y,z}\|\mathbf{B}_{\{L,R\}}(y,z) - \hat{\mathbf{B}}_{\{L,R\}}(y,z)\| \tag{11}$$

变量:
- $\mathbf{B}_{\{L,R\}}$ — 真实输入的 DP pair(left + right)
- $\hat{\mathbf{B}}_{\{L,R\}}$ — 用网络输出的 $(\hat{\mathbf{I}}, \hat{\mathbf{D}})$ 通过物理 simulator 重新 blur 出来的 DP pair
- $N$ — 像素数
- $\|\cdot\|$ — ℓ2 norm

**这个 loss 为啥聪明**:它是个 self-consistency constraint,强制网络输出的 (image, depth) 必须"解释得通"输入的 DP pair。如果 depth 错了,simulator 用错 depth 重新 blur,得到的图就对不上输入;如果 image 有 hallucination,重新 blur 也对不上。

更深一层的好处:**这个 loss 完全不需要 GT depth 和 GT sharp image,只用输入 B 就能算**。所以可以拿来在新 domain 上做 self-supervised fine-tune,不需要任何 GT。paper Section 5.2 实验验证了这点——在 DPD-disp 数据集上,用 reblur loss 做 self-sup FT 后接近 SOTA。

这种思路其实跟 NeRF 的 volume rendering 哲学同源:把 forward imaging process 写成可微 simulator,让网络输出的 latent variables 必须通过 forward model 解释 observation 才能降 loss。也是 Monodepth2、differentiable rendering、physics-informed learning 这条线上反复出现的范式。Karpathy 你应该特别 appreciate 这种 domain knowledge 显式 encode 到 loss 里的做法,它比纯 black-box regressor 更 sample-efficient 也更可解释。

---

## DP simulator: 解决数据稀缺

DP dataset 一直是个痛点:只有 Canon 5D IV 和 Google Pixel 两类设备能拿到 DP raw,而且很难拿到配套的 GT depth。深度学习又特别缺数据。

paper 用 Eq. (5) 那个 forward model 写了个 simulator:输入是任意 RGB-D dataset(比如 NYU Depth v2),输出是合成的 DP pair + GT depth + GT sharp image。

加速 trick 是 integral image / summed-area table(Crow 1984 经典 graphics 技巧,也是 Viola-Jones 那个 detector 的核心加速手段)。具体做法是把 aperture 近似为矩形,4 个角点通过 Eq. (5) 算出 blur kernel 的边界,然后在 differential mask 上对 4 个角点写正负权重,最后做一次 2D prefix sum 就能得到所有像素的 blur 结果。复杂度从 O(n × R_size) 降到 O(n),跟 blur kernel 大小无关。

这个 simulator 的实战价值很高:NYU v2 上合成 5000 train + 500 test,pretrain 后 fine-tune 到真实 DPD-blur 数据上,PSNR 26.92 dB,甚至**超过**直接在真实数据上训练的 26.76 dB。说明 synthetic DP image 学到的 prior 比 real 小数据更 generalizable。

paper 链接: https://arxiv.org/abs/2010.12052

---

## 网络架构

pipeline 是两 stage:

```
DP pair (B_L, B_R)
    ↓
[DepthNet g(·)]  (hierarchical NAS stereo, Cheng et al.)
    ↓
coarse inverse depth D̂_c
    ↓
concat (B_L, B_R, D̂_c)
    ↓
[DeblurNet f(·)]  (encoder-decoder + multi-patch, Zhang et al.)
    ↓
{sharp image Î, refined inverse depth D̂}
    ↓
[DP Simulator] (可微 forward model)
    ↓
reblurred pair B̂_L, B̂_R
    ↓
compare with B_L, B_R → reblur loss
```

两个 stage 的 intuition:
- DepthNet 先用 stereo cue 抓一个 noisy 但大致正确的 coarse inverse depth
- DeblurNet 把 coarse depth 当 prior,联合 refine depth 和 deblur——blur 越严重的地方 depth 越偏离 focal plane,两者互相 constrain

为什么用 inverse depth(1/d)不用 depth(d)?因为 disparity 在薄透镜模型下跟 inverse depth 近似线性,learning 上更友好。Garg et al. ICCV 2019 也是这么做的。

总 loss:

$$\mathcal{L} = \mathcal{L}_{res} + \mathcal{L}_d + \mathcal{L}_{reb} \tag{8}$$

- $\mathcal{L}_{res}$:L2 image restoration loss(Eq. 9)
- $\mathcal{L}_d$:smooth ℓ1 depth loss(Eq. 10,Huber loss,对 outlier 鲁棒,Fast R-CNN 引入)
- $\mathcal{L}_{reb}$:reblur self-consistency loss(Eq. 11)

---

## 实验数据

**Deblurring(Table 1, DPD-blur)**:

| Method | PSNR | SSIM | RMSE_rel |
|--------|------|------|----------|
| EBDB | 24.82 | 0.801 | 5.74 |
| DMENet | 23.93 | 0.812 | 6.36 |
| DPDNet | 25.53 | 0.826 | 5.29 |
| Ours (无 reblur) | 26.15 | 0.827 | 4.93 |
| **Ours (有 reblur)** | **26.76** | **0.842** | **4.59** |

reblur loss 让 PSNR 涨 0.6 dB,作者定义了 RMSE_rel = RMSE/255 来解释 1 dB 改善的物理意义——相对 intensity error 降 13%。

**Depth(DPD-disp,Table 2)**:

zero-shot(用 synthetic 训练直接测)就拿到第二,仅次于专门为这 task 设计的 DPdisp。**用 reblur loss 做 self-sup FT 后接近 SOTA,完全不依赖 GT depth**。

**Ablation(Table 3, Our-syn)**:

| Output | abs_rel | rmse | PSNR |
|--------|---------|------|------|
| Ours_b (DepthNet only) | 0.149 | 1.222 | - |
| Ours_wb (DeblurNet) | 0.091 | 0.599 | 32.171 |
| Ours_reb (final) | 0.083 | 0.461 | 33.218 |

abs_rel 从 0.149 → 0.091(joint depth+deblur)→ 0.083(+reblur loss)。三个组件的有效性清晰可分。

---

## 我觉得这篇 paper 真正重要的点

它把 dual-pixel 这个传感器从"fast autofocus 硬件"重新定义成了"single-snapshot depth-from-defocus-and-disparity 传感器"。blur 和 disparity 不再是两个独立信号,在数学上被 unification 了。

更深层的 takeaway:这篇 paper 印证了一种 research philosophy,在 deep learning 时代依然有生命力——**显式建模 forward imaging process,把它可微化,然后作为 constraint 或者 loss 嵌入到网络里**。这种 domain knowledge + learning 的 hybrid 路线比纯 black-box regressor 更 sample-efficient、更可解释、也更容易迁移到新 domain。

类似哲学在别的地方也出现过:

- NeRF 的 volume rendering 不同iable forward
- Monodepth2 的 photometric consistency loss: https://arxiv.org/abs/1806.01260
- Differentiable PDE solvers in physics-informed NN
- Differentiable ray tracing for inverse graphics

paper Section 6 说未来工作要 fully self-supervised,去掉 GT depth 依赖。这跟 Monodepth2 在 monocular 那条路的演化路径类似——先 supervised 跑通,再逐步去掉 GT 用 self-consistency 替代。

---

## 相关链接

- paper: https://arxiv.org/abs/2010.12052
- code: https://github.com/panpan81/dddnet
- baseline DPDNet (Abuolaim & Brown): https://arxiv.org/abs/2005.00305
- baseline DPdisp (Punnappurath et al.): https://arxiv.org/abs/2003.12781
- Du²Net (Zhang et al.): https://arxiv.org/abs/2003.14299
- Garg et al. ICCV 2019 (DP depth learning 开山): https://arxiv.org/abs/1904.08062
- Wadhwa et al. SIGGRAPH 2018 (SDoF): https://arxiv.org/abs/1805.05365
- Monodepth2 (self-sup mono depth 哲学同源): https://arxiv.org/abs/1806.01260
- Cheng et al. (DepthNet backbone, hierarchical NAS stereo): https://arxiv.org/abs/2010.13501
- NYU Depth v2 (simulator 输入数据): http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

这篇 paper 在 dual-pixel 这个 sub-field 里把数学建模做到位了,后续 ICCV/CVPR 不少 self-supervised DP、defocus-aware stereo 工作都能看到它的影子。

---

# Dual Pixel Exploration: Simultaneous Depth Estimation and Image Restoration 深度解析

这篇来自 ANU + Australian Centre for Robotic Vision 的 paper，作者是 Liyuan Pan、Shah Chowdhury、Richard Hartley（是的，就是 Hartley 那本《Multiple View Geometry》的 Hartley）、Miaomiao Liu、Hongguang Zhang、Hongdong Li。核心 idea 非常 elegant：**把 dual-pixel sensor 看作一个 unified 的 stereo-from-defocus 问题，而不是去模糊之后做 stereo**。

paper 链接: https://arxiv.org/abs/2010.12052
code: https://github.com/panpan81/dddnet (作者声明会 release)
相关 baseline DPDNet: https://arxiv.org/abs/2005.00305
DPdisp: https://arxiv.org/abs/2003.12781
Du²Net: https://arxiv.org/abs/2003.14299

---

## 1. 核心直觉：为什么 dual-pixel 是一个被低估的传感器

Dual Pixel sensor 原本是 DSLR 和 smartphone 给 autofocus 用的：每个像素被劈成两半（左右两个 photodiode），single snapshot 内同时抓两幅图像 I_L 和 I_R，分别对应光线穿过 aperture 的左半 (A_L) 和右半 (A_R)。

这就意味着：
- **Stereo cue**：两个 half-aperture 之间有 ~1mm 级别的 baseline，形成 stereo pair，可以做 disparity → depth
- **Defocus cue**：只有在 focal plane 上的点才在两幅图里都 sharp 且 zero-disparity；不在 focal plane 上的点会同时 blur 和 shift

之前的工作要么只抓 stereo cue（Wadhwa et al. SDoF, Garg et al., Du²Net），要么只抓 defocus cue 并盲目去 blur（DPDNet by Abuolaim）。这篇 paper 的关键 claim：**这两个 cue 同源同构，应该 joint model**，而且 defocus blur 本身就 carry 了 depth 信息，不该被当作噪声丢掉。

---

## 2. 理论建模：DP image formation 的数学

### 2.1 Thin-lens model + 半孔径投影

设 lens 在平面 X=0，lens 中心在原点；world W 在 X<0 一侧；camera 在 X>0 一侧。focal length f，focal plane 在 X=F（**注意 F 不等于 f**，F 是 sensor 到 lens 的物理距离，f 是 lens 的光学焦距）。

物点 X = (X, Y, Z) 通过 lens 形成 virtual world W' 中的像点 X' = (X', Y', Z')，由薄透镜方程：

$$X' = \frac{f}{f+X}(X, Y, Z) \tag{1}$$

变量解释：
- $X, Y, Z$ — 世界点坐标（X 是沿光轴的深度方向，取负值表示在 lens 左侧）
- $X', Y', Z'$ — virtual world W' 中的像点坐标
- $f$ — lens 焦距
- $X$ 出现在分母是因为薄透镜成像公式 $1/X + 1/X' = 1/f$ 的 rearrange

### 2.2 Key observation：DP 等价于针孔相机看 virtual world

这是 paper 里最 elegant 的一步（Fig. 2 那个 observation）：

> 通过 A_L 上一点 C_L 看到的 world W 的 image I_L，**等价于**以 C_L 为中心、以 F 为焦距的 pinhole camera 对 virtual world W' 的成像。

证明 sketch：world 点 X → lens 折射 → virtual 像点 X' → 经过 C_L 的光线继续走 → 打在 focal plane X=F 上的点 Y。这条路径 X→C_L→X'→Y 在反方向上看就是从 C_L 发出的射线穿过 X' 抵达 focal plane，正好是 pinhole projection。

**这个观察的意义**：把"half-aperture + thin-lens + defocus blur"统一化简为"针孔相机对 virtual world 做 stereo"。blur 来自于 A_L 是一个 region 而不是单点：所有 C_L ∈ A_L 形成一个锥，锥截面打在 focal plane 上形成 blur disk。

### 2.3 Image synthesis from RGB-D（这是 simulator 的基础）

给定 RGB-D image I_W（视角为 lens 中心），物点 X = (X, Y, Z) 在 image 上像素 (y, z)，按 pinhole 模型 X = -d(1, y/f, z/f)，其中 d 是 depth。

由薄透镜：

$$d' = \frac{fd}{d-f} \tag{3}$$

变量：
- $d$ — 物距（world depth）
- $d'$ — 像距（virtual world 中 X' 的 X 坐标）
- $f$ — lens 焦距

把 virtual world 中的 X' = d'(1, y/f, z/f) 再投影到 focal plane（X=F）：

$$T(\mathbf{X}, \mathbf{C}) = \frac{d'-F}{d'}\mathbf{C} + \frac{F}{d'}\mathbf{X}' = \frac{d'-F}{d'}\mathbf{C} + F(y/f, z/f) \tag{4, 5}$$

变量：
- $\mathbf{C} = (0, Y_0, Z_0)$ — lens 上 A_L 区域中某一点（X=0 平面）
- $F$ — focal plane 到 lens 的距离
- $d'$ — virtual 像点的 X 坐标
- $s = F/d'$ — scaling factor，由相似三角形得出
- 第一项 $\frac{d'-F}{d'}\mathbf{C}$ — blur 中心相对于 aperture region 的位移
- 第二项 $F(y/f, z/f)$ — virtual 像点投影到 focal plane 的位置（与 C 无关）

**直觉**：当 d' = F（物点恰好在 focal plane 上）时，第一项为 0，blur 退化为一个点 → 完全 sharp。当 d' 偏离 F，第一项 scaling 非零，整个 A_L 区域被平移+缩放成一个 blur disk A_L'。这也解释了 paper Fig. 4 那个关键观察：**blur kernel 的形状是 A_L 的几何相似形**（只是 scaling 可能正可能负，负号对应 inverted）。

而且第一项的符号 $\frac{d'-F}{d'}$ 直接编码了"远/近"——d' > F 时正、d' < F 时负（kernel 倒过来）。**这就是 disparity 和 defocus 在数学上统一的源头**：blur kernel 的位移就是 disparity，kernel 的尺寸就是 blur radius。

---

## 3. DP Simulator：高效合成 DP pair

朴素做法要对每个像素 × 每个区域 R 做 4 重循环，复杂度爆炸。作者用 **integral image / summed-area table**（Crow 1984 那个经典 graphics 技巧，Viola-Jones 也用）：

把 A_L 近似为矩形，4 个角点 p_tl, p_tr, p_bl, p_br 通过 Eq. (5) 算出在 focal plane 上的位置。在 differential mask $\mathcal{T}_L$ 上对这 4 个角点写 +I/|R_L| 和 -I/|R_L|（Eq. 6）：

$$\mathcal{T}_L(\mathbf{p}_{tl}) = \frac{I(y,z)}{|\mathbf{R}_L|}, \quad \mathcal{T}_L(\mathbf{p}_{tr}) = -\frac{I(y,z)}{|\mathbf{R}_L|}, \dots \tag{6}$$

然后对所有像素的 differential mask 求和，最后做一次 2D integral（Eq. 7）：

$$\hat{\mathbf{B}}_{\{L,R\}} = \tau(\mathcal{T}_{\{L,R\}}) \tag{7}$$

这就是经典的 "4 个角加权和再做 prefix sum = 矩形区域均匀填充"的 trick。复杂度从 O(n × R) 降到 O(n)，与 blur kernel 大小无关。

**这个 simulator 是 paper 的 side contribution 但价值很大**：之前 DP dataset 只有 Canon 5D IV、Google Pixel 两类，难以拿到 GT depth。NYU Depth v2 这类已有 RGB-D 数据可以直接喂进 simulator 合成 DP pair + GT depth + GT sharp image。作者用它合成 5000 train + 500 test。

---

## 4. DDDNet 架构

```
Input DP pair (B_L, B_R)
        ↓
[DepthNet g(·)] (基于 hierarchical NAS stereo, Cheng et al. 2020)
        ↓
coarse inverse depth D̂_c
        ↓
concat with (B_L, B_R) → 4-channel-ish input
        ↓
[DeblurNet f(·)] (encoder-decoder, multi-patch, Zhang et al. 2019)
        ↓
{deblurred image Î, refined inverse depth D̂}
        ↓
[DP Simulator]  ← 反过来用
        ↓
reblurred pair (B̂_L, B̂_R)
        ↓
compare with (B_L, B_R) → reblur loss
```

**两个 stage 的 intuition**：
- DepthNet 先用 stereo cue 拿一个 noisy + 模糊的 coarse inverse depth
- DeblurNet 把 coarse depth 当 prior，联合 refine depth 和 deblur——blur 越严重的地方，depth 越偏离 focal plane，两者互相 constraint

Inverse depth（不是 depth）作为输出：因为 disparity 在薄透镜模型下与 inverse depth 近似线性（Garg et al. 2019 也是这么做的），尺度上更便于 learning。

### 4.1 三个 loss 的设计

总 loss：

$$\mathcal{L} = \mathcal{L}_{res} + \mathcal{L}_d + \mathcal{L}_{reb} \tag{8}$$

**Image restoration loss**（L2）：

$$\mathcal{L}_{res} = \frac{1}{N}\sum_{y,z}\|\mathbf{I}(y,z) - \hat{\mathbf{I}}(y,z)\| \tag{9}$$

变量：N 是像素总数，$\|\cdot\|$ 是 ℓ2 norm，I 是 GT sharp image，Î 是 deblur 输出。

**Depth loss**（smooth ℓ1）：

$$\mathcal{L}_d = \frac{1}{N}\sum_{y,z} S(\mathbf{D}(y,z) - \hat{\mathbf{D}}(y,z)) \tag{10}$$

变量：D 是 GT inverse depth，S(·) 是 smooth ℓ1 = Huber loss（小残差时是 L2，大残差时是 L1，对 outlier 鲁棒，Fast R-CNN 引入）。这也是 depth estimation 任务的标准做法。

**Reblur loss**（这是 paper 最 important 的 contribution）：

$$\mathcal{L}_{reb} = \frac{1}{N}\sum_{y,z}\|\mathbf{B}_{\{L,R\}}(y,z) - \hat{\mathbf{B}}_{\{L,R\}}(y,z)\| \tag{11}$$

变量：B 是输入 DP pair（左+右），B̂ 是把网络输出 {Î, D̂} 喂给 simulator 重新 blur 出来的 DP pair。

**这个 loss 的 deep meaning**：它是个 self-consistency loss。要求网络输出的 (Î, D̂) "解释得通"输入 B。如果 D̂ 错了，simulator 用错 depth 重新 blur，得到的 B̂ 就不会等于 B；如果 Î 有 hallucination，重新 blur 也对不上。这强制 (Î, D̂) 落在由 DP model 定义的 manifold 上，而非任意 sharp image + depth 组合。

更妙的是——这个 loss **不需要 GT depth 和 GT sharp image**，只用输入 B 就能算。这意味着可以 self-supervised fine-tune 到新 domain（paper Section 5.2 验证了这点：在 DPD-disp 上无 GT fine-tune 后超过 SDoF）。

---

## 5. 实验数据深度解析

### 5.1 Deblurring 量化结果（Table 1）

| Dataset | Method | PSNR ↑ | SSIM ↑ | RMSE_rel ↓ |
|---------|--------|--------|--------|------------|
| DPD-blur | EBDB | 24.82 | 0.801 | 5.74 |
| DPD-blur | DMENet | 23.93 | 0.812 | 6.36 |
| DPD-blur | DPDNet | 25.53 | 0.826 | 5.29 |
| DPD-blur | Ours_wb | 26.15 | 0.827 | 4.93 |
| DPD-blur | **Ours_reb** | **26.76** | **0.842** | **4.59** |
| Our-syn | DPDNet | 31.45 | 0.926 | 2.68 |
| Our-syn | Ours_reb | **33.21** | **0.956** | **2.17** |
| Our-real | DPDNet | 22.65 | 0.808 | 7.09 |
| Our-real | Ours_reb | **24.03** | **0.850** | **6.13** |

**关键 takeaways**：
1. Reblur loss 让 PSNR 涨 ~0.6 dB（DPD-blur）、~1 dB（Our-syn）、~0.04 dB（Our-real，real 上小幅）。作者还定义了一个 RMSE_rel = RMSE/255 in %，用来解释 "1dB 改善"实际意义——相对 intensity error 改善 13%/19%/12%。
2. Our-real 是作者新收集的，Canon + 多个 aperture（f/4 到 f/22），150 scenes 室内外都有。这填补了 DP dataset 单一 aperture 的空白（Google Pixel 是 fixed narrow aperture）。

### 5.2 Depth 量化结果（Table 2，DPD-disp）

| Method | AI(1) ↓ | AI(w) ↓ | 1-ρ_s ↓ | Geo Mean ↓ |
|--------|---------|---------|---------|------------|
| BTS (mono) | 0.1070 | 0.1767 | 0.6149 | 0.2686 |
| Monodepth2 (mono) | 0.1139 | 0.1788 | 0.6153 | 0.2285 |
| SDoF (DP) | 0.0875 | 0.1294 | 0.2910 | 0.1443 |
| DPdisp (DP) | 0.0481 | 0.0845 | 0.1037 | 0.0671 |
| Ours (no FT) | 0.0906 | 0.1291 | 0.2456 | 0.1207 |
| **Ours_ft (self-sup FT)** | 0.0609 | 0.0985 | 0.1026 | 0.1098 |

变量解释：
- AI(1)、AI(w) — affine invariant error（DPD-disp 只有相对 depth，无 scale）
- ρ_s — Spearman rank correlation，1-|ρ_s| 越小越好（rank ordering 准确）
- Geo Mean — 几何平均，综合指标

**关键 takeaways**：
1. 仅用 synthetic 训练、直接 zero-shot 测 DPD-disp（Ours no FT），就拿到第二，仅次于专门为这个 task 设计的 DPdisp
2. **用 reblur loss 做 self-supervised fine-tune**（Ours_ft）后超过所有 baseline，逼近 DPdisp——证明 simulator 合成数据 + reblur 自监督能 bridge domain gap

### 5.3 Ablation：simulator 数据的迁移性（Table 3 + 文中段）

Ours-syn 训练 → 直接测 DPD-blur：20.28 dB / 0.650 SSIM（无 FT）→ FT 后 26.92 dB / 0.864 SSIM / 4.51% RMSE_rel。这个数字甚至**超过**直接在 DPD-blur 上训练的版本（26.76 dB）！证明 synthetic DP image 学到的 prior 比 real 小数据更 generalizable。

只用一半 DPD-blur 数据 FT 也拿到 26.52 dB——simulator 起到了 pretraining 的 data augmentation 作用。

### 5.4 Step-by-step ablation（Table 3，Our-syn）

| Output | abs_rel | rmse | rmse_log | δ<1.25 | PSNR |
|--------|---------|------|----------|--------|------|
| Ours_b (DepthNet only) | 0.149 | 1.222 | 0.224 | 0.743 | - |
| Ours_wb (DeblurNet) | 0.091 | 0.599 | 0.123 | 0.918 | 32.171 |
| Ours_reb (final) | 0.083 | 0.461 | 0.111 | 0.936 | 33.218 |

abs_rel 从 0.149 → 0.091（joint depth+deblur）→ 0.083（+reblur loss）。每一步都有明确改进，三个组件的有效性清晰可分。

---

## 6. 与其他方法的差异点 & 局限

**vs DPdisp (Punnappurath et al. ICCP 2020)**：DPdisp 假设 PSF 对称，只对 constant depth 区域成立，且分三步、非 end-to-end。本文数学模型直接处理 spatially-varying depth，单网络 forward。

**vs DPDNet (Abuolaim & Brown 2020)**：DPDNet 盲目去 blur，丢弃 blur 里 carry 的 depth 信号。本文把 blur 当 depth 信号的一部分，joint 出 depth + deblur。

**vs Du²Net (Zhang et al. 2020)**：Du²Net 要 DP + 另一个 wide-baseline stereo camera 硬件组合，只适配 Google Pixel 的窄 aperture small DoF 场景。本文单 sensor 通用。

**Limitations（paper 自己没明说但可推断）**：
1. Simulator 假设 A_L/A_R 是矩形，真实 aperture 是圆形或更复杂形状（虽然作者说可扩展）
2. 假设所有 lens 上点都能看到 world point（无 occlusion），但实际 DP baseline 极小，这个假设基本成立
3. 模型用 thin-lens，未建模 lens distortion、chromatic aberration
4. 单 focal plane 假设；focal length F 需要事先 calibrate
5. Real-world aperture 不是严格的左右两半（DSLR 是微透镜阵列分成的 two halves，与 paper 描述的 A_L/A_R 几何分割有偏差）

---

## 7. 我的 intuition: 为什么这个工作重要

这篇 paper 把 DP 传感器的 stereo cue 和 defocus cue 在数学上做了一个 beautiful unification。关键 elegant 之处在于 Eq. (4) 那一项 $\frac{d'-F}{d'}\mathbf{C}$：

- 它乘在 $\mathbf{C}$（lens 上点位置）上 → 编码 blur kernel 的几何位移 = disparity
- 它的 magnitude $|d'-F|/d'$ → 编码 blur kernel 半径
- 它的 sign → 编码远/近（kernel inverted 或 not）

**disparity 和 blur radius 是同一个量的两个 face**。传统 stereo 方法去 blur 再做 matching，本质上扔掉了一半信号；传统 defocus 方法用 blur radius 估 depth 但忽略了 stereo disparity。本文把它们 joint learn，reblur loss 又提供了一个 self-consistency regularizer，让网络不得不 honor 物理模型。

这其实跟 NeRF 的 volume rendering 思路异曲同工：**把 forward imaging model 写成可微 simulator，然后让网络输出 latent variables（depth + sharp image）通过 simulator 重建 observation 来约束**。这种 self-consistency loss 思路是 unsupervised/self-supervised vision 的金钥匙（参考 Monodepth2 的 photometric loss 也是同源思想，https://arxiv.org/abs/1806.01260）。

Andrej 你应该会特别 appreciate 这一点：**它不是 end-to-end 黑盒 regressor，而是把 domain knowledge 显式 encode 到网络结构 + loss 里**。这正是你之前在 supervised learning、physics-informed learning 等讨论中反复强调的方向。

未来工作（paper Section 6 提到）：fully self-supervised depth + sharp image estimation，去掉对 GT 的依赖。这跟 Monodepth2 在 monocular 那条路的演化路径很类似。

---

## 8. 一些值得 follow 的 references

- Hartley & Zisserman, *Multiple View Geometry*（作者 Hartley 本人的书，stereo 几何基础）
- Wadhwa et al., SDoF (SIGGRAPH 2018): https://arxiv.org/abs/1805.05365
- Garg et al., DP depth (ICCV 2019): https://arxiv.org/abs/1904.08062
- Abuolaim & Brown, DPDNet: https://arxiv.org/abs/2005.00305
- Punnappurath et al., DPdisp: https://arxiv.org/abs/2003.12781
- Crow, Summed-area tables (SIGGRAPH 1984): simulator 加速的经典 trick
- Viola & Jones, Integral image (IJCV 2001): https://arxiv.org/abs/1806.01260 (Monodepth2)
- Cheng et al., Hierarchical NAS stereo (DepthNet backbone): https://arxiv.org/abs/2010.13501
- Zhang et al., Multi-patch deblur (DeblurNet backbone, CVPR 2019)

这篇 paper 在 2020 年的 dual-pixel 这条 sub-field 上算是把数学建模做到位的一个工作，之后 ICCV/CVPR 的很多 follow-up（包括 defocus-aware stereo、self-supervised DP）都能看到它的影子。
