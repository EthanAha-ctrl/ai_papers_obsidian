---
source_pdf: PPISP.pdf
paper_sha256: 986b3f3921a0f74dee9d80cd2959eedb67df6301ff51982377be58eb79329fa0
processed_at: '2026-08-06T05:32:18-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 PPISP

好，我换个方式讲，假装咱们在咖啡店聊这个 paper。

---

## 一句话说清楚这论文干嘛

**你拍一圈照片想重建 3D 场景，结果每张照片亮度颜色都不一样——有的暗有的亮有的偏蓝有的偏黄。以前的做法是给每张照片配一个"万能补丁"去硬拟合，但补丁太强就把场景本身的信息也吃进去了，换个新视角就崩。这篇 paper 说：咱别瞎拟合了，老老实实把相机的物理过程拆开建模，再训练一个小网络当"自动测光表"，新视角直接让它猜该用什么参数。**

---

## 为什么照片会不一致？

想象你拿手机拍一圈。手机里有个 ISP 芯片，每拍一张都会自动决定：
- **曝光多少**（画面亮还是暗）
- **白平衡怎么调**（偏冷还是偏暖）
- 镜头边缘有暗角
- 传感器把光信号转成数字颜色的曲线（非线性）

这些设置每帧都在变，因为相机的 auto 模式在根据当前画面内容实时调整。所以你拍一圈回来，30 张照片亮度颜色各不相同。

NeRF/3DGS 的基本假设是"同一个 3D 点在所有视角下 radiance 一致"。这个假设被破坏了。

---

## 以前的土办法

### 办法 1：NeRF-W 的 GLO latent vector
给每张图配一个 256 维的 latent 向量，让网络自己学去。问题是这个向量什么都能塞——不光塞了曝光色温，还把场景几何、材质都塞进去了。**等于给每张图开了个后门，想怎么解释就怎么解释。** Novel view 没有对应的 latent 向量，咋办？不知道。

### 办法 2：Affine color transform
给每张图学一个 $3 \times 4$ 的仿射变换。比 GLO 约束强一点，但还是太自由，而且把"亮度"和"颜色"搅在一起——你想单独调曝光，白平衡也跟着动。

### 办法 3：BilaRF 的 bilateral grid
给每张图学一个 per-pixel 的局部仿射变换。**容量巨大**，training views 上 PSNR 能到 26.87，novel views 上只有 19.78——掉了 7 dB！这就是典型的过拟合：网格太灵活，把场景里每个像素的细节都吸进去了，换个视角全错。

### 办法 4：ADOP
物理建模思路，但 CRF 用 25 个离散节点插值，优化时会退化成非单调（绿色通道上下半段反转），而且白平衡用 per-channel gain，跟曝光耦合在一起分不开。

---

## PPISP 的思路：别贪，用物理约束自己

核心哲学就一句话：**容量小不可怕，可怕的是容量用错地方。**

把相机成像过程拆成四步，每步都用物理上正确的、参数极少的函数：

### 第 1 步：Exposure（曝光）

$$\mathbf{I}^{\text{exp}} = \mathbf{L} \cdot 2^{\Delta t}$$

就一个标量 $\Delta t$，单位是 stop。$\Delta t = 1$ 就是亮一倍，$\Delta t = -1$ 就是暗一倍。这跟摄影师说的 EV 值完全对应。

**为什么用 $2^{\Delta t}$ 不用 $\Delta t \cdot \mathbf{L}$？** 因为曝光在物理上是乘法关系——光子数翻倍就是亮度翻倍，不是加法。用 2 的幂让参数空间对齐摄影师直觉，也让 EXIF metadata 可以直接拿来用。

每帧一个 $\Delta t$，因为是 per-capture 的。

### 第 2 步：Vignetting（暗角）

$$v(r) = 1 + \alpha_1 r^2 + \alpha_2 r^4 + \alpha_3 r^6$$

$r$ 是像素到光学中心的距离，$\alpha_1, \alpha_2, \alpha_3$ 是三个系数。物理上 vignetting 一定是中心亮边缘暗的径向衰减，所以这个多项式刚好够用。每个颜色通道独立一组参数（因为不同波长的光在镜头里衰减不一样），per-camera 共享。

正则化还硬性约束 $\alpha_j \le 0$——**vignetting 物理上不可能是"边缘更亮"，所以这个约束就是物理常识写进 loss 里。**

### 第 3 步：Color Correction（白平衡 + 色彩校正）

这步最精妙。ADOP 的做法是给 R、G、B 三个通道各乘一个 gain：$\mathbf{I} \cdot (g_R, g_G, g_B)$。问题是你乘三个数的时候，既改了色温也改了亮度——你想只调白平衡不动曝光？做不到。

PPISP 的做法：先把 RGB 转成"色度坐标 + 强度"——也就是 $R/(R+G+B)$, $G/(R+G+B)$, 还有 $I = R+G+B$。在这个空间里，色度和强度是分开的。然后只对色度部分做一个 $3 \times 3$ 的 homography 变换，最后把强度归一化回去。

**直觉理解**：色度坐标描述的是"这个颜色偏红还是偏蓝"，强度描述的是"多亮"。你在色度空间做变换只改变色温不改变亮度，因为最后把强度拉回去了。这样就解耦了。

用 4 个控制点（R、G、B 三原色 + 白点）的 source→target 对应来构造 homography，每个控制点 2 个 offset 参数，总共 8 个参数。论文证明这等价于经典的 4-point DLT 算法。

### 第 4 步：CRF（相机响应函数）

$$
f_0(x) = \begin{cases} 
a (x/\xi)^\tau, & x \le \xi \\
1 - b((1-x)/(1-\xi))^\eta, & x > \xi
\end{cases}
$$

这是一个 S 形曲线：$\xi$ 是拐点位置，$\tau$ 控制暗部对比度，$\eta$ 控制亮部对比度。最后再套一个 gamma：$[f_0(x)]^\gamma$。

每通道 4 个参数，3 通道共 12 个，per-camera 共享。

**关键：这个函数 by design 单调递增。** 你不需要 smoothness loss 去强迫它单调——参数化本身就保证了。ADOP 的 25 个离散节点会优化成上下反转的怪物，PPISP 不会。

---

## 参数分配的逻辑

| 参数类型 | per-frame | per-camera |
|---------|-----------|------------|
| Exposure offset $\Delta t$ | ✓ 每帧一个 | |
| Color correction $\Delta \mathbf{c}_k$ | ✓ 每帧 8 个 | |
| Vignetting $\boldsymbol{\mu}, \boldsymbol{\alpha}$ | | ✓ 每相机 15 个 |
| CRF $\tau, \eta, \xi, \gamma$ | | ✓ 每相机 12 个 |

**逻辑**：exposure 和 white balance 是拍摄时变的（摄影师或 auto 模式每张都在调），所以 per-frame。Vignetting 和 CRF 是相机硬件物理特性，不会变，所以 per-camera。

这个 disentangle 是论文的核心贡献之一。Supplementary 里有个漂亮实验：在同一台相机不同 sequence 上跑，recover 出来的 vignetting 曲线和 CRF 曲线几乎完全重合——证明它们真的捕获了相机硬件特性，没有跟场景信息混在一起。

---

## Controller：最聪明的部分

训练阶段每帧有自己的 $\Delta t$ 和 $\Delta \mathbf{c}_k$，但 novel view 没有对应帧，参数从哪来？

**思路**：真实相机的 auto-exposure 和 auto-white-balance 也是"看画面内容决定参数"的。那咱也训一个小网络，输入是 rendered radiance，输出是这 9 个参数。

架构超简单：
1. 1×1 conv + pooling 把图像压成 $5 \times 5 \times 64$ 的特征图
2. Flatten 成 1600 维向量
3. 3 层 MLP，每层 128 神经元
4. 两个 linear head：一个输出 $\Delta t$，一个输出 8 个 $\Delta \mathbf{c}_k$

**$5 \times 5$ grid 是在模仿真实相机的测光区域分区**。相机测光也是把画面分成几个区域算统计量再决定曝光。这个类比很贴切。

训练时 freeze 场景重建和 per-camera 参数，只训 controller，loss 跟第一阶段一样。Controller 学到的是"看到这种 radiance 分布 → 应该输出这些 ISP 参数"的映射。这个映射是 per-camera 的（不同相机有不同的 AE/AWB 策略），但跨帧共享。

Fig. 3 的可视化特别直观：caterpillar 序列里，controller 预测的 $\Delta t$ 跟画面亮度形成合理的负相关——画面暗的帧，controller 说"该加曝光"；画面亮的帧，controller 说"该减曝光"。跟真实 auto-exposure 行为一模一样。

---

## 为什么容量小反而赢？

Table 5 是整篇论文最 insightful 的表：

| 方法 | Training PSNR | Novel View PSNR | Gap |
|------|--------------|-----------------|-----|
| BilaRF | 26.87 | 19.78 | **7.09** |
| ADOP | 26.08 | 20.28 | 5.80 |
| PPISP | 25.85 | **24.62** | **1.23** |

BilaRF training 上最高，novel view 上最低，gap 7 dB。PPISP training 上反而最低，但 novel view 最高，gap 只有 1.23 dB。

**这就是 bias-variance tradeoff 的教科书演示**。BilaRF 的高容量让它把 scene-specific 的细节都吸进 per-frame 参数里了——这些细节在新视角是错的。PPISP 的物理约束让它只能解释"相机 ISP 造成的变化"，scene 信息被迫留在 radiance field 里，所以 transfer 到新视角没问题。

混合实验更绝：给 PPISP 加上 BilaRF 的 per-frame bilateral grid，training PSNR 涨到 26.66，但 novel view 反而掉到 23.52——比纯 PPISP 的 24.62 还低。**加容量有害无益。**

---

## 评估协议的隐性贡献

以前所有 NeRF-W 系列工作报 PSNR 时，都偷偷做了 affine alignment——拿 GT novel view 跟渲染结果做最优仿射变换对齐后再算指标。理由是"我们只关心场景重建质量，不关心 appearance"。

但问题是：(1) 真实部署时没有 GT；(2) 这个 alignment 把方法之间的差异也一起"对齐"掉了，BilaRF 的烂泛化被 alignment 掩盖了。

PPISP 因为有 controller，可以直接输出 novel view 的 ISP 参数，不需要 GT 做 alignment。所以论文同时报 PSNR 和 PSNR-CC（affine aligned）两个指标。看 PPISP 的 PSNR（不 alignment）甚至比 BilaRF 的 PSNR-CC（alignment 后）还高——**你作弊都不如我不作弊。**

---

## 一句话总结

**用物理知识给可优化参数画个圈，让它们只能解释相机造成的变化；再训个小网络当"测光表"，新视角自己猜参数。容量小+物理约束=泛化好，这是老道理在神经渲染里的新胜利。**

---

## Reference

- 项目: https://research.nvidia.com/labs/sil/projects/ppisp/
- 代码: https://github.com/nv-tlabs/ppisp
- NeRF-W: https://arxiv.org/abs/2008.03924
- BilaRF: https://arxiv.org/abs/2405.04924
- ADOP: https://arxiv.org/abs/2110.06638
- Mip-NeRF 360: https://arxiv.org/abs/2111.11405
- Grossberg & Nayar CRF theory: https://www1.cs.columbia.edu/CAVE/publications/pdfs/Grossberg_TPAMI_2003.pdf
- Finlayson Color Homography: https://arxiv.org/abs/1612.01564
- Goldman Vignetting: https://ieeexplore.ieee.org/document/5404336

---

# PPISP: 物理可解释的 ISP 补偿与控制

## 1. 问题动机与核心洞察

这篇 NVIDIA 的工作针对一个长期被 NeRF/3DGS 社区"半隐藏"的问题：**多视角重建中的 photometric inconsistencies**。当你用 internet photo collection 或者多相机阵列采集时，每张图片都经过各自相机的 ISP (image signal processing) 管线处理，包含 auto-exposure、auto-white-balance、vignetting、CRF 等等。这些处理破坏了多视角 radiance consistency 的基本假设。

现有方法的痛点：
- **NeRF-W 的 GLO latent vectors** ([Martin-Brualla et al. CVPR 2021](https://arxiv.org/abs/2008.03924))：高维 latent 容易 entangle scene geometry 和 reflectance
- **Affine color correction (URF, [Rematas et al. CVPR 2022](https://arxiv.org/abs/2112.14324))**：太简单，无法建模 vignetting 等空间变化
- **BilaRF 的 bilateral grids** ([Wang et al. SIGGRAPH 2024](https://arxiv.org/abs/2405.04924))：per-pixel affine，容量大但严重过拟合 training views，对 novel views 泛化差
- **ADOP** ([Ruckert et al. SIGGRAPH 2022](https://arxiv.org/abs/2110.06638))：物理建模但 CRF 用 25 个离散 nodes，可能退化非单调

而评估协议本身也有问题：所有 prior work 都假设能拿到 novel view 的 GT，然后用 affine transform、quadratic polynomial alignment 等"作弊式"矫正后再算 PSNR。这在真实部署中不可能。

PPISP 的核心思路：**把 ISP 物理建模成一个低容量、可微、可解释的 post-processing pipeline**，并把 sensor-intrinsic (vignetting, CRF) 与 capture-dependent (exposure, white balance) **disentangle**。关键创新是引入一个 **controller**——类似于真实相机中的 auto-exposure / auto-white-balance 模块——直接从 rendered radiance 预测 per-frame 参数，从而让 novel view 评估不再需要 GT。

---

## 2. Pipeline 架构总览

PPISP 把 image formation $\mathcal{F}(\mathbf{L}; \Theta)$ 拆成四个串联模块：

$$
\mathbf{L} \xrightarrow{\text{Exposure}} \mathbf{I}^{\text{exp}} \xrightarrow{\text{Vignetting}} \mathbf{I}^{\text{vig}} \xrightarrow{\text{Color Correction}} \mathbf{I}^{\text{cc}} \xrightarrow{\text{CRF}} \mathbf{I}
$$

前三个模块在 **scene radiance 线性空间**操作，最后一个 CRF 提供非线性映射，呼应 Grossberg & Nayar 的经典相机理论 ([What is knowable? TPAMI 2003](https://www1.cs.columbia.edu/CAVE/publications/pdfs/Grossberg_TPAMI_2003.pdf))。

训练分两阶段：
1. **Phase 1 (30k iter)**：radiance field reconstruction（3DGUT 或 GSplat）+ ISP 所有 per-frame / per-sensor 参数联合优化
2. **Phase 2 (5k iter)**：freeze 重建结果和 per-sensor 参数，训练 controller

---

## 3. 四个模块的公式精解

### 3.1 Exposure Offset

$$
\mathbf{I}^{\text{exp}} = \mathcal{E}(\mathbf{L}; \Delta t) = \mathbf{L} \cdot 2^{\Delta t}
$$

变量含义：
- $\mathbf{L} \in \mathbb{R}^{H \times W \times 3}$: rendered scene radiance（从 3DGUT 或 3DGS 出来的 raw 辐射度）
- $\Delta t \in \mathbb{R}$: per-frame optimizable exposure offset，单位是 stop（EV，exposure value）

**为什么用 base-2 而不是自然对数或线性 scale？** 这是模仿摄影学中"曝光值"的物理定义——一个 stop 对应光量翻倍或减半。用 $2^{\Delta t}$ 让参数空间与 photographers 直觉对齐，也让 metadata（EXIF 中的 relative exposure）可以直接 init 或 concat 进 controller。Supplementary Sec. A.3 验证：在 HDR-NeRF 数据集上自由优化 $\Delta t$，结果与 EXIF metadata 高度对齐（Fig. 9），证明这个参数化是 identifiable 的。

Exposure 是 **per-frame** 参数——每帧一个标量，因为它由 shutter time / aperture / ISO 决定，每次拍摄都可能变。

### 3.2 Vignetting

$$
\mathbf{I}^{\text{vig}} = \mathcal{V}(\mathbf{I}^{\text{exp}}; \boldsymbol{\mu}, \boldsymbol{\alpha}) = \mathbf{I}^{\text{exp}} \cdot v(r; \boldsymbol{\alpha})
$$

$$
v(r) = \text{clip}_{(0,1)}\left(1 + \alpha_1 r^2 + \alpha_2 r^4 + \alpha_3 r^6\right), \quad r = \|\mathbf{u} - \boldsymbol{\mu}\|_2
$$

变量含义：
- $\boldsymbol{\mu} \in \mathbb{R}^2$: optical center（图像上的光学中心位置，可优化）
- $\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \alpha_3) \in \mathbb{R}^3$: 多项式系数，控制 falloff 强度
- $\mathbf{u} = (i, j)$: pixel location
- $r$: pixel 到 optical center 的距离

这个 6 次多项式模型出自 Goldman 的经典工作 ([Vignette Calibration TPAMI 2010](https://ieeexplore.ieee.org/document/5404336))。**关键点：vignetting 是 chromatic 的**，每个 color channel 有独立的 $\boldsymbol{\alpha}$ 和 $\boldsymbol{\mu}$（即每个相机 sensor 共享一组参数），所以参数量是 $3 \times (2+3) = 15$ per camera。

这是 **per-sensor** 参数——同一台相机所有帧共享，因为镜头和 sensor 物理特性不变。Supplementary Sec. A.2 (Fig. 8) 显示：在同一相机的不同 sequence 上跑，recover 出来的 vignetting 曲线几乎完全重合，证明它真的 disentangle 了 scene radiance。

正则项 $\mathcal{L}_{\text{vig}}$ 软性约束 $\alpha_j \le 0$（vignetting 物理上只可能是 falloff，不可能是 brightness boost），并对中心位置 $\boldsymbol{\mu}$ 加 $\ell_2$ penalty。

### 3.3 Color Correction（最 math-heavy 的部分）

这是 PPISP 与 ADOP 的核心差异。ADOP 用 per-channel white point gains（$g_R, g_G, g_B$），这会让 white balance 和 exposure **耦合**——因为 scaling 三个 channel 既改变总亮度也改变色温。PPISP 借鉴 Finlayson 的 color homography 理论 ([Finlayson TPAMI 2017](https://arxiv.org/abs/1612.01564))，在 **RG chromaticity + intensity** 空间做 3×3 homography，然后 **intensity 归一化**回去。

**核心变换**：
$$
\mathbf{I}^{\text{cc}} = \mathcal{C}(\mathbf{I}^{\text{vig}}; \{\Delta \mathbf{c}_k\}_{k \in \{R,G,B,W\}}) = h(\mathbf{I}^{\text{vig}}; \mathbf{H})
$$

第一步：把 RGB 转成 RGI 空间。设 $\mathbf{C} \in \mathbb{R}^{3 \times 3}$ 为 RGB→RGI 转换矩阵，$I = R + G + B$，则 $\mathbf{C} \mathbf{x} = (R/I, G/I, I)^T$（其中 $R/I, G/I$ 是 RG chromaticity）。

第二步：应用 homography $\mathbf{H}$。

第三步：intensity normalization，把 intensity 拉回来：
$$
n(\mathbf{x}; \mathbf{H}) \doteq \frac{\mathbf{x}_R + \mathbf{x}_G + \mathbf{x}_B}{[\mathbf{H} \cdot \mathbf{C}\mathbf{x}]_3 + \varepsilon}
$$

分子是原 RGB 的 intensity sum，分母是 homography 之后的 intensity（即 $[\mathbf{H} \cdot \mathbf{C}\mathbf{x}]_3$ 这一行）。$\varepsilon$ 防 0 除。最终输出：
$$
h(\mathbf{x}; \mathbf{H}) \doteq \mathbf{C}^{-1}\left(n(\mathbf{x}; \mathbf{H}) \cdot (\mathbf{H} \cdot \mathbf{C}\mathbf{x})\right)
$$

**Intuition**：homography 在 chromaticity 空间操作改变 color temperature / tint，但 normalization 把 intensity 还原——所以白平衡不会影响 exposure，exposure 也不会影响白平衡。Supplementary Fig. 6 用 Pearson Correlation Coefficient (PCC) 量化验证：PPISP 的 $\Delta t$ 与 $\Delta \mathbf{c}_W$ 的 PCC 显著低于 ADOP 的对应量。

**$\mathbf{H}$ 的构造**：用 4 对 source→target chromaticity correspondences。Source 固定为三原色 + 中性白：
$$
\mathbf{c}_{s,R} = (1,0)^T, \quad \mathbf{c}_{s,G} = (0,1)^T, \quad \mathbf{c}_{s,B} = (0,0)^T, \quad \mathbf{c}_{s,W} = (1/3, 1/3)^T
$$

Targets 是 sources 加可优化 offsets $\Delta \mathbf{c}_k$：
$$
\mathbf{c}_{t,k} = \mathbf{c}_{s,k} + \Delta \mathbf{c}_k, \quad k \in \{R, G, B, W\}
$$

总共 $4 \times 2 = 8$ 个参数（每个 $\Delta \mathbf{c}_k \in \mathbb{R}^2$）。把 source 和 target 三原色 lift 到齐次坐标，堆叠成 $\mathbf{S}, \mathbf{T} \in \mathbb{R}^{3 \times 3}$。然后构造约束矩阵：
$$
\mathbf{M} \doteq [\tilde{\mathbf{c}}_{t,W}]_\times \mathbf{T}
$$
其中 $[\cdot]_\times$ 是 skew-symmetric cross-product matrix。$\mathbf{k} \in \mathbb{R}^3$ 通过任意两行 independent rows $i, j$ 叉乘得到：
$$
\mathbf{k} \propto \mathbf{m}_i \times \mathbf{m}_j
$$

最后：
$$
\mathbf{H} = \mathbf{T} \, \text{diag}(\mathbf{k}) \, \mathbf{S}^{-1}, \qquad \mathbf{H} \leftarrow \frac{\mathbf{H}}{[\mathbf{H}]_{3,3}}
$$

Supplementary Sec. B.1 证明这个构造 **等价于 4-point DLT** (Direct Linear Transformation)——经典 homography 估计方法。当 targets 等于 sources 时 $\mathbf{H}$ 退化成 identity。

**Preconditioning**：因为 RGI 转换让 $\Delta \mathbf{c}_B$ 与 blue channel 梯度强相关，且 image 对 white point offset 比对 primary offsets 更敏感，作者用 ZCA preconditioning ([Kessy et al. 2018](https://arxiv.org/abs/1512.00809)) decorrelate 8 维 chromaticity offset 向量，分成四个 2×2 block 处理（不用完整 8×8）。

### 3.4 Camera Response Function (CRF)

$$
\mathbf{I} = \mathcal{G}(\mathbf{I}^{\text{cc}}; \tau, \eta, \xi, \gamma)
$$

每 channel 四个参数，灵感来自 Grossberg & Nayar ([Modeling CRF space TPAMI 2004](https://www.cs.columbia.edu/CAVE/publications/pdfs/Grossberg_TPAMI_2004.pdf))。基本 S-curve：
$$
f_0(x; \tau, \eta, \xi) = \begin{cases} 
a \left(\frac{x}{\xi}\right)^\tau, & 0 \le x \le \xi \\
1 - b\left(\frac{1-x}{1-\xi}\right)^\eta, & \xi < x \le 1
\end{cases}
$$

变量含义：
- $x$: input intensity（0~1 之间）
- $\xi \in (0, 1)$: inflection point（拐点位置）
- $\tau > 0$: 左半段 power（控制暗部对比度）
- $\eta > 0$: 右半段 power（控制亮部对比度）
- $a, b$: 端点匹配常数

为了 $C^1$ 连续（在 $\xi$ 处一阶导匹配）：
$$
a = \frac{\eta \xi}{\tau(1-\xi) + \eta \xi}, \quad b = 1 - a
$$

最后加 gamma：
$$
\mathcal{G}(x; \tau, \eta, \xi, \gamma) = [f_0(x; \tau, \eta, \xi)]^\gamma
$$

$\gamma$ 是 final gamma correction。总参数：4 per channel × 3 channels = 12 per camera。

**为什么这个比 ADOP 的 25-node linear interpolation 好？** PPISP 的 CRF **by design 单调递增且 smooth**，不需要 smoothness loss。ADOP 的 25 个离散节点在 large photometric variation 下可能退化成 Fig. 7 第三行那种 "split-and-reverse" 状态——green/red 通道上下半段反转，违反单调性。这种 degenerate CRF 在特定 exposure 下还能产出看起来 OK 的图，但只要 controller 改 exposure offset，立刻出现强烈 color artifacts。

CRF 是 **per-sensor** 参数。

---

## 4. Per-Frame Controller（核心创新）

所有 per-frame 参数（exposure offset $\Delta t$ + 8 个 color offsets $\Delta \mathbf{c}_k$）只在某个特定相机 pose 下有意义。Novel view 没有对应的 capture，参数怎么设？Prior work 假设有 GT 然后做 affine alignment——这是评估作弊。

PPISP 引入 controller：
$$
(\Delta t, \{\Delta \mathbf{c}_k\}_{k \in \{R,G,B,W\}}) = \mathcal{T}(\mathbf{L})
$$

**架构**（Supplementary Sec. B.2）：
1. **Feature extractor**: 
   - 1×1 conv: 3→16 channels
   - max pool ×3 spatial: $H \times W \to H/3 \times W/3$
   - ReLU
   - 1×1 conv: 16→32, ReLU
   - 1×1 conv: 32→64, output $\mathbf{F} \in \mathbb{R}^{H/3 \times W/3 \times 64}$
2. **Spatial aggregation**: adaptive average pooling to $5 \times 5$ grid → $\mathbf{F}_{\text{pool}} \in \mathbb{R}^{5 \times 5 \times 64} = 1600$ dim
3. **Metadata concat** (optional): e.g. EXIF exposure compensation
4. **Regressor**: MLP with 3 hidden layers, 128 neurons each, ReLU
5. **Two parallel linear heads**: 一个输出 $\Delta t$，一个输出 8 个 $\Delta \mathbf{c}_k$

**Intuition**：5×5 grid 模仿真实相机的 **metering zones**——相机测光时也是把图像分区域算统计量，然后决定 exposure。1×1 conv + pooling 是一个轻量的 "image statistics extractor"，不是 dense per-pixel 处理。整个 controller 极其轻量（0.74ms），且参数共享于 per-camera（不是 per-frame）。

训练时 freeze scene + ISP 参数，只更新 controller 权重，loss 与 phase 1 相同（photometric loss on training views）。Controller 在 training views 上学到"看到这种 radiance 分布应该输出这些 ISP 参数"，然后直接 transfer 到 novel views。

Fig. 3 是个漂亮的可视化：caterpillar sequence 中，controller 对每帧预测的 $\Delta t$ 跟图像内容（暗的帧→更亮 exposure，亮的帧→更暗 exposure）形成合理的负相关，就像真实 auto-exposure 行为。

---

## 5. 正则化细节

由于 scene radiance 和 ISP 参数之间存在 gauge ambiguity（亮度可以放在 radiance 也可以放在 exposure），需要正则打破对称：

$$
\mathcal{L}_b = \lambda_b \, \mathcal{L}_{\delta=0.1}\left(\frac{1}{F}\sum_{f=1}^{F} \Delta t^{(f)}\right)
$$
曝光 offset 跨帧 mean 趋零，Huber loss，$\delta = 0.1$ stop。

$$
\mathcal{L}_c = \lambda_c \sum_{k \in \{R,G,B,W\}} \mathcal{L}_{\delta=0.005}\left(\frac{1}{F}\sum_{f=1}^{F} \Delta \mathbf{c}_k^{(f)}\right)
$$
Color offset 跨帧 mean 趋零，$\delta = 0.005$。

$$
\mathcal{L}_{\text{var}} = \lambda_{\text{var}} \sum_{m \in \{\text{vig, crf}\}} \text{Var}_k(\boldsymbol{\theta}_{m,k})
$$
Vignetting 和 CRF 的 cross-channel 方差惩罚——鼓励三个 channel 的曲线相似，避免 CRF 把 color shift "吃"进去。

$$
\mathcal{L}_{\text{vig}} = \lambda_v\left(\|\boldsymbol{\mu}_k\|_2^2 + \sum_j [\alpha_j]_+^2\right)
$$
Vignetting 物理 constraint：optical center 靠近图像中心，$\alpha$ 不应为正（不应该是 brightness boost）。

权重 (Table 7)：$\lambda_b = 1.0, \lambda_c = 1.0, \lambda_{\text{var}} = 0.1, \lambda_v = 0.01$。

---

## 6. 实验结果解读

### 6.1 主结果 (Table 1)

PPISP + controller 在 5 个数据集上的 PSNR 全面 SOTA。重点看几个数字：

**Tanks & Temples**：3DGUT 基线 22.86，BilaRF 19.78（**严重退化**，因为高容量过拟合），ADOP 20.28，PPISP (w/o ctrl) 21.52，PPISP (w/ ctrl) **24.62**。Controller 单独贡献了 3.1 dB！这是论文最 dramatic 的提升。

**PPISP-AUTO**（他们自己采集的 3 相机数据集，故意用 auto exposure + auto WB 让 controller 难做）：3DGUT 22.05，PPISP (w/ ctrl) 22.87。这里 controller 提升主要来自 PPISP 帮助 disentangle 不同 camera 的特性。

**与 PSNR-CC 比较**：BilaRF 的 PSNR-CC (25.63) 仍然低于 PPISP 的原始 PSNR (24.12)——即 PPISP 不做 affine alignment 就比 BilaRF 作弊后还强。这是评估协议层面的胜利。

### 6.2 Ablation (Table 2)

| 配置 | NV PSNR |
|------|---------|
| Full | 24.62 |
| - exposure | 23.33 (掉 1.29) |
| - vignetting | 24.08 (掉 0.54) |
| - color correction | 24.27 |
| - CRF | 24.36 |

**Exposure 模块最重要**——这跟物理直觉一致：auto-exposure 在真实相机中是变化最剧烈的成分。Vignetting 排第二，说明 per-sensor 的 lens falloff 修正对多相机场景至关重要。

### 6.3 容量 vs 泛化 (Table 5) — 这是最 insightful 的表

| 方法 | TV PSNR | NV PSNR | Gap |
|------|---------|---------|-----|
| BilaRF + PC (per-camera) | 26.83 | 21.80 | 5.03 |
| PPISP + BilaRF (混合) | 26.66 | 23.52 | 3.14 |
| BilaRF (原始) | 26.87 | 19.78 | **7.09** |
| ADOP | 26.08 | 20.28 | 5.80 |
| PPISP | 25.85 | **24.62** | **1.23** |

**Key insight**：BilaRF 在 training views 上 PSNR 最高 (26.87)，但 novel views 上只有 19.78——**7 dB 的过拟合 gap**！它的 per-pixel affine 容量太大，把 scene-specific 信息都吸进去了，到 novel view 就崩。PPISP 的 gap 只有 1.23 dB，因为它的容量被物理模块严格限制：exposure 是 1 个标量，CRF 是 12 个数，无法吸收 scene 信息。

混合实验更妙：把 BilaRF 加到 PPISP 上（最后一行加 per-frame bilateral grid）→ TV 26.66 但 NV 23.52，比纯 PPISP 的 NV 24.62 还低。说明 **加容量在 training 上有用，在 generalization 上有害**——这是 PPISP 设计哲学的强力验证。

### 6.4 Runtime (Table 4)

| 方法 | 时间 | Overhead |
|------|------|----------|
| 3DGUT 基线 | 3.24 ms | - |
| BilaRF | 1.17 ms | 36% |
| ADOP | 0.10 ms | 3% |
| PPISP (w/o ctrl) | 0.10 ms | 3% |
| PPISP (w/ ctrl) | 0.84 ms | 26% |

PPISP w/o ctrl 与 ADOP 一样快（都是简单解析函数）。加 controller 后慢一些但仍然比 BilaRF 快。

### 6.5 Metadata 利用 (Table 3)

HDR-NeRF 数据集有 exposure bracketing metadata。给 controller 喂 EXIF exposure：PPISP PSNR 17.86→34.30 (带 metadata)，远超 ADOP 的 31.27。PPISP 的 controller 设计成可以 concatenate 任意 metadata 进 MLP regressor 输入，这是个很优雅的扩展接口。

---

## 7. 与 ADOP 的精细对比（Supplementary Sec. A.1）

ADOP 与 PPISP 都是物理 ISP 建模，但有几个关键差异：

1. **White balance 解耦**：ADOP 用 per-channel gains $(g_R, g_G, g_B)$，这与 exposure offset 强相关（scaling 三个 channel 既改亮度也改色温）。PPISP 在 chromaticity 空间操作 + intensity normalization，PCC 测量显示解耦明显更好（Fig. 6）。

2. **CRF 稳定性**：ADOP 25 个 nodes + smoothness loss，paper 中提到他们不得不把 ADOP 的 CRF regularization 加强 100× 才稳定。Fig. 7 第三行显示退化情况下 ADOP CRF 会出现 channel 反转。PPISP 的 piecewise power curve **by construction** 单调递增，不需要 smoothness loss。

3. **CRF 紧凑性**：PPISP 4 参数/channel vs ADOP 25 nodes/channel。

4. **Decoupling 的下游影响**：Fig. 7 顶行演示——ADOP 的 CRF 把 color artifact "bake" 进 radiance field 来补偿白平衡耦合。当用 controller 改 exposure offset 时，artifact 显露。PPISP 因为解耦干净，radiance field 本身保持 neutral，改 exposure 不会引发 color shift。

---

## 8. 局限性

作者自己提到：
- **Training views 上可能输给 BilaRF**（因为容量限制）——这是 trade-off，不是 bug
- **忽略 local tone-mapping**：现代手机相机的 spatially-adaptive tone-mapping 无法建模（vignetting 是 global 的，color correction 也是 global affine on chromaticity）
- **忽略 lens flares**：night scenes 中的耀斑
- **Controller 依赖 radiance-ISP 参数的相关性**：如果摄影师手动 override shutter/aperture/ISO（这种情况下 exposure 与 image content 不相关），controller 学不到有用映射，必须靠 metadata

---

## 9. 我的几个延伸思考

**为什么 PPISP 这种"低容量 + 物理结构"会赢过高容量 black-box？** 这是经典的 bias-variance trade-off 在神经渲染中的体现。BilaRF 的高容量在 training view 上吸收了所有残差，包括 scene-specific 信息——这些信息在 novel view 是错的。物理结构提供了 **inductive bias**：vignetting 一定是径向衰减的，CRF 一定是单调的，exposure 一定是全局 scale。这种 prior 让少量参数 expressive enough 解释真实 ISP 现象，又不至于 overfit。

**Controller 的"auto-exposure auto-white-balance 类比"非常 strong**。真实相机里 AE/AWB 算法也是从 image statistics 预测 exposure/WB 参数。PPISP controller 学到的是这个映射的神经版本。这意味着如果训练数据中 AE/AWB 行为符合典型相机控制规律（大部分场景），controller 能 transfer；如果用户手动 override，controller 就需要 metadata 辅助。这跟真实摄影师的经验完全一致。

**评估协议的革命性意义**：以前所有 NeRF-W 系列工作都偷偷用 GT novel view 做 alignment 才报 PSNR，PPISP 第一次让"无 GT 评估"成为可能。这对 real-world deployment 是必须的——你不可能在产品里要求 GT。

**与 3DGS MCMC 的结合**：论文用 3DGUT 和 GSplat 的 MCMC 配置。MCMC 倾向于稳定地分配 gaussian，与 PPISP 的物理建模互补——MCMC 防 gaussian 数量爆炸，PPISP 防 photometric 过拟合，两者都在限制表达性以提升泛化。

**可能的延伸方向**：
- 把 controller 扩展成 RNN/transformer，处理 temporal video sequence（auto-exposure 的时间滞后建模）
- 引入 spatially-adaptive tone-mapping（用低秩 spatial basis + few coefficients）
- 跟 RawNeRF ([Mildenhall et al. CVPR 2022](https://arxiv.org/abs/2111.13679)) 结合，直接在 raw Bayer 空间建模，跳过 sRGB ISP
- 把 controller 训练成跨相机通用（meta-learning），实现 zero-shot camera transfer

---

## 参考链接

- **项目主页**: https://research.nvidia.com/labs/sil/projects/ppisp/
- **代码**: https://github.com/nv-tlabs/ppisp
- **3DGUT** (基础重建框架): https://research.nvidia.com/labs/sil/projects/3dgut/
- **GSplat** (3DGS 实现): https://github.com/nerfstudio-project/gsplat
- **NeRF-W**: https://arxiv.org/abs/2008.03924
- **BilaRF**: https://arxiv.org/abs/2405.04924
- **ADOP**: https://arxiv.org/abs/2110.06638
- **Mip-NeRF 360**: https://arxiv.org/abs/2111.11405
- **HDR-NeRF**: https://arxiv.org/abs/2111.14461
- **Tanks and Temples**: https://www.tanksandtemples.org/
- **Waymo Open Dataset**: https://waymo.com/open/
- **Grossberg & Nayar CRF theory**: https://www1.cs.columbia.edu/CAVE/publications/pdfs/Grossberg_TPAMI_2003.pdf
- **Finlayson Color Homography**: https://arxiv.org/abs/1612.01564
- **Goldman Vignetting Calibration**: https://ieeexplore.ieee.org/document/5404336
- **RawNeRF**: https://arxiv.org/abs/2111.13679

总而言之，PPISP 是把"物理可解释 ISP"这个古典计算机视觉问题，以现代可微渲染的方式注入到神经辐射场重建中。它的核心贡献不在某个单点突破，而在于**整体哲学**：用低容量物理模块换取 generalization，用 controller 把 "per-frame 优化" 转化为 "per-frame 预测"，从而绕过 novel view 评估长期依赖 GT 的困境。这个思路在 NeRF/3DGS 走向真实部署的路上是重要的一步。
