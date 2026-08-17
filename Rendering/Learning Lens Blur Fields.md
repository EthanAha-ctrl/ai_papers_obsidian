---
source_pdf: Learning Lens Blur Fields.pdf
paper_sha256: c105ee9e0aa3c67654245b4e38298aac64bd1ce4c9b778da20a67e03c3e4e17c
processed_at: '2026-08-05T13:24:28-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Learning Lens Blur Fields

## 一句话版本

拿个MLP把一整台camera的"光学指纹"全记住，以后随便查询任何pixel、任何focus、任何距离下的blur长什么样。

---

## 这事儿为什么难

想象你想搞清楚一台iPhone拍照时到底blur成什么样。blur不是固定的——角落和中心不一样，对焦近和对焦远不一样，红绿蓝通道不一样，dual-pixel的左眼和右眼还不一样。这就是个**5D甚至6D的函数**。

以前三条路：

**第一条：拍点光源直接看PSF**
听起来最直觉，但点光源要做得很小很小（比pixel还小），会被diffraction污染，光效率极低，实际根本搞不了。

**第二条：每个位置独立解deconvolution**
拍一堆pattern，在sensor上切小patch，每个patch独立算"这个位置的PSF是什么"。问题是patch必须比PSF大，PSF一大patch就大，patch一大里面edge信号就稀疏，inverse problem就ill-posed。你想dense sample吧，位置太多算不完；sparse sample吧，中间插值又不准。**这就是经典的density-accuracy tradeoff**。

作者算了一笔账：iPhone 12 Pro的12MP sensor，75×100个sensor位置×15个focus×73×73 kernel×4 channel，用Joshi 2008的方法要**34天**，用Mannan 2016的方法要**1945天（5.3年）**，存下来要9.3 GB。

**第三条：用Zernike/Seidel多项式拟合**
假设lens是旋转对称的、圆形pupil的。但dual-pixel sensor根本不对称，smartphone的aperture blade造成八角形PSF也不对称，chromatic aberration更没法套。

---

## 关键insight

MLP天生就是个smooth interpolator。PSF这个物理量在sensor位置(x,y)和focus f维度上是slowly varying的smooth field，在kernel内部(u,v)维度上有结构但范围有限。这种"全局smooth、局部有结构"的信号刚好是MLP最擅长拟合的。

而且MLP把所有维度耦合在一个compact function里，不存在"每个位置独立解"的问题——**所有image的所有pixel共同supervise这个function**。某个位置信号弱不要紧，旁边位置的信息会通过MLP的smoothness prior"渗"过来。

这就绕开了density-accuracy tradeoff。Patch不再是supervision的boundary，只是计算的container。

---

## 具体怎么干

### 拍什么

拿台32寸4K显示器，对着camera显示几个pattern：

1. **Dot grid**（白点黑底+黑点白底）：用来算homography和radial distortion
2. **Black/White全屏**：用来做radiometric calibration
3. **Synthetic noise patterns**（5种频率，通过Fourier filtering生成）：真正训练用

然后对着monitor拍focal stack——7到24个focus位置，每个位置burst拍3张取平均。iPhone用app自动拍大概15分钟，SLR手动调focus ring要30分钟。

### 怎么算"如果没blur应该长什么样"

这是最难的一步。你拍了blurry image，但训练时需要blur-free image做对照。

流程：
1. 用dot grid算homography $H$（monitor坐标→sensor坐标）
2. 用dot grid算radial distortion $D$（OpenCV `calibrateCamera`）
3. 对每个focus position $f_i$，算scale transformation $S$（因为focus breathing会让magnification变化）
4. 合成 $\text{invproj} = D \circ S \circ H$

然后radiometric compensation有个很聪明的trick：拍一张全黑一张全白，因为monitor的irradiance在平面上近似uniform，convolve uniform signal等于乘PSF的integral等于signal本身。所以blur-free image就是black和white capture的affine组合，albedo做权重：

$$\hat{I}(\mathbf{x}) = (1-A) \cdot I_{\min}(\mathbf{x}) + A \cdot I_{\max}(\mathbf{x})$$

不需要拟合任何vignetting model，直接拿测量的两张图组合。

### MLP长什么样

```
Input: (x, y, f, [d], u, v)  # 5D or 6D坐标，归一化
  ↓
FC 512 + ReLU  ×6层
  ↓
FC → C channels + Sigmoid  # 输出PSF值，sigmoid保证非负
```

7层，每层512，没有positional encoding。为什么不需要？因为NeRF里radiance有view-dependent高频效应需要PE来fit，但PSF在(x,y,f)维度上是物理smooth的，spectral bias反而帮我们过滤掉noise和artifact。kernel内部(u,v)的结构靠MLP的capacity就够了，最大也就120×120。

### 怎么训练

每个iteration：
1. 随机sample一个focus $f_i$ 和sensor位置 $\mathbf{x}$
2. 在 $\mathbf{x}$ 周围sample一个 $u,v$ grid，MLP forward得到2D PSF
3. 在以 $\mathbf{x}$ 为中心的patch内（大小1.5×PSF size），用这个PSF跟blur-free image做convolution
4. 跟真实captured image算L2 loss

**Local stationarity假设**：patch内PSF不变。这是计算效率的关键——一个patch只做一次MLP forward（query中心点），然后convolution是matrix-vector product，GPU并行。

训练 $4 \times 10^6$ 到 $10^7$ iterations，Adam，$\beta_1=0.5$，$\beta_2=0.999$，lr $10^{-5}$，batch size 1。在A6000上跑14小时。

---

## 结果有多好

### Quantitative

Simulation上PSNR 32dB，比Joshi好一点，比Mannan好16dB。关键是validation lens position（训练时没见过的focus）——MLP interpolation 30.4 dB，linear interpolation 29.4 dB。在focus附近thin lens equation的defocus radius有derivative sign change，linear根本捕捉不到。

Memory：19.1 MB vs 9.3 GB，**500倍压缩**。
Time：14小时 vs 1945天，**3300倍加速**。

### Qualitative的发现

**发现1：同型号iPhone有不同的blur signature**
两台iPhone 14 Pro，各自重复拍4次，within-device的std小于between-device的RMS difference。这意味着manufacturing variability（lens element placement误差、sensor tilt、aperture shape微小差异）可以被这个方法检测到。以前没人能在commodity hardware上做到。

**发现2：Dual-pixel PSF根本不follow参数模型**
Pixel 4的left/right looking pixel，PSF的center of mass不总是按预期左右偏，边缘位置严重skewed。Punnappurath 2020的analytical model完全fit不了。

**发现3：SLR lens的特征能恢复出来**
Canon 50mm f/1.4的八角形PSF（aperture blade造成），24-70mm zoom的不规则PSF（vignetting造成），14mm和50mm f/1.2的圆滑PSF。这些都是Seidel model表达不了的。

**发现4：6D field的target distance维度变化小**
smartphone aperture小，target distance变化对PSF的影响大部分可以从5D field预测，5D可能够用了。

---

## 能拿来干嘛

### 1. Device fingerprinting
每台camera有独特的blur signature，可以做device authentication、quality control。

### 2. Photorealistic rendering
作者做了Blender add-on，把blur field当post-processing filter。渲个all-in-focus + depth map，然后用occlusion-aware model分层convolve再composite。不同lens的blur field能渲出不同的bokeh风格。

### 3. Image restoration
比Seidel model的deconvolution效果好——Siemens Star的高频细节保得更好，因为blur field PSF能capture aperture shape和chromatic aberration。

### 4. Computational photography的prior
Dual-pixel depth estimation、defocus map estimation这些task本来需要PSF model，现在可以用真实calibrated的blur field替代简化analytical model。

---

## 跟其他工作的关系

这篇属于**Neural Fields家族**，但跟NeRF的key区别是不需要positional encoding——PSF的物理smoothness帮了忙。

跟NeRF的类比：
- NeRF: $F_\theta(\mathbf{x}, \mathbf{d}) \to (\sigma, \mathbf{c})$，parameterize radiance field
- Blur field: $F_\theta(\mathbf{x}, f, \mathbf{u}) \to \text{PSF}$，parameterize optical response field

都是用MLP的interpolation property把high-dimensional physical quantity压成compact function。

跟不同iable optics（Tseng 2021, Sitzmann 2018）的区别：那些是design optics，这个是characterize现成optics。跟coded aperture（Levin 2007）的区别：那个改hardware，这个不改。

跟传统camera calibration（Zhang 2000）的区别：传统只估intrinsics和distortion，blur field给出完整的optical characterization——aberration、defocus、chromatic、diffraction、vignetting、dual-pixel disparity全包。

---

## Limitations和可能的下一步

1. 还是non-blind deconvolution——需要known patterns on monitor。能不能做blind的、in-the-wild的calibration？
2. Bayer分channel训练，没有显式model spectral PSF的连续wavelength dependence。
3. Local stationarity对very strong spatial variation可能不够。
4. 训练时间长（14小时），没尝试multi-resolution或coarse-to-fine加速。
5. 没有prior——完全靠data fit。能不能加wavefront/Zernike prior做regularization？

可能的extension：
- Joint估PSF + sharp image（blind deconvolution）
- 加wavelength维度做spectral PSF
- 加polarization维度
- 用meta-learning做few-shot device-specific fine-tune
- 把acquisition搬进main camera app做自动周期校准

---

## 我的take

这篇paper干净利落地把"光学characterization"这个传统上用parametric model或者sparse sampling解决的问题，用INR重新formulate了一遍。核心收益是scalability和expressiveness同时提升——这在以前是tradeoff。

最有意思的side discovery是device fingerprinting：原来同型号iPhone的光学差异大到可以被这个方法检测到。这暗示着computational photography的很多"奇怪表现"可能跟manufacturing variability有关，而以前没有工具能measure这个。

MLP不需要positional encoding这个发现也值得记住——说明INR需不需要PE取决于target signal的frequency content，不是所有neural field都需要PE。这个insight对做NeRF变体的人应该有用。

参考链接：
- [项目主页 blur-fields.github.io](https://blur-fields.github.io)
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [Neural Fields survey](https://arxiv.org/abs/2111.11426)
- [Joshi et al. 2008](https://www.cs.toronto.edu/~joshi/pubs/joshi_cvpr2008.pdf)
- [Punnappurath et al. 2020 dual-pixel](https://arxiv.org/abs/2003.09691)
- [Tancik et al. 2020 Fourier features](https://arxiv.org/abs/2006.10739)

---

# Learning Lens Blur Fields 深度技术解读

## 1. 核心intuition

这篇paper要解决的核心问题：**如何用continuous neural representation来parameterize一台camera的整个PSF manifold**，并且这个manifold是高维的(5D或6D)。传统方法要么用sparse discrete sampling造成density-accuracy tradeoff，要么用过于简化的analytical model损失expressiveness。作者用MLP做"光学指纹"的implicit neural representation (INR)，把所有维度interpolate起来。

关键insight：MLP本身就是一个**smooth interpolator**，对低维信号有spectral bias，刚好适合PSF这种空间上平滑变化但局部又有结构的物理量。这就解释了为什么**不需要positional encoding**——PSF的高频细节主要在`u`维度上(单个2D kernel的内部结构)，而在`(x, f)`维度上是相对低频的slowly-varying field。

---

## 2. Mathematical Formulation详解

### 2.1 Forward Image Formation Model

考虑scene point $\mathbf{p} \in \mathbb{R}^3$ 距离lens front pupil为$d$，其在sensor平面上的perspective projection为 $\tilde{\mathbf{x}} \in \mathbb{R}^2$ (忽略distortion)，考虑lens distortion后actual projection为 $\mathbf{x} \in \mathbb{R}^2$。

**Equation (1)**:
$$\mathbf{p} = \text{invproj}(\mathbf{x}, d, f, c)$$

变量含义：
- $\mathbf{p}$: 3D scene point (世界坐标)
- $\mathbf{x} \in \mathbb{R}^2$: sensor plane位置(像素坐标，考虑distortion后)
- $d$: scene-to-lens distance
- $f$: focus setting (lens位置)
- $c$: pixel type (如Bayer的R/G/B，dual-pixel的L/R)
- $\text{invproj}$: composition of $H$(homography), $S$(scale), $D$(distortion)的逆映射

注意 $f$ 和 $c$ 都会改变magnification和geometric mapping，这点在dual-pixel sensor上尤其重要——left和right photodiodes有disparity。

**Equation (2)**: blur-free image
$$\hat{I}^{(c)}(\mathbf{x}, d, f) = A^{(c)}(\mathbf{p}) \cdot E^{(c)}(\mathbf{p}, f)$$

- $A^{(c)}(\mathbf{p}) \in [0,1]$: albedo at scene point (材质属性，per-channel)
- $E^{(c)}(\mathbf{p}, f)$: total irradiance incident on sensor due to point $\mathbf{p}$ (包含vignetting, cosine-fourth falloff, solid angle等)

这个分解很巧妙：把scene的texture和camera的radiometric response分开，使得可以通过black/white calibration image来分别calibrate。

**Equation (3)**: blurry image formation (continuous convolution)
$$I^{(c)}(\mathbf{x}, d, f) = \int_{\mathcal{U}} \text{PSF}_\theta^{(c)}(\mathbf{x}, d, f, \mathbf{u}) \cdot \hat{I}^{(c)}(\mathbf{x} - \mathbf{u}, d, f) \, d\mathbf{u}$$

- $\text{PSF}_\theta^{(c)}$: parameterized by MLP weights $\theta$
- $\mathbf{u} \in \mathbb{R}^2$: displacement from $\mathbf{x}$ (kernel内部坐标)
- $\mathcal{U}$: PSF spatial support

这里有个关键假设——**local stationarity**：在小patch $\mathcal{P}$ 内，PSF对 $\mathbf{x}$ 不变，只在 $\mathbf{u}$ 上变化。这样convolution可以写成matrix-vector product，GPU parallelize。

### 2.2 Optimization Objective

**Equation (4) - Main Loss**:
$$\arg\min_{\theta} \sum_{t, c, i} \left\| I_t^{(c)}(\mathbf{x}, d_i, f_i) - \sum_{\mathbf{u}} \text{PSF}_\theta^{(c)}(\mathbf{x}, d_i, f_i, \mathbf{u}) \cdot \hat{I}_t^{(c)}(\mathbf{x} - \mathbf{u}, d_i, f_i) \right\|_2^2$$

subject to $\text{PSF}_\theta \geq 0$

变量含义：
- $t$: pattern index ($1 \leq t \leq N_T$)
- $i$: distance/focus setting pair index ($1 \leq i \leq N_I$)
- $c$: channel (pixel type)
- $\mathbf{x}$: sensor position
- $\theta$: MLP trainable weights

如果所有 $d_i$ 相同→5D blur field；不同→6D blur field。

### 2.3 Radiometric Compensation的关键技巧

**Equation (4) in paper text (the radiometric one)**:
$$I_{\min}(\mathbf{x}) = A_{\min} \int_{\mathbf{u}} \text{PSF}_\theta(\mathbf{x}, d_i, f_i, \mathbf{u}) \cdot E(\text{invproj}(\mathbf{x}, d_i, f_i) - \mathbf{u}) \, d\mathbf{u}$$
$$\approx A_{\min} \cdot E(\text{invproj}(\mathbf{x}, d_i, f_i))$$

关键近似：当 $E$ (irradiance) 在monitor平面近似constant时，convolution with PSF就是constant乘PSF的integral，等于constant本身。这就避开了不知道vignetting/irradiance分布的问题。

**Equation (5)**: blur-free image的affine组合
$$\hat{I}(\mathbf{x}) = [1 - A(\text{invproj}(\mathbf{x}, d_i, f_i))] \cdot I_{\min}(\mathbf{x}) + A(\text{invproj}(\mathbf{x}, d_i, f_i)) \cdot I_{\max}(\mathbf{x})$$

这是个非常聪明的trick——只需要拍摄black和white两张图就能得到vignetting的actual measurement，不用拟合任何parametric vignetting model。

---

## 3. MLP Architecture与训练细节

### 3.1 网络结构

```
Input: (x, y, f, [d], u, v)  -- 5D or 6D coordinates
       (each coord normalized)
       │
       ▼
FC Layer 1 → 512 units + ReLU
FC Layer 2 → 512 units + ReLU
FC Layer 3 → 512 units + ReLU
FC Layer 4 → 512 units + ReLU
FC Layer 5 → 512 units + ReLU
FC Layer 6 → 512 units + ReLU
FC Layer 7 → C units (C channels) + Sigmoid  (保证non-negativity)
```

- 7 fully-connected layers
- 512 channels each
- ReLU activations
- Final layer: **sigmoid** (满足 $\text{PSF} \geq 0$ constraint)
- **No positional encoding** (与NeRF/Neural Radiance Fields不同)
- Total parameters: ~19.1 MB (Table III)

为什么no positional encoding？因为PSF在 $(\mathbf{x}, f, d)$ 维度是平滑slowly-varying，spectral bias反而是个feature，避免high-frequency artifacts。只在 $\mathbf{u}$ 维度上有结构，但2D kernel size有限(最大120×120)，MLP的capacity足够。

### 3.2 训练超参数

- Optimizer: Adam with $\beta_1 = 0.5$, $\beta_2 = 0.999$
- Learning rate: $10^{-5}$, exponential decay
- Iterations: $4 \times 10^6$ to $10 \times 10^7$
- Batch size: 1
- Max PSF size: $120 \times 120$
- Patch size: 1.5× PSF size per dimension
- Framework: [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- GPU: single NVIDIA A6000
- $\epsilon = 10^{-15}$, weight decay $\|\mathbf{w}\|^2 = 10^{-6}$, $\gamma = 0.98$

### 3.3 训练循环细节

每个batch element:
1. 随机sample focus setting $f_i$ 和 sensor position $\mathbf{x}$
2. 在 $\mathbf{x}$ 周围sample $\mathbf{u}$ grid (2D PSF的内部坐标)
3. MLP forward得到2D PSF
4. 在patch $\mathcal{P}$ 内与 $\hat{I}$ 做convolution
5. 与captured image $I$ 计算L2 loss

**Locally stationary assumption**: 在patch $\mathcal{P}$ 内只用一次MLP forward pass for中心点 $\mathbf{x}$，但在 $\mathbf{u}$ grid上query多次。这是计算效率的关键。

---

## 4. Acquisition Procedure详解

### 4.1 三步preprocessing pipeline

**Step 1: Absolute Homography Estimation**
- 用dot grid pattern (white circles on black + conjugate)
- Binarize + Hough Transform + centroid refinement
- 估计camera intrinsics + radial distortion (OpenCV `calibrateCamera`)
- 计算homography $H$ mapping in-focus capture → sharp ground truth

**Step 2: Relative Homography Estimation**
- 对每个 $f_i$ 估计scale-only homography $S$ (绕principal point缩放)
- 这account for focus breathing effect (lens改变focus时magnification变化)

**Step 3: Mapping Sharp Ground Truth to Capture Space**
1. Apply $H^{-1}$ 把 ground truth 映到 in-focus 线性 perspective
2. Apply $S^{-1}$ 得到每个 $f_i$ 的 linear perspective sharp image
3. Apply $D$ (radial distortion) 到 raw capture space
4. Apply radiometric compensation (Eq. 5)

最终 $\text{invproj} = D \circ S \circ H$ 的composition。

### 4.2 Pattern选择

- **Dot grid**: 9×12，中心间距=4×dot radius，用于homography (robust to defocus)
- **Synthetic noise patterns** (Couture et al. ICCV 2011): 通过Fourier filtering控制spatial frequency content
  - 频率参数 $f \in \{10, 20, 50, 70, 100\}$
  - 保留 $[f, 2f]$ 频率band
  - Binarize based on mean
  - 对iPhone用 $f=50$ 即可，SLR用 $f=70$
- **Black/White patterns**: radiometric calibration

关键insight from ablation (Fig. S-10): noise pattern的feature size要 ≤ 最大PSF radius，否则PSF estimation失败。

### 4.3 Capture参数

- Monitor: 32-inch 4K LED (Dell UltraSharp)
- Exposure: 100ms (iPhone) / 250ms (Pixel, SLR) — 避免screen refresh/PWM/flicker
- ISO: lowest possible
- Burst: 3 images averaged
- Focal stack sampling: 7-24 positions, uniform in diopter space
- iPhone iOS API: focus值在[0,1]，需用linear rail calibration lookup table
- 总acquisition time: ~15 min (smartphone)

---

## 5. 实验数据深度分析

### 5.1 Simulation evaluation (Table I)

在合成的dual-pixel PSF (Punnappurath et al. 2020 model)上测试：

| Metric | Proposed (Train) | Joshi et al. | Mannan & Langer |
|--------|------------------|--------------|------------------|
| PSNR ↑ | 32.109 ± 1.201 | 31.829 ± 1.016 | 15.420 ± 0.381 |
| SSIM ↑ | 0.929 ± 0.072 | 0.950 ± 0.071 | 0.601 ± 0.021 |
| RMSE ↓ | 0.022 ± 0.004 | 0.025 ± 0.003 | 0.166 ± 0.010 |

关键观察：
- Proposed和Joshi在training lens positions相当
- Mannan & Langer严重失败(15dB) - 因为大PSF时ill-posed
- **Validation lens positions** (held out): proposed用MLP interpolation > linear interpolation of训练位置的PSF。这是MLP representation的关键优势 - thin lens equation的非线性defocus radius变化在focus附近有derivative sign change，linear interpolation无法捕捉。

### 5.2 Scalability (Table III)

最震撼的数据：

| Method | Single PSF | 5D PSF (4032×3024, 75×100 positions, 15 focus, 73×73 kernel, 4 channels) |
|--------|------------|---------------------------------------------------------------------------|
| Proposed | N/A | 14 hours / 19.1 MB |
| Joshi et al. | 6.5 sec / 39KB | 34 days / 9.3 GB |
| Mannan & Langer | 373.4 sec / 39KB | 1945 days / 9.3 GB |

Mannan要1945天 = 5.3年！而proposed只要14小时。这个**500x speedup + 500x memory reduction**是INR的核心价值。

### 5.3 Device fingerprinting (Section IV-C, Fig. 6)

通过repeatability实验验证：
- 同一iPhone连续4次capture：standard deviation很小
- 不同iPhone 14 Pro devices: RMS difference > std within device
- 结论：**同型号iPhone有distinct blur signature**，可以用于device fingerprinting

这是个新发现 - 以前没人能在commodity hardware上观察到manufacturing variability造成的光学差异。

### 5.4 Dual-pixel PSF complexity (Fig. 1)

Pixel 4的dual-pixel green channel PSF观察：
- PSF center of mass不总遵循expected left-right orientation
- 边缘PSF严重skewed
- Depart radically from Punnappurath et al.的parametric model

### 5.5 Restoration vs Seidel model (Table IV)

| Restoration | Siemens Star PSNR | Synthetic Blurry (Green) |
|--------------|-------------------|--------------------------|
| Proposed | 8.900 | 20.392 |
| Seidel Coefficients | 7.415 | 19.660 |

Proposed PSF更好保留Siemens Star的高频细节。Seidel model虽然可以拟合PSF尺寸，但无法捕捉aperture blade造成的octagonal shape、chromatic aberration等真实光学现象。

---

## 6. Applications

### 6.1 Rendering in Blender

Authors制作了Blender add-on (Fig. S-19)，集成blur field作为post-processing。流程：
1. 渲染 all-in-focus image + z-buffer (mist pass)
2. 量化depth成fronto-parallel planes
3. 用occlusion-aware image formation (Ikoma et al. 2021)

**Image Formation Models** (Section S-V-B):

Linear:
$$I = \sum_{k=0}^{K-1} \text{PSF}_k * (l_k \cdot a_k)$$

Approximate layered occlusion (Hasinoff & Kutulakos 2007):
$$I = \sum_{k=0}^{K-1} \text{PSF}_k * (l_k \cdot a_k) \prod_{k'=k+1}^{K-1} (1 - (\text{PSF}_{k'} * a_{k'}))$$

Nonlinear normalized (Ikoma et al. 2021):
$$I = \sum_{k=0}^{K-1} \tilde{l}_k \prod_{k'=k+1}^{K-1} (1 - \tilde{a}_{k'})$$

where $\tilde{l}_k = (\text{PSF}_k * (l_k \cdot a_k)) / E_k$, $\tilde{a}_k = (\text{PSF}_k * a_k) / E_k$, $E_k = \text{PSF}_k * \sum_{k'=0}^{k} a_{k'}$

非线性normalized model可以避免halo artifacts，效果最好。

### 6.2 Lens fingerprinting与device characterization

可以detect同model不同device的optical差异，可能用于：
- Quality control
- Device authentication
- Computational photography的device-specific calibration

---

## 7. 联系与延伸思考

### 7.1 与NeRF/Neural Fields的关系

这篇paper属于Neural Fields家族 ([Xie et al. 2022 survey](https://arxiv.org/abs/2111.11426))，但有几个关键区别：
- NeRF: $F_\theta(\mathbf{x}, \mathbf{d}) \to (\sigma, \mathbf{c})$，需要positional encoding
- Blur field: $F_\theta(\mathbf{x}, f, \mathbf{u}) \to \text{PSF}$，**不需要**positional encoding
- 这是因为PSF在 $(\mathbf{x}, f)$ 维度是物理smooth的，spectral bias反而帮助

类似工作：
- [Sitzmann et al. 2021](https://arxiv.org/abs/2006.09661) - differentiable optics
- [Tseng et al. 2021 SIGGRAPH](https://arxiv.org/abs/2101.05880) - end-to-end camera design

### 7.2 与auto-focus/dual-pixel的应用

Dual-pixel sensor的PSF建模是当前热点：
- [Punnappurath et al. 2020 ICCP](https://arxiv.org/abs/2003.09691) - parametric dual-pixel model
- [Xin et al. 2021 ICCV](https://arxiv.org/abs/2108.06094) - defocus map estimation
- Blur field可以为这些方法提供更accurate的PSF prior

### 7.3 与diffractive optics/coded aperture的关系

Paper提到可以extend到coded optics。这联系到：
- [Levin et al. 2007 SIGGRAPH](https://arxiv.org/abs/0709.0301) - coded aperture
- [Antipa et al. 2018 Optica](https://arxiv.org/abs/1709.00580) - DiffuserCam

### 7.4 与calibration的关系

传统的camera calibration (Zhang 2000, OpenCV)只估计intrinsics和distortion。Blur field提供了一个**完整的光学characterization**，包括：
- Aberration (higher-order)
- Defocus response
- Chromatic aberration (per-channel)
- Diffraction
- Vignetting (in Eq. 5)
- Dual-pixel disparity

### 7.5 Limitations与future work

从paper看几个limitation：
1. 仍是non-blind deconvolution - 需要known patterns
2. 没考虑wavelength dependence (Bayer CFA分开训练，但没显式model spectral PSF)
3. 5D/6D PSF是stationary assumption在小patch上，对于very strong spatial variation可能不够
4. 训练时间长 (4M-10M iterations)，没有用multi-resolution或coarse-to-fine
5. 需要monitor做calibration，不能in-the-wild

可能改进方向：
- Joint估PSF + sharp image (blind deconvolution)
- 加入wavefront representation (Zernike作为prior)
- 不同光照条件下calibration (ambient light的影响)
- Multi-resolution training加速
- 用meta-learning做few-shot device-specific fine-tuning

---

## 8. 关键Reference Web Links

- Paper project page: [blur-fields.github.io](https://blur-fields.github.io)
- [tiny-cuda-nn (NVIDIA)](https://github.com/NVlabs/tiny-cuda-nn)
- [Neural Fields survey - Xie et al. 2022](https://arxiv.org/abs/2111.11426)
- [Joshi et al. 2008 CVPR - PSF estimation using sharp edge prediction](https://www.cs.toronto.edu/~joshi/pubs/joshi_cvpr2008.pdf)
- [Mannan & Langer 2016 CRV - Blur calibration for depth from defocus](https://openreview.net/forum?id=BylQlQ5jQ)
- [Punnappurath et al. 2020 ICCP - Dual-pixel defocus-disparity](https://arxiv.org/abs/2003.09691)
- [Xin et al. 2021 ICCV - Dual-pixel defocus map](https://arxiv.org/abs/2108.06094)
- [Tseng et al. 2021 SIGGRAPH - Differentiable compound optics](https://arxiv.org/abs/2101.05880)
- [Sitzmann et al. 2018 SIGGRAPH - End-to-end optics optimization](https://arxiv.org/abs/1804.05909)
- [Ikoma et al. 2021 ICCP - Depth from defocus with learned optics](https://arxiv.org/abs/2104.08155)
- [Couture et al. 2011 ICCV - Unstructured light scanning](https://openaccess.thecvf.com/content_iccv_2011/papers/Couture_Unstructured_Light_Scanning_2011_ICCV_paper.pdf)
- [Tancik et al. 2020 NeurIPS - Fourier features for MLPs](https://arxiv.org/abs/2006.10739)
- [Heide et al. 2013 SIGGRAPH Asia - Computational imaging through simple lenses](https://www.cs.ubc.ca/labs/imager/tr/2013/SimpleLenses/heide13_simplified_lenses.pdf)
- [Levin et al. 2007 SIGGRAPH - Coded aperture](https://arxiv.org/abs/0709.0301)
- [Wadhwa et al. 2018 SIGGRAPH - Synthetic depth-of-field mobile phone](https://arxiv.org/abs/1806.07343)
- [Blender](https://www.blender.org/)
- [OpenCV camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [DxO lens correction database](https://www.dxo.com/technology/how-dxo-corrects-lens-flaws/)
- [Kohli et al. 2024 - Ring deconvolution microscopy](https://arxiv.org/abs/2401.17535)
- [Ramasinghe et al. 2022 - On regularizing coordinate-MLPs](https://arxiv.org/abs/2207.01225)

---

## 9. 建议的intuition building exercise

如果想真正理解这篇paper，建议按以下顺序思考：

1. **为什么non-parametric方法在大PSF上失败？** 答：patch必须 ≥ PSF size，但小patch内edge信号不足，inverse problem ill-posed。

2. **为什么MLP能overcome这个？** 答：MLP是global optimization，所有images的所有pixels共同supervise，不存在per-patch的signal-to-noise问题。Patch只是计算device，不是supervision boundary。

3. **为什么不需要positional encoding？** 画一下PSF在 $(\mathbf{x}, f)$ 上的变化 - 是slowly varying smooth field。NeRF需要PE是因为radiance有high-frequency view-dependent effects。

4. **为什么需要black/white calibration？** Eq. 4 (radiometric)的近似需要 $E$ 在scene plane uniform。但如果不知道vignetting，无法预测vignetting本身。Black/white图提供了vignetting的actual measurement。

5. **为什么scale homography绕principal point？** Focus breathing造成magnification change是绕光学中心的，不是绕image center。

6. **为什么sample in diopter space？** Diopter = 1/distance，thin lens的defocus radius在diopter space是linear的，uniform sampling更有效。

7. **同model iPhone为什么有不同blur signature？** Manufacturing variability - lens element placement误差，sensor tilt，aperture shape小差异，都贡献到PSF的高频结构上。

---

## 10. 总结

Learning Lens Blur Fields的核心贡献是把high-dimensional PSF estimation从"sample + interpolate"的discrete paradigm转变为"learn a continuous function"的neural paradigm。这种paradigm shift带来：
- **Scalability**: 5D PSF从9.3 GB降到19.1 MB
- **Speed**: 1945天到14小时
- **Smoothness**: MLP的spectral bias避免high-frequency artifacts
- **Expressiveness**: 能capture Seidel/Zernike无法建模的非对称、chromatic、dual-pixel等效应
- **Discovery**: 揭示同model device的manufacturing variability

这篇paper是INR在computational optics上的clean application，类似NeRF在view synthesis上的作用 - 都是利用MLP的interpolation properties来parameterize一个原本需要sparse sampling的high-dimensional physical quantity。可以预期未来会看到更多这种"physics-informed INR"的工作，例如wavefront sensing、spectral PSF、polarization-dependent blur等扩展。
