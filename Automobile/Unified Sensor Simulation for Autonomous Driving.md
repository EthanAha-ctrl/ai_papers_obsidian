---
source_pdf: Unified Sensor Simulation for Autonomous Driving.pdf
paper_sha256: 451e3c02c7abda7e704aa22e8ae631477ec530e095624dca36236bbc1b2a7c3b
processed_at: '2026-08-12T19:44:38-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 XSIM

## 这篇 paper 到底在干啥

想象你是自动驾驶公司的工程师。你想训练 perception model，但真实数据太贵、太慢、危险场景太少。于是你想搞仿真——把真实驾驶录像重建出来，然后能从任意视角、任意轨迹重新渲染相机和 LiDAR 数据。

这个事情听起来简单，做起来要命。因为真实传感器很"刁钻"：

- 相机有 **rolling shutter**（逐行曝光，不是瞬间拍照）
- 相机有 **optical distortion**（鱼眼、广角）
- LiDAR 是 **360 度旋转扫描**的，每个 beam 的时间都不一样
- 车在动、行人也在动，曝光/扫描期间整个 world 都在变

传统 3DGS 用 EWA splatting，核心假设是"在 Gaussian 中心附近，projection 是线性的"。这个假设在 pinhole camera + 静止场景下还凑合，碰到 rolling shutter + spherical LiDAR + 高速运动就崩了。

XSIM 干的事情：基于 3DGUT（一个 CVPR 2025 的工作，用 Unscented Transform 替代 EWA 的 linearization），把上面这些"刁钻"的东西统一搞定。

---

## 三个核心问题，一个个说

### 问题 1：Rolling Shutter 怎么处理

**现象**：相机每行曝光时间不同。LiDAR 每个 beam 扫描时间不同。扫描期间车走了几米，人也走了一步。

**传统做法**（SplatAD 等）：给相机和 LiDAR 各写一套 sensor-specific 的 rolling shutter model。工作量大，且每加一种传感器就要重写。

**XSIM 的思路**：把 rolling shutter 写成一个"自洽方程"。

具体来说，一个 3D 点 $\pmb{x}_w$ 要投影到图像坐标 $(u, v)$。但问题是：

- 要投影到 $(u, v)$，你需要知道"这个像素是什么时候曝光的"，也就是时间 $\eta$
- 要知道时间 $\eta$，你需要知道点投影到哪个像素 $(u, v)$
- 而像素 $(u, v)$ 又依赖于时间 $\eta$（因为 $\eta$ 时刻的车和物体的位置都变了）

所以这是一个鸡生蛋问题：

$$\eta = \tau(u(\eta), v(\eta))$$

左边是"观察时间"，右边是"该像素对应的曝光时间函数"。两边要相等。

**解法**：Newton-Raphson 迭代。先猜一个 $\eta_0$，投影一次，看算出来的 $\tau(u, v)$ 和 $\eta$ 差多少，沿着 Jacobian 修正一下，再来一次。Paper 说实际上 2-3 次迭代就收敛。

关键 trick 在于这个 formulation **完全独立于 camera model**。你给它 pinhole、fisheye、spherical，只要能写出 $\pi(\pmb{x}_c) \to (u, v)$ 和它的 Jacobian，就能 plug in。这就是为什么 XSIM 能"unified"地处理 camera 和 LiDAR。

**变量解释**：
- $\pmb{x}_w$: world coordinates 下的 3D 点
- $\pmb{x}_w(\eta)$: 该点在时间 $\eta$ 的位置（因为 dynamic actor 在动）
- $\pmb{q}(\eta), \pmb{t}(\eta)$: 相机在时间 $\eta$ 的 orientation (quaternion) 和 translation
- $\pmb{x}_c(\eta) = \pmb{q}^{-1}(\eta) \otimes (\pmb{x}_w(\eta) - \pmb{t}(\eta)) \otimes \pmb{q}(\eta)$: 点在 camera frame 下的坐标
- $\tau(u, v) = \tau_{start} + u\tau_u + v\tau_v$: 像素 $(u, v)$ 的曝光时间，$\tau_u, \tau_v$ 控制 scan 方向和速度

---

### 问题 2：LiDAR 跨越 ±π 边界会"裂开"

这个是 XSIM 最 novel 的点，我先描述现象再讲原理。

**现象**：LiDAR 旋转扫描，azimuth $\varphi = \text{atan2}(y, x) \in [-\pi, \pi]$。$\varphi = +\pi$ 和 $\varphi = -\pi$ 物理上是同一个方向，但 LiDAR 扫到 $+\pi$ 那侧是某个时刻 $t_1$，扫到 $-\pi$ 那侧是另一个时刻 $t_2 = t_1 + 100\text{ms}$。中间隔了整整一圈扫描周期。

如果一个 3D Gaussian 刚好横跨 $\varphi = \pm\pi$ 这条缝，会发生什么？

Unscented Transform 的做法是：取一堆 sigma points，每个 point 投影一次，然后拟合一个 2D Gaussian。但 sigma points 散落到 $+\pi$ 和 $-\pi$ 两侧之后，算 mean 会把两边"扯"到中间 $\varphi \approx 0$，算 covariance 会得到一个覆盖整个 range image 的巨大 Gaussian。

结果就是 Figure 1 那种"LiDAR range image 在 ±π 边界附近完全乱掉"的 artifact。

更恶劣的是，如果 dynamic actor 跨越这条缝，由于 rolling shutter，它会被 LiDAR 看到**两次**（一次在 $+\pi$ 侧 $t_1$ 时刻，一次在 $-\pi$ 侧 $t_2$ 时刻，中间它移动了）。这种"bimodal projection"是 UT 的 single Gaussian assumption 完全无法处理的。

**XSIM 的解法：Phase modeling**

思路很直接：既然 azimuth 是周期函数 $\varphi = \text{atan2}(y,x) + 2\pi k$，那就显式考虑 $k \in \{-1, 0, +1\}$ 三个 phase。

- **Central projection**：用 mid-exposure time $\tau_{mid}$ 初始化，做 standard UT
- **Negative shift**：用 start time $\tau_{start}$ 初始化，把 $\varphi \geq 0$ 的 sigma points 减 $2\pi$，wrap 到 $[-\pi, 0]$
- **Positive shift**：用 end time $\tau_{end}$ 初始化，把 $\varphi < 0$ 的 sigma points 加 $2\pi$，wrap 到 $[0, \pi]$

每个 phase 独立做一次 UT，得到独立的 2D Gaussian。如果两个 phase 都 valid（extent 合理、与 image range 相交），说明这个 particle 确实 bimodal，返回两个 2D Gaussian。否则返回 valid 的那一个。

**Intuition**：UT 假设"投影后还是 single Gaussian"。这个假设在 spherical image 的 $\pm\pi$ 缝处不成立。XSIM 的 fix 是把一个 particle 显式拆成最多三个"phase projection"，每个 phase 内部 UT 假设成立。本质上是 mixture of Gaussians，用 topology-aware 的方式分解。

这个 mechanism 看起来很 specific（只对 spherical + rolling shutter 有用），但思路其实可以推广。任何有 cyclic/discontinuous topology 的 sensor model 都可能遇到类似问题：fisheye 的 antipodal point、panoramic camera 的 seam、mirror-based sensor 的 reflection boundary。

---

### 问题 3：Camera 和 LiDAR 的"透明度"打架

这个看起来简单，但 idea 很关键。

**问题**：3DGS 里每个 Gaussian 有一个 opacity $\sigma$。但 camera 和 LiDAR 对 opacity 的物理要求是冲突的。

**LiDAR 角度**：我要的是 sharp depth，surface 要用 opaque Gaussian 精确表示。一个 ray 打到一个 surface，要么 hit 要么 miss，不要搞半透明。

**Camera 角度**：真实世界有 specular reflection、translucency、view-dependent shading。这些效果需要好几个 semi-transparent Gaussian 沿 viewing ray 叠加才能表达。

**玻璃**：对 LiDAR（905nm 近红外）可能直接穿透，对可见光可能反射或透射。物理上就是不同的 transparency。

如果硬让两边共享一个 $\sigma$：要么 LiDAR 变模糊（$\sigma$ 被拉小），要么 camera 失去 view-dependent 表达力（$\sigma$ 被拉大）。

**XSIM 的解法**：每个 Gaussian 带两个 opacity $\sigma_c$（camera）和 $\sigma_L$（LiDAR），独立优化。加一个 L1 regularization 防止两者 drift 太远：

$$\mathcal{L}_{opacity} = \sum_i |\sigma_{c,i} - \sigma_{L,i}|$$

普通表面会学到 $\sigma_c \approx \sigma_L$，玻璃等特殊材料会学到 $\sigma_c \neq \sigma_L$。Pruning 用 $\max(\sigma_c, \sigma_L)$ 做判据，避免某个 modality 下重要的 particle 被误删。

**为什么这个 idea 重要**：表面上是"加个参数"，实际上承认了一个物理事实——不同 sensing modality 探测的是不同的物理量，它们的"透明度"在物理上就不该共享。这个 decoupling 思路可以推广到 event camera、RGB-IR、ToF camera 等其他 sensor modality。

---

## 整体架构图

```
Driving Log (camera images + LiDAR sweeps + bounding boxes)
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  Scene Graph Initialization      │
    │  - Static background node        │
    │  - Rigid actor nodes (vehicles)  │
    │  - Deformable nodes              │
    │  - SMPL nodes (pedestrians)      │
    └──────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  Per-Gaussian Parameters         │
    │  μ, Σ (shape)                    │
    │  c_d, c_s (appearance)           │
    │  σ_c, σ_L (dual opacity) ◄── XSIM│
    └──────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   Camera Render             LiDAR Render
        │                       │
        ▼                       ▼
  ┌─────────────────────────────────────┐
  │  3DGUT Projection (Unscented        │
  │  Transform)                         │
  │  + Generalized Rolling Shutter      │
  │    (Newton-Raphson on η=τ(u,v))     │
  │  + Phase Modeling (spherical) ◄─XSIM│
  └─────────────────────────────────────┘
        │                       │
        ▼                       ▼
    RGB image              Range image
        │                       │
        └───────────┬───────────┘
                    ▼
    ┌──────────────────────────────────┐
    │  Loss = λL1 + (1-λ)L_SSIM        │
    │       + L_depth                  │
    │       + L_opacity (dual)          │
    │       + L_reg                    │
    └──────────────────────────────────┘
```

---

## 实验数据讲清楚

### 主表：Waymo 12 个 scene 重建质量

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CD↓ (LiDAR) |
|---|---|---|---|---|
| EmerNerf (ICLR24) | 27.15 | 0.806 | 0.462 | 0.71 |
| HUGS (CVPR24) | 26.90 | 0.851 | 0.335 | 44.58 |
| SplatAD (CVPR25) | 27.74 | 0.865 | 0.281 | 0.82 |
| **XSIM** | **30.75** | **0.903** | **0.223** | **0.08** |

PSNR 提升 +3.01 dB（在这个领域 +0.5 已经是大改进，+3 是 leap）。LiDAR CD 从 0.82 降到 0.08，**降 8.8 倍**。这是质变，不是量变。

### Novel view synthesis（训练隔帧，测试剩下）

| Dataset | Method | PSNR | CD |
|---|---|---|---|
| Waymo | SplatAD | 27.06 | 0.82 |
| Waymo | XSIM | 29.80 | 0.18 |
| Argoverse 2 | SplatAD | 28.40 | 2.68 |
| Argoverse 2 | XSIM | 29.44 | 1.26 |
| PandaSet | SplatAD | 26.77 | 1.69 |
| PandaSet | XSIM | 27.00 | 1.23 |

NVS 是衡量"sim 能不能从没见过的视角渲染"的关键。XSIM 在三个数据集上一致领先。PandaSet 上 RGB 质量只领先 0.23 dB（LPIPS 反而略输），但 LiDAR CD 还是有显著优势。

### Ablation：每个组件值多少钱

Waymo 6 个 scene 的 NVS ablation：

| 配置 | PSNR | SSIM | LPIPS | CD |
|---|---|---|---|---|
| Full XSIM | 30.03 | 0.8945 | 0.2122 | 0.21 |
| 去掉 camera RS | 28.95 | 0.8761 | 0.2407 | 0.21 |
| 去掉 LiDAR opacity | 29.55 | 0.8888 | 0.2189 | 0.25 |
| 去掉 LiDAR RS | 29.05 | 0.8816 | 0.2296 | 0.31 |
| 去掉 phase modeling | 29.32 | 0.8824 | 0.2245 | 0.28 |

读这张表的方式：
- **Camera rolling shutter** 贡献最大 PSNR gain (+1.08)。自动驾驶场景里 RS 对 image quality 影响巨大，这是预期的。
- **LiDAR rolling shutter** 对 LiDAR CD 影响显著 (0.21 → 0.31)。这是 LiDAR 仿真保真度的核心。
- **Phase modeling** 数值上看起来贡献小（PSNR +0.71, CD +0.07），但 qualitative 差别巨大。Figure 1 的合成实验专门 isolate 这个：phase modeling 修掉的是 azimuth boundary 附近的"撕裂"artifact，这种 local artifact 在全图平均 metric 上被稀释了。
- **Dual opacity** 同时改善 PSNR 和 CD，验证 geometry/appearance decoupling 是双赢，不是 trade-off。

---

## 更细节的技术讲解

### Unscented Transform 到底在干啥

UT 来自 state estimation 领域 [Julier & Uhlmann 2004](https://ieeexplore.ieee.org/document/1271396)。问题设定：你有一个 random variable $\pmb{x} \sim \mathcal{N}(\pmb{\mu}, \pmb{\Sigma})$，过一个 nonlinear function $\pmb{y} = f(\pmb{x})$，你想知道 $\pmb{y}$ 的 mean 和 covariance。

**EKF 做法**：在 $\pmb{\mu}$ 处 linearize，$f(\pmb{x}) \approx f(\pmb{\mu}) + J_f(\pmb{\mu})(\pmb{x}-\pmb{\mu})$，然后 $\pmb{y}$ 也是 Gaussian。问题：linearization 在高度 nonlinear 区域不准。

**UT 做法**：构造 $2n+1$ 个 sigma points：
- $\pmb{\chi}_0 = \pmb{\mu}$
- $\pmb{\chi}_i = \pmb{\mu} + (\sqrt{(n+\lambda)\pmb{\Sigma}})_i$ for $i=1..n$
- $\pmb{\chi}_i = \pmb{\mu} - (\sqrt{(n+\lambda)\pmb{\Sigma}})_i$ for $i=n+1..2n$

每个 sigma point 单独过 $f(\cdot)$：$\pmb{\gamma}_i = f(\pmb{\chi}_i)$。然后用 weighted mean 和 covariance 拟合 $\pmb{y}$ 的 distribution：

$$\hat{\pmb{\mu}}_y = \sum_i W_i^{(m)} \pmb{\gamma}_i$$
$$\hat{\pmb{\Sigma}}_y = \sum_i W_i^{(c)} (\pmb{\gamma}_i - \hat{\pmb{\mu}}_y)(\pmb{\gamma}_i - \hat{\pmb{\mu}}_y)^\top$$

权重 $W_i$ 由参数 $\lambda$ 决定。

**在 splatting 里**：3D Gaussian $\mathcal{N}(\pmb{\mu}, \pmb{\Sigma})$ 过 camera projection $\pi(\cdot)$ 得到 2D Gaussian。EWA 用 linearized $\pi$（Jacobian），3DGUT 用 UT。UT 的优势是 sigma points 各自过 nonlinear $\pi$，只在最后拟合 2D Gaussian 时才做 Gaussian approximation，所以 nonlinear distortion 被"采样"进来了。

**UT 失败的场景**：当 $\pi$ 把 Gaussian "撕成两半"（比如 spherical wrap），sigma points 散到两个 disjoint region，fit 出来的 single Gaussian 是 garbage。这就是 phase modeling 要解决的。

### Newton-Raphson 在 RS 投影上的具体形式

方程：$\Delta\eta(\eta) = \eta - \tau(u(\eta), v(\eta)) = 0$

Jacobian：
$$\frac{d\Delta\eta}{d\eta} = 1 - \tau_u \frac{du}{d\eta} - \tau_v \frac{dv}{d\eta}$$

其中 $\frac{du}{d\eta} = \frac{d\pi_u}{d\pmb{x}_c} \cdot \frac{d\pmb{x}_c}{d\eta}$，$\frac{dv}{d\eta}$ 类似。

$\frac{d\pmb{x}_c}{d\eta}$ 来自 actor 和 camera 的 motion model，是解析的。$\frac{d\pi}{d\pmb{x}_c}$ 是 camera model 的 Jacobian，比如 perspective：

$$\frac{d\pi_{perspective}}{d\pmb{x}_c} = \begin{pmatrix} \frac{f_x}{z} & 0 & -\frac{f_x x}{z^2} \\ 0 & \frac{f_y}{z} & -\frac{f_y y}{z^2} \end{pmatrix}$$

迭代：
$$\eta_{n+1} = \eta_n - \frac{\Delta\eta(\eta_n)}{d\Delta\eta/d\eta|_{\eta_n}}$$

Paper 说实际 2-3 次就收敛，因为 motion 是 constant velocity 假设下的小扰动。

### Phase Modeling Algorithm 详解

参考 paper Algorithm 2。给定 3D Gaussian $(\pmb{\mu}_{3D}, \pmb{\Sigma}_{3D})$ 和 rolling shutter projection $\pi_{rolling}(\pmb{x}, \eta)$：

1. **Central projection**: 初始化 $\eta = \tau_{mid}$，做 UT。如果投影不与 visible range 相交，返回空（particle 不可见）。
2. **Negative shift projection**: 初始化 $\eta = \tau_{start}$，对每个 sigma point 投影后检查 azimuth：如果 $\varphi \geq 0$ 就减 $2\pi$ 把它 wrap 到 $[-\pi, 0]$。再做 UT。
3. **Positive shift projection**: 初始化 $\eta = \tau_{end}$，如果 $\varphi < 0$ 就加 $2\pi$ 把它 wrap 到 $[0, \pi]$。再做 UT。
4. **返回策略**：
   - 如果两个 shift 都 valid 且 azimuth extent $< \pi$，返回两个 2D Gaussian（bimodal case）
   - 如果只有一个 valid，返回那个
   - 否则返回 central

$\varphi_{ext} < \pi$ 这个判据是为了过滤 wrap 后 extent 反而变大的退化情况（比如 particle 本来就很大，wrap 后从 $+\pi$ 端到 $-\pi$ 端的"绕路"距离很大）。

---

## 我的直觉和联想

### 1. UT 的 distributional assumption 是关键瓶颈

UT 假设"投影后还是 single Gaussian"。这个假设在大部分场景下成立，但只要 sensor model 有 topology discontinuity（spherical wrap、fisheye antipodal、mirror seam），就会破坏。

Phase modeling 的本质是 mixture of Gaussians，mixture components 数量由 sensor topology 决定。这个思路可以推广：
- **Fisheye camera**: antipodal point 附近投影会有类似问题
- **Panoramic 360 camera**: stitch seam 处
- **Multi-view splatting**: 不同 view 间 baseline 大时，同一 3D Gaussian 在不同 view 的 2D 差异巨大，可能也需要 mixture

### 2. Rolling shutter 的 implicit formulation 很优雅

传统 RS 处理有两种：
- **Discretize time + warp**: 把一帧拆成多个 sub-frame，每个 sub-frame 用 global shutter 渲染，最后 warp 合成。计算贵，且有 blending artifact。
- **Linearized approximation**: 在 mid-exposure 处 linearize，假设 RS 偏移是 small perturbation。在高速运动下不准。

XSIM 的 implicit equation $\eta = \tau(u(\eta), v(\eta))$ + Newton-Raphson 是第三条路。每个 3D point 在投影时找到自己的 self-consistent observation time。这等价于"per-point inverse rolling shutter"。

这种 formulation 让 sensor model 和 motion model 解耦——任意 $\pi(\cdot)$ 都能 plug in，只要能算 Jacobian。这是为什么 XSIM 能 unified 处理 camera 和 LiDAR 的根本原因。

### 3. Dual Opacity 是 physics-aware parameterization

3DGS 原版用 spherical harmonings 表达 view-dependent color，但 opacity 是 view-independent 的单一值。XSIM 的 dual opacity $\sigma_c, \sigma_L$ 把这个 view-independence 推广到 sensor-independence。

物理直觉：opacity 本质上是"这个 particle 对某种 sensing modality 的阻挡程度"。LiDAR 的近红外和 camera 的可见光波长差 1-2 个数量级，material 的 reflectance/transmittance 在这两个波段下完全不同（玻璃就是极端例子）。强行 share 一个 $\sigma$ 是物理上错误的。

推广方向：
- **Event camera**: log intensity change 有自己的 dynamics，opacity 概念可能要换
- **RGB-IR**: 不同波长可见光 + 红外，每个 wavelength band 一个 opacity
- **ToF camera**: multi-frequency phase，每个 frequency 一个 opacity
- **Radar**: mm-wave 和光学完全不同的 scattering behavior

### 4. Constant Velocity 假设的局限

Paper 假设 actor 和 camera 在 capture 期间都是 constant linear + angular velocity。这在大部分 driving 场景下 OK（capture duration 通常 30-100ms），但在以下情况会失效：
- Sharp turn（角速度突变）
- Hard brake（线加速度大）
- Pedestrian突然变向

下一步可能是：
- Constant acceleration model（二阶 motion model）
- IMU-integrated motion（用 IMU high-frequency 数据 interpolate）
- Per-actor learnable motion model（让网络学 motion pattern）

### 5. Sim-to-real 下游价值

XSIM 在 lane shift 3m 下的 consistency 是 closed-loop testing 的 prerequisite。这是 sim-to-real transfer 的关键 metric——sim 必须能从 training trajectory 之外的新视角渲染合理数据，否则只能做 data augmentation，不能做 closed-loop safety testing。

可以接的下游：
- [HUGSim](https://arxiv.org/abs/2412.01718): closed-loop simulator 框架
- [NeurONCap](https://arxiv.org/abs/2410.23275): photorealistic closed-loop safety testing
- [PreSight](https://arxiv.org/abs/2407.13316): city-scale NeRF priors for AV perception

### 6. 与 3D Gaussian Ray Tracing 的关系

[Moenne-Loccoz et al. 2024](https://arxiv.org/abs/2410.18482) 的 3D Gaussian Ray Tracing 走的是另一条路：把 splatting 推向 path tracing，物理上更准确但计算贵。XSIM 走的是"在 splatting framework 内逼近 ray-based rendering fidelity"（通过 UT + 3D evaluation），保持 rasterization 速度。

两个方向的 convergence point 可能是 hybrid rendering：关键 region 用 ray tracing，背景用 splatting。这有点像现代 game engine 的 rasterization + ray tracing hybrid pipeline。

### 7. Limitation 推测（paper 没明说）

- **LiDAR intensity 没建模**：真实 LiDAR return 有 intensity 值，反映 material reflectance、incident angle、distance 衰减。[SplatAD](https://arxiv.org/abs/2411.16816) 有 explicit intensity rendering，XSIM 看起来只 render range
- **Multi-return LiDAR 没建模**：真实 LiDAR 一个 beam 可能多次 return（比如透过树叶看到地面），3DGS 的 alpha compositing 是 implicit single-return
- **Motion blur 没建模**：rolling shutter 解决了 temporal offset，但每个 pixel 的 exposure time 内 integration 没建模（应该是 box filter over exposure time）
- **Phase modeling 只处理 azimuth**：elevation 边界（LiDAR 顶部/底部 beam）可能也有类似问题，但 extent 通常小，paper 没提

### 8. Newton-Raphson 和 EKF 的同构性

$\eta = \tau(u(\eta), v(\eta))$ 可以写成 measurement equation $h(\eta) = \eta - \tau(u(\eta), v(\eta)) = 0$。Newton-Raphson 是 Gauss-Newton 的特例（残差是 scalar）。

这种 formulation 让 sensor model 可以接入 robotics estimator 的数学语言。如果以后要把 XSIM 接到 SLAM 或 state estimator 里，这个 formulation 是自然的 interface。

### 9. Bimodal projection 推广到 multi-camera

如果一个 3D Gaussian 同时被多个 camera 看到，且 camera 间 baseline 大（比如 driving 的 surround camera），它的 2D projection 在不同 view 里差异巨大。Phase modeling 的 mixture 思路可以推广：每个 view 独立做 UT，而不是用全局一致的 2D Gaussian。这其实是 multi-view splatting 一直没解决的问题——大部分工作用 single 2D Gaussian 覆盖所有 view，baseline 大时 approximation 变差。

### 10. Topology-aware splatting 是新方向

传统 splatting 假设 image plane 是 $\mathbb{R}^2$。但实际 sensor 的 image space 可能有 non-trivial topology：
- Spherical: $\mathbb{S}^2$ (LiDAR) 或 $\mathbb{S}^1 \times \mathbb{R}$ (panoramic)
- Fisheye with antipodal: 投影到 disk 但 antipodal point 重合
- Cubemap: 6 个 face，seam 处有 discontinuity

XSIM 的 phase modeling 是第一个（据我所知）显式处理 spherical topology 的工作。这开了一个方向：topology-aware splatting，针对不同 sensor topology 设计不同的 projection strategy。

---

## 总结一句

XSIM 的核心 insight 是：autonomous driving 的 sensor simulation 需要三件事——nonlinear projection (3DGUT)、generalized rolling shutter (Newton-Raphson on implicit equation)、topology-aware handling (phase modeling)。加上 dual opacity 这个 physics-aware parameterization，把 camera 和 LiDAR 的 unified rendering 推到了 SOTA。

如果你想 build deeper intuition：
1. 读 [3DGUT paper](https://arxiv.org/abs/2412.20402) 理解 UT 在 splatting 里怎么 work
2. 读 [SplatAD](https://arxiv.org/abs/2411.16816) 对比 sensor-specific RS modeling 的做法
3. 读 [Julier & Uhlmann 2004](https://ieeexplore.ieee.org/document/1271396) 理解 UT 的 statistical foundation
4. 跑一下 [XSIM 代码](https://github.com/whesense/XSIM)，看 phase modeling 在 azimuth boundary 附近的实际效果

---

# XSIM: Unified Sensor Simulation for Autonomous Driving 详解

## 1. 核心动机与背景

XSIM 要解决的是自动驾驶仿真中 sensor rendering 的根本矛盾。传统 3DGS 基于 EWA splatting [Zwicker et al., 2001]，其核心在于通过 single point 处的 Jacobian 对 projection 做 linear approximation。这种 linearization 在 autonomous driving 场景下会出现严重问题，因为真实的 driving sensors 有三类非线性：(1) rolling shutter (RS) 导致的时空耦合；(2) optical distortions (fisheye、wide FOV)；(3) spherical projection (LiDAR) 的 cyclic azimuth。

3DGUT [Wu et al., 2025](https://arxiv.org/abs/2412.20402) 的关键 insight 是用 Unscented Transform 替换 EWA 的 linearized projection。UT 本来是 state estimation 里处理 nonlinear propagation 的工具 [Julier & Uhlmann, 2004]，这里被 repurpose：把一个 3D Gaussian 当成 posterior distribution，构造 2n+1 个 sigma points，每个 sigma point 单独通过 nonlinear camera model 投影，然后从投影后的 sigma points 拟合出一个 2D Gaussian。这等价于在 rasterization 框架里近似 ray-based rendering。

XSIM 在此基础上解决三个具体问题：

---

## 2. 方法详解

### 2.1 3DGUT Preliminary

每个 Gaussian particle 由 mean $\pmb{\mu} \in \mathbb{R}^3$ 和 covariance $\pmb{\Sigma} \in \mathbb{R}^{3\times3}$ 定义：

$$
\rho(\pmb{x}) = \sigma \exp\left(-\frac{1}{2}(\pmb{x}-\pmb{\mu})^\top \pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu})\right)
$$

这里 $\sigma \in \mathbb{R}$ 是 opacity（最大响应值），$\pmb{\mu}$ 是 Gaussian 在 3D space 的中心，$\pmb{\Sigma}$ 是 3D 形状矩阵。实际优化时 $\pmb{\Sigma}$ 被分解为 $\pmb{\Sigma} = \pmb{R}\pmb{S}\pmb{S}^\top\pmb{R}^\top$，其中 $\pmb{s} \in \mathbb{R}^3$ 是 scaling vector（对角线元素），$\pmb{q} \in \mathbb{R}^4$ 是 rotation quaternion。这种参数化保证 $\pmb{\Sigma}$ 始终 positive semi-definite。

每个 particle 还携带 diffuse color $\pmb{c}_d \in \mathbb{R}^3$ 和 view-dependent appearance feature $\pmb{c}_s \in \mathbb{R}^f$。注意 XSIM 不用 spherical harmonics，而是 follow SplatAD 的做法：render 出 feature map 后用一个 small CNN 解码成 RGB。

**Volumetric integration 的关键改动**：3DGUT 不在 2D conic 上做 EWA 那样的积分，而是对每个 camera ray $\pmb{o} + \tau \pmb{d}$ 找到 particle 响应最大的点：

$$
\pmb{x}_{max} = \pmb{o} + \tau_{max}\pmb{d}, \quad \tau_{max} = \frac{\pmb{d}^\top \pmb{\Sigma}^{-1}(\pmb{\mu}-\pmb{o})}{\pmb{d}^\top \pmb{\Sigma}^{-1}\pmb{d}}
$$

这里 $\pmb{o}$ 是 camera origin，$\pmb{d}$ 是 ray direction（通常 normalized），$\tau_{max}$ 是沿 ray 的参数（即深度，如果 $\pmb{d}$ 是单位向量）。这个公式其实是 "Gaussian 沿 ray 投影的极值点"，通过对 $\rho(\pmb{o}+\tau\pmb{d})$ 关于 $\tau$ 求导并令其为零得到。

然后标准 alpha compositing：

$$
T_i = \prod_{j<i}(1-\alpha_j), \quad \pmb{c} = \sum_i \pmb{c}_i \alpha_i T_i
$$

$T_i$ 是 transmittance（前 i-1 个 particle 的累积透明度），$\alpha_i = \rho_i(\pmb{x}_{max})$ 是第 i 个 particle 在该 ray 上的响应。Depth rendering 直接把 $\pmb{c}_i$ 换成 $\tau_{max,i}$。

**Intuition**：这种 3D-space evaluation 而不是 2D-screen-space evaluation 的好处是——projection approximation error 不会进入最终的 volumetric integration，只在 tiling 阶段起作用。这对于 LiDAR 这种 spherical sensor 尤其重要。

### 2.2 Extended 3D Gaussian Representation: Dual Opacity

这是 XSIM 的一个看起来简单但很关键的 insight。

**问题**：LiDAR 和 camera 对 opacity 的物理要求是 conflict 的。

- **LiDAR** 是 active sensing，每个 surface 应该被 single opaque Gaussian 精确表示，render 出来的 range 要 sharp。
- **Camera** 需要建模 specular reflection、translucency、view-dependent effects，通常需要 multiple semi-transparent Gaussians 叠加在同一条 viewing ray 上。
- **玻璃** 这种材料对 LiDAR（905nm/1550nm 近红外）和可见光透明性完全不同。LiDAR 可能直接穿透，camera 看到反射。

如果共用一个 opacity $\sigma$，要么 LiDAR 模糊（$\sigma$ 偏小），要么 camera 失去 view-dependent 建模能力（$\sigma$ 偏大）。

**Solution**：每个 Gaussian 携带两个独立 opacity $\sigma_c$（camera）和 $\sigma_L$（LiDAR），jointly optimized，但用 regularization 强制一致性：

$$
\mathcal{L}_{opacity} = \sum_i |\sigma_{c,i} - \sigma_{L,i}|
$$

这个 L1 penalty 是 soft constraint——允许 deviation，但 penalize 大的 deviation。这样玻璃等波长依赖的材料可以学出 $\sigma_c \neq \sigma_L$，而大多数普通表面则 $\sigma_c \approx \sigma_L$。

**Pruning 时**取 $\max(\sigma_c, \sigma_L)$ 作为判据，避免某个 modality 下重要的 particle 被误删。

### 2.3 General Rolling Shutter Modeling

这部分是 XSIM 的技术核心之一。公式比较多，逐个拆解。

**Rolling shutter 时间模型**：

$$
\tau(u,v) = \tau_{start} + u\tau_u + v\tau_v
$$

$(u,v) \in [0,1]^2$ 是 normalized image coordinates。$\tau_u, \tau_v$ 定义 scan 方向与速度。例如 horizontal RS 的 camera 通常 $\tau_v = 0, \tau_u = T_{line}/W$。$\tau_{mid} = \tau(0.5, 0.5)$ 是 mid-exposure time，作为 reference timestamp。

**Dynamic actor 运动**（constant velocity 假设）：

$$
\pmb{x}_w(\eta) = \pmb{x}_w(\tau_{mid}) + (\pmb{v}_a + \pmb{w}_a \times \pmb{r})\eta
$$

$\eta$ 是相对 $\tau_{mid}$ 的时间偏移。$\pmb{v}_a \in \mathbb{R}^3$ 是 actor 线速度，$\pmb{w}_a \in \mathbb{R}^3$ 是角速度。$\pmb{r} \in \mathbb{R}^3$ 是 point 在 actor local frame 中的位置。$\pmb{w}_a \times \pmb{r}$ 是 cross product，给出 angular velocity 引起的线速度。这其实是 screw motion 的一阶近似。

**Camera 运动**：

$$
\pmb{q}(\eta) = e^{\pmb{w}_c \eta / 2} \otimes \pmb{q}(\tau_{mid}), \quad \pmb{t}(\eta) = \pmb{t}(\tau_{mid}) + \eta \pmb{v}_c
$$

$\pmb{q} \in \mathbb{R}^4$ 是 unit quaternion 表示 orientation，$e^{\pmb{w}_c\eta/2}$ 是 quaternion exponential map（$\pmb{w}_c$ 是 camera 的 angular velocity in body frame），$\otimes$ 是 Hamilton product（quaternion 乘法）。$\pmb{t} \in \mathbb{R}^3$ 是 translation。这里 quaternion 上的 exponential 给出 constant angular velocity 下的精确积分。

**投影方程**（世界点到 camera coordinates）：

$$
\pmb{x}_c(\eta) = \pmb{q}^{-1}(\eta) \otimes (\pmb{x}_w(\eta) - \pmb{t}(\eta)) \otimes \pmb{q}(\eta)
$$

$(u(\eta), v(\eta)) = \pi(\pmb{x}_c(\eta))$

这里 $\pi(\cdot): \mathbb{R}^3 \to [0,1]^2$ 是任意 static camera 的 projection function（perspective, fisheye, spherical 都可以 plug in）。$\pmb{q}^{-1} \otimes \cdot \otimes \pmb{q}$ 是用 quaternion 旋转一个向量（这里把 $\pmb{x}_w - \pmb{t}$ 当成 pure quaternion）。

**核心难点**：observation time $\eta$ 依赖于 image position，而 image position 又依赖于 $\eta$：

$$
\eta = \tau(u(\eta), v(\eta))
$$

这是 fixed point equation，没有 closed-form solution。XSIM 用 **Newton-Raphson** 迭代求解。

定义 discrepancy $\Delta\eta = \eta - \tau(u(\eta), v(\eta))$，迭代：

$$
\eta_{n+1} = \eta_n - \frac{\Delta\eta}{d(\Delta\eta)/d\eta}
$$

Jacobian：

$$
\frac{d(\Delta\eta)}{d\eta} = 1 - \tau_u \frac{du}{d\eta} - \tau_v \frac{dv}{d\eta}
$$

而 $\frac{du}{d\eta} = \frac{d\pi_u}{d\pmb{x}_c}\frac{d\pmb{x}_c}{d\eta}$，$\frac{dv}{d\eta}$ 类似。$\frac{d\pmb{x}_c}{d\eta}$ 可以从上面的 motion equations 解析推导，不依赖 camera model。$\frac{d\pi}{d\pmb{x}_c}$ 就是 EWA splatting 里那个 Jacobian。

对于 perspective camera：

$$
\frac{d\pi_{perspective}}{d\pmb{x}_c} = \begin{pmatrix} \frac{f_x}{z} & 0 & -\frac{f_x x}{z^2} \\ 0 & \frac{f_y}{z} & -\frac{f_y y}{z^2} \end{pmatrix}
$$

$f_x, f_y$ 是 focal lengths。对于 spherical camera (LiDAR)：

$$
\pi_{spherical}(x,y,z) = \left(\text{atan2}(y,x), \arcsin\frac{z}{\sqrt{x^2+y^2+z^2}}\right)
$$

对应 Jacobian 在 paper appendix 里给出。Paper 实测 Newton-Raphson 比 fixed-point iteration 快 1-2 次迭代，实践中"few iterations" 就足够。

**Intuition**：这个 rolling shutter 模型的优雅之处在于——把 RS 看成一个 implicit equation，而不是 discretize time 然后多次 render。每个 3D point 投影时通过迭代找到 self-consistent 的 observation time，从而避免 ghosting 和 motion blur 的近似。

### 2.4 Phase Modeling: 解决 Spherical Camera 的 Bimodal Projection

这是 XSIM 最 novel 的部分，也是 Figure 1, 4, 6 的核心。

**问题本质**：spinning LiDAR 的 azimuth 是周期函数：

$$
\varphi = \text{atan2}(y, x) + 2\pi k, \quad k \in \mathbb{Z}
$$

通常只取 $k=0$ 的 principal value $[-\pi, \pi]$。但是 combined with rolling shutter 和 ego motion，会出现两类 discontinuity：

1. **时间不连续** (Figure 4a)：即使 LiDAR 静止，$\varphi = -\pi$ 这一侧的 beam 是 $t = \tau_{start}$ 时扫描的，$\varphi = +\pi$ 这一侧是 $t = \tau_{end}$ 时扫描的。中间隔了整个 scan period（通常 100ms 量级）。如果 scene 里有 dynamic object 跨越这条线，object 会被观测两次，且两次观测之间 object 已经移动。

2. **空间不连续** (Figure 4b)：ego motion + rolling shutter 让同一个 world point 在 $+\pi$ 和 $-\pi$ 附近投影到 range image 上不同的位置。

**UT 的失败模式**：Unscented Transform 假设 projected distribution 可以用 single 2D Gaussian 近似。但当一个 3D Gaussian 跨越 $\varphi = \pm\pi$ 边界时，它的 sigma points 会落到 $+\pi$ 和 $-\pi$ 两侧。UT 计算这些 sigma points 的 weighted mean 和 covariance 时，会把两端"扯"到中间（接近 $\varphi = 0$），得到一个覆盖整个 range image 的 spurious 巨大 Gaussian（Figure 6 中间）。这导致 tiling 阶段大量 false tile-particle intersections，且 depth sorting 出错。

**XSIM 的 Solution**：Phase modeling。显式考虑 $k \in \{-1, 0, +1\}$ 三个 phase：

- **Central projection** $\pi_{central}$：用 $\tau_{mid}$ 作为初始时间，做 standard UT。
- **Negative shift** $\pi_{negative}$：用 $\tau_{start}$ 初始化，对 $\varphi \geq 0$ 的 sigma points 减 $2\pi$，把它们 wrap 到 $[-\pi, 0]$ 侧。
- **Positive shift** $\pi_{positive}$：用 $\tau_{end}$ 初始化，对 $\varphi < 0$ 的 sigma points 加 $2\pi$，wrap 到 $[0, \pi]$ 侧。

每个 phase 独立做一次 UT，得到独立的 2D mean, covariance, depth。如果两个 phase 都 valid（投影 range 与 visible image range 相交且 azimuth extent 小于 $\pi$），说明 particle 确实 bimodal，返回两个 2D Gaussian；否则返回 valid 的那一个。

Algorithm 2 里的 $\varphi_{ext} < \pi$ 判据是为了过滤掉 wrap 之后 extent 反而变大的退化情况。

**Intuition**：这个 mechanism 本质上是把 spherical projection 的 topology 显式编码进 splatting pipeline。EWA splatting 完全没法处理这种 cyclic wrap，因为它假设 image plane 是 $\mathbb{R}^2$。XSIM 通过 phase shift 把 spherical image 拓扑上展开成三个 overlapping 的 patches，每个 patch 内 UT 假设成立。

---

## 3. Scene Representation 和 Training

Scene 是 graph 结构：static background node + dynamic actor nodes (rigid + deformable + SMPL for humans)。Human 用 SMPL [Loper et al., 2015](https://smpl.is.tue.mpg.de/) 建模，这是 follow OmniRE [Chen et al., 2025b](https://arxiv.org/abs/2409.05102) 的做法。Deformable actor 用 instance-conditional MLP 在 actor local frame 里 deform Gaussians。

**Initialization**：从 LiDAR sweeps 出发，用 bounding box 分离 static/dynamic points，project 到 camera images 染色，对称车辆沿 longitudinal axis symmetrize，inverse-distance sphere sampling 补 LiDAR 盲区。

**Loss**：

$$
\mathcal{L} = \underbrace{\lambda\mathcal{L}_1 + (1-\lambda)\mathcal{L}_{SSIM}}_{camera} + \underbrace{\mathcal{L}_{depth}}_{LiDAR} + \mathcal{L}_{opacity} + \mathcal{L}_{reg}
$$

$\lambda = 0.2$（与 SplatAD 一致）。$\mathcal{L}_{depth}$ 是 rendered ray length 和 GT ray length 的 L1。$\mathcal{L}_{reg} = 0.01\mathcal{L}_{mask} + 0.01\mathcal{L}_{pose} + \mathcal{L}_{SMPL}$。

**Densification trick**：3DGUT 不直接产生 2D positional gradient（因为它在 3D space evaluate），所以用 $|\nabla_{\pmb{\mu}}| \cdot depth$ 替代 3DGS 原来的 2D gradient norm。但 LiDAR supervision 会在 ray 方向产生强 gradient，导致 over-densification。XSIM 的 fix：densification criteria 只从 RGB supervision 累积 gradient，再加 2D scale criteria 与原版 3DGS 对齐。Pruning 用 $\max(\sigma_c, \sigma_L)$。

---

## 4. 实验数据解析

### 4.1 主表 (Table 1) 关键数字

**Waymo (12 scenes)** Reconstruction:
| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CD↓ |
|---|---|---|---|---|
| SplatAD (CVPR25) | 27.74 | 0.8650 | 0.2807 | 0.82 |
| **XSIM** | **30.75** | **0.9030** | **0.2228** | **0.08** |

PSNR 提升 **+3.01 dB** 是非常显著的 gap（在这个领域 +0.5 dB 已经是大改进）。CD 从 0.82 降到 0.08，**降低 8.8x**，说明 LiDAR rendering 质量有质变。

**Waymo NVS** (训练每隔一帧，测试剩下的): XSIM 29.80 vs SplatAD 27.06，提升 +2.74 dB。

**Argoverse 2 (10 scenes)** Reconstruction:
| Method | PSNR↑ | CD↓ |
|---|---|---|
| NeuRAD | 26.46 | 2.43 |
| SplatAD | 28.71 | 2.78 |
| **XSIM** | **29.44** | **0.57** |

这里 CD 从 2.78 到 0.57，**降低 4.9x**。

**PandaSet (10 scenes)** Reconstruction:
- XSIM PSNR 29.05 vs SplatAD 28.69
- LPIPS 0.1872 vs SplatAD 0.1853（这里 XSIM 略输，paper 诚实地承认"competitive, second place"）

### 4.2 Ablation (Table 2)

在 6 个 Waymo scenes 上的 NVS ablation：

| Config | PSNR | SSIM | LPIPS | CD |
|---|---|---|---|---|
| Full XSIM | 30.03 | 0.8945 | 0.2122 | 0.21 |
| – Camera RS | 28.95 | 0.8761 | 0.2407 | 0.21 |
| – LiDAR opacity | 29.55 | 0.8888 | 0.2189 | 0.25 |
| – LiDAR RS | 29.05 | 0.8816 | 0.2296 | 0.31 |
| – Phase modeling | 29.32 | 0.8824 | 0.2245 | 0.28 |

观察：

- **Camera rolling shutter** 贡献最大 PSNR gain (+1.08)。这印证了 autonomous driving 中 RS 对 appearance 影响巨大。
- **LiDAR RS** 对 CD 影响显著（0.21 → 0.31）。这是 LiDAR 仿真保真度的关键。
- **Phase modeling** 单独贡献 0.07 PSNR 和 0.07 CD。看似不大，但 qualitative（Figure 1, 7）显示在 azimuth boundary 区域差别显著——这种 artifact 在 quantitative metric 上被大量"normal"区域稀释了。
- **Dual opacity** 同时改善 PSNR (0.48) 和 CD (0.04)，验证了 geometry/appearance decoupling 的价值。

### 4.3 Qualitative 关键观察

Figure 7 的 LiDAR rendering 对比尤其 informative：XSIM 保留了 LiDAR 特有的 ring pattern（不同 beam 之间的同心环结构），而 SplatAD 和 OmniRE 的 rendering 出现 scan-line 扭曲和 geometry 缺失，特别是行人的 reconstruction。这印证了 phase modeling + dual opacity 的组合效应。

Figure 3 的 depth map 显示 XSIM 的 depth 是 smooth 且 dense 的，而 baseline 出现 noise 和 hole。这说明 unified representation 让 LiDAR supervision 直接 help camera depth rendering，反之亦然。

Figure 2 的 lane shift 3m 实验（extrapolation test）是 sim-to-real transfer 的关键 metric。XSIM 在 3m lateral shift 下仍保持 consistency，说明 representation 没有 overfit training trajectory。

---

## 5. 与相关工作的 positioning

XSIM 处于几个研究 line 的交叉口：

1. **3DGUT** [Wu et al., 2025](https://arxiv.org/abs/2412.20402) — XSIM 的 technical foundation。3DGUT 本身是 CVPR 2025 的工作，解决了 EWA 在 nonlinear camera 下的 approximation 问题。
2. **SplatAD** [Hess et al., 2024](https://arxiv.org/abs/2411.16816) — 直接 baseline，CVPR 2025。SplatAD 也做 LiDAR + camera rendering with RS，但用 sensor-specific models。XSIM 的 unified formulation 是对 SplatAD 的超越。
3. **OmniRE** [Chen et al., 2025b](https://arxiv.org/abs/2409.05102) — ICLR 2025，提供 scene graph + SMPL human 的 representation。XSIM 复用其 human modeling。
4. **NeuRAD** [Tonderski et al., 2024](https://arxiv.org/abs/2311.17822) — CVPR 2024，NeRF-based 的 multi-sensor rendering，是 3DGS 系方法的强 baseline。
5. **EmerNerf** [Yang et al., 2024](https://arxiv.org/abs/2404.02162) — ICLR 2024，self-supervised spatio-temporal decomposition。
6. **HUGS** [Zhou et al., 2024b](https://arxiv.org/abs/2312.07058) — CVPR 2024，holistic urban scene understanding。
7. **HUGSim** [Zhou et al., 2024a](https://arxiv.org/abs/2412.01718) — closed-loop simulator。
8. **NeurONCap** [Ljungbergh et al., 2025](https://arxiv.org/abs/2410.23275) — ECCV 2024 (按 reference 标注 2025)，photorealistic closed-loop safety testing。

**Spherical splatting 相关 prior art**：
- [Huang et al., 2025](https://arxiv.org/abs/2410.01803) "On the error analysis of 3D Gaussian Splatting and an optimal projection strategy" — ECCV 2024，分析 3DGS 投影误差，提出 optimal projection strategy。XSIM 的 phase modeling 可以看作针对 spherical camera 的专项 fix。

**Rolling shutter in NeRF/3DGS**：
- 早期 RS modeling 多见于 visual SLAM 和 visual odometry，如 [Forssén & Ringaby 2010](https://link.springer.com/article/10.1007/s11263-010-0364-2)。
- NeuRAD 和 SplatAD 把 RS 引入 driving simulation，但 model 是 sensor-specific 的。XSIM 的 generalized RS formulation 把任意 $\pi(\cdot)$ 都 plug in。

**Unscented Transform 来源**：
- [Julier & Uhlmann 2004](https://ieeexplore.ieee.org/document/1271396) "Unscented Filtering and Nonlinear Estimation" — UT 的经典 reference。3DGUT 把 UT 引入 splatting 是聪明的 repurpose。

**SMPL**：
- [Loper et al., 2015](https://smpl.is.tue.mpg.de/) ACM ToG — skinned multi-person linear model，human pose/shape 的 de facto standard。

---

## 6. 代码实现与工程细节

代码开源于 https://github.com/whesense/XSIM (paper 里写 whesense，应该是 WHE / Sense 的拼写)。

关键工程点：

- **Custom CUDA kernels** 实现 camera 和 LiDAR rendering，unified pipeline 共享 forward/backward rasterization pass。
- **LiDAR non-uniform beam angles** 处理：tiling 时迭代 elevation tile boundaries（follow SplatAD）。
- **Optimization**：Adam，40000 iterations，warm-up 500 iterations + exponential decay。Position LR 从 1.6e-4 到 1.6e-6。
- **Datasets**：Waymo 12 scenes (≈19s sequences, 5 cameras 1920×1080 or 1920×886)，Argoverse 2 10 scenes (≈15.5s, 7 cameras 2048×1550, crop bottom 250px for ego-vehicle)，PandaSet 10 scenes (≈8s, 6 cameras 1920×1080)。

---

## 7. 我的 intuition 与联想

几个我觉得有意思的角度：

**1. UT 的局限是 distributional assumption**。UT 是 Gaussian belief propagation 的标准工具，但它假设 posterior 仍是 Gaussian。Phase modeling 本质上是 mixture of Gaussians，且 mixture components 数量由 sensor topology 决定。这个思路可以推广到其他 cyclic/discontinuous projection：fisheye 的 antipodal point、panoramic camera 的 seam、mirror-based sensor 的 reflection boundary。

**2. Dual opacity 是 physics-aware parameterization**。表面上看是"加一个参数"，但实质是承认不同 sensing modality 探测的是不同的物理量。LiDAR 是 time-of-flight of coherent IR pulse，camera 是 spectral radiance integration。它们的"透明度"在物理上就不该共享。这种 decoupling 思路可以推广到：event camera（log intensity change）、RGB-IR（不同波长）、ToF camera（multi-frequency phase）。

**3. RS 投影的 fixed-point formulation 很优雅**。传统 RS 处理要么 discretize time 然后 warp，要么用 approximation（如 [Ait-Aider et al. 2006](https://ieeexplore.ieee.org/document/1640808) 的 linearized SfM）。XSIM 把它写成 implicit equation 然后用 Newton-Raphson 解，等价于在每一帧 render 时做 "inverse RS"。这种 formulation 让 sensor model 和 motion model 解耦——任意 $\pi(\cdot)$ 都可以 plug in，只要能算 Jacobian。

**4. 与 3D Ray Tracing 的对比**。[Moenne-Loccoz et al., 2024](https://arxiv.org/abs/2410.18482) 的 3D Gaussian Ray Tracing 把 splatting 推向 path tracing。XSIM 走的是相反方向：在 splatting framework 内尽量逼近 ray-based rendering 的 fidelity（通过 UT + 3D evaluation），但保持 rasterization 的速度。两个方向的 convergence point 可能是 hybrid rendering。

**5. Sim-to-real 的下游价值**。XSIM 在 lane shift 3m 下的 consistency 是 closed-loop testing 的 prerequisite。后续可以接 [HUGSim](https://arxiv.org/abs/2412.01718) 这类 closed-loop simulator，或者 [Ljungbergh et al. 2025](https://arxiv.org/abs/2410.23275) 的 safety testing pipeline。

**6. Limitation 推测**（paper 没明说）：
- Constant velocity assumption 在 sharp turn / hard brake 时失效。高阶 motion model (constant acceleration, IMU-integrated) 可能是下一步。
- Phase modeling 只处理 azimuth wrap，elevation 边界（LiDAR top/bottom beam）可能也有类似问题，但 extent 通常小。
- LiDAR 的 intensity/return 仿真没有建模（只有 range）。真实的 intensity 受 material reflectance、incident angle、distance 衰减影响。这点 [SplatAD](https://arxiv.org/abs/2411.16816) 有建模，XSIM 看起来没 explicit intensity rendering。
- 物体间的 occlusion 在 3DGS 框架里是 implicit 的（alpha compositing），但对 LiDAR 这种 active sensor，multi-return（同一 beam 多次 return）的物理 modeling 还缺。

**7. 与 Kalman Filter / EKF 的联系**。Newton-Raphson 在 RS 投影上的使用，本质上和 EKF 里 Newton iteration on measurement equation 是同构的。$\eta = \tau(u(\eta), v(\eta))$ 可以看作 measurement equation $h(\eta) = 0$，Newton-Raphson 是 Gauss-Newton 的特例。这种 formulation 让 sensor model 可以接入 robotics estimator 的数学语言。

**8. Bimodal projection 在 stereo / multi-camera 上的潜在应用**。如果一个 3D Gaussian 同时被多个 camera 看到，且 camera 间 baseline 大，它的 2D projection 在不同 view 里差异巨大。Phase modeling 的 mixture 思路可以推广到 multi-view splatting，让一个 3D Gaussian 在每个 view 里独立 UT，而不是用全局一致的 2D Gaussian。

---

## 8. 总结

XSIM 是一个工程完成度很高的工作，三个 contributions 各自解决一个具体、可验证的问题：

1. **Generalized RS**：把 RS 投影写成 implicit equation + Newton-Raphson，让任意 $\pi(\cdot)$ 可以 plug in，统一了 camera 和 LiDAR 的 RS modeling。
2. **Phase modeling**：显式处理 spherical camera azimuth boundary 的 cyclic topology，把 UT 的 unimodal assumption 升级为 mixture。
3. **Dual opacity**：physics-aware parameterization，让 geometry 和 appearance 各自优化又不完全脱钩。

在 Waymo / Argoverse 2 / PandaSet 三个数据集上 SOTA，特别是 LiDAR CD 的 8.8x / 4.9x 降低 是 qualitative leap。代码开源对 community 是利好。

如果你想 build deeper intuition，我建议从两个方向切入：(a) 读 [3DGUT](https://arxiv.org/abs/2412.20402) 的 original paper，理解 UT 在 splatting 里如何 work；(b) 读 [SplatAD](https://arxiv.org/abs/2411.16816) 对比 sensor-specific RS modeling 的做法，看 XSIM 的 unification 实际节省了什么。然后再看 [HUGSim](https://arxiv.org/abs/2412.01718) 和 [NeurONCap](https://arxiv.org/abs/2410.23275) 了解 closed-loop 下游怎么用这些 simulator。

主要参考 links：
- XSIM 代码: https://github.com/whesense/XSIM
- 3DGUT: https://arxiv.org/abs/2412.20402
- SplatAD: https://arxiv.org/abs/2411.16816
- OmniRE: https://arxiv.org/abs/2409.05102
- NeuRAD: https://arxiv.org/abs/2311.17822
- HUGS: https://arxiv.org/abs/2312.07058
- HUGSim: https://arxiv.org/abs/2412.01718
- EmerNerf: https://arxiv.org/abs/2404.02162
- 3D Gaussian Ray Tracing: https://arxiv.org/abs/2410.18482
- On the error analysis of 3DGS: https://arxiv.org/abs/2410.01803
- Waymo Open Dataset: https://waymo.com/open/
- Argoverse 2: https://www.argoverse.org/
- PandaSet: https://scale.com/open-datasets/pandaset
- SMPL: https://smpl.is.tue.mpg.de/
- Unscented Transform (Julier & Uhlmann): https://ieeexplore.ieee.org/document/1271396
- EWA Splatting (Zwicker et al. 2001): https://www.cs.umd.edu/~zwicker/publications/EWAVolSplatting.pdf
