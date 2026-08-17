---
source_pdf: PhySG.pdf
paper_sha256: 03392fb59201e300aa8e72adbb814e2f1ff6f7325c89e7388a528931d6e1a463
processed_at: '2026-08-06T03:15:30-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PhySG: 用人话讲清楚这篇 paper

## 1. 一句话概括 PhySG 在干啥

你给 PhySG 几张同一个 glossy 物体在**同一种光照**下拍的多视角照片，它能把 **geometry、material BRDF、environment illumination** 全部拆开。拆完之后，你可以换个环境光重新渲染，也可以把塑料涂成金属，然后再合成新图。NeRF 只能让你换个角度看同一个东西，PhySG 能让你换个"世界"看同一个东西。

## 2. 为什么 NeRF 那一套做不到

NeRF、IDR、DVR 本质上是把物体表面每个点当成一个小灯泡。它们学一个 MLP，输入位置和视角，直接输出颜色。数学上这叫 **surface light field**。它能插值出新视角，因为它直接记住了"从这个角度看是这个颜色"。

但是它完全无法区分"这个点看起来红"是因为光照红，还是因为物体本身反照率高，还是因为高光刚好扫到这。所以你改不了光，也改不了材质。

PhySG 走物理路线，老老实实解 **rendering equation**：光从 environment map 打过来，撞到表面，按 BRDF 规则散射进相机。只要把 lighting、BRDF、geometry 三个 unknown 解出来，改任何一个变量都能重新算出图。这就是 inverse rendering。

## 3. 核心魔法：Spherical Gaussians (SG)

Rendering equation 里有个半球积分，一般要用 Monte Carlo path tracing 采样几万条 ray 才能算准。而且可微 path tracing 边界处理极麻烦，参考 redner [Li et al. 2018, https://dl.acm.org/doi/10.1145/3272127.3275106]。

PhySG 的核心 trick 是：**把 integrand 里的每一项都用 Spherical Gaussian 表示**。SG 就是球面上的一个"凸起"，像探照灯的光斑：

$$G(\boldsymbol{\nu}; \boldsymbol{\xi}, \lambda, \boldsymbol{\mu}) = \boldsymbol{\mu} \, e^{\lambda(\boldsymbol{\nu}\cdot\boldsymbol{\xi} - 1)}$$

变量解释：
- $\boldsymbol{\nu} \in \mathbb{S}^2$：球面上的输入方向（你想算哪个方向的光强）
- $\boldsymbol{\xi} \in \mathbb{S}^2$：**lobe axis**，探照灯指向哪
- $\lambda \in \mathbb{R}_+$：**lobe sharpness**，光斑多集中。$\lambda$ 大=激光笔，$\lambda$ 小=泛光灯
- $\boldsymbol{\mu} \in \mathbb{R}_+^n$：**lobe amplitude**，亮度颜色（RGB 三维）

关键数学性质：**两个 SG 相乘在球面上的积分有 closed-form 解析解**。所以环境光（128 个 SG 混合）乘上 BRDF（1 个 SG）乘上 cos 项（也近似成 SG），可以直接套公式算出结果，完全不用采样。这就是 PhySG 高效且可微的根本原因。

对比 Spherical Harmonics (SH)：SH 也有 closed form，但 SH 是 low-frequency basis，表达不了 sharp 的 specular highlight。SG 的 sharpness 可调，既泛光也能尖锐。参考 Ramamoorthi & Hanrahan 2001 [https://doi.org/10.1145/383259.383309] 和 Wang et al. 2009 [https://dl.acm.org/doi/10.1145/1661412.1614292]。

## 4. 架构怎么转的（对应 Figure 2）

整个 pipeline 就是一条 ray 的流水线：

```
Camera ray r = o + t·d
    │
    ▼
[Geometry 模块]
SDF MLP: S(x; Θ)  →  sphere tracing 找交点 x
法线 n = ∇_x S  (auto-diff)
    │ x, n
    ▼
[Appearance 模块]
diffuse albedo MLP: a(x; Φ) → 底色
specular BRDF: {λ, μ}  (单个 SG, isotropic, monochrome)
environment map: {ξ_k, λ_k, μ_k}_{k=1..128}  (128 个 SG)
    │
    ▼
[SG Renderer]
把 L_i × BRDF × cos 全当 SG，closed-form 积分
    │
    ▼
L_o(ω_o; x) → 跟 GT 比 → backprop 更新 Θ, Φ, {λ,μ}, {ξ_k,λ_k,μ_k}
```

几何用 SDF 而非 occupancy field，因为 sphere tracing 每条 ray 大概只要 ~10 次 MLP 评估，occupancy 需要 100+ 次 root-finding [Niemeyer et al. 2020, https://arxiv.org/abs/1911.12055]。法线直接是 SDF gradient，自动满足 shape-normal 约束。

## 5. 关键公式逐个讲直觉

### 5.1 Environment map

$$L_i(\omega_i) = \sum_{k=1}^{128} G(\omega_i; \boldsymbol{\xi}_k, \lambda_k, \boldsymbol{\mu}_k)$$

128 个手电筒叠加，拼出整张环境光。lobe axis 用 spherical Fibonacci lattice 均匀初始化 [Keinert et al. 2015, https://dl.acm.org/doi/10.1145/2816795.2818131]，这是个低 discrepancy 序列，比纯随机均匀。

### 5.2 Specular BRDF

用简化的 Disney BRDF [Burley 2012, https://blog.selfshadow.com/publications/s2012-shading-course/]：

$$f_s(\omega_o, \omega_i) = \mathcal{M}(\omega_o, \omega_i) \cdot \mathcal{D}(\mathbf{h})$$

- $\mathbf{h} = (\omega_o + \omega_i)/\|\omega_o + \omega_i\|_2$：**half vector**，视角和入射的中间方向
- $\mathcal{D}$：**NDF** (normal distribution function)，microfacet 法线分布，描述"有多少 microfacet 的法线刚好指向 $\mathbf{h}$ 能把光反射进眼睛"
- $\mathcal{M}$：Fresnel + shadowing/masking 项

$\mathcal{D}$ 用一个 SG 表示，axis 沿 surface normal $\boldsymbol{\xi}=\mathbf{n}$（因为假设 isotropic），amplitude monochrome（三个 RGB 分量一样）。

为了能跟环境光做 closed-form 积分，需要做球面 warping 和常数化近似：

$$\mathcal{D}_{\mathbf{x}}(\mathbf{h}) = G\!\left(\mathbf{h};\, \mathbf{n},\, \frac{\lambda}{4\mathbf{h}\cdot\omega_o},\, \mu\right)$$

$4(\mathbf{h}\cdot\omega_o)$ 是 Jacobian 调整因子，因为从 microfacet 法线空间 $\mathbf{h}$ 变到入射方向空间 $\omega_i$ 时 measure 会变。

$$\mathcal{M}_{\mathbf{x}}(\omega_o, \omega_i) \approx \mathcal{M}(\omega_o, 2(\omega_o\cdot\mathbf{n})\mathbf{n} - \omega_o)$$

把 shadowing/masking 用 ideal mirror reflection direction $r_o = 2(\omega_o\cdot\mathbf{n})\mathbf{n} - \omega_o$ 处的值固定，避免它破坏 SG 形式。

### 5.3 cos 项的 SG 近似

$$\omega_i \cdot \mathbf{n} \approx G(\omega_i; 0.0315, \mathbf{n}, 32.7080) - 31.7003$$

$\omega_i \cdot \mathbf{n}$ 是 cos，本质上是 hemisphere 上的函数。用 sharpness=32.7 的窄 SG 减去一个常数来逼近。数值来自 Meder & Bruderlin 2018 [https://link.springer.com/chapter/10.1007/978-3-030-00692-1_1]。

### 5.4 最终积分

$$L_o(\omega_o; \mathbf{x}) = \int_{\Omega} L_i(\omega_i) \left(\frac{\mathbf{a}}{\pi} + f_s(\omega_o, \omega_i; \mathbf{x})\right) (\omega_i\cdot\mathbf{n}) \, d\omega_i$$

全都是 SG，所以 diffuse 部分是 128 次 SG×SG 积分，specular 部分也是 128 次。一次 forward pass 几百次浮点运算就完了，比 Monte Carlo 快几个数量级。

SG×SG 积分 closed form：

$$\int_{\mathbb{S}^2} G_1 \cdot G_2 \, d\nu = \frac{4\pi \mu_1 \mu_2}{e^{\lambda_1+\lambda_2-\mu_{12}} - e^{-(\lambda_1+\lambda_2-\mu_{12})}}$$

其中 $\mu_{12} = \lambda_1\lambda_2(1-\xi_1\cdot\xi_2)/(\lambda_1+\lambda_2)$ 是等效 sharpness。

## 6. Loss Function 三件套

$$\ell = \ell_{\text{recon}} + \beta_1 \ell_{\text{bg}} + \beta_2 \ell_{\text{eikonal}}$$

**Reconstruction loss**：rendered color 跟 GT 的 L1 距离。L1 比 L2 对 outlier 鲁棒。

**Background loss**：non-object ray 上 SDF 应该非负（背景在物体外）。用 softplus 平滑版 ReLU：

$$\ell_{\text{bg}} = \frac{\ln(1+e^{-\alpha S_i^{nobj}})}{\alpha}$$

$\alpha$ 从 50 curriculum 涨到 1600，越涨越接近硬 ReLU，让优化先平滑后收紧。

**Eikonal loss**：SDF 定义要求 $\|\nabla S\| = 1$ 处处成立。在 bounding box 随机采样点惩罚梯度模长偏离 1：

$$\ell_{\text{eikonal}} = \frac{1}{N_x}\sum_i (\|\nabla_{\mathbf{x}_i} S\|_2 - 1)^2$$

还有 patch-based normal smoothness loss（weight=10），在 2×2 patch 内惩罚法线方差。

## 7. 实验数据表解读

### Table 1: 合成数据 PhySG 自评

| 任务 | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|
| Diffuse albedo | 0.0339 | 0.989 | 33.43 |
| Novel view | 0.0170 | 0.990 | 35.93 |
| Relighting | 0.0227 | 0.988 | 33.25 |

Surface normal error 只有 2.528°，说明几何恢复得很准。Relighting PSNR 33.25 说明换光之后跟 GT 渲染差异很小。

### Table 2: 几何质量对比

| Method | Normal Error (°) | L1 Chamfer |
|---|---|---|
| PhySG | 2.528 | 0.00142 |
| NeRF | 36.05 | 0.01650 |
| IDR | 2.207 | 0.00136 |
| DVR | 38.90 | 0.13800 |

NeRF 几何很差是因为 volumetric 表示颜色不聚拢在表面。DVR 几何差是因为没建模 view-dependent，glossy 数据把它搞晕了。PhySG 跟 IDR 持平，而且 IDR 拿不到物理材质。

### Table 3: Novel view 质量

在真实数据上 PhySG 的指标略低于 IDR。作者分析是因为 PhySG 的物理 BRDF 模型有 bias，跟真实材质不完全吻合；而 IDR 的 surface light field 没 bias 但牺牲了编辑能力。这是 bias-variance tradeoff 的经典体现。

## 8. 关键 Intuition：为什么 specular 是反演的救星

如果物体纯 Lambertian（完全 diffuse），会出现 **lighting-texture ambiguity**：

$$\text{observed} = \text{albedo} \times \text{shading}$$

无数种 albedo 和 lighting 组合能给出同一张图。这跟 intrinsic image decomposition [Land & McCann 1971] 是亲戚问题。

specular highlight 提供 view-dependent 信号：高光位置随视角移动，这种动态信息能 disentangle lighting 和 material。Fig. 6 显示即使 roughness R=0.25（很粗糙）也能恢复 environment map，虽然模糊。R 太小（几乎 mirror）反而难，因为高光太尖锐，SG 表达不够。

## 9. 初始化为什么关键

作者强调如果 environment map 初始化太亮或太暗，diffuse albedo MLP 会 stuck 在全 0 或全 1。原因：gradient 信号失衡。如果光太强，albedo 减一点就 over-saturated，gradient 推它往 0 走；反之亦然。作者用 median intensity 归一化到 0.5 来稳住优化。

SDF 初始化用 IGR 方法 [Gropp et al. 2020, https://arxiv.org/abs/2002.10099] 让初始形状是个球，避免随机初始化导致的塌陷。

specular $\lambda$ 初始化在 [95, 125]，$\mu$ 在 [0.18, 0.26]，对应 typical glossy plastic。

## 10. Limitations 与直觉解释

### 10.1 无 indirect illumination

SG 积分只算 direct lighting，物体内部互反射没算。比如一个碗的内壁，光会弹好几次。所以 PhySG 只适合 single object with simple geometry。scene-level 需要 differentiable path tracing + deferred neural lighting [Gao et al. 2020, https://arxiv.org/abs/2008.08596]。

### 10.2 单一 specular BRDF

整个 object 共享一组 specular 参数。因为 scale ambiguity：lighting 强度 × reflectance = observed intensity，固定一个才能解另一个。未来用 learning-based prior 可能打破这个限制，类似 NeRF++ [Zhang et al. 2020] 用先验处理 unbounded scenes。

### 10.3 Isotropic 假设

lobe axis 必须沿 normal。拉丝金属、绸缎这种 anisotropic 不行。可以用 anisotropic SG mixture [Xu et al. 2013, https://dl.acm.org/doi/10.1145/2508363.2508386]，但参数量暴涨，优化更难。

## 11. 后续工作联想

PhySG 之后这条路很热：

- **NeRO [Liu et al. 2023, https://liuyuan-pal.github.io/NeRO/]**：结合 NeuS 和 PBR，用 split-sum approximation（Unreal Engine 4 那套 pre-filtered environment map [Karis 2013, https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf]）代替 SG，质量更高。
- **Ref-NeRF [Verbin et al. 2022, https://arxiv.org/abs/2111.32657]**：在 NeRF 里加 reflected radiance parameterization，用 reflection direction 查颜色，但还是 surface light field 范畴。
- **GaussianShader / Relightable 3DGS [2023+]**：把 SG 思想用到 3D Gaussian Splatting [Kerbl et al. 2023, https://arxiv.org/abs/2301.13025] 上，飞快且可 relight。
- **TensoIR [Zhuang et al. 2023]**：用 tensoRF 加速 inverse rendering，处理大场景。
- **NeRFactor [Zhang et al. 2021, https://arxiv.org/abs/2106.07070]**：跟 PhySG 同时期，用 cube 2D 参数化 environment map + visibility MLP，更通用但更慢。
- **Differentiable path tracing (redner, Mitsuba 2/3)**：完全物理但慢，PhySG 用近似换速度。

SG vs SH 的取舍一直贯穿图形学：SH 低频优雅，SG 高频灵活。PhySG 选 SG 是因为它要处理 glossy。如果只要 diffuse，SH 就够了 [Ramamoorthi & Hanrahan 2001]。

## 12. 对你（Karpathy）的 intuition building

从 deep learning 角度看，PhySG 是个**用 domain knowledge 强约束神经网络**的范例。NeRF 让 MLP 自由学 appearance，结果 overfit 到 surface light field，失去物理可编辑性。PhySG 把 appearance 用 SG 参数化，等于强加了一个 inductive bias："appearance 必须能分解成光照×BRDF×几何的 closed-form 积分"。这限制了 expressivity，但换来了 interpretability 和 editability。

这跟你喜欢的 "micrograd" 哲学有点像：把反向传播拆到最简，看 gradient 怎么流。PhySG 的可微性也是手工推导：sphere tracing 本身 forward only，但交点 x 的 gradient 通过 implicit differentiation $\partial t^*/\partial \Theta = -(\partial S/\partial \Theta)/(\mathbf{n}\cdot\mathbf{d})$ 手工接入计算图。几何模块和 appearance 模块的 gradient 各自独立推导，再 chain rule 串联。

如果想动手复现，看作者开源代码 https://github.com/Kai-46/PhySG，核心就是 `sg_render.py` 里那几百行 closed-form 积分。

## References

- PhySG: https://kaizhang.cn/PhySG/ | https://arxiv.org/abs/2104.00674
- Code: https://github.com/Kai-46/PhySG
- NeRF: https://arxiv.org/abs/2003.08934
- IDR: https://arxiv.org/abs/2006.09662
- IGR: https://arxiv.org/abs/2002.10099
- redner: https://dl.acm.org/doi/10.1145/3272127.3275106
- SG rendering (Wang 2009): https://dl.acm.org/doi/10.1145/1661412.1614292
- Disney BRDF: https://blog.selfshadow.com/publications/s2012-shading-course/
- Unreal Engine 4 split-sum: https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
- NeRO: https://liuyuan-pal.github.io/NeRO/
- Ref-NeRF: https://arxiv.org/abs/2111.32657
- 3DGS: https://arxiv.org/abs/2301.13025
- Anisotropic SG: https://dl.acm.org/doi/10.1145/2508363.2508386
- NeRFactor: https://arxiv.org/abs/2106.07070
- Mitsuba: https://www.mitsuba-renderer.org/

---

# PhySG: 基于 Spherical Gaussians 的物理可微渲染逆向管线详解

## 1. 核心问题与动机

PhySG 解决的是一个 long-standing 的 inverse rendering 问题：给定一组静态光照下拍摄的 posed multi-view RGB images，jointly recover **geometry + material BRDF + environment illumination**。这个 setting 极具挑战性，因为 appearance 是 material 和 lighting 的 entanglement，需要 disentangle 出两个 unknown。

之前的方法要么假设 RGBD 输入，要么要求 varying illumination（比如 co-located flashlight 或者 turntable），要么用 surface light field（像 NeRF 那样把 object 当成自发光源）。PhySG 的突破点在于：**用 Spherical Gaussians (SG) 把 rendering equation 解析地 closed-form evaluate**，从而避开 Monte Carlo path tracing 的高昂 cost，做到 end-to-end 可微。

## 2. Pipeline 架构解析

整个 pipeline 分三个模块，全部可微：

```
Camera ray r = o + t·d
        │
        ▼
┌─────────────────────────────┐
│ Geometry: SDF MLP S(x;Θ)     │  ← sphere tracing 求交点 x
│  - 8 layers, width 512       │  - 法线 n = ∇_x S (auto-diff)
│  - 6 freq positional encoding│
│  - skip connection @ 4th     │
└─────────────────────────────┘
        │ x, n
        ▼
┌─────────────────────────────┐
│ Appearance:                 │
│  ├ Diffuse albedo MLP a(x;Φ)│  ← 4 layers, width 512, 10 freq
│  ├ Specular BRDF {λ, μ}     │  ← 单个 SG lobe (isotropic)
│  └ Env map {ξ_k,λ_k,μ_k}    │  ← 128 SGs mixture
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ SG Renderer (closed form)   │  ← 所有项都是 SG → 乘积积分有 closed form
└─────────────────────────────┘
        │
        ▼
    L_o(ω_o; x)  →  与 GT 比较 →  backprop
```

关键 intuition：**只要把 rendering equation 里的每一项都变成 SG，乘积的积分就有解析解**。这避免了 Monte Carlo 采样的 high variance 和不可微的 boundary term（参考 redner [Li et al. 2018]）。

## 3. Geometry: SDF + MLP

### 3.1 为什么用 SDF 而不是 occupancy field

SDF 相比 occupancy [Mescheder et al. 2019; Niemeyer et al. 2020 DVR] 的优势：

- **Sphere tracing 高效**：每条 ray 大约只需要 ~10 次 MLP evaluation（occupancy 需要 100+ 次 root-finding）
- **法线自动获得**：surface normal = SDF gradient $\mathbf{n} = \nabla_{\mathbf{x}} S$，满足几何约束
- **内存友好**：MLP 参数远少于 voxel grid，且 infinite resolution

### 3.2 可微 ray casting

Sphere tracing 本身**不需要**可微（这是个 forward 过程，类似 mesh rasterization）。但需要把梯度 backprop 到 x 和 n，这里用 implicit differentiation [Yariv et al. 2020 IDR; Niemeyer et al. 2020 DVR]：

设 ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$，交点满足 $S(\mathbf{o} + t^*\mathbf{d}; \Theta) = 0$，对 Θ 求导：

$$\frac{\partial t^*}{\partial \Theta} = -\frac{\partial S / \partial \Theta}{\partial S / \partial t}\bigg|_{t=t^*}$$

分母 $\partial S/\partial t = \nabla_{\mathbf{x}} S \cdot \mathbf{d} = \mathbf{n} \cdot \mathbf{d}$，就是法线和 ray direction 的点积。

## 4. Appearance Modeling: Spherical Gaussians 是核心

### 4.1 SG 定义

n-dimensional Spherical Gaussian:

$$G(\boldsymbol{\nu}; \boldsymbol{\xi}, \lambda, \boldsymbol{\mu}) = \boldsymbol{\mu} \, e^{\lambda(\boldsymbol{\nu}\cdot\boldsymbol{\xi} - 1)}$$

变量含义：
- $\boldsymbol{\nu} \in \mathbb{S}^2$：**输入方向**（球面上的单位向量），例如 incident direction $\omega_i$
- $\boldsymbol{\xi} \in \mathbb{S}^2$：**lobe axis**，SG 的"中心方向"，lobe 在这个方向达到峰值 $\boldsymbol{\mu}$
- $\lambda \in \mathbb{R}_+$：**lobe sharpness**，控制 lobe 的宽度。$\lambda$ 越大，lobe 越尖锐集中；$\lambda \to 0$ 时退化为常数
- $\boldsymbol{\mu} \in \mathbb{R}_+^n$：**lobe amplitude**，对于 RGB 图像 $n=3$，是颜色向量

注意 $e^{\lambda(\boldsymbol{\nu}\cdot\boldsymbol{\xi}-1)}$ 在 $\boldsymbol{\nu}=\boldsymbol{\xi}$ 时为 1（达到峰值），当 $\boldsymbol{\nu} \perp \boldsymbol{\xi}$ 时为 $e^{-\lambda}$（很小）。

### 4.2 Environment map 表示

用 $M=128$ 个 SG 混合：

$$L_i(\omega_i) = \sum_{k=1}^{M} G(\omega_i; \boldsymbol{\xi}_k, \lambda_k, \boldsymbol{\mu}_k)$$

可优化参数：$\{\boldsymbol{\xi}_k, \lambda_k, \boldsymbol{\mu}_k\}_{k=1}^M$，共 $128 \times (2+1+3) = 768$ 维。

初始化用 spherical Fibonacci lattice 均匀分布在 unit sphere 上 [Keinert et al. 2015]，这是个采样低 discrepancy 序列的技巧。

### 4.3 Specular BRDF 表示

使用简化 Disney BRDF：

$$f_s(\omega_o, \omega_i) = \mathcal{M}(\omega_o, \omega_i)\, \mathcal{D}(\mathbf{h})$$

变量含义：
- $\mathbf{h} = (\omega_o + \omega_i)/\|\omega_o + \omega_i\|_2$：**half vector**，反射方向中间方向
- $\mathcal{M}$：Fresnel + shadowing/masking 项（G term）
- $\mathcal{D}$：normal distribution function (NDF)，描述 microfacet 法线分布

NDF 用一个 SG 表示：

$$\mathcal{D}(\mathbf{h}) = G(\mathbf{h}; \boldsymbol{\xi}, \lambda, \boldsymbol{\mu})$$

由于假设 **isotropic** specular，$\boldsymbol{\xi} = \mathbf{n}$（lobe axis 沿 surface normal）。又因为 **monochrome**（三个 RGB 通道 specular 系数相同），$\boldsymbol{\mu}$ 三个分量相同。

可优化参数：$\{\lambda, \boldsymbol{\mu}\}$（标量 sharpness 和标量 albedo），非常紧凑。

### 4.4 Warping 与关键近似

对每个 surface point x，half vector $\mathbf{h}$ 依赖于 $\omega_i$，所以 $\mathcal{D}$ 需要做球面 warping。PhySG 用近似：

$$\mathcal{D}_{\mathbf{x}}(\mathbf{h}) = G\!\left(\mathbf{h};\, \mathbf{n},\, \frac{\lambda}{4\mathbf{h}\cdot\omega_o},\, \mu\right)$$

变量：$4(\mathbf{h}\cdot\omega_o)$ 是 Jacobian 调整因子，因为 microfacet BRDF 的 $\mathcal{D}(\mathbf{h})$ 转换到 $\omega_i$ 空间时需要除以 $4(\mathbf{h}\cdot\omega_o)^3$，PhySG 做了简化处理。

$\mathcal{M}$ 项近似为常数（在 point x 处固定）：

$$\mathcal{M}_{\mathbf{x}}(\omega_o, \omega_i) \approx \mathcal{M}(\omega_o, 2(\omega_o\cdot\mathbf{n})\mathbf{n} - \omega_o)$$

这里 $2(\omega_o\cdot\mathbf{n})\mathbf{n} - \omega_o$ 是 ideal mirror reflection direction $r_o$，把 shadowing/masking 用 specular direction 的值代替。

### 4.5 cos term 的 SG 近似

rendering equation 里的 $\omega_i \cdot \mathbf{n}$（cosine term）也被一个 SG 近似：

$$\omega_i \cdot \mathbf{n} \approx G(\omega_i; 0.0315, \mathbf{n}, 32.7080) - 31.7003$$

这里 0.0315 是 amplitude（很小），32.7080 是 sharpness，31.7003 是常数 offset 用于让 SG 更贴合 cos 函数的形状。这个近似来自 Meder & Bruderlin 2018 [http://link.springer.com/chapter/10.1007/978-3-030-00692-1_1]。实际上 cos 在 lobe axis 方向值是 1，sharpness 32 表示 lobe 非常集中（接近 hemisphere-only）。

### 4.6 Rendering Equation 的 closed-form 积分

最终：

$$L_o(\omega_o; \mathbf{x}) = \int_{\Omega} \underbrace{L_i(\omega_i)}_{\text{SG mixture}} \underbrace{\left(\frac{\mathbf{a}}{\pi} + f_s(\omega_o, \omega_i; \mathbf{x})\right)}_{\text{diffuse + SG BRDF}} \underbrace{(\omega_i\cdot\mathbf{n})}_{\text{SG approx}} \, d\omega_i$$

diffuse 项：

$$L_o^{\text{diff}} = \frac{\mathbf{a}}{\pi} \int_{\Omega} L_i(\omega_i) (\omega_i\cdot\mathbf{n})\, d\omega_i$$

SG × SG 的积分有 closed form：

$$\int_{\mathbb{S}^2} G(\nu;\xi_1,\lambda_1,\mu_1) \cdot G(\nu;\xi_2,\lambda_2,\mu_2)\, d\nu = \frac{4\pi \mu_1 \mu_2}{e^{\lambda_1+\lambda_2-\lambda_{12}} - e^{-(\lambda_1+\lambda_2-\lambda_{12})}}$$

其中 $\lambda_{12} = \lambda_1 + \lambda_2 - \frac{\lambda_1\lambda_2(1-\xi_1\cdot\xi_2)}{\lambda_1 + \lambda_2}$ 是合并后的 lobe sharpness。

specular 项类似处理，因为 environment map 是 128 个 SG 的和，BRDF 是 1 个 SG，所以总共 128 次 closed-form 积分就能求出 specular contribution。

## 5. Loss Function

完整 loss:

$$\ell = \underbrace{\frac{1}{N_{obj}}\sum_{i=1}^{N_{obj}} \|\mathbf{c}_i^{obj} - \mathbf{c}_i^{gt}\|_1}_{\text{L1 reconstruction}} + \beta_1 \underbrace{\frac{1}{N_{nobj}}\sum_{i=1}^{N_{nobj}} \frac{\ln(1+e^{-\alpha S_i^{nobj}})}{\alpha}}_{\text{non-object SDF 应非负}} + \beta_2 \underbrace{\frac{1}{N_x}\sum_{i=1}^{N_x} \big(\|\nabla_{\mathbf{x}_i} S\|_2 - 1\big)^2}_{\text{Eikonal}}$$

变量：
- $N_{obj}=2048-N_{nobj}$: object pixel 数量
- $N_{nobj}$: non-object pixel 数量
- $N_x=1024$: bounding box 内随机采样点
- $S_i^{nobj}$: non-object ray 上 100 个均匀采样点中 SDF 的最小值
- $\alpha$: softplus 参数，从 50 逐渐增长到 1600（curriculum），控制 non-object penalty 的硬度
- $\beta_1=100$, $\beta_2=0.1$: 权重

Eikonal term $\|\nabla S\|=1$ 是 SDF 的定义性质（梯度模长处处为 1），是 regularization。

还有 patch-based normal smoothness loss（weight=10），在 2×2 patch 内 penalize normal 方差。

## 6. 实验数据表解读

### Table 1: 合成数据定量

| Task | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|
| Diffuse albedo | 0.0339 | 0.989 | 33.43 |
| Novel view | 0.0170 | 0.990 | 35.93 |
| Relighting | 0.0227 | 0.988 | 33.25 |

Surface normal error: 2.528° (非常小)

### Table 2: 几何对比

| Method | Surface Normal Error (°) | L1 Chamfer |
|---|---|---|
| PhySG (Ours) | 2.528 | 0.00142 |
| NeRF | 36.05 | 0.01650 |
| IDR | 2.207 | 0.00136 |
| DVR | 38.90 | 0.13800 |

PhySG 几何质量接近 IDR（state-of-art SDF 反演），远超 NeRF 和 DVR。

### Table 3: Novel view 对比

NeRF 在 view extrapolation 上表现差（volumetric 表达不能很好聚拢在 surface 上）；DVR 不支持 view-dependent，glossy 数据上完全失败；IDR 模型 view-dependence 但缺乏物理 appearance 模型，specular highlight 不准确；PhySG 因为有物理 prior，specular extrapolation 最合理。

## 7. 关键 Intuition

### 7.1 为什么 SG 闭环形式积分能 work

SG 的关键性质：**SG × SG 仍然是某种 closed-form 可积的形式**。这类似于 Spherical Harmonics (SH) 乘积有 closed form，但 SH 是 low-frequency basis，表达不了 sharp lobe。SG 可以表达 sharp lobe（specular BRDF 和 point light），同时还能解析积分。

参考 Ramamoorthi & Hanrahan 2001 [https://doi.org/10.1145/383259.383309] 的 SH irradiance，以及 Wang et al. 2009 SG work [https://dl.acm.org/doi/10.1145/1661412.1614292]。

### 7.2 Specular 是 lighting/material 反演的关键信号

如果物体是纯 Lambertian，会出现 **lighting-texture ambiguity**：albedo × lighting = observed color，无法分离。specular highlight 提供了 view-dependent 信号，是 disentangle 的关键。这也是为什么 PhySG 假设 glossy 物体。Fig. 6 显示即使 roughness R=0.25（比较粗糙）也能恢复 environment map，虽然更模糊。

### 7.3 Scale ambiguity

inverse rendering 有 inherent 的 scale ambiguity：lighting 强度 × material 反射率 = observed intensity。PhySG 通过固定 specular BRDF 假设（monochrome, shared across object）部分缓解这个问题，并在评估时用 channel-wise median scaling 对齐预测和 GT:

$$s_r = \text{Median}(I_r / \hat{I}_r)$$

### 7.4 初始化的重要性

paper 强调初始化很关键：
- SDF 初始化成 sphere [Gropp et al. 2020 IGR]
- diffuse albedo 初始化成 0.5（中等灰度）
- specular $\lambda \in [95,125]$，$\mu \in [0.18, 0.26]$
- environment map 用 Fibonacci lattice 均匀分布，amplitude scaling 到 median pixel intensity ~0.5

如果 environment map 初始化太亮或太暗，diffuse albedo MLP 会 stuck 在全 0 或全 1（gradient 信号失衡导致）。

## 8. Limitations 与后续方向

1. **无 indirect illumination**：SG 近似只处理 direct lighting，不能处理 interreflection、occlusion。scene-level 数据可能需要 differentiable path tracing + deferred neural lighting [Gao et al. 2020]。
2. **单 specular BRDF 假设**：因为 scale ambiguity，整个 object 共享一组 specular 参数，且 monochrome。spatially-varying specular 需要 learning-based prior 来解决 ambiguity。
3. **Isotropic**：无法表达 anisotropic BRDF（如拉丝金属），可以用 anisotropic SG mixture [Xu et al. 2013, https://dl.acm.org/doi/10.1145/2508363.2508386]。

## 9. 与后续工作的联系（相关联想）

- **NeRV [Bi et al. 2021]**：用 neural radiance 做类似的 inverse rendering，但用 volumetric 表示
- **Mip-NeRF 360 / Ref-NeRF [Verbin et al. 2022]**：在 NeRF 上加 view-dependent appearance（reflected radiance），但仍是 surface light field 范畴
- **NeRO [Liu et al. 2023]**：组合 NeuS + PBR，更高质量反演，用 split-sum approximations 代替 SG
- **TensoIR [Zhuang et al. 2023]**：用 tensoRF 加速 inverse rendering
- **Wang et al. 2023 "PhySG++" 方向**：后续工作用更复杂的 BRDF 表示（如 GGX），但牺牲了 closed-form 优势
- **3D Gaussian Splatting [Kerbl et al. 2023]** 与 relightable 3DGS：把 SG 思想用在 3DGS 上做 relighting（如 GaussianShader、Relightable 3D Gaussians）

## References

- **PhySG 原文**: Kai Zhang et al., "PhySG: Inverse Rendering with Spherical Gaussians for Physics-based Material Editing and Relighting", CVPR 2021. https://arxiv.org/abs/2104.00674
- **Project page**: https://kaizhang.cn/PhySG/
- **Code**: https://github.com/Kai-46/PhySG
- **NeRF**: Mildenhall et al., ECCV 2020. https://arxiv.org/abs/2003.08934
- **IDR**: Yariv et al., NeurIPS 2020. https://arxiv.org/abs/2006.09662
- **IGR (SDF init)**: Gropp et al. https://arxiv.org/abs/2002.10099
- **SG rendering (Wang 2009)**: https://dl.acm.org/doi/10.1145/1661412.1614292
- **redner (MC differentiable renderer)**: Li et al. SIGGRAPH Asia 2018. https://dl.acm.org/doi/10.1145/3272127.3275106
- **Mitsuba renderer**: https://www.mitsuba-renderer.org/
- **Disney BRDF**: Burley 2012. https://blog.selfshadow.com/publications/s2012-shading-course/
- **Anisotropic SG**: Xu et al. 2013. https://dl.acm.org/doi/10.1145/2508363.2508386
- **Ref-NeRF**: Verbin et al. CVPR 2022. https://arxiv.org/abs/2111.32657
- **3DGS**: Kerbl et al. SIGGRAPH 2023. https://arxiv.org/abs/2301.13025
