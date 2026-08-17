---
source_pdf: GS3 Efficient Relighting with Triple Gaussian Splatting.pdf
paper_sha256: 86a123a30815e47b76b8309e988d5ad6ec8ba7affd40022b20ab80e08b70e668
processed_at: '2026-08-04T22:53:51-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GS³ 用人话说

## 1. 这篇 paper 到底想干啥

你有一堆 object 的照片，每张照片用**一个点光源**从不同角度照着拍。现在要重建这个 object 的 3D 表征，之后随便换个视角、换个光照方向，都能**实时**渲染出逼真的图。

就是这么个事。

---

## 2. 为什么之前的 3DGS 不行

原始 3DGS [Kerbl et al. 2023] 里面每个 Gaussian 挂一坨 **Spherical Harmonics (SH)** 来表示颜色。问题是 SH 把"光照"和"材质"**焊死**了——它存的是"在这个固定环境光下从这个角度看是什么颜色"。换个光，整个 SH 就废了，因为材质本身没被显式拆出来。

打个比方：3DGS 像拍一张 360 全景照，光照固定在那一刻；你想把太阳挪到另一边重新打光，没门，因为照片里"光照 + 阴影 + 材质"已经混在一起烘焙好了。

项目主页: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## 3. GS³ 的核心思路：把"颜色"拆成三个物理量

原始 3DGS 渲染时每个 pixel 就是 alpha-blend 一堆带颜色的 Gaussian。GS³ 改成：每个 Gaussian 不存"颜色"，而是存一个 **appearance function**（就是个 BRDF），渲染时分三路算：

$$\text{最终像素} = \text{Shading} \times \text{Shadow} + \text{Residual}$$

这三路对应物理上的三件事：
- **Shading**: 这个点在直接光照下应该多亮（材质 + 光照方向夹角）
- **Shadow**: 这个点有没有被别的 Gaussian 挡住光
- **Residual**: 剩下的乱七八糟的（interreflection、subsurface scattering 等）

这就是 paper 标题里 "Triple Splatting" 的含义——三次 splatting 各算各的，最后合在一起。

---

## 4. 三次 splatting 分别在干啥

### 4.1 第一次 splatting：算 Shading（材质该多亮）

**问题**：怎么让每个 Gaussian 都能表达复杂材质？尤其 anisotropic 的（拉丝金属、织物、头发这种高光会拉长成一条线的）。

**做法**：每个 Gaussian 存一个 BRDF 函数：
$$f(\omega_i', \omega_o') = \rho_d f_d(\omega_i') + \rho_s f_s(\omega_i', \omega_o')$$

- $\rho_d, \rho_s$: diffuse / specular albedo（RGB 三个数）
- $f_d$: diffuse term, 改良版 Lambertian
- $f_s$: specular term, 用 **mixture of angular Gaussians (ASG)** 表达

为什么不用 SH？SH 是 isotropic 的，要表达 anisotropic lobe 得用很高阶，参数又多又不好优化。ASG 直接就是参数化的 anisotropic 钟形函数，天生适合。

**ASG 长啥样**：
$$G_{\text{ang}}(\mathbf{h}') = \frac{1}{\sigma_z} \exp\left(-\frac{1}{2}\left(\frac{\arccos(\mathbf{h}'\cdot\mathbf{z}) \sqrt{(\mathbf{s}'\cdot\mathbf{x}/\sigma_x)^2 + (\mathbf{s}'\cdot\mathbf{y}/\sigma_y)^2}}{\sigma_z}\right)^2\right)$$

变量含义：
- $\mathbf{h}'$: half vector（光线和视线的中间方向），决定高光出现位置
- $[\mathbf{x}, \mathbf{y}, \mathbf{z}]$: 这个 ASG 的局部坐标（z 是 lobe 中心轴方向）
- $\sigma_x, \sigma_y$: 切平面两个方向的 lobe 宽度（不等就 anisotropic）
- $\sigma_z$: 沿中心轴的衰减速度（决定 sharpness）
- $\mathbf{s}'$: $\mathbf{h}'$ 在 x-y 平面投影后归一化的结果
- $\arccos(\mathbf{h}'\cdot\mathbf{z})$: half vector 偏离 lobe 中心的球面角距离

**直觉**：想象球面上一个椭圆斑块，中心轴 $\mathbf{z}$ 指向哪高光就朝哪，$\sigma_x, \sigma_y$ 决定这块斑横竖各拉多宽，$\sigma_z$ 决定边缘是软还是硬。

**关键工程细节**: 8 个 basis ASG 在所有 spatial Gaussian 间**共享**，每个 Gaussian 只学 8 个 weight $\alpha_j$ 做线性组合。这跟 SVBRDF 的 basis decomposition [Lensch et al. 2003; Chen et al. 2014] 一脉相承——材质空间上通常是 coherent 的，basis sharing 让优化更好 conditioning。

**Diffuse term 的细节**：
$$f_d(\omega_i') = \frac{\text{ELU}(\mathbf{n}' \cdot \omega_i') + \varepsilon(1 - 1/e)}{(1 + \varepsilon(1 - 1/e))\pi}$$

为什么不直接用 $\max(0, \mathbf{n}\cdot\omega_i)/\pi$？因为下半球梯度为 0，优化一旦陷进去就爬不出来（"dead zone"）。ELU 在负值区域梯度是 $e^x > 0$，处处可微。$\varepsilon = 0.01$ 加一点偏置保证 $f_d$ 始终为正，分母保证能量守恒近似。

每个 Gaussian 还存一个独立 **shading frame** $[\mathbf{n}, \mathbf{t}, \mathbf{b}]$（用 quaternion 编码），**与 Gaussian 的几何朝向 $\Sigma$ 完全解耦**。这点很重要：头发拉成细长的 Gaussian，但高光方向可能垂直于发丝；几何朝向和材质朝向不能绑死。

参考:
- ASG 原始 paper: https://doi.org/10.1145/2508363.2508386
- Relightable Gaussian Codec Avatars: https://arxiv.org/abs/2312.03704

---

### 4.2 第二次 splatting：算 Shadow（可见性）

**问题**：要知道一个 Gaussian 有没有被其他 Gaussian 挡住光，标准做法是 ray tracing，但慢得离谱。

**GS³ 的 trick**：抄 shadow mapping [Williams 1978] 的思路——把**点光源当作虚拟相机**，对着 scene 渲染一遍。原本被挡住的点，opacity 累积就高，这就是它的 shadow value。

具体流程（对应 Fig. 3）：

1. 在点光源位置放一个 perspective camera，分辨率同输入图
2. 对每条 shadow ray（虚拟相机的一个 pixel），找到与之相交的所有 spatial Gaussians 的 2D splats
3. 按 distance to light 排序
4. 对每个 Gaussian 计算 cumulative opacity $T_m$（沿用 Eq. 3）：
$$T_m = \prod_{k=0}^{m-1}(1 - \beta_k \gamma_k)$$
   - $\beta_k$: 第 k 个 Gaussian 在该 ray 上投影的密度
   - $\gamma_k$: 第 k 个 Gaussian 的 opacity
5. 一个 Gaussian 的 2D splat 通常覆盖多条 ray，按 $\beta_m$ 加权平均得到这个 Gaussian 的 shadow value：
$$T = \frac{\sum_m \beta_m T_m}{\sum_m \beta_m}$$
6. **MLP refinement**: $T' = \Phi(T, \omega_i; \boldsymbol{\mu}, \mathbf{l})$
   - $T$: splatting 得到的 raw shadow
   - $\omega_i$: 光照方向
   - $\boldsymbol{\mu}$: Gaussian 中心位置（spatial awareness）
   - $\mathbf{l}$: per-Gaussian 6D latent vector
   - $\Phi$: 3-layer MLP, hidden {32, 32, 32}, leaky ReLU + sigmoid
   - 4-band positional encoding 加到 $\boldsymbol{\mu}$ 和 $\omega_i$ 上
7. 把每个 Gaussian 染上 refined $T'$, splat 到真实相机视角 → shadow image

**为什么这个 trick 聪明**：3DGS 的整个 differentiable rasterization pipeline（projection、tile culling、排序、alpha blend）是高度 GPU 优化的。把 light 当 camera 之后，shadow 计算可以直接复用整套 pipeline，从 $O(N)$ 的 ray tracing 降到 rasterization 复杂度。这就是从 1fps 拉到 90fps 的关键。

**为什么还要 MLP refine**：splatting 出来的 shadow 有 aliasing 和 discretization 噪声。MLP 提供空间平滑 + 细节补充，类似 ray tracing 里的 neural denoiser。Ablation 显示去掉 $\Phi$ SSIM 从 0.9715 掉到 0.9514（损失最大的一项）。

**Shadow bias = 0.015**: 类似传统 shadow mapping 解决 z-fighting 的小偏移。

---

### 4.3 第三次 splatting：算 Residual（其他效果）

$$\text{Residual} = \Psi(\omega_o; \boldsymbol{\mu}, \mathbf{l})$$

- $\Psi$: 3-layer MLP, hidden {128, 128, 128}, leaky ReLU + sigmoid
- $\omega_o$: view direction（注意：只依赖 view，不依赖 light）
- $\boldsymbol{\mu}$: spatial awareness
- $\mathbf{l}$: shared latent vector（与 $\Phi$ 共享）
- 4-band positional encoding

**为什么 Residual 只看 $\omega_o$ 不看 $\omega_i$**：这个 MLP 主要是补 interreflection、subsurface scattering 这类**间接光照**。在 real-time rendering 工程里，indirect illumination 通常被近似为 view-dependent 但与 direct light 解耦的低频成分 [Akenine-Möller et al. 2018]。这是简化但实用。

**作用**：处理掉没被 shading 和 shadow 模型覆盖的 light transport。Ablation 显示去掉 $\Psi$ 后结果变 noisy，因为其他组件会 over-fit 试图补偿。

---

## 5. 训练时为啥不用任何 regularization

大部分 GS-based inverse rendering 方法（GaussianShader, GS-IR, Relightable 3DGS）都要加 normal smoothness、depth consistency 等正则化。GS³ 啥都不加，只用 end-to-end image loss：
$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}, \quad \lambda = 0.2$$

为什么能 work？因为三路 splatting 在物理上 **天然 decoupled**：
- Shading 管直接光照
- Shadow 管可见性
- Residual 管间接光照

物理上不重叠，image loss 自然让它们各管一摊，无需人为加约束。这是 paper 的优雅之处。

而传统方法用 isotropic BRDF，normal 是唯一能 regularize 的量，所以不得不加约束。GS³ 的 ASG mixture 自带 anisotropy 表达能力，shading frame 可以独立学习，根本不需要 normal 正则化。

---

## 6. 两阶段训练的 intuition

**Stage 1 (0-15K iter)**: 只用 Lambertian $f_d$，specular term 关掉  
**Stage 2 (15K-115K iter)**: 启用 full appearance function

为什么？specular term 的 gradient 非常尖锐（高光区域小、值变化大），如果一开始 shading frame 还没稳定，specular gradient 会把 frame 拉得到处乱跑，陷入 local minimum。Lambertian term 梯度平缓，先把 shading frame 稳定下来，再加细节。类似 NeRF 的 coarse-to-fine。

---

## 7. 实验数据要点

### 7.1 性能
- 训练: 40-70 min（120K-750K Gaussians + 8 basis ASG）
- 渲染: >90 fps on RTX 4090
- 对比 NRHints [Zeng et al. 2023]: 训练 15 hrs, 渲染 <1fps — **GS³ 快 1-2 个数量级**

### 7.2 关键 Ablation（Table 1）

| Ablation | SSIM | PSNR |
|---|---|---|
| Full | 0.9715 | 31.39 |
| w/o shadow splatting (纯 MLP 算 shadow) | 0.9661 | 29.93 |
| w/o $\Phi$ (shadow refine MLP) | 0.9514 | 28.03 |
| w/o $\Psi$ (residual MLP) | 0.9707 | 31.30 |
| 1 basis ASG | 0.9655 | 29.70 |
| 8 basis ASG | 0.9715 | 31.39 |
| 16 basis ASG | 0.9721 | 31.50 |

**怎么读这张表**：
- $\Phi$ 去掉掉得最狠（PSNR -3.36）——纯 splatting 的 shadow 太糙，必须 refine
- 纯 MLP shadow 也不行（PSNR -1.46）——会 over-fit，必须物理 grounded 的 splatting 当底子
- Basis 数量边际收益递减，8 是 sweet spot，16 只多 0.11 PSNR 但参数翻倍
- $\Psi$ 去掉 PSNR 只掉 0.09，但视觉上会 noisy，因为其他组件会试图补偿

### 7.3 跟 SOTA 对比

vs environment-lit 输入方法（GaussianShader, GS-IR, Relightable 3DGS, TensoIR）—— GS³ 全方位碾压，MaterialBalls PSNR 28.01 vs 第二名 25.34。

原因：OLAT 输入信息量比 environment-lit 大得多（每个 light direction 独立采样），加上 ASG 能表达 anisotropic，传统方法用 isotropic BRDF 根本接不住。

vs NRHints（点光输入的 SOTA）—— GS³ 在 complex appearance 和 fur 上更好，在简单 scene 上略差。NRHints 用 neural implicit representation 表达能力略强，但慢 1000x。

---

## 8. 这篇 paper 真正的聪明之处

总结成几个 takeaway：

1. **Shadow mapping 思路 + Gaussian rasterization = 高效可微 shadow 计算**。这是工程上最大的 win，把 ray tracing 的复杂度直接拉到 rasterization 水平。

2. **物理拆解 → 无需 regularization**。Shading / Shadow / Residual 三路对应物理上独立的光传输过程，image loss 自然让它们 decouple。这比加一堆人工正则化干净多了。

3. **Analytic basis + Neural refinement 是个好范式**。ASG（解析函数）+ MLP $\Phi$（neural refinement）、Splatting + MLP、Shading + MLP $\Psi$——全是"物理 grounded 初值 + neural 补细节"的组合。比纯 MLP sample-efficient，比纯 analytic 更有表达力。

4. **Shading frame 与 geometry frame 解耦**。Geometry 表达"形状长啥样"（$\Sigma$），shading frame 表达"材质朝哪"（quaternion），两者独立优化。这对头发、织物等 anisotropic 材质是必须的。

5. **Shared basis ASG**。8 个 basis 在所有 Gaussian 间共享，每个 Gaussian 只学 8 个 weight。这是经典 SVBRDF basis decomposition 的思想迁移到 Gaussian 表征上。

---

## 9. 局限性（paper 自己承认的）

1. **透明材质**（玻璃、宝石）不支持，没显式建模 refraction / internal reflection
2. **极高频 anisotropic 高光** 会出现 blinking——spatial Gaussian 粒度不够细，需要更好的 density control
3. **Shadow crispness** 不如 mesh-based 表征——Gaussian 的 footprint 性质决定的，硬阴影做不锐利

---

## 10. 一些相关资源

- **GS³ 项目主页**: https://GSrelight.github.io/
- **原始 3DGS**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **ASG**: https://doi.org/10.1145/2508363.2508386
- **Relightable Gaussian Codec Avatars**: https://arxiv.org/abs/2312.03704
- **NRHints (对比 SOTA)**: https://research.adobe.com/publication/relighting-neural-radiance-fields-with-shadow-and-highlight-hints/
- **3DGS Survey**: https://arxiv.org/abs/2403.11134
- **OpenSVBRDF 数据库**: https://doi.org/10.1145/3618358

---

## 11. 一句话总结

GS³ 把 3DGS 从"固定光照 novel view synthesis"推到"joint novel view + novel lighting synthesis"，靠的是三件事：**ASG mixture 表达 anisotropic BRDF** + **shadow splatting 复用 rasterization 算 visibility** + **三路物理 decoupled splatting 让训练无需 regularization**。结果：40 分钟训练，90fps 渲染，质量超过 SOTA neural relighting。

如果你对某个具体点还想再深入（比如 ASG 的解析积分性质、quaternion 优化的具体技巧、或者 density control 在 relighting 里的特殊行为），告诉我，我可以再展开。

---

# GS³: Efficient Relighting with Triple Gaussian Splatting 深度解析

## 1. 核心问题与动机

这篇 paper 解决的是 **relightable 3D 表征** 问题：给定 500-2000 张多视角、单点光源轮流照射（OLAT, One-Light-At-a-Time）的照片，重建出一个可在新视角和新光照下实时渲染的 3D 表征。

### 为什么现有方法不够用？

**3DGS [Kerbl et al. 2023]** 用 low-order Spherical Harmonics (SH) 表示每个 spatial Gaussian 的 view-dependent color，但 SH 编码的是固定 environment lighting 下烘焙好的 appearance，无法 relight。
- 项目主页: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

**Neural relighting** (e.g., NRHints [Zeng et al. 2023], NeRV [Srinivasan et al. 2021]) 质量高但训练 15 小时、渲染 <1fps，工程上不实用。

**Existing GS-based relighting** (GaussianShader, GS-IR, Relightable 3DGS) 大多假设 isotropic BRDF + 需要 surface normal 正则化，无法处理 anisotropic 反射（如金属、织物）。

GS³ 的关键洞察：**用 angular Gaussian mixture 替代 SH 表示 BRDF**，并设计 **triple splatting** 流程让 shadow 计算复用 GS 的 rasterization pipeline。

---

## 2. Triple Splatting 的架构

GS³ 是一个 **deferred shading** pipeline，分三路并行 splatting：

### 2.1 Geometry 表征（继承 vanilla 3DGS）

每个 spatial Gaussian 的密度：
$$G_{\text{spa}}(\mathbf{p}) = \exp\left(-\frac{1}{2}(\mathbf{p}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{p}-\boldsymbol{\mu})\right)$$

变量解析：
- $\mathbf{p}$: 3D 空间中任意一点
- $\boldsymbol{\mu}$: Gaussian 的 3D 中心
- $\Sigma = R S S^\top R^\top$: 协方差矩阵
  - $S$: scaling matrix（3 个尺度参数）
  - $R$: rotation matrix（quaternion 表示，4 个参数）
- $\Sigma$ 描述 Gaussian 椭球的形状和朝向

每个 Gaussian 还携带 opacity $\gamma_j$ 和一个**完整的 appearance function**（替代 vanilla GS 的 SH color）。

### 2.2 Triple Splatting 三路拆解

**1st splatting — Angular Gaussian splatting (Shading image)**:
- 对每个 spatial Gaussian 在 half-vector 空间混合 angular Gaussians 评估 BRDF → 得到 per-Gaussian RGB color → splat 到屏幕 → **shading image**

**2nd splatting — Shadow splatting (Shadow image)**:
- 把所有 spatial Gaussians 朝 light 方向 splat（light 作为虚拟相机） → 计算 cumulative opacity → MLP refine → splat 到屏幕 → **shadow image**

**3rd splatting — Residual splatting (Residual image)**:
- MLP 预测 global illumination 等 unmodeled effects → per-Gaussian RGB → splat → **residual image**

**最终合成**：
$$\text{Final pixel} = \text{Shading} \times \text{Shadow} + \text{Residual}$$

这对应 rendering equation 的拆解：
$$L_o(\omega_o) = \int V(\omega_i) \cdot f_r(\omega_i, \omega_o) \cdot L_i(\omega_i) \cdot (\mathbf{n}\cdot\omega_i) \, d\omega_i + L_{\text{indirect}}$$

- $f_r$ → angular Gaussian mixture
- $V(\omega_i)$ → shadow splatting + MLP Φ
- $(\mathbf{n}\cdot\omega_i)$ → 嵌入 Lambertian $f_d$
- $L_{\text{indirect}}$ → MLP Ψ

对 point light，积分退化为单点求和，所以最终就是 shading × shadow + residual。

---

## 3. Appearance Function 设计细节

### 3.1 总体形式

每个 spatial Gaussian 的 appearance function：
$$f(\omega_i', \omega_o') = \rho_d f_d(\omega_i') + \rho_s f_s(\omega_i', \omega_o')$$

变量解析：
- $\omega_i'$: **local** lighting direction（变换到该 Gaussian 的 shading frame 下）
- $\omega_o'$: **local** view direction
- $\rho_d, \rho_s$: diffuse / specular albedo（各为 RGB 3 维）
- $f_d, f_s$: diffuse / specular appearance function

### 3.2 Shading Frame

每个 spatial Gaussian 有一个独立的 shading frame $[\mathbf{n}, \mathbf{t}, \mathbf{b}]$（normal, tangent, binormal）。
- 用 **unit quaternion** 表示，4 个参数（比 9 个数的 rotation matrix 紧凑）
- **关键设计**: shading frame 完全独立于 Gaussian 的 $\Sigma$ 轴
  - $\Sigma$ 描述几何形状（一个椭球的朝向）
  - shading frame 描述材质的局部坐标
  - 例如 hair：Gaussian 沿发丝拉长，但 anisotropic 高光方向可能垂直于发丝
  - 几何朝向 ≠ 材质朝向，解耦是必要的

### 3.3 Diffuse Term — Modified Lambertian

$$f_d(\omega_i') = \frac{\text{ELU}(\mathbf{n}' \cdot \omega_i') + \varepsilon(1 - 1/e)}{(1 + \varepsilon(1 - 1/e))\pi}$$

变量解析：
- $\mathbf{n}'$: shading frame 下的 normal（local frame 下就是 $(0,0,1)$）
- $\omega_i'$: local lighting direction
- ELU: Exponential Linear Unit, $\text{ELU}(x) = x$ if $x>0$ else $e^x - 1$
- $\varepsilon = 0.01$: 小正数
- $e$: 自然对数底
- $\pi$: 球面积分常数

**为什么用 ELU 而非 $\max(0, \cdot)$？**

标准 cosine-weighted Lambertian: $f_d = \max(0, \mathbf{n}\cdot\omega_i)/\pi$，当 $\omega_i$ 在下半球时梯度为 0，形成 **"dead zone"**。一旦优化陷入 $\mathbf{n}\cdot\omega_i < 0$ 区域，梯度消失，无法爬出来。

ELU 在 $x<0$ 区域梯度为 $e^x > 0$，处处可微。加上 $\varepsilon(1-1/e)$ 偏置保证 $f_d$ 始终为正。分母的 $(1+\varepsilon(1-1/e))\pi$ 是为了保证 $f_d$ 在上半球积分接近 1（能量守恒近似）。

### 3.4 Specular Term — Mixture of Angular Gaussians

$$f_s(\omega_i', \omega_o') = \sum_j \alpha_j G_{\text{ang}, j}(\mathbf{h}')$$

变量解析：
- $\alpha_j$: 第 $j$ 个 basis angular Gaussian 的 weight（per-spatial-Gaussian learnable）
- $\mathbf{h}' = \frac{\omega_i' + \omega_o'}{\|\omega_i' + \omega_o'\|}$: half vector（local frame 下）
- $G_{\text{ang}, j}$: 第 $j$ 个 angular Gaussian

**Angular Gaussian (ASG) 公式**:
$$G_{\text{ang}}(\mathbf{h}') = \frac{1}{\sigma_z} \exp\left(-\frac{1}{2}\left(\frac{\arccos(\mathbf{h}'\cdot\mathbf{z}) \sqrt{(\frac{\mathbf{s}'\cdot\mathbf{x}}{\sigma_x})^2 + (\frac{\mathbf{s}'\cdot\mathbf{y}}{\sigma_y})^2}}{\sigma_z}\right)^2\right)$$

变量解析：
- $[\mathbf{x}, \mathbf{y}, \mathbf{z}]$: angular Gaussian 的 local frame（3 个正交向量，由 quaternion 编码）
- $\mathbf{s}'$: $\mathbf{h}'$ 在 x-y 平面投影后的 normalized 结果
- $\sigma_x, \sigma_y, \sigma_z$: 三轴 standard deviation
- $\arccos(\mathbf{h}'\cdot\mathbf{z})$: $\mathbf{h}'$ 与中心轴 $\mathbf{z}$ 的球面角距离（弧度）

**几何直觉**：
- $\arccos(\mathbf{h}'\cdot\mathbf{z})$ 是 half vector 偏离 lobe 中心的"高度"
- $\sqrt{(\mathbf{s}'\cdot\mathbf{x}/\sigma_x)^2 + (\mathbf{s}'\cdot\mathbf{y}/\sigma_y)^2}$ 是在切平面上的椭圆"宽度"
- 整体是一个 anisotropic 钟形函数，在球面上展开
- $1/\sigma_z$ 是 amplitude normalization

**为什么 ASG 比 SH 更适合 anisotropic BRDF**：
- SH 是 isotropic 基函数，表达 anisotropic lobe 需要高阶项
- ASG 直接参数化 anisotropic lobe：sharpness（$\sigma_z$）、各向异性（$\sigma_x$ vs $\sigma_y$）、方向（$[\mathbf{x},\mathbf{y},\mathbf{z}]$ frame）
- 8 个 ASG mixture 足以表达复杂 all-frequency specular appearance

**与 [Xu et al. 2013] 原始 ASG 的差异**:
原始 ASG 的 smoothness 项不可微（用了 $\arccos$ 加绝对值等操作），不利于 differentiable optimization。GS³ 借鉴 [Saito et al. 2023] 的改写方式使其可微。
- ASG 原始 paper: https://doi.org/10.1145/2508363.2508386
- Relightable Gaussian Codec Avatars: https://arxiv.org/abs/2312.03704

### 3.5 Shared Basis Angular Gaussians

**关键设计**：8 个 basis angular Gaussians 在**所有 spatial Gaussians 间共享**，每个 spatial Gaussian 只学一组 weights $\{\alpha_j\}_{j=1}^8$ 线性组合它们。

每个 spatial Gaussian 的 learnable 参数：
- $[\mathbf{n}, \mathbf{t}, \mathbf{b}]$ (shading frame, quaternion, 4D)
- $[\mathbf{x}, \mathbf{y}, \mathbf{z}]$ (angular Gaussian local frame, 但既然 basis 共享，这部分是共享的 8 份)
- $[\sigma_x, \sigma_y, \sigma_z]$ (8 个 basis 各一份，共享)
- $\rho_d, \rho_s$ (3+3 = 6D per Gaussian)
- $\{\alpha_j\}_{j=1}^8$ (8D per Gaussian)

这种 **basis sharing** 类似 SVBRDF 的 basis decomposition [Lensch et al. 2003; Chen et al. 2014]，让优化更 conditioned（material 在空间上 coherent）。

---

## 4. Shadow Splatting — 核心创新点

### 4.1 核心思想

类比传统 **shadow mapping** [Williams 1978]：把 light source 当作虚拟 camera，复用 rasterization pipeline 计算深度，从而判断 visibility。

GS³ 把所有 spatial Gaussians 朝 light 方向 splat，用 alpha-blending 累积 opacity 作为 visibility。

### 4.2 详细流程（Fig. 3）

**Step 1: Light-view splatting**
- 设置一个 perspective camera（center 在 light 位置），分辨率同 input image
- 对每条 shadow ray $m$（即 light-view 的每个 pixel）：
  - 找到与该 ray 相交的所有 spatial Gaussians 的 2D splats
  - 按 distance to light 排序
  - 对每个相交 Gaussian，计算 cumulative opacity $T_m$（Eq. 3）

**Step 2: Per-Gaussian shadow value aggregation**
- 一个 Gaussian 的 2D splat 通常覆盖多条 shadow rays（多个 pixel）
- 每条 ray 给出一个 $T_m$，按 $\beta_m$（Gaussian 在该 pixel 处的 density）加权平均：
$$T = \frac{\sum_m \beta_m T_m}{\sum_m \beta_m}$$

这是 **footprint-averaged visibility**：越接近 splat 中心的 ray 权重越大。

**Shadow bias**: 0.015，类似 shadow mapping 中为了缓解 "z-fighting" 的 bias。

**Step 3: MLP refinement**
$$T' = \Phi(T, \omega_i; \boldsymbol{\mu}, \mathbf{l})$$

变量解析：
- $T$: splatting 得到的 raw shadow value
- $\omega_i$: lighting direction
- $\boldsymbol{\mu}$: spatial Gaussian center（spatial awareness）
- $\mathbf{l}$: per-spatial-Gaussian 6D latent vector
- $\Phi$: 3-layer MLP, hidden dims {32, 32, 32}, leaky ReLU + sigmoid output
- 4-band positional encoding 应用到 $\boldsymbol{\mu}$ 和 $\omega_i$

**为什么 MLP refine 必要**：
- Splatting 产生的 shadow 是离散的、有 aliasing 的
- MLP 提供 spatial smoothness + 细节补充
- 类似神经 shadow denoiser 在 ray tracing 里的作用
- Ablation 显示：去掉 $\Phi$ 后 SSIM 从 0.9715 → 0.9514（损失最大的一项）

### 4.3 为什么 Shadow Splatting 是聪明的复用？

3DGS 的核心是 **tile-based differentiable rasterization**，GPU 高度优化。把 light 当 camera 后：
- Projection、tile culling、排序、alpha blend 全部复用
- 比起 ray tracing 的 $O(N)$ per-ray traversal，rasterization 是 $O(\log N)$ 排序
- 这让 shadow 计算从 ~1fps 提升到 90fps

---

## 5. Other Effects (Residual) — MLP Ψ

$$\text{Residual} = \Psi(\omega_o; \boldsymbol{\mu}, \mathbf{l})$$

- 3-layer MLP, hidden dims {128, 128, 128}, leaky ReLU + sigmoid
- 仅以 $\omega_o$（view direction）为输入，这是 real-time rendering 中表示 indirect illumination 的常见参数化 [Akenine-Möller et al. 2018]
- $\boldsymbol{\mu}$: spatial awareness
- $\mathbf{l}$: shared latent vector（与 $\Phi$ 共享）
- 4-band positional encoding

**为什么 $\omega_o$ 而非 $\omega_i$**：
- Global illumination (interreflection, subsurface scattering 等) 主要表现为 view-dependent 但与 direct light 解耦的低频成分
- 这是 RT 渲染中常用的简化（precomputed radiance transfer 中 ambient term 通常只依赖 view）

**作用**：
- 处理 interreflection、subsurface scattering 等未显式建模的 light transport
- Ablation: 去掉 $\Psi$ 后 SSIM 0.9707（小幅下降但结果会 noisy，其他组件会 over-fit 补偿）

---

## 6. Training 细节

### 6.1 Loss

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}, \quad \lambda = 0.2$$

- $\mathcal{L}_1$: pixel-wise L1
- $\mathcal{L}_{\text{D-SSIM}}$: SSIM 的 1 - 形式

**关键设计**：**不施加任何 intermediate regularization**。不像其他 GS-IR 方法需要 normal smoothness、depth consistency 等。

为什么能 work？因为三路 splatting 在物理上是 decoupled 的：
- Shading 对应 direct lighting
- Shadow 对应 visibility
- Residual 对应 indirect illumination
- End-to-end image loss 自然让它们各管一摊，无需人为约束

### 6.2 Two-stage Training

**Stage 1 (0-15K iter)**: 只用 Lambertian $f_d$，让 shading frame 收敛稳定
**Stage 2 (15K-115K iter)**: 启用 full appearance function with specular

**为什么两阶段**：
- 直接训练 specular 会导致 shading frame 在 specular gradient 的高方差下不收敛
- Lambertian term 梯度平稳，先稳定 frame，再加细节
- 类似 NeRF 的 coarse-to-fine 策略

### 6.3 Initialization

- Spatial Gaussians: vanilla GS 初始化（SfM points）
- 每个 angular Gaussian: $\sigma_z \sim U[0.13, 0.69]$, $\sigma_x = 0.5$, $\sigma_y = 1.0$
- $\rho_d = \rho_s = (1, 1, 1)$
- $\alpha_j = 0.5$
- 所有 local frame 初始对齐世界坐标轴

### 6.4 Learning Rate

- $\rho_d, \rho_s$: 0.01
- Angular Gaussians: 0.01 (40K 前), 指数衰减到 0.0001 (90K 时), 之后固定
- Adam optimizer, momentum 0.9

---

## 7. Rendering

对 **point light**：直接用 perspective projection shadow splatting。

对 **directional light**：shadow splatting 切换为 orthographic projection。

对 **environment light**：采样若干 directional lights，每个按 directional light 渲染，最后 linear combination。这本质是 importance sampling 的简化版。

---

## 8. 实验数据解读

### 8.1 性能

- 训练时间: 40-70 分钟（120K-750K spatial Gaussians, 8 basis angular Gaussians）
- 渲染速度: >90 fps on RTX 4090
- 对比 NRHints: 训练 15 hrs, 渲染 <1fps — **GS³ 快 1 个数量级以上**

### 8.2 Ablation Studies (Table 1)

| Ablation | SSIM | PSNR | LPIPS |
|---|---|---|---|
| **Full** | **0.9715** | **31.39** | **0.0355** |
| w/o shadow splatting | 0.9661 | 29.93 | 0.0391 |
| w/o $\Phi$ (shadow refine) | 0.9514 | 28.03 | 0.0556 |
| w/o $\Psi$ (other effects) | 0.9707 | 31.30 | 0.0366 |
| 1 basis ang. Gauss | 0.9655 | 29.70 | 0.0407 |
| 2 basis | 0.9694 | 30.93 | 0.0377 |
| 4 basis | 0.9709 | 31.25 | 0.0363 |
| **8 basis** | **0.9715** | **31.39** | **0.0355** |
| 16 basis | 0.9721 | 31.50 | 0.0350 |
| 500 images | 0.9670 | 30.61 | 0.0390 |
| 1000 images | 0.9698 | 31.09 | 0.0370 |
| 2000 images | 0.9715 | 31.39 | 0.0355 |

**关键观察**:
- 去掉 shadow refinement $\Phi$ 损失最大（PSNR -3.36），是最关键的 MLP
- 去掉 shadow splatting 整体 PSNR -1.46，证明显式 splatting 优于纯 MLP
- Basis 数量边际收益递减，8 是 sweet spot
- 输入图像数边际收益递减

### 8.3 Comparisons (Table 2, Fig. 12)

与 environment-lit 输入的方法比较：

| Method | Hotdog PSNR | Lego PSNR | MaterialBalls PSNR |
|---|---|---|---|
| GaussianShader | 29.25 | 24.33 | 24.31 |
| GS-IR | 29.13 | 27.66 | 23.41 |
| Relightable 3DGS | 30.22 | 30.31 | 25.34 |
| TensoIR | 31.68 | 30.96 | 25.05 |
| **GS³ (Ours)** | **36.47** | **32.06** | **28.01** |

**为什么 GS³ 全方位碾压**：
- OLAT 数据比 environment-lit 信息量大（每个 light direction 独立采样，无歧义）
- 显式 ASG 表达 anisotropic 反射
- Shadow splatting 比 ray-traced visibility 高效且可微

### 8.4 vs NRHints (Table 3)

| Scene | NRHints SSIM | GS³ SSIM |
|---|---|---|
| Drums | 0.9745 | 0.9714 |
| FurBall | 0.9522 | 0.9669 |
| Lego | 0.9583 | 0.9511 |
| Fish | 0.9140 | 0.9252 |
| Cluttered | 0.9280 | 0.9521 |
| Cat | 0.8560 | 0.8981 |

GS³ 在 complex appearance 和 fur 上明显更好，在简单 scene 上略差（NRHints 的 neural representation 表达能力略强但慢 1000x）。

---

## 9. Intuition Building 总结

### 9.1 三层 Splatting 的物理对应

| Splatting | 物理意义 | 关键参数 |
|---|---|---|
| 1st (Angular) | BRDF evaluation in half-vector space | 8 basis ASG, $\alpha_j$, $\sigma$ |
| 2nd (Shadow) | Visibility / shadow ray | light-view camera, MLP $\Phi$ |
| 3rd (Screen) | Final pixel compositing | alpha-blend |

### 9.2 为什么 GS³ 能做到 No Regularization

传统 GS-IR 需要 normal smoothness 等正则化是因为它们用 isotropic BRDF，normal 是唯一可约束的量。GS³ 的 ASG mixture 自带 anisotropy 表达能力，shading frame 可独立学习，加上三路物理 decoupling，end-to-end image loss 就够。

### 9.3 为什么 ASG Mixture 而非 Neural BRDF

Ablation 1.1 显示：用一个 pre-trained on SGGX BRDFs 的 MLP 替代 ASG mixture，无法建模 all-frequency appearance。原因：
- MLP 表达 all-frequency 需要 high capacity，per-Gaussian 容量有限
- ASG 是 **analytic function**，对 anisotropic lobe 有 inductive bias
- Mixture of 8 ASG 是显式的 basis decomposition，比 dense MLP 更 sample-efficient

### 9.4 为什么 Shadow Splatting 而非 Neural Shadow

Ablation 1.2 显示：纯 MLP shadow 会 over-fit，generalization 差。Splatting 给出的是物理 grounded 的 visibility 信号，MLP 只做 refinement，分工明确。

---

## 10. Limitations & Future Directions

1. **Transparent materials** (glass, gems) 未处理 — 需要 refraction / internal reflection 显式建模
2. **Extremely high-frequency anisotropic** 会 blink — spatial Gaussian 粒度不够，需要更好的 density control
3. **Shadow crispness** 不如 mesh-based — Gaussian 表征的 footprint 性质决定
4. **Future**: active next-best-view/light 采样以减少输入图像数；大规模 GS³ 数据库以支持 generative tasks

---

## 11. 关键 References

- **3DGS (基础)**: Kerbl et al. 2023, https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **ASG (anisotropic spherical Gaussian)**: Xu et al. 2013, https://doi.org/10.1145/2508363.2508386
- **Relightable Gaussian Codec Avatars** (SG 先驱): Saito et al. 2023, https://arxiv.org/abs/2312.03704
- **NRHints (对比 SOTA)**: Zeng et al. 2023, ACM SIGGRAPH 2023
- **NeRF**: Mildenhall et al. 2020, ECCV
- **SGGX microflake**: Heitz et al. 2015
- **GS³ 项目主页**: https://GSrelight.github.io/
- **GS Survey**: Wu et al. 2024, https://arxiv.org/abs/2403.11134
- **3DGS Survey**: Fei et al. 2024, https://doi.org/10.1109/TVCG.2024.3397828
- **OpenSVBRDF 数据库**: Ma et al. 2023, https://doi.org/10.1145/3618358

---

## 12. 我对这篇 paper 的整体评价

**核心贡献**：把 GS 从 "static lighting novel view synthesis" 推进到 "joint novel view + novel lighting synthesis"，同时保持 90fps 实时性。

**精妙之处**：
1. **Triple splatting 的物理对应非常清晰**，每个 splatting 对应一个物理量，让 end-to-end training 无需 regularization
2. **Shadow splatting 复用 rasterization pipeline** 是工程上的神来之笔，把 $O(N)$ 的 ray tracing 降到 rasterization 复杂度
3. **ASG mixture 的 shared basis** 平衡了表达能力和优化 conditioning
4. **Shading frame 独立于 Gaussian geometry frame** 的解耦，让 hair 等各向异性材质可以正确建模

**潜在启发**：
- 这种 "analytic basis + neural refinement" 的混合模式（ASG + MLP $\Phi$, splatting + MLP $\Phi$）是一个值得借鉴的设计范式
- 不同iable rendering 中，物理 grounded 的初值 + neural refinement 通常比纯 neural 更 sample-efficient
- 三路 splatting 的 architecture 可以推广到其他 inverse rendering 任务（e.g., 分离 ambient occlusion、subsurface scattering）

如果你对某个具体组件还想深入（例如 quaternion parameterization 的优化细节、ASG 的解析积分性质、或 adaptive density control 在 relighting 中的特殊行为），我可以继续展开。
