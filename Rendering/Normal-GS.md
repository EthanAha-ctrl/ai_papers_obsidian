---
source_pdf: Normal-GS.pdf
paper_sha256: 5a7cbca164f4e4ee26acd028e906908484ca1417397b638c9a7b49317fe210e5
processed_at: '2026-08-05T22:39:15-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话聊聊 Normal-GS 的 Intuition

Andrej，咱们把那些 academic 的包装撕开，直接看这玩意儿在白板上到底画了什么。说白了，这篇 paper 的核心就是给 3D Gaussian Splatting (3DGS) 的优化计算图打了一个补丁，把原本“装死”的 Normal 梯度给救活了。

## 1. 原罪：为什么 3DGS 的 Geometry 一直很烂？

在 3DGS 里，每个 Gaussian 就是一个带着颜色和透明度的 3D 椭球。渲染 pixel 的时候，就是把这些椭球 alpha-blending 叠起来。每个 Gaussian 的 color $c$ 是这么算的：

$$c(\omega_{view}) = \sum_{l=0}^{3}\sum_{m=-l}^{l} k_l^m Y_l^m(\omega_{view})$$

变量解释：
- $\omega_{view}$：从 surface 指向 camera 的 viewing direction
- $k_l^m$：学出来的 Spherical Harmonics (SH) coefficient
- $Y_l^m$：SH basis function

你看这个公式，里面有 normal $\mathbf{n}$ 吗？压根没有。颜色只跟你看它的角度有关，跟它表面朝哪一点关系都没有。

这在 backprop 的时候就是个灾难。Photometric loss $\mathcal{L}_P$ 要通过 chain rule 把梯度传给 normal：

$$\frac{d\mathcal{L}_P}{d\mathbf{n}} = \frac{d\mathcal{L}_P}{dc} \cdot \frac{dc}{d\mathbf{n}} = 0$$

因为前向计算里 $c$ 压根不包含 $\mathbf{n}$，所以 $\frac{dc}{d\mathbf{n}}$ 直接就是 0。网络在优化图像质量的时候，Normal 就像个旁观者，完全收不到任何“你算错了”的信号。这跟 NeRF 完全不一样，NeRF 靠 density field 的 gradient 天然能抠出 normal，而 3DGS 的离散椭球没这个待遇。

以前大家怎么补救呢？加各种 regularization，或者拿个预训练的 normal network 去监督它。但是这些方法都在跟 rendering loss 打架，你强拉 normal 平滑了，渲染出来的图就糊了；你放开限制让画质好，normal 又成了一坨乱码。这就是 paper 里说的 "seesaw"（跷跷板）效应。

## 2. 核心 Trick：把 Normal 硬塞进 Color 公式

作者回到计算机图形学最底层的 Rendering Equation：

$$L_{out}(\omega_o) = L_E + \int_{\Omega^+} L_{in}(\omega_i) (\omega_i \cdot \mathbf{n}) f_r(\omega_i, \omega_o) d\omega_i$$

这公式太重了，没法直接在 3DGS 里实时跑。作者只考虑最简单的 Lambertian（纯漫反射）表面。这时候 BRDF $f_r$ 就退化成一个常数 albedo $k_D$。于是公式变成：

$$L_D = k_D \int_{\Omega^+} L_{in}(\omega_i) (\omega_i \cdot \mathbf{n}) d\omega_i$$

接下来就是见证奇迹的时刻。把 $(\omega_i \cdot \mathbf{n})$ 看作向量的点乘，然后把 $\mathbf{n}$ 从积分号里面硬拽出来：

$$L_D = k_D \cdot \mathbf{n} \cdot \left[ \int_{\Omega^+} L_{in}(\omega_i) \omega_i d\omega_i \right]$$

中括号里的东西只跟光照有关，作者给它起名叫 **Integrated Directional Illumination Vector (IDIV)** $\mathbf{l}$。

$$\mathbf{l} = \int_{\Omega^+} L_{in}(\omega_i) \omega_i d\omega_i$$

这样，漫反射颜色就被重写成了：

$$c = k_D \cdot \mathbf{n} \cdot \mathbf{l}$$

这一步太漂亮了。现在的 color 是 albedo、normal 和 lighting 的三元乘积。这时候你再算梯度：

$$\frac{d\mathcal{L}}{d\mathbf{n}} = \frac{d\mathcal{L}}{dc} \cdot (k_D \cdot \mathbf{l})$$

梯度不再是 0 了！如果渲染出来的颜色跟 Ground Truth 不对，网络现在有办法通过调整 normal 来弥补这个误差。Photometric loss 终于能直接监督 geometry 了，跷跷板被打破了。

这个 idea 本质上就是把 25 年前 Ramamoorthi 的 [Irradiance Environment Maps](https://www1.cs.columbia.edu/~ravir/papers/envmap/) 理论搬到了 3DGS 里，只不过当年是离线渲染，现在是可微渲染。

## 3. 别让自由度放飞自我：Anchor-based MLP

如果你给每个 3D Gaussian 都塞一个 3 维的 IDIV 向量 $\mathbf{l}$，参数量直接爆炸，而且必然过拟合——相邻的 Gaussian 可能学出完全相反的光照向量，导致表面全是噪点。

作者借用了 [Scaffold-GS](https://city-super.github.io/scaffold-gs/) 的架构。不在每个 Gaussian 上存 $\mathbf{l}$，而是在空间中撒一堆 Anchor。每个 Anchor $v$ 存一个 local feature $\mathbf{f}_v$，然后用一个全局共享的小 MLP $\theta_l$ 把这个 feature 解码成周围 $K$ 个 Gaussian 的 IDIVs：

$$\theta_l(\mathbf{f}_v) = \{\mathbf{l}_v^k\}_{k=1}^K$$

这招一石三鸟：
1. **Memory 大降**：不用存几百万个 3D 向量，只存稀疏的 anchor feature。
2. **隐式平滑**：因为 MLP 的 inherent smoothness，加上 local feature 共享，物理上合理的“光照在局部区域平滑变化”假设就被白嫖了，不用再手写什么 TV loss 或 Laplacian regularizer。
3. **抗造**：比那种假设全场景只有一个 global environment map 的方法（比如 [GaussianShader](https://github.com/Asparagus15/GaussianShader)）灵活多了。室内场景多光源、角落遮挡，局部 IDIV 都能处理。

## 4. Specular 高光：借 Ref-NeRF 的壳

Lambertian 只能管哑光表面。遇到金属、塑料这种高光材质，BRDF 复杂得很，法线藏在 Fresnel 项里，抽不出来。作者干脆借用了 [Ref-NeRF](https://gxberlin.github.io/ref-nerf-website/) 的思路：算出视线关于法线的镜面反射方向：

$$\omega_r = 2(\omega_o \cdot \mathbf{n})\mathbf{n} - \omega_o$$

然后把这个 $\omega_r$ 喂给一个叫 Integrated Directional Encoding (IDE) 的编码器，再过一个小 MLP 输出 specular 颜色 $L_S$。因为 $\omega_r$ 里面显式包含了 $\mathbf{n}$，所以 specular 分支也能给 normal 传梯度。最终颜色就是：

$$c = k_D \cdot (\mathbf{n} \cdot \mathbf{l}) + L_S(\phi_{IDE}(\omega_r), \mathbf{n}, \mathbf{f}_v)$$

这一步属于实用主义的妥协，没有强求把整个 Rendering Equation 都可微化，而是分而治之。

## 5. 实验数据背后的 Intuition

看实验结果最有意思。在 [Synthetic-NeRF](https://github.com/bmild/nerf) 数据集上测 Normal 的 Mean Angular Error (MAE)：

| Method | Normal MAE ↓ |
|---|---|
| SpecGaussian | 45.98° |
| ScaffoldGS | 25.56° |
| GShader | 23.56° |
| **Normal-GS** | **20.71°** |

注意 SpecGaussian，它的 PSNR 画质很高，但 Normal MAE 居然有 45.98 度！这说明什么？说明它的网络找到了一条“捷径”——它用神经网络隐式地把 color 拟合出来了，压根没用 normal。Normal 在它那里就是个摆设，所以网络随便乱填，只要 color 对就行。

而 Normal-GS 因为在前向公式里强制绑定了 $c \propto \mathbf{n} \cdot \mathbf{l}$，网络如果想降低 photometric loss，就必须把 normal 学对。这种强结构约束比靠网络自己悟要靠谱得多。

再看 [GaussianShader](https://github.com/Asparagus15/GaussianShader) 在 [Deep Blending](https://github.com/fgkoukoarenberg/DeepBlending) 数据集上直接崩了（PSNR 19.15）。因为它依赖一个 global environment map，复杂室内光照它学不出来，env map 一烂，normal 跟着全烂。Normal-GS 用 anchor-level 的 local IDIV，完全没有这个痛点。

## 6. 局限与脑洞大开

作者也坦承，那个 self-supervised 的 depth-normal consistency loss $\mathcal{L}_N$ 在 outdoor 远景的地方会失效，因为 depth 噪声太大。这很合理，远景的深度梯度跟实际法线差得十万八千里。

发散一下脑洞：
1. **集成 Monocular Prior**：完全可以把 [DSINE](https://github.com/baeggyx/DSINE) 或者 [Lotus](https://github.com/EnVision-Research/Lotus) 这种 2D normal estimator 跑出来的 normal map 当作 pseudo-GT，替掉那个脆弱的 self-regularizer，甚至直接在 loss 里加一项跟 IDIV 的联合监督。
2. **Relighting 的潜力**：现在 IDIV 是 baked 进去的，如果能想办法把 $\mathbf{l}$ 跟 scene lighting 全局解耦，说不定能在 3DGS 里做简单的 relighting，虽然不如 [GS-IR](https://github.com/lvzihao/GS-IR) 那种全 inverse rendering 彻底，但计算量小得多。
3. **跟 2DGS 联动**：[2DGS](https://buaavrcg.github.io/2d-gaussian-splatting/) 把椭球压扁成盘片，几何本来就更好。如果 2DGS 的表面加上 IDIV 的 shading，估计 normal 能准到逆天，而且画质不会掉。

总而言之，Normal-GS 的核心贡献就是一个极度简洁的数学重写，把 3DGS 从“盲人摸象”变成了“睁眼看几何”，而且代价极小，几乎可以 plug-in 到任何现有的 3DGS 变体里。这种 elegance 在如今堆模块的 paper 里挺少见的。

---

# Normal-GS 深度技术讲解

Andrej, 这篇 paper 解决了 3DGS 里一个非常 fundamental 的"seesaw 问题"——rendering quality 和 geometry accuracy 一直没法同时做好。让我把它的核心思路拆解给你看，重点 build intuition 关于**为什么 normal 在 3DGS 里从来没真正进入过 rendering pipeline**。

---

## 1. 核心问题：Normal 与 Color 的"断联"

在 vanilla 3DGS 中，每个 Gaussian 的 color 是这么算的：

$$c(\omega_{view}) = \sum_{l=0}^{3}\sum_{m=-l}^{l} k_l^m Y_l^m(\omega_{view})$$

变量说明：
- $l, m$：Spherical Harmonics (SH) 的 degree 和 order，$l \in \{0,1,2,3\}$, $m \in [-l, l]$
- $k_l^m$：SH coefficient，每个 Gaussian 优化得到
- $Y_l^m$：SH basis function
- $\omega_{view}$：viewing direction（从 surface 指向 camera）

**关键问题**：注意上式中**只有 $\omega_{view}$，没有 normal $n$**。这意味着 photometric loss $\mathcal{L}_P$ 通过 chain rule 回传时：

$$\frac{d\mathcal{L}_P}{d\mathbf{n}} = \frac{d\mathcal{L}_P}{dc} \cdot \frac{dc}{d\mathbf{n}} = 0$$

因为 $c$ 根本不依赖 $\mathbf{n}$，所以 $\frac{dc}{d\mathbf{n}} = 0$，normal 收不到任何来自 photometric loss 的梯度。这是 3DGS geometry 差的**根本原因**——不像 NeRF 那样 density gradient 天然就是 normal，3DGS 的 normal 只能靠额外的 prior/regularization 来约束，而那些 prior 又往往伤害 rendering quality，这就是 seesaw 效应。

---

## 2. 从 Rendering Equation 出发的 Re-parameterization

作者回到 Kajiya 1986 的经典 rendering equation ([The Rendering Equation](https://dl.acm.org/doi/10.1145/15922.15902))：

$$c(\omega_{view}) = L_{out}(\omega_o) = L_E(\omega_o) + \int_{\Omega^+} L_{in}(\omega_i)(\omega_i \cdot n) f_r(\omega_i, \omega_o) d\omega_i \tag{3}$$

变量说明：
- $\omega_o = -\omega_{view}$：outward radiance 方向
- $L_E(\omega_o)$：emitted radiance（自发光，对于非光源表面通常为 0）
- $\Omega^+$：surface tangent plane 上方的 upper hemisphere
- $\omega_i$：incident light direction（半球上每个入射方向）
- $L_{in}(\omega_i)$：从 $\omega_i$ 方向来的入射 radiance
- $\mathbf{n}$：surface normal（单位向量）
- $f_r(\omega_i, \omega_o)$：BRDF，描述光从 $\omega_i$ 反射到 $\omega_o$ 的比例
- $(\omega_i \cdot \mathbf{n})$：Lambert cosine term，描述投影面积衰减

---

## 3. Diffuse 分解：IDIV 的诞生

对于 Lambertian 表面，BRDF 退化成与 viewing direction 无关的 albedo $k_D$，所以 diffuse 项：

$$L_D = k_D \int_{\Omega^+} L_{in}(\omega_i)(\omega_i \cdot \mathbf{n}) d\omega_i$$

**关键 trick**（其实源自 Ramamoorthi & Hanrahan 2001 的 [Irradiance Environment Maps](https://www1.cs.columbia.edu/~ravir/papers/envmap/)，以及 [50] Xu et al. 2017 的 shape-from-shading 工作）：把 dot product 看作向量内积，从而把 $\mathbf{n}$ 从积分里"抽出来"：

$$L_D = k_D \cdot \mathbf{n} \cdot \left[\int_{\Omega^+} L_{in}(\omega_i) \omega_i \, d\omega_i\right] \tag{5}$$

定义 **Integrated Directional Illumination Vector (IDIV)**：

$$\mathbf{l} = \int_{\Omega^+} L_{in}(\omega_i) \omega_i \, d\omega_i \tag{6}$$

变量说明：
- $\mathbf{l} \in \mathbb{R}^3$：一个 3D 向量，是把 incident light field 按方向加权积分的结果
- 几何意义：$\mathbf{l}$ 的方向大致指向"平均光源方向"，模长大致反映"总入射 irradiance"
- 物理对应：这其实就是 irradiance 的 **1st-order Spherical Harmonics** 表示——Ramamoorthi 证明 1st-order SH (3 个系数) 就能 capture ~87.5% 的 diffuse irradiance energy

最终：

$$\boxed{c = L_D = k_D \cdot \mathbf{n} \cdot \mathbf{l}}$$

这是一个**漂亮的三元乘积**：albedo × normal × illumination，每一项都有明确物理意义。

---

## 4. 梯度流的"打通"：为什么这个 trick 真的工作

现在再算梯度（Eq. 7）：

$$\frac{d\mathcal{L}}{d\mathbf{n}} = \frac{d\mathcal{L}}{dc} \cdot \frac{dc}{d\mathbf{n}} = \frac{d\mathcal{L}}{dc} \cdot (k_D \cdot \mathbf{l}) \tag{7}$$

变量说明：
- $\frac{d\mathcal{L}}{dc}$：photometric loss 对 color 的梯度（来自 L1 + D-SSIM）
- $\frac{dc}{d\mathbf{n}} = k_D \cdot \mathbf{l}$：color 对 normal 的 Jacobian，是一个 3-vector
- 直觉：**normal 朝哪个方向调，能让 $k_D \mathbf{n} \cdot \mathbf{l}$ 更接近 GT color，梯度就往那个方向走**

这和 GShader ([GaussianShader CVPR 2024](https://github.com/Asparagus15/GaussianShader)) 那种用 global environment map 的方法本质不同：

$$\frac{d\mathcal{L}}{d\mathbf{n}} = \frac{d\mathcal{L}}{dE} \cdot \frac{dE}{d\mathbf{n}} \quad (\text{GShader-style})$$

这里梯度依赖 environment map $E$ 的质量；$E$ 学差了，normal 就跟着崩。而 IDIV 是 **per-anchor local** 的，不需要假设全局 environment map 存在——室内、复杂多光源场景都适用。这一点在 Fig.4 实验里非常明显：GShader 的 env map 一旦在复杂光照下学坏，specular 完全乱掉。

---

## 5. Anchor-based IDIV 隐式正则化

直接给每个 Gaussian 一个 $\mathbf{l} \in \mathbb{R}^3$ 会引入 $3 \times N_{Gaussians}$ 自由参数，过拟合严重。作者用 [Scaffold-GS](https://city-super.github.io/scaffold-gs/) 的 anchor 结构来隐式编码：

- 每个 anchor $v$ 存一个 local feature $\mathbf{f}_v$
- 一个 global MLP $\theta_l$ 把 $\mathbf{f}_v$ 解码成 $K$ 个 IDIVs，给该 anchor 周围的 $K$ 个 Gaussians 共享：

$$\theta_l(\mathbf{f}_v) = \{\mathbf{l}_v^k\}_{k=1}^K$$

变量说明：
- $v$：anchor 位置
- $\mathbf{f}_v$：anchor 上的 local feature（学习得到）
- $\theta_l$：global MLP，所有 anchor 共享参数
- $K$：每个 anchor 关联的 Gaussian 数量
- $\mathbf{l}_v^k$：第 $k$ 个 Gaussian 的 IDIV

**为什么这样做是关键**：
1. **Memory 效率**：从 $O(N_{Gaussians} \times 3)$ 降到 $O(N_{anchors} \times \dim(\mathbf{f}_v))$，通常 ~100x 压缩
2. **隐式 smoothness 正则**：MLP 的 inherent function smoothness 天然约束相邻 Gaussians 的 IDIV 连续，比 TV/Laplacian loss 更鲁棒（Xu et al. [50] 用 mesh 上做 TV/Laplacian，3DGS 离散稀疏不适用）
3. **Local sharing 假设合理**：incident lighting 在小邻域内确实变化平缓（low-frequency）

---

## 6. Specular 分支：来自 Ref-NeRF 的 IDE

对于非 Lambertian 表面，BRDF $f_r(\omega_i, \omega_o)$ 复杂且 view-dependent（Fresnel、roughness、microfacet 等），没法像 diffuse 那样把 $\mathbf{n}$ 抽出来。作者借鉴 [Ref-NeRF (Verbin et al. CVPR 2022)](https://gxberlin.github.io/ref-nerf-website/) 的思路：用 **reflection direction** 替代 viewing direction 作为输入：

$$\omega_r = 2(\omega_o \cdot \mathbf{n})\mathbf{n} - \omega_o$$

变量说明：
- $\omega_o$：outward viewing direction
- $\mathbf{n}$：surface normal
- $\omega_r$：理想镜面反射方向（mirror reflection）
- 几何意义：specular 高光出现在 $\omega_r$ 方向对应的环境光照位置

然后用 **Integrated Directional Encoding (IDE)** 编码 $\omega_r$：

$$L_S \doteq \theta(\phi_{IDE}(\omega_r), \mathbf{n}, \mathbf{f}_v)$$

变量说明：
- $\phi_{IDE}$：Ref-NeRF 提出的 IDE encoding，类似 SH 但专门为 reflection 设计
- $\theta$：另一个 MLP，输入 IDE 编码、normal、anchor feature，输出 specular color
- 因为 $\omega_r$ 显式包含 $\mathbf{n}$，所以 specular 分支也能向 normal 传梯度

最终 Gaussian color：
$$c = L_D + L_S = k_D \cdot (\mathbf{n} \cdot \mathbf{l}) + L_S(\phi_{IDE}(\omega_r), \mathbf{n}, \mathbf{f}_v)$$

---

## 7. Normal 的 Self-Regularization

3D Gaussian 的 shortest axis 当 normal，但这个 shortest axis 物理上不一定对齐 real surface normal（Gaussian 可能因为多视角监督噪声变成歪椭球）。作者加了一个 **self-supervised depth-normal consistency loss**：

1. Render 出 depth map $\mathcal{D}$ 和 normal map $\mathcal{N}$
2. 计算 depth map 的 image-space gradient：$\nabla_{(u,v)} \mathcal{D}$
3. Cross product 得到从 depth 推导的 normal：$\mathcal{N}_D$
4. Loss：$\mathcal{L}_N = 1 - \mathcal{N}_D \cdot \bar{\mathcal{N}}$

变量说明：
- $\nabla_{(u,v)} \mathcal{D}$：在 image plane $(u,v)$ 上对 depth 的 Sobel/sparse gradient
- $\mathcal{N}_D$：从 depth gradient cross product 得到的"几何法向"
- $\bar{\mathcal{N}}$：rendered normal map（来自 Gaussian shortest axis）
- 直觉：depth 和 normal 应该几何一致——depth 平的地方 normal 应该朝向 camera

最终 loss（Eq. 8）：

$$\mathcal{L} = \mathcal{L}_P + \lambda_{vol}\mathcal{L}_{vol} + \lambda_N \mathcal{L}_N$$

- $\mathcal{L}_P$：photometric (L1 + D-SSIM)，from 3DGS
- $\mathcal{L}_{vol}$：volume regularization from Scaffold-GS
- $\lambda_{vol} = 0.001$, $\lambda_N = 0.01$
- $\mathcal{L}_N$ 从 5k iteration 后才启用（前期 depth/normal 不准）

---

## 8. 架构图解析（Fig.2）

整体 pipeline：

```
[Anchors v] ── position x_v, feature f_v, scaling s_v, offsets O_v^k
     │
     ├── MLP θ_pos → Gaussian positions μ
     ├── MLP θ_scale → Gaussian scales
     ├── MLP θ_rot → Gaussian rotations (shortest axis = normal n)
     ├── MLP θ_opacity → opacities α
     ├── MLP θ_D → albedo k_D
     ├── MLP θ_l(f_v) → IDIVs {l_v^k}      ← diffuse lighting
     └── MLP θ_S(φ_IDE(ω_r), n, f_v) → specular L_S
                       │
                       ▼
              c = k_D (n · l) + L_S
                       │
                       ▼
       3DGS Rasterizer → image, depth, normal
                       │
                       ▼
       L = L_P + λ_vol L_vol + λ_N L_N
```

关键设计：所有 MLP 共享一个 anchor feature $\mathbf{f}_v$，所以 albedo、IDIV、specular 是**联合学习**的，互相提供 regularization 信号。

---

## 9. 实验数据深度分析

### Table 1: Rendering Quality

| Method | Mip-NeRF360 PSNR | T&T PSNR | DeepBlending PSNR |
|---|---|---|---|
| 3DGS | 28.691 | 23.142 | 29.405 |
| ScaffoldGS | 29.267 | 24.088 | 30.140 |
| ScaffoldGS w/N | 29.177 | 23.976 | 30.163 |
| GShader | 26.060 | 21.262 | 19.159* (失败) |
| SpecGaussian | 29.287 | 24.502 | 30.114 |
| **Normal-GS** | **29.341** | 24.219 | **30.187** |

**关键观察**：
- Normal-GS 在 Mip-NeRF360 和 DeepBlending 上 PSNR 最高
- GShader 在 DeepBlending 上完全崩了（19.159），因为它依赖 global env map，复杂室内场景 env map 学不出来
- SpecGaussian 在 T&T 上 PSNR 最高但 normal MAE 灾难性差（见下）
- "ScaffoldGS w/N"（把 normal 喂进 color MLP）效果反而比 ScaffoldGS 还差——证明 **隐式 normal 利用是不够的，必须显式 re-parameterize**

### Table 2: Normal MAE on Synthetic-NeRF

| Method | Normal MAE ↓ |
|---|---|
| SpecGaussian | 45.98° |
| ScaffoldGS | 25.56° |
| GShader | 23.56° |
| **Normal-GS** | **20.71°** |

**关键洞察**：
- SpecGaussian 虽然 PSNR 高，但 normal MAE 是 45.98°（基本随机了！）——证明靠 MLP 隐式集成 normal，网络找到了不依赖 normal 的"捷径"来拟合 color
- Normal-GS 比 GShader 低 3°，是因为 IDIV 的 local 性比 global env map 更灵活
- 20.71° 仍然不算 perfect（< 15° 才算很好），但已是 3DGS-based 方法 SOTA

---

## 10. 与相关工作的 intuition 联系

让我把这篇工作放到更大的 landscape 里，帮你 build mental model：

### (a) 与经典 Irradiance Environment Maps 的关系
[Ramamoorthi & Hanrahan 2001](https://www1.cs.columbia.edu/~ravir/papers/envmap/) 证明：diffuse irradiance 可以用 9 个 SH 系数（degree 2）精确表示，1st-order (3 系数) capture 87.5% 能量。IDIV 实际上就是这 1st-order SH 的 vector form。Normal-GS 是把这个 25 年前的图形学 trick 在 3DGS 里"复活"——但用 anchor MLP 替代显式 SH 系数，更 flexible。

### (b) 与 Inverse Rendering 的对比
GS-IR ([CVPR 2024](https://github.com/lvzihao/GS-IR))、Relightable 3DGS ([CVPR 2024](https://github.com/guochengqian/Relightable-3D-Gaussian)) 等 inverse rendering 方法用 Disney BRDF + split-sum approximation，能 relight 但 PSNR 显著低于原 3DGS（一般低 2-3 dB）。它们追求"分解出 material/lighting/geometry"，过约束导致 fitting 能力下降。Normal-GS 不追求分解，只追求"让 normal 进入 gradient flow"，所以保住了 rendering quality。

### (c) 与 2DGS / Gaussian Surfels 的对比
[2DGS (SIGGRAPH 2024)](https://buaavrcg.github.io/2d-gaussian-splatting/) 和 [Gaussian Surfels (SIGGRAPH 2024)](https://gaussian-surfels.github.io/) 把 3D 椭球压成 2D 盘片，强几何约束，surface 重建好但 rendering quality 掉。它们走的是"破坏 Gaussian 自由度换 geometry"的路线。Normal-GS 走的是"让 Gaussian 自由度服务 geometry"的路线——保持 3D Gaussian 表达能力，让 normal 通过 rendering 自然学到。

### (d) 与 GSDF / DeferredGS 的对比
[GSDF (arXiv 2403.16964)](https://arxiv.org/abs/2403.16964) 用 3DGS + SDF 双分支，[DeferredGS (arXiv 2404.09412)](https://arxiv.org/abs/2404.09412) 从额外 implicit surface 蒸馏 normal。两者都引入额外网络，训练慢。Normal-GS 单分支，无额外 surface 网络，更轻量。

### (e) 与 NeRF 的 normal 自动性的对比
NeRF 用 $\mathbf{n} = \nabla_\mathbf{x} \sigma / \|\nabla_\mathbf{x} \sigma\|$ 自动得到 normal，因为 density 场天然是 implicit surface。3DGS 是 explicit 离散基元，没这个 luxury。Normal-GS 的 IDIV trick 实际上是给 3DGS 注入了一个"伪 density gradient"——让 color 依赖 normal，相当于把 NeRF 的 inductive bias 显式带进来。

---

## 11. Limitation 与 Future Directions

作者承认：self-regularized $\mathcal{L}_N$ 在远处 outdoor 场景效果差（depth 噪声大，cross product 不稳定）。这暗示几个 future direction：

1. **Monocular Normal Priors**：用 [DSINE](https://github.com/baeggyx/DSINE)、[OmniData](https://github.com/EPFL-VILAB/omnidata)、[Lotus](https://github.com/EnVision-Research/Lotus) 等 pretrained normal estimator 提供 normal pseudo-GT，替代或辅助 self-regularization
2. **IDIV 的更精细分解**：当前 IDIV 是 1st-order，可以扩展到 2nd-order SH (9 维) 或 Spherical Gaussians，capture 更复杂 lighting
3. **Relighting**：当前 IDIV 是 baked 的，如果能 disentangle lighting 和 material，可以 relight
4. **Path Tracing 集成**：未来 3DGS + Monte Carlo path tracing 可以做更准确的 indirect lighting

---

## 12. Intuition 总结（Karpathy-style）

如果让我用一句话总结这篇 paper 的核心 insight：

> **3DGS 的 normal 一直没被 photometric loss "看见"，因为 color 在前向公式里压根不依赖 normal。Normal-GS 通过把 color 重新参数化为 $k_D \cdot \mathbf{n} \cdot \mathbf{l}$，让 normal 第一次真正进入 backprop 的计算图，于是 photometric loss 可以直接监督 normal。**

这个 idea 其实非常简单——就是把 graphics 里 25 年前的 irradiance decomposition trick 搬到 3DGS 里，但 timing 完美，因为 3DGS 正好需要解决 geometry 差的问题。这个工作的"beauty"在于：**没有引入复杂 BRDF、没有引入额外 surface network、没有引入强 prior**，仅仅是一个 re-parameterization，就打破了 seesaw。

值得思考的 open question：能否把这个 idea 推广到 **glossy BRDF**（介于 Lambertian 和 mirror 之间）？IDIV 是 diffuse 的"线性化"，specular 用 IDE 是"非线性编码"，中间地带（roughness 0.1~0.5 的 glossy 表面）是不是可以用 microfacet BRDF + IDIV 的某种混合？这是把 3DGS 推向真正 inverse rendering 的关键一步。

---

## References / Web Links

**主 paper 与 code**：
- Normal-GS (paper attachment above)
- [Scaffold-GS](https://city-super.github.io/scaffold-gs/) — anchor-based 3DGS
- [Ref-NeRF](https://gxberlin.github.io/ref-nerf-website/) — IDE 来源

**对比方法**：
- [3DGS (Kerbl et al. 2023)](https://repo.samuelgeorgi.at/) / [项目页](https://gao-xiao-bai.github.io/articles/2023-08/3d-gaussian-splatting)
- [SpecGaussian](https://arxiv.org/abs/2402.15870)
- [GaussianShader](https://github.com/Asparagus15/GaussianShader)
- [2DGS](https://buaavrcg.github.io/2d-gaussian-splatting/)
- [Gaussian Surfels](https://gaussian-surfels.github.io/)
- [SuGaR](https://anttwo.github.io/sugar/)
- [GS-IR](https://github.com/lvzihao/GS-IR)
- [GSDF](https://arxiv.org/abs/2403.16964)
- [DeferredGS](https://arxiv.org/abs/2404.09412)

**经典图形学背景**：
- [Kajiya 1986 — The Rendering Equation](https://dl.acm.org/doi/10.1145/15922.15902)
- [Ramamoorthi & Hanrahan 2001 — Irradiance Environment Maps](https://www1.cs.columbia.edu/~ravir/papers/envmap/)
- [Xu et al. 2017 — Shading-based Surface Detail Recovery](https://ieeexplore.ieee.org/document/7918371) (paper [50])

**Datasets**：
- [NeRF Synthetic](https://github.com/bmild/nerf)
- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/)
- [Tanks & Temples](https://tanksandtemples.org/)
- [Deep Blending](https://github.com/fgkoukoarenberg/DeepBlending)

**Normal Priors (future work 方向)**：
- [DSINE](https://github.com/baeggyx/DSINE)
- [OmniData](https://github.com/EPFL-VILAB/omnidata)
- [Lotus Normal Estimator](https://github.com/EnVision-Research/Lotus)

如果你感兴趣把这个 idea 推广到 glossy BRDF 或者集成 monocular normal prior，我我们可以进一步聊聊 microfacet BRDF 在 3DGS 里的可微实现细节。
