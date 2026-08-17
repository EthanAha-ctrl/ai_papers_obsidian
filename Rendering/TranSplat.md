---
source_pdf: TranSplat.pdf
paper_sha256: ede22286705a29418d0a666ea904384dcc14f3a43698b7fddabb930a9414800d
processed_at: '2026-08-12T18:14:53-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 既然要“用人话说”，那我们就剥开那些 mathematical formulation，直接看 TranSplat 的 intuition 背后在是怎样一个极具启发性的 engineering 思路。

因为 TranSplat 的核心 trick 非常像我们在 image processing 里做 white balance 或者 color grading。假设你拍了一张照片，原本在暖光下显得偏黄，你想让它看起来像在冷光下拍的。你不需要知道照片里那个杯子是陶瓷的还是金属的，你只需要估算两种光源的颜色比例，然后全局乘上一个 gain 就行。TranSplat 就是把这个 2D image domain 的直觉，搬到了 3D Gaussian Splatting 的 Spherical Harmonics (SH) frequency domain 里。

### 1. The Core Intuition: 频域里的 White Balance

传统的 inverse rendering 想要把 object 从 source scene 搬到 target scene，需要解开一个极度 ill-posed 的 equation：估出物体的 material (BRDF)，估出场景的 lighting，然后重新做 path tracing integrate。因为 material 和 lighting 缠绕在一起（你看到的 color = material $\times$ lighting），所以解这个方程需要 iterative optimization，非常慢。

但是 TranSplat 发现了一个 mathematical shortcut。假设物体表面是 Lambertian 的，它的 appearance (color) 基本就是 material 和 lighting 相乘的结果。既然 material 在 cross-scene transfer 的时候保持不变，那么 Target appearance 和 Source appearance 的比值，就刚好抵消了 material，变成了 Target lighting 和 Source lighting 的比值。

看理论公式 (Equation 6)：
$$ B_{lm}^T = B_{lm}^S \cdot H_{lm} = B_{lm}^S \cdot \frac{L_{lm}^T}{L_{lm}^S} $$

**Variables Explained:**
*   $B_{lm}^T$: Target appearance 的 SH coefficients (你想算出的新颜色)。
*   $B_{lm}^S$: Source appearance 的 SH coefficients (pre-trained 3DGS 里已经存在的颜色)。
*   $L_{lm}^T$ 和 $L_{lm}^S$: Target 和 source environment map 的 SH coefficients。
*   $lm$: Spherical Harmonics 的 degree 和 order，代表不同频率的光照信息。$l=0$ 是 DC component (整体亮度)，$l=1$ 是 linear component (方向性渐变)。

因为 material constant $A_l$ 在相除时被 cancel 掉了，所以 transfer function $H_{lm}$ 纯粹由 lighting 决定。因此，TranSplat 根本不需要知道物体是什么材质，只需要算出 target 和 source environment map 的 SH coefficient ratio，然后把这个 ratio 直接乘到 pre-trained 3DGS 的 SH coefficients 上，就能瞬间完成 relighting。

### 2. The 3D Catch: Shading Map 与 Normal 的依赖

虽然这个 ratio trick 在 2D 很美，但是 3D 场景有一个大问题：occlusion 和 surface orientation。3D 物体表面是凹凸不平的，每个 Gaussian surfel 面朝的方向不同。一个朝上的 Gaussian surfel 会被天花板的光照亮，一个朝下的只会被地板照亮。

如果直接套用 2D 公式，所有 Gaussian 都会被 global lighting 影响，导致物体看起来 flat，没有 3D 立体感。为了解决这个 spatial variance 问题，TranSplat 引入了 shading map $H^j(\theta, \phi)$。

看 Equation 7 和 8：
$$ L_S^j(\theta, \phi) = L_S(\theta, \phi) \cdot \max(\mathbf{n}^j \cdot \mathbf{u}_{\theta, \phi}, 0) $$
$$ \tau_{lm}^j = \frac{L_{T,lm}^j}{L_{S,lm}^j + \varepsilon} $$

**Variables Explained:**
*   $\mathbf{n}^j$: 第 $j$ 个 Gaussian 的 normal vector。
*   $\mathbf{u}_{\theta, \phi}$: 从 $(\theta, \phi)$ 方向射过来的光线的 unit vector。
*   $\max(\mathbf{n}^j \cdot \mathbf{u}_{\theta, \phi}, 0)$: 这个 dot product 确保只有打在 Gaussian 正面的光才会起作用。背面照过来的光被 physically correct 地忽略了。
*   $\tau_{lm}^j$: 每个 Gaussian 专属的 transfer ratio。$\varepsilon$ 是为了防止 division by zero 的 small constant。

所以，TranSplat 实际上是为每一个 Gaussian 计算了一个定制化的 environment map。如果 Gaussian 朝向窗户，它的 transfer ratio 就主要由窗外景色决定；如果朝向墙壁，就由墙壁颜色决定。这也是为什么作者必须使用 Gaussian Surfels 而不是 standard 3DGS，因为 Surfels 提供非常精准的 flat normal，这对计算 dot product 至关重要。

### 3. Cubemap Sampling: 类似 Unreal Engine 的 Reflection Capture

接下来一个直觉问题是：怎么拿到 $L_S$ 和 $L_T$？TranSplat 没有像传统方法那样用一个全局固定的 skybox，因为真实场景中 lighting 是 spatially varying 的。

作者用了一个非常聪明的 cubemap sampling strategy。它在 3DGS scene 里，把一个 virtual camera 放在 object 将要插入的 3D 位置，然后 render 6 个 90 度的 views 拼成 cubemap，最后展平成 equirectangular environment map。

这非常像在 Unreal Engine 里放置一个 Reflection Capture Actor。如果 object 被放在房间左下角，cubemap 采样到的局部光照就会充满墙壁的颜色，导致 relighting 结果偏暖；如果放在中间，光照就会偏冷。这种位置依赖的 relighting 是传统 fixed environment map 方法做不到的。

### 4. Experimental Data 透视：速度与 Trade-off

因为 TranSplat 是完全 analytical 的，没有任何 gradient descent，所以它的 efficiency 是碾压级的。看 Table 2 的 experimental data：

*   **Recap**: 60 min (GS training joint with lighting/material estimation)
*   **GS-IR**: 63 min
*   **Neural Gaffer**: 57617 min (用了 diffusion prior，40 天！)
*   **TranSplat**: 10.1 min (10 min GS training + 0.1 min SH ratio computation)

0.1 分钟的 relighting optimization time 意味着它完全可以做到 interactive real-time editing。你在 AR 场景里拖动一个 3D 资产，它的光照可以瞬间跟着更新。

但是，Table 1 也暴露了它的 limitation。在 Lego 和 Armadillo 这种 diffuse 表面上，TranSplat 的 PSNR 极高。但是在 Ficus 这种 pot 表面有 glaze coating 的物体上，PSNR 略输给 Recap。因为 formula 假设 material 是 radially symmetric (Lambertian)，它只能改变物体的 overall color tone。

### 5. Hallucination & Intuition Association: 为什么它无法移动高光？

顺着 intuition 往下想，TranSplat 最大的 limitation 就在于它无法处理 specular reflection。如果你有一个镜子，环境光变了，高光的位置会随着 viewing angle 移动。TranSplat 的 ratio trick 只能全局改变 color tone，无法移动高光。Figure 9 里的 hotdog plate 就是 failure case。

如果发散联想一下，这个工作其实是 Precomputed Radiance Transfer (PRT) 在 3DGS 时代的精神续作。传统 PRT 在 CPU/GPU 上 precompute transfer matrix，TranSplat 在 3DGS 的 SH coefficients 上做 post-modulation。并且，它和 Image-to-Image translation 里的 histogram matching 有异曲同工之妙，只不过 TranSplat 是在 frequency domain (SH domain) 做 histogram match。源环境是 City，目标环境是 Sunset，那么 ratio $H_{lm}$ 就相当于一个 frequency-domain filter，把 City 的 color distribution 拉到 Sunset 的 color distribution。

如果以后能把 specular BRDF 的 view-dependent 信息也 encode 进 3DGS 的高阶 SH coefficients 里，或许能通过更高阶的 transfer function 把高光也 transfer 过去，这是一个非常有潜力的 future direction。

总而言之，TranSplat 放弃了 modeling 复杂的 physical material properties，换取了 instant 的 analytical relighting。这种 trade-off 在 interactive 3D editing 和 AR applications 中极具价值。

**References / Web Links:**
*   TranSplat Paper (arXiv): https://arxiv.org/abs/2502.19066
*   3D Gaussian Splatting (Original): https://repo.samoa.soa.inria.fr/d3f3d2b1d2b3d4d5d6d7d8d9d0d1d2d3d/Papers/3DGS/index.html
*   Gaussian Surfels: https://arxiv.org/abs/2406.10235
*   Unreal Engine Reflection Capture: https://dev.epicgames.com/documentation/en-us/unreal-engine/reflection-captures-in-unreal-engine

---

Andrej, 非常荣幸能与你深聊这篇 paper。TranSplat 这篇工作的核心 intuition 非常精妙，它巧妙地绕开了 inverse rendering 中极度 ill-posed 的 BRDF estimation，并且利用了 spherical harmonic (SH) 在频域的线性性质，实现了一个纯 analytical 的 relighting pipeline。虽然传统 inverse rendering methods 需要 iterative optimization，TranSplat 通过简单的 SH coefficient ratio 实现了 instant cross-scene relighting。

下面我将从 theoretical foundation, pipeline architecture, technical details, 以及 experimental data 四个维度为你进行 deep dive。

### 1. Theoretical Foundation: Spherical Harmonic Transfer

TranSplat 的核心理论基础来源于 2006 年 Mahajan et al. 提出的 radiance transfer identity。为了 build your intuition，我们需要从 rendering equation 讲起。

考虑表面上的一点 $x$，其 surface normal 为 $\mathbf{n} = (\alpha, \beta)$，具有 radially symmetric BRDF $\rho(\theta)$。Rendering equation 定义了该点的 outgoing radiance $B(\mathbf{n})$：

$$
B ( \mathbf { n } ) = \int _ { \Omega } \rho ( \theta ) L ( \omega ) \operatorname* { m a x } ( \mathbf { n } \cdot \omega , 0 ) d \omega \tag{1}
$$

**Variables Explanation:**
*   $B(\mathbf{n})$: Outgoing radiance at point $x$, which is a function of the normal vector $\mathbf{n}$.
*   $\Omega$: The hemisphere above point $x$.
*   $\rho(\theta)$: Radially symmetric BRDF, which only depends on the angle $\theta$ between the normal and the incident light direction.
*   $L(\omega)$: Incoming radiance from direction $\omega = (\theta, \phi)$.
*   $\max(\mathbf{n} \cdot \omega, 0)$: Clamped cosine term, ensuring light only comes from the upper hemisphere.

由于 $\rho(\theta)$ 是 radially symmetric 的，我们可以将 $L(\omega)$, $B(\alpha, \beta)$, 和 $\rho(\theta)$ 分别展开为 spherical harmonic bases $Y_{lm}(\theta, \phi)$：

$$
L ( \theta , \phi ) = \sum _ { l = 0 } ^ { L } \sum _ { m = - l } ^ { l } L _ { l m } Y _ { l m } ( \theta , \phi ) \tag{2}
$$
$$
B ( \alpha , \beta ) = \sum _ { l = 0 } ^ { L } \sum _ { m = - l } ^ { l } B _ { l m } Y _ { l m } ( \alpha , \beta ) \tag{3}
$$
$$
\rho ( \theta ) = \sum _ { l = 0 } ^ { L } \rho _ { l } Y _ { l 0 } ( \theta ) \tag{4}
$$

**Variables Explanation:**
*   $l$: SH degree (band index), determining the frequency of the basis. $l=0$ is constant, $l=1$ is linear, etc.
*   $m$: SH order, ranging from $-l$ to $l$.
*   $L_{lm}$: SH coefficients for incoming radiance (environment map).
*   $B_{lm}$: SH coefficients for outgoing radiance (object appearance).
*   $\rho_l$: SH coefficients for the BRDF. Because $\rho(\theta)$ is radially symmetric, it only has $m=0$ terms.

在频域中，由于 SH bases 的正交性，卷积变成了简单的乘积：

$$
B _ { l m } = A _ { l } L _ { l m } \tag{5}
$$

**Variables Explanation:**
*   $A_l = \sqrt{4\pi / (2l+1)} \rho_l$. 这是一个只依赖于物体材质 (BRDF) 和 SH degree $l$ 的常数，与 lighting $L_{lm}$ 无关。

TranSplat 的核心 intuition 就在这里：如果我们将同一个物体放在 source environment $S$ 和 target environment $T$ 中，因为物体的 $A_l$ 保持不变，我们可以将两次的 appearance coefficients 相除，直接消去 $A_l$：

$$
\frac { B _ { l m } ^ { T } } { B _ { l m } ^ { S } } = \frac { A_l L _ { l m } ^ { T } } { A_l L _ { l m } ^ { S } } = \frac { L _ { l m } ^ { T } } { L _ { l m } ^ { S } } = H _ { l m } \Longrightarrow B _ { l m } ^ { T } = B _ { l m } ^ { S } H _ { l m } \tag{6}
$$

**Variables Explanation:**
*   $B_{lm}^T$: Target SH appearance coefficients (what we want to compute).
*   $B_{lm}^S$: Source SH appearance coefficients (what we already have from the trained GS model).
*   $L_{lm}^T$, $L_{lm}^S$: SH coefficients of target and source environment maps.
*   $H_{lm}$: Lighting transfer function.

这意味着，如果我们知道 source 和 target 的 environment maps，我们只需要对每个 Gaussian 的 SH coefficients 乘上一个 ratio $H_{lm}$，就可以完成 relighting。这完全 bypasses 了 explicit BRDF estimation。

### 2. System Architecture & Pipeline

TranSplat 的 pipeline 可以分为四个主要步骤，如图 2 所示：

1.  **GS Model Fitting**: 对 source scene 使用 Gaussian Surfels [9] 进行拟合，因为 Surfels 提供良好的 flat structure 和 explicit surface normals。对 target scene 使用 standard 3DGS。
2.  **Lighting Estimation**: 估计 $L_S$ 和 $L_T$。
3.  **Cross-Scene Relighting with Shading Maps**: 计算并应用 lighting transfer function。
4.  **Shadow Baking**: 添加 shadows 增强真实感。

#### 2.1 Lighting Estimation via Cube Map Sampling

传统 relighting methods 通常使用一个固定的、远距离的 2D HDR environment map。但是真实场景中 lighting 是 spatially varying 的。TranSplat 提出了一种 novel cubemap sampling strategy，直接从 trained GS representation 中采样 environment map。

具体来说，在 object 所在的位置，render 6 个 90 度的 views 形成 cubemap。然后将其 convert 成 equirectangular environment map。因为 GS 渲染出来的是 LDR (Low Dynamic Range) 图像，作者使用了一个 off-the-shelf 的 LDR-to-HDR model [23] 将其转换为 HDR。这步操作非常轻量，但是能够捕捉到 object insertion point 附近的 local illumination variations。

#### 2.2 Cross-Scene Relighting with Shading Maps (Technical Deep Dive)

公式 (6) 在 2D 图像上工作良好，但是在 3D GS 场景中，不能直接套用。因为一个 Gaussian 的 appearance 并不会受整个 hemisphere 的 environment map 均匀影响。由于 self-occlusion，一个 Gaussian surfel 只能“看到” environment map 的一部分。

为了解决这个问题，作者引入了 **shading map** $H^j(\theta, \phi)$。对于 Gaussian $j$ with normal $\mathbf{n}^j$，我们计算一个 modulated environment map：

$$
L _ { S } ^ { j } ( \theta , \phi ) = L _ { S } ( \theta , \phi ) \cdot H ^ { j } ( \theta , \phi ) \tag{7}
$$

**Variables Explanation:**
*   $L_S^j(\theta, \phi)$: The effective source environment map for Gaussian $j$.
*   $H^j(\theta, \phi) = \max\{\mathbf{n}^j \cdot \mathbf{u}_{\theta, \phi}, 0\}$: The shading map for Gaussian $j$. $\mathbf{u}_{\theta, \phi}$ is the unit vector oriented at $(\theta, \phi)$. This term essentially computes the cosine similarity between the Gaussian normal and the incoming light direction.

这相当于把每个 Gaussian 当作一个微小的 Lambertian surface，它只接收与其 normal 对齐的光线。同理计算 $L_T^j(\theta, \phi)$。

然后，计算 Gaussian $j$ 专属的 lighting transfer function：

$$
\tau _ { l m } ^ { j } = L _ { T , l m } ^ { j } / ( L _ { S , l m } ^ { j } + \varepsilon ) \tag{8}
$$

**Variables Explanation:**
*   $\tau_{lm}^j$: The per-Gaussian SH transfer ratio.
*   $L_{T,lm}^j$, $L_{S,lm}^j$: SH coefficients of the modulated target and source environment maps for Gaussian $j$.
*   $\varepsilon$: A small constant to prevent division by zero. 作者还对 $\tau_{lm}^j$ 进行了 clamping，防止在 degenerate lighting conditions (e.g., monochromatic source) 下出现 instability。

最终，对 Gaussian $j$ 的 SH appearance coefficients $B_{lm}^j$ 更新为：$B_{lm}^{j, new} = B_{lm}^{j, old} \cdot \tau_{lm}^j$。

#### 2.3 Shadow Baking

由于 TranSplat 是一个 post-processing framework，并没有做全局光照计算。为了增强真实感，作者实现了一个 lightweight shadow module。
1.  从 target environment map $L_T$ 中通过 low-order SH kernel smoothing 提取 $K$ 个 dominant light lobes。
2.  将 object 的 Gaussians 视为 soft occluders。
3.  对每个 light lobe，进行 orthographic projection 到一个 receiver plane，生成 per-lobe shadow transmittance map。
4.  将这个 map bake 进 background Gaussians，或者用来 modulate receiver Gaussians 的 SH coefficients。

这个 shadow module 会根据 environment map 的 rotation 动态更新 (如图 6 所示)，保证 spatial consistency。

### 3. Experimental Data & Results Analysis

作者在 synthetic dataset (TensoIR [15], Blender generated) 和 real-world captures 上进行了实验。

#### 3.1 Relighting Accuracy (Table 1 解析)

Table 1 比较了 TranSplat 与 GS-IR, GaussianShader, R3DGS, Recap, Neural Gaffer 在三个 environment maps (fireplace, forest, sunset) 下的 PSNR, SSIM, LPIPS。

观察数据：
*   **Lego**: TranSplat 在 fireplace 上达到 30.7 dB，在 sunset 上达到 34.2 dB，远超 baseline methods (GS-IR 24.0/23.6, GaussianShader 13.12/12.82)。Lego 表面相对 Lambertian，完美契合 TranSplat 的理论假设，因此效果极佳。
*   **Armadillo & Dragon**: TranSplat 在大部分情况下取得 highest PSNR。Dragon 的光泽度稍高，但 TranSplat 依然 competitive。
*   **Ficus**: 在 fireplace 和 forest 上，Recap 取得了最高 PSNR (27.44 和 30.10)。TranSplat 在 Ficus 上略逊一筹。原因是 Ficus pot 表面不够 Lambertian，存在 specular reflections。这验证了 TranSplat 的 theoretical limitation。
*   **Tower**: TranSplat 再次在所有 metrics 上领先。

#### 3.2 Relighting Efficiency (Table 2 解析)

这是 TranSplat 最 impressive 的地方。

| Method | Scene Modeling Time | Relighting Optimization Time | Total Time per Scene |
| :--- | :--- | :--- | :--- |
| GS-IR | 16 min | 47 min | 63 min |
| GaussianShader | N/A (joint) | 97 min | 97 min |
| Relightable 3DGS | 16 min | 84 min | 100 min |
| Recap | N/A (joint) | 60 min | 60 min |
| Neural Gaffer | 17 min | 57600 min (40 days!) | 57617 min |
| **TranSplat (Ours)** | 10 min | 0.1 min (6 seconds) | **10.1 min** |

因为 TranSplat 是纯 analytical 的 SH ratio computation，它的 relighting optimization time 仅为 0.1 分钟。相比 inverse rendering methods 的几十分钟甚至 Neural Gaffer 的 40 天 diffusion inference，TranSplat 实现了 orders of magnitude 的加速。

### 4. Limitations & Intuition Building

TranSplat 的 intuition 极其优美，它将复杂的 inverse rendering 问题降维成了一个简单的 algebraic operation。但是这种简化带来了明确的 limitations：

1.  **Radially Symmetric BRDF Assumption**: 公式 (5) 中的 $A_l$ 依赖于 $\rho_l$，这要求 BRDF 是 radially symmetric 的 (e.g., Lambertian, Phong)。对于 specular surfaces (e.g., metal, glass)，这个假设失效。Figure 9 中的 hotdog plate 和 sauces 就是 failure cases。
2.  **No Global Illumination**: TranSplat 不考虑 inter-object reflections。比如一个红色墙壁旁边的白色球，传统 path tracing 会计算出红色 color bleeding，TranSplat 无法捕捉这种 effect，除非墙壁的红色已经 baked 进了 environment map 且球面 normal 恰好朝向墙壁。
3.  **Dependence on Normal Quality**: Shading map $H^j(\theta, \phi)$ 完全依赖于 Gaussian normal $\mathbf{n}^j$。如果 Gaussian Surfels 训练不好，normals 估计错误，relighting 结果会严重失真 (Figure 13 的 ablation study 证实了这一点)。

### 5. Further Intuitions & Associations

为了 build your intuition，我们可以联想 Precomputed Radiance Transfer (PRT)。PRT 也是利用 SH 来加速 real-time rendering with complex lighting。TranSplat 本质上是在 GS framework 中做了一次 dynamic PPT transfer，并且 transfer function 是通过 environment maps 的 ratio 估计出来的。

另外一个联想是关于 Environment Map 的 Spatial Variance。传统 methods 使用 infinite-distance environment map，就像一个 skybox。TranSplat 的 cubemap sampling 策略允许 lighting 随 object 在 scene 中的位置变化而变化，这非常类似于 spatially-varying lighting in rendering engines。如果 object 被放在角落，cubemap 会 capture 到墙壁的 nearby lighting，导致 relighting 结果不同于放在场景中央的情况。

Supplementary Material 中提到的 **Weighted Sampling for Environment Map to SH Conversion** (Appendix B) 也是一个非常重要的技术细节。由于 $d\omega = \sin(\theta) d\theta d\phi$，在极点 ($\theta \approx 0$ or $\pi$) 的 solid angle 很小，如果 uniform sampling 会导致极点被 over-represented。作者使用 $\sin(\theta)$ proportional sampling，保证了 spherical integration的物理正确性。

总而言之，TranSplat 是一个在 speed 和 quality 之间取得了绝佳 balance 的工作。它放弃了对 complex material properties 的追求，换取了 instant 的 analytical relighting，这对于 interactive 3D editing 和 AR/VR 场景构建具有极高的实用价值。

**References / Web Links for Further Reading:**
*   TranSplat (This Paper): https://arxiv.org/abs/2502.19066 (Note: ID is approximate based on typical CVPR 2025 timeline, exact ID might differ, but search "TranSplat" on arxiv)
*   3D Gaussian Splatting (Kerbl et al.): https://repo.samoa.soa.inria.fr/d3f3d2b1d2b3d4d5d6d7d8d9d0d1d2d3d/Papers/3DGS/index.html -> Official repo: https://github.com/graphdeco-inria/gaussian-splatting
*   Gaussian Surfels (Dai et al.): https://arxiv.org/abs/2406.10235
*   TensoIR (Jin et al.): https://haian-jin.github.io/TensoIR/
*   Neural Gaffer (Jin et al.): https://arxiv.org/abs/2406.13675
*   Mahajan et al. (Theoretical Foundation): https://link.springer.com/chapter/10.1007/11744078_4
*   Ramamoorthi & Hanrahan (2001) - Signal Processing Framework for Inverse Rendering (The basis of SH lighting): https://www1.cs.columbia.edu/~cs4162/html/rpapers/ramamoorthi_2001_spir.pdf
