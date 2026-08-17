---
source_pdf: GI-GS.pdf
paper_sha256: 4abe0004c0f19b028dac199447b67be2c5ff80bc8fb0ee22bfc385e2e2462e78
processed_at: '2026-08-04T21:41:00-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GI-GS 讲人话版

## 一句话直觉

想象你在玩一个 3D 游戏，游戏引擎先把整个场景画一遍（拿到 depth、normal、材质），然后再发射一堆虚拟射线，看看每个点被周围几何挡住多少光、又能从邻居那里"借"到多少反射光。GI-GS 就是把这套游戏里用了十几年的 **two-pass global illumination** 思路搬到了 3D Gaussian Splatting 上，让 inverse rendering 在 relighting 时能动态算 indirect light。

Project page: https://stopaimme.github.io/GI-GS-site/

---

## 之前的方法卡在哪

先把"敌人"画清楚：

**Vanilla 3DGS** 把每个 Gaussian 的颜色存成 spherical harmonics 系数，这直接把 material 和 lighting 焊死在一起。换个环境光，SH 系数就废了。

**GS-IR、Relightable 3D Gaussian、GIR** 这些 3DGS-based inverse rendering 方法为了处理 indirect light，普遍用两招：
- 把 occlusion / indirect light 烘焙（bake）成每个 Gaussian 的额外属性
- 或者存到一个静态 volume 里

这个 baking 的毛病在于：你换光照的时候，直接光变了，但 indirect light 没跟着变。物理上等于"我换了太阳，但屋子里的反弹光还是原来那个太阳产生的"。Relighting 出来就不对味。

**NeRF-based 方法**（NeRFactor、InvRender、TensoIR）能用 MLP 学 visibility，把 indirect light 算得比较准，但速度慢。TensoIR 4.5 小时训练，PSNR 35 左右。

GI-GS 想同时拿到两个东西：3DGS 的速度 + 动态 indirect light 的物理自洽性。

参考：
- GS-IR: https://zhigao1990.github.io/GS-IR/
- TensoIR: https://haian-jin.github.io/TensoIR/
- Relightable 3D Gaussian: https://arxiv.org/abs/2311.16043

---

## GI-GS 的核心 trick：拿 G-buffer 当跳板

3DGS 是一堆点云椭球，你没法直接对它做 path tracing —— 点云没有"表面"，射线穿过一堆 Gaussian 没明确的交点。

GI-GS 的关键 insight：**3DGS 自带的 tile-based rasterizer 本质就是个 G-buffer 生成器**。你让它额外输出 depth map、normal map、albedo/roughness/metallic map，就有了 deferred shading 需要的全部素材。然后在这个 2.5D 的 depth map 上做 ray marching，就绕开了"对点云做 path tracing"这个老大难。

这思路和游戏里的 SSAO（Screen Space Ambient Occlusion）、SSR（Screen Space Reflection）、SSGI（Screen Space Global Illumination）一脉相承 —— 都是"我没有真表面，但我有屏幕空间的 depth，那就够了"。

参考：
- SSAO 原理：https://en.wikipedia.org/wiki/Screen_space_ambient_occlusion
- Deferred shading: https://en.wikipedia.org/wiki/Deferred_shading

---

## 跑起来的样子：三 stage pipeline

### Stage 1：先把几何立起来

跑 vanilla 3DGS，每个 Gaussian 多挂一个 normal 属性。

**Depth rendering 用 normalized weight**：
$$d = \sum_{i=1}^{N} w_i d_i, \quad w_i = \frac{T_i \alpha_i}{\sum_j T_j \alpha_j}$$

直觉：直接 α-blend depth 在 Gaussian 重叠区会跳，归一化后变成加权平均，平滑。

**Normal 不用最短轴 trick**：之前 GaussianShader / GS-IR 喜欢把 Gaussian 最短轴方向当 normal，再加正则化把 Gaussian 压成扁平 disk。GI-GS 直接把 normal 当属性学，不加正则 —— 因为正则会损害渲染质量。

**Pseudo normal 监督**：从 depth map 反推 normal 当伪标签。具体做法：每个 pixel 取 $3\times3$ 邻域，9 个 depth 投影回 3D 得 9 个点，用切向量叉积得 normal，取平均。这个 pseudo normal $\hat{\mathbf{n}}$ 用来监督学到的 normal $\mathbf{n}$：
$$\mathcal{L}_n = |\mathbf{n} - \hat{\mathbf{n}}| + \lambda \mathcal{L}_{TV}$$

TV loss 是 image-guided 的：相邻 pixel 如果 RGB 差大（边界），就不惩罚 normal 差异；RGB 相近（同表面），就逼 normal 平滑。权重是 $\exp(-|I_{i,j} - I_{i-1,j}|)$，相当于 edge-aware 的 smoothness。

### Stage 2：直接光用 PBR + split-sum

G-buffer 渲出 depth / normal / albedo $a$ / metallic $m$ / roughness $\rho$，然后用 IBL（image-based lighting）+ Unreal Engine 4 的 split-sum 近似算直接光。

**Cook-Torrance BRDF**：
$$f_r = (1-m)\frac{a}{\pi} + \frac{D F G}{4(\mathbf{n}\cdot\omega_i)(\mathbf{n}\cdot\omega_o)}$$

- $a$：diffuse albedo（RGB）
- $m$：metallic（0 = 电介质，1 = 纯金属）
- $\rho$：roughness（控制 specular lobe 宽度）
- $D$：micro-facet 法线分布（GGX）
- $F$：Fresnel 项
- $G$：shadowing-masking 项
- $\omega_i, \omega_o$：入射、出射方向
- $\mathbf{n}$：表面 normal

**Split-sum 近似**把 specular 积分拆两半：
$$L_s \approx \underbrace{\int_\Omega \frac{DFG}{4(\mathbf{n}\cdot\omega_o)} d\omega_i}_{\text{2D LUT } R} \cdot \underbrace{\int_\Omega L_i(\omega_i) D(\omega_i, \omega_o)(\omega_i\cdot\mathbf{n}) d\omega_i}_{\text{prefiltered env map } I_s}$$

第一项 $R$ 只依赖 $\cos\theta$ 和 roughness，预计算成 2D LUT；第二项 $I_s$ 用不同 mip-level 的 prefiltered env map 代替。这是 Epic Games 2013 那套，游戏工业标配。

参考：Unreal split-sum paper: https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf

**直接光最终公式**：
$$L_{dir} = (1-m)\frac{a}{\pi} O(\mathbf{x}) I_{dir} + R I_s$$

- $O(\mathbf{x})$：occlusion，stage 3 算出来
- $I_{dir}$：env map 的 diffuse irradiance
- 注意 specular 没乘 occlusion，是工程上的偷懒（specular occlusion 要沿 reflection 方向做，太贵）

### Stage 3：间接光用 path tracing（灵魂部分）

这是 paper 的真正贡献。

#### Ambient Occlusion 公式

$$O(\mathbf{x}) = 1 - \frac{1}{\pi}\int_\Omega V(\omega)\, \mathbf{n}\cdot\omega\, d\omega$$

- $V(\omega) = 1$ 当且仅当沿 $\omega$ 方向射线打到其他表面（被遮挡）
- $\Omega$：normal 指向的上半球
- $\mathbf{n}\cdot\omega$：cosine weight

直觉：从 $\mathbf{x}$ 往上半球均匀射射线，被挡住的比例（加权后）就是 occlusion。

#### Ray marching 在 depth map 上

这是工程关键。射线起点 $\mathbf{x}_0 = (x_0, y_0, z_0)$，方向 $\omega$，参数化：
$$\mathbf{x} = \mathbf{x}_0 + \omega t$$

把 $\mathbf{x} = (x,y,z)$ 投影回屏幕 $(u,v)$：
$$u = x/z - c_x, \quad v = y/z - c_y$$

去 depth map 查 $(u,v)$ 处的 depth $z_d = d(u,v)$。

**相交判定**：
$$V(\omega) = \begin{cases} 1 & z_d < z < z_d + \delta \\ 0 & \text{else} \end{cases}$$

- $z$ 是当前采样点的 depth
- $z_d$ 是 depth map 上对应 pixel 的 depth
- $\delta$ 是表面厚度容差

$\delta$ 的必要性（Appendix F.3）：如果只判 $z_d < z$，那射线只要往远离相机的方向走，就会一路"穿过"所有更近的 pixel，全部判定为遮挡 —— 这显然错。加 $\delta$ 让判定只在"刚好擦着表面厚度内"才成立。

**Adaptive step size**：透视投影下远处 3D 走很远屏幕才走几 pixel，固定步长会 over-sample。自适应：
$$t = t_0\left(1 + \frac{z_0}{z_{far} - z_{near}}\right)^2$$

- $t_0$：base step
- $z_0$：当前 ray 起点深度
- $z_{far}, z_{near}$：远 / 近裁剪面

远处 $z_0$ 大，$t$ 就大，步长大。这思路和 mip-map 一致：远处低采样率。

#### Indirect lighting 公式

核心近似：从 $\mathbf{x}$ 沿 $\omega$ 看到的 $\hat{\mathbf{x}}$ 反射回来的光，近似等于 first-pass 渲染图上 $\hat{\mathbf{x}}$ 对应 pixel 的 RGB：
$$L_i(\mathbf{x}, \omega) \approx I_{dir}(\hat{u}, \hat{v})$$

为什么能这么近似？Appendix F.2 给了推导：
$$\Delta L_o = L_o(\hat{\mathbf{x}}, -\omega) - L_o(\hat{\mathbf{x}}, \omega_1) = L_s(\hat{\mathbf{x}}, -\omega) - L_s(\hat{\mathbf{x}}, \omega_1)$$

也就是说，从 $\hat{\mathbf{x}}$ 反射给 $\mathbf{x}$ 的光，和相机看到 $\hat{\mathbf{x}}$ 的光，差异只在 specular 部分。Diffuse 各向同性，没差异。Specular 在大多场景相对小，工程上忽略。

**间接光积分**：
$$L_{ind} = (1-m)\frac{a}{\pi}\int_\Omega V(\omega_i)\, I_{dir}(\hat{u}, \hat{v})\, \mathbf{n}\cdot\omega_i\, d\omega_i$$

**最终渲染**：
$$L_o = \underbrace{(1-m)\frac{a}{\pi} O(\mathbf{x}) I_{dir} + R I_s}_{\text{first-pass (direct)}} + \underbrace{L_{ind}}_{\text{second-pass (indirect)}}$$

这就是 one-bounce 全局照明的近似。Relighting 时重跑这两 pass，indirect 自然跟随直接光变化，物理自洽。

#### Cubemap 扩展到 world space

Screen-space path tracing 只看得到相机 frustum 内的几何。对 Mip-NeRF 360 这种 unbounded 场景，indirect light 可能来自视野外。

解决方案：渲染一个 cubemap —— 当前视角当 front face，再旋转相机渲染其余 5 个面，构成 360° 的 depth cubemap + RGB cubemap。Path tracing 时从 cubemap 采样。

这就是 paper 里 "Ours-Cubemap" 变体。代价是多渲染 5 个面，速度变慢，但遮挡区域更平滑（Fig. 6）。

参考：Cube mapping: https://en.wikipedia.org/wiki/Cube_mapping

---

## 实验讲人话

### TensoIR Dataset（Table 1）

| 方法 | NVS PSNR | Albedo PSNR | Relighting PSNR | Time |
|---|---|---|---|---|
| TensoIR | 35.09 | 29.27 | **28.58** | 4.5 hrs |
| GS-IR | 35.33 | 30.29 | 24.37 | 33 min |
| **GI-GS** | **36.75** | **31.97** | 24.70 | **28 min** |

- NVS / Albedo SOTA，比 GS-IR 高 1.4 dB
- Relighting 输给 TensoIR 约 4 dB。原因：TensoIR 用 MLP 学 visibility，物体级几何简单，visibility 学得准；GI-GS 依赖 G-buffer 几何质量，TensoIR 物体几何太干净，path tracing 反而显不出优势
- 训练 28 min，比 GS-IR 快 5 min，比 TensoIR 快 10x

### Mip-NeRF 360（Table 2）

- Outdoor：23.81 vs GS-IR 24.01 —— 略输。户外开阔，indirect light 贡献小，path tracing 噪声反而拖累
- Indoor：29.07 vs GS-IR 27.46 —— 高 1.6 dB。室内遮挡多，indirect light 贡献大，GI-GS 优势明显

Table 5 单场景：garden 上 26.42 vs 3DGS 27.41 —— 唯一明显输的，garden 开阔，path tracing 噪声 + 没有 indirect light 优势，双重不利。

### Ablation（Table 3, 4）

- 去掉 occlusion：PSNR 降 0.38
- 去掉 indirect light：PSNR 降 1.15 —— indirect light 更关键
- 采样数 $N_s = 16/64/256$：PSNR 36.25/36.75/36.82，时间 27/28/32 min。64 已经够用，边际收益递减

---

## 这套思路让我联想到什么

**1. SSGI（Screen Space Global Illumination）的神经化版本**
游戏里 SSGI 就是这套：从 depth map 上做 ray marching 拿 indirect light，局限是只能看到屏幕内。Crytek、UE5 都有实现。GI-GS 等于把 SSGI 包装进 inverse rendering 的可微管线。

参考 SSGI: https://docs.unrealengine.com/5.0/en-US/screen-space-global-illumination-in-unreal-engine/

**2. Deferred shading 的胜利**
3DGS 之前做 inverse rendering 的方法都纠结于"怎么对点云做光追"。GI-GS 直接用 rasterizer 出 G-buffer，绕开问题。这思路很 Karpathy 式 —— 不发明新理论，而是把已有理论正确组合。

**3. 和 TensoIR 的对比有意思**
TensoIR 用 tri-plane 表示 + MLP visibility，物体级强但场景级弱。GI-GS 用 G-buffer + cubemap，物体级输但场景级赢。两种范式各有领地，物体级几何太规整，path tracing 优势显不出来；场景级遮挡复杂，MLP 学不动 visibility。

**4. 1-bounce 近似够不够**
真实 GI 需要 multi-bounce，GI-GS 只 1 bounce。但 TensoIR 也是 1 bounce，游戏 SSGI 也是 1 bounce。工业经验是 1 bounce 已经能拿到 80% 的视觉效果，剩余 20% 需要付出指数级代价。Multi-bounce 可以迭代 first-pass / second-pass 多次，是自然的下一步。

**5. Mitsuba 3 / nvdiffrast 路线**
另一条路是完全可微的 path tracer，像 Mitsuba 3 那样。优点是物理严格，缺点是慢。GI-GS 用工程近似换速度，是另一种权衡。

参考：
- Mitsuba 3: https://www.mitsuba-renderer.org/
- nvdiffrast: https://nvlabs.github.io/nvdiffrast/

**6. 几何质量是天花板**
Path tracing 依赖 depth / normal 精度。Appendix E 用 Omnidata 监督 normal 能显著提升 relighting，说明几何是瓶颈。用 2DGS、PGSR、GOF 这些高几何质量的 3DGS 变体做前端，可能直接提升 GI-GS 的 relighting 质量。

参考：
- 2DGS: https://arxiv.org/abs/2403.17888
- PGSR: https://arxiv.org/abs/2406.06521
- GOF: https://arxiv.org/abs/2404.10772

**7. Specular indirect 是大坑**
现在只做 diffuse indirect。Specular indirect 需要 Monte Carlo 沿 BRDF lobe 采样，CG 里也是 expensive 的（路径追踪、双向路径追踪）。SSR 是 hack 版本，质量一般。这是 GI-GS 的明确 limitation。

**8. Spatially varying lighting 没解决**
用 env map 当直接光，意味着全场景同一个光照。室内场景明明有窗户、台灯、吊灯，这些 spatial variation 都表达不了。NeILF 用 neural incident light field 试图解决，但效果一般。这是 inverse rendering 共同痛点。

参考 NeILF: https://arxiv.org/abs/2110.03953

**9. Sharp shadow 表达弱**
Hemisphere 积分得到的是 soft ambient occlusion，要 sharp shadow 还得 shadow map（Appendix F.5 给了 demo）。这意味着 GI-GS 对室内有明确光源的场景，shadow 会偏软。

**10. Importance sampling 可能加速**
现在用 uniform sampling，理论上沿 BRDF lobe importance sampling 可以减少 sample 数。但实现复杂度高，工程权衡上 uniform + 64 sample 已经够用。

---

## 我的吐槽 / 思考

**1. 这 paper 工程价值 > 科学价值**
没提出新 BRDF、没改 rendering equation，全是组合已有的：deferred shading + split-sum + path tracing + cubemap + image-guided TV loss。但组合得对，组合得巧，所以 work。这种 paper 我很欣赏 —— 计算机视觉很多时候不需要新理论，需要把对的东西放在一起。

**2. Relighting 输给 TensoIR 是合理的**
TensoIR 在物体级 relighting PSNR 28.58，GI-GS 24.70，差 4 dB。这是因为 TensoIR 物体级 GT visibility 学得好。但 TensoIR 训练 4.5 hrs，GI-GS 28 min。如果你只关心物体级 relighting，TensoIR 仍更好；如果关心场景级 + 速度，GI-GS 赢。

**3. Cubemap 版本 PSNR 反而降一点，但 qualitative 更好**
Table 2 上 Ours-Cubemap 在 indoor PSNR 29.07 vs Ours 29.29 —— 略降。但 Fig. 6 视觉上明显更平滑。这反映 PSNR 不能完全捕捉 indirect light 质量，人眼对平滑的间接光更敏感。这种 case 提醒我们看 paper 别只看 metrics。

**4. 把 3DGS rasterizer 当 G-buffer 生成器，这个视角值得推广**
3DGS 的 rasterizer 本质就是个"可微的 raster pipeline"，可以输出任意 per-pixel attribute。GI-GS 用它输出 depth / normal / BRDF，但其实还可以输出更多 —— motion vector、segmentation mask、object ID 等。这思路可以反过来影响 forward rendering 领域。

**5. 跟NeRF series 的对比**
NeRF 那套 inverse rendering 一直在和"如何对隐式表面做光追"作斗争，方案越来越复杂（NeRFactor 多阶段训练、InvRender MLP visibility、TensoIR tri-plane ray tracing）。3DGS 因为显式 + rasterizer，反而让 G-buffer trick 直接可用。某种程度上这是 representation 决定 algorithm complexity 的一个例证。

**6. 工业落地可能性**
28 分钟训练，relighting 可实时（重跑 two-pass 即可），适合 VR/AR、电商商品展示、数字人等场景。限制是需要高质量几何和静态光照假设。动态场景、动态光照还没碰。

---

## 总结

GI-GS 是一篇很"工程师"的 paper：它没发明新物理，只是把游戏工业的 deferred shading + SSGI + split-sum 那套搬到 3DGS 上，让 inverse rendering 的 indirect light 在 relighting 时动态算。核心 trick 是用 3DGS rasterizer 出 G-buffer，绕开"对点云做 path tracing"的难题。

实验上 NVS / Albedo SOTA，relighting 物体级输给 TensoIR 但场景级赢，速度比 TensoIR 快 10x。Limitation 集中在 specular indirect、sharp shadow 和 spatially varying lighting —— 这些都是后续工作可以攻的明确方向。

如果你想 build intuition，记住一句话：**3DGS 的 rasterizer 本质是个 G-buffer 生成器，所有 deferred shading 工业管线都能嫁接上来**。这个视角比 GI-GS 本身更有价值。

---

# GI-GS: 3DGS 上的 Global Illumination 分解

## 1. Paper 一句话总结与定位

这篇 paper 来自 HKUST 的 Hongze Chen、Zehong Lin、Jun Zhang，核心贡献是把 **deferred shading** + **path tracing** 这套游戏工业里的成熟管线，嫁接到 **3D Gaussian Splatting (3DGS)** 的 inverse rendering 上，让 3DGS 在 relighting 时能够**动态**计算 indirect lighting（多次弹射的光），而非像 GS-IR、Relightable 3D Gaussian 那样把 occlusion 和 indirect light 烘焙（bake）成静态属性。

Project page: https://stopaimme.github.io/GI-GS-site/

---

## 2. 为什么这个工作有意义：3DGS inverse rendering 的"间接光困境"

要 build intuition，先得想清楚现状里 3DGS-based inverse rendering 的几个核心痛点：

**(1) Vanilla 3DGS 把颜色当成 SH 系数**，每个 Gaussian 存一组 spherical harmonics 来表达 view-dependent color。这直接把 *material*（BRDF）和 *lighting*（环境光）耦合到一起。一旦要换光照（relighting），SH 系数就废了。

**(2) NeRF-based inverse rendering（NeRFactor、InvRender、TensoIR）** 走的是 MLP 路线，能做 indirect lighting 但慢，TensoIR 已经用了 tri-plane + ray tracing 来估 indirect lighting，PSNR ~35，但训练 4.5 hrs。

**(3) 3DGS-based inverse rendering（GS-IR、Relightable 3DGS、GIR、GaussianShader）** 通常用两种 hack 来处理 indirect lighting：
- 把 indirect lighting 当成每个 Gaussian 的额外 SH 属性（Relightable 3D Gaussian）
- 用 baked volume 存 occlusion + indirect lighting（GS-IR）

这两种做法本质上是"静态烘焙"，**问题在于**：relighting 时直接光变了，但 baked 出来的 indirect light 没变，物理上不自洽。

GI-GS 的核心 insight：**用 G-buffer 重建表面几何，在重建表面上做 path tracing，根据 first-pass 渲染结果（含直接光）动态算出 second-pass 的 indirect light**。这等于把 forward rendering 工业管线里很标准的"two-pass global illumination"思路拿过来用到 inverse rendering 里。

参考链接：
- 3DGS 原文：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- GS-IR: https://zhigao1990.github.io/GS-IR/
- TensoIR: https://haian-jin.github.io/TensoIR/
- NeRFactor: https://people.csail.mit.edu/xiuming/projects/nerfactor/

---

## 3. 方法详解：分三 stage 的 pipeline

### 3.1 Stage 1: Geometry Reconstruction

先跑一遍 vanilla 3DGS，但每个 Gaussian 多了一个 **normal** 属性。

**Depth rendering**：用 normalized weight 而不是直接 α-blending：
$$d = \sum_{i=1}^{N} w_i d_i, \quad w_i = \frac{T_i \alpha_i}{\sum_{i=1}^{N} T_i \alpha_i}$$

变量含义：
- $d$：渲染得到的 depth
- $d_i$：第 $i$ 个 Gaussian 在该 pixel 处的 depth
- $w_i$：normalized weight，把原始 transmittance 归一化到和为 1
- $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$：累积 transmittance
- $\alpha_i$：第 $i$ 个 Gaussian 的 opacity

这种归一化写法相当于让 weight 在最远和最近的 Gaussian 之间平滑过渡，比直接 α-blend 更鲁棒，能减少 depth 在 Gaussian 重叠区域的 artifact。

**Normal estimation**：normal $\mathbf{n}$ 直接作为 Gaussian 的属性，用 α-blending 渲染 normal map。**不**像 GaussianShader / GS-IR 那样把 Gaussian 最短轴方向当 normal 再加正则化把它压成 disk —— 因为正则化会损害渲染质量。

**Pseudo normal 监督**：从 depth map 反推 pseudo normal $\hat{\mathbf{n}}$。具体做法（Appendix B）：对每个 pixel $(u,v)$，取 $3\times3$ 邻域，把 9 个 depth 投影回 3D，用切向量叉积得 normal，最后取平均。然后用这个 pseudo normal 监督 rendered normal：
$$\mathcal{L}_n = \mathcal{L}_{n,p} + \lambda_{n-TV}\mathcal{L}_{TV_{normal}}, \quad \mathcal{L}_{n,p} = |\mathbf{n} - \hat{\mathbf{n}}|$$

TV loss 是 image-guided 的，公式见 Appendix Eq. 18-19：
$$\triangle_{ij}^N = \exp(-|I_{i,j} - I_{i-1,j}|)(N_{i,j} - N_{i-1,j})^2 + \exp(-|I_{i,j} - I_{i,j-1}|)(N_{i,j} - N_{i,j-1})^2$$

直觉上：相邻 pixel 如果 RGB 差异大（说明可能是物体边界），那 normal 的差异就不该被惩罚；RGB 相近（同一表面内），normal 该平滑。$\exp(-|I|)$ 当作 edge-aware weight。

### 3.2 Stage 2: Direct Lighting Modeling（PBR + Deferred Shading）

G-buffer 渲染出 depth map、normal map、BRDF maps（albedo $a$、metallic $m$、roughness $\rho$）。然后用 **Image-Based Lighting (IBL)** + **split-sum approximation**（Unreal Engine 4 的 Karis 2013 那套）算直接光。

**核心公式 (Cook-Torrance BRDF)**：
$$f_r(\omega_i, \omega_o) = \underbrace{(1-m)\frac{a}{\pi}}_{\text{diffuse } f_d} + \underbrace{\frac{D(h;\rho) F(\omega_o, h; a, m) G(\omega_i, \omega_o, h; \rho)}{4(\mathbf{n}\cdot\omega_i)(\mathbf{n}\cdot\omega_o)}}_{\text{specular } f_s}$$

变量含义：
- $a \in [0,1]^3$：diffuse albedo（RGB 三通道）
- $m \in [0,1]$：metallic 值，1 表示纯金属
- $\rho \in [0,1]$：roughness，控制 specular lobe 大小
- $h = \frac{\omega_i + \omega_o}{\|\omega_i + \omega_o\|}$：half vector，入射和出射方向的角平分线
- $D$：Normal Distribution Function（GGX 之类），描述 micro-facet 法线分布
- $F$：Fresnel 项，描述不同入射角的反射率
- $G$：Geometry / shadowing-masking 项，描述 micro-facet 自遮挡

**Split-sum 近似**（Eq. 9）：把 specular 积分拆成 BRDF integral + prefiltered env map：
$$L_s \approx \underbrace{\int_\Omega \frac{DFG}{4(\mathbf{n}\cdot\omega_o)} d\omega_i}_{R \text{ (2D LUT)}} \cdot \underbrace{\int_\Omega L_i(\omega_i) D(\omega_i, \omega_o)(\omega_i \cdot \mathbf{n}) d\omega_i}_{I_s \text{ (prefiltered env map)}}$$

- $R$：只依赖 $\cos\theta = \mathbf{n}\cdot\omega_o$ 和 roughness $\rho$，可以预计算成 2D LUT
- $I_s$：用 NDF $D$ 加权的环境光积分，不同 roughness 对应不同 mip-level 的 prefiltered environment map

**最终直接光公式 (Eq. 10)**：
$$L_{dir} = (1-m)\frac{\mathbf{a}}{\pi} O(\mathbf{x}) I_{dir} + R I_s$$

- 第一项是 diffuse 直接光，乘上 occlusion $O(\mathbf{x})$
- 第二项是 specular 直接光
- $O(\mathbf{x})$ 是 path tracing 算出的 occlusion（见 Stage 3）
- $I_{dir}$ 是 prefiltered environment map 中的 diffuse irradiance

注意：specular 没乘 occlusion，这是 deferred shading 常用近似 —— specular 的 occlusion 应该沿 reflection 方向做，复杂度高，先省略。

### 3.3 Stage 3: Indirect Lighting Modeling（核心创新）

这是 paper 的灵魂部分。

#### 3.3.1 Ambient Occlusion 公式

$$O(\mathbf{x}) = 1 - \frac{1}{\pi}\int_\Omega V(\omega)\, \mathbf{n}\cdot\omega\, d\omega$$

- $O(\mathbf{x})$：点 $\mathbf{x}$ 处的 occlusion（值越大越不遮挡）
- $\Omega$：以 $\mathbf{x}$ 为中心、normal $\mathbf{n}$ 指向的上半球
- $V(\omega)$：visibility 函数，沿 $\omega$ 方向 ray 出去，若碰到表面为 1（被遮挡），否则为 0
- $\mathbf{n}\cdot\omega$：cosine weight，符合 Lambert 投影

在球坐标下展开（Eq. 25）：
$$O(\mathbf{x}) = 1 - \frac{1}{\pi}\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi/2} V(\phi,\theta)\cos\theta\sin\theta\, d\phi\, d\theta$$

均匀采样估计（Eq. 26）：
$$O(\mathbf{x}) = 1 - \frac{\sum_{i=0}^{n_1}\sum_{j=0}^{n_2} V(\phi_i,\theta_j)\cos\theta_j\sin\theta_j}{\sum_{i=0}^{n_1}\sum_{j=0}^{n_2}\cos\theta_j\sin\theta_j}$$

直觉：分母是归一化常数，确保完全无遮挡时 $O=1$。

#### 3.3.2 Adaptive Path Tracing（关键工程）

3DGS 是点云表示，没法直接做传统 mesh-based ray tracing。GI-GS 的 trick：**从 G-buffer 的 depth map 反向恢复表面，然后在这个 depth map 上做 ray marching**。

**Ray 表达式**：
$$\mathbf{x} = \mathbf{x}_0 + \omega t$$

- $\mathbf{x}_0 \in \mathbb{R}^3$：ray 起点
- $\omega \in \mathbb{R}^3$：ray 方向（单位向量）
- $t$：沿 ray 的距离

把 3D 点 $\mathbf{x} = (x,y,z)$ 投影到屏幕坐标 $(u,v)$：
$$u = \frac{x}{z} - c_x, \quad v = \frac{y}{z} - c_y$$

- $z$：3D 点的 view space depth
- $(c_x, c_y)$：光心坐标（principal point）

**相交判定**：查屏幕坐标 $(u,v)$ 在 depth map 上的值 $z_d = d(u,v)$，比较 $z$ 和 $z_d$：
$$V(\omega) = \begin{cases} 1 & \text{if } z_d < z < z_d + \delta \\ 0 & \text{else} \end{cases}$$

- $z$：当前采样点的 depth
- $z_d$：该方向 depth map 上对应的 depth
- $\delta$：表面厚度容差，避免 ray 擦边过去被误判

这里 $\delta$ 的作用（Appendix F.3）：单纯比较 $z_d < z$ 会把所有 depth 更近的 pixel 都算遮挡，引入 $\delta$ 让遮挡判定更严格。

**Adaptive step size**：透视投影下，远处 3D 走很远，屏幕上才走几个 pixel，固定步长会 over-sample。改用：
$$\mathbf{x} = \mathbf{x}_0 + \omega k t, \quad k \in \mathbb{Z}$$
$$t = t_0 \left(1 + \frac{z_0}{z_{far} - z_{near}}\right)^2$$

- $t_0$：base step size
- $z_0$：当前 ray 起点的 depth
- $z_{far}$、$z_{near}$：far/near clipping plane 距离
- 离相机越远（$z_0$ 越大），步长越大

直觉：远处表面在屏幕上更密集，可以用大步长；近处需要细节，用小步长。这个 trick 类似 mip-mapping 的思想，远距离低分辨率。

#### 3.3.3 Indirect Lighting 公式

关键物理直觉（Eq. 29）：从 $\mathbf{x}$ 沿 $\omega$ 方向看到的 $\hat{\mathbf{x}}$ 反射回来的光，近似等于从相机看到 $\hat{\mathbf{x}}$ 的 RGB 值：
$$L_i(\mathbf{x}, \omega) = L_o(\hat{\mathbf{x}}, -\omega) \approx I_{dir}(\hat{u}, \hat{v})$$

近似成立的理由（Appendix F.2）：diffuse 是各向同性的，specular 部分相对小很多，所以 $L_o(\hat{\mathbf{x}}, -\omega) - L_o(\hat{\mathbf{x}}, \omega_1) \approx L_s(\hat{\mathbf{x}}, -\omega) - L_s(\hat{\mathbf{x}}, \omega_1)$，差异是 specular，工程上忽略。

**Indirect diffuse 公式 (Eq. 13)**：
$$L_{ind} = (1-m)\frac{a}{\pi}\int_\Omega L_i(\omega_i, \mathbf{x})\, \mathbf{n}\cdot\omega_i\, d\omega_i$$
$$L_i(\omega_i, \mathbf{x}) = V(\omega_i, \mathbf{x})\, I_{dir}(\hat{u}, \hat{v})$$

- $I_{dir}(\hat{u}, \hat{v})$：first-pass 渲染图上 $\hat{\mathbf{x}}$ 对应 pixel 的 RGB
- $V(\omega_i, \mathbf{x})$：visibility，1 表示该方向有"间接光源"（被其他物体遮挡即接收其反射光）

**最终渲染公式 (Eq. 14)**：
$$L_o(\omega_o, \mathbf{x}) = \underbrace{(1-m)\frac{\mathbf{a}}{\pi}O(\mathbf{x})I_{dir} + R I_s}_{\text{First-pass (direct)}} + \underbrace{L_{ind}}_{\text{Second-pass (indirect)}}$$

这是 paper 的核心方程，把直接光和间接光相加。

#### 3.3.4 Cubemap 扩展到 World Space（Sec. 4.4）

Screen-space path tracing 只考虑相机 frustum 内的几何。对 Mip-NeRF 360 这种真实场景，occlusion 和 indirect light 可能来自相机视野外。解决方案：渲染一个 cubemap，即把当前视角当 front face，再旋转相机渲染其余 5 个面，构成完整 360° 的 depth cubemap + RGB cubemap，然后 path tracing 时从 cubemap 采样。

这就是 paper 里 "Ours-Cubemap" 变体。Table 2 显示 cubemap 版本 PSNR 略低（24.01 vs 23.81 outdoor，29.07 vs 29.29 indoor —— 注意 indoor PSNR 反而稍降），但 qualitative 上遮挡区域更平滑（Fig. 6）。

### 3.4 Loss Function

**Stage 1 初始化 loss (Eq. 16)**：
$$\mathcal{L}_{init} = \mathcal{L}_1 + \mathcal{L}_{SSIM} + \mathcal{L}_n$$

**Stage 2-3 decomposition loss (Eq. 15/20)**：
$$\mathcal{L}_d = \underbrace{\mathcal{L}_1}_{\mathcal{L}_{color}} + \underbrace{\lambda_M \mathcal{L}_{TV_{mat}}}_{\mathcal{L}_{material}} + \underbrace{\lambda_E \mathcal{L}_{TV_{light}}}_{\mathcal{L}_{light}}$$

- $\mathcal{L}_1$：渲染图和 GT 的 L1 loss
- $\mathcal{L}_{TV_{mat}}$：material map 的 image-guided TV loss（Eq. 21-22），让相邻 pixel 的 material 平滑
- $\mathcal{L}_{TV_{light}}$：environment map 的 TV loss（Eq. 23-24），让 env map 平滑
- 权重：$\lambda_{n-TV} = 5.0$，$\lambda_M = 1.0$，$\lambda_E = 0.01$

训练 30K iter 跑 vanilla 3DGS + normal，再加 5K-10K iter 优化 material/lighting。单卡 A5000，28 分钟跑完 TensoIR。

---

## 4. 实验数据解读

### 4.1 TensoIR Dataset（Table 1）

| Method | NVS PSNR | Albedo PSNR | Relighting PSNR | Time |
|---|---|---|---|---|
| NeRFactor | 24.68 | 25.13 | 23.38 | >100 hrs |
| InvRender | 27.37 | 27.34 | 23.97 | 14 hrs |
| NVDiffrec | 30.70 | 29.17 | 19.88 | <1 hr |
| TensoIR | 35.09 | 29.27 | **28.58** | 4.5 hrs |
| GS-IR | 35.33 | 30.29 | 24.37 | 33 min |
| **GI-GS** | **36.75** | **31.97** | 24.70 | **28 min** |

关键观察：
- NVS / Albedo 是 SOTA，比 GS-IR 高 ~1.4 dB PSNR
- Relighting 输给 TensoIR —— 这是因为 TensoIR 用 MLP 学 visibility，对物体级数据更精确；而 GI-GS 的 path tracing 依赖 G-buffer 几何质量
- 训练时间 28 分钟，比 GS-IR 还快 5 分钟，比 TensoIR 快 10x

### 4.2 Mip-NeRF 360 Dataset（Table 2, 5-7）

Outdoor PSNR：23.81（Ours）vs 24.01（GS-IR）—— 略低
Indoor PSNR：29.07（Ours）vs 27.46（GS-IR）—— 高 1.6 dB

**Indoor 提升明显**的直觉：室内场景遮挡多、indirect light 贡献大，GI-GS 的动态 path tracing 比 GS-IR 的 baked volume 优势明显。

Table 5 单场景细分：garden 上 Ours 26.42 vs 3DGS 27.41 —— 这是唯一明显输的，garden 是开阔户外场景，indirect light 作用小，反而 path tracing 噪声带来退化。

### 4.3 Ablation（Table 3, 4）

- **无 occlusion**：TensoIR PSNR 36.37 → 36.75（去掉降 0.38）
- **无 indirect lighting**：TensoIR PSNR 35.60 → 36.75（去掉降 1.15）—— indirect light 更重要
- **Sample 数量**：$N_s = 16/64/256$ 对应 PSNR 36.25/36.75/36.82，时间 27/28/32 min —— 边际收益递减，64 已经够用

---

## 5. 关键 Insight 与 Limitation

### 5.1 核心 Insight

**(1) Deferred shading 是关键**：直接对 3DGS 点云做 path tracing 很难（点云没表面），通过 G-buffer 把几何先栅格化成 2D depth map + normal map，再在 depth map 上做 ray marching，绕过了这个难题。这和游戏工业的 G-buffer 思路完全一致。

**(2) Two-pass 渲染的物理自洽性**：first-pass 用 PBR 算直接光，second-pass 用 first-pass 结果算间接光 —— 这本质上是一次光线弹射的近似，物理上等价于把 rendering equation 在 first bounce 后截断。Relighting 时只要重跑这两 pass，indirect light 会自然跟随直接光变化，物理上自洽。

**(3) Adaptive ray marching 处理透视投影**：透视投影非线性，固定步长会 over-sample 远处。用 $t \propto (1 + z_0/(z_{far}-z_{near}))^2$ 自适应，是工程上很巧的 trick，类似 mip-map 思想。

**(4) Cubemap 把 screen space 扩展到 world space**：对 unbounded 场景，单一 depth map 不够，渲染 cubemap 覆盖 360° 几何，这个思路类似 SSR (Screen Space Reflection) 升级到 world space reflection。

### 5.2 Limitation

- **没考虑 specular indirect lighting**（Sec. 6）：只对 diffuse 做了 indirect，specular indirect 需要 Monte Carlo 多次弹射，CG 里也是难题
- **依赖几何质量**：path tracing 依赖 depth map 和 normal map 的精度，几何差则间接光算不准（Appendix E 用 Omnidata 监督 normal 能显著提升 relighting）
- **环境光模型太简单**：用 environment map 当直接光，无法表达 spatially varying lighting（如室内灯）
- **Sharp shadow 表达弱**（Appendix F.5）：hemisphere 积分得到的 occlusion 是软阴影，要 sharp shadow 还得用 shadow map

---

## 6. 与相关工作的联系

- **TensoIR**（Jin et al. 2023）：tri-plane 表示 + ray tracing 算 indirect light，物体级强但场景级弱。GI-GS 用 G-buffer + cubemap 扩展到场景级。https://haian-jin.github.io/TensoIR/
- **GS-IR**（Liang et al. 2024）：3DGS + PBR + baked volume 存 occlusion/indirect，速度快但 relighting 时不更新 indirect。GI-GS 直接竞争对象。https://zhigao1990.github.io/GS-IR/
- **Relightable 3D Gaussian**（Gao et al. 2023）：每个 Gaussian 多加 SH 存 indirect light，point-based ray tracing 算 occlusion，烘焙思路。https://arxiv.org/abs/2311.16043
- **GaussianShader**（Jiang et al. 2024）：用 shading function 处理反射表面，但没显式建模 indirect light。https://jidaspring.github.io/projects/GaussianShader/
- **NeRFactor / InvRender**：NeRF 系 indirect lighting 老大难，InvRender 用 MLP 学 visibility。https://people.csail.mit.edu/xiuming/projects/nerfactor/
- **PGSR / 2DGS / GOF**：高几何质量的 3DGS 变体，可以作为 GI-GS 的几何前端的更好替代。https://arxiv.org/abs/2406.06521 ; https://arxiv.org/abs/2404.10772
- **Unreal Engine 4 split-sum**：https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
- **Cook-Torrance BRDF**：1982 经典 BRDF 模型。https://en.wikipedia.org/wiki/Specular_highlight#Cook%E2%80%93Torrance_model
- **Deferred shading in games**：https://en.wikipedia.org/wiki/Deferred_shading

---

## 7. 我的思考 / 联想

**为什么 path tracing on depth map 是优雅的**：3DGS 的 rasterizer 输出 depth map 几乎免费，再在这个 2.5D 表示上做 ray marching，相当于把 3D path tracing 降维成 2.5D screen-space ray marching，可微、可并行、可 GPU 加速。这思路和 SSAO (Screen Space Ambient Occlusion)、SSR (Screen Space Reflection)、SSGI (Screen Space Global Illumination) 一脉相承 —— 都是 deferred shading 工业管线的神经化版本。

**潜在的下一步**：
1. 多次弹射：现在只 1 bounce，理论上可迭代 first-pass / second-pass 多次得到 multi-bounce indirect light
2. Specular indirect：用 split-sum 思路对 reflection 方向也做 path tracing，类似 SSR
3. Spatially varying lighting：env map 换成 learnable point lights 或 neural lighting field
4. Geometric prior：用 2DGS（Surfels）或 PGSR 的平面约束，得到更准确 normal，path tracing 质量自然提升
5. Monte Carlo 替代 uniform sampling：importance sampling 沿 BRDF lobe 采样，可能减少 sample 数
6. Differentiable path tracing 全栈：参考 Mitsuba 3 / nvdiffrast 的可微渲染思路，可能更通用但慢
7. NeRF-Gaussian hybrid：远距离用 NeRF 表示（cubemap 起作用），近距离用 3DGS（精细 path tracing）

**Mitsuba 3**: https://www.mitsuba-renderer.org/
**nvdiffrast**: https://nvlabs.github.io/nvdiffrast/
**2DGS**: https://arxiv.org/abs/2403.17888
**PGSR**: https://arxiv.org/abs/2406.06521

---

## 8. 总结

GI-GS 的工程价值大于科学价值：它没有提出新的 BRDF 模型或新的渲染方程近似，而是把 deferred shading + two-pass global illumination + adaptive ray marching + cubemap world-space extension 这套游戏工业里成熟的做法，巧妙嫁接到 3DGS 上。**核心创新点是把 3DGS 的 rasterizer 当作 G-buffer 生成器**，从而绕开"对点云做 path tracing"这个难题。这思路很 Karpathy 式 —— 不发明新理论，而是把已有理论正确组合，让它在新的表示框架下 work。

实验上 NVS / Albedo SOTA，relighting 输给 TensoIR（物体级），但训练快 10x。Indoor 场景优势明显，验证了 indirect light 在遮挡密集场景的重要性。Limitation 集中在 specular indirect、sharp shadow 和 spatially varying light，这些都是后续工作可以攻的方向。
