---
source_pdf: MESH SPLATTING FOR END-TO-END MULTIVIEW SURFACE RECONSTRUCTION.pdf
paper_sha256: 3d4ead5b71d81a4e1fd94f0f40200eed544cc162bd3d27366dfa4b5b94f5dc16
processed_at: '2026-08-05T17:44:55-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Mesh Splatting

## 一句话说清楚

这篇 paper 的核心 idea 就是:**把一个硬邦邦的 mesh "吹松"成一团半透明的"棉花糖"**,这样它就能像 NeRF 那样被 photometric loss 直接优化,但底子还是 mesh,可以 remeshing、可以控制顶点数。

---

## 为什么需要这么搞?先讲清楚两个痛点

### 痛点 1: Volumetric 方法的 "meshing 诅咒"

NeRF、NeuS、Neuralangelo 这些 volumetric 方法本质上是把整个 3D 空间都填上 density,然后沿 ray 一路积分。好处是 **receptive field 大**——ray 经过的地方都能产生梯度,几何再偏也能被 "pull" 回来。

但最后你要 mesh,就得跑 Marching Cubes 或者 Poisson Reconstruction。这一步就出问题了:

1. **mesh 过密**。DTU 上随便就 1000K vertices,实际应用用不了这么多。
2. **误差累积**。volumetric field 里有一点 noise,meshing 就会把这个 noise "固化"成几何误差。Fig.1 红圈里能看到 mesh 和 GT point cloud 对不齐。
3. **upper bound 被卡住**。Fig.1e 做了个实验:直接拿 GT point cloud 跑 Poisson,结果蓝圈处 Poisson 直接漏点。意思是**就算你有完美 point cloud,meshing 也会丢信息**,volumetric pipeline 的 ceiling 就是被 meshing 这步压住的。

### 痛点 2: Mesh 方法的 "单层近视眼"

NvdiffRec、IMLS-Splatting、SuGaR 这些直接优化 mesh 的方法避免 meshing,但 mesh 是个 **opaque single-layer surface**。你看 Fig.2a:

假设真实 surface 在 base mesh 下方 5cm。Base mesh 上有个点 A 漂在空中。Multi-view 训练时,A 在所有视角下都能被看到,优化器只能学到 "A 的颜色应该是什么"——它**完全不知道 A 下方 5cm 才是真实表面**。梯度信号在 A 附近完全没有 "往下拉" 的方向。

为了补救,这些方法只能靠 **priors**:
- Shading supervision(假设光照模型已知,但现实光照经常违反假设)
- Monocular normal prediction(Omnidata 这种网络预测的 normal 很 noisy)
- Depth supervision(Kinect 这种 sensor 有测量误差)

这些都是**间接信号**,远不如直接 photometric loss 信息量大。

---

## 核心 idea: "软化的 mesh" = mesh 的皮 + volumetric 的魂

### 直觉类比

想象 mesh 是一张硬纸板。硬纸板的问题:如果它不在正确位置,你看不到它后面有什么,梯度也穿不过去。

现在把这张硬纸板**变成 5 张半透明纸**,每张 opacity 不同,叠在一起形成一个 "soft band"。这个 soft band 就像 volumetric field 的局部版本——ray 穿过时会经过多个半透明层,每层都贡献 color,合在一起就是 alpha compositing。

关键 trick:**这 5 张半透明纸的位置是 base mesh 沿 normal 偏移出来的**。所以如果半透明层的 color 不对(说明真实 surface 在这附近),梯度可以顺着偏移关系**传回 base mesh**,把 base mesh 拉到正确位置。

---

## 数学上怎么实现?三个公式讲透

### 公式 (1): 怎么生成多层

$$\mathbf{v}_j^i = \mathbf{v}_j^0 + d_j^i \cdot \mathbf{n}_j$$

- $\mathbf{v}_j^0$: base mesh 上第 j 个顶点
- $\mathbf{n}_j$: 这个顶点的 normal 方向
- $d_j^i$: 第 i 层的偏移量(正负各几个值,范围 ±10cm)
- $\mathbf{v}_j^i$: 第 i 层上对应的顶点

意思就是:base mesh 上每个顶点,沿 normal 方向复制 5 份,有的往外凸、有的往内凹,形成 5 层 mesh。这 5 层 mesh 的顶点都**由 base mesh 决定**,所以是 differentiable 的。

### 公式 (2): stop-gradient trick(最关键的一个)

$$s_j^i = \text{sign}(d_j^i) \left\| \text{stop}(\mathbf{v}_j^i) - \mathbf{v}_j^0 \right\|_2$$

- $s_j^i$: 第 i 层第 j 个顶点到 base mesh 的 signed distance
- $\text{stop}(\cdot)$: stop-gradient operator(forward 时 identity,backward 时梯度不通过这个位置回传)
- $\text{sign}(d_j^i)$: 偏移方向(正/负)

**为什么需要 stop-gradient?**

如果不要 stop,直接把公式 (1) 代入:

$$s_j^i = \text{sign}(d_j^i) \cdot \| d_j^i \cdot \mathbf{n}_j \| = \text{sign}(d_j^i) \cdot |d_j^i| \cdot 1 = d_j^i$$

因为 $\mathbf{n}_j$ 是 unit normal,模长 1。结果 $s_j^i = d_j^i$,**完全跟 $\mathbf{v}_j^0$ 无关**。梯度 $\partial s_j^i / \partial \mathbf{v}_j^0 = 0$,base mesh 就无法被更新了。

加了 stop-gradient 后,forward 时 $\text{stop}(\mathbf{v}_j^i) = \mathbf{v}_j^i$,数值不变;backward 时 $\text{stop}(\mathbf{v}_j^i)$ 被当成常数,**只有 $\mathbf{v}_j^0$ 作为减数项在表达式中出现,梯度可以正常回传**:

$$\frac{\partial s_j^i}{\partial \mathbf{v}_j^0} = -\text{sign}(d_j^i) \cdot \frac{\mathbf{v}_j^0 - \mathbf{v}_j^i}{\|\mathbf{v}_j^i - \mathbf{v}_j^0\|_2}$$

直觉:stop-gradient 把 "偏移量 $d_j^i$ 是固定的" 这个事实硬编码进计算图,强迫梯度只能通过 "base mesh 位置 $\mathbf{v}_j^0$" 这一条路回来。这是一个非常 elegant 的 design pattern——**用 stop-gradient 控制梯度流向**,类似 Neural Radiance Caching 里用 stop-gradient 把 gradient 分配到不同 path。

### 公式 (3): signed distance → alpha

$$\alpha = \begin{cases} \frac{1}{\beta}(1 - \frac{1}{2}e^{s/\beta}), & s < 0 \\ \frac{1}{2\beta}e^{-s/\beta}, & s \geq 0 \end{cases}$$

- $s$: signed distance(负 = base mesh 内部,正 = 外部)
- $\beta$: learnable 参数,控制 "棉花糖的厚度"

直觉:
- $s = 0$ 时 $\alpha = 1/(2\beta)$,最大值
- $|s|$ 越大,$\alpha$ 越小,指数衰减
- $\beta$ 小 → band 薄,表面锐利
- $\beta$ 大 → band 厚,receptive field 大但表面模糊

这个 mapping 直接来自 VolSDF,但 VolSDF 里 $s$ 是网络预测的 SDF,这里 $s$ 是**几何上算出来的对 base mesh 的距离**。这是一个很好的 "reuse"——把 volumetric rendering 的成熟工具嫁接到 mesh 上。

参考 VolSDF: https://arxiv.org/abs/2106.12060

---

## 渲染怎么做?Mesh Splatting

### 核心思路

把 mesh triangle 当作 splatting primitive(类似 3DGS 把 Gaussian 当 primitive)。

对每个 pixel,找出所有覆盖它的 triangles,按 depth 排序,然后做标准 alpha compositing:

$$\mathbf{C}_p = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha_i \prod_{k=1}^{i-1}(1 - \alpha_k)$$

- $\mathcal{N}$: near-to-far 排序的 triangles
- $\mathbf{c}_i$: 第 i 个 triangle 在 pixel 处的 color
- $\alpha_i$: 第 i 个 triangle 的 opacity(来自公式 3)
- $\prod(1-\alpha_k)$: 前面所有 triangles 的累积透射率

这就是 NeRF/3DGS 那套,**只是 primitive 换成 mesh triangle**。

### Color 怎么算?

每个 triangle 三个顶点上存 feature,用 barycentric coordinate 插值到 pixel 处,再过一个小 MLP:

$$\mathbf{c}_i = \text{MLP}(\mathbf{f}_i, \mathbf{n}_i, \mathbf{r}_i, \text{Hash}(\mathbf{x}_i))$$

- $\mathbf{f}_i$: 插值后的 appearance feature
- $\mathbf{n}_i$: 插值后的 normal
- $\mathbf{r}_i$: view direction
- $\text{Hash}(\mathbf{x}_i)$: 多分辨率 hash encoding [Müller et al. 2022]

**为什么要 hash encoding?** 因为 mesh 顶点稀疏,直接 piecewise linear 插值 feature 会有 triangle 内的 over-smoothing artifact。Hash encoding 注入 non-linearity,等于在 mesh 表面贴了一个 implicit texture field,可以表达高频细节。

---

## Topology 控制:DMTet + Continuous Remeshing 的混合策略

### 为什么直接优化 mesh 不行?

Mesh optimization 容易出现 topological defects: self-intersection、holes、degenerate triangles。一旦出现就很难修复。

### 阶段 1: DMTet(前 5000 iterations)

DMTet 在 tetrahedral grid 上存 SDF,用 Marching Tetrahedra 提取 mesh。SDF 初始化为 sphere:

$$\text{SDF}(\mathbf{x}_g) = \|\mathbf{x}_g\|_2 - r$$

- $\mathbf{x}_g$: grid vertex
- $r$: 初始 sphere 半径

这阶段更新 grid 上的 SDF。SDF 符号变化自然表达拓扑变化,可以 robust 地收敛到正确 topology(比如有 holes 的物体)。

**注意**:这里的 SDF 和公式 (2) 的 SDF 是**两个不同的东西**。DMTet 的 SDF 是用来稳定拓扑的,公式 (2) 的 SDF 是用来生成 softened layer alpha 的。

### 阶段 2: Continuous Remeshing(后 10000 iterations)

DMTet resolution cubic scaling,提到 256 已经 8GB 内存(Table 4),不够用。所以 DMTet 收敛后就**冻结**,切换到直接优化 mesh + 每步 remeshing。

Continuous Remeshing 每步:
1. 优化 vertex positions 和 attributes
2. 做 isotropic remeshing(目标 minimum edge length ~5mm)
3. 维持 near-isotropic triangles,减少 degenerate elements

### 消融实验数据(Table 4)

| 配置 | Vertices | CD |
|---|---|---|
| w/o DMTet | 80K | 3.79 |
| DMTet (128) | 2K | 6.94 |
| DMTet (256) | 10K | 4.20 |
| Full Model | 306K | 1.57 |

- **w/o DMTet**: 拓扑都收敛不对,holes 都没有,CD 3.79
- **DMTet (128)**: 太稀疏,细节全丢,CD 6.94
- **DMTet (256)**: 还是稀疏,CD 4.20
- **Full Model**: DMTet 稳拓扑 + remeshing 加细节,CD 1.57

**insight**: DMTet 和 Continuous Remeshing 解决的是两个**不同 scale 的问题**。DMTet 解决 global topology(整个物体的连通性、handles、holes),Continuous Remeshing 解决 local element quality(三角形形状、density)。两者缺一不可。

---

## 实验数据的关键解读

### DTU(Table 1)

| Method | Mean CD | Verts (K) | Training (min) |
|---|---|---|---|
| Neuralangelo | 0.62 | 1000 | 600 |
| IMLS-Splatting | 0.57 | 300 | 11 |
| Ours | 0.62 | 300 | 23 |
| Ours w/o MS | 0.73 | 300 | 20 |

几个关键 takeaways:

1. **Ours 与 Neuralangelo 并列**,但 vertices 少 3.3x,training 快 26x。这是 mesh 方法相对 volumetric 的实用优势。

2. **比 IMLS-Splatting 略差**(0.62 vs 0.57)。IMLS 用 cubic grid 参数化,3D representation 更灵活。但 ablation "Ours w/o MS"(去掉 mesh softening,只保留 shading supervision)CD 0.73,**比 full model 差 0.11**。这 0.11 就是 mesh softening 带来的 volumetric supervision 的增益。

3. **SuGaR 1.33**,证明 single-layer Gaussian-mesh proxy 不行,因为它只有单层 receptive field。

### BlendedMVS(Table 2)

| Method | Mean CD |
|---|---|
| NeuS | 2.68 |
| IMLS-Splatting | 2.75 |
| Ours w/o MS | 1.94 |
| **Ours** | **1.71** |

BMVS 有更复杂几何(thin structures、indentations),这恰好是 mesh softening 的强项。**Ours 明显领先**,因为 soft band 在细节附近能形成有效 gradient field。

Fig.5 的 Stone 例子最直观:Stone 顶部有个 indentation,**只有 full model 能恢复**。"Ours w/o MS" 即使有更 dense mesh 也丢了 indentation,证明 shading supervision 在细节恢复上根本不够,必须靠 volumetric supervision。

### 渲染效率(Table 3)

DTU scan122, full resolution (1600×1200):

| Method | Memory (GB) | Training (min) |
|---|---|---|
| GS | 6 | 13 |
| IMR (iterative mesh rasterization) | **OOM** | N/A |
| MS (ours) | 13 | 22 |

- **IMR 直接 OOM**:5 层 mesh 用 depth peeling 渲染,每层一次 rasterization pass,5 个 framebuffer,32GB V100 都扛不住。
- **MS 通过 tile-based splatting 一次 compositing 所有 layers**,内存效率 2-4x 提升。
- **GS 仍比 MS 快**:说明 MS 实现还有工程优化空间(triangle culling、adaptive density)。

### 顶点数可控性

| 配置 | Vertices | CD |
|---|---|---|
| Sparse Mesh | 127K | 1.66 |
| Dense Mesh | 487K | 1.67 |
| Full Model | 306K | 1.57 |

127K 到 487K 顶点 CD 几乎不变(1.66 vs 1.67),**方法对顶点数 robust**。这是 mesh 方法相对 volumetric 的巨大实用优势——可以根据下游应用 budget 调整顶点数。

---

## 整个方法为什么 work?直觉总结

回到 Fig.2 的思想实验。Base mesh 漂在真实 surface 上方 $\delta = 5$cm。

### Regular mesh 的情况

点 A 在 base mesh 上,在所有 views 中都能被看到。优化器只能学 A 的 color——它**没有任何信号告诉它 A 应该往下移 5cm**。结果 mesh 漂在空中,color 伪装成 surface。这就是 "single-layer receptive field" 的死穴。

### Soft mesh 的情况

现在点 A 上下散开 5 层,每层 alpha 按 signed distance 衰减。同时点 B(真实 surface 上对应 A 的位置)在某个 soft layer 上也有非零 alpha。

Multi-view 渲染时:
- **点 A**:在所有 views 中位置一致,color 可以 fit 任何 image(像 regular mesh 一样)
- **点 B**:在不同 views 中位置不同(因为 ray-triangle intersection 随视角变化),如果 B 处的颜色与 GT 不符,会产生 photometric loss

关键:**点 A 和点 B 都在 base mesh 附近的 soft band 内**,都参与 compositing。如果 base mesh 应该往下移到 B 处:
1. 当前 base mesh 上 A 偏离 B,B 周围的 layer alpha 不为 0,贡献的 color 与 GT 偏差大
2. 梯度通过 $\partial \alpha / \partial s \cdot \partial s / \partial \mathbf{v}^0$(stop-gradient trick 让这条路通)传回 $\mathbf{v}^0$
3. 优化让 $\mathbf{v}^0$ 朝 B 移动,base mesh 下沉到真实 surface

这就是 volumetric 的 "large receptive field" 在 mesh 上的对应物。**Soft band 是 mesh 表面的"梯度可达区域",区域内任何点对 photometric loss 有贡献,从而 pull base mesh 到正确位置**。

$\beta$ 控制 band 厚度:$\beta$ 小则 band 薄、receptive field 不足;$\beta$ 大则 band 厚、失去 surface 锐度。论文里 $\beta$ 是 learnable,自动学到合适值。

---

## Shading supervision 是辅助(Appendix C)

除了 volumetric photometric loss,还保留了 IMLS-Splatting 风格的 shading supervision。通过 Nvdiffrast rasterize base mesh,得到 mask $\mathbf{I}_m$、normal map $\mathbf{I}_n$、feature map $\mathbf{I}_f$。MLP 把 feature 解码成 diffuse color $\mathbf{c}_d$、specular tint $\mathbf{s}$、specular feature $\mathbf{f}_s$:

$$\mathbf{c} = \mathbf{c}_d + \mathbf{s} \odot \Phi_s(\mathbf{f}_s, \omega, \omega_r)$$

- $\omega$: view direction
- $\omega_r = 2(-\omega \cdot \mathbf{n})\mathbf{n} + \omega$: reflection direction
- $\Phi_s$: 轻量 MLP,预测 specular color

这是简化的 split-sum PBR。**volumetric compositing loss 提供 geometric supervision,shading loss 提供 material/light supervision**。两者共同作用,前者靠 mesh softening,后者靠 base mesh 直接 rasterize。

---

## 局限性

### Scalability

DMTet 的 tetrahedral grid 在大场景下分辨率不够(128 resolution 只覆盖 2.5m bounding box)。所以 paper 主要在 object-centric 上做。Fig.7 的 scene-level 实验是先用 GaussianSurfel 粗 mesh 再 refine,本质上是**两阶段而非 end-to-end**。

### 远距离 base mesh failure

如果 base mesh 距真实 surface 太远(超过 ±10cm offset 范围),soft band 无法 overlap 真实 surface,**梯度为零**。这是任何 "soft" 方法的根本限制——band 必须能 "够到" 真实表面。

解决方案:**adaptive bandwidth**(根据当前 residual photometric error 动态扩大 band),或者 **coarse-to-fine schedule**(先大 $\beta$ 全局定位,再小 $\beta$ 细化)。

### 极薄结构

Fig.8 显示 ship 的 cable 都没重建出来。原因:isotropic remeshing 倾向生成等边三角形,对 cable 这种 1D 结构不友好。需要 **anisotropic remeshing**——在 cable 方向上拉长 triangle。

---

## 与相关工作对比

### vs. Gaussian Shell Maps / DELIFFAS / AdaptiveShell / Gaussian Frosting

这些工作也在 base mesh 周围放 transparent layers,但 **layers 与 base mesh 之间没有 differentiability**。它们的目标是 novel view synthesis(粗几何 + layered Gaussians 模拟 light field),base mesh 是 fixed 的(SMPL 之类预先定义)。

本 paper 的关键创新:**layers 是 base mesh 的 differentiable function,通过 stop-gradient trick 让 alpha 反传到 base mesh**,使 base mesh 可以被优化。这是 end-to-end surface reconstruction 而非 view synthesis 的前提。

### vs. IMLS-Splatting

IMLS-Splatting 是最接近的 baseline。它把点云转 grid,提取 mesh,然后用 shading-based supervision 优化。它用 cubic grid 作为 3D representation 但仍是 **single-layer surface optimization**。

"Ours w/o MS" ablation 就是只在 mesh-based pipeline 上做 shading supervision,CD 0.73 vs full model 0.62。**这 0.11 差距来自 mesh softening 提供的 volumetric receptive field**。如果本 paper 的 softening 配合 IMLS 的 cubic grid 参数化,可能继续提精度。

### vs. NvdiffRec

NvdiffRec 也是 mesh + DMTet 早期,但只做 shading supervision。本 paper 可以看作 **NvdiffRec 的 "volumetric supervision 升级版"**。

---

## 为什么这篇 paper 重要?Insight 层面

### 1. 重新定义问题

这篇 paper 精准定义了 mesh-based 和 volumetric 方法之间的二分问题,然后用 elegant 的方式 merge。这种 **"重新参数化 + stop-gradient 让梯度走特定路径"** 的思路与 NeuS 的 SDF density mapping 有精神上的相通——都是用 forward 数值正确但 backward 梯度被引导到期望方向的设计。

### 2. Design pattern: stop-gradient as differentiability control

公式 (2) 的 stop-gradient 是一个值得记住的 design pattern。**当你有一个表达式,某些路径的梯度是 trivial 的(会 collapse),用 stop-gradient 切断这些路径,强迫梯度走你想要的路径**。这在 implicit differentiation、gradient checkpointing、RL 里的 reward shaping 都有类似思想。

### 3. Mesh 的实用优势终于被保留

300K vertices + 23min training + SOTA 精度,这个 sweet spot 直接可用。Volumetric 方法经常给 1000K vertices,实际游戏/AR/VR 应用用不了这么多。Mesh 方法可以控制顶点数,但精度上不去。本 paper 拿到了两边的优点。

### 4. Hybrid topology control 的工程价值

DMTet 解决全局拓扑,Continuous Remeshing 解决局部 element quality。这个 hybrid 思路值得记住——**不同 scale 的问题用不同工具**,DMTet 的 cubic scaling 不适合高分辨率,remeshing 无法改变全局拓扑。两者分工明确。

---

## 个人直觉构建总结

回到 Karpathy 的 intuition 层面,这篇 paper 教给我几件事:

1. **Gradient flow 是 differentiable rendering 的核心**。不是 "forward 能算就行",而是 "backward 梯度能不能到达你想优化的参数"。公式 (2) 的 stop-gradient 是这个原则的极致体现——forward 时它什么都没做,backward 时它决定了梯度能不能 reach base mesh。

2. **Receptive field 不只是 CNN 的概念,differentiable rendering 也有**。Single-layer mesh 的 receptive field 太小,梯度信号弱。Volumetric 的 receptive field 大但 meshing 有损。Soft mesh 是两者的 interpolation。

3. **Mesh 的 structural advantages 是 real**。Remeshing、顶点数控制、topology control,这些都是 volumetric 做不到的。如果你能用 mesh 就别用 volumetric field,但要解决 mesh 的 gradient 问题。

4. **Priors 是 noisy 的,直接 image supervision 才是 ground truth signal**。Shading、normal、depth 这些 priors 都有误差,只有 photometric loss 是直接的。Soft mesh 让 mesh 能直接吃 photometric loss,这是 information efficiency 的提升。

5. **"Pseudo-volumetric" 是一个 useful design pattern**。不是真的 volume(不需要 3D grid),但行为像 volume(沿 ray 有多个 alpha splat)。这种 "假装是 volumetric" 的思路在效率上有优势,在 gradient 上有 volumetric 的好处。

参考链接:
- VolSDF: https://arxiv.org/abs/2106.12060
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- NvdiffRec: https://nvlabs.github.io/nvdiffrec/
- IMLS-Splatting: https://imls-splatting.github.io/
- DMTet: https://research.nvidia.com/labs/toronto-ai/dmtet/
- Continuous Remeshing: https://onlinelibrary.wiley.com/doi/10.1002/cav.2101
- Gaussian Shell Maps: https://research.nvidia.com/labs/dir-labs/gaussian-shell-maps/
- AdaptiveShell: https://arxiv.org/abs/2311.10091
- NeuS: https://arxiv.org/abs/2106.10689
- Neuralangelo: https://research.nvidia.com/labs/dir-labs/neuralangelo/

希望这个 "人话版" 帮你 build up 了对方法的 intuition。核心记住三件事:**soft band 提供 receptive field、stop-gradient 控制 gradient flow、DMTet+remeshing 的 hybrid topology control**。剩下的都是 supporting details。

---

# Mesh Splatting 论文深度讲解

## 1. 论文核心动机:两范式的痛点

这篇 paper 处理的是 multiview surface reconstruction 中一个长期存在的二分问题。整个 field 大致分为两类方法,各有缺陷:

**Volumetric methods**(NeuS [1], Neuralangelo [2], 2DGS [3], GOF [4], GaussianSurfel [5])占据整个 3D space,density 或 opacity 在整个 volume 中都有定义。沿着 ray 的每个点都有响应,这就是所谓的 "large effective receptive field along rays"。这种 large receptive field 让 volumetric rendering(eq.6 那种 alpha compositing)的梯度可以从远距离pull 几何到正确位置。但致命问题是最后必须 meshing(Marching Cubes [6], Marching Tetrahedra [7], Poisson Reconstruction [8])才能拿到 mesh,meshing 步骤会:
- 产生 overly dense meshes(DTU 上动辄 1000K vertices)
- 累积误差(Fig.1 中红圈显示 mesh 和 ground-truth point cloud 的 misalignment)
- 实际 upper bound 被 meshing 卡住(Fig.1e 蓝圈显示 Poisson 甚至漏点)

**Mesh-based methods**(NvdiffRec [9], IMLS-Splatting [10], SuGaR [11])直接优化 mesh,避免 meshing。但 mesh 是 opaque single-layer surface,只有 boundary geometry 的响应。Fig.2a 精确说明了问题:当 base mesh 不与真实 surface overlap 时,multi-view 只能优化 color(点A的 appearance),给不到 spatial gradient 去移动 geometry。所以 mesh-based 方法不得不依赖 priors——shading、monocular normals、depth estimation——而这些都是 noisy 或者违反假设的。

**Insight**:作者要 bridge 这两个世界,把 mesh "软化"成 pseudo-volumetric representation,既保留 mesh 的 topology control 优势(可 remeshing),又获得 volumetric 的 3D receptive field 和直接 image supervision。

参考链接:
- NeuS: https://arxiv.org/abs/2106.10689
- Neuralangelo: https://research.nvidia.com/labs/dir-labs/neuralangelo/
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- NvdiffRec: https://research.nvidia.com/labs/toronto-ai/nvdiffrec/

---

## 2. Soft Mesh 的核心数学设计

### 2.1 偏移生成多层

第 i 层 mesh 上的第 j 个 vertex 由 base layer $\mathcal{M}_0$ 偏移得到:

$$\mathbf{v}_j^i = \mathbf{v}_j^0 + d_j^i \cdot \mathbf{n}_j \quad (1)$$

- 上标 $i \in \{1, ..., N\}$:layer index(论文中 N=5)
- 下标 $j$:vertex index
- $\mathbf{v}_j^0$:base mesh 上第 j 个 vertex
- $d_j^i$:第 j 个 vertex 在第 i 层的 offset distance(限制在 ±10 cm)
- $\mathbf{n}_j$:vertex $\mathbf{v}_j^0$ 处的 unit normal

几何直觉:5 层 mesh 在 normal 方向上散开,形成一个 "soft band" 包裹 base mesh,从外到内 alpha 渐变。

### 2.2 SDF 到 alpha 的关键映射(这是 paper 最精妙的部分)

第 i 层第 j 个 vertex 到 base mesh 的 signed distance:

$$s_j^i = \text{sign}(d_j^i) \left\| \text{stop}(\mathbf{v}_j^i) - \mathbf{v}_j^0 \right\|_2 \quad (2)$$

这里 $\text{stop}(\cdot)$ 是 stop-gradient operator,**这个 stop-gradient 是整个方法的数学关键**。

**为什么 stop-gradient 必须存在?**

假设没有 stop-gradient,直接把 eq.(1) 代入 eq.(2):

$$s_j^i = \text{sign}(d_j^i) \left\| d_j^i \cdot \mathbf{n}_j \right\|_2 = \text{sign}(d_j^i) \cdot |d_j^i| \cdot \|\mathbf{n}_j\|_2 = d_j^i$$

因为 $\mathbf{n}_j$ 是 unit vector 所以 $\|\mathbf{n}_j\|_2 = 1$。结果 $s_j^i = d_j^i$,完全 independent of $\mathbf{v}_j^0$。这样梯度回传时 $\partial s_j^i / \partial \mathbf{v}_j^0 = 0$,base mesh 完全无法被更新。

加 stop-gradient 后,$\text{stop}(\mathbf{v}_j^i)$ 在反向传播时被当作常数,$\mathbf{v}_j^0$ 仍然在表达式中作为减数项出现,梯度变成:

$$\frac{\partial s_j^i}{\partial \mathbf{v}_j^0} = -\text{sign}(d_j^i) \cdot \frac{\mathbf{v}_j^0 - \text{stop}(\mathbf{v}_j^i)}{\|\text{stop}(\mathbf{v}_j^i) - \mathbf{v}_j^0\|_2}$$

forward 时 $\text{stop}$ 是 identity,所以 $s_j^i$ 的数值不变,但 backward 时梯度路径被切断,只剩下从 $\mathbf{v}_j^0$ 作为减数项贡献的梯度。这是 implicit differentiation 的一个 trick——用 stop-gradient 把"已经被定义的关系"和"需要被优化的参数"在反向图上拆开。

### 2.3 VolSDF alpha mapping

把 signed distance 转换成 alpha:

$$\alpha = \begin{cases} \frac{1}{\beta}(1 - \frac{1}{2}e^{s/\beta}), & s < 0 \\ \frac{1}{2\beta}e^{-s/\beta}, & s \geq 0 \end{cases} \quad (3)$$

- $s$:signed distance(负值表示在 base mesh 内部,正值表示外部)
- $\beta > 0$:learnable 参数,控制 density 围绕 base mesh 的集中程度

直觉:当 $s=0$ 时 $\alpha = 1/(2\beta)$,当 $|s| \to \infty$ 时 $\alpha \to 0$。这就形成了 base mesh 周围一个指数衰减的 "density cloud"。$\beta$ 小则 cloud 集中在表面附近,类似 thin shell;$\beta$ 大则 cloud 弥散,类似 thick volume。这个 mapping 直接来自 VolSDF [12],但在 VolSDF 中 SDF 是从网络预测的,这里 SDF 是相对于 base mesh 的几何量。

---

## 3. Differentiable Mesh Splatting

### 3.1 渲染管线

核心是 Fig.3 的 pipeline。给定 softened mesh,用 tile-based rasterization 渲染。对每个覆盖 pixel $p$ 的 triangle $\{\mathbf{v}_1^i, \mathbf{v}_2^i, \mathbf{v}_3^i\}$,计算 ray-triangle intersection $\mathbf{x}_i$ 和 **perspective-corrected barycentric coordinates**:

$$\mathbf{w}_i = \text{correct}(\mathbf{p}, \{\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3\}, \{\mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3\}) \quad (4)$$

- $\mathbf{p}$:pixel 坐标
- $\mathbf{u}_k$:vertex $k$ 在 image plane 上的 2D 投影
- $z_k$:vertex $k$ 的 depth(view space z)
- $\mathbf{w}_i = \{w_1, w_2, w_3\}$:corrected barycentric weights(标准透视校正: $w_k \propto w_k^{screen} / z_k$,归一化)

Barycentric 插值得到 per-intersection 属性 $\{\alpha_i, \mathbf{f}_i, \mathbf{n}_i, \mathbf{r}_i, \mathbf{x}_i\}$。然后 color 通过一个小 MLP 预测:

$$\mathbf{c}_i = \text{MLP}(\mathbf{f}_i, \mathbf{n}_i, \mathbf{r}_i, \text{Hash}(\mathbf{x}_i)) \quad (5)$$

- $\mathbf{f}_i$:appearance feature(从 vertex 插值)
- $\mathbf{n}_i$:normal
- $\mathbf{r}_i$:view direction(from camera center to intersection)
- $\text{Hash}(\mathbf{x}_i)$:multi-resolution hash encoding [13] of position $\mathbf{x}_i$,用来注入非线性避免三角形内 over-smooth 插值

这是个有意思的设计——为什么 mesh 顶点上要存 feature 然后过 MLP?因为 mesh 顶点稀疏,直接存 RGB 会有 piecewise-linear artifact。Hash encoding 让任意 3D 位置都能查到 non-linear feature,等于在 mesh 表面贴了一个 implicit texture field。

### 3.2 Volumetric compositing

最后所有覆盖 pixel $p$ 的 triangles 按 depth 排序后做 standard volumetric rendering:

$$\mathbf{C}_p = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha_i \prod_{k=1}^{i-1}(1-\alpha_k) \quad (6)$$

- $\mathcal{N}$:near-to-far sorted triangles
- $\alpha_i$:triangle $i$ 的 opacity(从 eq.3 得到)
- $\prod_{k=1}^{i-1}(1-\alpha_k)$:transmittance,前面所有 triangles 的累积透射

这个公式就是 3DGS / NeRF 的标准 alpha compositing。区别在于 primitive 是 mesh triangle 而非 Gaussian ellipsoid 或 sampled point。每个 triangle 自带 alpha(基于它的 layer 的 signed distance),所以多层 mesh 在 ray 方向上形成了 5 个 alpha "splats",compositing 后获得 smooth gradient。

直觉上,这套设计把 mesh "分身"成5个半透明幻影,每个幻影有不同的位置和不同的 alpha,组合起来模拟了一个连续的 density field。当 base mesh 偏离真实表面时,某些 layer 的 alpha 高,某些低,compositing 会偏好与 GT image 接近的 layer,梯度通过 stop-gradient trick 传回 base mesh。

---

## 4. Hybrid Topology Control

### 4.1 DMTet 阶段(早期)

DMTet(Deep Marching Tetrahedra) [7] 在 tetrahedral grid 上存 SDF,用 Marching Tetrahedra 提取 mesh。grid vertex 初始化为:

$$\text{SDF}(\mathbf{x}_g) = \|\mathbf{x}_g - \mathbf{0}\|_2 - r$$

- $\mathbf{x}_g$:tetrahedral grid vertex
- $r$:初始 sphere 半径
- $\mathbf{0}$:中心

这就初始化了一个半径为 r 的 sphere。注意:**这里的 SDF 与 eq.(2) 中的 SDF 是两个完全不同的东西**。DMTet 的 SDF 是用来稳定早期拓扑的(避免 mesh 出现 hole、self-intersection),eq.(2) 的 SDF 是用来生成 softened layer 的 alpha 的。

训练前 5000 iterations 用 DMTet,这阶段更新 tetrahedral grid 上的 SDF。

### 4.2 Continuous Remeshing 阶段(后期)

DMTet 后冻结,切换到 Continuous Remeshing [14]。这阶段直接优化 base mesh 的 vertex positions 和 attributes,每一步后做 remeshing 维持 near-isotropic triangles(目标最小边长 ~5 mm)。

为什么不能一直用 DMTet?因为 DMTet 的 resolution $R$ 是 cubic scaling($O(R^3)$),从 128 到 256 顶点数从 2K 跳到 10K,内存从 6GB 到 8GB,training time 从 15min 到 23min(Table 4)。而 mesh optimization 可以线性增加 vertex density,可控性强得多。

为什么需要 DMTet 启动?因为 mesh optimization 从一个不合适的初始拓扑(比如 sphere 与目标 holes 不匹配)很难收敛到正确拓扑。DMTet 通过 SDF 的符号变化可以自然地表达拓扑变化(零等值面的连通性变化)。

Table 4 显示:
- "w/o DMTet": CD 3.79(无法收敛到正确拓扑)
- "DMTet (128)": CD 6.94(过稀疏)
- "DMTet (256)": CD 4.20(细节仍不足)
- "Full Model": CD 1.57(最优)

这就是 hybrid 的核心收益:DMTet 保证全局拓扑正确,Continuous Remeshing 保证局部细节和 element 质量。

---

## 5. 实验结果分析

### 5.1 DTU 量化结果(Table 1)

| Method | Mean CD | Verts (K) | Training (min) |
|---|---|---|---|
| NeuS | 0.76 | 1000 | 600 |
| NeuS2 | 0.70 | 1000 | 3 |
| Neuralangelo | 0.62 | 1000 | 600 |
| 2DGS | 0.78 | 300 | 9 |
| GOF | 0.74 | 1000 | 18 |
| SuGaR | 1.33 | 1000 | 52 |
| IMLS-Splatting | **0.57** | 300 | 11 |
| Ours w/o MS | 0.73 | 300 | 20 |
| **Ours** | 0.62 | 300 | 23 |

**关键观察**:
1. Ours 与 Neuralangelo 并列第一(0.62),但 vertices 少 3.3 倍,training time 快 26 倍
2. 比 IMLS-Splatting 略差(0.62 vs 0.57),但 IMLS 用 cubic grid 在 3D representation 上更灵活;ablation "Ours w/o MS"(只保留 shading supervision,去掉 mesh softening)CD 跌到 0.73,证明 softening 带来的 volumetric supervision 比 shading-only 强
3. SuGaR 的 1.33 证明 single-layer Gaussian-mesh proxy 不行

### 5.2 BlendedMVS 量化结果(Table 2)

| Method | Mean CD |
|---|---|
| NeuS | 2.68 |
| Surfels | 2.46 |
| SuGaR | 8.71 |
| IMLS-Splatting | 2.75 |
| Ours w/o MS | 1.94 |
| **Ours** | **1.71** |

Ours 在 BMVS 上明显领先。BMVS 有更复杂几何(细薄结构、indentation),这恰好是 mesh softening 的强项——softened layers 在细节附近形成有效 gradient field。Fig.5 显示 Stone 顶部的 indentation 只有 full model 能恢复,"Ours w/o MS" 即使有更 dense mesh 也丢失了 indentation,直接证明 softening 提供的 3D receptive field 是 shading 提供不了的。

### 5.3 渲染效率(Table 3)

DTU scan122,full resolution (1600×1200):

| Method | Memory (GB) | Training (min) |
|---|---|---|
| GS (GaussianSurfel impl) | 6 | 13 |
| IMR (Iterative Mesh Rasterization w/ depth peeling) | **OOM** | N/A |
| MS (Ours) | 13 | 22 |

IMR 用 Nvdiffrast 的 depth peeling 渲染 5 层 mesh,full resolution 直接 OOM——5 层意味着 5 次 rasterization pass,每次都要存储 framebuffer。MS 通过 tile-based splatting 一次 compositing 所有 layers,内存效率 2-4 倍提升。

GS 仍比 MS 快——暗示 MS 实现还有工程优化空间(invisible triangle culling、adaptive vertex density)。

### 5.4 顶点数可控性(Table 4)

通过 Continuous Remeshing 的 minimum edge length 参数:

| 配置 | Vertices | CD |
|---|---|---|
| Sparse Mesh | 127K | 1.66 |
| Dense Mesh | 487K | 1.67 |
| Full Model | 306K | 1.57 |

顶点数从 127K 到 487K CD 几乎不变(1.66 vs 1.67),证明方法对顶点数 robust,可根据下游应用 budget 调整。这是 mesh-based 方法相对于 volumetric(只能给 dense mesh)的巨大实用优势。

---

## 6. 与相关工作的精细对比

### 6.1 vs. Gaussian Shell Maps / DELIFFAS / AdaptiveShell / Gaussian Frosting

这些工作也在 base mesh 周围放 transparent layers [15, 16, 17, 18],但**layers 与 base mesh 之间没有 differentiability**。它们的目标是 novel view synthesis(用粗几何 + layered Gaussians 模拟 light field),base mesh 是 fixed 的(SMPL [19] 之类预先定义的)。

本 paper 的关键创新:layers 是 base mesh 的 differentiable function(eq.1),并且通过 stop-gradient trick(eq.2)让 alpha 反传到 base mesh,使 base mesh 可以被优化。这是 end-to-end surface reconstruction 而非 view synthesis 的前提。

参考:
- GSM: https://research.nvidia.com/labs/dir-labs/gaussian-shell-maps/
- AdaptiveShell: https://arxiv.org/abs/2311.10091
- Gaussian Frosting: https://antoinelc.github.io/gaussian-frosting-website/

### 6.2 vs. Quadrature Fields / Volumetric Surfaces

Quadrature Fields [20] 和 Volumetric Surfaces [21] 也构造 layered 结构近似 SDF sampling,但同样 base mesh 是预提取的、不可优化的。它们的 layered structure 是渲染加速器,不是几何优化目标。

### 6.3 vs. IMLS-Splatting

IMLS-Splatting [10] 是最接近的 baseline。它把点云转成 grid,提取 mesh,然后用 shading-based supervision 优化 mesh。它用 cubic grid 作为 3D representation 但仍是 single-layer surface optimization。

本 paper 的 "Ours w/o MS" ablation 就是只在 mesh-based pipeline 上做 shading supervision,结果 CD 0.73 vs full model 0.62,差距来自 mesh softening 提供的 volumetric receptive field。如果本 paper 的 softening 配合 IMLS 的 cubic grid 参数化,可能继续提精度。

IMLS-Splatting: https://imls-splatting.github.io/

### 6.4 vs. NvdiffRec

NvdiffRec [9] 也是 mesh + DMTet 早期,但只做 shading supervision,几何细节难以恢复。本 paper 可以看作 NvdiffRec 的 "volumetric supervision 升级版"。

NvdiffRec: https://nvlabs.github.io/nvdiffrec/

---

## 7. 直觉构建:为什么 softening 能 work?

考虑 Fig.2 的场景。Base mesh 在真实 surface 上方,远离真实 surface 一段距离 $\delta$。

**Regular mesh**:multi-view 看到 base mesh 上的点 A(虚空中)。点 A 在所有 views 中的 appearance 是 mesh 自己学的 color,可以 fit 任何 image。没有梯度信号告诉 base mesh 应该往下移。结果是 mesh 浮在空中,color 伪装成 surface。

**Soft mesh**:在点 A 上下散开 5 层,每层 alpha 按 eq.(3) 衰减。点 A 自己(中心层)alpha 最大,上下两层 alpha 衰减。同时点 B(在真实 surface 上方但被 base mesh 偏离后"漏掉"的位置)出现在某个 soft layer 上。

Multi-view 渲染时:
- 点 A 在所有 views 中位置一致(color 假装正确),贡献稳定 color
- 点 B 在不同 views 中位置不同(因为 ray-triangle intersection 随视角变),如果 B 处的颜色与 GT 不符,会产生 photometric loss

**关键**:点 A 和点 B 都在 base mesh 附近的 "soft band" 内,都参与 compositing。如果 base mesh 应该往下移到 B 处,那么:
1. 当前 base mesh 上 A 偏离 B,在 B 周围的 layer 上 alpha 不为 0,贡献的 color 与 GT 偏差大
2. 梯度通过 $\partial \alpha / \partial s \cdot \partial s / \partial \mathbf{v}^0$(eq.2 的 stop-gradient trick)传回 $\mathbf{v}^0$
3. 优化让 $\mathbf{v}^0$ 朝 B 移动,base mesh 下沉到真实 surface

这就是 volumetric 的 "large receptive field" 在 mesh 上的对应物。soft band 是 mesh 表面的"梯度可达区域",区域内任何点对 photometric loss 有贡献,从而 pull base mesh 到正确位置。

**$\beta$ 的作用**:$\beta$ 决定 soft band 厚度。$\beta$ 太小,band 太薄,receptive field 不足;$\beta$ 太大,band 太厚,alpha 衰减慢,失去 surface 锐度。论文中 $\beta$ 是 learnable,自动学到合适值。

---

## 8. Shading supervision 细节(Appendix C)

除了 volumetric rendering 的 photometric loss,还保留了 IMLS-Splatting 风格的 shading supervision:

通过 Nvdiffrast 把 base mesh rasterize,得到 foreground mask $\mathbf{I}_m$、normal map $\mathbf{I}_n$、feature map $\mathbf{I}_f$。MLP $\Phi$ 把 feature 解码成 diffuse color $\mathbf{c}_d$、specular tint $\mathbf{s}$、specular feature $\mathbf{f}_s$。另一个轻 MLP $\Phi_s$ 预测 specular color:

$$\mathbf{c} = \mathbf{c}_d + \mathbf{s} \odot \Phi_s(\mathbf{f}_s, \omega, \omega_r) \quad (7)$$

- $\omega$:view direction(camera → surface point)
- $\omega_r = 2(-\omega \cdot \mathbf{n})\mathbf{n} + \omega$:reflection direction
- $\mathbf{n}$:surface normal from $\mathbf{I}_n$
- $\odot$:element-wise product

这是简化的 split-sum PBR [22],把 specular 拆成 tint $\times$ environment-dependent term。 $\Phi_s$ 模拟 view-dependent specular lobe。最后用 mask $\mathbf{I}_m$ 把 color $\mathbf{c}$ 限定到 foreground。

这是 dual supervision:volumetric compositing loss 提供 geometric supervision,shading loss 提供 material/light supervision。两者共同作用,前者靠 mesh softening,后者靠 base mesh 直接 rasterize。

---

## 9. 局限与未来方向

### 9.1 Scalability

DMTet 的 tetrahedral grid 在大场景下分辨率不够(128 resolution 只覆盖 2.5m bounding box)。这是为什么 paper 主要在 object-centric 上做实验。Fig.7 的 scene-level 实验是先用 GaussianSurfel 粗 mesh 再 refine,本质上是两阶段而非 end-to-end。

未来方向:用 IMLS-Splatting 那种 cubic grid 参数化或者 hierarchical softening(远处 layer 厚,近处 layer 薄)处理 unbounded scene。

### 9.2 远距离 base mesh failure

如果 base mesh 距真实 surface 太远(超过 ±10 cm offset 范围),soft band 无法 overlap 真实 surface,梯度信号为零。这是任何 "soft" 方法的根本限制——band 必须能 "够到" 真实表面。解决方案:adaptive bandwidth(根据当前 residual photometric error 动态扩大 band),或者 coarse-to-fine schedule(先大 $\beta$ 全局定位,再小 $\beta$ 细化)。

### 9.3 极薄结构

Fig.8 显示 ship 的 cable 都没重建出来。原因:isotropic remeshing 倾向生成等边三角形,对 cable 这种 1D 结构不友好。需要 anisotropic remeshing——在 cable 方向上拉长 triangle,在垂直方向上压缩。

### 9.4 渲染器工程优化

Table 3 显示 MS 仍慢于 GS。可优化方向:
- Invisible triangle culling(被 occluded 的 triangle 不参与 compositing)
- Adaptive vertex density(平坦区域少 vertex,曲率高处多 vertex)
- 利用 mesh topology 加速 sorting(GS 每次都要全局 sort,mesh 可以利用 connectivity 信息)

---

## 10. 个人评价

这篇 paper 的核心贡献是**发现并精准定义了 mesh-based 和 volumetric 方法之间的二分问题**,然后用一个数学上 elegant 的 stop-gradient trick(eq.2)让两者 merge。这种"重新参数化+stop-gradient 让梯度走特定路径"的思路与 Neural Radiance Caching、NeuS 的 SDF density mapping 有精神上的相通——都是用 forward 数值正确但 backward 梯度被引导到期望方向的设计。

实操上 23 分钟训练 + 300K vertices + SOTA 精度是一个 sweet spot,直接可用。BMVS 上的明显领先(1.71 vs IMLS 2.75)证明 softening 在复杂几何上的价值。

潜在 impact:
- 给 mesh-based inverse rendering 一个新的 supervision paradigm
- 给 3DGS 类方法一个 mesh-aware alternative(类似 SuGaR 但 end-to-end)
- 启发后续 work:layered differentiable representations、stop-gradient as design pattern

参考综述链接:
- Differentiable Rendering Survey: https://arxiv.org/abs/2006.12057
- Recent 3DGS Surface Reconstruction: https://ingra14m.github.io/3dgs-survey/

希望这些讲解 build up 了你对该方法的 intuition,尤其是 eq.2 的 stop-gradient 设计,这是整篇 paper 的数学核心。
