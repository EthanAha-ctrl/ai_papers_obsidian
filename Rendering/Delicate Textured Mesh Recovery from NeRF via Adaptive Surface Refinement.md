---
source_pdf: Delicate Textured Mesh Recovery from NeRF via Adaptive Surface Refinement.pdf
paper_sha256: 5e75e039c359e6a5e4b089847fff7d919f8c65e53d004837b2cfc56bf9e7c061
processed_at: '2026-08-03T19:19:27-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 NeRF2Mesh

## 1. 这篇论文到底在解决什么问题

一句话:**NeRF 出来的是"一团云",但游戏引擎和 3D 软件只认"一个壳"**。

NeRF 厉害是厉害,photos 再喂进去,出来一个 volumetric 的东西,渲染要用 ray marching,一点点地穿过这团云算颜色。这玩意儿:
- 慢(没有硬件加速)
- 不能编辑(你没法 select 一块云说"我要涂成红色")
- 没法塞进 Blender、Unity、Unreal 这些工具

但是 mesh(三角面片)是整个 graphics industry 几十年的标准,所有 GPU 都专门为它做了 rasterization 硬件加速。

所以问题来了:**怎么把 NeRF 这团云,变成一个干净的、带 texture 的 mesh?**

---

## 2. 之前的办法为什么不行

### 路线 A:直接学 SDF(Signed Distance Field)

NeuS、VolSDF 这些方法学一个"距离场",每个 3D 点都有一个值,正的在外面,负的在里面,0 就是 surface。

问题在于:**SDF 要每个点都有"内外"之分**。但是树叶、网、绳子这种 thin structures(薄结构),根本没有"里面"——它薄到两个面几乎贴一起。SDF 在这里就懵了,会把这些细节糊成一团。

### 路线 B:Marching Cubes 直接提 mesh

NeRF 训完,density field 有了,用 Marching Cubes 提出来。结果:**要么 face 数量爆炸**(为了保细节就高 resolution),要么细节丢失。

### 路线 C:NVdiffrec 直接优化 mesh

用 differentiable rasterization 直接在 mesh 上 backprop。问题:**mesh 的 topology 没法从头学**——你从什么 mesh 开始?如果没有好的初始 mesh,优化就卡死。所以 NVdiffrec 只能做 object-level,还要 foreground mask。

---

## 3. 他们的核心思路(三句话)

1. **先用 NeRF 学个大概**(几何 + 外观),用 density field,因为它不要求内外,能处理薄结构
2. **把 density field 提成 coarse mesh**,用 Marching Cubes
3. **关键招**:用一个"自适应"算法,根据渲染误差来决定哪里加密 face、哪里删 face,同时微调 vertex 位置

这等于:**NeRF 当 mesh 的"好老师",先教个大概,mesh 再自己精修**。

---

## 4. 为什么用 density field 不用 SDF(这是关键 intuition)

SDF 在 thin structure 处会退化,因为它假设每个点要么在物体外、要么在物体里。但树叶是"既不在外也不在里"——它太薄了。

Density field 不一样,它只说"这个点有多大概率有东西",不要求内外概念。所以一片薄叶子,density 就是一条窄窄的高斯,SDF 学不出来,但 density 能。

他们还用了一个 trick:**exponential activation**

$$\sigma = \phi(\text{MLP}(E^{\text{geo}}(\mathbf{x}))), \quad \phi(z) = e^z$$

为什么用 $e^z$ 不用 ReLU?因为 ReLU 会让 density 在 surface 附近"拖尾",形成一片软软的 volume。exponential 会让 density 在 surface 处"陡然"上升,接近一个 binary 的 indicator——哪里有东西,哪里没有,边界清楚。

这一步等于把 NeRF 从"软的体积"逼成"硬的表面",为后面提 mesh 做准备。

参考:https://arxiv.org/abs/2201.05989 (Instant-NGP)

---

## 5. 外观怎么 export(这是另一个关键)

NeRF 的外观是 view-dependent 的——换个角度看,颜色会变。这是好事(真实),但是麻烦(没法 bake 成静态 texture)。

他们的解法:**把颜色拆成两部分**

$$\mathbf{c} = \mathbf{c}_d + \mathbf{c}_s$$

- $\mathbf{c}_d$:diffuse(漫反射),只跟位置有关,换个角度看颜色不变。直接 bake 成普通 PNG texture,Blender/Unity 谁都能用
- $\mathbf{c}_s$:specular(镜面反射),跟视角有关。用一个很小的 MLP 存进 fragment shader,运行时实时算

具体怎么拆:

$$\mathbf{c}_d, \mathbf{f}_s = \psi(\text{MLP}_1(E^{\text{app}}(\mathbf{x})))$$
$$\mathbf{c}_s = \psi(\text{MLP}_2(\mathbf{f}_s, \mathbf{d}))$$

- $E^{\text{app}}$:color grid(只依赖位置 $\mathbf{x}$)
- $\text{MLP}_1$:输出 diffuse color $\mathbf{c}_d$ + specular feature $\mathbf{f}_s$
- $\text{MLP}_2$:拿 $\mathbf{f}_s$ 和 view direction $\mathbf{d}$,算出 specular color $\mathbf{c}_s$
- $\psi$:sigmoid,压到 [0,1]

**直觉**:diffuse 是"木头本身的颜色",specular 是"从某个角度看的高光"。分开存,既保留真实感,又能塞进 graphics pipeline。

参考:https://arxiv.org/abs/2208.00277 (MobileNeRF,这个 shader trick 来自这里)

---

## 6. 最核心的算法:自适应 face density

这个是我觉得全文最漂亮的地方。

### 问题

Mesh 的 face 是离散结构,不能 backprop。你不能说"这个 face 误差大,梯度下降加一个 face"——face 数量是整数,不是连续可微的。

### 他们的解法

借鉴 IRLS(Iteratively Reweighted Least Squares)思想:**根据上一轮的 error,重新分配表达预算**。

算法流程:

```
每隔一段时间:
  1. 渲染当前 mesh,算每个 pixel 的误差
  2. 把 2D pixel 误差 reproject 回 3D mesh face
  3. 排序所有 face 的误差
  4. 误差 top 5% 的 face → 细分
     误差 bottom 50% 的 face → 删除+remesh
  5. 重新初始化 vertex offsets,继续训练
```

公式:

$$e_{\text{subdivide}} = \text{percentile}(E_{\text{face}}, 95)$$
$$e_{\text{decimate}} = \text{percentile}(E_{\text{face}}, 50)$$

**为什么这么设计**:

- High-error face = 这里细节多但 face 不够 → 加 face
- Low-error face = 这里已经 flat → 删 face 省 space
- 这相当于 mesh 上的 **importance sampling**:预算往重要的地方倾斜

**直觉类比**:这跟你在 image compression 里给高频区域分更多 bits 是一回事。Mesh 的"face"就是"bits",哪里需要细节就往哪里投。

### 实验数据看效果

看 Mic 场景的 ablation(Table 5):

| 配置 | #V | #F | Size | PSNR |
|------|-----|-----|------|------|
| 完整版 | 58k | 117k | 54.8 MB | 31.30 |
| 不要 refinement | 150k | 300k | 74.8 MB | 31.06 |

不要 refinement,face 数停在 300k(初始),mesh 更大但渲染质量反而略低。说明 adaptive 分配 face 确实更聪明。

还有更猛的对比:**NeRF-Synthetic 上 fine mesh 的 face 数(192k)比 coarse mesh(300k)还少**。意思就是 refine 之后,既质量更好,又更小。这是 adaptive density 的功劳。

---

## 7. Vertex position 怎么优化

给每个 vertex $\mathbf{v}_i$ 加一个 trainable offset $\Delta\mathbf{v}_i$:

$$\mathbf{v}_i^{\text{new}} = \mathbf{v}_i + \Delta\mathbf{v}_i$$

用 nvdiffrast 做 differentiable rasterization,渲染 mesh → image,算 photometric loss,反传到 $\Delta\mathbf{v}_i$。

这里有个很聪明的设计:**appearance 不从头学**,直接用 Stage 1 训好的 color grid。所以渲染 mesh 的时候,rasterize 出每个 pixel 的 3D surface point,直接去 query color grid 拿颜色。这样 Stage 2 只需要 10k~30k steps,非常快。

约束 vertex 乱动用两个 loss:

Laplacian smooth(公式 11):

$$\mathcal{L}_{\text{smooth}} = \sum_i \sum_{j \in S_i} \frac{1}{|S_i|} \|(\mathbf{v}_i + \Delta\mathbf{v}_i) - (\mathbf{v}_j + \Delta\mathbf{v}_j)\|^2$$

- $S_i$:$\mathbf{v}_i$ 的邻居 vertex 集合
- $|S_i|$:邻居数量

直觉:每个 vertex 不要偏离邻居太远,等于最小化 mean curvature,保持光滑。

Offset L2(公式 12):

$$\mathcal{L}_{\text{offset}} = \sum_i \|\Delta\mathbf{v}_i\|^2$$

防止 vertex 跑得太远,破坏 Stage 1 学好的 topology。

参考:https://arxiv.org/abs/2010.03955 (nvdiffrast)

---

## 8. 无界场景怎么处理(Mip-NeRF 360 这种)

户外场景太大了,直接搞 mesh 会爆炸。他们的解法:**几何级数增长的 region**

$$[-2^k, 2^k]^3, \quad k \in \{0, 1, 2, \dots\}$$

- $k=0$:中心区域 $[-1,1]^3$,最 detail
- $k=1$:$[-2,2]^3$,外层
- $k=2$:$[-4,4]^3$,更外层
- ...

每个 region 提一个 mesh,重叠部分自动 exclude。外层 marching cubes resolution 递减(反正远处看不清),iterative refinement 只跑 $k=0$。

**直觉**:远处的 mesh 不需要 refine,因为远处占屏幕像素少,error 贡献低。这跟 game engine 里 LOD(Level of Detail)是同一个思路。

参考:https://arxiv.org/abs/2111.12077 (Mip-NeRF 360)

---

## 9. 实验数据里几个亮点

### Chamfer Distance(NeRF-Synthetic,Table 1,越低越好,单位 $10^{-3}$)

| 方法 | Ship | Ficus | Mean |
|------|------|-------|------|
| NeuS | 9.54 | 2.84 | 5.64 |
| NVdiffrec | 25.89 | 5.47 | 8.15 |
| Ours fine | **8.39** | **2.44** | **5.06** |

Ship 有绳网,thin structures,NVdiffrec 直接崩(25.89)。Ficus 是薄叶子,他们的方法最好。**这就是 density field 路线对 SDF 路线的结构性优势**。

### Mesh size(Table 2)

NeRF-Synthetic 上,NeuS 提出的 mesh 有 200 万 face,他们只有 19 万 face,质量还更好。这 10 倍的压缩比就是 adaptive refinement 的功劳。

### Rendering quality trade-off(Table 4)

| 配置 | NeRF-syn PSNR |
|------|---------------|
| Ours mesh (with smooth) | 29.76 |
| Ours mesh (no smooth) | 31.04 |

关掉 smooth,PSNR 涨 1.3,但 mesh 出现 self-intersection,UV 质量崩。**这是几何质量 vs 渲染质量的 explicit trade-off**——作者选择了前者,因为 mesh 本来就是为了 editing/ downstream 用,不能 self-intersect。

---

## 10. 这套思路的更深层 intuition

我觉得这篇论文最值得记住的不是具体算法,而是这个 design pattern:

**Use representation A to bootstrap representation B**

- Representation A(NeRF / density field):容易优化,能处理复杂 topology,但难用
- Representation B(mesh):难从头优化,但好用、有硬件加速、可编辑

A 给 B 当 initial guess,B 在 A 的基础上做 local refinement。

这跟你之前在 Tesla 讲过的"teacher-student distillation"是同构的——用一个大模型 bootstrap 一个小模型,各取所长。

在 3D vision 里这个 pattern 越来越流行:
- BakedSDF:SDF → mesh baking
- NVdiffrecMC:Monte Carlo + differentiable mesh
- NeuS → Marching Cubes → mesh post-processing

NeRF2Mesh 的独特贡献是那个 **adaptive face density** 算法,本质上是 mesh 上的 importance sampling,把"表达预算"自动分配到需要的地方。

---

## 11. 局限性(自己加的直觉)

这套方法有几个根本性短板:

1. **半透明物体**:单层 rasterization 没法表达透明。玻璃瓶会变成"灰玻璃",不是真透明。
2. **毛发/fur**:需要 volumetric 表达,Laplacian smooth 会把毛发糊成块。这是 mesh representation 的根本限制。
3. **Relighting**:lighting 直接 bake 进 texture,换光照就假。想真正 relight 需要 disentangle lighting,但那会降低重建质量(参考 Nerfactor 的经验)。
4. **复杂 view-dependent**:小 MLP 处理不了强高光时,模型会"用 vertex 位移"补偿光学效果,导致 geometry 出错。这是 hybrid representation 的矛盾:**appearance capacity 不够时,geometry 会被牺牲**。

最后这一点我觉得最 interesting——它揭示了一个深层 issue:**geometry 和 appearance 在重建中是耦合的**,你不可能同时把它们都做到最好,必须 trade off。

---

## 12. 相关阅读

如果你想顺着这条线深挖:

- **NeuS** (SDF-based surface): https://arxiv.org/abs/2106.10689
- **NVdiffrec** (differentiable mesh): https://arxiv.org/abs/2111.12540
- **NVdiffrecMC** (Monte Carlo 版): https://arxiv.org/abs/2206.03380
- **MobileNeRF** (shader trick 来源): https://arxiv.org/abs/2208.00277
- **Instant-NGP** (hash grid): https://arxiv.org/abs/2201.05989
- **BakedSDF** (SDF baking): https://arxiv.org/abs/2302.14859
- **NeuralUDF** (thin structures): https://arxiv.org/abs/2211.14173
- **nvdiffrast** (differentiable rasterization): https://arxiv.org/abs/2010.03955
- **Nerfactor** (shape + reflectance decomposition): https://arxiv.org/abs/2106.07685
- **DreamFusion** (exponential density trick): https://arxiv.org/abs/2209.14988

---

## 13. 一句话总结

**NeRF 是好老师,mesh 是好学生;先用 NeRF 把 topology 搞对,再用 mesh 把表面做精,自适应地把 face 预算花在需要的地方**。

这个 pattern 我觉得在 generative 3D、3D reconstruction、甚至 robotics 的 representation learning 里都会反复出现——用一个 flexible 但 inefficient 的 representation 做初始化,换到一个 rigid 但 efficient 的 representation 上做部署。这个 idea 价值远超这篇论文本身。

---

# NeRF2Mesh 论文深度解析

Andrej 你好,我来详细拆解这篇 NeRF2Mesh。这篇文章的核心直觉很清晰:把 NeRF 的 volumetric representation 当作"几何先验"的初始化器,然后用 mesh 表征做"精确化"与"压缩"。这个思路非常重要,因为它绕开了 SDF-based 方法在 thin structures 上的根本性困难。

---

## 1. Motivation:为什么需要这条 pipeline

在 NeRF 出现之后,大家发现 implicit volumetric representation 虽然渲染质量高,但是有三个硬伤:

1. **Hardware unfriendly**:ray marching 没办法用 GPU 的 rasterization pipeline 加速
2. **Editing unfriendly**:implicit function 无法直接做几何编辑、UV 切割、texture painting
3. **Representation gap**:从 density field 提取 mesh,经历 Marching Cubes 会有信息损失

已有路径的问题:

- **SDF-based 方法** (NeuS, VolSDF):假设 surface 是 zero level set,几何 smooth,thin structures(比如树叶、绳子、网)根本无法表达,因为 SDF 的"内外"概念在 thin structure 处退化
- **Marching Cubes 直接提取**:产生海量 redundant vertices/faces(为了保 detail 就要高 resolution)
- **NVdiffrec**:用 deformable tetrahedral grid + differentiable rasterization,但是只适用 object-level,需要 foreground mask,复杂 topology 学不动

NeRF2Mesh 的关键 insight:**density field 在 NeRF 训练好后其实已经隐含了正确的 topology**(用 exponential activation 后 density 集中在 surface 附近)。我们要做的是把这个 topology "固化"成 mesh,然后只用 mesh 做 local refinement。

参考:
- NeuS: https://arxiv.org/abs/2106.10689
- NVdiffrec: https://arxiv.org/abs/2111.12540
- MobileNeRF: https://arxiv.org/abs/2208.00277
- Instant-NGP: https://arxiv.org/abs/2201.05989

---

## 2. Stage 1: Efficient NeRF Training(几何+外观的初始化)

### 2.1 Geometry: density field with exponential activation

公式 (1):

$$\boldsymbol{\sigma} = \phi(\text{MLP}(E^{\text{geo}}(\mathbf{x})))$$

变量解释:
- $\mathbf{x} \in \mathbb{R}^3$:3D query point 的坐标
- $E^{\text{geo}}$:multi-resolution hash grid encoder(Instant-NGP 那套,16 levels,每 level 1-channel feature)
- $\text{MLP}$:2 层、32 hidden channels 的浅网络
- $\phi$: **exponential activation** $\phi(z) = e^z$

为什么用 exponential 而不是 ReLU/softplus?关键直觉:

- softplus/ReLU 会让 density 在 surface 附近"拖尾",形成一个厚度不均匀的 volume
- exponential 让 density 在 surface 处"陡峭地"上升,等于把 volume rendering 的 transmittance 退化成近似 binary 的 indicator function
- 这相当于把 NeRF 从"软的体积"逼成"硬的表面"

这个 trick 来自 DreamFusion 和 Instant-NGP 后期的发现:density 集中后 Marching Cubes 提取的 surface 质量大幅提升。

### 2.2 Appearance Decomposition:把 view-dependence 拆开

公式 (2)(3)(4):

$$\mathbf{c}_d, \mathbf{f}_s = \psi(\text{MLP}_1(E^{\text{app}}(\mathbf{x})))$$
$$\mathbf{c}_s = \psi(\text{MLP}_2(\mathbf{f}_s, \mathbf{d}))$$
$$\mathbf{c} = \mathbf{c}_d + \mathbf{c}_s$$

变量:
- $E^{\text{app}}$:color hash grid(16 levels,2-channel/level)
- $\text{MLP}_1$:3 层,64 hidden,输出 3-channel diffuse $\mathbf{c}_d$ + 3-channel specular feature $\mathbf{f}_s$
- $\text{MLP}_2$:2 层,32 hidden,输入 $\mathbf{f}_s$ 和 view direction $\mathbf{d}$,输出 3-channel specular color $\mathbf{c}_s$
- $\psi$:sigmoid(把输出压到 [0,1])

为什么这样拆?核心是为了**exportability**:

- Diffuse color $\mathbf{c}_d$ 只依赖 position,可以 bake 成标准 RGB PNG texture,任何 OpenGL 设备都能读
- Specular feature $\mathbf{f}_s$ 也是 position-only 的,可以 bake 成 feature texture
- $\text{MLP}_2$ 极小(2 层 32 维),可以直接编进 fragment shader,view-dependent 部分在 GPU shader 里实时算

注意这里**没有估计 environment lighting**,而是把 illumination bake 进 texture。论文明确说:从图像反演光照通常会降低渲染质量(参考 Nerfactor、NVdiffrec 的经验)。所以这里选择"烘焙光照"换来更高的 photometric fidelity。

### 2.3 Volumetric Rendering Loss

公式 (5):

$$\hat{\mathbf{C}}(\mathbf{r}) = \sum_i T_i \alpha_i \mathbf{c}_i, \quad T_i = \prod_{j<i}(1 - \alpha_j)$$

其中:
- $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$:per-point rendering weight
- $\delta_i = t_{i+1} - t_i$:step size
- $T_i$:transmittance(光线到达第 $i$ 个点还没被 block 的概率)

这是标准 NeRF quadrature,但配合 exponential activation,实际效果是 $\alpha_i$ 接近 one-hot,绝大多数点贡献接近 0。

### 2.4 Regularizations

公式 (7) specular L2:

$$\mathcal{L}_{\text{specular}} = \sum_i \|\mathbf{c}_s(\mathbf{x}_i)\|^2$$

直觉:specular 项的 L2 惩罚会逼模型"优先用 diffuse 解释颜色",只有 diffuse 解释不了的高光才用 specular 补。这是一种 sparse-inducing prior,保证 diffuse texture 干净(适合 export 和 edit)。

公式 (8) entropy:

$$\mathcal{L}_{\text{entropy}} = -\sum_i \big(\alpha_i \log \alpha_i + (1-\alpha_i)\log(1-\alpha_i)\big)$$

直觉:二元 entropy 在 $\alpha_i \to 0$ 或 $\alpha_i \to 1$ 时最小,在 $\alpha_i = 0.5$ 时最大。这个 loss 把每个点的 rendering weight 推向 binary,等于"集中"密度到 surface 上。配合 exponential activation,这是把 NeRF "硬化"成 surface 的关键。

公式 (11) Laplacian smooth:

$$\mathcal{L}_{\text{smooth}} = \sum_i \sum_{j \in S_i} \frac{1}{|S_i|} \|(\mathbf{v}_i + \Delta\mathbf{v}_i) - (\mathbf{v}_j + \Delta\mathbf{v}_j)\|^2$$

变量:
- $S_i$:vertex $\mathbf{v}_i$ 的邻居集合
- $\Delta\mathbf{v}_i$:trainable offset(Stage 2 才用)
- $|S_i|$:邻居数

这是 mesh 上的 Laplacian operator,惩罚"每个 vertex 偏离邻居平均位置"的程度,等价于最小化 mean curvature,让 surface 保持光滑。

公式 (12) offset L2:

$$\mathcal{L}_{\text{offset}} = \sum_i \|\Delta\mathbf{v}_i\|^2$$

防止 vertex 跑得太远,破坏 Stage 1 学到的 topology。

---

## 3. Stage 2: Iterative Mesh Refinement(本文最核心贡献)

这一阶段的逻辑是把 mesh 既当**渲染载体**又当**优化对象**,联合优化 geometry + appearance。

### 3.1 Vertex Position Optimization

从 Marching Cubes 提取 coarse mesh $\mathcal{M}_{\text{coarse}} = \{\mathcal{V}, \mathcal{F}\}$,给每个 vertex $\mathbf{v}_i$ 加一个可训练 offset $\Delta\mathbf{v}_i$。用 nvdiffrast 做 differentiable rasterization:

1. Rasterize mesh → image space
2. 拿到每个 pixel 对应的 3D surface point(通过 barycentric interpolation)
3. **关键**:直接用这个 3D point 去 query Stage 1 训好的 color grid $E^{\text{app}}$ 拿 diffuse/specular
4. 计算 photometric loss,反传到 $\Delta\mathbf{v}_i$ 和 color grid

这里有一个非常重要的设计:**appearance model 从 Stage 1 继承过来**,不需要从头学。所以 Stage 2 只需要 10k~30k steps 就能 converge,比 NVdiffrec 快很多。

### 3.2 Adaptive Face Density(本文最核心算法)

Mesh faces 是离散结构,不能直接 backprop。论文借鉴 **IRLS (Iteratively Reweighted Least Squares)** 思想:根据 error 重新分配"表达预算"。

算法:

```
每隔一定 iteration:
  1. 渲染当前 mesh,计算每 pixel 的 photometric error
  2. 把 2D pixel error reproject 到对应 mesh face,累积 per-face error
  3. 排序所有 face error E_face
  4. e_subdivide = percentile(E_face, 95)   # top 5% error 的 face
     e_decimate = percentile(E_face, 50)    # bottom 50% error 的 face
  5. 对 error > e_subdivide 的 face 做 midpoint subdivision
  6. 对 error < e_decimate 的 face 做 decimation + remeshing
  7. 重新 initialize vertex offsets 和 face errors
  8. 继续训练
```

直觉解释:

- High-error face = surface 在这里有 detail 但表达不够,需要更多 face 来 fit
- Low-error face = surface 在这里已经 flat/simple,可以省 face

这相当于**自适应的 LOD (Level of Detail) learning**:数据自己决定哪里需要 dense tessellation,哪里可以 sparse。

公式 (9)(10):

$$e_{\text{subdivide}} = \text{percentile}(E_{\text{face}}, 95)$$
$$e_{\text{decimate}} = \text{percentile}(E_{\text{face}}, 50)$$

为什么阈值是 95 和 50?这是经验值,但直觉是:每次迭代只细分 top 5% error 的 face(避免 mesh 爆炸),但 decimate 一半的 face(因为大部分 error 都集中在少数复杂区域)。

Subdivision 时机在 training 的 $\{0.1, 0.2, 0.3, 0.4, 0.5, 0.7\}$ ratio 处执行,后期不再 subdiv/deci,让网络稳定 converge。

### 3.3 Unbounded Scene 处理

Mip-NeRF 360 这种户外无界场景,论文用 **geometrically growing regions**:

$$[-2^k, 2^k]^3, \quad k \in \{0, 1, 2, \dots\}$$

每个 region 提一个 mesh,重叠部分自动 exclude。外层 $k \geq 1$ 的 marching cubes resolution 递减(因为远处 detail 不重要)。Iterative refinement 只跑 center region $k=0$。

这个设计的关键 insight 是:**远处的 mesh 不需要 refine**,因为反正它在屏幕上占的像素少,error 贡献低。这避免了 outdoor scene 的 mesh 体积爆炸问题。

---

## 4. Mesh Exportation(实用化的最后一公里)

### 4.1 UV Unwrap + Texture Baking

- 用 xatlas 做 UV unwrap
- Diffuse color $\mathbf{c}_d$ → RGB PNG texture $I_d$
- Specular feature $\mathbf{f}_s$ → PNG texture $I_s$(feature 通道)
- Center mesh texture 分辨率 4096,外层 mesh 按 power-of-2 递减到最小 1024
- UV seam 处用 1-pixel out-painting 修复接缝

### 4.2 Real-time Rendering with Custom Shader

Diffuse 部分:任何 OpenGL 设备都能直接读 $I_d$ 当 RGB texture。

Specular 部分:把 $\text{MLP}_2$ 的权重 hardcode 进 fragment shader,运行时:
1. Sample $I_s$ 拿 specular feature
2. 用 view direction $\mathbf{d}$ 算 $\mathbf{c}_s = \text{MLP}_2(\mathbf{f}_s, \mathbf{d})$
3. Final color = $I_d$ + $\mathbf{c}_s$

这就是 MobileNeRF 的 shader trick,把 NeRF 的 view-dependent 部分塞进 fragment shader。

---

## 5. 实验数据深度分析

### 5.1 Chamfer Distance (NeRF-Synthetic, Table 1, 单位 $10^{-3}$)

| Method | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship | Mean |
|--------|-------|-------|-------|--------|------|-----------|-----|------|------|
| NeuS | 3.95 | 6.68 | 2.84 | 8.36 | 6.62 | 4.10 | 2.99 | 9.54 | 5.64 |
| NVdiffrec | 4.13 | 8.27 | 5.47 | 7.31 | 5.78 | 4.98 | 3.38 | 25.89 | 8.15 |
| Ours coarse | 5.76 | 7.81 | 6.05 | 7.09 | 7.15 | 4.95 | 8.71 | 10.32 | 7.23 |
| Ours fine | **4.60** | 6.02 | **2.44** | **5.19** | 5.85 | 4.51 | 3.47 | 8.39 | **5.06** |

关键观察:

- **Ship** 场景:NeuS 9.54,NVdiffrec 25.89(Ship 有复杂绳网,NVdiffrec 直接崩),Ours fine 8.39。这说明 density field 路径在 thin structures 上有**结构性优势**
- **Ficus**:Ours fine 2.44,远低于 NeuS 的 2.84。Ficus 是薄叶子,density field 能保 topology
- **Materials**:Ours 略差(4.51 vs NeuS 4.10)。Materials 场景都是非 Lambertian,view-dependent 复杂,小 MLP 解释不了就用 vertex 位移补偿 → geometry 退化
- **Coarse → Fine 的提升**:Lego 从 7.15 → 5.85,Ship 从 10.32 → 8.39。说明 iterative refinement 是真有效的,不只是 vertex position 调整

### 5.2 Mesh Size (Table 2)

| Method | NeRF-syn #V | #F | LLFF #V | #F | 360 #V | #F |
|--------|-------------|-----|---------|-----|--------|-----|
| NeuS | 1020k | 2039k | - | - | - | - |
| MobileNeRF | 494k | 224k | 830k | 339k | 1436k | 609k |
| Ours coarse | 151k | 300k | 231k | 455k | 446k | 886k |
| Ours fine | 200k | **192k** | 397k | 446k | 718k | 816k |

注意 **NeRF-syn fine mesh 的 #F 反而比 coarse 少**(192k vs 300k)!这正是 adaptive face density 的力量:细分复杂区域 + decimation 简单区域,总 face 数下降。

### 5.3 Rendering Quality (Table 4, PSNR)

| Method | NeRF-syn | LLFF | Mip-NeRF 360 |
|--------|----------|------|--------------|
| NeRF (volume) | 31.00 | 26.50 | - |
| Ours (volume) | 30.88 | 26.42 | 22.33 |
| NVdiffrec (mesh) | 29.05 | - | - |
| Ours (mesh) | 29.76 | 24.75 | 22.36 |
| MobileNeRF | 30.90 | 25.91 | 23.06 |
| Ours mesh w/o $\mathcal{L}_{\text{smooth}}$ | **31.04** | 24.90 | 22.74 |

关键 trade-off:

- $\mathcal{L}_{\text{smooth}}$ 开了,mesh quality 好但 PSNR 掉一点(31.04 → 29.76)
- $\mathcal{L}_{\text{smooth}}$ 关了,PSNR 涨到 31.04,但 mesh 有 self-intersection,UV 质量差
- 这是一个 explicit 的渲染质量 vs 几何质量的 trade-off

### 5.4 Ablation (Table 5, Mic scene)

| Variant | #V | #F | Size (MB) | PSNR |
|---------|-----|-----|-----------|------|
| Ours | 58,649 | 116,698 | 54.8 | 31.30 |
| w/o $\mathcal{L}_{\text{smooth}}$ | 202,656 | 396,385 | 133.0 | 32.57 |
| w/o refinement | 150,276 | 300,000 | 74.8 | 31.06 |

观察:

- **无 smooth**:PSNR 32.57(高),但 mesh size 翻倍(133 vs 54.8 MB)。原因:无 smooth 约束,iterative refinement 的 error 分布变得乱七八糟,subdivision 失控
- **无 refinement**:face 数停在 300k(初始),没有 adaptive density 调整,渲染质量略掉

### 5.5 Efficiency

- 单 V100 GPU,两阶段 + export 共 ~1 小时
- NVdiffrec 数小时,MobileNeRF 数天
- 这里的速度优势主要来自:**Stage 1 用 Instant-NGP 风格 grid** (而非 dense MLP),**Stage 2 不重训 appearance**

---

## 6. Limitations(我自己补充的直觉)

1. **半透明物体**:单层 rasterization 没法表达,会硬学成 opaque surface。玻璃瓶会变成"灰玻璃"。
2. **Hair/Fur**:需要 volumetric 表达,Laplacian smooth 会把毛发糊成块。
3. **Relighting**:lighting 直接 bake 进 texture,换光照会假。
4. **复杂 view-dependent**:小 MLP 处理不了强高光时,模型会"用 vertex 位移"补偿光学效果,导致 geometry 出错。这是 hybrid representation 的根本矛盾:**当 appearance capacity 不够,geometry 就会被牺牲**。

---

## 7. Intuition 总结(为什么这个 pipeline work)

这个框架的成功可以用三个 insight 解释:

**Insight 1: Density field 比 SDF 更适合做 topology 初始化**
- SDF 在 thin structure 处退化为 0(没有 inside/outside)
- Density field 只是"哪里有东西"的概率,不要求 closed surface
- 用 exponential activation 把 density "硬化"后,既保了 topology 又接近 surface

**Insight 2: Iterative refinement 是 mesh 上的 "importance sampling"**
- 不在所有 face 上均匀花预算
- 根据渲染 error 重新分配 face density
- 等价于在 mesh 上做 adaptive computation

**Insight 3: Appearance decomposition 是 exportability 的关键**
- 把 view-independent 和 view-dependent 拆开
- view-independent 部分 bake 成标准 PNG,任何引擎都能用
- view-dependent 部分用极小 MLP 嵌入 shader

这个 pipeline 给了我一个更深的直觉:**NeRF 和 Mesh 不是对立的两种 representation,而是同一 3D scene 的不同 "精度级" 表达**。NeRF 是容易优化但难用的"软"表达,Mesh 是难优化但好用的"硬"表达。NeRF2Mesh 本质是"用 NeRF 做初值,用 Mesh 做精修"的 warm-start + refinement 策略。

---

## 8. 相关延伸阅读

如果你对这个方向感兴趣,以下论文很值得看:

- **NeuS** (SDF-based surface reconstruction):https://arxiv.org/abs/2106.10689
- **NVdiffrec** (differentiable mesh + materials):https://arxiv.org/abs/2111.12540
- **NVdiffrecMC** (Monte Carlo 版本):https://arxiv.org/abs/2206.03380
- **MobileNeRF** (rasterization-friendly NeRF):https://arxiv.org/abs/2208.00277
- **Instant-NGP** (hash grid):https://arxiv.org/abs/2201.05989
- **BakedSDF** (SDF → mesh baking):https://arxiv.org/abs/2302.14859
- **NeuralUDF** (unsigned distance field for thin structures):https://arxiv.org/abs/2211.14173
- **NeRFactor** (shape + reflectance decomposition):https://arxiv.org/abs/2106.07685
- **nvdiffrast** (differentiable rasterization):https://arxiv.org/abs/2010.03955
- **Marching Cubes** (经典):https://en.wikipedia.org/wiki/Marching_cubes

---

## 9. 给 Andrej 的几句话思考

这篇工作最有意思的地方不是 mesh 质量数字,而是它对 NeRF 表征的"两阶段使用":

**Stage 1 把 NeRF 当成"可微的、topology-free 的几何先验"**,这避开了 mesh-based 方法 (NVdiffrec) 在初始化时的难题——mesh 没法从头学复杂 topology 因为没有 face 的增删机制,而 density field 天然是 Eulerian 的,拓扑无关。

**Stage 2 把 mesh 当成"高效的可微渲染代理"**,用 differentiable rasterization 做精确化,用 adaptive subdivision 做表达预算分配。

这种 "use representation A to bootstrap representation B" 的思路,跟你在 Tesla 讲过的"先用大的 teacher 跑通 pipeline 再 distill 到 efficient student"是同构的。在 3D vision 里,这种 coarse-to-fine + representation switching 是非常实用的设计 pattern。

后续的 NVdiffrecMC、BakedSDF 都在这个方向上推进,但 NeRF2Mesh 的 adaptive face density 算法在 mesh compression 角度仍然是非常干净的解法。
