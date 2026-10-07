---
source_pdf: Native and Compact Structured Latents for 3D Generation.pdf
paper_sha256: 06a8dd9e0b9331a6e336172f96e6544d8a99b6a18b2b1340933e9a0bd6b8462d
processed_at: '2026-08-05T21:50:44-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

好嘞 Andrej，我换一种讲法，就当咱俩在咖啡馆白板上画图聊。

---

## 一句话总结

**教电脑用一种新的"3D 积木"表示法画 3D 模型，这种积木能处理各种奇形怪状（开口的、薄皮的、内部空心的），还能在压缩得很狠的情况下不丢细节，最后生成又快又好看。**

---

## 1. 之前大家怎么表示 3D 物体？为什么不够好？

想象你要让电脑理解一个 3D 模型，比如一朵花、一个齿轮、一个带驾驶舱的机甲。

**老方法 A：SDF / Occupancy Field（场函数）**

思路是："空间中任意一点，告诉我它离最近表面有多远"。就像给整个空间涂上颜色，表面是 0，外面是正，里面是负。

**问题**：它默认要求物体是个"闭合气球"。你画一片叶子（一片单层的薄面）、两个边贴一起但拓扑不对的 mesh（non-manifold）、或者一个密封驾驶舱内部，它就懵了——要么强行 flood-fill 闭合，要么直接表达不出来。而且要 evaluate 这个场、做 Marching Cubes 提取 mesh，慢。

**老方法 B：TRELLIS 的路子（multi-view 2D feature）**

思路是："我围着一个 3D 物体拍一圈 2D 照片，提取每张照片的 feature，反投影回 3D 空间"。

**问题**：本质上还是从 2D 拼回 3D，中间这一步会丢拓扑信息。内部结构拍不到，复杂 material 反推不准。而且每个 asset 要存 9.6K 个 token，量太大，generative model scale 上不去。

---

## 2. O-Voxel：这篇 paper 的核心新表示

**核心 idea**：把 3D 物体切成一个稀疏的 voxel grid（只有物体表面附近的 voxel 才有内容，空的全跳过）。每个 active voxel 里存三样东西：

- 这个格子里"表面代表点"的精确位置（不是格子中心，而是表面真正经过的地方）
- 这个格子的 X/Y/Z 三个方向上，表面有没有穿过（决定怎么和邻居连成 mesh）
- 这个格子的 PBR 材质：base color + metallic + roughness + **opacity**（前人没有，让玻璃、透光这种能表达）

**为什么能处理任意拓扑？**

灵感来自一个老算法叫 **Dual Contouring**。它在每个 voxel 里放一个"代表点"（dual vertex），这个点的位置通过解一个优化问题决定：让它到所有穿过这个 voxel 的表面平面距离最小。

直觉上：如果立方体的一个角正好落在这个 voxel 里，那一定有多个面（每个面是一个平面）穿过这个 voxel。优化问题会自动把代表点放在"所有平面的交汇处"——也就是那个角上。**这就是 sharp feature 保得住的根本原因**，不用任何额外处理，数学上自动给你对齐。

paper 在原始 DC 基础上加了两个东西：
- **boundary edge 项**：如果 voxel 里有 open mesh 的边界边（比如一片叶子的边缘），代表点会被拉过去对齐这条边 —— 这就是为什么能处理 open surface
- **regularization 项**：防止退化情况（比如所有平面平行时 QEF 病态）

**关键工程优势**：mesh → O-Voxel 只要把每个 triangle 和 voxel edge 求交点，然后每个 voxel 解一个 closed-form 的 QEF，**几秒搞定**。不需要 SDF evaluation，不需要 flood-fill，不需要迭代优化。反过来 O-Voxel → mesh 也是几十毫秒。

---

## 3. SC-VAE：怎么把 O-Voxel 压到很小？

目标：把一个 1024³ 的 asset（理论上 10 亿个 voxel）压成大约 9600 个 latent token。空间压缩率 16×，是之前 voxel-based 方法的 2-4 倍。

**为什么 16× 压缩很难？**

普通下采样就是平均池化：8 个 child voxel 平均成 1 个 parent。信息直接丢了，16× 压缩后重建质量崩盘。

**Sparse Residual Autoencoding 的 idea**（从 DC-AE 借来的）：

下采样时，不平均，而是把 8 个 child 的 feature **concat 到 channel 维度**，再用 group average 把 channel 数降下来。上采样时反过来：把 channel 维展开到 space 维。

**直觉**：这就像你要把 8 个小抽屉的东西塞进 1 个大抽屉。普通平均池化是把 8 个抽屉里的东西搅成一团。Residual autoencoding 是把 8 个抽屉里的东西分别打包，整齐放进大抽屉的 8 个小隔间里。要用的时候再分开拿出。信息被打包重排，但没被销毁。

paper 的 ablation 显示：32× 压缩下，去掉这个 trick，重建误差增加 526%，PSNR 掉 1.6dB。压缩越狠，这个 trick 越关键。

**还有个 early-pruning**：上采样前预测每个 parent 的 8 个 child 里哪几个该 active，inactive 的直接跳过不计算，省内存省时间。

---

## 4. Generative Model：怎么生成？

**三阶段 pipeline**，就像画家画画：

1. **打底稿**（sparse structure generation）：决定哪些 voxel 是 active 的 —— 整体的稀疏骨架
2. **画线条**（geometry generation）：在每个 active voxel 里生成几何 latent —— 精确的形状
3. **上色**（material generation）：基于 input image + 刚生成的 shape latent，生成每个 voxel 的 PBR 材质 —— 这是 paper 的 novel 部分

前两步沿用 TRELLIS 思路，第三步是新的 —— 直接在 3D native latent 空间里做材质生成，不再走 multi-view baking 那条老路。

**模型**：三个 sparse DiT，每个 ~1.3B 参数，合起来 4B。用 flow matching（rectified flow）训练，用 DINOv3-L 提 image feature，AdaLN-single 注 timestep，RoPE 给位置编码。

**为什么能用 vanilla DiT 而不用 TRELLIS 那种带 conv packing 的复杂结构？** 因为 latent 够 compact（9.6K token vs TRELLIS 也是 9.6K，但他们只有 4× 压缩，我们 16× 压缩），可以直接处理。

**训练 trick**：
- Progressive training：先 512³ 训练，再 scale 到 1024³，prior 跨分辨率 transfer
- 相机随机放 + shallow near plane 切穿表面，强制模型学内部结构

---

## 5. 结果有多好？

**重建质量**（Table 1）：

同样 9.6K token，TRELLIS 的 mesh distance 是 85.07，这篇是 **0.0042** —— 降了 4 个数量级。Normal PSNR 从 30.29 提到 43.11。用 1/100 的 token 数就能接近 SparseFlex 1024 的精度。

**生成质量**（Table 2）：

CLIP score 0.894（最高），user study 66.5% 偏好率（Hunyuan3D 2.1 只有 13.3%）。

**速度**：

- 512³ 生成 ~3 秒
- 1024³ ~17 秒
- 1536³ ~60 秒

单 H100 GPU，比 Hunyuan3D 2.1 等快一个数量级。

---

## 6. 两个有意思的额外 trick

**Cascaded inference**：因为 latent 够 compact，可以在 inference 时多次跑第二阶段。
- 想要更高分辨率？先生成 1024³，max-pool 到 96³ sparse structure，再生成 1536³
- 想要更高质量？先生成 512³，下采样到 64³ sparse structure（修正局部错误），再生成 1024³

**FlexGEMM**：自研的 sparse conv backend，用 Triton 写，NVIDIA 和 AMD 都能跑。用 Gray code 给 voxel 排序让相似 neighborhood 的 voxel 聚一起提升 SIMD 效率，比 spconv 快 ~2×。

---

## 7. 局限

老实讲：

1. **voxel size 是硬上限**：两个平行表面距离小于 voxel size 时，QEF 会把代表点放在中间，材质也变成两者平均 —— 细节会糊
2. **偶尔有小孔**：sparse decoder 保证不了完美闭合 manifold，要 post-process 修补
3. **没有 semantic / part 信息**：只编码了几何和材质，没说"这是椅子腿、那是椅背" —— 未来可以做 part-level + graph topology 扩展

---

## 最核心的三个 intuition

1. **Dual Contouring 天然保 sharp edges**：代表点位置由 QEF 自动对齐所有穿过 voxel 的平面，cube 角这种 sharp feature 自动涌现，不用任何后处理。

2. **Space-channel 重排是高压缩 VAE 的命门**：下采样时把空间信息打包到 channel，上采样时再展开 —— 信息没丢只是换了维度，比平均池化温和得多。

3. **Native 3D > 2D multi-view**：绕过 2D 中间步骤直接从 mesh 学，拓扑信息保住了，内部结构学得到了，material 也对齐了。

---

简单说，这篇就是把 3D 表示从"拍照拼回去"升级到"直接从 3D 数据学"，顺便把压缩率提到一个新 frontier，让 4B 参数的 generative model 能跑得又快又好。Representation 和 generative model 两端都打磨得很扎实，是少见的全面工作。

---

# Native and Compact Structured Latents for 3D Generation — 深度技术解析

Hey Andrej, 这篇是 TRELLIS 的下一代工作（可以理解为 TRELLIS.2），来自 Tsinghua + MSR + USTC 团队，主要解决之前 3D 生成领域中 representation 的两个核心痛点：**topology 表达受限**和**latent 不够 compact**。我把它拆成几块讲，每块都配公式、架构细节和实验数据。

---

## 1. 核心问题：为什么之前的 representation 不够好？

在 3D 生成领域，之前主流的两个 representation paradigm 都有硬伤：

**Field-based iso-surface**（SDF / Flexicubes / Occupancy）：
- 数学上要求 **watertight** 和 **manifold**
- 没法表达 open surfaces（一片叶子）、non-manifold（两条边共享一个 vertex 但拓扑不对）、enclosed interior（密封的驾驶舱内部）
- 需要 flood-fill / SDF evaluation / 迭代优化，慢

**TRELLIS 的 SLAT**（Structured Latents）：
- 输入是 **multi-view 2D image feature**，本质上是 2D → 3D 的反投影
- 纯 rendering-based supervision
- 不能很好捕捉 complex topology 和 material
- 用 4× spatial compression，~9.6K tokens per asset，token 数量太大，限制了 generative model scale

而这篇 paper 的核心 insight 是：**直接从 native 3D data (mesh + PBR texture) 学一个 structured latent space**，绕过 2D multi-view 这个中间步骤。

参考链接：
- TRELLIS 原版: https://microsoft.github.io/TRELLIS/
- Flexicubes (前驱): https://research.nvidia.com/labs/toronto-ai/flexicubes/
- DC-AE (SC-VAE 灵感来源): https://hanlab.mit.edu/blog/dc-ae

---

## 2. O-Voxel: 一个 field-free 的 omni-voxel 表示

### 2.1 基本结构

O-Voxel 把一个 3D asset 表示为 sparse voxel grid 上的 feature tuples：

$$\mathbf{f} = \{(\mathbf{f}_i^{\text{shape}}, \mathbf{f}_i^{\text{mat}}, \mathbf{p}_i)\}_{i=1}^{L}$$

变量含义：
- $L$: active voxel 的总数（其他空 voxel 是 inactive）
- $\mathbf{p}_i \in \{0, 1, \ldots, N-1\}^3$: 第 $i$ 个 active voxel 的 3D 整数坐标，$N$ 是 grid 分辨率
- $\mathbf{f}_i^{\text{shape}}$: 该 voxel 的几何 feature
- $\mathbf{f}_i^{\text{mat}}$: 该 voxel 的材质 feature

### 2.2 Flexible Dual Grid（几何部分核心创新）

这部分是整个 paper 最聪明的 idea，灵感来自 **Dual Contouring (DC)** [Ju et al. 2002]，但做了本质改造。

经典 DC 的工作原理：
1. 输入是一个 signed grid（每个 grid corner 有 SDF 值 + normal）
2. 找 sign 变化的 edge，记录 Hermite data (intersection point $\mathbf{q}$ + normal $\mathbf{n}$)
3. 在每个 sign 变化的 cell 里放一个 **dual vertex** $\mathbf{v}$
4. 通过 QEF 求解 $\mathbf{v}$ 的位置
5. 连接相邻 dual vertex 形成 quadrilateral faces

O-Voxel 的区别：
- **不需要 field**，直接用 mesh triangle 算 edge intersection
- 每个 mesh triangle 和 voxel edge 求交点，得到 Hermite data $\{\mathbf{q}_i, \mathbf{n}_i\}$
- 每个 active voxel 的几何 feature $\mathbf{f}_i^{\text{shape}}$ 包含：
  - **Dual vertex** $\mathbf{v}_i \in \mathbb{R}_{[0,1]}^3$：voxel 内部的 surface 代表点
  - **Edge intersection flags** $\boldsymbol{\delta}_i \in \{0,1\}^3$：在 X/Y/Z 三个 axis 的预定义 edge 上是否有 intersection（决定 quad 怎么连）
  - **Splitting weights** $\gamma_i \in \mathbb{R}_{>0}$：控制 quad 怎么 split 成两个 triangle（Flexicubes 风格）

### 2.3 QEF 求解：保持 sharp features 的关键

每个 active voxel 的 dual vertex 位置 $\mathbf{v}$ 通过求解如下 QEF 得到：

$$\min_{\mathbf{v} \in \text{voxel}} e(\mathbf{v}) = \sum_i d_{\Pi, i}^2 + \lambda_{\text{bound}} \sum_j d_{L, j}^2 + \lambda_{\text{reg}} d_{\hat{\mathbf{q}}}^2$$

变量和上标下标含义：

- $d_{\Pi, i}^2 = (\mathbf{n}_i \cdot (\mathbf{v} - \mathbf{q}_i))^2$：$\mathbf{v}$ 到第 $i$ 个 intersecting plane 的距离平方，平面由 Hermite data $(\mathbf{q}_i, \mathbf{n}_i)$ 定义，$\mathbf{q}_i$ 是交点，$\mathbf{n}_i$ 是该点法向。**这一项就是经典 DC 的 QEF**，让 dual vertex 同时对齐所有穿过该 voxel 的 surface 平面 → 这就是 sharp feature 保持的根源。
- $d_{L, j}^2 = \|(\mathbf{v} - \boldsymbol{\alpha}_j) - ((\mathbf{v} - \boldsymbol{\alpha}_j) \cdot \mathbf{d}_j)\|^2$：$\mathbf{v}$ 到第 $j$ 个 **boundary edge**（open mesh 的边界边）的距离平方，$\boldsymbol{\alpha}_j$ 是 edge 起点，$\mathbf{d}_j$ 是单位方向向量。**这是 O-Voxel 的新增项**，让 dual vertex 对齐 boundary edges，从而能正确表达 open surfaces（叶子、纸片等）。
- $d_{\hat{\mathbf{q}}}^2 = \|\mathbf{v} - \bar{\mathbf{q}}\|^2$：$\mathbf{v}$ 到所有交点平均位置 $\bar{\mathbf{q}}$ 的距离平方，**regularization** 项，防止 QEF 退化（比如所有 plane 平行时 QEF 病态）。
- $\lambda_{\text{bound}}, \lambda_{\text{reg}}$：权重超参。

这个 QEF 是 closed-form 解的，**无需迭代优化**——这就是为什么 mesh → O-Voxel 转换只要几秒。

### 2.4 Material 部分：PBR-aligned

每个 active voxel 的 material feature：

$$\mathbf{f}_i^{\text{mat}} = (\mathbf{c}_i, m_i, r_i, \alpha_i)$$

- $\mathbf{c}_i \in \mathbb{R}_{[0,1]}^3$：base color (RGB)
- $m_i \in \mathbb{R}_{[0,1]}$：metallic ratio
- $r_i \in \mathbb{R}_{[0,1]}$：roughness
- $\alpha_i \in \mathbb{R}_{[0,1]}$：**opacity**（前人方法都没有，这个让 glass / translucent surface 可表达）

**Texture → O-Voxel**：对每个 active voxel，将其中心投影到所有 intersecting triangle，按距离加权平均采样 PBR 属性（用合适的 mipmap level）。

**O-Voxel → Texture**：对每个 query point（mesh vertex 或 texture texel 对应的 3D surface point），trilinear 插值邻近 voxel 的 material 属性。

### 2.5 转换速度

| 方向 | 时间 |
|------|------|
| Mesh → O-Voxel | 几秒（单 CPU） |
| O-Voxel → Mesh | 几十毫秒 |

对比之下，SDF 方法要 evaluate field、要 flood-fill、要 Marching Cubes 迭代；TRELLIS 要 multi-view rendering + fusion。这是 O-Voxel 一个工程上的巨大胜利。

---

## 3. Sparse Compression VAE (SC-VAE)：16× 空间压缩

### 3.1 架构总览

完全 sparse-convolutional 的 U-Net 风格 VAE（不像 TRELLIS 用 transformer），总共 ~800M 参数（encoder 354M + decoder 474M）。Encoder 结构（来自 Table 4）：

| Stage (f_down) | Block | 重复 |
|---|---|---|
| 1× | Linear(6, 64), ResEnc(64, 128) | - |
| 2× | LayerNorm + Linear(128, 512) + SiLU + Linear(512, 128), ResEnc(128, 256) | ×4 |
| 4× | SubMConv(3, 256, 256), LayerNorm + Linear(256, 1024) + SiLU + Linear(1024, 256), ResEnc(256, 512) | ×8 |
| 8× | SubMConv(3, 512, 512), LayerNorm + Linear(512, 2048) + SiLU + Linear(2048, 512), ResEnc(512, 1024) | ×16 |
| 16× | SubMConv(3, 1024, 1024), LayerNorm + Linear(1024, 4096) + SiLU + Linear(4096, 1024) | ×4 |

注意几个细节：
- **ConvNeXt-style block**：用 1 个 sparse conv + 1 个 wide point-wise MLP（类似 transformer FFN）替代传统的 2 个 conv，参数效率更高
- **SubMConv = Submanifold Sparse Convolution** [Graham & van der Maaten 2017]，输出 sparse grid 和输入保持一致
- 16× 下采样后 channel 是 1024，对应 1024^3 asset → 64^3 latent grid

### 3.2 Sparse Residual Autoencoding（核心创新）

这是从 DC-AE 借来的 idea，适配到 sparse voxel data。核心问题：16× 空间压缩下，标准 VAE 训练不稳，信息瓶颈太严重。

**Downsampling block**（factor 2，把 8 个 child voxel 聚合到 1 个 parent）：

$$F_{\text{coarse}}^{\text{raw}} = \text{stack}(F_{\text{child}_1}, \ldots, F_{\text{child}_8}) \in \mathbb{R}^{8C}$$
$$F_{\text{coarse}} = \text{avg\_groups}(F_{\text{coarse}}^{\text{raw}}) \in \mathbb{R}^{C'}$$

变量：
- $F_{\text{child}_k} \in \mathbb{R}^C$：第 $k$ 个 child voxel 的 feature（共 8 个，对应 2×2×2 邻域）
- $F_{\text{coarse}}^{\text{raw}} \in \mathbb{R}^{8C}$：把 8 个 child feature concat 起来
- $\text{avg\_groups}$：把 8C 维分成若干组做 average，得到 $C'$ 维（通常 $C' = 2C$）
- Missing voxels（sparsity）贡献 zero vector

**Upsampling block**（对称操作）：

$$F_{\text{fine}}^{\text{raw}} = \text{unstack}(F_{\text{coarse}}) \in \mathbb{R}^{8C'/8}$$
$$F_{\text{fine}} = \text{dup\_groups}(F_{\text{fine}}^{\text{raw}}) \in \mathbb{R}^{C}$$

- $\text{unstack}$：把 $C'$ 维 feature 拆成 $8 C'/8$ 维（即每组 $C'/8$ 维）
- $\text{dup\_groups}$：在每组内 copy 通道，恢复到 $C$ 维

**Intuition**: 这是一种 **"space-to-channel" 和 "channel-to-space" 的可逆信息重排**。下采样时空间信息被打包到 channel 维度，上采样时再展开。这避免了平均池化那种"硬扔信息"的问题，让 VAE 在 16× 压缩下还能保持高保真度。

### 3.3 Early-Pruning Upsampler

在每个 upsampling 步之前，预测一个 binary mask $\hat{\boldsymbol{\rho}} \in \{0,1\}^8$，指定每个 parent 的哪些 child 是 active 的。Inactive node 直接跳过计算，大幅减少 runtime 和 memory。

### 3.4 训练 Loss

**Stage 1**（256^3，快速稳定训练，direct O-Voxel supervision）：

$$\mathcal{L}_{s1} = \lambda_v |\hat{v} - v|_2^2 + \lambda_\delta \text{BCE}(\hat{\delta}, \delta) + \lambda_\rho \text{BCE}(\hat{\rho}, \rho) + \lambda_{\text{mat}} |\hat{\mathbf{f}}^{\text{mat}} - \mathbf{f}^{\text{mat}}|_1 + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}$$

变量：
- $\hat{v}, v$：预测 / 真值 dual vertex 位置
- $\hat{\delta}, \delta$：预测 / 真值 edge intersection flags
- $\hat{\rho}, \rho$：预测 / 真值 pruning mask
- $\hat{\mathbf{f}}^{\text{mat}}, \mathbf{f}^{\text{mat}}$：预测 / 真值 material feature
- $\lambda_*$：各 loss 项权重
- $\mathcal{L}_{\text{KL}}$：KL divergence，standard VAE 正则

**Stage 2**（512^3，引入 rendering-based perceptual loss）：

$$\mathcal{L}_{s2} = \mathcal{L}_{s1} + \mathcal{L}_{\text{render}}$$

其中：

$$d_p(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_1 + 0.2 \cdot d_{\text{SSIM}} + 0.2 \cdot d_{\text{LPIPS}}$$

$$\mathcal{L}_{\text{render}}^{\text{shape}} = \|\hat{m} - m\|_1 + 10 \cdot \|\hat{d} - d\|_1 + d_p(\hat{\mathbf{n}}, \mathbf{n})$$

$$\mathcal{L}_{\text{render}}^{\text{mat}} = d_p(\hat{\mathbf{c}}, \mathbf{c}) + d_p(\widehat{\text{mra}}, \text{mra})$$

变量：
- $m$：silhouette mask
- $d$：depth map
- $\mathbf{n}$：normal map
- $\mathbf{c}$：base color map
- $\text{mra}$：metallic-roughness-alpha 拼接的 map
- $\hat{\cdot}$ 表示模型预测

**重要 trick**：相机随机放置 + **shallow near plane 切穿 surface**，强制模型同时学外部和内部结构。

### 3.5 解耦的 latent space

为了支持 sequential generation（先生成 shape，再基于 shape 生成 material），训练了 **两个独立的 SC-VAE**：
- Shape SC-VAE：建模 $\mathbf{f}^{\text{shape}}$
- Material SC-VAE：建模 $\mathbf{f}^{\text{mat}}$，但在 upsampling 时 condition on shape VAE 的 subdivision structure

这样 material latent 和 shape latent 在空间结构上自然对齐。

---

## 4. Generative Modeling：4B 参数 Flow-Matching

### 4.1 三阶段 pipeline

1. **Sparse structure generation**：预测 sparse voxel grid 的 occupancy layout
2. **Geometry generation**：在 active voxels 内生成 geometry latents
3. **Material generation**：基于 image + generated geometry latents，生成 material latents（**这是 paper 的新 stage**）

前两阶段沿用 TRELLIS 设计，第三阶段是 novel。

### 4.2 DiT 架构（Table 5）

| Stage | Block | 重复 |
|---|---|---|
| In_proj | Linear(32(+32), 1536) | - |
| Stem | AdaLN-single → SelfAttn(12×128) → LayerNorm → CrossAttn(12×128) → AdaLN-single → FFN(1536, 8192) | ×30 |
| Out_proj | LayerNorm → Linear(1536, 32) | - |

参数：width 1536, 30 blocks, 12 heads, MLP width 8192 → 每个 DiT ~1.3B 参数，三个 DiT 加起来约 4B。

Key 设计：
- **AdaLN-single** [PixArt-α]：timestep conditioning，比标准 AdaLN 省参数
- **RoPE** [Su et al.]：rotary position embedding，cross-resolution generalization 友好
- **QK-Norm + RMSNorm**：稳定 attention
- **DINOv3-L**：提取 image conditioning feature
- **Cross-attention**：image prompt 注入
- **Channel-wise concat**：material 生成 stage 把 shape latents 拼到 input tensor，显式注入几何条件

得益于 SC-VAE 的 16× 压缩，DiT 用 **vanilla 设计**，去掉了 TRELLIS 的 conv packing 和 skip connection，更干净、更 scalable。

### 4.3 Flow Matching 训练

采用 **Rectified Flow** [Liu et al. 2023]：

$$\mathbf{x}(t) = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$$

- $\mathbf{x}_0$：data sample（latent）
- $\boldsymbol{\epsilon}$：random noise sample
- $t \in [0, 1]$：timestep

训练目标（Conditional Flow Matching, CFM）：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \|\mathbf{v}_\theta(\mathbf{x}(t), t) - (\boldsymbol{\epsilon} - \mathbf{x}_0)\|_2^2$$

变量：
- $\mathbf{v}_\theta$：神经网络预测的 vector field
- $(\boldsymbol{\epsilon} - \mathbf{x}_0)$：从 noise 到 data 的 "straight" velocity target
- 这个目标让 $\mathbf{v}_\theta$ 学到从任意 $\mathbf{x}(t)$ 流回 $\mathbf{x}_0$ 的速度场

**Timestep sampling**：用 $\text{logitNorm}(1, 1)$ 分布（TRELLIS trick），让中间步骤采样更密，提高生成质量。

### 4.4 Progressive training

- 先 512×512 image + 512^3 output（32^3 latent）训练
- 再 scale 到 1024 image + 1024^3 output（64^3 latent）
- 这样 prior 可以跨分辨率 transfer，训练高效

---

## 5. 实验结果分析

### 5.1 Reconstruction（Table 1 关键数据）

| Method | #Token | f_down | Dec(s) | MD↓ (Toys4K) | F1_{1e-8}↑ | PSNR↑ |
|---|---|---|---|---|---|---|
| Dora (4.1K) | 4.1K | - | 43.0 | 360.8 | 0.019 | 27.32 |
| TRELLIS | 9.6K | 4× | 0.108 | 85.07 | 0.074 | 30.29 |
| Direct3D-S2 1024 | 17K | 8× | 13.0 | 73.17 | 0.001 | 27.38 |
| SparseFlex 1024 | 225K | 4× | 3.21 | 0.3132 | 0.845 | 37.34 |
| **Ours 512** | **2.2K** | **16×** | **0.077** | **0.0323** | **0.888** | **39.54** |
| **Ours 1024** | **9.6K** | **16×** | **0.301** | **0.0042** | **0.971** | **43.11** |

**关键观察**：
- Ours 1024 vs TRELLIS：**MD 降了 4 个数量级**（85.07 → 0.0042），token 数相同（9.6K）
- Ours 512 vs SparseFlex 1024：MD 略差一点（0.0323 vs 0.3132），但 token 数只有 1/100，decoder 时间 0.077s vs 3.21s
- 16× compression 是 prior voxel-based 方法没达到过的
- PSNR 43.11dB 是 SOTA by a wide margin

### 5.2 Generation（Table 2）

| Method | CLIP↑ | CLIP-N↑ | ULIP-2↑ | Uni3D↑ | Pref%↑ |
|---|---|---|---|---|---|
| TRELLIS | 0.876 | 0.748 | 0.470 | 0.414 | 6.40% |
| Hunyuan3D 2.1 | 0.869 | 0.753 | 0.474 | 0.427 | 13.3% |
| **Ours** | **0.894** | **0.758** | **0.477** | **0.436** | **66.5%** |

User study 中 66.5% 偏好率，碾压所有 baseline。

### 5.3 Runtime

- 512^3 generation: ~3s
- 1024^3: ~17s
- 1536^3: ~60s (35s shape + 25s texture)

全部在单张 H100 上，比 Hunyuan3D 2.1 等快一个数量级。

### 5.4 Ablation（Table 3）

**Sparse residual autoencoding 的效果**：
- 16× 压缩：去掉 residual AE 后 MD 增加 69%，PSNR 降 0.5dB
- 32× 压缩：去掉 residual AE 后 MD 增加 **526%**，PSNR 降 1.6dB
- 压缩率越高，residual AE 越关键——这印证了"信息重排"在强瓶颈下的必要性

**Optimized residual block**：
- MD 增加 16%，PSNR 降 0.6dB
- Runtime 不变（因为 point-wise MLP 比 conv 更高效）

---

## 6. Test-Time Compute 和 Resolution Scaling

这是 paper 一个很巧的 trick：利用 SC-VAE 的 compactness 做 **cascaded inference**。

**思路 1：超分辨率生成**
1. 第一阶段生成 sparse structure
2. 第二阶段生成 1024^3 O-Voxel
3. **Max-pool 1024^3 O-Voxel 到 96^3 sparse structure**（下采样）
4. 再用第二阶段生成 1536^3 O-Voxel

这样可以在训练分辨率之外生成更高分辨率的 asset。

**思路 2：质量增强（同分辨率）**
1. 第一阶段生成 sparse structure
2. 第二阶段生成 512^3 O-Voxel
3. **下采样 512^3 O-Voxel 到 64^3 sparse structure**（修正局部错误）
4. 再用第二阶段生成 1024^3 O-Voxel

这种 cascaded 机制提供了 efficiency vs quality 的可控 trade-off。Figure 8 显示了清晰细节增强。

---

## 7. FlexGEMM: 自研 Sparse Conv Backend

这是个工程亮点，值得单独提一下。

之前 sparse conv 库（spconv, TorchSparse++, fvdb, WarpConvNet）都和 NVIDIA CUDA 强耦合。这个工作用 **Triton** [Tillet et al. 2019] 实现了一个跨平台 backend，NVIDIA 和 AMD 都能跑。

关键技术：
1. **Masked Implicit GEMM**：把 im2col（feature gathering）和 GEMM 融合成一个 kernel，最小化 global memory I/O
2. **Gray code ordering**：把相似 neighborhood pattern 的 voxel 重排到一起，提升 SIMD 效率，减少 warp divergence
3. **Split-K**：把矩阵乘法的 accumulation 维度切成多个并行 task，提升 parallelism

Benchmark：在 FP16 上比 spconv 快 ~2×。

参考链接：
- Triton: https://triton-lang.org/
- spconv: https://github.com/traveller59/spconv
- fvdb: https://research.nvidia.com/labs/toronto-ai/fvdb/

---

## 8. Limitations

诚实说一下 paper 自己承认的问题：

1. **Voxel size 限制**：当两个平行 surface 距离 < voxel size 时，QEF 会把 dual vertex 放在两者之间（最小化误差），导致 aliasing；material 也会 blur 成两者平均
2. **小孔问题**：decoder 的 sparse 性质导致有时候生成的小孔，需要 post-process 修补
3. **没有 semantic / part-level 信息**：当前 O-Voxel 只编码 geometry + material，没有更高层的结构语义

第 3 点是我觉得未来最有意思的方向——把 O-Voxel 扩展到 part-level segmentation + graph topology，可以做更多下游任务（编辑、重组合、rigging 等）。

---

## 9. Build Intuition 的核心 takeaways

1. **Dual Contouring 是 sharp features 的天然解**：通过 QEF 让 vertex 同时对齐多个 plane，sharp edges 自然涌现。O-Voxel 把这个 idea 从 field-based 推广到 field-free，直接从 mesh 提取 Hermite data。

2. **Space-channel duality 是高压缩 VAE 的关键**：传统 pooling 在 16× 压缩下会丢太多信息。Sparse residual autoencoding 通过把 8 个 child feature 拼到 channel 维度，再 group average，本质上做了一种 "信息保持的下采样"。

3. **Native 3D > 2D multi-view**：TRELLIS 走 multi-view 2D feature 路线是因为数据/监督方便，但损失了 native 3D 的 topology 信息。O-Voxel 直接从 mesh 学，避免了 2D 中间表示。

4. **Compact latent = scalable generation**：9.6K tokens 而非 225K（SparseFlex）或 17K（Direct3D-S2 1024），让 4B 参数的 DiT 训练 + 推理都变得可行。

5. **Cascaded inference 利用 latent compactness**：因为 token 少，可以在 inference 时做多次 forward pass，实现 super-resolution 或质量增强——这是 compact latent 的额外红利。

---

## 相关联想

- **Neural Dual Contouring** [Chen et al. 2022]：O-Voxel 直接继承了这个 line of work
- **DC-AE** [Chen et al. 2024]：2D 高压缩 VAE，SC-VAE 的直接灵感来源，paper 中也明确 cite
- **XCube** [Ren et al. 2024]：sparse voxel hierarchy 生成，O-Voxel 在 sparse 表示上有相似 spirit
- **Sparc3D** / **SparseFlex**：同期 sparse voxel representation 工作
- **MeshAnything**：artist-created mesh generation，autoregressive，和 O-Voxel 的 explicit mesh 输出有互补性
- **Hunyuan3D 2.1**：当前 strong baseline，但走 multi-view + baking 路线，和 native 3D 思路不同
- **nvdiffrec**：split-sum PBR renderer，这篇用它做 material 评估

整体看，这篇 paper 在 representation 层做了一次重大升级，让 3D 生成摆脱了 multi-view 2D 中间步骤的桎梏，同时把 latent compactness 推到了一个新的 frontier。O-Voxel + SC-VAE 的组合非常优雅，engineering 上也打磨得很细（FlexGEMM、cascaded inference、progressive training），是少有的 representation 和 generative model 两端都做扎实的工作。

参考链接汇总：
- Paper 主页: https://microsoft.github.io/TRELLIS.2
- TRELLIS 原版: https://microsoft.github.io/TRELLIS/
- DC-AE: https://arxiv.org/abs/2410.10733
- Dual Contouring (原始 paper): https://faculty.cs.wisc.edu/~djmercer/duacontour.pdf
- Flow Matching: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003
- PixArt-α (AdaLN-single): https://arxiv.org/abs/2310.00426
- DINOv3: https://arxiv.org/abs/2508.10104
- Objaverse-XL: https://objaverse.allenai.org/
- Submanifold Sparse Conv: https://arxiv.org/abs/1706.01307
- ConvNeXt: https://arxiv.org/abs/2201.03545
