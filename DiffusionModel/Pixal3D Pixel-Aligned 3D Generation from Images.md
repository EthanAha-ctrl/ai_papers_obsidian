---
source_pdf: Pixal3D Pixel-Aligned 3D Generation from Images.pdf
paper_sha256: ddcd9538c8f6f3fb7df45ec247dc7e729a2f9b7fa066e189c8f18c6e533b6dcf
processed_at: '2026-08-06T04:19:49-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Pixal3D 用人话讲

## 一句话总结

Pixal3D 干的事情就是：**别让网络猜哪个像素对应哪个 3D 位置了，直接用几何把对应关系算出来，硬塞给网络。**

---

## 问题出在哪

你给一张图，要生成 3D 模型。现有的方法（TRELLIS、Hunyuan3D、Direct3D-S2）都这么干：

1. 在一个**标准坐标系**（canonical space）里生成 3D 形状——比如让椅子永远朝某个固定方向摆
2. 拿图像特征通过 **cross-attention** 喂给 3D latent

问题就在 cross-attention 这步。cross-attention 是什么？是让网络去**搜索**：3D 空间里某个位置，该看图像里哪个 token 的信息。这个"搜索"是 network 要学的，学得好不好全看 data 和 training。

结果就是网络经常**偷懒**——它不真的建立 pixel-to-3D 的精确映射，而是抓个大概的语义意思。"哦这地方有键盘，那生成一堆按键吧"，至于按键数量对不对、排布对不对，它不太care。fidelity 就这么塌了。

你看实验数据（Toys4K 数据集，Table 1）：

| Method | IoU↑ | Mean Error↓ | 30° Accuracy↑ |
|--------|------|------------|--------------|
| Hunyuan3D-2.1 | 83.33 | 21.19° | 75.83 |
| Direct3D-S2 | 74.23 | 29.99° | 63.20 |
| TRELLIS | 79.48 | 25.00° | 70.80 |
| **Pixal3D** | **93.57** | **16.63°** | **85.35** |

IoU 从 83 跳到 93，这是**质变**级别的提升。

---

## Reconstruction 为什么没这毛病

你看 3D reconstruction 领域（MVS、单目深度估计、DUSt3R、VGGT），人家压根没 fidelity 问题。为什么？

因为 reconstruction **天生就是 pixel-aligned 的**：
- 单目深度估计：每个 pixel 直接对应一个 depth 值，一对一，没歧义
- MVS：pixel correspondence + triangulation，几何上严丝合缝
- DUSt3R / VGGT：直接预测 pixel-aligned point map

每个 2D pixel 对应一条 camera ray，ray 上的 3D 位置就是"这个 pixel 看到的东西"。correspondence 是**几何给的**，network 不用学。

Reconstruction 的缺点是：只能搞可见表面，背面、遮挡区域全没。输出是残缺的，当不了 3D asset 用。

---

## Pixal3D 的核心 Insight

**把 reconstruction 的几何严谨性 + generation 的补全能力结合。**

具体做法：
- **可见部分**：用几何 back-projection 强制约束——像 reconstruction 一样忠实
- **不可见部分**：用 diffusion 的 generative prior 补——像 generation 一样 plausible

Correspondence 给定，completion 学习。各干各的活。

---

## 怎么做的：Back-Projection Conditioning

### 核心操作

给定输入图像 $I$，DINOv2 提取 2D feature map $F \in \mathbb{R}^{H' \times W' \times C}$。

对 3D volume 里的每个 voxel（坐标 $\mathbf{X} = (X, Y, Z)^T$，camera coordinate 下），**投影到 image plane**：

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
$$

变量解释：
- $K$：camera intrinsic matrix，$3 \times 3$
- $u, v$：image plane 上的坐标
- $X, Y, Z$：voxel 在 camera coordinate 下的 3D 坐标
- $\sim$：齐次坐标等价（差一个 scale）
- 下标 $i, j, k$：voxel 在 3D grid 里的 index

然后在 $F$ 上做 **bilinear interpolation** 采样，把采到的 feature 塞给这个 voxel。

所有 voxel 走一遍，得到一个 **3D feature volume** $V \in \mathbb{R}^{N^3 \times C}$（$N=64$）。

### 怎么用这个 volume

把 $V$ **直接加到 noise volume** 上，作为 conditioning signal：

$$
z_t' = z_t + V_{\text{feat}}(I)
$$

变量：
- $z_t$：diffusion 在 time step $t$ 的 noisy latent
- $z_t'$：加完 condition 后的 input
- $V_{\text{feat}}(I)$：back-projected image feature volume
- 上标 $'$：表示 modified version

这就相当于告诉 diffusion：**"每个 voxel 位置该长成什么样，图像已经告诉你了，你照着来"**。

### Multi-Scale Features 很关键

DINOv2-Large 输出 patch tokens 约 $37 \times 37$，太粗了。键盘按键、花瓣这种 fine detail 直接糊掉。

用 NAF [Chambon et al. 2025] 上采样到 518×518 full resolution，然后 multi-scale 采样：

$$
V_{ijk} = \frac{1}{S} \sum_{s=1}^{S} F_s(u_{ijk}, v_{ijk})
$$

变量：
- $F_s$：第 $s$ 个 scale 的 feature map
- $S$：scale 总数
- $s$：scale index（下标）
- $ijk$：voxel index（下标）

在 cross-attention 框架下，dense attend 到 high-res map 是 **prohibitively expensive**。但 back-projection 下每个 voxel 独立采样，**essentially cost-free**。这是 pixel-aligned paradigm 的额外红利。

---

## Canonical vs. Pixel-Aligned 直觉

### Canonical generation

Object 定义在 view-independent 的标准朝向。好处是容易学 category prior（椅子都长差不多）。坏处是 2D-3D correspondence 模糊——网络得猜"图像里这个像素，对应 canonical space 里哪个位置"。

### Pixel-aligned generation

Object 定义在 **camera coordinate frame**。"As seen from camera"。3D volume 和 image frustum 对齐。

每个 pixel ↔ 一条 camera ray ↔ 一个 3D locus。**Correspondence 从 learned stochastic behavior 变成 solid geometric prior**。

---

## Backbone：Direct3D-S2 的改造

Pixal3D 用 Direct3D-S2 [Wu et al. 2025b] 作 base，保留其核心：
- **Dense stage**：VAE + DiT 生成 coarse occupancy grid（$64^3$）
- **Sparse stage**：sparse DiT denoise sparse voxel latents，VAE decoder 解码成 sparse SDF
- **Marching Cubes**：SDF → mesh

**改造点**：
1. VAE 编码 **pixel-aligned SDF**（非 canonical SDF）——不同 view 对应不同 camera-space object
2. Cross-attention → **back-projection conditioning**
3. 保留 DINOv2 global token 的 cross-attention（global semantic guidance）

训练数据：TRELLIS-500K（Objaverse subset），对每个 mesh 做 random rotation + frontal rendering with varying FoV/distance，强制网络学 view-dependent, pixel-aligned generation。

Reference:
- Direct3D-S2: https://arxiv.org/abs/2505.17412
- TRELLIS: https://arxiv.org/abs/2412.01506
- DINOv2: https://arxiv.org/abs/2304.07193

---

## Multi-View 扩展：自然又优雅

因为 single-view formulation 是 explicit projection geometry，扩展到 multi-view 就是：

1. 每个 view 独立 back-project 成 feature volume
2. **Aggregation by averaging**

$$
V_{ijk} = \frac{1}{M} \sum_{v=1}^{M} V^{(v)}_{ijk}
$$

变量：
- $V^{(v)}$：view $v$ 的 back-projected feature volume
- $M$：view 数量
- $v$：view index（上标）

**为什么 averaging 就 work？**

因为 pixel-aligned formulation 保证了不同 views 的 features 在 3D space 上**几何对齐**——同一 voxel 接收来自不同 views 的信息，对应同一 3D 位置的不同观察。这隐含了 multi-view stereo 的核心原理：同一 3D 点在不同 views 下应该 consistent。Averaging 是 bayesian-style 的信息融合。

Cross-attention 的 multi-view 需要网络学会区分不同 views 的 features 并 implicit 融合。Pixal3D 的 averaging 是 geometrically meaningful 的。

实验数据（Table 3）：

| Method | Views | CD↓ | EMD↓ | F-Score↑ |
|--------|-------|-----|------|----------|
| VGGT | 2 | 613.55 | 19.60 | 9.57 |
| VGGT | 6 | 2791.10 | 25.33 | 9.67 |
| TRELLIS | 2 | 21.39 | 2.40 | 43.68 |
| TRELLIS | 6 | 18.13 | 2.16 | 46.02 |
| **Pixal3D** | 2 | 5.27 | 1.13 | 64.94 |
| **Pixal3D** | 6 | 4.16 | 1.00 | 69.04 |

注意 VGGT（pure reconstruction）**views 越多越差**——累积误差 + 无 generative prior 补全。Pixal3D **views 越多越好**——reconstruction cues 增强，generative ambiguity 下降。这就是 generative reconstruction 的精髓。

---

## Scene Generation：模块化 Pipeline

对比 SAM3D [SAM et al. 2025]：

**SAM3D 的坑**：生成 canonical-pose object，再预测 7-DoF pose（rotation + translation + scale）对齐到 camera frame。7-DoF pose estimation 从 image 是 non-robust 的，inter-object relations 经常错。

**Pixal3D 的 pipeline**：

1. **Segmentation & Completion**：SAM3 分割 → Qwen-image-edit 补全 occluded regions
2. **Pixel-Aligned Generation**：每个 object 直接在 camera space 生成，orientation 已经对齐
3. **Global Alignment**：MoGe 预测 global point map → least-squares 估计 scale 和 depth

Global alignment 的数学：

设 object $o$ 的 pixel-aligned point cloud $\{p_i^{(o)}\}$，MoGe 预测对应 $\{q_i^{(o)}\}$，估 scale $\sigma^{(o)}$ 和 depth offset $\delta^{(o)}$：

$$
\min_{\sigma^{(o)}, \delta^{(o)}} \sum_i \| \sigma^{(o)} p_i^{(o)} + \delta^{(o)} \mathbf{d}_i - q_i^{(o)} \|^2
$$

变量：
- $\sigma^{(o)}$：object $o$ 的 scale（待估参数）
- $\delta^{(o)}$：object $o$ 的 depth offset（待估参数）
- $p_i^{(o)}$：object $o$ 的第 $i$ 个 pixel-aligned 3D point
- $q_i^{(o)}$：MoGe 预测的对应 point
- $\mathbf{d}_i$：pixel $i$ 的 view direction（已知）
- 上标 $(o)$：object index

标准 least-squares，相对 7-DoF pose estimation **极度稳定**。因为 Pixal3D 已经搞定了最难的 rotation alignment（pixel-aligned 天然解决），只剩 scale 和 depth 要解。

Reference:
- SAM3D: https://arxiv.org/abs/2511.16624
- SAM3: https://arxiv.org/abs/2511.16719
- Qwen-Image: https://arxiv.org/abs/2508.02324
- MoGe: https://arxiv.org/abs/2410.19115

---

## Inference 时的 Cube Placement

Training 用 ground-truth camera intrinsics, distance $d$, cube scale $s$。

Inference 用一个 robust heuristic：
- 选较小 FoV
- Unit cube scale ($s=1$)
- 计算 $d$ 使得 image 四角 cast 的 ray 精确通过 cube back face 四个顶点

公式：
$$
\tan(\theta) = \frac{s/2}{d + s/2} \implies d = \frac{s/2}{\tan(\theta)} - s/2
$$

变量：
- $\theta$：half FoV 角度
- $s$：cube edge length
- $d$：camera plane 到 cube center 的距离

这保证 frustum 信息完整 captured，voxel 利用率不过度损失。论文说这个策略 stable and robust，all experiments 都用它。

---

## Ablation 的启示

Figure 8 的 ablation：

1. **去掉 feature upsampling**：用 $37 \times 37$ 的 coarse DINOv2 tokens → fine details 丢失，misalignment 出现。键盘按键、面部细节这种全糊。

2. **去掉 back-projection，换回 cross-attention**：训练 **slow to converge, unstable**，最终 fidelity 显著降低。

第二个尤其重要。它说明 back-projection 是**训练稳定性的关键**，单纯 design choice。直觉上，pixel-aligned generation 是一个更约束的问题空间，cross-attention 在这个空间下要学习一个已经 implicit 定义的 correspondence，引入 redundancy 和 instability。

---

## 为什么这个思路意义重大

### 对 image-to-3D 的直接影响

fidelity 一直 是 image-to-3D 的 central bottleneck。现有方法生成的东西"差不多像"，但没法做到"像素级忠实"。Pixal3D 把 fidelity 推到 **near reconstruction-level**，这是个 milestone。

### 更深层的启示

Cross-attention 在 image-to-3D 中的角色，本质上是让网络**重新发现投影几何**。这浪费 capacity——网络学 correspondence 的 capacity 本可以用来学更好的 generative prior。

Pixal3D 把 correspondence **hardcode** 进 backbone，让网络专注学 completion。这符合 Software 1.0 / 2.0 混合的哲学：**explicit、可形式化的先验应该 hardcoded；fuzzy、需要数据驱动的部分才 learned**。

这个 paradigm 对其他 generative tasks 也有启示——只要存在 explicit input-output correspondence（如 depth estimation 中的 image-to-depth, normal estimation 中的 image-to-normal），都应该优先利用它，而非让 cross-attention 从头学。

### 与 Classical CV 的融合

back-projection 是 classical multi-view geometry 的经典操作（Atlas [Murez et al. 2020], NeuralRecon [Sun et al. 2021] 都用过）。Pixal3D 把这个 classical 操作 integrate 到 modern diffusion backbone 里。

这预示着 CV 领域的一个趋势：**pure data-driven 的 end-to-end learning 可能不是最优解**。把 domain knowledge（几何、物理、光学）hardcode 进 architecture，让 network 只学 data-driven 的部分，可能更 sample-efficient、更 robust、更 interpretable。

Reference:
- Atlas: https://arxiv.org/abs/2003.10432
- NeuralRecon: https://arxiv.org/abs/2104.00681
- DUSt3R: https://arxiv.org/abs/2403.11279
- VGGT: https://arxiv.org/abs/2503.11651
- ReconViaGen: https://arxiv.org/abs/2510.23306
- CUPID: https://arxiv.org/abs/2510.20776

---

## Limitations

1. **Pixel-level noise sensitivity**：imperfect segmentation boundaries 会被 back-projected 并放大成 small geometric artifacts。Pixel-aligned 的双刃剑——精确对应也意味着精确传递 noise。

2. **Multi-view 需要已知 camera poses**：假设 camera parameters 准确已知。实际应用需要额外 SfM 或 pose estimation。

3. **Scene generation 依赖 2D inpainting**：Qwen-image-edit 在复杂 occlusions 下可能引入 errors，传递到 3D generation。

---

## 未来方向

1. **Texture & Material**：pixel-aligned paradigm 天然适合 appearance fidelity——back-project color / BRDF 信息
2. **3D Editing via 2D pixel manipulation**：pixel-aligned 意味着 2D image editing 直接对应 3D 修改
3. **Video-based 3D scene generation**：temporal consistency + pixel-aligned = high-fidelity world building
4. **Self-supervised pose estimation**：将 camera pose estimation 集成到 framework 内部
5. **Dynamic scene / 4D generation**：pixel-aligned paradigm 自然扩展到 temporal dimension

---

Project page: https://ldyang694.github.io/projects/pixal3d/

---

# Pixal3D: Pixel-Aligned 3D Generation 深度技术解析

## 1. 核心问题与 Motivation：为什么需要 Pixel-Aligned Generation？

### 1.1 Fidelity Bottleneck 的本质

当前 image-to-3D 生成方法（如 TRELLIS、Hunyuan3D-2.1、Direct3D-S2、TripoSG）尽管在 geometry quality 和 appearance realism 上取得巨大进展，但都存在一个**根本性 fidelity 问题**：生成的 3D 资产在像素级别上无法忠实于输入图像。

Karpathy 你肯定会敏锐地意识到，这本质上是一个 **correspondence ambiguity** 问题。让我从信息论角度剖析：

现有方法的 pipeline 是：
1. 3D generator 在 **canonical space**（对象中心坐标系）下生成 shape
2. Image 信息通过 **cross-attention** 注入 3D latent
3. Cross-attention 需要"学习"每个 image token 应该影响哪些 3D token

这种"学习式" correspondence 存在本质 ambiguity：
- Canonical space 中多个 3D 位置可以解释相同的 2D evidence（在 unknown pose 下）
- Model 倾向于 **cheat**——利用 global semantic cues 而建立数学上 faithfull 的 pixel-to-3D mapping
- 对于 repetitive parts（如键盘按键、花瓣）或者 multiple input views，confusion 进一步放大

### 1.2 Reconstruction 的启示

3D reconstruction 领域（MVS、SfM、单目深度估计、DUSt3R、VGGT、MoGe）几乎没有这个问题，因为它们**显式建立 2D-3D correspondence**：
- Multi-view geometry 基于 pixel correspondences 和 triangulation
- Single-view reconstruction 直接在 pixel-aligned 方式下预测 depth / normal / point map
- 每一个 2D pixel 对应一个唯一的 3D locus（相机射线）

Pixal3D 的核心 insight：**把 reconstruction 的几何严谨性 marriage 到 generation 的创造力上**。

Reference links:
- DUSt3R: https://arxiv.org/abs/2403.11279  
- VGGT: https://arxiv.org/abs/2503.11651
- MoGe: https://arxiv.org/abs/2410.19115
- Direct3D-S2: https://arxiv.org/abs/2505.17412
- TRELLIS: https://arxiv.org/abs/2412.01506

---

## 2. Pixel-Aligned Generation 的核心 Intuition

### 2.1 Canonical vs. Pixel-Aligned 范式对比

**Canonical generation**：
- Object 定义在 view-independent 的 default orientation
- Semantic components（车头、椅子座）锚定到 predefined axes
- 优势：易于学习 category-level priors
- 劣势：2D-3D correspondence underconstrained

**Pixel-aligned generation**：
- Object 定义在 input camera coordinate frame
- "As seen from camera"——3D volume 与 image frustum 对齐
- 每个 pixel ↔ 一条 camera ray ↔ 一个 3D 结构化 locus
- **Correspondence 从 learned stochastic behavior 变成 solid geometric prior**

### 2.2 为什么这个 shift 如此 powerful？

直觉上，cross-attention 是一个 **soft search** 操作——网络需要从 3D token 出发，在所有 image tokens 中找到相关的，这是一个高维的、容易塌缩的检索问题。而 back-projection 是一个 **hard geometric assignment**——给定 voxel 坐标，通过投影几何直接确定它对应 image 中的位置，没有歧义。

这就把一个**需要学习的问题**变成了一个**已知几何先验**——网络不必浪费 capacity 去学 correspondence，而是直接利用 correspondence 去做 generation。

---

## 3. Back-Projection Conditioning 的数学细节

### 3.1 坐标系与投影关系

设：
- $I \in \mathbb{R}^{H \times W \times 3}$：输入图像
- $F = \text{DINOv2}(I) \in \mathbb{R}^{H' \times W' \times C}$：2D feature map（patch tokens）
- $K \in \mathbb{R}^{3 \times 3}$：相机内参矩阵
- $V \in \mathbb{R}^{N^3 \times C}$：3D feature volume（分辨率 $N=64$）

Back-projection 的核心公式——给定 3D voxel 中心坐标 $\mathbf{X} = (X, Y, Z)^T$（在 camera coordinate 下），投影到 image plane：

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \cdot \mathbf{X} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
$$

其中：
- $f_x, f_y$：focal length（像素单位）
- $c_x, c_y$：principal point
- $u, v$：image plane 上的坐标
- $\sim$：定义到齐次坐标的等价

反向操作：从 voxel grid 出发，对每个 voxel center $\mathbf{X}_{ijk} = (X_i, Y_j, Z_k)^T$：
1. 投影到 $(u_{ijk}, v_{ijk})$
2. 在 image feature map $F$ 上做 **bilinear interpolation** 采样：$F(u_{ijk}, v_{ijk})$
3. 将采样到的 feature 赋给该 voxel：$V_{ijk} = F(u_{ijk}, v_{ijk})$

### 3.2 Cube Placement 参数化

3D generator 需要预定义 bounding box（unit cube），需要确定 cube 在 camera frustum 中的位置和大小。两个关键参数：

- $d$：camera plane 到 cube center 的距离
- $s$：cube scale（边长）

**Training**：使用 ground-truth 投影参数（camera intrinsics, $d$, $s$）。

**Inference**：采用一个 stable 的 heuristic 策略——选择较小的 FoV，unit cube scale ($s=1$)，计算 camera distance $d$ 使得从 image 四角 cast 的 ray 精确通过 cube back face 的四个顶点。这保证 frustum 信息完整 captured，同时 voxel 利用率不过度损失。

具体计算（推断时）：
设 cube 边长 $s=1$，back face 距离 $d + s/2$，half FoV 角度 $\theta$：
$$
\tan(\theta) = \frac{s/2}{d + s/2} \implies d = \frac{s/2}{\tan(\theta)} - s/2
$$

### 3.3 Multi-Scale 2D Features 的关键性

DINOv2-Large 输出 patch tokens 大小约 $37 \times 37$（input 518×518），这个分辨率对 fine-grained details（如 facial features、花瓣数量）太粗糙。

解决方案：使用 NAF (Neighborhood Attention Filtering) [Chambon et al. 2025] 将 DINOv2 patch tokens 上采样到 full resolution（518×518）。

Multi-scale back-projection：
$$
V_{ijk} = \frac{1}{S} \sum_{s=1}^{S} F_s(u_{ijk}, v_{ijk})
$$

其中 $F_s$ 是第 $s$ 个 scale 的 feature map，$S$ 是 scale 总数。这一步在 cross-attention 框架下成本高昂（dense attention 到 high-res map），但在 back-projection 下 **essentially cost-free**，因为每个 voxel 都独立采样。

Reference:
- DINOv2: https://arxiv.org/abs/2304.07193
- NAF: https://arxiv.org/abs/2511.18452

---

## 4. 整体架构详解

### 4.1 Backbone：Direct3D-S2 的继承与改造

Pixal3D 保留 Direct3D-S2 的核心架构：
- **Dense stage**：编码 / 采样 coarse occupancy grid（$64^3$）
- **Sparse stage**：sparse DiT denoise noisy sparse voxel latents，VAE decoder 解码成 sparse SDF
- **Marching Cubes**：从 SDF 提取 final mesh

**关键改造**：
1. VAE 编码 **pixel-aligned SDF**（而非 canonical SDF）——不同 input view 对应不同 camera-space object $X$，从而不同 latents $z_0$
2. Cross-attention conditioning → **Back-projection conditioning**
3. 保留 DINOv2 global token 的 cross-attention（提供 global semantic guidance）

### 4.2 三个核心组件（对应 Figure 2）

**(1) Pixel-Aligned Structured Latent Representation Learning**
- VAE 压缩 pixel-aligned sparse SDF 到 efficient sparse latents
- Pretrained VAE 在 pixel-aligned SDF 上 robust 工作，仅 finetune decoder

**(2) Image Back-Projection-based Conditioner**
- DINOv2-Large 提取 2D features
- NAF 上采样到 518×518
- Multi-scale back-projection 形成 3D feature volume
- Feature volume 直接加到 noise volume

**(3) Two-Stage Generative Process**
- Structure Generation（dense stage）：预测 coarse structure
- Structured Latents Generation（sparse stage）：预测 detailed latents

### 4.3 Conditional Diffusion 的形式化

标准 3D latent diffusion：
$$
z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Pixal3D 的 conditional formulation：
$$
z_t' = z_t + V_{\text{feat}}(I)
$$

其中 $V_{\text{feat}}(I)$ 是 back-projected image feature volume，作为 spatially-aligned 条件信号。这类似于 ControlNet 的思路但更直接——因为 feature volume 和 noise volume 在 3D 空间上**严格对齐**。

### 4.4 训练数据构造

- 数据集：TRELLIS-500K（Objaverse subset）
- 关键步骤：对每个 mesh 应用 random object-centric rotations，从 frontal perspectives 用 varying FoV 和 camera distances 渲染
- 这强制网络学习 view-dependent, pixel-aligned generation
- Watertight 每个 mesh，计算 SDF
- Multi-view training：fine-tune single-view model，随机采样 2-6 views 作为 condition

---

## 5. Multi-View 扩展的优雅性

### 5.1 公式化

给定 multi-view images $\{I_v\}_{v=1}^{M}$ 和已知 camera parameters $\{K_v, R_v, t_v\}_{v=1}^{M}$：

1. 每个 view 独立 back-project：
$$
V^{(v)}_{ijk} = F_v(u^{(v)}_{ijk}, v^{(v)}_{ijk})
$$
其中 $(u^{(v)}_{ijk}, v^{(v)}_{ijk})$ 是 voxel $\mathbf{X}_{ijk}$ 在 view $v$ 下的投影。

2. **Aggregation by averaging**：
$$
V_{ijk} = \frac{1}{M} \sum_{v=1}^{M} V^{(v)}_{ijk}
$$

这个简单 averaging 之所以 work，是因为 **pixel-aligned formulation 保证了不同 views 的 features 在 3D space 上几何对齐**——同一 voxel 接收来自不同 views 的信息，对应同一 3D 位置的不同观察。

### 5.2 为什么这比 cross-attention 方案更优？

Cross-attention 的 multi-view conditioning 需要网络学会区分不同 views 的 features，并在 attention 中 implicit 融合。而 Pixal3D 的 averaging 是一个**几何上 meaningful** 的操作——它隐含了 multi-view stereo 的核心原理：同一 3D 点在不同 views 下应该 consistent，averaging 实现了 bayesian-style 的信息融合。

### 5.3 View 数量对生成的影响

Table 3 数据显示：
| View 数 | CD (×10⁻⁴) | EMD (×10⁻²) | F-Score |
|---------|-----------|------------|---------|
| 2 | 5.27 | 1.13 | 64.94 |
| 4 | 4.73 | 1.05 | 67.85 |
| 6 | 4.16 | 1.00 | 69.04 |

随着 views 增加：
- Generative ambiguity 下降
- Reconstruction cues 增强
- 3D shape 更 deterministic

这正是 3D generative reconstruction 的 fundamental principle——**views 越多越接近 reconstruction，越少越依赖 generative prior**。

---

## 6. Modular Scene Generation Pipeline

### 6.1 与 SAM3D 的对比

**SAM3D** 的 pipeline：
1. SAM3 分割 objects
2. TRELLIS backbone 生成 canonical-pose object
3. **预测 7-DoF pose**（rotation + translation + scale）以对齐到 camera frame
4. 组装成 scene

**问题**：7-DoF pose estimation 从 image 是 non-robust 的，经常导致 inter-object relations 错误（wrong rotations, misaligned placements, incorrect contact/support）。

**Pixal3D** 的 modular pipeline：
1. **Segmentation & Completion**：SAM3 分割 → Qwen-image-edit 完成 occluded regions
2. **Pixel-Aligned Generation**：每个 object 直接在 camera space 生成
3. **Global Alignment**：MoGe 预测 global point map → 用 least-squares 估计 object scale 和 depth

### 6.2 Global Alignment 的数学

由于 Pixal3D 输出和 MoGe 预测都是 pixel-aligned，可以直接 formulate point-wise constraints：

设 object $o$ 的 pixel-aligned point cloud $\{p_i^{(o)}\}$，MoGe 预测的对应 points $\{q_i^{(o)}\}$，需要估计 scale $\sigma^{(o)}$ 和 depth offset $\delta^{(o)}$：

$$
\min_{\sigma^{(o)}, \delta^{(o)}} \sum_i \| \sigma^{(o)} p_i^{(o)} + \delta^{(o)} \mathbf{d}_i - q_i^{(o)} \|^2
$$

其中 $\mathbf{d}_i$ 是 pixel $i$ 的 view direction。这是一个标准 least-squares 问题，相对 7-DoF pose estimation **极度稳定**。

Reference:
- SAM3: https://arxiv.org/abs/2511.16719
- SAM3D: https://arxiv.org/abs/2511.16624
- Qwen-Image: https://arxiv.org/abs/2508.02324
- MoGe: https://arxiv.org/abs/2410.19115

---

## 7. 实验结果深度分析

### 7.1 Single-View Quantitative (Toys4K, Table 1)

| Method | IoU↑ | PSNR↑ | SSIM↑ | LPIPS↓ | Mean↓ | Median↓ | Mean_B↓ | 11.25°↑ | 22.5°↑ | 30°↑ |
|--------|------|-------|-------|--------|-------|---------|---------|---------|--------|------|
| TRELLIS | 79.48 | 20.98 | 0.883 | 0.204 | 25.00 | 17.97 | 36.04 | 46.82 | 63.99 | 70.80 |
| TripoSG | 73.54 | 19.73 | 0.873 | 0.250 | 28.55 | 21.20 | 41.71 | 39.85 | 57.18 | 64.81 |
| Hunyuan3D-2.1 | 83.33 | 21.96 | 0.889 | 0.179 | 21.19 | 14.05 | 32.46 | 51.37 | 69.08 | 75.83 |
| Direct3D-S2 | 74.23 | 19.49 | 0.851 | 0.268 | 29.99 | 23.46 | 41.04 | 37.56 | 55.46 | 63.20 |
| **Pixal3D** | **93.57** | **24.21** | **0.897** | **0.108** | **16.63** | **11.77** | **21.80** | **53.13** | **77.96** | **85.35** |

**关键观察**：
- **IoU 93.57 vs 83.33**（Hunyuan3D-2.1）：提升 10 个百分点，这是巨大 gap
- **Mean angular error 16.63 vs 21.19**：normal 误差降低 22%
- **11.25° accuracy 53.13**：虽然比 Hunyuan3D 仅高 2 点，但 22.5° 和 30° 阈值上优势扩大到 9 和 10 点
- 这说明 Pixal3D 在 fine details 上优势尤其明显——更多 pixels 落在低误差范围

### 7.2 In-the-Wild User Study (Table 2)

| Method | Uni3D↑ | ULIP2↑ | Fidelity↑ | Quality↑ |
|--------|--------|--------|-----------|----------|
| TRELLIS | 41.09 | 44.76 | 1.86 | 1.99 |
| TripoSG | 40.99 | 44.64 | 2.25 | 2.14 |
| Hunyuan3D-2.1 | 41.15 | 44.65 | 2.77 | 2.50 |
| Direct3D-S2 | 41.62 | 44.79 | 3.21 | 3.64 |
| **Pixal3D** | **42.11** | **45.04** | **4.91** | **4.74** |

**关键观察**：
- **Fidelity 4.91**（5 分制）：用户感知到接近 reconstruction 的 fidelity
- **Quality 4.74**：在 fidelity 提升的同时 quality 也最优
- Uni3D / ULIP2 提升 marginal，说明 semantic alignment 没有显著变化——这正是预期，因为 pixel-aligned 主要改善 geometric fidelity 而非 semantic understanding

### 7.3 Multi-View Evaluation (Table 3)

| Method | View | CD↓ | EMD↓ | F-Score↑ |
|--------|------|-----|------|----------|
| VGGT | 2 | 613.55 | 19.60 | 9.57 |
| VGGT | 4 | 881.53 | 21.71 | 10.25 |
| VGGT | 6 | 2791.10 | 25.33 | 9.67 |
| TRELLIS | 2 | 21.39 | 2.40 | 43.68 |
| TRELLIS | 6 | 18.13 | 2.16 | 46.02 |
| **Pixal3D** | 2 | 5.27 | 1.13 | 64.94 |
| **Pixal3D** | 6 | 4.16 | 1.00 | 69.04 |

**关键观察**：
- **VGGT 性能反而随 views 增加恶化**——这是 feed-forward reconstruction 的经典问题（多 views 累积误差，且无生成 prior 补全）
- **Pixal3D 性能随 views 增加单调改善**：完美体现了 generative reconstruction 的核心——views 越多越接近 reconstruction
- **F-Score 69.04 vs 46.02**：在 6 views 下相对 TRELLIS 提升 50%

### 7.4 Ablation Study (Figure 8)

两个关键 ablation：
1. **Without feature upsampling**：依赖 37×37 的 coarse DINOv2 patch tokens → fine details 丢失，misalignment 出现
2. **Without back-projection**（替换为 cross-attention）：训练 slow to converge, unstable，最终 fidelity 显著降低

第二个 ablation 尤其重要——它证明 back-projection 不仅是 design choice，更是**训练稳定性的关键**。直觉上，pixel-aligned generation 是一个更约束的问题空间，cross-attention 在这个空间下需要学习一个已经 implicitly 定义的 correspondence，反而引入 redundancy 和 instability。

---

## 8. 与相关工作的深度对比

### 8.1 ReconViaGen [Chang et al. 2025]

ReconViaGen 将 VGGT features 注入 canonical-space generator——仍然在 canonical space 生成，correspondence 通过 cross-attention。

**Pixal3D 的优势**：彻底建立 explicit 2D-3D correspondence，避免 canonical-pose generation 的 fidelity loss。

### 8.2 CUPID [Huang et al. 2025a]

CUPID 联合建模 canonical 3D object 和 camera pose——需要预测 pose。

**Pixal3D 的优势**：完全避免 camera estimation 的 brittleness，直接在 camera space 生成。

### 8.3 RaySt3R [Duisterhof et al. 2025]

RaySt3R 进行 ray-based novel-view prediction 并 fuse multi-view estimates——本质是 reconstruction。

**Pixal3D 的优势**：generative model 补全 occluded regions，输出直接 usable 3D asset。

### 8.4 Gen3R [Huang et al. 2025b]

Gen3R 耦合 feed-forward reconstruction backbone 和 diffusion——仍依赖 reconstruction backbone 的 pose estimation。

**Pixal3D 的优势**：pure generative backbone，pixel-aligned conditioning 替代 pose estimation。

Reference:
- ReconViaGen: https://arxiv.org/abs/2510.23306
- CUPID: https://arxiv.org/abs/2510.20776
- RaySt3R: https://arxiv.org/abs/2506.05285
- Gen3R: https://arxiv.org/abs/2601.04090

---

## 9. 局限性与未来方向

### 9.1 当前 limitations

1. **Pixel-level noise sensitivity**：imperfect segmentation boundaries 会被 back-projected 并放大成 small geometric artifacts。这是 pixel-aligned 的双刃剑——精确对应也意味着精确传递 noise。

2. **Multi-view 依赖已知 camera poses**：假设 camera parameters 准确已知。这在实际应用中需要额外的 SfM 或 pose estimation 步骤。

3. **Scene generation 依赖 2D inpainting**：Qwen-image-edit 在复杂 occlusions 下可能引入 errors，这些 errors 会传递到 3D generation。

### 9.2 未来方向（与论文方向一致 + 个人延伸）

1. **Texture & Material Synthesis**：pixel-aligned paradigm 特别适合 appearance fidelity——可以直接 back-project color / BRDF 信息。

2. **3D Editing via 2D Pixel Manipulation**：pixel-aligned 意味着 2D image editing 直接对应 3D 修改——这是一个 powerful interaction paradigm。

3. **Video-based 3D Scene Generation**：temporal consistency + pixel-aligned = high-fidelity world building。

4. **Self-supervised multi-view pose estimation**：将 camera pose estimation 集成到 framework 内部，避免 external 依赖。

5. **Dynamic scene / 4D generation**：pixel-aligned paradigm 自然扩展到 temporal dimension。

---

## 10. 我的 Intuition 总结

Karpathy，我认为这篇论文的核心贡献可以提炼为一个 insight：

**Correspondence 不应该是 learned，而应该是 given by geometry。**

Cross-attention 在 image-to-3D 中的角色本质上是让网络"重新发现"投影几何——这是一个浪费 capacity 的做法。Pixal3D 通过 back-projection 把这个先验 hardcoded 进去，让网络专注学习真正需要 generative prior 的部分（occluded regions 的 plausible completion）。

这背后的更深层启示：**当我们可以用 explicit geometry 时，不要让网络去 learn 它**。这是 classical CV 与 deep learning 融合的一个 elegant 例子——classical multi-view geometry 的 back-projection + modern diffusion 的 generative prior = best of both worlds。

更广泛地，这个 paradigm 对其他 generative tasks 也有启示——只要存在 explicit input-output correspondence（如 depth estimation 中的 image-to-depth），就应该优先利用它，而非让 cross-attention 从头学习。

这与你在 "Software 2.0" 论文中的思想一致：明确的、可形式化的先验应该 hardcoded，而 fuzzy 的、需要数据驱动的部分才应该 learned。Pixal3D 把 correspondence hardcode 进 backbone，把 completion 留给 learned prior——这是一个 excellent 的 Software 1.0 / 2.0 混合范例。

Project page: https://ldyang694.github.io/projects/pixal3d/

---

## References (Key Papers)

1. **Pixal3D** (本论文): SIGGRAPH 2026
2. **Direct3D-S2** (backbone): https://arxiv.org/abs/2505.17412
3. **TRELLIS**: https://arxiv.org/abs/2412.01506
4. **Hunyuan3D-2.1**: https://arxiv.org/abs/2506.15442
5. **TripoSG**: https://arxiv.org/abs/2502.06608
6. **DINOv2**: https://arxiv.org/abs/2304.07193
7. **DUSt3R**: https://arxiv.org/abs/2403.11279
8. **VGGT**: https://arxiv.org/abs/2503.11651
9. **MoGe**: https://arxiv.org/abs/2410.19115
10. **SAM3**: https://arxiv.org/abs/2511.16719
11. **SAM3D**: https://arxiv.org/abs/2511.16624
12. **Atlas** (back-projection 先驱): https://arxiv.org/abs/2003.10432
13. **NeuralRecon**: https://arxiv.org/abs/2104.00681
14. **ReconViaGen**: https://arxiv.org/abs/2510.23306
15. **CUPID**: https://arxiv.org/abs/2510.20776
16. **NAF**: https://arxiv.org/abs/2511.18452
