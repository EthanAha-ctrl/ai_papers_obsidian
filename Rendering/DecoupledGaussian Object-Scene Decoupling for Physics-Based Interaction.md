---
source_pdf: DecoupledGaussian Object-Scene Decoupling for Physics-Based Interaction.pdf
paper_sha256: e89f6f8242f3367f7a29bc978cb93d8f4fd63e786c78cddc1a2268c4683271a1
processed_at: '2026-08-18T04:41:54-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，如果抛开学术黑话，这篇 paper 其实就是在回答一个特别接地气的问题：**你拿手机拍个视频，视频里有个雕塑放在桌子上，你怎么把这雕塑“抠”下来，放到电脑里重新模拟它掉在地上碎掉，或者被车撞飞的过程？**

这事儿听起来简单，但在现在的 3D 技术里是个巨大的空缺。下面我用大白话给你拆解一下。

---

## 1. 痛点在哪？为什么以前做不到？

现在的 3D 重建技术（比如 Gaussian Splatting, 简称 GS）很厉害，输入一堆多视角照片，就能重建出特别逼真的 3D 场景。但是，如果你要在电脑里对这个场景做物理仿真（比如受力、碰撞、掉落），就会遇到一个极其尴尬的问题：

**物体和接触面是“粘”在一起的。**

你想啊，雕塑压在桌子上，从相机的视角看，雕塑底部和桌面接触的地方是被遮挡的。重建出来的 3D 模型里，雕塑和桌子在这里就是一坨连在一起的点云或者高斯球。

这个时候你想做物理仿真，给雕塑一个力，让它翻倒——做不到。因为雕塑的底面是缺的（没观察到），桌子的表面也是缺的（被挡住了），而且它们在 3D 数据结构上是相连的。你一算物理，雕塑要么穿透桌子，要么因为底部不完整直接在重力下“坍塌”成一滩烂泥。

以前的 paper 怎么搞？
- **PhysGaussian**: 假设你输入的物体本身是完美的、全视角的，内部也是填满的。这在野拍视频里根本不存在。
- **VR-GS**: 用 2D 的修图软件（LaMa）把视频里物体挡住的桌子部分“P”回去。但 2D 修图是没 3D 几何信息的，P 出来的桌子深度是错的，物理仿真时物体还是会穿模或者乱飞。（Paper 里的 Fig.2 给了 LaMa 翻车的例子）

所以，DecoupledGaussian 的核心 mission 就是：**把物体和场景从 3D 几何层面彻底“解绑”（Decouple），把各自缺失的 3D 结构补全，最后丢给物理引擎去模拟。**

---

## 2. 这套系统怎么干活？

整个 pipeline 分三步走，逻辑极其清晰。

### Step 1: 把场景拍扁，分清敌我 (3D Gaussians Preparation)

首先，你输入一段视频，系统先用 COLMAP 算出相机位姿，然后用 Gaussian Splatting 重建出 3D 场景。但原版 3DGS 重建出来的东西是一坨坨各向异性的“高斯雪球”，这些雪球的中心其实并不在物体的真实表面上。如果直接拿雪球中心去补洞，几何肯定不准。

所以作者用了他们之前的绝活 **PGSR (Planar-based Gaussian Splatting)**。核心 idea 是：加一个正则项，强迫每个高斯球变扁，变成像一块块小饼干一样贴在物体表面上。

公式长这样：
$$ \mathcal{L}_{planar} = \| \min(s_1, s_2, s_3) \|_1 $$

变量解释：
- $s_1, s_2, s_3$: 高斯球的三个轴向的缩放比例。
- $\min(\cdot)$: 取最小值。
- 直觉：强迫最短的那根轴趋近于 0，3D 雪球就退化成了 2D 平面。平面的法向量 $n_g$ 自然就是物体的表面法向。

有了贴在表面上的“饼干”，就能算出准确的 **Unbiased Depth（无偏深度）**。
$$ D(p) = \frac{d(p)}{n(p) K^{-1} p'} $$
变量解释：
- $p$: 像素坐标。
- $K$: 相机内参矩阵。
- $p'$: 像素齐次坐标。
- $n(p)$: $\alpha$-blending 融合后的法向量。
- $d(p)$: 融合后的平面到相机的距离。
- $K^{-1} p'$: 从相机射向该像素的光线方向。
- $D(p)$: 真实的物理深度。这个深度极其重要，后面补洞全靠它。

接着，用 SAM2 给视频打 2D mask，训练一个小 MLP，把 3D 空间里的高斯球分成“物体”和“场景”两类。现在，敌我分清了。

### Step 2: 切开并补全几何 (Object-Scene Restoration) —— 全文灵魂

现在把物体和场景从 3D 空间里切开。切开之后，物体底部有个大窟窿，场景上也有个物体形状的坑。怎么补？

作者的杀手锏：**Joint Poisson Fields（联合泊松场）**。

这里有个很深数学等价性：**泊松表面重建 等价于 计算 Winding Number（环绕数）场**。环绕数就是告诉你一个点是在物体内部（值为 1）还是外部（值为 0）。边界就是 0.5。

所以，补洞的思路就是：解泊松方程，把那些残缺的内部点找回来。

具体流程：
1. 分别对物体 O 和场景 S 解泊松方程，求出各自的内部指示函数 $\chi_O$ 和 $\chi_S$。用了一个 $128^3$ 的网格，只要 20 秒。
2. 解决冲突：因为物体原本是放在场景上的，它们算出来的内部空间肯定有一块是重叠的。怎么办？**信场景，不信物体**。因为场景（比如桌面）是平的，几何更简单、算得更准；物体底部被遮挡得厉害，算出来的边界可能往外膨胀了。所以重叠的部分，算作物体外部。
3. 提取点：从场景的泊松场里提网格，从物体的泊松场里提内部密集点 $P_O$（物理仿真要用）。

**但是这里有个巨大的坑**：物体的表面高斯中心不准啊！直接拿去解泊松方程，重建出来的物体表面坑坑洼洼。

**破局点：Proxy Points（代理点）**。
作者用刚才算的无偏深度 $D(p)$，结合多视角，跑了个 TSDF Fusion（一种传统的深度图融合算法）。这样融合出来的点云，极好地逼近了物体的真实表面。然后把这些点再分割出来，作为泊松重建的输入。Fig.5 的消融实验证明，用高斯中心重建出来的物体跟渣一样，用 Proxy Points 就极其平滑。

**最后一个问题：泊松重建太爱“平滑”了**。它为了把窟窿补上，经常补得过多，导致物体体积变大，生出了一堆在所有训练视角里都看不见的“游离结构”。

**终极杀招：UNCE (Unilateral Negative Cross Entropy，单边负交叉熵)**。
多视角雕刻：我拿 SAM2 的 2D mask 当监督，把泊松重建出来的多余部分“削”掉。损失函数是：
$$ \mathrm{UNCE}(p) = -(1 - M_O^{GT}(p)) \log(1 - \mathbb{1}_O(p)) $$
变量解释：
- $M_O^{GT}(p)$: 像素 $p$ 的真实 mask，1 代表这里是物体。
- $\mathbb{1}_O(p)$: 当前模型渲染出来在这个像素的 opacity（不透明度）。
- $1 - M_O^{GT}(p)$: 这是个 indicator，只在“非物体区域”为 1。

**为什么叫单边？** 这个 loss 只惩罚负样本！也就是：只有在背景区域，模型却渲染出了物体（多余的结构），我才惩罚你。但在物体区域，模型没渲染出来（因为被挡住了），我不惩罚你。

Intuition: 泊松方程补出来的部分，可能有些是合理猜测（虽然视角里看不到，但物体本来就有），有些是明显多余的。如果我两边都惩罚，就会把合理的猜测也削掉，物体就又破了个洞。UNCE 保证了只削明显的错误，不破坏封闭表面的完整性。这个设计极其精妙。

最后，给物体内部填上 dense 粒子 $P_O$，纹理用 2D 修图稍微 refine 一下。场景的坑也用扁平高斯贴在刚才重建的网格上。至此，完美解绑。

### Step 3: 物理仿真

拿着补全好的、内部有 dense particles 的物体和场景，扔进 MLS-MPM 物理引擎。
- MLS-MPM (Material Point Method): 把物体当成一堆拉格朗日粒子，每一步把质量和动量传到欧拉网格上算受力，再传回粒子更新位置。
- 场景作为 Dirichlet 边界：包含场景高斯球的网格节点，速度强制设为 0。物体撞上去就会停下，不会穿透。
- 用 RANSAC 找平面的法向量，把重力轴对齐，就可以开始掉落、碰撞、甚至改变材质参数（比如让物体融化）了。

---

## 3. 实验结果：按在地上摩擦

作者搞了个 Decoupling Benchmark，拿真实的重建场景，用 PyBullet 放物体进去渲染作为输入。Ground truth 是原始无物体的场景和完美填充的物体。

**场景修复对比:**
| Methods | PSNR ↑ | FID ↓ | CD ($10^{-3}$) ↓ |
|---|---|---|---|
| GScream | 17.82 | 42.28 | 44.00 |
| VR-GS | 25.13 | 58.50 | 6.41 |
| **Ours** | **27.32** | **32.07** | **4.40** |

Chamfer Distance (CD) 是衡量几何准确度的硬指标。DecoupledGaussian 的 CD 是 4.40，比依赖 2D 修图的 VR-GS (6.41) 和 GScream (44.00) 好太多了。而且速度极快，场景修复只要 1 分钟，GScream 要 70 分钟。

**交互仿真对比 (Motion-FID 越低越好):**
| Methods | Motion-FID ↓ |
|---|---|
| PhysGaussian | 54.79 |
| GIC | 47.47 |
| **Ours** | **31.69** |

物理运动的逼真度远超同类工作。因为别人连底子（几何）都没打好，物理引擎一跑就露馅。

**最酷的 Feature (Fig. 7)**：因为物体和场景彻底解绑了，你可以把 A 场景里的卡车抠出来，扔到 B 场景的自行车棚里做物理碰撞模拟。这种 Cross-scene 的组合能力，对做 robotics 数据增强的人来说简直是刚需。

参考链接：
- DecoupledGaussian 主页: https://wangmiaowei.github.io/DecoupledGaussian.github.io/
- PhysGaussian: https://github.com/Xharlor/PhysGaussian
- PGSR (拍扁高斯): https://arxiv.org/abs/2406.06521
- MLS-MPM 物理引擎: https://yuanming-hu.com/publications/2018/mls-mpm/
- SAM2 (分割): https://github.com/facebookresearch/sam2
- Taichi (物理仿真神器): https://github.com/taichi-dev/taichi

---

## 4. 一句话总结 Intuition

这篇 paper 的核心哲学就是：**不要用 2D 图片修图的结果去反推 3D 几何，2D 没有深度信息，反推出来的 3D 在物理仿真里必死无疑。正确的做法是：利用 3D 高斯自身的几何先验（扁平化、表面法向）、数学上的闭合表面先验（泊松环绕数），再结合多视角一致性，直接在 3D 空间里把残缺的几何补全。**

Geometry first, texture second. 只要把几何补对了，物理引擎就能自然地算出正确的运动。这是把基于视频的 3D 重建真正推向“可用、可交互”的 4D 资产的大路。

---

# DecoupledGaussian: 从视频中解开物体与场景的物理仿真系统

Andrej，这篇 paper 我觉得很值得深挖，因为它其实是在回答一个看似简单却被大家回避的问题：**给定一段野拍视频（in-the-wild video），重建出物体和它接触的表面，然后让物体能"真正脱离"接触面去做 Newtonian 物理 simulation**。这个 capability gap 之前在 GS-based simulation 社区其实没有人系统性地解决——大家要么做合成物体（全视角观测），要么做 elastic jittering（物体还在接触面上抖）。DecoupledGaussian 的核心 insight 是：**先 decouple，再 simulate；geometry 修复优先于 texture 修复**。

项目主页：https://wangmiaowei.github.io/DecoupledGaussian.github.io/
3DGS 原文：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
MLS-MPM (Hu et al. 2018)：https://yuanming-hu.com/publications/2018/mls-mpm/
PhysGaussian：https://github.com/Xharlor/PhysGaussian
SAM2：https://github.com/facebookresearch/sam2
Screened Poisson Reconstruction (Kazhdan & Hoppe)：https://www.poissonrecon.com/

---

## 1. Motivation: 为什么 Decoupling 是缺失的一环

先 build intuition。看现有的 GS-based simulation 工作：

- **PhysGaussian [Xie et al., CVPR 2024]**: 把 MLS-MPM 直接作用到 Gaussian kernels 上，做 elastic deformation。但它假设物体有"完整内部"，是从 synthetic 数据或者 well-captured 数据开始的，没法处理野拍视频中物体被接触面遮挡的 broken surface。
- **GIC [Cai et al., NeurIPS 2024]**: 用 isotropic Gaussians + coarse-to-fine density field 填充内部，但还是假设物体本身完整。
- **VR-GS [Jiang et al., SIGGRAPH 2024]**: 用 XPBD + bounding mesh，场景修复依赖 2D inpainting（LaMa），导致 geometry 不准确，simulation 时物体"穿过"场景。
- **GScream [Wang et al., ECCV 2024]**: 做的是 object removal 后的场景修复，用单张 reference image + monocular depth prior，view consistency 差。

**核心矛盾**：物体 O 和场景 S 在物理上是接触的，video 重建会把它们 merged 在一起；而 Newtonian simulation（尤其是 detach motion）需要它们是独立的 closed surfaces，且各自内部有 dense particles 用于 continuum simulation。所以必须先 decouple。

Fig.2 展示了 LaMa 和 PhotoRoom 这种 2D inpainting 在 3D 上的失败案例：artifacts、frame 之间 texture inconsistent。作者的关键论点是：**2D inpainting 在 RGB pixel space 做事，但 3D geometry position 信息丢失了；你必须先在 3D geometry 层面修复，再考虑 texture**。这其实呼应了早期 NeRF 时代大家发现的问题——geometry 和 appearance 的解耦对下游 task 至关重要。

---

## 2. Pipeline 全景

整个 system 可以分成四个 stage：

1. **3D Gaussians Preparation**（Sec. 4.1）：用 Planar-based GS (PGSR) 重建场景 + 用 SAM2 做 Gaussian-level segmentation
2. **Object-Scene Restoration**（Sec. 4.2）：核心创新——Joint Poisson Fields + Proxy Points + UNCE carving + Gaussian Restoration
3. **Interactive Simulation**（Sec. 4.3）：MLS-MPM continuum simulation，把 restored Gaussians 当作 Lagrangian particles

我下面逐 stage 拆开讲。

---

## 3. Stage 1: Planar-based Gaussian Splatting

### 3.1 为什么不用 vanilla 3DGS

原始 3DGS [Kerbl et al., SIGGRAPH 2023] 用 anisotropic 3D Gaussians，optimization 只用 photometric loss（L1 + D-SSIM），会陷入 local optima，导致：
- **Floaters**: 在空气中飘着的孤立 Gaussian kernels
- **Surface deviation**: Gaussian center $k_g$ 不在真实表面附近，因为 α-blending 的"blended rendering"会让 surface 看起来对，但 kernel center 实际上散乱分布

这对后续的 Poisson reconstruction 是致命的，因为 Poisson 需要准确的 oriented points 作为 input。

### 3.2 PGSR 的核心 idea

PGSR [Chen et al., 2024, arXiv:2406.06521] 的 idea 是：**强制把每个 Gaussian 压成 planar-like（扁的）**，让最小 scale dimension 趋近于 0，这样 normal 方向就是 shortest axis，自然就有了 surface normal。

正则项是：

$$\mathcal{L}_{planar} = \| \min(s_1, s_2, s_3) \|_1$$

其中 $s_1, s_2, s_3$ 是 covariance matrix $\Sigma_g$ 的 singular values（也就是 scaling matrix $S_g = \text{diag}(s_1, s_2, s_3)$ 的对角元）。

回忆 3DGS 的 covariance 分解：

$$\Sigma_g = R_g S_g S_g^T R_g^T$$

- $R_g \in SO(3)$: rotation matrix，决定 Gaussian 的朝向
- $S_g = \text{diag}(s_1, s_2, s_3)$: scaling matrix，三个轴向的 scale

如果 $\min(s_1, s_2, s_3) \to 0$，Gaussian 就退化成 plane，normal 就是 $R_g$ 对应那个最小 scale 轴的列向量。

### 3.3 Unbiased Depth

有了 plane-like Gaussians，就可以定义**unbiased depth**。每个 Gaussian 的深度不是简单取 center 的 z 坐标，而是用 plane 到 camera 的距离：

$$d_g = \| n_g^T d_g' \|$$

- $n_g$: Gaussian 的 normal（shortest axis 方向的单位向量）
- $d_g'$: 从 camera center 到 Gaussian center $k_g$ 的向量
- $d_g$: plane 到 camera center 的垂直距离（带符号信息通过 viewing direction disambiguate）

然后做 α-blending：

$$D(p) = \frac{d(p)}{n(p) K^{-1} p'}$$

- $p$: pixel 坐标
- $p'$: $p$ 的 homogeneous coordinate
- $K$: camera intrinsic matrix（3×3）
- $n(p)$: α-blended normal
- $d(p)$: α-blended plane-to-camera 距离
- $K^{-1} p'$: pixel 对应的 view ray 方向
- $n(p)^T K^{-1} p'$: 投影到 view ray 上的分量
- 最终 $D(p)$: 真正的 metric depth，而不是 vanilla GS 那种 blended center depth（biased）

这个 unbiased depth 后面要拿来做 TSDF fusion 生成 proxy points。

---

## 4. Stage 1.5: Gaussian Segmentation

用 Gaussian Grouping [Ye et al., ECCV 2024] 的方法：每个 Gaussian 附加 32 维 semantic affinity feature $\xi_g$。一个单层 MLP $\zeta$ 把 α-blended feature $\xi(p)$ 映射到 $C$ 个 class probability。

训练数据来自 SAM2 [Ravi et al., 2024] 给的 multi-view 2D segmentation mask。Loss 是 cross-entropy：

$$\mathcal{L}_{seg} = -\sum_p \sum_c y_{p,c} \log \zeta(\xi(p))_c$$

其中 $y_{p,c}$ 是 pixel $p$ 在 class $c$ 上的 SAM2 label。还加了 local feature smoothing [Cen et al., 2024] 来减少邻近 Gaussians 的 boundary artifacts。

### Intuition

为什么需要 Gaussian-level segmentation 而不是 image-level？因为后面要做 Poisson reconstruction，必须知道**哪些 Gaussian 属于 object O，哪些属于 scene S**。SAM2 给的是 2D supervision，但最终学到的 affinity 是 3D 一致的（因为 MLP input 是 3D feature）。

---

## 5. Stage 2: Object-Scene Restoration（核心创新）

这是整篇 paper 的灵魂。我分成四个 sub-module 讲。

### 5.1 Decoupling: 把 O 和 S 分开

用户点击一个位置 $p$（在 object 上），系统找 α-blended feature $\xi(p)$，然后用 affinity feature matching 找到 object 的所有 Gaussians $\mathcal{G}_O$，再用 KNN 移除附近的 residual artifacts。

但此时 $\mathcal{G}_O$ 是 broken 的（contact 面被遮挡），$\mathcal{G}_S$ 也少了一块（被 object 占据的区域）。要修复两者。

### 5.2 Joint Poisson Fields（最重要的创新）

**Key insight**: Poisson surface reconstruction 等价于 winding number field construction [Feng et al., SIGGRAPH 2023; Xu et al., SIGGRAPH 2023]。Winding number 是一个 indicator function 的光滑近似：内部 = 1，外部 = 0，边界 = 0.5。

作者提出 **Joint Poisson Fields** $\mathcal{W}$：分别对 O 和 S 做 screened Poisson reconstruction，然后用 heterogeneous constraints 解决它们的 intersecting regions。

三步走：

**(1) 独立重建 indicator functions**

对 scene 和 object 分别求 indicator function $\chi_S$ 和 $\chi_O$：

- $\chi_S > 0.5$ 表示 S 内部
- $\chi_S < 0.5$ 表示 S 外部
- $\chi_O$ 类似

用 Adaptive Multigrid Solvers [Kazhdan, 2024]，grid size $128^3$，每个 Poisson field 20 秒以内搞定。

数学形式上，screened Poisson reconstruction [Kazhdan & Hoppe, 2013] 是求解：

$$\min_\chi \int_\Omega \|\nabla \chi(x) - \vec{v}(x)\|^2 dx + \lambda \int_P (\chi(x) - s(x))^2 dx$$

- $\chi(x)$: 待求的 indicator function
- $\vec{v}(x)$: 由 oriented points 的 normals 定义的 vector field（在 surface 附近指向外，远处为 0）
- $P$: oriented points 集合
- $s(x)$: 在 $P$ 上的 values（通常是 1，表示 in-out boundary）
- $\lambda$: screening weight，控制 fidelity to input points
- 第一项：让 gradient 匹配 vector field，从而让 $\chi$ 在 surface 附近从 0 跳到 1
- 第二项：让 $\chi$ 在 input points 上接近 1，正则化

**(2) Transform $\chi_O$ 到 S 的 canonical frame**

把 $\chi_O$ 从 $\mathcal{W}_O$ 变换到 $\mathcal{W}_O^s$（在 S 的 canonical grid 空间）。然后处理 conflict region：

$$\text{conflict} = \{ x \mid \chi_S(x) > 0.5 \cap \chi_O^s(x) > 0.5 \}$$

即 O 和 S 重叠的 voxel。**作者选择保留 S（discarding $\chi_O^s$ 在 conflict 区域）**，因为 S 的 geometry 更简单、更可靠（planar surface），而 O 的 Poisson reconstruction 可能 over-extend。

这个选择很 sensible，因为 planar surface（地板、桌面）的 winding number 比物体的 winding number 更容易精确算。

**(3) 提取 mesh 和 dense points**

- 从 $\mathcal{W}_S$ 用 marching cubes [Lorensen & Cline, 1987] 提取 mesh，re-meshing [Pietroni et al., 2009]，再用 $P_O$ 的 scaled bounding box 裁出 patch mesh $\mathcal{M}_S$
- 从 $\overline{\mathcal{W}}_O^s$ 提取 dense interior points $P_O$（用于 MLS-MPM 的 Lagrangian particles）

### 5.3 Proxy Points $\mathcal{P}_O$

这里有个 critical observation：**Gaussian centers $\{k_g\}_{g \in \mathcal{O}}$ 不能直接作为 Poisson reconstruction 的 input**，因为 α-blending 让 center 偏离真实 surface（看 Fig. 5 的 ablation）。

Proxy points 的生成流程：

1. 渲染 RGB 和 unbiased depth map $D$ 在所有 training views 下
2. 用 zero-opacity trick 获取 projected mask $M_O^{proj}$：把 $\mathcal{G} \setminus \mathcal{O}$ 的 opacity 设为 0，渲染得到的 mask
3. 用 masked depth $D \circ M_O^{proj}$ 做 TSDF fusion [Newcombe et al., 2011]：把所有 training view 的 depth 集成
4. TSDF 结果里混入了 $\mathcal{G} \setminus \mathcal{O}$ 的 points（boundary smearing [Wang & Morris, WACV 2024]），用 nearest neighbor search 把 $\xi_g$ feature 从 raw Gaussian 继承到 proxy points，再做 segmentation 清除

最终 proxy points $\mathcal{P}_O$ 比 Gaussian centers 准确得多，作为 Poisson reconstruction 的 input。

### 5.4 UNCE (Unilateral Negative Cross Entropy)

Poisson reconstruction 倾向于 over-smooth，会引入 geometry expansion（超出观察 viewpoint 之外的部分）。作者用 multi-view carving 来 refine。

定义 isometric dense object Gaussians $\mathcal{G}_O$：
- centers: $\{k_g \in P_O \cup \mathcal{P}_O\}$
- 每个 opacity $\sigma_g = 0.1$
- isometric covariance: $\bar{\Sigma}_g = \text{diag}(s_g^2, s_g^2, s_g^2)$
- $s_g = c \left(\frac{3}{4\pi}\right)^{1/3}$，其中 $c$ 是 Poisson grid cell length（保证 Gaussian 体积等于 voxel 体积）

UNCE loss：

$$\text{UNCE}(p) = -(1 - M_O^{GT}(p)) \log(1 - \mathbb{1}_O(p))$$

- $M_O^{GT}(p)$: 来自 SAM2 的 object 2D ground truth mask，1 表示 object，0 表示非 object
- $\mathbb{1}_O(p)$: 当前 $\mathcal{G}_O$ 在 pixel $p$ 上 α-blended opacity
- $1 - M_O^{GT}(p)$: 非 object 区域的 indicator
- $\log(1 - \mathbb{1}_O(p))$: 当前 model 在非 object 区域预测的 "没有 object" 的 log-likelihood

**关键：这是 "unilateral" 的——只惩罚负样本（非 object 区域被预测为有 object 的部分），不惩罚正样本**。为什么？因为正样本（object 区域）本来就在 multi-view 上有 occlusion，可能某些 view 看不到，但 Poisson 已经 fill 了，这些 fill 的部分不应该被 push 掉。只 push 那些在所有 view 都明确是 background 但 Poisson over-extend 出来的部分。

**这是非常 elegant 的设计**。如果用普通 binary CE，会把所有 view 都看不到的 object 部分（合理 fill 的部分）也 push 掉，这就破坏了 closed surface 假设。UNCE 保留了 fill 的合理性，只移除明显的 over-extension。

每 100 iterations 清除 $\sigma_g \leq 0.05$ 的 Gaussians。

### 5.5 Gaussian Restoration

最终 Gaussian 集合：
- **Object** $\mathcal{G}_O$: 来自 $\mathcal{P}_O$（外表面）和 $P_O$（内部 dense），用 UNCE carve 过
- **Scene patch** $\mathcal{G}_S$: 把 flattened Gaussians bind 到 $\mathcal{M}_S$ 上，最小 scale $\epsilon$ along normal 方向

Texture refinement：
- Object 的 SH coefficients 从 TSDF 集成的颜色初始化，Gaussian-weighted interpolation from 15 nearest neighbors
- Scene patch 的 opacity 和 SH 从原始 broken $\mathcal{S}$ 的 nearest neighbors 初始化，用 LaMa [Suvorov et al., 2022] 的 inpainted images 做 texture supervision（**只用于 texture，不用于 geometry**）
- 每 iteration 用 random background 渲染（避免 scene 其他区域的影响）

---

## 6. Stage 3: MLS-MPM Simulation

### 6.1 MLS-MPM 回顾

MLS-MPM [Hu et al., SIGGRAPH 2018] 是 Material Point Method 的优化版本，用 Moving Least Squares 来近似 velocity gradient。

**Continuum 离散化**：
- Lagrangian particles $p$：携带 mass、momentum、deformation gradient $F_p$、affine matrix $C_p$
- Eulerian grid：每 $\Delta t$ 时间步，做 P2G → Grid Operation → G2P 三步

**Per-step 更新**：
1. **P2G**: particle mass 和 momentum 转移到 grid
2. **Grid Operation**: 
   - 计算 PK1 stress（first Piola-Kirchhoff stress），依赖 deformation gradient $F^E$
   - Forward Euler 更新 grid velocity $v$
3. **G2P**: 把 grid velocity 插值回 particle，更新 particle position

**Affine update**:
$$F_p^{n+1} = (I + \Delta t C_p^n) F_p^n$$

- $F_p^n$: time step $n$ 时 particle $p$ 的 elastic deformation gradient（描述从 reference configuration 到 current configuration 的形变）
- $C_p^n$: affine matrix，是 $\nabla v$ 的一阶近似（MLSM 用 MLS basis function 计算的）
- $\Delta t$: 时间步长
- $I$: 单位矩阵
- 这个公式的 intuition：deformation 由 velocity gradient 累积，$C_p$ 近似 $\nabla v$，所以 $\Delta t \cdot C_p$ 是一 timestep 的形变增量

**Material parameters**:
- Young's modulus $E$: 描述 material 的 stiffness（线性 elastic regime）
- Shear modulus $\mu$: 描述 shape change 的 resistance（与 Lame 参数相关）
- 这些参数影响 PK1 stress 的计算

### 6.2 Dirichlet Boundary Condition

为了模拟 restored scene $\mathcal{G}_S$ 作为 sticky boundary（让物体撞到场景表面后停下来）：

> "set the velocities of grid nodes containing Gaussians from the restored scene $\mathcal{S}$ to zero during Grid Operation stage in MLS-MPM"

这是 Dirichlet boundary condition [Bazilevs & Hughes, 2007] 的简单实现：包含 scene Gaussian 的 grid node velocity 强制为 0。

### 6.3 Gravity Alignment

用 RANSAC [Li et al., 2017] 估计 planar scene 的 normal，然后旋转所有 $\{k_g, \Sigma_g\}$ 让 z 轴对齐到 gravity direction。

**旋转 SH 的细节**：view-dependent spherical harmonics coefficients 要通过 Wigner D-matrices [Wigner, 2012] 旋转。Wigner D-matrix 是 SO(3) 在 spherical harmonics basis 上的 representation，对于 degree $l$ 的 SH，旋转矩阵是 $(2l+1) \times (2l+1)$ 的 $D^l_{m,m'}(R)$。

---

## 7. Experiments: 全面分析

### 7.1 Dataset

测试用 9 个 real-world 场景：
- Mip-NeRF360 [Barron et al., 2022]: BICYCLE, GARDEN, BONSAI, ROOM, KITCHEN
- Tanks&Temples [Knapitsch et al., 2017]: TRUCK
- Deep Blending [Hedman et al., 2018]: PLAYROOM
- LERF [Kerr et al., 2023]: FIGURINES
- Instruct-NeRF2NeRF [Haque et al., 2023]: BEAR

### 7.2 Baselines

四个 SOTA baselines，每个都做了合理 adaptation：
1. **PhysGaussian [Xie et al., CVPR 2024]**: anisotropy regularization + bounding-box based opacity field for interior filling
2. **GIC [Cai et al., NeurIPS 2024]**: isotropic Gaussians + coarse-to-fine density field
3. **VR-GS [Jiang et al., SIGGRAPH 2024]**: XPBD + bounding mesh，scene restoration 用 LaMa
4. **GScream [Wang et al., ECCV 2024]**: monocular depth estimation [Ke et al., 2024] + single reference image

### 7.3 User Study 结果（Table 1）

| Method | SRQ ↑ | Time ↓ |
|---|---|---|
| GScream | 1.94 | ~70m |
| VR-GS | 2.12 | ~7m |
| **Ours** | **3.48** | **~1m** |

| Method | ORQ ↑ |
|---|---|
| PhysGaussian | 1.40 |
| GIC | 1.60 |
| **Ours** | **4.03** |

| Method | ISF ↑ |
|---|---|
| VR-GS(S) + PhysGaussian(O) | 1.50 |
| Ours(S) + PhysGaussian(O) | 2.60 |
| Ours(S) + GIC(O) | 2.73 |
| **Ours(S) + Ours(O)** | **4.35** |

**关键 takeaways**:
- Scene restoration 时间：我们 1 分钟 vs GScream 70 分钟（70× 加速）vs VR-GS 7 分钟
- 完全用我们的方法 (S+O) 在 simulation fidelity 上达到 4.35/5.0
- 用我们的 S + 别人的 O，fidelity 大幅下降——说明 object restoration 是关键

### 7.4 Quantitative Benchmark（Table 2）

作者构造了一个新的 decoupling benchmark：从 PEGASET [Meyer et al., IROS 2024] 和 BlenderNeRF [Chen et al., 2024] 拿真实重建场景，用 PyBullet [Coumans & Bai, 2016] 放物体进去，render 作为输入。Ground truth 是原始 scene（无 object）+ filled object。

**Scene Restoration**:

| Method | PSNR ↑ | LPIPS ↓ | FID ↓ | CD × 10⁻³ ↓ |
|---|---|---|---|---|
| GScream | 17.82 | 0.56 | 42.28 | 44.00 |
| VR-GS | 25.13 | 0.32 | 58.50 | 6.41 |
| **Ours** | **27.32** | **0.30** | **32.07** | **4.40** |

CD (Chamfer Distance) 是衡量 geometry 的关键 metric，我们 4.40 vs VR-GS 6.41 vs GScream 44.00——基本上差了一个数量级。

**Object Restoration**:

| Method | PSNR ↑ | LPIPS ↓ | FID ↓ | CD × 10⁻³ ↓ |
|---|---|---|---|---|
| PhysGaussian | 24.46 | 0.07 | 227.60 | 0.53 |
| GIC | 26.62 | 0.06 | 201.91 | 0.73 |
| **Ours** | **30.32** | **0.04** | **138.75** | **0.17** |

Object CD 我们 0.17，比 PhysGaussian 的 0.53 好 3×。FID 从 227 → 138，提升明显。

**Interactive Simulation**:

| Method | PSNR ↑ | LPIPS ↓ | FID ↓ | Motion-FID ↓ |
|---|---|---|---|---|
| PhysGaussian | 19.48 | 0.37 | 112.55 | 54.79 |
| GIC | 20.90 | 0.31 | 134.56 | 47.47 |
| w/o dense $P_O$ | 21.19 | 0.29 | 98.19 | 48.39 |
| w/o Proxy $\mathcal{P}_O$ | 21.08 | 0.30 | 90.26 | 36.01 |
| w/o $\mathcal{W}$ (Joint Poisson) | 20.97 | 0.30 | 96.16 | 42.27 |
| **Ours** | **21.33** | **0.29** | **86.98** | **31.69** |

Motion-FID 用 RAFT [Teed & Deng, 2020] 提取 optical flow，colorize，再算 FID——衡量 motion realism。

### 7.5 Ablations 关键发现

1. **Dense Interior Points $P_O$**: 没有内部 dense particles 的物体在 gravity 下会 collapse（Fig. 9 Top-Middle）。这是 MLS-MPM 必须——continuum simulation 需要 volumetric particle representation，不能只有 surface。

2. **Proxy Points $\mathcal{P}_O$**: 用 Gaussian centers 直接做 Poisson reconstruction 质量差（Fig. 5）。Proxy points 通过 TSDF fusion 提供更准确的 surface samples。

3. **Joint Poisson Fields $\mathcal{W}$**: 独立做 Poisson（不做 conflict resolution）会留下 intersection artifacts（Fig. 9 Top-Right 红色区域）。

4. **UNCE**: 不加 UNCE 的话 Poisson over-extension 产生的 artifacts 会让 simulation 时物体有奇怪的悬空部分（Fig. 10 Top）。

5. **Planar-based GS**: vanilla GS 即使做 opacity filtering ($\sigma_g \leq 0.02$) 也会留下 floaters，限制物体运动（Fig. 10 Bottom 红圈）。

---

## 8. 几点 Intuition 联想

### 8.1 Winding Number vs. Poisson

作者引用了 [Feng et al., SIGGRAPH 2023] 和 [Xu et al., SIGGRAPH 2023] 的等价性结果。这个等价性其实很深：winding number field 在离散点云上做积分近似，本质上就是 screened Poisson 方程的解。这件事在 ACM ToG 上有详细推导。直觉上：indicator function $\chi$ 的 gradient 在 surface 附近应该集中（delta function），所以 $\nabla \chi$ 是一个集中在 surface 上的 vector field；把这个 vector field 作为 Poisson 方程的 RHS，解出来的 $\chi$ 自然就是 indicator function。

参考: https://www.cs.cmu.edu/~kmcrane/Projects/SignedDistance/

### 8.2 为什么 UNCE 只用 negative samples

这让我联想到 GAN 的 discriminator 早期 issue：如果 positive samples 太强，generator collapse。类似地，UNCE 保留了 Poisson 的 "hallucination" 部分（合理 fill），只 push 明显错误的 over-extension。这是非常 elegant 的 asymmetric loss 设计。

更深的 intuition：在 multi-view setting 下，positive samples 是高度 incomplete 的——某些 object region 在所有 view 都被遮挡。所以不能强求 model 在所有 view 都 fit positive samples；只能用 negative samples 做"边界约束"。

### 8.3 MLS-MPM 选型的 intuition

为什么选 MLS-MPM 而不是经典 MPM？MLS-MPM 用 Moving Least Squares 基函数，可以做 affine particle state update（$C_p$ 矩阵），让每个 particle 携带 velocity gradient 的局部近似。这意味着 energy/momentum conservation 更好，noise 更小，可以少用 grid resolution。

PhysGaussian 也是用 MLS-MPM，所以这里其实是 inherit 了 PhysGaussian 的 simulator，但加上了 decoupling 这一上游 capability。

### 8.4 与 GaussianEditor / Gaussian Grouping 的关系

DecoupledGaussian 在 segmentation stage 用了 Gaussian Grouping [Ye et al., ECCV 2024]，在 restoration stage 借鉴了 GaussianEditor [Chen et al., CVPR 2024] 的 inpainting 思路。但 key difference：GaussianEditor 做 object removal 后的 scene 补全，依赖 2D inpainting 提供 geometry guidance；DecoupledGaussian 反过来——**用 GS 自身的 geometry prior 提供 guidance，2D inpainting 只补 texture**。这个 reversal 是整个 system 工程的关键。

### 8.5 跟 mesh hole filling 的对比

[Hernandez-Bautista & Melero, 2024] 用 2D diffusion model 做 mesh hole filling。DecoupledGaussian 是 first 在 GS 上做这件事，但不用 2D diffusion 来补 geometry——这是 engineering trade-off：用 closed surface + winding number prior 来 fill，比用 2D diffusion 更 robust，因为 2D diffusion 没有 3D consistency 保证。

### 8.6 VR-GS 的失败 case 直觉

VR-GS 用 LaMa 修复 scene 后做 simulation，Fig. 6 显示物体在 BEAR 和 BONSAI 场景里"穿过" scene surface。Intuition 是：LaMa 给出的 RGB 上看起来 ok，但 3D geometry 实际是错的（深度不连续），MLS-MPM 把这些"假 Gaussians"当作 Dirichlet boundary，boundary 本身就错位了，物体撞上去自然会穿。

### 8.7 Cross-scene interaction 的 implication

Fig. 7 Bottom 展示 TRUCK（来自 Tanks&Temples）放到 BICYCLE 场景里 simulation。这暗示了**一个很 power 的 capability**：因为 object 和 scene 都 decoupled 了，可以 mix-and-match——把任意 object 放到任意 scene 里 simulate。这对 robotics simulation、VR content creation 是巨大的——可以做 in-the-wild 数据增强。

这种 capability 之前只有 synthetic assets 才有（Isaac Gym / Orbit [Mittal et al., 2023]），现在可以从 video 里直接 derived。

### 8.8 Limitations 的真正本质

作者在 Discussion 里提到：
- Multi-object contact configurations 没处理
- High-frequency texture completion 难
- Fine-grained object components decoupling 难

我补充一个 intuition：multi-object contact 本质上是 combinatorial 问题——N 个 object 两两 contact，Poisson reconstruction 要做 N+1 个 indicator function 的 conflict resolution。Joint Poisson Fields 的 formulation 在 N=1 时 elegant，N>1 时可能要重新设计 priority graph。

### 8.9 跟神经辐射场仿真的联系

PIE-NeRF [Feng et al., CVPR 2024] 和 PAC-NeRF [Li et al., ICLR 2023] 做 NeRF-based physics simulation，也用 MPM。但 NeRF 是 implicit representation，做 simulation 需要 explicit mesh extraction，每 frame 都要重新 extract，慢。Gaussian Splatting 是 explicit representation，可以直接当作 Lagrangian particles——这是为什么 PhysGaussian 比 PIE-NeRF 快很多。

DecoupledGaussian 站在 PhysGaussian 肩上，但把"输入是什么"扩展了：从 well-captured synthetic 到 in-the-wild captured。

---

## 9. 总结性 intuition

我觉得这篇 paper 最 powerful 的 insight 是：**3D geometry restoration 应该用 3D prior，而不是 2D prior 的 lift**。整个 system 围绕"closed surface + winding number + multi-view consistency"展开：

- PGSR 提供 planar geometry prior（让 Gaussian 自身成为 surface 表示）
- Joint Poisson Fields 提供 closed surface prior（winding number 自然 fill hole）
- Proxy Points 提供 multi-view depth consistency prior（TSDF fusion）
- UNCE 提供 multi-view observation prior（asymmetric carving）

这四个 prior 叠加起来，让系统不需要 2D diffusion model 也能做高质量 3D restoration——这是工程上的胜利，也是 modeling choice 上的胜利。

物理 simulation 部分相对 incremental（直接 reuse PhysGaussian 的 MLS-MPM），但 decoupling 是 prerequisite，没有 decoupling，simulation 就只能在 synthetic / well-captured data 上做。这篇 paper 真正打开了 "video → 4D interactive asset" 的 pipeline，对 VR / robotics / autonomous driving 都有意义。

相关参考链接汇总：
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- MLS-MPM Taichi 实现: https://github.com/yuanming-hu/taichi_elements
- PhysGaussian: https://github.com/Xharlor/PhysGaussian
- PGSR: https://arxiv.org/abs/2406.06521
- Gaussian Grouping: https://github.com/lkeab/gaussian-grouping
- SAM2: https://github.com/facebookresearch/sam2
- Screened Poisson Reconstruction: https://www.poissonrecon.com/
- Winding Number on Discrete Surfaces: https://www.cs.cmu.edu/~kmcrane/Projects/WindingNumber/
- Taichi (MLS-MPM 实现 framework): https://github.com/taichi-dev/taichi
- NVIDIA Warp (paper 用的 simulation framework): https://github.com/NVIDIA/warp
- LaMa inpainting: https://github.com/advimman/lama
- Mip-NeRF 360 dataset: https://jonbarron.info/mipnerf360/
- Tanks&Temples dataset: https://www.tanksandtemples.org/
- GaussianEditor: https://github.com/buaacyw/GaussianEditor
- VR-GS: https://github.com/zarklew/VR-GS
- GScream: https://github.com/W-Ted/GScream
- PEGASET: https://rmurai.eu.org/Pegaset/
- PyBullet: https://pybullet.org/

如果你有兴趣 dive deeper，我建议你从 PhysGaussian 开始读（因为 MLS-MPM 部分基本是 inherit 的），然后看 PGSR（planar-based GS 是关键 enabler），最后看 winding number 那篇 [Feng et al., SIGGRAPH 2023]——理解 winding number 和 Poisson 的等价性，能让你对整个 restoration pipeline 的数学本质有更深的把握。
