---
source_pdf: World Tracing Generative Pixel-Aligned Geometry.pdf
paper_sha256: 77597867040ef400d689e3fb86444f6ca7815458b00777e0525399f333a28c90
processed_at: '2026-08-13T05:09:40-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我用大白话讲一遍这篇 paper 的"故事"，把技术细节背后的直觉讲清楚。

---

## 这领域一直有个尴尬

你拍一张照片，想把它变 3D。两条路：

**第一条路：单目深度估计**（Depth Anything、MoGe、π³）。这类模型给你每个像素一个 depth，很准，pixel-aligned——你看到的每个像素都对应一个 3D 点。但问题是：它只给你**正面那层皮**。桌子后面是什么？墙后面是什么？一概不知。你拿到的是个"面具"，不是完整的 3D 世界。

**第二条路：image-to-3D generation**（TRELLIS、SAM 3D、Zero-1-to-3）。这类模型给你一个完整的 mesh 或 Gaussian，前后都有。但问题是：它生成在 **canonical frame** 里——一个标准的物体坐标系，和你拍的这张照片的相机坐标系对不上号。你拿到一个漂亮的 3D 椅子，但不知道这个椅子在你这张照片的相机里位置在哪、朝向如何。想拿它做"在原图场景里插入一个新物体"？做不到，因为 2D pixel 和 3D point 之间没有对应关系。

下游 pipeline 真正想要的是：**在 input 相机坐标系下，既 faithful（正面准）又 complete（背面有）的几何**。既要能 reproject 回原图做编辑，又要能在大角度 view rotation 时有背面支撑。

这两条路长期分离，各自 hack 各自的 corner case，没人把它们统一起来。

---

## WT 的核心 insight：一根 ray 上的所有交点本来就在同一根射线上

关键观察：从相机中心出发，穿过某个像素的那根 ray，它先碰 visible surface（layer 0），然后继续往前，碰 occluded surface（layer 1），再继续，碰 layer 2……这些交点本来就在同一根射线上，按 front-to-back 排好序了。

为什么以前没人把它们放进一个 tensor？因为大家把 "faithful visible surface" 和 "generative occluded completion" 当成两个 task，分别用 regression 和 generation 做。

WT 说：这就是一个 task——**在 image grid 上预测 L 张 depth 图**，每张是一层 ray-surface intersection。第一张是 visible surface（像 MoGe 那样），后面几张是 occluded surface（像 image-to-3D generation 那样）。把它们打包成 $\mathbf{X} \in \mathbb{R}^{L \times H \times W \times 3}$，一个 tensor 搞定。

这样做的直接好处：
- **pixel alignment 保留**：每层都在 input image grid 上，2D pixel 和 3D point 一一对应
- **camera pose 保留**：所有坐标都在 input camera space，不需要 canonical frame
- **intrinsics 不用输入**：网络直接输出 XYZ，需要 $K$ 时从 layer-0 pointmap closed-form 拟合

这是 representation-level 的选择，不是 model-level 的 trick。

---

## 为什么以前没人这么做？因为深层 supervision 太稀疏

真正的工程难题在这里：depth peeling 出来的深层非常 sparse。论文 Table 5 统计：

| Layer | 有效像素占比 |
|---|---|
| L0 | 8.14% |
| L1 | 7.90% |
| L2 | 2.53% |
| L3 | 1.74% |
| L4 | 0.72% |
| L5 | 0.60% |

L5 只有 0.6% 的像素有真实交点。LaRI（最接近的 predecessor）的做法是让网络同时回归 XYZ 和预测 per-layer validity mask。结果 mask head 因为严重 imbalanced 直接 collapse——预测"全无效"就行，loss 最低。而且 mask gradient 和 XYZ gradient 之间 EMA cosine similarity 是 -0.19，两个监督信号在 early decoder block 互相打架。

---

## WT 的 trick：forward-fill，把 sparse 变 dense

WT 的解法非常简洁。如果某个像素在 layer $\ell$ 没有真实交点，就拿前面最近的 valid layer 的点填进来：

$$\mathbf{x}_\ell(\mathbf{u}) \gets \mathbf{x}_{\ell'}(\mathbf{u}), \quad \ell' = \max\{k < \ell : \mathbf{x}_k(\mathbf{u}) \text{ valid}\}$$

这样每个 valid 像素在每一层都有 target。不需要 mask head。

关键 insight：**filling 进来的是已有表面的重复，不是新几何**。几何本身没变，只是 supervision 变 dense 了。网络如果想表达"这条 ray 已经终止了"（比如桌子前面是实的，后面没有更深表面），可以让深层 collapse 到前面那层——预测同一个点。这比预测一个 binary mask 容易得多。

这是个 elegant 的 reformulation：把 "regression + classification" 耦合问题变成 "pure dense regression" 问题。诊断显示原本 valid 的深层 pixel 和继承自前面层的 pixel，误差在同一个 regime，说明 filling 没引入 systematic bias。

这个思想其实和 DINOv2 的 dense image-level supervision、DiffusionDet 的 dense prediction、NeRF 的 occupancy collapse-to-surface 是同源的——**避免 head-based sparse supervision，用 dense target + loss mask 解决问题**。

---

## 架构上的关键：ray-wise attention

WT-DiT 是个 1.7B 参数的 flow-matching DiT。frozen MoGe ViT-L 做 image encoder（继承 in-the-wild visual prior），48 个 DiT block 做 decoder。

最有意思的设计是 **three-way factorized attention**。如果对 $L \times P = 6 \times 1296 = 7776$ 个 token 做全 attention，没结构、贵。作者把一个 block 拆成三种 attention shape 交替：

1. **Layer-wise**：每个 layer 内部做 2D attention with 2D RoPE。这是标准图像 self-attention，捕获 2D 空间结构。
2. **Ray-wise**：同一个像素的 6 个 layer token 之间做 attention。**这是 WT 区别于 VGGT 的核心**。VGGT 处理多视角时用 frame/global 交替，WT 处理 multilayer 时显式引入 **ray axis**，让深层 token 能"看见"自己对应的 layer 0，不会 drift away from visible surface。
3. **Global**：所有 token 全 attention，捕获 object/scene-level context，周期性出现。

ray-wise attention 是真正让 multilayer 几何 coherent 的关键。没有它，深层会和 visible surface 脱节，预测出飘在空中的点云。作者在 Sec 4.3 ablation 里验证了这点。

这个 pattern 可能 generalize——任何"同一位置多个 sample"的问题（multi-view、multilayer、time、BRDF samples）都可以加一个 axis-specific attention。

---

## 训练上的关键：layer-aware noise schedule

flow matching 的标准做法是所有 token 共享一个 timestep $t$ 的采样分布。但 WT 的不同 layer 不确定性 profile 完全不同：

- **Layer 0** 强约束于 input image，更像是 deterministic reconstruction——需要在 $t$ 小处多采样
- **Layer 1-5** 只被间接约束，更像是 conditional generation——需要在 $t$ 中间处多采样

一个 timestep 分布照顾不过来。Table 4 ablation 显示：

| Schedule | L0 CD-L2 | L1-L5 CD-L2 | All CD-L2 |
|---|---|---|---|
| Plateaued logit-normal（偏小 t） | 0.021 | 0.033 | 0.031 |
| Standard logit-normal（偏中 t） | 0.023 | 0.028 | 0.027 |
| **Mixture（两阶段 curriculum）** | **0.020** | **0.025** | **0.024** |

作者的 curriculum：
- **Phase (i) 早期**：Layer 0 用 plateaued logit-normal，深层用 standard logit-normal，各自在适合自己的 noise regime 训
- **Phase (ii) 稳定后**：整个 stack 切换到共享 timestep，从两个分布的 50/50 mixture 采样

这个 insight 对其他"混合 task"的 diffusion model 应该有启发——image inpainting 中 mask 内 vs mask 外、video generation 中 conditioned frames vs free frames，都是同一个 continuum 上的不同点，可能都需要 per-region-aware noise schedule。

---

## Mix-training：一个 mask gate 同时吃两种数据

传统深度模型绑死一个 supervision regime：单目深度只看 visible layer，image-to-3D 只看完整 mesh。WT 因为是 image-grid tensor，可以同时训两种数据。

公式 (4)：

$$A^{(\ell)} \leftarrow A^{(\ell)} (1 - b_{\text{single}} \mathbb{1}[\ell \geq 1])$$

$b_{\text{single}} \in \{0, 1\}$ 标记这个 sample 是不是只提供 single-layer supervision。效果：single-layer 样本（real RGBD photo）只在 layer 0 贡献 loss，layer 1-5 自动 mask 掉。

实操：60% multilayer 3D assets + 40% RGBD single-layer（ScanNet、MegaDepth、BlendedMVS、Hypersim、Taskonomy 等 12 个数据集）。Table 6 显示加上 RGBD mix 后，WT-S 在 NYU、ETH3D indoor/outdoor 上接近甚至超过专门做单目深度的 MoGe-2、Pi3X。50 步 mix-trained WT-S 在 ETH3D indoor AbsRel 0.0332 全场最好。

关键是只跑了 50K mix-training iterations，validation loss 还在降——这只是 lower bound。而且 multilayer capability 完全保留，因为 mix-training 只 mask 深层 loss，不动 multilayer supervision。

这个 mix-training 的简单性很美——一个 mask gate 公式就把 single-layer RGBD 数据塞进 multilayer 训练，不需要新 head、新 embedding、新 regime token。这是 representation choice 的红利：选对了 output format，数据扩展几乎是 free 的。对比 LaRI 这种 mask-head 路线，吃 RGBD single-layer 数据非常困难。

---

## 结果：faithful 和 complete 不再 trade off

Table 1（object）：WT-O 的 layer-0 visible surface 比 Depth Anything 3、MoGe-2、π³ 都准。**生成 occluded geometry 没有拖累 visible surface，反而通过更好的几何 prior 让 layer 0 更 sharp**。完整几何的 Chamfer distance 也赢过所有 image-to-3D generator。

Table 2（scene）：WT-S 在 3D-FRONT 和 internal test set 上 visible surface 全面领先，All-L metrics 大幅超过 LaRI-scene。Fig 4 显示 MoGe-2 在生成图上会 bend walls/facades，WT-S 保留更直的平面结构。

Table 3（dynamic）：WT-D 在 Obj.-Val 和 Truebone 上大幅领先 GVFD、SS4D、ActionMesh。

Table 6（real-scene depth）：WT-S 在 NYU、ETH3D 上接近甚至超过专门做单目深度的方法。**这验证了核心 claim：multilayer generative training 不牺牲 visible-surface fidelity，反而提升 L0**。

---

## 下游：training-free 接上三个 pipeline

WT 的 representation 一旦训好，下游三个应用都是 training-free 的组合：

1. **Text-driven 3D scene editing**：2D editor 编辑图 → WT-S lift 原场景 → WT-O lift 编辑区域 → 因为都在同一 camera frame + pixel correspondence，closed-form composite。canonical-frame generator 做不到——它能生成 plausible asset，但不知道这个 asset 在 input camera 里位置在哪。

2. **Geometry-guided novel-view video synthesis**：image-to-video world model（VACE、TrajectoryCrafter）常用 depth map 做 conditioning，但单层 depth 一旦 camera 移动，新暴露区域要从头 hallucinate。WT 提供完整 multilayer geometry memory，沿 target trajectory rasterize 生成 dense multi-view depth guidance。object orbit 场景特别受益——背面已经在 WT 输出里。

3. **Pose-aligned TRELLIS hybrid**：TRELLIS 输出 mesh 好看但 canonical frame，reproject 不回 input image。WT 的 pixel-aligned point stack 已经带 recoverable intrinsics，voxelize 后喂给 TRELLIS Stage 2，替换最弱的 Stage-1 structure prior。Table 1 显示 WT-O* 全面超过纯 TRELLIS.2 和 VGGT-guided 的 ReconViaGen / LaS-Comp。

---

## 我觉得这篇 paper 真正的价值在哪

这篇 paper 的 contribution 不是某一个 trick，而是一个 **representation choice**：把单目深度估计和 image-to-3D generation 这两个长期分离的 task，统一成一个 image-grid 上的 multilayer pointmap tensor。

围绕这个 representation，作者把所有该解决的工程问题都解了：
- dense supervision → depth filling
- multilayer attention → three-way factorization with explicit ray axis
- layer identity → LayerFiLM
- 不同 uncertainty profile → layer-aware noise schedule
- 数据 scale → mix-training with single-layer RGBD
- generalization → frozen MoGe encoder

最被低估的点是 **pixel-aligned 作为 3D interface 的价值**。canonical-frame generator 输出虽然漂亮但和 input image 失联，下游 pipeline 用不上。WT 这种 "camera-space + pixel correspondence + recoverable intrinsics" 的输出格式，是个更好的 2D-to-3D contract。这是个 interface-level innovation，意义可能比模型本身更大。

还有个意外发现：object model 没 scene training 也能在 scene image 上产出合理几何；没显式 multi-object augmentation 也 robust to multi-object image。因为 WT 每个 ray 一个 stack，多个 disconnected object 自然共存于同一 tensor。这暗示 next step 应该是 unify object / scene / dynamic into one model，而不是维护 specialist。

---

参考资源：
- WT 项目（推断）: https://world-tracing.github.io/
- MoGe: https://wangrc.github.io/MoGePage/
- LaRI: https://arxiv.org/abs/2504.18424
- TRELLIS: https://trellis3d.github.io/
- VGGT: https://vgg-t.github.io/
- Depth Anything v2: https://depth-anything-v2.github.io/
- π³ (Pi3X): https://arxiv.org/abs/2507.13347
- Flow matching: https://arxiv.org/abs/2210.02747
- SD3 / DiT: https://arxiv.org/abs/2403.03206
- DINOv2: https://arxiv.org/abs/2304.07193
- LayerScale (CaiT): https://arxiv.org/abs/2103.17239
- Depth peeling (Everitt): https://en.wikipedia.org/wiki/Order-independent_transparency#Depth_peeling
- Layered Depth Images (Shade 1998): https://dl.acm.org/doi/10.1145/280814.280882
- World Labs: https://www.worldlabs.ai/
- FLUX.2 representation comparison: https://bfl.ai/techblog/representation-comparison/
- VACE: https://arxiv.org/abs/2503.17001
- TrajectoryCrafter: https://arxiv.org/abs/2503.05638
- ReconViaGen: https://arxiv.org/abs/2510.23306
- LaS-Comp: https://arxiv.org/abs/2602.18735

---

# World Tracing: 把可见与不可见几何统一到一个 Pixel-Aligned Tensor 里

这篇 paper 来自 World Labs（Mildenhall、Lassner 等人）和 UIUC（Narendra Ahuja 组），第一作者 Hao Zhang 同时也是 MagicPose4D、S3O 的作者。核心贡献：提出一个 pixel-aligned multilayer geometry representation，把**单目深度估计的 faithfulness**和**image-to-3D generation 的 completeness**统一在同一个 camera-space tensor 里，让单张图既能精确还原 visible surface，又能 plausible 地补全 occluded geometry。

项目主页和资源：
- 论文 arXiv（推断）: https://arxiv.org/abs/2506.18828 (近似 ID)
- MoGe（WT 用的 frozen encoder）: https://wangrc.github.io/MoGePage/
- LaRI（最接近的 predecessor）: https://arxiv.org/abs/2504.18424
- TRELLIS（被 hybrid 的 mesh 生成器）: https://trellis3d.github.io/
- VGGT（多视角 baseline）: https://vgg-t.github.io/
- Flow matching 原始 paper: https://arxiv.org/abs/2210.02747
- DiT / SD3（AdaLN、logit-normal schedule）: https://arxiv.org/abs/2403.03206
- DINOv2: https://arxiv.org/abs/2304.07193
- Depth peeling (Everitt 2001): NVIDIA whitepaper，http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.21.7579&rep=rep1&type=pdf
- Layered Depth Images (Shade 1998): https://dl.acm.org/doi/10.1145/280814.280882
- World Labs: https://www.worldlabs.ai/

---

## 1. 问题本质：faithful 和 complete 的二选一

3D-from-image 领域长期有两个分离的 paradigm：

| 范式 | 代表 | 优点 | 缺点 |
|---|---|---|---|
| 单目 depth / pointmap | Depth Anything v1/v2, MoGe, MoGe-2, π³, UniDepth | pixel-aligned、faithful、可吃大量 RGBD 数据 | 只能预测 visible surface，背后是空的 |
| Image-to-3D generation | TRELLIS, SAM 3D, LGM, DreamGaussian, Zero-1-to-3 | 完整 mesh / Gaussians | canonical frame，丢掉 input pose 和 pixel correspondence |

下游 pipeline 真正想要的是**在 input camera 坐标系下的 faithful+complete 几何**——既要能 reproject 回原图（做编辑、insertion、view synthesis），又要能补齐看不到的背面（做大角度 view rotation）。

WT 的核心 insight 是：**faithfulness 和 completeness 本来就在同一根射线上**——一根 ray 从 camera 出发，先碰 visible surface（layer 0），然后穿过表面继续往前，再碰 occluded surface（layer 1, 2, …）。把它们放进同一个 `L × H × W × 3` tensor，把"补全"变成"在 image grid 上生成第 ℓ 张 depth 图"。

---

## 2. Representation：一个 image-grid 上的 multilayer pointmap

### 2.1 主公式

$$\mathbf{X} \in \mathbb{R}^{L \times H \times W \times 3}, \quad \mathbf{X}[\ell, \mathbf{u}] = \mathbf{x}_\ell(\mathbf{u})$$

变量含义：
- $L$：层数（论文用 $L=6$）
- $H, W$：input image 的高宽
- $3$：camera-space XYZ（不是 disparity、不是 metric depth）
- $\ell \in \{0,1,\ldots,L-1\}$：layer index，按 front-to-back 排序
- $\mathbf{u} = (u, v)$：input pixel 坐标
- $\mathbf{x}_\ell(\mathbf{u})$：从 camera center 出发穿过 pixel $\mathbf{u}$ 的那条 ray 上，第 $\ell$ 个 ray-surface intersection 的 3D 点

关键 property：
- **不需要 intrinsics 输入**：网络直接输出 XYZ，intrinsics 留到下游用 closed-form 从 layer-0 拟合（App. B 公式 (9)）。
- **保持 pixel-to-3D correspondence**：因为每层都生活在 input image grid 上，2D 像素和 3D 点一一对应。
- **object 和 scene 共享同一个架构**：只是 scale normalization 不同。

### 2.2 Depth filling：避免 per-layer mask collapse

经典的 depth peeling 在深层会变得非常 sparse——很多 ray 根本没碰到 6 个表面。Table 5 的统计触目惊心：

| Layer | Mean valid pixel % |
|---|---|
| L0 | 8.14% |
| L1 | 7.90% |
| L2 | 2.53% |
| L3 | 1.74% |
| L4 | 0.72% |
| L5 | 0.60% |

如果像 LaRI 那样让网络同时回归 XYZ 和预测 per-layer validity mask，深层 mask 监督严重 imbalanced（L5 只有 0.6% positive），mask head 会 trivially 预测"无效"，把深层抑制掉。作者实测 EMA cosine similarity 在 depth gradient 和 mask gradient 之间是 $-0.19$（early decoder block），两个监督信号互相打架。

WT 的解法是 forward-fill：

$$\mathbf{x}_\ell(\mathbf{u}) \gets \mathbf{x}_{\ell'}(\mathbf{u}), \quad \ell' = \max\{k < \ell : \mathbf{x}_k(\mathbf{u}) \text{ is valid}\}$$

如果 pixel $\mathbf{u}$ 在 layer $\ell$ 没有真实 hit，就拿前面最近的 valid layer 的点填进来。结果：
- **每条 valid ray 在每一层都有一个 dense target**——监督不再稀疏
- **几何不变**：filling 进来的是已有表面的重复，不是新几何
- 网络如果想表达"这条 ray 已经终止了"，可以让深层 collapse 到前面那层（预测同一个点）

这是一个非常 elegant 的 reformulation：把"regression + classification"耦合问题，变成"pure dense regression"问题。诊断显示原本 valid 的深层 pixel 和继承自前面层的 pixel，误差在同一个 regime，说明 filling 不会引入 systematic bias。

### 2.3 Scale normalization

object/dynamic 用 per-channel z-score：

$$\tilde{\mathbf{x}} = (\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}, \quad \boldsymbol{\mu}, \boldsymbol{\sigma} \in \mathbb{R}^3$$

$\boldsymbol{\mu}, \boldsymbol{\sigma}$ 是整个训练 corpus 算出来的 per-channel 均值标准差，只算一次。

scene 用 per-sample log-median（因为房间 depth 可能从 0.5m 到 50m，跨两个数量级）：

$$\tilde{z} = \ln(z/m), \quad \tilde{x} = \text{sign}(x)\ln(1 + |x|/m), \quad \tilde{y} = \text{sign}(y)\ln(1 + |y|/m)$$

$m$ 是这张 sample 所有 valid pixel 的 median depth。这个变换是 monotone 的，所以 front-to-back 顺序保留。

### 2.4 Invalid pixel 处理

input 的 alpha mask 外（比如 object 的背景、scene 的 sky）是 invalid。在 noisy geometry token 里直接灌 max noise：

$$\mathbf{x}_t^{\text{net}}(\ell, \mathbf{u}) = A(\mathbf{u}) \cdot \mathbf{x}_t(\ell, \mathbf{u}) + (1 - A(\mathbf{u})) \cdot \boldsymbol{\epsilon}(\ell, \mathbf{u}), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

$A(\mathbf{u}) \in \{0,1\}$ 是 layer-0 alpha，broadcast 到所有层和 XYZ 通道。endpoint loss 也用 $A$ 把 invalid pixel mask 掉。这样网络被训练成"对 invalid pixel 视而不见"，推理时只在 valid region 输出有意义的几何。

### 2.5 Monotonicity penalty

软性保证 layer 之间的 front-to-back 顺序：

$$\mathcal{L}_{\text{mono}} = \frac{1}{L-1} \sum_{\ell=0}^{L-2} \mathbb{E}_\mathbf{u}\left[\text{relu}\left(z_\ell(\mathbf{u}) - z_{\ell+1}(\mathbf{u})\right)^2\right]$$

$z_\ell(\mathbf{u})$ 是 layer $\ell$ 在 pixel $\mathbf{u}$ 的 normalized depth。one-sided penalty：顺序对了就是 0，错了就惩罚。$\lambda_{\text{mono}} = 0.1$。作者说这个 penalty 主要稳定 early training，对 final quality 影响很小——是个 structural prior 而不是 hard constraint。

### 2.6 Intrinsics 恢复（App. B）

$$\min_{f_x, f_y, c_x, c_y} \sum_\mathbf{u} \left\| (u, v) - \left(\frac{f_x X}{Z} + c_x, \frac{f_y Y}{Z} + c_y\right) \right\|^2$$

$(X, Y, Z) = \mathbf{x}_0(\mathbf{u})$ 是 layer-0 的预测点。这是个 least-squares，closed-form 可解。恢复出来的 $K$ 自洽于预测的 pointmap，对 downstream（object insertion、view synthesis）很关键。

---

## 3. WT-DiT 架构

### 3.1 整体 dataflow

```
RGBA (504×504×4)
   │
   ├─→ frozen MoGe ViT-L (DINOv2-based, ~300M)
   │      │
   │      └─→ last 4 blocks features → projection → f_I (pixel-aligned, 504/14 = 36 patches per side → 36×36 tokens)
   │
   └─→ noisy multilayer XYZ: x_t ∈ R^(L=6 × 504×504 × 3)
          │
          ├─→ patchify (14×14) → geometry tokens: (B, L=6, P=36×36=1296, D)
          │
          ├─→ at each (ℓ, u): concat(noisy_xyz, repeated f_I[u]) → linear → D
          │
          └─→ 48 DiT blocks (D=1536, 24 heads), each cycles through:
                layer-wise attention (B·L, P, D) + 2D RoPE
                ray-wise attention (B·P, L, D)         ← key ingredient
                global attention (B, L·P, D)
                + AdaLN(time t) + LayerFiLM(layer ℓ)
                [for WT-D: + temporal attention (B·L·P, T=16, D) + 1D RoPE on time]
          │
          └─→ linear projection → unpatchify → X̂_0 ∈ R^(6 × 504×504 × 3)
```

**总参数 1.7B，trainable 1.4B**（MoGe frozen）。

### 3.2 Three-way factorized attention（关键设计）

如果对 $L \times P = 6 \times 1296 = 7776$ 个 token 做全 attention，没结构、贵。作者把一个 block 拆成三种 attention shape 交替：

1. **Layer-wise** $(B \cdot L, P, D)$：把每个 layer 当成一张 36×36 的 image，内部做 2D attention with 2D RoPE over $(y, x)$。这是标准的图像 self-attention，捕获 2D 空间结构。
2. **Ray-wise** $(B \cdot P, L, D)$：同一个 pixel 的 6 个 layer token 之间做 attention。这是 WT 区别于 VGGT 的核心——VGGT 处理多视角时用 frame/global 交替，WT 处理 multilayer 时显式引入 **ray axis**，让深层不会 drift away from visible surface（Sec 4.3 ablation 验证）。
3. **Global** $(B, L \cdot P, D)$：所有 token 全 attention，捕获 object/scene-level context。代价高但只周期性出现。

这个 factorization 把 $O((LP)^2)$ 的全 attention 分摊成 $O(L \cdot P^2) + O(P \cdot L^2) + O((LP)^2)$ 三项，global 那项只在某些 block 出现，整体可控。

### 3.3 Layer-aware conditioning

每个 token 需要知道自己是第几层、当前 denoising timestep 是多少。

- **Layer FiLM**：$\mathbf{e}_\ell \xrightarrow{\text{MLP}} (\gamma_\ell, \beta_\ell)$，然后 $h \leftarrow \gamma_\ell \odot h + \beta_\ell$。这破坏了 layer permutation symmetry——否则 6 个层在网络看来完全等价。
- **AdaLN for time**：标准 DiT 的 AdaLN，所有 token 共享 time embedding 的 modulation。

注意这里**没有 learnable additive position token**，layer 身份完全靠 FiLM。这是个简洁的设计。

### 3.4 Temporal attention (WT-D)

对动态 clip（$T=16$）：
- flatten $(T \cdot L)$ 维度
- 每个 global-attention block 后插一个 temporal-attention block
- reshape 成 $(B \cdot L \cdot P, T, D)$，沿 $T$ 做 self-attention，1D RoPE on time axis
- 复用 host decoder 的 time AdaLN
- **LayerScale init $\gamma = 10^{-5}$**：从 WT-O checkpoint warm start 时，temporal block 残差接近 0，$T=1$ 输入下 bit-for-bit 复现静态行为，fine-tune 时慢慢 pick up temporal coupling

这个 init trick 和 CaiT (LayerScale) 同源：https://arxiv.org/abs/2103.17239

### 3.5 为什么用 MoGe ViT-L 而不是 DINOv2 直接

MoGe 本身是 monocular pointmap predictor，它的 ViT-L 已经在大量 RGBD/3D 数据上 fine-tuned 过，有强 in-the-wild visual geometry prior。WT 只训一个 projection layer 把 MoGe 最后 4 个 block 的 feature 投影到 decoder 维度，剩下 frozen。这让 WT 继承了 MoGe 在真实图上的 generalization（不像 TRELLIS 这类 canonical-frame generator 只见过 synthetic 3D assets，在真实图上容易 flip pose 或 hallucinate planes）。

---

## 4. Training：flow matching + layer-aware noise schedule

### 4.1 Flow matching objective

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t}\left[\left\| A \odot \left(F_\theta(\mathbf{x}_t^{\text{net}}, t, \mathbf{f}_I) - \mathbf{x}_0\right) \right\|_2^2\right]$$

- $\mathbf{x}_0 = \tilde{\mathbf{X}}$：clean endpoint（normalized target）
- $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：noise endpoint
- $t \sim p_{\text{train}}(t)$：timestep
- $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$：linear interpolation（rectified flow / flow matching）
- $A$：alpha mask broadcast 到 layer 和 XYZ 通道
- $F_\theta$ 输出 $\hat{\mathbf{x}}_0$（直接预测 endpoint，不是 velocity）
- velocity 推理时为 $\mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{f}_I) = (\mathbf{x}_t - F_\theta(\mathbf{x}_t^{\text{net}}, t, \mathbf{f}_I)) / t$
- 从 $t=1$（max noise）积分到 $t=0$，20 步 ODE

注意是 **pixel-space flow matching**，不用 VAE latent。这是受 "Back to basics" (Li & He 2025, ref [27]) 启发——denoising model 就该 denoise 原始信号。

### 4.2 Layer-aware noise schedule（关键 trick）

不同 layer 的不确定性 profile 完全不同：
- **Layer 0** 强约束于 input，更像 reconstruction——需要多在 $t$ 小处采样
- **Layer 1-5** 只被间接约束，更像 conditional generation——需要在 $t$ 中间处采样

单一 schedule 满足不了两边。作者设计了一个两阶段 curriculum：

**Phase (i)**（early training）：
- Layer 0：plateaued logit-normal（central plateau 更宽，$t$ 偏小，倾向 reconstruction）
- Layer 1-5：standard logit-normal（集中在 $t \approx 0.5$，倾向 generation）

**Phase (ii)**（layers 稳定后）：
- 整个 stack 用共享 timestep，从两个分布的 50/50 mixture 采样

Table 4 ablation 验证：

| Schedule | L0 CD-L2 | L1-L5 CD-L2 | All CD-L2 |
|---|---|---|---|
| Plateaued logit-normal | 0.021 | 0.033 | 0.031 |
| Logit-normal | 0.023 | 0.028 | 0.027 |
| **Mixture** | **0.020** | **0.025** | **0.024** |

Mixture 全面胜出——可见层和不可见层各得其所，又因为 phase (ii) 共享 timestep 保持了 layer coupling。

### 4.3 Mix-training：同时吃 multilayer 3D assets 和 single-layer RGBD

这是 WT 的另一个优势：传统深度模型绑死一个 supervision regime，WT 因为是 image-grid tensor，可以同时训两种数据。

公式 (4)：

$$A^{(\ell)} \leftarrow A^{(\ell)} (1 - b_{\text{single}} \mathbb{1}[\ell \geq 1])$$

- $b_{\text{single}} \in \{0, 1\}$：这个 sample 是否只提供 single-layer supervision
- $\mathbb{1}[\ell \geq 1]$：layer ≥ 1 时为 1
- 效果：single-layer 样本（real RGBD photo）只在 layer 0 贡献 loss，layer 1-5 自动 mask 掉

实操比例：$p_{\text{dojo}} = 0.6$（multilayer 3D assets），$p_{\text{rgbd}} = 0.4$（RGBD 单层）。RGBD 包含 12 个数据集：ScanNet v2、MegaDepth、BlendedMVS、ArkitScenes、Argoverse2、Waymo Open、Hypersim、Taskonomy 等。weight by $\sqrt{N_{\text{rows}}}$，real photo 再 1.5× boost。

Table 6 显示，加上 RGBD mix 后：
- NYU AbsRel: 0.0398 → 0.0382 → 0.0374（50 步）
- ETH3D indoor AbsRel: 0.0398 → 0.0345 → 0.0332（50 步，全场最好，超过 MoGe-2 和 Pi3X 的 0.0378）
- ETH3D outdoor AbsRel: 0.0533 → 0.0495 → 0.0451（50 步）

关键是只跑了 50K mix-training iterations，validation loss 还在降——这只是 lower bound。

### 4.4 Mask robustness

真实 mask 不完美（边界抖动、thin structures 丢失、内部 holes）。训练时 jitter alpha boundary，新暴露 / 新遮住的 pixel 用 nearest valid rendered geometry 做 pseudo-target，loss 降权。这让 checkpoint 在 TRELLIS hybrid 和 scene editing pipeline（用外部产生的 mask）下保持稳定。

### 4.5 数据规模

- ~300K objects, ~17M rendered views（Objaverse-XL, Objaverse, 3D-FUTURE, Toys4k, GSO, Truebones）
- Scene: 3D-FRONT + 内部 corpus
- Dynamic: ~16.8K clips
- 全部用 depth peeling 生成 multilayer supervision

### 4.6 优化

- 64 H100, batch 512, AdamW lr=1e-4 → 1e-5 cosine, 2000 warmup, wd=0.01, betas (0.9, 0.999)
- grad clip 1.0
- EMA decay 0.9995
- loss spike > 4× running EMA 跳过（保护 Adam 统计）
- 两阶段：196×196 / 100K iters → 504×504 / 100K iters
- WT-D 从 WT-O warm start + 50K iters

---

## 5. 实验：faithful 几何在三个 regime 都赢

### 5.1 Object (Table 1)

**Visible-surface depth（per-sample SSI alignment）**：

| Method | MAE↓ | RMSE↓ | AbsRel↓ | δ<1.25↑ |
|---|---|---|---|---|
| DA3 | 0.0703 | 0.0384 | - | 0.9998 |
| LaRI | 0.0920 | 0.9973 | 0.0440 | 0.9999 |
| Pi3X | 0.0506 | 0.0317 | 0.0172 | 0.9994 |
| MoGe-2 | 0.0368 | 0.0141 | - | 0.9995 |
| VGGT | 0.0370 | 0.0138 | - | 0.9995 |
| **WT-O** | **0.0243** | **0.0079** | - | **0.9996** |

WT-O 的 layer-0 不仅没被 multilayer generation 拖累，反而是最好的 visible-surface predictor。原因：generation 的多模态建模反而帮助 layer-0 学到更 sharp 的几何（不像 regression 平均化）。

**Full geometry（best-of-8 seeds）**：

| Method | L1↓ | L2↓ | F@0.01↑ | F@0.05↑ |
|---|---|---|---|---|
| TRELLIS.2 | 0.0566 | 0.00717 | 0.204 | 0.598 |
| SAM 3D | 0.0475 | 0.00501 | 0.203 | 0.675 |
| LaS-Comp | 0.0477 | 0.00739 | 0.225 | 0.677 |
| ReconViaGen | 0.0478 | 0.00528 | 0.228 | 0.677 |
| WT-O* (hybrid) | 0.0326 | 0.00530 | 0.321 | 0.808 |
| **WT-O (PC)** | **0.0213** | **0.00194** | **0.549** | **0.898** |

WT-O 直接输出 point cloud 已经比所有 mesh generator 好。和 TRELLIS.2 Stage 2 hybrid 之后（WT-O*）进一步提升——把 TRELLIS 最弱的 Stage-1 structure prior 换成 pixel-aligned WT point stack。

### 5.2 Scene (Table 2)

3D-FRONT held-out 50 samples：

| Method | MAE↓ | RMSE↓ | AbsRel↓ | L0 CD-L2↓ | L0 F@0.05↑ | All-L CD-L1↓ | All-L F@0.05↑ |
|---|---|---|---|---|---|---|---|
| VGGT | 0.0393 | 0.0568 | 0.0441 | 0.002889 | 0.7839 | - | - |
| DA3 | 0.0369 | 0.0521 | 0.0403 | 0.002893 | 0.7992 | - | - |
| LaRI-scene | 0.0319 | 0.0483 | 0.0359 | 0.001994 | 0.8576 | 0.0575 | 0.6671 |
| MoGe-2 | 0.0248 | 0.0373 | 0.0269 | 0.000973 | 0.9131 | - | - |
| Pi3X | 0.0234 | 0.0375 | 0.0260 | 0.000871 | 0.9206 | - | - |
| **WT-S** | **0.0102** | **0.0215** | **0.0114** | **0.000204** | **0.9867** | **0.0216** | **0.8951** |

WT-S visible surface 全面领先（即便和专门做深度的 MoGe-2/Pi3X 比）。All-L metrics 只对 multilayer 方法定义，LaRI-scene 在 All-L 上明显落后 WT-S。

Internal test set（200 samples，超出 3D-FRONT 分布）：WT-S 在 OOD 上也保持领先，而且 Fig 4 显示 MoGe-2 在生成图上会 bend walls/facades，WT-S 保留更直的平面结构。

### 5.3 Dynamic (Table 3)

Global CD-L2：

| Benchmark | GVFD | SS4D | AM | **WT-D** |
|---|---|---|---|---|
| ActionBench | 0.0879 | 0.0882 | 0.0243 | 0.0291 |
| Truebone | 0.0166 | 0.0145 | 0.0120 | **0.0063** |
| Obj.-Val | 0.0248 | 0.0249 | 0.0142 | **0.0034** |
| Mean | 0.0385 | 0.0381 | 0.0162 | **0.0105** |

WT-D 在 Obj.-Val 和 Truebone 大幅领先。ActionBench 输给 ActionMesh——因为 ActionBench 的 GT 是 tracked animated surface，正好匹配 AM 的 native output format，这是 evaluation bias 而非方法劣势。

### 5.4 Real-scene depth (Table 6)

WT-S 在 NYU Depth V2、ETH3D indoor/outdoor 上接近甚至超过专门做单目深度的方法。50 步 mix-trained WT-S 在 ETH3D indoor AbsRel 0.0332 全场最好。**这验证了 paper 的核心 claim**：multilayer generative training 不会牺牲 visible-surface fidelity，反而通过更好的几何 prior 提升 L0。

---

## 6. Downstream pipelines（不用 task-specific training）

WT 的 representation 一旦训好，下游三个应用都是 training-free 的组合。

### 6.1 Text-driven 3D scene editing

pipeline：
1. 用户用 2D editor（FLUX.2 / Inpainting）编辑 image
2. WT-S 把 original scene lift 成 3D
3. WT-O 把 edited region / inserted object lift 成 3D
4. 因为两个输出都在同一 camera frame、共享 pixel correspondence，**closed-form composite**

canonical-frame generator 做不到——它能生成 plausible asset，但不知道这个 asset 在 input camera 里位置在哪。

### 6.2 Geometry-guided novel-view video synthesis

image-to-video world model（VACE, TrajectoryCrafter, Wonderland）常用 depth map 做 conditioning。但单层 depth 一旦 camera 移动，新暴露区域要从头 hallucinate。WT 提供**完整 multilayer geometry memory**：

1. WT 预测完整 6 层几何
2. 沿 target trajectory rasterize，生成 dense multi-view depth guidance
3. frozen video model 拿到几何一致的信号，不用从零 hallucinate disoccluded region

object orbit 场景特别受益——背面已经在 WT 输出里。

### 6.3 Pose-aligned TRELLIS hybrid

TRELLIS 输出 mesh 好看但 canonical frame，reproject 不回 input image。WT 的 pixel-aligned point stack 已经带 recoverable intrinsics：

1. WT 预测 multilayer point stack
2. voxelize 这个 stack
3. 喂给 TRELLIS Stage 2（detail geometry + texture decoder）
4. 不重新训 TRELLIS，只替换 Stage-1 sparse structure

Table 1 显示 WT-O* 全面超过纯 TRELLIS.2 和 VGGT-guided 的 ReconViaGen / LaS-Comp。

---

## 7. 关键 design choices 的 ablation 和讨论

### 7.1 Depth filling vs mask prediction

LaRI 路线：regress XYZ + predict per-layer mask。问题：
- Mask 严重 imbalanced（L5 valid 0.6% vs L0 8.14%）
- Mask gradient 和 XYZ gradient EMA cosine sim 接近 0 或负（-0.19）
- Mask head collapse → 深层被抑制

WT 路线：dense XYZ regression，no mask head。Forward-fill 让每条 valid ray 在每层都有 target，深层 collapse 到前层是合法表达"ray terminated"。诊断显示原本 valid 的深层 pixel 和继承的 pixel 误差在同一 regime——filling 没引入 systematic bias。

### 7.2 XYZ pointmap vs depth + intrinsics

作者测了 depth+intrinsics 路线（更紧凑、更可解释），但发现：small focal length error 会 coherently warp 整个 shape。XYZ prediction 把 calibration 吸收进 pointmap 本身，global shape 更 plausible，尤其 real-world 图（crop、mask、metadata 都不确定）。这和 DUSt3R / MoGe 的 motivation 一致：直接输出 2D-to-3D map，需要时再从 L0 几何反求 intrinsics。

### 7.3 层数 L=6

Table 5 显示 valid pixel 在 L2 之后迅速下降。L=6 是 sweet spot：覆盖绝大多数 valid ray，同时保持 tensor 紧凑。更多层只对高度 perforated geometry（foliage、cage、grate）有帮助，但代价是 memory + sparse tail 监督。作者在 F.1 提到未来方向是 variable-depth / adaptive-ray representation。

### 7.4 LayerScale init

小规模 ablation：加 LayerScale init $10^{-4}$，10k iter 时 total loss 从 0.009258 降到 0.007018，XYZ loss 从 0.007242 降到 0.003537。对 early optimization 帮助巨大。RoPE 在小规模 neutral，但在 full model 跨分辨率 / camera crop extrapolation 时有用。

### 7.5 Generalization 两个意外发现

1. **Object model 没 scene training 也能在 scene image 上产出合理几何**——因为 pixel-aligned multilayer prediction 对 object/scene transfer 更自然（不像 canonical generator 在 cluttered scene 上崩）。
2. **Object model 没显式 multi-object augmentation 也 robust to multi-object image**——因为 WT 每个 ray 一个 stack，多个 disconnected object 自然共存于同一 tensor。

作者明确指出 next step：unify object / scene / dynamic into one model，而不是维护 specialist。

---

## 8. 局限和我的思考

作者承认的局限：
- 固定 6 层，对高度 perforated geometry 不够
- synthetic-to-real gap 在 textureless / reflective region 还在
- 20 步采样不实时，需要 distillation
- WT-D 只处理 short clip，long-range memory 和 persistent identity 留给未来
- WT-S 单帧，没融合 multi-frame observation

我想补充几点：

1. **Ray-wise attention 是真正的 architectural innovation**。VGGT 处理 multi-view 用 frame/global 交替，WT 处理 multilayer 显式引入 ray axis，让网络沿 ray "看见"自己。这可能是 generalizable 的 pattern——任何"同一位置多个 sample"的问题（multi-view、multilayer、time、BRDF samples）都可以加一个 axis-specific attention。

2. **Depth filling 这个 trick 思想上很接近 DINOv2 的 image-level supervision**——避免 head-based sparse supervision，用 dense target + mask 在 loss 端解决。它和 DiffusionDet 的 dense prediction 思路、和 NeRF 的 occupancy collapse-to-surface 都是同源思想。

3. **Layer-aware noise schedule 揭示了一个 deep insight**：multilayer geometry 上 reconstruction 和 generation 是 continuum，不是 binary。Layer 0 几乎是 deterministic mapping（深度估计），Layer 5 几乎是 conditional generation（背面 imagination）。一个 timestep 分布处理不了两端，需要 per-layer-aware schedule。这个思路对其他"混合 task"的 diffusion model 应该有启发——比如 image inpainting 中 mask 内 vs mask 外、video generation 中 conditioned frames vs free frames。

4. **Pixel-aligned 作为 3D interface 的价值被低估**。canonical-frame generator 输出虽然漂亮但和 input image 失联，下游 pipeline 用不上。WT 这种"camera-space + pixel correspondence + recoverable intrinsics"的输出格式，是个更好的 2D-to-3D contract。这是个 interface-level innovation，意义可能比模型本身更大。

5. **Mix-training 的简单性很美**。一个 mask gate 公式 (4) 就把 single-layer RGBD 数据塞进 multilayer 训练，不需要新 head、新 embedding、新 regime token。这是 representation choice 的红利——选对了 output format，数据扩展几乎是 free 的。对比 LaRI 这种 mask-head 路线，吃 RGBD single-layer 数据非常困难。

6. **没用 VAE latent 而在 pixel-space flow matching**——这是顺应当前 "Back to basics" (Li & He 2025) 的潮流。XYZ pointmap 维度可控（$6 \times 504 \times 504 \times 3 \approx 4.6M$），不需要 VAE 压缩，避免了 latent encoding 的信息损失。这是个值得关注的趋势。

---

## 9. 总结

WT 这篇 paper 的核心 contribution 不是某一个 trick，而是一个 **representation choice**：把单目深度估计和 image-to-3D generation 这两个长期分离的 task，统一成一个 image-grid 上的 multilayer pointmap tensor。围绕这个 representation，作者把所有该解决的工程问题都解了：
- dense supervision → depth filling
- multilayer attention → three-way factorization with explicit ray axis
- layer identity → LayerFiLM
- 不同 uncertainty profile → layer-aware noise schedule
- 数据 scale → mix-training with single-layer RGBD
- generalization → frozen MoGe encoder

最后这个 representation 在 object / scene / dynamic 三个 regime 都赢，visible surface 比 dedicated depth predictor 还准，downstream 三个 pipeline 都 training-free 接上。这是我读到的最近 image-to-3D 领域最有"统一感"的工作之一。

参考链接：
- WT 项目（推断）: https://world-tracing.github.io/
- MoGe: https://wangrc.github.io/MoGePage/
- MoGe-2: https://arxiv.org/abs/2507.02546
- LaRI: https://arxiv.org/abs/2504.18424
- TRELLIS: https://trellis3d.github.io/
- VGGT: https://vgg-t.github.io/
- Depth Anything v2: https://depth-anything-v2.github.io/
- π³ (Pi3X): https://arxiv.org/abs/2507.13347
- Flow matching: https://arxiv.org/abs/2210.02747
- SD3 / DiT: https://arxiv.org/abs/2403.03206
- DINOv2: https://arxiv.org/abs/2304.07193
- LayerScale (CaiT): https://arxiv.org/abs/2103.17239
- RoPE: https://arxiv.org/abs/2104.09864
- Depth peeling (Everitt): https://en.wikipedia.org/wiki/Order-independent_transparency#Depth_peeling
- Layered Depth Images (Shade 1998): https://dl.acm.org/doi/10.1145/280814.280882
- Objaverse: https://objaverse.allenai.org/
- Objaverse-XL: https://objaverse-xl.allenai.org/
- 3D-FRONT: https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future
- GSO: https://app.gigagenie.ai/p/42002496 (Google Scanned Objects, https://arxiv.org/abs/2204.11918)
- ScanNet: http://www.scan-net.org/
- NYU Depth V2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
- ETH3D: https://www.eth3d.net/
- Hypersim: https://github.com/apple/ml-hypersim
- Taskonomy: http://taskonomy.stanford.edu/
- DAVIS: https://davischallenge.org/
- World Labs: https://www.worldlabs.ai/
- FLUX.2 representation comparison: https://bfl.ai/techblog/representation-comparison/
- VACE: https://arxiv.org/abs/2503.17001
- TrajectoryCrafter: https://arxiv.org/abs/2503.05638
- ReconViaGen: https://arxiv.org/abs/2510.23306
- LaS-Comp: https://arxiv.org/abs/2602.18735
