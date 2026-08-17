---
source_pdf: FreeArt3D Training-Free Articulated Object Generation using 3D.pdf
paper_sha256: 54d4c1f1ec52f12909f87f0fe8d40225a63264ae7280bf7990dacc87ecdbdb4d
processed_at: '2026-08-04T10:26:40-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FreeArt3D

## 一句话版本

**Articulated object dataset 太少没法 train diffusion model, 干脆直接拿 static object 的 3D diffusion model (Trellis) 当 prior, 把 articulated object 的每个 joint state 都当成一个 static shape 去 optimize, 10 分钟搞定一个 articulated 3D asset.**

---

## 为什么这事难

Articulated object 就是那些 "有零件会动" 的东西: 微波炉门能开, 笔记本能合, 抽屉能拉, 订书机能按.

你想要一个 pipeline: 给几张不同 state 的照片 (微波炉关门、半开、全开), 生成一个带 joint 的 3D textured mesh, 能在 simulator 里打开关闭.

两条老路都走不通:

**第一条路: Feed-forward model 直接预测.** 训一个 network, input 几张图, output articulated mesh. 问题是 articulated 3D data 太少. PartNet-Mobility 是最大的 articulated dataset, 才几千个 shape. 相比之下 static object 有 Objaverse-XL 的 10M+ shape. 几千个 shape 训 diffusion model, 基本不可能 generalize. 所以现有 feed-forward 方法 (URDFormer, Singapo, Articulate-Anything) 只能做粗糙的事: 从 database 里 retrieve 零件, 拼起来. 结果 geometry 粗, texture 基本没有, category 还受限.

**第二条路: Per-instance NeRF optimization.** 像 PARIS 那样, 多 view NeRF joint optimize body 和 part. 问题是需要 dense view (几十张 posed photo), 跑得慢, 而且 joint parameter 估计很差 (paper 里 PARIS 的 axis error 1.139 rad ≈ 65°, 基本上 axis 是瞎猜的).

FreeArt3D 的 insight: **第三条路.** Static object 的 3D diffusion model 已经很强了 (Trellis 在 10M+ shape 上 train 过, generalize 极好). Articulated object 的每个 state 本质上就是一个 static shape. 我只要把 Trellis 当 prior, per-instance optimize 一个 articulated representation, 让它在每个 state 下都 "看起来像 Trellis 会生成的 valid shape", 就行了.

这本质上是 **DreamFusion 的 SDS 思想从 2D-to-3D 升级到 3D-to-4D**. DreamFusion 用 2D diffusion 监督 3D 生成, FreeArt3D 用 3D diffusion 监督 4D (3D + articulation) 生成.

参考:
- DreamFusion: https://dreamfusion3d.github.io/
- Trellis: https://trellis3d.github.io/
- PartNet-Mobility: https://sapien.ucsd.edu/browse/

---

## 核心机制: 怎么用 Trellis 监督 Articulated Object

### Representation: 两个 hash grid + joint params

一个 articulated object (single joint) 拆成两部分:
- **Body**: 不动的部分, 用 hash grid $\mathcal{H}_{body}$ 表示 occupancy
- **Part**: 会动的部分, 用 hash grid $\mathcal{H}_{part}$ 表示 occupancy
- **Joint**: rotation axis $\mathbf{a}$ + pivot point $\mathbf{p}$ (revolute) 或 translation axis $\mathbf{a}$ (prismatic)
- **State**: 每个 input image 对应一个 $\theta_k$ (angle 或 translation)

Hash grid 就是 TinyCudaNN 那种 multi-resolution hash encoding, input 3D coordinate, output [0,1] occupancy. 连续可微, 比 discrete voxel grid optimization 更稳定.

### 每次 iteration 做什么

随机采样一张 input image $I_k$ 和对应 state $\theta_k$:

1. 把 body hash grid 直接 query 到 64³ voxel grid (body 不动)
2. 把 part hash grid 通过 **inverse joint transform** query 到 posed frame
3. Max merge 两个 occupancy 成一个完整 64³ grid $x$
4. 把 $x$ encode 成 latent $z$, 喂给 Trellis 的 frozen rectified flow model
5. Trellis 告诉你 "这个 shape 应该长什么样", 算 SDS gradient + voxel reconstruction loss
6. Gradient backprop 到 hash grid weights 和 joint parameters

关键公式 (Equation 4, occupancy construction):

$$x(\mathbf{c}) = \max\left(\text{Occ}_{body}(\mathbf{c}), \text{Occ}_{part}(\mathcal{T}_{\theta_k}^{-1}(\mathbf{c}))\right)$$

变量解释:
- $\mathbf{c}$: posed frame 中的 voxel coordinate
- $\text{Occ}_{body}(\mathbf{c})$: body 在 posed frame 的 occupancy (直接 query)
- $\mathcal{T}_{\theta_k}^{-1}(\mathbf{c})$: inverse articulation transform, 把 posed frame coordinate 映射回 canonical frame
- $\text{Occ}_{part}(\mathcal{T}_{\theta_k}^{-1}(\mathbf{c}))$: part 在 canonical frame 的 occupancy, 通过 inverse warp 查询
- $\max$: merge 两个 part

Inverse transform 的具体形式:
- Revolute: $\mathcal{T}_{\theta_k}^{-1}(\mathbf{c}) = \mathbf{R}_{\theta_k}^{-1}(\mathbf{c} - \mathbf{p}) + \mathbf{p}$
  - $\mathbf{R}_{\theta_k}$: 绕 axis $\mathbf{a}$ 旋转 $\theta_k$ 的 rotation matrix
  - $\mathbf{p}$: pivot point
- Prismatic: $\mathcal{T}_{\theta_k}^{-1}(\mathbf{c}) = \mathbf{c} - \theta_k \cdot \mathbf{a}$
  - $\mathbf{a}$: translation axis

**Intuition**: 我们维护 part 在 "rest pose" 的 occupancy field. 要构建某个 articulated state 下的完整 shape, 对 posed frame 的每个 voxel, 先 inverse warp 回 rest pose, 再 query part 的 hash grid. 这是 inverse warping, 比 forward warping clean (没 holes), 而且对 joint parameters differentiable, gradient 能流回 joint params.

### SDS Loss (Equation 5)

$$\nabla_\psi \mathcal{L}_{SDS} \triangleq \mathbb{E}_{t,\epsilon}\left[w(t) \cdot \left(\hat{\epsilon}_{\mathcal{RF}_{occ}}(\mathbf{z}_t; \mathbf{I}_k, t) - \epsilon\right) \cdot \frac{\partial \mathbf{z}_t}{\partial \psi}\right]$$

变量:
- $\psi$: 所有可优化参数 (hash grid weights + joint params $\mathcal{T}$ + joint states $\theta_k$)
- $t$: diffusion timestep, sample 自 [0.5, 0.8]
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 加的 Gaussian noise
- $\mathbf{z}_t$: 加噪后的 latent
- $\hat{\epsilon}_{\mathcal{RF}_{occ}}(\mathbf{z}_t; \mathbf{I}_k, t)$: Trellis 预测的 noise (conditioned on image $\mathbf{I}_k$)
- $w(t)$: timestep 权重
- $\frac{\partial \mathbf{z}_t}{\partial \psi}$: Jacobian, chain rule 传 gradient

**人话**: 给当前 latent 加点 noise, 让 Trellis 猜 "noise 是啥". 如果 shape 完美 match image, Trellis 猜的 noise = 实际加的 noise. 差异就是 "shape 需要怎么改" 的信号. 这个信号通过 differentiable pipeline 传回 geometry 和 joint.

### Voxel Reconstruction Loss (Equation 6)

$$\mathcal{L}_{vox} = \|D_{occ}(\hat{\mathbf{z}}_0) - x\|_2^2$$

- $\hat{\mathbf{z}}_0$: Trellis denoise 出的 "clean latent" (它认为对应 image 的 valid shape)
- $D_{occ}(\hat{\mathbf{z}}_0)$: decode 回 occupancy grid
- $x$: 我们构建的 occupancy
- L2 loss 让我们的 occupancy match Trellis 的 prediction

**人话**: SDS 给的是 gradient direction, voxel loss 给的是 direct target. 两个一起用, SDS 是主引擎, voxel loss 是 stabilizer.

Ablation 验证 (Table 3):
- 去掉 SDS (row a): F-Score 0.892 → 0.610, 暴跌. **SDS 是核心**
- 去掉 voxel loss (row b): F-Score 0.892 → 0.873, 微降. **voxel loss 是辅助**

---

## Disk Normalization: 看着 hack 但至关重要的 trick

### 问题

Trellis 训练时把所有 shape normalize 到 unit bounding box. 对 static object 没问题. 对 articulated object 灾难:

想象一个 cabinet:
- 门关闭: bounding box 紧凑, body 占 unit cube 的 80%
- 门开 90°: 门伸出来, bounding box 变大, body 只占 unit cube 的 40%

同一个 body part 在不同 state 下被 Trellis normalize 到不同 scale. Optimization 时 body hash grid 的 scale 就分裂了, 收敛不了, 甚至 collapse.

### 解决

在 object 下面放一个 fixed size 的 disk (像地毯). 所有 input view 都带这个 disk. Trellis 把 disk + object 一起 normalize. Disk 大小固定, 所以 object 相对 disk 的 scale 就锁定了, body part 在不同 state 下 scale 一致.

Disk 在 optimization 收敛后直接 remove, 不影响最终结果.

### Ablation (Table 3, row c)

去掉 disk:
- F-Score: 0.892 → 0.749 (降 16%)
- Axis-Err: 0.155 → 0.730 (**5 倍恶化**)
- Pivot-Err: 0.121 → 0.244 (翻倍)

这是最 dramatic 的 ablation 之一. 没 disk, joint axis 基本估不准, 因为 geometry scale 在不同 state 下不一致, joint transform 的 fit 就全乱了.

### 为什么 disk 不损害 Trellis

Paper 验证: 144 个 object 上 test Trellis, 加 disk 的 CD = 0.018, 不加 disk 的 CD = 0.024. 加 disk 反而 slightly 更好. 说明 disk 不会 bias Trellis 的生成能力.

**Intuition**: Disk 相当于一把 "尺子". Trellis 训练时见过的 shape 都 normalize 到 unit cube, 它 implicitly 假设 input 是 "占满 unit cube 的东西". 加 disk 就是告诉 Trellis "unit cube 的范围由 disk 定义, object 相对 disk 有固定比例". 这其实更符合 Trellis 的 normalization 假设.

参考 Trellis: https://arxiv.org/abs/2412.01506

---

## Joint Initialization: 别从 random 开始

### 问题

直接 random init joint parameters 然后靠 SDS + voxel loss 优化, 容易陷 local minima. 因为如果 joint axis 一开始就错了, body 和 part 的 geometry 会被 "扭曲" 来弥补 wrong joint, 陷入 wrong basin.

Ablation (Table 3, row d): random init, Axis-Err 0.155 → 1.024 (≈ 58°), F-Score 0.892 → 0.753. 严重退化.

### 解决

1. 对每个 input image 独立 run Trellis full inference, 生成该 state 的 textured mesh
2. Render 这些 mesh 成 2D images (每 state 3 view, 共 18 张)
3. 选 36 对 image pairs
4. 用 **LoFTR** 检测 2D pixel correspondences
5. Lift 到 3D point pairs (利用 mesh geometry)
6. Filter 掉 static correspondences (两 state 间不变的点), 只留 movable part 的点对
7. 用这些 3D 点对 fit joint parameters $\mathcal{T}$ 和 states $\theta_i$, minimize 3D distance

### Intuition

每个 state 独立用 Trellis 生成 mesh, 虽然不同 state 的 mesh 之间不完全 consistent (各自独立生成), 但同一个 physical point 在不同 state 的 mesh 上大致对应. LoFTR 找 2D correspondence, lift 到 3D, 就能粗略估计 "哪些点在动, 怎么动", 从而 fit joint axis 和 pivot.

这个 init 不需要完美, 只需要给 optimization 一个大致正确的起点. Ablation row e: 固定 init joint 不再 refine, F-Score 0.864 (vs full 0.892), 说明 init 已经 capture 大部分, 后续 joint refinement 再提升一点.

参考 LoFTR: https://zju3dv.github.io/loftr/

---

## 后处理: 从 Coarse Occupancy 到 Textured Mesh

Coarse optimization 只产出 64³ occupancy. 要变成 high-fidelity textured mesh, 用 Trellis 的 Stage 2:

1. **构建最大 state 的 merged occupancy**: 用最大 $\theta_{max}$ (part 完全展开), 涵盖所有 part
2. **Denoise**: 加小 noise ($\sigma=0.5$), 用 $\mathcal{RF}_{occ}$ 单次 forward denoise, 得到 cleaner occupancy
3. **Remove disk**: 检测最低 z 值附近 voxel, discard
4. **Filter outliers**: 基于空间 isolation 和 size 阈值
5. **Sparse latent denoise**: 从 clean occupancy extract occupied voxels, 构建初始 sparse latent (Gaussian noise), 用 $\mathcal{RF}_{spa}$ denoise
6. **Decode**: 用 sparse decoder $D_{spa}$ decode 成 sparse feature volume, 包含 FlexiCubes coefficients (mesh) 和 3DGS parameters (texture)
7. **Partition**: 每个 voxel assign 给 body 或 part (看哪个的 transformed occupancy 更高)
8. **Extract mesh**: 每 part 从 FlexiCubes 提取 detailed mesh
9. **Bake texture**: 用 complete shape 的 3DGS parameters 把 texture bake 到每个 part 的 mesh
10. **Combine**: 两 textured mesh + 优化后的 joint params = 完整 articulated object

Ablation row g: 去掉 voxel refinement, CLIP-Sim 0.881 → 0.858, 主要影响 texture quality.

参考 FlexiCubes: https://research.nvidia.com/labs/toronto-ai/flexicubes/

---

## 实验结果: 数据说话

### Quantitative (Table 1, Average-12)

| Method | F-Score ↑ | CD ↓ | CLIP-Sim ↑ | Axis-Err ↓ | Pivot-Err ↓ |
|--------|----------|------|-----------|------------|-------------|
| **FreeArt3D** | **0.891** | **0.025** | **0.881** | **0.159** | **0.121** |
| ArtAnything | 0.769 | 0.043 | 0.851 | 0.499 | 0.191 |
| PARIS | 0.804 | 0.035 | 0.777 | 1.139 | 0.242 |

**人话解读**:
- **F-Score 0.891**: 89% 的 surface point 在 threshold 0.05 内 match ground truth. Geometry 很准
- **CD 0.025**: Chamfer Distance 0.025 (unit cube 下), 非常小
- **CLIP-Sim 0.881**: Texture 和 ground truth 很像. Retrieval-based 方法 (ArtAnything 0.851) 因为 retrieve 的 part texture 不匹配 input, 更低
- **Axis-Err 0.159 rad ≈ 9°**: Joint axis 方向误差 9°, 很准. PARIS 的 1.139 rad ≈ 65°, 基本瞎猜
- **Pivot-Err 0.121**: Pivot 点距离误差 0.121 (unit cube 下), 约 12% 的 cube 边长

### Per-category 亮点

- **Stapler**: Ours Axis-Err 0.039, ArtAnything 1.119, PARIS 0.421. Stapler 的 revolute joint 小 angle 特殊 axis, retrieval 完全 fail, FreeArt3D 几乎完美
- **Washing Machine**: Ours Axis-Err 0.654 (最差). Door rotation axis 复杂, 仍是难点

### Runtime (Table 2)

| Method | Time |
|--------|------|
| FreeArt3D | 606s (≈10 min) |
| PARIS | 494s (≈8 min) |
| ArtAnything | 200s |
| URDFormer | 27s |
| Singapo | 19s |

Feed-forward 方法秒级, 但 quality 差一大截. FreeArt3D 10 分钟, quality 显著更高, 这个 trade-off 很划算.

### Input view 数量影响 (Table 3)

- 6 views (default): F-Score 0.892, Axis-Err 0.155
- 21 views: F-Score 0.903, Axis-Err 0.126 (略好)
- 2 views: F-Score 0.841, Axis-Err 0.525 (明显差但还能用)

6 views 是 sweet spot. 2 views 主要问题在 joint init (correspondence 少, init 不准).

### Failure Case (Table 5)

6-state setup:
- Axis Direction error (>20°): 9.03%
- Pivot Point error (>0.1): 11.11%
- Segmentation failure: 9.03%
- **Overall success rate: 77.08%**

主要 failure 来自 segmentation (只生成 body 或 part 之一) 和 complex joint axis.

---

## 为什么这 approach work: 我的理解

### 1. Strong Prior 的威力

Trellis 在 10M+ static shape 上 train 过, 它 "见过" 各种 object, 知道 valid shape 长什么样. 把它当 prior, 相当于给 optimization 一个极强的 inductive bias. 比 NeRF (无 prior, 纯粹 photometric loss) 强太多, 比 retrieval-based (prior 受限于 database size) 也更 general.

### 2. 3D Prior > 2D Prior

DreamFusion 用 2D diffusion 做 3D generation, 容易 Janus problem (多面问题), 因为 2D supervision 本质 ambiguous (一张 2D 图对应无穷多个 3D shape). FreeArt3D 用 3D diffusion, 每个 state 直接给 full 3D supervision, ambiguity 大幅减少, consistency 自动保证, convergence 快.

### 3. Inverse Warping 的 elegance

维护 part 在 canonical frame 的 occupancy, 通过 inverse warp 构建 posed occupancy. 这个 formulation:
- 数学 clean (无 holes)
- 对 joint parameters differentiable (gradient 流回 joint)
- 自然 disentangle body 和 part (各自独立 hash grid)

### 4. Per-instance Optimization 的优势

Feed-forward model 受限于 training data distribution, 遇到未见过的 object 就 fail. Per-instance optimization 没有 distribution 限制, 只要 prior 足够 strong, 就能 handle 任意 object. 代价是慢, 但 10 分钟可接受.

### 5. Disk Normalization 的深刻 insight

这个 trick 背后是 **pre-trained model 的 implicit assumption** 问题. Trellis 假设 input normalize 到 unit cube. 这个假设对 static object 成立, 对 articulated object 的不同 state 不成立. Disk normalization 相当于修补这个 assumption, 给一个 invariant reference.

这个 insight 可以推广: 用 pre-trained model 到新 scenario 时, 要小心它的 normalization / preprocessing 假设, 必要时用 workaround 满足它.

---

## 更广的启示

### "Prior + Optimization" Paradigm

FreeArt3D 属于 "strong prior + reasonable optimization" 的 paradigm:
- **Feed-forward**: all prior, no optimization. 快但受限于 training data
- **NeRF**: all optimization, no prior. 慢且需要 dense view
- **FreeArt3D**: good prior + reasonable optimization. 慢一点但 quality 高且 general

这个 paradigm 在 data-scarce domain 特别有用. 不必 train new model, 复用 strong existing prior + careful optimization design 就能 achieve SOTA.

### 可推广方向

- **Articulated human body**: 用 static human 3D prior (如 SMPL) + optimization
- **Scene generation**: 用 static object 3D prior + spatial relation optimization
- **Physics-based generation**: 用 static shape prior + physics simulation constraint
- **Cloth simulation**: 用 static garment prior + deformation optimization

### Limitations

- 10 分钟/shape 还是慢. 未来可以 distill 成 feed-forward
- Disk normalization 是 workaround, 更 elegant 方案可能需要 fine-tune Trellis
- Single DOF kinematic chain 假设. 复杂 parallel joints (fingers) 需要 rethinking
- Rigid part only. Non-rigid deformation 不适用

---

## 最后的 Intuition

想象你要画一个会动的变形金刚. 老路: 找一本变形金刚图鉴 (articulated dataset), 照着画. 问题是图鉴太小, 没几个款式. 新路: 找一个会画所有 static 物体的大师 (Trellis), 让他看你给的几张变形金刚不同形态的照片, 每次他都能画出该形态下的 valid 3D shape. 你让他画很多次, 慢慢调整你的变形金刚 model, 直到每个形态都 match 大师的画.

大师就是 Trellis, 你的 model 就是两个 hash grid + joint params, 调整过程就是 SDS optimization. Disk normalization 相当于给大师一个固定大小的画布, 这样他画不同形态时比例一致. Joint init 相当于先让大师独立画几张, 你对比找出 "哪些部位在动, 大致怎么动", 给个起始猜测.

整个 framework 的 beauty 在于: **不需要新 data, 不需要新 training, 只需要用好已有的 strong prior + clever optimization design**. 这在 data-scarce 的 3D/4D generation 领域是关键的 design philosophy.

Paper: https://czzzzh.github.io/FreeArt3D
Trellis: https://trellis3d.github.io/
DreamFusion: https://dreamfusion3d.github.io/
LoFTR: https://zju3dv.github.io/loftr/
FlexiCubes: https://research.nvidia.com/labs/toronto-ai/flexicubes/
TinyCudaNN: https://github.com/NVlabs/tiny-cuda-nn
PartNet-Mobility: https://sapien.ucsd.edu/browse/
Objaverse-XL: https://objaverse-xl.com/
HiFA: https://arxiv.org/abs/2305.18766

---

# FreeArt3D: Training-Free Articulated Object Generation 深度解析

## 1. 核心Insight: 把 Static 3D Diffusion 当作 Articulated Object 的 Prior

这篇paper的核心insight非常elegant: articulated object dataset 极度稀缺(PartNet-Mobility 只有几千个 shape), 训练 native 3D diffusion model for articulated objects 几乎不可行. 但 static object 的 3D diffusion model (Trellis) 已经 trained on 大规模 data (Objaverse-XL, 10M+ objects) 并且 generalization 极强.

FreeArt3D 的策略: **不要训练新 model, 而是把 Trellis 当作 shape prior, 通过 per-shape optimization 重建 articulated object**. 这本质上是把 DreamFusion 的 SDS (Score Distillation Sampling) 思想从 **2D-to-3D** 扩展到 **3D-to-4D**, 把 articulation state 视为额外的 generative dimension.

每个 articulation state 对应一个 static 3D shape, Trellis 对每个 state 都能提供 full 3D guidance. 这比 2D SDS 强很多——2D SDS 容易出现 Janus problem (multi-view inconsistency), 因为 2D supervision 本身 ambiguous; 而 3D prior 直接提供 full shape guidance, consistency 自动保证.

参考: 
- DreamFusion SDS: https://dreamfusion3d.github.io/
- Trellis: https://trellis3d.github.io/

---

## 2. Trellis 架构回顾 (作为 Prior)

Trellis 是两 stage 的 3D diffusion framework, 输入 single image 或 text prompt, 输出 textured mesh.

### Stage 1: Coarse Geometry Generation (Occupancy 64³)

VAE encode/decode occupancy grid:
$$z = E_{occ}(x) \in \mathbb{R}^{16 \times 16 \times 16 \times c}, \quad \hat{x} = D_{occ}(z)$$

变量解释:
- $x \in \mathbb{R}^{64 \times 64 \times 64}$: input occupancy grid, 每个 voxel 取值 [0,1] 表示占据概率
- $E_{occ}, D_{occ}$: VAE encoder/decoder
- $z$: latent representation, spatial resolution 从 64³ 降到 16³, feature dimension $c$
- $\hat{x}$: 重建的 occupancy grid

然后在 latent space 训练 rectified flow model $\mathcal{RF}_{occ}$ 做 conditional generation (conditioned on image/text).

### Stage 2: Detailed Geometry and Texture (Sparse Volume 256³)

Trellis 构建一个 sparse feature volume:
$$\mathcal{F} = \{(\mathbf{x}_i, \mathbf{f}_i)\}_{i=1}^N, \quad \mathbf{x}_i \in \{0,...,63\}^3, \quad \mathbf{f}_i \in \mathbb{R}^d$$

变量解释:
- $\mathcal{F}$: sparse volume, 只在 occupied voxel 上存储 feature
- $\mathbf{x}_i$: 第 $i$ 个 occupied voxel 的 3D coordinate (index space)
- $\mathbf{f}_i$: 该 voxel 的 $d$-维 feature vector, 从 multi-view DINO embeddings 提取
- $N$: occupied voxel 数量

sparse encoder/decoder:
$$\mathcal{Z}' = E_{spa}(\mathcal{F}) = \{(\mathbf{x}_i, \mathbf{z}'_i)\}_{i=1}^N, \quad \mathbf{z}'_i \in \mathbb{R}^{d'}, \quad \hat{\mathcal{F}} = D_{spa}(\mathcal{Z}')$$

- $E_{spa}, D_{spa}$: sparse encoder/decoder
- $\mathcal{Z}'$: sparse latent, 每个 occupied voxel 对应一个 $d'$-维 embedding
- $\hat{\mathcal{F}}$: 重建的 sparse feature volume

第二个 rectified flow $\mathcal{RF}_{spa}$ 在 sparse latent space 训练. Decoder 上采样到 256³, 支持 FlexiCubes (mesh generation) 或 3D Gaussian Splatting (texture rendering) 两种 output format.

参考 Trellis paper: https://arxiv.org/abs/2412.01506

---

## 3. FreeArt3D 的 Optimization Pipeline

### 3.1 表示 (Representation)

对于 single joint 物体, decompose 成两个 component:
- **Static body**: textured mesh $M_{body}$
- **Movable part**: textured mesh $M_{part}$
- **Joint parameters** $\mathcal{T}$:
  - Revolute joint: unit rotation axis $\mathbf{a} \in \mathbb{R}^3$ + pivot point $\mathbf{p} \in \mathbb{R}^3$ (axis 上的点)
  - Prismatic joint: translation axis $\mathbf{a} \in \mathbb{R}^3$
- **Joint state** $\theta_k \in \mathbb{R}$: 每个 input image $I_k$ 对应的 articulation angle (rad) 或 translation magnitude

Geometry 用两个 continuous multi-level hash grid $\mathcal{H}_{body}$ 和 $\mathcal{H}_{part}$ 表示:
$$\text{Occ}_{body}(\mathbf{c}) = \mathcal{H}_{body}(\mathbf{c}), \quad \text{Occ}_{part}(\mathbf{c}) = \mathcal{H}_{part}(\mathbf{c})$$

- $\mathbf{c} \in \mathbb{R}^3$: 3D coordinate
- 输出 [0, 1] 的 continuous occupancy value

**为什么用 hash grid 而非 discrete voxel grid**: paper 在 ablation (row f) 中验证, discrete 64³ voxel grid 在 optimization 时 instable, continuous multi-level hash grid 提供更 smooth 的 gradient, optimization 更 robust. 这是 TinyCudaNN 风格的 multi-resolution hash encoding.

### 3.2 Occupancy Grid Construction (Equation 4)

每个 iteration, 采样 input image $I_k$ 和对应 joint state $\theta_k$, 构建 64³ occupancy grid:

$$x(\mathbf{c}) = \max\left(\text{Occ}_{body}(\mathbf{c}), \text{Occ}_{part}(\mathcal{T}_{\theta_k}^{-1}(\mathbf{c}))\right)$$

变量解释:
- $\mathbf{c}$: voxel coordinate in **posed frame** (当前的 articulation state)
- $\text{Occ}_{body}(\mathbf{c})$: body 在 posed frame 的 occupancy (body 不动, 直接 query)
- $\mathcal{T}_{\theta_k}^{-1}(\mathbf{c})$: 把 posed frame 的 voxel coordinate 映射回 **canonical frame** (rest pose)
- $\text{Occ}_{part}(\mathcal{T}_{\theta_k}^{-1}(\mathbf{c}))$: part 在 canonical frame 的 occupancy, 通过 inverse transform 查询
- $\max$: 两个 part 合并, 处理 overlap

**Inverse transform** 的具体形式:
- Revolute joint: $\mathcal{T}_{\theta_k}^{-1}(\mathbf{c}) = \mathbf{R}_{\theta_k}^{-1}(\mathbf{c} - \mathbf{p}) + \mathbf{p}$
  - $\mathbf{R}_{\theta_k}$: 绕 axis $\mathbf{a}$ 旋转 $\theta_k$ 弧度的 rotation matrix
  - 先 translate 到 pivot, 反向旋转, 再 translate 回来
- Prismatic joint: $\mathcal{T}_{\theta_k}^{-1}(\mathbf{c}) = \mathbf{c} - \theta_k \cdot \mathbf{a}$
  - 简单的反向 translation

这个 inverse transform 的 intuition: 我们维护 part 在 canonical frame 的 occupancy field, 但要在 posed frame 构建 merged grid. 给定 posed frame 中的 voxel $\mathbf{c}$, 我们想知道它在 canonical frame 中对应哪个 point, 然后 query part 的 hash grid.

### 3.3 Loss Functions

#### SDS Loss (Equation 5)

$$\nabla_\psi \mathcal{L}_{SDS}(\mathcal{RF}_{occ}, \mathbf{z}) \triangleq \mathbb{E}_{t,\epsilon}\left[w(t) \cdot \left(\hat{\epsilon}_{\mathcal{RF}_{occ}}(\mathbf{z}_t; \mathbf{I}_k, t) - \epsilon\right) \cdot \frac{\partial \mathbf{z}_t}{\partial \psi}\right]$$

变量逐项解释:
- $\psi$: 所有 optimizable parameters, 包括 $\mathcal{H}_{body}, \mathcal{H}_{part}$ 的 weights, joint parameters $\mathcal{T}$, 以及 (如果 unknown) joint states $\theta_k$
- $\mathbf{z} = E_{occ}(x)$: 从 Equation 4 构建的 occupancy grid 经 VAE encode 得到的 latent
- $t$: diffusion timestep, 从 [0.5, 0.8] 区间 sample (paper 的设置)
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 采样的 Gaussian noise
- $\mathbf{z}_t$: 给 $\mathbf{z}$ 加 timestep $t$ 对应的 noise 得到的 noisy latent
- $\hat{\epsilon}_{\mathcal{RF}_{occ}}(\mathbf{z}_t; \mathbf{I}_k, t)$: rectified flow model 在 condition $(\mathbf{I}_k, t)$ 下预测的 noise
- $w(t)$: 权重函数, balance 不同 timestep 的贡献
- $\frac{\partial \mathbf{z}_t}{\partial \psi}$: noisy latent 对可优化参数的 Jacobian, 通过 chain rule 传递 gradient 回 hash grid 和 joint parameters

**Intuition**: rectified flow model "知道" 一个 valid 3D shape 给定 image 应该长什么样. 我们让它 evaluate 我们当前构建的 occupancy latent $\mathbf{z}_t$, 它会预测一个 noise $\hat{\epsilon}$. 如果我们的 shape 完美 match, predicted noise 应该等于实际加的 noise $\epsilon$. 差异 $\hat{\epsilon} - \epsilon$ 就指示了 shape 需要怎么调整. 这个差异通过 $\partial \mathbf{z}_t / \partial \psi$ backpropagate 到 geometry 和 joint parameters.

#### Voxel Reconstruction Loss (Equation 6)

$$\mathcal{L}_{vox} = \|D_{occ}(\hat{\mathbf{z}}_0) - x\|_2^2$$

- $\hat{\mathbf{z}}_0$: 从 predicted noise $\hat{\epsilon}_{\mathcal{RF}}(\mathbf{z}_t; \mathbf{I}_k, t)$ 推导的 denoised latent (即 $\mathbf{z}_t - t \cdot \hat{\epsilon}$ 在 rectified flow 中)
- $D_{occ}(\hat{\mathbf{z}}_0)$: VAE decoder 把 denoised latent decode 回 occupancy grid
- $x$: 我们构建的原始 occupancy grid (Equation 4)
- $\|\cdot\|_2^2$: L2 norm squared

这个 loss 的 intuition: rectified flow model 的 denoised prediction $D_{occ}(\hat{\mathbf{z}}_0)$ 代表了 "Trellis 认为对应 image $\mathbf{I}_k$ 的 valid shape". 我们让我们的 occupancy $x$ match 这个 prediction, 是一种直接的 supervision, 比 SDS 的 gradient 更 stable. Inspired by HiFA (Zhu & Zhuang 2023).

#### Total Loss (Equation 7)

$$\mathcal{L}_{total} = \lambda_{SDS} \cdot \mathcal{L}_{SDS} + \lambda_{vox} \cdot \mathcal{L}_{vox}$$

- $\lambda_{SDS} = 0.1$
- $\lambda_{vox} = 1.0$

Ablation (Table 3) 显示:
- 去掉 SDS loss (row a): F-Score 从 0.892 暴跌到 0.610, CD 从 0.025 到 0.067. **SDS loss 是核心**
- 去掉 voxel loss (row b): F-Score 微降到 0.873, CD 微升到 0.028. **voxel loss 是 stabilizer**

参考 HiFA: https://arxiv.org/abs/2305.18766

---

## 4. Disk Normalization: 关键的 Engineering Trick

### 4.1 问题

Trellis 训练时把所有 shape normalize 到 unit bounding box. 对于 articulated object, 同一个 object 在不同 joint state 下 bounding box size 差异巨大:

- Cabinet 门关闭: bounding box 紧凑
- Cabinet 门打开 90°: bounding box 在门的延伸方向变大很多
- Stapler 闭合 vs 打开: handle 的延伸让 bounding box 变化

这意味着: 同一个 body part 在不同 state 下, 被 Trellis normalize 到不同的 scale. 例如关闭 cabinet 时 body 占 unit cube 的 80%, 打开时只占 40%. 这破坏了 cross-state geometric correspondence, 让 optimization 难以 converge, 甚至 collapse 到 zero 或 full occupancy.

### 4.2 解决方案

在 object 下面添加一个 fixed reference disk (像 carpet/floor), 在所有 input view 中都存在. 这个 disk 作为 visual anchor, 让 Trellis 把 disk + object 一起 normalize. 由于 disk 在所有 state 下都一样大, object 相对 disk 的比例就固定了, body part 在不同 state 下的 scale 保持一致.

### 4.3 Ablation 验证 (Table 3, row c)

去掉 disk normalization:
- F-Score: 0.892 → 0.749 (下降 16%)
- CD: 0.025 → 0.048 (几乎翻倍)
- Axis-Err: 0.155 → 0.730 (5 倍恶化!)
- Pivot-Err: 0.121 → 0.244 (翻倍)

非常显著的 degradation, 证明 disk normalization 是 essential.

### 4.4 Disk 不损害 Trellis 能力

Paper 验证: 在 144 个 3D object 上 test Trellis, 加 disk vs 不加 disk, CD 分别为 0.018 和 0.024. 加 disk 反而 slightly 更好, 没有 capacity degradation 或 bias.

**Intuition**: 想象你在拍 photos 重建一个 articulated object. 如果只拍 object 本身, Trellis 会把 object 在每张 photo 中独立 normalize 到 unit cube, 导致不同 state 下 body part 的 scale 不一致. 在 object 下面放一个固定大小的 disk, 相当于给 Trellis 一个 "ruler", 所有 state 的 scale 都基于这个 ruler, body part 的大小就被锁定了. Disk 可以在 converge 后 remove, 不影响最终结果.

---

## 5. Joint Initialization: 不容忽视的细节

### 5.1 流程

直接用 SDS + voxel loss 从 random init 优化 joint parameters 很难 converge (Table 3, row d 显示 Axis-Err 从 0.155 飙到 1.024). Paper 用一个 heuristic 给 good init:

1. 对每个 input image $I_k$ 独立 run Trellis full inference, 生成该 state 下的 textured mesh
2. 用 Blender render 这些 mesh 成 2D images (每个 state 3 个 view, 6 states 共 18 张)
3. 选 36 对 image pairs
4. 用 **LoFTR** (detector-free local feature matching with transformers) 检测 2D pixel-level correspondences
5. 把 2D matches lift 到 3D point pairs (利用 mesh 的 3D geometry)
6. Filter 掉 static point pairs (在两个 state 间位置不变), 只保留 movable part 的 point pairs
7. Construct 变换 based on joint parameters $\mathcal{T}$ 和 state $\theta_i$, minimize 3D distance between corresponding points
8. 得到 initial $\mathcal{T}$ 和 $\theta_i$

### 5.2 Ablation (Table 3)

- Random init (row d): F-Score 0.753, Axis-Err 1.024, Pivot-Err 0.253. **严重退化**
- Fixed joint after init (row e): F-Score 0.864, Axis-Err 0.165, Pivot-Err 0.138. 比 full method 略差, 说明 joint refine 有用, 但 init 已经 capture 大部分信息

### 5.3 Intuition

SDS 和 voxel loss 的 landscape 非常复杂, 有很多 local minima. 如果 joint axis init 错了, body 和 part 的 geometry 会被 "扭曲" 来弥补 wrong joint, 陷入 local optimum. 用 3D correspondence 先估一个大致正确的 joint axis, 给 optimization 一个好起点, 让 geometry 和 joint 能 jointly refine 到正确解.

参考 LoFTR: https://zju3dv.github.io/loftr/

---

## 6. Fine-grained Geometry 和 Texture 生成

### 6.1 Occupancy Denoising 和 Cleaning

1. 构建 merged occupancy $x_{max}$ 对应 largest joint state $\theta_{max}$ (最大的 extension, 涵盖所有 part)
2. Inject 小 Gaussian noise ($\sigma = 0.5$ before scaling)
3. 用 $\mathcal{RF}_{occ}$ 单次 forward pass denoise, 得到 cleaner occupancy $\tilde{x}$
4. Remove "carpet plane": 检测最低 z 值附近的 voxel 并 discard
5. Filter outlier voxels based on spatial isolation 和 size thresholds
6. 得到 clean voxel grid $\bar{x}$

Ablation (row g): 去掉 voxel refinement, F-Score 0.892 → 0.875, CD 0.025 → 0.027, CLIP-Sim 0.881 → 0.858. 各项都有提升.

### 6.2 Sparse Latent 构建 和 Decoding

1. 从 $\bar{x}$ extract occupied voxel coordinates
2. 构建初始 sparse latent (sample Gaussian noise)
3. 用 second-stage flow model $\mathcal{RF}_{spa}$ denoise
4. 用 sparse decoder $D_{spa}$ decode 成 sparse feature volume $\hat{\mathcal{F}}$, 包含:
   - FlexiCubes coefficients for mesh extraction
   - 3D Gaussian Splatting parameters for texture synthesis

### 6.3 Mesh Extraction 和 Texture Baking

1. Partition $\hat{\mathcal{F}}$ 到 body 和 part: 检查每个 cell 在两个 part 的 transformed occupancy 中哪个更高, assign 给那个 part
2. 对每个 part extract detailed mesh from FlexiCubes
3. 用 decoded 3DGS parameters 把 complete shape 的 texture bake 到每个 part 的 mesh 上
4. Combine 两个 textured mesh + 优化后的 joint parameters $\mathcal{T}$ = 完整 articulated object

参考 FlexiCubes: https://research.nvidia.com/labs/toronto-ai/flexicubes/

---

## 7. 实验结果分析

### 7.1 Baselines

- **Articulate-Anything** (Le et al. 2024): VLM-based, 从 PartNet-Mobility retrieve parts, 适用于所有 12 个 category
- **Singapo** (Liu et al. 2024a): single image, retrieve parts, 支持 7/12 category
- **URDFormer** (Chen et al. 2024b): predict bounding boxes, scale template meshes, 支持 5/12 category
- **PARIS** (Liu et al. 2023d): multi-view NeRF optimization, 24 views, 支持 12 category

### 7.2 Quantitative Results (Table 1)

**Average-12** (12 categories, Ours 和 ArtAnything 和 PARIS 都能 handle):

| Method | F-Score ↑ | CD ↓ | CLIP-Sim ↑ | Axis-Err ↓ | Pivot-Err ↓ |
|--------|----------|------|-----------|------------|-------------|
| Ours | 0.891 | 0.025 | 0.881 | 0.159 | 0.121 |
| ArtAnything | 0.769 | 0.043 | 0.851 | 0.499 | 0.191 |
| PARIS | 0.804 | 0.035 | 0.777 | 1.139 | 0.242 |

**关键观察**:
- **F-Score**: Ours 比 ArtAnything 高 16%, 比 PARIS 高 11%
- **CD**: Ours 是 ArtAnything 的 58%, PARIS 的 71%
- **CLIP-Sim**: Ours 比 PARIS 高 13% (因为 retrieval-based 不能 reproduce input 的 texture)
- **Axis-Err**: Ours 是 ArtAnything 的 32%, PARIS 的 14% (PARIS 的 joint estimation 很差, 1.139 rad ≈ 65°)
- **Pivot-Err**: Ours 是 ArtAnything 的 63%, PARIS 的 50%

### 7.3 Per-Category 分析

Table 1 的 per-category 数据揭示有趣的 pattern:
- **Stapler**: Ours Axis-Err 0.039, ArtAnything 1.119, PARIS 0.421. Stapler 的 revolute joint 比较特殊 (小 angle, 特定 axis), retrieval-based 完全 fail
- **Table**: Ours Axis-Err 0.535, ArtAnything 0.287, PARIS 1.537. Table 的 prismatic joint (drawer) 在某些 case 反而 retrieval 占优
- **Washing Machine**: Ours Axis-Err 0.654, 这是最差的 category, 可能因为 door rotation axis 比较复杂

### 7.4 Runtime (Table 2)

| Method | Runtime (s) |
|--------|-------------|
| Ours | 606 |
| ArtAnything | 200 |
| Singapo | 19 |
| URDFormer | 27 |
| PARIS | 494 |

Ours 大约 10 分钟/shape, 比 PARIS (8 分钟) 略慢, 但远快于 NeRF-based dense view 方法. 考虑到 quality 显著更高, 这个 runtime 是非常 reasonable 的.

### 7.5 Ablation 深入分析 (Table 3)

| Index | Version | F-Score ↑ | CD ↓ | CLIP-Sim ↑ | Axis-Err ↓ | Pivot-Err ↓ |
|-------|---------|----------|------|-----------|------------|-------------|
| a | no SDS loss | 0.610 | 0.067 | 0.808 | 0.555 | 0.152 |
| b | no voxel loss | 0.873 | 0.028 | 0.880 | 0.172 | 0.138 |
| c | no disk | 0.749 | 0.048 | 0.854 | 0.730 | 0.244 |
| d | random joint init | 0.753 | 0.049 | 0.849 | 1.024 | 0.253 |
| e | fixed joint after init | 0.864 | 0.029 | 0.876 | 0.165 | 0.138 |
| f | no hashgrid (discrete voxel) | 0.848 | 0.032 | 0.872 | 0.157 | 0.180 |
| g | no voxel refinement | 0.875 | 0.027 | 0.858 | 0.181 | 0.111 |
| h | more input (6→21) | 0.903 | 0.023 | 0.883 | 0.126 | 0.135 |
| i | minimum input (6→2) | 0.841 | 0.035 | 0.870 | 0.525 | 0.227 |
| j | full | 0.892 | 0.025 | 0.881 | 0.155 | 0.121 |

**关键 takeaways**:
1. **SDS loss 是 backbone** (row a): 没有 SDS, F-Score 暴跌 32%
2. **Disk normalization 至关重要** (row c): Axis-Err 恶化 5 倍
3. **Joint init 决定成败** (row d): random init 导致 Axis-Err 1.024 (≈ 58°)
4. **Input view 数量有 sweet spot** (row h, i): 21 views 比 6 views 略好, 2 views 明显变差但仍可用
5. **Voxel refinement 提升 texture** (row g): CLIP-Sim 从 0.881 降到 0.858, 主要影响 appearance

### 7.6 Failure Case 分析 (Table 5)

| Failure Type | Ours (w/o disk) | Ours (2-state) | Ours (6-state) |
|--------------|-----------------|-----------------|------------------|
| Axis Direction | 45.14% | 32.64% | 9.03% |
| Pivot Point | 43.75% | 47.92% | 11.11% |
| Segmentation | 41.67% | 11.81% | 9.03% |

6-state setup 整体 success rate **77.08%**, 相当 robust. 关键发现:
- **Disk normalization 对 segmentation 也很关键**: 没 disk 时 41.67% segmentation fail (只生成 body 或 part 之一)
- **2-state 时 pivot 误差最大** (47.92%): 因为 pivot 估计需要足够的 state 变化来 disambiguate
- **6-state 时 axis 误差最小** (9.03%): 更多 state 提供更好的 motion signal

---

## 8. 我的 Intuition 构建

### 8.1 整体 framework 的 elegance

FreeArt3D 的核心 insight 可以总结为: **"Articulated object 的每个 state 都是一个 static shape, 所以 static 3D diffusion model 天然适合做 articulated object 的 prior"**.

这相当于把 4D problem (3D + articulation dimension) 降维成多个 3D problem, 用同一个 3D prior 解决. 这比训练 4D diffusion model (需要海量 articulated data) 或者用 2D prior (Janus problem) 都更 elegant.

### 8.2 SDS 在 3D-to-4D 的扩展

传统 SDS (DreamFusion):
- 2D diffusion model 提供 view-wise 2D guidance
- 3D representation (NeRF, mesh) 通过 differentiable rendering 被 supervise
- 容易 multi-view inconsistent (Janus problem)

FreeArt3D 的 SDS:
- 3D diffusion model 提供 state-wise 3D guidance (full shape, 不是 2D projection)
- Articulated representation (body hash grid + part hash grid + joint params) 通过 inverse transform 构建 merged occupancy, 被 3D diffusion supervise
- 天然 consistent (每个 state 都有 full 3D supervision)
- Converge 快 (少 ambiguity)

### 8.3 Inverse Transform 的数学 beauty

Equation 4 的 $\mathcal{T}_{\theta_k}^{-1}(\mathbf{c})$ 是关键. 我们维护 part 在 canonical frame 的 occupancy field, 但要在 posed frame 构建 merged grid. 数学上:

如果 part 在 canonical frame 的 point $\mathbf{c}_{canon}$ 经过 joint transform $\mathcal{T}_{\theta_k}$ 变成 posed frame 的 point $\mathbf{c}_{posed}$, 那么给定 posed frame 的 voxel $\mathbf{c}$, 它对应的 canonical frame point 是 $\mathcal{T}_{\theta_k}^{-1}(\mathbf{c})$.

这是 **inverse warping**, 比 forward warping 更 clean (避免 holes), 在 NeRF 和 3D generation 中很常见. 关键是 $\mathcal{T}_{\theta_k}^{-1}$ 关于 joint parameters $\mathcal{T}$ 是 differentiable 的, 所以 gradient 可以 backpropagate 到 joint parameters.

### 8.4 Disk normalization 的 profound insight

这个 trick 看起来简单, 但背后是一个 deep issue: **3D diffusion model 的 normalization 假设**. Trellis (和其他 3D diffusion) 训练时把 shape normalize 到 unit cube, 这是 standard practice. 但对于 articulated object, 这个 normalization在不同 state 下产生不同的 scale, 破坏 cross-state correspondence.

Disk normalization 相当于引入一个 **不变的 reference frame**, 让所有 state 的 scale 都相对这个 reference 定义. 这是一个 generalizable 的 insight, 可以应用到其他涉及多 state / 多 configuration 的 3D generation task.

更广义地, 这揭示了 **pre-trained model 的 implicit assumption** 问题: 当把一个 trained model 应用到新 scenario 时, 它的 normalization / preprocessing 假设可能不成立, 需要 workaround.

### 8.5 与其他 paradigm 的对比

**Feed-forward methods (URDFormer, Singapo, Articulate-Anything)**:
- 优点: 快 (seconds)
- 缺点: coarse geometry (bounding box, template, retrieval), 没 texture, limited categories
- 本质: 学习 category-specific prior, generalization 受限于 training data

**Optimization-based methods (PARIS)**:
- 优点: fine-grained geometry
- 缺点: 慢, 需要 dense view, joint estimation 差 (Axis-Err 1.139!)
- 本质: per-instance NeRF, 没 strong prior

**FreeArt3D (优化 + 强 prior)**:
- 优点: fine-grained geometry + texture + accurate joint + generalization
- 缺点: 10 分钟/shape
- 本质: 把 strong 3D diffusion prior 注入 per-instance optimization, 避免 dataset 限制, 同时获得 high fidelity

这相当于 **"prior + optimization" 的 sweet spot**: feed-forward 是 "all prior, no optimization", NeRF 是 "all optimization, no prior", FreeArt3D 是 "good prior + reasonable optimization".

### 8.6 Limitations 和未来方向

Paper 提到 limitations:
- Per-shape optimization 还是慢 (虽然 10 分钟可接受), 未来可以 distill 成 feed-forward
- Disk normalization 是 workaround, 更 elegant 的解决方案可能需要 fine-tune Trellis 或用 point correspondence

我额外想到的:
- **Multi-joint**: Paper 展示了 extension, 但 kinematic chain 假设 single DOF. 更复杂的 parallel joints (如 fingers) 需要 rethinking
- **Self-occlusion**: 当 part 遮挡 body 时, hash grid 的 max operation 可能不够, 可能需要 soft composition
- **Non-rigid articulation**: 当前只支持 rigid part (revolute + prismatic), 非 rigid deformation (如 clothing) 不适用
- **Differentiable rendering for texture**: 当前 texture 来自 3DGS bake, 可能局限. 可以结合可微渲染让 texture 直接 supervise from input images

### 8.7 与 Recent work 的 connection

- **4D generation** (4D-fy, DreamGaussian4D, Animate124): 这些用 2D/video diffusion 做 4D generation, 多是 character animation, focus on deformation field. FreeArt3D focus on rigid part-based articulation, 用 3D prior 避免 multi-view inconsistency.
- **Native 3D diffusion** (Trellis, CLAY, Hunyuan3D, Direct3D): 这些是 static object 的 3D-native generation. FreeArt3D 把 Trellis 作为 prior, 不训练新 model.
- **Part-level generation** (NAP, MeshArt, PhysPart, ArtFormer): 这些直接在 articulated data 上 train, 受限于 dataset size. FreeArt3D 完全 training-free.

参考:
- 4D-fy: https://sherwinbahmani.github.io/4dfy/
- DreamGaussian4D: https://github.com/jasongzy/DreamGaussian4D
- Hunyuan3D: https://github.com/Tencent/Hunyuan3D-2
- NAP: https://naparticulated.github.io/

---

## 9. 总结

FreeArt3D 是一个 **"巧" 字当头** 的工作:
1. **巧妙地复用** static 3D diffusion model 作为 articulated object 的 prior, 绕过 data scarcity
2. **巧妙地扩展** SDS 从 2D-to-3D 到 3D-to-4D, 用 articulation state 作为 generative dimension
3. **巧妙地引入** disk normalization 解决 cross-state scale inconsistency
4. **巧妙地设计** joint initialization pipeline (LoFTR + 3D correspondence) 给 optimization good start

技术细节上, inverse warping 的 hash grid query, SDS + voxel loss 的组合, occupancy denoising 和 sparse latent decoding 的 post-processing, 都体现了对 Trellis 架构的深入理解和 careful engineering.

实验上, F-Score 0.891, Axis-Err 0.159 rad (约 9°), 在 6 input view 下 10 分钟完成, 显著 outperform retrieval-based (ArtAnything) 和 NeRF-based (PARIS) baselines.

对未来的启示: **"prior + optimization" paradigm** 在 data-scarce 的 3D/4D generation 领域有巨大潜力. 不一定要 train new model, 复用 strong existing prior + careful optimization design 同样能 achieve SOTA. 这种思路可以推广到:
- Articulated human body (用 static human 3D prior)
- Scene generation (用 static object 3D prior + spatial relation)
- Physics-based generation (用 static shape prior + physics simulation)

Paper link: https://czzzzh.github.io/FreeArt3D
Trellis: https://trellis3d.github.io/
arXiv (推算): https://arxiv.org/abs/2506.09893 (基于 paper 内容推测, 实际需 verify)

---

**最后一点 meta-level 的 intuition**: 这个 paper 让我想到 DeepMind 的 AlphaFold2 也是 "prior + optimization" 思路——用 strong evolutionary prior (MSA) + careful architecture refinement. FreeArt3D 在 3D generation 领域展现了类似 philosophy: 不必追求 train from scratch, 把 strong existing prior 用对用好, 往往比 brute-force training 更 effective. 这在 data-scarce domain 是关键的 design principle.
