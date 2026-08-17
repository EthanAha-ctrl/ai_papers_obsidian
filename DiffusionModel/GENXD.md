---
source_pdf: GENXD.pdf
paper_sha256: e9402454e1f495442d6c81da5a1932583d6de23a7d57198604415d96b5f600a9
processed_at: '2026-08-04T21:12:47-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GenXD 用人话讲讲

## 一句话总结

**GenXD 是一个 unified model，能同时做 3D 生成（static scene 的多视角图）和 4D 生成（dynamic scene 的 video），输入可以是 1 张图、3 张图、或任意张图。**

核心 trick：把 "多视角信息" 和 "时间信息" 在网络里 disentangle 开，让 3D data（没有 temporal）和 4D data（有 temporal）能在同一个 model 里互相补着训练。

---

## 1. 为什么这事难？

### 1.1 4D data 缺

2D generation 火了是因为 internet 上有海量图。3D generation 也能凑合，因为有 Objaverse 这种 synthetic mesh dataset。**但 4D data 呢？** real-world 的 4D 几乎不存在——你没见过哪个 dataset 给你 "一段真实世界视频 + 每帧的 camera pose + 标好的 moving object"。

唯一的来源是 real-world video。但 video 本身没 camera pose annotation，而且 video 里既有 camera 在动也有 object 在动，这俩是 entangled 的，你分不开。

### 1.2 3D 和 4D 怎么统一？

3D data：一堆 static 的多视角图，比如 Objaverse 里一个 mesh 从 12 个角度 render 出来。

4D data：video，既是 multi-view（camera 在动）又是 temporal（object 在动）。

**这俩共享 spatial representation，但 4D 多了 temporal。** 所以 paper 的核心 idea：在 network 里分两条 pathway，一条处理 multi-view consistency，一条处理 temporal dynamics，用一个 learnable weight α 控制是否 fuse temporal pathway。3D data 时 α=0（bypass temporal），4D data 时 α learnable。

---

## 2. CamVid-30K 是怎么造出来的？

### 2.1 Camera Pose Estimation

SfM 是标准方法，但 SfM 要 match feature 在 **static** region 上。如果 moving object 的 pixel 进了 feature matching，camera pose 就废了。

之前的工作 Particle-SfM (Zhao et al., 2022) 用一个 motion segmentation module 分 moving/static。问题是这个 module 在 wild video 上 generalize 差——**漏判 moving pixel**（false negative），导致 camera pose 错。

GenXD 的解法：**用 Mask2Former（instance segmentation）贪婪地把所有可能 moving 的 pixel 都 mask 掉**。宁可误杀静态 region（false positive），也不能漏判动态 region。因为 Mask2Former 在 COCO 等大类上 train 过，generalize 好得多。

然后 Particle-SfM 在 mask 后的 static region 上跑，得到 camera pose + sparse 3D point cloud。

### 2.2 Object Motion Estimation —— 这块最有意思

有了 camera pose 后，还要判断 video 里到底有没有 object 在动。**直接用 optical flow 不行**，因为 optical flow = camera motion + object motion 混在一起。

#### Step 1: 把 sparse depth 对齐到 dense depth

SfM 只能在 static region 给出 sparse depth（因为只在 static feature 上做 3D reconstruction）。但 dynamic region 也要 depth 才能 back-project object。

所以 GenXD 用 Depth Anything V2 (Yang et al., 2024) 预测 dense **relative** depth $d_{\text{rel}} \in [0,1]$。relative depth 没绝对 scale，所以要 align 到 SfM 的 sparse depth scale：

$$\alpha = \frac{\text{median}(d_{\text{SfM}})}{\text{median}(d_{\text{rel}})}, \quad \beta = \text{median}(d_{\text{SfM}} - \alpha \cdot d_{\text{rel}})$$
$$d_{\text{aligned}} = \alpha \cdot d_{\text{rel}} + \beta$$

变量说明：
- $d_{\text{SfM}}$: SfM 给的 sparse depth（只在 static region 有值）
- $d_{\text{rel}}$: Depth Anything V2 给的 dense relative depth
- $\alpha$: scale factor
- $\beta$: shift factor
- 用 **median** 而不是 mean，因为 median 对 SfM outlier 鲁棒

#### Step 2: Object Motion Field

**关键 insight**：如果 object 是 static 的，那么把 object 上一个 keypoint 从 frame $i$ back-project 到 3D，再 project 到 frame $j$，应该和 frame $j$ 中实际 tracked 到的 keypoint 位置重合。如果 object 真的在动，这俩位置会偏离。

具体：
1. 用 TAPIR (Doersch et al., 2023) 在 video 里 track object 上的 keypoints，得到 frame $j$ 的实际位置 $(u_j, v_j)$
2. 用 aligned depth 把 frame $i$ 的 keypoint back-project 到 3D：
$$kp_i = Z_i \cdot K^{-1} \cdot (u_i, v_i, 1)^T$$
   - $Z_i = d_{\text{aligned}}(u_i, v_i)$: 该 pixel 处的 dense depth
   - $K$: camera intrinsics
   - $K^{-1}$: inverse intrinsics，把 2D pixel 反投影到 3D camera space
3. 用 camera pose 把 3D keypoint project 到 frame $j$，得到预测位置 $(u_{ij}, v_{ij})$（**假设 object 静止**）
4. Motion field 是实际位置和预测位置的差：
$$(\Delta u_{ij}, \Delta v_{ij}) = \left(\frac{u_j - u_{ij}}{W}, \frac{v_j - v_{ij}}{H}\right)$$
   - 除以 $W, H$ 是 normalize
   - 如果 object 真静止，$\Delta u, \Delta v \approx 0$
   - 如果 object 在动，这个值会大

#### Step 3: Motion Strength

每个 video 取所有 object 中 max 的 motion magnitude 作为 motion strength。这个值有两大用途：
1. **Filtering**：剔除纯 static scene（没明显 object 动的 video）
2. **Training signal**：作为 condition 加到 temporal layer，告诉 network "这个 video 的 object 动得多厉害"

最后得到约 **30K real-world 4D videos** with camera pose + motion strength。

---

## 3. GenXD Model 架构

### 3.1 Mask Latent Conditioning —— 怎么支持任意张 input view？

之前方法有两个套路：

**套路 A: Concatenation**
把 condition image 的 latent concat 到 target latent 输入 U-Net。问题：必须**固定** input view 数量（比如固定 1 张），改 view 数量要改 channel。

**套路 B: CLIP embedding + cross-attention**
把 condition image encode 成 CLIP embedding，用 cross-attention 注入。问题：丢失多 view 之间的**位置关系**（CLIP embedding 是 global 的，没 spatial positional info），且要额外 CLIP encoder + cross-attention layer。

**GenXD 用 Mask Latent Conditioning**（idea 来自 CAT3D (Gao et al., 2024) 和 Video Interpolation Diffusion (Jain et al., 2024)）：
1. 所有 input/target images 都用 VAE encode 成 latent
2. **Target frames 加 Gaussian noise**（正常 diffusion）
3. **Condition frames 不加 noise**（保持 clean latent）
4. 全部按 sequence 位置输入 U-Net

**好处**：
1. **任意张** input view，不用改 network parameter
2. Condition frame **位置自由**（SVD 等强制 condition 必须在 first frame）
3. **不用 cross-attention** → 参数大幅减少

### 3.2 Plücker Ray 做 Camera Condition

Camera pose 怎么 inject 进网络？之前有些工作（Zero-1-to-3 (Liu et al., 2023b), ZeroNVS (Sargent et al., 2024)）把 camera extrinsics 转成 1D embedding，用 cross-attention 注入。

GenXD 用 **Plücker ray** —— 一个 6D 的 dense per-pixel camera representation：

$$\mathbf{r} = \langle \mathbf{d}, \mathbf{o} \times \mathbf{d} \rangle \in \mathbb{R}^6$$

变量说明：
- $\mathbf{o} \in \mathbb{R}^3$: camera center（光心位置，world space）
- $\mathbf{d} \in \mathbb{R}^3$: ray direction（camera center 到某个 pixel 的射线方向）
- $\mathbf{o} \times \mathbf{d}$: cross product，编码 camera center 和 ray direction 的几何关系
- 最终每个 pixel 都有一个 6D embedding

**直觉**：1D embedding 是 global 的，所有 pixel 共享。Plücker ray 是 **per-pixel** 的，每个 pixel 都知道 "这个 pixel 对应的 ray 从哪儿来、指向哪儿"，信息密度高得多。Table 7 ablation 证实：Plücker ray 在 Re10K 上 PSNR 22.96，Camera CA (cross-attention with 1D) 只有 21.73。

### 3.3 MultiView-Temporal Modules with α-Fusing —— 核心 design

GenXD 的 U-Net 在每个 block 里有两个 sub-layer：

**MultiView layer**：multi-view conv + self-attention，处理 cross-view consistency
**Temporal layer**：temporal conv + self-attention，处理 temporal dynamics

然后 α-fusing：
$$h_{\text{out}} = h_{\text{multiview}} + \alpha \cdot h_{\text{temporal}}$$

- **3D data**: $\alpha = 0$ 硬编码，temporal pathway bypass
- **4D data**: $\alpha$ learnable，融合 multi-view + temporal

**为什么这样设计 work**？

想象 3D 和 4D data 共享 spatial representation learning（都在 multi-view layer 里学）。如果 3D data 也通过 temporal layer，temporal layer 会试图找 temporal pattern，但 3D data 没有 temporal，会学 garbage。所以用 α=0 gate 掉。

4D data 则两个 layer 都用，因为既要 multi-view consistency 又要 temporal dynamics。

**Motion Strength 怎么用？**

SVD 等用 FPS 控制 motion magnitude，但 FPS 不区分 camera motion 和 object motion。GenXD 的 motion strength 是**纯 object motion**（已 disentangle），把它和 diffusion timestep $t$ 一起加到 temporal ResBlock：
$$h_{\text{temporal}} = \text{ResBlock}(h, \text{embed}(t) + \text{embed}(\text{motion\_strength}))$$

Fig 7 显示：增大输入 motion strength → 生成的视频里 car 开得更快。说明 motion strength 是 disentangled 的有效控制信号。

### 3.4 Lifting to 3D Representation

GenXD 生成的是 multi-view images / 4D video。要 render arbitrary view 还得 lift 到 explicit 3D representation：
- **3D**: 3D-GS (Kerbl et al., 2023) 或 Zip-NeRF (Barron et al., 2023)
- **4D**: 4D-GS (Wu et al., 2024a)

**关键区别 vs Animate124 (Zhao et al., 2023)**：Animate124 用 SDS (Score Distillation Sampling, Poole et al., 2022) 优化 4D NeRF，要 7 小时。GenXD **直接 generate 4D video 再 optimize 4D-GS**，只要 4 分钟——**100× speedup**。Table 3 显示 GenXD CLIP-I 90.32 vs Animate124 85.44，还解决 semantic drift 问题。

---

## 4. Training 三阶段

1. **Stage 1**: 只用 3D data，500K iterations → 建立 spatial prior
2. **Stage 2**: 3D + 4D data joint，single-view mode，500K iterations
3. **Stage 3**: single-view + multi-view joint，500K iterations

**初始化**：部分 init 自 SVD (Blattmann et al., 2023) pretrained weights——multiView layer 和 temporal layer 都从 SVD temporal layer 初始化。Cross-attention 删掉。这能加速 convergence，因为 SVD 已经学了 video temporal prior。

**训练资源**：32× A100, batch 128, 256×256 resolution, AdamW lr=5e-4。

---

## 5. Experiments 数据看 intuition

### 5.1 4D Scene Generation (Table 2, Cam-DAVIS benchmark)

| Method | FID ↓ | FVD ↓ |
|---|---|---|
| MotionCtrl | 118.14 | 1464.08 |
| CameraCtrl | 138.64 | 1470.59 |
| GenXD (1 view) | 101.78 | 1208.93 |
| GenXD (3 views) | **55.64** | **490.50** |

**Insight**: 3 views vs 1 view, FID 降 46%, FVD 降 59%。多 view condition 提供强 consistency constraint，让生成质量大幅提升。MotionCtrl/CameraCtrl 都基于 frozen SVD 加 camera branch，camera trajectory alignment 差且 object motion 弱。

### 5.2 Few-View 3D Reconstruction (Table 4)

| Baseline | PSNR↑ | + GenXD | 提升 |
|---|---|---|---|
| Zip-NeRF on Re10K | 20.58 | 25.40 | +4.82 |
| Zip-NeRF on LLFF (OOD) | 14.26 | 19.39 | +5.13 |
| 3D-GS on Re10K | 18.84 | 23.13 | +4.29 |
| 3D-GS on LLFF | 17.35 | 19.43 | +2.08 |

**Insight**: GenXD 作 generative prior，能补全 sparse-view 输入，让 Zip-NeRF/3D-GS 在 few-view 上 PSNR 提 4-5 dB。LLFF 是 OOD，提升更显著——说明 GenXD 学到了 generalizable prior 而非过拟合训练分布。

### 5.3 Ablation: Motion Disentangle (Table 5)

| Variant | Re10K PSNR | Cam-DAVIS FVD |
|---|---|---|
| w/o α-fusing | 20.75 | 1488.47 |
| GenXD (full) | 22.96 | 1208.93 |

去掉 α-fusing，3D 和 4D 都崩——3D data 被错误地通过 temporal layer 处理，4D data 也没法 disentangle camera vs object motion。

### 5.4 Ablation: Joint Training (Table 7)

| Setting | Re10K PSNR | Cam-DAVIS FVD |
|---|---|---|
| w/o 3D data | 16.38 | 1262.12 |
| w/o 4D data | 20.74 | 1240.57 |
| Full | 22.96 | 1208.93 |

**Insight**: 
- 去 3D data → PSNR 暴跌 6.58（3D data 提供 camera pose 多样性，没了 camera alignment 全崩）
- 去 4D data → PSNR 降 2.22（4D data 主要帮 object motion learning）

**3D 和 4D data 确实互补**——验证 paper 核心 hypothesis。

---

## 6. 我的 Intuition

### 6.1 为什么这工作重要？

之前的 3D 生成工作和 4D 生成工作是**分开**做的。3D 工作不处理 dynamic，4D 工作只 object-centric 不处理 scene-level。GenXD 第一个把 general 3D + 4D unify 到一个 model，而且实验显示 **unified training 比单独训更好**——3D 和 4D data 互相增强。

### 6.2 核心 contribution 怎么 break down？

1. **CamVid-30K dataset**：通过 back-project + re-project 的差值间接 measure object motion，避开 direct 3D motion estimation 的 scale ambiguity。Elegant 的设计。

2. **Mask Latent Conditioning**：解决任意张 input view 的问题，顺便删掉 cross-attention 省参数。

3. **Plücker Ray**：dense per-pixel camera encoding，比 1D global embedding 信息密度高得多。

4. **α-Fusing**：用一个 learnable scalar gate 把 temporal pathway 在 3D data 上 bypass 掉，让 3D/4D joint training 变 feasible。

### 6.3 Limitations 直觉

Paper 自己说的两个 limitation：

1. **Real-world dataset diversity 差**：Re10K 等多 forward-facing，没有 360° coverage → 单图生成复杂 scene 的 360° view 难。

2. **Large camera motion + Large object motion 难以同时出现**：video data 里大 camera motion 时 object 通常小动；大 object motion 时 camera 通常 static。这种 correlation 让 GenXD 难同时学两者。

这两个都是 **data limitation** 而非 model limitation——更多更好的 data 能直接解决。

---

## 7. 相关参考链接

**Core Paper & Project**:
- GenXD Project: https://gen-x-d.github.io
- arXiv (推测): 搜 "GenXD Generating Any 3D and 4D Scenes"

**Backbone / 基础模型**:
- Latent Diffusion (Stable Diffusion): https://arxiv.org/abs/2112.10752
- Stable Video Diffusion: https://arxiv.org/abs/2311.15127
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 4D Gaussian Splatting: https://arxiv.org/abs/2402.13210
- Zip-NeRF: https://arxiv.org/abs/2304.06785

**Data Curation 相关**:
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- TAPIR (keypoint tracking): https://arxiv.org/abs/2306.08637
- Particle-SfM: https://arxiv.org/abs/2207.08787
- Mask2Former: https://arxiv.org/abs/2112.01527
- Objaverse: https://objaverse.allenai.org
- Objaverse-XL: https://arxiv.org/abs/2307.05663
- DAVIS (video segmentation benchmark): https://davischallenge.org
- RealEstate10K: https://google.github.io/realestate10k/
- LLFF: https://local-light-fields.github.io
- Co3D: https://github.com/facebookresearch/co3d
- MVImgNet: https://github.com/robodmt/mvimgnet

**Baselines / 对比工作**:
- CAT3D: https://arxiv.org/abs/2405.10314
- SV4D: https://arxiv.org/abs/2407.17470
- SV3D: https://arxiv.org/abs/2403.12008
- CamCo: https://arxiv.org/abs/2406.02509
- MotionCtrl: https://arxiv.org/abs/2312.12795
- CameraCtrl: https://arxiv.org/abs/2404.02101
- Zero-1-to-3: https://arxiv.org/abs/2303.11328
- Animate124: https://arxiv.org/abs/2311.14603
- DreamFusion (SDS): https://arxiv.org/abs/2209.14988
- RealFusion: https://arxiv.org/abs/2302.12663
- IM-3D: https://arxiv.org/abs/2402.08682
- ImageDream: https://arxiv.org/abs/2312.02201
- One2345++: https://arxiv.org/abs/2402.19207
- ReconFusion: https://arxiv.org/abs/2312.02981
- Diffusion4D: https://arxiv.org/abs/2405.16645
- L4GM: https://arxiv.org/abs/2406.10324

**数学/几何**:
- Plücker coordinates: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
- Structure from Motion 综述: https://en.wikipedia.org/wiki/Structure_from_motion

---

## 8. 总结

GenXD 的故事一句话讲：**用 back-project/re-project 差值测 object motion → 造 30K real-world 4D dataset → 用 mask latent conditioning 支持任意 view 输入 → 用 Plücker ray 做 dense camera encoding → 用 α-fusing disentangle spatial/temporal → 让 3D 和 4D data 在 unified model 里互相补着训 → 实验证明 unified 比 separate 训更好**。

这套设计的 elegance 在于每个 component 都解决一个具体 bottleneck，且 solution 都是 mathematically well-motivated 而非 hack。CamVid-30K 和 code 承诺开源（撰写时尚未释放），如果能 release 出来对 community 推动 4D research 应该很大。

---

# GenXD: Generating Any 3D and 4D Scenes — 深度技术讲解

## Paper Overview

GenXD 是 Microsoft 与 NUS 合作的工作，核心目标是**统一 3D 和 4D scene 生成**。这里的 "3D" 指纯 spatial view changes（静态场景多视角），"4D" 指 dynamic scene（同时含 camera motion 与 object motion）。Project page: https://gen-x-d.github.io

关键 insight 是：**3D 和 4D data 在 spatial representation 上是相似的，差异只在 temporal dimension**。因此可以通过 disentangle spatial/temporal information 让 3D 和 4D data 互相补充训练。这点在之前的工作（如 CAT3D, SV4D, CamCo）都没做到 unified。

---

## 1. CamVid-30K Dataset Construction

### 1.1 Why Need New Dataset?

4D data 的核心困难：需要同时具备 **multi-view spatial info + temporal dynamics**。Synthetic 4D data（如 Objaverse-XL-Animation）只能覆盖 object-centric 场景，无法 cover real-world scenes。Real-world video 是唯一来源，但 video 缺少 camera pose 标注，且无法区分 camera motion vs object motion。

### 1.2 Camera Pose Estimation Pipeline

**核心思路**: SfM (Structure-from-Motion) 对 static region 估计 camera pose，但要 mask 掉 moving objects 否则会污染 feature matching。

具体步骤：
1. 用 instance segmentation model (Cheng et al., 2022, Mask2Former) **贪婪地 segment 所有可能移动的 pixels**
2. Particle-SfM (Zhao et al., 2022) 在 static part 上做 SfM
3. 得到 camera poses + sparse 3D point clouds

**关键设计选择**: 偏好 false positive（误判静态为动态）over false negative（漏判动态为静态）。因为漏判会导致 moving pixels 进入 SfM feature matching → camera pose estimation 严重错误。Mask2Former 比 Particle-SfM 的 motion segmentation module 更 generalize。

### 1.3 Object Motion Estimation — 最 mathematically interesting 的部分

#### Problem Formulation
有了 camera pose 后，需要判断 video 中是否真的有 object movement（而非纯 static scene 或仅 camera motion）。直接用 optical flow 不行，因为 optical flow = camera motion + object motion 的混合。

#### Depth Alignment (Eq. 1 & Eq. 2)

**Eq. 1** — SfM sparse depth 投影：
$$P_{\text{camera}} = R \cdot P_{\text{world}} + t, \quad (u, v, 1)^T = K \cdot (X_c/Z_c, Y_c/Z_c, 1)^T$$

变量解释：
- $P_{\text{world}} \in \mathbb{R}^3$: world space 中的 3D point（来自 SfM 稀疏点云）
- $R \in \mathbb{R}^{3\times3}$: world→camera rotation matrix
- $t \in \mathbb{R}^3$: world→camera translation
- $P_{\text{camera}} = (X_c, Y_c, Z_c)$: camera space 坐标
- $K \in \mathbb{R}^{3\times3}$: camera intrinsics（focal length + principal point）
- $(u, v)$: image pixel 坐标
- $d_{\text{SfM}}(u, v) = Z_c$: sparse depth at pixel $(u,v)$

**Eq. 2** — Monocular depth alignment：
$$\alpha = \text{median}(d_{\text{SfM}}) / \text{median}(d_{\text{rel}}), \quad \beta = \text{median}(d_{\text{SfM}} - \alpha \cdot d_{\text{rel}})$$
$$d_{\text{aligned}} = \alpha \cdot d_{\text{rel}} + \beta$$

变量解释：
- $d_{\text{rel}} \in [0,1]^{H \times W}$: pre-trained Depth Anything V2 (Yang et al., 2024) 预测的 relative depth
- $\alpha$: scale factor（median-based robust estimation）
- $\beta$: shift factor
- $d_{\text{aligned}}$: 与 SfM 深度尺度对齐的 dense depth

**为什么用 median 而非 mean?** median 对 outlier 鲁棒。SfM sparse points 可能含 mismatch outlier。

#### Object Motion Field (Eq. 3 & Eq. 4)

**核心想法**: 把 dynamic object 上的 keypoint 从 frame $i$ back-project 到 3D，再 project 到 frame $j$，比较实际位置与预测位置之差。

**Eq. 3** — Back-projection:
$$kp_i = Z_i \cdot K^{-1} \cdot (u_i, v_i, 1)^T$$

变量解释：
- $(u_i, v_i)$: keypoint 在 frame $i$ 的 2D 像素位置
- $Z_i = d_{\text{aligned}}(u_i, v_i)$: 该 pixel 处的 aligned depth
- $K^{-1}$: inverse intrinsics
- $kp_i \in \mathbb{R}^3$: 3D keypoint position in camera space

**Eq. 4** — Motion field:
$$(\Delta u_{ij}, \Delta v_{ij})^T = ((u_j - u_{ij})/W, (v_j - v_{ij})/H)^T$$

变量解释：
- $(u_j, v_j)$: keypoint 在 frame $j$ 的实际 tracked 位置（由 TAPIR (Doersch et al., 2023) 得到）
- $(u_{ij}, v_{ij})$: keypoint 由 frame $i$ 经 3D projection 到 frame $j$ 的**预测位置**（假设 object 静止）
- $H, W$: image 高宽
- 除以 $W, H$ 是为了 normalize 到 $[0,1]$ 范围

**关键 insight**: 如果 object 真的 static，那么 back-project + re-project 后预测位置应该与实际 tracked 位置重合，motion field ≈ 0。Object 真动时，预测位置会偏离实际位置。

**Motion strength**: 每个 video 取所有 object 的 max motion magnitude。这个值用于:
1. Filtering: 排除纯 static scenes
2. **Training signal**: 作为 temporal layer 的 condition，提供 object motion 大小提示

### 1.4 Dataset Statistics

最终获得 **~30K real-world videos with camera poses + object motion strength**。与 44K synthetic data from Objaverse-XL-Animation 合并训练。

---

## 2. GenXD Architecture

### 2.1 Mask Latent Conditioned Diffusion

#### Base Framework
Latent Diffusion Model (Rombach et al., 2022, LDM):

**Eq. 5** — Training objective:
$$L_{\text{LDM}} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|_2^2 \right]$$

变量解释：
- $\mathcal{E}(x)$: VAE encoder，image→latent
- $z_t$: noisy latent at diffusion timestep $t$
- $\epsilon \sim \mathcal{N}(0,1)$: Gaussian noise
- $\epsilon_\theta(\cdot)$: denoising U-Net with params $\theta$
- $c$: condition（text/image/camera）

#### Mask Latent Conditioning 机制

之前工作的两种 conditioning 方法都有问题：
1. **Concatenation** (Blattmann et al., 2023; Voleti et al., 2024): 把 condition latent concat 到 target latent → 必须固定 input views 数量，改 channel
2. **CLIP embedding via cross-attention** (Liu et al., 2023b; Radford et al., 2021): 可以支持多 views，但**丢失 positional info among views**，且需要额外 CLIP encoder

GenXD 采用 mask latent conditioning (源自 CAT3D (Gao et al., 2024), Video Interpolation Diffusion (Jain et al., 2024)):
- Reference frames **不加 noise**（保持 clean latent）
- Target frames 加 Gaussian noise
- 直接作为 sequence 输入 denoising U-Net
- 通过位置隐式表达多 view 之间的 spatial relationship

**Three benefits** (paper 明确列出):
1. 任意数量 input views，无需修改 network parameters
2. Condition frame 位置不受限（其他工作如 SVD 强制 condition 在 first frame）
3. 可移除 cross-attention layers → 大幅减少 parameters

### 2.2 Plücker Ray Camera Condition

**Eq. 6**:
$$\mathbf{r} = \langle \mathbf{d}, \mathbf{o} \times \mathbf{d} \rangle \in \mathbb{R}^6$$

变量解释：
- $\mathbf{o} \in \mathbb{R}^3$: camera center（光心位置）
- $\mathbf{d} \in \mathbb{R}^3$: ray direction（camera center 到 pixel 的射线方向）
- $\mathbf{o} \times \mathbf{d}$: cross product，编码 camera center 与 ray direction 的几何关系
- $\mathbf{r} \in \mathbb{R}^6$: 6D Plücker ray embedding

**为什么用 Plücker ray?** (源自 Plücker 1828 的几何代数)
- **Dense per-pixel encoding**: 不是 global camera embedding，而是每个 pixel 都有对应的 6D ray
- 隐式 encode camera pose + intrinsics
- 比 1D camera embedding + cross-attention（如 Zero-1-to-3 (Liu et al., 2023b)）信息更丰富

Table 7 ablation 显示 Camera CA (cross-attention with 1D embedding) 在 Re10K 上 PSNR 21.73 vs GenXD 22.96，LLFF 上 17.15 vs 17.94，Cam-DAVIS FVD 1331.62 vs 1208.93 — 全面劣于 Plücker ray。

### 2.3 MultiView-Temporal Modules with α-Fusing

这是 GenXD 最核心的设计。问题：3D data 只有 multi-view（no temporal），4D data 有 multi-view + temporal。如何在一个 model 中统一处理？

#### Architecture Layout

GenXD 的 U-Net 在每个 ResBlock 和 Transformer block 中插入:
1. **MultiView layer**: spatial conv + self-attention，处理 cross-view consistency
2. **Temporal layer**: temporal conv + self-attention，处理 temporal dynamics
3. **α-fusing**: learnable weight α 控制是否融合 temporal info

#### α-Fusing Mechanism

形式化（推导自 paper Sec 4.1 + Fig 4）:
$$h_{\text{out}} = h_{\text{multiview}} + \alpha \cdot h_{\text{temporal}}$$

- 3D data: $\alpha = 0$（hard-coded），temporal pathway 被 bypass
- 4D data: $\alpha$ learnable，fusion multi-view + temporal

**为什么这样设计 work?** 3D 和 4D 的 spatial 信息 encoding 是 shareable 的，但 temporal 只在 4D 中存在。通过 $\alpha$ gate，可以保证:
- 3D data 不会污染 temporal learning pathway
- 4D data 仍能 leverage spatial multi-view learning

#### Motion Strength Integration

Video generation models (SVD (Blattmann et al., 2023), MagicVideo (Zhou et al., 2022)) 通常用 FPS 或 motion id 控制 motion magnitude，但**没考虑 camera movement**。

GenXD 的 motion strength 来自 CamVid-30K curation，是 object motion 的真实度量。作为 scalar 与 diffusion timestep $t$ 一起加到 temporal ResBlock:
$$h_{\text{temporal}} = \text{ResBlock}(h, t + \text{embedding}(\text{motion\_strength}))$$

Fig 7 ablation 显示: 增大 motion strength 输入 → 视频 car 速度变快。证明 motion strength 是 disentangled 的有效控制信号。

### 2.4 Lifting to 3D Representations

GenXD 输出 multi-view images/videos，需要 lift 到 explicit 3D representation 才能 render arbitrary views。具体:
- **3D generation**: 3D-GS (Kerbl et al., 2023) 或 Zip-NeRF (Barron et al., 2023)
- **4D generation**: 4D-GS (Wu et al., 2024a)

**与 SDS (Poole et al., 2022) 区别**: GenXD **不做 score distillation**，直接用 generated views 作 supervision 优化 3D representation。这是为什么 GenXD 比 Animate124 快 100×（4min vs 7hrs）。

---

## 3. Training Strategy

### 3.1 三阶段训练

1. **Stage 1**: 只用 3D data 训 500K iterations（建立 spatial prior）
2. **Stage 2**: 3D + 4D data joint training，single-view mode，500K iterations
3. **Stage 3**: Single-view + multi-view mode joint training，500K iterations

### 3.2 初始化

GenXD 部分初始化自 Stable Video Diffusion (SVD) pretrained weights:
- MultiView layer ← SVD temporal layer
- Temporal layer ← SVD temporal layer
- Cross-attention layers **removed**

**为什么这样初始化?** SVD 学了 video temporal prior，可以迁移到 multi-view + temporal learning，加速 convergence。

### 3.3 Training Config

- 32× A100 GPUs
- Batch size 128
- Resolution 256×256
- AdamW optimizer, lr = 5e-4
- Stage 1: center crop to square
- Stage 3: square by crop OR padding → support 各种 image ratio

---

## 4. Experiments 深度分析

### 4.1 4D Scene Generation (Table 2)

Benchmark: **Cam-DAVIS** — paper 新提出，对 DAVIS (Perazzi et al., 2016) video 重新 annotate camera pose + filter 有 object motion 的 20 个 video。**OOD camera trajectory**，测 robustness。

| Method | FID ↓ | FVD ↓ |
|---|---|---|
| MotionCtrl | 118.14 | 1464.08 |
| CameraCtrl | 138.64 | 1470.59 |
| GenXD (Single View) | 101.78 | 1208.93 |
| GenXD (3 Views) | **55.64** | **490.50** |

**关键观察**: 3-view conditioning 比 single-view FID 降 46%，FVD 降 59%。Multi-view 提供 strong consistency constraint，让 generation 质量大幅提升。MotionCtrl/CameraCtrl 都用 SVD 作 base，但 camera trajectory alignment 差且 object motion 弱。

### 4.2 4D Object Generation (Table 3)

| Method | Time ↓ | CLIP-I ↑ |
|---|---|---|
| Zero-1-to-3-V | 4 hrs | 79.25 |
| RealFusion-V | 5 hrs | 80.26 |
| Animate124 | 7 hrs | 85.44 |
| GenXD (Single View) | **4 min** | **90.32** |

**100× speedup** 来自不做 SDS optimization，直接 generate 4D video 然后 optimize 4D-GS。CLIP-I 提升 4.88 vs Animate124，且解决 semantic drift 问题。

### 4.3 Few-View 3D Reconstruction (Table 4)

| Baseline | Method | Re10K PSNR | SSIM | LPIPS | LLFF PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|
| Zip-NeRF | baseline | 20.58 | 0.729 | 0.382 | 14.26 | 0.327 | 0.613 |
| Zip-NeRF | + GenXD | 25.40 | 0.858 | 0.223 | 19.39 | 0.556 | 0.423 |
| 3D-GS | baseline | 18.84 | 0.714 | 0.286 | 17.35 | 0.489 | 0.335 |
| 3D-GS | + GenXD | 23.13 | 0.808 | 0.202 | 19.43 | 0.554 | 0.312 |

**Key takeaway**: GenXD 作 generative prior，能补全 sparse-view 输入，让 Zip-NeRF/3D-GS 在 few-view setting 上 PSNR 提升 4-5 dB。LLFF 是 OOD，提升更显著（Zip-NeRF +5.13 dB）。

### 4.4 Single-View 3D Generation (Table 6, Appendix B)

| Method | Type | Time (min) | CLIP-I |
|---|---|---|---|
| ImageDream | 3D | 120 | 83.77 |
| One2345++ | 3D | 0.75 | 83.78 |
| IM-3D | 3D | 3 | 91.40 |
| GenXD | 3D&4D | 2 | 84.75 |

GenXD 比 IM-3D 略低（IM-3D 91.40 vs 84.75），但 IM-3D 是专门 3D model。GenXD 是 unified 3D/4D model，且能解决 over-saturation 和 Janus problem（SDS-based 方法常见 issue）。

### 4.5 Ablation: Motion Disentangle (Table 5)

| Method | Re10K PSNR/SSIM/LPIPS | LLFF | Cam-DAVIS FID/FVD |
|---|---|---|---|
| w/o Motion Disentangle | 20.75/0.635/0.362 | 16.89/0.397/0.560 | 122.73/1488.47 |
| GenXD | 22.96/0.774/0.341 | 17.94/0.463/0.546 | 101.78/1208.93 |

**Removing α-fusing 的代价**: 
- Re10K PSNR -2.21
- SSIM 0.635 vs 0.774（大幅下降）
- Cam-DAVIS FVD +279.54

证明 disentanglement 对 3D 和 4D **都**重要 — 不 disentangle 时 3D data 也会被 temporal module 误处理。

### 4.6 Ablation: Joint Training (Table 7)

| Setting | Re10K PSNR | LLFF PSNR | Cam-DAVIS FVD |
|---|---|---|---|
| w/o 3D Data | 16.38 | 14.98 | 1262.12 |
| w/o 4D Data | 20.74 | 17.35 | 1240.57 |
| GenXD (full) | 22.96 | 17.94 | 1208.93 |

**Insight**: 
- 移除 3D data → PSNR 暴跌 6.58（3D data 提供 camera pose 多样性）
- 移除 4D data → PSNR 降 2.22，但 FVD 仅升 31（4D data 主要帮助 object motion learning）

3D 和 4D data 确实互补 — 验证 paper 的核心 hypothesis。

---

## 5. 与 Related Work 的对比直觉

### 5.1 vs CAT3D (Gao et al., 2024)
- CAT3D: multi-view diffusion for 3D，用 mask latent conditioning
- GenXD **borrow** mask latent conditioning 思路，但 extend 到 4D，引入 multiview-temporal modules + α-fusing
- GenXD 可以视为 CAT3D 的 4D extension

### 5.2 vs SV4D (Xie et al., 2024)
- SV4D: 单 image → multi-view 4D object
- GenXD 支持 single **and** multi-view input，scene-level **and** object-level，更 general
- SV4D 用 static multi-view + temporal decomposition 但架构上没 α-fusing

### 5.3 vs CamCo (Xu et al., 2024)
- CamCo 也 annotate 4D data 类似，fine-tune video generation
- CamCo 受限 camera pose quality 和 diversity，处理不了 large camera motion
- GenXD 用 Plücker ray + multiview-temporal layer，更鲁棒

### 5.4 vs MotionCtrl (Wang et al., 2024) / CameraCtrl (He et al., 2024)
- 这两个: frozen video gen model + camera branch
- GenXD: full fine-tune with camera-aligned 4D data
- Table 2 显示 GenXD 显著优于二者

---

## 6. Limitations & Intuition

### 6.1 Dataset Diversity 限制
Real-world datasets (Re10K) 的 camera trajectory 多为 forward-facing，复杂场景缺 360° coverage → GenXD 难生成 complex scene 的 360° views from single image。

### 6.2 Large Camera + Large Object Motion 难
现有 video data 中: 大 camera motion 时 object motion 通常小；大 object motion 时 camera 通常 static。这种 correlation 让 GenXD 难以同时处理两者。

### 6.3 我的 Intuition
GenXD 的核心贡献在于把 3D/4D data 的 synergy 显式建模出来。**Mask latent conditioning** 解决 input flexibility，**α-fusing** 解决 spatial-temporal disentanglement，**Plücker ray** 解决 dense camera conditioning。三个组件 combine 让 unified training 变得 feasible。

Data curation pipeline 也是关键 — CamVid-30K 的 object motion field 设计很 elegant：通过 back-project + re-project 的差值间接 measure object motion，避免了 direct 3D motion estimation 的 scale ambiguity。

---

## 7. References & 相关工作链接

- Project page: https://gen-x-d.github.io
- Latent Diffusion (Rombach et al., 2022): https://arxiv.org/abs/2112.10752
- 3D Gaussian Splatting (Kerbl et al., 2023): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Zip-NeRF (Barron et al., 2023): https://arxiv.org/abs/2304.06785
- 4D-GS (Wu et al., 2024): https://arxiv.org/abs/2402.13210
- Stable Video Diffusion (Blattmann et al., 2023): https://arxiv.org/abs/2311.15127
- CAT3D (Gao et al., 2024): https://arxiv.org/abs/2405.10314
- SV4D (Xie et al., 2024): https://arxiv.org/abs/2407.17470
- CamCo (Xu et al., 2024): https://arxiv.org/abs/2406.02509
- MotionCtrl (Wang et al., 2024): https://arxiv.org/abs/2312.12795
- CameraCtrl (He et al., 2024): https://arxiv.org/abs/2404.02101
- Depth Anything V2 (Yang et al., 2024): https://arxiv.org/abs/2406.09414
- TAPIR (Doersch et al., 2023): https://arxiv.org/abs/2306.08637
- Particle-SfM (Zhao et al., 2022): https://arxiv.org/abs/2207.08787
- Mask2Former (Cheng et al., 2022): https://arxiv.org/abs/2112.01527
- Plücker coordinates (原始): https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
- Animate124 (Zhao et al., 2023): https://arxiv.org/abs/2311.14603
- DreamFusion SDS (Poole et al., 2022): https://arxiv.org/abs/2209.14988
- Zero-1-to-3 (Liu et al., 2023): https://arxiv.org/abs/2303.11328
- Objaverse (Deitke et al., 2023): https://objaverse.allenai.org
- Objaverse-XL (Deitke et al., 2024): https://arxiv.org/abs/2307.05663
- RealFusion (Melas-Kyriazi et al., 2023): https://arxiv.org/abs/2302.12663
- IM-3D (Melas-Kyriazi et al., 2024): https://arxiv.org/abs/2402.08682
- ImageDream (Wang & Shi, 2023): https://arxiv.org/abs/2312.02201
- One2345++ (Liu et al., 2024): https://arxiv.org/abs/2402.19207
- ReconFusion (Wu et al., 2024): https://arxiv.org/abs/2312.02981
- RealmDreamer (Shrivastava et al., 2024): https://realmdreamer.github.io
- SV3D (Voleti et al., 2024): https://arxiv.org/abs/2403.12008
- DAVIS dataset (Perazzi et al., 2016): https://davischallenge.org
- RealEstate10K (Zhou et al., 2018): https://google.github.io/realestate10k/
- LLFF (Mildenhall et al., 2019): https://local-light-fields.github.io
- Co3D (Reizenstein et al., 2021): https://github.com/facebookresearch/co3d
- MVImageNet (Yu et al., 2023): https://github.com/robodmt/mvimgnet

---

## 8. 总结

GenXD 的核心 contribution 是把 3D 和 4D data 的 training 显式 unify 到一个 architecture 里，通过 **multiview-temporal modules + α-fusing** 实现 spatial-temporal disentanglement。Data side 通过 CamVid-30K 的 curation pipeline 解决 4D scene data 缺失问题，object motion field 是数学上很 elegant 的设计。Model side 通过 **mask latent conditioning + Plücker ray + α-fusing** 让单一 model 能 handle 任意数量 condition views 的 3D/4D generation。

实验结果显示 unified training **不仅**没 sacrifice 单 task 性能，**反而**让 3D 和 4D generation 互相 benefit — 这是 paper 最强的 claim。代码与 dataset 承诺开源（撰写时仍未释放）。
