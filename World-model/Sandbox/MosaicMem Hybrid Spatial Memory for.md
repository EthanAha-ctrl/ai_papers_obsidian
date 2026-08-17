---
source_pdf: MosaicMem Hybrid Spatial Memory for.pdf
paper_sha256: cdb0c9c4401775f1ca050c123095ce2f4f9aa9eb8311a76b363018346ff15e3e
processed_at: '2026-08-05T20:32:52-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 MosaicMem

## 1. 这 paper 在解决什么真问题

你想用 video diffusion model 做 world simulator——给它一张图、一句 prompt、一条 camera 轨迹，它生成你能"进去逛"的 long video。核心卡点不是画质，是 **memory**。

举个具体场景：你拿着 camera 往前走 30 秒，看到一个红色沙发。你转身走 2 分钟，再转回来——model 还能认出这个沙发吗？还能生成同一个沙发从新角度看的样子吗？

现有方案两条路，都不行。

## 2. 为什么两条路都不行

**Explicit memory (3D cache 路线)**：把看到的东西建 point cloud 或 3D Gaussian。回来时投影到新视角做 condition。
- 问题：只能 render static scene。你说"沙发上跳上一只猫"，它干不了——因为 point cloud 是死的，它不理解 "dynamic"。
- 类比：像是你拿着一个乐高模型拍照，再换角度看——精确但没生命。

**Implicit memory (frame memory 路线)**：把之前看到的关键 frame 存起来，新视角生成时让 attention 自己 retrieve。
- 问题：你给它 perfect camera pose，它生成的 video 还是会 drift。走 2 分钟回来，沙发位置歪了。
- 类比：像你给 AI 一本相册让它"参考之前照片"——它能看图但不会精确对位，且每张照片都占大量 context window。

一个 dynamic 不行，一个 pose 不行。两条路互补但没人把它们结合对。

## 3. MosaicMem 的 key idea

**用 patch 作为 memory 的基本单位**——既不是 3D point (太死)，也不是整个 frame (太冗余)。

具体做三件事：

**(1) 每个 patch 带 3D 坐标标签**
看过的每个 patch，用 Depth Anything 估计 depth，算出它在 3D world 中的位置。这就像给每张小照片贴一张 3D GPS 标签。

**(2) 新视角来了，retrieve 相关 patch 并 warp 到对的位置**
回到沙发那个例子——你的 camera 想看沙发，model 把之前存的"沙发 patch"拿出来，用 multi-view geometry 公式算出"这个 patch 应该出现在新画面的哪个位置"，然后贴过去。

**(3) 把贴好的 patch 当 condition 喂给 DiT，让 model 自己 inpaint 剩下的**
贴了沙发 patch 之后，model 知道"沙发在这"，剩下的动态部分——猫跳上去、光影变化——交给 attention 自己生成。

这就是 paper 名字 Mosaic 的来源：把带坐标标签的小照片拼到新画面里，像马赛克拼贴。

## 4. 为什么 work

一个 surprising 的实验：把 MosaicMem 直接 plug 进 vanilla Wan 2.2，**完全不 finetune**，模型还是能正确把 memory patch 放到对的位置。

这说明 pre-trained video DiT 内部**已经隐含了 spatial correspondence 能力**——它知道"这块 region 应该对应之前看过的那块"。MosaicMem 只是给这个能力一个 explicit 的 interface 去激活它。

这其实是 deep learning 里一个反复出现的 pattern：model 已经会了，缺的是正确的"提问方式"。

## 5. 两个关键 trick

光有 patch 不够，还有两个 alignment 问题：

**(1) VAE 压缩导致坐标精度损失**
3D VAE 把 spatial 压 8×、temporal 压 4×，RoPE 坐标精度大幅下降。算出来 patch 应该贴到 $(u, v)$，但因为 compression，实际贴到的位置偏了。

解决：**Warped RoPE**——用 multi-view geometry 的 reprojection 公式 $\Pi(\mathbf{K}_j \mathbf{T}_j \mathbf{T}_i^{-1} \mathbf{K}_i^{-1} (u,v,D))$ 算出 patch 在新 view 的精确 fractional 坐标，保留小数部分，用高分辨率 sample RoPE。本质是 sub-pixel RoPE。

**(2) 光改坐标不够，feature 也得对齐**
**Warped Latent**——直接对 source latent 做 bilinear grid sampling，把 feature 也 warp 到新位置。

两个互补：Warped RoPE 管 "where" (位置编码层面)，Warped Latent 管 "what" (feature 层面)。混着用最稳。

## 6. PRoPE 干嘛的

MosaicMem 主要管 "memory 怎么贴回来"，但 camera 走得远或 memory 稀疏时，patch 不足以告诉 model 整个 camera motion。

**PRoPE (Projective Positional Encoding)** 把 camera projection matrix 直接 bake 进 attention 的 Q/K/V rotation——两个 token 互相 attend 时，attention score 自动 reflect 它们的 relative camera geometry。

一个细节 trick：VAE 把 time 压 4×，一个 latent frame 对应 4 个原始 frame。PRoPE "unfold" 一个 sub-index，让每个 sub-frame 用自己的 camera matrix——避免 frame-level averaging 损失精度。

## 7. 实验结果讲了什么

**对比 implicit memory (CaM, WorldMem)**:
- Camera error: MosaicMem 0.51° vs CaM 4.65°——10 倍 gap
- 这 gap 很直观：implicit memory 给 perfect pose 还是 drift，MosaicMem 几乎不 drift

**对比 explicit memory (GEN3C, SEVA)**:
- Dynamic score: MosaicMem 2.58 vs GEN3C 1.21——2 倍以上
- Explicit memory 本质是 render static scene，MosaicMem 能 prompt-driven 生成 dynamics

**Long-horizon**: 生成 2-minute navigation video，CaM 会逐渐 collapse，MosaicMem 稳住

**AR 蒸馏**: 把 bidirectional 模型用 Causal Forcing 蒸馏成 AR 版本，real-time 16 FPS @ 640×360，全面 beat RELIC 和 Matrix-Game 2.0

## 8. 真正的 insight

这篇 paper 的深层 insight 不在 patch 本身，而在这个观察：

> Video diffusion model 已经 implicitly 学会了 spatial correspondence，只是缺一个 explicit interface 去 unlock 它。

MosaicMem 的贡献是找到了这个 interface 的 sweet spot——比 3D structure 灵活 (能 handle dynamic)，比 frame memory 精确 (geometric grounded)，比 point cloud manipulable (patch 可直接 edit)。

**类比**：像是你给一个已经会画画的人一本带 GPS 标签的相册——他不需要重新学画画，只需要知道每张照片该贴到画布哪个位置，剩下的 creativity 留给他自己。

## 9. Limitation / 我的疑问

Paper 没深讲的几个点：

1. **Dynamic object 的 depth 估计**: 走动的人、移动的车，Depth Anything V3 估准了吗？Dynamic object 的 patch lifting quality 直接决定 memory 准不准
2. **Patch 大小怎么选**: 太小 retrieval cost 高，太大 alignment 不精确——paper 没讨论这个 trade-off
3. **Memory 增长**: 长期 navigation memory 越来越大，sparse retrieval 之外有没有 forgetting / pruning 机制？
4. **AR 蒸馏 loss**: bidirectional 蒸馏成 causal，consistency 是否完全保留？paper 只对比其他 AR system，没对比自己的 bidirectional baseline
5. **Training data bias**: 主要 UE5 + Cyberpunk + first-person，real-world long-tail scene (野外、医院、工厂) 表现未知

## 10. 一句话总结

MosaicMem 找到了 video world model memory 的 "missing middle"——用带 3D 坐标的 patch 作为 memory unit，既 unlock pre-trained DiT 的 implicit spatial capability，又保留 dynamic generation 的 flexibility，在 explicit-implicit spectrum 上找到了一个真正 work 的 hybrid point。

---

# MosaicMem: Hybrid Spatial Memory for Controllable Video World Models 深度解析

## 1. 背景: Video World Model 的 memory bottleneck

Video diffusion model 正从 short plausible clip 转向 world simulator——需要支持 camera motion、revisit、intervention 下的 consistency。Genie 3 (Google DeepMind, 2025) 是这个方向的 representative work，目标是 real-time interaction + long-duration persistence。核心瓶颈在 spatial memory。

现有 spatial memory 的两种 paradigm:

### Explicit spatial memory
- 代表: GEN3C (CVPR 2025) [1]、VMem (ICCV 2025) [2]、SEVA (ICCV 2025) [3]、VWM [4]
- 存储单元: 3D primitive (point cloud / 3D Gaussian splat / surfel)
- retrieval 方式: optics-based projection——把 cached 3D structure 投影到 queried viewpoint，通过 ControlNet 或 channel concat 注入
- 优势: geometry grounded by dedicated 3D inference，metric faithfulness 好
- 劣势: static scene friendly，multiple independently moving objects 难维护，global 3D reconstruction 导致 cross-view misalignment 累积

### Implicit spatial memory
- 代表: Context-as-Memory (SIGGRAPH Asia 2025) [5]、WorldMem (NeurIPS 2025) [6]、WorldPack [7]
- 存储单元: posed frame (或 frame-derived feature)
- retrieval 方式: 通过 DiT 的 attention mechanism 做 token concatenation
- 优势: end-to-end differentiable，handle dynamics/non-rigid 变化好
- 劣势: 即使给 perfect camera pose 仍然产生 inaccurate egomotion，drift over revisits，frame-based representation redundant，finite context window 限制 persistence

这里有个核心 insight: explicit 和 implicit 的 fundamental memory unit 不同——一个是 point/splat，一个是 entire frame。MosaicMem 提出了一个中间粒度: **patch**。

---

## 2. MosaicMem 核心思想: Patch-and-Compose

### 2.1 Task definition

给定:
- 真实输入 image $\mathcal{I}$
- text prompt 集合 $\mathcal{L} = \{\ell_1, \ldots, \ell_k\}$
- camera pose 序列 $\mathcal{C} = \{c_1, \ldots, c_T\}$

目标: 生成 long-horizon video rollout $\mathcal{X} = \{X_1, \ldots, X_T\}$，同时满足 camera trajectory adherence、spatial memory retrieval、text-prompt-driven dynamics。

### 2.2 Generative formulation (Flow Matching)

建立在 text+image-to-video (TI2V) model 之上，用 Flow Matching 学 joint distribution。$\lambda \in [0,1]$ 是 continuous flow time，$\mathcal{X}^\lambda = \{X_1^\lambda, \ldots, X_T^\lambda\}$ 是 flow time $\lambda$ 处的 video state。

从 Gaussian noise $\mathcal{X}^0 \sim \mathcal{N}(\bar{0}, \bar{I})$ 出发，学一个 neural vector field $u_\theta$ 把 $\mathcal{X}^0$ 传到 $\mathcal{X}^1$。probability-flow ODE 为:

$$
\frac{d\mathcal{X}^\lambda}{d\lambda} = u_\theta(\mathcal{X}^\lambda, \lambda \mid \mathcal{Z}, \mathcal{L}, \mathcal{A})
$$

$$
\mathcal{X}^1 = \mathcal{X}^0 + \int_0^1 u_\theta(\mathcal{X}^\lambda, \lambda \mid \mathcal{Z}, \mathcal{L}, \mathcal{C}, \mathcal{M}) \, d\lambda \quad (1)
$$

公式变量解释:
- $\lambda$: flow time，连续 0→1 表示从噪声到数据的演化进度
- $\mathcal{X}^\lambda$: flow time $\lambda$ 处的 video state
- $u_\theta$: 神经向量场，参数为 $\theta$
- $\mathcal{Z}$: latent 编码（VAE 编码后的 image/text 条件）
- $\mathcal{L}$: text prompt 集合
- $\mathcal{C}$: camera trajectory
- $\mathcal{M}$: spatial memory——这是 MosaicMem 引入的 key 新条件

**Intuition**: 这本质是 conditional flow matching，相比标准 TI2V 多了 $\mathcal{C}$ 和 $\mathcal{M}$ 两个 condition。memory 作为 conditional signal 注入 vector field，而不是直接 warp 输出。

### 2.3 Mosaic Memory 的 hybrid design

对每个 patch $\mathcal{P}$:
1. **Geometric lifting** (explicit-style front half): 用 off-the-shelf 3D estimator (Depth Anything V3 [8] 或 VIPE [9]) 推断 depth + camera，把 patch lift 到 3D
2. **Implicit conditioning** (implicit-style back half): 当 observer revisit patch $\mathcal{P}$，把 retrieved patch 作为 context 喂给 DiT，用 modified RoPE (§2.3) 传递 memory patch 和 noised latent token 之间的 correspondence

这个 "explicit lifting + implicit conditioning" 就像拼 mosaic——把 localized piece 拼成 coherent whole。这也解释了名字 MosaicMem 的来源。

### 2.4 Training-free validation

一个 striking 的实验: 把 Mosaic Memory 直接 plug 进 vanilla Wan 2.2 [10] 不做任何 finetune，模型仍然能 project memory condition 到 correct spatiotemporal location 并 generate meaningful content。这暗示了预训练 video DiT 已经具备某种 spatial correspondence 能力，MosaicMem 只是激活它。

---

## 3. Memory Alignment Through Warping

由于 3D VAE 的高 spatiotemporal compression (time ×4, space ×8 通常)，spatial-temporal ambiguity 严重，RoPE coordinate resolution 下降。retrieved patch 可能不 align 到 generated region 的中心，导致 local geometric inconsistency 或 blur。

提出两种 warping mechanism:

### 3.1 Warped RoPE

每个 retrieved memory patch $\mathcal{P}$ 关联 depth $D$ 和 source timestep 的 camera intrinsics/extrinsics $(\mathbf{K}_i, \mathbf{T}_i)$。给定 original RoPE 坐标 $(u, v)$，back-project 到 3D world space 再 re-project 到 target camera $(\mathbf{K}_j, \mathbf{T}_j)$:

$$
(u', v') = \Pi(\mathbf{K}_j \mathbf{T}_j \mathbf{T}_i^{-1} \mathbf{K}_i^{-1} (u, v, D)) \quad (2)
$$

公式变量:
- $(u, v)$: patch 在 source view 中的 original RoPE 坐标
- $D$: patch 对应的 depth 值
- $\mathbf{K}_i, \mathbf{T}_i$: source timestep $i$ 的 camera intrinsics / extrinsics
- $\mathbf{K}_j, \mathbf{T}_j$: target timestep $j$ 的 camera intrinsics / extrinsics
- $\mathbf{T}_i^{-1}$: source camera extrinsics 的逆，即 world-to-camera 的反向
- $\mathbf{K}_i^{-1}$: source camera intrinsics 的逆，即 pixel-to-camera 坐标变换
- $\Pi(\cdot)$: perspective projection，通过 perspective division 把 homogeneous 坐标转 image-plane 坐标
- $(u', v')$: patch re-project 到 target view 的新 RoPE 坐标

**Intuition**: 这是经典的 reprojection 公式 $\Pi(\mathbf{K}_j \mathbf{T}_j \mathbf{T}_i^{-1} \mathbf{K}_i^{-1} \mathbf{p}_i)$，其中 $\mathbf{p}_i = (u, v, D)$ 是 source view 的 3D point。本质是 multi-view geometry 中的 image-to-image warping，但作用在 RoPE 坐标上而不是 pixel 上。

tuple $(j, u', v')$ 联合定义 3D RoPE 坐标。保留 reprojected coordinate 的 fractional part，在更高 resolution 上 sample RoPE 保留精度——这其实是一种 sub-pixel RoPE。

### 3.2 Warped Latent

互补方案: 直接在 feature space 变换 retrieved memory patch，而非改 positional encoding。利用 Eq.(2) 得到的 dense correspondence $(u', v')$，对 source latent 做 differentiable bilinear grid sampling。

$$
\mathcal{P}'_{\text{warped}} = \text{GridSample}(\mathcal{P}_{\text{latent}}, (u', v'))
$$

**Intuition**: Warped RoPE 在 positional encoding 层面对齐，让 attention 自己搞 warp；Warped Latent 在 feature 层面对齐，让 attention 看到的就是 warp 后的 patch。前者保持 feature 原样但告诉模型"这个 patch 应该出现在 $(u', v')$"，后者直接把 patch 搬到 $(u', v')$ 但 RoPE 用原坐标。

### 3.3 两种 warping 的互补性

empirical 观察:
- **Warped Latent**: camera motion accuracy 更高
- **Warped RoPE**: visual quality 和 memory retrieval 更好
- **Warped RoPE 的 issue**: 在 autoregressive generation 中，previous frame 新观察到的 object 倾向于在 image boundary 附近反复 regenerate
- 训练用 mixture 两者最 robust

**Intuition**: 这是典型的 "what vs where" 分工。Warped Latent 在 "what" 层面更准（feature 已对齐），Warped RoPE 在 "where" 层面更准（位置编码告诉模型对应关系）。mixture 让模型同时拿到两种 signal。

---

## 4. PRoPE for Camera Control

### 4.1 为什么 MosaicMem 不够

三个 motivation:
1. **(a) Large motion / sparse memory**: 大 camera motion 或稀疏 memory 下，Mosaic Memory 主要作为 visual cue 而非 precise motion signal——需要 explicit trajectory specification
2. **(b) VAE temporal compression**: 3D VAE 把 time 压缩 4×，Mosaic Memory 不能捕获 fine-grained inter-frame motion
3. **(c) Latent reuse**: 加 camera control 后可以复用之前 generated 的 video latent 加速生成，camera module 校正已 encoded 的 inter-frame motion

### 4.2 PRoPE (Projective Positional Encoding)

采用 PRoPE [11] 作为 camera conditioning 接口。给定 per-frame camera projection matrix $\tilde{\mathbf{P}}_i \in \mathbb{R}^{4 \times 4}$，PRoPE 通过 projective transform $\tilde{\mathbf{P}}_{i_1} \tilde{\mathbf{P}}_{i_2}^{-1}$ 编码两个 view 的 complete relative relationship。

attention 操作为 GTA-style (Geometry-aware Transformed Attention):

$$
\text{Attn}_{\text{PRoPE}}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{D} \odot \text{Attn}(\mathbf{D}^\top \odot \mathbf{Q}, \bar{\mathbf{D}}^{-1} \odot \mathbf{K}, \bar{\mathbf{D}}^{-1} \odot \mathbf{V})
$$

每个 token $t$ 用 block-diagonal matrix:

$$
\mathbf{D}_t^{\text{PRoPE}} = \begin{bmatrix} \mathbf{D}_t^{\text{Proj}} & 0 \\ 0 & \mathbf{D}_t^{\text{RoPE}} \end{bmatrix}
$$

其中:
- $\mathbf{D}_t^{\text{Proj}} = \mathbf{I}_{d/8} \otimes \tilde{\mathbf{P}}_{i(t)}$: projective 部分
- $\mathbf{D}_t^{\text{RoPE}}$: 标准 2D patch RoPE 项
- $d$: hidden dimension
- $\mathbf{I}_{d/8}$: $d/8$ 维单位阵
- $\otimes$: Kronecker product
- $\bar{\mathbf{D}}$ 表示 $\mathbf{D}$ 的某种 normalization
- $\odot$: element-wise / block-wise multiplication

**Intuition**: PRoPE 把 camera projection matrix 直接 bake 进 attention 的 Q/K/V rotation。$\mathbf{D}_t^{\text{Proj}}$ 这一项让两个 view 的 token 互相 attend 时自动知道它们的 relative camera geometry。Block-diagonal 结构让 projective 部分 和 standard RoPE 部分独立作用。

### 4.3 Temporal compression 的处理

关键 challenge: 3D VAE 把 time 压缩 $s=4$ 倍，一个 latent frame index $\ell$ 对应 4 个 original frame $\{4\ell + k\}_{k=0}^{3}$。一个 latent slice 需要被 4 个 camera matrix $\{\tilde{\mathbf{P}}_{\ell, k}\}_{k=0}^{3}$ 条件化。

处理: "unfold" 一个 extra sub-index $k$:

$$
\mathbf{D}_{\ell, k}^{\text{Proj}} = \mathbf{I}_{d/8} \otimes \tilde{\mathbf{P}}_{\ell, k}
$$

等价地，把 camera pack 成 $\tilde{\mathbf{P}} \in \mathbb{R}^{B \times L \times 4 \times 4 \times 4}$ (Batch × Latent-frame × Sub-frame × Matrix-row × Matrix-col)，broadcast $k$-indexed transform 到 Q/K/V 的 rotation。

**Intuition**: 这是把 temporally-compressed latent frame "打开"成 4 个 sub-frame，每个 sub-frame 用自己的 camera matrix 做 PRoPE。本质是 fine-grained camera conditioning 而非 frame-level averaging。

---

## 5. Architecture 总览

基于 Wan 2.2 5B (TI2V DiT)，AdamW optimizer，learning rate $1 \times 10^{-5}$，250k steps，effective batch size 64，8×H100。Inference 用 Wan default sampler，50 denoising steps。

整体 pipeline:

```
Input image I + text L + camera trajectory C
    ↓
3D Estimator (DA V3 / VIPE) → depth + camera per frame
    ↓
Patch-level geometric lifting → 3D patch cache M
    ↓
For each target view j:
    Retrieve relevant patches P (dense or sparse)
    Warp via Eq.(2): get (u'_P, v'_P)
    ↓
Warped RoPE: position patch at (u'_P, v'_P)
Warped Latent: grid-sample patch feature
    ↓
Flatten + concat to token sequence
    ↓
PRoPE injects camera pose C into attention
    ↓
DiT denoise via Flow Matching (Eq.(1))
    ↓
Output video X
```

---

## 6. Data: MosaicMem-World benchmark

针对 first-person video dataset (Ego4D [12]、EgoVid-5M [13]、DL3DV-10K [14]) 的问题: forward navigation dominated，explicit revisitation rare，long-range return underrepresented。

构造 MosaicMem-World，四个 source 凑几十小时:
1. **Unreal Engine 5 curated scenes**: licensed asset，single/mixed action trajectory + explicit revisited segment
2. **Commercial game** (Cyberpunk 2077 [15]): dense interaction + complex dynamics
3. **Real-world first-person capture**: realistic appearance / noise / illumination
4. **Sekai dataset [16]**: 选 revisit frequency 最高的 sequence

annotation pipeline:
- Depth Anything V3 [8] 或 VIPE [15] 重建 depth + camera motion
- 32-frame segment 切分
- Gemini 3 [17] 生成两个 complementary description per segment:
  - **Static**: first frame 的 layout / salient object / spatial relation
  - **Dynamic**: 后 31 frame 的 camera motion / interaction / state change
- 这种 factorized "static + dynamic" labeling 支持 compositional training——concatenate dynamic description across segment 构造 arbitrary length clip

filter: 移除 3D estimate 不准 或 motion blur 过度的 video。

---

## 7. 实验: Quantitative Results

### 7.1 主表 (Table 1)

**Metric**:
- Camera: RotErr (°)↓, TransErr↓
- Visual quality: FID↓, FVD↓
- Consistency: SSIM↑, PSNR↑, LPIPS↓
- Dynamic↑: Average Optical Flow Magnitude

**Explicit memory baselines**:
| Method | RotErr↓ | TransErr↓ | FID↓ | FVD↓ | SSIM↑ | PSNR↑ | LPIPS↓ | Dynamic↑ |
|---|---|---|---|---|---|---|---|---|
| VMem | 1.59 | 0.14 | 77.12 | 363.34 | 0.64 | 21.64 | 0.17 | 1.18 |
| GEN3C | 1.61 | 0.13 | 77.41 | 372.08 | 0.64 | 21.58 | 0.17 | 1.21 |
| SEVA | 1.42 | 0.12 | 74.67 | 301.77 | 0.66 | 22.01 | 0.15 | 1.22 |
| VWM | 1.50 | 0.13 | 75.83 | 323.67 | 0.65 | 21.86 | 0.16 | 1.41 |

**Implicit memory baselines**:
| Method | RotErr↓ | TransErr↓ | FID↓ | FVD↓ | SSIM↑ | PSNR↑ | LPIPS↓ | Dynamic↑ |
|---|---|---|---|---|---|---|---|---|
| WorldMem | 5.87 | 0.49 | 85.72 | 403.50 | 0.47 | 15.34 | 0.46 | 1.67 |
| CaM | 4.65 | 0.43 | 85.32 | 392.11 | 0.49 | 15.78 | 0.42 | 1.72 |

**MosaicMem (full)**:
- RotErr=0.51°, TransErr=0.06, FID=65.67, FVD=232.95, SSIM=0.75, PSNR=23.57, LPIPS=0.11, Dynamic=2.58

**关键观察**:
1. Implicit memory 的 RotErr 4-5°，是 MosaicMem (0.51°) 的 ~10 倍——huge gap
2. Explicit memory 的 Dynamic score ~1.2，MosaicMem 2.58，多出 2 倍——dynamic 能力显著强
3. FVD 上 MosaicMem (232.95) vs SEVA (301.77) vs CaM (392.11)——visual quality 也是最好

**Intuition**: explicit baseline 给了 perfect pose 但 dynamic 受限，implicit baseline 给了 dynamic 但 pose drift。MosaicMem 同时改善两个 axis。

### 7.2 Ablation (Table 1 下半部)

| Ablation | RotErr↓ | TransErr↓ | FID↓ | FVD↓ | SSIM↑ | PSNR↑ | LPIPS↓ | Dynamic↑ |
|---|---|---|---|---|---|---|---|---|
| ControlMLP alone | 6.51 | 0.52 | 89.17 | 458.45 | 0.37 | 13.55 | 0.56 | 1.84 |
| PRoPE alone | 4.91 | 0.36 | 86.44 | 412.85 | 0.45 | 14.32 | 0.52 | 1.75 |
| MosaicMem w/o PRoPE | 0.79 | 0.11 | 73.18 | 250.84 | 0.68 | 22.33 | 0.14 | 2.11 |
| PRoPE + Warped Latent | 0.66 | 0.08 | 75.46 | 268.13 | 0.65 | 21.49 | 0.15 | 1.98 |
| PRoPE + Warped RoPE | 0.70 | 0.09 | 71.89 | 243.59 | 0.69 | 22.80 | 0.12 | 2.24 |
| **MosaicMem (full)** | 0.51 | 0.06 | 65.67 | 232.95 | 0.75 | 23.57 | 0.11 | 2.58 |

**关键 insight**:
1. ControlMLP → PRoPE: RotErr 6.51 → 4.91，PRoPE 明显胜出
2. PRoPE → MosaicMem w/o PRoPE: RotErr 4.91 → 0.79，加 Mosaic memory 大幅降低 pose error
3. MosaicMem w/o PRoPE → full: RotErr 0.79 → 0.51，PRoPE 补充 fine-grained intra-patch motion
4. Warped RoPE 的 FVD (243.59) < Warped Latent (268.13)，但 RotErr (0.70 vs 0.66) 略大——和 paper 中描述一致
5. Mixture (full) 在所有 metric 上 best——互补性 confirmed

### 7.3 Long-horizon Video Generation

每段 80-frame generation，更新 memory space，最后 frame 作为下一段 first frame，循环出 2-minute navigation video。CaM baseline 在长 horizon 下 artifact 累积最终 collapse，MosaicMem 保持 negligible drift + faithful reconstruction。

### 7.4 AR generation (Mosaic Forcing, Table 2)

把 bidirectional MosaicMem 蒸馏成 causal 架构——用 Causal Forcing [18] (升级版 Self Forcing [19] 和 Rolling Forcing [20])，加 Rolling Forcing strategy。Real-time 16 FPS @ 640×360。

对比 RELIC [21] 和 Matrix-Game 2.0 [22]，VBench metric:
- Total quality: 81.11 (MosaicMem) vs 79.08 (RELIC) vs 75.11 (Matrix-Game)
- RotErr: 0.89 vs 4.99 vs 5.32
- TransErr: 0.11 vs 0.36 vs 0.38

Mosaic Forcing 全面胜出。Warped RoPE alone 在 extreme scenario (极慢 camera motion) 下会在 image boundary 反复 regenerate object，引入 Warped Latent 解决。

---

## 8. 应用: Memory Manipulation

MosaicMem 存 patch + 它们的 3D spatiotemporal location，可以直接 edit 这些 location 实现:
- **Deletion**: 移除某 patch 的 3D location
- **Duplication**: 复制 location
- **Relocation**: 改 location 坐标
- **Concatenation**: 把两个 scene 的 memory 拼接——水平拼 (medieval → modern street → medieval) 或垂直拼 (Inception-like sky scene)

这种 manipulability 是 implicit memory 完全做不到的——因为 implicit 把 memory 存在 latent 里没法直接 edit。explicit memory 理论上能 edit 但因为 global structure 耦合，独立 patch 操作困难。

---

## 9. 核心贡献与 Intuition 总结

### 9.1 三大贡献
1. **Patch 作为 fundamental memory unit**: 介于 point/splat (explicit) 和 frame (implicit) 之间的新粒度
2. **Hybrid retrieval**: explicit-style lifting 提供 geometry grounding，implicit-style attention conditioning 保留 dynamic flexibility
3. **PRoPE + 双 warping**: 解决 fine-grained camera control 和 patch alignment 两个 axis

### 9.2 Build intuition

可以这样理解 MosaicMem:

- **Frame memory (implicit)** 像是给模型一本相册，让模型自己翻找对应——flexible 但翻找不精确，且相册占用大量 context
- **3D structure (explicit)** 像是给模型一个 3D scene model——精确但只能 render static view
- **Mosaic patch memory** 像是给模型一堆 "有 3D 坐标标签的 polaroid 照片"——既知道每张照片该贴到哪 (精确)，又能选择性参考 (flexible)，还能撕掉/复制/移动某张 (manipulable)

关键技术 insight:
- **Patch 是 video 的 natural unit**: video 高度 redundant，spatial 上的局部 patch 足以 reconstruct，temporal 上重复信息多
- **RoPE 的 sub-pixel resolution**: 通过保留 fractional coordinate + 高分辨率 sample RoPE，绕开 VAE compression 的精度损失
- **Projective encoding 进入 attention**: PRoPE 直接把 camera geometry bake 进 attention matrix，避免 ControlNet-style 外部 branch 的信息 bottleneck
- **Warping 互补**: positional-level warp (RoPE) + feature-level warp (Latent) 形成 "what" + "where" 双重对齐

### 9.3 可能的 limitation / 思考点

1. **3D estimator 依赖**: geometric lifting 依赖 Depth Anything V3 / VIPE，dynamic object 的 depth 估计可能不准——paper 没深入讨论 dynamic object 的 lifting quality
2. **Patch granularity**: paper 没说 patch 大小如何选，patch 太小 retrieval cost 高，太大 alignment 不精确
3. **Memory size 增长**: 长期 navigation 后 memory 会很大，虽然有 sparse retrieval 但没讨论 memory pruning / forgetting
4. **Long-tail scene**: training data 多为 game + UE5 + first-person，real-world long-tail scene (如野外) 表现未验证
5. **Causal forcing 蒸馏**: AR version 从 bidirectional 蒸馏而来，蒸馏过程中 consistency 是否完全保留？paper 给的对比是 vs 其他 AR system，没 vs 自己的 bidirectional 版本

---

## 10. 与相关工作的 position

- **GEN3C** (CVPR 2025) [1]: point cloud cache + video inpainting——pure explicit，MosaicMem 继承其 lifting 思路但用 patch 替代 point cloud
- **Context-as-Memory** (SIGGRAPH Asia 2025) [5]: rule-based selection + token concat——pure implicit，MosaicMem 保留其 conditioning 但加 geometric warp
- **WorldMem** (NeurIPS 2025) [6]: FOV overlap retrieval——implicit 但用 FOV 做 retrieval cue
- **WorldPack** [7]: trajectory packing + hierarchical compression——压缩 implicit memory 的尝试
- **PRoPE** (NeurIPS 2025) [11]: Cameras as relative positional encoding，MosaicMem 直接采用
- **PE-Field** [23]: 2D positional encoding 扩展到 3D field for NVS——和 Warped RoPE 思想相近但作用域不同
- **Genie 3** [24]: Google DeepMind 的 interactive world model，real-time + persistence，MosaicMem 是同一方向的不同技术路径
- **Causal Forcing** [18] / **Self Forcing** [19] / **Rolling Forcing** [20]: AR video diffusion 的 train-test gap 处理，MosaicMem 用这些方法做 AR 蒸馏

---

## 11. 代码与资源链接

- **Project page**: paper 提到有 additional visual results 的 project page，具体 URL 需要 author 发布
- **Wan 2.2**: https://github.com/Wan-Video/Wan2.2 (open-source 5B TI2V DiT)
- **Depth Anything V3**: https://github.com/DepthAnything/Depth-Anything-V3
- **VIPE**: https://arxiv.org/abs/2508.10934
- **PRoPE (Cameras as Relative Positional Encoding)**: NeurIPS 2025, https://arxiv.org/abs/2506.2ica (Li et al.)
- **Causal Forcing**: https://arxiv.org/abs/2602.02214
- **Self Forcing**: https://arxiv.org/abs/2506.08009
- **Rolling Forcing**: https://arxiv.org/abs/2509.25161
- **GEN3C**: https://research.nvidia.com/labs/toronto-ai/GEN3C/ (CVPR 2025)
- **VMem**: https://arxiv.org/abs/2505.01776 (ICCV 2025)
- **SEVA (Stable Virtual Camera)**: https://arxiv.org/abs/2506.15397 (ICCV 2025)
- **VWM**: https://arxiv.org/abs/2506.05284
- **Context-as-Memory**: SIGGRAPH Asia 2025
- **WorldMem**: https://xingtongxx.github.io/worldmem/ (NeurIPS 2025)
- **Sekai**: https://arxiv.org/abs/2506.15675
- **Genie 3 blog**: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
- **RELIC**: https://arxiv.org/abs/2512.04040
- **Matrix-Game 2.0**: https://arxiv.org/abs/2508.13009
- **Ego4D**: https://ego4d-data.org/
- **Cyberpunk 2077**: https://www.cyberpunk.net
- **VBench**: https://vchitect.github.io/VBench-project/

---

## 12. 总结

MosaicMem 的核心 contribution 在于找到了 **patch 这个 "missing middle" 作为 spatial memory unit**——既不像 explicit memory 那样 lock 到 global 3D structure 而失去 dynamic flexibility，也不像 implicit memory 那样 stuck 在 redundant frame representation 而失去 geometric precision。配合 PRoPE 的 projective encoding 和 Warped RoPE / Warped Latent 的双重 alignment，MosaicMem 在 long-horizon camera-controlled video world model 的道路上提供了可操作的 path forward。

更深层看，这篇 paper 暗示了 video diffusion model 内部已经隐含了 spatial correspondence 能力 (training-free validation 这个实验是关键证据)，问题在于如何 unlock 这种能力——MosaicMem 通过 geometric warp + memory patch + projective attention 提供 explicit cue，让 pre-trained model 的 implicit capability 显式化。这种 "implicit capability + explicit interface" 的思路，可能比 pure end-to-end learning 更 sample-efficient 且 interpretable。

对 world model 方向的更长 term implication: 真正的 persistent world simulator 需要某种形式的 addressable external memory——pure attention 体系受限于 context window 和 redundancy，pure 3D reconstruction 受限于 dynamic modeling。Patch-level addressable memory 可能是介于两者之间的 sweet spot，未来可能扩展到 multi-modal memory (audio patch、physics state patch、semantic tag patch) 形成 richer world state representation。
