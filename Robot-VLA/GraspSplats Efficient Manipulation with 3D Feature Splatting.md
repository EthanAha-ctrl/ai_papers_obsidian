---
source_pdf: GraspSplats Efficient Manipulation with 3D Feature Splatting.pdf
paper_sha256: d62956de29c09841b8d6166768ae9a4cb83f2ad6b916891332f5dfe78427d119
processed_at: '2026-08-19T09:57:37-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 GraspSplats

## 一句话概括

机器人要抓东西，得先"看懂"三维世界。以前的方法要么太慢（NeRF要train几分钟），要么太糙（直接把2D feature拍扁到3D point cloud）。GraspSplats 用 **3D Gaussian Splatting** 当中间桥梁——又快又准，还能实时跟着物体动。

---

## 为什么这事难

想象你在厨房，要抓一个 mug 的 handle。你的大脑做了几件事：

1. 认出"这是个 mug"（object-level semantics）
2. 认出"要抓 handle 那个细长的部分"（part-level semantics）
3. 知道 handle 在 3D 空间哪个位置、朝哪个方向（geometry）
4. 规划手指怎么伸过去（grasp sampling）

机器人卡在哪儿？CLIP 这种 2D foundation model 很会做 1、2，但它在 2D 图像上做。机器人活在 3D 世界里。这个 **2D-to-3D gap** 就是核心痛点。

---

## 以前的解法都有毛病

### NeRF 路线（F3RM、LERF-TOGO）

NeRF 就是把整个 scene 塞进一个 MLP。要 query 任何一个点，得 forward 一次 network。train 一次要 10 分钟。

更致命的：scene 一变就得 retrain。你把 mug 挪一下位置，整个 NeRF 就废了，得重来。这跟真实机器人场景完全脱节——真实世界里物体一直在动。

### Point cloud 路线（VoxPoser、ConceptGraphs）

快，直接把 2D CLIP feature 拍到 3D point cloud 上。但没经过 rendering-based optimization，feature 是"硬贴上去"的，不 consistent。part-level segmentation 特别糟——你问"mug handle"，它经常把整个 mug 都圈进去。

---

## GraspSplats 的核心 insight

3D Gaussian Splatting（3DGS）刚好卡在 NeRF 和 point cloud 中间，继承了两者优点：

- 像 point cloud 一样 **explicit**：每个 Gaussian 是独立的小椭球，你想 move 它、rotate 它，直接改坐标就行，不需要 retrain
- 像 NeRF 一样 **可微分渲染**：可以 backprop 把 2D CLIP feature distill 进 3D Gaussian，得到 consistent 的 semantic field

用一句话：**Gaussian 是一群可以单独操作的小 NeRF**。

这就是为什么 GraspSplats 能做 dynamic scene——物体动了，CoTracker 跟踪 2D point，depth lift 到 3D，Kabsch 算出 rigid transform，直接乘到对应 Gaussian 的 position/rotation 上。毫秒级。NeRF 做不了这事。

---

## 三个关键工程 trick

### Trick 1: Hierarchical reference feature

CLIP 有个臭名昭著的毛病——bag-of-words behavior。你 query "mug handle"，它会同时激活所有 mug 和所有 handle，分不清你想抓的到底是哪个 mug 的 handle。

LERF-TOGO 的解法是 multi-scale query + DINO regularization，很慢。

GraspSplats 的思路特别 human：**先认 object，再认 part**。用 MobileSAMV2 先 detect 出 mug 的 bounding box，在 box 里面 crop patch 再跑 MaskCLIP 算 part feature。这样 "handle" feature 只在 mug 的 context 里有意义。

两个 branch 的 MLP 同时 render object-level feature 和 part-level feature，loss 是 $\mathcal{L}_{obj} + 2.0 \cdot \mathcal{L}_{part}$。part-level weight 加倍，因为更难、更重要。

实验上 part-level IoU 从 39.0（LERF）干到 50.7（GraspSplats）。

### Trick 2: Depth 直接 init，跳过 SfM

原版 3DGS 用 Colmap 从 RGB 算 sparse point cloud 做 Gaussian initialization。这步本身要 11.6s，dense Colmap 要 623s。

GraspSplats 直接用 RGBD 的 depth map back-project 出 3D point 当 Gaussian center。这步 0.7s。

同时用 depth 做 supervision：$\mathcal{L}_{depth} = ||\hat{\mathbf{D}} - \mathbf{D}_{gt}||_2^2$。

这个 free lunch 让 train iteration 从 10,000 降到 3,000。整个 reconstruction 从 10min 降到 60s。

**Intuition**：RGBD camera 本来就给你 depth 了，干嘛还要从 RGB re-derive？很多 GS 方向的 paper 忽略了 RGBD 这个 prior，GraspSplats 把它捡起来用足。

### Trick 3: 直接在 Gaussian 上做 grasp sampling

以前做 grasp sampling 要么用 end-to-end GraspNet（受限于训练分布，泛化差），要么先 voxel 化再跑 GPG（慢）。

GraspSplats 直接用 Gaussian primitive 当 input：每个 Gaussian 的 scale/rotation 已经 encode 了 local surface geometry，rendered normal 就可以直接算。公式 2 那个 $M(p) = \sum \hat{n}(g)\hat{n}(g)^T$ 本质是 local normal 的 structure tensor，SVD 一下就得到 reference frame。

然后在这个 reference frame 里做 2D grid search（translation $y$ + rotation $\phi$），gripper 沿 normal 方向推进直到碰撞，碰撞检查通过就加 candidate，最后 geometry-aware scoring 选最佳。

整个 grasp sampling 0.5s，比 LERF-TOGO 的 4.8s 快一个量级，比 GraspNet-100（LERF-TOGO 的实际做法）的 10.3s 快 20×。

**关键 insight**：semantic affordance 由 CLIP feature 提供，geometry 由 Gaussian 提供，grasp 由 classical GPG 提供。三者解耦，各用各的 SOTA。不需要把 semantics 塞进 end-to-end grasp network。

---

## Dynamic scene 怎么 work

这是最 cool 的部分。流程：

1. Language query "mug" → segment 出 mug 的 3D Gaussians
2. Rasterize 到 camera view → 得到 2D mask
3. Discretize mask 成一堆 2D keypoint
4. 喂给 CoTracker，持续跟踪这些 2D point 的坐标
5. 用 depth 把 2D track 换成 3D correspondence
6. DBSCAN 过滤 noisy correspondence
7. Kabsach algorithm 算出 SE(3) rigid transform
8. 直接 apply 到 mug 的 Gaussian position/rotation 上

整个过程 millisecond 级。物体被人手挪了、被其他机器推了，都能 track。

NeRF 完全做不了这事——它的 representation 是 implicit 的，没法局部 edit。LERF-TOGO 和 F3RM 在 dynamic scene 上直接放弃（Table 1 里那两个 † 符号就是 "doesn't cope with dynamic scenes"）。

---

## 实验上到底赢多少

主表 Table 1：

- Training：60s vs LERF-TOGO 10min（10× faster）
- Grasping：1.3s vs 9.9s（7.6× faster）
- Static success：81.4% vs LERF-TOGO 65.1%（+16.3%）、vs F3RM 72.1%（+9.3%）
- Dynamic：74.2%，所有 baseline 都做不了

Latency breakdown（Table 2）：
- Segment：0.8s vs LERF-TOGO 5.1s
- Grasp sampling：0.5s vs LERF-TOGO 4.8s

Object vs Part-level breakdown（Table 6）：
- Object-level：96.3%（几乎完美）
- Part-level：85.2% vs LERF-TOGO 63.0%（+22.2%）

---

## 哪些地方还不行

1. **Rigid transformation 假设**：Kabsch 只能算 rigid transform。deformable object（dough、rope、cloth）做不了。但 Gaussian representation conceptually 是支持的，每个 Gaussian 可以独立 deform，未来可能结合 physics simulation 或 non-rigid registration（比如 Gaussian Mixture Registration 或 CPD）

2. **Tracking 对 texture 敏感**：单色物体、对称物体，CoTracker 没 feature 可 track，会 drift。fast rotation + occlusion 会直接 lose correspondence。Future work 可以考虑 re-sample keypoint during task，或者用 geometric feature（edge、corner）补强

3. **Placement 仍困难**：gripper 抓住 object 后会 occlude view，不知道 object 在 hand 里的 orientation，placement 困难。需要 in-hand re-observation 或者 tactile feedback

---

## 我（Karpathy 视角）怎么看这篇 paper

几个值得注意的点：

**1. Representation 决定上限**

这篇 paper 核心贡献就一件事：选对了 representation。3DGS 作为 explicit + differentiable 的 hybrid，在 manipulation 这种需要 edit + query 的场景下，碾压 NeRF 的纯 implicit。这呼应了一个 broader pattern：纯 end-to-end learned pipeline 在 controllability 和 sample efficiency 上不如 hybrid（learned prior + explicit structure）。

类比语言模型：pure attention 什么都学，但加 explicit structure（Mamba 的 selective state space、MoE 的 routing）能在特定维度上更强。Robotics manipulation 也是——learned semantics（CLIP）+ explicit geometry（Gaussian）+ classical grasp（GPG），三者解耦，每个 component 都用各自的 SOTA。

**2. Classical method 复活**

GPG 是 2016-2017 的工作，单独用的时候效果一般。但配上 CLIP feature 做的 semantic segmentation，居然 beat GraspNet（2020）的 end-to-end pipeline。

这说明：**semantic 和 geometry 解耦后，各自不需要 end-to-end**。GraspNet 想学 semantic prior 又想学 geometry，结果两边都不够强，而且受限于训练 distribution。GraspSplats 让 CLIP 管 semantic、Gaussian 管 geometry、GPG 管 grasp，三个 expert 各管一摊，反而更好。

这让人想到 autonomous driving 里 Waymo 的 approach（modular）vs Tesla 的 end-to-end。Modular 在 data 少的时候更 controllable、debuggable，end-to-end 在 data 多的时候 ceiling 更高。Robotics manipulation 现在 data 还不够多，modular 反而 work。

**3. Depth supervision 是 free lunch**

这个 trick 太朴素了，但有效。RGBD camera 给的 depth 直接用，跳过 SfM，直接 supervise。10min → 60s 的 speedup 一半来自这里。

很多 GS 论文 fall in love with RGB-only setting，因为更 general。但 robotics 场景下 RGBD camera 是标配，不用白不用。这种 "match representation to actual sensor modality" 的工程品味很重要。

**4. Hierarchical feature aligns with human cognition**

人识别物体也是 compositional 的：先认 mug 再认 handle，不会一步到位 query "mug handle"。CLIP 的 bag-of-words problem 本质是它 lacks compositionality。GraspSplats 的 hierarchical design 把 compositionality 显式编码进去，绕过了 CLIP 的这个 limitation。

这个思路在 VLM 里也有人做——比如 hierarchical text encoding、region-conditioned CLIP query。 robotics 这里又重现了。**Compositionality 是 AI 里反复出现的 theme，谁解决得好谁就赢**。

---

## Future work 我觉得有意思的方向

1. **Deformable object manipulation**：把 Kabsch 换成 non-rigid registration（CPD、GMMReg、甚至 Gaussian Process latent variable model）。Gaussian representation 天然支持 non-rigid，每个 Gaussian 独立 deform。结合 physics simulation 可以做衣服、绳索、软体食物的 manipulation。

2. **Tactile-visual fusion**：Gaussian 的 geometry 可以 predict contact point，gripper 接触后用 tactile sensor（DIGIT、GelSight）做 closed-loop adjustment。这能解决 placement 问题——gripper 视觉被遮挡后，用 tactile 推断 in-hand object orientation。

3. **LLM as task planner + GraspSplats as executor**：LLM 看 rendered scene + language feature，输出 task plan（"先开 cabinet，再抓 pineapple"），GraspSplats 执行每一步。UMI-style data collection + GraspSplats representation + LLM planning = zero-shot generalist manipulator 的 promising path。

4. **Active perception**：GraspSplats 可以 render future view（不同 camera pose），用 information gain 选 best next view。机器人主动 scan 来补全 occluded region。这比固定 Bezier curve scan 更 sample-efficient。

5. **4D Gaussian Splatting for manipulation**：4DGS [48] 已经能 model dynamic scene。结合 GraspSplats 的 feature embedding，可以直接 track deformable object 的 temporal evolution，不需要 separate CoTracker pipeline。

References:
- 3DGS original: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Feature Splatting (GraspSplats 的 base): https://arxiv.org/abs/2404.01223
- CoTracker: https://arxiv.org/abs/2307.07635
- LERF-TOGO (主要 baseline): https://lerf-togo.github.io/
- F3RM (另一个 baseline): https://f3rm.github.io/
- GPG (grasp sampling): https://arxiv.org/abs/1703.00239
- GraspNet-1Billion: http://graspnet.net/
- MobileSAMV2: https://arxiv.org/abs/2312.09579
- Kabsch algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm
- Project page: https://graspsplats.github.io

---

## TL;DR

GraspSplats = 3DGS（explicit + differentiable）+ hierarchical CLIP feature（object + part）+ depth supervision（跳过 SfM）+ GPG grasp sampling（classical geometry）+ CoTracker + Kabsch（real-time rigid tracking）。

每个 component 都不新，但组合起来：10× faster training、7× faster grasping、+16% static success、唯一能做 dynamic scene。这就是 engineering taste——选对 representation、解耦 component、match sensor modality。论文的 contribution 不在单点突破，在 system-level 的 coherent design。

---

# GraspSplats: Efficient Manipulation with 3D Feature Splatting 深度解析

## 1. Core Motivation: 为什么需要这篇 paper

机器人 manipulation 面临一个 fundamental gap: 2D foundation models (CLIP, SAM) 提供 rich semantics, 但 robot 需要 3D representation 来做 grasp sampling。现有方法分两类:

**NeRF-based methods** (F3RM [1], LERF-TOGO [2]): 用 differentiable rendering 把 CLIP features distill 进 NeRF。问题在于 NeRF 是 implicit representation — 整个 scene encode 在一个 MLP 里, scene 一变就要 retrain (minutes级别)。

**Point-based methods** (VoxPoser [5], ConceptGraphs [24]): 直接 back-project 2D features 到 3D point cloud。快, 但没有 rendering-based optimization, part-level localization 不准, occlusion 处理差。

GraspSplats 的核心 insight: **3D Gaussian Splatting (3DGS) [12] 天然是 explicit representation**, 每个 Gaussian 是独立的 ellipsoid, 可以直接 transform; 同时支持 differentiable rasterization 做 feature distillation。这把 NeRF 的 quality 和 point cloud 的 editability 结合起来了。

Project page: https://graspsplats.github.io

---

## 2. Method Architecture 详解

### 2.1 Background: Gaussian Splatting Rendering Equation

公式 (1) 是整个方法的 rendering backbone:

$$\{ \hat{\mathbf{D}}, \hat{\mathbf{F}}, \hat{\mathbf{C}} \} = \sum_{i \in N} \{ \mathbf{d}_i, \mathbf{f}_i, \mathbf{c}_i \} \cdot \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

**变量含义**:
- $\hat{\mathbf{D}}$: rendered depth map
- $\hat{\mathbf{F}}$: rendered dense visual feature map (CLIP/DINO features)
- $\hat{\mathbf{C}}$: rendered color image
- $\mathbf{d}_i$: 第 $i$ 个 Gaussian 到 camera origin 的距离
- $\mathbf{f}_i$: 第 $i$ 个 Gaussian 的 latent feature vector (假设 isotropic, 即 view-independent)
- $\mathbf{c}_i$: 第 $i$ 个 Gaussian 的 color
- $\alpha_i$: 第 $i$ 个 Gaussian 的 opacity
- $i \in N$: Gaussian indices, 按 $\mathbf{d}_i$ **升序排列** (front-to-back sorting, 这对 alpha blending 正确性关键)
- $\prod_{j=1}^{i-1}(1-\alpha_j)$: **transmittance**, 前面所有 Gaussian 的累积透过率, 表示光线到达第 $i$ 个 Gaussian 时剩余的能量比例

**Intuition**: 这本质上是 **alpha compositing** — 沿着 ray 从近到远累加, 每个 Gaussian 贡献 $\{\mathbf{d}_i, \mathbf{f}_i, \mathbf{c}_i\}$ 乘以它的 opacity $\alpha_i$ 再乘以前面没被挡住的比例。和 NeRF 的 volume rendering 数学上等价, 但 3DGS 不用 MLP query, 而是直接 rasterize explicit Gaussians, 所以快 10x+。

Reference: 3DGS original paper https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

### 2.2 Efficient Hierarchical Reference Feature Computation

这是 GraspSplats 的第一个 key contribution。现有 feature-enhanced GS methods (Feature Splatting [38], LangSplat [39]) 的 bottleneck 在于 **reference feature computation** — 在 optimization 开始前要花大量时间算 2D CLIP features。

#### Pipeline

**Step 1: Object detection via MobileSAMV2 [14]**

给定 input image, MobileSAMV2 预测:
- Class-agnostic bounding boxes $\mathbf{D}_{obj} := \{(x_i, y_i, w_i, h_i)\}_{i=1}^N$
  - $(x_i, y_i)$: box center
  - $w_i, h_i$: box width, height
  - $N$: object 数量
- Object masks $\{M\}$

**Intuition**: MobileSAMV2 在 SA-1B [25] 上训练, 有 strong object prior, 所以 $N$ 比 LERF [6] 的 uniform multi-scale queries 少很多 — 这是 speedup 的关键。

**Step 2: Object-level feature via Masked Average Pooling (MAP)**

公式 (3):

$$w_i = \mathbf{MAP}(\mathbf{M}, \mathbf{F}_C) = \frac{\sum_{i \in \mathbf{F}_C} \mathbf{M}(i) \cdot \frac{\mathbf{F}_C(i)}{||\mathbf{F}_C(i)||}}{\sum_{i \in \mathbf{F}_C} \mathbf{M}(i)}$$

**变量**:
- $\mathbf{M}$: object mask (binary)
- $\mathbf{F}_C \in \mathbb{R}^{H' \times W' \times C}$: MaskCLIP [15] 计算的 coarse CLIP feature map
  - $H', W'$: downsampled spatial dims
  - $C=768$: CLIP feature dim
- $\mathbf{M}(i)$: mask 在 pixel $i$ 处的值 (0 or 1)
- $\mathbf{F}_C(i)$: pixel $i$ 处的 CLIP feature
- $||\mathbf{F}_C(i)||$: L2 norm, 用于 normalize feature
- $w_i$: aggregated object-level feature vector

**Intuition**: MAP 就是把 mask 内所有 pixel 的 normalized CLIP features 平均 — 得到一个 object-level 的 semantic embedding。先 normalize 再 average 防止 high-norm feature 主导。

**Step 3: Part-level feature via patch inference**

从 $\mathbf{D}_{obj}$ crop image patches, resize 到 $(224, 224)$, 用 MaskCLIP batched inference 得到 $(28, 28, 768)$ feature map, interpolate 回原 bounding box size, paste 到 part-level feature map $\mathbf{F}_{part}$。重叠 pixel 做 average。

**Step 4: Two-branch shallow MLP**

在 differentiable rasterization 时, rendered feature $\hat{\mathbf{F}}$ 经过一个 shallow MLP:

$$\hat{\mathbf{F}}_{obj}, \hat{\mathbf{F}}_{part} := \mathbf{MLP}(\hat{\mathbf{F}})$$

- $\hat{\mathbf{F}}_{obj}$: object-level rendered feature, supervised by $\mathbf{F}_{obj}$ with cosine loss $\mathcal{L}_{obj}$
- $\hat{\mathbf{F}}_{part}$: part-level rendered feature, supervised by $\mathbf{F}_{part}$ with cosine loss $\mathcal{L}_{part}$
- Joint loss: $\mathcal{L}_{obj} + \lambda \cdot \mathcal{L}_{part}$, $\lambda = 2.0$

**Intuition**: $\lambda = 2.0$ 强调 part-level supervision, 因为 part-level grasping 更难、更重要。Two-branch design 让 object 和 part 的 semantics 分开学, 避免 feature 互相干扰。

---

### 2.3 Geometry Regularization via Depth

原版 3DGS [12] 用 SfM (Colmap [45]) 从 RGB 算 sparse point cloud 做 Gaussian initialization — 这步很慢 (Table 5: Colmap-S 要 11.6s, Colmap-D 要 623s)。

GraspSplats 直接用 RGBD 的 depth map project 出 3D points 作为 Gaussian centers, 同时用 depth 做 supervision:

$$\mathcal{L}_{depth} = ||\hat{\mathbf{D}} - \mathbf{D}_{gt}||_2^2$$

- $\hat{\mathbf{D}}$: rendered depth (Eq. 1)
- $\mathbf{D}_{gt}$: ground truth depth from RGBD sensor

**Intuition**: Depth supervision 做了两件事: (1) 跳过 SfM, initialization 直接 dense; (2) 约束 geometry, 让 Gaussian 的 position/scale/rotation 收敛到真实 surface, 这对 grasp sampling 的 normal estimation 至关重要。

Table 5 的 ablation 印证: GraspSplats initialization 只需 0.7s (vs Colmap-S 11.6s), train iteration 只需 3,000 (vs Colmap-S 10,000)。

---

### 2.4 Static Scene: Part-level Querying 和 Grasp Sampling

#### Open-vocabulary Object Querying (Appendix B 详解)

给定 language set $L = \{L_0^-, L_1^-, ..., L_n^-, L^+\}$:
- $L^+$: positive query (e.g., 'mug')
- $L_i^-$: negative queries (default: 'objects', 'things'; 可扩展)

CLIP encode 每个 word 到 768-dim:
$$\mathbf{F}_{text,i} = \text{CLIP.encode}(L_i), \quad i = 0, 1, ..., n$$

每个 Gaussian $j$ 有 16-dim latent feature $\mathbf{F}_{latent,j} \in \mathbb{R}^{16}$, decoder 到 768-dim:
$$\mathbf{F}_{CLIP,j} = \text{Decoder}(\mathbf{F}_{latent,j})$$

**Intuition**: 16-dim latent 是 bottleneck — 压缩 768→16→768, 起到 regularization 作用, 防止 overfit 到 noisy CLIP features, 同时节省 memory (768-dim per Gaussian 太贵)。

Cosine similarity:
$$\sin(\mathbf{F}_{CLIP,j}, \mathbf{F}_{text,i}) = \frac{\mathbf{F}_{CLIP,j} \cdot \mathbf{F}_{text,i}}{||\mathbf{F}_{CLIP,j}|| \cdot ||\mathbf{F}_{text,i}||}$$

Softmax over all queries:
$$\mathbf{S}_j = \text{softmax}(\{\sin(\mathbf{F}_{CLIP,j}, \mathbf{F}_{text,i})\}_{i=0}^n)$$

取 positive query 的 similarity: $\text{sim}_{positive,j} = \mathbf{S}_j[n]$

Threshold $\tau = 0.6$ 选 Gaussians, 再用 DBSCAN [43] 过滤 outliers。

**Intuition**: Softmax over positive + negatives 是 contrastive query — negatives 把无关的 semantics push down。这比单纯 cosine similarity with positive 更 robust。

#### Conditional Part-level Querying

CLIP 有 bag-of-words behavior: 'mug handle' 会同时激活所有 mug 和所有 handle。LERF-TOGO [2] 用 multi-scale queries + DINO regularization 解决, 很慢。

GraspSplats 的做法: 先 object-level query segment 出 mug 的 Gaussians, 再在这个 subset 上用 part-level feature 做 'handle' query — **conditional query natively supported by explicit primitives**。不需要 re-render。

#### Grasp Sampling (公式 2 + Appendix D)

**Step 1: 定义 workspace $\mathcal{R}_{obj}$**

从 segmented part 的 Gaussians 扩展 3D 空间:
$$\text{expansion radius} = \max(\text{Gaussian scales}) + \text{gripper collision radius}$$

**Step 2: 采样 $N$ 个 reference points**

从 $\mathcal{R}_{obj}$ 采样 $N$ 个 points, 每个 point $p$ 在 neighborhood $R_p$ 内聚合 Gaussian normals:

公式 (2):
$$M(p) = \sum_{g \in R_p} \hat{n}(g) \hat{n}(g)^T$$

**变量**:
- $M(p)$: 3×3 matrix at point $p$ (sum of outer products)
- $g$: gaussian primitive in neighborhood $R_p$
- $\hat{n}(g)$: unit surface normal of gaussian $g$ (3D vector)
- $\hat{n}(g)^T$: 它的 transpose (row vector)
- $\hat{n}(g)\hat{n}(g)^T$: 3×3 outer product matrix

**Intuition**: $M(p)$ 本质是 local surface normal 的 **second-moment matrix** (structure tensor 的变体)。做 SVD 后:
- 最大 eigenvalue 对应的 eigenvector $v_3(p)$: **normal direction** (surface 法线方向, 因为 normals 在这个方向上 variance 最大)
- 次大 $v_2(p)$: secondary direction
- 最小 $v_1(p)$: minimum direction (surface tangent 方向, normals 在这个方向上 variance 最小)

Reference frame: $F(p) = [v_3(p) \mid v_2(p) \mid v_1(p)]$

**Step 3: Grid search for grasps**

2D grid $G = Y \times \Phi$:
- $Y$: translation along $v_2(p)$
- $\Phi$: rotation about $v_3(p)$ (normal axis)

对每个 $(y, \phi) \in G$:
- 构造 homogeneous transform $T_{x,y,\phi}$: translation in x,y plane + rotation about z-axis
- 在 $F(p)$ frame 下, gripper 沿 negative x-axis 推进直到接触 point cloud
- $x^*$: minimum contact distance
- Gripper pose: $F(h_{y,\phi}) = F(p) T_{x^*, y, \phi}$

**Step 4: 碰撞检查**

如果 gripper closed region 内的 segmented object points $N_{obj} > N_{th}$, 则 grasp $h_{y,\phi}$ 加入 candidate set $H$。

**Step 5: Scoring**

用 geometry-aware scoring model [17, 49] rank, 选 highest score。

**Intuition**: 这是 GPG (Geometry-based Point Cloud Grasping) [16, 17] 的变种。关键区别: GraspSplats 直接用 Gaussian primitives 的 rendered normals, 不需要额外 point cloud normal estimation。Gaussian 的 scale/rotation 已经 encode 了 local surface geometry。

---

### 2.5 Dynamic Scene: Real-time Tracking 和 Editing

这是 GraspSplats 相比 NeRF methods 的 **killer feature**。

#### Multi-view Object Tracking

**Step 1: Segment via language**

Query language (e.g., 'mug') → segment 对应的 3D Gaussians。

**Step 2: Rasterize 2D mask**

把 segmented 3D Gaussians rasterize 到每个 calibrated camera, 得到 2D mask。

**Step 3: Discretize mask → keypoints**

把 rendered mask discretize 成一组 2D points, 作为 CoTracker [18] 的 input。

**Step 4: CoTracker 跟踪**

CoTracker [18] 持续跟踪这批 2D points 的 coordinates。Reference: https://arxiv.org/abs/2307.07635

**Step 5: 2D → 3D**

用 depth 把 2D correspondences 转成 3D points。

**Step 6: DBSCAN outlier filtering**

用 DBSCAN [43] 过滤 noisy 3D correspondences (depth noise, tracking drift)。

**Step 7: Kabsch algorithm [44]**

用 Kabsch algorithm 解 SE(3) transformation (rotation + translation) 最优拟合 correspondence pairs:

给定 source points $\{p_i\}$ 和 target points $\{q_i\}$:
1. Center: $\bar{p} = \frac{1}{n}\sum p_i$, $\bar{q} = \frac{1}{n}\sum q_i$
2. Cross-covariance: $H = \sum (p_i - \bar{p})(q_i - \bar{q})^T$
3. SVD: $H = U\Sigma V^T$
4. Rotation: $R = V U^T$ (correct for reflection if $\det(R) < 0$)
5. Translation: $t = \bar{q} - R\bar{p}$
6. SE(3): $T = [R \mid t]$

**Multi-camera**: 把所有 camera 的 3D correspondences append 到 Kabsch 的方程系统。

**Step 8: Apply transform to Gaussians**

直接把 $T$ 乘到 segmented Gaussians 的 position 和 rotation 上 — explicit representation 的优势。

**Intuition**: 这套 pipeline 本质是 **2D tracking + 3D back-projection + rigid registration**。CoTracker 提供 2D temporal correspondence, depth 提供 3D lift, Kabsch 求 rigid transform。整个过程 millisecond 级, 而 NeRF 要 retrain minutes。

#### Partial Fine-Tuning

物体被移走后, 原来被遮挡的 surface (e.g., 桌面) 没被 reconstruct, 会有 artifacts。GraspSplats 支持用 displacement 前后的 object mask 做 partial re-training — 只 optimize 受影响 region, 比完整 reconstruction 快。

---

## 3. Experiments 详解

### 3.1 Main Results (Table 1)

| Method | Training Latency | Grasping Latency | Static Succ. | Dynamic Succ. |
|---|---|---|---|---|
| Tracking Anything [46] | — | 3.1s | 41.9% | 45% |
| ConceptGraphs [24] | ~30s | 0.7s | 51.1% | — |
| LERF-TOGO [2] | ~10min | 9.9s | 65.1% | — |
| F3RM* [1] | ~3min | 1.6s | 72.1% | — |
| **GraspSplats** | **60s** | **1.3s** | **81.4%** | **74.2%** |

**Key takeaways**:
- Training: GraspSplats 60s vs LERF-TOGO 10min (10× faster), vs F3RM 3min (3× faster)
- Grasping: 1.3s vs LERF-TOGO 9.9s (7.6× faster)
- Static success: +16.3% vs LERF-TOGO, +9.3% vs F3RM
- Dynamic: GraspSplats 是唯一能做 dynamic 的 (74.2%), NeRF methods 根本不支持 (†: requires offline batch processing)

### 3.2 Latency Breakdown (Table 2)

| Method | Segment | Grasp Sampling |
|---|---|---|
| Tracking Anything* | 2.5±0.1s | 0.6±0.05s |
| ConceptGraph* | 0.1±0.05s | 0.6±0.05s |
| LERF-TOGO | 5.1±0.3s | 4.8±0.7s |
| F3RM | 1.0±0.1s | 6.9±0.45s |
| **GraspSplats** | **0.8±0.1s** | **0.5±0.06s** |

**Intuition**: 
- LERF-TOGO 慢是因为要从 implicit NeRF render CLIP features (5.1s segment) + 用 GraspNet 100 次 inference (4.8s grasp)
- F3RM 慢在 grasp: 6.9s, 也是 GraspNet 的 multi-view inference
- GraspSplats 直接 query explicit Gaussians (0.8s) + sampling-based GPG (0.5s)

### 3.3 Grasp Sampling Ablation (Table 3)

| Method | Query Time | Success |
|---|---|---|
| GraspNet-100 [2] | 10.3s | 76.7% |
| GraspNet-1 [32] | 0.6s | 65.1% |
| F3RM [1] | 6.9s | — |
| **GraspSplats** | **0.5s** | **81.4%** |

**Intuition**: GraspNet-100 是 LERF-TOGO 的做法 — 用 100 个不同 viewpoint 的 point cloud 累积 grasps, 解决 viewpoint variation。GraspSplats 用 GPG + Gaussian geometry, 0.5s 达到 81.4%, 比 GraspNet-100 的 76.7% 还高。这说明 **semantic affordance 由 CLIP features 提供, geometry 由 Gaussians 提供, sampling-based 方法够用** — 不需要 end-to-end learned grasp priors。

Reference: GraspNet-1Billion http://graspnet.net/

### 3.4 Segmentation Quality (Table 4)

| Method | IoU |
|---|---|
| LERF [6] | 39.0 |
| GraspSplats | 50.7 |

+11.7 IoU improvement, 主要归功于 hierarchical features (object-level + part-level 分开学)。

### 3.5 Initialization Ablation (Table 5)

| Method | Process Time | Train Iteration |
|---|---|---|
| Colmap-S [12] | 11.6s | 10,000 |
| Colmap-D [48] | 623.0s | 3,000 |
| GraspSplats (depth) | **0.7s** | **3,000** |

**Intuition**: Dense Colmap init (Colmap-D) 能减少 train iterations (3,000 vs 10,000), 但 Colmap 本身要 623s。GraspSplats 直接用 depth, init 只需 0.7s, 同样 3,000 iterations 收敛 — **depth 是免费的 dense geometry prior**。

### 3.6 Object/Part-level Success Breakdown (Table 6)

| Method | Object-level | Part-level |
|---|---|---|
| LERF-TOGO [2] | 81.5% | 63.0% |
| F3RM* [1] | 85.2% | 77.8% |
| **GraspSplats** | **96.3%** | **85.2%** |

**Intuition**: 
- Object-level GraspSplats 96.3% — 几乎完美, 说明 CLIP features + Gaussian geometry 足够
- Part-level gap 更大: GraspSplats 85.2% vs LERF-TOGO 63.0% (+22.2%) — hierarchical feature design 的直接体现
- Part-level 仍比 object-level 低 ~11%, 主要是细小 part (e.g., knife handle) 的 segmentation 和 grasp 都更难

---

## 4. Dynamic Scene 实验设计

实验分三个难度:
- **Easy**: translation only (no rotation)
- **Medium**: rotation 180°, 有 hand occlusion
- **Hard**: translation + rotation 同时, 有 occlusion

GraspSplats 在 dynamic 达到 74.2% (overall), 主要 fail 在 hard cases 的 fast rotation — tracking 丢失 correspondence, Kabsch 求 transform 不准。

**Failure analysis** (Appendix E):
- 单色、对称物体难 track (textureless → CoTracker 没 feature 可 track)
- Fast rotation 导致 motion blur + occlusion
- 假设 rigid transformation, 不支持 deformable objects (dough, clay)

---

## 5. Hardware Configuration (Appendix F)

- Robot: Franka Research (FR3)
- Cameras: 3× Intel RealSense D435 (1 in-wrist + 2 third-person)
- Gripper: UMI gripper [50]
- Compute: RTX 4090 + Intel i9-13900k

Reference: UMI gripper paper https://arxiv.org/abs/2402.10329

---

## 6. Intuition 总结: 为什么 GraspSplats work

### 6.1 Explicit vs Implicit Representation

NeRF 是一个 MLP $f_\theta(\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$, 整个 scene 编码在 $\theta$ 里。要 query 一个 point, 必须前向 MLP; 要 edit scene, 必须 retrain $\theta$。

3DGS 是一组 explicit Gaussians $\{\mathbf{G}_i\}$, 每个 $\mathbf{G}_i = \{\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \alpha_i, \mathbf{c}_i, \mathbf{f}_i\}$:
- $\boldsymbol{\mu}_i \in \mathbb{R}^3$: center position
- $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3\times3}$: covariance matrix (decomposed as $\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$ for differentiability)
- $\alpha_i$: opacity
- $\mathbf{c}_i$: color (spherical harmonics in original GS)
- $\mathbf{f}_i$: feature vector (GraspSplats extension)

要 query, 直接索引 Gaussians; 要 edit, 直接 transform $\boldsymbol{\mu}_i$ 和 $\mathbf{R}_i$。这是 GraspSplats 能做 dynamic scene 的根本原因。

### 6.2 Semantic + Geometry 解耦

GraspSplats 的设计哲学: **semantics 来自 CLIP (learned prior), geometry 来自 Gaussians (optimized representation), grasp sampling 来自 GPG (classical geometry)**。三者解耦, 各取所长:

- CLIP 提供 open-vocabulary semantics, 但只有 2D → 通过 differentiable rasterization distill 到 3D
- Gaussians 提供 explicit geometry, 通过 depth supervision + texture supervision 优化
- GPG 提供 viewpoint-invariant grasp sampling, 不需要训练数据

对比 end-to-end methods (GraspNet [32]): 它们要学 semantic priors, 但受限于训练分布; 对新 object category 泛化差。

### 6.3 Hierarchical Features 解决 CLIP 的 bag-of-words problem

CLIP 的 text-image alignment 是 global 的 — 'mug handle' 会被 decompose 成 'mug' + 'handle' 两个 concept 独立激活, 所以会同时匹配所有 mug 和所有 handle。

GraspSplats 的解法: 先 object-level MAP 得到 'mug' 的 global feature, 再在 mug 的 bounding box 内 crop patch 算 'handle' 的 local feature。这样 part-level feature 只在 object context 内有效, 避免 bag-of-words 混淆。

这在 Table 4 的 IoU +11.7 和 Table 6 的 part-level +22.2% 上直接体现。

---

## 7. Limitations 和 Future Directions

1. **Rigid transformation only**: Kabsch 假设 rigid, 不支持 deformable objects。但 Gaussian representation conceptually 支持 (deform 每个 Gaussian 独立), 未来可以结合 physics simulation。

2. **Tracking sensitivity**: Fast rotation + occlusion → CoTracker fail。可以考虑:
   - Re-sample keypoints during task (论文提到是 future work)
   - 用 semantic + geometric priors 做 optimization-based tracking
   - 用 more robust trackers (e.g., TAP-Vid [46])

3. **单色/对称物体**: Textureless objects 给 CoTracker 的 feature 少, 容易 drift。可以引入 geometric features (edge, corner) 补充。

4. **Placement**: Grasping 后 gripper 遮挡 view, object orientation 不确定, placement 困难。可以加 in-hand re-observation。

---

## 8. 和 Concurrent Work 的对比

- **Feature 3DGS [37]**, **LangSplat [39]**: 只做 appearance editing / language embedding, 不针对 manipulation
- **Feature Splatting [38]**: GraspSplats 的 base, 但 reference feature computation 慢; GraspSplats 加 hierarchical features + depth init, 快 10×
- **GaussianGrasper [40]**: Concurrent work, 也 combine GS + feature distillation for grasping, 但 (1) 不做 part-level query, (2) 假设 object 只被 robot arm 移动, (3) reference feature 仍 costly
- **Object-aware GS [41]**: 只 fuse fixed cameras, 不 address part-level manipulation

Reference: Feature Splatting https://arxiv.org/abs/2404.01223, GaussianGrasper https://arxiv.org/abs/2403.09637

---

## 9. 个人 Commentary

从 Karpathy 的视角看, 这篇 paper 有几个值得注意的点:

1. **Representation matters**: 3DGS 作为 explicit representation, 在 manipulation 这种需要 edit + query 的场景下, 天然优于 NeRF。这呼应了你在 "Software 2.0" 之外对 "explicit structure" 的重视 — learned priors (CLIP) + explicit structure (Gaussians) 的 hybrid 比纯 learned pipeline 更 sample-efficient + controllable。

2. **Classical methods 的价值**: GPG [16, 17] 是 2016-2017 的工作, 但配合 CLIP features + Gaussian geometry, 依然 beat GraspNet (2020) 的 end-to-end pipeline。这说明 **semantic priors 和 geometric reasoning 解耦后, 各自的 SOTA 组合起来很强**。不需要把所有东西塞进一个 end-to-end network。

3. **Depth supervision 是 free lunch**: RGBD camera 本来就有 depth, 直接用来 init + supervise, 跳过 SfM。这个工程选择让训练从 10min 降到 60s, 是 10× speedup 的主要来源。很多 GS 方向的工作忽略了 RGBD 的价值。

4. **Hierarchical features 对应 human cognition**: 人也是先认 object 再认 part ( mug → handle), 直接 query 'mug handle' 是 anti-pattern。这个 hierarchical design 不只是 engineering trick, 更 aligns with compositional generalization。

Potential future work 我觉得有意思的方向:
- Deformable object tracking (e.g., 衣服、绳索) — Gaussians 天然支持 non-rigid deformation, 只是 Kabsch 要换成 non-rigid registration (e.g., Gaussian mixture registration)
- Tactile feedback integration — Gaussian 的 geometry 可以用来 predict contact points, 结合 tactile sensor 做 closed-loop grasping
- LLM-driven task decomposition — 给 LLM 看 rendered scene + language features, 让 LLM 输出 manipulation plan, GraspSplats 执行 — 这条 path 是 zero-shot generalist manipulator 的 promising direction
