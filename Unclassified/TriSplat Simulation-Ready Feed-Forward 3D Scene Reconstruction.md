---
source_pdf: TriSplat Simulation-Ready Feed-Forward 3D Scene Reconstruction.pdf
paper_sha256: 1d7a20b8d99f448dda60a3f52da67861f73aafaa18fe8df731e8d1ea54cbbd1d
processed_at: '2026-08-12T18:18:04-07:00'
target_folder: 2026-05
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TriSplat 人话版

Andrej，咱们坐下来喝杯咖啡聊聊这篇paper。我先把核心故事讲清楚，再钻细节。

---

## 一、这篇paper到底在解决什么痛点

你想象一个robotics工程师，他要用feed-forward 3D reconstruction的输出做sim2real训练。他跑一个Gaussian splatting的feed-forward model，得到一堆Gaussian ellipsoid，看起来rendering效果挺漂亮。然后他想把这个scene导入Isaac Sim做collision detection，问题来了——**Isaac Sim只吃triangle mesh**。

他不得不跑一个TSDF fusion把Gaussian cloud转成mesh。这一转，quality掉6个dB，thin structures全没了，还要花十几秒到几百秒。feed-forward的"一次forward就出结果"承诺彻底破了。

这就是TriSplat要解决的core problem：**下游simulation engine的native language是triangle，你却硬要喂它Gaussian，中间的翻译过程既慢又损**。

TriSplat说，那我就直接让network输出triangle好了。rendering primitive = mesh primitive = simulation primitive，三方合一，没有翻译损失。

这听起来简单，做起来极难。paper的整个technical contribution都在解释"为什么直接让network输出triangle是tricky的，以及我们怎么让它work"。

---

## 二、为什么Triangle难，Gaussian容易

你拿一个Gaussian ellipsoid，它的rendering footprint是smooth radial falloff。你这个Gaussian的位置偏了10个pixel，它仍然覆盖目标区域的大部分，仍然能拿到gradient signal来纠正自己。orientation偏了15度，Gaussian的soft halo看起来只是略微变形，gradient依然友好。

Triangle完全不一样。它是hard-edged的。位置偏10个pixel，triangle可能完全miss目标pixel，gradient直接是zero。orientation偏15度，silhouette处直接暴露hard edge，rendering看起来像碎玻璃。Triangle的loss landscape比Gaussian崎岖得多，容易卡在local minimum。

这就是为什么前人Held et al.在triangle splatting原始paper里只做per-scene optimization——慢慢迭代能收敛。但feed-forward是one-shot prediction，没有迭代机会，必须一次预测对。**这是TriSplat要解决的核心technical challenge**。

---

## 三、TriSplat怎么让Triangle在feed-forward下work

paper给了四个关键insight，缺一不可，我一个个讲intuition。

### Insight 1: Normal不要让network自己学，从point map推

你让network预测每个pixel的3D point，得到一张dense point map $\mathbf{P} \in \mathbb{R}^{H\times W\times 3}$。然后对每个pixel，取它右边和下边的相邻pixel，cross product一下，就是surface normal：

$$
\mathbf{n}_{\text{geo}} = \text{normalize}(\Delta_x \times \Delta_y)
$$

这里 $\Delta_x$, $\Delta_y$ 是point map在horizontal和vertical方向的差分。$\times$ 是cross product。出来的就是unit normal。

这个trick的intuition是：adjacent pixel的point在物理上应该是相邻surface上的点，所以它们的差分天然给出local tangent vector，cross product给出normal。你不用让network单独去学一个3-DOF的quaternion——point map本身已经encode了geometry信息，normal可以free derive。

这跟DUSt3R、VGGT那套pointmap prediction的思路一脉相承。point map是更fundamental的representation，normal是它的derivative。

### Insight 2: 但finite difference太noisy，要learn一个refinement head

finite difference在depth edge附近会爆掉——你跨过一条edge，相邻pixel的point突然跳变，cross product给出garbage normal。paper训练一个轻量U-Net来refine：

$$
\mathbf{n}_{\text{ref}} = \text{normalize}\big(\mathbf{n}_{\text{sm}} + f_\theta(\mathbf{n}_{\text{geo}}, \mathbf{n}_{\text{sm}}, \mathbf{I}_v, \mathbf{D}_v, \mathbf{m})\big)
$$

输入是：raw geometry normal $\mathbf{n}_{\text{geo}}$、smoothed normal $\mathbf{n}_{\text{sm}}$、RGB image $\mathbf{I}_v$、depth map $\mathbf{D}_v$、validity mask $\mathbf{m}$。输出是correction term加到smoothed normal上。

**关键trick**：这个U-Net的output layer是**zero-initialized**的。训练初期 $f_\theta(\cdot) = 0$，所以 $\mathbf{n}_{\text{ref}} = \mathbf{n}_{\text{sm}}$，identity mapping。随着训练，network慢慢学到correction。这避免了random init的head在训练初期perturb orientation、把triangle orientation搞乱。

这种"identity start, gradually learn residual"的pattern在deep learning里到处都是——比如controlnet的zero convolution，比如layer scale的zero init。intuition都是一样的：**让network先copy input，再慢慢学delta，避免一开始就inject noise**。

### Insight 3: 训练初期point map太烂，finite difference出garbage，需要teacher bootstrap

这是chicken-and-egg problem：
- Point map不准 → normal不准 → triangle misoriented → rendering差 → photometric gradient差 → point map没法improve

paper用Omnidata这个pretrained monocular normal estimator做teacher。对每个input image跑Omnidata拿到teacher normal $\mathbf{n}_{\text{tch}}$。然后blend：

$$
\mathbf{n}_{\text{fwd}} = \text{normalize}\big(\alpha(t) \mathbf{n}_{\text{tch}} + (1-\alpha(t)) \mathbf{n}_{\text{ref}}\big)
$$

$\alpha(t)$ 是time-varying coefficient，用cosine schedule从1衰减到0：

- 训练前6000步：$\alpha = 1$，完全用teacher
- 6000到20000步：$\alpha$ cosine decay
- 20000步之后：$\alpha = 0$，完全用model自己

**这个bootstrap的关键点**：它作用在forward pass representation上，不是loss term。teacher normal直接进入triangle construction（影响Eq.2里的 $\mathbf{R}_n$），直接影响rendering output，直接shape所有downstream gradient。

如果你只是加一个teacher-matching loss $\|\mathbf{n}_{\text{ref}} - \mathbf{n}_{\text{tch}}\|$，network可以学到"normal匹配teacher但rendering还是烂"。但forward bootstrap强制teacher normal真的进入rendering pipeline，photometric gradient是从rendering回流到point map head的，这条path是consistent的。

这跟knowledge distillation的"forward-pass distillation"思想类似——不是让student match teacher的output，而是把teacher的intermediate representation直接inject进student的forward pass。

### Insight 4: Triangle对orientation太敏感，需要soft-to-hard curriculum

Gaussian的soft falloff给gradient提供forgiveness。Triangle没有。早期triangle position不准，可能完全miss目标pixel，gradient = 0，学不动。

paper用两个schedule解决：

**Opacity schedule**：
$$
o = \frac{1}{2}\Big(1 - (1-p)^{e(t)} + p^{e(t)}\Big)
$$

$p$ 是predicted density logit过sigmoid，$e(t)$ 从1线性增到2。当 $e=1$，$o = p$，identity。当 $e$ 增大，中间值被推向0或1，opacity逐渐binary化。再加一个temperature $\tau(t)$ 从1到5，进一步sharpen。

**Blur schedule**：
$$
\sigma = \text{sigmoid}(\hat{\sigma}) \cdot \beta(t)
$$

$\beta(t)$ 从1.0线性衰减到0.5。早期triangle有broad blur，alpha falloff柔和，覆盖范围广，gradient coverage好。后期blur减小，triangle收紧成crisp surface element。

**Intuition**：这其实是**把triangle临时变成"Gaussian-like"**来启动训练，然后逐步sharpen成真triangle。Curriculum learning的经典套路——先学容易的soft version，再逼近hard version。

---

## 四、Architecture的几何骨架

整个pipeline长这样：

```
Input images (V张, unposed)
        ↓
DINOv2 ViT-L/14 backbone (with 2D RoPE)
        ↓
Decoder (alternating intra-view self-attn + cross-view joint-attn)
        ↓
    ┌───┴───┬───────┐
   Point   Camera  Primitive
    Head    Head    Head
    ↓       ↓       ↓
Point map  SE(3)   Triangle attrs (density, scale, quat, SH, blur)
    ↓
Geometry normal (finite diff)
    ↓
U-Net refinement
    ↓
Mono-normal bootstrap blend
    ↓
Tangent frame construction → R_n
    ↓
Triangle vertex: v_k = R_c R_n (T_k ⊙ s) + c
    ↓
Differentiable triangle rasterizer
    ↓
Rendered image + depth + normal
    ↓
Losses: photo + cam + normal
```

几个architecture choice的intuition：

**DINOv2 backbone**: 已经被DUSt3R、NoPoSplat、YoNoSplat等无数工作验证，self-supervised feature对geometry和semantic都有响应，cross-view matching友好。

**Alternating attention decoder**: intra-view self-attention做单视图spatial reasoning，cross-view joint attention做multi-view correspondence。这是YoNoSplat/PI3的paradigm，比cost volume更general，不受plane sweep discretization限制。

**Point head参数化**: $\mathbf{p} = z \cdot (u, v, 1)^\top$，其中 $z = \exp(z')$。这把image formation model直接embed进prediction——3D point = depth × ray。DepthSplat也用类似idea。好处是depth不准时lateral也会scale，gradient signal强。

**Camera head用SVD orthogonalization**: 预测3×3 matrix，SVD投影到SO(3)。这是Levinson et al. NeurIPS 2020验证的方法，比quaternion-based预测更stable。

**Pairwise relative pose loss**: 对所有view pair算relative translation和rotation loss。好处是gauge invariant（不依赖global coordinate frame），supervision dense（V个view给V(V-1)个约束）。

---

## 五、Mesh Export有多trivial

Gaussian baseline的mesh export：
1. Render所有view的depth map
2. TSDF fusion（voxel size 0.005, truncation 0.1）
3. Marching cubes
4. Connected component analysis，保留top 50 clusters
5. Clean up floaters, holes
6. 总耗时：15秒到几百秒，scale with volume

TriSplat的mesh export：
1. Forward pass已经输出oriented triangle in world space
2. Low-opacity triangle pruning（threshold 0.10）
3. Winding order correction（用per-pixel normal对比face normal）
4. Vertex deduplication via quantized position hashing（precision $10^{-5}$）
5. 总耗时：< 0.1秒

这0.1秒里没有geometry inference，全是bookkeeping。**整个representation已经是mesh了，export只是format conversion**。

---

## 六、实验数据说什么

### Surface Quality（DL3DV, Table 1）

F1 score对比（6 views）：
- MVSplat: 0.118
- DepthSplat: 0.145
- AnySplat: 0.093
- YoNoSplat: 0.106
- **TriSplat: 0.287**

TriSplat的F1是baseline的2-3倍。Recall差距最大：TriSplat 0.448 vs YoNoSplat 0.189。这意味着TSDF-fused mesh只覆盖了19%的GT surface，TriSplat覆盖45%。

**Intuition**: TSDF是volumetric discretization，thin structures（椅子腿、桌沿、树枝）在voxel grid里被smooth掉。Triangle原生表达这些thin structure没问题，每个pixel一个triangle，密度由input resolution决定。

### Mesh Rendering Quality（RE10K, Table 3）

PSNR对比：
- YoNoSplat primitive rendering: 27.19 dB
- YoNoSplat mesh rendering (TSDF后): 21.07 dB（**掉6.12 dB**）
- TriSplat primitive rendering: 26.46 dB
- TriSplat mesh rendering: 23.25 dB（**掉3.21 dB**）

TriSplat的primitive rendering本来略低于YoNoSplat（26.46 vs 27.19），因为triangle的hard edge比Gaussian soft falloff损失一点rendering quality。但**export成mesh后TriSplat反超**（23.25 vs 21.07），因为Gaussian的TSDF conversion损失大，TriSplat几乎无损。

这就是paper的central claim的quantitative evidence：**representation match downstream consumer比representation本身optimization更重要**。

### Zero-shot ScanNet Normal（Table 4）

Mean angular error：
- MVSplat: 102.2°（基本没用）
- DepthSplat: 54.9°
- YoNoSplat: 59.8°
- **TriSplat: 27.9°**

<30° accuracy：
- DepthSplat: 29.4%
- YoNoSplat: 31.9%
- **TriSplat: 71.7%**

这个gap是**stunning**的。TriSplat的normal accuracy是baseline的2倍多。原因正是geometry-anchored normal pipeline + mono-normal bootstrap——它们直接优化orientation quality，不靠indirect rendering signal。

### Runtime（Figure 8）

6 views time-to-mesh：
- AnySplat: 18.7秒
- DepthSplat: 100+秒
- **TriSplat: 0.57秒**

33×到249×的speedup。这是structural speedup——Gaussian pipeline = feed-forward + TSDF，TriSplat pipeline = feed-forward only。TSDF的cost scale with reconstructed volume，跟network无关。

---

## 七、Ablation讲的story

Table 5的ablation非常clean，每个component针对不同failure mode：

| 移除什么 | F1下降 | PSNR下降 | 主要影响 |
|---|---|---|---|
| Normal anchoring | -0.057 | -1.11 | Orientation unconstrained，triangle center漂移 |
| Mono-normal bootstrap | -0.065 | -1.08 | 早期chicken-and-egg，point map卡住 |
| Normal refinement | -0.059 | -1.58 | Finite diff noise在edge处变rasterization artifact |
| Progressive sharpening | -0.062 | -1.44 | 早期hard triangle gradient = 0，学不动 |

四个component都important，互补性强。最值得注意：**normal refinement对LPIPS影响最大**（+0.111），因为finite difference noise在depth edges直接surface as visual artifact，perceptual损失大。**normal anchoring对F1影响最大**，因为unconstrained quaternion让triangle center漂移，破坏surface coverage。

---

## 八、跟相关工作的位置关系

### Gaussian feed-forward谱系

```
MVSplat (cost volume) → DepthSplat (cost volume + depth)
                          ↓
NoPoSplat (pose-free cost volume) → AnySplat, YoNoSplat (pose-free transformer)
                                     ↓
                                   MeshSplat, SurfelSplat (geometry-aware Gaussian)
                                     ↓
                                   TriSplat (triangle-native, pose-free)
```

TriSplat继承YoNoSplat的pose-free transformer架构，但primitive从Gaussian换成triangle，加了normal anchoring pipeline。

### Triangle splatting原始工作

Held et al. 2025的triangle splatting paper只在per-scene optimization下work。per-scene可以慢慢iterate收敛orientation。Feed-forward是one-shot，必须一次预测对，所以TriSplat需要normal anchoring + bootstrap + sharpening这三个per-scene不需要的innovation。

### Object-level mesh reconstruction

InstantMesh、MeshLRM、MeshFormer直接预测mesh，但只针对object，从controlled viewpoints。TriSplat extend到scene-level + unposed，难度量级不同——scene的scale、topology、viewpoint variation都远超object。

### DUSt3R / VGGT pointmap谱系

DUSt3R、MASt3R、VGGT预测dense pointmap + camera，但representation是point cloud，没有surface structure。TriSplat借鉴pointmap parameterization（Eq.1），但把point map作为intermediate，final是triangle。

---

## 九、Limitations和我的思考

Paper自己承认的：
1. **Non-manifold triangle soup**：导出的mesh不是watertight，FEM用不了。但90%的robotics/embodied AI应用只需要collision geometry，不需要watertight。
2. **Triangle density tied to input resolution**：224×224 input = 50K triangle，high-res input = million triangle，可能redundant。

我自己的几个observation：

**View-dependent appearance缺失**: 只用zeroth-order SH，specular和view-dependent reflection处理不了。DL3DV有rich material，这是limitation。可以extend到higher-order SH，但parameter count会涨。

**Triangle是planar的**: curved surface需要多个triangle拼接，density高。Gaussian的ellipsoid能更好fit curved surface。但实验显示mesh rendering quality反而更好——triangle的hard edge比Gaussian soft blending更sharp。这是trade-off，不是绝对劣势。

**Scale range hand-tuned**: Table E显示 $[s_{\min}, s_{\max}]$ 对结果敏感。能否让network自己学scale range是future direction。

**Long-horizon scene update**: Robot在scene里operate，object移动，如何incremental update triangle mesh？TriSplat是single-shot，continuous SLAM + reconstruction还需要研究。

**Generative 3D的implication**: 如果triangle-native feed-forward可行，能否直接generate triangle mesh scene-level？这跟InstantMesh方向类似但scale更大。

---

## 十、Meta-level insight

这篇paper给我的最大启发：**representation choice比algorithmic improvement更fundamental**。

Gaussian feed-forward方法在过去两年疯狂改进——cost volume、transformer、pose-free、geometry-aware——但都困在"Gaussian → mesh"这个translation gap里。每一步algorithmic improvement都被TSDF fusion的information loss吃掉。

TriSplat说，别在algorithm层面卷了，去representation层面commit to triangle。rendering quality略损一点（triangle hard edge vs Gaussian soft falloff），但simulation-readiness直接拿到。**下游consumer的native language决定representation choice**。

这跟你们在Tesla做simulation的intuition应该一致——physics engine吃triangle mesh，你feed它Gaussian再TSDF转回来，中间损失不可逆。不如直接在representation层面就commit到triangle。

paper的数学细节、公式推导、实验数据都围绕这个central claim展开。四个technical innovation（normal anchoring、mono-normal bootstrap、normal refinement、progressive sharpening）都是为了让triangle在feed-forward one-shot setting下收敛——这些是per-scene optimization不需要的，但feed-forward必须的。

Reference几个相关资源：
- Triangle Splatting原始paper: https://arxiv.org/abs/2505.05756
- TriSplat project page: https://lhmd.top/trisplat
- YoNoSplat: https://arxiv.org/abs/2511.07321
- DUSt3R: https://dust3r.europe.naverlabs.com/
- VGGT: https://vgg-t.github.io/
- Omnidata: https://omnidata.vision/
- DINOv2: https://dinov2.metademolab.com/
- Isaac Sim: https://developer.nvidia.com/isaac-sim
- 2DGS: https://surh.github.io/2d-gaussian-splatting/
- MeshSplat: https://arxiv.org/abs/2508.17811
- DepthSplat: https://arxiv.org/abs/2412.13509

---

希望这版"人话"版帮你把intuition build起来，Andrej。核心就是：**下游consumer是triangle mesh，你硬要feed它Gaussian然后TSDF翻译，翻译过程又慢又损。TriSplat让network直接说triangle的language，省掉翻译**。technical challenge在于triangle对orientation太敏感、feed-forward没法iterate，所以他们用geometry-derived normal + monocular teacher bootstrap + soft-to-hard curriculum三个trick联合解决。实验数据干净利落证明每个trick都necessary且complementary。

---

# TriSplat 深度解析

非常好的 paper 来读，Andrej。这篇工作在 2026 年的 feed-forward 3D reconstruction 领域里属于相当有洞察的一类——它把"representation 选择"这个看似表层的工程问题，提升到了决定 simulation-readiness 的核心问题层面。我会按照 intuition → architecture → 公式细节 → 实验解读 → 相关联想的层次展开。

---

## 1. 核心问题：为什么 Triangle-Native 是必要的

当前 feed-forward reconstruction 领域被 Gaussian primitive 统治（MVSplat, DepthSplat, NoPoSplat, YoNoSplat, AnySplat 等）。Gaussian 的优势在于 **soft radial falloff 提供局部 forgiving 的 gradient coverage**，对位置误差、scale 误差、orientation 误差都有"容差"。但 Gaussian 的 fundamental limitation 在于：它的 surface 是 **implicit** 的——你渲染出来的是一组 overlapping ellipsoids 的 alpha-blending 结果，没有一个明确的 surface boundary。

下游 task 如 robotics, embodied AI, physics simulation 需要 **explicit triangle mesh**。当前 pipeline 的做法是：
1. Feed-forward Gaussian prediction
2. TSDF fusion 或 Poisson reconstruction 把 Gaussian cloud 转成 mesh
3. Mesh cleanup（去 floater、补 hole、winding order 修正）

这个后处理步骤 **breaks the feed-forward promise**：训练时优化的是 Gaussian rendering quality，但 inference 时 downstream consumer 是 mesh，二者之间存在 representation gap。Table C 的数据触目惊心：YoNoSplat primitive rendering 27.19 dB，TSDF 转 mesh 后 21.07 dB，**掉了 6.12 dB**。MVSplat 掉了 10.70 dB。这就是 representation mismatch 的代价。

TriSplat 的核心洞察：**让 rendering primitive 本身就是 mesh primitive**，那么训练目标和 downstream consumption target 一致，gap 趋近于零。TriSplat 的 primitive-to-mesh degradation 只有 3.21 dB，且这部分 loss 主要来自 low-opacity triangle pruning 而不是 representation conversion。

Reference:
- Triangle Splatting原始论文: https://arxiv.org/abs/2505.05756
- YoNoSplat: https://arxiv.org/abs/2511.07321
- NoPoSplat: https://arxiv.org/abs/2410.24207

---

## 2. Architecture 整体解析

参考 Figure 2 的 overview，整个网络可以分解为 4 个 stage：

### Stage 1: Feature Extraction
- **Backbone**: DINOv2 ViT-L/14，patch size 14
- 附加 **2D rotary position embeddings** (RoPE 2D) 提供位置信息
- **Per-pixel ray-direction embeddings** 提供几何条件
- 对 pose-free 设置，额外有 **per-pixel intrinsic encoding**（4th-degree positional encoding）

DINOv2 是 self-supervised 预训练的 ViT，特征对几何/语义都有不错的响应，已经被 DUSt3R / MASt3R / VGGT / NoPoSplat 这一支线广泛验证。

### Stage 2: Cross-view Decoder
Decoder block 采用 **alternating attention**：
- **Intra-view self-attention**：单视图内的 spatial reasoning
- **Cross-view joint attention**：多视图之间的 correspondence aggregation
- 2D RoPE 贯穿整个 decoder
- Token dimension $d = 1024$

这个设计继承了 YoNoSplat / PI3 的 paradigm，关键在于 cross-view attention 替代传统 cost volume，把 multi-view matching 变成 transformer 内部的信息流动。

### Stage 3: Three Parallel Heads

#### Point Head
预测 dense local 3D point map $\mathbf{P} \in \mathbb{R}^{H \times W \times 3}$（per camera coordinate frame）。
- 5-layer transformer decoder (dim 1024, 16 heads, MLP ratio 4)
- 输出 3 个 unconstrained scalars $(u, v, z')$
- Pixel-shuffle upsampling 到 full resolution

#### Camera Head
预测 per-view SE(3) pose：
- Mean-pooling decoder tokens
- 输出 translation + 3×3 matrix
- 通过 **SVD orthogonalization** 投影到 SO(3)
- 所有 pose 相对第一个 view（消除 gauge ambiguity）
- **Scheduled sampling**：训练时线性 decay 使用 GT pose 的概率（防 distribution shift）

#### Primitive Head
预测 per-pixel triangle attributes：
- 结构同 point head
- 输入 features 与 zero-init patch-embedded RGB tokens additive fusion（直接访问 appearance）
- Output dim $1 + d_{\text{tri}} = 12$
  - 1 个 density logit
  - 3 个 scale logits
  - 4 个 quaternion components
  - 3 个 zeroth-order SH coefficients
  - 1 个 blur parameter

### Stage 4: Geometry-Anchored Normal Refinement

这是 paper 的核心创新。后面单独详细解析。

---

## 3. 公式深度解析

### Eq. (1): Point Map 参数化

$$
\mathbf{p} = z \cdot (u, v, 1)^\top
$$

其中：
- $\mathbf{p} \in \mathbb{R}^3$ 是该 pixel 在 camera coordinate frame 下的 3D 点
- $z = \exp(z')$ 是 depth，通过 exp 保证严格正
- $u, v$ 是 lateral coordinates（横向位置）
- $(u, v, 1)$ 形成一个 ray direction

**Intuition**: 这个参数化把 image formation model 直接嵌入到 prediction 里——3D point = depth × ray。这跟 DepthSplat 的 insight 一致：与其预测独立 xyz 让网络去隐式学习透视投影，不如 explicit 地乘起来。当 depth 不准时，lateral 也会随之 scale，gradient signal 更强。

### Eq. (2): Triangle Vertex 构造

$$
\mathbf{v}_k = \mathbf{R}_c \mathbf{R}_n (\mathcal{T}_k \odot \mathbf{s}) + \mathbf{c}, \quad k \in \{1, 2, 3\}
$$

变量含义：
- $\mathbf{v}_k \in \mathbb{R}^3$：第 $k$ 个 vertex 的 world space 坐标
- $\mathbf{R}_c \in \mathbb{R}^{3\times 3}$：camera-to-world rotation
- $\mathbf{R}_n \in \mathbb{R}^{3\times 3}$：tangent-frame rotation，由 geometry normal 构造（这是关键，见 Eq. 4-5）
- $\mathcal{T}_k \in \mathbb{R}^3$：canonical equilateral triangle template 的第 $k$ 个 vertex，paper 中具体为：
  - $\mathcal{T}_1 = (0, 0.577, 0)$
  - $\mathcal{T}_2 = (-0.5, -0.289, 0)$
  - $\mathcal{T}_3 = (0.5, -0.289, 0)$
  - 整体 pre-scaled by factor 4
- $\mathbf{s} \in \mathbb{R}^3$：三个 sigmoid-mapped scale logits，bounded 到 $[s_{\min}, s_{\max}]$，再乘上 depth 和 pixel footprint 转换到 world space
- $\mathbf{c} \in \mathbb{R}^3$：triangle center，由 predicted point map + camera pose 投影得到
- $\odot$：element-wise multiplication

**Intuition**: Triangle 不直接预测三个 vertex 位置（这样太 unconstrained），而是预测 center + scale + orientation，从 canonical template 变形得到。这跟 3DGS 的"预测 mean + scale + rotation + SH + opacity"的结构完全平行——只是把 Gaussian ellipsoid 换成 triangle。

### Eq. (3): Geometry Normal from Finite Difference

$$
\mathbf{n}_{\text{geo}} = \text{normalize}(\Delta_x \times \Delta_y)
$$

变量含义：
- $\Delta_x, \Delta_y \in \mathbb{R}^{H \times W \times 3}$：point map $\mathbf{P}$ 的 horizontal/vertical padded derivatives
- $\times$：cross product
- $\text{normalize}$：unit normalization

If $\mathbf{n}_{\text{geo}} \cdot \mathbf{p} > 0$（normal 指向远离 camera），flip 朝向 camera。

**Intuition**: 这是 computer graphics 里标准的 surface normal 估计方法。给定一张 depth map 或 point map，相邻 pixel 之间形成两个 tangent vector $\Delta_x$ 和 $\Delta_y$，cross product 给出 normal。问题在于 finite difference 对 noise 非常敏感，尤其在 depth 边界处会出现严重 artifacts，所以需要后续 refinement。

### Eq. (4): Learned Normal Refinement

$$
\mathbf{n}_{\text{ref}} = \text{normalize}\big(\mathbf{n}_{\text{sm}} + f_\theta(\mathbf{n}_{\text{geo}}, \mathbf{n}_{\text{sm}}, \mathbf{I}_v, \mathbf{D}_v, \mathbf{m})\big)
$$

变量含义：
- $\mathbf{n}_{\text{ref}} \in \mathbb{R}^3$：refined unit normal
- $\mathbf{n}_{\text{sm}} \in \mathbb{R}^3$：smoothed geometry normal（average-pooling + orientation-aware box filter）
- $f_\theta$：lightweight U-Net refinement network
- $\mathbf{n}_{\text{geo}}$：raw geometry normal
- $\mathbf{I}_v$：downsampled RGB image
- $\mathbf{D}_v \in \mathbb{R}^{H \times W}$：predicted depth map
- $\mathbf{m}$：validity mask（Boolean）

**关键细节**: $f_\theta$ 的 output layer **zero-initialized**，所以训练初期 $f_\theta(\cdot) = 0$，$\mathbf{n}_{\text{ref}} = \mathbf{n}_{\text{sm}}$（identity mapping）。随着训练，network 逐渐学到 correction。这是 **warm-start trick**，避免随机初始化的 head 在训练初期扰动 orientation。

U-Net 结构：
- 4 encoder-decoder scales
- Channel progression: $36 \to 72 \to 144 \to 288$（encoder），decoder 对称
- 11 input channels：raw normal (3) + smoothed normal (3) + RGB (3) + depth (1) + mask (1)
- Residual mode with scale factor 0.25

### Eq. (5): Mono-Normal Bootstrap Blending

$$
\mathbf{n}_{\text{fwd}} = \text{normalize}\big(\alpha(t) \mathbf{n}_{\text{tch}} + (1 - \alpha(t)) \mathbf{n}_{\text{ref}}\big)
$$

变量含义：
- $\mathbf{n}_{\text{fwd}} \in \mathbb{R}^3$：最终用于 triangle 构造的 normal
- $\mathbf{n}_{\text{tch}} \in \mathbb{R}^3$：来自 pretrained Omnidata [49] 的 monocular normal teacher
- $\alpha(t) \in [0, 1]$：time-varying blending coefficient
- $\mathbf{n}_{\text{ref}}$：refined normal from Eq. (4)

### Eq. (6): Cosine Decay Schedule

$$
\alpha(t) = \frac{1}{2}\bigg(1 + \cos\bigg(\pi \cdot \frac{t - t_{\text{tk}}}{t_{\text{bl}} - t_{\text{tk}}}\bigg)\bigg)
$$

变量含义：
- $t$：training step
- $t_{\text{tk}}$：takeover phase 结束时间（paper 中 6,000 steps）
- $t_{\text{bl}}$：blending phase 结束时间（paper 中 20,000 steps）
- $\pi$：角度到 $[0, \pi]$ 的映射

**Three phases**:
- $t \leq t_{\text{tk}}$：takeover phase，$\alpha = 1$，完全用 teacher
- $t_{\text{tk}} < t < t_{\text{bl}}$：blending phase，$\alpha$ 从 1 cosine decay 到 0
- $t \geq t_{\text{bl}}$：release phase，$\alpha = 0$，完全用 model 自身 normal

**关键洞察**: 这个 bootstrap 作用在 **forward pass representation** 上，不是 loss term。也就是说，teacher normal 直接进入 triangle 构造（Eq. 2 的 $\mathbf{R}_n$），影响渲染输出和所有 downstream gradient。这跟 teacher-matching loss 有本质区别——loss 只提供 additive optimization signal，而 forward bootstrap 直接 shape 了 representation 本身。Paper 同时使用两者以最大化 stability。

### Eq. (7): Opacity Mapping with Exponent and Temperature

$$
o = \frac{1}{2}\Big(1 - (1 - p)^{e(t)} + p^{e(t)}\Big)
$$

变量含义：
- $p \in [0, 1]$：predicted density logit 通过 sigmoid
- $e(t)$：time-varying exponent，linearly ramps from $e_{\text{init}} = 1$ to $e_{\text{final}} = 2$ during warm-up
- $o \in [0, 1]$：final opacity

**Properties**:
- 当 $e(t) = 1$：$o = \frac{1}{2}(1 - (1-p) + p) = \frac{1}{2}(2p) = p$（identity）
- 当 $e \to \infty$：$o \to \mathbf{1}_{p > 0.5}$（binarization）
- 边界保持：$o(0) = 0, o(1) = 1$ for all $e > 0$

然后还有一个 temperature factor $\tau(t)$：
$$
o_{\text{final}} = \sigma(\tau(t) \cdot \text{logit}(o))
$$

其中 $\tau(t)$ linearly ramps from $\tau_{\text{init}} = 1.0$ to $\tau_{\text{final}} = 5.0$ over 16,000 steps。

**Intuition**: 这是把 soft probability 逐步 sharpen 到 binary decision 的 curriculum。早期 triangle 是"半透明"的，gradient 友好；后期变成"全有或全无"的 hard surface element，便于 mesh extraction。

### Eq. (8): Blur Scheduling

$$
\sigma = \text{sigmoid}(\hat{\sigma}) \cdot \beta(t)
$$

变量含义：
- $\hat{\sigma}$：raw predicted blur value
- $\beta(t)$：time-varying multiplier，linearly decays from $\beta_{\text{init}} = 1.0$ to $\beta_{\text{final}} = 0.5$ over 16,000 steps
- $\sigma$：实际用于 rasterizer 的 blur parameter，控制 triangle edges 的 alpha falloff

**Intuition**: 早期 triangle 有 broad, overlapping soft footprint，确保即使位置不准也能覆盖到目标 pixel，提供 dense gradient。后期 footprint 收紧，triangle 变成 crisp surface element。

Opacity 控制"贡献强度"，blur 控制"贡献空间范围"。两者协同，提供了比单一 scheduling 更丰富的 soft-to-crisp curriculum。

### Eq. (9, 10, 11, 12, 13): Losses

Total loss:
$$
\mathcal{L} = \lambda_{\text{photo}} \mathcal{L}_{\text{photo}} + \lambda_{\text{cam}} \mathcal{L}_{\text{cam}} + \lambda_{\text{normal}} \mathcal{L}_{\text{normal}}
$$

Photometric:
$$
\mathcal{L}_{\text{photo}} = \lambda_{\text{mse}} \|\hat{\mathbf{I}} - \mathbf{I}^*\|_2^2 + \lambda_{\text{lpips}} \text{LPIPS}(\hat{\mathbf{I}}, \mathbf{I}^*)
$$

Camera (pairwise):
$$
\mathcal{L}_{\text{cam}} = \omega_t \mathcal{L}_{\text{trans}} + \omega_r \mathcal{L}_{\text{rot}}
$$

Normal:
$$
\mathcal{L}_{\text{normal}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} (1 - \mathbf{n}_{\text{ref},i}^\top \mathbf{n}_{\text{tch},i})
$$

变量含义：
- $\hat{\mathbf{I}}, \mathbf{I}^*$：rendered / GT image
- $\mathcal{L}_{\text{trans}}$：Huber loss on pairwise relative translation
- $\mathcal{L}_{\text{rot}}$：angular loss on pairwise relative rotation
- $\mathcal{V}$：valid pixel set
- $\mathbf{n}_{\text{ref},i}, \mathbf{n}_{\text{tch},i}$：第 $i$ 个 pixel 的 refined normal 和 teacher normal

**Intuition on pairwise camera loss**: 相对 pose loss（vs. 绝对 pose regression）有两个优势：
1. Invariant to global coordinate frame（消除 gauge freedom）
2. 更 dense supervision：$V$ 个 view 给 $V(V-1)$ 个 constraint，每个 view pair 独立

---

## 4. 实验数据深度解读

### Table 1: Surface Quality on DL3DV

| Method | 6-view CD↓ | 6-view F1↑ | 12-view CD↓ | 12-view F1↑ | 24-view CD↓ | 24-view F1↑ |
|---|---|---|---|---|---|---|
| MVSplat | 1.143 | 0.118 | 0.802 | 0.135 | 0.695 | 0.156 |
| DepthSplat | 1.116 | 0.145 | 0.907 | 0.152 | 0.786 | 0.152 |
| AnySplat | 1.012 | 0.093 | 0.731 | 0.096 | 0.699 | 0.100 |
| YoNoSplat | 0.920 | 0.106 | 0.664 | 0.092 | 0.687 | 0.088 |
| **TriSplat** | **0.613** | **0.287** | **0.323** | **0.279** | **0.310** | **0.277** |

注意几个 phenomena：

1. **TriSplat 的 F1 大约是 baseline 的 2-3 倍**。这非常夸张。原因在于 Gaussian baseline 都需要 TSDF fusion 转 mesh，而 TSDF 是 volumetric discretization，会丢掉 thin structures、产生 hole，所以 Recall 极低。

2. **TriSplat 的 CD 改进比 F1 改进小**。CD 是双向 Chamfer distance，对全局 shape 错误敏感；F1 是 Precision/Recall 在 threshold $\delta = 0.05$ 内的 harmonic mean，对 coverage 细节敏感。说明 TriSplat 的优势主要在 surface coverage 细节，不在全局 shape correctness（baseline 这部分已经做得不错了）。

3. **View 数量从 6 → 24 增加时**，baseline F1 改进很慢（MVSplat: 0.118 → 0.156，AnySplat: 0.093 → 0.100）。但 TriSplat 也只是 0.287 → 0.277（甚至略降）。这说明 view 数量并不是 surface quality 的 bottleneck，representation choice 才是。

### Table 2: NVS Quality under Mesh Rendering

| Method | 6-view PSNR↑ | 12-view PSNR↑ | 24-view PSNR↑ |
|---|---|---|---|
| MVSplat | 14.75 | 15.16 | 15.72 |
| DepthSplat | 14.86 | 14.82 | 15.13 |
| AnySplat | 18.58 | 16.58 | 16.42 |
| YoNoSplat | 18.88 | 16.90 | 16.71 |
| **TriSplat** | **20.84** | **18.71** | **18.06** |

关键 insight：**这是 mesh rendering（即用标准 triangle rasterizer 渲染 exported mesh）的 metric**。Gaussian baseline 在 primitive rendering（Table A）上其实很 competitive：YoNoSplat 6-view primitive PSNR 是 23.88 dB，但 mesh rendering 掉到 18.88 dB（-4.99 dB）。TriSplat primitive rendering 23.22 dB，mesh rendering 20.84 dB（-2.38 dB）。

这个 gap 的来源：
- Gaussian baseline：TSDF fusion 丢信息
- TriSplat：只有 low-opacity pruning 丢一点信息

### Table 3: RE10K 6-view Comparison

最 striking 的数据是 **Recall**: TriSplat 0.560 vs. 最强 baseline YoNoSplat 0.333。这意味着 TriSplat 覆盖了 56% 的 GT surface，而 YoNoSplat 只覆盖 33%。这 23 个百分点的差距直接体现在 thin structure 完整性上——Figure 6 的 textured mesh 可视化非常清楚地显示了这一点：TSDF-fused mesh 丢失 thin structures 和 scene extent，TriSplat 完整保留。

### Table 4: Zero-shot ScanNet Depth/Normal

| Method | AbsRel↓ | Mean↓ | <30°↑ |
|---|---|---|---|
| MVSplat | 0.708 | 102.247 | 17.204 |
| DepthSplat | 0.279 | 54.861 | 29.403 |
| YoNoSplat | 0.270 | 59.803 | 31.862 |
| **TriSplat** | **0.188** | **27.901** | **71.708** |

Normal accuracy 的差距是 **stunning**：TriSplat 71.7% < 30°，最强 baseline 只有 31.9%。Mean angular error 27.9° vs. 54-60°。这说明 **normal anchoring pipeline + mono-normal bootstrap 是真正 work 的**——它们直接优化 orientation quality，不依赖任何 indirect signal。

### Table 5: Ablation Study

| Configuration | CD↓ | F1↑ | PSNR↑ | LPIPS↓ |
|---|---|---|---|---|
| Full model | 0.190 | 0.708 | 23.25 | 0.318 |
| w/o normal anchoring | 0.198 | 0.651 (-0.057) | 22.14 (-1.11) | 0.396 |
| w/o mono-normal bootstrap | 0.198 | 0.643 (-0.065) | 22.17 (-1.08) | 0.397 |
| w/o normal refinement | 0.193 | 0.649 (-0.059) | 21.67 (-1.58) | 0.429 (+0.111) |
| w/o progressive sharpening | 0.191 | 0.646 (-0.062) | 21.81 (-1.44) | 0.416 |

每个 component 的 ablation 都让所有 metric 下降 comparable margin，说明四个 component 针对不同 failure mode，互补性强。

特别值得注意：**normal refinement 对 LPIPS 影响最大**（+0.111），这是因为 finite-difference noise 在 depth edges 会直接 surface as rasterization artifacts，perceptual quality 损失大。而 **normal anchoring 对 F1 影响最大**（-0.057），因为 unconstrained quaternion 会让 triangle center 漂移，破坏 surface coverage。

---

## 5. Runtime Efficiency Analysis

Figure 8 的数据：
- TriSplat: 6-view 0.57s, 12-view 0.62s, 24-view 1.23s
- AnySplat (fastest Gaussian): 6-view 18.7s, 24-view 33.0s
- DepthSplat: 24-view 306s

TriSplat 比 fastest Gaussian baseline 快 **33×**（6 views）到 **249×**（24 views，slowest baseline）。这个 speedup 不是 "engineering trick"，而是 **structural**：

Gaussian pipeline = feed-forward pass + TSDF fusion
TriSplat pipeline = feed-forward pass

TSDF fusion 的 cost scales with **reconstructed volume**，跟 network 无关。所以 view 越多、scene 越大，TSDF 越慢。而 TriSplat 的 export 只需要 low-opacity pruning + vertex deduplication，< 0.1s。

Reference:
- TSDF fusion: https://graphics.stanford.edu/papers/volrange/volrange.pdf
- Marching cubes: https://en.wikipedia.org/wiki/Marching_cubes

---

## 6. 与相关工作 Positioning

### 6.1 Feed-Forward Gaussian 方法谱系

- **Cost volume based**: MVSplat, DepthSplat — 用 plane sweep cost volume 做 multi-view matching
- **Pose-free cost volume**: NoPoSplat — pose 和 Gaussian 联合预测
- **Pure transformer**: YoNoSplat, AnySplat — 不用 cost volume，纯 attention 做 cross-view
- **Geometry-aware Gaussian**: MeshSplat, SurfelSplat — 加 normal prediction 和 surface regularization

TriSplat 在这个 spectrum 上的位置：继承 YoNoSplat 的 pose-free transformer architecture，但把 primitive 从 Gaussian 换成 triangle，并加入 geometry-anchored normal pipeline。

### 6.2 Triangle Splatting 的 per-scene 前作

Paper [30] (Held et al., 2025) 是 triangle splatting 的原始工作，但只在 per-scene optimization 设置下 work。TriSplat 把它 extend 到 feed-forward 设置，需要解决的关键问题：
1. Per-scene optimization 可以慢慢收敛 orientation，feed-forward 必须一次预测正确
2. Per-scene 可以迭代 refine，feed-forward 没有 iterate 机会
3. Triangle 对 orientation noise 极敏感，feed-forward 需要 strong prior

所以 TriSplat 的 normal anchoring + mono-normal bootstrap + progressive sharpening 都是 **针对 feed-forward 设置的 essential innovations**，不是可选优化。

### 6.3 Object-level Mesh Reconstruction

InstantMesh, MeshLRM, MeshFormer, Pixel2Mesh 等方法直接预测 mesh，但只 work 在 object-level，从 controlled viewpoints。TriSplat 把这个 idea extend 到 scene-level + unposed，需要解决:
1. Scene 的 scale 远大于 object
2. Sparse, unposed views（vs. controlled multi-view）
3. Scene 的 surface topology 远复杂于 object

### 6.4 DUSt3R / MASt3R / VGGT 谱系

这一支线预测 dense pointmap 和 camera pose，但 representation 是 point cloud，没有 surface structure。TriSplat 借鉴了它们的 pointmap parameterization（Eq. 1），但把 point map 作为 **intermediate**，final representation 是 triangle。

Reference:
- DUSt3R: https://dust3r.europe.naverlabs.com/
- MASt3R: https://arxiv.org/abs/2406.09756
- VGGT: https://arxiv.org/abs/2503.11651

---

## 7. Intuition Building: 为什么 Triangle 对 Orientation 敏感

这是 paper 的 fundamental insight，值得深入理解。

**Gaussian**：渲染方程是 $\alpha$-blending of Gaussian footprints。一个 slightly misoriented Gaussian 仍然会 cover 大致正确的 image region，因为它有 radial falloff。位置误差导致的是 "soft halo"，不是 hard artifact。

**Triangle**：渲染是 hard-edged rasterization。一个 slightly misoriented triangle 会：
1. 在 silhouette 处暴露 hard edge
2. 错过目标 pixel，gradient = 0（vs. Gaussian 总有 gradient）
3. 在 grazing angle 产生 moiré pattern
4. Coverage area 突变（discontinuity）

所以 triangle 的 loss landscape 比 Gaussian **更崎岖**，容易 stuck 在 local minimum。这是为什么 TriSplat 需要 progressive sharpening——早期用 broad blur 让 gradient surface 平滑，后期 sharpen 到 hard surface。

类比：这是 **curriculum learning** 的经典案例。直接学 hard-edged triangle 太难，从 soft Gaussian-like footprint 开始，逐步 sharpen。

---

## 8. 关键设计选择的 Intuition

### 8.1 为什么 Normal 必须从 Point Map 推导

Paper 写得很明确：把 orientation 作为 unconstrained latent variable 让 network 自己学，效果差。原因：
1. Triangle orientation 是 3-DOF（quaternion），搜索空间大
2. Without geometric prior，network 学到的 orientation 在 surface 上不 consistent
3. Pixel-level prediction 难以 enforce cross-pixel consistency

通过 finite difference 从 point map 推 normal，自动获得 pixel-level consistency——adjacent pixel 的 point 接近，所以 normal 也接近。这是 **structural prior**。

### 8.2 为什么需要 Mono-Normal Bootstrap

Chicken-and-egg problem:
- 训练初期，point map 不准 → finite difference normal 噪声大
- 噪声大 → triangle misoriented → rendering artifacts
- Artifacts → gradient 不 reliable → point map 难以改进
- 死循环

Mono-normal teacher (Omnidata) 是 **pretrained single-image normal estimator**，不依赖 multi-view geometry，所以稳定。用它 bootstrap 训练初期，让 triangle 先有大致正确的 orientation，rendering 看起来合理，photometric gradient 才能回流到 point map head。

注意：teacher 是 **per-image monocular**，所以无法区分 left-facing vs. right-facing surface。Blending 阶段会在 multi-view 之间产生 inconsistency，但 paper 通过 validity mask + blending decay 限制这个问题。

### 8.3 为什么 Forward Bootstrap 优于 Teacher-Matching Loss

Forward bootstrap: $\mathbf{n}_{\text{fwd}} = \alpha \mathbf{n}_{\text{tch}} + (1-\alpha) \mathbf{n}_{\text{ref}}$
- Teacher normal 直接进入 triangle construction
- 影响 rendering output
- 影响 photometric gradient

Teacher-matching loss: $\mathcal{L} = \| \mathbf{n}_{\text{ref}} - \mathbf{n}_{\text{tch}} \|$
- 只对 $\mathbf{n}_{\text{ref}}$ 提供额外 gradient
- 不影响 rendering，不影响 photometric gradient
- Network 可以学到 "match teacher normal but render badly"

Forward bootstrap 强制 teacher normal 真的进入 rendering pipeline，gradient 是从 rendering 来的，不是从 normal matching 来的。这跟 **knowledge distillation 的 forward-pass variant** 思想类似。

### 8.4 为什么 Opacity + Blur 协同 Scheduling

单独 opacity scheduling：opacity 控制 "贡献强度"，但 triangle footprint 仍然是 hard edge。早期 triangle 可能完全 miss 目标 pixel，gradient = 0。

单独 blur scheduling：blur 让 footprint 软化，但 opacity distribution 仍然 soft，难以决定哪些 triangle 是 "real" surface，哪些是 outlier。

协同：blur 早期广覆盖，确保 gradient 流；opacity 早期 soft，后期 binary，逐步决定 surface membership。两者一起 provide "soft Gaussian-like" → "hard triangle" 的完整 curriculum。

---

## 9. Limitations 和未来方向

Paper 自己提到的 limitations：
1. **Non-manifold triangle soup**：导出的 mesh 不是 watertight，无法用于 FEM (finite element method)
2. **Per-pixel prediction ties triangle density to input resolution**：高分辨率 input = 更多 triangle，可能 redundant
3. **Topology-aware export** 和 **adaptive tessellation** 是未来方向

我可以补充几个 observation：

1. **没有 view-dependent appearance**: 只用 zeroth-order SH，无法处理 specular / view-dependent reflections。对 DL3DV 这种丰富 material 的 dataset，这是 limitation。

2. **Triangle 是 planar**：无法表达 curved surface。需要多个 triangle 拼接，density 高。这跟 Gaussian 的 "ellipsoid 可以 fit curved surface" 相比有劣势。但 paper 的实验显示这在实践中 OK，因为 mesh rendering quality 反而更好——因为 mesh 是 discrete 的，比 Gaussian 的 soft blending 更 sharp。

3. **Pose-free joint prediction 的 chicken-and-egg**: pose 不准时 point map 不准，point map 不准时 normal 不准，normal 不准时 triangle rendering 不准。Paper 用 scheduled sampling 缓解，但 fundamental coupling 还在。

4. **Scale 范围 hand-tuned**: Table E 显示 scale range $[s_{\min}, s_{\max}]$ 对结果敏感。paper 用 $[0.5, 18.0]$（Stage 1）和 $[1.2, 15.0]$（Stage 2）。能否让 network 自己学 scale 范围是 future work。

---

## 10. 个人 Thoughts 和开放问题

1. **Triangle vs. Surfel vs. Gaussian**: 这个 paper 让我重新思考 primitive 选择问题。Gaussian 在 rendering quality 上 unbeatable（soft falloff, view-dependent, easy gradient），但 simulation 需要 explicit surface。Triangle 是 simulation 的 native language，但 rendering 不友好。Surfel 是中间道路——planar disk，orientation 敏感度比 triangle 低（因为 round edge），但 surface extraction 仍需要后处理。未来会不会有 **adaptive primitive selection**？同一 scene 里 flat surface 用 triangle，smooth surface 用 Gaussian？

2. **Watertight mesh 的必要性**: Paper 说 limitation 是 non-manifold。但很多 simulation 不需要 watertight——只要 collision geometry OK 就行。FEM 才需要 watertight。所以 TriSplat 的 representation 对 90% 的 robotics / embodied AI 应用已经够用。

3. **Feed-forward + per-scene refinement**: 完全 feed-forward 是 TriSplat 的优势，但也是 limitation——遇到 OOD scene 时无法 adapt。一个 hybrid pipeline：TriSplat 作为 warm start，然后 short per-scene refinement，可能是个 sweet spot。但这又回到 per-scene optimization 的复杂性。

4. **Long-horizon embodied AI 的 representation 问题**: 如果 robot 在 scene 里长时间 operate，scene 会变化（object moved, door opened）。如何 maintain / update triangle mesh？这是 continuous SLAM + reconstruction 问题。TriSplat 提供了 single-shot representation，但 incremental update 还需要研究。

5. **Generative 3D 的 implication**: 现在 3D generation 领域多用 triplane, NeRF, Gaussian 作为 representation。如果 triangle-native feed-forward 可行，能否直接 generate triangle mesh？这跟 InstantMesh, MeshLRM 的方向类似，但场景级而非 object 级。

Reference 一些相关 link:
- 3DGS original: https://repo.samuelgarcia.xyz/papers/23-0801.html
- 2DGS: https://surh.github.io/2d-gaussian-splatting/
- SurfelSplats: https://arxiv.org/abs/2507.15602
- MeshSplat: https://arxiv.org/abs/2508.17811
- DINOv2: https://dinov2.metademolab.com/
- Omnidata: https://omnidata.vision/
- Isaac Sim: https://developer.nvidia.com/isaac-sim
- Unity: https://unity.com/
- Triangle Splatting原始paper: https://arxiv.org/abs/2505.05756
- TriSplat project page: https://lhmd.top/trisplat
- PI3: https://arxiv.org/abs/2507.13347
- DUSt3R: https://arxiv.org/abs/2312.14132
- VGGT: https://vgg-t.github.io/
- DepthSplat: https://arxiv.org/abs/2412.13509

---

## 11. 总结

TriSplat 的核心 contribution 可以凝练成一句话：**让 representation match downstream consumer，而不是让 downstream consumer 去 fit representation**。

Gaussian feed-forward 方法的 implicit surface 问题在 simulation context 下是 fatal 的——TSDF fusion 的 quality loss + runtime cost 都是 deal-breaker。TriSplat 通过：
1. Triangle-native representation
2. Geometry-anchored normal prediction
3. Mono-normal bootstrap（forward pass, not loss）
4. Progressive opacity + blur sharpening

四个 design choice 协同工作，把 triangle primitive 从 per-scene optimization extend 到 feed-forward pose-free setting，并直接 export simulation-ready mesh。

实验数据干净利落地证明了每个 design choice 的价值（ablation 互补性强），并且 surface quality / mesh rendering quality / runtime 三个维度都 substantial 改进。Zero-shot ScanNet 的 normal accuracy 提升尤其 impressive——mean angular error 减半，< 30° accuracy 翻倍。

这 paper 让我想到的一个 meta-level insight：**representation choice 决定 downstream applicability，比 algorithmic improvement 更 fundamental**。Gaussian vs. Triangle 之争，本质上是 "rendering-first" vs. "simulation-first" 的路线之争。TriSplat 表明，针对 simulation 这个下游 task，从 representation 层面就 commit to triangle 是最优解，而不是用 Gaussian 然后后处理。

希望这个解读对你 build intuition 有帮助，Andrej。如果你对某个具体细节（比如 U-Net 结构、scheduled sampling、TSDF fusion 对比、Isaac Sim 集成）想深入讨论，我可以展开。
