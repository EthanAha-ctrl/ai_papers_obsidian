---
source_pdf: Explicit Memory through Online 3D Gaussian Splatting Improves Class-Agnostic
  Video Segmentation.pdf
paper_sha256: 45f0b0230714d37378928b7df9f145cff218811c88d4e56a28bc62fb777cf09f
processed_at: '2026-08-04T06:17:55-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲：这篇 paper 到底在干嘛

## 一句话版本

机器人要分割视频里所有东西，但现有模型要么没有记忆（FastSAM 逐帧独立分割，导致同一物体在每帧 ID 不同），要么有"隐式记忆"但不可查询（SAM2 用神经网络 latent feature 做 memory，物体被遮挡重新出现时会丢 ID）。作者说：既然机器人本来就有 depth 和 camera pose，那把每帧 2D 分割结果"抬起"到 3D，存成 3D Gaussian Splatting (3DGS) 这个显式的几何表示，后续帧再把 3DGS 投影回 2D 作为 memory，就能修掉这两类毛病。

---

## Motivation：为什么这是个真问题

想象你让机器人去厨房倒杯水，它需要持续地认出"杯子"、"桌子"、"水龙头"这些东西。现实世界里会有遮挡（杯子被手挡住）、光照变化、物体移动。两个 SOTA 模型各自有不同毛病：

- **FastSAM** (https://arxiv.org/abs/2306.12156)：本质是个 image segmentation model，每帧独立预测。第 1 帧认出杯子 ID=5，第 2 帧可能给同一杯子 ID=12，因为完全没有 temporal memory。这就叫 "flickering"，对下游 task（比如抓取）是灾难。
- **SAM2** (https://arxiv.org/abs/2408.00714)：Meta 的 video extension，加了 memory attention 机制，能跨帧 track。但 memory 是个 latent tensor，模型自己"看着办"，外部无法查询、无法纠错。论文引用 SAM2-Long (https://arxiv.org/abs/2410.16268) 的观察：长视频里 SAM2 会"error accumulation"——物体被遮挡再出现时，SAM2 经常给它一个新 ID，导致同一物体有两个 track。

作者的关键 insight：在 robotics 场景，我们有 RGB-D camera 和 SLAM/odometry 给的 camera pose，这是 geometric prior，本来就在用。那为什么不让它帮我们记东西？

---

## 三个核心 design choice，用大白话讲

### 1. 3DGS 当 memory bank：为什么是 3DGS

3DGS (https://arxiv.org/abs/2308.04034) 本质上是把场景表示成几百万个带颜色、带透明度的 3D 椭球。每个 Gaussian 有：
- 位置 $\mu \in \mathbb{R}^3$：椭球中心
- 朝向 $q \in \mathbb{R}^4$：quaternion 表示旋转
- 尺度 $s \in \mathbb{R}^3$：三个轴的拉伸
- opacity $\sigma \in [0,1]$：不透明度
- 颜色 $c \in \mathbb{R}^3$：RGB

给定一个相机视角，把所有 Gaussian 投影到 2D，按深度排序 alpha-blend，就能渲染出图。这套机制 differentiable，所以可以用梯度下降训练。

作者加了一个新参数：**segment ID feature** $f^{\mathrm{ID}} \in \mathbb{R}^{D_{\mathrm{ID}}}$。每个 Gaussian 不光记颜色，还记"我属于哪个 object"。这样 3DGS 可以渲染出 "ID map"，每个 pixel 一个 segment ID。

为什么用 3DGS 而不是别的 3D 表示？

- **vs point cloud**：点云不可 differentiable render，没法把 2D prediction loss 反传到 3D。
- **vs voxel grid**：voxel 固定分辨率，memory 爆炸，3DGS 是 sparse adaptive。
- **vs NeRF**：NeRF 要 per-pixel ray-marching，慢，且 implicit 难查询。3DGS explicit + 实时 render。
- **vs implicit feature volume（NIERF/NeRF-SLAM 那类）**：3DGS 可直接 project + rasterize，工程友好。

而且 recent SLAM 工作 SplaTAM (https://arxiv.org/abs/2312.09403) 已经证明 online incremental 3DGS 可以实时跑，正好满足 video segmentation 的 online 需求。

### 2. Segment ID 用 vector 不用 integer：最妙的 design

这一段是 paper 最值得咂摸的部分。

直觉上"segment ID = 5"这种 integer 看起来最简单。但 3DGS 渲染时多个 Gaussian 会 alpha-blend，公式大致是：

$$C(x,y) = \sum_i \alpha_i \cdot c_i \cdot \prod_{j<i}(1 - \alpha_j)$$

这里 $\alpha_i$ 是第 $i$ 个 Gaussian 在 pixel $(x,y)$ 处的 blended opacity，$c_i$ 是它的 color。如果 $c_i$ 是 integer ID 比如 5 和 7，blend 出来可能是 6.2——这个数字没有任何语义，没法 decode。

但如果 $c_i$ 是个 vector，blend 出来就是个 weighted average vector，仍然有意义：它指向 "5 和 7 的混合方向"。这就是为什么必须用 vector。

更进一步，作者把 vector 解耦成两部分：
- **Direction（方向）**：编码 identity（哪个 object）
- **Magnitude（长度）**：编码 confidence（我多确定这是这个 object）

公式上每个 codebook vector 归一化到 unit norm，rendered feature $F^{\mathrm{ID}}$ 在 blend 后 magnitude 会自然反映"这些 Gaussian 是否一致"——如果都同意 ID=5，blend 出来的向量长度接近 1；如果有 disagreement（一半说 5 一半说 7），blend 后互相抵消，magnitude 下降。

这个 decoupling 让 alpha-blending 自然产生 confidence 信号，无需额外建模。我个人觉得这是 paper 里最 elegant 的部分。

### 3. Codebook 怎么设计：max-min sphere packing

需要一个 codebook $C = \{c_1, \ldots, c_N\}$，每个 $c_i \in \mathbb{R}^{D_{\mathrm{ID}}}$ 是 unit vector，要保证不同 ID 之间足够分开。优化目标（公式 1）：

$$L = -\min_{1 \leq i \leq N}\left(\min_{1 \leq j \leq N, j \neq i} \|c_i - c_j\|\right)$$

这是个 max-min 问题：最大化"最近的两个 codeword 之间的距离"。等价于把 N 个点在 $D_{\mathrm{ID}}$ 维球面上"撑开"，让最差情况下也能区分。

直觉：假设有 100 个 object，要在球面上放 100 个 ID vector，希望任意两个之间夹角尽量大。这与 FaceNet (https://arxiv.org/abs/1503.03832) 的 embedding learning、Tammes problem（球面上均匀撒点）数学同构。

Decoding 时（公式 2-3）：

$$m_{x,y} = \arg\max_i \left\{ \langle c_i, F^{\mathrm{ID}}_{x,y} \rangle \cdot \mathbf{1}_{\langle c_i, F^{\mathrm{ID}}_{x,y} \rangle > 0.5} \right\}$$

阈值 0.5 对应夹角 60°（cos 60° = 0.5）。如果 rendered feature 和所有 codeword 的内积都低于 0.5，pixel 被判 background。

Ablation（Table II）显示 $D_{\mathrm{ID}}=1$（integer）VSQ 只 20%，$D_{\mathrm{ID}}=4$ 跳到 42.29%，$D_{\mathrm{ID}}=14$ 基本饱和到 43.66%。这数据强烈支持 vector encoding 的必要性。

---

## 两个具体 model

### FastSAM-Splat：给 image model 装上 memory

Pipeline (Fig. 2)：
1. 当前帧来，先把 3DGS memory 投影出来，得到一组 "splat segments"。
2. FastSAM 对当前帧独立预测，得到一组 "image segments"。
3. 用 Hungarian algorithm 做二部图匹配：哪几个 splat segment 对应哪几个 image segment？目标是最大化所有匹配对的 F-score 总和。这和 DETR (https://arxiv.org/abs/2005.12872) 的 set prediction 匹配思想一致。
4. 匹配结果分三类：
   - **Matched**：3DGS 有，FastSAM 也有，且配上了——保留 3DGS 的 ID（保持时序一致性）
   - **Unmatched predicted**：FastSAM 检测到，但 3DGS 没有——新 object 诞生，分配一个没用过的 codebook vector
   - **Unmatched splat**：3DGS 有，但 FastSAM 没检测到——可能是 occlusion 或 detector 漏检，magnitude 线性衰减 $C_{\mathrm{conf}}=0.1$，给个 grace period 而不是直接删

5. Fusion 完后得到 fused feature map $\hat{F}^{\mathrm{ID}}$，当作 pseudo ground truth 反向训练 3DGS，让它的 $F^{\mathrm{ID}}$ 渲染结果逼近 $\hat{F}^{\mathrm{ID}}$，loss 是 magnitude MSE + direction cosine similarity（公式 5）。

每帧做 20 步 SGD。

直觉：这是 classical Bayesian filtering 的 differentiable 实现。FastSAM 是 noisy sensor，3DGS 是 state estimate，fused result 是 posterior，gradient step 是 state update。但 representation 是 dense 3DGS 而非 Kalman 的 mean+covariance，所以信息密度高得多。

### SAM2-Splat：用 3DGS 给 SAM2 纠错

SAM2 已经有 memory，但会犯错。3DGS memory 当作 "ground truth reference" 检测三类错误：

1. **Not Tracked**：3DGS 说这里应该有 ID=3，但 SAM2 没追踪到——可能是 occlusion 后 re-appear。从 3DGS rendered region 内采 positive point，告诉 SAM2"这里有 object"。
2. **Incorrect Track**：SAM2 追踪了，但 ID 错了（把 ID=3 物体当 ID=7）。从 3DGS 的 ID=3 区域采 positive point，从 SAM2 错误分配的区域采 negative point。
3. **Duplicated Track**：SAM2 对同一 object 产生两个 track（re-appear 后当成新 object）。Positive 点标真实 region，negative 点标多余 track。

这本质上把 3DGS 变成 SAM2 的 "external prompt generator"，是一种 explicit-over-implicit 的纠错机制。SAM2-Long (https://arxiv.org/abs/2410.16268) 也解决类似问题但用 memory tree，没有 geometric prior。SAM2-Splat 用 depth+pose 做 3D grounding，更可靠地判断"是否同一 object"。

Ablation (Table III) 显示三类错误都修才有最大提升，单独修 "Duplicated Track" 提升最大（+2.02 VSQ），说明这是 SAM2 最常见的 error mode。

Click 数量 ablation (Table IV)：1-click 到 3-click 提升约 1.2，3-click 到 5-click 几乎不提升。SAM2 对 point prompt 高度敏感，少量 click 够用。

---

## 实验数据怎么读

### ScanNet-MV 主实验（Table I）

最戏剧的结果：**FastSAM-Splat 的 STQ = 38.39，超过了 SAM2 的 33.43**。

这意味着：一个 image segmentation model 加上 3DGS memory，在 temporal consistency 上打败了专门做 video segmentation 的 SOTA model。这是 paper 的核心 punchline。

具体对比：
- FastSAM (no memory) → FastSAM-Splat：VSQ +5.82，STQ +10.22
- SAM2 (implicit memory) → SAM2-Splat：VSQ +2.73，STQ +1.58

提升对 image model 更大，符合 paper 的 hypothesis：原本 memory 越少，加 explicit memory 收益越大。

### AQ vs SQ 分化

- AQ (Association Quality)：track ID 的一致性
- SQ (Segmentation Quality)：单帧 mask 的 IoU

FastSAM-Splat 的 AQ 从 19.74 跳到 32.83（+13.09），这理所当然——3DGS 天然给每个 segment 一个稳定 ID。

但 SQ 也从 40.19 涨到 44.89（+4.70），这有意思。说明 3DGS memory 不光改善 temporal consistency，还改善 single-frame mask quality。直觉：3DGS memory 是过去多帧信息的聚合，re-project 后相当于多帧信息融合，mask 边界自然更准。

### MVPd 大规模实验（Table V）

FastSAM-Splat VSQ = 56.76，比第二名高 +4.86。MVPd (https://arxiv.org/abs/2405.05560) 是作者团队自己的 dataset，180 scenes，平均每视频 94 个 object，最多 281。SAM2 在这上面跑不动（显存爆），这反向说明 3DGS memory 对 object 数量更 scalable（Gaussian 数量随 object 数线性，SAM2 memory attention 更像 quadratic）。

### 效率（Table VI）

- FastSAM-Splat：2.84 FPS（vs SAM2 的 2.83 FPS，几乎免费！）
- SAM2-Splat：1.46 FPS（慢，作者归因于 SAM2 prompt API 串行处理多 object，可工程优化）

3DGS memory 平均 28M 参数，是显著 overhead，但可以降 $D_{\mathrm{ID}}$ 压缩。

---

## 我的 take

### 1. 这个 representation design 是 transferable 的

Direction = identity, magnitude = confidence 这个 decoupling 不光适用于 3DGS。任何 dense + differentiable + discrete label + uncertainty 的场景都可以借鉴：
- Semantic SLAM 的 voxel map
- Neural map compression
- Multi-object tracking 里的 track embedding

它本质是把 discrete label 嵌入到 continuous space，让 differentiable pipeline 能处理。

### 2. "Image model + explicit memory" 可能比 "video model" 更实用

Video foundation model 训练成本巨大（SAM2 用了海量标注 video）。但 robotics 部署场景里，我们有 geometric prior 可用，与其硬训 video model，不如组合 image model + 3DGS memory。这个组合：
- 易于换 image model（domain shift 时换 backbone 即可）
- Memory 显式可查询，可解释、可纠错
- Scalability 对 object 数量更友好

### 3. Re-prompting 范式可推广

Explicit memory 给 prompt-conditioned model 自动生成 prompt——这个 idea 可推广到 Grounding DINO (https://arxiv.org/abs/2303.05499)、Florence-2、SAM2-URSA (https://arxiv.org/abs/2505.02057)。本质是 explicit representation + foundation model prompt interface = 可解释的纠错循环。

### 4. Open questions

- 没有 metric depth 和 pose 时还能用吗？monocular setting（Depth Anything V2 https://arxiv.org/abs/2406.09414 + DROID-SLAM https://arxiv.org/abs/2101.06553）能不能撑住 3DGS memory 质量？
- Memory 能否 hierarchical？目前每个 Gaussian 一个 ID，但 object 有 part/sub-object 结构（椅子有椅背、椅腿），需要 multi-granular codebook。
- 能否结合 language feature？LangSplat (https://arxiv.org/abs/2312.16008) 已经在 3DGS 上 splat CLIP feature，如果同时 splat segment ID + language feature，可以同时获得 class-agnostic segmentation 和 open-vocabulary grounding，直接对接 LLM agent。

### 5. 与 LSS 的隐秘联系

Lift-Splat-Shoot (https://arxiv.org/abs/2008.05700) 在 autonomous driving 用类似 lift-splat 做 BEV perception。本文是它在 robotics 视频分割领域的对应物，加了 segmentation memory。这两条线最终可能汇合成 general embodied perception 的统一架构。

---

## Reference 链接汇总

- 论文主页: https://topipari.com/projects/FastSAM-Splat
- SAM: https://arxiv.org/abs/2304.02643
- FastSAM: https://arxiv.org/abs/2306.12156
- SAM2: https://arxiv.org/abs/2408.00714
- SAM2-Long: https://arxiv.org/abs/2410.16268
- SAM2-URSA: https://arxiv.org/abs/2505.02057
- 3DGS (Kerbl et al.): https://arxiv.org/abs/2308.04034
- SplaTAM: https://arxiv.org/abs/2312.09403
- Gaussian Splatting SLAM (MonoGS): https://arxiv.org/abs/2312.06741
- Gaussian Grouping: https://arxiv.org/abs/2404.01858
- CoSSegGaussians: https://arxiv.org/abs/2401.05925
- LangSplat: https://arxiv.org/abs/2312.16008
- EmbodiedSAM: https://arxiv.org/abs/2408.11811
- FlashSplat: https://arxiv.org/abs/2409.08870
- SAGA: https://arxiv.org/abs/2312.00848
- Clio: https://arxiv.org/abs/2402.18835
- MVPd / FastSPAM: https://arxiv.org/abs/2405.05560
- ScanNet: http://www.scan-net.org/
- DETR: https://arxiv.org/abs/2005.12872
- FaceNet: https://arxiv.org/abs/1503.03832
- GLEE: https://arxiv.org/abs/2406.01804
- OmniSeg3D: https://arxiv.org/abs/2306.00814
- Lift-Splat-Shoot: https://arxiv.org/abs/2008.05700
- DROID-SLAM: https://arxiv.org/abs/2101.06553
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- Scaffold-GS: https://arxiv.org/abs/2312.00109
- LightGaussian: https://arxiv.org/abs/2311.17245
- Grounding DINO: https://arxiv.org/abs/2303.05499

---

# Explicit Memory through Online 3D Gaussian Splatting Improves Class-Agnostic Video Segmentation 深度解读

## 1. 核心问题与 motivation

这篇论文瞄准 robotics 领域的 class-agnostic video instance segmentation 这个 task。问题的本质可以归纳为一个判断:**当前 SOTA segmentation foundation models 在时序一致性上的瓶颈在哪里?**

作者观察到两类典型 baseline 的结构性缺陷:
- **FastSAM** (https://arxiv.org/abs/2306.12156): 逐帧独立预测,no memory。导致同一物体在不同帧获得完全不同的 segment ID,出现 "flickering" 与 inconsistency。
- **SAM2** (https://arxiv.org/abs/2408.00714): 用 recurrent neural network features 作 implicit memory。虽然能跨帧 track,但 memory 是 latent feature tensor,无法被显式查询、无法 re-project、无法纠正 "error accumulation"(尤其是 occlusion re-appear 之后)。

核心 hypothesis:**如果在 robotics 设定下还有 depth + camera pose 这种 geometric prior,我们可以把每帧的 2D segment predictions lift 到 3D,聚合后再 re-project 回 2D 作为 explicit memory**,这能显著提升 image segmentation model 的 consistency,并修复 video segmentation model 的 error accumulation。

这是一个 "lift → aggregate via 3DGS → re-project → re-prompt" 的闭环。论文 project page: https://topipari.com/projects/FastSAM-Splat

---

## 2. 3DGS 作为 memory 的 representation

每个 Gaussian primitive 参数(基于原始 3DGS https://arxiv.org/abs/2308.04034):
- $\mu \in \mathbb{R}^3$:3D 中心位置
- $q \in \mathbb{R}^4$:orientation quaternion
- $s \in \mathbb{R}^3$:各向异性 scale
- $\sigma \in [0,1]$:opacity
- $c \in \mathbb{R}^3$:color
- **新增**:$f^{\mathrm{ID}} \in \mathbb{R}^{D_{\mathrm{ID}}}$:segment ID feature vector

这里的关键设计是 segment identity 被编码为 *vector*,而不是 integer。原因直觉上很关键:3DGS rasterize 时多个 Gaussian 通过 alpha-blending 加权融合输出每个 pixel 的 value,如果 value 是 integer 没法 blend;必须是 continuous embedding,才能让反向传播的梯度流向每个 Gaussian 的 $f^{\mathrm{ID}}$。这与 NeRF 类 segmentation 工作(如 OmniSeg3D https://arxiv.org/abs/2306.00814 用 contrastive feature field)有同样的逻辑。

### Online Construction(基于 SplaTAM https://arxiv.org/abs/2312.09403)

每帧 RGB-D 输入,反投影每个 pixel 到 3D,生成新 Gaussian。各向同性初始化 scale:

$$s = \left[\frac{D}{f}, \frac{D}{f}, \frac{D}{f}\right]$$

其中 $D$ 是该 pixel 的 depth,$f$ 是 camera focal length。这个公式直觉上说的是:**远处 pixel 反投影出来的 3D 点,在 image space 上对应的 footprint 越大**,所以 Gaussian 的 scale 应该正比于 depth。这保证每个 Gaussian 在 image space 上的 footprint 都是约 1 pixel,既不会过稀疏(欠拟合几何),也不会过密(浪费参数)。

后续帧仅在两种情况下创建新 Gaussian:
1. 现有 3DGS 投影后未覆盖的 pixel
2. observed depth 与 rendered depth 差异 > 0.15m(depth disparity)— 用于发现新几何或几何更新

这是一种 *greedy incremental densification*,不做 global bundle adjustment-style 优化,因此 online friendly,但牺牲了全局一致性。

---

## 3. Segment ID Codebook 的设计哲学

这是论文最值得玩味的部分。定义 codebook:

$$C = \{c_1, \ldots, c_N\} \subset \mathbb{R}^{D_{\mathrm{ID}}}$$

其中 $N$ 是最大 segment 数,$D_{\mathrm{ID}}$ 是 embedding 维度。codebook 通过如下 contrastive loss 优化(公式 1):

$$L = -\min_{1 \leq i \leq N}\left(\min_{1 \leq j \leq N, j \neq i} \|c_i - c_j\|\right)$$

这是一个 **max-min sphere packing** 目标:最大化 *最近的* 两个 codeword 之间的距离。这与 Tammes problem(球面上均匀分布 N 个点)数学等价,本质是把 N 个 ID 在 $D_{\mathrm{ID}}$ 维单位球上"撑开",类似 FaceNet / ArcFace 的 angular margin 思想,但目标更激进 —— 要求所有 codeword 之间 *最小* 距离尽量大。

**重要 decoupling:direction = identity, magnitude = confidence**

所有 codeword 归一化为 unit norm。这意味着:
- 向量方向编码 segment ID(类 one-hot 但 continuous)
- 向量 magnitude 编码 model confidence

这种 decoupling 让 rasterize 时 alpha-blending 有明确语义:blended direction 表示 "融合的 ID prediction",blended magnitude 表示 "融合的 confidence"。如果某区域有 disagreement,blending 后 magnitude 自然下降,后续可以用 magnitude < threshold 来 reject pixel。这是个很优雅的设计。

### Decoding(公式 2-3)

$$m_{x,y} = \arg\max_i \left\{ d(c_i, F^{\mathrm{ID}}_{x,y}) \right\}$$

$$d(c_i, F) = \langle c_i, F \rangle \cdot \mathbf{1}_{\langle c_i, F \rangle > 0.5}$$

- $m_{x,y}$:pixel $(x,y)$ 处的 segment ID(整数索引到 codebook)
- $F^{\mathrm{ID}}_{x,y}$:rendered feature vector at pixel $(x,y)$
- $\langle \cdot, \cdot \rangle$:inner product
- 阈值 0.5:如果与所有 codeword 相似度都低于 0.5,该 pixel 被判为 background / unassigned

阈值 0.5 对应约 60° 的 angular separation(cos 60° = 0.5)。这意味着 codebook 的设计必须保证 codeword 之间夹角至少 > 120° 才能在 decoding 时不混淆,与 max-min 优化目标完全吻合。

---

## 4. FastSAM-Splat:从 image model 到 video model

FastSAM 本身是 image-level model,逐帧独立分割,没有任何 temporal memory。FastSAM-Splat 的设计思路:

### Pipeline(Fig. 2)

1. **3DGS render**:把当前 3DGS memory 投影到当前 viewpoint,得到 2D rendered segments(每个 pixel 一个 codebook index)。
2. **FastSAM predict**:对当前 RGB 帧独立预测 2D segments。
3. **Hungarian matching**:把两组 segments 做 bipartite matching,目标函数是最大化所有 matched pair 的 F-score 之和。这类似 DETR(https://arxiv.org/abs/2005.12872)的 set prediction matching,但用 F-score 而非 IoU loss。
4. **Categorize**:matched / unmatched predicted / unmatched splat 三类。
5. **Fusion**:
   - Matched:保留 3DGS 的 codebook ID
   - Unmatched predicted:随机分配未用过的 codebook vector(新 object instance 诞生)
   - Unmatched splat:linear decay magnitude(公式 4):
   
   $$f^{\mathrm{ID}'} = \frac{f^{\mathrm{ID}}}{\|f^{\mathrm{ID}}\|} \cdot \left(\|f^{\mathrm{ID}}\| - C_{\mathrm{conf}}\right)$$
   
   其中 $C_{\mathrm{conf}} = 0.1$。这个 "confident decrement" 的直觉是:**如果一个 3DGS 段在当前帧没被 detection 匹配上,可能是 (a) 物体被遮挡 (b) 物体暂时被 detector 漏检 (c) 真的消失了**。线性 decay 比 hard deletion 更 robust,给一个 grace period,体现 Bayesian filtering 的精神(类似 Kalman filter 的 prediction update)。

6. **3DGS optimization**(公式 5):

$$L = \lambda_{\mathrm{mag}} \mathrm{MSE}(\|\hat{F}^{\mathrm{ID}}\|, \|F^{\mathrm{ID}}\|) + \lambda_{\mathrm{dir}} (1 - S_C(\hat{F}^{\mathrm{ID}}, F^{\mathrm{ID}}))$$

- $\hat{F}^{\mathrm{ID}}$:fused objective feature map(从 fusion step 得到)
- $F^{\mathrm{ID}}$:rendered 3DGS feature map
- $S_C$:cosine similarity
- $\lambda_{\mathrm{mag}} = 50.0$, $\lambda_{\mathrm{dir}} = 1.0$:magnitude loss 权重远大于 direction,暗示作者认为 confidence 校准比 ID 校准更重要(或更不稳定,需要更大梯度)

每帧做 $N_{\mathrm{opt}} = 20$ 步 SGD。这相当于把每帧的 fused prediction 当作 pseudo ground truth 反向训练 3DGS,让 3DGS "记住" 当前 fused 估计。

### Intuition

整个 FastSAM-Splat 可以视为一种 **3D-space 非参数化 tracker**:每帧的 FastSAM 预测是 observation,3DGS 是 state,融合 step 是 measurement update,3DGS optimization 是 state update。这本质是把 classical multi-object tracking 的 Bayesian filter 思想用 differentiable 3D representation 实现。与 FastSPAM(https://arxiv.org/abs/2405.05560,作者前作)用 sparse centroid tracking 相比,这里用 dense 3DGS 作为 state,信息密度高得多。

---

## 5. SAM2-Splat:用 3DGS memory 做 re-prompting

SAM2 已经有 implicit memory,但作者观察到三类典型 failure:

1. **Not Tracked**:3DGS 期望某 object 但 SAM2 漏 track(occlusion 后 re-appear 时常见)
2. **Incorrect Track**:SAM2 跟踪了但 ID 错误(把 A 物体当 B 物体)
3. **Duplicated Track**:SAM2 对同一物体生成多个重复 track(常见于 view 变化大时)

SAM2-Splat 的核心 idea:用 3DGS memory 作 "ground truth reference" 检测这三类错误,然后用 **point prompts** 重新告诉 SAM2 "这里应该有 object / 这块应该属于 ID X / 这两个是同一个 object"。

### Re-prompting 设计

对每类错误,从相关 image region 采样 positive / negative point:
- Not Tracked:从 3DGS rendered segment 内部采 positive point,告诉 SAM2 "这里有 object"
- Incorrect Track:从 3DGS 该 ID 的 rendered region 采 positive point,从 SAM2 错误分配的区域采 negative point,告诉 SAM2 "这块是 X,那块不是 X"
- Duplicated Track:把重复 track 对应的 3DGS region 用 positive point 标记,把多余的 SAM2 track 用 negative point 标记

这与 SAM2-Long(https://arxiv.org/abs/2410.16268)用 memory tree 解决 error accumulation 的方向类似,但 SAM2-Long 没用 geometric prior,SAM2-Splat 利用 depth + pose 做 3D grounding,理论上能更准确地判断 "是否同一 object"。

### Click 数量 ablation(Table IV)

| Backbone | 1-click | 3-click | 5-click |
|----------|---------|---------|---------|
| YOLOv8 | 42.95 / 34.03 | 43.86 / 34.99 | 43.76 / 35.01 |
| GT | 54.89 / 37.29 | 56.10 / 38.72 | 56.42 / 39.19 |

3-click 到 5-click 的提升仅 0.10 / 0.02,基本饱和。1-click 到 3-click 提升约 1.2 / 1.4。这说明 SAM2 对单点 prompt 已经高度敏感(因为 SAM2 本身被训练为 prompt-conditioned),更多 click 收益递减。这与原始 SAM(https://arxiv.org/abs/2304.02643)的 observation 类似 —— 一个粗略的中心点 prompt 就能获得大部分 IoU。

### Re-prompt 类别 ablation(Table III)

| 配置 | VSQ / STQ |
|------|-----------|
| Baseline SAM2 | 41.03 / 33.43 |
| Only Not Tracked | 41.09 / 34.19 |
| Only Incorrect Track | 41.73 / 33.15 |
| Only Duplicated Track | 43.05 / 33.78 |
| All three | 43.76 / 35.01 |

有趣的是 "Duplicated Track" 单独就贡献了 +2.02 VSQ,而 "Incorrect Track" 单独甚至略微降了 STQ。Intuition:SAM2 在 re-appear 场景容易产生 duplicate ID(因为它把 re-appear 的物体当新 object),这是最常见 error。Incorrect Track 单独不提升可能因为它依赖 Not Tracked 检测作为前置 —— 你需要先 detect 到"应该有 ID X"才能判断"现在 SAM2 给的不是 X"。

---

## 6. Segment ID Feature 维度 ablation(Table II)

| $D_{\mathrm{ID}}$ | VSQ / STQ |
|------|-----------|
| 1 (integer) | 20.00 / 15.47 |
| 4 | 42.29 / 36.97 |
| 7 | 43.08 / 37.93 |
| 14 | 43.66 / 38.38 |
| 28 | 43.74 / 38.39 |

**关键观察**:
- $D_{\mathrm{ID}} = 1$(integer 编码)灾难性失败,VSQ 仅 20%。
- $D_{\mathrm{ID}} = 4$ 到 $D_{\mathrm{ID}} = 14$ 之间收益从 +22.29% 砍到 +1.37%,**14 维之后基本饱和**。

Intuition: $D_{\mathrm{ID}} = 1$ 失败本质是 rasterize 时无法 blend 标量(只能加权平均得到一个数,但这个数没有 ID 语义)。从 $D_{\mathrm{ID}} = 2$ 开始,vector 才有 direction 概念。$D_{\mathrm{ID}} = 14$ 饱和对应的是:典型场景下 object 数量 N 大约在 10-100 范围,要在 $D_{\mathrm{ID}} = 14$ 单位球上均匀分布 100 个点,平均最近邻距离角约 $\arccos(1 - 2 \cdot 100^{-1/13}) \approx 70°$ 左右,与 decoding 阈值 0.5(60°)之间还有 buffer,基本足够区分。再加大维度收益边际化 —— 这是经典的 dimensionality sufficiency 现象。

---

## 7. ScanNet-MV 主实验(Table I)

完整对比:

| Method | Backbone | VSQ | STQ | AQ | SQ |
|--------|----------|-----|-----|-----|-----|
| FastSAM | YOLOv8 | 37.92 | 28.17 | 19.74 | 40.19 |
| FastSPAM | YOLOv8 | 41.18 | 30.33 | 21.86 | 42.09 |
| SAM2 | YOLOv8 | 41.03 | 33.43 | 27.47 | 40.68 |
| SAM2 | GT | 52.42 | 33.83 | 35.96 | 31.82 |
| **FastSAM-Splat** | YOLOv8 | **43.74** | **38.39** | 32.83 | 44.89 |
| **SAM2-Splat** | YOLOv8 | **43.76** | **35.01** | 30.68 | 39.94 |
| **SAM2-Splat** | GT | **56.42** | **39.19** | 41.99 | 36.58 |

关键解读:

### 7.1 FastSAM-Splat 的跨越

VSQ 提升 **+5.82**(37.92 → 43.74),STQ 提升 **+10.22**(28.17 → 38.39)。这非常戏剧化 —— 一个 image model 加上 3DGS memory 后 STQ 超过了 SAM2(33.43)!这强烈支持了核心 hypothesis:**explicit 3D memory 能让 image segmentation model 在 consistency 上超越 video segmentation model**。

为什么 AQ 提升巨大(19.74 → 32.83, +13.09)?因为 3DGS memory 天然提供 track ID —— 只要每个 Gaussian 在 fusion 时被分配了 codebook vector,re-project 后就自然有 ID consistency。而 FastSAM 原生没有 ID 概念,AQ 几乎是随机匹配得分。

### 7.2 SAM2-Splat 的提升更小但一致

SAM2(YOLOv8) → SAM2-Splat(YOLOv8):VSQ +2.73, STQ +1.58。

注意 SAM2 已经有 implicit memory,3DGS 是补充而非替代。提升小,但 GT prompt 的设定下 SAM2-Splat 提升 +4.00 VSQ / +5.36 STQ 更显著,说明 **detector quality 是 SAM2-Splat 的瓶颈**:YOLOv8 漏检的 object,3DGS 也无从 render 提示。

### 7.3 AQ vs SQ 的分化

FastSAM-Splat 的 SQ 从 40.19 提升到 44.89,说明 3DGS memory 不仅提升 temporal consistency(AQ),还提升 *单帧 mask 质量*(SQ)。直觉:re-projected 3DGS memory 提供 "上几帧这个 object 长这样" 的 dense prior,等于多帧信息融合,mask 边界自然更准。

但 SAM2(GT) baseline 的 SQ 反而只有 31.82,远低于 FastSAM 的 40.19。这是因为 SAM2(GT)用 GT 起始 prompt 但后续靠 memory propagate,容易在长视频中 drift 出 boundary,降低 SQ。SAM2-Splat(GT)把 SQ 提回 36.58,但仍不及 FastSAM-Splat 的 44.89 —— 说明 FastSAM 每帧独立 detection 提供的 mask boundary 本身就比 SAM2 的 propagate mask 更准,3DGS memory 只是把它们对齐。

### 7.4 两个 hypothesis 的验证

论文提出两个 hypothesis:
1. explicit 3D memory 提升 image & video segmentation model 一致性 —— **验证**
2. 提升对 image model 更显著(因为 image model 原本 no memory) —— **验证**

FastSAM-Splat 比 SAM2-Splat(YOLOv8)的相对提升大 +3.09 VSQ / +8.64 STQ。这是 paper 的 story 顶梁柱 —— "memory 越少,加 memory 收益越大"。

---

## 8. MVPd 大规模仿真实验(Table V)

| Method | Backbone | VSQ |
|--------|----------|-----|
| Video K-Net | Swin-base | 42.52 |
| FastSAM | YOLOv8 | 49.17 |
| FastSPAM | YOLOv8 | 51.90 |
| **FastSAM-Splat** | YOLOv8 | **56.76** |

MVPd(https://arxiv.org/abs/2405.05560,作者前作)是作者团队自己造的 dataset,180 scenes × ~94 objects/video × 300-600 frames。FastSAM-Splat 在这个 large-scale benchmark 上仍能提升 +4.86 VSQ。

注意 SAM2 / SAM2-Splat 没在 MVPd 上跑 —— 论文解释是 SAM2 对多 object 时显存爆。这暴露了 SAM2 在 100+ objects 时的 scalability 问题,反向证明 FastSAM-Splat 的 memory 机制对 object 数量更友好(3DGS 是 distributed representation,与 object 数线性,而 SAM2 的 memory attention 是 quadratic-like)。

---

## 9. 效率分析(Table VI)

| Model | FPS | Params (M) |
|-------|-----|-----------|
| FastSAM | 64.57 | 72 |
| FastSPAM | 6.60 | 72 |
| SAM2 | 2.83 | 81 |
| **FastSAM-Splat** | **2.84** | 72 + 28 |
| **SAM2-Splat** | **1.46** | 81 + 26 |

观察:
1. **FastSAM-Splat 只比 SAM2 慢 0.01 FPS**(2.84 vs 2.83),cost 几乎免费!这是因为 FastSAM backbone 本身很快,3DGS render + 20 步 SGD 在 GPU 上 wall time 很短。
2. **SAM2-Splat 慢到 1.46 FPS**,作者归因于 SAM2 prompt API 串行处理多 object。如果能 batch prompt,有望恢复到接近 SAM2 的 2.83 FPS。这是个工程优化方向,与 algorithm 无关。
3. **3DGS memory 参数量**:ScanNet-MV 上平均 28M / 26M parameters。这是个相当大的 overhead,且与视频长度 / 场景规模成正比。长视频会无限增长。作者在 Sec. IV-B 指出可以降 $D_{\mathrm{ID}}$ 来压缩。

---

## 10. 局限性与未来方向

作者自陈的 limitations:
1. **依赖 input pose + depth**:这要求 RGB-D camera + odometry/SLAM,在 monocular setting 不可用。但可以用 monocular depth(如 Depth Anything V2 https://arxiv.org/abs/2406.09414) + visual odometry(如 DROID-SLAM https://arxiv.org/abs/2101.06553)替代,虽然 metric accuracy 会下降。
2. **Lack of global optimization**:3DGS 只做 per-frame local optimization,可能漂移。可以引入 global bundle adjustment 或定期 refinement(如 SplaTAM 的 global refinement)。
3. **3DGS storage 优化**:可借鉴 Scaffold-GS(https://arxiv.org/abs/2312.00109)的 anchor-based 稀疏化、或 LightGaussian(https://arxiv.org/abs/2311.17245)的 pruning + quantization,大幅降 memory。

我额外想到的几个方向:

- **Language features 也可以 splat**:LangSplat(https://arxiv.org/abs/2312.16008)已经在 3DGS 上 splat CLIP feature,如果 FastSAM-Splat 同时 splat language feature + segment ID,可以同时获得 class-agnostic segmentation 与 open-vocabulary grounding,直接对接 LLM agent。
- **Re-prompting 可以扩展到 box / mask prompt**:目前只用 point prompt,但 SAM2 支持 box 和 mask prompt。3DGS re-projected 的 binary mask 可以直接作为 mask prompt,可能比 point prompt 更准。
- **Active re-prompting**:用 3DGS 不一致性来触发 robot 的 active perception —— 让机器人主动移动相机去重新观察 ambiguous segment。这与 active SLAM 思想(https://arxiv.org/abs/2310.18364)相通。
- **3DGS + Semantic Uncertainty**:目前 segment ID 用 single point estimate,可以用 probabilistic embedding(如 VAE-style codebook)建模 ID uncertainty,在 re-prompt 决策时考虑 confidence。
- **与 SAM2.1 / SAM2-URSA 类 newer memory mechanism 结合**:近期 SAM2 改进工作(如 SAM2-URSA,https://arxiv.org/abs/2505.02057)用 unified memory,SAM2-Splat 思路可移植。
- **Lift-splat pipeline 与 LSS 的联系**:Lift-Splat-Shoot(https://arxiv.org/abs/2008.05700)在 autonomous driving 用类似 lift-splat 做 BEV,本质上这里是它的 robotics 视角对应物,且加了 segmentation memory。

---

## 11. 与相关工作的关联图

```
                Memory Memory Tree (SAM2-Long, NeurIPS'24)
                            │
                            │ video object segmentation
                            ▼
                       SAM2 (ECCV'24)
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
         FastSPAM'24    SAM2-Splat   EmbodiedSAM'24
         (sparse)     (this paper,  (online 3D
                       dense 3DGS)   segmentation)
                            ▲
                            │ dense 3D memory
                            │
              ┌─────────────┴─────────────┐
              │                           │
        SplaTAM (CVPR'24)         Gaussian Grouping (ECCV'24)
        MonoGS (CVPR'24)         SAGA (AAAI'25)
        (online 3DGS SLAM)       (offline 3DGS segmentation)
```

- **vs Gaussian Grouping**(https://arxiv.org/abs/2404.01858):offline 重建后再 segmentation,加 identity embedding。本文 online 做,且用 codebook vector 而非 learnable embedding。
- **vs FlashSplat**(https://arxiv.org/abs/2409.08870):用 linear programming 分配 segment ID,要求固定 3DGS。本文 incremental + 不同iable assignment。
- **vs CoSSegGaussians**(https://arxiv.org/abs/2401.05925):训练 decoder 输出 instance segment,本文直接用 codebook lookup,不需 decoder。
- **vs EmbodiedSAM**(https://arxiv.org/abs/2408.11811):同样是 online 3D segmentation,但用 implicit feature field 而非显式 Gaussian ID codebook。
- **vs GLEE**(https://arxiv.org/abs/2406.01804):bipartite feature matching 扩展 image model 到 video。本文走 3DGS memory 路线。
- **vs Clio**(https://arxiv.org/abs/2402.18835):task-driven 3D scene graph,本文 motivation 来源之一。

---

## 12. 我的总体 intuition 总结

1. **Codebook direction-magnitude decoupling 是 paper 最美妙的 design choice**。它让 alpha-blending 有明确 semantic 意义(direction 不会因 blending 而"平均",反而通过 magnitude weight 自动得到加权方向),并支持 graceful confidence decay。这种 representation design 值得借鉴到其他需要"dense + discrete label + uncertainty"的场景(如 semantic SLAM、neural map compression)。

2. **3D memory 对 image model 的提升远大于 video model**,这是个有重要工程意义的结果。意味着我们可能不需要训练昂贵的 video segmentation foundation model —— 在 robotics 场景下,把 image model + 几何 memory 组合可能 cost-effective 得多,且更易扩展到新 domain(domain shift 时 image model 更易替换)。

3. **Re-prompting 是 explicit memory 的天然应用**。SAM2 之类 model 有强大的 prompt interface,explicit memory 提供了"自动生成 prompt"的信号源。这个范式可推广到其他 prompt-conditioned foundation model(如 Grounding DINO、Florence-2)。

4. **Reconstruction 与 perception 的融合方向**:这篇 paper 把 3DGS 从 "scene reconstruction" 工具转变为 "perception memory substrate"。未来的 robotics perception 系统可能更多采用这种 "memory-first" 架构,而 3DGS 之所以合适,是因为它 (a) differentiable,(b) explicit,(c) 可 render 到任意 viewpoint,(d) 增量构建。NeRF 缺 (c) 的速度优势,vector-quantized map 缺 (a) 的可优化性。

5. **Open question**: 这个工作清晰展示了 explicit memory 在 robotics segmentation 上的价值,但留下两个核心问题未解:(a) 在没有 metric depth / pose 时能否推广?(monocular setting 是消费级 AR/VR 的实际需求);(b) memory 能否做 hierarchical / compositional(目前每个 Gaussian 一个 segment ID,但 object 可能有 part/sub-object 结构,需要 multi-granular memory)。

参考链接整理:
- Paper project page: https://topipari.com/projects/FastSAM-Splat
- FastSAM: https://arxiv.org/abs/2306.12156
- SAM: https://arxiv.org/abs/2304.02643
- SAM2: https://arxiv.org/abs/2408.00714
- SAM2-Long: https://arxiv.org/abs/2410.16268
- SplaTAM: https://arxiv.org/abs/2312.09403
- 3DGS (Kerbl et al.): https://arxiv.org/abs/2308.04034
- MVPd / FastSPAM: https://arxiv.org/abs/2405.05560
- LangSplat: https://arxiv.org/abs/2312.16008
- Gaussian Grouping: https://arxiv.org/abs/2404.01858
- CoSSegGaussians: https://arxiv.org/abs/2401.05925
- EmbodiedSAM: https://arxiv.org/abs/2408.11811
- Clio: https://arxiv.org/abs/2402.18835
- ScanNet: http://www.scan-net.org/
- GLEE: https://arxiv.org/abs/2406.01804
- Scaffold-GS: https://arxiv.org/abs/2312.00109
- LightGaussian: https://arxiv.org/abs/2311.17245
- Lift-Splat-Shoot: https://arxiv.org/abs/2008.05700
- OmniSeg3D: https://arxiv.org/abs/2306.00814
- DETR: https://arxiv.org/abs/2005.12872
- DROID-SLAM: https://arxiv.org/abs/2101.06553
