---
source_pdf: Generate, Transfer, Adapt Learning Functional Dexterous Grasping from
  a Single Human Demonstration.pdf
paper_sha256: 52b5d0856cdd2f4a44d8b3b2421d646c0bbab66dfc182324fe9549d5dc18dbc6
processed_at: '2026-08-04T13:41:05-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 CorDex

## 一句话说清楚这 paper 在干嘛

你拿手机拍一段自己抓锤子的 video，CorDex 就能自动造出 1100 万条训练数据，训出一个模型，让 robot hand 抓住各种没见过的锤子——不仅抓稳，食指还精准按在 trigger 上。

---

## 为什么这件事 hard

普通 grasping 的目标就一个字：**稳**。抓住别掉就行。

Functional grasping 多了一个要求：**对**。你得抓对地方。

举个 hammer 的例子。你抓 hammer，大拇指和另外几根手指要稳住 handle（这是 stability），食指得精准搭在 trigger 上（这是 functionality）。trigger 可能就指甲盖那么大一块区域——在 partial point cloud 里可能就 5-10 个点。模型稍微 sample 偏一点，食指就搭在 trigger 旁边了，抓得再稳也没用，因为按不下去。

所以 functional grasping = stability + semantic placement。前者是几何问题，后者是语义问题。之前的工作基本只解了前者。

---

## 之前的人怎么做的，为什么不行

### 路线 A：teleoperation / MoCap 收数据
RealDex、OakInk 这类。让人戴 glove 或者 teleoperation 几千次。

问题：太贵。你抓 9 类工具 × 100 个 object × 10 种 grasp = 9000 次 teleoperation，每次还要精确到毫米级。PhD student 抓到怀疑人生。

### 路线 B：从 web video 学
Deft、Web2grasp。去 YouTube 找人用工具的视频，reconstruct 出 hand-object pose。

问题：web video 质量参差不齐，reconstruction noise 一塌糊涂。一个 blurry 的 video 重建出来手指位置能差 2 cm——而 functional tolerance 是 1 mm。你得花大量人工去 curate。

### 路线 C：3D correspondence transfer
SparseDFF、DenseMatcher。给一个 reference object 上的 grasp，用 3D dense correspondence 找到新 object 上对应的位置，transfer 过去。

问题：3D correspondence 模型本身就没见过多少数据，跨 instance 泛化很差。你训练集里有个红色 hammer，测试来个蓝色异形 hammer，3D matcher 就懵了。

实验数据说话：DenseMatcher 在 Inspire Hand 上只有 **7.6%** success rate。比瞎猜好不到哪去。

### 路线 D：category-level pose estimation
AG-Pose 这类。估一个 6D pose，把 reference grasp 变换过去。

问题：coarse alignment。pose estimation 误差几毫米到几厘米，trigger 那么小的区域根本对不准。实验里 AG-Pose 用 CorDex 的数据训，也才 48.9%。

---

## CorDex 的三个 Aha Moment

### Aha 1：别在 3D 里做 correspondence，降维到 2D

这是整篇 paper 最大的 insight。

3D correspondence 为什么不行？因为训练数据少。整个 3D correspondence 领域的数据量可能就几十万 scan，跟 internet-scale 的 2D image data 不在一个量级。

但 2D image matching 不一样。DINOv2、MatchAnything 这些模型见过几亿张 internet image，跨 category、跨 appearance、跨 illumination 都 robust。你给它看一张人抓锤子的 photo 和一张新锤子的 rendering，它能 reliably 找到 "trigger 在哪" 这个 pixel-level correspondence。

所以 CorDex 的 pipeline 是：

1. 从 demo video reconstruct 出 3D hand mesh + object point cloud，提取 fingertip contact 点
2. 把 contact 点投影到 demo 的每一帧 2D image 上
3. 渲染新 object 的多视角 2D image
4. 用 2D matcher（MatchAnything / EfficientLoFTR）匹配 demo frame 和新 object rendering
5. 匹配上的 pixel 用 camera 参数反投影回 3D
6. 多个 view 反投影回来的点做 density-based clustering，取 top-3 cluster center

**关键细节**：保留 top-3 cluster centers，不是只取 1 个。因为 2D matching 本身有 ambiguity，硬选一个会丢信息。保留 3 个 hypothesis，让下游 physics optimization 自己挑最合理的。

Ablation 数据：换成 3D matching → 从 72.7% 掉到 18.5%。去掉 multiple candidates → 从 72.7% 掉到 66.1%。两个设计都关键。

### Aha 2：Physics optimization 当 filter

现在新 object 上有一堆 candidate contact points。但它们只是 "大概在这附近" 的 hint，不是精确到毫米的 grasp label。还得变成 robot hand 的 $(T, \theta)$——wrist pose + joint angles。

这里有几个 trap：

**Trap 1：scale mismatch**。Demo 里的 hammer handle 粗 3 cm，generated hammer 可能粗 5 cm。Transferred contact 点之间的间距跟 robot hand finger span 对不上。

**Solution**：不只在 fingertip 上定义 contact point，在 middle phalanx 和 distal phalanx 上都放 candidate。给 optimization 更多 freedom 去够到 transferred points。

**Trap 2**：transferred points 可能 unreachable。Object 太大，robot hand 张不开那么宽。

**Solution**：除了 contact-prior loss（让 hand 贴近 transferred points），还加 stability contact loss——让 hand points 贴近 object surface 上最近的点。意思是 "如果 transferred points 够不到，至少贴住 object 表面别悬空"。

Loss 公式回顾：

$$\mathcal{L}_{prior} = \sum_{l \in \mathcal{C}} \left( \| h_l(g) - o_p \|_2^2 + \alpha (1 - n_h^\top n_o) \right)$$

- $h_l(g)$：grasp $g$ 下 link $l$ 的 contact point 位置
- $o_p$：transferred prior point
- $n_h, n_o$：surface normal
- $\alpha$：position vs normal alignment 的权重

这个 normal alignment term 很 clever。光位置对不够，finger 接触 object 的角度也得对，否则 force 方向不对，抓不牢。$1 - n_h^\top n_o$ 就是两个 normal 的 cosine distance。

优化完之后还有一道 gate：IsaacGym 里用 6 个方向 external force 推 object，displacement < 2 cm 才算 stable，保留进 dataset。这一步把物理上不靠谱的 grasp 全 filter 掉了。

**Intuition**：2D matching 给的是 "软 hint"（有噪声、有 ambiguity），physics optimization 是 "硬约束"（必须稳定、必须不穿模、joint limit 不能超）。两者结合 = 先用弱信号 explore，再用强约束 refine。这比纯 correspondence transfer 或者纯 optimization 都靠谱。

### Aha 3：模型要同时看 RGB 和 depth，还要 smart sampling

有了数据，训模型。基于 D(R, O) 的 CVAE 框架，但加了两个东西。

**为什么必须看 RGB**：

光看 point cloud，你能知道 object 的几何形状，但你不知道哪个部分是 trigger。一个 spray bottle 的 point cloud——handle 和 trigger 在几何上都是圆柱体的一部分，长得很像。但 RGB image 里 trigger 是红色的，handle 是白色的，semantic 一下就区分开了。

Ablation：去掉 image input，从 72.7% 掉到 20.7%。掉了 52 个点。这是最大的 single ablation drop，说明 semantic information 是 functional grasping 的命脉。

**Importance-aware sampling**：

Uniform sample 4096 个点过 transformer 太贵，而且 trigger 那种小区域只占几十个点，信号被淹没。

CorDex 训了一个 lightweight transformer 估计每个 point 的 importance probability，按 probability 做 non-uniform downsampling 到 1024 点。contact 区域 density 提升，其他区域 sparse。

GT importance map 怎么来？object points 到 GT robot hand points 的距离，距离越近 importance 越高，softmax 成分布。训练用 KL divergence 让 predicted distribution match GT。

这本质是 **learned attention**——模型自己学 "哪里重要"，然后把计算 budget 投过去。

**Local-global fusion**：

两个 receptive field 并行：

- Local cross-attention + adaptive radius：捕捉 trigger 的细节。点稀疏的地方 radius 大一点 gather 更多 context，点密集的地方 radius 小一点 focus detail
- Global self-attention：捕捉整个 object 的 shape context

两者 fusion 后 cross-attend robot hand feature → predict distance matrix。

Ablation：去掉 local attention 从 72.7% 掉到 52.7%，掉 20 个点。Local detail 对 functional region 识别至关重要。

---

## 数据引擎的产出

跑 48 张 A100 × 3 天：

- 9 个 category（Drill, Pipette, Stapler, Spray Bottle, Hammer, Syringe, Hair Dryer, Aerosol Can, Glue Gun）
- 900 个 object（每 category 100 个，用 Rodin 从 internet image 生成 3D）
- 108 万张 RGB-D image
- 1100 万 image-grasp pair
- 2 个 embodiment（Shadow Hand 22-DoF + Inspire Hand 6-DoF）

对比一下：OakInk 大概 10 万 grasp，DexGraspNet 大概 130 万 grasp 但都是 stable grasp 没 functional。CorDex 的 1100 万 pair 是 **functional + stable 双重验证** 的高质量 label。

---

## 结果有多强

### Simulation（Shadow Hand）

| Method | Avg Success Rate |
|---|---|
| D(R,O) 原版 | 18.3% |
| D(R,O) + CorDex data | 36.0% |
| SparseDFF（one-shot，给完整 model） | 14.8% |
| DenseMatcher（one-shot，给完整 model） | 16.9% |
| AG-Pose + CorDex data | 67.5% |
| **CorDex** | **88.5%** |

注意 SparseDFF 和 DenseMatcher 给的是 **完整 3D model 或 two-view**，比 CorDex 的 single-view 输入信息更多，但效果反而差一个量级。这说明 3D correspondence 方法本身就不行，跟输入 completeness 无关。

### Real-world（OYMotion hand + Franka FR3）

6 category × 3 object × 5 pose = 90 trials：

| Method | Success |
|---|---|
| D(R,O) + CorDex data | 13/90 (14%) |
| SparseDFF | 11/90 (12%) |
| DenseMatcher | 6/90 (7%) |
| AG-Pose + CorDex data | 27/90 (30%) |
| **CorDex** | **62/90 (69%)** |

69% real-world functional grasp success。在这个领域是非常强的数字。Pipette 最弱（7/15），因为细长物体在 partial view 下几何 ambiguity 大。

---

## 这 paper 真正的 contribution 是什么

表面上看是 "一个新的 grasping framework"。但我觉得真正有意思的是方法论层面的 insight：

**Insight 1：2D matching > 3D matching for cross-instance transfer**

这个 insight 可能超越 grasping 领域。任何需要跨 instance transfer 3D information 的任务——manipulation、articulated object reasoning、scene understanding——都可能受益。核心 logic 是：2D pretrained model 见过海量数据，泛化能力强；3D 模型数据稀缺，泛化差。把 3D transfer 拆成 2D match + geometric back-projection + 3D aggregate，是降维打击。

**Insight 2：保留 multiple hypotheses，让下游 optimization resolve**

2D matching 有 noise，不要硬选一个 "best match"。保留 top-3 cluster centers，给 physics optimization 留 explore space。这跟 diffusion model sample 多个 candidate 再 pick best 的思路类似——ambiguous 的问题不要在早期 commit。

**Insight 3：Data quality > Data quantity，但 quality 要靠 physics gate 保证**

1100 万 pair 里，真正通过 IsaacGym 6-direction force test 的才保留。这个 filter 是 data quality 的关键。没有这一步，transferred contact 的 noise 会直接污染模型。

**Insight 4：Functional grasping 必须 multimodal**

纯 point cloud 不行，因为 functional region 是 semantic 概念不是 geometric 概念。RGB 提供 semantic，depth 提供 geometric，两者缺一不可。这个 ablation（掉 52 个点）是铁证。

---

## 我的额外思考

**这能 scale 到 open-set 吗？**

现在每个 category 要一段 demo + 训一个 model。9 个 category = 9 个 model。如果要做到 1000 个 category，这个范式就不 work 了——你不可能训 1000 个 model。

可能的方向：
- Multi-task training：所有 category 一起训，share backbone，task-specific head
- In-context learning：给 model 看一段新 category 的 demo 当 context，直接 inference 不 fine-tune
- Foundation model for grasping：类似 LLM 的思路，海量 multi-category data 预训练，emergent generalization

**Tactile 呢？**

Functional grasping 的终极验证是 "trigger 真的按下去了"。现在完全靠 vision，没有 tactile feedback。Trigger 按下去那一下的 force resistance 是一个很关键的 signal。未来可能要加 tactile sensor（GelSight 之类）+ tactile simulation（TACTO）做 sim-to-real。

**Downstream tool use？**

Grasp 只是第一步。抓完 hammer 之后要敲钉子，抓完 drill 之后要钻孔。CorDex 输出的 functional grasp 能不能直接 plug 进一个 manipulation policy？这个 policy 需不需要也从 demo 学？还是可以 RL？这些都是 open question。

---

## Web Links

- **CorDex project page**: https://cordex-manipulation.github.io
- **D(R,O) Grasp (predecessor)**: https://d-ro.github.io/
- **DINOv2 (semantic feature)**: https://github.com/facebookresearch/dinov2
- **MatchAnything (2D matcher)**: https://github.com/xingyihe/MatchAnything
- **EfficientLoFTR (2D matcher)**: https://github.com/zju3dv/EfficientLoFTR
- **WiLoR (hand reconstruction)**: https://github.com/Willow-WLR/WiLoR
- **VGGT (3D reconstruction)**: https://github.com/facebookresearch/vggt
- **Rodin / Hyper3D (2D-to-3D generation)**: https://hyper3d.ai/
- **IsaacGym (physics simulation)**: https://developer.nvidia.com/isaac-gym
- **SparseDFF (baseline)**: https://github.com/j96w/SparseDFF
- **DenseMatcher (baseline)**: https://github.com/Ju-Zhu/DenseMatcher
- **AG-Pose (baseline)**: https://github.com/linjiajia0707/AG-Pose
- **DexGraspNet (related dataset)**: https://github.com/PKU-EPIC/DexGraspNet
- **OakInk (hand-object dataset)**: https://github.com/ll4lab/oakink
- **Grounded SAM (segmentation)**: https://github.com/IDEA-Research/Grounded-Segment-Anything
- **DexMimicGen (related data gen)**: https://github.com/AGI-EdgerSimulator/DexMimicGen
- **Blender (rendering)**: https://www.blender.org/

---

# CorDex：从单个 Human Demo 学习 Functional Dexterous Grasping

这篇 paper 来自 Xingyi He、Kuan Fang 等人（西湖大学 + Stanford/CMU 方向的合作），核心解决一个非常 hard 的 problem：**怎么从一段 smartphone 拍的人类握持工具的 video，scale 出千万级 functional dexterous grasp 数据，并训出一个能泛化到 unseen objects 的 multimodal grasping policy**。Project page: https://cordex-manipulation.github.io

---

## 1. 问题定义：为什么 Functional Dexterous Grasping 难

普通 grasping（比如 AnyGrasp、DexGraspNet）只优化 stability——把 object 抓住、wrench-space force closure 满足即可。**Functional grasping** 多了一个 semantic constraint：必须 contact 到 object 的 *functional region*（比如 drill 的 trigger、spray bottle 的 handle+trigger 组合、pipette 的 plunger），否则抓得再稳也没有意义。

paper 公式 1 形式化了这一点：

$$\forall f \in \mathcal{F}, \exists p_f \in \mathcal{R}_f \quad \text{s.t.} \quad \text{dist}\big(h_f(g), p_f\big) < \epsilon$$

- $f$：index over functional fingers（比如食指按 trigger）
- $\mathcal{F} \subseteq \{1, \dots, M\}$：被指定为"functional"的 finger 子集，$M$ 是总 finger 数
- $\mathcal{R}_f \subseteq \mathbb{R}^3$：object 上对应 finger $f$ 的 functional region（一个 3D 空间集合）
- $h_f(g)$：在 grasp $g$ 下 finger $f$ 的 fingertip 3D 位置
- $\epsilon$：tolerance（real-world 里通常 1 mm，非常严格）

同时还有 **stability** constraint：stabilizing fingers $\mathcal{S}$ 必须能抵抗 external wrench。Grasp $g = (T, \theta)$，其中 $T \in SE(3)$ 是 wrist pose，$\theta \in \mathbb{R}^K$ 是 $K$ 个 joint angles（Shadow Hand $K=22$，Inspire Hand $K=6$）。

**Intuition**：普通 grasping 是一个 6D pose + K-dim joint 的 search 问题；functional grasping 在这个基础上又加了一个 *semantic placement* 约束——某些 finger 必须落在特定的几何小区域上。这就让 random sampling / pure optimization 几乎不可能，必须靠 data-driven。但 data 从哪来？这就是 paper 的切入点。

---

## 2. 两个 Bottleneck（paper 反复强调）

### 2.1 Data scarcity
- MoCap / teleoperation（RealDex、OakInk、DexGraspNet 的某些子集）成本高、scale 差
- In-the-wild video（Deft、Web2grasp）reconstruction noise 大，需要大量人工 curation

### 2.2 缺 semantic + geometric 联合 reasoning
- 多数 prior work（UniDexGrasp、GenDexGrasp、D(R,O)）只吃 point cloud，丢掉了 RGB 的 semantic cue
- 小 functional region（trigger / button）在 partial point cloud 里就几个点，容易被 uniform sampling 淹没

CorDex 同时打这两个 bottleneck：一个 data engine 解决 data，一个 multimodal network 解决 reasoning。

---

## 3. Data Engine：Generate → Transfer → Adapt

这是 paper 的真正核心。整个 pipeline 见 Fig. 2。三 stage 设计得很精巧。

### 3.1 Generate：Internet Image → 3D Models

输入只有一段 smartphone video（比如人握 hammer）。Engine：

1. 从 Internet retrieve 大量同 category images，用 DINOv2 feature similarity 过滤，保证 visual diversity 又 relevant
2. 如果某 category 图片不够，用 **GPT-Image inpainting** 补充
3. 用 **Rodin**（hyper3d.ai，https://hyper3d.ai/）做 2D-to-3D generation，得到 100 个 mesh per category

**Intuition**：作者没有用 ShapeNet 这种 fixed dataset（intra-class variation 太小），也没有用纯 text-to-3D（"hammer" 太模糊会生成奇怪 shape）。用 demo video 当 anchor 去 retrieve Internet image，再生成 3D，是一种**以 demonstration 为锚的 visual augmentation**——既保住 functional semantics（同一类工具），又拉满 shape variation。

最终 9 个 category × 100 object × 1200 RGB-D image × 10 grasp per hand × 2 hand = 约 **11M image-grasp pair**，48 A100 跑 3 天。

### 3.2 Transfer：Correspondence-based Contact Transfer

这是 paper 最关键的技术创新，也是 ablation 大跌的地方（72.7% → 18.5%）。

**为什么不直接做 3D correspondence**：SparseDFF、DenseMatcher 等 method 用 3D dense correspondence 跨 instance transfer grasp，但 3D correspondence 训练数据稀缺，跨 category、跨 instance 泛化差（ablation 里 DenseMatcher 在 Inspire hand 上只有 7.6%）。

**CorDex 的 trick**：把 3D transfer 拆成 **2D matching + 3D aggregation** 两步。

具体 pipeline：

1. **从 demo 提取 3D contact keypoints**：
   - 用 **WiLoR**（CVPR 2025，https://github.com/Willow-WLR/WiLoR）重建 human hand mesh
   - 用 **VGGT**（CVPR 2025，https://github.com/facebookresearch/vggt）重建 object point cloud
   - 取 fingertip 最近的 object surface point 作为 contact
   - 因为 VGGT 重建无绝对 scale，用 hand mesh 的 metric scale 反解 object scale（minimize fingertip-to-object distance）

2. **2D matching transfer**：
   - 把 fingertip contacts 投影到 demo 所有 valid frame
   - 把 novel object 从 sphere 上均匀采样的 viewpoint 渲染出来
   - 用 **MatchAnything**（Arxiv 2025，universal cross-modality matcher）或 **EfficientLoFTR**（CVPR 2024）做 demo frame ↔ novel object rendering 的 2D 匹配
   - 匹配上的 pixel 用 camera intrinsics/extrinsics + depth 反投影回 3D

3. **3D aggregation**：
   - 多 view 反投影回来的点很多且 noisy
   - 用 **density-based clustering**（DBSCAN 思路），取 top-3 largest cluster centers 作为 candidate contacts per fingertip
   - 每个 candidate 用其 cluster 内点的平均 2D matching confidence 加权
   - 小 cluster 直接当 outlier 丢掉

**Intuition**：2D matcher 在 internet-scale 预训练，跨 category、跨 appearance 都 robust；而 3D matcher 训练数据小、过拟合。把 hard 3D 问题降维成多个 2D 问题再聚合，是经典 *divide and conquer*——但**关键洞察是保留 multiple hypotheses**（top-3 cluster centers），而不是 single point。因为 2D matching 本质 ambiguous，硬选一个会丢信息；保留多个让下游 physics optimization 自己 resolve。

Ablation 验证了这一点：w/o multiple candidates 掉到 66.1%（从 72.7%），w/ 3D matching 直接掉到 18.5%。

### 3.3 Adapt：Physics-Informed Grasp Optimization

现在 novel object 上面有了一堆 candidate contact points（per fingertip），要把它们变成 robot hand 可执行的 $g = (T, \theta)$。

**关键 issue**：object scale 变了（demo 里 hammer head 离 handle 8 cm，generated hammer 可能 12 cm），transferred contact points 可能 robot hand 够不到。

**解决**：在 robot hand 的 **middle phalanx 和 distal phalanx 上都定义 candidate contact points**（Fig. 2c），给 optimization 更多 freedom。同时初始化 N 个 grasp 并行 optimize。

Loss 设计（公式 2–4）：

**Contact-prior loss**（公式 2）：

$$\mathcal{L}_{prior} = \sum_{l \in \mathcal{C}} \left( \| h_l(g) - o_p \|_2^2 + \alpha \big(1 - n_h^\top n_o\big) \right)$$

- $l$：finger link index，遍历 set $\mathcal{C}$（有 transferred prior 的 links）
- $h_l(g)$：grasp $g$ 下 link $l$ 上 sampled contact point 的 3D 位置
- $o_p$：transferred prior contact point on object surface
- $n_h, n_o$：hand contact 和 object contact 处的 surface normal
- $\alpha$：positional L2 vs normal alignment 的 trade-off 权重
- $1 - n_h^\top n_o$ 是 cosine distance（normal 越对齐越小）

**Stability contact loss**（公式 3）：

$$\mathcal{L}_{stab} = \sum_{l \in \mathcal{C}} \| h_l(g) - o_c \|_2^2$$

- $o_c$：hand contact point 在 object surface 上的 nearest neighbor
- 作用：scale mismatch 时 transferred points 太 dense/sparse，让 hand "退而求其次"贴到最近的 object surface，避免 floating gesture

**Auxiliary contact loss**（公式 4）：

$$\mathcal{L}_{aux} = \sum_{l \in \mathcal{A}} \| h_l(g) - o_c \|_2^2$$

- $\mathcal{A}$：auxiliary links（palm 等）
- transferred prior 只约束 functional fingers 的 middle/distal link，palm 这种 structural support 不约束就会乱飘

加上 standard **joint limit / collision / self-penetration loss**（DexGraspNet 风格），用 differentiable force closure estimator（Liu et al. RA-L 2021）optimize。

优化完用 IsaacGym 验证：六方向施加 external force，object displacement < 2 cm 才算 stable；functional region 距离 < 1 mm 且 avoidance region 不被 contact 才算 functional。**通过验证的 grasp 才进 dataset**——这一步是 data quality 的最后一道 gate。

**Intuition**：整个 Adapt stage 是个 **constrained optimization with soft priors**。prior loss 提供语义引导（"应该 contact 这里"），stab/aux loss 提供几何兜底（"实在不行贴住 object 表面"），physical loss 提供 feasibility。三者权衡出来的 grasp 既 functional 又 stable。

---

## 4. Grasp Prediction Network：Multimodal Fusion

这一部分基于 **D(R, O) representation**（Wei et al., ICRA 2025）。先回顾 D(R, O)：

### 4.1 D(R, O) 回顾

- 一个 grasp 被编码成 robot hand sampled points $\{r_i\}$ 和 object sampled points $\{o_j\}$ 之间的 **dense distance matrix** $D_{ij} = \|r_i - o_j\|$
- 从 $D$ 通过 **multilateration**（Norrdine 2012，algebraic solution）恢复 hand point 3D 位置，再 IK 到 joint angles
- 不需要 collision term in optimization（distance matrix 已经隐含 contact 结构）
- 跨 embodiment：换 hand 只要换 sampled point set
- Policy 是个 **CVAE**：encoder 把 GT grasp 压成 latent $z$，decoder 输入 $(z, \text{object feature})$ 输出 distance matrix；test 时直接从 prior $p(z)$ sample
- Loss = L1 matrix loss + KL regularization + pose reconstruction loss

**CorDex 在 D(R, O) 基础上的改动**：原来只吃 point cloud，现在吃 RGB-D，并加 importance sampling 和 local-global fusion。

### 4.2 Multimodal 输入处理

- RGB image → DINOv2 提 pixel-wise semantic feature
- Depth → back-project 成 3D point cloud
- 每个 3D point 关联 RGB 对应 pixel 的 DINOv2 feature（semantic embedding）+ 自己的 DGCNN geometric feature
- 得到 N=4096 个 multimodal pointwise features

### 4.3 Importance-Aware Sampling

**Motivation**：uniform sample 4096 个点，trigger/button 这种小 functional region 可能只占几十个点，特征淹没。同时 4096 个点过 transformer 太贵。

**Mechanism**：
- 输入：每个 point 的 concatenated [semantic; geometric] feature
- 一个 **lightweight transformer** + global self-attention 估计 per-point importance probability $\hat{p}_i$
- 按 $\hat{p}_i$ 做 sampling，从 N=4096 降到 N'=1024，contact 区域 density 提高
- **GT importance map**：object points 到 GT robot hand points 的距离 → softmax 成分布
- 训练用 **KL divergence** 让 $\hat{p}$ match GT distribution

**Intuition**：这本质是个 **learned non-uniform downsampling**。注意它不是 hard attention（不丢点），而是 sample——只是分布偏向 contact region。这避免了 RoI pooling 那种"看不见全局"的问题，同时把计算 budget 集中在 functional region。

Ablation 显示 w/o importance sampling 掉到 65.1%（72.7% - 7.6%），中等贡献。

### 4.4 Local-Global Fusion Module

这是 paper 的另一核心创新。Fig. 3 的中间部分。

**Motivation**：
- Local detail：trigger 的纹理、button 的边缘——需要 fine-grained geometric + semantic 联合
- Global context：整个 hammer 的形状——hand 怎么放才能稳定

**Mechanism**：
1. **Local cross-attention**：把 geometric feature 当 query，nearby semantic feature 当 key/value（反之亦然）
2. **Adaptive attention radius**：
   - 点稀疏区域 → **larger** receptive field，gather 更多 context
   - 点密集区域 → **smaller** radius，focus 在 local detail
   - 半径可能根据 local point density $\rho$ 自适应，比如 $r \propto \rho^{-1/3}$（保持 expected neighbor count 大致恒定）
3. **Global self-attention**：在 locally fused features 上做 standard self-attention，编码整体结构
4. **Fusion**：local + global feature 集成 unified representation
5. **Cross-attention with robot hand features** → predict distance matrix

**Intuition**：这是 **coarse-to-fine 的双通路设计**。Local 通路（cross-attention + adaptive radius）捕捉 "trigger 在哪、形状如何"，Global 通路（self-attention）捕捉 "整个 object 大形态"。两者 late fusion 后再去 cross-attend robot hand——这步 cross-attention 在做"对每个 hand point，object 上哪些点离它近"，等价于预测 distance matrix。

Ablation：w/o local attention 掉到 52.7%（掉了 20 个点），是最大跌幅之一，说明 local fine-grained fusion 极其关键。

### 4.5 Decoder

Standard D(R, O) decoder：
- CVAE 结构
- Decoder 输入 $(z, \text{fused object feature}, \text{robot hand feature})$
- 输出 dense distance matrix
- Multilateration + IK → $g = (T, \theta)$

Inference time：Shadow Hand 0.92 s，Inspire Hand 0.36 s（4090 GPU）。Shadow 慢是因为 DoF 多、IK 解空间大。

---

## 5. 实验

### 5.1 Dataset 统计

- 9 categories: Drill, Pipette, Stapler, Spray Bottle, Hammer, Syringe, Hair Dryer, Aerosol Can, Glue Gun
- 900 objects（100 per category）
- 1.08M RGB-D images
- ~11M image-grasp pairs
- 2 embodiments: Shadow Hand (22-DoF) + Inspire Hand (6-DoF)
- 48 A100 × 3 days

### 5.2 Simulation 结果（Table I）

Shadow Hand 平均 success rate：

| Method | Avg |
|---|---|
| D(R,O) [3] | 18.3% |
| D(R,O) w/ our data | 36.0% |
| SparseDFF* | 14.8% |
| DenseMatcher* | 16.9% |
| AG-Pose w/ our data | 67.5% |
| **Ours** | **88.5%** |

Inspire Hand：

| Method | Avg |
|---|---|
| D(R,O) w/ our data | 17.6% |
| SparseDFF* | 7.8% |
| DenseMatcher* | 7.6% |
| AG-Pose w/ our data | 48.9% |
| **Ours** | **74.7%** |

**几个关键 observation**：
1. D(R,O) w/ our data vs. 原版 D(R,O)：18.3% → 36.0%，说明 **data engine 单独就有 ~2× 提升**，但 model 架构改动还能再翻一倍多
2. SparseDFF / DenseMatcher 给的是 **complete object model / two-view**，比 CorDex 的 single-view 输入更完整，但效果反而差一个数量级——证明 3D correspondence 跨 instance transfer 是真不行
3. AG-Pose (category-level pose estimation) 在同 data 下也只到 67.5%，因为 coarse alignment 满足不了 1 mm 的 functional tolerance

### 5.3 Real-World 结果（Table II）

6 categories × 3 objects × 5 poses = 15 trials per category，OYMotion hand + Franka FR3：

| Method | Drill | Pipette | Stapler | Spray | Hammer | Glue Gun | Total |
|---|---|---|---|---|---|---|---|
| D(R,O) w/ our data | 2/15 | 0/15 | 3/15 | 2/15 | 4/15 | 2/15 | 13/90 |
| SparseDFF* | 3/15 | 0/15 | 3/15 | 1/15 | 3/15 | 1/15 | 11/90 |
| DenseMatcher* | 1/15 | 0/15 | 2/15 | 0/15 | 3/15 | 0/15 | 6/90 |
| AG-Pose w/ data | 3/15 | 2/15 | 6/15 | 3/15 | 9/15 | 4/15 | 27/90 |
| **Ours** | **10/15** | 7/15 | **11/15** | **11/15** | **13/15** | **10/15** | **62/90 (69%)** |

**69% 的 real-world success** 在 functional dexterous grasping 这个领域是相当强了。注意 pipette 这种细长物体最弱（7/15），可能因为 partial view 下细物体几何 ambiguity 大。

### 5.4 Ablation 总结（Table III，Inspire Hand）

| Variant | Avg | Δ |
|---|---|---|
| Full | 72.7% | – |
| (1) 3D matching 替代 2D+3D | 18.5% | -54.2% |
| (2) w/o multiple candidates | 66.1% | -6.6% |
| (3) w/o image input | 20.7% | -52.0% |
| (4) w/o importance sampling | 65.1% | -7.6% |
| (5) w/o local attention | 52.7% | -20.0% |

**关键 takeaway**：
- 3D matching 替代（1）和 w/o image（3）是两个最大跌幅，**都在 50% 量级**
- (1) 验证了 data engine 的核心设计：跨 instance correspondence 必须走 2D
- (3) 验证了 multimodal fusion 必要性：纯 point cloud 在 functional grasping 里语义信息严重不足
- Local attention（5）掉 20 点：fine-grained local 推理对 functional region 识别至关重要

---

## 6. 跟相关工作的关系

- **vs. DexGraspNet / UniDexGrasp**：他们做 stable grasping，没有 functional constraint；CorDex 加了 semantic 维度
- **vs. SparseDFF / DenseMatcher**：他们是 one-shot correspondence transfer，用 3D dense matching，跨 instance 泛化差；CorDex 用 2D+3D+physics，明显更鲁棒
- **vs. AG-Pose / category-level pose**：coarse alignment 不够精度；CorDex 直接 predict grasp gesture，bypass alignment
- **vs. Deft / Web2grasp**：从 web video 学，但 reconstruction noise 大；CorDex 只用一段 demo video 当 seed，剩下全靠 synthetic generation
- **vs. DexMimicGen / Robotwin**：automated data generation 思路类似，但 CorDex 针对 functional grasping 这个特定 problem 设计了 correspondence transfer
- **vs. D(R, O)**：直接 predecessor，CorDex 在它基础上加 multimodal fusion + importance sampling

---

## 7. Limitations 和 Future Direction

作者自己承认：

1. **Depth noise sensitivity**：虽然训练时 inject 了 depth noise，real-world 严重 corrupted depth 仍会让 model 失效。这个是 synthetic→real gap 的老问题，可能需要更精细的 depth simulation（比如模拟 Kinect/RealSense 的 multipath interference、flying pixels）
2. **Category-specific training**：每个 category 要一段 demo + 训一个 model，还不是 open-set。未来要做 universal model，可能需要 multi-task training + in-context learning

我自己的额外思考（build intuition）：

- **Demo video 数量 vs. generalization**：现在 9 个 category 各一段 demo。如果改成 100 个 category，会不会出现 emergent open-set generalization？这跟 LLM 的 scaling 假设类似
- **Latent space 的可解释性**：CVAE 的 $z$ 现在只是 sample 多样 grasp，但能不能让它 encode "grasp type"（power grasp vs. precision grip）做可控生成？
- **Tool use downstream**：functional grasp 只是第一步，抓完之后还要 use（按 trigger、敲钉子）。CorDex 输出的 grasp 能不能直接 plug 进一个 manipulation policy？
- **Tactile feedback**：现在完全是 vision-driven，没有 tactile。Functional grasping 里 trigger 按下去的 force feedback 很关键，未来可能要加 tactile simulation（TACTO、Taxim）做 sim-to-real

---

## 8. 参考 Web Links

- Project page: https://cordex-manipulation.github.io
- D(R, O) Grasp (predecessor): https://d-ro.github.io/
- DINOv2: https://github.com/facebookresearch/dinov2
- MatchAnything: https://github.com/xingyihe/MatchAnything
- EfficientLoFTR: https://github.com/zju3dv/EfficientLoFTR
- WiLoR (hand reconstruction): https://github.com/Willow-WLR/WiLoR
- VGGT (3D reconstruction): https://github.com/facebookresearch/vggt
- IsaacGym: https://developer.nvidia.com/isaac-gym
- Rodin / Hyper3D: https://hyper3d.ai/
- SparseDFF: https://github.com/j96w/SparseDFF
- DenseMatcher: https://github.com/Ju-Zhu/DenseMatcher
- AG-Pose: https://github.com/linjiajia0707/AG-Pose
- DexGraspNet: https://github.com/PKU-EPIC/DexGraspNet
- OakInk: https://github.com/ll4lab/oakink
- Grounded SAM: https://github.com/IDEA-Research/Grounded-Segment-Anything
- Blender: https://www.blender.org/

---

**总结成一句话 build intuition**：CorDex 把 "从一段 video 学 functional grasping" 这件事拆成了 *visual augmentation（generate 3D assets）+ cross-modal correspondence transfer（2D matcher + 3D clustering）+ physics-aware refinement（multi-objective optimization）* 三段式 data engine，配上一个 *importance-sampled multimodal local-global fusion* 的 policy network，本质上是用 **internet-scale pretrained 2D matcher** 当 "semantic bridge" 跨越 instance gap，用 **physics optimization** 当 "filter" 过滤掉 ambiguous transfer，再用 **learned sampling** 把模型 capacity 集中到 functional region——三个 bottleneck（data、correspondence、reasoning）各打一发，最终在 real world 拿到 69% 的 functional grasp success rate。
