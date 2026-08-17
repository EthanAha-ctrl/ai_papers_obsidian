---
source_pdf: IKEAManuals at Work 4D Grounding of Assembly Instructions on Internet
  Videos.pdf
paper_sha256: d3cac3e6879e922903a39b878d8d11c4309e09fa43bfc419b756b22380f1ad1d
processed_at: '2026-08-05T09:03:39-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

好，我换个角度，把这篇paper"翻译"成人话。

---

## 一句话概括

Stanford的Jiajun Wu组跟J.P. Morgan AI Research一起，做了一个dataset，把IKEA家具组装这件事在video里的"4D grounding"给标注出来了——也就是说，videos里每一帧、每个零件在3D空间中的位置和朝向，还有它跟说明书第几步对应，全都标好了。

这听起来好像没什么，但仔细想想，这件事**从来没人做过**。

---

## 为什么这事难，为什么之前没人做

你想想，一个人录了一段YouTube video，在自己家客厅组装一把IKEA椅子。video里画面抖、光线变、手挡住零件、镜头切来切去、四条腿长得一模一样分不清谁是谁。

你作为一个AI researcher，想让机器理解这段video里发生了什么，你需要知道：

- 这帧画面里这个木板的6-DoF pose是什么（位置xyz + 朝向roll/pitch/yaw）
- 这个木板跟说明书里画的哪个零件对应
- 这帧属于组装过程的第几步、第几个substep
- 这个木板跟旁边那个螺丝在3D空间里是什么relative关系

之前的数据集，**每个都只解决了一部分**：

- IKEA-Manual (2022 NeurIPS, 同组前作) 有3D模型 + 说明书，但没video，只有static images
- IKEA ASM (WACV 2021) 有video，但是lab里拍的，4种家具，5个环境，有depth但是calibrated的
- IKEA in the Wild (CVPR 2023) 有Internet videos，14类420个，但是没有3D model，没有pose标注
- Assembly101 (CVPR 2022) 有101个toy vehicle assembly videos，但是lab拍的，没有3D model

所以gap很清楚：**没人把Internet video + 3D model + manual + 6-DoF pose对齐**。这paper就是来填这个gap的。

---

## 他们到底标注了什么

每一帧（1 FPS采样，共34,441帧），他们标了：

1. **2D segmentation mask** — 画面里哪些pixel属于哪个part/sub-assembly
2. **6-DoF pose** — 每个part在camera坐标系下的位置和朝向
3. **Part identity** — 这个pixel blob对应3D model里第几号零件
4. **Camera intrinsics** — 这一帧用的camera参数（焦距之类的）
5. **Temporal alignment** — 这帧属于manual的第几步，以及第几个substep

其中"substep"是他们新引入的概念。Manual里说"把4个腿装到座板上"算一步，但video里是一个一个装，每个装一条腿是一个substep。平均一个step包含7.59个substeps，总共1120个substeps。

这个substep的区分非常重要，因为manual是high-level illustration，video是fine-grained demonstration，两者granularity差一个数量级。

---

## 标注pipeline怎么搞的

这是paper里最工程的部分，但也是最有价值的——因为如果标注方法不robust，dataset就废了。

### Step 1: 找数据来源

3D models从IKEA-Manual拿（36个家具），videos从IAW (IKEA in the Wild) 拿（98个assembly videos），通过IKEA product ID做matching。

### Step 2: Temporal segmentation两层

先从IAW的coarse step alignment开始，然后手动调整start/end time让segment更完整（从拿起零件到拧紧螺丝）。

再引入substep——每当新零件出现，或者新sub-assembly通过positioning/fastening形成时，切一刀。

### Step 3: Part identity标注

这里有个intuition很关键：**annotator必须先看完整段video，才能开始标**。

为什么？因为IKEA的零件太像了。四条椅子腿长得一模一样，你不看完全段video，根本不知道现在画面里这条腿是manual里的第几号腿。

他们参考IKEA-Manual的assembly order——如果manual说leg_3是第一个装的，那video里第一个装的腿就标为3。

还有一种恶心情况：人装错了又拆开重装。Annotator得识别这是"装错重装"而不是"正常组装"，identity保持不变。

### Step 4: Mask标注

用SAM (Segment Anything Model) 辅助。Annotator点几个keypoint，SAM生成mask。SAM失败的地方（比如两个texture一样的木板挨着，SAM分不开边界），用brush手动修。

这里有个intuition：**SAM对"物体之间边界"的segmentation其实很差**，特别是当两个物体表面颜色相近、紧贴在一起时。Furniture assembly恰恰全是这种场景。

### Step 5: Camera intrinsics估计

Internet video的camera参数会变——focal length会调、可能切换前后摄像头。所以他们：

1. 人工标注camera change points（哪里焦距变了、哪里切摄像头了）
2. 对每个segment，标2D-3D keypoint correspondences
3. 用PnP算法估计pose
4. 用RANSAC过滤outlier keypoints
5. 生成多个candidate intrinsics，选reprojection error最小的
6. 取top-10里minimal set的intrinsics

$$E = \sum_i \|x_i - \pi(K(R X_i + \mathbf{t}))\|^2$$

这个公式里：
- $x_i$ 是keypoint在2D image里的位置
- $X_i$ 是对应3D point
- $K$ 是camera intrinsic matrix（要估计的）
- $R, \mathbf{t}$ 是camera extrinsic（pose，要估计的）
- $\pi$ 是perspective projection

这个reprojection error就是把3D point用估计的 $K, R, \mathbf{t}$ 投影到2D，跟实际标的2D keypoint比，差的平方和。

### Step 6: Pose refinement

这是最累的一步。即使2D projection看起来对，3D里parts的relative pose可能错。为什么？因为2D projection有歧义——多个3D configuration可以project到同样的2D image。

所以他们做了一个interface，让annotator从side/front/top等多个orthographic视角看3D scene，确认coplanarity、inter-part distance、left/right/front/back关系都对。还用前一帧的pose初始化当前帧，保证temporal smoothness。

**Intuition**: 单纯depth estimation（比如MiDaS）在real-world video上不可靠，occlusion和challenging viewpoint会让depth估计崩掉，所以必须human-in-the-loop。

---

## 他们跑了5个baseline实验，结果都很差

这是这篇paper有意思的地方——他们不是光放一个dataset就走人，而是跑了5个task的baseline，全部显示现有SOTA方法在real-world video上struggle。

### Task 1: Assembly Plan Generation

给video，predict assembly的hierarchical plan（DAG）。每个node是一组parts，edge是assembly order。

两个heuristic baseline：
- SingleStep：所有parts直接连root（一步装完）
- GeoCluster：用DGCNN提取3D feature，迭代group geometrically similar parts

Metric是precision/recall/F1，分Simple Matching（只看parts集合对不对）和Hard Matching（parts集合和parent-child关系都对）。

结果：Hard Matching F1最高只有16.30%（IKEA-Manual上）和11.51%（Ours上）。原因——video-derived plan更长更多样，Laiva shelf有8种不同的assembly order，baseline的deterministic方法根本capture不到这种diversity。

### Task 2: Part-Conditioned Segmentation

给frame和sub-assembly，predict binary mask。

Baselines: CNOS, SAM-6D。

结果：IoU最高0.16 (SAM-6D)，Top-5 IoU最高0.40。非常低。Common failures：heavily occluded parts, visually complex backgrounds, textureless 3D shapes。

**Intuition**: 即使给3D CAD model，在real-world video里做part segmentation也是open problem。

### Task 3: Part-Conditioned 6D Pose Estimation

给frame和sub-assembly（用GT mask，upper bound测试），predict 6-DoF pose。

Baselines: SAM-6D (用MiDaS depth), MegaPose, Differentiable Rendering (MSE loss), Differentiable Rendering (Occlusion-Aware, PHOSA风格)。

Metric:
- ADD: $\text{avg}_i \| (R X_i + \mathbf{t}) - (R' X_i + \mathbf{t}') \|$
  - $(R, \mathbf{t})$ 是GT pose, $(R', \mathbf{t}')$ 是predicted pose, $X_i$ 是model point $i$
  - 就是predicted pose下model点跟GT pose下对应model点的平均距离
- ADD-S: 对symmetric object用 $\min_j$ 而不是对应点

结果：MegaPose最好，ADD=1.36, ADD-S=0.89。但所有方法都挣扎。MegaPose confused by symmetric parts和challenging viewpoints。SAM-6D受MiDaS depth质量限制。Differentiable rendering worst，因为silhouette loss在large textureless家具上不reliable。

### Task 4: Video Object Segmentation

给substep的video sequence和第一帧GT mask，predict后续帧mask。

Baselines: SAM2 Hiera-L, Cutie-base。

Metric: J&F（region similarity J + boundary accuracy F的mean）。

结果：SAM2在Ours上73.6（其他benchmark上75.6-91.6），Cutie在Ours上54.7（其他60.7-88）。

**Intuition**: 我们的dataset比DAVIS, YTVOS, MOSE都难。原因：camera movements, similar-looking parts, small parts, frequent occlusions, extended sequences。

### Task 5: Shape Assembly with Instruction Videos

给3D parts和instruction video，predict所有parts的6-DoF poses for final assembly。

他们提出modular pipeline：
1. Keyframe detection (找parts正在被combined的frame)
2. Segmentation + part identification
3. Pose estimation from keyframe
4. Iterative assembly

两个settings：
- Setting 1: 用dataset annotation提取keyframe + 用GT poses → Chamfer Distance = 0.33
- Setting 2: 用GPT-4o检测keyframes → Chamfer Distance = 0.55

$$\text{CD}(S_1, S_2) = \frac{1}{|S_1|}\sum_{x \in S_1} \min_{y \in S_2} \|x - y\|^2 + \frac{1}{|S_2|}\sum_{y \in S_2} \min_{x \in S_1} \|y - x\|^2$$

这里 $S_1$ 是predicted assembled furniture point cloud, $S_2$ 是GT point cloud。每个点找对方最近点，距离平方平均。

GPT-4o失败case：15个video里5个没识别出final assembly step，家具不完整。

**Intuition**: 即使给GT pose (Setting 1)，Chamfer Distance也不为0——因为substep的last frame不一定parts完全connected。这揭示keyframe detection本身有ambiguity。

---

## Error Analysis特别有启发

Appendix H分析了6D pose estimation的4类failure：

### 1. Close-up Views with Partial Visibility

画面里只看到家具一小部分。模型推断不出完整object的scale。**这种场景在real-world assembly video很常见，但existing datasets几乎没有**。

### 2. Ambiguous Semantic Information

家具正面背面长得像。MegaPose把orientation估错180度。SAM-6D好一点（用了depth），但仍然uncertain。

### 3. Depth Discontinuities Due to Occlusions

手挡在物体前面。SAM-6D分不清occluder和target的depth，predicted bounding box排除了被挡区域。MegaPose因为appearance-based影响小但也reduced accuracy。

### 4. Challenging Viewpoints

Top-down看家具腿、部分被挡的角度、难以判断full scale/shape的perspective。Both方法都misinterpret scale或shape。

### 成功case对比

Full object visibility + minimal occlusion时两个方法都better。这contrast很有意思——说明dataset确实capture了real-world的难度分布。

---

## 我的几个核心takeaway

### 1. "4D grounding"这个词用得precise

3D空间 + 1D时间 = 4D。每帧都有6-DoF pose，时间上形成trajectories。这需要三个约束同时满足：
- Temporal smoothness（帧间pose不能跳）
- Geometric consistency（同帧parts间relative pose要合理）
- Cross-frame consistency（同一part跨帧identity和pose连续）

### 2. Substep是conceptual contribution

Manual说"装4条腿"，video里是4个独立action。这种hierarchical decomposition让manual和video的granularity差一个数量级。这个gap一直是prior work回避的——要么只看manual，要么只看video——这paper把它explicitly bridge了。

### 3. Real-world比lab难得太多

所有SOTA方法在这dataset上都struggle。即使给GT mask，6D pose估计ADD还是1.36（MegaPose最好）。即使给第一帧GT mask，VOS的J&F只有73.6（SAM2）。即使simple heuristic，plan generation的Hard F1只有11.51%。

这说明real-world visual grounding远未solved。Furniture assembly是一个excellent proxy——它有large multi-part objects, complex spatial relationships, long sequences, frequent occlusions, similar parts——这些challenge在很多domain都common。

### 4. Annotation pipeline本身是contribution

Algorithm-assisted manual annotation：SAM帮mask，PnP+RANSAC帮initial pose，human从多视角refine。这种hybrid在scale和quality间找balance。Fully manual太慢，fully automatic不够准。他们的pipeline是reproducible的framework。

### 5. 未来方向清晰

- 加入audio和text narration作为additional modality
- 用这dataset训练end-to-end model（目前baselines都是pretrained没finetune）
- Scale up annotation（目前98 videos）
- Test transferability到non-furniture assembly domains
- 用VLM（GPT-4o失败说明这里有机会）做keyframe detection和assembly step identification

---

## 跟更广领域的联系

我想到一些paper没cite但相关的方向：

### Embodied AI / Robotics

Robot要assemble家具需要三层能力：
- Top: task planning from manual
- Middle: visual grounding from video (这dataset的focus)
- Bottom: motion planning and control

这dataset提供middle层的training/eval data。Robotics community的Behavioral Cloning, Diffusion Policy等方法可能能用这dataset的4D trajectories做imitation learning。

参考: 
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Open-X-Embodiment: https://robotics-transformer-x.github.io/

### 3D Vision

最新的6D pose methods like FoundationPose可能在furniture parts上表现更好：
- FoundationPose: https://nvlabs.github.io/FoundationPose/
- BundleSDF: https://github.com/NVlabs/BundleSDF

### VLM / Video Understanding

GPT-4o在keyframe detection上失败说明VLM还不懂assembly process。这dataset可以作为VLM的fine-tuning data或者evaluation benchmark。

参考:
- LLaVA: https://llava-vl.github.io/
- Video-LLaVA: https://github.com/PKU-YuanGroup/Video-LLaVA

### Generative 3D Models

有了4D grounding，可以训练生成模型predict assembly trajectories或者generate assembly plans。这跟shape assembly领域的Neural Assembly Graph、PartNet-Mobility等工作connect。

参考:
- PartNet: https://partnet.github.io/
- ShapeAssembly: https://cs.stanford.edu/~ericyi/shapeAssembly/

---

## 总结人话版

这paper就是：Stanford的人花了大力气，把IKEA组装video里的每个零件每一帧在3D空间中的位置、朝向、跟说明书的对应关系全标出来了。这是第一次有人这么干，之前要么没video，要么没3D model，要么没pose，要么lab里拍的不真实。

然后他们拿现有SOTA方法在这dataset上跑5个task，全部表现差。说明real-world assembly理解是个open problem，这dataset可以作为benchmark。

技术上有几个intuition值得记：
- Annotator要先看完全段video才能标part identity（因为零件太像）
- SAM对物体间边界segmentation差，furniture场景全是这种case
- 2D projection对了不代表3D pose对，需要多视角refine
- Substep概念bridge了manual和video的granularity gap
- Real-world的close-up, occlusion, challenging viewpoint是existing dataset没capture的

未来方向是scale up、加modality、train end-to-end model、test transferability。这dataset本身是open challenge而不是solved problem。

---

**Dataset link**: https://github.com/yunongLiu1/IKEA-Manuals-at-Work  
**Paper arXiv**: 搜"IKEA Manuals at Work 4D Grounding"  
**License**: CC-BY-4.0

Karpathy你会感兴趣的是：这paper其实是在做vision社区的"Assembly任务"。之前的assembly task要么缺3D grounding要么缺真实video，这个dataset把gap填了。而且所有baseline都struggle，说明这是个有挑战性的benchmark。从build intuition角度，这dataset让你能看到real-world assembly的complexity到底有多高——occlusion, similar parts, camera changes, ambiguous views——这些在lab dataset里根本不会出现。这跟你在Stanford教CS231N时强调的"real-world data is messy"完全consistent。

---

# IKEA Video Manuals at Work: 4D Grounding of Assembly Instructions on Internet Videos 深度解析

## 1. 核心动机与问题定位

这篇paper from Stanford (Jiajun Wu lab) + J.P. Morgan AI Research 解决了一个非常具体的gap in the assembly understanding literature。Furniture assembly requires reasoning at **multiple levels of abstraction**:

- **Top level**: task planning — decompose task into subtasks, compute dependency ordering (manual 给出)
- **Middle level**: visual grounding — register each part at pixel level, identify geometry and pose (translate manual to actionable steps)
- **Bottom level**: motion planning and control — execute steps avoiding collisions given embodiment

之前的工作都是各自解决一部分：
- Agrawala et al. [1] 设计assembly instructions但是 **not visually grounded**
- IKEA-Manual [2] 把assembly steps ground到3D part models并register at pixel level，但 **没有action trajectories**
- IKEA ASM [3], IKEA-FA [9,10], Assembly101 [11] 在controlled lab环境采集，限制diversity
- IKEA in the Wild (IAW) [5,32] 有diverse videos但 **没有3D part correspondence和manual alignment**

IKEA Video Manuals 的key innovation就是把这三者(first time)统一align到一起，并且提供 **6-DoF pose annotations on internet videos**，这是之前的dataset都没有做的。

Reference: 
- IKEA-Manual: https://github.com/rwang92/ikea-manual  
- IAW paper: https://arxiv.org/abs/2303.15260  
- Assembly101: https://assembly-101.github.io/

---

## 2. Dataset定义的数学formalization

让我把Section 3.1的formalization讲清楚，因为这是理解整篇paper的key。

**Furniture S** 由一组3D parts组成：
$$S = \{p_1, p_2, \ldots, p_N\}$$

其中 $p_i$ 是第 $i$ 个part的3D shape（point cloud或mesh），$N$ 是parts总数。

**6-DoF pose** of part $i$ 记为 $\zeta_i = (R_i, \mathbf{t}_i)$，其中：
- $R_i \in SO(3)$ 是rotation matrix（3个自由度：roll/pitch/yaw）
- $\mathbf{t}_i \in \mathbb{R}^3$ 是translation vector（3个自由度：x/y/z）

**Pose变换**作用于part：
$$\zeta_i(p_i) = R_i \cdot p_i + \mathbf{t}_i$$

整个furniture的assembled state：
$$\theta = \{\zeta_1(p_1), \ldots, \zeta_N(p_N)\}$$

**Video frames**: $\{f_1, f_2, \ldots, f_T\}$，$T$是总帧数。

**Sub-assembly** $A$: 在assembly过程中形成的intermediate结构，可以是individual part或者是already-combined parts的集合。这点很关键 — paper区分了part和sub-assembly，因为assembly是hierarchical的：两个parts组成一个sub-assembly，sub-assembly再和新的part组成更大的sub-assembly。

**Instruction manual**: $\{m_1, \ldots, m_L\}$，$L$是manual steps数。

**Temporal alignment**: $\phi(m_i) \rightarrow \{f_j, \ldots, f_k\}$ where $j \le k$，即把每个manual step映射到video的一个segment。

**4D grounding** = 3D空间 + 1D时间，所以叫"4D" — 每帧都有6-DoF pose，时间上形成trajectories。

---

## 3. 数据集统计与对比

| 指标 | 数值 |
|------|------|
| # Videos | 98 |
| # Annotated frames | 34,441 |
| Avg frames/video | 316 |
| # Manual steps | 137 |
| # Substeps | 1,120 |
| # Furniture models | 36 |
| # Parts total | 268 |
| Avg parts/furniture | 7 |
| # Furniture categories | 6 (chair/table/bench/desk/shelf/misc) |
| # Environments | ~90 |
| Longest video | 49 minutes |
| Avg video duration | 6 minutes |
| Avg substeps per step | 7.59 |
| Multiple assembly sequences | 25% of items (Laiva shelf有8种variations) |

这个 **25% items 有multiple valid assembly sequences** 很有意思 — Laiva shelf 有8种不同的assembly order。这反映了real-world assembly的non-determinism，跟IKEA-Manual那种 "the plan" 的视角很不一样。

**与现有dataset对比** (Table 1):

| Dataset | # Class | # Obj | Source | # Env | 3D Model | 3D Info | Camera |
|---------|---------|-------|--------|-------|----------|---------|--------|
| **Ours** | 6 | 36 | Internet | ~90 | √ | **6-DoF Pose** | Estimated* |
| Assembly101 | 15 | 101 | Lab | 1 | × | Depth | Calibrated |
| HA-ViD | 1 | 35 parts | Lab | 1 | √ | Depth | Calibrated |
| IKEA-Manual | 6 | 102 | / | / | √ | 6-DoF Pose | Estimated |
| IKEA Ego 3D | 4 | 4 | Lab | 1 | × | Depth | Calibrated |
| IKEA ASM | 3 | 4 | Lab | 5 | × | Depth | Calibrated |
| IKEA in Wild | 14 | 420 | Internet | ~1000 | × | / | Uncalibrated |

注意 *: IKEA Video Manuals的camera parameters虽然estimated，但做了 **额外的processing来确保within each video segment的consistency**（除非有明显的camera change）。这区别于IKEA-Manual，后者对每个part单独estimate intrinsic，可能造成parts之间relative pose不consistent。

---

## 4. Data Collection与Annotation Pipeline (Section 4)

这是paper最技术性的部分，让我详细讲。

### 4.1 数据源

- **3D models**: 来自IKEA-Manual [2]的36个segmented 3D furniture models
- **Videos**: 来自IAW [5,32]的98个assembly videos
- 通过IKEA product ID matching建立3D model和video的correspondence

### 4.2 Temporal Segmentation的两层结构

**Coarse layer (steps)**: 来自IAW的manual step alignment，但作者 **手动调整了start/end time** 以包含更完整的assembly process（从pick up part到positioning再到tightening）。

**Fine layer (substeps)**: 作者新引入的概念。每个manual step可能涉及multiple parts的assembly，但video里parts是 **一个一个** 装的。Substep的标记规则：
1. 当new part出现时
2. 当new sub-assembly通过positioning或fastening形成时

平均每个step包含7.59个substeps。这种hierarchical decomposition非常符合physical reality，跟manual的high-level illustration形成互补。

### 4.3 Part Identity Annotation

这是关键的一步。为什么难？因为IKEA parts有ambiguities（Fig. A4）：
- (a) Wrongly assembled parts — 装错位置又重装，annotator会confuse
- (b) Similar/identical-looking parts — 比如四条table leg看起来一样
- (c)(d) Heavy occlusion — 看不到part boundary

解决方案：**annotator先watch整个video**，然后在每个substep的first frame标记part identity，并参考IKEA-Manual的assembly order。比如如果IKEA-Manual说leg_3是第一个装的，就把video里第一个装的leg标为3。

这种 **全局观看再局部标注** 的策略是处理part identity ambiguity的关键intuition。

### 4.4 Segmentation Mask Annotation

基于SAM (Segment Anything Model) [33] 构建 annotation interface:
1. Annotator看到3D model (target part highlighted) + first frame的part 2D location
2. 在当前frame标记keypoints，fed to SAM生成mask
3. SAM失败时（similar texture between parts, low-light regions），用brush/eraser手动修改

SAM的限制在于它对part之间边界的segmentation不准，特别是当parts表面颜色相近时。这正是furniture assembly的common case。

Reference: SAM at https://segment-anything.com/

### 4.5 2D-3D Correspondence与Camera Estimation

这是最technical的部分。整个pipeline：

**Step 1**: 识别camera change points（focal length adjustment或multi-camera切换）

**Step 2**: 对每个video segment，annotate 2D-3D point correspondences（keypoints在3D model上和在2D image上）

**Step 3**: 使用 **PnP (Perspective-n-Point)** 算法 [34] 估计object pose。PnP解决的问题是：给定n个3D points $\{X_i\}$ 和它们在image plane上的2D projections $\{x_i\}$，以及camera intrinsic matrix $K$，估计camera的extrinsic parameters $(R, \mathbf{t})$ 使得：

$$x_i = \pi(K(R X_i + \mathbf{t}))$$

其中 $\pi$ 是perspective projection。EPnP [34] 是 $O(n)$ 的精确解法。

**Step 4**: **RANSAC** [35] 过滤outlier keypoints。RANSAC的intuition：
1. 随机sample最小subset of points fit model
2. 计算所有points的inlier count (within threshold)
3. 重复多次，选inlier最多的model

**Step 5**: 对每个segment生成多个candidate intrinsics，选reprojection error最小的。Reprojection error定义为：
$$E = \sum_i \|x_i - \pi(K(R X_i + \mathbf{t}))\|^2$$

**Step 6**: 从top-10 intrinsics选minimal set的intrinsics（避免过拟合）

**Step 7**: 用interactive interface人工refine poses。Annotator可以从不同orthographic视角查看3D scene，identify错误。

**Step 8**: Temporal smoothness — 每帧的pose初始化用前一帧的refined pose。

这个pipeline的关键intuition是：单纯从depth estimation恢复pose在real-world video上不可靠（occlusion, partial visibility, challenging viewpoints），所以需要manual annotation + algorithm辅助的hybrid approach。

而且annotation不仅最小化2D projection error，还强调 **part间relative pose的accuracy** 和 **cross-frame consistency**，因为physical assembly有geometric constraints — 比如两个parts在final assembly state的relative pose必须与3D furniture model中的pose align。

Reference:
- EPnP: https://www.epfl.ch/labs/cvlab/software/top-the-epnp-algorithm/  
- RANSAC: classic Fischler & Bolles 1981

### 4.6 Pose Refinement

Fig. A7展示了一个典型问题：2D projection看起来正确，但3D中parts的relative position错了。这是因为2D projection loss is ambiguous — 多个3D configurations可以project to同样的2D image。

Interface允许annotator从多个视角（side, front, top）查看3D parts，确认coplanarity、inter-part distances、正确relative location（right/left, front/back, up/down）。

---

## 5. 五个Application Tasks的实验

### 5.1 Assembly Plan Generation

**任务**: 给定video frames $\{f_1, \ldots, f_T\}$，predict hierarchical assembly plan as DAG $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：
- 每个node $v \in \mathcal{V}$ 对应subset of K parts $\{p_1, \ldots, p_K\}$
- 每个edge $e \in \mathcal{E}$ 表示assembly order和parent-child关系
- Root node $v_r$ 是final assembled shape

**Baselines**:
- **SingleStep**: 所有parts直接连到root（一步装完）
- **GeoCluster**: 用pre-trained DGCNN [36] 提取3D features，迭代地group geometrically similar parts

**Metrics**:
- **Simple Matching**: predicted node正确如果parts集合匹配ground truth
- **Hard Matching**: 必须parts集合**和**parent-child关系都匹配

**Results (Table 2)**:

| Method | Dataset | Simple P | Simple R | Simple F1 | Hard P | Hard R | Hard F1 |
|--------|---------|----------|----------|-----------|--------|--------|---------|
| SingleStep | IKEA-Manual | 100.00 | 35.77 | 48.64 | 10.78 | 10.78 | 10.78 |
| GeoCluster | IKEA-Manual | 44.90 | 48.46 | 43.53 | 16.54 | 16.50 | 16.30 |
| SingleStep | Ours | 98.98 | 16.86 | 26.88 | 3.06 | 2.55 | 2.72 |
| GeoCluster | Ours | 43.04 | 24.16 | 29.74 | 14.98 | 9.49 | 11.51 |

关键观察：
- 两个baseline在Ours dataset上比IKEA-Manual差，因为video-derived plans更长更多样
- Hard Matching分数普遍很低（10-16%），说明structure prediction很难
- GeoCluster略好于SingleStep因为利用了geometric features

**Intuition**: assembly plan的多样性是关键挑战。同一件furniture可以有8种不同的assembly order (Laiva shelf)，而manual往往只展示一种"canonical"的order。Video captures real-world diversity。

DGCNN reference: https://github.com/WangYueFt/dgcnn

### 5.2 Part-Conditioned Segmentation

**任务**: 给定frame $f$ 和sub-assembly $A$，预测binary segmentation mask for $A$。

**Setup**: 12,296 examples, only unique-shape sub-assemblies (去除歧义)。

**Baselines**:
- **CNOS** [37]: CAD-based novel object segmentation
- **SAM-6D** [38]: 考虑额外geometric features (shape, size)

**Metrics**: IoU, Top-5 IoU

**Results (Table 3)**:

| Method | IoU | Top-5 IoU |
|--------|-----|-----------|
| CNOS | 0.09 | 0.21 |
| SAM-6D | 0.16 | 0.40 |

**Intuition**: 即使SOTA的CAD-based方法在internet video上也只能达到0.16 IoU。Common failures是heavily occluded parts, visually complex backgrounds, textureless 3D shapes。说明real-world grounding远比controlled environment难。

CNOS: https://github.com/nv-nguyen/cnos  
SAM-6D: https://github.com/JiehongLin/SAM-6D

### 5.3 Part-Conditioned 6D Pose Estimation

**任务**: 给定frame $f$ 和sub-assembly $A$，predict 6-DoF pose。

**Setup**: 7,795 annotations, 用ground truth mask (上限测试)。

**Baselines**:
- **SAM-6D** [38]: 需要depth (用MiDaS [40] estimate)
- **MegaPose** [39]: render & compare
- **Differentiable Rendering (MSE loss)**
- **Differentiable Rendering (Occlusion-Aware, PHOSA [41])**: 20 random initial poses, refine top 5 with lowest loss for 500 iterations

**Metrics**:
- **ADD** (Average Distance of Model Points): 
$$\text{ADD} = \text{avg}_i \| (R X_i + \mathbf{t}) - (R' X_i + \mathbf{t}') \|$$
  where $(R, \mathbf{t})$ 是ground truth, $(R', \mathbf{t}')$ 是prediction, $X_i$ 是model point $i$。衡量predicted pose和ground truth pose下model points的平均距离。
- **ADD-S** (ADD-Symmetric): 对于symmetric objects，用min over correspondences：
$$\text{ADD-S} = \text{avg}_i \min_j \| (R X_i + \mathbf{t}) - (R' X_j + \mathbf{t}') \|$$

**Results (Table 4)**:

| Method | ADD | ADD-S |
|--------|-----|-------|
| SAM-6D | 2.34 | 1.85 |
| MegaPose | 1.36 | 0.89 |
| Diff. Rendering (MSE) | 3.33 | 2.91 |
| Diff. Rendering (Occlusion-Aware) | 3.29 | 2.86 |

注意：这里数值 **越小越好**，单位是某种normalized distance。

**Intuition**: 即使给GT mask，所有方法都挣扎。MegaPose best overall但confused by symmetric parts和challenging viewpoints。SAM-6D受限于MiDaS depth estimation accuracy。Differentiable rendering worst因为silhouette loss在large textureless furniture上unreliable。

MegaPose: https://github.com/megapose6d/megapose6d  
MiDaS: https://github.com/isl-org/MiDaS  
PHOSA: https://github.com/yufuwang83/phosa

### 5.4 Video Object Segmentation

**任务**: 给定video sequence of a substep $\{f_1, \ldots, f_T\}$ 和initial mask $M_1$，predict $\{M_2, \ldots, M_T\}$。Mask只在该part identity不变期间valid（part连接到其他part形成新sub-assembly之前）。

**Setup**: 至少20 frames的substep，用GT mask初始化第一帧。

**Baselines**:
- **SAM2 Hiera-L** [43]
- **Cutie-base** [48]

**Metric**: J&F (standard VOS metric, region similarity J和boundary accuracy F的mean)

**Results (Table 5)**:

| Method | Ours | SA-V | MOSE | DAVIS 2017 | LVOS | YTVOS 2019 |
|--------|------|------|------|------------|------|------------|
| SAM2 | **73.6** | 75.6 | 77.2 | 91.6 | 76.1 | 89.1 |
| Cutie | **54.7** | 60.7 | 69.9 | 87.9 | 66.0 | 87.0 |

**Intuition**: 我们的dataset比现有benchmarks更难。SAM2 drop moderate (73.6 vs 76-91)，Cutie drop more significant (54.7 vs 60-88)。原因：camera movements, similar-looking parts, small parts, frequent occlusions, extended sequences。

SAM2: https://ai.meta.com/sam2/  
Cutie: https://github.com/hkchengrex/Cutie  
DAVIS: https://davischallenge.org/

### 5.5 Shape Assembly with Instruction Videos

**任务**: 给定3D parts $\{p_1, \ldots, p_N\}$ 和instruction video $\{f_1, \ldots, f_K\}$，predict 6-DoF poses $\{\zeta_1, \ldots, \zeta_N\}$ for final assembly。

**Modular Pipeline**:
1. **Keyframe detection**: 找出两个parts/sub-assemblies正在被combined的frames (通常是每个substep的最后一帧)
2. **Segmentation + Part ID**: identify哪些3D parts在frame里 + 2D locations
3. **Pose estimation**: 从first keyframe开始estimate poses并combine成sub-assembly $A_i = \{\zeta_i(p_i)\}_{i=1}^K$
4. **Iterative assembly**: move到next keyframe, estimate pose of previous sub-assembly + new part, incrementally build furniture

**Two settings**:
- Setting 1: 用dataset annotation提取keyframe + 用GT poses — Chamfer Distance = **0.33**
- Setting 2: 用GPT-4o [49]检测keyframes (15 videos) — Chamfer Distance = **0.55**

**GPT-4o failure**: 15个video中5个failed to identify final assembly step，导致incomplete furniture。

**Chamfer Distance**定义:
$$\text{CD}(S_1, S_2) = \frac{1}{|S_1|}\sum_{x \in S_1} \min_{y \in S_2} \|x - y\|^2 + \frac{1}{|S_2|}\sum_{y \in S_2} \min_{x \in S_1} \|y - x\|^2$$

其中 $S_1, S_2$ 是两个point sets（assembled vs ground truth furniture），$x, y$ 是points。衡量两个surfaces的distance。

**Intuition**: 即使有GT poses (Setting 1)，Chamfer Distance = 0.33也不为零 — 因为substep的last frame不一定parts完全connected。这揭示了keyframe detection本身的inherent ambiguity。

GPT-4o reference: https://arxiv.org/abs/2303.08774

---

## 6. Error Analysis (Appendix H) — 这部分对build intuition特别有用

作者identify了4类6D pose estimation failure modes：

### H.2.1 Close-up Views with Partial Visibility
只有小部分furniture visible。SAM-6D和MegaPose都struggle infer complete object pose from limited visual info。**Both fail to determine scale correctly**。这种场景在real-world furniture assembly videos很常见，但在existing datasets很rare。

### H.2.2 Ambiguous Semantic Information
Objects有similar appearance from不同angles。Fig. A13展示furniture piece front和back views相似。MegaPose错估180度orientation。SAM-6D略好因为用了depth info，但仍然uncertain。

### H.2.3 Depth Discontinuities Due to Occlusions
Hand在target object前面引入depth discontinuities。SAM-6D struggle区分occluder和target的depth，predicted bounding box excludes occluded region。MegaPose因为appearance-based受影响较小但仍reduced accuracy。

### H.2.4 Challenging Viewpoints
Top-down views of furniture legs, partially obscured angles, perspectives that make full scale/shape hard to discern。Both seen and unseen object methods都misinterpret scale或shape。

### H.2.5 Success Cases (for contrast)
Full object visibility + minimal occlusion时两个method都perform better。

**Broader intuition**: Furniture比small manufactured objects有less-defined shapes，silhouette-based approaches less reliable。Wide range of real-world challenges (partial visibility, occlusions, ambiguous views, diverse viewpoints) test limits of current algorithms。Furniture assembly的unique combination of challenges (large, multi-part, complex spatial relationships) makes it excellent proxy for wider real-world pose estimation problems。

---

## 7. 与IKEA-Manual的Key Differences (Appendix J)

1. **Detailed 3D motion information**: IKEA-Manual只有static poses for each step，我们有dense temporal sampling
2. **Diverse real-world environments**: Internet videos vs IKEA-Manual的controlled images
3. **Video-based part assembly task**: 我们用temporal info from videos，IKEA-Manual只依赖3D part info
4. **Consistency in camera information**: 我们确保within segment consistent，IKEA-Manual每个part单独estimate intrinsic（可能unrealistic relative poses）
5. **Real-world visual grounding challenges**: 同样的task在real images上比manual images上难得多

---

## 8. 我的Intuition Building — Why this paper matters

### 8.1 关于"4D Grounding"
"4D"这个词用得precise：3D空间 + 1D时间。每帧都有6-DoF pose，时间上形成continuous trajectories。这不是简单把3D grounding拼上时间维度，而是要保证temporal smoothness + geometric consistency across frames + relative pose accuracy between parts in same frame。三个约束同时满足才valid。

### 8.2 关于Substep概念
Substep是这篇paper的核心conceptual contribution。Manual给出coarse steps（"把4个leg装到seat上"），但video里是4个独立的装leg动作，每个都是substep。这种hierarchical decomposition让我们既能利用manual的high-level structure，又能capture video的fine-grained dynamics。平均7.59 substeps/step这个数字很重要 — 说明video比manual要fine-grained一个数量级。

### 8.3 关于Annotation的难点
Real-world annotation远比lab annotation难。几个issues叠加：
- Camera change（focal length, multi-camera）
- Part ambiguity（similar-looking, wrongly assembled then relocated）
- Occlusion（hands, other parts）
- 2D-3D correspondence ambiguity（多个3D configs project to same 2D）

Paper的pipeline是 **algorithm-assisted manual annotation** — SAM帮segment，PnP+RANSAC帮initial pose estimate，但最终quality靠human refinement from multiple viewpoints。这种hybrid approach在scale和quality间取得balance。

### 8.4 关于Benchmark Difficulty
所有5个tasks的结果都show SOTA methods表现差。这说明：
- Internet video远比lab data复杂
- 即使给GT mask (pose estimation实验)，6D pose estimation仍难
- 即使给GT mask in first frame (VOS实验)，tracking仍难
- 即使simple heuristic (plan generation)，matching real-world plan仍难

这dataset therefore serves as **a meaningful benchmark** — 它不是被solved的问题，而是open challenge。

### 8.5 关于Limitations
作者承认：
- Dataset规模limited (98 videos) — 不能large-scale training
- 只focus visual + 3D，没有audio或text
- Manual annotation still limits scale
- Furniture-specific，transferability to其他assembly domains未验证
- Current baselines没利用dataset进行advanced model development

### 8.6 与更广领域的connection

**Embodied AI**: 这个dataset直接服务于robotics的assembly tasks。一个robot要assemble IKEA furniture需要：(1) 理解manual的高层plan (2) 在video中ground每个part (3) plan motion to execute assembly。这dataset提供training/eval data for所有三层。

**LLM/VLM for assembly**: GPT-4o在Setting 2的失败说明 VLM 还不能reliable理解assembly process。这正是future work的方向 — train VLM with这种4D grounding data。

**3D generative models**: 有了4D grounding，可以train models生成assembly plans or predict part trajectories — 这是shape assembly领域的新方向。

**Procedural learning**: 跟YouCook2, COIN, EPIC Kitchens等instructional video dataset比，这个dataset有3D grounding，能capture spatial relations和object interactions，而cookingdataset只有2D。

---

## 9. 项目资源

- **Code & Dataset**: https://github.com/yunongLiu1/IKEA-Manuals-at-Work
- **DOI**: https://doi.org/10.5281/zenodo.11623997
- **License**: CC-BY-4.0
- **Built upon**: IKEA-Manual (https://github.com/rwang92/ikea-manual) + IAW (https://arxiv.org/abs/2303.15260)

---

## 10. 相关未在paper中提及但intuitively connected的工作

为了build更完整的intuition，提一些paper没cite但相关的工作：

- **Neural Assembly Graph**: 类似idea在shape assembly里用graph表示assembly process
- **AVD (Assembly Video Dataset)**: 家具assembly video理解相关
- **Behavioral Cloning for assembly**: Robotics里从demo学习assembly policies
- **Diffusion Policy**: 最近的robotic learning method，可能能apply到4D grounding data上学习assembly trajectories
- **LLaMA-VID, Video-LLaVA**: Video-language models可能在assembly plan generation上有用
- **BundleSDF, FoundationPose**: 最新的6D pose tracking方法可能在furniture parts上表现更好
- **Track-Anything**: SAM-based video tracking可能improve VOS results
- **Point-E, Shap-E**: 生成式3D models，可能用于补全occluded parts
- **DROID, Open-X-Embodiment**: Robotics datasets，assembly是其中一个task

Reference for these:
- FoundationPose: https://nvlabs.github.io/FoundationPose/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Open-X-Embodiment: https://robotics-transformer-x.github.io/

---

## 总结

这篇paper的core contribution是 **first dataset to provide 4D grounding of assembly instructions on internet videos** — 把manual + 3D model + real-world video三者spatio-temporally align。36个furniture, 98 videos, 34K annotated frames, 137 steps, 1120 substeps, 6-DoF poses for parts/sub-assemblies in every frame。

Five application experiments都show current SOTA methods struggle — 说明dataset有挑战性且useful作为benchmark。Key challenges identified：occlusions, varying viewpoints, similar parts, partial visibility, camera changes, ambiguous 2D-3D correspondence。

**Intuition for why this matters**: Real-world assembly理解是embodied AI的core problem。Robot要assemble furniture需要bridging high-level planning (manual) + mid-level grounding (video→3D) + low-level control (motion)。这dataset提供evaluating和training data for中间这层grounding，而这层一直被existing datasets忽视。作者自己也说future work应该augment with更多modalities和develop algorithms leveraging instructional videos for 3D-grounded assembly plans — 这是一个rich future direction。
