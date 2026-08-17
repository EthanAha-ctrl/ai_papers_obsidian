---
source_pdf: Can Generative Video Models Help Pose Estimation.pdf
paper_sha256: dc498f11f39bc78adeea1cfa3fe125d1b9ecd4055b1002865fe28fe894089ccd
processed_at: '2026-08-03T14:52:24-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这paper到底在干嘛

## 一句话版本

两张照片拍的是同一个场景但几乎没重叠，传统pose estimator全跪了，于是作者想了个招——让video model在中间"脑补"几帧过渡画面，这些假帧居然能帮pose estimator干活。

## 为啥这事是个问题

想象你拍了两张教室照片。一张拍的是教室左半边，另一张拍的是右半边。两张之间几乎没有visual overlap。你问computer：这两张照片的相对camera pose是什么？

传统方法（SIFT、LOFTR这些）的逻辑是：先找两张图里共同出现的feature points，比如同一个桌角、同一块黑板，然后match它们，用eight-point algorithm算essential matrix。没有overlap就等于没有correspondences，这套路直接报废。

DUSt3R呢？它在Habitat、MegaDepth这些3D dataset上train了transformer，能直接从image pair预测pointmap，再从中解出pose。听起来很厉害，但它本质还是依赖某种implicit correspondence——毕竟pointmap就是per-pixel的3D坐标，没overlap的时候它也瞎猜。

关键bottleneck在哪？**3D data太少了**。DUSt3R train的那些dataset加起来也就那么点scene diversity。但video data呢？web上有billions级的video，scene diversity甩3D dataset几百条街。

## Aha moment

作者的核心insight：video model（Sora、Runway、Luma这些）在billions级video上train过，它内部已经encode了极强的visual world prior。你给它两张照片让它interpolate，它其实在做一件事——**用自己学到的world knowledge去"解释"这两张照片之间的空间关系**。

这跟人类怎么判断那张教室照片是一个道理。我们不靠feature matching，我们靠prior knowledge：教室长啥样、桌子怎么摆、拍到左半边桌子之后再往右拍会拍到什么。Video model也在做类似的事，只不过它输出的是pixel而不是reasoning。

## 咋做的

Pipeline简单得惊人：

**Step 1**: 给两张input image，用GPT-4o生成两句caption描述scene。为啥要caption？因为video model是text-to-video的，给它一个description能让生成的video更plausible。

**Step 2**: 把两张图丢给video model（DynamiCrafter开源版，或Runway/Luma商业版），让它生成一段从image A过渡到image B的video。比如14帧的video，A是第一帧，B是最后一帧，中间12帧是model"脑补"的。

**Step 3**: 把这14帧全部丢给DUSt3R。DUSt3R本来是pairwise的，但作者有个extension能处理N张image。它对每张image预测pointmap，然后jointly optimize所有camera pose。

**Step 4**: 从优化结果里提取A和B之间的relative pose。

听起来简单，但有个huge problem：**video model经常生成垃圾**。

## Video model的失败模式

Fig.2展示了几个典型failure：
- 一个microwave突然从sink上方冒出来（object appearing out of nowhere）
- 图像morph without geometric consistency（物体形状变但不符合3D transformation）
- Appearance drift（物体外观在帧间不自然变化）
- Shot cut（突然镜头切换，根本不是smooth transition）

这些artifacts会让DUSt3R被带偏。如果video有shot cut，DUSt3R可能给出完全错误的pose。

## Self-consistency score——这paper的真正贡献

作者的解法特别elegant。核心idea：**如果一个video真的3D-consistent，那我从它里面随机抽不同subset of frames去跑pose estimation，结果应该差不多**。反之如果video有artifacts，不同subset会给出完全不同的pose prediction。

具体操作：

对每个生成的video，随机sample 11个不同的5-frame subsets（每次都包含原始A和B这对anchor）。每个subset丢给DUSt3R，得到一个pose prediction。于是每个video产生11个pose estimate。

然后算**medoid distance**：在11个pose里找一个"中心点"（medoid，类似centroid但必须是实际存在的点），使得它到其他10个pose的平均distance最小。这个最小平均distance就是 $D_{med}$。

$D_{med}$ 小 = 11个prediction聚在一起 = video一致性好 = 可以trust
$D_{med}$ 大 = 11个prediction散开 = video有artifacts = 不能trust

但还有个degenerate case：万一某个video有systematic bias，11个prediction都聚在一起但全错了呢？比如video model一直生成某个错误方向的motion，DUSt3R一致地给出错误pose。

作者的fix：加个bias term。medoid pose不应该离"只用原始A、B两帧跑DUSt3R的结果"太远。这相当于说：video可以refine pose，但不能radically推翻原始prediction。

最终score = $D_{med}$ + (medoid pose到原始pair pose的distance)

选score最低的video，输出它的medoid pose。

## 为啥要生成4个video

每个image pair，作者生成4个video：
- 2个caption × 2个ordering（A→B和B→A）

为啥要swap ordering？因为video model有**left-to-right bias**。不管你给它A→B还是B→A，它都倾向生成rightward panning。swap ordering能force model生成不同方向的camera motion hypothesis，增加找到正确motion的概率。

Fig.6展示了这个bias——同一个pair，A→B和B→A生成的video看起来camera motion几乎一样。这有点像diffusion model sampling，你需要multiple samples来cover distribution。

## 结果如何

几个key numbers：

**ScanNet（indoor，outward-facing，最难）**：
- DUSt3R baseline: MRE 21.31°, MTE 24.72°
- Ours (Dream Machine + medoid): MRE 17.65°, MTE 15.88°
- 提升明显，rotation error降3.7°，translation error降8.8°

**Cambridge Landmarks（outdoor，rotation-dominant）**：
- DUSt3R: MRE 13.28°
- Ours (Runway): MRE 10.78°
- 降2.5°

**DL3DV-10K（outdoor，center-facing）**：
- DUSt3R: MRE 10.72°（本来就好，因为center-facing有overlap）
- Ours: MRE 9.13°
- 提升小但仍consistent improvement

**Oracle（upper bound）**：
- ScanNet: MRE 5.80°
- 跟所有方法的实际performance差huge gap

Oracle的含义：在所有4个video × 11个subset = 44个pose prediction里，挑一个跟ground truth最接近的。这说明video model**确实能生成informative frames**，只是我们的selection heuristic太粗糙，找不到那个最好的prediction。

## Average vs Medoid——为啥naive averaging会hurt

ScanNet上Dream Machine的对比：
- DUSt3R baseline: MRE 21.31°
- Average all predictions: MRE 21.85°（比baseline还差！）
- Medoid selection: MRE 17.65°

Average会hurt的原因：有些video是纯garbage，它的pose prediction完全错误。把这些garbage prediction跟good prediction一起average，结果被拉偏。Medoid selection相当于先filter掉garbage video，只从consistent video里选。

这paper用最simple的heuristic（medoid distance + bias term）做到了consistent improvement。但Oracle的gap告诉我们：一个learned scorer（比如train一个小network来predict "这个video的pose prediction准不准"）可能大幅提升performance。

## 这paper真正的big picture

这paper让我想到一个更大的趋势：**generative model正在变成implicit world prior的query interface**。

传统computer vision pipeline：hand-crafted features → learned features → end-to-end networks。每一步都显式modeling geometry、correspondence、structure。

这paper展示了一个新范式：**不显式modeling geometry，而是query一个generative model来"幻觉"出geometry**。Video model生成intermediate frames，本质上是在回答"这两张照片之间的3D transformation最plausible是什么"。

类似的思路在别的地方也出现：
- DDRM、DPI用diffusion model作为inverse problem的prior
- Sora被OpenAI称为"world simulator"
- Genie用generative model做environment for agent learning

Karpathy你自己常说"the implicitly learned representation is where the magic happens"。这paper就是活例子——video model从未被explicitly taught geometry，但它在billions frames上学到的implicit prior比explicit geometry modeling更强。

## 一些开放问题

1. **Can we do better than medoid?** Oracle的巨大gap暗示yes。一个learned scorer用contrastive learning on (video, pose accuracy) pairs可能work。或者更激进——把video model和pose estimator joint train，让video model学会生成geometrically useful frames。

2. **Why does 5 frames work best?** Table 5的ablation显示2+3（original pair + 3 generated）最优，更多frames反而degenerate。可能因为更多frames让DUSt3R的post-optimization overweight generated frames而忽略anchor frames。这有点像diffusion sampling里steps太多反而artifacts多。

3. **Is left-to-right bias fixable?** Fig.6的bias很有趣。可能video training data里rightward pan比较多，或者text prompt有subtle bias。Future work可以explore de-biasing video model for geometric tasks。

4. **Extreme non-overlap（yaw > 110°）全fail**：Table 7显示yaw [110°, 180°]时DUSt3R MRE=105°，Ours=108°。Video prior也救不回来。这说明video model的prior也有limit——当两张照片的scene overlap完全为零且context clues也弱时，连人类都难判断relative pose。

5. **Cost问题**：$5,500跑300 pairs × 4 datasets。这限制了evaluation scale。如果video model inference能快100x，可能可以sample更多videos、更多subsets，让medoid selection更robust。

## 我觉得最clever的地方

公式9的bias term。这个设计特别elegant。作者没有单纯信任self-consistency——因为"consistently wrong"是可能的。加一个到原始pair prediction的anchor，相当于说：video可以refine pose estimate，但必须stay close to the original estimate。

这跟Bayesian inference的logic一致——prior（原始pair prediction）+ likelihood（video consistency）→ posterior（medoid）。只不过这里是heuristic combination而非principled Bayesian。

另一个clever的地方是swap ordering。如果video model有directional bias，那你给它B→A时，它生成的motion方向可能跟A→B时一样（都rightward pan）。但A→B和B→A的ground truth motion方向是相反的。所以swap ordering实际上是在sampling不同的motion hypothesis distribution。这有点像data augmentation——通过input transformation来cover output distribution的多样性。

## 给你的take-away

这paper的contribution不在method complexity——medoid distance这种东西 undergrad都能想到。真正的contribution是**demonstrate了一个counterintuitive的事实：generative video model内部encode了geometrically useful prior，可以被query来帮助3D vision task**。

这开启了一个research direction：把generative model当作scene prior的oracle，而不是当作output generator。Pose estimation只是testbed，同样的idea可以用到3D reconstruction、depth estimation、optical flow、scene completion等等。任何需要"填补missing information"的inverse problem都可以query generative model。

而且这paper的finding暗示了一个uncomfortable truth：**我们花大量精力collect和label 3D data（Habitat、MegaDepth、ScanNet、DL3DV），但video model在unlabeled web video上学到的东西可能比这些labeled 3D data更有用**。这跟LLM的story一模一样——self-supervised learning on web data beat supervised learning on labeled data。3D vision可能也在走同一条路。

Project page: https://inter-pose.github.io

---

# Can Generative Video Models Help Pose Estimation? - 深度解析

## 1. Motivation与Core Insight

这篇paper的核心insight非常优雅。当人类观察两张几乎无重叠的classroom照片时，我们能通过prior knowledge推断它们的空间关系——例如，左图中桌子右边缘对应右图中桌子左边缘。传统pose estimation方法依赖visual correspondences（如SIFT、LOFTR），在non-overlap场景下完全失败。即使是state-of-the-art的DUSt3R（基于CroCo pre-training的transformer，在Habitat、MegaDepth等大规模3D dataset上训练）也struggle，因为3D data的diversity远不如web-scale video data。

**Key insight**：Generative video models（如DynamiCrafter、Runway Gen-3、Luma Dream Machine）在billions级video data上训练，内部encode了强大的visual world prior——它们能生成plausible的camera motion、reflection、dynamic interactions。如果我们让video model在两张input images之间"幻觉"出intermediate frames，这些frames就提供了一种scene explanation，相当于让pose estimator有了"视觉桥梁"。

**Project page**: https://inter-pose.github.io

## 2. Method - InterPose Pipeline详解

### 2.1 Problem Formulation

给定两张images $I_A$ 和 $I_B$，ground truth world-to-camera transforms为：

$$T_A = \begin{bmatrix} R_A & t_A \\ 0 & 1 \end{bmatrix}, \quad T_B = \begin{bmatrix} R_B & t_B \\ 0 & 1 \end{bmatrix}$$

其中 $R_A, R_B \in SO(3)$ 是rotation matrices，$t_A, t_B \in \mathbb{R}^3$ 是translation vectors。目标是恢复relative pose：

$$T_{rel} = T_B T_A^{-1}$$

展开得到：
- Relative rotation: $R_{rel} = R_B R_A^{-1}$ (注意paper中写作 $R_B^{-1}R_A^{-1}$ 看起来是typo，标准定义应该是 $R_B R_A^T$)
- Relative translation: $t_{rel} = t_B - R_{rel} t_A$

### 2.2 Pose Distance Metrics

公式2定义total distance：

$$\text{dist}(T_1, T_2) = \text{dist}_R(R_1, R_2) + \text{dist}_t(t_1, t_2)$$

**Rotation error（公式3）**：
$$\text{dist}_R(R_1, R_2) = \arccos\left(\frac{\text{Trace}(R_2 R_1^\top) - 1}{2}\right)$$

变量含义：$R_1^\top$ 是 $R_1$ 的转置（inverse for rotation matrices）；$\text{Trace}(\cdot)$ 是矩阵迹；这个公式源自 $R_2 R_1^\top$ 的eigenvalue性质——两个rotation matrices的乘积仍是rotation matrix，其trace = $1 + 2\cos\theta$，其中 $\theta$ 是相对旋转角。除以2再arccos就得到geodesic distance on $SO(3)$。

**Translation error（公式4）**：
$$\text{dist}_t(t_1, t_2) = \arccos\left(\left|\frac{t_1}{\|t_1\|} \cdot \frac{t_2}{\|t_2\|}\right|\right)$$

这里 $t_1, t_2$ 先归一化为单位向量，再求dot product。绝对值符号使得direction"对称"（即 $t$ 与 $-t$ 视为相同方向），这处理了pose estimation中scale ambiguity和direction ambiguity问题。

### 2.3 Pose Estimator as Black Box

公式5定义pose estimator：

$$f_{\text{pose}}(\{I_A, I_B, I_1, \ldots, I_{N-2}\}) = \hat{T}_B \hat{T}_A^{-1} = \hat{T}$$

这里实际使用DUSt3R的multi-image extension。DUSt3R backbone预测per-image pointmaps $\hat{X}^{i,j} \in \mathbb{R}^{H \times W \times 3}$（在camera coordinate of image $i$ 下表示image $j$ 的3D points），然后通过global alignment optimization（基于Levenberg-Marquardt）联合优化camera poses和point cloud positions。

### 2.4 Generative Video Model

公式6：

$$f_{\text{vid}}(I_A, I_B, p) = [I_1, I_2, \ldots, I_N]$$

其中 $I_1 = I_A$, $I_N = I_B$，$p$ 是text prompt。论文测试3个模型：
- **DynamiCrafter**: 开源，基于text-to-video diffusion，在WebVid10M上finetune，输出16 frames @ 320×512
- **Runway Gen-3 Alpha Turbo**: 商业，112 frames @ 1280×768
- **Luma Dream Machine**: 商业，114 frames @ ~1MP

### 2.5 Self-Consistency Score（核心创新）

Video model有较大variance，且会产生artifacts（morphing、shot cuts、object appearing/disappearing，见Fig.2）。论文的策略：生成 $n=4$ 个videos（2 prompts × 2 orderings），每个video采样 $m=11$ 个subsets（10次random + 1次uniform spacing），每个subset包含 $k=5$ frames（2 original + 3 generated）。

公式7：
$$f_{\text{pose}}(\{I\}^{(i)}) = \hat{T}^{(i)}$$

对每个subset预测一个relative pose。

公式8 - **Medoid Distance**：
$$D_{\text{med}} = \min_i \frac{1}{m-1} \sum_{j \neq i} \text{dist}(\hat{T}^{(i)}, \hat{T}^{(j)})$$

变量含义：
- $i$ 是候选medoid index
- $j$ 遍历其他所有samples
- $\min_i$ 选择使average distance最小的那个pose作为medoid
- medoid类似centroid，但必须是数据集中实际存在的点（类似k-medoids clustering）

Intuition：如果video是3D-consistent的，那么不同frame subsets应该给出相似的pose prediction，medoid distance小；如果video有shot cut或morphing，不同subsets会给出divergent predictions，medoid distance大。

公式9 - **Total Distance with Bias Term**：
$$D_{\text{total}} = D_{\text{med}} + \text{dist}(\hat{T}_{\text{med}}, f_{\text{pose}}(\{I_A, I_B\}))$$

第二项是medoid pose与"仅用原始pair预测的pose"之间的distance。这是为了防止degenerate case：如果一个video一直"自信地"给出错误prediction（例如总是180° off），medoid distance会低，但与原pair prediction偏差大。

### 2.6 最终Pipeline

1. GPT-4o生成2个caption描述scene
2. 对4个组合（2 captions × 2 orderings）生成4个videos
3. 对每个video，sample 11个5-frame subsets
4. 每个subset输入DUSt3R获得pose
5. 计算每个video的 $D_{\text{total}}$
6. 选择 $D_{\text{total}}$ 最低的video，输出其 $\hat{T}_{\text{med}}$

**为何要swap image order**：Fig.6展示了video model有left-to-right panning bias——无论input是A→B还是B→A，模型都倾向生成rightward pan。Swapping order提供diverse camera motion hypotheses。

## 3. 实验设计深度解析

### 3.1 Benchmark Construction

论文构建了4个datasets的challenging subsets，按yaw angle range筛选：

| Dataset | Type | # Pairs | Yaw Range | Characteristics |
|---------|------|---------|-----------|------------------|
| Cambridge Landmarks | Outdoor, scene-scale | 290 | [50°, 65°] | Rotation-dominant, minimal translation |
| ScanNet | Indoor, scene-scale | 300 | [50°, 65°] | Outward-facing viewpoints |
| DL3DV-10K | Outdoor, center-facing | 300 | [50°, 90°] | POI videos, large frustum overlap |
| NAVI | Object-centric, center-facing | 300 | [50°, 90°] | Multi-device capture |

**关键区别**：Outward-facing datasets（Cambridge, ScanNet）的相邻frames frustum overlap少；center-facing datasets（DL3DV, NAVI）由于摄像头始终指向同一POI，即使yaw变化大也有较多overlap。

### 3.2 Quantitative Results - Table 1 (Outward-facing)

以ScanNet为例，关键数据：

| Method | Input | MRE↓ | MTE↓ | R@5° | R@30° | t@30° | AUC30° |
|--------|-------|------|------|------|-------|-------|--------|
| SIFT+NN | Pair | 112.95 | 48.99 | 2.06 | 23.02 | 31.62 | 1.82 |
| LOFTR | Pair | 64.46 | 45.49 | 8.33 | 28.33 | 35.33 | 6.43 |
| DUSt3R | Pair | 21.31 | 24.72 | 65.33 | 79.00 | 73.67 | 60.34 |
| Ours(DC)-Avg | Video | 19.97 | 18.87 | 62.33 | 83.00 | 74.33 | 58.84 |
| Ours(DC)-Medoid | Video | 18.96 | 16.42 | 68.00 | 84.33 | 80.33 | 62.14 |
| Ours(DM)-Medoid | Video | 17.65 | 15.88 | 68.67 | 85.33 | 82.33 | 63.06 |
| Oracle | All | 5.80 | 5.00 | 81.33 | 95.00 | 96.67 | 81.19 |

**几个critical observations**：
1. Classic feature matching（SIFT+NN, LOFTR）在non-overlap case几乎完全失败（MRE > 60°）
2. DUSt3R大幅超越classical methods（MRE 21.31°），但仍不理想
3. **Average strategy有时甚至hurt performance**：Dream Machine的Avg得到MRE=21.85° > DUSt3R的13.28° in Cambridge，因为低质量videos拉低平均
4. **Medoid selection有效**：Dream Machine Avg→Medoid从21.85°降到11.96°
5. **Oracle远超所有方法**：说明video model确实能生成有用frames，但selection strategy还有巨大改进空间

### 3.3 Table 2 (Center-facing)

DL3DV-10K上DUSt3R baseline表现已经很强（MRE=10.72° vs ScanNet的21.31°），因为center-facing天然有overlap。Ours仍然有slight improvement：MRE 10.72° → 9.13° (Dream Machine)。

### 3.4 Ablation: Distance Metrics (Table 4)

比较 $D_{\text{total}}$, $D_{\text{med}}$, $D_{\text{bias}}$ 单独使用：
- Cambridge + Dream Machine: 仅用 $D_{\text{med}}$ 得到MRE=19.37°，加 $D_{\text{bias}}$ 后降到11.96°
- 这证实了bias term的必要性——防止video"自信但错误"的degenerate case

### 3.5 Ablation: Number of Input Images (Table 5)

ScanNet上Dream Machine的实验：
- 2+0 (DUSt3R baseline): MRE=21.31°
- 2+1: MRE=20.41°
- **2+3 (main paper setting): MRE=17.65°** ← 最优
- 2+8: MRE=17.98°
- 2+38: MRE=18.43°
- 2+114: MRE=17.77°

Oracle也呈现degenerate trend：2+3时Oracle MRE=5.80°，2+114时Oracle MRE=9.21°。原因：more frames意味着less randomness in sampling，且post-optimization可能overweight generated frames而忽略original pair。

### 3.6 Yaw Angle Breakdown (Tables 6-9)

ScanNet上：
- [0°, 50°] overlap pairs: DUSt3R MRE=11.33°, Ours MRE=9.12° (improvement)
- [65°, 180°] non-overlap: DUSt3R MRE=83.48°, Ours MRE=83.94° (slightly worse on rotation, but MTE improved 58.93°→37.81°)
- [110°, 180°] extreme non-overlap: Both methods catastrophically fail (MRE > 100°), 说明video prior也无法处理完全non-overlap的极端case

DL3DV-10K上即使[90°, 180°]也work（MRE 19.20°→16.06°），因为center-facing数据inherently有overlap。

### 3.7 MASt3R Results (Table 3, 10, 11)

MASt3R是DUSt3R的follow-up，增加了feature matching head。它在center-facing datasets上表现优异（DL3DV MRE=4.13° vs DUSt3R的10.72°），但在outward-facing上反而更差（Cambridge MRE=36.55° vs DUSt3R的13.28°）——因为MASt3R依赖correspondences，在non-overlap case反而被误导。

Ours + MASt3R在Cambridge上仍然improvement：MRE 36.55° → 27.47° (Dream Machine)，验证方法的generalization across pose estimators。

## 4. Limitations与Future Directions

1. **Cost**: 论文花了$5,500在commercial video models上，限制了evaluation scale到300 pairs/dataset
2. **Multi-view consistency不保证**: Video model可能在帧间morph object appearance
3. **Selection strategy suboptimal**: Oracle远超medoid selection，说明better selection method存在大量research opportunity——可能的方向包括learned scoring model、optical flow consistency check、Scene coordinate network verification
4. **Sensitivity**: Video models对prompt wording、camera intrinsics、aspect ratio高度敏感
5. **Extreme non-overlap failure**: Yaw > 110°时所有方法都失败，video prior也难以bridge完全无visual overlap的case

## 5. 与Related Work的联系

### 5.1 DUSt3R Lineage
- CroCo (Weinzaepfel et al.): cross-view completion pretraining
- DUSt3R (Wang et al., CVPR 2024): pointmap prediction
- MASt3R (Leroy et al., ECCV 2025): 加入feature matching head

### 5.2 Probabilistic Pose Estimation
- RelPose (Zhang et al., ECCV 2022): factorized distribution over rotations
- RelPose++ (Lin et al.): sparse-view extension
- PoseDiffusion (Wang et al., ICCV 2023): diffusion-based bundle adjustment
- Cameras as Rays (Zhang et al.): ray diffusion
- PF-LRM (Wang et al.): pose-free large reconstruction

### 5.3 Video Generation
- Early: GAN-based (MoCoGAN, SVG)
- Diffusion-based pixel space: Video Diffusion Models, Imagen Video, Make-A-Video
- Latent diffusion: Stable Video Diffusion, Align-Your-Latents, Lumiere, Phenaki
- Image animation/interpolation: DynamiCrafter (本文开源option)

### 5.4 概念上的联系
这paper的哲学与Sora的"world simulator"概念呼应——generative model作为implicit world model。similar in spirit to:
- V-JEPA (Meta): video作为self-supervised representation
- Genie (Google): generative environment for agent learning
- Diffusion as prior for inverse problems (e.g., DDRM, DPI)

## 6. Personal Thoughts on Intuition

这篇paper给me最大的intuition shift是：**generative model不仅是output tool，更是implicit scene prior的查询接口**。当video model生成intermediate frames时，它实际上在回答"这两张图之间最plausible的3D scene transformation是什么"这个query。这与传统pose estimation显式建模geometry的方式完全不同。

公式9的bias term特别精彩——它隐式假设了"original pair prediction虽然不完美但大致direction正确"，所以medoid不应该偏离太远。这种heuristic虽简单，但effectively避免degenerate failure mode。

Oracle的巨大gap暗示：如果未来能train一个learned video scorer（可能用contrastive learning on (video, pose consistency) pairs），performance可以大幅提升。或者更激进地，可以尝试joint training: video model + pose estimator end-to-end，让video model学会生成geometrically useful frames。

另一个有趣的extension：是否可以用multiple generated videos做ensemble pose estimation，类似diffusion model的classifier-guided sampling？即用pose estimator的gradient guidance来guide video generation process，使生成的frames不仅plausible而且geometrically informative。

## Reference Links

- Project page: https://inter-pose.github.io
- DUSt3R: https://dust3r.europe.naverlabs.com
- MASt3R: https://github.com/naver/mast3r
- DynamiCrafter: https://github.com/Doubiiu/DynamiCrafter
- Runway Gen-3: https://runwayml.com/product
- Luma Dream Machine: https://lumalabs.ai/dream-machine
- CroCo pretraining: https://github.com/naver/croco
- ScanNet: http://scannet.cs.princeton.edu
- DL3DV-10K: https://github.com/DL3DV-10K/Dataset
- NAVI: https://navi-dataset.info
- Cambridge Landmarks (PoseNet): https://github.com/alexgkendall/opencv_map_server
- LOFTR: https://github.com/zju3dv/LoFTR
- LightGlue: https://github.com/cvg/LightGlue
- SuperPoint: https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork
- Sora (world simulators): https://openai.com/research/video-generation-models-as-world-simulators
- V-JEPA: https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
- Genie: https://sites.google.com/view/genie-2024
- Stable Video Diffusion: https://stability.ai/news/stable-video-diffusion-open-ai-video-model
- PoseDiffusion: https://github.com/google/diffusion-mat
- RelPose: https://github.com/dcharatan/relpose-pp
- COLMAP: https://colmap.github.io
- WebVid10M: https://maxbain.com/webvid/
- GPT-4: https://openai.com/research/gpt-4
