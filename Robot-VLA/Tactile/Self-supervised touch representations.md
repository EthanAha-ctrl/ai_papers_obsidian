---
source_pdf: Self-supervised touch representations.pdf
paper_sha256: 033b3ea13ff404e42bd352df5a424642cde564a6f57b1534840d50fc00c36ac7
processed_at: '2026-08-12T05:00:36-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲Sparsh

## 一句话版本

Meta的FAIR团队把computer vision里那套self-supervised pretraining的玩法搬到了touch sensor上，搞了一堆通用touch encoder，发现不用label就能学到很好的touch representation，在6个task上平均吊打从scratch训练的model 95%。

---

## 为什么要做这件事

想象你在搞robot manipulation，你想让robot能抓东西、插东西、转东西。光靠camera不够，因为occlusion、lighting、contact这些信息camera看不到。所以你给robot手指头上装了tactile sensor——就是那种gel做的、里面有个小camera拍gel变形的sensor，叫vision-based tactile sensor。

现在问题来了：你换了三种sensor——DIGIT、GelSight 2017、GelSight Mini，它们的optics不一样、有的有markers有的没有、分辨率不同、lighting不同。然后你要做force estimation、slip detection、pose estimation、grasp stability、textile recognition、bead maze六个task。

传统做法是每个task每个sensor配一对，训练一个custom model。你有3 sensors × 6 tasks = 18个model要train。而且touch data的label极其难收集——force要F/T sensor标定，slip要friction cone模型，pose要ArUco marker追踪，每个label都是真金白银的lab time。

这就很painful了。所以作者想：CV里ImageNet pretraining + downstream fine-tuning的paradigm这么成功，touch能不能也这么搞？

答案是能，但有坑。

---

## 核心思路：把SSL搬到touch

作者试了5种SSL方法，都是ViT-B/14 backbone，在460k张unlabeled touch image上pretrain：

1. **MAE** - 遮住75%的patch，让model重建pixel
2. **DINO** - student-teacher self-distillation，latent space对齐
3. **DINOv2** - DINO + iBot的组合
4. **I-JEPA** - 在latent space做masked prediction
5. **V-JEPA** - video版的JEPA，处理4帧clip

这5个方法的区别本质上是**在哪里做prediction**：

- MAE在pixel space重建，要把gel deformation的每个pixel细节都reconstruct出来
- DINO在latent space做self-distillation，student和teacher看同一张图的不同crop，output distribution要对齐
- JEPA在latent space做predictive learning，context network看masked image，predict target network对local crop的latent embedding

---

## Touch和Vision的本质区别

这是这篇paper最valuable的insight，我来详细说说。

**Touch image不是natural image。** 你拍的cat photo有semantic content——cat的shape、texture、背景。但touch image是gel deformation的indirect measurement，它长什么样完全取决于sensor的optics、lighting、gel的物理特性。

所以touch image有几个vision没有的问题：

**第一，noise profile完全不同。** Touch image里有sensor manufacturing带来的lighting variation、gel aging、marker placement discrepancy。这些是distractor，不是你要学的content。MAE在pixel space重建，会把capacity浪费在reconstruct这些noise上。DINO和JEPA在latent space学习，能filter掉这些noise。这就是为什么paper里latent-space SSL整体优于pixel-space SSL。

**第二，temporal dimension是必需的。** Vision的SSL大多用single image就够了。但touch信号里slip、pose change这种dynamic property，单帧根本看不出来。你需要~80ms的历史。为什么是80ms？因为human neurophysiology研究（Zangrandi et al. 2021）发现人类检测partial slip后调整grip force的反应时间就是80ms。作者把两帧image沿channel拼接（stride=5，60FPS对应80ms），或者用V-JEPA处理4帧clip（~100ms）。

**第三，cross-sensor variation巨大。** DIGIT是markerless + RGB LED，GelSight 2017有markers，GelSight Mini是markerless + HD。同样是"按压5N normal force"，三个sensor拍出来的image完全不一样。传统方法每个sensor要train一个model。Sparsh通过background subtraction + multi-sensor pretraining，让representation跨sensor transferable。Appendix E.3的10-shot cross-sensor实验是关键evidence：GelSight上train的textile classifier，用10个DIGIT sample adapt，Sparsh (DINO)达到61.8% accuracy，E2E只有10.9%。

---

## 每个SSL方法的直觉

### MAE

$$\mathcal{L}_{\mathrm{MAE}} = \|\mathbf{I}_{\mathrm{target}} - \mathbf{I}_{\mathrm{recon}}\|_2^2$$

$\mathbf{I}_{\mathrm{target}}$是被mask的原始pixel，$\mathbf{I}_{\mathrm{recon}}$是decoder重建的pixel。L2 norm平方就是MSE。

直觉：遮住一大块，让model猜被遮住的gel deformation长什么样。这强迫model学习gel的spatial structure和deformation pattern。

问题：touch image里很多pixel variation是sensor noise，不是task-relevant signal。MAE会花capacity去reconstruct这些noise。

结果：MAE在textile recognition上反而最强（0.599 vs DINO 0.527），因为textile是pixel-level texture task，MAE的pixel reconstruction恰好capture了texture的fine-grained appearance。但在force estimation、pose estimation这种physics task上不如latent-space方法。

### DINO

$$\mathcal{L}_{\mathrm{DINO}} = -\sum \mathbf{p}_t \log \mathbf{p}_s$$

$\mathbf{p}_s$是student的softmax probability，$\mathbf{p}_t$是teacher（EMA updated）的probability。这是cross-entropy form，但target是teacher的soft prediction。

直觉：student和teacher看同一张touch image的不同crop（比如student看global crop，teacher看local crop），它们的output distribution要对齐。这就让model学习"不管怎么看这张touch image，semantic content应该一样"。

关键trick：teacher是student的EMA copy，stop gradient through teacher，避免degenerate solution。

结果：DINO在force estimation和pose estimation这种physics-based regression task上最强。DIGIT force estimation RMSE 36.09 mN（full data），1/3 data下44.03 mN，而E2E在1/3 data下退化到61.42 mN。

### I-JEPA

$$\mathcal{L}_{\mathrm{jepa}} = \sum_{i \in M} \sum_{j \in B_i} \|\hat{\mathbf{s}}_{y_j} - \mathbf{s}_{y_j}\|_2^2$$

$M$是context masks集合，$B_i$是第$i$个context对应的target crops集合。$\hat{\mathbf{s}}_{y_j}$是context network + predictor预测的target representation，$\mathbf{s}_{y_j}$是target network（EMA teacher）对crop $y_j$的实际latent embedding。

直觉：context network看masked image的大块context，predict被mask区域的latent representation。Target network看完整image的local crop，提供target embedding。Loss是latent space L2 distance。

和MAE的区别：MAE reconstruct pixel，JEPA predict latent embedding。JEPA不需要reconstruct gel deformation的pixel细节，只需要predict semantic-level representation。

和DINO的区别：DINO用multi-crop + self-distillation，JEPA用masked prediction。JEPA的masking strategy是spatial block masking（I-JEPA）或tube masking（V-JEPA）。

结果：I-JEPA在grasp stability（0.802）和textile recognition上表现好，适合semantic understanding task。

### V-JEPA

和I-JEPA一样，但处理4帧video clip而不是single image。用tube masking（temporal + spatial masking）。

结果：V-JEPA在slip detection上碾压所有method，F1=0.820，而且1/100 data下还有0.760。这是因为slip本质上是temporal event——你需要看几帧才能判断是不是在slip。V-JEPA的4帧clip直接建模temporal dynamics。

代价：inference慢一倍（60 FPS vs 112 FPS），因为要处理video。

---

## 数据集的故事

作者curated了4个dataset，总共661k images，70%用于pretrain：

1. **YCB-Slide** (Suresh et al. CoRL 2022) - 180k frames，DIGIT，10个YCB物体sliding
2. **Touch-and-Go** (Yang et al. 2022) - 220k frames，GelSight，human in-the-wild contact
3. **ObjectFolder** (Gao et al. CVPR 2022) - 81k frames，robot discrete contact
4. **Touch-Slide** (new) - 180k frames，DIGIT，9个toy-kitchen物体

关键：这些data都是unlabeled的，只用SSL pretraining。Labeled data是另外collect的，用于TacBench evaluation。

460k这个数字听起来不小，但和ImageNet的1.4M、DINOv2的142M比还是小得多。Touch data的收集成本远高于web image crawling，所以这个scale已经是significant effort了。

---

## TacBench的6个Task

作者设计了6个task，从low-level properties到high-level manipulation：

### [T1] Force Estimation

预测3-axis force（normal + 2 shear）。用robot arm把sensor按在indenter上，F/T sensor给ground truth。

数据：75k samples，3种indenter（sphere, sharp, flat），train用sphere+sharp，test用flat。

Metric：RMSE in mN。

关键发现：
- DIGIT上E2E在full data下还能竞争（39.34 vs Sparsh (DINOv2) 29.31），但1/3 data时E2E退化到61.42，Sparsh (DINOv2)只有26.85
- GelSight Mini上E2E即使full data也不行（57.21），因为HD resolution导致contact region小，from-scratch训练困难。Sparsh (DINO)达到20.25，提升64%

直觉：force是continuous regression task，需要precise物理量估计。Latent-space SSL（DINO, DINOv2）在这里最强，因为它们学到了abstraction的contact representation，能generalize across indenter shape。MAE在pixel space重建，对indenter shape的spatial pattern敏感，generalization差一些。

### [T2] Slip Detection

判断sensor是否在slip。用friction cone model标label：shear force magnitude > μ × normal force时slip。

数据：125k samples，只有13%是slip（imbalanced）。

Metric：F1 score（不用accuracy因为imbalance）。

关键发现：
- Sparsh (V-JEPA) F1=0.820，1/100 data下还有0.760
- E2E在1/3 data时F1掉到0.238（接近random）

直觉：slip是temporal event，单帧看不出slip，需要看几帧的force change trend。V-JEPA的4帧video clip天然适合这个task。作者还发现joint training slip detection + force change (Δ) prediction能互相帮助，因为slip和force change高度相关。

### [T3] Pose Estimation

估计object相对sensor的SE(2) pose change (Δx, Δy, Δθ)。用Allegro hand + DIGIT，ArUco tag追踪object pose。

用regression-by-classification：每个DOF离散化到11 bins（log-uniform space），train 3个classification head。

数据：49k samples。

Metric：accuracy。

关键发现：
- Sparsh (DINO) 0.913 accuracy (full data)，1/3 data下0.834
- E2E在1/3 data时崩到0.245（接近random的1/11=0.09）

直觉：pose estimation需要spatial reasoning——你要从gel deformation的pattern推断object往哪个方向移动了。DINO的latent space representation capture了这种spatial structure。E2E在low data下无法区分相邻bin（比如0.5°和1.0°的区别），default到zero或max。

### [T4] Grasp Stability

预测grasp成功还是失败。用Feeling of Success dataset（Calandra et al. 2017），GelSight 2017 with markers，9.3k grasps。

输入'before'和'during'两帧。

关键发现：
- Sparsh (I-JEPA) 0.802 accuracy，只用touch single finger
- 原paper用touch + vision才达到~75%

直觉：grasp stability是semantic understanding task——你要从touch history判断"这个grasp稳不稳"。I-JEPA的latent predictive learning适合这种high-level判断。

### [T5] Textile Recognition

识别20种textile（leather, cotton, polyester等）。用Clothing Dataset（Yuan et al. 2018），GelSight 2017 with markers，4467 video clips。

关键发现：
- Sparsh (MAE) 0.599 accuracy，反超DINO (0.527)和IJEPA (0.506)
- E2E只有0.437

直觉：textile recognition是pixel-level texture task。不同textile的区分在于gel接触时的surface texture pattern，这是fine-grained pixel信息。MAE的pixel reconstruction恰好capture了这种texture detail。DINO和JEPA在latent space学习，丢失了部分pixel-level texture信息。

这个result很重要，因为它说明**没有one-size-fits-all的SSL方法**。不同task有不同的inductive bias，pixel-space和latent-space SSL各有优势。

### [T6] Bead Maze

这是最接近real robot manipulation的task。Robot用DIGIT sensor拿着bead在wire maze上移动。

用Diffusion Policy（Chi et al. RSS 2023），把Sparsh encoder替换原来的CNN encoder。50 demonstrations，~34k training pairs。

关键发现：
- Position error：Sparsh (DINO) 5.54mm vs E2E 8.46mm（full data）
- Real rollout distance：Sparsh (DINO) 10.80cm vs E2E 6.70cm，提升61%
- 但所有model都没完成full maze

意外发现：fine-tuning反而比frozen差（Sparsh (DINO) 10.80 → 8.45）。作者推测是domain mismatch + overfitting。

直觉：bead maze是sequential decision making，compounding error是核心问题。Pretrained representation提供了good touch features，但policy learning本身有covariate shift问题。Fine-tuning破坏了pretrained representation的general features，overfit到narrow task distribution。

---

## 为什么Latent Space SSL > Pixel Space SSL for Touch

这是paper的核心finding，我来build一下intuition。

Touch image的information可以分为两类：
1. **Task-relevant signal**：gel deformation pattern反映的contact geometry、force、slip state
2. **Distractor noise**：sensor manufacturing variation、lighting fluctuation、gel aging、marker placement discrepancy

MAE在pixel space重建，loss treats所有pixel equally。Model必须reconstruct task-relevant signal和distractor noise的sum。如果distractor占pixel variation的大头，model的capacity就被浪费在reconstruct noise上。

DINO和JEPA在latent space学习，loss只要求latent representation对齐。Latent space可以filter掉distractor noise，只保留task-relevant signal。这就是为什么latent-space SSL在大多数task上更强。

例外是textile recognition——这个task的signal本身就是pixel-level texture，和noise在同一个granularity，所以MAE的pixel reconstruction反而有优势。

这个insight和vision domain不同。Vision里MAE很强，因为natural image的pixel variation主要是semantic content（object shape, texture, scene structure），distractor相对少。Touch image反过来，distractor占比高，所以latent-space SSL更 advantageous。

---

## Cross-Sensor Generalization

这是Sparsh的killer feature。作者用了两个trick：

**1. Background Subtraction**

对markerless sensor（DIGIT, GelSight Mini）做background subtraction：用no-contact image作为reference，减掉background。

为什么有用？因为不同sensor instance的background lighting不同。减掉background后，剩下的就是contact-induced deformation，这个是cross-sensor invariant的。

**2. Multi-Sensor Pretraining**

Pretraining data包含DIGIT、GelSight 2017、GelSight Mini三种sensor。Model被迫学习跨sensor invariant的touch representation。

**Evidence：Cross-sensor n-shot transfer (Appendix E.3)**

[T5] Textile decoder在GelSight (markers)上训练，迁移到DIGIT做n-shot evaluation：

| | zero-shot | 1-shot | 5-shot | 10-shot |
|---|-----------|--------|--------|---------|
| Sparsh (DINO) | 9.1% | 19.1% | 28.2% | 61.8% |
| E2E | 3.6% | 0.0% | 15.5% | 10.9% |

Zero-shot时Sparsh (DINO)只有9.1%（接近random的5%），因为GelSight和DIGIT的image appearance差太多。但10-shot就达到61.8%，说明representation已经capture了cross-sensor invariant的textile feature，只需要少量sample adapt decoder。E2E的10-shot只有10.9%，因为它没有cross-sensor prior。

---

## Label Efficiency

这是SSL在robotics的核心价值。作者做了systematic ablation：full, 1/3, 1/10, 1/100的label budget。

**Average improvement of best SSL over E2E: 98.75%**（Table 13）

具体看几个task：

**Force estimation (DIGIT), 1/3 data**：
- E2E: 61.42 mN
- Sparsh (DINOv2): 26.85 mN
- Improvement: 128%

**Slip detection, 1/100 data**：
- E2E: F1 = 0.214
- Sparsh (V-JEPA): F1 = 0.760
- Improvement: 255%

**Pose estimation, 1/3 data**：
- E2E: 0.245 accuracy
- Sparsh (DINO): 0.834 accuracy
- Improvement: 240%

直觉：E2E在low data regime下严重overfit或underfit，因为它要从scratch学习touch image的基本structure。Sparsh已经通过SSL学到了touch的general structure，downstream task只需要学习task-specific的mapping，label需求大幅降低。

这对robotics意义重大——force labeling需要F/T sensor + robot arm + 标定时间，slip labeling需要friction cone模型 + 人工验证，pose labeling需要ArUco tracking + 手动perturbation。每个label都是真金白银。如果SSL能把label需求降到1/3甚至1/100，data collection成本就降一个数量级。

---

## Fine-tuning的Paradox

Appendix E.1做了fine-tuning ablation，发现一个counterintuitive的结果：

**Bead maze task上，frozen Sparsh (DINO)比fine-tuned更好**：
- Frozen: 10.80 cm
- Fine-tuned: 8.45 cm

这和vision pre-training for control（Hansen et al. 2022）的发现一致。直觉是：

Pretrained representation已经capture了broad touch features，这些features对downstream task有用但not task-specific。Fine-tuning会把representation往narrow task distribution push，丢失broad features的generalization能力。

在narrow task domain（single maze pattern）下，fine-tuned model可能overfit到这个maze的specific pattern，test-time遇到slight variation就fail。

这说明**pretrained touch representation的最佳用法可能是frozen + lightweight task-specific decoder**，而不是full fine-tuning。这和NLP里GPT-3的few-shot prompting思路类似——保持backbone frozen，只adapt task head。

但作者也发现regression task（force, pose）在full fine-tuning下能进一步提升，特别是low data regime。所以fine-tuning策略要task-specific：

- Classification task（slip, stability, textile）：frozen就够好
- Regression task（force, pose）：full fine-tuning有额外收益
- Policy learning（bead maze）：frozen反而更好

---

## V-JEPA的Trade-off

V-JEPA在slip detection上碾压所有method，但在其他task上不如I-JEPA。而且inference慢一倍（60 vs 112 FPS）。

直觉：V-JEPA的4帧video clip建模temporal dynamics，对temporal-dependent task（slip）有unique优势。但对static task（force estimation是instantaneous的，textile recognition是single-frame的），video clip的额外temporal information是redundant的，反而可能introduce noise。

而且V-JEPA的inference cost高——处理4帧比1帧慢。Real-time robot manipulation对latency敏感，60 FPS可能不够（robot control loop通常要求100+ Hz）。

所以V-JEPA的value proposition是：如果你的task strongly temporal（slip detection, dynamic event detection），用V-JEPA；如果task是static或weakly temporal，用I-JEPA或DINO就够了。

---

## 对Robotics的Implication

这篇paper的long-term vision是让touch representation成为robot manipulation的foundation model。

具体implications：

**1. Contact-rich manipulation的touch encoder**

Insertion、in-hand rotation、grasping这些task现在可以用Sparsh作为touch encoder，不用从scratch训练。你只需要collect少量task-specific labeled data，train一个lightweight decoder。

**2. Multi-modal fusion**

Sparsh的touch representation可以和vision representation在latent space融合。比如Neural Feels（Suresh et al. 2023, https://arxiv.org/abs/2312.13460）做visuo-tactile perception for in-hand manipulation，如果用Sparsh替换他们的custom touch encoder，可能label efficiency更高。

**3. Tactile-language alignment**

Touch-and-Go（Yang et al. 2022）和Binding Touch to Everything（Yang et al. 2024, https://arxiv.org/abs/2401.18084）做touch-vision-language alignment。Sparsh可以作为更好的touch encoder，提升alignment quality。

**4. Sim2Real**

如果tactile simulator（TACTO, Taxim, DiffTactile）能生成足够realistic的touch image，Sparsh可以在sim data上pretrain，进一步降低real data需求。但目前simulator的realism还不够，特别是shadow和per-sensor-instance discrepancy。

**5. Dexterous manipulation的scaling**

如果Sparsh能scale到10M+ touch images + 更大backbone（ViT-L, ViT-H），可能解锁更复杂的dexterous manipulation task。In-hand rotation、tool use、fabric manipulation这些task都需要rich touch understanding。

---

## Limitations和Open Questions

作者坦诚承认的limitation：

**1. Bead maze没完成full maze**

所有model都只能partial complete，compounding error导致bead掉出来。这暴露了representation learning → policy learning的gap。Pretrained representation在dense prediction task上work，但在sequential decision making上还有距离。

可能的解决方向：
- Temporal ensembling（Zhao et al. 2023, Aloha, https://arxiv.org/abs/2304.13705）平滑policy output
- Force control（目前Franka arm用的position control，bead maze需要force feedback）
- Error recovery mechanism（一旦bead松动，如何重新grasp）

**2. Dataset偏discrete contact**

现有dataset大多是press-and-release或short sliding，shear interaction不够丰富。Real manipulation有大量continuous shear、torsional slip、multi-finger contact，这些scenario在pretraining data里underrepresented。

**3. Temporal history length没ablate**

stride=5是heuristic选择（匹配human reaction time），但更长history可能对certain task更好。比如grasp stability可能需要500ms+的history来判断force trend。

**4. No sim2real**

所有pretraining都在real data上。Real touch data收集成本高，限制了scale。如果能sim pretrain + real finetune，scalability更好。但simulator的realism是bottleneck。

**5. No comparison with multimodal pretraining**

Touch-and-Go、Binding Touch to Everything这些vision-touch alignment work没直接对比。Sparsh是touch-only SSL，如果加入vision或language supervision，能不能学到更好的touch representation？

---

## 我的Intuition Summary

读完这篇paper，几个关键take-away：

**第一，touch的SSL和vision的SSL有本质区别。** Touch image是gel deformation的indirect measurement，noise profile、temporal dynamics、cross-sensor variation都和natural image不同。简单照搬vision SSL方法不够，需要domain-specific adaptation。但一旦adapt好，SSL的label efficiency优势在touch domain比vision domain更显著，因为touch的labeling成本更高。

**第二，latent space SSL > pixel space SSL for touch。** 这和vision里MAE很强的现象相反。原因是touch image的distractor noise占比高，pixel reconstruction浪费capacity。Latent space learning能filter noise，保留task-relevant signal。例外是pixel-level texture task（textile recognition），MAE有优势。

**第三，temporal dimension对touch至关重要。** 80ms窗口能capture slip、pose change这种dynamic property。V-JEPA的video clip建模在temporal task上碾压single-image方法。但temporal modeling的inference cost要权衡。

**第四，cross-sensor generalization是touch representation的killer feature。** Background subtraction + multi-sensor pretraining让representation跨sensor transferable。10-shot cross-sensor transfer的61.8% vs 10.9%是convincing evidence。

**第五，不同SSL方法有task-specific优势。** 没有one-size-fits-all：
- DINO: force, pose (physics regression)
- I-JEPA: stability, textile (semantic)
- V-JEPA: slip (temporal)
- MAE: textile (pixel-level texture)

Future work可能需要multi-objective SSL，或者根据downstream task选backbone。

**第六，fine-tuning不一定比frozen好。** Policy learning task上frozen反而更好，因为fine-tuning破坏pretrained representation的general features。Regression task上full fine-tuning有额外收益。Fine-tuning策略要task-specific。

**第七，representation learning → policy learning有gap。** Bead maze的compounding error问题暴露了这个gap。Pretrained representation在dense prediction task上work，但sequential decision making还需要更好的policy learning framework。

---

## 批判性思考

**Strong points：**
1. 实验设计严谨，6个task覆盖properties → perception → manipulation的hierarchy
2. 5种SSL方法的systematic comparison，ablation充分
3. Cross-sensor evaluation是真实痛点，10-shot transfer结果convincing
4. Label efficiency的data budget ablation（full, 1/3, 1/10, 1/100）很全面

**Weak points：**
1. Bead maze没成功完成，暴露representation → policy的gap
2. 460k images还很小，scaling law没探索
3. Temporal history length没ablate
4. No multimodal pretraining comparison
5. V-JEPA的value proposition局限于temporal task，inference cost高

**Future directions值得探索：**
1. Multi-modal Sparsh（touch + vision + language pretraining）
2. RL with Sparsh（sample efficiency提升）
3. Larger scale（10M+ touch images, ViT-L/H）
4. Sim2Real Sparsh（sim pretrain + real finetune）
5. Tactile foundation model for dexterous manipulation

---

**最终intuition：** Sparsh是touch representation learning的里程碑，它证明SSL paradigm可以成功迁移到touch domain，但需要domain-specific adaptation。Touch的noise profile、temporal dynamics、cross-sensor variation是三个核心challenge，paper分别用latent-space SSL、temporal tokenization、background subtraction + multi-sensor pretraining来address。TacBench提供了标准化evaluation framework，为future work的 apples-to-apples comparison奠定基础。期待看到Sparsh在更大scale、更多modality、更复杂manipulation task上的扩展。

---

# Sparsh: Self-supervised Touch Representations 深度讲解

## 1. 论文核心动机与定位

这篇paper来自FAIR at Meta，作者团队包括Carolina Higuera、Akash Sharma等，与CMU和UW合作。核心问题是：vision-based tactile sensors（如DIGIT、GelSight系列）虽然在robot manipulation中越来越普及，但目前每个task、每个sensor都要训练custom model，极度fragmented。

Margaret Atwood那句"Touch comes before sight, before speech"在论文intro被引用，作者反过来说：今天AI领域恰恰相反，touch反而落后于vision和language。这篇论文的目标就是把NLP和CV中成功的SSL paradigm搬到touch domain。

项目页面：https://sparsh-ssl.github.io/

参考相关work：
- DINO: https://arxiv.org/abs/2104.14294
- DINOv2: https://arxiv.org/abs/2304.07193
- MAE: https://arxiv.org/abs/2111.06377
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2304.12391

## 2. Vision-based Tactile Sensors 背景

先快速过一下sensors，因为这影响整个representation learning的设计：

- **GelSight 2017** (Yuan, Dong, Adelson)：带markers的elastomer，可以通过marker tracking获取dense shear field，分辨率高
- **DIGIT** (Lambeta et al., 2020)：Meta出的compact sensor，无markers，靠RGB LED照明 + camera捕捉gel deformation
- **GelSight Mini**：markerless，HD resolution

关键区别在于markers vs markerless。带markers的sensor可以通过传统CV方法（marker tracking）拿到shear force field，但markerless sensor要靠learning来infer这些物理量。Sparsh的卖点之一就是让markerless sensor也能做shear/slip detection这种传统上需要markers的任务。

sensor参考：
- GelSight: https://arxiv.org/abs/1708.00896
- DIGIT: https://arxiv.org/abs/2005.14697

## 3. Sparsh的SSL方法家族

这是论文的核心技术贡献。作者把三种SSL paradigm适配到touch domain：

### 3.1 Sparsh (MAE) - Masked Autoencoder

经典MIM方法。把image大量mask（典型75%），encoder只处理visible patches，decoder重建masked patches。

Loss（论文Eq. 1）：

$$\mathcal{L}_{\mathrm{MAE}} = \|\mathbf{I}_{\mathrm{target}} - \mathbf{I}_{\mathrm{recon}}\|_2^2$$

变量含义：
- $\mathbf{I}_{\mathrm{target}}$: 被mask的原始pixel patch
- $\mathbf{I}_{\mathrm{recon}}$: decoder重建的patch
- $\|\cdot\|_2^2$: L2 norm的平方，pixel-space MSE

**intuition**：MAE在pixel space做重建，model必须学习gel deformation的fine-grained appearance。但touch image有大量noise（lighting variation、manufacturing discrepancy），pixel-level重建可能浪费capacity在distractor上。

### 3.2 Sparsh (DINO) / Sparsh (DINOv2) - Self-distillation

student-teacher架构，两个identical networks。student预测teacher的output distribution。

Loss（Eq. 2）：

$$\mathcal{L}_{\mathrm{DINO}} = -\sum \mathbf{p}_t \log \mathbf{p}_s$$

变量：
- $\mathbf{p}_s$: student network经softmax normalize后的class probability分布
- $\mathbf{p}_t$: teacher network（EMA updated）的probability分布
- 这是cross-entropy form，但target是teacher的soft prediction，不是hard label

**intuition**：DINO在latent space学习，通过multi-crop augmentation让student和teacher看同一image的不同crops，输出要一致。这就避开了pixel-level reconstruction，让model把capacity花在semantic特征上。论文发现DINO在force estimation和pose estimation这种physics-based task上表现最好。

DINOv2是DINO + iBot（MIM）的组合，论文中也做了Sparsh (DINOv2)变体。

### 3.3 Sparsh (I-JEPA) / Sparsh (V-JEPA) - Joint-Embedding Predictive Architecture

LeCun提出的JEPA框架。context network观察masked image，predict target network对local crops的latent representation。

Loss（Eq. 3）：

$$\mathcal{L}_{\mathrm{jepa}} = \sum_{i \in M} \sum_{j \in B_i} \|\hat{\mathbf{s}}_{y_j} - \mathbf{s}_{y_j}\|_2^2$$

变量：
- $M$: global context masks的集合（context network观察的masked regions）
- $B_i$: 第$i$个context对应的target local crops集合
- $\hat{\mathbf{s}}_{y_j}$: context network输出经predictor预测的target representation
- $\mathbf{s}_{y_j}$: target network（EMA teacher）对local crop $y_j$ 输出的latent embedding
- $\|\cdot\|_2^2$: latent space L2 distance

**intuition**：JEPA在latent space做predictive learning，既不用pixel reconstruction（避开noise），又不用multi-crop contrastive（避开batch size需求）。I-JEPA用spatial block masking，V-JEPA用tube masking处理video clips。论文发现JEPA系列在slip detection、grasp stability、textile recognition这种semantic understanding task上更强。

V-JEPA在slip detection上表现最好（F1 = 0.820），因为它直接处理4帧video clip，temporal reasoning能力更强。

参考JEPA路线图：https://openreview.net/pdf?id=bZoJ3qXCaN

## 4. 关键技术细节

### 4.1 Temporal Tokenization

Touch信号是时序的，slip detection、pose estimation都需要temporal context。作者的处理方式：

- **Image-based SSL (MAE, DINO, I-JEPA)**: 把两帧image沿channel dimension拼接
  $$\mathbf{I}_t \oplus \mathbf{I}_{t-5} \to x \in \mathbb{R}^{h \times w \times 6}$$
  stride = 5 samples，60FPS sensor对应~80ms inference window
  
- **Video-based SSL (V-JEPA)**: 4帧clip
  $$[t, t-2, t-4, t-6] \in \mathbb{R}^{4 \times h \times w \times 3}$$
  ~100ms inference window

为什么是80ms？作者引用了Zangrandi et al. 2021的neurophysiology研究：人类检测partial slip后调整grip force的反应时间就是~80ms。这个设计是bio-inspired的。

### 4.2 Background Subtraction

针对markerless sensors（DIGIT, GelSight Mini）做background subtraction，给model一个no-contact reference。这能传递static shear information。Empirically帮助model跨同类型sensor generalize。

### 4.3 ViT Architecture

所有encoder都是ViT-B/14，参数量~86M：
- Sparsh (MAE): 86.25M, 104 FPS
- Sparsh (DINO): 86.26M, 112 FPS
- Sparsh (IJEPA): 86.39M, 112 FPS
- Sparsh (VJEPA): 86.54M, 60 FPS（video处理慢一些）

ViT-B/14表示patch size 14×14，image resize到224×224，所以有16×16=256个patches。

作者还用了ViT registers（Darcet et al. 2024, https://arxiv.org/abs/2401.09209）来替代[cls] token做classification。

### 4.4 Training Hyperparameters

Table 1的关键信息：
- 8× A100 80G GPU
- 150 epochs
- AdamW optimizer
- weight decay: cosine schedule 0.04 → 0.4
- LR warmup: 30 epochs
- EMA decay: DINO 0.998, IJEPA/VJEPA 0.996

注意MAE没有EMA（因为它是single network）。

### 4.5 Dataset Curation

总计~661k images，70% (462.7k)用于SSL pretraining：

- **YCB-Slide** (Suresh et al., CoRL 2022): 180k frames，DIGIT，10个YCB物体sliding
- **Touch-and-Go** (Yang et al. 2022, https://arxiv.org/abs/2211.12498): 220k frames，GelSight，human-collected in-the-wild
- **ObjectFolder** (Gao et al., CVPR 2022): 81k frames，robot discrete contact
- **Touch-Slide** (new): 180k frames，DIGIT，9个toy-kitchen物体

数据规模比prior work大一个数量级（previous: Cao et al. 2023, Yang et al. 2024, Dou et al. 2024都是小规模）。

## 5. TacBench: 6个Tasks的Benchmark

作者把tasks分成三类：tactile properties、physical perception、manipulation planning。

### 5.1 [T1] Force Estimation

预测3-axis normal + shear forces。Data: 75k time-aligned samples，用3种indenter（hemisphere, sharp, flat）。

训练用normalized force ∈ [-1, 1]，L1 loss，Adam optimizer。Metric: 平均3-axis RMSE。

**DIGIT结果** (Table 5, RMSE in mN):
| Model | Full (50k) | 1/3 | 1/10 | 1/100 |
|-------|-----------|-----|------|-------|
| E2E | 39.34 | 61.42 | 98.22 | 187.51 |
| Sparsh (DINOv2) | **29.31** | **26.85** | **37.66** | 185.86 |
| Sparsh (DINO) | 36.09 | 44.03 | 51.89 | 97.95 |

**关键观察**：E2E在full data下还能竞争（39.34 vs 29.31），但data减少到1/3时大幅退化（61.42 vs 26.85，差57%）。Sparsh (DINOv2)在1/3 data下居然比E2E在full data下还好。1/100 data时所有model都退化严重。

**GelSight Mini结果** (Table 6)：
E2E在GelSight Mini上即使full data也只有57.21 mN，因为HD resolution导致contact region相对background很小，from-scratch训练困难。Sparsh (DINO)能达到20.25 mN，提升64%。

### 5.2 [T1A] Force Field Visualization

定性任务，可视化normal field（类比depth estimation）和shear field（类比optical flow）。用DPT decoder (Ranftl et al., ICCV 2021, https://arxiv.org/abs/2103.13413)做dense prediction。

unsupervised learning思路：
- Normal field: monocular depth estimation的reprojection loss
- Shear field: optical flow的photometric consistency (Charbonnier + SSIM) + smoothness regularization

Figure 10展示Sparsh (DINO)能capture各种motion pattern：torsional slip、sliding on edge、diverging field upon contact。

这对markerless sensor特别有价值，因为传统marker tracking不可用。

### 5.3 [T2] Slip Detection

125k samples，13% slip instances（高度imbalanced）。用friction cone model + 实测static friction coefficient标注。

metric用F1 score（不用accuracy，因为imbalance）。

**结果** (Table 7):
| Model | Full | 1/3 | 1/10 | 1/100 |
|-------|------|-----|------|-------|
| E2E | 0.767 | 0.238 | 0.299 | 0.214 |
| Sparsh (VJEPA) | **0.820** | **0.828** | **0.800** | **0.760** |
| Sparsh (IJEPA) | 0.776 | 0.791 | 0.775 | 0.726 |

**惊人结果**：Sparsh (VJEPA)在1/100 data下F1=0.760，比E2E在full data下的0.767还差不多！而且Sparsh (VJEPA)在1/3 data下居然比full data还略高（0.828 vs 0.820），说明label效率极高。

V-JEPA的优势来自video clip的temporal modeling，4帧clip直接建模时序。

### 5.4 [T3] Pose Estimation

SE(2) relative pose (Δx, Δy, Δθ) estimation。用regression-by-classification：把每个DOF离散化到11 bins（log-uniform space）。

Data: 49k samples，DIGIT + Allegro hand。

**结果** (Table 8, accuracy):
| Model | Full | 1/3 | 1/10 | 1/100 |
|-------|------|-----|------|-------|
| E2E | 0.812 | 0.245 | 0.162 | 0.162 |
| Sparsh (DINO) | **0.913** | **0.834** | 0.460 | 0.242 |
| Sparsh (MAE) | 0.896 | 0.719 | 0.417 | 0.223 |

E2E在1/3 data时崩盘到0.245（接近random），Sparsh (DINO)还能保持0.834。这是95%+的relative improvement。

### 5.5 [T4] Grasp Stability

用Feeling of Success dataset (Calandra et al. 2017, https://arxiv.org/abs/1710.05512)，9.3k grasps，GelSight 2017 (markers)。输入'before'和'during'两帧。64% success / 36% failure。

**结果** (Table 9, accuracy):
| Model | Full | 1/3 | 1/10 | 1/100 |
|-------|------|-----|------|-------|
| E2E | 0.784 | 0.725 | 0.682 | 0.478 |
| Sparsh (IJEPA) | 0.802 | **0.782** | 0.768 | **0.598** |
| Sparsh (VJEPA) | **0.809** | 0.702 | 0.743 | 0.523 |

原paper [8]结合touch + vision才达到~75%，Sparsh (IJEPA)只用touch single finger就80.2%。

### 5.6 [T5] Textile Recognition

Clothing Dataset (Yuan et al. 2018, https://arxiv.org/abs/1710.11832)，4467 video clips，20类textile，GelSight 2017 (markers)。

**结果** (Table 10, accuracy, chance = 0.05):
| Model | Full | 1/3 | 1/10 | 1/100 |
|-------|------|-----|------|-------|
| E2E | 0.437 | 0.365 | 0.373 | 0.171 |
| Sparsh (MAE) | **0.599** | **0.588** | **0.527** | **0.330** |

这里Sparsh (MAE)反超DINO/IJEPA！因为textile recognition是pixel-level texture task，MAE的pixel-space reconstruction恰好capture这种fine-grained appearance。这印证了论文的finding：不同SSL方法在不同task上有不同优势。

### 5.7 [T6] Bead Maze

Diffusion Policy (Chi et al. RSS 2023, https://arxiv.org/abs/2303.04137) + Sparsh encoder替换CNN encoder。50 demonstrations，~34k training pairs。Franka arm + DIGIT。

observation horizon=2, action prediction horizon=8, predict Δq ∈ ℝ^7。

**Position error** (Table 12, mm):
| Model | Full | 1/2 | 1/10 |
|-------|------|-----|------|
| E2E | 8.46 | 7.14 | 9.80 |
| Sparsh (DINO) | **5.54** | 5.98 | 5.71 |
| Sparsh (IJEPA) | 5.47 | 5.72 | 5.46 |

**Real rollout distance** (Table 11, cm before failure):
| Model | Pre-trained | Fine-tuned |
|-------|-------------|------------|
| E2E | 6.70 ± 1.67 | 6.70 ± 1.67 |
| Sparsh (DINO) | **10.80 ± 3.68** | 8.45 ± 3.21 |
| Sparsh (IJEPA) | 9.40 ± 3.10 | 10.02 ± 5.37 |
| Sparsh (MAE) | 10.20 ± 4.90 | 11.25 ± 3.85 |

Sparsh (DINO)比E2E多走61%距离。但所有model都没完成full maze，因为高precision task + covariate shift导致compounding error。

**意外发现**：fine-tuning反而比frozen差（Sparsh (DINO): 10.80 → 8.45）。作者推测是domain mismatch + overfitting。

## 6. 整体性能汇总

Table 13的关键数字：
- **Best SSL vs E2E average improvement: 98.75%**
- DINO vs IJEPA: DINO平均好8.91%
- MAE vs Best: MAE平均差5.57%
- VJEPA vs Best: VJEPA平均差24.47%（但slip detection上VJEPA最强）

**核心finding**：
1. Latent space SSL (DINO, IJEPA) > Pixel space SSL (MAE)，因为touch image有大量noise/distractor，latent space能filter掉
2. DINO适合physics-based regression task（force, pose）
3. IJEPA/VJEPA适合semantic understanding task（slip, stability, textile）
4. V-JEPA在temporal task（slip）上有独特优势

## 7. Ablations亮点

### 7.1 Fine-tuning (Appendix E.1)

- **Full fine-tuning**: latent-space SSL (DINO, IJEPA, VJEPA)在regression task上提升明显，特别是low-data regime
- **Partial fine-tuning** (只调最后一个transformer block): 效果接近frozen，minor improvement
- **MAE fine-tuning效果差**: 作者hypothesize MAE weights更brittle，因为没用EMA，minima basin窄

### 7.2 ViT-Small (Appendix E.2)

把ViT-B (768 dim)换成ViT-S (384 dim)：
- Regression task (force, pose)受影响最大：Sparsh (DINO)在force estimation上DIGIT误差增74%，GelSight Mini增50.3%（33% data时）
- Classification task (slip, stability, textile)基本不受影响，除非1% data

说明regression需要更高dimensional representation来编码continuous物理量。

### 7.3 Cross-sensor Transfer (Appendix E.3)

[T5] Textile decoder训练在GelSight (markers)上，迁移到DIGIT做n-shot evaluation：

| | zero-shot | 1-shot | 5-shot | 10-shot |
|---|-----------|--------|--------|---------|
| Sparsh (DINO) | 9.1 | 19.1 | 28.2 | **61.8** |
| E2E | 3.6 | 0.0 | 15.5 | 10.9 |

10-shot时Sparsh (DINO)达到61.8% accuracy，E2E只有10.9%。这证明SSL pretraining学到了cross-sensor invariant的touch representation。

## 8. Limitations & Future Work

作者坦诚承认：
1. **Dataset偏discrete contact**，shear interaction不够丰富
2. **没ablate temporal history length**，可能更长的history对certain task更好
3. **Bead maze没完成full maze**，compounding error + 缺force control
4. **Real robot deployment有system-level confounding variables**

## 9. 我的Intuition构建

读完这篇paper，几个关键take-away：

**第一，touch的SSL和vision的SSL有本质区别**。Touch image不是natural image，它是gel deformation的indirect measurement，充满sensor-specific noise。所以pixel-space reconstruction（MAE）不如latent-space learning（DINO, JEPA）。这和vision里MAE反而很强的现象相反。

**第二，temporal dimension对touch至关重要**。Vision的SSL大多用single image，但touch信号80ms窗口才能capture slip、pose change这种dynamic property。V-JEPA在slip detection上的优势印证了这一点。

**第三，cross-sensor generalization是touch representation的killer feature**。不同sensor（DIGIT vs GelSight vs GelSight Mini）光学特性、marker有无都不同，传统方法每个sensor都要custom model。Sparsh通过background subtraction + 大规模multi-sensor pretraining，让representation跨sensor transferable。10-shot cross-sensor transfer的61.8% vs 10.9%是convincing evidence。

**第四，label efficiency是SSL在robotics的核心价值**。Robotics的labeled data收集成本极高（force需要F/T sensor，slip需要friction cone labeling，pose需要ArUco tracking）。Sparsh在33-50% label budget下平均95.1% improvement over E2E，这意味着同样performance可以用1/3到1/2的labeling effort。

**第五，不同SSL方法有task-specific优势**。没有one-size-fits-all：
- DINO: force, pose (physics regression)
- IJEPA: stability, textile (semantic)
- VJEPA: slip (temporal)
- MAE: textile (pixel-level texture)

这暗示future work可能需要multi-objective SSL，或者根据downstream task选backbone。

**第六，fine-tuning不一定比frozen好**。Bead maze实验中fine-tuning反而hurt performance，这和vision pre-training for control (Hansen et al. 2022, https://arxiv.org/abs/2212.05749)的发现一致。Pretrained representation已经capture了task-relevant features，fine-tuning反而破坏这些features去fit narrow task distribution。

## 10. 与Concurrent Work对比

论文提到两个concurrent work：

- **T3** (Zhao et al. 2024, https://arxiv.org/abs/2406.13640): sensor-specific encoders + shared trunk + MAE + task supervision。需要labels。
- **UniT** (Xu et al. 2024, https://arxiv.org/abs/2408.06481): VQGAN + patch discriminator，只针对GelSight Mini (markers)。

Sparsh的差异化：
1. Pure SSL（不需要labels for pretraining）
2. Cross-sensor（DIGIT, GelSight 2017, GelSight Mini）
3. Standardized benchmark (TacBench)

## 11. 对Robotics Manipulation的Implication

这篇paper的long-term vision是touch representation成为robot manipulation的foundation model。类比vision里ImageNet pretraining + task-specific fine-tuning的paradigm，touch domain终于有了类似的backbone。

具体implications：
1. **Insertion、in-hand manipulation、grasping**这些contact-rich task现在可以用Sparsh作为touch encoder，不用从scratch训练
2. **Multi-modal fusion**：Sparsh representation可以和vision representation（如R3M, https://arxiv.org/abs/2203.12601; VIP, https://arxiv.org/abs/2210.00030）在latent space融合
3. **Tactile-language alignment**：类似Touch and Go (https://arxiv.org/abs/2211.12498)和Binding Touch to Everything (https://arxiv.org/abs/2401.18084)，Sparsh可以作为touch encoder做touch-language alignment
4. **Sim2Real**：如果simulator（TACTO, https://arxiv.org/abs/2012.08456; Taxim, https://arxiv.org/abs/2109.04027; DiffTactile, https://openreview.net/forum?id=eJHnSg783t）能生成足够realistic的touch image，Sparsh可以在sim data上pretrain

## 12. 技术细节深挖

### 12.1 Attentive Probe架构

Evaluation用frozen encoder + attentive probe (Caron et al. DINO; Context Autoencoder, https://arxiv.org/abs/2202.03018)。

架构（Table 4）：
- Cross-attention layer: embedding dim 768, 12 heads, MLP ratio 4.0, depth 1
- 后接2-layer MLP做task-specific prediction

**为什么用attentive probe而不是linear probe？** 因为touch task可能需要spatially-pooled features + local patch features的混合，cross-attention能learnable地aggregate。

### 12.2 DPT Decoder for Dense Prediction

Force field visualization用DPT (Ranftl et al. ICCV 2021, https://arxiv.org/abs/2103.13413)：

从ViT的layer 2, 5, 8, 11取patch tokens，经过reassemble + fusion modules progressively upsample到full resolution。这和dense prediction task（depth, segmentation）的标准做法一致。

### 12.3 Friction Cone Labeling for Slip

Slip label用friction cone model：
$$\sqrt{f_x^2 + f_y^2} > \mu \cdot f_z$$

其中$f_x, f_y$是shear force分量，$f_z$是normal force，$\mu$是实测static friction coefficient。

当shear force magnitude超过$\mu \cdot f_z$时，标记为slip。这个labeling有noise（$\mu$是empirical估计），作者在Figure 13展示了failure case。

### 12.4 Regression-by-Classification for Pose

SE(2) pose change $(\Delta x, \Delta y, \Delta \theta)$被离散化到11 bins per DOF（log-uniform space）。

**为什么log-uniform？** 因为大多数pose change集中在zero附近，log-uniform binning让小change有更高分辨率，大change有较低分辨率。这和IEBins (Shao et al. NeurIPS 2024, https://arxiv.org/abs/2309.11719) for depth estimation的思路一致。

Translation: ±5mm resolution
Rotation: ±2° resolution

每个DOF训练一个head，输出11-class probability distribution，cross-entropy loss。

## 13. 批判性思考

**Strong points**：
1. 实验设计严谨，6个task覆盖properties → perception → manipulation的hierarchy
2. Cross-sensor evaluation是真实痛点，10-shot transfer结果convincing
3. Label efficiency的data budget ablation（full, 1/3, 1/10, 1/100）很全面
4. 同时对比5种SSL方法（MAE, DINO, DINOv2, IJEPA, VJEPA），ablation充分

**Weak points / Open questions**：
1. **Bead maze没成功完成**：作者归因于system-level confounders，但这暴露了representation learning → policy learning的gap。Pretrained representation在dense prediction task上work，但在sequential decision making上还有距离
2. **No sim2real**：所有pretraining都在real data上，但real touch data收集成本高。如果能在sim pretrain + real finetune，scalability会更好
3. **Temporal history length没ablate**：stride=5是heuristic选择（匹配human reaction time），但更长history可能对certain task更好
4. **No comparison with multimodal pretraining**：Touch-and-Go、Binding Touch to Everything这些vision-touch alignment work没直接对比
5. **V-JEPA只在slip detection上最强**：其他task上不如I-JEPA，但V-JEPA inference慢一倍（60 vs 112 FPS）。是否值得video-level modeling存疑

**Future directions I'd explore**：
1. **Multi-modal Sparsh**：把touch和vision、proprioception、language一起pretrain，类似Voltron (https://arxiv.org/abs/2302.12766)但touch-centric
2. **Reinforcement learning with Sparsh**：用Sparsh作为RL policy的touch encoder，看sample efficiency提升
3. **Larger scale**：460k images还很小（vs ImageNet 1.4M, DINOv2 142M）。如果能到10M+ touch images，scaling law会怎样？
4. **Tactile foundation model for dexterous manipulation**：把Sparsh和in-hand manipulation（如AnyRotate, https://arxiv.org/abs/2405.07391; General In-Hand Rotation, https://arxiv.org/abs/2310.13024）结合

## 14. 总结

Sparsh是touch representation learning的里程碑paper。它把CV的SSL paradigm系统地适配到touch domain，解决了cross-sensor generalization、label efficiency、temporal modeling三个核心问题。TacBench提供了标准化evaluation framework。

**核心贡献**：
1. 5个SSL backbone（MAE, DINO, DINOv2, IJEPA, VJEPA）的touch domain adaptation
2. 460k+ multi-sensor pretraining dataset curation
3. TacBench: 6个task的standardized benchmark
4. 实验证明latent-space SSL > pixel-space SSL for touch

**核心insight**：Touch image不是natural image，它的noise profile、temporal dynamics、cross-sensor variation都和vision不同。简单照搬vision SSL方法不够，需要domain-specific adaptation（background subtraction, temporal tokenization, multi-sensor pretraining）。但一旦adapt好，SSL的label efficiency优势在touch domain甚至比vision domain更显著，因为touch的labeling成本更高。

这篇paper为touch-based robot manipulation的foundation model时代开了门。期待看到Sparsh在更大scale、更多modality、更复杂manipulation task上的扩展。

---

**参考链接汇总**：
- Project page: https://sparsh-ssl.github.io/
- DINO: https://arxiv.org/abs/2104.14294
- DINOv2: https://arxiv.org/abs/2304.07193
- MAE: https://arxiv.org/abs/2111.06377
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2304.12391
- JEPA roadmap: https://openreview.net/pdf?id=bZoJ3qXCaN
- DIGIT: https://arxiv.org/abs/2005.14697
- GelSight: https://arxiv.org/abs/1708.00896
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DPT: https://arxiv.org/abs/2103.13413
- Touch and Go: https://arxiv.org/abs/2211.12498
- ObjectFolder: https://arxiv.org/abs/2206.10979
- Feeling of Success: https://arxiv.org/abs/1710.05512
- ViT Registers: https://arxiv.org/abs/2401.09209
- IEBins: https://arxiv.org/abs/2309.11719
- T3: https://arxiv.org/abs/2406.13640
- UniT: https://arxiv.org/abs/2408.06481
- Binding Touch to Everything: https://arxiv.org/abs/2401.18084
- TACTO: https://arxiv.org/abs/2012.08456
- Taxim: https://arxiv.org/abs/2109.04027
- DiffTactile: https://openreview.net/forum?id=eJHnSg783t
- R3M: https://arxiv.org/abs/2203.12601
- VIP: https://arxiv.org/abs/2210.00030
- Voltron: https://arxiv.org/abs/2302.12766
- Pretraining for visuo-motor control: https://arxiv.org/abs/2212.05749
- General In-Hand Rotation: https://arxiv.org/abs/2310.13024
- AnyRotate: https://arxiv.org/abs/2405.07391
- MidasTouch: https://arxiv.org/abs/2212.14065
- Neural Feels: https://arxiv.org/abs/2312.13460
