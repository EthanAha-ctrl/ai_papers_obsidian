---
source_pdf: V-JEPA2.pdf
paper_sha256: 9cfcfde5fb0d9730637da5b9e7317825c3f3d09e91f3553e22eeba42c74d2226
processed_at: '2026-08-13T00:04:49-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# V-JEPA 2 人话版

Andrej, 咱换个方式讲，把这篇paper的"灵魂"拎出来，用大白话+比喻讲清楚为什么这件事在2026年是个big deal。

---

## 一、这篇paper在讲一个什么故事

你可以把它想成一个小孩的成长过程：

**第一阶段**：小孩在YouTube上刷了100万小时的短视频。刷视频的时候没人给他讲解，他也不需要做任何动作，就是看。看人类做饭、看人打球、看机器人抓东西、看猫跳上桌子。看着看着，小孩脑子里慢慢形成了一种"世界的感觉"——什么东西会怎么动、物体掉下去会怎样、手碰到杯子会发生什么。这个过程叫 **V-JEPA 2 pretraining**。

**第二阶段**：给这个小孩一台机器人手臂，让他玩62小时。没人告诉他"这是在学抓杯子"或者"你失败了要重来"，就是让他随便玩，记录下video和手臂的位置。这一阶段叫 **V-JEPA 2-AC post-training**。

**结果**：把这个小孩（模型）搬到两个他从来没见过的实验室，给他一张目标图片说"把杯子从这儿挪到那儿"，他就能做到。65-80%的成功率，不训练、不调参、不收集任何新数据，直接上。

这件事听起来挺神奇，因为之前大家觉得要让机器人学会抓东西，得在同一个实验室里反复teleop几千次，或者写很复杂的reward function让RL慢慢探索。V-JEPA 2说：你给我看看视频就行了。

参考：https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks

---

## 二、核心insight：为什么不在pixel space生成？

这是整篇paper最深的哲学选择，值得慢慢讲。

想象你要预测一段视频的下一帧。两种思路：

### 思路A：Pixel Generation（Cosmos, GAIA-1, Genie, Sora走的路）

你要预测下一帧每个pixel的颜色。问题来了——草地上的每一根草、树叶的每一条纹理、背景墙上的颗粒——这些东西下一秒变成什么样，本质上不可预测。但你用diffusion model硬要生成的话，模型会怎么做？要么blur掉（uncertainty下取平均），要么在几种可能状态之间randomly跳。这就是为什么video generation models经常出现"物体突然消失又出现""背景在呼吸"这种诡异现象。

更糟的是，做planning时你每试一个action trajectory，都要跑几十步diffusion sampling。Cosmos 7B每action要4分钟。一个pick-and-place trajectory跑下来一小时过去了，机器人都凉了。

### 思路B：Latent Prediction（V-JEPA 2走的路）

你不去predict pixel，而是predict "abstract state"。什么叫abstract state？想象你看一段视频，脑子里其实只关心几件事：杯子在桌上、手在杯子左边、手正在合拢。至于杯子上的花纹精确到pixel什么样，你根本不在乎，因为这对"抓杯子"这个task毫无帮助。

V-JEPA 2的encoder学的就是把video变成这样一个abstract representation。然后predictor学的不是"下一帧长什么样"，而是"下一帧的abstract state会变成什么样"。

好处是什么？**predictable的信息被保留，unpredictable的细节被自动丢弃**。Energy landscape（你衡量"离goal有多远"的那个function）变得非常smooth，像一口碗而不是一片崎岖的山地。Planning用Cross-Entropy Method采样action sequence就能可靠收敛，每action 16秒。

Figure 9那张energy landscape图特别有说服力——在Δx-Δy平面上画出来是个smooth的碗，minimum就在ground truth action附近。这种smoothness是pixel-space model永远拿不到的，因为pixel space的noise会把landscape搞得全是local minimum。

LeCun说了好多年"放弃生成pixel，predict抽象"，这篇paper算是把这个idea推到了real robot zero-shot deployment的程度。

---

## 三、JEPA的objective为什么能学到东西

这个其实蛮巧妙的，值得讲讲。

你训练一个encoder把video变成feature，再加一个predictor预测masked部分的feature。问题来了：怎么防止model偷懒把所有feature都collapse成一个常数？那样loss也是0啊。

V-JEPA 2用了两个trick：

**Trick 1: EMA Teacher**

Target不是由current encoder生成的，而是由一个"慢半拍"的encoder生成。这个慢半拍版本的weights是current weights的exponential moving average（EMA decay 0.99925，相当于大概几千步才追上）。你让student去predict一个慢慢移动的target，target不能跟着student一起跑，所以student没法把target拽到collapse点。

这个思路BYOL最早提出来，DINO/MoCo都用了类似idea。JEPA的特殊之处是target在 **representation space** 而不是pixel space。

**Trick 2: Stop-Gradient**

Target那个branch的梯度被切断，gradient只流过student encoder和predictor。如果允许梯度流过teacher，teacher会被推着往"容易被predict"的方向走，最后还是collapse。Stop-gradient强制teacher只能被EMA passively更新。

公式1里那个 $\text{sg}(\cdot)$ 就是stop-gradient的意思。整个loss是：

$$
\| P_\phi(\Delta_y, E_\theta(x)) - \text{sg}(E_{\bar{\theta}}(y)) \|_1
$$

左边是predictor的输出，右边是EMA teacher的输出加stop-gradient。L1 loss只在masked patches位置算。

**Intuition**：这就像让一个学生（student encoder + predictor）去模仿一个慢慢进步的老师（EMA teacher）。老师进步慢，学生追得上但追过头就没意义了，所以学生只能学到"真正的结构"，没法靠让老师降级来作弊。

参考I-JEPA：https://arxiv.org/abs/2301.08243

---

## 四、Scaling四件套——怎么从ViT-H/2M视频scale到ViT-g/22M视频

这部分很engineering，但很关键。Scaling不是把模型变大就完事，有四个维度同时要scale：

1. **Data**: 2M → 22M videos。VM22M混合了SSv2、Kinetics、HowTo100M、curated YT1B、ImageNet。ImageNet被duplicate成16帧一样的video混进来，权重0.25——这是为了补appearance（texture、shape、color），因为video数据偏motion，纯video pretrain做ImageNet classification会差。

2. **Model**: ViT-L (300M) → ViT-g (1B)。Width 1024→1408, Depth 24→40, Heads 16→22。Predictor始终是ViT-s (22M)不scale。

3. **Training length**: 90K → 252K iterations。换成warmup-constant-decay schedule，这样constant phase的checkpoint可以reuse跑多个cooldown ablation，HPO便宜。

4. **Resolution + Duration**: 256×256×16frames → 384×384×64frames。

第四点是最有意思的engineering trick——**progressive resolution training**。

如果在384×384×64frames从头训ViT-g，需要60 GPU-years。没人能跑得起。他们做的是：

- 前12K iter：16 frames @ 256×256 warmup
- 中间228K iter：同上，constant LR
- 最后12K iter：升到64 frames @ 384×384，linear decay LR

8.4× speedup，性能不掉。Intuition是constant phase后representation已经学好，cooldown只是adapt到high-resolution input distribution。

这招对所有做video pretrain的人都有用，因为你训练时永远想用短clip省GPU，推理时又想用长clip拿好性能。

参考：https://arxiv.org/abs/2405.18392

---

## 五、V-JEPA 2-AC：从"看视频"到"能act"

这是最magic的部分。讲三个点：

### 5.1 Frozen Encoder是关键

V-JEPA 2 pretrain完之后，encoder就freeze了，不再update。然后训一个新的action-conditioned predictor，300M params。

为什么要freeze？因为如果你让predictor训练时也update encoder，predictor会"作弊"——它会把encoder的output改成"对predict来说最方便的形式"，而不是"对理解世界最有用的形式"。这就像让翻译员自己改原文，他肯定改成他翻译起来最轻松的版本，但原文就毁了。

Frozen encoder保证representation始终是general-purpose的，可以同时支持action anticipation、video QA、robot planning等多个downstream。

### 5.2 输入序列怎么组织

每帧video过frozen encoder得到 $z_k \in \mathbb{R}^{16\times16\times1408}$（spatial 16×16, channel 1408）。然后序列被组织成：

$$
(a_1, s_1, z_1), (a_2, s_2, z_2), \ldots, (a_{15}, s_{15}, z_{15})
$$

每个 $(a_k, s_k, z_k)$ 三元组里：
- $a_k$ 是action（7D end-effector delta）
- $s_k$ 是end-effector state（7D: 3 pos + 3 Euler + 1 gripper）
- $z_k$ 是visual feature map

Predictor用block-causal attention：时间维度causal（不能看未来），同一时间步内spatial维度bidirectional（同一帧的patches可以互相attend）。

为什么是block-causal而不是full causal？因为同一帧的patches之间没有causality关系——杯子的左上角和右下角是同时存在的，没必要按raster-scan顺序处理。这样比autoregressive image model效率高很多。

### 5.3 两个loss

**Teacher forcing loss**：每一步predict下一帧的representation，用ground truth历史：

$$
\mathcal{L}_{\text{tf}} = \frac{1}{T} \sum_{k=1}^T \| \hat{z}_{k+1} - z_{k+1} \|_1
$$

**Rollout loss**：从第一步开始autoregressively rollout两步，用第一步的真实state：

$$
\mathcal{L}_{\text{rollout}} = \| P_\phi(a_{1:2}, s_1, z_1) - z_3 \|_1
$$

为什么要rollout loss？因为纯teacher forcing会让model永远看ground truth历史，inference时需要自己rollout看自己的预测，会有exposure bias。Rollout loss让模型提前适应"看自己的输出"。

T=2是个折中——再长gradient会unstable，再短效果不明显。

---

## 六、Planning：在imagination里search action

这点我觉得是整篇paper最优雅的地方。

### 6.1 Energy Function

给定当前frame $x_k$ 和goal frame $x_g$，把它们都encode成 $z_k$ 和 $z_g$。然后要找action sequence $\hat{a}_{1:T}$ 让imagined future state representation尽量接近goal representation：

$$
\mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g) = \| P(\hat{a}_{1:T}; s_k, z_k) - z_g \|_1
$$

最小化这个energy就找到了best action sequence。Intuition非常clean：在imagination里try各种action，看哪个action让imagined trajectory end up最接近goal。

### 6.2 为什么用CEM而不是gradient descent？

CEM (Cross-Entropy Method) 是个sampling-based optimizer：
1. 从一组Gaussian distributions采样action trajectory
2. 跑world model看每个trajectory的energy
3. 选top-k（lowest energy的）samples
4. 用这些samples的statistics更新Gaussian的mean和variance
5. 重复几轮，返回最终Gaussian的mean

为什么不直接gradient descent on action space？因为world model虽然differentiable，但energy landscape可能有local minimum，gradient method会卡。CEM是global search，更robust。

但paper Section 10也提到future work可以做gradient-based planning加速10×——因为Figure 9显示landscape是smooth and locally convex的，gradient descent应该work。这是个很好的next step。

参考CEM: https://www.sciencedirect.com/science/article/pii/S037722179700882X

### 6.3 Multi-Goal Trick

Pick-and-place是长horizon task，但他们planning horizon只用1步。怎么做到？用sub-goals：

- Sub-goal 1: 物体被grasped的图片
- Sub-goal 2: 物体在target附近的图片
- Sub-goal 3 (final): 物体在target位置的图片

Schedule：4 steps对sub-goal 1 → 10 steps对sub-goal 2 → 4 steps对sub-goal 3。手动设计的。

这是个limitation——理想情况应该有个high-level planner自动生成sub-goals。但作为proof-of-concept够了。

---

## 七、为什么这件事"反直觉"

### 7.1 没用任何language supervision，居然能做video QA

SigLIP2、Perception Encoder这些SOTA vision encoder都是用image-text contrastive learning训练的，天然aligned with language。V-JEPA 2完全self-supervised，不碰任何text。

按conventional wisdom，没language supervision的encoder应该做不好video QA，因为video QA本质是language task。但Table 8显示V-JEPA 2 + Llama 3.1 8B在PerceptionTest、MVP、TempCompass、TemporalBench、TOMATO上拿了SOTA。

Intuition：video encoder不需要"懂language"，它需要"懂video dynamics"。Language alignment可以靠后期用一个MLP projector + 大量alignment data来补。但"懂dynamics"是pretrain阶段必须学好的，后期补不上。

这个发现对整个VLM领域有implication：可能我们对"vision encoder必须language-aligned"的执念是错的。Self-supervised video encoder + 适量alignment data就够。

### 7.2 62小时unlabeled robot video就够zero-shot deploy

Octo baseline在Open-X-Embodiment（1M+ trajectories）上pretrain，然后在Droid上fine-tune with behavior cloning。结果reach很好，但涉及object interaction的task大幅掉。

V-JEPA 2-AC只用62小时Droid raw video（无label、无reward、无success flag），zero-shot deploy到两个新lab，pick-and-place cup 80% / box 65%。

为什么差这么多？因为BC学的是"demonstrator会做什么"——本质是reactive policy，看到frame就输出action。对precise gripper control、object dynamics的compositional generalization不够强。

V-JEPA 2-AC学的是"dynamics"——给定state和action，next state会怎样。然后planning时search action让imagined trajectory match goal。因为world model学的是dynamics本身（task-agnostic），可以compose到没见过的task。

这就像小孩学物理 vs 小孩背答案的区别。背答案的（BC）遇到没背过的问题就抓瞎；学物理的（world model）可以reasoning through新问题。

---

## 八、限制和future work

### 8.1 Camera Sensitivity

因为没有explicit calibration，model要从monocular RGB infer action coordinate axis。Appendix B.4实验显示inferred axis的rotation error几乎linearly scale with camera position offset。

这是real-world deployment的痛点。VLA没这个问题，因为BC训练时policy已经learned camera-conditioned mapping。V-JEPA 2-AC要解决这个问题可能需要：
- Multi-camera training data
- Explicit camera calibration conditioning
- 或者train一个unsupervised calibration phase（paper末尾提了这个idea）

### 8.2 Long Horizon Planning

Autoregressive rollout有error accumulation——Figure 15a能看到cup位置预测在第16帧时已经drift了。Long horizon task（如pick-and-place without sub-goals）目前做不了。

可能的solution：
- Hierarchical world model（多个time scale的abstraction）
- Latent state smoothing（类似encoder-decoder with skip connections across time）
- Better rollout loss training（更长T，但需要gradient stability tricks）

### 8.3 Planning Speed

16秒/action对reactive task太慢。VLA是real-time。如果能做gradient-based planning，可能再加速10×，到1-2秒/action，就competitive了。

### 8.4 No Language Goal

目前只能用image goal。Paper Section 9明确说future work要align with language。Section 7已经把V-JEPA 2 align到LLM做VidQA了，但还没把language接进world model。一旦接通，就可以说"把那个红色的杯子放到左上角"这种natural language goal。

### 8.5 Gripper State Hardcoded

Action space是7D end-effector delta，对different robot morphology需要重新定义。不像π0/Gr00t那种VLA可以多robot generalize。

---

## 九、更大的picture：这条路接下来怎么走

我自己的几个predictions：

**1年内**：会有人做language-conditioned V-JEPA 2-AC。把LLM接进来生成sub-goals（language或image），low-level用V-JEPA 2-AC execute。这等于SayCan + V-JEPA 2-AC的hierarchy，但比SayCan + RT-2更clean，因为high-level reasoning和low-level dynamics真正decouple了。

**1-2年**：gradient-based planning替代CEM。Figure 9显示energy landscape smooth and locally convex，gradient descent应该work。如果能work，planning速度从16秒/action降到1秒/action以下，就能做reactive closed-loop task。

**2-3年**：V-JEPA scale到10B+ params。Carreira et al. 2024 (https://arxiv.org/abs/2412.15212) 已经在explore 4D representation scaling。V-JEPA 2到1B就停了但scaling curve没saturate，20B应该能再上一台阶。

**3-5年**：V-JEPA + VLA hybrid。用V-JEPA 2-AC做imagination rollout + 用diffusion policy做action proposal + 用world model做filter/refine。最近FLARE (https://arxiv.org/abs/2505.15659) 和CoT-VLA (https://arxiv.org/abs/2503.22020) 在这个方向试探。

**长期**：multi-scale world model。人类cognitive architecture是有hierarchy的——millisecond level motor control, second level action, minute level activity, hour level plan。现在的V-JEPA 2-AC只在second level。真正的AGI-level world model需要multi-scale temporal abstraction，能在不同time scale planning。这是LeCun 2022 proposal (https://openreview.net/pdf?id=BZ5a1r-kVsf) 里的愿景，但还没人做出来。

---

## 十、和你之前讲的"mode collapse in next-token prediction"的关联

Andrej, 你之前讲过LLM的hallucination和next-token prediction的mode collapse问题——model在uncertain的时候会blur或者jumping between modes。

V-JEPA路线是对"pixel-level next-frame prediction"的批评和修正。LLM在discrete token space做next-token prediction works是因为token已经是abstracted representation of meaning；V-JEPA把同样的philosophy带到video——在abstract representation space做next-state prediction，避免pixel-level mode collapse。

但这引出一个deep question：什么是"right level of abstraction"？

V-JEPA 2用encoder自己学出来的representation，但这个representation是否capture了planning所需的所有信息？比如：
- Object permanence（Figure 15b显示学到了——open gripper时cup不动）
- Physical constraints（gripper force limit? deformable object dynamics?）
- Long-range dependencies（"5分钟前我把钥匙放哪了"）

可以用MVP-style minimal pair benchmark做更精细的probing。MVP (https://arxiv.org/abs/2410.01867) 这种minimal video pair设计很clever——同一对video只differ in一个关键temporal event，其他都一样，强迫model真正理解dynamics而不是靠appearance bias蒙。

V-JEPA 2在MVP上拿44.5 paired accuracy（SOTA），说明dynamics理解确实学到了。但还有55.5%做不对——这部分是representation还是predictor的limitation？做个ablation：用V-JEPA 2 encoder + 更大的predictor，看MVP能不能涨。如果能涨，说明是predictor bottleneck；如果不涨，说明是representation没capture到那些fine-grained dynamics。

---

## 十一、最最后的人话总结

这篇paper干了一件这样的事：

**把"看视频学世界"和"在脑子里想象+规划"这两件事拼起来了，第一次证明能在真实机器人上zero-shot work。**

关键决定：
1. 在representation space而不是pixel space做prediction——快、准、smooth
2. Stage-wise training——先self-supervised学general representation，再freeze+lightweight post-train学action-conditioned dynamics
3. Planning用MPC in imagination——search action让imagined trajectory match goal

惊人结果：
- 62小时unlabeled robot video → zero-shot deploy到两个新lab
- Pick-and-place 65-80% success rate
- 比Cosmos这种7B pixel-generation world model快15× planning
- 比Octo这种BC-based VLA在object interaction task上好得多

残留问题：
- Camera sensitive
- Long horizon要手动sub-goal
- 没language goal
- Planning 16秒/action对reactive task太慢

但作为一个proof-of-concept，它把LeCun的JEPA路线从"理论proposal"推到了"real robot zero-shot deployment"——这是质变。如果接下来有人接通language goal + gradient-based planning + 10B scale，这条线可能就是通向autonomous embodied AI最有希望的路径之一。

---

## 关键参考链接

- V-JEPA 2 paper: https://arxiv.org/abs/2506.07963
- V-JEPA 2 code: https://github.com/facebookresearch/vjepa2
- Blogpost: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks
- LeCun JEPA proposal: https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA (原版): https://arxiv.org/abs/2404.08471
- Droid dataset: https://arxiv.org/abs/2403.12945
- Cosmos: https://arxiv.org/abs/2501.03575
- Octo: https://arxiv.org/abs/2405.12213
- MVP benchmark: https://arxiv.org/abs/2410.01867
- PerceptionTest: https://arxiv.org/abs/2305.13745
- DINO-WM: https://arxiv.org/abs/2411.04983
- CEM: https://www.sciencedirect.com/science/article/pii/S037722179700882X
- FLARE: https://arxiv.org/abs/2505.15659
- CoT-VLA: https://arxiv.org/abs/2503.22020

---

# V-JEPA 2 深度讲解：从 Self-Supervised Video 到 Zero-Shot Robot Planning

Andrej, 这篇paper是FAIR在LeCun的JEPA路线上的一个里程碑式工作，它把self-supervised video pretraining与robot manipulation zero-shot deployment串了起来。我从intuition、architecture、objective、scaling recipe、planning loop、到实验数据逐层拆解，最后给一些critique和open questions。

---

## 1. High-Level Story: 为什么这件事重要

核心claim：用 **JEPA (Joint-Embedding Predictive Architecture)** 在web-scale video上做self-supervised pretraining，学到的是"abstract world model"而不是pixel-level生成；再叠加极少量的unlabeled robot interaction data (62小时Droid)，就能zero-shot部署到两个从未见过的lab的Franka arm上做pick-and-place。整个过程 **没有task-specific reward、没有from-scratch teleoperation in target lab、没有language supervision during pretraining**。

这件事的重要性在于：它验证了LeCun多年主张的 **"world model + planning via latent imagination"** 路线在真实机器人上是可行的，且data efficiency惊人。与VLA (Vision-Language-Action) 路线如RT-2, OpenVLA, π0, Gr00t N1等的根本差异在于——VLA本质是behavior cloning，学的是"看到X就执行Y"的reactive policy，没有显式的predictive model；而V-JEPA 2学的是"dynamics in latent space"，用MPC在imagination里search actions。两条路线philosophy不同：一个是 **imitation**, 一个是 **model-based planning**。

参考链接：
- Paper: https://arxiv.org/abs/2506.07963 (V-JEPA 2)
- Code: https://github.com/facebookresearch/vjepa2
- Blogpost: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks
- LeCun的JEPA原始proposal: https://openreview.net/pdf?id=BZ5a1r-kVsf

---

## 2. V-JEPA 2 Pretraining: 在 Representation Space 做 Mask Denoising

### 2.1 Objective 公式拆解

核心objective（公式1）：

$$
\min_{\theta, \phi, \Delta_y} \left\| P_\phi(\Delta_y, E_\theta(x)) - \text{sg}(E_{\bar{\theta}}(y)) \right\|_1
$$

变量解释：
- $x$: masked view of video $y$，即从原video里drop掉一部分patches
- $y$: 原始video（target view，通常用不同的masking pattern）
- $E_\theta(\cdot)$: encoder (ViT-g, 1B params)，参数为 $\theta$
- $P_\phi(\cdot)$: predictor (ViT-s, 22M params)，参数为 $\phi$
- $\Delta_y$: learnable mask token，标示被drop掉的patch的位置
- $\bar{\theta}$: encoder weights $\theta$ 的EMA (exponential moving average)，teacher
- $\text{sg}(\cdot)$: stop-gradient，防止representation collapse
- $\|\cdot\|_1$: L1 loss，只在masked patches位置上计算

**Intuition**：和MAE很像，但关键区别——MAE在pixel space重建，JEPA在representation space预测。Pixel-level重建会强迫模型memorize "哪一片草地的哪根草"这种不可预测的高频细节；JEPA只要求predictor在abstract feature space里填空，unpredictable details被自然忽略掉。这正是LeCun反复强调的 **"放弃生成像素，预测抽象"** 的具体instance。

防止collapse的两个机制：
1. **EMA teacher**: target不是由当前 $\theta$ 生成，而是由慢变的 $\bar{\theta}$ 生成，类似BYOL/DINO/MoCo的bootstrap思路。EMA decay = 0.99925。
2. **Stop-gradient on target**: 梯度只流过predictor和encoder，不更新teacher。如果没有这两个trick，model可以直接把所有representation collapse到一个常数点使loss=0。

### 2.2 Architecture 细节

| Component | 架构 | 参数量 | Width | Depth | Heads | MLP dim |
|---|---|---|---|---|---|---|
| Encoder $E_\theta$ (ViT-L) | ViT | 300M | 1024 | 24 | 16 | 4096 |
| Encoder $E_\theta$ (ViT-H) | ViT | 600M | 1280 | 32 | 16 | 5120 |
| Encoder $E_\theta$ (ViT-g) | ViT | 1B | 1408 | 40 | 22 | 6144 |
| Predictor $P_\phi$ | ViT-s | 22M | 384 | 12 | 12 | 1536 |

**Patchification**: video被切成tubelet size $2 \times 16 \times 16$ (T × H × W)，即时间维度2帧一组，空间16×16。这意味着一个16帧的256×256 video产生 $8 \times 16 \times 16 = 2048$ 个tokens。

**Position encoding**: 换成 **3D-RoPE** (Rotary Position Embedding)。把feature dim切成三段（temporal/height/width各占约1/3），每段独立做1D rotation。和原版V-JEPA用的sincos absolute PE相比，3D-RoPE在大模型上更稳定，对resolution change也更友好（这点对progressive resolution training至关重要）。

参考：3D-RoPE在Su et al. 2024 RoFormer里：https://arxiv.org/abs/2104.09864

**Masking strategy**: multi-block masking (Bardes et al. 2024)，spatial mask scale ∈ [0.15, 0.7]，aspect ratio ∈ [0.75, 1.5]，temporal mask scale = [1.0, 1.0]（即不做temporal masking在主phase）。

### 2.3 Scaling Ingredients 四件套

这是这篇paper在engineering上的核心贡献，回答了"如何把V-JEPA从ViT-H/2M videos scale到ViT-g/22M videos":

1. **Data scaling**: 2M → 22M videos (VM22M)
2. **Model scaling**: 300M (ViT-L) → 1B (ViT-g)
3. **Longer training**: 90K → 252K iterations
4. **Higher resolution + longer clips**: 256×256×16frames → 384×384×64frames

累计贡献：每个ingredient单独贡献 +1.0, +1.5, +0.8, +0.7 average points across 6 tasks，合计约+4 points over baseline ViT-L。

### 2.4 VM22M Dataset 构成

| Source | Samples | Type | Hours | Weight |
|---|---|---|---|---|
| SSv2 | 168K | EgoVideo | 168 | 0.056 |
| Kinetics (400/600/700) | 733K | ExoVideo | 614 | 0.188 |
| HowTo100M | 1.1M | ExoVideo | 134K | 0.318 |
| YT-Temporal-1B (curated) | 19M | ExoVideo | 1.6M | 0.188 |
| ImageNet | 1M | Images | n/a | 0.250 |

**Curation**：YT1B未curate版本有1.4M小时video，里面有大量cartoon、clipart、slide show等噪声。他们用DINOv2 ViT-L对每个scene的中帧提取embedding，做1.5M k-means clusters，然后用Kinetics/SSv2/COIN/EpicKitchens作为target distribution做cluster-based retrieval，最终保留210k clusters / 115M scenes。这个pipeline借鉴DINOv2的retrieval-based curation (Oquab et al. 2023)。

**Intuition**：Image data被duplicate成16-frame identical video混入训练，权重0.25——这是为了补video数据里appearance/texture覆盖度不够的问题。Video data天然偏motion，image data补appearance，二者互补。这和DINOv2、SigLIP2等纯image SSL路线相比，是V-JEPA 2在appearance任务上保持competitive的关键。

DINOv2 paper: https://arxiv.org/abs/2304.07193

### 2.5 Progressive Resolution Training Schedule

这是scaling到384×384×64frames的关键trick。如果从头在384×384×64frames上训练ViT-g，需要约60 GPU-years。他们做：

- **Warmup phase (12K iter)**: 16 frames, 256×256, linear LR warmup
- **Constant phase (228K iter)**: 同上，constant LR = 5.25e-4
- **Cooldown phase (12K iter)**: 升到64 frames, 256×256或384×384，linear LR decay to 1e-6

结果：8.4× speedup，性能不掉。这是因为constant phase后representation已经基本学好，cooldown阶段只是adapt到high-resolution input distribution。

LR schedule从cosine换成warmup-constant-decay的好处：可以reuse constant phase的checkpoint跑多个cooldown ablation，做HPO很便宜。这点Hägele et al. 2024 (https://arxiv.org/abs/2405.18392) 也验证过。

---

## 3. V-JEPA 2-AC: Action-Conditioned World Model

这是从"看视频"到"能act"的关键一跃。

### 3.1 设定

数据：Droid dataset (Khazatsky et al. 2024, https://arxiv.org/abs/2403.12945) 的raw videos，只用left exocentric camera，filter掉短于4秒的clip，剩 **62 hours** video。注意——只用video + 7-DoF end-effector state，**不用reward、不用task label、不用success flag**。

输入设定：
- 4秒clip @ 4fps → 16 frames
- Resolution 256×256
- Frame $x_k \in \mathbb{R}^{256\times256\times3}$
- End-effector state $s_k \in \mathbb{R}^7$ (3 pos + 3 Euler + 1 gripper)
- Action $a_k = s_{k+1} - s_k \in \mathbb{R}^7$ (相邻帧end-effector变化)

### 3.2 Loss Function 详解

**Teacher Forcing Loss (公式2)**：

$$
\mathcal{L}_{\text{tf}}(\phi) := \frac{1}{T}\sum_{k=1}^{T} \left\| \hat{z}_{k+1} - z_{k+1} \right\|_1 = \frac{1}{T}\sum_{k=1}^{T} \left\| P_\phi\left((a_t, s_t, E(x_t))_{t \le k}\right) - E(x_{k+1}) \right\|_1
$$

变量：
- $T = 15$ (16帧 → 15个transitions)
- $z_k = E(x_k) \in \mathbb{R}^{16\times16\times1408}$：每帧通过frozen V-JEPA 2 ViT-g得到的feature map，spatial 16×16, channel 1408
- $\hat{z}_{k+1}$: predictor在看到 $t \le k$ 的所有信息后，对下一帧representation的预测
- $P_\phi$: 300M param transformer, 24 layers, 16 heads, 1024 hidden dim
- 关键：encoder $E$ 是 **frozen** 的，只训练 $\phi$

**Rollout Loss (公式3)**：

$$
\mathcal{L}_{\text{rollout}}(\phi) := \left\| P_\phi(a_{1:T}, s_1, z_1) - z_{T+1} \right\|_1
$$

变量：
- 这里 $P_\phi(a_{1:T}, s_1, z_1)$ 表示从 $(s_1, z_1)$ 出发，**autoregressively rollout** T步得到最终状态预测
- 实践中 $T=2$，即只differentiate one recurrent step
- $z_{T+1}$ 是ground truth第T+1帧的encoder output

**Intuition**：纯teacher forcing会让model在训练时永远看ground truth历史，但inference时需要自己rollout，会有exposure bias。Rollout loss虽然只展开1步，但让模型提前适应"看自己上一时刻预测的输出"。这是video prediction / world model里的经典trick，类似scheduled sampling。

**Total loss (公式4)**：

$$
L(\phi) := \mathcal{L}_{\text{tf}}(\phi) + \mathcal{L}_{\text{rollout}}(\phi)
$$

### 3.3 Predictor 架构: Block-Causal Attention

Predictor输入序列：$(a_k, s_k, z_k)_{k \in [15]}$，每个token经过separate的learnable affine projection到1024维。

**Block-causal attention pattern**：在时间维度上causal（不能看未来），但在同一时间步内的spatial patches之间是bidirectional的。这样设计的原因：
- 因果性（causality）保证可以autoregressive rollout
- 同时间步spatial bidirectional让model能整合一帧内的全局信息，不必被迫做raster-scan那种低效的顺序

Position encoding：
- Video patches用3D-RoPE（temporal/height/width三段）
- Action和pose tokens只用temporal段（它们没有spatial坐标）

### 3.4 为什么这样设计能work：Intuition

整个pipeline的intuition链：
1. V-JEPA 2 pretraining让encoder学到"physical world的abstract state"——这里有物体、有机器人、有空间关系，但丢掉了像素级噪声
2. Frozen encoder输出 $z_k$ 就是一个稳定的"state abstraction"，预测这个abstraction比预测pixel容易得多（维度低1408 vs 几万pixel，且语义稳定）
3. Action-conditioned predictor学的是"在这个abstract state下，执行action $a_k$ 会走到哪个abstract state"
4. 因为abstract space已经过滤掉unpredictable details，predictor只需focus on "action对哪些物体、机器人、场景元素产生了可预测的变化"
5. Planning时在abstract space里search action sequence，让imagined trajectory match goal image的abstract representation

这条chain的每个环节都避免了pixel-level generation方法（如Cosmos, GAIA-1, Genie）的痛点：生成pixel需要建模所有不可预测的细节，diffusion sampling又慢得离谱。

---

## 4. Planning: 通过 Energy Minimization 做 MPC

### 4.1 能量函数 (公式5)

$$
\mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g) := \left\| P(\hat{a}_{1:T}; s_k, z_k) - z_g \right\|_1
$$

变量：
- $z_k = E(x_k)$: 当前帧representation
- $s_k$: 当前end-effector state
- $z_g = E(x_g)$: goal image的representation
- $\hat{a}_{1:T}$: 待优化的action sequence（horizon T）
- $P(\hat{a}_{1:T}; s_k, z_k)$: 从 $(s_k, z_k)$ 出发rollout T步得到的imagined final state representation

**Optimization**：

$$
(a_i^\star)_{i \in [T]} := \arg\min_{\hat{a}_{1:T}} \mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g)
$$

用 **Cross-Entropy Method (CEM, Rubinstein 1997)** 求解：
- 每个planning step从一系列Gaussian distributions采样action trajectory
- 用population statistics of top-k samples更新Gaussian的mean和variance
- 多轮refine后返回mean作为selected action
- Execute first action, observe new state, re-plan (receding horizon control / MPC)

CEM参考：https://www.sciencedirect.com/science/article/pii/S037722179700882X

### 4.2 实际planning参数

| Method | #Samples | Iterations | Horizon | Time/action |
|---|---|---|---|---|
| Cosmos (latent diffusion 7B) | 80 | 10 | 1 | **4 minutes** |
| V-JEPA 2-AC | 800 | 10 | 1 | **16 seconds** |

Action被constrain在以原点为中心、半径0.075的L1-ball里——单步最大end-effector displacement约13cm。这是为了避免sample到训练分布外的极端action。

### 4.3 Multi-Goal Pick-and-Place 的 Sub-Goal Schedule

Pick-and-place需要长horizon planning，但他们的planning horizon只用1。Trick是用 **3个sub-goal images**：
1. Sub-goal 1: 物体被grasped
2. Sub-goal 2: 物体到达target位置附近
3. Sub-goal 3 (final): 物体放在target位置

Schedule: 4 steps对sub-goal 1 → 10 steps对sub-goal 2 → 4 steps对sub-goal 3。手动engineer的sub-goal decomposition，相当于把长horizon task切成3个短horizon task。

**Limitation**：这个approach不够general，需要人为提供sub-goals。paper在Section 4.3和9明确指出这是future work——需要hierarchical planning或language-conditioned goals。

---

## 5. 实验结果：Zero-Shot Robot Manipulation

### 5.1 主表（Table 2）

| Method | Reach | Grasp Cup | Grasp Box | Reach w/ Obj Cup | Reach w/ Obj Box | Pick-&-Place Cup | Pick-&-Place Box |
|---|---|---|---|---|---|---|---|
| Octo (avg) | 100% | 15% | 0% | 15% | 70% | 15% | 10% |
| V-JEPA 2-AC (avg) | **100%** | **65%** | **25%** | **75%** | **75%** | **80%** | **65%** |

Octo baseline：在Open-X-Embodiment (1M+ trajectories)上pretrain，然后在Droid上fine-tune with behavior cloning + hindsight relabeling。结果：reach很好，但涉及object interaction的task大幅掉——这是BC-based VLA的通病，BC学到的是"demonstrator会做什么"，对precise gripper control、object dynamics的compositional generalization不够强。

V-JEPA 2-AC：62小时unlabeled video post-training，没有reward、没有task label、没有success/failure信号，但pick-and-place cup 80% / box 65%，跨两个lab平均。

### 5.2 Planning Speed 对比 (Table 3)

Cosmos 7B做planning：每action 4分钟。Pick-and-place trajectory 18步+ → 1小时+ robot execution。V-JEPA 2-AC：每action 16秒。**15×加速**，且success rate更高（pick-and-place cup 80% vs Cosmos 0%）。

**Intuition**：Latent imagination vs pixel generation的本质区别。Pixel diffusion需要iterative denoise几十步，每步走一遍7B network；latent prediction是single forward pass + 几步autoregressive rollout。Energy evaluation时V-JEPA 2-AC只需一次forward，Cosmos要跑完整diffusion。

### 5.3 Energy Landscape 可视化 (Figure 9)

在 Δx-Δy 平面上sweep，固定 Δz=0，画出energy function。Ground truth action在 (0, -0.1)，energy minimum在 (0, -0.05)。Energy landscape是 **smooth and locally convex** 的——这一点非常关键，意味着CEM这种sampling-based optimizer能可靠找到global minimum。如果landscape是multi-modal + flat region + cliff，CEM会陷入local minimum或者sample效率崩溃。

Smoothness的来源：V-JEPA 2 representation space本身就是"abstract state"，对action的response在abstract space里自然是Lipschitz连续的。Pixel space里landscape会非常noisy（每一点pixel variation都贡献loss）。

---

## 6. Understanding: Probe-Based Classification

### 6.1 主表 (Table 4) - 6 Tasks Average

| Method | Params | Avg | SSv2 | Diving-48 | Jester | K400 | COIN | IN1K |
|---|---|---|---|---|---|---|---|---|
| DINOv2 | 1.1B | 81.1 | 50.7 | 82.5 | 93.4 | 83.6 | 90.7 | 86.1 |
| PE_core G | 1.9B | 82.3 | 55.4 | 76.9 | 90.0 | 88.5 | 95.3 | 87.6* |
| SigLIP2 | 1.2B | 81.1 | 49.9 | 75.3 | 91.0 | 87.3 | 95.1 | 88.0 |
| V-JEPA (原版) ViT-H | 600M | 85.2 | 74.3 | 87.9 | 97.7 | 84.5 | 87.1 | 80.0 |
| InternVideo2 s2-1B | 1B | 87.0 | 69.7 | 86.4 | 97.0 | 89.4 | 93.8 | 85.8 |
| **V-JEPA 2 ViT-g** | 1B | **87.5** | **75.3** | **90.1** | **97.7** | 86.6 | 90.7 | 84.6 |
| **V-JEPA 2 ViT-g384** | 1B | **88.2** | **77.3** | **90.2** | **97.8** | 87.3 | 91.1 | 85.1 |

**关键观察**：
- SSv2 (motion-heavy): V-JEPA 2 77.3 vs DINOv2 50.7 vs PE_core 55.4 vs InternVideo2 69.7 → **领先+20 points**
- ImageNet (appearance): V-JEPA 2 85.1 vs SigLIP2 88.0 → 落后约3 points，但还competitive
- DINOv2/PE_core/SigLIP2都是image encoders，做video task时按frame独立encode后concatenate
- V-JEPA 2 **没有任何language supervision**，但仍能在video QA上超过SigLIP2/PE——这点打破了"做video QA必须有language-aligned visual encoder"的conventional wisdom

### 6.2 Attentive Probe 架构

4-layer transformer：
- 前3层：standard self-attention, 16 heads
- 第4层：cross-attention with learnable query token
- Output → linear classifier

Probe只在frozen encoder output上训练，不update encoder weights。这比linear probe更强，但比full fine-tune更弱——是介于二者之间的"attentive probe"范式，Bardes et al. 2024首先用。

对Jester/Diving-48两个task用multi-layer strategy：取encoder的4个中间层（如ViT-g的layer 24, 29, 34, 39）的tokens一起送给probe——deeper layers提供更abstract semantics。

---

## 7. Prediction: EK100 Action Anticipation

EK100 task: 给一段context video（结束于某action开始前1秒），预测1秒后会执行什么action。Metric是mean-class recall@5。

| Method | Params | Verb | Noun | Action |
|---|---|---|---|---|
| InAViT | 160M | 51.9 | 52.0 | 25.8 |
| Video-LLaMA | 7B | 52.9 | 52.0 | 26.0 |
| PlausiVL | 8B | 55.6 | 54.2 | 27.6 |
| V-JEPA 2 ViT-L | 300M | 57.8 | 53.8 | 32.7 |
| V-JEPA 2 ViT-g384 | 1B | **63.6** | **57.1** | **39.7** |

V-JEPA 2-AC ViT-g384 (1B) 比 PlausiVL (8B) 高出 **+12.1 points** recall@5 on action，relative improvement 44%。

### 7.1 Anticipation Probe 设计

很巧妙：除了用encoder输出，还用V-JEPA 2的predictor预测"1秒后的frame representation"，把这个predicted representation和encoder output concat后送probe。Probe有3个query tokens，分别预测verb/noun/action三类，分别用focal loss训练。

公式上，相当于：

$$
\hat{z}_{\text{future}} = P_\phi(\Delta_y, E_\theta(x_{\text{context}}))
$$

这里 $\Delta_y$ 是对应"未来1秒"那个时间步的mask token。Probe输入是 $[\text{tokens}(E_\theta(x_{\text{context}})); \text{tokens}(\hat{z}_{\text{future}})]$。

**Intuition**：pure supervised anticipation baselines (InAViT)用hand-crafted hand-object interaction features；PlausiVL/Video-LLaMA靠LLM的common sense补全。V-JEPA 2靠 **学到的video dynamics prior** 直接预测未来state——这是prediction真正发生的地方，不靠language补全。

### 7.2 Long-Horizon Prediction 衰减 (Figure 18 left)

Anticipation time从1s → 2s → 4s → 10s，recall@5大幅下降。这是world model的fundamental limitation——video是非确定性的，1秒后可能wash sink, turn on water, clean wall都合理（paper里这个example特别clear），10秒后possibility space爆炸。这和人类认知一致：预测"1秒后我会做什么"容易，"10秒后我会做什么"难。

---

## 8. Video Question Answering: 对齐 LLM

### 8.1 Controlled Setup (Table 6)

固定Qwen2-7B-Instruct作为LLM backbone，固定18M image/video-text pairs做alignment，比较不同vision encoder。Vision encoder frozen。

| Method | Enc/LLM | Avg | PerceptTest | MVP | TempCompass | TemporalBench | TVBench | TOMATO | MVBench |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2 ViT-g518 | 1.1B/7B | 45.7 | 67.1 | 22.4 | 62.3 | 26.8 | 47.6 | 32.0 | 61.8 |
| SigLIP2 ViT-g384 | 1.1B/7B | 48.1 | 72.4 | 26.2 | 66.8 | 25.7 | 48.7 | 33.2 | 64.0 |
| PE ViT-G/14/448 | 1.9B/7B | 49.1 | 72.3 | 26.7 | 67.0 | 27.5 | 51.6 | 34.0 | 64.7 |
| **V-JEPA 2 ViT-g512** | **1B/7B** | **52.3** | 72.0 | **31.1** | **69.2** | **33.3** | **55.9** | **37.0** | **67.7** |

**关键观察**：
- V-JEPA 2在temporal reasoning benchmarks (MVP, TempCompass, TemporalBench, TVBench, TOMATO)上一致领先
- PerceptionTest基本打平，说明appearance-heavy task里language-aligned encoder还有优势
- **V-JEPA 2完全无language supervision pretrain**，居然能在VidQA上超过SigLIP2/PE——这点很反常识

### 8.2 Scaling Data to 88.5M (Table 8)

换Llama 3.1 8B backbone，用PerceptionLM的训练recipe，scale alignment data到88.5M:

| Method | Enc/LLM | PerceptTest | MVP | TempCompass | TemporalBench | TOMATO | TVBench | MVBench |
|---|---|---|---|---|---|---|---|---|
| InternVL-2.5 | 300M/7B | 68.9 | 39.9 | 68.3 | 24.3 | 29.4 | 61.6 | 72.6 |
| Qwen2.5VL | 1B/7B | 70.5 | 36.7 | 71.7 | 24.5 | 24.6 | 50.5 | 69.6 |
| PerceptionLM 8B | 1B/8B | 82.7 | 39.7 | 72.7 | 28.3 | 33.2 | 63.5 | **77.1** |
| **V-JEPA 2 + Llama 3.1 8B** | 1B/8B | **84.0** | **44.5** | **76.9** | **36.7** | **40.3** | 60.6 | 73.5 |

SOTA on 5个benchmarks。在PerceptionTest/MVP/TempCompass/TemporalBench/TOMATO上全面提升，TVBench/MVBench上输给PLM但还competitive。

**Intuition**：MVP (Krojer et al. 2024, https://arxiv.org/abs/2410.01867) 是minimal video pair benchmark，专门设计来避免text bias和appearance bias——它要求真正理解video dynamics。V-JEPA 2在此拿到SOTA说明其temporal abstraction确实是dynamics-aware的，不仅仅是frame appearance的堆叠。

---

## 9. 相关工作脉络 & V-JEPA 2 的位置

### 9.1 World Model 谱系

- **Pixel-space world model**: Visual Foresight (Finn et al. 2017, Ebert et al. 2018), Gaia-1/2 (Hu et al. 2023, Russell et al. 2025), Cosmos (Agarwal et al. 2025, https://arxiv.org/abs/2501.03575), Genie (Bruce et al. 2024) — 在pixel space直接生成，diffusion或autoregressive
- **Latent dynamics world model**: E2C (Watter et al. 2015), World Models (Ha & Schmidhuber 2018), Dreamer series (Hafner et al. 2019a/b, 2023), TD-MPC2 (Hansen et al. 2023), DINO-WM (Zhou et al. 2024, https://arxiv.org/abs/2411.04983)
- **JEPA-based**: I-JEPA (Assran et al. 2023), V-JEPA (Bardes et al. 2024), V-JEPA 2 (本文), DINO-WM

V-JEPA 2相对于DINO-WM的差异：DINO-WM用DINOv2 features + small world model + small-scale task evaluation；V-JEPA 2是从pretrain到deploy end-to-end设计，用JEPA objective（mask denoising in representation space）而非DINOv2的self-distillation，且做了real robot zero-shot deployment。

### 9.2 VLA (Vision-Language-Action) 谱系

- RT-1/RT-2 (Brohan et al. 2022/2023)
- OpenVLA (Kim et al. 2024, https://arxiv.org/abs/2406.09246)
- Octo (Octo Model Team 2024, https://arxiv.org/abs/2405.12213)
- π0 / π0.5 (Black et al. 2024/2025, https://arxiv.org/abs/2410.24164, https://arxiv.org/abs/2504.16054)
- Gr00t N1 (Bjorck et al. 2025, https://arxiv.org/abs/2503.14734)

VLA的哲学：BC + 大规模teleop data + internet VLM knowledge → reactive policy
V-JEPA 2-AC的哲学：unlabeled video + 62h unlabeled interaction → world model + MPC planning

两条路线本质不同：VLA学的是"input → action"映射，inference是single forward；V-JEPA 2-AC学的是"input + action → next state"的dynamics，inference需要在imagination里search。前者快但需要task-specific demos，后者慢但可以zero-shot generalize到没见过的task。

### 9.3 V-JEPA 2 vs Cosmos：Planning Speed 的本质差异

Cosmos是7B latent diffusion world model，pretrained on 20M hours video。在Droid上action-conditioned fine-tune后，理论上也能做planning。本文Table 3显示Cosmos planning每action 4分钟，V-JEPA 2-AC每action 16秒，且success rate远高。

**根本原因**：
- Cosmos每eval一个action trajectory要跑完整diffusion sampling (10+ steps × 7B network)
- V-JEPA 2-AC每eval一个action trajectory只需一次forward (300M predictor + autoregressive rollout)
- 此外Cosmos在pixel space生成，包含很多对task-irrelevant的细节，energy landscape更noisy，CEM sample效率低

但Cosmos的优势是生成的video更"漂亮"，可以做visualization、simulation等用途；V-JEPA 2-AC不能直接可视化imagination（必须额外训一个frame decoder，如Appendix B.3所示）。

---

## 10. Critical Analysis & Open Questions

### 10.1 强的地方

1. **Data efficiency惊人**：62小时unlabeled Droid video → zero-shot deploy到两个未见过的lab。这是world model + planning路线相对于VLA的key advantage。
2. **Stage-wise training合理**：先action-free pretrain学representation，再freeze encoder + 训action-conditioned predictor——两个stage decouple了"理解世界"和"理解action effect"，engineering上可复用。
3. **Smooth energy landscape** (Figure 9)：latent space planning的物理直觉很好验证——landscape smooth + locally convex，CEM收敛可靠。
4. **Cross-task generalization**：V-JEPA 2同一encoder同时撑住SSv2/EK100/VidQA/Robot manipulation——representation确实是general-purpose的。

### 10.2 弱的地方 & 限制

1. **Camera sensitivity (Section 4.3, Appendix B.4)**：因为没有explicit calibration，model要从monocular RGB infer action coordinate axis，对camera position敏感。Appendix B.4实验显示rotation error几乎linearly scale with camera position offset。这是部署的real-world痛点——VLA模型不需要这个，因为BC时已经learn了camera-conditioned policy。
2. **Long horizon planning受限**：autoregressive rollout有error accumulation（Figure 15a能看到cup位置预测drift），且search space exponential blow up。目前必须靠sub-goal decomposition，需要人手设计sub-goals。
3. **Planning速度仍比VLA慢**：16秒/action vs Octo的real-time。对需要reactive closed-loop control的task（如接抛物、双指旋转物体）会太慢。
4. **No language goal**：只能用image goal。论文Section 9明确说future work要align with language。这点在VidQA部分（Section 7）已经做了一半，但还没接进world model。
5. **ImageNet输给SigLIP2/PE**：V-JEPA 2在appearance-only task上仍有差距。如果想做OCR、document understanding这类fine-grained appearance task，language-aligned encoder可能仍是首选。
6. **Gripper state 7-DoF hardcoded**：action space是固定的7D end-effector delta，对于不同robot morphology需要重新定义action space。不像π0/Gr00t那种"any-robot"的VLA。

### 10.3 我的Intuition: 这条路会怎么走

1. **Hierarchical planning**：未来肯定会有high-level planner（可能是个LLM-based）生成sub-goals（language或image），low-level V-JEPA 2-AC负责execute。类似SayCan + RT-2的hierarchy但更clean——high-level reasoning和low-level dynamics真正decouple了。
2. **Joint train action-conditioned predictor from scratch**：当前是先action-free pretrain再action-conditioned post-train，但理论上可以混合训练。Sobal et al. 2025 (https://arxiv.org/abs/2502.14819) 和Zhou et al. 2024 (DINO-WM) 探索了这条路。
3. **Gradient-based planning**：现在用CEM (sampling-based)，因为latent space energy landscape smooth，其实可以做gradient descent on action space——predictor是differentiable的。可能再加速10×。
4. **V-JEPA 2-AC + Diffusion policy hybrid**：用V-JEPA 2-AC做imagination rollout + 用diffusion policy做action proposal + 用world model做filter/refine。最近Zheng et al. 2025 (FLARE, https://arxiv.org/abs/2505.15659) 和Zhao et al. 2025 (CoT-VLA, https://arxiv.org/abs/2503.22020) 在这个方向。
5. **Longer video context**: 现在cooldown phase到64 frames (16秒)。如果scale到1000+ frames，可能可以做真正长horizon planning。memory mechanism（类似RAG或hierarchical state abstraction）是key。
6. **Push to 10B+ params**: V-JEPA 2到1B就停了，但scaling curve看起来没saturate。Carreira et al. 2024 (https://arxiv.org/abs/2412.15212) 探索了4D representation scaling到更大。如果能scale到20B，intuition是representation quality会再上台阶，所有downstream task跟着水涨船高。

### 10.4 和我自己的工作联系

Andrej, 你之前讲过"mode collapse in next-token prediction"和LLM的hallucination问题——V-JEPA路线是对"pixel-level next-frame prediction"的批评和修正。LLM在discrete token space做next-token prediction works因为token是abstracted；V-JEPA把同样的philosophy带到video——在abstract representation space做next-state prediction，避免pixel-level mode collapse（要么blurred要么jumping between modes）。

但这引出一个deep question：什么是"right level of abstraction"？V-JEPA 2用encoder自己学出来的representation，但这个representation是否capture了planning所需的所有信息？比如object permanence、physical constraints、gripper-object contact——Figure 15b显示model对open vs closed gripper的prediction不同（open时cup不动），说明physics intuition学到了；但gripper force、deformable object dynamics等更复杂physics可能没学到。可以用MVP-style minimal pair benchmark做更精细的probing。

---

## 11. 一些 Implementation Details 容易被忽略

1. **Predictor只300M params**：encoder 1B但predictor只有300M。这和I-JEPA的发现一致——predictor不必大，因为它只需"在已知representation space做interpolation"，不需要重新learn semantics。
2. **EMA decay = 0.99925**：很慢的EMA。fast EMA会让teacher追上student太快，slow EMA让teacher提供稳定的target。Bardes et al. 2024 ablate过。
3. **No dropout in predictor**：因为是regression task，dropout会引入noise让training不稳定。
4. **Frozen encoder during V-JEPA 2-AC training**：这很重要——如果unfreeze encoder，predictor会"作弊"把representation改成容易预测的形式，破坏general-purpose utility。
5. **Two-step rollout only**：训练时只rollout 2步。再多步会让gradient unstable（vanishing/exploding through recurrent steps）。
6. **Action L1-ball radius 0.075**：相当于13cm max displacement per step。这是从Droid训练数据统计来的——超过这个范围的action是OOD。
7. **Sub-goal timing (4-10-4)**：手动调的。未来应该让high-level planner自动decide何时切换sub-goal。
8. **Frame decoder for visualization**：Appendix B.3专门训一个feedforward ViT-L decoder把V-JEPA 2 representation decode回pixel——只是为了visualization，不是inference的一部分。这和MAE-style reconstruction decoder不同，它只是interpretability tool。

---

## 12. 总结：What to Take Away

V-JEPA 2证明了一个完整的chain：

> **Self-supervised video pretraining (1M+ hours, no labels) → general-purpose visual representation → freeze + lightweight action-conditioned predictor (62h unlabeled robot video) → latent world model → MPC planning in representation space → zero-shot deployment to unseen robots in unseen labs**

这件事在3年前还像是LeCun的blue-sky proposal，现在变成可复现的recipe（code + weights都open）。它给的两个关键insight：

1. **Latent imagination > pixel generation for control**：在representation space做planning比在pixel space快15×且更准，因为abstract state过滤了unpredictable details，energy landscape更smooth。
2. **Self-supervised > supervised for world models**：互联网video是无穷的unlabeled observation data，(robot) interaction data是稀缺的。把它们stage-wise combine，能avoid supervised RL的data bottleneck。

接下来18个月，我会期待看到：language-conditioned V-JEPA 2-AC、gradient-based planning、10B+ scale V-JEPA 2、hierarchical sub-goal generation、multi-robot morphology generalization。这条线如果走通，是通向LeCun设想的"autonomous machine intelligence"的最有希望路径之一。

---

## Reference Links 汇总

- V-JEPA 2 paper: https://arxiv.org/abs/2506.07963
- V-JEPA 2 code: https://github.com/facebookresearch/vjepa2
- Meta blogpost: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks
- V-JEPA (原版): https://arxiv.org/abs/2404.08471
- I-JEPA: https://arxiv.org/abs/2301.08243
- LeCun JEPA proposal: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Droid dataset: https://arxiv.org/abs/2403.12945
- DINOv2: https://arxiv.org/abs/2304.07193
- SigLIP2: https://arxiv.org/abs/2502.14786
- Perception Encoder: https://arxiv.org/abs/2504.13181
- PerceptionLM: https://arxiv.org/abs/2504.13180
- Cosmos: https://arxiv.org/abs/2501.03575
- Octo: https://arxiv.org/abs/2405.12213
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- Gr00t N1: https://arxiv.org/abs/2503.14734
- DINO-WM: https://arxiv.org/abs/2411.04983
- TD-MPC2: https://arxiv.org/abs/2310.16828
- Dreamer V3: https://arxiv.org/abs/2301.04104
- MVP benchmark: https://arxiv.org/abs/2410.01867
- PerceptionTest: https://arxiv.org/abs/2305.13745
- EK100: https://doi.org/10.1007/s11263-021-01531-2
- RoFormer (RoPE): https://arxiv.org/abs/2104.09864
- Cross-Entropy Method: https://www.sciencedirect.com/science/article/pii/S037722179700882X
- Hägele et al. (scaling schedules): https://arxiv.org/abs/2405.18392
- Sobal et al. (reward-free offline): https://arxiv.org/abs/2502.14819
- CoT-VLA: https://arxiv.org/abs/2503.22020
- FLARE: https://arxiv.org/abs/2505.15659
- Scaling 4D representations (Carreira et al.): https://arxiv.org/abs/2412.15212
- InternVideo2: https://arxiv.org/abs/2312.06846
- VideoMAEv2: https://arxiv.org/abs/2303.16719
