---
source_pdf: HowMuch3DDoVideoFoundation Models Encode.pdf
paper_sha256: 97fda46d74d7857a8933888a597380a7eaf7407a9e37f607f741cdbfc1889d52
processed_at: '2026-08-05T07:32:53-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

## 一句话概括

**现在的video生成模型，比如WAN2.1-14B，从来没见过3D数据，但它们内部居然已经"懂"3D了，而且懂的成都快赶上甚至超过专门训练做3D的模型。**

## 这帮人到底干了啥

想象你有个朋友，号称"画画特别好"，你不知道他到底understand不understand透视、空间、geometry这些3D的东西。怎么test？

**你给他看几帧视频，让他"凭直觉"画出这些frame对应的3D点云和camera trajectory。如果他画得准，说明他脑子里真的有3D model。**

paper就是这么干的。他们拿各种video foundation model（VidFMs），freeze住weights不动，从里面抽出features，然后train一个很轻的probe（就是个shallow transformer），让probe从这些features decode出三个东西：

1. **Point map**：每个pixel在3D空间里的xyz坐标（以第一帧为origin）
2. **Depth map**：每个pixel的深度
3. **Camera pose**：每帧相对第一帧的旋转和平移

probe很shallow，capacity有限，所以如果它能decode出准确3D，那说明3D information本来就在feature里了，backbone自己学到了。

## 为什么这事有意思

之前大家觉得video生成模型就是"frame predictor"——学会pixel-level synthesis，internal可能啥geometry都没有。Sora出来时OpenAI喊了句"video models are world simulators" [https://openai.com/research/video-generation-models-as-world-simulators]，但没给硬evidence。

这篇paper给了quantitative evidence：**WAN2.1-14B在DL3DV这个out-of-distribution的dataset上，3D reconstruction quality超过Fast3R——Fast3R是专门用3D data train的SOTA model。**

WAN从来没见过3D label，只见过2D video。但它内部feature encode的3D understanding比见过百万3D data的expert还强。这挺震撼的。

## 关键findings用大白话

### Finding 1：Temporal reasoning是灵魂

拿DINOv2（image model，per-frame extract feature）跟V-JEPA（video model）比，DINOv2的single-frame depth estimation其实不差——0.209 vs 0.214，差不多平手。但一比global 3D（point map、camera pose），DINOv2就崩了。

CO3Dv2上：DINOv2 Point 0.559，V-JEPA Point 0.439。
DL3DV上：DINOv2 Point 2.814，V-JEPA Point 1.576。

**Intuition**：单张图片里的depth cue（远小近大、遮挡、shading）能给你depth，但同一物体在不同帧的correspondence、camera怎么动的，光看单帧永远猜不出来。Video model让信息跨帧mixing，才能build起global 3D representation。这是image model和video model的本质区别。

### Finding 2：3D fine-tuning不是free lunch

Aether在CogVideoX基础上加了3D-aware loss finetune。在DL3DV（large scenes）上确实比base好，但在CO3Dv2（object-centric turntable videos）上反而更差了。

**Intuition**：Aether训练数据是game/simulator里的synthetic large scenes，跟CO3Dv2这种物体绕圈的video distribution不match。3D fine-tuning让模型specialize到training distribution，反而hurt了原本broad的generalization。

这跟LLM里看到的phenomenon很像——你用某个domain的instruction tuning能让模型在那个domain上更强，但broad capability可能degrade。Base model的"未specialized"状态反而保留更多general knowledge。

### Finding 3：Scaling不是万能

WAN从1.3B scale到14B，3D awareness相对提升23%。
CogVideoX从2B scale到5B，3D awareness反而稍微降了2%。

**Intuition**：scale不just是parameter count。WAN-14B训练data可能也更多，CogVideoX-5B可能data没跟上。这跟Chinchilla [https://arxiv.org/abs/2203.15556]揭示的data-optimal scaling类似——parameter多了data也得跟上，否则overparameterized但undertrained。

3D awareness的scaling behavior跟generation quality的scaling behavior可能不一致。光看生成视频好看不mean 3D awareness强。

### Finding 4：Layer和timestep的sweet spot

对所有probed diffusion models，最优feature都来自**mid-layer + early-but-not-first timestep**。这个consistency很surprising。

**Layer的intuition**：
- 太early的layer：feature还没abstract到high-level，还停留在low-level texture
- 太late的layer：feature specialized到"接下来要生成什么pixel"，high-level 3D info被suppress
- Mid layer：刚刚好，已经abstract但还没specialize到pixel synthesis

**Timestep的intuition**：
Diffusion model timestep $\tau$决定注入多少noise。太early的timestep（接近clean signal）让denoising task太trivial，feature没work to do；太late的timestep（noise很大）input signal被corrupt，feature质量差。Mid-early timestep让model perform non-trivial denoising，feature最有informative。

**为什么"not-first"**：完全clean的feature（$\tau = 0$）没经过denoising process，diffusion model特有的denoising-aware representation还没被activate。需要稍微加点noise让model"思考"一下，feature才encode rich信息。

这跟DIFT [https://arxiv.org/abs/2306.03841]的发现一致——diffusion feature在mid timesteps最semantic informative。

### Finding 5：VidFM features > DINO features for 3D

VGGT [https://arxiv.org/abs/2503.17351]是SOTA feedforward 3D reconstructor，原本用DINO features end-to-end train。把DINO features换成frozen WAN2.1-14B features，所有metric都大幅提升。

CO3Dv2 Point从0.476降到0.289。
DL3DV Point从2.751降到1.034（降了62%！）。

更夸张的是supplementary里的data scaling实验：**VidFM-VGGT用不到10%的3D training data，就能超过原版VGGT用100% data训出来的performance。**

**Intuition**：VGGT从DINO features学3D，等于从零learn"如何从2D appearance cue推断3D"。VidFM features已经encode了大量3D prior（因为video model训练时被迫学这个才能generate coherent video），probe只需要learn"如何read out 3D from这些features"。前者需要海量3D supervision，后者只需要少量。

这跟LLM里"pretraining提供world knowledge，finetuning只需少量task data"的pattern一模一样。

## 我觉得最insightful的点

### Point 1：Multi-view consistency是biased proxy

Supplementary Sec. C那个分析很sharp。他们测了cross-view correspondence error（同一3D点跨frame的feature matching error）vs 3D probe error。

发现：**DINOv2的multi-view feature consistency特别好，但3D probe performance很差。** 而video diffusion model的feature consistency比feedforward model差，但3D probe performance强。

**Intuition**：
- DINOv2 features对appearance invariant——同一3D点跨view虽然look不同，但DINOv2把它们project到相近feature space。这让feature matching好做，但features本身没encode多少3D structure info，只是"appearance signature"。
- Video diffusion features包含denoising-specific信息，受noise影响，跨frame的同一3D点feature可能不一样。但features里encode了rich 3D structure，shallow probe能decode出来。

**这给3D awareness evaluation敲了个警钟**：之前Probe3D [https://arxiv.org/abs/2404.00651]用multi-view consistency作为3D awareness的proxy，但这个proxy可能misleading。一个model的features可能很容易跨view match，但根本没understand 3D structure。Direct 3D prediction才是更strict的test。

### Point 2：Video generation objective implicitly learns 3D

最philosophical的point：为什么generation objective能学3D？

我的intuition：video generation要求model predict未来帧。如果model只学2D texture synthesis、local pattern matching，在simple scene里能凑合，但遇到complex camera motion、object occlusion、large scene，纯2D策略会fail。Model必须build一个internal world model——understand 3D structure、camera ego-motion、object dynamics——才能generate coherent future frames。

所以generation objective在scale够大、scene够diverse时，3D understanding是**必然的emergent property**，因为没3D understanding就generate不好。

这跟LLM emergent abilities [https://arxiv.org/abs/2206.07682]的mechanism类似——能力不是为了某个task explicit train的，而是被model capacity + data scale + objective逼出来的。

### Point 3：VidFM作为3D prior的practical path

3D vision的瓶颈一直是data scarcity。COLMAP重建慢、人工标注贵、synthetic data distribution gap大。这篇paper指出一条路：

**Pretrain on video（cheap, abundant）→ extract 3D-aware features → 用少量3D data finetune for specific 3D tasks。**

这比"从少量3D data end-to-end learn 3D"效率高得多。VidFM-VGGT用10% data beat 100% baseline已经证明了。

对real-world deployment意义重大——比如robotics、AR/VR、autonomous driving这些需要3D perception的domain，data acquisition bottleneck可能解除。

## 我会怎么critique这篇paper

### Critique 1：Distribution match的问题

WAN2.1-14B training data我们不知道具体是什么。如果WAN training data里本来就包含类似CO3Dv2、DL3DV的videos（turntable物体拍摄、户外large scenes），那它的3D awareness可能partly来自distribution match，而pure generalization能力可能没数字显示的那么强。

要更convincing，需要在truly out-of-distribution domain上测——microscopy videos、astronomical images、underwater footage等。这些domain的3D structure跟WAN training data应该差异巨大。

### Critique 2：Probe的artifacts

Probe design（4层alternating attention, 1024 channels）虽然比VGGT shallow很多，但仍然是个non-trivial model。Probe可能"learn"一些3D heuristics不from feature而从training data statistics。比如probe可能learn"CO3Dv2的物体一般绕轴旋转，translation pattern大致是X"这种prior，不真正从feature decode 3D。

Paper做了probe size ablation（4 layers 1024 channels vs 4 layers 512 channels），rankings不变。但更strict的test是train probe on one dataset, test on another——如果probe学的是dataset-specific prior，cross-dataset transfer应该collapse。

### Critique 3：3D awareness的维度

Paper只测了point map / depth / camera pose。但3D understanding远不止这些：
- Dynamic scene understanding（deformable objects, non-rigid motion）
- Physical reasoning（gravity, contact, support）
- Material properties（transparency, reflectance）
- Multi-object interactions

一个model可能static 3D geometry强，但dynamic scene understanding差。需要更comprehensive的probe。

### Critique 4：Architectural generalization

所有probed generators都是latent diffusion transformer（DiT-based）。Findings（mid-layer + early-timestep best）可能architecture-specific。Pixel-space diffusion、autoregressive video models、flow-based models上不一定成立。

尤其现在有 trend toward autoregressive "next-frame prediction" video models（类似LLM的next-token prediction），这些model的3D awareness可能不同——可能更好（autoregressive的sequential nature可能force stronger temporal modeling），也可能更差（next-frame prediction比denoising更local）。

## 我对future work的猜测

1. **Mechanistic interpretability**：哪些attention heads、neurons encode 3D info？能否找到"3D circuit" in video models？类似Anthropic在LLM上做的mechanistic analysis [https://transformer-circuits.pub/]。

2. **Distillation**：WAN2.1-14B feature extraction很贵（要跑denoising step）。能否distill一个small student model mimic VidFM的3D-aware features，让deployment cheaper？

3. **Combine VidFM prior with explicit 3D supervision at scale**：paper只在小dataset上test了VidFM-VGGT。如果在massive 3D data（比如所有公开的3D dataset pooled）上train，VidFM-VGGT vs DINO-VGGT的gap会不会反而缩小？或者继续widen？

4. **Video objective design for 3D awareness**：什么objective最能induce 3D awareness？纯generation vs. masked prediction（V-JEPA）vs. contrastive？目前data point是generation > masked prediction，但只是observation，没mechanism explanation。

5. **Beyond 3D: 4D awareness**：能否probe temporal evolution of 3D（即4D）？比如预测future frame的3D structure？这对world model application直接relevant。

## 一句话总结我的intuition

这篇paper告诉我：**Large video models trained on 2D data, despite never seeing 3D supervision, develop strong 3D awareness as an emergent property. 这不是"被骗"的artifact，因为shallow probe能从features里decode出准确3D。3D awareness是generation objective + scale逼出来的side product——你要generate coherent complex videos，必须有internal world model。**

这对整个AI field有broader implication：**general intelligence的pathway可能是scale + right objective，而explicit supervision是secondary。** 这跟LLM emergent abilities的故事一致，也跟LeCun的JEPA vision [https://openreview.net/pdf?id=BZ5a1r-kVsf]有共鸣——world model不需要explicit 3D label，joint-embedding prediction足够induce它。

VidFM as 3D prior这条路很promising。3D data bottleneck可能解除——pretrain on video，用少量3D data calibrate，deploy到any 3D task。对robotics、embodied AI、AR/VR都是好消息。

**Core references for further reading**:
- 论文主页: https://vidfm-3d-probe.github.io/
- VGGT (probe architecture借鉴): https://arxiv.org/abs/2503.17351  
- DIFT (diffusion feature extraction方法): https://arxiv.org/abs/2306.03841
- Probe3D (前作，probe image models): https://arxiv.org/abs/2404.00651
- WAN2.1: https://arxiv.org/abs/2503.20314
- Fast3R (3D expert baseline): https://arxiv.org/abs/2506.03240
- Sora blog (world simulator hypothesis): https://openai.com/research/video-generation-models-as-world-simulators
- Emergent abilities of LLMs (类比): https://arxiv.org/abs/2206.07682
- V-JEPA (LeCun的self-supervised video model): https://arxiv.org/abs/2305.14582

---

# How Much 3D Do Video Foundation Models Encode? 深度解析

## 一、论文的Motivation和核心问题

这篇paper问了一个非常根本的问题：**当视频基础模型（VidFMs）在大规模2D视频数据上训练后，3D understanding是否会作为emergent property自然涌现？**

这个问题的背景是这样的：3D vision领域一直受限于高质量3D data的scarcity。Native 3D assets（如COLMAP重建、人工标注的3D scans）获取成本极高，规模有限，这从根上限制了3D foundation models的scaling。而video是3D world的2D projection，可以大规模获取（YouTube-8M [https://arxiv.org/abs/1609.08675]、Panda-70M [https://arxiv.org/abs/2405.07690]、HowTo100M [https://arxiv.org/abs/1906.03327] 等数据集已经curated）。所以video prior成为scalable 3D learning的一条有前景的pathway。

之前的工作要么finetune video generators加3D control（CameraCtrl [https://arxiv.org/abs/2404.02101]、CamCo [https://arxiv.org/abs/2406.02509]、AC3D [https://arxiv.org/abs/2502.07063]），要么让video model同时输出3D caches（Matrix3D [https://arxiv.org/abs/2503.22616]、Geo4D [https://arxiv.org/abs/2504.07961]）。但这些工作存在confounds：3D inconsistency artifacts、3D fine-tuning requirement、task-specific engineering，让"video data alone能否induce strong 3D awareness in general-purpose setting"这个问题clouded。

作者提出第一个model-agnostic的framework，直接probe VidFMs的3D awareness，沿着四个axes：
1. **Extent**：VidFMs的3D awareness相比image models或specialized 3D models如何？
2. **Factor**：哪些因素影响3D awareness？包括temporal reasoning、3D finetuning、model scaling
3. **Localization**：3D信息集中在哪里？哪个layer？哪个diffusion timestep？
4. **Implication**：在limited 3D data和compute下，VidFM features能否实用地用于3D reconstruction？

## 二、Probe方法的核心思想

### 2.1 Probe的philosophy

核心idea非常clean：**如果一个video model真的understand 3D world，那么应该可以用shallow feedforward readout从它的features中extract出准确的3D properties**，不需要post-optimization或fine-tuning base model。

固定probe capacity和training set，stronger 3D awareness意味着shallow readout能达到lower reconstruction error。这是衡量"3D information是否已经encod在feature space"的直接方法，比用2.5D proxy（如depth、multi-view consistency）更严格。

### 2.2 Feature Extraction的细节

给定video $\mathbf{V} \in \mathbb{R}^{T_v \times 3 \times H_v \times W_v}$，这里：
- $T_v$ = 视频总帧数
- $3$ = RGB通道
- $H_v, W_v$ = 视频的空间分辨率

对每一帧 $t$，从frozen VidFM中提取一个spatial feature map $\mathbf{F}_t \in \mathbb{R}^{C \times H_f \times W_f}$，这里：
- $C$ = feature通道数（例如WAN2.1是几十到上百的通道数）
- $H_f, W_f$ = feature map的空间分辨率（通常是输入分辨率的1/16或1/32）

**对diffusion-based video generators**，提取方法类似DIFT [https://arxiv.org/abs/2306.03841]：
1. 选择一个denoising timestep $\tau$
2. 向VAE latents注入高斯噪声
3. 执行一次denoising step
4. 从指定的network layer读取hidden activations作为features

使用empty text embedding，image-to-video models用第一帧condition。Layer index和 $\tau$ 作为hyperparameters，固定across experiments。

**对V-JEPA、DINOv2、Fast3R**：标准forward pass，取last-layer spatial features（经验上发现last layer最好）。

**Long video处理**：不同VidFMs operating on不同clip lengths。处理长video时，将input video $\mathbf{V}$ split成short chunks，从beginning以fixed stride subsampling。每个chunk前面prepend第一帧，确保所有chunks共享同一个first-frame reference。维护一个frame-to-feature index map $\pi(t)$，记录每个raw frame $t$对应的chunk和local index。Probe time根据input frame indices $\{t_i\}_{i=1}^S$ 和 $\pi(t)$，gather对应的features $\{\mathbf{F}_{t_i}\}_{i=1}^S$。

### 2.3 Probe Architecture详解

Probe model是个shallow VGGT-like transformer。VGGT [https://arxiv.org/abs/2503.17351]（Visual Geometry Grounded Transformer）是最近一个state-of-the-art的feedforward 3D reconstructor。

**Input**：每个input video取 $S=4$ 帧——第一video frame作为reference，另外3帧以minimum temporal gap of 5 frames sampling。

**为什么minimum gap = 5**：太小gap的frames motion不够，pose estimation task可能degenerate到identity；太大gap可能导致large baseline让correspondence难。5是一个empirical sweet spot。

从对应的feature maps $\{\mathbf{F}_{t_i}\}_{i=1}^4$，获取per-frame tokens，apply **4个alternating-attention blocks**。

**Alternating-attention block**包含：
- **Frame attention**：mixes tokens within each frame（intra-frame信息聚合）
- **Global attention**：mixes tokens across frames（inter-frame信息聚合）

这种设计mirror了VGGT但much shallower（VGGT通常有几十层）。

**三个readout heads**：
1. 两个 **DPT heads**（Dense Prediction Transformer，从Vision Transformers for Dense Prediction [https://arxiv.org/abs/2102.02744]借鉴）：
   - 产出 dense point maps $\hat{\mathbf{X}}_{t_i} \in \mathbb{R}^{H_v \times W_v \times 3}$（在第一frame的coordinate system下，每个pixel的3D坐标）
   - 产出 dense depth maps $\hat{\mathbf{D}}_{t_i} \in \mathbb{R}^{H_v \times W_v}$
2. 一个 **camera head**：predicts每帧相对第一帧的pose

### 2.4 Loss Function详解

Multi-task objective：

$$\mathcal{L} = \lambda_{\mathrm{pmap}} \mathcal{L}_{\mathrm{pmap}} + \lambda_{\mathrm{depth}} \mathcal{L}_{\mathrm{depth}} + \lambda_{\mathrm{cam}} \mathcal{L}_{\mathrm{cam}}$$

其中 $\lambda_{\mathrm{pmap}} = \lambda_{\mathrm{depth}} = \lambda_{\mathrm{cam}} = 1$（默认）。

**$\mathcal{L}_{\mathrm{pmap}}$ 和 $\mathcal{L}_{\mathrm{depth}}$**：confidence-weighted $\ell_2$ loss，between predicted point/depth maps和groundtruth。

Confidence weighting的形式类似VGGT：

$$\mathcal{L}_{\mathrm{pmap}} = \frac{1}{N} \sum_i c_i \cdot \|\hat{\mathbf{X}}_i - \mathbf{X}_i\|_2^2 - \log c_i$$

其中 $c_i$ 是预测的confidence，第一项惩罚high confidence但high error的pixels，第二项 $\log c_i$ 防止 $c_i$ 退化到无穷大。模型自动学会在ambiguous区域（如object boundaries、textureless regions）output低confidence。

**Scene normalization**：groundtruth scenes在loss计算前normalized，remove scale ambiguity。因为单目video无法确定绝对尺度，必须去掉这个ambiguity才能让loss有意义。

**$\mathcal{L}_{\mathrm{cam}}$**：Huber loss [https://en.wikipedia.org/wiki/Huber_loss]between predicted poses和groundtruth poses。Huber loss形式：

$$\mathcal{L}_{\mathrm{cam}}(x) = \begin{cases} \frac{1}{2}x^2 & \text{if } |x| \leq \delta \\ \delta(|x| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

Huber loss相比 $\ell_2$ 更robust to outliers（pose outliers常见），相比 $\ell_1$ 更smooth at zero。$\delta$ 通常取1。

## 三、实验设置详解

### 3.1 Datasets

**CO3Dv2** [https://arxiv.org/abs/2009.00559]：object-centric dataset，turntable-type videos（相机绕物体旋转）。Curate：filter掉heavy truncation或portrait-oriented videos（无法形成border-less horizontal object-centric crops）。Filtered split有11k videos。从每个video sample consecutive frames作为feature extraction pipeline input，training用first 76 frames。Video level 9:1 train-test split。Ablation subset包含10个diverse categories，2.7k videos。

**DL3DV** [https://arxiv.org/abs/2406.12865]：large, cluttered scenes，比CO3D更challenging。Use first 6k splits，9:1 train-test split by video。

**Groundtruth生成**：两个datasets都run VGGT [https://arxiv.org/abs/2503.17351]生成每帧的groundtruth：dense point maps, depth, camera poses。Point/depth maps也保存confidence maps用于loss。Probe time只sample 4帧，但groundtruth用all frames生成——这比原dataset提供的groundtruth更准确。

### 3.2 Metrics

**Point map error**：先normalize每个scene去除global scale，然后用 **Umeyama algorithm** [https://en.wikipedia.org/wiki/Umeyama_algorithm] align predicted和groundtruth point clouds，报告mean $\ell_2$ error。

Umeyama algorithm求解两个point clouds之间的最优rigid transformation（rotation $\mathbf{R}$，translation $\mathbf{t}$，scale $s$）：

$$\min_{s, \mathbf{R}, \mathbf{t}} \sum_i \| s \mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \|^2$$

通过SVD closed-form求解。Align之后point map error才反映geometry本身的准确性，而非坐标系差异。

**Depth error**：同样scene normalization后，报告mean $\ell_2$ error。

**Camera pose error**：计算所有frame pairs的relative pose error：
- Rotation error $e_R$ = SO(3)上的geodesic angle，即 $\arccos\left(\frac{\mathrm{tr}(\mathbf{R}_1^T \mathbf{R}_2) - 1}{2}\right)$
- Translation error $e_T$ = translation directions之间的angle

**Joint accuracy at threshold $\theta$**：

$$\Pr[\max(e_R, e_T) \leq \theta]$$

即rotation和translation都必须 $\leq \theta$。

**AUC@$\Theta$**：$\theta$从 $0°$ sweep到 $\Theta°$（如 $\Theta \in \{5, 30\}$），joint accuracy curve下的area。AUC同时反映accuracy和robustness（在多个thresholds下的表现）。

### 3.3 Control Groups设计

这是论文非常elegant的设计：用两个controls上下contextualize VidFM results。

**Per-frame Image control (lower reference)**：probe DINOv2 features from each frame of video。Features是isolated提取的，所以任何global 3D understanding must be induced by probe rather than supplied by backbone。为了让task well-posed，append一个reference-frame indicator token标记第一帧。所有losses、schedules、hyperparameters mirror VidFM setting。

这个control回答的问题是：probe本身有多强？如果probe本身能从isolated image features中reconstruct 3D，那VidFM results的practical意义就打折扣。

**Native 3D control (upper reference)**：probe Fast3R [https://arxiv.org/abs/2506.03240] features。Fast3R是state-of-the-art multi-view 3D point map predictor。因为Fast3R直接optimized for同样的target，probing它在同probe architecture和supervision下提供strong reference。

有意思的细节：CO3D是Fast3R training sets的一部分，但DL3DV不是。这允许study Fast3R的generalization behavior——在CO3Dv2上是in-distribution performance，在DL3DV上是out-of-distribution performance。

## 四、主要发现详解

### 4.1 Extent：VidFMs的3D awareness有多强？

**Table 1核心结果**（CO3Dv2部分）：

| Probed Feature | Point Err(↓) | Depth Err(↓) | AUC@5(↑) | AUC@30(↑) |
|---|---|---|---|---|
| DINOv2 | 0.559 | 0.209 | 0.051 | 0.508 |
| V-JEPA | 0.439 | 0.214 | 0.076 | 0.619 |
| CogVideoX | 0.485 | 0.231 | 0.051 | 0.569 |
| Aether | 0.501 | 0.249 | 0.054 | 0.571 |
| Open-Sora2.0 | 0.391 | 0.196 | 0.096 | 0.643 |
| WAN2.1-14B | 0.284 | 0.151 | 0.200 | 0.736 |
| Fast3R | 0.262 | 0.145 | 0.272 | 0.769 |

在CO3Dv2上，WAN2.1-14B [https://arxiv.org/abs/2503.20314] 在所有metric上second only to Fast3R：
- Point: 0.284 vs 0.262（差距很小）
- Depth: 0.151 vs 0.145（几乎平手）
- AUC@30: 0.736 vs 0.769（差距也很小）

更surprising的是DL3DV结果：

| Probed Feature | Point Err(↓) | Depth Err(↓) | AUC@5(↑) | AUC@30(↑) |
|---|---|---|---|---|
| DINOv2 | 2.814 | 0.534 | 0.013 | 0.245 |
| V-JEPA | 1.576 | 0.613 | 0.076 | 0.558 |
| CogVideoX | 1.748 | 0.608 | 0.061 | 0.486 |
| Aether | 1.566 | 0.574 | 0.067 | 0.527 |
| Open-Sora2.0 | 1.306 | 0.445 | 0.115 | 0.607 |
| WAN2.1-14B | **1.051** | **0.323** | **0.136** | **0.660** |
| Fast3R | 1.379 | 0.514 | 0.134 | 0.637 |

WAN2.1-14B在DL3DV上**surpasses Fast3R on all metrics**！Point 1.051 vs 1.379，Depth 0.323 vs 0.514，AUC@30 0.660 vs 0.637。

**Intuition**：Fast3R在CO3Dv2上in-distribution表现最好（这是它训练数据的一部分），在DL3DV上out-of-distribution时generalization不够强。WAN2.1-14B从未见过任何3D supervision，但只通过大规模video生成训练，在challenging out-of-distribution scenes上比3D expert还好。

这非常震撼——意味着video generation objective本身在scale够大时，能implicitly learn到非常general的3D understanding，可能比explicit 3D supervision的domain-specific generalization更强。

### 4.2 Factor #1：Temporal Reasoning的关键性

Per-frame DINOv2在CO3Dv2上的depth error是0.209，competitive with video models。但global 3D understanding上显著worse：Point 0.559，AUC@30 0.508，比所有video models差，包括self-supervised V-JEPA（Point 0.439，AUC@30 0.619）。

**Intuition**：image和video models的核心区别在于video models allow information exchange along time axis。Per-frame DINOv2提取的features对single-frame depth有暗示（来自2D appearance cues），但global 3D properties（如跨frame的point map在common coordinate frame下，camera pose相对reference frame）本质上需要跨frame reasoning。Probe虽然能mix tokens across frames，但backbone本身没supply跨frame information，所以probe要从零开始induce这个mapping。

DL3DV上这个gap更大：DINOv2 Point 2.814，AUC@30 0.245 vs V-JEPA Point 1.576，AUC@30 0.558。DINOv2的depth estimation在DL3DV上仍然competitive——这再次证实image model能做单帧depth，但完全做不到global 3D understanding。

Figure 1的radar plots也mirror这个pattern：methods with explicit temporal reasoning的polygons expand along Point和Pose axes，not just Depth。

### 4.3 Factor #2：3D Fine-tuning的双刃剑

Aether [https://arxiv.org/abs/2503.18945] 在CogVideoX基础上用3D-aware objectives和conditions finetune。

**DL3DV上Aether确实improves over CogVideoX**：
- Point: 1.566 vs 1.748
- Depth: 0.574 vs 0.608
- AUC@30: 0.527 vs 0.486

**但在object-centric CO3Dv2上slightly worse than base model**：
- Point: 0.501 vs 0.485
- Depth: 0.249 vs 0.231

**Intuition**：Aether的training data大部分是synthetic large scenes from games/simulators，与CO3Dv2的object-centric turntable videos distribution不match。3D-aware fine-tuning让model在in-domain data上更specialized，但可能overfit到training distribution的statistics，hurt out-of-domain generalization。

这是个非常important的observation，对想build scalable 3D world models的researcher有practical implication：3D fine-tuning不free lunch，可能牺牲base model的broad generalization能力。

### 4.4 Factor #3：Model Scaling的mixed impact

**WAN scaling**：1.3B → 14B，point-map error从0.0468降到0.0360（相对-23%），显著improve。

**CogVideoX scaling**：2B → 5B，point-map error从0.0576到0.0590（相对+2%），slightly worsen。

**Intuition**：Parameter count alone不guarantee stronger 3D awareness。WAN的scaling伴随着training data scale的increase，CogVideoX-5B和2B可能在data上没有相应scaling。Paper假设additional training data plays important role。

这其实和LLM scaling laws类似——scaling不just parameter count，data和compute同样critical。3D awareness的scaling behavior可能与generative quality的scaling behavior different，不是monotonic关系。

### 4.5 Localization：哪个Layer，哪个Timestep？

通过sweeping 3个network layers和4个denoising timesteps，所有diffusion models的optimum惊人consistent：**mid-network layers + early-but-not-first timesteps**。

**Layer choice intuition**：
- **Late layers** specialized to per-frame RGB synthesis task，suppress high-level 3D-related features（因为RGB synthesis需要pixel-level detail而非global 3D structure）
- **Too early layers** high-level features还没form（feature abstraction是渐进的过程）
- **Mid layers** best balance——已经abstract到high-level representation，还没specialize到pixel synthesis

**Timestep choice intuition**：
- Diffusion model timestep $\tau$对应noise level。Earlier timesteps = less noise added
- Either too little或too much noise让denoising task degenerate（too easy或too hard），features less useful
- Early steps比late stepswork better，因为input signal less corrupted by noise
- "Not-first"意味着pure clean features不行（因为denoising process没initiate，没extract denoising-specific features）
- 整体：mid-layer + moderately early features strike balance，retain global 3D cues while less influenced by large noise

这个发现对想用diffusion model features的practitioner非常valuable——给了一个general的heuristic，不需要每个model都做expensive sweep。

### 4.6 Implication：VidFM Features用于Feedforward 3D

VGGT [https://arxiv.org/abs/2503.17351]是state-of-the-art feedforward 3D reconstructor，依赖DINO features。Question：用VidFM features替换DINO features效果如何？

**Table 2结果**：

| Method | CO3Dv2 Point | CO3Dv2 Depth | CO3Dv2 AUC@5 | CO3Dv2 AUC@30 | DL3DV Point | DL3DV Depth | DL3DV AUC@5 | DL3DV AUC@30 |
|---|---|---|---|---|---|---|---|---|
| Original VGGT (DINO) | 0.476 | 0.205 | 0.076 | 0.565 | 2.751 | 0.518 | 0.058 | 0.363 |
| VidFM-VGGT (WAN2.1-14B) | **0.289** | **0.145** | **0.178** | **0.718** | **1.034** | **0.319** | **0.183** | **0.686** |

VidFM-VGGT在所有metrics上大幅outperform原VGGT。CO3Dv2 Point从0.476降到0.289，DL3DV Point从2.751降到1.034（相对-62%）。

**Supplementary Sec. B的data scaling实验**更impressive：Figure 6显示，VidFM-VGGT通常用less than 10% training data就能surpass原VGGT用100% training data的performance。

**Intuition**：当3D supervision limited到small datasets（如CO3Dv2和DL3DV），从DINO features end-to-end learn 3D understanding需要大量3D data。VidFM features已经encode了大量3D prior，3D model只需要learn如何decode这些features到3D outputs，所需3D supervision大大减少。

## 五、Supplementary的关键发现

### 5.1 Probe Size Ablation

Table 3在DL3DV上对比original probe（4 layers, 1024 channels）和smaller probe（4 layers, 512 channels）：

| Probed Feature | Original Point | Original Depth | Original AUC@30 | Smaller Point | Smaller Depth | Smaller AUC@30 |
|---|---|---|---|---|---|---|
| DINOv2 | 2.814 | 0.534 | 0.245 | 3.344 | 0.623 | 0.163 |
| V-JEPA | 1.576 | 0.613 | 0.558 | 1.707 | 0.657 | 0.505 |
| WAN2.1-14B | 1.051 | 0.323 | 0.660 | 1.317 | 0.374 | 0.567 |
| Fast3R | 1.379 | 0.514 | 0.637 | 1.551 | 0.572 | 0.549 |

Smaller probe让所有方法performance都degrade，但**relative rankings和conclusions unchanged**。WAN2.1-14B仍然best。

这说明findings robust——3D awareness是feature property，不artifact of probe capacity。

### 5.2 Multi-view Consistency vs 3D Probe Performance

Supplementary Sec. C做了非常insightful的分析。Cross-view correspondence error定义为：sample anchor view A和pixels，用groundtruth 3D reproject到view B，记录locations if not occluded。Predicted correspondence用nearest neighbor query in feature space——对A中每个anchor point，retrieve top-1 nearest neighbor in B based on VidFM features。Average Euclidean pixel distance作为cross-view correspondence error。

**Figure 7发现**：3D probe error（x轴，lower better）vs cross-view correspondence error（y轴，lower better）的scatter plot：

**Within video diffusion models**：positive correlation，lower probe error伴随lower correspondence error。CogVideoX最差，Open-Sora2.0和WAN2.1-1.3B intermediate，WAN2.1-14B best（bottom-left）。

**Feedforward models（Fast3R, V-JEPA, DINOv2）在comparable probe error下显示better multi-view consistency**——它们位于diffusion models下方。DINOv2尤其strong multi-view consistency，但3D probe performance很差。

**Intuition解析**：

**Diffusion models为什么worse multi-view consistency at same 3D awareness**？Diffusion features通过注入noise到VAE features + 单次denoising step提取。这让features在large noise locations noisy，representation也包含specifically tailored to denoising的features（受random noise影响）。同一3D point在不同frames的pixels可能carry不同features，导致feature discrepancies。Probe能decode 3D但raw feature consistency差。

**DINOv2为什么strong multi-view consistency but poor 3D awareness**？Image model features可能更"invariant"——同一3D point跨views appearance变化时，DINOv2学到的features相对consistent。但DINOv2 features缺乏temporal information，无法提供global 3D structure cues。

**Video models为什么比image model less "consistent"**？Video model一些channels correlate with local motions at current frame；同一3D point跨frames可能exhibit不同local motions。这richer temporal information aids 3D decoding，但让features在nearest-neighbor matching下appear less "consistent"。

**这是paper一个重要warning**：cross-view feature similarity alone is a **biased** evaluation for 3D awareness，especially comparing across model families。Prior work（如Probe3D [https://arxiv.org/abs/2404.00651]）用multi-view consistency作为3D awareness proxy可能misleading。

## 六、Intuition Building和深度联想

### 6.1 "Video Generation = World Simulation" Hypothesis

Sora release时OpenAI提出"video generation models as world simulators" [https://openai.com/research/video-generation-models-as-world-simulators]。这篇paper为这个hypothesis提供了quantitative evidence：state-of-the-art video generators如WAN2.1-14B确实encode了strong, generalizable 3D understanding，即使没见过任何3D supervision。

这暗示video generation objective在scale够大时，可能implicitly learn到physics-aware world models。3D structure和ego-motion作为video的implicit factor，被model pick up用于更好的generation质量。

### 6.2 与LLM Emergent Abilities的类比

Paper [10]在References提到"Emergent abilities of large language models" [https://arxiv.org/abs/2206.07682]。LLM在scale够大时涌现出chain-of-thought reasoning、in-context learning等能力。VidFM的3D awareness可能类似的emergent property——单frame或small-scale video training可能不sufficient，但scale到数十亿parameters + 百万小时video后自然emerge。

### 6.3 与Self-Supervised Learning的联系

V-JEPA [https://arxiv.org/abs/2305.14582]是Yann LeCun的self-supervised video representation learning工作，基于latent video prediction。V-JEPA在Table 1中3D awareness moderate，比DINOv2好但比top video generators差。

**Intuition**：V-JEPA的objective是predict masked regions in latent space，可能更关注local prediction能力而非global generative modeling。Video generation models的denoising objective可能force model learn更完整的world model，包括3D structure。

### 6.4 与DUSt3R/MASt3R/Fast3R的evolution

DUSt3R [https://arxiv.org/abs/2404.06284]是第一个strong feedforward 3D reconstructor for image pairs。MASt3R [https://arxiv.org/abs/2406.09756]改进它加matching。Fast3R [https://arxiv.org/abs/2506.03240] scaling到1000+ images in one forward pass。这些都rely on explicit 3D supervision。

VGGT用DINO features end-to-end learn，scale到strong performance but需大量3D data。这篇paper的VidFM-VGGT提示：future方向是combine large video priors with limited 3D supervision——best of both worlds。

### 6.5 与"Latent World Models"的联系

Recent work如Genie [https://arxiv.org/abs/2402.15391]、Genie 2 [https://deepmind.google/discover/blog/genie-2-a-large-foundation-world-model/]探索latent world models。如果video generators already encode 3D understanding，那它们已经某种程度是world models。Next step是extract这个world model用于planning、control等downstream tasks，不仅仅生成视频。

### 6.6 与Multiview Consistency Probing的反思

Probe3D [https://arxiv.org/abs/2404.00651]和Feat2GS [https://arxiv.org/abs/2405.19224] probe image models用multi-view consistency或Gaussian Splatting。这篇paper show这些indirect probes可能biased——DINOv2有strong multi-view feature consistency但poor 3D understanding。

**Implication**：Future 3D awareness probes应该用direct 3D prediction tasks（point maps, camera poses），not just feature consistency。Feature consistency是means to an end，不是end itself。

### 6.7 Open Questions和Future Directions

Paper Sec. 4.5讨论limitations：
1. 公开checkpoints而非controlled conditions——无法strictly attribute 3D awareness差异到specific factors
2. 无open-source models提供多个versions只在training data scale上differ——无法isolate data scale effect
3. 资源限制无法train large-scale 3D reconstruction models from scratch on massive datasets with VidFM features

**Future directions联想**：
1. **Scaling laws for 3D awareness**：系统study parameter count、data scale、compute各自如何影响3D awareness
2. **VidFM features for更多3D tasks**：除了point/depth/camera，还有novel view synthesis、3D segmentation、dynamic scene reconstruction
3. **Diffusion model internals probing**：哪些attention heads或neurons对应3D properties？mechanistic interpretability方向
4. **Pretraining objective design**：什么objective最能induce 3D awareness？纯generation vs. prediction vs. contrastive？
5. **Combine VidFM priors with explicit 3D supervision at scale**：能否achieve比两者都stronger的model？

### 6.8 与Physics-aware Generation的联系

Paper [42]在References提到"Shadows don't lie and lines can't bend" [https://arxiv.org/abs/2405.17953]，show generative models don't know projective geometry well. 这篇paper的findings看似contradict，但实则complement：早期generation models（如Stable Diffusion）确实geometry poor，但scale到WAN2.1-14B这种level后，3D awareness emerges。

可能机制：small models用2D shortcuts（texture synthesis, local patterns）就能generate plausible frames；large models必须learn真正的3D structure才能maintain coherence across frames in complex scenes。这是emergent behavior。

### 6.9 与Embodied AI的connection

Paper introduction提到embodied AI applications。如果VidFMs已经encode 3D understanding，可以用于：
- Robot manipulation：从video demonstrations extract 3D scene understanding
- Autonomous navigation：从driving videos learn 3D scene layouts
- AR/VR：从sparse video reconstruct 3D environments

VidFM-VGGT的data efficiency（10% data surpass 100% DINO baseline）对real-world deployment critical——3D data acquisition bottleneck解除。

### 6.10 Compute Cost和Practical Considerations

虽然paper没详细discuss compute，但值得思考：

**Feature extraction cost**：对diffusion model，每次feature extraction需要单次denoising step + VAE encoding，相比DINO的single forward pass显著更expensive。For大规模deployment，这是bottleneck。

**Probe training cost**：Shallow probe训练相对cheap，但data preprocessing（VGGT生成groundtruth on all frames）expensive。

**Trade-off**：VidFM features的quality vs. extraction cost。可能distillation方向——train smaller student model to mimic VidFM features。

## 七、Limitations和Critique

虽然paper findings impressive，但需要保持critical视角：

### 7.1 Probe作为Evaluation的Validity

Probe的performance同时取决于backbone feature quality和probe本身capacity。如果probe太强，可能"hallucinate"3D information不是真正encoded in features。Paper用smaller probe ablation缓解这个concern，但probe design space巨大（depth、width、attention pattern等），仍可能有unexplored confounds。

### 7.2 Distribution Match between VidFM Training Data和Probe Datasets

WAN2.1-14B在CO3Dv2和DL3DV上strong performance，可能partly因为WAN training data包含similar videos。虽然paper argument是WAN从未见过3D supervision，但video distribution match可能still help。需要test on truly out-of-distribution videos（如medical, satellite, niche categories）。

### 7.3 3D Awareness的Spectrum

Paper operationalize 3D awareness为point/depth/camera prediction。但3D understanding有broader spectrum：
- Dynamic scenes（moving objects, deformable surfaces）
- Physical reasoning（gravity, friction, contact）
- Material properties（reflectance, transparency）
- 4D understanding（temporal evolution of 3D）

Future work应该extend probing到这些aspects。

### 7.4 Generalization到Non-Latent-Diffusion Models

所有tested video generators都是latent diffusion models（VAE + denoiser）。Findings可能not generalize到：
- Pixel-space diffusion models
- Autoregressive video models（如VideoPoet [https://arxiv.org/abs/2312.14125]）
- Flow-based or VAE-based generative models

Mid-layer + early-timestep的finding可能architecture-specific。

## 八、Conclusion

这篇paper做出了重要贡献：

1. **First systematic, model-agnostic evaluation** of 3D awareness in VidFMs
2. **Methodology innovation**：shallow feedforward probes for direct 3D prediction，比indirect proxies更strict
3. **Surprising findings**：state-of-the-art video generators exhibit 3D awareness接近或超过3D experts，never trained on 3D data
4. **Practical implications**：VidFM features for feedforward 3D reconstruction under limited data
5. **Open questions and future directions** for scalable 3D world models

整体上，这篇paper是"understanding foundation models"研究方向的重要contribution。它quantitatively证实了"video generation models as world simulators"hypothesis，并为build next-generation 3D models提供insights。

对Karpathy这种对foundation models和emergent properties感兴趣的researcher，这篇paper的findings应该resonate——emergent 3D awareness是video model scaling的side effect，类似LLM中emergent reasoning能力。这暗示general intelligence的pathway可能是scale + right objective，而explicit supervision是secondary。

**References**:
- Paper主页：https://vidfm-3d-probe.github.io/
- VGGT: https://arxiv.org/abs/2503.17351
- WAN2.1: https://arxiv.org/abs/2503.20314
- Open-Sora2.0: https://arxiv.org/abs/2503.09642
- CogVideoX: https://arxiv.org/abs/2408.06072
- Aether: https://arxiv.org/abs/2503.18945
- V-JEPA: https://arxiv.org/abs/2305.14582
- DINOv2: https://arxiv.org/abs/2304.07193
- Fast3R: https://arxiv.org/abs/2506.03240
- DUSt3R: https://arxiv.org/abs/2404.06284
- MASt3R: https://arxiv.org/abs/2406.09756
- Probe3D: https://arxiv.org/abs/2404.00651
- Feat2GS: https://arxiv.org/abs/2405.19224
- DIFT: https://arxiv.org/abs/2306.03841
- Sora blog: https://openai.com/research/video-generation-models-as-world-simulators
- Umeyama algorithm: https://en.wikipedia.org/wiki/Umeyama_algorithm
- CO3Dv2: https://arxiv.org/abs/2009.00559
- DL3DV: https://arxiv.org/abs/2406.12865
- DPT (Dense Prediction Transformer): https://arxiv.org/abs/2102.02744
- Huber loss: https://en.wikipedia.org/wiki/Huber_loss
- Emergent abilities of LLMs: https://arxiv.org/abs/2206.07682
- Panda-70M: https://arxiv.org/abs/2405.07690
- YouTube-8M: https://arxiv.org/abs/1609.08675
- HowTo100M: https://arxiv.org/abs/1906.03327
- CameraCtrl: https://arxiv.org/abs/2404.02101
- CamCo: https://arxiv.org/abs/2406.02509
- AC3D: https://arxiv.org/abs/2502.07063
- Matrix3D: https://arxiv.org/abs/2503.22616
- Geo4D: https://arxiv.org/abs/2504.07961
- Shadows don't lie paper: https://arxiv.org/abs/2405.17953
- VideoPoet: https://arxiv.org/abs/2312.14125
- Genie: https://arxiv.org/abs/2402.15391
- Genie 2 blog: https://deepmind.google/discover/blog/genie-2-a-large-foundation-world-model/
