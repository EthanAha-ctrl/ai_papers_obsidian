---
source_pdf: RealisID.pdf
paper_sha256: 206c86de217462fa2140af756d5f6199eec28dd979bee167a0cc6717c16a85f5
processed_at: '2026-08-11T21:27:03-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RealisID用人话讲

Andrej你好！我换个方式讲，保持技术深度但用更直觉的方式。把这篇paper当成一个engineering story来讲。

## 这篇paper想解决什么真问题

ID customization这个领域已经被做了很多轮了——InstantID、PuLID、PhotoMaker、IP-Adapter都是比较有名的工作。但如果你真的拿这些方法去productize，会发现它们在一个特别common的场景下崩掉：**当生成的face在画面里比较小的时候**。

这个场景有多common？你想想手机拍照、全身照、合影、电影截图——face占画面比例小于1/6的case到处都是。但现有方法几乎都只在face占画面1/4到1/2这种"大头照"场景下才work well。

RealisID的作者团队（Alibaba DAMO + 武汉理工 + 上海AI Lab）就盯着这个gap做。他们发现small face下identity fidelity退化不是random failure，而是**structural problem**——在latent space里face信息被背景信息淹没了。

除了small face，他们还顺手解决了两个相邻问题：
- Fine control（location, pose, expression的精确控制）
- Multi-person customization（不用multi-person训练数据）

## 小face为什么会崩——intuition

这是整篇paper最核心的insight，值得仔细讲。

Stable Diffusion在latent space工作，假设latent shape是 $z_t \in \mathbb{R}^{h \times w \times c}$，对SDXL来说 $h=w=128$（1024×1024 image经VAE下采样8倍）。

现在考虑两种case：
- **Big face**: face bbox占image 1/2，对应latent里大约64×64的region
- **Small face**: face bbox占image 1/7，对应latent里大约18×18的region

ControlNet的工作方式是把condition（比如facial landmarks）inject到U-Net decoder的intermediate feature里。这个injection要在face spatial position跟condition对齐。问题是：当face只占18×18，整个latent是128×128，**face signal只占latent的2%面积，剩下98%是background noise**。

ControlNet的trainable copy blocks要从这个被background-dominated的latent里提取face-relevant feature去跟landmarks对齐，这个alignment task难度极高。模型学到的更多是background的spatial pattern，而不是face detail的精细control。

更糟的是，不同reference image的face大小不同，latent里face占比也不同。模型要同时学"face占64×64时怎么对齐"和"face占18×18时怎么对齐"，这是个scale-dependent learning task，泛化困难。

## RealisID的解法——Local Branch的核心trick

作者借用了small object detection领域的一个老idea：**feature super-resolution**。在object detection里，小物体之所以难检测，是因为它在feature map上占的pixel太少。SOD-MTGAN (Bai et al. 2018) 和后续工作的思路是crop小物体区域，up-sample到固定size，再单独处理。

RealisID把这个idea搬到diffusion latent space：

```
z_t (128×128×4, whole image latent)
        │
        ▼ crop face region based on bbox
z_t_face (18×18×4 if face is small)
        │
        ▼ bilinear up-sample
ẑ_t (64×64×4, fixed size)
        │
        ▼ feed to local ControlNet
local features (64×64×C)
        │
        ▼ relocation: down-sample + place at face position
î_t_l (128×128×C, sparse, only face region non-zero)
        │
        ▼ inject to U-Net
```

这个trick做了三件事：

1. **把scale variation从learning problem变成preprocessing**：所有reference image的face都被强制resize到64×64，模型不需要学scale invariance。这就好比BatchNorm把activation distribution normalize掉，让网络不必学distribution shift。

2. **过滤background interference**：crop之后ControlNet只看到face region的latent，background根本进不来。latent跟facial landmarks的对齐变得trivial——两者spatial size匹配。

3. **通过relocation保留location信息**：crop会丢失face在画面里的位置，所以处理完要down-sample回relative size，再根据location guidance放到正确位置。这一步用binary mask $c_{loc}$ 指导。

公式Eq. 5写的是：

$$i_{t\_l} = \mathcal{Z}(\mathcal{F}(\hat{z}_t + \mathcal{Z}(c_{p\&e}; \Theta_{z1\_l}), p_{id}, t; \Theta_{c\_l}); \Theta_{z2\_l})$$

翻译成人话：
- $\hat{z}_t$: crop+up-sample后的face-only latent
- $c_{p\&e}$: facial landmarks condition
- $\mathcal{Z}(c_{p\&e}; \Theta_{z1\_l})$: landmarks经zero conv后的feature
- $\hat{z}_t + \mathcal{Z}(c_{p\&e}; \Theta_{z1\_l})$: latent和condition相加（ControlNet的标准做法）
- $\mathcal{F}(\cdot; \Theta_{c\_l})$: ControlNet的trainable copy blocks处理
- 外层$\mathcal{Z}$: 再过一个zero conv准备inject
- $p_{id}$: ID embedding作为prompt signal（不是text，是face identity）

注意外层没有直接relocation，实际是先得到$i_{t\_l}$再通过$\mathcal{R}(\cdot, c_{loc})$ relocate到正确位置。论文Eq. 4和Eq. 5写法有点绕，但逻辑就是这样。

## Global Branch的角色

只有local branch够吗？不够。Local branch只管face region，但整张图还需要harmony——background、body、lighting、style要跟face一致。如果只inject local feature，生成的face会很清晰但跟周围格格不入，像个贴上去的sticker。

Global branch的公式Eq. 6：

$$i_{t\_g} = \mathcal{Z}(\mathcal{F}(z_t + \mathcal{Z}(c_{loc}; \Theta_{z1\_g}), p_{id}, t; \Theta_{c\_g}); \Theta_{z2\_g})$$

跟local branch的区别：
- 输入用整个$z_t$而不是cropped $\hat{z}_t$
- condition用$c_{loc}$（binary mask）而不是$c_{p\&e}$（landmarks）
- 不做relocation

为什么condition用binary mask？因为$c_{loc}$隐式编码了整图layout——face bbox位置、body相对位置、background区域分布。ControlNet从这个mask学到的是global layout，配合ID embedding $p_{id}$，就能control整图的harmony和face location。

## 两个branch怎么协作

这是设计的精妙处。两个branch的injection都加到U-Net decoder的intermediate feature上，U-Net自己learn如何从两路信号里取信息：

- Local branch提供face details、identity、expression、pose的fine signal
- Global branch提供layout、harmony、face location的coarse signal

这其实是multi-scale processing的经典思路——fast path处理detail，slow path处理context。类似的思想在super-resolution、image inpainting里都常见。

Ablation study（Table 2）特别有意思：

| Config | ASP | FaceNet | CLIP-I |
|---|---|---|---|
| w/o $B_{local}$ | 6.07 | 0.681 | 0.673 |
| w/o $B_{global}$ | 5.97 | 0.734 | 0.689 |
| Full | 6.11 | 0.767 | 0.701 |

去掉local branch，FaceNet从0.767掉到0.681（-11%），证明local是identity主力。
去掉global branch，identity metrics反而略升（0.734 vs 0.767 face region专注了），但ASP（aesthetic score）掉到最低5.97——生成的face清晰但整图不协调。

这说明identity和harmony有trade-off，两个branch协作才达到Pareto optimal。

## Multi-person怎么扩展——这个设计很巧妙

训练只用single-person data（CosmicMan数据集），但推理要支持multi-person。怎么做到？

关键是两个branch的injection性质不同：

**Local branch**：每个reference image经过local branch得到的injection feature只在face region非零（因为input是cropped face latent + relocate到对应位置）。所以两个人的local injection相加，它们在不同spatial position非零，互不干扰：

$$i_{t\_l}^{multi} = \sum_{k=1}^{K} \mathcal{R}(i_{t\_l}^{(k)}, c_{loc}^{(k)})$$

每个人的injection在自己face position贡献signal，互不interference。

**Global branch**：每个人的$c_{loc}$是各自face的binary mask，分别经过ControlNet得到global injection。如果直接相加会冲突，所以取平均：

$$i_{t\_g}^{multi} = \frac{1}{K}\sum_{k=1}^{K} i_{t\_g}^{(k)}$$

平均后global layout是K个人的layout的"union"，跟multi-person scene的合理layout对应。

这个设计让single-person训练直接迁移到multi-person推理，是工程上的free lunch。Table 9的实验验证：multi-person下FaceNet几乎不退化（0.791 vs 0.788 at face size 1/4）。

## 训练Loss的设计

Eq. 7：

$$\mathcal{L} = \mathbb{E}[||\epsilon_\theta(z_t, p_{text}, t) - \epsilon||_2 + \lambda ||(\epsilon_\theta(z_t, p_{text}, t) - \epsilon) \odot c_{loc}||_2]$$

第一项是标准SD noise prediction loss，覆盖整图。
第二项只在face region（$c_{loc}=1$的区域）额外加noise prediction loss，weight是$\lambda=1.0$。

这个设计鼓励U-Net在face region预测noise更准。由于两个branch的injection都影响U-Net的noise prediction，这个加权loss间接强化两个branch在face region的学习信号。

只训练两个branch的参数（$\Theta_{z1\_l}, \Theta_{z2\_l}, \Theta_{c\_l}, \Theta_{z1\_g}, \Theta_{z2\_g}, \Theta_{c\_g}$）和projection layer，SDXL本身freeze。这跟ControlNet原paper的训练protocol一致。

## Inference的Classifier-Free Guidance

Eq. 8用了三条件CFG：

$$\epsilon_{prd} = \epsilon_{none} + \lambda_t(\epsilon_t - \epsilon_{none}) + \lambda_i(\epsilon_{t\&i} - \epsilon_t)$$

- $\epsilon_{none}$: 无ID无text的noise prediction
- $\epsilon_t$: 只有text的noise prediction
- $\epsilon_{t\&i}$: 有ID有text的noise prediction
- $\lambda_t = 7.5$, $\lambda_i = 5.0$

训练时按0.05概率随机drop image prompt / text prompt / both，生成三种条件的training signal。推理时用这个公式extrapolate，text和ID分别guide。

这个CFG变体在PhotoMaker、InstantID里都有类似设计，不是RealisID的创新，但用了ID-aware的版本。

## 实验设置的关键细节

- **Base model**: SDXL-1.0（比SD-1.5强很多，但IP-Adapter-face-plus和FlashFace用SD-1.5，所以comparison有base model差异，作者有说明）
- **训练数据**: CosmicMan，2M single-person image-text pairs，这是Li et al. 2024提出的高质量数据集
- **训练硬件**: 8× NVIDIA H20 GPU，batch size 16，lr 1e-5，Adam
- **Inference**: 30-step DDIM sampler，delayed subject conditioning（借鉴FastComposer）

Evaluation用40个CelebA-HQ unseen identity × 35 prompts × 2 images = 2800 images per method。

Small face定义：face bbox long side < 1/6 image edge。对于不能控制face size的方法，加text prompt "a full-body people image"引导生成small face。

## 主实验数据解读

Table 1的核心数字：

**Regular case**（face占1/4到1/2）：
- RealisID在四个metric都是第二，PuLID在ASP和CLIP-T第一但FaceNet和CLIP-I落后
- 这个case大家都做得不错，gap不大

**Small face case**（face占1/7到1/6）：
- IP-Adapter-face-plus和FlashFace完全无法生成small face（论文原文说"fail to generate"）
- PhotoMaker也fail
- InstantID FaceNet 0.693, PuLID暴跌到0.497
- RealisID 0.767，比InstantID高10.7%，比PuLID高54%

Table 3的scale robustness对比更直观：

| Face size | 1/4 | 1/5 | 1/6 | 1/7 |
|---|---|---|---|---|
| InstantID | 0.765 | 0.745 | 0.708 | 0.664 |
| RealisID | 0.791 | 0.787 | 0.772 | 0.748 |

InstantID从1/4到1/7掉0.101，RealisID只掉0.043。这就是crop+up-sample把scale variation剥离出learning的量化体现。

Table 7的pose control精度（L1距离，越低越好）：

| Face size | 1/4 | 1/7 |
|---|---|---|
| InstantID | 0.0313 | 0.0339 |
| RealisID | 0.0187 | 0.0231 |

RealisID的pose控制精度比InstantID高40%左右，因为local branch的latent跟landmarks spatial size对齐，alignment更准。

## 我的几个疑问

读完paper有几个想深入的点：

1. **Local branch的crop边界artifact**：crop face bbox再up-sample，face边界外的信息（hair、neck、shoulder）丢失。生成的face跟周围body的衔接是否会有seam？论文没讨论这个，但实际productize可能遇到。

2. **Relocation的down-sample精度**：从64×64 down-sample回18×18（如果face占1/7）会有信息损失。这个loss是否影响fine detail preservation？

3. **Multi-person overlapping case**：论文Figure 7和13展示了overlap场景，但没quantitative评估overlap程度对identity fidelity的影响。如果两人face bbox重叠50%，local injection sum在重叠区域会有interference吗？

4. **$c_{loc}$的coarse性**：binary mask只表示face/non-face，没有提供face orientation、body pose的详细信息。Global branch能否学到足够丰富的layout prior？还是大部分layout靠SDXL的text-to-image能力？

5. **为什么不用cross-attention注入ID**：IP-Adapter用cross-attention把ID features inject到U-Net，RealisID用ControlNet-style的zero conv injection。两种方式的representational capacity差异在哪？作者没对比。

## 相关联想

这个工作让我想到几个更广的topic：

**Spatial normalization的普适性**：crop+up-sample本质是把spatial scale variation从learning中剥离。这个idea在face detection（S³FD的scale-equitable design）、small object detection（SOD-MTGAN）、medical imaging（patch-based processing）都有应用。RealisID把它搬到diffusion latent space，是合理迁移。

**Local-Global complementarity的范式**：双branch处理不同scale的思路在super-resolution（Laplacian pyramid）、image generation（coarse-to-fine GAN）、video processing（short-long term attention）都很常见。RealisID是把这个范式应用到ID customization的control signal injection。

**Zero-shot multi-person的工程价值**：通过injection的sum/average规则实现single-person到multi-person的迁移，避免了multi-person训练数据稀缺问题。这个思路在FastComposer（localized attention）也有类似设计，但RealisID用ControlNet实现更clean。

**Limitation的诚实**：论文最后承认multi-person下text prompt无法精确控制每个人的clothing/hairstyle，因为text encoder的spatial understanding弱。这其实是整个text-to-image领域的开放问题，StableVITON、MM-Diff等试图解决但都不彻底。RealisID没claim解决这个，是诚实的。

## Web Links

主流程相关：
- [Stable Diffusion / Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [SDXL](https://arxiv.org/abs/2307.01952)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
- [DDIM Sampler](https://arxiv.org/abs/2010.02502)

ID customization baselines:
- [IP-Adapter](https://arxiv.org/abs/2308.06721)
- [InstantID](https://arxiv.org/abs/2401.07519)
- [PuLID](https://arxiv.org/abs/2404.02601)
- [PhotoMaker](https://arxiv.org/abs/2312.04461)
- [FlashFace](https://arxiv.org/abs/2403.17008)
- [FastComposer](https://arxiv.org/abs/2305.10431)
- [DreamBooth](https://arxiv.org/abs/2208.12242)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Textual Inversion](https://arxiv.org/abs/2208.01618)

辅助工具：
- [CLIP](https://arxiv.org/abs/2103.00020)
- [FaceNet](https://arxiv.org/abs/1503.03832)
- [MTCNN](https://arxiv.org/abs/1604.02878)
- [BiSeNet](https://arxiv.org/abs/1808.00897)
- [MediaPipe](https://arxiv.org/abs/1906.08172)
- [DECA](https://arxiv.org/abs/2012.04012)

Inspiration来源：
- [SOD-MTGAN (small object detection)](https://arxiv.org/abs/1804.02646)
- [Feature super-resolution for small object detection](https://arxiv.org/abs/1909.07360)

训练数据：
- [CosmicMan](https://arxiv.org/abs/2404.01294)
- [LAION-5B](https://arxiv.org/abs/2210.08402)
- [CelebA-HQ](https://arxiv.org/abs/1710.10196)

相关多person/controllable generation:
- [InstantFamily (multi-ID)](https://arxiv.org/abs/2404.19427)
- [StableVITON (spatial control)](https://arxiv.org/abs/2312.01725)
- [T2I-Adapter](https://arxiv.org/abs/2311.17084)

希望这个版本讲清楚了RealisID的engineering story和核心insight。如果还想深入哪个部分（比如zero conv的gradient flow、relocation的tensor操作细节、CFG的推导），可以继续聊。

---

# RealisID深度解析

你好Andrej！这篇RealisID是Alibaba DAMO Academy联合Wuhan University of Technology、Shanghai AI Lab等机构的工作，投往AAAI 2025。它聚焦于一个具体且实际的问题：identity customization在**small face**场景下identity fidelity退化，且现有方法缺乏fine control和multi-person支持。这工作体现了清晰的engineering insight——通过**local/global complementarity**来解耦different concerns。让我深入讲解。

## 核心问题动机

现有ID customization方法的痛点在三个维度上同时存在：

| 维度 | 现状问题 | RealisID对策 |
|---|---|---|
| Small face identity fidelity | IP-Adapter-face-plus和FlashFace完全无法生成small face；PhotoMaker/InstantID/PuLID在small face下identity显著退化 | Local branch crop + up-sample实现scale-robust |
| Location/Pose/Expression控制 | text prompt难以精确描述这些空间因素 | facial landmarks + binary mask作为显式condition |
| Multi-person customization | 需要multi-person training data，稀缺 | 两个ControlNet variant的injection可以分别sum/average |

特别值得思考的是small face问题——这其实是个**feature dominance**问题。当face只占图像一小部分（比如1/7 image edge），在latent space $z_t \in \mathbb{R}^{H \times W \times C}$中，face region只占少数spatial positions，face-irrelevant背景信息dominate了整个latent representation。这就导致latent embedding与face-focused condition（如facial landmarks）难以对齐，从而fine control失效。

## 架构深度解析

### 整体框架

RealisID基于SDXL-1.0，构建两个ControlNet variant作为branch：

```
                    Reference Image
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         Face detect   Landmarks   BBox mask
              │           │           │
              ▼           ▼           ▼
           $p_{id}$    $c_{p\&e}$   $c_{loc}$
              │           │           │
              │           ▼           ▼
              │     Local Branch   Global Branch
              │     (ControlNet)    (ControlNet)
              │           │           │
              │      $\hat{i}_{t\_l}$│  $i_{t\_g}$
              │           │           │
              └───────────┴───────────┘
                          ▼
                  SDXL U-Net Denoiser
                          ▼
                     Generated Image
```

### 条件信号提取

1. **ID embedding** ($p_{id}$):
   - MTCNN检测face bounding box → 调整为square（按longer edge）
   - crop face region → BiSeNet face parsing zero-out背景
   - CLIP image encoder penultimate hidden layer → projection layer → 对齐到U-Net latent space
   - $p_{id}$作为两个branch的prompt signal

2. **Pose-Expression representation** ($c_{p\&e}$):
   - MediaPipe提取cropped face的facial landmarks
   - 作为local branch的condition input
   - 这跟InstantID思路类似，但InstantID用whole-image landmarks

3. **Location guidance** ($c_{loc}$):
   - Binary single-channel mask: face bbox内为1，其余为0
   - 隐式包含body和background的layout信息
   - 同时用于global branch input和local branch的relocation

## 关键公式解析

### Preliminary

**SD loss (Eq. 1)**:

$$\mathcal{L}_{sd} = \mathbb{E}_{z_t, p, \epsilon \sim \mathcal{N}(0,1), t \sim \mathcal{U}(1,T)} [||\epsilon_\theta(z_t, p, t) - \epsilon||_2]$$

- $z_t$: noisy latent at timestep $t$
- $p$: prompt signal
- $\epsilon$: 从标准Gaussian采样的noise
- $T$: 最大timestep
- $\epsilon_\theta$: U-Net denoiser，参数为$\theta$

**ControlNet injection (Eq. 2)**:

$$i_t = \mathcal{Z}(\mathcal{F}(z_t + \mathcal{Z}(c; \Theta_{z1}), p, t; \Theta_c); \Theta_{z2})$$

- $c$: input condition
- $\mathcal{Z}(\cdot; \Theta_{z1})$, $\mathcal{Z}(\cdot; \Theta_{z2})$: 两个zero convolution layers（参数初始化为0，保证训练初期不破坏pre-trained SD）
- $\mathcal{F}(\cdot, \cdot, \cdot; \Theta_c)$: trainable ControlNet copy blocks

### Local Branch

**原始形式 (Eq. 3)**:

$$i_{t\_l} = \mathcal{Z}(\mathcal{F}(z_t + \mathcal{Z}(c_{p\&e}; \Theta_{z1\_l}), p_{id}, t; \Theta_{c\_l}); \Theta_{z2\_l})$$

这里$z_t$是whole-image noisy latent，face-irrelevant信息会污染injection。

**改进形式 (Eq. 4 + 5)**:

$$\hat{i}_{t\_l} = \mathcal{R}(i_{t\_l}, c_{loc})$$

$$i_{t\_r l} = \mathcal{Z}(\mathcal{F}(\hat{z}_t + \mathcal{Z}(c_{p\&e}; \Theta_{z1\_r l}), p_{id}, t; \Theta_{c\_r l}); \Theta_{z2\_r l})$$

这里：
- $\hat{z}_t$: 从$z_t$中crop face region后bilinear up-sample到fixed input size的latent
- $\mathcal{R}(\cdot, \cdot)$: relocation操作——把$\hat{z}_t$经过ControlNet得到的feature先down-sample回face相对整图的相对大小，然后根据$c_{loc}$放到zero tensor的对应position

**核心intuition**: 
- crop+up-sample把所有reference image的face强制到同一spatial size，**消除scale variation**
- 这样local branch不需要学习scale invariance，只需学习face details
- relocation保证location信息不丢失，face仍然在正确位置

这思路其实借鉴自small object detection中的feature super-resolution工作（Bai et al. 2018 SOD-MTGAN; Noh et al. 2019），但巧妙应用到diffusion latent space。

### Global Branch (Eq. 6)

$$i_{t\_g} = \mathcal{Z}(\mathcal{F}(z_t + \mathcal{Z}(c_{loc}; \Theta_{z1\_g}), p_{id}, t; \Theta_{c\_g}); \Theta_{z2\_g})$$

- 直接用整个$z_t$，没有crop
- condition是$c_{loc}$而不是$c_{p\&e}$
- 因为$c_{loc}$隐式编码了body和background layout
- 这个branch管overall harmony和face location

### Training Loss (Eq. 7)

$$\mathcal{L} = \mathbb{E}_{z_t, p_{text}, \epsilon, t} [||\epsilon_\theta(z_t, p_{text}, t) - \epsilon||_2 + \lambda ||(\epsilon_\theta(z_t, p_{text}, t) - \epsilon) \odot c_{loc}||_2]$$

- $\odot$: element-wise multiplication
- 第一项: 标准SD noise prediction loss（全局）
- 第二项: face region内的noise prediction loss增强项
- $\lambda = 1.0$: 平衡系数
- 只训练两个branch和projection layer，SDXL freeze

这个loss设计很关键：第二项让noise prediction在face region更准确，间接增强local branch学习信号。

### Inference with CFG (Eq. 8)

$$\epsilon_{prd} = \epsilon_{none} + \lambda_t(\epsilon_t - \epsilon_{none}) + \lambda_i(\epsilon_{t\&i} - \epsilon_t)$$

- $\epsilon_{none}$: no ID, no text
- $\epsilon_t$: only text prompt
- $\epsilon_{t\&i}$: both ID embedding and text prompt
- $\lambda_t = 7.5$, $\lambda_i = 5.0$
- 这是classifier-free guidance的双condition扩展

### Multi-person Inference

- Local branch: 多人injection information **相加**（因为每人injection只在local region非零）
- Global branch: 多人injection information **取平均**（共享global layout）
- 训练只用single-person data，但推理可扩展到multi-person

## 实验数据深度分析

### Table 1: 主对比

| Method | Base | Regular ASP | Regular CLIP-T | Regular FaceNet | Regular CLIP-I | Small ASP | Small CLIP-T | Small FaceNet | Small CLIP-I |
|---|---|---|---|---|---|---|---|---|---|
| IP-Adapter-face-plus | SD-1.5 | 5.64 | 0.201 | 0.760 | 0.714 | - | 0.229 | 0.516 | 0.658 |
| FlashFace | SD-1.5 | 5.46 | 0.212 | 0.809 | 0.754 | - | - | - | - |
| PhotoMaker | SDXL | 5.74 | 0.223 | 0.508 | 0.651 | 5.78 | - | - | - |
| InstantID | SDXL | 6.01 | 0.211 | 0.768 | 0.706 | 5.72 | 0.210 | 0.693 | 0.686 |
| PuLID | SDXL | 6.37 | 0.254 | 0.772 | 0.697 | 5.95 | 0.249 | 0.497 | 0.585 |
| **RealisID** | **SDXL** | **6.22** | **0.234** | **0.796** | **0.739** | **6.11** | **0.236** | **0.767** | **0.701** |

关键观察：
1. **Small face scenario下RealisID全面领先**：FaceNet 0.767 vs 0.693 (InstantID) → **+10.7%** identity fidelity提升
2. PuLID在small face下FaceNet暴跌到0.497，几乎丧失identity
3. IP-Adapter-face-plus和FlashFace完全无法生成small face（论文中定义为long side < 1/6 image edge）

### Table 2: Ablation

| Config | ASP | CLIP-T | FaceNet | CLIP-I |
|---|---|---|---|---|
| w/o $B_{local}$ | 6.07 | 0.243 | 0.681 | 0.673 |
| w/o $B_{global}$ | **5.97** | 0.241 | 0.734 | 0.689 |
| **Full Model** | **6.11** | 0.236 | **0.767** | **0.701** |

非常有意思的发现：
- 去掉$B_{local}$后FaceNet从0.767暴跌到0.681（-11.2%），证明local branch是identity fidelity的关键
- 去掉$B_{global}$后identity metrics反而**升高**（FaceNet 0.734 > 0.767? 实际是降低），但ASP掉到5.97最低——说明global branch管overall harmony，去掉后face孤立
- 这个ablation揭示了**identity fidelity和overall harmony之间存在trade-off**，两个branch协作才能兼顾

### Table 3: Scale Robustness

| Face Relative Size | 1/4 | 1/5 | 1/6 | 1/7 |
|---|---|---|---|---|
| InstantID | 0.765 | 0.745 | 0.708 | 0.664 |
| RealisID | **0.791** | **0.787** | **0.772** | **0.748** |

- RealisID从1/4到1/7只下降0.043，而InstantID下降0.101
- RealisID在1/7的small face下FaceNet仍达0.748，InstantID只剩0.664
- 这就是crop+up-sample的威力——把scale variation从模型学习中剥离

### Table 7 & 8: Fine Control (Pose & Expression)

Pose control (L1, lower better):
| Face Size | 1/4 | 1/5 | 1/6 | 1/7 |
|---|---|---|---|---|
| InstantID | 0.0313 | 0.0316 | 0.0333 | 0.0339 |
| RealisID | **0.0187** | **0.0194** | **0.0205** | **0.0231** |

Expression control (L1):
| Face Size | 1/4 | 1/5 | 1/6 | 1/7 |
|---|---|---|---|---|
| InstantID | 0.2591 | 0.2622 | 0.2704 | 0.2738 |
| RealisID | **0.1792** | **0.1821** | **0.1966** | **0.2032** |

RealisID在pose和expression控制精度上都明显优于InstantID，且gap随face变小而扩大——这又是scale robustness的体现。

### Table 9: Multi-person

| Setting | 1/4 | 1/5 | 1/6 | 1/7 |
|---|---|---|---|---|
| RealisID (single) | 0.791 | 0.787 | 0.772 | 0.748 |
| RealisID (multi) | 0.788 | 0.786 | 0.772 | 0.746 |

multi-person下FaceNet几乎无退化，证明local injection的sum策略确实保持了identity independence。

## Implementation Details

- **训练数据**: CosmicMan (2M image-text pairs, single individual)
- **GPU**: 8× NVIDIA H20
- **Optimizer**: Adam, batch size 16, lr=1e-5, weight decay=1e-2
- **Prompt drop**: 0.05概率drop image prompt / text prompt / both（IP-Adapter策略，为CFG训练）
- **Sampler**: 30-step DDIM
- **Delayed subject conditioning**: 借鉴自FastComposer
- **Pre-processing**: MTCNN (detection) + BiSeNet (parsing) + MediaPipe (landmarks)

## 相关联想与Intuition构建

### 1. 为什么Local + Global比单一branch更好？

直觉上，**face generation本质是two-scale problem**：
- Face region需要pixel-level精确identity和expression
- 整图需要scene-level harmony和合理layout
- 把这两个scale耦合在一个branch，模型会在trade-off中挣扎
- RealisID通过物理分离两个scale的processing，让每个branch专注自己的concern

### 2. Crop + Up-sample的深层意义

这其实是**spatial normalization**思想：
- 不同reference image的face大小不同，导致latent representation scale variation
- 通过crop+up-sample到fixed size，把scale factor从learning问题转化为preprocessing
- 模型只需在normalized space学习face details，无需scale invariance

类比：BatchNorm把activation分布normalize到固定mean/variance，让网络不必学习distribution shift。RealisID的crop+up-sample是spatial的类似操作。

### 3. Zero Convolution的重要性

ControlNet的zero conv设计保证了：
- 训练初始时injection = 0，不破坏pre-trained SD
- gradient可以flow到copy blocks训练
- 渐进式加入新能力，避免catastrophic forgetting

RealisID用了**两套zero conv**（每个branch独立），保证两branch独立学习。

### 4. Multi-person扩展的优雅性

最精彩的设计在于：
- Local branch injection只在face region非零，所以多人相加不会interference
- Global branch injection代表global layout，多人平均得到合理layout
- 这个设计使得**single-person训练直接迁移到multi-person**，无需multi-person data
- 这是工程上的"free lunch"

### 5. Limitation的诚实

论文坦诚承认：multi-person下text prompt控制face-irrelevant区域（如clothing, hairstyle）困难，因为text encoder缺乏spatial understanding。这跟StableVITON、MM-Diff等工作的spatial control问题相通。

### 6. 与其他工作的关系

- **InstantID**: 也用landmarks作为pose control，但用whole-image latent，没有scale robustness
- **IP-Adapter**: ID features直接cross-attention到U-Net，没有local/global分离
- **PuLID**: contrastive alignment提升identity，但small face退化严重
- **PhotoMaker**: stacked ID embedding，但project到text space丢失fine details
- **FastComposer**: localized attention处理multi-subject，但需要attention manipulation
- **CosmicMan**: 提供高质量训练数据，但不解决inference控制问题

## Web Links Reference

- [Stable Diffusion / LDM](https://arxiv.org/abs/2112.10752)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [SDXL](https://arxiv.org/abs/2307.01952)
- [IP-Adapter](https://arxiv.org/abs/2308.06721)
- [InstantID](https://arxiv.org/abs/2401.07519)
- [PuLID](https://arxiv.org/abs/2404.02601)
- [PhotoMaker](https://arxiv.org/abs/2312.04461)
- [FlashFace](https://arxiv.org/abs/2403.17008)
- [FastComposer](https://arxiv.org/abs/2305.10431)
- [Classifier-free Guidance](https://arxiv.org/abs/2207.12598)
- [CosmicMan](https://arxiv.org/abs/2404.01294)
- [MTCNN](https://arxiv.org/abs/1604.02878)
- [BiSeNet](https://arxiv.org/abs/1808.00897)
- [MediaPipe](https://arxiv.org/abs/1906.08172)
- [DDIM](https://arxiv.org/abs/2010.02502)
- [FaceNet](https://arxiv.org/abs/1503.03832)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [DECA](https://arxiv.org/abs/2012.04012)
- [DreamBooth](https://arxiv.org/abs/2208.12242)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Textual Inversion](https://arxiv.org/abs/2208.01618)
- [SOD-MTGAN (small object detection inspiration)](https://arxiv.org/abs/1804.02646)
- [StableVITON](https://arxiv.org/abs/2312.01725)
- [RealisID arxiv (推测)](https://arxiv.org/abs/2501.01304)

## 最终思考

RealisID的工程美感在于**问题decomposition**：把ID customization拆成local face details + global harmony两个sub-problem，每个用专门branch解决。Local branch通过crop+up-sample获得scale robustness是核心insight，类似BatchNorm的spatial版。Multi-person inference通过injection sum/average的简单规则，从single-person训练直接扩展，是工程上的优雅free lunch。

不过有些值得思考的点：
1. Local branch的crop+up-sample是否丢失了face region的上下文信息（如hair连接）？
2. Relocation后的injection在face boundary处是否有artifact？
3. Multi-person overlapping区域的identity mixing如何避免？
4. Loss第二项的$\lambda$为何选1.0？更大会怎样？

希望这些解析能帮你build对ID customization中local/global complementarity的intuition。
