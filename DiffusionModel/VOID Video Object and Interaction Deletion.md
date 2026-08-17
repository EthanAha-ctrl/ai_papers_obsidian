---
source_pdf: VOID Video Object and Interaction Deletion.pdf
paper_sha256: 7d7acde8762c6323614e4af57d5f13d1b5f2859b1ab9b21b6095c719dcb47c65
processed_at: '2026-08-13T03:16:55-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VOID 用人话说一遍

好，Andrej，咱们坐下来用大白话重新过一遍这篇paper的intuition。

---

## 这篇paper到底在解决什么问题

想象你在看一段视频：一排domino正在倒下，一个接一个。你想把中间几块domino删掉，做成"这几块domino从来就不存在"的效果。

这时候问题来了 —— 如果中间几块domino不存在，后面那些domino**根本就不应该倒**，因为没人推它们。

但传统video object removal工具是怎么做的呢？它们看到中间有空缺，就**filling holes**，把背景补上，然后完事了。后面的domino该倒还是倒，因为你删掉的那几块domino对它们造成的"已经发生"的影响，工具根本不管。这就像你把一个人从照片里P掉，但他的影子还留在地上一样荒谬。

传统的video inpainting方法（ProPainter、DiffuEraser、MiniMax-Remover等）本质上都是**spatial hole filling**。它们的工作逻辑是：

$$\hat{I}_t(p) = g(I_{t-1}, I_{t+1}, \text{context}(p))$$

这里 $I_t$ 是第 $t$ 帧，$p$ 是pixel位置，$g$ 是某种propagation或generation函数。意思是：当前帧某个pixel的值，由前后帧和周围context决定。这种做法能处理shadows、reflections这些**appearance-level**的东西，因为背景的pixel在别的帧里能看到，复制过来就行。

但collisions、support、physical interactions这些东西，**counterfactual outcome在observed pixels里根本不存在**。你不能从视频里"找到"domino不倒的样子，因为视频里domino就是倒了。这个outcome需要被**synthesize**出来，需要model理解物理世界会怎样演化。

所以这篇paper的核心论点是：**video object removal本质上是一个counterfactual world simulation问题**，而不是pixel inpainting问题。

---

## VOID的核心idea：三个trick的组合

VOID没有发明什么全新的architecture，它的聪明之处在于把三个东西拼到一起：

1. **用物理引擎造counterfactual训练数据**
2. **用quadmask告诉model"哪里要改、哪里要保留"**
3. **用VLM做因果推理 + diffusion model做视觉生成**

咱们一个一个说。

---

### Trick 1: 用物理引擎造"假如..."的数据对

要训练一个model生成counterfactual video，你需要成对的数据：**有object O的视频** vs **没有object O的同一个视频**。

现实中你不可能同时拍到这两个版本，但物理引擎可以。

作者用了两个data source：

**Kubric**（Google的物理仿真引擎）：
- 随机生成一堆objects，给它们初始位置 $\{s_i^0\}$ 和速度 $\{v_i^0\}$
- 跑一遍物理仿真，录下来，得到 $\mathbf{V}$（有object O的版本）
- 删掉object O，**保持其他objects的初始条件完全一样**，重新跑一遍仿真，得到 $\hat{\mathbf{V}}$（没有O的版本）

$$\mathbf{V} = \text{Simulate}(\{s_i^0, v_i^0\}_{\text{all objects}})$$
$$\hat{\mathbf{V}} = \text{Simulate}(\{s_i^0, v_i^0\}_{\text{all except } O})$$

这就拿到了一个ground truth counterfactual pair。约1900对。

**HUMOTO**（人体动作捕捉数据集）：
- 人在跟各种物体交互（推、拿、踢等）
- $O$ 就是人本身
- 两次渲染：有人 vs 没人
- 随机化textures让数据更多样
- 约4500对

**关键的细节**：渲染时randomize camera trajectories和focal zoom。为什么？因为如果camera motion总是一样的，model可能会学到spurious correlation，比如"camera往左走的时候object就应该往右飞"。Randomize camera能force model去理解真正的physical dynamics，而不是memorize camera-object的co-occurrence。

这个思路本质上是Pearl的**do-calculus**在video generation中的应用。你不是在观察"O不存在"的数据（observational），你是在**干预**"让O不存在"（interventional），然后看world state怎么变。物理引擎让你能做这种intervention。

---

### Trick 2: Quadmask —— 告诉model因果结构

有了训练数据，下一个问题是：怎么告诉model"哪里该改，哪里不该改"？

之前的Generative Omnimatte用了一个**trimask**（三色mask）：
- **黑色**：要删的object
- **浅灰**：可能受影响的区域
- **白色**：保留不动的区域

但trimask有两个问题：

**问题1：浅灰太大**。Gen-Omnimatte把几乎整张图都涂成浅灰，只把明确不变的object涂成白色。这样model学到的是"通常只需要在浅灰区域里改很小一部分"。guidance太弱了，model不知道到底哪里需要改。

VOID的fix：把浅灰区域**精确聚焦**到effects真正发生的地方，然后gridify（切成grid blocks），这样inference时VLM也能用grid format生成。

**问题2：Overlap的ambiguity**。想象一个男孩在接球，你要删掉男孩。男孩upper body那个区域：
- 应该涂黑色吗？因为男孩要被删掉
- 还是应该涂浅灰？因为球会飞过这个区域（counterfactual里球没人接，会继续飞）

这两个信息都是true的，但trimask只能选一个颜色。

VOID的fix：引入**第四色dark grey**，表示"这个pixel既要被删掉（属于object），又是新event要发生的地方（属于affected region）"。

所以quadmask $\mathbf{M}_q$ 的定义是：

$$\mathbf{M}_q(p) = \begin{cases} \text{black} & p \in \mathbf{M}_o \setminus \mathbf{M}_a \\ \text{dark grey} & p \in \mathbf{M}_o \cap \mathbf{M}_a \\ \text{light grey} & p \in \mathbf{M}_a \setminus \mathbf{M}_o \\ \text{white} & \text{otherwise} \end{cases}$$

这里：
- $\mathbf{M}_o$：要删除的object的mask
- $\mathbf{M}_a$：受影响区域的mask
- $p$：pixel位置
- $\setminus$：集合差
- $\cap$：集合交
- $\vee$：集合并

换句话说：
- **Black** = 只属于object，不属于affected area → 纯删除
- **Dark grey** = 既属于object又属于affected area → 删掉的同时这里会发生新event
- **Light grey** = 只属于affected area，不属于object → 这里要有新的物理event发生
- **White** = 都不属于 → 保持不变

这个quadmask本质上就是一个**local causal graph的pixel-space encoding**。它告诉model："这里有个cause（black/dark grey），那里会有个effect（light grey/dark grey），其他地方别碰（white）"。

---

### Trick 3: VLM做因果推理 + Diffusion做视觉生成

训练时，quadmask是从物理引擎的GT数据直接提取的。但inference时，用户只给你一个binary mask（点几下要用SAM 2生成），你怎么知道affected region在哪？

这时候需要**因果推理**：看到一个人拿着气球，你得reason出"如果这个人消失了，气球会飞走"。这是world knowledge，不是视觉feature能告诉你的。

VOID的方案：用VLM（主paper用Gemini 3 Pro）做这个推理。Pipeline是：

1. 用户sparse clicks → **SAM 2** → binary object mask $\mathbf{M}_o$
2. VLM看视频 + $\mathbf{M}_o$ → 产出一份"哪些objects会受影响"的描述
3. **SAM 3**根据这份描述 → 生成affected objects的mask $\mathbf{M}_a^{orig}$
4. VLM再预测这些affected objects在counterfactual里会去哪 → overlay一个coarse spatial grid，VLM标出哪些grid cells会有effect → block-structured mask $\mathbf{M}_a^{count}$
5. 合并：$\mathbf{M}_a = \mathbf{M}_a^{orig} \vee \mathbf{M}_a^{count}$
6. 根据上面的公式构造quadmask $\mathbf{M}_q$

这里有个很巧妙的设计：为什么要预测counterfactual position？因为affected object在counterfactual里可能**移动到别的位置**。比如一个人举着一个ball，ball在人头顶。删掉人之后，ball会掉到地上。如果只标ball在头顶的原始位置（$\mathbf{M}_a^{orig}$），model不知道ball会掉到哪里去。所以VLM还需要预测ball在counterfactual里的新位置（$\mathbf{M}_a^{count}$），这样affected area mask才能覆盖ball的完整trajectory。

这个VLM + diffusion的分离设计是这篇paper的philosophical core：**VLM擅长high-level causal reasoning但不擅长visual synthesis，diffusion model擅长visual synthesis但不擅长causal reasoning。把两者组合，各做自己擅长的事。**

---

## Two-Pass：为什么需要一个第二遍

VOID的generation分两pass。

### Pass 1: 初步counterfactual生成

$$\hat{\mathbf{V}}_{p1} = \mathrm{VOID}(\mathbf{z}, \mathbf{V}, \mathbf{M}_q) \tag{2}$$

这里：
- $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：标准Gaussian noise，diffusion的起始noise
- $\mathbf{V}$：input video
- $\mathbf{M}_q$：quadmask
- $\hat{\mathbf{V}}_{p1}$：Pass 1的输出

Pass 1通常能capture正确的motion hypothesis（比如"这个object应该free fall"），但会有一个烦人的问题：**object会变形**。

为什么？因为CogVideoX只有5B参数，在生成复杂motion时，temporal coherence不够好 [4, 28]。传统inpainting里，input video提供了强constraint —— 背景的motion和geometry不变，所以model只需要filling。但counterfactual场景下，model需要**generate new motion**，这时候lightweight video diffusion model就容易出artifacts：object会bending、stretching、structural drift。

你可以想象成：model知道"球应该往下掉"，但在逐帧denoise的过程中，球的shape在不同帧之间drift了，最后球看起来像被压扁了或者拉长了。

### Pass 2: 用warped noise fix变形

这个fix来自一个叫**Go-with-the-Flow** [2]的insight。

Standard diffusion里，noise $\mathbf{z}$ 是每个pixel独立采样的：

$$\mathbf{z}^{(t)}(p) \sim \mathcal{N}(0, 1) \quad \text{independently for each } p, t$$

这里 $t$ 是frame index，$p$ 是pixel位置。每个pixel每个frame的noise都是独立random的。

但如果我们已经知道了motion trajectory（从Pass 1的output算optical flow），那么同一个object在不同frame的noise应该是**correlated的** —— 因为同一个物理点在不同frame出现的位置不同，但它是同一个点，noise应该"跟着它走"。

所以Pass 2的做法是：从Pass 1的output $\hat{\mathbf{V}}_{p1}$ 计算optical flow field，然后warp noise让noise沿motion trajectory保持correlated：

$$\mathbf{z}_{warp}^{(t+1)}(p') = \mathbf{z}_{warp}^{(t)}(p) \quad \text{where } p' = p + \text{flow}(p, t)$$

这里：
- $\text{flow}(p, t)$：pixel $p$ 在frame $t$ 到 $t+1$ 的optical flow vector
- $p'$：pixel $p$ 在下一frame的新位置
- $\mathbf{z}_{warp}^{(t)}(p)$：frame $t$ 在pixel $p$ 处的warped noise

然后用这个warped noise做第二pass：

$$\hat{\mathbf{V}} = \mathrm{VOID}_{warp}(\mathbf{z}_{warp}, \mathbf{V}, \mathbf{M}_q) \tag{3}$$

这里 $\mathrm{VOID}_{warp}$ 是用同样的数据和quadmask训练、但用flow-aligned noise的VOID变体。

**直觉上为什么这能work**：warped noise让diffusion process"沿着"object的运动轨迹denoise。如果noise跟着object走，那么denoising过程也会跟着object走，object的shape就不会在frame之间drift。就像你画动画的时候，如果你先画好motion path再沿着path画object，object就不容易变形。

**Pass 2是conditional触发的**：VLM判断这个case是否需要substantial dynamic reconfiguration（free-fall、trajectory change等），只在需要时才跑第二pass。在75个real-world test cases里，只有10个被标记为需要Pass 2。

从Table 6的数据看，这10个case上Pass 2的贡献很大：

| Pass | Int.Phys | Obj.Rem | Bg.Art | Temp | Pres | Sharp | Total |
|------|----------|---------|--------|------|------|-------|-------|
| Pass 1 | 2.90 | 4.20 | 3.70 | 3.80 | 4.90 | 4.00 | 23.5 |
| Pass 2 | 3.90 | 4.90 | 4.00 | 4.20 | 4.80 | 4.20 | 26.0 |

Interaction Physics从2.90跳到3.90，涨了整整1分（满分5分）。这说明deformation确实是Pass 1在难case上的主要failure mode，warped noise有效解决了它。

---

## 实验结果的人话解读

### Human Preference（Table 1）

25个人，每人看5个场景，从7个model的output里选最好的。

| Model | Win % |
|-------|-------|
| **VOID** | **64.8** |
| Runway | 18.4 |
| Gen-Omnimatte | 11.2 |
| DiffuEraser | 4.0 |
| ROSE | 1.6 |
| MiniMax-Remover | 0.0 |
| ProPainter | 0.0 |

注意Runway是**closed-source commercial model**，而且作者还**额外给它写了text prompt**告诉它应该发生什么（比如"remove the person and ensure the held object falls naturally"）。即使这样，Runway也只有18.4%。ProPainter和MiniMax-Remover直接0%。

这说明什么？问题不是出在model capacity上（Runway肯定比5B的CogVideoX大得多），而是出在**task formulation**上。你把counterfactual video editing当成inpainting来做，再大的model也做不好，因为问题的structure就不对。

### VLM-as-Judge（Table 2）

三个不同的VLM（Gemini 3 Pro、GPT 5.2、Qwen 3.5-32B）当judge，每个output在6个维度上打分（0-5），总分30。

VOID在三个judge下都是总分第一。最有说服力的是**Interaction & Physics**这个维度（直接评估"删掉object后物理后果对不对"）：

- Gemini judge: VOID 3.66 vs Runway 2.61 vs Gen-Omni 2.30
- GPT judge: VOID 3.19 vs Runway 1.85 vs Gen-Omni 1.39
- Qwen judge: VOID 2.64 vs DiffuEraser 2.19 vs Gen-Omni 2.19

VOID在物理reasoning上的优势是consistent的，跨所有judge。

### Synthetic Benchmark（Table 3）

40个test videos，有GT counterfactual，可以用PSNR、LPIPS、DreamSim、DINOv2、FVD、VLM-Judge来评。

| Model | PSNR↑ | LPIPS↓ | DreamSim↓ | DINOv2↑ | FVD↓ | VLM-Judge↑ |
|-------|-------|--------|-----------|---------|------|------------|
| **VOID** | **31.49** | 0.12 | **0.07** | **0.92** | **260.31** | **25.10** |
| MiniMax-Remover | 29.96 | **0.11** | 0.09 | 0.91 | 448.43 | 22.83 |
| ProPainter | 30.48 | **0.11** | 0.10 | 0.89 | 471.13 | 21.38 |
| Gen-Omnimatte | 29.44 | 0.12 | 0.12 | 0.87 | 437.88 | 20.40 |
| Runway | 26.68 | **0.11** | 0.15 | 0.85 | 442.76 | 21.67 |

VOID在**FVD**（video-level distribution distance）上碾压：260.31 vs 第二名437.88。FVD是衡量整个video distribution的最comprehensive的metric，最能反映counterfactual合理性。

LPIPS是VOID唯一没拿第一的metric。Paper给了一个很好的解释：LPIPS对local translation敏感。比如你正确地让stick掉下来了，但掉的速度跟GT差一点点，LPIPS会penalize你。但如果你直接把stick也删了（什么effect都没处理），LPIPS反而可能更好，因为stick在output和GT里都不存在。Appendix Figure 9专门展示了这个case。

这是一个很好的reminder：**传统perceptual metrics在counterfactual evaluation里可能misleading**，因为它们measuring的是appearance similarity，但counterfactual quality的core是physical plausibility。

### Ablation（Table 4）

在75个real-world test cases上用Gemini 3 Pro当judge。

**数据组成的ablation**（都1200 samples）：

| Dataset | Int.Phys | Total |
|---------|----------|-------|
| Kubric-Only | 2.63 | 20.36 |
| HUMOTO-Only | 2.50 | 20.12 |
| Both | 3.04 | 21.93 |

两种数据混合比单独用任何一种都好，即使总sample数一样。Kubric教rigid body physics（collisions、falling），HUMOTO教articulated manipulation（人跟物体交互），两者互补。这符合intuition —— diversity比scale更重要，至少在这个task上。

**Masking strategy的ablation**：

| Mask | Int.Phys | Bg.Art | Total |
|------|----------|--------|-------|
| Gen-Omni Mask (Full data) | 3.30 | 3.04 | 23.39 |
| VOID Quadmask (Full data) | 3.66 | 4.10 | 26.12 |

详细的quadmask + VLM生成pipeline贡献了+2.73总分。特别有意思的是Background & Artifacts从3.04跳到4.10（+1.06）。这说明好的mask guidance不仅help物理reasoning，还help model不去乱改不该改的地方。

---

## 最让人impress的：Generalization

Figure 6展示了VOID在real-world视频上的generalization，这些effects在训练数据里**从未出现过**：

1. **气球浮起**：一个人拿着气球，删掉人，气球往上飘走了。但训练数据里**没有floating objects**。
2. **Blender不启动**：一个人按blender开关，删掉人，blender不转了，里面的食物不动。但训练数据里**没有blenders或任何电子设备**。
3. **Jenga tower不倒**：一个人和一只猫同时在推Jenga tower，删掉猫，tower不倾斜了。
4. **Bowling pins保持站立**：删掉bowling ball和扔球的人，pins保持站立。
5. **Big Ben的reflection消失**：删掉Big Ben tower，水面上的reflection也消失了。
6. **Stick掉落**：狗咬着stick，删掉狗，stick掉地上。
7. **Ball滚过障碍**：ball碰到ducky障碍物，删掉ducky，ball滚过去了。

这些generalization不是从训练数据里recall的，是VLM的world knowledge和diffusion model的visual priors**组合**出来的emergent behavior。

比如balloon的例子：VLM知道"人拿着气球，人没了，气球没人抓了，气球比空气轻所以会往上飘"。这个reasoning是VLM的world knowledge。然后diffusion model把"气球往上飘"这个hypothesis render成视觉上plausible的frames。

这就是为什么这个framework的意义超越了object removal本身 —— 它展示了**VLM reasoning + video generation的组合可以产生emergent physical reasoning**。

---

## 整体架构的intuition总结

把整个VOID pipeline用一段话总结：

用户给一个video和sparse clicks → SAM 2生成binary mask → VLM分析"删掉这个object会影响什么"和"affected objects在counterfactual里会去哪" → SAM 3根据VLM的分析生成affected area mask → 组合成quadmask → VOID Pass 1用quadmask guide CogVideoX生成初步counterfactual video（motion对了但object可能变形）→ 如果需要，VLM判断要不要跑Pass 2 → Pass 2从Pass 1的output算optical flow，warp noise，再跑一遍diffusion fix变形 → 最终output。

整个pipeline的哲学是：**把world simulation问题decompose成reasoning（VLM）、conditioning（quadmask）、generation（diffusion）三个可工程化的部分，每部分用最合适的tool来做。**

---

## 这篇paper的bigger picture

Andrej，我觉得这篇paper真正重要的insight是：

**Video editing at its core is world simulation.**

当你删除一个object，你本质上是在问一个counterfactual question："如果这个object从未存在，world state会怎样演化？"这个问题需要internal world model。

传统方法把video editing当pixel manipulation来做，所以遇到physics就崩了。VOID把video editing当counterfactual world simulation来做，用物理引擎造supervision、用VLM做reasoning、用diffusion做synthesis，三个组件各司其职。

这跟你们一直在推的"video diffusion models should be world models"的agenda [28]完全一致。物理引擎提供的counterfactual supervision本质上就是在教model **interventional reasoning**（Pearl的do-calculus），而不是observational pattern matching。

如果把这个framework scale up —— 更大的backbone、更多样的物理场景、更长的video、更复杂的causal chains —— 我觉得这是一个很有希望通向"真正理解物理世界的video foundation model"的路径。

---

References:
- VOID project page: https://void-model.github.io
- CogVideoX: https://arxiv.org/abs/2408.06072
- Kubric: https://github.com/google-research/kubric
- Generative Omnimatte: https://arxiv.org/abs/2411.16683
- Go-with-the-Flow: https://go-with-the-flow.github.io
- SAM 2: https://arxiv.org/abs/2408.00714
- SAM 3: https://arxiv.org/abs/2511.16719
- HUMOTO: https://humoto.github.io
- PhysBench: https://arxiv.org/abs/2501.16411
- TraVL (作者prior work): https://arxiv.org/abs/2510.07550
- Do gen video models understand physics? (作者prior): https://arxiv.org/abs/2501.09038
- ProPainter: https://shangchenzhou.github.io/projects/ProPainter/
- DiffuEraser: https://arxiv.org/abs/2501.10018
- ROSE: https://arxiv.org/abs/2508.18633
- MiniMax-Remover: https://arxiv.org/abs/2505.24873
- Runway Gen-4: https://runwayml.com
- Veo 3: https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf
- DreamSim: https://dreamsim-nights.github.io
- DINOv2: https://arxiv.org/abs/2304.07193
- FVD: https://arxiv.org/abs/1812.01717
- VideoJam: https://arxiv.org/abs/2502.02492
- Wan: https://arxiv.org/abs/2503.20314
- VACE: https://arxiv.org/abs/2503.07598
- LTX-2: https://arxiv.org/abs/2601.03233

---

# VOID: Video Object and Interaction Deletion 详细解读

Andrej，这篇paper的核心insight非常漂亮 —— 它把video object removal从**pixel-level inpainting**重新frame成**counterfactual world simulation**。这个reframe本身就是一个很重要的思想转变。让我深入展开。

---

## 1. Problem Framing: 为什么传统inpainting会失败

传统的video object removal方法（ProPainter、DiffuEraser、ROSE、MiniMax-Remover等）本质上都在解决一个**spatial hole-filling**问题：

$$\hat{I}_t(p) = g\left(I_{t-1}, I_{t+1}, \text{context}(p)\right)$$

其中 $p$ 是pixel位置，$g$ 是一个propagation或generation函数。这类方法可以很好地处理：
- **Photometric effects**：shadows、reflections、translucency
- **Static background recovery**：被遮挡的背景像素恢复

但是当target object与其他object存在**kinematic coupling**时（collisions、support、articulated manipulation），它们就崩了。原因很简单：counterfactual outcome**不在observed pixels中存在**，需要被synthesize出来，而inpainting本质上是**reconstructive**的。

Paper中Figure 1的domino example是这个failure mode的完美illustration —— Gen-Omnimatte删掉中间三块domino，但最后一块yellow block仍然倒了，因为没有force作用却还在倒，这违反了Newton's first law。

---

## 2. 核心Formulation

把整个problem抽象一下。给定：

- Input video: $\mathbf{V} = \{I_t\}_{t=1}^{T}$，其中 $I_t$ 是第 $t$ 帧，$T$ 是总帧数
- Object mask: $\mathbf{M}_o = \{m_t\}_{t=1}^{T}$，标识要删除的object $O$ 在每帧的位置

目标是学一个model $f$：

$$\hat{\mathbf{V}} = f(\mathbf{V}, \mathbf{M}_o) \tag{1}$$

这里的关键insight是：$f$ 需要**三个能力**：
1. **Eliminate** target object
2. **Regenerate** affected regions through complex causal relationships
3. **Preserve** unaffected regions

这不是spatial inpainting的三个步骤，而是**counterfactual world simulation**的三个步骤。这个区别在intuition上很重要。

---

## 3. 数据构建：Counterfactual Pairs的生成

这是这篇paper最clever的地方之一。没有counterfactual ground truth，就没法supervise这种reasoning。作者用两个物理simulation source：

### 3.1 Kubric (Rigid-body dynamics)
- 来源：Kubric engine [10]，Google团队开发的scalable dataset generator
- 思路：
  1. Sample初始条件 $\{s_i^0, v_i^0\}_{i=1}^{N}$（位置和速度）
  2. Forward simulate得到 $\mathbf{V}$
  3. 选定object $O$
  4. **重新simulate**，但去掉 $O$，保持其他objects初始条件不变
  5. 得到 $\hat{\mathbf{V}}$
- 规模：约1900 video pairs

这相当于在rigid-body physics simulator里执行**do-calculus**：
$$\hat{\mathbf{V}} = \text{Simulate}(\{s_i^0, v_i^0\}_{i \neq O})$$
$$\mathbf{V} = \text{Simulate}(\{s_i^0, v_i^0\}_{\text{all}})$$

数据对 $(\mathbf{V}, \hat{\mathbf{V}})$ 就是causal effect of $O$ 的supervision。

### 3.2 HUMOTO (Articulated interactions)
- 来源：HUMOTO [25]，4D motion capture of human-object interactions
- $O$ 对应human
- 两次pass渲染：with human和without human
- 随机化textures（objects、background wall、human）
- 规模：约4500 video pairs

**关键的trick**：randomize camera trajectories和focal zoom。这有助于disentangle object effects from camera motion，防止model学到spurious correlations。

---

## 4. Quadmask: 解决Trimask的Ambiguity

Generative Omnimatte [19]提出的**trimask**有三色：
- Black: object to remove
- Light gray: affected region
- White: preserve

但trimask有两个ambiguity，作者通过**quadmask** $\mathbf{M}_q$ 解决：

### Ambiguity 1: Light gray过大
Gen-Omnimatte把几乎整张图标为light gray，只把特定object标white。这样model学到"通常只需要修改light gray里很小一部分"，guidance太弱。

**VOID的fix**: 把light gray区域**closely focus**到effects真正发生的地方，并gridify以匹配inference时VLM生成的grid mask。

### Ambiguity 2: Overlap情况
考虑Figure 3的例子：要删掉catch ball的男孩。男孩upper body区域：
- 应该是black（因为要删男孩）？
- 还是light gray（因为ball会经过这个区域）？

**VOID的fix**: 引入第四色**dark grey**表示overlap。

### Formal Definition
$$\mathbf{M}_q(p) = \begin{cases} \text{black} & p \in \mathbf{M}_o \setminus \mathbf{M}_a \\ \text{dark grey} & p \in \mathbf{M}_o \cap \mathbf{M}_a \\ \text{light grey} & p \in \mathbf{M}_a \setminus \mathbf{M}_o \\ \text{white} & \text{otherwise} \end{cases}$$

其中 $\mathbf{M}_a$ 是affected region mask。

这个设计intuition上很清晰：model需要知道"这个pixel既要被删掉，又是新event要发生的地方" —— 这两个信息不能用一个binary mask表达。

---

## 5. Architecture & Training

### Backbone
- **CogVideoX** [40] diffusion transformer (5B参数)
- 从Generative Omnimatte [19]的weights初始化

为什么选CogVideoX而不是更大的模型？Paper里提到"relatively lightweight models like the 5 billion parameter CogVideoX"。这意味着这个方法可能scalable到更大的backbone（Veo 3、Runway Gen-4等）。

### Two-Pass Pipeline

#### Pass 1: Counterfactual Trajectory Synthesis

$$\hat{\mathbf{V}}_{p1} = \mathrm{VOID}(\mathbf{z}, \mathbf{V}, \mathbf{M}_q) \tag{2}$$

变量解释：
- $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Gaussian diffusion noise，从标准正态分布采样
- $\mathbf{V}$: input video
- $\mathbf{M}_q$: quadmask
- $\hat{\mathbf{V}}_{p1}$: Pass 1输出

Pass 1会capture broadly correct motion hypothesis（free-fall、trajectory continuation），但会有**structural deformation**问题。这是lightweight video diffusion models的通病 [4, 28]。

#### 为什么会有deformation？

CogVideoX这类5B模型在motion-heavy generation时很难maintain temporal coherence [4, 28]。传统inpainting里input video提供强constraint（surface motion和geometry不变），但在counterfactual场景下需要**generate new motion**，导致：
- Bending
- Stretching  
- Structural drift

类比：就像没有motion guidance的CogVideoX image-to-video model一样artifacts。

#### Pass 2: Flow-Warped Noise Stabilization

基于**Go-with-the-Flow** [2]的insight：使用temporally correlated noise based on predicted motion trajectories能encourage diffusion model沿这些trajectories denoise consistently。

$$\hat{\mathbf{V}} = \mathrm{VOID}_{warp}(\mathbf{z}_{warp}, \mathbf{V}, \mathbf{M}_q) \tag{3}$$

变量解释：
- $\mathbf{z}_{warp}$: 从Pass 1 optical flow field warp过的noise
- $\mathrm{VOID}_{warp}$: 用flow-aligned noise训练的VOID variant
- $\mathbf{V}$, $\mathbf{M}_q$: 与Pass 1相同

**Pass 2的触发是conditional的** —— VLM判断是否需要substantial dynamic reconfiguration（free-fall、trajectory change），只在需要时触发。

#### 为什么warped noise有用？直觉解释

Standard diffusion的 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 在每个pixel独立采样。如果motion field告诉我们"frame t的pixel p对应frame t+1的pixel p'"，那么noise也应该沿这个trajectory保持correlated。这样denoising过程会"沿着"object motion走，避免object在generation过程中被"撕裂"。

形式上，warped noise定义大致是：
$$\mathbf{z}_{warp}^{(t+1)}(p') = \mathbf{z}_{warp}^{(t)}(p) \quad \text{where } p' = p + \text{flow}(p, t)$$

---

## 6. Inference-time VLM-Guided Quadmask Generation

这是paper的另一个亮点。Training时有GT quadmask，但inference时只有user click + SAM mask，怎么生成quadmask？

### Pipeline

1. **User input**: sparse clicks → SAM 2 [32] → binary mask $\mathbf{M}_o$
2. **VLM scene analysis**: input $\mathbf{V}$ + $\mathbf{M}_o$ → VLM produces list of affected objects
3. **SAM 3** [3]: list → mask $\mathbf{M}_a^{orig}$
4. **Counterfactual position prediction**: VLM predicts where affected objects go in counterfactual scenario，通过overlay coarse spatial grid → block-structured mask $\mathbf{M}_a^{count}$
5. **Combine**: $\mathbf{M}_a := \mathbf{M}_a^{orig} \vee \mathbf{M}_a^{count}$
6. **Construct quadmask**:
   - Black: $p \in \mathbf{M}_o \setminus \mathbf{M}_a$
   - Dark grey: $p \in \mathbf{M}_o \cap \mathbf{M}_a$
   - Light grey: $p \in \mathbf{M}_a \setminus \mathbf{M}_o$
   - White: otherwise

### VLM选择
主paper用**Gemini 3 Pro**作为VLM。Appendix Table 5比较了三种VLM：
- Qwen3-32B: Total 23.91
- GPT 5.2: Total 24.34
- Gemini 3-Pro: Total 26.12

Gemini 3-Pro最好，尤其在Int.Phys (3.66 vs 3.49-3.75)和Obj.Rem (4.82)上优势明显。

### 为什么要VLM？

VLM提供了**high-level causal reasoning**和**world knowledge** —— 这是diffusion model本身缺乏的。例如，paper提到VOID能generalize到unseen effects：
- 气球在holder被删除后浮起（训练数据中没有floating objects）
- 搅拌机在pressing的人被删除后不启动（训练数据中没有blenders）

这种generalization只能来自VLM + diffusion base model的world knowledge，而不是recall训练数据。

---

## 7. 实验结果

### 7.1 Human Preference Study (Table 1)

25 participants, 每个5 scenarios, 共125 comparisons。七种model对比：

| Model | Win % |
|-------|-------|
| **VOID (ours)** | **64.8** |
| Runway | 18.4 |
| Gen-Omnimatte | 11.2 |
| DiffuEraser | 4.0 |
| ROSE | 1.6 |
| MiniMax-Remover | 0.0 |
| ProPainter | 0.0 |

Runway作为text-guided editor，**已经被额外告知了counterfactual expected scene evolution**（例如"remove the person and ensure the held object falls naturally"），但仍然只有18.4%。这说明问题不在于language grounding，而在于reasoning + generation的整合。

### 7.2 VLM-as-Judge (Table 2)

三个VLM judges：Gemini 3 Pro, GPT 5.2, Qwen 3.5-32B

六个dimensions (0-5 each, total 30)：
1. Interaction & Physics
2. Object Removal  
3. Background & Artifacts
4. Temporal Consistency
5. Preservation
6. Sharpness

VOID在所有三个judge下都总分第一。最consistent的优势在**Interaction & Physics**：
- Gemini Pro judge: VOID 3.66 vs Runway 2.61 vs Gen-Omni 2.30
- GPT 5.2 judge: VOID 3.19 vs Runway 1.85 vs Gen-Omni 1.39
- Qwen 3.5-32B judge: VOID 2.64 vs DiffuEraser 2.19 vs Gen-Omni 2.19

注意Qwen judge下，ranking略有变化，Runway排到第四。但VOID始终第一。

### 7.3 Synthetic Benchmark (Table 3)

40个test videos (10 classic + 30 dynamic counterfactual)

| Model | PSNR↑ | LPIPS↓ | DreamSim↓ | DINOv2↑ | FVD↓ | VLM-Judge↑ |
|-------|-------|--------|-----------|---------|------|------------|
| **Ours** | **31.49** | 0.12 | **0.07** | **0.92** | **260.31** | **25.10** |
| MiniMax-Remover | 29.96 | **0.11** | 0.09 | 0.91 | 448.43 | 22.83 |
| ProPainter | 30.48 | **0.11** | 0.10 | 0.89 | 471.13 | 21.38 |
| Gen-Omnimatte | 29.44 | 0.12 | 0.12 | 0.87 | 437.88 | 20.40 |
| Runway | 26.68 | **0.11** | 0.15 | 0.85 | 442.76 | 21.67 |

VOID在**FVD**（video-level metric）上大幅领先：260.31 vs 第二437.88 (Gen-Omnimatte)。FVD衡量video distribution distance，最能反映counterfactual合理性。

**LPIPS**唯一不如baselines。Paper解释得很好：LPIPS对local translations敏感，可能penalize"effect发生在slightly wrong region"。例如，正确让stick掉下但速度稍微不对，比直接删掉stick的LPIPS更差。Appendix Figure 9专门展示了这个failure mode of LPIPS。

### 7.4 Ablation (Table 4)

75个real-world test cases, Gemini 3 Pro judge。

**Data composition** (都1200 samples):
- Kubric-Only: 20.36
- HUMOTO-Only: 20.12
- Both Datasets: 21.93

**Diversity matters** —— 两种数据混合比单独用任何一种都好，即使total size相同。Kubric教rigid body physics，HUMOTO教articulated manipulation，互补。

**Masking strategy**:
- Gen-Omni Mask (Full data): 23.39
- VOID (Full data): 26.12

详细的quadmask + VLM生成pipeline贡献了**+2.73**总分，其中Int.Phys从3.30→3.66 (+0.36)，Bg.Art从3.04→4.10 (+1.06)。

### 7.5 Second-Pass Analysis (Table 6)

VLM在75个case中标记了10个需要Pass 2 refinement：

| Pass | Int.Phys | Obj.Rem | Bg.Art | Temp | Pres | Sharp | Total |
|------|----------|---------|--------|------|------|-------|-------|
| Pass 1 | 2.90 | 4.20 | 3.70 | 3.80 | 4.90 | 4.00 | 23.5 |
| Pass 2 | 3.90 | 4.90 | 4.00 | 4.20 | 4.80 | 4.20 | 26.0 |

Pass 2在"难case"上贡献了**+2.5**总分，Int.Phys从2.90跃升到3.90 (+1.0)，这证实了deformation是Pass 1的主要failure mode，warp noise有效解决。

---

## 8. 与Related Work的关系网络

### Video Decomposition系列
- **Omnimatte** [24] (CVPR 2021): self-supervised layer decomposition，但reconstructive，不能synthesize new content
- **OmnimatteRF** [22]: 用3D radiance fields表示static background
- **Generative Omnimatte** [19]: 集成video inpainting prior + trimask，VOID的direct predecessor
- **OmnimatteZero** [34]: training-free via attention maps

VOID继承了Gen-Omnimatte的trimask思路，但扩展到quadmask并改造成counterfactual generation。

### Video Inpainting
- **ProPainter** [44]: dual-domain propagation (image + feature)
- **DiffuEraser** [20]: flow-based pixel propagation + transformer
- **AVID** [43] / **FDM** [9]: sampling pipelines for longer videos
- **MiniMax-Remover** [45]: efficient architecture + distillation
- **ROSE** [26]: photometric effects removal (shadows, reflections, light, translucency)
- **Object-Wiper** [17]: training-free photometric effects

这些都是**pixel-perfect but physically implausible**的代表。VOID的contribution是在他们之上加了causal reasoning layer。

### VLM-augmented video editing
- **Video-Repair** [18]: misalignment evaluation + localized refinement
- **Veggie** [41]: grounded generation for video concepts
- **LangDriveCtrl** [13]: driving scene specific

这些都是narrow domain或simple reasoning task。VOID是**first** to apply VLM reasoning for complex counterfactual video editing。

### Motion-aware video generation
- **Go-with-the-Flow** [2]: warped noise based on motion trajectories
- **VideoJam** [4]: joint appearance-motion representations
- **Track4Gen** [14]: point tracking supervision
- **Diffusion as Shader** [11]: 3D-aware video diffusion control
- **Intragen** [23]: trajectory-controlled object interactions

VOID借鉴了Go-with-the-Flow的warped noise思路，但adapted到object removal context。

### Foundation Video Models
- **Veo 3** [8]: Google DeepMind
- **Runway Gen-4** [33]: closed-source commercial
- **WAN** [37], **VACE** [15], **CogVideo** [40], **LTX-2** [12]: open-source

VOID基于CogVideoX，但framework原则上可以transfer到任何video diffusion backbone。

### VLM reasoning & physics
- **Physbench** [6]: benchmarking VLM for physical understanding
- **TraVL** [27]: 作者自己的prior work，VLM as physics implausibility judge
- **MLLM-as-a-judge** [5], **LLaVA-Critic** [38]: judge framework
- **Do generative video models understand physical principles?** [28]: 作者团队的prior analysis

TraVL和[28]是同一作者group的工作，VOID是这条research line的continuation。

---

## 9. Intuition: 为什么这个framework能work

我想强调几个deep insights：

### Insight 1: 分离Reasoning和Generation
VLM做high-level causal reasoning（什么会受影响，affected objects会去哪），diffusion model做low-level generation。这个separation of concerns是关键 —— diffusion model本身不擅长causal reasoning，但擅长visual synthesis；VLM相反。

### Insight 2: Counterfactual supervision解锁了新能力
传统video editing data是unstructured web data，model学到的是statistical co-occurrence。Counterfactual pairs通过physics simulator提供了**true causal supervision**：
$$P(\hat{\mathbf{V}} | \text{do}(O = \emptyset), \mathbf{V})$$
而不是
$$P(\hat{\mathbf{V}} | \text{observe } O = \emptyset)$$

这是Pearl的do-calculus思想在video generation中的应用。

### Insight 3: Quadmask是compressed representation of causal graph
Quadmask本质上是把causal relationship encode成pixel-space guidance：
- Black: $\text{cause} \setminus \text{effect}$ (要删的部分)
- Dark grey: $\text{cause} \cap \text{effect}$ (删掉后会触发新effect)
- Light grey: $\text{effect} \setminus \text{cause}$ (新的effect location)
- White: $\text{neither}$ (preserve)

这个mask就是一个**local causal graph**的visual encoding。

### Insight 4: Two-pass对应"先想清楚再画"
Pass 1是rough hypothesis（"object会掉"），Pass 2是refinement（"保证object不变形"）。这种coarse-to-fine的reasoning-then-render pattern在human cognition中也存在 —— 我们先predict会发生什么，再visualize细节。

### Insight 5: Generalization来自inductive bias
VOID在unseen effects上（气球浮起、blender不启动）的generalization，说明真正的物理reasoning发生了，而不是memorization。这是VLM world knowledge和diffusion model visual priors的emergent composition。

---

## 10. Limitations & Future Work

Paper提到的：
1. **Camera angles**: unusual angle或太close时会degrade
2. **Video length**: 仍然只有几秒
3. **Resolution**: 可以improve

我能想到的更多limitation和可能的extension：

1. **Multi-object counterfactuals**: 删多个objects之间的interaction怎么处理？paper例子都是single object removal，但实际场景中causal chain可能更长
2. **Human intent ambiguity**: "删除这个person"可能意味着"假装这个人从未存在"或"突然消失"，counterfactual semantics不同
3. **Real-time interaction**: VLM-based mask generation是slow的，interactive editing需要acceleration
4. **Backbone scaling**: 现在用CogVideoX 5B，transfer到Veo 3或Sora-like 100B+ models是否能保持causal reasoning？
5. **Counterfactual faithfulness evaluation**: 真实video没有GT counterfactual，human study和VLM-judge都有biases

---

## 11. 我对这篇paper的take

Andrej，从build intuition的角度，这篇paper最重要的思想是：

**Video editing = world simulation in disguise**

当你删除一个object，你不是在做inpainting，你是在问"如果这个object从未存在，world state会怎样演化？"。这个问题本质上需要一个internal world model。

作者很clever地把这个抽象问题**decompose**成三个可工程化的部分：
1. **Data**: physics simulator生成counterfactual pairs
2. **Conditioning**: quadmask表达local causal structure
3. **Reasoning**: VLM做causal analysis，diffusion model做synthesis

这个framework其实指向了一个更大的vision —— **video diffusion models should be world models**，而不仅仅是pixel generators。PhysBench [6]和作者团队的TraVL [27]、[28]工作都是在推进这个agenda。

我特别感兴趣的是：如果把这个framework scale up到Veo 3级别，combined with reinforcement learning on counterfactual predictions（类似AlphaGo的self-play），我们是否能得到一个真正能做counterfactual physical reasoning的video foundation model？

Reference links:
- Project page: https://void-model.github.io
- CogVideoX: https://arxiv.org/abs/2408.06072
- Kubric: https://github.com/google-research/kubric
- Omnimatte (original): https://omnimatte.github.io
- Generative Omnimatte: https://arxiv.org/abs/2411.16683
- Go-with-the-Flow: https://go-with-the-flow.github.io
- SAM 2: https://arxiv.org/abs/2408.00714
- SAM 3: https://arxiv.org/abs/2511.16719
- PhysBench: https://arxiv.org/abs/2501.16411
- TraVL (作者prior work): https://arxiv.org/abs/2510.07550
- Do gen video models understand physics? (作者prior): https://arxiv.org/abs/2501.09038
- HUMOTO: https://humoto.github.io
- ProPainter: https://shangchenzhou.github.io/projects/ProPainter/
- DiffuEraser: https://arxiv.org/abs/2501.10018
- ROSE: https://arxiv.org/abs/2508.18633
- MiniMax-Remover: https://arxiv.org/abs/2505.24873
- Veo 3 Tech Report: https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf
- Runway Gen-4: https://runwayml.com
- DreamSim: https://dreamsim-nights.github.io
- DINOv2: https://arxiv.org/abs/2304.07193
- FVD: https://arxiv.org/abs/1812.01717
- Wan: https://arxiv.org/abs/2503.20314
- VACE: https://arxiv.org/abs/2503.07598
- LTX-2: https://arxiv.org/abs/2601.03233
