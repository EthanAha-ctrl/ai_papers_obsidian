---
source_pdf: ReconVLA.pdf
paper_sha256: de568f5eb3e26a355160e7b79c286327eae38f3585a419a51a061c186e55f011
processed_at: '2026-08-11T21:54:33-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 ReconVLA

## 一句话总结

**现在的机器人视觉太"涣散"了，看哪儿都不聚焦，导致抓错东西。ReconVLA的训练方法就是逼着模型学会"盯着目标看"。**

---

## 问题出在哪

想象你教一个小孩去桌上拿水杯。正常小孩眼睛会盯着水杯，伸手过去拿。但现在这些VLA模型（不管是OpenVLA还是RT-2），它们的**注意力是散开的**——水杯看一眼，旁边的书本看一眼，墙上的花纹也看一眼。

结果就是：模型可能去抓旁边的书，或者手臂伸到了水杯和书本之间的空气里。

作者画了attention map（Figure 4第一行），肉眼可见baseline的attention像泼了一盆水洒在整个画面上，哪哪儿都有，哪哪儿都不强。

---

## 别人怎么解决，为什么不行

### 方法一：找个帮手帮你框出来（Explicit Grounding）

拿一个现成的物体检测器（比如YOLOv11），先把水杯框出来，crop放大，和原图一起塞给VLA。

**问题**：你只是给模型喂了更多信息，但模型本身的"眼力"没变。下次换个场景，没有检测器帮忙，它还是抓不准。而且两张图一起输入，信息冗余，模型反而困惑。

### 方法二：让模型自己说出坐标再行动（Chain-of-Thought Grounding）

让模型先输出 `[x1, x2, y1, y2]` 这种bounding box坐标，像写推理过程一样，然后再输出action。

听起来很美，实际上**灾难性失败**。Table 1里这个方法的5/5成功率是**0.0%**，连baseline都不如。

为什么？因为LLM输出离散token，让它同时学"精确坐标"和"连续动作值"这两种完全不同性质的distribution，相互干扰。就像让一个人同时用左手写字用右手画画，两只手都抖。

### 方法三（ReconVLA）：让模型"复述"它看到了什么

这是这篇paper的核心思路，也是我觉得最elegant的地方。

---

## ReconVLA的核心idea

**别让模型说坐标，也别给它crop好的图。直接问它："你能把目标区域重建出来吗？"**

具体做法：

1. 拿一张图，比如桌上有个水杯，instruction是"pick up the watermelon"
2. 用Grounding DINO先把watermelon那块小区域crop出来，这就是**gaze region**（人眼中央凹盯住的那块清晰区域）
3. 把gaze region过一遍frozen VAE，变成latent tokens $\mathbf{z}_0$
4. 给LLM的visual outputs加一个diffusion decoder，让它从noise出发，conditioned on自己的visual representation，把 $\mathbf{z}_0$ 重建出来

如果LLM的visual representation里没编码"watermelon长什么样"的信息，diffusion decoder就重建不出来。Gradient回传，强迫LLM学会focus attention到watermelon上。

### 用比喻说

像训练侦探的观察力。方法一是给他个放大镜（external detector），方法二是让他报告嫌疑人坐标（CoT），方法三是让他**默写嫌疑人长相**——你不用管他怎么观察，但如果他能把嫌疑人face sketch出来，说明他真的看见了。

---

## 几个关键技术细节

### 1. 为什么用VAE tokenize，不直接reconstruct RGB像素？

RGB像素太低级，模型可以通过统计pattern cheat（比如background的平均颜色），不需要真正理解gaze region的内容。VAE latent space保留了spatial structure但压缩了redundancy，逼模型学真正有意义的feature。

这也呼应了LeCun的[JEPA](https://arxiv.org/abs/2301.08243)思想：在latent space做prediction，不在pixel space做。

### 2. 为什么reconstruct gaze region而不是整张图？

Table 2的ablation很说明问题：

| 重建目标 | 5/5成功率 |
|---------|----------|
| 不重建（baseline） | 49.0% |
| 重建整张图 | 46.5%（反而更差！） |
| 重建gaze region | 64.1% |

重建整张图为什么更差？因为background占95%的pixel，gradient大部分浪费在学无关的background上。而且在unseen environment下，background distribution shift大，diffusion拟合不了，gradient变成噪声干扰action学习。

Gaze region的distribution compact、task-relevant，模型能学好。

### 3. 为什么instruction要放在image前面？

LLM是causal attention，后面的token看不到前面的。如果instruction在image后面，image tokens就看不到instruction，无法基于"pick up watermelon"来focus到watermelon上。

把instruction前置后，每个image token都能通过causal attention看到前面的instruction，相当于给visual encoder一个**language query**：你要根据这个instruction来决定看哪儿。

这就像人类听话做事：先听到"拿水杯"，然后眼睛才去找水杯。而不是先看完整个场景再听指令。

### 4. Pretraining为什么重要？

作者构建了**100k trajectories、2M samples**的pretraining dataset，用Grounding DINO自动标注gaze region。

Table 2显示pretraining带来5.9%的提升（58.2%→64.1%）。原因是在test-time unseen environment下，模型要面对没见过的物体和背景，如果没在大规模数据上学过"如何grounding + reconstruction"，generalization就上不去。

---

## 实验结果讲人话

### CALVIN ABC→D（在没见过的环境D上测试）

| 方法 | 5/5成功率 |
|------|----------|
| OpenVLA | 43.5% |
| UniVLA | 56.5% |
| GR-1 | 40.1% |
| **ReconVLA** | **64.1%** |

在unseen environment上ReconVLA完胜。因为它的perception能力被reconstruction supervision塑造好了，遇到新环境也能grounding。

### CALVIN ABCD→D（在见过的环境上测试）

| 方法 | 5/5成功率 |
|------|----------|
| GR-1 | 73.1% |
| **ReconVLA** | **70.5%** |

在seen environment上ReconVLA略逊GR-1。因为GR-1预测future images，在familiar environment下dynamics modeling有优势。但这trade-off值得：ReconVLA的perception-first哲学在distribution shift下更robust。

### Real-world实验

四个任务：Stack bowls, Put fruit into bowl, Flip cups, Bus table。

ReconVLA基本都90%+。**Unseen objects上OpenVLA和PD-VLA几乎全失败**（0%成功率），ReconVLA还能成功。这说明pretraining带来的visual generalization是真的，不是paper上的数字游戏。

---

## 这篇paper的"美"在哪

1. **Architecturally simple**：只在LLaVA上加一个diffusion transformer作为auxiliary head，inference时可以skip
2. **Conceptually elegant**：用gaze region重建模拟人眼foveation机制，biologically plausible
3. **No external detector at inference**：Grounding DINO只在data preparation阶段用，inference时model自己grounding
4. **Empirically strong**：CALVIN long-horizon和real-world unseen objects都SOTA

---

## 给你的intuition

Karpathy你应该会喜欢这个angle：

ReconVLA本质上是把**"prediction as representation learning"**这个哲学apply到VLA的perception上。你之前在[makemore](https://github.com/karpathy/makemore)和[zero to hero](https://karpathy.ai/zero-to-hero.html)里反复讲的"next token prediction学到了representation"的思想，在这里换了个形式：**reconstruction prediction学到了visual grounding**。

不用显式标注bounding box，不用显式输出坐标，就让模型重建目标区域的latent representation，attention自然就聚焦了。这和自监督学习的精髓一脉相承——**让模型自己发现什么信息重要**。

更深的联系：这和[DeepMind的BYOL](https://arxiv.org/abs/2006.07733)、[FAIR的DINO](https://arxiv.org/abs/2104.14294)系列都是同一个family——不靠contrastive，不靠explicit labels，靠prediction task本身塑造representation。ReconVLA把这个idea搬到了robot manipulation的visual grounding问题上。

References:
- [ReconVLA Project](https://zionchow.github.io/ReconVLA/)
- [JEPA](https://arxiv.org/abs/2301.08243)
- [MAE](https://arxiv.org/abs/2111.06377)
- [BYOL](https://arxiv.org/abs/2006.07733)
- [DINO](https://arxiv.org/abs/2104.14294)
- [Latent Diffusion](https://arxiv.org/abs/2112.10752)
- [Karpathy's makemore](https://github.com/karpathy/makemore)

---

# ReconVLA 论文深度讲解

## 1. 核心问题与Motivation

这篇paper要解决的核心问题是：当前VLA模型的**visual attention是dispersed的**，无法聚焦到target object上。这导致robot抓错东西。作者通过可视化attention map（Figure 4 Row 1）发现baseline VLA的attention分布在整个image上，包括irrelevant的distractors。

这个insight非常直观——想象你伸手抓水杯时，眼睛应该聚焦在水杯上，而baseline模型像是在看整个房间的模糊倒影。作者把这个mechanism类比为人眼的**fovea（中央凹）机制**：人眼在fovea区域sharp clarity，周边blurred，这种"gaze region"的机制让人类能precise manipulation。

Reference: 人眼fovea机制的review [Journal of Vision 2020](https://jov.arvojournals.org/article.aspx?articleid=2772035)

---

## 2. 三种Grounding Paradigm的对比

作者把现有的visual grounding方法归为三类，这是这篇paper最有价值的conceptual contribution：

### (a) Explicit Grounding (EG)
代表：[RoboGround](https://arxiv.org/abs/2503.04636)、[VIP](https://arxiv.org/abs/2412.01000)

用external detector（如LISA、YOLOv11）检测target object，crop出来后和原图一起输入VLA。缺点是**依赖外部专家模型**，VLA本身的grounding能力没有提升，而且visual information redundancy（重复输入）。

### (b) Chain-of-Thought Grounding (CG)
代表：[ECoT](https://arxiv.org/abs/2407.08693)、[GraspVLA](https://arxiv.org/abs/2505.03233)

让VLA先输出bounding box坐标 `[x1 x2 y1 y2]`，再输出action。这是CoT风格。问题非常严重：Table 1显示CG的5/5成功率只有**0.0%**，avg length 0.63，比baseline还差很多。原因是**discrete token空间下同时输出精确坐标和action values非常困难**——LLM要同时学两种不同semantics的distribution，coordinate tokens污染action tokens的训练。

### (c) Implicit Grounding (IG) — ReconVLA
直接supervise visual outputs，通过reconstructive tokens $\mathbf{h}_R$去condition一个diffusion transformer重建gaze region。这种**auxiliary visual supervision**强迫LLM的visual outputs编码region-specific的fine-grained information，从而implicit地实现grounding。

---

## 3. 架构详解（Figure 3解析）

### 3.1 整体pipeline

ReconVLA基于[LLaVA-7b](https://llava-vl.github.io/)，其中：
- LLM backbone: [Qwen2-7b](https://arxiv.org/abs/2407.10671)
- Vision encoder: [siglip-so400m-patch14-384](https://arxiv.org/abs/2303.15343)
- Visual tokenizer $\mathcal{F}$: frozen VAE（来自[Latent Diffusion / Stable Diffusion](https://arxiv.org/abs/2112.10752)）
- Denoiser $\mathcal{D}$: Transformer encoder blocks（with self-attention）

### 3.2 标准VLA的形式化

输入image $I$ 和text instruction $S$，VLA预测action $\mathcal{A}$：

$$\mathcal{A} = \mathcal{Q}(\text{LLM}(\mathbf{h}_I, \mathbf{h}_S)) = \mathcal{Q}(\text{LLM}(\mathcal{E}(I), \mathcal{T}(S)))$$

变量含义：
- $\mathcal{A}$: 最终executable action（7-DoF，包含末端位姿、gripper开合等）
- $\mathcal{Q}$: action detokenizer，把discrete action tokens映射回continuous action
- $\mathcal{E}$: vision encoder（siglip）
- $\mathcal{T}$: text tokenizer
- $\mathbf{h}_I = \mathcal{E}(I)$: image tokens
- $\mathbf{h}_S = \mathcal{T}(S)$: text tokens

Autoregressive generation：
$$p(\mathbf{a}) = \prod_{i=1}^{N} p_{\text{LLM}}(\mathbf{a}_i \mid \mathbf{a}_{1\sim i-1}; \mathbf{h}_I; \mathbf{h}_S)$$

- $\mathbf{a}_i$: 第$i$个action token
- $N$: action token总数（通常7个action dimensions各bin化后约7-256 tokens）

### 3.3 Reconstruction部分的关键设计

这里有几个**non-obvious的设计选择**值得深挖：

**Design 1: 用frozen VAE tokenize gaze region**

gaze region image $I'$ 通过frozen VAE $\mathcal{F}$ 编码成latent tokens：
$$\mathbf{z}_0 = \mathcal{F}(I')$$

为什么用VAE而不是直接用pixel或者用siglip features？我推测原因有三：
1. **Pixel-level reconstruction太easy也太难**：太easy是说LLM可以cheat通过低级statistical pattern；太难是说reconstruct RGB values需要exponential capacity
2. **SigLIP features语义太high-level**：重建siglip features会丢失fine-grained spatial信息，达不到"逼迫model注意target region细节"的目的
3. **VAE latent space的inductive bias**：VAE的latent保留了spatial structure但压缩了redundancy，这个[latent diffusion](https://arxiv.org/abs/2112.10752)的insight直接迁移过来

**Design 2: Reconstruction loss用标准DDPM形式**

$$\mathcal{L}_{\text{VLA}}^{\text{visual}}(\mathbf{h}_R, I') = \mathbb{E}_{t,\epsilon}\left[||\mathcal{D}(\mathbf{z}_t; \mathbf{h}_R, t) - \epsilon||^2\right]$$

变量含义：
- $t$: diffusion timestep，从$\{1, ..., T\}$均匀采样
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 注入的Gaussian noise
- $\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\mathbf{z}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$: forward process加噪后的tokens，其中$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$是cumulative noise schedule
- $\mathbf{h}_R = \text{LLM}(\mathbf{h}_I)$: reconstructive tokens，是LLM输出的visual tokens（不是action tokens！）
- $\mathcal{D}$: denoiser，Transformer encoder stack

**关键insight**：$\mathbf{h}_R$作为condition输入denoiser，等价于把LLM的visual representation作为diffusion的conditioning signal。如果$\mathbf{h}_R$没有编码gaze region的fine-grained信息，denoiser就没办法denoise出来correct的$\mathbf{z}_0$。这个gradient回传到LLM，强迫visual attention集中在target上。

**Design 3: Instruction tokens prepended before image tokens**

为了让image tokens能attend to instruction tokens（"put the watermelon into the yellow bowl"），作者把instruction tokens prepend到image tokens前面，用causal attention让image tokens fuse prefix text的信息。这点很关键——否则image tokens是先于instruction tokens出现的，causal mask下image tokens看不到后续的text，grounding就失败了。

这个设计有点像[Flamingo](https://arxiv.org/abs/2204.14198)的interleaved format，但目的是**让visual representation conditional on language**，这是grounding的本质。

**Design 4: Reconstruction和action supervision同时backprop**

总loss：
$$\mathcal{L}_{\text{ReconVLA}} = \mathcal{L}_{\text{VLA}}^{\text{action}} + \mathcal{L}_{\text{VLA}}^{\text{visual}}$$

其中$\mathcal{L}_{\text{VLA}}^{\text{action}}$是cross-entropy loss on action tokens。两个loss同时优化，reconstruction的gradient通过$\mathbf{h}_R$传回visual encoder和LLM，间接影响action generation。

---

## 4. 大规模Pretraining Dataset的构建

这是另一个key contribution，作者构建了**100k+ trajectories, 2M samples**的pretraining dataset。

数据来源：
- [BridgeData V2](https://arxiv.org/abs/2308.12952): 真实世界robot data
- [LIBERO](https://arxiv.org/abs/2306.03310): simulation，knowledge transfer benchmark
- [CALVIN](https://arxiv.org/abs/2112.03227): long-horizon manipulation benchmark

数据pipeline：
1. 用[Grounding DINO](https://arxiv.org/abs/2403.05699)（fine-tuned）自动检测每个frame中的target object
2. crop出gaze region image
3. 整理成(original image, gaze region image) pair

这里有个**值得注意的细节**：作者fine-tune Grounding DINO，说明off-the-shelf Grounding DINO在robot数据上的检测精度不够。这也解释了为什么pretraining对generalization如此重要——Table 2 ablation显示，去掉pretraining后5/5成功率从64.1%降到58.2%，掉5.9%。这个gap主要来自test-time unseen environment下gaze region grounding的generalization挑战。

---

## 5. 实验结果深度解读

### 5.1 Paradigm Comparison (Table 1, CALVIN ABC→D)

| Paradigm | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg.Len |
|----------|-----|-----|-----|-----|-----|---------|
| Baseline | 88.8 | 76.1 | 63.7 | 57.0 | 49.0 | 3.36 |
| EG | 94.4 | 82.5 | 70.9 | 62.2 | 50.2 | 3.61 |
| CG | 47.0 | 14.3 | 1.6 | 0.0 | 0.0 | 0.63 |
| **IG (ReconVLA)** | **95.6** | **87.6** | **76.9** | **69.3** | **64.1** | **3.95** |

几个observation：
1. **CG的catastrophic failure**：5/5=0.0%，avg len=0.63，比baseline差极多。这印证了让VLA autoregressively输出精确coordinates和action values的training difficulty。这是为什么ECoT、GraspVLA那种approach在long-horizon上挣扎。
2. **EG的提升有限**：EG 5/5=50.2%，只比baseline高1.2%。原因是crop+resize引入visual redundancy，而且external detector的error propagate进来。
3. **IG的dramatic improvement**：5/5从49.0%→64.1%，提升15.1%。这说明implicit supervision机制确实让model学会了fine-grained grounding。

### 5.2 Ablation Study (Table 2)

| Recon. | Gaze Region | Pretrain | 5/5 | Avg.Len |
|--------|-------------|----------|-----|---------|
| ✓ | ✓ | ✓ | 64.1 | 3.95 |
| ✓ | ✓ | ✗ | 58.2 | 3.85 |
| ✓ | ✗ | ✗ | 46.5 | 3.42 |
| ✗ | ✗ | ✗ | 49.0 | 3.36 |

关键insights：
- **Reconstruct whole image vs gaze region**： reconstruct whole image的5/5=46.5%，反而比baseline（49.0%）还低！这说明reconstruct整个image在unseen scene下太困难，diffusion model拟合不了distribution，gradient噪声反而干扰了action learning。只有reconstruct gaze region才能让supervision signal focused。
- **Pretraining的marginal gain**：pretrain带来5.9%提升，主要在unseen environment下的generalization。

### 5.3 与SOTA对比 (Table 3 & 4)

CALVIN ABC→D：
- ReconVLA: 64.1% (5/5), 3.95
- [UniVLA](https://arxiv.org/abs/2505.06111): 56.5%, 3.80
- [OpenVLA](https://openvla.github.io/): 43.5%, 3.27
- [GR-1](https://arxiv.org/abs/2402.06543): 40.1%, 3.06

CALVIN ABCD→D：
- ReconVLA: 70.5%, 4.23
- GR-1: 73.1%, 4.21
- RoboFlamingo: 66.0%, 4.08

注意到**ABCD→D上ReconVLA略逊GR-1**，这很有意思：GR-1是generative method预测future images，在seen environment（ABCD→D是in-distribution）上dynamics modeling有优势；但**ABC→D（unseen environment D）上ReconVLA完胜**，说明perception-focused的approach对distribution shift更robust。

### 5.4 Real-world Experiments (Figure 6)

四个任务：Stack bowls, Put fruit into bowl, Flip cups, Bus table。ReconVLA在real-world task上success rate接近或超过90%。**Unseen tasks上OpenVLA和PD-VLA几乎0%**，而ReconVLA仍能成功，证明pretraining带来的visual generalization。

---

## 6. 与相关工作的深度联系

### 6.1 与[Reconstructive Visual Instruction Tuning](https://arxiv.org/abs/2410.09575) (Wang et al. 2024)
这篇是ReconVLA的直接inspiration来源。那个工作在VLM上加reconstruction task，让VLM学习fine-grained visual representation。ReconVLA把这个idea迁移到VLA，并且target region用gaze region而不是whole image。

### 6.2 与[GR-1](https://arxiv.org/abs/2402.06543)的对比
GR-1也是generative VLA，但predict **future** images，目的是学dynamics。ReconVLA reconstruct **current** gaze region，目的是学perception/grounding。这是两种不同的philosophy：
- GR-1: "我能不能imagine未来" → planning capability
- ReconVLA: "我现在该看哪儿" → perception capability

Table 4显示在ABCD→D上GR-1略胜（73.1% vs 70.5%），但ABC→D上ReconVLA完胜（64.1% vs 40.1%），印证了这个分析。

### 6.3 与[3D-VLA](https://arxiv.org/abs/2403.09631)的对比
3D-VLA引入depth information做grounding。Table 4上3D-VLA的5/5只有0%，远低于ReconVLA。推测原因：3D信息需要explicit depth input，在long-horizon task中depth estimation误差累积。

### 6.4 与[JEPA](https://arxiv.org/abs/2301.08243)的联系
Yann LeCun的JEPA思想：在latent space做prediction而不是pixel space。ReconVLA在VAE latent space做reconstruction，本质上也是JEPA-style的latent prediction。这是为什么用frozen VAE而不是直接pixel reconstruction。

### 6.5 与[Diffusion Policy](https://arxiv.org/abs/2303.04137)的区别
Diffusion Policy用diffusion生成action sequence，ReconVLA用diffusion做visual reconstruction的auxiliary task。两者方向相反：DP是diffusion for action，ReconVLA是diffusion for perception supervision。

---

## 7. 个人的intuition构建

让我尝试build一些更深的intuition：

### Intuition 1: 为什么implicit grounding比explicit grounding好？

Explicit grounding把crop过的image作为extra input，等价于告诉model"看这里"，但model的attention机制还是没变，下次遇到类似场景还是会disperse。Implicit grounding通过supervision signal塑造model的internal representation，让model主动学会"哪里该看"。

类比：explicit grounding像给学生划重点，implicit grounding像训练学生自己识别重点的能力。

### Intuition 2: 为什么gaze region比whole image好？

Reconstruct whole image有两个问题：
1. **Pixel redundancy**：background的pixel占了95%以上，gradient大部分浪费在学background上
2. **Distribution shift**：在unseen environment下background变化大，whole image distribution难拟合

Gaze region的distribution更compact、更task-relevant，diffusion model能更好建模。

### Intuition 3: Reconstruction的auxiliary supervision如何改善action？

通过**shared visual encoder和LLM**。Reconstruction的gradient流回visual encoder和LLM的early layers，改善了$\mathbf{h}_I$的quality。而action prediction也用$\mathbf{h}_I$作为输入，因此action quality也提升。这是**representation learning**的经典套路。

### Intuition 4: 为什么instruction prepend很重要？

LLM的causal attention是单向的，如果image tokens在text之前，image tokens就看不到text，无法基于instruction做grounding。把instruction前置后，每个image token都能通过causal attention看到前面的instruction，相当于给visual encoder一个**language-conditioned的"过滤"先验**。

---

## 8. 潜在Limitations与Future Directions

虽然paper没有详细讨论limitation，从技术细节可以推测：

1. **Grounding DINO的依赖**：gaze region标注质量取决于Grounding DINO的精度，可能引入label noise
2. **Inference cost**：训练时需要跑diffusion，inference时reconstruction branch是否还要跑？从architecture看inference时应该可以skip reconstruction branch
3. **Gaze region定义**：对于interactive task（如pour water），gaze region可能ambiguous——是water stream还是cup？paper没有详细讨论
4. **3D information缺失**：相比3D-VLA，没有depth信息，可能在occlusion-heavy场景受限
5. **Pretraining数据bias**：100k trajectories主要来自BridgeData V2（桌面场景），可能不cover legged locomotion等场景

### 可能的extension方向
1. **Video reconstruction**：不只是当前frame的gaze region，而是predict未来几帧的gaze region trajectory
2. **Multi-object gaze**：同时reconstruct多个candidate objects的gaze regions，让model自己选
3. **End-to-end gaze detection**：去掉Grounding DINO，让model自己从language instruction推断gaze region
4. **3D gaze**：结合depth，reconstruct 3D point cloud的gaze region

---

## 9. 总结

ReconVLA的核心contribution可以总结为：

> 通过把visual reconstruction作为auxiliary supervision，implicit地塑造VLA的visual attention allocation，实现precise manipulation。

这个idea的**beauty**在于：
1. **Architecturally simple**：只加一个diffusion transformer作为auxiliary head
2. **Conceptually elegant**：用gaze region的重建模拟人眼foveation机制
3. **Empirically strong**：在CALVIN long-horizon task上SOTA，real-world unseen object上generalize

对于你（Karpathy）关心的representation learning angle：这篇paper本质上是把[JEPA-style latent prediction](https://arxiv.org/abs/2301.08243)和[MAE-style reconstruction](https://arxiv.org/abs/2111.06377)的思想，apply到VLA的visual grounding问题上，用diffusion作为proxy task来shape visual representation。这和你之前在[dracc](https://github.com/dracc)或者[Spring](https://arxiv.org/abs/2305.16291)讨论的"prediction is the key to representation"的哲学是一脉相承的。

## References

- [ReconVLA Project Page](https://zionchow.github.io/ReconVLA/)
- [LLaVA](https://llava-vl.github.io/)
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [DDPM](https://arxiv.org/abs/2006.11239)
- [CALVIN Benchmark](https://calvinrobot.github.io/)
- [LIBERO](https://libero-project.github.io/)
- [BridgeData V2](https://arxiv.org/abs/2308.12952)
- [OpenVLA](https://openvla.github.io/)
- [GR-1](https://arxiv.org/abs/2402.06543)
- [UniVLA](https://arxiv.org/abs/2505.06111)
- [Grounding DINO](https://arxiv.org/abs/2403.05699)
- [RoboGround](https://arxiv.org/abs/2503.04636)
- [ECoT](https://arxiv.org/abs/2407.08693)
- [GraspVLA](https://arxiv.org/abs/2505.03233)
- [3D-VLA](https://arxiv.org/abs/2403.09631)
- [Reconstructive Visual Instruction Tuning](https://arxiv.org/abs/2410.09575)
- [JEPA](https://arxiv.org/abs/2301.08243)
- [MAE](https://arxiv.org/abs/2111.06377)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [OpenFlamingo](https://arxiv.org/abs/2308.01390)
