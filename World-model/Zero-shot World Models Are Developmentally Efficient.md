---
source_pdf: Zero-shot World Models Are Developmentally Efficient.pdf
paper_sha256: 122df68e9b2d1084fcc7c43edbd6718131a018c1ba061741c87934389c904aea
processed_at: '2026-08-13T06:51:41-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说ZWM

## 这篇paper到底在问什么？

一个很简单的问题：**小屁孩怎么学的这么快？**

你想想，一个两三岁的小孩，每天清醒时间也就10小时左右，他能看到的视频素材加起来可能就几百小时。但这个小孩已经会判断东西远近、追踪运动的物体、知道一个东西被挡住还在那儿、知道推一个球球会滚。

我们现在的AI呢？ImageNet喂1400万张图，V-JEPA2喂几万小时视频，DINOv3也是海量数据。喂完了想干个新活儿（比如分割），还得专门标一堆数据再训一个head。

**小孩用几百小时能干的事儿，AI用几万小时还干不到。这中间到底差了什么？**

这就是这篇paper想回答的问题。

---

## ZWM的三个核心idea

### Idea 1：训练时把"长啥样"和"怎么动"分开

先说最底层的训练。ZWM用的架构很普通——就是一个ViT，[Dosovitskiy 2021](https://arxiv.org/abs/2010.11929)那个。没什么花活。但训练方式很巧。

普通MAE怎么做？[He et al. 2021](https://arxiv.org/abs/2111.06377)。一张图随机mask掉75%的patches，让模型reconstruct。这是spatial masking。

ZWM改了一个关键东西：**temporal masking**。它用两帧视频，$f_1$和$f_2$，时间隔150到450毫秒。然后——

- $f_1$：完全可见，一个patch都不mask
- $f_2$：mask掉90%，只露出10%的patches

让模型从完整的$f_1$加上残缺的$f_2$去predict完整的$f_2$。

$$\Psi_{\Theta^*}: (f_1, f_2^{masked}) \mapsto \hat{f}_2; \quad \Theta^* := \arg\min_\Theta \mathbb{E} \|f_2 - \hat{f}_2\|^2$$

- $f_1$：第一帧，完整RGB，代表"过去"
- $f_2^{masked}$：第二帧，90%被盖住，只露出10%
- $\hat{f}_2$：模型预测的完整第二帧
- $\Theta^*$：最优参数
- $\|\cdot\|^2$：像素级L2 loss

**为什么这样做？**

你想啊，要从10%的sparse patches重建整个$f_2$，模型面临一个选择难题。它光从$f_2$那10%的patches根本看不出物体的appearance——一个像素patch告诉你这里是红色，但整个物体长啥样？不知道。那怎么办？

**只能去$f_1$里借appearance。** $f_1$是完整的嘛，物体长啥样$f_1$里全有。

那$f_2$里那10%的patches有什么用？**用来推断motion。** 因为$f_1$和$f_2$隔了几百毫秒，物体动了。那10%的patches告诉你"这个点从$f_1$到$f_2$跑哪儿去了"，模型就能推断出整个motion field。

所以网络被逼着学会两件事：
- 从$f_1$编码appearance（高维信息，texture、color、shape）
- 从$f_2$的sparse patches编码motion（低维信息，物体怎么动了）

这就是"temporally factored"的意思——**appearance和motion被天然地factorize成两个stream**。

作者做了ablation：如果把两帧都mask成45%-45%或90%-90%（symmetric），效果差很多。为什么？因为symmetric masking没给motion那一条窄通道，模型可以用各种取巧方式reconstruct，不需要真正学motion。

**intuition**：自然视频的motion是低维的（相机6-DoF + 刚体运动），appearance是高维的。asymmetric masking刚好match这个natural factorization。逼模型走窄路，它反而学对了。

参考[MAE原始paper](https://arxiv.org/abs/2111.06377)和[ZWM前作Bear et al. 2023](https://arxiv.org/abs/2306.01828)。

---

### Idea 2：训练完了怎么"问"模型？

这是这篇paper最clever的地方。

模型训练完了，它**隐式**学会了motion、appearance、objectness等等一堆东西。但这些knowledge都藏在网络参数里，你怎么把它"问"出来？

传统做法：给每个task收集labeled data，训一个linear probe或finetune一个head。但这在developmental上不make sense——小孩不会为每个新task去收集labels。

ZWM的方案：**做反事实实验**。

[Pearl的causal inference](https://www.cambridge.org/core/books/causality/9407819B7E7A4D6F2A4C9F4F4B4D4F4D)思想：想知道X对Y有没有causal影响？改一下X看Y变不变。

形式化：

$$x_\delta := \text{perturb}(x); \quad \delta\Psi := \text{compare}(\Psi(x), \Psi(x_\delta)); \quad \text{output} := \text{aggregate}(\delta\Psi)$$

- $x$：原始输入
- $x_\delta$：perturbed后的输入
- $\Psi(x), \Psi(x_\delta)$：模型两次forward的输出
- $\delta\Psi$：两次输出的difference
- output：aggregation后得到的visual quantity

**具体例子：optical flow怎么做？**

假设我要track一个点$x_q$在$f_1$到$f_2$之间怎么动了。

1. **Perturb**：复制$f_1$变成$\tilde{f}_1$，在$x_q$位置画一个白色Gaussian dot（amplitude 255, $\sigma=3$ pixels）。这个dot就像一个tracer。
2. **Compare**：用同一个$f_2^{masked}$，forward两次：
   - clean：$(f_1, f_2^{masked}) \to \hat{f}_2$
   - perturbed：$(\tilde{f}_1, f_2^{masked}) \to \tilde{f}_2^{pred}$
   - 算difference：$\Delta = \tilde{f}_2^{pred} - \hat{f}_2$
3. **Aggregate**：$\text{flow}(x_q) = \arg\max(|\Delta|) - x_q$

**intuition**：你给$f_1$加了个白点tracer，predictor在预测$f_2$的时候，会沿着motion的因果链把这个tracer"搬运"到$f_2$中对应的位置。$\arg\max(|\Delta|)$告诉你tracer跑哪儿去了，减去原始位置$x_q$就是flow vector。

这就好像你在水流里滴一滴墨水，看墨水跑到哪里，就知道水流方向。**模型内部的motion representation就是那条"河流"**。

**Object segmentation怎么做？**

这就更巧妙了。借用[Gestalt psychology的common fate原则](https://linkinghub.elsevier.com/retrieve/pii/0010028583900178)：同一物体的pixels会一起动。

1. 选一个candidate object的patch，用"hypothetical motion"把它displace到新位置（在$f_2^{masked}$里直接挪过去）
2. 让predictor预测剩下的masked区域，产生hypothetical scene $\tilde{f}$
3. 计算原图$f$和hypothetical scene $\tilde{f}$之间的optical flow
4. 同一物体的pixels会跟着displacement走，flow magnitude大；其他pixels几乎不动
5. Threshold flow magnitude → binary mask
6. 8个方向各做一次，aggregate得到robust的segment

**intuition**：你在心里问模型"如果这个杯子往右移了3厘米，整个画面会变成什么样？"模型给你一个hypothetical scene。你比较原scene和hypothetical scene，发现杯子整体平移了，但桌子、背景没动。那些跟着动的pixels就是杯子。

这就是把objectness定义成一个**反事实实验**：if this patch moves, what else moves with it?

参考[SpelkeBench](https://arxiv.org/abs/2507.16038)的"Spelke segments"概念。

---

### Idea 3：简单能力组合成复杂能力

ZWM的prompts是**可以compose**的，形成一个computational graph：

```
RGB pixels
    ↓ optical flow prompt
optical flow
    ↓ 应用到stereo pair
relative depth
    ↓ 
hypothetical motion + optical flow
    ↓
object segmentation
    ↓ + flow
intuitive physics
```

**Relative depth怎么做？**

人有两只眼睛，binocular vision。给stereo pair $(f_L, f_R)$，在$f_L$的query point上加tracer，计算到$f_R$的optical flow。flow magnitude就是binocular disparity，disparity和depth成反比。

$$\text{depth} \propto \frac{1}{\text{disparity}}$$

近的东西disparity大（两只眼睛看它角度差大），远的东西disparity小。

**Intuitive physics怎么做？**

这个最有意思。benchmark测试5类物理reasoning：cohesion（物体完整性）、support（支撑关系）、force transfer（力传递）、force separation（力隔离）。

做法：
1. Context frame $f_1$（手还没动）
2. Target frame $f_2^{masked}$，reveal手的位置（32×32 green patch）+ 背景anchor patches（32×32 red patches，fix illumination和camera pose）
3. 让模型predict整个$f_2$
4. 判断prediction更接近ground-truth $f_2$还是更接近 $f_1$

**intuition**：如果模型理解physics，它应该predict"手推动了物体，物体移动了"——prediction接近真实的$f_2$。如果模型不理解physics，它会做模糊的interpolation——prediction接近$f_1$。

作者还做了attention head分析：在deeper layers，从moved object的query patch到hand patches的attention weight**显著高于**到background或random patches的attention。这暗示模型在deeper layers学到了"agent → object"的causal relationship。

---

## 实验结果怎么样？

### Optical Flow

在[TAP-Vid-DAVIS](https://arxiv.org/abs/2211.03726)（real-world videos）上，BabyZWM和supervised的[CoTracker3](https://arxiv.org/abs/2307.07635)、[SeaRAFT](https://arxiv.org/abs/2405.14793)competitive。

**BabyZWM从来没见过flow label，却达到了supervised水平。**

在TAP-Vid-Kubric（synthetic）上略低，因为supervised models用了synthetic training data。

### Relative Depth

在[UniQA-3D](https://arxiv.org/abs/2410.10799)上，ZWM超过90% accuracy。

- 超过Gemini-1.5、GPT-4-Turbo、GPT-4o
- 和supervised monocular的[MiDaS](https://arxiv.org/abs/1907.01341)、self-supervised的[MonoDepth2](https://arxiv.org/abs/1806.01260)comparable
- 略低于supervised binocular的[FoundationStereo](https://arxiv.org/abs/2501.09898)（这个用了binocular supervision）

### Object Segmentation

在[SpelkeBench](https://arxiv.org/abs/2507.16038)上，BabyZWM和Mask2Former（trained on [COCO](https://arxiv.org/abs/1405.0312)）相当，略低于[SAM2](https://arxiv.org/abs/2408.00714)（用了大规模human annotation）。

**注意**：BabyZWM是class-agnostic的，它不知道"这是杯子那是球"，但它知道"这是一个object那是另一个object"。

### Intuitive Physics

5个category全部接近100%。

对比：V-JEPA2（在BVD上训练）也接近100%，但Baby V-JEPA2（在BabyView上训练）明显下降。**说明pixel-space prediction（ZWM）比feature-space prediction（V-JEPA2）在data-sparse regime下更data-efficient。**

---

## 最striking的发现：Single-Child实验

作者做了一个很狠的实验：只用**一个小孩**的132小时视频训练，看模型能不能学到generalizable的能力。

这个小孩9到30个月大，132小时，视觉diversity极度受限——就是这小孩家里的那些东西。

结果：**Single-Child BabyZWM在大部分task上和用34个小孩868小时训练的BabyZWM表现相似！**

这说明什么？**Total exposure比diversity更重要。** 关键不是见过多少种场景，而是见过足够多的motion、enough的frame pairs让模型学到"world如何work"。

而且作者还试了age-ordered curriculum（按小孩年龄顺序训练，不shuffle），发现和shuffled curriculum表现相似。**ZWM对catastrophic forgetting鲁棒，支持continual learning。**

---

## Developmental Trajectory

作者在不同training checkpoint评估，画出了"发展曲线"：

| Checkpoint | Optical flow | Depth | Segmentation | Intuitive physics |
|---|---|---|---|---|
| 0 | ~random | ~random | ~random | ~random |
| 5k | 快速上升 | 陡升 | 缓慢提升 | 缓慢提升 |
| 40k | 接近plateau | 高 | 持续提升 | 持续提升 |
| 200k | plateau | 高 | 接近peak | 接近peak |

这和儿童发展曲线qualitatively parallel：
- Optical flow早期快速发育 → 对应婴儿single/multi-object tracking发展 [Trick et al. 2005](https://linkinghub.elsevier.com/retrieve/pii/S0885201405000249)
- Depth早期steep发育 → 对应早期stereopsis [Held et al. 1980](https://pnas.org/doi/full/10.1073/pnas.77.9.5572)
- Segmentation持续提升 → 对应object perception发展 [Johnson 2010](https://onlinelibrary.wiley.com/doi/10.1111/j.1551-6709.2010.01127.x)
- Intuitive physics稳步提升 → 对应从coarse expectations到precise reasoning [Baillargeon et al. 2012](https://www.tandfonline.com/doi/abs/10.1080/15475441.2012.630610)

---

## Neural Predictivity：模型的"脑子"像人脑吗？

用两个benchmark测：
- [NSD](https://www.nature.com/articles/s41593-021-00962-x)：human fMRI
- [TVSD](https://linkinghub.elsevier.com/retrieve/pii/S089662732400881X)：macaque单神经元电生理

方法：fit一个linear regression从model features到neural responses，看noise-corrected correlation。

发现：
1. **Hierarchical correspondence**：early model layers对应V1/V2，deep layers对应V4/ventral regions。这和[Felleman & Van Essen 1991](https://academic.oup.com/cercor/article-lookup/doi/10.1093/cercor/1.1.1)的hierarchical visual organization一致。
2. **"Early-first" trajectory**：early visual cortex的noise ceiling更早达到，higher regions需要更多training。这和儿童visual cortex发展顺序一致 [Gogtay et al. 2004](https://pnas.org/doi/full/10.1073/pnas.0402680101)。

**一个self-supervised world model，跨species、跨measurement modality都展现出brain-like的representational structure。**

---

## 这事儿为什么重要？

### 1. 对发展心理学

[Spelke 2000](https://doi.apa.org/doi/10.1037/0003-066X.55.11.1233)的core knowledge theory说婴儿有innate的object、physics等concept。这是nativist立场。

ZWM提供了一个computational middle ground：
- **Innate的部分**：架构（ViT）、learning algorithm（masked prediction）、prompt procedures（causal perturbation）
- **Learned的部分**：所有representational content，从data中学

所以**core knowledge的"core"可能是inference procedures，而不是representational content本身**。content是从experience中emerge的，但extract content的mechanism是innate的。

### 2. 对AI工程

当前visual SSL最大的deployment bottleneck：**每个task需要labeled readout**。想segment？标数据训head。想depth？标数据训head。想flow？标数据训head。

ZWM消除了这个依赖。**一个predictor yield多个能力，zero-shot。**

这和NLP领域从BERT-style pretrain+finetune向GPT-style zero-shot的shift完全类比。但ZWM用**3 orders of magnitude less data**实现了这个shift。

参考[Sutton的Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)：ZWM是hybrid——不是纯scaling，也不是纯hand-engineering，而是**正确的inductive bias + scaling**。

### 3. 对Robotics和Embodied AI

Robotics最大的问题是label稀缺。ZWM证明：**正确的inductive bias可以大幅压缩data需求**。这对robotics、medical imaging、embodied AI等领域有直接意义。

---

## 我的intuition总结

### 为什么temporally-factored masking如此关键？

本质是**bottleneck-based disentanglement**。类比β-VAE：限制latent capacity强迫disentangle。ZWM通过限制$f_2$的visible pixels，强迫motion pathway压缩成低维tokens。

更深层的intuition：**自然视频的motion是低维的**（相机6-DoF + 刚体运动），appearance是高维的。asymmetric masking恰好match这个natural factorization。

### Causal perturbation作为universal interface

这是最深刻的贡献。和NLP prompt engineering结构相似：

- NLP：用自然语言描述task → LLM infer → answer
- ZWM：用perturbation描述query → Ψ propagate → visual quantity

但ZWM的prompt是**结构化的、可组合的**，不是文本的natural language。这可能比NLP prompt更robust。

### "Data-driven World Model"的含义

ZWM叫"world model"但input没有actions。怎么回事？

作者的clever insight：**用cheap data operations（pixel-patch displacement）proxy expensive true actions**。当你displace一个patch，相当于"如果这个物体这样动了会怎样"的hypothetical action。模型从未见过真实action labels，但因为trained on raw video，它学到了"world如何work"，所以能competently做hypothetical。

这接近[Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122)的精神，只是把"sense → model → act"中的act用data perturbation替代。

---

## Limitations和Future Work

### Limitations

1. **没有semantic concepts**：ZWM只学到physics-grounded quantities，没涉及named categories。未来需要integrate linguistic/auditory data。
2. **Deterministic regression导致mode collapse**：在uncertain情况下prediction会blur，限制long-horizon prediction。
3. **缺乏human developmental behavioral/neural datasets**做精细对比。

### Future Work

作者提到一个intriguing方向：**把zero-shot extracted intermediates feed back to $\Psi$ as additional targets**，形成bootstrapping cycle。每个intermediate（flow、depth、segment）都成为新的learning target，反过来enrich predictor。

这和[Kotar et al. 2025](https://arxiv.org/abs/2509.09737)的probabilistic structure integration思路一致，可能是下一步的关键。

---

## 一句话总结

**ZWM用temporally-biased masked prediction把appearance和motion factorize开，用causal perturbation作为universal interface zero-shot extract出各种visual-cognitive能力，用compositional prompting把简单能力组合成复杂能力。用868小时一个小孩的第一人称视频训练，就能在多个task上达到supervised SOTA水平，同时recapitulate儿童发展轨迹和brain-like representations。**

这或许正在见证visual AI从"representation learning"范式向"world modeling"范式的shift，正如NLP从BERT-style向GPT-style shift一样。

---

## 关键参考链接

**主paper和相关系列：**
- [ZWM主paper](https://arxiv.org/abs/2509.09737)
- [Bear et al. 2023 (ZWM前作)](https://arxiv.org/abs/2306.01828)
- [Kotar et al. 2025 (BVD扩展)](https://arxiv.org/abs/2509.09737)
- [Kim et al. 2025 (zero-shot flow extraction)](https://arxiv.org/abs/2507.09082)
- [Lee et al. 2025 (3D scene understanding)](https://arxiv.org/abs/2504.03875)

**数据集和benchmark：**
- [BabyView](https://arxiv.org/abs/2406.10447)
- [SpelkeBench](https://arxiv.org/abs/2507.16038)
- [TAP-Vid](https://arxiv.org/abs/2211.03726)
- [UniQA-3D](https://arxiv.org/abs/2410.10799)
- [NSD](https://www.nature.com/articles/s41593-021-00962-x)
- [TVSD](https://linkinghub.elsevier.com/retrieve/pii/S089662732400881X)

**Baseline models：**
- [V-JEPA2](https://arxiv.org/abs/2506.09985)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [SAM2](https://arxiv.org/abs/2408.00714)
- [MAE](https://arxiv.org/abs/2111.06377)
- [ViT](https://arxiv.org/abs/2010.11929)

**Cognitive science背景：**
- [Spelke "Core knowledge" (2000)](https://doi.apa.org/doi/10.1037/0003-066X.55.11.1233)
- [Spelke "What Babies Know" (2022)](https://academic.oup.com/book/43912)
- [Baillargeon et al. (1985)](https://linkinghub.elsevier.com/retrieve/pii/0010277785900083)
- [Carey "Origin of Concepts" (2009)](https://global.oup.com/academic/product/the-origin-of-concepts-9780195367638)
- [Sutton "Bitter Lesson"](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
- [Frank "Bridging data gap" (2023)](https://linkinghub.elsevier.com/retrieve/pii/S1364661323002036)

**World model传统：**
- [Ha & Schmidhuber "World Models" (2018)](https://arxiv.org/abs/1803.10122)
- [Dreamer](https://arxiv.org/abs/1912.01603)
- [MuZero](https://www.nature.com/articles/s41586-020-03051-4)

**Neural predictivity：**
- [Brain-Score](https://www.biorxiv.org/lookup/doi/10.1101/407007)
- [Yamins et al. 2014](https://pnas.org/doi/full/10.1073/pnas.1403112111)

---

# Zero-shot World Models Are Developmentally Efficient Learners — 深度解读

这篇paper来自Stanford的Khai Loong Aw、Klemen Kotar、Daniel Yamins和Michael Frank等人的工作，地址在 [arXiv](https://arxiv.org/abs/2509.09737)（实际ZWM主paper可能引用编号稍异，相关系列工作见文末链接)。它试图回答一个非常深刻的问题：**婴幼儿如何在极少数据下快速获得灵活的、zero-shot的视觉认知能力，同时还能泛化到训练中从未见过的新任务？** 这直接挑战了今天大规模视觉模型data-hungry的现状。

## 一、核心问题与动机

### 1.1 生物学背景

婴儿在几个月大时就能estimate depth、track motion、perceive object coherence、做intuitive physics reasoning（参考Spelke的core knowledge理论 [Spelke 2000](https://doi.apa.org/doi/10.1037/0003-066X.55.11.1233)；[Baillargeon et al. 1985](https://linkinghub.elsevier.com/retrieve/pii/0010277785900083)）。这有两层意义：
- **Data-efficient**：只用单一个体的第一人称视觉经验，远比ImageNet规模小
- **Flexible (zero-shot)**：无需task-specific labeled examples即可执行多种视觉认知任务

[BabyView数据集](https://arxiv.org/abs/2406.10447) 提供了34个儿童、868小时的高分辨率egocentric视频，让这个computational question变得可验证。

### 1.2 现有self-supervised learning的局限

现代SSL方法（如DINOv3 [arXiv](https://arxiv.org/abs/2508.10104)、V-JEPA2 [arXiv](https://arxiv.org/abs/2506.09985)）在BabyView上训练虽然比predictive coding进步，但远未达到人类水平。更关键的是，它们**无法zero-shot执行下游任务**——必须为每个task训练一个readout head，这在发展心理学上是不plausible的（婴儿不会为每个新task单独收集labels）。这与NLP领域形成鲜明对比，LLM可以zero-shot执行多种任务，但代价是海量数据。

## 二、ZWM的三大设计原则

ZWM的intuition可以归纳为：**让模型在训练时学会预测"未来"，在推理时通过"反事实perturbation"来quote出它隐式学到的视觉概念**。

### 2.1 Principle 1: Sparse Temporally-Factored Prediction

核心是把masked autoencoder ([MAE, He et al. 2021](https://arxiv.org/abs/2111.06377))的mask从"spatially random"改为"temporally biased"。

#### 数学形式

给定相邻两帧 $(f_1, f_2)$，时间间隔150–450ms。Predictor $\Psi_\Theta$ 接收 $(f_1, f_2^{masked})$，其中 $f_2$ 只reveal 10%的patches，$f_1$ 完全visible。

$$\Psi_{\Theta^*}: (f_1, f_2^{masked}) \mapsto \hat{f}_2; \quad \Theta^* := \arg\min_\Theta \mathbb{E}_{(f_1,f_2) \in \mathcal{D}} \|f_2 - \hat{f}_2\|^2$$

变量含义：
- $f_1 \in \mathbb{R}^{H \times W \times 3}$：第一帧（完整RGB），代表"过去"
- $f_2^{masked} \in \mathbb{R}^{H \times W \times 3}$：第二帧被mask后，只有约10%的8×8 patches保留（其余替换为shared learnable mask token），代表"未来稀疏观察"
- $\hat{f}_2$：模型预测的完整第二帧
- $\Theta$：网络所有可学习参数的集合
- $\Theta^*$：训练收敛后的最优参数
- $\mathcal{D}$：训练数据集分布
- $\|\cdot\|^2$：像素级L2 norm

#### 为什么这种asymmetric masking会factorize appearance和motion？

Intuition是这样的：要从10%的sparse patches重建整个$f_2$，模型必须从$f_1$中"借"appearance信息（因为$f_2$只reveal极少像素，无法仅从$f_2$自身重建appearance），同时从$f_2$中那10%的sparse revealed patches推断motion/dynamics（因为同一物体在$f_1$和$f_2$间的位移只能从那些sparse patches中读取）。

这相当于强制网络把"什么"（appearance）和"怎么动"（motion）解耦到不同的latent dimensions。这是**information bottleneck**的应用——给motion通路留一条窄通道，强迫它压缩成低维motion tokens，而appearance通路则可以从$f_1$无限制地borrow。

Ablation实验验证了这一点：symmetric masking（45%-45%或90%-90%）表现差很多（见原文Figures 2,3）。这说明**temporal asymmetry**本身是核心inductive bias，而非"masked prediction"这个generic operation。

### 2.2 Principle 2: Zero-shot Extraction via Approximate Causal Inference

这是ZWM最巧妙的部分。训练好的predictor $\Psi$ 隐式编码了大量visual-cognitive knowledge，但如何**zero-shot**地把它quote出来？

借鉴[Judea Pearl的causal inference](https://www.cambridge.org/core/books/causality/9407819B7E7A4D6F2A4C9F4F4B4D4F4D)思想：做minimal intervention，看output如何改变。

形式化：

$$x_\delta := \text{perturb}(x); \quad \delta\Psi := \text{compare}(\Psi(x), \Psi(x_\delta)); \quad \text{output} := \text{aggregate}(\delta\Psi)$$

变量含义：
- $x$：原始输入（如 $(f_1, f_2^{masked})$）
- $x_\delta$：对$x$做minimal perturbation后的输入
- $\Psi(x), \Psi(x_\delta)$：分别forward两次得到的预测
- $\delta\Psi$：两次预测的difference，反映了perturbation如何在model内部propagate
- output：对$\delta\Psi$做aggregation得到的最终visual quantity

**关键insight**：把$\Psi$视为"learned structural equation for world dynamics"。当你perturb某个pixel patch，predictor会把这个perturbation"传播"给与它causally相关的pixels。**相关的pixels其实就是同一物体上的pixels**（common fate principle）。这给了我们一个objectness detector：perturb一个patch，看哪些pixels跟着动。

#### Optical Flow的zero-shot实现

具体例子：要估计query point $x_q$在$f_1$到$f_2$的flow：

1. **Perturb**：在$f_1$的$x_q$位置加一个Gaussian white dot（amplitude 255, $\sigma=3$ pixels），形成 $\tilde{f}_1$
2. **Compare**：同一 $f_2^{masked}$ 下forward两次：
   - clean: $(f_1, f_2^{masked}) \to \hat{f}_2$
   - perturbed: $(\tilde{f}_1, f_2^{masked}) \to \tilde{f}_2^{pred}$
   - 计算 $\Delta = \tilde{f}_2^{pred} - \hat{f}_2$
3. **Aggregate**：$\text{flow}(x_q) = \arg\max(|\Delta|) - x_q$

Intuition：那个Gaussian dot是"tracer"，predictor会沿着motion的因果链把这个tracer"搬运"到$f_2$中对应的位置。argmax的位置就是物体在$f_2$中的corresponding point。

### 2.3 Principle 3: Compositional Prompting

简单prompt组合成复杂query，形成computational graph of visual intermediates：

```
RGB pixels
    ↓ optical flow prompt
optical flow
    ↓ stereo pair flow
relative depth
    ↓ 
hypothetical motion + optical flow
    ↓
object segmentation
    ↓ + flow
intuitive physics
```

这种composition非常elegant：**每个新能力都建立在前一个能力之上，就像婴儿发展一样**。

## 三、架构与训练细节

### 3.1 模型架构

Backbone是[ViT (Dosovitskiy et al. 2021)](https://arxiv.org/abs/2010.11929)，两种配置：

| Hyperparameter | ZWM-170M | ZWM-1B |
|---|---|---|
| Transformer layers | 24 | 48 |
| Attention heads | 12 | 16 |
| Embedding dim | 768 | 1280 |
| Patch size | 8×8 | 8×8 |
| Input resolution | 256×256 | 256×256 |
| Tokens/frame | 1024 | 1024 |
| Total params | ~170M | ~1B |

输入tokenization：$f_1$ → 1024个192-dim tokens（每个是8×8×3=192维向量）+ 位置embedding；$f_2$ → 102个visible tokens + 922个shared learnable mask tokens + 位置embedding。两组token concat后送入transformer。

### 3.2 训练超参

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| $\beta_1, \beta_2$ | 0.9, 0.95 |
| Peak learning rate | 3e-4 |
| Weight decay | 1e-1 |
| LR schedule | Cosine decay, 2000 warmup |
| Batch size | 512 |
| Total steps | 200,000 |
| Inter-frame gap | 150–450ms (5–14 frames @ 30fps) |
| $f_1$ patches revealed | 100% |
| $f_2$ patches revealed | 10% |

**Compute**：
- ZWM-170M: 32 H100 GPUs, ~11h, ~352 GPU-hours
- ZWM-1B: 64 H100 GPUs, ~24h, ~1536 GPU-hours

200k steps × 512 batch ≈ 1.02亿 frame pairs，对应~950 video hours，约95天的婴儿waking experience（按~10 awake hours/day计算，参考[Iglowstein et al. 2003](https://publications.aap.org/pediatrics/article/111/2/302/66745/)）。

### 3.3 数据集对比

| Dataset | Hours | Source | ZWM variant |
|---|---|---|---|
| BabyView | 868 | 34 children, egocentric | BabyZWM |
| Single-Child BabyView | 132 | 1 child (9–30 months) | Single-Child BabyZWM |
| Random 132h subset | 132 | 34 children | — |
| Kinetics-400 | ~670 | YouTube | ZWM-Kinetics |
| BVD | ~7,000 | CV datasets + Internet | ZWM-BVD |

## 四、Zero-shot Prompts详解

下表总结了所有prompt的 Perturb/Compare/Aggregate 结构：

| Task | Perturb | Compare | Aggregate | Composes from |
|---|---|---|---|---|
| Optical flow | Gaussian tracer at $x_q$ in $f_1$ | RGB diff $\Delta$ between perturbed & clean $\hat{f}_2$ | argmax$|\Delta|$ − $x_q$ = flow vector | Primitive |
| Hyp. motion | Displace object patch in $f_2^{masked}$ | Predict remaining masked regions | Full hypothetical scene | Primitive |
| Relative depth | Flow tracer on $f_L$ in stereo pair $(f_L, f_R)$ | Optical flow from $f_L$ to $f_R$ | Rank points by disparity (inverse of depth) | Optical flow |
| Object seg. | Displace object patch via hyp. motion | Optical flow between original & hyp. scene | Threshold flow, aggregate over 8 directions | Hyp. motion + flow |
| Intuitive physics | Reveal hand's GT location (32×32 green) + red background patches in $f_2^{masked}$ | MSE/LPIPS between prediction & $f_2$ | Closer to target or context? | Hyp. motion + flow + seg. |

### 4.1 Object Segmentation的intuition

这是最elegant的部分。借用[Gestalt psychology的"common fate"原则](https://linkinghub.elsevier.com/retrieve/pii/0010028583900178)：同一物体的pixels会一起move。

- 给定一张图$f$，选一个candidate object的patch，用hypothetical motion把它displace 25–35 pixels到新位置（在8个不同方向各做一次）
- 让predictor预测剩下的masked区域，产生hypothetical scene $\tilde{f}$
- 计算$f$和$\tilde{f}$之间的optical flow
- 同一物体的pixels会跟着displacement走，flow magnitude大；其他物体或背景的pixels几乎不动
- Threshold flow magnitude → binary mask
- Aggregate over 8 directions → robust object segment

这就是把"objectness"转化为一个**反事实实验**：if this patch moves, what else moves with it?

### 4.2 Intuitive Physics的intuition

Intuitive physics benchmark测试5类：cohesion、support (top moves)、support (bottom moves)、force transfer、force separation。

- Context frame $f_1$ + 部分unmask的 $f_2^{masked}$（reveal手的位置 + 背景anchor patches）
- 模型predict整个$f_2$
- Accuracy: prediction更接近ground-truth $f_2$ 还是更接近 $f_1$？

如果模型理解了physics，它应该能predict出"手推动了物体，物体移动了"——prediction会接近真实的 $f_2$。如果模型不理解physics，它会做模糊的interpolation，prediction会接近 $f_1$。

#### Attention head分析

作者做了mechanistic interpretability：观察deeper layers的attention从moved object的query patch指向哪里。结果发现deep layers的attention**优先指向hand patches**（causal agent），而不是background或random patches（见原文Figure S1）。

这暗示模型在deeper layers学习到了"agent → object"的causal relationship，这是intuitive physics的核心。

## 五、实验结果

### 5.1 Optical Flow (TAP-Vid)

| Model | TAP-Vid-DAVIS ($<\delta_{\text{avg}}^x$) | TAP-Vid-Kubric | Occlusion Acc |
|---|---|---|---|
| CoTracker3 (supervised) | High | High | High |
| DPFlow (supervised) | High | High | — |
| SeaRAFT (supervised) | High | High | — |
| **BabyZWM (zero-shot)** | Competitive | Slightly below supervised | Matches supervised |
| DINOv3 | Lower | Lower | — |
| V-JEPA2 | Lower | Lower | — |

参考[TAP-Vid benchmark](https://arxiv.org/abs/2211.03726)、[CoTracker3](https://arxiv.org/abs/2307.07635)、[SeaRAFT](https://arxiv.org/abs/2405.14793)。

BabyZWM在real-world DAVIS上与supervised baselines competitive，在synthetic Kubric上略低（supervised models用了synthetic training data）。**关键insight**：BabyZWM从未见过flow label，却达到了supervised水平。

### 5.2 Relative Depth (UniQA-3D)

| Model | Accuracy |
|---|---|
| FoundationStereo (supervised binocular) | Highest |
| **ZWM / BabyZWM (zero-shot, binocular)** | >90% |
| MiDaS-CNN (supervised monocular) | Comparable |
| MonoDepth2 (self-supervised monocular) | Comparable |
| Gemini-1.5 / GPT-4-Turbo / GPT-4o | Below ZWM |
| ResNet50 / DINOv3 / V-JEPA2 (zero-shot probe) | Lower |

参考[MiDaS](https://arxiv.org/abs/1907.01341)、[MonoDepth2](https://arxiv.org/abs/1806.01260)、[FoundationStereo](https://arxiv.org/abs/2501.09898)。

ZWM用的是binocular input（stereo pair），所以比monocular baselines强合理，但仍逊于supervised binocular。

### 5.3 Object Segmentation (SpelkeBench)

参考[SpelkeBench](https://arxiv.org/abs/2507.16038)、[Mask2Former](https://arxiv.org/abs/2112.01527)、[SAM2](https://arxiv.org/abs/2408.00714)。

BabyZWM在SpelkeBench（class-agnostic object segmentation）上与Mask2Former（trained on COCO）相当，但略低于SAM2（用了大规模human annotation）。**Single-Child BabyZWM**（只训练一个孩子的132小时视频）表现也很接近，说明**diversity不是必要的，total exposure足够即可**。

### 5.4 Intuitive Physics

| Model | Cohesion | Support (top) | Support (bot) | Force transfer | Force sep. |
|---|---|---|---|---|---|
| ZWM (BVD) | ~100% | ~100% | ~100% | ~100% | ~100% |
| **BabyZWM** | ~100% | ~100% | ~100% | ~100% | ~100% |
| V-JEPA2 | ~100% | ~100% | ~100% | ~100% | ~100% |
| Baby V-JEPA2 | Lower | Lower | Lower | Lower | Lower |

V-JEPA2在BabyView上训练后下降明显，说明**pixel-space prediction（ZWM）比feature-space prediction（V-JEPA2）在data-sparse regime下更有data-efficient**。

### 5.5 Developmental Trajectory

原文Figure 5展示training checkpoints（0, 5k, 10k, 20k, 40k, 80k, 120k, 160k, 200k）的性能发展曲线：

- **Optical flow**：先快速上升后plateau，对应儿童single/multi-object tracking发展 ([Trick et al. 2005](https://linkinghub.elsevier.com/retrieve/pii/S0885201405000249))
- **Relative depth**：陡升后保持，对应早期stereopsis发育 ([Held et al. 1980](https://pnas.org/doi/full/10.1073/pnas.77.9.5572))
- **Object segmentation**：持续提升，对应婴儿object perception发展 ([Johnson 2010](https://onlinelibrary.wiley.com/doi/10.1111/j.1551-6709.2010.01127.x))
- **Intuitive physics**：稳步提升，对应婴儿从coarse expectations到precise reasoning的过程 ([Baillargeon et al. 2012](https://www.tandfonline.com/doi/abs/10.1080/15475441.2012.630610))

**Age-ordered curriculum**（按儿童年龄排序训练）与shuffled curriculum表现相似，说明ZWM对catastrophic forgetting鲁棒，支持continual learning。

### 5.6 Neural Predictivity

参考[Brain-Score](https://www.biorxiv.org/lookup/doi/10.1101/407007)、[NSD](https://www.nature.com/articles/s41593-021-00962-x)、[TVSD](https://linkinghub.elsevier.com/retrieve/pii/S089662732400881X)。

- **NSD (human fMRI)**：BabyZWM的早期层对应V1/V2，深层对应V4/ventral regions
- **TVSD (macaque electrophysiology)**：同样hierarchical correspondence，从V1到V4到IT
- **"Early-first" trajectory**：早期visual cortex的noise ceiling更早达到，符合hierarchical visual development ([Felleman & Van Essen 1991](https://academic.oup.com/cercor/article-lookup/doi/10.1093/cercor/1.1.1); [DiCarlo et al. 2012](https://linkinghub.elsevier.com/retrieve/pii/S089662731200092X))

这给ZWM提供了**mechanistic mapping** ([Cao & Yamins 2021](https://arxiv.org/abs/2104.01490))：模型layer → cortical region，且这个mapping跨species、跨measurement modality一致。

## 六、深层讨论

### 6.1 ZWM与"Bitter Lesson"的关系

Richard Sutton的[Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)说复杂hand-built inductive bias没必要，scaling + general learning就够了。ZWM采取**hybrid立场**：架构、learning algorithm、task readout programs是innate的structural priors；但representational content和network parameters从experience中学。

这暗示了一种新的"innateness"定义：进化可以hard-wire **the inference procedures**（如何从world model中extract信息），但**the content of the world model**本身要从data中学习。

### 6.2 "Data-driven World Model"的含义

ZWM叫"world model"，但input没有actions，怎么算world model？参考[Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122)、[Dreamer](https://arxiv.org/abs/1912.01603)、[MuZero](https://www.nature.com/articles/s41586-020-03051-4)。

作者的回答：**用cheap data operations（如pixel-patch displacement）proxy expensive true actions**。当你displace一个patch，相当于"如果这个物体这样动了会怎样"的hypothetical action。模型从未见过真实action labels，但因为trained on raw video，它学到了"world如何work"，所以能competently做hypothetical。

这其实非常接近[World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122)的精神，只是把"sense → model → act"中的act用data perturbation替代。

### 6.3 与V-JEPA2的对比

[V-JEPA2](https://arxiv.org/abs/2506.09985)在feature space预测，避免pixel-level reconstruction的blur问题。但实验显示，在BabyView这种data-sparse regime下，ZWM的pixel-space prediction反而更data-efficient。

可能的解释：pixel prediction的supervision signal更dense（每个pixel都是loss target），而feature prediction的target是另一个network的output，bootstrap可能引入更多instability。在数据充足时V-JEPA2的feature space更robust，但在data-sparse时pixel supervision的优势更突出。

### 6.4 Limitations

1. **没有semantic concepts**：ZWM只学到physics-grounded quantities（depth, motion, objects），没涉及named categories。未来需要integrate linguistic/auditory data。
2. **Deterministic regression导致mode collapse**：在uncertain情况下prediction会blur，限制long-horizon prediction和control。
3. **缺乏human developmental behavioral/neural datasets**做精细对比。

### 6.5 未来方向

作者提到一个intriguing的future work：**把zero-shot extracted intermediates feed back to $\Psi$ as additional targets**，形成bootstrapping cycle。这样每个intermediate（flow、depth、segment）都成为新的learning target，反过来enrich predictor。这与[Kotar et al. 2025](https://arxiv.org/abs/2509.09737)的probabilistic structure integration思路一致。

## 七、个人intuition与思考

### 7.1 为什么temporally-factored masking如此关键？

这本质是一种**bottleneck-based disentanglement**。可以类比β-VAE：通过限制latent capacity强迫disentangle。这里通过限制$f_2$的visible pixels，强迫motion pathway压缩成低维tokens，appearance pathway则从$f_1$无限制borrow。

更深层的intuition：**自然视频的motion是低维的**（相机6-DoF + 物体刚性运动），而appearance是高维的。asymmetric masking恰好match这个natural factorization。

### 7.2 Causal inference作为universal interface

ZWM最深刻的贡献是：**用causal perturbation作为universal zero-shot interface**。这与NLP的prompt engineering有结构相似性：

- NLP prompt：用自然语言描述task → LLM infer → answer
- ZWM prompt：用perturbation描述query → Ψ propagate → visual quantity

但ZWM的prompt是**结构化的、可组合的**，不是文本的natural language。这或许能避免NLP prompt的brittleness。

### 7.3 发展心理学意义

ZWM给developmental nativism vs. empiricism debate提供了一个具体的computational middle ground：
- **Nativism**部分对：架构、algorithm、prompt procedures是innate的
- **Empiricism**部分对：content从data学，无需labeled examples

这挑战了强nativist accounts（如[Spelke 2022](https://academic.oup.com/book/43912)对core knowledge的strong interpretation）。

### 7.4 对AI的启示

当前visual SSL最大的deployment bottleneck是**task-specific labeled readout**。ZWM消除了这个依赖，一个predictor yield多个能力。这对robotics、medical imaging、embodied AI等领域（label稀缺）有直接意义。

更provocatively：**ZWM展示了zero-shot能力可以用3 orders of magnitude less data实现**（vs. LLM-scale）。这暗示visual domain的inductive structure比language更"稠密"，正确的inductive bias可以大幅压缩data需求。

## 八、相关参考链接

主paper系列：
- ZWM/BVD系列: [Kotar et al. 2025 (World Modeling with Probabilistic Structure Integration)](https://arxiv.org/abs/2509.09737)
- [3D Scene Understanding Through Local Random Access Sequence Modeling (Lee et al. 2025)](https://arxiv.org/abs/2504.03875)
- [Taming generative video models for zero-shot optical flow extraction (Kim et al. 2025)](https://arxiv.org/abs/2507.09082)
- [Unifying (Machine) Vision via Counterfactual World Modeling (Bear et al. 2023)](https://arxiv.org/abs/2306.01828)

数据集与benchmark：
- [BabyView dataset (Long et al. 2024)](https://arxiv.org/abs/2406.10447)
- [SpelkeBench (Venkatesh et al. 2025)](https://arxiv.org/abs/2507.16038)
- [TAP-Vid (Doersch et al. 2023)](https://arxiv.org/abs/2211.03726)
- [UniQA-3D (Zuo et al. 2024)](https://arxiv.org/abs/2410.10799)
- [Natural Scenes Dataset (Allen et al. 2022)](https://www.nature.com/articles/s41593-021-00962-x)
- [TVSD (Papale et al. 2025)](https://linkinghub.elsevier.com/retrieve/pii/S089662732400881X)

Baseline models:
- [V-JEPA2 (Assran et al. 2025)](https://arxiv.org/abs/2506.09985)
- [DINOv3 (Siméoni et al. 2025)](https://arxiv.org/abs/2508.10104)
- [SAM2 (Ravi et al. 2024)](https://arxiv.org/abs/2408.00714)
- [MAE (He et al. 2021)](https://arxiv.org/abs/2111.06377)
- [ViT (Dosovitskiy et al. 2021)](https://arxiv.org/abs/2010.11929)

Vision-language baselines:
- [Gemini 1.5](https://arxiv.org/abs/2403.05530)
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [GPT-4o System Card](https://arxiv.org/abs/2410.21276)

Cognitive science背景：
- [Spelke "Core knowledge" (2000)](https://doi.apa.org/doi/10.1037/0003-066X.55.11.1233)
- [Spelke "What Babies Know" (2022)](https://academic.oup.com/book/43912)
- [Baillargeon et al. "Object permanence" (1985)](https://linkinghub.elsevier.com/retrieve/pii/0010277785900083)
- [Carey "The Origin of Concepts" (2009)](https://global.oup.com/academic/product/the-origin-of-concepts-9780195367638)
- [Sutton "The Bitter Lesson"](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
- [Frank "Bridging the data gap" (2023)](https://linkinghub.elsevier.com/retrieve/pii/S1364661323002036)
- [BabyLM Challenge](https://aclanthology.org/2023.conll-babylm)

Neural predictivity framework:
- [Brain-Score (Schrimpf et al. 2018)](https://www.biorxiv.org/lookup/doi/10.1101/407007)
- [Yamins et al. 2014 "Performance-optimized hierarchical models"](https://pnas.org/doi/full/10.1073/pnas.1403112111)

World model传统：
- [Ha & Schmidhuber "World Models" (2018)](https://arxiv.org/abs/1803.10122)
- [Dreamer (Hafner et al. 2019)](https://arxiv.org/abs/1912.01603)
- [PlaNet (Hafner et al. 2018)](https://arxiv.org/abs/1811.04551)
- [MuZero (Schrittwieser et al. 2019)](https://www.nature.com/articles/s41586-020-03051-4)

## 九、总结

ZWM是一个三层贡献叠加的工作：

1. **Algorithmic level**：temporally-factored masked prediction + causal perturbation + compositional prompting 构成了一个data-efficient + zero-shot visual cognition framework
2. **Cognitive science level**：提供了一个具体的computational hypothesis，说明infants如何在有限data下获得灵活visual cognition——core priors是结构性的，content是learned
3. **AI engineering level**：演示了~170M参数、~868小时video即可达到多个SOTA supervised baselines的zero-shot性能，挑战了"必须scale到internet-scale data"的常识

最有意思的开放问题：**这种causal perturbation作为universal interface的范式，能否推广到其他模态（如language、audio）？能否推广到longer-horizon prediction和control？如果能，我们或许正在见证visual AI从"representation learning"范式向"world modeling"范式的shift，正如NLP从BERT-style pretrain+finetune向GPT-style zero-shot shift一样。**

如果你想build deeper intuition，我强烈推荐先看一下[Unifying (Machine) Vision via Counterfactual World Modeling (Bear et al. 2023)](https://arxiv.org/abs/2306.01828)（ZWM的前身思想），以及[Kotar et al. 2025](https://arxiv.org/abs/2509.09737)（BVD扩展到probabilistic、longer-horizon）。这两篇加上本paper构成一个相对完整的研究lineage。
