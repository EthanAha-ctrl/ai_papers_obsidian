---
source_pdf: VLM2-Bench.pdf
paper_sha256: 1b0dccf2dfd0f778dbe1f5d913d1370f7ecdfd3148a3d2fbd13756ea43f79dd0
processed_at: '2026-08-13T03:10:29-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VLM2-Bench:用人话说

## 一句话版本

这篇 paper 问的是:**VLM 能不能像人一样,光靠看——看脸、看衣服、看物体细节——把"这是同一个人/同一个东西"在不同照片或视频帧里 link 起来,完全不需要知道他是谁、叫什么名字。**

答案是不能。人类 94%,GPT-4o 60%,7B 模型在随机猜的水平附近,有几个模型甚至比随机猜还差。

## 为什么这事重要——一个 thought experiment

你打开手机相册,看到一张三年前聚会的照片,里面有个戴红帽子的女生。然后你翻到上周的照片,又看到一个女生,也戴着类似的帽子,但是不同角度、不同光线、不同场景。你一眼就知道"这是同一个人"。

你不需要知道她叫什么、在哪工作、多大年纪。你只是 **link 了 visual cue** ——脸型、眉眼间距、穿衣风格、体态。

这个能力人类 trivial。婴儿几个月大就有 face preference,不需要 identity knowledge。Bruce & Young (1986) 的 face recognition 模型讲的就是这个——recognition 和 identification 是两个模块,前者 pure visual,后者才需要 semantic memory。Treisman & Gelade (1980) 的 feature-integration theory 也讲的是,视觉注意先把 feature bind 成 object,这个过程不需要 language 介入。

现在 VLM 拥有海量 knowledge,能写代码、能解数学、能讲笑话。但它能不能做这个"最 dumb、最 fundamental"的视觉任务?

**这篇 paper 就是来测这件事的。**

参考:Bruce & Young 1986 https://doi.org/10.1111/j.2044-8295.1986.tb02197.x ; Treisman & Gelade 1980 https://doi.org/10.1016/0010-0285(80)90005-5

---

## Benchmark 怎么设计的

三个 category,九个 subtask,3060 个 QA pair。Inter-annotator agreement Fleiss' κ = 0.983,质量极硬。

### General Cue (GC) —— 960 个 T/F 题

给两张图,一张 original 一张 edited。问两种事:

- **Matching (Mat)**: "X 在两张图里是同一个吗?" —— 模型要 link 一致 cue,排除变化 cue。
- **Tracking (Trk)**: "X 在第二张图里变成什么了?" —— 跨图跟踪 transform。

数据来自 image editing 数据集,但用了一个很聪明的 difficulty filter。核心是 Eq. 1:

$$
S_{\text{salient}} = \frac{1}{|\mathcal{P}|} \sum_{i=1}^{|\mathcal{P}|} \log P_\theta(p_i \mid C \cup p_{<i})
$$

变量讲清楚:
- $\mathcal{P}$: edit instruction 分词后的 token 序列,$|\mathcal{P}|$ 是 token 数
- $p_i$: 第 $i$ 个 token
- $p_{<i}$: 第 $i$ 个 token 之前的 prefix
- $C$: 用 VLM 给两张图生成的 caption 拼起来的 text context
- $P_\theta$: Llama3-8B 的条件概率
- $S_{\text{salient}}$: 平均 per-token log-prob

直觉:**这个分数衡量"从 caption 能不能猜出 edit instruction"**。如果分数高,说明 caption 已经泄露了视觉差异(比如 caption 说"图里没有花瓶",那 edit instruction 大概就是"remove the vase"),这种样本太简单,扔掉。保留 $S_{\text{salient}} < -2.0$ 的样本——必须看图才能解的 hard case。

这是个很 elegant 的 difficulty filter,跟 contrastive learning 的 hard negative mining 思路相通。

### Object-centric Cue (OC) —— 1280 个题

测"同一个物体在不同场景下"的识别。8 个 category:pet, plush, bag, book, cup, shirt, shoes, toy。每个 category 5 个 main meta-object,每个 4 张图。每个 main 配 4 个 distractor——同 category 但不同 instance,共享某些 visual cue 制造 confusion。

三个 subtask:

- **Comparison (Cpr)**: T/F 配对。"The pets in these images are the same pet" (GT=T) vs "are not the same pet" (GT=F)。**两题都答对才算对**。这是为了抵消 model 的 affirmative bias——你如果不做这个,模型总猜 T 就能拿 50%。
- **Counting (Cnt)**: 数 unique object 数量。
- **Grouping (Grp)**: MC 题,"哪几张图是同一个 object"。

### Person-centric Cue (PC) —— 820 个题

测"同一个人在不同照片/视频里"的识别。6 个 (race × gender) 组 × 5 个 meta-human × 4 张图 = 120 张图。

Distractor 选择是关键:**计算每张图的 CLIP embedding,做 image-to-image similarity search,选来自不同人但最相似的图**。这样 distractor 在 race / gender / clothing / scene 上尽量接近,只在 fine-grained facial geometry 上不同。

为什么 person 不需要像 object 那样 dedicated distractor?因为 person 只有一个"category"(都是人),任意 meta-human 都能当另一个的 distractor,CLIP 选最相似的就行。Object 不行——pet 和 bag 差异太大,必须 type-specific distractor。

四个 subtask,前三个跟 OC 一样,多了:

- **Video Identity Describing (VID)**: 视频里描述人。两种结构:
  - `P → ¬P`: 两段不同人的 clip 拼接,看模型能不能区分
  - `P → ¬P → P`: 三段拼接,看模型能不能把第一个和第三个 P link 起来

### 为什么这个设计好

对比现有 benchmark:

| 现有 benchmark | VLM2-Bench |
|---|---|
| ActivityNet-QA, MMDU: 不要求跨图 link cue | 显式 require |
| MIBench, MMMU: 依赖 external knowledge | 纯视觉,不依赖 |
| TempCompass, Img-Diff: abstract 比较 | specific cue matching |
| MuirBench: retrieval | direct association |

参考:MuirBench https://arxiv.org/abs/2406.09411 ; MIBench https://arxiv.org/abs/2407.15272

---

## 关键数字

Table 1 是核心:

| Model | Avg | Δ_human |
|---|---|---|
| Human | 94.44 | 0 |
| Chance | 33.72 | -61.44 |
| GPT-4o-2024-08-06 | 59.56 | -34.88 |
| Claude-3.7-sonnet | 59.57 | -34.87 |
| Qwen2.5-VL-7B | 55.86 | -38.58 |
| GPT-4o-2024-11-20 | 55.73 | -38.71 |
| InternVL2.5-26B | 48.58 | -45.86 |
| Gemini-2.0-flash | 28.33 | **-66.11**(低于 chance!) |
| LongVA-7B | 24.95 | -69.49(显著低于 chance) |

**几个 sharp observation**:

1. **Gemini-2.0-flash 和 LongVA-7B 低于 chance level**。这说明它们有 systematic bias——比如总答 T,或者总答某个固定 count。在 Cpr 这种 50% chance 的任务上掉到 21-49%,bias 极严重。这是一个反向 sanity check:benchmark 真的能区分"有能力"和"假装有能力"。

2. **GPT-4o 新版本 (2024-11-20) 反而比老版本 (2024-08-06) 差**。Avg 55.73 vs 59.56。GC-Mat 从 37.45 跌到 18.53,GC-Trk 从 39.27 跌到 29.68。这暗示 RLHF / post-training 在优化 conversational 能力时损害了细粒度 visual perception。跟 Tong et al. 2024 "Eyes Wide Shut" 的发现一致——后训练经常伤害 perception。

参考:Eyes Wide Shut https://arxiv.org/abs/2407.06481

---

## 为什么这么差——从 architecture 讲

这是 Karpathy 你会问的问题。让我从几个 angle 拆:

### Angle 1: ViT 没有 viewpoint-invariant 归纳偏置

ViT 把图切成 patch,每 patch 一个 token。同一个人在不同角度 / 光线 / pose 下,patch token 序列在 embedding space 里的位置不连续。ViT 没有 viewpoint-invariant 的归纳偏置,要靠 training data 学。

对比:face recognition 网络 (ArcFace, CosFace) 用 metric learning + triplet loss 显式训练 identity-invariant embedding。VLM 没有这种 training signal。

这解释了 PC > OC——person 数据有 implicit name 当 metric learning 的 label(Pi et al. 2024b "Personalized Visual Instruction Tuning"),模型隐式学到 face embedding。Object 只用 category name,没有 instance-level anchor。

参考:Personalized VIT https://arxiv.org/abs/2410.07113

### Angle 2: Cross-Attention 的长程依赖问题

VLM 的 image tokens 在 LLM context 里是 sequence。要 link "image 1 的 cue A" 和 "image 4 的 cue A",模型要在 self-attention 里跨长距离 attend。Multi-image scenario 下,image tokens 可能隔几千 token。LLM 的 long-range attention 在 softmax 上会衰减,sharp 度不够。

更关键:**没有 bidirectional cross-image binding 机制**。CLIP-style contrastive training 只做 image-text alignment,不做 image-image instance alignment。模型从来没被显式训练过"这两个 image token 是同一个 instance"这种 signal。

### Angle 3: Training Data 的任务偏置

主流 instruction tuning 数据(VQA, captioning, conversational)很少要求"link same instance across views"。MMDU / MuirBench 是 multi-image,但任务是 story / retrieval,不是 instance re-identification。

所以 VLM 的 linking 能力是从 face/person 数据 + general vision foundation model **emergent** 来的,质量参差,没人 guarantee。

### Angle 4: Language Bottleneck

这是 paper 最 sharp 的 finding 的根。Visual feature → text → matching 是两步有损:
- Step 1: vision encoder → image tokens
- Step 2: LLM 把 image tokens 翻译成自然语言描述
- Step 3: LLM 用语言做匹配

Step 2 是 bottleneck。"dinosaur with sunglasses" 是一个 vocab-limited compression,丢失了"holding skateboard vs keyboard"这种 sub-cue。CoT 把整个 reasoning 链路 push 到 language domain,放大了这个 loss。

GC 上 CoT 有效,因为 GC 的 cue (vase, chandelier) 是语言可表达的 noun。OC-Grp 失败,因为 sub-cue 是 fine-grained visual pattern。PC 彻底失败,因为 face 是 abstract pattern,语言几乎 zero expressive power。

这就是 Karpathy 你之前讲过的 "language is a lossy compression of thought" 的一个 concrete example。

### Angle 5: Resolution Sensitivity 的含义

Table 3,Qwen2.5-VL-7B 和 InternVL2.5-8B 在 ↓×2 / ×4 / ×8 / ×16 下:

| Resolution | Qwen2.5-VL OC-Cpr | Qwen2.5-VL PC-Grp |
|---|---|---|
| Origin | 71.39 | 69.00 |
| ↓×2 | 64.17 | 70.00 |
| ↓×4 | 52.78 | 61.00 |
| ↓×8 | 43.33 | 52.00 |
| ↓×16 | 34.17 | 41.00 |

单调下降。**如果模型依赖 layout / coarse semantic shortcut,降分辨率不该让它崩**。它崩了,说明它确实依赖 fine-grained pixel-level cue。

深层含义:**VLM 的"视觉理解"很大程度上还是 pixel-level template matching,而非 abstraction**。人类看 16×16 的马赛克图也能粗略识别是不是同一个人,因为人类有 viewpoint-invariant 的 face representation。VLM 没有,所以分辨率一降就崩。

---

## Prompting 能救吗——三个故事

Paper 测了 CoT (language-side) 和 Visual Prompting (vision-side)。

### Story 1: GC 上 CoT 有效,VP-grid 看模型

GC 的 cue 是语言可表达的 noun (vase, chandelier)。CoT 让模型先列出每张图的元素,再比较,再结论——这个流程帮模型结构化逻辑,降错误率。

**CoT-special**(Table 23)是模仿人类视觉匹配流程的 4 步:
1. Understand question
2. Perceive(列出每张图元素)
3. Connect(比较推理)
4. Conclude

这个 prompt 让 InternVL2.5-8B 在 GC 上涨 25%+,比普通 CoT 的 13% 翻倍。

**VP-grid** 是在图上叠点阵 + 三维坐标 (image_idx, col, row)。结果很分裂:
- Qwen2.5-VL-7B 掉 20%。Case study(Figure 16):模型识别出"vest at (2,5,3)"但当成第一张图的坐标,尽管 prompt 明确说 image_idx=2 是第二张图。**模型有 spatial indexing 的 comprehension 缺陷**——把坐标当 token 处理,没真正建立 "coordinate system in vision" 的 grounding。
- GPT-4o 涨 10%。Case study(Figure 17):识别 cat's nose 在 (1,2,4) 和 (2,2,4),正确 deduce "nose changed from pink to black"。

**这暗示 spatial-coordinate-grounding 是 emerging ability,scale 起来才行**。7B 模型规模 + 训练数据的复合短板。

### Story 2: OC-Grp 上 CoT 有害,VP-zoom-o 看模型

OC-Grp 要求 group similar objects based on fine-grained detail。

**CoT 在 OC-Grp 上反而有害**。Case(Figure 18):InternVL2.5-26B 的 CoT response 说"image 2 和 3 都有 dinosaur with sunglasses",选 C) 2 and 3。但 image 3 的 dinosaur 拿 keyboard,image 2 拿 skateboard——正确答案是 D) None。

失败原因:
1. **Insufficient visual cue coverage**:CoT 不强制 systematic verify 所有细节
2. **Language inconsistency**:"dinosaur with sunglasses" 这种 high-level 描述丢失了 disambiguating cue

这是 language bottleneck 的典型案例:**把视觉细节编码成自然语言,有损压缩导致信息丢失**。

**VP-zoom-o** 用 Grounded-SAM 检测 bounding box,crop 出 object 局部放大。结果:GPT-4o 涨分,其他模型至少不掉。这是一个 free lunch——没有信息丢失,只是 reformatting。弱模型不涨可能是 detector 不够准或 vision encoder 不够强去利用 crop,但至少不会因为 visual prompt 而崩。

### Story 3: PC 上 CoT 和 VP 都失败

**Facial feature 是 highly abstract visual pattern**。
- CoT 失败:语言很难描述 face geometry(脸型、眼距、鼻梁、嘴唇轮廓)的细微差异。你不可能用文字描述"两个人的 nose bridge curvature 差 0.5mm"。
- VP-zoom-p 失败:crop 出 face 局部后,VLM 的 vision encoder 本身不够强去区分细微 facial difference。这不是 prompt 能解决的,是 **base visual capability 的 hard limit**。

这是 paper 最 sharp 的 finding:**有些 visual cue 是 language-inaccessible 的,你只能靠 vision encoder 本身的能力**。

---

## 这意味着什么——未来方向

### Paper 自己总结的三条

1. **Strengthen Fundamental Visual Capabilities**: 不只是 scale up ViT。需要 viewpoint-invariant representation learning。可能方向:metric-learning loss (instance triplet)、synthetic identity data(一个人 100 个 angle 渲染做 contrastive)、video identity pretraining(MoCo / DINO 在 video frame 上的 instance discrimination)。

2. **Balance Language-Based Reasoning in Vision-Centric Tasks**: 建立 "when language helps / hurts" 的原则。Paper 的 boundary:helps 当 cue 是 noun-level 可描述时;hurts 当 cue 是 fine-grained sub-element;fails 当 cue 是 abstract pattern。未来 work 应该探索 non-language intermediate reasoning,比如 visual sketchpad / spatial reasoning in pixel space。

3. **Evolve Vision-Text Training Paradigm**: 当前 instruction tuning 是 "vision → text → answer"。需要演化到 "vision → visual structure → visual reasoning → answer",让模型能在 vision domain 内部组织 cue 之间的关系。

### 我额外想到的方向

1. **Bidirectional cross-image binding**:在 attention 里加显式的 cross-image identity matching head,像 DETR 的 bipartite matching 那样。

2. **Resolution-adaptive tokenization**:不同 patch 用不同 resolution,face / object 局部 high-res,background low-res。这其实就是 VP-zoom-o 的训练时版本——不用 inference 时 crop,而是训练时就让模型学会 dynamic resolution allocation。

3. **Identity-aware pretraining objective**:把 "is this the same instance as before" 当作 MLM-style auxiliary task 在大规模无标签视频上预训练。视频天然有 instance continuity signal。

4. **Visual Scratchpad**:让模型在 vision domain 内部"画"——不是输出 text CoT,而是输出 visual annotation(比如框出 face 的 keypoint),再基于这些 annotation 做推理。这绕过了 language bottleneck。

---

## Reference Links 汇总

**Paper**: 搜 "VLM2-Bench Zhang Fung HKUST 2025"

**核心相关工作**:
- Eyes Wide Shut (Tong et al., 2024): https://arxiv.org/abs/2407.06481
- Personalized VIT (Pi et al., 2024b): https://arxiv.org/abs/2410.07113
- MuirBench (Wang et al., 2024a): https://arxiv.org/abs/2406.09411
- MIBench (Liu et al., 2024a): https://arxiv.org/abs/2407.15272
- HEMM (Liang et al., 2024a): https://arxiv.org/abs/2407.03418
- NaturalBench (Li et al., 2024a): https://arxiv.org/abs/2410.14669
- Grounded-SAM (Ren et al., 2024): https://arxiv.org/abs/2401.14159
- CLIPScore (Hessel et al., 2021): https://arxiv.org/abs/2104.08718

**模型**:
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- InternVL2.5: https://arxiv.org/abs/2412.05271
- LLaVA-OneVision: https://arxiv.org/abs/2408.03326
- mPLUG-Owl3: https://arxiv.org/abs/2408.04840
- GPT-4o system card: https://arxiv.org/abs/2410.21276
- LongVA: https://arxiv.org/abs/2406.16852

**Cognitive Science 背景**:
- Bruce & Young (1986): https://doi.org/10.1111/j.2044-8295.1986.tb02197.x
- Treisman & Gelade (1980): https://doi.org/10.1016/0010-0285(80)90005-5

---

## 最后一句

VLM2-Bench 揭示了一个 fundamental gap:**VLM 有 knowledge 但缺 perception mechanism**。人类 trivial 的 visual cue linking,最好的商业模型落后 30%+,7B 开源模型在 chance level 附近。Resolution ablation 证明这是真实 perception 缺陷,CoT 在 abstract cue (face) 上失效证明 language reasoning 不是万能药。这条 research line 应该会引发一波 "visual-side reasoning" 的工作——visual scratchpad、cross-image binding、identity-aware pretraining 这些方向。

---

# VLM2-Bench: 评估 VLM 的 Implicit Visual Cue Linking 能力

## 1. Core Question & Motivation

这篇 paper 的核心问题非常具体:**当 VLM 处理多图/视频时,它能不能像人一样,在不知道"这是谁"的前提下,纯靠视觉特征 (facial geometry, clothing pattern, object texture) 把"同一个 instance"在不同 frame/image 里 link 起来?**

人类这个能力是 Bruce & Young (1986) 那一套 face recognition 模型 + Treisman & Gelade (1980) feature-integration theory 的范畴 —— 不需要 identity 知识,只需要 feature matching。作者 argue 现有 multi-image/video benchmark 都没真正测这个:

| Existing benchmark 缺陷 | VLM2-Bench 的对应 |
|---|---|
| 不要求跨图 link visual cue (e.g. ActivityNet-QA, MMDU) | 显式 require |
| 依赖外部 knowledge (e.g. MIBench, MMMU) | 不依赖,纯视觉 |
| 做的是 abstract 视觉比较 (e.g. TempCompass, Img-Diff) | 做 specific cue matching |
| 是 retrieval (e.g. MuirBench) | 是 direct association |

这是 Karpathy 你会喜欢的 angle —— **不是 knowledge-capacity gap,是 perception-mechanism gap**。

arXiv 链接(基于 paper 信息推测): https://arxiv.org/abs/2502.something — 实际可搜 "VLM2-Bench Zhang Fung HKUST"

---

## 2. Benchmark 结构:3 Categories × 9 Subtasks, 3060 QA pairs

```
VLM2-Bench
├── General Cue (GC)         — 960 QA (T/F)
│   ├── Matching (Mat)        — 520 (260 pairs)
│   └── Tracking (Trk)       — 440 (220 pairs)
├── Object-centric Cue (OC)  — 1280 QA
│   ├── Comparison (Cpr)     — 720 (360 T/F pairs)
│   ├── Counting (Cnt)       — 360 (numerical)
│   └── Grouping (Grp)       — 200 (MC)
└── Person-centric Cue (PC)  — 820 QA
    ├── Comparison (Cpr)     — 400 (200 T/F pairs)
    ├── Counting (Cnt)       — 120 (numerical)
    ├── Grouping (Grp)       — 100 (MC)
    └── Video Identity Describing (VID) — 200 (open-ended)
```

Inter-annotator agreement: **Fleiss' κ = 0.983** (3 annotators, 98.74% consensus),数据质量极硬。

### Subtask 直觉

- **GC-Mat**: 给两张图,问"X 在两张图里是同一个吗"。模型要 link 一致 cue,排除变化 cue。
- **GC-Trk**: 给两张图,问"这个 cue 在第二张里变成了什么"。模型要跨 frame 跟踪 transform。
- **OC/PC-Cpr**: T/F 配对验证 —— "pets in these images are the same pet" (T) vs "are not the same pet" (F)。必须两题都对才算对。这是为了抵消 model 的 affirmative bias (Goyal et al. 2017 VQA-CP 的发现)。
- **OC/PC-Cnt**: 数 unique instance 数量。考验"不重复计数同一 object"。
- **OC/PC-Grp**: MC 题,问"哪几张图是同一个 object/person"。选项含 "None" 作为干扰项。
- **PC-VID**: 视频里描述人。视频有两种结构:
  - `P → ¬P`: 两段不同人的 clip 拼接,看模型能不能区分。
  - `P → ¬P → P`: 三段拼接,看模型能不能把第一个和第三个 P link 起来。

---

## 3. Data Construction (细节)

### 3.1 GC:Salient Sampling 控制难度

复用 image editing 数据集,每条样本有 $(I_{ori}, I_{edit}, \mathcal{P})$,其中 $\mathcal{P}$ 是 edit instruction。

三阶段 pipeline:

**Stage 1: Manual Screening** —— 人工验证 $\mathcal{P}$ 的 correctness / uniqueness / clarity。

**Stage 2: Salient Sampling** —— 自动剔除"太简单"的样本。核心公式 (Eq. 1):

$$
S_{\text{salient}} = \frac{1}{|\mathcal{P}|} \sum_{i=1}^{|\mathcal{P}|} \log P_\theta(p_i \mid C \cup p_{<i})
$$

变量含义:
- $\mathcal{P} = \{p_1, p_2, ..., p_{|\mathcal{P}|}\}$: edit instruction $\mathcal{P}$ 分词后的 token 序列,$|\mathcal{P}|$ 是 token 数。
- $p_i$: 第 $i$ 个 token。
- $p_{<i}$: 第 $i$ 个 token 之前的所有 token(prefix)。
- $C = \mathcal{T}(Cap_{ori}, Cap_{edit})$: 把 VLM 生成的两张图 caption 拼到 template $\mathcal{T}$ 里形成的 text context。
- $P_\theta$: 参数为 $\theta$ 的 LM(用 Llama3-8B)的条件概率。
- $S_{\text{salient}}$: 平均 per-token log-prob,本质是 instruction 在已知 caption 下的"可预测性"。

直觉:**如果 $S_{\text{salient}}$ 高,说明从 caption 就能猜出 $\mathcal{P}$ 是什么,意味着 edit 太 salient(语言已经泄露了视觉差异),样本太简单。保留 $S_{\text{salient}} < -2.0$ 的样本**,即模型从 caption 推不出 edit、必须看图才能解的"难样本"。这是一个非常 elegant 的 difficulty filter,与 contrastive difficulty 有关。

**Stage 3: Pair-wise Answer Generation** —— 双层级 cue 抽取:
- (a) 从 VLM 生成的 caption parse 出 cue(open-set detector 处理不了 OOD 场景)
- (b) open-set detector (GRiT, Wu et al. 2022) 抽 fine-grained cue(VLM 可能漏掉)
合起来 prompt LLM 生成 (positive answer, negative answer) 一对。

### 3.2 OC:Meta-Object 结构化采集

8 个 object categories:Pet, Plush, Bag, Book, Cup, Shirt, Shoes, Toy。每类 5 个 main meta-objects,每个 meta-object 4 张图(同一物体不同角度/场景)。每个 main meta-object 配 4 个 distractor meta-objects(同 category 但不同 instance,共享某些 visual cue 制造 confusion)。

图像序列构造(360 个序列):

| 长度 | 类型 | 描述 | Cpr GT | Cnt GT | Grp GT |
|---|---|---|---|---|---|
| 2 | AA | 同 obj 2 图 | T | 2 | — |
| 2 | AB | 主 + 干扰 | F | 1 | — |
| 3 | AAA | 同 obj 3 图 | T | 3 | — |
| 3 | AAB | 主2+干扰1 | F | 2 | [I_i, I_j] |
| 3 | ABC | 主1+干扰2 | F | 3 | [] |
| 4 | AAAA | 同 obj 4 图 | T | 4 | — |
| 4 | AAAB | 主3+干扰1 | F | 2 | [I_i,I_j,I_k] |
| 4 | AABC | 主2+干扰2 | F | 3 | [I_i,I_j] |
| 4 | ABCD | 主1+干扰3 | F | 3 | [] |

### 3.3 PC:CLIP-based Distractor Selection

按 (Asian/Black/White) × (Male/Female) = 6 组 × 5 meta-humans × 4 张图 = 120 张图。Age / makeup / styling 都控制接近(避免"明显不同人"这种 trivial 样本)。

Distractor 选择:**计算每张图的 CLIP embedding 存到 reference base,需要 distractor 时做 image-to-image similarity search,选来自不同 meta-human 但最相似的图**。这是关键设计 —— 让 distractor 在 race/gender/clothing/scene 上尽量接近,只在 fine-grained facial geometry 上不同。

为什么 object 需要 dedicated distractor(同 category)而 person 不需要?因为 object 8 个类差异巨大(pet vs bag),需要 type-specific distractor;person 只有一个"category",任意 meta-human 都可作另一个的 distractor,CLIP 选最相似即可。

---

## 4. Metrics(公式详解)

### 4.1 T/F: Paired Accuracy (Eq. 2)

$$
Acc_{\text{pair}} = \frac{\sum_{i=1}^{N} (T_i^+ \cap F_i^-)}{N}
$$

变量:
- $N$: T/F pair 总数
- $T_i^+$: 第 $i$ 个 pair 里 positive 题(GT=T)答对的 indicator
- $F_i^-$: 第 $i$ 个 pair 里 negative 题(GT=F)答对的 indicator
- $\cap$: 逻辑 AND,两个都答对才算这一 pair 通过

**这是关键设计** —— 抵消 affirmative bias。如果模型只会猜 T,positive 答对但 negative 全错,Acc=0。如果你不做 paired evaluation,Qwen2-VL-7B 在 OC-Cpr 上能拿 68% (Table 1),看起来不错;但这是 bias-driven 的虚高。

### 4.2 Numerical: Weighted Normalized Error (Eq. 3, 4)

$$
\epsilon_i = \frac{|\hat{N}_i - N_i|}{\max(N_i - 1, \; N_i^{\text{img}} - N_i)}
$$

变量:
- $\hat{N}_i$: 模型预测的 unique 数
- $N_i$: ground-truth unique 数
- $N_i^{\text{img}}$: 输入图总数(序列长度)
- 分母的 $\max(N_i-1, N_i^{\text{img}}-N_i)$: 这是 error 的"动态范围"。$N_i-1$ 是"漏数"方向的最大偏差;$N_i^{\text{img}}-N_i$ 是"重复计数"方向的最大偏差。取 max 是为了 normalize 时用更大的那个,使误差不被分母放大成虚高。

直觉:序列长度 4,GT=2,模型答 4(全重复)。$\epsilon = |4-2|/\max(1, 2) = 2/2 = 1$(满误差);模型答 3,则 $\epsilon = 1/\max(1,2) = 0.5$。

$$
Acc_{\text{num}} = 1 - \frac{1}{n}\sum_{i=1}^{n} w_i \cdot \epsilon_i^\alpha
$$

- $n$: case 总数
- $w_i = \max(\{N_i^{\text{img}}\}_{i=1}^n) / N_i^{\text{img}}$: 长度惩罚权重。$L_{\max}=4$,如果某 case 只有 2 张图,$w_i = 4/2 = 2$。**短序列错更严重**。
- $\alpha$: error amplification factor(paper 没明确给值,推测 α≥1,用来惩罚大误差)。

### 4.3 Open-ended: GPT-4o-as-Judge + 人工验证

VID 用 GPT-4o 打分,人工抽 180 个核对,准确率 178/180 = 98.89%。这种 LLM-as-judge 的 reliability 检验在当前 paper 里越来越标配。

### 4.4 Chance Level 计算(附录 D)

**GC-Mat / Trk**:每个 pair 是独立的 (positive, negative),每个 1/2 chance,都答对 $= (1/2)^2 = 25\%$。

**OC/PC-Cpr**:positive 题是"X is Y"(GT=T),negative 题是"X is not Y"(GT=F)。**这两题逻辑上等价**,所以模型对一题,如果 free of bias 必对另一题。Chance = $1/2 = 50\%$。

**Cnt**:随机猜测 $\hat{N}_i \sim \text{Uniform}\{1, ..., L\}$,$L$ 是序列长度。期望:

$$
E(L) = 1 - \frac{1}{L^2} \sum_{N=1}^{L} \sum_{\hat{N}=1}^{L} w(L) \cdot \epsilon(N, \hat{N})^\alpha
$$

其中 $w(L) = L_{\max}/L$,$L_{\max}=4$。按 OC-Cnt 的 length 分布(80×L2 + 120×L3 + 160×L4)/360 ≈ 34.88%。PC-Cnt 按 (30×L2 + 25×L3 + 65×L4)/120 ≈ 34.87%。

---

## 5. 主要结果:人类 94.44%,最好的模型 59.57%

Table 1 关键数字:

| Model | Avg | Δ_human |
|---|---|---|
| Human | **94.44** | 0 |
| Chance | 33.72 | -61.44 |
| GPT-4o-2024-08-06 | 59.56 | -34.88 |
| Claude-3.7-sonnet | 59.57 | -34.87 |
| Qwen2.5-VL-7B | 55.86 | -38.58 |
| GPT-4o-2024-11-20 | 55.73 | -38.71 |
| InternVL2.5-26B | 48.58 | -45.86 |
| Gemini-2.0-flash | 28.33 | **-66.11**(低于 chance!) |
| LongVA-7B | 24.95 | -69.49(显著低于 chance) |

**两个尖锐观察**:

1. **Gemini-2.0-flash 和 LongVA-7B 低于 chance level**。这意味着它们有 system-level bias(比如总答 T 或者总答某固定 count)。在 Cpr 这种 50% chance 的任务上掉到 21-49%,说明 bias 极严重。这是一个反向 sanity check —— benchmark 真的能区分"有能力"和"假装有能力"。

2. **GPT-4o 新版本 (2024-11-20) 反而比老版本 (2024-08-06) 差**:Avg 55.73 vs 59.56。具体看 GC-Mat 从 37.45 跌到 18.53,GC-Trk 从 39.27 跌到 29.68。这暗示 RLHF/post-training 在优化 conversational 能力时可能损害了细粒度 visual perception。这与 Tong et al. 2024 "Eyes Wide Shut" 的发现一致 —— 后训练经常伤害 perception。

### 5.1 Finding II:Mat 在 swap 上错,Trk 在 add/remove 上错

Table 2 breakdown。这个 pattern 非常有 intuition 价值:

- **Mat 的 swap 错误率高**:swap 意味着"两个 cue 互换"。模型必须先 link 所有其他 cue 才能确定哪两个被 swap 了 —— 这是一个 **second-order reasoning**。模型擅长 detect "something changed",不擅长 "which two swapped"。
- **Trk 的 add/remove 错误率高**:Trk 要求跟踪一个"只在一图出现"的 cue。模型不知道"它在另一图里 absence"这件事意味着什么 —— 缺失 cue 的 link 是更难的,因为 attention 没有锚点。

### 5.2 Finding III:PC > OC

Qwen2.5-VL / InternVL2.5-8B / 26B 三个模型:
- Cpr:PC 比 OC 高 7.65%
- Cnt:高 9.75%
- Grp:高 11.83%

假设原因:训练数据里 person 有 explicit name 作为 anchor(Pi et al. 2024b "Personalized Visual Instruction Tuning", HumanVLM 数据集),模型隐式学到 face embedding。Object 只用 category name ("cat", "bag"),没有 instance-level anchor。这与 recognition literature 里 face-specific processing (Kanwisher 的 FFA) 的争论 echo。

### 5.3 Sanity Check:Resolution Ablation

Table 3,Qwen2.5-VL-7B 和 InternVL2.5-8B 在 ↓×2 / ×4 / ×8 / ×16 下:

- Qwen2.5-VL OC-Cpr: 71.39 → 64.17 → 52.78 → 43.33 → 34.17(单调下降)
- Qwen2.5-VL PC-Grp: 69.00 → 70.00 → 61.00 → 52.00 → 41.00

这是关键的 validity check。**如果模型依赖 layout / coarse semantic shortcut,降分辨率不该让它崩**。它崩了,说明它确实依赖 fine-grained pixel-level cue。这反过来证明 benchmark 测的是真东西,而非 spurious correlation。

---

## 6. Prompting 实验与 Findings IV–VIII

### 6.1 GC:CoT-normal, CoT-special, VP-grid

**CoT-special**(Table 23)设计模仿人类视觉匹配流程,4 步:
1. Understand question
2. Perceive (列出每张图的元素)
3. Connect (比较、推理)
4. Conclude

**Finding IV**: CoT 在 GC 几乎都涨分。语言 reasoning 帮模型**结构化逻辑流**,降低错误率。

**Finding V**: **VP-grid 在 Qwen2.5-VL 上掉 20%,在 GPT-4o 上涨 10%**。VP-grid 是在图上叠点阵 + 三维坐标 (image_idx, col, row)。

为什么 Qwen2.5-VL 失败?Case study (Figure 16):模型识别出"vest at (2,5,3)"但把它当成第一张图的坐标,尽管 prompt 明确说 image_idx=2 是第二张图。**模型有 spatial indexing 的 comprehension 缺陷** —— 它把坐标当 token 处理,但没真正建立 "coordinate system in vision" 的 grounding。这是 7B 模型规模 + 训练数据的复合短板。

GPT-4o 成功 (Figure 17):识别 cat's nose 在 (1,2,4) 和 (2,2,4),正确 deduce "nose changed from pink to black"。**模型容量 → structured multi-modal prompt comprehension**。这暗示 spatial-coordinate-grounding 是 emerging ability,scale 起来才行。

### 6.2 OC:CoT, VP-zoom-o

VP-zoom-o 用 Grounded-SAM 检测 bounding box,crop 出 object 局部放大。

**Finding VI**: **CoT 在 OC-Grp 上反而有害**。Case (Figure 18):InternVL2.5-26B 的 CoT response 说"image 2 和 3 都有 dinosaur with sunglasses",选 C) 2 and 3。但 image 3 的 dinosaur 拿 keyboard,image 2 拿 skateboard —— 正确答案是 D) None。

失败的两个原因:
1. **Insufficient visual cue coverage**:CoT 不强制 systematic verify 所有细节。
2. **Language inconsistency**:"dinosaur with sunglasses" 这种 high-level 描述丢失了"holding skateboard vs keyboard"这种 disambiguating cue。

这是 language bottleneck 的典型案例:**当你把视觉细节编码成自然语言,有损压缩导致信息丢失**,grouping 这种依赖 fine-grained detail 的任务就崩。

**Finding VII**: **VP-zoom-o 对 GPT-4o 涨分,对其他模型至少不掉**。Crop 出 object 局部放大 → 突出 object-centric cue → 去掉 background 干扰。这是一个 free lunch,因为没有信息丢失,只是 reformatting。弱模型不涨可能是 detector 不够准或 vision encoder 不够强去利用 crop,但至少不会因为视觉提示而崩。

### 6.3 PC:CoT, VP-zoom-p

VP-zoom-p 用 face detector (Geitgey 2016 的 dlib-based MTCNN/HOG detector) crop face 局部。

**Finding VIII**: **CoT 和 VP-zoom-p 在 PC 都失败**。

直觉:**Facial feature 是 highly abstract visual pattern**。
- CoT 失败:语言很难描述 face geometry(脸型、眼距、鼻梁、嘴唇轮廓)的细微差异。你不可能用文字描述"两个人的 nose bridge curvature 差 0.5mm"。
- VP-zoom-p 失败:crop 出 face 局部后,VLM 的 vision encoder 本身不够强去区分细微 facial difference。这不是 prompt 能解决的,是 **base visual capability 的 hard limit**。

这是 paper 最 sharp 的 finding:**有些 visual cue 是 language-inaccessible 的,你只能靠 vision encoder 本身的能力**。这指向 "vision-text training paradigm 需要演进" 的 takeaway。

---

## 7. 为什么 VLM 在 Linking 上这么差?— First Principles 分析

Karpathy 你应该会问:从 architecture 角度,VLM 缺什么?让我从几个 angle 拆:

### 7.1 ViT Patch Tokenization 的非不变性

ViT 把图切成 patch,每 patch 一个 token。同一个 person 在不同 angle / lighting / pose 下,patch token 序列在 embedding space 里的位置不连续。**ViT 没有 viewpoint-invariant 的归纳偏置**,要靠 training data 学。

对比:face recognition 网络 (ArcFace, CosFace) 用 metric learning + triplet loss 显式训练 identity-invariant embedding。VLM 没有这种 training signal。

这解释了 PC > OC(person 数据有 implicit name anchor 当 metric learning 的 label)+ PC-VP-zoom-p 失败(face crop 之后没有 identity-tuned encoder)。

### 7.2 Cross-Attention 的单向性 + 长程依赖

VLM 的 image tokens 在 LLM context 里是 sequence。要 link "image 1 的 cue A" 和 "image 4 的 cue A",模型要在 self-attention 里跨长距离 attend。Multi-image scenario 下,image tokens 可能隔几千 token。LLM 的 long-range attention 在 softmax 上会衰减,sharp 度不够。

更关键的:**没有 bidirectional cross-image binding 机制**。CLIP-style contrastive training 只做 image-text alignment,不做 image-image instance alignment。

### 7.3 Training Data 的任务偏置

Personalized VIT (Pi et al. 2024b) 是少数显式训练 instance identity 的工作。主流 instruction tuning 数据(VQA, captioning, conversational)很少要求"link same instance across views"。MMDU / MuirBench 是 multi-image,但任务是 story / retrieval,不是 instance re-identification。

所以 VLM 的 linking 能力是从 face/person 数据 + general vision foundation model **emergent** 来的,质量参差。

### 7.4 Language Bottleneck

这是 Finding VI / VIII 的根。Visual feature → text → matching 是两步有损:
- Step 1: vision encoder → image tokens
- Step 2: LLM 把 image tokens 翻译成自然语言描述
- Step 3: LLM 用语言做匹配

Step 2 是 bottleneck。"dinosaur with sunglasses" 是一个 vocab-limited compression,丢失了 skateboard / keyboard 这种 sub-cue。CoT 把整个 reasoning 链路 push 到 language domain,就放大了这个 loss。

GC 上 CoT 有效,因为 GC 的 cue (vase, chandelier) 是语言可表达的 noun。OC-Grp 失败,因为 sub-cue 是 fine-grained visual pattern。PC 彻底失败,因为 face 是 abstract pattern,语言几乎 zero expressive power。

这是一个非常重要的 takeaway:**language reasoning 不是 vision-centric 任务的 universal booster**。Karpathy 你之前讲过 "language is a lossy compression of thought",这里就是个 example。

### 7.5 Resolution Sensitivity 的含义

Table 3 的 ablation 还有一个深层含义:**VLM 的视觉能力是 "raw pixel dependent" 而非 "concept abstraction dependent"**。人类看 16×16 的图也能识别是不是同一个人(粗略),因为人类有 viewpoint-invariant 的 face representation。VLM 没有,所以分辨率一降就崩。

这意味着 VLM 的"视觉理解"很大程度上还是 **template matching at pixel level**,而非 abstraction。这与 Tong et al. 2024 "Eyes Wide Shut" 的核心论点一致。

---

## 8. Takeaways 和 Commentary

Paper 自己总结三条,我觉得都对但需要展开:

### (i) Strengthen Fundamental Visual Capabilities

不只是 "scale up ViT"。需要的是:**viewpoint-invariant representation learning**。可能的方向:
- 在 VLM 训练里加 metric-learning loss (instance triplet)
- 用 synthetic identity data (一个人 100 个 angle 渲染出来做 contrastive)
- 引入 video identity pretraining(MoCo / DINO 在 video frame 上的 instance discrimination)

### (ii) Balance Language-Based Reasoning in Vision-Centric Tasks

需要建立 "when language helps / hurts" 的原则。Paper 的实验给出的 boundary:
- **Helps**: 当 visual cue 是 noun-level 可描述时(GC-Mat 的 vase)
- **Hurts**: 当 cue 是 fine-grained sub-element(OC-Grp 的 skateboard)
- **Fails**: 当 cue 是 abstract pattern(PC 的 face)

未来 work 应该探索 **non-language intermediate reasoning**,比如 visual sketchpad / spatial reasoning in pixel space。

### (iii) Evolve Vision-Text Training Paradigm

当前 instruction tuning 是 "vision → text → answer"。需要演化到 "vision → visual structure → visual reasoning → answer",让模型能在 vision domain 内部组织 cue 之间的关系。这个方向接近 SuHa (Surface Hand-Drawn) 之类的 visual scratchpad 工作。

### 我额外想到的几个方向

1. **Bidirectional cross-image binding**:在 attention 里加显式的 cross-image identity matching head,像 DETR 的 bipartite matching 那样。
2. **Resolution-adaptive tokenization**:不同 patch 用不同 resolution,face / object 局部 high-res,background low-res。这其实就是 VP-zoom-o 的训练时版本。
3. **Identity-aware pretraining objective**:把 "is this the same instance as before" 当作 MLM-style auxiliary task 在大规模无标签视频上预训练。

---

## 9. Reference Links

**Paper 本身**: 搜索 "VLM2-Bench Karpathy Fung HKUST 2025" 应该能找到 arXiv 链接。

**相关工作**:
- Eyes Wide Shut (Tong et al., 2024): https://arxiv.org/abs/2407.06481 — 同样研究 VLM visual shortcomings
- Personalized VIT (Pi et al., 2024b): https://arxiv.org/abs/2410.07113 — person identity tuning
- MuirBench (Wang et al., 2024a): https://arxiv.org/abs/2406.09411 — multi-image understanding
- MIBench (Liu et al., 2024a): https://arxiv.org/abs/2407.15272 — multi-image benchmark
- HEMM (Liang et al., 2024a): https://arxiv.org/abs/2407.03418 — holistic multimodal eval
- NaturalBench (Li et al., 2024a): https://arxiv.org/abs/2410.14669 — adversarial samples
- Grounded-SAM (Ren et al., 2024): https://arxiv.org/abs/2401.14159 — 用在 VP-zoom-o
- CLIPScore (Hessel et al., 2021): https://arxiv.org/abs/2104.08718 — 用在 PC distractor selection
- ImageNet Image Editing (Wei et al. 2024 / Ku et al. 2023): GC 数据源

**模型**:
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923 (推测)
- InternVL2.5: https://arxiv.org/abs/2412.05271
- LLaVA-OneVision: https://arxiv.org/abs/2408.03326
- mPLUG-Owl3: https://arxiv.org/abs/2408.04840
- GPT-4o system card: https://arxiv.org/abs/2410.21276
- LongVA: https://arxiv.org/abs/2406.16852

**Cognitive Science 背景**:
- Bruce & Young (1986) face recognition model: https://doi.org/10.1111/j.2044-8295.1986.tb02197.x
- Treisman & Gelade (1980) feature-integration theory: https://doi.org/10.1016/0010-0285(80)90005-5

---

## 一句话总结

VLM2-Bench 揭示了 VLM 的一个根本性短板:**视觉 cue linking 能力** —— 一个人类 trivial、不需 knowledge 的能力。最好的商业模型落后人类 30%+,7B 开源模型在 chance level 附近。Resolution ablation 证明这是真实的 perception 缺陷,CoT 在 abstract cue (face) 上失效证明 language reasoning 不是万能药。这条 research line 应该会引发一波 "visual-side reasoning" 的工作,类似 visual scratchpad / cross-image binding 之类。
