---
source_pdf: Vidi2.5.pdf
paper_sha256: 570e4489152654df6d27ebf105a337f88167789ab1e92cbc808aae1808b463c5
processed_at: '2026-08-13T01:09:18-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲Vidi2.5这篇paper

## 一句话概括

ByteDance这个team做了个12B的多模态模型，叫Vidi2.5，它干了一件别人没干成的事：**给一段视频和一句自然语言query，直接端到端输出一串"时间戳+bounding box"对，把query描述的东西在时空两个维度上都定位出来**。这个东西他们叫"tube"，就是一个object在视频中随时间移动的轨迹。

这个能力有什么用？想象你剪一个30分钟的视频，说"找出那个球员被抬上救护车的片段"，模型不光告诉你这个片段在视频的什么时间位置，还告诉你每一帧里那个球员（或救护车）在画面的哪个位置。这就比单纯给个时间区间强太多了。

---

## Vidi2.5到底是什么

先把几个版本理清楚：

**Vidi2**: 基础模型，SFT训练出来，主要能力是spatio-temporal grounding (STG) + temporal retrieval (TR) + video QA。LLM backbone是Gemma-3的12B配置。

**Vidi2.5**: 在Vidi2基础上加了一层RL训练，用verifiable rewards（就是IoU这种有标准答案的reward），把STG的vIoU从32.57拉到38.64，提升6个点。

**Vidi2.5-Think**: 加了thinking能力的版本，类似OpenAI o1或DeepSeek-R1的思路，但reward是hybrid的——一部分是verifiable reward（比如定位准不准），一部分是LLM-as-judge（比如reasoning过程好不好）。这个版本专门攻复杂剧情理解。

**Vidi-Edit**: Vidi2.5经过post-training后的变种，专门用于video editing planning，输出结构化的编辑计划。

---

## 核心技术问题：怎么端到端做STG

### 以前的pipeline做法

传统做法是分阶段：先做temporal grounding找到时间段，再做spatial grounding在每帧找bbox。问题是error会累积，而且两个stage的训练objective不一致。

### Vidi2的做法

Vidi2把整个tube当成一个structured text sequence来生成。给定query，模型直接吐出一串：

```
(timestamp_1, [x0_1, y0_1, x1_1, y1_1])
(timestamp_2, [x0_2, y0_2, x1_2, y1_2])
...
```

这就把STG变成了一个sequence-to-sequence的生成问题，跟LLM本身的能力天然契合。

intuition是这样的：LLM本质就是个token生成器，你让它生成natural language也好，生成structured JSON也好，对它来说都是token sequence。那为什么不直接让它生成tube呢？关键在于训练数据要够，让模型学会"在什么时候该输出什么坐标"。

### Tube的数学表示

paper里用这个公式定义tube：

$$B(t) = \big( x_0(t), y_0(t), x_1(t), y_1(t) \big)$$

这里：
- $t$ 是时间（秒）
- $x_0(t), y_0(t)$ 是box左上角坐标
- $x_1(t), y_1(t)$ 是box右下角坐标
- 坐标都归一化到[0,1]

实际操作中按1 FPS采样，所以 $t$ 是离散整数秒。模型输出的就是 $B(0), B(1), B(2), ...$ 这串离散sample。

---

## 评测指标的设计——这是paper的一个亮点

### 为什么vIoU是primary metric

paper定义了几个metric，最核心的是vIoU。我们来拆解intuition：

先算每一帧的frame-level IoU：

$$\text{IoU}_t = \begin{cases} \text{bIoU}(B^{\text{pred}}(t), B^{\text{gt}}(t)), & t \in T_\cap \\ 0, & \text{otherwise} \end{cases}$$

意思就是：在predicted和gt时间重叠的部分($T_\cap$)，算bbox的IoU；不在重叠部分，直接给0。

然后累加这些IoU：

$$S = \sum_{t \in T_\cap} \text{IoU}_t$$

vIoU的定义：

$$\text{vIoU} = \frac{S}{|T_\cup|}$$

$|T_\cup|$ 是predicted和gt时间的并集长度。

intuition：分子 $S$ 是"在时间对齐的帧上，spatial对齐有多好"；分母 $|T_\cup|$ 是"时间总跨度"。如果时间对齐差（并集大），分母变大，vIoU变小；如果spatial对齐差，分子变小，vIoU也变小。**所以vIoU同时惩罚temporal misalignment和spatial inaccuracy**，是个综合性指标。

### vIoU-Int是complementary metric

$$\text{vIoU-Int.} = \frac{S}{|T_\cap|}$$

这里分母是 $|T_\cap|$（交集），只在时间对齐的帧上算spatial accuracy。

这个指标有什么用？它能帮你诊断模型的failure mode：

- vIoU低 + vIoU-Int高 → 时间对齐有问题，但一旦对齐了，spatial定位是准的
- vIoU低 + vIoU-Int低 → spatial定位本身就有问题

看Table 7的数据：
- Vidi2: vIoU 32.57, vIoU-Int 60.30 → 差距27.7，说明Vidi2主要错在temporal alignment
- Gemini 3 Pro: vIoU 4.61, vIoU-Int 16.59 → 两个都低，spatial本身就不行

这种诊断能力在iterative改进模型时非常有用。

---

## Benchmark设计——VUE-STG为什么重要

### 跟现有STG dataset的区别

paper里列了四个critical improvements：video duration, query format, annotation quality, evaluation metric。我觉得最关键的是duration和query format。

**Duration**: 现有STG dataset大多是短视频（几秒到几十秒），VUE-STG覆盖10秒到30分钟。这对real-world editing是必须的——你剪一个vlog或tutorial，动辄几分钟到几十分钟。

看Table 1的分布：
- Ultra-short (<1min): 126 videos, 0.82h
- Short (1-10min): 294 videos, 26.28h  
- Medium (10-30min): 562 videos, 177.69h

Medium档占了大部分，这才是真实场景。

**Query format的disambiguation**：

paper给了个好例子。如果query是"a player is loaded into an ambulance"，这是模糊的——你到底要定位player还是ambulance？

VUE-STG会改写成：
- "the ambulance which the player is being loaded into" （要ambulance）
- "the player who is loaded into the ambulance" （要player）

这种明确指向target object的query让评测更可靠，也让模型不用猜。

**Fragmented tubes**：真实视频里object会因为镜头切换、遮挡等原因消失再出现，VUE-STG保留这种fragmented annotation。这比强制连续的tube更真实。

---

## 为什么Vidi2在STG上碾压Gemini/GPT

看Table 7的overall数据：

| Model | vIoU |
|-------|------|
| Vidi2.5 | 38.64 |
| Vidi2 | 32.57 |
| Gemini 3 Pro | 4.61 |
| GPT-5 | 5.47 |
| Qwen3-VL-32B | 5.12 |

差距是30多个点，这很惊人。为什么？

### 1. 端到端 vs 拼接

Gemini、GPT这些模型虽然能处理video，但它们的spatio-temporal grounding本质上是"分段处理+格式化输出"。paper Section 4.1.1详细描述了怎么让它们公平参与评测：

- Gemini: 接受video URL，输出JSON with MM:SS timestamp + [0,1000] box
- GPT-5: 只能接受最多120帧image，<2min视频1 FPS采样，>2min视频uniform subsample到120帧
- Qwen3-VL: interleaved timestamp-image序列

GPT-5的120帧限制是致命的。一个30分钟视频1800帧，压到120帧意味着每15秒采一帧，temporal resolution太粗了。

### 2. Adaptive token compression

Vidi2重新设计了token压缩策略，对short video保留更多spatial细节，对long video做更aggressive的temporal downsampling。

这在Table 6/7的medium-length (10-30min)结果上体现得最明显：
- Vidi2.5 tIoU: 51.63
- Gemini 3 Pro tIoU: 21.13
- GPT-5 tIoU: 4.10

长视频上差距更大，说明token压缩策略对长视频理解至关重要。

### 3. Task-specific training data

Vidi2在训练时用了大量STG-specific数据：
- 用image-level spatial grounding dataset合成video STG pairs
- 真实video STG annotation

这些task-specific data让模型学会"在什么时间该输出什么坐标"，而general-purpose LMM没有这个训练信号。

---

## Vidi2.5的RL为什么有效

Vidi2.5在Vidi2基础上加RL，vIoU从32.57提升到38.64。关键在于**verifiable rewards**。

STG/TR这类任务的ground truth是明确的（time range + bbox），可以直接用IoU作为reward。这跟RLHF那种需要human preference或LLM-as-judge的情况完全不同——reward signal是clean的，没有噪声。

看具体提升：
- vR (recall): 36.32 → 44.71 (+8.39)
- tR: 59.80 → 68.00 (+8.20)

提升主要集中在recall上。intuition是：SFT训练的模型容易"保守"，只输出高confidence的prediction，导致漏掉一些ground truth segments。RL训练鼓励模型在保持precision的前提下提高coverage，所以recall提升明显。

---

## Vidi2.5-Think：thinking model的multimodal版本

### 设计思路

Vidi2.5-Think借鉴了DeepSeek-R1和OpenAI o1的inference-time scaling思路，但做了两个关键adaptation：

**Adaptation 1: Multimodal reasoning**

原来的thinking model主要处理math/code/logic，Vidi2.5-Think要处理video + audio + text的multimodal reasoning。训练数据覆盖：
- Video perception
- Audio understanding
- Narrative comprehension
- Professional filming/editing techniques

**Adaptation 2: Hybrid reward**

纯verifiable reward只适用于有明确答案的task。但plot understanding这种任务，reasoning process本身的质量也很重要。所以Vidi2.5-Think用：

- Verifiable rewards: 对temporal grounding、bbox定位这种
- LLM-as-judge: 对rationale quality、response quality、rationale-response consistency

这种hybrid design让模型既能在可验证的perception上精进，又能在open-ended reasoning上提升。

### 效果

在VUE-PLOT的Character track上：

| Model | tIoU | sIoU |
|-------|------|------|
| Vidi2.5-Think | 71.63 | 66.04 |
| Gemini 3 Pro | 50.68 | 13.24 |
| GPT-5 | 55.89 | 6.80 |

sIoU差距42点，这是huge gap。Vidi2.5-Think在speaker face tracking上碾压。

在Reasoning track上，Vidi2.5-Think总体跟Gemini 3 Pro持平(64.33 vs 64.58)，但在某些子task上有明显优势：

- Speech/Audio reasoning: 74.43 vs 66.03 (+8.40)
- Filming/Editing: 61.35 vs 50.92 (+10.43)

这种advantage来自Vidi的多模态训练（含audio）和task-specific的filming/editing知识训练。

---

## Vidi-Edit：从understanding到creation

### 任务定义

Vidi-Edit做的是video editing planning：给定raw assets（images + videos）+ optional user intent，生成一个结构化的editing plan。

这个plan包含四个aspect：
1. **Narrative structure**: 怎么选clip、怎么排列
2. **Voiceover content**: 旁白说什么、什么风格
3. **Audio attributes**: 音乐风格、mood、BPM、speaker
4. **Visual editing intent**: 转场、强调、风格指令

### 为什么planning和execution要分离

paper Section 5.4描述的pipeline很值得细看：

```
Raw Assets + User Intent
  ↓
Vidi-Edit (high-level planning)
  ↓
Editing Plan (structured text)
  ↓
Translation Stage:
  - Music attributes → music database retrieval
  - Voiceover → TTS
  - Visual intent → effect retrieval + parameterization
  ↓
Rendering System
  ↓
Final Video
```

intuition是这样的：high-level planning需要semantic reasoning能力（理解asset内容、构造narrative、设计pacing），这正好是LLM擅长的。但具体的music retrieval、TTS、video rendering，这些是specialized module的活，LLM不擅长也不需要擅长。

分离的好处：
- Vidi-Edit专注semantic reasoning，不需要学渲染细节
- Execution module独立优化，可以接入更好的music library或TTS engine
- Plan是可解释的，用户可以理解和修改
- 这种架构天然支持agent iteration——planning→execution→feedback→re-planning

### Editing Plan的具体格式

paper Figure 12给了一个example：

```json
{
  "scenario": "A pampered cat is on a mission to reclaim his food bowl...",
  "viral_strategy": "The story is framed as a 'sibling showdown'...",
  "pacing_strategy": "A quick, punchy setup...",
  "editing_script": [
    {
      "scene": "A fluffy white cat cautiously approaches an automatic feeder.",
      "voiceover": "POV: You just want your own food bowl.",
      "timestamps": ["asset 3, 00:02-00:10"],
      "music_tag": ["Lo-Fi Hiphop", "Chill Beats", "Funny"],
      "music_bpm": "Medium (90-110)",
      "tts_speaker": "Bestie",
      "visual_effect": "Begin with the cat cautiously approaching..."
    }
  ]
}
```

这种结构化plan的设计很聪明：
- 语义层面：scenario、strategy是high-level intent
- 执行层面：timestamps、music_tag、tts_speaker是具体的operational spec
- 每个scene独立可控，便于下游module解析

---

## 几个有意思的实验观察

### GPT-5的spatial能力其实不差，temporal是瓶颈

Table 7里，GPT-5的vIoU-Int是33.64，比Gemini的16.59和Qwen的18.47都高。但GPT-5的vIoU只有5.47，说明它的spatial localization能力其实不错，问题出在temporal alignment上。

这跟GPT-5只能处理120帧的限制直接相关。长视频被压到120帧后，temporal resolution太粗，模型难以准确定位time range。一旦time range对了，spatial定位反而还可以。

### RL主要提升recall

Vidi2.5 vs Vidi2的提升：
- vR: 36.32 → 44.71 (+8.39)
- vP: 44.56 → 47.26 (+2.70)

precision提升小，recall提升大。这说明SFT模型的问题是"过于保守，漏太多"，RL训练让模型学会在保持precision的同时更aggressive地覆盖ground truth。

### Long video是真正的分水岭

Table 6的medium (10-30min)档：
- Vidi2.5 tIoU: 51.63
- Gemini 3 Pro tIoU: 21.13
- GPT-5 tIoU: 4.10

长视频上，Vidi2.5的tIoU是Gemini的2.4倍，是GPT-5的12.6倍。这说明long video understanding不是简单scale up context window就能解决的，需要architecture-level的设计（如adaptive token compression）和task-specific training。

---

## 这篇paper给我什么intuition

### 1. End-to-end structured generation是可行且强大的

Vidi2把STG做成end-to-end text generation，效果远超pipeline。这个intuition可以推广到其他task：只要你能把output定义成structured token sequence，并且有足够training data，LLM就能学会直接生成。

### 2. Verifiable reward是RL的sweet spot

STG/TR这种有明确ground truth的任务，RL训练效果显著。reward signal clean，没有RLHF那种preference modeling的噪声。paper里Vidi2.5的提升主要来自verifiable reward RL。

### 3. Hybrid reward是thinking model的合理设计

纯verifiable reward只适用于有答案的task，纯LLM-judge又可能unreliable。Vidi2.5-Think的hybrid设计——对可验证部分用verifiable reward，对open-ended部分用LLM-judge——是个pragmatic的平衡。

### 4. Planning-execution separation是agent的合理架构

Vidi-Edit不做渲染，只做high-level planning。这种separation让模型能专注于semantic reasoning，下游module独立优化。这种架构可以推广到其他creative task——coding、design、writing都可以考虑类似的planning-execution分离。

### 5. Benchmark设计要贴近real-world

VUE-STG的四个关键improvement——duration、query format、annotation quality、evaluation metric——都是针对academic benchmark脱离real-world的痛点。特别是long video和disambiguated query，这俩在real editing场景里是必须的。

---

## 可能的extension方向

1. **Multi-object STG**: 当前主要单tube，multi-object scene下的query disambiguation更challenging
2. **Interactive editing**: Vidi-Edit目前是one-shot，可以扩展为iterative refinement with user feedback
3. **Cross-video STG**: 多video中同一object的grounding，对asset retrieval有用
4. **Hierarchical planning**: Vidi-Edit可以做成hierarchical——high-level narrative → mid-level scene → low-level cut
5. **Reward design for creative tasks**: editing plan quality怎么自动评估？LLM-judge可能不够，可以探索human preference model
6. **Continuous time representation**: 目前1 FPS discrete sampling，可以考虑neural temporal fields做continuous time

---

## 参考链接

- Vidi project page: https://bytedance.github.io/vidi-website/
- Vidi原版arXiv: https://arxiv.org/abs/2504.15681
- Vidi2 arXiv: https://arxiv.org/abs/2511.19529
- Gemma-3: https://arxiv.org/abs/2503.19786
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- OpenAI o1: https://openai.com/index/openai-o1-system-card/
- VERL: https://arxiv.org/abs/2409.19256
- VideoMME: https://arxiv.org/abs/2405.21075
- LVBench: https://arxiv.org/abs/2406.14380
- LongVideoBench: https://arxiv.org/abs/2410.10734

---

# Vidi2.5: Large Multimodal Models for Video Understanding and Creation 深度解析

## 1. Paper总体定位与核心贡献

这篇paper来自ByteDance的Intelligent Editing Team，是Vidi系列的第三个版本迭代。Vidi2.5的核心贡献可以归纳为四个层次：

**Model层**: Vidi2基础模型 + Vidi2.5 (RL-enhanced) + Vidi2.5-Think (inference-time scaling)

**Benchmark层**: 三个新评测集 - VUE-STG (spatio-temporal grounding), VUE-TR-V2 (temporal retrieval升级版), VUE-PLOT (plot understanding)

**Capability层**: 端到端spatio-temporal grounding (STG)，video QA，complex plot reasoning

**Application层**: Vidi-Edit用于video editing planning的实际应用

关键intuition：**Vidi2.5把spatio-temporal grounding从"分段pipeline"变成了"end-to-end text generation"**。给定一个text query，模型直接输出一串 (timestamp, [x0, y0, x1, y1]) tuple序列，整个tube作为一个structured text sequence生成出来。

References:
- Project page: https://bytedance.github.io/vidi-website/
- arXiv (Vidi原版): https://arxiv.org/abs/2504.15681
- arXiv (Vidi2): https://arxiv.org/abs/2511.19529

---

## 2. Architecture深度解析

### 2.1 整体架构

Vidi2保留Vidi的多模态架构设计，三模态联合处理text/visual/audio：

```
Input: text + video frames + audio
  ↓
[Modality-specific Encoders]
  - Visual encoder (with adaptive token compression)
  - Audio encoder
  - Text tokenizer
  ↓
[Projected to shared embedding space]
  ↓
[LLM Backbone: Gemma-3 12B]
  ↓
Output: text with structured spatio-temporal tokens
```

**关键设计点**:
- LLM backbone使用Gemma-3 [15] (https://arxiv.org/abs/2503.19786)，12B参数配置
- Single image被treat为1-second silent video，这样image和video共享unified encoding interface
- Adaptive token compression策略重新设计，平衡short/long video的token效率

### 2.2 Adaptive Token Compression的intuition

对于长视频（30分钟），如果按1 FPS采样就有1800帧，每帧224×224 patch化后token数量爆炸。Adaptive compression的核心思路：

- **Short video**: 保留更多spatial细节，因为short video通常需要fine-grained spatial grounding
- **Long video**: 更aggressive的temporal downsampling，因为long video的核心是temporal reasoning而非spatial precision

这种design choice直接反映在Table 6/7的结果中：在medium-length video (10-30min)上，Vidi2的tIoU达到47.27%，而Gemini 3 Pro只有21.13%，GPT-5跌到4.10%。差距随video length拉大，说明token compression策略对long video至关重要。

### 2.3 Training Pipeline

Vidi2沿用Vidi的训练pipeline但scale up：

**Stage 1 - Pretraining**: 大规模multimodal alignment
**Stage 2 - Temporal-aware multimodal alignment**: 关键阶段，对temporal retrieval和video QA都有generalizable的提升
**Stage 3 - SFT**: 扩展temporal retrieval数据 + 加入generic QA数据 + STG task-specific数据
**Stage 4 - RL (Vidi2.5新增)**: multimodal RL with verifiable rewards

STG训练数据构建的核心trick：
- 利用image-level spatial grounding datasets合成large-scale spatio-temporal video grounding pairs
- 这种"以图造视频"的合成策略有效bridge了spatial alignment和temporal alignment

---

## 3. Spatio-Temporal Grounding技术细节

### 3.1 Tube Representation

论文用tube来建模spatio-temporal object：

$$B(t) = \big( x_0(t), y_0(t), x_1(t), y_1(t) \big)$$

变量解释：
- $B(t)$: time $t$ 时刻的bounding box
- $x_0(t), y_0(t)$: bounding box左上角坐标
- $x_1(t), y_1(t)$: bounding box右下角坐标
- $t$: 时间变量，连续时间轴

实际discretization：1 frame per second采样，所以 $t \in \{0, 1, 2, ..., T\}$，$T$是video总秒数。

**关键intuition**: Tube不是一个static box，而是time-dependent function。模型需要输出的是这个function的discrete samples，即一串(timestamp, box) pairs。

### 3.2 Bounding Box IoU

$$\text{bIoU}(B_1, B_2) = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}$$

标准的2D IoU，用于spatial alignment评估。

### 3.3 Temporal Metrics

定义 $T_{\text{pred}}, T_{\text{gt}} \subset \mathbb{R}$ 为predicted和ground-truth的temporal support，$T_\cap = T_{\text{pred}} \cap T_{\text{gt}}$，$T_\cup = T_{\text{pred}} \cup T_{\text{gt}}$。

**Temporal Precision (tP)**:
$$\text{tP} = \frac{|T_\cap|}{|T_{\text{pred}}|}$$

intuition: predicted time span中有多少是真正命中ground truth的，衡量prediction的精度（避免over-prediction）。

**Temporal Recall (tR)**:
$$\text{tR} = \frac{|T_\cap|}{|T_{\text{gt}}|}$$

intuition: ground truth time span中有多少被predicted覆盖，衡量coverage（避免under-prediction）。

**Temporal IoU (tIoU)**:
$$\text{tIoU} = \frac{|T_\cap|}{|T_\cup|}$$

intuition: 标准IoU形式的temporal alignment，综合precision和recall。

### 3.4 Spatio-Temporal Metrics（核心创新）

**Frame-level IoU**:
$$\text{IoU}_t = \begin{cases} \text{bIoU}(B^{\text{pred}}(t), B^{\text{gt}}(t)), & t \in T_\cap \\ 0, & \text{otherwise} \end{cases}$$

变量解释：
- $B^{\text{pred}}(t)$: 模型在time $t$预测的bounding box
- $B^{\text{gt}}(t)$: ground truth在time $t$的bounding box
- 当 $t \notin T_\cap$（即时间没对齐），frame-level IoU直接置0

**Accumulated IoU over temporal intersection**:
$$S = \sum_{t \in T_\cap} \text{IoU}_t$$

intuition: 在时间对齐的部分，把每一帧的spatial IoU累加起来。这个 $S$ 是后续所有spatio-temporal metrics的基础。

**Spatio-temporal Precision (vP)**:
$$\text{vP} = \frac{S}{|T_{\text{pred}}|}$$

intuition: 在predicted time span上平均的frame-level IoU，惩罚over-prediction（predicted太长但spatial不准）。

**Spatio-temporal Recall (vR)**:
$$\text{vR} = \frac{S}{|T_{\text{gt}}|}$$

intuition: 在ground truth time span上平均的frame-level IoU，惩罚under-prediction（漏掉了gt的时间段）。

**Spatio-temporal IoU (vIoU) - Primary Metric**:
$$\text{vIoU} = \frac{S}{|T_\cup|}$$

intuition: 在temporal union上平均的frame-level IoU。这个metric同时惩罚temporal misalignment（因为 $|T_\cup|$ 会变大）和spatial inaccuracy（因为 $S$ 会变小）。

**vIoU-Int** (complementary metric):
$$\text{vIoU-Int.} = \frac{S}{|T_\cap|}$$

intuition: 只在time对齐的部分评估spatial accuracy。这个metric非常有用，因为它能区分两种failure mode：
- vIoU低 + vIoU-Int高 → temporal alignment有问题
- vIoU低 + vIoU-Int低 → spatial localization有问题

看Table 7的数据：Vidi2的vIoU 32.57 vs vIoU-Int 60.30，差距27.7，说明很多错误来自temporal alignment。Gemini 3 Pro的vIoU 4.61 vs vIoU-Int 16.59，差距12，但其vIoU-Int也低，说明Gemini的spatial localization本身就有问题。

---

## 4. Benchmark设计深度分析

### 4.1 VUE-STG: Spatio-Temporal Grounding Benchmark

**数据规模** (Table 1):
- 982 videos, 1600 queries, 12,147 boxes
- 总时长204.79小时
- 三档duration: ultra-short (126 videos, 0.82h), short (294, 26.28h), medium (562, 177.69h)

**关键设计创新**:

1. **Video duration**: 涵盖10秒到30分钟，远超现有STG dataset [2,14,23,24]。这一点很重要，因为传统STG dataset多在短视频上评估，但real-world editing需要处理长视频。

2. **Query format**: 人工verified，明确指向target object。论文给的例子很有启发性：
   - 模糊: "a player is loaded into an ambulance"
   - 明确: "the ambulance which the player is being loaded into" 或 "the player who is loaded into the ambulance"
   
   这种disambiguation策略显著降低了semantic ambiguity。

3. **Fragmented tubes**: 真实场景中objects会因camera cut、occlusion、scene transition消失再出现。VUE-STG保留这种fragmented annotation而不是强制temporal continuity。这是realistic的考量。

4. **Object size distribution** (Table 2):
   - Small (<10%): 746 tubes (最常见)
   - Medium (10-30%): 511 tubes
   - Large (>30%): 343 tubes
   
   Small object占多数，这对模型是巨大挑战。

5. **Tube duration** (Table 3):
   - Micro-short (<3s): 179
   - Ultra-short (3-10s): 1013 (最多)
   - Short (10-60s): 408

### 4.2 VUE-TR-V2: Temporal Retrieval升级版

**核心改进** (Table 4):
- 总时长从VUE-TR的107.87h增加到310.72h (2.88x)
- 五档duration: ultra-short (<1min)到ultra-long (>60min)
- Long (30-60min): 197 videos, 145.95h - 新增档
- Ultra-long (>60min): 39 videos, 47.50h - 新增档

**Evaluation Metric**: AUC of precision, recall, IoU，记作 $\bar{P}, \bar{R}, \overline{\text{IoU}}$。用AUC而非单点metric，可以评估模型在不同threshold下的整体表现。

### 4.3 VUE-PLOT: Plot Understanding Benchmark

**Character Track** (Table 5):
- 546 videos, 13554 speech segments, 33083 bboxes
- 22.45小时
- 任务: dense speaker localization + speech recognition

**Reasoning Track**:
- 137 videos, 1214 QA pairs
- 6.11小时
- 五种task类型 (Figure 6 right):
  1. Narrative and Structural Understanding
  2. Perception and Understanding
  3. Professional Filming and Editing Techniques
  4. Social Cognition and Knowledge Integration
  5. Speech, Audio, and Sound Effect Reasoning

**Reasoning Track的metrics**:

**Temporal Grounding IoU**:
$$\text{tIoU} = \frac{1}{N_{\text{match}}} \sum_{i=1}^{N_{\text{match}}} \text{IoU}(s_i^{\text{gt}}, s_i^{\text{pred}})$$

变量：
- $N_{\text{match}}$: 匹配的segment对数
- $s_i^{\text{gt}}, s_i^{\text{pred}}$: 第 $i$ 对gt和pred的segment

**Word Error Rate (WER)**:
$$\text{WER} = \frac{S + D + I}{N}$$

变量：
- $S$: substitutions (替换错误数)
- $D$: deletions (删除错误数)
- $I$: insertions (插入错误数)
- $N$: ground truth transcript的word总数

**Spatial IoU (sIoU)**:
$$\text{sIoU} = \frac{1}{N_{\text{box}}} \sum_{j=1}^{N_{\text{box}}} \text{IoU}(b_j^{\text{gt}}, b_j^{\text{pred}})$$

变量：
- $N_{\text{box}}$: matched box对数
- $b_j^{\text{gt}}, b_j^{\text{pred}}$: 第 $j$ 对gt和pred的bounding box

**Multiple-Choice VQA Accuracy**:
$$\text{Acc}_{\text{MC}} = \frac{1}{N_q} \sum_{i=1}^{N_q} \mathbb{I}(a_i^{\text{pred}} = a_i^{\text{gt}})$$

变量：
- $N_q$: 问题总数
- $a_i^{\text{pred}}, a_i^{\text{gt}}$: 第 $i$ 题的predicted和gt answer
- $\mathbb{I}(\cdot)$: indicator function

---

## 5. Vidi2.5的RL Training与Thinking Model

### 5.1 Vidi2.5: Multimodal RL with Verifiable Rewards

Vidi2.5在Vidi2基础上加入RL训练，核心是**verifiable rewards**：对于STG、TR这类任务，ground truth是确定的(time range, bounding box)，可以直接用metric (IoU)作为reward，无需LLM-as-judge。

效果对比 (Table 7):
- Vidi2.5 vIoU: 38.64 vs Vidi2: 32.57 (+6.07)
- Vidi2.5 tIoU: 58.34 vs Vidi2: 53.19 (+5.15)

verifiable reward的RL对STG这种有明确ground truth的任务特别有效，因为reward signal清晰无噪声。

### 5.2 Vidi2.5-Think: Inference-Time Scaling

Vidi2.5-Think的核心设计:
- 灵感来自DeepSeek-R1 [7] (https://arxiv.org/abs/2501.12948) 和 OpenAI o1 [8] (https://openai.com/index/openai-o1-system-card/)
- 基于VERL framework [12] (https://arxiv.org/abs/2409.19256)
- **Hybrid reward**: verifiable rewards + LLM-as-judge

hybrid reward的intuition:
- Verifiable rewards: 对有明确答案的subtask（如temporal grounding）
- LLM-as-judge: 对open-ended reasoning（如rationale quality评估）

这种组合设计让Vidi2.5-Think既能优化可验证的perception能力，又能提升reasoning质量。

### 5.3 Vidi2.5-Think的训练数据

训练数据覆盖：
- Video perception
- Audio understanding  
- Narrative comprehension
- Professional filming and editing techniques

任务形式：
- Multiple-choice VQA
- Open-ended reasoning

这种多任务设计让thinking model能在不同reasoning depth的任务上都发挥作用。

---

## 6. 实验结果深度分析

### 6.1 VUE-STG结果 (Table 6/7)

**Temporal Grounding (Table 6) - Overall**:
| Model | tIoU | tP | tR |
|-------|------|-----|-----|
| Vidi2.5 | 58.34 | 72.14 | 68.00 |
| Vidi2 | 53.19 | 73.00 | 59.80 |
| Gemini 3 Pro | 27.50 | 51.91 | 35.26 |
| GPT-5 | 16.40 | 38.29 | 19.53 |
| Qwen3-VL-32B | 25.91 | 45.29 | 39.19 |

Vidi2.5相对Vidi2的tR大幅提升(59.80→68.00)，说明RL训练主要改善了coverage（少漏）。

**Spatio-Temporal Grounding (Table 7) - Overall**:
| Model | vIoU | vIoU-Int | vP | vR |
|-------|------|----------|-----|-----|
| Vidi2.5 | 38.64 | 64.84 | 47.26 | 44.71 |
| Vidi2 | 32.57 | 60.30 | 44.56 | 36.32 |
| Gemini 3 Pro | 4.61 | 16.59 | 8.95 | 5.71 |
| GPT-5 | 5.47 | 33.64 | 13.01 | 6.50 |
| Qwen3-VL-32B | 5.12 | 18.47 | 8.61 | 7.49 |

关键观察：
1. Vidi2.5/2 vs其他模型差距巨大（vIoU差距30+点），这是end-to-end STG架构的优势
2. Vidi2.5的vR提升明显(36.32→44.71)，RL训练改善了temporal coverage
3. GPT-5的vIoU-Int 33.64相对高，但vIoU只有5.47，说明GPT-5的spatial localization能力其实不弱，但temporal alignment严重失败
4. Long video (medium 10-30min)上，Gemini/GPT/Qwen都崩溃(vIoU < 3)，Vidi2.5仍维持33.27

### 6.2 VUE-TR-V2结果 (Table 8)

**Overall IoU**:
| Model | IoU | P̄ | R̄ |
|-------|-----|-----|-----|
| Vidi2.5 | 49.62 | 59.78 | 71.09 |
| Vidi2 | 48.75 | 62.45 | 64.93 |
| Gemini 3 Pro | 37.58 | 48.61 | 56.30 |
| GPT-5 | 17.15 | 29.64 | 26.63 |

**按video length分**:
- Ultra-long (>60min): Vidi2.5 42.22 vs Gemini 21.19 vs GPT 12.49
- Long (30-60min): Vidi2.5 48.54 vs Gemini 38.41 vs GPT 9.39

Vidi2.5在ultra-long上的IoU (42.22)甚至超过Gemini在ultra-short上的IoU (38.41)，差距非常显著。

**按query format分**:
- Keyword: Vidi2.5 48.56 vs Gemini 38.41
- Phrase: Vidi2.5 51.22 vs Gemini 37.84
- Sentence: Vidi2.5 48.83 vs Gemini 36.73

Sentence query上Vidi2.5优势明显，说明对complex natural language的理解能力强。

### 6.3 VUE-PLOT结果 (Table 10)

**Character Track**:
| Model | tIoU | sIoU | WER↓ |
|-------|------|------|------|
| Vidi2.5-Think | 71.63 | 66.04 | (lowest) |
| Gemini 3 Pro | 50.68 | 13.24 | 23.20 |
| GPT-5 | 55.89 | 6.80 | 29.00 |
| Qwen3-VL | 58.12 | - | - |
| Qwen3-Omni | - | - | - |

Vidi2.5-Think的sIoU 66.04 vs Gemini的13.24，差距超42点，这是巨大的lead。

**Reasoning Track - Overall**:
| Model | Acc |
|-------|-----|
| Vidi2.5-Think | 64.33 |
| Gemini 3 Pro | 64.58 |
| GPT-5 | 54.37 |
| Qwen3-VL | 33.94 |
| Qwen3-Omni | 28.01 |

**Reasoning分task**:
| Task | Vidi2.5-Think | Gemini 3 Pro |
|------|---------------|--------------|
| Perception | 66.25 | 80.42 |
| Speech/Audio | 74.43 | 66.03 |
| Social Cognition | 55.83 | 55.34 |
| Narrative | 35.00 | 45.42 |
| Filming/Editing | 61.35 | 50.92 |

Vidi2.5-Think在Speech/Audio (+8.40)和Filming/Editing (+10.43)上明显超过Gemini，这与Vidi的multimodal alignment training (含audio)的训练设计有关。在Narrative和Perception上Gemini更强，反映Vidi在general perception上还有提升空间。

### 6.4 Video QA (Table 9)

| Model | LVBench | LongVideoBench | VideoMME |
|-------|---------|----------------|----------|
| Gemini-2.5-Pro | 78.7 | 84.3 | - |
| Qwen2.5-VL-7B | 45.3 | 54.7 | 65.1 |
| Vidi2 | 45.8 | 57.1 | 63.5 |
| Vidi2.5 | 45.2 | 58.9 | 63.6 |

Vidi2.5在LongVideoBench上比Qwen2.5-VL-7B高4.2点，但LVBench持平。Vidi2.5在LongVideoBench比Vidi2提升1.8，说明RL训练对long video QA有改善。

---

## 7. Inference Setup的技术细节

论文Section 4.1.1详细描述了如何与LMM baselines公平对比，这部分很值得学习。

### 7.1 不同模型的input/output格式

**Gemini 3 Pro Preview**:
- Input: video via URL
- Output: JSON array with MM:SS timestamp + [0,1000] normalized box
- 直接处理完整video

**GPT-5**:
- Input: 最多120 image frames per request
- 采样策略: <2min视频1 FPS，>2min视频均匀采样到120帧
- Output: frame index + [0,1] normalized box

**Qwen3-VL-32B**:
- Input: interleaved timestamp-image序列
- Output: seconds + [0,1000] box

### 7.2 Normalization策略

时间归一化：
- Gemini: MM:SS → seconds
- GPT-5: frame index → seconds (根据采样率)
- Qwen3-VL: 已经是seconds

空间归一化：
- Gemini/Qwen3-VL: [0,1000] → [0,1] (除以1000)
- GPT-5: 已经是[0,1]

这种统一归一化确保了metric计算的公平性。

---

## 8. Vidi-Edit: Video Editing Planning应用

### 8.1 任务定义

给定raw assets (images + videos) + optional user prompts，生成editing plan包含：
1. **Narrative structure**: clip selection, segment extraction, segment ordering
2. **Voiceover content**: narration text + delivery style + temporal alignment
3. **Audio attributes**: music style, mood, BPM, speaker
4. **Visual editing intent**: transition design, emphasis cues, stylistic directives

### 8.2 Execution Pipeline (Figure 12)

```
Raw Assets + User Intent
  ↓
[Vidi-Edit - High-level Planning]
  ↓
Editing Plan (structured text/JSON)
  ↓
[Translation Stage]
  - Music attributes → Music database retrieval
  - Voiceover → TTS synthesis
  - Visual intent → Effect retrieval + parameterization
  ↓
[Rendering System]
  ↓
Final Video
```

关键设计insight：**High-level planning与execution分离**。Vidi-Edit专注于semantic reasoning，把具体的asset selection、synthesis、rendering交给downstream specialized modules。这种separation of concerns让planning model能专注于high-level decision making。

### 8.3 Editing Plan示例 (Figure 12)

```json
{
  "scenario": "A pampered cat is on a mission to reclaim his food bowl...",
  "viral strategy": "The story is framed as a 'sibling showdown'...",
  "pacing strategy": "A quick, punchy setup showing the problem...",
  "editing_script": [
    {
      "scene": "A fluffy white cat cautiously approaches an automatic feeder.",
      "voiceover": "POV: You just want your own food bowl.",
      "timestamps": ["asset 3, 00:02-00:10"],
      "music_tag": ["Lo-Fi Hiphop", "Chill Beats", "Funny"],
      "music_bpm": "Medium (90-110)",
      "tts_speaker": "Bestie",
      "visual_effect": "Begin with the cat cautiously approaching..."
    }
  ]
}
```

这种结构化plan的设计有几个优势：
- 可解析: 下游module可以直接读取字段
- 可组合: 不同aspect独立可控
- 可解释: 用户能理解模型的reasoning

### 8.4 Agent架构extension

论文提到这种pipeline可以扩展为editing agent架构：
- Vidi-Edit作为high-level decision-making module
- Execution pipeline作为tool使用
- 可迭代planning-execution循环

这是agent-based video editing的promising方向。

---

## 9. 关键Intuition总结

### 9.1 为什么Vidi2在STG上能大幅领先？

1. **End-to-end training**: 直接从query到tube sequence，避免了pipeline的error accumulation
2. **Task-specific data**: 大规模合成 + 真实STG annotation
3. **Architecture design**: Adaptive token compression适应不同video length
4. **Multimodal alignment**: temporal-aware alignment stage对STG有generalizable的提升

### 9.2 为什么Vidi2.5的RL有效？

1. **Verifiable reward**: STG/TR的ground truth明确，reward signal清晰
2. **Direct optimization**: 直接在target metric (IoU)上优化
3. **Coverage improvement**: tR/vR提升明显，说明RL帮助模型"少漏"

### 9.3 为什么Vidi2.5-Think在Character track上特别强？

1. **Hybrid reward**: verifiable + LLM-judge同时优化perception和reasoning
2. **Multimodal training data**: 含audio的training让speaker recognition更强
3. **Inference-time scaling**: thinking过程帮助复杂plot分解

### 9.4 Vidi-Edit的planning-execution分离为什么重要？

1. **Modularity**: planning和execution独立优化
2. **Interpretability**: high-level plan可被人理解和修改
3. **Extensibility**: 可接入新的execution module (新music library, 新visual effect)
4. **Agent-ready**: 为multi-step agent iteration提供基础

---

## 10. 与相关工作的联系

### 10.1 STG相关工作

- [5] Context-guided STG: 早期STG工作
- [6] Target-aware transformer: 强调target的role
- [20] VideoGrounding-DINO: open-vocabulary STG
- Vidi2的差异化: end-to-end LMM-based STG, 支持长视频

### 10.2 Thinking Model相关工作

- DeepSeek-R1 [7]: RL with verifiable rewards for reasoning
- OpenAI o1 [8]: inference-time scaling
- Vidi2.5-Think的差异化: multimodal + hybrid reward

### 10.3 Video Editing相关工作

- Vidi (原版) [17]: temporal retrieval foundation
- Vidi2/2.5: 加入STG + plot understanding + editing planning

---

## 11. 我的思考与可能的extension方向

1. **Tube representation的连续化**: 目前是1 FPS discrete sampling，可以考虑continuous time representation (如neural temporal fields)

2. **Multi-object STG**: 当前paper主要是单tube，multi-object场景下的query disambiguation更具挑战

3. **Interactive editing**: Vidi-Edit目前是one-shot planning，可以扩展为iterative refinement (用户反馈→plan修改)

4. **Cross-video STG**: 多个video中的同一object grounding，对video editing的asset retrieval有用

5. **Audio-visual causal reasoning**: VUE-PLOT的Reasoning track中Speech/Audio reasoning是Vidi2.5-Think的强项，可以进一步探索audio-visual causal model

6. **Hierarchical planning**: Vidi-Edit可以扩展为hierarchical (高-level narrative → mid-level scene → low-level cut)

7. **Reward design for creative tasks**: editing plan的quality如何自动评估？目前的LLM-judge可能不够，可以考虑human preference model

References:
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- OpenAI o1: https://openai.com/index/openai-o1-system-card/
- VERL: https://arxiv.org/abs/2409.19256
- Gemma-3: https://arxiv.org/abs/2503.19786
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- Qwen3-Omni: https://arxiv.org/abs/2506.09310 (实际需查证)
- VideoMME: https://arxiv.org/abs/2405.21075
- LVBench: https://arxiv.org/abs/2406.14380
- LongVideoBench: https://arxiv.org/abs/2410.10734

这篇paper的核心价值在于：**用end-to-end LMM重新定义了spatio-temporal grounding的SOTA**，并通过Vidi-Edit展示了multimodal reasoning在video editing这种complex real-world task上的潜力。Vidi2.5-Think的hybrid reward设计也值得深思，为multimodal reasoning model的post-training提供了新思路。
