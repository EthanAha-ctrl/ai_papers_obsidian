---
source_pdf: VideoEspresso.pdf
paper_sha256: a455ab04b39e59a390249607e163f0d15951e365e05d2781695b279613e5176d
processed_at: '2026-08-13T00:54:20-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说VideoEspresso

好，我把之前的技术分析翻译成人话。

---

## 这篇paper在干嘛

假设你想训练一个能看视频回答复杂问题的AI。你遇到一个很尴尬的局面：

**数据不够，而且现有的数据都挺差。**

传统做法是雇人看视频写问答对——质量还行，但太贵太慢，而且人写的往往比较浅层，问不出什么有深度的问题。

后来大家想了个偷懒办法：用GPT-4这种大模型，给个视频简介，让它自动生成问答对。问题是，视频简介太粗糙了，很多重要细节没写进去，生成的问答对也就浮于表面。

还有个更笨的办法：把视频每一帧都喂给模型分析。问题是视频帧太多了，大量都是冗余的——一个人走路的视频，几十帧画面几乎一模一样，全喂进去既费钱又把关键信息淹没了。

VideoEspresso的核心idea就是：**像泡espresso咖啡一样，把视频里的精华"压"出来。**

---

## 它是怎么做的

整个pipeline分三步，我类比成"做一杯espresso"的过程：

### 第一步：选豆子（去掉冗余帧）

视频帧太多了，大部分都是废话。怎么去掉冗余？

他们的做法很聪明：**先把每一帧用文字描述出来，然后在文字层面去重。**

具体来说，用一个小模型（InternVL2-8B）给每一帧生成一句描述。然后用一个文本相似度工具（BGE-M3）比较相邻帧的描述——如果两句描述高度相似，就说明这两帧内容差不多，扔掉后一个。

这个做法的妙处在于：**判断两张图片是不是相似，比判断两段文字是不是相似要难得多。** 把视觉问题转化为文本问题，一下子就好处理了。

而且他们用了一个叫"LIFO"的策略——后进先出，意思是连续相似时保留最早那一帧。直觉上这也对，因为故事的开头往往定了整个场景的基调。

### 第二步：磨豆子（生成问答对）

去重后，他们把剩下的帧按15个一组分组。为什么是15？大概是因为既不能太少（缺少上下文），也不能太多（GPT-4o的context window装不下，也容易走神）。

然后把每组帧的描述喂给GPT-4o，让它生成复杂的推理类问答对。注意，不是简单问"视频里有什么"，而是问"为什么""怎么""基于什么推断"这种需要推理的问题。

生成完之后，再用另一个LLM当"质检员"，过滤掉质量差的、有幻觉的、太主观的问题。

### 第三步：拉花（标注推理链）

光有问答对还不够，他们还想标注出"答案是怎么推出来的"——也就是Chain-of-Thought。

具体标注三样东西：
1. **哪些帧是回答这个问题的关键帧**（不是所有帧都相关）
2. **关键帧里哪些物体是关键证据**（用GroundingDINO画框标记）
3. **把这些关键物体组织成自然语言描述**作为推理依据

这个步骤的价值在于：模型不仅知道"答案是X"，还知道"因为看到A帧里的B物体做了C动作，所以推断出X"。这就从"背答案"变成了"讲道理"。

---

## 模型架构：小工加大师

光有数据不够，还得设计一个能利用这数据的模型。他们的架构很有意思，我类比成"小工加大师"：

**小工（Frame Selector）**：一个1B参数的小视觉模型加一个0.5B参数的小语言模型。它的活儿是看视频、给每一帧写描述，然后根据问题选出最相关的几帧。

**大师（Reasoning LVLM）**：一个7B级别的大模型。它只看小工选出来的2-3张关键帧，然后基于这些帧做推理回答。

为什么这么设计？因为大模型看视频太贵了。如果让它看128帧，计算量爆炸。但如果只让它看2-3张精华帧，既省钱效果还好——因为冗余帧反而会干扰它的注意力。

这就像看病：不需要让专家翻你所有的体检报告，先让实习医生挑出有问题的几页，专家只看那几页就够了。

---

## 训练：先学找证据，再学推理

他们把训练分成两阶段，这个设计也很有讲究：

**阶段一**：告诉模型"请找出回答这个问题的证据"。模型学着从画面里提取关键信息——哪些物体重要、它们在做什么、时间上怎么衔接。

**阶段二**：告诉模型"根据你找到的证据回答问题"。模型学着基于证据做逻辑推理。

为什么不直接端到端训练？因为如果一上来就让它既找证据又推理，两个任务互相干扰，模型容易学歪。分开训练，每个阶段的gradient信号更干净，学得更扎实。

---

## 结果怎么样

效果确实不错：

- 用2.36帧（别人的1/50到1/7）达到34.1%准确率
- 比InternVL2高5.4%，比GPT-4o高7.7%
- 计算量只有别人的15%左右

而且他们做了个很有说服力的消融实验：如果用"完美标注"的CoT来指导，准确率能飙到72.95%。这说明什么？说明CoT这个方向是对的，只是当前模型还没充分学会利用它——天花板还很高。

---

## 我觉得哪里聪明，哪里有疑问

**聪明的地方**：

1. **用语言空间做视觉去重**——这个idea本身就很优雅，把难题转化到容易的领域
2. **Question-driven选帧**——同一个视频问不同问题选不同帧，这个直觉非常对
3. **两阶段训练**——先grounding再reasoning，符合人类认知发展顺序
4. **小模型选帧+大模型推理**——工程上非常实用，类似MoE的思想

**我有疑问的地方**：

1. **严重依赖GPT-4o**——整个数据pipeline靠GPT-4o生成caption和CoT。如果GPT-4o对某个domain不熟，数据质量就崩了。能不能做成self-supervised？
2. **长视频还是搞不定**——这个framework本质还是sample frames，对真正长视频（10分钟以上）的long-range dependency还是无能为力
3. **Selector的上限不够高**——GT selector也才37.54%，说明选帧这个问题本身就很难。也许该让selector也学会"我不确定时多选几帧"
4. **评测用GPT-4o打分有偏好**——subjective evaluation用GPT-4o当judge，可能有self-preference，GPT-4o容易偏好GPT-4o风格的表达

---

## 一句话总结

这篇paper的核心insight：**把video reasoning的难题，通过"视觉转文字"这个桥梁，转化为language model擅长的问题**。从去重、选帧、到CoT生成，都是这个思路的体现。代价是引入了对GPT-4o和captioning模型的依赖，收益是pipeline变得modular、scalable、可解释。

---

# VideoEspresso 深度解析

Andrej，这篇paper的核心insight其实非常elegant：它解决了video reasoning中一个根本性的tension——**granularity vs scalability**。让我从intuition层面来拆解整个pipeline。

---

## 1. 核心动机与设计哲学

### 1.1 问题本质

传统VideoQA datasets面临一个trilemma：
- **Manual annotation**（如NExT-QA、TVQA、MovieChat）：质量高但scale小，granularity受限
- **Metadata-based LLM generation**：scale大但丢失了fine-grained visual details
- **Frame-by-frame analysis**：detail充分但computational expensive且information overload

VideoEspresso的关键insight是：**通过semantic-aware mapping把视觉冗余问题转化为language space的retrieval问题**。这个idea非常聪明，因为它把video frame selection从visual similarity的模糊判断，转移到textual semantic similarity的精确计算上。

### 1.2 为什么叫"Espresso"

Espresso是"pressed out"的coffee——通过pressure提取essence。这个metaphor完美对应了paper的pipeline：从raw video frames中"press"出core semantic content。这个命名暗示了整个methodology的哲学——**compression with information preservation**。

参考：[VideoEspresso GitHub](https://github.com/hshjerry/VideoEspresso)

---

## 2. 数据集构建Pipeline深度解析

### 2.1 Redundancy Removal的数学形式化

Section 3.2的公式(1)-(2)看似简单，但蕴含深刻的设计：

$$\mathcal{S} = \underset{c_i, c_j \in \mathcal{C}}{\arg \max} \cos(\phi_T(c_i), \phi_T(c_j))$$

$$\mathcal{C} \rightarrow \mathcal{C}' \quad (c \in \mathcal{C}', \text{if } S(c) < \tau)$$

**变量含义**：
- $\mathcal{C}$：原始caption集合（所有sampled frames的textual descriptions）
- $\mathcal{C}'$：filter后的core caption集合
- $c_i, c_j$：caption集合中的第i、j个元素
- $\phi_T(\cdot)$：text encoder（BGE-M3）的feature extraction function
- $\cos(\cdot, \cdot)$：cosine similarity
- $\mathcal{S}$：adjacent captions之间的similarity matrix
- $\tau$：threshold for redundancy判断
- $S(c)$：caption c与adjacent caption的similarity score

**关键设计——LIFO filtering**：
这个Last-In-First-Out策略的intuition是：当连续frames语义高度相似时，保留earlier frame（语义变化的起点），丢弃later redundant frame。这与video narrative的natural structure一致——故事的beginning往往anchor了整个sequence的context。

**为什么用BGE-M3而不是CLIP text encoder**：
BGE-M3是multilingual、multi-functionality、multi-granularity的embedding model，比CLIP的text encoder在semantic retrieval上更powerful。reference: [BGE-M3 paper](https://arxiv.org/abs/2402.03216)

### 2.2 Frame Captioning的FPS自适应

Paper提到：
- Dynamic scenes：FPS = 2-4
- Static scenes：FPS = 1

这个adaptive sampling的intuition是：video的information density与motion complexity正相关。使用InternVL2-8B做frame-level captioning，本质上是在做一次**visual-to-textual projection**，把visual redundancy转化为textual redundancy，后者更容易用NLP tools处理。

参考：[InternVL2](https://github.com/OpenGVLab/InternVL)

### 2.3 QA Pair Construction的Grouping Strategy

Section 3.3提到"every 15 consecutive frame captions grouped into $G_i$"。这个magic number 15的intuition：

1. **Token budget**：GPT-4o的context window需要留出space for prompt instructions和QA generation
2. **Semantic coherence**：15 frames大约覆盖5-15秒video content，保证narrative unit的completeness
3. **Reasoning complexity**：太少frames无法construct complex reasoning QA，太多frames导致attention dilution

### 2.4 Multimodal CoT的时空对齐

Section 3.4的公式(3)是temporal grounding的核心：

$$t = \arg \max_k \cos(\phi_T(c_j), \phi_T(c_k))$$

**变量含义**：
- $t$：temporal grounding information（时间戳）
- $c_j \in \mathcal{G}_{GPT}$：GPT-4o生成的core frame caption
- $c_k \in \mathcal{G}_i$：原始caption group中的第k个caption
- $\phi_T(\cdot)$：BGE-M3 text encoder
- $\cos(\cdot, \cdot)$：cosine similarity

**为什么需要这个alignment**：
GPT-4o生成CoT时，会paraphrase原始captions，导致string-level matching失败。用semantic similarity做retrieval可以recover temporal position。这是一个**cross-modal temporal grounding**问题——text space的semantic matching对应video space的时间定位。

**Spatial annotation**：
- GroundingDINO：open-vocabulary object detection，标记key items的bounding boxes
- CLIP-ViT-B/32：验证label与bounding box内容的一致性

这个verification step很重要，因为GroundingDINO可能hallucinate。CLIP的verification是一个**cross-check mechanism**，降低spatial annotation的noise。

参考：[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [CLIP](https://github.com/openai/CLIP)

---

## 3. Hybrid LVLMs Collaboration Framework

### 3.1 Architecture的Economic Intuition

Figure 5的architecture本质上是一个**two-tier computational economy**：

**Tier 1 - Frame Selector（cheap worker）**：
- InternVL2-1B（1B params）做frame captioning
- QwenLM-0.5B（0.5B params）做question-relevant frame selection
- 总共1.5B parameters，computational cost极低

**Tier 2 - Reasoning LVLM（expensive expert）**：
- LLaVA-Next-interleave backbone（7B级别）
- 只处理selected core frames（average 2.36 frames vs 16-128 frames）

这个design的intuition类似Mixture of Experts的spirit——用cheap model做routing，用expensive model做reasoning。但这里是sequential而非parallel。

### 3.2 Core Frame Selection的形式化

公式(4)-(6)描述了完整的selection process：

$$\{f_i\}_{i=1}^N = \text{SampleFrames}(V, \text{FPS})$$

$$\{c_i\}_{i=1}^N = \text{LVLM}(\{f_i\}_{i=1}^N)$$

$$\{c_j'\}_{j=1}^M = \text{LLM}(\{c_i\}_{i=1}^N, q)$$

**变量含义**：
- $V$：input video
- $f_i$：第i个sampled frame
- $N$：总sampled frame数
- $c_i$：第i个frame的caption
- $c_j'$：第j个selected core frame的caption
- $M$：selected core frame数（$M \leq N$）
- $q$：question
- LVLM：InternVL2-1B
- LLM：QwenLM-0.5B

**关键insight**：这个selection是**question-driven**的，不是question-agnostic的uniform sampling。同一个video，不同question会select不同的core frames。这是与传统keyframe extraction的根本区别。

### 3.3 Two-Stage Training的Curriculum Learning Intuition

Section 4.2的两阶段SFT本质上是一个**curriculum learning**策略：

**Stage 1 - Evidence Extraction**：
- Instruction: "Please provide evidence to help answer the question."
- Model学到：如何从visual content中extract relevant evidence
- 这stage训练的是**visual grounding ability**

**Stage 2 - Answer Generation with Evidence**：
- Instruction: "Please answer the question with the help of evidence."
- Model学到：如何基于evidence做logical reasoning
- 这stage训练的是**reasoning ability**

**为什么不直接end-to-end**：
分阶段training的好处是**gradient signal更clean**。Stage 1的loss只关注evidence quality，不被answer generation的复杂reasoning干扰。Stage 2的input已经包含grounding evidence，model可以focus on reasoning pattern learning。

Table 3的ablation验证了这个design：
- Baseline (no CoT): 34.13%
- w/o Bbox: 33.14% (−0.99%)
- w/o CoT: 31.32% (−2.81%)
- GT-CoT: 72.95% (+38.82%)

GT-CoT的+38.82%提升说明：**CoT的upper bound非常高**，当前model还没有fully exploit这个potential。这暗示future work的方向是提升CoT generation quality。

---

## 4. 实验结果深度分析

### 4.1 Main Results（Table 1）的关键insights

**Performance vs Efficiency trade-off**：

| Model | #Frames | TFLOPs | Avg. Acc. |
|-------|---------|--------|-----------|
| LongVA-DPO | 128 | 465.4 | 24.4% |
| LLaVA-1.5 | 4 | 14.50 | 18.02% |
| InternVL2 | FPS=1 | 73.23 | 28.7% |
| **Ours** | **2.36** | **9.26** | **34.1%** |

Ours用**2.36 frames**（LongVA的1.8%）和**9.26 TFLOPs**（LLaVA-Next-interleave的14.74%），achieve了**34.1% accuracy**，比InternVL2高5.4%，比GPT-4o高7.7%。

这个efficiency gain的来源：
1. Frame Selector过滤了>85%的redundant frames
2. 每个frame的token数固定，input length大幅缩减
3. Two-stage training让reasoning更focused

**Task-specific analysis**：
- **Causal Inference**：Ours 45.2% vs GPT-4o 41.7%——CoT对causal reasoning的提升显著
- **Influence Tracing**：Ours 55.6% vs 次优48.6%——temporal grounding的direct benefit
- **Theme Analysis**：InternVL2 42.6%最高——可能因为training data中有similar content
- **Cooking Process**：LongVA-DPO 37.7%最高——long video training的优势

### 4.2 Subjective Evaluation（Table 2）

| Model | Logic | Factuality | Accuracy | Conciseness | Overall |
|-------|-------|-----------|----------|-------------|---------|
| GPT-4o | 73.15 | 61.66 | 70.02 | 66.13 | 66.13 |
| **Ours** | **72.25** | **61.28** | **75.73** | **65.84** | **65.84** |

有趣的是：Ours在**Accuracy**上超过GPT-4o（75.73 vs 70.02），但在**Logic**上略低（72.25 vs 73.15）。这暗示：VideoEspresso训练让model更grounded（accuracy高），但可能在complex logical chain上还不如GPT-4o的prior knowledge。

**Conciseness**：Ours 75.73 vs GPT-4o 66.13——这说明CoT training让model生成更focused的answer，减少了hallucination和over-analysis。Figure 13和14的case study正好印证这一点：GPT-4o容易over-analyze，generate看似rich但visually ungrounded的内容。

### 4.3 Selector Ablation（Table 4）

| Selector | #Frame | GPU hr | Memory | Acc. |
|----------|--------|--------|--------|------|
| Uniform | 8 | = | 0+14+40G | 33.74% |
| GT | 2.98 | | 0+14+15G | 37.54% |
| 1B/1.5B | 2.77 | 1.33 | 5+14+14G | 34.76% |
| **1B/0.5B** | **2.36** | **0.37** | **3+14+12G** | **34.13%** |

**关键insight**：
- **GT selector**上界37.54%，说明selector的quality还有3.4%的提升空间
- **1B/0.5B**比**1B/1.5B**更快（0.37 vs 1.33 GPU hr），memory更少（29G vs 33G），accuracy反而更高（34.13 vs 34.76）
- 这说明：**selector不需要太强**，只要能做question-frame relevance ranking就行

### 4.4 Cross-Model Generalization（Table 5）

| Model | Sampling | #Frame | Acc. |
|-------|----------|--------|------|
| GPT-4o | Uniform | 16 | 26.86 |
| GPT-4o | 1B/0.5B | 2.36 | 28.26 |
| InternVL2 | Uniform | 16 | 28.57 |
| InternVL2 | 1B/0.5B | 2.36 | 29.23 |
| LongVA | Uniform | 128 | 24.41 |
| LongVA | 1B/0.5B | 2.36 | 23.18 |

**Plug-and-play capability**：
- GPT-4o和InternVL2用selector后accuracy提升，frame数减少85%
- LongVA和LLaVA-Next-interleave accuracy略降，但frame数减少98%
- 这个trade-off说明：selector对**reasoning-focused models**帮助大，对**long-context models**主要是efficiency gain

---

## 5. Dataset的Statistical Insights

### 5.1 Core Frame Distance Distribution（Figure 3a）

不同task的key frame距离分布差异巨大：
- **Causal Inference**：距离分布广，需要跨frame reasoning
- **Traffic Analysis**：距离集中，relevant frames通常连续
- **Cooking Process**：中等距离，step-by-step progression

这个distribution说明：**uniform sampling对所有task都不是optimal的**。Adaptive sampling是必须的。

### 5.2 与MVBench的对比（Figure 4）

**Token Length**：
- MVBench：QA长度短，distribution集中
- VideoEspresso：answer长度长，distribution广

**Word Cloud**：
- VideoEspresso Question：reasoning-oriented（"considering", "based", "inferred"）
- MVBench Question：basic inquiry（"object", "person", "action"）
- VideoEspresso Answer：reasoning process（"Initially", "Finally"）
- MVBench Answer：spatial definition（"left", "forward"）

这说明VideoEspresso的QA pairs确实更侧重于**reasoning chain**而非**factoid retrieval**。

### 5.3 Dataset Scale（Table 6）

| Dataset | #Questions | CoT |
|---------|------------|-----|
| How2QA | 2,852 | × |
| ActivityNet-QA | 8,000 | × |
| NExT-QA | 8,564 | × |
| MovieChat | 13,000 | × |
| TVQA | 15,253 | × |
| MSRVTT-QA | 72,821 | × |
| VideoCoT | 11,182 | Text only |
| **VideoEspresso** | **203,546** | **Text + Visual** |

VideoEspresso是**最大规模**且**唯一包含visual CoT**的video reasoning dataset。

---

## 6. Objective Evaluation Algorithm分析

Algorithm 1的two-step evaluation设计很精妙：

```
Step 1: Sim(output, reference) > τ=80%?
  No → Incorrect
  Yes → Step 2

Step 2: For each distractor D_i:
  Sim(output, D_i) > Sim(output, reference)?
  Any Yes → Incorrect
  All No → Correct
```

**Intuition**：
- Step 1：semantic correctness的necessary condition
- Step 2：confusion test——如果output与distractor更相似，说明model可能猜对了但reasoning错误

这个evaluation比传统exact match或BLEU更robust，因为它：
1. 允许paraphrase（semantic similarity）
2. 防止lucky guess（distractor comparison）
3. 适合open-ended generation evaluation

---

## 7. 我的个人insights和未来方向联想

### 7.1 这篇paper的真正贡献

1. **Methodology contribution**：证明了"language as bridge for visual redundancy reduction"的feasibility
2. **Data contribution**：第一个large-scale multimodal CoT video dataset
3. **Architecture contribution**：hybrid small-large model collaboration的successful demonstration

### 7.2 可能的future directions

**1. Self-supervised CoT learning**：
当前依赖GPT-4o生成CoT，如果能self-supervised从video content中extract reasoning chain，可以avoid GPT-4o的bias和cost。

**2. Hierarchical Frame Selection**：
当前selector是flat的。可以设计hierarchical selection——先select coarse-grained scene segments，再select fine-grained frames within each segment。类似human的attention机制。

**3. Interactive CoT**：
当前CoT是one-pass generation。可以设计interactive CoT——model生成evidence后，如果uncertainty高，可以query more frames。类似Active Learning in inference time。

**4. Multi-granularity CoT**：
当前CoT只包含spatial（bbox）和temporal（frame index）。可以扩展到action-level、object-level、scene-level的hierarchical CoT。

**5. Benchmark limitations**：
当前benchmark只有1,382 test questions。可以扩展test set，并加入adversarial examples——比如修改frame顺序，test model的temporal sensitivity。

### 7.3 与相关工作的联系

- **Visual CoT（Shao et al.）**：image-level的visual CoT，VideoEspresso扩展到video-level
- **VoT（Fei et al.）**：spatial relationship reasoning，VideoEspresso增加temporal dimension
- **VideoCoT（Wang et al.）**：text-only CoT，VideoEspresso增加visual grounding
- **LLaVA-NeXT-Interleave**：multi-image understanding backbone，VideoEspresso fine-tune它做video reasoning

参考：
- [Visual CoT](https://arxiv.org/abs/2403.16999)
- [VoT](https://proceedings.mlr.press/v235/fei24a.html)
- [VideoCoT](https://arxiv.org/abs/2407.05355)
- [LLaVA-NeXT-Interleave](https://arxiv.org/abs/2407.07895)

### 7.4 Technical Concerns

1. **GPT-4o dependency**：整个pipeline依赖GPT-4o的quality，如果GPT-4o对某些domain不熟悉，CoT quality会受影响
2. **Selector的ceiling**：GT selector只有37.54%，说明还有很大提升空间
3. **Long video limitation**：当前framework对long video（>1min）的处理还是依赖frame sampling，没有真正的long-range temporal modeling
4. **Evaluation bias**：使用GPT-4o做subjective evaluation可能有self-preference bias

---

## 8. 总结

VideoEspresso这篇paper的核心价值在于：**它提供了一个scalable的pipeline来construct fine-grained video reasoning data**。这个pipeline的每个component都经过精心设计：

- **Semantic-aware frame reduction**：用language space做redundancy removal
- **Question-driven frame selection**：adaptive而非uniform sampling
- **Multimodal CoT annotation**：spatial + temporal grounding
- **Two-stage curriculum training**：evidence extraction → answer generation
- **Hybrid model collaboration**：small model routing + large model reasoning

整个framework的design philosophy是：**用language作为visual和reasoning之间的bridge**。Frame captioning把visual转化为text，text similarity做redundancy reduction和frame selection，text-based reasoning做CoT generation。这个language-centric design让整个pipeline高度modular和scalable。

对于你Andrej关注的area，这个paper最值得思考的点是：**如何用language model的reasoning ability来bootstrap visual reasoning**。这本质上是一种**cross-modal knowledge transfer**——用LLM的reasoning prior来guide visual model的reasoning learning。这个paradigm可能比直接end-to-end training更sample-efficient。

---

**Key References**:
- [VideoEspresso GitHub](https://github.com/hshjerry/VideoEspresso)
- [InternVL2](https://github.com/OpenGVLab/InternVL)
- [BGE-M3](https://arxiv.org/abs/2402.03216)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [MVBench](https://github.com/OpenGVLab/LLaVA-NeXT)
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
- [mPLUG-Owl3](https://github.com/X-PLUG/mPLUG-Owl)
- [LongVA](https://arxiv.org/abs/2406.16852)
- [Visual CoT](https://arxiv.org/abs/2403.16999)
- [VoT (Video-of-Thought)](https://proceedings.mlr.press/v235/fei24a.html)
- [VideoCoT](https://arxiv.org/abs/2407.05355)
- [CLIP](https://github.com/openai/CLIP)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [LoRA](https://arxiv.org/abs/2106.09685)
