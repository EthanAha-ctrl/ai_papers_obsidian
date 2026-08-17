---
source_pdf: Qwen3-VL-Embedding and Qwen3-VL-Reranker A Unified.pdf
paper_sha256: d7cab1b67d4777abc1c85a28b73004e27ad556f5932bb960fb2a8d9d80398b85
processed_at: '2026-08-11T20:29:48-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

---

## 这东西是干嘛的

就是做一个**搜图搜视频搜文档**的模型。你给它一句话，它能从一堆图片、视频、文档里找到最相关的东西。也能反过来——给它一张图，找到匹配的文字。

---

## 怎么做的

核心idea特别朴素：**拿一个已经很聪明的VLM（Qwen3-VL），教它学会"判断相关性"这个技能。**

就像一个博学的大学毕业生，知识都有，但没学过怎么做搜索引擎。你给他看几万个"这个query该匹配哪个document"的例子，他就学会了。

---

## 两个模型分工

**Embedding model** — 快速粗筛
- 把每个东西变成一个向量
- 算cosine similarity就行
- 便宜，能处理million-scale corpus

**Reranker** — 精挑细选
- 把query和document拼一起，让VLM判断"相关吗？yes/no"
- 贵但准
- 只对top-100候选跑

就像招人：HR先快速筛简历（embedding），部门主管再面试精挑（reranker）。

---

## 训练的聪明之处

### 1. 用大模型当"出题老师"
公开的multimodal retrieval数据又少又不均衡。所以他们让Qwen3-VL-32B去看图片和视频，自己生成大量的"query-document对"，包括正确的和故意搞错的hard negative。

就像让学霸出模拟题——学霸知道哪些错答案最像对的，学生练了就更强。

### 2. 四阶段进化

- **s0**：先拿合成数据训一遍，建立baseline
- **s1**：用s0去mining更高质量数据，再训，同时训reranker
- **s2**：让reranker当老师，用distillation教embedding model
- **s3**：把s1和s2的权重merge一下，解决"s2检索变强但分类变弱"的trade-off

每一轮模型变强→数据mining更好→下一轮模型更强。这个loop很elegant。

### 3. Distillation的intuition

Reranker能给每个query-document对打一个细粒度的分数（比如0.3 vs 0.05）。Embedding model通过KL divergence去match这个分布，比单纯学"相关/不相关"的binary label信息量大得多。

就像老师不只告诉你"对/错"，还告诉你"这个错得离谱，那个差点对了"——学生学到的信号更丰富。

### 4. Model merging解决task conflict

s2在retrieval任务上变强了，但classification和QA变弱了。作者直接把s1和s2的权重merge（因为是LoRA training，权重线性可加），结果两边的好都能保留。

这招在LLM时代越来越重要——不同任务训出来的model weight可以像调色一样混合。

---

## 省钱的两招

### Matryoshka Representation Learning (MRL)

像俄罗斯套娃。训练时强制所有"截断维度"都要semantic meaningful——比如2048维的embedding，你只取前512维也要有用。

部署时按需截断。存1024维省一半空间，只掉1.4%性能。

### Quantization-Aware Training (QAT)

训练时就让模型适应低精度存储。int8几乎无损，binary会掉点。

对billion-scale corpus，从16KB/条降到4KB/条（int8），省的是**TB级**的存储和对应的retrieval latency。

---

## 效果

- MMEB-V2 multimodal benchmark：**77.8分**，SOTA，超第二名6.7%
- 纯text也能打：MTEB上67.9分（略低于Qwen3-Embedding的70.6，multimodal有点tax）
- Reranker-8B在visual document retrieval上碾压ColPali-style模型

---

## 一句话总结

**把strong VLM当backbone，用合成数据+多阶段训练+distillation+model merging，sculpt出retrieval能力，再用MRL+QAT让它能在production部署。** 整个pipeline每个环节互相feed forward，是self-improving的设计。

---

# Qwen3-VL-Embedding & Qwen3-VL-Reranker 深度解析

Andrej, 这篇paper来自Alibaba Tongyi Lab, 是Qwen3-VL foundation model在retrieval领域的extension。核心idea是: 把一个strong VLM通过multi-stage training改造成unified multimodal retrieval system, 同时输出embedding model和reranker model, 两者通过distillation互相bootstrap。我下面从architecture、training pipeline、loss design、data synthesis、efficiency techniques几个维度展开, 重点是build你的intuition。

---

## 1. Architecture: 为什么用VLM做retrieval backbone

传统multimodal retrieval走CLIP-style dual encoder路线(image encoder + text encoder + contrastive loss), 这种架构在fine-grained document understanding上比较弱。Qwen3-VL-Embedding直接继承Qwen3-VL的causal attention backbone, 保留world knowledge和instruction-following能力, 然后用retrieval-specific data去sculpt出relevance estimation能力。

**Embedding Model (Bi-encoder):**
- Input format遵循Qwen3-VL chat template: instruction作为system message, instance作为user message
- 在input末尾append一个`<|PAD|>` token (类似BERT的[CLS]或E5的`</s>`), 取这个token的last hidden state作为dense vector
- Similarity用cosine
- 2B: 28 layers, dim=2048; 8B: 36 layers, dim=4096
- 支持32k context length

**Reranker Model (Cross-encoder):**
- Pointwise formulation: 把query和document拼在一起, 让VLM预测"yes"/"no" token
- Input template:
  ```
  <system> Judge whether the Document meets the requirements...
  <user> <Instruct>: {I} <Query>: {q} <Document>: {d}
  <assistant>
  ```
- Relevance score = sigmoid(logit(yes) - logit(no)) — 这个设计很巧妙, 用logit difference而不是单纯yes probability, 可以calibrate掉prior bias

**Intuition:** Bi-encoder负责粗筛(cheap, scalable), cross-encoder负责精排(expensive, accurate)。两者通过distillation形成teacher-student关系, 这是整个pipeline的核心。

参考: [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631), [VLM2Vec](https://arxiv.org/abs/2507.04590), [GME](https://arxiv.org/abs/2504.07499)

---

## 2. Multi-stage Training Pipeline: 4个model checkpoint的演化

这是paper最精彩的部分。整个pipeline产出4个embedding model版本(s0→s1→s2→s3)和一个reranker, 形成一个iterative bootstrapping loop。

### Stage 1: Contrastive Pre-training → s0
- 用大规模synthetic data(由Qwen3-VL-32B生成)
- Hard negative mining用open-source GME model做recall
- Loss: 完整InfoNCE with in-batch query-query和document-document negatives
- 目的: 建立baseline relevance understanding

### Stage 2: Multi-Task Contrastive Learning + SFT → s1 + Reranker
- 用s0去做更高质量的data mining(模型变强→数据变好→模型更强)
- Embedding: 去掉query-query和document-document项, 只保留query-document contrastive terms
- 同时train reranker在retrieval-specific subset上
- 这个stage引入curated public + proprietary data

### Stage 3: Distillation → s2
- 用Stage 2训好的reranker给compact subset打fine-grained relevance scores
- Embedding model通过KL divergence去match reranker的score distribution
- s2在retrieval tasks上大幅提升, 但classification和QA tasks轻微退化

### Stage 3.5: Model Merging → s3
- 把s1和s2通过model merging(Li et al., 2024的方法)合并
- 解决s2的task conflict问题
- s3是最终release的版本

**Table 6的ablation数据(2B model on MMEB-V2):**

| Stage | Image Overall | Video Overall | VisDoc Overall | All |
|-------|--------------|--------------|----------------|-----|
| s0 | 65.8 | 57.5 | 74.8 | 66.6 |
| s1 | 74.8 | 60.3 | 77.1 | 72.1 |
| s2 | 71.3 | 59.5 | 80.9 | 71.5 |
| s3 | 75.0 | 61.9 | 79.2 | 73.2 |

注意s2在VisDoc上从77.1跳到80.9(distillation对document retrieval帮助巨大), 但Image从74.8掉到71.3。s3 merging后Image回到75.0, VisDoc保持79.2, 证明了model merging解决task conflict的有效性。

参考: [Qwen3 Embedding](https://arxiv.org/abs/2506.05176), [Model Merging for embedding](https://arxiv.org/abs/2410.15035)

---

## 3. Loss Functions: 详解公式与设计哲学

### 3.1 Retrieval Loss (InfoNCE扩展版)

$$\mathcal{L}_{\text{retrieval}} = -\frac{1}{N} \sum_{i}^{N} \log \frac{e^{s(q_i, d_i^+)/\tau}}{Z_i}$$

变量含义:
- $N$: batch size
- $q_i$: 第i个query的embedding
- $d_i^+$: 第i个query对应的positive document embedding
- $s(\cdot, \cdot)$: cosine similarity
- $\tau$: temperature parameter(可学习或固定)
- $Z_i$: normalization term, 包含5类contrastive terms

$Z_i$的完整展开:
$$Z_i = e^{s(q_i, d_i^+)/\tau} + \sum_k^K m_{ik} e^{s(q_i, d_{i,k}^-)/\tau} + \sum_{j \neq i} m_{ij} e^{s(q_i, q_j)/\tau} + \sum_{j \neq i} m_{ij} e^{s(d_i^+, d_j)/\tau} + \sum_{j \neq i} m_{ij} e^{s(q_i, d_j)/\tau}$$

5个term分别对应:
1. Positive pair $d_i^+$
2. K个hard negatives $\{d_{i,k}^-\}_{k=1}^K$
3. In-batch其他queries $\{q_j\}_{j \neq i}$ (query-query contrast)
4. In-batch其他documents与$d_i^+$的对比 (doc-doc contrast via positive)
5. In-batch其他documents与$q_i$的对比 (standard in-batch negative)

**Masking factor $m_{ij}$防false negative:**
$$m_{ij} = \begin{cases} 0, & \text{if } s_{ij} > s(q_i, d_i^+) + 0.1 \text{ or } d_j = d_i^+ \\ 1, & \text{otherwise} \end{cases}$$

Intuition: 如果某个in-batch negative的similarity已经超过positive + 0.1的margin, 很可能是false negative(语义上其实相关), 直接mask掉。这个trick在multimodal场景特别重要, 因为image-text的semantic overlap比纯text更模糊。

**Stage 2的modification:** 去掉term 3和term 4(query-query和doc-doc), 只保留query-doc contrast。作者观察到high-quality multimodal data上这样更稳定。这个观察和SimCSE、E5系列的发现一致 — 当数据质量高时, in-batch query作为mutual negative会引入noise。

### 3.2 CoSent Loss for STS

$$\mathcal{L}_{\text{sts}} = \log\left(1 + \sum_{\hat{s}(q_i, d_j) > \hat{s}(q_m, d_n)} \exp\left(\frac{\cos(q_m, d_n) - \cos(q_i, d_j)}{\tau}\right)\right)$$

变量:
- $\hat{s}(q_i, d_j)$: ground-truth similarity score(连续值, 0-1或0-5)
- $\cos(q_i, d_j)$: 模型预测的cosine similarity
- 求和遍历所有ground-truth ordering违反的pair对

Intuition: STS数据是对称的, 没有natural query-doc asymmetry, 而且label是连续score而非binary。CoSent loss本质是pairwise ranking loss — 强制模型预测的cosine ordering和ground-truth ordering一致。比直接MSE regression更robust, 因为它只关心relative ordering。

参考: [CoSent](https://ieeexplore.ieee.org/document/10386181)

### 3.3 Distillation Loss

$$\mathcal{L}_{\text{distill}} = -\sum_{i=1}^{k+1} P_{\text{reranker}}(d_i | q) \log P_{\text{embedding}}(d_i | q)$$

变量:
- $k+1$: 1个positive + k个negatives
- $P_{\text{reranker}}(d_i|q)$: reranker对query q下document $d_i$的softmax概率(offline pre-computed)
- $P_{\text{embedding}}(d_i|q)$: embedding modelonline计算的softmax概率

这是标准KL divergence(等价于cross-entropy with soft targets)。Reranker的soft label比binary hard label包含更多信息 — 比如reranker可能给negative A打0.3, negative B打0.05, 这个relative hardness信息通过distillation传递给embedding。

### 3.4 Reranking Loss

$$\mathcal{L}_{\text{reranking}} = -\log p(l | I, q, d)$$

变量:
- $l$: "yes"或"no" token
- $I$: instruction
- $p(\cdot|*)$: VLM的next token probability

Inference时:
$$s = \text{sigmoid}(\text{logit}(\text{yes}) - \text{logit}(\text{no}))$$

这个logit difference的设计比直接用$p(\text{yes})$更稳健 — VLM对"yes"的prior可能偏高(训练数据positive bias), 用logit difference可以cancel out这个bias, 类似于contrastive formulation。

参考: [Supervised fine-tuning vs contrastive for reranking](https://arxiv.org/abs/2510.14824)

---

## 4. Data Synthesis: 用Qwen3-VL-32B做annotator

这是整个工作能scale的关键。公开multimodal retrieval数据严重imbalance, 作者用synthesis补齐。

### Seed Pool Construction Pipeline:
1. **Coarse filtering**: 剔除低分辨率、异常aspect ratio的image/video
2. **Structural refinement**: video做scene cut detection, 移除static/corrupted segments
3. **Category labeling**: Qwen3-VL-32B打fine-grained category labels
4. **Cross-modal alignment filtering**: 用GME embedding similarity score过滤visual-text correspondence差的样本
5. **Category rebalancing**: 确保类别分布均衡

### Synthesis Tasks:

**Image (3类):**
- Classification: object recognition, scene parsing, landmark ID, action recognition
- QA: factoid, visual reasoning, OCR extraction, domain knowledge
- Retrieval: visual description, narrative scenario, compositional logic, knowledge-centric localization

**Video (4类):**
- Classification: activity, scene, event, sentiment/intent
- QA: factual, temporal grounding, thematic reasoning, cinematic analysis
- Retrieval: entity/action, temporal-event, thematic/emotional, tutorial
- **Moment Retrieval**: text query → specific video segment(带temporal boundary), 这是video特有的fine-grained task

### Hard Negative Mining (两阶段):

**Recall阶段:** 用embedding model对每个query $q_j$ retrieve top-K candidates, 得到relevance scores $S = \{s_{j,k}\}_{k=1}^K$

**Relevance Filtering:**
- Positive refinement: 保留$q_j$当且仅当至少一个positive $d^+$的score $s > t^+$(threshold)
- Hard negative selection: 对valid query, 计算positive平均score $\bar{s}^+$, 选非positive且score满足 $s < \bar{s}^+ + \delta^-$ 的document作为hard negative

$\delta^-$是safety margin防止false negative。这个pipeline本质是用模型自己的retrieval能力去curate训练数据, 形成self-improving loop。

参考: [Contrastive Learning with Hard Negatives](https://arxiv.org/abs/2101.03231), [MegaPairs](https://aclanthology.org/2025.acl-long.990/)

---

## 5. Efficiency Techniques: MRL + QAT

这两个技术让模型在production部署时storage和compute大幅下降。

### Matryoshka Representation Learning (MRL)

训练时不仅在全维度embedding上计算loss, 还在truncated低维prefix上同时计算loss。比如dim=2048的模型, 同时在2048, 1024, 512, 256, 128, 64维上计算contrastive loss。

$$\mathcal{L}_{\text{MRL}} = \sum_{d \in \mathcal{D}} \mathcal{L}(\text{truncate}(e, d))$$

其中$\mathcal{D}$是dimension set, $\text{truncate}(e, d)$取embedding $e$的前$d$维。

**Intuition:** Matryoshka doll套娃 — 大embedding包含小embedding的信息, 小embedding是大embedding的coarse approximation。训练时强制所有prefix维度都semantic meaningful, 这样部署时可以按需截断。

**Figure 6的实验结果:**
- Text retrieval (MSMARCO): dim从1024降到512, MRR@10只掉1.4%, 但storage减半, retrieval速度翻倍
- T2I retrieval (VL3-Syn, 2M image corpus): 同样趋势
- Int8 quantization几乎无损, binary quantization显著掉点
- 维度越低, quantization的degradation越明显

### Quantization-Aware Training (QAT) with LSQ

LSQ (Learned Step Size Quantization):
- 量化scale (step size) $\Delta$作为learnable parameter, 和model weight一起backprop
- 用Straight-Through Estimator (STE)绕过rounding operation的不可导问题
- 前向: $e_q = \Delta \cdot \text{round}(e / \Delta)$
- 反向: gradient直接传过round操作, 当作identity

训练时同时用full-precision和quantized embedding计算loss, 让model学会produce quantization-robust的embedding。

**为什么QAT重要:** 大规模retrieval system里, corpus embedding要offline存billions条。FP32存4096维embedding = 16KB/条, 1B条=16TB。Int8 = 4KB/条, 1B条=4TB。Binary = 512 bytes/条, 1B条=512GB。QAT让int8几乎无损, 这对production是game-changer。

参考: [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147), [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153), [Straight-Through Estimator](https://arxiv.org/abs/1308.3432)

---

## 6. Visual Granularity Analysis (Figure 7)

作者做了resource consumption vs performance的scaling分析:

**Image (spatial):** 增加visual token数(即提高分辨率), performance提升但有diminishing return, 最高token consumption时甚至regression(长context degradation)

**Video (temporal + spatial):**
- Temporal axis: 增加frame数, 早期提升明显, 后期plateau
- Spatial axis: 增加per-frame token budget, 类似趋势

**Implementation details:**
- Image: 保持aspect ratio, max 1280 tokens (~1.3M pixels)
- Video: 1 FPS采样, max 64 frames, total 4500 tokens (~9.2M pixels)

这个analysis对deployment很重要 — 不是越多token越好, 要找sweet spot。

---

## 7. Benchmark Results

### MMEB-V2 (Table 2) - 主战场
- Qwen3-VL-Embedding-8B: **77.8** overall, 比之前SOTA开源模型提升6.7%
- Image: 80.1, Video: 67.1, VisDoc: 82.4
- 超过closed-source的IFM-TTE (74.1)和Seed-1.6-embedding-1215 (76.9)

### Visual Document Retrieval (Table 3)
- 和ColPali-style模型(ColQwen2.5, ColNomic)comparable, 但ColPali用late interaction需要更高compute
- Reranker-8B: **80.3** avg, 显著超过所有embedding model

### MTEB Multilingual (Table 4) - 纯文本
- 8B: 67.9 mean task score
- 比Qwen3-Embedding-8B (70.6)略低 — multimodal training对纯text有轻微tax
- 但仍competitive with同size text-only模型

### Reranking (Table 5)
- 8B reranker比2B embedding + reranker提升明显
- MMEB-v2 retrieval avg: 73.4(embedding) → 75.2(reranker-2B) → 79.2(reranker-8B)
- ViDoRe v3: 52.9 → 60.8 → 66.7

参考: [MMEB](https://arxiv.org/abs/2507.04590), [MTEB](https://arxiv.org/abs/2502.13595), [ColPali](https://arxiv.org/abs/2407.01449), [Jina VDR](https://arxiv.org/abs/2506.18902)

---

## 8. 我的几点观察与联想

**1. VLM-as-embedder的trend:** 这篇paper、VLM2Vec、GME、BGE-VL都在走同一条路 — 把instruction-tuned VLM直接改造为embedder。好处是inherit world knowledge和multilingual能力, 坏处是参数效率不如dedicated dual encoder。Qwen3-VL-Embedding-8B在纯text上不如Qwen3-Embedding-8B就是这个tax的体现。

**2. Distillation loop的设计很elegant:** Reranker和Embedding不是独立训练, 而是Reranker先训好→给Embedding做teacher→Embedding变强后可以做更高质量data mining→下一轮Reranker更好。这个iterative bootstrapping是self-improving的。

**3. Model Merging解决task conflict:** s2在retrieval上强但classification弱, s1反之。直接合并weight(不是ensemble inference)就能得到balanced model, 这依赖LoRA training让weight space线性可加。这个trick在LLM era越来越重要。

**4. Pointwise reranker的logit difference trick:** $s = \sigma(\text{logit}_{\text{yes}} - \text{logit}_{\text{no}})$这个formulation比单纯用$p(\text{yes})$更robust, 因为它cancel了VLM对"yes"的prior bias。类似idea在reward model里也有(用preference pair的logit difference)。

**5. MRL + QAT的组合:** 这两个技术叠加让embedding的storage cost可以从16KB/条降到512B/条(int8+低维), 对billion-scale corpus是enabler。QAT训练时STE的gradient估计虽然有bias, 但实验证明int8几乎无损。

**6. 与ColPali的对比:** ColPali用late interaction(每个token做match, MaxSim aggregation), 精度高但inference贵。Qwen3-VL-Embedding用single vector representation, 在document retrieval上能match ColPali, 说明strong VLM backbone + good data可以弥补architectural simplicity。

**7. Missing pieces:** Paper没详细讨论的 — (a) synthetic data的diversity如何量化验证; (b) reranker的inference cost如何optimize(32k context的cross-encoder很贵); (c) multilingual能力在non-English上的breakdown; (d) compositional reasoning的failure case analysis。

**8. 与Moon Embedding、MegaPairs的关联:** 同期工作也在做类似synthesis + VLM backbone的multimodal retrieval, 说明这是industry consensus direction。Qwen3-VL-Embedding的优势在于multi-stage pipeline的系统性设计 + MRL/QAT的production-readiness。

---

## 9. 实用资源

- **Models**: [HuggingFace Collection](https://huggingface.co/collections/Qwen), [ModelScope](https://modelscope.cn/organization/qwen)
- **Code**: [GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding)
- **Qwen3-VL backbone**: [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)
- **Qwen3 Embedding (text-only predecessor)**: [arXiv:2506.05176](https://arxiv.org/abs/2506.05176)
- **MMEB-V2 benchmark**: [arXiv:2507.04590](https://arxiv.org/abs/2507.04590)
- **MTEB**: [arXiv:2502.13595](https://arxiv.org/abs/2502.13595)
- **ColPali (baseline对比)**: [arXiv:2407.01449](https://arxiv.org/abs/2407.01449)
- **Matryoshka Representation Learning**: [arXiv:2205.13147](https://arxiv.org/abs/2205.13147)
- **LSQ Quantization**: [arXiv:1902.08153](https://arxiv.org/abs/1902.08153)
- **VLM2Vec**: [arXiv:2507.04590](https://arxiv.org/abs/2507.04590)
- **GME**: [arXiv:2504.07499](https://arxiv.org/abs/2504.07499)

---

总结一句build intuition的话: 这篇paper的核心贡献是把VLM foundation model通过"synthetic data generation → multi-stage contrastive training → cross-encoder distillation → model merging"这个pipeline, 系统性地sculpt出retrieval能力, 同时用MRL+QAT保证production efficiency。整个设计是iterative self-improving的, 每个stage的output都feed forward到下一个stage的data quality或model quality提升上。
