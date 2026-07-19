---
source_pdf: Action100M.pdf
paper_sha256: 8edecc2c28b304e7b130fad54a69de174a586a89b4bfb52a7a7623b8ca54b708
processed_at: '2026-07-18T01:01:16-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Action100M Paper 深度解析

Andrej，这篇paper的核心贡献是把video action understanding这个长期受数据规模限制的领域，用一个完全自动化的pipeline推到了O(100M)的scale。下面我从pipeline技术细节、VL-JEPA训练protocol、experimental结果以及与相关工作关联四个层面来拆解。

## 1. 整体Pipeline架构

整个pipeline的设计哲学是"先压缩成text再推理"。换句话说，避免直接把heavy VLM应用在长video上，而是先用轻量模型把video变成结构化的Tree-of-Captions，最后用纯text-based reasoning LLM做聚合。这个设计跟Pyramid-of-Captions（Chen et al., 2024a）思路一脉相承。

### Stage 1: Hierarchical Temporal Segmentation

**输入处理**：原始video按每4帧取1帧的stride做uniform sampling，模拟V-JEPA 2预训练时的temporal resolution。然后用sliding window切分：
- 每个window：64 sampled frames
- 相邻window的stride：8 frames (87.5% overlap)
- Encoder：V-JEPA 2 ViT-g-384
- 每个window独立encode后做spatial average pooling，得到per-frame feature (维度 = encoder hidden size)

由于window overlap，同一帧会有多个representations。pipeline对这些representations做accumulation and average，得到整段video上时间一致的per-frame embedding序列。

**Hierarchical Agglomerative Clustering (HAC) with Ward linkage**：

Ward linkage的目标是最小化每次merge带来的intra-cluster variance增量。具体定义：

$$d(A, B) = \sqrt{\frac{|A| \cdot |B|}{|A| + |B|}} \|\mu_A - \mu_B\|_2$$

其中：
- $A, B$ 是两个待merge的cluster
- $|A|, |B|$ 是cluster中frame数
- $\mu_A, \mu_B$ 是cluster centroid (即frame embedding的均值)
- $d(A, B)$ 是Ward distance，衡量merge后intra-cluster variance的增长

每次选择$d(A, B)$最小的相邻cluster对进行merge，重复直到收敛。Local temporal connectivity constraint要求每个frame只能与immediate neighbor连接，所以segment必须是contiguous time span，这样得到的tree是一棵segment tree（类似interval tree）。

最终输出是一个hierarchy，底层是细粒度的atomic motion（如"add sugar"），高层是procedural step（如"make Irish coffee"）。保留duration > 0.5s的node。

**Intuition**：V-JEPA 2 embedding本身已经编码了motion-aware的语义信息，相同action的帧embedding在feature space里是接近的，所以Ward linkage会自然把"持续做同一件事"的帧聚类在一起。这其实类似DINOv2在image上的feature clustering表现出的emergent semantic segmentation，但V-JEPA 2是temporal version。

### Stage 2: Multi-mode Caption Generation

Tree的每个node都会得到caption，两种mode互补：

**Mid-frame captioning (leaf nodes)**：
- 模型：Llama-3.2-Vision-11B
- 输入：node时间区间midpoint的一帧image
- Prompt: "Describe this image in detail."

**Video-segment captioning (high-level nodes)**：
- 模型：Perception-LM-3B
- 输入：在segment start到end之间均匀sample 32 frames，分辨率 320²
- Prompt: "Describe this video in detail."

两个模型都限制1024 tokens，都能在单卡V100 32GB上跑。这种"轻量VLM + 局部window"的分配方式把cost控制在可承受范围内：1.3M V100 GPU-hours做segmentation + captioning。

### Stage 3: LLM Aggregation with Self-Refine

这是整个pipeline最关键的一步，用GPT-OSS-120B做reasoning aggregation。

**输入构造**（depth-first Markdown serialized）：
- Current node的caption
- Children captions (depth-first order)
- Global root captions (限制depth)
- Video metadata (title, description, ASR transcript)

**Self-Refine (3 rounds)**：
- Round 1: high reasoning effort，生成initial draft
- Round 2-3: 把previous output + original context重新feed，让模型verify, correct factual errors, remove unsupported statements

**Output schema** (JSON)：
- `summary.brief`: 单句video caption
- `summary.detailed`: dense comprehensive描述
- `action.brief`: 单个verb phrase (不能用-ing形式)
- `action.detailed`: imperative sentence描述how
- `action.actor`: 主语（noun phrase或完整sentence）

整个Stage 3消耗0.3M H100/H200 GPU-hours。

**Intuition on why this works**: Pyramid-of-Captions paper (Chen et al., 2024a)证明，把多个caption聚合起来能有效降低hallucination。本质上是multi-view evidence aggregation——每个caption可能错，但majority consensus往往对。GPT-OSS-120B作为reasoning model特别擅长这种"分析-verify-修正"的任务。

## 2. 数据集统计

- 1,199,096 videos (来自HowTo100M, face-blurred)
- 总时长：14.6 years
- ASR transcripts retrieval rate: 72%
- Annotated segments: 147,092,653
- 4类annotation的字数分布：
  - Brief action: avg 3.2 words
  - Brief caption: avg 19.2 words  
  - Detailed action: avg 27.8 words
  - Detailed caption: avg 95.3 words
- 总token量：21.27B English words
- "N/A" action标签占比：3.23% (对应intro/广告/订阅提醒等)
- Segment duration分布：64%在0-3s, 23.8%在3-10s, 10.2%在10s-1min, 2% > 1min
- 磁盘占用：~205 GB

数据集的long-tail问题很严重——paper Figure 11显示有7.58M duplicate groups, 141.8M duplicate instances。"speak to camera"这种pattern占比过高，直接训练会bias model。

## 3. VL-JEPA训练Protocol

VL-JEPA是Chen et al., 2025c提出的vision-language JEPA，核心idea是用V-JEPA 2作为visual encoder，用text encoder生成target embedding，用InfoNCE loss对齐embedding space。

### InfoNCE Loss公式

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(z_i^v, z_i^t)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(z_i^v, z_j^t)/\tau)}$$

其中：
- $z_i^v \in \mathbb{R}^d$：第$i$个video样本的visual embedding (由V-JEPA 2 + projector得到)
- $z_i^t \in \mathbb{R}^d$：第$i$个样本的text target embedding (由text encoder + stop-gradient得到)
- $z_j^t$：batch内第$j$个text embedding (作为negative)
- $\text{sim}(\cdot, \cdot)$：cosine similarity
- $\tau > 0$：temperature scalar (可学习或固定)
- $N$：batch size

分子是positive pair的similarity，分母是所有negative pairs的sum。这是标准contrastive learning的InfoNCE，源自CPC (van den Oord et al., 2018)。

### 3-Stage训练

| Stage | Vision Encoder | #Frames | Data | Batch | #Iter | LR |
|-------|---------------|---------|------|-------|-------|-----|
| Stage 1 | Frozen | 1 | DataComp-1B + YFCC-100M (image-text) | 24,576 | 100k | 5e-5 |
| Stage 2 | Frozen | 8 | Action100M (mix 4 fields, detailed downsampled 0.5×) + PLM-3B labels | 12,288 | 60k | 5e-5 |
| Stage 3 | Unfrozen | 32 | Action100M | 3,072×4 (grad accum) | 10k | 1e-5 |

Stage 3的gradient accumulation是4，所以effective batch size = 12,288。Unfreeze V-JEPA 2 encoder需要更高显存，因此物理batch变小，用grad accum补偿。

**Intuition on 3-stage设计**：Stage 1先把vision-language alignment建立起来（image-text数据充足且clean）；Stage 2在frozen encoder上学temporal pattern，用8 frames；Stage 3 unfreeze encoder让整个系统joint finetune。这种"先冻后解"的protocol在CLIP-style训练里很常见，能避免early stage的noisy gradient破坏pretrained representation。

### Semantic Resampling

针对long-tail问题，paper提出semantic resampling（借鉴DINOv2 Vo et al., 2024）：

1. 用EmbeddingGemma-300M把所有brief action descriptions encode成embedding
2. Hash dedup (7.58M groups → 141.8M duplicates removed)
3. K-means clustering with $k \in \{10^3, 10^4, 10^5\}$
4. 从每个cluster uniform sampling with replacement直到target dataset size

K-means的objective：
$$\arg\min_{\{c_k\}_{k=1}^K, \{r_i\}_{i=1}^N}\sum_{i=1}^N \sum_{k=1}^K r_{ik}\|e_i - c_k\|_2^2$$

其中：
- $e_i \in \mathbb{R}^{300}$：第$i$个action description的EmbeddingGemma embedding
- $c_k \in \mathbb{R}^{300}$：第$k$个cluster centroid
- $r_{ik} \in \{0, 1\}$：assignment indicator ($r_{ik}=1$ iff sample $i$ assigned to cluster $k$)
- $K$：cluster总数
- $N$：sample总数

实验结果（Figure 10）：在10M subset上训练10k steps，$k=10^3$ 比 $k=10^5$ 和no resampling都更好。说明aggressive down-sampling frequent actions确实能提升sample efficiency。

## 4. 主实验结果

Table 3最关键的对比：
- VL-JEPA ViT-L (256px, 3.3B params, 3B samples seen)
- 对比对象：CLIP (13B), SigLIP2 (40B), Perception Encoder (86B)

VL-JEPA在样本量小一个数量级以上的情况下，在motion-focused任务上显著领先：

| Benchmark | CLIP ViT-L | SigLIP2 ViT-L | PE-Core ViT-L | VL-JEPA Stage 3 |
|-----------|------------|---------------|---------------|-----------------|
| SSv2 | 30.7 | 38.6 | 42.9 | **52.5** |
| EK-100 | 3.8 | 5.9 | 9.3 | **19.3** |
| EgoExo4D | 3.7 | 4.5 | 6.0 | **21.8** |
| COIN (step) | 63.5 | 78.5 | 83.3 | **89.6** |
| CrossTask (step) | 20.8 | 35.1 | 37.5 | **64.5** |

EK-100和EgoExo4D这种egocentric + fine-grained motion任务，VL-JEPA比PE-Core提升2倍以上。这说明V-JEPA 2的motion-aware representation + Action100M的dense action supervision是关键。

Figure 8显示3-stage的scaling曲线，log-scale x轴。可以看到：
- Stage 1 → Stage 2有显著跳跃，证明image-only训练不足以capture motion
- Stage 2 → Stage 3 (8 → 32 frames, frozen → unfrozen) 继续提升
- 整体趋势是monotonic improvement with more samples

### Ablation (Figure 9)

固定20k steps训练，对比不同数据源：
- Action100M brief action descriptions > 直接PLM-3B pseudo-labeling
- Action100M detailed captions > PLM-Video-Auto captions
- Ego4D atomic action description对EK-100和EgoExo4D有效，对其他domain无效

这证明hierarchical captioning + LLM aggregation比直接pseudo-labeling更有效。

## 5. 与相关工作的关联与联想

**与V-JEPA 2的关系**：V-JEPA 2 (Assran et al., 2025)是LeCun的JEPA思想在video上的扩展，self-supervised预训练后能emergent地capture motion、enable planning。Action100M用V-JEPA 2做segmentation相当于"让一个已经理解video的模型来决定segment boundary"。

**与Pyramid-of-Captions / Tree-of-Captions**：Chen et al., 2024a的Pyramid-of-Captions最初是为image captioning设计，证明hierarchical caption aggregation能减少hallucination。Action100M把这个idea扩展到temporal dimension，形成Tree-of-Captions。

**与世界模型的关系**：Paper反复强调Action100M对"world modeling"的价值。Reference里有Genie 3 (Ball et al., 2025), Cosmos World Foundation Model (Agarwal et al., 2025a), Gaia-2 (Russell et al., 2025), VL-JEPA本身（Chen et al., 2025c），以及Planning with Reasoning using VL World Model (Chen et al., 2025b)。这些都依赖dense action understanding来预测state transition。

**与PE Video Dataset的对比**：Cho et al., 2025的PE Video Dataset用PLM + Llama-3.3-70B + human refinement，规模0.49 years / 1M videos / 120K clips。Action100M是30×的scale且fully automated。PLM-Video-Auto扩展到6.06 years但仍远小于Action100M的14.6 years。

**与InternVid / Panda-70M的对比**：InternVid (Wang et al., 2024c) 86.8 years / 234M clips，但caption来自Tag2Text + BLIP2，平均17.6 words，偏短偏noisy。Panda-70M用多个cross-modal teacher但平均13.2 words。Action100M的detailed caption平均95.3 words，detailed action 27.8 words，密度高得多。

**与DINOv2 clustering curation**：Vo et al., 2024的automatic data curation用k-means + semantic dedup来curate pretraining data。Action100M的semantic resampling是这个思路在action space的应用——把action description embedding做clustering，然后uniform sample。

**与OpenAI Sora / Video Generation**：虽然paper主要讲understanding，但dense caption对video generation极有价值。Koala-36M (Wang et al., 2025)就是专门为video generation构建的高质量caption dataset，平均202.1 words，但video时长19.6 years且偏生成导向。Action100M的detailed caption (540 words avg!) 可以作为更强的conditioning signal给generation model。

**与HowTo100M的关系**：Action100M的源数据是HowTo100M (Miech et al., 2019)，这是2019年的1.3亿clip dataset。但HowTo100M的caption是ASR-derived，noisy且与visual弱对齐。Action100M相当于对HowTo100M做了"重新标注升级"，用2025年的frontier open-source models替代ASR。

## 6. 关键Reference Links

- Paper GitHub: https://github.com/facebookresearch/Action100M
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- VL-JEPA: https://arxiv.org/abs/2512.10942
- PerceptionLM: https://arxiv.org/abs/2504.13180
- Perception Encoder: https://arxiv.org/abs/2504.13181
- HowTo100M: https://arxiv.org/abs/1906.03334 (Miech et al., ICCV 2019)
- GPT-OSS-120B Model Card: https://arxiv.org/abs/2508.10925
- Pyramid-of-Captions (What makes for good image captions?): https://arxiv.org/abs/2405.00485
- Planning with Reasoning VL World Model: https://arxiv.org/abs/2509.02722
- WorldPrediction benchmark: https://arxiv.org/abs/2506.04363
- EmbeddingGemma: https://arxiv.org/abs/2509.20354
- DINOv2 curation: https://arxiv.org/abs/2405.15613
- Cosmos World Foundation Model: https://arxiv.org/abs/2501.03575
- Genie 3: https://arxiv.org/abs/2507.09404 (DeepMind)
- SigLIP 2: https://arxiv.org/abs/2502.14786
- DataComp: https://arxiv.org/abs/2304.14106 (Gadre et al., NeurIPS 2023)
- InternVid: https://arxiv.org/abs/2307.06942
- Panda-70M: https://arxiv.org/abs/2402.14107
- Koala-36M: https://arxiv.org/abs.2410.08680 (Wang et al., CVPR 2025)
- EgoExo4D: https://arxiv.org/abs/2311.18258
- EPIC-KITCHENS-100: https://arxiv.org/abs/2004.12264
- Something-Something v2: https://arxiv.org/abs/1706.05031
- Kinetics-400: https://arxiv.org/abs/1705.06950
- COIN: https://arxiv.org/abs/1903.04595
- CrossTask: https://arxiv.org/abs/1812.02741
- YouCook2: https://arxiv.org/abs/1709.06054
- MSR-VTT: https://arxiv.org/abs/1610.07002
- ActivityNet Captions: https://arxiv.org/abs/1705.00754

## 7. 我的几点Intuition与延伸思考

1. **"先text化再reasoning"是scaling的关键trick**。如果直接让GPT-OSS-120B看32-frame video input做per-segment reasoning，cost会爆炸（120B参数 × 147M segments × 多轮refine）。改成"轻量VLM生成caption → text LLM聚合"，cost从video-token scale降到text-token scale，整体可承受。这种"text as middle representation"的思路在tool use里很常见，但用在data curation上是新的scale。

2. **Hierarchical segmentation的emergent semantic**：V-JEPA 2 embedding能让Ward linkage自动产生semantic segment，这印证了self-supervised video model的feature已经encode了task structure。这跟DINOv2在image上emergent segmentation properties是类似的phenomenon。

3. **Self-Refine 3轮的ROI**：Round 1产生draft，Round 2-3修正。这是cost-quality的tradeoff。Paper没做Round数量的ablation，是个limitation。我猜测Round 2比Round 1提升显著，Round 3可能marginal——这种diminishing return在iterative refinement里很常见。

4. **Long-tail的semantic resampling**：Figure 10显示$k=10^3$效果最好，意味着aggressive down-sampling frequent actions有正则化效果。这与LM pretraining里的data dedup和balance curation（如Penedo et al.的FineWeb-Edu）思路一致——raw data分布是不平衡的，需要intentional rebalancing。

5. **Cross-domain generalization**：Action100M源数据来自HowTo100M（WikiHow 12 categories的instructional video），但zero-shot在EK-100（egocentric kitchen）、EgoExo4D（多视角skill activity）、Kinetics-400（通用action）上都强。这暗示procedural video学到的action representation有跨domain transfer能力。

6. **关于Embodied AI的下游**：Paper的motivation里反复提到embodied learning、wearable assistive、physical world modeling。Action100M的structured annotations（brief/detailed action + actor + caption）天然适合作为robot learning的semantic prior，特别是action-conditioned world model这种需要action-label的训练范式。

7. **未explored的方向**：
   - Action anticipation（从past segments预测future action）
   - Hierarchical planning（用Tree-of-Captions的tree structure做planning）
   - Action-conditioned video generation（用brief action作为condition）
   - VLM的action reasoning evaluation（很多benchmark只测object-centric reasoning）

整体看，Action100M是LeCun JEPA路线 + Meta FAIR开源生态 + LLM-as-annotator trend三者结合的产物。它把video action understanding从million-scale推到hundred-million-scale，且fully automated。这对整个embodied AI / world model领域是重要的data foundation。

参考的相关web links（补充几个有用的）：
- Yann LeCun on JEPA: https://openreview.net/forum?id=BZ5a1r-kVsf
- Meta FAIR open source: https://ai.meta.com/blog/
- HowTo100M dataset: https://www.di.ens.fr/willow/research/howto100m/
- HuggingFace Datasets for video understanding: https://huggingface.co/datasets?task_categories=task_categories:video-classification
