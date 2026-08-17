---
source_pdf: FineViT Progressively Unlocking Fine-Grained Perception with.pdf
paper_sha256: 3dca4f5602e5b2856fb6fb189244a9fa7cb9ab60bbb51bf31a3b2ab67cba978c
processed_at: '2026-08-04T08:27:02-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FineViT 用人话讲

好，Karpathy，我抛开那些公式和表格，用最直白的方式跟你聊聊这篇 paper 到底在干啥。

## 一句话说清楚

**现在 MLLM 的 vision encoder 是个 bottleneck，原因是 CLIP 系列又模糊又短视。这篇 paper 的核心 idea 是：别想着一步到位，分三步走，每步给 encoder 喂不同粒度的 supervision，从"学会看"到"学会懂"再到"学会精准定位"。**

## 问题到底出在哪

你想想现在 MLLM 的架构。一个 ViT 当 encoder，把图片变成 tokens，塞给 LLM。这个 ViT 基本是 CLIP 或者 SigLIP，pre-trained 好了直接用。

听起来合理，但有几个要命的问题：

**第一，resolution 太低。** CLIP 训练时图片都是 224×224 或 336×336。你拿一张 4K 的照片，硬生生 resize 到 336×336，那图片里的小字、远处的人脸、细小的 logo——全没了。这就像你让一个人戴着老花镜去看远处的招牌，然后问他招牌上写的啥，他肯定说不清。

OCR、grounding 这些任务要的就是细节。你把细节在 encoder 入口就抹掉了，后面 LLM 再强也补不回来。

**第二，数据太糙。** CLIP 那些数据是 web-crawled 的，LAION-400M 之类的。caption 呢？通常是 alt text，就一句话："A photo of a dog"。甚至连狗的品种都没说，更别提狗在干啥、背景是啥、狗旁边有啥。

这种 caption 喂给 model，model 学到的就是"这张图里有个狗"这种粗粒度对齐。你后面让它做 grounding——"图里那个穿红衣服的人左边的狗"——它根本没学过这种 level 的 alignment。

**第三，训练目标打架。** CLIP 用 contrastive learning，目标是让图和文的 global embedding 接近。LLM 用 autoregressive，目标是 token-by-token 预测。这两个目标根本不在一个频道上。

你拿一个 contrastive 训练好的 encoder，去接一个 autoregressive 的 LLM，中间用个 projector 硬接，这就是个"婚后搭伙"的状态——表面上在一个系统里，底下的 representation 其实没真正对齐。

## FineViT 的思路：分三步走

作者的核心 insight 是：**fine-grained perception 不是一蹴而就的，得循序渐进**。

这就像教小孩认东西。你不会一上来就让小孩学"图里左上角那个红色汽车旁边穿蓝衣服的人手里拿的手机的牌子"。你得先让他学会看，再让他学会认，最后让他学会精准定位。

### Stage I：学会"看"（MIM）

第一步，啥 language 都不给，就让 encoder 学会"看懂图片"。

怎么做呢？把图片的 patch 遮住 75%，让 model 猜遮住的部分是啥。猜的时候不是猜 raw pixel，而是猜 DINOv3 这个 teacher model 输出的 feature。这就相当于让 encoder 学 DINOv3 的 representation。

为什么这么做？因为 DINOv3 是 self-supervised 训练出来的，它的 feature 已经包含了很好的 spatial 和 geometric 信息。你让 FineViT 去对齐 DINOv3 的 feature，等于站在巨人肩膀上，避免了从零学视觉表征的 cost。

这一步在 256×256 的低分辨率上做。分辨率低是因为这一步只关心基础视觉表征，不追求细节。batch size 4096，跑 1.8B 图片。

**直觉**：这一步让 encoder 的 weights 有了好的 initialization。它现在知道啥是边缘、啥是 texture、啥是 object 的轮廓了。但它还不知道这些东西用语言怎么描述。

### Stage II：学会"懂"（Contrastive Learning）

第二步，开始引入 language。用 image-text contrastive learning，让 encoder 知道"这张图里的东西用语言怎么表达"。

这里有几个关键设计：

**用 SigLIP loss 而不是 CLIP loss。** CLIP 那个 softmax loss 要求整个 batch 做 normalization，batch size 必须 huge 才有足够 negatives。SigLIP 用 sigmoid，每对 (image, text) 独立判断 positive/negative，batch size 49,152 在工程上能搞定。

**Text encoder 从 SigLIP2 Giant 初始化。** 这很聪明。SigLIP2 Giant 的 text encoder 已经和 visual feature 对齐得很好了，你直接拿来用，等于跳过了 text encoder 的学习成本，只 fine-tune visual branch。

**全程 native resolution。** 这点和 SigLIP2 不一样。SigLIP2 只在最后阶段用 native resolution，FineViT 从头到尾都用。这意味着 model 一开始就见到各种 aspect ratio 的图片，不会形成"图片必须是正方形"的 bias。

**Resolution 渐进从 336² 到 448²。** 一开始低分辨率，后面慢慢提高。这避免了直接高分辨率训练的不稳定性。

**Text context 从 64 到 256 tokens。** 这很关键。因为他们的 data 是 recaptioned 的，caption 比原始 web caption 长得多。CLIP 的 77 token cap 根本 hold 不下这些 caption。

**数据是 recaptioned 的 1.56B pairs。** 他们拿 1.8B 原始图片，用 Qwen2.5-7B、Intern3-VL-8B、MiniCPM-V-8B 三个 MLLM 重新 caption。用三个 model 是为了避免单一 model 的 bias。然后过滤掉 null 和重复的 caption，得到 1.56B 高质量 pairs。

**直觉**：这一步让 encoder 学会了"图里的东西用语言怎么说"。现在它既能看懂 visual content，又知道这些 content 对应什么 text。但这一步学的是 global alignment——整张图和整段 caption 的对齐。

### Stage III：学会"精准定位"（LLM Alignment）

第三步，把 encoder 接到 LLM 上，让它学会"图里某个具体位置的东西怎么说"。

这一步是论文的核心，也是 FineCap-450M 这个 dataset 发挥作用的地方。

**Resolution 提到 1K。** 细节拉满。

**任务是 multi-granularity 的 QA**：
- Global caption QA：整张图的描述
- bbox-to-string：给坐标，描述那个 region
- string-to-bbox：给描述，输出坐标
- bbox-to-ocr / ocr-to-bbox：文字识别与定位

所有这些任务都 reformulate 成 sequence-to-sequence 格式，让 LLM 统一处理。

**Loss 是标准 autoregressive**：给定 visual features 和前面生成的 tokens，预测下一个 token。

**直觉**：这一步让 visual tokens 和 language tokens 在 LLM 内部真正对齐。encoder 输出的 visual features 不再只是"这张图大概是什么"，而是"图里每个位置具体是什么"。这就是 grounding 能力的来源。

## FineCap-450M：真正的杀手锏

说实话，这篇 paper 的 architecture 不算激进——标准 ViT + 2D RoPE，28 层，0.86B params。真正的创新在 data。

FineCap-450M 是 450M 个 region-level annotations，覆盖 80M 张图片。这是迄今最大的 fine-grained annotated dataset。

**怎么构建的呢？**

1. **筛选高质量图片**：short side ≥ 448，排除模糊、过曝、过暗、饱和度过高的。aspect ratio 限制在 [1/3, 3]。

2. **Multi-granularity recaptioning**，用 Qwen3-VL 32B：
   - 先给整张图生成 global caption（150-300 tokens）
   - 从 global caption 里抽 noun phrases
   - 用 Grounding DINO 生成 candidate bounding boxes
   - 过滤 confidence < 0.3 的 boxes
   - NMS (IoU=0.7) 去重
   - **Class-balanced sampling**：高频类 downsampling，低频类全保留。这解决了 long-tail 问题。
   - **Context-aware local captioning**：同时输入 local crop 和 global image，让 MLLM 描述 local detail 时参考 global context

这个 context-aware 的设计很聪明。你只给 local crop，MLLM 可能描述得和全局脱节。你同时给 global image 和 local crop，MLLM 就能在全局语境下描述局部细节。比如 local crop 是一个人，global image 显示这个人在海滩上，那 local caption 会说"一个站在沙滩上的人"而不是只说"一个人"。

3. **Rich text & Document OCR**：自然图里的文字 region，MLLM 的 caption 经常就返回 "text" 这种废标签。他们用 PaddleOCR 检测文字，替换掉这些废 caption。Document 图像同理。

最终 dataset 构成：
- 63M 张图的 global caption
- 226M 个 general region 的 local caption
- 142M 个 rich-text OCR box
- 86M 个 document OCR box
- 总共 454M regions

## 实验结果说明了什么

### Zero-shot retrieval 的大幅领先

| | COCO T2I | COCO I2T | DCI T2I |
|---|---|---|---|
| SigLIP2-so400m | 55.8 | 71.7 | 66.8 |
| FineViT | **60.7** | **80.7** | **84.8** |

COCO I2T 高了 9 个点，DCI T2I 高了 18 个点。这是巨大的 gap。

**为什么这么强？** 因为 FineViT 训练时见到的 caption 都是 dense 的，而且 context 到 256 tokens。它学到的是"图里这个小蓝船、左边那个帆船、天上的鸟"这种 fine-grained alignment。传统 CLIP 学到的是"图里有个码头"这种粗粒度 alignment。

你给它一段长 caption，它能 match 到里面的细节描述。传统 CLIP 给它长 caption，它只能 match 个大概，细节全丢。

### Local task 的压倒性优势

当 FineViT 接入 MLLM 后，local task（OCR、grounding、counting）全面领先：

| | DocVQA | OCRBench | RefCOCO avg | CountBenchQA |
|---|---|---|---|---|
| Qwen3-VL | 89.49 | 846 | 88.04 | 86.65 |
| FineViT-VL | **93.47** | **841** | **90.60** | **93.84** |

DocVQA 高 4 个点，CountBenchQA 高 7 个点。这直接证明了 FineCap-450M 的 region-level supervision 的价值。

### Progressive training 的 ablation

这是最能说明问题的 ablation：

| Stage | General Avg | Local Avg |
|---|---|---|
| Stage I only | 57.12 | 48.04 |
| Stage I + II | 64.80 | 48.57 |
| Stage I + II + III | 73.29 | 70.33 |

你看这个趋势：
- Stage I → II：General task 涨 7.68，Local task 几乎没动（+0.53）
- Stage II → III：Local task 暴涨 21.76

这说明 contrastive learning 主要提升 global semantic，对 spatial localization 帮助不大。真正让 model 学会 grounding 的是 Stage III 的 region-level QA。

**有个有意思的细节**：Stage II 的 RefCOCO 反而比 Stage I 下降了（66.47 → 63.18）。作者的 explanation 是 contrastive objective 关注 global alignment，可能让 features 变得"过于 global"而丢失 local discriminability。Stage III 正好修复了这个。

### Frozen vs Unfrozen backbone

Figure 5 显示，unfreeze backbone 在 local task 上 gains 最大。这说明 frozen encoder 对 general task 够用，但 fine-grained localization 必须更新 backbone。

**直觉**：frozen encoder 已经有 global semantic，但 spatial precision 需要在 task-specific objective 下 refine。这解释了为啥很多 MLLM general VQA 还行但 grounding 很差——backbone 冻住了，spatial features 没 adapt。

## 我的 Take

这篇 paper 传递的 core message 是：**vision encoder 的 fine-grained 能力，归根到底是 data 问题，不是 architecture 问题**。

Architecture 上 FineViT 很保守——标准 ViT + 2D RoPE，没啥花活。真正的 innovation 在 training recipe 和 data curation。

三阶段 progressive training 的设计 logic 很清晰：先建立基础视觉表征（MIM），再建立 global semantic alignment（contrastive），最后建立 fine-grained spatial alignment（LLM + region QA）。每个阶段解决不同 level 的问题，避免 gradient conflict 和 training instability。

FineCap-450M 的 context-aware local captioning 是关键创新。同时输入 local crop 和 global image 让 MLLM 生成 local caption，这避免了 local caption 和全局语义脱节。450M region-level annotations 是迄今最大的，这是真正的 moat。

**但我也有几个疑问**：

1. **三阶段的必要性有多强？** 如果直接从 SigLIP2 Giant 初始化 visual branch，跳过 Stage I，效果会差多少？Stage I 的 MIM initialization 真的不可替代吗？paper 没做这个 ablation。

2. **Stage II 的 native resolution throughout 真的比 SigLIP2 的 final-stage-only 好吗？** 也没看到直接 ablation。

3. **FineCap-450M 的质量如何？** 他们用 Qwen3-VL 32B 生成 local caption，但 Qwen3-VL 32B 自己的 grounding 能力如何？如果 teacher model 的 grounding 不准，那 student model 学到的也会不准。这个 dataset 的 quality assessment 没详细讨论。

4. **Generalization 到 video？** 作者承认只做了 static image。region-level supervision 如何 extend 到 temporal dimension 是个 open question。

5. **Long-context retrieval 的优势是否 overfit 到 recaptioned data 的 style？** DCI、Urban-1k 这些 benchmark 的 caption style 和他们 recaption 的 style 可能很像，导致 inflated performance。

总的来说，这篇 paper 的 contribution 是 solid 的。它证明了一个简单但重要的道理：**你想要 fine-grained perception，就得用 fine-grained supervision 去训**。光靠 contrastive learning 学 global alignment，再指望 LLM alignment 阶段补上 local precision，是不够的。得从 encoder 训练阶段就注入 region-level 的 grounding 信号。

这对你做 MLLM 的启发应该是：**别把 vision encoder 当黑盒，得从 data 和 training recipe 上就为 fine-grained 任务做 design**。

References:
- [FineViT paper](https://arxiv.org/abs/2505.07062)
- [SigLIP2](https://arxiv.org/abs/2502.14786)
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [Grounding DINO](https://arxiv.org/abs/2404.04911)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [LongCLIP](https://arxiv.org/abs/2404.04910)
- [Recaptioning with LLaMA-3](https://arxiv.org/abs/2406.08478)

---

# FineViT 深度技术讲解

这篇 paper 来自 Huawei，核心贡献是提出了一个 **从零训练的 vision encoder**，专门为 fine-grained perception 设计。让我系统性拆解。

## 1. 核心问题诊断

Karpathy 你应该很熟悉 CLIP 系列的痛点，作者归纳了三个：

**(1) Resolution-induced information loss**：传统 encoder 强制 resize 到 224×224 或 336×336，对 OCR、small object grounding 等 dense tasks 是致命的。这本质是个 lossy compression，把高频细节抹掉。

**(2) Data noise**：LAION-400M 这类 web-crawled 数据，caption 简短、错配、缺乏细节。这种 coarse supervision 让模型只学到 global semantic alignment，丢失 local discriminative power。

**(3) Modality gap**：CLIP 用 contrastive learning（global semantic），LLM 用 autoregressive（token-level generative），两个训练目标不兼容，导致 alignment 阶段很 opaque。

## 2. Architecture Details

| Hyperparam | Value | 说明 |
|---|---|---|
| Depth | 28 | 比 ViT-L/14 (24) 略深 |
| Patch Size | 14 | 标准选择 |
| Hidden Size | 1536 | 比 ViT-L (1024) 大 |
| Intermediate Size | 4608 | FFN expansion ratio ≈ 3x |
| Heads | 16 | 96 dim/head |
| Activation | SiLU | Swish activation |
| Position Embed | 2D RoPE | **关键设计** |
| Params | ~0.86B | 介于 SigLIP2-so400m (0.4B) 和 SigLIP2-g (1B) 之间 |

**为什么用 2D RoPE 而不是 absolute position embedding 或 1D RoPE？**

2D RoPE 把位置编码 $(x, y)$ 拆成两半，分别旋转前半和后半的 hidden dim：

$$
\text{RoPE}_{2D}(q, x, y) = R_x \cdot R_y \cdot q
$$

其中 $R_x, R_y$ 是 block-diagonal 旋转矩阵，分别对应 x 和 y 维度。优势是：
- **Native resolution friendly**：不同分辨率、aspect ratio 都能处理，不像 absolute PE 需要插值
- **Relative position inductive bias**：attention 自然编码相对位置，对 grounding 任务友好
- **Length extrapolation**：训练时见过的 resolution 范围可推广到更长序列

输入图片 resize 到最近的 28×28 倍数（28 = patch size 14 的 2 倍），保留 aspect ratio。

## 3. Progressive Training Paradigm（核心方法）

这是论文的精髓。三个 stage，每个 stage 解决不同层次的问题：

### Stage I: MIM Initialization

**目的**：建立 spatial awareness 和 geometric reasoning 的底层表征，不依赖 language。

**Loss**（Eq. 1）：
$$
\mathcal{L}_{\text{MIM}} = \sum_{i \in \mathcal{M}} \| \Phi(\boldsymbol{x})_i - \mathcal{T}(\boldsymbol{x})_i \|_2^2
$$

变量说明：
- $\Phi(\cdot)$：FineViT encoder（student）
- $\mathcal{T}(\cdot)$：DINOv3 teacher network（frozen）
- $\boldsymbol{x}$：input image
- $\mathcal{M}$：masked patch indices（75% masking ratio）
- $i$：patch index
- $\| \cdot \|_2^2$：squared L2 norm

**关键设计点**：
- **Feature-space reconstruction**（而非 pixel-space）：跟随 EVA、MVP 路线，target 是 DINOv3 的 feature，不是 raw pixel。这避免 encoder 浪费 capacity 在 low-level texture reconstruction 上，直接对齐 semantic features。
- **75% masking**：比 MAE 默认的 75% 一致，强制模型从 context 推断 masked region。这种 high masking ratio 意味着 model 必须学到全局结构先验。
- **Resolution 256×256**：低分辨率先建立基础视觉表征。
- **Batch size 4096，lr 1e-3**：标准 self-supervised 配置。

**直觉**：这个 stage 相当于让 encoder 学会"看懂图片"，建立 geometric 和 spatial 先验，但不学语言对齐。这避免了直接 contrastive learning 时，model 既要学视觉表征又要学 alignment，容易出现 shortcut learning。

### Stage II: Large-Scale Contrastive Learning

**目的**：对齐视觉和语言语义空间，建立 global semantic discriminability。

**Loss**（Eq. 2）：SigLIP-style sigmoid loss
$$
\mathcal{L}_{\text{CL}} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{n} \log \left( \sigma(\boldsymbol{z}_{i,j} \cdot \boldsymbol{y}_{i,j}) \right)
$$

其中：
$$
\boldsymbol{z}_{i,j} = \tau \left( \Phi(\boldsymbol{x}_i) \cdot \Psi(t_j) + b \right)
$$

变量说明：
- $n$：batch size
- $\boldsymbol{z}_{i,j}$：image $i$ 和 text $j$ 的 pairwise similarity
- $\boldsymbol{y}_{i,j} \in \{1, -1\}$：正负样本 label
- $\tau$：learnable temperature
- $b$：learnable bias
- $\Phi(\cdot)$：visual encoder（从 Stage I 初始化）
- $\Psi(\cdot)$：text encoder（从 SigLIP2 Giant 初始化）
- $\sigma$：sigmoid function

**为什么用 sigmoid loss 而不是 softmax loss？**

Softmax CLIP loss：
$$
\mathcal{L}_{\text{softmax}} = -\frac{1}{n} \sum_i \log \frac{\exp(z_{i,i}/\tau)}{\sum_j \exp(z_{i,j}/\tau)}
$$

需要整个 batch 的 normalisation，导致 batch size 必须 huge（CLIP 用 32k batch）。SigLIP 用 pairwise sigmoid 解耦了 batch size 和 loss，每个 对独立判断 positive/negative，batch size 49,152 在工程上更可行。

**关键设计点**：
- **Visual branch 从 Stage I 初始化**，text branch 从 SigLIP2 Giant 初始化（已有良好 vision-language alignment）
- **初期 freeze text encoder**，后期 unfreeze joint training
- **Native resolution throughout**：区别于 SigLIP2 只在最后 stage 用 native resolution，FineViT 全程保持。这让 model 一开始就适应 diverse aspect ratios。
- **Resolution scaling 336² → 448²**：渐进提升
- **Text context 64 → 256 tokens**：因为 recaptioned data 比 raw caption 长，需要更长 context
- **Batch size 49,152**：超大 batch，提供足够 negatives

**9.3B samples seen**：相当于在 1.56B unique pairs 上跑 ~6 epochs。

### Stage III: LLM Autoregression with Multi-Granularity Alignment

**目的**：把 visual features 真正整合进 MLLM，赋予 fine-grained perception。

**Loss**（Eq. 3）：标准 autoregressive loss
$$
\mathcal{L}_{\text{MLLM}} = -\sum_{i=1}^{L} \log P\left( y_i \mid y_{<i}, g\left( \Phi(\boldsymbol{x}) \right) \right)
$$

变量说明：
- $y_i$：第 $i$ 个 target token
- $y_{<i}$：前面生成的 tokens（teacher forcing 下是 ground truth prefix）
- $\Phi(\boldsymbol{x})$：visual features
- $g(\cdot)$：trainable projector（2-layer MLP），把 visual features 映射到 language embedding space
- $L$：sequence length

**关键设计点**：
- **Resolution 1K**：大幅提升 spatial precision
- **Multi-granularity tasks**：
  - Global caption QA
  - bbox-to-string：给坐标，输出 region 描述
  - string-to-bbox：给描述，输出坐标
  - bbox-to-ocr / ocr-to-bbox：字符识别与定位双向
- **统一 sequence-to-sequence 格式**：所有任务都变成 token sequence，让 LLM 统一处理

**直觉**：这个 stage 让 visual tokens 和 language tokens 在 autoregressive framework 内对齐，强制 encoder 输出的 visual features 在 LLM 内部能够被精确 grounding 到 spatial location。这是 dense captioning / grounding 能力的来源。

## 4. FineCap-450M 数据集（核心数据贡献）

这是论文最 solid 的部分。450M region-level annotations，是迄今为止最大的 fine-grained annotated dataset。

### Pipeline（Figure 3）

**Step 1: Image filtering**
- Short side ≥ 448 pixels
- 排除 blurry、over/under-exposed、oversaturated
- Aspect ratio ∈ [1/3, 3]
- ~3% 被排除
- 加入 document-like samples

**Step 2: Multi-granularity recaptioning**
用 Qwen3-VL 32B（平衡 compute 和 quality）：

**Global caption**：整个图的描述，~150-300 tokens

**Local caption pipeline**：
1. 从 global caption 抽取 noun phrases
2. 用 Grounding DINO 生成 candidate bounding boxes
3. 过滤 confidence < 0.3 的 boxes
4. NMS (IoU=0.7) 合并重叠 boxes
5. **Class-balanced sampling**：高频类别 downsampling（cap at 100k/batch），低频类别（<1000 instances）全保留
6. **Context-aware local captioning**：同时输入 local crop 和 global image，让 MLLM 描述 local detail 时考虑 global context（这很聪明，避免 local caption 和全局语义脱节）

**Rich text & Document OCR**：
- 自然图像中的 rich text regions，local caption 经常返回不准确的 "text" 标签
- 用 PaddleOCR 检测 text，替换 MLLM 输出的 caption
- Document 图像同样用 PaddleOCR 做 localized text detection

### Dataset Statistics（Table 2）

| 类别 | Images | Regions | Avg tokens |
|---|---|---|---|
| Global Caption | 63M | - | 211.25 |
| Local Caption | 226M | 226M | 56.58 |
| Rich-text OCR | 631,252 categories | 142M | 3.88 |
| Doc OCR | - | 86M | 4.33 |
| **Total** | 80M | **454M** | - |

关键观察：
- **Token length 分布**：Global caption 150-300 tokens，Local caption ~50 tokens，OCR <10 tokens。这反映了不同任务的信息密度。
- **Bbox aspect ratio**：General boxes 接近 1:1，OCR boxes 偏长条形（文字的线性结构）。
- **Class balance**：经过 sampling 后，post-sampling 分布比 raw 分布 balanced 很多。

## 5. 实验结果分析

### 5.1 Zero-shot Classification & Retrieval（Table 4）

| Model | IN-1k val | COCO T2I | COCO I2T | Flickr I2T |
|---|---|---|---|---|
| SigLIP2-so400m | 84.1 | 55.8 | 71.7 | 94.9 |
| SigLIP2-g (1B) | 85.0 | 56.1 | 72.8 | 95.4 |
| FineViT (0.86B) | 84.2 | **60.7** | **80.7** | **96.7** |

**关键洞察**：FineViT 在 classification 上持平 SigLIP2，但在 retrieval 上显著领先（COCO I2T +9.0%）。这表明 dense recaption 数据让 model 学到了更精细的 image-text alignment，而非只是粗粒度 category 级别。

### 5.2 Long-text Retrieval（Table 5）— 这是论文的 killer result

| Model | DCI T2I | DCI I2T | Urban-1k T2I |
|---|---|---|---|
| SigLIP2-so400m | 66.8 | 66.9 | 75.6 |
| LongCLIP-L | 63.8 | 56.2 | 84.0 |
| FixCLIP-L | 74.2 | 72.0 | 96.3 |
| **FineViT** | **84.8** | **83.4** | **99.1** |

DCI T2I 上，FineViT 比 SigLIP2 高 **18 points**，比专门为 long text 优化的 FixCLIP 高 10+ points。

**为什么 FineViT 在 long-text retrieval 这么强？**

1. **Recaptioned data**：1.56B pairs 都是 MLLM 重新 caption 的，平均长度远超 raw web caption（77 token cap 是 CLIP 的瓶颈）。
2. **Text context extension to 256 tokens**：训练时就处理长文本。
3. **Progressive resolution scaling**：从 336 到 448 到 1K，model 学到的 visual representation 既包含 global semantic 又包含 local detail，能匹配 long caption 中的 fine-grained 描述。

Figure 4 的可视化很能说明问题：长 caption 提到 "small blue boats"、"guy working on one of them near the front"、"sailboats on the left"、"cloudy sky with a little blue"、"birds in the sky"——这些是非常 specific 的细节，传统 CLIP 因为压缩到 224² 加上 short caption training，根本 capture 不到这种 level。

### 5.3 MLLM Integration（Table 6）

FineViT-VL vs Qwen3-VL / Intern3.5-VL / Aquila-VL：

| Task | FineViT-VL | Qwen3-VL | Intern3.5-VL |
|---|---|---|---|
| General Avg | 65.90 | 63.20 | 65.51 |
| Local Avg (OCR+Grounding) | **81.44** | 78.02 | 75.32 |
| - OCRBench | 93.41 | 86.46 | 86.65 |
| - DocVQA | 93.47 | 89.49 | 80.76 |
| - RefCOCO avg | 90.60 | 88.04 | 88.28 |
| - CountBenchQA | 93.84 | 86.65 | 70.22 |

**Local task 的优势是压倒性的**。这直接证明了 FineCap-450M 的价值。

### 5.4 Progressive Training Ablation（Table 8）

| Stage | General Avg | Local Avg | OCRBench | CountBenchQA | RefCOCO+ avg |
|---|---|---|---|---|---|
| Stage I only | 57.12 | 48.04 | 417 | 55.00 | 66.47 |
| Stage I+II | 64.80 | 48.57 | 681 | 56.67 | 63.18 |
| Stage I+II+III | 73.29 | 70.33 | 745 | 60.00 | 83.70 |

**关键观察**：
- **Stage I → Stage II**：General task 大幅提升（+7.68），但 Local task 几乎没变（+0.53）。这说明 contrastive learning 主要提升 global semantic alignment，对 spatial localization 帮助不大。
- **Stage II → Stage III**：Local task 暴涨（+21.76），尤其是 RefCOCO+ 从 63.18 到 83.70。这是 LLM alignment + FineCap-450M 的功劳。
- **OCRBench**：417 → 681 → 745，三阶段持续提升。

**有意思的 negative finding**：Stage II 的 RefCOCO (66.47→63.18) 和 RefCOCO+ (76.59→69.51) 反而下降了！作者的解释是 contrastive objective 偶尔 suppresses instance-level spatial sensitivity，因为 contrastive learning 关注的是 global alignment，可能让 features 变得"过于 global"而丢失 local discriminability。这是个很重要的 trade-off，Stage III 正是为了修复这个。

### 5.5 Frozen vs Unfrozen ViT（Figure 5）

Unfrozen 的 gains 在 local tasks 上最大（RefCOCO, ChartQA），frozen encoder 对 general task 够用，但 fine-grained localization 必须更新 backbone。

**直觉**：frozen encoder 已经有良好的 global semantic，但 fine-grained spatial features 需要在 task-specific objective 下进一步 refine。这解释了为什么很多 MLLM 在 general VQA 上好但 grounding 差——backbone frozen 导致 spatial precision 不够。

### 5.6 Cross-LLM Scalability（Table 7）

FineViT vs SigLIP2-naflex，在 Qwen3-1.7B 和 Qwen3-8B 上：

**Local task advantage（Qwen3-1.7B）**：
- DocVQA: +18.52%
- InfoVQA: +13.68%
- BLINK-count: +10.0% (8B version)

这个 gap 在 LLM scale up 时依然存在，说明 FineViT 的优势来自 encoder 本身，不会被 LLM 容量稀释。

## 6. 与相关工作的对比

### SigLIP2 ([arXiv:2502.14786](https://arxiv.org/abs/2502.14786))
- 用 sigmoid loss + dense supervision (masked prediction + self-distillation)
- Native resolution 只在 final stage
- **FineViT 优势**：native resolution throughout + region-level supervision（vs SigLIP2 的 patch-level）

### Seed-ViT ([arXiv:2505.07062](https://arxiv.org/abs/2505.07062))
- 强调 data quality，用 synthetic recaptions
- **FineViT 优势**：更大 region-level dataset（450M vs Seed-ViT 的规模）+ 三阶段 progressive

### Qwen-ViT ([arXiv:2502.13923](https://arxiv.org/abs/2502.13923))
- Naive Resolution + window attention + 2D-RoPE
- **FineViT 优势**：multi-granularity supervision（global + local + OCR）

### AIMv2 ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fini_Multimodal_Autoregressive_Pre-Training_of_Large_Vision_Encoders_CVPR_2025_paper.html))
- Multimodal autoregressive pretraining（next patch + next token）
- Token-level dense supervision
- **FineViT 区别**：contrastive learning 仍是主要 alignment 机制，autoregressive 只在 Stage III

## 7. 我的 Intuition Building 总结

这篇 paper 的核心 insight 是：**fine-grained perception 不能只靠 architecture 或 resolution，而需要 progressive 的 supervision refinement**。

具体来说：

1. **Stage I (MIM)**：建立 "看" 的能力——spatial awareness、geometric reasoning。这是 self-supervised 的，不依赖 noisy label。
2. **Stage II (Contrastive)**：建立 "理解" 的能力——visual semantic 和 language 对齐。这步是 global alignment。
3. **Stage III (LLM alignment)**：建立 "精准定位" 的能力——region-level QA 让 model 学会 visual token 到 spatial location 的精确 grounding。

三阶段的设计动机：**不同 task 需要不同 level 的 supervision，越 fine-grained 越需要后期 task-specific supervision**。如果一开始就用 region-level supervision，model 还没建立基础视觉表征，会训练不稳定；如果只用 contrastive，global alignment 有了但 local precision 不够。

数据方面，**FineCap-450M 的核心创新是 context-aware local captioning**：同时输入 local crop 和 global image，让 caption 既描述 local detail 又保持 global coherence。这避免了 local caption 与全局语义脱节的问题。

**Long-text retrieval 的优势来源**：1.56B recaptioned data（vs raw web caption）+ 256 token context + progressive resolution scaling。这三者结合，让 model 既能处理长文本，又能匹配长文本中的 fine-grained visual detail。

**Limitation**：作者承认只处理 static image，没做 video。这其实是个 open question——region-level supervision 如何 extend 到 temporal dimension。

## 8. 工程实践启示

对于想用或复现 FineViT 的实践者：

1. **Training cost 是巨大的**：1.8B + 9.3B + 0.5B samples，需要 cluster 级别资源。Stage II 的 batch size 49,152 在单机 8×H100 上都很难。
2. **Data curation 是真正的 moat**：FineCap-450M 的构建需要 Grounding DINO + Qwen3-VL 32B + PaddleOCR 的 pipeline，这本身就是个 engineering 项目。
3. **Three-stage 比直接 end-to-end 更 sample efficient**：每个 stage 解决不同 level 的问题，避免 gradient conflict。
4. **2D RoPE 是关键 architecture choice**：如果用 absolute PE，native resolution 的 flexibility 会受限。

## 9. 可能的后续方向

- **Video extension**：region-level supervision 如何 temporal extend？
- **More efficient Stage III**：450M region-level QA 训练昂贵，能否 distill from larger MLLM？
- **Hierarchical resolution**：是否可以让 model 自适应选择 resolution（节省 compute）？
- **Multi-modal RoPE**：2D RoPE for image，能否设计 3D RoPE for video？

References:
- [SigLIP2 paper](https://arxiv.org/abs/2502.14786)
- [Seed-ViT](https://arxiv.org/abs/2505.07062)
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [AIMv2](https://openaccess.thecvf.com/content/CVPR2025/html/Fini_Multimodal_Autoregressive_Pre-Training_of_Large_Vision_Encoders_CVPR_2025_paper.html)
- [LongCLIP](https://arxiv.org/abs/2404.04910) (Springer)
- [FixCLIP](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Fix-CLIP_Dual-Branch_Hierarchical_Contrastive_Learning_via_Synthetic_Captions_for_Better_ICCV_2025_paper.html)
- [Grounding DINO](https://arxiv.org/abs/2404.04911)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [EVA-CLIP](https://arxiv.org/abs/2303.15389)
- [SigLIP original](https://arxiv.org/abs/2303.15343)
- [RoPE (RoFormer)](https://arxiv.org/abs/2104.09864)
- [MAE](https://arxiv.org/abs/2111.06377)
- [LAION-400M](https://arxiv.org/abs/2111.02114)
- [Recaptioning with LLaMA-3](https://arxiv.org/abs/2406.08478)

整篇 paper 我觉得最 solid 的贡献是 FineCap-450M 这个 dataset 和 context-aware local captioning pipeline，long-text retrieval 的结果证明了 dense supervision 的价值。Architecture 本身比较保守（标准 ViT + 2D RoPE），真正的 innovation 在 training recipe 和 data。
