---
source_pdf: CogVLM Visual Expert for Pretrained Language Models.pdf
paper_sha256: bf4a205fbfb39717c4fa898c23fd9beb67c15e1390ef21a6a38fb04e0444c5a0
processed_at: '2026-08-03T16:31:59-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CogVLM 用人话讲讲

## 一句话概括

**冻结LLM不动，给它每一层都配一个"视觉翻译官"，专门负责把图片信息翻译成LLM能听懂的语言。**

---

## 这个paper在解决什么问题？

先说背景。你想把一个已经train得很好的LLM（比如Vicuna-7B）变成一个能看图说话的VLM。摆在你面前有两条路：

**路线A：浅层对齐（BLIP-2, MiniGPT-4, InstructBLIP的做法）**

LLM冻死，ViT也冻死，中间塞一个小adapter，把image feature映射到text embedding space。

听起来很省事，但问题在哪？你想想，image feature经过adapter之后进入LLM第一层，这时候还算"对齐"了。但LLM有32层啊！image feature每经过一层transformer，就在那一层的text-specific weights里被"扭曲"一次。等到第10层、第20层，image feature早就偏离了LLM期望的distribution。这就好比你把一个外国游客送进一家本地餐厅，门口给他配了个翻译，但进到厨房、跟服务员交流、跟厨师沟通，全都没有翻译了，最后他点的东西完全不是他想要的。

**路线B：直接解冻LLM训（PaLI, Qwen-VL, LLaVA-1.5的做法）**

既然浅层对齐不够，那我把LLM也一起训不就行了？确实性能能上去，但代价是**catastrophic forgetting**。Paper里Figure 3给了一个很吓人的数据：用LAION数据直接训LLM，2500步之内MMLU score从正常水平暴跌到24.9，基本等于random guess。PaLM-E的report也提到，8B的LM用于VLM pretraining会导致NLG性能下降87.3%。

这就很尴尬了：浅层对齐性能不够，深层对齐又把NLP能力搞没了。

---

## CogVLM的答案

**能不能两全其美？既深层融合，又不破坏NLP能力？**

CogVLM说：能。方法就是给LLM的每一层都加一个trainable的visual expert module。

### 具体怎么做？

看Figure 4就很清楚了。LLM的每个transformer layer本来长这样：

```
Input → Attention(Q,K,V) → FFN → Output
```

CogVLM改成了这样：

```
Image tokens → Attention with W_I (trainable) → FFN_I (trainable) → Output
Text tokens  → Attention with W_T (frozen)   → FFN_T (frozen)   → Output
```

也就是说，**image tokens和text tokens在同一个layer里走不同的QKV weights和FFN weights**。

用公式说，attention部分：

$$Q = \text{concat}(X_I W_I^Q, X_T W_T^Q)$$
$$K = \text{concat}(X_I W_I^K, X_T W_T^K)$$  
$$V = \text{concat}(X_I W_I^V, X_T W_T^V)$$

- $X_I$: image hidden states
- $X_T$: text hidden states  
- $W_I$: visual expert的weights，trainable
- $W_T$: 原LLM的weights，frozen

FFN同理：
$$\text{FFN}(X) = \text{concat}(\text{FFN}_I(X_I), \text{FFN}_T(X_T))$$

### 这个设计妙在哪？

**妙处1：NLP能力100%保留**

因为text tokens永远只走frozen的$W_T$和$\text{FFN}_T$，如果输入序列里没有image，那整个模型的行为和原Vicuna-7B完全一样。这就是为什么CogVLM不会catastrophic forgetting。

**妙处2：Deep fusion**

Image tokens在每一层都有专属的trainable weights来处理它们，不会被text-specific的weights扭曲。相当于给那个外国游客从门口到厨房到餐桌全程配了翻译。

**妙处3：参数效率**

Visual expert的weights是从LLM的weights初始化的，不是random init。Paper的ablation证明从LLM init比random init好不少（COCO CIDEr 142.8 vs 138.0）。这说明LLM的weights本身就有某种modality-agnostic的computation structure，可以作为visual processing的好起点。

### 灵感来源

作者自己说这个灵感来自P-Tuning vs LoRA的对比：
- P-Tuning：只在input层加prefix embedding → 性能不稳定
- LoRA：在每一层通过low-rank matrix调整weights → 性能更好更稳定

Shallow alignment就像VLM版的P-Tuning（image features只在input层对齐），CogVLM就像VLM版的LoRA（每一层都有adaptation）。

参考：https://arxiv.org/abs/2106.09685 (LoRA), https://arxiv.org/abs/2103.10385 (P-Tuning)

---

## 还有一个很巧的设计：Position Embedding

CogVLM让**所有visual tokens共享同一个position id**。

为什么？想象一下，一张490×490的图片，经过ViT后会变成几百到上千个tokens。如果用常规的RoPE，给这些tokens分配连续的position id 0,1,2,...,999，那么后面的text query会因为RoPE的remote attenuation特性，过度关注离它最近的image tokens，也就是图片的底部区域。这显然不对。

而且，visual tokens在ViT阶段已经encode了spatial information（通过patch的位置），不需要LLM的position embedding再来一遍。所以干脆让所有visual tokens共享一个position id，相当于在position层面把它们看成一个"超级token"，但在content层面保留所有细节。

这个设计很聪明，很多后来的VLM都采用了类似策略。

参考：https://arxiv.org/abs/2104.09864 (RoPE)

---

## 训练流程

Paper里的训练分三个阶段：

### Stage 1: Pretraining on 1.5B image-text pairs
- 数据：LAION-2B + COYO-700M，过滤后约1.5B
- 任务：image captioning（next token prediction）
- 120K steps, batch size 8192, lr 1e-4
- Trainable: visual expert + MLP adapter, 共6.5B参数

### Stage 2: 加入REC任务
- 数据：captioning + Referring Expression Comprehension
- REC格式：Question "Where is the object?" → Answer `[[x0,y0,x1,y1]]`，坐标0-999归一化
- 60K steps, batch size 1024, lr 1e-5
- 前30K steps用224×224分辨率，后30K切换到490×490

为什么分两个resolution？先低分辨率学粗粒度的语义理解，再高分辨率学细节（尤其是OCR和small object）。

### Stage 3: SFT
两个模型：
- **CogVLM-Chat**：通用对话，用VQAv2, OKVQA, TextVQA, OCRVQA, ScienceQA, LLaVA-Instruct等
- **CogVLM-Grounding**：grounding专用，用RefCOCO, Visual7W, VisualGenome等

SFT阶段ViT也解冻了，但lr是其他参数的1/10，为了微调visual encoder但不破坏它。

参考：https://arxiv.org/abs/2301.12597 (BLIP-2的数据策略类似)

---

## 性能有多强？

Paper声称在17个benchmark上SOTA，挑几个最亮眼的：

### Image Captioning
| Model | 数据量 | NoCaps OOD | Flickr |
|-------|--------|-----------|--------|
| GIT2 | 12.9B | 130.6 | 50.7 |
| **CogVLM** | **1.5B** | **132.6** | **94.9** |

用不到GIT2十分之一的数据，性能还更好。Flickr上94.9分，比同时期的Qwen-VL高9.1分。

### LVLM Benchmarks（最看重这些）
| Benchmark | CogVLM | 之前SOTA | 提升 |
|-----------|--------|---------|------|
| MM-Vet | 51.1 | 48.5 (Emu2-33B) | +2.6 |
| MMBench | 77.6 | 71.6 (SPHINX-13B) | +6.0 |
| LLaVA-Bench | 77.8 | 67.7 (LLaVA-13B) | +10.1 |
| MMMU | 41.1 | 34.1 (Emu2-33B) | +7.0 |

注意CogVLM用的是Vicuna-7B，而Emu2用的是LLaMA-33B，参数量差好几倍，CogVLM还是赢。

### Visual Grounding
| Model | RefCOCO val | RefCOCO+ test-A |
|-------|-------------|-----------------|
| UNINEXT-H (specialist) | 92.64 | - |
| ONE-PEACE (specialist) | 92.58 | 92.21 |
| **CogVLM-Grounding (generalist)** | **92.76** | **92.91** |

Generalist模型第一次在grounding任务上打败specialist模型。

### 计算效率
| Model | Pretraining Compute (PFLOPS·days) |
|-------|----------------------------------|
| PaLI-17B | 453 |
| Flamingo-80B | 1381 |
| GIT2 | 5513 |
| **CogVLM** | **230.1** |

CogVLM的pretraining计算量只有GIT2的4.2%。

---

## Ablation Study的几个有意思发现

### 1. 浅层对齐确实不行
只训MLP adapter（BLIP-2式）：COCO CIDEr 131.2
加visual expert：COCO CIDEr 142.8

差了11.6分，这是很显著的差距，直接证明了deep fusion的必要性。

### 2. Visual expert比直接训LLM好
| 方法 | Trainable Params | OKVQA |
|------|-----------------|-------|
| 训LLM + adapter | 6.9B | 56.8 |
| Visual expert | 6.6B | **59.3** |

参数量差不多，但VE在需要external knowledge的OKVQA上明显更好。这很make sense，因为frozen LLM保留了预训练积累的知识，而train LLM可能把这部分知识forget了。

### 3. Causal mask比full mask好
这个很反直觉。按理说bidirectional attention能看更多信息。但CogVLM发现对visual tokens用causal mask效果更好。

我的理解：LLM的每一层都是在causal assumption下训练的，内部的computation pattern已经适配了causal structure。如果突然给visual tokens用bidirectional，会破坏这种pattern，导致visual和text feature的interaction不自然。

### 4. 从LLM init比random init好
| Init | COCO | OKVQA |
|------|------|-------|
| From LLM | 142.8 | 59.3 |
| Random | 138.0 | 55.9 |

这说明LLM的weights不是纯language-specific的，它们包含某种modality-agnostic的computation structure。这个发现对build intuition很重要。

### 5. VE不需要每层都加
VE every 4 layers（1.7B trainable params）已经能达到full VE（6.6B）的大部分性能。这是一个实用的compute-performance trade-off发现。

### 6. Image SSL loss没用
加了CLIP feature的self-supervised loss，下游任务没有提升。大模型里可能被其他signal dominate了。

### 7. Visual encoder大小影响不大
4.4B的EVA2-E换成300M的EVA2-L，大多数benchmark只略有下降。唯一明显的是TextVQA下降2.5分，说明OCR能力更依赖visual encoder的capacity。

---

## 我的Intuition总结

### 1. Modality-specific routing是关键
当你想把一个frozen foundation model扩展到新modality时，应该在每一层都做modality-specific adaptation，而不是只在input层。这个principle可能也适用于audio、video、3D等其他modality。

### 2. Frozen LLM有双重价值
- 保留NLP能力（直接价值）
- 为visual expert提供好的init（间接价值）

所以冻LLM不只是"不破坏NLP"这么简单，它还为visual processing提供了modality-agnostic的computation structure作为起点。

### 3. Causal mask的隐含意义
LLM的computation pattern是和causal structure深度绑定的。即使对visual tokens，也应该保持causal，才能让visual和text feature的interaction自然。这个发现可能对video VLM也有指导意义。

### 4. Position sharing很优雅
让所有visual tokens共享position id，既解决了RoPE的remote attenuation问题，又保留了spatial information（在ViT阶段已encode）。这是工程优雅的体现。

### 5. 与MoE的关系
Visual expert可以看作一种特殊的Mixture of Experts：
- Expert selection by modality（deterministic）
- 而非by token routing（learned）

这避免了MoE的training不稳定，同时获得了modality-specific processing的好处。这个思路可能启发了后来的MoE-LLaVA等工作。

参考：https://arxiv.org/abs/2401.15947 (MoE-LLaVA)

---

## 对后续工作的影响

CogVLM发布后，这个"visual expert"的思路被很多工作借鉴和扩展：

- **CogVLM2**：升级版，支持更高分辨率和video
  https://github.com/THUDM/CogVLM2
- **InternVL**：也采用了类似的多层adaptation策略
  https://arxiv.org/abs/2312.14238
- **DeepSeek-VL**：探索了更高效的visual-text融合
  https://arxiv.org/abs/2403.05525
- **Cambrian**：系统研究了visual encoder的选择和融合策略
  https://arxiv.org/abs/2406.16860

从更大的视角看，CogVLM代表的paradigm是：**保留foundation model的general capability，通过modality-specific experts来扩展capability**。这个思路比直接fine-tune整个model更优雅，也更sample efficient。我觉得这个principle在未来multi-modal model的设计中会越来越重要。

---

## 一句话再总结

CogVLM告诉你：想给LLM加新能力，别动LLM本身，给每一层都配个专属小弟（visual expert）来处理新modality的信息，这样既能深层融合，又不破坏原有能力。简单粗暴但有效。

---

# CogVLM: Visual Expert for Pretrained Language Models 深度解析

## 一、核心问题与动机

这篇paper要解决一个根本性的矛盾:**如何在一个已经pretrain好的LLM上添加视觉理解能力, 同时完全不破坏其NLP性能?**

### 1.1 Shallow Alignment的瓶颈

当前主流的VLM方法(InstructBLIP, MiniGPT-4, BLIP-2)采用所谓的"shallow alignment": 把frozen ViT + frozen LLM, 中间只通过一个trainable的Q-Former或linear adapter连接。这种做法的核心缺陷在于**visual features在进入LLM深层之前, 只经过了一次线性变换**, 而LLM的每一层都是为text token的distribution设计的。当visual features经过多层transformer后, 会逐渐偏离LLM期望的input distribution。

具体来说, 在image captioning任务里, writing style、caption length这种task-specific的信息, shallow alignment只能将其"压缩"到visual features里, 但这些信息本质上不属于visual modality, 表达能力受限。

### 1.2 Catastrophic Forgetting问题

另一个极端是PaLI、Qwen-VL的做法: 直接unfreeze LLM, 让其在image-text pair数据上继续训练。Paper里Figure 3展示了一个触目惊心的现象: 用LAION数据直接训LLM, MMLU score在2500步内从原始水平暴跌到24.9(接近random)。

PaLM-E的report也显示, 把8B的LM用于VLM pretraining会导致NLG performance下降87.3%。这是因为text-only pretraining数据(C4, Raffel et al. 2020)和image-text pair数据(LAION, COYO)的distribution差异巨大, LLM会发生catastrophic forgetting。

### 1.3 CogVLM的洞察

CogVLM的灵感来自P-Tuning vs LoRA的对比:
- P-Tuning: 只在input层加一个task prefix embedding → 性能不稳定
- LoRA: 在每一层通过low-rank matrix调整weights → 性能更好更稳定

Shallow alignment相当于VLM版的P-Tuning, image features就像prefix embedding。CogVLM则选择了VLM版的LoRA思路 —— 在每一层加入trainable的visual expert。

参考链接:
- CogVLM GitHub: https://github.com/THUDM/CogVLM
- LoRA paper: https://arxiv.org/abs/2106.09685
- P-Tuning paper: https://arxiv.org/abs/2103.10385
- BLIP-2: https://arxiv.org/abs/2301.12597
- InstructBLIP: https://arxiv.org/abs/2305.06500
- MiniGPT-4: https://arxiv.org/abs/2304.10592

---

## 二、Architecture深度解析

### 2.1 整体架构

CogVLM-17B由4个组件构成:

1. **ViT encoder**: EVA2-CLIP-E (4.4B参数), 但去掉了最后一层(因为它specialize在aggregating [CLS] features for contrastive learning, 对generation无用)
2. **MLP adapter**: 2层SwiGLU, 把ViT输出映射到text embedding space
3. **Pretrained LLM**: Vicuna-1.5-7B (frozen)
4. **Visual Expert module**: 每层都加, trainable

### 2.2 Visual Expert的核心设计

这是paper最关键的创新。在LLM的每个transformer layer里, image tokens和text tokens使用**不同的QKV matrix和FFN weights**。

#### 公式详解(Attention with Visual Expert):

输入: $X \in \mathbb{R}^{B \times H \times (L_I + L_T) \times D}$
- $B$: batch size
- $H$: attention head数量
- $L_I$: image sequence长度
- $L_T$: text sequence长度  
- $D$: hidden size

X被split成两部分:
- $X_I$: image hidden states, shape $\mathbb{R}^{B \times H \times L_I \times D}$
- $X_T$: text hidden states, shape $\mathbb{R}^{B \times H \times L_T \times D}$

**Attention计算**:
$$\text{Attention}(X, W_I, W_T) = \text{softmax}\left(\frac{\text{Tril}(QK^T)}{\sqrt{D}}\right)V$$

其中:
$$Q = \text{concat}(X_I W_I^Q, X_T W_T^Q)$$
$$K = \text{concat}(X_I W_I^K, X_T W_T^K)$$
$$V = \text{concat}(X_I W_I^V, X_T W_T^V)$$

关键点:
- $W_I = (W_I^Q, W_I^K, W_I^V)$: visual expert的QKV weights, trainable
- $W_T = (W_T^Q, W_T^K, W_T^V)$: 原LLM的QKV weights, frozen
- $\text{Tril}(\cdot)$: lower-triangular causal mask

**FFN with Visual Expert**:
$$\text{FFN}(X) = \text{concat}(\text{FFN}_I(X_I), \text{FFN}_T(X_T))$$

- $\text{FFN}_I$: visual expert的FFN, trainable
- $\text{FFN}_T$: 原LLM的FFN, frozen

### 2.3 为什么这样设计能build intuition?

我的理解是, 这个设计本质上是一种**modality-specific routing**。每个attention head在原LLM里capture某种semantic aspect。当image tokens进来时, 它们的feature distribution和text tokens完全不同。如果用同一套QKV weights处理两者, 会产生distribution mismatch。

Visual expert相当于给image tokens一套"专属的transform weights", 让它们能被transform到LLM能理解的空间。同时text tokens依然走原frozen path, 所以NLP能力完全保留。

这有点像Mixture of Experts的思想, 但是expert的选择是由modality决定的, 而不是by token routing。

### 2.4 Position Embedding的特殊处理

CogVLM让**所有visual tokens共享同一个position id**。原因有二:
1. Visual tokens在ViT阶段已经encode了spatial information
2. 避免RoPE的remote attenuation问题 —— 如果给几百上千个visual token分配连续的position id, 后面的query会过度关注靠近它的image patches(即图片底部)

这个设计选择非常重要, 它实际上把image tokens当成了一个"compressed single token"在position层面处理, 但在content层面保留了所有的spatial detail。

参考链接:
- EVA-CLIP: https://arxiv.org/abs/2303.15389
- SwiGLU: https://arxiv.org/abs/2002.05202
- RoPE: https://arxiv.org/abs/2104.09864
- Vicuna: https://vicuna.lmsys.org/

---

## 三、Pretraining策略

### 3.1 两阶段Pretraining

**Stage 1: Image Captioning**
- 数据: LAION-2B + COYO-700M, 过滤后约1.5B images
- Loss: next token prediction on text part
- 120,000 iterations, batch size 8,192
- Learning rate: 1e-4
- Trainable params: 6.5B (visual expert + adapter)

**Stage 2: Captioning + Referring Expression Comprehension (REC)**
- 60,000 iterations, batch size 1,024
- Learning rate: 1e-5 (cosine decay)
- 前30,000 steps用224×224分辨率
- 后30,000 steps切换到490×490分辨率(提升细节理解)

REC任务的格式:
- Question: "Where is the object?"
- Answer: `[[x_0, y_0, x_1, y_1]]`
- 坐标范围000-999, 是normalized position

### 3.2 Visual Grounding数据集

CogVLM自己构建了40M images的grounding数据集:
- 从LAION-115M(LAION-400M的subset)中过滤
- 用spaCy提取caption中的nouns
- 用GLIPv2预测bounding boxes
- 保留75%+的images包含至少2个bounding boxes

这个数据集很关键, 它让CogVLM具备了visual grounding能力, 这是很多其他VLM不具备的。

### 3.3 SFT阶段: 两个Generalist模型

**CogVLM-Chat**:
- 数据: VQAv2, OKVQA, TextVQA, OCRVQA, ScienceQA, LLaVA-Instruct, LRV-Instruction, LLaVAR
- 6,000 iterations, lr=1e-5, batch size 1,024
- 对VQA用"Question: Short answer:"prompt
- 对对话用"Question: Answer:"prompt
- ViT也unfreeze了, 但lr是其他参数的1/10

**CogVLM-Grounding**:
- 4类grounding数据: GC, REG, REC, GroundedVQA
- 来源: Flickr30K Entities, RefCOCO, Visual7W, VisualGenome, Grounded CoT-VQA

参考链接:
- LAION-5B: https://arxiv.org/abs/2210.08314
- COYO-700M: https://github.com/kakaobrain/coyo-dataset
- GLIPv2: https://arxiv.org/abs/2206.05836
- LLaVA: https://arxiv.org/abs/2304.08485
- spaCy: https://spacy.io/

---

## 四、实验结果深度分析

### 4.1 Image Captioning (Table 1)

| Method | Train Data | NoCaps val OOD | NoCaps val overall | Flickr Karp. | COCO Karp. | TextCaps test |
|--------|-----------|----------------|-------------------|--------------|------------|---------------|
| GIT2 | 12.9B | 130.6 | 126.9 | 50.7 | 145.0 | 145.0 |
| PaLI-X-55B | - | - | 126.3 | - | 149.2 | 147.0 |
| **CogVLM** | **1.5B** | **132.6** | **128.3** | **94.9** | 148.7 | 144.9 |

关键insight: CogVLM只用1.5B数据(比GIT2少一个数量级), 在NoCaps OOD上超出GIT2 2.0分, 在Flickr上超出Qwen-VL 9.1分。这说明deep fusion比单纯scale数据更effective。

### 4.2 VQA & LVLM Benchmarks (Table 2)

最亮眼的数字:

| Benchmark | CogVLM-Chat | 前SOTA | 提升幅度 |
|-----------|-------------|--------|----------|
| MM-Vet | 51.1 | 48.5 (Emu2) | +2.6 |
| SEED-Bench | 72.5 | 71.6 (SPHINX) | +0.9 |
| MMBench | 77.6 | 71.6 (SPHINX) | +6.0 |
| LLaVA-Bench | 77.8 | 67.7 (LLaVA-13B) | +10.1 |
| POPE | 87.9 | 87.7 (Unified-IO2) | +0.2 |
| MMMU | 41.1 | 34.1 (Emu2) | +7.0 |
| MathVista | 34.5 | 33.8 (Qwen-VL) | +0.7 |

特别注意MMMU上41.1分, 这个benchmark是expert-level multimodal understanding, CogVLM比Emu2(LLaMA-33B backbone)高出7分, 而CogVLM只用Vicuna-7B。

### 4.3 Visual Grounding (Table 3)

CogVLM-Grounding在RefCOCO val上达到92.76, 超过了specialist模型UNINEXT-H(92.64)和ONE-PEACE(92.58)。这是generalist模型首次在grounding任务上超越specialist。

RefCOCO+ test-A: 92.91 (超过ONE-PEACE的92.21)
RefCOCOg test: 90.79 (超过UNINEXT-H的89.37)

### 4.4 Computational Efficiency (Table 8)

| Model | Pretraining Data | Pretraining Compute (PFLOPS·days) |
|-------|------------------|----------------------------------|
| PaLI-17B | 1.6B | 453 |
| Flamingo-80B | 2.3B | 1381 |
| GIT2-5.1B | 12.9B | 5513 |
| **CogVLM** | **1.5B** | **230.1** |

CogVLM的pretraining compute仅为GIT2的4.2%, Flamingo-80B的16.7%, 这得益于高质量数据和optimized architecture。

参考链接:
- MM-Vet: https://arxiv.org/abs/2308.02490
- MMBench: https://arxiv.org/abs/2307.06281
- SEED-Bench: https://arxiv.org/abs/2307.16125
- MMMU: https://arxiv.org/abs/2311.16502
- POPE: https://arxiv.org/abs/2305.10355
- MathVista: https://arxiv.org/abs/2310.02257
- RefCOCO: https://arxiv.org/abs/1612.06818

---

## 五、Ablation Study的深刻Insights

这是paper最有价值的部分之一, 我逐条分析:

### 5.1 Tuned Parameters (Table 4)

| Setting | Trainable Params | COCO CIDEr | NoCaps CIDEr | OKVQA | TextVQA |
|---------|-----------------|------------|--------------|-------|---------|
| Only MLP Adapter (BLIP-2式) | 140M | 131.2 | 111.5 | 55.1 | 40.7 |
| LLM + MLP Adapter | 6.9B | 140.3 | 118.5 | 56.8 | 44.7 |
| VE every layer | 6.6B | **142.8** | **120.1** | **59.3** | **45.3** |
| VE every 4th layer | 1.7B | 138.7 | 117.4 | 58.9 | 44.1 |
| Only VE-FFN | 4.4B | 140.0 | 118.7 | 58.2 | 45.1 |

关键发现:
1. **Shallow alignment(只训adapter)performance最差**, 验证了deep fusion的必要性
2. **VE优于直接训LLM**, 尤其在需要external knowledge的OKVQA上(59.3 vs 56.8), 即使trainable params相近(6.6B vs 6.9B)
3. **VE每隔4层加一次**就能达到接近full VE的效果, 可作为compute-performance trade-off
4. **只加VE-FFN(去掉attention part)**也有不错的performance, 说明FFN层的modality-specific processing更重要

### 5.2 Initialization Method

| Init | COCO | NoCaps | OKVQA | TextVQA |
|------|------|--------|-------|---------|
| From LLM | 142.8 | 120.1 | 59.3 | 45.3 |
| Random | 138.0 | 117.9 | 55.9 | 44.0 |

从LLM初始化visual expert明显更好。这说明**transformer在language pretraining后, 已经具备了一定的process visual tokens的能力**, 可以作为multimodal pretraining的更好起点。这个发现对build intuition很重要 —— LLM的weights不仅仅是language-specific的, 它们包含某种modality-agnostic的computation structure。

### 5.3 Visual Attention Mask

| Mask | COCO | NoCaps | OKVQA | TextVQA | VQAv2 |
|------|------|--------|-------|---------|-------|
| Causal | 142.8 | 120.1 | 59.3 | 45.3 | 80.0 |
| Full (bidirectional) | 141.0 | 117.2 | 57.4 | 45.1 | 79.1 |

**反直觉的发现**: causal mask比full mask效果好! 按理说bidirectional attention能access更多信息。作者的解释是causal mask更好地fit了LLM的inherent structure。我的intuition是: LLM的每一层都是trained under causal assumption的, 如果突然给visual tokens用bidirectional, 会破坏LLM内部的某些computation pattern。

### 5.4 Image SSL Loss

加入self-supervised learning loss(每个visual feature预测下一个位置的CLIP feature)并**没有带来提升**。这与PaLI-X的观察一致。但作者提到在小模型early experiment里确实有提升。这说明SSL loss在大模型里可能被其他signal dominate了。

### 5.5 Visual Encoder Scale

| Encoder | Params | TextVQA | 其他benchmark |
|---------|--------|---------|--------------|
| EVA2-E | 4.4B | 45.3 | baseline |
| EVA2-L | 300M | 42.8 (-2.5) | slight decrease |

大visual encoder主要在text-intensive任务(TextVQA)上有明显优势, 其他任务影响不大。这暗示OCR能力更依赖visual encoder的capacity。

### 5.6 EMA

EMA(Exponential Moving Average)在多数任务上带来improvement, 是个稳定的training trick。

参考链接:
- PaLI-X: https://arxiv.org/abs/2305.18565
- EVA-CLIP: https://arxiv.org/abs/2303.15389

---

## 六、与其他方法的深度对比

### 6.1 vs BLIP-2 / InstructBLIP (Shallow Alignment)

| 维度 | BLIP-2 | CogVLM |
|------|--------|--------|
| LLM状态 | Frozen | Frozen |
| Visual-text融合 | 仅在input层(adapter) | 每层(visual expert) |
| Trainable params | ~100M (Q-Former) | 6.5B |
| NLP能力保留 | 完全保留 | 完全保留 |
| VQA performance | 中等 | SOTA |

BLIP-2的Q-Former本质上是一个bottleneck, visual information必须通过它压缩成少量tokens。CogVLM让visual tokens直接进入LLM的每一层, 没有bottleneck。

### 6.2 vs LLaVA-1.5 (直接训LLM)

| 维度 | LLaVA-1.5 | CogVLM |
|------|-----------|--------|
| LLM状态 | Trainable | Frozen + VE |
| Catastrophic forgetting | 有风险 | 无 |
| NLP能力 | 可能下降 | 完全保留 |
| Visual grounding | 弱 | 强 |

LLaVA-1.5通过unfreeze LLM来获得deep fusion, 但代价是可能损害NLP能力。CogVLM通过VE实现deep fusion, 同时完全保留NLP能力。从实验看, CogVLM在多数benchmark上超过LLaVA-1.5-13B, 即使后者用更大的LLM。

### 6.3 vs Qwen-VL / PaLI (直接训LLM)

这些方法选择直接训LLM, 接受catastrophic forgetting的风险。CogVLM证明了这是不必要的 —— visual expert能达到相同甚至更好的performance, 同时保留NLP能力。

### 6.4 vs Mixture of Experts (MoE)思路

CogVLM的visual expert可以看作一种特殊的MoE:
- Expert selection: by modality(image vs text), 而非by token routing
- 两个experts(text expert frozen, image expert trainable)
- 没有routing network, routing是deterministic的

这种设计避免了MoE的training不稳定问题, 同时获得了modality-specific processing的好处。

参考链接:
- LLaVA-1.5: https://arxiv.org/abs/2310.03744
- Qwen-VL: https://arxiv.org/abs/2308.12966
- PaLI: https://arxiv.org/abs/2209.06794
- MoE survey: https://arxiv.org/abs/2202.07165

---

## 七、Build Intuition: 为什么Visual Expert Work?

让我总结几个核心insight:

### 7.1 Distribution Mismatch是核心问题

LLM的每一层都期望特定的input distribution。Shallow alignment只在input层做对齐, 但经过多层transformation后, visual features会drift。Visual expert在每一层都做modality-specific的transform, 保证visual features始终在LLM能理解的distribution内。

### 7.2 深度 > 广度

从ablation看, VE every 4 layers(1.7B params)就接近full VE(6.6B params)的效果。这说明关键是要在**深层**有modality-specific processing, 而非每一层都有。这与deep learning的一般insight一致: 深层的features更task-specific。

### 7.3 Frozen LLM的value

从LLM init优于random init可以看出, LLM的weights包含某种modality-agnostic的computation structure。冻结LLM不仅保留了NLP能力, 还为visual expert提供了一个好的starting point。

### 7.4 Causal Mask的隐含意义

Causal mask优于full mask这个发现很深刻。它暗示LLM的每一层都trained to expect causal structure, 即使对visual tokens也是如此。Bidirectional attention虽然理论上更powerful, 但破坏了LLM的inherent computation pattern。

### 7.5 Position Sharing的智慧

所有visual tokens共享一个position id, 这个设计避免了RoPE的remote attenuation问题, 同时保留了spatial information(在ViT阶段已encode)。这是一个很优雅的工程选择。

---

## 八、Limitations与Future Directions

Paper没有明确讨论limitations, 但从reading between the lines可以看出:

1. **Trainable params doubled**: Visual expert doubled the number of parameters, 但FLOPs相同(因为image tokens数量远少于text tokens in practice)
2. **No RLHF**: Paper提到RLHF是future direction, 当前模型没有alignment
3. **Anti-hallucination**: POPE上87.9分还有提升空间
4. **Video理解**: 当前只支持image, video extension是自然方向

参考链接:
- RLHF survey: https://arxiv.org/abs/2203.02155
- Anti-hallucination in VLM: https://arxiv.org/abs/2310.03641

---

## 九、总结

CogVLM的核心贡献是**证明了deep fusion与NLP能力保留可以兼得**。通过visual expert这一精巧设计, 它在17个benchmark上达到SOTA, 同时完全保留了Vicuna-7B的NLP能力。

这篇paper给我的最大启发是: **当我们想在frozen foundation model上添加新modality时, 应该在每一层都做modality-specific adaptation, 而不是只在input层**。这个principle可能也适用于其他modality(audio, video, 3D)的添加。

从更宏观的角度看, CogVLM代表了一种新的paradigm: 保留foundation model的general capability, 通过modality-specific experts来扩展capability。这比直接fine-tune整个model更优雅, 也更sample efficient。

相关work扩展阅读:
- CogVLM2: https://github.com/THUDM/CogVLM2
- LLaVA-NeXT: https://arxiv.org/abs/2310.03744
- Qwen-VL2: https://arxiv.org/abs/2308.12966
- InternVL: https://arxiv.org/abs/2312.14238
- DeepSeek-VL: https://arxiv.org/abs/2403.05525
- Cambrian: https://arxiv.org/abs/2406.16860
