---
source_pdf: MM1.5.pdf
paper_sha256: 6db6fcdfaa493c217bb606fd28de23f01399bd0b8d51f63291755511a99b00f8
processed_at: '2026-08-05T19:09:19-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MM1.5 用人话说

## 一句话总结

Apple团队把自家MM1模型拿过来，架构一点没改，纯靠调数据配方和训练流程，把性能刷上去了。核心就一句话：**现在MLLM的瓶颈不在架构，在数据**。

---

## 为什么要搞MM1.5？

MM1是Apple去年的工作，pre-training做得很扎实，但有几个毛病：

1. **看不清小字**：pre-training用378×378分辨率，文档上的文字糊成一团，DocVQA这种任务做不好
2. **不会画框**：没法指着图里某个东西说"这就是你问的那个"，grounding能力缺失
3. **多图推理弱**：虽然interleaved data训了不少，但实际多图任务表现一般

MM1.5就是奔着这三个痛点去的。

---

## 他们怎么干的？三个阶段

### Stage 1：Pre-training（沿用MM1）

跟MM1基本一样，大规模image-text对齐，低分辨率378×378。这一步是打地基。

**关键改动**：数据比例从 **45:45:10** 改成 **50:10:40**

啥意思呢？原来是 image-caption : interleaved : text-only = 45% : 45% : 10%。现在变成 image-caption 50%，interleaved 砍到10%，text-only 暴涨到40%。

**为什么砍interleaved？** MM1时代用interleaved data是为了in-context learning，但继续加量边际收益递减了。

**为什么加text-only？** 因为MMMU、ScienceQA这种知识密集benchmark，瓶颈在LLM的语言推理能力，不在vision。你图像看得再清楚，LLM本身不会推理也白搭。

这个调整效果立竿见影：knowledge +0.99，text-rich +0.85，refer&ground +1.4。就改个数据比例，没动模型，白捡的提升。

### Stage 2：Continual Pre-training（新加的）

这是MM1.5最大的创新点。在SFT之前，插一个**高分辨率OCR专项训练**。

用45M的OCR数据（PDFA、IDL、Renderedtext、DocStruct-4M），在1344×1344分辨率下继续pre-train 30K steps。

**为什么需要这一步？** 因为直接从低分辨率跳到SFT的高分辨率，模型对文档细节的适应不够。先在OCR数据上"适应"高分辨率，再去SFT，效果好得多。

**反直觉发现**：如果用378×378做continual pre-training，反而比不做还差。因为低分辨率下看文档，字都糊了，模型学不到有用的东西，反而把原有表征搞坏了。分辨率必须够高，OCR data才有价值。

**合成caption的坑**：他们试了公开的合成caption（LLaVA-Recap-3M、ShareGPT4V-PT），发现没比纯OCR data好。但用自己MM1 3B模型生成的caption就有效。推测原因是**分布匹配**——自己模型生成的caption风格、长度、分布跟下游task更一致。外部大模型生成的caption虽然"质量高"，但风格不对路。

### Stage 3：SFT（精细调参）

SFT数据分成6类：general、text-rich、refer&ground、science、math、code。作者逐一加进来，看每类数据对各个能力的影响。

**核心发现**：
- **text-rich数据是万金油**：加了之后text-rich涨，knowledge也跟着涨（因为MMMU里有很多图表）
- **refer&ground数据有副作用**：加进来grounding大涨，但其他能力轻微下降。学grounding需要"忘掉"一点general的东西
- **要学好grounding，数据量得是general的2倍**（α=2.0）。这是个很重的比例，说明grounding是全新的能力，需要大量数据"灌"进去

最终SFT配比：single-image 80%，multi-image 10%，text-only 10%。

---

## Dynamic Image Splitting：怎么把大图塞进模型？

这是工程上最巧妙的部分。

### 问题

文档图片动辄2000×3000，vision encoder原生只支持672×672。怎么办？

### 朴素方案（Static Splitting）

固定切成2×2网格，每块resize到672×672。问题：正方形图片浪费，长条图片全是padding。

### MM1.5的方案（Dynamic Splitting）

根据图片宽高比，动态选grid。比如长条文档就选(1,4)或(1,3)，正方形就选(2,2)。

**公式核心**：在能覆盖原图的所有grid里，选padding最少的那个。

训练时用 $(n_{min}, n_{max}) = (4, 9)$，推理时可以放宽到 $(4, 16)$，支持到4 Megapixels。这种train-test解耦很聪明——训练省compute，推理拼性能。

### 一个小细节：Overview图放哪？

切完子图后，还要加一张原图缩略图（overview）。放前面还是放后面？

放后面更好。因为autoregressive LLM的causal mask，overview放后面就能"看到"所有子图，相当于做一次全局汇总。放前面的话，overview啥都看不到，后面子图也看不到overview。

---

## MoE：3B参数打7B的脸

MM1.5的MoE版本特别有意思。

**做法**：把dense LLM decoder的FFN换成64个expert，top-2 gating。Vision encoder和connector不动。用GShard的"upcycling"策略——所有expert共享原dense权重初始化，然后训练分化。

**结果**：3B-MoE在knowledge、general、refer&ground、multi-image上都超过7B-dense，唯独text-rich略逊。

**为什么text-rich不行？** 推测是OCR任务需要dense的细粒度表征，expert的稀疏激活对识别小字不利。

**为什么knowledge大涨？** MoE天然适合knowledge-intensive任务，不同expert可以专精不同领域，存的知识量比dense大得多。

这个结果很有启发：**与其训7B dense，不如训3B MoE**。参数效率高得多。

---

## 两个衍生模型

### MM1.5-Video

没设计专门的video架构，就把video当多图输入——均匀采样24帧，每帧144 token，关掉dynamic splitting防序列爆炸。

**Training-free版本**：直接拿image模型跑video，在multiple choice QA上已经能打赢7B的专门video模型。说明multi-image reasoning能力可以从image迁移到video。

**SFT版本**：用ShareGPTVideo + VideoChat2 + ActivityNet-QA微调一下，7B模型在ActivityNet-QA达到60.9%，超过LLaVA-OneVision-7B的56.6%。

### MM1.5-UI

专注iPhone截图理解。用Ferret-UI数据微调。

**震撼结果**：1B模型在elementary UI tasks上**全面超过13B的Ferret-UI**。Refer-i: 90.0 vs 80.5，Grd-i: 86.5 vs 79.4。

为什么1B能赢13B？因为MM1.5的SFT已经培养了很强的refer&ground和OCR能力，UI任务本质上就是这俩能力的应用。Dynamic high-resolution对识别小图标也关键。

---

## 最值得记住的几个Insight

### 1. 数据配方比架构重要

Apple团队用MM1原架构，纯靠调数据就刷出一堆SOTA。这说明现在MLLM的瓶颈在data curation，不在模型设计。

### 2. Continual Pre-training性价比极高

30K steps的OCR continual pre-training，让text-rich性能大幅提升。比单纯堆SFT数据有效得多。相当于给模型"补课"——专门补文档理解这一块。

### 3. Grounding是新能力，需要超量数据

α=2.0意味着grounding数据量要是general QA的2倍。你要教模型一个全新的能力（输出坐标），就得喂足够多的数据，否则学不会。

### 4. 自生成caption比公开caption好

用自己模型生成的caption做continual pre-training，比用外部大模型生成的好。分布匹配比绝对质量重要。这跟VILA²的self-augmentation思路一致。

### 5. MoE upcycling是穷人家的 scaling

3B-MoE > 7B-dense，训练成本远低于从零训7B。把dense model转成MoE，是一种很实用的scaling策略。

### 6. 分辨率要分阶段提升

低分辨率pre-train（省compute）→ 高分辨率continual pre-train（补OCR）→ SFT（调能力）。直接在高分辨率pre-train太贵，分阶段更经济。

---

## 跟其他模型什么关系？

- **vs LLaVA系列**：LLaVA主要做SFT data scaling，MM1.5加了continual pre-training这个中间阶段，更完整
- **vs InternVL2**：InternVL2强在vision encoder scaling，MM1.5强在data recipe。两者互补
- **vs Qwen2-VL**：Qwen2-VL用M-RoPE处理变长，更优雅；MM1.5用grid + position indicator，更explicit
- **vs Cambrian-1**：Cambrian-1搞multi-encoder融合，MM1.5完全不动encoder。一个走vision-side，一个走data-side
- **vs Ferret**：Ferret专做refer&ground，MM1.5把refer&ground整合进general模型里，不牺牲其他能力

---

## 我觉得有什么不足？

1. **所有ablation只在3B上做**，然后直接scale到1B/7B/30B和MoE。MoE对data ratio的敏感度没测
2. **Continual Pre-training这个名字起得不好**，它其实是domain-adaptive pre-training，不是continual learning
3. **没跟Qwen2-VL-2B直接比**，Table 4里InternVL2-2B是2B的，Qwen2-VL-2B也是2B但没列
4. **Video部分比较简单**，没做temporal modeling，长视频（EgoSchema）表现有限
5. **UI部分只有iPhone**，没测Android和desktop，产品导向太明显

---

## 给你的实操建议

如果你要复现或改进这个工作：

1. **先定架构**：vision encoder + connector + LLM，选成熟的组合就行
2. **Pre-training ratio要调**：别盲目45:45:10，text-only给多点（40%），interleaved砍到10%
3. **加continual pre-training**：用OCR data在高分辨率训30K steps，ROI很高
4. **SFT要分6类数据**：text-rich和refer&ground是重点，后者比例要大
5. **Dynamic splitting用(4,9)训练，(4,16)推理**：这是经过ablation验证的好默认
6. **如果预算够，上MoE**：3B-MoE比7B-dense划算

---

## 相关链接

- MM1.5 paper: https://arxiv.org/abs/2409.20566
- MM1 paper: https://arxiv.org/abs/2403.09611
- Honeybee (C-Abstractor): https://arxiv.org/abs/2312.06742
- Ferret: https://arxiv.org/abs/2310.07704
- Ferret-UI: https://arxiv.org/abs/2404.05719
- InternVL2: https://arxiv.org/abs/2404.16821
- LLaVA-OneVision: https://arxiv.org/abs/2408.03326
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- Cambrian-1: https://arxiv.org/abs/2406.16860
- GShard: https://arxiv.org/abs/2006.16668
- ST-MoE: https://arxiv.org/abs/2202.08906
- VILA²: https://arxiv.org/abs/2407.17453
- DocOwl 1.5: https://arxiv.org/abs/2403.12895
- AXLearn: https://github.com/apple/axlearn
- Apple Intelligence Foundation Models: https://arxiv.org/abs/2407.21075
- Set-of-Mark: https://arxiv.org/abs/2310.11441
- SlowFast-LLaVA: https://arxiv.org/abs/2407.15841
- Phi-3-Vision: https://arxiv.org/abs/2404.14219
- MiniCPM-V: https://arxiv.org/abs/2408.01800
- MoE-LLaVA: https://arxiv.org/abs/2401.15947

---

人话版就这些。核心takeaway：**data curation是当前MLLM进步的最大杠杆**，Apple这篇paper把这个杠杆用到了极致。架构不动，数据动，性能照样涨。

---

# MM1.5 深度解读：Data-Centric MLLM 的工程化探索

Karpathy 好，这篇 MM1.5 是 Apple 团队承接 MM1（https://arxiv.org/abs/2403.09611）路线的"工程化续作"，定位上和 MM1 互补：MM1 关注 pre-training 阶段的大规模 ablation，而 MM1.5 关注 pre-training 之后如何通过 continual pre-training + SFT 把基础模型变成一个多能力的实用模型。核心 takeaway 在于——**在 LLM 跑分饱和的今天，MLLM 的进步很大程度来自 data curation 而非架构创新**。作者刻意保留 MM1 架构不变，就是为了隔离架构变量，纯粹做 data-centric 研究。

---

## 1. 三阶段训练 Pipeline 的整体设计

```
Stage 1: Pre-training (378×378, low-res)
   ↓
Stage 2: Continual Pre-training (1344×1344, OCR data, 45M)
   ↓
Stage 3: SFT (Dynamic High-Res, 多能力数据 mixture)
```

### 1.1 为什么需要 Continual Pre-training？

这是 MM1.5 相对 MM1 最显著的改动。MM1 原始 pre-training 主要在 378×378 分辨率，对 text-rich 任务（DocVQA、InfoVQA、ChartQA 这种 document/infographic 理解）能力不足。作者发现，在 SFT 之前插入一个**高分辨率 continual pre-training stage**，专门用 OCR data 训练，能显著提升 text-rich 性能。

这种 stage-by-stage 的 resolution 提升 策略，本质上是一种 **curriculum**：先在低分辨率学会视觉-语言对齐，再在高分辨率学会精细的文本识别。

### 1.2 关键 ablation：分辨率的影响（Figure 9a）

| Continual PT Resolution | Text-rich | Knowledge | General | Refer&Ground | Average |
|---|---|---|---|---|---|
| 378×378 (no split) | 较低 | — | — | — | 低于 no CPT |
| 756×756 (split, no PE interp) | 中等 | — | — | — | 中等 |
| 1344×1344 (split + PE interp) | **最高** | — | — | — | **最高** |

注意一个反直觉发现：**378×378 的 continual pre-training 反而比不做 continual pre-training 还差**。作者推测原因是低分辨率下模型无法从 document OCR data 中获取足够的视觉细节，反而干扰了原有表征。

---

## 2. 架构与 Dynamic Image Splitting

### 2.1 架构沿用 MM1

- **Vision Encoder**: CLIP-based, in-house
- **Vision-Language Connector**: C-Abstractor (Honeybee, https://arxiv.org/abs/2312.06742)
- **LLM Backbone**: in-house, 1B/3B/7B/30B
- **Coordinate tokens**: 支持 `<box>` 和 `<point>` 表示视觉指代

### 2.2 Dynamic Image Splitting 的核心公式

公式 (1) 是这篇 paper 在工程上最值得琢磨的细节：

$$
g^* = \arg\min_{(n_h, n_w) = g \in G} n_h n_w r^2 - h_g w_g
$$

变量含义：
- $g^* $: 最优的 grid 配置，例如 (2,3) 表示 2 行 3 列
- $G$: 候选 grid 集合，约束为 $n_{\min} \leq n_h \cdot n_w \leq n_{\max}$
- $n_h, n_w$: grid 的行数和列数
- $r$: vision encoder 的原生分辨率（例如 378 或 672）
- $h_g, w_g$: 图像在 longer side resize 到 grid 后的实际高宽

约束条件：
- $n_h r \geq h_g \geq h$
- $n_w r \geq w_g \geq w$

这个公式的直觉是：**在保证 grid 能完整覆盖原图的前提下，最小化 padding 区域**。如果没有 grid 能覆盖原图，则选择 resolution loss 最小的 grid（即 scaling down 最少的）。

### 2.3 Global-Local Format 的设计

每张图除 sub-images 外，还额外加一张 overview image（原图 resize 到 encoder 分辨率）。两种放置方式：

- **before**: overview 在 sub-images 之前 → sub-images 可以 attend 到 overview（因为 causal mask）
- **after**: overview 在 sub-images 之后 → overview 可以 attend 到所有 sub-images

实验结果（Table 3）显示 **after 略好**，因为 overview 图像作为"汇总"，能看到所有局部细节后再做整体理解，这跟人类先扫视局部再综合的视觉认知不太一样，但符合 autoregressive LLM 的信息流。

### 2.4 Sub-image Position Indicator

两种方案：
- **Index**: $(k, i, j)$ tuple，$k$ 是 image index，$i, j$ 是行列
- **Seps**: 用 `:` `,` `<n>` 三个 text token 分隔

Table 3 显示，Index 对 refer&ground 略有帮助（74.8 vs 74.0），但对 text-rich 影响不大。最终选择 Index + overview after 的组合。

### 2.5 Train-Inference 分辨率解耦

一个很聪明的工程技巧：

- **Training**: $(n_{\min}, n_{\max}) = (4, 9)$
- **Inference**: $(n_{\min}, n_{\max}) = (4, 16)$，支持高达 4 Megapixels

Table 2 显示，原生训练到 (4, 16) 比仅推理时扩展更好（rows 3 vs 6）。但训练到 (4, 9) 然后推理扩展到 (4, 16) 仍然有部分收益（row 6 vs row 1）。这种解耦策略在 video 等长序列场景下尤其重要——24 帧 × 动态 split 会让 sequence 长度爆炸，所以 video 推理时反而关闭了 dynamic splitting。

---

## 3. SFT Data Mixture 的精细 ablation

这是 paper 最有价值的部分。作者把 SFT 数据分成 6 个 sub-category，逐一研究它们之间的**协同与冲突**。

### 3.1 数据类别与 Category Average Score

定义 **MMBase Score** = (General + Text-rich + Knowledge) / 3，用来衡量"核心能力"。Refer&Ground 和 Multi-image 单独评估。

### 3.2 单类数据的影响（Figure 5）

| 加入的数据 | General | Text-rich | Knowledge | Refer&Ground |
|---|---|---|---|---|
| Base (general) | 基线 | 基线 | 基线 | — |
| + text-rich | 持平 | **大幅↑** | ↑ | — |
| + math | 持平 | ↑ | ↑ | — |
| + science | 持平 | 微↑ | **↑** | — |
| + code | 持平 | 微↑ | 持平 | — |
| + refer&ground | 微↓ | 微↓ | 微↓ | **大幅↑** |

关键发现：
1. **text-rich data 是多面手**，不仅提升 text-rich benchmark，还顺带提升 knowledge（因为 MMMU 等 benchmark 包含大量图表）
2. **refer&ground data 有 negative transfer**，加入后会轻微损害其他能力——这是 trade-off 的核心
3. **science data 量小但有效**，因为 ScienceQA 和 AI2D 是知识密集型

### 3.3 混合比例 α 的搜索（Figure 6）

定义 $\alpha$ = target category 数据量 / general category 数据量（per batch）。

| Category | 最优 α | 备注 |
|---|---|---|
| Science | 0.1 | 数据量本来就小，少量即可 |
| Math | 0.5 | 需要 1:2 的相对比例 |
| Code | 0.2 | 类似 science，少量 |
| Refer&Ground | 2.0 | 需要 2 倍于 general 的数据才能学好 |

Refer&Ground 的 α=2.0 是一个关键发现：要学好 grounding 能力，需要**远多于 general QA 的数据**，因为 grounding 是一个"新能力"，general QA 数据无法覆盖。这与 Ferret（https://arxiv.org/abs/2310.07704）的观察一致。

### 3.4 Single / Multi / Text 的最终比例

$w_{\text{single}} + w_{\text{multi}} + w_{\text{text}} = 1$

最优：$w_{\text{single}}=0.8, w_{\text{multi}}=0.1, w_{\text{text}}=0.1$

注意 text-only 数据的 weight 选择 $w_{\text{text}}=0.1$ 而非更高，是因为 SFT 阶段过多 text-only 数据会挤占多模态数据。这与 pre-training 阶段完全相反——pre-training 中 text-only 占 40%。

---

## 4. Pre-training Data Ratio 的关键调整

### 4.1 从 45:45:10 到 50:10:40

MM1 原始比例：image-caption (45%) : interleaved (45%) : text-only (10%)

MM1.5 调整为：image-caption (50%) : interleaved (10%) : text-only (40%)

这是非常显著的调整——**interleaved data 大幅减少**，**text-only data 大幅增加**。原因：

1. **Knowledge benchmark 的瓶颈在 LLM 而非 vision**：MMMU、ScienceQA 这类任务依赖 LLM 的知识储备，纯文本 reasoning 能力是关键
2. **Interleaved data 的边际收益递减**：MM1 已经证明了 interleaved 对 in-context learning 很重要，但继续增加比例收益有限
3. **评估方式改变**：MM1 用 few-shot pre-training metric 选比例，MM1.5 改用 post-SFT 的下游 benchmark。Few-shot metric 可能无法预测 SFT 后的最终性能

### 4.2 HQ-Text 数据的引入

引入 Apple 的 HQ-Text 数据集（来自 Apple Intelligence Foundation Models paper, https://arxiv.org/abs/2407.21075），这些数据专注于 general knowledge, math, coding，比通用 web text 质量更高。

仅替换 text-only 数据，knowledge average 就提升 0.85 分。再加上比例调整（50:10:40），整体提升：
- Text-rich: +0.85
- Knowledge: +0.99
- Refer&Ground: +1.4
- Multi-image: -0.05（因为 interleaved 减少）

---

## 5. Continual Pre-training 的合成 Caption 之谜

### 5.1 公开合成 Caption 无效（Figure 9b）

作者测试了：
- LLaVA-Recap-3M（基于 LLaVA-NeXT 34B 生成）
- ShareGPT4V-PT
- 两者与 OCR data 混合

结果：**所有 continual pre-trained 模型都比 baseline 好，但合成 caption 没有比纯 OCR data 更好**。这与 LLaVA-OneVision（https://arxiv.org/abs/2408.03326）等工作的结论相反。

### 5.2 In-house 合成 Caption 有效（Appendix A.1, Figure 13）

作者用 MM1 3B 模型 fine-tune 在 8K 人工标注的 paragraph-length caption 上，然后生成 290M web image 的 caption，concept filtering 后得到 7M 高质量 caption。

加入这 7M caption 后，**性能持续提升，且与数据量正相关**（1.4M → 7M）。

为什么公开 caption 无效而 in-house 有效？作者推测：
- Caption 的**风格、长度、分布**至关重要
- LLaVA-Recap 用 34B 生成，但风格可能与 MM1.5 的目标不匹配
- In-house captioner 虽然只有 3B，但 fine-tune 在特定风格的 human caption 上，分布更一致

这呼应了 VILA²（https://arxiv.org/abs/2407.17453）的 self-augmentation 思路——**模型生成的 caption 应该匹配模型自身的"方言"**。

---

## 6. MoE 设计的工程细节

### 6.1 MoE 配置

- 64 experts，替换每两层的 FFN（dense layers）
- Top-2 gating
- Load balance loss: 0.01
- Router z-loss: 0.001（用于稳定训练，参考 ST-MoE https://arxiv.org/abs/2202.08906）
- Vision encoder 和 connector 保持不变，只替换 LLM decoder

### 6.2 MoE 的收益模式

Table 5 的对比显示一个有趣模式：

| Model | Knowledge | General | Text-rich | Refer&Ground | Multi-image |
|---|---|---|---|---|---|
| MM1.5-3B (dense) | 65.7 | 64.3 | 60.0 | 74.6 | 63.1 |
| MM1.5-3B-MoE | **69.9** | **73.3** | 60.1 | **76.1** | **68.0** |
| MM1.5-7B (dense) | 72.2 | 73.4 | 64.5 | 77.7 | 67.5 |

观察：
1. **3B-MoE 在多数 benchmark 上超过 7B-dense**，说明 expert routing 比单纯 scale 参数更高效
2. **Text-rich 例外**：3B-MoE (60.1) 仍不如 7B-dense (64.5)，可能因为 text-rich 任务需要 dense 的 representation，而 expert 的稀疏激活不利于细粒度文本识别
3. **Knowledge 大幅提升**（+4.2）：这符合 MoE 在 knowledge-intensive 任务上一贯的优势，因为不同 expert 可以专精不同领域

### 6.3 与 GShard 的关联

这种 dense → MoE 转换策略直接来自 GShard（https://arxiv.org/abs/2006.16668），即不重新 pre-train MoE，而是把已经训好的 dense model 作为 MoE 的初始化（所有 expert 共享同一个 dense weight，然后训练分化）。这种"upcycling"策略在 Mixtral 等模型上也验证过有效。

---

## 7. 实验结果的关键比较

### 7.1 3B 规模的横向对比（Table 5, 6, 7, 8）

| Benchmark | MM1.5-3B | MiniCPM-V2-3B | InternVL2-2B | Phi-3-Vision-4B |
|---|---|---|---|---|
| MMMU (val) | 37.1 | 38.2 | 36.3 | 40.4 |
| MathVista | 44.4 | 38.7 | 46.0 | 44.5 |
| DocVQA (test) | **87.7** | 71.9 | 86.9 | 83.3 |
| InfoVQA (test) | **58.5** | 37.6 | 58.9 | 49.0 |
| RefCOCO avg | **85.6** | — | 78.3 | 37.6 |
| VL-ICL avg | **56.3** | — | 18.5 | 19.5 |

几个观察：
1. **MM1.5-3B 在 text-rich 和 refer&ground 上压倒性领先**
2. **Phi-3-Vision 在 knowledge 上略强**（因为参数更大，4.2B）
3. **VL-ICL 上 MM1.5-3B 完胜**（56.3 vs 19.5），这是大规模 interleaved pre-training 的红利

### 7.2 30B 规模对比（Table 6）

| Benchmark | MM1.5-30B | Cambrian-34B | GPT-4V | GPT-4o |
|---|---|---|---|---|
| DocVQA | **91.4** | 75.5 | 88.4 | 92.8 |
| InfoVQA | **67.3** | — | — | — |
| ChartQA | **83.6** | 75.6 | 78.5 | 85.7 |

MM1.5-30B 在 text-rich 上大幅超过 Cambrian-34B，且逼近 GPT-4o。考虑到 30B vs GPT-4o 的规模差距，这个结果非常 impressive。

---

## 8. MM1.5-Video：Training-free 与 SFT 双版本

### 8.1 设计哲学

MM1.5-Video 的有趣之处在于它**没有为 video 设计专门架构**。直接把 video 当作 multi-image 输入：
- 24 frames 均匀采样
- 每帧 144 tokens
- 关闭 dynamic splitting（避免序列过长）
- 不做 frame 间 temporal modeling

### 8.2 Training-free 的能力来源

Table 10 显示，MM1.5-Video-3B (training-free) 在 Multiple Choice VideoQA 上**已经超过 7B 的 SlowFast-LLaVA**（如 NExTQA: 72.8 vs 64.2）。这说明 MM1.5 在 SFT 阶段已经具备了 multi-image reasoning 的泛化能力，能 zero-shot 迁移到 video。

但 Open-Ended VideoQA（如 ActivityNet-QA）上 training-free 表现一般，因为 SFT 数据主要是 multiple choice 格式，泛化到自由形式 answer 能力不足。

### 8.3 SFT 数据

| Dataset | Size | 类型 |
|---|---|---|
| ShareGPTVideo | 556K | 开放问答 |
| VideoChat2 | 225K | 多任务 |
| ActivityNet-QA | 31.5K | 开放问答 |

SFT 后 MM1.5-Video-7B 在 ActivityNet-QA 达到 60.9%，超过 LLaVA-OneVision-7B 的 56.6%。

---

## 9. MM1.5-UI：小模型的 UI 理解

### 9.1 与 Ferret-UI 的对比（Table 12）

| Model | S2W | WiC | TaP | Ref-i | Ref-A | Grd-i | Grd-A |
|---|---|---|---|---|---|---|---|
| Ferret-UI-13B | 113.4 | 142.0 | 78.4 | 80.5 | 82.4 | 79.4 | 83.5 |
| MM1.5-UI-1B | 103.0 | 144.4 | 79.3 | **90.0** | **88.6** | **86.5** | **88.2** |
| MM1.5-UI-3B | 103.3 | 145.0 | 80.4 | **90.8** | **89.2** | **87.3** | **88.8** |

1B 模型在 elementary UI tasks 上**全面超过 13B 的 Ferret-UI**。关键原因：
1. MM1.5 的 SFT 已经培养了 strong 的 refer&ground 能力，可以迁移到 UI
2. Dynamic high-resolution 对小图标识别至关重要
3. MM1.5 的 OCR 能力直接服务于 UI 文本识别

### 9.2 一个有意思的 ablation

Table 12 最后两行：
- MM1.5-UI-3B (1 ep.): 完整 MM1.5 SFT 后再训 UI
- MM1.5-UI-3B (1 ep., w/o MM1.5 SFT): 跳过 MM1.5 SFT，直接从 pre-trained checkpoint 训 UI

后者 Ref-i 掉了 2.4 分，说明 MM1.5 的 SFT 数据 mixture 对下游 UI 任务有**迁移价值**，尤其是 refer&ground 数据教会了模型空间坐标的语义。

---

## 10. 关键 Insights 与相关联想

### 10.1 Data Curation > Architecture

整篇 paper 反复证明：在固定架构下，数据 recipe 的调整能带来巨大收益。这与 LLaVA 系列（https://arxiv.org/abs/2304.08485）、Cambrian-1（https://arxiv.org/abs/2406.16860）的趋势一致。MM1.5 把 SFT 数据分成 6 类逐一 ablation 的方法学，应该成为 MLLM 训练的标准操作。

### 10.2 Resolution 的 Curriculum

低分辨率 pre-train → 高分辨率 continual PT → SFT 的三阶段 resolution 提升策略，与近期 SiT（Scalable Interleaved Transformer）、InternVL2（https://arxiv.org/abs/2404.16821）的"coarse-to-fine"思路一致。这种策略可以大幅节省 compute（低分辨率阶段覆盖大量数据），同时保证最终高分辨率能力。

### 10.3 MoE 在 MLLM 中的潜力

3B-MoE > 7B-dense 的现象非常有启发性。Vision encoder 的 dense 计算成本高，但 vision encoder 通常是 frozen 的，所以 MoE 只用在 LLM decoder 上。这种 hybrid 设计（dense vision + sparse LLM）可能是未来 MLLM 的主流方向。联想到最近 MoE-LLaVA（https://arxiv.org/abs/2401.15947）和 CuMo（https://arxiv.org/abs/2405.05949），MoE 在 MLLM 中的应用还远未饱和。

### 10.4 Set-of-Mark vs. Native Grounding

MM1.5 强调的一个差异点：它原生支持 bounding box 输入输出，而不依赖 GPT-4o 那样的 Set-of-Mark (SoM) prompting（https://arxiv.org/abs/2310.11441）。SoM 需要预先用 segmentation model 标注图像区域，然后让 LLM 引用这些标记。Native grounding 则让模型直接输出坐标。

两者各有优劣：
- **SoM** 优势在于精度高（segmentation 模型强），但 pipeline 复杂
- **Native grounding** 优势在于端到端，但训练数据要求高

MM1.5 的 refer&ground α=2.0 说明 native grounding 需要大量数据才能学好。这预示着未来 hybrid 方案：native grounding 处理粗粒度，SoM 处理细粒度。

### 10.5 Self-Training 的 Caption 闭环

Appendix A.1 的 self-augmented captioner 实验很有意思。用 3B MM1 fine-tune 在 8K human caption 上，然后生成 290M caption。这形成闭环：
1. Pre-train MM1
2. Fine-tune captioner
3. Generate captions at scale
4. Use captions for continual pre-training of MM1.5

这呼应了 VILA²、LLaVA-NeXT 的 self-improvement 路线。关键 insight：**captioner 与 downstream model 同源**，分布匹配度高，效果优于外部大模型生成的 caption。

### 10.6 Multi-image SFT 数据的稀缺

Appendix A.3 提到 in-house multi-image 数据只有 ~2K（coco-instruct-interleaved）+ ~500（icl-instruct）。这个数据量级非常小，但效果显著。说明 multi-image reasoning 主要来自 pre-training 的 interleaved data，SFT 只需要少量"激活"就能释放能力。

这与 Flamingo（https://arxiv.org/abs/2204.14198）的 few-shot in-context learning 通过大规模 interleaved pre-training 涌现的观察一致。

### 10.7 与 Qwen2-VL 的对比

Qwen2-VL（https://arxiv.org/abs/2409.12191）采用类似的 dynamic resolution（称为 "Naive Dynamic Resolution"），但用 1D absolute position encoding 处理变长，而 MM1.5 用 2D grid + position indicator。前者更优雅，后者更 explicit。Qwen2-VL 在 video 上做了更多工作（M-RoPE），而 MM1.5-Video 则更简单（uniform sampling，无 temporal modeling）。

### 10.8 UI 理解的 mobile-first 取向

MM1.5-UI 专注 iPhone screenshot，这与 Ferret-UI（https://arxiv.org/abs/2404.05719）一脉相承。考虑到 Apple 的产品定位，这种 mobile-first 的 UI 理解路线很自然。对比 CogAgent（https://arxiv.org/abs/2310.17126）专注 desktop GUI，未来 UI agent 的分化会越来越细。

---

## 11. 我的批判性思考

### 11.1 Continual Pre-training 的命名

"Continual Pre-training" 这个命名有点误导，因为它本质上是一个**domain-adaptive pre-training**（类似 BERT 的 DAPT）。Continual learning 通常指 sequential task learning with forgetting prevention，而这里只是单纯在 OCR data 上继续 pre-train。

### 11.2 SFT Ablation 的 Generalization 风险

所有 ablation 都在 3B dense model 上做，然后直接 scale 到 1B/7B/30B 和 MoE。这种 scaling 假设不一定 always hold——尤其 MoE 的 expert routing 可能对 data ratio 敏感。作者没有提供 MoE 上的 SFT ratio ablation。

### 11.3 Training-Free Video 的"trick"

MM1.5-Video training-free 表现好，部分原因是 Multiple Choice QA 的"答案格式"很容易从 image QA 迁移过来。真正考验 video understanding 的 temporal reasoning（如 EgoSchema），training-free 3B 只有 48.4%，仍然有限。

### 11.4 缺少的对比

- 没有跟 Qwen2-VL-2B 直接对比（只在 Table 4 提了 InternVL2-2B）
- 没有 BLIP-3（https://arxiv.org/abs/2408.08872）的 detailed 比较
- 没有跟 Llama-3.2-Vision 比较（11B/90B）

### 11.5 与 Cambrian-1 的方法论对比

Cambrian-1（https://arxiv.org/abs/2406.16860）也是 data-centric，但聚焦于 vision encoder 的 multi-encoder 融合。MM1.5 则完全不动 vision encoder，只调数据。两种路线代表了 MLLM 优化的两个极端：Cambrian-1 是 vision-side engineering，MM1.5 是 data-side engineering。

---

## 12. 总结：MM1.5 给我们的实操启示

1. **Pre-training ratio 比 SFT ratio 更重要**：从 45:45:10 到 50:10:40 的调整带来的收益（+1.4 refer&ground, +0.99 knowledge）超过很多 SFT 技巧
2. **Continual Pre-training 是高 ROI 的中间阶段**：相对 SFT，它更便宜（30K steps），但对 text-rich 任务提升巨大
3. **Refer&Ground 需要超量数据**：α=2.0 意味着 grounding 数据应该是 general QA 的 2 倍
4. **Dynamic splitting 的工程细节决定 text-rich 上限**：(n_min, n_max) = (4, 9) 训练，(4, 16) 推理是一个好默认
5. **MoE upcycling 性价比极高**：3B-MoE 在多数 benchmark 上 > 7B-dense，训练成本却远低于从零训 7B
6. **Self-generated caption 比 public caption 更有效**：分布匹配比模型大小更重要

---

## References

- MM1: https://arxiv.org/abs/2403.09611
- MM1.5: https://arxiv.org/abs/2409.20566
- LLaVA: https://arxiv.org/abs/2304.08485
- Ferret: https://arxiv.org/abs/2310.07704
- Ferret-UI: https://arxiv.org/abs/2404.05719
- C-Abstractor (Honeybee): https://arxiv.org/abs/2312.06742
- InternVL2: https://arxiv.org/abs/2404.16821
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- LLaVA-OneVision: https://arxiv.org/abs/2408.03326
- Phi-3-Vision: https://arxiv.org/abs/2404.14219
- MiniCPM-V: https://arxiv.org/abs/2408.01800
- Cambrian-1: https://arxiv.org/abs/2406.16860
- DocOwl 1.5: https://arxiv.org/abs/2403.12895
- Idefics2: https://arxiv.org/abs/2405.02246
- Set-of-Mark: https://arxiv.org/abs/2310.11441
- GShard: https://arxiv.org/abs/2006.16668
- ST-MoE: https://arxiv.org/abs/2202.08906
- VILA²: https://arxiv.org/abs/2407.17453
- ShareGPT4V: https://arxiv.org/abs/2311.12793
- VL-ICL Bench: https://arxiv.org/abs/2403.13164
- Mantis: https://arxiv.org/abs/2405.01483
- Flamingo: https://arxiv.org/abs/2204.14198
- CogAgent: https://arxiv.org/abs/2310.17126
- BLIP-3: https://arxiv.org/abs/2408.08872
- MoE-LLaVA: https://arxiv.org/abs/2401.15947
- CuMo: https://arxiv.org/abs/2405.05949
- AXLearn: https://github.com/apple/axlearn
- Apple Intelligence Foundation Models: https://arxiv.org/abs/2407.21075
- SlowFast-LLaVA: https://arxiv.org/abs/2407.15841

希望这个深度解读能帮你 build 起对 MM1.5 的 intuition。如果你想深入某个具体模块（比如 C-Abstractor 的实现细节、或 dynamic splitting 的 padding 计算示例），可以告诉我，我再展开讲。
