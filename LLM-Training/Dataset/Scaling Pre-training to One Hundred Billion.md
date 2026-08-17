---
source_pdf: Scaling Pre-training to One Hundred Billion.pdf
paper_sha256: 6c2e3741f26c2622594ea8a43ed5e7e878f822ff7cfd19422fbcdefb0bd27442
processed_at: '2026-08-12T03:38:32-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，如果我们用最直白的人话来拆解这篇 paper，它其实在讲一个非常反直觉的故事：**当你把 vision-language model 的训练数据从 10B 砸到 100B，你心心念念的 ImageNet 准确率几乎没动，但你的模型突然能看懂因纽特人的冰屋和 Telugu 语的图片描述了。**

这篇 paper 的核心直觉，用一句话总结就是：**多看 10 倍的数据，对于模型“已经见过无数次的东西”没用，但对于“长尾的、罕见的文化和语言”是零到一的质变。**

下面我用更直白的方式，结合底层的技术细节，帮你 build 起对这个现象的 intuition。

---

### 1. 为什么 100B 数据对 ImageNet 没用？

我们先看经典 scaling law 公式：
$$L(N) = \alpha N^{-c} + \varepsilon$$

变量解释：
- $L(N)$: 模型的 error rate
- $N$: training data 的大小
- $\alpha$: 缩放系数，代表数据质量等乘子
- $-c$: 幂律指数（负数），决定曲线下降有多快
- $\varepsilon$: 不可约误差，也就是当数据无限多时，模型性能的“天花板”

对于 ImageNet 这种 Western-centric 的任务，模型在 10B 数据时，$c$ 的绝对值已经变得很小（比如从 -0.92 缩小到 -0.82），意味着曲线开始“躺平”。更关键的是 $\varepsilon$，对于 ImageNet 里的“狗”、“猫”、“跑车”，10B 规模的 web 数据已经把它们覆盖了无数遍。你再给它 90B 新数据，这些数据里关于“狗”的信息全是冗余的，根本无法把 $\varepsilon$ 再往下推。

所以 paper 里做了一个 compute-matched 的对比：
- 在 10B 数据上跑 10 epochs（看 100B 次）
- 在 100B 数据上跑 1 epoch（看 100B 次）

结果发现，在 ImageNet 上，两者几乎打平。**多 epoch 看重复数据，跟看新数据，在 head 概念上没区别。**

### 2. 为什么 100B 对 Dollar Street（文化多样性）有大用？

直觉上，文化相关的概念在 web data 里分布是极度长尾的。

假设一个概念在 web 里的占比是 $p$。在 $N$ 规模的数据集里，它的期望样本数是 $N \times p$。
- 对于英语，$p \approx 35\%$。10B 数据里有 3.5B，100B 里有 35B。早就饱和了。
- 对于 Telugu 语，$p \approx 0.036\%$。10B 数据里只有 3.6M，100B 里有 36M。

3.6M 对训练一个 ViT-L 来说，信号太弱了，模型根本“记不住”长尾特征。但到了 36M，模型开始能从噪声中提取出 Telugu 的语义结构。

这反映在 scaling law 的参数上：对于 Dollar Street 10-shot 任务，ViT-L/16 在 100B 时的 $c$ 是 -0.94，绝对值极大！这说明曲线还在陡峭下降，**根本没有触碰到 $\varepsilon$**。

所以这里的技术 insight 是：**不同任务的 $\varepsilon$ 天然不同，长尾任务的 $\varepsilon$ 所需的临界数据量远大于 head 任务。** 100B 规模的真正价值，是把那些“原本几乎没机会被学到”的概念，推过了可学习的阈值。

### 3. CLIP Filter 是一个“自欺欺人”的陷阱

工业界特别爱用 CLIP score 来过滤 noisy web data。直觉是：用 CLIP 算 image-text 相似度，把分低的（图文不匹配的）扔掉，留下高分的，数据质量不就上去了吗？

这篇 paper 櫟穿了这层美好滤镜。

我们看 SigLIP 的 loss 公式：
$$\mathcal{L} = -\frac{1}{B^2} \sum_{i,j} \log \sigma(z_{ij} \cdot (s_i^\top t_j / \tau - b))$$
其中 $s_i^\top t_j$ 就是 image embedding 和 text embedding 的点积。

CLIP filter 的逻辑是：如果 $s_i^\top t_j$ 高，就是好数据。但是，你用来打分的 CLIP 模型本身就是在充满 Western bias 的数据上训练出来的！它觉得“西装+办公室”是高分匹配，“传统服饰+部落”是低分匹配（因为它没见过）。

所以，用 CLIP 过滤数据，本质上是在做一个**强化已有 bias 的循环**：
1. 模型有 bias -> 
2. 过滤掉长尾数据 -> 
3. 训练出的新模型 bias 更重 -> 
4. 再用它过滤，长尾更少。

paper 里的 Table 10 把这个现象钉死了：在 5B 数据量下，CLIP filter 确实让 ImageNet 0-shot 错误率从 28.17% 降到了 26.76%（好看了！）。但在 Dollar Street 10-shot 上，未过滤的 baseline 错误率是 77.18%，用 CLIP filter 过滤后，错误率反而飙到了 84.59%！

直觉上，你为了把 ImageNet 提高 1.4 个点，直接把模型识别非西方文化的能力废掉了。这在做 inclusive AI 时是致命的。

### 4. Language Rebalancing 的数学逻辑

既然 raw scale 有用，那我们能不能用更聪明的方法模拟 scale 的效果？paper 试了 language rebalancing。

原本 7 个低资源语言总占比 $p_{low} \approx 0.5\%$。作者强行把它们 upsample，每个语言拉到 1%，总占比 7%。

从概率角度看，如果你按原始分布采样，采到低资源语言的 probability mass 极小。Rebalancing 相当于在计算 loss 时，给这些长尾语言乘上了一个巨大的 weight。原本 ViT-L/16 在 100B 数据下，低资源语言的平均 error rate 是 75.01%，rebalance 后直接降到 70.10%。

有趣的是，高资源语言（如英语）的性能几乎没掉（45.43% -> 45.75%）。这说明 ViT-L 的 capacity 足够大，你给低资源语言多一点 representation，并不会挤占高资源语言的内存空间。当然，代价是 Western-centric 的平均 error 从 26.87% 微升到了 27.55%。

这就是一个非常划算的 trade-off：用一点点 head 性能，换取 tail 的巨大提升。

### 5. 为什么对 PaliGemma (Generative Transfer) 提升不明显？

这是个非常 subtle 的技术点。作者把不同 scale 训练的 ViT-L 挂到 PaliGemma 上做 generative pre-training，发现 1B（跑 100 epoch）、10B（跑 10 epoch）、100B（跑 1 epoch）在下游 caption/VQA 任务上几乎没区别。

直觉上，contrastive learning 阶段的 vision encoder 学到的是“全局表征”。当你把它 freeze 住去喂给 LLM 时，LLM 只关心图像里“有哪些大概的 semantic 块”。对于 captioning 这种任务，哪怕你没见过 Telugu 的特定舞蹈，只要你能提取出“人在跳舞”、“有传统服饰”的通用语义，LLM 就能 generate 出还过得去的 caption。

所以，contrastive scale 带来的长尾细节识别能力（比如能 attention 到冰屋的 dome 结构），在 generative transfer 的 coarse semantic 层面被“抹平”了。除非你的下游任务显式 require 那些长尾细节（比如 zero-shot geo-localization），否则 100B 的 scale benefit 在 transfer 时会被吸收掉。

### 总结给你的 Intuition

如果我们像写代码一样来描述这个 paper 的逻辑：

```python
def scale_data(N):
    performance = {}
    # Head concepts 已经 saturate
    performance['western_benchmarks'] = alpha * N**(-c_saturated) + epsilon_west
    # Tail concepts 严重依赖 unique data
    performance['cultural_diversity'] = alpha * N**(-c_steep) + epsilon_culture
    
    # CLIP filter 会干掉 tail 的数据
    if use_clip_filter:
        N_tail = reduce_long_tail(N)
        performance['cultural_diversity'] = collapse(N_tail)
        
    return performance
```

1. **Raw scale 比什么都重要**：如果你要覆盖长尾概念，不要用 CLIP filter，直接喂 raw data，把 $N$ 推到 100B。
2. **多 epoch 是“假 scale”**：在 10B 上跑 10 epoch，模型只是在 head 概念上 overfit，根本碰不到 tail 的分布。Unique data 的多样性才是把曲线推下去的唯一动力。
3. **Scaling law 是 task-dependent 的**：别再用 ImageNet 的 $c$ 和 $\varepsilon$ 来预测 VLM 的未来了。对于 inclusive AI，我们的 $\varepsilon$ 还在天上，远没到 diminishing returns 的时候。

这篇 paper 最大的贡献，就是用 100B 的真实实验证明了：**在追求 AGI 的路上，我们离长尾概念的 saturation 点还远得很，scale up 依然是最有效的手段，只要你不瞎过滤数据。**

参考链接:
- SigLIP: https://arxiv.org/abs/2303.15343
- PaliGemma: https://arxiv.org/abs/2407.07726
- PaLI-3: https://arxiv.org/abs/2310.09199
- Dollar Street: https://datasets.neurips.cc/paper_files/paper/2022/hash/9bba9306c0b8b49ca9b1c4e9a0bb31ab-Datasets_and_Benchmarks-Datasets_and_Benchmarks_track.html
- Crossmodal-3600: https://arxiv.org/abs/2205.12522

---

# Scaling Pre-training to 100B: 一份深度解读

这篇 paper 由 Xiao Wang、Ibrahim Alabdulmohsin、Xiaohua Zhai 等 Google DeepMind 作者完成，核心问题非常直白：**当我们把 vision-language pre-training 数据从 10B 扩到 100B image-text pairs（10 倍量级），到底能得到什么？** 答案出人意料：传统 Western-centric benchmark 几乎没有提升，但 cultural diversity、multilinguality、performance disparity 这类"inclusivity"指标大幅改善。这个发现重新定义了"数据 scale"这件事的价值定位。

---

## 1. 背景与动机：为什么是 100B？

### 1.1 Scaling Law 的形式

paper 在 introduction 给出经典 power law:

$$L(N) = \alpha N^{-c} + \varepsilon$$

变量含义:
- $L(N)$: model 性能度量，比如 error rate
- $N$: training data size (image-text pairs 数量)
- $\alpha$: pre-factor（scaling coefficient），代表 data quality 维度的"乘子"
- $c$: scaling exponent，决定曲线下降速率（绝对值越大递减越快）
- $\varepsilon$: irreducible error / asymptotic limit，也就是当 $N \to \infty$ 时性能的"天花板"

这个公式隐含两个关键 intuition:
1. **diminishing returns**: 加倍 data，性能按 $2^{-c}$ 衰减，绝对提升越来越小
2. **任务相关 $\varepsilon$**: 不同任务有不同的 irreducible error，这决定了"再加 data 还有没有意义"

paper 一个核心发现就是: Western-centric benchmark 的 $\varepsilon$ 已经被 10B 规模 approximated，而 cultural diversity benchmark 的 $\varepsilon$ 仍然远离，所以 100B data 在后者上还有大量 "headroom"。

### 1.2 数据集演进的脉络

paper 引用了一条清晰的演进路线:
- 早期人工标注: COCO Captions (~100K)、Flickr30k (~30K)
- heuristic filtering: Conceptual Captions (3.1M)
- web-scale noisy: LAION-400M → LAION-2B → LAION-5B
- private 工业级: WebLI (10B)
- 本文: WebLI-100B (10× scale up)

之前的 plateau 在 ~10B，所以 100B 是 uncharted territory。

参考链接:
- LAION-5B: https://laion.ai/blog/laion-5b/
- WebLI / PaLI: https://arxiv.org/abs/2209.06794
- DataComp: https://arxiv.org/abs/2304.14108
- SigLIP: https://arxiv.org/abs/2303.15343

---

## 2. WebLI-100B 数据集构建

### 2.1 Raw dataset

- 100 billion image-text pairs，从 web 抓取
- 仅做"essential filters": 移除有害图像、PII
- **不做 quality filtering**，最大限度保留 multilingual 与 cultural diversity
- text source: alt-text + page title
- 对 >90 个常用 VLM benchmark 做 near-duplicate removal，防止 contamination

### 2.2 子集构造

为做 fair comparison:
- **1B subset**: 从 100B 随机采样 1%
- **10B subset**: 从 100B 随机采样 10%
- **100B**: 完整数据集

这种 sample 方式确保三个 scale 的 distribution 一致（除了 sample noise），因此任何性能差异都可以归因于"size"本身，而非"distribution shift"。这是非常干净的实验设计。

### 2.3 Quality-filtered 子集

为研究 data filtering 的影响，作者构造:
1. **CLIP-L/14 filter**: 用 CLIP-L/14 score 过滤后保留 5B English pairs
2. **Classifier filter**: 训练一个 VLM 分类"aligned/misaligned"，调 threshold 保留同样大小的 5B
3. **English baseline**: 从 raw 直接采样 5B English 作为对照

这里的关键是三个 dataset 都是 5B English，所以规模不变，只对比"filtering strategy"。

### 2.4 Language-rebalanced 子集

针对 7 个 low-resource languages: Bengali (bn), Filipino (fil), Hindi (hi), Hebrew (iw), Maori (mi), Swahili (sw), Telugu (te)
- 原始占比: 0.001% ~ 0.267%（见 Table 12）
- 上采样到每个 1%，合计 7%，剩余 93% 是原数据

Table 12 显示 Maori 在数据集中只占 0.001%，Telugu 0.036%。这意味着在 10B 数据里，Maori 大约只有 100K 个 example，几乎不可能训练出能 generalization 的能力。100B 时是 1M，仍然少，但已经进入"可学习"区间。

---

## 3. 训练 setup 细节

### 3.1 模型与训练算法

- 模型: **SigLIP** (Sigmoid Loss for Language-Image Pre-training)
- 与 CLIP 的 softmax-over-batch loss 不同，SigLIP 用 per-pair sigmoid loss:
  $$\mathcal{L} = -\frac{1}{B^2} \sum_{i,j} \log \sigma(z_{ij} \cdot (s_i^\top t_j / \tau - b))$$
  其中:
  - $z_{ij} \in \{+1, -1\}$: 第 $i$ 个 image 与第 $j$ 个 text 是否为正对
  - $s_i, t_j$: image embedding 与 text embedding
  - $\tau$: temperature (learnable)
  - $b$: bias term (learnable)
  - $B$: batch size
  
  这个 loss 不需要全局 softmax，因此 batch size 可以很大，对 scale 友好。

- 模型 size:
  - ViT-B/16 (image & text encoder 对称)
  - ViT-L/16
  - ViT-H/14

### 3.2 关键 hyperparameters

| 参数 | 值 |
|------|-----|
| Batch size | 32K |
| LR schedule | inverse sqrt |
| Peak LR | 0.001 |
| Weight decay | 0.0001 |
| Warmup examples | 200M |
| Cooldown examples | 200M |
| Image resolution | 224 × 224 |
| Text tokenizer | mT5 multilingual |
| Max text length | 64 tokens |
| Total seen examples | up to 100B |

### 3.3 Compute-matched comparison 的精髓

这是个非常巧妙的设计:
- ViT-L 在 1B 上训练 100 epochs → 100B seen examples
- ViT-L 在 10B 上训练 10 epochs → 100B seen examples
- ViT-L 在 100B 上训练 1 epoch → 100B seen examples

三者在 seen examples 上匹配，**但唯一变量是"unique data 量"**。如果 10 epochs × 10B 与 1 epoch × 100B 表现接近，说明模型已经 saturate，更多 unique data 没用；如果后者更好，说明 unique data 还能提供新的信息。

Cooldown 在 seen 3, 7, 10, 17, 26, 33, 49, 66, 100B 处分别 evaluate，这样得到的是完整的 scaling curve，而不只是终点。

---

## 4. Scaling Law 拟合: 揭示 asymptotic limit

paper 在 Table 2 和 Table 3 给出三个 scale 下的 scaling law 拟合: exponent 和 limit。fit recipe 来自 Alabdulmohsin et al. (NeurIPS 2022)，参考: https://arxiv.org/abs/2305.16266

### 4.1 Western-centric 结果

以 ViT-L/16 ImageNet 0-shot 为例:

| Scale | err (%) | exponent | limit (%) |
|-------|---------|----------|-----------|
| 1B | 31.2 | -0.92 | 30.7 |
| 10B | 29.7 | -0.91 | 29.0 |
| 100B | 28.5 | -0.82 | 27.1 |

观察:
- exponent 从 -0.92 → -0.91 → -0.82，**绝对值在缩小**，意味着"递减收益"已经显著，再加 data 边际收益递减
- limit 从 30.7% → 27.1%，收敛中
- 1B → 100B 误差绝对减少 2.7%，但 10B → 100B 只减少 1.2%

Wilcoxon signed-rank test 给出 p-value = 0.9，**statistically not significant**。

### 4.2 Cultural diversity 结果（关键差异！）

ViT-L/16 Dollar Street 10-shot:

| Scale | err (%) | exponent | limit (%) |
|-------|---------|----------|-----------|
| 1B | 63.6 | -1.09 | 63.2 |
| 10B | 64.1 | -0.38 | 60.1 |
| 100B | 58.3 | -0.94 | 57.5 |

观察:
- 10B → 100B 误差减少 5.8%，**绝对提升远超 Western-centric 任务**
- exponent 在 100B 处是 -0.94，绝对值更大，说明曲线**远未 saturate**
- limit 57.5%，但当前 100B 还在 58.3%，说明距离 limit 还有空间

ViT-H/14 Dollar Street 10-shot:
- 10B: 50.2% → 100B: 44.7%（误差下降 5.5%，更强模型上更明显）

Wilcoxon test p-value = 0.002，**99% 置信水平显著**。

### 4.3 直觉: $\varepsilon$ 不同导致饱和点不同

为什么会出现这种"分歧"? 关键是不同 benchmark 的 $\varepsilon$（irreducible error）对应所需的数据覆盖量不同:

- **ImageNet/COCO**: 概念主要来自 Western-centric 视觉世界（如 specific 的 car model、特定 bird species），web data 在 10B 已经"看过无数次"，再 unique data 是 redundancy。
- **Dollar Street/GeoDE**: 概念是 long-tail 的文化相关物体（如 Igorot dance 装饰、Igloo 结构、Indian puja 物品），每个概念在 web data 中出现频率低，10B 看到的不够多，100B 才开始 accumulate 足够信号。

这与 Sharma & Kaplan 提出的"数据流形维度"理论吻合: high-dimensional 的概念 manifold 需要 $O(C^{d})$ 样本才能 well-sampled，其中 $C$ 是概念数量，$d$ 是 inherent dimension。文化概念的 $C$ 很大且 frequency 不均，所以需要 10× scale 才能从 0.001% Maori 类的稀疏分布里"挤出" signal。

---

## 5. Multilinguality: 低资源语言显著受益

Crossmodal-3600 是 36 语言、3600 张图的 zero-shot retrieval benchmark。reference: https://arxiv.org/abs/2205.12522

### 5.1 关键观察

Figure 3 显示一个**分化的 pattern**:
- High-resource languages（English, Spanish, German 等）: 10B → 100B 的提升很小
- Low-resource languages（Bengali, Telugu, Swahili 等）: 提升显著

具体数据（Table 8, ViT-L/16）:
| Language | 1B | 10B | 100B | Δ(100B-10B) |
|----------|----|----|------|----|
| Telugu (I2T) | 96.08 | 76.67 | 69.69 | -6.98 |
| Swahili (I2T) | 89.94 | 81.33 | 79.47 | -1.86 |
| Bengali (I2T) | 90.81 | 66.36 | 63.75 | -2.61 |
| English (I2T) | 54.14 | 52.42 | 51.67 | -0.75 |

注意这是 error rate，所以负值代表提升。

### 5.2 为什么 Telugu 提升如此显著?

Telugu 在数据集中只占 0.036%，意味着:
- 10B 数据里: 3.6M Telugu example
- 100B 数据里: 36M Telugu example

10 倍的 unique data 直接让模型从"几乎没学到"过渡到"开始能 retrieve"。这是从 0 到 1 的相变，而 English 在 10B 已经过饱和点了。

Figure 1 右侧用一个例子说明: Telugu 在 web 上只占 < 0.04%，所以 100B 才让它进入"learnable" regime。

### 5.3 模型 size 越大，gap 越大

Figure 3 显示一个值得注意的现象: ViT-H 上 low/high resource 的 gap 比 ViT-B 上更大。这暗示**更大的模型能更好地利用额外 unique data**，符合 "compute-optimal" scaling 的直觉——给定足够 data，big model 更能 exploit 长尾结构。

---

## 6. Fairness 三维分析

### 6.1 Representation Bias (RB)

paper 用 1st-order statistics 度量 model 对 gender label 的"偏好"。

Table 4 ViT-L/16:

| Data | 1B | 10B | 100B |
|------|----|----|------|
| Male preference (%) | 88.2 | 86.4 | 85.5 |

观察:
- 100B 时仍 85.5% 偏向 Male label，**scaling 不会自动 fix bias**
- 1B → 100B 微弱下降，但仍然严重
- 作者明确说"this finding aligns with previous research highlighting the necessity of bias mitigation strategies, such as data balancing"——单纯 scale 不能替代 explicit debiasing

### 6.2 Association Bias (AB)

Figure 2 测试 gender × occupation 的关联，用 Fairface 数据集（https://arxiv.org/abs/2108.04784）。对比 5 个 occupation pair:
- librarian vs. scientist
- nurse vs. doctor
- housekeeper vs. homeowner
- receptionist vs. executive
- secretary vs. manager

对每个 pair，用 model 对 male/female 图像分别预测两个 label 的 preference 概率。如果 model 倾向把 female 图预测为 "nurse"，把 male 图预测为 "doctor"，则存在 association bias。

关键发现: scaling from 10B → 100B **没有显著减少** association bias。这跟 RB 的结论一致——bias 是 dataset 内 inherent 的 statistical pattern，scaling 反而可能**强化**这种 pattern（数据越多，pattern 越显著）。

### 6.3 Performance Disparity

定义: 各 subgroup 间的 max accuracy gap。Table 5:

Dollar Street 按 income 分组（ViT-L/16）:

| Income | 1B | 10B | 100B |
|--------|----|----|------|
| 0-200 | 33.7 | 35.7 | 33.7 |
| 200-685 | 44.7 | 47.8 | 46.6 |
| 685-1998 | 57.3 | 58.7 | 59.5 |
| >1998 | 63.4 | 65.5 | 64.1 |
| **Disparity** | **29.7** | **29.8** | **30.4** |

Hmm，disparity 反而略升? 但 GeoDE 按 region（ViT-L/16）:
| Region | 1B | 10B | 100B |
|--------|----|----|------|
| Africa | 92.0 | 91.8 | 93.5 |
| Americas | 94.0 | 94.4 | 95.1 |
| East-Asia | 94.0 | 94.0 | 95.4 |
| Europe | 95.2 | 95.8 | 96.2 |
| SE Asia | 94.2 | 94.2 | 95.0 |
| West Asia | 94.9 | 94.7 | 95.8 |
| **Disparity** | **3.2** | **4.0** | **2.8** |

ViT-H/14 上 disparity 从 3.6 → 2.7。

所以综合来看，scaling 在 cultural diversity 提升的同时**也会改善 disparity**。这是 paper 声称 100B 对"inclusivity"重要的核心证据之一。

---

## 7. Data Filtering 的双刃剑

这是 paper 最有 practical value 的一节。

### 7.1 CLIP filter 的标准做法

工业界用 CLIP score $s = \cos(I, T)$ 作 threshold 来过滤 noisy web data。这确实提升 Western-centric benchmark，paper Table 10 也验证:
- ImageNet 0-shot, 5B examples seen: CLIP filter err 26.76 vs baseline 28.17

但 Table 10 同时显示 cultural diversity benchmark 表现**变差**:
- Dollar Street 10-shot: CLIP 84.59 vs baseline 77.18（CLIP "好"了，但实际 err 高出 7.4 个点？）

等等，仔细看 Table 10 的 Dollar Street 10-shot 行: 
- Baseline (en): 84.87 → 79.27 → 77.18 → 76.21 → 72.54
- CLIP filtered: 88.86 → 84.59 → 84.73 → 82.80 → 82.80

哦不对，paper 用的 metric 是 error rate，越低越好。所以 baseline 72.54 比 CLIP filter 82.80 **更好**。这里 CLIP filter 反而损害 cultural diversity performance。这跟 paper 的 claim 一致。

### 7.2 Mechanism: 为什么 CLIP filter 会 hurt diversity?

我的理解: CLIP filter 是用**已经在 Western-centric 数据上训练好的 model** 来给 data 打分。CLIP 本身就有 Western-centric bias（比如对"婚礼"的概念偏向西方白色婚纱），所以打分时:
- 高分: 与 CLIP 已有 concept manifold 接近的 sample（多 Western）
- 低分: long-tail 或 culturally distant 的 sample

这等价于一个**自我强化的 bias loop**: 用 biased model 过滤 → 得到更 biased 数据 → 训练更 biased model → ...

Figure 4 的"middle"子图就是显示这个 gap。

### 7.3 "Other filtered"（classifier-based filter）

作者还训练了一个 VLM 做 aligned/misaligned 分类器过滤。这个 filter 表现介于 raw 和 CLIP filter 之间，但也比 raw baseline 差。

这暗示**任何 quality filter** 都可能损害 cultural diversity，不仅 CLIP。

paper 的 limitation section 明确说"We focus on the impact of scaling raw, unfiltered data, and leave the improvement of data quality at the 100 billion scale for future work."——这是诚实的研究态度。

---

## 8. Language Rebalancing: 上采样低资源语言

### 8.1 设计

Section 5.2 把 7 个低资源语言从总占比 0.5% 上采样到每个 1%，总 7%，剩余 93% 原数据。

### 8.2 结果（Table 11, ViT-L/16, 100B seen）

对比 Before vs After rebalancing:

| Metric | Before | After |
|--------|--------|-------|
| Avg Multiling: Low-Resource Lang | 75.01 | 70.10 |
| Avg Multiling: High-Resource Lang | 45.43 | 45.75 |
| Avg Cultural Diversity | 44.01 | 43.29 |
| Average Western-centric | 26.87 | 27.55 |

关键观察:
- **Low-resource 大幅提升**（error rate 从 75 → 70，绝对 5 点）
- **High-resource 几乎不变**（45.43 → 45.75）
- **Western-centric 略降**（26.87 → 27.55）

这是一种 trade-off: 牺牲 Western-centric benchmark 一点点，换取 low-resource 语言大幅提升。**实用上**完全值得，特别当目标是 inclusive multimodal system。

### 8.3 这个实验的 broader implication

scaling data 本身就能让 long-tail 收益，但如果想进一步 explicit 把 representation 拉平，upsampling 是最直接的方法。这跟 LLM 里 Llama-3 之类模型做的 language rebalancing 思路一致。

---

## 9. Transfer 到 Generative Models

### 9.1 PaliGemma 实验

paper 把 contrastively trained ViT-L/16 作为 PaliGemma 的 vision encoder，做 stage-1 pretraining 50M examples，224×224 resolution。两个 setup:
- **Frozen**: ViT 权重不动
- **Unfrozen**: 联合 fine-tune

Table 6 给出 aggregate 结果:

| ViT-L pretrained on | Semantics | OCR | Multiling | RS | Avg |
|---|---|---|---|---|---|
| Frozen 1B | 76.0 | 66.8 | 67.0 | 92.3 | 73.6 |
| Frozen 10B | 75.4 | 65.2 | 66.3 | 91.9 | 72.7 |
| Frozen 100B | 76.4 | 67.0 | 66.9 | 92.1 | 73.9 |
| Unfrozen 1B | 77.1 | 69.5 | 66.9 | 92.0 | 75.1 |
| Unfrozen 10B | 76.4 | 66.9 | 66.0 | 91.8 | 73.7 |
| Unfrozen 100B | 77.2 | 70.0 | 67.0 | 91.8 | 75.3 |

观察:
- Frozen setup: 100B 比 10B 略好（73.9 vs 72.7）
- Unfrozen: 100B 比 10B 略好（75.3 vs 73.7）
- 但 1B（100 epochs）和 100B（1 epoch）几乎相同

paper 的解读: "when taking the noise level into consideration, we do not observe consistent performance gains across downstream tasks as we scale the pre-training dataset."——也就是说 generative transfer 这条路径下，contrastive encoder 的 scale benefit 比较微弱。

### 9.2 一个有趣的对比

这个结果 vs. cultural diversity benchmark 上的大幅提升，暗示:
- 当下游任务的 distribution 与 pretraining 数据 close（如 generative captioning on COCO），contrastive encoder 的 inclusive 能力"用不上"
- 当下游任务**直接 measure** inclusive（如 Dollar Street geolocalization），scale benefit 立即显现

这跟 PaLI-3 论文（https://arxiv.org/abs/2310.09199）的发现一致: web-scale contrastive encoder 对 generative VLM 的 transfer benefit 主要在 long-tail task。

---

## 10. Attention Map 定性分析

Table 1 和 Table 7 用 attention map visualization 对比 1B/10B/100B 训练的 ViT-L/16。

例子:
- **Igorot Dance**: 100B 模型能 focus 在传统装饰的 fine-grained pattern，1B 只看到 image 整体
- **Igloo**: 100B 模型 focus 在 dome 结构，1B/10B 被 background（mountains, ice）分散
- **Bison**: 100B 模型更精确 focus 在 bison 本身，而非 landscape

这暗示 scale 不仅改善 representation quality，还改善 **object localization**（attention 的"聚焦"能力）。这可能是因为更多 unique data 让 model 学到"什么细节是 culturally/semantically diagnostic"的更鲁棒 prior。

---

## 11. 关键 Takeaways 与我的思考

### 11.1 Scaling 不是 monolithic 的

paper 最重要的 takeaway: **"scaling"这个笼统的概念需要拆开看**。10× data 对 saturated task 几乎无用，但对 long-tail task 是质变。所以决定要不要投资 100B scale，要先问"你的下游 task distribution 是 head 还是 tail?"

### 11.2 Power law 的任务依赖性

每个 task 有自己的 $\varepsilon$ 和 $c$。Western-centric 的 $\varepsilon$ 在 10B 附近被 approximate，cultural diversity 的 $\varepsilon$ 还需要更大数据才能 approximate。这意味着 scaling law 研究需要**多 benchmark 拟合**，不能只看 ImageNet 0-shot。

### 11.3 Data filtering 的 trade-off

CLIP filter 提升 Western-centric benchmark，**但放大 cultural bias**。这是个被低估的 risk: 当大家用 CLIP-score 过滤大规模 data 时，实际是在"过滤掉 inclusive 能力"。paper 的 limitation section 呼吁 community 开发 diversity-preserving filter，是个开放的 research direction。

可能的解决方向:
- **Multilingual CLIP**: 用 multilingual model 替代 English CLIP 做 filter
- **Diversity-aware score**: score function 同时考虑 alignment 与 coverage
- **Stratified filtering**: 按地理/语言分桶独立过滤
- **Adversarial debiasing**: 训练 filter 时 explicit 加入 anti-bias loss

### 11.4 对 Inclusive AI 的 broader implication

如果目标是 build 一个 "truly inclusive multimodal system"，paper 给出明确建议:
1. **用 raw data 而非 filtered data**，哪怕 trade off 一点 Western-centric benchmark
2. **加 language rebalancing**，explicit 上采样 low-resource
3. **Scale data 到 100B+**，特别是想覆盖 long-tail 概念

这跟一般"更多 data 总是更好"的 naive 直觉不同——paper 揭示了 scale 的"任务选择性"。

### 11.5 局限性

paper 自己指出的局限:
- Inclusivity benchmark 有限（Crossmodal-3600 只 36 语言，Dollar Street 只 63 国家）
- 只研究了 raw data scaling，没研究 100B scale 上的 quality filtering
- 只研究了 SigLIP contrastive 方法，其他方法（如 captioning、masked modeling）上的 scaling behavior 未知
- 没研究 100B 之上还有没有 saturation（也许 1T scale 又有新发现）

### 11.6 与 Karpathy 风格的 intuition

如果你 build 一个 VLM，这篇 paper 的直觉是:

1. **Compute budget 分配**: 如果只关心 ImageNet-style benchmark，10B + 多 epoch 就够；如果想 inclusive，必须 spread 到 100B unique data
2. **不要轻易 filter**: CLIP-filter 在 short term 提升 SOTA 数字，但 long-term 让 model 失去 inclusive 能力。这对 closed benchmark 上的"刷分"和真实世界"泛化"是个 trade-off
3. **多 epoch 是"假 scaling"**: 10 epochs × 10B ≠ 1 epoch × 100B，对 long-tail 任务尤其如此。unique data 才是真正的 scaling
4. **Explicit upsampling 比 implicit scaling 更经济**: 如果只是想提升 low-resource 语言，upsampling 比 scale up 10× data 更高效
5. **Power law 的 c 是关键诊断指标**: $|c|$ 越大说明越远离 saturation，可以放心 scale；$|c|$ 接近 0 说明 saturated，应该转向 quality 改善而非 quantity 增加

---

## 12. 总结

这篇 paper 用一个非常 clean 的实验设计回答了一个看似简单但被低估的问题: **100B unique data 带来的不是 ImageNet+1%，而是 Igloo recognition、Telugu retrieval、African household items identification 的从 0 到 1**。它把"scaling"这个口号细化成了"task-dependent saturation theory + inclusive capability"。

它没有 propose 新的 architecture 或 loss function，但提供了关键 empirical evidence 让 community 重新思考"data scale"到底意味着什么。对我个人而言，这篇 paper 改变了我对"CLIP filter"的默认信任，也让我重新看待"多 epoch"这件事在 long-tail 任务上的局限性。

参考链接汇总:
- Paper: https://arxiv.org/abs/2502.???(由于这是较新 paper，具体 arxiv ID 可能需要查阅)
- SigLIP: https://arxiv.org/abs/2303.15343
- PaliGemma: https://arxiv.org/abs/2407.07726
- WebLI / PaLI: https://arxiv.org/abs/2209.06794
- PaLI-3: https://arxiv.org/abs/2310.09199
- Dollar Street: https://datasets.neurips.cc/paper_files/paper/2022/hash/9bba9306c0b8b49ca9b1c4e9a0bb31ab-Datasets_and_Benchmarks-Datasets_and_Benchmarks_track.html
- GeoDE: https://arxiv.org/abs/2304.04124
- Crossmodal-3600: https://arxiv.org/abs/2205.12522
- DataComp: https://arxiv.org/abs/2304.14108
- Scaling Laws (Alabdulmohsin et al.): https://arxiv.org/abs/2305.16266
- Fairface: https://arxiv.org/abs/2108.04784
- Chinchilla: https://arxiv.org/abs/2203.15556
- Kaplan et al. scaling laws: https://arxiv.org/abs/2001.08361

如果你要我针对某个 section 深入再聊，比如 SigLIP loss 的 gradient 分析、power law fit 的具体 method、或者 language rebalancing 的更细数学，我可以展开。
