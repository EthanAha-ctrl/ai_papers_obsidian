---
source_pdf: GPIC A Giant Permissive Image Corpus for Visual Generation.pdf
paper_sha256: 38fb0c838a67409875850037aff87341937c3c510adfdf571e792bf1c83df6e5
processed_at: '2026-08-04T22:02:45-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GPIC 用人话讲

## 一句话总结

ImageNet-1K 当 benchmark 用了十几年，**FID 这个指标已经刷爆了**，刷到模型生成的图比"真实图片和真实图片之间的差异"还小。这说明大家不是在改善生成质量，是在**钻 metric 的空子**。GPIC 就是来换掉这个老 benchmark 的。

---

## 这个问题有多严重？

想象一个考试，满分 100 分。一开始大家考 60 分，慢慢爬到 80、90、95。但某一天你发现——**这套卷子本身有 bug，瞎蒙都能拿 98 分**。这时候还继续刷这套卷子有意义吗？

ImageNet-1K 上的 FID 就是这个情况。paper 里 Figure 9 是个很震撼的图：

- SiD2、REPA、MAR 这些模型，FID 比"**拿 5 万张真实图和训练集比**"的 FID 还低
- 也就是说，模型生成的图和训练集"更像"，比真实测试图和训练集还像

这在物理上不可能，除非模型在 memorize 训练集，或者 FID 这个指标本身就坏了。**两个原因都有**。

---

## 为什么 FID 坏了？

FID 的原理是：把图片塞进 Inception-v3 网络，拿到 2048 维 feature，然后算两个分布的"距离"（Fréchet Distance）。

问题在于 Inception-v3 是 2015 年在 ImageNet 上训练的，它的 feature space 只认 ImageNet 那 1000 个类别。你给它一张"柴犬在月球上"的图，它提取的 feature 和一张"柯基在草地上"可能差不多——因为它只关心"这是不是狗"，不关心月亮、草地、柴犬还是柯基。

所以 FID **对 semantic 细节瞎了**。模型只要把"狗的 texture"画对，FID 就很低，管你画的是柴犬还是柯基。

GPIC 的解法：换 feature extractor，用 **DINOv2**。DINOv2 是 self-supervised 训练的，feature space 对 semantic 细节敏感得多，和 human judgment 的相关性也高很多。这个新 metric 叫 **FD-DINOv2**。

---

## 但换 metric 不够，还得换 dataset

光换 metric 还有一个坑：**ImageNet 本身太小、太窄**。

- 只有 1000 个类，128 万张图
- 只有 class label，没有 text caption
- 但现在的模型（DALLE-3、Sora、FLUX）都是 **text-to-image** 的，用自由文本控制

在 ImageNet 上训练的模型，和真实世界的 text-to-image 模型，完全是两回事。你在 ImageNet class-conditional 上调出来的 hyperparameter，搬到 text-to-image 上根本不 work。paper 里 baseline 的 CFG=6.25 就是证据——ImageNet 上 CFG 1.5-2.5 最优，到了 GPIC text-conditional 得拉到 6.25 才好。

所以 GPIC 要做的不是"换个 metric"，是**从头建一个 dataset + benchmark**，让它和真实 text-to-image 的 practice 对齐。

---

## 建 dataset 的四个硬约束

paper 提了四个标准，听起来简单，同时满足极难：

### 1. Permissive（license 干净）
每张图必须能商用。ImageNet 很多图来自 Flickr non-commercial license，你拿它训练模型然后商业化，法律上是个雷。LAION-5B 更夸张，14 亿张图的 URL，license 五花八门，还有大量受版权保护的。

GPIC 只用 **CC BY、CC0、Public Domain** 这几类。代价是：可用的图少很多。Flickr 上 CC BY 的图就那么多，挖了十几年，这次又挖了 9700 万张，基本到顶了。

### 2. Stable（不能变）
LAION、DataComp 是 URL 列表。你今天下载，图还在；明年下载，30% 的链接 404 了；后年某个网站改版，又一堆链接失效。**同一个 dataset，不同时间下载，内容不一样**。那大家 paper 里报的数字怎么比？

GPIC 的解法：把所有图打包成 **8000 个 tar 文件，冻在 Hugging Face 上**。12.9 TB，一次性 release，以后不改了。你要么下不动（带宽问题），但只要下下来，内容就是固定的。

### 3. Large（规模够大）
100M 训练图 + 200K validation + 1M test。每张图配 text caption。规模和 LAION-5B 比小很多（5B 是 50 亿），但 LAION 没法 stable + permissive。100M 已经足够训 1B 参数级别的 text-to-image model 一个 epoch。

### 4. Accessible（好下载）
不用自己写 crawler，不用 img2dataset 去 14 亿个 URL 爬图。直接 HF 上下 tar shards，stream training。这对没大算力的 lab 特别重要。

---

## Caption 怎么做？

这部分是我觉得 paper 里最讲究的设计。

**不用 alt text**（网页里 image 的替代文字），因为 alt text 噪声极大——经常是 "image001"、"" （空）、"photo of my dog"。拿这种东西训练 text-to-image model，模型学到的"text-image alignment"是垃圾。

**用 VLM 重新 caption 每张图**。选了 Qwen3-VL-4B，理由很实用：
- 4B 比 30B 只差一点点质量（1.68 vs 1.73）
- 但 throughput 快 5 倍以上
- 100M 张图，4B 要 1500 H100 小时，30B 估计得 8000+ 小时

而且有个反直觉发现：**4B 在 counting 和 spatial understanding 上比 30B 还好**。因为 30B 是 sparse MoE，在 free-form caption 任务上 active experts 可能不如 dense 4B。所以大不是永远好。

Caption 分四种风格，比例 1/45/45/9：

| 类型 | 比例 | 例子 |
|---|---|---|
| Tag | 1% | `gorilla, rocky stream, green foliage, wet stones` |
| Short | 45% | "A bird perches on a bare tree branch against a cloudy sky." |
| Medium | 45% | "A mother duck with brown body and orange beak swims through green vegetation..." |
| Long | 9% | 一大段场景描述，包括颜色、位置、背景、光照 |

为什么是这个比例？因为**真实 user 输入 prompt 的分布就是这样的**。大多数人输入一句话（short），少数人写一大段（long），极少数人输入关键词（tag）。如果你 100% 用 long caption 训练，模型看到 short prompt 就会"脑补"一堆 user 没说的东西，偏离 user 意图。SDXL 之所以要 prompt upsampling，就是因为训练数据和 user 输入分布不匹配。

---

## Dedup 的讲究

去重这件事，说起来简单——找重复图片删掉。但实际操作里有个**两难**：

- 阈值太松（比如 SSCD similarity > 0.85 就删），会误杀很多**只是相似但不同的图**（同一场景不同角度、同一物体不同光照）
- 阈值太紧（比如 > 0.99 才删），漏掉大量**经过编辑的重复图**（加滤镜、改尺寸、调色温）

paper 里 Figure 5 给了个很直观的展示：similarity 在 0.95 到 0.9625 之间，居然还能看到明显不同的图（同一建筑不同时间拍）。

他们的解法分两步：

**第一步：先在小数据集上拟合 power law，预测大规模会删多少**

$$D(N) = A \cdot N^{\beta}$$

在 6 个小 subset（10 万到 340 万图）上跑 dedup，数每个 threshold 删多少图，拟合 $A$ 和 $\beta$。然后外推到 1.1 亿图。发现 threshold=0.95 时，预计删 962 万张，剩 1.01 亿。实际跑完剩 1.013 亿，**预测准得惊人**。

为什么用 power law？因为 duplicate 的增长不是线性的，类似 birthday paradox——数据越多，碰撞概率多项式增长。

**第二步：用两层规则，保守去重**

```
对所有 similarity > 0.90 的 pair 建图：
  规则1: 如果某 pair 的 similarity > 0.9625，删低分辨率的那个
  规则2: 如果某个 connected component 有 ≥5 张图，只留最高分辨率的
```

规则 1 抓**明确的 pair duplicate**（同一张图改了改）。规则 2 抓**连拍 / 相册 cluster**（同一个人上传了 10 张几乎一样的旅游照）。两层结合，既不误杀单独的相似图，又能抓到 cluster。

最后用 SHA-256 hash 验证没有 byte-level 完全相同的图——这是个 safety net，理论上前面步骤应该已经处理了。

---

## Evaluation 怎么做？

这是我觉得 paper 里**最该被其他 benchmark 学习**的部分。

### 三个关键设计

**1. Reference 是 held-out 1M test set，不是 train set**

ImageNet FID 的传统做法：拿生成的图和**训练集**比。问题：如果模型 memorize 了训练集，FID 就很低，但这不是真本事。

GPIC：拿生成的图和**100 万张 held-out test set**比。模型没见过这些图，要 FID 低必须**真的 generalize**。

**2. 提供 "oracle reference"**

拿 5 万张真实 test 图和 100 万张 test 图比，FD-DINOv2 = 7.44。这就是 **saturation floor**。任何模型用 5 万张生成图测，FD 不能低于 7.44，否则就是在 hack metric。

baseline JiT-T2I 的最好 FD 是 76.25。**离 floor 还差 10 倍**，说明这个 benchmark 远远没有刷到顶，还有大量改进空间。

**3. 明令禁止用 DINOv2 features 训练**

因为 DINOv2 训练数据可能和 GPIC test set overlap。如果你用 DINOv2 features 做 alignment loss（比如 REPA 那类方法），等于在"偷看答案"——你的 loss 直接优化 metric 用的 feature space。paper 明确说：**这么做算 non-standard GPIC result，必须披露**。

这一条直接 ban 掉了最近 representation alignment 这条 line of work 在 GPIC 上的"刷分"路径，逼大家回到纯 generative modeling 的比拼。

---

## Baseline 实验透露了什么

paper 训了个 JiT-T2I（1.1B 参数的 pixel-space flow matching model）作为 reference baseline。

几个有意思的点：

**1. 只训 1 个 epoch**

100M 张图，1 epoch，40 小时 8×H100。为什么不训更多？因为 100M × 256 tokens ≈ 25.6G tokens，对 1B 参数模型来说，chinchilla optimal 是 20B tokens，1 epoch 已经够了。多训会过拟合，而且 baseline 太强了后续工作不好比。

**2. CFG=6.25 最好，不是 ImageNet 的 1.5-2.5**

这个差异是 **text-conditional vs class-conditional 的本质区别**。class label 信息密度低（1000 类），CFG 不用很强。text caption 信息密度高，模型 capacity 有限，需要更强的 CFG 来"放大"condition 信号。SDXL / FLUX 的 CFG 5-8 也是这个原因。

**3. pixel-space flow matching**

不用 VAE，直接在 pixel space 训 flow matching。baseline 设计哲学是**最 minimal、最 reproducible**——没有 tokenizer pretraining，没有 auxiliary loss，没有 multi-stage。这样后续工作改进时，每个改动的影响都能清晰看到。

---

## 这篇 paper 真正的意义

表面看是 dataset paper，实际是**重新定义 visual generation 的游戏规则**：

1. **ImageNet-1K 时代的终结**：class-conditional benchmark 不再代表真实 practice，FID saturation 让 metric 失去区分度。GPIC 用 text caption + FD-DINOv2 + held-out test 重建可信 benchmark。

2. **Legal-clean dataset 的稀缺性**：CC BY 数据被挖了十几年，Flickr 接近枯竭。GPIC 把剩余可挖的 permissive data 一次性整理好、冻住、开源。**未来想做 commercial-open model 的 lab，这是目前唯一能大规模用的 clean dataset**。

3. **Synthetic captioning pipeline 成为 standard**：以后 academic lab 不会再抓 alt text 训模型，会抓 raw image + 自己用 VLM caption。GPIC 的 prompt 设计、microbenchmark、4 种 caption mixture 会被复用。

4. **对 representation alignment 的限制**：明确禁止用 DINOv2 features 训练，逼这条 line of work 重新思考 target。

---

## 我的担忧

1. **Flickr + Wikimedia 的地理 bias**：用户主体是北美欧洲，模型训出来会有 western bias。paper 没做 diversity audit。

2. **Caption 风格 bias**：Qwen3-VL-4B 的 caption 有自己风格，模型会继承。与真实 user caption 的 distribution 会不会 mismatch？

3. **跨 dataset leakage**：如果别的 lab 训 model 用的 dataset 也含 Flickr CC 图，和 GPIC test set 可能 overlap。paper 没给 overlap 估计。

4. **1 epoch baseline 太弱**：FD 76.25 vs floor 7.44，差 10 倍。不知道是 dataset 太难还是 baseline 太弱。后续需要 multi-epoch + larger model 的 scaling curve 实验。

5. **Hugging Face infra 压力**：12.9 TB，stream training 的带宽需求巨大。未来会不会限流？

---

## 最后

GPIC 不是终点，是**新十年的起跑线**。ImageNet-1K 从 2009 到 2026 用了 17 年，FID 从 200+ 刷到 1.5。GPIC 的 FD-DINOv2 现在 baseline 76，floor 7.44，还有 10 倍空间。**未来两三年的 image generation paper，大概率都会在 GPIC 上报数字**。

参考链接：
- Paper: https://gpic.stanford.edu
- Hugging Face dataset: 搜 GPIC
- FD-DINOv2 原始 paper: https://arxiv.org/abs/2310.00598
- JiT: https://arxiv.org/abs/2511.13720
- DINOv2: https://arxiv.org/abs/2304.07193

---

# GPIC: A Giant Permissive Image Corpus - 深度技术解析

## 1. 这篇 paper 真正要解决什么问题？

GPIC 的核心 motivation 不是 "再做一个大数据集"，而是**针对 ImageNet-1K 上的 "Goodharting" 现象**。论文中 Figure 9 是一个非常关键的可视化：在 ImageNet-256 上，多个方法（如 SiD2, REPA, MAR）的 FID 比 **50K held-out real images vs train set 的距离**还要低。这意味着 FID 已经被 hill-climbing 到了 metric 自身的 noise floor 以下，再优化 FID 实际上是在过拟合 evaluator（Inception-v3 features），而不是在改善生成质量。

这个问题有 mathematically 严格的一面：Fréchet Distance 的计算方式为

$$d^2 = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

其中：
- $\mu_r, \mu_g$：real / generated feature 分布的均值向量（在 Inception-v3 或 DINOv2 的 2048/768-d 特征空间里）
- $\Sigma_r, \Sigma_g$：两个分布的协方差矩阵
- $\text{Tr}(\cdot)$：矩阵 trace
- $(\Sigma_r \Sigma_g)^{1/2}$：矩阵乘积的对称平方根（通过 SVD 或 Schur 分解计算）

当用 **train set 统计** 作为 reference 时，如果模型 memorize 了一部分 train 数据，$\mu_g \to \mu_r$，$\Sigma_g \to \Sigma_r$，FD 就会人为地趋近于 0，但模型其实没有真正的 generalization 能力。GPIC 的核心 fix 之一就是用 **1M held-out test set** 作为 reference，让 memorization 不能骗过 metric。

参考链接：
- 关于 ImageNet FID saturation 的本质问题：https://arxiv.org/abs/2306.04675
- FD-DINOv2 原始 paper（Stein et al., NeurIPS 2023）：https://arxiv.org/abs/2310.00598

---

## 2. 四个标准的设计哲学

Table 1 是这篇 paper 的"立论"基础，值得逐条深思：

| 标准 | 含义 | 之前 dataset 失败的原因 |
|---|---|---|
| **Permissive** | 每张图 license 必须允许 commercial use，且 metadata 也是 permissive license | ImageNet 部分图片来自 Flickr non-commercial，YFCC100M 有 license 但分布长尾，DataComp 也有 license 限制 |
| **Stable** | 数据集内容不能随时间变化 | URL-based datasets (LaION, DataComp) 会 link rot，今天的 ImageNet "imagenet-256" 也存在不同 resize 预处理导致的隐性漂移 |
| **Large** | 规模够大、配丰富 caption | ImageNet-1K 只 1.28M 图、class label，OpenImages 9M 但 image-level label 弱 |
| **Accessible** | 直接 sharded 下载，不需要 crawler | YFCC100M / LaION 是 URL list，要 img2dataset 重新爬，墙内墙外差异巨大 |

注意一个关键直觉：这四个标准其实**互相对立**。
- Permissive 数据很难 large（Flickr CC BY 占比小）
- Stable 数据很难 accessible（一个 12.9TB 的镜像要让 user 都能下载，Hugging Face 给的免费带宽有限）
- Stable 的 dataset 不能迭代，但这与"前沿模型需要不断清洗新的 dirty web data"矛盾

GPIC 的解法是用 **frozen tar shards**（8000 shards，每个 ≈12,500 张）+ **Hugging Face 镜像** + **CC BY/CC0/Public Domain source pool**，牺牲"覆盖最 front-edge web image"换 permissive + stable。

参考：CC license 的 commercial use 边界讨论 https://creativecommons.org/about/cclicenses/

---

## 3. Construction Pipeline 的技术细节

### 3.1 Source pool 的统计直觉

最终 source pool：**110,569,761 张**，87.7% Flickr + 12.3% Wikimedia。注意 Flickr 在过去十几年其实是 academic dataset 的"金矿"（YFCC100M / Flickr30k / CC3M / CC12M 全来自这里），剩下可挖的"permissive Flickr 长尾"其实接近 exhausted 了。GPIC 这次又从 Flickr 抓 ≈97M，已经接近 Flickr CC 部分的物理上限。

这里有个**负反馈循环**的隐忧：当所有 academic lab 都从 Flickr CC 抓图训练，未来的 generative model 会越来越多地"输出 Flickr CC 风格的图"，然后这些图又被上传回 Flickr CC，下一轮 dataset 就会被 model-generated image 污染。GPIC 的 SSCD dedup 在某种程度上能捕捉这种 cycle（model generated image 与训练 image 高度相似），但只能 detect 1st-order 重复，对 styled 但 semantically different 的污染无效。

### 3.2 Filtering 的损失预算

- Aspect ratio / extreme resolution filter：去除 ≈0.01%（极小）
- Longest side < 256 px：与 SDXL/Flux 等模型 256 px 训练对齐
- **VLM-based quality filter** (Qwen3-VL-4B)：去除 ≈0.3%（near-blank, blur, over/underexposure）
- **Safety filter**：去除 ≈0.35%

合计只去除 ≈0.66% 的图。这个比例很保守，原因是 permissive 图太贵，不能激进过滤。直觉上，如果你在 LAION-5B 上做同样的 filter，loss rate 应该到 10-20%（因为 LAION 充满 meme / screenshot / ad banner）。

### 3.3 SSCD-based Deduplication 的核心技术

这部分是 paper 中最有 math 内容的地方。SSCD (Self-Supervised Copy Detection) 是 Meta 在 CVPR 2022 提出，专门为 copy detection 训练的 self-supervised feature。直觉上 SSCD 的设计目标是**对 copy-paste / edit / crop / color shift 不变，对 semantic identity 敏感**：

$$f_{\text{SSCD}}(x) \in \mathbb{R}^{512}, \quad \cos(f(x_i), f(x_j)) \in [0, 1]$$

- 越接近 1 表示越相似
- 完全不同的 image cos similarity ≈ 0.3-0.5
- Semantically related but distinct ≈ 0.6-0.85
- Near-duplicate ≈ 0.85-0.95
- Strict duplicate ≈ 0.95+

Paper 的 Figure 5 给出了**6 个 similarity bin 的 qualitative sample**，值得仔细看：在 [0.85, 0.9) 已经能看到同 object 不同 pose，[0.95, 0.9625) 仍可能有 visible difference（不同曝光、不同光线下同一建筑）。

**Power-law 估计** 是这里的关键技术：

$$D(N) = A \cdot N^{\beta}$$

- $D(N)$：当数据集大小为 $N$ 时，threshold $\theta$ 下被移除的图像数
- $A, \beta$：通过 6 个 subset ($N_i \in [108\text{K}, 3.4\text{M}]$) 拟合得到的参数
- $N$：subset 大小

为什么要用 power law 而不是 log 或 linear？因为 **collision 概率随 $N$ 多项式增长**，类似 birthday paradox 的 $O(N^2)$ 增长模式，但因为有大量真正不同的 image，碰撞数 sub-quadratic 增长更合理。$\beta < 1$ 表示 duplicate 集中度比 linear 还低（如果 duplicate 是均匀分散的，$\beta \approx 1$；如果集中在某个 cluster，$\beta < 1$）。

在 $\theta = 0.95$ 下外推到 $N = 110\text{M}$，估计 $D = 9.62 \times 10^6$（约 8.7% removal）。这是个**预测**，实际执行后剩下 101.3M，与预测的 100.9M 非常吻合，说明 power-law extrapolation 在这个 scale 上 well-behaved。

**Two-tier removal rule** 是个很聪明的 design：

```
For all pairs with cos > 0.90:
    if cos(i,j) > 0.9625:
        remove lower-resolution image
    else:
        do nothing yet

For all connected components with >=5 images (using threshold 0.90):
    keep only highest-resolution image
```

直觉：
- Rule 1 是 high-confidence dedup（target pairwise duplicate）
- Rule 2 是 cluster-level dedup（target burst photography 的连拍 / album 上传）
- 两层结合避免：单张图因 cos similarity 偶然高而被错杀，又能抓到 5+ 连拍的明显 cluster
- 最终用 SHA-256 验证无 byte-level exact duplicate（safety net）

参考：
- SSCD paper: https://arxiv.org/abs/2111.06249
- Power law in dataset deduplication discussion: https://arxiv.org/abs/2304.14108

---

## 4. Captioning Pipeline 的工程细节

### 4.1 Caption 类型分布的设计直觉

| 类型 | 比例 | 描述风格 | 训练直觉 |
|---|---|---|---|
| Tag | 1% | unordered keywords: `gorilla, rocky stream, green foliage` | Tag-like caption 用来诱导模型学习 keyword-conditioned generation（用于 inference 时 user 输入短 tag 的鲁棒性） |
| Short | 45% | 一句简单描述 | 多数 inference 场景的真实输入 |
| Medium | 45% | 2-3 句详细描述 | 与 DALLE-3 / Sora / FLUX prompt 风格对齐 |
| Long | 9% | 长段 scene description | 教模型对 spatial / detail binding 的长 prompt 鲁棒 |

这个 1/45/45/9 的 mixture 实际上**反映真实 user 行为分布**。如果你 100% 都用 long caption 训练，模型在 short prompt 下会"过度发挥"，偏离 user 意图（SDXL / DALLE-3 都有这个问题，所以现在 SDXL 仍要做 prompt upsampling）。

### 4.2 Qwen3-VL-4B 选择背后的 microbenchmark

Microbenchmark 设计：
- 1,520 张图：720 short + 640 medium + 160 long caption
- 5 个 axis 评分：overall summary, counting accuracy, spatial understanding, attribute binding, OCR
- 0-2 分制，LLM-as-a-judge

为什么 5 个 axis 这些维度？
- **Counting**：VLM 经常错算 object 数（"five" vs "seven" volleyball players）
- **Spatial understanding**：能区分 first-base line vs third-base line（这决定 caption 能否用于训练 layout-aware generation）
- **Attribute binding**：颜色 / 物体对应关系（red car vs car red 这种常见的 VLM 弱项）
- **OCR**：能否正确读 sign / text（DALLE-3 等模型对 OCR 极度依赖 caption 质量）

Qwen3-VL-4B vs 30B-A3B 的 trade-off：
- Quality：1.68 vs 1.73（只差 0.05）
- Spatial：1.60 vs 1.55（4B 反而更好！）
- Attribute：1.55 vs 1.50（4B 更好）
- Throughput（short）：56.10 vs （30B 慢很多）
- Total cost：1,500 H100-hours for 100M images

**关键直觉**：4B model 在 detail accuracy 上反而强于 30B，这是因为 30B-A3B 是 sparse MoE，在 caption 这种 free-form task 上 active experts 可能比 dense 4B 弱。这也是为什么 Qwen3-VL-4B 是开源 captioning 的 sweet spot。

参考：
- Qwen3-VL technical report: https://arxiv.org/abs/2511.21631
- vLLM (used for serving): https://arxiv.org/abs/2309.06180

---

## 5. Benchmarking Protocol 的核心问题

### 5.1 为什么用 FD-DINOv2 而不是 FID？

FID 用 Inception-v3 features 的问题：
1. **Inception-v3 trained on ImageNet-1K**，feature space 高度 class-biased，对 ImageNet-1K class 外的 image 区分度低
2. ImageNet-1K FID 已饱和（Figure 9 left）
3. Inception features 对 texture / color 敏感，对 semantics 不敏感

DINOv2 features 是 self-supervised 训练，更接近 human perception：
- DINOv2 ViT-L/14 输出 1024-d feature
- 在 multi-crop, scene-level, object-centric image 上都 well-behaved
- 与 human ranking 相关性更高（per Stein et al.）

但这里有个**关键的合规约束**（paper 专门强调了）：

> **DINOv2 was trained on LVD-142M dataset, 部分图可能与 GPIC test set overlap**。所以任何 method 不能在 DINOv2 features 上 train（如 REPA / REPaE 这类 representation alignment loss），否则就是 metric hacking。

这一条直接 ban 掉了最近 representation alignment 的整条 line of work（如 REPA: https://arxiv.org/abs/2410.06938），它们都依赖 DINOv2 features 作为 target。

### 5.2 Oracle Reference 与 saturation monitoring

Table 2 给出 real-vs-real 距离：

| Subset | FD | Precision | Recall | Density | Coverage |
|---|---|---|---|---|---|
| Full vs Test-1M | 1.19 | 0.947 | 0.950 | 1.000 | 0.972 |
| Lite vs Test-1M | 1.25 | 0.951 | 0.947 | 1.010 | 0.973 |
| Nano vs Test-1M | 1.60 | 0.946 | 0.946 | 1.002 | 0.968 |
| Val vs Test-1M | 2.37 | 0.948 | 0.949 | 0.993 | 0.966 |
| **Test-50K vs Test-1M** | **7.44** | 0.949 | 0.953 | 0.997 | 0.967 |

**这个 7.44 就是 saturation floor**。任何 method 用 50K generated image 测试，FD 不能低于 7.44，否则就是 metric hacking。对比之下 JiT-T2I baseline 的 best FD 是 76.25，离 saturation floor 还差 10×，说明这个 benchmark 还有大量 hill-climbing 空间。

直觉：当真实数据集非常大（1M test）时，sample 50K 子集本身就有 sampling variance，这个 variance 设了 metric 的 noise floor。FD 不是 sample-size invariant 的，越小的 subset 距离越大。

### 5.3 FDμ 和 FDΣ 的分解

Table B.2 提供了一个非常细致的分解：

$$d^2 = \underbrace{\|\mu_r - \mu_g\|^2}_{\text{FD}_\mu} + \underbrace{\text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)}_{\text{FD}_\Sigma}$$

- FD_μ：分布均值的 squared distance
- FD_Σ：协方差的 Fréchet distance

对 Test-50K vs Test-1M：DINOv2 上 FD_μ = 0.040，FD_Σ = 7.404。**几乎所有 FD 都来自 variance mismatch，不是 mean mismatch**。

这个直觉很重要：随着 subset 变小，mean 估计仍然准确（小样本 mean 还是收敛的），但 covariance 估计噪声急剧增长，所以小 subset 的 FD 主要由 covariance mismatch 主导。

这也意味着未来优化 GPIC benchmark，应该关注 **diversity 而不是 fidelity**（这与 paper 主张用 Recall / Coverage 而不只看 FD 一致）。

参考：
- Fréchet Distance 原始定义：https://arxiv.org/abs/1706.08500
- Precision/Recall metric (Kynkäänniemi et al.): https://arxiv.org/abs/1904.06991

---

## 6. JiT-T2I Baseline 实验细节

### 6.1 为什么选 JiT 作为 reference baseline？

JiT (Just a Transformer) 是 Kaiming He / Tianhong Li 在 2025 年提出的 **pixel-space flow matching** model。paper [33] 是 https://arxiv.org/abs/2511.13720。设计哲学是：

- **Single-stage training**：不预训 VAE / VQ-VAE
- **No tokenizer pretraining**：直接在 pixel space 做 flow matching
- **No auxiliary losses**：不用 perceptual loss / GAN loss / LPIPS loss（区别于 PixelGen [34]）

直觉：这种 baseline 是为了提供一个**最 minimal、最 reproducible 的 benchmark 起点**。如果用 SDXL / FLUX 这种复杂 baseline，涉及 VAE pretraining + multi-stage training + classifier-free guidance tuning + DDIM solver selection 等多个 confounder，就失去了"benchmark 一个 dataset"的干净性。

### 6.2 训练超参

- Architecture：PixGen-XXL/16 1.1B（**patch size 16**，意味着每个 16×16 patch 是一个 token；256×256 image = 16×16 = 256 tokens）
- Text encoder：Qwen3-1.7B
- Resolution：256×256
- Global batch size：256
- Optimizer：AdamW(lr=1e-4, β1=0.9, β2=0.95, weight_decay=0)
- Schedule：constant LR with 0.1% warmup（不用 cosine decay 是为了 single-epoch 实验简单）
- Data augmentation：random crop scale ∈ [0.8, 1.0]，再 resize 到 256×256
- Max text length：300 tokens
- Total：**1 epoch, 40 hours on 8×H100**

直觉：1 epoch 而不是 multi-epoch 是有意为之。100M 张图 × 256 tokens ≈ 25.6G tokens per epoch，已经远超 1B 参数的 chinchilla optimal（≈20B tokens for 1B params），所以 1 epoch 已经足够收敛；multi-epoch 反而会过拟合。但 GPIC 是 large + diverse dataset，1 epoch 也是为了避免 baseline 太强，留给后续工作改进空间。

### 6.3 Flow matching 的核心公式

虽然 paper 没直接给公式，但 JiT 用的是 **rectified flow / flow matching**（Lipman et al., ICLR 2023）：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1, \epsilon}\left[\|v_\theta(x_t, t, c) - (x_1 - x_0)\|^2\right]$$

其中：
- $t \sim U[0, 1]$：时间步
- $x_0$：noise，通常 $\sim \mathcal{N}(0, I)$
- $x_1$：real image (在 pixel space)
- $x_t = (1-t) x_0 + t x_1$：linear interpolation
- $v_\theta$：神经网络预测的 velocity
- $c$：text condition

Inference 时用 Euler ODE solver：

$$x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c)$$

Paper 用 50 步 Euler sampling，对应 $\Delta t = 1/50$。50 步是 flow matching 的典型 setting，比 DDPM 的 1000 步少很多（rectified flow 的优势）。

### 6.4 CFG scale 的反常发现

| CFG | FD | Precision | Recall | Density | Coverage |
|---|---|---|---|---|---|
| 1.75 | 204.01 | 0.917 | 0.530 | 1.034 | 0.806 |
| 4.00 | 87.80 | 0.933 | 0.765 | 1.012 | 0.906 |
| 6.25 | **76.25** | 0.942 | 0.792 | 1.014 | 0.908 |

直觉：在 ImageNet class-conditional benchmark，通常 CFG 1.5-2.5 是最优（再高就 mode collapse）。在 GPIC 上，最优 CFG 6.25，远高于 ImageNet。这是因为：
- **Class-conditional**：condition 信息少（仅 1000 类），CFG 不需要很强
- **Text-conditional**：condition 信息密度高，但模型 capacity 有限，需要更强的 CFG 来"放大"condition 信号
- 这与 SDXL / FLUX 的经验一致（CFG 5-8 是常见配置）

参考：
- JiT paper: https://arxiv.org/abs/2511.13720
- PixelGen paper: https://arxiv.org/abs/2602.02493
- Flow matching 原始 paper: https://arxiv.org/abs/2210.02747

---

## 7. Appendix 中的关键细节

### 7.1 DINOv2 backbone size & registers 对 FD 的影响

Paper B.3 section 测试了 8 种 DINOv2 配置：
- ViT-S/14, ViT-B/14, ViT-L/14, ViT-g/14
- ± register tokens

发现：
- With registers 比 without registers 在 S/B/L 上 FD 显著更低（feature distribution 更窄）
- ViT-g 上 with/without registers 几乎一样
- 8 种配置的 pairwise Pearson = 0.847，Kendall's W = 0.795（强相关）

**默认选择**：ViT-L/14 **without registers**，与 Stein et al. 一致。

直觉：register tokens 是为了减少 DINOv2 feature map 上的 high-norm artifacts（Darcet et al. 发现 ViT 会在 low-information patches 上"垃圾 dump"）。register tokens 把这些 artifact 收容到 register，所以 feature value 分布更窄，自然 FD 也更低。但因为 with/without registers 在 ranking 上高度一致，用哪个其实不影响 relative comparison。

### 7.2 Resize 的影响

Paper B.1 说明了 Imagenet-256 / GPIC-256 的 resize protocol：
1. Center crop along longer edge → square
2. Bicubic downsampling to 256×256（Pillow 实现）

警告：**OpenCV / PIL / torchvision 的 bicubic kernel 不同**（PIL 是 Mitchell-Netravali, torchvision 是 Catmull-Rom），会导致 1-2 点 FID 差异。这是 ImageNet-256 benchmark 长期以来的隐性噪声来源之一。Paper 选用 Pillow 与 Stein et al. 一致，避免后续 reproduction 出错。

参考：
- DINOv2 register paper: https://arxiv.org/abs/2309.12988
- Parmar et al. on aliased resizing: https://arxiv.org/abs/2104.11222

---

## 8. 我的整体直觉评估

### 8.1 GPIC 的真正价值

GPIC 最大的贡献不是 28T pixels 这个数字（其实 ImageNet-1K 全集 ≈ 1.5T pixels，100M 图 256×256 也只 ≈ 6.7G pixels），而是 **为 text-to-image generation 提供 legal-clean + frozen + reproducible 的 benchmark**。这对 academic research 是关键基础设施：

- DALLE-3 / Sora / FLUX 这些 frontier 模型的 caption quality / diversity 在 DPO / RLHF 之后的不断提升，academic 模型如果只在 ImageNet class label 上 train 永远追不上
- 没有 legal-clean dataset，academic lab 想做 commercial-usable open model 也不行（这也是为什么 SDXL 是 LAION 但商业 release 时被 SC 控诉）
- 没有 frozen test set，每个 paper 都自己挑 50K test caption，metric 不可比

### 8.2 GPIC 的潜在弱点

我读 paper 后觉得以下几方面需要警惕：

1. **Source bias**：Flickr + Wikimedia 的 image 分布**严重偏向西方**（Flickr 用户主体北美欧洲，Wikimedia 类似）。如果未来 GPIC-trained model 出现 geographic bias，这是 source-level bias 而不是 model-level bias。paper 没有显式做 geographic / cultural diversity audit。

2. **Caption 风格 bias**：Qwen3-VL-4B 生成的 caption 有其自身风格（每张图都强调某些 attribute 类型，如颜色、位置）。如果未来 model 在 GPIC 上训出来，会继承这个"Qwen3-VL caption 风格"，与真实 user caption 的 distribution mismatch。

3. **Deduplication 的 trade-off**：保守 dedup 留下更多图但保留了 near-duplicate。Paper 在 Limitations 承认 "some near-duplicates may remain"。这对 generative model evaluation 不一定是问题，但对 memorization 评估可能是隐患。

4. **Test set 的 leakage risk**：因为 source 是 Flickr / Wikimedia，如果其他 lab 训 model 用的 dataset 也包含这些 source，会有跨数据集 leakage。Paper 没有显式声明 GPIC 与 LAION / DataComp / CC12M 等常见 dataset 的 overlap 估计。

5. **Single epoch baseline 可能不足以体现 dataset 价值**：1 epoch JiT-T2I 的 best FD=76.25，离 7.44 oracle floor 还差 10×，说明 baseline 远未 saturate。但 paper 也没给 multi-epoch 的对照，无法判断这是 dataset 复杂度高还是 baseline 太弱。后续工作（特别是 1B+ params）应该在 GPIC 上做 multi-epoch，看 scaling curve。

### 8.3 对学术研究的影响预测

GPIC 可能在未来 1-2 年产生以下连锁反应：

1. **REPA / REPaE 这类 representation alignment 方法**在 GPIC 上不允许直接用 DINOv2 features 做 alignment target，会被迫切换到 ImageNet features / self-supervised features trained on GPIC train set。这其实是个公平挑战——是 dataset shift 还是 metric shift 决定的方法 design。

2. **Pixel-space generative model 复兴**：JiT / PixelGen / Simpler Diffusion (SiD2) 都在挑战 latent-space diffusion。GPIC 提供了一个 large + clean 数据集，pixel-space model 在大数据集上能更 fair 地对比 latent-space model，看哪个 scaling law 更好。

3. **Synthetic captioning pipeline 成为新 standard**：以后 academic lab 不再会去 LAION 抓 alt-text（noise + privacy risk），而是会抓 raw image + 自己用 Qwen3-VL 重新 caption。GPIC 的 captioning prompt + microbenchmark 设计会被复用。

4. **Hugging Face 镜像 infra 的承压**：12.9TB dataset 跑流式训练，对 HF bandwidth 是很大压力。未来如果更多 lab 用 GPIC + stream training，可能出现 IP-based rate limit / quota。值得 watch。

参考：
- GPIC 主页: https://gpic.stanford.edu
- Hugging Face dataset (具体链接 paper 未给出，应在 huggingface.co/datasets 搜 GPIC)
- JiT code: 可在 GitHub 找
- REPA: https://arxiv.org/abs/2410.06938
- Simpler Diffusion (SiD2): https://arxiv.org/abs/2410.19324
- DINOv3: https://arxiv.org/abs/2508.10104
- SigLIP: https://arxiv.org/abs/2303.15343

---

## 9. 总结：构建直觉的核心几点

如果只记 5 件事：

1. **ImageNet FID 已 saturate 到"模型 FID < real-vs-real FID"的反直觉程度**，GPIC 用 FD-DINOv2 + 1M held-out test set 重新建立可信赖的 evaluation。
2. **Permissive + Stable + Large + Accessible** 是个工程上的 4-way trade-off，GPIC 的解法是 Flickr CC BY + frozen tar shards + HF 镜像。
3. **Deduplication 的 power-law extrapolation** $D(N) = AN^\beta$ 是个 elegant 的方法，让 small-subset 实验预测 large-scale removal。
4. **Caption mixture 1/45/45/9 (tag/short/medium/long)** 设计反映 user 行为分布，不是平均分配。
5. **Baseline 的 CFG=6.25 比 ImageNet 的 CFG=2 高很多**，是 text-conditional vs class-conditional benchmark 的根本差异。

这篇 paper 表面上是 dataset paper，实际上是为整个 visual generation community 重新设定**评价规则**——把 ImageNet-1K 的 decaying benchmark 替换成 GPIC 的 scalable benchmark。后续 1-2 年的 image / video generation paper 可能都会在 GPIC 上报告数字。
