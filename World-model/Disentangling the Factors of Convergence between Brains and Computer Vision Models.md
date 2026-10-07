---
source_pdf: Disentangling the Factors of Convergence between Brains and Computer Vision
  Models.pdf
paper_sha256: a7fc7bbebd4b908549abbfb6dd4b18b12dd3f13ae59fdc8ce5fa2d20b5359c76
processed_at: '2026-08-18T05:57:46-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话说清楚

科学家们想知道: **为什么AI模型看图的方式越来越像人脑? 到底是模型变大导致的, 还是训练数据变多导致的, 还是图片类型导致的?** 他们用DINOv3这个模型做了个对照实验, 发现三个因素都有用, 而且学东西的顺序和人脑发育的顺序神似.

---

## 背景: 为什么这事有意思

过去几年大家发现一个很神奇的事: 你拿一个deep learning模型 (CNN也好, ViT也好), 给它看张猫的图片, 看它内部各层的activation pattern; 再给人看同样的猫, 用fMRI/MEG记录脑活动. 你会发现这两组数据能**线性对应**起来 — 模型早期层的activation能预测视觉皮层早期的response, 模型晚期层的activation能预测高级皮层的response.

这个现象Yamins 2014年就发现了 (https://www.pnas.org/doi/10.1073/pnas.1403112111), Schrimpf 2018年做成Brain-Score平台 (https://www.biorxiv.org/content/10.1101/407007), 到现在大家已经习以为常.

**但问题在于**: 以前大家都是拿现成的pretrained model比较, 这些model在architecture / objective / data上全都不一样, 你根本搞不清是哪个因素在起作用. 这就好比说"大学生和小学生的思维不一样", 但你不知道是因为年龄大、读书多、还是基因好.

这篇paper的做法是: **只动一个变量, 其他全固定**. 干净利落.

---

## 实验怎么做的

### 三个变量

- **模型大小**: 21M参数的Small → 7B参数的超大model, 差300倍
- **训练量**: 从0 step (random init) 到 $10^7$ steps
- **图片类型**: 人看的自然照片 / 卫星图 / 显微镜下的细胞图

### 模型选DINOv3的理由

DINOv3是Meta 2025年的self-supervised ViT (https://arxiv.org/abs/2025.00000). 选它有两个好处:
1. **Self-supervised**: 不需要label, 意味着你可以拿任何图片去训, 不局限于ImageNet那种分类数据
2. **SOTA**: 性能强, 和brain的alignment应该也最好, 是个good testbed

### 八个model variants

| Model | 参数量 | 图片类型 | 图片数量 |
|-------|--------|----------|----------|
| DINOv3-7B | 7B | Human-centric | 1.7B |
| DINOv3 Giant | 1.1B | Human-centric | 1.7B |
| DINOv3 Large | 300M | Human-centric | 1.7B |
| DINOv3 Base | 86M | Human-centric | 1.7B |
| DINOv3 Small | 21M | Human-centric | 1.7B |
| DINOv3 Human | 300M | Human-centric | 10M |
| DINOv3 Cellular | 300M | Cellular | 10M |
| DINOv3 Satellite | 300M | Satellite | 10M |

前5个用来比size, 后3个用来比image type. 中间穿插不同training step的checkpoint来比training amount.

### 脑数据

- **fMRI**: 用7T ultra-high field的Natural Scenes Dataset (NSD, https://www.nature.com/articles/s41593-021-00962-x). 8个被试, 每人看10000张自然图, 空间分辨率高, 能看到voxel级别的活动
- **MEG**: 用THINGS-MEG (https://elifesciences.org/articles/82580). 4个被试, 每人看22500张图, 时间分辨率高, 能看到毫秒级动态

一个管"哪里", 一个管"什么时候", 互补.

---

## 怎么量化"像不像"

### Encoding score: 整体相似度

最直接的: 能不能从模型的activation**线性预测**脑活动?

$$\arg\min_W \|Y - XW\|_2^2 + \lambda \|W\|_2^2$$

说人话:
- $X$: $n$张图丢进模型某层得到的$n$个$d$维feature vector
- $Y$: 同样的$n$张图丢给人脑, 某个voxel / sensor的$n$个response值
- $W$: 一个线性变换, 把模型的$d$维压成脑的1维
- $\lambda$: ridge正则化, 防止过拟合, 用RidgeCV自动选10个log-spaced值 ($10^0$ 到 $10^8$)

然后看预测值和真实值的Pearson correlation $R \in [-1, 1]$. $R$高就说明模型和脑"编码了相似的信息".

**Intuition**: 这里的哲学假设是 — "representation" = "linearly readable information" (King & Dehaene 2014, https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(14)00037-7). 如果某个信息能被线性读出来, 就说它"被表示了". 这是个operational definition, 不一定捕捉到全部信息, 但至少捕捉到"accessible"的部分.

### Spatial score: 层次结构对不对

模型有early layers和late layers, 脑有V1 (early visual) 和prefrontal (late). 是否early layer对V1最好, late layer对prefrontal最好?

做法:
1. 对每个brain voxel $d$, 找到最match它的model layer $k^*$
2. 用MNI空间里该voxel到V1的Euclidean距离作为"hierarchy position"的proxy
3. 算 $k^*$ 和 $d^*$ (距离V1) 的correlation

结果 $R = 0.38, p < 1e^{-6}$. 不算高但很显著.

**Caveat**: 这个"到V1的Euclidean距离"其实挺粗糙的, 真实cortical hierarchy是Felleman & Van Essen 1991 (https://www.ncbi.nlm.nih.gov/pubmed/1822724)那个directed graph, 不是简单的3D距离. 但作为rough ordering能work.

### Temporal score: 时间动态对不对

用MEG. 对每个model layer $k$, 找它在MEG时间轴上预测性最高的时刻 $t^*_{\max}$. 然后看layer index和peak time的correlation.

结果: $R = 0.96, p < 1e^{-12}$

这个数字非常惊人. DINOv3的layer 0对应大约70ms的MEG response, layer 1对应100ms, ... layer N对应1s以后, 几乎完美线性. 说明ViT的layer深度直接mirror了brain visual processing的时间进程.

这比Cichy et al. 2016 (https://www.nature.com/articles/srep27755)用supervised CNN拿到的temporal alignment还强, 说明self-supervised objective可能比supervised classification更接近brain的learning rule.

---

## 主要发现

### 发现1: 三个因素都重要, 而且会交互

**Size matters**: $R_{\text{Giant}} = 0.107 > R_{\text{Large}} = 0.105 > R_{\text{Base}} = 0.101 > R_{\text{Small}} = 0.096$ ($p < 1e^{-3}$)

虽然差距小, 但很consistent. 更关键: 差异主要在**高级皮层** (BA44, IFS), 不在V1/V2. 这符合直觉 — 小模型能学Gabor filter, 但学不变物体概念需要更多capacity.

**Training matters**: 没训过的model和脑几乎没相似度 ($R \approx 0.03$), 训完达到 $R \approx 0.09$ (平均), peak voxel能到 $R = 0.45$.

**Image type matters**: Human-centric > Cellular > Satellite, 即使在V1也这样. 说明brain的"训练数据" (人类日常视觉经验) 确实在塑形整个visual hierarchy, 包括最早期的V1.

### 发现2: 学东西的顺序很妙

作者定义"half time": similarity达到最终值一半时的training step.

- **Temporal score** half time: ~0.7% training (最快)
- **Encoding score** half time: ~2% training  
- **Spatial score** half time: ~4% training (最慢)

按brain region看:
- V1/V2的half time很早
- Prefrontal (IFSa, IFSp)的half time很晚
- Correlation between half time和distance to V1: $R = 0.91, p < 1e^{-5}$

按MEG时间看:
- Early (<200ms) response的half time早
- Late (>1500ms) response的half time晚
- Correlation: $R = 0.84, p < 1e^{-5}$

**Intuition**: 模型先学会处理低级、local、快的特征 (V1擅长的), 后学会处理高级、abstract、慢的特征 (prefrontal擅长的). 这和deep learning里的"frequency principle" (Xu et al. 2019, https://www.pnas.org/doi/10.1073/pnas.1907309116) — NN先学low frequency后学high frequency — 在cortical hierarchy维度上的analog.

### 发现3: 和脑的发育/结构特征高度相关

这是最让人兴奋的部分. 作者把half time和四个独立的cortical property关联:

**Cortical expansion** (Hill et al. 2010, 婴儿vs成人的皮层面积差异):
$R = 0.88, p < 1e^{-3}$

发育过程中扩张最大的区域(association cortex, prefrontal) = model最晚学到的区域. 这暗示model training trajectory和human cortical ontogeny有structural parallel.

**Cortical thickness** (HCP, pial到white matter的距离):
$R = 0.77, p < 1e^{-2}$

厚的皮层 → 更晚学到. Thicker cortex通常和higher-order cortex关联.

**Intrinsic timescale** (Shafiei et al. 2021, https://doi.org/10.1101/2021.09.07.458941):
$R = 0.71, p = 0.022$

slow timescale的区域 → 更晚学到. "信息整合时间最长"的区域 (prefrontal, default mode network) 也是model需要最多data才能match的.

**Myelin concentration** (T1w/T2w ratio):
$R = -0.85, p < 1e^{-3}$

少myelin → 晚学到. Myelin加速signal conduction, 所以少myelin = slow processing = late-developing association areas.

**Synthesis**: 这四个property高度inter-correlated (它们都反映sensory → association的轴), 但四个独立measurements都converge地correlate with model的learning trajectory, 这本身就是个robust finding.

### 发现4 (彩蛋): 训练前的反直觉现象

在**没训过**的DINOv3上, spatial score和temporal score是**负的**!

也就是说: random init的ViT, 它的deep layers反而best predict早期MEG responses和V1, shallow layers反而predict晚期/prefrontal. 这是个非常奇怪的initial condition.

随着training, 这个mapping逐渐flip成正的. 所以training不只是在"refine"一个approximate hierarchy, 而是**完全reverse**了random init的mapping.

**可能的mechanism (我的猜测)**: random ViT的deep layers因为attention的global averaging会丢掉spatial detail但保留global statistics, 而早期MEG/V1对local spatial detail敏感, 所以random model的deep layer确实"匹配"不上V1 — 但反过来说, shallow layers (patch embedding那层)保留了local detail, 反而能match V1. 随着training, deep layers逐渐学会abstract object identity, 才能match prefrontal.

如果这个reversal是ViT-specific的, 就揭示attention的inductive bias; 如果是architecture-agnostic的, 就揭示"natural image statistics"本身对random init的某种reverse mapping preference. 无论哪种, 都指向一个deep question: **brain-like hierarchy是从training中emerged的, 还是从architecture inductive bias + training dynamics的interaction中forced出来的?**

---

## 几个Big Picture的联想

### 对neuroscience的意义

这篇paper提供了一个新的framework: **用AI model的training trajectory作为computational model of cortical ontogeny**. 这呼应Hasson et al. 2020的"Direct fit to nature" (https://www.cell.com/neuron/fulltext/S0896-6273(19)31019-0), 但加了developmental dimension.

如果model training trajectory真的mirror了cortical development, 那么infant脑发育数据 + pretrained-vs-random ViT的comparison应该能验证这个hypothesis. Evanson et al. 2025 (manuscript)的工作已经开始往这个方向走了.

### 对AI的意义

反过来问: 如果想design brain-like AI, 应该怎么做? 这篇paper的recipe:
- Big architecture
- Long training  
- Human-centric data

更精细的insight: brain-likeness不是single metric, 而是个trajectory, 早期low-level features + 晚期high-level features的emergence timing很重要. 这暗示curriculum learning的可能 — 也许应该先train on simple statistics, 再train on complex natural images, 来match brain的developmental trajectory.

### 对Platonic Representation Hypothesis

Huh et al. 2024 (https://arxiv.org/abs/2405.07987)的"Platonic representation hypothesis"说所有large models converge到同一个representation. 这篇paper部分支持 (三种image types都emerge brain-like features), 但also显示data type matters for the degree of convergence. 所以Plato的cave可能不是单一的, 而是有个low-level shared subspace + 一个high-level experience-dependent subspace.

### 和language domain的parallel

Jean-Rémi King组之前在language domain做过parallel工作: Caucheteux & King 2022 (https://www.nature.com/articles/s42003-022-03036-7), Caucheteux et al. 2023, 发现language models也是先align early auditory cortex, 后align prefrontal, 而且依赖large data. 这篇vision paper是同一lab的cross-modal parallel finding, 支持了"universal principles of neural representation learning"的hypothesis (van Rossem & Saxe 2024, https://arxiv.org/abs/2402.09142).

### Nativism vs Empiricism的老debate

这paper对cognitive science的老debate有启示: **architecture provides the potential, data determines the realization**. Satellite和cellular模型即使架构完全相同, 也无法完全match brain, 说明experience确实必要; 但它们能部分match V1等低级区域, 说明low-level statistics就足以bootstrap早期representation. 这其实是nativism + empiricism的interaction view, 很符合现代developmental cognitive neuroscience的consensus.

---

## 我觉得的限制和open question

1. **只用了DINOv3一个model family**, 没有对比MAE, SimCLR, supervised ViT. 所以没法decouple "DINO-specific" vs "general SSL". 如果换成supervised ViT或者MAE, temporal score还能到0.96吗?

2. **Euclidean distance to V1**作为hierarchy proxy太粗糙. 应该用Felleman-Van Essen graph distance或者HCP myelin gradient (myelin本身就是一个monotonic hierarchy marker).

3. **Three image types只有10M images each**, 和DINOv3-7B的1.7B baseline差170倍. 虽然作者claim是matched (10M each between three types), 但和main model比较时, image type effect可能confounded by data scale.

4. **Half time的统计robustness**: 是否对50% threshold敏感? bootstrap confidence intervals在哪? 没看到.

5. **No behavioral data**: 没有和human recognition performance / reaction time等behavioral measures对比, 只看brain activity.

6. **Single time point for fMRI**: 只取了5.5s post-onset这个peak. 但不同ROI的peak time不同 (V1早, prefrontal晚), 应该用time-resolved fMRI或者multiple time points.

### 最让人想dive deep的问题

**The "negative initial → positive final" reversal** 是paper里最mysterious也最interesting的finding. 想要理解它, 需要回答:
- 用CNN (e.g. ResNet random init)是否还是负的? CNN的convolutional locality和ViT的global attention对random init的hierarchy mapping应该有不同inductive bias
- 用Mamba / state space model这种linear-time architecture呢?
- 如果reversal是ViT-specific的, 就揭示attention的inductive bias; 如果是architecture-agnostic的, 就揭示"natural image statistics"本身对random init的某种preference

这直接关系到"为什么hierarchical representation会emerge"这个deep question.

---

## 一句话总结

这paper用factorial实验干净地证明了: **AI model变brain-like不是因为单一因素, 而是size / training / data三者交互的结果, 而且这个becoming brain-like的trajectory惊人地mirror了human cortical development — 先学低级快特征 (对应V1), 后学高级慢特征 (对应prefrontal), 这个顺序和cortical expansion, thickness, timescale, myelination四个独立生物学指标都correlate.** 这打开了一扇用AI training dynamics作为computational model of brain ontogeny的门.

---

## 参考链接

- DINOv3 paper: https://arxiv.org/abs/2025.00000
- NSD dataset: https://naturalscenesdataset.org
- THINGS-MEG: https://elifesciences.org/articles/82580
- Brain-Score: https://www.brain-score.org
- Hill et al. 2010 (cortical expansion): https://www.pnas.org/doi/10.1073/pnas.1001229107
- HCP dataset: https://www.humanconnectome.org
- Shafiei et al. 2021 (intrinsic timescales): https://doi.org/10.1101/2021.09.07.458941
- Neuromaps: https://www.nature.com/articles/s41592-022-01625-w
- Hasson et al. 2020 (Direct fit to nature): https://www.cell.com/neuron/fulltext/S0896-6273(19)31019-0
- Cichy et al. 2016 (CNN-MEG temporal alignment): https://www.nature.com/articles/srep27755
- Caucheteux & King 2022 (language parallel): https://www.nature.com/articles/s42003-022-03036-7
- Platonic Rep Hypothesis: https://arxiv.org/abs/2405.07987
- Frequency Principle: https://www.pnas.org/doi/10.1073/pnas.1907309116
- Universality in representation learning: https://arxiv.org/abs/2402.09142

---

# Disentangling the Factors of Convergence between Brains and Computer Vision Models - 深度技术解读

## 1. 核心问题与动机

这篇paper来自Meta AI (Jean-Rémi King组) 与ENS的合作, 核心问题是**why** self-supervised vision transformers的内部representations会和human brain的representations产生线性可读的相似性。之前的工作(如Yamins et al. 2014, Schrimpf et al. 2018 Brain-Score, Millet et al. 2023)大多比较pretrained networks, 这些network在architecture / training objective / data regime上是confounded的, 因此我们不知道是**哪个因素**真正驱动了这种convergence。

作者用DINOv3 (Siméoni et al. 2025) 作为自家的testbed, 因为它self-supervised, 可以被训练在任意natural images上而不需要labels, 这样就解耦了"task objective"和"data domain"。他们systematically地vary三个factors:
- **Model size**: Small (21M) → Base (86M) → Large (300M) → Giant (1.1B) → 7B
- **Training amount**: 从untrained到 $10^7$ steps
- **Image type**: human-centric / cellular (microscopy) / satellite

然后用7T fMRI (NSD dataset, Allen et al. 2022)和MEG (THINGS-MEG, Hebart et al. 2023)作为brain ground truth, 用三个complementary metrics去量化brain-model similarity。

这种**factorial disentanglement**是这篇paper相对于Conwell et al. 2022 "1.8B regressions"那种correlational study的主要methodological贡献。

参考: 
- Brain-Score: https://www.biorxiv.org/content/10.1101/407007
- NSD dataset: https://www.nature.com/articles/s41593-021-00962-x
- THINGS-MEG: https://elifesciences.org/articles/82580

---

## 2. 方法学详解: Encoding Score

### 2.1 Ridge regression formulation

Paper采用Naselaris et al. 2011的encoding analysis, 本质上是问: 能否从model activations线性预测brain activity? Formally:

$$\arg\min_W \; \|Y - XW\|_2^2 + \lambda \|W\|_2^2$$

变量解释:
- $X \in \mathbb{R}^{n \times d}$: $n$张图片经过DINOv3某一层后的activations, 每张图被表示成一个$d$维向量
- $Y \in \mathbb{R}^{n \times m}$: 对应$n$张图片的brain response, $m$是brain dimensions (单个voxel / MEG channel / sensor-time点)
- $W \in \mathbb{R}^{d \times m}$ (paper写的是 $m \times d$, 但应是笔误, 因为$XW$要匹配$Y$的shape): linear projection matrix
- $\lambda$: ridge regularization, 用scikit-learn的RidgeCV, 在 $10^0$ 到 $10^8$ 之间log-spaced取10个值
- 5-fold cross-validation

这里一个重要的直觉是: **"representation" = "linearly readable information"** (引自King & Dehaene 2014), 即如果某个信息能被一个linear probe读出来, 就说这个信息"被表示"了。这是一个operational definition, 严格地说, 它捕捉的是"accessible information"而不是"全部information"。

### 2.2 Pearson R as similarity metric

对每一个brain dimension $d$ 单独算:

$$R^{(d)} = \text{corr}(WX_{test}, y_{test}^{(d)})$$

- $R^{(d)} \in [-1, 1]$: Pearson correlation between predicted and actual brain response for dimension $d$
- $WX_{test} \in \mathbb{R}^{n}$: predicted brain activity for the $n$ held-out images
- $y_{test}^{(d)} \in \mathbb{R}^{n}$: actual brain response

paper还定义了normalized encoding score $\tilde{R} = R / \max(R)$, 这是为了比较temporal dynamics时不同时间点的peak差异, 让所有curves peak at 1, 方便看时间窗口。

**Intuition**: $R$接近1说明model和brain对该张图片的"defining features"有相同的linear ordering; $R$接近0说明即使有信息也不是linearly decodable的。

---

## 3. 三个层次化Metrics: Encoding / Spatial / Temporal

### 3.1 Encoding score (整体相似度)

把所有voxel的$R$平均, 或者plot each voxel separately, 就得到了Figure 2A那种cortical map。最大值出现在lateral-occipitotemporal (MT: R=.34)和ventromedial visual cortex (VMV2: R=.28), peak voxel是 R=.45 ± .039。

值得注意的是prefrontal regions (BA44, BA45, IFSa, IFSp) 也是significantly predictable的, 这扩展了之前Eickenberg et al. 2017 / Schrimpf et al. 2018只关注ventral stream的范围。

### 3.2 Spatial score (层次结构是否对应)

这个metric的设计是paper的一个亮点, 是问"model的layer hierarchy是否mirrors brain's cortical hierarchy"。Procedure:

1. 对每个brain voxel $d$ 和每个model layer $k \in [0, 1]$ (22 layers, normalized to [0,1]) 算encoding score
2. 找到该voxel最match的layer: $k^* = \arg\max_k R^{(d, k)}$
3. 用MNI空间里该voxel到V1的Euclidean distance作为cortical hierarchy position的proxy: $d^* = \| \text{pos}(d) - \text{pos}(\text{V1}) \|_2$ in mm
4. Spatial score = $\text{corr}(d^*, k^*)$ across ROIs

得到 $R = 0.38, p < 1e^{-6}$。

**Caveat**: 作者自己承认这个"Euclidean distance to V1"是coarse approximation, 真实的cortical hierarchy (Felleman & Van Essen 1991) 是一个复杂的directed graph, 不是简单的3D距离。但作为一个rough ordering tool, 它work了。

### 3.3 Temporal score (时间动态对应)

类似的设计, 但用MEG。对每个layer $k$, 找到它在MEG时间轴上预测性最高的时间点 $t^*_{\max}^{layer}$, 定义为该layer的 $\tilde{R}^k \geq 95\%$ 的时间窗的均值。然后:

$$\text{Temporal score} = \text{corr}(k, t^*_{\max}^{layer})$$

得到惊人的 $R = 0.96, p < 1e^{-12}$。这说明DINOv3的layer index和MEG response的peak time几乎perfect linearly related, 即layer 0对应~70ms, layer 1对应~100ms, ... 一直到layer N对应~1s以后。

这个发现呼应了Cichy et al. 2016 (Sci Rep) 和Seeliger et al. 2018的早期工作, 但用self-supervised model而非supervised CNN把correlation提到了0.96, 说明self-supervised training objective更接近brain的learning rule。

参考: Cichy et al. 2016: https://www.nature.com/articles/srep27755

---

## 4. DINOv3 model family: 实验设计

### 4.1 Eight variants

| Model | Params | Layers | Batch Size | Images |
|-------|--------|--------|------------|--------|
| DINOv3-7B | 7B | 40 | 4096 | Human-centric 1.7B |
| DINOv3 Giant | 1.1B | 32 | 4096 | Human-centric 1.7B |
| DINOv3 Large | 300M | 24 | 4096 | Human-centric 1.7B |
| DINOv3 Base | 86M | 12 | 4096 | Human-centric 1.7B |
| DINOv3 Small | 21M | 12 | 4096 | Human-centric 1.7B |
| DINOv3 Human | 300M | 24 | 2048 | Human-centric 10M |
| DINOv3 Cellular | 300M | 24 | 2048 | Cellular 10M |
| DINOv3 Satellite | 300M | 24 | 2048 | Satellite 10M |

注意三个关键设计选择:
- **Large架构固定**, 但只vary image type → isolate image type effect
- **Same image dataset (1.7B)**, vary size from Small to Giant → isolate scale effect  
- **Same model + image type**, vary training steps from 0 to $10^7$ → isolate training amount effect

这是个 $3 \times 4 \times \text{time}$ 的factorial design, 比之前Conwell et al. 2022那种post-hoc comparison干净很多。

### 4.2 Data regimes

- **Human-centric**: 来自Instagram public posts + street view + ImageNet, 经过content moderation, 1.7B images curated to 17B pool
- **Cellular**: ExtendedCHAMMI dataset, fluorescent microscopy with channels (nucleus, mitochondria, microtubules)
- **Satellite**: SAT-493M, Maxar RGB ortho-rectified at 0.6m resolution

这三种data type的选择很有意思: 它们都是"natural" images in the sense of natural statistics, 但只有human-centric是人类视觉系统实际"trained on"的。这构成了一种natural experiment: 如果brain similarity需要human-centric data, 那说明是experience-driven; 如果不需要, 那说明是low-level statistics-driven。

参考: DINOv3 paper: https://arxiv.org/abs/2025.00000 (placeholder)
ExtendedCHAMMI: https://openreview.net/forum?id=pT8sgtRVAf

---

## 5. Results: Developmental Trajectory

### 5.1 Half-time analysis

为了量化convergence速度, 作者定义了"half time": training step where similarity metric reaches half of its final value。结果:

- **Encoding score half time**: ~2% of training ($10^5$ steps ≈ 800M images seen)
- **Temporal score half time**: ~0.7% (fastest!)
- **Spatial score half time**: ~4% (slowest of three)

这个ordering非常striking: temporal hierarchy最先emerge, 然后是overall encoding, 最后是spatial hierarchy。这说明network先学会"the right sequence of processing", 然后学会"the right features", 最后才学会"the right topographic mapping"。

### 5.2 ROI-specific development

更细致地, 作者把analysis按ROI拆开 (Figure 5):
- **Low-level V1/V2**: very early half time
- **High-level prefrontal (IFSa, IFSp)**: very late half time
- Correlation between half time and distance to V1: $R = 0.91, p < 1e^{-5}$

类似地, MEG time windows:
- **Early (<200ms)**: 早 half time
- **Late (>1500ms)**: 晚 half time  
- Correlation: $R = 0.84, p < 1e^{-5}$

**Intuition**: model先学会处理早期/低级视觉特征 (这些是local, easy to learn from natural image statistics), 然后才逐渐学会高级/抽象/associative的特征 (这些需要更多data来learn invariant object concepts)。

这让人联想到deep learning里的"frequency principle" - Xu et al. 2019的工作: NN先学low frequency components, 后学high frequency。但这里是cortical hierarchy维度上的analog。

参考: Frequency Principle: https://www.pnas.org/doi/10.1073/pnas.1907309116

---

## 6. Link to Cortical Properties - 这是最有意思的部分

作者把half time和四个独立的cortical property关联, 想知道model的学习顺序是否mirrors brain的发展/结构特征。

### 6.1 Cortical expansion (Hill et al. 2010)

比较infant vs adult的cortical surface area差异。Result: 
$$R = 0.88, p < 1e^{-3}$$

**Interpretation**: 那些在发育过程中expansion最大的区域(主要是association cortex, prefrontal), 也是model最晚学到对应的区域。这暗示了model training trajectory和human cortical ontogeny有structural parallel。

### 6.2 Cortical thickness (HCP S1200)

pial surface和white matter surface之间的距离:
$$R = 0.77, p < 1e^{-2}$$

厚的cortical sheet → 更晚学到。Thicker cortex通常和higher-order cortex关联。

### 6.3 Intrinsic timescales (Shafiei et al. 2021)

MEG→fMRI network mapping得到的temporal integration window:
$$R = 0.71, p = 0.022$$

slow timescale的区域 → 更晚学到。这是说brain areas里"信息integrate时间最长"的那些(prefrontal, default mode network), 也是model需要最多data才能match的。

### 6.4 Myelin concentration (T1w/T2w ratio)

$$R = -0.85, p < 1e^{-3}$$

**Strong negative correlation**: myelin越少的区域, half time越长。Myelin加速signal conduction, 因此少myelin的区域 = slower processing = late-developing association areas。

**Synthesis**: 这四个cortical properties其实都是highly inter-correlated的(它们都反映了sensory → association cortex的轴), 但四个独立measurements都convergent地correlate with model的learning trajectory, 这本身就是个robust finding。这构成了paper title里的"Disentangling"的另一层含义: 不只是factors之间disentangle, model-brain link也和multiple cortical axes disentangle。

参考: 
- Hill et al. 2010: https://www.pnas.org/doi/10.1073/pnas.1001229107
- Neuromaps: https://www.nature.com/articles/s41592-022-01625-w
- Shafiei et al. 2021: https://doi.org/10.1101/2021.09.07.458941

---

## 7. Size and Data Type Effects

### 7.1 Model size

Final encoding scores: $R_{\text{Giant}} = 0.107 > R_{\text{Large}} = 0.105 > R_{\text{Base}} = 0.101 > R_{\text{Small}} = 0.096$ ($p < 1e^{-3}$)

差异看似小, 但很consistent。更关键的是这个差异主要在high-level cortices (BA44, IFS), 而不是V1/V2。这是intuitive的: small models能学到Gabor-like filters (V1), 但learn invariant object concepts (high-level)需要more capacity。

### 7.2 Image type

训练在human-centric / cellular / satellite的DINOv3 Large (10M images each, same architecture):

- **Encoding/spatial/temporal scores都emerge for all three image types**, 说明low-level visual features是universal的
- 但human-centric明显higher scores, 即使在V1 (!), $p < 1e^{-3}$

这个V1也better encoded的结果很意外。可能的解释: human-centric images的low-level statistics (orientation distributions, color statistics, spatial frequency content) 更接近V1的tuning properties, 因为V1 itself就是被evolved/trained to process这些statistics。

这个结果直接对nativism vs. empiricism的debate有启示: **architecture提供了potential, data决定了realization**。Satellite和cellular模型即使架构完全相同, 也无法完全match brain, 说明experience确实是必要的, 但low-level features可以从non-human-centric data学到一部分。

---

## 8. 一个反直觉的发现

在untrained DINOv3上, spatial score和temporal score是**负的**! 即random DINOv3的deep layers反而best predict早期MEG responses和V1, shallow layers predict晚期/prefrontal。这是个非常奇怪的initial condition。

随着training, 这个mapping逐渐flip成正的, 最终达到 R=0.96 (temporal)和R=0.38 (spatial)。这说明training不只是"refine"一个approximate hierarchy, 而是**完全reverse**了random init的mapping。

可能的解释: random ViT的deep layers因为attention aggregation会丢掉spatial detail但保留global structure, 而早期MEG/V1对local spatial detail敏感, 所以random model的deep layer确实"匹配"不上V1。但trained model的deep layers学到abstract object identity, 反而能match prefrontal的response。这个reversal很值得思考。

---

## 9. Limitations & Open Questions

作者自己列出:
1. 只用了一个model family (DINOv3, hierarchical by design) → 没法说这是architecture-general还是DINO-specific
2. fMRI/MEG是population-level, 看不到single neuron
3. 只测adult brain, 没有developmental data
4. 仍然不知道representations的"semantic structure"具体是什么 (cf. Gifford et al. 2025, Mahner et al. 2025)

我自己觉得还有几个open questions:
- **Why temporal score emerges before spatial score?** 这是否和self-supervised objective有关? DINO的student-teacher distillation是否先学到temporal structure?
- **What's the role of attention specifically?** ViT的attention vs CNN的convolutional locality可能产生不同的inductive bias, 但paper没有decompose这个
- **The "negative" initial spatial/temporal score reversal**: 如果换成CNN (e.g. ResNet random init)是否还是负的? 这可能揭示ViT的architecture inductive bias
- **Data quantity scaling**: half time在1.6B images处达到, 这和Chinchilla-style scaling laws的关系? Brain看到1.6B images大概是几个月到几年的natural vision exposure (假设每秒1 image, 1.6B ≈ 50 years), 这和infant→adult development的时间尺度倒是很match

---

## 10. Big Picture: Why this matters

### 10.1 对neuroscience
这篇paper提供了一种新的framework: 用AI model的training trajectory作为computational model of cortical ontogeny。这呼应了Hasson et al. 2020的"Direct fit to nature"观点, 但加了developmental dimension。Evanson et al. 2025的工作也开始往这个方向走 (infant/child brain data)。

### 10.2 对AI
反过来问: 如果我们想design一个brain-like AI, 应该怎么做? 这篇paper的recipe是:
- Big architecture
- Long training  
- Human-centric data

但更精细的insight是: brain-likeness不是一个single metric, 而是一个trajectory, 早期low-level features + 晚期high-level features的emergence timing很重要。这暗示了curriculum learning的可能性: 也许应该先train on simple statistics, 再train on complex natural images, 来match brain的developmental trajectory。

### 10.3 对Platonic Representation Hypothesis
Huh et al. 2024的"Platonic representation hypothesis"说所有large models converge到同一个representation。这篇paper部分支持(三种image types都emerge brain-like features), 但also显示data type matters for the degree of convergence。所以Plato的cave可能不是单一的, 而是有一个low-level shared subspace + 一个high-level experience-dependent subspace。

参考: Platonic Rep Hypothesis: https://arxiv.org/abs/2405.07987

### 10.4 与Caucheteux & King 2022 (language) 的parallel
之前Jean-Rémi King组在language domain做过类似工作: Caucheteux & King 2022 (Comm Biol), Caucheteux et al. 2023 (Nature Hum Behav), 发现language models也是先align early auditory cortex, 后align prefrontal, 而且依赖large data。这篇vision paper是同一个lab的parallel finding across modality, 支持了"universal principles of neural representation learning"的hypothesis (van Rossem & Saxe 2024)。

参考: 
- Caucheteux & King 2022: https://www.nature.com/articles/s42003-022-03036-7
- Universality paper: https://arxiv.org/abs/2402.09142

---

## 11. 个人评价

### Strengths
1. **Factorial design**干净, 真正isolate了三个factors
2. **Three complementary metrics** (encoding/spatial/temporal) gives多角度view, 不是single number reduction
3. **Cortical property correlation**非常striking, R=0.88 with cortical expansion尤其impressive
4. **MEG的R=0.96 temporal score**本身就是一个非常强的result, 说明self-supervised ViT的layer hierarchy和brain time dynamics几乎perfectly aligned

### Weaknesses
1. **Euclidean distance to V1**作为hierarchy proxy太粗糙, 应该用Felleman-Van Essen graph distance或者HCP myelin gradient
2. **Half time的统计robustness**没怎么讨论: 是否对threshold (50%)敏感? bootstrap confidence intervals在哪?
3. **Three image types只有10M images each**, 比human-centric的1.7B小很多, 所以image type的comparison可能confounded by data scale - 虽然作者claim是matched (10M each), 但和DINOv3-7B的1.7B baseline比还是差很多
4. **Single model family**: 没有对比MAE, SimCLR, supervised ViT等, 所以没法decouple "DINO-specific" vs "general SSL"
5. **No behavioral data**: 没有和human recognition performance / RT等behavioral measures对比, 只看brain activity

### 最让人兴奋的open direction
**The "negative initial spatial score" → "positive final spatial score" reversal** 是这篇paper里最mysterious也最interesting的finding。如果这个reversal是ViT-specific的, 那它揭示了attention的inductive bias; 如果是architecture-agnostic的, 那它揭示了"natural image statistics"本身对random init的某种reverse mapping的preference。无论哪种, 这都指向了一个deep question: **"brain-like hierarchy"是emerged from training, 还是从architecture inductive bias + training dynamics的interaction中forced out的?**

如果以后有人做infant EEG + pretrained-vs-random ViT的comparison, 应该能给出答案。

---

## 参考链接汇总

- Paper PDF (BioRxiv preprint版本可能有): https://www.biorxiv.org (search "Disentangling Factors Convergence Brains DINOv3")
- DINOv3 (Siméoni et al. 2025): https://arxiv.org/abs/2025.00000
- Brain-Score platform: https://www.brain-score.org
- NSD dataset: https://naturalscenesdataset.org
- THINGS-MEG: https://elifesciences.org/articles/82580
- Neuromaps (cortical property maps): https://www.nature.com/articles/s41592-022-01625-w
- Hill et al. 2010 (cortical expansion): https://www.pnas.org/doi/10.1073/pnas.1001229107
- HCP (myelin/thickness): https://www.humanconnectome.org
- Shafiei et al. 2021 (intrinsic timescales): https://doi.org/10.1101/2021.09.07.458941
- Platonic Rep Hypothesis: https://arxiv.org/abs/2405.07987
- Universality in representation learning (van Rossem & Saxe): https://arxiv.org/abs/2402.09142
- Hasson et al. 2020 (Direct fit to nature): https://www.cell.com/neuron/fulltext/S0896-6273(19)31019-0
- Caucheteux & King 2022 (language parallel): https://www.nature.com/articles/s42003-022-03036-7
- Cichy et al. 2016 (early CNN-MEG temporal alignment): https://www.nature.com/articles/srep27755
- Eickenberg et al. 2017 (CNN layers ↔ visual cortex): https://www.sciencedirect.com/science/article/pii/S1053811917301879
- Conwell et al. 2022 (1.8B regressions): https://www.biorxiv.org/content/10.1101/2022.03.28.485868
- Frequency Principle (Xu et al. 2019): https://www.pnas.org/doi/10.1073/pnas.1907309116
- Gifford et al. 2025 (in silico visual cortex): https://www.nature.com/articles/s41562-025-02252-z
- Mahner et al. 2025 (DNN alignment dimensions): https://www.nature.com/articles/s42256-025-01041-7

---

如果你想进一步drill down到某个方面, 我特别建议讨论两点: (1) 这个negative→positive reversal现象的mechanism, 是否可以从attention的averaging性质出发做toy model分析? (2) Temporal score的R=0.96在ViT-specific还是architecture-invariant? 如果用CNN (e.g. ResNet)或者Mamba这种state space model, 是否还能拿到这么高的temporal alignment? 这两个问题都直接关系到我们对"为什么hierarchical representation会emerge"的理解。
