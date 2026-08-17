---
source_pdf: Sapiens Foundation for Human Vision Models.pdf
paper_sha256: 4cbfeeeec8574634d2629da2c06a41a9ed805de4defd41c0b054f4343d94df73
processed_at: '2026-08-12T02:56:19-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Sapiens 用人话讲

## 一句话概括

Meta 的一帮人搞了个专门看人的大模型，叫 Sapiens。核心 idea 特别朴素：**与其用通用图像 pretrain 再 finetune 到人，不如一开始就只用人的图像来 pretrain**。听起来 obvious，但他们真的花了 1024 块 A100 烧了 18 天，把这事做到极致了。

## 为什么这事有意思

现在 foundation model 圈子的主流信仰是 "general is better"——GPT 什么都能写，CLIP 什么都能认，DINOv2 啥图像 feature 都能提。但 Sapiens 反其道而行：**narrow but deep**。只看人，但看得很透。

这就像说，与其培养一个什么都懂一点的通才，不如培养一个对人体结构烂熟于心的专家。专家在通用知识上不行，但在本专业上秒杀通才。

## 他们具体干了啥

### 1. 收数据

从 10 亿张图里筛出 3 亿张只有人的图。筛选标准很粗暴：
- 用 Detectron2（https://github.com/facebookresearch/detectron2）跑 person detector
- detection score 大于 0.9
- bounding box 大于 300 像素

就这么简单。没有 fancy 的标注，全是 self-supervised，不需要任何 label。

### 2. Pretrain

用 MAE（Masked Autoencoder，https://arxiv.org/abs/2111.06377）。MAE 的原理你肯定熟，但简单说就是：

把图片切成 patch，随机挡住 75%，让模型猜挡住的是啥。

Sapiens 的特别之处在于**分辨率**：1024×1024，patch size 16，一张图 4096 个 token。对比之下，标准 ViT 在 224 分辨率下只有 196 个 token。这意味着 Sapiens 每个 token 只看图像面积的 0.02%，而标准 ViT 是 0.4%。

**直觉**：这就像给你看一个人的脸，普通模型看到的是 "眼睛、鼻子、嘴" 这种粗粒度，Sapiens 看到的是 "左眼外眼角、鼻翼右侧、上唇左侧" 这种细粒度。对手指、脚趾、微表情这种需要毫米级精度的任务，这个区别是致命的。

### 3. 四个下游任务

pretrain 完了，finetune 到四个任务：

- **Pose estimation**：预测 308 个 keypoints，包括 243 个 face landmarks
- **Body-part segmentation**：28 类，连上下嘴唇、牙齿、舌头都分
- **Depth estimation**：单目深度
- **Surface normal**：每个像素的表面法向量

这四个任务都是 3D 人体重建的基础组件。Meta Reality Labs 做 Codec Avatars（https://about.meta.com/realitylabs/codecavatars/）需要这些。

## 最关键的实验结果

Table 7 是整篇论文的灵魂。同样的 0.3B 模型，同样的训练 schedule，只换 pretraining data：

| Pretraining Data | Pose (mAP) | Segmentation (mIoU) |
|---|---|---|
| 3 亿张通用图像 | 37.3 | 52.8 |
| 3 亿张人像图像 | **47.0** | **66.5** |

差距巨大。Pose 高了 9.7，Segmentation 高了 13.7。

更狠的是：1 亿张人像 vs 3 亿张通用——人像数据量只有通用数据的三分之一，但 pose 依然高 6.3（43.6 vs 37.3）。

**直觉**：这就好比学法语。你可以先学 10 种语言再专攻法语，也可以一开始就只学法语。后者虽然 "见过的语言少"，但在法语上的熟练度远超前者。因为模型的 capacity 全用来记法语的语法、词汇、语境了，没浪费在无关语言上。

## 跟 SOTA 的对比

### Pose

Sapiens-2B 在 Humans-5K 上 61.1 AP。之前 SOTA 是 DWPose-l 的 53.5 AP，提升 7.6。

更值得注意：**Sapiens-0.3B（53.4）≈ DWPose-l（53.5）**。DWPose 用了复杂的 student-teacher distillation，Sapiens 就一个 plain ViT + MAE pretrain + finetune。同参数量打平，说明 pretraining data distribution 比 architecture trick 重要得多。

按 body part 看 gap 更清楚：
- Foot：Sapiens-2B 69.4 vs DWPose 56.5，差 12.9
- Hand：Sapiens-2B 57.1 vs DWPose 40.1，差 17.0
- Face：Sapiens-2B 76.9 vs DWPose 74.3，差 2.6

foot 和 hand 的 gap 巨大，face 的 gap 小。因为 foot 和 hand 在图像里占比小，高分辨率 + 细粒度 token 的优势在这里爆发。face 本来就占图像较大区域，低分辨率也能看清。

### Segmentation

Sapiens-0.3B（76.7 mIoU）已经超过 Mask2Former（58.7）12 分。Sapiens-2B 达到 81.2。

### Depth

只用了 600 个 RenderPeople（https://renderpeople.com/）扫描的合成数据 finetune，在真实数据 Hi4D 上 RMSE 0.114，比 Depth Anything（https://arxiv.org/abs/2401.10891）的 0.147 低 22.4%。

**这很疯狂**。纯合成数据 finetune，在真实 multi-human interaction 场景上超越用 60M 数据训练的 general depth model。说明 human-centric pretraining 的 prior 足够强，能 bridge 合成到真实的 domain gap。

### Normal

Sapiens-2B 在 THuman2.0 上 angular error 11.84°，ECON（https://arxiv.org/abs/2212.07422）是 25.45°，降了 53.5%。

ECON 用了 4000 个 scans 训练，Sapiens 只用 600 个。数据少 6.67 倍，性能反而好一倍。pretraining 的威力。

## 技术细节里几个有意思的点

### Depth loss 的设计

$$\mathcal{L}_{\text{depth}} = \sqrt{\overline{(\Delta \mathbf{d})^2} - \frac{1}{2}(\overline{\Delta \mathbf{d}})^2}$$

其中 $\Delta \mathbf{d}_i = \log(\mathbf{d}_i) - \log(\hat{\mathbf{d}}_i)$，$\overline{(\Delta \mathbf{d})^2} = \frac{1}{M}\sum_{i=1}^{M}(\Delta \mathbf{d}_i)^2$，$\overline{\Delta \mathbf{d}} = \frac{1}{M}\sum_{i=1}^{M}\Delta \mathbf{d}_i$，$M$ 是 human pixel 数量。

这里 $M$ 只算 human pixel，background 不参与 loss。所以模型完全专注人体深度，不浪费 capacity 学背景。

log-space 是因为 depth 是 multiplicative——远处的 1 米和近处的 1 米，感知上差异不同。log 把 multiplicative 变 additive。

那个 $\frac{1}{2}$ 系数很有意思。如果完全 scale-invariant，应该是 $\overline{(\Delta \mathbf{d})^2} - (\overline{\Delta \mathbf{d}})^2$，完全去掉 mean shift。但用 $\frac{1}{2}$ 意味着**保留一半的 mean shift 惩罚**。这是个折衷：完全 scale-invariant 会让模型忽略绝对深度（对 AR/VR 不好），完全 MSE 又对 scale 敏感。半 scale-invariant。

### Normal loss 的设计

$$\mathcal{L}_{\text{normal}} = ||\mathbf{n} - \hat{\mathbf{n}}||_1 + (1 - \mathbf{n} \cdot \hat{\mathbf{n}})$$

$\mathbf{n}$ 和 $\hat{\mathbf{n}}$ 都是单位法向量，$\mathbf{n} \cdot \hat{\mathbf{n}}$ 是 cosine similarity。

两项：L1 惩罚 component-wise 差，cosine 惩罚方向差。单独 L1 对方向不敏感，单独 cosine 对 magnitude 不敏感。组合起来更 robust。

### LLRD = 0.85

Finetune 时 layer-wise learning rate decay 0.85。这意味着第 1 层 lr 是 base lr × 0.85^(L-1)，到最后一层基本接近 base lr。早期层 lr 被压得很低，保护 pretraining feature 不被 finetune 破坏。

0.85 是个比较 aggressive 的 decay。BERT finetune 常用 0.8-0.9。这个选择说明 Sapiens 的 pretraining feature 很 valuable，不想在 finetune 时 wash out。

### Width over Depth

从 0.3B 到 2B，hidden size 1024 → 1920（1.875x），layers 24 → 48（2x）。作者明确说 "prioritize width over depth"。

这个选择跟 LLaMA（https://arxiv.org/abs/2302.13971）的结论一致。宽网络的好处：
- Finetune 更 stable
- LLRD 下梯度传播更均匀
- 短而宽比长而窄更容易优化

## 我的思考

### 1. 这验证了 "data distribution is everything"

你以前在 CS231n 讲过（https://cs231n.github.io/），深度学习的 success 三要素是 data、architecture、training。Sapiens 把 data distribution 这个要素的重要性推到了极致。

architecture 是最 boring 的 plain ViT，training 是最 boring 的 MAE + finetune，但 data distribution 选对了，性能就爆了。这说明 modern DL 的 bottleneck 不在 architecture innovation，而在 data curation。

### 2. 跟 general foundation model 的路线分歧

DINOv2、CLIP、SAM 这些走 general 路线。Sapiens 走 domain-specific 路线。两条路都有道理，但适用场景不同：

- General model：适合 zero-shot、多任务、research exploration
- Domain-specific model：适合 production、高精度、特定场景

我觉得未来会分化。通用 foundation model 会继续 scale 到 trillion params，做 reasoning。但每个垂直领域（medical、driving、human、satellite）会有自己的 foundation model，参数量可能没那么大，但在本领域秒杀通用 model。

### 3. 跟你在 Tesla 的工作的联想

Tesla 的 vision system 处理 driving scene。Sapiens 的方法论可以直接迁移：

- 收 10 亿张 driving 图像（含 ego-car、other cars、pedestrians、lanes）
- MAE pretrain at high resolution
- Finetune 到 detection、lane segmentation、depth、planning

这种 narrow-domain foundation model 可能比用 SAM 或 DINOv2 做 perception backbone 更 efficient。因为 driving scene 的 distribution 跟 ImageNet 差太远了。

### 4. 1K 分辨率的工程代价

Sapiens-2B 的 8709 GFLOPs 推理很贵。论文里没提 latency。但 0.3B 已经很强，可能是 production 的 sweet spot。2B 更像 research showcase。

### 5. 合成数据的胜利

Depth 和 normal 都用合成数据 finetune 就 SOTA 了。这对未来 data pipeline 设计有启示：

- Pretraining 用大规模真实无标注数据学 distribution
- Finetuning 用小规模高精度合成数据学 task-specific mapping
- Domain gap 由 pretraining 的强 prior 来 bridge

这个 recipe 可能比 "收集百万张真实标注" 更 efficient，尤其是标注成本高的 task（如 depth、normal 需要 3D scan）。

### 6. 没解决的问题

- 多人遮挡场景还是不行（Limitations 部分承认）
- 1K 分辨率推理太贵
- Humans-300M 是 Meta 内部数据，外部无法复现
- 没跟 DINOv2 在相同 downstream task 上直接对比
- Video 还没做

## 最后的 takeaway

Sapiens 给我的最大启示：

**如果你要做某个垂直领域的 perception，与其用 general foundation model finetune，不如自己 curate 该领域的大规模无标注数据，pretrain 一个 domain-specific foundation model。**

这个 recipe 简单到不像话：好的 data + plain ViT + MAE + finetune。没有 architecture innovation，没有 training trick，就是 data + scale。

这呼应了 Rich Sutton 在 "The Bitter Lesson"（http://www.incompleteideas.net/IncIdeas/BitterLesson.html）里的观点：最终胜出的是 compute + data，不是 clever method。Sapiens 的 method 不 clever，但 data 和 compute 用到位了，性能就爆了。

**相关链接**：
- Sapiens 项目页：https://about.meta.com/realitylabs/codecavatars/sapiens
- MAE 论文：https://arxiv.org/abs/2111.06377
- DINOv2 论文：https://arxiv.org/abs/2304.07193
- Depth Anything 论文：https://arxiv.org/abs/2401.10891
- ECON 论文：https://arxiv.org/abs/2212.07422
- PIFuHD 论文：https://arxiv.org/abs/2004.00452
- Chinchilla 论文：https://arxiv.org/abs/2203.15556
- Bitter Lesson：http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- DWPose 论文：https://arxiv.org/abs/2307.01846
- RenderPeople：https://renderpeople.com/
- COCO-WholeBody：https://arxiv.org/abs/2007.11858
- Hi4D：https://arxiv.org/abs/2305.02185
- THuman2.0/Function4D：https://arxiv.org/abs/2105.02060

如果你想聊某个具体部分更深，比如 MAE 为什么 work、depth loss 的数学推导、或者怎么把这套方法搬到别的 domain，随时说。

---

# Sapiens: Foundation for Human Vision Models 深度解析

## 1. 论文核心 Thesis

Sapiens 的核心论点非常清晰：**给定相同 compute budget，在 curated human images 上做 self-supervised pretraining，能显著提升各种 human-centric 任务的性能**。这听起来像一句废话，但 Table 7 的 ablation 把它量化得很有说服力——同样的 0.3B 模型、同样的训练 schedule，Humans-300M vs General-300M 在 pose 任务上 mAP 差了 9.7（47.0 vs 37.3），在 normal 任务上 angular error 差了 5°（21.8 vs 26.8）。

这让我想起 Chinchilla 论文里关于 data quality 与 data quantity 的讨论（https://arxiv.org/abs/2203.15556），以及 DINOv2 论文里 LVD-142M 数据 curation 的故事（https://arxiv.org/abs/2304.07193）。Sapiens 把这个思路推到了 human domain 的极端——直接把数据集限定到只有人类的图像，然后 scale 到 2B 参数。

## 2. Architecture 与 Scaling 设计

### 2.1 ViT 配置

Table 2 给出了完整的 model specs，这里展开看一下 scaling 策略：

| Model | Params | Hidden | Layers | Heads | FLOPs | Batch |
|-------|--------|--------|--------|-------|-------|-------|
| Sapiens-0.3B | 0.336B | 1024 | 24 | 16 | 1.242T | 98,304 |
| Sapiens-0.6B | 0.664B | 1280 | 32 | 16 | 2.583T | 65,536 |
| Sapiens-1B | 1.169B | 1536 | 40 | 24 | 4.647T | 40,960 |
| Sapiens-2B | 2.163B | 1920 | 48 | 32 | 8.709T | 20,480 |

注意几个细节：
- **Width 优先于 depth**：从 0.3B → 2B，hidden size 从 1024 → 1920（1.875x），而 layers 从 24 → 48（2x），大致同步 scale，但作者在文中明确说"prioritize scaling models by width rather than depth"。这个选择跟 LLaMA 系列的结论一致（https://arxiv.org/abs/2302.13971），宽一点的网络 finetune 时更 stable，尤其是 layer-wise learning rate decay（LLRD）下。
- **Head 数 scaling**：从 16 → 32，head_dim 从 64 → 60，没有严格保持 head_dim 不变，可能是工程上的折衷。
- **Batch size 反向 scale**：模型越大 batch 越小（98k → 20k），这是因为显存约束。但 20k batch 对于 2B 模型做 MAE 来说仍然很大，这意味着 gradient accumulation 也不小。

### 2.2 高分辨率的代价与收益

这是 Sapiens 最激进的设计选择。原生 1024×1024 输入，patch size 16，意味着 **64×64 = 4096 个 tokens**，而标准 ViT-Large@224 只有 14×14 = 196 个 tokens，约 20× 多的 tokens。FLOPs 从 ViT-L 的 ~17.6 GFLOPs 跳到 Sapiens-0.3B 的 1.242 TFLOPs（注意是 T 不是 G），这跟作者自己说的"twentyfold more FLOPs"对得上。

每个 patch token 在 Sapiens 中只占图像面积的 **0.02%**，标准 ViT 是 0.4%。这意味着每个 token 的 receptive field（在 patch 层面）更小，inter-token reasoning 更细粒度——这对 human keypoints 这种需要毫米级精度的任务至关重要。Table 3 中 Sapiens-0.3B（58.1 body AP）超过 ViTPose+-L（61.0 body AP）实际上 body 上略低，但在 foot（56.8 vs 62.4）和 hand（49.6 vs 41.5）上明显领先，这正是高分辨率带来的细粒度收益。

## 3. Pretraining: MAE on Humans-300M

### 3.1 数据集构造

Humans-300M 来自约 1B in-the-wild 图像的筛选：
1. 去水印、文字、艺术图、非自然元素
2. 用 person detector（Detectron2，https://github.com/facebookresearch/detectron2）过滤
3. detection score > 0.9 且 bbox > 300 pixels

Figure 2 显示超过 248M 张图像包含多个人——这个细节很重要，因为 multi-person 场景的 distribution learning 对下游 multi-person segmentation（Fig 7）和 Hi4D 这种 interaction 场景（Table 5）的 generalization 至关重要。

### 3.2 MAE 的细节

MAE（https://arxiv.org/abs/2111.06377）的选择理由作者说得很直接：
- Single-pass inference，比 contrastive（DINO/iBOT）或 multi-view（SimCLR）效率高
- 同样的 compute 能处理更多图像

预训练 1.2 trillion tokens，masking ratio 75%（标准 MAE 设置）。Figure 3 下半部分展示了 inference 时 mask ratio 从 0.75 提到 0.95 仍然能 reconstruct 出 plausible 的人体结构——这是 model 真正学到了 human body prior 的强信号，而不只是 low-level texture completion。

**Intuition**：当你在 300M 张人类图像上做 MAE，模型本质上学到的是"human body 的 plausible completion manifold"。这个 manifold 对 pose（missing keypoint 可由其他 keypoint 推断）、segmentation（被遮挡的 body part）、depth（深度结构的约束）、normal（surface 几何）都是直接 useful 的 inductive bias。这跟 MAE 在 ImageNet 上学到的"texture + shape"prior 不同，Sapiens 学到的是更 narrow 但更深的 human-centric prior。

### 3.3 一个有趣的 ablation

Figure 10：随着 pretraining 中 unique human images 数量增加，normal estimation（% within 30°）持续上升，没有 saturation 迹象。这暗示 300M 可能还没到这个 architecture 的 data scaling ceiling。如果推到 1B human images，2B 模型可能还能继续涨——但这是 Meta 内部数据，外部研究者很难验证。

## 4. 下游任务 Finetuning

### 4.1 通用 finetune 架构

所有四个任务共享一个 encoder-decoder 结构：
- Encoder：pretrained MAE encoder，用 LLRD（layer-wise learning rate decay = 0.85），weight decay 0.1
- Decoder：随机初始化的 lightweight head（deconv + conv），task-specific

这个 design choice 很重要——它意味着 pretraining 学到的是 general human features，而 task-specific 信息主要靠 decoder + finetune 注入。这也是为什么 Sapiens-0.3B（架构类似 ViT-L）能在所有四个任务上超过专门的 SOTA model（如 DWPose 用了 student-teacher distillation）。

LLRD 0.85 是个偏 aggressive 的 decay，这意味着早期 layer 学习率被压得很低，保留 pretraining feature。0.85 这个数字跟 BERT finetune 的常用值一致。

### 4.2 Pose Estimation

#### 4.2.1 308 keypoints 的设计

新 skeleton 的 breakdown：
- Body: 17（COCO）+ 扩展
- Face: 243（vs 之前的 68 marking points）
- Hand: 每只手 21
- Foot: 每只脚 包括脚趾
- Surface: body surface 上的额外点

243 facial keypoints 是个很激进的设计——现有 face landmark 通常是 68（https://ibug.doc.ic.ac.uk/resources/300-W/）或 98（https://arxiv.org/abs/1906.06337）。Sapiens 把它推到 243 是为了捕捉细微表情，这对后续的 Codec Avatars 类应用（Meta Reality Labs 的核心方向，https://about.meta.com/realitylabs/codecavatars/）是直接必要的。

#### 4.2.2 Loss

$$\mathcal{L}_{\text{pose}} = \text{MSE}(\mathbf{y}, \hat{\mathbf{y}})$$

其中 $\mathbf{y} \in \mathbb{R}^{H \times W \times K}$ 是 ground truth heatmap，$\hat{\mathbf{y}} = \mathcal{P}(\mathbf{I})$ 是预测。K=308，输入 1024×768（4:3 ratio）。

这里有个细节：position embedding 通过 interpolation 适配 4:3 比例（reference [58] 是 SAM 的位置编码处理）。pretrain 时是 1024×1024 square，finetune 时变 1024×768，需要 2D bicubic interpolation。

#### 4.2.3 Table 3 解读

Sapiens-2B 在 Humans-5K 上达到 **61.1 whole-body AP**，比 DWPose-l 的 53.5 高了 7.6 AP。更有意思的是：
- **Sapiens-0.3B (53.4) ≈ DWPose-l (53.5)**，即同样参数量下，仅靠 human pretraining 就追平了用复杂 distillation 的 SOTA
- **Sapiens-0.6B (56.2) > DWPose-l (53.5) by 2.7 AP**，scaling 开始拉开差距

按 body part 拆分：
- Foot AP：Sapiens-2B 69.4 vs DWPose-l 56.5，差 **12.9 AP**——这是最大的 gap，因为 foot 在低分辨率下基本看不见
- Hand AP：Sapiens-2B 57.1 vs DWPose-l 40.1，差 **17.0 AP**——同样是高分辨率的胜利
- Face AP：Sapiens-2B 76.9 vs DWPose-l 74.3，差 2.6 AP——face 本来就 small region，差距没那么大

### 4.3 Body-Part Segmentation

#### 4.3.1 28-class 词汇表

扩展了 ATL（Look into Person，https://arxiv.org/abs/1703.05446）的 20-class，加入：
- upper/lower lip（分开）
- teeth
- tongue
- upper/lower limb halves（四肢上下半段分开）
- torso 细分

这个细粒度词汇表对 3D 重建尤其重要——比如 PIFuHD（https://arxiv.org/abs/2004.00452）和 ECON（https://arxiv.org/abs/2212.07422）都需要 body part segmentation 来 guide 几何先验。

#### 4.3.2 Loss

$$\mathcal{L}_{\text{seg}} = \text{WeightedCE}(\mathbf{p}, \hat{\mathbf{p}})$$

其中 $\mathbf{p} \in \mathbb{R}^{H \times W \times C}$ 是 GT class probability map，$\hat{\mathbf{p}} = \mathcal{S}(\mathbf{I})$。Weighted 是为了处理 class imbalance（torso 大，teeth/tongue 小）。

#### 4.3.3 Table 4 解读

- Sapiens-0.3B (76.7 mIoU) > Mask2Former (58.7) by **12.6 mIoU**——这是个很大的 gap
- Sapiens-2B (81.2 mIoU) 比 Sapiens-0.3B 高 4.5
- Scaling 从 0.3B → 2B 带来 +4.5 mIoU，这是 clear scaling law signal

### 4.4 Depth Estimation

#### 4.4.1 Loss 详解

公式 (1)-(3) 是 Eigen et al. (https://arxiv.org/abs/1406.2283) 提出的 scale-invariant loss 变体：

$$\Delta \mathbf{d} = \log(\mathbf{d}) - \log(\hat{\mathbf{d}})$$

对 GT depth $\mathbf{d}$ 和预测 $\hat{\mathbf{d}}$ 取 log，做差。log-space 是为了处理 depth 的 multiplicative nature（远处的物体 depth 差异大，绝对差不能反映感知差异）。

$$\overline{\Delta \mathbf{d}} = \frac{1}{M} \sum_{i=1}^{M} \Delta \mathbf{d}_i, \quad \overline{(\Delta \mathbf{d})^2} = \frac{1}{M} \sum_{i=1}^{M} (\Delta \mathbf{d}_i)^2$$

$M$ 是图像中 human pixel 数量（只在 human region 上算 loss）。$\overline{\Delta \mathbf{d}}$ 是 log-space residual 的均值，$\overline{(\Delta \mathbf{d})^2}$ 是二阶矩。

$$\mathcal{L}_{\text{depth}} = \sqrt{\overline{(\Delta \mathbf{d})^2} - \frac{1}{2}(\overline{\Delta \mathbf{d}})^2}$$

这个公式有意思。展开看：
$$\mathcal{L}_{\text{depth}}^2 = \overline{(\Delta \mathbf{d})^2} - \frac{1}{2}(\overline{\Delta \mathbf{d}})^2$$

如果是 standard variance，应该是 $\overline{(\Delta \mathbf{d})^2} - (\overline{\Delta \mathbf{d}})^2$。这里系数是 $\frac{1}{2}$，意味着**部分保留了 mean shift 的惩罚**，而不是完全 scale-invariant。这是个折衷：完全 scale-invariant 会让模型忽略绝对 depth scale（对 AR/VR 应用重要），完全 MSE 又对 scale 敏感。

#### 4.4.2 Synthetic data only

Depth 任务只用了 600 个 RenderPeople scans（https://renderpeople.com/）render 出 500K 合成图像，4K 分辨率，random HDRI environment maps。

**Intuition**：合成数据理论上 domain gap 很大（光照、材质、background），但 Sapiens 在 Hi4D（真实 multi-human interaction）上 RMSE 0.114 vs DepthAnything-L 的 0.147，**降低了 22.4% relative**。这说明 human-centric pretraining 提供的 prior 足够强大，能 overcome 合成→真实的 domain gap。这也是 Figure 8 中 $\nabla \text{depth}$ visualization 比 DepthAnything 平滑很多的原因——DepthAnything 在 background 上也给了 noisy depth，而 Sapiens 专注 human region。

### 4.5 Surface Normal Estimation

#### 4.5.1 Loss

公式 (4)：
$$\mathcal{L}_{\text{normal}} = ||\mathbf{n} - \hat{\mathbf{n}}||_1 + (1 - \mathbf{n} \cdot \hat{\mathbf{n}})$$

两项：
- $||\mathbf{n} - \hat{\mathbf{n}}||_1$：L1 距离，对 outlier robust
- $1 - \mathbf{n} \cdot \hat{\mathbf{n}}$：cosine distance（$\mathbf{n} \cdot \hat{\mathbf{n}}$ 是 cosine similarity，因为 normal 是单位向量）

**为什么要 L1 + cosine？** L1 惩罚 component-wise 差异，cosine 惩罚方向差异。单独 L1 对方向不敏感（比如 (0.7, 0.7, 0) vs (0.9, 0.4, 0.1) L1 = 0.5 但 cosine = 0.82），单独 cosine 对 component scale 不敏感。两者结合更 stable。

#### 4.5.2 Table 6 解读

Sapiens-2B 在 THuman2.0 上 mean angular error **11.84°**，ECON（https://arxiv.org/abs/2212.07422）是 25.45°，降低了 53.5% relative。这是个 huge gap。

注意：ECON 用了 4000 scans 训练（super set of Sapiens 的 600），但 Sapiens 只用合成 + human pretraining 就大幅超越——这说明 in-the-wild pretraining 比 in-studio supervision 更 generalizable。

## 5. 关键 Ablation: Pretraining Data Source

Table 7 是这篇论文最核心的 ablation，值得仔细看：

| Pretraining | Pose ↑ | Seg ↑ | Depth ↓ | Normal ↓ |
|-------------|--------|-------|---------|----------|
| Random Init | 30.2 | 40.3 | 0.720 | 35.4 |
| General-100M | 35.7 | 50.1 | 0.351 | 27.5 |
| General-300M | 37.3 | 52.8 | 0.347 | 26.8 |
| Humans-100M | 43.6 | 61.2 | 0.316 | 24.0 |
| **Humans-300M** | **47.0** | **66.5** | **0.288** | **21.8** |

几个观察：

1. **General-300M vs General-100M**：scaling general data 收益递减（pose +1.6, seg +2.7, depth -0.004, normal -0.7）。这跟 DINOv2 的发现一致，general pretraining 很快 saturate。

2. **Humans-100M vs General-300M**：**用 1/3 的数据但限定到 human domain，反而更好**（pose 43.6 vs 37.3, seg 61.2 vs 52.8）。这是 domain curation 的胜利。

3. **Humans-300M vs Humans-100M**：scaling human data 仍然有显著收益（pose +3.4, seg +5.3）。说明 human data 的 scaling 还没 saturate。

**Intuition**：这背后的 logic 是——general pretraining 学到的 feature 对 human 任务来说大部分是 wasted capacity（cat vs dog 的判别对 pose estimation 没用）。限定到 human domain 后，模型的 capacity 都被用来学习 human 的 variability（pose、shape、clothing、occlusion），这些正是 downstream 需要的。

这也呼应了 CLIP 的发现（https://arxiv.org/abs/2103.00020）——data distribution 决定 feature 的 useful domain。但 Sapiens 走的是 narrow-but-deep 路线，而不是 CLIP 的 broad-but-shallow。

## 6. 跟相关工作的对比

### 6.1 vs DINOv2

DINOv2（https://arxiv.org/abs/2304.07193）用 142M general images + iBOT contrastive loss，1B params。Sapiens 用 300M human images + MAE，2B params。两者都 native 高分辨率吗？DINOv2 是 224，Sapiens 是 1024。Table 1 显示 Sapiens-2B 的 8709 GFLOPs 远超 DINOv2 的 291——但 DINOv2 是 general foundation model，Sapiens 是 human-specific，所以 FLOPs 比较意义不大。

### 6.2 vs AIM

AIM（https://arxiv.org/abs/2401.08541）是 autoregressive pretraining，6.5B params，224 resolution。Sapiens 选择 MAE 而非 autoregressive，理由是 MAE 的 single-pass inference 效率更高。这点在 scaling data 时确实重要——AIM 处理 2B 图像需要更多 compute。

### 6.3 vs Depth Anything

Depth Anything（https://arxiv.org/abs/2401.10891）是最近的 monocular depth SOTA，用 60M labeled + unlabeled images。Table 5 显示在 human images 上 Sapiens 大幅超越 Depth Anything。但 fair comparison 要注意——Depth Anything 是 general depth model，Sapiens 是 human-specific，且 Sapiens 用了 high-quality synthetic human depth supervision。两者本质上不在同一个赛道。

## 7. 我对这篇论文的 Critique

### 7.1 优点

1. **Domain curation 的 quantification**：Table 7 把"用 human data 训 human task"这个直觉量化了，非常有说服力。
2. **Native 1K resolution**：这个工程决策很大胆，对细粒度任务（foot、hand、face expression）的收益清晰可见。
3. **Synthetic data 充分性**：depth 和 normal 都用合成数据 finetune 就能超越 SOTA，这对未来数据收集策略有重要启示。
4. **简单 architecture**：没有用 fancy 的 multi-stage、distillation、attention manipulation，就是 plain ViT + MAE + finetune，这说明 pretraining data quality 才是关键。

### 7.2 局限与疑问

1. **数据可获得性**：Humans-300M 是 Meta 内部数据，外部研究者无法复现。这是 foundation model paper 的通病，但对 community 价值打折。
2. **1K resolution 的 cost**：8709 GFLOPs 对 inference 不友好。作者没讨论 latency/throughput tradeoff。
3. **Mask ratio 95% 的 reconstruction**：Figure 3 下半部分看起来 impressive，但缺乏 quantitative 评估。95% mask 下 reconstruction 真的能用于 downstream task 吗？还是只是 plausible-looking 但 feature 质量下降？
4. **Multi-person 限制**：Limitations 部分提到"crowding and severe occlusion are challenging"——这是 multi-person 场景的常见问题，Sapiens 没有专门处理。
5. **Pretraining data scale 是否饱和**：Figure 10 显示 normal 还在涨，但只到 300M。如果推到 1B human images 会怎样？这点没回答。
6. **没有跟 CLIP/DINOv2 feature 在同一 downstream 上比较**：比如把 DINOv2 encoder 也拿来 finetune pose/depth/normal，作为对照。Table 1 只列了 model size 对比，没列 downstream 性能对比。

## 8. 跟你（Karpathy）可能关心的点的联想

### 8.1 Scaling Laws 视角

Sapiens 的 scaling 表现（0.3B → 2B）显示 clear performance scaling，但每个 task 的 scaling rate 不同：
- Pose: 53.4 → 61.1，scale factor 6.67x params → 1.14x AP
- Seg: 76.7 → 81.2，6.67x params → 1.06x mIoU
- Depth (RMSE): 0.148 → 0.114，6.67x params → 0.77x error
- Normal (angular): 13.02 → 11.84，6.67x params → 0.91x error

这个 scaling 跟 Chinchilla 的 compute-optimal scaling 不完全一致——Sapiens 是在 fixed data 上 scale params，而 Chinchilla 强调 data 与 params 共同 scale。如果按 Chinchilla logic，Sapiens-2B 可能 undertrained（300M images × 4096 tokens/image = 1.2T tokens，对应 2B params，token/param ratio = 600，低于 Chinchilla 推荐的 20）。

但 MAE 的 token efficiency 跟 autoregressive LM 不同——每个 masked patch 都是 supervision signal，effective tokens 数远高于 nominal。这个 regime 下 scaling laws 还不清晰。

### 8.2 跟 you 在 Tesla 的工作

Tesla 的 vision system 也用 high-resolution input（https://www.youtube.com/watch?v=j0z4F-xCi6Q 你在 AI Day 的演讲），但 domain 是 driving scene。Sapiens 的方法论——domain curation + high-resolution + MAE pretraining——可以直接迁移到 driving domain：
- 用 1B driving images（含 ego-car, other cars, pedestrians, road）做 MAE pretraining
- 1K resolution 处理远距离 object detection
- Finetune 到 detection, lane segmentation, depth 等

这种 narrow-domain foundation model 可能比 general foundation model（如 SAM, https://arxiv.org/abs/2304.02643）更 efficient。

### 8.3 跟 nanoGPT / minGPT 的联系

从 minGPT（https://github.com/karpathy/minGPT）和 nanoGPT（https://github.com/karpathy/nanoGPT）的角度看，Sapiens 的 architecture 实际上非常 simple——就是 ViT + MAE。复杂度全在 data pipeline 和 pretraining scale 上。这呼应了你常说的一个观点：modern deep learning 的 complexity 在 data 与 scale，不在 architecture。

### 8.4 关于 "pretraining data distribution matters"

Table 7 让我想起你在 CS231n 讲过的 data distribution 重要性（https://cs231n.github.io/）。Sapiens 把它推到 extreme——pretraining data 与 downstream task 同分布时，efficiency 最高。这跟 multi-task learning 的 negative transfer 现象（https://arxiv.org/abs/1706.05137）相关——general pretraining 对 human task 有点像 negative transfer，capacity 被 general statistics 占用。

### 8.5 与 Codec Avatars 的关系

Sapiens 作者都来自 Meta Reality Labs Codec Avatars 团队（https://about.meta.com/realitylabs/codecavatars/）。Codec Avatars 需要 high-fidelity face/body tracking，正好对应 Sapiens 的 308 keypoints + 243 face landmarks + normal/depth。可以预见 Sapiens 会作为 Codec Avatars 2.0/3.0 的 perception backbone。

### 8.6 关于 evaluation 的 concerns

Humans-5K 和 Humans-2K 是 Meta 内部 dataset，benchmark 不公开。这意味着 SOTA comparison 难以独立验证。但 THuman2.0（https://arxiv.org/abs/2105.02060）和 Hi4D（https://arxiv.org/abs/2305.02185）是 public dataset，depth 和 normal 的 evaluation 可信度更高。

## 9. 未来方向猜测

1. **3D extension**：作者在 conclusion 提到"extending Sapiens to 3D"——可能是 multi-view pretraining，或者直接 predict 3D features（如 SMPL parameters, https://smpl.is.tue.mpg.de/）。
2. **Multi-modal**：human video often has audio（speech）和 motion capture。结合 audio + visual pretraining 可能进一步提升 face expression 任务的 performance。
3. **Video pretraining**：当前 Sapiens 是 image-based，但 human motion 是 time-series。Video MAE（https://arxiv.org/abs/2203.12602）+ human video pretraining 是 natural extension。
4. **Generative extension**：当前 Sapiens 是 discriminative。把它作为 diffusion model 的 condition（类似 ControlNet, https://arxiv.org/abs/2302.05543）可能产生高保真 human generation。
5. **On-device**：2B params 太大，edge deployment 需要 distillation 或 pruning。Sapiens-0.3B 已经很强，可能是 on-device 的 sweet spot。

## 10. 总结

Sapiens 的核心 message 我觉得可以浓缩为一句：

> **在 human domain，pretraining data 的 specificity 比 generality 更重要；在 compute budget 固定的情况下，narrow-but-deep pretraining beats broad-but-shallow。**

这个 message 跟当前 foundation model 领域的 "bigger is better, more general is better" 主流叙事形成对比。它暗示了未来 foundation model 可能会分化为：
- General foundation models（GPT-4, Gemini, Claude）做 reasoning
- Domain-specific foundation models（Sapiens for humans, driving-specific for AV, medical-specific for radiology）做 perception

这种分化在 2024-2025 年可能会越来越明显。

**Reference links**:
- Sapiens project page: https://about.meta.com/realitylabs/codecavatars/sapiens
- MAE: https://arxiv.org/abs/2111.06377
- DINOv2: https://arxiv.org/abs/2304.07193
- ViTPose+: https://arxiv.org/abs/2212.04246
- Depth Anything: https://arxiv.org/abs/2401.10891
- PIFuHD: https://arxiv.org/abs/2004.00452
- ECON: https://arxiv.org/abs/2212.07422
- Hi4D: https://arxiv.org/abs/2305.02185
- THuman2.0 / Function4D: https://arxiv.org/abs/2105.02060
- AIM: https://arxiv.org/abs/2401.08541
- SAM: https://arxiv.org/abs/2304.02643
- ControlNet: https://arxiv.org/abs/2302.05543
- VideoMAE: https://arxiv.org/abs/2203.12602
- COCO-WholeBody: https://arxiv.org/abs/2007.11858
- Detectron2: https://github.com/facebookresearch/detectron2
- RenderPeople: https://renderpeople.com/
- Chinchilla: https://arxiv.org/abs/2203.15556
- CLIP: https://arxiv.org/abs/2103.00020
- LLaMA: https://arxiv.org/abs/2302.13971
- nanoGPT: https://github.com/karpathy/nanoGPT
- minGPT: https://github.com/karpathy/minGPT
- CS231n: https://cs231n.github.io/

如果你想 dive deeper 到某个具体方面（比如 normal estimation 的 loss 设计、synthetic data pipeline、或者跟 specific baseline 的 head-to-head 比较），我可以继续展开。
