---
source_pdf: SPATIAL FORCING IMPLICIT SPATIAL REPRESENTATION.pdf
paper_sha256: c7d15ca562f5639ab88c60e7de0de4c34965e3c57710e79febfa249b8bddada5
processed_at: '2026-08-12T09:10:14-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Spatial Forcing 的人话版

## 一句话总结

VLA 模型的眼睛其实是个"色盲"——它看得到颜色和形状,但感觉不到深度和距离。这篇 paper 的做法是:找个 3D 视觉特别厉害的老师(VGGT),让 VLA 学生偷偷看老师的笔记本,把中间层的 feature 对齐过去,自己就开窍了。关键是,inference 的时候老师完全不在场,学生自己就能打。

## 问题的本质

你想想 OpenVLA 这种模型是怎么训练的。它的 vision backbone 基本就是 SigLIP + DINOv2,这俩哥们都是在 2D 图像上 pretrain 的。它们看到一张图,能告诉你"这是个杯子"、"那是个红色的方块",但是你问它"这个杯子离 robot gripper 多远"、"那个方块比这个高几厘米",它两眼一抹黑。

作者做了一个特别 elegant 的实验来证明这件事。他们把 OpenVLA-OFT 的 visual embedding 全部冻住,然后在上面接一个 DPT head,只训练这个 head 去预测 depth map。结果呢?预测出来的 depth 简直是糊的,跟瞎猜差不多。这说明什么?说明 VLA 的 visual token 里压根就没编码 depth 信息。它生成的 action 看起来 ok,很大程度上是在 overfit 训练数据里的 spurious correlation,而不是真正理解了 3D 空间。

这个观察特别重要。因为 VLA 是 auto-regressive 的,action token 是 conditioned on visual token 生成的。visual token 是垃圾,action 能好到哪去?Garbage in garbage out。

## 现有方案为什么不行

想给 VLA 加 3D 能力,之前的人怎么做的?两条路,都有硬伤。

**第一条路:直接喂 depth camera / point cloud。** 比如 GeoVLA、3D-CAVLA、PointVLA。听起来最直接对吧?但工程上全是坑。第一,depth camera 的数据质量很差,Realsense 这种级别的 sensor,透明物体反光物体直接丢失,边缘全是 noise。第二,不同 robot 装的 sensor 位置不同、型号不同、calibration 状态不同,数据 heterogeneity 极大。第三,也是最致命的,Open-X-Embodiment 这种大规模数据集里,很大一部分 episode 根本没录 depth。你想 scale up?没门。

**第二条路:用 depth estimator 从 2D 图估 depth。** 比如 SpatialVLA。这避开了 sensor 问题,但你用的是一个 depth estimator,它本身的误差就很大,尤其对 transparent object、thin structure 这种 challenging case。你把一个 60 分的 estimator 的输出喂给 VLA,VLA 最多也就 60 分天花板。

所以核心矛盾是:explicit 3D input 有各种工程限制,但 VLA 又确实需要 3D understanding。怎么办?

## Spatial Forcing 的思路

作者的 insight 特别巧妙。与其改 input,不如改 representation。与其在 inference 时依赖 3D sensor,不如在 training 时偷偷把 3D 知识"灌"进 VLA 的 visual embedding 里。

具体怎么做?找一个 3D 视觉能力极强的 pretrained model 当"老师"。这里选的是 VGGT(Visual Geometry Grounded Transformer),这玩意能从一组 2D 图像直接预测 camera pose、point map、depth map、3D point track,是 3D 视觉 foundation model 里的天花板级别。VGGT 论文在这:https://arxiv.org/abs/2503.11651

然后,在 VLA 训练的过程中,除了正常的 action loss 之外,额外加一个 alignment loss:把 VLA 中间某一层(实验发现第 24 层最好)的 visual embedding,经过一个简单的 MLP 投影后,跟 VGGT 输出的 spatial representation 做 cosine similarity 对齐。

就这么简单。没有改架构,没有改 input,没有改 inference 流程。只是在 training 时多加了一个 loss term。

## 为什么这个能 work

这里有几个层次的 intuition,我一层层剥开。

### Intuition 1: Representation 是 action 的天花板

Auto-regressive VLA 里,visual token 是 action token 的"输入材料"。Action token 是基于 visual token 生成的。如果 visual token 里没有 depth 信息,你指望 action 怎么精准地控制 gripper 在 z 轴上的位置?

这就像你让一个厨师做菜,但食材库里根本没盐。厨师再厉害,做出来的菜也没盐味。你得先让食材库里 有盐,厨师才能用。

SF 做的事情就是:往 visual token 这个"食材库"里偷偷塞进 3D 信息这个"盐"。Chef(action head)不需要知道盐是从哪来的,它只需要能尝到味道就行。

### Intuition 2: 中间层对齐比输入层对齐更优雅

你可能会问:为什么不直接把 VGGT 的 representation 当 input 喂进去?

因为 input 层的对齐有几个问题。第一,input 改了,inference 时就需要一直带着 VGGT,计算开销大。第二,VGGT 的 representation 和 VLA 原本的 visual encoder(SigLIP+DINOv2)的 representation space 不一样,直接 concat 或 replace 会破坏 VLA 原本的 semantic understanding 能力。第三,你改了 input,整个模型的 normalization、scale 都得重新调。

中间层对齐就优雅多了。VLA 的 visual encoder 照常工作,产出自己的 embedding。只是在中间某一层,我们"拽"一下这个 embedding,让它往 VGGT 的 representation 靠拢。这个"拽"的力就是 alignment loss。拽完之后,后续的 layer 继续处理这个被"拽过"的 embedding,最终生成的 action token 自然就带有 3D 信息。

这跟 REPA(Yu et al. 2024)在 diffusion model 里的思路一模一样:diffusion transformer 的中间层 hidden state 对齐到 DINOv2 的 representation,生成质量大幅提升。REPA 论文:https://arxiv.org/abs/2410.06940

### Intuition 3: "Relatively deep but not deepest" 是个 universal pattern

Ablation 显示,对齐第 24 层(总共 32 层)效果最好。为什么不是第 1 层?为什么不是第 32 层?

第 1 层太浅。浅层 feature 还停留在 low-level 的 edge、texture 级别,还没形成 high-level 的 spatial representation。你在这一层对齐,相当于逼一个小学生学大学的 spatial geometry,他还没那个认知基础。

第 32 层(最后一层)太深。随着 layer 加深,visual 和 language modality 会逐渐 converge 到一个 modality-agnostic 的空间(Huang et al. 2024 的 Modality Integration Rate 理论)。最后几层已经丢失了很多 vision-specific 的信息,变成了一种"通用语义"的 representation。在这种 representation 上对齐 spatial 信息,就像在一张已经被擦干净的纸上重新写字,写不进去。

第 24 层是 sweet spot。这一层既已经形成了足够 high-level 的 spatial understanding(比浅层强),又还保留了足够的 vision-specific 信息(比深层强)。对齐这一层,反向传播会让浅层也趋向 spatial representation,同时不会破坏深层的 action generation 能力。

这个 pattern 在 REPA、3DRS、Geometry Forcing 里都出现了。"Relatively deep but not deepest"可能是一个 universal principle for representation supervision。

### Intuition 4: Structure transfer without identity collapse

t-SNE 可视化显示了一个特别有意思的现象。SF 训练后,VLA 的 visual feature 在 t-SNE 空间里的分布形状跟 VGGT 的 target 几乎一模一样,但是 cluster center 是分开的。

这说明什么?说明 SF 不是在做简单的"复制粘贴"。它让 VLA 学到了 VGGT representation 的 relational structure——也就是 feature 之间的相对关系、manifold 的拓扑结构。但是 VLA 保留了自己的 cluster center,也就是说它没有丢失自己 modality 独特的信息。

这就像你学画画,你临摹大师的构图、笔触、用色(relational structure),但你画的还是你自己的画(cluster center 独立)。你学的是 structure,不是 copy。

如果 alignment 太强($\alpha = 12.5$),VLA 会完全 collapse 到 VGGT 的 representation space,丢失自己的 visual modality 信息,action 性能反而下降。所以 $\alpha = 0.5$ 这个 sweet spot 是在"学 structure"和"保 identity"之间的完美平衡。

## Positional Embedding 的重要性

这个细节特别容易被忽略,但在 LIBERO-Long 上差了 10 个点。

VLA 是 causal attention 的 auto-regressive 结构。这意味着 token 的顺序本身就是信息。如果只对齐 visual token 的 content,而不管它的 position,模型就不知道"哪个 visual token 对应场景的哪个位置"。

举个具体例子。假设场景里有两个物体,左边的杯子离 gripper 10cm,右边的方块离 gripper 30cm。如果 visual token 丢了 position 信息,模型可能知道"有个东西在 10cm,有个东西在 30cm",但不知道哪个是杯子哪个是方块。Action 就会错乱。

加 positional embedding $E$ 到 target representation 里,就是告诉 VLA:"这个 spatial feature 对应这个 position"。这样 alignment 不仅对齐了 content,还对齐了 position。在长 horizon 任务里,这种 position-aware 的 alignment 尤其关键,因为 robot 需要记住一系列 spatial state 的变化。

## VGGT 为什么是好老师

Ablation 比较了三个 target representation:SigLIP、DINOv2、VGGT。

SigLIP 是 image-text alignment 训练的,强项是 semantic understanding("这是个杯子"),spatial 能力一般。用它当 target,SR 从 92.7% 涨到 94.0%。

DINOv2 是 self-supervised 训练的,spatial grounding 比 SigLIP 强一些(因为 contrastive learning 天然保留了 spatial 结构)。用它当 target,SR 涨到 94.1%。

VGGT 是在 2D-3D paired data 上训练的,天生就是为 3D 几何设计的。用它当 target,SR 涨到 96.9%。

这说明什么?说明 alignment paradigm 本身是 general 的(任何 strong representation 都能带来提升),但 target 的质量决定了天花板。VGGT 之所以最好,是因为它最直接地编码了 3D spatial information。这也反向验证了论文的核心 hypothesis:VLA 缺的就是 3D understanding,补上它就能大幅提升。

## Training Efficiency 为什么能快 3.8x

这个结果我觉得是最 actionable 的。

Baseline 在 150K iterations 达到 ~92.7%。SF 在 20K iterations 就超过这个数字了。

为什么?因为正常训练 VLA 时,模型需要从 robotic data 里同时学两件事:(1) visual spatial understanding,(2) action generation。这两件事纠缠在一起,学习效率低。

SF 把 (1) 这件事 "外包" 给了 VGGT。VGGT 已经在海量 2D-3D paired data 上训练过,它的 spatial representation 是现成的、高质量的。SF 通过 alignment loss,直接把 VLA 的 visual embedding "拉" 到这个已经富含 spatial 信息的 manifold 上。VLA 不需要从头学 spatial understanding,只需要在 robotic data 上学 task-specific 的 action generation。

这就像你学开车。如果你得先从零开始学"什么是距离"、"什么是速度",那学习周期会很长。但如果有个 simulator 已经帮你建立了 distance 和 speed 的 intuition,你上车只需要学"方向盘怎么打"、"油门怎么踩",自然快很多。

VGGT 就是那个 simulator。它帮 VLA 建立了 spatial 的 prior,VLA 在这个 prior 基础上学 action,效率自然高。

## Data Efficiency 为什么能好 5.9x

5% 的数据,SF 达到 75.8% SR,baseline 在 5% 数据下大概只有 50% 左右。

这个现象的 intuition 跟 training efficiency 是同一个根源。Robotic data 稀缺且昂贵(每个 demo 都要人遥操作采集)。如果 VLA 需要从 robotic data 里同时学 spatial understanding 和 action generation,那一点点数据根本不够学。

但有了 SF,VGGT 的 spatial representation 提供了一个强大的 prior。VLA 只需要很少的 robotic data 就能学到 "在这个 spatial manifold 上,怎么做 task-specific 的 action"。Spatial understanding 不需要 data 来学(已经从 VGGT 那里 "继承" 了),data 只用来学 action mapping。

这跟 pre-training 的哲学一脉相承。为什么 BERT pre-training 能让下游任务 data efficient?因为 language 的 statistical structure 已经被学到了,下游任务只需要学 task-specific 的 mapping。SF 做的事情就是 spatial representation 的 "pre-training",只不过这个 pre-training 是通过 alignment loss 间接 transfer 过来的。

## Real-World 实验的 insight

Real-world 实验设计得特别好,每个 task 都考察 spatial understanding 的不同维度。

**Stack glass cups**:透明杯子会反射环境光,颜色变化巨大。一个没有 spatial understanding 的模型会被 color 搞晕,因为它 overfit 到了 color 这个 spurious correlation。但 SF 学到的是 underlying 的 spatial structure(杯子的 3D 位置、gripper 的相对距离),color 变化不影响 spatial structure。所以 SF 比 baseline 高 47.5%。

**Place green block with height variation**:这个直接考察 depth estimation。不同高度放置,SF 达到 85% SR,说明 visual embedding 里确实编码了 height 信息。

**Bimanual lift pot**:这个考察的是 spatial horizontal balance。两只 arm 要协调,保持 pot 不倾斜。这需要对 pot 的 3D 姿态有精确感知。SF 能做到,说明 alignment 带来的 spatial information 不只是 per-pixel depth,还包括 object-level 的 3D structure。

这些 real-world 结果共同证明了一件事:SF 学到的 spatial understanding 是 robust 的、generalizable 的,不是在 simulation 里 overfit 的 artifact。

## 跟其他 paradigm 的关系

### 跟 Knowledge Distillation 的关系

SF 本质上是一种 feature-level knowledge distillation。VGGT 是 teacher,VLA visual encoder 是 student。但跟传统 KD 有几个区别:

1. 传统 KD 通常对齐 final output 或者所有 layer。SF 只对齐一个中间层(layer 24)。
2. 传统 KD 的 student 只做 distillation。SF 的 student 同时在做 end-to-end 的 action prediction,distillation 只是 auxiliary。
3. 传统 KD 的 teacher 在推理时可能还需要(比如 logit distillation)。SF 的 teacher 在推理时完全不需要,zero overhead。

这更像 FitNets(Romero et al. 2014)的 hint-based training,但应用场景和设计完全不同。FitNets:https://arxiv.org/abs/1412.6550

### 跟 JEPA 的哲学共鸣

Yann LeCun 的 JEPA 哲学是:不要做 pixel-level reconstruction(保留太多 redundant detail),而要做 representation-level 的 prediction(学 high-level abstraction)。https://openreview.net/pdf?id=bN0oYDJN7P

SF 跟这个哲学共鸣。它没有让 VLA 重建 depth map(reconstruction-based supervision),而是对齐到 VGGT 的 latent representation(alignment-based supervision)。Latent representation 是 VGGT 已经过滤掉 redundant detail 之后的高质量 abstraction,用它当 target 比 用 depth map 当 target 更高效。

这也解释了为什么 reconstruction-based 的方法(比如 ReconVLA)效果不如 alignment-based 的 SF。Reconstruction 会逼模型记住所有 detail,包括那些对 action 无关的 detail(比如背景纹理),浪费 model capacity。

### 跟 RLHF 的结构相似性

结构上,SF 跟 RLHF 有点像。RLHF 里,你有一个 reward model 来提供 supervision signal,训练 policy。SF 里,你有一个 3D foundation model(VGGT)来提供 supervision signal,训练 VLA 的 visual representation。

区别在于,RLHF 的 reward model 是在 human preference data 上训练的,SF 的 teacher 是在 2D-3D paired data 上训练的。RLHF 的 supervision 是 scalar reward,SF 的 supervision 是 dense representation。但 underlying 的 pattern 是一样的:用另一个 model 的 knowledge 来 regularize 训练。

## 我的 Takeaway

读完这篇 paper,我脑子里留下的几个核心 picture:

**Picture 1: Bottleneck 在 representation,不在 architecture。** VLA 社区花了大量精力设计更好的 action head(diffusion、flow matching、action chunking),但 visual representation 这个 bottleneck 一直被忽视。SF 证明,只要 representation 质量上去了,action 精度自然就上去了。这跟你经常强调的 "找到真正的 bottleneck" 思维一致。

**Picture 2: Pretrained model 是 "免费的老师"。** VGGT 已经花了大量算力训练好,它的 spatial representation 是现成的。SF 本质上是 "transfer learning 的极致简化版"——不需要 fine-tune teacher,不需要 distill output,只需要在 student 的中间层加一个 cosine similarity loss。这种 elegance 在工程上特别 actionable。

**Picture 3: Implicit > Explicit 当 explicit 有工程包袱时。** Explicit 3D input 听起来更直观,但 sensor noise、hardware heterogeneity、data scarcity 这些工程问题让它难以 scale。Implicit alignment 绕开了所有这些问题,同时达到了同样甚至更好的效果。这是 "less is more" 的典范。

**Picture 4: Structure transfer without identity collapse 是 representation supervision 的核心美学。** t-SNE 实验特别美。VLA 学到了 VGGT 的 relational structure,但保留了自己的 cluster center。这种 "学 structure 不学 identity" 的平衡,可能是所有 representation alignment 方法能 work 的深层原因。

**Picture 5: "Relatively deep but not deepest" 可能是 universal principle。** 这个 pattern 在太多地方出现了。REPA、3DRS、SF 都验证了这一点。如果你在设计 representation supervision,layer selection 是一个关键 hyperparameter,而且大概率不是最后一层。

这篇 paper 的影响可能会超出 VLA 领域。任何 "student model 的 intermediate representation 需要被增强" 的场景,都可以用这个 paradigm。Video generation、3D generation、embodied navigation、autonomous driving,都有可能借鉴 SF 的思路。当你有一个 strong pretrained teacher,而你的 student 需要在某个 modality 上补课时,SF 提供了一个 elegant 的 recipe。

Project page: https://spatial-forcing.github.io/
VGGT: https://arxiv.org/abs/2503.11651
REPA: https://arxiv.org/abs/2410.06940
OpenVLA-OFT: https://arxiv.org/abs/2502.19645

---

# Spatial Forcing: Implicit Spatial Representation Alignment for VLA 深度解析

## 1. 核心问题与 Motivation

这篇 paper 处理一个 VLA (Vision-Language-Action) 模型领域的关键痛点:当前的 VLA backbone 比如 PaliGemma、Prismatic VLM 都是只在 2D 图像上 pretrain 的，它们对 3D 物理世界的 spatial awareness 严重不足。作者用一个叫做 **depth probing** 的实验直接证实了这一点:把一个主流 VLA (OpenVLA-OFT) 的 visual embeddings 冻住，只训练一个 DPT head 去预测 depth map，结果显示这些 embedding 根本 reconstruct 不出有意义的 spatial structure。这说明 VLA 模型虽然能输出 action，但它的 visual tokens 实际上并没有编码足够的 3D 几何信息。

这里有一个很关键的 insight:既然 action tokens 是 auto-regressive 地 conditioned on visual tokens 生成的，那么如果 visual tokens 本身富含 spatial information，action 的精度自然会提升。这个链条非常清晰:

$$\pmb{x}_t^{\mathcal{A}} \sim p_\theta\left(\pmb{x}_t^{\mathcal{A}} \mid \{\pmb{x}_i^{\mathcal{V}}\}_{i=1}^{N}, \{\pmb{x}_j^{\mathcal{L}}\}_{j=1}^{M}, \pmb{x}_{<t}^{\mathcal{A}}\right)$$

其中 $\pmb{x}_t^{\mathcal{A}}$ 是第 $t$ 个 action token，$\{\pmb{x}_i^{\mathcal{V}}\}_{i=1}^{N}$ 是 $N$ 个 visual tokens，$\{\pmb{x}_j^{\mathcal{L}}\}_{j=1}^{M}$ 是 $M$ 个 linguistic tokens，$\pmb{x}_{<t}^{\mathcal{A}}$ 是之前生成的 action tokens。这个公式告诉我们 action 的质量被 visual token 的质量 bottleneck 了。

参考 VGGT 原始论文: https://vgg-t.github.io/

## 2. 现有方案的问题

作者对比了三类 paradigm:

**(a) Explicit 3D sensor input**:比如 GeoVLA、3D-CAVLA、PointVLA，直接用 depth camera 或 LiDAR 获取 depth map / point cloud 作为额外输入。问题有三个:(1) sensor 噪声大、quality 差;(2) 不同 robot 的 sensor 类型、位置、calibration 状态差异大，引入 heterogeneity;(3) 大规模数据集比如 Open-X-Embodiment 中相当一部分 episode 根本没有 depth 信息，scale 不上去。

**(b) Depth estimation from 2D**:比如 SpatialVLA、Evo-0，用 depth estimator 从 2D 图像估 depth 再喂进去。问题是性能被 depth estimator 本身的天花板限制，sub-optimal。

**(c) Spatial Forcing (本文)**:不修改输入，而是用 representation supervision 在中间层强制对齐到一个 3D foundation model 的 representation。这个思路非常优雅——inference 的时候跟普通 VLA 完全一样，没有额外计算开销。

这个 paradigm 跟 REPA (Representation Alignment for Generation, Yu et al. 2024) 在 diffusion model 领域的思路高度类似，REPA 论文: https://arxiv.org/abs/2410.06940

## 3. 方法细节: Spatial Forcing

### 3.1 监督信号来源: VGGT

作者选择 **VGGT (Visual Geometry Grounded Transformer)** 作为 3D supervision 的来源。VGGT 是一个 feed-forward 模型，输入一组 2D 图像，直接输出 camera parameters、point maps、depth maps、3D point tracks 等多种 3D 属性。它的核心设计是 **Alternating-Attention mechanism**:交替进行 frame-wise self-attention (关注单帧内部) 和 global self-attention (跨帧全局聚合)。这样每帧的 latent representation 既包含 local 细节又包含 global context。

作者的关键论点是:VGGT transformer backbone 输出的 **latent representation** 本身就编码了丰富的 spatial information，可以直接当作 supervision target，而不需要用 VGGT 的 prediction head 输出的显式 depth/point map。这一点很重要——用 latent 比用显式 3D 输出更灵活，因为 latent 是 dense 的、continuous 的、information-rich 的。

VGGT 论文: https://arxiv.org/abs/2503.11651

### 3.2 Alignment Loss

技术实现上，对于 VLA 中的 visual token $\pmb{x}_i^{\mathcal{V}}$，先用 Batch Normalization $\Gamma$ 归一化，再过两层 MLP 投影到与 VGGT representation 兼容的维度，然后跟 VGGT 输出的 spatial representation $f_i^{3D}(I)$ (加上 positional embedding $E$) 做 cosine similarity 最大化:

$$\mathcal{L}_{\mathrm{align}} = -\frac{1}{N} \sum_{i=1}^{N} S\left[\mathrm{MLP} \cdot \Gamma(\pmb{x}_i^{\mathcal{V}}), f_i^{3D}(I) + E\right]$$

变量含义:
- $N$: visual token 的总数 (per-pixel token 数量)
- $S[\cdot, \cdot]$: cosine similarity
- $\pmb{x}_i^{\mathcal{V}}$: VLA 第 $i$ 个 visual token
- $\Gamma$: Batch Normalization
- $\mathrm{MLP}$: 两层 MLP，做维度对齐
- $f_i^{3D}(I)$: VGGT 对应 pixel location 的 spatial representation
- $E$: positional embedding，保证 token 的位置顺序信息不丢，这在 auto-regressive 过程中至关重要

为什么加 positional embedding $E$ 这么重要?因为 VLA 是 causal attention 的 auto-regressive 结构，token 的相对位置本身就是信息。如果只对齐 content 而丢掉 position，模型在后续生成 action 时会丢失 "哪个 visual token 对应哪个空间位置" 的对应关系。Ablation (Table 2) 显示:不加 PE 的 VGGT 在 LIBERO-Long 上只有 84.4% SR，加了 PE 暴涨到 94.2%，差了整整 10 个点。

### 3.3 在哪一层对齐

这是一个非常 fine-grained 的设计选择。VLM backbone 有 32 层 causal attention layer。Ablation 结果:

| Layer | Average SR (%) |
|-------|----------------|
| 1     | 94.6           |
| 8     | 95.7           |
| 16    | 93.8           |
| 24    | **96.9**       |
| 32    | 94.8           |

第 24 层最优。作者的解释有两层:
1. **Supervising deep features implicitly enforces shallow features to align** — 在深层施加约束，反向传播会让浅层也趋向 spatial representation，从而在 global level 获得更好的 spatial understanding。
2. **Last layers lose vision-specific features** — 随着 layer 加深，visual 和 language modality 会 converge 到一个 modality-agnostic space (Huang et al. 2024, Modality Integration Rate)。最后几层已经 vision-specific 信息流失，不适合再做 vision representation supervision。

这个发现跟 REPA 在 diffusion transformer 里的发现一致:不是越深越好，而是 "relatively deep but not the deepest" 是 sweet spot。

参考 MLLMs need 3D-aware representation supervision: https://arxiv.org/abs/2506.01946

### 3.4 总 Loss

$$\mathcal{L}_{\mathrm{SF}} = \mathcal{L}_{\mathrm{action}} + \alpha \mathcal{L}_{\mathrm{align}}$$

其中 $\alpha$ 是权重因子。Appendix Table 3 显示 $\alpha = 0.5$ 最优:

| $\alpha$ | 0   | 0.02 | 0.1  | 0.5  | 2.5  | 12.5 |
|----------|-----|------|------|------|------|------|
| SR (%)   | 73.2| 92.2 | 92.8 | 93.6 | 86.6 | 81.2 |

$\alpha$ 太大 ($\alpha = 12.5$) 会让 visual modality 失稳，干扰原始 action prediction。$\alpha = 0$ 就是没 SF 的 baseline。这个 weight sensitivity 分析告诉我们 alignment loss 是 auxiliary 的，不能喧宾夺主。

## 4. 实验结果分析

### 4.1 LIBERO Benchmark

LIBERO 有四个 task suite: Spatial (空间布局泛化)、Object (物体泛化)、Goal (目标泛化)、Long (长 horizon)。

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| OpenVLA-OFT (baseline) | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| GeoVLA (explicit 3D) | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| 3D-CAVLA (explicit 3D) | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| **Spatial Forcing** | **99.4** | **99.6** | **98.8** | 96.0 | **98.5** |

SF 在不使用任何 explicit 3D sensor input 的情况下，超越了所有 explicit 3D VLA。这一点意义重大——它证明了 implicit representation alignment 可以替代 explicit 3D input。LIBERO-Long 上略低于 3D-CAVLA (96.0 vs 96.1)，基本持平。

### 4.2 RoboTwin Benchmark

RoboTwin 是 bimanual benchmark，有 easy (in-domain) 和 hard (domain randomization: clutter、background texture、lighting、tabletop height) 两个 setting。Hard setting 的提升尤其明显，说明 SF 让模型 focus 在 object 的 relative spatial relationship 上，而不是 overfit 到 background/lighting 这种 shortcut correlation。这跟 self-supervised learning 里 "过滤掉 spurious correlation" 的哲学一致 (LeCun JEPA 论文: https://openreview.net/pdf?id=bN0oYDJN7P)。

### 4.3 Training Efficiency

这是我个人觉得最 impressive 的结果。Table 2 显示:

| Training Iterations | Average SR (%) |
|---------------------|----------------|
| 2K                  | 72.7           |
| 5K                  | 87.5           |
| 20K                 | 93.7           |
| 50K                 | 96.5           |
| 150K                | 96.9           |

而 baseline 在 150K iterations 才达到 ~92.7%。也就是说 SF 用 20K iterations 就超过了 baseline 150K iterations 的效果，加速 **3.8×**。

这个现象的 intuition 是:VGGT 的 representation 提供了一个 "good initialization" in representation space。VLA 不需要从 robotic data 中从头学习 spatial structure，而是被直接 "拉" 到一个已经富含 spatial information 的 manifold 上。robotic data 只需要在这个 manifold 上做 task-specific 的 fine-tuning。这跟知识蒸馏中 "soft target 提供暗知识" 的机制有异曲同工之妙。

### 4.4 Data Efficiency

| Data | SF SR (%) | 
|------|-----------|
| 1%   | 42.3      |
| 5%   | 75.8      |
| 33%  | (Fig 5b)  |
| 100% | 96.9      |

5% data 就能达到 75.8% SR，baseline 在 5% data 下大概只有 50% 左右。这相当于 **5.9× data efficiency**。对于 real-world robotic data 稀缺的场景，这个价值巨大。

### 4.5 Target Representation 的 Ablation

Table 2 还比较了不同 target:

| Target | Average SR (%) |
|--------|----------------|
| (none) | 92.7           |
| SigLIP | 94.0           |
| DINOv2 | 94.1           |
| VGGT w/o PE | 94.7     |
| VGGT (full) | 96.9     |

SigLIP 和 DINOv2 都能带来提升，说明 "representation alignment" 本身是一个 general paradigm。但 VGGT 最强，因为它 trained on 2D-3D paired data，spatial perception 能力最强。这进一步验证了 "compensating for the lack of 3D understanding is crucial"。

## 5. t-SNE 可视化的深层含义

Appendix B 提供了一个很有深度的分析。SF alignment 后，VLA feature 的 t-SNE 分布形状跟 target 几乎一样，但 cluster center 保持独立。这个观察有两层含义:

1. **Distribution shape 相似**:说明 SF 不只是做了一个简单的 linear mapping，而是 force VLA 学习了 target spatial representation 的 underlying manifold。VLA 学到的是 "feature 之间的 relational structure"，而不只是 "feature 的绝对位置"。

2. **Cluster center 独立**:说明没有发生 representational collapse。如果 alignment 只是让 VLA feature 复制 target feature，两个 cluster 会完全 overlap。但 SF 保留了 VLA 自己 modality 的独特信息，只是借用了 target 的 relational geometry。

这一点让我想到 mode-seeking vs mode-covering GAN loss 的区别。SF 像是在做一个 "structure-preserving" 的 alignment，既学到 target 的结构，又不丢失 source 的 identity。这种平衡非常微妙，也是为什么 $\alpha$ 不能太大的原因。

参考 t-SNE 原始论文: https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

## 6. Real-World 实验

Real-world 实验设计得非常 comprehensive，覆盖了 spatial capability 的多个 axis:

1. **Stack glass cups (light variation)**:透明杯子反射不同光照颜色，高度 deceptive。SF 比 baseline 高 47.5%，因为它捕获 underlying spatial relationship 而非 overfit spurious correlation。
2. **Grasp right-side vegetable (target object variation)**:不同物体需要不同 gripper pose 和 clamping width，考验 3D appearance understanding。
3. **Place green block (height variation)**:考验 spatial height estimation，SF 达到 85% SR。
4. **Lift pot (bimanual, new embodiment)**:考验 spatial horizontal balance awareness，防止 pot 倾斜。

只用 40 demos (single-arm) / 20 demos (bimanual) 训练，data efficiency 极高。

## 7. 与 Related Work 的关系

### 7.1 Representation Supervision

这个方向有两支:

**Reconstruction-based**: L-DAE、Genhancer、ROSS、ReconVLA。用 denoising architecture 重建输入图像来监督 visual embedding。问题是 reconstruction 会保留 redundant details，LeCun 在 JEPA 里反复论证过 generative reconstruction 不适合学 high-level representation。

**Alignment-based**: REPA、3DRS、Geometry Forcing。直接把中间 hidden state 对齐到 pretrained encoder。SF 属于这一支。

SF 跟 REPA 的核心区别:REPA 对齐的是 diffusion transformer 的 hidden state 到 DINOv2/VGGT，目标是提升生成质量。SF 对齐的是 VLA 的 visual token 到 VGGT，目标是提升 action precision。但 underlying philosophy 完全一致:用 strong pretrained representation 作为 "teacher" 来 regularize 学习过程。

Geometry Forcing (Wu et al. 2025): https://arxiv.org/abs/2507.07982

### 7.2 跟 Knowledge Distillation 的关系

从更高层面看，SF 本质上是一种 **feature-level knowledge distillation**，其中 VGGT 是 teacher，VLA 的 visual encoder 是 student。但区别在于:
- 传统 KD 通常对齐 final output 或所有 layer
- SF 只对齐某一层 (layer 24)，且 student 的 primary task (action generation) 仍然是 end-to-end 训练的
- VGGT 的输出不直接参与 inference，只是 training 时的 supervision signal

这让我想到 FitNets (Romero et al. 2014) 的 hint-based training，思路类似但应用场景完全不同。FitNets 论文: https://arxiv.org/abs/1412.6550

## 8. 我的 Intuition Building

读完这篇 paper，我提炼出几个关键 intuition:

**Intuition 1: Representation 是 Action 的 bottleneck**。在 auto-regressive VLA 中，action tokens 完全 conditioned on visual tokens。如果 visual tokens 是 "spatial-blind" 的，action 不可能 spatial-precise。这跟 LLM 中 "garbage in garbage out" 是同一个道理。

**Intuition 2: Implicit guidance > Explicit input**。直接给 depth map 作为 input 听起来直观，但会引入 sensor noise、heterogeneity、data scarcity 等一堆工程问题。而 implicit alignment 在 representation 层面注入 spatial knowledge，inference 时零开销，且能 leverage pretrained 3D foundation model 的全部 power。

**Intuition 3: 中间层 supervision 是 sweet spot**。太浅的层还没形成 high-level representation，太深的层已经丢失 modality-specific 信息。"Relatively deep but not deepest" 这个规律在 REPA、3DRS、SF 中反复出现，可能是一个 universal principle。

**Intuition 4: Pretrained representation 是 "cheap teacher"**。VGGT 已经在海量 2D-3D paired data 上训练过，它的 representation 是 "free" 的 spatial knowledge source。SF 本质上是把这种 knowledge "transfer" 到 VLA 中，而且 transfer 的 cost 极低 (只是一个 cosine similarity loss)。

**Intuition 5: Structure-preserving alignment 避免 collapse**。t-SNE 实验显示 SF 学到了 target 的 relational structure 但保留了自己的 cluster center。这种 "structure transfer without identity collapse" 是 representation supervision 能 work 的关键。如果 alignment 太强 ($\alpha = 12.5$)，VLA 丢失自己的 modality 信息，action 性能反而下降。

## 9. 潜在的局限与未来方向

虽然 paper 没有明确讨论，但我推测几个 potential limitation:

1. **VGGT 的 multi-view assumption**:VGGT 需要 multiple views 来做 3D reconstruction。如果 robot 只有 single view (比如只有 wrist camera)，VGGT 的 representation quality 可能下降。Paper 中用的是 primary + wrist cameras，算 multi-view。

2. **Domain gap**:VGGT 是在 general images 上训练的，它的 spatial representation 可能不完全 match robotic manipulation 的特定需求 (比如对 gripper 末端的高精度 spatial 估计)。

3. **Layer selection 的 generalization**:Layer 24 是在 OpenVLA-OFT (32 层) 上调出来的。换一个不同深度的 backbone，最优 layer 可能不同，需要重新 search。

未来方向可能是:
- 把 SF 应用到更多 VLA backbone (π0、RDT-1B、CogACT)
- 探索其他 3D foundation model 作为 teacher (比如 DUSt3R、MASt3R)
- 结合 explicit 3D input 和 implicit alignment，看是否能进一步提升
- 把这个 idea 扩展到 video VLA，对齐 temporal representation

DUSt3R: https://dust3r.europe.naverlabs.com/
MASt3R: https://arxiv.org/abs/2406.09756

## 10. 总结

Spatial Forcing 是一个 elegant、simple、effective 的方法。它的核心贡献不是发明新架构，而是发现了一个被忽视的问题 (VLA visual embedding 缺乏 spatial information) 并提供了一个干净的解决方案 (中间层 representation alignment 到 VGGT)。3.8× training speedup、5.9× data efficiency、SOTA performance on LIBERO 和 RoboTwin，这些数字背后是一个深刻的 insight:在 auto-regressive 架构中，提升 intermediate representation 的质量是提升 output 质量的最直接路径。

这个工作让我联想到你在 Neural Nets: Zero to Hero 系列里讲的一个观点:deep learning 的很多突破来自于 "找到正确的 supervision signal"。SF 找到的 supervision signal 是 VGGT 的 spatial representation，这个 signal 既 rich 又 free (pretrained)，而且完美契合 VLA 的 bottleneck (visual token → action token)。这种 "用 strong pretrained model 的 representation 来 regularize 另一个 model 的中间层" 的 paradigm，我预见会在更多领域 (video generation、3D generation、embodied AI) 反复出现。

Project page: https://spatial-forcing.github.io/
OpenVLA-OFT: https://arxiv.org/abs/2502.19645
π0: https://arxiv.org/abs/2410.24164
