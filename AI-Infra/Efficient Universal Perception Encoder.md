---
source_pdf: Efficient Universal Perception Encoder.pdf
paper_sha256: bc55f1066344f0eee91b9230ddb3a08c2879438a7ba3d1dec675c8cd2fa7b2e2
processed_at: '2026-08-04T01:59:29-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用大白话来讲，这篇 paper 讲了一个“博导教小学生”的故事。

我们希望手机、VR头显这些 edge devices 上能跑一个全能的视觉模型。这个模型必须小（<100M参数），同时什么都要会：认图、分割、看图说话。以往的做法是找几个各个领域的绝顶高手（比如懂图文匹配的 PEcore/SigLIP，懂空间几何的 DINOv3），直接把他们的知识硬塞给这个小模型。结果小模型直接学崩了。原因很简单，小模型的“脑容量”不够，无法同时调和这些完全不同的几何空间和语义空间，梯度互相打架。

EUPE 的解法非常符合人类直觉：**找个“班主任”做中间人**。

你不能让5个不同领域的博导同时给一个小学生上课，小学生会疯。你得先让这些博导把知识教给一个“班主任”——一个超大的 Proxy Teacher（1.9B参数）。这个班主任脑容量足够大，能把不同博导的知识内化、融合成一个统一的、不矛盾的知识体系。然后再让小模型只跟着这个班主任学。这在信息论上等于先把多源异构的 manifold 拉平，再让小模型去拟合这个平滑的 manifold。

具体执行上，这套 pipeline 分为三个 stage：

**Stage 1: Scaling Up（博导教班主任）**
拿几个大牛模型（PEcore, PElang, DINOv3）当 Teacher，教出一个 1.9B 的 Proxy Teacher。这个大模型有足够的高维空间去吸收并融合不同的 inductive biases。

**Stage 2: Scaling Down（班主任教小学生）**
让小模型跟着 Proxy Teacher 学。保持图片分辨率固定在 $256 \times 256$，踏踏实实学很久（390k iterations）。在固定分辨率下，计算效率高，小模型可以充分吸收班主任那套统一过的知识。

**Stage 3: Multi-resolution 微调（让小学生见世面）**
现实世界里的图片有各种尺寸。如果你只学过 $256 \times 256$，碰到大图 ViT 的 positional encoding 就会错位，dense prediction 就会崩。所以最后阶段给图片做个金字塔（256, 384, 512），Teacher 和 Student 各自随机挑一个尺寸学习，让小模型学会 scale-invariant，修复空间感知。

为了 build your intuition，我们看看具体的数学公式里藏了什么猫腻：

公式 (1) 和 (2) 就是 Student 和 Teacher 的前向传播：
$$ y_S^c, y_S^p = S(x_S; \theta) $$
$$ y_{T_i}^c, y_{T_i}^p = T_i(x_{T_i}; \phi_i) $$
*   $S$ 是 Student，$\theta$ 是它的参数。$T_i$ 是第 $i$ 个 Teacher，$\phi_i$ 是它的 frozen 参数。
*   $y^c$ 是 class token（全局语义特征），上标 $c$ 代表 class。它存在于空间 $\mathbb{R}^d$ 中，$d$ 是 feature dimension。
*   $y^p$ 是 patch tokens（空间细节特征），上标 $p$ 代表 patch。它存在于 $\mathbb{R}^{N \times d}$ 中，$N$ 是 patch 的数量。

公式 (4) 和 (5) 是 Loss 计算：
$$ L_i^c = L_{cos}(z_{T_i}^c, \bar{y}_{T_i}^c) $$
$$ L_i^p = \alpha L_{cos}(z_{T_i}^p, \bar{y}_{T_i}^p) + \beta L_{smooth-L1}(z_{T_i}^p, \bar{y}_{T_i}^p) $$
*   对 class token $y^c$，只用 cosine similarity loss $L_{cos}$。Cosine loss 只管方向对不对，不管绝对数值，这对全局语义对齐非常合适。
*   对 patch tokens $y^p$，用 0.9 的 cosine loss 加 0.1 的 smooth L1 loss。Smooth L1（Huber loss）对异常值鲁棒，能把空间几何结构的绝对距离信息稳稳锁住。$z$ 是 Student 经过 adapter head 映射后的特征，$\bar{y}$ 是 Teacher 经过归一化后的特征。

这里有个极关键的细节：**Feature Normalization**。算 loss 之前，必须把 Teacher 的输出先减均值、除标准差（$y \to \bar{y}$）。因为不同大牛模型的“嗓门”大小天差地别，CLIP 的特征 norm 可能极大，DINO 的 norm 可能极小。如果不归一化，那个 norm 最大的模型会瞬间主导整个 gradient，小模型当场崩溃。

实验里还有几个反常识的发现，非常有意思：

在 Table 3 里，如果你加一个 SigLIP2 当 Teacher，VLM（视觉语言）性能反而变差了。原因在于 SigLIP2 和 PEcore 都是 CLIP 家族的，同质化太严重，梯度互相打架。换成 PElang（从 PEcore 蒸馏出来的语言对齐版）效果才好。这告诉我们，多教师蒸馏需要的是“多样性”，同质化的老师只会帮倒忙。

在 Appendix Table 8 里，作者尝试把 Teacher 加大到 7B。Proxy Teacher 变强了，但小模型反而学不好了。这就是经典的 **Capacity Gap**。班主任太牛了，小学生听不懂他在讲什么。解决办法只能是未来加个 Teaching Assistant（比如先蒸馏到 1B，再蒸馏到 100M）做缓冲。

最后聊聊 edge deployment。Table 11 里的数据很有意思。在手机 CPU 上（iPhone 15 Pro），虽然 ConvNext 的 FLOPs 比 ViT 小，但 ViT 跑得反而更快。因为手机 CPU 对矩阵乘法（GEMM，ViT的主要操作）的底层优化远好于卷积操作。对于未来 edge AI 的架构选择，这是一个极强的 insight：在 mobile 端，FLOPs 是个谎言，GEMM 友好的架构才是王道。

**Reference Links:**
1.  Perception Encoder (PEcore / PElang): https://arxiv.org/abs/2504.13181
2.  DINOv3: https://arxiv.org/abs/2508.10104
3.  AM-RADIO: https://arxiv.org/abs/2312.06709
4.  Teacher Assistant Knowledge Distillation (TAKD): https://arxiv.org/abs/1902.03393
5.  Smooth L1 Loss (Fast R-CNN): https://arxiv.org/abs/1504.08083

---

你好 Andrej！非常荣幸能与你交流这篇来自 Meta Reality Labs 与 FAIR 的 paper——《Efficient Universal Perception Encoder》(EUPE)。这篇 paper 探讨了如何在 edge devices 有限的 compute budget 下，构建一个既 efficient 又具备 universal representation 的 vision encoder。为了 build your intuition，我将从核心动机、pipeline 架构、数学公式解析、实验数据挖掘，以及更深层的 geometry 与 optimization intuition 进行全方位的拆解，并附上相关联想与 reference links。

### 1. 核心动机与背景 Intuition

当前的 foundation vision encoders 各有所长：基于 contrastive learning 的 CLIP / SigLIP / PEcore 擅长 image-text alignment 与 zero-shot classification；基于 self-supervised learning 的 DINOv2 / DINOv3 在 dense prediction（如 segmentation, depth estimation）上表现卓越；SAM 则在 promptable segmentation 上一骑绝尘。

如果在 edge devices 上运行多任务 AI 系统，通常面临一个困境：部署多个 domain-specific encoders 会导致 memory 与 compute 爆炸；如果只部署一个 encoder，则在 out-of-domain tasks 上性能断崖式下跌。之前的 agglomerative methods（如 AM-RADIO, DUNE）尝试通过 multi-teacher knowledge distillation 将多个 foundation models 蒸馏到一个 student 中。这种方法在 large models（>300M parameters）上有效，但在 efficient backbones（<100M parameters, 如 ViT-B, ViT-S）上彻底失败。

EUPE 发现，直接从多个 domain experts 蒸馏到 small student 失败的根本原因在于 **capacity bottleneck 与 representation conflict**。小模型没有足够的 parameter capacity 去强行融合极度异构的特征空间（例如 CLIP 的 semantic-aligned 空间与 DINO 的 spatial-structural 空间）。为了解决这个问题，EUPE 提出了 **"scaling up, then scaling down"** 的三阶段 distillation recipe。这本质上是一个信息论中的 bottleneck 缓解策略，通过引入一个具有足够 capacity 的 large proxy model 作为中间态，先将异构知识统一到一个 coherent 的 manifold 中，再让 small student 去拟合这个单一的 manifold。

### 2. Multi-stage Pipeline 架构深度解析

EUPE 的 pipeline 包含三个 stage，设计极其精巧：

*   **Stage 1: Multi-teacher distillation to a large proxy teacher**
    将多个 domain experts（PEcore-G, PElang-G, DINOv3-H+）蒸馏到一个 1.9B parameters 的 ViT-G proxy model 中。Proxy model 拥有足够的高维空间去吸收并融合不同的 inductive biases。
*   **Stage 2: Fixed-resolution distillation to efficient student**
    从 Stage 1 的 proxy teacher 蒸馏到目标 efficient student（如 ViT-B）。固定输入分辨率为 $256 \times 256$。这个 stage 采取较长的 training schedule（390k iterations），在固定分辨率下让 student 充分吸收 proxy teacher 的 universal representation。
*   **Stage 3: Multi-resolution finetuning**
    为了适应 downstream tasks 中多变的分辨率，引入 image pyramid（256, 384, 512）。Teacher 与 student 在每次 iteration 中独立随机选择一个 scale。这使得 student 能够学习 scale-invariant 的特征，修复 ViT 在推理时因分辨率不匹配导致的 positional encoding 错位问题。

### 3. 公式与 Loss 机制技术讲解

Paper 中的 distillation flow 采用了非常标准但经过精细调整的设计。我们来逐个拆解公式中的变量与上下标：

**前向传播输出：**
$$ \left( \boldsymbol { y } _ { S } ^ { c } , \boldsymbol { y } _ { S } ^ { p } \right) = S ( \boldsymbol { x } _ { S } ; \boldsymbol { \theta } ) , \qquad \boldsymbol { y } _ { S } ^ { c } \in \mathbb { R } ^ { d _ { S } } , \boldsymbol { y } _ { S } ^ { p } \in \mathbb { R } ^ { N _ { S } \times d _ { S } } $$

*   $S(\cdot; \theta)$: Student encoder，$\theta$ 为其可学习参数。
*   $x_S$: Student 的输入 image。
*   $y_S^c$: Student 输出的 class token（即 CLS token）。上标 $c$ 代表 class。它存在于空间 $\mathbb{R}^{d_S}$ 中，$d_S$ 是 student 的 feature dimension（例如 ViT-B 的 768）。
*   $y_S^p$: Student 输出的 patch tokens。上标 $p$ 代表 patch。它存在于 $\mathbb{R}^{N_S \times d_S}$ 中，$N_S$ 是 patch tokens 的数量（例如分辨率为 $256 \times 256$ 且 patch size 为 $16 \times 16$ 时，$N_S = 16 \times 16 = 256$）。

对于第 $i$ 个 Teacher：
$$ \left( { y _ { T _ { i } } ^ { c } , y _ { T _ { i } } ^ { p } } \right) = T _ { i } ( x _ { T _ { i } } ; \phi _ { i } ) $$
*   $T_i(\cdot; \phi_i)$: 第 $i$ 个 teacher，$\phi_i$ 为其 frozen 参数。
*   $d_{T_i}, N_{T_i}$: 分别代表第 $i$ 个 teacher 的 feature dimension 和 patch token 数量。

**Adapter Head 对齐维度：**
因为 student 与 teacher 的维度不一致，paper 设计了简单的 2-layer MLP adapter heads：
$$ z _ { T _ { i } } ^ { c } = H _ { i } ^ { c } ( y _ { S } ^ { c } ; \psi _ { i } ^ { c } ) $$
*   $H_i^c, H_i^p$: 分别为针对 class token 和 patch tokens 的 adapter heads，参数为 $\psi_i^c, \psi_i^p$。
*   $z_{T_i}^c \in \mathbb{R}^{d_{T_i}}$: Student 特征经过映射后，在维度上与第 $i$ 个 teacher 对齐。如果 $N_S \neq N_{T_i}$，patch tokens 会通过 bicubic interpolation 对齐空间分辨率。

**Loss 函数：**
$$ L _ { i } = L _ { i } ^ { c } ( z _ { T _ { i } } ^ { c } , \bar { y } _ { T _ { i } } ^ { c } ) + L _ { i } ^ { p } ( z _ { T _ { i } } ^ { p } , \bar { y } _ { T _ { i } } ^ { p } ) $$
$$ L _ { i } ^ { p } ( z _ { T _ { i } } ^ { p } , \bar { y } _ { T _ { i } } ^ { p } ) = \alpha L _ { c o s } ( z _ { T _ { i } } ^ { p } , \bar { y } _ { T _ { i } } ^ { p } ) + \beta L _ { s m o o t h - L 1 } ( z _ { T _ { i } } ^ { p } , \bar { y } _ { T _ { i } } ^ { p } ) $$

*   $L_i^c$: Class token loss，仅使用 cosine similarity loss $L_{cos}$。Cosine loss只关注特征方向的 alignment，忽略了 magnitude，这对于 global semantic representation 非常合适。
*   $L_i^p$: Patch token loss，结合了 cosine similarity loss 与 smooth L1 loss。Smooth L1 loss（Huber loss）对 outliers 更加鲁棒，同时保留了绝对距离的度量信息。这里的权重设置为 $\alpha = 0.9, \beta = 0.1$，表明模型更依赖方向一致性，但辅以距离约束来稳固 dense feature 的几何结构。
*   $\bar{y}$: 经过 Feature Normalization 的 teacher 输出。

**Feature Normalization 的关键作用：**
$$ y_{T_i} \to \bar{y}_{T_i} $$
Paper 明确提出，必须对 teacher 的输出进行 feature normalization（减去 mean，除以 std）。这是由于不同 teacher 的输出统计分布差异极大。例如，CLIP 的 class token 往往具有极大的 norm，而 DINO 的 patch tokens norm 较小。如果直接蒸馏，gradient 会被 norm 最大的 token 主导，导致 student 崩溃。不同于 RADIOv2.5 复杂的 PHI-S normalization，EUPE 采用了极简的 pre-computed 统计量，避免了 on-the-fly EMA 带来的 cross-GPU communication overhead，极大地提升了 training throughput。

### 4. 实验数据表深度挖掘

在 Table 1 中，EUPE-ViT-B 在只有 86M parameters 的情况下，展现了惊人的 universal balance：

*   **Image Understanding**: IN1k-ZS 达到 79.7，IN1k-KNN 达到 84.1，超越了专门的 domain experts PEcore-B (78.4) 和 SigLIP2-B (78.2)。这证明 proxy model 帮助 student 过滤掉了 CLIP 特征中的 noise，学到了更纯粹的 semantic manifold。
*   **Dense Prediction**: 在 ADE20k 上达到 52.4 mIoU，甚至超越了 dense prediction expert DINOv3-ViT-B (51.8)。在 SPair-71k 上达到 51.3，与 DINOv3 持平。
*   **VLM**: 在 RealworldQA 上达到 55.5，GQA 达到 67.3，显著超越了 PEcore-B 和 SigLIP2-B。这得益于 PElang-G 作为 teacher 的一部分，将 strong vision-language grounding 注入了 proxy model。

在 Table 2 的 ablation study 中，验证了三阶段不可分割：
*   如果只用 "Stage 2 only"（直接多 teacher 蒸馏到 ViT-B），SPair 仅为 35.1，NYUv2 高达 0.616。小模型直接面对多 teacher 时，dense prediction 能力完全丢失。
*   加入 Stage 1 后（"Stage 1&2"），VLM 与 Dense 均有提升，但 Dense 提升有限（SPair 41.0）。
*   加入 Stage 3 的 multi-resolution finetuning 后（"Stage 1&2&3"），SPair 飙升至 51.3，NYUv2 降至 0.391。这证实了 multi-resolution 训练对于 dense spatial features 的决定性作用。

Table 3 探讨了 teacher combination。一个极其有趣的发现是：加入 SigLIP2-G 反而导致 TextVQA 从 48.6 跌至 44.8。这说明 SigLIP2 与 PEcore 同属 CLIP-style 模型，特征空间高度重合。在 multi-teacher distillation 中，同质化的 teacher 会导致 gradient 冗余与互相干扰。引入异构的 PElang-G（从 PEcore 通过 language alignment 微调而来）才是最优解。

### 5. 更细节的技术联想与 Intuition 构建

#### A. Capacity Gap 与 Teaching Assistant Phenomenon
在 Appendix Table 8 中，作者做了一个非常有价值的探索：将 DINOv3 teacher 和 proxy model 都 scale up 到 7B（DINOv3-7B & ViT-7B proxy）。结果显示，7B Proxy model 的性能全面超越 1.9B Proxy model，但是，当用这个 7B Proxy model 去蒸馏 ViT-B student 时，性能反而下降了（TextVQA 从 50.4 跌到 48.5，Realworld 从 55.5 跌到 53.9）。

这是一个经典的 **Capacity Gap** 现象。当 teacher 与 student 的 parameter scale 差距过大时，student 无法在有限的 training steps 内拟合 teacher 极度复杂的高维 manifold。这让我联想到 NLP 领域的 distillation，比如从 GPT-4 蒸馏到 7B Llama 相对容易，但直接蒸馏到 1B 模型往往失败。Paper 也提到，未来可能需要引入 Teaching Assistant Knowledge Distillation (TAKD)，即 7B -> 1B -> 100M 的渐进式蒸馏。

#### B. ViT 的 Positional Encoding 与 Resolution Mismatch
Stage 3 中的 multi-resolution 设计触及了 ViT 架构的一个核心痛点：Positional Encoding 的 extrapolation 问题。标准的 ViT 使用 fixed sinusoidal 或 learned absolute positional embedding。当 inference resolution 大于 training resolution 时，需要对 positional encoding 进行 2D interpolation。这种 interpolation 会破坏 patch tokens 之间的 relative spatial relationship，导致 dense prediction 性能断崖式下跌（如 Figure 5 所示，"Stage 1&2 only" 的 feature map 出现了明显的 blocky 现象，semantic coherence 被割裂）。

通过在 image pyramid 中随机 scale 输入，student 被迫在所有 scale 的 positional embedding 上进行联合优化，相当于学习了一个更加平滑的 positional embedding manifold，极大地增强了 scale-invariant 能力。这与 NaViT 和 FlexiViT 处理 variable resolution 的思路有异曲同工之妙。

#### C. PCA Feature Visualization 的深层含义
Figure 4 与 Figure 5 使用 PCA 将 patch tokens 降维到 RGB 空间进行可视化。这种手法最早源自 DINO paper。从几何角度看，PCA 提取的是特征方差最大的前 3 个正交方向。如果模型的 dense feature 质量高，那么同一个 object 的 patches 会在特征空间中形成一个紧凑的 cluster，在 PCA 投影下就会呈现出一致的色彩。

在 Figure 4 中，SigLIP2 与 PEcore 的 feature map 非常 noisy，缺乏空间一致性，这是因为 contrastive loss 只作用于 class token，patch tokens 没有受到强约束（虽然 PEcore 通过 alignment tuning 改善了一些）。DINOv3 的 feature map 边缘极其锐利，但在 fine-grained details（如食物与盘子）上缺乏 discrimination。EUPE 成功融合了两者的优势，既保持了 semantic coherence，又具备了 fine-grained discrimination。这从可视化层面证明了 proxy model 成功地 restructured 了一个更优的 representation manifold。

#### D. Edge Deployment 与 Inference Cost
Table 11 提供了在 iPhone 15 Pro CPU 上的 latency 数据。ViT-B 在 $256 \times 256$ 分辨率下仅需 55.2ms，在 $512 \times 512$ 下需 305.2ms。ConvNext-B 在 $256 \times 256$ 下为 59.3ms。这说明在 < 100M parameters 的区间内，模型完全可以胜任 real-time edge AI tasks。值得注意的是，ConvNext 虽然在 FLOPs 上看似小于 ViT，但在 mobile CPU 上 latency 却更高。这是因为 mobile CPU 对 GEMM (Matrix Multiplication) 的优化远好于 convolution 操作，ViT 的全矩阵乘法架构在 edge hardware 上反而更具优势。这对于未来 edge AI 的架构选型是一个极其重要的 insight。

### 6. Reference Links

为了进一步拓展阅读，这里提供相关的 reference links：

1.  **Perception Encoder (PEcore / PElang)** - EUPE 的核心 teacher 模型：
    https://arxiv.org/abs/2504.13181
2.  **DINOv3** - 提供强大的 dense prediction teacher：
    https://arxiv.org/abs/2508.10104
3.  **AM-RADIO** - 之前的 Agglomerative multi-teacher distillation baseline：
    https://arxiv.org/abs/2312.06709
4.  **DUNE** - 异构 2D/3D multi-teacher distillation：
    https://arxiv.org/abs/2504.09958
5.  **Teacher Assistant Knowledge Distillation (TAKD)** - 解决 capacity gap 的经典方法：
    https://arxiv.org/abs/1902.03393
6.  **Smooth L1 Loss (Fast R-CNN)** - 理解 patch token loss 中的 Huber loss：
    https://arxiv.org/abs/1504.08083

希望这个详尽的拆解能够帮助你在直觉上建立对 EUPE 及 multi-teacher distillation 的深刻理解！如果有任何想要深挖的细节，欢迎随时探讨。
