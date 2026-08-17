---
source_pdf: OpenVLA.pdf
paper_sha256: 353c37df34458f12f969b14dfd8b77175b727b9cddea7bb891759beddeefe1be
processed_at: '2026-08-06T01:05:59-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OpenVLA 用人话版

好，我把那些技术细节先放一边，用大白话讲讲这帮人到底干了啥。

## 一句话说清楚

他们做了一个**7B参数的机器人大脑**，喂一张照片和一句指令（"把那个茄子放进锅里"），它就一帧一帧吐出机器人该怎么动。关键是这个大脑开源了，谁都能拿去改、拿去用，而且比Google那个55B的闭源版本还猛一点。

## 这事儿为什么难

机器人学里有个老问题：你拿几千条demo训个policy，它换张桌子就废了，换个杯子颜色就废了，更别说新指令了。但你看GPT-4V那种VLM，给它看张图它能聊半天，语义理解强得离谱。于是大家就想：**能不能把这种"看图说话"的大脑，改成"看图动手"的大脑？**

RT-2（Google）先证明了这条路走得通——把PaLI这种VLM fine-tune一下，让它输出的token不是文字而是机器人action，效果惊艳。但问题有两个：RT-2是55B的closed model，你摸不着；而且他们没告诉你怎么把这个模型fine-tune到你自己的机器人上。

OpenVLA就是来填这两个坑的。

## 大脑怎么搭

三块积木，很直觉：

**第一块：眼睛**——用两个vision encoder并行看图。一个叫SigLIP，管"这是啥"（语义）；一个叫DINOv2，管"这在哪、长啥样"（空间）。两个的feature拼一起。为什么非要两个？因为机器人不光要知道"那是茄子"，还得知道"茄子在桌子左边、朝这个角度、gripper得斜着抓"。单靠SigLIP这种contrastive language pretrain的encoder，spatial detail不够细。DINOv2是self-supervised的，对patch级别的空间结构特别敏感，刚好补上。

paper里做了ablation，去掉DINOv2性能掉5%——不算大，但在精细操作上那5%就是"抓得到"和"抓不到"的区别。

**第二块：翻译官**——一个2层MLP，把visual features投到Llama 2的embedding space。没啥好说的，就是个维度对齐的adapter。

**第三块：大脑**——Llama 2 7B，就是那个开源LLM。它的输入是[image patch tokens] + [instruction tokens] + [action tokens]，训练目标是标准的next-token prediction，但loss只算在action tokens上。

这里有个巧妙的hack：机器人的action是7维连续向量（xyz位移 + 旋转 + gripper开合），但LLM只会输出离散token。怎么办？每维独立分256个bin，bin的边界用数据的1%和99%分位数（不是min/max，避免outlier撑爆区间）。这样7维action变成7个离散整数，每个对应Llama vocab里的一个token。

问题是Llama tokenizer只给100个special token的额度，256个action bin塞不下。他们的解法简单粗暴：**直接覆盖Llama vocab里最不常用的256个token**。这些token本来就是些极低频的怪字符，覆盖了基本无损。这种工程直觉很Karpathy——别跟tokenizer较劲，能跑就行。

## 数据怎么搞

Open X-Embodiment是个大杂烩，70多个数据集、200万条轨迹。他们筛到970k条，标准是：有第三人称相机 + 单臂end-effector control。mixture weight基本沿用Octo的配比，Bridge/Fractal/Kuka各占~13%。

有个很诚实的细节：DROID这个新出的in-the-wild大数据集，他们只给10%的权重，后来发现action token accuracy一直上不去，干脆在训练后1/3阶段把它踢了。这说明7B的capacity其实吃不下所有数据的diversity——数据多不等于模型学得动，得match model size。

## 训练那些坑

这部分是paper最值钱的section，全是踩坑出来的直觉：

**1. Vision encoder必须fine-tune**。这在VLM圈是反直觉的——VLM领域通常freeze vision encoder更好（保住Internet pretrain的robust feature）。但VLA里freeze了性能崩一半。原因：Internet pretrain教vision encoder"这是猫那是狗"，但机器人需要的是"这个物体的精确边界、和gripper的相对位姿"，这种fine-grained spatial detail得在robot数据上重新学。所以vision encoder也得跟着adapt。

**2. 224×224就够了**。试了384，性能没涨但训练时间3x。VQA那种任务高分辨率有用，但机器人控制224px够用，省下来的算力换成更高的control frequency更划算。

**3. 训27个epoch**。LLM/VLM通常1-2 epoch就停，VLA得27 epoch，而且性能一直涨到action token accuracy超过95%。robot数据"信息密度低"，每个trajectory里action高度重复，得多过几遍才能学会action generation。这跟language pretrain的"一遍过"完全不同。

**4. Learning rate 2e-5，不warmup**。跟VLM pretrain一样的LR就行，说明这个LR对pretrain权重不算"破坏性"，主要是在VLM feature space上做task adaptation，不需要小心翼翼地warm up。

## 到底有多猛

### BridgeData V2 WidowX（170 rollouts，17个任务）

OpenVLA在15/17个任务上最好或并列最好，aggregate比RT-2-X高出一截。RT-2-X只在semantic generalization（从没见过的物体/概念）上略胜——预期内，人家55B + co-training，semantic prior保留得更好。

有个偷笑的细节：OpenVLA在Bridge上碾压RT-2-X，部分原因是他们发现BridgeData V2原始数据每个demo第一个action是all-zero no-op，直接训会让policy学到"冻结"行为。他们把这些no-op过滤了，RT-2-X是closed model没法重训，只能用query second-most-likely action的workaround。这个不对等比较有点占便宜，但paper很坦诚地承认了。

### Google Robot（60 rollouts，12个任务）

OpenVLA和RT-2-X打平，都把RT-1-X和Octo按在地上摩擦。在"Move Coke Can to Taylor Swift"这种纯semantic任务上，RT-2-X 3/5，OpenVLA 2/5——55B和co-training的优势在这里显出来了。

### Fine-tuning到新机器人（Franka，10-150 demos）

这是RT-2-X完全没碰的领域。结论很清晰：
- **窄单指令任务**：Diffusion Policy从scratch训最强，能到80-100%。它action space连续、有temporal smoothing，天生适合窄而精的任务。
- **多指令多物体任务**：OpenVLA和Octo明显领先。预训练给的language grounding能力在这里发力。
- **OpenVLA scratch（不经过OpenX预训练）远不如OpenVLA**：证明增益来自robot data diversity，不只是VLM backbone。

OpenVLA是唯一在所有任务上都≥50%的方法，说明它是robust的default choice——你不知道任务长啥样时，选它不会翻车。

### LoRA fine-tuning

| 方法 | 成功率 | 训练参数 | VRAM |
|---|---|---|---|
| Full FT | 69.7% | 7.2B | 163GB |
| LoRA r=32 | 68.2% | 97.6M | 59.7GB |

LoRA只训1.4%参数，性能几乎打平full FT，VRAM降到60GB。一张A100 80G甚至4090 24G（配合quantization）就能fine-tune一个robot foundation model。这是democratization的关键——没有这个，970k数据预训练出来的model对普通lab没意义。

### int4 quantization

| 精度 | 成功率 | VRAM |
|---|---|---|
| bfloat16 | 71.3% | 16.8GB |
| int8 | 58.1% | 10.2GB |
| int4 | 71.9% | 7.0GB |

int4性能跟bfloat16一样，VRAM只要7GB！4090甚至3090都能跑。int8反而差——quantization ops有overhead，推理慢，non-blocking control下dynamics变了，性能就掉。Appendix D.4用blocking control做了control experiment，确认int8的下降纯粹是速度问题，不是quantization破坏了policy。

部署guidance：**就用int4，别犹豫**。

## 它没解决的问题

paper很诚实，limitation列得清清楚楚：

**1. 只吃单张图**。没wrist camera、没proprioception、没history。很多精细操作wrist view是刚需，proprioception对recovery from perturbation关键。这俩加上去应该能再涨一截。

**2. 6Hz太慢**。ALOHA那种50Hz bimanual直接没戏。action chunking（一次predict 16步）是明显fix，但paper没做。

**3. 不到90%成功率**。SOTA但没到production grade。要deploy得再往上推10-20%。

**4. Action representation**。256-bin discretization对gripper这种binary signal浪费，对6D pose连续控制有quantization error。Diffusion Policy那种continuous action更优雅。后续π0、CogACT都在这块做了改进。

## 我的多嘴

OpenVLA真正的贡献不是技术多novel——它的每个组件（VLM backbone、action discretization、LoRA、quantization）都是现成的，recipe不复杂。它的贡献是**把这条路走通了、走全了、全开源了**。

在它之前，VLA是Google DeepMind的特权游戏。在它之后，任何有张4090的lab都能下载checkpoint、LoRA fine-tune到自己的机器人、发paper改进某个组件。这跟LLM领域Llama 2的角色一模一样——closed model（GPT-4）证明可行，open model（Llama 2）普及生态。

从这往后捋，几个方向社区已经在猛跑：
- **π0**：flow matching替discretization，continuous action更优雅
- **RDT-1B**：bimanual的diffusion transformer VLA
- **OpenVLA-OFT**：OpenVLA官方follow-up加flow matching
- **HPT**：处理不同robot的proprioception异构性
- **SmolVLA**：更小更快的方向
- **3D-VLA**：3D scene representation替2D image

每个都在补OpenVLA limitation的某个洞。这就是open ecosystem的威力——baseline一出来，所有人都知道往哪使劲。

你如果要进robot foundation model这个坑，OpenVLA是必读必跑的baseline。codebase在github.com/openvla/openvla，写得挺干净，从那fork比自己搭快10倍。

参考链接：
- https://openvla.github.io
- https://github.com/openvla/openvla
- https://huggingface.co/openvla
- https://arxiv.org/abs/2406.09246（OpenVLA论文）
- https://arxiv.org/abs/2402.07865（Prismatic VLMs）
- https://arxiv.org/abs/2310.08864（Open X-Embodiment）
- https://arxiv.org/abs/2307.15818（RT-2）
- https://arxiv.org/abs/2106.09685（LoRA）
- https://arxiv.org/abs/2305.14314（QLoRA）
- https://droid-dataset.github.io（DROID）
- https://libero-project.github.io（LIBERO）

---

# OpenVLA 深度解析

这篇paper是Stanford、TRI、Google DeepMind等多家机构合作的成果，主打一个**开源7B参数的Vision-Language-Action Model**，在多个机器人embodiment上直接可用，且支持efficient fine-tuning。我先从整体定位讲起，再钻到技术细节。

## 1. 核心定位：为什么需要OpenVLA

机器人manipulation的痛点在于**generalization**——单一任务训练的policy对新物体、新场景、新指令几乎零迁移能力。而VLM（CLIP、SigLIP、Llama 2）在Internet-scale数据上学到了强大的semantic priors。RT-2/RT-2-X证明了把VLM直接fine-tune成action predictor的可行性，但存在两个问题：

1. **Closed source**：RT-2-X是55B的closed model，社区无法复现、改进、做ablation
2. **Fine-tuning practice缺失**：RT-2-X论文没系统研究如何高效fine-tune到新机器人

OpenVLA的目标就是补上这两块，同时用7B参数打过55B的RT-2-X（在29个任务上绝对成功率高出16.5%）。

参考链接：
- 项目主页：https://openvla.github.io
- Prismatic VLMs（backbone来源）：https://arxiv.org/abs/2402.07865
- Open X-Embodiment：https://arxiv.org/abs/2310.08864
- RT-2：https://arxiv.org/abs/2307.15818

## 2. 架构详解

整体三件套，跟主流VLM一致，但每个组件都有讲究。

### 2.1 Vision Encoder：双路融合

这是OpenVLA区别于RT-2的关键设计之一。它用两个visual encoder并行处理同一张224×224的图像：

- **SigLIP**（600M params的一部分）：来自Google，用sigmoid loss替代softmax loss做对比学习，输出high-level semantic features
- **DINOv2**（Meta，self-supervised）：输出low-level spatial features，对物体边界、位置、几何结构敏感

两者的patch embeddings在channel维度concat：

$$\mathbf{F}_{\text{ fused}} \in \mathbb{R}^{P \times (d_{\text{SigLIP}} + d_{\text{DINOv2}})}$$

其中$P$是patch数量（224/14=16，所以$P=16 \times 16 = 256$个patch），$d$是各encoder的embedding维度。

**Intuition**：机器人需要同时知道"这是什么"（SigLIP负责semantic）和"这在哪里、什么形状"（DINOv2负责spatial）。单一的CLIP/SigLIP encoder在spatial reasoning上不够强，DINOv2的self-supervised features恰好补上这块。paper在Appendix D.2做了ablation：去掉DINOv2，Bridge任务平均成功率从45.6%掉到40.6%（5%绝对下降）。这个gain看起来不大，但在fine-grained manipulation场景下spatial信息至关重要。

### 2.2 Projector：2-layer MLP

把fused visual features映射到Llama 2的token embedding space：

$$\mathbf{F}_{\text{proj}} = \text{MLP}_2(\mathbf{F}_{\text{fused}}) \in \mathbb{R}^{P \times d_{\text{LLM}}}$$

这里$d_{\text{LLM}} = 4096$（Llama 2 7B的hidden size）。Projector只有2层，是个轻量级adapter，不喧宾夺主。

### 2.3 LLM Backbone：Llama 2 7B

整个架构的核心计算引擎。输入序列是：

$$[\text{image patches}] \oplus [\text{instruction tokens}] \oplus [\text{action tokens}]$$

通过standard next-token prediction训练，cross-entropy loss只算在action tokens上（instruction和image patches是input context）。

Llama 2 7B的具体配置：32层transformer decoder，4096 hidden dim，32 attention heads，grouped-query attention，RoPE位置编码，SwiGLU激活。

## 3. Action Discretization：把连续控制变成token

这是VLA范式的关键trick。机器人的action通常是7维连续向量（6D pose + gripper open/close），但LLM只能输出discrete tokens。

### 3.1 分箱策略

对每个action维度独立做256-bin的uniform discretization。bin的边界用training data的**1st和99th percentile**（不是min/max）：

$$\text{bin}_i = \text{UniformDiscretize}(a_i, [Q_1(a_i), Q_{99}(a_i)], 256)$$

其中$a_i$是第$i$维action，$Q_1, Q_{99}$是1%和99%分位数。bin index $\in [0, 255]$。

**为什么不直接用$[\min, \max]$？** Outlier actions会过度拉伸discretization range，导致大部分正常action挤在少数几个bin里，effective resolution严重降低。用quantile截尾就把outliers的影响隔离掉了。

### 3.2 Tokenizer hack

Llama 2的tokenizer只预留了100个special tokens给fine-tuning用，但256个action bin需要256个token。怎么办？直接**覆盖Llama tokenizer vocabulary里最少用的256个token**（即最后256个）。

这个操作很暴力但很有效——这256个token原本就是极低频的Unicode字符或rare tokens，覆盖掉它们对自然语言能力几乎无影响，但换来了完整的action表达空间。

最终一个7维action序列变成：

$$[a_1^{(0)}, a_2^{(0)}, \ldots, a_7^{(0)}, a_1^{(1)}, \ldots, a_7^{(T)}]$$

每个$a_i^{(t)} \in \{0, 1, \ldots, 255\}$对应Llama vocab里的一个token。

## 4. 训练数据：Open X-Embodiment的curated mixture

总共970k条真实机器人轨迹，来自OpenX的70+个数据集。curation目标有两个：

### 4.1 输入输出空间统一
只保留：有第三人称相机 + 单臂end-effector control的数据集。这一步筛掉了大量多臂、无相机、特殊sensor的数据集。

### 4.2 Embodiment/scene/task平衡
用Octo的mixture weights作为基础（Octo已经做了heuristic down-weighting低多样性数据集、up-weighting高多样性数据集），再加一些新数据集。最终mixture见Table 3，最大几个是：
- BridgeData V2：13.3%
- Fractal（RT-1数据）：12.7%
- Kuka：12.7%
- BC-Z：7.5%
- FMB：7.1%
- DROID：10%（但只在训练前2/3阶段用，后面因为action token accuracy上不去移除了）

**DROID的有趣故事**：DROID是新发布的大规模in-the-wild数据集，理论上应该大力用。但OpenVLA团队发现DROID的action token accuracy一直很低，说明7B模型还不够吸收DROID的全部多样性。最终保守起见，在训练最后1/3阶段把DROID从mixture里拿掉。这是个很诚实的工程决策——数据多不等于数据好，得看model capacity够不够。

## 5. 关键设计决策（Section 3.4的精华）

这部分是paper里最有工程价值的section，每个决策都来自小规模BridgeData V2上的ablation。

### 5.1 VLM Backbone选择
比较了三个VLM：
- IDEFICS-1
- LLaVA
- Prismatic-7B

结果：在单物体任务上三者差不多，但在**多物体language grounding任务**上（即场景里有多个物体，根据instruction选对目标），LLaVA比IDEFICS-1高35%绝对成功率，Prismatic又比LLaVA高~10%。差异主要来自Prismatic的fused SigLIP-DINOv2 encoder——spatial reasoning对"哪个物体是目标"至关重要。

### 5.2 Image resolution
比较224×224和384×384。**没有性能差异**，但384的训练时间3x。所以选224。

这点跟VLM benchmark的趋势相反——在VQA、captioning这些任务上高分辨率确实有用。但在机器人控制里，224px已经够用，且control对推理速度敏感，低分辨率省下来的compute可以换成更高的control frequency。

### 5.3 Vision encoder必须fine-tune
这是个反直觉的发现。在VLM领域，freeze vision encoder通常更好（保留Internet预训练的robust features）。但在VLA里，**fine-tune vision encoder至关重要**——freeze会导致性能大幅下降。

paper在Appendix D.3给了量化：两个不同VLM backbone上，frozen vision分别只有46.7%和很差的performance，fine-tune后到80%+。

**Intuition**：Internet pretrain的vision encoder对"识别物体类别"很在行，但对"物体精确位置、形状、与gripper的相对位姿"这种fine-grained spatial detail不够。机器人控制需要pixel-precise的spatial understanding，所以必须让vision encoder在robot数据上进一步adapt。

### 5.4 训练27个epoch
LLM/VLM通常只训1-2 epoch，VLA需要27 epoch。原因：robot数据相对"信息密度低"，每个trajectory里action高度重复，需要多次pass才能让action token accuracy到95%+。这是过拟合风险和action learning需求的trade-off——VLA阶段model已经在VLM pretrain里学完了language/vision understanding，剩下的全部capacity用来学action generation。

### 5.5 Learning rate
固定2e-5，不要warmup。sweep了多个量级，2e-5最好（跟VLM pretrain一样）。不需要warmup说明这个LR对预训练权重不是"破坏性"的——主要在VLM的feature space上做task adaptation。

## 6. 实验结果分析

### 6.1 BridgeData V2 WidowX（Figure 3 + Table 4）

17个任务，每任务10次trial，共170 rollouts。任务覆盖5个generalization axis：
- Visual gen（5 tasks）：unseen背景、distractor、颜色
- Motion gen（2 tasks）：unseen位置/朝向
- Physical gen（3 tasks）：unseen尺寸/形状
- Semantic gen（4 tasks）：unseen目标物体/指令
- Language grounding（3 tasks）：多物体场景按指令选对目标

OpenVLA在15/17个任务上达到或并列最好，aggregate成功率显著高于RT-2-X。值得注意的是RT-2-X在semantic gen上略胜一筹——这预期内，因为RT-2-X用更大规模Internet data co-train，semantic priors保留得更好。

### 6.2 Google Robot（Figure 4 + Table 6）

12个任务，每任务5 trials。OpenVLA和RT-2-X性能相当，都显著高于RT-1-X和Octo。

**有意思的细节**：在"Move Coke Can to Taylor Swift"这种纯semantic generalization任务上（场景里有3张名人照片，需要把可乐放到Taylor Swift那张上），RT-2-X 3/5，OpenVLA 2/5。差距不大，但RT-2-X的55B参数和co-training确实给了它一点edge。

### 6.3 与RT-2-X对比的关键优势来源（Appendix C）

OpenVLA在Bridge上碾压RT-2-X的原因不光是架构，还有**数据preprocessing**。OpenVLA团队发现BridgeData V2原始数据里每个demo第一个action是all-zero（no-op），直接训练会导致policy学到"冻结"行为。OpenVLA把这些no-op transition过滤掉了。

RT-2-X是closed model，不能重训。它的workaround是query second-most-likely action（因为first-most-likely经常是all-zero）。这个workaround能跑，但suboptimal。paper很坦诚地承认了这个不对等比较的现实约束。

### 6.4 Data-efficient fine-tuning（Figure 5 + Table 7）

Franka-Tabletop（5Hz）和Franka-DROID（15Hz），10-150 demos的fine-tuning。比较对象：
- Diffusion Policy（from scratch）
- Diffusion Policy（matched input/output spec）
- Octo fine-tuned
- OpenVLA fine-tuned
- OpenVLA scratch（直接从Prismatic VLM fine-tune，不经OpenX预训练）

结果pattern：
- **窄单指令任务**（如Put Carrot in Bowl, Pour Corn into Pot）：Diffusion Policy最强，能达到80-100%
- **多指令多物体任务**（如Move <obj> onto Plate, Knock <obj> Over）：OpenVLA和Octo明显领先
- **OpenVLA scratch远不如OpenVLA**：证明OpenX预训练的增益来自robot data diversity，不只是VLM backbone

OpenVLA aggregate 67.2%，Diffusion Policy 48.5%，Octo 43.4%。OpenVLA是唯一在所有任务上都至少50%的方法——这说明它是个robust的default选择，尤其在任务结构未知时。

### 6.5 Parameter-efficient fine-tuning（Table 1）

这是paper的重要contribution之一，RT-2-X完全没碰这块。比较：

| Strategy | Success Rate | Train Params | VRAM |
|---|---|---|---|
| Full FT | 69.7% | 7.19B | 163GB |
| Last layer only | 30.3% | 465M | 51GB |
| Frozen vision | 47.0% | 6.76B | 156GB |
| Sandwich | 62.1% | 914M | 64GB |
| LoRA r=32 | **68.2%** | 97.6M | 59.7GB |
| LoRA r=64 | 68.2% | 195M | 60.5GB |

**LoRA是sweet spot**：只训1.4%参数（97.6M），性能匹配full FT，VRAM只需59.7GB（可以在单个A100 80GB或4090 24GB上跑）。

LoRA的公式（对每个linear layer $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$）：

$$W' = W + \Delta W = W + B A$$

其中$A \in \mathbb{R}^{r \times d_{\text{in}}}$，$B \in \mathbb{R}^{d_{\text{out}} \times r}$，$r \ll \min(d_{\text{in}}, d_{\text{out}})$是rank。paper发现r=32和r=64性能一样，推荐r=32。

**为什么frozen vision和last layer only都不行？** 印证了"vision encoder必须adapt到target scene"的结论——robot control对visual feature的spatial precision要求极高，frozen features不够用。Sandwich（vision encoder + token embedding + last layer）比frozen vision好但仍不如LoRA，说明LLM内部的中间层也需要adaptation。

### 6.6 Quantization inference（Table 2 + Appendix D.4）

7B模型默认bfloat16需要15GB VRAM。试了int8和int4：

| Precision | Success | VRAM |
|---|---|---|
| bfloat16 | 71.3% | 16.8GB |
| int8 | 58.1% | 10.2GB |
| int4 | **71.9%** | 7.0GB |

int4和bfloat16性能一样，VRAM减半！int8反而差——因为int8的quantization ops有overhead，推理速度慢，在non-blocking control下系统dynamics被改变。

Appendix D.4做了关键control experiment：用**blocking control**（每个action执行完才query下一个）消除推理速度差异。结果int8/bfloat16/int4性能都差不多（74.4%/70.0%/68.8%）。这证实了int8在non-blocking下的性能下降纯粹是speed导致的，不是quantization本身破坏了policy quality。

**Practical takeaway**：部署OpenVLA就用int4，VRAM只要7GB，4090甚至3090都能跑，速度还快。

### 6.7 Inference speed（Figure 6）

不同GPU上的control frequency：
- RTX 4090 + bfloat16：~6Hz
- RTX 4090 + int4：更高
- H100：更快
- A5000 + int8：1.2Hz（太慢了，影响task performance）
- A5000 + int4：3Hz（勉强可用）

这给了一个清晰deployment guideline：consumer GPU上int4是必选项，server GPU可以bf16。

## 7. LIBERO simulation验证（Appendix E）

为了reproducibility，paper在LIBERO benchmark上做了sim实验。4个task suite：Spatial/Object/Goal/Long。

| Method | Avg SR | Avg Rank |
|---|---|---|
| Diffusion Policy from scratch | 72.4% | 2.5 |
| Octo fine-tuned | 75.1% | 2.0 |
| OpenVLA fine-tuned (LoRA r=32) | **76.5%** | **1.5** |

OpenVLA仍是最好的，但margin比real-world实验小。paper解释：OpenVLA和Octo都是纯real-world预训练，sim有domain gap。如果pretraining mixture加sim数据，gain应该更大。这是个清晰的future work方向——sim+real co-training。

## 8. Limitations和我的思考

paper很诚实地列了limitation：

1. **Single image only**：不支持多相机、wrist camera、proprioceptive input、observation history。这对precise manipulation是硬伤——很多任务wrist camera提供关键的close-up view。
2. **6Hz推理速度**：对ALOHA这种50Hz bimanual setup完全不够。Action chunking（一次predict K步）和speculative decoding是potential fix。
3. **可靠性<90%**：state-of-the-art但还不到production-grade。
4. **未探索的design space**：base VLM size的影响、robot data + Internet data co-training的效果、best visual features都还是open question。

我补充几个观察：

**A. Action representation的局限**：256-bin per dimension的discretization对gripper这种binary signal很浪费，对6D pose的连续控制又有quantization error。Diffusion Policy的continuous action space在这方面更优雅。future work可能需要hybrid representation——discrete token for coarse, continuous refinement head for precision。

**B. 没用proprioception**：robot的joint state、gripper state完全没作为input。这对recovery from perturbation很不利——policy不知道自己当前end-effector pose在哪，只能靠visual推断。加上proprioception token应该能显著提升。

**C. No action chunking**：每个timestep都重新run整个7B model，浪费严重。如果像Diffusion Policy那样一次predict 16步action sequence，effective control frequency能提升16x。

**D. VLM backbone的天花板**：Llama 2 7B在2024-2025的标准下已经偏小。换成Llama 3 8B、Mistral 7B、甚至Gemma 2 9B，semantic reasoning能力会有质的提升。paper里提到"base VLM size影响"是open question，但社区已经在快速验证。

**E. Data scaling**：970k轨迹听起来多，但跟Internet text/image data的trillion scale比还是太少。BridgeData V2、DROID这种大规模采集 + sim2real + synthetic data generation是必须的scaling路径。

## 9. 对整个VLA field的影响

OpenVLA的真正价值在于**lowered barrier to entry**。在它之前，VLA研究基本是Google DeepMind的特权——只有他们有compute、data、closed model的API。OpenVLA之后：

- 任何lab都能在HuggingFace下载checkpoint：https://huggingface.co/openvla
- 用单卡4090 + LoRA就能fine-tune到自己的机器人
- 可以做architecture ablation、data mixture study、new training objective
- 能复现RT-2的核心idea并改进

这跟LLM领域的Llama路线图一模一样：closed model（GPT-4）证明可行性 → open model（Llama 2）普及生态 → 社区迭代加速。OpenVLA是robot foundation model的"Llama时刻"。

## 10. 相关工作联想

顺着OpenVLA往下捋，几个值得关注的后续方向：

1. **π0 (Physical Intelligence)**：flow matching + VLA，处理continuous action space更优雅
2. **RDT-1B**：bimanual的VLA，用Diffusion Transformer
3. **OpenVLA-OFT**：OpenVLA的follow-up，加flow matching做continuous action
4. **CogACT**：discrete + continuous hybrid action head
5. **HPT (Heterogeneous Pre-trained Transformer)**：处理不同robot的proprioception异构性
6. **Octo 2 / SmolVLA**：更小更快的方向
7. **3D-VLA**：用3D scene representation替代2D image
8. **GR-2 / Helix**：video pretraining + VLA

每个都针对OpenVLA limitation的某个方面做了改进。

## 11. 总结

OpenVLA是2024年机器人foundation model领域的里程碑paper。它的价值分三层：

**Layer 1 - Engineering**：证明7B参数 + 双encoder + OpenX data + 27 epoch训练能打过55B closed model，给出了清晰可复现的recipe。

**Layer 2 - Methodology**：系统研究VLA的fine-tuning（LoRA）、quantization（int4）、design decision（vision encoder必须fine-tune），这些RT-2 paper完全没碰。

**Layer 3 - Ecosystem**：开源checkpoint + codebase + fine-tuning notebook，把VLA从closed research拉到open community，类比Llama对LLM的意义。

如果你要进入robot foundation model领域，OpenVLA是必读且必跑的baseline。它的codebase（https://github.com/openvla/openvla）写得相当干净，从这里fork做改进比从scratch搭快10倍。

---

**进一步阅读建议**：
- Prismatic VLMs: https://arxiv.org/abs/2402.07865
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- RT-2: https://arxiv.org/abs/2307.15818
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- DROID dataset: https://droid-dataset.github.io/
- LIBERO benchmark: https://libero-project.github.io/
- DINOv2: https://arxiv.org/abs/2304.07193
- SigLIP: https://arxiv.org/abs/2303.15343
