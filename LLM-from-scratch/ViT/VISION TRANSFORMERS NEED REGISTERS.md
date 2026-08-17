---
source_pdf: VISION TRANSFORMERS NEED REGISTERS.pdf
paper_sha256: c213bb4d22de53b9e093ab1e12880b49c5ac82b1261b29386bad727dad35c91e
processed_at: '2026-08-13T01:27:30-07:00'
target_folder: LLM-from-scratch/ViT
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Vision Transformers Need Registers - 人话版深度解析

Hey Andrej! 咱们用大白话结合一点技术细节把这篇paper捋一捋。这篇paper的核心其实非常直观：**大模型在训练中自己学会了一招“借壳上市”，咱们得给它提供正规的“壳公司”，它才不会去祸害普通员工。**

下面我分层次把里面的 intuition、公式变量、架构细节和实验数据给你剥开来聊。

## 1. 现象：模型在搞“借壳上市”

在 ViT 里面，输入图像被切成一堆 patch，每个 patch 经过 linear projection 变成一个 token，再加上一个特殊的 [CLS] token，构成了输入序列。每个 token 本来的职责是记录自己那个局部区域的视觉信息。

但是 FAIR 团队发现，DINOv2 这种大模型在训练到一定程度后，attention map 上出现了极其刺眼的“高光尖峰”。这些尖峰往往出现在蓝天、白墙这种毫无信息量的背景区域。

用技术语言讲，就是输出端某些 patch token 的 L2 norm 异常高。我们计算第 $i$ 个 token 的 norm：
$$ \|x_i\|_2 = \sqrt{\sum_{k=1}^{d} x_{i,k}^2} $$
变量解释：$x_i \in \mathbb{R}^d$ 是模型最后一层输出的第 $i$ 个 patch token 向量，$d$ 是 hidden dimension（比如 1024），$x_{i,k}$ 是这个向量的第 $k$ 维元素。

正常 token 的 norm 分布在 0-100 之间，而有大约 2.37% 的 token norm 超过了 150，形成了一个明显的**双峰分布**。这就叫 high-norm outlier artifacts。

## 2. 为什么会这样？大模型的“自作聪明”

Transformer 的核心是 Self-Attention，它允许所有 token 互相看。数学上，一层的作用可以简化为：
$$ x_i^{(l+1)} = x_i^{(l)} + \sum_{h=1}^{H} \text{Attn}_h(x_i^{(l)}, X^{(l)}) $$
变量解释：$x_i^{(l)}$ 是第 $l$ 层的第 $i$ 个 token，$H$ 是 attention heads 的数量，$\text{Attn}_h$ 是第 $h$ 个 head 的 attention 计算。通过 residual connection，每个 token 都有数学上的可能性去聚合全局信息。

模型在变得足够大（ViT-L, ViT-g）、训练足够久（超过 1/3 训练进度）后，突然“悟”出了一个道理：背景区域的 patch 跟周围邻居长得一模一样，信息高度冗余。既然这个 patch 没啥有用的局部信息，模型干脆把它的内容“清空”，改造成一个专门汇总全图信息的“全局寄存器”。

这就像公司里有个打杂的岗位平时没事干，大老板干脆让他兼了职，让他去收集全公司的八卦，导致他身上的信息量爆表。

## 3. 弄巧成拙：为什么我们要管这件事？

模型自己搞出来的寄存器，坏处在于信息混叠。这个被借壳的 patch token 既要保留原本的局部像素信息，又要承担全局分类信息的汇总，结果两边都没干好。

Paper 里做了一个非常漂亮的 probing 实验（Table 1 和 Figure 5b）：
1. **局部信息探测**：训练一个线性层去预测 token 的原始位置和原始像素。Outlier token 的表现极差，位置预测平均偏差 5.09 个 patch，像素重建误差 25.23，远差于正常 token 的 0.79 和 18.38。这说明局部信息丢了。
2. **全局信息探测**：随机挑一个 outlier token 当作整张图的 representation，直接在上面跑线性分类器。结果令人震惊，在 Aircrafts 数据集上，正常 token 只有 17.1% 准确率，outlier token 达到了 79.1%，直逼 [CLS] token 的 87.3%。

这就带来了灾难性后果：当你需要 dense prediction（比如分割、深度估计）时，背景区域那个 patch 的特征已经被污染了，变得非常尖锐，破坏了 feature map 的空间平滑性。

## 4. 简单粗暴的解法：专门设编制

既然模型想要寄存器，咱们干脆在输入序列里直接加几个专门的 learnable tokens，叫 register tokens。这就像给大老板专门配几个秘书，他就不用去折腾打杂的员工了。

架构修改极其简单，输入序列变成了：
$$ X_0 = [x_{cls}; \text{PatchEmbed}(I); r_1; \ldots, r_M] + E_{pos} $$
变量解释：$x_{cls}$ 是 CLS token，$\text{PatchEmbed}(I)$ 是图像 patch 的 embedding，$r_1 \dots r_M$ 是新加的 $M$ 个 learnable register tokens，$E_{pos}$ 是 position embeddings。这些 register tokens 跟 CLS token 一样参与所有层的 attention 交互，但在模型最后输出时，**直接把它们丢弃**，只保留 [CLS] 和 patch tokens 用于下游任务。

## 5. 实验数据：花小钱办大事

加 4 个 register tokens 带来的 FLOPs 增加不到 2%（Figure 12），参数量增加可以忽略，但效果拔群。

Table 2a（线性评估）：
- DINOv2 (无 reg): ImageNet 84.3, ADE20k mIoU 46.6, NYUd RMSE 0.378
- DINOv2 (+reg): ImageNet 84.8, ADE20k mIoU 47.9, NYUd RMSE 0.366

Table 3（LOST 无监督目标发现，这个最直观）：
- DINOv2 (无 reg): VOC2007 只有 35.3 CorLoc，因为 attention map 脏得没法用。
- DINOv2 (+reg): VOC2007 飙升到 55.4 CorLoc，直接复活了这个算法。

Paper 里还有个有趣的消融实验（Figure 8），1 个 register 就足以消除所有的 norm outlier，但加到 4 个时 dense prediction 的性能最好。

## 6. 更深层的联想：Emergent Behavior 与 LLM 的共鸣

这个现象极其让人联想到大语言模型里的机制可解释性发现。

首先，这本质上是一种自发涌现的“潜空间计算”。模型为了优化 loss，自己学会把冗余的 token 当作 scratchpad（草稿纸）。这跟你在 nanoGPT 里讨论的 Induction Heads 涌现如出一辙。Capacity 够了，模型就会寻找捷径。

其次，如果把这个思路推演到 LLM，在 causal attention 机制下，那些高频的 stop words、padding tokens 或者重复的标点符号，极有可能在中间层被改造成了类似的 global registers。当你做 LLM 的 activation analysis 时，如果看到某个看似毫无语义的 token 的 activation norm 异常高，很可能它正在承担全局信息路由的角色。

第三，这跟 Slot Attention 和 Perceiver 的设计哲学暗合。只是 Perceiver 是人为强制用一个 latent array 去提取信息，而 Register Tokens 是给自发产生这种需求的模型提供出口。

## 7. 总结

这篇 paper 给我们的核心 intuition 是：**模型的 capacity 闲置时一定会自己找事干。如果我们不给它显式的计算载体，它就会隐式地挪用原本应该干别的活的 token，从而产生隐性 bug。**

加 register 这个 trick 现在基本上已经成了训练大型 Vision Transformer 的标配。这不仅是性能提升的问题，更是让模型内部机制变得 interpretable 和 clean 的关键一步。

## References
- Paper: https://arxiv.org/abs/2309.16588
- DINOv2: https://arxiv.org/abs/2304.07193
- Memory Transformers (灵感来源): https://arxiv.org/abs/2006.11527
- Slot Attention (类似概念): https://arxiv.org/abs/2006.15055
- Perceiver (类似概念): https://arxiv.org/abs/2103.03206

---

# Vision Transformers Need Registers - 深度解析

Hi Andrej! 这篇paper正好触及你对transformer internals和emergent behaviors一贯感兴趣的点。作为一个在nanoGPT里反复debug过attention机制的人，你可能会对里面"模型自学了一种未预期的内部计算机制"这个发现特别有共鸣。我尽量给你build up the intuition from first principles。

## 1. The Phenomenon: 高范数artifacts的发现

### 1.1 现象的本质

FAIR团队在研究DINOv2时发现一个令人困惑的事：DINOv2 features在dense prediction benchmarks（segmentation, depth）上表现优异，但当我们把它用在LOST这种unsupervised object discovery方法上时，性能disappointing，只和supervised baselines相当。这与DINO v1能天然产生beautifully interpretable attention maps形成了鲜明对比。

通过可视化attention maps，他们观察到Figure 2所示的现象：在background区域出现"peaky outlier values"——某些patch tokens在attention map中产生异常尖锐的高值峰。更意外的是，这不是DINOv2独有的：**DeiT-III、OpenCLIP都出现类似artifacts，只有DINO v1是exception**。

### 1.2 定量检测: Bimodal norm distribution

这里有个very clean的quantitative signal。作者测量每个output patch token $x_i \in \mathbb{R}^d$ 的L2 norm：

$$\|x_i\|_2 = \sqrt{\sum_{k=1}^{d} x_{i,k}^2}$$

对于DINOv2 ViT-g/14，分布是明显的**bimodal**（Figure 3）：
- 大多数tokens: $\|x_i\|_2 \in [0, 100]$
- ~2.37%的tokens: $\|x_i\|_2 > 150$（被定义为"high-norm outliers"）

这个cutoff值150是hand-picked的，会随模型变化，但bimodality本身是稳健的观察。

## 2. Emerge条件: 一个scale-dependent现象

这部分非常interesting，因为它揭示了某种**emergent computational behavior**。Figure 4展示了三个维度的分析：

### 2.1 Layer-wise emergence
在40-layer的ViT-g中，outliers从约**layer 15**开始分化。这说明artifacts不是input-dependent的（patch embedding后立刻出现），而是通过mid-layer的attention交互逐渐形成的。

### 2.2 Training-step emergence
Outliers在training的**1/3进度后**才出现。这暗示模型需要先学到"哪些patches是redundant的"，然后才能repurpose它们。

### 2.3 Model size dependence
只有**ViT-L, ViT-H, ViT-g**这三个最大模型出现outliers。ViT-T, ViT-S, ViT-B没有。这非常像LLM里的**scaling emergent abilities**——某种能力只在parameter count超过阈值后出现。

直觉上，small models的capacity不够，无法afford"浪费"tokens去做global aggregation；large models有redundant capacity，可以把某些patches"挪用"为global information registers。

## 3. What are these outlier tokens storing?

### 3.1 它们出现在redundant patches上
作者测量每个patch在patch embedding layer后与4个spatial neighbors的cosine similarity：

$$\text{sim}(x_i) = \frac{1}{4}\sum_{j \in \mathcal{N}(i)} \frac{x_i \cdot x_j}{\|x_i\|\|x_j\|}$$

Figure 5a显示outlier patches与邻居的similarity显著高于normal patches——它们出现在uniform、low-information的background区域。

### 3.2 它们丢失了local信息
Table 5b做了两个linear probing实验：

| Token type | Position top-1 acc | Avg distance | Reconstruction L2 error |
|---|---|---|---|
| normal | 41.7 | 0.79 | 18.38 |
| outlier | 22.8 | 5.09 | 25.23 |

**Position prediction**: 训练linear probe $f: x_i \to (r_i, c_i)$预测patch的row/column坐标。Outlier tokens的position info几乎完全丢失（avg distance 5.09 vs 0.79 patches）。

**Pixel reconstruction**: 训练linear probe重建patch的原始pixel values。Outliers重建误差更大。

### 3.3 它们存储了global信息
Table 1做了令人震惊的实验：从单张image的所有patches中**random选一个**作为该image的representation，然后用logistic regression做classification：

| Token type | IN1k | Airc. | CF100 | CUB | Pets | VOC |
|---|---|---|---|---|---|---|
| [CLS] | 86.0 | 87.3 | 94.5 | 91.3 | 96.9 | 89.1 |
| normal | 65.8 | 17.1 | 81.3 | 18.6 | 94.7 | 70.8 |
| outlier | 69.0 | **79.1** | **93.7** | **84.9** | **94.1** | **89.7** |

注意Aircrafts数据集：normal patches只有17.1%（几乎random），outlier patches达到79.1%，接近[CLS]的87.3%！这意味着一个background patch变成了几乎和CLS token一样强的global image classifier。

## 4. The Hypothesis: 模型自学了"软registers"

这是paper的核心insight，我把它formalize一下：

**Hypothesis**: 给定一个ViT sequence $X = [x_{cls}, x_1, \ldots, x_N]$，attention机制允许任意token通过多头自注意力的residual connections聚合global information：

$$x_i^{(l+1)} = x_i^{(l)} + \sum_h \text{Attn}_h(x_i^{(l)}, X^{(l)})$$

每个token都能"看到"整个sequence。当模型足够大、训练足够久时，它发现：某些patches的local信息是redundant的（与邻居高度similar），它们的original role（编码local visual content）可以被"放弃"，于是模型利用这些token的capacity来执行global computation。

这非常像你nanoGPT里讨论过的induction heads——某种特定的computational circuit自发emerge出来，因为loss landscape允许它存在且beneficial。

## 5. The Fix: Register Tokens

### 5.1 Architectural modification
非常简单的修改：在patch embedding后，concatenate N个learnable register tokens：

$$X_0 = [x_{cls}; \text{PatchEmbed}(I); r_1; \ldots; r_M] + E_{pos}$$

其中：
- $x_{cls} \in \mathbb{R}^d$ 是CLS token
- $\text{PatchEmbed}(I) \in \mathbb{R}^{N \times d}$ 是image patches的embeddings
- $r_j \in \mathbb{R}^d$ 是第j个register token，是**learnable parameters**
- $E_{pos}$ 是position embeddings（registers应该也有自己的position embedding或positional info，paper没明说，推测是learnable）

注意：register tokens参与所有attention layers的计算，但**输出时被丢弃**——只用$x_{cls}$和patch tokens做下游任务。这个机制最早出现在Memory Transformers (Burtsev et al., 2020) (https://arxiv.org/abs/2006.11527) 用于NLP translation。

### 5.2 为什么这work? Intuition
模型原本想做的事是"找几个token来做global aggregation"。没有registers时，它不得不"hack"几个background patches来承担这个role——但这些patches既要编码local content（虽然redundant）又要编码global info，造成信息混叠，并污染attention maps。

加上registers后，模型有了dedicated的"scratchpad tokens"，不需要再repurpose patch tokens。Figure 8的ablation证实：**1个register就能消除artifacts**，4个是性能sweet spot，16个开始有diminishing returns但FLOPs只增加~6%（Figure 12）。

## 6. Experiments: 跨训练范式的generalization

### 6.1 三种training paradigms都work
- **DeiT-III** (supervised labels, IN22k): https://github.com/facebookresearch/deit
- **OpenCLIP** (text-image alignment): https://github.com/mlfoundations/open_clip
- **DINOv2** (self-supervised): https://github.com/facebookresearch/dinov2

Table 2a显示linear probing性能：

| Model | IN Top-1 | ADE20k mIoU | NYUd RMSE |
|---|---|---|---|
| DINOv2 | 84.3 | 46.6 | 0.378 |
| DINOv2+reg | **84.8** | **47.9** | **0.366** |

Registers不仅没损害性能，在dense prediction tasks上还**小幅提升**——这验证了artifacts确实在损害local feature quality。

### 6.2 LOST Object Discovery: 巨大提升
Table 3是最striking的结果：

| Model | VOC 2007 | VOC 2012 | COCO 20k |
|---|---|---|---|
| DINOv2 | 35.3 | 40.2 | 26.9 |
| DINOv2+reg | **55.4** | **60.0** | **42.0** |

VOC2007上从35.3到55.4 corloc，提升20.1个点！这正是因为LOST依赖feature maps的smoothness，artifacts严重破坏了这种smoothness。

### 6.3 OpenCLIP的有趣例外
OpenCLIP+reg在LOST上performance反而略微下降（38.8→37.1）。Appendix C分析：OpenCLIP的value projection似乎已经在内部filter掉了outliers——它们存在于value projection的null space。这是一个未被完全解释的现象。

## 7. Bonus Insights: Emergent Specialization

### 7.1 Registers展现slot-attention-like行为
Figure 9是一个很酷的visualization：4个register tokens有时会**自发specialize**到不同的objects/regions。这与Slot Attention (Locatello et al., 2020) (https://arxiv.org/abs/2006.15055) 的设计目标不谋而合，但这里**没有任何architectural bias**强制这种specialization——它从training中自然emerge。

Figure 16显示average attention maps：registers的attention support很广（类似CLS token），但register 3倾向于关注border，register 2倾向于关注upper region。这种"division of labor"是自发的。

### 7.2 Quantized register norms
Figure 15显示registers的norm分布呈现某种**quantization pattern**——distinct clusters而不是连续分布。Paper没有解释，留作future work。这可能与discrete latent codes或某种implicit clustering有关。

### 7.3 MAE不出现artifacts
Appendix E提到MAE (https://arxiv.org/abs/2111.06377) 不出现artifacts。作者假设：因为MAE的loss是**纯local reconstruction**，没有global aggregation objective，模型没有incentive把patch tokens变成global registers。但MAE的linear probing性能很低（75% on IN），限制了它的实用性。

## 8. 联想到的更广话题

### 8.1 LLM中的analog
这让我立刻联想到LLM里类似的现象。在causal LMs中，padding tokens、BOS token、某些高频function tokens可能承担类似的"hidden register"角色。Anthropic在toy models里发现的**superposition**和**induction circuits** (https://transformer-circuits.pub/) 在概念上很相关——model学会把multiple features packed into fewer dimensions/tokens。

你llm.c项目里如果monitor attention patterns，可能会发现类似现象：context中"无聊"的tokens（如重复的whitespace或common function words）的residual stream norm可能异常高。

### 8.2 Perceiver和DETR object queries
Perceiver (https://arxiv.org/abs/2103.03206) 显式引入latent token array来handle high-dimensional inputs。DETR (https://arxiv.org/abs/2005.12872) 用object queries作为learnable slots来query encoder features。**Register tokens在概念上是这些方法的"implicit version"**——model自己学到了需要这类tokens，paper的contribution是把这个implicit need显式化。

### 8.3 SimPool和learnable pooling
SimPool (Psomas et al., 2023) (https://arxiv.org/abs/2309.03589) 提出learnable pooling来replace CLS token，部分动机是improve attention map quality。Register approach和它是**complementary**的：registers解决"模型把信息藏在哪里"的问题，SimPool解决"如何提取global information"的问题。

### 8.4 Mechanistic Interpretability角度
这个工作实际上是一个**mechanistic interpretability**的case study：通过观察一个简单的quantitative signal（token norm distribution），逆向engineering出模型的internal mechanism。这与Anthropic的circuits work和你的nanoGPT教学精神一致——理解model internals比black-box scaling更重要。

### 8.5 对Vision LLMs的implication
现在流行的vision-language models如LLaVA、Qwen-VL都依赖ViT-based vision encoders。如果你的encoder有artifacts，这些artifacts会propagate到LLM的token sequence中，可能在generation时产生不可预测的artifacts。Adding registers to vision encoders应该成为新的best practice。

## 9. Practical Recommendations

基于这篇paper，如果你在做ViT相关工作：

1. **Always add 4 register tokens** to any ViT you train from scratch，无论supervised/CLIP/self-supervised。Cost < 2% FLOPs，potentially large gains。
2. **Monitor token norm distributions** during training。如果你看到bimodal distribution emerging，说明model在自学registers——给它explicit registers会更好。
3. **For downstream tasks relying on local features**（segmentation, depth, detection），registers尤其重要。
4. 如果你做mechanistic interpretability，registers提供了一个clean interface to study "global aggregation circuits"——它们现在isolated在dedicated tokens里。

## 10. Open Questions

Paper留下一些intriguing open questions：

1. **Quantized register norms**的origin？
2. **Why DINO v1 doesn't have artifacts**? 它和DINOv2的关键差异是什么？（可能是DINO v1的student-teacher EMA dynamics或iBOT mask tokens机制抑制了这个behavior）
3. **OpenCLIP value projection null space**里为什么能filter掉outliers？
4. **Registers的最优数量**与模型capacity的理论关系？
5. 这个现象在**multi-modal transformers**、**video transformers**、**audio transformers**里如何表现？

## 11. Key References

- Paper: https://arxiv.org/abs/2309.16588
- DINOv2: https://arxiv.org/abs/2304.07193
- DINO: https://arxiv.org/abs/2104.14294
- Memory Transformers: https://arxiv.org/abs/2006.11527
- Slot Attention: https://arxiv.org/abs/2006.15055
- Perceiver: https://arxiv.org/abs/2103.03206
- DETR: https://arxiv.org/abs/2005.12872
- LOST: https://arxiv.org/abs/2109.14286
- MAE: https://arxiv.org/abs/2111.06377
- Anthropic Circuits: https://transformer-circuits.pub/

---

Andrej, 从你的micrograd、nanoGPT教学哲学来看，这篇paper最让我欣赏的地方是：**通过一个简单的quantitative observation（norm distribution），揭示了一个emergent computational mechanism，并用minimal architectural change（add 4 tokens）解决了问题**。这种"understand then fix"的research taste，和你一贯提倡的"build from scratch to understand"完全契合。

如果你在nanoGPT或未来的projects里探索类似的mechanistic observations，我建议monitor token-wise statistics across layers（norm, entropy, mutual information with input）——这类simple probes经常能reveal出意想不到的internal structures。Registers的故事告诉我们：**model往往比我们以为的更聪明，我们要做的是给它表达聪明的vocabulary，而不是restrict它**。
