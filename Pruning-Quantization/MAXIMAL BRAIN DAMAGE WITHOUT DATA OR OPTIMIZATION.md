---
source_pdf: MAXIMAL BRAIN DAMAGE WITHOUT DATA OR OPTIMIZATION.pdf
paper_sha256: 0aff914563f2b308febf1ddcc2339a9c2e9be7530e1a2154f994f844a92afbcd
processed_at: '2026-08-05T16:56:10-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲这篇paper

Karpathy，咱就别整那些学术腔了，跟你直接讲我的理解和直觉。

## 这paper其实在说啥

一句话：**只要你能改到模型的权重bit，flip两三个sign bit就能把任何DNN砸废，而且你不需要数据、不需要跑模型、不需要任何优化。**

就这么简单。这个简单背后其实藏着一个很深的事实：训练好的神经网络，functional capacity极度集中在极少数parameters上。绝大多数weights是"陪跑"的，删了也没事（这就是pruning的前提）；但极少数是"承重墙"，动一个就塌。

## 先说威胁模型有多严格

attacker只能做一件事：flip几个bit。其余什么都干不了。

- 没有训练数据，连一张图、一个token都没有
- 不能forward pass，不能backward pass（1P-DNL稍微放松，允许跑一次random input）
- 不能观察模型输出
- 就只能直接读写存储的parameters

这相当于啥？相当于一个小偷潜入你家，不知道你家住几口人、不知道你家布局，但能精准地把承重墙里的一根钢筋锯断一根半根，房子就塌了。

为什么这个威胁模型重要？因为之前所有的weight attack（BFA、DeepHammer、ZeBRA）都需要数据来算梯度，需要反复forward/backward来search。这些在真实部署里都不现实——你黑进了一个自动驾驶公司的服务器，还想用他们的数据集跑forward pass调参？DNL告诉你：完全不需要，扫一遍权重magnitude就够了。

## 为什么sign bit是攻击目标

IEEE 754 FP32：1位sign + 8位exponent + 23位mantissa。

Flip sign bit = 把权重从正变负或反之。$0.05 \to -0.05$。

这个操作的"性价比"极高：
- 改动极小（1个bit）
- 但语义上完全reverse了这个weight的作用——原来它push某个方向，现在pull
- magnitude不变，所以模型的整体scale、normalization都不会被破坏，反而更隐蔽

对比exponent flip：把 $0.001$ 变成 $1000$，这种极端rescaling很容易被检测（NaN、inf、layernorm爆炸）。Sign flip是个"surgical"的破坏，正好破坏功能但不破坏numerical stability——这正是攻击者想要的。

paper里有个有意思的finding：vision models对sign flip更敏感（因为Gabor filter的lobes对方向极敏感），而LLMs对exponent flip更敏感（因为attention对scale极敏感）。这个domain dependence本身就值得深挖。

## 三个heuristic：DNL的全部秘密

DNL的核心就三个规则，组合起来catastrophic。

### 1. 选magnitude最大的

直觉：训练好的网络里，大权重是"重要节点"。Pruning说小权重可以删——反过来，大权重就是不能动的。Flip大权重的sign = 让最重要的节点从助力变成阻力。

为啥这个成立？paper给了个理论motivation，来自Optimal Brain Damage（[LeCun 1989](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)）。

对loss做二阶Taylor展开：
$$\Delta \mathcal{R} \approx g^\top \Delta\theta + \frac{1}{2}\Delta\theta^\top H \Delta\theta$$

变量解释：
- $g = \nabla_\theta \mathcal{R}(\theta)$：梯度
- $H$：Hessian矩阵
- $\Delta\theta$：参数扰动

训练收敛时$g \approx 0$，curvature主导。Flip $\theta_i$后$\Delta\theta_i = -2\theta_i$。对角Hessian近似下：
$$\Delta \mathcal{R}_i \approx 2\theta_i^2 H_{ii}$$

如果$H_{ii}$在同一层内近似常数（empirical上early conv layer成立），那选最大$|\theta_i|$就是最大化damage的greedy策略。

更robust的版本：如果$H \succeq \mu I$（Hessian半正定），则
$$\Delta \mathcal{R} \geq 2\mu \sum_{i \in S} \theta_i^2$$

选最大magnitude等于最大化一个certified lower bound。这不是"理论上最优"，但给出了为啥这么简单的heuristic能work的数学直觉。

### 2. 瞄准前10层

这是最反直觉但最关键的insight。

直觉上你会想：最后几层离分类器近，应该最critical。但empirical完全相反——**前几层最致命**。

为什么？因为信息是层级流动的。早期层是"地基"，处理所有后续输入。早期一个feature detector（比如Sobel-like edge detector）被破坏，所有下游representation都基于错误输入，error逐层compounding。

paper的Figure 1特别直观：一个RegNetY-400MF的第一层卷积核，原本是水平edge detector。Flip其中一个高magnitude权重的sign，整个kernel的功能就废了——feature map完全变形。Dalmatian的图片，特征提取错了，后面所有层都基于错的特征，最终预测崩溃。

这跟神经科学里"早期lesion致盲"（[Kandel Principles of Neural Science](https://books.google.co.il/books?id=yzEFK7Xc87YC)）类似。视网膜损伤致全盲，V1区损伤致皮质盲，但高级视觉区损伤只导致特定agnosia（不认脸但认得物体之类）。DNN的层级信息处理和生物视觉有convergent evolution的味道。

Lipschitz composition给个数学motivation：layer $\ell$的Lipschitz constant $L_\ell$，perturbation最坏放大$\prod_{\ell > 1} L_\ell$。早期perturbation被所有后续层放大。但这只是motivation，paper明确说不是proof。

最有意思的case是ShuffleNetV2。它的最大magnitude weights集中在后期，所以naive magnitude attack效果差。但一旦你redirect到前几层，2 flips就能99.6% AR。这说明early-layer prior比magnitude prior更universal。

### 3. 一个kernel只flip一个

这个insight专门针对CNN，特别elegant。

观察：同一个kernel内flip多个bit，效果会互相抵消。为什么？

看个Sobel-like edge detector：kernel长这样
```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

左右两列opposite sign。Flip左边一个权重（比如-1变1），kernel变成：
```
[ 1  0  1]
[-2  0  2]
[ 1  0  1]
```

这个kernel基本废了，不再是edge detector。

但如果再flip右边一个（比如1变-1）：
```
[ 1  0 -1]
[-2  0  2]
[ 1  0 -1]
```

注意——这其实又变回了一个edge detector，只是方向反了！它依然能检测edges，只是方向翻转。Functional capacity没被破坏，只是orientation变了。

数学上更精确：对卷积响应$y = w^\top x$，两个flip at $i, j$：
$$\Delta y = -2(w_i x_i + w_j x_j)$$

如果$w_i w_j < 0$（opposite-signed lobes）且patch局部相关（$\Sigma_{ij} > 0$，natural images成立），则
$$\mathbb{E}[(\Delta y)^2] = 4(w_i^2 \Sigma_{ii} + w_j^ \Sigma_{jj} + 2 w_i w_j \Sigma_{ij})$$

cross-term $2 w_i w_j \Sigma_{ij} < 0$，部分抵消第一个flip的damage。所以同kernel内多flip是低效的。

策略：每个kernel至多flip一个，spread across更多kernels，破坏更多features。

这个insight是CNN-specific的，transformer里不用（因为没卷积kernel结构）。

## 1P-DNL：加一次forward/backward

DNL已经很强，但加一次random input上的forward+backward pass能进一步refine。

定义hybrid score：
$$\mathcal{S}(\theta_i) = \alpha|\theta_i| + \beta\left|\frac{\partial \mathcal{R}}{\partial \theta_i}\theta_i + \frac{1}{2}H_{ii}\theta_i^2 + \sum_{j \neq i} H_{ij}\theta_i\theta_j\right|$$

变量：
- $\alpha, \beta$：调权系数，paper里都设1
- $\mathcal{R}(\theta) = \sum_i f_\theta(X)[i]$：random input上输出之和
- $H$：Hessian
- $H_{ii}$：Hessian对角
- $H_{ij}$：Hessian off-diagonal

简化：
- $H_{ij} = 0$ for $j \neq i$（对角近似，经典OBD用过）
- $H_{ii} \approx g_i^2$（Gauss-Newton近似，等于Fisher info matrix对角）

退化成：
$$\mathcal{S}(\theta_i) = |\theta_i| + |\theta_i g_i + \frac{1}{2}\theta_i^2 g_i^2|$$

这个score融合了magnitude（一阶信息）和Taylor saliency（二阶信息）。Paper在Appendix D做了ablation，比较magnitude、GraSP、SynFlow、OBD等，发现hybrid最robust——因为不同architecture对不同signal敏感，hybrid始终work。

为啥要用random input而不是real data？因为threat model不让用real data。但random input上算的gradient依然能提供有用的saliency signal——这是因为网络的局部几何对input分布不极度敏感，random input也算个noisy estimate。

注意这个score跟pruning里的Taylor importance（[Molchanov 2017](https://openreview.net/forum?id=SJGCiw5gl)）形式一样。Pruning想找"可以删的"，attack想找"必须保留的"——是镜像问题。

## 跨domain的vulnerability

### Vision Classification

48个ImageNet模型。亮点结果：
- ResNet-50：1P-DNL 1 flip → 99.4% AR
- VGG-11：DNL 3 flips → 99.9%
- MobileNet-V2：2 flips → 99.8%
- ViT-B/16：4 flips → 99.1%

跨架构都成立。更重要的：random flips 10万个bit，很多模型accuracy几乎不掉——证实"绝大多数参数是redundant的"。这是Lottery Ticket Hypothesis（[Frankle & Carbin 2018](https://arxiv.org/abs/1803.03635)）的反面证据：importance mass高度集中。

模型size也不影响（Figure 14, 15）：ResNet-18到152、RegNet不同GFLOPs、EfficientNet B0-B7、ConvNeXt、ViT——都类似collapse。这说明vulnerability是trained DNN的fundamental性质，不是小模型的bug。Bigger model只是更多redundancy包着critical core，critical core本身依然稀疏。

### Reasoning LLMs

最有意思的结果。在MATH-500上：

Qwen3-30B-A3B（MoE模型）：DNL 2 flips → 78% → 0% accuracy。

两个flip分别打在layer 3 expert 82和layer 1 expert 68的down_proj上。MoE里每个token只route到少数experts，被attack的expert在整个response上只在4.14%的tokens上被route。但response还是完全崩了。

为啥？因为corrupted hidden state通过self-attention propagate。即便后续token不route到那个expert，attention已经"看到"前面被poison的token representation，corruption就这么传下去了。

这是这篇paper一个相当深的finding：**MoE的sparsity不提供robustness**。你以为稀疏routing能"隔离"故障，实际上attention机制会把corruption broadcast到所有token。

Figure 5展示的corrupted generation很有意思：模型开始重复boilerplate（"I am a student..."）或者nonsense。不是数学错误，是generation process本身的stability被破坏了。所以这种corruption mode大概率不止影响MATH-500，会transfer到其他generation benchmark。

注意LLMs对exponent flip更敏感——single targeted exponent flip直接把三个模型砸到0%。Random exponent flip单次就把Qwen3-30B-A3B砸到6%。LLM的attention机制对hidden state scale极敏感，exponent flip引入极端rescaling直接毁掉computation。

### Object Detection & Segmentation

COCO上，只攻击backbone，不碰task head：

- Mask R-CNN / ResNet-50：1 flip → bbox AP 0.38 → 0.01（97.36% AR）；2 flips → 0
- Mask R-CNN / ResNet-101：1 flip → 0.40 → 0.01
- YOLOv8-seg：1 flip → bbox AP 0.33 → 0.05（83.66% AR）

Figure 8的qualitative对比特别informative：

**Mask R-CNN-R101**：仍能segment object with high fidelity，但assign wrong semantic class。Localization保留，semantics corrupted。这非常危险——backbone是representation extractor，head是task-specific decoder。Backbone被破坏导致semantic representation错，但head依然能基于错的representation做精确的localization。

**YOLOv8-seg**：完全检测不到dog，反而在tail上"hallucinate"一个bird detection。Dense prediction结构对backbone corruption更敏感，failure mode完全不同。

这两种failure mode对safety implications不同：Mask R-CNN的"看似正常但语义错"比YOLO的"明显失败"更危险，因为更难detect。

### Text Encoders on GLUE

BERT/DistilBERT/RoBERTa on MRPC/QNLI/SST-2。mAR(10)范围70%-83%。证明vulnerability不止autoregressive generation，encoder-based classification同样脆弱。

有意思的是DistilBERT（distilled、更小）反而最vulnerable（SST-2 mAR 83.07%）。可能因为distillation压缩了redundancy，让剩下的参数更"critical"。

## 跟prior work对比

| 方法 | 需要data? | 需要optimization? | ResNet-50 flips→AR |
|---|---|---|---|
| BFA ([Rakin 2019](https://ieeexplore.ieee.org/document/9012120)) | 是 | 是（iterative） | 5 → 99.7% |
| DeepHammer ([Yao 2020](https://www.usenix.org/conference/usenixsecurity20/presentation/yao)) | 是 | 是 | 23* → 75.4% |
| ZeBRA ([Park 2021](https://arxiv.org/abs/2105.04049)) | 否（但生成synthetic） | 是 | 5 → 99.7% |
| **DNL** | 否 | 否 | 1 → 6.6%, 8 → 99.7% |
| **1P-DNL** | 否 | 1次pass | 1 → 99.4% |

DNL/1P-DNL是唯一同时满足"不要data + 不要optimization"的，效果还最好或并列最好。

复杂度对比（Table 7）：
- BFA/DeepHammer/ZeBRA：$\mathcal{O}(k \times B \times \theta \times m)$
- DNL/1P-DNL：$\mathcal{O}(\theta) + \mathcal{O}(k)$

差了几个数量级。这意味着DNL在real-time部署场景下actually feasible，prior art理论上需要很多forward/backward pass，真实黑进系统时根本不实用。

## 防御：选择性保护

最有意思的防御思路：既然critical parameters这么少，那就只保护这些。

Table 5：
- ResNet-18：no defense → BFA AR(10) = 88.87%；保护0.001%（100个参数）→ 58.83%；保护1%（10万个）→ 0.00%
- ResNet-50：no defense → 93.87%；保护0.001%（250个）→ 39.08%；保护1%（25万个）→ 1.30%

只保护千分之一的参数就把BFA效果砍半；保护1%几乎完全中和。这反向证实DNL确实精准识别了critical parameters——BFA通过exhaustive search找的就是这些。

Figure 16 vs 17的对比更直观：
- Selective protect 1%-20% high-scoring bits：100K random flips的damage大幅降低
- Random protect 20% bits：几乎无效

**Which bits are protected 比 how many 重要**。这跟attack的insight完全镜像：少数bits是critical的，剩下的都是redundant。

现有防御被bypass：
- **DeepNcode**（[Velcicky 2024](https://arxiv.org/abs/2405.13891)）：用Hamming distance > 1的codeword编码weight，decoder自动correct单bit error。但gray-box下，attacker可以flip encoded bits让decoded value变相反sign——绕过correction。
- **Weight scaling**（[Fuengfusin 2024](https://arxiv.org/abs/2411.18993)）：存$c\theta$，推理除以$c$。对additive perturbation有效。但sign flip是multiplicative：$\frac{-c\theta}{c} = -\theta$，完全无效。
- **Binarization**：Binary ResNet-18 AR(10) = 96.50%，几乎无保护——sign flip直接invert weight。

## 我的intuition和理解

**为啥训练后的DNN必然importance集中？**

SGD/Adam优化出来的solution从来不是uniform的。Optimization会找到sparse critical subnetwork（[Frankle & Carbin 2018](https://arxiv.org/abs/1803.03635)），剩下的是"陪跑"参数。这种concentration似乎是SGD的emergent property，跟overparameterization关系不大——bigger model不更robust，只是更多redundancy包着同样稀疏的critical core。

**为啥早期层最critical？**

早期层是地基。Sobel/Gabor features是所有后续representation的building blocks。这些primitive features被破坏，所有高层semantic都基于错的输入。这跟神经科学的Hubel & Wiesel发现一致：视网膜/LGN/V1的hierarchical processing，早期lesion致盲，晚期lesion致specific agnosia。DNN在这个意义上有convergent evolution。

**为啥MoE的sparsity不提供robustness？**

这是paper最反直觉的finding之一。直觉上：MoE每个token只route到少数expert，attack一个expert应该只影响那个expert的tokens。但实际上corrupted hidden state通过self-attention broadcast到所有后续token——即便后续token不route到那个expert，attention已经看到了前面的poison。

这对MoE-based LLM的safety analysis很重要：sparse routing ≠ fault isolation。Attention是global的，任何corruption都会传播。

**Pruning和attack的duality**

这是paper的intellectual depth所在。Pruning研究几十年"which weights can be removed"，attack研究"which weights must be kept"。两者saliency criterion完全mirror：
- Magnitude pruning ↔ DNL
- Taylor/OBD pruning ↔ 1P-DNL
- GraSP、SynFlow ↔ Appendix D的variants

Pruning saliency = attack criticality。两端是同一spectrum。这暗示未来pruning和attack literature应该更紧cross-pollinate。

**Failure modes的多样性**

不同架构的failure mode不同：
- Vision classification：collapse到随机精度
- Mask R-CNN：保留localization，corrupted semantics（极危险——plausible but wrong）
- YOLOv8-seg：完全失败+hallucinated detection
- Reasoning LLMs：degenerate到repetitive boilerplate
- Text encoders：accuracy drop

这种多样性说明：vulnerability是structural的，但failure mode是architecture-specific的。Safety analysis需要per-architecture考虑。

**为啥model size不mitigate？**

Figure 14, 15显示5个model family不同capacity都类似vulnerable。这暗示vulnerability不是"小模型不够robust"，是trained DNN的fundamental性质。Scaling laws在robustness维度不帮我们——bigger LLMs同样flip一两个sign就崩。

**这篇paper真正的contribution**

不是"提出了新attack"——bit flip attack已经有人做过。是**重新定义了weight-space attack的lower bound**：在strictest possible threat model下（no data, no optimization, or single pass），仍catastrophic。这把weight integrity提升到first-class security concern。

同时显式establish了**pruning-attack duality**：把两个看似无关的subfield unify到同一个saliency framework下。这个unifying视角是真正深刻的contribution。

**为啥这篇paper让我兴奋**

它用一个极简的heuristic揭示了一个fundamental事实：trained DNN的functional capacity是power-law distributed的。极少数"hub"参数承担大部分load，绝大多数是supporting infrastructure。这种distribution在复杂系统里ubiquitous——互联网、社交网络、生物网络、经济系统——DNN也不例外。

更exciting的是：这个事实既enable pruning（删non-critical），也enable attack（破坏critical）。一个硬币两面。这意味着任何用SGD训练的网络都inherently vulnerable——除非我们改变训练过程本身，explicitly鼓励importance distribution更uniform。但这可能hurt generalization（pruning literature暗示sparse importance是trained DNN的特性）。

这是一个深度学习的fundamental tension，paper没解但pointed out了。

## 一些开放问题

1. **Quantized models (INT8, FP8, BF16)**：FP32的sign bit在INT8（2's complement）里不存在。BF16仍有sign bit但mantissa不同。DNL在不同numeric format上的efficacy是个fascinating实验问题。INT8的"sign flip"对应什么操作？

2. **Training-time defense**：能否在训练时explicitly minimize importance集中度？比如regularizer鼓励importance distribution更uniform。但这可能hurt generalization——pruning literature暗示sparse importance是trained DNN的特性，强行distribute可能让模型表达力下降。

3. **MoE的routing-aware defense**：能否detect corrupted expert output并mask？但paper显示corruption通过attention传播，即便expert不被route也derail整个response。单纯的routing-level defense可能不够，需要attention-level isolation。

4. **Critical parameter detection作为interpretability工具**：DNL可以作为一种model analysis工具——哪些参数最critical揭示model的functional structure。这跟Anthropic的mechanistic interpretability（circuits、induction heads、IOI circuits）的cross-pollination是promising direction。Critical parameters对应哪些circuits？

5. **Adversarial training against sign flips**：能否训练模型使sign flips不catastrophic？类似adversarial training for input perturbations。但weight-space的adversarial training计算成本高（每步要sample flip configurations）。

6. **Hardware mitigations的efficacy**：ECC memory、Rowhammer防御（TRR）等。Paper显示software-level selective protection已经极有效，hardware-level可能更省事但cost更高。

7. **Differential access threat models**：在distributed serving、federated learning中attacker只能access部分weights的情景。Paper的limitation提到这点，但没实验。

8. **Connection to mechanistic interpretability**：critical parameters跟anthropic-style的circuits（[Anthropic Circuits](https://transformer-circuits.pub/)）有什么对应？如果某个induction head的某个weight是DNL-critical，那这个weight在circuit里扮演什么角色？这种mapping可能双向enrich两个领域。

## 参考链接

- Optimal Brain Damage: https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf
- Optimal Brain Surgeon: https://proceedings.neurips.cc/paper_files/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf
- Lottery Ticket Hypothesis: https://arxiv.org/abs/1803.03635
- Rethinking Pruning: https://arxiv.org/abs/1810.05270
- BFA (Rakin 2019): https://ieeexplore.ieee.org/document/9012120
- DeepHammer (Yao 2020): https://www.usenix.org/conference/usenixsecurity20/presentation/yao
- Rowhammer (Project Zero): https://googleprojectzero.blogspot.com/2015/03/exploiting-dram-rowhammer-bug-to-gain.html
- timm: https://github.com/rwightman/pytorch-image-models
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- SynFlow (Tanaka 2020): https://proceedings.neurips.cc/paper/2020/file/46a4378f835dc8040c8057beb6a2da52-Paper.pdf
- GraSP (Wang 2020): https://openreview.net/forum?id=SkgsACVKPH
- SNIP (Lee 2019): https://openreview.net/forum?id=B1VZjDXcF
- Molchanov Taylor Pruning: https://openreview.net/forum?id=SJGCiw5gl
- Yosinski Transferability: https://papers.nips.cc/paper/2014/hash/7acd6a8d870b14b8a5b8e43e8a0f5aeb-Abstract.html
- DeepNcode: https://arxiv.org/abs/2405.13891
- Weight Scaling Defense: https://arxiv.org/abs/2411.18993
- Lipschitz Networks (Gouk): https://link.springer.com/article/10.1007/s10994-020-05929-w
- Kandel: https://books.google.co.il/books?id=yzEFK7Xc87YC
- Anthropic Circuits: https://transformer-circuits.pub/

Karpathy，这paper真就是用三个simple heuristic揭示了一个deep fact。简单背后是trained DNN的fundamental性质——importance集中、early layer disproportionate、kernel-internal cancellation。Pruning literature几十年累积的saliency criterion直接mirror到attack端，这或许是paper最深的contribution。

---

# Deep Neural Lesion: 一篇关于 DNN Sign-Bit Flip Vulnerability 的深度解析

Karpathy, 这篇 paper 我读得非常仔细。下面我把它的核心 idea、技术细节、直觉,以及与 pruning、neuroscience、floating-point arithmetic 的联系都讲清楚。

## 1. 核心问题与 Threat Model

这篇文章问了一个非常 sharp 的问题:**一旦 attacker 获得对 model parameters 的 write access,他需要多少 access 和 computation 才能摧毁一个 DNN?** 答案令人不安 — 极少。

Threat model 极其严格(对 attacker 而言约束极强):
- (i) attacker 没有任何 data samples,无法访问 $P(X)$ 或 $P(Y)$
- (ii) attacker 不能 evaluate model on any input(no forward/backward passes),1P-DNL 版本稍微 relax,允许一个 random input 上一次 forward + 一次 backward pass

攻击目标形式化:
$$\min_k \max \mathbb{E}_{(X,Y)\sim \mathcal{D}}\left[\mathcal{L}(f_{\theta'_{(k)}}(X), Y)\right]$$

其中 $\theta'_{(k)}$ 是对 $\theta$ 的 B 个 memory bits 中翻转 k 个所得。$\text{bits}(\theta) \in \{0,1\}^B$ 表示 IEEE-754 编码。

实现 bit flip 的硬件/软件向量包括 rootkit、firmware exploit、DMA from Thunderbolt/FireWire、Rowhammer(参考 [Rowhammer Project Zero blog](https://googleprojectzero.blogspot.com/2015/03/exploiting-dram-rowhammer-bug-to-gain.html))、GPU cache tampering、voltage/frequency glitching。

## 2. IEEE 754 FP32 表示与攻击位的选择

IEEE 754 FP32:
$$(-1)^s \times 2^{(e-127)} \times \left(1 + \frac{m}{2^{23}}\right)$$

- $s$: 1 sign bit
- $e$: 8 exponent bits
- $m$: 23 mantissa bits

Sign bit flip: $\theta_i \to -\theta_i$,magnitude 不变,sign 直接 negate。
Exponent MSB flip: 极端 rescaling(可能从 $10^{-3}$ 跳到 $10^{3}$ 量级)。

在 vision models 中,sign flip 比 exponent flip 更 selective、更稳定地破坏 features(详见 paper Appendix C Table 8)。而在 LLMs 中,exponent flip 反而更 destructive,因为 LLM 对 hidden state 的 scale 极其敏感。这种 domain-dependent 现象本身就是一个有意思的 finding。

## 3. 三个核心 Heuristics 及其直觉

### 3.1 Magnitude-Based Selection

$$S(\theta_i) = |\theta_i|$$

这个看似 trivial 的 score 背后有非常深的 theory。理论 motivation 来自 Optimal Brain Damage (OBD, [LeCun et al. 1989](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)) 和 Optimal Brain Surgeon (OBS, [Hassibi et al. 1992](https://proceedings.neurips.cc/paper_files/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf))。

对 trained network 的 loss $\mathcal{R}(\theta)$ 做二阶 Taylor 展开:
$$\Delta \mathcal{R} \approx g^\top \Delta\theta + \frac{1}{2}\Delta\theta^\top H \Delta\theta$$

其中:
- $g = \nabla_\theta \mathcal{R}(\theta)$ 是 gradient
- $H$ 是 Hessian
- $\Delta\theta$ 是 parameter perturbation

在 convergence 处 $g \approx 0$,所以 curvature 项 dominate。Flip sign of $\theta_i$: $\Delta\theta_i = -2\theta_i$,其他坐标不变。在 diagonal Hessian 近似下:
$$\Delta \mathcal{R}_i \approx \frac{1}{2}(-2\theta_i)^2 H_{ii} = 2\theta_i^2 H_{ii}$$

给定 budget k flips,greedy maximizer 就是选取 k 个 largest $\theta_i^2 H_{ii}$。

关键 empirical observation: 在早期 convolutional layers 中,$H_{ii}$ 近似 constant。这种情况下 criterion 退化为 largest $|\theta_i|$,正是 DNL 的 zero-pass criterion。

更 robust 的 version: 若 $H \succeq \mu I$(Hessian 半正定,bounded below by $\mu I$),则
$$\Delta \mathcal{R} \geq 2\mu \sum_{i \in S} \theta_i^2$$

选择最大 magnitude 等价于最大化 loss damage 的 certified lower bound。这是一个相当强的 theoretical guarantee,即便 $\mu$ 未知。

### 3.2 Early-Layer Targeting

直觉来自神经科学:早期 lesions(retina、optic nerve)导致 severe 或 total blindness([Kandel et al. Principles of Neural Science](https://books.google.co.il/books?id=yzEFK7Xc87YC))。类比 DNN,early convolutional filters 编码 generic edge/texture features (Sobel-like, Gabor-like),disrupting 它们会 degrade 所有 downstream representations。Figure 1 直观展示:flip 一个 Sobel-like kernel 中单个 high-magnitude weight 的 sign,整个 feature map 完全变形。

数学 motivation(Lipschitz composition bound):对 layer $\ell$ 的 Lipschitz constant $L_\ell$,perturbation 的 worst-case amplification 最多是 $\prod_{\ell > 1} L_\ell$。早期 perturbation 被所有后续 layer 处理。([Burago et al. 2001](https://www.ams.org/books/gsm/033/); [Gouk et al. 2021](https://link.springer.com/article/10.1007/s10994-020-05929-w); [Virmaux & Scaman 2018](https://papers.nips.cc/paper/2018/hash/7e0a81791dbebf4dd38c8e0a1e67cf61-Abstract.html))

Empirical evidence 强烈支持:Figure 4a 显示 targeting 前 l 层($l \in [1, 10]$)consistently 比 random 或 naive magnitude 更 destructive。Table 4b 对 ShuffleNetV2 的细化分析特别 informative:
- All layers: AR(5) = 39%
- First 10 layers: AR(2) = 99.6%, AR(5) = 99.8%
- First 2 layers: AR(2) = 99.6%

ShuffleNetV2 的 largest weights 集中在后期,所以 naive magnitude attack 效果差。Redirect 到 early layer 立即 catastrophic。这暗示 architectural quirks 会 naive 偏 magnitude 的方法,但 early-layer prior 是 universal 的。

### 3.3 One-Flip-Per-Kernel Constraint (CNN-specific)

这是对 CNN architecture 最重要的 insight。观察:同一 kernel 内多次 flip 会 partially offset。

数学分析:对 convolution kernel response $y = w^\top x$,两个 flip at indices $i, j$:
$$\Delta y = -2(w_i x_i + w_j x_j)$$

如果 $i, j$ 在 opposite-signed lobes($w_i w_j < 0$),且 patch entries locally correlated($\Sigma_{ij} > 0$,这在 natural images 上成立),则 cross-term negative:
$$\mathbb{E}[(\Delta y)^2] = 4(w_i^2 \Sigma_{ii} + w_j^2 \Sigma_{jj} + 2 w_i w_j \Sigma_{ij})$$

第二个 flip 的 cross-term $2 w_i w_j \Sigma_{ij} < 0$,partially cancels 第一个 flip 的 squared perturbation,而不是 compound。这非常符合 Sobel-like filter 的结构 — opposite-signed lobes 是 edge detector 的特征([Yosinski et al. 2014](https://papers.nips.cc/paper/2014/hash/7acd6a8d870b14b8a5b8e43e8a0f5aeb-Abstract.html))。

所以策略:每个 kernel 至多一个 flip,spreading across more kernels amplifies overall impact。这个 heuristic 是 CNN-specific,不用于 transformers。

## 4. DNL (Pass-free) Algorithm

```
Algorithm 1: DNL
Input: θ, k (flips), L (layers)
1. θ_L ← first L layers' parameters
2. Sort θ_L by |θ_i| descending
3. K ← top-k
4. For CNNs: enforce ≤ 1 entry per kernel
5. For each θ_i in K: θ_i ← -θ_i
Output: modified θ
```

L = 10 是 default。Complexity: $\mathcal{O}(\theta) + \mathcal{O}(k)$ — 几乎 free。这与 prior work (BFA $\mathcal{O}(k \times B \times \theta \times m)$, DeepHammer 同类)相比,attacks 都不需要任何 forward/backward pass。

## 5. 1P-DNL (Single-Pass) Algorithm

当允许一次 forward + backward pass on random input,引入 hybrid importance score:

$$\mathcal{S}(\theta_i) = \alpha|\theta_i| + \beta\left|\frac{\partial \mathcal{R}}{\partial \theta_i}\theta_i + \frac{1}{2}H_{ii}\theta_i^2 + \sum_{j \neq i} H_{ij}\theta_i\theta_j\right|$$

变量解释:
- $\alpha, \beta$: tunable coefficients(本文设 $\alpha = \beta = 1$)
- $\mathcal{R}(\theta) = \sum_i f_\theta(X)[i]$:random input 上 output 之和(对 vision 是 Gaussian noise image,对 LLM 是 random token sequence)
- $H$: Hessian of $\mathcal{R}$
- $H_{ii}$: Hessian 对角元素
- $H_{ij}$: Hessian off-diagonal 元素

简化:
- $H_{ij} = 0$ for $j \neq i$ — diagonal approximation,经典 OBD 用过
- $H_{ii} \approx g_i^2$ — Gauss-Newton approximation(Fisher information matrix 的对角)

于是 score 退化为:
$$\mathcal{S}(\theta_i) = |\theta_i| + \left|\theta_i g_i + \frac{1}{2}\theta_i^2 g_i^2\right|$$

$\alpha = 0$ 时纯 second-order(OBD-like); $\beta = 0$ 时纯 magnitude。Hybrid 在不同 architecture 间最 robust(Figure 9 的 ablation)。

值得注意的是,这等于 Taylor pruning saliency $\propto |\theta_i g_i|$ 的 adversarial analogue([Molchanov et al. 2017](https://openreview.net/forum?id=SJGCiw5gl); [Lee et al. SNIP 2019](https://openreview.net/forum?id=B1VZjDXcF); [Wang et al. GraSP 2020](https://openreview.net/forum?id=SkgsACVKPH); [Tanaka et al. SynFlow 2020](https://proceedings.neurips.cc/paper/2020/file/46a4378f835dc8040c8057beb6a2da52-Paper.pdf))。Pruning 和 sign-flip attack 是同一 saliency spectrum 的两端。

## 6. 实验结果深度分析

### 6.1 Image Classification:60 models, 48 ImageNet

| Model | Method | Flips → AR (%) |
|---|---|---|
| VGG-11 | DNL | 3 → 99.9 |
| VGG-11 | 1P-DNL | 2 → 99.8 |
| ResNet-50 | DNL | 1 → 6.6, 8 → 99.7 |
| ResNet-50 | 1P-DNL | 1 → 99.4 |
| MobileNet-V2 | DNL | 2 → 99.8 |
| MobileNet-V2 | 1P-DNL | 2 → 99.9 |
| ViT-B/16@224 | DNL | 5 → 99.3 |
| ViT-B/16@224 | 1P-DNL | 4 → 99.1 |

注意 ResNet-50:1P-DNL 仅 1 flip 就 99.4% AR,这是非常 extreme 的结果。对比 BFA [Rakin et al. 2019](https://ieeexplore.ieee.org/document/9012120) 需要 5 flips 才到 99.7%,且需要 data + iterative optimization。

Figure 2a 显示 random flips 即便 100K 个,很多模型 accuracy 不显著下降 → **绝大多数参数 non-critical**。这是 lottery ticket hypothesis([Frankle & Carbin 2018](https://arxiv.org/abs/1803.03635))的另一面证据:importance mass 高度集中。

Figure 2c:10 flips 下,43/48 ImageNet models AR > 60%。

模型 size 不影响 vulnerability(Figure 14, 15):ResNet family 从 18 到 152,RegNet 不同 GFLOPs,EfficientNet B0-B7,ConvNeXt,ViT — 都类似 collapse level。这暗示 vulnerability 是 DNN 训练 dynamics 的 fundamental 性质,而非 architecture quirk。

### 6.2 Reasoning LLMs on MATH-500

最 striking 的结果之一,Table 1:

| Model | Targeted Layers | DNL Flips → AR | 1P-DNL Flips → AR |
|---|---|---|---|
| Qwen3-30B-A3B | First 5 blocks | 2 → 100.0 | 1 → 71.8, 4 → 100.0 |
| Qwen3-4B | First 5 blocks | 30 → 2.3 | 28 → 95.3 |
| Nemotron Nano 8B | First 5 blocks | 3 → 100.0 | 17 → 100.0 |

Qwen3-30B-A3B 是 MoE 模型 — 每个 token 只 route 到 small subset of experts。DNL 的两个 flip 分别 target layer 3 expert 82 和 layer 1 expert 68 的 down_proj。仅这两个 flip,78% → 0%。

为什么 MoE 这么脆弱?即便被 attack 的 expert 在整个 response 上只在 4.14% 的 tokens 上被 route,corrupted hidden state 通过 attention 机制 propagate forward,poisons 后续所有 token representations。Figure 5 的示例 generation 显示 model 在 attack 后陷入 repetitive boilerplate("I am a student...")或 nonsensical text。

这与 vision models 的 early-layer intuition 一致:corruption 一旦进入 hidden state,会通过 self-attention compounding。

Random sign flips 远弱于 targeted:Qwen3-30B-A3B 27 random flips 后还 70% accuracy。但 random exponent flip 单次就把 Qwen3-30B-A3B 砸到 6% — exponent attack 对 LLM 极度 selective 不重要,因为极端 rescaling 几乎任何 parameter 都致命。

### 6.3 Object Detection & Instance Segmentation (COCO)

| Model / Backbone | Metric | Baseline | k=1 | AR(1) | k=2 | AR(2) |
|---|---|---|---|---|---|---|
| Mask R-CNN / ResNet-50 | bbox AP | 0.38 | 0.01 | 97.36 | 0.00 | 100 |
| Mask R-CNN / ResNet-50 | segm AP | 0.35 | 0.00 | 100 | 0.00 | 100 |
| Mask R-CNN / ResNet-101 | bbox AP | 0.40 | 0.01 | 97.51 | 0.01 | 97.51 |
| YOLOv8-seg | bbox AP | 0.33 | 0.05 | 83.66 | 0.05 | 86.33 |

Attack 只针对 backbone,task-specific heads 不动。Figure 8 的 qualitative 结果特别有意思:

- **Mask R-CNN-R101**:仍能 segment object with high fidelity,但 assign wrong semantic class。Localization 保留,semantics corrupted。这非常符合 "backbone 是 representation extractor,head 是 task-specific decoder" 的分工。这种 failure mode对 safety 很危险 — 看起来 plausible 但 semantically wrong。
- **YOLOv8-seg**:complete object-level failure,hallucinated bird detection on dog's tail。Failure mode 不同,可能因为 YOLO 的 dense prediction structure。

### 6.4 Text Encoders on GLUE

BERT/DistilBERT/RoBERTa on MRPC/QNLI/SST-2:

| Model | Task | Baseline | mAR(10)% |
|---|---|---|---|
| BERT | SST-2 | 93.16% | 82.43 |
| DistilBERT | SST-2 | 91.21% | 83.07 |
| RoBERTa | MRPC | 91.18% | 69.99 |

mAR(10) 范围 69.99%-83.07%,证明 vulnerability 不仅限于 autoregressive generation,encoder-based classification 同样脆弱。DistilBERT(更小、distilled)反而最 vulnerable,可能因为 distillation 压缩了 redundancy。

## 7. 与 Prior Bit-Flip Attacks 对比

Table 4 & 6 对比:

| Method | OF | DA | ResNet-50 | VGG-11 | MobileNet-V2 | ViT-B/16 |
|---|---|---|---|---|---|---|
| BFA (Rakin 2019) | ✗ | ✗ | 5 → 99.7 | 17 → 99.7 | 3 → 99.8 | 5 → 30.1, 10 → 90.9 |
| DeepHammer (Yao 2020) | ✗ | ✗ | 23* → 75.4 | — | 2* → 99.8 | — |
| ZeBRA (Park 2021) | ✗ | ✓ | 5 → 99.7 | 8 → 99.8 | 2 → 99.7 | 5 → 5.1, 10 → 45.8 |
| DNL (ours) | ✓ | ✓ | 1 → 6.6, 8 → 99.7 | 3 → 99.9 | 2 → 99.8 | 5 → 99.3 |
| 1P-DNL (ours) | ✓ | ✓ | 1 → 99.4 | 2 → 99.8 | 2 → 99.9 | 4 → 99.1 |

OF = optimization-free, DA = data-agnostic。DNL/1P-DNL 是唯一同时满足 OF + DA 的方法,且效果 ≥ prior art,常常 fewer flips。

Complexity (Table 7):
- BFA / DeepHammer / ZeBRA: $\mathcal{O}(k \times B \times \theta \times m)$
- DNL / 1P-DNL: $\mathcal{O}(\theta) + \mathcal{O}(k)$

ZeBRA 通过 generate synthetic data 绕开 real data,但本质上仍是 iterative optimization。DNL 完全没有任何 optimization。

## 8. 防御机制

### 8.1 Selective Defense

核心 insight:只有极少数 sign bits 是 catastrophic 的。Protect 这些 high-scoring weights 就够。

Method:用 DNL identify critical parameters,然后对这些少数用 bit replication 或 ECC(Hamming codes, [Peterson & Weldon 1972](https://mitpress.mit.edu/9780262160391/error-correcting-codes/))。

Table 5:

| Model | # Defended | BFA AR(10) |
|---|---|---|
| ResNet-18 | No Defense | 88.87 |
| ResNet-18 | ~0.001% (100 params) | 58.83 |
| ResNet-18 | ~1% (100K params) | 0.00 |
| ResNet-50 | No Defense | 93.87 |
| ResNet-50 | ~0.001% (250 params) | 39.08 |
| ResNet-50 | ~1% (250K params) | 1.30 |

仅 protect 0.001% 的 parameters 就把 BFA 效果 halve;protect 1% 几乎完全 nullify。这说明 DNL 可靠地识别了 BFA 通过 exhaustive search 寻找的同一个 critical parameter set。

### 8.2 现有防御被 bypass

**DeepNcode** ([Velcicky et al. 2024](https://arxiv.org/abs/2405.13891)):把每个 weight 编码成更长 codeword,Hamming distance > 1,decoder 自动 correct 单 bit error。但 gray-box 下(attacker 不知 codebook,但能 observe decoded values),可以 selectively flip encoded bits使 decoded value 变相反 sign — 绕过 correction。

**Weight scaling** ([Fuengfusin & Tamukoh 2024](https://arxiv.org/abs/2411.18993)):存 $c\theta$,推理时除以 $c$。对 additive perturbation 有效。但 sign flip 是 multiplicative:
$$\frac{-c\theta}{c} = -\theta$$
防御完全无效。Empirical 确认 AR 不变。

**Binarization**:Binary ResNet-18 (Table 9):AR(10) = 96.50%。Binarization 假设 weight perturbation 影响小,但 sign flip 直接 invert weight,防御几乎为零。

### 8.3 Random vs Selective Protection

Figure 16 vs 17:protect 1%-20% 的 high-scoring(DNL-selected)bits 显著 reduce damage from 100K random flips。但 protect 20% 的 random bits 几乎无效。**Which bits are protected 比 how many 重要**。

## 9. Weight Score Ablation (Appendix D)

比较多种 scoring functions:

- Magnitude: $|\theta_i|$
- GraSP: $|\theta_i \odot Hg|$ ([Wang et al. 2020](https://openreview.net/forum?id=SkgsACVKPH))
- GraSP (Gauss-Newton): $H \approx g^2$
- SynFlow: $|g \odot \theta_i|$ ([Tanaka et al. 2020](https://proceedings.neurips.cc/paper/2020/file/46a4378f835dc8040c8057beb6a2da52-Paper.pdf))
- OBD: $\frac{1}{2}\theta_i^t H_{ii}\theta_i$
- Hybrid (1P-DNL): $|\theta_i| + |\theta_i g_i + \frac{1}{2}\theta_i^2 g_i^2|$

Finding:某些 models 对 second-order 敏感(OBD 强),对 magnitude robust;另一些相反。Hybrid 在所有 architecture 上最 stable。这反映 magnitude 和 gradient 信息是 complementary 的 signals for criticality。

## 10. 与 Pruning 的深刻联系

这是 paper 隐藏的 intellectual depth。Pruning literature 几十年都在研究 "which weights are important to keep"。DNL 翻转问题:which weights are important to break。

- Magnitude pruning ([Frankle & Carbin 2018](https://arxiv.org/abs/1803.03635); [Liu et al. 2019](https://arxiv.org/abs/1810.05270)) ↔ DNL
- Taylor/OBD pruning ([LeCun et al. 1989](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf); [Molchanov et al. 2017](https://openreview.net/forum?id=SJGCiw5gl)) ↔ 1P-DNL
- GraSP, SynFlow ↔ variants in Appendix D

DNL 把 pruning saliency 用作 attack saliency。Pruning 想 remove least important;attack 想 break most important。两端 spectrum 的 criterion 是镜像的。

Empirical 上,early-layer targeting 在 pruning 中也成立 — 早期层 disproportionately salient。DNL 的 early-layer targeting 是 pruning empirical evidence 的 adversarial counterpart。

Lottery ticket hypothesis 提示 importance mass 高度集中在 sparse subnetwork。DNL 提示 criticality mass 也高度集中 — flip 极少数 bits 就能 catastrophic collapse。这两个 phenomenon 是同一硬币的两面:**trained DNN 的 functional capacity 集中在极少数 parameters,其余是 redundancy**。

## 11. Exponent Bit Attacks 的 Domain Dependence

Vision(Table 8):sign > exponent 在多数 architectures。例:VGG-11 sign AR(10) = 91.8 vs exp AR(10) = 53.89;ViT-B/16 sign 99.84 vs exp 82.38。

LLMs:exponent 远强于 sign。Single targeted exponent flip 直接把三个 reasoning LLMs 砸到 0%。Random exponent flip on Qwen3-30B-A3B k=1 已经 6% accuracy。

直觉:LLM 对 hidden state scale 极敏感(attention scores, softmax, layernorm),exponent flip 引入极端 rescaling 毁掉整个 computation。Vision models 用 ReLU/BN,Gabor features 对 sign 更敏感(改变 edge direction)。

这暗示 quantized models(INT8)可能在 LLM 上对 exponent attack 不再适用,因为 exponent 不存在 — 留作 future work。

## 12. Failure Modes 的多样性

值得 highlight 的 qualitative 观察:

- **Vision classification**:collapse to near-random accuracy,具体 failure mode 取决于哪个 kernel 被 flip(Figure 1 的 Dalmatian 例子)
- **Mask R-CNN**:保留 localization,corrupted semantics(图 8 左)。这是 partial collapse — backbone 破坏但 head 仍工作。Safety 视角极危险:看似 plausible 但 wrong answer
- **YOLOv8-seg**:complete failure + hallucinated detection(图 8 右)。Dense prediction 结构对 backbone corruption 更敏感
- **Reasoning LLMs**:degenerate to repetitive boilerplate(Figure 5)。不是 near-miss 数学错误,是 generation 完全 collapse。这暗示 corruption 不只影响 specific capability,而是破坏了生成过程本身的 stability
- **Text encoders**:accuracy drop,但具体 failure mode 未在 paper 中详述

## 13. Open Questions / Future Directions

Paper 提到的 limitation:DNL 假设 attacker 可以 global search all weights。在 model sharding、partial exposure、compartmentalization 部署下,attack 可能 less effective。

我额外想到的几个方向:

1. **Quantized models (INT8, FP8, BF16)**:FP32 的 sign bit 在 INT8 中不存在(2's complement)。BF16 仍有 sign bit 但 mantissa 不同。Attack 在不同 numeric format 上的 efficacy 是一个 fascinating 实验问题
2. **Training-time defense**:能否在 training 时 explicitly minimize 集中在少数 weights 的 saliency?E.g., regularizer 鼓励 importance distribution 更 uniform?但这可能 hurt generalization(pruning literature 暗示 sparse importance 是 trained DNN 的特性)
3. **MoE 架构的 routing-aware defense**:能否 detect corrupted expert output 并 mask?但 paper 显示 corruption 通过 attention 传播,即便 expert 不被 routed 也 derail 整个 response
4. **Critical parameter detection as interpretability tool**:DNL 可作为一种 model analysis 工具 — 哪些 parameters 最 critical 揭示 model 的 functional structure
5. **Connection to mechanistic interpretability**:critical parameters 与 circuits、induction heads、IOI circuits 等的对应关系?Anthropic-style interpretability + DNL-style criticality detection 的 cross-pollination 是 promising direction
6. **Adversarial training against sign flips**:能否训练模型使 sign flips 不 catastrophic?类似 adversarial training for input perturbations
7. **Hardware-level mitigations**:ECC memory、Rowhammer defenses(TRR)等。但 paper 显示 software-level selective protection 已经非常有效
8. **Differential access threat models**:在 distributed serving、 federated learning 中 attacker 只能 access 部分 weights 的情景

## 14. 与 Model Size 的关系

Figure 14, 15 显示 5 个 model families (ResNet, RegNet, EfficientNet, ConvNeXt, ViT) 不同 capacity 都类似 vulnerable。Model size 不 mitigate vulnerability。

直觉解释:训练 dynamics 似乎 always 产生 importance 集中的 solution,无论 model size。这与 [Frankle & Carbin 2018](https://arxiv.org/abs/1803.03635) 的 lottery ticket observation 一致 — 任何 trained DNN 都有一个 sparse critical subnetwork。Bigger model ≠ more distributed importance;只是更多 redundancy 包着 critical core。

这也暗示 scaling laws 在 robustness 维度可能不帮助我们 — bigger LLMs 同样 flip 一两个 sign 就 collapse。

## 15. 为什么这篇文章重要

从 research 角度:

1. **New threat model**:strictest possible(data-free, optimization-free 或 single-pass),仍 catastrophic。重新定义了 weight-space attack 的 lower bound
2. **Universal vulnerability**:跨 vision/detection/segmentation/LLM/encoder LM 都成立,说明这是 DNN 的 fundamental 性质而非 architecture quirk
3. **Pruning-attack duality**:explicitly connects pruning saliency 与 adversarial criticality,unifying 两个 subfields
4. **Practical defense**:selective ECC protection 只需 0.001%-1% parameters,immediately deployable
5. **MoE vulnerability**:corrupted hidden state propagation through attention 是新 mechanism,可能启发新的 robustness research

从 deployment 角度:

任何 deployed DNN(autonomous driving、medical imaging、LLM serving、recommendation)只要 attacker 获得 parameter write access,极低成本就能 catastrophic。这把 weight integrity 提升到 first-class security concern。

传统 cybersecurity 关心 confidentiality、integrity、availability。DNL 暴露 AI systems 的 integrity dimension 是 fundamentally fragile 的。Rowhammer、DMA、firmware 等 attack vectors 之前主要 concern 是 OS-level compromise,现在直接 translate 到 ML model collapse。

## 16. 个人 Reflections

这篇 paper 让我想到几个 deep 的点:

**On parameter criticality 的 concentration**:
训练后的 DNN 像一个高度 optimized 的 economy — 极少数 "hub" parameters 承担大部分 functional load,大多数 parameters 是 supporting infrastructure。Pruning 删 supporting infrastructure 没事;attack 删 hub 立即 collapse。这种 power-law distribution of importance 在复杂系统中 ubiquitous(internet、social networks、biology),DNN 不例外。

**On early layers 的特殊性**:
早期层是 representation 的 foundation。Gabor/Sobel features 编码 most fundamental visual primitives — 任何破坏都 cascade。这与 neuroscience 的 Hubel & Wiesel 发现一致(retina → LGN → V1 → ... 的 hierarchical feature extraction)。早期 lesions 致盲,后期 lesions 导致 specific agnosias。DNN 在这点上与 biological vision 类似 convergent evolution。

**On MoE 与 sparsity 的 false security**:
MoE 给人一种 "sparsity provides robustness" 的直觉 — 每个 token 只用 few experts,attack 单个 expert 应该只影响那个 expert 的 tokens。Paper 证伪了这种 intuition。Hidden state corruption 通过 attention 传播,即便 expert 只被 routed on 4% tokens,整个 response 仍 collapse。这对 MoE-based LLM 的 safety analysis 是 important finding。

**On the elegance of simple heuristics**:
DNL 的 magnitude + early-layer + one-flip-per-kernel 三个 heuristic 都很 simple,但配合起来 catastrophic。这反映 DNN 的 vulnerability 是 structural 的,任何 suffices to identify the load-bearing parameters 的简单 prior 都能 exploit。Complex attack methods(BFA、DeepHammer)的 complexity 不必要 — simple 已经 optimal-ish。

**On the unity of pruning and attack**:
Pruning 几十年研究 "what to remove" 的 criterion(magnitude、Taylor、GraSP、SynFlow)。Attack 同样的 criterion flip 反向。这是 saliency 的 dual interpretation。未来 pruning 和 attack literature 应该更紧密 cross-pollinate — pruning 的 criterion 直接 translate 到 attack 的 criterion,反之亦然。

## 17. 总结

Deep Neural Lesion 是一个 deceptively simple 但 profound 的 attack。它表明:

1. **几乎 free**:无需 data、无需 optimization(DNL)或单次 pass(1P-DNL)
2. **Catastrophic**:1-4 sign flips 即可砸碎 ResNet-50、Mask R-CNN、Qwen3-30B-A3B
3. **Universal**:跨 vision、detection、segmentation、LLM、encoder LM
4. **Defensible**:selective ECC protection 仅需 0.001%-1% parameters
5. **Theoretically grounded**:与 OBD/OBS pruning 的二阶 Taylor expansion 直接连接
6. **Pruning-attack duality**:pruning saliency = attack criticality,两端 spectrum

Paper 的 broader significance 在于揭示了 trained DNN 的 functional capacity 极度集中在少数 parameters,这既是 pruning 的基础也是 attack 的 vector。这种 concentration 似乎是 SGD/Adam 训练的 fundamental emergent property。

## 参考链接

- Optimal Brain Damage: https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf
- Optimal Brain Surgeon: https://proceedings.neurips.cc/paper_files/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf
- Lottery Ticket Hypothesis: https://arxiv.org/abs/1803.03635
- Rethinking Pruning: https://arxiv.org/abs/1810.05270
- BFA (Rakin et al. 2019): https://ieeexplore.ieee.org/document/9012120
- DeepHammer (Yao et al. 2020): https://www.usenix.org/conference/usenixsecurity20/presentation/yao
- Rowhammer (Project Zero): https://googleprojectzero.blogspot.com/2015/03/exploiting-dram-rowhammer-bug-to-gain.html
- timm (PyTorch Image Models): https://github.com/rwightman/pytorch-image-models
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- SynFlow (Tanaka et al. 2020): https://proceedings.neurips.cc/paper/2020/file/46a4378f835dc8040c8057beb6a2da52-Paper.pdf
- GraSP (Wang et al. 2020): https://openreview.net/forum?id=SkgsACVKPH
- SNIP (Lee et al. 2019): https://openreview.net/forum?id=B1VZjDXcF
- Molchanov Taylor Pruning: https://openreview.net/forum?id=SJGCiw5gl
- Yosinski Transferability: https://papers.nips.cc/paper/2014/hash/7acd6a8d870b14b8a5b8e43e8a0f5aeb-Abstract.html
- DeepNcode: https://arxiv.org/abs/2405.13891
- Weight Scaling Defense: https://arxiv.org/abs/2411.18993
- Lipschitz Networks (Gouk et al.): https://link.springer.com/article/10.1007/s10994-020-05929-w
- Kandel Principles of Neural Science: https://books.google.co.il/books?id=yzEFK7Xc87YC

Karpathy, 这篇 paper 的 core insight 极其 simple — magnitude + early-layer + per-kernel constraint。但 simple 背后是 DNN 训练的 fundamental 性质:importance 集中、early layers disproportionate、kernel-internal cancellation。Pruning literature 几十年累积的 saliency criterion 直接 mirror 到 attack 端。这或许是 paper 最 deep 的 contribution — 显式 establish pruning-attack duality, unify 两个看似无关的 subfields。
