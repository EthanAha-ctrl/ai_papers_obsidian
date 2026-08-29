---
source_pdf: Finer-CAM  Spotting the Difference.pdf
paper_sha256: b5913501ee8bb2a21e138522871336e81f9894e323f759b1c1758941aba7c78d
processed_at: '2026-08-04T08:20:32-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Finer-CAM

Andrej，咱们抛开 paper 的学术腔调，像在白板前聊天那样讲。

---

## 一、这帮人在解决啥问题

你训了个 classifier，认鸟的种类。给它一张 Blue Grosbeak 的图，它说"这是 Blue Grosbeak"，对的。你问它"你凭啥这么判断？"——这就要 Grad-CAM 出场，画个 heatmap 告诉你"我看了这块区域"。

问题是：Grad-CAM 画出来的 heatmap **糊一大片**，把整只鸟的身体都涂红了。你看着这个 map，心想："行吧，你看了鸟的身体，但这怎么能帮我理解你怎么区分 Blue Grosbeak 和长得几乎一模一样的 Grandala？"

**这就是病根**：Grad-CAM 回答的是"哪些区域对预测 Blue Grosbeak 有贡献"，但"对 Blue Grosbeak 有贡献"和"能区分 Blue Grosbeak 和 Grandala"是两码事。蓝色身体对 Blue Grosbeak 的 logit 有正贡献，对 Grandala 的 logit 也有正贡献——这 feature 是 shared 的，根本不 discriminative。

真正能区分这俩鸟的，是翅膀颜色的一个小细节。但 Grad-CAM 把这个细节淹没在整片蓝色身体的高 activation 里了。

---

## 二、他们的 insight 是啥

一句话：**别问"啥能预测这个 class"，问"啥能区分这个 class 和它最像的那个 class"**。

这就像玩"找茬"游戏。给你一张图，问你"这图里有啥"——你会描述整张图。但给你两张几乎一样的图，问你"这俩有啥区别"——你会瞬间锁定那个小差异。

Finer-CAM 就是把这个"找茬"机制塞进 CAM 里。具体怎么做？Grad-CAM 原来算的是 gradient of $y^c$（target class 的 logit），现在改成算 gradient of $(y^c - \gamma \cdot y^d)$（target class logit 减去 similar class logit）。

就这么一个改动。一行代码级别的事。

---

## 三、为什么这个改动 work

数学上特别好理解。Grad-CAM 在最后一层的 channel importance weight $\alpha_k^c$，实际上就等于 classifier weight $w_k^c$（up to 一个常数）。所以 Grad-CAM 本质在做：

$$L^c = \text{ReLU}\left(\sum_k w_k^c \cdot A_k\right)$$

用 class $c$ 的 classifier 权重对 feature maps 加权求和。

Finer-CAM 把权重换成 $w_k^c - \gamma \cdot w_k^d$：

$$L^{c,d} = \text{ReLU}\left(\sum_k (w_k^c - \gamma \cdot w_k^d) \cdot A_k\right)$$

**直觉**：如果某个 channel $k$ 的特征对 class $c$ 和 class $d$ 都很重要（shared feature），那 $w_k^c$ 和 $w_k^d$ 都大，相减后抵消，这个 channel 被抑制。如果某个 channel 只对 $c$ 重要对 $d$ 不重要（discriminative feature），$w_k^c$ 大 $w_k^d$ 小，相减后保留，这个 channel 被突出。

这就是一个**在 classifier weight 空间里的差分操作**，把 shared component 减掉，留下 discriminative component。

---

## 四、一个关键的 subtle 点：为啥不能直接减两个 saliency map

你可能想：那我跑两次 Grad-CAM，得到 $L^c$ 和 $L^d$，直接 $L^c - \gamma L^d$ 不就完了？

不行。因为 ReLU。

Finer-CAM 是在 ReLU **之前**做减法：
$$\text{ReLU}\left(\sum_k (w_k^c - \gamma w_k^d) A_k\right)$$

直接减 saliency map 是在 ReLU **之后**做减法：
$$\text{ReLU}\left(\sum_k w_k^c A_k\right) - \gamma \cdot \text{ReLU}\left(\sum_k w_k^d A_k\right)$$

这俩数学上不一样。ReLU 是非线性的，$\text{ReLU}(a-b) \neq \text{ReLU}(a) - \text{ReLU}(b)$。

具体说，Finer-CAM 的减法在 ReLU 之前，能保留 "对 $c$ 正、对 $d$ 负"的 signal——这个 signal 在 spatial 某个位置可能是负的，意味着"这块区域对区分 $c$ vs $d$ 是负贡献的"，ReLU 之后 clip 掉，干净。

直接减 saliency map 呢？两个 ReLU 输出都非负，相减后各种乱七八糟的 noise 全出来了，因为 sign information 已经在各自 ReLU 时丢掉了。Fig. 9b 里直接减的结果是糊的 noise map，Finer-CAM 是干净的。

**一句话：减法要在 ReLU 之前做，sign matters。**

---

## 五、$\gamma$ 这个旋钮

$\gamma$ 是 comparison strength，控制你多狠地 suppress similar class 的 features。

- $\gamma = 0$：退化成普通 Grad-CAM，heatmap 覆盖整个 object，coarse
- $\gamma = 0.6$：paper 的 default，平衡点
- $\gamma = 0.8$：Fig. 6b 显示 relative drop metric 在这达到 peak
- $\gamma = 2.0$：Fig. 13 的 extrapolation，heatmap 极度聚焦在 micro details，几乎只剩一个小点

Intuition：$\gamma$ 太小，suppress 不够，还是糊一片；$\gamma$ 太大，过度 suppress，连对 $c$ 真正重要的 features（因为它们对 $d$ 也有一点点贡献）都被减掉了，反而伤 localization。

这给用户一个**可调的"聚焦度"**：想要 coarse contour 就调小 $\gamma$，想要 fine detail 就调大 $\gamma$。这个 flexibility 在实际用起来挺方便的。

---

## 六、Aggregation：和多个 similar class 比较

不只和一个 similar class 比，可以和 top-3 similar classes 都比，然后 aggregate：

$$L^c = \text{ReLU}\left(\frac{1}{T} \sum_{t=1}^{T} \sum_k \alpha_k^{c,t} A_k\right)$$

$t$ 索引第 $t$ 个 reference class，$T$ 是 reference 总数（paper 用 3）。

**关键：average 在 ReLU 之前做**。Tab. 7 显示这个策略最好。

为啥？因为 average 在 ReLU 之前，多个 comparison 的"减法 signal"能叠加——某个 channel 如果同时被多个 similar class 共享，它在每个 comparison 里都被减一点，加起来就被狠狠 suppress。After ReLU 的 average 丢掉了 sign，没法做这种"加权抵消"。

Intuition：和你最像的 3 个 class 比，比只和 1 个比更稳。因为和单一 reference 比，万一那个 reference 选得不好（logit similarity 高但 visual 不像），就翻车。多个 reference 平均下来更鲁棒。

---

## 七、Score-CAM 版本的微妙之处

Score-CAM 不用 gradient，用"把某个 feature map 上采样回原图大小，盖在原图上，看 logit 变多少"来定权重：

$$\alpha_k^c = f(\mathbf{x} \circ \mathbf{H}_k)^c - f(\mathbf{x}_b)^c$$

$\mathbf{x}$ 是原图，$\mathbf{H}_k$ 是第 $k$ 个 feature map 上采样到原图大小，$\circ$ 是 element-wise 乘，$\mathbf{x}_b$ 是 baseline（全黑图），$f(\cdot)^c$ 是 class $c$ 的 logit。

Finer-CAM 版本：

$$\alpha_k^{c,d} = f(\mathbf{x} \circ \mathbf{H}_k)^c - \gamma \cdot f(\mathbf{x} \circ \mathbf{H}_k)^d - f(\mathbf{x}_b)^c$$

注意一个 asymmetry：减的 baseline 是 $f(\mathbf{x}_b)^c$，不是 $f(\mathbf{x}_b)^d$。

这里我的理解是：第一项 $f(\mathbf{x} \circ \mathbf{H}_k)^c - f(\mathbf{x}_b)^c$ 衡量的是"加上 $\mathbf{H}_k$ 后 $c$ 的 logit 增加了多少"——contribution to $c$。第二项 $\gamma \cdot f(\mathbf{x} \circ \mathbf{H}_k)^d$ 不减 baseline，因为这里要的是"加上 $\mathbf{H}_k$ 后 $d$ 的 logit 变成多少"这个绝对值，用来 suppress。

严格写应该是：

$$\alpha_k^{c,d} = [f(\mathbf{x} \circ \mathbf{H}_k)^c - f(\mathbf{x}_b)^c] - \gamma [f(\mathbf{x} \circ \mathbf{H}_k)^d - f(\mathbf{x}_b)^d] + \text{const}$$

后面常数不影响 ranking，所以 paper 省了。这是 paper 写法 minimalism 的一个体现，但读的时候要理解。

---

## 八、评估 metric 的聪明设计：Relative Drop

这是 paper 里我觉得第二聪明的地方。

**标准 deletion metric**：按 saliency 从高到低 mask pixels，看 target class confidence 怎么掉，AUC 越小越好。

问题：Finer-CAM 和 baseline 在这个 metric 上几乎没差别（Tab. 1：Grad-CAM 0.079 vs +Finer 0.076）。因为 mask 掉 discriminative region（翅膀）和 mask 掉 shared region（身体）都会让 target class confidence 下降，deletion metric 分不清这俩。

**Relative Drop**：

$$\text{RD} = (p^c - p_*^c) - (p^d - p_*^d)$$

$p^c$ 是 mask 前 target class $c$ 的 confidence，$p_*^c$ 是 mask 后的。$p^d$ 和 $p_*^d$ 同理，对应 similar class $d$。

Intuition：如果你 mask 掉的 regions 真的是 discriminative 的，那它应该**主要打击 $c$ 的 confidence，不太动 $d$ 的 confidence**——因为这些 regions 对 $d$ 本来就不重要。

- Mask shared region（baseline CAM 倾向做的）：$p^c$ 掉，$p^d$ 也掉 → RD 小
- Mask discriminative region（Finer-CAM 倾向做的）：$p^c$ 大掉，$p^d$ 几乎不动 → RD 大

这 metric 直接 measure "你找到的 regions 在多大程度上 specifically belong to $c$ 而不是 $d$"。Tab. 1 显示 Finer-CAM 在 RD@0.05 上稳定提升 8-10%。

**这个 metric 设计的哲学很深**：explanation 的好坏不是看"能不能解释 target"，是看"能不能区分 target from alternatives"。这和 Bayesian evidence 的逻辑一致——evidence for hypothesis $H$ 是 evidence that raises $H$ 的 posterior **相对于** alternatives，不是绝对 raise $H$ 的概率。

---

## 九、Multi-modal extension：CLIP 上的玩法

CLIP 没有 linear classifier，logit 是 image embedding 和 text embedding 的 cosine similarity。Finer-CAM 直接把 text embedding 当"classifier weight"，比较两个 text prompts。

最 cute 的 application 是 Fig. 2 的 red epaulets 例子。你想 localize "red epaulets"（红色肩羽）这个 attribute 在图上的位置。

直接用 "red epaulets" 作为 prompt 跑 Grad-CAM：activation 很弱，因为 CLIP 的 similarity 对 whole-object alignment 敏感，对局部 attribute 不敏感。

Finer-CAM 的做法：比较 "red epaulets" 和 "bird" 两个 prompt 的 similarity 差异。

- "red epaulets" prompt 和整个 bird 都有 similarity（epaulets 是 bird 的一部分）
- "bird" prompt 和整个 bird 也有 similarity
- 相减后留下来的就是 "red epaulets" 比 "bird" 额外强调的部分——即 epaulets 本身

这相当于把"红色肩羽"作为一个相对于"鸟"的 **incremental concept** 来 localize。像在说："告诉我和一只普通鸟相比，这只鸟额外多了啥"。极妙。

更进一步，Sec 4.3 提出一个 faithfulness examination：
1. 用 Finer-CAM 对 classifier 跑 saliency（比较 target class vs similar class 的 logit 差异）
2. 用 Finer-CAM 对 CLIP 跑 attribute saliency（比较 attribute prompt vs "bird" prompt）
3. 看两个 saliency map align 不 align

如果 align：classifier 在用正确的 trait 区分 class。如果不 align：classifier 学了 spurious correlation，或者 dataset attribute annotation 不全。

**这是用 XAI 工具反过来 audit classifier**，挺有意思的用法。

---

## 十、实验数据里值得注意的点

Tab. 1 Birds-525：
- Grad-CAM RD@0.05 = 0.174 → +Finer = 0.192（+10.3%）
- Layer-CAM 0.186 → 0.201（+8.1%）
- Score-CAM 0.151 → 0.163（+7.9%）

Layer-CAM + Finer 在 CUB 的 localization：0.625 → 0.682（+9.1%），显著。

Tab. 5/6 DINOv2 backbone 上 Finer-CAM 提升更小。Paper 解释：DINOv2 linear classifier accuracy 更高（CUB 66.4 vs CLIP 58.4），classifier weight separation 更好，similar class 间 weight similarity 更低，Finer-CAM 能"减"的东西更少。

**这其实反向验证了核心假设**：Finer-CAM 的效果取决于 similar class 之间 classifier weight 有多大 similarity。如果 classifier 已经 well-separated，Grad-CAM 本身就够用，Finer-CAM 边际收益就小。

Tab. 2 的 aggregation ablation：单个 2nd prediction 比较 RD=0.198，aggregation 0.192，2nd pred 单独反而略高。但 paper 选 aggregation 是因为更稳定，避免 2nd pred 在某些 case 不真 visually similar 时翻车。

---

## 十一、Failure cases

Sec C.1 诚实地说了：
- Classifier 预测错时，Finer-CAM 也救不了——因为 reference class 选错了
- Logit similarity 不反映 visual similarity 时（某 class weight 和 target 接近但视觉上不像），Finer-CAM degenerate 到 baseline

核心假设是"logit similarity ≈ visual similarity"。well-trained fine-grained classifier 上通常成立，pathological cases 会 break。

---

## 十二、Intuition 的三层

**操作层**：把 CAM 的 target 从 $y^c$ 改成 $y^c - \gamma y^d$，一行代码。

**表示层**：classifier weight 空间里 similar classes 的 weight vectors 重叠。Grad-CAM 用单一 weight vector 投影，highlight shared regions。Finer-CAM 用差分 vector 投影，filter 掉 shared component，留下 orthogonal complement 中的 discriminative signal。

**哲学层**：explanation 不是 "what activates this class"，是 "what distinguishes this class from its competitors"。Bayesian evidence for $H$ 是 evidence that raises $H$ 的 posterior **相对于** alternatives。Grad-CAM 只看 numerator（likelihood for $c$），Finer-CAM 看 likelihood ratio $p(\text{features}|c) / p(\text{features}|d)^\gamma$。

第三层是真正的 insight，可以推广到所有 XAI 工作。

---

## 十三、我的几个 critical thoughts

1. **Reference class 选择是 hyperparameter**。Paper 用 classifier weight cosine similarity 选 top-K，依赖 classifier 学得好。如果 classifier 有 bias，reference 也偏。改进方向：用 visual feature space（CLIP image embedding）选 reference。

2. **$\gamma$ 应该 class-pair-specific**。不同 pair similarity 不同，固定 $\gamma$ 不最优。Data-driven 选择：$\gamma_{c,d} = \text{sim}(w^c, w^d)$，越像 suppress 越多。

3. **Relative Drop metric 的 $d$ 选择**。用 "most similar class" 单一 $d$，但如果有多个 similar classes，更严的 metric 应该是 $\text{RD} = (p^c - p_*^c) - \max_{d \neq c}(p^d - p_*^d)$，衡量 mask 打击 $c$ 相对于所有 competitors 的 margin。

4. **Connection to mutual information**。Finer-CAM 实质上在 highlight "features that have high mutual information with class label conditional on the reference class"。这是 information-theoretic view，paper 没点破但有潜力。

5. **Open-set / long-tailed 场景的潜在 failure**。target class 的 "most similar class" 可能 noisy，假设 break。

---

## 十四、相关链接

- **Finer-CAM 代码**：https://github.com/Imageomics/Finer-CAM
- **Grad-CAM 原文**：https://arxiv.org/abs/1610.02391
- **Score-CAM 原文**：https://arxiv.org/abs/1910.06476
- **Layer-CAM**：https://arxiv.org/abs/2104.08617
- **CLIP**：https://arxiv.org/abs/2103.00020
- **DINOv2**：https://arxiv.org/abs/2304.07193
- **CUB-200**：https://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- **RISE（对比方法）**：https://arxiv.org/abs/1806.07421
- **GEM（multi-modal grounding 对比）**：https://arxiv.org/abs/2404.00791
- **Bayesian evidence / likelihood ratio 的哲学基础**：https://en.wikipedia.org/wiki/Bayes_factor

---

## 一句话总结

**CAM 的问题不在"怎么解释"，在"解释啥"。原来解释"啥对预测有贡献"，现在解释"啥能把这 class 和它最像的 class 区分开"。改一行代码，从 absolute explanation 变成 contrastive explanation，fine-grained localization 就大幅提升。**

这工作的 long-term 价值在 conceptual shift：explanation should be contrastive, not absolute。这个 insight 可以渗透到很多 XAI 工作里。

---

# Finer-CAM 深度讲解

Andrej，这篇 paper 是一个很漂亮的小工作，核心 idea 非常简单但 insight 深刻。让我从直觉出发，一层一层剥开。

## 一、核心 insight 的建立

**问题根源的诊断：CAM 不是 "how" 出了问题，是 "what" 出了问题。**

标准 Grad-CAM 做的事情是：找一个 saliency map $L^c$，使得这个 map 上的 regions "对 target class $c$ 的 logit 有正贡献"。但这里有个 subtle 的 bug——在 fine-grained classification 里，**对 class $c$ 有正贡献的 features，很可能也对 similar class $d$ 有正贡献**。

Consider Fig. 1 的例子：Blue Grosbeak 和 Grandala 都是蓝色的鸟。当你对 Blue Grosbeak 跑 Grad-CAM，它会 highlight 整个身体（蓝色部分），因为蓝色身体的 features 确实推高了 Blue Grosbeak 的 logit。但同样的 features 也推高了 Grandala 的 logit，因为这两个 class 的 classifier weight 高度相似（Fig. 1 left 的 cosine similarity 图）。

所以 highlight 整个蓝色身体其实**没告诉你怎么区分这两个 class**。真正区分它们的是翅膀的颜色——这才是 discriminative detail。

Fig. 1 middle 的两张 saliency map 几乎一模一样，这正是这个 pathology 的可视化证据。它说明：**当你 explain class $c$ 时，你其实在 explain 一组 shared features**。

## 二、数学层面的精妙之处

### 2.1 Grad-CAM 的权重等于 classifier weight

Eq. (4) 是一个关键结果。对最后一层（在 linear classifier 之前），Grad-CAM 的 channel importance weight $\alpha_k^c$ 实际上正比于对应的 classifier weight $w_k^c$：

$$w_k^c = \sum_i \sum_j \frac{\partial y^c}{\partial A_k^{ij}}$$

这里 $w_k^c$ 是 classifier 对第 $k$ 个 channel、第 $c$ 个 class 的 weight；$A_k^{ij}$ 是第 $k$ 个 feature map 在 spatial location $(i,j)$ 的 activation；$y^c$ 是 class $c$ 的 logit。

**直觉**：Grad-CAM 的 channel 权重 $\alpha_k^c$ = "这个 channel 对 class $c$ 有多重要" = classifier 自己学到的 $w_k^c$。所以 Grad-CAM 本质上是在做 $\sum_k w_k^c A_k$，即用 class $c$ 的 classifier 权重对 feature maps 加权求和。

### 2.2 Finer-CAM 的核心操作

Eq. (5) 是全文最重要的一个 equation：

$$\alpha_k^{c,d} = \frac{1}{Z} \sum_i \sum_j \frac{\partial (y^c - \gamma \times y^d)}{\partial A_k^{ij}}$$

变量说明：
- $y^c$：target class $c$ 的 logit
- $y^d$：一个 visually similar 的 reference class $d$ 的 logit
- $\gamma$：comparison strength coefficient，控制 suppress reference class 的强度
- $A_k^{ij}$：第 $k$ 个 feature map 的 $(i,j)$ 位置
- $Z$：归一化常数（feature grid 总数）

由于 gradient 的 linearity（Eq. 6），可以 decompose：

$$\frac{\partial (y^c - \gamma y^d)}{\partial A_k^{ij}} = \frac{\partial y^c}{\partial A_k^{ij}} - \gamma \frac{\partial y^d}{\partial A_k^{ij}}$$

代入 Eq. (2) 的定义，得到 Eq. (7)：

$$\alpha_k^{c,d} = \alpha_k^c - \gamma \cdot \alpha_k^d$$

**这是关键 insight 的数学化**：Finer-CAM 的 channel 权重 = target class 的 channel 权重 - $\gamma$ × similar class 的 channel 权重。

如果一个 channel 对 $c$ 和 $d$ 都重要（shared feature），它会被减掉；如果一个 channel 只对 $c$ 重要而对 $d$ 不重要（discriminative feature），它会保留。

### 2.3 为什么不能直接减去两个 saliency maps

这是 paper 里很 subtle 但很重要的一个点（Sec 4.3 + Fig. 9b）。

你可能想：既然 $\alpha_k^{c,d} = \alpha_k^c - \gamma \alpha_k^d$，那我先跑两次 Grad-CAM 得到 $L^c$ 和 $L^d$，然后相减不就行了？

不行。因为 Eq. (1) 里有 ReLU：

$$L^c = \text{ReLU}\left(\sum_k \alpha_k^c A_k\right)$$

ReLU 不是线性的，所以：

$$\text{ReLU}\left(\sum_k (\alpha_k^c - \gamma \alpha_k^d) A_k\right) \neq \text{ReLU}\left(\sum_k \alpha_k^c A_k\right) - \gamma \cdot \text{ReLU}\left(\sum_k \alpha_k^d A_k\right)$$

左边是 Finer-CAM（在 ReLU 之前做减法），右边是 naive subtraction。左边能保留 "对 $c$ 正、对 $d$ 负" 的 regions，右边会因为两个 ReLU 输出都非负，相减后出现大量 noise。Fig. 9b 的对比非常清楚——naive subtraction 产生糊状的 noise map，而 Finer-CAM 产生干净的 discriminative region。

## 三、Aggregation 策略的细节

### 3.1 多个 reference class 的聚合

Eq. (8)：

$$L^c = \text{ReLU}\left(\frac{1}{T} \sum_t \sum_k \alpha_k^{c,t} A_k\right)$$

- $T$：比较的 reference class 数量（paper 默认用 top 3 similar classes）
- $t$：第 $t$ 个 reference class

关键：**average 是在 ReLU 之前做的**。Tab. 7 比较了三种策略：
- Before ReLU, Max：$\max_t \sum_k \alpha_k^{c,t} A_k$，然后 ReLU
- Before ReLU, Avg：$\frac{1}{T}\sum_t \sum_k \alpha_k^{c,t} A_k$，然后 ReLU ← **最好**
- After ReLU, Avg：$\frac{1}{T}\sum_t \text{ReLU}(\sum_k \alpha_k^{c,t} A_k)$

为什么 before ReLU averaging 最好？因为 averaging 在 ReLU 之前，可以让多个 comparison 的"减法信号"叠加，把同时被多个 similar class 共享的 features 更彻底地 suppress 掉。After ReLU averaging 则丢失了 sign information（每个 comparison 已经 clip 掉负值），无法做"加权抵消"。

### 3.2 Comparison strength $\gamma$ 的设计

$\gamma$ 是一个很有意思的 design knob：
- $\gamma = 0$：退化到 baseline Grad-CAM，coarse map 覆盖整个 object
- $\gamma = 0.6$：paper 的 default，平衡点
- $\gamma = 0.8$：Fig. 6b 显示这是 relative drop 的 peak
- $\gamma = 2.0$：Fig. 13 的 extrapolation，activation 极度聚焦在 micro details

Fig. 6b 的曲线很有意思——relative drop 在 $\gamma=0.8$ 达到 peak 后开始下降。Intuition：$\gamma$ 太大会过度 suppress，把一些对 $c$ 真正重要的 features 也一起减掉了（因为它们对 $d$ 也有少量贡献），反而损害 localization。

Paper 最终选 $\gamma=0.6$ 而不是 $0.8$，可能是为了 stability across datasets。

## 四、Score-based Finer-CAM 的微妙之处

Eq. (9) 是 Score-CAM 的扩展：

$$\alpha_k^{c,d} = f(\mathbf{x} \circ \mathbf{H}_k)^c - \gamma \cdot f(\mathbf{x} \circ \mathbf{H}_k)^d - f(\mathbf{x}_b)^c$$

变量：
- $\mathbf{x}$：输入 image
- $\mathbf{H}_k$：第 $k$ 个 activation map 上采样到原图大小
- $\circ$：Hadamard product（element-wise）
- $\mathbf{x}_b$：baseline input（默认 zero input）
- $f(\cdot)^c$：取 class $c$ 的 logit

注意一个非常 subtle 的 asymmetry：**减去的是 $f(\mathbf{x}_b)^c$，不是 $f(\mathbf{x}_b)^d$**。

为什么不减 $f(\mathbf{x}_b)^d$？我的理解是：Score-CAM 的 baseline 是为了 normalize 掉"什么都不输入时的 default logit"。对 class $c$，我们要 measure "加入 $\mathbf{H}_k$ 之后 $c$ 的 logit 增加了多少"，所以减 $f(\mathbf{x}_b)^c$。但减去 $\gamma \cdot f(\mathbf{x} \circ \mathbf{H}_k)^d$ 时，我们不需要再减 $f(\mathbf{x}_b)^d$，因为我们要的就是"加入 $\mathbf{H}_k$ 让 $d$ 的 logit 变成多少"这个绝对值（用来 suppress），而不是"增加了多少"。

数学上更清楚的形式应该是：

$$\alpha_k^{c,d} = \underbrace{[f(\mathbf{x} \circ \mathbf{H}_k)^c - f(\mathbf{x}_b)^c]}_{\text{contribution to } c} - \gamma \cdot \underbrace{[f(\mathbf{x} \circ \mathbf{H}_k)^d - f(\mathbf{x}_b)^d]}_{\text{contribution to } d} + \gamma \cdot f(\mathbf{x}_b)^d - f(\mathbf{x}_b)^c$$

后两项是常数（不依赖 $k$），对 ranking 不影响，所以省略掉。这解释了 paper 写法的 minimalism。

## 五、Relative Confidence Drop 这个 metric 的设计哲学

这是 paper 我觉得第二聪明的地方。

### 5.1 标准 deletion metric 的问题

标准 deletion curve：按 saliency 从高到低 mask 掉 pixels，看 target class 的 confidence 怎么下降。AUC 越小越好。

但 Tab. 1 显示 Finer-CAM 和 baseline 的 deletion AUC 几乎一样（Grad-CAM 0.079 vs +Finer 0.076）。Paper 在 Sec 4.2 自己承认了这点。

为什么？因为 deletion metric 只看 target class 的 confidence。你 mask 掉 discriminative regions（翅膀）和 mask 掉 shared regions（身体）都会降低 Blue Grosbeak 的 confidence——只是降低的机制不同。Deletion AUC 区分不了这两种 mask。

### 5.2 Relative Drop 的 insight

Eq. (10)：

$$\text{RD} = (p^c - p_*^c) - (p^d - p_*^d)$$

- $p^c$：mask 前 target class $c$ 的 confidence
- $p_*^c$：mask 后 target class $c$ 的 confidence
- $p^d$：mask 前 similar class $d$ 的 confidence
- $p_*^d$：mask 后 similar class $d$ 的 confidence

核心 insight：**如果你 mask 掉的 regions 真的是 discriminative 的，那么它应该 specifically 打击 $c$ 的 confidence，而尽量不动 $d$ 的 confidence**（因为这些 regions 对 $d$ 本来就不重要）。

- Mask 掉 shared features（baseline CAM 倾向做的）：$p^c$ 下降，$p^d$ 也下降 → RD 小
- Mask 掉 discriminative features（Finer-CAM 倾向做的）：$p^c$ 下降明显，$p^d$ 几乎不变 → RD 大

这是一个非常 elegant 的 metric 设计，它直接 measure "你找到的 regions 在多大程度上 specifically belong to $c$ 而不是 $d$"。

Tab. 1 的结果就是证据：Grad-CAM 在 Birds-525 的 RD@0.05 是 0.174，+Finer 提升到 0.192（约 +10%）；Layer-CAM 从 0.186 提升到 0.201。这些提升在 deletion AUC 上是看不到的。

Fig. 5 的 deletion curve 可视化更直观：上面那条线是 target class confidence 随 mask 下降，下面那条是 similar class confidence。Finer-CAM 的两条线 gap 更大（colored area），说明 mask 主要打击了 target class。

## 六、Multi-modal extension 的妙用

### 6.1 CLIP 场景下的 setup

CLIP 没有 linear classifier，logit 是 image embedding 和 text embedding 的 cosine similarity：

$$y^c = \text{sim}(\mathbf{z}_{\text{img}}, \mathbf{t}^c)$$

这里 $\mathbf{t}^c$ 是 class $c$ 的 text prompt embedding。Finer-CAM 直接把 $\mathbf{t}^c$ 当作"classifier weight"，比较两个 text prompts：

$$\text{gradient of } [\text{sim}(\mathbf{z}_{\text{img}}, \mathbf{t}^c) - \gamma \cdot \text{sim}(\mathbf{z}_{\text{img}}, \mathbf{t}^d)]$$

### 6.2 Fig. 2 的 red epaulets 例子

这是 paper 最 cute 的 application。你想 localize "red epaulets"（一种鸟的红色肩羽）这个 attribute 在图上的位置。

直接用 "red epaulets" 作为 prompt 跑 Grad-CAM：activation 很弱，因为 CLIP 的 image-text similarity 本质上对 whole-object 的 alignment 更敏感，对局部 attribute 不敏感。

Finer-CAM 的做法：比较 "red epaulets" 和 "bird" 这两个 prompt 的 similarity 差异。Intuition 是：
- "red epaulets" prompt 和整个 bird 都有 similarity（因为 epaulets 是 bird 的一部分）
- "bird" prompt 和整个 bird 也有 similarity
- 两者相减，留下来的就是 "red epaulets" 比 "bird" 额外强调的部分——即 epaulets 本身的位置

这相当于把"红色肩羽"作为一个相对于"鸟"的 **incremental concept** 来 localize。非常巧妙。

### 6.3 Faithfulness examination

Sec 4.3 + Fig. 9a 提出了一个 verification 流程：
1. 用 Finer-CAM 对 classifier 跑 saliency map（比较 target class 和 similar class 的 logit 差异）
2. 用 Finer-CAM 对 CLIP 跑 attribute saliency map（比较 attribute prompt 和 "bird" prompt）
3. 比较两个 saliency map 是否 align

如果 align：classifier 在用正确的 trait 区分 class；如果不 align：classifier 可能学到了 spurious correlation，或者 dataset 的 attribute annotation 不全。

这相当于一个 **behavioral test for the classifier**，用 XAI 工具反过来 audit classifier。

## 七、实验数据的仔细解读

### 7.1 Tab. 1 的关键数字

Birds-525：
- Grad-CAM: Del=0.079, RD@0.05=0.174
- +Finer: Del=0.076, RD@0.05=0.192 ← +10.3%
- Layer-CAM: RD@0.05=0.186
- +Finer: RD@0.05=0.201 ← +8.1%
- Score-CAM: RD@0.05=0.151
- +Finer: RD@0.05=0.163 ← +7.9%

Layer-CAM + Finer 在 CUB 的 localization：0.625 → 0.682（+9.1%），这是非常显著的提升。

Cars 上 Loc：Layer-CAM 0.581 → +Finer 0.592，提升不大。可能因为 cars 的 discriminative features 比较分散（轮子、车灯、格栅等），bounding box 评估不够精细。

### 7.2 DINOv2 vs CLIP 的对比（Tab. 5, Tab. 6）

DINOv2 backbone 上 Finer-CAM 也有效但提升幅度更小。Paper 给的解释是：DINOv2 的 linear classifier accuracy 更高（Tab. 3：CUB 66.4 vs 58.4），意味着 classifier weight 之间的 separation 更好，similar class 之间的 weight similarity 更低，所以 Finer-CAM 的"减法"能减的东西更少。

这个观察支持了 paper 的核心 narrative：**Finer-CAM 的效果取决于 similar class 之间的 classifier weight 有多大 similarity**。如果 classifier 已经 well-separated，Grad-CAM 本身就能 highlight discriminative regions，Finer-CAM 的边际收益就小。

### 7.3 Aggregation 的 ablation（Tab. 2）

Comparing target with:
- None（baseline）: RD@0.05 = 0.174
- 2nd Pred: 0.198 ← 单个 reference 中最好
- 3rd Pred: 0.178
- 4th Pred: 0.174
- Aggregation of top 3: 0.192

有趣的是，2nd prediction 单独比较比 aggregation 还好。Paper 仍选 aggregation 是因为 aggregation 在更多 datasets 上稳定，而 2nd pred 在某些 case 可能因为 second-most-similar 不真的 visually similar 而失败。

## 八、Failure cases 的诚实

Sec C.1 + Fig. 10 提到了 failure cases：当 classifier 本身预测错误时，Finer-CAM 也无能为力；当 logit similarity 不反映 visual similarity 时（即某个 class 的 weight 和 target 接近但 visual 上很不同），Finer-CAM 也 degenerate 到 baseline。

这说明 Finer-CAM 是建立在 "logit similarity ≈ visual similarity" 这个假设上的。这个假设在 well-trained fine-grained classifier 上通常成立，但在某些 pathological cases 会 break。

## 九、Intuition 总结

我会用三个层次总结 Finer-CAM 的 intuition：

**Layer 1（操作层面）**：把 CAM 的 explanation target 从 "maximize $y^c$" 改成 "maximize $y^c - \gamma y^d$"。一行代码的改动。

**Layer 2（表示层面）**：classifier weight 空间中，similar classes 的 weight vectors 高度重叠。Grad-CAM 用单一 weight vector 投影 feature maps，自然 highlight shared regions。Finer-CAM 用"差分 vector"投影，自动 filter 掉 shared components，留下 orthogonal complement 中的 discriminative signal。

**Layer 3（哲学层面）**：explanation 不是 "what activates this class"，而是 "what distinguishes this class from its competitors"。这是一个 Bayesian-style 的思维：evidence for a hypothesis 是 evidence that **raises** the hypothesis 的 posterior 相对于 alternatives。Grad-CAM 只看 numerator（likelihood for $c$），Finer-CAM 看的是 likelihood ratio $p(\text{features}|c) / p(\text{features}|d)^{\gamma}$。

第三层是最深的 insight，也最值得吸收到其他 XAI 工作里。

## 十、相关 references

- **Finer-CAM 代码**：https://github.com/Imageomics/Finer-CAM
- **Grad-CAM 原文**：https://arxiv.org/abs/1610.02391
- **Score-CAM 原文**：https://arxiv.org/abs/1910.06476
- **Layer-CAM**：https://arxiv.org/abs/2104.08617
- **CLIP**：https://arxiv.org/abs/2103.00020
- **DINOv2**：https://arxiv.org/abs/2304.07193
- **CUB-200 dataset**：https://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- **RISE**（对比方法）：https://arxiv.org/abs/1806.07421
- **GEM**（multi-modal grounding 对比）：https://arxiv.org/abs/2404.00791

## 十一、我的一些 critical thoughts

1. **Reference class 的选择是 hyperparameter**。Paper 用 "top-K similar by classifier weight cosine similarity"，但这依赖 classifier 学得好。如果 classifier 有 bias，reference 选择也会偏。一个可能的改进：用 visual feature space（如 CLIP image embedding）选 reference，而不是 classifier weight。

2. **$\gamma$ 应该是 class-pair-specific**。不同 class pair 的 similarity 不同，固定 $\gamma$ 不是最优。一个 data-driven 的选择：$\gamma_{c,d} = \text{sim}(w^c, w^d)$，越相似 suppress 越多。

3. **Relative Drop metric 的 $p^d$ 选择**。Paper 用 "most similar class" 作为 $d$。但如果有多个 similar classes，mask 掉 discriminative region 应该打击 $c$ 相对所有 similar classes 的 margin。一个更严格的 metric：$\text{RD} = (p^c - p_*^c) - \max_{d \neq c} (p^d - p_*^d)$。

4. **理论上的 connection to mutual information**。Finer-CAM 实质上在 highlight "features that have high mutual information with class label conditional on the reference class". 这是 information-theoretic 的 view，paper 没点破但很有潜力。

5. **可能的 failure mode**: 在 open-set 或者 long-tailed classification 里，target class 的 "most similar class" 可能是 noisy 的。Finer-CAM 的假设 break。

总结：这是一个 idea-clean、execution-solid 的工作。它的贡献更多在 conceptual insight（explanation should be contrastive）而非 technical novelty。这种工作往往比堆 complexity 的工作更有 long-term 价值，因为它 shift 了 community 的 thinking。Karpathy 你在 CS231n 里讲过的 "gradient is all you need" 系列思路，这篇 paper 在 CAM 范畴内做了类似的 shift：**contrastive gradient is all you need for fine-grained explanation**。
