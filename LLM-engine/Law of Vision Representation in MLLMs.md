---
source_pdf: Law of Vision Representation in MLLMs.pdf
paper_sha256: b1a9a600a6348002cb1b42d365528433fecdfe1ee171b56870ee4d1558efaf09
processed_at: '2026-08-05T12:25:04-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇Paper

## 一句话总结

**MLLM表现好不好，基本就看两件事：vision encoder跟语言模型"对得上话"吗？vision features自己内部"长得像的东西feature也像"吗？这两个指标一测，就能预测performance，不用再去跑昂贵的finetuning。**

---

## 问题是什么

现在做MLLM的人选vision encoder，方式很原始：拿一堆candidate（CLIP、SigLIP、DINOv2、diffusion features...），一个一个塞进pipeline里finetune LLM，跑benchmark，看谁高用谁。

这方法的问题：

- **贵得离谱**：测10个encoder要3840个A100小时，花2万美金。如果要测combinations（10个encoder的组合有1023种），电费够电动车绕地球13圈。
- **不知道为什么**：CLIP+DINOv2比单独用CLIP好，大家都知道，但没人说清楚到底好在哪。是alignment？是detail？是resolution？全混在一起，ablation做不清楚。

这篇paper说：别猜了，我给你两个数，算一下就能预测performance。

---

## 两个核心指标

### A Score（Alignment）—— "vision tokens能不能被LLM听懂"

直觉：把image经过vision encoder + projector之后的vision tokens，直接喂给**frozen的LLM**，看它能不能predict出正确的caption。

$$A = \frac{1}{T}\sum_t \log P(y_t | f(I), y_{<t})$$

就是在算frozen LLM对正确caption的log-likelihood。如果vision features跟language space对齐得好，LLM拿到vision tokens后自然能"读出"图片内容。CLIP这类contrastive pretrained的encoder，A score天生就高，因为它们本来就是对着text训的。

**计算成本极低**：只需要训一个小projector（0.3%的参数），LLM完全frozen，跑100张图就够。

### C Score（Correspondence）—— "图片里语义相同的地方，feature是不是也相似"

直觉：给两张图（比如两只不同的猫），在第一张上标个点（左耳尖），用vision features算similarity，看能不能在第二张图里找到对应的左耳尖。

$$C = \frac{1}{m}\sum_i \mathbb{1}[\|p_i^{pred} - p_i^{GT}\| < T]$$

就是标准的PCK metric。用[SPair-71k](https://arxiv.org/abs/1908.10543)这个semantic correspondence benchmark来算。

DINOv2的C score高（24.51），因为它用DINO + iBOT self-supervised训的，patch-level consistency被显式保留。CLIP的C score低（15.66），因为contrastive loss只关心global image-text matching，不在乎patch间像不像。

**这解释了为什么CLIP "lacks visual details"**——它的features每个patch都带有language semantics，但patch之间distinguishability差。LLM拿到CLIP features，能知道"这是只猫"，但分不清"左耳"和"右耳"。

---

## 核心发现：Performance ≈ 二次函数(A, C)

$$\text{AC Score} = w_{00} + w_{10}A + w_{01}C + w_{20}A^2 + w_{11}AC + w_{02}C^2$$

15个encoder setting，8个benchmark，拟合出来**R² = 94.06%**。

这个R²高得离谱。你想想，MLLM performance受多少因素影响——training data、hyperparameters、random seed、projector初始化...结果光靠A和C两个数就能解释94%的variance。

### 为什么是二次的，不是线性的

线性模型假设A和C是独立additive的。但实际上有个**synergy效应**：

- 当A很低时（比如DINOv2，A=-2.32），C再高也没用——LLM根本听不懂vision tokens在说什么，你visual detail再丰富它也"接收"不到。**Language barrier是bottleneck。**
- 当A高时（CLIP，A=-1.97），C的提升会amplify效果——LLM能听懂了，这时候给它更多visual detail，它就能利用上。

这个synergy需要cross term $AC$来捕捉。单纯的线性模型 $w_1 A + w_2 C$ 捕捉不了。

实验上，6/8个benchmark的 $w_{11}$（cross term）是正的，证实了synergy。只有MMMU和OKVQA是slight negative（可能这两个更靠reasoning不靠visual detail）。

---

## 为什么Combination Work——终于有解释了

社区里大家都知道CLIP+DINOv2 concatenation比单独用CLIP好。[Prismatic VLMs](https://arxiv.org/abs/2402.07865)、[Cambrian-1](https://arxiv.org/abs/2406.16860)都这么干，但都是empirical的，没解释why。

这篇paper的framework给了解释：

| Encoder | A Score | C Score |
|---------|---------|---------|
| CLIP@336 | -1.97 | 15.66 |
| DINOv2 | -2.32 | 24.51 |
| **CLIP+DINOv2@336** | **-1.95** | **26.08** |

Concatenation之后，A基本保持（继承CLIP的alignment），C大幅提升（继承DINOv2的correspondence）。AC score整体上去了，performance自然上去。

**这不是魔法，是两个orthogonal优势的叠加。** Paper第一次把这事说清楚了。

---

## AC Policy：省钱省到极致

### 流程

1. 对k个candidate encoder，每个都算A和C score（成本极低，只训projector）
2. 选其中k'个（比如k=15, k'=4），做完整的Stage 1+Stage 2 training，拿到真实performance
3. 用这k'个数据点拟合quadratic regression
4. 用拟合好的模型predict剩下k-k'个的performance
5. 排序，选top-3

### 采样策略

不能random选k'个，否则可能选到AC空间里扎堆的点，regression underdetermined。

用**region-based sampling**：把(A, C)画在2D plane上，递归地分成4个region（quadtree），每个region最多采一个点，保证coverage。

### 效果

- Budget = 4次full training
- Random selection: **26.7% Recall@3**（基本瞎猜）
- AC Policy: **87.72% Recall@3**

意思是：只花4次finetuning的成本，就能在15个candidate里87.7%的概率把最优encoder圈进top-3。

对比一下：random要做到85.6% Recall@3，得跑13/15个encoders。

**Compute reduction: 99.7%**（因为A score只需要训projector的0.3%参数，不用动7B LLM）。

---

## 理论Intuition（Appendix里的）

### 为什么Alignment高 → 省training

如果vision embedding $E_c^{image}$ 和对应text embedding $E_c^{text}$ 很接近（$\|E_c^{image} - E_c^{text}\| \leq \epsilon$），那么用vision tokens替换text tokens喂给LLM，output变化很小（因为transformer是[Lipschitz continuous](https://arxiv.org/abs/2106.05283)的）：

$$\|f([\text{vision tokens}]) - f([\text{text tokens}])\| \leq L\epsilon$$

所以用well-aligned vision representation训练MLLM，training dynamics接近text-only finetuning。LLM不需要额外花capacity去"学习翻译vision到language"，data efficiency自然高。

### 为什么Correspondence高 → 看到更多detail

Attention score里，如果text token $E_2$ attend到patch $E_0$，而 $E_0$ 和 $E_1$ 的key相似（high correspondence），那么 $E_2$ 也会间接attend到 $E_1$。这就是**transitivity**——attention通过feature similarity扩散到semantic相关的patches。

DINOv2的features让attention能"连片"地retrieve visual information，而CLIP的features是"孤立的"，每个patch都带着language tag但patch间没联系。

---

## 一些有意思的细节

### Diffusion features出奇地好

SD1.5的C score是22.02，仅次于DINOv2和combination。这跟[Emergent Correspondence from Image Diffusion](https://arxiv.org/abs/2306.03881)的发现一致——diffusion model的denoising objective implicit地学到了patch-level consistency。

但SD3/DiT（rectified flow架构）的features极差（C=3.09, A=-4.13）。Flow matching的latent space可能跟vision encoder需要的feature geometry不兼容。

### OCR是Achilles' Heel

AC score在vision-based benchmark上R²=94%，但OCR-based只有83.85%。原因：

- SPair-71k只有natural images（猫、火车...），没有text-heavy images
- CLIP在text images上correspondence其实很好（能match所有"LLaVA"文本），但SPair测不出来
- LLaVA-558K captions也缺OCR数据

所以SigLIP2-L的A和C都比CLIP高，但在MME（OCR-heavy）上反而输给CLIP。**Domain mismatch between measurement and evaluation。**

Paper承认这个limitation，呼吁社区建OCR-specific correspondence dataset。

### Cross-LLM generalization

换了Qwen 2.5 14B当backbone，R²依然>80%。Law不绑定specific LLM。

---

## 我的几个联想

### 1. 这本质是Surrogate Model Optimization

AC Policy跟Bayesian Optimization是同一类思路——用便宜的surrogate function替代昂贵的true objective。区别是BO通常用Gaussian Process，这里用simple polynomial。GP的好处是能给uncertainty estimate，exploration-exploitation更 principled。如果后面有人用GP替代polynomial，可能还能再提升。

### 2. A Score跟Mutual Information的关系

$\log P(y|f(I))$ 本质在maximize conditional log-likelihood，等价于maximize mutual information $I(Y; f(I))$。CLIP的contrastive loss就是A score的proxy——InfoNCE就是在approximate mutual information。所以这个framework跟information theory是自洽的。

### 3. "Alignment Tax"的multimodal版本

Anthropic讲RLHF有"alignment tax"——跟human preference对齐会hurt raw capability。这里也类似：vision encoder跟text对齐（contrastive training，优化A）会hurt visual correspondence（C）。CLIP high A low C，DINOv2 high C low A，就是这个tax的体现。

Combination策略本质是**用两个encoder分别承担两个objective**，avoid alignment tax。这跟LoRA用两个adapter分别serve不同task的思路有异曲同工之妙。

### 4. Resolution的AC trade-off

Paper没explore，但我猜：higher resolution → C升高（更多spatial detail）但A可能降（更多tokens要压缩进language space）。这可能是AC trade-off的一个来源。[LLaVA-NeXT](https://arxiv.org/abs/2401.02914)的any-resolution empirically work，AC framework可能能解释——高resolution提升C的收益超过A的损失。

### 5. 这种"Law"的范式

Title叫"Law of Vision Representation"，明显在evoke scaling laws。但scaling laws是performance ~ compute/params/data（observable quantities），这里performance ~ A/C（latent qualities需要proxy measurement）。

这种"latent factor law"如果work，可能能推广到其他component：
- "Law of Projector Design"：projector的什么property决定MLLM performance？
- "Law of Instruction Tuning Data Quality"：data的什么measurable property决定finetune效果？
- "Law of MLLM Mixture-of-Experts Routing"：什么factor决定哪个expert该被激活？

---

## 给Practitioner的Takeaway

1. **选encoder前先算AC**：花0.3%的cost算A和C，就能大致预测performance，别盲目跑full training
2. **Combination不是万灵药**：要看A和C的trade-off，如果两个encoder都是low A high C，combine了还是low A
3. **Task-aware selection**：vision-heavy task（VQA、spatial understanding）优先C高的encoder；language-heavy task优先A高的
4. **OCR task要小心**：AC score对OCR task预测不准，CLIP在OCR上的优势是SPair-71k测不出来的
5. **新encoder开发方向**：同时优化contrastive loss（A）和self-supervised consistency loss（C），SigLIP 2就是这么做的（A=-1.81最高，C=16.75也不错）

---

## 一句话再总结

**Vision encoder好不好，看两个数：跟LLM能不能对上话（A），自己features内部语义一致不一致（C）。两个都高performance就高，组合起来能互补，算出来不用跑finetune。**

---

# Law of Vision Representation in MLLMs 深度讲解

## 1. Paper核心问题与Motivation

这篇paper由Stanford、UC Berkeley、HK PolyU等机构的研究者提出，要解决一个长期被empirical方法掩盖的根本问题：**什么fundamentally makes a vision representation optimal for MLLMs?**

当前MLLMs社区选择vision encoder的方式是"train-and-test"：固定一个MLLM pipeline，把candidate encoders逐个试一遍，finetune LLM，看benchmark performance，选最好的。这种方式的cost极其惊人：

- 单个7B LLM的MLLM pipeline，测试10个encoders需要**3,840 NVIDIA A100 GPU hours**，约$20,000
- 10个encoders的全部combinations是$2^{10}-1 = 1023$种，需要约**100,000 kilowatt-hours**（足够让一辆EV绕地球13圈）
- 这种combinatorial explosion让"feature combination"方向变得practically intractable

更深层的问题在于：**我们不知道为什么某些encoder更好**。是cross-modal alignment？是visual detail preservation？是resolution？是training paradigm？这些因素是coupled的，empirical ablation无法disentangle。

Paper的insight：MLLM performance其实可以由两个orthogonal的、可测量的factor来预测——**cross-modal Alignment (A)** 和 **Correspondence (C)**，它们与performance呈quadratic关系，R²高达94.06%。

参考：[LLaVA series](https://arxiv.org/abs/2304.08485), [Cambrian-1](https://arxiv.org/abs/2406.16860), [Prismatic VLMs](https://arxiv.org/abs/2402.07865)

---

## 2. Law of Vision Representation的形式化

核心公式（公式1）：

$$Z \propto f(A, C)$$

其中：
- $Z$：MLLM的performance（在某benchmark上的score）
- $A$：cross-modal Alignment score
- $C$：Correspondence score
- $f$：quadratic function

这个Law的前提假设非常关键：

**(1) Architectural assumption**：仅适用于**decoder-only MLLMs**（如LLaVA系列），即vision encoder → projector (MLP) → LLM的pipeline。Cross-attention based MLLMs（如Flamingo, Qwen-VL）被排除在外，因为downsampling module（perceiver resampler）混淆了vision representation的作用。

**(2) Controlled variable assumption**：vision representation是唯一independent variable，alignment module和LLM architecture保持固定。如果vision encoder unfrozen，它实际上充当了alignment module的一部分，使实验uncontrolled。

这个assumption的严谨性很重要——它意味着Law适用于"selecting among frozen vision encoders"的场景，不适用于"end-to-end finetuning vision encoder"的场景。

---

## 3. A SCORE的精确定义与Intuition

### 3.1 公式

$$\text{A\_SCORE}(I, y) = \frac{1}{T} \sum_{t=1}^{T} \log P(y_t \mid f(I), y_{<t})$$

变量详解：
- $I$：input image
- $y = (y_1, y_2, \dots, y_T)$：与image配对的caption，由$T$个tokens组成
- $y_{<t}$：第$t$个token之前的所有tokens（autoregressive context）
- $f(I)$：projected visual representation，即vision encoder输出经过projector MLP后的vision tokens
- $P(y_t \mid f(I), y_{<t})$：frozen LLM在给定vision tokens和text prefix条件下，生成正确token $y_t$的概率
- $\log P$：log-likelihood（避免underflow，加法方便）
- $\frac{1}{T}$：对sequence length归一化

### 3.2 测量protocol

- Vision encoder + LLM **都frozen**，只训练projector（Stage 1 of LLaVA training）
- 这一步trainable parameters仅占0.298%，相比Stage 2 (train projector + LLM) computation可忽略
- Caption来自LLaVA-558K dataset
- 在100张randomly sampled images上average

### 3.3 Intuition

A SCORE本质是**"vision tokens被frozen LLM解读为正确text的能力"**。如果vision representation与language distribution pre-aligned得好，那么把vision tokens喂给frozen LLM，LLM自然能predict出正确的caption tokens。

这与"contrastive pretraining"的intuition一致——CLIP/SigLIP通过contrastive loss把image和text嵌入同一空间，本质上就是在做A SCORE优化的proxy。

### 3.4 与传统cross-modal alignment metric的区别

传统的cross-modal retrieval metric（如Image-to-Text R@1）需要labeled image-text pairs，且metric定义在encoder自身的embedding空间。而A SCORE直接测量**"在特定LLM的language space中vision tokens的有效性"**——它是conditional on LLM的，更贴近MLLM的实际使用场景。

参考：[CLIP](https://arxiv.org/abs/2103.00020), [SigLIP](https://arxiv.org/abs/2303.15343), [SigLIP 2](https://arxiv.org/abs/2502.14786)

---

## 4. C SCORE的精确定义与Intuition

### 4.1 公式

$$\text{C\_SCORE} = \frac{1}{m} \sum_{i=0}^{m} \mathbb{1}_{\|p_i^{pred} - p_i^{GT}\|_2 < T}$$

变量详解：
- $m$：一对image中annotated semantic key points的数量
- $p_i^{pred}$：用vision features预测的第$i$个key point在target image中的位置（通过feature similarity最大化得到）
- $p_i^{GT}$：第$i$个key point的ground truth位置
- $\|\cdot\|_2$：Euclidean distance
- $T$：threshold，proportional to目标object的bounding box size（即PCK metric的标准做法）
- $\mathbb{1}(\cdot)$：indicator function，条件满足返回1，否则0
- 整体是**PCK (Percentage of Correct Keypoints)** metric

### 4.2 计算流程

给定一对source-target image $(I_1, I_2)$ 和source image上的keypoints $K_1$：

1. 提取feature maps：$F_1 = E(I_1) \in \mathbb{R}^{l \times c}$, $F_2 = E(I_2) \in \mathbb{R}^{l \times c}$
   - $l$：sequence length (transformer) 或 $H \times W$ (grid)
   - $c$：hidden dimension
2. 计算similarity matrix：$S_{sim} = F_1 \cdot F_2^T \in \mathbb{R}^{l \times l}$
3. 对每个keypoint $k \in K_1$，在$I_2$中找similarity最大的位置作为predicted keypoint
4. 与ground truth keypoints比较，统计PCK

使用[SPair-71k dataset](https://arxiv.org/abs/1908.10543)，这是standard semantic correspondence benchmark。

### 4.3 Intuition

C SCORE测量的是**"vision features的intra-image semantic consistency"**。如果一只猫的左耳在image A和image B中（不同pose、不同光照）的feature vector相似，那么C SCORE高。

这解释了CLIP family"lacks visual details"的现象——CLIP的contrastive training只care about global image-text alignment，不care about local patch consistency。所以CLIP的C SCORE低（CLIP@336只有15.66）。而DINOv2通过self-supervised DINO loss + iBOT masked image modeling，explicitly保留了patch-level consistency，C SCORE高达24.51。

### 4.4 Semantic vs. Geometric Correspondence

Paper明确指出这里测的是**semantic correspondence**（matching same semantic concept across different instances），是**geometric correspondence**（matching exact same point across images，用于pose estimation、SLAM）。这个区别很关键：semantic correspondence关注"这是左耳vs左耳"，geometric correspondence关注"这是同一个3D点的projection"。

参考：[DINOv2](https://arxiv.org/abs/2304.07193), [Diffusion features for correspondence](https://openreview.net/forum?id=ypOiXjdfnU), [LightGlue](https://arxiv.org/abs/2306.13643)

---

## 5. AC SCORE的Quadratic Form

### 5.1 公式

$$\text{AC\_SCORE} = \sum_{\alpha=0}^{2} \sum_{\beta=0}^{2-\alpha} w_{\alpha\beta} A^{\alpha} C^{\beta}$$

变量详解：
- $\alpha, \beta$：polynomial的powers，$\alpha + \beta \leq 2$（保证是second-degree polynomial）
- $w_{\alpha\beta}$：trainable regression coefficients
- 展开共**6项**：$w_{00} + w_{10}A + w_{01}C + w_{20}A^2 + w_{11}AC + w_{02}C^2$

### 5.2 为什么是Quadratic？

Appendix A.3给出了intuitive解释：

- **Linear不够**：A和C不只是additive。DINOv2有高C但低A，单纯相加会被A的低分拉下来，但实际DINOv2的performance不算差。
- **Cross-term $AC$很关键**：它捕捉"synergy"——当A和C都高时，performance提升是super-additive的。
- **$A^2, C^2$捕捉diminishing returns / acceleration**：例如A极低时，提高C帮助不大（language barrier bottleneck）。
- **Higher degree overfits**：作者在ablation中尝试了更高degree，发现并不更好，反而obscure了A和C的individual contribution。

### 5.3 Interaction Term $\alpha_{AC}$的实验观察

Table 4展示了各benchmark的cross-term coefficient：

| Benchmark | $\alpha_{AC}$ | 类型 |
|-----------|---------------|------|
| MMBench | +1.1301 | Synergy |
| MME | +1.5819 | Synergy |
| MMMU | -0.4872 | Mild trade-off |
| OKVQA | -0.0364 | Mild trade-off |
| TextVQA | +4.9804 | Strong synergy |
| VizWiz | +2.9964 | Synergy |
| ScienceQA | +4.4644 | Strong synergy |
| SeedBench | +0.5772 | Synergy |

6/8 benchmarks显示**positive synergy**——A和C互相reinforce。MMMU和OKVQA有slight negative trade-off，可能因为这些task更依赖reasoning而非visual detail。

### 5.4 Empirical Trade-off Observation

虽然ideal scenario是A和C都高，但实际现有encoders呈现trade-off：

- **CLIP family**: high A (≈-1.97), low C (≈15.66) — 对齐好但视觉细节差
- **DINOv2**: low A (-2.32), high C (24.51) — 视觉细节好但对齐差
- **Diffusion features (SD1.5)**: medium A (-2.53), high C (22.02)
- **CLIP+DINOv2 combination**: medium A (-1.95), highest C (26.08)

这就是为什么**feature combination在empirical上work**——它把两个orthogonal的优势拼到一起。Paper第一次为这个empirical现象提供了theoretical framework。

---

## 6. Theoretical Justification

### 6.1 Cross-modal Alignment的理论（Appendix A.1）

**假设**：vision embedding distribution $D_{image}$ 和 text embedding distribution $D_{text}$ well-aligned，对shared concept $c$：

$$\|E_c^{image} - E_c^{text}\| \leq \epsilon$$

变量：
- $E_c^{image} \sim D_{image}$：concept $c$的image embedding（after projector）
- $E_c^{text} \sim D_{text}$：concept $c$的text embedding
- $\epsilon$：small constant

**关键性质**：pre-normed transformer是**Lipschitz continuous**（[Kim et al., 2021](https://arxiv.org/abs/2106.05283)证明self-attention是Lipschitz）。这意味着small input changes导致small output changes：

$$\|f([E_c^{image}, E_1, \dots, E_n]) - f([E_c^{text}, E_1, \dots, E_n])\| \leq L\epsilon$$

变量：
- $f$：language model function
- $L$：Lipschitz constant
- $E_1, \dots, E_n$：其他context tokens的embeddings

**结论**：用well-aligned vision representation训练MLLM，training dynamics ≈ text-only finetuning。这意味着：
1. 不需要extra capacity来bridge modality gap
2. Data efficiency提高（同样数据量下学得更好）
3. 不会"破坏"pretrained LLM的language能力

### 6.2 Correspondence的理论（Appendix A.2）

考虑input $[E_0^{image}, E_1^{image}, E_2, \dots, E_n]$，其中$E_0^{image}, E_1^{image}$来自high-correspondence representation的不同patches。

**Attention score公式**：

$$\text{score}(E_2, E_0^{image}) = \frac{(E_2 W^Q) \cdot (E_0^{image} W^K)}{\sqrt{d_k}}$$

变量：
- $E_2$：text token的embedding
- $W^Q \in \mathbb{R}^{d \times d_k}$：query projection matrix
- $W^K \in \mathbb{R}^{d \times d_k}$：key projection matrix
- $d_k$：key/query的dimension
- $\sqrt{d_k}$：scaling factor防止dot product过大

**Transitivity argument**：
如果 $\text{score}(E_2, E_0^{image})$ 高（text token attend to patch 0），且 $(E_0^{image} W^K)^T (E_1^{image} W^K)$ 也大（patch 0和patch 1的key相似），则 $\text{score}(E_2, E_1^{image})$ 也likely高。

**Intuition**：correspondence让attention通过**transitivity扩散**到semantic相关的visual regions。即使text token没有直接attend到某个patch，通过path transitivity也能间接retrieve它的信息。这解释了为什么high-correspondence features（如DINOv2）能让MLLM"看到"更多细节。

### 6.3 理论的局限性

这个theoretical analysis有几个caveats：
1. **Lipschitz argument是worst-case bound**，实际中$L\epsilon$可能很大
2. **Transitivity假设了$W^K$不distort vectors**——但实际中$W^K$是learned matrix，可能严重改变feature geometry
3. **没有量化A和C的具体functional form**——只是qualitative argument说"higher is better"

这些caveats解释了为什么paper还需要empirical fitting来得到quadratic form。

---

## 7. AC Policy的具体实现

### 7.1 Problem Formulation

给定$k$个candidate vision representations，原本需要finetune LLM $k$次。AC Policy只finetune $k' \ll k$次。

Regression model（公式4）：

$$\mathbf{y} = \mathbf{X}_s \mathbf{w} + \boldsymbol{\epsilon}$$

变量：
- $\mathbf{X} \in \mathbb{R}^{k \times 6}$：所有$k$个encoders的AC features（6个：$1, A, C, A^2, AC, C^2$）
- $\mathbf{X}_s \in \mathbb{R}^{k' \times 6}$：subsample后的training set
- $\mathbf{w} \in \mathbb{R}^{6}$：regression coefficients
- $\boldsymbol{\epsilon} \in \mathbb{R}^{k'}$：error terms
- $\mathbf{y} \in \mathbb{R}^{k'}$：observed benchmark performance

### 7.2 Region-based Sampling Strategy

直接random sampling可能sample到AC space相近的点，导致regression underdetermined。Region-based sampling的做法：

1. 将$k$个encoders的normalized $(A, C)$画在2D plane上
2. 第$j$次iteration将plane分成$4^j$个equal regions（quadtree-like subdivision）
3. 移除空regions和已采样过的regions
4. 从remaining regions随机选一个，从中随机选一个model

伪代码（Algorithm 2）：

```
function Region_based_Sampling(ACs, past_sampled, level):
    regions = {}
    for AC in ACs:
        region_key = determine_region(A, C, level)
        regions[region_key].append((model, A, C))
    remove past_sampled from regions
    remaining_regions = keys of regions
    chosen_region = random_select(remaining_regions)
    model = random_select(regions[chosen_region])
    return model
```

### 7.3 完整AC Policy流程（Algorithm 3）

```
function AC_Policy(V, k'):
    ACs = [(A_score(v), C_score(v)) for v in V]  # Compute all AC scores
    past_sampled = []
    train_ACs = []
    train_performance = []
    
    for i in 1 to k':
        model = Region_based_Sampling(ACs, past_sampled)
        performance = fully_train(model)  # Stage 1 + Stage 2
        train_ACs.append(AC of model)
        train_performance.append(performance)
        past_sampled.append(model)
    
    poly = PolynomialFeatures(degree=2)
    X_train = poly.fit_transform(train_ACs)
    regression = LinearRegression().fit(X_train, train_performance)
    ranking = rank V by regression.predict(poly.transform(ACs))
    return ranking
```

### 7.4 Computational Savings

- **Stage 1** (train projector only)：trainable parameters = 2-layer MLP，约0.298% of total
- **Stage 2** (train projector + LLM)：trainable parameters = 7B LLM + MLP
- 计算AC score只需要Stage 1（A SCORE用Stage 1后的frozen LLM评估；C SCORE只需要vision encoder）
- **Parameter ratio ≈ 0.003 → 99.7% reduction**

### 7.5 Experimental Results (Figure 4, 5)

- Budget = 4 full training runs：
  - Random subset selection: **26.7% Recall@3**
  - AC Policy: **87.72% Recall@3** (averaged over 6 benchmarks)
  - OKVQA最佳: **91.7% Recall@3**

- 1000次simulated ablation：要达到85.6% Recall@3，random需要train至少13/15个encoders
- AC Policy只需4次training就达到87.72%

参考：[AC Policy code](https://github.com/bronyayang/Law-of-Vision-Representation-in-MLLMs)

---

## 8. Experimental Setup详解

### 8.1 Vision Representations Tested (15 settings)

| 类别 | Encoder | Resolution |
|------|---------|------------|
| **Single feed-forward** | OpenAI CLIP ViT-L/14 | 224, 336 |
| | OpenCLIP ViT-L/14 | 224 |
| | DINOv2 ViT-L/14 | 224 |
| | SigLIP ViT-B/16 | 224 |
| | SigLIP ViT-L/16 | 256 |
| | SigLIP2 ViT-L/16 | 256 |
| **Single diffusion** | SD 1.5 | 768 |
| | SD 2.1 | 768 |
| | SD Image Variations | - |
| | SD XL | 512 |
| | DiT | 512 |
| | SD 3 | 512 |
| **Combination** | CLIP+DINOv2 ViT-L/14 | 224, 336 |

### 8.2 Diffusion Features Extraction（公式5）

$$x_t = \sqrt{a_t} \cdot \text{VAE}(I) + \sqrt{1-a_t} \cdot \epsilon$$

变量：
- $x_t$：noised latent at timestep $t$
- $I \in \mathbb{R}^{H \times W \times 3}$：input image
- $\text{VAE}(I)$：VAE encoder的output latent
- $a_t$：noise schedule coefficient at timestep $t$（控制保留多少original signal）
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$：standard Gaussian noise
- $t$：timestep，paper采用**little-noise strategy** $t=1$

**关键设计**：$t=1$意味着只加minimal noise，diffusion model只denoise一次，one-step denoising latents作为vision features。这避免了multi-step denoising的computational overhead，同时retains了diffusion model的semantic features。

这个设计与[Diffusion features work](https://arxiv.org/abs/2306.08857)和[Deem](https://arxiv.org/abs/2405.15232)的发现一致——diffusion model的intermediate features具有excellent correspondence properties。

### 8.3 U-Net vs Transformer Features

- **Transformer models (CLIP, DINOv2, SigLIP, DiT)**：提取last hidden state $F \in \mathbb{R}^{l \times c}$
  - $l$：sequence length (number of patches)
  - $c$：hidden dimension
  
- **U-Net models (SD 1.5, SD 2.1, SDXL)**：提取first upsampling block后的activation $F \in \mathbb{R}^{\hat{H} \times \hat{W} \times c}$
  - $\hat{H}, \hat{W}$：feature map spatial dimensions
  - $c$：channels

两种format可通过reshape/flatten互换。

### 8.4 Benchmarks (8 total)

**Vision-based (4)**：
- [MMBench](https://arxiv.org/abs/2307.06281)：20 ability dimensions的multiple-choice
- [MME](https://arxiv.org/abs/2306.13394)：yes/no questions on existence, counting, position, color
- [OKVQA](https://arxiv.org/abs/1906.00067)：open-ended VQA requiring external knowledge
- [SEED-Bench](https://arxiv.org/abs/2307.16125)：spatial + temporal understanding

**OCR-based (4)**：
- [MMMU](https://arxiv.org/abs/2311.16502)：college-level subject knowledge
- [TextVQA](https://arxiv.org/abs/1904.08920)：text reading VQA
- [VizWiz](https://arxiv.org/abs/1802.08217)：blind users的VQA
- [ScienceQA](https://arxiv.org/abs/2209.10658)：science questions with images

---

## 9. Key Experimental Results

### 9.1 R² Values (Table 2)

| Fitting Data | R² (Vision) | R² (OCR) |
|--------------|-------------|-----------|
| **No transformation** | | |
| Random | 4.03% | 1.75% |
| A Score only | 80.53% | 58.00% |
| C Score only | 39.02% | 14.57% |
| AC Score | 80.55% | 62.06% |
| **Polynomial (degree=2)** | | |
| Random | 36.90% | 31.26% |
| A Score only | 91.48% | 79.67% |
| C Score only | 56.56% | 30.11% |
| **AC Score** | **94.06%** | **83.85%** |

**Key observations**：
1. AC Score with polynomial transformation达到最高R²
2. C Score alone比A Score差很多——说明alignment是更dominant的因素
3. 但A Score alone比AC Score差3%——C Score的contribution是significant的
4. OCR benchmarks的R²普遍低于Vision benchmarks——OCR需要不同的features

### 9.2 Cross-LLM Generalization (Table 3)

| LLM Backbone | R² (Vision) | R² (OCR) |
|--------------|-------------|-----------|
| Vicuna 1.5 7B | 94.06% | 83.85% |
| Qwen 2.5 14B | 91.50% | 82.27% |

Law在不同LLM backbone上都hold，R²都>80%。

### 9.3 AC Scores of All Encoders (Table 6)

| Encoder | C Score (PCK@0.10) | A Score (Log Likelihood) |
|---------|---------------------|--------------------------|
| CLIP@336 | 15.66 | -1.97 |
| CLIP@224 | 14.30 | -1.98 |
| OpenCLIP | 16.22 | -1.93 |
| SigLIP-B | 12.89 | -1.92 |
| SigLIP-L | 13.66 | -1.83 |
| C+D@224 | 23.62 | -1.96 |
| C+D@336 | 26.08 | -1.95 |
| SigLIP2-L | 16.75 | -1.81 |
| DINOv2 | 24.51 | -2.32 |
| DiT | 1.91 | -3.76 |
| SDXL | 16.52 | -2.69 |
| SD3 | 3.09 | -4.13 |
| SD2.1 | 6.99 | -2.81 |
| SD1.5 | 22.02 | -2.53 |
| SDim | 20.90 | -2.37 |

**Critical insights**：
- **SigLIP2-L**：A最高(-1.81)，C中等(16.75) — 当前最强single encoder
- **DINOv2**：C最高之一(24.51)，A很低(-2.32) — 单独用performance差
- **C+D@336**：C最高(26.08)，A中等(-1.95) — combination策略最优
- **DiT/SD3**：A和C都极差 — rectified flow模型features不适合做MLLM vision encoder
- **SD1.5**：C很高(22.02)，A中等(-2.53) — latent diffusion的features surprisingly good

### 9.4 Performance of All Encoders (Table 5)

Vision-based benchmarks (MMBench/MME/OKVQA/SEED-Bench)上：
- SigLIP2-L和C+D@336表现最好
- DINOv2单独使用MMBench只有58.50（vs CLIP@336的64.26）
- DiT/SD3表现极差（MMBench 32-33）

OCR-based benchmarks (MMMU/TextVQA/VizWiz/ScienceQA)上：
- SigLIP2-L在TextVQA达47.2，超过C+D@336的46.17
- DINOv2在TextVQA崩溃到14.27
- Diffusion features在TextVQA全部崩溃（10-13）— 这是关键limitation

---

## 10. Limitations与OCR问题

### 10.1 The OCR Problem

OCR-based benchmarks与AC score相关性低（R² 83.85% vs Vision 94.06%），原因：

1. **SPair-71k只有natural images**：cats, trains, etc.，没有text-heavy images
2. **CLIP对text有special correspondence**：通过contrastive learning with OCR-like captions学到
3. **LLaVA-558K captions缺少OCR/numeric/symbolic data**

Figure 6可视化了这个差异：
- DINOv2在natural images上correspondence完美
- CLIP在text-containing images上correspondence完美（能match所有"LLaVA"或"VQAv2"文本）
- DINOv2和diffusion features在text images上完全失败

### 10.2 具体例子：SigLIP2 vs CLIP

- SigLIP2-L@256：A=-1.81, C=16.75
- CLIP-L@336：A=-1.97, C=15.66

SigLIP2在A和C上都优于CLIP，但在MME上表现差，因为MME是OCR-heavy。这exposes了AC framework的limitation——当benchmark domain与AC score的measurement domain不match时，预测失败。

### 10.3 Future Work Suggestion

Paper呼吁社区构建OCR-specific correspondence dataset和systematically designed OCR short caption datasets。这对于tables, charts, document understanding至关重要。

---

## 11. Intuition总结与深度联想

### 11.1 The "Two Orthogonal Axes" Intuition

可以把A和C看作两个orthogonal axes：
- **A axis**：vision features与language space的"翻译通畅度"
- **C axis**：vision features内部的"语义结构保持度"

理想的vision representation需要两者都高，但实际中training paradigm决定了一个trade-off：
- Contrastive learning (CLIP) → 优化A，牺牲C
- Self-supervised DINO (DINOv2) → 优化C，牺牲A
- Diffusion features → C高，A取决于具体architecture
- **Combination是empirical workaround**，paper提供了theoretical justification

### 11.2 为什么Quadratic而非Linear

Linear假设A和C是independent additive contributions。但实际上：
- 当A极低时，C的提升几乎没用（language bottleneck）— $C^2$ or $AC$ term会体现这点
- 当A高时，C的提升amplify效果 — positive synergy
- 这是"multiplicative interaction"，需要cross term $AC$

这与[ResNet中的residual connection](https://arxiv.org/abs/1512.03385)有类似intuition：identities和transformations是multiplicative的，单纯additive model无法捕捉。

### 11.3 Connection to Information Theory

A SCORE本质是**conditional log-likelihood**，与**mutual information** $I(Y; f(I))$相关：

$$I(Y; f(I)) = H(Y) - H(Y | f(I))$$

最大化$\log P(y|f(I))$等价于最小化conditional entropy $H(Y|f(I))$，等价于maximizing mutual information。这与[InfoNCE](https://arxiv.org/abs/1807.03748)的contrastive learning objective一致——CLIP的training本质就是在优化A SCORE的proxy。

### 11.4 Connection to Mechanistic Interpretability

Correspondence的transitivity argument（Appendix A.2）实际上是attention mechanism的一个mechanistic interpretation：high-correspondence features让attention head能间接retrieve semantic-related patches。这与[induction heads in LLMs](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/)的工作有conceptual parallel——都是通过feature similarity实现information routing。

### 11.5 Connection to Sparse Distributed Representations

A高C低的CLIP features是"sparse distributed"——每个patch feature都有language semantics但patch间不distinguishable。C高A低的DINOv2 features是"dense localized"——patch间well-distinguished但缺乏language anchor。Combination相当于在两种representation间interpolate，类似于[Polysemantic neurons vs monosemantic neurons](https://distill.pub/2020/circuits/zoom-in/)的debate。

### 11.6 Diffusion Features的Surprising Finding

SD1.5的C Score (22.02) 仅次于DINOv2和combinations，这很surprising。可能的解释：
- Diffusion model的denoising objective implicitly学习了patch-level consistency
- VAE latent space的locality preservation
- Cross-attention layers在text-conditioned generation中学到了semantic grouping

这与[Emergent Correspondence from Image Diffusion](https://arxiv.org/abs/2306.03881)的发现一致——diffusion features在correspondence任务上出奇好。

### 11.7 为什么Cross-attention MLLMs被排除

Cross-attention架构（如Flamingo）有perceiver resampler，它对vision features做downsampling和aggregation。这意味着：
1. Vision encoder和perceiver resampler共同构成"effective vision representation"
2. 改变vision encoder会同时改变resampler的作用
3. Variables not controlled，Law的mathematical formulation不成立

这是paper严谨性的体现——明确scope比overclaim好。

### 11.8 AC Policy与Bayesian Optimization的对比

AC Policy本质是**surrogate model-based optimization**，类似于Bayesian Optimization with quadratic surrogate。但区别在于：
- Bayesian Optimization通常用Gaussian Process，提供uncertainty estimate
- AC Policy用simple polynomial regression，uncertainty估计crude
- Region-based sampling ≈ space-filling design (like Latin Hypercube Sampling)

这个connection提示可能的extension：用GP替代polynomial，可能得到更好的uncertainty quantification和更好的exploration-exploitation trade-off。参考：[Bayesian Optimization tutorial](https://arxiv.org/abs/1807.02811)

### 11.9 与Scaling Laws的对比

这篇paper的title "Law of Vision Representation"明显evokes [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)和[Kaplan et al. scaling laws](https://arxiv.org/abs/2001.08361)。但区别在于：
- Scaling laws：performance ~ compute, parameters, data (continuous variables)
- Vision Representation Law：performance ~ A, C (latent quality factors)

这种"latent factor law"的形式更接近[Physics-inspired laws in deep learning](https://arxiv.org/abs/2102.02774)的spirit——找到observable proxies for latent qualities。

### 11.10 Practical Implications for MLLM Development

对于practitioners，这篇paper的take-home messages：
1. **不要盲目换encoder**——先算AC score，predict performance
2. **Combination不一定总是好**——取决于A和C的trade-off
3. **OCR tasks需要specialized measurement**——不要用natural image的AC score预测OCR performance
4. **Vision encoder的选择应该task-aware**——vision-heavy task优先C高，language-heavy task优先A高
5. **新encoder开发方向**：同时优化contrastive loss (A) 和 self-supervised consistency loss (C)

### 11.11 与CLIP is Blindly Routing的Critique

[Eyes Wide Shut?](https://arxiv.org/abs/2401.06309) (Tong et al., 2024)批判CLIP家族"lacks visual detail"。这篇paper提供了quantitative explanation：CLIP的C Score确实低（15.66 vs DINOv2的24.51）。但paper也说CLIP的A Score高，所以在不需要visual detail的task上CLIP依然强。这refine了社区的understanding——不是CLIP"blind"，是CLIP"language-fluent但visually-impaired"。

### 11.12 与Prismatic VLMs的设计一致性

[Prismatic VLMs](https://arxiv.org/abs/2402.07865)通过systematic ablation发现CLIP+DINOv2 combination是最强的，但他们没解释why。这篇paper的Law of Vision Representation给出了quantitative explanation——CLIP提供A，DINOv2提供C，combination maximizes AC score。这是empirical work被theoretical framework解释的good example。

### 11.13 Connection to Multimodal Pretraining Dynamics

A SCORE的formulation与[multimodal pretraining objectives](https://arxiv.org/abs/2102.03334)有deep connection：
- Contrastive loss (CLIP) → 直接优化image-text alignment
- Generative loss (BLIP, COCA) → 优化conditional log-likelihood，本质是A SCORE
- VQA-style losses → 优化task-specific A SCORE

所以A SCORE可以作为统一的evaluation metric，横跨不同pretraining paradigm。

### 11.14 The "Alignment Tax" Hypothesis

Anthropic在[Constitutional AI](https://arxiv.org/abs/2212.08073)中提出"alignment tax"概念——RLHF会hurt model capability。在vision representation中也有类似phenomenon：contrastive alignment with text (优化A) 会hurt visual correspondence (C)。这是"alignment tax"的multimodal版本。Paper的Table 6数据支持这点：CLIP (high A) C低，DINOv2 (high C) A低。

### 11.15 与Tokens-per-Image的Trade-off

Paper没explore但值得思考：higher resolution → more vision tokens → higher C (更多spatial detail) 但可能lower A (more tokens to compress into language space)。这可能是AC trade-off的一个source。这与[NaViT](https://arxiv.org/abs/2307.06304)和[LLaVA-NeXT](https://arxiv.org/abs/2401.02914)的any-resolution design相关——他们empirically发现higher resolution helps，但没theoretical explanation。AC framework可能能解释。

### 11.16 Future Directions I'd Speculate

基于这篇paper的framework，可能的future work：

1. **Differentiable AC Score**：把A和C SCORE做成differentiable loss，直接end-to-end训练vision encoder
2. **Multi-modal AC**：扩展到audio, video等modality
3. **Task-conditional AC**：不同task可能需要不同A-C weighting，learn task-conditional $w_{\alpha\beta}$
4. **AC-guided architecture search**：用AC score作为NAS的surrogate
5. **Dynamic AC**：在inference时根据input动态选择vision encoder
6. **AC for Video MLLMs**：temporal correspondence需要新的C SCORE定义
7. **OCR-specific AC**：构建text-heavy correspondence dataset

### 11.17 Critique of the Paper

客观地看，这篇paper有几个weakness：
1. **15 settings偏少**——quadratic fit在6个parameters上，R²=94%可能overfit
2. **C SCORE只用SPair-71k**——一个dataset不够robust
3. **A SCORE只用100 images**——sample size偏小
4. **Theoretical justification是qualitative的**——没有quantitative bound
5. **Cross-LLM generalization只测了2个LLM**——不够comprehensive
6. **OCR limitation没solved**——只是acknowledged

但overall这是solid empirical work，为MLLMs community提供了first quantitative framework for vision representation selection。

---

## 12. 最终Take-away

**核心发现**：MLLM performance ≈ quadratic function of (A, C)，其中A = cross-modal alignment，C = visual correspondence。

**Practical价值**：用AC Policy可以99.7% reduce cost of vision encoder selection，从exhaustive finetuning变成surrogate model-based prediction。

**Theoretical价值**：第一次为"CLIP lacks visual detail"和"feature combination works"提供了quantitative explanation。

**Limitation**：OCR-heavy tasks的prediction不准，需要domain-specific AC measurement。

**Broader impact**：这种"latent factor law"思路可能inspire其他modality或component的similar laws——例如"Law of Projector Design", "Law of Instruction Tuning Data Quality"等。

参考资源：
- [Paper GitHub](https://github.com/bronyayang/Law-of-Vision-Representation-in-MLLMs)
- [LLaVA](https://llava-vl.github.io/)
- [CLIP](https://openai.com/research/clip)
- [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [DINOv2](https://dinov2.metademolab.com/)
- [SPair-71k](https://arxiv.org/abs/1908.10543)
- [Prismatic VLMs](https://arxiv.org/abs/2402.07865)
- [Cambrian-1](https://arxiv.org/abs/2406.16860)
- [Eyes Wide Shut](https://arxiv.org/abs/2401.06309)
- [Diffusion Correspondence](https://openreview.net/forum?id=ypOiXjdfnU)
- [Scaling Laws](https://arxiv.org/abs/2203.15556)
- [Bayesian Optimization](https://arxiv.org/abs/1807.02811)
