---
source_pdf: Good Token Hunting A Hitchhiker’s Guide to Token Selection for Visual
  Geometry Transformers.pdf
paper_sha256: 0b0e1407e5fa8d00604cf31dcaf21c7b1e43497d498061c7261f87b455cca650
processed_at: '2026-08-04T22:01:22-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

Andrej，咱们把那些公式和表格都扔一边，我就用大白话给你讲讲这帮人到底干了啥。

---

## 这事儿到底是个什么问题

你想想，现在有一类模型叫visual geometry transformers，比如VGGT、π³这些。你给它N张照片，它一把forward pass就能吐出来camera pose、depth map、point cloud这些东西。听起来很爽对吧？

但有个要命的瓶颈：**global attention的复杂度是 $\mathcal{O}(N^2 L^2)$**。N是帧数，L是每帧token数。你喂100张图还好，喂500张图就要288秒。喂1000张？显卡直接炸了。

本质上就是，每个query token要跟所有N×L个key/value tokens做attention。N一大，就quadratic爆炸。

---

## 他们怎么解决的

思路特别朴素：**你别让每个query跟所有tokens交互了，挑一部分出来交互就行了**。

但关键问题是——挑哪些？这就是整篇paper的核心。

他们搞了个两阶段的hierarchical selection：

### 第一阶段：Inter-frame selection（选帧）

你500张图，不可能全用来做global attention吧？选25张出来。

那怎么选？他们试了一堆intuitive的策略，全崩了：

- **选时间上挨得近的**：全是相似画面，scene覆盖一塌糊涂
- **选跟当前帧最相似的**：全看同一个区域，其他区域没人管
- **选attention score最高的**：不同query想看的东西不一样，选不出统一的anchor set

最后work的策略是：**用place recognition model提取feature，然后做diversity-based selection**。

说白了就是：**选出来的25张图，要尽可能"散开"覆盖整个scene**。你站在一个房间拍了500张，我选25张，让你从门口、墙角、窗户边、桌子旁边各选几张，保证整个房间都被覆盖到。

具体用Farthest Point Sampling（FPS）——贪心地每次选离已选集合最远的那张。这个算法在point cloud processing里用了几十年了，经典得很。

**最关键的一点：选出来的这25帧是所有query共享的**。每个query token都attend到同一组25帧上。这保证了cross-view information aggregation的一致性。

### 第二阶段：Intra-frame selection（帧内token pruning）

选了25帧还不算完，每帧内部还能再砍。

但这里有个巨关键的发现：**不同layer的attention pattern完全不一样**。

他们分析了VGGT的24个global attention layers，发现：

- **Early layers（0-2层）**：attention几乎是uniform distribution，每个key token的weight都差不多。说白了就是attention在"摸鱼"，啥也没区分出来
- **Middle layers（9-16层）**：attention开始spiking，某些token的权重特别高，其他接近0
- **Late layers**：又有一些spiking，但不如middle那么极端

这个发现直接决定了pruning策略要**layer-adaptive**：

- Early layers既然attention是uniform的，那丢掉一些token无所谓——反正大家weight都差不多。甚至可以直接把global attention换成local attention（只在帧内做），因为cross-view信息在这些layer本来就没被有效利用
- Middle layers有spiking，你要是uniform downsampling把high-activation token丢了，attention pattern直接崩了，性能就掉

所以他们的final design是分三段：

```
Layer 0 到 l_local-1:  用local attention替换global attention（最aggressive）
Layer l_local 到 l_sample-1:  downsampling by factor σ（moderate）
Layer l_sample 到 end:  保留selected frames的全token（保守）
```

默认 $l_{\text{local}}=2$, $l_{\text{sample}}=9$。

---

## 结果如何

用大白话说：**又快又好**。

500帧的场景：
- VGGT base：288秒
- GoToHunt：41秒，快了85%
- 而且性能没掉，有些指标还**超过**了base model

对比其他方法：
- **FastVGGT**：training-free但比GoToHunt慢一倍
- **LiteVGGT**：速度快一点点但要expensive retraining，GoToHunt是training-free的
- **SparseVGGT**：在高sparsity下直接OOM了
- **Speed3R**：要retraining，而且只适用于π³

更夸张的是，有些实验里GoToHunt的accuracy比原始VGGT还好。这说明什么？**原始VGGT的global attention其实是over-parameterized的**，很多attention计算其实是noise，砍掉反而更好。

---

## 为什么work的intuition

我帮你理一理最核心的intuition：

**1. Inter-frame层面：diversity > similarity**

这个道理其实很深。你想想SLAM里keyframe selection几十年前就明白了：你要的是information coverage，不是redundant information。选25张全是同一个角落的照片，不如选25张分散在整个scene的照片。

visual geometry transformer的global attention本质上是在做cross-view information aggregation。每个query想找的是"跟我最相关的那张参考图"。如果你选的anchor frames能覆盖整个scene，每个query都能找到自己的"知音"。

**2. Intra-frame层面：attention pattern的layer-wise特性**

这个发现很有意思。early layers的attention近uniform，说明这些layer在做cross-view aggregation方面没什么实质贡献。你把global attention换成local attention（只在帧内做），loss几乎为零。

middle layers才有spiking，说明真正的cross-view information aggregation发生在这里。这些layer不能随便prune。

这跟language model里的发现一致——early layers的attention确实是diluted的。

**3. Training-free但能match甚至beat retraining方法**

这说明现有的visual geometry transformers架构可能根本就没设计对。如果token selection能让模型更好，那说明原始的全attention包含很多redundant甚至harmful的information。

---

## 几个我觉得特别妙的点

**1. Shared anchor set across all queries**

所有query token用同一组25帧做attention。这看似限制了flexibility，但实际work得很好。Intuition是：一个好的anchor set应该是scene-level的，每个query都能从中找到relevant的信息。

**2. FPS的效率**

FPS只需要一次 $N \times N$ 的similarity计算 + K次迭代更新。对于N=500, K=25，基本秒级完成，相比attention本身的计算量可以忽略。

**3. Robustness to hyperparameters**

Table 9显示，$l_{\text{local}}$ 从1到4，$l_{\text{sample}}$ 从8到10，性能波动很小。这说明策略的设计是对的——只要大致符合attention pattern的观察，具体阈值不太敏感。

---

## 几个我没想明白的点

**1. 为什么K增大到一定程度性能反而降？**

Table 8显示K从25增到40-60时性能最好，但增到100反而略降。作者说这是future investigation。我的猜测是：更多frames引入了更多redundant甚至conflicting information，attention的noise变多了。

**2. 为什么GoToHunt能超过base model？**

这个挺反直觉的。Token selection相当于给模型加了inductive bias：只看diverse的anchor frames。如果原始模型在全attention时会被redundant frames干扰，那pruning掉那些redundant frames可能反而帮模型focus到有用的信息上。

**3. Place recognition model的dependency**

他们用MegaLoc提取feature做co-visibility approximation。如果scene很symmetric或者object-centric，place recognition的feature可能unreliable。作者在Section G承认了这个limitation。

---

## 对未来的启发

这paper最大的价值可能不是GoToHunt本身，而是它揭示的insight：

**1. 架构设计上**

现有visual geometry transformers的early global attention layers基本在"摸鱼"。既然如此，为什么不在training时就设计成early layers用local attention，later layers才用global attention？省compute还可能提升性能。

**2. Routing-based attention**

Inter-frame selection用offline的FPS做的。如果train一个轻量级router network，per-query地预测该attend哪些frames，可能更灵活。类似MoE的routing思想。

**3. 和TTT方法的结合**

Concurrent works用Test-Time Training layers替换quadratic attention实现linear scaling。GoToHunt和这些方法complementary——可以在TTT框架内进一步做token selection。

---

## 一句话总结

**用diversity-based frame selection + layer-adaptive token pruning，training-free地把visual geometry transformers加速85%以上，性能还能保持甚至提升**。

核心insight就两个：
1. 选帧要选diverse的不要选相似的
2. 不同layer的attention pattern不一样，pruning策略要跟着变

就这么简单，但它work得很好。

---

## References

- Project page: https://zsh2000.github.io/good-token-hunting.github.io/
- VGGT repo: https://github.com/facebookresearch/vggt  
- π³ repo: https://github.com/yyfz/Pi3
- DUSt3R: https://github.com/naver/dust3r
- K-center FPS原始paper: https://www.sciencedirect.com/science/article/pii/0304397585900225

---

# Good Token Hunting: 深度技术解析

## 1. Paper的核心动机和问题背景

Andrej, 这篇paper处理的是visual geometry transformers的效率问题。让我先build up the intuition关于为什么这是个hard problem。

**Visual Geometry Transformers的架构背景:**
VGGT [83], π³ [87], MapAnything [41], Depth Anything 3 [52] 这类模型的核心设计是: 给定N张图片 $\mathcal{I} = \{I_i\}_{i=1}^N$, 通过一个feedforward pass同时预测camera pose $[\mathbf{R}_i | \mathbf{t}_i]$, point maps $\mathbf{P}_i$, depth maps等3D属性。

架构由两类attention layer交替组成:
- **Frame-wise attention**: 在每帧内部独立操作, 复杂度 $\mathcal{O}(N \cdot L^2)$
- **Global attention**: 跨所有帧的所有tokens做attention, 复杂度 $\mathcal{O}(N^2 L^2)$

其中 $N$ 是帧数, $L$ 是每帧的token数。当 $N$ 变大(比如500帧), global attention成为dominant bottleneck, 因为它是quadratic in $N$。

**Key insight**: 这个瓶颈的本质是, 每个query token需要和所有 $N \times L$ 个key/value tokens做attention。如果我们能限制每个query只和一部分key/value tokens交互, 复杂度就能降到 $\mathcal{O}(N \cdot K \cdot L')$ 其中 $K \ll N$ 是selected frames数, $L' \leq L$ 是每帧selected tokens数。

---

## 2. Problem Formulation的精妙之处

Paper的核心formulation极其简洁:

> 在global attention layers中, 限制每个query token能attend的key/value token数量。

这个formulation的general性在于它不依赖具体架构。任何用global attention的visual geometry transformer都可以套用。

**为什么不用直接从全部 $N \times L$ tokens中选?** 因为这需要先计算所有tokens的features再做selection, 计算量太大。所以作者采用hierarchical的两阶段策略:
1. **Inter-frame selection**: 先在frame level选 $K$ 个frames
2. **Intra-frame selection**: 在每个selected frame内部进一步prune tokens

这种coarse-to-fine的hierarchical设计使得selection的计算成本很低。

---

## 3. Inter-frame Selection: 为什么Diversity是关键

### 3.1 Intuitive strategies为什么失败

Paper做了一个非常informative的ablation study (Table 1), 测试了几种intuitive strategies, budget $K=25$ from 250/500 frames:

| Strategy | ATE | RPE-rot | RPE-trans |
|----------|-----|---------|-----------|
| Temporal nearest | 0.7588 | 1.8485 | 0.0563 |
| High co-visibility | 0.3813 | 2.9934 | 0.1197 |
| Low co-visibility | 0.1840 | 2.4761 | 0.1038 |
| Max attention pool | 0.3879 | 7.2494 | 0.1257 |
| Mean attention pool | 0.3627 | 7.2988 | 0.0988 |
| **Diversity (Ours)** | **0.0676** | **0.4421** | **0.0167** |
| VGGT base | 0.0698 | 0.4953 | 0.0178 |

**Intuition building**: 
- **Temporal proximity**失败因为相邻frames包含redundant information, 无法覆盖整个scene
- **High co-visibility**失败因为选出来的frames都看向同一区域, scene coverage差
- **Low co-visibility**看似合理但失败, 因为可能选到outlier frames
- **Attention-based**失败因为attention scores在不同query之间不一致, 难以选出shared anchor set

### 3.2 Diversity-based Frame Selection的数学formulation

给定 $N$ 张图片, 用place recognition model [4] (MegaLoc)提取 $d$-维features $\{f_i\}_{i=1}^N$。定义cosine distance:

$$d(i, j) = 1 - \frac{\langle f_i, f_j \rangle}{\|f_i\|_2 \|f_j\|_2}$$

其中:
- $f_i, f_j \in \mathbb{R}^d$ 是图片 $i$ 和 $j$ 的feature vectors
- $\langle \cdot, \cdot \rangle$ 是inner product
- $\|\cdot\|_2$ 是L2 norm
- $d(i,j) \in [0, 2]$, 值越大表示两帧越不相似

目标是在budget $K$ 下找到subset $S^* \subseteq \{1, ..., N\}$ with $|S^*| = K$, 最小化任何frame到其最近selected frame的最大距离:

$$S^* = \underset{S \subseteq \{1,...,N\}, |S|=K}{\arg\min} \underset{i \in \{1,...,N\}}{\max} \underset{j \in S}{\min} d(i, j)$$

这个objective的intuition: 我们希望选出的 $K$ 个frames能"覆盖"整个view space, 使得任何一帧到其最近的"anchor frame"都不会太远。这就是经典的**K-center problem**。

### 3.3 Farthest Point Sampling (FPS)算法

K-center是NP-hard问题, 作者采用greedy FPS heuristic [30]:

```
Algorithm A: Diversity-based Inter-frame Selection
Input: Feature map F ∈ R^(N×d), budget K, random seed σ
Output: Selected index set S with |S| = K

1. Normalize: f̃_i = f_i / ||f_i||_2
2. Build similarity matrix: C = F̃ F̃^T  (cosine similarity)
3. Convert to distance: D_ij = max(C) - C_ij
4. Random initial pick: b_1 ~ Uniform{0,...,N-1}
5. S = {b_1}, d_min = D_{b_1,:}
6. For k = 2 to K:
   - b = argmax_j d_min[j]  (选最远的frame)
   - S = S ∪ {b}
   - d_min[j] = min(d_min[j], D_{b,j})  (更新距离)
7. Return S
```

**计算复杂度分析**:
- Similarity matrix: $\mathcal{O}(N^2 d)$ (一次性)
- FPS iteration: $\mathcal{O}(K \cdot N)$
- 总计: $\mathcal{O}(N^2 d + KN)$, 对于 $N=500, K=25$ 非常efficient

**Key insight**: 这个selected set $S$ 是shared across all query tokens的。这意味着所有queries都attend到同一组"anchor frames"。这个设计很重要, 因为它保证了cross-view representation processing的一致性。

### 3.4 为什么Diversity work的深层原因

我的理解是, 这和**keyframe-based SLAM** [40, 48]的设计哲学一致。在SLAM中, keyframe selection的目标也是maximize information coverage而不是选最多features的frames。

Visual geometry transformers的global attention本质上是在做cross-view information aggregation。如果选出的frames能覆盖整个scene, 那么每个query token都能找到和自己最相关的参考view。相比之下, 选co-visible frames会导致redundant information, 而选attention-activated frames可能偏向某些dominant regions。

---

## 4. Intra-frame Selection: Layer-adaptive策略

### 4.1 Uniform downsampling的问题

有了selected frames, 下一步是intra-frame token pruning。AVGGT [76]的做法是在所有global attention layers做uniform downsampling, factor $\sigma$:

将 $h \times w$ 的token map downsample到 $\lfloor \frac{h}{\sigma} \rfloor \times \lfloor \frac{w}{\sigma} \rfloor$。

Table 2显示即使 $\sigma=2$ 也有明显performance drop:
- $K=25, \sigma=2$: ATE=0.0831 (base: 0.0698)
- $K=25, \sigma=3$: ATE=0.1393

### 4.2 Attention Pattern Analysis: 关键发现

这是paper最informative的部分。作者分析了VGGT的24个global attention layers的attention pattern, 用两个统计量:

**Normalized Entropy**:
$$\mathcal{H}_{\text{norm}} = \frac{\sum_{0 \leq h < H, 0 \leq q < Q} \mathcal{H}(h, q)}{H \cdot Q \cdot \mathcal{H}_{\max}}$$

其中:
- $\mathcal{H}(h, q)$ 是attention head $h$, query token $q$ 的attention score distribution的Shannon entropy
- $H = 4$ 是sampled attention heads数量
- $Q = 50$ 是sampled query tokens数量
- $\mathcal{H}_{\max} = \log(NL)$ 是maximum possible entropy (uniform distribution over all $N \times L$ key tokens)
- $\mathcal{H}_{\text{norm}} \in [0, 1]$, 越接近1表示attention越diluted/uniform

**Top-1 token weight**: attention score中最大那个token的weight, 衡量attention的spikiness。

**关键观察** (Figure 4):
- **Layer 0-2**: $\mathcal{H}_{\text{norm}} \approx 1$, attention极其diluted, 近似uniform distribution
- **Layer 3-8**: 逐渐从diluted过渡, 但entropy仍然较高
- **Layer 9-16**: entropy较低, 出现spiking attention values
- **Layer 17-23**: 又有所回升, 但仍有spiking

### 4.3 这个pattern和language models的类比

Paper提到这个现象在language models中也有 [15, 102, 111]。我的理解是:

在transformer的early layers, token的representations还很general, query和key的dot product分布比较uniform, 所以attention接近uniform distribution。随着layers加深, token学到更specific的semantic features, 某些key tokens变得特别relevant, attention开始spiking。

这个观察的**直接implication**: 
- **Early layers**: attention近uniform, 丢掉任何token的影响大致相同, 可以aggressive pruning
- **Middle/late layers**: attention有spiking values, 如果丢掉high-activation tokens会严重破坏attention pattern

Table 3验证了这个hypothesis:
- Layer 9-16, $\sigma=2$, Standard (uniform downsample): ATE=0.0792
- Layer 9-16, $\sigma=2$, Activation (保留high-activation tokens): ATE=0.0687
- Activation strategy明显更好, 说明middle layers确实有重要tokens不能随意丢弃

### 4.4 Layer-adaptive策略的设计

基于以上分析, 作者引入两个thresholds:
- $l_{\text{local}}$: 对于layer $l < l_{\text{local}}$, 用local attention替换global attention
- $l_{\text{sample}}$: 对于layer $l_{\text{local}} \leq l < l_{\text{sample}}$, 用downsampling (factor $\sigma$)
- 对于layer $l \geq l_{\text{sample}}$, 保持原始global attention

**Default设置**: $l_{\text{local}} = 2, l_{\text{sample}} = 9$

**为什么用local attention替换early layers?** 因为当 $\mathcal{H}_{\text{norm}} \approx 1$ 时, global attention本质上是在做average pooling over all tokens, 但cost是 $\mathcal{O}(N^2 L^2)$。换成local attention (只在帧内做attention) cost降到 $\mathcal{O}(N \cdot L^2)$, 而且因为这些layers本来就不怎么利用cross-view信息, 性能影响很小。

### 4.5 为什么不在late layers也做pruning

Paper在Appendix B.4探索了这个方向 (Table G, H), 引入 $l_{\text{late}}$ threshold。结果显示performance对 $l_{\text{late}}$ 更敏感, 不如 $l_{\text{local}}$ 和 $l_{\text{sample}}$ robust。作者归因于late layers接近output, 小扰动会有大影响。

---

## 5. 整体Pipeline和计算复杂度

完整pipeline (Figure 2):

```
Input: N images
  ↓
[Patchify] → N × L tokens
  ↓
[Frame-wise attention layers] (unchanged)
  ↓
[Global attention layer 0 to l_local-1] → Local attention (替换)
  ↓
[Global attention layer l_local to l_sample-1] → Inter-frame selection + intra-frame downsampling
  ↓
[Global attention layer l_sample to end] → Inter-frame selection (K frames only)
  ↓
[Task-specific heads] → camera, depth, point maps
```

**计算复杂度**:
- Original: $\mathcal{O}(N^2 L^2)$ per global attention layer
- GoToHunt: $\mathcal{O}(N \cdot K \cdot (L/\sigma)^2)$ for sampling layers, $\mathcal{O}(N \cdot K \cdot L^2)$ for non-sampling layers
- 当 $K=25, N=500$, 加速比约 $\frac{N}{K} = 20\times$ on the global attention layers

---

## 6. 实验结果深度分析

### 6.1 Camera Pose Estimation (Table 4)

在7-Scenes, Neural RGB-D, TUM-Dynamics三个数据集上:

**GoToHunt vs Base model**:
- 7-Scenes: ATE 0.0673 (σ=2) vs base 0.0698 — **超过base model**
- Neural RGB-D: ATE 0.0267 vs base 0.0374 — **显著超过**
- TUM-Dynamics: ATE 0.0115 vs base 0.0118 — 略微超过

**GoToHunt vs其他方法**:
- FastVGGT [70]: training-free, 但性能略差
- SparseVGGT [80]: training-free, 但在高sparsity下OOM
- LiteVGGT [72]: 需要expensive retraining, 性能差于GoToHunt
- Co-Me [10]: 需要lightweight training, 性能最差
- Speed3R [67]: 需要retraining, 只适用于π³

### 6.2 3D Point Cloud Reconstruction (Table 5)

在7-Scenes和Neural RGB-D上, 用500帧dense reconstruction:
- GoToHunt在大多数metrics上超过base model
- 特别在Neural RGB-D上, Acc=0.0127 (σ=2) vs base 0.0160, 提升明显

### 6.3 Video Depth Estimation (Table 6)

在Bonn数据集 (332-895帧长序列):
- SparseVGGT在48GB GPU上OOM, 即使SR=75%
- GoToHunt scales reliably, 且超过base model: Abs Rel 0.0288 vs base 0.0333
- 比Speed3R (需retraining) 更好: 0.0288 vs 0.0314

### 6.4 Inference Time (Table 7)

500帧场景的inference time:
- VGGT base: 288.0s
- GoToHunt: 41.2s — **加速85.7%**
- LiteVGGT: 36.5s (但需expensive retraining)
- FastVGGT: 84.6s

GoToHunt的scaling接近linear, 因为global attention的cost不再依赖 $N$。

### 6.5 Inter-frame budget K的analysis (Table 8)

- $K=10$: ATE=0.0722, 32.3s
- $K=25$: ATE=0.0677, 41.2s
- $K=40-60$: ATE≈0.0674, 性能最好
- $K=100$: ATE=0.0685, 性能略降

**有趣发现**: 性能不是monotonically increasing with $K$。作者leave这个作为future investigation。我的intuition是: 当 $K$ 接近 $N$, 模型回到原始的redundant attention pattern, 可能引入noise。

### 6.6 Layer thresholds的robustness (Table 9)

变化 $l_{\text{local}} \in \{1,2,3,4\}$ 和 $l_{\text{sample}} \in \{8,9,10\}$:
- 所有配置的ATE都在0.0567-0.0578之间
- 说明method对hyperparameter选择robust

---

## 7. 与相关工作的context

### 7.1 Feed-forward 3D Reconstruction的发展

- **DUSt3R** [85] (CVPR 2024): 开创pairwise 3D point map prediction
- **MASt3R** [47] (ECCV 2024): grounding image matching in 3D
- **VGGT** [83] (CVPR 2025): 扩展到multi-view joint prediction
- **π³** [87] (ICLR 2026): permutation-equivariant design
- **MapAnything** [41] (3DV 2026): universal metric 3D reconstruction
- **Depth Anything 3** [52] (ICLR 2026): recovering visual space from any views

### 7.2 Efficiency improvement方法分类

**Training-free方法**:
- **FastVGGT** [70]: token merging, 保留reference和salient tokens
- **SparseVGGT** [80]: block-sparse global attention, CDF threshold
- **AVGGT** [76]: 重新思考global attention, uniform downsampling
- **GoToHunt (本文)**: hierarchical token selection

**需要retraining的方法**:
- **LiteVGGT** [72]: geometry-aware cached token merging, 需要full retraining
- **Speed3R** [67]: sparse feed-forward, 只适用于π³
- **Co-Me** [10]: confidence-guided token merging, lightweight training

**Test-time training方法** (concurrent works):
- **ZipMap** [37], **Scal3R** [95], **tttLRM** [79], **VGG-T³** [23], **LoGeR** [108]: 用TTT layers [77]替换quadratic attention

**Quantization方法**:
- **Quantized VGGT** [27], **Tail-aware quantization** [63]

### 7.3 Token selection in vision transformers的broader context

Token pruning/merging在ViT中已有大量研究:
- **Token merging (ToMe)** [Rao et al.]: 基于similarity merging
- **DynamicViT** [Rao et al.]: learning-based pruning
- **A-ViT** [Yin et al.]: adaptive token halting

但visual geometry transformers的特殊性在于: 有frame structure, 需要cross-view consistency, 而且global attention的pattern有layer-wise特性。

---

## 8. 技术细节的额外思考

### 8.1 为什么不直接在feature space做token-level FPS?

Appendix B.1探索了Token-Level Diversity (TLD)策略: 在每个selected frame内用FPS选tokens, 并考虑cross-frame redundancy。结果(Table A-C)显示:
- Video depth: TLD略好
- Pose estimation: TLD和Standard相当
- 3D reconstruction: Standard略好

且TLD需要额外5秒计算overhead per scene, 所以最终没用。我的理解是: intra-frame的token selection如果太aggressive, 会破坏spatial structure, 而uniform downsampling保留了spatial regularity。

### 8.2 Mean pooling替换early layers的失败

Appendix B.2尝试用mean pooling替换local attention。Table D显示Pool strategy明显差于Local:
- Local: ATE=0.0266
- Pool: ATE=0.0286

这表明即使early layers的attention近uniform, 也不能完全skip attention mechanism。可能的原因: attention有positional information, 而mean pooling完全忽略spatial structure。

### 8.3 Entropy-based adaptive layer partitioning

Appendix B.3探索了用entropy thresholds $\tau_1, \tau_2$ 自动决定layer partitioning。Table E, F显示性能comparable, 但需要online计算entropy, 每scene额外7秒。这说明固定thresholds已经足够好。

### 8.4 算法的几个微妙之处

1. **FPS的随机初始化**: Algorithm A用random seed $\sigma$ 选第一帧。作者没分析不同seed的影响, 但K-center的FPS approximation通常对initialization不敏感。

2. **Shared anchor set across queries**: 所有用queries用同一组selected frames $S$。这降低了selection的complexity, 也保证了attention的consistency。如果per-query selection会更flexible但复杂得多。

3. **Place recognition model的选择**: 用MegaLoc [4]提取features。这个model的选择可能影响效果, 在object-centric scenes或symmetric structures上可能unreliable (Section G的limitation)。

---

## 9. Future Directions和更深层implication

### 9.1 Routing-based attention mechanisms

Paper Section 5提到inter-frame selection指向routing-based mechanisms的可能性。我的理解是: 可以train一个轻量级router network, 预测每个query应该attend哪些frames, 替代FPS的offline selection。类似Mixture-of-Experts的routing思想。

### 9.2 Early layers的architectural implication

观察到early global attention layers基本是uniform attention, 说明这些layers对cross-view information aggregation贡献很小。这暗示在training时可以:
1. 直接skip这些layers的global attention
2. 用更轻量的cross-view interaction替代 (如mean pooling + positional encoding)
3. Progressive complexity: early layers local, later layers global

### 9.3 Token selection的learning-based方法

GoToHunt是training-free的, 但learning-based selection可能更好:
1. Train一个small prediction network预测token importance
2. 使用Gumbel-Softmax做differentiable token selection
3. 强化学习approach: agent学习selection策略

### 9.4 与test-time training的结合

Section F提到concurrent works用TTT layers。GoToHunt和这些方法complementary, 可以组合: 用TTT处理long sequences, 同时用GoToHunt在每个chunk内加速。

### 9.5 Beyond visual geometry transformers

这个hierarchical selection的思想可以推广到:
- **Video transformers**: temporal frame selection + spatial token pruning
- **Multi-modal transformers**: modality selection + token pruning
- **Long-context language models**: document selection + token selection

---

## 10. 实用建议

基于paper的实验, 给practitioners的建议:

1. **Default配置**: $K=25, \sigma=2$ or $3$, $l_{\text{local}}=2, l_{\text{sample}}=9$, 适用于大多数场景
2. **追求极致速度**: $K=10, \sigma=3$, 性能略降但大幅加速
3. **追求精度**: $K=40-60, \sigma=2$, 可能超过base model
4. **超长序列 (>500帧)**: 考虑结合chunk-based processing [17] 或streaming approaches [115]
5. **Memory constrained**: GoToHunt比SparseVGGT更memory-efficient, 在48GB GPU上能处理800+帧

---

## 11. Web References

**Main paper**: https://zsh2000.github.io/good-token-hunting.github.io/

**Base models**:
- VGGT: https://github.com/facebookresearch/vggt
- π³: https://github.com/yyfz/Pi3

**Datasets**:
- 7-Scenes: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
- Neural RGB-D: https://github.com/dazinovic/neural-rgbd-surface-reconstruction
- Bonn RGB-D: https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html
- TUM RGB-D: https://cvg.cit.tum.de/data/datasets/rgbd-dataset

**Related works**:
- DUSt3R: https://github.com/naver/dust3r
- FastVGGT: https://arxiv.org/abs/2512.xxxxx (ICLR 2026)
- LiteVGGT: https://arxiv.org/abs/2512.xxxxx (CVPR 2026)
- MegaLoc (place recognition): https://github.com/gabrieleberton/MegaLoc

**K-center and FPS**:
- Gonzalez T.F. (1985): "Clustering to minimize the maximum intercluster distance", Theoretical Computer Science

---

## 12. 总结

GoToHunt的beauty在于它的simplicity和effectiveness。通过hierarchical token selection + layer-adaptive策略, training-free地实现了85%+加速且performance不降反升。核心insights:

1. **Diversity > similarity**: inter-frame selection应该maximize scene coverage, 不是选最相似的frames
2. **Layer-adaptive > uniform**: attention pattern有layer-wise特性, 需要adaptive pruning策略
3. **Training-free > retraining**: 简单的selection策略能match甚至超过expensive retraining方法

这个work对visual geometry transformers的未来设计有重要implication: 现有架构可能over-parameterized, 用simpler的attention pattern (early layers local, later layers global with selected tokens)能达到comparable甚至更好的performance。

Andrej, 希望这个分析帮你build了intuition。这个paper的方法虽然简单, 但背后的analysis很solid, 特别是attention pattern的layer-wise分析很有启发性。
