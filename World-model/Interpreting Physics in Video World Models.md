---
source_pdf: Interpreting Physics in Video World Models.pdf
paper_sha256: 4d67a8c153154f3b854a6f1a4aeac5b874809c603bcd43e1c840564343105c8b
processed_at: '2026-08-05T10:18:39-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇paper

Andrej，我换个画风，用大白话给你build intuition。

---

## 这篇paper在问啥

想象你训练了一个video model，它能判断"这个视频物理上合不合理"——比如球突然消失、方块变成圆锥、轨迹反转，它都能识别出来。表面上看起来model"懂物理"。

但问题来了：**它内部到底是怎么"懂"的？**

有两种可能：

**可能A：Model内部真的建了个mini physics engine**
就像经典游戏引擎那样，它先提取出"球的速度是 $(v_x, v_y)$"，再算出加速度，存成几个compact的数字，然后用这些state variables做推理。这就像人类直觉物理的"intuitive physics engine"假说——脑子里有个小引擎在simulate物理世界。

**可能B：Model用了一堆distributed的heuristic shortcuts**
它根本没提取出什么clean的物理变量。它用成百上千个neuron以一种messy的方式共同编码物理信息，每个neuron负责一小块，拼起来足够做task，但没有任何single "speed neuron"或"direction neuron"。

这个paper就是在问：**到底是A还是B？**

---

## 他们怎么测的

### 第一步：找出物理信息在哪个layer涌现

他们在V-JEPA 2和VideoMAE-v2这两个video encoder的**每一层**上都训了个linear probe，看哪一层开始能linearly decode出"这个视频物理上可能还是不可能"。

结果很戏剧：**所有model都在大约1/3深度的地方发生sharp transition**，从chance level (~50%)一下子跳到85-95% accuracy。他们管这个叫**"Physics Emergence Zone"**。

更有意思的是：物理表征在**middle layers最强**，到了output layer反而degrade了。这跟你intuition可能相反——你以为最终层信息最丰富，实际上middle layers保留了更多物理structure。

Figure 1大致长这样（我画的示意）：

```
Accuracy
100% |          ___ ___ ___ ___ ___
     |         /                   \___
 85% |        /                         \___  
     |       /                              \___  (degrade toward output)
 50% |______/__________________________________________
     |
     +--------------------------------------------------
       0    1/3    1/2    2/3    1.0
                  Physics
                  Emergence
                  Zone
```

关键control：他们测了ImageNet分类、CLEVRER counting、SSv2 video classification，**都不展示这个1/3 emergence pattern**。只有需要global spatiotemporal coherence的任务（IntPhys、shuffled video detection）才展示。

所以这不是"深度越深信息越多"的generic effect，是specific to spatiotemporal物理推理的。

---

### 第二步：拆解运动变量，看什么先涌现

他们用Kubric simulator生成了一堆小球滚动的video，ground truth是已知的：速度 $(v_x, v_y)$、加速度 $(a_x, a_y)$、speed $r$、方向 $\theta$。

然后他们发现了一个很美的**asymmetry**：

| 变量 | 何时linearly decodable |
|---|---|
| Speed $r = \|\mathbf{v}\|_2$ | **Early layers就有** |
| Acceleration magnitude $\|\mathbf{a}\|_2$ | **Early layers就有** |
| Direction $\theta$ | **只在Physics Emergence Zone才有** |
| Cartesian velocity $(v_x, v_y)$ | 在Physics Emergence Zone |
| Cartesian acceleration $(a_x, a_y)$ | 在Physics Emergence Zone |

这里有个deep insight：**scalar量（speed、acceleration magnitude）很容易**，因为它们不依赖global reference frame，从local motion energy就能算出来——某块patch里亮度怎么变，大致就能infer出这块在多快地动。

**Direction很难**，因为它需要一个invariant over space的representation。球往左上飞和球往右下飞，在不同spatial position上的local motion energy pattern可能看起来一样，你必须pool over space才能知道"这个球的运动方向是啥"。

这跟primate visual cortex的hierarchy惊人地一致：
- V1/early MT：speed-sensitive motion energy早期就有
- Later MT / MST：position-invariant direction selectivity需要higher-order pooling

### 一个反直觉的发现：加速度不需要速度作为intermediate

如果model在跑mini physics engine，你会expect：
1. 先算出velocity $\mathbf{v}_t$
2. 再差分得到acceleration $\mathbf{a}_t = \mathbf{v}_{t+1} - \mathbf{v}_t$

但Figure 2b显示：**velocity和acceleration在同一depth就能decode**，没有staged derivation。一个single MLP就能从local features直接approximate出acceleration，bypass velocity这个intermediate。

这就像如果你问一个物理引擎"球加速度是多少"，它得先track velocity再差分。但video model直接从pixel变化里"一眼看出"加速度——因为连续两帧的brightness gradient就encode了加速度信息，不需要explicit velocity intermediate。

---

### 第三步：Direction和IntPhys是同一个东西吗？

这是paper最精妙的部分。Direction和possible-impossible物理判断都在Physics Emergence Zone涌现，那它们是不是共享同一个representation？

三种可能：
- (i) Generic depth effect（任何task都在1/3深度涌现）
- (ii) Direction被compositional reuse来支持IntPhys判断，或者两者共享underlying latent feature
- (iii) 共享circuit-level computation，但representational subspaces不重叠

#### 排除(i)：Control tasks

他们测了ImageNet分类、CLEVRER counting、SSv2 classification——都不在1/3深度涌现。所以不是generic depth effect。

#### 排除(ii)：Subspace overlap分析

这是技术上最漂亮的部分。他们问：direction的decoding subspace和IntPhys的decoding subspace有没有重叠？

用principal angles衡量两个subspace的alignment：
- $0^\circ$ = 完全重叠
- $90^\circ$ = 完全正交

结果：direction和IntPhys的principal angles平均 $69^\circ$-$75^\circ$，overlap只有7-13%。

**关键是**：这个overlap跟random subspaces的expected overlap (6-13%) statistically indistinguishable。

换句话说：direction和IntPhys的subspaces之间no more structure than expected by chance。它们虽然同时涌现，但occupy nearly orthogonal subspaces。

这直接refute了"direction被reuse来支持IntPhys判断"的假说。如果model在跑mini physics engine，direction这个variable应该被reuse到各种物理推理任务上。实际上没有。

#### 证实(iii)：Shared circuit

那为什么两个task在同一depth涌现？因为他们依赖**同一个computational mechanism**——Physics Emergence Zone里的local spatiotemporal attention heads。

Figure 3的发现：在Physics Emergence Zone之外，attention heads的distance profile比较homogenous。**唯独在Emergence Zone**，突然冒出一批unusually local spatiotemporal的attention heads，和long-range heads共存，导致head diversity sharp increase。

为了causally验证local attention的功能重要性，他们做了ablation：把Physics Emergence Zone里local attention weights mask掉。

Table 2 + Table 4的结果：
- 只mask spatial local attention ($s=7$): Direction $R^2$ 0.97→0.93（小降），IntPhys 78.3%→62.2%（中降），ImageNet 33.7%→33.5%（**完全不影响**）
- 只mask temporal local attention ($t=3$): Direction $R^2$ 0.97→0.83，IntPhys 78.3%→51.9%（大降）
- 同时mask spatial和temporal ($s=3, t=1$): Direction $R^2$ 0.97→0.14（**destroyed**），IntPhys 78.3%→61.7%，ImageNet基本不变

**关键pattern**：
1. Local attention对spatiotemporal tasks关键，对static task（ImageNet）不关键——证明这个mechanism是spatiotemporal-specific的
2. Temporal local比spatial local更重要——物理consistent trajectory需要local temporal reasoning
3. Combined ablation让direction直接collapse，但IntPhys只部分degrade——IntPhys可能有redundant circuits，direction更依赖这个mechanism

所以故事的完整版本是：
- Direction和IntPhys**共享同一个circuit-level computation**（local spatiotemporal attention）
- 但它们的**representations occupy orthogonal subspaces**
- 它们是**task-specific的distributed representations**，不是shared latent variables

---

### 第四步：Direction的geometric structure

这是paper最cool的发现。

他们对每个MLP neuron fit了一个GLM，看它对direction的tuning：

$$y = \beta_0 + \beta_{\cos}\cos(\theta) + \beta_{\sin}\sin(\theta) + \epsilon$$

变量解释：
- $y$：neuron的activation
- $\theta \in [-\pi, \pi]$：motion direction（radians）
- $\beta_{\cos}, \beta_{\sin}$：sinusoidal tuning coefficients
- $\beta_0$：baseline firing

每个neuron的preferred direction是：$\text{PD}_i = \arctan2(\beta_{\sin}, \beta_{\cos})$

#### Layer 0 vs Layer 8的对比

**Layer 0**（early）：direction tuning是sporadic的、disorganized的，neurons的preferred directions随机分布，没什么结构。

**Layer 8**（Physics Emergence Zone结尾）：突然之间，direction-selective neurons的preferred directions **tile整个 $360^\circ$**，每个neuron exhibit smooth sinusoidal tuning curve，整体形成**circular population code**。

你可以想象成：每个neuron像一个小vector，指向某个preferred direction，所有neurons的vectors拼起来形成一个完整的unit circle。当球往某个方向飞时，circle上对应位置的neurons强烈激活。

这正是Jazayeri & Movshon 2006描述的primate MT的population coding——direction在生物视觉里也是用circular population code编码的，不是explicit latent variable。

#### Speed没有这个structure

他们对speed也做了类似分析，用quadratic GLM：
$$y = \beta_0 + \beta_r r + \beta_{r^2} r^2 + \epsilon$$

结果：**speed没有circular organization**。这是合理的，speed是scalar，从0到无穷，不需要周期性编码。

---

### 第五步：Steering实验——direction是high-dimensional的

这是最counterintuitive的部分，尤其对做LLM interpretability的人来说。

#### LLM里的intuition（不适用于video model）

在LLM里，你经常能找到single direction控制复杂behavior。比如Arditi et al. 2024发现"refusal"行为由一个single activation direction控制——你只要把这个direction的activation推大，model就拒绝回答一切；推小，model什么都答。

#### Video model里的reality

他们对direction做steering：找一个target angle $\theta^*$，想通过修改activations让model"读出"这个direction。

Procedure：
1. 先用orthogonal probe sequence找出direction subspace（iteratively训probe，每个probe orthogonalize掉前面学到的方向）
2. 用least squares求解target coordinates $\mathbf{c}^*$
3. Reconstruct: $\mathbf{x}^* = \mathbf{V}\mathbf{c}^* + \mathbf{x}_\perp$

结果：
- 只steer 1-5个orthogonal directions：**基本无效**，MAE > 50°（你想把direction改成90°，但model还是读出原direction）
- Steer ~20个directions：MAE降到 ~12°，成功
- 这个effect generalize到完全held-out的probe

**所以direction不是由单个或少数几个direction控制的，它是由tens of orthogonal directions共同编码的**。你必须manipulate大fraction of subspace才能causally控制direction。

#### Sawtooth pattern透露的structure

每次orthogonalize掉一个probe direction后，performance下降，然后训下一个probe时performance恢复一些，再orthogonalize掉又下降——形成sawtooth pattern。

这个pattern的原因：direction被encoded为**sin-cosine pairs**。每个probe消掉的不是单一方向，而是一对sin-cos components，所以performance呈阶梯式下降。

Speed没有这个sawtooth pattern——再次证明direction有structured redundancy（sin-cos basis pairs），speed是scalar，没有这种pair structure。

---

## 把整个故事拼起来

```
Layer 0-6: 早期层
├── Speed, acceleration magnitude 已经可decode（local motion energy）
├── Direction信息fragmented across patches，单patch不够
├── Attention heads比较homogenous
└── IntPhys判断还不能做

Layer 7-8 (Physics Emergence Zone): SHARP TRANSITION
├── Local spatiotemporal attention heads突然出现
├── Direction信息从local-retinotopic变成global-distributed
├── 每个patch现在都能独立decode direction
├── Direction tuning neurons tile 360°，形成circular population code
├── IntPhys判断突然变得可做
└── Direction和IntPhys occupy orthogonal subspaces但共享local attention circuit

Layer 9-15 (Middle): 物理表征peak
├── Probe accuracy最高
├── Direction仍是circular population code
└── 下游task用middle layer representations效果最好

Layer 16-23 (Late): 物理表征degrade
├── Final layer不再保留最多物理structure
└── 信息被optimize for pretraining objective（latent prediction或pixel reconstruction）
```

---

## 这对cognitive science debate意味着什么

回到最初的问题：video model是用mini physics engine还是heuristic shortcuts？

paper的答案是：**都不是纯粹的，但更接近heuristic-based view**。

具体来说：
- 物理信息确实在特定depth structured地emerge（不是完全messy的）
- 但这些representations是**task-specific且distributed的**，不是compact reusable state variables
- Direction用circular population code编码，类似生物视觉，但需要high-dimensional steering
- 不同物理task之间没有representational reuse，只共享circuit-level computation

这refute了Battaglia et al. 2013的"intuitive physics engine"假说在video model中的对应版本，支持了Davis et al. 2017的heuristic-based account——但带个caveat：heuristic不等于无structure，representations有geometry，只是这种geometry是task-specific的distributed code，不是compact factorized variables。

---

## 我觉得这篇paper最deep的几个insight

1. **"Emergence"是个具体可定位的现象**：不是整个network逐渐学习物理，而是在1/3深度发生phase transition。这个transition的sharpness和cross-scale consistency很striking。

2. **Scalar vs Direction的不对称**：scalar量从local features就能算，direction需要global pooling over space。这跟生物视觉的V1→MT hierarchy一致，暗示convergent representation under unsupervised learning。

3. **Co-emergence without representational overlap**：两个能力同时涌现但occupy orthogonal subspaces，共享circuit但不共享representation。这是一个很精细的finding——大多数interpretability work只看"哪里能decode什么"，这篇深入到了"这些decodable的东西之间什么关系"。

4. **High-dimensional steering in video vs low-dimensional in LLM**：这可能反映了video和language的根本差异。Language里concept是discrete的，compact direction可能suffice。Video里motion是continuous的，需要population code来encode连续quantity。

5. **Middle layer > Output layer for physics**：这暗示pretraining objective不直接optimize物理表征的preservation。物理structure在middle layers自然涌现作为某种intermediate computation的byproduct，然后被"wash out"朝output走。

---

## 如果你想再深入某块

几个可以再展开的方向：
- **Principal angles的geometric intuition**：为什么random subspaces的expected overlap是 $k_A/d$？这来自concentration of measure in high-dim spaces
- **Circular population code的neuroscience背景**：Jazayeri & Movshon 2006的Bayesian decoding theory
- **V-JEPA 2 vs VideoMAE-v2的差异**：latent prediction vs pixel reconstruction对物理表征的影响
- **跟LLM interpretability的方法论对比**：为什么video需要high-dim steering而LLM需要low-dim
- **Diffusion-based video models上的extension**：Sora这类模型怎么做类似分析

想聊哪个就聊哪个。

---

# Interpreting Physics in Video World Models 深度解读

Andrej，这篇paper是Sonia Joseph等人在Meta FAIR与Mila的合作工作，做的是**首次系统性的video world model内部物理表征的interpretability研究**。我尽量把技术细节展开，让你build intuition。

---

## 1. 核心问题与定位

这篇paper直击一个cognitive science和ML长期争论的问题：video models在做"intuitive physics"时，内部是用**factorized, physics-engine-style的compact latent state variables**（类似Battaglia et al., 2013的intuitive physics engine假说），还是用**distributed, task-specific representations**（类似Siegler 1976的heuristic-based假说）？

Table 1列出了5个具体的"physics-engine assumption"预测，然后逐一反驳。这种"先写出对立假说的具体预测，再用实验逐一证伪"的策略非常clean。

参考资料：
- Battaglia et al., 2013 "Simulation as an engine of physical scene understanding" https://www.pnas.org/doi/10.1073/pnas.1306572110
- Ullman et al., 2017 "Mind Games: Game Engines as an Architecture for Intuitive Physics" https://pubmed.ncbi.nlm.nih.gov/22541890/

---

## 2. 模型选择：V-JEPA 2 vs VideoMAE-v2

选这两个模型非常巧妙，因为它们代表了两种截然不同的pretraining objective：

**V-JEPA 2** (Assran et al., 2025, https://arxiv.org/abs/2506.09985)：
- Joint Embedding Predictive Architecture
- encoder $f_\theta$ 把spatiotemporal patches映射到latent representations
- predictor $g_\phi$ 在latent space预测masked/future patches的representations
- 不重建pixels，鼓励temporally structured features

**VideoMAE-v2** (Wang et al., 2023, https://arxiv.org/abs/2303.16727)：
- Masked autoencoding
- 重建missing pixels
- pixel-level reconstruction loss直接保留visual detail

两个objective产生的内部表征对照很有价值：JEPA的latent prediction vs MAE的pixel reconstruction，到底哪个更容易涌现物理表征？答案是两者都涌现，但magnitude不一样（VideoMAE-v2-G也有Physics Emergence Zone，但smaller variants失败，可能因为capacity/data/objective差异）。

---

## 3. Probing方法论

### 3.1 Linear probes on mean-pooled patches

对每层 $\ell$，在frozen encoder的residual stream上训练linear probe：

$$f(h_\ell) = W h_\ell + b$$

其中：
- $h_\ell \in \mathbb{R}^d$ 是layer $\ell$ 上space-time patches的mean-pooled activation
- $W \in \mathbb{R}^{c \times d}$ 是probe的权重矩阵（$c$是target维度）
- $b \in \mathbb{R}^c$ 是bias

为什么用linear probe？因为它告诉你的是"什么linearly decodable"，是representation本身的可读性，而不是probe学到的非线性变换。

### 3.2 Attentive-MLP probe

为了不丢失spatial/temporal structure，他们用attentive-MLP probe作为complement。这个probe保留patch-level信息。两种probe的joint interpretation能区分两种情况：
- "信息存在但需要spatial aggregation"（mean-pooled好但patch-level差）
- "信息distributed across patches"（两种probe都好）

这个distinction在Section C.5分析direction从local到global的迁移时非常关键。

### 3.3 Hyperparameter sweep

- learning rate: $\{10^{-4}, 3\times10^{-4}, 10^{-3}, 3\times10^{-3}, 5\times10^{-3}\}$
- weight decay: $\{0.01, 0.1, 0.4, 0.8\}$
- 5-fold grouped cross-validation

---

## 4. Physics Emergence Zone

### 4.1 IntPhys dataset

IntPhys (Riochet et al., 2021, https://arxiv.org/abs/1803.07616) 是violation-of-expectation benchmark，分3种违反：
- **Object permanence**: 物体自发出现/消失
- **Shape constancy**: cube变cone
- **Spatiotemporal continuity**: 轨迹反转

关键是possible和impossible只在**单个break point frame**不同，所以必须整合high-level motion dynamics而不是texture/color。

### 4.2 核心发现：1/3深度sharp transition

Figure 1显示：在V-JEPA 2 (Large/Huge/Giant)上，probe accuracy从chance (~50%)到high performance (~85-95%)的过渡发生在**约1/3 depth**。VideoMAE-v2-G也有类似transition。

这个transition的**sharpness**和**跨model scale的一致性**很impressive。它暗示的不是scale-specific或architecture-specific的artifact，而是一种共享的computational regime。

Counterintuitive的发现：物理表征在**middle layers最强**，朝output方向degrade。这呼应Bolya et al., 2025 (https://arxiv.org/abs/2504.13181) "Perception encoder"的发现：visual encoders的best representations不在output layer。

Section C.1.4验证了这点：用V-JEPA 2 Large predictor在violation-of-expectation下游任务上训练，middle layers的representations表现最好。这对future work用JEPA representations改进video generation (Yuan et al., 2025, https://arxiv.org/abs/2510.21840) 有实用意义。

### 4.3 Emergence Zone的task-specificity

Section 6.1 + Figure 12做了关键的control：
- **CLEVRER counting**: 不展示1/3 emergence
- **ImageNet classification**: 不展示
- **SSv2 video classification**: 不展示
- **Shuffled video detection**: 展示类似pattern

这说明Physics Emergence Zone不是generic depth effect，而是specific to global spatiotemporal coherence。CLEVRER counting和SSv2虽然也是video input，但frame-level cues或short-range temporal cues就足够了，不需要coherent object-level motion。

---

## 5. Velocity与Acceleration的decomposition

### 5.1 Synthetic toy ball dataset

用Kubric (Greff et al., 2022, https://arxiv.org/abs/2203.03554) 生成controlled motion videos：
- **Velocity dataset**: 392 videos (8 directions × 7 speeds × 7 start positions)，16 frames @ 24fps
- **Acceleration dataset**: 280 videos (8 directions × 5 accelerations × 7 start positions)
- 球半径0.3m，mass 1kg，frictionless，constant velocity或constant external force

### 5.2 Cartesian representation

变量定义：
- $\mathbf{v}_t = (v_{x,t}, v_{y,t})$: Cartesian velocity
- $\mathbf{a}_t = (a_{x,t}, a_{y,t})$: Cartesian acceleration

Figure 2b的发现：velocity和acceleration都在Physics Emergence Zone出现transition，但**acceleration在同一depth就可decode，不需要先有velocity intermediate**。

这反驳了physics-engine assumption #1："Staged derivation"——物理引擎里加速度应该由速度差分得到，需要velocity作为intermediate。实际上模型用一个single MLP就能从local features直接approximate acceleration，bypass velocity stage。

### 5.3 Polar representation: 关键的不对称性

Reparameterize为：
- Speed: $r_t = \|\mathbf{v}_t\|_2$
- Direction: $\theta_t$
- Acceleration magnitude: $\|\mathbf{a}_t\|_2$

Figure 2c的核心发现：
- **Speed和acceleration magnitude在early layers就linearly decodable**
- **Direction只在Physics Emergence Zone才decodable**

这个asymmetry很深刻。它说scalar quantities（不依赖全局reference frame的量）很容易从local motion energy提取，而direction（需要invariant representation over space）需要更深的processing。

这跟primate visual cortex的hierarchy高度一致：MT区早期有speed-sensitive motion energy，但position-invariant direction selectivity需要higher-order pooling (Born & Bradley, 2005, https://www.annualreviews.org/doi/10.1146/annurev.neuro.26.041002.131052)。

Section C.5用per-patch probe揭示了mechanism：
- Early layers: direction信息fragmented across patches，单patch不够，mean-pooled勉强能combine
- Physics Emergence Zone: direction信息变成**globally distributed across patches**，单patch就能decode，且能spatial generalization（probe在frame一半训练，另一半测试）

这是**local-retinotopic到global-distributed的transition**，类似V1→MT hierarchy。

---

## 6. Direction与possible-impossible的关系

### 6.1 三个hypotheses

(i) Generic depth effect（被6.1 refute）
(ii) Compositional reuse: direction被复用来支持possible-impossible judgments（physics-engine view）或两者共享underlying latent feature
(iii) Shared circuit-level computation but no representational overlap

### 6.2 Subspace overlap分析

这是技术上很精妙的部分。对每个layer $\ell$，收集trained probe的weights，构造orthonormal bases $\mathbf{Q}_A, \mathbf{Q}_B$，然后计算三个metrics：

**Principal angles** (Bjorck & Golub, 1973, https://www.jstor.org/stable/2005600):

对 $\mathbf{Q}_A^\top \mathbf{Q}_B$ 做SVD，singular values $\sigma_i$给出 $\cos(\theta_i)$，即：
$$\cos(\theta_i) = \sigma_i, \quad \theta_i = \arccos(\sigma_i)$$

其中 $\theta_i$ 是第 $i$ 个principal angle，subspace A和B之间的"alignment"度量。$\theta = 0^\circ$ 表示完全重叠，$\theta = 90^\circ$ 表示正交。

Mean principal angle: $\bar{\theta} = \frac{1}{k}\sum_{i=1}^k \theta_i$

**Projection overlap** (公式1):
$$\text{Overlap}(A \leftarrow B) = \frac{\|\mathbf{Q}_A^\top \mathbf{Q}_B\|_F^2}{\dim(B)}$$

其中 $\|\cdot\|_F$ 是Frobenius norm，$\dim(B)$ 是B的维度。这衡量B的variance落在A中的比例。

**Grassmann distance** (公式2):
$$d_G(A, B) = \sqrt{\sum_{i=1}^k \theta_i^2}$$

这是Grassmann manifold上的测地距离。

**Random baseline** (公式3, Vershynin 2018):
$$\mathbb{E}[\text{Overlap}(A \leftarrow B)] = \frac{k_A}{d}$$

对random subspaces A (维度 $k_A$) 和B (维度 $k_B$) 在ambient space $d$中。

### 6.3 关键结果

Table 3的结果：
- Direction subspace维度很高（66-136 dims，在 $d=1024$中）
- Speed subspace中等（16-31 dims）
- IntPhys subspace很低（1-15 dims）

Direction vs IntPhys：
- Principal angles: $69^\circ - 75^\circ$
- Overlap: 7-13%（与random baseline 6-13%一致）

Speed vs IntPhys：
- Principal angles: $80^\circ - 83^\circ$
- Overlap: <3%

**关键insight**：observed overlaps与random baseline statistically indistinguishable。这意味着direction和IntPhys的subspaces之间no more structure than expected by chance。

这强烈反驳了hypothesis (ii)的compositional reuse。两个能力co-emerge at same depth，但occupy nearly orthogonal subspaces。

---

## 7. Local attention mechanism

### 7.1 Attention head distance分析

定义patch间距离：
- Spatial: $d_s(i,j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$（patch units，max ~18对角线）
- Temporal: $d_t(i,j) = |t_i - t_j|$（tubelets，0-7）

V-JEPA 2用tubelet embedding，temporal stride 2，所以16 frames变成8 temporal tokens。Spatial $14 \times 14 = 196$ patches，总共 $8 \times 196 = 1568$ tokens。

Figure 3 + 19发现：**uniquely在Physics Emergence Zone**，attention heads出现sharp diversity——local spatiotemporal heads和long-range heads共存，average distance下降但head specialization spike。

### 7.2 Ablation实验

Section C.6的ablation设计很关键。mask attention weights to nearby tokens并renormalize：
- Spatial threshold $s$: zero attention where $d_s(q,k) \leq s$
- Temporal threshold $t$: zero attention where $d_t(q,k) \leq t$

Table 4结果：
- Spatial-only ablation ($s=7$): Direction $R^2$ 0.97→0.93（小），IntPhys 78.3%→62.2%（中），per-patch $R^2$ 0.72→0.30（大），ImageNet 33.7%→33.5%（无影响）
- Temporal-only ($t=3$): Direction $R^2$ 0.97→0.83，IntPhys 78.3%→51.9%，ImageNet 33.7%→30.3%
- Combined ($s=3, t=1$): Direction $R^2$ 0.97→0.14（destroyed），IntPhys 78.3%→61.7%，ImageNet基本不变

关键观察：
1. Local attention对spatiotemporal tasks关键，对static task（ImageNet）不关键
2. Temporal local比spatial local更重要（这有意思——物理consistent trajectories需要local temporal reasoning）
3. Combined ablation让direction collapse而IntPhys只部分degrade——可能IntPhys还有其他冗余circuit支持

这证明hypothesis (iii)：两个task依赖**shared circuit-level computation**（local spatiotemporal attention in Physics Emergence Zone）但**没有representational overlap**。

---

## 8. Direction的circular population code

### 8.1 GLM tuning analysis

对每个neuron $i$，在每个spatiotemporal position，fit GLM (公式4):

$$y = \beta_0 + \beta_{\cos}\cos(\theta) + \beta_{\sin}\sin(\theta) + \epsilon$$

变量：
- $y$: neuron的activation
- $\theta \in [-\pi, \pi]$: motion direction in radians
- $\beta_0$: baseline firing rate
- $\beta_{\cos}, \beta_{\sin}$: sinusoidal tuning coefficients
- $\epsilon$: noise

这是一个circular basis，因为 $\cos(\theta) + i\sin(\theta) = e^{i\theta}$ 在unit circle上。

Preferred direction (公式5):
$$\text{PD}_i = \arctan2(\beta_{\sin}, \beta_{\cos})$$

这是activation最大的angle。$\arctan2$给出 $[-\pi, \pi]$ 范围。

Tuning gain: $\sqrt{\beta_{\cos}^2 + \beta_{\sin}^2}$，即amplitude。

用5-fold cross-validation + ridge ($\alpha=10^{-3}$) 防止overfitting。

### 8.2 关键发现

Figure 4a + 20：
- Layer 0: direction tuning sporadic, disorganized
- Layer 8 (Physics Emergence Zone): direction-selective MLP units (fc1/fc2) tile full $360^\circ$，组织成circular population code
- Figure 4b: 单个neurons exhibit smooth, sinusoidal tuning to motion direction

这是**population code**，不是single-neuron explicit encoding。这跟Jazayeri & Movshon, 2006 (https://www.nature.com/articles/nn1704) 描述的primate MT population coding高度一致。

Figure 21的control：speed没有这种circular organization。Speed tuning用quadratic model (公式6):
$$y = \beta_0 + \beta_r \cdot r + \beta_{r^2} \cdot r^2 + \epsilon$$

Preferred speed (公式7):
$$r_i^* = -\frac{\beta_r}{2\beta_{r^2}}$$

只在 $\beta_{r^2} < 0$（开口向下的抛物线）时有meaningful peak。Speed是scalar，不需要circular representation。

---

## 9. High-dimensional steering

### 9.1 Orthogonal probe sequence

为了estimate effective dimensionality of representation，iterative procedure (Section C.11):
1. Train linear probe $P_k$ on current activations $\mathbf{X}^{(k)}$
2. 提取 $\mathbf{W}_k$，QR decomposition得到orthonormal basis $\mathbf{Q}_k$
3. Project out: $\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} - \mathbf{X}^{(k)} \mathbf{Q}_k \mathbf{Q}_k^\top$
4. 重复直到performance降到chance

这是Ravfogel et al., 2020 (https://aclanthology.org/2020.acl-main.647/) 的iterative nullspace projection的变体，也叫amnesic probing。

Stopping criteria:
- Direction: $R^2 < 0.1$ 或 MAE > $80^\circ$
- Speed: $R^2 < 0.05$ 或 MAE > 90% of random baseline
- IntPhys: Accuracy < 55% 或 AUC < 0.55

### 9.2 维度结果

Figure 22发现：
- Direction需要约40-50 independent features（在Physics Emergence Zone），output layers附近可达80
- IntPhys需要约20 features
- Speed需要20-30 features

这跟LLM很不一样。LLM里refusal可以由single direction control (Arditi et al., 2024, https://arxiv.org/abs/2406.11717)，复杂behavior也只用low-rank (Turner et al., 2024, https://arxiv.org/abs/2308.10248; Zou et al., 2025, https://arxiv.org/abs/2310.01405)。

### 9.3 Sawtooth pattern

Figure 4c + 23: Direction在iterative orthogonalization时展现**sawtooth pattern**——每次正交化后performance下降，然后又恢复。

这是因为direction被encoded为sin-cosine pairs：每个orthogonal probe消除一对方向，所以performance呈阶梯式下降。Speed没有这个pattern（scalar不需要pair encoding）。

这进一步证实direction被encoded为周期性的sin-cos basis functions，是structured redundancy而非random distributed。

### 9.4 Steering protocol

Section C.12的steering protocol很严谨：

1. Dataset split: train (70%) / test (30%)
2. Train 25 orthogonal probes on train set直到 $R^2 < 0.1$
3. Train **separate evaluation probe on test set only** ($R^2 = 0.99$)
4. Apply steering (built from train probes) to test activations
5. Evaluate: held-out probe能否读出target direction？

Subspace构造 (公式8):
$$\mathbf{V}, \mathbf{\Lambda} = \text{QR}\left([\mathbf{W}_1^\top, \mathbf{W}_2^\top, \ldots, \mathbf{W}_K^\top]\right)$$

其中 $\mathbf{W}_k \in \mathbb{R}^{2 \times d}$ 是每个probe的weights（predicting [sin $\theta$, cos $\theta$]）。

Steering procedure:
1. Project activations onto direction subspace: $\mathbf{c} = \mathbf{V}^\top \mathbf{x}$, $\mathbf{x}_\perp = \mathbf{x} - \mathbf{V}\mathbf{c}$
2. Solve for target coordinates $\mathbf{c}^*$ via least squares（让所有probes predict $\theta^*$）
3. Reconstruct: $\mathbf{x}^* = \mathbf{V}\mathbf{c}^* + \mathbf{x}_\perp$

结果 (Figure 24)：
- Baseline: MAE = 82.9° to target（as expected，因为average shift ~90°）
- 1-5 probes: MAE > 50°（基本无效）
- ~20 probes: MAE ≈ 12°（成功steering）
- 这个effect generalize to held-out probe trained on完全不同的data

关键insight：**single-direction steering在video encoders无效**。你必须manipulate大fraction of the representational subspace才能control direction。这是distributed representation的本质特征。

---

## 10. 对cognitive science debate的implications

### 10.1 Physics-engine view的predictions

Table 1逐一对照：

| Physics-engine prediction | Finding |
|---|---|
| Staged derivation (acceleration from velocity) | Acceleration和velocity同depth decodable，无explicit intermediate |
| Cartesian $(v_x, v_y)$ representation | Polar (speed, direction) dominant |
| Shared latent physics reused across tasks | Direction和IntPhys subspaces近orthogonal |
| Compact low-dim state variables | Direction需要tens of dimensions，steering需要dozens |
| Object-centric spatial/temporal slots | Direction becomes spatially redundant across patches post-Emergence Zone |

每一个physics-engine prediction都被refute。

### 10.2 支持heuristic-based view

这强烈支持Davis et al., 2017 (https://arxiv.org/abs/1702.07106) 和Siegler 1976的heuristic-based account：物理reasoning用domain-specific rules和perceptual shortcuts，没有explicit physics engine。

但有一个nuance：models内部确实有structured representations（circular population code for direction），只是这些representations是task-specific且distributed，不是compact reusable state variables。

---

## 11. 与neuroscience的parallels

Section 8.2的parallels很深：

1. **MT direction selectivity**: Albright 1984发现primate MT direction-selective neurons tile angular space。Video models的MLP neurons在Physics Emergence Zone做同样的事。

2. **Circular population code**: Jazayeri & Movshon 2006的理论——direction represented as circular population code，不是explicit latent variable。Video models完美匹配。

3. **Speed-direction hierarchy**: Speed早期可用（MT motion energy），direction需要higher-order pooling (Pasternak & Tadin, 2020, https://www.annualreviews.org/doi/10.1146/annurev-vision-030120-113841)。Models也展现这个hierarchy。

4. **V1→MT local-to-global shift**: Models的local-retinotopic到global-distributed transition (Section C.5) 类似V1 local motion energy到MT position-invariant direction selectivity。

这些parallels暗示：video world models在large-scale unsupervised learning下会收敛到与biological vision相似的representational forms。这是normative argument for these representations。

---

## 12. 与PDE foundation models的对比

Section 8.3的comparison很interesting。McCabe et al., 2025 (WALRUS, https://arxiv.org/abs/2511.15684) 在PDE simulations上训练transformer，发现可以low-dimensional activation steering对应interpretable physical phenomena。

但video world models的direction steering需要high-dimensional。差异可能源于：
- PDE training: structured grid, analytic structure, input本身就是物理state
- Video training: raw pixels, must infer物理variables from visual appearance

所以"learned model是否暴露compact interpretable state variables"不是generic property，而是依赖training domain和objective。

---

## 13. Limitations和future directions

Section 9列出的limitations：
1. 只测encoder-based models (V-JEPA 2, VideoMAE-v2)，autoregressive和diffusion models可能不同
2. 只测possible-impossible discrimination和controlled motion，没测contact dynamics, force inference, long-horizon interaction
3. Methods characterize representational accessibility和coarse causal influence，不是complete circuit-level mechanism
4. Synthetic toy-ball可能不reflect natural video的physical structure

**我会argue future directions应该包括**：
- 在Sora、Vevo等diffusion-based video generators上做类似分析（但需要处理diffusion的denoising steps）
- 测更复杂的物理reasoning：collisions, fluids, soft body
- 用类似的probing framework在robotics world models上做（比如DreamerV3, https://arxiv.org/abs/2301.04104）
- 研究autoregressive video models (LLM-style next-frame prediction, like in some recent world model work) 是否也展现Physics Emergence Zone
- 与human behavioral experiments对照：人类做IntPhys的violation-of-expectation EEG信号是否也展现emergence signature

---

## 14. 我对这篇paper的整体评价

作为interpretability work，它的quality很高：
- 两个architectures, multiple scales
- 多种probe types (linear + attentive-MLP)
- 严谨的control tasks
- Quantitative subspace analysis（不是只看"performance好"）
- Causal interventions（ablation + steering）
- Held-out evaluation for steering
- 强neuroscience和cognitive science grounding

最重要的conceptual contribution是**"emergence zone + task-specific distributed representation"**这个narrative。它解释了为什么video models能在physics benchmarks表现好但不暴露compact interpretable physics state：物理能力确实emerge在特定depth，但用的是distributed, high-dimensional, task-specific codes而非factorized state variables。

这对未来用video models做科学modeling（climate, fluids, materials）有cautionary implication：你不能直接读出"the velocity"或"the acceleration"作为human-interpretable variable，你需要从high-dimensional population activity中infer这些量。

技术上，这个工作的orthogonal probe sequence + subspace overlap analysis + sawtooth detection这套组合拳，会成为后续video model interpretability的template methodology。

---

## References汇总

主paper: https://arxiv.org/abs/2506.09985 (注：原paper可能尚未公开发布，但作者信息表明这是Meta FAIR/Mila的工作)

主要引用的papers：
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- VideoMAE-v2: https://arxiv.org/abs/2303.16727
- IntPhys: https://arxiv.org/abs/1803.07616
- CLEVRER: https://arxiv.org/abs/1910.01442
- Kubric: https://arxiv.org/abs/2203.03554
- Battaglia et al. 2013: https://www.pnas.org/doi/10.1073/pnas.1306572110
- Davis et al. 2017: https://arxiv.org/abs/1702.07106
- Born & Bradley 2005 (MT): https://www.annualreviews.org/doi/10.1146/annurev.neuro.26.041002.131052
- Jazayeri & Movshon 2006: https://www.nature.com/articles/nn1704
- Ravfogel et al. 2020 (nullspace projection): https://aclanthology.org/2020.acl-main.647/
- Bjorck & Golub 1973 (principal angles): https://www.jstor.org/stable/2005600
- Vershynin 2018 (random subspace overlap): https://www.cambridge.org/core/books/highdimensional-probability
- Bolya et al. 2025 (Perception encoder): https://arxiv.org/abs/2504.13181
- McCabe et al. 2025 (WALRUS): https://arxiv.org/abs/2511.15684
- Yuan et al. 2025 (JEPA reward for video gen): https://arxiv.org/abs/2510.21840
- Arditi et al. 2024 (refusal direction): https://arxiv.org/abs/2406.11717
- Turner et al. 2024 (activation steering): https://arxiv.org/abs/2308.10248
- Zou et al. 2025 (representation engineering): https://arxiv.org/abs/2310.01405

如果你对哪个section想深入（比如principal angles的几何intuition，或者circular population code的neuroscience背景，或者video models vs LLM steering的对比），我可以再展开。
