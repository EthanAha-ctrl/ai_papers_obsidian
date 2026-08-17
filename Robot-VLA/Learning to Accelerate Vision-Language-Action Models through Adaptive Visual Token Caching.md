---
source_pdf: Learning to Accelerate Vision-Language-Action Models through Adaptive
  Visual Token Caching.pdf
paper_sha256: 0a9a0173cf42d45787d4a5b6f7bda9b4278ff9dd6e4865567d80c3b4606819fb
processed_at: '2026-08-05T13:47:09-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 LAC这篇paper

## 一句话总结

VLA model每一步都要重新看一遍整张图，但图里大部分东西根本没动，太浪费了。这篇paper的核心想法：**让模型自己学该看哪儿、该偷懒哪儿**，直接拿task loss来教，比起靠attention score瞎猜，效果好太多了。

## 为什么这件事重要

先说说VLA model的痛点。OpenVLA这种model，你给它一句话"把banana盖住"，它要output action token告诉robot怎么动。每一步控制都要跑一遍完整的Vision-Language-Action pipeline，从image encoding到LLM reasoning到action decoding。

问题是：robot控制是real-time的，你要至少10-20Hz的control frequency，OpenVLA一次inference要50多ms，根本扛不住高频控制。

这里有个特别obvious的observation：**视频里大部分pixel根本没动**。robot的gripper动了几厘米，但background的桌子、墙壁、远处的物体完全static。你每一步都re-encode全部visual tokens，这90%的computation是浪费的。

KV cache在LLM里已经是standard practice了——token之间temporal redundancy很大，不用每步都重算K和V。VLA里更夸张，是spatial-temporal redundancy，不仅时间维度有冗余，空间维度也有。

## 现有方法为什么不行

之前的acceleration方法基本都是heuristic的，靠各种proxy metric来决定哪些token重要：

- **VLA-Cache**: 用attention score当proxy，attention低的token就cache
- **SparseVLM**: 用visual saliency pruning tokens
- **FastV**: 在特定layer做token compression

这些方法的核心问题是：**proxy metric和task success之间存在gap**。

Paper里Figure 1举了个特别好的例子：robot要把东西放进basket里，basket现在是static的，attention score很低，rule-based方法就把basket cache了。但basket马上要被interaction啊！结果robot arm卡在basket rim上fail了。

这就是"where the model attends ≠ what the task requires"。Attention score是个correlation signal，not a causal signal。Model可能attend到某些visually salient但task-irrelevant的东西，也可能忽略task-critical但visually subtle的cue。

还有一个counter-intuitive的发现：SparseVLM在LIBERO上FLOPs降低了，但CUDA time反而增加了（从51.91ms到83.39ms）。因为pruning logic本身的overhead超过了pruning带来的saving，在robotics这种short action sequences上根本amortize不过来。这告诉我们一个lesson：**theoretical FLOPs和wall-clock speedup是两回事**，一定要测实际latency。

## LAC的核心idea

LAC（Learnable Adaptive Caching）的核心insight其实很简洁：**与其用proxy metric猜哪些token重要，不如直接用task loss来教模型做这个decision**。

把inference acceleration重新formulate成一个policy learning problem：
- 决策内容：哪些visual token需要recompute，哪些可以从cache reuse
- 优化目标：task success rate + efficiency
- 训练方式：end-to-end，task loss直接backprop到decision module

这就像把speculative decoding从heuristic变成learned policy一样，把computation allocation本身当成一个可学习的policy来optimize。

## 架构怎么设计的

### Motion-aware input

$$V_t = [I_t; O_t]$$

很简单，就是把current frame $I_t$ 和optical flow $O_t$ 在channel维度concat起来。$O_t$ 用RAFT-small算，是个lightweight optical flow model。

为什么用optical flow？这是个关键design choice。你想想，robotics场景里"哪些pixel在动"这个信号几乎完美对应"哪些token需要recompute"。Gripper在动、被抓的object在动，这些pixel的flow magnitude就大；static background的flow几乎为零。

Paper里ablation了用language guidance代替optical flow的variant，结果从85.6%掉到83.0%。Language guidance有两个问题：
1. 要做好vision-language alignment需要大encoder，overhead太大
2. Lightweight language encoder又align不好语义

Optical flow则是个pixel-level、直接、便宜的motion signal，perfect fit这个场景。

### 两个cooperative module

**Cached Token Selector**：决定which tokens to cache

$$S_t = f_{\text{sel}}(V_t; \theta_{\text{sel}})$$

这里 $S_t = \{s_t^{(1)}, \ldots, s_t^{(N)}\}$，每个 $s_t^{(i)} \in [0,1]$ 是token $i$ 的saliency score。$f_{\text{sel}}$ 是个小CNN，参数是 $\theta_{\text{sel}}$。

$s_t^{(i)}$ 高 → 这个token在动或task-critical → recompute
$s_t^{(i)}$ 低 → static background → cache

用CNN是因为它的inductive bias对局部motion detection特别合适，而且computation cost远低于backbone transformer。

**Cache Ratio Predictor**：决定how much to cache

$$L_t = f_{\text{pred}}(V_t; \theta_{\text{pred}})$$

Output是logits $L_t = \{l_t^{(1)}, \ldots, l_t^{(C)}\}$，对应一个discrete set of candidate ratios $\mathcal{R} = \{r_1, \ldots, r_C\}$。比如 $\mathcal{R} = \{0.2, 0.4, 0.6, 0.8\}$，那 $C=4$。

Scene很static的时候predict high ratio（多cache省算力），scene很dynamic的时候predict low ratio（多recompute保accuracy）。这是scene-level的budget allocation。

**两模块怎么cooperate**：
1. Ratio Predictor看全局scene dynamics，决定budget $k_t = N \cdot r_t$
2. Token Selector给每个token打分，按score排序
3. 选score最低的 $k_t$ 个token去cache，其余recompute

这是一个hierarchical decision process：先决定"多少"，再决定"哪些"。两个decision不同granularity，互相complement。

## 训练为什么难

这是这篇paper技术上最interesting的部分。

Decision是discrete的：argmax选ratio、top-k选token。但discrete operation不可微，gradient从task loss传不回decision module。

**Solution**: Gumbel-Softmax + Straight-Through Estimator (STE)

### Cache Ratio Predictor的可微化

$$\tilde{p}_t^{(j)} = \frac{\exp((l_t^{(j)} + g_j)/\tau)}{\sum_{k=1}^{C} \exp((l_t^{(k)} + g_k)/\tau)}$$

变量解释：
- $l_t^{(j)}$：ratio $r_j$ 的logit
- $g_j \sim \text{Gumbel}(0,1)$：Gumbel noise，采样自标准Gumbel distribution
- $\tau$：temperature，控制softmax的"软度"
- $\tilde{p}_t^{(j)}$：ratio $r_j$ 的soft probability

Forward pass用hard argmax：
$$p_t = \text{one\_hot}(\arg\max_j(\tilde{p}_t^{(j)}))$$

Backward pass用STE，直接把gradient传给 $\tilde{p}_t$，绕过argmax。

这就是"hard forward, soft backward"的trick：inference时用discrete decision保证效率，training时用soft approximation保证gradient flow。

### Token Selector的可微化

$$\tilde{M}_t^{(i)} = \sigma\left(\frac{s_t^{(i)} - \theta_k}{\tau_s}\right)$$

变量解释：
- $s_t^{(i)}$：token $i$ 的saliency score
- $\theta_k$：第k个token的score threshold（top-k的边界）
- $\tau_s$：temperature，很小时sigmoid接近step function
- $\sigma$：sigmoid

Forward用hard top-k mask $M_t$，backward用steep sigmoid $\tilde{M}_t$ 做differentiable approximation。$\tau_s \to 0$ 时steep sigmoid就变成step function，gradient集中在threshold附近。

### 两阶段training

**Stage I: Attention Alignment**

$$\mathcal{L}_{\text{align}} = \text{MSE}(f_{\text{sel}}(V_t; \theta_{\text{sel}}), S_{\text{VLA}})$$

让Token Selector先去mimic frozen VLA的attention map $S_{\text{VLA}}$。VLA attention不是optimal policy，但是个reasonable "visual saliency prior"。

这一步很关键。Table 5的ablation显示，without Stage I直接从scratch学，success rate从85.6%掉到79.2%，掉了6.4pp。从scratch学discrete selection policy在sparse supervision下不稳定，容易collapse到suboptimal local minimum。

这和RL里的"behavior cloning before policy gradient"思想一样：先让policy有个reasonable starting point，再optimize。

**Stage II: Joint Optimization**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{VLA}} + \lambda \mathcal{L}_{\text{ratio}}$$

$$\mathcal{L}_{\text{ratio}} = -\mathbb{E}_{\tilde{p}_t}[r] = -\sum_{j=1}^{C} \tilde{p}_t^{(j)} r_j$$

这里：
- $\mathcal{L}_{\text{VLA}}$：VLA的task loss（action prediction的loss）
- $\lambda$：trade-off权重
- $\tilde{p}_t^{(j)}$：Gumbel-Softmax的soft probability
- $r_j$：candidate cache ratio
- 负号：鼓励predictor选higher ratio（maximize efficiency）

$\mathcal{L}_{\text{VLA}}$ 保证task performance不degrade，$\mathcal{L}_{\text{ratio}}$ 提供efficiency pressure，$\lambda$ 控制两者balance。

## Inference怎么做

Inference时把stochastic sampling换成deterministic argmax，保证稳定。

1. Ratio Predictor argmax → $r_t$
2. Token Selector按score排序，选最低的 $N \cdot r_t$ 个token去cache
3. 生成binary mask $M_t \in \{0,1\}^N$
4. Active tokens重新encode计算新K,V；cached tokens直接用上一步的K,V
5. 混合的K,V送进action decoder

还有一个**Stochastic Recovery Mechanism**：每步以小概率 $p_{\text{recover}}$ 随机refresh一部分cached tokens。这是为了防止error accumulation——一直cache的话stale信息会累积，periodic refresh像checkpoint一样保证long-horizon robustness。Figure 3里绿色的tiles就是recovered tokens。

这个mechanism在ablation里贡献了+2.2pp success rate。

## 实验数据解读

### LIBERO Benchmark

```
Method                  Avg Success    FLOPs(T)   CUDA(ms)
OpenVLA (baseline)      75.0%          1.864      51.91
SparseVLM              64.7%          1.407      83.39  ← FLOPs降了但latency反增！
FastV                  73.3%          1.864      53.28
VLA-Cache              74.7%          1.355      31.83
LAC (Ours)             76.9%          1.392      29.51
```

几个值得关注的点：

1. **LAC同时提升performance和efficiency**：76.9% > 75.0%，29.51ms < 51.91ms，1.76× speedup + 1.9pp success rate提升。这种"free lunch"在acceleration方法里很罕见。

2. **SparseVLM的反例**：FLOPs降低24.6%，但CUDA time反增60%。Pruning logic的overhead超过benefit，在short action sequences上amortize不过来。This is why wall-clock measurement matters more than theoretical FLOPs。

3. **LIBERO-Long提升最大**：从53.2%到59.2%，+6pp。Long-horizon task里error accumulation更严重，learned policy + stochastic recovery的优势更明显。

### SIMPLER Benchmark

在CogAct（diffusion action decoder）上：
- Visual Matching: 74.8% → 75.5%（+0.7pp），1.42× speedup
- Variant Aggregation: 61.3% → 63.0%（+1.7pp）

这说明LAC是architecture-agnostic的，不管是autoregressive还是diffusion action decoder都work。

### Real-World Robot

```
Method        KnockCrisp  PickMango  CoverBanana  KnockBottle  Avg
OpenVLA       48.0%       16.0%      24.0%        44.0%       33.0%
VLA-Cache     48.0%       12.0%      20.0%        40.0%       30.0%
LAC           52.0%       12.0%      24.0%        64.0%       38.0%
```

Real-world提升+5pp比simulation的+1.9pp更大。KnockBottle从44%到64%尤其impressive。Real-world有更多visual noise，learned policy能更好distinguish task-relevant dynamics from noise，而rule-based方法容易被noise误导。

## Ablation给我们的insight

### Component-wise

```
Selector only                    82.20%   1.283T   28.48ms
+ Reuse Predictor                83.40%   1.325T   29.04ms
+ Recovery (Full)                85.60%   1.377T   29.32ms
```

有意思的是，加Reuse Predictor后FLOPs和时间略增，因为predictor会adaptive在critical moments降低cache ratio。这说明**optimal efficiency-accuracy tradeoff不是maximal caching**，而是adaptive caching with safety mechanisms。

### Two-Stage Training

```
w/o Stage I Initialization       79.2%
Language-guided Policy          83.0%
Ours (LAC full)                 85.6%
```

Without Stage I掉6.4pp，证实cold-start problem严重。Language guidance掉2.6pp，证实optical flow比language更便宜更有效。

## Computational Complexity分析

Paper Appendix C给了theoretical analysis。

**Baseline per layer**:
$$\mathcal{C}_{\text{base}} \approx 4ND^2 + 2N^2D + 2NDM$$

- $4ND^2$：QKV projection ($3ND^2$) + output projection ($ND^2$)
- $2N^2D$：attention score computation + aggregation
- $2NDM$：FFN的两个linear layer（$D \to M$和$M \to D$）
- $N$ tokens, $D$ embedding dim, $M$ FFN intermediate dim, $L$ layers

**LAC per layer**:
$$\mathcal{C}_{\text{lac}} \approx 4N_{\text{act}}D^2 + 2(N_{\text{act}} \cdot N)D + 2N_{\text{act}}DM$$

where $N_{\text{act}} = (1-\rho)N$

注意attention部分：active tokens ($N_{\text{act}}$) 的queries要attend to full context (active + cached, total $N$)，所以是 $N_{\text{act}} \cdot N$ 而不是 $N_{\text{act}}^2$。但FFN和projection的cost线性减少。

**Policy overhead**:
$$\mathcal{C}_{\text{policy}} \approx \mathcal{O}(H \cdot W \cdot C_{\text{cnn}})$$

只跟image resolution有关，跟model depth $L$ 和sequence length无关。在$L$大的时候overhead被amortize掉。

LIBERO上实测FLOPs从1.864T降到1.392T，减少25.3%，跟理论分析吻合。

## 这个工作更深的implication

LAC揭示了一个deeper insight：**inference computation的allocation本身就是一个policy**，应该被end-to-end optimized for task performance，而不是由proxy metric决定。

这个insight其实可以extend到很多其他场景：

1. **LLM inference中的speculative decoding**：现在靠heuristic draft model，可以learned
2. **Video understanding中的frame sampling**：哪些frame需要process，哪些skip
3. **Neural rendering中的ray sampling**：哪些pixel需要更多采样
4. **MoE中的expert routing**：现在靠router network learned，但和这里思想类似
5. **Multi-modal fusion中的modality selection**：什么时候trust vision，什么时候trust language

本质上这些都是**computation allocation as policy**的问题，都可以用类似的learned approach来optimize。

## 和我熟悉的工作的connection

LAC让我联想到几个相关工作：

**Conditional Computation**: MoE (Mixture of Experts) 是conditional computation的一种形式，每个token activate部分expert。LAC是spatial-temporal conditional computation，每个frame activate部分token的computation。两者本质上都是"不均匀分配computation资源"。

**Learned Scheduling**: Database里的query optimizer从rule-based升级到learned optimizer是个类似trajectory。LAC把VLA inference scheduling从rule-based升级到learned。

**Policy Distillation**: Stage I的attention alignment本质是policy distillation from frozen expert。RL里这个pattern很常见：先用sub-optimal但stable的expert warm-up，再做policy gradient optimization。

**Active Perception**: Robotics里的active vision传统思路是控制camera去看哪里。LAC是"active perception的computation版本"——不是控制camera，而是控制computation资源allocation。

**Speculative Execution**: CPU里的speculative execution用branch prediction猜执行路径。LAC的caching也是某种speculation——bet on "这个token下一帧不会变"。Learned policy让这个speculation更accurate。

## Limitations和未来方向

Paper自己提了两个limitation：
1. Optical flow在extreme visual conditions下可能失效
2. 高度dynamic scene下efficiency gain减少

我看到几个potential extensions：

1. **Cross-frame semantic consistency**: 现在只考虑motion，可以引入object tracking做semantic-level caching
2. **Learnable recovery probability**: 现在fixed $p_{\text{recover}}$，可以做成state-dependent
3. **Multi-scale caching**: 不同Transformer layer用不同cache ratio
4. **Joint vision-action caching**: 结合FAST [34]的action tokenization，jointly optimize vision caching和action chunking
5. **3D VLA的extension**: 在point cloud based VLA上extend，π0.5 [15]那种flow matching framework
6. **Full policy distillation in Stage I**: 现在只distill attention，可以distill full behavior

## 实用takeaways

如果你要deploy一个VLA model，LAC给几个immediate actionable insights：

1. **Visual token caching在robotics上确实work**：1.76× speedup是real的
2. **Rule-based caching不够**：需要task-aware learned policy
3. **Optical flow是efficient motion prior**：比language guidance便宜且有效
4. **Two-stage training很重要**：warm-up + end-to-end optimization比pure RL-style learning stable
5. **Stochastic recovery是cheap insurance**：+2.2pp with minimal cost
6. **Wall-clock > FLOPs**：一定测实际CUDA latency，别只看theoretical FLOPs

## Web References

- LAC paper (arxiv): https://arxiv.org/abs/2502.02175
- VLA-Cache (most related baseline): https://arxiv.org/abs/2502.02175
- OpenVLA: https://openvla.github.io/
- CogAct: https://arxiv.org/abs/2411.19650
- LIBERO benchmark: https://libero-project.github.io/
- SIMPLER: https://simpler-env.github.io/
- Gumbel-Softmax: https://arxiv.org/abs/1611.01144
- Straight-Through Estimator: https://arxiv.org/abs/1308.3432
- RAFT optical flow: https://arxiv.org/abs/2003.12039
- SparseVLM: https://arxiv.org/abs/2410.04417
- FastV: https://arxiv.org/abs/2403.06764
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- RT-2: https://robotics-transformer2.github.io/
- PaLM-E: https://palm-e.github.io/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- LoRA: https://arxiv.org/abs/2106.09685
- FAST (action tokenization): https://arxiv.org/abs/2501.09747
- HIRT (hierarchical robot transformers): https://arxiv.org/abs/2410.05228
- Deer-VLA (dynamic inference): https://arxiv.org/abs/2410.07205
- Mole-VLA (layer skipping): https://arxiv.org/abs/2503.20384

## Final Intuition

把LAC整个故事压缩成一句话：

**让task loss直接教模型该看哪里、该偷懒哪里，而不是用attention score当proxy猜**。

这个paradigm shift从heuristic到learned、从proxy到direct、从static到adaptive，在robotics这种real-time、safety-critical的应用中尤为重要。Deep down，LAC告诉我们：**computation allocation itself is a policy, optimize it end-to-end**。

这个insight远远超出VLA acceleration本身，触及ML里一个universal pattern——什么时候都不要满足于proxy metric，要optimize你真正care的objective。

---

# LAC: Learnable Adaptive Caching for Vision-Language-Action Models - 深度技术解析

Andrej，这篇paper从robotics deployment的实际痛点出发，做了一个相当elegant的工作：把VLA inference acceleration从rule-based heuristic升级成end-to-end learnable policy。让我从intuition、architecture、math到experimental evidence逐层剖析。

## 1. 核心Intuition: 为什么existing methods会fail

### 1.1 The Fundamental Misalignment

现有的VLA acceleration方法（VLA-Cache [48], SparseVLM [52], FastV [7]）都依赖proxy metrics：
- Attention scores
- Visual saliency
- Static spatial heuristics

这些proxy metrics和actual task success之间存在semantic gap。paper里Figure 1给出了一个特别好intuition example：rule-based method会cache那个static target basket，because它的attention score低，but basket正是即将被interaction的目标，导致robot arm卡在rim上fail。

**Key insight**: "where the model attends ≠ what the task requires"。这是一个经典的criterion-target mismatch problem，类似于early days of CNN中用gradient saliency来解释decision，但saliency和causality之间有gap。

### 1.2 Temporal Redundancy在Robotics中的特殊性

VLA model在timestep t和t+1之间：
- Gripper pose可能移动了几cm
- Manipulated object姿态变化
- **但90%+的background pixels完全static**

然而standard VLA model每一步都re-encode全部N个visual tokens，这是massive redundancy。这点和LLM inference中的KV cache motivation类似，但VLA的redundancy是**spatial-temporal**而不仅仅是**temporal**（LLM中是token-level temporal）。

## 2. 架构解析: 双模块协同设计

### 2.1 Motion-Aware Input Representation

$$V_t = [I_t; O_t]$$

其中：
- $I_t$: current frame的visual features
- $O_t$: optical flow（由RAFT-small计算）
- $[;]$: channel-wise concatenation

这里有个重要design choice：**为什么用optical flow而不是language guidance**？
- Optical flow是pixel-level的motion signal，直接对应"哪些pixel在动"
- Language guidance需要额外的encoder（computationally expensive）
- 在Table 5的ablation中验证：language-guided variant只有83.0% success vs optical flow的85.6%

RAFT-small的选择是efficiency-accuracy tradeoff：full RAFT太重，small version在robotic场景下足够。

### 2.2 Cached Token Selector (Token-Level Decision)

$$S_t = f_{\text{sel}}(V_t; \theta_{\text{sel}})$$

变量解释：
- $S_t = \{s_t^{(1)}, \ldots, s_t^{(N)}\}$：N个token的saliency scores
- $s_t^{(i)} \in [0, 1]$：第i个token在第t步的importance
- $\theta_{\text{sel}}$：selector网络参数
- $f_{\text{sel}}$：small CNN（lightweight，关键design choice）

**Intuition**: $s_t^{(i)}$高 → token需要recompute（比如gripper在移动的区域）；$s_t^{(i)}$低 → token可以cache（static background）。

为什么用CNN而不是Transformer？因为CNN的inductive bias对"局部motion detection"更高效，且computation cost远低于backbone。

### 2.3 Cache Ratio Predictor (Scene-Level Decision)

$$L_t = f_{\text{pred}}(V_t; \theta_{\text{pred}})$$

变量解释：
- $L_t = \{l_t^{(1)}, \ldots, l_t^{(C)}\}$：C个candidate ratios的logits
- $\mathcal{R} = \{r_1, \ldots, r_C\}$：discrete set of candidate cache ratios（比如{0.2, 0.4, 0.6, 0.8}）
- $l_t^{(j)}$：选择ratio $r_j$的confidence

**Intuition**: 在static scene中predict high ratio（maximize efficiency）；在dynamic scene中predict low ratio（preserve accuracy）。这是scene-level budget allocation，complement token-level的fine-grained decision。

### 2.4 两模块的Cooperative机制

```
Scene dynamics → Cache Ratio Predictor → overall budget k_t = N · r_t
                                                        ↓
Frame + Flow → Cached Token Selector → S_t (ranking)
                                                        ↓
                                          Top-k_t selection → M_t (binary mask)
                                                        ↓
                              Active tokens (recompute) | Cached tokens (reuse)
```

这是一个hierarchical decision process：ratio决定"how much"，selector决定"which"。

## 3. Training Procedure: 两阶段策略的必要性

### 3.1 Stage I: Attention Alignment Initialization

$$\mathcal{L}_{\text{align}} = \text{MSE}(f_{\text{sel}}(V_t; \theta_{\text{sel}}), S_{\text{VLA}})$$

变量解释：
- $S_{\text{VLA}}$：frozen VLA model的attention map（aggregated across layers）
- MSE：mean squared error between selector output和VLA attention

**Why this stage is critical** (Table 5 ablation: 79.2% without it vs 85.6% with it)：
- 从scratch学习discrete selection policy在sparse supervision下unstable
- VLA attention虽然不是optimal policy，但提供了reasonable "visual saliency prior"
- 类似于policy distillation的思想：先用sub-optimal但stable的expert warm-up

这和RL中的"behavior cloning before policy gradient"思想一致：先让policy有一个reasonable starting point，再做optimization。

### 3.2 Stage II: Joint Optimization with Task Gradients

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{VLA}} + \lambda \mathcal{L}_{\text{ratio}}$$

$$\mathcal{L}_{\text{ratio}} = -\mathbb{E}_{\tilde{p}_t}[r] = -\sum_{j=1}^{C} \tilde{p}_t^{(j)} r_j$$

变量解释：
- $\mathcal{L}_{\text{VLA}}$：VLA task loss（action prediction的cross-entropy或diffusion loss）
- $\lambda$：trade-off权重
- $\tilde{p}_t^{(j)}$：Gumbel-Softmax输出的soft probability for ratio $r_j$
- $r_j$：candidate cache ratio
- $\mathcal{L}_{\text{ratio}}$的negative sign：鼓励predictor选择higher ratio（maximize efficiency）

**Intuition**: $\mathcal{L}_{\text{VLA}}$确保task performance不degrade；$\mathcal{L}_{\text{ratio}}$提供efficiency pressure。两者balance就是$\lambda$的作用。

### 3.3 The Differentiability Challenge

关键问题：decision是discrete的（argmax和top-k都不可微），但我们需要gradient flow from $\mathcal{L}_{\text{VLA}}$ back to决策模块。

**Solution**: Gumbel-Softmax + Straight-Through Estimator (STE)

#### For Cache Ratio Predictor:

$$\tilde{p}_t^{(j)} = \frac{\exp((l_t^{(j)} + g_j)/\tau)}{\sum_{k=1}^{C} \exp((l_t^{(k)} + g_k)/\tau)}$$

变量解释：
- $g_j \sim \text{Gumbel}(0, 1)$：Gumbel noise（采样自Gumbel distribution）
- $\tau$：temperature parameter（控制softness）
- $l_t^{(j)}$：logit for ratio $r_j$

**Forward pass** (deterministic):
$$p_t = \text{one\_hot}(\arg\max_j(\tilde{p}_t^{(j)}))$$

**Backward pass**: STE直接把gradient传给$\tilde{p}_t$（绕过argmax的non-differentiability）。

这是"hard forward, soft backward"的hybrid策略：inference时用discrete decision保证efficiency；training时用soft approximation保证gradient flow。

#### For Cached Token Selector:

$$\tilde{M}_t^{(i)} = \sigma\left(\frac{s_t^{(i)} - \theta_k}{\tau_s}\right)$$

变量解释：
- $s_t^{(i)}$：token i的saliency score
- $\theta_k$：第k-th token的score threshold（因为要选top-k）
- $\tau_s$：temperature（steep sigmoid ≈ step function when $\tau_s \to 0$）
- $\sigma$：sigmoid function

**Forward**: hard mask $M_t$ via top-k
**Backward**: gradient through $\tilde{M}_t^{(i)}$

Steep sigmoid是top-k的可微approximation：当$\tau_s$很小时，sigmoid接近step function，gradient集中在threshold附近。

## 4. Inference Pipeline: 实际部署细节

### 4.1 Deterministic Decision

Inference时stochastic sampling被deterministic argmax替代：
1. Cache Ratio Predictor: $\arg\max$ → $r_t$
2. Cached Token Selector: top-$(N \cdot r_t)$ lowest scores → cached
3. 生成binary mask $M_t \in \{0, 1\}^N$

### 4.2 Efficient Forward Pass

在Transformer forward pass中的关键modifications（Appendix B）：

**Position Management**:
- Cached tokens保留previous step的positional encoding
- Active tokens获得new rotary embeddings

**Attention Mask Pruning**:
- Self-attention只计算active tokens之间的attention
- Active queries attend to full context（active + cached的K, V）

**Dynamic KV Cache Updates**:
$$\mathbf{K}_t^l = \text{update}(\mathbf{K}_{t-1}^l, \mathbf{K}_{\text{new}}^l, M_t)$$
$$\mathbf{V}_t^l = \text{update}(\mathbf{V}_{t-1}^l, \mathbf{V}_{\text{new}}^l, M_t)$$

这是partial update：active tokens用新计算的K, V；cached tokens用上一步的K, V。

### 4.3 Stochastic Recovery Mechanism

为了mitigate error accumulation：
- 每步以small probability $p_{\text{recover}}$随机refresh一部分cached tokens
- Figure 3和Figure 5中的green tiles就是recovered tokens
- 在Ablation Table 2中，加recovery后从83.4% → 85.6%（+2.2pp）

**Intuition**: 完全cache会让stale信息一直累积，periodic refresh像是一个"checkpoint"，保证long-horizon robustness。

## 5. Theoretical Complexity Analysis

### 5.1 Baseline Cost per Layer

$$\mathcal{C}_{\text{base}} \approx \underbrace{4ND^2 + 2N^2D}_{\text{MSA}} + \underbrace{2NDM}_{\text{FFN}}$$

变量解释：
- $N$: tokens per frame
- $D$: embedding dimension
- $M$: FFN intermediate dimension
- $L$: Transformer layers

分项：
- $4ND^2$: QKV projection ($3ND^2$) + output projection ($ND^2$) = $4ND^2$
- $2N^2D$: attention score computation ($N^2D$) + attention aggregation ($N^2D$)
- $2NDM$: FFN的两个linear layer（$D \to M$和$M \to D$）

### 5.2 LAC Cost per Layer

$$\mathcal{C}_{\text{lac}} \approx \underbrace{4N_{\text{act}}D^2 + 2(N_{\text{act}} \cdot N)D}_{\text{Partial MSA}} + \underbrace{2N_{\text{act}}DM}_{\text{Partial FFN}}$$

where $N_{\text{act}} = (1-\rho)N$

**Key insight**: 
- FFN和projection的cost线性减少（只处理active tokens）
- Attention的cost：$N_{\text{act}}$个queries attend to $N$个keys（cached keys仍然需要被attend to）
- 所以attention的reduction是$N_{\text{act}}/N$倍queries，但keys不变

### 5.3 Policy Overhead

$$\mathcal{C}_{\text{policy}} \approx \mathcal{O}(H \cdot W \cdot C_{\text{cnn}})$$

变量解释：
- $H, W$: image height, width
- $C_{\text{cnn}}$: CNN的constant parameter cost

**关键**: policy overhead与Transformer depth $L$和sequence length无关，是per-frame的fixed cost。在$L$很大时，overhead被amortized。

### 5.4 Total FLOPs Reduction

$$\Delta\text{FLOPs}_{\text{total}} \approx \sum_{l=1}^{L} \Delta\text{FLOPs}_{\text{layer}} - \mathcal{C}_{\text{policy}}$$

实验数据验证：LIBERO上FLOPs从1.864T → 1.392T，减少25.3%。

## 6. Experimental Results Deep Dive

### 6.1 LIBERO Benchmark (Table 1)

| Method | Spatial | Object | Goal | Long | Avg | FLOPs(T) | CUDA(ms) |
|--------|---------|--------|------|------|-----|----------|----------|
| OpenVLA (baseline) | 84.4% | 86.6% | 75.6% | 53.2% | 75.0% | 1.864 | 51.91 |
| SparseVLM | 79.8% | 67.0% | 72.6% | 39.4% | 64.7% | 1.407 | 83.39 |
| FastV | 83.4% | 84.0% | 74.2% | 51.6% | 73.3% | 1.864 | 53.28 |
| VLA-Cache | 83.8% | 85.8% | 76.4% | 52.8% | 74.7% | 1.355 | 31.83 |
| **LAC (Ours)** | **85.6%** | 86.2% | 76.6% | **59.2%** | **76.9%** | 1.392 | **29.51** |

**Key observations**:
1. **SparseVLM的counter-intuitive result**: FLOPs降低但CUDA time反而增加（83.39ms vs 51.91ms）。这是因为pruning logic的overhead超过benefit，在short action sequences上尤其明显。这验证了"theoretical FLOPs ≠ wall-clock speedup"。
2. **FastV**: FLOPs几乎不变（因为只在特定layer pruning），speedup微乎其微。
3. **LAC vs VLA-Cache**: LAC在所有4个subtask上都有提升，特别是Long task提升5.6pp（52.8% → 59.2%），说明learned policy在long-horizon任务上更有优势。

### 6.2 LIBERO-Long的提升来源

Long task是LIBERO中最challenging的（需要long-horizon planning）。LAC的5.6pp提升来源于：
1. Stochastic recovery mechanism防止error accumulation
2. Adaptive ratio在critical moments自动降低cache ratio
3. Task-aware selection避免critical cue被错误cache

### 6.3 SIMPLER Benchmark (Table 3)

在CogAct（diffusion action decoder）上的结果：
- Visual Matching: 74.8% → 75.5%（+0.7pp），1.42× speedup
- Variant Aggregation: 61.3% → 63.0%（+1.7pp）

**Significance**: 验证LAC的architecture-agnostic特性，能port到不同的action decoder（autoregressive和diffusion-based都work）。

### 6.4 Real-World Robot (Table 4)

| Method | KnockCrisp | PickMango | CoverBanana | KnockBottle | Avg | FLOPs(T) | CUDA(ms) |
|--------|------------|-----------|-------------|------------|-----|----------|----------|
| OpenVLA | 48.0% | 16.0% | 24.0% | 44.0% | 33.0% | 1.893 | 37.38 |
| VLA-Cache | 48.0% | 12.0% | 20.0% | 40.0% | 30.0% | 1.534 | 33.36 |
| LAC | 52.0% | 12.0% | 24.0% | **64.0%** | **38.0%** | 1.569 | 32.47 |

**Notable**: KnockBottle从44% → 64%（+20pp！），这暗示在dynamic interaction tasks上，learned policy的优势更明显。Real-world的+5.0pp average提升比simulation的+1.9pp更大，可能因为real-world有更多visual noise，learned policy能更好distinguish task-relevant dynamics from noise。

## 7. Ablation Studies - 组件贡献分析

### 7.1 Component-wise Ablation (Table 2)

| Method | Success(%) | FLOPs(T) | Time(ms) |
|--------|-----------|----------|----------|
| Selector only | 82.20 | 1.283 | 28.48 |
| + Reuse Predictor | 83.40 | 1.325 | 29.04 |
| + Recovery (Full) | 85.60 | 1.377 | 29.32 |

**Interesting tradeoff**:
- 加Reuse Predictor后FLOPs和时间略增（因为它会adaptive降低cache ratio在critical moments）
- 但success rate提升+1.2pp
- 加Recovery后FLOPs再增，success再提升+2.2pp

这说明**optimal efficiency-accuracy tradeoff不是maximal caching**，而是adaptive caching with safety mechanisms。

### 7.2 Two-Stage Training Necessity (Table 5)

| Variant | Success(%) |
|---------|-----------|
| w/o Stage I Initialization | 79.2 |
| Language-guided Policy | 83.0 |
| LAC (full) | 85.6 |

- Without Stage I: 下降6.4pp，证明cold-start problem严重
- Language-guided: 下降2.6pp，证明optical flow比language更efficient且effective

## 8. 与Related Work的Positioning

### 8.1 vs VLA-Cache [48]

VLA-Cache是most related work：
- 都做KV caching across timesteps
- VLA-Cache用static, rule-based policy（基于attention scores）
- LAC用learned, task-driven policy
- LAC在LIBERO上比VLA-Cache高2.2pp success，快2.32ms

VLA-Cache的limitation：rule-based无法adapt to scene dynamics，比如在critical interaction moment可能cache了不该cache的token。

### 8.2 vs Token Pruning Methods (SparseVLM, FastV)

这些方法源于VLM acceleration：
- **Spatial reduction**: 在single frame内prune unimportant tokens
- **No temporal reuse**: 每步仍需要处理全部remaining tokens

LAC的优势：
- **Temporal reuse**: 利用cross-frame redundancy
- **Learned**: 直接优化task performance
- **Wall-clock efficient**: 没有额外overhead

SparseVLM在LIBERO上的failure（CUDA time反而增加）证明了VLM acceleration方法直接apply到VLA上不work，因为robotic action sequences短，overhead无法被amortize。

### 8.3 vs High-Frequency Control Methods [18, 34, 50]

这些方法（如FiVA, HIRT）通过action tokenization或hierarchical structure加速，但**没有address vision processing bottleneck**。LAC是complementary的，可以和这些方法结合。

## 9. Limitations & Future Directions

### 9.1 Stated Limitations

1. **Optical flow在extreme visual conditions下可能失效**（fast motion, motion blur）
2. **Highly dynamic scenes下efficiency gain减少**（policy会自动选择低cache ratio）

### 9.2 我看到的Potential Extensions

1. **Cross-frame semantic consistency**: 目前只考虑motion，未来可以引入object tracking
2. **Learnable recovery probability**: 现在是fixed $p_{\text{recover}}$，可以做成state-dependent
3. **Multi-scale caching**: 不同Transformer layer用不同cache ratio（shallow layers可能更robust to caching）
4. **Adaptive action chunking**: 结合FAST [34]的action tokenization，jointly optimize vision caching和action chunking
5. **3D scene understanding**: 在3D point cloud的VLA model上extend（如π0.5 [15]的flow matching framework）
6. **Distillation from larger VLA**: Stage I的attention alignment可以扩展成full policy distillation

### 9.3 Connection to Broader ML Trends

这个工作有几个connection到broader ML research：

1. **Conditional Computation**: 类似于Mixture of Experts (MoE)，但这里是spatial-temporal conditional computation
2. **Learned vs Heuristic Scheduling**: 类似于learned query scheduling in database，vs rule-based optimizer
3. **Differentiable Discrete Optimization**: Gumbel-Softmax + STE是standard toolkit，这里apply到robotics inference scheduling
4. **Policy Distillation**: Stage I的attention alignment本质是policy distillation from frozen expert

## 10. 实用Takeaways

如果你要deploy一个VLA model，LAC给了几个immediate actionable insights：

1. **Visual token caching在robotics上work**：cross-frame redundancy确实huge，1.76× speedup是real的
2. **Rule-based caching不够**：需要task-aware learned policy
3. **Optical flow是efficient motion prior**：比language guidance更便宜
4. **Two-stage training很重要**：warm-up + end-to-end optimization比pure RL-style learningstable得多
5. **Stochastic recovery是cheap insurance**：+2.2pp with minimal cost
6. **Wall-clock > FLOPs**: 实测CUDA time比theoretical FLOPs更重要（SparseVLM的反例）

## 11. Web References

- Paper (arxiv): https://arxiv.org/abs/2502.02175 (VLA-Cache, the most related baseline)
- OpenVLA: https://openvla.github.io/ 
- CogAct: https://arxiv.org/abs/2411.19650
- LIBERO benchmark: https://libero-project.github.io/
- SIMPLER: https://simpler-env.github.io/
- Gumbel-Softmax: https://arxiv.org/abs/1611.01144
- Straight-Through Estimator: https://arxiv.org/abs/1308.3432
- RAFT (optical flow): https://arxiv.org/abs/2003.12039
- SparseVLM: https://arxiv.org/abs/2410.04417
- FastV: https://arxiv.org/abs/2403.06764
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- RT-2: https://robotics-transformer2.github.io/
- PaLM-E: https://palm-e.github.io/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- LoRA: https://arxiv.org/abs/2106.09685

## 12. Final Intuition Building

把LAC的整个思想浓缩成一句话：

**"Let the task loss teach the model what to see and what to remember, rather than using attention scores as a proxy for importance."**

这是一个从heuristic到learned，从proxy到direct，从static到adaptive的paradigm shift。在robotics这种real-time, safety-critical的应用中，这种task-aligned efficiency优化尤为重要。

Deep down，LAC揭示了一个deeper insight：**inference computation的allocation本身就是一个policy**，应该被end-to-end optimized for task performance，rather than being determined by some arbitrary proxy metric。这个insight其实可以extend到很多其他场景：LLM inference中的speculative decoding，video understanding中的frame sampling，甚至是neural rendering中的ray sampling——都是computation allocation policy，都可以被learned。

希望这个technical deep dive对你build intuition有帮助，Andrej！这个工作的 elegance在于把一个看似engineering的问题（acceleration）reformulate成learning problem，并给出了clean的solution。
