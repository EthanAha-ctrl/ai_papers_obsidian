---
source_pdf: Elastic Attention Cores for Scalable.pdf
paper_sha256: 91b1dbcede9ce90aef3de64347d679d6afeb9e3895d2d4de1561602e377dc470
processed_at: '2026-08-18T10:30:37-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VECA 用人话说

## 一句话总结

**把所有 patch 之间互相看来走去掉的 attention，改成让所有 patch 只跟一小撮 "core" tokens 说话，core tokens 之间互相说话。这样就从 $N^2$ 变成 $O(N)$，而且效果没怎么掉。**

## 这篇 paper 到底在干啥

### 痛点

ViT 的问题老生常谈了：每个 patch 都要跟其他所有 patch 算 attention，N 个 patch 就是 $N^2$ 次比较。图片分辨率一高，直接爆炸。

之前的人想了一堆招：linear attention、sliding window、token merging... 但都是在 "保留 self-attention" 这个大框架下做 approximation。

VECA 的作者直接说：**也许 patch 之间根本不需要直接说话。**

### 核心想法

想象一个房间里有 256 个人（patches），现在有两种开会方式：

**传统 ViT**: 每个人跟其他 255 个人挨个握手聊天。信息交流很充分，但 $256 \times 256 = 65536$ 次握手，累死。

**VECA**: 房间里放 8 个 "core" tokens 当中间人。所有人只跟这 8 个中间人说话，中间人之间互相聊。握手次数变成 $256 \times 8 \times 2 + 8 \times 8 = 4160$ 次。

关键发现是：**这玩意儿效果居然没差多少**。

## 架构细节讲讲

### Attention 矩阵长啥样

Standard self-attention 是个满矩阵，$N \times N$ 全有值。

VECA 是个 block-sparse 矩阵：

```
       Cores    Patches
Cores  [ 全满 ]  [ 全满 ]
Patches[ 全满 ]  [ 全零 ]  ← patch 之间不互相 attend
```

- Core tokens 的 query 看所有东西（包括 patches 和其他 cores）
- Patch tokens 的 query 只看 core tokens
- Patch 之间 **直接 attention 被完全砍掉**

### 那信息怎么在 patch 之间流动？

靠 core tokens 当中转站。Patch A 想知道 Patch B 的信息，路径是：

Patch A → (attend to cores) → Core → (core 之间互相 attend) → Core → (patch attend to core) → Patch B

需要 **2 hop**，所以 graph diameter 是 2。Standard self-attention 是 1 hop。

这意味着 VECA 至少得堆 2 层 block 才能让信息在整个 image 里传开。一层不够。

### Core tokens 是啥

- 一组 learnable embeddings，从 scratch 训
- 数量固定（比如 64 个），跟 image 分辨率无关
- 有自己的 spatial coordinate，会随着 layer 往里走慢慢更新
- 第一个 core token 当 CLS token 用

Core coordinate 的更新公式（论文公式 3）：

$$\rho_i^{\ell+1} = \rho_i^\ell + \alpha_\ell \cdot f_\ell(r_i^\ell)$$
$$u_i^{\ell+1} = \tanh(\rho_i^{\ell+1})$$

说人话就是：core 每层都根据自己的 feature 预测一个 delta，往自己的坐标上加。$\alpha_\ell$ 是个很小的 scalar 控制步长，$\tanh$ 把坐标限制在 $[-1, 1]^2$ 的 image plane 里。

初始化用 **farthest-point sampling** 在 image plane 上撒点，类似 k-means++ 的思路，保证一开始 cores 在空间上散开。

### Elastic 机制（最骚的设计）

训练时每个 step 随机 sample 一个 core budget $C \in \{8, 16, 24, 32, 40, 48, 56, 64\}$，只用前 $C$ 个 cores。

采样权重偏向更大的 budget：$(1, 1, 2, 2, 3, 3, 4, 4)$。

因为总是取 prefix（前 $C$ 个），所以：
- Core 1 在所有 budget 下都 active，必须学最 essential 的东西
- Core 64 只在 C=64 时才 active，可以学很 specific 的细节

这叫 **nested dropout**，类似 Matryoshka 套娃。

效果是：inference 时你可以随便选 C。要快就 C=8，要准就 C=64，同一个 model 就行，不用 retrain。

### 训练方式

没有 label，用 **DINOv3 当 teacher 做 feature distillation**。训练数据是 Object365 的无标注图片。

Loss = global feature 对齐 + dense patch feature 对齐。公式 4 那个：

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{dense}}$$

Global loss 是 cosine distance，dense loss 是 cosine + MSE 的组合。

## 结果怎么样

### 分类任务（ImageNet-1K）

| Model | Acc | 跟 DINOv3 比 |
|-------|-----|-------------|
| DINOv3-B | 83.56% | - |
| VECA-B (C=64) | 81.93% | -1.63% |
| VECA-B (C=8) | 79.99% | -3.57% |

掉 1.6 个点，但 attention interaction 少了 87%。

### 分割任务（PASCAL Context）

| Model | mIoU | 跟 DINOv3 比 |
|-------|------|-------------|
| DINOv3-B | 57.74 | - |
| VECA-B (C=64) | 57.46 | -0.28 |
| VECA-B (C=8) | 53.30 | -4.44 |

这个差距小得惊人，dense prediction 几乎没掉点。

### 效率（1024×1024 分辨率）

| Model | FLOPs | Speedup |
|-------|-------|---------|
| DINOv3-B | 71.02G | 1× |
| VECA-B | 21.25G | 3.34× 更少 FLOPs |
| 延迟 | 4.08ms → 2.20ms | 1.86× 更快 |

分辨率越高，优势越明显，因为 quadratic 的 $N^2$ 项被干掉了。

## 几个有意思的发现

### 1. Core tokens 自然变成 object-centric

没人逼它，但训练完发现不同 core tokens 会 attend 到不同物体上。比如一个 core 专门看碗里的鸡蛋，另一个看背景。类似 slot attention 的效果，但 VECA 是 feedforward 的，不需要 recurrent。

### 2. 浅层 isotropic，深层 semantic

Figure 7 那个可视化很 striking：

- **Layer 2-3**: core attention 是 blob 状的、各向同性的，就像没聚好的 k-means
- **Layer 8-12**: core attention 变成 semantic clusters，每个 core 对应一个有意义的 region

这个 emergence 完全是 self-organized 的，没有任何 segmentation supervision。

### 3. Classification 对 C 不敏感，dense 对 C 敏感

- 分类任务：C=8 到 C=64，ImageNet acc 只掉 2 个点
- 分割任务：C=8 到 C=64，mIoU 差好几个点

这符合直觉：分类靠 global info（早期 cores 就够了），分割要 fine-grained spatial info（需要更多 cores 来 spatially 分辨）。

## 跟之前工作的区别

### vs Perceiver

Perceiver 也有 latent bottleneck tokens，但只更新 latents，inputs 不更新。要 dense output 还得加 decoder。VECA 同时更新 cores 和 patches，直接拿 patch features 就能用，不需要额外 decoder。

### vs Slot Attention

Slot Attention 是 recurrent 的，每轮迭代更新 slots。VECA 是 feedforward 的，一层一层往前走。而且 Slot Attention 通常只做 object discovery，VECA 是完整 vision backbone。

### vs Token Merging (ToMe)

ToMe 动态把相似 patch merge 掉，是 input-dependent 且 merge 后丢了 spatial alignment。VECA 的 patches 始终保持 spatial 对齐，对 dense prediction 友好。

### vs Linear Attention

Linear attention（Performer、Linformer 之类）通过 approximation 把 softmax 搞成可分解的。VECA **保留 softmax**，只是把 attention matrix 的结构从 dense 改成 block-sparse。expressive power 保留得更好。

### vs Swin Transformer

Swin 在 local window 内做 full attention。VECA 是 global 的，但通过 bottleneck 限制连接数。Swin 的 receptive field 受 window 限制，VECA 每层都是 global 的。

## 我的直觉解释

### 为什么 patch 之间不直接 attend 也行？

我觉得 core tokens 本质上是在做 **soft clustering**。每个 core 像一个 cluster centroid，patches 通过 attend 到 cores 来确定自己属于哪个 semantic group。

信息的流动是：patch 把自己的信息汇总到 core 里 → core 之间互相 mix → patch 从 core 拿回 global context。这跟 k-means 的 E-step + M-step 很像，只是 differentiable 且 end-to-end。

之前的 ViT 里，每个 patch 都要跟所有其他 patch 算 attention，但其实大部分 attention weight 都很小，information 真正流动的可能就那么几个关键的 tokens。Core tokens 等于显式地把这些 "关键信息节点" learn 出来了。

### 为什么 dense prediction 掉点更少？

Dense prediction 需要每个 patch 的 local representation 保持 spatial alignment。VECA 里 patches 始终保持自己的 identity，只是 attention 被 bottleneck 了。而 Perceiver 那种把 inputs 压成 latents 再 decode 回来的方案，spatial alignment 容易丢。

### Elastic 的 tradeoff 为什么 work？

Nested dropout 强制前面的 cores 学 broadly useful 的东西。这跟 PCA 的 principal components 类似——第一个 component 解释最多 variance，后面的越来越 specific。

训练时如果 C=8，那只有前 8 个 cores 参与计算，gradient 只流到这 8 个。所以 $r_1$ 被 gradient 更新的次数远多于 $r_{64}$。这种 implicit curriculum 让 cores 自然按 importance 排序。

## 我觉得的亮点和问题

### 亮点

1. **想法干净**：不需要 approximation，直接改 attention 的结构
2. **Elastic inference**：一个 model 多种 speed，实用
3. **Emergent object-centricity**：没 supervision 就 emerge 出 object discovery，有意思
4. **Dense prediction 几乎不掉点**：比 classification 表现还好，有点反直觉但 make sense

### 问题

1. **Graph diameter 2**：信息流动慢，浅 model 可能吃亏。Paper 的 model 都是 12-24 层，信息能传开。如果是 4 层的 model，可能 2 hop 的 bottleneck 就太紧了。

2. **Fixed maximum M=64**：不管图片简单还是复杂，最多就 64 个 cores。一张只有一个物体的图也用 64 cores，一张有 50 个物体的街景图也 64 cores。Content-adaptive allocation 可能更优。

3. **Core redundancy 没量化**：到底有几个 cores 是真正 useful 的，有几个是 redundant 的？Pruning 之后能不能更 efficient？

4. **对 LLM 的启发**：这篇是 vision 的，但 idea 能不能搬到 NLP？Text tokens 之间是否也需要 full pairwise attention？如果用 core-periphery 结构，long context 的 $N^2$ 问题能不能缓解？这个方向值得想。

## 总结

VECA 的核心 insight 是：**vision 里，patch 之间的直接 interaction 是冗余的**。一个小的 bottleneck 就能 capture 大部分 information flow，而且还能 elastic 调整。这 opens up 了一个新的 architectural design space。

References：
- [VECA paper](https://arxiv.org/) (this work)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [Nested Dropout (Rippel 2014)](https://proceedings.mlr.press/v32/rippel14.html)
- [Matryoshka Representation Learning](https://proceedings.neurips.cc/paper/2021/hash/a3f5d28b9fe1792692a3efc68321f6ad-Abstract.html)
- [Perceiver](https://arxiv.org/abs/2103.03206)
- [Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c5ab053bd5c81ed380c9137b-Abstract.html)
- [Token Merging](https://arxiv.org/abs/2210.09461)
- [Agent Attention](https://link.springer.com/chapter/10.1007/978-3-031-72390-2_8)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [Core-Periphery Structure](https://epubs.siam.org/doi/10.1137/130917502)

---

# VECA: Visual Elastic Core Attention - 深度解析

## 1. 核心问题与动机

Vision Transformers (ViTs) 的核心痛点在于 self-attention 的 **quadratic complexity**。对于 N 个 patch tokens，standard self-attention 需要 $O(N^2)$ 次比较，这在 high-resolution scenarios 下变得 prohibitive。

VECA 提出了一个反直觉的假设: **pairwise patch-to-patch interaction 对于学习 rich visual-semantic representations 可能不是必需的**。他们用一个小的、resolution-invariant 的 core token set 来 mediate 所有 token interactions。

## 2. Core-Periphery Attention 架构

### 2.1 结构设计

VECA 的 attention matrix 是一个 block-sparse 结构:

- **Core tokens** $R_C = \{r_1, ..., r_C\}$: C 个 learnable tokens，构成 fully connected clique
- **Patch tokens** $Z = \{z_1, ..., z_N\}$: N 个 image patches，只能 attend 到 core tokens

公式 (2) 定义了 attention 操作:

$$R' = \text{Attn}(R_C, X, X), \quad Z' = \text{Attn}(Z, R_C, R_C)$$

其中 $X = [R_C; Z]$ 是 concatenation。Core tokens 的 query attend 到整个 sequence (包括其他 cores 和所有 patches)，而 patch tokens 的 query 只 attend 到 active core prefix $R_C$。

### 2.2 计算复杂度分析

Standard self-attention: $O(N^2)$

VECA attention:
- Core-to-everything: $C \times (C + N) = C^2 + CN$
- Patch-to-core: $N \times C = NC$  
- Total: $2NC + C^2$

当 C 是 predetermined constant 时，复杂度变为 $O(N)$，即 **linear** relative to image resolution。

值得注意的是，这个结构的 **graph diameter 是 2** (而 self-attention 是 1)，意味着信息需要至少两层才能从一个 patch 传递到另一个 patch。这看起来是个限制，但实验表明 performance 依然 competitive。

### 2.3 Core Coordinate 的动态更新

公式 (3) 描述了 core token 的 spatial coordinate evolution:

$$\rho_i^{\ell+1} = \rho_i^{\ell} + \alpha_\ell \cdot f_\ell(r_i^{\ell}), \quad u_i^{\ell+1} = \tanh(\rho_i^{\ell+1})$$

变量解释:
- $\rho_i^{\ell} \in \mathbb{R}^2$: core token $i$ 在 layer $\ell$ 的 unconstrained coordinate state
- $r_i^{\ell}$: core token $i$ 在 layer $\ell$ 的 feature representation  
- $f_\ell$: lightweight coordinate-update head (一个 linear layer)
- $\alpha_\ell$: learned layer-wise scalar，initialized 到 small value
- $u_i^{\ell+1} \in [-1,1]^2$: bounded coordinate，用于 2D axial RoPE
- $\tanh$ 确保 coordinate bounded 在 image plane 内

这个设计让 core tokens 既有 semantic representation 又有 evolving spatial position，类似于 **slot attention** 但 feedforward 而非 recurrent。

### 2.4 与 Perceiver / Set Transformer 的区别

| 方法 | 更新 latents | 更新 inputs | Dense output | Elastic |
|------|-------------|-------------|--------------|---------|
| Perceiver [67] | ✓ | ✗ | 需要 special handling | ✗ |
| Set Transformer [66] | ✓ | ✗ | ✗ | ✗ |
| **VECA** | **✓** | **✓** | **✓ (直接)** | **✓** |

VECA 的关键区别在于 **同时 iterative 更新** core tokens 和 patch tokens，因此天然支持 dense prediction tasks。Perceiver 只 refine latent queries，要得到 dense features 需要 decoder。

## 3. Budget-Adaptive Training (Nested Dropout)

### 3.1 Nested Core Learning

VECA 训练时从分布 $p_C(\cdot)$ 中 sample active core budget $C$，然后取 ordered prefix $R_C = R_M[:C]$。这类似于 **Matryoshka Representation Learning** [23] 和 **nested dropout** [16]。

实际实现中:
- Maximum capacity: $M = 64$
- Active budgets: $\{8, 16, 24, 32, 40, 48, 56, 64\}$
- Sampling weights: $(1, 1, 2, 2, 3, 3, 4, 4)$ (偏向更大 budget)

由于 early cores 在更多 budgets 下被 activate，它们被 encouraged 编码最 broadly useful 的信息；later cores 提供 additional capacity for higher-fidelity representations。这 induce 了一个 **coarse-to-fine** 的 representation hierarchy。

### 3.2 为什么这 work?

我的 intuition: nested dropout 实际上是在训练一个 **ordered basis**，类似于 PCA 的 principal components。第一个 core token 必须能独立 work (当 C=1 时)，所以它需要 capture 最 global、最 essential 的信息。后续 core tokens 可以 specialize 到更 local 或更 specific 的 aspects。

这种 implicit ranking 在 inference time 给了 elastic tradeoff: 用 8 个 cores 快速 inference，或用 64 个 cores 高精度 inference，**无需 retraining**。

## 4. 训练 Objective

### 4.1 Distillation from DINOv3

VECA 用 feature distillation 训练，teacher 是 frozen 的 DINOv3 [21]:

$$\mathcal{L}(x, C) = \mathcal{L}_{\text{global}}(y^{(C)}(x), y^{\star}(x)) + \lambda_{\text{dense}} \cdot \mathcal{L}_{\text{dense}}(Z^{(C)}(x), Z^{\star}(x))$$

其中:
- $y^{(C)}(x)$: VECA 的 global feature (first final core token $r_1^L$)
- $Z^{(C)}(x) = \{z_i^{(C)}(x)\}_{i=1}^N$: VECA 的 dense patch features
- $y^{\star}(x), Z^{\star}(x)$: DINOv3 teacher 的对应 targets
- $\lambda_{\text{dense}} = 1.0$

**Global loss** 是 cosine distance:
$$\mathcal{L}_{\text{global}} = 1 - \frac{\langle y^{(C)}(x), y^{\star}(x) \rangle}{\|y^{(C)}(x)\|_2 \|y^{\star}(x)\|_2}$$

**Dense loss** 结合 cosine distance 和 MSE:
$$\mathcal{L}_{\text{dense}} = \frac{1}{N} \sum_{i=1}^{N} \left(1 - \cos(z_i^{(C)}, z_i^{\star})\right) + \beta_{\text{mse}} \cdot \frac{1}{ND} \sum_{i=1}^{N} \|z_i^{(C)} - z_i^{\star}\|_2^2$$

$\beta_{\text{mse}} = 1.0$。

### 4.2 两阶段训练

- **Stage 1**: $256 \times 256$ resolution, 135 epochs
- **Stage 2**: Multi-resolution finetuning, $\{256, 384, 512, 768\}$, 50K steps

训练数据是 Object365 [137] 的 unlabeled images。

## 5. 实验结果分析

### 5.1 Dense Prediction (Table 1)

在 PASCAL Context segmentation 上:
- DINOv3-B/16: **57.74 mIoU**
- VECA-B/16 (C=64): **57.46 mIoU** (gap: -0.28)
- VECA-B/16 (C=8): 53.30 mIoU (gap: -4.44)

在 NYUv2 depth estimation 上:
- DINOv3-B/16: **0.3684 RMSE**
- VECA-B/16 (C=64): **0.3705 RMSE** (gap: +0.0021)

这些结果相当 impressive: VECA 移除了 **87.1% fewer attention interactions per layer** (C=64 vs full self-attention at 512 resolution)，但 dense prediction performance 几乎持平。

### 5.2 Image Classification (Table 2)

在 ImageNet-1K 上:
- DINOv3-B/16: **83.56%**
- VECA-B/16 (C=64): **81.93%** (gap: -1.87)
- VECA-B/16 (C=8): 79.99% (gap: -3.81)

Classification 的 gap 比 dense prediction 稍大，这可能是因为 classification 更依赖 global token 的 direct interaction。但即使 C=8 (仅 6.3% 的 connections)，仍保持 96% 的 full model performance。

### 5.3 Multi-Resolution Evaluation (Table S.10-S.12)

VECA 的 scaling behavior 在高 resolution 下表现更好。Table S.11 显示在 768 resolution 下:
- DINOv3-B Context: 58.40 mIoU
- VECA-B Context: 58.05 mIoU (gap: -0.35)

但 FLOPs 差距巨大 (Figure S.7):
- DINOv3-B at 1024×1024: **71.02 GFLOPs**
- VECA-B at 1024×1024: **21.25 GFLOPs** (3.34× reduction)
- Latency: 4.08ms → 2.20ms (1.86× speedup)

### 5.4 Budget Ablation (Table S.15, S.16)

Classification 对 budget 减少相对 robust: C=32 vs C=64 在 ImageNet-1K 上只 drop ~0.3%。

Dense prediction 对 budget 更敏感: C=32 vs C=64 在 ADE 上 drop ~1.2 mIoU，在 NYUv2 上 drop ~0.008 RMSE。

这 aligns with intuition: classification 需要 global information (early cores suffice)，dense prediction 需要 fine-grained spatial details (需要更多 cores)。

## 6. Emergent Behaviors

### 6.1 Object-Centric Cores

Figure 6 显示不同 core tokens 会 attend 到 distinct objects/parts。例如一个 core 可能 attend 到碗里的鸡蛋，另一个 attend 到背景。这 emergent behavior 类似于 **slot attention** [111] 的 object-centric learning，但 VECA 没有 explicit objectness supervision。

### 6.2 Isotropic-to-Semantic Evolution

Figure 7 展示了一个 striking 的现象: core attention maps 在浅层是 **isotropic** (spherical, blob-like)，在深层逐渐变成 **semantically clustered** structures。这个 evolution 发生在 **feedforward** processing 中，不像 slot attention 需要 recurrent updates。

公式 (5) 定义了 output-contribution attention:
$$e_{ij}^{\ell} = \sum_{h=1}^{H} A_{hij}^{\ell} (v_{hj}^{\ell} W_h^O), \quad s_{ij}^{\ell} = \frac{\|e_{ij}^{\ell}\|_2}{\sum_{j'} \|e_{ij'}^{\ell}\|_2}$$

这个 metric 比 raw attention averaging 更能反映 actual output contribution，因为它 account for value magnitude 和 output projection [170]。

## 7. 我的 Intuition 与联想

### 7.1 为什么 removing patch-to-patch attention 仍然 work?

我的 hypothesis: **Core tokens 本质上是在做 soft clustering**。每个 core token 学习 attend 到 image 中某个 semantic/spatial region，类似一个 cluster centroid。Patch tokens 通过 attend 到 cores 来确定自己属于哪个 "cluster"，并从 cluster centroid 获取 context。

这类似 **k-means 的 soft 版本**，但 differentiable 且 end-to-end learned。Isotropic-to-semantic 的 evolution 正是 cluster centroids 从 random initialization 收敛到 semantic groups 的过程。

### 7.2 与 Mamba / Linear RNNs 的关系

Paper 提到 linear RNNs [72-84] 作为 efficient sequence modeling 的 alternative。但 VECA 的 approach 本质不同:
- Linear RNNs 用 **state-space models** 实现 causal sequence processing
- VECA 用 **bottleneck tokens** 实现 bidirectional information flow

VECA 的优势在于保留 **softmax attention** (expressive) 同时实现 linear complexity，而 linear RNNs 通常需要 sacrifice expressivity。

### 7.3 与 Agent Attention [93] 的区别

Agent Attention 也试图 combine softmax 和 linear attention，用 nested linear attention。但 VECA **不需要 nested softmax**，直接用 standard scaled dot-product attention，更简洁。

### 7.4 与 Token Merging [44] 的区别

Token merging (ToMe) 动态 merge similar tokens 来 reduce count。但 ToMe 是 **input-dependent** 且 merge 后的 tokens 不一定 spatially aligned。VECA 的 patch tokens 始终 maintain spatial alignment，这对 dense prediction 重要。

### 7.5 Graph Theory 视角

Core-periphery structure 是 network science 中的经典概念 [13,14]。在 social networks 中，core nodes 是 high-degree hubs，periphery nodes 只 connect 到 cores。VECA 把这个结构 explicit 强加到 attention graph 上，creating 一个 **star-of-cliques** topology。

Graph diameter 2 意味着 information propagation 需要 2 hops: patch → core → patch。这类似 **message passing neural networks** 的 2-layer propagation。

### 7.6 与 Mixture of Experts (MoE) 的联系

Nested core selection 有点像 **elastic MoE** [47,48]: 不同 budget 下 activate 不同的 "expert" cores。但 VECA 的 cores 不是 conditional activated，而是 prefix-selected，更 deterministic。

### 7.7 Potential Limitations

1. **Fixed maximum capacity M**: 不适应 content complexity。简单 image 可能只需 8 cores，复杂 scene 可能需要 128 cores。
2. **Two-layer information mixing**: 由于 graph diameter 2，需要至少 2 个 VECA blocks 才能 achieve full information propagation。Very shallow VECA 可能 underperform。
3. **Core redundancy**: Paper 没有量化 core 之间的 redundancy。如果多个 cores attend 到相同 region，就 wasteful。
4. **No explicit objectness supervision**: Emergent object-centric behavior 可能 fragile，需要更多 analysis。

### 7.8 Future Directions 我的猜想

1. **Content-adaptive core allocation**: 用 a small predictor network 根据 image complexity 动态决定 C
2. **Hierarchical cores**: 不同 layer 用不同 resolution 的 cores，类似 FPN
3. **Cross-modal cores**: Cores 可以 shared across modalities (image-text)，类似 Perceiver IO [68]
4. **Core pruning**: 训练后 prune redundant cores，类似 lottery ticket hypothesis
5. **Slot attention hybrid**: 用 slot attention 做 initial core assignment，VECA 做 refinement

## 8. Implementation Details 我觉得有意思的

### 8.1 Optimizer Setup

Paper 用了一个 unusual 的 **two-optimizer setup**:
- **NorMuon** [160,161] for selected 2D linear weights
- **AdamW** [162,163] for remaining parameters (cores, coordinates, biases, etc.)

NorMuon 是 Muon optimizer 的改进版，使用 **Polar Express** [167] 做 orthogonalization (5 iterations)。Cautious weight decay [166] 也被使用。这显示训练 stability 是一个 concern。

### 8.2 Core Coordinate Initialization

Cores 用 **farthest-point sampling** 在 normalized image plane $[-1,1]^2$ 内初始化，确保 spatial coverage。这比 random initialization 更 sensible，类似 k-means++ 的思想。

### 8.3 First Core as CLS Token

$y(x) = r_1^L$ — 第一个 core token 用作 global representation。由于 nested training 时 $r_1$ 在所有 budgets 下都 active，它被 forced 学习最 essential 的 global information。这是一个 elegant 的 design choice。

### 8.4 SwiGLU FFN

$$[u, v] = W_1 z, \quad \text{FFN}(z) = W_2 (\text{SiLU}(u) \odot v)$$

这是 LLaMA-style 的 FFN，比 standard MLP 更 efficient (fewer params for same capacity)。

## 9. 总结评价

VECA 是一个 **conceptually clean** 的 work，核心 idea 简单但 effective:
- 用 core-periphery structure 替代 full self-attention
- Linear complexity with softmax attention
- Nested dropout enables elastic inference
- Competitive performance with DINOv3 (gap < 2% on classification, < 0.3 mIoU on segmentation)

最 striking 的发现是 **patch-to-patch interaction 可能不是必需的**。这 challenge 了 ViT 的基本 assumption，opens up 新的 architectural design space。

实验结果表明 VECA 在 dense prediction 上尤其 strong，这可能是因为 patch tokens 始终 maintain spatial alignment，而 core tokens 提供 global context。Classification 稍弱可能是因为 global token interaction 受 bottleneck 限制。

**Emergent object-centric behavior** 是另一个 exciting 发现，suggesting core tokens 自然 emerge 出 object discovery capability，即使没有 explicit supervision。这连接到 slot attention 和 object-centric representation learning 的 literature。

## References

- VECA paper (this work)
- [DINOv3](https://arxiv.org/abs/2508.10104) - Teacher model
- [Nested Dropout](https://proceedings.mlr.press/v32/rippel14.html) - Rippel et al., ICML 2014
- [Matryoshka Representation Learning](https://proceedings.neurips.cc/paper/2021/hash/a3f5d28b9fe1792692a3efc68321f6ad-Abstract.html) - Kusupati et al., NeurIPS 2022
- [Perceiver](https://arxiv.org/abs/2103.03206) - Jaegle et al., ICML 2021
- [Perceiver IO](https://arxiv.org/abs/2107.14795) - Jaegle et al., ICLR 2022
- [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) - Lee et al., ICML 2019
- [Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c5ab053bd5c81ed380c9137b-Abstract.html) - Locatello et al., NeurIPS 2020
- [Core-Periphery Structure](https://epubs.siam.org/doi/10.1137/130917502) - Rombach et al., SIAM 2014
- [Matformer](https://proceedings.neurips.cc/paper_files/paper/2024/hash/578dcccce48703b75f5a2b8e9ee8e538-Abstract-Conference.html) - Kudugunta et al., NeurIPS 2024
- [Agent Attention](https://link.springer.com/chapter/10.1007/978-3-031-72390-2_8) - Han et al., ECCV 2024
- [Token Merging](https://arxiv.org/abs/2210.09461) - Bolya et al., 2022
- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) - Darcet et al., 2023
- [Muon Optimizer](https://kellerjordan.github.io/posts/muon/) - Jordan et al., 2024
- [AM-RADIO](https://arxiv.org/abs/2312.00752) - Ranzinger et al., CVPR 2024
- [DINOv2](https://arxiv.org/abs/2304.07193) - Oquab et al., 2023
- [Objects365](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_paper_2019.pdf) - Shao et al., ICCV 2019

这篇 paper 让我联想到一个更 general 的问题: **在 attention-based models 中，有多少 interactions 是 redundant 的?** VECA 的结果 suggest 在 vision domain，大部分 patch-to-patch interactions 可以被一个 small bottleneck 替代。这个 insight 是否 extend 到 language domain? 如果 LLMs 也能用 core-periphery attention，能否 drastically reduce inference cost? 这值得 future exploration。
