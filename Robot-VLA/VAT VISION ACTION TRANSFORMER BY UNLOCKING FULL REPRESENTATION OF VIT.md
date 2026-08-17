---
source_pdf: VAT VISION ACTION TRANSFORMER BY UNLOCKING FULL REPRESENTATION OF VIT.pdf
paper_sha256: d34841e8a8a06d57f9ff08c2852ff7d2eb1f98929519fea27b912deadc19745d
processed_at: '2026-08-13T00:09:41-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 VAT

## 一句话版本

大家用ViT都只拿最后一层输出当feature喂给policy，VAT说"凭啥只用最后一层啊，中间层全是宝贝"，于是把action token塞进ViT每一层，让它们逐层去"看"vision feature。

## 为啥只看最后一层有问题

你训练一个SigLIP，最后一层是被contrastive loss硬拽到跟text对齐的semantic space里。这个space做zero-shot classification很爽，但pixel-level的几何细节被挤没了。DINOv2好一点，dense prediction友好，但照样会丢local low-level info。

ViT是27层堆叠的transformer，每层输出一个representation。整个forward过程是一条trajectory：从浅层的edge/contour/texture，到中层的object part，再到深层的semantic concept。主流做法只取trajectory终点，相当于看完一整部电影只记住结局。

VAT的核心claim：中间那些层的representation没被任何loss显式约束过，反而保留了"没被压缩干净"的visual evidence。Robot manipulation恰恰需要这些fine-grained geometry——你要抓个杯子，得知道杯把在哪、啥形状、跟桌面的相对位置。这些信息在SigLIP最后一层里早就被semantic label "cup" 吃掉了。

## 架构怎么搞的

ViT原来就跑vision tokens的self-attention，VAT在旁边并行加了个action module。两个module各跑各的，action module通过cross-attention去"看"vision tokens。

每一层发生的事：

1. Vision module正常跑self-attention，参数来自pretrained SigLIP/DINOv2，不动
2. Action tokens先经过FiLM注入task info——就是把task id查表得到embedding，再生成一组scale和shift，对action token做affine变换
3. Action tokens拿自己当Query，拿vision tokens当Key和Value做cross-attention，把视觉信息吸进来
4. Action tokens再过自己的MLP，参数也是独立的新参数

注意一个小细节：第 $l$ 层的action tokens是cross-attend到第 $l-1$ 层的vision tokens，不是当前层。这点paper没ablation，可能是为了稳定也可能是实现顺手。

Action tokens的构造：chunk size K=8，每个action 7维（6-DoF end-effector + 1维gripper），所以一个chunk 56个action token。最后送进一个轻量decoder head出robot action。Loss用L1。

如果用diffusion loss，得加timestep embedding——但cross-attention的输出只保留query长度，timestep token会被吃掉。他们的trick是在第一层把timestep token拼接两次，让它能"穿过"cross-attention。Proprioception token同理处理。这trick读起来有点hacky但确实work。

## 为啥这个设计work

一个关键问题：为啥不直接把action token塞进ViT sequence里让它跟vision tokens共享self-attention，就像CLS token那样？

Paper做了这个ablation（叫VAT-ViT）：把action token当CLS token处理，共享ViT参数。结果97.05%，只比完整VAT的98.15%低1.1个点。这说明**hierarchical representation access本身就是performance gain的主要来源**，独立action module只是锦上添花。

完整VAT相比VAT-ViT的优势主要在long-horizon的LIBERO-10上（96.8% vs 92.4%）。独立参数空间让action representation的learning dynamics不被vision backbone的pretraining拽住，复杂多阶段任务里更灵活。

## 最有说服力的实验

**Last-Layer Baseline**：所有层的action tokens都cross-attend到**只来自倒数第二层**的vision feature（模仿OpenVLA-OFT做法）。LIBERO-10从96.8%直接崩到74.6%。这是核心ablation——证明gain不是参数量带来的，是hierarchical access带来的。

**Layer Skipping**：把action tokens从第 $l$ 层直接送decoder，相当于只用前 $l$ 层vision feature。发现即使只用第1层，success rate仍 >85%，training time还减少5-10x。说明ViT浅层feature已经足够informative，深层是refinement不是qualitative change。这对real deployment的speed-accuracy trade-off很有价值。

**Attention Heatmap**：SigLIP-based VAT是"focus-then-disperse"——浅层均匀分布抓contour，中层聚焦到task-relevant object，深层扩散到global。DINOv2-based VAT是attention sink——信息累积到background tokens上。这暴露了不同ViT backbone的内部trajectory有qualitatively不同的特性，VAT把这些都暴露给policy用，比只取final layer更能利用backbone的特异性。

## 跟其他工作的关系

VLM领域早有multi-layer fusion工作，分两派：

**External fusion**：vision feature送进LLM前先融合多层。比如Dense Connector直接concat多层feature。问题：vision token sequence变长，compute变贵。

**Internal fusion**：在LLM内部不同层注入vision feature。比如DeepStack、Qwen3-VL。VAT跟这个精神一致，但不需要heuristic选哪几层——直接全用。这绕开了ablation search的overhead。

还有一类native multimodal model（EVE、Fuyu），干脆不要ViT，patch直接送进unified transformer。VAT借鉴了"concurrent layer-wise refinement"的philosophy，但保留pretrained ViT——不放弃foundation model的powerful representation，只extend不replace。

## 实验结果

LIBERO四个sub-benchmark平均98.15%，比OpenVLA-OFT高1.05个点，比π0高3.95个点。最大提升在LIBERO-10 long-horizon任务上比π0高11.6个点。RoboTwin上40.66%，比ACT高10.9点比Diffusion Policy高12.6点，但比3B参数的π0低5.76点——backbone小一半还能竞争已经不错。

## 一句话我的看法

这篇paper的insight其实很朴素：**pretrained model的中间层是未被开发的information commons**。我们一直把foundation model当"encoder + head"，但它其实是deep information processing pipeline，每层都observable。下游任务应该能query pipeline任意位置，不只是接收final output。

这个principle不只对robot policy有用。Dense prediction、spatial reasoning QA、world model——任何需要fine-grained perception的downstream task都可能受益。某种意义上U-Net的skip connection就是这个思想的早期形态，VAT把它移植到了transformer架构上。

如果让我follow-up，我会做layer-wise gating让policy自己学哪几层最informative，或者搞multi-backbone VAT让SigLIP+DINOv2的trajectory互补。Diffusion + VAT也值得重新design——paper里diffusion loss略低于L1 loss，可能是timestep conditioning设计得不够精细。

核心reference就这些，GitHub在 https://github.com/sellerbubble/VAT 。

---

# VAT: Vision Action Transformer 深度讲解

Hey Andrej, 这篇paper确实切中了一个我长期觉得被忽视的点。下面我从intuition到technical一层层拆给你看。

## 1. The Core Intuition: "Representation Trajectory" 而不是 "Final Embedding"

先建立mental model。一个ViT有N层，每层输出一个representation $x^{(l)} \in \mathbb{R}^{P \times D}$，其中P是patch数，D是hidden dim。整个forward过程产生一个trajectory：

$$\{x^{(0)}, x^{(1)}, x^{(2)}, \ldots, x^{(L)}\}$$

主流robot learning pipeline只取 $x^{(L)}$，把它喂给一个policy head。这相当于只看trajectory的终点。

**问题在哪？** SigLIP的 $x^{(L)}$ 经过contrastive loss显式优化，被push到与text embedding对齐的semantic manifold上——这个manifold是为zero-shot classification服务的，pixel-level geometry被压缩掉了。DINOv2的 $x^{(L)}$ 经过self-distillation优化，对dense prediction友好，但paper指出它仍然会丢弃某些local low-level information。

而中间层 $x^{(1)}, x^{(2)}, \ldots, x^{(L-1)}$ 没有被显式loss约束，它们保留了大量"未压缩"的visual evidence——object contour、spatial layout、texture gradient等等。这些恰恰是robot manipulation需要的fine-grained几何信息。

VAT的claim：**既然trajectory的每一层都携带独特信息，policy就该沿着整条trajectory progressive地吸收，而不是只在终点处接一个static summary**。

这让我联想到你之前在micrograd/nanoGPT里强调的："中间层的hidden state是模型思考过程的空间记录"。VAT本质上是把这个insight用到action generation上。

## 2. Architecture Walkthrough

### 2.1 Token Sequence构造

输入有两路token序列：
- **Vision tokens** $x_{\text{vision}} \in \mathbb{R}^{P \times D_v}$: 来自ViT的patch embedding，P个patch，每个D_v维
- **Action tokens** $x_{\text{action}} \in \mathbb{R}^{(K \cdot L_a + E) \times D_a}$: K是chunk size (K=8)，$L_a$是每个action的token数（默认7，对应6-DoF + gripper），E是extra tokens（diffusion timestep embedding, proprioception token）

注意action tokens的维度 $D_a$ 可以与 $D_v$ 不同（VAT-Small里 $D_a = D_v / 4$）。这意味着action module是独立的parameter space。

### 2.2 单层计算详解

每层有两个并行module：

**Vision Module (Eq 1-2)** — 标准ViT block，参数来自pretrained SigLIP/DINOv2：

$$x'_{\text{vision}} = x_{\text{vision}} + \text{Attention}(\text{LayerNorm}_1(x_{\text{vision}}))$$

$$x_{\text{vision\_out}} = x'_{\text{vision}} + \text{MLP}(\text{LayerNorm}_2(x'_{\text{vision}}))$$

这里 $x_{\text{vision}}$ 是上一层的vision token输出。Attention是标准的self-attention，Q=K=V都来自vision tokens。

**FiLM Conditioning (Eq 3-6)** — 在action module之前，把task info注入：

$$t_{\text{embed}} = \text{TaskEmbeddingLayer}(\text{task\_id})$$

$$\Theta_{\text{film}} = \text{FiLMModulator}(t_{\text{embed}})$$

$$\gamma, \beta = \text{Split}(\Theta_{\text{film}}, \text{dim}=2)$$

$$x_{\text{action}} = x_{\text{action}} \odot (\gamma + 1) + \beta$$

变量含义：
- $t_{\text{embed}} \in \mathbb{R}^{D_t}$: task id查表得到的task embedding
- $\Theta_{\text{film}} \in \mathbb{R}^{2D_a}$: FiLM parameter vector
- $\gamma, \beta \in \mathbb{R}^{D_a}$: scale和shift，沿feature dimension
- $\odot$: element-wise product
- $(\gamma + 1)$: 这里+1是为了让γ=0时是identity，类似于residual initialization

这个机制来自Perez et al. 2018的FiLM paper (https://arxiv.org/abs/1709.07871)，原本用于visual reasoning。这里把task condition作为affine transform的source，让每个task shift/scale action embedding的manifold。

**Action Module (Eq 7-8)** — 关键创新，cross-attention到vision tokens：

$$x'_{\text{action}} = x_{\text{action}} + \text{CrossAttention}(\text{LN}_3(x_{\text{action}}), \text{LN}_1(x_{\text{vision}}))$$

$$x_{\text{action\_out}} = x'_{\text{action}} + \text{MLP}_{\text{action}}(\text{LN}_4(x'_{\text{action}}))$$

这里关键细节：CrossAttention(Q, K, V)中
- Query: $\text{LN}_3(x_{\text{action}})$ — action tokens做query
- Key, Value: $\text{LN}_1(x_{\text{vision}})$ — vision tokens做K和V

**重要**: 论文里写 "$x_{\text{vision}}$ in equation 1 and equation 7 refers to the vision tokens from adjacent lower layer"。意思是action tokens在第 $l$ 层cross-attend到第 $l-1$ 层的vision tokens。这是一个"延迟一层"的设计——可能为了让vision信息先经过一层self-attention refinement再被action query，也可能只是实现上的简化。

Cross-attention公式展开：

$$\text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{D_a}}\right) V$$

Attention matrix shape: $[K \cdot L_a + E, P]$。每个action token对每个vision patch有一个attention score。后面heatmap可视化就是把这个matrix reshape回spatial维度。

### 2.3 Extra Tokens的trick

Diffusion variant需要timestep $t$。Paper的做法：把timestep embedding作为一个extra token拼接到第一层的 $x_{\text{action}}$。但cross-attention的输出只保留query的长度，所以timestep token会消失。

他们的解决方案：**也在第一层的 $x_{\text{action}}$ 里再拼接一次timestep token**。这样经过FiLM和cross-attention之后，action tokens里仍然隐含timestep信息。这个trick读起来有些awkward，但实现上work。Proprioception token同样处理。

## 3. 实验数据深度解析

### 3.1 主结果 Table 1

| Model | Spatial | Object | Goal | 10 | Avg |
|---|---|---|---|---|---|
| Diffusion Policy (scratch) | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| π0 (fine-tuned) | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **VAT** | **98.8** | **99.4** | **97.6** | **96.8** | **98.15** |

关键observation：VAT在LIBERO-10 (long-horizon) 上比π0高11.6个点（85.2→96.8）。Long-horizon任务最考验对scene的fine-grained理解——这恰好印证了"中间层信息对复杂多阶段任务至关重要"的hypothesis。

### 3.2 Layer Skipping 实验 (Figure 2)

这是最有启发性的实验。把action tokens从第 $l$ 层直接送进decoder，相当于只用了前 $l$ 层的vision feature。

发现：
- 用深层（接近L=27）作为final layer：performance最好
- 用很浅的层（甚至第1层）：success rate仍 >85%，但training time减少5-10x

**Intuition**: 这说明ViT的浅层feature已经包含了足够的semantic + geometry信息让policy工作。深层带来的增益是refinement，不是qualitative change。这呼应了CLIP-like模型早期layer就形成object-centric representation的发现 (https://arxiv.org/abs/2304.06147, Probing Vision Transformer models)。

这对deployment意义重大——可以用early-exit做speed-accuracy trade-off。

### 3.3 Last-Layer Baseline Ablation (Table 2)

对照：把所有层的action tokens都cross-attend到**只来自倒数第二层**的vision feature（模仿OpenVLA-OFT的做法）。结果：

| Variant | Spatial | Object | Goal | 10 | Avg |
|---|---|---|---|---|---|
| VAT | 98.8 | 99.4 | 97.6 | 96.8 | 98.15 |
| Last-Layer Baseline | 99.2 | 94.2 | 98.2 | 74.6 | 91.55 |

LIBERO-10从96.8崩到74.6（-22.2点）。这是**最关键的ablation**——证明performance gain不是来自更多参数或更多action token capacity，而是真正来自hierarchical visual feature access。

### 3.4 Attention Heatmap (Figure 3, 4)

SigLIP-based VAT的pattern："focus-then-disperse"
- 浅层：attention均匀分布，捕捉object contour
- 中层：聚焦到task-relevant object
- 深层：扩散到global view

DINOv2-based VAT：注意力"sink"到background tokens。这是ViT里著名的attention sink现象 (https://arxiv.org/abs/2304.02815, Vision Transformers Need Registers)。DINOv2在最后几层会把信息累积到某些无semantic意义的register-like token上。

这暗示一个更深的问题：**不同ViT backbone的内部trajectory有qualitatively different properties**。VAT把这些都暴露出来给policy用，比单纯取final layer更能利用backbone的特异性。

### 3.5 FiLM vs Task Embedding (Table 3)

| Variant | Avg |
|---|---|
| VAT (FiLM) | 98.15 |
| No FiLM | 70.35 (Goal: 8.4!) |
| Task Embedding | 97.05 |

No FiLM在LIBERO-Goal上崩到8.4%——因为Goal任务需要根据goal object不同执行不同action sequence，没有task info完全无法disambiguate。但即便用简单的additive task embedding也能达到97.05，说明**主要performance来自hierarchical architecture，FiLM只是锦上添花**。

### 3.6 Architecture Variants (Table 5)

| Variant | Params | LIBERO-10 | Avg |
|---|---|---|---|
| VAT (separate action module) | 1.3B | 96.8 | 98.15 |
| VAT-Small (D_a = D_v/4) | 490M | 93.8 | 96.7 |
| VAT-ViT (shared weights like CLS) | 430M | 92.4 | 97.05 |

VAT-ViT这个变体最有意思——把action token直接塞进ViT序列里，用**shared** self-attention处理，就是CLS token的扩展版。结果仍有97.05%。这证明核心insight成立：hierarchical representation access本身就work，independent parameter space是marginal gain。

## 4. 与相关工作的positioning

### 4.1 External vs Internal Multi-layer Fusion (VLM领域)

VLM领域已有大量multi-layer fusion工作：

- **External fusion**: 在vision feature送入LLM前融合多层。例如Dense Connector (Yao et al. 2024, https://arxiv.org/abs/2405.13800) 直接concat多层feature。问题：vision token sequence变长，compute变贵。
- **Internal fusion**: 在LLM内部不同层注入vision feature。例如DeepStack (Meng et al. 2024, https://arxiv.org/abs/2406.04334)，Qwen3-VL (https://qwen.com/blog/qwen3-vl)。

VAT与internal fusion精神一致，但有个根本不同：**VLM的internal fusion仍需要heuristic选哪几层注入，VAT直接用全部层**。这绕开了ablation search的overhead。

### 4.2 Native Multimodal Models

EVE (Diao et al. 2024, https://arxiv.org/abs/2406.11832), Fuyu这类no-ViT架构把patch embedding直接送进unified transformer。Concurrent processing of vision + language。

VAT借鉴了这个"concurrent layer-wise refinement"的philosophy，但**保留pretrained ViT**——避免从头训练vision参数的高昂成本。这是关键的engineering decision：站在巨人肩膀上，只extend不replace。

### 4.3 与CLS Token的analogy

Action tokens与CLS token结构同构：都是designated aggregation agent，都直接接supervision。但关键差异：CLS token共享backbone参数，会被training dynamics拉向image classification的目标；action tokens有独立参数空间，专门学习action-relevant feature extraction。Table 5的VAT-ViT vs VAT对比验证了dedicated parameter space的value，尤其在long-horizon任务上。

## 5. 思考与局限

### 5.1 为什么cross-attention而不是把action token直接塞进ViT sequence？

VAT-ViT实验表明后者也能work (97.05%)。但separate action module有两个优势：
1. Action representation的dimension可以独立选择（VAT-Small减小到D_v/4）
2. Action module的训练不会污染vision module的pretrained representation

第2点对continual learning和multi-task可能很重要。如果未来要online adapt action policy，冻结vision module只训action module会更stable。

### 5.2 局限

1. **LIBERO偏向closed-set manipulation**。RoboTwin结果（40.66% vs π0的46.42%）说明VAT在更复杂的bimanual任务上还有差距——可能因为1.3B backbone相比3B的π0表达力不足。
2. **Cross-attention的compute cost**。每层都做 $[K \cdot L_a, P]$ cross-attention，深度L=27时累计cost可观。Layer skipping实验提示early-exit是practical mitigation。
3. **Pre-trained ViT的inductive bias**。如果未来robot需要的视频理解能力与SigLIP/DINOv2的pretraining objective差异大，中间层也不一定有相关信息。VAT依然受限于backbone ceiling。
4. **Delay-one cross-attention**。Eq 7注释说action tokens在第 $l$ 层attend到第 $l-1$ 层vision。这个设计选择paper没有ablation——可能是实现细节，可能是为了稳定性。如果action tokens能attend到当前层的vision tokens（after this layer's self-attention），信息流更紧凑。

### 5.3 更广的联想

这个"layer-wise representation trajectory"的视角其实不只适用于robot policy。任何需要fine-grained perception的downstream task都可能受益：

- **Dense prediction** (depth, segmentation): 已经有DeepStack在做
- **Visual question answering with spatial reasoning**: 中间层的几何信息可能解答"which object is to the left of X"
- **World models**: latent dynamics prediction需要保留spatial detail
- **Diffusion U-Net的skip connection**: 某种意义上U-Net的multi-scale feature已经是这类思想的实现，VAT把这个principle移植到transformer

更远一点——这暗示一个general design principle：**pretrained model的中间层是未被开发的"information commons"**。我们一直把foundation model当作"encoder + head"，但其实它是一个deep information processing pipeline，每一层都是observable。下游任务应该被empowered去query这个pipeline的任意位置，而不是只能接收final output。

### 5.4 一些可能的follow-up方向

1. **Layer-wise gating mechanism**: 学习每一层cross-attention的weight，让policy自己决定哪几层最informative
2. **Diffusion + VAT**: paper里diffusion loss variant (96.7%)略低于L1 (98.15%)，但理论上diffusion policy的多模态表达力应该更强。可能需要更精心的timestep conditioning设计
3. **Cross-attention的sparse variant**: 不是每个action token都要attend到所有vision patches，可以引入locality bias或deformable attention
4. **Multi-backbone VAT**: 同时用SigLIP + DINOv2，让两条trajectory的feature互补。这会回到external fusion的complexity，但hierarchical access可能使complexity可控

## 6. 关键参考资料

- VAT GitHub: https://github.com/sellerbubble/VAT
- LIBERO benchmark: https://libero-project.github.io/
- SigLIP: https://arxiv.org/abs/2303.15343
- DINOv2: https://arxiv.org/abs/2304.07193
- FiLM: https://arxiv.org/abs/1709.07871
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- π0: https://arxiv.org/abs/2410.24164
- DeepStack: https://arxiv.org/abs/2406.04334
- Dense Connector: https://arxiv.org/abs/2405.13800
- EVE (encoder-free VLM): https://arxiv.org/abs/2406.11832
- Vision Transformers Need Registers (attention sink): https://arxiv.org/abs/2304.02815
- Qwen3-VL: https://qwen.com/blog/qwen3-vl
- ACT: https://arxiv.org/abs/2304.13705
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- RoboTwin: https://arxiv.org/abs/2506.18088

---

总结一句给你的intuition：**VAT本质上是把policy head从ViT的"出口"移到ViT的"内部"，让action representation沿着ViT的representation trajectory逐层refine，而不是一次性接收一个被压缩过的final summary**。这个architectural shift让pretrained vision backbone的中间层information从"side effect of training"变成"first-class resource for downstream tasks"。

我觉得这个方向的潜力还远没被挖完——尤其是如果结合你之前对training dynamics和representation learning的兴趣，layer-wise probing + VAT-style aggregation会是一个很自然的research direction。
