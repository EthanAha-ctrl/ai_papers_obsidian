---
source_pdf: The Latent Space Foundation, Evolution.pdf
paper_sha256: 57d1ccd9534403c7d7f8f29f867851d8ba0bdbdb5a3a87c37b769a20428ef46f
processed_at: '2026-08-12T14:24:09-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说Latent Space这篇survey

## 一句话版本

**LLM现在干所有事都要用文字"说出来"，但其实大部分thinking用连续向量"想"就够了，而且更快、更省、更强。**

---

## 为什么这事有意思

先想一个简单问题：你做心算 $17 \times 23$ 的时候，你脑子里是跑一遍 "$17 \times 23 = 17 \times 20 + 17 \times 3 = 340 + 51 = 391$" 的文字呢，还是直接 "感觉" 出答案？

大概是某种介于两者之间的东西——有些step你verbalize了，有些step你是直接在某种abstract representation里搞定的。

现在的LLM是个偏执狂：**每一步thinking都必须翻译成人话**。"Let me think step by step. First, I need to..."——这些token大部分是为了语法连贯和衔接，不是真正的semantic content。你付钱让model生成1000个token的CoT，可能真正承载reasoning的semantic content只需要几十个latent vector就能encode。

这就是这篇survey讨论的整个field的核心motivation。

---

## 这field在argue什么

**Claim**: latent space应该从"hidden implementation detail"升级成"first-class computational substrate"。

翻译成人话：现在model内部有hidden states，但这些states只是中间产物，最终的interface是token。这篇survey argue的是——**让latent space本身成为reasoning、planning、perception、memory、communication的native medium**，不要强迫model把每一步都翻译回text。

**为什么text是bad medium for machine computation**:
1. **冗余**: "Let me think about this problem" 这种token对reasoning本身没有贡献，是为了linguistic coherence
2. **量化损失**: continuous state → discrete token是个有损压缩，fine-grained information会丢
3. **串行锁死**: discrete tokenization锁死了sequential pipeline，每个token都要full forward pass
4. **带宽低**: agent之间通信只能通过text，semantic信息被压缩到text bottleneck

**为什么latent space是better medium**:
1. **Compact**: 几个vector就能encode一段reasoning
2. **Parallelizable**: 可以同时explore多条路径
3. **High-fidelity**: 不经过discrete bottleneck，信息不丢
4. **Differentiable**: 可以gradient-based optimization，可以做intervention和steering

---

## Field怎么发展起来的

Paper给了一个四阶段的narrative，我觉得这个narrative本身就很有信息量：

### Stage 1: Prototype (2025年3月之前) —— "这玩意能work吗？"

关键paper是 **COCONUT**。Idea很直接：把model的last hidden state直接feed back作为下一步的input embedding，不要project回vocab再sample。这样model就在continuous space里"think"了，不需要每一步都翻译成文字。

发现一个很神奇的现象：continuous thought vector可以encode **multiple potential next steps的superposition**。就是说，discrete CoT每一步commit到一个token，失去了其他possibility；continuous space里一个vector可以同时"是"多个candidate的叠加。这有点像quantum superposition——在measurement之前，system处于多个state的叠加。

这个阶段的problem：没人能解释清楚为什么这work，什么时候比explicit CoT好，怎么evaluate。

### Stage 2: Formation (2025年4-7月) —— "为什么这work？"

理论paper出来了。**Reasoning by Superposition** 给了formal proof：continuous thought vectors确实可以encode multiple search frontiers simultaneously，这给COCONUT的empirical observation提供了理论解释。

**Looped Transformers** 证明了一个separation result：looped transformers with latent iterations能express **strictly more complex** computations than standard feedforward transformers。这是说latent reasoning不只是更efficient，而是strictly more powerful（至少理论上）。

这个阶段也开始向multimodal延伸。**Mirage** 让VLM "think visually"——把hidden states recast成latent visual tokens，interleave with text。**UniVLA** 学latent action representations from internet-scale video for robotics。

### Stage 3: Expansion (2025年8-11月) —— "还能用在哪些地方？"

Field爆炸性diversify。Latent space methods扩展到：
- **Visual reasoning**: LVR, Monet, 3DThinker, VisMem, Latent Sketchpad
- **Multi-agent communication**: C2C (KV-cache直接传), LatentMAS (shared latent working memory)
- **Embodied AI**: LAPA (latent action pretraining from video), OccVLA (3D occupancy supervision), SRPO (RL with latent world representations)
- **Memory**: MemGen (generative latent memory for agents)

这个阶段的problem：field开始fragment。不同工作用不同的architecture assumptions, optimization objectives, evaluation criteria, latent interfaces。很难比较方法，很难identify stable design principles。

### Stage 4: Outbreak (2025年12月至今) —— "怎么做得更好更系统？"

最近的paper开始显示maturity的signal：

**Architectural specialization**: 不只是把standard transformer改造一下，而是设计专门为latent computation优化的architecture。
- **Dreamer**: depth-recurrent，sequence-depth sparse attention mixture
- **LoopFormer**: elastic-depth，loop iterations随input complexity变化
- **DLCM**: 把computation从token-level移到concept-level，adaptive conceptual boundaries

**Optimization sophistication**: 
- **LED**: 用entropy variation across recurrent depth来解决post-training的exploration collapse
- **Latent Thinking Optimization**: 发现latent thoughts本身可以encode reward-relevant information，不需要external reward model

**Embodied VLA的surge**: latent action representations成为VLA的central design paradigm。
- **Motus**, **VLA-JEPA**: unified latent action world models
- **WholeBodyVLA**, **SwiftVLA**, **LoLA**: latent action representations for various VLA settings

---

## 技术上怎么categorize

Paper用一个2D taxonomy来organize所有方法：**Mechanism** (怎么build和use latent space) × **Ability** (latent space enable什么capability)。

### Mechanism的四个axis

**1. Architecture**: latent space在哪里被integrate？

- **Backbone-based**: latent computation是backbone的native mechanism。比如Huginn用shared recurrent block反复iterate，Ouro做recursive inference，Dreamer是depth-recurrent design。这些是改architecture本身。
- **Component-based**: 保持backbone不变，加plug-in module。比如加一个VAE来generate latent states，加一个MLP来project representations，加一个gating head来control何时进入latent mode。
- **Auxiliary model-based**: 用external model提供supervision signal或intermediate features。比如用teacher model的hidden states做distillation target，用3D foundation model提供spatial priors。

**2. Representation**: latent variable $\mathbf{z}$ 长什么样？

- **Internal**: 直接用backbone自己的hidden states，不加参数。比如COCONUT用last hidden state，Soft Thinking用probability-weighted embedding combination（$\mathbf{z} = \mathbf{E}^\top \alpha$，是所有token embedding的加权叠加）。
- **External**: 从frozen auxiliary encoder来。比如用pre-trained vision encoder的features，用teacher model的KV cache。
- **Learnable**: 用dedicated trainable module构造。比如学一个codebook，学一个adapter，学一个VAE的latent space。
- **Hybrid**: 先用learnable module构造，再作为external signal注入。

**3. Computation**: latent space怎么participate in信息处理？

- **Compressed**: 把verbose explicit traces压成compact latent。比如HCoT把CoT压成special token，KaVa把KV cache压成distilled version，DeltaKV存cache residual。
- **Expanded**: 在depth或width维度expand computation。Depth就是recurrent/looped，width是parallel paths，structural是新的topology。
- **Adaptive**: input-dependent分配computation。TaH的think-and-halt，PonderLM-3的per-token adaptive depth，Dreamer的depth-recurrent attention。
- **Interleaved**: heterogeneous sequence，token和latent交替。Assorted混latent和text token，Mirage在text里interleave vision latent，LatentMem在multi-agent间interleave shared memory。

**4. Optimization**: 在什么时候、怎么optimize latent space？

- **Pre-training**: 从头训练时就embed latent capacity。Looped Trans.的recurrent depth scaling，LAPA的latent action pretraining from video，LoopRPT的reinforcement pre-training。
- **Post-training**: 在pre-trained model上fine-tune。包括explicit supervision（task loss only）、implicit supervision（distillation + contrastive + reconstruction）、reinforcement learning（self-rewarding + policy gradient + preference optimization）。
- **Inference**: inference时直接manipulate latent states，model weights frozen。LatentSeek的self-reward sampling，LTPO的online policy gradient，REVIS的sparse intervention for hallucination mitigation。

### Ability的七个domain

- **Reasoning**: implicit inference, compact trace, continuous refinement, branching path, modal generalization
- **Planning**: controllable exploration, efficient search, adaptive budget, sequential decision
- **Modeling**: rich expression, self inspection, robust control, scalable computation
- **Perception**: multimodal inference, heuristic imagination, faithful grounding
- **Memory**: working retention, persistent mind, multimodal recall
- **Collaboration**: semantic fidelity, shared cognition, heterogeneous interoperability
- **Embodiment**: unsupervised grounding, implicit thinking, predictive foresight, spatial cognition, generalized transfer

---

## 几个我觉得最interesting的technical insights

### 1. Superposition是latent reasoning的核心mechanism

COCONUT发现continuous thought vectors能encode superposition of multiple potential next steps。Reasoning by Superposition给了formal proof。

这个idea的intuition：discrete CoT每一步commit到一个token，你失去了其他possibilities。Continuous space里，一个vector可以是 $\mathbf{z} = \sum_i \alpha_i \mathbf{e}_i$，其中 $\alpha_i$ 是weight，$\mathbf{e}_i$ 是不同candidate token的embedding。这相当于同时explore multiple paths。

这跟quantum mechanics的superposition有analogy，也跟BFS vs DFS的区别类似。Discrete CoT是DFS——每一步commit到一个branch。Latent reasoning可以是BFS——同时维护多个frontier的superposition。

### 2. Looped Transformers的separation result

Saunshi et al. 证明looped transformers with latent iterations能express strictly more complex computations than standard feedforward transformers。

这个result的意义：latent reasoning不只是更efficient的explicit CoT，而是strictly more powerful的computational model。至少理论上，有些computations只有looped/latent才能express，standard autoregressive做不到。

### 3. Latent space和visual generation的latent space不是一回事

这个区分很重要。Visual generation的latent space（VAE, Latent Diffusion）是reconstruction objective训练出来的，manifold是smooth、locally Euclidean的，linear interpolation有意义（两张脸之间中间还是脸）。

Language model的latent space是next-token prediction训练出来的，geometry是emergent的，没有explicit constraint。所以你不能直接把visual latent space的intuitions搬过来。

### 4. Embodiment可能是latent space的killer application

机器人control面临几个discrete tokens很难handle的问题：
- **Data scarcity**: 每个robot platform要重新collect demonstrations
- **Cross-embodiment transfer**: 不同robot的action space完全不同
- **Spatial reasoning**: 3D geometry很难用text encode

Latent action representations提供了body-agnostic abstraction layer。UniVLA从internet video学latent actions，Motus做unified latent action world model，LoLA学long horizon latent actions。这些work suggest latent space可能是通往generalist embodied intelligence的关键。

### 5. Multi-agent latent communication是被underexplored的opportunity

C2C直接传KV cache between agents，LatentMAS用shared latent working memory，Wormhole connect heterogeneous visual agents。

这些work的potential：agent之间不通过text通信，直接传continuous representations。Higher bandwidth, lower latency, no semantic loss。如果未来large-scale multi-agent systems become practical，latent communication可能是necessary而非optional。

---

## 我的overall takeaway

这篇survey给我几个intuition：

**1. 这field正在从trick变成paradigm。** COCONUT当初是个interesting idea，现在有theoretical justification (Reasoning by Superposition, Looped Transformers separation result), 有architectural specialization (Dreamer, LoopFormer, DLCM), 有systematic optimization (SofT-GRPO, LED, Latent Thinking Optimization)。这让我想起deep learning从2006年Hinton DBN到2012年AlexNet的trajectory。

**2. 核心tension是interpretability vs efficiency。** Latent space methods gain efficiency和expressiveness，但lose interpretability和evaluability。CoT的好处是人类能读，能verify。Latent reasoning是black box。未来breakthrough需要新的evaluation paradigms——也许是基于latent state probing、causal intervention、或者新的information-theoretic measures。

**3. 最promising的方向是architecture-level的latent computation。** 不是在现有transformer上加个adapter，而是设计backbone本身就为latent computation优化的architecture。Huginn的shared recurrent block, Ouro的recursive inference, Dreamer的depth-recurrent attention——这些可能是future foundation model architecture的precursors。

**4. Multimodal和embodied是latent space的natural应用场景。** Text-only reasoning用latent已经不错，但真正让latent space indispensable的是那些text根本express不好的domain——visual reasoning, spatial understanding, robot action, multi-agent coordination。这些domain里latent space不只是efficiency optimization，而是representational requirement。

**5. Theory还far behind practice。** 现在有formal expressiveness results，有complexity analysis，但还没有unified theory of latent computation。什么时候latent比explicit好？为什么好？under what conditions？这些fundamental questions还没answer。

---

## References

- Survey paper: https://github.com/YU-deep/Awesome-Latent-Space
- COCONUT: https://arxiv.org/abs/2412.06769
- Huginn: https://arxiv.org/abs/2502.05171
- Reasoning by Superposition: https://arxiv.org/abs/2505.12514
- Looped Transformers: https://openreview.net/forum?id=din0lGfZFd
- Ouro: https://arxiv.org/abs/2510.25741
- Dreamer: https://arxiv.org/abs/2601.21582
- LoopFormer: https://arxiv.org/abs/2602.11451
- DLCM: https://arxiv.org/abs/2512.24617
- UniVLA: https://arxiv.org/abs/2505.06111
- Motus: https://arxiv.org/abs/2512.13030
- LAPA: https://arxiv.org/abs/2410.11758
- C2C: https://arxiv.org/abs/2510.03215
- LatentMAS: https://arxiv.org/abs/2511.20639
- Wormhole: https://arxiv.org/abs/2602.15382
- SofT-GRPO: https://arxiv.org/abs/2511.06411
- LED: https://arxiv.org/abs/2602.01698
- Latent Thinking Optimization: https://arxiv.org/abs/2509.26314

---

# The Latent Space Survey: 给Karpathy的深度解读

## 1. Paper的定位与核心论点

这篇survey来自一个跨机构的collaboration（NUS, Fudan, Tsinghua, Zhejiang等），作者名单里能看到Shuicheng Yan、Yu-Gang Jiang这些senior PI，还有Xinlei Yu作为organizer。Paper的核心claim是：**latent space正在从"hidden implementation detail"升级为"machine-native computational substrate"**，成为下一代language-based models的原生operating medium。

GitHub repo: https://github.com/YU-deep/Awesome-Latent-Space

这个positioning本身值得unpack。传统上我们理解的LLM是：token in → embedding → transformer layers → logits → token out。Computation在hidden states里happen，但interface是discrete tokens。这篇survey argue的shift是：**让latent space本身成为reasoning、planning、perception、memory、communication、embodiment的first-class medium**，而不仅是中间产物。

---

## 2. Foundation: What is Latent Space?

### 2.1 Formal Definition

Paper给的定义比较直接。Standard autoregressive generation：

$$\mathbf{y} \sim \Phi_\theta(\cdot \mid \mathbf{x})$$

其中 $\mathbf{x} \in \mathcal{V}$ 是input token sequence，$\mathbf{y} \in \mathcal{V}$ 是output，$\Phi_\theta$ 是参数为 $\theta$ 的model。Computation internally走hidden states $\mathbf{h} \in \mathcal{H}$，但generation interface仍然是token-to-token。

Latent space methods extend这个formulation：

$$\mathbf{y} \sim \Phi_\theta(\cdot \mid \mathbf{x}, \mathbf{z})$$

其中 $\mathbf{z} \in \mathcal{H}$ 是continuous latent representation。这个 $\mathbf{z}$ 提供了额外的channel来encode那些难以用token直接express的信息：global semantics、multimodal features、intermediate reasoning states、structural constraints等。

**Intuition**: 想象一下CoT（chain-of-thought）。Standard CoT把reasoning externalize成text tokens："Let's think step by step. First, I need to..."。但这些tokens大部分是linguistic redundancy——语法、衔接词、模板化表达。真正承载reasoning的semantic content可能只需要几个latent vectors就能encode。Latent space methods就是把这些intermediate computation保持在continuous space，不project回discrete vocabulary。

### 2.2 Explicit Space vs Latent Space的对比

Paper在Section 2.2做了一个系统的对比，用四个representational properties和四个functional capabilities来区分。让我用table形式重新组织：

| Dimension | Explicit Space | Latent Space |
|-----------|---------------|--------------|
| **Representation** | Human-readable, discrete & symbolic | Machine-native, continuous & flexible |
| **Efficiency** | Linguistic redundancy, sequential decoding bottleneck | Compact, parallelizable, no mandatory conversion |
| **Fidelity** | Semantically lossy (quantization to finite vocab) | High-fidelity, preserves fine-grained info |
| **Operability** | Non-differentiable, limited token-level ops | Differentiable manifold, supports steering/intervention |
| **Expressiveness** | Constrained by vocab & grammar | Richer substrate, can encode non-linguistic info |
| **Scalability** | Linear cost with trace length | Compact vectors, parallelizable |
| **Generalization** | Surface linguistic patterns | Abstract semantic structures |
| **Evaluability** | Human-readable, evaluable | Opaque, hard to inspect |

这个对比的核心intuition是：**natural language是为人类communication设计的，不是为machine computation优化的**。当我们强迫model把每一步reasoning都externalize成text，我们引入了三类inefficiency：
1. **Linguistic redundancy**: 大部分tokens是为了grammaticality和coherence，不是semantic content
2. **Representational transformation cost**: 每步都要把continuous state project回discrete vocab
3. **Sequential decoding overhead**: discrete tokenization锁死了sequential pipeline

### 2.3 与Generative Visual Models的Latent Space对比

这个对比很重要，因为visual generation community早就用latent space了（VAE、VQ-VAE、Latent Diffusion）。但paper指出language model的latent space和visual model的latent space有fundamental difference：

| Aspect | Visual Generative Models | Language Models |
|--------|------------------------|-----------------|
| **Training Objective** | Reconstruction objective → smooth, locally Euclidean manifold | Predictive criterion (next-token) → no explicit geometry constraint |
| **Structure** | Spatiotemporal grid (patches, temporal axis) | Linguistic semantics, no spatial topology |
| **Controllability** | Architectural pathways (pose, depth, segmentation) | No such dedicated control pathways |

**Intuition**: Visual latent space是anchored到pixel statistics的，linear interpolation有意义（两张脸之间的中间脸还是脸）。Language model的latent space是anchored到next-token prediction的，geometry是emergent的，没有同样的可解释structure。这也解释了为什么直接把visual latent space的intuitions搬到language model上不work。

---

## 3. Evolution: 四个发展阶段

Paper把field的evolution分成四个阶段，用timeline figure展示：

### Stage 1: Prototype (Previous – Mar 2025)

这个阶段是"theory validation + early exploration"。

**Theory Validation** 部分的key insights：
- HCoT [122]: 完整CoT trace可以压缩成compact special-token representation via contrastive semantic alignment。这说明explicit CoT里的很多信息对model本身是redundant的。
- Zhang & Viteri [277]: 从model activations里提取steering vectors，inference时inject这些vectors能elicit CoT-like reasoning without fine-tuning。
- CoE [217]: LLMs可以通过latent embeddings做self-evaluation，不需要explicit verbal output。

**Early Exploration** 的representative works：
- **COCONUT** [58]: 这是第一个完整的continuous latent space reasoning framework。Key idea是把last hidden state feed back作为next input embedding，形成continuous thoughts的loop，bypass discrete vocabulary bottleneck。发现continuous thought vectors可以encode superposition of multiple potential next steps，enable emergent breadth-first search in latent space。
  
  Formal一点说，COCONUT的recurrence是：
  $$\mathbf{h}_{t+1} = \Phi^{back}(\mathbf{h}_t, \text{context})$$
  然后 $\mathbf{h}_{t+1}$ 直接作为下一个step的input embedding，而不是project回vocab再sample。

- **CCoT** [31]: 引入contemplation tokens，把explicit reasoning chains压缩成dense latent form。
- **Huginn** [50]: 用recurrent depth来scale test-time compute in latent space。Shared transformer block iterate variable次数，perform all reasoning implicitly。
- **SoftCoT** [243]: 第一个plug-in approach，project instance-specific soft thought tokens到frozen backbone的representation space。

这个阶段的bottleneck：缺乏systematic account of why latent reasoning works, when it outperforms explicit CoT, how to evaluate it。

### Stage 2: Formation (Apr – Jul 2025)

这个阶段是"theoretical systematization + technical formation"。

**Theory Systematization**:
- **Reasoning by Superposition** [294]: 第一个formal complexity analysis。证明continuous thought vectors作为superposition states可以同时encode多个search frontiers。这给COCONUT的empirical observation提供了rigorous explanation。

  核心idea：在discrete CoT里，每一步commit到一个token，失去了其他possibilities。在continuous space里，一个vector可以是多个candidate tokens的superposition：$\mathbf{z} = \sum_i \alpha_i \mathbf{e}_i$，其中 $\alpha_i$ 是weight，$\mathbf{e}_i$ 是token embedding。这相当于parallel explore multiple paths。

- **CoT2** [52]: quantify parallelism和embedding dimension的关系，引入continuous supervision和RL for continuous thought optimization。
- **Looped Transformers** [167]: prove looped transformers with latent iterations能express strictly more complex computations than standard transformers。这是theoretical separation result。

**Technical Formation**:
- **Assorted** [181]: mixing latent discrete tokens with text tokens，achieve shorter traces with improved accuracy。
- **CODI** [174]: self-distillation procedure，同一个model同时作为teacher和student，分别在explicit和latent space生成reasoning chains。
- **HRPO** [266], **System-1.5** [214], **CoLaR** [188]: 分别introduce latent RL methods、adaptive computation allocation、dynamic inference-time controllable compression。

这个阶段也看到multimodal的initial exploration：
- **Mirage** [251]: VLMs think visually，recast hidden states as latent visual tokens interleaved with text。
- **UniVLA** [14]: task-centric latent actions for cross-embodiment robot policies。

### Stage 3: Expansion (Aug – Nov 2025)

Field diversify into multi-modal, multi-domain ecosystem。

**Technical Maturation in LLM**:
- **MemGen** [273]: latent memory for agents，interweaving reasoning和memory，planning/procedural/working memory types emerge without explicit supervision。
- **LTPO** [258]: treat latent thought vectors as optimizable parameters with online policy gradient。
- **SofT-GRPO** [291]: Gumbel-reparameterized policy optimization解决RL应用于continuous latent reasoning的differentiability challenge。

**Visual Latent Methods**:
- **LVR** [95], **Monet** [211]: autoregressive reasoning in visual embedding space，generating latent states interleaved with text。
- **3DThinker** [28]: extend to 3D mental simulation from limited 2D views。
- **VisMem** [264]: cognitively inspired short-term和long-term latent vision memory modules。
- **Latent Sketchpad** [276]: visual scratchpads for planning。

**Latent Communication for Multi-Agent**:
- **C2C** [48]: direct semantic communication between LLMs via KV-cache projection和fusion。
- **LatentMAS** [300]: latent collaboration through shared latent working memory。

**Embodied Domain**:
- **LAPA** [256], **LAWM** [196]: self-supervised latent action pretraining from unlabeled video data through world modeling。
- **OccVLA** [118]: integrate implicit 3D occupancy supervision。
- **SRPO** [45]: RL with latent world representations for VLA training。

### Stage 4: Outbreak (Dec 2025 – Present)

Explosive acceleration，hallmarks包括architectural specialization、optimization sophistication、multi-scenario surge。

**Architectural Specialization**:
- **Dreamer** [86], **LoopFormer** [72]: depth-recurrent designs combining sequence attention with depth-wise computation和elastic looping。
- **MLRA** [121]: multi-head low-rank attention via low-rank projections。
- **DLCM** [158]: shift computation from tokens to compressed concept space with adaptive conceptual boundaries。

**Optimization Sophistication**:
- **ReLaX** [280], **Active Latent Planning** [292]: move beyond imitation-based learning toward RL-based planning in latent space。
- **LED** [189]: address post-training exploration collapse by leveraging entropy variation across recurrent depth。
- **Latent Thinking Optimization** [41]: latent thoughts themselves可以encode reward-relevant information，enable直接optimizing latent trajectories without external reward models。

**Visual Multi-step Inference**:
- **ILVR** [39], **CrystaL** [283]: visual-text interleaved reasoning，report emergence of visual latent representations during reasoning。
- **LIVR** [100], **Mull-Tokens** [160]: push visual reasoning more fully into latent space，propose modality-agnostic latent thinking。

**Multi-Agent Systems**:
- **Dery et al.** [38]: latent-space communication via K-V cache alignment with lightweight adapters。
- **L2-VMAS** [265], **Wormhole** [124]: extend to visual和heterogeneous multi-agent settings。
- **LatentMem** [47]: shared latent memory for multi-agent experience accumulation。

**Embodied VLA**:
- **Motus** [10], **VLA-JEPA** [184]: unified latent action world models integrating action generation和environment understanding。
- **Villa-X** [26], **JALA** [131]: improve expressiveness和scalability of latent action modeling。
- **CoWVLA** [249]: world-model reasoning in latent motion space。
- **WholeBodyVLA** [77], **SwiftVLA** [142], **LoLA** [213]: latent action representations成为VLA pretraining和deployment的central design。

---

## 4. Mechanism: How Does Latent Space Work?

这是paper的技术核心，用四个axis来organize：Architecture, Representation, Computation, Optimization。

### 4.1 Architecture

Architecture axis关注latent space在哪里被integrated into model。Formally：

$$\Phi = \{\Phi^{back}, \Phi^{comp}, \Phi^{aux}\}$$

其中 $\Phi^{back}$ 是backbone，$\Phi^{comp}$ 是functional component，$\Phi^{aux}$ 是auxiliary model。

#### 4.1.1 Backbone

Latent computation intrinsically embedded in primary architecture：

$$\mathbf{h}_{t+1} = \Phi^{back}(\mathbf{h}_{1:t}, \mathbf{x}, \mathbf{y}_{1:t})$$

每个subsequent output token基于updated hidden state产生，latent operation是backbone的native mechanism。

**Parameter-shared Backbone**:
- **Huginn** [50]: decoder-only，shared recurrent block reused across depth steps。Hidden dim 5280, 8 layers, 3.5B params。Key feature是test-time scaling via additional recurrent steps。
- **Looped Trans.** [167]: explicit layer-looping mechanism，looping-based regularization稳定hidden-state dynamics。5120 dim, 24 layers, 1.5B。
- **PHD-Trans.** [223]: cache management + sliding-window attention，reduce memory overhead。2048 dim, 16 layers, 1.2B。

**Iterative Backbone**:
- **Ouro** [296]: recursive inference framework，iterations作为alternative scaling axis。Entropy-regularized objective for consistent latent reasoning。2048 dim, 24/48 layers, 1.4B/2.6B。
- **LoopFormer** [72]: elastic-depth looped transformer，loop iterations不fixed，可以vary across inputs。2048 dim。
- **PonderLM2** [267]: iterative refinement via Jacobi-style parallel updates。Multi-step hidden-state evolution in parallel。2048 dim, 24 layers, 0.5B/1.4B。

**Augmented Backbone**:
- **Heima** [223]: hierarchical encoder-decoder，progressive adaptive decoding。4096 dim, 72 layers, 19B。
- **DLCM** [158]: encoder-decoder，shift computation from tokens to compressed concept space。Large Concept Model。1536 dim, 32 layers, 2.3B。
- **Dreamer** [86]: depth-recurrent，sequence-depth-sparse attention mixture。1024 dim, 16/32 layers, 1B/2B。
- **MLRA** [121]: multi-head low-rank attention，four-way tensor parallelism decoding。3072 dim, 24 layers, 2.9B。

**Intuition**: Backbone-based methods代表most intrinsic form of architecture-level latent modeling。Parameter-sharing提高efficiency通过reusing layers；iterative-refinement增强flexibility通过dynamic iterations；augmentation-based designs提供broader view of architectural shifts。

#### 4.1.2 Component

Preserve backbone architecture但augment with functional modules：

$$\mathbf{z} = \Phi^{comp}(\mathbf{h}, \mathbf{x})$$

Backbone $\Phi^{back}$ 仍principal generator，component $\Phi^{comp}$ 是plug-in operator over latent space。

**Generation Component**: Construct intermediate latent states。
- **ETD** [87]: encode-think-decode mechanism，shift part of process into latent computation。
- **Palette** [129], **iCLP** [25], **ATP-Latent** [292], **ReGuLaR** [205]: VAE-style components to modulate high-level contexts和encourage diverse exploration。
- **JEPA-Reasoner** [110]: reasoning as chain of latent predictions under JEPA framework。
- **MemGen** [273], **VisMem** [264], **LatentMem** [47]: LoRA attached to backbone to weave latent memories。

**Projection Component**: Project existing representations into different target space。
- **SoftCoT++** [244], **PLaT** [207]: linear layers project hidden states to target semantic space。
- **SpiralThinker** [155], **LiteReason** [55]: MLPs for projection。
- **LF-Steering** [250]: SAE to project activations to semantic subspace。
- **Wormhole** [124]: encoder-decoder bridge projecting visual representations to shared latent space。
- **OccVLA** [118]: project 3D spatial features via transformer adapter。

**Alignment Component**: Enforce correspondence between latent representations和external grounding signals。
- **AlignVLM** [137]: mixed alignment layers binding visual tokens to textual semantics。
- **PREGEN** [169]: align generated video embeddings to retrieval-relevant textual semantics。
- **LaDiR** [83], **LoLA** [213]: align hidden states to diffusion-compatible or transformer-based semantic spaces。
- **Interpreter** [82]: LoRA to transfer latent abilities across domains。

**Control Component**: Determine when/how model enters/exits/delegates latent modes。
- **FR-Ponder** [62]: MLP gating head predicts whether input requires extended latent pondering。
- **TaH** [49]: MLP-style decider at each layer votes on whether to enter latent deliberation。
- **MemGen** [273]: trigger via LoRA or entropy signals。
- **Kelp** [101]: MLP-based module for risk-sensitive inputs in safety scenarios。

**Storage Component**: Maintain persistent latent states across steps/turns/episodes。
- **IMM** [148], **L2-VMAS** [265]: differentiable vector library as latent memory bank。
- **G-MemLLM** [242]: gated write-read logic selectively updates memory。
- **PolarMem** [30]: graph-topology structure clusters和links visual memory across episodes。

#### 4.1.3 Auxiliary Model

External auxiliary model provides supervision signals or intermediate features：

$$\mathbf{z} = \Phi^{aux}(\mathbf{x})$$

**Supervision-oriented**: Provide training signals。
- **HCoT** [122], **LaViT** [227]: assistant models generate explicit reasoning chains，distill into host hidden states。
- **SoftCoT** [243]: project auxiliary chain to continuous latent embedding。
- **SIM-CoT** [220], **SemCoT** [61]: contrastive objective aligning host latent representations with external reference model。
- **CTRLS** [226]: small LLM synthesizes intermediate reasoning states as exploration waypoints。
- **CoLT** [293]: auxiliary model decomposes latent tool-calling states。

**Feature-oriented**: Generate and inject intermediate representations。
- **CoVT** [156]: chain of visual thought，auxiliary vision models iteratively construct intermediate visual representations。
- **3DThinker** [28]: specialized 3D foundation model provides spatially-grounded geometric priors。
- **LaRe** [135], **MM-CoT** [171], **VaLR** [73]: diffusion-architected generative models reconstruct/imagine visual states。
- **UniVLA** [14]: dedicated vision encoder generates task-specific feature sequences。
- **LCDrive** [187], **DW-VLA** [75]: world models和extra vision encoders for autonomous driving。
- **WholeBodyVLA** [77], **LatentVLA** [235]: latent action models as feature encoders for humanoid control。

### 4.2 Representation

Representation axis描述latent variable $\mathbf{z} \in \mathcal{H}$ 的form。Paper用2D taxonomy：subject（how $\mathbf{z}$ is constructed）和parameterization（fixed states vs trainable modules）。

#### 4.2.1 Internal

$\mathbf{z}$ derived exclusively from endogenous activations，no additional parameters。

$$\mathbf{z} = g(\{\mathbf{H}_l\}_{l \in S})$$

其中 $g(\cdot)$ 是parameter-free aggregation function。

**Hidden State**: 
$$\mathbf{z} = \mathbf{h}_L^T \quad \text{or} \quad \frac{1}{T}\sum_{t=1}^T \mathbf{h}_l^t$$

COCONUT [58] 建立foundational pattern，feed last hidden state back as next input。SIM-CoT [220], LatentMAS [300] adopt这个recurrent paradigm。

**Weighted Embedding**: Soft, probability-weighted combination over vocab embedding matrix：
$$\mathbf{z} = \mathbf{E}^\top \alpha, \quad \alpha = \text{softmax}(\mathbf{o})$$

其中 $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$ 是embedding matrix，$\mathbf{o} \in \mathbb{R}^{|\mathcal{V}|}$ 是pre-softmax logits。$\mathbf{z}$ constrained to convex hull of vocab embeddings。

**Intuition**: 这是superposition of candidate token embeddings。不是commit到一个token，而是keep所有possibilities的weighted combination。Gradient可以flow through这个discrete generation step。

**Cache**: Treat KV pairs as structured latent memory：
$$\mathbf{k}_l = \mathbf{H}_l \mathbf{W}_l^k, \quad \mathbf{v}_l = \mathbf{H}_l \mathbf{W}_l^v$$

其中 $\mathbf{W}_l^k, \mathbf{W}_l^v \in \mathbb{R}^{d \times d_k}$ project sequence to key/value spaces。SALS [139] exploit sparse attention patterns over KV cache。LatentMAS [300] uses KV cache as shared working memory。

#### 4.2.2 External

$\mathbf{z}$ originates from auxiliary encoder $\Phi^{aux}$，structurally independent of backbone：

$$\mathbf{z} = \Phi^{aux}(\mathbf{x}_{aux})$$

$\Phi^{aux}$ frozen during backbone training。需要alignment：$\hat{\mathbf{z}} = \psi(\mathbf{z})$，其中 $\psi: \mathbb{R}^{d_{aux}} \to \mathbb{R}^{d_{back}}$。

**Reasoning Priors**:
- **CODI** [174]: self-distillation loop，frozen teacher's hidden states作为continuous supervision targets。
- **SoftCoT** [243]: assistant model generates speculative reasoning chains，project到soft-token embeddings。
- **KaVa** [91]: distill teacher's compressed KV cache。

**Perceptual Priors**:
- **3DThinker** [28]: spatially grounded 3D tokens for geometric priors。
- **SkiLa** [197]: pre-trained sketch tokens interleave with textual reasoning。
- **VL-JEPA** [21]: pre-trained embeddings from predictive-coding model。
- **OneLatent** [133]: hidden states from strong VLM condense into single latent token。

**Embodied Priors**:
- **OccVLA** [118]: pre-trained 3D occupancy tokens作为supervisory signal。
- **LCDrive** [187]: external world model generates action和scene tokens。
- **LaRA-VLA** [5]: pre-trained visual和action tokens bridging perception和motor control。

#### 4.2.3 Learnable

$\mathbf{z}$ actively constructed by parameterized module $\Phi^{comp}$ with learnable parameters $\theta$：

$$\mathbf{z} = \Phi^{comp}(\mathbf{c}; \theta)$$

$\Phi^{comp}$ structurally coupled with backbone，optimized end-to-end。

**Compression Learning**:
- **CoLaR** [188]: aggregate consecutive reasoning tokens into compressed embeddings using variance-preserving scaling factor。
- **CoLT** [293]: condense long reasoning trajectories into continuous seed tokens。
- **LIVR** [100]: visual bottleneck during training，learn implicit spatial compressions。
- **DeltaKV** [57]: encode residual differences between successive cache states。

**Distribution Learning**:
- **CTRLS** [226]: reasoning as Markov decision process，state transitions via Dirichlet distributions。
- **MARCOS** [115]: conditional hidden Markov model with step-level latent variables，variational training。
- **UniCog** [116]: latent variable model，optimize evidence lower bound to project activations to high-dimensional sparse space。
- **LatentGuard** [179]: VAE to model latent space，manipulate semantic distributions for robust refusal。

**Alignment Learning**:
- **KVCA** [38]: globally shared latent manifold via cross-attention to translate KV caches between heterogeneous models。
- **C2C** [48]: neural MLP to project KV caches between specific models by aligning terminal-layer representations。
- **Interlat** [42]: weighted Jensen-Shannon divergence to align transmitted hidden states。

#### 4.2.4 Hybrid

Combines Learnable和External sequentially：

$$\mathbf{z} = \Phi^{comp}(\mathbf{c}; \theta)$$

$\Phi^{comp}$ architecturally disjoint from backbone，$\mathbf{z}$ deployed as External conditioning signal or supervision target。

**Traces**: Distill discrete reasoning trajectories into compact continuous vectors。
- **HCoT** [122]: compress multi-step reasoning into specialized thought token。
- **Assorted** [181]: VQ-VAE codebook compresses reasoning segments。
- **Latent-SFT** [37]: restrict latent space to column space of pre-trained vocabulary matrix。
- **EBM-CoT** [29]: Langevin dynamics calibration toward lower-energy regions。
- **ThoughtComm** [290]: sparsity-regularized autoencoder transmit latent thoughts between agents。

**Grounding**: Translate continuous sensory/control signals to structured latent tokens。
- **AURORA** [11]: VQ-VAE produce discrete visual latent codes。
- **Monet** [211]: continuous embeddings as intermediate visual thoughts via multi-stage distillation。
- **UniVLA** [14]: task-centric latent actions from heterogeneous videos。
- **Motus** [10]: pixel-level delta actions via optical flow。
- **VITA** [136]: visual-action dynamics mapped to unified codebooks。

**Augmentation**: Condense large contextual histories into compressed soft prompts。
- **DCA** [117]: latent embeddings via offline coprocessor appended to KV cache。
- **CLaRa** [59]: lengthy documents encoded to compact memory tokens。
- **DEP** [157]: sparse autoencoder isolates user-specific interaction patterns。
- **VisMem** [264], **CoMEM** [229]: dedicated visual memory tokens for long-horizon planning。

### 4.3 Computation

Computation axis captures how latent space participates in information processing。

#### 4.3.1 Compressed

Reduce volume of explicit traces：

$$\mathbf{z} = \Phi(\mathbf{h}), \quad |\mathbf{z}| \ll |\mathbf{h}|$$

**Traces Compression**: 
- **HCoT** [122], **SoftCoT** [243]: semantic alignment across abstraction levels。
- **CCoT** [31]: variable-length latent allocation。
- **CODI** [174], **CoLaR** [188]: adaptive compaction through self-distillation和dynamic token-level control。

**States Compression** (targeting KV cache):
- **KaVa** [91]: KV-cache compression as knowledge distillation。
- **SALS** [139]: training-free strategy projecting cache to low-rank principal subspace。
- **DeltaKV** [57]: store semantically compressed residuals across reasoning steps。

**Features Compression** (multimodal/embodied):
- **RoT** [216]: render intermediate states as low-resolution image patches。
- **OneLatent** [133]: single latent visual token distills image context。
- **LatentVLA** [235]: project visual inputs to compact action latents。
- **Future-VLA** [44]: future-conditioned latents incorporating anticipated states。

#### 4.3.2 Expanded

Augment computational capacity along depth or width：

$$\mathbf{h}_{t+1}^{(k)} = \Phi\left(\{\mathbf{h}_t^{(k)}\}_{k=1}^K\right), \quad t=1,\dots,T, \quad k=1,\dots,K$$

其中 $\mathbf{h}_t^{(k)}$ 是k-th latent trajectory at step t，所有K paths share initialization $\mathbf{h}^{(0)}$。

**Depth Expansion**:
- **Huginn** [50]: recurrent depth，fixed transformer block applied variable次数，decoupling parameter count from inference-time compute。
- **RD-VLA** [198]: recurrent latent refinement for long-horizon manipulation planning。
- **Loop** [167]: repeatedly applying full decoder stack induces genuinely multi-step reasoning。
- **Ouro** [296]: favorable scaling along looping dimension。
- **ETD** [87]: encode-think-decode framework，latent thought state iteratively refined。

**Width Expansion**:
- **SoftCoT++** [244]: multiple parallel reasoning paths in continuous embedding space，aggregated before decoding。
- **LatentTTS** [262]: concurrent latent tree search，prune with learned value function。
- **PCCoT** [224]: multiple latent chains in parallel，exchange information at synchronization points。
- **CoT2** [52]: parallel sampling in continuous thought space。
- **Bubbles** [113]: parallel "bubbles" whose pooled representations support zero-shot unsupervised learning。

**Structural Expansion**:
- **Latent-SFT** [37]: superposed latent chains——continuous compressions of multi-step reasoning。
- **Laser** [218]: multi-scale visual features fused into shared latent representation。
- **ColaVLA** [152]: parallel interactive high-level semantic reasoning和low-level motor planning streams。

#### 4.3.3 Adaptive

Input-conditioned allocation：

$$(T, K) = \mathcal{T}(\mathbf{h}_t; \mathbf{x})$$

Halting function $\mathcal{T}$ determines instance-specific termination。

**Depth/Width Adaptation**:
- **TaH** [49]: think-and-halt mechanism，exit early on easy tokens。
- **LWS** [146]: halting as learned policy。
- **PLaT** [207]: learn when to terminate latent token generation。
- **Dreamer** [86], **SpiralFormer** [263]: weight-tied recurrent transformers。
- **I2B-LPO** [36]: expand branching factor in high uncertainty regions。

**Semantic Adaptation**:
- **AL-CoT** [267]: token-level adaptation within semantic。
- **PonderLM-3** [97], **AdaPonderLM** [180]: per-token learned depth or halting decisions。
- **DLCM** [158]: shift adaptation from tokens to semantically coherent concepts。

**Control Adaptation**:
- **System-1.5** [214]: cognitive shortcuts bypass intermediate transformer blocks。
- **FR-Ponder** [62]: instance-adaptive activation steering。
- **RISER** [257]: composition of latent reasoning skills as steering directions。
- **PRE** [125]: selective extraction of intermediate-layer representations。

#### 4.3.4 Interleaved

Heterogeneous generation sequence alternating discrete tokens with continuous latents：

$$\mathbf{r} = [r_1, z_1, r_2, z_2, \dots, r_M, z_N], \quad r_i \in \mathcal{V}, \quad z_j \in \mathcal{H}$$

**Explicit-latent Interleaving**:
- **Assorted** [181]: replacing selected verbal CoT steps with latent activations。
- **SpiralThinker** [155]: progressively internalises explicit reasoning through spiral curriculum。
- **LiteReason** [55]: lightweight model learns mixed explicit-latent reasoning。
- **SwiReasoning** [175]: step-level switching policy via RL。
- **LT-Tuning** [123], **ThinkRouter** [241]: adaptive explicit-latent boundary。

**Modality Interleaving**:
- **AURORA** [11]: explicit perceptual reasoning steps。
- **Mirage** [251], **LVR** [95], **VisMem** [264], **Monet** [211]: interleave vision latents with text。
- **IVT-LR** [20]: couple text和vision activations through cross-attention。
- **SkiLa** [197], **LS** [276], **MM-CoT** [171]: visual latents as internal sketchpad。
- **3DThinker** [28]: spatial latents for geometric reasoning。
- **DMLR** [111], **ILVR** [39]: sparse latent interleaving preserves grounding。

**Task Interleaving**:
- **LCR-SER** [177]: compressed history buffer interleaved with inference for sequential recommendation。
- **MemGen** [273], **VisMem** [264], **FlashMem** [65]: persistent working memory through generated memory tokens。
- **CLaRa** [59]: retrieval和generation interleaved in jointly trained latent lookup。
- **LatentMem** [47], **L2-VMAS** [265]: shared latent memory for multi-agent coordination。

### 4.4 Optimization

Three stages: pre-training, post-training, inference。

#### 4.4.1 Pre-training

$$\theta^\star = \arg\min_{\theta \in \Phi} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}} \left[\mathcal{L}(\mathbf{x}, \mathbf{y}, \mathbf{z}; \Phi_\theta)\right]$$

**Autoregressive Supervision**:
- **PonderLM-2** [267]: Jacobi-iteration-based parallel training。
- **Looped Trans.** [167]: recurrent depth scaling via looped layers。
- **Ouro** [296]: entropy-regularized objective，scaled to 2.6B params。
- **PHD-Trans.** [223]: dynamic exploration of continuous states。

**Auxiliary Supervision**:
- **CoCoMix** [186]: continuous semantic concepts via cross-entropy和reconstruction losses。
- **LAPA** [255]: latent action pretraining via VQ-VAE from unannotated videos。
- **CLAP** [272]: contrastive pretraining aligning visual latent spaces with proprioceptive action spaces。
- **JALA** [131]: predictive action embeddings jointly aligned with inverse dynamics。
- **ConceptLM** [126]: latent representations natively for efficient multi-task prediction。

**Reinforcement Pre-training**:
- **LoopRPT** [190]: next-token prediction as reasoning task via noisy latent rollouts。

#### 4.4.2 Post-training

$$\theta^\star = \arg\min_{\theta \in \mathcal{W}} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}} \left[\mathcal{L}(\mathbf{x}, \mathbf{y}, \mathbf{z}; \Phi_\theta)\right] - \beta \mathbb{E}_{\mathbf{r} \sim \Phi_\theta(\cdot|\mathbf{x})} \left[R(\mathbf{x}, \mathbf{r}, \mathbf{z})\right]$$

**Explicit Supervision**:
- **LATPC** [260]: task losses for safety against jailbreak。
- **GainRouter** [288]: dynamically routes features for adaptive latent reasoning。
- **GeoSteer** [84]: steering hidden states on learned latent manifold。
- **PILOT** [289]: internalize strategic oversight into intrinsic latent guidance。

**Implicit Supervision** (distillation + contrastive + reconstruction):
- **SPOT** [32]: compress explicit reasoning trajectories to compact latent tokens。
- **SemCoT** [61]: distill ground-truth reasoning into semantically aligned implicit tokens。
- **Latent-SFT** [37]: latent tokens as vocabulary-space superpositions，KL divergence和cross-entropy。
- **LTA-Thinker** [206], **EPR-Latent** [215]: contrastive objectives。
- **LaViT** [227]: jointly predict next tokens和reconstruct visual features。
- **RoT** [216]: render CoT as images，MSE和cross-entropy。
- **BNPO** [102]: reconstruction和task losses for embodied action spaces。

**Reinforcement Learning**:
- **LaTRO** [22]: reasoning as variational sampling，self-rewarding。
- **I2B-LPO** [36]: iterative information bottleneck with self-rewards。
- **SofT-GRPO** [291]: Gumbel reparameterization for stable group relative policy optimization。
- **GANPO** [76]: adversarial regularizer to robustify preference optimization。
- **DLR** [170]: contrastive stability constraints with reward signals。
- **HRPO** [266]: learnable gate progressively integrates continuous hidden states with discrete tokens。
- **LARES** [112], **RL-Latent** [149], **SCM** [208], **ATP-Latent** [292]: KL penalties alongside reward objectives。

#### 4.4.3 Inference

$$\mathbf{z}^\star = \arg\min_{\omega \in \Omega} \mathcal{J}(\mathbf{z}; \mathbf{x}, \Phi_\theta)$$

Model weights frozen，latent states $\mathbf{z}$ directly manipulated at test time。

**Inference Scaling**:
- **LTO** [41]: continuous-space classifier as latent reward model，prune incorrect thinking patterns。
- **TGR** [298]: manifold-informed latent foresight search。
- **GTS** [209]: Gaussian Thought Sampler for context-dependent perturbation distributions。
- **LatentSeek** [98]: self-reward sampling alleviates catastrophic forgetting。
- **LatentPrompt** [17]: automatically evaluates和optimizes prompts via intrinsic self-rewards。

**Inference Tuning** (gradient-based):
- **LTPO** [258]: online policy gradient on latent thought vectors。
- **∇-Reasoner** [210]: gradient descent in continuous sample space within decoding loop。
- **LatentEvolve** [274]: dynamically samples和fine-tunes experimental memory states。

**Inference Guidance** (structural constraints):
- **REVIS** [225]: sparse interventions to mitigate object hallucination。
- **STIR** [178]: value-modulated trajectory intervention via anchor-based gating。
- **SoftCoT++** [244]: contrastive learning enforces diversity among soft representations。
- **VTI** [120]: contrastive learning at inference time mitigates hallucinations。
- **L2V-CoT** [270]: contrasts visual features at test time。
- **Control++** [228]: targeted task losses during alignment generation。

---

## 5. Ability: What Does Latent Space Enable?

Paper identifies seven capability domains。

### 5.1 Reasoning

Six abilities:
- **Implicit Inference**: reasoning-like behavior already encoded in continuous activation spaces。
- **Compact Trace**: long reasoning traces absorbed into compact latent states。
- **Continuous Refinement**: sustain, blend,和iteratively revise thought as continuous state。
- **Branching Path**: explore several candidate trajectories at once via parallel latent reasoning。
- **Modal Generalization**: latent reasoning paradigm applies across linguistic, visual, heterogeneous substrates。

### 5.2 Planning

- **Controllable Exploration**: RL-based trajectory optimization over continuous thought representations。
- **Efficient Search**: geometric smoothness和continuity exploited for latent manifold navigation。
- **Adaptive Budget**: dynamic, input-dependent resource allocation。
- **Sequential Decision**: temporal structure of user behavior/system states maps onto trajectory optimization。

### 5.3 Modeling

- **Rich Expression**: continuous thought vectors encode multiple search frontiers simultaneously。
- **Self Inspection**: latent debate, geometry visualization, polarity-aware probing。
- **Robust Control**: attack vectors exploit latent fusion；defense mechanisms provide controllable steering。
- **Scalable Computation**: looped transformers express strictly more complex computations than feedforward counterparts。

### 5.4 Perception

- **Multimodal Inference**: VLMs reason about visual content through internal latent representations。
- **Heuristic Imagination**: generate和manipulate internal visual representations（mental imagery）。
- **Faithful Grounding**: representation-level intervention improves output faithfulness，addresses hallucination。

### 5.5 Memory

- **Working Retention**: KV cache as actively managed working memory。
- **Persistent Mind**: knowledge stores persist across context resets，update selectively。
- **Multimodal Recall**: continuous encoding compresses multimodal knowledge into fixed-length embeddings。

### 5.6 Collaboration

- **Semantic Fidelity**: direct latent state transfer preserves full semantic content。
- **Shared Cognition**: shared和private thought components identifiable from observable outputs。
- **Heterogeneous Interoperability**: coordination across agents of different architectures, specializations, modalities。

### 5.7 Embodiment

- **Unsupervised Grounding**: action semantics from internet-scale video without teleoperation labels。
- **Implicit Thinking**: multi-step planning as continuous latent computation without explicit CoT。
- **Predictive Foresight**: simulate future states in latent space for training supervision和real-time guidance。
- **Spatial Cognition**: reconstruct 3D/4D geometric structure from 2D observations within policy latent space。
- **Generalized Transfer**: body-agnostic abstraction layers for cross-hardware deployment。

---

## 6. Outlook: Challenges和Future Directions

### 6.1 Challenges

**Evaluability**: Latent trajectories not directly accessible to human inspection。Difficult to determine whether intermediate computation is correct, complete, or relevant。Field lacks mature benchmarking protocols for latent-space reasoning。

**Controllability**: Fine-grained interventions can reshape model behavior，but often suffer from low controllability。Tension between continuous internal dynamics和discrete interpretable objectives。

**Interpretability**: High-dimensional, distributed, entangled representations。Difficult to explain why model reaches particular conclusion，trace information transformation, identify error sources。

### 6.2 Future Directions

**Theory**: Need principled theoretical understanding——when, why, under what constraints latent space surpasses explicit space。Foundational theory of latent representation, computation, capability gains。

**Multimodal**: Transition from text-mediated multimodality to modality-native latent computation。Latent space as shared computational workspace for language, vision, action, memory, communication。

**Downstream Task**: Latent space as internal workspace for computation in search-intensive reasoning, sequential planning, visual perception, long-horizon memory, multi-agent coordination, embodied control。

**Governable**: Develop latent space into observable和governable substrate——benchmark suites, supervision strategies, controllable latent interfaces, explainable frameworks。

---

## 7. 我的Intuition和观察

读完这篇survey，我build的intuition是：

1. **Latent space是computation的"native substrate"**: Language是为human communication设计的，不是为machine computation优化的。当我们强迫model把每一步reasoning都externalize成text，我们引入了大量redundancy和inefficiency。Latent space methods本质上是让model在它最native的representation space里compute，只在必要时候interface回human-readable tokens。

2. **这个field正在从"trick"变成"paradigm"**: 从COCONUT的initial exploration，到Reasoning by Superposition的理论justification，到现在的architectural specialization（Dreamer, LoopFormer, DLCM），field正在systematize。这让我想起deep learning从2006年Hinton的DBN到2012年AlexNet的trajectory——从"interesting trick"到"new paradigm"。

3. **Key open question是interpretability-efficiency tradeoff**: Latent space methods gain efficiency和expressiveness，但lose interpretability和evaluability。这是fundamental tension。未来的breakthrough可能需要新的evaluation paradigms——也许是基于latent state probing、causal intervention、或者新的information-theoretic measures。

4. **Embodiment可能是latent space的killer application**: 机器人control的data scarcity、cross-embodiment transfer、spatial reasoning——这些都是discrete tokens很难handle的。Latent action representations（UniVLA, Motus, LoLA）提供了body-agnostic abstraction，这可能是通往generalist embodied intelligence的关键。

5. **Multi-agent latent communication是被underexplored的opportunity**: C2C, LatentMAS, Wormhole的工作展示了latent communication的potential——higher bandwidth, lower latency, no semantic loss。但目前还停留在proof-of-concept。如果未来large-scale multi-agent systems（比如1000+ agents协作）become practical，latent communication可能是必要的。

References:
- Paper: https://arxiv.org/abs/2507.06203 (推测的arXiv link，实际需要查证)
- GitHub: https://github.com/YU-deep/Awesome-Latent-Space
- COCONUT: https://arxiv.org/abs/2412.06769
- Huginn: https://arxiv.org/abs/2502.05171
- Reasoning by Superposition: https://arxiv.org/abs/2505.12514
- Looped Transformers: https://openreview.net/forum?id=din0lGfZFd
- Ouro: https://arxiv.org/abs/2510.25741
- Dreamer: https://arxiv.org/abs/2601.21582
- LoopFormer: https://arxiv.org/abs/2602.11451
- DLCM: https://arxiv.org/abs/2512.24617
- UniVLA: https://arxiv.org/abs/2505.06111
- Motus: https://arxiv.org/abs/2512.13030
- LAPA: https://arxiv.org/abs/2410.11758
- C2C: https://arxiv.org/abs/2510.03215
- LatentMAS: https://arxiv.org/abs/2511.20639
- Wormhole: https://arxiv.org/abs/2602.15382

Karpathy，这篇survey在我看来是一个signal——latent space正在从"implementation detail"升级为"architectural principle"。如果你在思考next-generation model design，这篇paper的taxonomy提供了一个systematic的map来navigate这个rapidly evolving landscape。特别是Architecture axis的backbone-based methods（Huginn, Ouro, Dreamer, LoopFormer）和Computation axis的adaptive methods（TaH, PonderLM-3, AdaPonderLM）——这些可能是future foundation model architecture的precursors。
