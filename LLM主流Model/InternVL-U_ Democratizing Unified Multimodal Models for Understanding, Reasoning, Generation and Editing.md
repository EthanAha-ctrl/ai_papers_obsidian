---
source_pdf: InternVL-U_ Democratizing Unified Multimodal Models for Understanding,
  Reasoning, Generation and Editing.pdf
paper_sha256: b7f4bac080886bb877ad815df04bf76e8ad8486ec4f988dd1d2cd717658fb2c6
processed_at: '2026-08-05T10:15:57-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 InternVL-U

好，我把这篇paper拆成几个你觉得有意思的点讲讲。

## 问题到底出在哪

现在想做一个"全能模型"，既能看图理解、又能画图、还能编辑图。但问题是，**理解**和**生成**这两件事，它们要的东西根本不一样。

理解一张图，你要的是high-level semantic——"这图里有只猫，猫在桌上"。但生成一张图，你要的是low-level pixel——"猫毛的纹理长什么样，光影怎么打"。

这两类training data的distribution也完全不同。生成模型吃的data都是portraits、landscapes这种texture rich但semantically sparse的图。理解模型吃的data都是GUI截图、infographic、OCR文档这种texture简单但semantic dense的图。

你硬要把这两个东西塞进一个模型，就会出现一个encoder既想抓high-level abstraction又想保low-level pixel detail，两头不讨好。

## 别人怎么搞的，为什么不行

两条路：

**Fully-native**：从零开始训一个unified model，比如Chameleon、Emu3把图像也token化当autoregressive来predict。问题是你放弃了已有SOTA MLLM（像InternVL3.5这种理解能力已经很强的），等于把理解能力重新发明一遍，cost巨大。而且把图像discretize成token去做next-token prediction有quantization bottleneck，spatial建模不直接。

**Fully-ensemble**：拿一个pretrained MLLM，外挂一个pretrained image generator当head。但要么generator做得很大（Qwen-Image 20B，deployment贵），要么做得很小但需要复杂的conditioning pipeline（SD3用两个text encoder），这俩东西的hidden state空间很难对齐。

## InternVL-U的key insight

三个原则，但我觉得核心就一句话：**不要假装所有modality都一样，该统一的统一，该分开的分开**。

**统一context，分开generation target**。在context阶段，视觉和语言token都进同一个latent space做causal attention，让它们深度fuse。但到了generation，text就是离散AR（cross-entropy），image就是连续Flow Matching（regression velocity field）。每种modality用自己最合适的建模方式。

**理解用ViT，生成用VAE**。这俩representation完全decouple。理解的时候，ViT抓semantic feature，不需要pixel-level detail。生成的时候，VAE latent专门设计成reconstruction-friendly。你拿理解用的representation去当生成target，反而会变成累赘——就像人能看懂蒙娜丽莎但画不出来。

**backbone专注reasoning，专门加一个generation head**。InternVL3.5当backbone，加一个1.7B的MMDiT head。backbone不用管pixel-level synthesis，专心做semantic reasoning；head接收backbone的hidden state当conditioning，在VAE latent space里做Flow Matching。

## Generation Head的几个工程细节

**Dual projectors + variance normalization**。MLLM的hidden state magnitude比VAE latent大很多，且有outlier。直接concat会让training不稳定。所以在VLM branch加一层normalization把variance压到1，再project。简单但重要。

**Gated Attention**。attention output过一个sigmoid gate，element-wise控制每个feature维度的pass-through：
$$O' = O \odot \sigma(XW_g)$$
这是MMDiT里第一次加gating。intuition是高分辨率长context下attention容易"塌"到某些sink token上，gate让模型学会主动抑制一些维度，类似high-pass filter。参数开销很小，但expressivity更好。

**Resolution Interpolation for RoPE**。低分辨率预训练时，不缩小position index range，而是增大stride。比如最终目标1024px，512px训练时用整个1024的range但stride加倍。这样模型一开始就学到global spatial layout，升到高分辨率不会有tiling artifact。这招挺聪明的，相当于让RoPE的频率basis提前adapt大坐标空间。

## Training的三阶段策略

**Stage 1**：冻backbone，只训generation head。从512px起（跳过256px），同时喂T2I和image editing数据。逼head同时attend to text instruction和visual context。

**Stage 2**：引入variable resolution（512-1024px，aspect ratio 0.5-2.0）。image editing任务还要把condition image的VAE latent显式注入head，保证pixel alignment。

**Stage 3**：全模型unfreeze，end-to-end训。加入CoT reasoning data，让模型先做textual reasoning再visual execution。

## 数据工程才是真正的大招

paper里method section其实不长，data section才是核心。他们搭了好几个domain-specific的数据合成pipeline。

**SVG-based physics editing**：拿physics textbook的图，用Gemini-3-Flash转成SVG code，通过manipulate SVG来生成editing pair。成本从$0.16/sample降到$0.03/sample。因为SVG是structured representation，所以生成的ground truth质量有保证，不会像直接用image editing model那样出现质量参差。

**Computer Science editing**：用matplotlib和Graphviz直接render tree/graph/FSM。定义15类任务（BST插入、K-hop neighborhood、cycle detection、bipartite coloring等）。用fixed anchor points保证同一node在不同图里position一致。occlusion check剔除重叠样本。这等于把"图像编辑"变成了"程序化合成"，ground truth绝对正确。

**Solid Geometry**：GeoGebra + matplotlib，5类任务（旋转体、平面对称、点对称、平移、投影）。

**Spatial Rotation**：Objaverse 3D objects + 背景合成。两种策略：Object-First先确定object再合成背景，Background-First先用Flux.1 Kontext做object removal拿干净背景再paste新角度的object。

## Reasoning-centric CoT：最关键的trick

这个我觉得是paper最有意思的贡献。

用户给指令通常很vague："生成一个表达开心的meme"。模型直接拿这种指令去生成，结果往往不对。因为指令缺了具体的scene composition、emotional stance、typography约束。

他们的做法：在raw instruction和visual supervision之间插一个reasoning module，自动把abstract instruction展开成structured specification——refined objectives、decomposed sub-tasks、verifiable constraints、ordered operations。

比如"中秋主题图"，CoT会展开成"月饼、满月、桂花、团圆意象"等具体visual elements。"banana after one week"会展开成"brown spots on peel"。

训练时，数据组织成（abstract instruction, reasoning trace, execution target）三元组。模型学会先reasoning再execute。

效果在RISEBench上非常夸张：overall score从3.6直接飙到9.4。这个benchmark测的就是需要logical deduction的编辑任务，比如"画明天日历长什么样"、"把88插入BST并用红框标出"。没有CoT，模型完全搞不定；加了CoT，大幅超过BAGEL（6.1）和Qwen-Image-Edit（8.9）。

WISE上CoT也让score从0.46涨到0.58。GenExam从20.8涨到22.9。

intuition是：**abstract intent到visual execution之间有巨大gap**，CoT相当于一个"interpreter"把gap填上。模型先在language space里把逻辑理清楚，再到visual space执行，比直接end-to-end要可靠得多。

## 结果到底怎么样

4B参数的InternVL-U：
- Understanding基本保持InternVL3.5水平，MMMU 54.7接近BAGEL 7B+7B的55.3
- GenEval 0.85，在unified model里最高
- LongText-Bench ZH 0.860，BAGEL只有0.310，差距巨大
- TextEdit F1 0.71，match Nano Banana Pro，Ovis-U1只有0.35
- RISEBench加CoT后9.4，超过专门做editing的Qwen-Image-Edit（8.9）

但也要看gap：
- TIIF overall 74.9，Qwen-Image 86.1， specialized generation model还是更强
- GenExam 22.9，GPT-Image-1.5是82.3，差4倍
- RISEBench 9.4，Nano Banana Pro是47.2，差5倍

4B model跟大model比能有这种表现已经不错，但要说"democratize"还差点意思，gap还是明显的。

## 我的几个观察

1. **Decoupled representation是正确的方向**。ViT和VAE本来就optimize不同目标，硬要统一就是自找麻烦。这篇paper用工程方式证明了这一点。

2. **CoT for generation是个被低估的方向**。大家都在做CoT for reasoning，但把CoT用在generation/editing上当"intent interpreter"，效果这么显著，说明generation model缺的不是capacity而是structured planning signal。

3. **数据工程比model architecture更重要**。paper的method其实不复杂，真正花功夫的是各种domain-specific data synthesis pipeline。SVG-based、程序化合成、anchor point约束，这些都是"用structure换correctness"的思路。

4. **Gated attention in MMDiT可能是个通用改进**。作者说这是首次，但gating在FFN里已经很常见。把它用在attention output上mitigate attention sink，值得在其他diffusion transformer上试。

5. **Resolution interpolation for RoPE**这个trick其实挺通用，任何用RoPE做多分辨率训练的模型都能借鉴。

6. **Reasoning-centric data paradigm**可能改变future UMM data construction的思路。不再追求更长的caption或更dense的annotation，而是在instruction和execution之间插入reasoning bridge。

paper链接：
- [arXiv: InternVL-U](https://arxiv.org/abs/2508.18265)
- [GitHub: OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL)
- [GenEditEvalKit](https://github.com/open-compass/GenEditEvalKit)
- [TextEdit Benchmark](https://github.com/open-compass/TextEdit)

你觉得哪个点最值得深挖？我个人觉得CoT for generation这个方向可能还有很大空间没被探索。

---

# InternVL-U: 技术深度解析

## 1. 核心问题与设计动机

这篇paper试图解决一个根本性的tension: **Unified Multimodal Models (UMMs)** 需要同时具备 high-level semantic comprehension 和 low-level pixel synthesis capability, 但这两者在representation层面存在冲突。传统做法要么 fully-native (从头训练, 如 Chameleon, Emu3), 要么 fully-ensemble (post-hoc align, 如 Qwen-Image + 外挂 generator), 两者都有缺陷。

Fully-native 的问题在于: 从头训练成本高, 且无法利用已有 SOTA MLLM (如 InternVL3.5) 的 understanding capability; Fully-ensemble 的问题在于 generator scale 要么很大 (deployment cost 高), 要么很小但需要复杂的 multi-encoder conditioning pipeline (如 Stable Diffusion 3 的双文本编码器), 难以和 MLLM hidden state 对齐。

InternVL-U 的核心 insight 是: **不要强行homogenize所有modality**, 而是基于三个设计维度做 principled decoupling。

## 2. 三大架构设计原则

### 2.1 Unified Contextual Modeling with Modality-Adaptive Generation

核心 insight: **context phase 要统一, generation target 要分模态**。

**Context phase**: 视觉和语言 token 都投影到 shared latent space, 用 causal masking 的 autoregressive (AR) 范式。这保证了 deep semantic fusion。

**Generation target**: 文本是 discrete + sequential, 适合 categorical distribution + cross-entropy; 视觉信号是 continuous + spatially correlated, 适合 Flow Matching (diffusion 的 generalization)。

这避免了 "tokenization-for-all" approach (如 Emu3) 的 quantization bottleneck。

### 2.2 Structural Efficiency via Modality-Specific Modular Design

对比 Mixture-of-Transformer (MoT) 这类 fully modality-agnostic 架构, 它们把所有 modality 当成 uniform token sequence 处理, 浪费 FLOPs。

InternVL-U 用 encoder-based MLLM initialization (pre-trained ViT) 作为 modality-specific encoding stem, 再加一个 dedicated MMDiT generation head。这样 backbone 专注 semantic reasoning, stems/heads 负责模态翻译。

### 2.3 Decoupled Visual Representations

这是我觉得最 interesting 的设计。**理解图像用的 representation 不必和 生成图像用的 representation 相同**。

- **Understanding**: 用 pre-trained ViT 提取 high-level semantic features
- **Generation target**: 用专门训练的 VAE 压缩到 reconstruction-friendly latent space

这避免了单一 encoder 在 high-level abstraction 和 low-level pixel detail 之间的 optimization trade-off。作者用人类类比: 人能理解复杂场景, 但未必能画出来。

## 3. Visual Generation Head 架构细节

这个 head 是 1.7B 参数的 MMDiT-based module, 是 paper 的核心 technical contribution。

### 3.1 Dual Projectors

Multimodal hidden states (context) 和 VAE image latents (target) 的 feature distribution 差异很大。用 independent linear projectors 映射到 conditioning space。

关键 observation: multimodal context embeddings 的 magnitude 更大, 有更明显的 outliers。所以在 VLM branch 加一层 normalization, 显式 normalize variance 到 1, 减少 scale mismatch, 提升 training stability。

### 3.2 Dual-Stream MMDiT Block with Gated Attention

这是 fully dual-stream: 两个 stream 通过 joint self-attention 交互, 但 QKVO projections 和 FFN 参数 disentangled。

**Gating Mechanism** (公式1):
$$\mathbf{O}' = \mathbf{O} \odot \sigma(\mathbf{X}\mathbf{W}_g)$$

变量含义:
- $\mathbf{O}$: attention layer 的原始 output
- $\mathbf{O}'$: gating 后的 modulated output
- $\sigma$: sigmoid 函数
- $\mathbf{X}$: attention layer 的 input
- $\mathbf{W}_g$: learnable gating projection matrix, 每个 stream disentangled
- $\odot$: element-wise (Hadamard) product

这个 gating 通过 sigmoid 产生 [0,1] 的 gate 值, element-wise 控制每个 feature 维度的 pass-through。作者 claim 这是 MMDiT 架构中首次集成 gating, 用 minimal parameter overhead 提升 expressivity, 同时 mitigate 高分辨率长 context 下的 "attention-sink" 现象。

这个思路类似 GLU (Gated Linear Units) 在 LLaMA FFN 中的应用, 但用在了 attention output 上。Intuition: 让模型学会 "抑制" attention output 中的某些维度, 类似 high-pass filter, 减少 attention sink 的 degenerate behavior。

### 3.3 Unified MSRoPE with Resolution Interpolation

MSRoPE (Multi-Scale Rotary Positional Embeddings) 用 3D encoding (temporal, height, width), 同时作用于 generative target 和 context 中的 visual token。这统一了 positional encoding, 对 image editing 这类需要精确空间 reasoning 的任务特别有利。

**Resolution Interpolation** 解决了一个实际问题: 高分辨率 fine-tuning 时直接 extrapolate position index 会导致 "tiling artifact"。

做法: 定义最大 target resolution (如 1024px) 的 position embedding range。低分辨率预训练 (512px) 时不缩小 index range, 而是增大相邻 token 间的 stride。这样模型从一开始就学 consistent global spatial representation, 缩小 resolution scaling 时的 domain gap。

Intuition: 类似于在低分辨率图上用 "稀疏采样" 的方式覆盖整个高分辨率 coordinate space, 让 RoPE 的频率 basis 提前适应大范围坐标。

## 4. Training Objective

联合优化的数学形式:

### 4.1 Autoregressive Text Generation (NTP)

$$\mathcal{L}_{\text{NTP}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t | x_{<t}, \mathbf{c})$$

变量:
- $T$: text sequence 长度
- $x_t$: 第 $t$ 个 text token
- $x_{<t}$: 前面的 tokens
- $\mathbf{c}$: multimodal context sequence
- $p_\theta$: 参数为 $\theta$ 的模型预测概率
- $\mathcal{L}_{\text{NTP}}$: negative log-likelihood, 标准 next-token prediction loss

### 4.2 Flow Matching for Image Generation

这里采用 velocity parameterization, 不是 noise prediction ($\epsilon$-prediction):

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim \mathcal{U}[0,1], \mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \mathbf{z}_1 \sim p_{\text{data}}} \left[ \| v_\theta(\mathbf{z}_t, t, \mathbf{c}) - (\mathbf{z}_1 - \mathbf{z}_0) \|^2 \right]$$

变量:
- $t$: flow time, $\mathcal{U}[0,1]$ 均匀分布
- $\mathbf{z}_0$: Gaussian noise, $\sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- $\mathbf{z}_1$: ground-truth image latent (data distribution $p_{\text{data}}$)
- $\mathbf{z}_t = t\mathbf{z}_1 + (1-t)\mathbf{z}_0$: 线性插值的中间状态
- $v_\theta(\mathbf{z}_t, t, \mathbf{c})$: 模型预测的 velocity vector, conditioned on context $\mathbf{c}$
- $(\mathbf{z}_1 - \mathbf{z}_0)$: ground-truth instantaneous velocity (沿 linear trajectory 的常数 drift)
- $\|\cdot\|^2$: L2 squared norm

Intuition: Flow Matching 学的是一个 vector field, 把 Gaussian 分布 "transport" 到 data 分布。Linear interpolation path 是 Optimal Transport 的特例, trajectory 是直线, velocity 恒定。对比 diffusion 的 $\epsilon$-prediction, velocity prediction 在 linear path 下更 direct, gradient signal 更稳定。

### 4.3 Unified Training Objective

$$\mathcal{L}_{\text{Total}} = \alpha \cdot \mathcal{L}_{\text{NTP}} + \beta \cdot \mathcal{L}_{\text{FM}}$$

$\alpha, \beta$ 是 scalar hyperparameter, 在不同 training stage 动态调整 (预训练 vs. SFT 时侧重不同 capability)。

## 5. 三阶段 Training Pipeline

| Stage | Backbone | Gen. Head | LR | Resolution | Data | Loss Weight (NTP:VP) |
|-------|----------|-----------|-----|-----------|------|----|
| Stage 1: Gen Head Pre-training | Frozen | Trainable | 3e-4 (Constant) | 512px | T2I + IT2I | 4:1 |
| Stage 2: Any-res Continued Pre-training | Frozen | Trainable | 1e-4 (Cosine) | 512-1024px | T2I + IT2I | 3:4 |
| Stage 3: Unified SFT | Unfrozen | Trainable | 1e-5 (Cosine) | 512-1024px | T2I + IT2I + Understanding | 1:1:2 |

**Stage 1**: Freeze MLLM, 只训 generation head + projectors。从 512px 起步 (跳过 256px), 同时用 T2I 和 image editing 数据, 让 head 同时 attend to text instruction 和 visual context。

**Stage 2**: 引入 variable resolution (512-1024px, aspect ratio 0.5-2.0)。对于 image editing, 显式 inject condition image 的 VAE latent 到 generation head, 保证 pixel-level alignment。

**Stage 3**: 全模型 unfrozen, end-to-end 优化。加入 CoT reasoning data, 让模型在 visual execution 前先做 textual reasoning planning。

## 6. Data Construction Pipeline

这是 paper 的另一大贡献。核心论点: generation models 训练在 natural-image corpora (texture-rich, semantically sparse), understanding models 训练在 text-rich structured data (GUIs, infographics, OCR)。这个 domain gap 是 UMM 的根本障碍。

### 6.1 General Data Pipeline

四步: Filter (aesthetic, resolution, safety, watermark) → Expansion (retrieval-based + synthesis-based) → Deduplication (p-hash) → Captioning (concise / dense / human-centric, 用 Qwen2.5-VL)。

Image editing 用 multi-agent framework: Router (Qwen2.5-VL-72B) 分配任务 → Instruction Generation Agent + Image Editing Agent (heterogeneous ensemble: Flux-Text, Nano Banana, Qwen-Image) → 三维 verification (Instruction Following, Editing Consistency, Generation Quality)。

### 6.2 Text-centric Data

三类:
1. Semantically related text on natural images (用 paired caption rendering, 考虑 mask, font, adaptive layout)
2. Text on pure-color backgrounds
3. Text editing in existing images (OCR → MLLM instruction generation → Flux-Text editing)

### 6.3 Science-centric Data

最 interesting 的部分:

**SVG-based Physics Editing**: 用 Gemini-3-Flash 把 physics 图像转为 SVG code, 通过 manipulate SVG 生成 editing pair。成本从 \$0.16/sample (Nano Banana Pro) 降到 \$0.03/sample。

**Computer Science Editing**: 基于 Python libraries (matplotlib for trees/graphs, Graphviz for FSM), 定义 15 类任务 (BST 操作, K-hop neighborhood, cycle detection, bipartite coloring 等)。用 fixed anchor points 保证 node position 跨图像一致, occlusion check 去除重叠样本。

**Solid Geometry**: GeoGebra + matplotlib, 5 类任务 (Solid of Revolution, Plane/Point Symmetry, Translation, Projection)。

### 6.4 Spatial-centric Data

- **Multi-view CAD**: 基于 ABC dataset + OCC library, 渲染 isometric/front/side/top views
- **Spatial Rotation**: Objaverse 3D objects + 4 candidate backgrounds → Bounding Box Detection + Object Consistency + Generation Quality filtering → 两种策略 (Object-First 用 Qwen-Image, Background-First 用 Flux.1 Kontext 做 object removal 再 paste)

### 6.5 Reasoning-centric Data (核心创新)

这是 paper 的 key insight: **用户指令往往 brief 且 abstract**, 缺少 attribute specification, spatial relationship, executable steps, domain constraints。

解决方案: 在 raw instruction 和 final supervision 之间插入 explicit reasoning module, 自动 derive structured, actionable specification (refined objectives, decomposed sub-tasks, verifiable constraints, ordered operations)。

四类应用:
1. **General Images**: 抽象概念 → 详细 visual description (objects, backgrounds, styles)
2. **Knowledge-infused**: "中秋" → "月饼"; "banana after one week" → "brown spots" (cultural/commonsense association)
3. **Meme Images**: short instruction → (concrete scene details, humor structure, typography constraints)
4. **Science Images**: scientific concept → intermediate reasoning steps (conceptual analysis + layout planning)

## 7. 实验结果分析

### 7.1 Understanding (Table 4)

InternVL-U (2B+1.7B) 在 MMMU 54.7, 接近 BAGEL (7B+7B) 的 55.3, 说明 unified training 没有显著 degrade understanding capability。MME-P 1607.5, OCRBench 83.9, 超过 BAGEL。

### 7.2 Text-to-Image Generation

- **GenEval** (Table 5): 0.85 overall, 在 unified models 中最高, 超过 BAGEL (0.82)
- **DPG-Bench** (Table 6): 85.18 overall
- **TIIF** (Tables 7-8): 73.9-74.9 overall, 在 advanced instruction following 上 strong
- **CVTG-2k** (Table 11): 0.623 word accuracy, 在 unified models 中 SOTA
- **LongText-Bench** (Table 12): EN 0.738, ZH 0.860, 远超 BAGEL (0.373/0.310), 解决了 unified model 的 text rendering deficiency

### 7.3 Knowledge-informed Generation

- **WISE** (Table 13): CoT 让 overall 从 0.46 → 0.58, 接近 UniWorld-V1 (7B+13B) 的 0.55
- **GenExam** (Table 14): CoT 让 overall 从 20.8 → 22.9, 远超 BAGEL (11.9)

### 7.4 Image Editing

- **ImgEdit** (Table 15): CoT 让 overall 从 3.67 → 3.82
- **GEdit-Bench** (Table 16): CoT 6.88, 超过 BAGEL (6.52) 和 Ovis-U1 (6.42)
- **RISEBench** (Table 19): **CoT 让 overall 从 3.6 → 9.4**, 巨大提升, 超过 Qwen-Image-Edit (8.9)。IR (Instruction Reasoning) 从 35.6 → 43.9, AC (Appearance Consistency) 从 52.7 → 64.4

### 7.5 TextEdit Benchmark (新提出)

Table 17-18: InternVL-U F1=0.71, 匹配 Nano Banana Pro, 远超 Ovis-U1 (0.35)。MLLM-based avg: Real 0.88, Virtual 0.83, 超过 BAGEL (0.53/0.54)。

## 8. TextEdit Benchmark 设计

新提出的 benchmark, 2148 samples, 18 sub-classes (Virtual: posters, comics, slides, GUIs; Real: objects, signage, boards, accessories, transport, watermarks, paper media)。

Evaluation 公式:

**OCR Accuracy** (公式6):
$$\text{Acc} = \max_{t \in \mathcal{T}_{gen}} S(t, t_{tgt}) \times \mathbb{P}_{fail}$$

- $\mathcal{T}_{gen}$: 生成图中 IoU > 0.5 与 target region overlap 的 detected text set
- $t_{tgt}$: ground-truth target text
- $S(\cdot, \cdot)$: normalized Levenshtein similarity (公式5)
- $\mathbb{P}_{fail}$: penalty = 0.2 if source text 仍可检测但 target 缺失, else 1.0

**MLLM Overall** (公式13-14):
$$V_{score} = w_1 s_1' + \mathbb{I}_{(s_1 \geq 4)} \cdot \sum_{i=2}^{5} w_i s_i'$$

- $s_i'$: 第 $i$ 维度的 normalized score ($s_i' = (s_i - 1)/4$, 从 [1,5] 映射到 [0,1])
- $w_i$: 权重 (默认 $w_1=0.4, w_2=0.3, w_{3,4,5}=0.1$)
- $\mathbb{I}_{(s_1 \geq 4)}$: indicator, 如果 primary text accuracy $s_1 < 4$, 其他维度 score 归零

这个 cutoff mechanism 很聪明: 文本编辑失败时, 视觉质量评估无意义, 直接归零避免 score inflation。

## 9. 关键 Intuition 总结

1. **Decoupled representation**: 理解和生成用不同 encoder (ViT vs VAE), 因为它们 optimize 不同目标 (semantic abstraction vs pixel reconstruction)

2. **Hybrid AR + Flow Matching**: text 用 discrete AR, image 用 continuous flow matching, 各自发挥 statistical property 优势

3. **Gated Attention**: 在 MMDiT 中首次引入, element-wise gate 控制 attention output, mitigate attention sink

4. **Resolution Interpolation for RoPE**: 低分辨率用全 range + 大 stride, 避免高分辨率 fine-tuning 的 tiling artifact

5. **Reasoning-centric CoT**: 关键创新, 把 abstract user intent 转成 executable plan, 在 RISEBench 上让 score 从 3.6 飙到 9.4

6. **Domain-specific data engines**: SVG-based physics editing, Python-based CS editing, GeoGebra-based geometry, 把 "生成" 变成 "可编程合成", 保证 ground truth 正确性

## 10. 与相关工作的关联

- **BAGEL** (7B+7B): 同为 unified, 但参数大 3.5×, InternVL-U 用 4B 超越它
- **Janus-Pro**: decoupled visual encoding, 但 text rendering 能力弱
- **Ovis-U1** (2.4B+1.2B): 类似 scale, 但 text rendering (CVTG 0.093) 和 editing (TextEdit F1 0.35) 远弱于 InternVL-U
- **Emu3**: fully tokenized AR approach, 有 quantization bottleneck
- **Qwen-Image** (20B): specialized generation model, InternVL-U 用 4B 接近其部分指标

## 11. 局限与未来方向

- TIIF 上 unified models 整体仍落后 specialized generation models (如 Qwen-Image 86.1 vs InternVL-U 74.9)
- GenExam 上 CoT 后 22.9, 但 GPT-Image-1.5 达到 82.3, 差距巨大
- RISEBench CoT 后 9.4, 但 Nano Banana Pro 47.2, 仍有 5× gap
- 推理时 CoT 需要 model 自身生成 reasoning, 但训练数据是合成 CoT, 可能存在 distribution mismatch

参考链接:
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [GenEditEvalKit](https://github.com/open-compass/GenEditEvalKit)
- [TextEdit Benchmark](https://github.com/open-compass/TextEdit)
- [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)
- [BAGEL Paper](https://arxiv.org/abs/2505.14683)
- [Janus-Pro](https://arxiv.org/abs/2501.17811)
- [Flow Matching Guide](https://arxiv.org/abs/2412.06264)
- [MMDiT (SD3)](https://arxiv.org/abs/2403.03206)
- [InternVL3.5](https://arxiv.org/abs/2508.18265)
- [Gated Attention Unit](https://arxiv.org/abs/2505.06708)
