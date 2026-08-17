---
source_pdf: Towards Enhanced Image Generation via Multi-Modal Chain of Thought.pdf
paper_sha256: c9d3d2fa3cbbf04b840ecf5af0fbd035737bc8949797fc6a723cb04ebacf2f77
processed_at: '2026-08-12T17:06:17-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 FoX 这篇 paper

## 一句话说清楚

**让 AI 画图的时候，别上来就画，先打草稿、再检查、再修改——就像真正的人类画家那样。**

## 这篇 paper 在解决什么痛点

你让现在的 unified model 画一张 "a blue baseball bat and a pink book"（一支蓝色棒球棒和一本粉色书），它会犯这种错误：

- 把棒球棒画成粉色，书画成蓝色（颜色串了）
- 画 "a giraffe and a computer mouse"，长颈鹿的脖子变成了鼠标线（物体纠缠）
- 让它画 "four dogs"，它给你画 3 只或者 5 只（数量错）
- 让它画 "a laptop right of a cow"，位置反了

本质原因：**现在的 unified model 画图是 "one-shot" 的**——给一句 prompt，咔嚓一下直接出图。没有任何 reasoning，没有任何 intermediate step。就像让一个画家看一眼需求就立即画完，不让他构思、不让他打草稿、不让他检查修改。

简单 prompt 还行，complex compositional prompt（涉及多个物体、属性、空间关系）就直接崩。

Reference: GenEval benchmark 就是专门测这种 compositional 场景的 https://arxiv.org/abs/2310.11525

## 两个核心创新

### 创新 1：FoXperts —— 按功能分工，不按模态分工

**先说背景**。现在的 unified model（Show-o、Janus、Transfusion、LLaMAFusion 这些）大多用 "modality-oriented" 设计：一个 visual expert 啥都干，既负责"看图理解"，又负责"生成图像"。

**问题在哪**：

- Visual understanding 是把图像"压缩"成语义信息——扔掉细节，留概念
- Visual generation 是把 noise "展开"成细节——要保留所有视觉信息

**这俩目标本质是矛盾的**。让一个 expert 同时干，就像让一个人同时负责"摘要"和"扩写"——左手拧紧右手拧松，最后俩都做不好。gradient 会打架。

**FoX 的做法**：把视觉 expert 一分为二

| Expert | 干啥 | 类比 |
|--------|------|------|
| Linguistic Expert (T) | 处理文本 | 文字秘书 |
| Semantic Vision Expert (C) | 看图理解 | "鉴赏家"——评价图画好不好 |
| Generative Vision Expert (N) | 生成图像 | "画家"——动手画图 |

鉴赏家和画家是两套技能，分开训练不冲突。但它们通过 shared multimodal attention 互相沟通——鉴赏家可以告诉画家"你画得哪里不对"。

这个设计的理论依据来自 Zhang et al. 2023 的研究：预训练 Transformer 里的 neurons 本来就会自发形成 functional modules。FoX 只是把这种 emergent behavior 显式化。

Reference: Emergent Modularity https://arxiv.org/abs/2305.18390

### 创新 2：MCoT —— 让 AI 画图走 "人类画家流程"

这是 paper 的核心。MCoT = Multimodal Chain of Thought，把画图分成 4 步：

#### Step 1: Planning（打草稿前先构思）

人类画家动笔前会想：构图怎么安排？每个物体放哪？细节是什么？

FoX 的 Planning 有两个 sub-step：

**a. Detailed caption planning**: 把简短 prompt 扩展成详细描述
- "a couch" → "a comfortable gray couch placed in a modern living room with soft lighting"
- 这步类似于 DALL-E 3 用 GPT-4 改写 prompt 的做法

**b. Layout box planning**: 给每个物体分配位置坐标
- "a vase and a broccoli" → 
  ```json
  {"a vase": "0.617, 0.482, 0.832, 0.92", "a broccoli": "0.025, 0.416, 0.486, 0.91"}
  ```
- 坐标是 normalized 的 (x1, y1, x2, y2)，范围 [0, 1]

为什么有用？因为之前的 research（DALL-E 3、PixArt-α）早就发现：**更详细的 prompt 带来更好的图像质量**。Layout box 则解决 spatial 关系问题——直接告诉模型物体该放哪。

Reference: 
- DALL-E 3 prompt engineering: https://cdn.openai.com/papers/dall-e-3.pdf
- PixArt-α: https://arxiv.org/abs/2310.00426

#### Step 2: Acting（按计划画图）

根据 dense caption + layout box，生成第一版图像。这一步用 Rectified Flow（SD3 同款技术）在 Generative Vision Expert 里跑。

注意：这一版图像大概率有缺陷。但这没关系，下一步会检查。

#### Step 3: Reflection（自我检查）

模型拿"第一版图像 + input prompt"，识别哪里画错了，输出一张 **artifact heatmap**——亮的地方表示"这里需要修改"。

paper 在 Figure 3 列了四类典型缺陷，Reflection 都能处理：

1. **Structural Incompleteness**：椅子缺一条腿
2. **Object Entanglement**：两个人融合在一起
3. **Object Redundancy**：让画 2 只狗，画了 3 只
4. **Object Distortion**：人脸扭曲

这一步本质是 visual question answering："图和 prompt 一致吗？哪里不一致？" 由 Semantic Vision Expert 负责——这就是为什么 FoXperts 要把 understanding 和 generation 拆开。**Reflection 是 understanding task，让 generation expert 干就废了**。

类比：让画家自己评价自己的画往往不客观，要请鉴赏家来点评。

#### Step 4: Correction（针对性修改）

模型拿 artifact heatmap + 之前的 planning 结果，对问题区域做 targeted inpainting（局部重绘）。不是整张图重画，只改 heatmap 标出来的地方。

类比：画家听了鉴赏家的意见，拿橡皮擦掉画错的部分，重新画那块。

## 训练怎么做：绕过数据难题

### 难点

端到端训练 MCoT 需要这种数据 tuple：
```
{input prompt, 详细 caption, layout box, 第一版"错误"图, artifact map, 最终"正确"图}
```

**最大的麻烦是 "第一版错误图"**：它必须
1. 与 prompt 大体一致（不能完全乱画）
2. 与 planning 一致
3. 与最终正确图有合理差异
4. 包含 realistic 的错误（供 reflection 学习）

这种数据几乎不可能大规模收集。LLM 里可以让模型自己生成 reasoning trace（DeepSeek-R1, OpenAI o1 的做法），但图像的 "错误" 定义太模糊，没法 end-to-end 训练。

Reference:
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Let's Verify Step by Step (o1): https://arxiv.org/abs/2305.20050

### FoX 的解法：拆成三个独立任务训练

不用端到端，把 MCoT 拆成 3 个独立的训练 task，每个 task 用容易获取的数据，按 1:1:1 比例交替训练：

**Task 1: Planning + Acting**
- 数据：`{prompt, detailed caption, layout box, image}` 四元组
- 数据来源：CC12M（100K 样本）
- Pipeline：
  - Dense caption 用 Qwen-VL 生成
  - Bounding box 用 Grounding-DINO + SAM 从 caption 里 extract object 名词短语
- 训练目标：给 prompt，输出 caption + box + image

**Task 2: Reflection**
- 数据：`(第一版图, prompt) → artifact map`
- 数据来源：RichHF-18K + 100K 手工标注（标出错误区域的 bbox，再转成 heatmap）
- 训练目标：给图和 prompt，输出 heatmap 标出错误区域

**Task 3: Correction**
- 数据：`(masked image, caption) → inpainted image`
- 数据来源：CC12M + SAM-1B（100K 样本）
- Mask 生成：follow BrushNet 方法，random mask + segmentation mask
- 训练目标：标准 inpainting 任务

**Intuition**: 这种 disentangled training 避开了 "构造错误图" 的难题。每个 sub-task 用的数据都是现成的或容易构造的。模型 inference 时把三个 capability 串起来跑 MCoT，但训练时各自独立。

Reference:
- CC12M: https://github.com/google-research-datasets/conceptual-12m
- SAM: https://arxiv.org/abs/2304.02643
- Grounding DINO: https://arxiv.org/abs/2303.05499
- BrushNet: https://arxiv.org/abs/2403.06976

## 训练 schedule

三个阶段：

**Stage I**: T2I only，256×256 分辨率，batch 4096，300M image-text pairs。训练 Linguistic + Generative Visual Expert。从 Qwen2-0.5B 初始化。

**Stage II**: T2I + I2T 混合训练（比例 8:1），512×512 分辨率，batch 1024。120M 生成数据 + 20M 理解数据。三个 expert 都训练。Semantic Vision Expert 从 Stage I 的 Generative Visual Expert 初始化（warm start）。

**Stage III**: MCoT 多任务训练，1:1:1 交替。三个 expert 都训练。

**为什么 Stage II 要加 I2T 训练**？因为 Reflection 步骤本质是 visual understanding——模型得先学会"看懂图"，才能 critique 自己的画。I2T 数据激活 Semantic Vision Expert，为 MCoT 铺路。

## 结果有多猛

### GenEval（核心 compositional benchmark）

FoX 用 **1.3B 参数** 在 GenEval 拿到 **0.77 overall score**：

| Model | Params | Overall |
|-------|--------|---------|
| SD3 | 12.7B | 0.68 |
| DALL-E 3 | - | 0.67 |
| Transfusion | 7.3B | 0.63 |
| JanusFlow | 1.3B | 0.63 |
| **FoX** | **1.3B** | **0.77** |

**1.3B 干翻了 12.7B 的 SD3**，这不是小提升，是 +0.09 的巨大差距。

子任务提升最有说服力：
- **Position**：0.60（JanusFlow 0.53）→ layout planning 直接见效
- **Attr. Binding**：0.64（JanusFlow 0.42）→ detailed caption 显式列出每个属性
- **Counting**：0.71 → reflection + correction 修复了"多画少画"的问题
- **Two Obj.**：0.86（JanusFlow 0.59）→ 多物体场景大幅提升

### MS-COCO（基础生成能力）

- FID: 7.24（DALL-E 2 是 10.39，NExT-GPT 是 10.07）
- CLIP Score: 26.8
- CIDEr: 126.5

**关键**：FoX 在 compositional 上提升的同时，基础生成能力没退化。这说明 FoXperts 的功能解耦设计避免了 trade-off。

### Image Understanding（视觉理解）

- MME-P: 1339.7（Janus 是 1338.0）
- MMBench: 73.6（Janus 是 69.4）
- VQAv2: 79.4

**关键**：拆出 Semantic Vision Expert 后，视觉理解能力反而略升——因为 disentangled 优化避免了之前 understanding 被 generation loss 拖累。

## Ablation 最有说服力的两个实验

### Ablation 1: MCoT 真的有用吗？

| Setting | Overall |
|---------|---------|
| T2I Gen. Twice（baseline 生成两次取最好的） | 0.67 |
| MCoT Planning & Acting Only | 0.73 |
| MCoT Full Process | 0.77 |

**这个 baseline 设计很 clever**：不是简单 T2I 一次，而是 T2I 生成两次取最好——这控制了 "MCoT 多用了 compute" 这个变量。结果 +0.10 的提升是真正来自 reasoning 的价值，而不是单纯多算几次。

### Ablation 2: FoXperts 的功能解耦有用吗？

| Architecture | CIDEr↑ | FID↓ |
|--------------|--------|------|
| Dense（一个 transformer 啥都干） | 116.2 | 11.3 |
| Modality-Oriented（按模态分 expert） | 121.1 | 9.56 |
| **FoX（按功能分 expert）** | **126.5** | **7.24** |

从 dense 到 modality-oriented 是 +4.9 CIDEr，从 modality-oriented 到 FoX 又 +5.4 CIDEr。**功能解耦的提升幅度比从 dense 到 MoE 还大**，说明这是独立的、重要的 design dimension。

## 为什么 MCoT 有效？三个深层原因

### 1. Compositional Generalization 的救星

直接 T2I 把 prompt → image 当一个端到端 mapping。如果训练集里 "red dog" 和 "blue cat" 多，"blue dog" 没见过，模型就不会组合。

MCoT 在 Planning 阶段把 "blue dog" 拆成 "blue" + "dog" 两个独立 attribute，每个都见过，组合起来就行。**Decompose-then-compose 天生对 compositional 友好**。

### 2. Capacity 在时间上的分配

Single-pass 生成要同时干三件事：
- 语义推理（理解 prompt）
- 空间布局（安排物体）
- 细节渲染（生成像素）

每个都吃 capacity，一起干会互相干扰。MCoT 在不同 step 分配 capacity：
- Planning 专注 semantic + spatial
- Acting 专注 rendering
- Reflection 专注 alignment check
- Correction 专注 detail fix

就像 LLM 的 CoT——每个 reasoning step 让模型专注一部分推理，避免 capacity bottleneck。

### 3. Iterative Refinement 的容错

Single-pass 错了就错了，没救。MCoT 的 Reflection + Correction 提供了"自我修正"的机会，即使 Acting 阶段画错了，还有 Reflection 能发现，Correction 能修补。

## 我的几个思考

### 1. Inference cost 是 4 倍

MCoT 4 个 step 都要 forward pass，成本是直接 T2I 的 4 倍。Table 5 的 baseline 是 "T2I Gen. Twice"（2 倍 cost），但 MCoT 是 4 倍。

**未来方向**：Adaptive MCoT——简单 prompt 直接 T2I 出图，复杂 prompt 才走完整 MCoT。类似 LLM 的 adaptive reasoning length。

### 2. Reflection 的能力上限

Reflection 本质是 "判断图像与 prompt 是否一致"。如果 Semantic Vision Expert 看不出的错误，Correction 就修不了。

这可能解释为什么 Single Obj. 能到 0.99，但 Two Obj. 只有 0.86——多物体场景的 reflection 更难，鉴赏家也会看走眼。

### 3. Layout Planning 的 robustness

Layout box 依赖 Grounding-DINO + SAM 从 caption 中 extract object。如果 Grounding-DINO 漏检了某个 object，layout 就错了，后续全错。Pipeline 的 weakest link 决定了整体下限。

### 4. RL 训练 MCoT 是个 promising 方向

paper 用 supervised multi-task training。但更 exciting 的方向是用 RL：
- Reward: 用 VLM 评价 final image 与 prompt 的 alignment
- 算法: PPO 或 GRPO
- 让模型自己学"什么时候需要 correction"，而不是固定 4 步

这就是把 DeepSeek-R1 的思路搬到图像生成。但 reward 设计和 multi-modal RL 的稳定性是大挑战。

### 5. 跟 GPT-5o 的对比

paper introduction 里提到 GPT-5（OpenAI 2025）作为 unified generative model 的代表。GPT-5o 的 native image generation + reasoning 很可能用了类似 MCoT 的思想——但 GPT-5o 是 end-to-end RL，FoX 是 disentangled supervised。

FoX 的方法更适合当前 data-scarce 的现实，但 RL 的 ceiling 可能更高。两条路殊途同归。

Reference: GPT-5o: https://openai.com/index/introducing-gpt-5o/

### 6. Scale behavior 是个 open question

paper 只展示了 1.3B 模型。MCoT 的收益是否会随 scale 衰减？通常 reasoning 的收益在小模型上更明显（大模型 single-pass 也能 implicit reasoning）。但 FoXperts 的功能解耦可能在大模型上更受益（大模型 expert 专业化更重要）。

这个需要更多实验验证。如果 FoXperts + MCoT 在 7B、13B 上仍能保持优势，那就是真正的 breakthrough。

## 一句话总结

**FoX = "功能分工的专家团队" + "人类画家式的四步工作流" + "绕过数据难题的拆解训练"。1.3B 参数打爆 12.7B 的 SD3，证明了一件事：图像生成领域也需要 reasoning，one-shot black-box mapping 的时代快过去了。**

这篇 paper 给图像生成领域指了一个新方向：把 NLP 里已经验证有效的 CoT 思想，结合图像生成的特殊性（多模态、空间布局、视觉细节），设计成 MCoT 这样的 explicit reasoning pipeline。这可能是图像生成从 "art" 到 "engineering" 的一个关键 step。

Reference (paper 主页，链接需要核对): https://arxiv.org/abs/2505.17923

---

# FoX: Towards Enhanced Image Generation via Multi-Modal Chain of Thought — 深度技术解析

## 1. Paper 的核心动机与问题定义

这篇 paper 由 Zhejiang University 与 Alibaba Group 联合完成，核心 problem statement 非常清晰：**现有的 unified generative models 在处理 complex compositional instructions 时存在根本性缺陷**。具体表现为四类 failure modes:

1. **Concept confusion** in multi-object scenarios（多物体场景中的概念混淆）
2. **Attribute errors**（如 misaligned color binding，颜色绑定错位）
3. **Spatial inconsistencies**（如 incorrect positioning，位置关系错误）
4. **Object defects**（如 incomplete structures，结构不完整）

Figure 1 中的对比非常直观：直接 T2I generation 生成 "a blue baseball bat and a pink book" 时，模型会把颜色错位；生成 "a giraffe and a computer mouse" 时，会把长颈鹿的脖子变成鼠标线。这本质上是当前 unified generative models 把图像生成当作 "one-shot" black-box mapping 来处理，缺乏显式的 reasoning 过程。

**Intuition**: 在 NLP 领域，我们早就知道 CoT (Chain of Thought) 对复杂任务至关重要。Karpathy 你自己也讨论过 LLM 的 reasoning 能力。这篇 paper 的核心 insight 是：**图像生成本质上也是一个需要 reasoning 的任务，特别是当 prompt 涉及多个物体、属性、空间关系时**。直接 text→image 的 mapping 在分布外（OOD）的 compositional case 上泛化能力差。

Reference links:
- Chain-of-Thought 原始 paper: https://arxiv.org/abs/2201.11903
- DeepSeek-R1 (end-to-end CoT): https://arxiv.org/abs/2501.12948
- Let's Verify Step by Step (OpenAI o1): https://arxiv.org/abs/2305.20050

## 2. FoXperts 架构：从 Modality-Oriented 到 Functionality-Oriented

### 2.1 设计动机：Function-Domain Conflict

这是 paper 最有洞察力的部分之一。当前主流 unified models（如 Show-o、Janus、JanusFlow、Transfusion、Chameleon、LLaMAFusion）大多采用 **modality-oriented** 的 expert 设计——即一个 visual expert 同时处理 visual understanding 和 visual generation 两个任务。

paper 指出这存在 **functional domain conflict**:

- **Visual Understanding** 的目标：将 image features 与 text 对齐，由 comprehension loss (Eq. 2 形式的 LM loss) 优化。本质上是把图像 "压缩" 成语义表征，保留与 task-relevant 的信息，丢弃细节。
- **Visual Generation** 的目标：在 latent space 中预测 noise velocity（Rectified Flow 的 Eq. 4），用于 denoising。本质上是把 noise "展开" 为 high-frequency details，需要保留与生成相关的所有视觉信息。

**Intuition**: 这两个目标的 gradient 方向在参数空间中是 conflict 的。understanding 要 "summarize"，generation 要 "elaborate"。一个共享 expert 被同时训练这两个目标时，会被迫做 trade-off，导致两个任务都 sub-optimal。

这与 Zhang et al. 2023 的 "Emergent Modularity in Pre-trained Transformers" 研究发现一致：预训练 Transformer 中 neurons 会自发地形成 functionally specialized modules。paper 引用这个工作作为 functionality-oriented 设计的理论依据。

Reference: Emergent Modularity in Pre-trained Transformers: https://arxiv.org/abs/2305.18390

### 2.2 三专家设计

FoXperts 引入三个并行 experts：

| Expert | 功能 | 初始化 | 优化目标 |
|--------|------|--------|----------|
| **Linguistic Expert** (T) | Text understanding & generation | Qwen2-0.5B | LM loss (Eq. 2) |
| **Semantic Vision Expert** (C) | Visual understanding | Stage I Generative Vision Expert | LM loss on image captions |
| **Generative Vision Expert** (N) | Visual generation | Qwen2-0.5B + 训练 | Rectified Flow loss (Eq. 4) |

**Key design**: 文本保持单 expert，因为 text understanding 和 text generation 共享相同的 next-token prediction 目标（Eq. 2）。视觉拆分为两个 expert，因为 understanding 和 generation 的目标本质不同。

**Intuition**: 这其实是 Mixture of Experts (MoE) 的一个 variant。传统 MoE 通常按 token routing（如 Switch Transformer、GShard），而这里按 **功能域** routing，是一种更结构化的 sparse design。

### 2.3 Forward Pass 数学解析

公式 (5) 是 FoXperts 单层的 forward process，非常关键：

$$\hat{\mathbf{x}}_i = W_i(Router(LN(\mathbf{x}))), \quad i \in \{T, C, N\}$$

- $\mathbf{x}$: 输入 token sequence $\mathbf{x} = \mathbf{x}_T \oplus \mathbf{x}_C \oplus \mathbf{x}_N$
  - $\mathbf{x}_T$: text tokens（Linguistic Expert 处理）
  - $\mathbf{x}_C$: clean image tokens（Semantic Vision Expert 处理）
  - $\mathbf{x}_N$: noise image tokens（Generative Vision Expert 处理）
- $LN$: Layer Normalization
- $Router$: multimodal routing module，决定每个 token 应该走哪个 expert
- $W_i$: 第 $i$ 个 expert 的 projection matrix，参数独立
- $\hat{\mathbf{x}}_i$: 路由到 expert $i$ 的 token 表征

$$\hat{\mathbf{x}}_i^{q,k,v} = W_i^{Q,K,V}(\hat{\mathbf{x}}_i), \quad i \in \{T, C, N\}$$

- $W_i^{Q,K,V}$: Query, Key, Value 的 projection matrices
- 上标 $q, k, v$ 分别对应 attention 的三要素

$$\hat{\mathbf{x}}^{rep} = \hat{\mathbf{x}}_T^{rep} \oplus \hat{\mathbf{x}}_C^{rep} \oplus \hat{\mathbf{x}}_N^{rep}, \quad rep \in \{q, k, v\}$$

- $\oplus$: concatenation 操作
- 三个 expert 的 Q, K, V representations 重新 concatenate 成完整 sequence，进入 shared multimodal attention

$$\hat{\mathbf{x}} = Attn(\hat{\mathbf{x}}^q, \hat{\mathbf{x}}^k, \hat{\mathbf{x}}^v) + \mathbf{x}$$

- $Attn$: shared multimodal attention module
- residual connection

公式 (6) 是 FFN 部分，结构类似，每个 expert 有独立的 FFN。

**Attention pattern 的关键设计**:
- **Local**: text tokens 用 causal attention（autoregressive），vision tokens (clean + noise) 用 bidirectional attention（diffusion 需要全局信息）
- **Global**: 所有 tokens 之间是 causal sequence，确保 loss 和 gradient 计算时没有 future information leakage

这个设计与 LLaMAFusion、Mixture-of-Transformers 类似，但 expert 划分依据不同。

Reference:
- Show-o: https://arxiv.org/abs/2408.12528
- Janus: https://arxiv.org/abs/2410.13848
- JanusFlow: https://arxiv.org/abs/2411.07975
- Transfusion: https://arxiv.org/abs/2408.11039
- LLaMAFusion: https://arxiv.org/abs/2412.15188
- Mixture-of-Transformers: https://arxiv.org/abs/2411.04996

### 2.4 Input Representation

Text 端用 Qwen2 tokenizer，输出 embedding sequence $\mathbf{x}_{\text{text}} \in \mathbb{R}^{L \times d}$，其中 $L$ 是 sequence length，$d$ 是 embedding dimension。

Image 端用 SD3 的 VAE (Esser et al. 2024)，并 follow Transfusion 的方法把 $2 \times 2$ patches 压缩成一个 vector，得到 $\mathbf{x}_{\text{image}} \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times d}$。每个 image token 对应原图 $16 \times 16$ pixels 的 patch。

**Intuition**: 这里采用 hybrid tokenization 策略，text 用离散 token（autoregressive），image 用连续 latent token（diffusion）。这避开了纯离散 tokenization（如 Emu3, Chameleon）在视觉细节上的损失。论文里明确说："discrete tokenization approaches do not align with the continuous nature of images and videos, thereby limiting the visual generative potential"。

Reference:
- SD3 (MMDiT): https://arxiv.org/abs/2403.05230
- Emu3 (next-token prediction for all modalities): https://arxiv.org/abs/2409.18869
- Chameleon: https://arxiv.org/abs/2405.09818

## 3. Rectified Flow 数学基础

公式 (3) 和 (4) 是 FoX 生成图像的核心数学基础：

$$\mathbf{x}_t = t\mathbf{x} + (1-t)\epsilon, \quad t \in [0, 1]$$

- $\mathbf{x} \sim \mathcal{D}$: 真实图像（latent space 中）
- $\epsilon \sim \mathcal{N}(0, I)$: 标准 Gaussian noise
- $t$: 时间步，$t \in [0, 1]$
- $\mathbf{x}_t$: 在时间 $t$ 的中间状态
- 当 $t = 0$: $\mathbf{x}_0 = \epsilon$ (纯噪声)
- 当 $t = 1$: $\mathbf{x}_1 = \mathbf{x}$ (真实数据)

这是 linear interpolation，从 noise 到 data 的直线路径，contrast to DDPM 的 stochastic differential equations。

$$\mathcal{L}_{RF} = \mathbb{E}_{t \sim U(0,1), \mathbf{x}, \epsilon, c} \left[\|(\mathbf{x} - \epsilon) - v_\theta(\mathbf{x}_t, t, c)\|_2^2\right]$$

- $v_\theta$: 速度场模型，预测从 $\mathbf{x}_t$ 到 $\mathbf{x}$ 的方向
- $c$: 条件（文本 embedding）
- $\mathbf{x} - \epsilon$: target velocity（沿直线路径的速度）
- $\| \cdot \|_2^2$: L2 norm 的平方

**Intuition**: Rectified Flow 是 optimal transport 的思想。DDPM 用 SDE 模拟扩散过程，路径是弯的；Rectified Flow 直接拉直这条路径，使 transport 更高效，sampling 需要的 step 更少。这是 SD3、Flux、Stable Diffusion 3.5 等现代生成模型的标配。

Reference:
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- Rectified Flow (Liu et al.): https://arxiv.org/abs/2209.03003
- SD3: https://arxiv.org/abs/2403.05230

## 4. MCoT: Multimodal Chain of Thought

这是 paper 的核心创新。MCoT 把图像生成过程显式分解为 4 个步骤，模仿人类艺术家的 workflow:

### 4.1 Planning

包含两个 sub-steps:

**Detailed caption planning**: 把简短的 input prompt 扩展为更详细、精确的 caption，但不扭曲原意。比如 "a couch" → "a comfortable gray couch placed in a modern living room with soft lighting"。这一步类似于 DALL-E 3 用 GPT-4 re-captioning prompt 的做法。

**Layout box planning**: 为 prompt 中的每个 object 分配合理的 bounding box 位置。例如 prompt "a vase and a broccoli" 生成:
```json
{"a vase": "0.617, 0.482, 0.832, 0.92", "a broccoli": "0.025, 0.416, 0.486, 0.91"}
```
Bounding box 是 normalized 坐标 (x1, y1, x2, y2)，范围 [0, 1]。

**Intuition**: Planning 步骤对应 CoT 中的 "decompose the problem"。Caption planning 增加 fidelity（之前的 paper 如 DALL-E 3、PixArt-α 都验证了更详细的 prompt 带来更好的图像质量），layout planning 增加 spatial accuracy。这两步是显式的 reasoning。

Reference:
- DALL-E 3 prompt engineering: https://cdn.openai.com/papers/dall-e-3.pdf
- PixArt-α: https://arxiv.org/abs/2310.00426
- GLIGEN (layout-to-image): https://arxiv.org/abs/2305.11874

### 4.2 Acting

FoX 根据 planning 阶段产生的 dense caption + layout box 生成第一版图像。这一步用 Rectified Flow 在 Generative Vision Expert 中执行。

**Intuition**: Acting 步骤是 "execute the plan"。Layout box 通过某种 spatial conditioning（可能类似 ControlNet 的方式）引导生成位置。

### 4.3 Reflection

模型拿第一次生成的图像 + input prompt 作为输入，识别其中的缺陷区域。输出是 **artifact heatmap**，confidence score 越高表示该区域越需要 correction。

paper 在 Figure 3 展示了四类典型的图像缺陷，Reflection 步骤可以处理:

1. **Structural Incompleteness**: 物体结构不完整（如椅子缺腿）
2. **Object Entanglement**: 物体之间纠缠（如两个人融合在一起）
3. **Object Redundancy**: 多余的物体（如要求 2 个狗，生成 3 个）
4. **Object Distortion**: 物体变形（如人脸扭曲）

**Intuition**: 这一步是 self-critique，类似 LLM 的 Self-Refine、Reflexion 方法。但 LLM 是文本 critique 文本，这里是 vision critique vision。模型需要从 prompt 和 generated image 中推理出 misalignment。

Reference:
- Self-Refine: https://arxiv.org/abs/2303.17651
- Reflexion: https://arxiv.org/abs/2303.11366

### 4.4 Correction

模型整合 artifact reflection map + planning rationale，进行 targeted inpainting。这是一种 conditioned image inpainting，只对 artifact heatmap 高 confidence 的区域进行修改。

**Intuition**: 这一步类似于 BrushNet 的 inpainting，但有更精细的 mask 来源（不是随机 mask，而是 reflection 出来的 artifact mask）。这种 "iterative refinement" 与 LLM 中的 "verify-then-correct" 思想一致。

Reference:
- BrushNet: https://arxiv.org/abs/2403.06976
- ControlNet: https://arxiv.org/abs/2302.05543

## 5. Multi-Task Joint Training Paradigm

这是 paper 解决数据问题的关键创新。

### 5.1 端到端训练的困难

完整 MCoT 需要 tuple: `{input prompt, detailed planning caption, layout planning boxes, first "wrong" image, artifact map, final "correct" image}`。

其中 "first wrong image" 极难构造：它必须 (1) 与 prompt 一致，(2) 与 planning 一致，(3) 与 final image 有合理的差异，(4) 包含 realistic errors 供 reflection 学习。

**Intuition**: 这就是为什么不能端到端训练。在 LLM 中，我们可以让模型自己生成 reasoning trace 然后用 RL 或 SFT 训练（如 DeepSeek-R1, OpenAI o1）。但图像生成中，"wrong image" 的定义本身就很模糊，且需要 multi-modal alignment，远比文本 trace 难构造。

### 5.2 Disentangled Three Tasks

paper 把 MCoT 训练拆分为三个独立的 task，比例 1:1:1:

**Task 1: Planning and Acting**
- 数据: `{input prompt, detailed caption, layout box, image}`
- 数据来源: CC12M, 100K samples
- Pipeline:
  - Dense caption: Qwen-VL 生成
  - Bounding box: Grounding-DINO + SAM 从 caption 中提取 object noun phrases
- Model input: prompt → output: caption + box + image

**Task 2: Reflection**
- 数据: `(first generated image, input prompt) → artifact map`
- 数据来源: RichHF-18K + 100K 手工标注的 bounding boxes（错误区域）
- 人工标注的 bbox 后续转化为 heatmap-style representation

**Task 3: Correction**
- 数据: `(masked image, caption) → inpainted image`
- 数据来源: CC12M + SAM-1B, 100K samples
- Mask 生成: follow BrushNet 方法，包括 random mask 和 segmentation-based mask

### 5.3 Training Schedule

Stage I (Pre-training):
- T2I only
- Image resolution: 256×256
- Batch size: 4096
- Data: 300M image-text pairs
- Trainable: Linguistic Expert + Generative Visual Expert
- 初始化: Qwen2-0.5B

Stage II (Pre-training continued):
- T2I + I2T (ratio 8:1)
- I2T 包括 captioning + VQA
- Image resolution: 512×512
- Batch size: 1024
- Data: 120M (generation) + 20M (understanding)
- 所有 experts 可训练
- Semantic Visual Expert 从 Stage I 的 Generative Visual Expert 初始化（warm start）

Stage III (MCoT Training):
- 三个 tasks 交替训练，1:1:1 比例
- 所有 experts 可训练

**Intuition**: Stage II 的 I2T 训练激活了 Semantic Vision Expert（视觉理解能力），这是 MCoT 的 Reflection 步骤的前提条件——Reflection 需要 model 能"理解"图像与 prompt 的差异，这本质是 visual understanding task。

Reference:
- CC12M: https://github.com/google-research-datasets/conceptual-12m
- SAM-1B (Segment Anything): https://arxiv.org/abs/2304.02643
- Grounding DINO: https://arxiv.org/abs/2303.05499
- RichHF-18K: https://huggingface.co/datasets/open-r1/RichHF-18K (需要核对)

## 6. 实验结果深度分析

### 6.1 GenEval Benchmark

Table 1 是核心实验，FoX 用 1.3B 参数量超越所有 baseline:

| Model | Params | Type | Overall |
|-------|--------|------|---------|
| SD3 | 12.7B | Uni. | 0.68 |
| DALL-E 3 | - | Uni. | 0.67 |
| Transfusion | 7.3B | Multi. | 0.63 |
| JanusFlow | 1.3B | Multi. | 0.63 |
| **FoX** | **1.3B** | **Multi.** | **0.77** |

FoX 在 6 个子任务上的表现：
- Single Obj.: 0.99 (近完美)
- Two Obj.: 0.86 (显著优于 JanusFlow 的 0.59)
- Counting: 0.71 (优于所有 baseline)
- Colors: 0.82
- Position: 0.60 (远超 JanusFlow 的 0.53)
- Attr. Binding: 0.64 (远超 JanusFlow 的 0.42)

**Intuition**: FoX 在 Position 和 Attribute Binding 上的提升最大，这正对应 MCoT 的两个核心改进点: layout box planning 提升 Position, detailed caption planning 提升 Attribute Binding。Counting 的提升对应 reflection + correction 修复 object redundancy。

Reference: GenEval: https://arxiv.org/abs/2310.11525

### 6.2 T2I-CompBench Benchmark

Table 2 验证 generalization:
- Color: 82.37 (超越 SD3 的 81.32)
- Spatial: 35.71 (超越 SD3 的 32.00)
- Complex: 42.78 (超越 PixArt-α 的 41.17)

Reference: T2I-CompBench: https://arxiv.org/abs/2310.01794

### 6.3 MS-COCO 基础生成能力

Table 3 验证基础能力没退化:
- FID: 7.24 (DALL-E 2 是 10.39, NExT-GPT 是 10.07)
- CLIP Score: 26.8
- CIDEr: 126.5

**Intuition**: 这点很重要——很多 paper 在某个能力上提升时会牺牲其他能力。FoX 通过 FoXperts 的功能解耦设计避免了这种 trade-off。

Reference: MS-COCO: https://arxiv.org/abs/1405.0312

### 6.4 Image Understanding Benchmarks

Table 4 显示 FoX 保持了强 image understanding:
- MME-P: 1339.7 (Janus 是 1338.0)
- MMBench: 73.6 (超越 Janus 的 69.4)
- VQAv2: 79.4

这证明了 FoXperts 拆分 Semantic Vision Expert 与 Generative Vision Expert 后，视觉理解能力没有退化，反而因为 disentangled optimization 略有提升。

Reference:
- MME: https://arxiv.org/abs/2306.13394
- MMBench: https://arxiv.org/abs/2307.06281
- VQAv2: https://arxiv.org/abs/1612.00837

### 6.5 Ablation Studies

**Table 5 (Ablation of MCoT on GenEval)** 是 paper 最重要的 ablation:

| Setting | Overall |
|---------|---------|
| T2I Gen. Twice (baseline) | 0.67 |
| MCoT Planning & Acting Only | 0.73 |
| MCoT Full Process | 0.77 |

这个对比非常 clever：baseline 不是简单的 T2I 一次，而是 T2I 生成两次取最好。这控制了 "twice computation" 这个混淆变量，纯粹衡量 MCoT 的 reasoning 价值。结果证明 MCoT 的 +0.10 是真正来自 reasoning，不只是多了 compute。

**Table 7 (Ablation of FoXperts on MS-COCO)**:

| Model | CIDEr↑ | FID↓ |
|-------|---------|------|
| Dense | 116.2 | 11.3 |
| Modality-Oriented | 121.1 | 9.56 |
| FoX | 126.5 | 7.24 |

对比非常清晰：
- Dense → Modality-Oriented: CIDEr +4.9, FID -1.74
- Modality-Oriented → FoX: CIDEr +5.4, FID -2.32

FoX 的提升幅度甚至比从 dense 到 modality-oriented 还大，说明 function-disentanglement 是一个独立的、重要的 design dimension。

**Table 9 (Ablation of Reflection & Correction)**:

| Setting | Overall |
|---------|---------|
| T2I Gen. Twice | 42.26 |
| T2I Gen. + Reflection & Correction | 48.86 |

这个对比很有意思：把 MCoT 的前两个 step（Planning & Acting）换成普通的 T2I Gen.，但保留 Reflection & Correction，结果仍有 +6.6 的提升。这说明 **Reflection & Correction 是一个独立的、强大的 capability**，可以 plug-in 到任何 T2I model 上。

Reference: Aesthetic Score, HPS v2: https://arxiv.org/abs/2306.09341

## 7. 与相关工作的深度对比

### 7.1 vs. Janus / JanusFlow

Janus (DeepSeek) 是 unified model 的代表作，采用 decoupled visual encoding: understanding 用 SigLIP encoder, generation 用 VQ tokenizer。JanusFlow 进一步用 Rectified Flow 替代 VQ。

**Key difference**:
- Janus: 在 input encoding 层面 decouple（不同 encoder for understanding vs generation）
- FoX: 在 transformer backbone 层面 decouple（不同 expert for understanding vs generation）

FoX 的 decoupling 更深层，一直 decouple 到 backbone weights。但代价是参数量更多（虽然都是 1.3B）。

Reference:
- Janus: https://arxiv.org/abs/2410.13848
- JanusFlow: https://arxiv.org/abs/2411.07975

### 7.2 vs. Show-o

Show-o 用 single transformer 同时处理 understanding 和 generation，采用离散 tokenization 加 discrete diffusion。

**Key difference**:
- Show-o: 共享 backbone，用不同 head 处理不同 task
- FoX: 功能解耦的 backbone，shared multimodal attention

Show-o 的设计有更严重 function-domain conflict，这正是 FoX 要解决的问题。

Reference: Show-o: https://arxiv.org/abs/2408.12528

### 7.3 vs. Transfusion

Transfusion 是 FoX 最直接的 baseline。Transfusion 用 single transformer，text 用 next-token prediction, image 用 diffusion loss，混合训练。

**Key difference**:
- Transfusion: single shared transformer
- FoX: functionality-oriented experts + shared attention

FoX 的 FoXperts 可以看作是 Transfusion 的 sparse 扩展。

Reference: Transfusion: https://arxiv.org/abs/2408.11039

### 7.4 vs. Mixture-of-Transformers (MoT)

Mixture-of-Transformers (Liang et al., 2024) 是最相似的工作，也用 expert-parallel 架构。但 MoT 是 modality-oriented（一个 visual expert）。

**Key difference**: FoX 把 MoT 的 modality 划分进一步细化为 functionality 划分。这正符合 paper 的核心 thesis。

Reference: Mixture-of-Transformers: https://arxiv.org/abs/2411.04996

### 7.5 vs. LLaVA-CoT / LLaVA-Reasoner

LLaVA-CoT (Xu et al., 2024) 把 CoT 引入 visual understanding，但只用于回答 visual question，没有图像生成。

**Key difference**: FoX 的 CoT 是用于图像生成，而非 visual understanding 推理。MCoT 是第一个把 CoT 显式应用到图像生成的工作之一。

Reference: LLaVA-CoT: https://arxiv.org/abs/2411.10440

### 7.6 vs. Can We Generate Images with CoT (Guo et al., 2025)

这是与 FoX 最接近的 concurrent work，把 CoT 用于图像生成并用 RL 优化。

**Key difference**: 
- Guo et al.: end-to-end RL training，需要 reasoning trace 数据
- FoX: human-defined key steps + disentangled multi-task training，不需要 reasoning trace 数据

FoX 的方法更适合当前的 data-scarce 状态。

Reference: Can We Generate Images with CoT: https://arxiv.org/abs/2501.13926

### 7.7 vs. CoT-VLA

CoT-VLA (Zhao et al., 2025) 把 CoT 引入 vision-language-action model，分解为 subgoal image generation + sequential action generation。

**Key difference**: CoT-VLA 是 robotics 场景，FoX 是 image generation 场景。但都是 human-defined key steps paradigm。

Reference: CoT-VLA: https://arxiv.org/abs/2502.13700

## 8. 训练优化细节

### 8.1 Optimizer

- AdamW, $\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = 0.02
- Learning rate: $5 \times 10^{-5}$, constant schedule with 10K warmup steps
- DeepSpeed ZeRO-2

**Intuition**: 这个配置非常 conservative。LR $5 \times 10^{-5}$ 对 0.5B 模型来说偏低，可能是为了保持预训练知识不破坏。10K warmup 在 300M 样本规模下合理。

Reference:
- AdamW: https://arxiv.org/abs/1711.05101
- DeepSpeed ZeRO: https://arxiv.org/abs/1910.02054

### 8.2 Resolution Strategy

- Stage I: 256×256, batch 4096
- Stage II: 512×512, batch 1024

**Intuition**: 这是 progressive resolution training，先低分辨率学粗结构，再高分辨率学细节。这与 Stable Diffusion 系列训练策略一致。

## 9. 深层 Intuition: 为什么 MCoT 有效？

让我从更根本的角度解释为什么 MCoT 有效。

### 9.1 分布外泛化问题

直接 T2I generation 把 prompt → image 当作一个 end-to-end mapping $f_\theta: \text{prompt} \rightarrow \text{image}$。这个 mapping 在训练分布上 well-defined，但在 OOD 的 compositional case 上泛化差。

**原因**: Compositional case 是 prompt space 的高维组合。如果训练集中 "red dog" 和 "blue cat" 出现多，但 "blue dog" 没出现，直接 T2I 难以正确组合。

MCoT 通过显式分解：
1. **Planning**: 把 "blue dog" 分解成 "blue" + "dog" 两个独立的 attribute
2. **Acting**: 在 plan 的约束下生成
3. **Reflection**: 检查 "blue" 和 "dog" 是否都正确
4. **Correction**: 修复错误部分

这种 decompose-then-compose 的策略天生对 compositional generalization 友好，因为每个 attribute 在 planning 阶段被显式表示。

### 9.2 容量分配问题

直接 T2I 模型要在一次 forward pass 中同时处理:
- Semantic reasoning (理解 prompt)
- Spatial layout (安排物体位置)
- Detail rendering (生成像素)

每个 sub-task 都需要模型 capacity。Single-pass 把所有 capacity 用于一个 joint objective，容易产生 interference。

MCoT 把 capacity 在时间维度上分配:
- Planning 阶段专注 semantic + spatial
- Acting 阶段专注 rendering
- Reflection 阶段专注 semantic alignment check
- Correction 阶段专注 detail fix

这类似 LLM 的 CoT——每个 reasoning step 占用模型一部分 forward pass，避免 capacity bottleneck。

### 9.3 Error Correction 的 Iterative Refinement

Single-pass generation 的 error 是 "all-or-nothing" 的。Reflection + Correction 引入了 **iterative refinement** 的机会。即使 Acting 阶段有错误，Reflection 可以识别，Correction 可以修复。

这与 LLM 的 self-correction 类似，但在视觉领域更难，因为视觉错误更难"describe"。

## 10. Limitations & 个人思考

paper 在 Section E 提到 limitation: 没有处理 fine-grained 和 customized image editing。

我个人的进一步思考：

### 10.1 MCoT 的 Cost

MCoT 的 inference cost 是直接 T2I 的 ~4 倍（4 个 step 都需要 forward pass）。这是显著的开销。Table 5 的 ablation 中 baseline 是 "T2I Gen. Twice"，已经控制了 2x compute 的因素，但 MCoT 是 4x。一个 interesting 的 future work 是 adaptive MCoT——简单 prompt 跳过 reflection/correction，只对复杂 prompt 执行完整 MCoT。

### 10.2 Reflection 模型的能力上限

Reflection 步骤本质是 visual question answering: "图像与 prompt 是否一致？哪里不一致？" 这个能力受限于 Semantic Vision Expert 的理解能力。如果第一步生成的图像有 Reflection 模型都看不出的问题，correction 就失败了。

这可能解释为什么 Table 5 中 MCoT Full Process 的 Single Obj. 是 0.99（几乎完美），但 Two Obj. 是 0.86（仍有错误）。多物体场景的 reflection 更难。

### 10.3 Layout Planning 的 Robustness

Layout box 依赖 Grounding-DINO + SAM 在 caption 中提取物体。如果 Grounding-DINO 漏检或错检某些 object，layout 就错了。这是 pipeline 的 bottleneck。

### 10.4 与 RLHF 的结合

paper 用的是 supervised multi-task training。一个 interesting 的方向是用 RL 来训练 MCoT。比如：
- Reward: final image 与 prompt 的 alignment score (用 VLM 作为 reward model)
- RL 算法: PPO 或 GRPO

这可以让模型学习 "什么时候需要 correction" 而不是固定的 4-step 流程。类似 DeepSeek-R1 在 LLM 中的做法。

### 10.5 Scaling Behavior

paper 只展示了 1.3B 模型的结果。MCoT 的收益是否会随 scale 衰减？通常 reasoning 的收益在小模型上更明显（因为大模型 single-pass 已经能 implicit reasoning）。但 FoXperts 的设计也可能在大模型上更受益（因为 expert 专业化在大模型上更重要）。这需要更多实验验证。

### 10.6 与 GPT-5o 的关系

paper 在 introduction 中提到 GPT-5 (OpenAI, 2025) 作为 unified generative models 的代表。FoX 的 MCoT 思想其实和 GPT-5o 的 native image generation + reasoning 可能有相似之处——都是用 reasoning 来引导图像生成。但 GPT-5o 是 end-to-end RL，FoX 是 disentangled supervised training。

Reference: GPT-5o: https://openai.com/index/introducing-gpt-5o/

### 10.7 与 Diffusion Models 的 Reflection 比较

最近有一些 diffusion-specific 的 reflection 工作，如 Self-Correcting Diffusion (Lee et al., 2023)。这些方法专门为 diffusion 设计。FoX 的 reflection 更 general，适用于任何 unified model。但专门化方法可能在 diffusion-only 场景下更高效。

Reference: Self-Correcting Diffusion: https://arxiv.org/abs/2311.01620

## 11. 总结：Paper 的核心贡献与启示

FoX 这篇 paper 的核心贡献有三个层次：

1. **Insight 层次**: 发现 unified model 中 visual understanding 与 visual generation 的 function-domain conflict，并提出按功能解耦的 FoXperts 设计。
2. **Method 层次**: 提出 MCoT，把图像生成显式分解为 planning-acting-reflection-correction 四个 step，模仿人类艺术家 workflow。
3. **Training 层次**: 提出 disentangled multi-task training，绕过 end-to-end 训练需要 consistent multi-step data 的困难。

**对 unified generative model 领域的启示**:
- Modality-oriented 可能不是 expert 划分的最佳粒度，functionality 是更自然的划分维度
- CoT 不止适用于 NLP reasoning，也是图像生成的重要 tool，特别是 compositional generation
- Disentangled training 在 multi-modal reasoning 中可能比 end-to-end 更实用，因为 multi-modal alignment data 极难收集

**对 Karpathy 这种研究者的启示**:
- 这是 "explicit reasoning > implicit reasoning" 的另一个证据。在 NLP 中我们看到 CoT 的巨大价值，在图像生成中也开始看到类似的 pattern
- 但图像生成的 CoT 设计空间比 NLP 大得多——可以是文本 reasoning（detailed caption）、空间 reasoning（layout box）、视觉 reasoning（artifact map）、修正 reasoning（correction）。FoX 是一个 early exploration
- Future: 更灵活的 MCoT（adaptive steps）+ RL training + 更大规模验证

**可能的 follow-up 方向**:
- RL 训练 MCoT（用 VLM 作为 reward model）
- Adaptive MCoT（根据 prompt 复杂度调整 step 数）
- 把 MCoT 扩展到 video generation（planning scene → acting → reflection → correction）
- 把 MCoT 扩展到 multi-turn image editing（与 DALL-E 3 的 conversational editing 结合）
- End-to-end MCoT 数据生成（用更好的 model 生成 "wrong image" 来训练 reflection）

## 12. 关键公式汇总

| 公式 | 用途 | 关键变量 |
|------|------|----------|
| Eq. (1) | Autoregressive factorization | $z, V, N, \theta, z_{<i}$ |
| Eq. (2) | LM loss (for understanding) | $\mathcal{D}, N, \theta$ |
| Eq. (3) | Rectified Flow trajectory | $\mathbf{x}, \epsilon, t, \mathbf{x}_t$ |
| Eq. (4) | RF loss (for generation) | $v_\theta, c, \mathbf{x}_t$ |
| Eq. (5) | FoXperts attention forward | $W_i, Router, LN, T/C/N, \oplus$ |
| Eq. (6) | FoXperts FFN forward | $FFN_i, T/C/N$ |

整体而言，FoX 是 unified generative model 领域一个相当 solid 的工作，把 NLP 中的 CoT 思想创造性扩展到图像生成，并通过 FoXperts 和 disentangled training 解决了实际工程挑战。1.3B 参数在 GenEval 上达到 0.77 是非常 impressive 的结果，远超 12.7B 的 SD3 (0.68)。这个工作预示着图像生成领域可能迎来类似 NLP 的 "reasoning 时代"。

Reference (paper 本身): https://arxiv.org/abs/2505.17923 (注：实际链接需要核对，paper title 是 "Towards Enhanced Image Generation via Multi-Modal Chain of Thought in Unified Generative Models")
