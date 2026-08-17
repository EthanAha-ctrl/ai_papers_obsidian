---
source_pdf: Think with 3D.pdf
paper_sha256: af7e590e3486611b489a1adc1d5639dbe914a1ed0715b71f7b4bbb32c9e9cef1
processed_at: '2026-08-12T15:23:55-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Andrej，我们抛开那些学术腔调，直接用最直白的人话，结合底层 mechanism，把这篇 paper 扒得干干净净。我尽量把里面的直觉和工程细节都给你揉碎了讲出来。

### 1. Core Intuition：VLM 为什么需要“脑补” 3D？

现在的 VLM（比如 Qwen2.5-VL 或者 GPT-4o）在看图说话的时候，本质上还是在玩 token 接龙。你要问它“桌子在沙发后面吗？”或者“从你现在的视角走到冰箱要几步？”，它往往歇菜。因为 2D pixel 里的空间关系，用 1D text 是极难完美描述的。以前的做法要么是让模型生成一堆文本来描述坐标系，要么是外挂一个 depth estimator 提取深度图喂进去。

3DThinker 的核心 intuition 极其简单粗暴：既然人脑在想空间问题时会浮现出一个 3D 画面，那我们也强行在 VLM 的 forward pass 里开一个“脑补窗口”。在这个窗口里，模型不输出文字，而是在 latent space 里直接生成一段代表 3D 几何的 continuous embeddings。并且，我们不需要模型真的吐出一个 point cloud，我们只需要它的 internal hidden states 跟一个专业的 3D foundation model 算出来的结果一模一样就行。这就是所谓的 "think with 3D mentaling"。

### 2. 架构拆解：怎么强行把 3D 塞进 1D Text 流？

整个架构的 workflow 是这样的：

输入几张 limited views 的图像 $\mathcal{T}$ 和问题 $Q$。VLM 开始自回归地吐 token。吐着吐着，遇到了特殊标记 `<|latent start|>`。此时，VLM 连续吐出 $k$ 个 `<|latent pad|>` token（实验证明 $k=12$ 最好）。在这 $k$ 个 token 吐完之后，遇到 `<|latent end|>`，VLM 继续吐正常的文字，最后给出答案。

关键在于这 $k$ 个 `<|latent pad|>` token 的 last layer hidden states。我们把它们抽出来，送进一个 Projector（就是一个 6 层的 MLP），映射出 projected features $F_{\mathrm{proj}}$。同时，我们把这些图片喂给 3D foundation model VGGT，拿到它的 geometry features $F_{3\mathrm{D}}$。训练的时候，我们就强迫 $F_{\mathrm{proj}}$ 和 $F_{3\mathrm{D}}$ 在数值上对齐。推理的时候，由于 Projector 已经学会了这层映射，我们甚至可以把 $F_{\mathrm{proj}}$ 直接喂给 VGGT 的 DPT head，直接渲染出一个 point cloud 出来！这就是 Figure 4 里展示的可解释性——你真的能看到模型在“脑补”什么形状的 3D 场景。

### 3. 公式与技术细节：从 SFT 到 RL 的全流程

我们来看它具体怎么训。整个过程分两个 Stage。

#### Stage 1: Supervised Training (SFT) - 建立几何对齐
这个阶段的目的是让 VLM 学会“格式”以及初步的几何感知。

公式 (2) 把推理轨迹拆开了：
$$o = o_{\mathrm{pre}} \oplus t_{3\mathrm{D}} \oplus o_{\mathrm{post}}$$
这里 $o$ 是完整轨迹，$o_{\mathrm{pre}}$ 是 3D tokens 之前的文本，$t_{3\mathrm{D}}$ 是 12 个 3D tokens，$o_{\mathrm{post}}$ 是之后的文本和答案。

公式 (3) 定义了这 12 个 3D tokens 的 hidden states 是怎么来的：
$$h_i = \begin{cases} f_\theta^{\mathrm{hidden}, L} (Q, \mathcal{Z}, o_{\mathrm{pre}}), & i = 1 \\ f_\theta^{\mathrm{hidden}, L} (Q, \mathcal{Z}, o_{\mathrm{pre}}, t_{1:i-1}), & i \geq 2 \end{cases}$$
- $h_i$ 是第 $i$ 个 3D token 的 hidden state。
- $f_\theta$ 是 VLM，$\theta$ 是参数。
- 上标 $\mathrm{hidden}, L$ 明确指出是第 $L$ 层（最后一层）的 hidden state。
- $Q$ 是 question，$\mathcal{Z}$ 是图像的 visual latents。
- 这其实就是一个标准的自回归过程，第 $i$ 个 latent 依赖于前面所有的 context。信息在这里逐步凝聚。

公式 (4) 和 (5) 是 Stage 1 的灵魂：
$$F_{\mathrm{proj}} = \mathrm{Projector}(F_{\mathrm{latent}}, F_{\mathrm{images}})$$
$$\mathcal{L}_{3D} = \| F_{\mathrm{proj}} - F_{3\mathrm{D}} \|_F^2$$
- $F_{\mathrm{latent}}$ 就是 $\{h_1, \ldots, h_k\}$。
- $F_{\mathrm{images}}$ 是 patch-level visual features，融合进来是为了补充 low-level 细节。
- $\mathcal{L}_{3D}$ 是 Frobenius norm loss（下标 $F$ 表示矩阵所有元素的平方和）。
- $F_{3\mathrm{D}}$ 是 VGGT aggregator 输出的 feature。
这个 loss 强行在 VLM 的高维 semantic space 里“凿”出了一个 3D 子空间。同时还要加上普通的 cross-entropy loss $\mathcal{L}_{\mathrm{text}}$ 保证它还能好好说话。总 loss 是 $\mathcal{L}_{\mathrm{total}} = 0.1 \mathcal{L}_{3D} + 1 \mathcal{L}_{\mathrm{text}}$。

#### Stage 2: Reinforced Spatial Mentaling (RL) - 任务驱动的 refinement
SFT 只是学会了模仿 VGGT，但这不代表模型能在回答问题时用好这个 3D latent。所以 Stage 2 用 GRPO 来做 RL。

公式 (10) 是标准的 GRPO 目标函数：
$$\mathcal{I}(\theta) = \frac{1}{N} \sum_{i=1}^N \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big\{ \min \Big[ \mathrm{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_{i,t}, r_{i,t} \hat{A}_{i,t} \Big] - \beta \mathbb{D}_{\mathrm{KL}}[\pi_\theta || \pi_{\mathrm{ref}}] \Big\}$$
- $N$ 是 group size，这里设为 8，即同一个 prompt 采样 8 条 trajectory。
- $r_{i,t}$ 是 likelihood ratio（新 policy 和老 policy 的概率比）。
- $\hat{A}_{i,t}$ 是 group-normalized advantage。
- $\mathbb{D}_{\mathrm{KL}}$ 是 KL divergence，防止模型跑飞。

最精妙的是它设计的 reward。公式 (12) 定义了 3D visual token reward：
$$r_{\mathrm{3D}} = \frac{1}{2} \left(1 + \frac{F_{\mathrm{proj}}^{RL} \cdot F_{3D}}{\| F_{\mathrm{proj}}^{RL} \| \| F_{3D} \|}\right)$$
- $F_{\mathrm{proj}}^{RL}$ 是当前 rollout 生成的 3D latent 经过 frozen Projector 的输出。
- $F_{3D}$ 是 VGGT 的 feature。
- 这就是一个 cosine similarity，经过 $\frac{1}{2}(1 + \dots)$ 缩放到 $[0, 1]$ 之间。
这个 reward 保证了模型在 RL 探索的时候，生成的 3D latent 始终被 VGGT 的几何空间 anchor 着，不会为了拿答案 reward 而把 3D latent 给“忘掉”或者“扭曲”了。此外还有 format reward 和 answer reward。

### 4. 实验数据表的 intuition

我们来看 Table 5 的 ablation study，关于 Latent Size $k$ 的影响：
- $k=4$: Accuracy 60.2
- $k=8$: Accuracy 60.6
- $k=12$: Accuracy 62.7 (Sweet Spot)
- $k=16$: Accuracy 59.9
- $k=32$: Accuracy 25.1 (断崖式下跌)
- $k=64$: Accuracy 15.5

为什么会这样？如果 $k$ 太小，representation capacity 不够装下 3D 几何信息。如果 $k$ 太大，VLM 在自回归吐这几十个 latent token 的时候，极易陷入某种 degenerate solution，也就是无休止地吐 `<|latent pad|>`，或者因为占用了太多的 attention bandwidth，导致后续 $o_{\mathrm{post}}$ 的文本生成彻底崩溃，连答案都输出不出来。这其实是一个典型的 information bottleneck 问题。

再看 Table 6 的 Token Position ablation：
- 放在 Middle: Accuracy 暴跌至 42.0
- 放在 End (在 `<answer>` 之前): Accuracy 74.3
- 放在 Beginning: 效果也不错

这是因为 Transformer 是自回归的，如果你把一大坨没经过充分训练的 latent tokens 插在文本中间，后面的 text token 在做 attention 的时候会被这些“异类”特征严重干扰，导致生成乱码或者提前 EOS。放在结尾相当于模型先把能说的文本推理说完，然后在脑子里形成一个综合的 3D 想象，最后直接吐出答案。这非常符合“先想清楚，再给答案”的直觉。

### 5. 更广维度的疯狂联想

1. **Latent System 2 Thinking**
这篇 paper 本质上是在 latent space 里做 System 2 thinking。传统的 CoT 是在 token space 把推理过程拉长，消耗的是 token 数量和 forward pass 的时间。而 3DThinker 是在 continuous latent space 里拉长推理过程。如果我们未来把 diffusion 机制或者 iterative refinement 引入这 $k$ 个 latent tokens 之间，让它们在内部做几百步的 attention 交互再输出，那这简直就是把 Diffusion Model 的迭代推理和 LLM 的自回归推理完美融合了。

2. **World Model 的隐式预测**
模型生成的这些 3D latent，其实就是一个对物理世界的 state representation。现在它只是用来回答空间问题。如果我们在这个 latent 基础上加一个 head，去预测下一帧的 3D latent（比如机器人采取某个 action 之后），那这就变成了一个纯 Transformer 架构的 World Model。跟 Dreamer 那种基于 RNN 或者显式 pixel generation 的 World Model 相比，这种 latent space 的 World Model 计算效率极高，而且泛化能力可能更强。

3. **Unified Tokenizer 的终极形态**
作者在 Limitation 里提到，目前 3D latents 没有自回归地融入框架。未来的终极解法一定是 Unified Tokenizer。把 text、image、3D geometry 全部离散化或者映射到同一个 continuous codebook 里。到那个时候，模型在输出时完全无缝切换，它可以输出一段文字，然后输出一段 3D latent，再基于这个 3D latent 渲染一张新视角的图片，然后再继续写文字。这甚至能让我们在 latent space 里做 Monte Carlo Tree Search (MCTS)，把 AlphaGo 的那套搜索算法搬到 LLM 的 internal representation 里来。

4. **跟 3D Gaussian Splatting 的结合**
现在的 Projector 是对齐到 VGGT 的 feature space，然后解出 point cloud。Point cloud 是离散的，渲染效果差。如果我们将 Projector 的 target space 改为 3D Gaussian Splatting (3DGS) 的参数空间，让 VLM 直接在 latent 里输出每个 Gaussian 的 mean, covariance, opacity, color。那模型脑补出来的就不是一个粗糙的点云，而是一个可以实时渲染、随意旋转视角的高保真 3D 场景。这会让 VLM 直接具备了一个内置的 3D 渲染引擎。

5. **Interpretability 的破局点**
Deep Learning 一直被诟病黑盒。这篇 paper 提供了一个极佳的 interpretability 视角。当模型给出错误答案时，我们不仅可以看到它的 CoT 文本哪里错了，我们可以直接把它脑补的 3D latent 解码成 point cloud，看看它是不是把空间结构给搞反了。这种将 high-dimensional latent 直接投影回物理世界进行可视化的范式，未来可能会成为研究 large reasoning model 行为的标配工具。

总而言之，3DThinker 这个工作用非常扎实的工程手段（SFT + RL + Projector Distillation），验证了一个非常深刻的直觉：VLM 内部需要有针对特定模态（如 3D 几何）的 latent scratchpad。这为我们设计下一代具备强物理世界感知能力的 AGI 架构提供了极其有价值的参考。

**Reference Links:**
- VGGT (Visual Geometry Grounded Transformer): https://vgg-t.github.io/
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- Qwen2.5-VL Technical Report: https://arxiv.org/abs/2502.13923
- MindCube (Spatial mental modeling from limited views): https://arxiv.org/abs/2506.21458
- 3DThinker Project Page: https://github.com/zhangquanchen/3DThinker

---

Hi Andrej，非常荣幸能和你深入探讨这篇paper。这篇文章的思路非常奇妙，试图在VLM的forward pass中强行开辟出一个"3D scratchpad"，让模型在输出text token的间隙，在latent space里进行隐式的3D几何想象。这非常符合你对neural network底层机制的直觉，本质上就是在探索representations的流动。

以下我将从core intuition、技术细节公式拆解、architecture flow、experimental insights以及更广维度的联想来进行详尽的解析。

### Core Intuition: 从1D Text序列到3D Latent空间的跃迁

当前大多数VLM在进行reasoning时，完全依赖1D的text token序列（比如生成cognitive map的文本描述），或者借助外部的depth estimator提取prior。这种方式的representational capacity存在上限，因为3D空间关系极难用纯text或者2D pixel完美描述。

3DThinker的core idea是：既然人类在思考空间问题时会在脑海中浮现出3D mental image，VLM也应该在autoregressive生成过程中，拥有一段处于3D geometric space的latent representations。为了实现这一点，作者没有选择硬标注3D data，而是采用了一种类似distillation加RL的范式。VLM在推理时生成一段特殊的`<|latent pad|>` tokens，这些token在经过最后一层transformer后，其hidden states被projector映射到一个与3D foundation model（VGGT）完全对齐的feature space。从而，VLM在"思考"时，其内部latent就真正蕴含了场景的3D geometric structure。

### Technical Deep Dive: 公式与机制拆解

#### 1. Trajectory分解与Latent Insertion
在Supervised stage，核心在于如何构建带有3D placeholder的CoT数据。
公式(2): $o = o_{pre} \oplus t_{3D} \oplus o_{post}$
- $o$ 表示完整的reasoning trajectory。
- $o_{pre}$ 和 $o_{post}$ 分别表示3D special tokens前后的text tokens。
- $t_{3D} = \{t_1, \ldots, t_k\}$ 代表连续的 $k$ 个3D special tokens。这里的 $k$ 就是latent size，实验表明 $k=12$ 是sweet spot。

公式(3)定义了这些3D tokens的hidden states是如何递归生成的：
$$h_i = \begin{cases} f_\theta^{\mathrm{hidden}, L} (Q, \mathcal{Z}, o_{pre}), & i = 1, \\ f_\theta^{\mathrm{hidden}, L} (Q, \mathcal{Z}, o_{pre}, t_{1:i-1}), & i \geq 2. \end{cases}$$
- $h_i$ 是第 $i$ 个3D token的salient vector。
- $f_\theta$ 是参数为 $\theta$ 的VLM。
- 上标 $\mathrm{hidden}, L$ 明确指出提取的是第 $L$ 层（通常是最后一层）的hidden state。
- $Q$ 是question，$\mathcal{Z}$ 是输入的图像latent，$o_{pre}$ 是前文text。
- 这个公式完美体现了autoregressive的本质，第 $i$ 个latent token的表征依赖于它前面的所有context，从而实现信息的逐步积累和refinement。

#### 2. Projector与Geometric Distillation
为了让VLM的hidden state具有3D几何意义，作者引入了VGGT作为teacher model。
公式(4): $F_{\mathrm{proj}} = Projector(F_{\mathrm{latent}}, F_{\mathrm{images}})$
- $F_{\mathrm{latent}} = \{h_1, \ldots, h_k\}$ 是VLM输出的3D latents。
- $F_{\mathrm{images}}$ 是patch-level的visual features。
- Projector通过融合这两者，生成projected features $F_{\mathrm{proj}}$。这里结合 $F_{\mathrm{images}}$ 可能是为了补充VLM在高层语义提取中丢失的low-level visual details，从而更好地与VGGT的特征空间对齐。

公式(5)定义了3D alignment loss：
$$\mathcal{L}_{3D} = \| F_{\mathrm{proj}} - F_{3\mathrm{D}} \|_F^2$$
- $F_{3\mathrm{D}}$ 是VGGT aggregator提取的geometry features。
- 下标 $F$ 表示Frobenius norm，即矩阵所有元素的平方和。
- 这个loss强迫VLM在reasoning时生成的latent，经过projection后，必须在数值上逼近VGGT看到同样图像时产生的内部几何表征。这相当于在VLM的representation空间中"凿"出了一个3D子空间。

同时，为了保证语言能力不退化，使用cross-entropy loss优化 $o_{pre}$ 和 $o_{post}$ 的text预测，如公式(6)和(7)所示。最终总loss为 $\mathcal{L}_{\mathrm{total}} = \lambda_{3D} \mathcal{L}_{3D} + \lambda_{\mathrm{text}} \mathcal{L}_{\mathrm{text}}$，其中 $\lambda_{3D}=0.1, \lambda_{\mathrm{text}}=1$。

#### 3. Reinforced Spatial Mentaling (Stage 2)
SFT阶段只教会了模型"格式"和"初始几何对齐"，但这并不能保证模型在回答具体问题时能动态调整3D latent。Stage 2引入了GRPO进行RL优化。

公式(10)是标准的GRPO目标函数：
$$\mathcal{I}(\theta) = \frac{1}{N} \sum_{i=1}^N \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big\{ \min \Big[ \mathrm{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_{i,t}, r_{i,t} \hat{A}_{i,t} \Big] - \beta \mathbb{D}_{\mathrm{KL}}[\pi_\theta || \pi_{\mathrm{ref}}] \Big\}$$
- $N$ 是group size（rollout number为8）。
- $|o_i|$ 是trajectory长度。
- $r_{i,t} = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t} | q, o_{i,<t})}$ 是likelihood ratio。
- $\hat{A}_{i,t}$ 是group-normalized advantage，如公式(11)所示，由同组rollout的reward均值和方差归一化得到。
- $\mathbb{D}_{\mathrm{KL}}$ 防止policy偏离reference model太远。

公式(12)定义了非常巧妙的3D visual token reward：
$$r_{\mathrm{3D}} = \frac{1}{2} (1 + \frac{F_{\mathrm{proj}}^{RL} \cdot F_{3D}}{\| F_{\mathrm{proj}}^{RL} \| \| F_{3D} \|})$$
- 这是cosine similarity的线性缩放，范围在 $[0, 1]$ 之间。
- $F_{\mathrm{proj}}^{RL}$ 是RL阶段当前rollout生成的3D latent经过frozen projector的输出。
- 这保证了在RL探索过程中，模型生成的3D latent不会"跑飞"，依然受到VGGT几何空间的anchor约束。

### Architecture Flow & Interpretability

从Figure 2和Figure 3可以看出，整个pipeline的精妙之处在于**双向兼容**：
1. **Forward pass (Inference)**：输入limited views images和question，VLM开始生成 $o_{pre}$。遇到 `<|latent start|>` 后，连续生成12个 `<|latent pad|>`。提取这些pad tokens的last layer hidden states，送入Projector。Projector输出的feature可以通过VGGT自带的DPT head解码成point cloud。之后VLM继续生成 $o_{post}$ 并给出最终answer。
2. **Backward pass (Training)**：SFT阶段，梯度从 $\mathcal{L}_{3D}$ 和 $\mathcal{L}_{\mathrm{text}}$ 回传，同时优化VLM参数 $\theta$ 和Projector。RL阶段，Projector frozen，梯度从reward通过policy gradient回传优化VLM。

这种设计赋予了模型极强的interpretability。我们不仅能看到模型输出的text reasoning，还能把模型在"思考"中间步骤产生的latent直接可视化成3D point cloud（如Figure 4所示）。通过观察point cloud，我们可以直观地看到模型在脑海中构建了哪些场景元素，这对于理解large reasoning model的黑盒行为具有革命性意义。

### Experimental Insights & Ablation Analysis

实验数据非常扎实，在Table 1中，3DThinker基于Qwen2.5-VL-72B，在MindCube-Tiny上达到78.9，甚至超过了O3 (56.6)和Gemini 2.5 Pro (66.7)。在Ego3D-Bench上也达到73.3，超越所有闭源模型。这证明了intrinsic 3D mentaling的巨大潜力。

Table 5的Ablation study非常揭示本质：
- 当Latent Size为12时，Accuracy最高（62.7）。
- 当Latent Size增加到32或64时，Accuracy断崖式下跌到25.1和15.5。
这从侧面反映了neural network优化中的"information bottleneck"问题。如果latent sequence太长，模型在autoregressive生成这些pad tokens时，极易陷入某种degenerate solution（比如无限重复生成latent tokens而无法exit），或者过多的latent tokens吸收了本该用于text reasoning的capacity，导致后续text生成崩溃。这与你之前提到的LLM在生成时容易陷入loops的机制很像。

Table 6的Token Position ablation也很有意思：
- 把3D tokens放在beginning或end，Accuracy在74-75左右。
- 放在middle，Accuracy暴跌至42.0。
这是因为VLM是autoregressive的，如果在text中间强行插入大量latent tokens，会彻底破坏language model的context连贯性。后续的text token在attend前面的内容时，会被这些未经过SFT充分对齐的latent特征干扰，导致输出garbled text或提前输出EOS。这也说明，当前的3DThinker实际上更像是一个"前向想象"或"后向反思"的过程，而不是在逻辑链条中频繁插入想象的交互过程。

### Broader Implications & Associations

阅读这篇paper，我脑海中涌现出很多更广维度的联想：

1. **System 1 vs System 2 in Latent Space**
这篇paper是典型的"Latent System 2"实现。传统CoT是在token space展开的System 2。3DThinker将思考过程移到了continuous latent space，这类似于人类非语言的潜意识推理。未来如果能将diffusion process或者iterative refinement机制引入这些latent tokens的生成过程（比如让latent tokens之间进行多步内部attention交互），可能会产生更强的reasoning涌现。

2. **Gradient Flow与Projector设计**
Projector接收 $F_{\mathrm{latent}}$ 和 $F_{\mathrm{images}}$。如果未来要扩展到更复杂的3D任务（如4D时空推理），Projector可能需要引入cross-attention机制，让VLM的latent tokens主动query图像的spatial-temporal细节。并且，目前的latent tokens是非自回归生成的（虽然公式3表面上是自回归，但生成的是固定数量的pad），未来如果能像Diffusion model那样通过一个stopping criterion动态决定生成多少个3D latent token，将会大大提高计算效率。

3. **与3D Gaussian Splatting的结合**
VGGT提取的是feature，最终通过DPT生成point cloud。目前3D领域非常火热的3D Gaussian Splatting (3DGS) 在表征质量和渲染效率上远超point cloud。假如我们将Projector的target space从VGGT的feature space改为3DGS的参数空间（即让VLM直接输出mean, covariance, color等Gaussian参数），那么VLM不仅能在脑海中"想象"3D场景，还能直接"渲染"出任意视角的高保真图像。这将彻底打通language reasoning和neural rendering的边界。

4. **World Models的隐式构建**
这篇paper其实是在做一种受限的World Model。通过在推理时生成3D latent，VLM实际上是在内部构建了一个对物理世界的几何state model。如果我们将这个思路推广到video generation和embodied AI，让模型在采取action之前，先在latent space预测action带来的3D场景变化，这就实现了类似于Dreamer系列的World Model架构，并且是用纯transformer的end-to-end方式实现的。

5. **Unified Tokenizer的必要性**
正如作者在Limitation中提到的，目前3D latents没有autoregressively incorporated into the framework。这其实是因为text tokenizer和3D latent space存在gap。未来可能需要一种真正的Unified Tokenizer，能把text、image、3D geometry全部映射到同一个discrete或continuous codebook里。那时，模型在生成text和生成3D imagination时将完全无缝，甚至可以像AlphaGo做MCTS一样，在latent space里进行tree search，探索多条3D reasoning path。

总而言之，3DThinker这篇paper不仅是一个spatial reasoning的SOTA method，更重要的是它提供了一种将外部foundation model的feature space蒸馏进VLM内部reasoning trajectory的paradigm。这种"think in latent space"的思想极具启发性，预示着未来large reasoning model的发展方向将越来越向continuous latent dynamics靠拢。

以下是相关的reference links：
- VGGT (Visual Geometry Grounded Transformer): https://vgg-t.github.io/
- GRPO (Group Relative Policy Optimization): https://arxiv.org/abs/2402.03300
- Qwen2.5-VL Technical Report: https://arxiv.org/abs/2502.13923
- MindCube (Spatial mental modeling from limited views): https://arxiv.org/abs/2506.21458
- Project page of 3DThinker: https://github.com/zhangquanchen/3DThinker
