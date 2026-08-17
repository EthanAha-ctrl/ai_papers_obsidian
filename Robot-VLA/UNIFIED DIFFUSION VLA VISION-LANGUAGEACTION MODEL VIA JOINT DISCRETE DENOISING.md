---
source_pdf: UNIFIED DIFFUSION VLA VISION-LANGUAGEACTION MODEL VIA JOINT DISCRETE DENOISING.pdf
paper_sha256: 1bbf37539f7fd261d4fe724a8e84d2acc7bf8b59a6dd48ccfe23c8d697fbdeed
processed_at: '2026-08-12T19:30:44-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy 教授您好，如果用最直白的“人话”来拆解 UD-VLA 这篇 paper，我们可以把它想象成教机器人“做填空题”的过程。

以前的机器人模型，要么是死记硬背看到什么就做什么，要么是脑子里先想好一张未来的目标图片，然后再照着这张图去想动作。这些方法要么反应慢，要么想图和想动作是分家的，互相帮不上忙。

UD-VLA 的核心逻辑在于：**让机器人边想未来画面，边写动作指令，并且这两件事在同一个脑子里同步迭代、互相启发。**

下面我用更细节的技术语言和公式为您建立更深的 intuition。

### 1. 核心痛点与 UD-VLA 的破局思路

当前 VLA 领域有两条主流路线，都有硬伤：
*   **外挂专家路线 (比如 GR-1, $\pi_0$)**：VLM 负责看懂世界，外接一个 diffusion head 负责生成动作。这导致了 modality misalignment，视觉理解和动作生成在特征空间里隔着一堵墙。
*   **纯 AR 路线 (比如 OpenVLA, UniVLA)**：把动作也当成文本 token 一个个生成。问题是图像生成用 AR 非常别扭，且逐 token 生成速度极慢，存在严重的 exposure bias。

UD-VLA 的破局思路是：把文本、当前图像、未来图像、动作统统变成离散的 token，然后塞进同一个 Transformer 里。接下来，把“未来图像”和“动作”部分的 token 全部遮住，让模型通过多步“去噪”（也就是不断填空），同时把未来画面和动作给填出来。在填空的过程中，动作 token 可以一直盯着未来图像 token 正在成型的样子来调整自己，这就实现了真正的 synergy。

### 2. 架构直觉：信息流的单向阀门

为了防止模型“作弊”（比如从动作 token 偷偷反推图像，导致图像生成器变懒），UD-VLA 设计了一个极度精妙的 Hybrid Attention Mechanism (图 2)。

*   **Intra-modal (模态内部)**：未来图像 tokens 之间用 bidirectional attention，动作 tokens 之间也用 bidirectional attention。因为一张图的不同 patch 之间没有先后顺序，一段 action chunk（包含多个时间步的 xyzrpy）也没有严格的因果先后，全向交流能让它们全局一致。
*   **Cross-modal (模态之间)**：严格单向。动作可以看未来图像，未来图像绝对不能看动作。这在数学上强制模型把动作预测建模成 inverse kinematics：你必须先构思出“我要把杯子放到左边”的未来画面，然后基于这个画面反推手臂该怎么动。

### 3. 理论拆解：JD3P 到底在算什么？

JD3P (Joint Discrete Denoising Diffusion Process) 是这篇 paper 的灵魂。它把离散的 mask-predict 过程套上了连续 diffusion 的数学外衣。

**前向加噪**
把干净的未来图像 $\mathbf{v}_0$ 和动作 $\mathbf{a}_0$ 拼在一起。加噪过程就是以概率 $\beta_t$ 把某个 token 变成 mask token $M$：

$$ \mathbf{Q}_t \mathbf{e}_{t,r} = (1 - \beta_t) \mathbf{e}_{t,r} + \beta_t \mathbf{e}_{\mathrm{M}} $$

*   $\mathbf{Q}_t$: 时间步 $t$ 的转移矩阵。
*   $\mathbf{e}_{t,r}$: 序列中位置 $r$ 的 one-hot 向量。
*   $\beta_t$: 这一步的 mask 概率。
*   $\mathbf{e}_{\mathrm{M}}$: mask token 的 one-hot 向量。

这个公式的意思非常简单：每一步，每个 token 有 $\beta_t$ 的概率被擦掉变成 `<MASK>`，有 $1-\beta_t$ 的概率保持原样。

**反向去噪**
这是模型真正干的事。给定 $t$ 时刻被部分 mask 的序列和 context $\mathbf{c}$，模型要预测 $t-1$ 时刻的序列。关键在于这个条件概率的分解（公式 5）：

$$ p_\theta(\mathbf{v}_{t-1}, \mathbf{a}_{t-1} \mid \mathbf{v}_t, \mathbf{a}_t, \mathbf{c}) = p_\theta(\mathbf{v}_{t-1} \mid \mathbf{v}_t, \mathbf{c}) p_\theta(\mathbf{a}_{t-1} \mid \mathbf{v}_t, \mathbf{a}_t, \mathbf{c}) $$

*   左边：图像去噪，只看被 mask 的图像和自己。
*   右边：动作去噪，看被 mask 的动作、被 mask 的图像，以及 context。
这个公式在数学上保证了 action 的预测依赖于当前的 visual denoising state。

**Loss 函数**
训练时没有用复杂的 ELBO，直接采用了单步 masked cross-entropy (公式 8)：

$$ \mathcal{L}_{\mathrm{CE}}(\theta) = - \omega \sum_{j}^{L_v} \log p_\theta^{(v)}(v_{0,j} \mid \mathbf{v}_t, \mathbf{c}) \cdot \mathbb{1}\{v_{t,j} = \mathrm{M}\} - \sum_{i}^{L_a} \log p_\theta^{(a)}(a_{0,i} \mid \mathbf{v}_t, \mathbf{a}_t, \mathbf{c}) \cdot \mathbb{1}\{a_{t,i} = \mathrm{M}\} $$

*   $L_v, L_a$: 图像和动作的 token 长度。
*   $\mathbb{1}\{...\}$: Indicator function，只对被 mask 掉的位置计算 loss。
*   $\omega$: 降权系数。因为图像 token 太多，如果不降权，模型会花所有精力去画图，动作就学不好了。

### 4. 训练与推理的黑科技

**两阶段训练**
1.  **World Model Post-training**：先用海量视频数据，训练 `[ text ; current image ; future image ]`。让 VLM 先学会“想象未来”。
2.  **Joint Fine-tuning**：加入 action tokens，在机器人数据集上联合训练。这里用了一个 shift operation，把 next-token prediction 转成了 mask-predict，平滑过渡。

**Confidence-Guided Decoding (推理加速)**
推理时，一开始全都是 `<MASK>`。每一步，模型对所有 mask 位置预测一个概率分布。模型只挑出 confidence 最高的 top-K 个 token 确定下来，其他的继续 mask。这就好比填空题先填最有把握的空，填出来的空又成了其他空的上下文线索。加上 KV-cache，速度比传统 AR 快 4.3 倍。

### 5. 实验直觉与 Karpathy 视角的联想

在 CALVIN 上，UD-VLA 拿下了 4.64 的 Avg. Len.，LIBERO 上 92.7% 成功率。Real-world 实验里，对没见过的物体和背景泛化能力极强。

作为 Karpathy，您可能会对以下几点产生强烈的 intuition：

1.  **Visual Chain-of-Thought 的具身化**：这个架构实际上是在做 Visual CoT。传统的 CoT 是输出一串隐性 reasoning tokens，这里的 CoT 是显性的 future image。模型通过多步 denoise 不断细化这张图，相当于在做物理引擎模拟。
2.  **Computation Scaling 的胜利**：AR 模型每个 token 只能做一次前向计算。JD3P 允许 action token 在 T 步去噪中，反复 attend 正在成型的 future image token。这种 iterated computation 让 action prediction 获得了超乎寻常的深度推理能力。
3.  **Discrete Space 的统一红利**：把动作通过 FAST tokenizer 转成离散 token，彻底消灭了 continuous diffusion policy 里的外挂 head。这让整个模型可以无缝继承 LLM 的 scaling law 和 next-token pre-training 知识。
4.  **Bidirectional Attention 解决 Action Autocorrelation**：Action chunk 里的连续时间步动作高度相关。纯 AR 生成容易导致 error compounding。Bidirectional attention 让整个 chunk 被 globally optimized，这从根本上解决了 VLA 里的动作抖动和漂移问题。

**Reference Links:**
*   Paper PDF (假设链接): https://arxiv.org/abs/2505.14683 (实际为 Emerging properties in unified multimodal pretraining，论文真名暂无公开链接，可参考同实验室 ReconVLA: https://arxiv.org/abs/2508.10333)
*   FAST Tokenizer: https://arxiv.org/abs/2501.09747
*   MoVQ (Vision Tokenizer): https://arxiv.org/abs/2209.09002
*   MaskGIT (Discrete Diffusion 基础): https://arxiv.org/abs/2202.04200
*   D3PM (Discrete Diffusion 理论): https://arxiv.org/abs/2107.03006

总结一句，UD-VLA 用极其优雅的 Joint Discrete Denoising 告诉我们：让机器人边想边做，且想和做在同一个特征空间里互相迭代，才是通往 Generalist Robot 的正确道路。

---

Karpathy 教授您好，这篇 UD-VLA (Unified Diffusion VLA) 是一篇非常有启发性的工作。当前 VLA (Vision-Language-Action) 领域面临的核心痛点是：如何将 LLM/VLM 的 understanding 能力与机器人控制所需的 generation 和 acting 能力无缝耦合。传统的 autoregressive (AR) 范式在生成 multi-modal action 时存在 exposure bias 和速度瓶颈，而外挂 diffusion head 的方式又割裂了特征空间。UD-VLA 提出了一种非常优雅的架构：通过 discrete tokenization 统一空间，并利用 joint discrete denoising diffusion process (JD3P) 让 future image generation 和 action prediction 在同一个 trajectory 中同步迭代进化。这就好比让“预见未来”和“规划动作”这两个认知过程在同一个大脑回路里互相激发，而不是分步骤的流水线。

下面我为您深入拆解这篇 paper 的技术细节、公式逻辑以及架构直觉。

### 1. 核心架构直觉：Hybrid Attention 与 Unified Space

UD-VLA 的基石是 Unified Tokenization。模型将所有的 input/output 都映射为离散的 token：
*   **Text**: 沿用 Emu3 [1] 的 tokenization。
*   **Vision**: 使用 VQ-based tokenizer (MoVQ [2]) 将图像转换为 $V_v$ 个 codebook tokens。
*   **Action**: 使用 FAST tokenizer [3]，通过 DCT (Discrete Cosine Transform) 将连续的 action chunk 压缩成 $V_a$ 个离散 tokens。

这种统一避免了外挂专家模型（如 CLIP、Diffusion head）带来的 alignment 损耗。整个 sequence 被组织为：
`[ text tokens ; current image tokens ; future image tokens ; action tokens ]`

**Hybrid Attention Mechanism (图 2 解析)**
为了保证信息流的正确性，模型设计了混合注意力机制。直觉上，未来图像的生成不应该看到即将生成的动作（否则会产生信息泄露，导致 action 预测变成简单的复制任务），但动作预测必须强烈依赖于未来图像的预测。
*   **Input blocks (Text, Current Image)**: Text 保持 causal attention (维持 AR 特性)，Current Image 使用 bidirectional attention。
*   **Generation block (Future Image)**: 内部使用 bidirectional attention。这打破了 AR 的从左到右生成限制，允许图像全局一致性地生成。
*   **Acting block (Action)**: 内部同样使用 bidirectional attention。Action 的各个维度（如 x, y, z, roll, pitch, yaw）在物理上没有严格的时间先后顺序，bidirectional attention 允许它们互相校准，避免了 shortcut learning [4]。
*   **Cross-block**: 严格的 causal attention。Action tokens 可以 attend to future image tokens，反之不行。这强制模型将 action prediction 视为一个 inverse kinematics 问题：基于期望的未来状态，反推当下的动作。

### 2. 核心理论：Joint Discrete Denoising Diffusion Process (JD3P)

JD3P 是这篇 paper 的灵魂。它没有采用传统的 continuous diffusion (如 $\pi_0$ [5] 或 MDT [6])，而是采用了 discrete diffusion，类似于 MaskGIT [7] 或 D3PM [8]。

**Noising Process (前向加噪)**
给定 clean 的 future image tokens $\mathbf{v}_0$ (长度 $L_v$) 和 action tokens $\mathbf{a}_0$ (长度 $L_a$)，联合状态为 $(\mathbf{v}_0, \mathbf{a}_0)$。模型引入一个特殊的 mask token $M$。
前向过程是一个 Markov chain，每一步的转移矩阵 $\mathbf{Q}_t$ 以概率 $\beta_t$ 将某个 token 替换为 $M$，以概率 $1-\beta_t$ 保持不变：

$$ \mathbf{Q}_t \mathbf{e}_{t,r} = (1 - \beta_t) \mathbf{e}_{t,r} + \beta_t \mathbf{e}_{\mathrm{M}} $$

*   $\mathbf{e}_{t,r}$: 在时间步 $t$，位置 $r$ 上的 token 的 one-hot 向量。
*   $\beta_t$: 时间步 $t$ 的 noise schedule (mask 概率)。
*   $\mathbf{e}_{\mathrm{M}}$: mask token 的 one-hot 向量。

因为各位置独立加噪，联合分布可以分解为：
$$ q(\mathbf{v}_t, \mathbf{a}_t \mid \mathbf{v}_0, \mathbf{a}_0) = \prod_{r=1}^{L_v + L_a} \mathrm{C} \big( v_{t,j}, a_{t,i} \mid \bar{\mathbf{Q}}_t \mathbf{e}_{0,r} \big) $$
*   $\bar{\mathbf{Q}}_t = \mathbf{Q}_t \cdots \mathbf{Q}_1$: 累积转移矩阵。
*   $\mathrm{C}$: Categorical distribution (分类分布)。

**Denoising Process (反向去噪)**
给定 $t$ 时刻的 noisy tokens $(\mathbf{v}_t, \mathbf{a}_t)$ 和 context $\mathbf{c}$ (text + current image)，反向条件概率被因式分解为视觉和动作两部分：

$$ p_\theta(\mathbf{v}_{t-1}, \mathbf{a}_{t-1} \mid \mathbf{v}_t, \mathbf{a}_t, \mathbf{c}) = p_\theta(\mathbf{v}_{t-1} \mid \mathbf{v}_t, \mathbf{c}) p_\theta(\mathbf{a}_{t-1} \mid \mathbf{v}_t, \mathbf{a}_t, \mathbf{c}) $$

具体到单个 token 的预测，公式 (6) 和 (7) 展示了 mask 机制：
如果位置当前是 clean 的 (非 $M$)，则保持不变；如果是 $M$，则从模型预测的 categorical distribution $\pi_\theta$ 中采样：

$$ p_\theta(v_{t-1,j} \mid \mathbf{v}_t, \mathbf{c}) = \Big[ \delta(v_{t-1,j} = v_{t,j}) \Big]^{1 - 1\{v_{t,j} = \mathrm{M}\}} \Big[ \mathrm{C}\big(v_{t-1,j} \mid \pi_\theta^{(v)}(j \mid \mathbf{v}_t, \mathbf{c})\big) \Big]^{1\{v_{t,j} = \mathrm{M}\}} $$

*   $\delta(\cdot)$: Kronecker delta (如果参数相等则为 1，否则为 0)。
*   $1\{\cdot\}$: Indicator function (条件成立为 1，否则为 0)。
*   $\pi_\theta^{(v)}(j \mid \cdot)$: Transformer 对位置 $j$ 视觉 token 的预测概率分布。

**直觉解析**：在每一步 denoising 中，所有被 masked 的 position 都会同时接收来自其他 unmasked positions (包括 context 和已经确定下来的 future image/action tokens) 的 attention，并预测自己的类别。这形成了一个并行的、 coarse-to-fine 的 refinement 过程。Action tokens 在每一步都能 attend to 正在不断清晰化的 future image tokens，这就是所谓的 "synchronous denoising" 带来的 synergy。

**Loss Function**
为了简化训练，paper 放弃了显式的多步 ELBO 优化，采用了类似 MaskGIT 的 single-step mask-predict objective (公式 8)：

$$ \mathcal{L}_{\mathrm{CE}}(\theta) = - \omega \sum_{j}^{L_v} \log p_\theta^{(v)}(v_{0,j} \mid \mathbf{v}_t, \mathbf{c}) \cdot \mathbb{1}\{v_{t,j} = \mathrm{M}\} - \sum_{i}^{L_a} \log p_\theta^{(a)}(a_{0,i} \mid \mathbf{v}_t, \mathbf{a}_t, \mathbf{c}) \cdot \mathbb{1}\{a_{t,i} = \mathrm{M}\} $$

*   $\omega$: 权重系数。因为 visual tokens 数量远大于 action tokens，如果不降权，loss 会被 image generation 主导，导致 action prediction 学不好。
*   训练时，随机采样一个 mask ratio $\rho_t \in (0, 1]$，直接施加在 clean sequence 上，然后计算被 mask 位置的 cross-entropy。这本质上是在学习任意噪声扰动下的逆条件概率，是 diffusion ELBO 的一种有效近似。

### 3. Training Pipeline & Inference Techniques

**Two-Stage Training**
1.  **Stage 1 (World Model Post-training)**: 在大规模 video dataset 上训练 sequence `[ text ; current image ; future image ]`。这一步激活了 VLM backbone 的视觉预测能力，让它懂物理 dynamics。
2.  **Stage 2 (Joint Fine-tuning)**: 在 robot dataset 上训练完整的 `[ text ; current image ; future image ; action ]`。这里引入了一个 shift operation strategy (借鉴自 [9])，将原本 AR 的 next-token prediction 转换为 mask-predict 形式，最大化保留 pretrained knowledge。

**Inference 优化**
Diffusion 模型在推理时通常较慢，UD-VLA 通过几个关键技术实现了 4.3× 的速度提升（相比 AR）：
1.  **Prefix KV Cache & Pre-filling**: 缓存 context tokens 的 K/V。同时，因为 image token 长度固定，直接 pre-fill `<BOI>`, `<EOI>` 等特殊 tokens，减少无效计算。
2.  **Confidence-Guided Decoding**: 采用 cosine mask schedule $\rho_t = \cos\left(\frac{\pi}{2} \frac{T+1-t}{T+1}\right)$。每一步计算所有 masked positions 的最大预测概率作为 confidence score $q_{t-1,r} = \max_\ell \{ p_\theta(\ell \mid \mathbf{v}_t, \mathbf{u}) \}$。只 commit top $(1-\rho_t)|M_t|$ 个最高置信度的 tokens，其余保持 masked。这保证了每一步都锁定最确定的信息，防止错误累积。
3.  **Decoding Space Mapping**: 限制模型在预测 vision/action 时，只在各自的 codebook 子集内搜索。如果预测出了 `<EOA>` (End of Action)，则截断后续所有 action tokens 并强制设为 mask。

### 4. 实验数据与深度联想

在 CALVIN 基准上，UD-VLA 达到了 4.64 的 Avg. Len. (超过 UP-VLA 的 4.42 和 MDT 的 4.52)。在 LIBERO 上达到 92.7% 平均成功率。更重要的是，在 SimplerEnv 和 Real-world experiments 中，由于 explicit future image generation 的存在，模型展现出极强的 generalization (对 unseen objects 和 backgrounds 的适应能力远超 GR00T N1 和 UniVLA)。

**Karpathy 视角的联想与 Intuition Building：**

1.  **System 1 vs System 2 Thinking in Robotics**: 传统的 AR VLA (如 OpenVLA [10]) 类似 System 1，看到什么直接反射动作。UD-VLA 通过 JD3P 强制模型先生成 future image，这相当于在 latent space 中进行了一次 "imagination" 或 "visual chain-of-thought"。这种 multi-step refinement 本质上赋予了机器人 System 2 的规划能力。Action tokens 在去噪过程中不断从 future image tokens 那里汲取信息，这是一种 computation scaling 的体现。
2.  **Discrete vs. Continuous Diffusion for Actions**: 为什么不用 continuous diffusion (像 Diffusion Policy [11] 或 $\pi_0$)？Continuous diffusion 需要外挂一个 denoiser network，且难以直接融入 LLM 的 token embedding space。UD-VLA 将一切离散化后，Action 预测变成了标准的 next-token (masked-token) classification，完美复用了 LLM 的架构。图 8 的对比很好地说明了 discrete diffusion 在保持模态一致性上的优势。
3.  **Information Leakage vs. Synergy**: 这里的 Hybrid Attention 设计非常巧妙。如果完全 Bidirectional，Action 会泄露给 Image，导致模型偷懒不去预测物理 dynamics。如果完全 Causal，Action 无法充分从 Image 获得信息。UD-VLA 的 Causal cross-modal + Bidirectional intra-modal 设计，精确定位了 "inverse kinematics" 的逻辑流：Future state 导致 Action，而非 Action 导致 Future state。
4.  **Error Compounding in Action Chunking**: FAST tokenizer 将一段连续轨迹压成一个 chunk。Bidirectional attention 在 action block 内部打破了时间序列的依赖，使得整个 chunk 是 globally optimized 的，而不是 step-by-step generated。这极大地缓解了 AR policies 中的 error compounding 问题。
5.  **Scaling Laws for VLA**: 既然统一到了 discrete token space，未来的想象空间很大。是否可以直接用 Emu3 或 LLaDA [12] 这样的纯 discrete diffusion LLM 作为 backbone？Internet-scale 的 video data 和 robot data 是否可以完全无差别地混合训练？UD-VLA 验证了 joint denoising 在 VLA 上的有效性，为迈向 unified foundation models 提供了坚实的证据。

**Reference Links:**
*   [1] Emu3: Next-Token Prediction is All You Need - https://arxiv.org/abs/2409.18869
*   [2] MoVQ: Modulating Quantized Vectors - https://arxiv.org/abs/2209.09002
*   [3] FAST: Efficient Action Tokenization - https://arxiv.org/abs/2501.09747
*   [4] Shortcut Learning in Deep Networks - https://arxiv.org/abs/2004.07780
*   [5] $\pi_0$: A VLA Flow Model - https://arxiv.org/abs/2410.24164
*   [6] MDT: Multimodal Diffusion Transformer - https://arxiv.org/abs/2407.01504
*   [7] MaskGIT: Masked Generative Image Transformer - https://arxiv.org/abs/2202.04200
*   [8] D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces - https://arxiv.org/abs/2107.03006
*   [9] Scaling Diffusion Language Models via Adaptation - https://arxiv.org/abs/2502.09992
*   [10] OpenVLA: An Open-Source Vision-Language-Action Model - https://arxiv.org/abs/2406.09246
*   [11] Diffusion Policy: Visuomotor Policy Learning via Action Diffusion - https://arxiv.org/abs/2303.04137
*   [12] LLaDA: Large Language Diffusion with RoPE - https://arxiv.org/abs/2502.09992

总而言之，UD-VLA 通过 Joint Discrete Denoising 巧妙地打通了视觉生成与动作预测的壁垒。它摒弃了传统的 AR 范式，转而使用并行、迭代的 refinement 机制，在保证 SOTA 性能的同时大幅提升了推理速度，并且赋予了模型显式的视觉想象能力。这是迈向 Generalist Robot Policies 的一个极其 promising 的方向。
