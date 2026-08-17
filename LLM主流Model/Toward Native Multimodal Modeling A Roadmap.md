---
source_pdf: Toward Native Multimodal Modeling A Roadmap.pdf
paper_sha256: dc45f3d27ac9759c7a350c37dc7105e28db380af536bb826d626b83f5b9dd16a
processed_at: '2026-08-12T16:42:10-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，我们用最直白的话来拆解这篇 paper。这篇 roadmap 其实就在讲一个核心故事：**我们的 multimodal model 怎么从“用胶带把不同的模态粘在一起”，演化到“从出生起就是一个原生的多模态大脑”。**

下面我尽量用 build intuition 的方式，把这篇论文最核心的洞察、那些复杂的公式背后的物理直觉，以及工业界的 trade-off 给你讲透。

---

### 1. 架构的演化：从“盲人摸象”到“全息共生”

论文定义了三种 fusion 范式，你可以把它想象成大脑处理感官信息的进化史。

**Late-Fusion (胶带拼接，非 Native)**
$$\mathcal{F}_{\text{late}} = \mathcal{G}\big(\text{LLM}(\{\mathcal{P}_i(E_i(m_i))\})\big)$$
变量解释：$m_i$ 是图像/音频输入，$E_i$ 是独立的预训练 encoder，$\mathcal{P}_i$ 是个浅层 projector，LLM 是冻结的语言模型，$\mathcal{G}$ 是输出头。
人话：这就是 LLaVA 的做法。大脑（LLM）是冻结的、盲的。眼睛（Encoder）看了图像，压缩成一串特征，通过细管子（Projector）喂给大脑。大脑只能看到被嚼碎的语义摘要，根本不知道原始的视觉细节长什么样。生成图像时，还得在外面外挂一个 Diffusion head $\mathcal{G}$。理解和生成完全是割裂的两个模块。

**Mid-Fusion (打通任督二脉，半 Native)**
$$\mathcal{F}_{\text{mid}} = \text{Backbone}\big(\mathcal{C}(E_1(m_1), \ldots, E_n(m_n))\big)$$
变量解释：$\mathcal{C}$ 是 cross-attention 或者深层 adapter，Backbone 是联合训练的多模态主干。
人话：大脑终于不冻结了，眼睛（Encoder）和大脑一起联合训练。梯度可以流回 Encoder 了。这派以 CogVLM、Qwen2.5-VL 为代表。它们通过 $\mathcal{C}$ 操作把视觉特征深度注入到语言模型的中间层。大脑开始懂得视觉的细节了。但是，视觉和语言依然有明确的架构边界，上游还是有独立的 $E_i$，结构上依然是不对称的。

**Early-Fusion (万物皆 Token，终极 Native)**
$$\mathcal{F}_{\text{early}} = \text{Transformer}\big(\tau(\mathcal{U}_i m_i)\big)$$
变量解释：$\tau$ 是统一的 tokenizer，把所有东西变成 token。
人话：这是 Chameleon、Emu3.5 的路线。不分什么视觉 encoder、语言 encoder了。图像、音频、文本，从宇宙诞生那一刻起（从输入第一步起），就全部被映射成同一种 token，扔进同一个 Transformer 里算。在这个世界里，图像 token 和文本 token 完全平等，没有谁是谁的附庸。这就是论文说的 "born-native"。

---

### 2. 输入输出对偶性：模型能干什么

论文按输入输出模态流，把 NMM 分成三类。这维度和上面的 fusion depth 是正交的。

*   **Multi-to-Text (M2T)**：看图说话、听音答题。输入图片/音频/视频，输出只有文本。这是目前绝大多数 MLLM 的状态。瓶颈在于如何把多模态信息 ground 到语言空间。
*   **Multi-to-Target (M2G)**：文生图、文生视频。输入提示词，直接输出特定模态。现在的原生模型抛弃了外挂 Diffusion head 的做法，直接从主干的 hidden representation 里 decode 出视频 voxel 或音频波形，保证了极高的语义一致性。
*   **Multi-to-Multi (M2M)**：终极形态。输入图像，输出音频；输入视频，输出图文交织的教程。理解和生成在同一个网络里无缝共存。这就是未来世界模型的样子。

---

### 3. 训练的核心洞察：为什么 Early-Fusion 这么难炼？

这是论文最有 teaching value 的地方。它指出训练配方和架构是死死绑定的。Early-fusion 不是简单把模态拼一起就行，它会引发数值和梯度的灾难。

#### 3.1 Late-Fusion 训练：躺平
因为 encoder 冻结，LLM 冻结，只练 projector。一个全局 learning rate，单一的 text cross-entropy loss，毫无稳定性压力。代价就是能力上限低。

#### 3.2 Mid-Fusion 训练：小心翼翼地解冻
一旦梯度流到 encoder，同一个 learning rate 就会搞崩一切。对 encoder 太大，对 LLM 太小。所以必须用 differential learning rates（比如 CogVLM 给 encoder 用 LLM 1/10 的学习率），还要设计 resolution curriculum（先低分辨率练，再高分辨率）。理解和生成还要用 decoupled loss 分别计算。这就像同时驾驭两匹野马。

#### 3.3 Early-Fusion 训练：在爆炸边缘走钢丝
所有模态从 step 0 开始一起训练，直接塞进同一个 Softmax 算 cross-entropy。这会带来两个致命问题：

**问题一：Logit 爆炸与表征竞争**
文本 token 信息密度极高，图像 token 极度稀疏。它们在同一个 Softmax 里竞争，大规模数据下会让 partition function $Z$（所有 logit 的指数和）指数级爆炸，导致梯度发散。
解法：Z-loss 正则化。
$$\mathcal{L}_{z\text{-loss}} = 10^{-5} \cdot \log^2 Z$$
直觉：强迫把 $Z$ 拉回有界范围。同时必须上 QK-Norm，在算 attention 的 dot product 前对 Query 和 Key 做 layer norm，防止高熵模态吞掉低熵模态的表征。Chameleon 的 ablation 证明，没这俩玩意，训练到 20% 就直接 diverge。这是 early-fusion 的生死线。

**问题二：模态配比的致命性**
Mid-fusion 有独立 loss head，batch 里图像和文本比例只影响 aggregate gradient 的 magnitude。但在 Early-fusion 的 unified loss 下，模态比例直接决定 gradient direction！
Chameleon 明确警告：如果模态配比不平衡，模型会学到 degenerate unconditional prior（退化成无视条件只生成噪声的先验）。所以必须极其精细地调度 batch 里的 text/image 比例，这叫做 "Modality-mixture scheduling"，它是 early-fusion 独有的命门。

---

### 4. RL 的灾难与 MOPD 的救赎

在 SFT 和 RL 阶段，架构同样决定命运。论文最精彩的地方在于它剖析了 Early-fusion RL 的 failure mode。

**Late-Fusion RL**：只调那个生成文本的头，简单粗暴。
**Mid-Fusion RL**：分别调理解和生成路径，互不干扰。
**Early-Fusion RL**：全模型更新，灾难降临。

在 unified softmax 下，语言先验和视觉证据在同等地位上竞争。由于语言先验更“平滑”、更容易拟合，naive RL 会让模型走捷径，靠文本胡编乱造来拿 reward，完全无视图像 evidence，这就是 **Visual-grounding hacking**。

更头疼的是，如果你针对数学做一次 RL，针对 code 做一次 RL，针对 agentic 做一次 RL，checkpoint 之间会互相 trade-off，练好数学毁了 code，这就是 **See-saw effect**。

**解法：On-Policy Distillation (OPD / MOPD)**
论文重点推崇 MiMo-V2.5 的 MOPD 策略。公式如下：
$$\hat{A}_{i,t} = \text{sg}\left[\log \frac{\pi_{\text{teacher}}(y_{i,t} \mid x, y_{i,<t})}{\pi_{\text{student}}(y_{i,t} \mid x, y_{i,<t})}\right]$$
变量解释：$\hat{A}_{i,t}$ 是 student 在第 $t$ 个 token 的 advantage。$\text{sg}$ 是 stop-gradient。$\pi_{\text{teacher}}$ 是专家模型的概率分布。
直觉：不直接用 rule-based reward 了。搞一个专家池，有专门练数学的 teacher，练视觉的 teacher，甚至包含 student 自己的一个 frozen snapshot。student 每生成一个 token，就去看看专家们在这个 token 上的概率分布，通过 KL 散度对齐。同时，算上 ORM（Outcome Reward Model）的结果 $\hat{A}_{i,t}^{\text{OPD}} + \alpha \tilde{A}_{i,t}^{\text{ORM}}$。那个 frozen snapshot 充当 anchor，防止 student 被外面的专家带偏到完全陌生的分布去。这其实是用 multi-teacher 把能力平滑地蒸馏进来，避开了 unified softmax 下的 reward hacking。

---

### 5. 工程与部署的深层矛盾

当你真的要把 early-fusion 模型跑起来，工程问题极其棘手。

*   **Sequence Explosion**：一张高分辨率图就是几千个 token，一段长视频上百万 token。Attention 的 $O(N^2)$ 直接把显存撑爆。目前的解法是动态分辨率（Q-Zoom）和 Visual Token 压缩，ResAdapt 实验证明砍掉 90% 的 visual token 反而能让长视频推理提升 15%，因为去除了背景噪声。
*   **Causal vs Bidirectional 冲突**：纯 Early-fusion 统一用 causal mask，但生成图像时用 bidirectional attention 效果更好（因为扩散需要全局上下文）。Transfusion 试图混合，但这破坏了 FlashAttention 的内存对齐假设，导致底层 kernel 失效。现在的解法是 FlexAttention 和 FlashMask，用 JIT 编译动态生成融合计算图，在 causal 和 bidirectional block 间无缝切换。这是底层 tensor operation 的解放。
*   **Full-Duplex 实时交互**：真原生的语音对话，必须边听边说。Moshi、MiniCPM-o 在架构层面做 streaming state prediction 和 KV cache 管理，而不是离线跑完一段再生成下一段。这要求模型 thinking-in-speaking，后台维护 hidden reasoning token 的同时并发输出 audio token。

---

### 6. 总结：我的 Critical Insights

1.  **Fusion-Coupled Training Signature 是这篇论文的精髓**。它告诉你，不要孤立地看架构。当你看到 Chameleon，你要立刻问：它用了 z-loss 吗？QK-Norm 在哪？batch 里 text/image token 比例怎么调？因为这些训练配方是被它的 early-fusion 架构“锁死”的必然选择。
2.  **M2M 对称建模是通向世界模型的关键**。现在的 unified model 很多还是用 hybrid loss（Transfusion: $\mathcal{L} = \mathcal{L}_{\text{LM}} + 5 \cdot \mathcal{L}_{\text{DDPM}}$），文本用 AR，图像用 Diffusion。真正的终局是同一个 probabilistic objective、同一个 token space 同时搞定理解和生成，且不退化。这也是为什么 TUNA-2 这种把 raw pixel 喂进去、抛弃 CLIP encoder 的 encoder-free 路线值得深挖。
3.  **Evaluation Gap 严重滞后**。现在的 benchmark 仍在静态评单一模态。但 native model 的核心能力是交互、是 timing、是 when to respond。ThinkStream 提出的 Watch-Think-Speak 协议才是对真交互智能体的正确评估方向。

这篇 roadmap 划定了 NMM 领域的 vocabulary 和边界，明确了 mid-fusion 只是过渡，early-fusion 才是归宿。但在 early-fusion 范式下，如何平衡 RL 中的多目标冲突、如何解决 deployment 时的 sequence explosion，依然是完全开放的工业级难题。

**References**:
*   [Chameleon (Early-fusion 先驱)](https://arxiv.org/abs/2405.09818)
*   [Transfusion (AR + Diffusion 混合)](https://arxiv.org/abs/2408.11039)
*   [Emu3.5 (纯 NTP 统一)](https://arxiv.org/abs/2510.26583)
*   [Moshi (全双工音频原生)](https://arxiv.org/abs/2410.00037)
*   [Janus-Pro (解耦理解与生成)](https://arxiv.org/abs/2501.17811)
*   [FlexAttention (解决 mask 冲突)](https://arxiv.org/abs/2412.05496)
*   [FlashMask (底层加速)](https://arxiv.org/abs/2410.01359)
*   [ThinkStream (流式评估)](https://arxiv.org/abs/2603.12938)
*   [ResAdapt (视觉 token 自适应)](https://arxiv.org/abs/2603.28610)
*   [TUNA-2 (Encoder-free)](https://arxiv.org/abs/2604.24763)

---

# Toward Native Multimodal Modeling: A Roadmap — 深度解读

这篇由 Tencent Youtu Lab 联合 Tsinghua、HKU 等机构发布的 roadmap 论文，试图回答一个在 2025-2026 年 NMM (Native Multimodal Modeling) 爆发期被反复追问但缺乏形式化的问题：**到底什么算 "native"？late/mid/early-fusion 的边界在哪里？工业级 NMM 的全栈 trade-off 长什么样？**

论文的野心不仅仅是 survey，而是想给社区一个**形式化 taxonomy + 工程实践 playbook + 未来方向预测**的三合一文档。下面我尽量按你喜欢的"build intuition"的方式来拆解，公式都讲清楚每个符号的含义。

---

## 1. 论文的 Motivation 与核心主张

论文开篇就抛出 LLM 的根本瓶颈：text-only interface 让模型对真实世界的感知是 indirect 的，缺乏 grounding。而通往 AGI / world model 的关键一跃必须经过 multimodal modeling。

但问题在于，过去几年的 "multimodal" 其实绝大多数是 **late-fusion**：拿一个预训练好的 vision encoder (CLIP/SigLIP) 通过一个 shallow projector 接到 frozen LLM 上，最典型的就是 LLaVA 系列、DeepSeek-VL、Qwen-Image。这种架构有几个本质问题：

1. Backbone 对 raw sensory signals 是 blind 的，只能看到经过 projector 压缩后的 high-level features；
2. Encoder 在训练中被冻结，无法 adapt 到 language objective，cross-modal capacity 有上限；
3. 输出侧必须 graft 一个 decoder head（比如 SD3、CosyVoice），understanding 和 generation 是两个孤岛。

论文提出 NMM 作为对这种 modular assembly 的反叛，强调 multimodal synergy 应该是 **intrinsic architectural property**，而不是 post-hoc 拼接。然后形式化定义了两种 native regime。

---

## 2. Nativity 的形式化定义

这是论文的核心贡献之一。设输入模态集合 $\mathcal{M} = \{m_1, m_2, \ldots, m_n\}$，$E_i$ 是 modality-specific encoders，$\mathcal{P}_i$ 是 projection/alignment layers，$\tau$ 是 unified tokenization operator，$\mathcal{G}$ 是 output head。

### 2.1 Late-Fusion (非 native, baseline)

$$\mathcal{F}_{\text{late}} = \mathcal{G}\Big(\text{LLM}\big(\{\mathcal{P}_i(E_i(m_i))\}_{i=1}^{n}\big)\Big)$$

含义：每个模态 $m_i$ 先经过独立 encoder $E_i$ 提取特征，再通过 projector $\mathcal{P}_i$ 投影到 LLM 的 hidden space，全部串接后喂给 frozen LLM，最后由 output head $\mathcal{G}$ 生成。Backbone 本身对原始信号是盲的，只看到一个被预先压缩的语义摘要。

### 2.2 Mid-Fusion (第一级 native)

$$\mathcal{F}_{\text{mid}} = \text{Backbone}\Big(\mathcal{C}\big(E_1(m_1), \ldots, E_n(m_n)\big)\Big)$$

其中 $\mathcal{C}$ 是 cross-modal alignment/injection operator（典型实现：cross-attention 或 deeply stacked adapters）。

关键点：gradient 第一次真正流到 encoder 了，backbone 是 **Joint Multimodal Backbone**，不再是 frozen LLM。代表：CogVLM (Visual Expert)、Qwen-Audio、Qwen2.5-VL、Qwen3-VL、InternVL-3.5、Kimi K2.5、GLM-5V-Turbo。

但这里仍有 **explicit modality-aware boundaries**——上游还有独立的 $E_i$，结构上是不对称的，所以只是"transitional native"。

### 2.3 Early-Fusion (终极 native)

$$\mathcal{F}_{\text{early}} = \text{Transformer}\Big(\tau\big(\mathcal{U}_i m_i\big)\Big)$$

其中 $\tau$ 是 unified tokenization operator，$\mathcal{U}_i$ 表示所有模态被统一映射。**完全 bypass 独立 frozen encoders**，所有模态从第一步就被映射到同一个 shared embedding space，由同一个 Transformer 处理。代表：Transfusion、Chameleon、AnyGPT、Emu3.5。

Intuition：这就是论文 Figure 1 里所谓的 "born-native" 状态，所有 modality 是 fundamentally equivalent tokens，没有 perceptor / renderer 的概念区分。

---

## 3. Input-Output Duality 三分类

论文用 input-output modality flow 这个正交维度把 NMM 分成三类。注意这个维度和 fusion depth 是正交的，所以每个 functional category 都可以既有 mid-fusion 也有 early-fusion 代表。

### 3.1 Multi-to-Text (M2T) — 不对称理解

$$\mathcal{F}_{M2T}: \mathcal{M} \to T, \quad T \in \mathcal{M}$$

输入任意交错的多模态流，输出只有 text。优化瓶颈在 cross-modal alignment 和 perceptual grounding，不在 textual synthesis。代表：MiniCPM-V-4.6、Nemotron3-Nano-Omni、MiMo-V2.5、Qwen3.6、Gemma-4、Kimi K2.5、GLM-5V-Turbo、Llama-4-Scout/Maverick、InternVL-3.5、Qwen3-VL、Qwen2.5-VL、CogVLM、Video-LLaVA、Qwen-Audio。

### 3.2 Multi-to-Target (M2G) — 不对称生成

$$\mathcal{F}_{M2G}: \mathcal{M} \to y_k, \quad y_k \in \mathcal{M}$$

输出是单一非文本 modality（video voxel、audio waveform 等）。关键优势：output pathway 直接从 native hidden representation decode，语义一致性比 graft 一个 SD head 强很多。代表：HiDream-O1-Image、OmniVoice、LTX-2.x、Ming-Flash-Omni-2.0、MiniCPM-o-4.5、Kling-Omni、HunyuanVideo-1.5、Qwen3-Omni、Wan2.2-T2V、Seedream3.0。

### 3.3 Multi-to-Multi (M2M) — 对称统一

$$\mathcal{F}_{M2M}: \mathcal{M}_{in} \rightleftharpoons \mathcal{M}_{out}, \quad \mathcal{M}_{in} \subseteq \mathcal{M}, \mathcal{M}_{out} \subseteq \mathcal{M}$$

输入和输出都是任意 modality 组合。理解与生成在同一个 Transformer 里 co-exist。这是论文认为的终极形态。代表：Moshi、Emu3.5、BAGEL-7B、OneCAT-3B、Show-o2、Janus-Pro、TUNA-2、Mamoda2.5、LLaDA2.0-Uni、LongCat、SenseNova-U1、Lance。

---

## 4. Model Architecture 的技术深挖

这部分论文按 M2T/M2G/M2M 三个 category 拆解，每个 category 内列出核心技术挑战和当前 SOTA 的解决路径。我重点挑几个有 teaching value 的细节讲。

### 4.1 M2T — Image Comprehension 的三个挑战

**Modality Unification**: 当前 SOTA 主要走 continuous projection route（避免 discrete quantization 的信息损失）。

- *Vision-Encoder-Based Fusion*：Llama-4-Scout/Maverick 用增强 vision encoder 把图像转成 continuous patch embeddings，从最早的 transformer layer 就开始 joint 处理；Kimi K2.5 用 MoonViT 编码图像后送进 sparse MoE backbone；Gemma-4-31B 用 hybrid-attention 把 continuous soft token 和 text interleave。
- *Unified Stream Mapping*：Qwen3.6 把所有 modality 当成 unified token stream 喂进单一 transformer；Nemotron3-Nano-Omni 用 compact unified architecture 实现低延迟 cross-modal alignment。

**Multi-image Reasoning**: visual token 会 overwhelm attention 导致 attention saturation + quadratic compute。四条技术路线：
1. *Extreme Visual Compression*：Kimi K2.5、InternVL-3.5 用 Visual Resolution Router + temporal pooling 压缩 visual token 数量。
2. *Deep Feature Alignment*：Qwen3-VL/Qwen2.5-VL 用 deep-stack multi-level feature injection；CogVLM 保留 dedicated Visual Expert module。
3. *Advanced Positional Encoding*：Llama-4、Gemma-4-E4B 用 iRoPE / p-RoPE 在 interleaved sequence 上稳定 retrieval。
4. *Perception-Reasoning Decoupling*：GLM-5V-Turbo、MiMo-V2.5 实现 "thinking mode"，把 raw visual perception 和后续逻辑推理解耦以降低 latency 和 hallucination。

**Multi-scale Encoding**: 处理非标准 aspect ratio 时不丢 fine-grained detail。四条策略：
1. *Structure-Aware Tiling*：InternVL-3.5、MiniCPM-V-4.6 把高分辨率输入切成 dynamic tiles，附带 structural identifiers 帮模型从 1D token 重建 2D layout。
2. *Dimension-Decoupled Positional Encoding*：Qwen3-VL、GLM-5V-Turbo 用 2D-RoPE，把坐标分解成 x 和 y 分量。
3. *Semantic-Driven Resampling*：InternVL-3.5 用 perceiver-based 架构自适应把背景 patch 压缩到固定 latent space。
4. *Resolution-Agnostic Projection*：Gemma-4-31B、Llama-4-Maverick 绕过 fixed-grid 约束。

### 4.2 M2T — Audio Comprehension

**Semantic-Acoustic Conflict**: 连续 audio signal 与离散 textual semantics 本质不兼容。
- MiMo-V2.5 的 MiMo-Audio-Tokenizer 在 shared latent space 同时输出 semantic 和 acoustic features；RVQ 系统在 initial layers 优先保留 semantic structure，later layers refine acoustic detail。
- Gemma-4-E4B 直接处理 log-Mel spectrogram，用 Conformer-based audio encoder 输出 continuous embedding 保留完整 acoustic info。
- Nemotron-3-Nano-Omni 用 FastConformer encoder 提取 deep acoustic features，通过 2-layer MLP 投影到 language backbone。

**High Latency & Computation**: 
- Gemma-4-E4B 用 long frame duration 的 acoustic encoder，每秒音频压缩成少量 vector。
- Nemotron-3-Nano-Omni 实现 algorithmic-architectural co-optimization：log-mel spectrogram + 3 个 convolutional subsampling 层得到 8× temporal downsampling；TDT decoder 在推理时基于预测 token duration 动态跳帧；底层是 31B Mamba2-Transformer hybrid MoE，每 forward 只激活 3B 参数；Mamba2 的 linear complexity $O(N)$ 替代 attention 的 quadratic $O(N^2)$。

### 4.3 M2T — Video Comprehension

输入维度从 $H \times W$ 扩展到 $T \times H \times W$，引发三个 bottleneck：

**Computational Explosion**: video 每秒 token 数极冗余，且 attention cost 与 sequence length 成平方增长。
- *Compression & Feature Aggregation*：Kimi K2.5 把连续帧打包成 spatiotemporal volume，在 patch level 做 temporal averaging；GLM-5V-Turbo 在 encoder 用 3D conv 替代 2D conv，在 feature extraction 时沿时间轴 downsample。
- *Dynamic Token Allocation*：InternVL-3.5 的 Visual Resolution Router 给语义丰富 patch 分配 256 token，背景压缩到 64 token，整体冗余减半；Gemma-4-31B 让用户 per task 手动设置 token budget。

**Temporal & Logical Inconsistency**:
- *Temporal Coordinate Encoding*：Qwen3.6/Qwen3-VL 把 RoPE 分解成三个交错维度 (T, H, W)，每个 token 在 3D spatiotemporal 坐标下有唯一表示。
- *Explicit Time Tokens*：GLM-5V-Turbo 在 video-frame sequence 里显式插入 time token，让模型像读自然语言一样感知物理时间。

**Long-range Dependency**: 处理小时级 video 需要高效 working memory。
- *Modular Long-Term Memory*：InternLM-XComposer2.5 建独立 memory pool 压缩存储 perceptual video features 到 long-term memory bank，Q&A 时 on-demand 检索，支持无限长 streaming interaction。
- *Distributed Clustering*：Kimi K2.5 的 agent-swarm mode，中央 dispatcher 把长 video 任务分解给数百 specialized sub-agent 并行分析。

### 4.4 M2G — Image Generation

传统 workflow是 LLM 生成 prompt → standalone diffusion model。Native image generation 走 joint modeling of text + image 路线。两个核心挑战：

**High Visual Fidelity**: Ming-Flash-Omni-2.0 把 Transformer 和 Diffusion 在 shared latent space 结合，用 Mask-based Discrete Diffusion 作为统一 mask-aware architecture，学习 cross-modal token 的联合分布。Hidden layer 同时预测 next text token 和输出 continuous features 指导 image denoising。Unified self-attention 让 text 结构和 image 空间 layout 在早期 fusion 阶段就对齐。

**Compositional Controllability**: 严格遵循多 entity 交互和精确位置约束。
- Seedream3.0 通过 cross-modality RoPE 实现 spatial perception。
- HiDream-O1-Image 集成 coordinate-aware representation，把离散 layout instruction 直接 project 到 localized generation process。

### 4.5 M2G — Audio Generation

三个挑战：(i) Semantic-Prosody Alignment, (ii) Latency Control, (iii) Reasoning-Streaming Synergy。

**Semantic-Prosody Alignment**:
- LTX-2 用 RoPE 处理 audio，bidirectional cross-attention layer 捕捉触发 acoustic feature 的 transient dependency。
- CosyVoice 做 semantic-acoustic decoupling：supervised semantic tokenizer 控制 content，flow-matching module 渲染 timbre 和 emotion。

**Latency Control**:
- Qwen3-Omni 用 Multi-Token Prediction (MTP) 同时输出 residual codebook，配 Code2Wav renderer 做 frame-level streaming synthesis，实现 first-packet latency。
- GLM-4-Voice 用 Single-codebook，在 VQ bottleneck 引入 ASR encoder (Whisper-v3)。
- MiniCPM-o 4.5 优化 token density，把 audio 压缩到每秒极少 token 数，专为移动端带宽优化。

**Reasoning-Streaming Synergy**: Thinker-Talker 架构是当前 SOTA——高容量 Thinker 在后台做 long-form reasoning，轻量 Talker (如 OmniVoice) 做超低延迟 speech 输出。Mini-Omni-Reasoner 实现 thinking-in-speaking：维护 hidden reasoning token 的同时并发输出 audio token。

### 4.6 M2G — Video Generation

**Physics Understanding**: 生成 video 经常违反基本物理定律（物体飘浮、无外力运动、穿模等）。
- *Explicit Physics Rules*：NewtonRewards 用 frozen visual network 提取可测物理 metric，把 Newtonian motion law 和质量守恒转化为 RL 的数学 penalty；PhysRVG 用 SAM2 frame-by-frame derive motion mask 追踪 object trajectory；Wan2.2 用 optical-flow-based Newtonian penalty 显著改善 free fall、projectile motion、inclined plane sliding 的 temporal consistency。
- *Implicit Emergence*：Kling-Omni 用 understand-reason-generate 架构 + intelligent prompt enhancer 解释物理意图 + DiT-based Omni-Generator 配合 DPO；HunyuanVideo-1.5 不用 explicit RL physics reward，纯靠海量带精确 multimodal caption 的真实 video 数据自然学到 temporal coherence 和 long-term physical reasoning。

**Token Explosion**: DiT 架构下高分辨率长 video 的 token 数让 self-attention 计算平方爆炸，OOM + 推理慢。
- *Extreme Spatiotemporal VAE Compression*：LTX-2.3 把 patchify 操作移到 VAE input，single-step denoising 直接生成 native 4K；Wan2.2 用 Wan-VAE 大幅降低空间 token 数，配 Flow Matching 降低 1080P 长视频生成内存压力。
- *Dynamic Sparse Attention Pruning*：HunyuanVideo-1.5 的 SSTA 机制自动 prune 冗余 spatiotemporal block；Ming-Flash-Omni 2.0 用 modality-level routing 的 MoE。

**Audio-Visual Alignment**: 毫秒级时间 + 物理同步。
- *Strict Audio-Visual Anchoring via Unified Timelines*：MiniCPM-o 4.5 的 Omni-Flow full-duplex framework 把 audio-visual 输入和 text/speech 输出在单一 timeline 上 token-level 对齐；Qwen3-Omni 用 TM-RoPE 锚定绝对时间，放弃相对 segment alignment。
- *Synchronous Generation via Deep Architectural Coupling*：LTX-2.3 用 asymmetric dual-stream architecture + bidirectional cross-modal attention + cross-modal AdaLN + modality-CFG；Seedance2.0 在 diffusion 每一毫秒构造 Attention Bridge，visual branch 的 action intensity 传给 audio branch，audio emotion 和 rhythm 影响 visual lighting；OmniVoice 和 Qwen3-Omni 用 non-autoregressive discrete codec-based acoustic mapping 跳过两阶段 pipeline。

### 4.7 M2M — Fully Discretized Unified

这条路的核心矛盾：

**Loss from Discretizing**: 连续信号压到离散 vocab 必然 lossy，high-res image/audio 量化后丢掉 low-level feature。
- LongCat-Next 的 Semantic Completeness 原则，设计 dNaViT 作为 visual tokenizer——codebook embedding 不是 fixed 而是 randomly initialized，与 language token 在 shared autoregressive objective 下 co-evolve。
- Moshi 的 in-house neural audio codec Mimi 用 RVQ 分解连续 audio，通过 knowledge-distillation 让 early acoustic token 强制匹配 self-supervised speech model 的 semantic representation。
- AnyGPT 用 multilingual 策略，为每个 continuous modality 部署高度专门化 discrete tokenizer。

**Competition-Driven Latency**: 高信息密度 text token 和极稀疏 visual/audio token 塞进同一个 Softmax 计算 cross-entropy，不同 entropy level 的 feature 在权重上竞争，大规模数据下导致 output norm 指数爆炸 → gradient divergence。同时纯 AR 一步步预测数千 image token 导致不可忍受的推理延迟。
- Chameleon 修改 attention：引入 QK-Norm，在计算 dot product 前对 Query 和 Key vector 做 layer normalization 以抑制 representation competition。
- LLaDA2.0-Uni 用 Sprint Inference，通过 Adaptive Unmasking 和 confidence-based Batch Acceptance 打破单步 decoding 延迟瓶颈。
- Emu3.5 的 Discrete Diffusion Adaptation 把严格 token-by-token serial decoding 改成 bidirectional parallel prediction，单图推理 ~20× 加速。

### 4.8 M2M — Modality-Specificity Preserving

这派认为视觉空间连续性无法被离散 vocab 无损表达，主张 continuous feature space + decoupled encoder + hybrid loss。

**Comprehension-Generation Dilemma**: 理解需要高压缩的 high-level semantic abstraction，生成需要 fine-grained low-level pixel feature 重建。共享 representation 会导致 task interference。
- *Physical Decoupling*：Janus-Pro 用独立 visual encoder 分别做理解和生成；BAGEL 在 backbone 用 Mixture-of-Transformer-Experts (MoT)，hard routing 把 token 分到 Understanding 或 Generation 专家。
- *Encoder-Free Modeling*：TUNA-2 和 SenseNova-U1 干脆移除传统 CLIP encoder 和 VAE，把 raw image patch 直接喂入网络，消除预训练 inductive bias。SenseNova-U1 即使理解分支 frozen，也能用 raw pixel stream 重建精确 microscopic texture。

**Bridging AR and Diffusion**: 在单一网络内融合离散 AR 和连续 Diffusion。
- Transfusion：unified Transformer 对 text 用 discrete NTP loss，对 image patch 用 continuous denoising Diffusion loss；hybrid attention——text 用 causal mask 保逻辑，image patch 用 bidirectional attention 捕捉空间连续性。Loss 是 $\mathcal{L} = \mathcal{L}_{\text{LM}} + 5 \cdot \mathcal{L}_{\text{DDPM}}$，scaling coefficient $\lambda = 5$ 通过 preliminary search 确定。
- Show-o2：Spatial-Temporal Fusion via 3D Causal VAE，独立 semantic layer 提取 high-level info，cascading + MLP 融合 low-level feature；顶部独立 AR head 和 Flow-Matching head 管理 text 和 video flow。
- OneCAT-3B：纯 decoder 架构内实现 Modality-MoE，引入 multi-scale visual AR 机制绕过 serial bottleneck。
- Mamoda2.5：用 MetaQueries 桥接——AR backbone 生成高度浓缩的逻辑 plan，continuous features 直接连到 backend DiT-MoE 模块做高速 pixel rendering。

---

## 5. Training — Fusion-Coupled Signature

这是论文最有 insight 的部分。核心主张：**训练策略不是 architecture 的附属品，每种 fusion regime 都有自己的 training signature**。

PT 的 signature 用 5 个维度刻画：freezing topology, learning-rate topology, loss formulation, stability prescription, curriculum scheduling。

### 5.1 Late-Fusion PT

Encoder 是 "connectivity peripheral"，feature 通过 thin projector 流入 frozen LLM，gradient 不触及 encoder。5 个维度全部 degenerate：单一 global learning rate 就够，loss 是 text-only AR cross-entropy，无需额外 stabilizer，resolution/mixture schedule 被 frozen encoder subgraph 和 dataset 吸收。Visual token 只是 text sequence 的 prefix。

**Trade-off**: 用训练简单换 capped cross-modal capacity——encoder 无法 adapt 到 language objective。

### 5.2 Mid-Fusion PT

gradient 第一次流到 encoder，每个训练 signature 元素都是对这一变化的响应。

**Progressive Unfreezing**:
- Qwen2-VL 是 symmetric 变种：Stage 1 ViT 训练 + LLM frozen，Stage 2 一起 unfreeze。
- CogVLM、Janus-Pro、MiniCPM-V 把 encoder unfreeze 推迟到 SFT。
- Qwen2-VL 在 chat-tuning 阶段再次 freeze ViT——信号是：1.4T token 联合 PT 后 vision-text alignment 已巩固，instruction tuning 阶段不需要再更新 encoder。

**Differential Rates Mandatory**: encoder 接到 gradient 后单一 LR 不稳定——对 encoder 太高，对 LLM 太低。
- CogVLM 在 SFT-time unfreeze 时给 EVA2-CLIP-E encoder 应用 base rate 的 1/10，成为 canonical mid-fusion prescription。
- Janus-Pro 三阶段 LR decay: $10^{-3} \to 10^{-4} \to 4 \times 10^{-5}$，25× total reduction；最大 rate 对应 adapter-only 训练，最小对应 full-model SFT。
- Moshi 训 temporal transformer 用 $3 \times 10^{-5}$，depth transformer 用 $2 \times 10^{-4}$，~7× gap 反映不同收敛动态。

**Decoupled Loss of Understanding/Generation**:
- Janus-Pro 对 text token 算 cross-entropy (理解)，对 discrete VQ token 算 cross-entropy (生成)，两个独立 visual encoder 喂 shared LLM。
- BAGEL 通过 MoT 路由理解 (SigLIP feature 上的 text cross-entropy) 和生成 (continuous VAE latents 上的 Next-Group-Token Prediction MSE)，用 task-specific batch toggle 防止两条 pathway 干扰。共享参数但 modality-specific layer 不共享 gradient signal——partial 而非 full unification。

**Resolution & Context-Length Curricula**:
- MiniCPM-V: $224 \to 448 \to 1344^+$ over three stages。
- CogVLM 在 late pretraining 把 input 从 224 增到 490。
- Qwen2.5-Omni 在 final PT 阶段把 context window 从 8,192 扩到 32,768。
- Emu3 在 post-training 把生成分辨率从 512 提到 720px（理解到 1024px）。

关键 insight：**mid-fusion 把 "哪些参数 unfreeze" 和 "在什么分辨率 unfreeze" 绑在一起**，只用一个不用另一个会失稳。

### 5.3 Early-Fusion PT

消除模态间架构防火墙后：每个 component 从 step 0 就背 gradient，loss collapse 成单一 shared vocab 上的 objective，没有任何 modality-specific buffer，所以 stabilizer 是必需而非可选。

**Joint-from-Start**: 所有模块从第一步同时优化，无 freeze-to-unfreeze 转换。
- Chameleon vocab 扩 8,192 image token；Emu3.5 扩 32,768；AnyGPT 扩 8,192 image + 1,024 speech + 8,192 music。
- 回到单一 global learning rate：Chameleon $10^{-4} \to 10^{-5}$，Transfusion $3 \times 10^{-4} \to 1.5 \times 10^{-5}$ 用 cosine decay；依赖 architectural stabilizer 而非 rate engineering。
- 例外是 Llama-4 的 MetaP，方向相反——下降到 algorithmically determined per-layer rate。

**Unified NTP & Modal-Aware Attention**:
- 纯 NTP 模型 (Chameleon, Emu3.5, AnyGPT, LongCat-Next) 统一 cross-entropy loss over shared vocab。
- Hybrid 变体：Transfusion 用 $\mathcal{L} = \mathcal{L}_{\text{LM}} + 5 \cdot \mathcal{L}_{\text{DDPM}}$；Show-o 用 Mask Token Prediction (image) + NTP (text)；LLaDA2.0-Uni 对 text 和 image token 统一用 discrete diffusion masked-denoising objective 替代 AR loss。
- Moshi 的 RVQ：semantic codebook loss weight $\alpha_k = 100$，acoustic codebook weight $\alpha_k = 1$，确保 linguistic content 优先于 acoustic detail。
- Attention pattern：纯 NTP 模型跨 modality 统一 causal attention，用 structural token 划分 modality 边界；hybrid 模型在需要时放松约束——Transfusion 和 Show-o 允许 image region 内 bidirectional attention，因为 diffusion 本质上需要全图上下文。

**Z-loss 和 QK-Norm for Stability**:

Z-loss regularization 形式化：
$$\mathcal{L}_{z\text{-loss}} = 10^{-5} \cdot \log^2 Z$$

其中 $Z$ 是 softmax partition function（所有 logit 的 exp 之和）。这个正则让 logit 在异构 token distribution 上保持有界。

Chameleon 的 ablation 铁证：**没有 QK-Norm，模型在训练 ~20% 后 diverge**。这是 discrete-token early-fusion 的工程前置条件，不是 generic transformer trick。

**Modality-Mixture Scheduling**: 关键变化是——每个 modality 的独立 loss head 被消除，所以 batch 里的 modality mixture 直接决定 gradient 方向。这迫使每个 early-fusion 系统把 mixture scheduling 当成核心 training hyperparameter。

- Transfusion: 1:1 text-to-image token ratio，80% 时候 caption 在对应 image 前面；用 BOI/EOI token 作为 attention-pattern trigger 在 causal 和 bidirectional region 间切换。
- Chameleon: image 和 text token 用 special delimiter interleave，每张图固定 1,024 token (512×512 center crop)，确保每张图的 gradient 恒定。
- Moshi: text 和 audio 在 12.5 Hz frame level interleave——每个 timestep 1 个 text position + 8 个 audio codebook position。关键：**一半的 pretraining batch 分配给 text-only data**，作为语言能力的显式 anti-forgetting buffer。
- Open-Sora 2.0: T2V → I2V 通过 prepend reference-frame latent；resolution curriculum 256px → 768px 并行运行。
- HunyuanVideo: 256px → 960px；Wan: 256px → 720px；全程 mixed image-video batch。

Chameleon 文档明确警告：**imbalanced modality mixture 导致 early-fusion model 学到 degenerate unconditional prior，扭曲生成**。这是 late-fusion (inert generation pathway) 和 mid-fusion (decoupled losses) 结构上能避免的退化行为。所以 modality-mixture scheduling 是 early-fusion 的 differential learning rate 等价物——是成功训练的基本前置条件，不只是优化 refinement。

### 5.4 SFT 维度的两个新 axis

PT 的 5 维度 signature 延续到 SFT，但多了两个 regime-specific axis：
1. **Freezing Rewiring**: mid-fusion 独享的特权——SFT 可以 unfreeze PT 时 frozen 的 encoder，或 re-freeze PT 时训练过的 component。Late-fusion 没东西可 unfreeze，early-fusion 的 joint-throughout 承诺禁止 re-freeze。
2. **Distribution Rebalancing**: SFT corpus 比 PT 小得多且偏 text-heavy，需要在更小 corpus 上恢复合适 modality mixture。

**Mid-Fusion SFT 的两种 rewiring 策略**：
- *Unfreeze-at-SFT*：CogVLM、Janus-Pro、MiniCPM-V 在 PT 保持 encoder frozen，SFT 时 unlock；Janus-Pro 选择性保留 generation tokenizer frozen，体现 mid-fusion 允许的 asymmetric component-by-component thaw。
- *Train-then-re-freeze*：Qwen2-VL 在 SFT 阶段把训练过两阶段 PT 的 ViT re-freeze，只 tune LLM on ChatML。这是 mid-fusion 独有特征。

**Mid-Fusion SFT 的 Pathway-Specific Tightening**:
- Janus-Pro 把 generation/understanding data ratio 从 PT Stage II 的 50/50 移到 SFT Stage III 的 40/60，向理解倾斜而不丢生成能力。
- HunyuanVideo 最后阶段用 ~1M 个 aesthetic + motion-scored 人工标注样本；Wan 用 resolution-dependent quality filter；Open-Sora 2.0 同时把任务从 T2V 移到 I2V。

**Early-Fusion SFT**：所有 freezing rewiring 都被 foreclosed。SFT 缩减到只在 universal layer 上操作（lower LR、prompt-token loss masking、dropout 比如 Chameleon 34B 加 0.05）。Regime-specific 唯一责任是 re-balance modality mixture。

边界 edge case：AnyGPT 在 SFT 反转典型模式——freeze LLM backbone，只更新新加的 multimodal embedding 和 prediction layer，5000 步；BAGEL 的 all-trainable all-stage joint optimization 是它的对立极端。

### 5.5 RL — Fusion Regime 决定 Policy Scope

论文这里有一个核心论断：**fusion regime 而非 algorithm 决定 RL scope**。这是 RL 设计中 single most consequential choice。

**Late-Fusion RL**: 结构上最小——架构 cleanly 分离 quality-localizable head，RL 只 target 这个 head。Qwen2.5-Omni 和 Qwen3-Omni 对 Talker 应用 DPO (over WER/pause-error-ranked triplets)，Thinker 和所有 encoder 不动。Toolkit collapse 到最简：offline DPO + rule-based scoring。Projector 太薄无法 overwhelm visual evidence，policy 不会偏离 visual conditioning 太远。

**Mid-Fusion RL**: 继承 mid-fusion SFT 的 pathway-decoupled trainable set，gradient 只路由到正在优化的 pathway。HunyuanVideo、Wan、T2I-R1、Flow-GRPO 保持 VAE 和 text encoder frozen，RL gradient 只进 diffusion transformer。Reward 通常 rule-based (CLIP, ImageReward, aesthetic/motion scorer)，pathway-locality 匹配 update 的 pathway-locality。

第一个 regime-specific failure mode 出现在理解 pathway：naive DPO on multimodal preference pair 过度依赖 language prior，忽略 image condition，最终学到 text-only preference。**mDPO** 显式把 image 作为 preference loss 的 condition，让 chosen-vs-rejected gap 直接依赖 visual input。

**Early-Fusion RL**: pathway-locality 在 early-fusion 下不可用。Unified softmax 下没有 isolated head 的 update 能让其他部分不变，所以 RL scope 必然扩展到 full backbone。

Emu3.5、UniRL 用 GRPO 更新整个 policy。Unified softmax 强制扩展同时交付主要 reward——单一 scalar 到达任何 output token 时通过 shared parameter 跨 modality 做 credit-assignment，这是 mid-fusion decoupled pathway 下做不到的。Reward model 通常从 SFT checkpoint 初始化，让 policy 和 reward natively share representation。

但扩展 scope 暴露两个 mid-fusion decoupling 结构性抑制的 failure mode：

1. **Visual-grounding hacking**: full-policy update 可以在不 grounding claims 到 image 的情况下推高 textual reward proxy (length, formatting, certainty)。Fact-RLHF 和 shortcut-aware MM-RM 从 reward 侧应对；policy 侧加 explicit visual-faithfulness term。

2. **Perceptual vs Logical Errors in Process Supervision**: multimodal CoT error 分两类——logical (computation, derivation) 和 perceptual (misread chart, mislocalize region)。Outcome-only RL 把两类混淆；multimodal PRM (URSA, GM-PRM) 分开它们。这在 early-fusion 下尤其关键——两类 error 都 route 到同一组参数。

两个 failure mode 共享同一机制：**unified softmax 下 language prior 和 visual evidence 平等竞争，naive RL 让 prior 赢**。第二个结构性 cost：每个 capability (math, code, agentic, instruction-following, safety) 各自做 specialized RL run，checkpoint 互相 trade-off，造成 see-saw effect。两个 cost 共同驱动下一个 post-RL primitive。

### 5.6 On-Policy Distillation (OPD)

OPD 是 GRPO 的 single-line 修改。把 group relative advantage 替换为对 teacher 的 stop-gradient reverse-KL log-ratio：

$$\hat{A}_{i,t} = \text{sg}\left[\log \frac{\pi_{\text{teacher}}(y_{i,t} \mid x, y_{i,<t})}{\pi_{\text{student}}(y_{i,t} \mid x, y_{i,<t})}\right]$$

变量含义：
- $\hat{A}_{i,t}$：student 在第 $i$ 个 sample 第 $t$ 个 token 的 advantage 估计。
- $\text{sg}[\cdot]$：stop-gradient operator，里面计算结果不传梯度给 teacher。
- $\pi_{\text{teacher}}$：teacher policy 给出 token $y_{i,t}$ 在条件 $(x, y_{i,<t})$ 下的概率。
- $\pi_{\text{student}}$：student policy 同条件下的概率。
- $x$：prompt，$y_{i,<t}$：第 $i$ 个 sample 前 $t-1$ 个已生成 token。

Intuition：每个从 student 采样的 token 都获得 dense per-position teacher supervision，同时保持 on-policy。

**MiMo-V2.5 是首个公开报道的 NMM 上 MOPD 部署**：
- 流程：text PT → projector warmup → multimodal PT → SFT + agentic post-training (context 32K → 1M) → RL + MOPD。
- MOPD 作为 terminal consolidation step，explicit 任务是强化 perception、reasoning、agentic capability 在 shared backbone 上。
- 三个结构组件：
  1. **Specialist Teacher Pool**：通过独立 domain RL 获得。
  2. **Outcome-Reward Augmentation**: $\hat{A}_{i,t} = \hat{A}_{i,t}^{\text{OPD}} + \alpha \tilde{A}_{i,t}^{\text{ORM}}$，decouple student 不被任一 teacher ceiling 限制。
  3. **Permissive Teacher Pool**: 接受 domain SFT model、RL specialist、student 自己的 frozen snapshot——snapshot 充当 anti-drift anchor，在其他 teacher 把 student 推到不熟悉区域时稳定它。

---

## 6. Inference & Deployment

### 6.1 Long-Context Multimodal Inference 的 Sequence Explosion

Native multimodal PT 显著放大经典 long-context 问题。高分辨率 image、多图文档、长 video 不再是 compact side feature，而是被转成数百、数千、甚至数百万 visual 和 temporal token 与 language token 共存。

两条互补路径：

**Visual Resampling and Token Compression**:
- Fixed-budget resampler 和 pooling module 把 dense patch grid 映射到少量 latent token，稳定 prefill latency 不受原始分辨率影响。MiniCPM-V 4.5、Gemma3 用这思路。
- Adaptive 方法：VisionZip、SparseVLM、FitPrune、LLaVA-PruMerge 根据 information density、attention behavior、similarity structure select/prune/recycle/merge visual token；VisionSelector、LaCo 把压缩移到学习的 visual pathway 内。

**Dynamic Resolution & Spatially Sparse Perception**:
- Qwen2-VL、Qwen2.5-VL 用 dynamic visual tokenization + multimodal RoPE 让 image/video token 在任意分辨率下保持 spatial 和 temporal grounding。
- LLaVA-UHD、LLaVA-OneVision、Oryx、InternVL 2.5 用 AnyRes-style slicing、spatial schema、on-demand compression 保留高分辨率细节同时防止 token 数随像素数机械增长。
- Q-Zoom 让分辨率决策条件化于 user instruction：先在 coarse view 上 reason，只在可能影响答案的区域花高分辨率 token。

### 6.2 Heterogeneity & Scale in MLLMs

**Heterogeneity**: 人类语言是 abstract、discrete、symbolic 的，而视觉/听觉/sensory signal 是 high-dimensional、continuous、physical-observable 的。差异涵盖 information density、temporal granularity、noise characteristic。

**Scale**: trillion-parameter + 千张高分辨率 image/video/audio 长 context 让 attention quadratic complexity 变成 prohibitive bottleneck。Activation memory、inter-layer communication、gradient sync overhead 碰撞 HBM 容量和 interconnect 带宽的物理极限。

**Pure Discrete Tokenization**: 把高维 continuous signal vector-quantize 成有限 discrete integer ID。
- Chameleon: 8,192-entry 独立 image codebook 处理 unified 1D sequence，消除 hardware-level branching overhead。
- Emu3.5: 大幅扩 discrete image vocab + feature distillation，完全抛弃 diffusion，证明单一 transformer 能纯靠 next-token prediction 实现混合 modality 训练。
- Seedance 2.0: 把多达 12 channel mixed input 标准化成 unified spatiotemporal 和 waveform token 并行处理。
- AnyGPT: 验证 discrete data-level preprocessing 对任意 modality 对话的普适性。

**MoE 和 Hybrid Paradigm 优化 Routing**:
- Kimi2.5 用严格 routing 策略 prune activation，支持多 agent 并发推理极低成本。
- Janus-Pro 引入 fine-grained isolated expert 实现 modality-aware implicit computational bifurcation。
- Transfusion 同时优化 discrete AR 和 continuous denoising；但强行 nest causal 和 bidirectional mask 会破坏 FlashAttention 的 memory alignment 假设。
- **FlexAttention** 用 JIT compilation 动态生成 fused computation graph；**FlashMask** 允许在 causal 和 bidirectional block 间快速切换。Tensor operation 的解放最终实现 hybrid multimodal architecture 的企业级部署。

### 6.3 Real-Time Streaming & Full-Duplex

为应对动态到达 multimodal stream 的 latency bottleneck 和 first-token delay，NMM 正从静态 offline generation 转向 unified inference paradigm——streaming decoding、duplex concurrency、resource-adaptive serving。

**Incremental Multimodal Token Decoding**: 不再等整个 visual/acoustic sequence 编码完才响应，progressive emit visual/audio token (patch-by-patch, frame-by-frame, streaming) 降低 TTFT。配 adaptive visual granularity 和 dynamic input reduction——只保留最 task-relevant visual token。

**Full-Duplex State Management**: 支持 incoming sensory stream 和 outgoing generation stream 并发 inference。duplex dialogue control、streaming state prediction、dynamic KV-cache management 缓解 cache contention 和 sequential blocking。

**Inference-Time Adaptive Bitrate Control**: 在 runtime bandwidth约束下动态降低 discrete visual code granularity 换 latency。Runtime visual token budgeting——adaptive resolution selection 和 visual token compression 是 bitrate-aware streaming 的近似。

**Modality-Aware Mixed Quantization**: 给 visual encoder、projector、language backbone 分配不同 precision，结合 runtime-aware visual simplification (dynamic resolution degradation, adaptive preprocessing, token pruning, energy-aware visual reduction) 让 edge system 根据 latency/energy/hardware pressure 降低 visual input token 数。

---

## 7. Evaluation

### 7.1 Image

**Understanding**:
- General Perception: VQAv2, GQA, SEED-Bench, MMBench, MMStar。
- Knowledge Reasoning: MMMU (30 学科 college-level), MathVista。
- Hallucination: POPE (polling-based binary probing), RLHF-V (segment-level)。
- Document & OCR: DocVQA, ChartQA, InfoVQA, OCRBench。

**Generation**: 从 distribution-level metric 进化到 compositional, semantic-level。
- FID: distributional similarity but insensitive to compositional accuracy。
- GenEval: 分解 T2I 成 attribute binding, spatial relationship, counting。
- DPG-Bench: dense prompt following with long, compositionally complex description。
- T2I-CompBench: attribute binding, object relationship, complex composition。
- CLIPScore: reference-free text-image alignment metric。

论文这里有一个关键诊断 insight：**Rao and Rachuri 证明 DPO on VQ-based unified model 即使 understanding metric 改善也 fails to improve CLIPScore**，揭示 discrete tokenization 给 offline preference optimization 制造 structural bottleneck。

### 7.2 Audio

- ASR: WER on LibriSpeech, CommonVoice, FLEURS。
- TTS: MOS (subjective naturalness, prosody, speaker similarity)；first-token latency, word-level sync accuracy, voice cloning fidelity。
- Full-Duplex: turn-taking accuracy, barge-in handling, response latency, false interruption rate。Moshi 200ms 目标；SoulX-Duplug 240ms bilingual streaming turn detection；Full-Duplex-Bench 多维评估。

### 7.3 Video

**Understanding**:
- Offline: VideoMME, EgoSchema, MVBench (20 fine-grained temporal task), PerceptionTest, LongVideoBench, MLVU。
- Streaming: OVO-Bench (real-time perception + backward tracing), StreamingBench (latency constraint 下 video comprehension), ThinkStream (Watch-Think-Speak protocol 不仅评 accuracy 还评 response timing), AURA (proactive QA + multi-response QA)。
- Efficiency-aware: ResAdapt 消除 >90% visual token 同时扩展 temporal horizon 16×，在复杂长视频 reasoning 上 >15% relative gain。

**Generation**: FVD (UCF-101, Kinetics-600), VBench (temporal consistency, motion smoothness, subject preservation, aesthetic), SeedVideoBench 2.0 (6 维: motion quality, video prompt adherence, aesthetics, audio quality, audio-visual sync, audio prompt following), Arena.AI T2V/I2V (community-scale human preference Elo), Next Block Prediction (semi-AR video generation 的 spatiotemporal coherence 评估), LTX-2 (joint audio-visual generation quality)。

---

## 8. Future Outlook

### 8.1 Architectural Convergence

论文期望 M2T/M2G/M2M 分裂逐步 collapse，三条收敛 axis：
1. **统一 understanding 和 generation 在单一 backbone**——单一 probabilistic objective、unified tokenization scheme 或 continuous latent grammar 能否同时支撑两 front 而不质量退化？discrete-token unification (Chameleon, AnyGPT, Janus-Pro) vs continuous-latent path (TUNA-2, Mamoda2.5) 仍是 unresolved design choice。
2. **Scaling sparsity 和 modality-aware expert**——formalize expert nativity 作为 architectural nativity 的对应。
3. **超越四个 canonical modality**——proprioception, depth, tactile, action sequence, code, graph, 3D scene。

### 8.2 Data

- Cross-modal data scarcity 和 synthesis: 长视频 + 同步 audio + transcript + action + reasoning trace 是最硬瓶颈。Self-distilled multimodal pipeline 的 filter/de-bias/anti-collapse 方法论缺失。
- Interaction-grounded data at scale: full-duplex audio、streaming video、proactive agent trace 需要捕捉 not only what to respond but also when to respond。
- Preference data for generative modality: cross-modal reward modeling 联合训练 policy 将成为核心 data engineering effort。

### 8.3 Training

- Modality-balanced optimization: 不同 information density 的 token 混合 (32K-token long-doc SFT sample vs sequence-packed image grid) 制造 loss-scale 和 gradient-norm asymmetry。principled token-budget allocation, per-modality loss weighting, curriculum scheduling 仍 underexplored。
- RL for cross-modal generation: 把 verifiable reward 扩展到 image/audio/video generation 和 interleaved interaction trace。policy-gradient 与 diffusion/flow-based generative objective 的统一（可能通过 stepwise multimodal advantage estimation）将是核心技术 thrust。
- OPD for omni capability: M2M 行为 distill 到 compact deployable model 在 streaming 和 full-duplex 约束下 largely uncharted。

### 8.4 Inference & Deployment

- Native long-context 和 adaptive perception: 256K context window 之外，selectively spending compute on informative region。ResAdapt 消除 >90% visual token 同时扩展 16× temporal horizon 指向 emerging accuracy-efficiency Pareto frontier。
- System-algorithm co-design for sparse multimodal MoE: disaggregated prefill/decoding、expert offloading、modality-aware KV-cache management。
- Born-streaming, born-duplex deployment: 真 native interactive agent 需要 streaming by construction。Moshi, ELLSA, FireRedChat, ThinkStream 暗示这个未来，但稳定可部署低延迟跨 modality 一致质量系统仍是工业 open problem。

### 8.5 Evaluation

四个 open direction：
1. Symmetric M2M benchmark 评 aligned understanding-generation pair (describe-then-render, listen-then-speak, watch-then-act)。
2. Temporally-aware metric: 评 answer quality + response timing。
3. Efficiency-aware protocol: accuracy + token budget + latency + energy。
4. Robustness 和 safety under multimodal attack surface: adversarial cross-modal prompt, image/audio jailbreak, generated content hallucination。

### 8.6 Native World Model

终极愿景：NMM 演化成 genuine world model——unified backbone 感知 raw sensory stream、maintain persistent state across long horizon、act in continuous time。架构层面从 late-fusion stitching 到 early-fusion convergence 的 roadmap 已清晰，但通往可部署 born-native world model 的路径尚不明确。

---

## 9. 我的 Critical Insights

这篇论文最大的贡献不是技术细节本身，而是把过去两年散落的工程实践**形式化为 regime-specific signature**——特别是 PT 5 个维度（freezing topology, LR topology, loss formulation, stability prescription, curriculum scheduling）+ SFT/RL 的两个额外 axis (freezing rewiring, policy scope) 的框架。这给社区一个诊断工具：拿到一个 NMM，问 "训练时 encoder 何时 unfreeze？rate 怎么设？loss 是 unified 还是 decoupled？用了 z-loss + QK-Norm 吗？mixture schedule 怎么走？"——这些答案能直接告诉你它在哪个 fusion regime，以及它的瓶颈在哪。

几个值得深挖的点：

1. **z-loss + QK-Norm 是 early-fusion 的 engineering precondition**。Chameleon 的 ablation 铁证——没有 QK-Norm 训练 20% 就 diverge。这背后的物理直觉是：unified softmax 把信息密度差几个数量级的 token (text token vs image patch token vs audio codebook token) 塞到同一个 Softmax，partition function $Z$ 会被高熵 modality 主导，logit 爆炸。$10^{-5} \cdot \log^2 Z$ 把 $Z$ 拉回有界——这不是 generic trick，是异构 token distribution 下的 numerical 必要条件。QK-Norm 在 attention 端做类似事情：modality 间 representation norm 不匹配会让 attention score 被 high-norm modality 主导，QK-Norm 强制 Q 和 K 在 dot product 前 normalize，抑制这种 competition。

2. **Modality-mixture scheduling 是 early-fusion 的 differential LR 等价物**。这可能是论文最反直觉的 insight。在 mid-fusion 下，两条 pathway 有独立 loss head，所以 batch 内 modality 比例只影响 aggregate gradient magnitude 而非方向；但 early-fusion 把所有 modality 塞到 unified cross-entropy，mixture 直接决定 gradient 方向。Chameleon 的"degenerate unconditional prior"警告应该被每个想做 early-fusion 的人读三遍——这不只是优化问题，是模型会学到错分布的根本性问题。

3. **Early-Fusion RL 的两个 failure mode 都源于 unified softmax 下 language prior 和 visual evidence 的平等竞争**。这暗示一个更深层问题：在 unified softmax 下，naive RL 让 prior 赢是因为 prior 是更"光滑"的 reward surface，而 visual grounding 需要稀疏的 per-image evidence。MOPD 的 anti-drift anchor (student 自己的 frozen snapshot) 是结构性应对——它在 unfamiliar territory 提供 familiar prior，让 student 不会在 teacher 集体推动下 drift 到 novel region。这是多 teacher distillation 中我没见过 explicit formalization 的设计。

4. **Pathway-Locality 的 Trade-off**: mid-fusion 的 decoupled loss 让 RL tractable 但 cap 了 cross-modal credit assignment；early-fusion 强制 full-policy update 让 credit assignment 自由但暴露 hacking 和 see-saw。这是 NMM 的 fundamental tension——你无法在不放弃 structural decoupling 的同时获得 unified credit assignment。OPD 的出现是对这一 tension 的工程响应，而非彻底解决。

5. **Evaluation Gap**: 论文 §7 揭示两个 systemic gap——多数 benchmark 评单一 modality 而非联合 understanding-generation；accuracy-only metric 忽略 native deployment 最关心的 when to respond, how much compute, how gracefully handle streaming。ResAdapt 90% visual token 消除 + 16× temporal horizon 的结果暗示我们正在进入 accuracy-efficiency Pareto frontier 时代，但现有 benchmark 普遍不报告 efficiency dimension。

几个可能被质疑的地方：

- 论文把 mid-fusion 和 early-fusion 的边界划得很清晰，但实际部署中很多 model 是 hybrid——比如 BAGEL 用 MoT 把理解/生成路由到不同 expert，这在架构上是 mid-fusion (有显式 modality-aware boundary) 还是 early-fusion (单一 Transformer 处理所有 modality)？论文把它归到 M2M Modality-Specificity Preserving 下，但严格说它跨界。

- 论文引用的 2026 年文献 (Kimi K2.5, GLM-5V-Turbo, MiMo-V2.5, Nemotron3-Nano-Omni, Mamoda2.5, TUNA-2, LongCat-Next, Lance) 我没有独立 verify 它们的实际架构细节——这部分需要 cross-check 原始 technical report。

- "Native World Model" §8.6 是 vision 部分，但论文对 continuous time 和 persistent state 的处理基本没展开——这其实是更深层的问题，因为 next-token prediction 本质是 discrete time，如何 reconcile continuous-time sensory stream 和 discrete token 是开放问题，论文承认但没给方向。

整体上，这篇论文对 2025-2026 NMM 爆发期做了一个有形式化野心的总结，特别是 fusion-coupled training signature 的框架有 teaching value。但它本质仍是 taxonomy + roadmap，没有提出新的 architecture 或 training method——这是它的 limitation，也是它的 contribution 边界。

**References**:

- 论文 arXiv 链接暂时无（从内容看是 2026 年中后期工作），code 在 https://nmm-roadmap.github.io
- Chameleon: https://arxiv.org/abs/2405.09818
- Transfusion: https://arxiv.org/abs/2408.11039
- Emu3.5: https://arxiv.org/abs/2510.26583
- AnyGPT: https://arxiv.org/abs/2402.12226
- Moshi: https://arxiv.org/abs/2410.00037
- Janus-Pro: https://arxiv.org/abs/2501.17811
- BAGEL: https://arxiv.org/abs/2505.14683
- Show-o2: https://arxiv.org/abs/2506.15564
- TUNA-2: https://arxiv.org/abs/2604.24763
- Mamoda2.5: https://arxiv.org/abs/2605.02641
- LLaDA2.0-Uni: https://arxiv.org/abs/2604.20796
- LongCat-Next: https://arxiv.org/abs/2603.27538
- SenseNova-U1: https://arxiv.org/abs/2605.12500
- Lance: https://arxiv.org/abs/2605.18678
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- InternVL-3.5: https://arxiv.org/abs/2508.18265
- Kimi K2.5: https://arxiv.org/abs/2602.02276
- GLM-5V-Turbo: 来自 ZhipuAI technical report
- MiMo-V2.5: 来自 Xiaomi MiMo Team
- Nemotron3-Nano-Omni: https://arxiv.org/abs/2604.24954
- MiniCPM-V-4.5: https://arxiv.org/abs/2509.18154
- MiniCPM-o-4.5: https://arxiv.org/abs/2604.27393
- HunyuanVideo-1.5: https://arxiv.org/abs/2511.18870
- Kling-Omni: https://arxiv.org/abs/2512.16776
- Wan2.2: 来自 Wan Team technical report
- Seedream3.0: https://arxiv.org/abs/2504.11346
- LTX-2: https://arxiv.org/abs/2601.03233
- Ming-Flash-Omni-2.0: https://arxiv.org/abs/2510.24821
- OmniVoice: https://arxiv.org/abs/2604.00688
- HiDream-O1-Image: https://arxiv.org/abs/2605.11061
- Qwen3-Omni: https://arxiv.org/abs/2509.17765
- FlexAttention: https://arxiv.org/abs/2412.05496
- FlashMask: https://arxiv.org/abs/2410.01359
- ResAdapt: https://arxiv.org/abs/2603.28610
- ThinkStream: https://arxiv.org/abs/2603.12938
- AURA: https://arxiv.org/abs/2604.04184
- NewtonRewards: https://arxiv.org/abs/2512.00425
- PhysRVG: https://arxiv.org/abs/2601.11087
- mDPO: https://arxiv.org/abs/2406.11839
- URSA: https://arxiv.org/abs/2501.04686
- GM-PRM: https://arxiv.org/abs/2508.04088
- Fact-RLHF: https://arxiv.org/abs/2309.14525
- T2I-R1: https://arxiv.org/abs/2505.00703
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- UniRL: https://arxiv.org/abs/2505.23380
