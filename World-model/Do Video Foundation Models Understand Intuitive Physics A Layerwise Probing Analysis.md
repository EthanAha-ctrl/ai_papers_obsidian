---
source_pdf: Do Video Foundation Models Understand Intuitive Physics A Layerwise Probing
  Analysis.pdf
paper_sha256: 6541d93cf7bf1c84e5ff402b31b52e1aff8af27d97ae7101cdf6306a804c8a20
processed_at: '2026-08-18T06:30:53-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

简单来讲，这篇 paper 就是在给现在的视频大模型做“体检”，看看它们到底是真的懂物理常识，还是只是死记硬背了画面的纹理。

我们人类看个视频，如果一个球飞过去突然穿墙了，你会觉得“这不对劲”，因为我们脑子里有直觉物理（intuitive physics）。这篇论文的作者就想知道：现在的 Video Foundation Models（比如 V-JEPA, VideoMAE，还有做视频生成的 Diffusion 模型），它们看了几千万个视频之后，脑子里有没有也长出这种“物理直觉”？

为了搞清楚这个问题，他们没有去跑传统的跑分测试，而是搞了个“解剖实验”：

1. **把模型冻住，不许它临时抱佛脚**：他们把预训练好的模型参数全部冻结（frozen），不让它为了应付考试再去微调学习。这样就排除了“考前背题”的嫌疑，只能看它原本脑子里有什么。
2. **拿探针去“扎”不同层**：模型是一层一层处理信息的。他们把模型切开了，在 25%、50%、75%、100% 的深度都插了“探针”（probe）。这就好比去测大脑皮层不同区域负责什么功能。
3. **用最难的卷子考它**：普通的视频测试，模型可能靠作弊（比如只看背景颜色）就猜对答案了。所以他们用了两套防作弊的卷子：IntPhys2（专门考物体不能凭空消失、穿墙等）和 MVP（Minimal Video Pairs，放两段长得几乎一模一样、但物理结果相反的视频，看你能不能分辨）。

**体检结果是什么呢？**

* **V-JEPA 家族是物理课代表**：V-JEPA 的训练方法是“预测接下来会发生什么”（在 latent space 里预测未来的画面状态），而不是像 VideoMAE 那样去“把被遮住的马赛克拼回去”（pixel reconstruction）。结果发现，V-JEPA 脑子里的物理直觉最准。这就说明，逼着模型去“猜未来”，比逼着它去“补像素”，更能让它领悟世界的物理规律。
* **知识藏在中层**：物理常识并不是在模型最深层（最后一层）最强。通常在中间偏后的深度（75% 左右），物理信息最容易被提取出来。到了最后一层，模型往往专注于它被训练的具体任务，反而把通用的物理常识给“扭曲”了。这就跟人一样，太钻牛角尖反而失去了常识。
* **它确实是在看视频动起来的规律，不是看截图**：作者做了个狠测试，把视频帧的顺序打乱，或者干脆把一帧画面复制成视频。结果模型的成绩暴跌。这证明它确实是靠观察时间线上的动态变化来理解物理的，不是靠单张截图蒙混过关。
* **但也有点小猫腻**：在 IntPhys2 这个测试里，如果把视频帧打乱，有些模型成绩居然没掉！这说明 IntPhys2 这个数据集本身可能有点漏洞，模型可以通过数一数画面里有几个球、出现了几次这种“不走心”的方式来作弊，根本没去真正理解因果顺序。相反，MVP 这个测试打乱顺序成绩就彻底崩了，说明它才是真正在测动态物理。

**一句话总结**：通过“解剖”发现，现在的视频大模型（尤其是像 V-JEPA 这种预测未来的模型）确实已经自发地学会了不少物理常识。这些常识藏在模型的中间层，而且它们是真靠看视频的动态变化学会的，不是死记硬背。这也证明了 LeCun 力推的 JEPA 架构（在抽象空间里预测未来）确实是通往世界模型（world model）的一条对的路。

---

Hey Andrej, 很高兴能和你 discuss 这篇 paper。结合你平时对 representation learning, world models 以及 model 内部 mechanism 的深刻关注，我把这篇 paper 的核心 intuition、架构细节、公式拆解以及实验数据提炼出来，尽量多地补充相关的联想和技术细节。

### 1. Paper Overview & Core Motivation

这篇 paper 的核心目标是探索 pretrained video foundation models 是否在 frozen representations 中隐式 encode 了 intuitive-physics 信息。Author 使用了 frozen-feature probing methodology，去 evaluate 三大主流 pretraining paradigm，分析 physics information 在 layers 间的 distribution，以及不同 readout mechanism 对结果的影响。

这里的关键在于 "frozen-feature probing"：backbone weights 完全 frozen，不进行 end-task finetuning。这就剥离了 downstream training 带来的 masking effect，纯粹检验 pretraining 本身是否在 latent space 里 build 出了符合真实世界物理规律的结构。这和 [El Banani et al. (2024)](https://arxiv.org/abs/2403.17856) 探讨 3D awareness 的逻辑一脉相承，以及你之前常提的 "looking inside the black box" 的精神完全一致。

### 2. Pretraining Paradigms 分析

Paper 比较了三个极具代表性的 family，它们各自代表了不同的 inductive bias：

*   **V-JEPA Family** (Predictive Joint-Embedding): 基于 Yann LeCun 的 [JEPA architecture](https://openreview.net/forum?id=BDcoE1eo4SZ)。包括 V-JEPA, V-JEPA 2, 和 V-JEPA 2.1 (ViT-Gigantic/16)。
    *   **Intuition**: V-JEPA 不 reconstruct raw pixels，它在 latent space predict target representations from context。由于摆脱了 pixel-level reconstruction 的干扰（比如 texture, lighting 的 precise prediction），model 被迫去 learn 更 abstract, semantically meaningful，且 temporally grounded 的 dynamics。这就好比你常说的 LLM predicting next token 学到了 syntax 一样，V-JEPA predicting future latent state 强迫它学到了 object permanence 和 physical constraints。
*   **VideoMAE Family** (Masked Video Reconstruction): 包括 VideoMAE (ViT-H/16) 和 VideoMAE-v2 (ViT-G/16)。
    *   **Intuition**: 采用 extreme masking ratio (比如 90% tube masking) 去重建 masked spatio-temporal content。虽然 high masking 逼迫 model 利用 temporal context，但最终 objective 依然落在 pixel space，导致 representation 可能 encode 了大量 superficial visual statistics。
*   **LTX-Video** (Diffusion-based Generator): 13B parameter DiT (Diffusion Transformer)。
    *   **Intuition**: 优化目标是 progressive denoising 去 generate video。Paper 将其作为 counterpart，探讨为 generation 优化的 latent space 是否也能隐式 encode physics。

### 3. Datasets & Metrics Technical Deep Dive

为了严谨性，paper 选用了两个专门针对 "shortcut-aware" 设计的 benchmarks。

#### 3.1 IntPhys2
基于 Developmental psychology 的 "Violation of Expectation" (VOE) 原理。覆盖四个物理原则：permanence, immutability, spatio-temporal continuity, solidity。每个 scene 包含 4 个 clips（2 possible, 2 impossible）。

核心 metric 是 **VOE accuracy**，公式如下：
$$ \text{VOE} = \mathbb{1}\left[ \min(\text{score}_{\text{possible}}) > \max(\text{score}_{\text{impossible}}) \right] $$

*   $\text{score}_{\text{possible}}$ : 指在同一个 scene 下，所有 physically possible clips 被 probe 给出的 scalar plausibility estimate。
*   $\text{score}_{\text{impossible}}$ : 所有 impossible clips 的 score。
*   $\min$ 和 $\max$ : 集合的最小/最大值函数。
*   $\mathbb{1}[\cdot]$ : 指示函数，条件为真输出 1，否则 0。

**Intuition 解析**: 这个公式极其 strict。模型不仅要给每个 clip 打出合理的高低分，还必须在同一个 scene 的 4 个 clip 中保持绝对的 ranking consistency。只要最差的 possible clip 的分数低于最好的 impossible clip，整个 scene 就算作 fail。这避免了 model 靠着整体 accuracy 蒙混过关。

#### 3.2 MVP (Minimal Video Pairs)
来自 [Krojer et al. (2025)](https://arxiv.org/abs/2502.00748)。其精髓在于 minimal pairs：两个 video 视觉上极度相似，搭配同一个 question，但是 require opposite answers。这彻底摧毁了 model 依靠 dataset bias 或者 superficial visual cue 去做 classification 的捷径。

因为大多数 video backbone（如 V-JEPA）不接受 text query，author 把 MVP 从 text-conditioned VideoQA 转化成了 binary plausibility prediction。Metric 采用 **pair consistency**：只有当 pair 中的两个 video 都被 classify 正确时，才算 1 分。

### 4. Probing Methodology & Architecture Details

为了彻底榨干 frozen features 里的信息，author 使用了三种 expressivity 递增的 probes，对应 input pooling 策略和 architecture 的演进：

1.  **Linear Probe**: 直接将 average pooled clip embedding 映射到 output space。
    *   $y = W x + b$ (其中 $x$ 为 pooled feature)。
    *   Test：如果 physics 信息是 linearly decodable 的，那么说明信息在 representation space 中是 explicit linearly separated 的。
2.  **MLP Probe**: 加入 hidden layers，GeLU activation 和 LayerNorm。
    *   Test：信息存在但是呈 nonlinear manifold 结构，需要非线性的 readout mechanism 去 unfold。
3.  **Temporal Attentive Probe**: 直接在 token sequence 上操作。
    *   **Architecture**: 1 个 self-attention layer + 1 个 final cross-attention layer，16 个 attention heads。
    *   **Input**: 不做 pooling，直接输入 spatio-temporal token embeddings。
    *   **Intuition**: Average pooling 会把 time 这个 dimension 压扁，丢失 temporal interactions。Attention mechanism 允许 probe 动态地去 attend 到不同时间步的 tokens，去 model physical causal chain。这个设计极度贴近 [Bardes et al. (2024)](https://arxiv.org/abs/2404.08471) 里的 attentive probing 协议。

### 5. Experimental Results & Data Table Analysis

#### Analysis 1: Paradigm Comparison (Table 1 解析)

在 Temporal Attn. probe 下（因为它能 best leverage temporal structure）：

| Model | IntPhys2 (VOE) | MVP (Pair Acc) |
| :--- | :--- | :--- |
| V-JEPA 2.1 | **66.67%** | 93.73% |
| V-JEPA (base) | 56.86% | **94.03%** |
| VideoMAE | 58.82% | 92.01% |
| VideoMAE-v2 | 15.69%* | 91.10% |
| LTX-Video | 47.06% | 84.33% |

*VideoMAE-v2 在 IntPhys2 上的 15.69% 是个 outlier。Appendix A.6 解释了这是由于 undertraining 导致的 probe 学习不充分（延长 epochs 后从 15.69% 升到了 31.37%）。

**Intuition Build**: V-JEPA family 完胜。这证明在 latent space 做 predictive learning 比 reconstruct pixels 更能 extract abstract physics。VideoMAE 虽然在 MVP 上很 competitive，但在需要深度 physical understanding 的 IntPhys2 上 gap 明显。LTX-Video 落后最多，说明 denoising objective 虽然能生成逼真 video，但其 latent space 并没有为了 human-like physics reasoning 而 explicitly organize。

#### Analysis 2: Layerwise Emergence of Physics Information (Figure 2 解析)

Author 提取了 25%, 50%, 75%, 100% depth 的 features。
*   **MVP**: Performance 呈现清晰的 late-layer dominated 趋势。V-JEPA 在 MLP probe 下从 0.25 depth 的 59.45% 一路升到 final layer 的 87.26%。
*   **IntPhys2**: 不像 MVP 那样 monotonic。通常在 50% - 75% depth 处 peak，然后在 final layer 出现下降。比如 V-JEPA 2 在 0.75 depth 达到 56.86%，但在 final layer 掉到 47.06%。

**Intuition Build**: 为什么 IntPhys2 在最后一层会掉？这和 LLM 里的现象极其相似。最后一层的 features 被极度 specialized 用于 pretraining objective (比如 predicting next masked patch 或 latent state)。Physics 这种 general, abstract world knowledge 往往在 intermediate-to-late layers 形成了最优的 linear/non-linear representation manifold，但还没被最终 task objective "overfit" 或 "collapse" 到 output space。Intermediate layers 是最 rich 的泛化特征地带。MVP 要求的 minimal pair discrimination 更偏向 high-level semantic/visual mismatch，所以越往后越好；而 IntPhys2 要求细粒度的 physical plausibility ranking，中间层最 unpolluted。

对于 LTX-Video 的额外发现：作者固定 transformer block，改变 noise level (0.1 to 1.0)。发现 performance 并没有随着 denoising progress 单调提升，最强信号集中在 backbone 中间且 specific noise level 处。这说明 Diffusion model 的 denoising trajectory 并非一个 uniform 释放 physical information 的过程，它在中间阶段将 overall layout 和 motion resolve 后，后续的 refinement 反而会 focus on high-frequency pixel details，从而稀释了 physics representation。

#### Analysis 3: Probe Expressivity (Linear vs MLP vs Attn)

*   **MVP**: Linear probe 表现极差 (V-JEPA 只有 48.74%)。加 MLP 大幅升到 87.26%，加 Temporal Attn. 升到 94.03%。
*   **IntPhys2**: Linear probe 表现尚可 (V-JEPA 达到 50.98%)。MLP 帮助不大，Temporal Attn. 有明显提升 (56.86%)。

**Intuition Build**: MVP 里的 physical plausibility 信息在 feature space 里是高度 non-linear 且 tangled 的，单纯的 linear hyperplane 切不开，需要 strong readout 去 warp the space。而 IntPhys2 因为 scene 设计本身更具 contrastive 特质（clear possible vs impossible），部分信息在 linear space 就已经 separable，但利用 temporal attention 去显式捕捉 causal event 的 token-level interaction，依然能挖出更多 latent signal。

#### Analysis 4: Temporal Controls (Table 5 深度解析)

这是最 revealing 的实验，用两个极其 brutal 的控制条件去 cross-check model 到底有没有学到 dynamics。

1.  **Frame-shuffled**: 随机打乱 frame 顺序，保留 set of frames。
2.  **Single-frame**: 随机取 1 帧 repeat 填满整个 clip length。

Relative degradation formula: $\Delta_{\%} = 100 \times \frac{\text{Control} - \text{Main}}{\text{Main}}$

观察数据：
*   在 MVP 上，V-JEPA 的 Temporal Attn. 遭遇 shuffle 时掉 -23.33%，遭遇 single frame 掉 -88.71%。而 LTX-Video 的 Temporal Attn. 遭遇 shuffle 暴跌 -95.92%，single frame 暴跌 -96.52%。
*   在 IntPhys2 上，V-JEPA Temporal Attn. 遭遇 shuffle 降幅为 **0.00%**，single frame 掉 -75.86%。

**Intuition Build**: 
Firstly, Single-frame control 的毁灭性打击证明了：所有 model 都 heavily rely on multi-frame evidence，static appearance 搞不定 physics。这推翻了 model 仅仅在 memorize static visual priors 的假设。

Secondly, IntPhys2 上 shuffle drop 为 0.00% 极其 interesting。这说明虽然 model 需要 multiple frames，但它甚至不需要 frames 之间有正确的 temporal order！这就暴露了 IntPhys2 dataset 的一个 limitation：部分 VOE 信号可以通过 unordered multi-frame evidence（比如统计某些 objects 的 presence/absence 频次）就能提取，不需要真正的 causal physics trajectory modeling。相反，MVP 在 shuffle 下大规模 collapse，证明 MVP 真正 test 了 temporal order 和 causal dynamics。这也解释了为什么 MVP 对 Temporal Attn. probe 的需求极高。

### 6. Broader Connections & Final Thoughts

这篇 paper 的结果对于未来 world models 的设计有极强的指向性。在 LeCun 的 [JEPA framework](https://arxiv.org/abs/2301.08243) 中，predicting in latent space 从一开始就是为了 avoid the curse of dimensionality 和 pixel-level shortcuts。这篇 paper 用 IntPhys2 和 MVP 这两个极其 hard 的 benchmarks 证实了这一点：放弃 pixel reconstruction，转而在 joint-embedding space 预测 future representation，是目前让 model develop internal world model and intuitive physics 的最优 inductive bias。

如果联系到你之前在 [Tesla AI day 或其他探讨](https://www.youtube.com/watch?v=hx7LBaoB2KE) 中提到的 "system 1 vs system 2" 思维，V-JEPA 通过 pretraining 已经构建了一个相当强大的 "System 1" physics intuition engine。接下来的问题是如何把这个 frozen representation 接上 "System 2" 的 reasoning module。Paper 最后提到的 limitation 也很真实：confounding factors (比如 V-JEPA 用了 ViT-Gigantic 而 VideoMAE 只有 ViT-H) 没有 fully isolated。未来如果在 exact same architecture scale 上对比 V-JEPA vs VideoMAE 的 latent predictive pretraining，将能给出更纯粹的 causal claim。

Hope this breakdown builds your intuition on how video FMs are currently encoding physics. If you want to dig deeper into the V-JEPA architecture's specific masking strategy or the attention probe's gradient flow, let's discuss further!
