---
source_pdf: Einstein World Models.pdf
paper_sha256: 603087922559d8045e4831c18ae9c807f26bb1f331a7322f820141c86ea34096
processed_at: '2026-08-18T10:27:48-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，让我用最直白的大白话，结合底层的技术 intuition，带你重新 hardcode 一遍这篇 paper 的核心思想。

### 1. 核心直觉：给 LLM 装一个“心灵的眼睛”

你想象一下自己解一道物理题：一个杂技演员把蓝球抛起一米，紫球抛起两米，然后她慢吞吞地爬上高梯子，头上还顶着气球。问此时紫球在哪？

如果你是个纯 Text LLM，你的 working memory 里只有 token。你会试图用 text 去形式化这个问题：“设抛出速度为 $v$，重力加速度为 $g$，爬梯子时间为 $t$... 但等等，题目没给时间啊？”于是你卡在了 combinatorial explosion 的逻辑分支里。

但是人类怎么做的？人类会在脑海里“播放一段视频”：球往上飞，然后掉下来，啪啪两下落地，这时候那个女的才刚爬了两步梯子。你看一眼脑海画面的最后一帧，发现两个球都在地上了。得出答案：一样高。

EWM 的逻辑就这么简单：**让 LLM 在生成 text reasoning 的时候，能够主动“暂停”，调用一个外部的 video generator 放一段短视频，然后把视频看进来，再继续写 reasoning。**

这里的 "E" 既是 Einstein（爱因斯坦靠脑海中追光速的视频想出了相对论），也是 Externalised（把脑海中隐式的想象变成 context window 里显式可检查的 tokens）。

### 2. 架构拆解：Trace 是怎么拼起来的

在 implementation 层面，这其实就是一种特殊的 tool use。就像现在的 LLM 会调用 search engine 或者 calculator 一样，只不过这里调用的 tool 是一个 World-Module $\mathcal{W}$（比如 HunyuanVideo 或者 Sora 这样的 video diffusion model）。

对于输入 $x$，模型初始化 trace 为 $\mathcal{T}_0 = x$。接着 LLM 开始 autoregressive 地生成 token。在每一步 $t$，它有两种选择：

$$
\mathcal{T}_{t+1} = \mathcal{T}_{t} \oplus \left\{ \begin{array}{ll} s_t, & \text{if } \mathcal{W} \text{ is not queried}, \\ [q_t, v_t], & \text{if } \mathcal{W} \text{ is queried}. \end{array} \right.
$$

**变量解释**：
- $\mathcal{T}_t$: 到第 $t$ 步为止，context window 里的全部 reasoning trace。
- $\oplus$: 拼接操作符。
- $s_t$: 纯文本的推理 token（比如 "Let's think about the gravity..."）。
- $q_t$: LLM 生成的调用工具的 prompt（比如 `<tool_call>{"name": "world_module", "query": "A juggler throws a blue ball..."}</tool_call>`）。
- $v_t$: $\mathcal{W}$ 吐出来的视频帧。这些 frames 会被 vision encoder 转成 visual tokens，塞回 LLM 的 context window。

**Intuition 深度解析**：
为什么要强调 $v_t$ 是 inspectable hypothesis？因为现在的 video generator 物理直觉其实很烂，经常会把球穿过桌面，或者水往上流。但是在 EWM 框架下，这不要紧！EWM 并不把 $v_t$ 当作最终答案，而是把它当作一个“草案”。LLM 拿到这个视觉草案后，可以用它极强的 semantic reasoning 去做 sanity check：“等等，视频里这个球穿过桌子了，这违反物理常识，所以真实情况应该是球在桌子上。”这个 Visual-Text 的 round-trip，相当于把 LLM 内部 latent space 里混乱的物理概念，投影到一个 structured spatiotemporal manifold 上，然后再读回来，强行做了一次 dimensionality reduction 和 alignment。

### 3. 训练机制：SFT 与 RLVR 怎么配合

这里就涉及到非常经典的 LLM 训练范式了。你不可能直接拿个预训练模型就能让它完美地知道“什么时候该调用 video，调用的 prompt 怎么写，拿回来视频怎么用”。所以需要两阶段训练。

#### 阶段一：SFT (Supervised Fine-Tuning)
先教它格式。我们需要一批数据，里面有完整的 EWM trace，包含 `text` -> `tool_call` -> `visual_rollout` -> `text answer`。

算 cross-entropy loss 的时候有个大坑：那个 `visual_rollout` 里的 token 是 video generator 生成的，不是 LLM 生成的。如果你把这些 token 也算进 LLM 的 loss 里去预测，模型优化目标就乱了。

所以要用 Masked SFT loss：

$$
\mathcal{L}_{\text{SFT}}(\theta) = - \underset{(x, \mathcal{T}^\star) \sim \mathcal{D}_{\text{SFT}}}{\mathbb{E}} \left( \frac{1}{\sum_t \mathbb{I}_t} \sum_t \mathbb{I}_t \log \pi_\theta(z_t^\star | \mathcal{T}_{<t}^\star) \right)
$$

**变量解释**：
- $\mathcal{D}_{\text{SFT}}$: 监督学习数据集。
- $z_t^\star$: 第 $t$ 个 target token。
- $\mathbb{I}_t$: Indicator function。如果是 LLM 应该生成的 token（text 或 query），设为 1；如果是外部 $\mathcal{W}$ 吐回来的 video observation token，设为 0。
- 分母 $\sum_t \mathbb{I}_t$: 确保只对 policy 自己生成的 tokens 做平均，不被大量的 video tokens 稀释。

#### 阶段二：RLVR (Reinforcement Learning with Verifiable Rewards)
SFT 只是教会了模型“怎么调用”，但模型不知道“什么时候调用最划算”。调用 video gen 是极其消耗 compute 的，不能每道题都调用。这时候我们引入 GRPO (Group Relative Policy Optimization)。

我们设计一个 reward function：
$$
r_{\mathcal{M}}(\mathcal{T}, y^\star) = r(\hat{y}, y^\star) + r_{\mathcal{W}}(\mathcal{T})
$$
- $r(\hat{y}, y^\star)$: 最终答案对不对，对了给 1，错了给 0。
- $r_{\mathcal{W}}(\mathcal{T})$: 工具使用惩罚。比如每次调用扣 0.1 分，防止模型变成 call-abuse 狂魔。

然后针对同一个 prompt $x$，我们让旧的 policy 采样出 $\mathcal{G}$ 条不同的 trajectory，计算 group-relative advantage：
$$
A_i = \frac{r_i - \bar{r}}{s_r + \epsilon_{\text{adv}}}
$$
- $r_i$: 第 $i$ 条 trajectory 的总 reward。
- $\bar{r}$, $s_r$: 组内的均值和标准差。
- $\epsilon_{\text{adv}}$: 防止除零的极小数。

**这里的 Intuition 非常关键**：假设对于那道杂技题，模型采样了 8 条轨迹。4 条没有调用 $\mathcal{W}$，全答错了，reward 是 0。3 条调用了 $\mathcal{W}$ 但没用对视频信息，扣了 0.1，reward 是 -0.1。1 条调用 $\mathcal{W}$ 并看懂了球落地了，答对了，reward 是 0.9。
此时 $\bar{r}$ 很低，那条 0.9 的轨迹 $A_i$ 会是个极大的正数。GRPO 会极大地拉高这条轨迹里每一个 action 的 probability。模型就学到了：“遇到这种 temporal physics puzzle，我要主动写 prompt 去生成视频，然后看最后几帧。”

优化目标的完整公式（带 masking 和 clipping 的 PPO objective）：
$$
J_E(\theta) = \mathbb{E}_{x, \tau_{1:\mathcal{G}}} \left[ \frac{1}{\mathcal{G}} \sum_{i=1}^{\mathcal{G}} \frac{1}{L_i^g} \sum_{t=1}^{L_i} \mathbb{I}_{it} \min(\rho_{it} A_i, \text{clip}_\epsilon(\rho_{it}) A_i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$
- $L_i^g = \sum_{t=1}^{L_i} \mathbb{I}_{it}$: 第 $i$ 条 trajectory 里 policy 自己生成的 token 数。
- $\rho_{it} = \frac{\pi_\theta(z_{it} | \tau_{i, <t})}{\pi_{\text{old}}(z_{it} | \tau_{i, <t})}$: 重要性采样比率。
- $\text{clip}_\epsilon(\rho_{it})$: 截断在 $[1-\epsilon, 1+\epsilon]$，防止梯度爆炸。
- $\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$: 防止模型为了刷 reward 把自己的语言能力搞坏了，强行往 SFT 后的 reference policy $\pi_{\text{ref}}$ 上靠拢。
- $\mathbb{I}_{it}$: 依然是对 video rollout token 的 mask，梯度不回传给 video generator，video generator 在这里是个 frozen 的 environment。

### 4. World-Module 选型与 Ensembling 脑洞

论文里把 $\mathcal{W}$ 分成 Renderer, Simulator, Planner 三类。在 EWM 里，Planning 依然留在 LLM 脑子里，所以 $\mathcal{W}$ 主要是 Renderer（Text-to-Video）或者 Simulator（如 Genie 2 那种可干预的）。

**最有意思的技术联想**：Diffusion models 的 denoising objective 其实可以转化为 likelihood estimation。这意味着你可以算出一个 video rollout 符合直觉物理的概率。基于此，论文提出了一个 Ensembling 的脑洞：
如果接一个偏重真实感的 $\mathcal{W}_1$，一个偏重物理一致性的 $\mathcal{W}_2$。两个模型分别生成 rollout，如果它们的结果分歧很大，说明这个 counterfactual 假设触及了物理边界。这就像多个人对同一件事有不同想象，把分歧暴露出来，正是 debug 和深层次 reasoning 的切入点。

### 5. 为什么这是个巨大的 Dataset Call for Action

作者最后指出，限制这个范式起飞的是没有数据。现在的 benchmark 比如 PHYRE, CLEVRER 都是直接喂图或者视频给模型做 VQA。但 EWM 需要的是：纯 text 输入，模型自己决定要不要生成 video，然后输出 text 答案。

SimpleBench 就是那种数据，只有 200 多道题，人类准确率爆杀所有 frontier LLM，因为人类能在脑子里 roll out，而 LLM 只能死磕 text。EWM 的终极目标是训练出能自主决定“何时想象”的 LLM，这需要构建大规模的、包含 text + tool_call + visual rollout 交错结构的 dataset。

### 6. 总结 & 我的技术 Hallucination

如果把这个思路推到极致：现在我们在 RLVR 阶段是冻结 $\mathcal{W}$ 的，梯度只更新 $\pi_\theta$。如果未来算力管够，$\mathcal{W}$ 可以是 differentiable 的（比如基于 score matching 或者 flow matching 的可微渲染层）。把 final answer 的 reward signal 直接通过 chain rule 回传给 $\mathcal{W}$ 的 U-Net weights。
这意味着什么？LLLM 发现物理题做错了，它会通过梯度“责怪” video generator：“你刚才生成的球没有按重力掉下来，导致我答错了，你给我更新权重，下次生成准点。”
这将是真正的 multimodal end-to-end reasoning，不再是对接 frozen tool，而是 LLM 和 World Model 在 latent space 和 objective space 上的共生进化。

### References & Web Links

1. **SimpleBench (神级物理直觉测试集)**: [SimpleBench GitHub](https://github.com/SimpleBench/SimpleBench)
2. **DeepSeek-R1 (RLVR & GRPO 范式起源)**: [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.14221)
3. **Search-R1 (LLM 调用外部环境 RL 训练的代表作)**: [Search-R1 Paper](https://arxiv.org/abs/2503.09516)
4. **Genie 2 (Simulator 类 World-Module 代表)**: [Google DeepMind Genie 2 Blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
5. **HunyuanVideo (Renderer 类 World-Module 代表)**: [HunyuanVideo GitHub](https://github.com/Tencent/HunyuanVideo)
6. **Whiteboard-of-Thought (Static Visual Sketchpad 前身)**: [Whiteboard-of-Thought Paper](https://arxiv.org/abs/2406.04685)
7. **VideoPhy (衡量 Video Gen 物理常识的利器)**: [VideoPhy Project Page](https://videophy.github.io/)
8. **Thinking with Video (探索 Video Gen 作为 Reasoner 的前沿)**: [Thinking with Video Project](https://thinking-with-video.github.io/)

---

这篇 paper 提出 Einstein World Models (EWM)，核心思想是把 visual-temporal rollout 嵌入到 LLM 的 reasoning trace 中，从而让 LLM 能够进行可视化的 thought experiment。EWM 的 "E" 既是 Einstein 的首字母，也代表 Externalised，即把内部隐式的 visualization 变成外显的、可检查的 reasoning step。相比于传统的 Chain-of-Thought (CoT) 只在 text space 内进行自回归推理，EWM 允许 LLM 在推理的 sparse intermediate steps 调用一个 world-module $\mathcal{W}$，生成 short video sequence，然后将这个 video rollout 作为 inspectable hypothesis 反馈给 LLM 继续推理。这种机制类似于 LLM 调用 web search 或者 code interpreter，EWM 把 tool use 的范畴扩展到了 visual thought experiments。

下面我为你深入拆解这篇 paper 的技术细节，build 你的 intuition。

### 1. Inference 架构与 Trace 构建机制

在 inference 阶段，EWM 系统由两个核心组件构成：Einstein reasoner $\pi_\theta$ (参数为 $\theta$ 的 LLM policy) 和 world-module $\mathcal{W}$ (负责生成 video rollout 的 video generator，如 video diffusion model)。

对于纯 text 输入的问题 $x$，推理过程初始化为 $\mathcal{T}_0 = x$。随后在每个 step $t$，reasoner 定义一个 next segment 的 conditional distribution $\pi_\theta(\cdot | \mathcal{T}_t)$。这里生成的 segment 有两种可能：
- **Non-tool segment $s_t$**: 普通的 language reasoning 或者 final answer。
- **World-module query segment $q_t$**: 触发可视化的 query。

如果生成了 query $q_t$，world-module 会返回一个 visual-temporal rollout $v_t \sim \mathcal{W}(q_t)$。Trace 的更新规则如公式 (1) 所示：

$$
\mathcal{T}_{t+1} = \mathcal{T}_{t} \oplus \left\{ \begin{array}{ll} s_t, & \text{if } \mathcal{W} \text{ is not queried}, \\ [q_t, v_t], & \text{if } \mathcal{W} \text{ is queried}. \end{array} \right.
$$

**变量解析与 Intuition**:
- $\mathcal{T}_t$: 第 $t$ 步时的 partial reasoning trace。
- $\oplus$: 序列拼接操作符。
- $s_t$: 纯文本推理 token segment。
- $q_t$: LLM 生成的、用于调用 $\mathcal{W}$ 的 query prompt。
- $v_t$: 从 $\mathcal{W}(q_t)$ 采样返回的 video rollout (由 frames 序列组成)。

在实现中，trace 的序列化类似于目前的 ReAct 或者 Search-R1 范式，使用特殊的 tags 比如 `<tool_call>{"name": "world_module", "query": q_t}</tool_call>`，以及 `<visual_rollout> ... </visual_rollout>`。

**技术联想与 Hallucination**:
这里 $v_t$ 作为 video frames 返回时，实际上需要经过一个 vision encoder (比如 ViT 或者 spatial-temporal tokenizer) 转换成 visual tokens，然后注入到 LLM 的 context window 中。这个跨模态的对齐是关键。如果 world-module 是一个 latent diffusion model，$v_t$ 甚至可以直接在 latent space 中产生，然后通过一个 projection layer 映射到 LLM 的 input embedding space，从而避免 decode 到 pixel space 再 encode 的信息损耗。由于 autoregressive reasoning 和 video generation 都具有 unidirectional temporal structure，token-by-token 的生成与 frame-by-frame 的展开在维度上具有天然的同构性。这使得 LLM "等待" video rollout 完成后再继续生成 text token 变得非常自然。

### 2. Training Objective: SFT 与 RLVR 联合优化

训练 EWM 分为两个阶段：Supervised Fine-Tuning (SFT) 和 Reinforcement Learning with Verifiable Rewards (RLVR)。

#### 2.1 SFT 阶段：Format 学习与 Masked Cross-Entropy
在 SFT 阶段，目标是教会 reasoner $\pi_\theta$ 产生合法的 EWM trace 格式。由于 $v_t$ 是由 $\mathcal{W}$ 生成的 observation，属于环境反馈，而不是 reasoner 的 action，因此在计算 cross-entropy loss 时必须被 mask 掉，否则模型会试图去预测不属于自己控制的 video tokens，导致优化目标混乱。

Masked SFT loss 定义为：

$$
\mathcal{L}_{\text{SFT}}(\theta) = - \underset{(x, \mathcal{T}^\star) \sim \mathcal{D}_{\text{SFT}}}{\mathbb{E}} \left( \frac{1}{\sum_t \mathbb{I}_t} \sum_t \mathbb{I}_t \log \pi_\theta(z_t^\star | \mathcal{T}_{<t}^\star) \right)
$$

**变量解析**:
- $\mathcal{D}_{\text{SFT}}$: 监督数据集，包含 input $x$ 和 target trace $\mathcal{T}^\star$。
- $z_t^\star$: target trace 中第 $t$ 个 token。
- $\mathbb{I}_t$: Indicator function。如果 $z_t^\star$ 是由 reasoner 生成的 (text reasoning 或 query)，则 $\mathbb{I}_t = 1$；如果是 world-module 返回的 rollout observation，则 $\mathbb{I}_t = 0$。
- 分母 $\sum_t \mathbb{I}_t$ 确保了 loss 只在 policy-generated tokens 上做归一化。

#### 2.2 RLVR 阶段：GRPO-style Optimization
在 SFT warm-start 之后，采用 GRPO (Group Relative Policy Optimization) 算法进行强化学习，以优化最终答案的正确率并鼓励合理使用 world-module。

Reward 函数定义为：
$$
r_{\mathcal{M}}(\mathcal{T}, y^\star) = r(\hat{y}, y^\star) + r_{\mathcal{W}}(\mathcal{T})
$$

其中 $r(\hat{y}, y^\star)$ 是 final answer $\hat{y}$ 和 ground truth $y^\star$ 之间的 verifier reward (比如 exact match)。$r_{\mathcal{W}}(\mathcal{T})$ 是针对 world-module 调用行为的 shaping reward。为了防止模型过度调用 $\mathcal{W}$ (因为调用通常伴随高昂的 computational cost)，可以设置 $r_{\mathcal{W}}(\mathcal{T}) = -\lambda M(\mathcal{T}) / B$，其中 $M(\mathcal{T})$ 是 trace 中调用 $\mathcal{W}$ 的次数，$B$ 是 call budget，$\lambda$ 是惩罚系数。

GRPO 的目标函数 (公式 3) 如下：

$$
J_E(\theta) = \mathbb{E}_{x, \tau_{1:\mathcal{G}}} \left[ \frac{1}{\mathcal{G}} \sum_{i=1}^{\mathcal{G}} \frac{1}{L_i^g} \sum_{t=1}^{L_i} \mathbb{I}_{it} \min(\rho_{it} A_i, \text{clip}_\epsilon(\rho_{it}) A_i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$

**变量解析**:
- $\tau_{1:\mathcal{G}} = \{\tau_i\}_{i=1}^{\mathcal{G}}$: 从 frozen old policy $\pi_{\text{old}}$ 中针对同一个 input $x$ 采样出的 $\mathcal{G}$ 条 EWM trajectories。
- $A_i = \frac{r_i - \bar{r}}{s_r + \epsilon_{\text{adv}}}$: Group-relative advantage。$r_i$ 是第 $i$ 条 trajectory 的总 reward，$\bar{r}$ 和 $s_r$ 是 group 内 reward 的均值和标准差。
- $L_i^g = \sum_{t=1}^{L_i} \mathbb{I}_{it}$: 第 $i$ 条 trajectory 中由 policy 生成的 token 数量。
- $\rho_{it} = \frac{\pi_\theta(z_{it} | \tau_{i, <t})}{\pi_{\text{old}}(z_{it} | \tau_{i, <t})}$: Importance sampling ratio (重要性采样比率)。
- $\text{clip}_\epsilon(\rho_{it})$: 将 $\rho_{it}$ 截断在 $[1-\epsilon, 1+\epsilon]$ 区间内，防止 importance weight 爆炸。
- $\beta$: KL penalty 系数。
- $\pi_{\text{ref}}$: 参考策略 (通常是 SFT 后的模型)，用于防止 RL 阶段模型发生 catastrophic forgetting 或 reward hacking 导致语言能力崩溃。
- $\mathbb{I}_{it}$: 同样是 masking function，如果 token $z_{it}$ 是 $\mathcal{W}$ 返回的 observation，则 $\mathbb{I}_{it} = 0$，梯度不回传给 $\pi_\theta$。

**Intuition 深度解析**:
RLVR 训练的核心难点在于 credit assignment。如果一条 trace 最终答案对了，模型怎么知道是因为中间调用了 $\mathcal{W}$ 得到了 rollout，还是因为自身的 text reasoning 就足够了？GRPO 通过 group-relative baseline 自然地解决了这个问题：如果大多数没有调用 $\mathcal{W}$ 的 trajectory 都失败了，而那条调用了 $\mathcal{W}$ 并利用了 rollout 信息的 trajectory 成功了，那么 $A_i$ 会显著偏高，从而增强生成那个 `<tool_call>` query 以及后续利用 `<visual_rollout>` 进行推理的 action probability。反之，如果调用 $\mathcal{W}$ 但答案依然错误，或者 $r_{\mathcal{W}}$ 的惩罚超过了正确答案带来的收益，模型就会学会在不需要 visualization 的时候直接输出 text。

### 3. World-Module $\mathcal{W}$ 的选择与 Ensembling

World-module 是 EWM 的物理直觉引擎。Paper 将其分为三类：

1. **Renderers**: 直接基于 text prompt 生成 video frames 的模型。目前的主流是 diffusion-based 或 flow-matching video generators (如 HunyuanVideo, Wan, LTX-Video)。它们是无条件或条件式的 future predictor。
2. **Simulators**: 允许 reasoner 在 visualised world 中进行干预并观察后果。比如 Genie 系列这种 interactive world model。
3. **Planners**: 在 EWM 框架下，planning 依然保留在 LLM reasoner 内部，因为 EWM 的目的是进行 thought experiment，而非 embodied robot action。

**World-Module Quality & Ensembling**:
Paper 提到了 diffusion models 作为 $\mathcal{W}$ 的一个巨大优势：可以通过其 denoising objective 计算出 human-verifiable likelihood estimates，从而衡量视频模型的 intuitive physics 质量。

更进一步，EWM 支持 **Ensembling**。不同的 LLM reasoners 连接具有不同 inductive biases 的 world-modules (例如，一个侧重 visual realism，一个侧重 physical consistency，一个侧重 temporal continuity)。这些 reasoners 可以交换它们的 rollouts 和 critiques。由于 rollouts 是 externalised 的，它们之间的 disagreement (分歧) 直接暴露了不同 visualisation 假设下的逻辑分歧点，这为后续的 inspection 提供了明确的靶标。这种多模型辩论机制类似于 LLM 领域的 Multi-Agent Debate，只不过辩论的介质是 visual hypotheses。

### 4. Dataset 瓶颈与 SimpleBench

这篇 paper 的一个重要贡献是指出了当前领域的 dataset vacuum。现有的 physical reasoning datasets (如 PHYRE, CLEVRER, IntPhys) 往往已经提供了 visual scene 作为 input。而 EWM 需要的是一种全新的 setting：纯 text-only 输入，但模型需要自己决定是否生成 visualization。

Paper 引用了 **SimpleBench** 作为这类任务的理想雏形。SimpleBench 只有 200 多道题，纯 text 描述，但人类准确率远超当前的 frontier LLMs。

**Juggler 抛球问题分析**:
> 杂技演员把一个蓝色实心球抛向空中一米高，然后又把一个紫色实心球抛向空中两米高。她随后小心翼翼地爬上一把高梯子的顶端，头上顶着一个黄色气球。此时紫色球最可能在哪？
> A. 和蓝色球同高
> B. 和黄色气球同高
> C. 在蓝色球内部
> D. 在黄色气球上方
> E. 蓝色球下方
> F. 蓝色球上方
> 正确答案: A.

**Intuition 解释**: 纯 text LLM 容易过度形式化，认为缺少抛出速度和爬梯子时间的参数而无法判断。但在人类直觉中，抛出一两米高的球只需要不到一秒就会落地，而爬梯子并保持平衡需要长得多的时间。当杂技演员爬到梯顶时，两个球早就掉在地上了，所以它们处于同一高度 (地面)。

在 EWM 中，LLM 在读到这个 prompt 时，会触发生成一个 query $q_t$ 传给 $\mathcal{W}$，$\mathcal{W}$ 生成一段杂技演员抛球、爬梯子、球落地的 video rollout $v_t$。LLM 观察这个 $v_t$ 的最后几帧，发现蓝紫两球都在地上，从而输出答案 A。这就把 latent 的 temporal simulation 外化为了显式的 video observation。

### 5. 相关工作的对比定位

- **Chain-of-Thought (CoT)**: 只在 text space 展开。无法捕捉 object identity, containment, contact, heat, motion 等 commonsense variables。
- **VLMs (Vision-Language Models)**: 接收外部提供的 visual input，但不能在推理过程中主动 "想象" 并生成 visual rollout 作为中间步骤。
- **VL-JEPA / World Models**: 学习 passive visual prediction，通常 tethered to observed states 或 chosen actions。EWM 的 thought experiment 可以是 counterfactual (反事实) 的，类似 Einstein 追光实验，无需真正执行或经历过。
- **Whiteboard-of-Thought / Visual Sketchpad**: 给 multimodal LLM 一个 visual scratchpad，让它们通过 code 画图 (bounding box, 辅助线等) 辅助推理。这些是 static visual annotations，而 EWM 产生的是动态的 visual-temporal rollout。
- **Visualization-of-Thought (VoT)**: 通过 text-form grids 或 maps 进行 state tracking，本质还是 symbolic manipulation，没有跳出 text modality。
- **Thinking with Video (Tong et al., 2025)**: 探讨 video generation model 本身能否作为 reasoner 产出带答案的视频。EWM 的 philosophy 截然不同：EWM 保持 LLM 作为 reasoner，video generator 仅仅是它调用的一个 cognitive tool。

### 总结与 Intuition Building

EWM 的深层 intuition 在于 **Cognitive Externalization**。人类在思考复杂物理或空间问题时，会在脑海中 (或纸上) 模拟场景演化，这是一种 offloading，把工作记忆无法承载的复杂变量转移到视觉皮层处理。LLM 的 text context window 相当于它的 working memory，当面对高维度的 spatiotemporal dynamics 时，text representation 会遭遇 combinatorial explosion (组合爆炸)。

EWM 本质上是在 reasoning trace 中创造了一个 "hallucination 通道"。这个通道把 LLM 对物理世界隐含的、碎片化的理解，通过一个 specialized decoder ($\mathcal{W}$) 投影到一个 structured spatiotemporal manifold (video pixels) 上，然后再通过 vision encoder 重新读回 LLM 的 context。这个 "生成-再观测" 的 round-trip 起到了一种类似 denoising 或 iterative refinement 的作用。即使 $\mathcal{W}$ 生成的视频不是 100% 物理准确的，只要它能把大致的时间线、空间遮挡、重力下落等 low-dimensional physics constraints 显式化，LLM 就能利用其强大的 semantic reasoning 能力对这个 visual hypothesis 进行纠错和推理。

未来的 frontier 在于训练 $\mathcal{W}$ 和 LLM 的 end-to-end alignment。如果在 RLVR 阶段，梯度只通过 LLM 的 text reasoning 回传，而 $\mathcal{W}$ 是 frozen 的，那么 LLM 只能去适应 $\mathcal{W}$ 的 inductive bias。若日后算力允许，通过 reward signal 回传梯度到 $\mathcal{W}$ (比如通过不同的iable rendering 或 score matching) 微调其 video generation 分布，使其生成对 reasoning 最有帮助的 rollout，这将会是下一代多模态 foundation model 的终极形态。

### References & Web Links

1. **DeepSeek-R1 / GRPO 机制**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
2. **SimpleBench**: [SimpleBench: Text benchmark where unspecialized human performance exceeds frontier models](https://simplebench.github.io/)
3. **Toolformer**: [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
4. **Search-R1**: [Search-R1: Training LLMs to Reason and Leverage Search Engines with RL](https://arxiv.org/abs/2503.09516)
5. **Whiteboard-of-Thought**: [Whiteboard-of-Thought: Thinking Step-by-Step Across Modalities](https://arxiv.org/abs/2406.04685)
6. **Visual Sketchpad**: [Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal LMs](https://arxiv.org/abs/2406.0994)
7. **V-JEPA 2.1**: [V-JEPA 2.1: Unlocking dense features in video self-supervised learning](https://arxiv.org/abs/2603.14482) (基于 arXiv 编号推测的 future work link)
8. **Genie 2 (Simulator example)**: [Genie 2: A Large-Scale Foundation World Model](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
9. **HunyuanVideo (Renderer example)**: [HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603)
10. **Thinking with Video**: [Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm](https://thinking-with-video.github.io/)
11. **Platonic Representation Hypothesis**: [Position: The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.10318)
12. **Causal-JEPA**: [Causal-JEPA: Learning world models through object-level latent interventions](https://arxiv.org/abs/2602.11389)
