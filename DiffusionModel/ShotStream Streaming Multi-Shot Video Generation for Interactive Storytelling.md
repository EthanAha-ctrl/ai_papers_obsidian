---
source_pdf: ShotStream Streaming Multi-Shot Video Generation for Interactive Storytelling.pdf
paper_sha256: ddb891b145332a7d31149c5c504b9d7a7091c343129c5fcb3289e626c8061d44
processed_at: '2026-08-12T05:51:12-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ShotStream 用人话讲

## 1. 它到底在干嘛

你想做个讲故事的 AI，生成一段有多个镜头切换的视频。比如你输入"镜头一：一个金发女人严肃地说话；镜头二：一个亚洲女人自信微笑"，模型要生成连续的、人物身份保持一致的两段视频。

现有做法的问题：模型一次性吃进所有 prompt，然后 bidirectional attention 把所有 shots 一起做。你要是想改其中一个 shot，得整个重新生成。HoloCine 生成 240 帧要 25 分钟，基本没法用。

ShotStream 换了个思路：**像 LLM 做 next-token prediction 一样，做 next-shot prediction**。生成完第一个 shot，你看一眼觉得 OK，再给它下一个 shot 的 prompt，它基于之前的内容继续生成。用户就像导演，实时指挥故事走向。

---

## 2. 核心思路拆解

### 2.1 第一步：训练一个 Bidirectional Teacher

先 fine-tune Wan2.1-1.3B，让它学会"给定历史 shots，生成下一个 shot"。

几个关键 trick：

**Dynamic sampling**：历史 shots 有几百帧，全保留太贵。比如 context budget 是 6 帧，有 3 个历史 shots，就从每个 shot 采样 2 帧。剩下的 budget 给最近的 shot 多分点。这比"每个 shot 只取第一帧"或"取首尾帧"效果好，Table 3 里 Semantic Consistency 从 0.671 提到 0.762。

**Multi-caption**：历史 shot 的 frames 不能用 target shot 的 caption。每个 shot 的 frames 要 attend 到自己的 local caption + 全局 global caption。这个细节很容易被忽略，但效果显著——Text Alignment 从 0.194 跳到 0.234。intuition 是：历史画面和当时的文字描述是 binding 在一起的，你扔掉历史 caption 就等于扔掉了"这个画面在讲什么"的语义锚点。

**Frame Concat 而不是 Channel Concat**：condition tokens 和 noise tokens 沿时间维拼接，让 3D self-attention 直接建模它们的关系。比 channel concat 好不少（Aesthetic 0.571 vs 0.509）。

**只训 3D attention**：cross-attention 和 FFN 冻住。反直觉但 work，Dynamic Degrees 从 60.85 涨到 63.06。说明 base model 的 text grounding 已经足够好，你只需要调整它的时序建模能力。

### 2.2 第二步：蒸馏成 4-step Causal Student

Teacher 要 50 步去噪，太慢。用 DMD 蒸馏成 4 步 causal model。

DMD 的核心想法：不要 match teacher 和 student 的逐个 output，而是 match 它们的 output distribution。通过两个 score function 的差来近似 reverse KL 的梯度，一个 score 学真实数据分布，一个 score 学 student 的输出分布，差值告诉你 student 应该往哪个方向调。

数学上：
$$\nabla_\phi \mathcal{L}_{\text{DMD}} \approx -\mathbb{E}_t \int (s_{\text{data}} - s_{\text{gen},\xi}) \frac{dG_\phi(\epsilon)}{d\phi} d\epsilon$$

这里 $s_{\text{data}}$ 是真实数据分布的 score，$s_{\text{gen},\xi}$ 是 student 输出分布的 score，$G_\phi(\epsilon)$ 是 student generator。直觉就是：两个 score 的差指明了 student 要怎么移动才能贴近真实分布。

---

## 3. 两个真正关键的创新

### 3.1 Dual-Cache + RoPE Discontinuity

转成 causal 后，推理时要维护两套 context：

- **Global cache**：存历史 shots 的 sparse frames，保证跨 shot 的人物、场景一致
- **Local cache**：存当前 shot 内已生成的 chunk，保证 shot 内的连续性

问题来了：直接拼接送进 causal model，RoPE 的 temporal position 是连续递增的，模型根本分不清某个 frame 是"历史回放"还是"当前 shot 的前面几帧"。这就导致 attention 权重分配混乱，shot 之间的 identity 会串味。

解决办法特别 elegant：给每个 shot 加一个 phase offset。第 $k$ 个 shot 的第 $t$ 帧的 RoPE 旋转角度是：
$$\Theta_t = \phi t + k\theta$$

$\phi$ 是 base frequency，$\theta$ 是 shot boundary 的 phase jump。这样每个 shot 之间会有一个明显的"断层"，模型看到 phase 跳变就知道"哦，这里换 shot 了"。

这个 trick training-free，不增加参数，但效果比 learnable embedding 还好（Sub. Cons. 0.825 vs 0.811）。Table 4 的 ablation 说得很清楚：不加 indicator，inter-shot Sub. Cons. 掉到 0.507，加了就回到 0.654。

intuition：RoPE 本来就是 encoding 时序位置的，shot boundary 是时序里的一种特殊事件，给它一个显式的 marker 是最自然的做法。

### 3.2 Two-Stage Distillation

Error accumulation 是 autoregressive generation 的老大难。训练时你用 ground-truth history，推理时 history 是自己生成的，有误差会累积放大。

**Stage 1：Intra-shot self-forcing**

History 用 ground-truth，但当前 shot 内部是 chunk-by-chunk causal rollout。相当于说"历史我给你保证是完美的，你先学会怎么 causally 生成一个 shot"。500 步就收敛。

**Stage 2：Inter-shot self-forcing**

History 也换成自己生成的，完整 shot-by-shot rollout，模拟真实推理。对每个新 shot 应用 DMD loss，前一个 shot 生成完就更新 global cache，local cache 清零。用 5-shot 子集训 1000 步，LoRA tuning。

ablation 说明两阶段缺一不可：
- 只 Stage 1：inter-shot Sub. 0.604，还凑合
- 只 Stage 2：inter-shot Sub. 0.583，更差，因为模型连基本 next-shot 能力都没学会就被迫处理 imperfect history
- 两阶段：0.654，最优

这是 curriculum learning 的经典思路：先在干净环境学基本技能，再在 noisy 环境学 error recovery。一步到位的训练方式相当于让模型同时学两件事，梯度信号互相干扰。

---

## 4. 结果到底多好

Quantitative（Table 1）：
- FPS 15.95，比 bidirectional 快 25 倍
- Inter-shot Subject Consistency 0.654，比第二名 LongLive 0.594 高 10%
- Transition Control 0.978，碾压所有 baseline（第二名 Infinity-RoPE 0.715）

Transition Control 这个指标特别值得关注。它衡量模型能不能精确地在 shot boundary 处做切换，而不是模糊过渡。RoPE discontinuity indicator 的作用在这里体现得淋漓尽致——模型真的"学会"了在 boundary 处干净利落地切换 scene。

User study：54 人投票，24 个 prompt，ShotStream 在 visual consistency、prompt following、visual quality 三个维度都被 76-87% 的用户选中。这是压倒性的 preference。

---

## 5. 我的联想

这个 work 让我想到几条线索：

**跟 LLM 走的路高度相似**。BERT 是 bidirectional，GPT 是 causal autoregressive，最后 GPT 赢了。Video generation 现在也在走这条路——从 bidirectional multi-shot model 到 causal next-shot predictor。ShotStream 就是这个方向的 GPT 时刻。

**Dual-cache 跟 human cognition 的类比**。Global cache 像 episodic memory，记住故事大背景；local cache 像 working memory，处理当前 scene 的细节。RoPE discontinuity 像是 cognitive 里的 "event boundary"——心理学研究早就发现人类也是这样切分连续体验的。

**Self-forcing 跟 RL 的类似**。Self-forcing 让模型 condition on 自己的输出，跟 RLHF 里 model roll out 自己的 trajectory 然后从 feedback 里学的逻辑很像。Stage 1 → Stage 2 的 progression 也像 SFT → RLHF 的 curriculum。更进一步想，未来 video generation 可能也会引入真正的 reward signal，不只是 DMD 的 distribution matching。

**Context budget 的动态调整是个 open problem**。现在固定 6 帧，shot 数多了可能不够。未来可以做 hierarchical memory——远期 shot 用更稀疏的 representation，近期 shot 用更密集的。或者引入 retrieval mechanism，根据当前 prompt 动态 retrieve 相关的历史 frames。

**Base model 的 scale up**。Paper 自己承认 1.3B backbone 在复杂场景下会出 artifact。如果换成 14B 或更大的 Wan 模型，加上这个 causal framework，质量会进一步提升。但训练成本也会指数增长，蒸馏的稳定性可能要重新调。

**Sparse attention 的集成**。现在 global cache 是 2 chunks，local cache 是 7 chunks，总共才 9 chunks，attention 成本还好。但要想支持更长 context（比如 10+ shots），sparse attention 或 attention sink 是必须的。LongLive 的 KV-recache 思路可以借鉴——定期回收远的 KV，只保留关键信息。

---

## 6. 关键 References 的链接

- ShotStream 项目主页: https://luo0207.github.io/ShotStream/
- Wan2.1 base model: https://arxiv.org/abs/2503.20314
- CausVid（蒸馏的基础）: https://arxiv.org/abs/2412.07772
- Self Forcing: https://arxiv.org/abs/2506.08009
- LongLive: https://arxiv.org/abs/2509.22622
- DMD2: https://arxiv.org/abs/2305.17439
- HoloCine（对比 baseline）: https://arxiv.org/abs/2509.22622
- VBench（评测工具）: https://vchampion.github.io/VBench-project/

---

## 7. 一句话总结

ShotStream 把 multi-shot video generation 重新 formulate 成 next-shot prediction，用 dual-cache + RoPE discontinuity 解决跨 shot 一致性，用 two-stage self-forcing 解决 error accumulation，最终跑出 16 FPS 的实时交互式多镜头叙事生成。这是 video generation 从 bidirectional 走向 causal autoregressive 的一个标志性工作。

---

# ShotStream 深度解析

这篇 paper 来自 CUHK MMLab 和 Kuaishou 的合作，提出了一个 causal multi-shot video generation 架构，实现了 16 FPS 的实时流式多镜头叙事视频生成。下面我从多个维度详细拆解。

---

## 1. 核心问题与动机

现有的 multi-shot video generation 方法（如 LCT、MoC、HoloCine）大多采用 **bidirectional architecture**，即所有 shots 在一个 sequence 内做 full attention。这类方法存在两个根本性问题：

- **Lack of interactivity**：所有 prompts 必须预先给定，无法在 runtime 流式输入和调整
- **High latency**：bidirectional attention 的计算量与 context length 成 quadratic 关系，例如 HoloCine 生成 240 frames 需要 25 分钟

ShotStream 的核心 insight 是：**把 multi-shot generation 重新 formulate 为 next-shot prediction task**，就像 LLM 的 next-token prediction 一样，每个 shot 基于历史 shots autoregressive 生成。这样自然支持 streaming prompts，用户可以像导演一样实时指挥故事走向。

---

## 2. 整体架构：两阶段训练 Pipeline

### Stage 1: Bidirectional Next-Shot Teacher Model

第一步是 fine-tune 一个 text-to-video 模型（Wan2.1-T2V-1.3B）成为 bidirectional next-shot generator。关键设计：

#### (a) Dynamic Sampling Strategy

给定 $S_{\text{hist}}$ 个历史 shots 和最大 context budget $f_{\text{context}}$（实验中设为 6 frames），从每个历史 shot 采样 $\lfloor f_{\text{context}} / S_{\text{hist}} \rfloor$ 帧。剩余 budget 分配给最近的 shot，充分利用 budget。

这里的 intuition 很清晰：**历史 shots 蕴含大量冗余信息，没有必要全部保留**。但均匀采样又可能丢失最近 shot 的重要细节，所以 dynamic sampling 在"覆盖所有历史"和"保留最新信息"之间做了 trade-off。

#### (b) Context Frame Captioning

这是这篇 paper 一个细节但关键的 insight。传统方法（如 FullDiT、CamCloneMaster、UniIC）在做 condition frame injection 时，对所有 condition frames 和 target frames 统一使用 target frame 的 caption。这在 next-shot generation 中是有问题的：**历史 shot 的 caption 包含了将过去 visual information 与文本描述 binding 的重要信息**，对于模型理解"接下来该生成什么"至关重要。

具体实现：每个 shot 的 frames 通过 cross-attention 同时 attend 到 global caption（描述整个叙事弧线）和该 shot 的 local caption（描述具体动作和内容）。

#### (c) Temporal Token Concatenation

condition frames $V_{\text{context}}$ 经过 3D VAE $\varepsilon$ 编码成 latents：
$$z_{\text{context}} = \varepsilon(V_{\text{context}}) \in \mathbb{R}^{f_{\text{context}} \times c \times h \times w}$$

其中 $f_{\text{context}}$ 是 condition frame 数，$c$ 是 channel 数，$h \times w$ 是空间分辨率。

然后 patchify 成 tokens：
$$x_j = \text{Patchify}(z_j), \quad z_j \in \{z_{\text{context}}, z_t\}$$

condition tokens $x_{\text{context}} \in \mathbb{R}^{b \times f_{\text{context}} \times s \times d}$ 和 noisy target tokens $x_t \in \mathbb{R}^{b \times f \times s \times d}$ 沿 frame dimension concat：
$$x_{\text{input}} = \text{FrameConcat}(x_{\text{context}}, x_t) \in \mathbb{R}^{b \times (f_{\text{context}} + f) \times s \times d}$$

其中 $b$ 是 batch size，$s$ 是每帧的 spatial token 数，$d$ 是 feature dimension。Noise 只加到 target tokens，context tokens 保持 clean。

这个设计的优势：**完全复用 DiT 原生的 3D self-attention，不需要新增任何模块或参数**。对比 channel concat 方式（如 [31] 中 MotionStream 的做法），frame concat 让 attention 能直接建模 condition 和 target 之间的时序关系，效果更好（Table 3 中 Aesthetic Quality 0.571 vs 0.509）。

#### (d) 只训练 3D self-attention

一个有趣的 ablation：只 fine-tune 3D spatial-temporal attention layers，比 full-parameter fine-tuning 效果更好（Sub. 0.825 vs 0.816）。这暗示 base T2V model 的 cross-attention 和 FFN 已经足够好，只需要调整时序建模能力。

---

### Stage 2: Causal Architecture via Distribution Matching Distillation

Bidirectional teacher 需要 ~50 denoising steps，延迟过高。通过 DMD 蒸馏成 4-step causal student。

#### Distribution Matching Distillation (DMD) 公式

DMD 的核心是 minimize reverse KL divergence between smoothed data distribution $p_{\text{data}}$ 和 student generator 的 output distribution $p_{\text{gen}}$：

$$\nabla_\phi \mathcal{L}_{\text{DMD}} \triangleq \mathbb{E}_t \big( \nabla_\phi \text{KL}(p_{\text{gen},t} \| p_{\text{data},t}) \big)$$

通过两个 score function 的差来近似梯度：
$$\approx -\mathbb{E}_t \bigg( \int \big( s_{\text{data}}(\Psi(G_\phi(\epsilon), t), t) - s_{\text{gen},\xi}(\Psi(G_\phi(\epsilon), t), t) \big) \frac{dG_\phi(\epsilon)}{d\phi} d\epsilon \bigg)$$

变量解释：
- $\phi$: student generator $G_\phi$ 的参数
- $\epsilon$: 随机 Gaussian noise
- $\Psi$: forward diffusion process
- $t$: random timestep
- $s_{\text{data}}$: 在真实数据分布上训练的 score function
- $s_{\text{gen},\xi}$: 在 student generator 输出分布上训练的 score function，参数为 $\xi$
- $G_\phi(\epsilon)$: student 生成的样本

intuition：**两个 score function 的差值告诉我们 student 输出分布应该往哪个方向移动才能 match 真实数据分布**。这是 DMD2 的核心思想，比 GAN-style 的对抗训练更稳定。

---

## 3. 两大核心创新

### Innovation 1: Dual-Cache Memory + RoPE Discontinuity Indicator

这是这篇 paper 最 elegant 的设计。

#### 问题背景

转成 causal 架构后，需要同时维护两类 context：
- **Global cache**：存储 sparse conditional frames from historical shots，保证 inter-shot consistency
- **Local cache**：存储当前 shot 内已生成的 frames，保证 intra-shot consistency

实验设置：chunk size = 3 latent frames，global cache = 2 chunks，local cache = 7 chunks。

#### 问题：Temporal Ambiguity

如果 naive 地把两个 cache 直接拼接送进 causal model，模型会**无法区分某个 frame 是来自历史 shot 还是当前 shot**，因为 RoPE 的 temporal position 是连续递增的。这会导致 attention 权重分配混乱。

#### 解决方案：Discontinuous RoPE

对第 $k$ 个 shot 中的第 $t$ 个 latent $z_t$，其 temporal rotation angle 设为：
$$\Theta_t = \phi t + k\theta$$

其中：
- $\phi$: base temporal frequency
- $t$: shot 内的 frame index
- $k$: shot index
- $\theta$: phase shift，表示 shot boundary 的 discontinuity

这样每个 shot 之间会有一个明显的 phase jump，**模型通过 RoPE 的 phase 就能明确知道"这里有一个 shot boundary"**。这是 training-free 的，不需要额外参数，比 learnable embedding 效果更好（Table 4: Sub. 0.825 vs 0.811）。

intuition：**RoPE 本质上是在 encoding 时序位置信息，shot boundary 是一种特殊的时序事件，应该有显式的 encoding**。用 phase shift 而不是 learnable embedding 的好处是，它是一种"结构化"的归纳偏置，泛化性更好。

### Innovation 2: Two-Stage Distillation Strategy

这是解决 autoregressive generation 中 **error accumulation** 的关键。核心问题是 train-test gap：训练时 condition on ground-truth history，推理时 condition on 自己生成的不完美 history。

#### Stage 2.1: Intra-Shot Self-Forcing

- Global context: 来自 **ground-truth** historical shots
- Local context: 来自 **self-generated** chunks（当前 shot 内 chunk-by-chunk rollout）
- 训练目标：建立 next-shot generation 的基础能力

这一阶段相当于"我给你完美的历史，你只需要学会怎么 causally 生成一个 shot"。

#### Stage 2.2: Inter-Shot Self-Forcing

- Global context: 来自 **self-generated** historical shots
- 完整的 shot-by-shot rollout，模拟真实推理过程
- 只对 newly generated shot 应用 DMD loss
- 使用 5-shot 子集训练

这一阶段相当于"现在历史也是你自己生成的，你需要学会从自己的错误中恢复"。

#### 为什么需要两阶段

Table 4 的 ablation 很清楚：
- Stage 1 Only：Inter-shot Sub. Cons. 0.604，还行但不够好
- Stage 2 Only：Inter-shot Sub. Cons. 0.583，更差，因为模型还没学会基本的 next-shot 能力就被迫处理 imperfect history
- Two Stage：0.654，最优

intuition：**这是一个 curriculum learning 的思想**。先学会"在完美条件下完成任务"，再学会"在不完美条件下完成任务"。直接跳到第二阶段，模型既要学基本能力又要学 error recovery，负担太重。

---

## 4. 实验结果分析

### Quantitative Results (Table 1)

对比了 7 个 baselines：
- Bidirectional: Mask2DiT, EchoShot, CineTrans
- Causal: Self Forcing, LongLive, Rolling Forcing, Infinity-RoPE

ShotStream 的关键优势：
- **Intra-shot Sub. Cons.**: 0.825（最高）
- **Inter-shot Sub. Cons.**: 0.654（最高，比第二名 LongLive 的 0.594 高 10%）
- **Transition Control**: 0.978（碾压所有 baseline，第二名 Infinity-RoPE 0.715）
- **FPS**: 15.95（与 causal baselines 持平，比 bidirectional 快 25×）

Transition Control 的高分特别值得注意，这说明 **RoPE discontinuity indicator 真的让模型学会了在 shot boundary 处做明确的切换**，而不是模糊过渡。

### User Study (Table 2)

54 个参与者，24 个 multi-shot prompts，8 个方法同时对比。ShotStream 在三个维度上都被 76%-87% 的用户选中，这是非常 strong 的 preference。

### Ablation Studies

#### Bidirectional Teacher (Table 3)

四个 ablation 都很 informative：
1. **Sampling Strategy**: Dynamic > First & Last > First Only（Semantic 0.762 vs 0.709 vs 0.671）
2. **Captioning**: Multi-Captions > Target Caption（Semantic 0.762 vs 0.681，Text Align 0.234 vs 0.194）
3. **Injection**: Frame Concat > Channel Concat（Sub. 0.825 vs 0.814，Bg. 0.819 vs 0.802）
4. **Training**: Only 3D > Full（Aesthetic 0.571 vs 0.546，Dynamic Degrees 63.06 vs 60.85）

#### Causal Student (Table 4)

Dual-Cache Distinction 和 Two-Stage Distillation 都被验证为必要。特别是 **Stage 1 Only vs Ours 的 Dynamic Degrees 差异（59.66 vs 63.06）**，说明 inter-shot self-forcing 不仅能提升 consistency，还能让视频更 dynamic——这可能是因为模型学会了在保持 identity 的同时允许 motion 多样性。

---

## 5. 实现细节

- Base model: Wan2.1-T2V-1.3B，分辨率 832×480
- 训练数据：320K multi-shot videos，每个 2-5 shots，最多 250 frames
- Hardware: 32× NVIDIA H800
- Teacher training: 10K steps, lr=1e-5, batch=64
- Causal adaptation: 5K ODE pairs, 2K steps, lr=1e-6
- Stage 1 distillation: 500 steps, generator lr=2e-6, critic lr=4e-7, 1:5 update ratio
- Stage 2 distillation: 1K steps with LoRA

推理时 shot-by-shot 生成，每个 shot 内 chunk-by-chunk，利用 KV caching 加速。

---

## 6. 与相关工作的关系

### Multi-Shot Video Generation 谱系

- **Keyframe-based** (StoryDiffusion, VideoGen-of-Thought, Captain Cinema): 生成 keyframes 然后用 I2V 扩展，intra-shot 内容孤立
- **Unified Sequence Modeling** (LCT, MoC, HoloCine): bidirectional full attention，缺乏交互性
- **ShotStream**: causal autoregressive，兼顾交互性和质量

### Autoregressive Long Video Generation 谱系

- **CausVid** [49]: 4-step causal generator via distillation，ShotStream 的技术基础
- **Self Forcing** [12]: condition on self-generated outputs，ShotStream 的核心思想来源
- **Rolling Forcing** [20]: condition on progressive noise levels
- **LongLive** [43]: KV-recache 机制支持 dynamic prompting
- **Infinity-RoPE** [45]: training-free RoPE Cut

ShotStream 的贡献在于把 autoregressive video generation 从 single-scene 扩展到 **multi-shot narrative**，这是关键一步。

---

## 7. 我的 Intuition 与思考

### 为什么这个 approach 行得通

1. **Next-shot prediction 是 multi-shot generation 的自然 formulation**：就像 GPT 用 next-token prediction 统一了各种 NLP task，next-shot prediction 统一了 multi-shot generation 的各种需求（interactivity, long generation, consistency）。

2. **Dual-cache 对应了人类叙事的两种记忆**：global cache 像 long-term memory（记住故事大背景），local cache 像 working memory（记住当前 scene 的细节）。RoPE discontinuity 就像是"场景切换"的 cognitive marker。

3. **Two-stage distillation 是经典 curriculum**：先学简单条件下的能力，再学复杂条件下的能力。这比一步到位的训练方式 stable 得多。

### 潜在的 limitations

1. **Backbone 容量有限**：paper 自己承认，复杂场景下会出现 visual artifacts。Wan2.1-1.3B 相对较小，scale up 到更大模型可能显著改善。

2. **Sparse attention 的集成空间**：paper 提到可以集成 sparse attention 和 attention sink 进一步加速。LongLive 的 KV-recache 机制可能也能借鉴。

3. **Context budget 的动态调整**：固定 6 frames 的 context budget 在 shot 数量很多时可能不够。可以考虑 hierarchical memory 或 retrieval-augmented context。

4. **Caption 的质量依赖**：multi-caption 机制依赖高质量的 shot-level caption，这在实际应用中可能是 bottleneck。

### 与 LLM 的类比

ShotStream 的架构与 LLM 的 autoregressive generation 高度相似：
- Next-shot prediction ≈ Next-token prediction
- Global cache ≈ KV cache 中的 long-range context
- Local cache ≈ Sliding window attention
- RoPE discontinuity ≈ Special tokens（如 `<eos>`）
- Two-stage distillation ≈ SFT → RLHF 的 curriculum

这种类比让我觉得，**video generation 正在重走 LLM 的路**，从 bidirectional（BERT-style）到 causal（GPT-style），从 batch generation 到 streaming generation。ShotStream 是这条路上的一个重要 milestone。

---

## 8. Web Links

- Paper project page: https://luo0207.github.io/ShotStream/
- Wan2.1 base model: https://arxiv.org/abs/2503.20314
- DMD2: https://arxiv.org/abs/2305.17439
- CausVid: https://arxiv.org/abs/2412.07772 (From Slow Bidirectional to Fast Causal Video Generators)
- Self Forcing: https://arxiv.org/abs/2506.08009
- LongLive: https://arxiv.org/abs/2509.22622
- Rolling Forcing: https://arxiv.org/abs/2509.25161
- HoloCine: https://arxiv.org/abs/2509.22622 (related)
- VBench: https://vchampion.github.io/VBench-project/
- TransNet V2 (shot boundary detection): https://github.com/SoC-research-group/transnetv2

---

## 9. 总结

ShotStream 的核心 contribution 是把 multi-shot video generation 从 bidirectional 转向 causal，同时解决了两个关键问题：inter/intra-shot consistency（通过 dual-cache + RoPE discontinuity）和 error accumulation（通过 two-stage distillation）。最终实现了 16 FPS 的实时交互式多镜头叙事视频生成。

这个工作让我看到 video generation 领域的一个清晰趋势：**causal autoregressive + distillation + self-forcing** 正在成为主流范式，就像 LLM 中的 decoder-only architecture 一样。未来的工作可能会在这个框架上继续 push——更大的 backbone、更高效的 attention、更智能的 context management、更好的 error recovery 机制。
