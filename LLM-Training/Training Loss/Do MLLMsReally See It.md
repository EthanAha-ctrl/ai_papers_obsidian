---
source_pdf: Do MLLMsReally See It.pdf
paper_sha256: b9316ed4d68f865ca9188fd02b62c6df8bc3cae6cb22c637a5edaca1b5ad0fbb
processed_at: '2026-08-03T22:55:50-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SAYO 这篇 paper

## 一句话总结

**现在的多模态大模型其实挺聪明的，推理能力也够用，但它"看图"的时候经常看错地方。SAYO 的做法就是：在 RL 训练里，当模型对自己生成的某个 token 不确定（entropy 高）的时候，就逼它去"看一眼"正确的图片区域，而不是靠语言模型瞎猜。**

就这么简单。

---

## 问题出在哪？先看一个生活中的类比

想象你让一个视力不好但脑子很好的人去解一道几何题。题目画了个三角形，标了几个边长。这个人脑子很聪明，数学公式都会，但他近视眼，看错了边长——把 5 看成了 8，然后一顿猛算，答案错了。

你怪他数学不好吗？不是。他数学很好。问题是**输入信号就错了**——garbage in, garbage out。

现在 MLLM 就是这个状态。Qwen3-VL、InternVL3.5 这些模型，language reasoning 能力早就够强了（海量文本预训练带来的），但在视觉任务上还是经常翻车。原因不是推理不行，是**视觉注意力放错了地方**。

更糟的是，CoT（chain-of-thought）一旦开始走错，就回不来了。paper 里 Figure 1 展示得很清楚：早期 attention 到了错误的 region，后面整个 reasoning chain 都建立在错误的 visual premise 上，model 不会自我纠正。

这叫 **error propagation**——一步错，步步错。

---

## 怎么证明"看对地方"很重要？

作者设计了一个指标叫 **Target Attention Score (TAS)**，其实就是算：model 生成 token 的时候，attention weight 有多少落在了"正确区域"上，多少落在了"整张图"上，然后取个比值。

公式长这样：

$$R_a = \frac{1}{2}\left(1 + \tanh\left(\log\frac{a + \varepsilon}{v + \varepsilon}\right)\right)$$

人话翻译：
- $a$：model 对"目标区域"（bounding box 对应的 image tokens）的平均 attention
- $v$：model 对"整张图"所有 visual tokens 的平均 attention
- $a/v$ 这个比值越大，说明 model 越聚焦于正确区域
- $\tanh + \log$ 就是把这个比值平滑映射到 $(0, 1)$ 区间，方便当 reward 用
- $\varepsilon$ 防止除零，很小的一个数

然后作者拿一堆模型在 GQA 数据集上跑，结果见 Figure 2：

**TAS 和 accuracy 强正相关。** TAS 越高的模型，答题越准。

但关键是——**所有现有模型的 TAS 都很低**。包括那些用了 RL 训练的 reasoning model。说明现有 RL 方法优化了文本推理链，但完全没碰视觉注意力这个维度。

这就是 paper 的核心诊断：**visual attention misalignment 是一个 credit assignment failure**。现有 training objective 没有给"视觉注意力"这个维度提供任何学习信号。

---

## SAYO 怎么解决？核心 idea

SAYO 用 GRPO 做强化学习训练，reward 有两部分：

$$r_o = r_v + r_f$$

- $r_f$：format reward，检查输出格式对不对（有没有 `<answer>` 标签之类的）
- $r_v$：**visual attention reward**，这是 SAYO 的核心创新

$r_v$ 怎么算？先看公式：

$$r_v = \tanh\left(\log\frac{a_q + \varepsilon}{v_q + \varepsilon}\right)$$

和前面的 $R_a$ 结构一样，但有个关键区别：**只对 high-entropy tokens 算**。

$a_q$ 和 $v_q$ 的计算：

$$a_q = \frac{1}{|\mathcal{Q}_{\mathrm{high}}|} \sum_{t_g \in \mathcal{Q}_{\mathrm{high}}} \frac{1}{H} \sum_{h=1}^{H} \frac{1}{|\mathcal{T}_{\mathrm{target}}|} \sum_{t_i \in \mathcal{T}_{\mathrm{target}}} \alpha_{t_i, t_g}^{(h)}$$

人话拆解：
- $\mathcal{Q}_{\mathrm{high}}$：所有生成 token 中 entropy 排名 top 30% 的那些 token（"高信息量 token"）
- $|\mathcal{Q}_{\mathrm{high}}|$：这些 token 的数量
- $H$：attention head 的总数
- $\mathcal{T}_{\mathrm{target}}$：目标区域对应的 image token 集合
- $\alpha_{t_i, t_g}^{(h)}$：生成 token $t_g$ 对 image token $t_i$ 在第 $h$ 个 head 上的 attention weight
- 整个公式就是：对高 entropy token，算它们对目标区域的平均 attention

$v_q$ 同理，只是把 $\mathcal{T}_{\mathrm{target}}$ 换成 $\mathcal{T}_{\mathrm{all}}$（所有 image tokens）。

---

## 为什么只选 high-entropy tokens？这是最 brilliant 的设计

这个设计背后的 intuition 特别深。

**什么是 high-entropy token？** 就是 model 在生成这个 token 的时候，概率分布很平坦，很不确定，"不知道该输出什么"。

**为什么不确定？** 两种可能：
1. 这个 token 确实需要看图才能确定（比如"图中有几个人？"→ 数字 token）
2. 纯语言层面的不确定（很少见）

大多数情况下，high entropy = **model 在这个位置需要 visual evidence 来做决策，但它没有去看**。

Standard next-token prediction 的目标是：

$$\min -\log p(t_k | v, t_{<k})$$

这个目标有个致命漏洞：**model 可以完全忽略 $v$（图片），只靠 $t_{<k}$（前面的文本上下文）来"猜"出 $t_k$**。因为 language model 有强大的统计先验，它知道"图中有___个人"这个位置大概率填个位数数字。

这就是 **hallucination 的根源**——model 靠语言先验蒙混过关，没有真正"看"图。

SAYO 的做法：**在 model 最不确定的地方（high entropy），强迫它的 attention 落在正确区域上**。如果你不确定，你就得去看，不准猜。

作者管这个叫 **"Look-to-Verify" policy**——先看再确认。

---

## 数据怎么构造？

很简单，两步（见 Figure 3）：

1. 从 question-answer pair 里提取 target object（比如问"红色的车在哪"，target 就是"红色的车"）
2. 用现成的 segmentation 工具找到这个 object 的 bounding box，然后根据 model 的 image tokenization 方式，把 bounding box 转换成对应的 visual token range

训练数据：
- GQA（真实场景，dense objects）：~16k
- ReFocus（结构化图表）：~4k

**总共才 20k 数据**。就这么点数据，效果就很显著。说明 attention reward 的 sample efficiency 很高——你只需要告诉 model "看这里"就够了，不需要教它怎么推理。

---

## 实验结果：几个让人惊讶的点

### 结果一：全面提升（Table 1）

以 Qwen3-VL-8B 为 base：

| Benchmark | Baseline | SAYO | 提升 |
|-----------|----------|------|------|
| MMERealWorld | 56.23 | 62.85 | +6.62 |
| M3CoT | 64.71 | 68.46 | +3.75 |
| V* | 81.15 | 82.20 | +1.05 |
| MMStar | 62.60 | 65.27 | +2.67 |
| MathVision | 22.20 | 25.26 | +3.06 |
| We-Math | 52.64 | 64.83 | **+12.19** |
| ChartQA | 78.96 | 81.84 | +2.88 |
| AI2D | 75.55 | 83.06 | **+7.51** |
| CharXiv | 42.70 | 42.50 | -0.20 |

平均从 59.64 涨到 64.03。

而且 SAYO-Qwen-8B 在 MMStar 上**超过了 GPT-4o**（65.27 vs 64.70）。

### 结果二：数学能力提升，但训练时没用数学数据

这个最 counter-intuitive。训练数据只有 GQA（场景理解）和 ReFocus（图表），但 We-Math 涨了 12 分，MathVision 涨了 3 分。

**为什么？** 因为数学推理能力和视觉感知能力是 **decoupled** 的。

Qwen3-VL 在文本预训练阶段早就学会了数学推理。但在视觉数学题上，它的瓶颈是：看错了三角形的边、读错了图表的坐标轴。一旦 visual attention 修正了，pre-existing 的数学引擎就能正常运转。

类比：一个数学很好的学生，以前考试时总是看错题目条件。现在给他配了副眼镜，成绩就上去了。不是他数学变强了，是他终于能看对题了。

### 结果三：Attention reward 比 accuracy reward 更有效（Table 2）

这个 ablation 是全文最重要的实验：

| 训练方式 | Avg |
|---------|-----|
| Baseline (Qwen3-VL-8B) | 59.64 |
| 只用 attention reward | 63.96 |
| 只用 accuracy reward | 60.92 |
| attention + accuracy | 64.27 |

**Attention-only 和 combined 效果差不多，accuracy-only 只有微弱提升。**

这说明什么？**模型不缺推理能力，缺的是视觉感知能力。** 你给它 accuracy reward，它知道"答案错了"但不知道"哪里看错了"。你给它 attention reward，它直接学会"该看哪里"，推理能力自然就释放出来了。

### 结果四：只奖励 high-entropy token 比奖励所有 token 好（Table 3）

| 策略 | Avg |
|------|-----|
| 只奖励 top 30% high-entropy token | 63.96 |
| 奖励所有 token | 60.51 |

差了 3.45 分。原因：很多 token 是功能性的（"the"、"is"、"a"），它们不需要看图。把这些 token 也纳入 reward 计算，就是往训练信号里掺噪声。

### 结果五：entropy 范围有 sweet spot（Table 6）

| 范围 | We-Math |
|------|---------|
| Top 20% | 59.08 |
| Top 30% | 64.83 |
| Top 40% | 66.84 |

太窄（20%）会漏掉重要 token，太宽（40%）会引入噪声。30% 是个平衡点。（We-Math 上 40% 反而更高，但其他 benchmark 上 30% 更稳定。）

---

## 训练细节

- Base model：Qwen3-VL-4B/8B、InternVL3.5-8B
- 算法：GRPO
- 硬件：6 × NVIDIA H200
- Epochs：4
- Learning rate：5e-6
- KL divergence coefficient：1e-3
- Rollout：16（每个 prompt 采样 16 条 response）
- Rollout temperature：1.0
- Max response length：1100 tokens
- Framework：TRL (HuggingFace)

训练 prompt 里有一句关键的话：

> "When reasoning, focus on the areas of aim object in the image."

这相当于在 prompt 层面就提醒 model 要看目标区域，和 reward 信号形成呼应。

---

## 和其他方法的区别

| 方法 | 怎么做 | 问题 |
|------|--------|------|
| Visual Prompt Engineering (ViP, BLINK) | 用 bounding box / 高亮来标出区域 | 依赖 model 本身的 attention 能力，那个能力本来就不行 |
| Look-Back | 在 CoT 里插入"回头看图"的标签 | 需要 lengthy textual reasoning |
| Reflection-V | reward 整个 thought chain 对图片的 attention | 没有区分 high/low entropy token，信号有噪声 |
| Standard GRPO + accuracy reward | 只看答案对不对 | credit assignment 失败，不知道哪步看错了 |
| **SAYO** | **只在高 entropy token 上 reward 区域级 attention** | **精准、低噪声、可迁移** |

---

## 我觉得这篇 paper 最 deep 的 insight

### 1. 重新定义了问题

以前大家觉得 MLLM 视觉推理不行，要么怪 reasoning 不够强（→ 加更长的 CoT），要么怪 vision encoder 不够好（→ 换更大的 ViT）。

SAYO 说：都不是。问题在 **attention 的 credit assignment** 上。model 的 reasoning engine 和 perception 都够用，但它们之间的 **interface**（attention）没有被正确训练。

### 2. Entropy 作为"需要看图"的信号

这是一个非常 principled 的设计。Entropy 高 = model 不确定 = 需要外部信息 = 应该去看图。这建立了一个从 model 内部状态到外部行为（看图）的自然映射。

而且这个 insight 是 **domain-agnostic** 的。不管你做几何题、读图表、还是数物体，"不确定就去看"这个 policy 都是通用的。这就是为什么训练数据只有场景理解和图表，但数学能力也涨了。

### 3. "Look-to-Verify" 是 meta-skill

这个概念很有意思。model 学到的不是"看特定的 pixel pattern"，而是一个 **meta-level 的 attention policy**：遇到不确定性，consult visual evidence。

这和人类的认知行为很像——你不确定的时候会回头去看一眼题目。这个 skill 一旦学会，可以迁移到任何视觉任务上。

### 4. Attention reward > Accuracy reward

这个发现对整个 RL + MLLM 领域都有启发意义。大家一直在用 accuracy reward 做 RL，但 accuracy reward 的问题是 **sparse**——只有最后才知道对错。Attention reward 是 **dense** 的——每个 high-entropy token 都有 feedback。

而且 attention reward 更接近 **root cause**。答案错是因为看错了，那直接奖励"看对"比奖励"答对"更高效。

---

## 局限性（paper 没明说但我觉得存在的）

1. **需要 bounding box annotation**。虽然 GQA 和 ReFocus 有现成的，但扩展到新领域需要标注成本。未来可以探索用 model 自己的 attention 来 pseudo-label target region。

2. **假设单一 target region**。有些问题需要看多个区域（比如"图中的红车和蓝车哪个更快？"），当前 reward 只能 reward 一个 region。

3. **只用了 last layer 的 attention**。中间层的 attention 可能也有信息，但 paper 没探索。参考 [SparseMM](https://arxiv.org/abs/2506.05344) 的研究发现 MLLM 的 attention head 有 sparse pattern，不同 head 负责不同 visual concept，也许可以更精细地设计 per-head reward。

4. **只在 static image 上做了**。Video MLLM 有 temporal dimension，attention reward 需要扩展到时空维度。

5. **Figure 12 的 case study 其实暴露了一个问题**。SAYO 在那个例子里答错了（选了 A "4FT12"，正确答案是 D "4FT11"）。虽然 attention 更好了，但在 OCR 这种需要 fine-grained pixel-level perception 的任务上，attention reward 可能不够——你看对了区域，但还是读错了字。这说明 attention 是必要条件，但不是充分条件。

---

## 对未来的启发

这篇 paper 让我想到几个方向：

1. **Attention reward 可以用到 LLM 的 tool use 上**。当 LLM 不确定时，reward 它去"查工具"（搜索引擎、calculator），和 reward 它去"看图"是一回事——都是在 high-entropy 状态下 consult external evidence。

2. **可以和 process reward model 结合**。Attention reward 是一种 process-level signal，可以和其他 process reward（比如 step-level correctness）组合。

3. **Self-supervised attention target**。与其用 bounding box，不如让 model 自己生成"我觉得重要的区域"，然后 reward consistency。这样就不需要外部标注了。

4. **Attention reward 用于 hallucination 抑制**。hallucination 的本质就是 model 在不确定时靠语言先验瞎编。Attention reward 直接惩罚这个行为，可能是一个通用的 anti-hallucination 方法。

---

## 参考

- [SAYO Paper (arXiv:2509.12132)](https://arxiv.org/abs/2509.12132)
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [InternVL3.5](https://arxiv.org/abs/2508.18265)
- [GQA Dataset](https://arxiv.org/abs/1902.09506)
- [ReFocus](https://arxiv.org/abs/2501.05452)
- [Look-Back](https://arxiv.org/abs/2507.03019)
- [Reflection-V](https://arxiv.org/abs/2509.12132)
- [High-Entropy Minority Tokens](https://arxiv.org/abs/2506.01939)
- [Eyes Wide Shut? - Visual Shortcomings of MLLMs](https://arxiv.org/abs/2401.06209)
- [Seeing but Not Believing](https://arxiv.org/abs/2510.17771)
- [SparseMM - Head Sparsity in MLLMs](https://arxiv.org/abs/2506.05344)
- [TRL Framework](https://github.com/huggingface/trl)
- [BLINK - Multimodal LLMs Can See but Not Perceive](https://arxiv.org/abs/2404.12390)
- [Visual SketchPad](https://arxiv.org/abs/2411.17646)
- [V* - Guided Visual Search](https://arxiv.org/abs/2312.14135)
- [MMStar Benchmark](https://arxiv.org/abs/2403.20330)
- [We-Math](https://arxiv.org/abs/2407.01584)
- [MathVision](https://arxiv.org/abs/2402.14804)
- [CharXiv](https://arxiv.org/abs/2406.18521)
- [MME-RealWorld](https://arxiv.org/abs/2408.13257)

---

# SAYO: Reinforcing Visual Attention in MLLMs - 深度解析

## 1. 核心Intuition: 问题是什么?

这篇paper的核心洞察非常elegant. 作者发现了一个fundamental的问题: **MLLMs在long-chain reasoning过程中, 一旦early-stage出现visual attention misalignment, 这个error会propagate through整个chain of thought, 而model几乎无法self-correct**.

这本质上是一个**credit assignment failure**. 现有的training objectives (比如标准的next-token prediction或accuracy-based RL rewards) 没有提供有效的signal来学习reliable visual attention behaviors. Model可以develop strong abstract reasoning capabilities, 但这些capabilities没有reliably grounded在correct visual evidence上.

Figure 1展示了一个典型的failure case: model在推理早期就attend到了错误的visual region, 然后整个CoT trajectory都建立在错误的visual premise上, 最终导致systematic inference failure.

**关键诊断指标 - Target Attention Score (TAS)**:

作者引入了一个quantitative metric来measure这个问题. 从final transformer layer提取attention weights (因为这里multimodal fusion已经fully realized):

$$a = \frac{1}{H} \sum_{h=1}^{H} \frac{1}{|\mathcal{T}_{\mathrm{target}}|} \sum_{t_i \in \mathcal{T}_{\mathrm{target}}} \alpha_{t_i, t_g}^{(h)}$$

变量解释:
- $H$: total number of attention heads
- $\mathcal{T}_{\mathrm{target}}$: set of image tokens corresponding to the target region (通过bounding box转换得到)
- $\alpha_{t_i, t_g}^{(h)}$: attention weight from generated token $t_g$ (query) to image token $t_i$ (key) at attention head $h$
- $|\mathcal{T}_{\mathrm{target}}|$: number of image tokens in target region

类似地定义entire image的attention score $v$, 然后用normalized attention advantage score:

$$R_a = \frac{1}{2}\left(1 + \tanh\left(\log\frac{a + \varepsilon}{v + \varepsilon}\right)\right)$$

这里$\varepsilon$是small constant for numerical stability, $\tanh$和$\log$的组合将ratio映射到$(0, 1)$区间, 提供smooth gradient.

**Figure 2的关键发现**: Across multiple models (Qwen3-VL series, InternVL3.5 series等), TAS与accuracy有**strong positive correlation**, 但所有evaluated models的TAS都consistently low. 这说明现有RL techniques虽然improve了textual reasoning trajectories, 但fail to provide effective learning signals for precise visual focus.

## 2. 方法架构解析

### 2.1 整体Workflow (Figure 3)

SAYO的training pipeline包含两个stages:

**Stage 1: Data Construction with Visual Focus**
- 从question-answer pairs中extract target object text
- Match with image segmentation information → bounding box coordinates
- 根据model的image processing methodology, 将bounding boxes转换为visual token ranges
- 这些token ranges就是attention reward的target

**Stage 2: GRPO Training with Attention Reward**
- 使用GRPO算法
- Reward = format reward + visual attention reward
- 关键创新: entropy-selective attention reward

### 2.2 核心创新: Entropy-Based Target Attention Reward

这是这篇paper最brilliant的部分. 作者没有对所有tokens uniform地apply attention reward, 而是只对**high-entropy tokens** (top 30%) apply reward.

**Why high-entropy tokens?** 这里的reasoning非常deep:

对于一个high-entropy token $t_k \in \mathcal{Q}_{\mathrm{high}}$, model表现出high epistemic uncertainty. 这种uncertainty通常源于**insufficient grounding in visual context $v$**. 

Standard Next-Token Prediction最小化:
$$-\log p(t_k | v, t_{<k})$$

这个objective有一个critical flaw: 它允许model **bypass visual verification** by relying on linguistic priors (这就是hallucination的来源). Model可以通过language model的statistical regularity"猜"出答案, 而不需要真正"看"图.

SAYO的objective explicitly penalizes这种行为:

$$\mathcal{L}_{SAYO} = \mathbb{E}_{t \sim \pi}[r_v(a_t) \cdot \nabla \log \pi(t|s)]$$

变量解释:
- $\pi$: policy (即the model being trained)
- $s$: state (包含visual context $v$, question $q$, 和已经generated的tokens $t_{<k}$)
- $t$: token being generated
- $r_v(a_t)$: visual attention reward, acts as regularizer
- $\nabla \log \pi(t|s)$: policy gradient

通过enforcing high Attention Ratio $R_a$ specifically at high-entropy states, 我们impose一个**visual verification constraint**: 当model uncertain的时候, 它必须去"看"图来resolve uncertainty, 而不是rely on linguistic hallucination.

作者称这为**"Look-to-Verify" policy** - 这是一个domain-agnostic meta-skill.

### 2.3 Reward计算细节

对于high-entropy tokens集合 $\mathcal{Q}_{\mathrm{high}}$:

$$a_q = \frac{1}{|\mathcal{Q}_{\mathrm{high}}|} \sum_{t_g \in \mathcal{Q}_{\mathrm{high}}} \frac{1}{H} \sum_{h=1}^{H} \frac{1}{|\mathcal{T}_{\mathrm{target}}|} \sum_{t_i \in \mathcal{T}_{\mathrm{target}}} \alpha_{t_i, t_g}^{(h)}$$

$$v_q = \frac{1}{|\mathcal{Q}_{\mathrm{high}}|} \sum_{t_g \in \mathcal{Q}_{\mathrm{high}}} \frac{1}{H} \sum_{h=1}^{H} \frac{1}{|\mathcal{T}_{\mathrm{all}}|} \sum_{t_i \in \mathcal{T}_{\mathrm{all}}} \alpha_{t_i, t_g}^{(h)}$$

最终的visual attention reward:

$$r_v = \tanh\left(\log\frac{a_q + \varepsilon}{v_q + \varepsilon}\right)$$

- $r_v \in (-1, 1)$: 正值表示model allocates相对更多attention到target region
- Overall reward: $r_o = r_v + r_f$ (format reward)

## 3. 实验结果深度分析

### 3.1 Main Results (Table 1)

| Model | MMERealWorld | M3CoT | V* | MMStar | MathVision | We-Math | ChartQA | AI2D | CharXiv | Avg |
|-------|-------------|-------|-----|--------|------------|---------|---------|------|---------|-----|
| Qwen3-VL-8B | 56.23 | 64.71 | 81.15 | 62.60 | 22.20 | 52.64 | 78.96 | 75.55 | 42.70 | 59.64 |
| **SAYO-Qwen-8B** | **62.85** | **68.46** | **82.20** | **65.27** | **25.26** | **64.83** | **81.84** | **83.06** | **42.50** | **64.03** |
| GPT-4o | 73.06 | 74.20 | - | 64.70 | 30.40 | 69.00 | 75.32 | 84.60 | 48.90 | - |

关键观察:
1. SAYO在MMStar上outperforms GPT-4o (65.27 vs 64.70)
2. 在We-Math上有**+12.19**的巨大提升 (64.83 vs 52.64), 尽管training时**没有用mathematical datasets**
3. AI2D提升**+7.51** (83.06 vs 75.55)

### 3.2 Cross-domain Generalization的深层原因

这个counter-intuitive finding非常重要: 为什么在GQA (dense scenes)和ReFocus (structured documents)上训练, 能improve数学推理?

作者的explanation非常insightful: **visual parsing和logical reasoning是decoupled的**.

Current SOTA base models (比如Qwen-VL) 已经possess strong latent mathematical reasoning capabilities (来自massive textual pre-training). 但它们的performance被**visual misalignment** bottleneck了 - 比如attend到错误的geometric line, 或者misread chart axis.

SAYO学到了一个robust **structure-aware attention policy**. 通过correcting input signal (确保model "sees" correct triangle side), SAYO effectively eliminates the "garbage in" phase. 这允许model的pre-existing mathematical engine处理valid visual premises.

**Intuition**: 这就像给一个聪明的blind person配了一副合适的眼镜 - 他的大脑(reasoning engine)一直很强大, 只是之前看不清楚.

### 3.3 Ablation Study (Table 2) - 最key的实验

| Model | MMERealWorld | M3CoT | MMStar | We-Math | AI2D | Avg |
|-------|-------------|-------|--------|---------|------|-----|
| Qwen3-VL-8B (baseline) | 56.23 | 64.71 | 62.60 | 52.64 | 75.55 | 59.64 |
| w/ Attn. Reward only | 62.85 | 67.86 | 65.27 | 64.83 | 83.06 | 63.96 |
| w/ Acc. Reward only | 57.01 | 65.96 | 65.20 | 56.61 | 78.01 | 60.92 |
| Full (Attn + Acc) | 61.59 | 67.77 | 66.00 | 66.15 | 83.03 | 64.27 |

**这个结果非常striking**: Attention-only reward的效果与combined reward相当, 而accuracy-only reward只有marginal gains!

这说明: **current MLLMs的deficiencies主要来自insufficient visual perception和localization, 而不是limited reasoning capacity**. 一旦relevant information被correctly identified, model的reasoning capabilities就能unlock.

### 3.4 Entropy Selection Ablation (Table 3)

| Token Selection | MMERealWorld | M3CoT | We-Math | AI2D | Avg |
|----------------|-------------|-------|---------|------|-----|
| key tokens (top 30%) | 62.85 | 67.86 | 64.83 | 83.06 | 63.96 |
| all tokens | 56.59 | 66.01 | 58.39 | 77.43 | 60.51 |

Selectively rewarding high-information tokens比uniformly applying rewards across all tokens效果好很多. 原因: 很多tokens只是syntactic或connective roles, 不需要direct visual attention. 包含这些low-information tokens会introduce noise, weaken learning signal.

### 3.5 Entropy Range Ablation (Table 6)

| Range | MMERealWorld | M3CoT | MathVision | We-Math | AI2D | CharXiv |
|-------|-------------|-------|------------|---------|------|---------|
| Top 20% | 58.36 | 65.19 | 17.80 | 59.08 | 79.40 | 40.90 |
| **Top 30%** | **62.85** | **68.46** | **25.26** | **64.83** | **83.06** | **42.50** |
| Top 40% | 62.48 | 66.18 | 20.13 | 66.84 | 82.93 | 39.30 |

Top 30%是sweet spot: 太少会exclude important information tokens, 太多会introduce training noise.

## 4. Attention Behavior Analysis (Figure 4, 5, 6, 7)

### 4.1 Token-level Attention Patterns

Figure 4展示了两个key findings:

1. **Throughout generation sequence**: SAYO consistently exhibits significantly higher visual attention weights toward target region than baseline Qwen3-VL, 尤其在later stages of inference when generating answers.

2. **Entropy-attention relationship**: SAYO在大多数high-entropy tokens上maintains consistently high visual attention, 只在extremely low entropy tokens上有lower attention (这些tokens信息量minimal).

### 4.2 Case Study (Figure 5)

Figure 5的example非常illuminating: SAYO在reasoning的early stages就correctly identified和focused on target object, 然后在critical junctures of reasoning process持续maintain high visual attention weighting. 这种sustained和accurate visual attention能continuously guide reasoning process away from erroneous visual information.

## 5. Training Details

### 5.1 Hyperparameters (Table 4)

| Parameter | Value |
|-----------|-------|
| Epochs | 4 |
| Per Device Batch Size | 64 |
| Rollout | 16 |
| Rollout Temperature | 1.0 |
| Rollout Top-P | 0.9 |
| KL divergence coefficient | 1e-3 |
| Learning rate | 5e-6 |
| Weight Decay | 1e-2 |
| Max Grad Norm | 0.8 |
| Optimizer | AdamW |

### 5.2 Training Data

| Dataset | Size |
|---------|------|
| GQA (dense scenes) | ~16k |
| ReFocus_Data (structured documents) | ~4k |

仅仅20k数据就能achieve这样的improvement, 说明attention reward的sample efficiency非常高.

### 5.3 Training Dynamics (Figure 9)

Figure 9显示: 使用top 30% high-entropy tokens的reward values能normally increase during training, 而使用all tokens的reward由于noise signals而less stable. 这进一步验证了entropy-selective design的necessity.

## 6. 与Related Works的对比

### 6.1 vs. Visual Prompt Engineering (ViP, BLINK, ReFocus, ControlMLLM)

这些方法通过external tools (bounding boxes, visual highlighting)来influence visual perception. 但它们的effectiveness depends on model's pre-existing attention behavior, 而这个behavior本身remains insufficiently optimized. 而且这些methods难以transfer到new visual reasoning tasks.

### 6.2 vs. Look-Back, Reflection-V

- **Look-Back**: 通过incorporating look-back labels到long-term reasoning chains来enhance focus
- **Reflection-V**: 进一步introduces attention reward mechanism, rewards overall attention of thought chain text toward image

SAYO的关键区别: 使用**visual bounding boxes annotated data**来enhance visual attention capabilities, 而且model能maintain visual focus和reasoning on target objects**without requiring lengthy textual inference和reflection processes**.

### 6.3 vs. Standard GRPO with Accuracy Reward

Standard GRPO只optimizes final answer accuracy, 没有explicit signal for visual attention. SAYO的ablation证明: attention reward比accuracy reward更effective, 因为它directly addresses root cause (visual misalignment)而不是symptom (wrong answer).

## 7. 深层Intuition和Implications

### 7.1 Credit Assignment视角

从RL的角度看, 这篇paper解决了一个credit assignment problem. 在long CoT中, 最终答案的对错很难attributed到specific reasoning steps. 如果只reward最终accuracy, model不知道哪一步visual attention出了问题.

Attention reward提供了**dense, per-step supervision signal**: 每个high-entropy token都得到关于其visual attention quality的feedback. 这大大simplified了credit assignment.

### 7.2 "Look-to-Verify" as Meta-Skill

作者提出"Look-to-Verify"是一个**domain-agnostic meta-skill**. 一旦model学会ground its attention in complex natural scenes (dense objects)和charts (structured elements), 这种attention-sharpening capability自然transfer到其他visual domains (比如geometric diagrams).

这解释了cross-domain generalization: model不是学到了specific visual patterns, 而是学到了一个**general attention policy** - "当uncertain时, consult visual evidence".

### 7.3 Entropy作为Uncertainty Signal

使用entropy来select tokens是一个非常principled的设计. High entropy = high epistemic uncertainty = model在"猜" = 需要visual verification. 这建立了一个自然的connection between model的internal uncertainty state和external visual grounding需求.

### 7.4 Garbage In, Garbage Out的深刻含义

这篇paper的core message可以summarized为: **MLLMs的reasoning engine一直很强大, 但它们一直被fed garbage visual premises**. 通过fixing visual attention, 我们让pre-trained reasoning engine终于能process valid inputs.

这有一个重要implication: 也许我们不需要更大的reasoning models, 而是需要更好的perception grounding.

## 8. Limitations和Future Directions

1. **Data dependency**: 需要bounding box annotations, 这限制了scalability. 未来可以explore weakly-supervised或self-supervised approaches来identify target regions.

2. **Single target assumption**: 当前reward假设每个question有单一target region. 对于需要multi-region reasoning的complex tasks, 可能需要extend到multi-target attention reward.

3. **Computational overhead**: 计算attention weights需要access to model internals, 这在very large models上可能有scalability concerns.

4. **Generalization to video**: 当前只处理static images. Video MLLMs有temporal dimension的attention, 需要新的reward design.

## 9. Personal Reflections

作为AI researcher reading这篇paper, 我觉得有几个特别brilliant的点:

1. **Problem formulation**: 将visual attention misalignment formulate为credit assignment failure, 这是一个非常clean和principled的framing.

2. **Entropy-selective design**: 这个设计既principled (基于information theory) 又practical (reduces noise). 它建立了一个elegant bridge between model的internal uncertainty和external visual grounding.

3. **Minimal architectural change**: SAYO不需要modify architecture, 不需要inference-time visual prompts, 只需要training-time reward engineering. 这使得它非常practical to adopt.

4. **Cross-domain generalization**: 证明attention policy是transferable meta-skill, 这个insight对整个MLLM field都有implications.

这篇paper让我想到一个问题: 我们是否over-invested在reasoning capabilities上, 而under-invested在perception grounding上? 如果model的visual attention是reliable的, 也许我们不需要那么复杂的CoT reasoning - 简洁的reasoning + precise perception可能就够了.

References:
- [SAYO Paper (arXiv)](https://arxiv.org/abs/2509.12132) - 原paper
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300) - GRPO算法来源
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) - Base model
- [InternVL3.5](https://arxiv.org/abs/2508.18265) - 另一个base model
- [GQA Dataset](https://arxiv.org/abs/1902.09506) - 训练数据之一
- [ReFocus](https://arxiv.org/abs/2501.05452) - 训练数据来源和related work
- [Reflection-V / Look Again, Think Slowly](https://arxiv.org/abs/2509.12132) - Related work on visual reflection
- [Look-Back](https://arxiv.org/abs/2507.03019) - Related work on implicit visual re-focusing
- [High-Entropy Minority Tokens](https://arxiv.org/abs/2506.01939) - Entropy-based token selection的理论基础
- [TRL Framework](https://github.com/huggingface/trl) - 训练框架
- [Eyes Wide Shut?](https://arxiv.org/abs/2401.06209) - Visual shortcomings of MLLMs的早期分析
- [Seeing but Not Believing](https://arxiv.org/abs/2510.17771) - Attention和correctness的disconnect研究
