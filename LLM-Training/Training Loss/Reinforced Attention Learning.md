---
source_pdf: Reinforced Attention Learning.pdf
paper_sha256: b9c6799e698e45964eee1f68c7c35d263a6dc27fa040c73850c7d2203b5412ce
processed_at: '2026-08-11T22:07:13-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Reinforced Attention Learning 用人话讲

## 一句话总结

传统 RL 训练告诉模型"**你说错了**"，RAL 告诉模型"**你看错地方了**"。

## 为什么需要这个方法

想象你问一个多模态模型："图片里那个人手里拿的咖啡杯上写的什么字？"

模型答错了。传统 RL（GRPO）的做法是：
- 看答案 → 错了 → 降低这类答案的概率
- 看答案 → 对了 → 提高这类答案的概率

问题在于，模型答错可能根本不是因为不会说，而是因为它的注意力飘到了图片的其他区域，压根没看到杯子上的字。你反复通过答案对错来惩罚它，它学会了各种 hack 比如编一个看起来合理的答案，但内部看图的能力其实没真正改善。

这就解释了 paper 里一个很反直觉的发现：**GRPO 在某些感知任务上反而比 base model 更差**。因为 token-level RL 容易 reward hacking——模型学会了"说什么能拿分"，但内部视觉 grounding 反而被带偏了。

## RAL 的核心思路

既然瓶颈在"看哪里"，那就直接优化"看哪里"。

具体来说，Transformer 每生成一个 token 时，会对前面所有 token 分配 attention weights。RAL 把这个 attention 分布本身当成一个 policy 来做 RL：

- 如果最终答案对了（reward 高）→ 鼓励当前这种 attention 模式，让它更靠近这次的 attention 分布
- 如果最终答案错了（reward 低）→ 惩罚这种 attention 模式，让它远离这次的 attention 分布

用 paper 的公式说就是：

$$L_{\text{AttnRL}} = \mathbb{E}_t \left[ A_t \cdot D(p_\theta^t \parallel p_{\text{old}}^t) \right]$$

- $A_t$：advantage，就是这次回答比平均水平好还是差
- $D(\cdot)$：JSD divergence，衡量两个 attention 分布的差异
- $p_\theta^t$：当前模型的 attention 分布
- $p_{\text{old}}^t$：采样时 old policy 的 attention 分布

$A_t > 0$ 时，最小化这个 loss 就是让 attention 靠近成功的模式；$A_t < 0$ 时，最小化 loss 反而让 attention 远离失败的模式（因为 JSD 恒正，乘以负 advantage 后整个项变负，梯度方向翻转）。

## 和传统 RL 怎么结合

RAL 没有完全抛弃 token-level RL，而是在 GRPO 的基础上加了一个 attention 正则项：

$$L_{\text{total}} = L_{\text{RL}} + \lambda_{\text{attn}} \cdot L_{\text{AttnRL}}$$

$\lambda_{\text{attn}}$ 控制attention监督的强度。所以模型同时在学"说什么"和"看哪里"，两条信号互补。

## Attention 怎么提取的

实操细节很关键：
- 取 **最后一层** Transformer 的 attention weights
- **所有 head 平均**（不做 head-specific 的区分，简单粗暴但有效）
- 用 eager attention 实现（因为 flash attention 不暴露中间 attention weights）
- 对每个生成的 token $t$，把它对前面所有位置 $i < t$ 的 attention 归一化成一个分布 $p_\theta^t(i)$

这里有个设计选择值得注意：它用的是 **causal attention distribution**，即生成 token $t$ 时往前看所有位置的 attention。这意味着它不仅监督对 visual tokens 的 attention，也监督对已生成 reasoning tokens 的 attention——整个信息收集过程都被塑造。

## On-Policy Attention Distillation

除了 RL，paper 还把同样的思路用到 distillation 上。

传统 distillation 是让 student 模仿 teacher 的**输出概率分布**。Attention distillation 是让 student 模仿 teacher 的**内部 attention 分布**——也就是 teacher 在回答同一个问题时，目光落在哪里。

$$L_{\text{AttnDistill}} = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=P+1}^{T} \text{JSD}(p_\theta^t \| p_\phi^t) \right]$$

注意这里没有 advantage 项，就是纯粹的 attention 对齐。轨迹 $\tau$ 从 student 自己采样（on-policy），但在 student 自己的轨迹上算 teacher 的 attention，然后拉齐。

直觉上：teacher 是个 32B 的大模型，它看图的 patterns 更好。你不光让 student 学 teacher 说什么，还教它在每一步该看哪里。这比单纯模仿输出 logits 信息密度高得多。

## 最有意思的实验发现

### 1. RAL-zero：不需要 thinking process 也能涨点

这个 ablation 很 striking。把 thinking block 完全去掉，模型直接输出答案，只靠 attention policy gradient 训练：

| | NExTQA | VideoMME | LVBench |
|---|---|---|---|
| Base | 73.7 | 61.6 | 40.5 |
| GRPO | 70.7 | 62.0 | 43.9 |
| RAL-zero | **76.2** | **65.1** | **45.9** |

没有 chain-of-thought reasoning，没有 verbose rationale，纯靠优化 internal attention，居然在 temporal reasoning 任务上全面超越。这说明什么？

**Attention policy space 本身就蕴含了大量未被挖掘的推理能力。** 之前大家觉得 MLLM 需要 CoT 来做推理，但可能 CoT 只是一种间接手段，真正重要的是 attention 到了正确的位置。你直接优化 attention，绕过语言生成的中间步骤，反而更高效。

### 2. 分辨率越高，RAL 优势越大

在 V* benchmark（fine-grained visual search）上：

| Tokens/image | GRPO vs RAL 差距 |
|---|---|
| 512 | +1.6 |
| 1024 | ~+4 |
| 2048 | +6.3 |

视觉信息越密集、越细粒度，attention 选择就越关键，RAL 的优势就越明显。这完全符合直觉：低分辨率时反正信息就那些，attention 飘一飘影响不大；高分辨率时几十个 patch 里找对位置就变成 bottleneck。

### 3. GRPO 会 degrade base model，RAL 不会

看 image benchmarks：

| | V* | VizWiz | MuirBench |
|---|---|---|---|
| Base | 70.7 | 71.2 | 44.9 |
| GRPO | 68.6 ↓ | 67.9 ↓ | 43.9 ↓ |
| RAL | 73.3 ↑ | 71.7 ↑ | 47.4 ↑ |

GRPO 在 fine-grained perception 上会掉点——这是 pure token-level RL 的 known issue。RAL 不光不掉，还稳稳提升。核心原因就是 attention-level 监督提供了更 stable 的 gradient signal，避免了 token-level 的 reward hacking。

## 我的理解和评价

这篇 paper 做的事情本质上是：**把 RL 的优化对象从模型的输出端拉到了中间表示端。**

从更宏观的视角看，这是一个很自然的方向。LLM 时代大家习惯了只看 input/output 做优化（pretrain 预测 next token，RLHF 也预测 next token with reward），中间的 internal representations 基本是黑箱。但 multimodal 场景下，cross-modal alignment 的瓶颈恰恰在中间——attention 怎么从 visual encoder 的输出里 select relevant information。

几个我觉得值得深究的点：

**为什么用最后一层而不是中间层？** Paper 没做这个 ablation。直觉上最后一层离 output 最近，attention 最直接服务于 generation。但中间层的 attention 可能对应更抽象的信息 routing。Multi-layer attention supervision 可能是个 extension 方向。

**为什么所有 head 平均？** 不同 head 本来就学不同功能（有的 head 做 syntactic，有的做 coreference，有的做 visual grounding）。一棍子打平平均掉可能损失信息。Head-specific 的 advantage weighting 可能更精细。

**JSD 的选择。** Paper 说 JSD 是 symmetric bounded divergence 所以稳定，但没和 KL 做对比。KL 有 mode-seeking behavior（$p_\theta \to p_{\text{old}}$ 时会 collapse），JSD mode-covering 可能更适合探索。但这个 claim 需要 ablation 支撑。

**和 process reward model 的关系。** 最近 process supervision 很火（PRM800K 那一套），本质也是给中间步骤 credit。RAL 可以看作一种 implicit process supervision——不过它监督的是 attention pattern 而不是 reasoning step。两者能否结合是个有意思的方向。

**泛化到其他 internal structures。** Paper 最后提到可以优化 MoE routing 或 cross-modal fusion。这个 vision 很对。如果 attention 可以做 policy，那任何可微的 internal computation path 理论上都可以做 policy。这可能是 post-training 的下一个 paradigm。

## 参考

- 原文没有公开链接（这是 2025 年的 paper），但作者背景和引用可以追溯：
  - GRPO 原始 paper: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
  - On-policy distillation: [Agarwal et al. ICLR 2024](https://openreview.net/forum?id=f3A53PYkB6)
  - Attention transfer 经典: [Zagoruyko & Komodakis 2016](https://arxiv.org/abs/1612.03928)
  - Video-R1 dataset: [arxiv](https://arxiv.org/abs/2503.21776)
  - Qwen2.5-VL: [arxiv](https://arxiv.org/abs/2502.13923)

---

# Reinforced Attention Learning 深度解析

## 核心思想直觉

这篇paper的核心洞察非常深刻：传统的RL后训练优化的是 **"what to generate"**（下一个token的概率分布），而 RAL 提出优化 **"where to attend"**（内部注意力分布）。作者认为 MLLM 在感知任务上的瓶颈不在于语言生成能力，而在于 cross-modal 的信息选择机制——即 Transformer 的 attention weights 如何分配到 visual tokens 上。

这个想法的直觉来源可以这样理解：当一个 MLLM 回答 "图片左下角的红色杯子是什么品牌" 时，错误往往不是因为语言模型不会组织答案，而是因为 attention 没有正确地聚焦到包含品牌 logo 的那几个 visual patch tokens 上。GRPO 优化 token likelihood 是一种**间接监督**——它只能通过最终答案的对错来反推，而 RAL 想做**直接监督**——直接塑造内部的信息流。

## 方法论数学细节

### 1. Aggregated Causal Attention Distribution Policy

设完整序列 $S = (x_1, \dots, x_T)$，其中：
- $x_1, \dots, x_P$ 是 prompt（包含 visual tokens + question tokens）
- $x_{P+1}, \dots, x_T$ 是 generated response（包含 `
