---
source_pdf: STEP3-VL-10B Technical Report.pdf
paper_sha256: 5c5dd52f86f5d9f81753235b82a5d8a9a0f83c6be01b19abf49fe34a275f6c38
processed_at: '2026-08-12T11:05:53-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲STEP3-VL-10B

## 先说结论

StepFun这帮人做了一件挺疯狂的事：用10B参数的小模型，在MathVision、AIME2025这些hard benchmark上打赢了Gemini-2.5-Pro和235B的Qwen3-VL。这不是渐进提升，是打脸"参数规模决定一切"的叙事。

核心trick就两件事：**第一，预训练阶段把vision encoder和LLM绑在一起从零训练，建立真正的vision-language synergy；第二，后训练阶段砸1400轮RL，加上一个叫PaCoRe的parallel reasoning trick，让模型在inference时能并行探索多个hypothesis再综合**。

[Paper link](https://stepfun-ai.github.io/Step3-VL-10B) | [PaCoRe paper](https://arxiv.org/abs/2601.05593)

---

## 架构选择：为什么是PE-lang而不是DINOv3

这里有个很有意思的ablation。论文比较了Perception Encoder的语言对齐版本（PE-lang, 1.8B）和DINOv3（ViT-large-16, 300M）。

DINOv3在纯视觉任务上强，这是公认的事实。但放进多模态setting后收敛很慢，因为modality gap——vision features和language space之间有个gap需要bridge。

PE-lang预先用contrastive learning和LLM对齐过，features已经在language space附近。即使后续有trillion-scale的生成训练，这个初始alignment仍然是收敛效率的前提。

| Vision Encoder | OCRBench | MMVP | MMMU |
|---|---|---|---|
| DINOv3 | 57.60 | 28.00 | 46.56 |
| PE-lang | 70.10 (+12.50) | 32.00 (+4.00) | 47.67 (+1.11) |

OCRBench上的+12.5%特别说明问题——text-rich场景下语言对齐优势巨大。

intuition：vision encoder不只是"看图"，它要把视觉信息翻译成LLM能消化的语言。如果encoder输出已经在language manifold附近，decoder的工作量就小得多。

[Perception Encoder](https://arxiv.org/abs/2504.13181) | [DINOv3](https://arxiv.org/abs/2508.10104)

---

## 预训练：1.2T tokens的"先宽后窄"

总训练配置：
- 1.2T tokens
- 370K iterations
- batch size 8,192
- sequence length 4,096
- AdamW: $\beta_1=0.9, \beta_2=0.95$（$\beta_2$比默认0.999低，适应大scale训练的梯度噪声）

学习率分两阶段：

| Phase | Tokens | LR | 数据 |
|---|---|---|---|
| Phase 1 | 前900B | $5\times10^{-5} \to 1\times10^{-5}$ | broad mixture |
| Phase 2 | 后300B | $1\times10^{-5} \to 6\times10^{-6}$ | 高质量数据，cool-down |

Phase 2的cool-down阶段专门巩固OCR和grounding这类fine-grained感知能力。这个pattern在LLM训练中很经典——前期学broad representation，后期在高质量数据上做精细打磨。

数据构成里几个有意思的点：

1. **Mosaic augmentation**：4张图拼成1张。同时提升input resolution和spatial reasoning压力，逼模型学会跨region reasoning
2. **OCR数据80M document + 100M formulas**：从arXiv渲染LaTeX，explicitly处理references和hyperlinks防止visual-textual mismatch
3. **GUI数据23M**：包含2M trajectories，每个trajectory是12个atomic actions（CLICK/SLIDE/TYPE...）的序列。这和简单captioning不一样，trajectory modeling让模型理解"动作-状态"的sequential dependency

---

## RL：核心引擎

1400轮RL分三个阶段：

| Stage | Iterations | Max Seq | Prompts/iter | Rollouts/prompt | 目的 |
|---|---|---|---|---|---|
| RLVR | 600 | 24k | 512 | 16 | verifiable reasoning |
| RLHF | 300 | 32k | 512 | 8 | human preference |
| PaCoRe | 500 | 64k | 64 | 16 | parallel reasoning |

### PPO+GAE的数学

给定$(I, q)$，policy $\pi_\theta$生成轨迹$\tau = (s_0, a_0, \ldots, s_{T-1}, a_{T-1})$。

GAE优势函数：
$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

变量：
- $l$: 时间步偏移，从0到$T-t-1$
- $\gamma$: discount factor，这里设为1（无折扣）
- $\lambda$: GAE平滑参数，这里设为1
- $\delta_{t'} = r_{t'} + \gamma V_\varphi(s_{t'+1}) - V_\varphi(s_{t'})$: TD误差
- $r_{t'}$: step $t'$的reward
- $V_\varphi$: value function

$\gamma=\lambda=1$时GAE退化成Monte Carlo return——bias最低但variance最高。这是off-policy setting下的选择。

PPO clipped surrogate：
$$\mathcal{J}_{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(\rho_t\hat{A}_t, \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

- $\rho_t = \pi_\theta(a_t|s_t)/\pi_{\text{old}}(a_t|s_t)$: 新旧policy概率比
- $\varepsilon$: clipping threshold，**这里设为8**（标准PPO是0.1-0.2）

$\varepsilon=8$是个aggressive的选择，允许更大的policy ratio deviation。这是为了mitigate training-inference inconsistency——off-policy correction需要更大的更新步长。

actor LR $2\times10^{-6}$，critic LR $5\times10^{-6}$，每iteration分4个minibatch。encoder冻结，只更新decoder。

[PPO paper](https://arxiv.org/abs/1707.06347) | [GAE paper](https://arxiv.org/abs/1506.02438)

---

## Reward系统：bifurcated design

最有工程价值的部分。分两大类：

### Verifiable Rewards

**Perception Rewards**：grounding/pointing任务用IoU或Euclidean distance。关键设计是**distance-decay reward shaping**——reward随distance指数衰减，保证optimization landscape光滑，避免discrete jump导致RL训练崩溃。

**Model-Based Verification**：用GPT-OSS-120B当judge，而不是string matching。优势：
- Parse-invariant（对LaTeX formatting variation鲁棒）
- 识别semantic equivalence（数学等价表达式）
- **Penalize false positives**——错误推理得到正确答案也要扣分

这第三点特别重要。标准reward只看final answer对不对，GPT-OSS会看reasoning process。如果模型通过flawed logic碰巧得到正确答案，reward会被zero out。这迫使模型学真正的reasoning，而不是reward hacking。

### Non-Verifiable Rewards

**GenRM**：pairwise preference，但在打分前先做explicit reasoning judgment。先判断再打分，能辨别plausible responses之间的subtle差异。

**Behavioral Regularization**三组penalty：
1. Language consistency：penalize code-switching和language mismatch
2. Citation verification：检测fabricated references直接zero reward
3. Epistemic calibration：penalize overconfident claims，鼓励适当表达uncertainty

这三组penalty是orthogonal的guardrails，从不同角度压制reward hacking。

[GPT-OSS model card](https://arxiv.org/abs/2508.10925) | [Perception-R1](https://arxiv.org/abs/2504.07954)

---

## 最重要的发现：Length Diminishment

text-only RL的常识：reasoning path会越来越长。DeepSeek-R1、o1都观察到这个现象，CoT length和performance正相关。

STEP3-VL-10B观察到反直觉现象：**average rollout length先升后降，最终回到起点**。

论文解释为两种opposing scaling的cancellation：

### Reasoning Tasks的Sequential Scaling
STEM/puzzles这类任务，更长CoT = 更多inference compute = 更好performance。标准pattern。

### Perception Tasks的Length Diminishment
grounding/OCR这类deterministic任务，RL gain来自**entropy reduction**。

机制：RL optimization通过pruning redundant exploratory tokens系统性collapse搜索空间。概率质量集中到singular deterministic mode上。效果是把high-temperature Pass@N exploration转换成robust Pass@1 accuracy。

在这个regime下，**更短的rollout = 更高model confidence = sharpened perceptual focus**。这和reasoning任务的intuition完全相反。

### "Missing Trace" Hypothesis

论文提出一个深刻假设解释为什么perception不能sequential scale。

人类视觉认知依赖iterative, coarse-to-fine过程——"glance-and-focus"、"try-error-correct"。但这些internal mental states在训练语料中几乎never被explicitly verbalized。

你看过几篇paper写"我先瞥了一眼左上角，没找到目标，然后转向右下角，发现三个候选，我逐一验证..."？几乎没有。人类perceptual process是implicit的，训练数据里缺这些cognitive traces。

所以RL optimization landscape没有足够的signal来incentivize sequential perceptual reasoning。模型学到的是"直接给答案"，而不是"先探索再聚焦"。

[Entropy mechanism paper](https://arxiv.org/abs/2505.22617) | [Does RL really incentivize reasoning](https://arxiv.org/abs/2504.13837) | [Perception in Reflection](https://arxiv.org/abs/2504.07165)

---

## PaCoRe：用Parallel绕过Missing Trace

既然perception不能sequential scale，那就parallel scale。

### 核心idea

类比Faster R-CNN的Region Proposal Network：
1. **Proposers并行生成多个perceptual hypotheses**
2. **Controller做sequential cross-checking与synthesis**

16个SeRe rollout并行生成，然后serialize成context喂给模型做synthesis。

### Serialization模板

```jinja
You are given a problem and a list of reference responses.
Your job is to analyze these references and provide your own response.
Original Problem:
{{ original_prompt }}
Reference Responses:
{% for response in ref_responses %}
Reference {{ loop.index }}:
{{ response }}
{% endfor %}
Now, based on the original problem and reference responses above,
please provide your own comprehensive solution.
```

当message set为空时bypass这个template，原始problem直接传入。

### 训练数据的Synthesis Filtration

复用RLVR阶段的24 rollouts作为message cache pool。对some-accept prompts：
1. 模拟parallel reasoning：从pool采样16-24 messages作为synthesis context
2. 喂回模型重新生成
3. **严格保留在coordinated setting下仍是some-accept的实例**

关键：防止task trivialization。如果有了16个reference后任务变trivially solvable，reward signal就没了，模型学不到东西。必须保证即使有parallel context，任务仍有难度。

### Emergent Behavior

论文Table 7展示了一个case study，模型在PaCoRe模式下自发产生human-like verification行为：

```
The references have different x-coordinates (like 220, 228, 227...) 
but the common y is 185 or around there.
Let's check the problem's context. The references' labels mention...
Let's see reference 5 has 228, reference 6 227, reference 9 228...
To create a new point, we need to follow the logic.
Let's check the references' labels and their coordinates' variations.
```

模型在：聚合references → cross-validate → 识别pattern → derive final answer。这是multi-agent synthesis的primitive form。

### 性能增益

| Benchmark | SeRe | PaCoRe | Δ |
|---|---|---|---|
| MathVision | 70.81 | 75.95 | +5.14 |
| DynaMath | 56.39 | 61.48 | +5.09 |
| CountQA | 33.69 | 38.29 | +4.60 |
| All-Angles-Bench | 51.58 | 59.08 | +7.50 |
| SpatialViz-Bench | 45.51 | 52.03 | +6.52 |
| AIME2025 | 87.66 | 94.43 | +6.77 |
| HMMT25 | 78.18 | 92.14 | +13.96 |

增益最大的是两类任务：
- **Intensive reasoning**（MathVision, AIME, HMMT）
- **Exhaustive perception with high recall requirement**（counting, OCR, spatial understanding）

HMMT25上的+13.96%特别夸张——这接近从78%到92%的跃升，是质变级别的提升。

### 推理配置

- SeRe: ``包裹thoughts, max 65,536 tokens
- PaCoRe: 16个SeRe rollouts合成context, max 131,072 tokens
- temperature=1.0, top-p=1.0, top-k=0

[PaCoRe paper](https://arxiv.org/abs/2601.05593) | [Faster R-CNN](https://arxiv.org/abs/1506.01497)

---

## 对比大模型的结果

Table 3的核心数据：

| Benchmark | STEP3-VL PaCoRe 10B | GLM-4.6V 106B | Qwen3-VL 235B | Gemini-2.5 Pro | Seed-1.5-VL |
|---|---|---|---|---|---|
| MMMU | 80.11 | 75.20 | 78.70 | 83.89 | 79.11 |
| MathVision | **75.95** | 63.50 | 72.10 | 73.30 | 68.70 |
| AIME2025 | **94.43** | 71.88 | 83.59 | 65.68 | 64.06 |
| HMMT25 | **92.14** | 57.29 | 88.36 | 74.53 | 51.30 |

MathVision、AIME2025、HMMT25三个benchmark上，10B的STEP3-VL在PaCoRe模式下**全部超越Gemini-2.5-Pro和Seed-1.5-VL**。

这是test-time compute scaling的胜利。参数少20倍，但通过parallel coordinated reasoning把inference compute拉满，在hard reasoning任务上反超大模型。

[Gemini 2.5 Pro](https://arxiv.org/abs/2507.06261) | [Qwen3-VL](https://arxiv.org/abs/2511.21631) | [GLM-4.6V](https://arxiv.org/abs/2507.01006) | [Seed-1.5-VL](https://arxiv.org/abs/2505.07062)

---

## 几个ablation的教训

### Muon optimizer
Muon用Newton-Schulz iteration正则化weight topology，理论上适合大scale多模态数据的噪声。SimpleVQA上+6.48%。

但最终没用——Muon对AdamW预训练的权重敏感，需要prolonged warmup，反而限制整体效率。这是个important negative result：新optimizer和预训练权重initialization的compatibility是实际部署的关键constraint。

### Deepstack
Qwen3-VL用的depth-extension技术。能加速训练收敛，但optimization-level的改进**没有转化为downstream benchmark提升**。

这是个值得记住的教训：训练loss曲线好看不等于下游性能好。有时候加速的只是loss landscape的descent速度，模型最终学到的东西本质一样。

[Muon optimizer](https://kellerjordan.github.io/posts/muon/) | [Deepstack](https://arxiv.org/abs/2507.19427)

---

## 更深层的思考

### 为什么PaCoRe work

PaCoRe本质上是在inference时materialize了implicit visual cognitive process。

人类做perception任务是parallel探索+sequential verification：先扫一眼整个scene形成多个candidate hypotheses，然后逐一验证。这个process在训练数据里missing，所以RL学不到sequential版本。

PaCoRe的工程解法：在inference时强行把16个parallel hypotheses塞进context，让模型做synthesis。这相当于在test-time人为注入了missing cognitive traces。

### System 2 to System 1

论文最后提到future direction：用self-distillation把PaCoRe产生的parallel coordinated reasoning traces internalize到模型参数中。

Kahneman的System 1/System 2框架：
- System 2: slow, deliberate, effortful（PaCoRe模式）
- System 1: fast, automatic, intuitive（目标）

如果self-distillation成功，expensive的parallel search会变成instinctive的sequential response。这和o1/o3的"把test-time reasoning压缩回训练"是同一个philosophy。

### Physical World Model

论文最后画出宏大narrative：
1. 当前：semantic world model（text-based multi-agent synthesis）
2. 下一步：physical world model（massive video trajectories + sensorimotor actions）
3. Ultimate: physics as verifier（用simulation environment，reward由physical laws governed）

这指向embodied AI的真正pathway——从"看图说话"到"理解物理因果"。

[Thinking Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) | [Gemini Robotics](https://arxiv.org/abs/2503.20020)

---

## 给Karpathy的intuition summary

这篇paper对你最relevant的几点：

1. **Vision encoder的语言对齐比纯视觉能力更重要**。DINOv3在vision benchmark强，但进VL setting后被PE-lang吊打。pre-alignment是zero-cost的head start。

2. **Perception RL和Reasoning RL是两个不同的game**。Reasoning RL让CoT变长，perception RL让output变短（entropy reduction → deterministic mode collapse）。这两个scaling law方向相反，混在一个training run里会cancel out。

3. **Missing Trace Hypothesis指出了当前MLLM training data的根本缺陷**。人类的perceptual cognitive process是implicit的，never verbalized。这是为什么perception不能sequential scale。PaCoRe用parallel绕过了这个限制，但根本解法可能需要重新设计training data来capture这些traces。

4. **Test-time compute是新的scaling dimension**。10B + PaCoRe打赢235B，说明parameter efficiency和inference compute之间存在trade-off。在reasoning-heavy任务上，额外inference compute的ROI可能高于额外参数。

5. **Reward design比algorithm选择更关键**。PPO的$\varepsilon=8$这种aggressive配置都能work，前提是reward signal足够clean。Model-based verification + behavioral regularization的组合比单纯调algorithm hyperparameter重要得多。

这篇paper的真正贡献不只是"又一个SOTA model"，而是揭示了perception scaling和reasoning scaling的本质差异，以及如何用parallel reasoning绕过training data的cognitive trace deficiency。这些insight对下一代MLLM架构设计有直接指导价值。

---

# STEP3-VL-10B 技术报告深度解析

## 核心叙事：小模型的逆袭

这篇报告的核心论点非常清晰：**一个10B参数的模型，通过精心的架构选择、统一的全参数预训练，以及超过1000轮的RL后训练，可以在多模态感知与推理任务上挑战106B-235B的巨型模型，甚至Gemini-2.5-Pro和Seed-1.5-VL这样的闭源旗舰**。关键数字：MMMU 80.11%、MathVision 75.95%、AIME2025 94.43%。这些不是渐进提升，而是量级跨越。

项目的两条战略主线：
1. **Unified Pre-training**：1.2T tokens多模态语料上全参数不冻结训练，让Perception Encoder与Qwen3-8B decoder建立intrinsic vision-language synergy
2. **Scaled RL + PaCoRe**：从sequential reasoning scaling到parallel coordinated reasoning，把test-time compute当作一等公民

Reference: [StepFun STEP3-VL-10B Homepage](https://stepfun-ai.github.io/Step3-VL-10B), [arXiv PaCoRe paper](https://arxiv.org/abs/2601.05593)

---

## 1. 架构设计（Section 2.1）

### 1.1 组件选型

| 组件 | 选择 | 参数量 | 理由 |
|------|------|--------|------|
| Vision Encoder | Perception Encoder (PE-lang) | 1.8B | 预先与LLM对齐，收敛快 |
| LLM Decoder | Qwen3-8B | 8B | 文本基础强，对多模态可塑性好 |
| Projector | 2层stride-2卷积 | - | 16×空间下采样 |
| 位置编码 | 1D RoPE | - | 高级变体无显著增益 |

### 1.2 Multi-crop策略

采用Caron等人的multi-crop思路：
- **Global view**：728 × 728
- **Local crops**：多个504 × 504

关键技巧是用batch维度的并行性绕开变长packing（variable-length packing）的复杂性（Shah等人的FlashAttention-3思路）。在patch rows之间插入newline tokens来编码2D空间结构。

### 1.3 为什么选PE-lang而不是DINOv3

这是论文中最有意思的ablation之一（Table 4）。DINOv3在纯视觉任务上强，但在多模态setting下收敛慢，因为modality gap。PE-lang已经预先与LLM对齐，即使后续有trillion-scale生成训练，这种语言对齐仍然是VL建模效率的前提。

| Vision Encoder | OCRBench | MMVP | MMMU | ReMI |
|---|---|---|---|---|
| DINOv3 | 57.60 | 28.00 | 46.56 | 24.50 |
| PE-lang | 70.10 (+12.50) | 32.00 (+4.00) | 47.67 (+1.11) | 26.08 (+1.58) |

OCRBench上的+12.5%提升尤其说明问题——语言对齐的vision encoder在text-rich场景下优势巨大。

Reference: [Perception Encoder paper](https://arxiv.org/abs/2504.13181), [DINOv3 paper](https://arxiv.org/abs/2508.10104)

---

## 2. 预训练数据构建（Section 2.2）

### 2.1 数据规模与领域

总计1.2T tokens，覆盖七大领域：

1. **Knowledge**（interleaved + image-text pairs）
   - Interleaved: CommonCrawl + StepCrawl（国内互联网）+ 关键词搜索
   - Image-Text Pairs四类：开源（LAION/COYO/BLIP-CCS/Zero）、关键词检索、从interleaved提取、Mosaic augmentation
   - Mosaic把4张图拼成1张，提升resolution与position reasoning

2. **Education**：约15M samples
   - K-12: 数理化人文，含化学公式/结构图（CoSyn合成）
   - 大学STEM/医学/艺术/金融
   - 成人教育：驾照/CPA/法律

3. **OCR**：
   - Image→Text: 10M真实 + 30M合成（SynthDog）
   - Image→Code: Markdown/LaTeX/Matplotlib（10M开源 + 15M合成infographic），TikZ/Graphviz约5M
   - Document→Text: 80M全页文档（PaddleOCR/MinerU 2.0标注）
   - Document→Code: HTML/Markdown/LaTeX，含arXiv的4M表格和100M公式

4. **Grounding & Counting**：约400M samples
   - 来源：OpenImages/COCO/Merlin/PixMo + 自建文本段落检测
   - Counting从detection标注转换得到

5. **VQA**：约10M samples + 20M OCR-VQA

6. **GUI**：约23M samples（参照Step-GUI）
   - 700K UI captions
   - 1M Knowledge VQA
   - 2M trajectories（12个atomic actions: CLICK/SLIDE/TYPE...）
   - 19M grounding samples
   - 30M web pages OCR

### 2.2 关键数据工程细节

- **CLIP-based clustering**做concept-balanced resampling，缓解长尾概念不平衡
- **Mosaic augmentation**同时提升分辨率和position reasoning
- **CoSyn**（Yang et al. 2025c）合成化学公式和几何图
- **细粒度rendering rules**：不让LLM完全接管生成，而是跨多个render工具强制细粒度渲染规则

Reference: [CoSyn paper](https://arxiv.org/abs/2502.14846), [PixMo/Molmo paper](https://arxiv.org/abs/2409.17146), [MinerU paper](https://arxiv.org/abs/2409.18839)

---

## 3. 预训练Recipe（Section 2.3）

### 3.1 优化器配置

AdamW:
- $\beta_1 = 0.9$（一阶矩衰减率）
- $\beta_2 = 0.95$（二阶矩衰减率，比默认0.999低，适应大scale训练的梯度噪声）
- $\varepsilon = 10^{-8}$
- weight decay = 0.01

训练配置：
- 总tokens: 1.2T
- iterations: 370K
- global batch size: 8,192
- sequence length: 4,096

### 3.2 两阶段学习率调度

| 阶段 | Tokens | LR范围 | 目的 |
|------|--------|--------|------|
| Phase 1 | 前900B | $5\times10^{-5} \to 1\times10^{-5}$ | 广泛表示学习 |
| Phase 2 | 后300B | $1\times10^{-5} \to 6\times10^{-6}$ | 高质量数据mixture，cool-down巩固OCR/grounding |

这是经典的"先宽后窄"思路：前期学broad representation，后期anneal到高质量数据上做fine-grained感知。

### 3.3 Muon vs AdamW的Ablation（Table 5）

Muon（Keller 2024）是matrix-wise optimizer，用Newton-Schulz iteration正则化weight topology。理论上能解决大规模多模态数据的噪声与不平衡问题。

实验结果显示Muon在tail-knowledge任务上确实有显著优势：
- SimpleVQA: +6.48%（Muon 27.08 vs AdamW 20.60）

但**最终没有采用Muon**，原因是initialization mismatch。Muon对AdamW预训练的权重敏感，需要prolonged warmup期来稳定transition，反而限制了整体训练效率。

Reference: [Muon optimizer](https://kellerjordan.github.io/posts/muon/), [Muon is scalable for LLM training](https://arxiv.org/abs/2502.16982)

### 3.4 Deepstack Ablation（Table 6）

Deepstack（Meng et al. 2024）是Qwen3-VL采用的depth-extension技术。实验发现：Deepstack能加速训练收敛，但**这种optimization-level的改进没有转化为downstream benchmark的实质提升**。考虑到计算开销，最终排除。

| Technique | OmniSpatial | MMVP | OCRBench | MMMU |
|---|---|---|---|---|
| w/ DeepStack | 42.92 | 26.00 | 71.20 | 47.44 |
| w/o DeepStack | 43.57 (+0.65) | 31.33 (+5.33) | 69.30 (-1.90) | 47.78 (+0.34) |

这是一个重要的negative result：训练loss曲线好看不等于下游性能好。

Reference: [Deepstack paper](https://arxiv.org/abs/2507.19427)

---

## 4. 后训练：两阶段SFT（Section 3.1）

### 4.1 数据构造

- 从开源社区收集millions of prompts（Guha et al. 2025的OpenThoughts等）
- 覆盖数学/编码/科学/逻辑推理
- 视觉感知：grounding/OCR/复杂文档图表理解
- 用internal frontier model蒸馏高质量responses
- **"Two-pipe"过滤**：
  1. 规则过滤（消除infinite repetition等degenerate patterns）
  2. Benchmark decontamination（exact matching + N-gram matching, $N=64$）

### 4.2 两阶段SFT策略

| 阶段 | Text:MM比例 | Tokens | 目的 |
|------|-------------|--------|------|
| Stage 1: Text-Dominant | 9:1 | 190B | 建立逻辑与语言基础 |
| Stage 2: Multimodal Integration | 1:1 | 36B | 平衡文本推理与视觉智能 |

训练配置：
- global batch size: 32
- sequence length: 128k（支持long-context）
- Cosine LR scheduler, 200-step warmup
- LR peak: $1\times10^{-4}$, anneal到$1\times10^{-5}$
- Domain-specific sampling weights（不同domain不同epoch数）

这里9:1到1:1的ratio shift值得注意——先让模型在纯文本上打好reasoning底子，再逐步注入多模态。

Reference: [OpenThoughts](https://arxiv.org/abs/2506.04178)

---

## 5. 强化学习核心（Section 3.2）

### 5.1 PPO + GAE算法详解

给定多模态输入$(I, q)$（image $I$, textual prompt $q$），policy network $\pi_\theta$生成轨迹：
$$\tau = (s_0, a_0, \ldots, s_{T-1}, a_{T-1})$$

其中：
- $s_t$: 状态，包含$(I, q)$和step $t$之前生成的token序列
- $a_t$: 在step $t$采样的action（token）
- $T$: 轨迹长度

#### GAE优势估计

$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

变量解释：
- $\hat{A}_t$: 时刻$t$的优势函数估计
- $l$: 时间步偏移量（从0到$T-t-1$）
- $\gamma \in [0,1]$: discount factor（折扣因子），此处设为1（无折扣）
- $\lambda \in [0,1]$: GAE平滑参数，此处设为1（相当于Monte Carlo return）
- $\delta_{t'}$: TD误差

TD误差定义：
$$\delta_{t'} = r_{t'} + \gamma V_\varphi(s_{t'+1}) - V_\varphi(s_{t'})$$

- $r_{t'}$: step $t'$的reward
- $V_\varphi$: 由参数$\varphi$定义的value function
- 当$\gamma=\lambda=1$时，GAE退化为完整的Monte Carlo return，bias最低但variance最高

#### PPO Clipped Surrogate Objective

$$\mathcal{J}_{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(\rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

变量解释：
- $\theta$: policy网络参数
- $\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$: 重要性采样比率（新旧policy概率比）
- $\varepsilon$: clipping超参数，限制policy更新幅度
- $\hat{A}_t$: GAE优势估计

#### Value Function Loss

$$\mathcal{J}_{\text{value}}(\varphi) = \frac{1}{2}\mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}}\left[\sum_{t=0}^{T-1}(V_\varphi(s_t) - V_t^{\text{target}})^2\right]$$

- $\varphi$: value function参数
- $V_t^{\text{target}} = \hat{A}_t^{\text{GAE}(\gamma,\lambda)} + V_\varphi(s_t)$: target value

### 5.2 实际训练配置

- **Off-policy PPO变体**：$\gamma=1, \lambda=1$，省略标准importance sampling
- 每iteration分4个minibatches
- Actor LR: $2\times10^{-6}$
- Critic LR: $5\times10^{-6}$
- Truncated importance sampling ratio threshold $\varepsilon=8$（Yao et al. 2025）
- **1400 iterations**，只更新decoder，encoder冻结

$\varepsilon=8$这个阈值比标准PPO的0.1-0.2大得多，是为了mitigate training-inference inconsistency——允许更大的policy ratio deviation以容纳off-policy correction。

Reference: [PPO paper](https://arxiv.org/abs/1707.06347), [GAE paper](https://arxiv.org/abs/1506.02438), [Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290), [Open-Vision-Reasoner](https://arxiv.org/abs/2507.05255)

---

## 6. Reward系统设计（Section 3.2.2）

这是论文最精妙的部分——bifurcated reward framework。

### 6.1 Verifiable Rewards

#### Perception Rewards
对于pointing/grounding任务，用IoU或Euclidean distance对齐几何输出与ground truth。关键设计：**strict, distance-decay reward shaping**，保证optimization landscape明确，RL收敛robust。

#### Model-Based Verification
用GPT-OSS-120B作为answer verifier（而不是简单string matching或mathverify heuristics）。优势：
- **Parse-invariant evaluation**：对formatting variations鲁棒（如idiosyncratic LaTeX）
- **Semantic equivalence recognition**：识别数学等价表达式或重排的derivation steps
- **Process consistency enforcement**：penalize false positives（错误推理得到正确答案）

### 6.2 Non-Verifiable Rewards

#### Generative Reward Modeling (GenRM)
Pairwise preference framework，GenRM评估rollouts与teacher model responses。关键创新：**在deriving fine-grained scalar score之前先做explicit reasoning judgment**，能辨别plausible responses之间的subtle quality差异。

#### Behavioral Regularization
三组penalty terms作为guardrails：

1. **Language consistency penalties**： discourage code-switching和question-answer language mismatch
2. **Strict citation verification**：检测到fabricated references/links时reward归零，直接从源头打击hallucination
3. **Epistemic calibration penalties**：suppress unjustified certainty或overconfident claims，鼓励在ambiguous/underspecified settings中适当表达uncertainty

这套设计哲学很清晰：reward hacking不是靠单个reward解决，而是靠一组orthogonal的constraints构建稳定optimization landscape。

Reference: [GPT-OSS model card](https://arxiv.org/abs/2508.10925), [Perception-R1](https://arxiv.org/abs/2504.07954)

---

## 7. Sequential Reasoning Scaling（Section 3.2.3）

### 7.1 RLVR阶段

训练数据：Open-Vision-Reasoner（数学/几何/物理/科学推理/感知/识别/图表推理/拼图）+ Perception-R1（视觉grounding）+ 内部K-12教育资源。

**三维过滤pipeline**：

| 维度 | 方法 | 作用 |
|------|------|------|
| Axis 1: Checkability | GPT-OSS-120B做4次独立验证，只保留全一致样本 | 确保监督质量 |
| Axis 2: Visual relevance | 早期STEP3-VL-10B评估image-question语义相关性 | 过滤冗余/错位对 |
| Axis 3: Difficulty | 24 rollouts/prompt，保留some-accept样本（既非trivially solvable也非consistently failed） | 控制难度分布 |

配置：
- 600 iterations
- max sequence length: 24k
- 512 prompts/iteration × 16 rollouts/prompt

### 7.2 RLHF阶段

在RLVR基础上做human preference alignment：
- Prompts来自open-source arena datasets（Chiang et al. 2024; Chou et al. 2024）+ 内部instruction pools
- 显式过滤uncheckable queries（无deterministic ground-truth）
- 用internal strongest models生成reference responses
- 300 iterations
- max sequence length: 32k
- 512 prompts/iteration × 8 rollouts/prompt

Reference: [Chatbot Arena](https://arxiv.org/abs/2412.08687), [Perception-R1](https://arxiv.org/abs/2504.07954)

---

## 8. PaCoRe: Parallel Coordinated Reasoning（Section 3.2.4）

这是论文最具创新性的部分，也是test-time compute scaling的关键。

### 8.1 核心思想

PaCoRe的思想类似Faster R-CNN的Region Proposal Networks：
1. **Proposers并行生成多个perceptual hypotheses**
2. **Controller做sequential cross-checking与self-verification，综合出最终输出**

### 8.2 训练数据构造

复用RLVR阶段的24 rollouts作为message cache pool。对some-accept prompts做**Synthesis Filtration**：
1. 模拟parallel reasoning：从pool采样16-24 messages作为"synthesis context"喂回模型重新生成
2. 严格保留在coordinated setting下仍然是some-accept的实例

关键：防止task trivialization，维持有效reward信号，迫使模型做multi-perspective self-verification与cross-checking。

### 8.3 训练配置

- PPO严格on-policy
- 500 iterations
- max sequence length: 64k（容纳aggregated context）
- 64 prompts/iteration × 16 rollouts/instance

### 8.4 Serialization模板（Table 8）

```jinja
You are given a problem and a list of reference responses. 
Your job is to analyze these references and provide your own response.
Original Problem:
{{ original_prompt }}
Reference Responses:
{% for response in ref_responses %}
Reference {{ loop.index }}:
{{ response }}
{% endfor %}
Now, based on the original problem and reference responses above, 
please provide your own comprehensive solution.
```

当message set为空时（$M=\emptyset$），bypass该template，原始problem直接传入。

### 8.5 推理时配置

- Sequential Reasoning (SeRe): thoughts包裹在``中，max length 65,536 tokens
- PaCoRe: 16个SeRe rollouts合成context，max length 131,072 tokens
- temperature=1.0, top-p=1.0, top-k=0

### 8.6 PaCoRe的Emergent Behavior（Table 7）

一个感知任务的例子显示模型自发产生了人类式的verification行为：
```
The references have different x-coordinates (like 220, 228, 227...) 
but the common y is 185 or around there.
Let's check the problem's context. The references' labels mention...
Let's see reference 5 has 228, reference 6 227, reference 9 228...
To create a new point, we need to follow the logic. 
Let's check the references' labels and their coordinates' variations.
```

模型在做：聚合reference answers → cross-validate validity → 识别final answer的underlying patterns。

### 8.7 PaCoRe的性能增益

| Benchmark | SeRe | PaCoRe | Δ |
|---|---|---|---|
| MathVision | 70.81 | 75.95 | +5.14 |
| DynaMath | 56.39 | 61.48 | +5.09 |
| CountQA | 33.69 | 38.29 | +4.60 |
| OCRBench | 86.75 | 89.00 | +2.25 |
| All-Angles-Bench | 51.58 | 59.08 | +7.50 |
| SpatialViz-Bench | 45.51 | 52.03 | +6.52 |
| AIME2025 | 87.66 | 94.43 | +6.77 |
| HMMT25 | 78.18 | 92.14 | +13.96 |

增益在两类任务上最大：
- **Intensive reasoning**（MathVision, DynaMath, AIME）
- **Exhaustive perception**（特别是高recall率的counting/OCR/spatial understanding）

Reference: [PaCoRe paper](https://arxiv.org/abs/2601.05593), [Faster R-CNN](https://arxiv.org/abs/1506.01497)

---

## 9. RL Dynamics的核心发现（Section 5.2）

### 9.1 Length Diminishment现象

这是论文最反直觉也最重要的发现。Text-only RL中通常观察到"sequential scaling"——reasoning path逐渐变长。但STEP3-VL-10B中**average rollout length先升后降，最终回到起点**。

论文识别出两种opposing scaling properties的cancellation effect：

1. **Reasoning Tasks**（STEM, Puzzles）：标准sequential scaling，性能与CoT长度正相关
2. **Deterministic Perception Tasks**（Grounding, OCR）：**length diminishment via policy refinement**

### 9.2 Length Diminishment的机制

Perception RL的增益来自**entropy reduction**（Cui et al. 2025b）。RL optimization通过pruning redundant exploratory tokens系统性collapse搜索空间，将概率质量集中在singular deterministic mode上。

效果：把high-temperature Pass@N exploration转换为robust Pass@1 accuracy（Yue et al. 2025）。在这个regime下，**更短的rollout length直接proxy更高的model confidence和sharpened perceptual focus**。

### 9.3 "Missing Trace" Hypothesis

论文提出一个深刻假设：perceptual "unscaling"源于critical data deficiency。

人类视觉认知依赖iterative, coarse-to-fine过程（如"glance-and-focus"或"try-error-correct"，类似o3或RePer的机制），但这些internal mental states在训练语料中很少被explicitly verbalized。因此RL optimization landscape缺乏必要的"cognitive traces"来自发incentivize sequential perceptual reasoning。

### 9.4 PaCoRe作为解法

PaCoRe通过test-time scaling策略explicitly explicate这些implicit visual processes。两个distinctive scaling properties：

1. **Response length steady, deliberate growth**：表明模型能有效allocate additional compute做hypothesis verification
2. **显著性能增益**：在reasoning-heavy和perception-centric benchmarks上均超越vanilla SeRe

### 9.5 Compress System 2 to System 1

PaCoRe作为primitive multi-agent framework：proposers并行生成massive visual proposals，controller做sequential cross-checking。未来方向：用self-distillation将parallel coordinated reasoning traces internalize到模型参数中，把expensive "slow-thinking"（Kahneman 2011）转化为high-fidelity intrinsic intuition。

Reference: [Entropy mechanism of RL for reasoning LLMs](https://arxiv.org/abs/2505.22617), [Does RL really incentivize reasoning capacity](https://arxiv.org/abs/2504.13837), [Perception in Reflection](https://arxiv.org/abs/2504.07165), [Thinking Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)

---

## 10. 评估结果（Section 4）

### 10.1 多模态核心结果（Table 1, 与7B-10B对比）

| Domain | Benchmark | STEP3-VL-10B | GLM-4.6V Flash 9B | Qwen3-VL Thinking 8B | InternVL 3.5 8B | MiMo-VL RL-2508 7B |
|---|---|---|---|---|---|---|
| STEM | MMMU | **78.11** | 73.53 | 71.69 | 71.14 | 70.81 |
| STEM | MathVision | **83.97** | 78.50 | 76.78 | 79.86 | 63.37 |
| STEM | MathVista | **66.89** | 64.37 | 54.03 | 39.47 | 51.65 |
| STEM | HLE | **10.73** | 3.82 | 5.98 | 4.51 | 5.90 |
| Recognition | MMBench EN | **92.05** | 91.04 | 90.55 | 88.20 | 89.91 |
| Recognition | MMBench CN | **91.55** | 89.56 | 89.75 | 86.24 | 88.79 |
| OCR | OCRBench | **86.75** | 85.97 | 48.75 | 46.25 | 59.38 |
| GUI | ScreenSpot-V2 | 92.61 | 92.14 | 93.60* | 63.95 | 90.82 |
| GUI | OSWorld-G | **59.02** | 54.71 | 56.70* | 31.91 | 50.54 |

### 10.2 Text-Centric结果（Table 2）

| Domain | Benchmark | STEP3-VL-10B | GLM-4.6V Flash 9B | Qwen3-VL Thinking 8B | InternVL 3.5 8B | MiMo-VL RL-2508 7B |
|---|---|---|---|---|---|---|
| Math | AIME2024 | **90.94** | 37.92 | 74.06 | 78.18 | 75.36 |
| Math | AIME2025 | **87.66** | 33.02 | 62.92 | 62.50 | 66.51 |
| Math | HMMT25 | **78.18** | 19.17 | 45.21 | 35.78 | 47.34 |
| Math | IMO-AnswerBench | **62.12** | 22.62 | 38.69 | 35.00 | 48.44 |
| Code | LiveCodeBench | **75.77** | 22.17 | 51.05 | 45.90 | 39.65 |
| Subjective | Arena-Hard-V2 | **58.57** | 9.26 | 47.34 | 15.57 | 28.59 |

STEP3-VL-10B在文本任务上的表现甚至超过了专门的text-only模型——关键设计选择是**没有trade-off text与vision**。

### 10.3 与10×-20×大模型对比（Table 3）

| Benchmark | STEP3-VL SeRe 10B | STEP3-VL PaCoRe 10B | GLM-4.6V 106B-A12B | Qwen3-VL Thinking 235B-A22B | Gemini-2.5 Pro | Seed-1.5-VL |
|---|---|---|---|---|---|---|
| MMMU | 78.11 | **80.11** | 75.20 | 78.70 | 83.89 | 79.11 |
| MathVision | 70.81 | **75.95** | 63.50* | 72.10 | 73.30* | 68.70* |
| AIME2025 | 87.66 | **94.43** | 71.88 | 83.59 | 65.68 | 64.06 |
| HMMT25 | 78.18 | **92.14** | 57.29 | 88.36 | 74.53 | 51.30 |
| MMBench EN | 92.05 | 92.38 | 92.75 | 92.70 | **93.19** | 92.11 |

PaCoRe模式下的STEP3-VL-10B在MathVision、AIME2025、HMMT25上**超越了Gemini-2.5-Pro和Seed-1.5-VL**。这是小模型+test-time compute scaling对抗大模型的最强证据。

Reference: [Gemini 2.5 Pro](https://arxiv.org/abs/2507.06261), [Qwen3-VL](https://arxiv.org/abs/2511.21631), [GLM-4.6V](https://arxiv.org/abs/2507.01006), [Seed-1.5-VL](https://arxiv.org/abs/2505.07062)

---

## 11. 未来方向（Section 6）

### 11.1 Maximizing Token Efficiency via Universal RL Scaling

- **从pre-training向RL shift compute**：RL scaling展示continuous, saturation-free的性能跃升，pre-training alone无法维持
- **Optimizing Reasoning Density**：internalize parallel exploration benefits，消除redundant "over-thinking"，把explicit coordinated search压缩成efficient sequentiality，最终distill成"System 1"式instinctive responses

### 11.2 Bridging the Reality Gap

- **From Semantic to Physical World Models**：从text-based multi-agent synthesis扩展到massive video trajectories和sensorimotor action sequences，构建holistic world model
- **Physics as Ultimate Verifier**：用high-fidelity simulation environments，rewards严格由immutable physical laws governed，从surface-level imitation转向interaction-driven mastery
- **Embodied Chain-of-Thought (E-CoT)**：扩展reasoning context到explicitly model temporal dynamics和physical state transitions，训练模型通过预测dynamics prior to action来articulate "physical intuition"

---

## 12. 关键Insight总结

### 12.1 为什么10B能打106B+

1. **Vision Encoder的语言对齐**是前提（PE-lang vs DINOv3 ablation）
2. **1.2T tokens统一全参数预训练**建立intrinsic vision-language synergy
3. **1400轮RL**（600 RLVR + 300 RLHF + 500 PaCoRe）是性能跃升的主引擎
4. **PaCoRe把test-time compute当作scaling维度**，在小模型上实现大模型的reasoning depth
5. **Length diminishment的发现**指出perception RL的机制与reasoning RL根本不同

### 12.2 最值得深挖的技术细节

1. **GAE with $\gamma=\lambda=1$**：完全Monte Carlo return，bias最低，配合truncated importance sampling ratio threshold $\varepsilon=8$
2. **Model-based verification（GPT-OSS-120B）**比string matching在noisy ground truth下更robust
3. **Synthesis Filtration**：防止PaCoRe训练时task trivialization
4. **Behavioral Regularization的三组penalty**：language consistency + citation verification + epistemic calibration
5. **"Missing Trace" Hypothesis**：解释为什么perception不能像reasoning那样sequential scale

### 12.3 对MLLM发展的启示

这篇paper传递的核心信息：**raw capability ≠ systemic maturity**。当前MLLM在digital tasks上excellence，但"reality gap"是critical frontier。未来方向不是单纯scale参数，而是：

1. 把compute从pre-training shift到RL
2. 把parallel reasoning通过self-distillation internalize到参数中
3. 用physics-based simulation作为ultimate verifier
4. 发展E-CoT处理temporal dynamics

这是一个清晰的从"statistical correlation"到"interaction-driven mastery"的范式转变。

---

## 13. 评估细节补充（Appendix C）

### 13.1 Text-Centric Benchmark的repeated evaluation

为减少metric variance：
- AIME 2024/2025, HMMT25, CNMO2024, BeyondAIME: Repeat=64
- GPQA-Diamond: Repeat=16
- LiveCodeBench: Repeat=16
- IFEval, IFBench: Repeat=4
- MMLU-Pro, SuperGPQA, LiveBench: Repeat=1

AIME系列做64次重复评估——这种rigor在开源工作中不常见。

### 13.2 GUI Grounding的统一prompt

所有GUI grounding任务（ScreenSpot-Pro/V2, OSWorld-G, MMBench-GUI-L2）使用统一格式：
```
<image>
Based on the instruction '{question}', locate the target element 
and output its coordinate point in JSON format.
```

### 13.3 Ablation的evaluation setup

所有ablation在相同billions tokens预训练的checkpoints上做，但不extend到最终checkpoint（computational cost考虑）。评估用few-shot manner。

---

## 14. 论文的深层贡献

这篇paper不只是"又一个10B VL model"，它在三个层面贡献了重要insight：

1. **方法论层面**：PaCoRe把test-time compute scaling从sequential扩展到parallel，并展示了如何通过training让模型学会coordinated reasoning
2. **科学发现层面**：Length diminishment现象和"Missing Trace" Hypothesis揭示了perception RL与reasoning RL的本质差异
3. **哲学层面**：提出从"semantic world model"到"physical world model"的演进路径，用physics作为ultimate verifier

对Karpathy这样的研究者来说，最值得关注的可能是Section 5.2的RL dynamics分析——它指向了一个深刻问题：**当前training corpora缺乏人类视觉认知的explicit cognitive traces，这限制了perception的sequential scaling**。PaCoRe是工程解法，但根本解法可能需要重新思考如何capture和encode人类的perceptual cognitive processes。

Reference: [STEP3-VL-10B ModelScope](https://modelscope.cn/collections/stepfun-ai/Step3-VL-10B), [HuggingFace collection](https://huggingface.co/collections/stepfun-ai/step3-vl-10b), [Step-3 technical report](https://arxiv.org/abs/2507.19427)
