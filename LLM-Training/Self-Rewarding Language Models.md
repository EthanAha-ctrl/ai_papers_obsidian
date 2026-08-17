---
source_pdf: Self-Rewarding Language Models.pdf
paper_sha256: 61bed62389be6445c7a8dc6608641e86a3cc1a4d9b0b19a1f8645fc425997a96
processed_at: '2026-08-12T04:56:37-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Self-Rewarding Language Models

## 一句话总结

让 LLM 自己当裁判给自己的回答打分，然后用这些打分构造偏好数据来训练自己，循环往复，结果发现模型不仅回答变好了，当裁判的能力也变好了。

## 为什么这件事有意思

### 传统 RLHF 的痛点

想象你在训练一个 LLM，标准做法是：

1. 找一堆人标注"回答A比回答B好"
2. 用这些数据训练一个 **reward model**（裁判模型）
3. 把裁判冻起来，用 PPO 训练 LLM 去讨好这个裁判

问题在于：
- 裁判的水平上限就是标注人员的水平
- 裁判被冻住了，LLM 在变强，裁判却不变，迟早脱节

### Self-Rewarding 的提议

干脆让 LLM 自己当裁判。同一个模型，既负责生成回答，也负责给自己（或自己的采样版本）打分。关键在于：这个裁判不是冻的——它跟 generator 共享参数，一起进化。

这就像一个学生做题、自己批改自己的卷子，然后把批改经验用来下次做得更好。听起来像在自欺欺人，但实验表明这个 loop 确实 work，至少在 3 轮迭代内。

## 方法详解

### 整体流程

```
M0 (base Llama 2 70B)
   ↓ SFT on IFT + EFT data
M1 (会做题 + 会当裁判)
   ↓ 用 M1 生成数据，M1 自己打分，构造 preference pairs，DPO 训练
M2 (做题更好 + 当裁判更好)
   ↓ 用 M2 生成数据，M2 自己打分，构造 preference pairs，DPO 训练  
M3 (做题更好+ + 当裁判更好+)
```

每一轮做两件事：
1. **Self-Instruction Creation**：生成新 prompt → 生成 N=4 个 candidate response → 自己给每个 response 打分
2. **Instruction Following Training**：取最高分和最低分的 response 组成 (winner, loser) pair，用 DPO 训练

### 当裁判的 Prompt 设计（关键细节）

论文设计了一个 **additive scoring** prompt，要求模型按 5 个维度逐项评分，每项 0-1 分，累加得 0-5 分：

1. **Relevance** - 回答跟问题相关吗
2. **Coverage** - 要点覆盖了吗
3. **Usefulness** - 有用吗
4. **Clarity** - 清晰吗
5. **Expertise** - 体现专业度吗

模型要先写 chain-of-thought justification，再给 final score。

这个设计比"multiple choice bucket"式 prompt 好太多：

| Prompt 类型 | Pairwise Accuracy |
|---|---|
| Multiple choice (Li et al. 的) | 26.6% |
| Additive (本文的) | 65.1% |

**Intuition**：additive 让模型把"评估质量"分解成 5 个子问题，每个子问题单独推理。multiple choice 要求模型直接跳到"这是 good/average/bad"的 bucket，缺少 reasoning 路径。这暗示 LLM-as-Judge 能力高度依赖 prompt 结构——裁判能力不是 model 的 intrinsic property，是被 prompt scaffold 出来的。

### DPO Loss 详解

DPO 的核心 loss：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]
$$

逐项解释：
- $\pi_\theta$：正在训练的模型（比如 $M_2$）
- $\pi_{\text{ref}}$：reference policy，冻住的上一轮模型（比如 $M_1$）
- $y_w$：winner response（自己给自己打分高的那个）
- $y_l$：loser response（打分低的那个）
- $\beta = 0.1$：KL penalty 系数，控制新模型不要离 reference 太远
- $\sigma$：sigmoid 函数，把 log-odds 压到 [0,1] 概率
- $\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$：这个量就是 implicit reward，衡量"当前模型比 reference 模型有多喜欢这个 response"

整个 loss 在说：让 $\pi_\theta$ 对 $y_w$ 的概率相对于 $\pi_{\text{ref}}$ 提升，对 $y_l$ 的概率相对于 $\pi_{\text{ref}}$ 下降，提升/下降的幅度用 sigmoid 控制饱和。

**有意思的点**：系统里同时存在两个 reward signal：
- **Explicit reward**：LLM-as-Judge 给的 0-5 分，用来构造 preference pair
- **Implicit reward**：$\beta \log[\pi_\theta / \pi_{\text{ref}}]$，DPO 实际优化的量

这两个 reward 不一定 aligned。Explicit judge 可能被 length 欺骗，但 implicit reward 通过 DPO 转化为梯度时还会经过 $\pi_{\text{ref}}$ 的 normalization。

### 数据来源

**Seed IFT data**：3200 examples from Open Assistant，只用 rank 0（最高质量）的第一轮对话。

**Seed EFT data**：1630 train + 541 eval，构造方式有点 tricky：
- Open Assistant 有人类给 multiple responses 排名
- 把 evaluation task 放进 Figure 2 的 prompt format
- target（chain-of-thought + score）由 SFT baseline 生成
- 只接受 score ranking 跟 human ranking 一致的生成结果
- 重采样避免 score 全集中在 4

### Self-Instruction Creation 的细节

1. **生成新 prompt**：用 Llama 2-Chat 70B（固定模型，不是 self-rewarding 模型）+ 8-shot，6 个 IFT seed examples + 2 个 model-generated examples。$T=0.6, p=0.9$。应用 ROUGE-L similarity check、keyword filtering、length filtering（来自 Self-Instruct 方法）。

2. **生成 candidate responses**：从 $M_t$ 采样 $N=4$ 个，$T=0.7, p=0.9$。

3. **Self-evaluate**：从 $M_t$ 采样 3 次取平均分（减少 variance）。

数据规模增长：
- AIFT($M_1$)：3,964 pairs
- AIFT($M_2$)：6,942 pairs

## 实验结果

### AlpacaEval 2.0 Win Rate（vs GPT-4 Turbo）

| 模型 | Win Rate |
|---|---|
| M1 | 9.94% |
| M2 | 15.38% |
| M3 | 20.44% |
| Claude 2 | 17.19% |
| Gemini Pro | 16.85% |
| GPT-4 0613 | 15.76% |
| GPT-4 0314 | 22.07% |

M3 超过 Claude 2、Gemini Pro、GPT-4 0613。起点只是 Llama 2 70B + 3200 条 seed data，没有从更强模型蒸馏。

### Reward Modeling 能力也在提升（核心发现）

| 模型 | Pairwise Acc | Spearman | Kendall τ |
|---|---|---|---|
| SFT Baseline | 65.1% | 0.253 | 0.233 |
| M1 | 78.7% | 0.279 | 0.253 |
| M2 | 80.4% | 0.331 | 0.315 |
| M3 | 81.7% | 0.349 | 0.324 |

关键点：M1→M2→M3 没有再加 EFT data，只加了 preference pair 数据，但 judge 能力还是在提升。

**Intuition**：这说明"判断什么是好回答"和"生成好回答"share underlying capability。当模型在 preference pair 上被 DPO 训练时，它学到了"好 response vs 差 response"的更精细 representation，这个 representation 对生成和评估都有用。类似 multitask learning 的 transfer effect。

### 分项表现（MT-Bench）

| 类别 | SFT | M3 | 变化 |
|---|---|---|---|
| Writing | 8.83 | 9.58 | +0.75 |
| Roleplay | 8.15 | 8.73 | +0.58 |
| Extraction | 6.90 | 7.80 | +0.90 |
| STEM | 9.18 | 9.45 | +0.27 |
| Math | 3.00 | 3.50 | +0.50 |
| Coding | 3.50 | 4.20 | +0.70 |
| Reasoning | 5.30 | 4.80 | **-0.50** |

**Pattern**：writing/roleplay/extraction 大涨，reasoning 反而下降。

**Intuition**：LLM-as-Judge 用的是 verbal evaluation，judge 自己也不擅长评估 math/reasoning 的 correctness。所以 self-reward loop 对 verbal artistry 形成正反馈，对 logical correctness 没有有效 signal。这是 self-reference 系统的典型 pathology——裁判的盲点就是 generator 的盲点，无法自我修正。

### Length Bias（令人担忧的观察）

| 模型 | 平均生成长度 |
|---|---|
| M1 | 1092 tokens |
| M2 | 1552 tokens |
| M3 | 2552 tokens |

长度翻倍。GPT-4 evaluator 已知有 length bias，所以部分 win rate 提升可能是 spurious。

**更深的担忧**：judge 和 generator 共享参数，如果两者都认为 longer = better，就会形成正反馈环——generator 生成更长，judge 给更长打更高分，下一轮 generator 生成更长。这是 self-referential system 的 failure mode，类似 Hofstadter 描述的 strange loop，但这里是 pathological 的。

### EFT Data 的必要性

只用 IFT data 训练的模型系列 $M_1', M_2', M_3'$：
- 经常无法输出正确 score 格式
- Score 容易 degenerate 到 4
- 只能收集 541 / 429 pairs（vs 3964 / 6942）
- 性能远差于 IFT+EFT 版本

**Intuition**：EFT data 不只是教模型"如何评估"，更重要的是稳定输出格式。没有 format stability，self-reward loop 根本 bootstrap 不起来。这暗示 self-improving system 对 initialization 极其 sensitive。

### Positive Examples Only 失败

只加 score=5 的 response 做 SFT augmentation（11254 条）→ 没有提升。必须用 preference pair + DPO。

**Intuition**：SFT on positives 只是 oversample model 已经擅长生成的内容，没有 contrastive signal。DPO 的 preference loss 学习 ranking boundary，告诉模型"什么不好"，这个 negative signal 更 informative。

## 深层思考

### 与 AlphaGo Self-Play 的本质区别

AlphaGo self-play 能超越人类，因为胜负是规则定义的 ground truth。

Self-Rewarding 没有 ground truth。"什么是好回答"由 LLM 自己定义。这更像 self-consistent system 而非 truth-seeking system。

Table 4 显示 Spearman correlation with human 从 0.253 提升到 0.349，确实在向 human alignment 移动。但这是因为模型真的"更懂 quality"，还是在模仿 human data 的 pattern？难以 disentangle。

### Self-Reference 的 Strange Loop

这个系统的核心是 self-reference：$M_t$ 评估 $M_t$ 的生成，训练出 $M_{t+1}$。

- **乐观视角**：如果 judge 能力 generalize，这是 bootstrap ladder
- **悲观视角**：如果 judge bias 和 generator bias correlated，只是 reinforce existing bias

Length bias 观察提示后者至少部分发生。

### 与其他方法的关系

| 方法 | Reward 来源 | Frozen? | Bottleneck |
|---|---|---|---|
| RLHF | Human data → trained RM | Yes | Human performance |
| Constitutional AI | LLM feedback → trained RM | Yes | LLM judge ability |
| RLAIF | LLM-as-Judge → trained RM | Yes | LLM judge ability |
| SPIN | Human labels as winner | No | Human performance |
| ReST | External fixed reward | Yes | Reward model |
| **Self-Rewarding** | LLM-as-Judge = LLM itself | No | Self-reference pathology |

Self-Rewarding 独特之处：reward model 和 policy 共同 evolve。代价是没有 external anchor，系统可能 drift。

### Potential Failure Modes

1. **Length hacking**：已观察到
2. **Format gaming**：模型可能学会 judge-friendly format 而非 genuinely better content
3. **Echo chamber**：judge 和 generator 共享参数，可能强化 shared biases
4. **Mode collapse**：持续 DPO 可能 reduce diversity
5. **Reward hacking through judge**：模型可能生成 judge 评分高但实际不好的 response

### 关于 Scaling Laws 的未探索

论文只跑 3 轮。Open questions：
- Iteration 4, 5, ... 会继续 improve 吗
- 在哪个 iteration saturate
- 是否存在 divergence risk
- Scaling 与 base model size 的关系

参考 [Iterative DPO (Xu et al.)](https://arxiv.org/abs/2312.16682) 显示 iterative DPO 比 non-iterative 更好，但用了 external reward model。

## 可能的 Extensions

### 1. Reasoning 的 Self-Rewarding

当前 seed data 偏 humanities。如果用 math/code seed data + specialized judge prompt（例如 verification-based judge），可能改善 reasoning。参考 [Process Reward Models](https://arxiv.org/abs/2305.20050) 显示 step-level reward 对 math 有帮助。

### 2. Multimodal Self-Rewarding

扩展到 vision-language models，让模型 evaluate 自己的 image descriptions 或 visual reasoning。

### 3. Safety Self-Rewarding

用 safety-specific judge prompt 进行 self-rewarding，可能让 safety 随 iteration 提升。

### 4. Decoupling Judge and Generator

实验 partial decoupling：judge 是 larger model, generator 是 smaller，类似 distillation 但保留 iterative 更新。

### 5. Active Self-Rewarding

用 acquisition function 主动选择 uncertain regions 来 generate，提高 data efficiency。

### 6. Theoretical Analysis

需要理论分析 self-rewarding system 的 fixed points 和 convergence properties。可能借用 iterated function systems 或 Markov chain theory。

## My Take

这篇 paper 最让我兴奋的是 conceptual reframing：**reward model 不是 separate artifact，而是 policy 本身的一个 aspect**。

Implications：
1. **Architectural simplicity**：不用维护 separate reward model
2. **Continual improvement**：消除 frozen bottleneck
3. **Multitask transfer**：judge 和 generation share representations

Worry：
1. **Lack of external anchor**：系统可能 drift
2. **Self-reference pathologies**：length bias, format gaming
3. **Bootstrap dependence**：仍然需要 human-provided seed data

后续可能方向：
1. **Hybrid systems**：结合 self-reward 和 occasional human verification
2. **Process-level self-rewarding**：step-by-step judge 而非 outcome-only
3. **Self-rewarding with verifiable rewards**：对 code/math 等可验证 domain，结合 execution feedback
4. **Multi-agent self-rewarding**：多个模型互相 judge，减少 self-reference pathology

这种 self-improvement paradigm 如果 work，是迈向 superhuman agents 的关键 step。如果不 work（because of self-reference pathologies），我们需要 hybrid approaches with external grounding。

## Web Links

- [Self-Rewarding Language Models (arXiv)](https://arxiv.org/abs/2401.10020)
- [Self-Instruct (Wang et al.)](https://arxiv.org/abs/2212.10560)
- [DPO (Rafailov et al.)](https://arxiv.org/abs/2305.18290)
- [Iterative DPO (Xu et al.)](https://arxiv.org/abs/2312.16682)
- [Constitutional AI (Bai et al.)](https://arxiv.org/abs/2212.08073)
- [SPIN (Chen et al.)](https://arxiv.org/abs/2401.01335)
- [Instruction Backtranslation (Li et al.)](https://arxiv.org/abs/2310.11531)
- [MT-Bench & LLM-as-Judge (Zheng et al.)](https://arxiv.org/abs/2306.05685)
- [ReST (Gulcehre et al.)](https://arxiv.org/abs/2308.08998)
- [Prometheus (Kim et al.)](https://arxiv.org/abs/2310.08491)
- [Open Assistant (Köpf et al.)](https://arxiv.org/abs/2304.07327)
- [Llama 2 (Touvron et al.)](https://arxiv.org/abs/2307.09288)
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)
- [I Am a Strange Loop (Hofstadter)](https://en.wikipedia.org/wiki/I_Am_a_Strange_Loop)
- [Process Reward Models (Lightman et al.)](https://arxiv.org/abs/2305.20050)

---

# Self-Rewarding Language Models 深度解析

## 一、核心Insight：为什么这个工作重要

这篇paper触及了当前LLM alignment的一个根本矛盾。传统RLHF pipeline有一个structural bottleneck：从human preference data训练一个reward model → freeze → 用PPO训练LLM。这造成两个问题：

1. Reward model被human performance ceiling限制
2. Frozen reward model在LLM训练过程中无法继续学习

Self-Rewarding的核心提议是消除"separate reward model"这个abstraction，让generation和rewarding共享同一个模型参数，并且两者在iterative training中互相bootstrap。这有点类似AlphaGo的self-play思想，但有一个关键区别——AlphaGo的reward有ground truth（胜负），而LLM quality的"truth"由LLM自己定义，所以这是一个self-referential system。

参考：[Self-Rewarding Language Models (arXiv)](https://arxiv.org/abs/2401.10020)

## 二、方法的Formal Description

### 2.1 Two Skills Unified in One Model

模型需要同时具备两种能力：
- **Instruction following**: 给定prompt $x$ 生成高质量response $y$
- **Self-Instruction creation**: 生成新的(prompt, response) pairs并self-evaluate

### 2.2 Self-Instruction Creation Pipeline

每个iteration $t$ 的数据生成过程：

**Step 1 - Generate prompt** $x_i$:
用Llama 2-Chat 70B with 8-shot prompting (6个来自IFT seed, 2个来自model-generated), $T=0.6, p=0.9$。应用ROUGE-L similarity filtering, keyword filtering, length filtering (来自Self-Instruct)。

**Step 2 - Generate N candidate responses** $\{y_i^1, \dots, y_i^N\}$:
$N=4$, $T=0.7, p=0.9$, 从当前model $M_t$ 采样。

**Step 3 - Self-evaluate**:
用同一个 $M_t$ 作为LLM-as-Judge，对每个candidate打分 $r_i^n \in [0, 5]$。为减少variance，采样3次取平均。

### 2.3 LLM-as-a-Judge Prompt Design（关键设计）

这是论文中一个被低估的细节。他们设计了**additive score-counting** prompt（Figure 2），包含5个cumulative criteria:
1. **Relevance** (0-1分)
2. **Coverage** (0-1分)
3. **Usefulness** (0-1分)
4. **Clarity** (0-1分)
5. **Expertise** (0-1分)

总分累加至5分。模型先chain-of-thought给出justification，然后给出final score。

对比实验（Table 5）显示这种design相比multiple choice bucket形式（来自Li et al. 2024的Instruction Backtranslation）有巨大差距：

| EFT Prompt | Pairwise Acc | Spearman | Kendall τ |
|---|---|---|---|
| Multiple choice | 26.6% | -0.18 | -0.16 |
| Additive (theirs) | 65.1% | 0.25 | 0.23 |

**我的解读**：additive prompt让模型能decompose evaluation为sub-problems，每个criterion单独评估。Multiple choice形式要求模型一次性映射到quality bucket，缺少结构化reasoning路径。这暗示了LLM-as-Judge能力高度sensitive to prompt structure——这本身也说明"reward model ability"在LLM中是prompt-dependent的，不是well-defined的property。

### 2.4 Iterative DPO Training

Model sequence:

$$
M_0 \xrightarrow{\text{SFT on IFT+EFT}} M_1 \xrightarrow{\text{DPO on AIFT}(M_1)} M_2 \xrightarrow{\text{DPO on AIFT}(M_2)} M_3
$$

Preference pairs构造：对每个prompt $x_i$，取score最高和最低的responses作为 $(y_i^w, y_i^l)$，若score相同则discard。

数据规模：
- $\text{AIFT}(M_1)$: 3,964 pairs → train $M_2$
- $\text{AIFT}(M_2)$: 6,942 pairs → train $M_3$

### 2.5 DPO Loss Function详解

DPO的loss如下：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]
$$

变量含义：
- $\pi_\theta$: 当前训练的policy（即 $M_{t+1}$）
- $\pi_{\text{ref}}$: reference policy（frozen，初始化为 $M_t$）
- $y_w$: winning response（self-judge score最高的）
- $y_l$: losing response（score最低的）
- $\beta = 0.1$: KL penalty coefficient，控制policy偏离reference的程度
- $\sigma(\cdot)$: sigmoid function

**隐式reward**: DPO实际定义了一个implicit reward:
$$
r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

所以Self-Rewarding system中有两个reward signal：
1. **Explicit**: LLM-as-Judge给出的score $r_i^n \in [0, 5]$（用于构造preference pair）
2. **Implicit**: DPO优化中的 $\beta \log[\pi_\theta / \pi_{\text{ref}}]$（用于实际梯度更新）

这两个reward不一定aligned——这是个值得深究的点。

### 2.6 训练Hyperparameters

| 阶段 | LR | Decay | Batch | Dropout | β | Loss |
|---|---|---|---|---|---|---|
| SFT | 5.5e-6 | cosine→1.1e-6 | 16 | 0.1 | - | target tokens only |
| DPO | 1e-6 | →1e-7 | 16 | 0.1 | 0.1 | - |

Early stopping：每200步save checkpoint，用Claude 2在253 validation examples上pairwise compare with previous checkpoint (AlpacaEval prompt format)。

## 三、实验结果深度分析

### 3.1 AlpacaEval 2.0 (Win Rate vs GPT-4 Turbo)

| Iteration | Win Rate |
|---|---|
| M1 (SFT on IFT+EFT) | 9.94% |
| M2 (DPO on AIFT(M1)) | 15.38% |
| M3 (DPO on AIFT(M2)) | 20.44% |

对比reference models:
- Claude 2: 17.19%
- Gemini Pro: 16.85%
- GPT-4 0613: 15.76%
- GPT-4 0314: 22.07%

M3超过Claude 2, Gemini Pro, 和GPT-4 0613。这是从Llama 2 70B base + 仅3200 IFT seed examples出发，没有任何distillation from stronger model。

### 3.2 Reward Modeling能力也提升（这是关键发现）

| Model | Training Data | Pairwise Acc | 5-best % | Spearman | Kendall τ |
|---|---|---|---|---|---|
| SFT Baseline | IFT | 65.1% | 39.6% | 0.253 | 0.233 |
| M1 | IFT+EFT | 78.7% | 41.5% | 0.279 | 0.253 |
| M2 | +AIFT(M1) | 80.4% | 44.3% | 0.331 | 0.315 |
| M3 | +AIFT(M2) | 81.7% | 43.2% | 0.349 | 0.324 |

**关键观察**：从M1到M2到M3，reward modeling能力monotonically提升，**尽管没有任何额外的EFT data**。AIFT数据是preference pair格式，不是judge格式。

**我的interpretation**：这暗示general instruction following能力与judge能力share underlying capability。可能机制：
1. 当模型在preference pair上训练时，它learn了"什么是好response"的更精细representation
2. 这个representation对generation和evaluation都有用
3. 类似于multitask learning的transfer effect（参考Collobert & Weston 2008）

### 3.3 MT-Bench分项分析（Table 10）

| Model | Writing | Roleplay | Reasoning | Math | Coding | Extraction | STEM | Humanities | Overall |
|---|---|---|---|---|---|---|---|---|---|
| SFT | 8.83 | 8.15 | 5.30 | 3.00 | 3.50 | 6.90 | 9.18 | 9.95 | 6.85 |
| M3 | 9.58 | 8.73 | 4.80 | 3.50 | 4.20 | 7.80 | 9.45 | 9.95 | 7.25 |

**Pattern**: 
- 大幅提升：Writing (+0.75), Roleplay (+0.58), Extraction (+0.90)
- 小幅提升：STEM, Math, Coding
- Reasoning几乎不变甚至下降 (-0.50)

这个pattern提示self-rewarding主要增强**verbal artistry**而非**logical reasoning**。这可能与LLM-as-Judge prompt本身是verbal evaluation有关——judge自己也无法well-evaluate correctness of math/reasoning。

### 3.4 Length Bias Concern

| Model | Avg Generation Length |
|---|---|
| M1 | 1092 tokens |
| M2 | 1552 tokens |
| M3 | 2552 tokens |

Length几乎doubles。GPT-4 evaluator已知存在length bias，所以一部分win rate提升可能是spurious。论文承认这是limitation，未深入分析。这是self-rewarding系统的一个潜在failure mode：**模型可能learn to generate verbose response而非substantively better response**，因为judge（同一个模型）也认为longer=better。这是self-reference system的典型pathology。

### 3.5 NLP Benchmarks的Alignment Tax

| | ARC-C | HellaSwag | GSM8K | MMLU | NQ |
|---|---|---|---|---|---|
| Llama 2 | 57.40 | 85.30 | 56.80 | 68.90 | 25.30 |
| M3 | 53.13 | 83.29 | 57.70 | 69.37 | 31.86 |

M3在某些benchmark上略降，重现InstructGPT的"alignment tax"现象。MMLU和NQ反而提升（因为instruction-following能力transfer到QA）。

### 3.6 EFT Data的必要性（Appendix A.3）

只用IFT data训练的模型系列 $M_1', M_2', M_3'$：
- 无法可靠输出score格式
- Score经常degenerate到4
- 只能收集到541 / 429 pairs (vs 3,964 / 6,942)
- 性能远差于IFT+EFT的版本

**Insight**: EFT data不只是"judge能力"的来源，还是**judge输出的format stability**的来源。没有format stability，self-reward loop无法bootstrap。

### 3.7 SFT with Positive Examples Only失败（Appendix A.4）

只加score=5的responses做SFT augmentation（11,254 examples）→ 没有提升。必须用preference pair + DPO。

**为什么？** 我推测：
- SFT on positive examples只是oversample model already generates well的内容
- 没有contrastive signal告诉模型"什么不好"
- DPO的preference loss本质上学习ranking boundary，提供更informative的gradient

## 四、深度思考与可能的Limitations

### 4.1 Self-Reference的Strange Loop

这个系统的核心是self-reference：模型 $M_t$ 评估模型 $M_t$ 的generation，然后训练出 $M_{t+1}$。Hofstadter在《I Am a Strange Loop》中讨论的self-reference现象在这里以algorithmic形式实现。

关键question：这种系统能否真的escape human-level ceiling？

- **乐观视角**：如果judge能力能generalize improvement，那么这是一个bootstrap ladder
- **悲观视角**：如果judge bias和generation bias是correlated的，那只是reinforcing existing bias

Length bias的观察提示后一种情况至少部分发生。

### 4.2 与其他self-improvement methods的关系

| 方法 | Reward来源 | Frozen? | Bottleneck |
|---|---|---|---|
| RLHF (Ouyang et al. 2022) | Human data → trained RM | Yes | Human performance |
| Constitutional AI (Bai et al. 2022b) | LLM feedback → trained RM | Yes | LLM (frozen) judge ability |
| RLAIF (Lee et al. 2023) | LLM-as-Judge → trained RM | Yes | LLM (frozen) judge ability |
| SPIN (Chen et al. 2024) | Human labels as y_w | No | Human performance |
| ReST (Gulcehre et al. 2023) | External fixed reward | Yes | Reward model |
| **Self-Rewarding (this work)** | LLM-as-Judge = LLM itself | No | Self-reference pathology |

Self-Rewarding的独特之处是reward model与policy共同evolve。这同时也意味着没有external anchor，系统可能drift。

### 4.3 与AlphaGo Self-Play的本质区别

AlphaGo self-play能超越human，因为胜负是objective ground truth（规则定义的）。

Self-Rewarding没有这样的ground truth。"什么是good response"由LLM自己定义。这更像是一个self-consistent system而非truth-seeking system。所以能否真正超越human-level judge能力是open question。

论文Table 4显示Spearman correlation with human从0.253提升到0.349，确实在向human alignment方向移动。但这是否因为模型真的"更懂quality"，还是因为模型在模仿human data的pattern？难以disentangle。

### 4.4 潜在Failure Modes

1. **Length hacking**: 已经观察到
2. **Format gaming**: 模型可能learn judge-friendly format rather than genuinely better content
3. **Echo chamber**: judge和generator share parameters, 可能强化shared biases
4. **Mode collapse**: 持续DPO可能reduce diversity
5. **Reward hacking through judge**: 模型可能learn to generate responses that its own judge scores high but are not actually good

### 4.5 关于"Scaling Laws"的未探索

论文只跑了3个iterations。Open questions:
- Iteration 4, 5, ... 会不会继续improve？
- 在哪个iteration saturate？
- 是否存在divergence risk？
- 这种scaling与base model size的关系？

参考：[Iterative DPO (Xu et al.)](https://arxiv.org/abs/2312.16682)显示iterative DPO比non-iterative更好，但用了external reward model。

### 4.6 关于Multitask Transfer的机制

论文声称"task transfer between reward modeling and instruction following"，但没有直接probe这个mechanism。一个可能的实验：分别track judge ability和generation ability on held-out数据，看两者的correlation。如果两者强相关，说明shared capability；如果只一个提升而另一个不提升，说明是independent improvements。

### 4.7 关于Reward Model能力的来源

M1的judge能力从EFT data来，EFT data的justifications是SFT baseline生成的（基于Open Assistant human rankings筛选）。所以initial seed还是human-derived。Self-Rewarding能否真正bootstrap from weaker signal是question。

参考：[Prometheus (Kim et al. 2023)](https://arxiv.org/abs/2310.08491)也train LLM to be judge，但只用judge data，没有combine with instruction following。

### 4.8 关于DPO的Limitation

DPO的preference loss假设preference是deterministic（Bradley-Terry model）。实际上不同response之间的preference可能是noisy的，特别是当self-judge有variance时。论文采样3次取平均，但没有报告score的variance是否大。

## 五、可能的Extensions和Future Work

### 5.1 推理/Reasoning的Self-Rewarding

当前Open Assistant seed data偏重humanities。如果用math/code seed data并设计specialized judge prompt（例如verification-based judge），可能改善reasoning performance。

参考：[Process Reward Models](https://arxiv.org/abs/2305.20050)和[PRM800K](https://arxiv.org/abs/2305.20050)显示step-level reward对math有帮助。

### 5.2 Multimodal Self-Rewarding

扩展到vision-language models，让模型evaluate自己的image descriptions或visual reasoning。

### 5.3 Safety Self-Rewarding

用safety-specific judge prompt（例如evaluate harmlessness）进行self-rewarding，可能让safety随iteration提升。

### 5.4 Decoupling Judge and Generator

虽然论文强调unification的好处，但也可以实验partial decoupling: e.g., judge是larger model, generator是smaller，类似distillation但保留iterative更新。

### 5.5 Active Self-Rewarding

不是random sample N=4 responses，而是用acquisition function主动选择uncertain regions来generate，提高data efficiency。

### 5.6 Theoretical Analysis

需要理论分析self-rewarding system的fixed points和convergence properties。可能借用iterated function systems或Markov chain theory。

## 六、Personal Takeaways

这篇paper最让我兴奋的不是实验结果，而是conceptual proposal：**reward model不是separate artifact，而是policy本身的一个aspect**。这个reframing有几个implications：

1. **Architectural simplicity**: 不用maintain separate reward model
2. **Continual improvement**: 消除frozen bottleneck
3. **Multitask transfer**: judge和generation share representations

但也有worry：
1. **Lack of external anchor**: 系统可能drift
2. **Self-reference pathologies**: length bias, format gaming
3. **Bootstrap dependence**: 仍然需要human-provided seed data

我认为后续工作可能朝几个方向：
1. **Hybrid systems**: 结合self-reward和occasional human verification
2. **Process-level self-rewarding**: step-by-step judge而非outcome-only
3. **Self-rewarding with verifiable rewards**: 对于code/math等可验证domain，结合execution feedback
4. **Multi-agent self-rewarding**: 多个模型互相judge，减少self-reference pathology

这种self-improvement paradigm如果work，是迈向superhuman agents的关键step；如果不work（because of self-reference pathologies），我们需要hybrid approaches with external grounding。

参考链接：
- [Self-Rewarding Language Models (arXiv)](https://arxiv.org/abs/2401.10020)
- [Self-Instruct (Wang et al.)](https://arxiv.org/abs/2212.10560)
- [DPO (Rafailov et al.)](https://arxiv.org/abs/2305.18290)
- [Iterative DPO (Xu et al.)](https://arxiv.org/abs/2312.16682)
- [Constitutional AI (Bai et al.)](https://arxiv.org/abs/2212.08073)
- [SPIN (Chen et al.)](https://arxiv.org/abs/2401.01335)
- [Instruction Backtranslation (Li et al.)](https://arxiv.org/abs/2310.11531)
- [MT-Bench & LLM-as-Judge (Zheng et al.)](https://arxiv.org/abs/2306.05685)
- [ReST (Gulcehre et al.)](https://arxiv.org/abs/2308.08998)
- [Prometheus (Kim et al.)](https://arxiv.org/abs/2310.08491)
- [Open Assistant (Köpf et al.)](https://arxiv.org/abs/2304.07327)
- [Llama 2 (Touvron et al.)](https://arxiv.org/abs/2307.09288)
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)
- [I Am a Strange Loop (Hofstadter)](https://en.wikipedia.org/wiki/I_Am_a_Strange_Loop)
