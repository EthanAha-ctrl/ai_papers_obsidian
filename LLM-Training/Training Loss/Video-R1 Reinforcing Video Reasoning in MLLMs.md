---
source_pdf: Video-R1 Reinforcing Video Reasoning in MLLMs.pdf
paper_sha256: 69574ff35e6bf69c726d7afa34b0e0a49074b685b95f5687aff0361d508e2178
processed_at: '2026-08-13T00:42:00-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Video-R1 用人话版

Andrej，好，我把学术黑话剥掉，用最直白的方式讲讲这篇paper到底干了啥。

## 一句话总结

**他们教一个7B的小模型看视频推理，方法是在训练时偷偷把视频帧打乱，看模型还能不能答对——如果打乱了照样答对，说明它在偷懒只看单帧，就惩罚它。**

就这么简单。剩下的都是工程细节。

---

## 问题是什么？

想象你训练一个学生做视频理解题。你给他看一段视频，问他"球从左边滚到右边花了多久？"

学生有两种答题策略：
- **认真策略**: 看完整段视频，观察球的运动轨迹，数帧数
- **偷懒策略**: 只看最后一帧球在右边，猜一个数

如果你只看答案对不对给reward，学生会发现偷懒策略往往也能蒙对，而且更快。久而久之，他就永远只看单帧了。

这就是原版GRPO的问题——它只奖励"答案对"，不关心"你怎么得到答案的"。在text domain这个问题不明显（文字推理必须一步步推），但在video domain，模型可以走捷径：很多video QA其实看一帧就能答。

DeepSeek-R1 [https://arxiv.org/abs/2501.12948] 在text domain work得很好，但直接搬到video，模型学会作弊。Video-UTR [https://arxiv.org/abs/2502.12081] 也独立发现了这个问题，叫"unhackable temporal rewarding"。

---

## T-GRPO 的核心idea

**作弊检测器。**

训练时，对每个问题，做两次：

1. 正常顺序播放视频，让模型答8次，记录正确率 $p$
2. 把视频帧随机打乱，让模型答4次，记录正确率 $\tilde{p}$

如果 $p \geq \tilde{p}$，说明模型在正常顺序上至少不比打乱时差——可能真的在用时间信息。给一个extra reward $\alpha = 0.3$。

如果 $p < \tilde{p}$，说明打乱了反而答得更好——这不可能，说明模型在乱猜。不给temporal reward。

就这么个道理。**通过shuffle frames构造一个"作弊基线"，强迫模型证明自己配拿temporal reward。**

---

## 公式拆解（用人话翻译）

### 公式1: temporal reward

$$r_t = \begin{cases} \alpha, & \text{if } p \geq \tilde{p} \\ 0, & \text{otherwise} \end{cases}$$

- $r_t$: temporal bonus，0.3分
- $p$: 正常顺序正确率，比如8个答对5个就是0.625
- $\tilde{p}$: 打乱顺序正确率
- $\alpha$: 0.3，拍脑袋调出来的，Figure 7做了sensitivity analysis显示0.2~0.3都行

**翻译**: "你正常看视频比看乱序视频强，就奖励你。"

### 公式2: temporal-augmented reward

$$R_i = \begin{cases} r_i + r_t, & \text{if } o_i \text{ is correct} \\ r_i, & \text{otherwise} \end{cases}$$

- $R_i$: 第i个回答的最终reward
- $r_i$: 基础reward（答案对不对 + 格式对不对）
- $r_t$: 只加在"答对的回答"上

**关键点**: temporal bonus只加给那些本来就答对的回答。为什么？因为如果加给所有回答，模型答错时也能拿temporal bonus，等于变相鼓励模型用temporal策略乱猜。只加给correct response，意思是"你用了temporal info并且答对了，很好，强化这个策略"。

### 公式3: advantage

$$A_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$$

- $A_i$: 第i个回答的advantage（该被强化还是削弱）
- $\{R_j\}$: 同一个group里所有回答的rewards
- mean / std: 在group内归一化

**翻译**: "在一个group里，比平均好的advantage为正，差的为负。归一化让数值稳定。"

这是GRPO [https://arxiv.org/abs/2402.03300] 的标准操作。不用绝对的critic（value network），用group内的相对排名来估计advantage。好处是不用训critic，坏处是group size要够大（论文用G=8）。

### 公式4: policy update

$$\mathcal{J}_{\text{T-GRPO}}(\theta) = \mathbb{E}_{q,\{o_i\}}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)A_i\right) - \beta\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)\right]$$

- $\rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}$: 新policy比旧policy的概率比
- $\epsilon$: clip范围，防policy更新太猛
- $\beta = 0.04$: KL惩罚系数，防policy飞走
- $\pi_{\text{ref}}$: SFT后的reference policy，frozen

这就是标准PPO [https://arxiv.org/abs/1707.06347] 那套，没啥新东西。

**重要**: 只有ordered group参与policy update。shuffled group纯粹用来算 $\tilde{p}$ 然后被丢弃。如果让shuffled group也参与update，就等于在乱序数据上训练，违背初衷。

### 公式5: length reward

$$R_i = R_i + \omega \quad \text{if correct and } 320 \leq \text{len}(o_i) \leq 512$$

- $\omega = 0.2$
- $l_{\min} = 320$ tokens, $l_{\max} = 512$ tokens

**为什么要这个？** 因为不加的话，模型会越答越短——既然短答案也能拿correctness reward，为啥要长篇大论？Figure 8的ablation显示去掉length reward后response长度持续下降，性能也跟着掉。

这是个通用问题：RL training如果不加约束，model会collapse到"最省力"的答题方式。length reward是个简单粗暴的fix。

---

## 数据：image当video的"预训练"

Video-R1-260k这个数据集：
- Video 116k（通用视频）
- Image 146k（chart, OCR, math, knowledge, spatial）

Image数据比video还多，这看着奇怪，其实很聪明。

**Intuition**: video reasoning = 空间推理 + 逻辑推理 + temporal推理。前两个在image上就能学。先在image上学会multi-step reasoning（数学题、图表题），这些技能是modality-agnostic的——逻辑链就是逻辑链，不管输入是图还是视频。然后迁移到video，只需要额外学temporal维度。

类比：教小孩写作文，先教他写句子（image reasoning），再教他讲故事（video reasoning）。不会一开始就让他写长篇叙事。

Ablation证实：去掉image数据，VSI-Bench从34.6掉到32.3，VideoMMMU从49.8掉到45.8。掉了2-4个点，证明image data确实在bootstrap reasoning能力。

---

## 训练流程

### Stage 1: SFT cold start
- 数据: Video-R1-CoT-165k（用Qwen2.5-VL-72B生成CoT标注）
- 目的: 让模型学会"怎么写reasoning chain"
- 产物: Qwen2.5-VL-7B-SFT

### Stage 2: RL with T-GRPO
- 数据: Video-R1-260k
- 只跑1k steps（约15小时，8×A100）
- 产物: Video-R1-7B

**只1k steps就能显著提升**，这是R1范式最惊人的特性。rule-based RL的sample efficiency远高于传统RLHF。Appendix里扩展到10k steps还有进一步提升，说明远未收敛。

---

## 结果：7B小模型干翻GPT-4o

| Model | VSI-Bench | VideoMMMU | MMVU |
|-------|-----------|-----------|------|
| GPT-4o | 34.0 | 61.2 | 75.4 |
| Qwen2.5-VL-7B base | 27.7 | 47.8 | 59.2 |
| Qwen2.5-VL-7B-SFT | 31.8 | 47.4 | 61.3 |
| **Video-R1-7B (16f)** | 34.6 | 49.8 | 64.2 |
| **Video-R1-7B (64f)** | **37.1** | 52.4 | 63.8 |

VSI-Bench [https://arxiv.org/abs/2412.14171] 是spatial reasoning benchmark，需要模型从多帧构建3D空间记忆。Video-R1-7B用64帧跑出37.1%，**超过GPT-4o的34.0%**。

一个7B小模型，在spatial reasoning上超过商业大模型——这结果挺炸裂的。

### 几个有意思的现象

**1. SFT不总是work**: Qwen2.5-VL-7B-SFT在VideoMME上反而比base model低（52.8 vs 53.1）。但RL后Video-R1提升到57.4+。

这印证了"SFT memorizes, RL generalizes" [https://arxiv.org/abs/2501.17161] 的现象。SFT容易过拟合到训练分布的format，RL才真正学到task-solving ability。

**2. 更多帧=更好**: 16→32→64帧几乎在所有benchmark上都单调提升。VideoMME从57.4→59.3→61.4，+4个点。说明long-context video reasoning是未来重要方向。

**3. Aha moment**: Figure 4的例子，模型在推理过程中会停下来重新审视视频证据。这种self-reflection行为是emergent的，没人explicit教过。这是R1范式最迷人的特性。

---

## Ablation 三个关键验证

| Variant | VSI-Bench | 干了啥 |
|---------|-----------|--------|
| wo-image | 32.3 | 去掉所有image data |
| wo-temporal | 32.7 | 用原版GRPO，不用T-GRPO |
| zero | 31.8 | 跳过SFT直接RL |
| **full** | **34.6** | 完整Video-R1 |

三个组件各贡献约2-3个点。T-GRPO单独贡献1.9点，看似不大，但Figure 6的temporal reasoning percentage分析更说明问题：

- Video-R1 (T-GRPO): **75.0%** responses包含temporal reasoning
- Video-R1-wo-temporal (GRPO): **60.2%**

差15个点！这是用独立的Qwen2.5-VL-72B来evaluate的，比较客观。说明T-GRPO确实改变了模型的推理策略，而不只是刷分。

---

## Training Curves 里的小故事

Figure 5(c)的response length曲线：**先降后升再稳定**。

作者猜测这是个transition phase：
1. 刚开始RL，模型先抛弃SFT学到的rigid reasoning style（长度下降）
2. 然后探索新的、更长的reasoning policy（长度上升）
3. 最后稳定在新的平衡点

DeepSeek-R1原文里也观察到类似现象。这个"unlearn then relearn"的过程可能是R1范式emergent reasoning的机制之一。

---

## 给Karpathy的几个intuition takeaways

**1. Contrastive reward是一个underexplored design space。** T-GRPO本质是"用adversarial input检测shortcut，把检测结果作为reward"。这个思想可以推广：任何inference shortcut都可以通过construct adversarial input来detect。

比如：
- 数学推理中"背答案"shortcut → 用数字扰动检测
- 代码生成中"抄模板"shortcut → 用变量重命名检测
- Spatial reasoning中"只看局部"shortcut → 用视角变换检测

**2. Image data作为video reasoning的"骨架预训练"。** 这个idea可以推广到任何modality扩展场景。新modality难收集数据时，先用旧modality的reasoning data建"推理骨架"，再迁移。

**3. RL training会collapse到最省力策略。** 不加length reward，response越来越短。这是RLHF的通用问题。需要explicit constraint维持reasoning effort。但固定length window [320, 512]是粗糙的fix，理想情况应该adaptive——难题长答，简单题短答。

**4. 7B + 1k steps就能超GPT-4o。** 说明video reasoning benchmark远未饱和，特别是需要temporal modeling的任务。R1范式在多模态的扩展才刚开始。

**5. Aha moment是emergent的。** self-reflection行为不需要explicit supervision，rule-based RL + enough exploration就能实现。这是R1范式最deep的insight。

---

## 联想：这篇paper的更大意义

Video-R1是R1范式在video domain的first attempt，但更重要的是它提出了一种**general方法论**：

> 当你想让模型用某种specific能力（temporal, spatial, causal），但outcome reward无法区分"真用"和"走捷径"时，construct adversarial input来检测shortcut，把检测结果作为reward shaping。

这个framework的应用远不止video：
- **Causal reasoning**: 用counterfactual input检测因果推理
- **Multi-hop reasoning**: 用单跳问题检测是否真的多跳
- **Tool use**: 用broken tool检测是否真的在用tool
- **Grounding**: 用干扰物检测是否真的ground到evidence

Video-R1的T-GRPO是这个framework的一个具体实例：adversarial input = shuffled frames, shortcut = single-frame reasoning, reward = contrastive bonus。

---

## References

- Video-R1 paper: https://github.com/tulerfeng/Video-R1
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- Video-UTR: https://arxiv.org/abs/2502.12081
- VSI-Bench: https://arxiv.org/abs/2412.14171
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- SFT memorizes, RL generalizes: https://arxiv.org/abs/2501.17161
- Kimi k1.5: https://arxiv.org/abs/2501.12599
- PPO: https://arxiv.org/abs/1707.06347

---

Andrej，最核心的intuition就一句：**shuffle frames是最简单的"测谎仪"，测模型有没有真在用时间信息**。这个idea简单到优美，engineering上也好实现。整个paper的contribution就是把这个intuitionformalize成reward function，然后证明它work。

---

# Video-R1: Reinforcing Video Reasoning in MLLMs 深度解析

Andrej, 这篇paper非常有意思，它在DeepSeek-R1的rule-based RL范式基础上，尝试解决一个很本质的问题：**如何让MLLM真正"看懂"视频的时间结构，而非走捷径看单帧**。我来build一下intuition。

## 1. 核心Motivation: "Shortcut"问题

DeepSeek-R1在text domain取得成功后，社区自然想把这个paradigm推到multimodal。Kimi k1.5 [https://arxiv.org/abs/2501.12599] 和 Skywork R1V [https://arxiv.org/abs/2501.12599] 已经在image reasoning上做了尝试，但video domain几乎空白。

作者识别出两个核心问题：

**问题1: Temporal Shortcut。** 原版GRPO [https://arxiv.org/abs/2402.03300] 只给outcome-based reward (答案对不对)，没有任何explicit signal告诉模型"你应该用temporal information"。结果是模型学会走捷径——只看一两个key frame就能蒙对答案，因为很多video QA task的答案其实藏在某一帧里 (e.g., 物体是什么颜色)。这种policy在训练distribution上能work，但泛化到真正需要temporal reasoning的任务就崩了。Figure 1里给出了一个很好的例子。

Video-UTR [https://arxiv.org/abs/2502.12081] 也独立发现了类似问题，称为"unhackable temporal rewarding"，说明这是community共识的痛点。

**问题2: Data Scarcity。** 现有video dataset大多是perception task (e.g., action recognition, video captioning)，真正需要long chain-of-thought reasoning的高质量video data非常稀缺。

## 2. T-GRPO: Contrastive Temporal Reward — 这是这篇paper的真正创新

核心idea极其elegant: **通过shuffle frames构造contrastive signal，强迫模型证明自己在用temporal信息**。

### 2.1 算法intuition

想象你在训练一个学生做video quiz。如果学生能在正常顺序的视频上答对，但把视频帧打乱后还能答对——那说明他根本没在用时间信息，只是看了某一帧。T-GRPO本质上是一个"作弊检测器"：只有当模型在ordered sequence上的正确率高于shuffled sequence时，才给extra reward。

### 2.2 数学详解

**Step 1: 生成两组responses。** 对同一个question $q$ + video $v$:
- Ordered组: 用原始帧顺序 $\{o_i\}_{i=1}^G$，G=8个samples
- Shuffled组: 用随机打乱的帧顺序 $\{\tilde{o_i}\}_{i=1}^{\tilde{G}}$，$\tilde{G}=G/2=4$（为了效率减半）

设 $p$ = ordered组正确率， $\tilde{p}$ = shuffled组正确率。

**Step 2: Temporal reward (公式1)。**

$$r_t = \begin{cases} \alpha, & \text{if } p \geq \tilde{p} \\ 0, & \text{otherwise} \end{cases}$$

- $r_t$: temporal reward标量
- $\alpha$: hyperparameter，论文设0.3，控制temporal reward的magnitude
- $p$: ordered组正确率，e.g., 8个responses里有5个对，p=5/8=0.625
- $\tilde{p}$: shuffled组正确率
- $p \geq \tilde{p}$ 是触发条件——模型在正常顺序上至少要和打乱时一样好

**Step 3: Temporal-augmented reward (公式2)。**

$$R_i = \begin{cases} r_i + r_t, & \text{if } o_i \text{ is correct} \\ r_i, & \text{otherwise} \end{cases}$$

- $R_i$: 第i个response的最终reward
- $r_i$: 第i个response的base reward，包含correctness reward + format reward (遵循DeepSeek-R1)
- $r_t$: 只加在correct responses上——这非常关键，否则会"稀释"signal

**为什么只加在correct responses上？** 这里有一个微妙的design choice。如果把 $r_t$ 加在所有responses上，那model在ordered组答错时也会拿到temporal bonus，这会push model去"打赌"temporal策略，可能反而学到无意义的策略。只加在correct responses上，确保了"强化已经在work的策略"，符合RLHF的reward shaping原则。

**Step 4: Group-relative advantage (公式3)。**

$$A_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$$

- $A_i$: 第i个response的advantage
- $\{R_j\}$: 当前group内所有responses的rewards
- mean/std: 在group内normalize，让advantage成为zero-mean, unit-variance

这是标准GRPO的advantage计算，no batch-level normalization，而是group-level——这是DeepSeekMath [https://arxiv.org/abs/2402.03300] 引入的关键设计，避免不同difficulty的questions之间的reward scale差异问题。

**Step 5: Policy update (公式4)。**

$$\mathcal{J}_{\text{T-GRPO}}(\theta) = \mathbb{E}_{q,\{o_i\}}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right)A_i\right) - \beta\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)\right]$$

让我逐项讲：
- $\mathcal{J}_{\text{T-GRPO}}(\theta)$: T-GRPO的expected objective (要maximize)
- $q$: 输入question
- $\{o_i\}_{i=1}^G$: G个sampled responses (ordered组)
- $\pi_\theta(o_i|q)$: 当前policy对response $o_i$ 的概率
- $\pi_{\theta_{\text{old}}}(o_i|q)$: 旧policy (上一iteration) 的概率
- $\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}$: PPO的importance ratio, 记作 $\rho_i$
- $\min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)A_i)$: clipped surrogate objective (PPO核心)
- $\epsilon$: clipping range, 限制policy update幅度, 通常0.1-0.2
- $\beta$: KL penalty coefficient, 论文设0.04
- $\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$: 当前policy和reference policy之间的KL散度, 防止policy偏离太远
- $\pi_{\text{ref}}$: reference policy (通常是SFT后的初始policy, frozen)

注意：虽然ordered和shuffled两组responses都生成，但只有ordered组参与policy update。Shuffled组纯粹用于计算 $\tilde{p}$ 来决定 $r_t$ 的值。这点很容易忽略但很关键——否则会变成在shuffled data上训练，违背初衷。

### 2.3 Length Reward (公式5)

$$R_i = \begin{cases} R_i + \omega, & \text{if } o_i \text{ is correct and } l_{\min} \leq \text{len}(o_i) \leq l_{\max} \\ R_i, & \text{otherwise} \end{cases}$$

- $\omega$: length reward magnitude, 论文设0.2
- $l_{\min} = 320$, $l_{\max} = 512$ (token数)
- $\text{len}(o_i)$: response $o_i$ 的token长度

这个设计的intuition是：避免overthinking (太长) 同时避免浅薄reasoning (太短)。DeepSeek-R1 paper里观察到RL训练会让response自然变长，但如果不管它，模型可能collapse到极端长的无意义循环，或者为了快速拿reward而短路回答。Length window [320, 512]是个heuristic sweet spot。

Figure 8的ablation很有说服力：去掉length reward后，response长度持续下降，性能也跟着下降。说明模型会"偷懒"——既然短回答也能拿correctness reward，干嘛长篇大论？

## 3. Dataset Construction: Image-Video Hybrid的精妙

### 3.1 为什么混入image data？

Video-R1-260k的组成（见Figure 2）：
- General (Video, 116k): 通用video
- General (Image, 15k): 通用image QA
- Chart (Image, 21k): 图表reasoning
- OCR (Image, 16k): 文字识别
- Math (Image, 37k): 数学推理
- Knowledge (Image, 37k): 知识推理
- Spatial (Image, 20k): 空间推理

Image数据 (146k) 反而比video数据 (116k) 还多。这个配比很有意思。

**Intuition**: image reasoning是video reasoning的"基础技能"。模型先在static context里学会multi-step reasoning (math, chart, knowledge)，这些技能是modality-agnostic的——逻辑链、空间推理、OCR。然后再迁移到video domain，只需要额外学temporal维度。

类比一下: 教一个人弹钢琴曲，你会先让他练指法 (image) 再学乐感 (video)，而不是一开始就让他即兴演奏。Image data承担了"教推理骨架"的角色。

Ablation (Table 2)证实了这点：Video-R1-7B-wo-image (去掉所有image data) 在VSI-Bench上从34.6掉到32.3，VideoMMMU从49.8掉到45.8——掉了2-4个点，相当显著。

### 3.2 Rule-based Reward Design

| Task Type | Reward Function |
|-----------|----------------|
| Multiple Choice | Binary match (最稳定) |
| Numerical QA | Exact match |
| OCR | Word Error Rate (WER) |
| Free-form QA | Average of ROUGE-1/2/L |
| Regression | $1 - \text{relative error}$ |

绝大多数数据是multiple choice / numerical，保证reward signal精确可靠。少量free-form/OCR/regression是为了泛化能力。

## 4. Training Pipeline

### Stage 1: SFT Cold Start
- Dataset: Video-R1-CoT-165k
- 用Qwen2.5-VL-72B-Instruct生成CoT annotations
- 1 epoch, ~40小时
- 产物: Qwen2.5-VL-7B-SFT

### Stage 2: RL with T-GRPO
- Dataset: Video-R1-260k (image-video混合)
- Optimizer: Adam, lr=1e-6
- KL coefficient $\beta$=0.04
- weight decay=0.01, max grad norm=5
- max response length=768 tokens
- Ordered group G=8, Shuffled group $\tilde{G}$=4
- 8×A100 80GB, ~15小时 for 1k steps
- 产物: Video-R1-7B

**只有1k steps就能取得显著提升**——这是R1范式的一个惊人特性。DeepSeek-R1原文也提到，rule-based RL的sample efficiency远高于传统RLHF。

Appendix A.1扩展到10k steps后，多数benchmark还有进一步提升 (Table 3)，说明1k远未收敛。

## 5. 关键实验结果分析

### 5.1 Main Results (Table 1)

| Model | Frames | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|-------|--------|-----------|-----------|------|---------|-------------|----------|
| GPT-4o | - | 34.0 | 61.2 | 75.4 | - | 71.9 | - |
| Qwen2.5-VL-7B (CoT) | 16 | 27.7 | 47.8 | 59.2 | 57.4 | 72.2 | 53.1 |
| Qwen2.5-VL-7B-SFT | 16 | 31.8 | 47.4 | 61.3 | 59.4 | 69.2 | 52.8 |
| **Video-R1-7B** | 16 | 34.6 | 49.8 | 64.2 | 62.7 | 72.6 | 57.4 |
| **Video-R1-7B** | 64 | **37.1** | 52.4 | 63.8 | 64.8 | 73.2 | 61.4 |

几个关键观察：

1. **Video-R1-7B 64 frames在VSI-Bench上达到37.1%，超过GPT-4o的34.0%**。这是个strong result，VSI-Bench [https://arxiv.org/abs/2412.14171] 是测试spatial reasoning的，需要模型从多帧中构建3D空间记忆——是真正的reasoning任务，不是pattern matching。

2. **SFT不一定能提升**：Qwen2.5-VL-7B-SFT在VideoMME上反而比base model低 (52.8 vs 53.1)。但RL后Video-R1提升到57.4+。这印证了 "SFT memorizes, RL generalizes" [https://arxiv.org/abs/2501.17161] 的现象——SFT容易过拟合到training distribution的format，RL则真正学到task-solving ability。

3. **更多帧 = 更好reasoning**: 16→32→64 frames几乎在所有benchmark上都单调提升。VideoMME从57.4→59.3→61.4，提升4个点。这暗示long-context video reasoning是未来重要方向。

### 5.2 Ablation (Table 2)

| Variant | VSI-Bench | VideoMMMU | MMVU | MVBench |
|---------|-----------|-----------|------|---------|
| Video-R1-7B-wo-image | 32.3 | 45.8 | 60.6 | 60.9 |
| Video-R1-7B-wo-temporal | 32.7 | 48.3 | 62.1 | 61.1 |
| Video-R1-7B-zero | 31.8 | 49.5 | 63.8 | 60.4 |
| **Video-R1-7B** | **34.6** | **49.8** | **64.2** | **62.7** |

- **wo-image**: 去掉image数据，VSI-Bench掉2.3点，证明image data确实在bootstrapping reasoning
- **wo-temporal**: 把T-GRPO换成原版GRPO，VSI-Bench掉1.9点——T-GRPO的贡献明确
- **zero**: 跳过SFT直接RL，VSI-Bench掉2.8点——SFT cold start的必要性

### 5.3 Temporal Reasoning Percentage (Figure 6)

这是T-GRPO最直接的validation: 用Qwen2.5-VL-72B来evaluate每个response是否"使用了temporal reasoning"。

- Video-R1 (T-GRPO): 75.0% responses包含temporal reasoning
- Video-R1-wo-temporal (原版GRPO): 60.2%

差了15个点，且这是用独立的strong model (72B)来evaluate的，比较客观。说明T-GRPO确实push model去使用temporal cues，而不仅是表面提升benchmark分数。

### 5.4 Training Curves (Figure 5)

- (a) Accuracy reward: 单调上升，模型持续学到的correct answer
- (b) Temporal reward $r_t$: 上升，说明越来越多questions满足 $p \geq \tilde{p}$ 条件
- (c) Response length: **先降后升再稳定**。作者猜测这是transition phase——模型先抛弃SFT的rigid reasoning style，然后找到新的、更长的reasoning policy。这个现象和DeepSeek-R1里观察到的相似。

## 6. Aha Moment (Section 3.4)

Video-R1能产生self-reflective behavior——模型会重新检查自己的理解，特别是在temporal cues模糊或多步推理时。Figure 4给了一个VSI-Bench上的例子，模型在推理过程中停下来重新审视video evidence。

这是R1范式最迷人的emergent property之一。DeepSeek-R1原文里用"aha moment"形容模型突然意识到自己推理错误的瞬间。Video-R1把这个现象带到了video domain。

## 7. 联系与联想

### 7.1 与Video-UTR的关系 [https://arxiv.org/abs/2502.12081]
Video-UTR (Unhackable Temporal Rewarding) 和T-GRPO都解决了同一个问题：temporal shortcut。但方法不同:
- Video-UTR: 通过对比单帧和多帧input的prediction consistency来design reward
- T-GRPO: 通过ordered vs shuffled frame的contrastive

T-GRPO更直接，因为shuffled sequence本身就是"作弊检测"——如果shuffled后还能答对，说明没用temporal info。

### 7.2 与STAR-R1的关系 [https://arxiv.org/abs/2502.14768]
STAR-R1做spatial transformation reasoning，也是R1范式扩展到多模态reasoning。说明R1 paradigm在多模态的扩展是2025年的重要趋势。

### 7.3 与MME-Reasoning的关系 [https://arxiv.org/abs/2505.21327]
MME-Reasoning是Co-author Kaituo Feng的另一个工作，做comprehensive logical reasoning benchmark for MLLMs，Video-R1也用了它。

### 7.4 与Critique-GRPO [https://arxiv.org/abs/2506.03106]
也是同一group的工作，把natural language critique加入GRPO，进一步提升reasoning quality。

### 7.5 训练cost的联想
8×A100 80GB, SFT 40小时 + RL 15小时 = 总共约440 GPU-hours。这个cost相当低——7B model能在中等scale compute上跑出超GPT-4o的spatial reasoning能力，说明R1范式极其sample efficient。

## 8. 局限性与未来方向

作者诚实地列出limitations:
1. **Frame数限制**: 训练只用16 frames，难以处理long-range temporal dependency
2. **Computational overhead**: T-GRPO需要double inference (ordered + shuffled)，可以vLLM加速
3. **固定length reward**: $l_{\min}=320, l_{\max}=512$对所有问题一刀切，理想情况应该adaptive
4. **Image-to-video迁移机制粗糙**: 只是混合数据，没有principled transfer mechanism
5. **Rule-based reward的局限**: 需要verifiable answer，无法处理open-ended reasoning tasks

未来方向最exciting的是**Generalist Video Reward Model**——一个能跨各种video reasoning task提供consistent reward的model，类似RLHF里的reward model但专门为video reasoning设计。这会解锁更广泛的video RL training。

## 9. 给Karpathy的几个Intuition Takeaways

1. **Contrastive reward是RL的一个underexplored design space**: T-GRPO本质是"用shuffled input作为negative sample"，这种self-supervised contrastive思想在RL里很有潜力。

2. **Image reasoning data作为video reasoning的"pretraining"**: 这个insight可以推广——任何需要新增modality时，先用已有modality的reasoning data建立"reasoning骨架"，再迁移。

3. **Length reward的必要性**: 暗示RL training如果不加constraint，model会collapse到短回答。这是RLHF的一个general issue。

4. **Aha moment的emergence**: 这种self-reflection行为的emergence不需要explicit supervision，rule-based RL + enough exploration就能实现。这是R1范式的deep insight。

5. **Video reasoning远未饱和**: 7B model + 1k RL steps就能超GPT-4o，说明video reasoning benchmark还有很大提升空间，特别是真正需要temporal modeling的任务。

## References

- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- VSI-Bench: https://arxiv.org/abs/2412.14171
- VideoMMMU: https://arxiv.org/abs/2501.13826
- MMVU: https://arxiv.org/abs/2501.12380
- MVBench: https://arxiv.org/abs/2310.00602 (Li et al.)
- TempCompass: https://arxiv.org/abs/2403.00476
- VideoMME: https://arxiv.org/abs/2405.21075
- Kimi k1.5: https://arxiv.org/abs/2501.12599
- Skywork R1V: https://arxiv.org/abs/2501.05993 (Peng et al.)
- Video-UTR: https://arxiv.org/abs/2502.12081
- Vision-R1: https://arxiv.org/abs/2503.06749
- Open Reasoner Zero: https://arxiv.org/abs/2503.24290
- SFT memorizes, RL generalizes: https://arxiv.org/abs/2501.17161
- Video-R1 GitHub: https://github.com/tulerfeng/Video-R1
- STAR-R1: https://arxiv.org/abs/2505.15804
- MME-Reasoning: https://arxiv.org/abs/2505.21327
- Critique-GRPO: https://arxiv.org/abs/2506.03106

---

Andrej, 这篇paper的核心创新是T-GRPO的contrastive design——把"作弊检测"直接encode进reward function。这是一个非常elegant的idea，值得你在后续工作中思考：**对任何inference shortcut，我们都可以通过construct adversarial input来detect it，然后把detection result作为RL reward**。这个思想可以推广到很多其他reasoning shortcut问题。

如果想深入build intuition，我建议重点思考两点：
1. 为什么 $r_t$ 只加在correct responses上？这和DPO里的preference pair设计有什么联系？
2. Shuffled group不参与policy update，但它的存在改变了ordered group的advantage distribution——这种"auxiliary group"的设计在RL里是一个有意思的pattern。
