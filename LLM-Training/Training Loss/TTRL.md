---
source_pdf: TTRL.pdf
paper_sha256: 3828cc886a0133749fe857dc12fff190f76c17dd7421dccfca01c96ec1dd5882
processed_at: '2026-08-12T18:20:28-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 TTRL

## 一句话说清楚

**模型考试的时候，不用标准答案，自己考自己，居然分数还能涨。**

---

## 问题是什么

现在训练 reasoning model，比如 DeepSeek-R1、o1，都得有标准答案。数学题你得知道正确答案是 42，代码题你得能跑通，才能算 reward，才能做 RL。

但现实是，真正难的问题，没人知道答案。ARC-AGI-2 这种 benchmark，o3 只能做 4%，谁给你标答案？

所以问题来了：**没答案，能不能还做 RL？**

---

## 他们怎么干的

很简单，三步：

1. **一道题做 64 遍**，每次采样一个不同的答案
2. **看哪个答案出现最多**，假装它是对的（majority voting）
3. **用这个假答案当标准**，匹配的给 reward=1，不匹配的给 reward=0，然后跑 GRPO

就这。没有 label，没有 verifier，没有 reward model，纯靠模型自己跟自己的 consensus 学。

---

## 为什么听起来不靠谱但实际 work

直觉上，模型很弱的时候，做 64 遍选出来的多数答案也是错的，那 reward 不全是噪声吗？

对，label 确实经常错。AIME 2024 上 label accuracy 只有 37%。

**但 reward accuracy 是 92%。**

这就是 paper 里最骚的点，叫 "Lucky Hit"。

举个例子。正确答案是 5，模型很弱，64 次采样出来的答案乱七八糟：3 出了 10 次，4 出了 8 次，7 出了 6 次，2 出了 5 次，等等。majority 选了 3（错的）。

现在算 reward：
- 答案是 3 的那些 → reward=1（这其实是错的 reward，因为 3 本身就是错的）
- 答案是 4、7、2 的那些 → reward=0（这是**正确的** reward，因为它们确实错了）

所以 64 个里面，10 个拿到了错误的正 reward，54 个拿到了正确的负 reward。**大部分 reward 是对的。**

为什么？因为弱模型的 prediction 很散，wrong answer 之间互相不重复，所以大部分 wrong prediction 都不会撞上那个错的 pseudo-label，自动获得正确的负 reward。

**越弱的模型，输出越散，Lucky Hit 概率越高，reward 反而越准。** 这太反直觉了。

---

## 最让人意外的结果

传统 self-training 的逻辑是：用 majority voting 选答案，拿去 SFT。这种做法的天花板就是 majority voting 的准确率，因为你在拟合一个固定信号。

TTRL 用 RL 之后，**pass@1 超过了初始模型的 maj@16，涨了 20 多分。**

意思就是：模型自己给自己当老师，老师只有 37 分水平，但学生最后考了 80 分。

怎么做到的？两个原因：

1. **RL 和 SFT 本质不同**。SFT 是死记硬背 majority answer 的 token sequence，上限就是 majority 准确率。RL 只把 reward 当方向信号，模型可以探索更广的策略空间，reward noisy 一点也能学（Chu et al. 2025 的 "SFT memorizes, RL generalizes" 说得就是这个）

2. **Online learning 的正反馈**。模型变强一点 → voting 更准一点 → reward 更准一点 → 模型再变强一点。Figure 6 里 pass@1 和 maj@16 两条线一起往上涨，不是零和博弈。

---

## 具体涨了多少

Qwen2.5-Math-7B：

- AIME 2024：12.9 → 40.2（**+211%**）
- AMC：35.6 → 68.1（+91%）
- MATH-500：46.7 → 83.4（+79%）
- GPQA：29.1 → 27.7（-5%）

最后一个 GPQA 是 graduate-level 科学，不是数学，模型 prior 弱，TTRL 反而掉了。这告诉我们 **TTRL 不能凭空创造能力，得有 prior 才行**。

甚至已经经过大量 post-training 的 DeepSeek-R1-LLaMA-8B，TTRL 还能再涨 17 分（AIME 51.7 → 69.2）。说明 self-evolution 没到天花板。

---

## 什么时候会挂

两个坑：

1. **Hyperparameter 敏感**。AIME 2024 上 temperature 0.6 挂，1.0 才行。难的数据需要高 entropy 来 explore。Episodes 也得调，数据越难越多轮。RL 调参本来就难，TTRL 因为 reward 有噪声，更敏感。

2. **Prior 不够**。把 MATH-500 按难度分 5 级，分别做 TTRL：
   - Level 1（最简单）：+175%
   - Level 5（最难）：+75%
   
   越难涨越少。模型对某个 domain 完全没概念的话，voting 出来的全是噪声，Lucky Hit 也救不了。

---

## 我觉得最有意思的几个 insight

1. **Majority voting 不只是 inference 技巧**。以前 self-consistency 只是在 inference 时选答案，weights 不变。TTRL 把 voting 的结果变成 reward signal，weights 更新了，knowledge 持久化了。同一个 voting 操作，用法的差别产生质变。

2. **弱模型的散乱输出是 feature 不是 bug**。传统觉得模型弱 → 输出散 → 不好。但 TTRL 里，散乱的输出恰好让 Lucky Hit 概率高，reward 反而准。这让我想到 boosting：weak learner 组合起来能变 strong，TTRL 更进一步，用这个组合信号反过来 train weak learner 自己。

3. **这本质是 self-play**。AlphaGo 自己跟自己下棋变强，TTRL 自己跟自己的 consensus 对齐变强。"对手"不是另一个 policy，是 majority voting 这个 aggregation operator。

4. **Test-time compute 能转化为持久的能力提升**。Snell et al. 2024 说 test-time compute 比 pretraining compute 更高效。TTRL 更进一步：test-time compute 不只是临时推理，还能 update 参数，复利式增长。这可能是通往 self-evolving AI 的一条 cheap path。

---

## 一个我的担忧

TTRL 在 test set 上训练，不管 reward 多 noisy，终究是利用了 test distribution 的 information。虽然 Figure 4 的 OOD 实验是 positive 的，但只测了几个相近的 benchmark。真正跨 domain 的 OOD（数学→代码，数学→agent）能不能 generalize，GPQA 那个 -5% 已经是个 warning signal 了。

而且这和 standard ML 评估范式的冲突还在。你可以说 "我不是 cheating，我是 self-supervised"，但如果每次 eval 都顺便 train 了，怎么定义 fair evaluation？这是个需要 community 一起想的问题。

---

## 和你的直觉对照

Andrej，你之前在多个场合提过 "RL is where the magic happens" 和 "model can self-improve through experience"。这篇 paper 给了一个具体的 instantiation：

- 不需要 human label
- 不需要 external reward model  
- 不需要 verifier
- 只要模型有 prior + 大量 unlabeled data + compute

就能 self-improve。这和 Silver & Sutton 的 "Era of Experience" 愿景完全对齐。

我觉得最有意思的是 Lucky Hit 这个现象——**弱的时候反而 reward 准**。这意味着 TTRL 在模型最需要提升的早期阶段，reward signal 恰好最可靠。随着模型变强，label accuracy 上来了，Lucky Hit 的红利逐渐消失，但那时候 voting 本身已经准了。两个阶段无缝衔接。

这个 self-correcting 的 property，可能是 TTRL 真正 work 的深层原因。

---

# TTRL: Test-Time Reinforcement Learning 深度解析

Andrej, 这篇 paper 我读了挺久, 有点反直觉的地方值得细嚼。我用 build intuition 的方式给你拆解一下, 公式、机制、为什么 work 都讲透。

---

## 1. 背景: 这篇 paper 想解决什么

现在的 Large Reasoning Models (LRMs), 比如 DeepSeek-R1、OpenAI o1, 都需要 RL + ground-truth labels 来训练。问题是:
- 真实世界的复杂问题 (如 ARC-AGI-2, o3 只能解 4%) 没有 label
- 人工标注不可扩展
- Test-time 遇到 distribution shift 的难题时, 模型是 frozen 的

TTRL 的核心 question: **能否在 test-time, 没有任何 ground-truth label 的情况下, 用 RL 让模型自我进化?**

这呼应了 Silver & Sutton 在 2025 年提出的 "Era of Experience" 的愿景 (https://ama.google/research/era-of-experience/)。

---

## 2. 方法: 用 Majority Voting 当 Reward Proxy

### 2.1 Pipeline 概览

给定一个 prompt $x$ (没有 label), TTRL 做以下几步:

1. **Repeated Sampling**: 从当前 policy $\pi_\theta(y \mid x)$ 采样 $N$ 个 candidate outputs $\{y_1, y_2, \ldots, y_N\}$ (paper 里 $N=64$)
2. **Answer Extraction**: 用 extractor 把每个 $y_i$ 的 final answer 抽出来, 得到 $\hat{y}_i$
3. **Majority Voting**: 找出 $\{\hat{y}_i\}_{i=1}^N$ 中出现频次最高的 answer, 记为 $y^*$ (estimated pseudo-label)
4. **Reward Computation**: 对每个 $y_i$, rule-based reward

$$R(\hat{y}_i, y^*) = \begin{cases} 1, & \text{if } \hat{y}_i = y^* \\ 0, & \text{otherwise} \end{cases}$$

5. **RL Update**: 用 GRPO / PPO / PRIME 优化

$$\max_\theta \, \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ r(y, y^*) \big] \tag{1}$$

参数更新 (gradient ascent):

$$\theta \leftarrow \theta + \eta \, \nabla_\theta \, \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} [r(y, y^*)] \tag{2}$$

变量含义:
- $\theta$: policy model 参数
- $\eta$: learning rate (paper 里用 cosine schedule, peak $5 \times 10^{-7}$)
- $y^*$: majority voting 得到的 pseudo-label
- $r(y, y^*)$: rule-based reward (匹配则 1, 不匹配则 0)

### 2.2 为什么用 Majority Voting 而不是其他 aggregation?

我猜本质原因是:
- Majority voting 是 Test-Time Scaling (TTS) 里最 robust 的免费信号
- Self-consistency (Wang et al., 2022, https://arxiv.org/abs/2203.11171) 已经验证了: 当模型 sampling 多次, majority answer 往往是对的
- 这是 "free" 的, 不需要 external verifier 或 reward model

---

## 3. 核心 Counterintuitive 现象: "Lucky Hit"

这是 paper 最有意思的部分。直觉上, 如果模型很弱, majority voting 估出的 label 也是错的, 那 reward 不就全是噪声了吗?

但 paper 的实验发现: **AIME 2024 上, label accuracy 只有 37%, 但 reward accuracy 高达 92%**!

### 3.1 Lucky Hit 的机制

考虑一个数学题, ground truth 是 $C$。模型采样 64 次, 各种 wrong answer $A, B, D, E, \ldots$ 都有, 但很分散。Majority voting 选出来的 pseudo-label 假设是 $B$ (也是错的)。

现在对每个 prediction 计算 reward:
- 如果 prediction 是 $B$ (匹配 pseudo-label), reward = 1 (但这个 prediction 其实是错的, 这是 false positive)
- 如果 prediction 是 $A, D, E, \ldots$ (不匹配 $B$), reward = 0

关键 insight: 这些 $A, D, E$ 也是 wrong answers, 它们得到 reward=0 (negative), **这恰好是正确的 reward**! 因为它们确实错了。

Figure 10 那个 toy case 把这个讲得很清楚: 当 true label 是 5, 但 majority estimated 是 3, 模型实际 generate 出的是 3, 3, 4, 4, 7。Reward 计算时:
- 3 vs 3 → reward = 1 (false positive, 这是 noise)
- 4, 4, 7 vs 3 → reward = 0 (correct! 它们确实错)

所以只要 wrong predictions 之间高度分散, 不撞上 pseudo-label, 就会自动得到正确的负 reward。这就是 "Lucky Hit"。

### 3.2 数学直觉

设模型对某题的预测分布是 $p(\hat{y})$, ground truth 是 $y_{gt}$, majority 估计是 $y^*$。Reward accuracy 可以写成:

$$\text{Reward Acc} = P(\hat{y} = y^*) \cdot \mathbb{1}[y^* = y_{gt}] + P(\hat{y} \neq y^*) \cdot \mathbb{1}[y^* \neq y_{gt}]$$

当模型弱时:
- $P(\hat{y} = y^*)$ 较小 (因为输出分散, majority 占比低; paper 里 AIME 初始 majority 只占 16.6%)
- 所以即使 $y^* \neq y_{gt}$, 大部分 prediction 也会得到 reward=0 (correct negative)

paper 的实测: majority ratio 只有 16.6%, 但 reward accuracy 92%。即 84% 的 outputs 都"幸运地"撞上了正确的负 reward。

### 3.3 多输出 rollouts 的 robustness

单一 rollout 时, label 错就全错。但 64 个 outputs 让 reward 信号 dense 很多——即使 label 错了, 单个 rollout 内部还能"自救"很多 correct negative rewards。这是和 self-consistency at inference 的本质区别: voting 不仅用来 pick answer, 还用来 densify reward signal。

---

## 4. 让人震惊的实验结果

### 4.1 Surpass Upper Bound: 超 maj@n

传统 self-training (Huang et al., 2022, https://arxiv.org/abs/2210.11610) 用 majority voting 选 CoT 做 SFT, 性能上限就是初始模型的 maj@n。

但 TTRL 用 RL 后, **avg@16 (即 pass@1 的平均) 超过了初始模型的 maj@16**, Figure 6 上差了 20+ 分。

为什么? 我的理解:
- SFT 是 offline + 监督式的, 学到的是 majority answer 这个 specific token sequence, 上限是 maj@n
- RL 是 online + 探索式的, reward 只是 directional signal, 让模型探索更广的策略空间
- 模型在训练中变强 → voting label 越来越准 → reward 越来越准 → 模型更强 (self-reinforcing loop, Figure 6 双曲线一起上升)
- RL 对 reward noise 的 robustness 比 SFT 强 (Chu et al., 2025, https://arxiv.org/abs/2501.17171; Razin et al., 2025, https://arxiv.org/abs/2503.15477)

### 4.2 接近 RL(leakage) 上限

RL(leakage) 是直接用 test set 的 ground truth label 做 RL, 理论上是这个 test set 上的最强上限 (information leakage)。TTRL 居然能接近它 (Figure 8)!

这说明 majority voting 的 reward 信号虽然 noisy, 但在 RL 框架下足够 informative。

### 4.3 具体数字 (Qwen2.5-Math-7B)

| Benchmark | Backbone | +TTRL | Gain |
|-----------|----------|-------|------|
| AIME 2024 | 12.9 | 40.2 | +211.6% |
| AMC | 35.6 | 68.1 | +91.3% |
| MATH-500 | 46.7 | 83.4 | +78.6% |
| GPQA | 29.1 | 27.7 | -4.8% |
| Avg | 31.1 | 54.9 | +76.5% |

注意 GPQA 是 -1.4, 略降。GPQA 是 graduate-level science, 不是数学, 模型 prior 弱, 落到了 Q3 那个 "lack of prior knowledge" 的失败模式。

### 4.4 On LRMs (Figure 3)

即使像 Qwen3-8B (thinking mode)、DeepSeek-R1-LLaMA-8B 这种已经经过昂贵 post-training 的 LRMs, TTRL 还能再涨 ~10 分。这说明 self-evolution 没有饱和。

DeepSeek-R1-LLaMA-8B: AIME 从 51.7 → 69.2 (+17.5), AMC 从 81.6 → 88.9 (+7.3)。

### 4.5 Out-of-Distribution Generalization (Figure 4)

在 AMC 上做 TTRL, 再 evaluate 在 AIME / MATH-500 / GPQA, 性能也全面提升。说明 TTRL 不是 overfit test data, 而是获得 generalizable 的 reasoning 能力。

---

## 5. 三大失败模式 (Q3)

### 5.1 Hyperparameter Sensitivity

- **Temperature**: AIME 2024 上, $T=0.6$ 训练失败 (entropy 一直高), $T=1.0$ 训练成功。困难 benchmark 需要更高 entropy 来 explore, 利用 prior。这和 Open-Reasoner-Zero (https://arxiv.org/abs/2503.24290) 的发现一致
- **Episodes**: 数据小/难的需要更多 episodes。AIME 80, AMC 30, MATH-500 10

### 5.2 Prior Knowledge 不足

Table 3 在 MATH-500 上按 5 个 difficulty level 分别做 TTRL:
- Level 1: +175.3%
- Level 5: +75.3%

难度越高, 增益越小。说明 TTRL 不能凭空创造能力, 依赖 backbone 的 prior。这和 R-Zero 系列 (https://arxiv.org/abs/2503.18892) 里 "spurious rewards" 的发现呼应——base model 太弱时 RL 会 collapse。

---

## 6. 和相关工作对比

| 工作 | 数据需求 | 方法 | Reward 来源 |
|------|---------|------|------------|
| DeepSeek-R1 (https://arxiv.org/abs/2501.12948) | Labeled | GRPO | Rule-based + GT |
| Self-consistency (https://arxiv.org/abs/2203.11171) | Unlabeled | Inference only | N/A |
| STaR (Zelikman) | Labeled | SFT iteration | GT filter |
| Self-rewarding LM (https://arxiv.org/abs/2401.10020) | Unlabeled | DPO | LLM-as-judge |
| SPIN (https://arxiv.org/abs/2401.01335) | Unlabeled | DPO | Self-play |
| **TTRL** | **Unlabeled** | **Online RL** | **Majority voting** |

TTRL 的关键不同:
1. **Online RL** (GRPO/PPO) 而非 offline DPO/SFT, 能利用 evolving policy
2. **Majority voting** 而非 LLM-as-judge, 避免 reward hacking
3. 用 rule-based reward 而非 preference, 信号更 dense

---

## 7. 我的 Intuition & 联想

### 7.1 这本质是 "Self-Play against Consensus"

TTRL 让模型和自己的 majority consensus 对弈。这有点像 AlphaGo 的 self-play, 但 "对手" 不是另一个 policy, 而是 majority voting 这个 aggregation operator。模型的探索被 majority 这个 prior 约束, 又通过 RL 突破这个 prior (surpass maj@n)。

### 7.2 和 EM 算法的关系

我觉得 TTRL 隐含了一个 EM-like 结构:
- **E-step**: 用当前 policy $\pi_\theta$ 采样, 用 majority voting 估 latent label $y^*$ (相当于 E-step 估后验)
- **M-step**: 用 estimated $y^*$ 做 RL update, 最大化 expected reward (相当于 M-step 最大化 likelihood)

但和 EM 不同的是, TTRL 是 online 的, $y^*$ 随 $\theta$ 变化而变化, 形成动态的 self-improvement。

### 7.3 和 R-Zero / RFT 类工作的区别

R-Zero 类 (DeepSeek-R1, Open-Reasoner-Zero) 假设有 verifiable reward (数学有标准答案, 代码能跑)。TTRL 把这个限制去掉了: 即使没有 verifier, 只要模型自己能形成 consensus, 就能 self-supervise。

但代价是: TTRL 在 GPQA 这种 model prior 弱的 domain 上不行。这是 paper 自己承认的 limitation。

### 7.4 和 Self-Consistency CoT 的区别

Self-consistency 在 inference 时 voting 选答案, 模型权重不变。TTRL 把 voting 的结果当作 reward, 改模型权重。所以:
- Self-consistency: 用 compute 换 accuracy, 但 compute 不能 reuse
- TTRL: 用 compute 换参数, knowledge 持久化, 还能 transfer 到 OOD

paper Figure 4 证明 TTRL 的 gain 是 generalizable 的, 不是 overfit。

### 7.5 "Lucky Hit" 和 Weak Supervision 的联系

这让我想起 Snorkel (https://arxiv.org/abs/1605.07723) 的 weak supervision 理论: 多个 noisy labeler 的 majority 可以很准。但 TTRL 是单一 model 的多次 sampling, noise 来源是 sampling randomness, 不是多个独立 labeler。

更近的联系是 **Cohen 的 boosting** 思想: weak learner (单次 prediction) 通过加权组合 (majority voting) 能成 strong learner。TTRL 进一步: 用这个 strong-but-imperfect signal 当 reward, 再让 weak learner 变 strong。

### 7.6 一个潜在的 Theoretical Puzzle

paper 在 Future Works 里提到想做 convergence analysis。我觉得 key question 是:

**TTRL 能 surpass maj@n 的本质原因是什么?**

我的猜测: maj@n 是 model 在 sample space 的 mode, 但 model 的 expected reward (pass@1) 可以通过 sharpening distribution 而提高, 即使 mode 不变。RL 通过降低 entropy, 让 model 更确信 majority answer, 提升 pass@1 但 maj@n 几乎不变。再加上 online 的 voting label 也在变好, 形成两重提升。

但 paper Figure 6 显示 maj@16 也在涨, 说明 mode 本身也在变。所以是分布整体右移, 不是单纯 sharpening。这点值得细究。

### 7.7 Agentic / Streaming 场景的延伸

paper 在 Future Work 提到 "Online Learning with Streaming Data"。想象一个部署中的 agent, 每次遇到新 task 就采样 64 次, voting 当 reward, RL update。这就实现了 Silver & Sutton 的 "Era of Experience"——agent 持续从自己的 experience stream 学习, 不需要人类 label。

但 stability 是个大问题。paper Q3 已经说了 hyperparameter 敏感, 在 streaming 场景下, 数据分布漂移会让 hyperparameter 调优更难。Catastrophic forgetting 也是潜在问题。

### 7.8 一个我自己的怀疑

TTRL 在测试集上训练, 这本质上还是 "training on the test set", 只是 self-supervised。虽然 reward 信号 noisy, 但终究是利用 test distribution 的 information。这和 standard ML 评估范式有冲突。

paper 在 4.1 说 "RL(leakage) represents the most efficient way to improve performance on the particular dataset"——确实, 但这真的是我们想要的吗? TTRL 的 OOD generalization (Figure 4) 是个 positive signal, 但 paper 只测了几个 benchmark, 真正的 OOD (比如数学→代码, 数学→科学) 是否还能 generalize? GPQA 那个 -1.4 已经是个 warning。

### 7.9 和 Constitutional AI / RLAIF 的对比

Anthropic 的 RLAIF (https://arxiv.org/abs/2204.05862) 用 LLM-as-judge 替代 human feedback, 但还是需要 preference data。TTRL 更激进: 直接用 model 自己的 majority 当 supervision。这避免了 LLM-as-judge 的 reward hacking (model 可能学会生成 "judge 喜欢的" 而非 "正确的")。

但 TTRL 也有自己的 hacking 风险: 模型可能 collapse 到 always output majority answer, 退化为 mode collapse。Paper 提到 entropy 监控, 但没深入分析。

### 7.10 Scaling Law of Self-Evolution

paper 提到 "TTRL naturally scales" (1.5B → 7B → 32B 越大越好)。这给出了一个新 scaling 维度: **self-evolution 的 scaling law**。Snell et al. 2024 (https://arxiv.org/abs/2408.03314) 说 test-time compute 比 pretraining compute 更 efficient。TTRL 说: test-time compute 还能转化为持久的参数改进, 复利式增长。

这可能是通往 AGI 的一个 cheap path: 不需要新 human data, 只要 model 有 prior + 大量 unlabeled data + compute, 就能 self-improve。

---

## 8. 总结 (Build Intuition)

一句话: **TTRL = Test-Time Self-Play RL with Majority Voting as Reward Proxy**。

核心机制:
1. **Voting as label estimation**: majority 估 latent label (E-step-like)
2. **Lucky Hit makes noisy reward work**: 弱模型时, 散乱 wrong predictions 自动获得 correct negative reward
3. **Online RL breaks maj@n ceiling**: voting label 随模型变强而变准, 正反馈循环
4. **Prior is the bottleneck**: 模型必须有足够 prior 才能形成有意义的 consensus

最反直觉的两点:
- **越弱的模型, reward 越准** (因为 predictions 越散乱, Lucky Hit 概率越高)
- **TTRL 能超过它自己的 supervision signal** (因为 RL explore vs SFT memorize 的本质差异)

我认为这是 post-DeepSeek-R1 时代一个真正有 insight 的工作, 把 TTS 和 RL 统一了起来 (Table 5 的 taxonomy)。值得 follow-up 的方向:
- Streaming version (continual TTRL)
- 非数学 domain (代码、agent) 的 TTRL
- Convergence analysis (为什么能超 maj@n)
- Mode collapse / reward hacking 的 theoretical bound
- 和 process reward model 的结合 (vote on steps 而非 final answer)

希望这个拆解对你的 intuition 有帮助, Andrej。如果某个具体点你想再深挖 (比如 Lucky Hit 的 probability bound, 或 online EM 的视角), 告诉我, 我可以再展开。
