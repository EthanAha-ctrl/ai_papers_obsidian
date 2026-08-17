---
source_pdf: Intrinsic Credit Assignment for Long Horizon Interaction.pdf
paper_sha256: 9d5ccc85bee01a2b4e01e968a4c2351f60598909408aea1b275a42e4e54f0e1a
processed_at: '2026-08-05T10:22:23-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 paper

## 一句话总结

模型心里其实知道自己离答案有多近,把这个"心里知道的变化"拿来当训练信号,模型就学会怎么问好问题了。

---

## 问题出在哪

你给 LLM 一个数学题,它能做。但你让它当侦探问问题破案、当客服诊断故障、当朋友聊天搞清楚你想要什么——它就很蠢,要么乱问一通,要么死磕一个方向。

根本原因是 **web data 不教这个**。网上到处是 question-answer pair,但没人记录"提问者当时心里在想什么、为什么觉得这个问题有用"。一个问题对 A 有用,对 B 可能完全是废话,因为两个人的知识不一样。所以光靠 pretraining 学不会"会问问题"这件事。

RL 能不能救?标准做法是看最终结果——猜对了给 reward,猜错了给 0。但一条 20 turn 的 trajectory 里,到底哪个问题问得好、哪个问得烂?sparse reward 分不清,就像你考试只看总分,不知道哪道题做对了哪道做错了,没法改进。训一个大 value network 当 critic 来做 credit assignment?太贵了,LLM 这么大训不起。

---

## 核心观察

作者发现一个很简单的事实:每轮对话之后,你拿 model 问一句"你觉得答案是 X 吗",它给 X 这个 token 的 probability 其实会变化。

比如你问 "Is it an animal?" 得到 "Yes",model 对 "dog"、"cat"、"elephant" 这些词的 log-prob 全都会涨。问得越好,涨得越快。这个概率变化就是 **model 心里"我离答案更近了"的量化指标**。

而且这个信号几乎是免费的——你本来就要跑 forward pass 来 sample 下一个问题,再多跑一次 forward 把 target word 喂进去取 log-prob 就行,不用训额外的 network。

---

## 怎么用这个信号

每轮算一个 $\Delta\text{Belief} = \log b_t - \log b_{t-1}$,就是"这一轮让 model 对 ground-truth 的信心涨了多少"。涨了就给奖励,跌了不惩罚(因为探索阶段暂时走偏是正常的)。

然后跟原来的 sparse reward 混在一起:

$$r_t = r^{\text{最终结果}} + 0.1 \times \max(\Delta\text{Belief}_t, 0) + \text{效率惩罚}$$

关键改动是 **GRPO 的 advantage 从 trajectory 级别降到 turn 级别**。原来 16 条 trajectory 比谁的总分高,现在每个 turn 单独比——第 3 turn 的 16 个候选问题之间比谁让 belief 涨得多。这样每个问题都能得到精准的 credit,不会被同一条 trajectory 里其他 turn 的好坏淹没。

---

## 为什么 work,用类比讲

想象你在玩 20 Questions,猜一个 secret word。

**trajectory-level GRPO** 就像:你玩完一整局,赢了拿 1 分输了拿 0 分,然后回头想"我这 20 个问题里哪个问得好"。你根本想不清,因为 signal 太粗了。

**ΔBelief-RL** 就像:每问完一个问题,旁边有个神仙告诉你"你现在离答案更近了 30%"。你立刻就知道这个问题问得好,下次多问类似的。这个"神仙"就是 model 自己的 log-prob 变化。

ReLU clip 的道理也简单:开局问 "Is it alive?" 这种大范围问题是好策略,哪怕答案是 No 让某些候选概率暴跌。如果你惩罚"信心下降",模型就不敢问 broad question 了,会变成一上来就瞎猜具体答案。所以只奖励"信心上升",不惩罚"下降",让模型放心 explore。

---

## 结果怎么样

几个数字说话:

1. **同规模碾压 baseline**:1.7B 和 4B 上,CIA 比 STARPO(标准 multi-turn GRPO)绝对涨 8-9 个点
2. **小模型干翻大模型**:CIA-1.7B 击败 DeepSeek-V3.2(670B)10+ 个点。会问问题这件事,不是靠堆参数就能解决的,得专门训
3. **训练时只见过 20 turn,测试时给 50 turn 性能继续涨**:说明学到的是通用的"怎么缩小假设空间"的元能力,不是 overfit 到 20 turn 的套路
4. **迁移到没见过的任务**:在 20Qs 上训的,直接拿去测 customer service、user personalization、murder mystery,都涨点。CIA 当客服会在 10 轮内锁定 engine mount 故障,STARPO 20 轮还在纠结车主开的到底是 Honda 还是 Toyota

---

## 最值得记住的三点

**第一,LLM 自己就是免费的 critic。** 任何有 ground-truth 的任务,拿 policy 对 target 的 conditional log-prob 变化当 per-step reward,不用训 value network、不用标 step-level 数据、不用 LLM-as-judge。这个思路可以推广到很多 long-horizon 场景。

**第二,long-horizon RL 必须做 turn-level credit assignment。** 一个 scalar advantage 覆盖整条 trajectory 是不够的,variance 太大。把 group-relative baseline 下沉到每个 turn,每个 action 都有自己的 counterfactual 对照组,学习 signal 立刻清晰起来。

**第三,information-seeking 是可训练的 capability,不是 scaling 自动涌现的。** Web data 缺这一块,pretraining 补不上,但用 intrinsic belief reward 做 RL 可以专门教。1.7B 小模型经过这种训练能超过 670B 大模型,说明这条路效率很高。

---

## 局限也得说

- 只能在训练时有 ground-truth 的任务上用,完全 open-ended 的任务(比如写诗)不直接适用
- 小模型(1.7B)的 belief signal 本身比较 noisy,效果不如 4B 明显,因为小模型内部 world model 不够 calibrated
- 理论上有 reward hacking 风险:模型可能学到生成某些 token pattern 让 log-prob 短暂跳一下,而非真正推进任务。论文里靠 λ 调小 + ReLU clip + 最终 reward 联合约束压住了,但更复杂任务上要小心

---

论文地址:https://bethgelab.github.io/delta-belief-rl

一句话:**别花钱训 critic 了,你的 model 自己就知道自己离答案有多近,把它说出来当 reward 就行。**

---

# Intrinsic Credit Assignment for Long Horizon Interaction 讲解

## Core Problem & Motivation

当前 LLM agents 在 fully specified tasks(数学、coding)上很强,但对 under-specified、open-ended、需要 information-seeking 的任务很差。Web pre-training data 的根本缺陷:它记录了 user 提出的问题和 answer,但**不 capture** 导致 user 提出该问题的 internal belief state / knowledge gap。一个问题对 agent A 有用,对 agent B 可能完全没用——utility 是 belief-dependent 的。

Long-horizon multi-turn RL 的痛点是 credit assignment:轨迹长度 $N$ 上,哪个 action 推动了 success?标准做法是 train 一个 learned critic(value function)或者 PRM,但对 large-scale LLM 而言成本太高,所以大家 fall back 到 sparse outcome reward(只看 final 是否猜对),希望 GRPO 这种 trajectory-level baseline 能间接推 policy 走向全局最优。在 single-turn 上还能勉强 work,但 partial-information + 需要主动 explore 的场景里 sparse reward 走不通。

这篇 paper 的核心 insight:**agent 自己的 token probability distribution 就是一个免费的 critic**。每轮 interaction 后,target concept $y$ 在 agent posterior 里的概率变化直接反映该 action 的信息价值,无需单独训练任何 reward model。

参考链接:
- 论文页面:https://bethgelab.github.io/delta-belief-rl
- arXiv 版本(2025):https://arxiv.org/abs/2510.23746
- StarPO(对比 baseline)Wang et al. 2025:https://openreview.net/forum?id=UeB3Hdrhda
- STaR-GATE 数据集(Andukuri et al. 2024):https://openreview.net/forum?id=CrzAj0kZjR

---

## Method: ΔBelief-RL

### Step 1 — Eliciting agent beliefs

对每个 candidate target concept $y_i \in \mathcal{V}$,用一个 elicitation prompt $e_i$ 把 agent 内部 belief 暴露出来:

$$b_t = p_\theta(y_i \mid h_t, e_i)$$

变量解释:
- $b_t \in [0,1]$:turn $t$ 时 agent 对 ground-truth $y_i$ 分配的概率
- $\theta$:当前 policy 参数(SFT 后的 Qwen3-1.7B / 4B)
- $y_i$:候选 secret word(ground-truth 仅在训练时可知)
- $h_t = \{(a_1,o_1), \dots, (a_{t-1},o_{t-1})\}$:截至 turn $t$ 的 interaction history
- $e_i$:elicitation template,20Qs 里固定为 "Is the secret \<target\>"

这里 $b_t$ 不是 agent 真正输出的 token 概率,而是**在给定 history 的条件下,模型给 target concept 这一 token sequence 分配的 conditional likelihood**。实践中用 constrained decoding 把 target $y_i$ 强行 feed 进去取 log-prob。

### Step 2 — ΔBelief: belief change as intrinsic reward

$$\Delta \mathrm{Belief}_t = \log \frac{b_t}{b_{t-1}} = \log b_t - \log b_{t-1} \tag{1}$$

变量解释:
- $\Delta \mathrm{Belief}_t$:turn $t$ 的 belief 增量(可正可负)
- $b_t, b_{t-1}$:相邻两轮 agent 对 ground-truth 概率(取 log 防止 underflow)
- 用 log-ratio 形式而非 raw ratio,保证数值稳定且对 small probabilities 友好

**直觉**:这一项直接度量"这一轮的 $(a_t, o_t)$ 让 agent 的世界观向 ground-truth 移动了多少"。它是一个 information-gain 风格的信号,但**不需要预先 train 一个 critic**,因为 $b_t$ 直接从 policy 自身读取。

### Step 3 — Per-turn composite reward

$$r_t = \underbrace{r^{\mathrm{eog}}}_{\text{trajectory}} + \underbrace{\lambda \max(\Delta \mathrm{Belief}_t, 0)}_{\text{intrinsic exploration}} + \underbrace{r_p}_{\text{efficiency}} \tag{3}$$

各 component 含义:
- $r^{\mathrm{eog}}$:end-of-game verifiable reward,只在最终猜对时给($r^{\mathrm{eog}} = 2.0$ scale),其它 turn 都是 0
- $\lambda$:scaling hyperparameter,消融最优 $\lambda = 0.1$($\lambda \geq 0.5$ 训练 collapse,因为 ΔBelief unbounded 会 dominate 固定范围的 eog reward)
- $\max(\cdot, 0)$:ReLU clip,只奖励"使 belief 上升"的 action,**不惩罚临时下降**——这点很关键,因为 exploration 阶段 belief 短暂下降是正常的(询问 broad question 会暂时 dilute posterior)
- $r_p$:penalty bundle,包括 format error $r^{\text{inv}} = -5.0$、repeated question $r^{\text{rep}} = -1.0$、per-turn efficiency penalty $r^{\text{traj}} = -0.05$

Table 11(b) 的 scale 配置:
| Signal | Scale |
|---|---|
| $r^{\text{eog}}$ | 2.00 |
| $r^{\text{traj}}$ | -0.05 |
| $r^{\text{rep}}$ | -1.00 |
| $r^{\text{inv}}$ | -5.00 |

### Step 4 — Turn-wise GRPO advantage

标准 GRPO 在 trajectory level 算 advantage:

$$\widehat{A}^i = \frac{r(\tau^i) - \mathrm{mean}(\{r(\tau^i)\}_{i=1}^G)}{\mathrm{std}(\{r(\tau^i)\}_{i=1}^G)}$$

对整条 trajectory 用同一标量——long-horizon 下 variance 巨大、credit assignment 模糊。ΔBelief-RL 改成 turn-level:

$$\widehat{A}_t^i = \frac{r_t^i - \mathrm{mean}(\{r_t^i\}_{i=1}^G)}{\mathrm{std}(\{r_t^i\}_{i=1}^G)} \tag{4}$$

变量解释:
- $\widehat{A}_t^i$:第 $i$ 个样本在第 $t$ turn 的 advantage
- $r_t^i$:第 $i$ 个样本第 $t$ turn 的 composite reward(来自公式 3)
- $G=16$:group size,每个 prompt 采样 16 条 trajectory
- mean、std:在 group 内对**同一 turn $t$** 的所有样本计算

这个 advantage 序列 $(\widehat{A}_1^i, \dots, \widehat{A}_N^i)$ 被赋给第 $t$ turn 生成的所有 token(注意要 mask 掉 environment simulator 产生的 token,只对 agent token 反传)。

**关键直觉**:每个 turn 都有一个独立的 baseline($\mathrm{mean}(r_t^i)$),所以 advantage 是在"同一 turn 的不同 candidate action"之间相对比较,而非整条 trajectory 之间比较。这等价于在 turn 级别做 counterfactual baseline,大幅降低 variance。

---

## Validating the ΔBelief Signal

### 实验 1: Beliefs 是否反映 interactive progress?

Figure 2 用 DeepSeek-v3.2 生成 off-policy trajectory,然后 measure Qwen3-1.7B / 4B 在这些 trajectory 上的 $b_t$。结论:
- 成功 trajectory 的 $b_t$ 单调上升且增长更快
- 失败 trajectory 的 $b_t$ 几乎平的
- belief 增长 rate 与 final outcome 强相关

这说明 LLM 内部 world model 确实在 integrate evidence,belief update 是 calibrated 的——这是 ΔBelief reward 可行的**前提条件**。

### 实验 2: 优化 belief updates 是否真能提升 task success?

Best-of-8 sampling:每 turn 采样 $n=8$ 个 candidate question,模拟每个问题的 environment response,选 $\arg\max_k \Delta\mathrm{Belief}(a_{t,k})$。

$$a_t \gets \arg\max_{k \in \{1,\dots,n\}} \Delta\mathrm{Belief}(a_{t,k}) \tag{2}$$

Figure 3 显示这个 lookahead 启发式在所有 model scale 上都显著优于 baseline sampling,证明 ΔBelief 是一个有效的 exploration heuristic。注意:这个方法需要 oracle 访问 ground-truth $y$,所以**只在训练时可用**,推理时不能用——这也是为何要把这个信号蒸馏进 RL policy。

---

## Experimental Setup

### Training environment: 20 Questions

- Target concept $y$:secret word from Google Trillion Word Corpus,经 Grok 3 mini LLM-as-judge 过滤后 4,678 个 noun
- Data splits:SFT 341 / RL train 1,000 / validation 198 / test 433
- Agent:Qwen3-1.7B、Qwen3-4B(non-thinking mode)
- User simulator:Qwen3-14B
- Max turns:20

### RL training details

- 起点:SFT checkpoint(用 Gemini 2.0 Flash 的 8,918 rejection-sampled demonstration turns 微调)
- Algorithm:turn-wise GRPO,$G=16$,temperature 1.0
- LoRA:rank 64,$\alpha=64$,LR $3 \times 10^{-5}$
- Hardware:2× NVIDIA H100(一块 train questioner,一块 run user simulator)
- Asymmetric clipping(来自 DAPO):$\epsilon_{\text{high}}=0.28$, $\epsilon_{\text{low}}=0.2$
- 无 KL penalty(ablation Figure 12 显示无效果)

---

## Main Results

### Table 1: 20Qs test set 性能

| Method | Mean@8 ± std | Pass@8 |
|---|---|---|
| BASELINE (1.7B) | 9.97% ± 1.04% | 32.03% |
| STARPO (1.7B) | 16.54% ± 1.32% | 45.73% |
| **CIA (1.7B)** | **24.80% ± 1.10%** | **53.10%** |
| BASELINE (4B) | 13.34% ± 1.05% | 36.87% |
| STARPO (4B) | 24.36% ± 1.18% | 59.12% |
| **CIA (4B)** | **33.72% ± 1.26%** | **63.97%** |
| DeepSeek-V3.2 (670B) | 14.35% ± 0.87% | 47.34% |
| Qwen3-235B-Instruct | 8.83% ± 0.87% | 27.71% |

关键观察:
1. CIA 在两个 scale 上都比 STARPO(trajectory-level GRPO)绝对涨 8-9 个点,说明 dense ΔBelief signal 真的提供了更精细的 credit assignment,而非单纯 RL 的功劳
2. CIA-1.7B(2.5B 参数量级)击败 DeepSeek-V3.2(670B)10+ 个点,说明 information-seeking 这种能力**不是靠 scaling pretraining 就能解决的**,需要专门的训练
3. Pass@8 比 Mean@8 涨幅更大(1.7B: 24.80→53.10),说明 CIA 不只是 sharpening mode、确实扩大了"能找到解的 trajectory manifold"

### Figure 4: Exploration efficiency

训练过程中 CIA 比 STARPO 更快降低:
- 平均每 episode 提问数(更高效)
- repeated question 比例(更少冗余)

单纯 trajectory-length penalty 也能压 turn 数,但效果远不如 ΔBelief——因为 ΔBelief 直接优化"信息量"而非"长度"。

### Figure 5: Belief update dynamics

在 4B 上,CIA 训练后 agent 的 per-turn belief 上升 rate 显著大于 SFT baseline 和 STARPO,说明 policy 真的学到了 ask more informative questions。1.7B 上没观察到差异——作者解释是**小模型 internal world model 不够 calibrated**,belief update 信号本身就 noisy。

### Figure 6: Pass@k up to k=128

CIA 在 $k=128$ 时仍领先 baseline,说明 RL 后处理没有 collapse 到 narrow distribution,exploration 能力保持住了。这是对"RL just sharpens pass@1"反驳的有力证据。

### Figure 8: Test-time interaction scaling

训练时 cap 20 turns,但测试时把 budget 扩到 50 turns,CIA 的成功率持续上升,而 STARPO 和 SFT 接近 plateau。这意味着 CIA 学到的是**通用的 information-seeking strategy**,而非 overfit 到 20-turn 的 specific policy。4B 上 50-turn 时 CIA 比 STARPO 绝对高 26 个点。

---

## Ablations

### λ (intrinsic reward weight)

- $\lambda = 0.0$:退化为 STARPO
- $\lambda = 0.05$:稳定但提升有限
- $\lambda = 0.1$:最优
- $\lambda \geq 0.5$:训练 collapse

原因:ΔBelief 是 unbounded 的(论文里观察到 raw value 范围约 $[-22, 24]$),而 $r^{\text{eog}} \in \{0, 2\}$ 是 bounded 的。$\lambda$ 过大会让 intrinsic reward dominate,agent 可能学到 hack 信号(比如反复问 "Is it X?" 让 log-prob 短暂跳一下)而不真正推进任务。

### Normalization scheme

| Method | 效果 |
|---|---|
| Naive(无 normalization) | 差,大 negative 值 dominate |
| EMA(lagged reference policy) | 差,同上 |
| tanh squashing | 差,信号被过度压缩 |
| **ReLU(positive-only shaping)** | **最优** |
| PACR(min-max scaled positive) | 接近 ReLU 但略差 |

ReLU 的优势:保留 informative variation,同时避免 negative signal 干扰 verifiable reward。

### Table 2: Generalization to stronger user simulators

训练时 user simulator 是 Qwen3-14B,测试时换成 Qwen3-235B 或 DeepSeek-v3.2。CIA 性能基本保持(35.65% → 31-37%),证明 agent 没有_overfit_ 到 train-time simulator 的 artifact。

---

## OOD Generalization

### Figure 9: Guess My City & Murder Mystery

- **Guess My City**(185 cities, open-ended questions 允许):CIA-4B Pass@1 比 STARPO 高 12.6 个点;CIA 用 "top-down" abstract question 策略,STARPO/SFT 用 brute-force 枚举具体城市
- **Murder Mystery**(50 scenarios, 5 suspects):CIA-4B Pass@1 比 STARPO 高 28.5 个点——这是最大 gap,因为任务 hypothesis space 小、对单步推理 quality 极其敏感

### Figure 10: Practical applications

- **User Personalization**(STaR-GATE, 1,000 samples, 3 turns):CIA 比 STARPO 提升最高 15%,因为 agent 学到了 eliciting latent user preferences 的能力
- **Customer Service**(200 scenarios, 20 turns):CIA-4B Pass@1 比 STARPO 高 11.13%,且 Pass@k gap 随 $k$ 增大——CIA 学到了**根据 user 响应动态调整 inquiry** 的能力,STARPO 容易被 irrelevant 概念 side-track

Appendix A.2.3 的 customer service 例子很说明问题:
- SFT agent 问一长串 undirected 问题
- STARPO agent 稍 directed 但被 irrelevant car model 反复绕进死循环(20 turns 还没诊断完)
- CIA agent 在 10 turns 内通过排除法锁定 engine mount 问题

---

## Architecture & Training Pipeline 解析

整体 pipeline(我整理的):

```
┌──────────────────────────────────────────────────────────────┐
│ 1. SFT bootstrap                                             │
│    Gemini 2.0 Flash → rejection sampling → 8,918 turns      │
│    Fine-tune Qwen3-1.7B/4B on 341 secret words              │
│    → Baseline SFT checkpoint                                 │
├──────────────────────────────────────────────────────────────┤
│ 2. ΔBelief-RL training loop                                  │
│    For each prompt (secret word y):                         │
│      ┌─ Sample G=16 trajectories {τ^i} from π_θ             │
│      │   Each turn:                                          │
│      │     a_t ~ π_θ(·|h_t)   (agent question)              │
│      │     o_t ~ π_user(·|h_t, a_t)  (Qwen3-14B response)   │
│      │     b_t = p_θ(y | h_t, e_i)  ← elicit belief         │
│      │     ΔBelief_t = log b_t - log b_{t-1}                │
│      │     r_t = r^eog + λ·max(ΔBelief_t,0) + r_p           │
│      └─ End of trajectory: compute r^eog                    │
│                                                              │
│    Compute turn-wise advantage:                             │
│      Â_t^i = (r_t^i - mean_t) / std_t                       │
│                                                              │
│    Update π_θ via GRPO with:                                │
│      - Token masking (only agent tokens)                    │
│      - Asymmetric clipping (ε_high=0.28, ε_low=0.2)         │
│      - No KL penalty                                         │
│      - LoRA rank=64                                          │
└──────────────────────────────────────────────────────────────┘
```

关键设计 decisions:
1. **Non-thinking mode**:为了把完整 20-turn trajectory 塞进 context window for backprop,关掉 Qwen3 的 thinking mode(thinking token 会爆 context)
2. **Single-turn SFT**:把 trajectory 拆成 single-turn samples 训练,而非整条 trajectory 一起训——作者发现这样更好,可能是因为降低了 off-policy gap
3. **Judge 设计**:允许 indirect guess(question 里只要包含 secret 就算 "Finished"),所以 CIA 学到的 optimal policy 是"inquire broad category + 在 question 里 enumerate category elements"

---

## Connection to Broader Literature

### Intrinsic motivation / curiosity in classic RL

ΔBelief reward 本质上是 **prediction error / information gain** 类的 intrinsic motivation,经典 RL 文献里很多 precedent:
- Schmidhuber 1991: curiosity as prediction error
- Pathak et al. 2017 ICM: forward model prediction error as curiosity
- Burda et al. 2018 RND: random network prediction as exploration bonus

ΔBelief 的创新点在于:**用 LLM 自己对 ground-truth target 的 conditional probability 作为"预测"**,而非单独训练 forward model。LLM 的 conditional distribution 隐式包含了 world model。

### Active learning / Bayesian Experimental Design

20Qs 是经典 active learning benchmark。BED-LLM(Choudhury et al. 2025)用 Bayesian Experimental Design 选 question,但需要显式 likelihood model。ΔBelief 把 LLM 自身当成 implicit Bayesian model。

### Process Reward Models vs. Verifier-free RL

- PRM 路线(Lightman et al. 2024 Let's Verify Step by Step):需要昂贵 step-level human annotation
- Verifier-free 路线:NOVER(Liu et al. 2025)用 reasoning perplexity 作 reward;PACR(Yoon et al. 2025)用 step-wise confidence 上升
- RLP(Hatamizadeh et al. 2025):pretraining 阶段就最大化 information gain
- ΔBelief-RL 的区别:**在 interaction(turn)级别** measure belief update,而非 sequence/reasoning step 级别

### Multi-turn RL for LLM agents

- ArCHer(Zhou et al. 2024):hierarchical multi-turn RL
- RAGEN(Wang et al. 2025)/ StarPO:trajectory-level GRPO
- ARIA(Yang et al. 2025):intention-driven reward aggregation

ΔBelief-RL 与 ARIA 思路相近(都试图给中间 turn credit),但信号来源不同:ARIA 用 intention aggregation,ΔBelief 用 intrinsic probability shift。

参考链接:
- Pathak ICM:https://arxiv.org/abs/1705.05363
- Lightman PRM:https://openreview.net/forum?id=v8L0pN6EOi
- ArCHer:https://openreview.net/forum?id=b6rA0kAHT1
- NOVER:https://aclanthology.org/2025.emnlp-main.378/
- RLP:https://arxiv.org/abs/2510.01265
- RND:https://arxiv.org/abs/1810.12894
- BED-LLM:https://arxiv.org/abs/2508.21184

---

## Intuition Building: 为什么这个方法 work?

让我用几个 angle 帮你 build intuition:

### Angle 1: LLM 是 implicit Bayesian agent

当你问 LLM "Is the secret X?" 并收到 "Yes",模型内部对 $X$ 的 conditional probability $p_\theta(X \mid h_t)$ 必然上升(假设 attention 机制正常工作)。这等价于 Bayesian update:

$$p_\theta(y \mid h_t, o_t) \propto p_\theta(o_t \mid y, h_t) \cdot p_\theta(y \mid h_t)$$

LLM 没有显式做这个 update,但下一个 token prediction 的 conditional distribution 隐式反映了这个 posterior shift。所以 $b_t = p_\theta(y \mid h_t, e_i)$ 是一个**近似 posterior**,而 $\Delta\mathrm{Belief}_t$ 是 posterior log-ratio,正好对应 Bayesian update 的 log-likelihood ratio。

### Angle 2: ΔBelief 是 free 信号,因为 forward pass 已经在做

计算 $b_t$ 需要一次 forward pass 把 $y_i$ 喂进去取 log-prob。这个 forward pass 跟 sampling $a_t$ 的 forward pass 共享 KV cache(只是不同 target),所以 marginal cost 很低。相比训练一个 value network,这几乎免费。

### Angle 3: Turn-level baseline 解决 variance

Long-horizon RL 最大问题是 variance:同一条 trajectory 不同 turn 之间 reward 不平衡。trajectory-level GRPO 用一个 scalar advantage 覆盖所有 token,导致:
- 前 5 turn 的好 action 和后 15 turn 的烂 action 拿同样 credit
- variance 巨大,学习 signal 被 noise 淹没

Turn-level advantage $\widehat{A}_t^i$ 在每个 turn 用 group-internal baseline,等价于把 trajectory 切成 N 个 mini-RL problem,每个 mini-problem 内部用 leave-one-out baseline。这是 REINFORCE leave-one-out baseline(PPO/GRPO 的核心 trick)在时间维度的推广。

### Angle 4: 为什么 ReLU clip 比 symmetric 好?

如果允许 negative ΔBelief 作为惩罚,agent 会学到 avoid "broad exploration questions"——因为 broad question 必然让某些 candidate 的 posterior 暂时下降(你问 "Is it animal?" 答案是 No,所有动物概率暴跌)。但 broad question 是 20Qs 最有效的开局策略。

ReLU clip $\max(\Delta\mathrm{Belief}, 0)$ 只奖励 "belief 提升",不惩罚 "belief 下降",允许 agent 自由 explore broad hypothesis space 而不被惩罚。这跟 prediction-error-based curiosity 不同——后者会鼓励 surprising outcome,反而可能 prefer wrong broad question。

### Angle 5: 与 model-based RL 的关系

ΔBelief-RL 本质上是 model-based RL,其中 **LLM 自身就是 world model**。$b_t = p_\theta(y \mid h_t, e_i)$ 是模型对 latent state $y$ 的 belief。ΔBelief reward 就是 "latent state estimation 的改进量"——这在 POMDP 文献里叫 "information gain reward" 或 "belief-based exploration"。

经典 POMDP 公式:information gain reward $= \mathrm{KL}[b_t \| b_{t-1}]$。ΔBelief 是它的一个特例(只看 ground-truth $y$ 这一个维度的 marginal posterior 变化,而非整个 belief distribution)。这暗示一个**改进方向**:用 full distribution 的 KL divergence 而非 single-target 的 log-ratio,会更 principled 但更 expensive。

---

## Limitations & Future Directions

论文自己点出的限制:

1. **Single reference target 限制 diversity**:对 multiple viable solutions 的任务(creative writing),只有一个 ground-truth 会让 agent 过度 sharpen 到这个 reference。Future work 可以用 distribution over plausible targets。
2. **Calibration 依赖 model scale**:1.7B 上 belief dynamics 没观察到明显变化,说明小模型 world model 不够 calibrated。可能需要 scale 到一定 size 才能受益。
3. **需要 supervised reference**:训练时必须知道 ground-truth $y$。对完全 open-ended 任务(没有 verifiable answer)不直接适用。但作者暗示,未来若 agent 有更好的 implicit world model,可以 self-generate reference。

我补充几点 intuition:

4. **Reward hacking 风险**:$\Delta\mathrm{Belief}$ 是 model 内部信号,理论上 agent 可以学到 "生成某些特定 token pattern 让 log-prob 短暂跳一下" 而不真正推进任务。论文里 ReLU + λ 调节 + eog reward 联合作用缓解了这个,但更复杂任务上可能需要额外约束。
5. **Sample efficiency 的真正来源**:作者 claim "more sample-efficient training"。从 Figure 4 看,CIA 比 STARPO 用更少 turns 解决问题,但 training step 数没明显减少。这里的 "sample efficient" 主要指 inference time efficiency 而非 training data efficiency——这两者要区分。
6. **Belief elicitation 的 prompt 敏感性**:$e_i = $ "Is the secret \<target\>" 这个 template 是 hand-crafted。对其他任务需要重新设计 elicitation prompt,可能引入 inductive bias。

---

## Code & Reproducibility

论文 project page: https://bethgelab.github.io/delta-belief-rl

关键 implementation 细节(Appendix F):
- 用 verl 框架的 modified SFTTrainer 和 GRPO
- Token masking:只对 agent 生成 token 算 loss,environment token 被 mask 掉
- Asymmetric clipping from DAPO(Yu et al. 2025):https://arxiv.org/abs/2503.14476
- 无 KL penalty(Figure 12 ablation 证明无 effect)
- 2× H100,LoRA rank 64,LR $3 \times 10^{-5}$

---

## Take-aways for building intuition

1. **LLM 自身是免费的 critic**:任何有 verifiable outcome 的任务,都可以用 "policy 对 ground-truth 的 conditional log-prob 变化" 作为 dense per-step reward,无需单独训 value network 或 PRM。

2. **Turn-level advantage > trajectory-level advantage**:long-horizon RL 一定要做 turn/step-level credit assignment,trajectory-level scalar advantage variance 太大。Group-relative baseline 在 turn 级别做 leave-one-out 是低成本高回报的改法。

3. **ReLU clip 比 symmetric penalty 好**:exploration 阶段 belief 暂时下降是 healthy 的,不应该惩罚。只奖励 "向 target 移动" 的 action 即可。

4. **Information-seeking 是可训练的 capability**:不是单纯靠 scaling pretraining 自动涌现的。Web data 不 capture belief-dependent question utility,需要专门 RL 训练。1.7B/4B 的小模型经过 ΔBelief-RL 可以击败 670B 的 DeepSeek-V3.2,说明这个 capability 有"trainable ceiling"。

5. **Generalize 到 unseen task**:在 20Qs 上训练的 information-seeking 能力,可以 transfer 到 customer service、user personalization、murder mystery——说明学到的不是 task-specific heuristic,而是 meta-strategy of "如何高效提问以缩小 hypothesis space"。

6. **Test-time scaling 持续收益**:训练时 cap 20 turns,推理时给 50 turns 性能继续涨。这说明 policy 学到的是 "如何继续 explore" 的 open-ended 能力,而非 "如何在 20 turn 内关门" 的特定策略。这对未来 long-horizon agent 很重要。

7. **Limitation 暗示下一步**:从 single-target log-prob 推广到 full belief distribution KL divergence、从 supervised reference 到 self-generated reference,是把这套方法推向真正 open-ended learning 的关键路径。

希望这帮你 build 起对 intrinsic credit assignment 的 intuition。如果你想深入某个具体方面(比如 turn-wise GRPO 的 variance 分析、或 belief elicitation 的 prompt engineering),可以继续聊。
