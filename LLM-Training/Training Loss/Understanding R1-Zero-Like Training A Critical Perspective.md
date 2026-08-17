---
source_pdf: Understanding R1-Zero-Like Training A Critical Perspective.pdf
paper_sha256: 98243d51297f011fb5baad8a70a972d061ffbef618f1d7cfc10deea37c5887d0
processed_at: '2026-08-12T19:13:05-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 到底在说什么

Paper: https://arxiv.org/abs/2503.20783
Code: https://github.com/sail-sg/understand-r1-zero

---

## 一句话总结

DeepSeek-R1-Zero 看起来像是"纯 RL 一步登天教会模型推理"，但这篇 paper 说：**你看到的"奇迹"有一大半是 base model 偷偷练过的，还有一小半是 optimizer 在帮模型"注水"把回答拉长而已**。

---

## 故事背景：R1-Zero 讲了个什么神话

DeepSeek-R1-Zero 的故事是这样的：拿一个 base model（DeepSeek-V3-Base），不经过 SFT，直接用 GRPO 做 RL，训练它做数学题。结果发现：
- 模型回答越来越长；
- 中途突然出现"等等，我重新想想"这种 self-reflection，叫 "Aha moment"；
- 最终推理能力暴涨。

大家一看，哇，RL 厉害，不用 SFT 也行，于是纷纷开始复现，基本都用 Qwen2.5-Math 系列。Sea AI Lab 这篇 paper 就是来拆台的。

---

## 拆台一：Qwen2.5-Math 根本不是"纯 base model"

### 观察

作者拿 Qwen2.5-Math-7B 做数学题，试了三种问法：
1. 加 R1 template（`<answer>...</answer>` 那套）；
2. 加 Qwen-Math template（`\boxed{}` 那套）；
3. **啥也不加，直接把题目甩给它**。

结果（Table 1，五个 benchmark 平均）：

| 问法 | 平均准确率 |
|---|---|
| 4-shot prompting | 23.8% |
| R1 template | **0.0%** |
| Qwen template | 26.5% |
| **No template（裸问）** | **38.2%** |

裸问最强，加 R1 template 直接归零。

### 这意味着什么

一个"真正"的 base model，你直接甩题目给它，它大概率会接着题目继续写下去，而不是回答。比如你问"2+3=?"，它可能续写成"2+3=5, 3+4=7, ..."，因为它训练目标是 sentence completion。

但 Qwen2.5-Math 不一样，你裸问它就乖乖答题。**这说明 Qwen2.5-Math 在 pretraining 阶段就已经见过大量"问题直接接答案"的文本，相当于在 pretraining 里偷偷做了 SFT**。作者猜测他们把 question 和 answer 直接拼起来 `log p(q; o)` 训练了。

### 对 R1-Zero 复现的影响

这就有意思了——大家复现 R1-Zero 都用 Qwen2.5-Math，然后说"看，纯 RL 就能让模型会推理"。但你起点已经是一个"半 SFT"模型了，RL 只是在这个基础上微调，不是从零开始。**所以"纯 RL 不需要 SFT"这个叙事，在 Qwen2.5-Math 上是不成立的**。

对比一下 DeepSeek-V3-Base：它不用 template 时 answering rate 最低，是个更"干净"的 base model。这也是为什么真正的 R1-Zero 用的是它。

---

## 拆台二：Aha moment 不是 RL 涌现的，base model 本来就有

### 观察

作者把 DeepSeek-V3-Base-685B 自己 host 起来，用 R1 template 问它 500 道 MATH 题，统计 self-reflection 关键词（recheck、rethink、wait 等）出现的频率。

结果（Fig. 3 right + Fig. 13）：**DeepSeek-V3-Base 本身就会说"Wait"、"Aha"这种话**，根本不用 RL 教。

Fig. 13 里有两个真实例子，比如模型算着算着说："*awkward silence* Wait, I'm overthinking. Let's try again."

### 这意味着什么

"Aha moment 是 RL 涌现的"这个说法，至少在 DeepSeek 这条线上是错的。base model 已经有了这种行为，RL 只是让它的频率变高。之前 Liu et al. 2025b 和 Yeo et al. 2025 在开源模型上就发现过这点，但没测过 DeepSeek-V3-Base，这篇补上了。

更有意思的是 Sec. F：作者对比了 DeepSeek-V3-Base 和真正的 DeepSeek-R1-Zero，发现有 self-reflection 的回答，准确率**并不比没有 self-reflection 的回答高**。也就是说，"说 wait"和"真的做对题"没有正相关。Self-reflection 可能更多是个 surface pattern，不是真正的能力 indicator。

直觉上想：模型说"等等我重新算"，可能只是它训练数据里见过这种表达，并不代表它真的有个"内心反思"的过程。这跟之前有 work 说 CoT 里的 "let me think step by step" 类似——surface form 和 underlying capability 是两回事。

参考：
- Liu et al. 2025b: https://oatllm.notion.site/oat-zero
- Yeo et al. 2025 "Demystifying long CoT": https://arxiv.org/abs/2502.03373

---

## 拆台三：response 越来越长，部分是 GRPO 的 bug

这是这篇 paper 技术上最重要的部分。

### GRPO 长什么样

GRPO 的 objective（Eq. 3）大致是：

$$
\mathcal{J}_{GRPO} = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|\mathbf{o}_i|} \sum_{t=1}^{|\mathbf{o}_i|} \min\left[\rho_{i,t}\hat{A}_{i,t},\ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right]
$$

变量解释：
- $G$：每个 question 采样几条 response（group size）；
- $|\mathbf{o}_i|$：第 $i$ 条 response 的 token 长度；
- $\rho_{i,t} = \pi_\theta(o_{i,t}|\mathbf{q},\mathbf{o}_{i,<t}) / \pi_{\theta_{old}}(o_{i,t}|\mathbf{q},\mathbf{o}_{i,<t})$：新旧 policy 概率比；
- $\hat{A}_{i,t}$：第 $i$ 条 response 第 $t$ 个 token 的 advantage；
- $\epsilon$：PPO clip 范围，一般 0.2。

advantage 怎么算：

$$
\hat{A}_{i,t} = \frac{R(\mathbf{q},\mathbf{o}_i) - \text{mean}(\{R(\mathbf{q},\mathbf{o}_1),...,R(\mathbf{q},\mathbf{o}_G)\})}{\text{std}(\{R(\mathbf{q},\mathbf{o}_1),...,R(\mathbf{q},\mathbf{o}_G)\})}
$$

$R$ 是 outcome reward（答对=1，答错=0），mean 和 std 都是在同一个 question 的 $G$ 条 response 上算的。

### 两个 bias

作者指出 GRPO 有两个"偷偷加进去"的 normalization，会 bias 优化：

**Bias 1: Length normalization（$\frac{1}{|\mathbf{o}_i|}$）**

这个除法把每条 response 的 loss 平均成"每个 token"的 loss。看起来很合理，但它带来了一个副作用：

- 如果某条 response 是**对的**（$\hat{A} > 0$），除以 $|\mathbf{o}|$ 意味着**短回答的梯度更大**——模型被推向"答对了就赶紧停"；
- 如果某条 response 是**错的**（$\hat{A} < 0$），负的梯度被 $|\mathbf{o}|$ 稀释，**长回答的惩罚更轻**——模型在"答错的时候倾向于啰嗦"。

合起来就是：**正确答案变短，错误答案变长**。这跟 R1-Zero 观察到的"response 越来越长"现象完美吻合，但方向是反的——大家以为是推理变复杂了，其实部分是错误回答被"放水"拉长。

**Bias 2: Std normalization（除以 std(R)）**

advantage 除以 group 内 reward 的 std。问题在于：如果一个 question 太简单（所有 response 都对，std=0 或接近 0），或太难（都错，std 也接近 0），除以一个很小的 std 会产生巨大的 advantage，**这些"没信息量"的 question 反而权重最大**。

正常的 RL 实现里，advantage normalization 是在整个 batch 上做的，不会让单个 question 的 std 决定它自己的权重。GRPO 这个 per-question normalization 是个奇怪的设计。

### 开源 PPO 实现也中招

作者还检查了 trl、OpenRLHF、verl、SimpleRL-Zero、Open-Reasoner-Zero 这些主流开源 RLHF 框架，发现它们在 PPO loss 里也都有 length normalization（Listing 1）。源头可能是 pretraining 时的 `loss.mean()` 习惯，因为 pretraining 把 token pack 成固定长度 context，除以 context length 没问题。但 RL 里 response 长度是变的，这就引入了 bias。

Table 2 列了一排 ✗，挺震撼的：大家都在用有 bias 的实现。

---

## 修复：Dr. GRPO（GRPO Done Right）

### 改法

超级简单，就两步：
1. **去掉 $\frac{1}{|\mathbf{o}_i|}$**：不再按 response 长度归一化，改用固定常数（比如 generation budget MAX_TOKENS）做 batch 级归一化；
2. **去掉 std normalization**：advantage 只减 mean，不除 std。

改完之后的 advantage：

$$
\tilde{A}_{i,t} = R(\mathbf{q},\mathbf{o}_i) - \text{mean}(\{R(\mathbf{q},\mathbf{o}_1),...,R(\mathbf{q},\mathbf{o}_G)\})
$$

作者证明这个 $\tilde{A}$ 等价于 REINFORCE Leave-One-Out（RLOO）的 advantage，up to 一个 scaling factor，理论上是 unbiased 的。推导在 Appendix A：

$$
\frac{G}{G-1}\tilde{A}_{i,t} = \frac{G}{G-1}R_i - \frac{1}{G-1}\sum_{j\neq i}R_j - \frac{1}{G-1}R_i = \hat{A}^{RLOO}_{i,t}
$$

直觉：leave-one-out baseline 是 RL 里经典的 variance reduction 手法，把"其他 $G-1$ 条 response 的平均 reward"当成 baseline，是 unbiased 的。GRPO 除以 std 后虽然还是 unbiased 的（std 不依赖 action），但它**改变了不同 question 之间的相对权重**，导致 difficulty bias。

参考 REINFORCE with baseline：Sutton & Barto Chapter 13，https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf

### 效果（Fig. 5）

训练动态对比：
- **GRPO**：reward 上升一段后趋于平稳，但 response length 继续疯狂增长——典型的"优化器在注水"；
- **Dr. GRPO**：reward 上升，response length 稳定在一个合理值，不再无限增长。

评估对比：
- 正确回答长度两者差不多；
- **错误回答长度 Dr. GRPO 显著更短**——说明 GRPO 确实在让错误回答变啰嗦，Dr. GRPO 修掉了这个 bug；
- 最终 benchmark 准确率 Dr. GRPO 略高或持平，但 token efficiency 好很多。

Fig. 9 给了 3 个 random seed 的结果，Dr. GRPO 在准确率和 token efficiency 上都稳定优于 GRPO，统计显著。

---

## 拆台四：template 和 question set 的微妙互动（Sec. 3.3）

### 实验

拿 Qwen2.5-Math-1.5B，用 Dr. GRPO 做 RL，交叉两个维度：
- Template：R1 / Qwen-Math / No template；
- Question set：ORZ（57k 大覆盖）/ MATH（12k）/ GSM（8k 简单）/ ASDiv（2k 基础）。

### 发现（Fig. 6）

1. **RL 能把不同 template 的起点拉到差不多**（都到 ~40% AIME）。所以"R1 template 带来的巨大提升"其实部分是 RL 在弥补 template 造成的损伤。

2. **R1 template + GSM（简单题）= 灾难**。因为 R1 template 和 Qwen2.5-Math 的 mismatch 很大，需要 question set 有足够 coverage 来"重建"被 template 破坏的能力。GSM 太窄太简单，撑不起来。

3. **Qwen-Math template + GSM = 反而最好**。起点高（Qwen-Math template 和 base model 兼容），用简单题做 RL 就能强化已有推理行为，不要求 question set 教新知识。

### 直觉

这跟"pretraining 已经给了什么"强相关：
- 如果 template 和 base model 兼容（Qwen-Math template 配 Qwen base），RL 只需强化，简单题就够；
- 如果 template 和 base model 不兼容（R1 template 配 Qwen base），RL 要先修复再提升，需要 question set 提供 coverage。

这对实际 recipe 选择很有指导意义：**先看你的 base model 是不是已经被 SFT 过，再决定要不要套 template、用什么 question set**。

---

## 拆台五：弱 base model 也能救，但需要 domain pretraining（Sec. 3.4）

### 实验

拿 Llama-3.2-3B（数学很弱），看 RL 能不能救：
- 原版 Llama-3.2-3B + Dr. GRPO → 提升很小（Avg 3.3% → 6.8%）；
- Llama-3.2-3B-FineMath（continual pretrain 数学）+ Dr. GRPO → 14.8%；
- Llama-3.2-3B-NuminaQA（再拼 Q-A 训练）+ Dr. GRPO → **20.7%**。

### 直觉

RL 不是凭空变出能力，它是在 base model 已有的"潜力"上做 shaping。Llama 原版数学太弱，RL 没东西可塑；加了 domain pretraining 后，有了潜力，RL 才能放大。

NuminaQA 那个拼接 Q-A 训练就更有意思了——这其实就是在模仿 Qwen2.5-Math 可能做过的事。作者等于是在 Llama 上"复现"了 Qwen 的 pretraining bias，然后证明这样做确实能让 RL 效果变好。

这条线说明：**"纯 RL 不需要 SFT"的叙事要打折扣。你可以不做显式 SFT，但 pretraining 里得埋下相应的 capability**。

---

## 最终 recipe：27 小时 8×A100 打到 7B SOTA

基于以上分析，作者提出 minimalist recipe：
- Base model: Qwen2.5-Math-7B；
- Algorithm: Dr. GRPO（无 KL，无 length/std normalization）；
- Template: Qwen-Math template；
- Data: MATH level 3-5（12k 题）；
- Reward: Math-Verify 规则验证，答对=1 否则=0；
- 硬件: 8×A100，27 小时。

结果（Table 4）：Oat-Zero-7B 在 AIME 2024 上 43.3%，平均 51.4%，超过 SimpleRL-Zero、OpenReasoner-Zero 等，是 7B 量级 SOTA。

1.5B 版本 Oat-Zero-1.5B 平均 42.1%，也超过了 R1-Distill-Qwen-1.5B @ 3k（22.0%）。

---

## 我觉得这篇 paper 最重要的几个 take-away

1. **Base model 的 pretraining bias 决定 RL 起点**。Qwen2.5-Math 很可能 Q-A 拼接训练过，所以"无 SFT 纯 RL"在它上面是个错觉。用 Qwen2.5-Math 复现 R1-Zero 时要意识到这点。

2. **GRPO 的 length normalization 是个 bug**，它让错误回答越长越"便宜"，正确回答越短越"划算"，造成 response length 增长的假象。Dr. GRPO 简单修掉就好。

3. **Aha moment 不是 RL 涌现**，DeepSeek-V3-Base 本身就有。self-reflection 关键词出现频率和准确率不正相关，surface pattern ≠ capability。

4. **Template 和 question set 要配套**。template 和 base model 兼容时简单题就够；不兼容时需要大覆盖 question set 来修复。

5. **RL 的天花板由 pretraining 决定**。Llama 数学弱需要 domain pretraining 才能让 RL 起作用，没有潜力 RL 也无能为力。

---

## 相关联想和 open questions

- **PRIME（Cui et al. 2025）**：https://arxiv.org/abs/2502.01456 也用 Qwen2.5-Math 做 zero RL，他们的 process reward 会不会也受 length bias 影响？我觉得会，process reward 本质上还是 token-level，除以长度一样有 bias。

- **DeepSeek 官方 R1 和 R1-Zero 的差别**：R1 是 SFT + RL，R1-Zero 是纯 RL。这篇 paper 没直接对比两者，但从分析看，R1 的 SFT 阶段可能在帮 base model 跨过 template mismatch 这道坎，让 RL 能更专注提升而非修复。

- **overthinking 问题**：Chen et al. 2024 "Do not think that much for 2+3=?" (https://arxiv.org/abs/2412.21187) 指出 o1-like 模型在简单题上也会长篇大论。Dr. GRPO 短错误回答的结果正好和这个问题相关——可能 length bias 就是 overthinking 的成因之一。

- **OpenAI o1 的训练细节**：我们不知道 o1 用的是什么 RL 算法，但如果它也是 PPO-like 且有 length normalization，那 o1 的"思考链越来越长"也可能有 optimization artifact 的成分。这点没法验证，但是个有趣的猜测。

- **value function 的问题**：GRPO 本质是想避开 value model $V_\phi$，用 group statistics 当 baseline。但 per-question std normalization 引入的 difficulty bias 其实有点像"学了一个 bad value model"。如果用一个真正的 learned $V_\phi$ 跨 question 归一化，可能更干净。但 $V_\phi$ 在 LLM 上确实难训，这也是 GRPO 流行的原因。

- **更进一步的去 bias**：Dr. GRPO 去掉了 length 和 std normalization，但 PPO clip 本身、advantage 的 group size $G$、KL penalty（这里 $\beta=0$）都是 design choice。作者 $\beta=0$ 是因为用 rule-based reward，不怕 distribution shift。但如果是 RLHF with learned reward model，$\beta > 0$ 还是必要的。

- **Batch composition 的影响**：GRPO 的 difficulty bias 本质是让 batch 里"极端难度"的 question 权重过大。这让我想到 OpenAI 的 "filtering easy/hard prompts" 的 RLHF recipe——可能就是为了类似的目的。Anthropic 的 Constitutional AI 也是用 batch-level normalization。

---

## 公式小结卡

**GRPO advantage**（有 bias）：

$$
\hat{A}_{i,t} = \frac{R_i - \bar{R}}{\sigma_R}
$$

- $R_i$：第 $i$ 条 response 的 reward；
- $\bar{R} = \frac{1}{G}\sum_j R_j$：group 内 mean；
- $\sigma_R$：group 内 std。

**Dr. GRPO advantage**（unbiased）：

$$
\tilde{A}_{i,t} = R_i - \bar{R}
$$

等价于 RLOO baseline 乘以 $\frac{G}{G-1}$。

**完整 objective**（Dr. GRPO）：

$$
\mathcal{J} = \mathbb{E}_{q, \{\mathbf{o}_i\}}\left[\frac{1}{G}\sum_{i=1}^G \sum_{t=1}^{|\mathbf{o}_i|}\min\left[\rho_{i,t}\tilde{A}_{i,t},\ \text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\tilde{A}_{i,t}\right]\right]
$$

注意：**没有 $\frac{1}{|\mathbf{o}_i|}$**，按 token 求和后直接 batch 平均。

---

总结一句：这篇 paper 是对 R1-Zero 叙事的一次重要"祛魅"，提醒大家别被表面现象（response 变长、Aha moment）迷惑，要看清楚 base model 带来了什么、optimizer 偷偷做了什么。Dr. GRPO 的 fix 简单到几乎免费，但效果显著。强烈建议做 LLM RL 的人都试一下。

---

# Understanding R1-Zero-Like Training: A Critical Perspective 深度讲解

这篇 paper 来自 Sea AI Lab（Min Lin 团队，第一作者 Zichen Liu），核心贡献是对 DeepSeek-R1-Zero 范式做了一次"祛魅"式的 critical analysis，并提出了 Dr. GRPO（GRPO Done Right）。我把关键内容拆开讲，顺便把我能联想到的相关 work 和直觉都串起来。

Paper link: https://arxiv.org/abs/2503.20783
Code: https://github.com/sail-sg/understand-r1-zero
Oat 框架: https://github.com/sail-sg/oat

---

## 1. 大局观：这篇 paper 在质疑什么

DeepSeek-R1-Zero 的"神话"有三条：
1. 纯 RL（不做 SFT）就能让 base model 学会推理；
2. response length 越来越长 = 推理能力 emergent；
3. "Aha moment" 是 RL 训练涌现的。

这篇 paper 的核心论点是：**这三条在很大程度上都被 base model 的 pretraining bias 和 GRPO 的 optimization bias 伪装了**。具体来说：
- Qwen2.5-Math 本身就已经在 pretraining 阶段做了类似 SFT 的事情（concat Q-A 训练），所以"不做 SFT"是一个错觉；
- response length 增长部分是 GRPO 的 length normalization bias 人为造成的，尤其在错误回答上；
- Aha moment 在 DeepSeek-V3-Base 本身就存在，RL 只是放大了它，而且 self-reflection 与 accuracy 不正相关。

---

## 2. Base Model 分析（Sec. 2）

### 2.1 Template 决定 base policy 是否能"回答问题"

Base model 训练目标是 sentence completion $p_\theta(x)$，而 RL 需要的是 conditional policy $\pi_\theta(\cdot | q)$。中间的桥梁就是 template。paper 测试了三种 template：

- **Template 1 (R1 template)**：用 `<answer>...</answer>` 的 chat 格式；
- **Template 2 (Qwen-Math template)**：`<|im_start|>system\nPlease reason step by step, and put your final answer within \boxed{}.<|im_end|>...`；
- **Template 3 (No template)**：直接 `{question}`。

实验结果（Fig. 3 left）很有意思：
- **Llama-3.1-8B、DeepSeek-Math-7B、DeepSeek-V3-Base** 用 R1 template 后 answering rate 大幅提升；不用 template 时 DeepSeek-V3-Base 的 answering rate 最低，说明它是一个"几乎纯净"的 base model；
- **Qwen2.5 全家** 反而是 No template 时 answering rate 接近 100%。这是一个非常强的信号，暗示 Qwen2.5 在 pretraining 时就见过大量 "Q directly followed by A" 的格式。

**Pass@8 探索能力**（Fig. 3 middle）：所有模型 pass@8 > 0，说明 RL 至少有 reward signal 可用；Qwen2.5-Math-7B 的 pass@8 甚至超过 DeepSeek-V3-Base-685B。这解释了为什么大家复现 R1-Zero 都喜欢用 Qwen2.5。

### 2.2 Qwen2.5-Math 不用 template 反而最强（关键发现）

Table 1 是整个 paper 最炸裂的实验之一。Qwen2.5-Math-7B 在五个 benchmark 上的平均准确率：

| Setting | Avg. |
|---|---|
| 4-shot prompting | 23.8 |
| R1 template | 0.0 |
| Qwen-Math template | 26.5 |
| **No template** | **38.2** |

No template 比 4-shot 提升约 60%，而 R1 template 直接让模型"失语"（0.0）。这说明 R1 template 的 `
