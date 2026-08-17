---
source_pdf: STAPO Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious
  Tokens.pdf
paper_sha256: e7e89576d9cd9b920e53826ee3471bd2e4bfccb3e3fcf21575d57dcc9ed3a8e2
processed_at: '2026-08-12T10:50:42-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 STAPO

## 一句话总结

训练 reasoning LLM 的时候，response 里大概每 1 万个 token 才有 1 个"坏 token"，但这 1 个 token 会把 gradient 搞得特别大，把整个训练带歪。STAPO 就是把这 0.01% 的 token 挑出来 mute 掉，训练立马稳定，性能还涨 7%。

Reference: paper 里 Figure 6a 显示 spurious token ratio 始终在 0.01% 以下
https://arxiv.org/abs/2503.14476 (DAPO, baseline)
https://arxiv.org/abs/2402.03300 (GRPO)

---

## 问题是什么

假设你让 LLM 做一道数学题，它写了 2000 个 token 的解题过程，最后答案对了，reward = +1。

GRPO/DAPO 这类算法的做法是：**整个 response 都共享这个 +1 的 reward**。也就是说，response 里的每一个 token——不管是关键的 "Therefore"、还是某个莫名其妙的 "Wait"、还是某个写错的中间数字——都拿到同一个 positive advantage。

这听起来好像也还好？毕竟大部分 token 都是 reasonable 的，被强化一下也没坏处。

问题在于：**有极少数 token，模型本来就不太想生成它（low probability），分布却很尖锐（low entropy），恰好又被采样出来了**。这三个条件凑齐，根据 Theorem 3.1 的 gradient bound，这种 token 的 gradient norm 会被自动放大——比正常 token 大 16.7%（Figure 2b 的实测数据）。

更糟的是，根据 Lemma 3.2 的 entropy dynamics，这种 low-prob + positive-advantage 的组合还会**主动 push entropy 向上**，引发 entropy explosion。

所以你看到的训练不稳定，本质上是这 0.01% 的 token 在背后疯狂搞事。

---

## 一个直觉比喻

想象一个合唱团录了一张专辑，乐评人听了整体感觉不错，给了五星好评。于是合唱团老板决定给每个人发奖金——**每个人，包括那个在第三首歌第 47 秒走音了 0.5 秒的 soprano**。

走音的 soprano 拿到奖金之后会怎么想？"哦原来走音也行啊"。下次录音她可能走音走得更厉害。

更糟糕的是，因为是"整体好评"，所以乐评人没注意到这个走音——它被 sequence-level reward 完全掩盖了。

STAPO 做的事就是：录音工程师在回放的时候，精准定位到那个 0.5 秒的走音片段，**在给 soprano 发奖金的时候跳过这一段**。其他 token 该奖的奖，这个走音的瞬间就当没发生过。

---

## 为什么是"low prob + low entropy + positive advantage"这三件套

这是整篇 paper 最微妙的地方，我用最朴素的话拆开讲。

**Low probability**: 模型本来就不想生成这个 token。比如该写 "removed" 的地方，它写成了 "broken"，"broken" 的概率只有 0.05%。

**Low entropy**: 这一步的整个分布很尖锐，99.5% 的 mass 都压在 "removed" 上，剩下 0.5% 均匀散在 15 万个其他 token 上。Local entropy 很低，因为模型其实"很确定"该写 removed。

**Positive advantage**: 整个 response 最后答案对了，所以这个 "broken" token 也跟着拿到了 +1 的 reward credit。

这三个条件合起来意味着什么？

- 模型本来就知道该写 "removed"，分布已经尖锐了（low entropy）——**再去强化 "broken" 没有任何 learning value**（Lemma 3.3）。
- "broken" 的概率只有 0.05%，importance sampling ratio $\rho = \pi_\theta / \pi_{\theta_{\mathrm{old}}}$ 容易变得很大，gradient 被放大（Theorem 3.1）。
- "broken" 拿到 positive advantage，相当于告诉模型"你写 broken 是对的"，这强化了一个错误 pattern。

把这三点叠起来，你就得到了一个**"零学习价值 + 巨大 gradient + 错误方向"**的完美风暴。

如果 advantage 是负的（response 错了），那 low-prob token 的 gradient 放大反而是好事——它在帮模型把概率往下推，是 corrective。所以 STAPO 只 mask positive advantage 的情况。

Reference: Theorem 3.1 的 proof 在 Appendix B
https://arxiv.org/abs/2505.12929 (Yang et al. 的原始 low-prob 分析)
https://arxiv.org/abs/2505.22617 (Entropy mechanism, Lemma 3.2 的来源)

---

## STAPO 的做法

超级简单。Algorithm 1 的核心就一行：

```
if (advantage > 0) and (probability < 0.002) and (entropy < bottom-20%-quantile):
    mask this token (gradient = 0)
else:
    keep it
```

三个 threshold：
- $\tau_p = 0.002$：probability 绝对值，低于这个就算"稀有"
- $\tau_h = 80\%$ quantile（注：paper 里 Section 5.1 和 5.3 描述有歧义，实际用法建议看 code）：entropy 在 mini-batch 内的相对位置
- advantage 必须是正的

Mask 之后，loss 的 normalization term 也相应调整——只数没被 mask 的 token，避免 mute 掉的 token 稀释 learning rate。

就这么简单。Algorithm 1 全文 15 行，没有什么 fancy 的 moving average、scheduler、auxiliary loss。

Reference: DAPO baseline
https://arxiv.org/abs/2503.14476
veRL framework (STAPO 的实现基础): https://github.com/volcengine/verl

---

## 一个关键设计选择：为什么 $\tau_p$ 用 absolute value，$\tau_h$ 用 quantile

这个 Remark 3.5 看起来是细节，其实体现了作者对 problem structure 的理解。

**Probability 是绝对的，entropy 是相对的。**

Probability = 0.001 就是客观稀有——不管什么 model、什么 training stage、什么 prompt，0.001 的概率就是低。如果某个 token 真的关键，它的概率应该被 training 拉高到 0.01、0.1、0.5，而不是一直卡在 0.001。所以用 absolute threshold 没问题。

Entropy 不一样。Entropy 取决于当前 step 的整个分布形状，而分布形状在不同 prompt、不同 position 上差异巨大。一个"决策点"（比如该用哪种方法解题）的 entropy 天然就高，一个"机械点"（比如写 "The answer is"）的 entropy 天然就低。如果用 absolute threshold，你会在机械点上 mask 一堆正常 token，在决策点上又 mask 不够。所以 quantile 更鲁棒——每个 mini-batch 内部自己比较，mask 掉相对低 entropy 的那批。

---

## 实验结果的人话版

**Setup**: Qwen3 1.7B / 8B / 14B base，DAPO-Math-17K 训练，64× H20 GPU，6 个数学 benchmark 评测。

**Table 2 的核心数字（training-aligned setting）:**

| Model | 最强 baseline avg | STAPO avg | 提升 |
|---|---|---|---|
| 1.7B | 33.64% (20-Entropy) | 38.18% | +13.5% relative |
| 8B | 57.63% (20-Entropy) | 58.76% | +2.0% relative |
| 14B | 60.77% (20-Entropy) | 64.38% | +5.9% relative |

**最 striking 的事实**：整个训练过程中，被 mask 的 token 比例始终 < 0.01%。换句话说，STAPO 用 **0.01% 的干预**换来了 **7% 的平均提升**（abstract 里报的数字，跨三个 model size 平均）。这个 leverage ratio 在 RL tuning 里非常罕见。

**Training dynamics（Figure 3）**：
- GRPO：entropy collapse（掉到接近 0，model 变得极度 confident 但其实 quality 下降）
- 20-Entropy / JustRL：entropy explosion（entropy 飙升，输出变得随机、repetitive）
- STAPO：entropy 平稳，既不 collapse 也不 explosion

**Ablation（Figure 5）的关键 insight**：
- 如果只 mask low-prob token（不看 entropy）：性能**下降**。因为很多 low-prob token 是有意义的 exploration。
- 如果 mask low-prob + **high**-entropy token：8B/14B 还行，1.7B **catastrophic collapse**。小模型依赖 high-entropy token 做 exploration。
- STAPO（low-prob + **low**-entropy + pos adv）：唯一在所有 scale 上都涨的。

这个 ablation 说明：**"low probability" 本身不是问题，"low probability + low entropy" 才是问题**。前者是探索，后者是 noise。STAPO 的精髓是用 entropy 这个维度把两者区分开。

---

## Spurious Token 长什么样

Figure 6b 的 word cloud + Appendix D 的案例分类很直观。三类：

**Category I: Uncommon Syntax**
该写 "removed"（85.53% prob）的地方，写成了 "broken"（0.05% prob）。语法没错，意思也对，但严重偏离标准术语。RL 强化这种 token 会让 model 的语言 drift away from canonical math language。

**Category II: Hallucinations & Math Errors**
Case 3 特别经典：model 写 $6901 = 67 \times 103 - 1$。其实 $67 \times 103 = 6901$，多减 1 是错的。但因为后面某个地方又"圆回来了"，最终 boxed answer 恰好正确，于是这个错误的 "-" token 被 positive advantage 强化。下次 model 可能更倾向于写这种"中间错但最后对"的奇怪推导。

**Category III: Formatting Errors**
LaTeX 里该有空格的地方没空格，比如 "$11^3+10^3+9^3$" 写成 "$11^3+10^3+9^3$"（"3" 后面紧跟 "+"，没空格）。LaTeX renderer 会自动补空格，视觉上看不出来，但 model 学到了一个 non-standard token sequence。这种错误最隐蔽，因为 rule-based verifier 完全检测不到。

Reference: paper Appendix D, Table 4-6 有完整的 case 列表

---

## 我觉得这篇 paper 真正的贡献

**Conceptual contribution > algorithmic contribution**。

Algorithm 本身非常简单——一个 binary mask，15 行伪代码。真正有价值的是 **diagnosis**：把 RL instability 的来源定位到 token level，并且给出一个可操作的特征组合（low prob + low entropy + pos adv）。

之前的 work 要么在 macro level 调（entropy regularization、KL penalty、clip range），要么在 sample level 调（reweighting、sample selection）。STAPO 在 token level 调，而且干预极小（0.01%）。这种"高杠杆"的干预说明：**instability 不是分布式的问题，是 localized 的问题**。

这跟传统 RL 里"value function high-variance samples"的直觉一致——少数 outlier sample dominate gradient。STAPO 把这个直觉 port 到 LLM RL，并且用 entropy + probability 把 outlier 的特征刻画得更精细。

Reference:
- Yang et al. 的原始 low-prob token 分析: https://arxiv.org/abs/2505.12929
- Lp-Reg (类似的思路但只用 probability): https://arxiv.org/abs/2510.03222
- 20-Entropy (反方向，强调 high-entropy token 的重要性): https://arxiv.org/abs/2506.01939

---

## 跟其他方向的联想

### 与 Process Reward Model 的关系

如果有一个完美的 Process Reward Model（PRM, https://arxiv.org/abs/2305.20050），sequence-level reward 会被分解到每个 token，spurious tokens 自然拿到低 reward，问题自动解决。

但 PRM 难训、容易 reward hack、需要额外标注。STAPO 的优雅之处是**不需要 PRM，只用 model 自己的 entropy 和 probability signal 就能识别 spurious tokens**。代价是只能处理"sequence 对但 token 坏"的 case，处理不了"sequence 错但某些 token 好"的 case。

完整版的 credit assignment 可能需要 PRM + STAPO 的结合：PRM 给 token-level reward，STAPO 的 mask 机制处理 PRM 也识别不出的 distributional anomaly。

### 与 DPO 的关系

DPO (https://arxiv.org/abs/2305.18290) 绕过显式 reward model，直接用 preference data 训练。STAPO 没绕过 reward，但绕过了 sequence-level reward 的**粗粒度**——在 token level 做 selective masking，相当于一种 implicit 的 credit assignment。

两者精神类似：**reward signal 太粗，就在更细的粒度上做 sanity check**。

### 与"Goodhart's Law"的关系

Spurious tokens 本质上是 **microscopic reward hacking**。Reward function 是 sequence-level 的，verifier 只看 final answer 对不对。在这个粗粒度 reward 下，某些 token 的更新方向和真正的 reasoning quality 脱钩——model 学会了"凑出正确答案的奇怪路径"。

STAPO 引入了一个 token-level 的 sanity check：即使 reward 说"对"，如果这个 token 本身的 distributional 特征不健康（low prob + low entropy），就拒绝强化它。

这跟 alignment 文献里"reward model over-optimization"的讨论是同一个 family 的问题，只不过发生在 token level。

### 与 Outlier Gradient Hypothesis 的关系

传统深度学习里有个经验观察：少数 outlier sample 会 dominate gradient，导致 training instability。各种 gradient clipping、sample reweighting 都在处理这个。

STAPO 在 LLM RL 里识别了一种**特定的 outlier**——不是 loss 大的 outlier，而是 distributional 特征（low prob + low entropy + pos adv）导致的 outlier。这种 outlier 的危险之处在于：它们的 loss 看起来正常（因为被 clip 了），但 gradient direction 是错的（强化了 spurious pattern）。

### 可能的扩展：Negative samples 里的 spurious tokens

Paper 在 conclusion 里承认只分析了 correct responses。一个自然的扩展是**错误 response 里的"好 token"**。

比如 response 整体错了（reward = -1），但前半段的 setup 是对的。这些"好 token"会被负 advantage 惩罚，这显然不合理。如果能识别它们（可能特征是 high prob + high entropy + negative advantage？）并 mask 掉，sample efficiency 会进一步提升。

这其实就是 token-level credit assignment 的完整版：sequence 对 → mask 坏 token（STAPO 做的），sequence 错 → mask 好 token（未来工作）。

### 与 Information Bottleneck 的可能联系

如果把 RL 看作一个 information bottleneck 问题——maximize reward subject to minimal policy change——那 spurious tokens 就是**高 information cost（-log π 很大）+ zero reward contribution** 的最坏 case。STAPO 的 mask 相当于在 information-cost-vs-reward 的平面上砍掉左上角的 outlier。

这个角度可能能给出一个更 principled 的 derivation：从 information bottleneck objective 出发，自然推出"low prob + low entropy + pos adv"的 mask criterion，而不是靠经验观察。这是个潜在的理论扩展方向。

---

## 一些 skepticism

公平地说，这篇 paper 也有几个地方我会追问：

**1. $\tau_p = 0.002$ 的来源**
Paper 没给系统的 grid search。这个值是怎么定的？对不同 vocab size、不同 model scale 是否需要调整？如果 vocab 从 15 万变到 3 万（比如某些 specialized model），这个 threshold 还对吗？

**2. 只在 math reasoning 上验证**
Code generation 里的 spurious token 形态完全不同——错误的 API name、unused variable、syntax error。mask criterion 可能需要 re-design。Paper 在 conclusion 里也承认了。

**3. Section 5.1 和 5.3 关于 $\tau_h$ 的描述不一致**
5.1 说 $\tau_h = 80\%$，5.3 说"optimal 是 20%，提到 80% 性能下降"。这两个数字对不上，要么是 typo，要么是描述方式不同（一个是 quantile percentile，一个是 mask 比例）。建议直接看 code 确认。Code 没在 paper 里给 link，但用了 veRL framework，可能后续开源。

**4. Long-term drift**
Mask 掉 token 的 gradient，意味着这些 position 的 policy 不会被 update。长时间训练下，这些 position 的 policy 可能 drift away from 一个 reasonable distribution。Paper 没显式处理这个，因为 0.01% 的比例确实小，但 scale 到更长训练时可能是个问题。

**5. Theoretical bound 的 tightness**
Theorem 3.1 的 upper bound 在 peaked distribution（实际 LLM 的大部分 step）上比较 loose。Paper 主要用 lower bound 来 argue"gradient 会被放大"，这是合理的，但 upper bound 部分的理论价值有限。

---

## 最终的 takeaway

如果你只能记住一件事，记住这个：

**RL instability in LLM fine-tuning is highly localized. A tiny fraction (0.01%) of tokens with a specific pathological signature (low prob + low entropy + positive advantage) can dominate gradient updates and destabilize training. Silencing them is enough.**

这个 insight 的价值不在于 STAPO 这个具体算法，而在于它揭示的 **token-level structure of RL instability**。后续的 RL fine-tuning work，不管是不是叫 STAPO，大概率都会需要处理这个问题——要么用 mask，要么用 reweighting，要么用 PRM，但都绕不开"少数 pathological token dominate gradient"这个微观结构。

Reference 汇总:
- STAPO paper (2026.02, 暂无 arXiv link，可能是 Tsinghua + DiDi 内部)
- DAPO: https://arxiv.org/abs/2503.14476
- GRPO: https://arxiv.org/abs/2402.03300
- JustRL: https://arxiv.org/abs/2512.16649
- 20-Entropy: https://arxiv.org/abs/2506.01939
- Yang et al. low-prob analysis: https://arxiv.org/abs/2505.12929
- Lp-Reg: https://arxiv.org/abs/2510.03222
- Entropy mechanism: https://arxiv.org/abs/2505.22617
- Qwen3: https://arxiv.org/abs/2505.09388
- veRL: https://github.com/volcengine/verl
- DPO: https://arxiv.org/abs/2305.18290
- PRM: https://arxiv.org/abs/2305.20050
- PPO: https://arxiv.org/abs/1707.06347

---

# STAPO: Stabilizing RL for LLMs by Silencing Rare Spurious Tokens 详解

这篇paper来自清华 + DiDi Voyager Labs（2026年2月），针对RL fine-tuning LLM时一个特别细的稳定性问题：sequence-level reward会让极少量（约0.01%）的"spurious tokens"获得不成比例的大梯度更新，从而破坏训练稳定性和reasoning quality。我会把理论、算法、实验全部讲透，并尽量build你的intuition。

参考链接:
- arXiv: 暂未找到对应的paper link，可能是2026年的新paper
- DAPO (baseline之一): https://arxiv.org/abs/2503.14476
- GRPO/DeepSeekMath: https://arxiv.org/abs/2402.03300
- 20-Entropy: https://arxiv.org/abs/2506.01939
- JustRL: https://arxiv.org/abs/2512.16649
- Yang et al. "Do not let low-probability tokens over-dominate": https://arxiv.org/abs/2505.12929
- Lp-Reg (Low-Probability Regularization): https://arxiv.org/abs/2510.03222
- Entropy mechanism paper: https://arxiv.org/abs/2505.22617
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- veRL framework: https://github.com/volcengine/verl

---

## 1. 核心Idea的Intuition

想象一个LLM生成了一段正确的数学解答，最后拿到了+1 reward。在sequence-level reward assignment下，response中所有token都共享这个正的advantage，包括那些"凑巧"出现但对reasoning毫无贡献、甚至语义错误的token。

问题在于：这些token往往是**低概率**的（模型本身就不太想生成它们），同时由于分布尖锐，**局部entropy也低**（模型"装作"很确定），加上正的advantage（因为sequence对了），三者组合就会触发一个病理性的更新模式——梯度被异常放大，去强化一个本不应该存在的pattern。

类比一下Figure 1(a)的vocalist比喻：一个合唱团整体和声完美，但有一个走音的soprano在某个瞬间冒出来一个不和谐的高音——观众（reward verifier）只听到了整体正确，给所有人鼓掌，于是这个走音的soprano也得到了鼓励，下次还会更响地走音。

---

## 2. 理论分析：为什么low prob + low entropy + positive advantage是危险的

### 2.1 Theorem 3.1 — Policy Gradient Norm Bounds

这是整篇paper的理论基石。考虑group-style objective在第t步对第i个样本的更新，gradient w.r.t. logits **a** ∈ ℝ^|V| 的squared L2-norm被如下bound：

$$
|w_{i,t}|^2 \left( 1 - 2\pi_\theta(y_{i,t} \mid x, y_{i,<t}) + e^{-\mathcal{H}(\pi_\theta)} \right) \le \|\nabla_a \mathcal{J}(y_{i,t})\|^2 \le |w_{i,t}|^2 \left( 2 - 2\pi_\theta(y_{i,t} \mid x, y_{i,<t}) - C_V \mathcal{H}(\pi_\theta)^2 \right)
$$

**变量解释:**
- $w_{i,t}$: 权重项。如果在clip区外（$\hat{A}_i>0 \wedge \rho_{i,t}>1+\epsilon$ 或 $\hat{A}_i<0 \wedge \rho_{i,t}<1-\epsilon$），$w_{i,t}=0$（被clip掉）；否则 $w_{i,t} = \rho_{i,t} \hat{A}_i$，即importance sampling ratio乘以advantage。
- $\pi_\theta(y_{i,t} \mid x, y_{i,<t})$: 当前token在policy分布下的概率。
- $\mathcal{H}(\pi_\theta)$: 当前policy的Shannon entropy。
- $C_V = \frac{|V|-1}{|V|(\ln|V|)^2}$: 与vocab size相关的常数，对15万级别的vocab约等于 $\frac{1}{(\ln|V|)^2} \approx \frac{1}{(11.9)^2} \approx 0.007$。

**Intuition拆解:**
- Lower bound中关键项是 $1 - 2\pi + e^{-\mathcal{H}}$。当 $\pi$ 小（target token概率低），$1-2\pi$ 变大；当 $\mathcal{H}$ 小（entropy低，分布尖锐），$e^{-\mathcal{H}}$ 变大。两者叠加 → 梯度下界显著变大。
- 这意味着即使advantage不大，只要target token稀有且local分布尖锐，gradient magnitude也会被"自动放大"。

**证明思路（Appendix B）:**
1. 先用Lemma A.1分解：$\|\nabla_a \mathcal{J}\|^2 = |w|^2 (1 - 2\pi + \sum_n \pi(v^n)^2)$，其中 $\sum_n \pi(v^n)^2$ 是collision probability。
2. Lower bound用Renyi entropy $H_2 = -\ln(\sum_n \pi(v^n)^2) \le \mathcal{H}$（Shannon entropy永远≥order-2 Renyi entropy），得到 $\sum_n \pi(v^n)^2 \ge e^{-\mathcal{H}}$。
3. Upper bound用一个discrete distribution偏离uniformity的bound: $\sum_n \pi(v^n)^2 \le 1 - C_V \mathcal{H}^2$。

### 2.2 Lemma 3.2 — Entropy Update Mechanism

来自paper [14]（https://arxiv.org/abs/2505.22617），描述entropy在一步natural policy gradient后的变化：

$$
\mathcal{H}(\pi_{\theta_{k+1}}(\cdot \mid x, y_{i,<t})) - \mathcal{H}(\pi_{\theta_k}(\cdot \mid x, y_{i,<t})) \approx -\eta \, \mathrm{Cov}_{y_{i,t} \sim \pi_{\theta_k}}\Big( \log \pi_{\theta_k}(\cdot \mid x, y_{i,<t}), \hat{A}_i \Big)
$$

**变量解释:**
- $\eta$: learning rate
- $\mathrm{Cov}$: covariance，在 $\pi_{\theta_k}$ 下采样 $y_{i,t}$ 的期望

**直觉:** 如果高概率token（log π大）正好对应正advantage，covariance为正，entropy会**下降**（policy变尖锐，collapse）；如果低概率token对应正advantage，covariance可能为负，entropy会**上升**（policy变平，explosion）。

对spurious tokens（低概率 + 正advantage）来说，它们会**主动push entropy向上**，这是entropy explosion的一个微观来源。

### 2.3 Lemma 3.3 — Learning Potential

Low-entropy tokens是模型已经"很确定"的区域，再更新收益有限；high-entropy tokens是有多个plausible continuation的"决策点"，是有效学习的位置。

### 2.4 三者合成的Token Phase Diagram (Table 1)

把entropy高/低、probability高/低、advantage正/负组合成 $2^3 = 8$ 种token类别：

| Token Prob | Advantage | Entropy | Gradient Norm | Entropy Change | Learning Potential |
|---|---|---|---|---|---|
| High | 负 | High | Low | ↑ | High |
| High | 正 | High | Low | ↓ | High |
| **Low** | **负** | **High** | **High** | **↓** | **High** (好的探索) |
| Low | 正 | High | High | ↑ | High (好的探索) |
| High | 负 | Low | Low | ↑ | Low |
| High | 正 | Low | Low | ↓ | Low |
| Low | 负 | Low | High | ↓ | Low |
| **Low** | **正** | **Low** | **High** | **↑** | **Low** ← Spurious! |

最后一行就是spurious token的"完美风暴"：
- **Gradient Norm = High**: 由Theorem 3.1，低prob + 低entropy → 大梯度。
- **Entropy Change = ↑**: 由Lemma 3.2，低prob + 正advantage → 推动entropy上升，可能引发explosion。
- **Learning Potential = Low**: 由Lemma 3.3，低entropy意味着模型已经"自以为确定"，继续更新没意义。

三个criterion都给出负面信号，这种token的更新纯属"扰动"。

---

## 3. Spurious Tokens的实证识别

### 3.1 Figure 2 — 经验上的聚类

作者在Qwen3-8B上跑JustRL setting，记录训练中所有token的 (probability, entropy) 二维分布（Figure 2a），token自然形成四个cluster，对应phase diagram的四个组合。

**Figure 2b的统计:** spurious tokens（低prob + 低entropy + 正advantage）的mean gradient norm比"高prob + 高entropy"baseline高约**+16.7%**。这是量化证据：稀有错误确实引发不成比例的更新。

### 3.2 Figure 2c — 三个典型例子

作者展示了三类典型的spurious tokens（Appendix D详细列出更多）:

- **Category I: Uncommon Syntax** — Case 2: 在描述图论"边的移除"时，模型用了 "broken" (prob 0.05%)，而Top-5是 "removed" (85.53%), "closed" (3.32%), "deleted" (2.58%)。语法上没错，但严重偏离标准术语。
- **Category II: Hallucinations & Math Errors** — Case 3: $6901 = 67 \times 103 - 1$。其实 $67 \times 103 = 6901$，多减了1是错的，但最终答案"恰好"对了，于是这个错误的"-" token被强化。
- **Category III: Formatting Errors** — Case 4: 在LaTeX summation中，"3" 后省略了空格" " token，渲染时LaTeX自动补空格所以视觉上看不出错，但policy学到了非标准序列。

### 3.3 Definition 3.4 — Spurious Tokens

> Spurious tokens are tokens that contribute negligibly to the correctness of a reasoning outcome but receive disproportionately large positive updates due to sequence-level reward assignment.

---

## 4. STAPO算法

### 4.1 S2T (Silencing Spurious Tokens) Mask

定义一个binary mask：

$$
\mathbb{I}_{i,t}^{\mathrm{S2T}} = \begin{cases} 0, & \text{if } \hat{A}_i > 0 \,\land\, \pi(y_{i,t}) < \tau_p \,\land\, \mathcal{H}_t < \tau_h \\ 1, & \text{otherwise} \end{cases}
$$

**变量解释:**
- $\tau_p$: probability threshold，论文中用 **fixed absolute value = 0.002**。
- $\tau_h$: entropy threshold，论文中用 **dynamic quantile = 80%**，即每个mini-batch内mask掉entropy最低的20% token。

**为什么τ_p用absolute value而τ_h用quantile?** （Remark 3.5）

这是很关键的设计选择。如果τ_p也用quantile（比如always mask掉最低20%概率的token），就会**不分情况地**砍掉20% token——包括那些高概率的合理token（比如某些必然出现的关键词）。这会破坏训练。而entropy因为是相对量（取决于当前step的uncertainty），用quantile更鲁棒。

直觉上：**probability有"绝对正确性"含义**（0.001就是真的稀有），**entropy只有"相对尖锐度"含义**（每个step内部比较）。

### 4.2 STAPO Objective

基于DAPO objective改造：

$$
\mathcal{J}_{\mathrm{STAPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\mathrm{old}}}} \left[ \frac{1}{\sum_{i=1}^G \sum_{t=1}^{|y_i|} \mathbb{I}_{i,t}^{\mathrm{S2T}}} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \mathbb{I}_{i,t}^{\mathrm{S2T}} \cdot \min\Big( \rho_{i,t}(\theta)\hat{A}_i, \, \mathrm{clip}(\rho_{i,t}(\theta), 1-\epsilon_{\mathrm{low}}, 1+\epsilon_{\mathrm{high}})\hat{A}_i \Big) \right]
$$

**与DAPO/Eq.(1)的两个关键区别:**
1. **Mask项** $\mathbb{I}_{i,t}^{\mathrm{S2T}}$ 乘在loss里，spurious tokens的loss贡献直接被置零。
2. **Normalization term** 改为 $\sum \mathbb{I}_{i,t}^{\mathrm{S2T}}$，只数"未被mask的token"。这很重要：如果不改normalization，mask的token会稀释loss scale，效果像降低learning rate；改了之后，只有真正"有效"的token参与平均，spurious tokens的影响被完全移除。

### 4.3 Algorithm 1 伪代码

```
输入: dataset D, initial policy π_θ, group size G=8, 
     thresholds τ_p=0.002, τ_h=80%, batch size B=256
1. 初始化 θ
2. for each iteration:
3.   θ_old ← θ (同步)
4.   采样prompts x^B ∼ D, 用π_θ_old生成 {y_1,...,y_G}^B (每个prompt 8 rollouts)
5.   计算advantages Â_i^B (用Eq.3，group内normalize)
6.   for each mini-batch of size B:
7.     for each (response, token) (i, t):
8.       获取 p_{i,t} = π_θ(y_{i,t}|x, y_{i,<t}) 和 h_{i,t} = H(π_θ(·|x, y_{i,<t}))
9.       I_{i,t}^silence ← 1 (默认保留)
10.      if Â_i > 0 AND p_{i,t} < τ_p AND h_{i,t} < τ_h:
            I_{i,t}^S2T ← 0 (标记为spurious)
11.    end
12.    用Eq.(2)更新θ (带mask)
13.  end
14. end
```

注意step 10的判断在**每个mini-batch内部**重新计算（quantile），所以是动态的。

### 4.4 Advantage计算 (Eq.3)

$$
\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t} \mid x, y_{i,<t})}, \qquad \hat{A}_i = \frac{R(x, y_i) - \mathrm{mean}(\{R(x, y_j)\}_{j=1}^G)}{\mathrm{std}(\{R(x, y_j)\}_{j=1}^G)}
$$

GRPO-style group normalization：每个prompt采样G=8个response，reward在group内做z-score normalize，所以**同一个sequence内所有token共享同一个advantage**——这正是spurious token问题的源头：sequence对了，所有token都拿到正advantage。

---

## 5. 实验

### 5.1 Setup

- **Training dataset**: DAPO-Math-17K (https://huggingface.co/datasets/Open-Reasoner-Zero/DAPO-Math-17k)
- **Models**: Qwen3-1.7B / 8B / 14B base
- **Baselines**: GRPO, 20-Entropy, JustRL
- **Hardware**: 64× NVIDIA H20 GPUs
- **Training hyperparameters**:
  - Batch size: 256
  - Mini-batch: 64
  - 4 PPO updates per rollout
  - LR = 1e-6, AdamW
  - Warmup: 10 steps
  - Max response length: 15k tokens
  - Group size G = 8 rollouts per prompt
  - Clip range: [0.8, 1.28] (DAPO asymmetric clip-higher)
  - No KL penalty, no value network
- **STAPO hyperparameters**: $\tau_p = 0.002$, $\tau_h = 80\%$ quantile
- **Benchmarks**: AIME24, AIME25, AMC23, MATH500, Minerva, OlympiadBench
- **Evaluation configs**:
  - Training-aligned: $\rho_T=1.0$, top-p=1.0
  - JustRL setting: $\rho_T=0.7$, top-p=0.9
  - N=4 (MATH500/Minerva/OlympiadBench), N=32 (AIME24/AIME25/AMC23) independent samples per problem
  - CompassVerifier-3B 用于二次校验rule-based verification

### 5.2 主结果 (Table 2)

| Model | Method | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | **Avg** |
|---|---|---|---|---|---|---|---|---|
| **1.7B** | GRPO | 6.88 (10.83) | 4.06 (4.06) | 40.47 (41.17) | 68.60 (67.15) | 32.44 (32.53) | 30.64 (30.64) | 30.52 (31.06) |
| 1.7B | 20-Entropy | 13.33 (13.85) | 8.44 (7.60) | 46.56 (44.53) | 69.85 (69.10) | 28.03 (29.04) | 35.65 (34.35) | 33.64 (33.08) |
| 1.7B | JustRL | 8.85 (14.58) | 6.25 (10.93) | 45.39 (51.09) | 62.70 (69.75) | 22.61 (28.89) | 29.15 (34.16) | 29.16 (34.90) |
| 1.7B | **STAPO** | **17.40 (16.04)** | **15.42 (14.27)** | **55.94 (52.42)** | **73.55 (70.30)** | 28.22 (29.42) | **38.54 (37.87)** | **38.18 (36.72)** |
| **8B** | GRPO | 31.25 (32.40) | 24.69 (24.17) | 75.23 (74.14) | 88.90 (88.85) | 55.88 (53.58) | 61.50 (58.34) | 56.24 (55.25) |
| 8B | 20-Entropy | 31.25 (35.10) | 27.50 (27.39) | 79.92 (80.00) | 89.85 (89.85) | 54.78 (54.23) | 62.50 (60.39) | 57.63 (57.83) |
| 8B | JustRL | 25.21 (34.06) | 21.98 (26.04) | 73.52 (81.48) | 84.90 (87.55) | 47.33 (51.03) | 51.26 (56.57) | 50.70 (56.12) |
| 8B | **STAPO** | **33.44 (39.48)** | **28.65 (29.37)** | **79.92 (80.23)** | **90.40 (91.40)** | **57.17 (55.70)** | **62.98 (63.32)** | **58.76 (59.92)** |
| **14B** | GRPO | 42.19 (40.52) | 30.31 (32.40) | 81.95 (78.12) | 90.05 (92.00) | 55.15 (57.54) | 64.76 (64.28) | 60.74 (60.81) |
| 14B | 20-Entropy | 42.08 (51.25) | 34.06 (39.48) | 84.45 (87.73) | 91.65 (92.30) | 51.10 (54.78) | 61.30 (63.95) | 60.77 (64.92) |
| 14B | JustRL | 37.40 (52.08) | 26.56 (38.96) | 81.95 (89.61) | 87.65 (94.25) | 43.84 (58.92) | 57.01 (68.81) | 55.74 (67.11) |
| 14B | **STAPO** | **46.98 (54.27)** | **35.21 (41.67)** | **87.11 (90.62)** | **92.45 (93.85)** | **59.47 (59.93)** | **68.21 (71.62)** | **64.38 (68.66)** |

括号内是JustRL setting（$\rho_T=0.7$, top-p=0.9）的结果。

**关键观察:**
- **1.7B**: STAPO比最强baseline (20-Entropy) 平均高 **+13.50%** relative。1.7B这种小模型尤其受益，因为小模型的representation redundancy少，spurious token的扰动影响更大。
- **8B**: STAPO全面领先。
- **14B**: STAPO在training-aligned setting下比最强baseline高约**+5.94%** relative。
- 在JustRL setting (greedy-ish decoding)下，STAPO依然领先，但gap变窄。作者解释：baseline因为entropy不稳定，分布有长尾，top-p=0.9的decoding正好砍掉了不稳定部分，相当于"作弊"；而STAPO本身分布就稳定，不需要这种decoding heuristic的帮助。

### 5.3 Training Dynamics (Figure 3)

跨三个scale的训练曲线显示:
- **20-Entropy & JustRL**: 经常出现entropy explosion（unstable, oscillatory high entropy）。
- **GRPO**: entropy collapse（rapidly下降到接近0）。
- **STAPO**: entropy保持平稳，既不collapse也不explosion，同时training reward最高，AIME24 acc稳步上升。

**最惊人的数字**: 整个训练过程中，被mask的spurious tokens比例始终在 **0.01% 以下**（Figure 6a）。早期14B模型短暂时升至0.03%，之后迅速回落到0.01%以下。**用0.01%的干预换来了7%的性能提升**——这表明RL instability的"病灶"高度集中。

### 5.4 Hyperparameter Sensitivity (Figure 4)

**τ_p sensitivity (Qwen3-1.7B):**
- τ_p 从 2e-3 增到 2e-1，AIME24 acc 从 17.4% 暴跌到 7.2%。
- 解释：太aggressive的probability threshold会砍掉真正稀有但语义重要的token（比如某些关键数字、罕见但正确的术语）。
- 这印证了为什么用absolute value而不是quantile：quantile会自动收紧，误伤有效token。

**τ_h sensitivity:**
- 论文最优设置是 τ_h = 20% quantile，意味着mask掉**底部80%的低entropy tokens**——而不是底部20%。
- 等等，这里需要核对一下。重新读Remark 3.5和Section 5.3...

仔细看Section 5.3:
> "In our optimal configuration (τ_h = 20%), we mask the bottom 80% of low-entropy tokens. As τ_h increases, performance consistently declines. Raising τ_h from 20% to 80% reduces AIME24 accuracy from 17.4% to 11.6%..."

我的理解：τ_h是"分位数threshold"。"mask the bottom 80%"指的是**80% quantile的entropy值作为下界**——entropy低于这个值的token被认为是低entropy token。Section 5.1中写的"τ_h = 80% for the entropy threshold"应该是指quantile percentile=80%。

把两个一致起来读：τ_h作为quantile百分位是80%意味着mask掉entropy最低的20% tokens（在80%分位线以下的token是80%，但我们要mask的是"低entropy"，所以应该是bottom 20%以下）...

这里有歧义。再仔细看Algorithm 1 line 10: `if Â > 0 ∧ p < τ_p ∧ h < τ_h`. 所以τ_h是entropy的上限，h < τ_h表示token的entropy低于threshold。如果τ_h是80% quantile，那80%的token的entropy都低于这个值——按算法这些token都可能被mask（如果同时满足其他条件）。

Section 5.3的实际描述说"mask the bottom 80% of low-entropy tokens"，即80% quantile作为threshold，h < 这个值的token（占80%）都算低entropy候选。这和Algorithm一致。但作者也说"raising τ_h from 20% to 80% reduces accuracy"——意思是把quantile从20%提到80%（mask更多低entropy token）反而降低性能。

这看起来有点矛盾。我推测Section 5.1中"τ_h = 80%"对应"mask bottom 80%"，而Section 5.3讨论的"raising τ_h from 20% to 80%"是错误描述或笔误。合理推断：**最优是mask底部20%（τ_h=20% quantile），过度mask（提到80%）会损害性能**。或者反过来——总之这是一个不太一致的地方，建议看原文确认。

实际更合理的reading：Figure 4b的x轴应该是τ_h quantile，从20%到80%。Optimal在20%（即只mask底部20%的低entropy tokens），提到80%（mask底部80%）会损害性能。这和Section 5.1的"τ_h = 80%"冲突——可能是paper内部typo。

这里我标注一下存疑。

### 5.5 Masking Strategy Ablation (Figure 5)

测试三种mask策略:
1. **Only probability threshold** (mask 低prob token): 全面逊于baseline。说明indiscriminately砍低prob token会丢失有意义的稀有探索。
2. **High entropy + low prob** (mask 高entropy 低prob token): 在8B/14B上competitive，但在1.7B上**catastrophic collapse**。说明大模型有structural redundancy能扛住high-entropy noise，小模型依赖这些token做exploration。
3. **STAPO (low entropy + low prob + pos adv)**: 唯一在**所有scale**上都比baseline好的方法。

这个ablation非常informative——它说明"低prob"本身不是问题，"低prob + 高entropy"（探索性稀有token）和"低prob + 低entropy"（spurious token）是截然不同的东西。STAPO的精髓是**用entropy这个额外维度区分两者**。

### 5.6 Word Cloud Analysis (Figure 6b-c)

**被mask的spurious tokens高频词:** 数字如 "4", "1", "9"；数学符号如 "/", "+", "-"; 过渡词如 "Wait", "But", "Since"。这些token在correct response里出现，但贡献微薄，加上大梯度就会overfit到不稳定模式。

**保留的normal tokens高频词:** "Let", "find", "we", "can"——结构化数学reasoning vocabulary，承担logical skeleton。

---

## 6. 与相关工作的关系

### 6.1 Yang et al. "Do not let low-probability tokens over-dominate" (https://arxiv.org/abs/2505.12929)

这篇STAPO的理论基石来自这篇工作。Yang et al.首次证明了low-prob token的gradient dominance现象，提出用importance sampling ratio reweighting来抑制。STAPO的Theorem 3.1基本就是基于Yang et al.的bound加上entropy维度的扩展。

### 6.2 Lp-Reg (https://arxiv.org/abs/2510.03222)

Lp-Reg试图在保留有意义的稀有token（exploration）同时filter掉noise，但用的是scalar probability threshold。STAPO的作者在related work里批评Lp-Reg：没有joint treatment of confidence和probability，无法区分"informative exploration"和"aleatoric noise"。

### 6.3 DAPO (https://arxiv.org/abs/2503.14476)

DAPO引入了四个trick：dynamic sampling、token-level normalization、clip-higher、overlong reward shaping。STAPO的baseline就是基于DAPO的clip-higher + token normalization（这是JustRL configuration），STAPO在这个基础上加S2T mask。

### 6.4 20-Entropy (https://arxiv.org/abs/2506.01939)

20-Entropy的发现是"high-entropy minority tokens"驱动有效RL learning，所以选择性地regularize高entropy token。STAPO指出了问题的另一面：低entropy + 低prob的token才真正有害，high-entropy token反而是learning potential高的位置。两者方向相反。

### 6.5 Entropy mechanism paper (https://arxiv.org/abs/2505.22617)

这篇是Lemma 3.2的来源。它建立了entropy dynamics的Cov(log π, A)公式，但没具体分析token-level微观原因。STAPO在它的基础上细化到token level。

---

## 7. 我的思考和Intuition Building

### 7.1 为什么"低prob + 低entropy"听起来矛盾？

第一眼看上去，"低prob"意味着模型不 confident，而"低entropy"意味着模型很 confident——这两个怎么同时发生？

关键在于**local entropy和target token probability是两个不同的量**。Local entropy是整个分布 $\pi(\cdot|x, y_{<t})$ 的Shannon entropy，反映模型在这个位置的整体不确定性；target token probability是 $\pi(y_{i,t}|...)$，是这个分布下特定token的概率。

可以想象一个分布：99.5% mass在某个token A上，0.5% mass均匀分散在剩下的15万个token上。这时local entropy非常低（因为A dominate），但如果你恰好采样到了0.5%中的某个稀有token B，它的probability就极低。

实际中这就是spurious token的scenario：模型在这个位置本来"应该"输出A，分布尖锐（低entropy），但因为sampling温度、stochasticity等原因，刚好采到了某个稀有token。这个token被spurious地放大了梯度。

### 7.2 为什么正advantage是必要条件？

如果advantage是负的（response错了），低prob token的"梯度放大"其实在帮模型把概率**推低**——这是corrective，有益的。但如果advantage是正的（response对了），梯度放大就在**推高**这个spurious token的概率——这强化了错误模式。

所以STAPO只mask $\hat{A} > 0$ 的case，保留 $\hat{A} < 0$ 的corrective updates。

### 7.3 与"Goodhart's Law on Token Level"的联系

这让我想到强化学习的alignment文献里关于"reward hacking"的讨论。在sequence-level reward + token-level gradient的setup下，spurious tokens就是一种**microscopic reward hacking**：reward function粗粒度，让某些token的更新方向与真正的reasoning quality脱钩。STAPO本质上是引入了一个**token-level的局部sanity check**。

类似的精神在RLHF的preference modeling里也有——比如DPO ([https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290))也试图绕过显式reward model直接用preference data。STAPO则是绕过sequence-level reward的粗粒度，直接在token level做selective masking。

### 7.4 与Importance Sampling和Off-policy Correction的关系

$w_{i,t} = \rho_{i,t} \hat{A}_i$ 这个weight其实就是standard PPO的surrogate weight。Theorem 3.1说gradient norm被 $|w_{i,t}|^2$ 和 (1-2π+e^{-H}) 共同determine。前者由clip控制，后者是论文的新洞察。

DAPO的clip-higher（$\epsilon_{\mathrm{high}} > \epsilon_{\mathrm{low}}$）实际上扩大了 $\rho$ 可以变大的范围，从而促进exploration；但这同时意味着low-prob token在pos adv下$\rho$容易变大，进一步放大gradient。STAPO在另一端通过entropy/probability mask来平衡。

### 7.5 一个可能的扩展：spurious tokens in negative samples?

作者在Conclusion里提到：当前分析focus on tokens in **correct responses**。一个自然扩展是研究**错误response里的spurious tokens**——比如response整体错了，但其中某些token（比如开头的正确setup）本应被强化。如果把它们的advantage也mask掉（或重新归因），可能进一步提升sample efficiency。

这个方向有点像credit assignment的精细化：sequence-level reward到token-level credit assignment是个经典难题。STAPO做了一个简化版——只处理"sequence对但token坏"的case，未处理"sequence错但token好"的case。完整版可能需要某种token-level reward decomposition。

### 7.6 一个理论可能：information-theoretic justification

可以更formal地论证：spurious tokens的"信息量" $\approx -\log \pi(y_{i,t})$ 很大，但它们的"对正确性的贡献"$\approx 0$。如果把RL看作"信息瓶颈"——maximize reward subject to minimal change in policy——那么spurious tokens就是高information cost + zero reward contribution的最坏case。这可能是另一个build intuition的角度。

### 7.7 与RLOO, ReMax等variance reduction方法的关系

RLOO (REINFORCE Leave-One-Out, https://arxiv.org/abs/2402.14740)和ReMax (https://arxiv.org/abs/2310.10505)都是通过leave-one-out baseline来reduce variance。STAPO没直接处理variance，而是识别"pathological high gradient"的来源。一个有意思的组合：在leave-one-out baseline下，spurious tokens的advantage估计可能更准确——因为它们在多个rollout中可能出现也可能不出现，留下来的rollout会pull baseline更高，从而自动减小spurious token的advantage。这可能是个有趣的研究方向。

### 7.8 关于0.01%的spurious token比例

这个数字真的很小。在15k token的response中，0.01%意味着平均1.5个token per response被mask。在batch size 256, G=8 rollout的setup下，每个training step大概有 $256 \times 8 \times 15000 \approx 30M$ tokens，0.01%就是3000个token被mask掉。3000个token的mask能让训练稳定性和最终性能显著提升——这再次说明instability是高度localized的。

这跟"outlier in RL"的文献有共鸣——比如value function的high-variance samples、reward shaping里的spurious correlations。但在LLM RL这个scale下，能精准定位到token level，是非常impressive的工程+理论结合。

### 7.9 与Curiosity-Driven Exploration的对比

在传统RL里，curiosity-driven exploration (ICM, RND) 鼓励agent去访问high prediction error的state。这里STAPO反向操作——把低entropy + 低prob的"模型已确定但实际错误"的token silence掉。这两者其实不冲突：high entropy的token鼓励exploration（20-Entropy精神），low entropy但rare spurious token discourage（STAPO精神）。

---

## 8. 可能的批评和改进方向

### 8.1 Threshold sensitivity的robustness

$\tau_p = 0.002$ 是怎么定的？论文没给出系统的sensitivity曲线在更细的网格上（比如1e-4到1e-2之间）。对不同的vocab size、不同的training stage，这个值应该自动adjust吗？可能需要一个adaptive schedule。

### 8.2 entropy threshold的quantile含义

前面提到的Section 5.1和5.3的描述不一致——paper内部可能有typo。建议看code或者直接联系作者确认。Codebase link没在paper里明说，但作者用了veRL framework (https://github.com/volcengine/verl)，可能后续会开源。

### 8.3 只在数学reasoning上验证

Conclusion里也承认了：没在code generation、其他reasoning domain上验证。Spurious tokens在code任务里的形态可能完全不同（比如syntax错误、未使用的变量、错误的API name），mask criteria可能需要re-design。

### 8.4 与reference policy drift的interaction

mask掉token的gradient，是否会让policy在某些位置drift away from reference policy（即使是implicit reference）？STAPO没用KL penalty，所以这点没显式处理。长时间训练下，spurious token位置的分布可能变得unanchored。

### 8.5 Theoretical bound的tightness

Theorem 3.1的upper bound用了 $\sum_n \pi(v^n)^2 \le 1 - C_V \mathcal{H}^2$，这个bound在distribution接近uniform时tight，接近peaked时loose。对实际训练中的LLM policy（大部分step都是peaked distribution），upper bound可能比实际gradient大很多。但lower bound反而可能更informative——这正是paper的核心使用方式。

### 8.6 Comparison with token-level reward models

如果有一个好的process reward model (PRM, https://arxiv.org/abs/2305.20050)，sequence-level reward会被分解到token level，spurious tokens自然拿到低reward，问题自动解决。STAPO的优雅之处是不需要PRM，只用entropy/prob signal就能identify spurious tokens。但代价是PRM能更精细地处理credit assignment。

---

## 9. 总结性Intuition

把STAPO的核心idea用一句话总结：**RL fine-tuning LLM的不稳定性，本质上不是"很多token都small wrong"，而是"极少数token大错特错"**。这些"大错特错"的token的特征是模型本不想生成它们（low prob），分布却看似确定了（low entropy），又赶上整个response对了（positive advantage）——三者合谋，让梯度更新变成"用大力气强化一个本不该出现的pattern"。

STAPO的解法极简：识别它们（用一个binary mask），silence它们（gradient置零），保留其他token的正常学习。0.01%的干预换7%的提升，这种"杠杆率"在RL tuning里非常罕见，也指向了训练不稳定的真正微观结构。

这让我想到Karpathy你自己关于"micrograd"的精神——理解训练动态要从微观的gradient flow入手，而不是宏观的loss curve。STAPO正是这种精神的体现。

---

## 10. 进一步阅读

- DAPO原paper: https://arxiv.org/abs/2503.14476
- GRPO原paper (DeepSeekMath): https://arxiv.org/abs/2402.03300
- Yang et al. low-prob token analysis: https://arxiv.org/abs/2505.12929
- 20-Entropy: https://arxiv.org/abs/2506.01939
- JustRL: https://arxiv.org/abs/2512.16649
- Lp-Reg: https://arxiv.org/abs/2510.03222
- Entropy mechanism paper (Lemma 3.2 source): https://arxiv.org/abs/2505.22617
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- veRL framework: https://github.com/volcengine/verl
- DAPO-Math-17K dataset: https://huggingface.co/datasets/Open-Reasoner-Zero/DAPO-Math-17k
- DPO原paper: https://arxiv.org/abs/2305.18290
- PRM (Process Reward Models): https://arxiv.org/abs/2305.20050
- RLOO: https://arxiv.org/abs/2402.14740
- ReMax: https://arxiv.org/abs/2310.10505
- PPO原paper: https://arxiv.org/abs/1707.06347
- CompassVerifier: https://aclanthology.org/2025.emnlp-main.1982/
- AIME24 (NuminaMath): https://huggingface.co/datasets/Numina/Math-CoT
- AIME25: https://huggingface.co/datasets/opencompass/AIME2025
- MATH500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- OlympiadBench: https://aclanthology.org/2024.acl-long.836/

希望这个详解能让你build起对STAPO的完整intuition。如果哪个部分想深入（比如Theorem 3.1的证明细节、不同masking策略的代码实现、或者扩展到其他domain的设想），可以继续聊。
