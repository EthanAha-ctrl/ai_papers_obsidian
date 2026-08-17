---
source_pdf: Your Group-Relative Advantage Is Biased.pdf
paper_sha256: bac0fdd810c0c7611db634a48be9757a172a1241d98667535548615490148258
processed_at: '2026-08-13T06:35:18-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

Okay so let me break this down in plain terms.

## 一句话版本

GRPO这个算法有个hidden bug：它丢掉"全对"或"全错"的samples只保留mixed results，这个filtering会systematically低估难题的learning signal、高估简单题的learning signal。结果就是你的model在medium difficulty上spinning wheels，hard problems永远学不会，easy problems被over-rewarded。这篇paper发现这个bug并给了一个cheap fix。

## 为什么这是个bug？

Imagine you're teaching a kid math. You give them 8 problems of similar difficulty. If they get all 8 right or all 8 wrong, you say "okay no signal here, skip this batch." Sounds reasonable right? That's literally what GRPO does.

But think about what this actually does to your data distribution. For a really hard problem where the kid only has 10% success rate:
- 43% of the time they get 0/8 right → discarded
- Almost never get 8/8 right
- So the batches you actually train on are biased toward "kid did better than expected" cases

The baseline (average score within the batch) gets artificially inflated because you threw away all the 0/8 cases. So when the kid gets 1/8 right, instead of thinking "wow that's amazing, 10% vs expected 10%, that correct answer deserves big reward", GRPO thinks "1 correct out of 8, baseline should be ~2-3, so this correct answer is actually below average, small reward."

You're literally penalizing the kid for getting hard problems right. That's messed up.

Symmetric thing happens for easy problems. Kid has 90% success rate, 43% of batches are 8/8 (discarded), so the surviving batches have artificially low baseline, wrong answers get over-penalized, correct answers get over-rewarded.

## 为什么之前没人发现？

Honestly I think it's because the RLVR community has been moving so fast. GRPO came out of DeepSeekMath, everyone jumped on it, R1 dropped, everyone went "wow RLVR works," and we're all kind of cargo-culting the algorithm without doing the basic statistical analysis.

The bias is subtle in the sense that:
1. It only shows up when you condition on the non-degenerate event
2. The magnitude depends on group size G and difficulty p_t
3. For p_t = 0.5 (medium difficulty) the bias is zero, so if you only looked at medium problems you'd never notice

But it's huge in the sense that:
1. With G=8 (what everyone uses), for hard problems p_t < 0.125, the underestimate is deterministic—100% probability
2. The hard problems are exactly the ones you need to learn from to push capability frontier

## 数学直觉

The key formula is:

$$\mathbb{E}[\hat{p}_t | S] = \frac{p_t(1 - p_t^{G-1})}{1 - (1-p_t)^G - p_t^G}$$

Where:
- $p_t$ = true success probability
- $\hat{p}_t$ = your group baseline estimate
- $S$ = event that not all 8 are same (the non-degenerate filter)
- Denominator = P(S) = probability batch survives filtering

For p_t = 0.1, G = 8:
- P(S) = 1 - 0.9^8 - 0.1^8 ≈ 0.57
- Numerator = 0.1 × (1 - 0.1^7) ≈ 0.1
- E[p̂_t | S] ≈ 0.1 / 0.57 ≈ 0.175

So your baseline thinks the "average" is 17.5% when reality is 10%. Every correct answer gets advantage 1 - 0.175 = 0.825 instead of 1 - 0.1 = 0.9. You're losing 8% of the reward signal on every correct hard answer. And for the wrong answers, you're penalizing them less than you should.

For p_t = 0.01 (really hard), G = 8:
- P(S) ≈ 1 - 0.99^8 ≈ 0.077
- E[p̂_t | S] ≈ 0.01 / 0.077 ≈ 0.13

Baseline is 13% when reality is 1%! That's 13x overestimate. The one correct answer out of 8 gets advantage 1 - 0.13 = 0.87 instead of 1 - 0.01 = 0.99. And the 7 wrong answers get advantage -0.13 instead of -0.01, so they're over-penalized too. The whole batch is messed up.

## 他们的fix

HA-DW = History-Aware Adaptive Difficulty Weighting. Two parts:

**Part 1: Track model's evolving capability $C_t$**

They maintain a running estimate of "how good is the model right now" using a Kalman-filter-style update:

$$C_t^+ = (1-\eta_t) C_t^- + \eta_t y_t$$

Where $y_t$ is current batch accuracy, $\eta_t$ adapts based on training stability (high when capability changing fast, low when stable). This gives you a "what's normal for this model at this point in training" reference.

**Part 2: Reweight advantages based on difficulty relative to $C_t$**

For each prompt, compute:
- $\text{diff}_t = \hat{p}_t - C_t$ (is this prompt easier or harder than model's current level?)
- $D_{t,i} = -\text{sgn}(\hat{A}_{t,i}) \cdot \text{sgn}(\text{diff}_t)$ (should we amplify or suppress?)
- $\Phi_{t,i} = \lambda_{\text{scale}} \cdot \exp(D_{t,i} \cdot |\text{diff}_t|)$ (multiplicative weight)

The four cases:
1. Hard prompt + correct answer: amplify (model beat its average on hard problem, big deal!)
2. Hard prompt + wrong answer: suppress (expected, don't over-penalize)
3. Easy prompt + correct answer: suppress (expected, don't over-reward)
4. Easy prompt + wrong answer: amplify (model failed on easy problem, big deal!)

This is just common sense pedagogy! You reward surprising successes and surprising failures, not expected ones. The math just formalizes this.

## 实验结果有多impressive？

Look at Table 3, this is the killer result:

| Setup | MATH500 |
|-------|---------|
| GRPO, G=8 | 75.4 |
| GRPO, G=16 | 76.2 |
| GRPO+HA-DW, G=8 | **78.0** |

So HA-DW with 8 rollouts beats vanilla GRPO with 16 rollouts. You're getting better than 2x rollout efficiency for free. And G=32 just OOMs, so you literally can't brute-force your way out of this with more sampling.

Across the board on Qwen3-8B with DAPO:
- MATH500: 79.2 → 82.8 (+3.6)
- AMC23: 67.5 → 70.0 (+2.5)
- AVG: 50.7 → 53.4 (+2.7)

These are not marginal gains. And they're consistent across GRPO, GSPO, DAPO, across Qwen and LLaMA, across 3B/4B/8B scales. That's a lot of robustness.

The difficulty-stratified analysis on MATH500 (Figure 1c) tells the story:
- Easy problems: no change (already learned)
- Medium problems: no change
- **Hard problems: +3.4%**

Exactly where the bias is worst, exactly where the fix helps most. Theoretical prediction matches empirical result. Beautiful.

## 我的take

This is a good paper for a few reasons:

1. **It finds a real bug that everyone missed.** That's the best kind of paper. Not "we proposed yet another variant" but "hey everyone, you've been doing this wrong and here's why."

2. **The fix is cheap.** No extra inference, no extra model, just a reweighting factor computed from batch statistics you already have. This is the kind of thing that should just get absorbed into the standard recipe.

3. **The theory is complete.** They don't just wave hands—they prove the bias exists, characterize it probabilistically, extend to non-binary rewards, and prove their fix reduces bias (Theorem 3). 

4. **Honest about limitations.** They acknowledge this only applies to group-relative methods, and bias is everywhere so future work abounds.

What I want to see next:

**Does HA-DW accelerate reasoning emergence?** The training dynamics in Figure 4 show response length increasing faster with HA-DW. This hints that the model is hitting "aha moments" earlier. If HA-DW can reproduce R1-style reasoning emergence with less compute, that's huge.

**Step-level version.** Current HA-DW is sequence-level. But PRMs give step-level rewards. Each step has its own "correctness" and the same truncation bias would apply. A step-level HA-DW could be even more effective.

**Multi-turn RL.** Dialogue turns have the same issue—some turns are "easy" (model nails it), some are "hard" (model struggles). The truncation/filtering logic in multi-turn RL is more complex but the bias structure should be analogous.

**Interaction with MoE.** GSPO was designed for MoE models and HA-DW helps there too. But MoE has this extra dimension of expert routing—different experts fire on different problems, so the "model capability" isn't a single scalar $C_t$ but a vector over experts. Extending $C_t$ to per-expert tracking could be a nice follow-up.

**Why not just use a critic?** The whole point of GRPO was to avoid the value network. But if you need Kalman filters and adaptive weighting to fix the baseline, maybe a small critic isn't so bad? Worth comparing the compute/performance tradeoff honestly.

## 更深的联想

This paper got me thinking about a meta-principle: **filtering is never free**. Whenever you discard data based on some criterion, you bias your remaining data. We see this everywhere:

- **Dropout in neural networks**: dropping neurons biases the remaining computation (we just normalize to compensate)
- **Sequence packing in LLM training**: packing short sequences biases toward short-sequence-dominated batches
- **Rejection sampling in DPO**: filtering preference pairs based on log-prob ratio biases the preference distribution
- **Data deduplication**: removing duplicates biases toward diverse but possibly less representative data
- **Active learning**: selecting samples to label based on model uncertainty biases the labeled set

In each case, the filtering seems reasonable but introduces subtle bias. The HA-DW approach—acknowledge the bias and correct for it algorithmically—is a generalizable strategy.

Actually wait, this connects to something I've been thinking about with RLHF/DPO. In DPO you filter preference pairs where the policy already strongly prefers one option (because gradient is ~0). But this filtering biases toward "uncertain" pairs, which might be exactly the pairs where the preference signal is noisy! You're keeping the bad data and throwing away the clean data. Has anyone analyzed this? Probably similar bias structure.

Reference: [DPO paper](https://arxiv.org/abs/2305.18290)

And in PPO itself, the clipping mechanism (min(ratio, clipped_ratio)) effectively filters out large-ratio updates. But which updates have large ratios? The ones where the policy changed a lot, which are the ones where you're learning the most. Clipping systematically biases toward small-gradient updates. This is a known "issue" but I haven't seen a formal bias analysis like this paper does for GRPO.

Reference: [PPO](https://arxiv.org/abs/1707.06347)

## 实操建议

If you're training an LLM with GRPO/GSPO/DAPO right now:

1. **Just add HA-DW.** It's plug-and-play, no extra inference cost. The reweighting is computed from batch statistics you already have.

2. **Start with $\lambda_{\text{scale}} = 1.3$.** Table 7 shows this is the sweet spot. Anything in [1.0, 1.5] works reasonably.

3. **Don't worry about exact $C_t$ initialization.** The Kalman filter will converge after a few batches. Maybe initialize $C_0$ = 0.5 if you have no prior.

4. **Sliding window $m$ = 5-10 batches** seems reasonable based on the description.

5. **Monitor the bias empirically.** Do what they did in Appendix E.1: sample some prompts at G=8 and at G=128, compare. If you see the bias pattern (hard prompts underestimated), HA-DW is helping.

6. **Don't just crank up G.** Table 3 shows G=8 + HA-DW > G=16 vanilla. And G=32 OOMs. So algorithmic correction beats brute force.

## 最终intuition

Think of it this way: GRPO is like a teacher who only gives feedback when the class has mixed performance. "Everyone got it right? No feedback. Everyone got it wrong? No feedback." Sounds efficient—why waste time on cases where everyone's the same?

But this means the teacher only gives feedback on problems where the class is "confused"—and the baseline for "confused" depends on problem difficulty. For a hard problem, even 1 kid getting it right is amazing, but the teacher compares to the confused-class average (which is artificially high because all-zero classes were excluded) and thinks "meh, below average." For an easy problem, 1 kid getting it wrong is surprising, but the teacher compares to the confused-class average (artificially low because all-correct classes were excluded) and thinks "meh, above average, no big deal."

HA-DW is the teacher who remembers "this class usually gets 60% right" and uses that as the baseline instead of the within-batch average. So when a hard problem has 1 kid getting it right, the teacher goes "1 out of 8 on a problem the class usually fails at—amazing!" When an easy problem has 1 kid getting it wrong, the teacher goes "7 out of 8 on a problem the class usually aces—what happened to that 1 kid?"

It's just basic pedagogy formalized. The math tells you the bias is real and the fix works. The experiments tell you the fix is practical. That's a complete story.

The deeper lesson: when your algorithm has any kind of filtering/selection mechanism, check whether that filtering introduces systematic bias. GRPO's truncation seemed innocent—zero gradients mean no update, why not skip? But that skipping changes the data distribution in a way that systematically hurts learning on exactly the hard examples you need most.

Reference: [The paper itself (Your Group-Relative Advantage Is Biased)](https://arxiv.org/abs/2506.10992)

Honestly I think this should just become standard in GRPO implementations. It's like batch norm—you don't argue about it, you just add it.

---

# Your Group-Relative Advantage Is Biased 深度解析

## TL;DR: 核心发现的intuition

这篇paper揭示了一个在GRPO及其variants中普遍存在却被overlooked的**statistical bias**：当你用group-relative baseline $\hat{p}_t = \frac{1}{G}\sum_{i=1}^G r_{t,i}$ 作为advantage估计时，estimator $\hat{A}_{t,i} = r_{t,i} - \hat{p}_t$ 相对于true advantage $A_{t,i} = r_{t,i} - p_t$ 是**systematically biased**的——hard prompts的advantage被underestimate，easy prompts的advantage被overestimate。这个bias的root cause是GRPO family的**truncation机制**：当 $R=0$ 或 $R=G$ 时gradient为零，这些degenerate groups被silently discarded，conditioning在这个non-degenerate event $S = \{1 \leq R \leq G-1\}$ 上就会引入selection bias。

---

## 1. 为什么会有bias？从直觉到数学

### 1.1 直觉：Selection bias from filtering

Consider一下GRPO的实际操作：你对每个prompt $x_t$ 采样 $G$ 个responses，如果 $G$ 个全部correct或全部wrong，advantage $\hat{A}_{t,i} = 0$ for all $i$，零梯度，不更新参数。这些group在effectively被filter out了。

现在考虑一个**hard prompt**，$p_t = 0.1$（模型平均10%能做对）。如果你采样 $G=8$ 个responses：
- $P(R=0) = 0.9^8 \approx 0.43$，43%的groups直接被丢弃
- $P(R=8) = 0.1^8 \approx 10^{-8}$，几乎不可能

被保留下来的groups是那些 $R \in \{1,...,7\}$ 的，这些groups的 $\hat{p}_t = R/G$ 的期望会被**pushed upward**，因为低 $R$ 值（尤其是 $R=0$）被systematically剔除了。所以 $\mathbb{E}[\hat{p}_t | S] > p_t$，导致 $\hat{A}_{t,i}$ 普遍偏小。

对称地，对**easy prompt** $p_t = 0.9$，$P(R=8) = 0.9^8 \approx 0.43$，这些全部correct的groups也被丢弃，剩下的groups的 $\hat{p}_t$ 被pushed downward，$\mathbb{E}[\hat{p}_t | S] < p_t$，导致 $\hat{A}_{t,i}$ 普遍偏大。

这就是Theorem 1的intuition。

### 1.2 Theorem 1的数学推导细节

设 $r_{t,i} \sim \text{Bernoulli}(p_t)$ i.i.d.，$R = \sum_{i=1}^G r_{t,i}$，$\hat{p}_t = R/G$。

Conditional expectation on $S$：

$$\mathbb{E}[\hat{p}_t | S] = \frac{\mathbb{E}[R \cdot \mathbf{1}_{\{S\}}]}{G \cdot \mathbb{P}(S)}$$

其中：
- $\mathbb{P}(S) = 1 - (1-p_t)^G - p_t^G$（保留probability）
- $\mathbb{E}[R \cdot \mathbf{1}_{\{S\}}] = \mathbb{E}[R] - G \cdot \mathbb{P}(R=G) = G p_t - G p_t^G = G p_t(1 - p_t^{G-1})$

所以：

$$\mathbb{E}[\hat{p}_t | S] = \frac{p_t(1 - p_t^{G-1})}{1 - (1-p_t)^G - p_t^G}$$

**关键步骤**：对比这个值和 $p_t$。Lemma 2给出了bias的closed form：

$$\mathbb{E}[\hat{p}_t | S] - p_t = \frac{p_t(1-p_t)^G + p_t^{G+1} - p_t^G}{1 - (1-p_t)^G - p_t^G}$$

变量说明：
- $p_t$：prompt $x_t$ 在policy $\pi_{\theta_t}$ 下的expected reward（true success probability）
- $G$：group size（每个prompt采样的response数量）
- $(1-p_t)^G$：$G$ 次全错的概率
- $p_t^G$：$G$ 次全对的概率
- 分母 $1 - (1-p_t)^G - p_t^G$：non-degenerate event $S$ 的概率

当 $p_t < 0.5$，分子 $p_t(1-p_t)^G + p_t^{G+1} - p_t^G$ 的符号分析：因为 $(1-p_t)^G \gg p_t^G$（比如 $p_t=0.1, G=8$ 时 $0.9^8 \approx 0.43$ vs $0.1^8 \approx 10^{-8}$），第一项dominate，整体为正，所以 $\mathbb{E}[\hat{p}_t | S] > p_t$。

这就完成了Theorem 1的证明。Advantage $A_{t,i} = r_{t,i} - p_t$，estimated advantage $\hat{A}_{t,i} = r_{t,i} - \hat{p}_t$，所以 $\mathbb{E}[\hat{A}_{t,i} | S] - A_{t,i} = p_t - \mathbb{E}[\hat{p}_t | S] < 0$ 当 $p_t < 0.5$。

### 1.3 Theorem 2: 概率级别的刻画

Theorem 1只给了expectation，但没说"有多大概率over/underestimate"。Theorem 2给出了精确的probability mass：

对于hard prompt ($p_t < 0.5$)，overestimation probability of $\hat{p}_t$：

$$\mathbb{P}(\hat{p}_t - p_t > \epsilon | S) = \frac{\sum_{k=\lfloor G(p_t+\epsilon) \rfloor + 1}^{G-1} \binom{G}{k} p_t^k (1-p_t)^{G-k}}{1 - (1-p_t)^G - p_t^G}$$

变量说明：
- $\epsilon \in (0, \mathbb{E}[\hat{p}_t | S] - p_t)$：deviation threshold
- $\binom{G}{k}$：binomial coefficient，从 $G$ 个response中选 $k$ 个correct的方式数
- $p_t^k (1-p_t)^{G-k}$：恰好 $k$ 个correct的概率
- 求和下界 $\lfloor G(p_t+\epsilon) \rfloor + 1$：要求 $\hat{p}_t = R/G > p_t + \epsilon$，即 $R > G(p_t+\epsilon)$
- 分母：normalization by $\mathbb{P}(S)$

### 1.4 Corollary 1-3: 实用regime下的量化

Corollary 1给出了实际 $G \in [2,8]$ regime下的concrete数字：

| Setting | Probability |
|---------|-------------|
| $\mathbb{P}(\hat{A}_{t,i} < A_{t,i} \| S, p_t < 0.5)$ | $> 0.63$ |
| $\mathbb{P}(\hat{A}_{t,i} > A_{t,i} \| S, p_t > 0.5)$ | $> 0.63$ |
| $\mathbb{P}(\hat{A}_{t,i} < A_{t,i} \| S, p_t < 0.25)$ | $> 0.78$ |
| $\mathbb{P}(\hat{A}_{t,i} > A_{t,i} \| S, p_t > 0.75)$ | $> 0.78$ |
| $\mathbb{P}(\hat{A}_{t,i} < A_{t,i} \| S, p_t < 0.125)$ | $= 1.00$ |
| $\mathbb{P}(\hat{A}_{t,i} > A_{t,i} \| S, p_t > 0.875)$ | $= 1.00$ |

Corollary 3更极端：当 $p_t < 1/G$（比single correct还罕见），underestimate is **sure** event；当 $p_t > (G-1)/G$，overestimate is sure event。这是deterministic bias，no stochasticity can save you。

Figure 2展示了bias magnitude $|A_{t,i} - \mathbb{E}[\hat{A}_{t,i} | S]|$ 作为 $p_t$ 和 $G$ 的function。$G$ 越小bias越大，$p_t$ 离0.5越远bias越大。这印证了"small group size + extreme difficulty = worst bias"。

### 1.5 为什么这个发现重要？

GRPO literature里大家普遍以为"group-relative baseline is unbiased estimator of $p_t$"，因为 $\hat{p}_t = R/G$ 在无truncation时确实是unbiased的。但**truncation机制改变了这一切**。这个bias会导致：

1. **Hard prompts under-learned**：advantage被低估，gradient信号被削弱，模型hardly explore hard problems
2. **Easy prompts over-exploited**：advantage被高估，gradient过大，模型在已经掌握的题目上浪费capacity
3. **Imbalanced exploration-exploitation**：训练dynamics被distort

这跟curriculum learning的intuition相反——你本来希望hard problems获得更多attention来push capability frontier，但GRPO的内在bias恰恰penalize了这些critical prompts。

---

## 2. HA-DW算法：History-Aware Adaptive Difficulty Weighting

### 2.1 整体架构

Figure 3展示了HA-DW的两阶段pipeline：

**Phase 1: Evolving Difficulty Anchor**
跨batch追踪model capability $C_t$，整合long-term reward trend。

**Phase 2: Adaptive Advantage Reweighting**
根据prompt相对于 $C_t$ 的deviation，动态调整advantage weight $\Phi_{t,i}$。

### 2.2 Evolving Difficulty Anchor的Kalman-style更新

Core idea：把model的"solving capability" $C_t$ 当作latent state，用batch accuracy $y_t$ 作为observation，做Bayesian filtering。

**Observation model**：
$$y_t = \frac{K_t}{B_t}, \quad K_t = \sum_{i=1}^{B_t} r_{t,i}$$

变量：
- $B_t$：batch $t$ 中的total response数量
- $K_t$：batch $t$ 中的correct response数量
- $y_t$：batch accuracy，作为model capability的noisy observation

**Kalman update**：
$$C_t^+ = (1-\eta_t) C_t^- + \eta_t y_t$$

变量：
- $C_t^-$：prior belief（来自上一batch的posterior $C_{t-1}^+$）
- $C_t^+$：posterior belief after incorporating $y_t$
- $\eta_t \in [0,1]$：forgetting factor，控制historical information的weight

**Adaptive forgetting factor**：
$$\bar{C}_t = \frac{1}{m} \sum_{j=1}^m C_{t-j}, \quad \sigma_t = \sqrt{\frac{1}{m}\sum_{j=1}^m (C_{t-j} - \bar{C}_t)^2}, \quad \eta_t = \eta \cdot \sigma_t$$

变量：
- $m$：sliding window size
- $\bar{C}_t$：过去 $m$ 个batch的average capability
- $\sigma_t$：过去 $m$ 个batch的standard deviation，衡量model stability
- $\eta$：task-dependent base hyperparameter

Intuition：early training阶段 $\sigma_t$ 大（capability快速变化），$\eta_t$ 大，新observation权重大，快速适应；late training阶段 $\sigma_t$ 小（稳定），$\eta_t$ 小，historical information权重大，减少noise。

**Prior propagation**：
$$C_t^+ \to C_{t+1}^-$$

Appendix F还提供了hard update variant：
$$C_t^+ = \frac{1}{h}\left(\sum_{j=1}^{h-1} y_{t-j} + y_t\right)$$

这是简单moving average，虽然lose了short-term oscillation info，但algorithmic complexity大幅降低。

### 2.3 Adaptive Reweighting公式

**History-based difficulty**：
$$\text{diff}_t^{\text{his}} = \hat{p}_t - C_t^-$$

Intuition：如果 $\hat{p}_t > C_t^-$，说明当前group的performance高于model的historical level，这个prompt相对easy；反之则hard。

**Direction of adjustment**：
$$D_{t,i} = -\text{sgn}(\hat{A}_{t,i}) \cdot \text{sgn}(\text{diff}_t^{\text{his}})$$

变量：
- $\text{sgn}(\hat{A}_{t,i})$：advantage符号，positive表示correct response，negative表示wrong response
- $\text{sgn}(\text{diff}_t^{\text{his}})$：difficulty方向，positive表示easy prompt，negative表示hard prompt

四种组合：
1. Hard prompt + correct response：$\hat{A}_{t,i} > 0$, $\text{diff}_t^{\text{his}} < 0$，$D_{t,i} = -1 \cdot (-1) = 1$，amplify weight
2. Hard prompt + wrong response：$\hat{A}_{t,i} < 0$, $\text{diff}_t^{\text{his}} < 0$，$D_{t,i} = -(-1) \cdot (-1) = -1$，suppress weight（减小负advantage的magnitude）
3. Easy prompt + correct response：$\hat{A}_{t,i} > 0$, $\text{diff}_t^{\text{his}} > 0$，$D_{t,i} = -1 \cdot 1 = -1$，suppress weight
4. Easy prompt + wrong response：$\hat{A}_{t,i} < 0$, $\text{diff}_t^{\text{his}} > 0$，$D_{t,i} = -(-1) \cdot 1 = 1$，amplify weight（增大负advantage的magnitude，强penalize easy prompt上的wrong answer）

这个设计极其精妙：它simultaneously boost了hard prompt上的learning signal和easy prompt上的penalty signal。

**Magnitude of adjustment**：
$$M_t = |\text{diff}_t^{\text{his}}|$$

**Final reweighting factor**：
$$\Phi_{t,i} = \lambda_{\text{scale}} \cdot \exp(D_{t,i} \cdot M_t)$$

变量：
- $\lambda_{\text{scale}}$：global scaling constant
- $\exp(\cdot)$：确保smooth、multiplicative adjustment

为什么用exponential？因为：
1. 严格positive，不会flip advantage符号
2. Multiplicative effect自然preserve advantage的相对ranking
3. Smooth gradient，无discontinuity

### 2.4 HA-DW Objective

$$L_{\text{HA-DW}}(\theta) = \frac{1}{G} \sum_{i=1}^G \psi\left(\frac{\pi_\theta(y_{t,i}|x_t)}{\pi_{\theta_{\text{old}}}(y_{t,i}|x_t)}\right) \cdot \phi(\hat{A}_{t,i}) \cdot \Phi_{t,i}$$

变量：
- $\psi(\cdot)$：importance sampling ratio的function（identity、clip、log等，取决于具体algorithm）
- $\phi(\cdot)$：advantage的transform（identity、normalization等）
- $\Phi_{t,i}$：HA-DW的reweighting factor

HA-DW是plug-and-play module，可以seamlessly集成到GRPO、GSPO、DAPO等任何group-relative algorithm中。Appendix B给出了具体instantiation：

**GRPO+HA-DW**：
$$J_{\text{GRPO+HA-DW}}(\theta) = \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_{t,i}|}\sum_{\tau=1}^{|o_{t,i}|} \min\left(r_{t,i,\tau}(\theta)\hat{A}_{t,i,\tau} \cdot \Phi_{t,i}, \text{clip}(r_{t,i,\tau}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{t,i,\tau} \cdot \Phi_{t,i}\right)$$

变量：
- $o_{t,i}$：第 $i$ 个response的token sequence
- $|o_{t,i}|$：response长度
- $\tau$：token index
- $r_{t,i,\tau}(\theta) = \frac{\pi_\theta(y_{t,i,\tau}|x_t, y_{t,i,<\tau})}{\pi_{\theta_{\text{old}}}(y_{t,i,\tau}|x_t, y_{t,i,<\tau})}$：token-level importance sampling ratio
- $\epsilon$：clipping hyperparameter

注意HA-DW的 $\Phi_{t,i}$ 是**sequence-level**的，对所有tokens应用同一个weight，这preserve了GRPO的token-level结构。

**GSPO+HA-DW**（sequence-level）：
$$J_{\text{GSPO+HA-DW}}(\theta) = \frac{1}{G}\sum_{i=1}^G \min\left(r_{t,i}(\theta)\hat{A}_{t,i} \cdot \Phi_{t,i}, \text{clip}(r_{t,i}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{t,i} \cdot \Phi_{t,i}\right)$$

GSPO的importance ratio是sequence-level的：
$$r_{t,i}(\theta) = \frac{\pi_\theta(y_{t,i}|x_t)}{\pi_{\theta_{\text{old}}}(y_{t,i}|x_t)} = \prod_{\tau=1}^{|y_{t,i}|} \frac{\pi_\theta(y_{t,i,\tau}|x_t, y_{t,i,<\tau})}{\pi_{\theta_{\text{old}}}(y_{t,i,\tau}|x_t, y_{t,i,<\tau})}$$

**DAPO+HA-DW**（token-level with decoupled clipping）：
$$J_{\text{DAPO+HA-DW}}(\theta) = \frac{1}{\sum_{i=1}^G |o_{t,i}|}\sum_{i=1}^G \sum_{\tau=1}^{|o_{t,i}|} \min\left(r_{t,i,\tau}(\theta)\hat{A}_{t,i,\tau} \cdot \Phi_{t,i}, \text{clip}(r_{t,i,\tau}(\theta), 1-\epsilon, 1+\epsilon')\hat{A}_{t,i,\tau} \cdot \Phi_{t,i}\right)$$

DAPO的特点是decoupled clipping（$\epsilon \neq \epsilon'$）和dynamic sampling。

---

## 3. Theoretical Guarantee: Theorem 3

### 3.1 Lemma 1: Baseline Rectification

Core question：是否存在一个scaling factor $c$ 使得 $\tilde{p}_t = c \cdot \hat{p}_t$ 的conditional expectation接近 $p_t$？

Lemma 1给出了 $c$ 的feasible range。关键definitions：

$$\epsilon_\delta := \sqrt{\frac{1}{2G}\log\left(\frac{2}{\delta(1-(1-\Delta)^G - \Delta^G)}\right)}$$

变量：
- $\delta \in (0,1)$：failure probability
- $\Delta \in (0, 1/2]$：prompt的boundary，assume $p_t \in [\Delta, 1-\Delta]$
- $G$：group size
- $1-(1-\Delta)^G - \Delta^G$：worst-case non-degenerate probability

$$I_t := [\hat{p}_t - \epsilon_\delta, \hat{p}_t + \epsilon_\delta] \cap [\Delta, 1-\Delta]$$

这是 $p_t$ 的confidence interval（基于Hoeffding bound + truncation correction）。

$$c_{\text{low}} := \sup_{p \in I_t} \frac{(p-\epsilon)A(p)}{p(1-p^{G-1})}, \quad c_{\text{high}} := \inf_{p \in I_t} \frac{(p+\epsilon)A(p)}{p(1-p^{G-1})}$$

其中 $A(p) := 1 - (1-p)^G - p^G$ 是non-degenerate probability。

当 $c \in (c_{\text{low}}, c_{\text{high}})$，with probability $\geq 1-\delta$ conditional on $S$：

$$\mathbb{E}[\tilde{p}_t | S] \in (p_t - \epsilon, p_t + \epsilon)$$

即rectified baseline $\tilde{p}_t$ 的期望在true $p_t$ 的 $\epsilon$-neighborhood内。

### 3.2 Theorem 3: HA-DW provably reduces bias

Theorem 3建立了 $\lambda_{\text{scale}}$ 的feasible range：

$$\lambda_{\text{scale}} \in \left(\frac{1 + \frac{(1-c_{\text{high}})\hat{p}_t}{1-\hat{p}_t}}{\exp(D_{t,i} M_t)}, \frac{1 + \frac{(1-c_{\text{low}})\hat{p}_t}{1-\hat{p}_t}}{\exp(D_{t,i} M_t)}\right) \cup \left(\frac{c_{\text{low}}}{\exp(D_{t,i} M_t)}, \frac{c_{\text{high}}}{\exp(D_{t,i} M_t)}\right)$$

当这个condition满足时：

$$\left|\mathbb{E}[\hat{A}_{t,i} \cdot \Phi_{t,i} | S] - A_{t,i}\right| < \left|\mathbb{E}[\hat{A}_{t,i} | S] - A_{t,i}\right|$$

即HA-DW后的advantage估计的bias严格小于原始 $\hat{A}_{t,i}$ 的bias。

### 3.3 Proof sketch (Appendix D.4)

考虑 $\Phi_{t,i}\hat{A}_{t,i} = r_{t,i} - c\hat{p}_t$ 的四种情况：

1. **Hard prompt + correct** ($r_{t,i}=1$, $D_{t,i}=1$)：$\Phi_{t,i} = \frac{1 - c\hat{p}_t}{1-\hat{p}_t}$，需要 $\lambda_{\text{scale}} \in (\cdot, \cdot)$ 使得amplification sufficient
2. **Hard prompt + wrong** ($r_{t,i}=0$, $D_{t,i}=-1$)：$\Phi_{t,i} = c$
3. **Easy prompt + correct** ($r_{t,i}=1$, $D_{t,i}=-1$)：类似case 1
4. **Easy prompt + wrong** ($r_{t,i}=0$, $D_{t,i}=1$)：类似case 2

这四个interval的union就是Theorem 3的feasible range。这给实践中的 $\lambda_{\text{scale}}$ 选择提供了principled guidance。

### 3.4 Non-binary Extension (Appendix D.5)

Theorem 4将分析扩展到continuous bounded reward（如Beta、truncated Gaussian）。定义non-degenerate event：

$$S_\sigma := \{\exists i \neq j: |r_{t,i} - r_{t,j}| > \sigma\}$$

即group内的rewards不完全相同（允许小的deviation $\sigma$）。结论是：bias现象在continuous reward下依然存在，magnitude随 $p_t$ 偏离0.5而increases。Corollary 4和5分别给出了Beta和truncated Gaussian的closed form expressions。

---

## 4. 实验深度解析

### 4.1 Main Results (Table 1)

| Model | Algorithm | MATH500 | AIME25 | AMC23 | Minerva | OlympiadBench | AVG |
|-------|-----------|---------|--------|-------|---------|---------------|-----|
| Qwen3-4B | GRPO | 75.4 | 19.6 | 60.3 | 33.8 | 43.5 | 46.5 |
| | +HA-DW | 78.0 | 20.4 | 63.4 | 36.8 | 44.7 | 48.7 |
| | GSPO | 75.8 | 20.0 | 62.2 | 35.3 | 42.3 | 47.1 |
| | +HA-DW | 77.6 | 19.6 | 68.6 | 37.1 | 43.2 | 49.2 |
| | DAPO | 76.8 | 18.3 | 60.0 | 35.7 | 43.2 | 46.8 |
| | +HA-DW | 78.6 | 21.3 | 65.0 | 37.5 | 45.3 | 49.5 |
| Qwen3-8B | GRPO | 78.8 | 20.4 | 64.2 | 38.2 | 46.4 | 49.6 |
| | +HA-DW | 80.0 | 22.9 | 72.8 | 39.7 | 47.1 | 52.5 |
| | DAPO | 79.2 | 20.4 | 67.5 | 39.3 | 47.2 | 50.7 |
| | +HA-DW | 82.8 | 23.3 | 70.0 | 40.8 | 50.0 | 53.4 |
| LLaMA-3.2-3B | GRPO | 51.4 | 2.7 | 31.7 | 22.8 | 19.9 | 25.7 |
| | +HA-DW | 53.2 | 3.3 | 35.0 | 23.9 | 20.1 | 27.1 |
| | DAPO | 52.4 | 2.5 | 35.0 | 22.4 | 20.2 | 26.5 |
| | +HA-DW | 53.2 | 3.1 | 37.5 | 24.6 | 22.3 | 28.1 |

Key observations：
1. **Consistent improvement across models**：Qwen3-4B/8B、LLaMA-3.2-3B都稳定提升
2. **Consistent improvement across algorithms**：GRPO、GSPO、DAPO都work
3. **Bigger gain on larger models**：Qwen3-8B上DAPO+HA-DW从50.7到53.4（+2.7），比4B上的+2.7相当，但绝对performance更高
4. **Hard benchmarks benefit more**：AIME25（极难）上Qwen3-8B DAPO+HA-DW从20.4到23.3（+2.9），相对gain 14.2%

### 4.2 Difficulty-stratified analysis (Figure 1c)

MATH500按难度分三档：
- Easy (Level 1)：GRPO vs GRPO+HA-DW comparable
- Mid (Levels 2-3)：comparable
- **Hard (Levels 4-5)：+3.4% improvement**

这正是Theorem 1预测的——hard prompts是biased estimation最严重的regime，HA-DW的corrective effect最显著。

### 4.3 Ablation: Dynamic threshold $C_t$ (Table 2)

| Threshold | MATH500 | AIME25 | AMC23 | Minerva | OlympiadBench | AVG |
|-----------|---------|--------|-------|---------|---------------|-----|
| Base | 75.4 | 19.6 | 60.3 | 33.8 | 43.5 | 46.5 |
| 0.4 (fixed) | 77.0 | 18.5 | 63.1 | 37.5 | 44.3 | 48.1 |
| 0.5 (fixed) | 76.6 | 20.0 | 62.7 | 35.7 | 44.0 | 47.8 |
| 0.6 (fixed) | 76.8 | 21.3 | 61.1 | 36.4 | 44.3 | 48.0 |
| $C_t$ (dynamic) | **78.0** | 20.4 | **63.4** | 36.8 | **44.7** | **48.7** |

Key insight：fixed threshold也有improvement（partially mitigate bias），但dynamic $C_t$ 整体最优。这说明cross-batch information至关重要——capability是drifting的，static threshold会逐渐mismatch。

### 4.4 Group size ablation (Table 3)

| Dataset | Rollout=8 GRPO | Rollout=16 GRPO | Rollout=8 GRPO+HA-DW |
|---------|----------------|-----------------|----------------------|
| MATH500 | 75.4 | 76.2 | **78.0** |
| AIME25 | 19.6 | 19.2 | **20.4** |
| AMC23 | 60.3 | 61.6 | **63.4** |
| Minerva | 33.8 | 34.2 | **36.8** |
| OlympiadBench | 43.5 | 43.9 | **44.7** |

**Stunning result**：Rollout=8+HA-DW **全面beat** Rollout=16 vanilla GRPO。这意味着HA-DW相当于**免费获得2x的rollout efficiency**。而Rollout=32直接OOM。

从理论角度，这完全consistent：增大 $G$ 确实reduce bias（Corollary 2显示 $G$ 越大bias probability越低），但cost是linear的。HA-DW用algorithmic correction替代了brute-force sampling，更efficient。

### 4.5 Scaling parameter $\lambda_{\text{scale}}$ (Table 7)

| $\lambda_{\text{scale}}$ | MATH500 | AIME25 | AMC23 | Minerva | OlympiadBench | AVG |
|------------------------|---------|--------|-------|---------|---------------|-----|
| 0.5 | 75.4 | 18.1 | 61.1 | 34.2 | 43.7 | 46.5 |
| 0.8 | 76.8 | 19.2 | 61.3 | 34.9 | 43.7 | 47.2 |
| 1.0 | 76.8 | 18.5 | 61.6 | 36.0 | 44.3 | 47.4 |
| 1.3 | **78.0** | 20.4 | **63.4** | 36.8 | **44.7** | **48.7** |
| 1.5 | 77.8 | **20.8** | 63.1 | **37.1** | 44.0 | 48.6 |
| 1.7 | 76.4 | 20.0 | 63.4 | 36.4 | 44.3 | 48.1 |
| 2.0 | 76.8 | 19.0 | 61.9 | 35.3 | 43.5 | 47.3 |

Sweet spot在 $\lambda_{\text{scale}} \in [1.3, 1.5]$。太小（0.5）等于no adjustment，太大（2.0）over-amplify。这跟Theorem 3的feasible range理论一致——存在一个optimal $\lambda_{\text{scale}}$ 平衡各种difficulty prompts的adjustment。

### 4.6 Training dynamics (Figure 4)

- **Accuracy**：HA-DW converge到更高的plateau
- **Training reward**：HA-DW获得更高reward，说明exploration更efficient
- **Response length**：HA-DW的response更长！这是关键副产品——因为hard prompts被amplify，model被incentivize产生longer reasoning chain来tackle challenging problems

这呼应了DeepSeek-R1的"aha moment"——longer reasoning emerges when model is properly incentivized。

### 4.7 Empirical bias verification (Appendix E.1, Figure 6)

实验设计精巧：
1. 在rollout=8下evaluate Qwen3-4B-Base
2. 筛选出"只有1个correct response"的hard prompts（50个）和"只有1个wrong response"的easy prompts（50个）
3. 对这些prompts在rollout=128下evaluate（足够多rollouts反映true $p_t$）

Results：
- Hard prompts (rollout=8时1 correct)：24/50个在rollout=128时correct count < 16，证明rollout=8的advantage被低估
- Easy prompts (rollout=8时1 wrong)：12/50个在rollout=128时wrong count < 16，证明rollout=8时overestimate了wrong advantage

这是对Theorem 1的direct empirical confirmation。

---

## 5. 与相关工作联系与intuition延伸

### 5.1 GRPO family的演进

- **PPO** (Schulman 2017)：需要value network，expensive
- **GRPO** (Shao 2024, DeepSeekMath)：用group baseline替代value network
- **Dr.GRPO** (Liu 2025)：remove heuristic normalizations
- **DAPO** (Yu 2025)：decoupled clipping + dynamic sampling
- **GSPO** (Zheng 2025)：sequence-level ratio，improve stability for large/MoE models
- **GMPO** (Zhao 2025)：geometric mean of token rewards
- **HA-DW (this paper)**：first to address the **estimation bias** issue

Reference: [DeepSeekMath GRPO paper](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476), [GSPO](https://arxiv.org/abs/2507.18071)

### 5.2 与curriculum learning的对比

传统curriculum learning（如Reference [Learning like humans](https://arxiv.org/abs/2505.08364)）是**explicitly order** training samples from easy to hard。HA-DW的哲学不同——它**不改变data order**，而是**adaptively reweight** advantage based on dynamic difficulty assessment。这更robust因为：
1. 不需要prior knowledge of difficulty
2. Difficulty是model-dependent（$p_t$ 是 $\pi_{\theta_t}$ 的函数），static ordering会过时
3. HA-DW的 $C_t$ 自动追踪evolving capability

### 5.3 与importance sampling bias corrections的联系

Off-policy RL literature里类似问题被广泛研究：
- **Retrace** (Munos 2016, [paper](https://papers.nips.cc/paper/2016/hash/30c5d0d2d6f9f19e4f8a5c5e4c5c5c5c-Abstract.html))：retrace technique mitigate off-policy bias
- **V-trace** (Espeholt 2018, IMPALA)：importance-weighted corrections for value function
- **DR-OVR** (Jiang & Li 2016)：doubly robust off-policy evaluation

HA-DW的contribution在于：揭示了**on-policy** group-relative方法也存在analogous bias，且bias的source不是off-policy mismatch而是truncation。

### 5.4 Cross-batch signals的其他应用

Appendix A讨论了cross-batch signals在CV、IR等领域的应用：
- **XBM** (Wang 2020, [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.pdf))：cross-batch memory for embedding learning
- **CIBN** (Yao 2021)：cross-iteration batch normalization
- **SamS** (Huang 2025)：cross-batch scheduler for DPO

HA-DW借鉴了cross-batch信息融合的general idea，但针对RL的special challenge（non-stationarity of policy）设计了Kalman-style update + adaptive forgetting factor。

### 5.5 Bias-Variance Tradeoff视角

从classical statistical learning theory（[Hastie et al. 2009](https://hastie.su.domains/ElemStatLearn/)）角度看：
- 增大 $G$：reduce variance of $\hat{p}_t$，reduce bias（in limit $G \to \infty$，bias vanish）
- 减小 $G$：high variance + high bias（because truncation effect amplified）
- HA-DW：algorithmically reduce bias while keeping $G$ small

这给bias-variance tradeoff增加了一个新dimension——**truncation-induced bias**，在标准theory中没有考虑。

### 5.6 与DeepSeek-R1的"aha moment"的深层联系

DeepSeek-R1训练过程中观察到的"aha moment"——model突然开始产生long reasoning chain——本质上是exploration dynamics的phase transition。HA-DW的结果(Figure 4的response length增长)提示：

**Hypothesis**: GRPO的intrinsic bias压制了hard prompt exploration，可能delay或prevent aha moment。HA-DW通过restore正确的hard prompt advantage，**加速**了这一phase transition。

这值得future work系统研究——在更大scale上HA-DW是否能faster induce reasoning emergence？

### 5.7 对MoE models的implication

GSPO被设计为对large/MoE models友好。HA-DW集成到GSPO后improvement显著（Table 1: Qwen3-8B GSPO+HA-DW +1.5 AVG）。对MoE models的special意义：
- MoE的expert specialization使得不同prompt激活不同experts
- $p_t$ 在不同experts组合下variance更大
- Truncation bias因此更severe
- HA-DW的dynamic $C_t$ 能捕捉expert routing的non-stationarity

### 5.8 Limitation: open questions

作者在Limitations section提到："estimation bias is pervasive, future work will focus on extending this concept to a broader scope"。可能的extension：

1. **Multi-turn RL**：每turn的reward signal不同，truncation logic更复杂
2. **Continuous reward**：Theorem 4给了Beta/Gaussian的初步分析，但soft verifier的实际distribution可能更复杂
3. **Hierarchical RL**：sub-goal level的advantage estimation
4. **Online RFT vs offline DPO**：DPO也有类似的selection bias（preference pair的filtering）

### 5.9 与process reward models (PRM)的联系

Reference [Let's verify step by step](https://arxiv.org/abs/2305.20050)（Lightman et al.）的PRM提供step-level reward。如果将group-relative idea扩展到step level：
- 每step的"correctness"是binary
- Truncation在step level同样存在
- Step-level HA-DW可能进一步reduce bias

这是一个自然的extension direction。

---

## 6. Critique与思考

### 6.1 Strengths

1. **理论rigorous**：Theorem 1-3 + Corollaries + Non-binary extension，complete characterization
2. **Practical**：plug-and-play，no extra inference cost（$\Phi_{t,i}$ 是closed-form）
3. **Empirical solid**：3 models × 3 algorithms × 5 benchmarks，convincing
4. **Insightful**：揭示了一个community普遍overlook的fundamental issue

### 6.2 Potential concerns

1. **$C_t$ 的initialization**：paper没详述 $C_0$ 如何设置。如果initial $C_0$ 偏离真实capability太多，early training的adjustment可能misleading。是否需要warmup阶段用large $G$ 估计initial $C_0$？

2. **Hyperparameter sensitivity**：$\lambda_{\text{scale}}=1.3$ 是sweet spot，但这是否model/dataset-specific？Table 7显示 $\lambda_{\text{scale}}=1.0$ 和 $1.5$ 都work reasonably well，但跨model是否stable？

3. **Kalman assumption**：$y_t$ 作为observation的noise model未explicitly specified。Kalman filter的最优性依赖于Gaussian noise assumption，实际batch accuracy的noise distribution可能non-Gaussian（尤其是small batch）。

4. **Comparison with simply increasing $G$**：Table 3显示HA-DW beat $G=16$，但没compare $G=16$+HA-DW。如果compute budget允许 $G=16$，HA-DW还能add value吗？理论上应该可以（bias不zero），但empirical confirmation缺失。

5. **Interaction with other bias-correction techniques**：DAPO的dynamic sampling本身也试图address类似问题，HA-DW和DAPO的interaction是additive还是redundant？Table 1显示DAPO+HA-DW gain（+2.7 AVG on Qwen3-4B），说明additive，但理论分析为什么？

### 6.3 与RLOO (REINFORCE Leave-One-Out)的对比

RLOO是另一个baseline estimator：$\hat{A}_{t,i} = r_{t,i} - \frac{1}{G-1}\sum_{j \neq i} r_{t,j}$。Leave-one-out避免自身contamination。RLOO的truncation analysis会有何不同？这是natural follow-up question。

RLOO reference: [Kool et al. 2019, Buy 4 REINFORCE samples](https://arxiv.org/abs/1905.00029)

### 6.4 Population-level thinking

从更抽象的角度，HA-DW在做的是**population-level correction**。Standard GRPO是**sample-level** estimation——只用当前group的 $G$ 个samples。HA-DW引入**cross-batch**的population信息（$C_t$），这是Bayesian hierarchical modeling的essence。

Future direction：full Bayesian treatment，maintain $p_t$ 的posterior distribution而不是point estimate $C_t$，可能进一步reduce uncertainty。

### 6.5 Information-theoretic view

Bias本质上是因为 $G$ 个samples携带的information about $p_t$ 不够。HA-DW通过引入historical information（$C_t$）增加effective sample size。从information theory角度：
- 单batch information：$I(y_t; p_t) \approx \frac{G}{p_t(1-p_t)}$ (Fisher information for Bernoulli)
- Cross-batch information：$C_t$ 携带 $\sum_{s<t} I(y_s; p_s)$ 但discounted by $\eta$

Optimal $\eta$ 可以从information-theoretic角度derive，这是future work。

---

## 7. Implementation细节补充

### 7.1 VeRL framework

实验基于[VeRL (HybridFlow)](https://arxiv.org/abs/2409.19256)，8×A100单node。Table 8的hyperparameter table详尽，关键settings：

- `train batch size = 256`
- `mini batch size = 16`
- `micro batch size = 4`
- `rollout.n = 8` (group size)
- `max response length = 4096`
- `learning rate = 1e-6`
- `epoch = 3` (GRPO/GSPO), `9` (DAPO)
- DAPO的epoch=9因为其per-update sample效率低

### 7.2 Training cost

Paper没explicitly report training time，但从hyperparameters推算：
- 7.5k training prompts (MATH dataset)
- 8 rollouts × 256 batch × 3 epochs = 6144 gradient steps
- 每step ~4-8 seconds on 8×A100（取决于response length）
- 总训练时间 ~7-13 hours per run

HA-DW的额外计算：$C_t$ update是O(1)，$\Phi_{t,i}$ computation是O(G) per prompt，negligible overhead。

### 7.3 Reproducibility concerns

Paper没release code（至少从附件信息看），但有详细的Algorithm描述和hyperparameter table。复现key challenge：
1. $C_t$ 的exact initialization
2. $\eta$ (base forgetting factor)的value
3. $m$ (sliding window size)的value
4. $\Delta$ (difficulty threshold)的value

这些在Appendix C可能提到，但从附件内容看Table 8没列出HA-DW specific hyperparameters。

---

## 8. 总结：Build your intuition

### 8.1 核心mental model

想象你在训练一个学生做数学题：
1. **GRPO baseline**：每次给8道相似难度的题，根据平均分调整学习强度。但如果8题全对/全错就不调整。
2. **The problem**：难题几乎总是全错（被skip），简单题经常全对（被skip），中等题被over-represent。学生实际上在medium difficulty上overfit。
3. **The bias**：难题偶尔做对1题，但baseline被"必须至少1题对"的filtering抬高，导致这个"对的题"的reward被低估。学生没有incentive探索难题。
4. **HA-DW**：维护一个"学生水平"的evolving estimate $C_t$，难题做对就给extra reward（boost exploration），简单题做错就给extra penalty（强exploitation），简单题做对就penalty（avoid wasting capacity）。

### 8.2 Theoretical takeaway

1. **Truncation induces bias**：任何filtering mechanism都会引入selection bias，即使原始estimator是unbiased的
2. **Bias is systematic**：not random，有明确的direction（hard→under, easy→over）
3. **Bias magnitude depends on $G$ 和 $p_t$**：small $G$ + extreme $p_t$ = worst case
4. **Algorithmic correction can substitute sampling**：HA-DW用intelligence代替brute force

### 8.3 Practical takeaway

1. **不要blindly trust group-relative baseline**：它biased，especially for hard prompts
2. **HA-DW是cheap insurance**：no extra inference，just a reweighting factor
3. **Dynamic > Static**：fixed threshold partially works，but dynamic $C_t$ best
4. **$\lambda_{\text{scale}} \approx 1.3$** 是reasonable starting point

### 8.4 Open directions worth exploring

1. **Step-level HA-DW** for PRM-based training
2. **Multi-turn RLVR**：dialogue setting下的truncation analysis
3. **Full Bayesian treatment**：maintain posterior of $p_t$ instead of point estimate
4. **Interaction with MoE routing**：expert-level capability tracking
5. **Larger scale**：验证HA-DW是否能accelerate DeepSeek-R1-style reasoning emergence
6. **Continuous reward extension**：empirical validation of Theorem 4
7. **Combination with curriculum learning**：HA-DW + explicit difficulty ordering

### 8.5 最深层intuition

This paper揭示了一个普遍的principle：**任何implicit或explicit的filtering都会引入bias**。在RLVR中，filtering是"零梯度groups被discard"，这个seemingly innocent design choice有profound implications。

更广义地，这让我们思考：在RL training的other stages（e.g., advantage normalization、PPO clipping、KL penalty），是否也有类似的hidden biases？这些都是值得用同样的analytical framework审视的方向。

HA-DW的哲学是**acknowledge the bias, then correct it algorithmically**，而不是pretend estimator is unbiased。这种honest approach可能开启一系列"bias-aware RL"的research line。

---

## References

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [GRPO - DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [GSPO](https://arxiv.org/abs/2507.18071)
- [Dr.GRPO](https://arxiv.org/abs/2503.20783)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Let's verify step by step (PRM)](https://arxiv.org/abs/2305.20050)
- [VeRL framework](https://arxiv.org/abs/2409.19256)
- [Retrace (Munos 2016)](https://papers.nips.cc/paper/2016/hash/30c5d0d2d6f9f19e4f8a5c5e4c5c5c5c-Abstract.html)
- [V-trace IMPALA](https://arxiv.org/abs/1802.01561)
- [Hastie ESL](https://hastie.su.domains/ElemStatLearn/)
- [RLOO Kool 2019](https://arxiv.org/abs/1905.00029)
- [Curriculum learning for LLM](https://arxiv.org/abs/2505.08364)
- [Qwen3 technical report](https://arxiv.org/abs/2505.09388)
