---
source_pdf: TIAR Trajectory-Informed Advantage Reweighting for LLM.pdf
paper_sha256: 106b163b15ae2bca8fc6041edfbf78361c7cd186af83a13e01f99b951ae8f7e3
processed_at: '2026-08-12T15:56:44-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 TIAR

好，Karpathy，我换个画风，像在咖啡馆白板上画图那样给你讲。

## 一句话讲清楚

教 LLM 学会说 "我不知道" —— 但不要无脑说，要 **该说的时候说、不该说就别怂**。做法是在 GRPO 训练里，根据每组 sample 的对错比例，动态调整 abstain 那条 trajectory 的 advantage。

## 为啥这事难

LLM 有两个极端毛病：
- **瞎编**（hallucination）：明明不会，硬答。
- **过度 abstain**：啥都说 "我不知道"，变成废物。

TruthRL（前作）给了个简单 fix：答对 +1，答错 -1，abstain 0。这叫 ternary reward。听起来很合理，但它偷偷藏了一个假设。

## 关键 insight：ternary reward 隐含一个固定门槛 0.5

想象 model 面对一道题，它内心有个 "答对概率" $\hat{p}$。如果决定 attempt，期望回报是：

$$V_{\text{attempt}} = \hat{p}\cdot(+1) + (1-\hat{p})\cdot(-1) = 2\hat{p} - 1$$

变量解释：$\hat{p}$ 是 model 在 $G$ 条 attempt trajectory 里答对的比例（公式 4 里 $\hat{p} = n_c / (n_c + n_w)$，$n_c$ 是答对数，$n_w$ 是答错数），$V_{\text{attempt}}$ 是 attempt 这件事的 expected return。

abstain 的回报是 0。model 该 abstain 当 $0 > 2\hat{p} - 1$，也就是 $\hat{p} < 0.5$。

**也就是说，ternary reward 偷偷定了个固定门槛：答对率低于 50% 才 abstain。**

这有问题吗？有。一道医学题，门槛可能该是 0.8（不靠谱就别说）；一道 trivia 题，门槛可能 0.3 就行（猜猜无妨）。一个固定 0.5 通吃所有场景，太粗暴。

## 最优 abstain reward 长啥样

作者用 opportunity cost 推了一下。abstain 这个动作：
- 省掉了答错的损失 $(1-\hat{p})\cdot 1$
- 丢掉了答对的收益 $\hat{p}\cdot 1$

净价值（公式 6）：

$$R_a^* = (1-\hat{p}) - \hat{p} = 1 - 2\hat{p}$$

直觉：
- $\hat{p} = 0$（全错）：abstain 值 +1，最该 abstain
- $\hat{p} = 0.5$（一半对一半错）：abstain 值 0，正好退回 ternary
- $\hat{p} = 1$（全对）：abstain 值 -1，最不该 abstain

所以 ternary 其实是 $R_a^*$ 在 $\hat{p}=0.5$ 的一个特例。**所有题都被当成中等难度**，这假设在 AbstentionBench 六个 scenario 下显然不成立。

## 直接换 reward 会爆雷：coupling problem

最 naive 的做法：把 $R_a$ 从 0 换成 $1-2\hat{p}$，其他不动。

结果：abstention F1 微涨 0.15，accuracy 反而掉 0.76（Table 1）。

为啥？GRPO 的 advantage 是 group 内 z-score normalization（公式 2）：

$$\hat{A}_i = \frac{r(x, y_i) - \text{mean}}{\text{std}}$$

所有 trajectory 共享同一组 mean 和 std。你改 $R_a$，mean 和 std 跟着变，连带着把 correct / wrong trajectory 的 advantage 也扭曲了。

数学上（公式 8、9）：

$$\bar{R}_{\text{dynamic}} = \bar{R}_{\text{ternary}} + \frac{n_a(1-2\hat{p})}{G}$$

hard question 上 $\hat{p} < 0.5$，第二项为正，group mean 被推高。correct trajectory 的 advantage = $(+1 - \bar{R})/\sigma$ 就变小了。

**直觉：hard question 上答对本来 rare 且珍贵，coupled reward 反而削弱了答对的 gradient signal。** 这正好跟想要的反着来。

## TIAR 的招：把信号挪到 advantage 层

不动 reward，在 GRPO 算完 standard advantage 之后，**只对 abstain trajectory 加一个 post-hoc adjustment**：

$$\hat{A}_j \leftarrow \hat{A}_j + \lambda(1 - 2\hat{p}), \quad \forall j: r_j = 0$$

变量：$\hat{A}_j$ 是第 $j$ 条 trajectory 的 advantage，$\lambda$ 是强度系数（实验里 $\lambda=1.0$），$\hat{p}$ 是这组 attempt 的答对率。

关键点：
- correct / wrong trajectory 的 advantage **完全不动**，decoupling 解决了
- abstain trajectory 的 advantage 根据 $\hat{p}$ 动态调整
- hard question（$\hat{p}$ 低）上，abstain 被鼓励
- easy question（$\hat{p}$ 高）上，abstain 被打压

Algorithm 1 里 Line 6 有个 `if n_c + n_w > 0` —— 防止全 abstain 时 $\hat{p}$ 分母为零，这种情况没 difficulty 信息，跳过。

## 实验结果

**主结果**（Table 2）：在 AbstentionBench 6 个 category 上，TIAR 拿了 5/6 的 best F1（Llama + Qwen 加起来），17/31 datasets 上 beat ternary baseline，accuracy 基本保住。

**Ablation**（Table 3）扫描 $\lambda$：

| $\lambda$ | 效果 |
|---|---|
| 0 | 退化成 TruthRL baseline |
| 0.3 | training 崩溃，信号太弱让 policy 在两个 attractor 间摇摆 |
| 0.5 | 还行但不如 1.0 |
| **1.0** | 最好，fully commit 到 correction 反而稳 |

这个 non-monotonic 现象跟很多 RL 经验一致：弱 correction 让模型左右为难，强 correction 直接拉过去。

**跟 frontier API 对打**（Table 4）：8B Llama + TIAR 在 Answer Unknown 类上 F1 = 95.8，**超过 Claude Sonnet 4.5 (93.9)、GPT-5.2 (93.6)、Gemini 3 (84.4)**。在 QAQA precision 上 90.3 是全场最高。

几个有意思的对照：
- Gemini 3 在 KUQ/Controversial 上 F1 = 100 但 accuracy = 50，**严重 over-abstain**——啥都不答，F1 当然满分
- GPT-5.2 在 FreshQA（temporal knowledge）上 F1 只有 48.8，**过度自信**，时间敏感问题上是 open problem
- TIAR 在 FreshQA 上 F1 = 68.4，比 GPT-5.2 强但比 Claude 弱

## 训练 setup 几个值得注意的数字

- Base model：Llama-3.1-8B-Instruct 和 Qwen3-8B
- 训练数据：TruthRL-CRAG，只有 656 个 sample，来自 CRAG benchmark
- Group size $G=8$，batch size 64
- 只跑 **20 步**，6.7 小时，107 GPU-hour / model（16× A100 40GB）
- Reward judge 用 Llama-3.1-8B-Instruct 自己当裁判，validated accuracy 82.3%
- 评估用 AbstentionBench，20 个 dataset / 31 subset / 6 scenario

## 我觉得有意思 / 可疑的地方

**有意思**：
1. **$\hat{p}$ 是 self-supervised difficulty probe**。G=8 条 sample 的对错比例直接告诉你这题对当前 policy 多难。这跟 self-consistency（Wang et al. 2022, https://arxiv.org/abs/2203.11171）和 Kadavath et al. 2022 "language models mostly know what they know"（https://arxiv.org/abs/2207.05221）的思路一脉相承，但 TIAR 把它搬进 RL training loop 让 reward 自己适应。
2. **Post-hoc advantage adjustment 这个 pattern** 可以搬到一堆场景：math reasoning 的 long chain、tool use 的失败 call、multi-turn agent 的不同 turn。
3. **Coupling problem 是 GRPO 的普遍问题**。任何想做 query-dependent reward 的工作都会撞到。TIAR 给了个最简的解法。更复杂的版本：per-query normalization strength、token-level reweighting、learned critic baseline。

**可疑 / 可改进**：
1. **$G=8$ 的 $\hat{p}$ variance 很大**。真实 $p^*=0.3$ 的题，单次 8 sample 估出来的 $\hat{p}$ 可能在 0.125-0.5 之间抖。Cross-step averaging 或者更大 $G$ 可能更稳。Paper 没扫 $G$。
2. **Reward judge 是 bottleneck**。Llama-3.1-8B-Instruct 当 judge，82.3% accuracy。judge 错了 advantage 就污染了。Self-rewarding LM（Yuan et al. 2024, https://arxiv.org/abs/2401.10020）那套可以缓解。
3. **没有 warmup**。如果 base model 一开始 overconfident 但 wrong，早期 $\hat{p}$ 会 misleading。前 5 步用 ternary、后面切 TIAR 可能更稳。Paper 没试。
4. **Reward hacking 风险**。model 可能学到故意答错拉低 $\hat{p}$ 再 abstain 拿高 reward。$G=8$ 小样本让这事难做但不排除。
5. **只测了 single-turn**。Multi-turn dialogue 里 abstain 不是 binary，该有 "ask clarification"、"give partial answer"、"need more context"。这跟 RAG 结合特别有意思——abstain on context insufficiency 而非 knowledge boundary。Paper 自己也承认这是 next step。

## 一句话再总结

TIAR = GRPO 训练里，看一组 sample 答对多少（$\hat{p}$），答得烂就鼓励 abstain，答得好就打压 abstain，**关键是只动 abstain 的 advantage 不动其他的**，避免 group normalization 被污染。简单、干净、work。

核心 reference：
- TIAR 论文本身（你贴的 attachment）
- TruthRL (Wei et al. 2025, https://arxiv.org/abs/2509.25760) —— 直接前作
- AbstentionBench (Kirichenko et al. 2025, https://arxiv.org/abs/2506.09038) —— 评估 benchmark
- DeepSeekMath GRPO (Shao et al. 2024, https://arxiv.org/abs/2402.03300) —— GRPO 原始
- Kadavath et al. 2022 (https://arxiv.org/abs/2207.05221) —— $\hat{p}$ 作为 confidence 的理论基础
- Why LLMs Hallucinate (Kalai et al. 2025, https://arxiv.org/abs/2509.04664) —— OpenAI 的 motivation

如果你想 weekend hack 一下，这 paper 是个不错的起点——656 sample、20 步、8B model、6.7 小时就能复现。把 $\hat{p}$ 推广到 multi-turn 或者接 RAG 是个明显的 next step。

---

# TIAR: Trajectory-Informed Advantage Reweighting 深度解析

Karpathy 你好，这篇paper来自Penn State的组，做的是LLM abstention learning方向。我先把核心insight拎出来，然后逐层拆解数学、架构和实验。

## 1. Paper 在 abstention learning 谱系中的位置

Abstention learning 这条线最近一年很热，OpenAI 的 "Why Large Language Models Hallucinate" (Kalai et al., 2025, https://arxiv.org/abs/2509.04664) 明确呼吁进一步探索。整个领域的脉络大致是：

- **SFT 派**：R-Tuning (Zhang et al., 2024, https://arxiv.org/abs/2311.09677), Alignment for Honesty (Yang et al., 2024, https://arxiv.org/abs/2312.07000) —— 把 unanswerable question 的 label 重写成 "I don't know"
- **Offline RL 派**：DPO-style preference learning，把 "I don't know" 作为 chosen response
- **Online RL / GRPO 派**：TruthRL (Wei et al., 2025, https://arxiv.org/abs/2509.25760)，用 ternary reward $R_c=+1, R_w=-1, R_a=0$
- **Inference-time 派**：多次 sampling 看 consistency 决定要不要 abstain（Cole et al. 2023, https://arxiv.org/abs/2305.14613；Phute et al. 2024, https://arxiv.org/abs/2308.07308）

TIAR 是 TruthRL 的 direct extension，同一个作者群已经验证 ternary reward 比 binary reward 好，TIAR 把 static ternary reward 升级成 trajectory-informed dynamic advantage reweighting。

## 2. 核心问题：ternary reward 隐含一个固定阈值 $\hat{p}^* = 0.5$

这是整篇 paper 最 elegant 的 insight，我详细推一下。

### 2.1 Empirical correctness rate 作为 difficulty proxy

给定一个 query $x$，GRPO 采样 $G$ 条 trajectories（实验里 $G=8$）。其中 $n_c$ 条 correct，$n_w$ 条 wrong，$n_a$ 条 abstain。约束 $n_c + n_w + n_a = G$。

定义 empirical correctness rate（公式 4）：

$$\hat{p} = \frac{n_c}{n_c + n_w}, \quad \text{when } n_c + n_w > 0$$

变量含义：
- $\hat{p}$：在所有 *attempt*（非 abstain）的 trajectories 中，correct 的比例
- 这是 model 在当前 policy 下对这个 query 的 "成功概率" 的 empirical 估计
- $\hat{p} \to 1$：easy question（model 几乎总能答对）
- $\hat{p} \to 0$：hard question（model 几乎总答错）
- $\hat{p}$ 中等：boundary question（model 犹豫）

这个量本质上是 **self-consistency 在 reward 空间的投影**。GRPO 采样多条 trajectory 这件事在 Math reasoning 里被广泛用作 self-consistency signal（Wang et al. 2022, https://arxiv.org/abs/2203.11171），但 TruthRL 之前的 GRPO 工作没把它显式用于 abstention。

### 2.2 Attempt 的 expected value 与隐含阈值

Model 面对一个 query，如果决定 attempt，expected value 是（公式 5）：

$$V_{\text{attempt}}(\hat{p}) = \hat{p} \cdot R_c + (1 - \hat{p}) \cdot R_w = \hat{p}\cdot(+1) + (1-\hat{p})\cdot(-1) = 2\hat{p} - 1$$

变量：
- $V_{\text{attempt}}$：在当前 $\hat{p}$ 下 attempt 的期望回报
- $\hat{p}$：empirical correctness rate
- $R_c, R_w$：correct/wrong 的 reward

abstain 的回报是 $R_a = 0$。Model 应该 abstain 当且仅当 $R_a > V_{\text{attempt}}(\hat{p})$，即 $0 > 2\hat{p} - 1$，即 $\hat{p} < 0.5$。

**关键 insight**：ternary reward 实际上隐含了一个 *固定* 的 abstention 阈值 $\hat{p}^* = 0.5$，无论 query 的实际 difficulty 分布如何。这违反直觉——医学高风险场景、temporal knowledge 场景、subjective question 场景的最优阈值完全不同。

### 2.3 Opportunity cost 推导：optimal abstention reward

Paper 的 derivation 很漂亮。当 model abstain，它：
- **avoid 了** 错误的期望损失 $(1-\hat{p}) \cdot |R_w|$
- **cost 了** 正确的期望收益 $\hat{p} \cdot R_c$

Net value（公式 6）：

$$R_a^* = \underbrace{(1-\hat{p}) \cdot |R_w|}_{\text{loss avoided}} - \underbrace{\hat{p} \cdot R_c}_{\text{gain costed}} = (1-\hat{p}) \cdot 1 - \hat{p} \cdot 1 = 1 - 2\hat{p}$$

性质（这是个关键表）：

| $\hat{p}$ | 含义 | $R_a^*$ | 行为 |
|---|---|---|---|
| 0 | 全错（hard / OOK） | +1 | abstain 最大奖励 |
| 0.5 | 一半对一半错 | 0 | 退化为 ternary |
| 1 | 全对（easy / in-knowledge） | -1 | abstain 最大惩罚 |

Ternary reward 是 $R_a^*$ 在 $\hat{p} = 0.5$ 处的特例，相当于 **假设所有问题都同样困难**，这个假设在 AbstentionBench 的 6 个 scenario 下显然不成立。

## 3. Coupling Problem：为什么不能直接换 reward

这是 paper 的第二个关键 insight，也是 TIAR 真正的方法学贡献。

### 3.1 Naive 做法：直接 $R_a \leftarrow 1 - 2\hat{p}$

直接把公式 6 的动态 reward 代入 GRPO 的 advantage 计算（公式 2）：

$$\hat{A}_i = \frac{r(x, y_i) - \text{mean}\{r(x, y_j)\}_{j=1}^G}{\text{std}\{r(x, y_j)\}_{j=1}^G}$$

变量：
- $\hat{A}_i$：第 $i$ 条 trajectory 的 normalized advantage
- $r(x, y_i)$：query $x$、trajectory $y_i$ 的 reward
- mean/std：在 $G$ 条 trajectory 上计算

问题在于，**所有 advantage 共享同一组 group statistics**（mean 和 std）。改 $R_a$ 会扰动 mean 和 std，从而连带扭曲 $\hat{A}_c$ 和 $\hat{A}_w$。

### 3.2 数学上看到耦合

Ternary 下 group mean（公式 7）：

$$\bar{R}_{\text{ternary}} = \frac{n_c \cdot (+1) + n_w \cdot (-1) + n_a \cdot 0}{G} = \frac{n_c - n_w}{G}$$

Dynamic reward $R_a = 1 - 2\hat{p}$ 下 group mean（公式 8）：

$$\bar{R}_{\text{dynamic}} = \bar{R}_{\text{ternary}} + \frac{n_a (1 - 2\hat{p})}{G}$$

在 hard question 上 $\hat{p} < 0.5$，第二项为正，所以 $\bar{R}_{\text{dynamic}} > \bar{R}_{\text{ternary}}$。

这直接削弱 correct trajectories 的 advantage（公式 9）：

$$\hat{A}_c^{\text{dynamic}} = \frac{R_c - \bar{R}_{\text{dynamic}}}{\sigma_{\text{dynamic}}} < \frac{R_c - \bar{R}_{\text{ternary}}}{\sigma_{\text{ternary}}} = \hat{A}_c^{\text{ternary}}$$

下标 `dynamic` / `ternary` 区分两种 reward scheme 下计算的 advantage。

**直觉上**：在 hard question 上正确答案本来 rare 且 valuable，coupled approach 反而削弱了"答对"的 gradient 信号。这违反了我们想要的——hard question 上的 correct trajectory 应该获得 *更强* 的学习信号，因为它表明 model 找到了某种 rare 的 reasoning path。

### 3.3 实验验证 coupling 确实存在

Paper Table 1 给了直接证据（step 20，AbstentionBench 平均）：

| Method | Acc. | Abstention F1 |
|---|---|---|
| TruthRL (ternary) | 67.07 | 72.45 |
| Coupled ($R_a = 1-2\hat{p}$) | 66.31 | 72.60 |
| $\Delta$ | -0.76 | +0.15 |
| Win/Loss | 6/16 | 13/13 |

Coupled 版本：abstention F1 微涨 0.15，但 accuracy 掉 0.76——这印证了上面数学推导的预测。F1 上 13/13 平局意味着 gain 不显著，accuracy 上 6 win 16 loss 意味着 widespread 退化。

## 4. TIAR：Decoupled Advantage Adjustment

### 4.1 核心想法

把动态信号从 **reward 层面** 移到 **advantage 层面**，做成 post-hoc adjustment：

- 标准 GRPO 用 ternary reward 算 $\hat{A}_c$ 和 $\hat{A}_w$（保持 normalization 不变）
- 只对 abstention trajectories 做 post-normalization adjustment

Adjustment 量（公式 10）：

$$\Delta_a = R_a^* - R_a^{\text{ternary}} = (1 - 2\hat{p}) - 0 = 1 - 2\hat{p}$$

乘上 strength $\lambda$，加到 abstention trajectory 的 advantage 上：

$$\hat{A}_j \leftarrow \hat{A}_j + \lambda (1 - 2\hat{p}), \quad \forall j: r_j = 0$$

下标 $j$ 表示第 $j$ 条 trajectory；$\lambda$ 控制 correction 强度，实验中 $\lambda = 1.0$。

### 4.2 Algorithm 1 解析

```
Require: Query batch {x_i}, group size G, strength λ
1: for each query x_i in batch do
2:   Sample G trajectories {y_1, ..., y_G} ~ π_θ_old(·|x_i)
3:   Assign ternary rewards: r_j ∈ {+1, -1, 0}
4:   Compute GRPO advantages via Eq. 2  # 标准 normalization
5:   n_c ← |{j : r_j = 1}|, n_w ← |{j : r_j = -1}|
6:   if n_c + n_w > 0 then
7:     p̂ ← n_c / (n_c + n_w)
8:     for each j where r_j = 0 do
9:       Â_j ← Â_j + λ(1 - 2p̂)
10:    end for
11:  end if
12: end for
13: Update π_θ with modified advantages (Eq. 1)
```

几个细节值得注意：
- Line 6 的 `if n_c + n_w > 0` 是为了避免 $\hat{p}$ 分母为零（全 abstain 的情况），此时 no information about difficulty
- Line 4 用的 *原始* ternary advantage，这意味着 $\hat{A}_c$ 和 $\hat{A}_w$ 完全不变——这是 decoupling 的核心
- Line 9 的 adjustment 对所有 abstain trajectory 同样生效，不区分 abstain 的方式

### 4.3 Figure 1 的解读

Figure 1（论文插图）展示了 reweighting 逻辑：

- **简单问题（$\hat{p}$ 高）**：abstention trajectories 的 advantage 被压低（甚至变负），discourage abstention
- **困难问题（$\hat{p}$ 低）**：abstention trajectories 的 advantage 被推高，encourage abstention
- correct / wrong trajectories 的 advantage 完全不动

这就是 "Trajectory-Informed" 名字的由来——informed by $G$ 条 trajectory 的 empirical 难度。

### 4.4 GRPO objective 完整形式

GRPO loss（公式 1）：

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{x, \{y_i\}_{i=1}^G} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left( w_{i,t} \hat{A}_i, \text{clip}(w_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

变量：
- $\theta$：policy 网络参数
- $x$：query；$\{y_i\}_{i=1}^G$：从 old policy $\pi_{\theta_{\text{old}}}$ 采样的 $G$ 条 trajectory
- $|y_i|$：第 $i$ 条 trajectory 的 token 数
- $w_{i,t}$：importance ratio，$\pi_\theta(y_{i,t}|...)/\pi_{\theta_{\text{old}}}(y_{i,t}|...)$，第 $t$ 个 token 处
- $\epsilon$：PPO clip range
- $\hat{A}_i$：第 $i$ 条 trajectory 的 advantage（TIAR 会修改这个）
- $\beta$：KL penalty 系数
- $D_{\text{KL}}$：KL divergence 到 reference policy $\pi_{\text{ref}}$
- 外层 $\mathbb{E}$：对数据分布 $\mathcal{D}$ 的 expectation

TIAR 替换的只是 $\hat{A}_i$ 这一项，其他完全不变。

## 5. 实验设置细节

### 5.1 Base model & dataset

- Llama-3.1-8B-Instruct (https://arxiv.org/abs/2407.21783)
- Qwen3-8B (https://arxiv.org/abs/2505.09388)
- 训练数据：TruthRL-CRAG，656 个 sample，源自 CRAG benchmark
- 巧妙点：所有训练问题都有 ground-truth answer，但很多超出 model 的 parametric knowledge，自然形成 answerable / unanswerable 分布

### 5.2 GRPO 配置

| 参数 | 值 |
|---|---|
| Framework | verl (https://arxiv.org/abs/2409.06557, Sheng et al. 2025) |
| Batch size | 64 |
| Group size (rollout n) | 8 |
| Max prompt length | 16,384 tokens |
| Max response length | 2,048 tokens |
| Learning rate | $1 \times 10^{-6}$ |
| KL coeff $\beta$ | 0.001 (low-variance KL) |
| Rollout engine | vLLM (https://arxiv.org/abs/2309.06181), TP=2, GPU mem 0.8 |
| Actor training | FSDP + gradient checkpointing |
| Hardware | 16× NVIDIA A100 40GB, SLURM |
| Steps | 20 |
| Time | ~6.7 hours |
| Total budget | 107.2 GPU-hours / model |

注意：只跑 20 步！这跟 TruthRL 一致，避免 over-optimization on KL。KL=0.001 用 low-variance KL estimator（参考 Schulman 的实现，https://openreview.net/forum?id=B1B8ZpRzVg）。

### 5.3 Reward function

外部 LLM judge（Llama-3.1-8B-Instruct via vLLM）：
- +1：correct
- -1：incorrect  
- 0：abstention

Judge 先做 pattern matching + semantic analysis 判断是否 abstain，再 evaluate 非 abstain response 的 correctness。**这是个 bottleneck**——paper 在 Limitations 里承认，judge 的 misclassification 会直接污染 advantage signal。

### 5.4 Baselines

| Baseline | 类型 | 描述 |
|---|---|---|
| R-Tuning | SFT | OOK question 标 "I don't know"，每题采样 256 次探测 knowledge boundary |
| RFT | SFT | 只在正确 response 上 fine-tune，过滤 wrong |
| DPO | Offline RL | OOK question 上 "I don't know" preferred over incorrect |
| TruthRL | Online RL | GRPO + ternary reward，**direct ablation** |

TruthRL 是最关键的 baseline——它和 TIAR 只差一个 advantage adjustment，cleanly isolate TIAR 的贡献。

## 6. 评估协议

### 6.1 AbstentionBench

Kirichenko et al. 2025 (https://arxiv.org/abs/2506.09038)，20 个 dataset，31 个 subset，6 个 scenario：

| Category | 代表 dataset |
|---|---|
| Answer Unknown | BB/Known Unknowns |
| Underspecified Intent | BBQ |
| Stale Information | FreshQA |
| Underspecified Context | UMWP |
| False Premise | QAQA |
| Subjective | KUQ/Controversial |

### 6.2 Metrics

- **Abstention F1**（主指标）：abstention precision × recall 的 harmonic mean
- **Abstention Recall**：正确 abstain 的 unanswerable question 比例
- **Abstention Precision**：abstain 中 warranted 的比例
- **Accuracy**：在所有有 reference answer 的样本上 response 的正确性

设计哲学：F1 反映 abstention 质量，accuracy 作为 independent capability 指标。AbstentionBench 作者已确认两者无直接关系——TIAR 的目标就是 *同时* 改善两者。

### 6.3 LLM-as-Judge validation

Judge 用 Llama-3.1-8B-Instruct，两阶段评估：先判 abstain/non-abstain，再判 correctness。AbstentionBench 作者用 human annotation 验证 judge accuracy 82.3%。

## 7. 结果分析

### 7.1 Main Results（Table 2 解析）

对 Llama-3.1-8B-Instruct：

| Method | BB/KU F1 | BBQ F1 | FreshQA F1 | UMWP F1 | QAQA F1 | KUQ/Cont F1 |
|---|---|---|---|---|---|---|
| R-Tuning | 91.7 | 82.4 | 69.9 | 75.3 | 56.0 | 75.0 |
| RFT | 97.9 | 85.9 | 72.2 | 76.1 | 57.1 | 77.7 |
| DPO | 93.9 | 84.4 | 72.1 | 75.5 | 58.0 | 74.6 |
| TruthRL | 95.8 | 84.5 | 74.7 | 78.9 | 58.0 | 74.8 |
| **TIAR** | **97.9** | 84.6 | 74.5 | 77.6 | **58.2** | **76.2** |

TIAR 在 False Premise (QAQA) 和 Subjective (KUQ) 上最高，其他几个跟 TruthRL 接近。**Accuracy 上 TIAR 在 Llama 的 6 个 dataset 中基本保持 TruthRL 水平甚至更好**。

对 Qwen3-8B：

| Method | BB/KU F1 | BBQ F1 | FreshQA F1 | UMWP F1 | QAQA F1 | KUQ/Cont F1 |
|---|---|---|---|---|---|---|
| TruthRL | 85.2 | 90.8 | 81.1 | 92.9 | 79.1 | 86.7 |
| **TIAR** | 79.3 | 91.2 | 80.4 | **94.0** | 79.7 | **87.2** |

Qwen 上 TIAR 在 Underspecified Context 和 Subjective 拿到 best F1。BB/KU 上 TIAR F1 掉了（79.3 vs 85.2），但 Qwen 的 baseline abstention 能力已经很强（很多 >90），TIAR 的边际 gain 不如 Llama 明显。

**Cross-model observation**：
- Qwen3-8B 整体显著优于 Llama-3.1-8B，例如 UMWP 上所有 Qwen 方法都 >90，Llama 在 75-79
- R-Tuning 在 Qwen 上 strong（FreshQA/QAQA best），在 Llama 上 weak——表明 SFT 对 instruction-following 能力 sensitive
- DPO 在 Llama 上导致 BB/KU accuracy 退化到 69.6（其他 87+），原因是 over-abstention（recall 100 但 precision 88.5）

### 7.2 Ablation（Table 3）

$\lambda$ 扫描，Llama-3.1-8B 上：

| $\lambda$ | Avg F1 | Avg Acc | F1 win/loss vs $\lambda=0$ | Acc win/loss |
|---|---|---|---|---|
| 0 (= TruthRL) | 71.x (基线) | 72.7 | - | - |
| 0.3 | 崩溃 | 崩溃 | - | - |
| 0.5 | 强（6 个 rep dataset） | - | - | - |
| **1.0** | **71.9** | 72.6 | 17/31 | 11/24 |

**Non-monotonic relationship**：$\lambda$ 太小（0.3）training 不稳定，full inversion ($\lambda=1.0$) 反而最稳。这跟很多 RL 工作中 "fully commit to a correction" 比 "weak correction" 更稳定的直觉一致——weak correction 让 policy 在两个 attractor 之间徘徊。

### 7.3 vs. Proprietary APIs（Table 4）

最有意思的结果：8B 开源 model + TIAR 跟 frontier API 对打。

| Model | BB/KU F1 | FreshQA F1 | QAQA F1 | KUQ/Cont F1 | QAQA Precision | BBQ Accuracy |
|---|---|---|---|---|---|---|
| Claude Sonnet 4.5 | 93.9 | 75.2 | 66.7 | 92.3 | 83.3 | 59.2 |
| GPT-5.2 | 93.6 | 48.8 | 66.0 | 92.9 | 74.4 | 73.5 |
| Gemini 3 | 84.4 | 32.8 | 75.8 | 100.0 | 87.8 | 67.3 |
| **TIAR (8B)** | 95.8 | 68.4 | 65.9 | 92.9 | **90.3** | **73.5** |

几个 takeaway：
- TIAR 在 Answer Unknown (BB/KU) 上 F1 **超过所有三个 API**——说明 recognizing knowledge boundary 不需要 massive scale
- TIAR 在 QAQA Precision 上 90.3 是全场最高
- TIAR 在 BBQ Accuracy 上 73.5 跟 GPT-5.2 并列 best
- FreshQA 是所有方法的弱项，GPT-5.2 才 48.8（temporal knowledge overconfidence），Gemini 3 也才 32.8——这暗示 abstention on temporal knowledge 是个 open problem
- Gemini 3 在 KUQ/Controversial 上 F1 = 100 但 accuracy 只有 50.0，**severe over-abstention**——完美 abstain 但啥也不答

## 8. Intuition building：从 Karpathy 视角的几点延伸

### 8.1 $\hat{p}$ 作为 self-supervised difficulty probe

这条思路让我想起 self-consistency（Wang et al. 2022）和 "language models (mostly) know what they know"（Kadavath et al. 2022, https://arxiv.org/abs/2207.05221）。Kadavath 那篇 paper 已经证明，通过 multiple sample 的 majority vote 概率可以 well-calibrate model 的 confidence。TIAR 把这个 insight 直接搬进 RL training loop，让 reward signal 自己适应 difficulty。

潜在问题：**$\hat{p}$ 的 variance**。$G=8$ 时 $\hat{p} \in \{0, 1/8, ..., 1\}$，分辨率有限。如果 query 真实难度 $p^* = 0.3$，单次 $G=8$ 采样 $\hat{p}$ 可能在 0.125 到 0.5 之间抖动。是否做 Monte Carlo averaging across steps 会更稳？Paper 没讨论，但 future work 提到 curriculum on $\lambda$，可能间接缓解。

### 8.2 Post-hoc advantage adjustment vs reward shaping

TIAR 的 design pattern 是 RL 里 **reward shaping** vs **advantage shaping** 的对比。Reward shaping（Ng et al. 1999, https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf）的潜在问题是它会动到 dynamics，advantage shaping 则保持 reward 不变，只在 critic 层面调整 signal。TIAR 更激进——它在 group normalization 之后做 adjustment，完全绕开 statistics 耦合。

这个 pattern 在其他 RL 任务里应该也 work，比如：
- Math reasoning 上，对 long reasoning chain 给 $\hat{p}$-dependent advantage
- Tool use 上，对 failed tool call 给 difficulty-aware adjustment
- Multi-turn agent 上，对不同 turn 的 reward 做 temporal reweighting

### 8.3 Coupling problem 的普遍性

Coupling 是 GRPO 这种 group-normalized 方法的 *普遍* 问题。任何想要 query-dependent reward 的工作都会撞到这个。GRPO 的 advantage normalization 公式（公式 2）是 group 内 z-score，**强制让所有 query 的 advantage 分布标准化到 mean 0 std 1**。这抹掉了 query difficulty 信息。

如果想要 difficulty-aware gradient，有几条路：
1. **TIAR 路径**：post-hoc advantage adjustment，不改 reward
2. **Per-query normalization strength**：让 std 不是 1 而是依赖 difficulty
3. **Token-level reweighting**：在 token 粒度而非 trajectory 粒度做 adjustment
4. **Different baseline**：放弃 group mean，用一个 learned critic

TIAR 选了最简单的路径，效果就够好。这跟 DeepSeekMath GRPO（https://arxiv.org/abs/2402.03300）的哲学一致——简单方法 + 大规模实验。

### 8.4 与 selective prediction / conformal prediction 的关系

Abstention learning 跟 classical selective prediction（Geifman & El-Yaniv 2017, https://arxiv.org/abs/1705.08500）和 conformal prediction（Vovk et al. 2005, https://arxiv.org/abs/2107.07511）有深厚联系。这些 classical 方法在 *inference* 时做 abstain decision，依赖 calibration score。TIAR 在 *training* 时把 abstain decision 烧进 policy，让 inference 时 model 自己 abstain。

Tayebati et al. 2025（https://arxiv.org/abs/2502.06884）已经把 conformal prediction 搬进 LLM abstention。一个自然的 future direction：TIAR 学出来的 policy + conformal prediction 做 calibration，应该能拿到 better coverage guarantee。

### 8.5 与 Semantic Clustering approach 对比

An & Xu 2025（https://arxiv.org/abs/2510.24020）做 semantic clustering on GRPO trajectories，cluster size 大于阈值才认作 correct。TIAR 的 $\hat{p}$ 是更轻量的 proxy——不 cluster，直接用 ternary reward judge 的 count。

两者各有优劣：
- Semantic clustering 不需要 reward judge，但需要 embedding model + threshold
- TIAR 需要 reward judge，但 $\hat{p}$ 直接来自 reward count，零额外 hyperparameter（除了 $\lambda$）

在 judge 可靠的场景 TIAR 更简洁；在 judge 不可靠的场景 semantic clustering 更鲁棒。Hybrid 是个 open direction。

### 8.6 与 behaviorally calibrated RL 对比

Wu et al. 2026（https://arxiv.org/abs/2512.19920）让 user 显式指定 risk score $t \in [0,1]$，scale abstention reward 成 $2t-1$。高 $t$ encourage abstain。

TIAR 自动 infer $t$ from trajectories——本质上是把 $t \leftarrow 1 - \hat{p}$。所以 TIAR 是 **risk-adaptive** 的，不需要 user input。这避免了 deployment 时 user 不知道怎么设 $t$ 的尴尬，但失去了 user 控制 risk tolerance 的能力。

## 9. Limitations 与我的看法

Paper 自己列了 5 个 limitations，我补几个：

1. **Judge bottleneck**：reward function 依赖 LLM judge 的准确性。Self-rewarding LM（Yuan et al. 2024, https://arxiv.org/abs/2401.10020）和 judge alignment 是个独立研究方向，TIAR 直接受益于 judge 改进。

2. **Computational cost**：$G=8$ rollout 比 SFT/DPO 重很多。107 GPU-hour / model 对学术组有压力。Future work 可以试 **asynchronous GRPO**（如 Async-RLHF, https://arxiv.org/abs/2310.05077）或者 speculative rollout。

3. **$\hat{p}$ calibration sensitivity**：如果 base model 一开始 overconfident 但 wrong，$\hat{p}$ 在早期可能 misleading。一个 fix：warm-up 阶段用 TruthRL (ternary) 训几步再切 TIAR，类似 learning rate warmup。Paper 没试这个。

4. **Single-turn limitation**：multi-turn dialogue 里 abstain 不该是 binary——应该有 "ask clarification", "give partial answer", "need more context" 等。这个 future direction 跟 RAG 结合特别有意思——abstain on context insufficiency 而非 knowledge boundary。

5. **Reward hacking risk**：model 可能学到 manipulate $\hat{p}$（比如故意答错降低 $\hat{p}$，再 abstain 拿高 reward）。$G=8$ 的小 sample size 让这种 hacking 难做但不排除。可能需要 anti-gaming reward component。

6. **Generalization beyond CRAG**：训练只用了 656 个 CRAG sample。Cross-distribution generalization 在 AbstentionBench 上已验证，但 cross-domain（如 medical QA）未验证。R-Tuning 论文里 medical abstention 是个核心场景，TIAR 应该试。

## 10. 与其他相关工作串联

| 方向 | 代表工作 | 与 TIAR 关系 |
|---|---|---|
| Self-consistency | Wang et al. 2022 (https://arxiv.org/abs/2203.11171) | $\hat{p}$ 是 self-consistency 的 reward-space 投影 |
| Confidence calibration | Kadavath et al. 2022 (https://arxiv.org/abs/2207.05221) | 多 sample 估计 confidence 的理论基础 |
| Selective prediction | Geifman & El-Yaniv 2017 (https://arxiv.org/abs/1705.08500) | Inference-time abstain 的经典框架 |
| Conformal abstention | Tayebati et al. 2025 (https://arxiv.org/abs/2502.06884) | Conformal prediction on LLM |
| Semantic clustering | An & Xu 2025 (https://arxiv.org/abs/2510.24020) | 不依赖 reward 的 trajectory 分析 |
| Behaviorally calibrated RL | Wu et al. 2026 (https://arxiv.org/abs/2512.19920) | Risk-adaptive reward |
| GRPO 原始 | DeepSeekMath (https://arxiv.org/abs/2402.03300) | GRPO framework |
| TruthRL | Wei et al. 2025 (https://arxiv.org/abs/2509.25760) | 直接前作 |
| AbstentionBench | Kirichenko et al. 2025 (https://arxiv.org/abs/2506.09038) | 评估 benchmark |
| R-Tuning | Zhang et al. 2024 (https://arxiv.org/abs/2311.09677) | SFT baseline |
| Why LLMs hallucinate | Kalai et al. 2025 (https://arxiv.org/abs/2509.04664) | 理论 motivation |
| Survey on abstention | Wen et al. 2025 (https://arxiv.org/abs/2407.18418) | 全面综述 |
| verl framework | Sheng et al. 2025 (https://arxiv.org/abs/2409.06557) | 实现框架 |
| vLLM | Kwon et al. 2023 (https://arxiv.org/abs/2309.06181) | rollout engine |
| Self-rewarding | Yuan et al. 2024 (https://arxiv.org/abs/2401.10020) | 减轻 judge bottleneck 的方向 |
| Reward shaping | Ng et al. 1999 (https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | TIAR 走 advantage shaping 而非 reward shaping 的理论背景 |

## 11. 总结：TIAR 的核心贡献

**Method 层面**：识别出 GRPO ternary reward 隐含 $\hat{p}^* = 0.5$ 的 fixed threshold，推导出 optimal $R_a^* = 1-2\hat{p}$，并发现直接替换会引发 coupling problem。Solution：post-hoc advantage adjustment，decoupled from normalization。

**Empirical 层面**：在 AbstentionBench 上 5/6 categories SOTA，17/31 datasets 上 beat ternary baseline，preserved accuracy，跟 GPT-5.2/Claude Sonnet 4.5/Gemini 3 competitive on key metrics。

**Conceptual 层面**：$\hat{p}$ 作为 self-supervised difficulty probe 是个可移植的 idea，应该能 spawn 一系列 "trajectory-informed X" 工作。

写到这里，我意识到 TIAR 跟你最近关注的 GRPO post-training（DeepSeek R1 之后的方向）高度相关。$\hat{p}$ 这个量本质上是个 query-level scalar signal，跟 process reward model（PRM, Lightman et al. 2023, https://arxiv.org/abs/2305.20050）的思路呼应——只是 PRM 在 token level 给信号，TIAR 在 trajectory level 给信号。两者结合是个自然 next step：用 PRM 在 attempt trajectory 上做 token-level reweighting，用 $\hat{p}$ 在 abstention trajectory 上做 trajectory-level reweighting。这可能比单独 TIAR 拿到更强的 accuracy preservation。

如果你想 push 这个方向，我建议三个实验：
1. **Warmup TIAR**：前 5 步用 ternary，第 6 步起切 TIAR，看 early-training calibration 是否更稳
2. **$G$ scaling**：$G=8 \to 16 \to 32$，看 $\hat{p}$ variance 降低后 F1 提升幅度
3. **Multi-turn extension**：把 $\hat{p}$ 推广到 dialogue state level，每轮都 update

这 paper 的 limitation 部分写得相当诚实，明确点出 multi-turn 和 RAG extension 是 next step——这跟你之前 micrograd / nanoGPT 的 "教学优先" 风格很 match，是个值得 hack on 的 weekend project 量级工作。
