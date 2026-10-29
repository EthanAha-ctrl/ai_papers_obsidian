---
source_pdf: Right Question is Already Half the Answer.pdf
paper_sha256: 9e7017c495b3ee357f848f743b54a6e1914081000d68e0ed1ea3dc1ceda186da
processed_at: '2026-08-11T23:50:40-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 EMPO

## 一句话版本

**让模型对同一个问题自己回答好多次，然后看大家意见一不一致，意见一致的答案就奖励它，意见不一致的就惩罚——全程不需要标准答案。**

---

## 1. 这篇 paper 在解决什么痛点

之前的 LLM reasoning training 流程是这样的：

```
pre-training → SFT（喂带 reasoning trace 的标注数据）→ RL（用 golden answer 做 reward）
```

这个流程需要三样东西：
- **问题 q**：user query
- **reasoning trace r**：人类专家写的解题过程
- **golden answer a**：标准答案

DeepSeek-R1-Zero 把 SFT 砍掉了，直接从 base model 做 RL，但**还是要 golden answer 来做 rule-based reward**（数学题可以对答案算分）。

问题来了：**golden answer 哪里来？** 数学题还好，有标准答案。但像 TruthfulQA 这种 free-form QA，根本没有唯一的 "正确答案"，你没法写一个 verifier 来判对错。code generation 也类似，test case 不一定有。

EMPO 的野心：**把 golden answer 这个最后一层 supervision 也拿掉，只喂一堆 unlabeled question 就能 train reasoning**。

---

## 2. 核心直觉：用 "一致性" 代替 "正确性"

### 一个生活中的比喻

想象一个考试场景：有一道难题，你不知道标准答案。怎么判断学生到底掌握没掌握？

**方法一**：看学生 A 的答案对不对（需要标准答案）。

**方法二**：让同一个学生把这道题做 7 遍，看 7 次的答案一不一致。
- 如果 7 次答案都一样 → 学生其实 "知道"，大概率对
- 如果 7 次答案全不一样 → 学生在猜，大概率错

这就是 self-consistency 的核心：**一致 = 大概率正确**。

EMPO 把这个 intuition 拿来当 training signal：
- 对同一个 question 采样 7 个 response
- 把 "意思相同" 的 response 聚成一类
- 多数派的那个 cluster 给高 reward，少数派给低 reward
- RL update 让 model 倾向输出多数派的答案

---

## 3. 怎么判断 "意思相同"

这是工程上最关键的一步：

### 数学题
简单：用 regex 提取 `\boxed{}` 里的 final answer，两个 answer 字符串一样就是同 cluster。比如 `\boxed{42}` 和 `\boxed{42}` 合并，`\boxed{42}` 和 `\boxed{43}` 不合并。

### Free-form QA
难：要用 NLI 模型（DeBERTa-v3-large，304M 参数）判断两个 output 是不是 bidirectional entailment。比如 "The capital of France is Paris" 和 "Paris is the capital of France" 应该合并。

这里有个工程上的取舍：
- 用大 LLM 当 judge 太贵
- DeBERTa 这种小 NLI 模型够用且快
- 只看 final answer，不看 reasoning trace（避免 reward hacking 在 trace 里塞废话）

---

## 4. 为什么这玩意儿能 work

这是最 counterintuitive 的地方。直觉上，"让模型变 confident" 不等于 "让模型变对"——模型可能自信地一直答错。

但 paper 的实验显示 accuracy 真的上升了。作者给出的解释，我翻译成人话：

**Pretraining 已经教会模型大部分东西，post-training 只是把 output distribution 拉到对的位置。**

打个比方：Qwen2.5-Math-7B Base 就像一个 "学过但没做过题" 的学生，脑子里其实有知识，但每次答题都会跑偏（采样 7 次可能 7 个不同答案）。EMPO 做的事相当于：

> "你自己多答几次，哪个答案出现次数最多，就多往那个方向靠。"

因为 model 脑子里其实 "知道" 正确答案（pretraining 学的），多次采样时正确答案大概率是多数派，于是 RL 会把 output distribution 推到正确答案上。这是一个 **"self-bootstrap from internal knowledge"** 的过程。

这跟之前 R1-Zero 出现的 "Aha moment"（response 变长、reflection emerge）形成对比——EMPO 训练时**没有这个现象**。作者的解读：

> R1-Zero 的 "长 CoT + reflection" 可能不是 reasoning 能力的 emergent，而是 rule-based reward 下 model 学会 "写长" 来撞对答案的 format alignment。EMPO 不奖励长度，所以 model 直接奔向 consistent answer，不绕弯。

这个观察其实挺 subversive 的：**我们之前以为 RL 让 model "学会思考"，可能 RL 只是让 model "学会用思考的格式说话"**。

---

## 5. 实验结果有多猛

| 任务 | Baseline | EMPO | 对比 supervised |
|---|---|---|---|
| 7B Math reasoning (avg) | 30.7% | **48.1%** | GRPO 46.8%, Instruct 49.4% |
| 7B TruthfulQA | 87.16% | **97.25%** | — |
| 7B TrivialQA | 54.94% | **70.22%** | — |

几个 striking 的事实：
1. **7B 数学题 EMPO 比 GRPO 还高**（48.1 vs 46.8），但 GRPO 用了 golden answer，EMPO 没用。
2. **EMPO 几乎追平 Instruct 模型**（48.1 vs 49.4），但 Instruct 经过了 multi-stage SFT + RL on private data。
3. **TruthfulQA 提升 10 个百分点**——这种 "为什么政府隐瞒 UFO" 的 trap 题，EMPO 让 model 学会质疑 false premise（因为 model 内部其实知道 UFO 是阴谋论，多次采样多数派会反对 premise）。

---

## 6. 为什么不会崩塌成 trivial solution

**问题**：model 完全可以让所有 output 都输出 `\boxed{?}` 或一个固定无意义字符串，这样 semantic entropy = 0，reward 最大化，但 model 啥也没学。

**Appendix B 就展示了这个**：free-form QA 上不加 KL penalty 时，model 真的学会了输出 `\boxed{?}`。

**解法**：
1. **KL penalty**：约束 model 别离 reference model 太远
2. **Entropy thresholding**：过滤掉 entropy 太低（已 collapse）或太高（太 confused）的 question，只在 "中间难度" 的 question 上 train

这俩是工程 patch，不是 fundamental solution。但 paper 里说在他们的 setting 下够用。

---

## 7. 哪些地方还很 sketch

### 7.1 Majority 可能是错的

如果一道题 model 大多数时候都答错，那 majority cluster 就是错的，EMPO 会 reinforce 错误答案。Paper 用 $\delta_{\text{high}}$ 过滤 high-entropy question 缓解，但这只是治标——如果 model 对一类题系统性 bias，entropy 不一定高。

### 7.2 Semantic clustering 是 bottleneck

DeBERTa 判 entailment 也有错的时候。对 subjective question（"最好的电影是哪部"）multiple correct answer 的情况，clustering 本身就 ill-defined。code generation 更难，可能需要 execution-based clustering。

### 7.3 只在 final answer 上做 clustering

EMPO 只看 final answer 一致性，reasoning trace 完全忽略。这意味着：
- 一个 "猜对答案但 reasoning 全错" 的 output 会得到高 reward
- 一个 "reasoning 完全对但 final answer 抄错" 的 output 会被 penalize

这跟 PRM（process reward model）的方向相反，长期看可能不如 step-level supervision 稳。

### 7.4 Distribution 依赖

EMPO 假设 unlabeled question 的 distribution 与 eval 接近。paper 里用 NuminaMath-CoT 训练，用 MATH/AIME eval，distribution 接近。真实场景 user query 是 mixture（写诗、debug、闲聊...），EMPO 在这种 mixture 上效果未知。

---

## 8. 这件事的 bigger picture

我觉得 EMPO 真正有意思的不是 benchmark 数字，而是它提出的几个 conceptual question：

### Q1: Post-training 到底在做什么？

如果 unsupervised EMPO 能逼近 supervised GRPO，那 supervised RL 提供的额外信号（golden answer）到底贡献了什么？作者的 hypothesis：**post-training 主要是 output distribution 的 shaping，不是 knowledge injection**。

这意味着：reasoning ability 的天花板主要由 pretraining 决定，post-training 只是 alignment。这跟 Anthropic ReFT paper 的观点一致。

### Q2: "Emergent reasoning behavior" 是真的 emergent 吗？

R1-Zero 的 "Aha moment" 一直被宣传为 RL 让 model "学会思考"。但 EMPO 在 unsupervised setting 下没观察到这个现象，强化了一个怀疑：**reflection behavior 可能 pretraining 时就在 base model 里了，rule-based reward 只是让它显式表达出来**。

这有点像 "language emergence in emergent communication" 领域的争议——agents 真的学会新 communication protocol，还是只是 align 到已有的 latent space？

### Q3: Unsupervised reasoning 的 scaling law 是什么？

EMPO 只需要 unlabeled question，互联网上有海量这种数据（Reddit、Stack Overflow、forum）。如果这个 method 能 scale 到 trillion-level unlabeled queries，可能比 R1-Zero 路线还 scalable。但 scaling law 没人测过。

### Q4: Consistency 为什么是 correctness 的好 proxy？

这背后可能有个 deep 的 information-theoretic 原因：一个 well-trained model 的内部 posterior distribution 在正确答案上 should be peaked，多次采样应该 collapse 到 mode。如果采样不 collapse，说明 model 的 posterior 是 multi-modal 或者 flat，即 model 不 confident，大概率 wrong。这是 Bayesian 视角下的 self-consistency，与 Infomax principle 也有关联。

---

## 9. 我觉得最 elegant 的地方

EMPO 最 elegant 的地方是它把三件事优雅地缝合在一起：

1. **Classical SSL 的 entropy minimization**（Grandvalet & Bengio 2004）——传统半监督学习的经典 trick
2. **Semantic entropy for LLM**（Oxford 的 Nature paper）——针对 free-form text 的 uncertainty metric
3. **GRPO**（DeepSeekMath）——LLM RL 的 SOTA algorithm

这三件事各自都成熟，但拼在一起产生了一个 "完全 unsupervised 的 reasoning incentivization"，这个 idea 本身就很 beautiful——**用 model 自己的 consistency 当 reward，完全 self-referential，没有任何 external supervision**。

有点像 GAN 的精神：generator 和 discriminator 互相博弈。EMPO 里 model 自己既是 policy 又是 reward provider（通过 cluster frequency）。

---

## 10. 一句话总结

**EMPO 证明：只要给一堆 unlabeled question，通过 minimize "多次采样答案的 semantic entropy"，LLM 就能 self-bootstrap 出 reasoning 能力，性能逼近甚至超越用了 golden answer 的 supervised RL——这暗示 LLM 的 reasoning ability 主要来自 pretraining，post-training 可能主要是 output distribution shaping 而非 knowledge injection。**

---

# EMPO: Fully Unsupervised LLM Reasoning Incentivization 深度解读

## 1. 核心Intuition: "问题本身就是答案的一半"

这篇 paper 来自 Tianjin University + Tencent AI Lab，核心 idea 极其 elegant：**完全 unsupervised 地 incentivize LLM reasoning，仅靠 unlabeled user queries 通过 minimize semantic entropy 来做 RL**。Paper title 借用了 "Right Question is Already Half the Answer" 这个谚语，呼应 R1-Zero 路线的同时，把 supervision 要求压到极限——连 rule-based reward / golden answer / reward model 都不要。

为什么这件事有意思？DeepSeek-R1-Zero 已经证明可以从 base model 直接做 RL 跳过 SFT，但它仍然需要 rule-based verifier（数学题能验算答案对错）。EMPO 把这最后一层 supervision 也拿掉，claim 只要有 unlabeled questions 就够了。这相当于把 LLM reasoning 的 self-improvement 推到了 pure self-supervised learning 的边界，让人联想到 self-play AlphaGo 或者 wake-sleep algorithm 的精神。

直觉上的 key insight：**一个 well-trained LLM 对同一个 question 多次采样，如果它真的 "知道" 答案，那么采样的多个 response 在 semantic level 应该 converge 到同一个 meaning cluster**。Semantic entropy 低 → 模型 confident 且 consistent → 大概率 correct。这个观察在 Oxford 的 Farquhar et al. (Nature 2024) 已经被验证：semantic entropy 与 accuracy 有 strong negative correlation，能 detect hallucination。EMPO 直接把这个 diagnostic metric 拿来当 training reward。

参考链接：
- Paper: https://arxiv.org/abs/2505.18114 (EMPO)
- Semantic entropy Nature paper: https://www.nature.com/articles/s41586-024-07421-0
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Tent (entropy minimization TTA): https://arxiv.org/abs/2006.10726
- Grandvalet & Bengio 2004 (entropy minimization SSL): https://proceedings.neurips.cc/paper/2004

---

## 2. Background: 三条线索的汇合

EMPO 是三条 research line 的交汇点：

### 2.1 RL for LLM Reasoning: GRPO 与 R1-Zero

GRPO (Group Relative Policy Optimization) 来自 DeepSeekMath，是 PPO 的简化版，省掉 critic，用 group-based baseline。其 objective（论文 Eq. 1）：

$$\mathcal{J}_{\text{GRPO}} = \mathbb{E}_{q\sim P(Q), \{o_i\}_{i=1}^G \sim \pi_\theta(O|q)} \left[ \frac{1}{G}\sum_{i=1}^G \Big( \min(A_i, \text{clip}(1, 1-\epsilon, 1+\epsilon)A_i) - \beta \text{KL}(\pi_\theta || \pi_{\text{ref}}) \Big) \right]$$

变量含义：
- $q$: 从 prompt distribution $P(Q)$ 采样的 question
- $\{o_1, \dots, o_G\}$: 对同一个 $q$ 从 policy $\pi_\theta$ 采样 $G$ 个 outputs（group）
- $A_i$: 第 $i$ 个 output 的 advantage，用 group-internal normalization: $A_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(r_1,\dots,r_G)}$
- $\beta$: KL penalty 系数，让 $\pi_\theta$ 别离 reference $\pi_{\text{ref}}$ 太远
- $\epsilon$: PPO-style clip，限制 ratio 区间 $[1-\epsilon, 1+\epsilon]$，stability
- $r_i$: reward，在 math reasoning 里通常是 rule-based verifier 给的 0/1

GRPO 的核心 trick 是用 group mean/std 做 baseline，避免训练 critic network，省一半显存。R1-Zero 直接在 base model 上跑 GRPO + rule-based reward，self-emerge 出 reflection、self-correction 等 behavior。

### 2.2 Semantic Entropy: 把 Shannon Entropy 搬到 LLM

Kuhn, Gal, Farquhar (Oxford) 提出的 semantic entropy (arXiv 2302.09664) 解决了一个根本问题：**LLM 输出是 token sequence，token-level entropy 不能反映 semantic uncertainty**。"The answer is 42" 和 "Forty-two is the answer" 在 token 层面完全不同，在 semantic 层面是同一个意思。

他们的做法：
1. 对 question $q$ 采样 $G$ 个 generations $\{o_1, \dots, o_G\}$
2. 用 bidirectional entailment（DeBERTa 类小模型做 NLI）判定两两是否 same meaning
3. 把同义的 outputs 合并成 meaning cluster $c_j$
4. 估计 cluster probability: $p(c_j|q) \approx |c_j|/G$
5. Semantic entropy: $H = -\sum_{c_j \in \{c\}} p(c_j|q) \log p(c_j|q)$

这个 $H$ 衡量 "model 在 meaning 空间的不确定度"。Farquhar et al. 在 Nature 2024 证明它能 detect hallucination / confabulation，因为它捕捉的是 model 内部的不一致，而 token-level entropy 经常被 surface form 噪声掩盖。

### 2.3 Entropy Minimization in Classical SSL

Grandvalet & Bengio (NeurIPS 2004) 在 semi-supervised learning 提出：在 unlabeled data 上 minimize 预测 distribution 的 Shannon entropy，能让 model 更 confident，相当于 implicit cluster assumption——decision boundary 应该落在 low-density region。

Tent (Wang et al. 2020) 把这个 idea 用到 test-time adaptation：测试时只 update normalization layer 的参数，minimize prediction entropy，就能 fill domain gap。COME (Zhang et al. 2024, 即本文作者之一) 进一步加 conservative 约束避免 degenerate solution。

EMPO 把这个 line 搬到 LLM RL，用 semantic entropy 替代 Shannon entropy（因为 LLM 输出是 free-form text，不能用 categorical softmax entropy）。

---

## 3. Method 详解: EMPO Algorithm

### 3.1 Problem Setup

输入：unlabeled questions $\{q_i\}_{i=1}^n$（来自 user query log，没 golden answer），pre-trained LLM $\pi_\theta$。

目标：增强 $\pi_\theta$ 的 reasoning ability，不用任何 supervision（no golden answer, no reasoning trace, no reward model）。

### 3.2 Semantic Entropy Minimization Objective

对每个 $q$：
1. 采样 $G$ 个 outputs $\{o_1, \dots, o_G\} \sim \pi_\theta(O|q)$
2. 按 semantic equivalence 聚类成 $M$ 个 clusters $\{c_1, \dots, c_M\}$
   - 数学题：用 regular expression + Math-Verify package 提取 `\boxed{}` 里的 final answer，按答案是否相同聚类
   - Free-form QA：用 DeBERTa-v3-large（304M 参数）判定 bidirectional entailment
3. 估计 cluster probability：$p(c_j|q) \approx |c_j|/G$（论文 Eq. 3）
4. Semantic entropy：$H = -\sum_{c_j} p(c_j|q) \log p(c_j|q)$（论文 Eq. 4）

### 3.3 EMPO Objective

EMPO 的核心是把 semantic entropy minimize 这个 objective 包装成 GRPO 形式。对每个 output $o_i$，它所属的 cluster $l(o_i) = c_j$，reward 定义为这个 cluster 的 likelihood：

$$r_i = p(c_j|q), \text{ where } l(o_i) = c_j \quad \text{(论文 Eq. 6)}$$

直觉：**多数派 cluster 的成员获得高 reward，少数派 cluster 获得低 reward**。这其实是在做 "soft majority voting"——GRPO 的 advantage normalization 让 minority cluster 获得负 advantage 被 penalize，majority cluster 获得 positive advantage 被 reinforce。

最终 objective（论文 Eq. 7）：

$$\mathcal{J}_{\text{EMPO}} = \mathbb{E}_{q\sim P(Q), \{o_i\}_{i=1}^G \sim \pi_\theta(O|q)} \left[ \frac{1}{|G|} \sum_{i=1}^{|G|} \Big( \min(A_i, \text{clip}(1, 1-\epsilon, 1+\epsilon)A_i) - \beta \text{KL}(\pi_\theta || \pi_{\text{ref}}) \Big) \right]$$

$$\text{s.t. } \delta_{\text{low}} < H < \delta_{\text{high}}$$

变量补充说明：
- $A_i$ 与 GRPO 一样：$A_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(r_1,\dots,r_G)}$
- $r_i$ 不再是 rule-based，而是 cluster frequency $p(c_j|q)$
- $\delta_{\text{low}}, \delta_{\text{high}}$：entropy 上下阈值，过滤掉两个极端的 question
  - $H > \delta_{\text{high}}$：model 对这个 question 太 confused，所有 answer 都不一致，reward 信号是噪声
  - $H < \delta_{\text{low}}$：model 太 confident，所有 samples 已经在同一个 cluster，没有 learning signal，且容易 reward hack

### 3.4 Architecture Pipeline 解析

从 Figure 1 看 EMPO 的 pipeline：

```
[Unlabeled Questions q] 
        ↓
[Sample G responses from π_θ]  ← 例如 G=7
        ↓
[Extract final answers / meaning]
        ↓
[Semantic clustering]   ← math: regex; QA: DeBERTa-v3-large
        ↓
[Compute cluster probabilities p(c_j|q) = |c_j|/G]
        ↓
[Assign reward r_i = p(c_j|q) for each o_i]
        ↓
[Filter by δ_low < H < δ_high]
        ↓
[GRPO-style advantage normalization + PPO clip + KL penalty]
        ↓
[Update π_θ via backprop through log-prob ratio]
```

关键 design choice：
- **Clustering 只看 final answer，不看 reasoning trace**——这避免了 reward hacking 在 trace 里堆积无意义 token。论文 supplementary case 里显示，original Instruct model 会写很长但错误的 reasoning，EMPO tuned model 倾向简洁正确的 reasoning。
- **DeBERTa NLI 304M 比 LLM judge 小很多**，clustering 几乎不增加计算开销。
- **G=7**（同 GRPO），group size 不需要太大。

### 3.5 为什么这个 objective 能 incentivize reasoning？

这是最 counterintuitive 的部分。直觉上，minimize entropy 只会让 model 输出 mode-collapsed 的常见答案，似乎不会真的提升 reasoning。但实验显示 accuracy 实际上升了。作者的解释 (Section 5) 引用 Wu et al. ReFT 的话：

> "Pretraining does all the hard work. One big bet is that the pretraining phase grants all the abilities to the base LM, and finetuning is simply like a style transfer which positions the model to the right output space."

这暗示 Qwen2.5-Math-7B Base 其实已经 "knows" 怎么做数学，只是 output distribution 没对齐到 "采样多次会 consistent" 的状态。EMPO 做的事相当于：**把 model 的 output distribution 从多个互不相同的 reasoning path "推" 到那些彼此一致的 reasoning path 上**，而 consistent path 因为是 majority，所以更可能是 correct path（self-consistency 的假设）。

这与 STaR (Zelikman et al.) 的 idea 有点像：bootstrap from self-generated correct trajectories。区别是 STaR 用 ground truth filter，EMPO 用 consistency filter。

另一种解读：**EMPO 是在做 implicit EM (Expectation-Maximization)**。E-step 是 sample + cluster（推断 latent "true answer" 是哪个 cluster），M-step 是 update policy 让 output 落到最大 cluster。当 majority cluster 是 correct 的概率 > 50% 时，这个 EM 会收敛到 correct solution。

参考：
- STaR: https://arxiv.org/abs/2203.14465
- Self-consistency: https://arxiv.org/abs/2203.11171
- ReFT (representation finetuning): https://arxiv.org/abs/2404.03592

---

## 4. 实验结果深度分析

### 4.1 数学推理 (Table 1)

| Model | Supervision | MATH | Minerva | Olympiad | AIME24 | AMC23 | Avg |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B | None | 64.8 | 15.1 | 26.7 | 6.7 | 40.0 | 30.7 |
| Qwen2.5-Math-Instruct | {q,r,a} | 82.8 | 43.8 | 41.2 | 16.7 | 62.5 | 49.4 |
| Qwen2.5-Math w/SFT | {q,r,a} | 72.2 | 34.6 | 33.2 | 10.0 | 45.0 | 39.0 |
| Qwen2.5-Math w/ODPO | {q,a} | 76.8 | 30.9 | 37.9 | 26.7 | 62.5 | 47.0 |
| Qwen2.5-Math w/GRPO | {q,a} | 77.8 | 39.7 | 39.1 | 20.0 | 57.5 | 46.8 |
| **Qwen2.5-Math w/EMPO** | **{q}** | **78.0** | **40.4** | **37.3** | **20.0** | **65.0** | **48.1** |

关键观察：
1. **EMPO 在 7B 上 average 48.1%，竟然略高于 GRPO 的 46.8% 和 ODPO 的 47.0%**——只用 questions，没用 answers。这非常 striking。
2. **EMPO 达到了 Qwen2.5-Math-Instruct (48.1% vs 49.4%) 的 97%**——Instruct 模型用了 multi-stage SFT + RL on private data。
3. 1.5B 上 EMPO 42.1% 也接近 Instruct 40.5%，甚至超过了 Instruct。
4. AMC23 上 EMPO (65.0) 反而高于 Instruct (62.5)。

潜在解释：GRPO/ODPO 的 rule-based reward 是稀疏的（AIME 上 6.7% 基线，意味着 7 个 sample 大概率全错，reward 全 0，没 learning signal），而 EMPO 的 reward 永远是 dense 的（总有 cluster frequency），所以 training signal 更稳。这与 paper 里 Figure 2 显示的 entropy 与 accuracy 强负相关一致。

### 4.2 Free-form QA (Table 2)

| Model | Supervision | TruthfulQA True | Info | True×Info | MC1 | MC2 | TrivialQA EM |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | — | 87.16 | 99.69 | 86.89 | 39.45 | 56.14 | 54.94 |
| + CoT | — | 94.19 | 98.17 | 92.47 | 37.31 | 57.98 | 67.42 |
| **+ EMPO** | {q} | **97.25** | **98.48** | **95.77** | **41.59** | **60.22** | **70.22** |

TruthfulQA 7B 上从 87.16% 拉到 97.25%——10 个百分点的提升。这个 benchmark 的特点是有大量 human misconception trap（"为什么政府隐瞒 UFO"这类），original Instruct 模型会顺着 question 的 false premise 编 reasoning，EMPO tuned 模型会先质疑 premise（见 Appendix C case）。

这点挺有意思：**EMPO 实际上在训练 model 拒绝 false premise**，因为对这类 question，multiple samples 里既有顺着 premise 编的、也有质疑的，但 model "真正知道" 的部分会一致地反对 false premise。Majority cluster 在这里起到了 "model 内在知识投票" 的作用。

### 4.3 Training Dynamics (Figure 2)

Figure 2 显示 7B model 训练 20K NuminaMath-CoT prompts 的过程：
- **Semantic entropy 持续下降**——表明 model outputs 越来越 consistent
- **Unsupervised reward (cluster frequency) 上升**——majority cluster 越来越大
- **Model accuracy 同步上升**——unsupervised proxy 确实对应 real accuracy

这验证了核心假设：**semantic entropy 是 accuracy 的有效 proxy**。

---

## 5. 关键 Discussion: 为什么没有 "Aha Moment"？

DeepSeek-R1-Zero 训练时观察到 response length 暴增和 emergent reflection behavior ("Aha moment")，但 EMPO 训练时**没有这个现象**。作者认为这暗示两件事：

1. **Pretraining 已经 grant 了 reasoning ability**，post-training 只是 style transfer 把 output distribution 拉到合适位置。R1-Zero 的 "Aha moment" 可能是 base model 在 rule-based reward 下慢慢学会 "长 chain of thought" 的 format，而 EMPO 直接 minimize entropy 让它直奔 consistent answer，不需要拉长。

2. R1-Zero 的 length 增长可能是 reward hacking 的一种——longer CoT 更可能撞到正确答案（verifier 只看 final answer），所以 model 学会 "写长"。EMPO 因为 reward 不看长度，所以没有这个 incentive。

这个观察对 reasoning model 的未来 design 很关键：**"长 chain of thought" 可能不是 reasoning 的本质，而是 rule-based reward 下的 emergent exploitation**。Anthropic 最近的一些工作（如 extended thinking 的 ablation）也暗示类似结论。

参考：
- Anthropic Claude extended thinking: https://www.anthropic.com/news/claude-3-7-sonnet
- Online-DPO-R1: https://online-dpo-r1.github.io/

---

## 6. Reward Hacking 与 Entropy Thresholding

Appendix B 展示了一个 reward hacking case：在 free-form QA 上不加 KL penalty 时，model 学会输出 `\boxed{?}`——一个 trivial 的 degenerate solution，所有 samples 都 collapse 到同一个 "未知" cluster，semantic entropy = 0，reward 最大化。

Entropy thresholding $\delta_{\text{low}} < H < \delta_{\text{high}}$ 的作用：
- **下界 $\delta_{\text{low}}$**：过滤掉已经 mode-collapsed 的 question，防止 model 在它们上面进一步 hack
- **上界 $\delta_{\text{high}}$**：过滤掉太 confused 的 question，防止噪声 reward 信号污染训练

这是一个 elegant 的 engineering trick，但它暗示了一个 fundamental limitation：**纯 entropy minimization 在 unconstrained 情况下有 trivial solution**。未来工作可能需要更强的 anti-degeneration 机制，比如：
- Diversity bonus（类似 ETPO 的 entropy regularization 但在 token level）
- Contrastive objective（让 model 同时 maximize inter-cluster distance）
- Implicit reward modeling from preference data
- Process-level entropy（在 reasoning steps 上 minimize entropy，而不是只在 final answer 上）

参考 ETPO: https://arxiv.org/abs/2402.04304

---

## 7. 与相关工作的更深联系

### 7.1 Minimum Bayes Risk (MBR) Decoding

SeaLong (Li et al. 2024) 在 long-context reasoning 上用 MBR decoding：sample 多个 outputs，选与其他 outputs average similarity 最高的。这与 EMPO 在 inference-time 的精神一致：consistency ≈ correctness。区别是 MBR 只在 inference 做 selection，EMPO 把这个 objective 放回 training 通过 RL propagate 回 weights。

参考: https://arxiv.org/abs/2411.08147

### 7.2 Self-Play 与 AlphaGo

EMPO 让我强烈联想到 AlphaGo/AlphaZero 的 self-play：没有 human supervision，只靠 self-consistency 和 environment reward。EMPO 的 "environment" 是 LLM 自己的 meaning space——model 既是对手也是 judge。这与 Self-Rewarding Language Models (Yuan et al.) 的 LLM-as-Judge 思路不同，EMPO 不需要 explicit judge prompt，cluster frequency 隐式 vote。

参考: https://arxiv.org/abs/2401.10020

### 7.3 Information Theory 与 Mutual Information

Semantic entropy minimization 可以看作最大化 model output 与某个 latent "true answer" variable 之间的 mutual information。形式上，如果 $Z$ 是 latent true answer，$O$ 是 output，最大化 $I(Z; O)$ 在 unsupervised setting 下近似为 minimize $H(O|q)$ 给定 $q$（假设 $Z$ 由 $q$ 确定）。这与 Infomax principle (Bell & Sejnowski 1995) 的精神一致，也与 SIMCLR 等对比学习的 mutual information lower bound 有联系。

### 7.4 Wake-Sleep Algorithm

EMPO 的 sample-then-update 流程与 Helmholtz machine 的 wake-sleep algorithm 结构很像：
- "Wake" phase: 用当前 policy 采样，相当于 inference
- "Sleep" phase: 用采样结果 update policy，相当于 learning
区别是 wake-sleep 显式建模 generative + recognition 两个 network，EMPO 只有一个 LLM。

### 7.5 ReFT (Representation Finetuning)

Wu et al. ReFT 的核心 insight（被作者引用）：pretraining 已经给 base model 所有 ability，finetuning 只是 style transfer。这与 EMPO 的观察（不需要 golden answer 也能逼近 Instruct 性能）互为印证。ReFT 通过 intervention on hidden representations 来做 finetuning，比 SFT 参数效率高几个数量级。EMPO 可以看作 RL 版的 ReFT——不动 weights 的 specific layers，而是通过 RL signal 调整 output distribution。

### 7.6 DeepSeek-R1-Zero 与 emergent behavior 的本质

R1-Zero 的 emergent reflection 一直被部分研究者怀疑是 "rule-based reward 下 base model 已有 ability 的 format alignment"，而非真正新 emerge 的能力。EMPO 没观察到 "Aha moment" 强化了这种怀疑：**reflection 这种 behavior 可能 pretraining 时就在 base model 里了，只是 RL+verifier 让它在 surface form 上显现出来**。

这对未来 reasoning model 的研究方向有启示：
- 不应该迷信 "RL 产生 reasoning emergent behavior" 的叙事
- 应该研究 base model 到底 "knows" 什么，post-training 只是 surface alignment
- 真正的 reasoning improvement 可能来自 pretraining data 和 architecture，不是 RL algorithm

---

## 8. Limitations 与 Future Work

### 8.1 Semantic Equivalence 判定的 limit

EMPO 在 free-form QA 上依赖 DeBERTa 判定 bidirectional entailment，但这有几个问题：
- **DeBERTa 本身有 bias**，它判定 "是否 same meaning" 也会错
- **Subjective question 没有 ground truth meaning**——"最好的电影是什么" 多个 answer 都合理
- **Code generation** 几乎无法用 NLI 判 equivalence，需要 execution 或 formal verification（作者在 Section 6 提到这是 future work）

未来可能的 generalization：
- 用 LLM-as-Judge 替代 DeBERTa（但成本高）
- 用 contrastive embedding 做 cluster（如 E5-Mistral 等 SOTA embedding model）
- 用 execution-based verification for code（passing test cases as cluster key）
- 用 LLM 自己 generate cluster identifier（"这些 outputs 的共同答案是 X"）

### 8.2 Distribution shift 问题

EMPO 假设 unlabeled questions 来自与 evaluation 同 distribution。如果 user query log 与 eval benchmark distribution 不一致，semantic entropy minimization 可能 push model 到错误方向。Paper 里 train on NuminaMath-CoT 20K, test on MATH/AIME 等，这些 dataset distribution 比较接近。真实部署中 user question 可能极为 diverse（"帮我写诗"、"解释量子力学"、"debug 这段代码"），EMPO 在这种 mixture 上效果未知。

### 8.3 Minority cluster 是 correct 的情况

Self-consistency 的根本假设是 majority = correct。但有些难题 model 大多数时候都答错，correct answer 反而在 minority cluster。EMPO 会 reinforce majority（错误）cluster，让 model 学错。这与 STaR 的 rejection sampling 有同样的 cold-start 问题。Paper 里用 $\delta_{\text{high}}$ 过滤 high-entropy question 缓解，但这是治标不治本。

可能的解法：
- Curriculum learning：从 model 已经 mostly correct 的 easy question 开始
- Iterative refinement：用 EMPO 后用 self-rewarding (Xiong et al. R3) 做 minority cluster mining
- Multi-step verification：在 reasoning trace 内部做 consistency check，而不是只在 final answer

### 8.4 与 process reward 的关系

EMPO 只在 final answer level 做 semantic clustering，reasoning trace 被忽略。这与 PRM (Process Reward Model, OpenAI's "Let's verify step by step") 的方向相反。理论上可以扩展 EMPO 到 step level：每个 reasoning step 采样多次，minimize step-level semantic entropy。但 step-level clustering 更难，因为 step 之间 dependency 强。

参考: https://arxiv.org/abs/2305.20050

---

## 9. 我的 Intuition Building

读完这篇 paper 我有几个 takeaway：

**Intuition 1: Consistency is a surprisingly strong unsupervised signal.**  
LLM 内部其实有大量 "knows" 但 surface 上 noisy 的 knowledge。多次 sampling + clustering 相当于 implicit ensemble，能 denoise 出 model 的 internal belief。这与 Bayesian model averaging 的精神一致——单次 sample 是 noisy estimate，多次 sample 平均接近 posterior mean。

**Intuition 2: Post-training 可能远没有 pretraining 重要.**  
EMPO 在 7B Base 上几乎复现 Instruct 性能，强烈暗示 Qwen2.5-Math 7B Base 已经 "会做数学"，只是 output distribution 没对齐。如果这个观察 generalizes，那么 reasoning model 的 scaling law 可能主要在 pretraining compute 上，post-training 只是 surface alignment。这呼应了 Anthropic、Meta 最近一些工作对 "SFT 只是 style transfer" 的观点。

**Intuition 3: RL 的本质可能是 distribution shaping, 不是 knowledge injection.**  
EMPO 用纯 unsupervised signal 做的 RL 能逼近 supervised RL，说明 RL 的主要作用不是注入新 knowledge，而是 shape output distribution 到 consistent + confident 状态。这个 view 对 RL 在 LLM 中的角色是个 interesting reframing。

**Intuition 4: Semantic entropy 是 LLM uncertainty 的 best proxy.**  
Token-level entropy 经常 misleading，因为 surface form variation 与 semantic content 不相关。Semantic entropy 通过 meaning clustering 逼近 "true uncertainty"。这个 metric 不仅可以做 training objective，也可以做 inference-time confidence estimation、active learning sample selection、hallucination detection等。Oxford 的工作和 EMPO 一起把这个 metric 的应用范围扩展了。

**Intuition 5: Reward hacking 是 unsupervised RL 的 fundamental challenge.**  
Appendix B 的 `\boxed{?}` case 提示我们：任何 unsupervised objective 都有 trivial solution。Entropy minimization 经典文献里就有这个问题（Press et al. 2024 "The Entropy Enigma" 系统分析过）。EMPO 用 entropy thresholding 是工程 patch，不是 fundamental solution。未来真正 robust 的 unsupervised reasoning 可能需要 multi-objective 组合（consistency + diversity + coverage）或者 explicit anti-collapse mechanism。

参考 Press et al. entropy enigma: https://arxiv.org/abs/2405.05012

**Intuition 6: Reasoning model 的 "Aha moment" 可能是 emergent format, 不是 emergent ability.**  
R1-Zero 的 reflection behavior 在 EMPO 上没出现，强化了 "base model 已经 knows，RL 只是 surface alignment" 的 view。这对如何理解 reasoning emergence 有重要影响——也许我们一直在 measure surface behavior 而非 underlying ability。如果 true reasoning ability 来自 pretraining，那 reasoning data 的 scaling law 应该在 pretraining 阶段，不是 RL 阶段。

**Intuition 7: 这条路线的 scaling potential 巨大.**  
EMPO 只需要 unlabeled questions，而互联网上有海量 user query log（Reddit、Stack Overflow、forum 等）。如果这个 method scale up 到 trillions of unlabeled queries，可能 produce 比 R1-Zero 还强的 reasoning model，且 cost 极低。这是一个真正 scalable 的路线。

**Intuition 8: 与 Q-learning / offline RL 的潜在结合.**  
EMPO 是 on-policy RL，每步都从 current policy 采样。可以想象 offline 版本：从 fixed dataset of (question, multiple responses) 用 semantic entropy 做 reward 做 offline RL。这能极大降低训练 cost，因为不再需要每步 sample。

---

## 10. 总结

EMPO 是一个 conceptually beautiful 且 empirically strong 的工作。它把 semantic entropy（Oxford 的工作）、entropy minimization（classical SSL）、GRPO（DeepSeekMath）三条线整合，做出一个真正 fully unsupervised 的 reasoning incentivization 方法。实验上：
- 7B math reasoning: 30.7% → 48.1%（逼近 Instruct 49.4%）
- 7B TruthfulQA: 87.16% → 97.25%
- 7B TrivialQA: 54.94% → 70.22%

更重要的是它引发的 conceptual questions：
- Post-training 是 knowledge injection 还是 distribution shaping？
- "Aha moment" 是 emergent ability 还是 emergent format？
- Supervised RL 真的比 unsupervised RL 强吗，还是只是 reward signal 更 dense？
- Pretraining 到底给 model 注入了多少 reasoning ability？

这些 questions 比单纯的 benchmark 提升更值得思考。EMPO 的 limitation（semantic equivalence 判定难、minority cluster 是 correct 的情况、reward hacking）也清晰指出了 future work 方向。

主要参考链接汇总：
- EMPO paper: https://arxiv.org/abs/2505.18114
- Semantic entropy (Kuhn et al.): https://arxiv.org/abs/2302.09664
- Semantic entropy Nature (Farquhar et al.): https://www.nature.com/articles/s41586-024-07421-0
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DeepSeekMath/GRPO: https://arxiv.org/abs/2402.03300
- Tent: https://arxiv.org/abs/2006.10726
- COME: https://arxiv.org/abs/2410.10894
- Online-DPO-R1: https://online-dpo-r1.github.io/
- Self-Rewarding LM: https://arxiv.org/abs/2401.10020
- R3 (reverse curriculum RL): https://arxiv.org/abs/2402.05808
- PRM (Let's verify step by step): https://arxiv.org/abs/2305.20050
- Self-consistency: https://arxiv.org/abs/2203.11171
- STaR: https://arxiv.org/abs/2203.14465
- ReFT: https://arxiv.org/abs/2404.03592
- Entropy enigma: https://arxiv.org/abs/2405.05012
- SimpleRL: https://hkust-nlp.notion.site/simplerl-reason
- NuminaMath: https://huggingface.co/AI-MO/NuminaMath-CoT
- TrivialQA: https://arxiv.org/abs/1705.03551
- TruthfulQA: https://arxiv.org/abs/2109.07958
- Tulu 3: https://arxiv.org/abs/2411.15124
- Open-R1: https://github.com/huggingface/open-r1
- TRL: https://github.com/huggingface/trl

Paper 让我最兴奋的是它把 "RL 在 LLM 中到底在做什么" 这个 question 摆到了台面上。如果 unsupervised EMPO 能逼近 supervised GRPO/ODPO，那么我们一直在 benchmark 上 measure 的 reasoning improvement 里有多少是真正的 capability gain，有多少只是 output distribution 的 alignment？这个问题对 reasoning model 的未来 design 至关重要，可能比单纯刷 SOTA 更值得研究 community 投入精力。
