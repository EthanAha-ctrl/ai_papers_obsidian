---
source_pdf: STaR Bootstrapping Reasoning With Reasoning.pdf
paper_sha256: d22a472fce2631755849988a78d78ad6d80999166eb4bac4f5315c97dfdf6d9a
processed_at: '2026-08-12T10:53:18-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 STaR

## 一句话讲完

STaR 干的事：**让 LLM 像学生一样，自己做题，把做对的解题过程收集起来当教材，下一轮用更好的自己再做一遍题，循环往复**。做错的题怎么办？把答案告诉它，让它倒着推一个能解释答案的解题过程，也收进教材。就这么简单。

---

## 用学生做类比

想象一个学生准备数学考试，手头只有：
- 一本厚厚的习题集（带答案，没解题过程）
- 老师给的 10 道带完整解题过程的范例

STaR 这个学生怎么学习？

**第一轮**：看着 10 道范例，硬着头皮把习题集全做一遍。做完对答案，把做对的题（连同自己写的解题过程）撕下来贴进笔记本。做错的题扔掉。

**然后**：把笔记本当新的教材，从头开始学（注意，是把人脑重置回初始状态，再学一遍新笔记本，避免过度记忆第一轮的内容）。

**第二轮**：学完新笔记本的学生又变强了一点，再做一遍原习题集。这次能做对更多题，收集更多解题过程。

**循环**：每轮都变强一点，能做对的题更多，笔记本越来越厚越来越准。

**问题**：到某一轮，这学生已经做不出任何新题了——做不出来的题永远进不了笔记本，卡住了。

**Rationalization 解决方案**：对于做不出的题，老师直接告诉答案。学生问自己："既然答案是 90，那怎样的推理过程能推出 90？"倒着推一个 reasoning，假装是自己想的，贴进笔记本。这样难题也能进教材。

这就是 STaR 的全部。

---

## 为什么这个 naive 想法居然 work？

这是 paper 最有意思的地方。直觉上，"自己做题自己教自己"听起来像 self-reference disaster，但作者证明了它其实是 **REINFORCE policy gradient 的简化版**。

**关键 reframe**：把 LLM 看成在 reasoning 上有隐变量的模型。

$$p(y|x) = \sum_r p(r|x) \cdot p(y|x, r)$$

翻译成人话：模型答对一道题的概率，等于"采样各种 reasoning 路径 × 每条路径推出正确答案的概率"之和。

那么怎么优化这个 objective？

**REINFORCE 的标准做法**：sampling reasoning 路径，答对的给 reward 1，答错的给 0，gradient ascent。

**STaR 做了两个 hack**：
1. **不 random sample，用 greedy decode**：variance 小很多，代价是 exploration 差
2. **答错的 sample 直接扔掉**——因为 indicator reward 是 0，gradient 本来就是 0，扔掉等价

所以 STaR **= REINFORCE with greedy decode + zero-variance filtering**。它不是某种新算法，就是一个被极度简化到能用 supervised learning pipeline 实现的 policy gradient。

这就是为什么它能用标准 GPT-J finetune script 跑起来——根本不需要 RL infra。

---

## Rationalization 到底干了什么

这是 paper 里最 subtle 的部分。表面看：给答案，让模型反推 reasoning。

但数学上它做的事更深刻：**改变了 sampling distribution**。

- 正常 rationale generation：从 $p(r|x)$ 采样——"给这个问题，一般怎么想？"
- Rationalization：从 $p(r|x,y)$ 采样——"给这个问题和答案，怎么解释？"

由 Bayes：

$$p(r|x,y) \propto p(y|x,r) \cdot p(r|x)$$

意思是：rationalization 偏向于采样那些**能推到正确答案**的 reasoning，而正常 sampling 是 unbiased 的。

这相当于 importance sampling from 一个更好的 proposal distribution。作者在 Section 5 提到这可以看成 **off-policy estimate**。

Intuition：难题上，"知道答案再推 reasoning"比"不知道答案硬想"容易太多了——人类也是这样，看答案倒推比硬做容易。

---

## 最 striking 的实验结果

### Arithmetic（加法）

baseline（不用 reasoning，直接训练 predict 答案，10k 样本 5000 steps）：76.3%

STaR 16 轮：**89.5%**

更厉害的是 **Figure 5**：在 20 轮后突然给 model 见 6-8 位数训练，然后测它从没见过的 9-10 位数加法——居然也能做对很多题。这说明 STaR 不只是 memorize，它学到了 **systematic algorithm**（进位、对齐等结构），这种东西可以 out-of-distribution generalize。

### CommonsenseQA

6B GPT-J + STaR：72.5%
175B GPT-3 fine-tuned：73.0%

**6B 模型追平 30 倍大的 model**。这说明 explicit reasoning 的 signal 比 raw answer prediction 信号信息密度高很多——同样参数 budget 下，能让 model 学到更可迁移的 representation。

### Rationalization 的 dramatic effect（Figure 4）

without rationalization：学习是 stagewise 的——3 位数加法做不好之前，4 位数完全不动。要一档一档爬。

with rationalization：1 轮训练后 2-digit 从 <1% 直接跳到 32%。能**多档同时学**。

为什么？因为 rationalization 把难题也变成 training data 了，model 不需要先解决简单题才能拿到难题的 signal。

---

## 作者老实承认的几个坑

### 1. 初始 model 必须"足够好"

GPT-2 在 arithmetic 上根本 bootstrap 不起来——因为 few-shot performance 接近 0%，第一轮生成的 rationale 全是垃圾，没有 positive example 可以学。STaR 需要 **seed capability**。

### 2. High chance accuracy 的任务会崩溃

二分类任务，random guess 50% 正确。这 50% 的"正确答案"背后 rationale 可能完全胡说八道——训练就把 model 带偏了。

### 3. Bias amplification

CQA 本身有 gender bias 等。STaR 会**放大**这些 bias，因为"对 bias 友好的 rationale"更容易答对题、更容易进 training set。Rationalization 更糟——把 model 平时不输出的 biased answer 也"pull out"了。

### 4. Faithfulness 问题

model 写的 rationale **未必反映它真实的内部推理过程**。完全可能是：model 隐式先选了答案，再 post-hoc 编一个能解释这个答案的 rationale。这是 explainable AI 经典问题，STaR 也没解决。

### 5. Appendix A 里的 reasoning fallacies

作者诚实列了一堆 model 的失败模式，特别 enlightening：

- **Question Implies Answer**："answer must be X. Y is X. So Y."——没解释 why Y 满足 X
- **Begging the Question**：reasoning 里偷偷 imply 答案
- **World State Assertions**：说"James feels exhilaration"而不是"people feel exhilaration"——把变量当 specific person
- **Hint Short-cutting**：rationalization 时 model 学到 hint 总是答案，于是乱写一通然后 output hint

这些 failure modes 直接启发了后续 OpenAI 的 PRM800K——**outcome-only reward 无法监督 process**，需要给每一步打分。

---

## 与其他工作的联想

### 跟 AlphaGo / Expert Iteration 的关系

STaR 就是 Language Model 版的 Expert Iteration [Anthony 2017]。ExIt 的 loop：apprentice self-play → expert(MCTS)给 feedback → imitation learn → 替换 expert。STaR 把 "expert" 换成了 ground-truth answer，把 MCTS 换成 LLM 自己 greedy decode。

### 跟 AlphaCode 的关系

AlphaCode：generate 一堆 candidate solutions → 用 tests 过滤 → clustering → submit。STaR 是 single-sample 版，用 answer correctness 当 filter。

### 后续 work 的链条

STaR 是 self-improvement LLM 的祖师爷，后面一系列工作都从它扩展：

- **V-STaR**：用 verifier model 打分，保留 top-k 而不是 only correct
- **Quiet-STaR** [Zelikman 2024]：在 every token 后 silent 思考，把 internal thought 当 rationale——把 reasoning 推到 token level
- **ReST** [Singh 2023]：把 STaR 用到 RLHF 场景
- **ReFT**：直接用 RL 优化 rationale generation
- **Self-Rewarding LM** [Yuan 2024]：model 自己当 reward model
- **Constitutional AI** [Anthropic 2022]：critique-revise 替代 human feedback
- **PRM800K** [Lightman 2023]：给每步 reasoning 打分而非只看 final answer
- **OpenAI o1**：把 reasoning 推到 test-time compute，inference 时多 sample + self-verify

### 跟 RLHF/PPO 的对比

| 维度 | RLHF (PPO) | STaR |
|------|-----------|------|
| Reward | learned reward model | indicator(correct/wrong) |
| Optimization | on-policy + KL penalty | off-policy-ish, no KL |
| Sample efficiency | 低 | 高 |
| Variance | 高 | 低 |
| Exploration | 好 | 差 |

STaR 用 simplicity 换 power，代价是 exploration 差——greedy decode 让它只 explore 最 likely 的 reasoning path，错过 long tail 的正确推理。这是 quiet-STaR、Tree of Thoughts 等后续工作要补的洞。

### Bootstrapping 的历史血脉

- **Self-training** (Yarowsky 1995)：NLP 经典 semi-supervised
- **Co-training** (Blum & Mitchell 1998)：两个 model 互相 teach
- **AlphaGo Zero**：self-play from scratch
- **Yann LeCun 的 JEPA**：self-supervised prediction

STaR 的独特贡献：用 **language 的 compositional structure** 作为 bootstrap 媒介。AlphaGo 的 move 是 low-dimensional categorical，而 rationale 是 discrete, compositional, interpretable 的 rich structure。这让 self-improvement 在 reasoning 任务上第一次 work。

---

## Build Intuition 的核心 mental model

如果只记一件事，记这个：

**Rationale 是 latent variable，answer 通过 rationale 生成。STaR 用 answer correctness 作为 implicit reward，通过 greedy decode 把 REINFORCE 简化到 supervised loop。Rationalization 改变 sampling distribution 到 p(r|x,y)，相当于 importance sampling from a better proposal。每次 reset 到 pretrained M 是为了避免 overfitting 和 catastrophic forgetting。**

这个 framework 让你能预测 STaR 的所有行为：
- 为什么需要 seed capability（greedy explore 太弱，初始必须能命中几个 right answer）
- 为什么 rationalization 加速这么多（proposal distribution 更好）
- 为什么 high temperature 反而更差（bad reasoning 也会偶尔答对，被 reward 1，污染 training set）
- 为什么 binary task 会崩（chance accuracy 太高，noise > signal）

后续所有 self-improvement 工作都在这个 framework 上加东西：dense reward（PRM）、test-time compute（o1）、multiple samples（V-STaR）、process supervision（PRM800K）。STaR 是这一切的简洁起点。

---

## Reference Links

- Paper: [STaR: Bootstrapping Reasoning With Reasoning (NeurIPS 2022)](https://arxiv.org/abs/2203.14465)
- Author page: [Eric Zelikman](https://ezelikman.github.io)
- Few-shot CoT: [Wei et al. 2022](https://arxiv.org/abs/2201.11903)
- Scratchpad: [Nye et al. 2021](https://arxiv.org/abs/2112.00114)
- Expert Iteration: [Anthony et al. 2017](https://arxiv.org/abs/1705.08439)
- Self-Consistency: [Wang et al. 2022](https://arxiv.org/abs/2203.11171)
- Quiet-STaR: [Zelikman et al. 2024](https://arxiv.org/abs/2403.09629)
- Constitutional AI: [Anthropic 2022](https://arxiv.org/abs/2212.08073)
- PRM800K: [Lightman et al. 2023](https://arxiv.org/abs/2305.20050)
- Self-Rewarding LM: [Yuan et al. 2024](https://arxiv.org/abs/2401.10020)

---

# STaR: Self-Taught Reasoner — 深度技术讲解

## 一、Core Intuition

STaR 的核心思想可以用一句话概括：**let a model teach itself to reason by learning from its own correct reasoning**。具体来说，给定一个pretrained LLM M 和一个只有 question-answer pairs 的大数据集 D = {(x_i, y_i)}，STaR 通过一个 iterative loop 让模型自己生成 rationale，把"答对了"的 rationale 收集起来做 finetune data，下一轮再用更好的模型生成更好的 rationale。这是一个 **synergistic bootstrapping process**：rationale generation 改善 → training data 改善 → rationale generation 进一步改善。

这个工作最优雅的地方在于，它只需要 **10个左右的 few-shot rationale examples** 作为 seed，就能把一个 6B 的 GPT-J bootstrap 到接近 175B GPT-3 fine-tuned 的水平。

## 二、方法详解

### 2.1 Rationale Generation Bootstrapping（无 rationalization 的 STaR）

**Setup**:
- Pretrained LLM: M
- Dataset: D = {(x_i, y_i)}_{i=1}^D（只有 question 和 answer，没有 rationale）
- Few-shot prompt set: P = {(x_i^p, r_i^p, y_i^p)}_{i=1}^P，其中 P ≪ D（例如 P = 10）
- Concatenated input: x_i' = (x_1^p, r_1^p, y_1^p, ..., x_P^p, r_P^p, y_P^p, x_i)

**Loop**:
1. 用当前 model M_{n-1} 对每个 x_i 生成 (r̂_i, ŷ_i)
2. Filter：只保留 ŷ_i = y_i 的 (x_i, r̂_i, y_i)
3. 在 filtered dataset 上 finetune **原始 pretrained M**（不是继续训练 M_{n-1}，避免 overfitting）
4. 重复直到 performance plateau

### 2.2 为什么可以看作 Policy Gradient 的近似

这是 paper 最精彩的理论分析部分。把 M 看作一个 **discrete latent variable model**：

$$p_M(y | x) = \sum_r p(r | x) \cdot p(y | x, r)$$

其中：
- $p_M(y|x)$：marginal probability of answer y given question x
- $r$：latent rationale（隐变量）
- $p(r|x)$：rationale 的先验分布，由 model 采样
- $p(y|x,r)$：给定 rationale 后 answer 的概率

定义 **indicator reward function** $\mathbb{1}(\hat{y} = y)$（答对得 1，答错得 0），total expected reward：

$$J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot | x_i)} \mathbb{1}(\hat{y}_i = y_i) \quad (1)$$

变量解释：
- $J$：objective function，要 max 的 expected total reward
- $M$：current model
- $X = \{x_i\}$，$Y = \{y_i\}$：dataset 的 questions 和 ground-truth answers
- $\hat{r}_i, \hat{y}_i$：从 model 分布 $p_M(\cdot | x_i)$ 采样得到的 rationale 和 answer
- $\mathbb{1}(\cdot)$：indicator function

通过 **log-derivative trick**（REINFORCE 的核心 trick），gradient 为：

$$\nabla J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot | x_i)} \left[ \mathbb{1}(\hat{y}_i = y_i) \cdot \nabla \log p_M(\hat{y}_i, \hat{r}_i | x_i) \right] \quad (2)$$

变量解释：
- $\nabla J$：objective 的 gradient
- $\nabla \log p_M(\hat{y}_i, \hat{r}_i | x_i)$：log-probability 的 gradient，即 score function
- $\mathbb{1}(\hat{y}_i = y_i)$ 作为 reward weight，**直接 discard 所有答错的 samples 的 gradient**——这正好对应 STaR 的 filtering step（Algorithm 1 的 Line 5）

**STaR 做的两个 approximation**：
1. **Greedy decoding** 代替 sampling：减少 variance，代价是可能 bias exploration（只 explore 最 likely 的 rationale，错过 long tail 的正确 reasoning path）
2. **Multiple gradient steps on same batch**：类似 PPO 等 on-policy 算法的 multi-step update，提高 sample efficiency

这就是为什么 STaR "不需要显式 RL machinery 也能 work"——它本质上是 REINFORCE 的一个特殊化、简化版本。

### 2.3 Rationalization：处理"卡住"的问题

纯 rationale generation 的致命问题：**当 model 无法 solve training set 中的新问题时，loop 就停了**——因为这些 fail 的 examples 不产生任何 gradient signal。

**Rationalization 的解决方案**：给 model 一个 **hint**（即 ground-truth answer），让它 generate 一个能 lead 到这个 answer 的 rationale。例如 Figure 2 中，prompt 里直接告诉 model "(b) grocery cart" 是正确答案，然后让 model 生成 reasoning。

数学上，这相当于从 **$p(r | x, y)$** 采样，而不是从 $p(r | x)$ 采样。这两个 distribution 的关系：

$$p(r | x, y) = \frac{p(y | x, r) \cdot p(r | x)}{p(y | x)}$$

即 rationalization 是在给定答案的条件分布上采样 rationale。paper 在 Section 5 中提到，这可以理解为 **off-policy estimate** of the original objective，hint-augmented model 作为 proposal distribution。

**关键 trick**：把 rationalization 生成的 rationale 加入 training data 时，**不包含 hint**——假装 model 是"自己想出来的"。这其实是一种 **teacher forcing + student distillation** 的变体。

**Rationalization 的两个好处**：
1. 暴露 model 给 difficult problems（否则永远不会出现在 training set 中）→ "think outside the box"
2. 增加 dataset size

### 2.4 完整 Algorithm 1

```
Input: M (pretrained LLM), D = {(x_i, y_i)} (with few-shot prompts)
1: M_0 ← M
2: for n in 1...N do                          # Outer loop
3:   (r̂_i, ŷ_i) ← M_{n-1}(x_i)  ∀i           # Rationale generation
4:   (r̂_i^rat, ŷ_i^rat) ← M_{n-1}(add_hint(x_i, y_i))  ∀i  # Rationalization
5:   D_n ← {(x_i, r̂_i, y_i) | ŷ_i = y_i}      # Filter correct
6:   D_n^rat ← {(x_i, r̂_i^rat, y_i) | ŷ_i ≠ y_i ∧ ŷ_i^rat = y_i}  # Filter rationalized
7:   M_n ← train(M, D_n ∪ D_n^rat)            # Finetune original M
8: end for
```

注意第 7 行：每次都从 **原始 pretrained M** 重新 finetune，而不是 continual training M_{n-1}。这是为了 **avoid overfitting** 和 **catastrophic forgetting**。

## 三、实验细节与数据表

### 3.1 Arithmetic（n-digit addition）

Setup：跟随 [Nye et al., 2021] 的 scratchpad 格式。Figure 3 的例子：

```
Input: 624 + 259
Target:
<scratch>
624 + 259, C: 0      # 从个位开始，C = carry
2+5, 3, C: 1         # 2+5=7? 错，应是 4+9=13, 写 3 进 1
6+2, 83, C: 0        # 6+2+1(carry)=9? 这里是 6+2=8
, 883, C: 0
0883
</scratch>
883
```

**结果对比**：
| Method | Accuracy |
|--------|----------|
| Few-shot direct | ~0% (2-digit < 1%) |
| Baseline (no rationale, 10k examples, 5000 steps) | 76.3% |
| STaR (16 iterations) | **89.5%** |

**Rationalization 的 dramatic effect**（Figure 4）：without rationalization 时，performance 是 **stagewise**——model 在 (n-1)-digit 做好之前，n-digit 几乎不会进步。with rationalization 后，**many lengths 一起学**，1 iteration 后 2-digit 从 <1% → 32%。

**Out-of-distribution 泛化**（Figure 5）：在第 20 iteration 引入更多 digits，model 在 **9 和 10 digit**（从未见过）上也能 solve 很多问题——这是 emergence of systematic generalization 的 evidence。

### 3.2 CommonsenseQA

| Method | Dev Acc (%) | Train Data Used (%) |
|--------|------------|---------------------|
| GPT-3 Direct Finetuned (175B) | 73.0 | 100 |
| Few-shot Direct GPT-J | 20.9 | ~0 |
| Few-shot CoT GPT-J | 36.6 | ~0 |
| Few-shot CoT LaMDA 137B | 55.6 | ~0 |
| GPT-J Direct Finetuned | 60.0 | 100 |
| STaR without rationalization | 68.8 | 69.7 |
| **STaR with rationalization** | **72.5** | 86.7 |

Insight：6B 的 GPT-J + STaR ≈ 175B 的 GPT-3 fine-tuned，**用 86.7% 的 training data**。这说明 reasoning 信号比单纯的 answer prediction signal 信息量大得多——相同参数 budget 下，explicit reasoning 让 model 学到更 transferable 的 representation。

**Human Evaluation**：20 个 crowdworkers 评估 50 个 rationales，
- STaR-generated 比 few-shot CoT 的高 30%（p = .039）
- STaR-generated 比 human-generated 的高 74%（p < .001）

但作者谨慎地说：这 **不是** 说明 STaR 达到人类水平，而是说明 human annotation 的 rationale 质量也很差——eliciting high-quality rationales from humans 本身就很难。

### 3.3 GSM8K

| Method | Test Acc (%) | Train Data (%) |
|--------|--------------|----------------|
| Few-shot Direct GPT-J | 3.0 | ~0 |
| Few-shot CoT GPT-J | 3.1 | ~0 |
| GPT-J Direct Finetuned | 5.8 | 100 |
| STaR without rationalization | 10.1 | 25.0 |
| STaR with rationalization | 10.7 | 28.7 |

**有意思的发现**（Figure 6）：model 计算步数与 human ground truth 的 agreement 在 53%-57% 之间。disagreement 时，**model 通常用更少步数**——有时是 skip steps，有时是 **找到更简洁的 solution**（Figure 8 的例子：ground truth 用 7 步，STaR 用 1 步直接 180/2=90）。

## 四、Discussion 中的关键技术细节

### 4.1 Temperature vs. Rationalization

直觉上，要 expand training set，可以用 **higher temperature sampling**。但实验发现这 **counterproductive**：

- High temperature → correct answer with incorrect reasoning 的概率大幅增加
- Training on bad reasoning → generalization 崩溃
- Arithmetic 上特别明显：high-temp 的 scratchpad "diverge into meaninglessness"

而且 high temperature 在 computation 上 **far less efficient**：10 samples 慢 10 倍，而 rationalization 只多一次 forward pass。

**相关 work**：作者提到可以用 [Wang et al., 2022] 的 **self-consistency** 思路——majority vote 多个 high-temp samples 作为 ground truth，再用 low-temp sample 训练。这能让 STaR 在 **只有 questions 没有 answers** 的 dataset 上 work。这是后续 V-STaR、ReST 等工作的 seed idea。

### 4.2 Few-shot Prompting during Training

包含 few-shot prompts 在 sampling 时有 **两个好处**：
1. 减少 "drift"——rationale 不会越来越 dissimilar from 初始 few-shot style
2. Computational：shorter prompt → shorter sequence → faster

**Performance 影响**（CQA）：
- Without rationalization：60.9% → 68.8%（+7.9%）
- With rationalization：69.9% → 72.5%（+2.6%）

所以 paper 建议至少 training 初期保留 few-shot prompts。

### 4.3 Failure Modes（Appendix A，非常 enlightening）

作者列出 6 类 reasoning fallacies：

1. **Question Implies Answer**：rationale 形式为 "answer must be X. Y is X. Therefore Y."——没有解释 why Y satisfies X
2. **Begging the Question**：rationale 里直接 imply 答案
3. **Exercise to the Reader**：直接给答案不解释
4. **World State Assertions**：假装知道某个 specific person 的事（"James feels X"）而不是 general statement（"people feel X"）
5. **Red Herrings**：说一句 technically true 但 irrelevant 的话
6. **Hint Short-cutting**：rationalization 时 model 学到"hint 总是答案"的 shortcut，导致 rationale 和 final answer 不一致（"answer is train station (e)" 但 rationale 推理出 "airport"）

这些 failure modes 对后续 work（如 process reward model、PRM800K）启发很大——**outcome-only reward 无法区分 good reasoning 和 bad reasoning with correct outcome**。

## 五、Limitations

1. **初始 few-shot performance 必须 above chance**：GPT-2 在 arithmetic 上都无法 bootstrap
2. **High chance accuracy 的 task**（如 binary classification）会产生大量 bad rationales——怎么 filter 是 open problem
3. **Bias amplification**：CQA 自身有 gender bias 等，STaR 会 **amplify** 这些 bias，rationalization 尤其严重（把 model 平时不输出的 biased answer "pull out"）
4. **Faithfulness**：rationale 可能 **不反映** model 内部真实的 reasoning process——model 可能先 implicit 选 answer 再 post-hoc 编 rationale

## 六、相关联想与后续工作

### 6.1 与 AlphaGo / Expert Iteration 的关系

STaR 本质上是 **Expert Iteration (ExIt)** [Anthony et al., 2017] 的 language model 版本：
- ExIt：apprentice self-play → expert（MCTS+slow search）提供 feedback → imitation learning → 用 improved apprentice 替换 expert
- STaR：model generate rationale → ground-truth answer 作为 expert feedback → finetune → repeat

区别：STaR 的 "expert" 是 **fixed** 的（ground-truth answer），不需要训练 separate value function。

### 6.2 与 AlphaCode、Codex 的关系

AlphaCode 也用 **generate-filter-cluster** 的思路：生成大量 candidate solutions，用 tests 过滤，clustering 后 submit。STaR 是 single-sample version，用 answer correctness 作为 filter。

### 6.3 后续工作 chain

STaR 是一系列 "self-improvement" 工作的起点：

- **V-STaR** (Hosseini et al., 2023): 用 verifier model 给 generated rationales 打分，保留 top-k 而非仅 correct 的
- **Quiet-STaR** (Zelikman et al., 2024): 在每个 token 后 silent 思考，把 internal thought 作为 rationale
- **ReST** (Singh et al., 2023): 把 STaR 应用到 RLHF 场景，grow dataset iteratively
- **Reasoning with Reinforced Finetuning (ReFT)**: 用 RL 直接优化 rationale generation
- **Self-Rewarding Language Models** (Yuan et al., 2024): model 自己当 reward model
- **Constitutional AI** (Anthropic): 用 critique-revise 替代 human feedback，思想类似 rationalization

### 6.4 与 RLHF / PPO 的对比

| 维度 | RLHF (PPO) | STaR |
|------|-----------|------|
| Reward | learned reward model | indicator (correct/wrong) |
| Optimization | on-policy, KL penalty | off-policy-ish, no KL |
| Sample efficiency | 低（每步都 sample） | 高（greedy decode） |
| Variance | 高 | 低（greedy） |
| Exploration | high | limited（greedy 偏 bias） |

STaR 用 simplicity 换 power，但 lose 了 exploration 能力——这是后续 quiet-STaR、ToT 等工作要补的洞。

### 6.5 与 Process Reward Model (PRM) 的关系

STaR 的 failure modes（Section A）暴露的问题：**outcome reward 无法监督 process**。OpenAI 的 PRM800K 工作正是要解决这个问题——用 human annotator 给每一步 reasoning 打分，而不是只看 final answer。这可以看作 STaR 的 **dense reward** 版本。

### 6.6 与 Test-Time Compute 的关系

STaR 把 reasoning 放在 training time（finetune）。后来的 work（如 o1、Quiet-STaR）把更多 reasoning 推到 **test-time compute**——inference 时 generate 多个 rationale，self-verify，select best。这是 STaR 思想的 inference-time 推广。

### 6.7 与 bootstrapping 历史的联系

Bootstrapping 在 AI 历史上有 deep 传统：
- **Self-training** (Yarowsky, 1995)：NLP 中的经典 semi-supervised 方法
- **Co-training** (Blum & Mitchell, 1998)：两个 model 互相 teach
- **Yann LeCun 的 JEPA**：self-supervised prediction
- **AlphaGo Zero**：从零开始 self-play

STaR 的独特之处：**用 language 的 compositional structure 作为 bootstrap 媒介**——rationale 本身是 discrete, compositional, 可解释的。这比 AlphaGo 的 move distribution 更 rich。

## 七、对 Build Intuition 的总结

理解 STaR 的核心 mental model：

1. **Rationale 是 latent variable**：answer 通过 rationale 生成，rationale 通过 question 生成。p(y|x) = Σ_r p(r|x)p(y|x,r)。
2. **Filtering 等价于 zero reward for wrong samples**：indicator function 自动 discard 错误 rationale 的 gradient。
3. **Rationalization 是 posterior sampling**：从 p(r|x,y) 采样而非 prior p(r|x)，信息更富。
4. **每次 reset 到 pretrained M**：避免 overfitting 和 catastrophic forgetting，类似 actor-critic 中定期 reset target network。
5. **Greedy decode = low variance, high bias**：STaR 选择 simplicity over exploration，这是它能 scale 的关键。

## 八、Reference Links

- Paper: [STaR: Bootstrapping Reasoning With Reasoning (NeurIPS 2022)](https://arxiv.org/abs/2203.14465)
- Author page (Eric Zelikman): [https://ezelikman.github.io](https://ezelikman.github.io)
- Few-shot CoT paper: [Wei et al., 2022](https://arxiv.org/abs/2201.11903)
- Scratchpad paper: [Nye et al., 2021](https://arxiv.org/abs/2112.00114)
- Expert Iteration: [Anthony et al., 2017](https://arxiv.org/abs/1705.08439)
- Self-Consistency: [Wang et al., 2022](https://arxiv.org/abs/2203.11171)
- Quiet-STaR (后续): [Zelikman et al., 2024](https://arxiv.org/abs/2403.09629)
- Constitutional AI: [Anthropic, 2022](https://arxiv.org/abs/2212.08073)
- PRM800K (OpenAI process reward): [Lightman et al., 2023](https://arxiv.org/abs/2305.20050)
- CommonsenseQA dataset: [Talmor et al., 2019](https://aclanthology.org/N19-1421/)
- GSM8K dataset: [Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)

---

总结一句 build intuition 的话：**STaR 把 reasoning 当 latent variable，用 answer correctness 作为 implicit reward，通过 greedy decode 把 REINFORCE 简化成一个 pure supervised learning loop**——这是它能 scale、能 plug into standard LLM training pipeline 的根本原因。后续所有 self-improvement 工作几乎都在这个框架上加东西：dense reward、test-time compute、multiple samples、process supervision 等。
