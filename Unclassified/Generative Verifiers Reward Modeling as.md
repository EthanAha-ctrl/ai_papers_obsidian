---
source_pdf: Generative Verifiers Reward Modeling as.pdf
paper_sha256: 0ca07ace7efe542a81ca5f58d10099011288f105aaba467b98b005ec5af918ec
processed_at: '2026-08-04T14:16:20-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 GenRM

## 这篇 paper 在解决什么问题

LLM 做数学题，会犯各种 subtle 错误。一个常见办法是 Best-of-N——让 model 生成 N 个答案，然后用一个 "verifier" 打分，选最好的。

问题是：**怎么训练这个 verifier**。

传统做法是在 LLM 上面接一个 classification head，输出一个 0 到 1 的分数。这就像让一个会写作文的人，只准他用手指比个数字来打分——浪费了他写作文的能力。

GenRM 的 idea 非常简单：**别加 head 了，直接问 LLM "答案对不对"，让它回答 Yes 或 No**。分数就是它说 Yes 的概率。

## 为什么这个简单的 idea work

因为一旦 verification 变成了 "回答 Yes/No"，LLM 就能用上它所有 pretraining 学到的能力：

1. **CoT reasoning**：可以先说 "让我一步步检查"，写一段 reasoning，最后再给 Yes/No。传统 discriminative RM 做不到，因为它不能生成文本。

2. **Majority voting**：让 verifier 生成 32 次推理路径，取平均。单次推理可能出错，多次取平均就 robust 了。传统 RM 做不到，因为它每次输出都一样。

3. **和 SFT 统一**：同一个 model 既能学怎么解题，又能学怎么判题。两件事互相帮助——会解题的人更会判题，会判题的人也更会解题。

## 最关键的 insight

论文里最打动我的一个 example：

题目说 Hulu 和 Disney Plus **各** $10，bundle 享 20% 折扣。solution 算成了 $10 × 80% = $8，忽略了 "各" 字，应该是 $20 × 80% = $16。

Discriminative RM 给了 0.999 分——完全被骗了，因为这个 solution 读起来很 convincing，数字也对得上。

GenRM-CoT 先写推理："Step 3: bundle 应该是 $10 + $10 = $20，折扣后 $16，solution 说 $8 是错的"。然后给 No。

**区别就在这里**：discriminative RM 只看 surface pattern，GenRM-CoT 会真正去 reason 每一步。

## Synthetic CoT 怎么来的

训练 GenRM-CoT 需要 verification rationale，但人工写太贵。论文的 trick：

给 Gemini 看题目 + 待验证的 solution + 一个正确答案的 reference solution，让它对照着检查。然后用最终 Yes/No 是否正确来 filter。

关键：训练时只给 verifier 看题目和 candidate solution，不给 reference——否则 test 时没有 reference 就 mismatch 了。

## 结果有多好

几个数字：

- GSM8K：73% → 93.4%，超过 GPT-4 和 Gemini 1.5 Pro
- MATH（从 grade-school math generalize 到高中竞赛题）：28% → 44.6%
- MMLU abstract algebra：37.9% → 53.5%
- Algorithmic tasks：几乎 match oracle verifier

而且越难的任务，GenRM-CoT 比 discriminative RM 提升越大。easy task 直接看答案就行，hard task 必须 step-by-step reason 才能发现问题。

## 一个有意思的发现

把 verification data 加到 generator 训练里，generator 自己的解题能力也变强了。

这听起来反直觉——教 model 怎么挑别人的错，怎么会让它自己解题更好？

intuition 是：判题和解题 share 同一个 underlying understanding。你真正理解了 "什么是对的推理"，才能既做对题又挑得出错。这就像一个 good debugger 往往也是 good programmer。

## 和其他方法比

- **vs Discriminative RM**：GenRM 能 CoT，能用 inference compute，能统一训练。性能 consistently 更好。
- **vs LLM-as-a-Judge**：直接 prompt 一个更强的 model（比如 Gemini 1.0 Pro）当 judge，不如 fine-tune 一个更小的 Gemma 当 GenRM。说明 fine-tuning 比 model size 更重要。
- **vs DPO verifier**：DPO 把 reward 藏在 policy logits 里，架构上 tied，不能同时做 generation。GenRM 简单直接，性能更好。

## 对未来的启示

最大的启示是：**LLM 的本质能力是 generation，任何把它当 classifier 用的设计都在浪费这个能力**。

verification 是 generation 的一个 special case。tool use 是。planning 也是。只要你能把任务 cast 成 "predict next token"，LLM 的全部武器库都能用上。

这和 Sutton 的 Bitter Lesson 一脉相承——general methods that leverage computation 胜过 domain-specific tricks。discriminative head 是 domain-specific trick，next-token prediction 是 general method。

未来 agent 的形态可能就是一个 unified LLM：能 generate solution，能 verify solution，能生成 verification rationale，能根据 verification 修正 solution。这就是 self-improvement loop 的基础。

---

# Generative Verifiers: Reward Modeling as Next-Token Prediction 深度解析

## 一、论文核心 Intuition

这篇 paper 来自 Google DeepMind（Lunjun Zhang, Arian Hosseini, Hritik Bansal 等），核心 idea 非常 elegant：**把 verifier 的训练从 discriminative classification 重构为 next-token prediction**。

传统 discriminative RM（如 Cobbe et al. 2021 的 ORM）在 LLM 顶部加一个 classification head，用 sigmoid 输出一个 scalar score $r_\theta(x,y) \in [0,1]$。这个做法的问题在于——LLM 本质是 generative model，pretraining 学到的是 token 分布 $p_\theta(y_t | x, y_{<t})$，而 discriminative head 强行把表征压成一个标量，**丢弃了 LLM 的 generation 能力**：无法做 CoT reasoning，无法用 inference-time compute，无法和 SFT 统一。

GenRM 的解法：让 verification decision 就是 "yet another token"。用 prompt "Is the answer correct (Yes/No)?"，score 就是 $p_\theta(\text{'Yes'} | x, y, I)$。这样 verification 天然嵌入到 next-token prediction 框架里，所有 LLM 的 tools（CoT、majority voting、instruction tuning）都能直接用。

论文链接：https://arxiv.org/abs/2408.15240

## 二、方法细节与公式解析

### 2.1 Preliminaries

autoregressive LM 的基本概率：

$$p_\theta(\mathbf{y} | \mathbf{x}) = \prod_{t=1}^{T} p_\theta(y_t | \mathbf{x}, y_{<t})$$

- $\mathbf{x}$: input context（如数学题）
- $\mathbf{y} = (y_1, ..., y_T)$: output token sequence
- $\theta$: model parameters
- $y_{<t} = (y_1, ..., y_{t-1})$: previous tokens

next-token probability 通过 softmax with temperature $\gamma$：

$$p_\theta(y_t | \mathbf{x}) = \frac{\exp(z_t / \gamma)}{\sum_{i=1}^{M} \exp(z_i / \gamma)}$$

- $z_t = \text{logit}_\theta(y_t | \mathbf{x}, y_{<t})$: token $t$ 的 logit
- $M$: vocabulary size
- $\gamma$: temperature，$\gamma=0$ 为 greedy decoding

SFT loss（标准 cross-entropy）：

$$\mathcal{L}_{\text{SFT}}(\theta, \mathcal{D}) = -\mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}} \left[\sum_{t=1}^{|\mathbf{y}|} \log p_\theta(y_t | \mathbf{x}, y_{<t})\right]$$

### 2.2 Discriminative RM Baseline（对比方法）

传统 verifier 训练用 binary cross-entropy：

$$\mathcal{L}(\theta, \mathcal{D}_{RM}) = -\mathbb{E}_{(\mathbf{x}, \mathbf{y}^+) \sim \mathcal{D}_{\text{correct}}} [\log r_\theta(\mathbf{x}, \mathbf{y}^+)] - \mathbb{E}_{(\mathbf{x}, \mathbf{y}^-) \sim \mathcal{D}_{\text{incorrect}}} [\log(1 - r_\theta(\mathbf{x}, \mathbf{y}^-))]$$

其中 $r_\theta(\mathbf{x}, \mathbf{y}) = \text{sigmoid}(z_{cls})$，$z_{cls} = \text{logit}_\theta(cls | \mathbf{y}, \mathbf{x})$，$cls$ 是一个 special vocabulary token。

这里的关键问题：$r_\theta$ 只是一个标量，无法生成文本，无法做 CoT，无法用更多 inference compute。

### 2.3 GenRM Direct Verifier

训练数据格式：

$$\mathcal{D}_{\text{Direct}} = \{(\mathbf{x}, \mathbf{y}^+, I, \text{'Yes'})\} \cup \{(\mathbf{x}, \mathbf{y}^-, I, \text{'No'})\}$$

- $I = $ "Is the answer correct (Yes/No)?"
- $\mathbf{y}^+$: correct solution
- $\mathbf{y}^-$: incorrect solution

推理时 verifier score：

$$r_{\text{Direct}}(\mathbf{x}, \mathbf{y}) = p_\theta(\text{Yes} | \mathbf{x}, \mathbf{y}, I)$$

**intuition**：这个 score 利用了 model 对正确性的 confidence，reduces test-time error compared to 硬性 binary prediction。

### 2.4 Unified Generation + Verification（关键创新）

GenRM 的 unified loss：

$$\boxed{\mathcal{L}_{\text{GenRM}}(\theta, \mathcal{D}_{\text{verify}}) = \mathcal{L}_{\text{SFT}}(\theta, \mathcal{D}_{\text{verify}}) + \lambda \mathcal{L}_{\text{SFT}}(\theta, \mathcal{D}_{\text{correct}})}$$

- $\mathcal{D}_{\text{verify}}$: verification data（含 Yes/No token 或 CoT rationale）
- $\mathcal{D}_{\text{correct}}$: 生成正确 solution 的 SFT data
- $\lambda > 0$: mixture ratio（algorithmic tasks 用 $\lambda=1/3$，GSM8K 用 $\lambda=1/4$）

**直觉**：generation 和 verification 是 related tasks——知道如何生成正确 solution，有助于判断 solution 是否正确（positive transfer）。Figure 9 的实验证实了这一点：加 SFT loss 后 verification 性能一致提升。Figure 10 反过来显示：加 verification data 到 generator 训练里，generator 的 Pass@N 也提升。这是 bidirectional synergy。

这个 unified training 是 DPO verifier（Hosseini et al. 2024, Rafailov et al. 2024）做不到的，因为 DPO 把 reward 隐式表示在 policy logits 里，架构上 tied，难以同时做 generation。

### 2.5 GenRM-CoT：Chain-of-Thought Verification

这是论文最有价值的部分。训练数据：

$$\mathcal{D}_{\text{CoT}} = \{(\mathbf{x}, \mathbf{y}^+, I_{\text{CoT}}), (\mathbf{v}_{\text{CoT}}, I, \text{'Yes'})\} \cup \{(\mathbf{x}, \mathbf{y}^-, I_{\text{CoT}}), (\mathbf{v}_{\text{CoT}}, I, \text{'No'})\}$$

- $I_{\text{CoT}} = $ "Let's verify step by step."
- $\mathbf{v}_{\text{CoT}}$: verification rationale（逐步检查 reasoning）

推理时 score：

$$r_{\text{CoT}}(\mathbf{x}, \mathbf{y}) = p_\theta(\text{Yes} | \mathbf{x}, \mathbf{y}, I_{\text{CoT}}, \mathbf{v}_{\text{CoT}}, I)$$

其中 $\mathbf{v}_{\text{CoT}} \sim p_\theta(\cdot | \mathbf{x}, \mathbf{y}, I_{\text{CoT}})$——verifier 先自己生成一段 CoT rationale，再基于这段 rationale 给出 Yes/No probability。

### 2.6 Majority Voting for Inference-Time Compute

$$r_{\text{MajV@K}}(\mathbf{x}, \mathbf{y}) = \frac{1}{K} \sum_{i=1}^{K} p_\theta(\text{Yes} | \mathbf{x}, \mathbf{y}, I_{\text{CoT}}, \mathbf{v}_{\text{CoT}}^{(i)}, I)$$

- $K$: vote 数量（默认 $K=32$）
- $\mathbf{v}_{\text{CoT}}^{(i)}$: 第 $i$ 次采样的 CoT rationale

**intuition**：单条 CoT 可能有 reasoning error，marginalize 掉这些 path 的 noise——average 多条 rationale 的 Yes 概率。这是 self-consistency（Wang et al. 2022）思想在 verification 上的应用。discriminative RM 做不到这点，因为它无法生成多个 rationale。

## 三、Synthetic Verification CoT Rationales 生成

这是一个关键的工程创新。问题是：人类写 verification rationale 太贵，且 LLM reasoning 超过人类后无法 scale。

**naive approach**：直接 prompt "Let's verify step by step"，filter 掉最终 Yes/No 不对的 rationale。问题：即使 filter 了，rationale 质量仍然差（50% random guessing accuracy 意味着 filter 后的 rationale 仍有大量 subtle errors）。

**Reference-Guided Grading**（论文的解法）：

prompt 格式（Table A.2）：
```
You are a math teacher. Grade the Solution, verifying correctness step by step.
Use Expected Answer to find any erroneous step in the Solution.
At the end of the Solution verification, when you give your final grade, write it in 
the form "Verification: Is the answer correct (Yes/No)? X"

Question: {problem}
Solution: {solution}
Expected Answer: {a solution that arrives at the correct answer}
```

关键点：
1. 给 LLM 一个 correct reference solution 作为 privileged information，帮助它定位 candidate solution 的错误
2. **训练时不包含 reference**，避免 train/test mismatch
3. 用同一个 model（Gemini 1.0 Pro）生成 solution 和 rationale，不需要更强大的 model

filtering 策略：只保留 >50% rationale 与 answer checker 一致的 solution 的 rationale。

Figure 13 显示 reference guidance 至关重要：91.7% vs 87.8%（Gemma-7B on GSM8K）。
Figure 14 显示 scaling rationale 数量 per solution 也有帮助（ensembling effect 防止 overfitting 到 rationale noise）。

## 四、实验结果深度分析

### 4.1 主要结果（Figure 1, 5）

| Task | Base | Disc-RM | GenRM | GenRM-CoT | Improvement |
|------|------|---------|-------|-----------|-------------|
| Algorithmic (Gemma-2B) | 5% | ~30% | ~35% | 45.3% | +15.3% over Disc |
| GSM8K (Gemma2-9B) | 73% | ~88% | ~90% | 93.4% | +5.4% over Disc |
| MATH500 (Best-of-32) | 28% | ~38% | ~40% | 44.6% | +6.6% over Disc |

GenRM-CoT 在 algorithmic tasks 上 nearly matches oracle verifier performance——说明在有 clean CoT data 的 ideal scenario 下，generative verification 接近上限。

### 4.2 MMLU Easy-to-Hard Generalization（Table 1）

verifier 只在 grade-school math（GSM8K）上训练，测试 college-level math：

| MMLU Dataset | Base | Disc-RM | GenRM-CoT | Improvement |
|--------------|------|---------|-----------|-------------|
| elementary_mathematics | 80.1% | 90.6% | 91.1% | +0.5% |
| high school mathematics | 52.2% | 74.8% | 76.1% | +1.3% |
| college_mathematics | 47.6% | 53.0% | 56.1% | +3.1% |
| abstract_algebra | 37.9% | 50.0% | 53.5% | +3.5% |

**关键 insight**：越难的任务，GenRM-CoT 的 improvement 越大。abstract algebra 上 +3.5%，而 elementary 只 +0.5%。这说明 CoT reasoning 对 hard problems 更有价值——easy problems 直接看答案就能判断，hard problems 需要逐步推理才能发现 subtle errors。

### 4.3 Sample Efficiency（Figure 5 right）

在 MATH 上，GenRM-CoT 用 6.4× fewer solutions 就能达到 discriminative RM 的 Best-of-32 performance。这是巨大的 compute saving。

### 4.4 Scaling Laws（Figure 11, 12）

**Inference-time compute scaling**：GenRM-CoT 的 performance 随 vote 数 $K$ graceful scaling。$K=2$ 就超过 greedy decoding。LLM-as-a-Judge（用更强的 Gemini 1.0 Pro）即使同样用 32 votes，仍然不如 fine-tuned Gemma GenRM-CoT——说明 training 比单纯用更强 model 更重要。

**Model size scaling**：Gemma 2B → 7B → 9B，GenRM 和 GenRM-CoT 都有 positive scaling trend，且 consistently outperform discriminative RM。bigger model 更能利用 text generation 能力做 CoT reasoning。

### 4.5 Unified Training 的双向 benefit（Figure 9, 10）

- Figure 9：加 SFT loss（$\lambda > 0$）后 verification 性能提升，对所有 task 和 GenRM/GenRM-CoT 都成立
- Figure 10：加 verification data 到 generator 训练后，generator 的 Pass@N（用 oracle verifier）也提升

这是 bidirectional positive transfer 的强证据。intuition：教会 model "判断 solution 是否正确" 和 "生成正确 solution" 是 complementary skills，shared underlying understanding。

### 4.6 Weighted Self-Consistency（Figure 8）

self-consistency 和 verification 可以 combine：每个 solution 用 verifier score 加权，选 weight 最大的 answer。

在 MATH 上，weighted SC based on GenRM-CoT 比 based on Discriminative RM 用 2.5× fewer solutions 达到同样性能。

实现细节（Appendix C）：对每个 answer，只 sum top-K scores（K=4 for MATH, K=6 for GSM），rather than sum all scores——slightly improves performance。

## 五、定性案例分析

### Figure 2 案例（GSM8K）

Problem: Tim 取消 cable，买 streaming services。Netflix $10/月，Hulu 和 Disney Plus 各 $10/月但 bundle 享 20% discount。cable $60/月。save 多少？

Solution 错误：忽略了 "each" 这个词，把 Hulu+Disney 算成 $10 而非 $20。

- Discriminative RM score: 0.999（**完全被骗**，因为 solution 看起来很 convincing）
- GenRM-CoT: 正确识别 Step 3 错误，"The bundle of Hulu and Disney Plus costs $10 * 80% = $8" 是错的，应该是 $10 + $10 = $20，discount 后 $16。score: 0.0015

这个案例完美展示了 CoT reasoning 的价值：discriminative RM 只看 surface pattern，GenRM-CoT 能 explicit reason about each step。

### Figure 4 案例（MATH transfer from GSM）

Line parameterization 问题，solution 在简化中间步骤时出错：

$y = -1 + 3\left(\frac{x-8}{2}\right) = \frac{3}{2}x - \frac{23}{2}$

正确应该是 $-1 + (3x-24)/2 = -1 + 3x/2 - 12 = 3x/2 - 13$

- Discriminative RM: 0.827（被骗）
- GenRM-CoT: 0.438（检测到错误）

注意：这个 verifier 只在 GSM8K 上训练，但能 generalize 到 MATH 的 algebra 错误。

### Figure 15 案例

90 minutes 没转换成 1.5 hours 就除以 7.5 hours，得到 12% 而非 20%。Discriminative RM: 0.841（被骗，因为 90/7.5=12 看起来 valid）。GenRM-CoT: 0.25（检测到单位错误）。

## 六、与 Related Work 的对比

### 6.1 vs Discriminative RM（Cobbe et al. 2021, Lightman et al. 2023）

- Cobbe et al.: https://arxiv.org/abs/2110.14168
- Lightman et al. (PRM): https://arxiv.org/abs/2305.20050

Discriminative RM 输出 scalar，无法 CoT，无法用 inference compute。GenRM 保留 generation 能力。

PRM（Lightman et al.）是 process-level supervision，给每个 step 打分，仍然是 discriminative。GenRM 可以 extend 到 process-level（论文 future work 提到）。

### 6.2 vs LLM-as-a-Judge（Zheng et al. 2024）

- 论文: https://arxiv.org/abs/2306.05685

LLM-as-a-Judge 直接 prompt off-the-shelf LLM，不 fine-tune。论文实验显示：即使 用更强的 Gemini 1.0 Pro 做 judge，仍然不如 fine-tuned Gemma GenRM-CoT。

intuition：fine-tuning 让 verifier 学到 task-specific verification patterns，off-the-shelf LLM 即使更大也 lack this specialization。

### 6.3 vs DPO Verifier（Hosseini et al. 2024, Rafailov et al. 2024）

- DPO: https://arxiv.org/abs/2305.18290
- V-STaR: https://arxiv.org/abs/2402.06457

DPO 把 reward 隐式表示为 $\log \pi_{\text{DPO}}(y|x) - \log \pi_{\text{ref}}(y|x)$。问题：
1. 无法 unified generation + verification（架构 tied）
2. 需要 reference policy
3. 容易 erroneous extrapolation（Pal et al. 2024, Pang et al. 2024）

GenRM 用简单 next-token prediction，不需要 reference policy，性能显著更好（Figure 1）。

有趣的 ablation（Figure D.5）：DPO verifier 用 $\log \pi_{\text{DPO}}(y|x)$ 直接作 score（不减 reference log prob）效果更好——和 Hosseini et al. 2024 的发现一致。

### 6.4 vs Critique-Out-Loud RM（Ankner et al. 2024）

- 论文: https://arxiv.org/abs/2408.11791

concurrent work，也用 critique 做 RM，但 RM head 仍然是 discriminative。GenRM 用 next-token prediction，能 unified generation + verification。

### 6.5 vs Self-Taught Evaluators（Wang et al. 2024）

- 论文: https://arxiv.org/abs/2408.02666

用 LLM-as-a-Judge 提取 preference signal，不直接训练 verifier 生成 CoT。GenRM 直接训练 verifier 生成自己的 CoT。

## 七、关键 Hyperparameters（Appendix B）

| Parameter | GenRM | Disc-RM | DPO |
|-----------|-------|---------|-----|
| Learning Rate | $2e-6$ (best) | $1e-7$ | $1e-6$ |
| Weight Decay | $1e-2$ | $1e-2$ | - |
| $\lambda$ (gen loss) | $1/3$ (algo), $1/4$ (GSM) | - | - |
| $\beta$ (DPO) | - | - | $0.1$ |
| Batch Size | 64 | 64 | 64 |
| Steps | 300K | 300K | 300K |
| Optimizer | AdamW | AdamW | AdamW |

Disc-RM 用 z-loss = $10^{-4} \cdot \log^2 Z$ 正则化（$Z$ 是 softmax normalizer），来自 PaLM（Chowdhery et al. 2023）和 Wortsman et al. 2023 的稳定性技术。

## 八、数据生成细节（Appendix A）

### Algorithmic Tasks

- **Last Letter Concatenation**: 给 word list，concatenate 每个词的 last letter。训练 length {2,3,4}，测试 length 6（length generalization）。每个 length 350 problems × 128 attempts from Gemma-2B。~50K training points。
- **Word Sorting**: alphabetical sort。训练 {2,3,4}，测试 5。每个 length 4096 lists × 64 attempts。~100K training points。

这两个 task 可以 algorithmically 生成 ground-truth verification CoT（Table A.1），是 ideal scenario 用来验证 GenRM-CoT 的上限。

### GSM8K

- 7.2K train, 1.3K test, 128 validation
- 每题生成 50 solutions，随机采样 max 16 correct + 16 incorrect
- 用 Gemini 1.0 Pro + reference guidance 生成 synthetic rationales
- 评估用 16 solutions per test problem

## 九、Limitations 与 Future Work

论文自己提到的方向：
1. 扩展到 coding, alignment, text-to-image, open-ended generation
2. Process-level supervision（像 PRM 那样给每个 step 打分）
3. 用 RL 训练 CoT verifier（当前只是 SFT）
4. 结合 RAG, many-shot learning, tool use

我补充一些 potential issues：
1. **CoT rationale 的 bias**：synthetic rationale 来自同一个 model，可能继承 model 的 blind spots。如果 model 在某类问题上 systematic 错误，rationale 也会 reinforce 这个错误。
2. **Inference cost**：32 votes 的 majority voting 意味着 32× 的 inference cost。虽然 paper 说 sample efficient，但 absolute cost 仍高。可以 explore adaptive voting（easy problems 少 vote，hard problems 多 vote）。
3. **Calibration**：paper 提到 GenRM 有 better calibrated uncertainty（Kapoor et al. 2024），但没有详细 calibration 分析。这对 RLHF pipeline 重要。
4. **OOD generalization 机制**：为什么 GSM8K 训练的 verifier 能 generalize 到 MATH？是 CoT reasoning 的 general reasoning ability，还是 math concepts 的 overlap？需要更深入分析。

## 十、对 LLM 训练的 broader implications

### 10.1 Verification as Universal Interface

GenRM 的 deep insight：**next-token prediction 是 universal interface**。任何 task 只要能 cast 成 "predict next token"，就能利用 LLM 的全部能力。verification 如此，tool use 如此，planning 也可以。

这呼应了 Sutton 的 "Bitter Lesson"——general methods that leverage computation 胜过 domain-specific methods。discriminative RM 是 domain-specific（classification head），GenRM 是 general（next-token prediction）。

### 10.2 Inference-Time Compute Scaling

论文显示 GenRM-CoT 能 scale with inference compute。这和 Brown et al. 2024（Large Language Monkeys, https://arxiv.org/abs/2407.21787）、Snell et al. 2024 的 test-time compute scaling 趋势一致。

未来方向：adaptive compute allocation——简单 problem 用 GenRM direct，难 problem 用 GenRM-CoT with many votes。这需要 verifier 能 estimate 自己的 uncertainty。

### 10.3 Unified Agent Architecture

unified generation + verification 暗示一个 future agent architecture：同一个 LLM 既能 generate solution，又能 verify solution，还能 generate verification rationale。这是 self-improvement loop 的基础——agent 可以自己 generate, verify, critique, refine。

类似思想见：
- Self-Refine (Madaan et al. 2023): https://arxiv.org/abs/2303.17651
- Reflexion (Shinn et al. 2023): https://arxiv.org/abs/2303.11366
- Self-Rewarding LM (Yuan et al. 2024): https://arxiv.org/abs/2401.10020

### 10.4 Process Reward Models 的 generative 版本

PRM（Lightman et al. 2023）给每个 step 打分，是 discriminative。GenRM 可以 natural extend 到 process-level：对每个 step 生成 CoT verification，marginalize step scores。

Math-Shepherd（Wang et al. 2023, https://arxiv.org/abs/2312.08935）和 Luo et al. 2024（https://arxiv.org/abs/2406.06592）探索 label-free process supervision，可以和 GenRM-CoT 结合——用 synthetic CoT rationale 做 step-level verification。

### 10.5 与 RLHF/RLAIF 的关系

GenRM 作为 reward model 可以直接插入 RLHF pipeline。相比 discriminative RM，GenRM 的优势：
1. Better verification accuracy → better policy gradient signal
2. CoT reasoning → 能提供 interpretable reward signal
3. Unified with policy → 可以做 actor-critic style training

但 paper 没有做 RL 实验，这是 future work。concurrent work 如 Wang et al. 2024 的 Implicit Reward Modeling（https://arxiv.org/abs/2402.13210）探索类似方向。

## 十一、复现考量

如果想复现或 extend 这篇工作：

1. **Model**: Gemma-2B / 7B / 9B，open weights
2. **Framework**: seqio (Roberts et al. 2022, https://arxiv.org/abs/2203.17189) for data mixture
3. **Key trick**: reference-guided grading for synthetic rationale generation——这个 trick 对性能 critical
4. **Data balancing**: 50% correct / 50% incorrect，必须 balanced
5. **$\lambda$ tuning**: 太大会 hurt verification（Figure D.3），GSM 用 1/4 是 sweet spot
6. **Majority voting**: $K=32$ 是 default，但 $K=2$ 就超过 greedy——可以根据 budget 调整

## 十二、总结

GenRM 是一个 conceptually clean 的 idea：把 verification cast 成 next-token prediction。这个 simple reframing 解锁了 CoT reasoning, inference-time compute scaling, unified generation+verification 三个 LLM 的核心能力。

实验结果 convincing：在 algorithmic, GSM8K, MATH, MMLU 上 consistently outperform discriminative RM, LLM-as-a-Judge, DPO verifier。easy-to-hard generalization 尤其 impressive——GSM8K 训练的 verifier 能在 MATH 和 college math 上显著提升。

最重要的 insight for building intuition：**LLM 的 power 在于 generation，任何把 LLM 当 classifier 用的方法都在浪费这个 power**。verification 是 generation 的一个 special case，应该用 generation 的方法做。

这个 work 对未来的 agent architecture、self-improving systems、test-time compute scaling 都有重要启示。

---

**相关参考文献链接**：
- 论文: https://arxiv.org/abs/2408.15240
- Cobbe et al. 2021 (Training Verifiers): https://arxiv.org/abs/2110.14168
- Lightman et al. 2023 (PRM): https://arxiv.org/abs/2305.20050
- Wei et al. 2022 (CoT): https://arxiv.org/abs/2201.11903
- Wang et al. 2022 (Self-Consistency): https://arxiv.org/abs/2203.11171
- Rafailov et al. 2024 (DPO): https://arxiv.org/abs/2305.18290
- Hosseini et al. 2024 (V-STaR): https://arxiv.org/abs/2402.06457
- Zheng et al. 2024 (LLM-as-a-Judge): https://arxiv.org/abs/2306.05685
- Ankner et al. 2024 (Critique-Out-Loud): https://arxiv.org/abs/2408.11791
- Saunders et al. 2022 (Self-Critiquing Models): https://arxiv.org/abs/2206.05802
- McAleese et al. 2024 (LLM Critics): https://arxiv.org/abs/2407.00215
- Brown et al. 2024 (Large Language Monkeys): https://arxiv.org/abs/2407.21787
- Agarwal et al. 2024 (Many-Shot ICL): https://arxiv.org/abs/2404.11018
- Singh et al. 2023 (Beyond Human Data): https://arxiv.org/abs/2312.06585
- Zelikman et al. 2022 (STaR): https://arxiv.org/abs/2203.14465
- Wang et al. 2023 (Math-Shepherd): https://arxiv.org/abs/2312.08935
- Luo et al. 2024 (Automated Process Supervision): https://arxiv.org/abs/2406.06592
- Gemma: https://arxiv.org/abs/2403.08295
- Gemma 2: https://arxiv.org/abs/2408.00118
