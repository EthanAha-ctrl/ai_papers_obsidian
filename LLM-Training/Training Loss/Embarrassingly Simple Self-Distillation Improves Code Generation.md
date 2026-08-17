---
source_pdf: Embarrassingly Simple Self-Distillation Improves Code Generation.pdf
paper_sha256: 96536a682ffd8f35632f2b3d775c649b67455c0f8f2a3e4218249e17bbfeab32
processed_at: '2026-08-04T03:11:54-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 让我换一种更接地气的方式来讲这个故事，尽量用大白话把 intuition 给 build 起来。

---

## 一句话总结

让 model 自己写代码（不管写得多烂），然后拿这些代码去 fine-tune 自己，model 就变强了。没有标准答案，没有判题器，没有老师模型，没有任何外部信号。

---

## 故事怎么开始的

Apple 这帮人提出了一个特别朴素的问题：

> 现在的 LLM 写代码已经很强了，但是要让它更强，传统路子要么靠人标数据（贵），要么靠更强的 teacher model distillation（有 ceiling），要么靠 execution-based RL（复杂且不稳定）。那能不能什么都不靠，就让 model 自己玩自己的？

答案居然是 yes。而且方法简单到让人觉得这论文是不是搞错了什么。

---

## 方法到底有多简单

三步走：

**第一步：让 model 写作业**

拿一批 competitive programming 的题目（大约 10K 道，来自 rSTARcoder 数据集），每道题让 base model 用特定 temperature $T_{\text{train}}$ 和 top-k/top-p 采样 **一个** solution。

注意：**不验证对错**。不管 model 写的是对的、错的、还是 halfway 就开始胡言乱语的，全部保留。唯一的 filter 是把空输出和一行代码的占位符去掉。

**第二步：用这些 solution 做标准 SFT**

就是最朴素的 cross-entropy loss：

$$\mathcal{L}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{SSD}}} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})$$

这里 $x$ 是题目 prompt，$y$ 是 model 自己生成的 solution（可能完全跑不通），$y_t$ 是第 $t$ 个 token，$y_{<t}$ 是前面所有 token。就这，标准的 next-token prediction，和 pre-training 一模一样。

**第三步：用 fine-tuned model 去评测**

用 evaluation temperature $T_{\text{eval}}$ 和对应的 top-k/top-p 去 decode。

完事了。没有 trick，没有 auxiliary loss，没有 contrastive learning，没有 RL reward shaping，什么都没有。

---

## 结果有多离谱

Qwen3-30B-Instruct 在 LiveCodeBench v6 上：

| 指标 | Base | +SSD | 提升幅度 |
|:---|:---|:---|:---|
| pass@1 (Overall) | 42.4% | 55.3% | +12.9pp (+30%) |
| pass@1 (Hard) | 18.3% | 33.6% | +15.3pp |
| pass@5 (Hard) | 31.1% | 54.1% | +23.0pp |

五个 model（Llama-8B, Qwen 4B/30B 的 instruct 和 thinking 变体）全部提升。越难的题提升越大。pass@5 的提升普遍大于 pass@1，说明 diversity 不降反升。

---

## 为什么会有效：Lock 和 Fork

这是整篇论文最深刻的 insight。作者观察到了 code generation 中存在两种截然不同的 context：

**Lock position（锁位）**

比如你写了 `if n == `，下一个 token 几乎肯定是某个具体的值。这时候 distribution 长这样：一个 dominant token 占了绝大部分概率，但 long tail 上还有一堆 syntactically plausible 但 semantically wrong 的 distractor token 在那儿挂着。

这些 distractor 平时不出问题，但是一旦你在 inference 时把 temperature 调高，它们就会 flood 回来，把正确答案挤掉。

**Fork position（岔路口）**

比如你刚写完 `def solve(arr):`，接下来可以用 quick sort、merge sort、insertion sort、built-in sort，四条完全不同的 algorithmic path。此时 distribution 是 spread out 的，每个 branch 都有 nontrivial 的概率。

这些 fork 是你**想要** explore 的地方。如果 temperature 太低，model 就只会走最高概率的那一条路，错过了其他 viable 的解法。

**矛盾来了**

Inference 时的 temperature $T_{\text{eval}}$ 是全局的，对每个 token position 一视同仁。

- 低温：锁位很安全（distractor 被压住），但岔路口没有 diversity（只会走一条路）
- 高温：岔路口可以 explore（多条路都有机会），但锁位的 distractor tail 复活（语法错误）

你没法用一个 global temperature 同时满足两者。这就是 **precision-exploration conflict**。

---

## SSD 怎么解决这个矛盾的

关键在于：SSD 不是在 inference 时调参数，而是在 **training 时改变 model 的 internal distribution**，让它变成一个在锁位自然 sharp、在岔路口自然 broad 的 distribution。

这样到了 inference 时，你就可以放心地用较高的 $T_{\text{eval}}$ 去 explore，因为锁位的 distractor tail 已经在 training 阶段被模型自己 "内部化" 地 suppress 掉了，不怕高温把它们复活。

具体怎么做到的？通过 $T_{\text{train}}$ 和 truncation（top-k/top-p）的组合。

**Truncation 的作用：Support Compression**

当你用 top-k=20 和 top-p=0.8 去采样 training data 时，相当于告诉 model：只在这个 retained set 里面待着，外面的 token 一概不要。

在 lock position，retained set 可能只有 1-2 个 token（因为 distribution 本来就很 sharp），所以 model 学到的就是：把所有 probability mass 集中到这一个 token 上，tail 彻底砍掉。

在 fork position，retained set 可能有 5-8 个 token（因为 distribution 本来就 flat），所以 model 学到的是：在这几个 viable branch 之间分配概率，但 tail 同样被砍掉。

**$T_{\text{train}}$ 的作用：Within-Support Reshaping**

当你用 $T_{\text{train}} = 1.6$（高于 1）去采样时，temperature 会 flatten distribution。在 retained set 内部，model 会把概率往均等方向推，形成 plateau。

这在 fork position 是好事（多条 branch 都有公平机会），在 lock position 影响不大（因为 retained set 本来就只有 1-2 个 token，flatten 也没什么可 flatten 的）。

**两个机制加在一起**

Lock 处：truncation 砍 tail + temperature 无影响 → 变成 sharp spike
Fork 处：truncation 砍 tail + temperature flatten head → 变成 broad plateau

这就是 Figure 5 里那个很直观的图：lock 变成 spike，fork 变成 plateau。

---

## 理论上怎么证明

Paper 在 Appendix B 做了一个精确的三项分解。SSD 的 loss 等价于：

$$\mathcal{L}_s(\theta) = \underbrace{-\log \text{KeptMass}_\theta(s)}_{\text{Support Compression}} + \underbrace{(1-T) H_{1/T}\big(p_\theta(\cdot \mid s, S_s)\big)}_{\text{Within-Support Reshaping}} + \underbrace{T \cdot \text{KL}\big(q_s \| p_{\theta,T}(\cdot \mid s, S_s)\big)}_{\text{Alignment}} + \text{const}$$

逐项解释：

- **$-\log \text{KeptMass}_\theta(s)$**：$\text{KeptMass}_\theta(s) = \sum_{v \in S_s} p_\theta(v \mid s)$，即 model 在 retained set $S_s$ 上的总概率质量。要 minimize loss，就要 maximize 这个 mass，也就是把概率往 retained set 里面推，tail 里面的 mass 就被挤掉了。这一项由 truncation 引入，是 lock 位置 tail suppression 的主力。

- **$(1-T) H_{1/T}$**：$H_{1/T}$ 是 Rényi entropy，order 为 $1/T$。当 $T > 1$ 时，$1/T < 1$，这是 sub-Shannon regime，对 diffuse tail 敏感。系数 $(1-T) < 0$，所以 minimize loss 等价于 **maximize** 这个 entropy。在 retained set 内部，model 会主动 flatten distribution。这一项由 $T_{\text{train}} \neq 1$ 引入，是 fork 位置 diversity 保留的主力。

- **$T \cdot \text{KL}$**：KL divergence 确保 flatten 不会跑偏太远，保持和 base model 的 alignment。

**为什么 naive self-training 不行？**

如果你用 $T=1$ 且不 truncation，第一项消失（因为 $S_s = \mathcal{V}$，$\text{KeptMass} = 1$），第二项消失（因为 $(1-1) = 0$），第三项也消失（因为 target 就是 model 自己）。Loss 的梯度期望为零，model 什么都不学。这就是 self-training 的 fixed point problem。

SSD 的 $T_{\text{train}}$ 和 truncation 同时打破了第一和第二项的惰性。

---

## Temperature Composition

还有一个很漂亮的结论。如果假设 student 完美拟合了 training target，那么在 inference 时：

$$q_{s,\tau}(v) \propto p_0(v \mid s)^{1/(T_{\text{train}} \cdot T_{\text{eval}})}$$

也就是说，在 retained set 内部，$T_{\text{train}}$ 和 $T_{\text{eval}}$ 是 **乘法关系**，构成 effective temperature $T_{\text{eff}} = T_{\text{train}} \cdot T_{\text{eval}}$。

实验验证了这一点：在没有 truncation 的情况下，performance 几乎只取决于 $T_{\text{eff}}$，在 $T_{\text{eff}} \approx 1.2$ 处达到 peak。加上 truncation 后，performance ceiling 进一步抬高，但对 $T_{\text{eff}}$ 的依赖变得不那么干净。

---

## 为什么调 inference 参数无法复现

这是 paper 里另一个 important point。你可能会想：既然 SSD 的效果是在 fork 处更 broad、在 lock 处更 sharp，那我在 inference 时对 base model 调 temperature 和 truncation 不就行了？

答案是不行。原因有二：

**Prefix Rigidity（前缀刚性）**

任何 decode-only 的 top-k/top-p policy 都必须保留 frozen model 原始 ranking 的一个 prefix。如果你想包含 rank-5 的一个 useful fork branch，你必须同时包含 rank 1-4 的所有 token，哪怕它们是 distractor。你没有 context-dependent 的能力去说"在这个 lock 位置只保留 rank 1，在那个 fork 位置保留 rank 1-5"。

**Power Rigidity（幂次刚性）**

Temperature scaling 对所有 logit 施加同一个 exponent $\alpha = 1/T_{\text{eval}}$。你没法在 lock 位置用大的 exponent（sharpen）同时在 fork 位置用小的 exponent（flatten）。

实验也证实了：对 base model 狂搜 $T_{\text{eval}} \in [0.6, 1.5]$，pass@1 只在 1.5-3.0pp 之间波动。而 SSD 带来 11.8pp 以上的提升。Gap 是 structural 的，不是 parametric 的。

---

## 极端实验：垃圾数据也能涨点

为了证明 gain 来自 distributional reshaping 而非 "学到了正确代码"，作者做了一个 stress test：

$T_{\text{train}} = 2.0$，关闭所有 truncation（top-k = $\infty$, top-p = 1.0）。

生成的 training data 约有 **62%** 包含无法提取的 code，甚至中英日多语言混杂的 gibberish：

```python
# COHERENT DEGRADED
# the number convinced lø be Fall
# Memorizzazione rethinknowledge Past
# found librore re inherently carry (
# Serv pull excitedtonspector franch danger
# money seasons domestic unicorn. complexity
```

就这种垃圾数据，fine-tune 之后依然涨了：pass@1 从 42.4% → 48.1% (+5.7pp)，pass@5 从 53.5% → 64.0% (+10.5pp)。

原因：虽然 Support Compression 项消失了（因为没有 truncation），但 Within-Support Reshaping 项依然 active（因为 $T = 2.0 \neq 1$）。Model 依然在学习如何在 full vocabulary 上 flatten distribution，这部分信号本身就足以提升 fork 位置的 exploration 能力。

当然，由于 training 时没有帮忙砍 tail，evaluation 时必须靠 $T_{\text{eval}}$ 的 truncation 来后处理，而且 viable region 变得更窄（$T_{\text{eval}} \in [0.8, 1.1]$）。这和有 truncation 的 headline 结果相比，gain 更小且更 fragile，但方向一致。

---

## 实验细节补充

**Training 超参**

- Megatron-LM on 8×B200 GPUs，MoE 用 EP=8
- AdamW，$\beta_1=0.9, \beta_2=0.95$，weight decay 0.1
- Peak LR $5 \times 10^{-6}$，cosine decay 到 $1 \times 10^{-6}$
- Global batch size 32，sequence length 65,536
- Instruct model 训 2,500 iterations，thinking model 训 300 iterations
- 每 250 / 50 iterations 存一个 checkpoint

**每个 model 的 decoding 配置**

| Model | $T_{\text{train}}$ | top-k | top-p | $T_{\text{eval}}$ | top-k | top-p |
|:---|:---|:---|:---|:---|:---|:---|
| Llama-3.1-8B-Instruct | 0.8 | 20 | 0.8 | 0.7 | 20 | 0.8 |
| Qwen3-4B-Instruct | 1.6 | 20 | 0.8 | 1.1 | 20 | 0.8 |
| Qwen3-4B-Thinking | 1.1 | 20 | 0.95 | 0.7 | 20 | 0.95 |
| Qwen3-30B-Instruct | 1.6 | 20 | 0.8 | 0.9 | 20 | 0.8 |
| Qwen3-30B-Thinking | 1.2 | 20 | 0.95 | 0.7 | 20 | 0.95 |

注意 thinking model 的 $T_{\text{train}}$ 普遍低于 instruct model，因为 thinking model 本身的 distribution 已经经过了 reasoning 训练的塑造，不需要太激进的 reshaping。

**Out-of-Domain Transfer**

只训了 competitive programming 数据，30B model 在 AIME、HumanEval、CruxEval、MMLU 上基本保持稳定（±2pp 以内）。4B 和 8B 的小 model 有一些 tradeoff，比如 Llama-8B 在 AIME 上掉了（因为它开始在数学题里输出 code block 而不是数字答案）。这说明大 model 的 representation 足够 robust，不会被 narrow domain 的 SFT 破坏 general capability。

---

## Toy Simulation 详解

Paper 还做了一个 V=16 token 的 toy FSM 来精确验证机制。FSM 结构是：

- Root state：最高概率 token 是错的（tok2 → FAIL），正确的两条 path 从 rank 2 和 rank 3 开始
- 每个 path 经过 1 个 fork + 3 个 lock 到达 PASS
- Fork state：correct token 在 rank 2，head 有 4 个 near-tied token
- Lock state：correct token 在 rank 1，占 75%，但 tail 有 15 个 distractor token 分走 25% 的 mass

SSD 之后：
- Lock collapse 成 2-token support，correct token 占 94.8%，tail 14 个 token 全部 prune
- Fork 保留 5-token support，correct token 从 16.9% 升到 plateau 级别，tail 11 个 token prune

最优 decoding temperature 从 $T^* = 0.63$（teacher）升到 $T^* = 2.09$（student），success probability 从 8.32% 升到 13.77%。

这个 toy 做到了精确可计算：因为 FSM 完全已知，success probability 可以写成 closed form：

$$P = [q_{\text{root}}(A) + q_{\text{root}}(B)] \cdot q_{\text{fork}}(\text{correct}) \cdot q_{\text{lock}}(\text{correct})^3$$

每个 $q$ 都是 post-truncation + post-temperature 的 operational probability，可以直接算。

---

## 我的一些联想

这篇 paper 的 result 在多个方向上都有启发：

**1. 和 RL 的关系**

GRPO/RLVR 通过 reward signal 来 break self-training fixed point，SSD 通过 target distribution shift 来 break。两者本质上都是在改变 "model 优化的 target"，只是一个是显式的 scalar reward，另一个是隐式的 distributional shaping。SSD 的结果暗示，至少在 code domain，reward signal 中很大一部分价值可能不在于告诉你 "哪个答案是对的"，而在于通过 reward 的 shaping 效应间接改变了 token distribution。如果这个假设成立，那 RL 的很多 gain 可能可以用更便宜的 distributional reshaping 来替代。

参考：DeepSeek-R1 (https://arxiv.org/abs/2501.12948), GRPO (https://arxiv.org/abs/2402.03300)

**2. 和 Speculative Decoding 的关系**

SSD 的 $T_{\text{eff}} = T_{\text{train}} \cdot T_{\text{eval}}$ 的乘法 composition 让人联想到 speculative decoding 中 draft model 和 target model 的 distribution composition。如果 SSD 的 student 可以看作是 teacher 在特定 $T_{\text{train}}$ 和 truncation 下的 " distilled approximation"，那 speculative decoding 的 accept/reject 机制是否也可以用类似 lens 来分析？

参考：Speculative Decoding (https://arxiv.org/abs/2211.17192)

**3. 和 Emergent Abilities 的关系**

SSD 的 gain 在 hard problems 上最大，这暗示 model 本来就 "知道" 怎么解这些 hard problems，只是 default decoding policy 无法 unlock 这部分 capability。这和 emergent abilities 的 threshold narrative 形成对比：capability 可能不是在某次 training 中突然出现，而是一直 latent 存在但被 decoding bottleneck 封印。SSD 只是解开封印的一种方式。

参考：Emergent Abilities (https://arxiv.org/abs/2206.07682)

**4. 和 Mode Collapse 的关系**

RLHF 众所周知的 mode collapse 问题，在 SSD 中没有出现（pass@5 提升大于 pass@1）。这可能是因为 SSD 的 reshaping 是 context-dependent 的（只在 fork 处 flatten），而 RLHF 的 KL penalty 是 global 的（对所有 token 一视同仁）。这暗示 RLHF 的 mode collapse 可能不是 RL 本身的问题，而是 global KL penalty 的设计缺陷。

参考：Mode Collapse in RLHF (https://arxiv.org/abs/2307.08632)

**5. 和 Information Bottleneck 的关系**

SSD 的三项目标分解中的 Support Compression 和 Within-Support Reshaping，和 Information Bottleneck (IB) 的 compression term 和 relevance term 有 deep connection。IB 的 Lagrangian form 是 $\min I(X;Z) - \beta I(Z;Y)$，SSD 的 loss 也可以理解成 $\min \text{Tail Mass} - (T-1) \cdot \text{Head Entropy}$。前者 compress 无关信息，后者 retain task-relevant 信息。

参考：Information Bottleneck (https://arxiv.org/abs/1610.03094)

**6. 和 Test-Time Compute Scaling 的关系**

SSD 提升了 pass@5，说明 model 能 generate 更多 diverse 的 viable solutions。这直接有利于 test-time compute scaling（如 majority voting, best-of-N selection）。如果 SSD + best-of-N 可以逼近 RLVR + verifier 的效果，那 post-training 的成本可以大幅下降。

参考：Inference Scaling Laws (https://arxiv.org/abs/2408.03314)

---

## 最后的 takeaway

这篇 paper 最让我震撼的点是：它证明了 LLM 的 capability 远未被充分利用。我们花了巨大精力去收集人类数据、训练 reward model、做复杂的 RL，结果 model 自己 generate 的 gibberish 上面做 SFT 就能涨 30%。

这暗示了一个更 deep 的可能性：pre-training 阶段注入的 information 量远大于 post-training 阶段能 unlock 的量。Post-training 的核心任务可能不是 "教 model 新东西"，而是 "帮 model 找到更好的 internal distribution 来表达它已经知道的东西"。SSD 只是找到的一种很 crude 的方式，可能还有大量类似的 simple post-training trick 等着被发现。

Paper 的 GitHub repo: https://github.com/apple/ml-ssd

---

Andrej, 这篇 Apple 的 paper 揭示了一个极其 elegant 且反直觉的 phenomenon: 强大的 LLM 在 code generation 领域存在未被充分利用的 latent capability。我们完全不需要 verifier、teacher model 或复杂的 RLHF/GRPO 流程，仅仅通过让 model 采样自己的 raw output 并进行 standard SFT，就能大幅提升 pass@1。

这里的核心洞察在于揭示了 LLM decoding 过程中的 **Precision-Exploration Conflict**，并且 SSD 提供了一种 context-dependent 的 distribution reshaping 机制来 alleviate 这个 conflict。

以下我将从 core intuition、mathematical formulation、experimental data 以及 related associations 层面为你进行极度细节的拆解，旨在 build your intuition。

---

### 1. Core Intuition: Precision-Exploration Conflict

在 code generation 中，token 序列的生成存在两种截然不同的 context state：

*   **Lock positions (Precision-bound)**: 在语法和语义高度确定的 context 下（例如 `if n == ` 之后），model 应该输出一个 specific value。此时 distribution 是 sharply peaked 的，但是在 long tail 上仍然存在非零概率的 distractor tokens。
*   **Fork positions (Exploration-bound)**: 在算法路径选择的 context 下（例如 function body 的开头），存在多种 plausible continuations（比如 quick sort vs merge sort vs built-in sort），每种 continuation 都会导致 fundamentally different downstream trajectories。此时 distribution 是 spread out 的。

这带来了一个 dilemma：
如果我们在 inference 阶段使用单一的 global decoding temperature $T_{eval}$：
*   **Low $T_{eval}$**: 会 sharpen lock 的 peak，suppress distractor tail，但是会 starve fork 的 diversity，导致 model 无法 explore 不同的 solution branches。
*   **High $T_{eval}$**: 会 flatten distribution，在 fork 处 restore exploration，但是会 revive lock 处的 distractor tail，导致 syntax error。

任何固定的 decoding 配置都是在这两端之间的 compromise。SSD 的 magic 在于，它通过 training-time 的 temperature shift 和 truncation，直接修改了 model 的 weight，从而重塑了 token distribution，使其在 lock 处变得更 sharp（suppressing tail），在 fork 处保留有用的 head entropy（preserving diversity）。这种 context-adaptive 的 behavior 是单纯调 decoding 参数无法做到的。

---

### 2. Methodology: Embarrassingly Simple Self-Distillation (SSD)

SSD 的 pipeline 简单到令人发指，主要分为三步：

**Step 1: Data Synthesis (Sampling)**
给定一个 frozen pre-trained LLM $p_\theta$ 和一组 prompts $X$，我们使用特定的 training-time temperature $T_{train}$ 和 truncation configuration $\rho_{train}$（即 top-k 和 top-p）来采样 solutions：

$$
y \sim \mathsf{Decode}_{T_{\mathsf{train}}, \rho_{\mathsf{train}}} \big[ p_\theta \big( \cdot \mid x \big) \big]
$$

这里的关键在于 **no verification**。没有 execution，没有 test cases，没有 correctness filtering。生成的 raw outputs 直接构成 dataset $\mathcal{D}_{\mathsf{SSD}}$。实践中 $N=1$（每个 prompt 只采一个 sample）就足够了。

**Step 2: Training (SFT)**
使用标准的 cross-entropy loss 在 $\mathcal{D}_{\mathsf{SSD}}$ 上进行 fine-tune：

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(x,y) \sim \mathcal{D}_{\mathsf{SSD}}} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})
$$

**Step 3: Inference**
Fine-tuned model $p_{\theta^*}$ 使用 evaluation-time 的 decoding configuration $(T_{\mathsf{eval}}, \rho_{\mathsf{eval}})$ 进行推理：

$$
\hat{y} \sim \mathsf{Decode}_{T_{\mathsf{eval}}, \rho_{\mathsf{eval}}} \big[ p_{\theta^*} \big( \cdot \mid x \big) \big]
$$

#### Decoding Pipeline 细节 (vLLM Implementation)
为了精确复现，Paper 在 Appendix A 中给出了 vLLM v0.11.0 的 exact 4-step pipeline：
1.  **Temperature Scaling**: Logits $z_v$ 全部除以 $T$ ($z_v \leftarrow z_v / T$)。这在 softmax 之后等价于 $p(v)^{1/T}$。
2.  **Top-k Filtering**: Logits 升序排列，保留 k 个最大的，其余设为 $-\infty$。
3.  **Top-p Filtering**: 在 top-k survivors 中计算 softmax，从最小概率开始累加，移除累积质量低于 $1 - p$ 的 tokens，确保至少保留 1 个 token。
4.  **Gumbel-max Sampling**: 不使用 `torch.multinomial` (避免 CPU-GPU sync)，而是采样 $Exp(1)$ 噪声 $q$，取 $\arg\max(p_v / q)$。

---

### 3. Mathematical Formulation: 为什么 SSD 会有用？

这是这篇 paper 最 beautiful 的部分。为什么训练自己生成的、未经核验的 gibberish 能提升 model 性能？

如果仅仅是 $T=1$ 且无 truncation 的 naive self-training，这是一个 fixed point（期望梯度为 0，因为 $\mathbb{E}_{v \sim p_\theta}[\nabla_\theta \log p_\theta(v)] = \nabla_\theta \sum p_\theta = 0$）。SSD 打破这个 fixed point 的原因在于 $T_{train}$ 和 $\rho_{train}$ **修改了 target distribution**。

Paper 在 Appendix B 推导了 SSD 损失函数的精确三项分解。

令 $s = (x, y_{<t})$ 为 context。$S_s$ 为经过 $T_{train}$ 和 truncation 后保留的 token set (Retained Support)。定义 model 分配给 $S_s$ 的总质量为 $\mathsf{KeptMass}_\theta(s) = \sum_{v \in S_s} p_\theta(v \mid s)$。定义 $p_{\theta, T}(\cdot \mid S)$ 为在 $S$ 上 tempered 后的 conditional distribution。

SSD 的 loss 可以被 exact 分解为：

$$
\mathcal{L}_s(\theta) = \underbrace{- \log \mathsf{KeptMass}_\theta(s)}_{\text{Support Compression}} + \underbrace{(1 - T) H_{1/T} \big( p_\theta(\cdot \mid s, S_s) \big)}_{\text{Within-Support Reshaping}} + \underbrace{T \cdot \mathrm{KL} \big( q_s \| p_{\theta, T}(\cdot \mid s, S_s) \big)}_{\text{Alignment to Base Model}} + \text{const}
$$

**变量与上标下标解释：**
*   $T$: 训练时的温度 $T_{\mathsf{train}}$。
*   $S_s$: Context $s$ 下，经过 temperature scaling 和 top-k/top-p 筛选后存活的 token 集合。
*   $\mathsf{KeptMass}_\theta(s)$: 参数为 $\theta$ 的 model 在 context $s$ 下分配给存活集合 $S_s$ 的总概率质量。
*   $H_{1/T}(\cdot)$: Order 为 $1/T$ 的 Rényi entropy。定义为 $\frac{1}{1 - \alpha} \log \sum \pi(v)^\alpha$，此处 $\alpha = 1/T$。当 $T > 1$ 时，$\alpha < 1$，处于 sub-Shannon regime，对 diffuse tails 极其敏感。
*   $q_s$: Teacher (即 base model) 经过 temperature 和 truncation 后在 $S_s$ 上的 target distribution。

**每一项的物理意义：**

1.  **Support Compression ($-\log \mathsf{KeptMass}_\theta$)**: 这是由 truncation 引入的 gate。因为 target $q_s$ 在 $S_s$ 外概率为 0，所以 model 必须把所有 mass 推进 $S_s$ 中。在 logit 空间，任何在 $S_s$ 外的 token $v$，其梯度都是 $+p_\theta(v \mid s)$，这是一个恒正的向下推力。这直接 suppresses 了 distractor tail，且永远不会消失（除非 logit 趋于 $-\infty$）。
2.  **Within-Support Reshaping ($(1 - T) H_{1/T}$)**: 在 $S_s$ 内部，当 $T > 1$ 时，$(1 - T) < 0$，所以 minimizing loss 等价于 **maximizing** Rényi entropy $H_{1/T}$。这意味着在 retained set 内部，model 会 flatten distribution，这在 Fork positions 给了 exploration 极大的空间。
3.  **Alignment ($T \cdot \mathrm{KL}$)**: 确保这种 flattening 不会偏离 base model 的 fundamental preferences 太远。

#### 锁与岔路口的 Asymmetry
这个三项目标解释了为什么 SSD 会有 context-dependent 的效果：
*   **At Locks**: $S_s$ 很小（只有 1-2 个 token）。Support Compression 项主导，因为 $H_{1/T} \le \log|S_s| \approx 0$，Reshaping 项没有发挥空间。结果是 distractor tail 被强力切除，lock 变得极其 robust。
*   **At Forks**: $S_s$ 较大（包含多个 plausible branches）。Support Compression 依然 active，但 Reshaping 项获得了巨大的发挥空间。$(1-T) < 0$ 导致 model 在这些 branches 之间平滑概率，形成 plateau，从而保留了 exploration 所需的 diversity。

#### Temperature Composition
在 local ideal-fit approximation（假设 student 完美拟合了 $q_s$）下，evaluation-time 的 distribution 表现为：

$$
q_{s, \tau}(v) = \frac{q_s(v)^{1/\tau}}{\sum u_s(u)^{1/\tau}} \propto p_0(v \mid s)^{1 / (T_{\mathsf{train}} \cdot \tau)}
$$

这里 $\tau$ 是 $T_{\mathsf{eval}}$。可以看出，在 retained support 内，$T_{\mathsf{train}}$ 和 $T_{\mathsf{eval}}$ 是 **multiplicatively composable** 的，构成了 effective temperature $T_{\mathsf{eff}} = T_{\mathsf{train}} \cdot T_{\mathsf{eval}}$。这就解释了为什么 Figure 3 中，性能的 peak 集中在 $T_{\mathsf{eff}} \approx 1.2$ 附近的一条对角线上。

---

### 4. Experimental Data Analysis

Table 2 展示了在 LiveCodeBench v6 上的惊艳表现：

| Model | Metric | Base | +SSD | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-30B-Instruct | pass@1 (All) | 42.4 | 55.3 | +12.9 pp |
| Qwen3-30B-Instruct | pass@1 (Hard) | 18.3 | 33.6 | +15.3 pp |
| Qwen3-30B-Instruct | pass@5 (Hard) | 31.1 | 54.1 | +23.0 pp |
| Qwen3-4B-Instruct | pass@1 (All) | 34.0 | 41.5 | +7.5 pp |
| Llama-3.1-8B-Instruct | pass@1 (All) | 12.7 | 16.2 | +3.5 pp |

**关键观察：**
1.  **Hard problems 获益最大**: Qwen3-30B-Instruct 在 Hard split 上 pass@1 提升了 +15.3pp，而 Easy 只提升了 +6.5pp。这非常反直觉，通常 SFT 容易在 easy 上 overfit 而在 hard 上 collapse。
2.  **Diversity 保留**: pass@5 的提升往往大于 pass@1。对于 Qwen3-30B-Instruct，pass@5 提升了 +18.1pp (All) 和 +23.0pp (Hard)。这说明 SSD 并非仅仅 sharpen 单一的 dominant mode，它同时 preserve 甚至 enhance 了 multi-modal 的 exploration 能力。
3.  **Global Decoding 调参无法复现**: Figure 2 显示，对 base model 狂搜 $T_{eval}$ 和 $\rho_{eval}$，pass@1 的变化幅度只有 1.5-3.0pp。而 SSD 带来的提升在 +11.8pp 以上。证明了 SSD 修改的是 model 的 internal representation，打破了 Prefix Rigidity 和 Power Rigidity（见 Appendix B.5，decode-only policy 必须在原 logit ranking 的前缀上进行 global power transform，无法做到 contextual 的 tail suppression）。

---

### 5. 极度反直觉的 Stress Test: Bad Data, Good Results

为了验证 "gain 完全来自于 distributional reshaping 而非 training on correct code"，作者做了一个极端实验：

设定 $T_{\mathsf{train}} = 2.0$，关闭所有 truncation（top-k = $\infty$, top-p = 1.0）。
结果生成的数据约 62% 包含无法提取的 code，甚至夹杂多语言乱码。
即使在这样的 "garbage" 数据上做 SFT，Qwen3-30B-Instruct 依然提升到了 48.1% pass@1 (+5.7pp) 和 64.0% pass@5 (+10.5pp)！

**为什么？** 因为在无 truncation 的情况下，第一项 Support Compression 消失了，但是第二项 Within-Support Reshaping ($(1-T) H_{1/T}$) 依然 active。Model 依然在学习如何 smooth 其 internal distribution。不过，由于 training 时没有 truncation 帮忙清理 distractor tail，model 在 evaluation 时极度依赖 $T_{\mathsf{eval}}$ 的 truncation 来进行后处理。这个实验完美佐证了三项目标分解的物理意义。

---

### 6. 相关联想与 Web Links 参考

这篇 paper 让我联想到多个领域的前沿工作，为你提供以下 references 来 build a broader intuition:

1.  **On-Policy Distillation vs RLVR**: 传统的 RLHF/GRPO (如 DeepSeek-R1) 依赖 verifier 提供的 scalar reward 来 break self-training fixed point。SSD 展示了一种纯 unsupervised 的途径。这与近期讨论的 "Learning to Reason without External Rewards" 异曲同工。
    *   DeepSeek-R1: https://arxiv.org/abs/2501.12948
    *   TTRL (Test-Time Reinforcement Learning): https://arxiv.org/abs/2504.16084
    *   Unsupervised RLVR: https://arxiv.org/abs/2603.08660

2.  **Entropy & Information Bottleneck**: Rényi entropy $H_{1/T}$ 的出现非常 profound。在 statistical mechanics 中，$-T \log \sum p^{1/T}$ 正是 free energy 的表现形式。SSD 实际上是在最小化 model 在 retained support 上的 free energy，这与 Information Bottleneck Theory 中压缩无关信息、保留 task-relevant 信息的过程高度一致。
    *   Rényi entropy original paper: https://link.springer.com/chapter/10.1007/978-3-642-99589-1_7
    *   Truncation Sampling as Desmoothing (Hewitt et al.): https://arxiv.org/abs/2210.15191

3.  **Critical Tokens / Forking Paths**: Paper 提到了 "fork positions" 概念，这与 Ari Holtzman 等人的 Forking Paths 理论以及近期 Wang 等人关于 "High-Entropy Minority Tokens Drive Effective RL" 的研究紧密相连。SSD 提供了一种无需 RL 即可增强 fork position 探索能力的方案。
    *   Forking Paths in Neural Text Generation: https://openreview.net/forum?id=ewQpwoI2m7
    *   Beyond the 80/20 Rule: High-Entropy Minority Tokens: https://arxiv.org/abs/2506.04223

4.  **VLLM Decoding Implementation**: 了解底层 sampler 实现对于复现至关重要。
    *   VLLM GitHub repo: https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py

总结来说，SSD 通过极简的 SFT 形式，在 latent space 内实现了一种基于 Rényi entropy 的 context-dependent distribution shaping。它告诉我们，当前 LLM 的 capability 远未被其 default decoding policy 充分挖掘，通过 self-distillation 我们可以 unlock 这部分被 precision-exploration conflict 封印的潜能。
