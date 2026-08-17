---
source_pdf: The Lessons of Developing Process Reward Models.pdf
paper_sha256: 07c71a7f12b7b15ad012624d755b7b9c053b0ff77c1c9b4c5cb44032034a13b3
processed_at: '2026-08-12T14:25:33-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用人话讲：

**这论文在说啥？**

大家想训练一个"数学过程检查员"(PRM)，专门检查推理每一步对不对。现在主流做法是：从某一步往下采样8次，看能不能蒙对答案，蒙对了就说这步对，蒙不对就说这步错。

这论文发现这个方法**根本不靠谱**。因为模型可能瞎蒙对了答案但过程是胡扯的，也可能过程对但后续发散了没蒙对。用这种充满噪声的数据训出来的PRM，表面上BoN分数还行，真要它定位具体哪步错了，一塌糊涂。

**怎么解决？**

简单一招：让72B大模型当裁判重新标一遍，两套标注结果取交集——只有两边都说"这步错了"才保留这条数据。数据砍掉60%，效果反而更好。说明数据质量远比数据量重要。

**BoN评价有啥坑？**

policy model经常"答案对但过程胡编"。BoN只看答案对不对，所以PRM越严格越吃亏——它正确地拒掉了过程有问题的response，但按BoN标准算它"选错了"。反过来，一个啥都通过的傻瓜PRM，BoN分数反而虚高。而且发现很多PRM实际上退化成了只看最后一步答案的ORM，完全背离了process监督的初衷。

**一句话总结：** 用MC采样标数据训PRM是在用value function的逻辑训deterministic verifier，方法不对；BoN作为唯一评价标准会被"假对"response骗到。两者都有系统性问题，需要靠consensus filtering和step-level benchmark来修正。

---

Hi Andrej! 这篇 Qwen Team 的 paper 在 PRM (Process Reward Model) 的 data construction 和 evaluation 方法论上挖得非常深，可以说是给目前 LLM 数学推理领域的 reward modeling 敲了一记警钟。这篇文章的很多 insight 直击 RLHF 和 verifier 训练的痛点，我来给你详细拆解一下其中的核心机制和技术细节，帮你 build 更强的 intuition。

### 1. PRM vs Value Model 的本质分野

要理解这篇 paper，首先要厘清 PRM 与 Value Model 在数学推理中的概念混淆。

*   **PRM (Process Reward Model)**: 作为 deterministic evaluator，评估当前 reasoning step $s_t$ 的绝对正确性。它的输出 $r(s_t) \in [0, 1]$ 代表“这一步本身是对是错”。
*   **Value Model**: 作为 predictive estimator，估计从当前 step $s_t$ 出发，未来到达正确 final answer 的概率。这在 RL 中对应 $V^\pi(s_t) = \mathbb{E}_{\pi}[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} | s_t = s]$。

目前主流的 data synthesis 方法（如 Math-Shepherd）采用 Monte Carlo (MC) estimation 来构造 PRM 训练数据。其逻辑是：从 step $s_t$ 开始，用 completion model 采样 $K$ 次，如果有 $K_c$ 次得到正确答案，则：
*   Hard label: $y_t = \mathbb{I}(K_c > 0)$
*   Soft label: $y_t = \frac{K_c}{K}$

这篇 paper 戳破了一个致命的直觉误区：**用 MC estimation 训练出来的模型，本质上学到的是 Value Function，而他们想把它当作 PRM 来用**。这导致了严重的性能崩塌。

### 2. MC Estimation 的 Noise 来源与公式解析

为什么 MC estimation 会引入大量 noise？核心在于 completion model 的不可靠性导致了 credit assignment 的扭曲。文章指出了两种极端情况：
1.  **Correct steps leading to incorrect answers**: 一步推导完全正确，但 completion model 在后续采样中发散了，导致 $K_c = 0$，于是正确的 step 被打上了 $y_t = 0$ 的负标签。
2.  **Incorrect steps leading to correct answers**: 一步推导逻辑断裂或计算错误，但 completion model “瞎猫碰上死耗子”或者通过后续的胡言乱语蒙对了答案，导致 $K_c > 0$，错误的 step 被打上了 $y_t = 1$ 的正标签。

为了 mitigate 这个问题，作者提出了 **Consensus Filtering Mechanism**。这个机制非常简单粗暴但有效：用 LLM-as-a-judge (这里是 Qwen2.5-72B-Instruct) 对同样的 trajectory 重新打标，只有 MC estimation 和 LLM-as-a-judge 对错误 step 的定位达成共识时，该 data instance 才会被保留。
用集合语言表达，假设 $D_{MC}$ 是 MC 标注的数据集，$D_{LLM}$ 是 Judge 标注的数据集，则过滤后的数据集 $D_{filtered} = D_{MC} \cap D_{LLM}$。

实验数据表明，经过 consensus filtering 后，3 million 的数据量缩减到了 1.5 million（保留了约 40%），但在 PROCESSBENCH 上的表现却大幅跃升，甚至超越了使用全量数据的 LLM-as-a-judge。这证明了**高精度的 signal 比 scale of data 更重要**。

### 3. Hard Label vs Soft Label 的实验深度剖析

作者在 Consensus Filtering 之前和之后，分别对比了 Soft label 和 Hard label 的表现。
*   **Before filtering**: Soft 和 Hard 差异不大。因为 data 本身 noise 极大，Soft label 里的分数 $y_t \in (0, 1)$ 全是方差和噪声，掩盖了它与 Hard label 的区别。
*   **After filtering**: Hard label 显著超越 Soft label。

直觉上，Soft label 保留了更多的信息（比如 $K_c=3/8$ 对应 0.375 的概率）。但这篇 paper 给出了反直觉的结论。解释如下：
PRM 作为 deterministic verifier，其目标空间是二值的。Soft label 强行将 future 的 probability 注入到了 current step 的 correctness 里。如果一个 step 本身完全正确，但被赋予了 0.375 的 soft label，模型在学习 MSE loss $\mathcal{L}_{MSE} = (y_i - \hat{y}_i)^2$ 时，会降低对 positive sample 的判别力。

此外，作者探究了 threshold 的选择。公式 $y_t = \mathbb{I}(\frac{K_c}{K} > \tau)$ 中，随着 $\tau$ 从 $1/8$ 增加到 $7/8$，性能在 Best-of-8 和 PROCESSBENCH 上均下降。最佳策略是 $\tau = 0$，即“只要有一条路走通，这步就算对”。这其实是在近似 logic 中的 $A \lor B \lor C \dots$ 逻辑，极大地容忍了 completion model 的发散性。

### 4. BoN Evaluation 的结构性 Bias 与 Process-to-Outcome Shift

这篇文章对 Best-of-N (BoN) 评估的批判极为犀利。传统的 BoN metric $prm@N$，是 N 个 sampling 里选 PRM 打分最高的那个 response。整体 response 的 score 计算方式通常有两种：
*   $S_{prod} = \prod_{i=1}^{N_{step}} \hat{y}_i$ (所有 step 分数连乘)
*   $S_{min} = \min_{i \in \{1..N_{step}\}} \hat{y}_i$ (取所有 step 中的最小值)

这两种方式的致命缺陷在于：**Policy model 经常生成出答案正确但过程胡编乱造的 response**。由于 BoN 的 ground truth 只看 final answer 对不对（$Acc_{final\_answer}$），这就产生了 Misalignment：
如果 PRM 真的很强，能识别出过程中的错误，它就会给这种“假对”的 response 打低分，从而错过这个 response，导致 BoN 分数下降。相反，如果 PRM 很弱，容忍了这种“假对”的 response，BoN 分数反而会虚高。

这就解释了 Table 3 和 Table 4 中诡异的现象：
*   在 BoN 上，MC estimation 数据训练的 PRM 表现最好（Avg. 65.9）。
*   在 PROCESSBENCH（要求精准定位错误 step）上，MC estimation 表现最差（Avg. F1 40.1），而 Human annotation 表现最好（Avg. F1 56.5）。

作者还发现了一个现象：**Process-to-Outcome Shift**。在分析多个开源 PRM 时，发现它们给出 minimum score 的 step，超过 40% 集中在 final answer step。这意味着，这些号称是 PRM 的模型，在实际运作中退化成了 ORM (Outcome Reward Model)。BoN 优化目标导向了这种退化。

### 5. Scoring Strategy 的直觉重建

这部分非常有 Karpathy 风格。对于不同 data 构造方式训练出的 PRM，最优的 scoring strategy 完全不同：
*   **Human Annotation / LLM-as-a-judge**: 因为每一步的 label 是 deterministic 的，代表本步正确性，所以 $S_{prod}$ 或 $S_{min}$ 最合理。
*   **MC estimation**: 因为每个 step 的 label 是 future-reaching probability，这些 estimated probabilities 之间是高度依赖的（conditional probability chain），所以既不能连乘（会过度惩罚概率衰减），也不能取 min。最合理的是取 **Last score** $S_{last} = \hat{y}_{N_{step}}$。因为最后一个 step（即给出最终答案前一步）的 MC probability，天然整合了整条 solution trajectory 的成功概率。

### 6. Architecture & Training Details

模型架构上，Qwen2.5-Math-7B/72B-Instruct 的 LM head (next token prediction 层) 被替换为一个 scalar-value head。这个 head 由 two linear layers 组成。
Loss function 的设计：
*   对于 Hard label (二分类任务): 采用 Cross-Entropy (CE) loss。
    $$ \mathcal{L}_{CE} = - \frac{1}{M} \sum_{j=1}^{M} [y_j \log(\hat{y}_j) + (1-y_j) \log(1-\hat{y}_j)] $$
    其中 $M$ 是一个 batch 里的 step 总数，$y_j \in \{0, 1\}$ 是 ground truth，$\hat{y}_j$ 是 scalar head 经过 sigmoid 后的输出。
*   对于 Soft label (回归任务): 采用 Mean Squared Error (MSE) loss。
    $$ \mathcal{L}_{MSE} = \frac{1}{M} \sum_{j=1}^{M} (y_j - \hat{y}_j)^2 $$

### 7. Extensive Intuitions & Hallucinations (扩展联想)

既然 Andrej 你喜欢更广阔的联想，我从这篇 paper 出发，谈谈几个 deep thoughts：

**A. The "Plausible Reasoning" Paradox (System 2 的幻觉陷阱)**
这篇 paper 揭示了 Policy Model 生成“正确答案但 flawed process”的现象（图6）。这在 RL 圈子里类似于 Reward Hacking 的变种。在 math reasoning 中，模型可以通过 pattern matching 直接猜出 final answer，然后倒推捏造一套看似合理的 derivation。这也就是为什么仅仅依赖 final answer 做强化学习（比如 RL with verifiable rewards, RLVR）是有风险的。模型可能并没有学会真正的 deductive reasoning，而是学会了如何“自圆其说”。o1 类模型的 System 2 thinking 如果没有 process supervision，可能会陷入高维空间的 hallucination local optimum。

**B. AlphaGo 的 Value Network 梦魇重现**
MC estimation 训练 PRM 的失败，简直就是 AlphaGo 时期训练 Value Network 踩过的坑的翻版。当年 DeepMind 也是用 policy rollout 来估计局面胜率，结果发现 Policy 本身的偏差会严重污染 Value Network 的估计。这篇 paper 的 Consensus Filtering (MC $\cap$ LLM-as-judge) 有点类似于 AlphaGo 中用 Tree Search 来修正 Value network 的直觉。未来的 PRM 训练，可能会回归到更复杂的 MCTS 框架中，结合 LLM 的 self-play 来做更精确的 credit assignment。

**C. ORM 的文艺复兴？**
在 Table 7 中有一个极其有趣的数据：Qwen2.5-Math-RM-72B (一个纯粹的 ORM，给整个 response 打一个分) 在 PROCESSBENCH 上的 Avg F1 达到了 38.9，甚至超过了绝大多数专门设计的 7B+ PRM。这个直觉很诡异：既然我们要 process supervision，为什么 ORM 也能找错？其实直觉上，当一个 response 中途出错时，后续的 trajectory 会呈现出一种“语义上的断裂”或者“分布偏移”。ORM 在整段 response 上做 pooled attention 时，能够敏锐捕捉到这种 logit 分布的不连续性。这说明 PRM 的 scalar head 可能破坏了 LLM 原本的 contextual representation，未来或许需要 hybrid reward (ORM global pooling + PRM local projection) 的结合。

**D. PRM 与 Q* / Tree Search 的耦合困境**
在 Appendix A 中，作者尝试了 PRM guided greedy search。每一步生成 8 个 candidates，用 PRM 选最高分继续 expand。但结果（Table 8）表明，这种简单的 greedy search 并没有比 ORM 的 BoN 好多少。这里的直觉是：PRM 只告诉你“当前这步好不好”，但在 tree search 中，我们常常需要走入一个“看似平庸但实则是唯一正解”的 step。PRM 缺乏 forward-looking value，无法指导 search 算法突破局部最优。这呼应了文章中“PRM 与 Value Model 的本质区别”。如果要把 PRM 用在 Search (比如 Q* 或 MCTS) 上，我们必须显式地建模和利用 PRM 分数随 trajectory 深度的衰减特性，或者强行引入 dual-network (PRM + Value Model) 共同指导节点扩展。

**Web Links for Reference:**
*   PROCESSBENCH paper: https://arxiv.org/abs/2412.06559
*   Math-Shepherd paper: https://aclanthology.org/2024.acl-long.510
*   Let's Verify Step by Step (PRM800K, OpenAI): https://arxiv.org/abs/2305.20050
*   Qwen2.5-Math Technical Report: https://arxiv.org/abs/2409.12122
*   RLHFlow (Generative PRM): https://github.com/RLHFlow/RLHF-Reward-Modeling
*   Qwen2.5-Math-PRM-72B Model on HuggingFace: https://hf.co/Qwen/Qwen2.5-Math-PRM-72B
