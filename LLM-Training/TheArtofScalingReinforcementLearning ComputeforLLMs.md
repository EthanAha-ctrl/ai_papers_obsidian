---
source_pdf: TheArtofScalingReinforcementLearning ComputeforLLMs.pdf
paper_sha256: ddb5eb8f43fa3b3dbb0f7fa14cdf16eb170cee919c0713c7e38aabfa7647c441
processed_at: '2026-08-12T15:08:42-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，要把这篇 paper 用人话讲透，我们可以把它当成是 **RL for LLMs 领域的 Chinchilla 论文**。Pre-training 时代大家也是瞎炼，直到 Kaplan 和 Hoffmann 给出了 power law，大家才找到方向。这篇 paper 做的就是同一件事：给 RL training 找一个能预测的 scaling law，并且告诉你到底该怎么调参。

我帮你把这篇 paper 的骨架和血肉剥开，重点讲讲里面的 intuition 和那些只有炼丹师才会关心的血泪经验。

### 1. 核心直觉：为什么 RL Scaling 是一条 Sigmoid 曲线？

Pre-training 的 loss 是 unbounded 的，只要给 compute，loss 就能顺着 power law 往下降。RL training 的 metric（通常是 pass rate 或 reward）是被死死钉在 $[0, 1]$ 之间的。你不可能有 120% 的 pass rate。

因为 metric 有天花板，作者发现用 sigmoid 曲线来拟合 RL 的 compute-performance 简直是完美契合。这就是论文的核心公式 Eq. (1)：

$$ R_C - R_0 = \frac{A - R_0}{1 + (C_{mid}/C)^B} $$

**人话翻译变量：**
*   $R_C$：你花了 $C$ 这么多 GPU hours 时，模型在 validation set 上的平均 reward。
*   $R_0$：刚开始训练（$C=0$）时模型的本底水平。下标 $0$ 就是 initial 的意思。
*   $A$：**Asymptote（天花板）**。这是整篇论文最核心的指标。无论你砸多少 compute，这个 recipe 最终能达到的最高 reward 极限就是 $A$。
*   $C_{mid}$：**拐点**。达到一半 reward gain $(A-R_0)/2$ 所需要的 compute。$C_{mid}$ 越小，说明配方起效越快。
*   $B$：**Efficiency（效率）**。曲线的陡峭程度。$B$ 越大，说明你从拐点冲向天花板的速度越快，compute 用得越狠。

**Build Intuition:**
你可以把 RL 训练想象成一个人学解魔方。
一开始（Low compute），他在死记硬背公式，对应 RL 里模型在学习 `...` 这种输出格式，这时候 reward 几乎是线性上升的。
到了中期（Mid compute），他开始理解魔方的底层逻辑，探索不同的解法，reward 快速飙升。
到了后期（High compute），简单的 cases 都秒解了，剩下的都是极其刁钻的盲拧 cases，进步极其缓慢，最后趋近于他智力和手速的物理极限 $A$。

作者在 Appendix A.4 里证明了一个很优美的数学性质：当 compute $C$ 远大于拐点 $C_{mid}$ 时，sigmoid 公式会自动退化成 pre-training 的 power law 公式 $R_C \approx A - D/C^B$。所以 sigmoid 是 power law 的 bounded 推广。

### 2. ScaleRL 配方大揭秘：把玄学变成工程

这篇 paper 花了 400,000 GPU-hours 就是为了搞清楚一件事：到底哪些 trick 能抬高 $A$，哪些只能改变 $B$？
他们最终拼凑出的 ScaleRL recipe，每一个组件都有明确的工程意义。

#### 2.1 PipelineRL-8：消灭 GPU 闲置时间
传统的 PPO-off-policy 是“阻塞式”的：Generator 攒够一个 batch 的 rollouts，Trainer 才开始训；Trainer 训的时候，Generator 在旁边抽烟等。
PipelineRL 改成了“流水线式”：Generator 一边吐 rollout，Trainer 一边吃。Trainer 刚算完一个 gradient step 更新了 weights，立马把新 weights 推给 Generator。哪怕 Generator 之前的 KV cache 是旧的，它也能用新 weights 继续生成后面的 tokens。

**Intuition:** 这让训练极度逼近理想的 on-policy 状态。分布偏移小了，天花板 $A$ 稍微涨了一点，但更重要的是 efficiency $B$ 大幅提升，因为 GPU 不再有空转的间隙。

#### 2.2 CISPO Loss：对超参数彻底脱敏
GRPO 和 DAPO 都是对 importance sampling ratio $\rho$ 直接做 clip。这就导致一个非常头疼的问题：$\epsilon_{max}$ 这个 clip 阈值极度敏感。论文在 Appendix A.17 里展示了，DAPO 的 $\epsilon_{max}$ 从 0.2 改到 0.28，天花板 $A$ 就会剧烈波动。

ScaleRL 采用了 CISPO loss：
$$ \mathcal{I}_{CISPO}(\theta) = \mathbb{E} \left[ \frac{1}{T} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \text{sg}(\min(\rho_{i,t}, \epsilon_{max})) \hat{A}_i \log \pi_{train}^\theta(y_{i,t}) \right] $$

**人话翻译：**
$\text{sg}$ 是 stop-gradient。CISPO 的精髓在于：它用 clipped 的 ratio $\rho_{i,t}$ 去乘以 advantage $\hat{A}_i$，但因为加了 stop-gradient，梯度在反向传播时完全忽略了前面的 clip 操作，直接流到 $\log \pi_{train}$ 里面去。
这就像是说：“我承认你的旧策略和新策略偏差很大（ratio 大），但我只用这个 ratio 来**缩放**你这次的梯度大小，我**不去修改**你更新的方向。”
这使得 $\epsilon_{max}$ 设成 4 还是 8 都无所谓，训练稳如老狗。

#### 2.3 FP32 Precision at LM Head：被忽视的数值刺客
这是全篇最让人拍案叫绝的发现。
Generator 通常用高度优化的 inference kernel（比如 vLLM），而 Trainer 用 training kernel（比如 FSDP+FlashAttention）。这两套代码在算 logits 时的浮点数截断规则不同。
在 supervised learning 里，这点误差无所谓。但在 RL 里，我们要计算 IS ratio $\rho_{i,t} = \pi_{train} / \pi_{gen}$。如果 purely on-policy，$\rho$ 应该恒等于 1。但因为 kernel 不同，算出来的 $\rho$ 可能是 1.0001 或者 0.9998。
这个微小噪声在 softmax 之前的 LM head（词汇表通常好几万）被极度放大。模型把大量的梯度用来去拟合这种“本不存在”的分布偏移了。

**Intuition:** 强制把 LM head 的计算升级到 FP32，把 $\rho$ 的噪声抹平，天花板 $A$ 硬生生从 0.52 拔高到了 0.61。这是纯粹的工程胜利。Horace He 在 Thinking Machines Lab 写的那篇 defeating nondeterminism 博客也是这个道理，在 RL 里，determinism 就是性能。

#### 2.4 Data 过滤：不浪费一滴 GPU Hour
*   **Zero-Variance Filtering**: 一个 prompt 生成 16 个 rollouts，如果 16 个全对或全错，advantage 就是 0，梯度就是 0。这种数据留着就是白白烧 GPU。Drop 掉它们，用 effective batch 算梯度。
*   **No-Positive-Resampling**: 记录每个 prompt 的历史正确率。一旦某题正确率 $\ge 0.9$，说明模型已经彻底掌握了，永久踢出训练集。

**Intuition:** RL 的 compute 太贵了，不能用来复习已经学会的题，也不能用来死死纠结完全做不出来的题。把 compute 集中在“跳一跳够得着”的 prompt 上，才能最大化梯度的信噪比。

### 3. 实验数据表的深度解读

我们来看 Section 5 的 Table 1，这里藏着关于如何分配 compute 的终极答案。

| Experiment | $C_{mid}$ | B | A |
| :--- | :--- | :--- | :--- |
| ScaleRL (8B, 14k context) | 2542 | 1.92 | 0.610 |
| ScaleRL-32k (8B, 32k context) | 11272 | 1.89 | 0.645 |
| ScaleRL-Scout (17Bx16 MoE) | 4242 | 1.65 | 0.710 |
| ScaleRL-bs2048 | 10909 | 1.70 | 0.645 |

**Build Intuition:**
*   **Generation Length (14k vs 32k)**: 32k 的 $C_{mid}$ 变大了（2542 -> 11272），意味着早期学得慢；但 $A$ 变高了（0.610 -> 0.645）。给模型更长的思考预算，模型在早期会因为生成太长导致 throughput 下降，看起来不如 14k 的跑得快。但只要 compute 给足，32k 最终能超越 14k。**Long-context RL 是一个提升天花板的 knob，牺牲的是早期效率。**
*   **Model Scale (8B vs MoE)**: Scout MoE 的 $A$ 达到了 0.710。更关键的是，Scout 只用了 8B model 1/6 的 compute 就达到了更高的性能。大模型不仅 ceiling 高，而且 truncation rate 极低（<2%），因为它 instruction-following 更好，你让它停下它就停下，不会像 8B 那样学会“拖时间”的恶习。
*   **Batch Size (768 vs 2048)**: 小 batch 早期看起来好，但在 downstream benchmark 上会 stagnate。大 batch 天花板更高。这与 pre-training 的 Chinchilla 结论一致：大 batch 能看到更全局的 gradient landscape。

### 4. 那些“坑”：Truncation 与 Entropy 的迷思

这篇 paper 还澄清了社区里的两个错误直觉：

1.  **Truncation 是 Instability 的头号杀手**：很多人发现 RL 训着训着就崩了。作者发现，崩盘前兆就是 truncation rate 飙升。模型发现答不出难题，就开始“钻空子”，无限输出废话直到撞到 12k token 的长度上限。一旦超过 10% 的 generations 被截断，effective batch size 就会缩水，梯度噪声爆炸，训练直接崩溃。ScaleRL 用了硬核的 forced interruption（直接强行插入 `

---

format and starts producing valid outputs that pass the verifier. Once this format is locked in, the real reasoning capability scaling begins, following the sigmoid curve. This is a strong parallel.

Another association: The generator-trainer split. This is essentially an actor-learner decoupling, similar to IMPALA or RLHF setups like those in DeepSpeed-Chat. The key issue highlighted here is the numerical determinism between inference kernels (e.g., vLLM, TensorRT-LLM) and training kernels (e.g., FlashAttention, FSDP). This nondeterminism in logits makes $\rho_{i,t}$ noisy. In on-policy RL (where generator and trainer are exactly the same model and kernel), $\rho_{i,t}$ should be exactly 1. Any deviation from 1 due to numerical error introduces spurious gradient signal. FP32 at the head fixes this.

Let's write.Andrej，这篇 paper 的核心贡献是给 LLM 的 RL compute scaling 建立了一个 predictive framework，类似于 pre-training 的 Chinchilla law。作者通过超过 400,000 GPU-hours 的大量实验，拟合出一个 sigmoidal compute-performance curve，并基于此 ablate 了 RL 训练中的几乎所有关键 design choices，最终组装出一个名叫 ScaleRL 的 recipe。ScaleRL 在 100k GPU-hours 的 scale 下展现了惊人的 predictability，并且超越了 DeepSeek (GRPO)、Qwen-2.5 (DAPO)、Magistral 和 MiniMax-M1 等主流 recipe。

下面我为你拆解这篇 paper 的核心技术细节、公式变量含义以及背后的 intuition。

### 1. Sigmoidal Scaling Framework

作者抛弃了 pre-training 常用的 power law，采用了 sigmoidal curve 来拟合 RL 的 compute-performance 关系。核心公式如 Eq. (1)：

$$ R_C - R_0 = \frac{A - R_0}{1 + (C_{mid}/C)^B} $$

**变量与上下标解析：**
*   $R_C$: 在 compute budget $C$ 下的 expected reward。下标 $C$ 表示 reward 是 compute 的函数。这里具体指 iid validation set 上的 mean@16 pass rate。
*   $R_0$: 训练初始（$C=0$）时的 reward。下标 $0$ 代表初始状态。
*   $A$: Asymptotic pass rate。代表当 compute 趋于无穷大时，reward 收敛的 ceiling。这是衡量一个 method 潜力的最核心指标。
*   $C_{mid}$: Midpoint of the curve。代表达到一半 reward gain $(A-R_0)/2$ 时所需的 compute。下标 $mid$ 顾名思义是中间值的意思。$C_{mid}$ 越小，模型在 early stage 增长越快。
*   $B$: Scaling exponent。控制 curve 的陡峭程度，代表 compute efficiency。$B$ 越大，意味着用更少的 compute 就能达到 asymptote $A$。
*   $C$: Compute budget（如 GPU hours）。

**Intuition: 为什么用 Sigmoid 而非 Power Law？**
Pre-training 的 loss 是 unbounded 的，power law $R_C = A - D/C^B$ 在 high compute 表现好，但在 low compute 会预测出负数或无穷大，因此 pre-training 拟合时通常会砍掉 low compute 的 warmup 阶段。RL 的 metric（accuracy/reward）是 bounded [0,1] 的，且 RL 训练的 evaluation points 极其有限（例如 100k GPU-hours 的 run 只有约 75 个点）。砍掉早期数据会导致无法 fit。Sigmoid 在 low compute 自动趋于 $R_0$，在 high compute 趋于 $A$，更 robust 且更符合 RL 训练的三阶段直觉：
1.  **Low compute (格式拟合)**: 模型快速学习 SFT 数据里遗留的 `...` format，reward 几乎线性增长。
2.  **Mid compute (策略探索)**: 模型开始探索新的 reasoning strategy，reward 加速增长。
3.  **High compute (难度饱和)**: 容易学的 strategy 都学完了，剩下的都是 hard cases，reward 饱和。

在 high compute regime ($C \gg C_{mid}$)，Sigmoid 可以近似为 Power law：
$$ R_C \approx A - \frac{(A-R_0)C_{mid}^B}{C^B} = A - \frac{D}{C^B} \quad \text{where } D = (A-R_0)C_{mid}^B $$
这说明 Sigmoid 是 Power law 的一个 bounded 推广。

### 2. ScaleRL Recipe 的技术拆解

作者 ablate 了大量的 design choices，并做 Leave-One-Out (LOO) 实验。ScaleRL 的核心组件如下：

#### 2.1 Asynchronous RL Setup: PipelineRL-8
作者比较了 PPO-off-policy-k 和 PipelineRL-k。
*   **PPO-off-policy-k**: Generator 生成一个 batch (B prompts)，trainer 分成 k 个 mini-batches 来更新。这是 block-wise 的，trainer 必须等 generator 生成完才开始训练，存在 GPU idle time。
*   **PipelineRL-k**: Generator 和 trainer 是 streaming 的。Generator 生成一点，trainer 就训一点，trainer 更新完立马把 new weights push 给 generator（即使 generator 的 KV cache 是 stale 的）。

**Intuition:** PipelineRL 的 streaming 特性让它更接近 on-policy 训练，减少了 generator 和 trainer 之间的 distribution mismatch。它不仅提升了 compute efficiency $B$（因为减少了 idle time），还微微提升了 asymptote $A$。作者最终选择 PipelineRL-8。

#### 2.2 Loss Type: CISPO
ScaleRL 使用了 Truncated importance sampling RL loss (CISPO)，取代了 GRPO 和 DAPO。

$$ \mathcal{I}_{CISPO}(\theta) = \mathbb{E} \left[ \frac{1}{T} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \text{sg}(\min(\rho_{i,t}, \epsilon_{max})) \hat{A}_i \log \pi_{train}^\theta(y_{i,t}) \right] $$

**变量与上下标解析：**
*   $\theta$: Model parameters。
*   $T$: Batch 中 token 的总数。
*   $i$: Rollout index（上标 $G$ 代表 group size）。
*   $t$: Token index within a rollout（上标 $|y_i|$ 代表 rollout $i$ 的长度）。
*   $\text{sg}$: Stop-gradient function，括号内的值在 backprop 时被视为常数。
*   $\rho_{i,t}$: Token-level importance sampling ratio。$\rho_{i,t} = \pi_{train}^\theta(y_{i,t}) / \pi_{gen}^{\theta_{old}}(y_{i,t})$。下标 $train$ 和 $gen$ 分别代表训练 backend 和生成 backend。
*   $\epsilon_{max}$: Upper clipping threshold。
*   $\hat{A}_i$: Advantage for rollout $i$。

**Intuition:** 相较于 DAPO 将 IS ratio 直接乘在 advantage 上，CISPO 把 truncated IS ratio $\text{sg}(\min(\rho_{i,t}, \epsilon_{max}))$ 只用来 scale gradient 的 magnitude。这种 stop-gradient 的设计让它对 $\epsilon_{max}$ 极其 robust（实验中 $\epsilon_{max}$ 从 4 到 8 几乎无影响），而 DAPO 对 $\epsilon_{max}$ 极其 sensitive，调错了 asymptote $A$ 会直接掉。

#### 2.3 FP32 Precision at LM Head
这可能是最 surprising 且 impactful 的 finding。Generator 和 Trainer 使用不同的 kernel（optimized inference vs training backend），导致 token probability 在 numerical 上有微小 mismatch。这个 mismatch 直接影响 IS ratio $\rho_{i,t}$ 的计算。在 LM head（softmax over vocabulary）这个 mismatch 被放大。

**Intuition:** 在 on-policy RL 理想情况下，$\rho_{i,t}$ 应该恒为 1。任何由 numerical noise 导致的偏差都会引入 spurious gradient。把 LM head 的计算强制升到 FP32，能把 asymptote $A$ 从 0.52 拉到 0.61。这让我想起 Thinking Machines Lab 关于 defeating nondeterminism in LLM inference 的工作。对于 RL 这种对 numerical precision 极其 sensitive 的算法，任何 nondeterminism 都会在 long-horizon training 中 accumulate 成 catastrophic shift。

#### 2.4 Loss Aggregation & Advantage Normalization
*   **Prompt-level loss averaging**: 每个 prompt 的 loss 是其所有 rollout 的 token loss 平均，然后再跨 prompt 平均。这避免了长 answer 的 prompt 主导 loss。
*   **Batch-level advantage normalization**: 使用整个 batch 所有 rollout 的 std 来 normalize advantage，而非 GRPO 的 prompt-level std。
    $$ \hat{A}_i^{norm} = \hat{A}_i / \hat{A}_{std} $$
    这里 $\hat{A}_{std}$ 是全 batch advantage 的标准差。

**Intuition:** Batch-level normalization 让不同难度的 prompt 产生的 gradient signal 量级一致，防止那些 reward variance 极大的 prompt 主导梯度更新。

#### 2.5 Zero-Variance Filtering & No-Positive-Resampling
*   **Zero-Variance Filtering**: 如果一个 prompt 的所有 rollout reward 全一样（全对或全错），advantage 就是 0，gradient 就是 0。把这些 prompt 从 batch 里 drop 掉，使用 effective batch，提高了 signal-to-noise ratio。
*   **No-Positive-Resampling**: 维护一个 prompt 的 pass rate history。如果某个 prompt 的 pass rate $\ge 0.9$，说明模型已经 mastered，永久移除该 prompt。

**Intuition:** 这是一种简单粗暴但有效的 curriculum learning。RL 的 compute 极其昂贵，不应该浪费在已经学会的题上。这也暗示了 RL training data 的 quality 和 difficulty distribution 比 quantity 更重要。

### 3. Experimental Design & Scaling Results

作者分三阶段进行 experiment：
1.  在 3.5k-4k GPU-hours 上 ablate 单个 design choice。
2.  把 stable 的 best choices 组合成 ScaleRL，在 16k GPU-hours 上做 LOO ablation，验证每个 component 在 combined recipe 里是否仍然必要。
3.  把 ScaleRL scale 到 100k GPU-hours，验证 predictability。

**LOO 实验数据表解析 (Figure 7 & Table 1)：**
在 16k GPU-hours 的 LOO 实验中，作者发现大部分 variant 在 asymptote $A$ 上差异不大（均在 $\pm 0.02$ error margin 内），但在 efficiency $B$ 上有差异。这说明 ScaleRL 的 components 主要在提升 efficiency 和 stability，而非单纯提升 ceiling。

在更大 scale 的实验中（Table 1），我们可以看到不同 axis 的 scaling 效果：
*   **ScaleRL (8B, 14k context)**: $A=0.610, B=1.92, C_{mid}=2542$
*   **ScaleRL-32k (8B, 32k context)**: $A=0.645, B=1.89, C_{mid}=11272$
*   **ScaleRL-Scout (17Bx16 MoE)**: $A=0.710, B=1.65, C_{mid}=4242$

**关键发现：**
1.  **Generation length**: 扩展 context length 从 14k 到 32k，early stage progress 变慢（$C_{mid}$ 从 2542 增至 11272），但 asymptote $A$ 更高。Long-context RL 是一个 ceiling-raising knob，并非单纯的 efficiency trade-off。
2.  **Model scale**: Scout MoE 用 1/6 的 compute 就超越了 8B dense model 的 asymptote。更大的模型 reasoning 能力更强，truncation rate 也更低（<2% vs 8B 的 <5%），因为 instruction-following 能力更强，更容易响应 forced interruption。
3.  **Batch size**: 从 512 到 2048，small batch 在 early stage 看起来好，但 larger batch 的 asymptote 更高，且在 downstream benchmark 上不会 stagnate。

### 4. Build My Intuition: 核心洞察

1.  **Bitter Lesson in RL**: 在 small compute 下看起来好的 method，在 large compute 下可能崩。例如 baseline GRPO 在 3.5k GPU-hours 就 destabilize 了，因为 truncation rate 超过 10%。只有 fit scaling curve 才能判断真正的潜力。Early stage 的 reward 增长极具欺骗性。
2.  **Truncation as Instability Indicator**: Generation length 爆炸是 RL 训练 instability 的 leading indicator。当模型学会 "拖时间" 而非 "想清楚" 时，truncation rate 就会飙升，effective batch size 下降，训练崩溃。ScaleRL 用 forced interruptions（如 "Okay, time is up. Let me stop thinking..."）控制 length，比 length penalty 更直接有效。
3.  **Asymptote vs Efficiency Trade-off**: 在做 scaling 决策时，优先提升 asymptote $A$，再优化 efficiency $B$。一个 method 如果 $A$ 低，无论 $B$ 多高，最终都会被 $A$ 高的 method 超越。FP32 fix 是少数能同时提升 $A$ 和 $B$ 的 intervention，而大部分 trick（如 loss aggregation, normalization）主要只调 $B$。
4.  **Generator-Trainer Mismatch**: RL training 对 numerical determinism 极其敏感。Inference kernel 和 training kernel 的微小差异会在 LM head 放大，导致 IS ratio 计算误差。这在 supervised learning里可以忽略，但在 RL 里是致命的。

### 5. References & Related Links

*   **ScaleRL curve fitting code**: [http://www.devvrit.com/scalerl_curve_fitting](http://www.devvrit.com/scalerl_curve_fitting)
*   **GRPO (DeepSeekMath)**: [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
*   **DAPO**: [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)
*   **MiniMax-M1**: [https://arxiv.org/abs/2506.13585](https://arxiv.org/abs/2506.13585)
*   **Magistral**: [https://arxiv.org/abs/2506.10910](https://arxiv.org/abs/2506.10910)
*   **DeepSeek-R1**: [https://arxiv.org/abs/2501.12148](https://arxiv.org/abs/2501.12148)
*   **PipelineRL**: [https://huggingface.co/blog/ServiceNow/pipelinerl](https://huggingface.co/blog/ServiceNow/pipelinerl)
*   **Thinking Machines Lab (FP32 nondeterminism)**: [https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
*   **ProRL**: [https://arxiv.org/abs/2505.24864](https://arxiv.org/abs/2505.24864)
*   **GSPO**: [https://arxiv.org/abs/2507.18071](https://arxiv.org/abs/2507.18071)
*   **Polaris Dataset**: [https://hkunlp.github.io/blog/2025/Polaris](https://hkunlp.github.io/blog/2025/Polaris)
