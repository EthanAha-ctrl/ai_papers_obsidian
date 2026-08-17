---
source_pdf: HDPO Hybrid Distillation Policy Optimization via Privileged.pdf
paper_sha256: 72b0f07ff5eb9d69a91df4fdbcbb243bbb1d153e00decd28aed8b428b2c1a3ac
processed_at: '2026-08-04T23:34:27-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HDPO 用人话讲

## 1. 问题：RL 训练数学模型的死穴

想象你训练一个 model 做数学题。每道题让 model 生成 16 个答案，对的全算正分，错的算零分。然后算 advantage，告诉 model "这些方向好，多走；那些方向差，少走"。

问题出在最难的题上。16 个 rollout 全错。advantage 算出来全是 0。gradient 直接消失。model 完全学不到东西。

这叫 cliff problem。最需要学的题，恰恰学不到。

打个比方。你教小孩做题，简单的题他全对，没进步空间；中等难度的题有时对有时错，这正好是学习的时候；最难的题他每次全错，你给他零分，他每次还是全错。你只说"你又错了"，没告诉他怎么对，他永远学不会。

GRPO 的 advantage 公式 $\hat{A}_i = (r_i - \mu)/\sigma$ 告诉我们：当所有 $r_i = 0$，$\mu = 0$，$\sigma = 0$，advantage 就是 $0/0$ 无定义，实际实现里赋成 0。gradient 死掉。

Le et al. 2025 的 paper（https://arxiv.org/abs/2509.21880）专门讨论这个 zero-variance prompt problem，发现相当一部分 prompt 落在这个 regime。

## 2. 核心 Insight：把答案塞进 prompt，让 model 教自己

HDPO 的 idea 简单到让人拍大腿。

cliff prompt 全错怎么办？把 ground truth 答案直接拼到 prompt 后面，让 model 再生成一遍。这次 model 看到答案了，大概率能生成正确推理。然后把这种"看到答案能做对"的能力，蒸馏回"不看到答案"的 model。

关键巧思在 teacher 和 student 是同一个 model，同一套 weights $\theta$。只是输入不同：
- Teacher 输入：`[problem; ground_truth; partial_solution]`
- Student 输入：`[problem; partial_solution]`

为什么这一点重要？传统 knowledge distillation 用一个大 model 当 teacher，小 model 当 student，两个 model 架构不同，分布总有 gap，这个 gap 不可控。HDPO 里 teacher 和 student 是同一个 function，分布的差异只来自"有没有看到答案"这一件事，可控、可 bound。

这个 idea 的源头是 Vapnik 和 Vashist 2009 的 LUPI（Learning Using Privileged Information）框架（https://doi.org/10.1016/j.neunet.2009.06.042）。训练时给 model 额外信息，推理时不给。HDPO 把这个思想用到 RLVR 上。

## 3. 算法流程：用最简单的话讲

每个 training step 做两件事：

**第一件：标准 GRPO**

正常 sample batch，每个 prompt 生成 16 个 rollout，score，算 advantage，算 GRPO loss。该干嘛干嘛。

**第二件：privileged self-distillation on cliff prompts**

1. 找出所有 cliff prompts：16 个 rollout 全错的那些
2. 对每个 cliff prompt，把 ground truth 拼到 prompt 后面，用同一个 model 生成新的 rollout
3. 过滤：只保留正确的 privileged rollout
4. 算 JSD loss：让 student 在原始 prompt 上的 token 分布，去 match teacher 在 privileged prompt 上的 token 分布
5. 总 loss = GRPO loss + λ × JSD loss

就这么简单。没有 curriculum scheduler，没有 replay buffer，没有 process reward model，没有 scaffolding heuristics。唯一的额外开销是 cliff prompts 上多一次 forward pass 生成 privileged rollouts。

## 4. 为什么 JSD 而不是 KL

这个选择很关键。

Reverse KL $D_{KL}(P\|Q)$ 是 mode-seeking 的。student 会集中在 teacher 最强的那个 mode 上，其他 mode 不管。结果是 policy collapse 到单一解题策略。pass@1 上去了，pass@k 下来——你 sample 多次还是拿到同一个答案，diversity 丢失。

Forward KL $D_{KL}(Q\|P)$ 是 mode-covering 的。student 必须覆盖 teacher 的所有 mode，包括低密度的。这样 diversity 保住了，但可能 mass 分散，greedy accuracy 受影响。

JSD $JSD(P\|Q) = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$，其中 $M = (P+Q)/2$，介于两者之间，但偏 mode-covering。

HDPO 用 JSD 是为了让 distillation 覆盖 teacher 的多种解题策略，不 collapse 到单一 mode。这关联到 Li et al. 2025 的 DPH-RL paper（https://arxiv.org/abs/2509.07430）——他们发现 RLVR fine-tuning 普遍导致 diversity collapse，建议用 mode-covering divergence 替代 mode-seeking 的 reverse KL。

公式里 JSD 的形式：

$$JSD(P\|Q) = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$$

变量：
- $P$: teacher 的下一 token 分布
- $Q$: student 的下一 token 分布
- $M = (P+Q)/2$: 两个分布的平均
- $D_{KL}(P\|M)$: P 偏离 M 的程度
- $D_{KL}(Q\|M)$: Q 偏离 M 的程度

JSD 对称、bounded（值域 $[0, \ln 2]$），数值稳定。实际实现用 top-64 logits renormalize 近似，对 top-k 外的 student mass 做 tail correction $P_{rest} \cdot \ln 2$。

## 5. 两个 Proposition 用人话讲

### 5.1 Proposition 1：同一 model 蒸馏比跨 model 蒸馏严格更优

**问题**：teacher 分布和 student 分布之间总有 gap，这个 gap 越小越好。小 gap 意味着 student 通过适当参数更新理论上能 match teacher。

**Same-model 的 bound**：

$$D_{KL}(P_T \| P_S) \leq \frac{L_\theta^2 \cdot \Delta(g)^2}{2}$$

变量：
- $P_T$: teacher 分布（输入含 ground truth）
- $P_S$: student 分布（输入不含 ground truth）
- $L_\theta$: model 的 local Lipschitz constant，衡量输入小扰动导致 logit 多大变化
- $\Delta(g)$: ground truth tokens $g$ 在 input space 引入的距离

**直觉**：同一个 model，输入差多少，logit 就差多少（被 $L_\theta$ 放大），分布就差多少。gap 完全由"ground truth 多大程度改变 model 预测"决定。

**Cross-model 的 bound**：

$$D_{KL}(P_\phi(\cdot|c_T) \| P_\theta(\cdot|c_S)) \leq \frac{(L_\theta \cdot \Delta(g) + \|f_\phi(c_T) - f_\theta(c_T)\|_\infty)^2}{2}$$

变量多了一个 $\|f_\phi(c_T) - f_\theta(c_T)\|_\infty$，这是两个不同 model 在同一 privileged 输入上的 logit 差，叫 model-mismatch term。

**直觉**：跨 model 蒸馏，gap 来自两部分——输入扰动（跟 same-model 一样）+ model 架构差异（same-model 没有）。所以 same-model 严格 tighter。

**Lemma 1 的数学美感**：

$$D_{KL}(\text{softmax}(z) \| \text{softmax}(z+\delta)) \leq \frac{\|\delta\|_\infty^2}{2}$$

证明用 Taylor's theorem 写成 $\int_0^1 (1-s) Var_{Q_s}(\delta) ds$，再用 $Var \leq \|\delta\|_\infty^2$。这个 bound 在信息论里是标准的，论文 Appendix A 完整给出了推导。Lipschitz continuity 假设来自 Kim et al. 2021（https://arxiv.org/abs/2006.04710）对 self-attention 的分析，per-layer Lipschitz constant 大约 $O(C\sqrt{n})$，其中 $n$ 是 sequence length。

### 5.2 Proposition 2：R=1 filter 就是 KL-regularized RL 的最优 policy

**问题**：用 ground truth 让 model 生成 + 只保留正确的，这种 rejection sampling 目标是什么？是某个 well-defined 的最优 policy 吗？

**KL-regularized RL objective**：

$$\max_\pi \mathbb{E}_{\tau \sim \pi}[R(\tau)] - \beta \cdot KL(\pi \| \pi_{ref})$$

变量：
- $\pi$: 待优化 policy
- $\pi_{ref}$: reference policy，防止 policy 漂太远
- $\beta$: KL penalty 强度，越大越保守，越小越激进
- $R(\tau) \in \{0, 1\}$: binary reward
- $\tau$: trajectory

**Closed-form optimal**（DPO paper Rafailov et al. 2023 推导，https://arxiv.org/abs/2305.18290）：

$$\pi^*(\tau) = \pi_{ref}(\tau) \cdot \frac{\exp(R(\tau)/\beta)}{Z(\beta)}$$

变量：
- $Z(\beta)$: partition function，归一化常数
- $\exp(R(\tau)/\beta)$: reward 的 Boltzmann weight

**Hard-threshold limit $\beta \to 0^+$**：

错误的 trajectory：$R=0$，weight 是 $\exp(0) = 1$。但 $Z(\beta) \sim p \cdot \exp(1/\beta) \to \infty$，所以 $\pi^*(\tau) \to 0$。

正确的 trajectory：$R=1$，weight 是 $\exp(1/\beta)$。分子分母同除 $\exp(1/\beta)$：

$$\pi^*(\tau) \to \frac{\pi_{ref}(\tau)}{p} = \frac{\pi_{ref}(\tau)}{P_{\pi_{ref}}(R=1)} = \pi_{ref}(\tau | R=1)$$

**直觉**：当 KL penalty 趋于 0，optimal policy 就是 reference policy 在正确 trajectory 上的条件分布。机械实现就是 rejection sampling from reference + filter R=1。

**HDPO 的应用**：

在 cliff prompts 上，原始 reference $\pi_{ref}(\cdot | \text{prompt})$ 在 correct trajectories 上几乎没有 support（$P(R=1) \approx 0$），所以 $\pi_{ref}(\cdot | R=1)$ 退化，rejection sampling 拿不到样本。

HDPO 把 reference 换成 proxy $\pi_\theta(\cdot | x, g)$（注入 ground truth 的 model）。这个 proxy 在 correct space 上有 non-degenerate support，所以 $\pi_\theta(\cdot | x, g, R=1)$ 是 well-defined target。HDPO 用 R=1 rejection sampling from proxy 来 finite-sample 估计这个 target。

两个近似误差来源：
1. Proxy reference 跟真实 reference 的 gap（Proposition 1 bound）
2. Sampling noise（privileged pass rate 在难题上可能远小于 1）

## 6. 实验：到底 work 不 work

主实验在 OpenMathInstruct-2 上用 Qwen2.5-Math-1.5B-Instruct，2000 training steps，8×H200 GPU。

| Method | pass@1 | pass@4 | pass@8 |
|--------|--------|--------|--------|
| GRPO Baseline | **0.6519** | 0.7749 | 0.8228 |
| HDPO (frozen, λ=0.01) | 0.6519 | 0.7812 | 0.8218 |
| HDPO (frozen, λ=0.1) | 0.6304 | 0.7812 | **0.8398** |
| HDPO (drifting, λ=0.01) | 0.6514 | **0.7861** | 0.8271 |
| HDPO (drifting, λ=0.1) | 0.6294 | 0.7856 | 0.8364 |

**用大白话解读**：

**λ=0.01 是 free lunch regime**。distillation 信号很轻，pass@1 几乎不掉（0.6514 vs 0.6519），但 pass@4 和 pass@8 都涨。Drifting teacher + λ=0.01 是最好的组合，pass@4 +1.1%，pass@8 +0.4%。相当于白嫖一些 coverage。

**λ=0.1 是 explicit tradeoff regime**。distillation 信号强，pass@8 涨到 0.84 附近（+1.4-1.7%），但 pass@1 掉 2-3%。你主动选择牺牲 greedy accuracy 换 broader coverage。

**Drifting vs Frozen**：drifting teacher 用当前 policy weights，realizability gap 小（Proposition 1 里 mismatch term 是 0），在低 λ 时占优。Frozen teacher 用初始 weights，保留了初始 model 的 diversity（还没被 RL mode-seeking 塑形），在高 λ 时 pass@8 最高。这暗示 diversity 和 realizability 之间有 tradeoff。

**Hardware robustness**：8×H100 上重复实验（Table 3），λ=0.1 的 pass@8 提升一致（+1.4-1.5%），λ=0.01 的提升较小（接近 noise floor），因为 floating-point non-determinism 跨 GPU 微架构累积 2000 步产生 subtly different weights。

## 7. pass@1 vs pass@k Tradeoff 的 Intuition

论文给了一个有意思的 hypothesis。

当 privileged model 解一道题有多条策略，JSD loss 训练 student 把 mass 放到所有这些策略上。对 1.5B 这种小 model，capacity 有限，多个 mode 竞争同一组参数。结果是分布变 flat，没有单一策略能 dominant，greedy decoding 拿不到 clean answer（pass@1 降）。但 broader support 意味着多次采样能 discover 不同策略（pass@k 升）。

理想分布不是 uniform，是 concentrated：一个 dominant mode（greedy 可靠 recover）+ 若干 small secondary modes（repeated sampling 可 discover）。纯 RL 自然产生这个 shape，通过 mode-seeking dynamics。但 cliff 上纯 RL 找不到 mode。

这 motivate 了论文 Section 6 的 expand-then-sharpen curriculum：
1. HDPO 阶段：在 cliff 上 broaden strategy support
2. RL 阶段：re-inject 已经可解的 cliff 回 RL training，让 mode-seeking dynamics sharpen dominant mode

关键：re-injection 要 delay。先让 distilled strategies 稳定 encode 进参数，再 sharpen，避免 transient accessibility。这跟 simulated annealing 高温 explore、低温 exploit 一个道理。

更深层联想：跟 biological evolution 的 speciation + selection 类似。distillation 引入新 genotypes（broaden），RL mode-seeking 是 selection pressure（sharpen）。

## 8. 跟相关工作的对比，用大白话

**VCRL (Jiang et al. 2025, https://arxiv.org/abs/2509.19803)**：看 group reward variance 当 prompt 难度 proxy，动态调度采样 focus 高 variance 的 frontier prompt。问题：cliff prompt 是 variance=0，直接被跳过了。HDPO 直接学 cliff。

**DAPO (Yu et al. 2025, https://arxiv.org/abs/2503.14476)**：filter 掉 zero-variance prompts，oversample 直到找到 informative 的。同上问题：cliff 被跳过。HDPO 学 cliff。

**Scaf-GRPO (Zhang et al. 2025b, https://arxiv.org/abs/2510.19807)**：往 prompt 里 inject 分层 hints（从抽象概念到具体步骤），随 model 进步 fade。需要设计 hint 生成策略、决定何时 inject 何时 fade、管理 scaffolded vs unscaffolded generation 的 distributional gap。HDPO 用 ground truth 当 privileged info，不需要设计 hint 层次。

**HINT (Wang et al. 2025, https://arxiv.org/abs/2510.09388)**：给 ineffective rollout 提供 targeted guidance 帮它们 navigate 到正确解。需要 guidance 策略。HDPO 用 ground truth 直接做 guidance。

**Retrospective Replay (Dou et al. 2025, https://arxiv.org/abs/2504.14363)**：存早期 exploratory trajectory，model 变强后 revisit。需要 buffer 管理，replay trajectory 是 off-policy，可能 stale。HDPO 是 online distillation，没有 staleness 问题。

**RLEP (Zhang et al. 2025a, https://arxiv.org/abs/2507.07451)**：收集 verified correct trajectory 混进 mini-batch。同样 buffer + staleness 问题。HDPO 用同一个 model 当 teacher，no external trajectory。

**Le et al. 2025 (https://arxiv.org/abs/2509.21880)**：对 zero-variance prompt 用 entropy-scaled gradient，按 token entropy 比例 scale gradient。reclaim 一点学习信号。不需要额外 model，但信号弱。HDPO 用 privileged distillation 提供强信号。

**PRIME (Cui et al. 2025, https://arxiv.org/abs/2502.01456)**：online process reward model，用 policy rollout 和 outcome label 训 PRM。需要训和 maintain 一个单独 reward model。HDPO 不需要 PRM。

**ReLIFT (Ma et al. 2025, https://arxiv.org/abs/2506.07527)**：概念上最接近 HDPO，interleave RL + online SFT 在 model 不会的题上。但需要两阶段 training loop + external 高质量解 + interleaving schedule 调参。HDPO 单一 unified objective，用 model 自己 privileged rollout 当监督，有理论保证。

**OPSD (Zhao et al. 2026, https://arxiv.org/abs/2601.18734)**：单 LLM 同时 teacher 和 student，teacher 条件在 privileged info 上，student 只看问题，在 student on-policy rollout 上最小化 per-token divergence。HDPO 不同：只在 cliff 上 distill，用 R=1 filtering 选最优 target。

**SDPO (Hübotter et al. 2026, https://arxiv.org/abs/2601.20802)**：把 textual feedback（runtime error, judge evaluation）转成 dense signal，蒸馏 feedback-conditioned prediction 回 unconditional policy。HDPO 用 ground truth 当 privileged info，target cliff prompts。

**KDRL (Xu et al. 2025, https://arxiv.org/abs/2506.02208)**：同时 minimize reverse KL between student and teacher + maximize expected reward。HDPO 用 JSD 而非 reverse KL，且只在 cliff 上 trigger。

**RLAD (Zhang et al. 2026, https://arxiv.org/abs/2602.22495)**：把 importance ratio 换成 old policy 和 teacher 的 geometric mixture。HDPO 用 separate loss term，不修改 importance ratio。

**G-OPD (Yang et al. 2026, https://arxiv.org/abs/2602.12125)**：理论上证明 on-policy distillation 是 dense KL-constrained RL 的特例，引入 reward extrapolation 让 student 超越 teacher。HDPO focus 在 cliff 上，用 R=1 hard filter 实现 $\beta \to 0$ limit。

**π-Distill (Penaloza et al. 2026, https://arxiv.org/abs/2602.04942)**：独立得到相同 objective 结构，在 finite $\beta$ 下对所有 prompts 做 gradient ascent。HDPO 在 $\beta \to 0$ limit 用 hard R=1 filtering 机械实现，只在 cliff prompts 上 trigger。

HDPO 的 elegance 在于用最少的额外复杂度实现 cliff 上的 learning signal。其他方法都需要 curriculum scheduler、hint generator、replay buffer、PRM、multi-phase training loop 中的一个或多个。HDPO 只需要 append ground truth、generate、filter、JSD。

## 9. 更深层的联想与 Intuition

### 9.1 跟 Offline RL 的联系

HDPO 在 cliff prompts 上本质是 online 形式的 offline RL。从 privileged generation（off-policy data）+ R=1 filter 得到 demonstrations，然后蒸馏。类似 AWAC（Nair et al. 2020, advantage-weighted regression）、CQL（Kumar et al. 2020, conservative Q-learning）、IQL（Kostrikov et al. 2022, implicit Q-learning）。不同点：HDPO 的 offline data 是 on-the-fly 生成的，用 JSD 而非 regression。

### 9.2 跟 Mean Teacher 的联系

Drifting teacher 跟 semi-supervised learning 里的 Mean Teacher（Tarvainen & Valpola 2017, https://arxiv.org/abs/1703.01780）有结构相似性。Mean Teacher 用 student weights 的 EMA 当 teacher，提供 consistency loss。HDPO drifting 用 student 当前 weights（无 EMA），不同 input 当 teacher。共同点：teacher 和 student 同源，gap 来自某种 perturbation——Mean Teacher 是 weight perturbation，HDPO 是 input perturbation。

### 9.3 跟 DPO 的联系

DPO（Rafailov et al. 2023, https://arxiv.org/abs/2305.18290）直接 optimize KL-regularized RL objective 的 closed form，避免显式 RL。HDPO 的 Proposition 2 用 DPO 的 closed form 推 hard-threshold limit $\beta \to 0$，optimal = $\pi_{ref}(\cdot | R=1)$，机械实现就是 R=1 rejection sampling。可以说 HDPO 在 cliff prompts 上用 rejection sampling 代替 gradient-based optimization，因为 cliff 上 gradient fails 但 sampling 在 privileged proxy 上可行。这是 RLVR + rejection sampling 的 hybrid。

### 9.4 Information-Theoretic 视角

Cliff problem 本质是 reference policy 在 correct trajectory space 上 entropy 为 0。$P_{\pi_{ref}}(R=1) \to 0$ 时 conditional distribution $\pi_{ref}(\cdot | R=1)$ 不再 well-defined。HDPO 通过 ground truth 注入，在 input side 引入额外信息 $\Delta(g)$，这个 information content 转化为 reference 在 correct space 上的 non-degenerate support。Information injection → distributional support expansion → effective learning target。

从信息论看，HDPO 是 information-theoretic regularization：把 ground truth 的 information content 注入 reference，再通过 distillation 传递回 student 参数。

### 9.5 Lipschitz Bound 的 Loose 但 Useful

Proposition 1 的 bound 严格说很 loose。Deep transformer 的 $L_\theta$ 可能很大（per-layer $O(C\sqrt{n})$，L 层 composition 可能指数级）。但论文 Section A.5 的 Remark 关键：比较同一 bound 下的两个 case，looseness 同样 affect 两者，不改变比较结论。absolute magnitude 不准，relative comparison 仍然有效。这是 inequality reasoning 的常见技巧，在 differential privacy composition bound、generalization bound 里也常见。

### 9.6 R=1 Filter 跟 Importance Sampling 的等价

R=1 rejection sampling 在 binary reward 下是 importance sampling 的 extreme form。Importance weight $w(\tau) = \exp(R(\tau)/\beta)/Z(\beta)$，当 $\beta \to 0$，$w(\tau) \to \mathbb{1}[R(\tau)=1]/P(R=1)$。这就是 rejection sampling：接受概率 $\propto \mathbb{1}[R(\tau)=1]$。

更精细的 view：HDPO 用 hard filter 是 $\beta \to 0$ limit，general $\beta$ 对应 soft filter（weighted sampling with $\exp(R/\beta)$）。Future work 可以探索 finite $\beta$ soft filter 是否更 smooth。

### 9.7 Privileged Info 的 Generalization

论文用 math（ground truth = solution），但 privileged info 形式可以 generalize：
- Code generation：ground truth = test cases passing，privileged = 看到 test cases
- Scientific reasoning：ground truth = 最终结论，privileged = 知道结论
- Tool use：ground truth = 正确工具调用序列，privileged = 看到调用 schema
- Multi-step planning：ground truth = goal state，privileged = 看到 goal

关键是 ground truth 能 inject 进 prompt 且不破坏 solution 的 validity。

### 9.8 Capacity Bottleneck 的 Open Question

1.5B 上 pass@1 vs pass@k tradeoff 明显。Hypothesis：更大 model capacity 充足，可能不损失 pass@1。Mixture model capacity 分析一致：小 model K modes 竞争 N parameters，大 model N >> K 每个 mode 独立 fit。如果成立，HDPO 在大 model 上是 pure win。这是 future work 关键方向。

### 9.9 Ground Truth 形式的 Curriculum

不一定每次注入完整 ground truth。可以：
- 早期：完整 ground truth（max privileged）
- 中期：partial hint（intermediate privileged）
- 后期：无 hint（vanilla RL）

这是 privileged info 的 fade-out curriculum，跟 Scaf-GRPO 的 hint fade 思路类似但更 principled。

### 9.10 跟 Process Reward Models 的 Complementarity

HDPO 在 outcome level 工作（R=1 filter）。可以与 process reward（Lightman et al. 2024, https://arxiv.org/abs/2305.20050）结合：
- Non-cliff prompts：process reward densify signal
- Cliff prompts：HDPO privileged distillation

两者 complementary，覆盖不同 regime。

## 10. Limitations 说人话

**Single model scale**：只在 1.5B 上验证。Larger models 可能 fewer cliffs（baseline 成功率高，HDPO 边际收益降低），或更大 capacity（coverage gain 不损失 pass@1，HDPO 是 pure win）。Open question。

**Single dataset**：只在 OpenMathInstruct-2（MATH + GSM8K）上验证。Generalization 到 code、logic、scientific reasoning 需要 experiment。

**Computational overhead**：privileged generation 一个额外 forward pass per cliff prompt，top-k teacher logits 额外 logits 处理，JSD loss 额外 backward pass。Overhead ∝ cliff prompts per step。对 small batch / many cliffs，overhead 显著。

**Frozen vs Drifting 的 Tradeoff**：frozen 在 λ=0.1 时 pass@8 最高，drifting 在 λ=0.01 时更好。暗示 diversity 和 realizability 之间有 tradeoff。理想可能是 EMA teacher（between frozen 和 drifting），未探索。

**Tail Correction 的 Validity**：Top-64 logits renormalize 假设 teacher 分布 mass 集中在 top-64。对 rare token（数字、变量名）可能 violated。Tail correction $P_{rest} \ln 2$ 假设 student rest mass 是 uniform，实际 student rest distribution 也 non-trivial。更精确做法：full softmax JSD，但计算开销大。

## 11. 一句话总结

HDPO 的故事：RLVR 训数学 model 时最难的题学不到（cliff problem），把 ground truth 塞进 prompt 让 model 自己生成正确解，过滤保留对的，用 JSD 蒸馏回原始 prompt 上的 model。Teacher 和 student 同一个 model 同一套 weights，gap 完全来自"有没有看到答案"这件事，理论可 bound。R=1 filter 等价于 KL-regularized RL 在 $\beta \to 0$ limit 的最优 policy。实验显示 λ=0.01 时保 pass@1 提升 coverage（free lunch），λ=0.1 时主动 trade pass@1 换 pass@8。Expand-then-sharpen curriculum 是 future work：HDPO 先 broaden strategy support，RL 再 sharpen dominant mode。

核心美感：simple idea + tight theory + modest but consistent empirical gains。Simple 不等于 obvious，理论分析让 simple idea 变 principled。

参考资料汇总：
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen2.5-Math: https://arxiv.org/abs/2409.12122
- OpenMathInstruct-2: https://arxiv.org/abs/2410.01560
- LUPI: https://doi.org/10.1016/j.neunet.2009.06.042
- DPO: https://arxiv.org/abs/2305.18290
- Lipschitz of self-attention: https://arxiv.org/abs/2006.04710
- Le et al. zero-variance: https://arxiv.org/abs/2509.21880
- DPH-RL diversity collapse: https://arxiv.org/abs/2509.07430
- GKD: https://arxiv.org/abs/2306.13649

---

# HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation 深度解析

## 1. Cliff Problem 的本质：为什么 RL 在数学推理上"学不动"

要真正理解 HDPO 解决的问题，得先看 GRPO 在 binary reward 下的 advantage 计算：

$$\hat{A}_i = \frac{r_i - \mu}{\sigma}$$

变量解释：
- $r_i \in \{0, 1\}$: 第 $i$ 个 rollout 的 binary reward（数学题对错）
- $\mu = \frac{1}{G}\sum_i r_i$: group 内 reward 均值
- $\sigma$: group 内 reward 标准差
- $G$: 每个 prompt 的 rollout 数量（论文中 G=16）

考虑三种 prompt 难度 regime：

| Case | 现象 | $\mu$ | $\sigma$ | $\hat{A}_i$ | Gradient |
|------|------|-------|----------|-------------|----------|
| (i) 所有 rollout 成功 | model 已学会 | 1 | 0 | 0 | 0 |
| (ii) 部分成功 | 标准 learning regime | $0<\mu<1$ | $>0$ | 有正有负 | 有信号 |
| (iii) 所有 rollout 失败 | **cliff** | 0 | 0 | 0 | **0** |

Case (iii) 就是 cliff problem 的核心：最难的题（model 完全不会的），恰恰是 gradient 完全消失的地方。Le et al. 2025 在他们的 benchmark 上观察到相当一部分 prompts 落在这个 regime，形成一个 persistent learning dead zone。

这个问题的本质在于：policy gradient 方法需要 contrastive signal（成功 vs 失败的对比）才能形成 advantage。没有对比，就没有学习信号。这跟 supervised learning 有本质区别——SFT 看到 ground truth 答案就能学，而 RLVR 在所有 rollout 都错的时候，gradient 直接归零。

参考资料：GRPO 的原始论文 https://arxiv.org/abs/2402.03300 (DeepSeekMath)；Le et al. 的 entropy-guided advantage shaping https://arxiv.org/abs/2509.21880

## 2. HDPO 的核心 Insight：Self-Teaching via Privileged Information

HDPO 的核心 insight 极其简洁优雅：

**对于 cliff prompts，把 ground truth 注入 prompt，让同一个 model 重新生成，就能拿到 correct trajectories。然后把这些 correct trajectories 蒸馏回 model 在原始 prompt 上的 policy。**

关键点在于"同一个 model"。这意味着：
- Teacher $\pi_T$ 和 student $\pi_\theta$ share 相同的 weights $\theta$
- 唯一差别：teacher 的输入是 $c_T = [x; g; y_{<t}]$（包含 ground truth $g$），student 的输入是 $c_S = [x; y_{<t}]$（不含 $g$）
- 因为是同一个 function，distributional gap 完全来自 privileged information $g$ 本身

这跟传统的 cross-model distillation 形成鲜明对比：传统蒸馏引入 teacher model $\phi$ 和 student model $\theta$ 之间的 architectural mismatch，这个 mismatch term 在 same-model 设置下被完全消除。

这种思路的灵感来自 Vapnik & Vashist 2009 的 LUPI (Learning Using Privileged Information) 框架：https://doi.org/10.1016/j.neunet.2009.06.042

## 3. 算法详解

### 3.1 总 loss 形式

$$\mathcal{L}_{HDPO}(\theta) = \mathcal{L}_{GRPO}(\theta) + \lambda \cdot \mathcal{L}_{JSD}(\theta)$$

变量：
- $\mathcal{L}_{GRPO}$: 标准 GRPO clipped policy gradient loss
- $\mathcal{L}_{JSD}$: token-averaged JSD over filtered teacher trajectories on cliff prompts
- $\lambda$: distillation weight，控制 exploration-exploitation tradeoff

### 3.2 JSD Loss 的具体形式

$$\mathcal{L}_{JSD}(\theta) = \frac{1}{N_{tok}} \sum_{(x,y) \in \mathcal{T}} \sum_{t=1}^{|y|} JSD_k(\pi_T(\cdot | y_{<t}) \| \pi_\theta(\cdot | y_{<t}))$$

变量详解：
- $\mathcal{T}$: distillation set，由两层 filtering 构造
- $N_{tok} = \sum_{(x,\bar{y}) \in \mathcal{T}} |\bar{y}|$: 全局总 token 数（across all data-parallel ranks，rank-invariant normalization）
- $y_{<t}$: trajectory 的前 $t-1$ 个 token（teacher 生成的 prefix）
- $\pi_T(\cdot | y_{<t})$: teacher 在 prefix $y_{<t}$ 下的下一 token 分布
- $\pi_\theta(\cdot | y_{<t})$: student 在相同 prefix 下的下一 token 分布
- $JSD_k$: 用 teacher 的 top-k（k=64）logits renormalized 后计算的 JSD

注意一个微妙之处：这里的 condition 是 **teacher 生成的 trajectory 的 prefix** $y_{<t}$，不是 student 自己的 on-policy prefix。对 teacher 是 on-policy，对 student 是 off-policy。这种 design 让 distillation 更接近 teacher 的真实分布，而不是 student 当前能 reach 的分布。

### 3.3 两层 Filtering 构造 $\mathcal{T}$

**第一层：识别 cliff prompts**

$$\mathcal{C} = \{x \in B : \sum_k R(x, y^{(k)}) = 0\}$$

变量：
- $B$: 当前 batch 的 prompt set
- $y^{(k)} \sim \pi_\theta(\cdot | x)$: 第 $k$ 个 standard rollout
- $R(x, y) \in \{0,1\}$: binary reward function
- 条件：所有 K 个 rollouts 都失败（reward 之和为 0）

**第二层：privileged rollouts 上做 R=1 rejection sampling**

$$\mathcal{T} = \{(x, \bar{y}) : x \in \mathcal{C}, \bar{y} \sim \pi_\theta(\cdot | x \oplus y^*), R(x, \bar{y}) = 1\}$$

变量：
- $y^*$: ground truth solution
- $x \oplus y^*$: 把 ground truth 拼接进 prompt（privileged context）
- $\bar{y}$: privileged rollout
- 条件：privileged rollout 正确（R=1）

注意 $\pi_\theta$ 和 $\pi_T$ 共享 weights，所以这里 $\bar{y} \sim \pi_\theta(\cdot | x \oplus y^*)$ 其实就是 teacher 的 generation。

### 3.4 Top-k JSD 的 Tail Correction

$JSD_k$ 用 teacher 的 top-64 logits renormalize。对 student 在 top-k support 之外的 mass $\breve{P}_{rest}$，tail correction 是：

$$\breve{P}_{rest} \cdot \ln 2$$

这个 correction 怎么来的？JSD 的标准定义：

$$JSD(P\|Q) = \frac{1}{2} KL(P\|M) + \frac{1}{2} KL(Q\|M), \quad M = \frac{P+Q}{2}$$

对 top-k 之外的 token，teacher mass 假设为 0（因为 renormalize 后丢掉了），student 总 mass 是 $P_{rest} = 1 - \sum_{i \in \text{top-}k} P_S(i)$。则：

$$M_{rest} = \frac{0 + P_{rest}}{2} = \frac{P_{rest}}{2}$$

$$KL(Q\|M)_{\text{rest}} = P_{rest} \log\frac{P_{rest}}{P_{rest}/2} = P_{rest} \log 2 = P_{rest} \ln 2$$

（用自然对数时是 $\ln 2$，用 $\log_2$ 时是 1）

这个 correction 确保了 student 不在 teacher top-k support 内的 mass 仍然被 penalize，避免 student 在 top-k 之外任意分 mass。

### 3.5 完整 Algorithm 1 流程

```
Require: Policy π_θ, prompt set X, ground truth {y*}, reward R, 
         learning rate α, distillation weight λ, rollouts per prompt K

1: for step = 1, ..., N do
2:   // ============ Standard GRPO ============
3:   Sample prompt batch B ⊂ X
4:   for all x ∈ B do
5:     Generate K rollouts y^(k) ~ π_θ(·|x)
6:   end for
7:   Score: Â_i = (r_i - μ) / σ
8:   Compute L_GRPO via clipped policy gradient with leave-one-out advantages
9:   
10:  // ============ Privileged Self-Distillation ============
11:  Identify cliffs: C = {x ∈ B : Σ_k r^(k) = 0}
12:  for all x ∈ C do
13:    Generate ȳ^(j) ~ π_θ(·|x ⊕ y*)
14:  end for
15:  Filter: T = {(x, ȳ) : R(x, ȳ) = 1}
16:  Compute L_JSD = (1/N_tok) Σ JSD_k(π_T(·|ȳ_<t) || π_θ(·|ȳ_<t))
17:  
18:  // ============ Update ============
19:  θ ← θ - α ∇_θ (L_GRPO + λ · L_JSD)
20: end for
```

实现细节（来自 Table 2）：
- Base model: Qwen2.5-Math-1.5B-Instruct
- GRPO 配置: G=16 rollouts/prompt, 32 prompts/step, ε=0.2 clip
- Teacher type: drifting（用当前 policy weights）或 frozen（用初始 weights）
- Top-k=64, λ ∈ {0.01, 0.1}
- vLLM colocated generation, temperature=1.0
- AdamW, lr=1e-6, linear warmup 50 steps

## 4. Proposition 1 详解：Realizability Gap 比较

这个 proposition 是 HDPO 理论分析的基石，证明 same-model distillation 的 realizability gap 严格小于 cross-model distillation。

### 4.1 Lemma 1: KL Divergence under Logit Perturbation

**Statement**:
$$D_{KL}(\text{softmax}(z) \| \text{softmax}(z + \delta)) \leq \frac{\|\delta\|_\infty^2}{2}$$

变量：
- $z \in \mathbb{R}^{|\mathcal{V}|}$: 原始 logits
- $\delta \in \mathbb{R}^{|\mathcal{V}|}$: logit perturbation（每个 token 一个分量）
- $\|\delta\|_\infty = \max_v |\delta_v|$: $\ell_\infty$ norm
- $|\mathcal{V}|$: vocabulary 大小

**Proof**:

设 $P = \text{softmax}(z)$, $Q = \text{softmax}(z+\delta)$。

$$D_{KL}(P\|Q) = \sum_v P(v) \log\frac{P(v)}{Q(v)}$$

代入 softmax：

$$P(v) = \frac{\exp(z_v)}{Z}, \quad Q(v) = \frac{\exp(z_v + \delta_v)}{Z'}, \quad Z = \sum_j \exp(z_j), \quad Z' = \sum_j \exp(z_j + \delta_j)$$

展开：

$$D_{KL}(P\|Q) = \sum_v P(v) \left[(z_v - \log Z) - (z_v + \delta_v - \log Z')\right] = -\mathbb{E}_P[\delta] + \log Z' - \log Z$$

关键观察：

$$\frac{Z'}{Z} = \frac{\sum_j \exp(z_j) \exp(\delta_j)}{\sum_j \exp(z_j)} = \sum_j \frac{\exp(z_j)}{Z} \exp(\delta_j) = \mathbb{E}_P[\exp(\delta)]$$

所以：

$$D_{KL}(P\|Q) = \log\mathbb{E}_P[\exp(\delta)] - \mathbb{E}_P[\delta]$$

这是 $\delta$ 在 $P$ 下的 centered cumulant generating function 在 $s=1$ 处的值。

定义 $\psi(s) = \log\mathbb{E}_P[\exp(s\delta)] - s\mathbb{E}_P[\delta]$ for $s \in [0,1]$。

性质：
- $\psi(0) = 0$
- $\psi(1) = D_{KL}(P\|Q)$
- $\psi'(s) = \mathbb{E}_{Q_s}[\delta] - \mathbb{E}_P[\delta]$，其中 $Q_s(v) \propto P(v)\exp(s\delta_v)$ 是 exponentially tilted distribution
- $\psi''(s) = Var_{Q_s}(\delta)$

用 Taylor's theorem with integral remainder：

$$\psi(1) = \psi(0) + \psi'(0) \cdot 1 + \int_0^1 (1-s) \psi''(s) ds = \int_0^1 (1-s) Var_{Q_s}(\delta) ds$$

（因为 $\psi(0) = 0$ 和 $\psi'(0) = \mathbb{E}_P[\delta] - \mathbb{E}_P[\delta] = 0$）

对于 bounded domain，$Var(X) \leq \mathbb{E}[X^2] \leq \|X\|_\infty^2$。因为每个 $\delta_v \leq \|\delta\|_\infty$：

$$D_{KL}(P\|Q) \leq \|\delta\|_\infty^2 \int_0^1 (1-s) ds = \frac{\|\delta\|_\infty^2}{2} \quad \square$$

**Intuition**：logit 的最大 perturbation $\|\delta\|_\infty$ 决定了 KL 的上界。perturbation 越大，KL 越大。这个 bound 的形式是 $\|\delta\|_\infty^2 / 2$，平方关系意味着小 perturbation 的影响被 quadratic 抑制。

### 4.2 Assumption 1: Local Lipschitz Continuity

$$\|f_\theta(c_1) - f_\theta(c_2)\|_\infty \leq L_\theta \cdot d(c_1, c_2)$$

变量：
- $f_\theta$: model 的 logit function（context → logits）
- $c_1, c_2$: 两个输入 contexts
- $L_\theta$: local Lipschitz constant
- $d(\cdot, \cdot)$: input space distance metric

对 transformer，Kim et al. 2021 证明了 self-attention 在 bounded input 上的 local Lipschitz bound，per-layer 大约是 $O(C\sqrt{n})$，其中 $n$ 是 sequence length，$C$ 依赖 weight norms 和 attention distribution concentration。

参考：https://arxiv.org/abs/2006.04710

### 4.3 Part I: Same-Model Bound

**Statement**:
$$D_{KL}(P_T \| P_S) \leq \frac{L_\theta^2 \cdot \Delta(g)^2}{2}$$

变量：
- $P_T = P_\theta(\cdot | c_T)$, $c_T = [x; g; y_{<t}]$ (teacher, 含 ground truth)
- $P_S = P_\theta(\cdot | c_S)$, $c_S = [x; y_{<t}]$ (student, 不含 ground truth)
- $\Delta(g) = d(c_T, c_S)$: ground truth tokens $g$ 引入的输入空间距离
- $L_\theta$: model 的 local Lipschitz constant

**Proof**:

因为 teacher 和 student 用同一个 function $f_\theta$，只是输入不同：

$$\|z_T - z_S\|_\infty = \|f_\theta(c_T) - f_\theta(c_S)\|_\infty \leq L_\theta \cdot d(c_T, c_S) = L_\theta \cdot \Delta(g)$$

代入 Lemma 1 with $\delta = z_T - z_S$：

$$D_{KL}(P_T \| P_S) \leq \frac{\|z_T - z_S\|_\infty^2}{2} \leq \frac{L_\theta^2 \cdot \Delta(g)^2}{2} \quad \square$$

**关键性质**：
1. 只依赖 model $\theta$（通过 $L_\theta$）和 ground truth $g$ 的信息内容（通过 $\Delta(g)$）
2. 不依赖任何 capacity gap
3. distillation 的难度取决于 ground truth 多大程度改变 prediction，不是 architectural difference

### 4.4 Part II: Cross-Model Bound

**Statement**: 用 separate teacher $\phi$ 和 student $\theta$：

$$D_{KL}(P_\phi(\cdot|c_T) \| P_\theta(\cdot|c_S)) \leq \frac{(L_\theta \cdot \Delta(g) + \|f_\phi(c_T) - f_\theta(c_T)\|_\infty)^2}{2}$$

**Proof**: Logit 差分解：

$$f_\phi(c_T) - f_\theta(c_S) = \underbrace{[f_\phi(c_T) - f_\theta(c_T)]}_{\text{model mismatch}} + \underbrace{[f_\theta(c_T) - f_\theta(c_S)]}_{\text{input perturbation}}$$

Triangle inequality：

$$\|f_\phi(c_T) - f_\theta(c_S)\|_\infty \leq \|f_\phi(c_T) - f_\theta(c_T)\|_\infty + L_\theta \cdot \Delta(g)$$

代入 Lemma 1 得 bound。$\square$

### 4.5 Interpretation：为什么 same-model 严格更优

比较两个 bound：
- Same-model: $\frac{L_\theta^2 \Delta(g)^2}{2}$
- Cross-model: $\frac{(L_\theta \Delta(g) + \text{mismatch})^2}{2}$

Cross-model bound 多了一个 additive model-mismatch term $\|f_\phi(c_T) - f_\theta(c_T)\|_\infty$，这个 term 只有当两个 model 在相同 privileged input 上产生 identical logits 时才 vanish，实际上不可能。

对 drifting teacher（teacher 共享 student 当前 weights），mismatch term 全程为 0。对 frozen teacher（初始 weights），mismatch term 非零，bound 接近 cross-model regime。

**Intuition**：same-model privileged distillation 的 elegance 在于，teacher 和 student 之间的 distributional gap 完全归因于"看到了 vs 没看到 ground truth"这件事本身，没有 model 能力差异这个 confounding factor。这让 distillation target 变得"可实现"——student 通过适当的参数更新，理论上可以 match teacher 分布，gap 由 information content 单独决定。

## 5. Proposition 2 详解：R=1 Filtering = RL-Optimal Policy

### 5.1 KL-Regularized RL Objective

$$\max_\pi \mathbb{E}_{\tau \sim \pi}[R(\tau)] - \beta \cdot KL(\pi \| \pi_{ref})$$

变量：
- $\pi$: 待优化的 policy
- $\pi_{ref}$: reference policy（防止偏离太远）
- $\beta > 0$: KL penalty 强度
- $R(\tau) \in \{0,1\}$: binary reward
- $\tau$: trajectory

这是 PPO/RLHF 的标准 objective。$\beta \to \infty$ 时 policy 紧贴 reference；$\beta \to 0$ 时 policy 完全 maximize reward（可能剧烈偏离 reference）。

### 5.2 Optimal Solution 的 Closed Form

由 Lagrangian / calculus of variations（Rafailov et al. 2023, DPO paper）：

$$\pi^*(\tau) = \pi_{ref}(\tau) \cdot \frac{\exp(R(\tau)/\beta)}{Z(\beta)}$$

变量：
- $Z(\beta)$: partition function，归一化常数

对 binary reward $R(\tau) \in \{0, 1\}$：

$$Z(\beta) = P_{\pi_{ref}}(R=0) \cdot 1 + P_{\pi_{ref}}(R=1) \cdot \exp(1/\beta) = (1-p) + p \cdot \exp(1/\beta)$$

其中 $p = P_{\pi_{ref}}(R=1)$ 是 reference 生成 correct trajectory 的概率。

### 5.3 Hard-Threshold Limit $\beta \to 0^+$

对 incorrect trajectory $R(\tau) = 0$：

$$\pi^*(\tau) = \frac{\pi_{ref}(\tau) \cdot 1}{Z(\beta)} \to 0 \quad \text{as } \beta \to 0^+$$

因为 $Z(\beta) \sim p \exp(1/\beta) \to \infty$，而分子固定为 $\pi_{ref}(\tau)$。

对 correct trajectory $R(\tau) = 1$：

$$\pi^*(\tau) = \frac{\pi_{ref}(\tau) \cdot \exp(1/\beta)}{p \cdot \exp(1/\beta) + (1-p)}$$

分子分母同除 $\exp(1/\beta)$：

$$\pi^*(\tau) \to \frac{\pi_{ref}(\tau)}{p} = \frac{\pi_{ref}(\tau)}{P_{\pi_{ref}}(R=1)} = \pi_{ref}(\tau | R(\tau)=1)$$

即：

$$\pi^*(\tau) \xrightarrow{\beta \to 0^+} \pi_{ref}(\tau | R=1) = \frac{\pi_{ref}(\tau) \cdot \mathbb{1}[R(\tau)=1]}{P_{\pi_{ref}}(R=1)}$$

**Intuition**：在 hard-threshold limit，KL penalty 完全消失，optimal policy 就是 "reference policy 限制在 correct trajectories 上的条件分布"。这就是 rejection sampling from reference + filter R=1 的精确分布。

### 5.4 HDPO 的应用

在 **non-cliff prompts** 上：GRPO 的 policy gradient 直接优化这个 objective，$\pi_{ref}(\cdot | R=1)$ 有非平凡 support，gradient 有信号。

在 **cliff prompts** 上：$P_{\pi_{ref}}(R=1) \approx 0$，所以 $\pi_{ref}(\cdot | R=1)$ 退化（没有支持），rejection sampling 拿不到任何样本，gradient vanishes。

HDPO 的解决方案：把 $\pi_{ref}$ 换成 proxy $\pi_\theta(\cdot | x, g)$（注入 ground truth）。这个 proxy 在 correct trajectories 上有 non-degenerate support（因为看到答案后 model 能正确生成）。

新的 KL-regularized objective：

$$\max_\pi \mathbb{E}[R(\tau)] - \beta \cdot KL(\pi \| \pi_\theta(\cdot | x, g))$$

这个 objective 的 optimal solution $\pi_\theta(\cdot | x, g, R=1)$ 有 well-defined support。HDPO 用 R=1 rejection sampling from $\pi_\theta(\cdot | x, g)$ 来 finite-sample 估计这个 optimum。

**两个近似误差来源**：
1. **Proxy reference 替代真实 reference**: gap 由 Proposition 1 bound
2. **Sampling noise**: privileged pass rate 在难题上可能远小于 1

## 6. 实验结果分析

### 6.1 主实验（Table 1, 8×H200）

| Method | pass@1 | pass@4 | pass@8 |
|--------|--------|--------|--------|
| GRPO Baseline | **0.6519** | 0.7749 | 0.8228 |
| HDPO (frozen, λ=0.01) | 0.6519 | 0.7812 | 0.8218 |
| HDPO (frozen, λ=0.1) | 0.6304 | 0.7812 | **0.8398** |
| HDPO (drifting, λ=0.01) | 0.6514 | **0.7861** | 0.8271 |
| HDPO (drifting, λ=0.1) | 0.6294 | 0.7856 | 0.8364 |

关键观察：

**Observation 1: λ=0.01 时保 pass@1，提升 coverage**
- Drifting, λ=0.01: pass@4 +1.1%, pass@8 +0.4%, pass@1 几乎不变（0.6514 vs 0.6519）
- 这是 "free lunch" regime：distillation 是 gentle nudge，不破坏 greedy decoding

**Observation 2: λ=0.1 时显著提升 pass@8 但牺牲 pass@1**
- 两种 teacher 都达到 ~0.84 pass@8（+1.4-1.7%）
- pass@1 下降 ~2.3-2.8%
- 这是 explicit exploration-exploitation tradeoff

**Observation 3: Drifting vs Frozen**
- λ=0.01 时 drifting > frozen（pass@4 上明显）
- λ=0.1 时差距缩小，frozen 在 pass@8 上最高（0.8398）
- 解释：drifting 的 realizability gap 小；frozen 保留初始 model 的 diversity（还没被 RL mode-seeking 塑形）

### 6.2 Hardware Variation (Table 3, 8×H100)

| Method | pass@1 | pass@4 | pass@8 |
|--------|--------|--------|--------|
| GRPO Baseline | 0.6509 | 0.7739 | 0.8223 |
| HDPO (frozen, λ=0.01) | 0.6484 | 0.7773 | 0.8252 |
| HDPO (frozen, λ=0.1) | 0.6343 | 0.7856 | **0.8369** |
| HDPO (drifting, λ=0.01) | 0.6499 | 0.7783 | 0.8213 |
| HDPO (drifting, λ=0.1) | 0.6343 | 0.7832 | 0.8359 |

定性一致：λ=0.1 提升 pass@8 +1.4-1.5%；λ=0.01 提升幅度小，可能接近 noise floor（floating-point non-determinism 跨 GPU microarchitecture 累积）。

### 6.3 pass@1 vs pass@k Tradeoff 的直觉

论文给了一个重要的 hypothesis：

> When the privileged model solves a problem via multiple distinct strategies, the JSD loss trains the student to place mass on all of them. For a small model, limited capacity means these modes compete for the same parameters: the model cannot maintain multiple fully coherent reasoning strategies simultaneously. The result is a flatter distribution where no single strategy dominates cleanly, degrading greedy accuracy (pass@1), while the broader support means additional samples discover genuinely different approaches (improving pass@k).

**Intuition**: 理想分布是"一个 dominant mode + 一条 long tail"。纯 RL 通过 mode-seeking 自然产生这个 shape，但 cliff 上找不到 mode。HDPO 先 broaden strategy support，后续 RL 可以 sharpen dominant mode，同时保留 secondary strategies 作为 long tail。这就是论文 Section 6 提出的 "expand-then-sharpen" curriculum 的动机。

## 7. 为什么用 JSD 而不是 Reverse KL？

这个选择直接关联到 Li et al. 2025 的 diversity collapse insight：https://arxiv.org/abs/2509.07430

RLVR fine-tuning 普遍观察到的现象：pass@1 提升但 pass@k 下降，policy collapse 到 single mode。

Divergence 选择的影响：
- **Reverse KL** $D_{KL}(P\|Q)$: mode-seeking。student 倾向集中在 teacher 的高密度 region，导致 mode collapse
- **Forward KL** $D_{KL}(Q\|P)$: mode-covering。student 必须 cover teacher 的所有 modes，包括低密度的
- **JSD** $JSD(P\|Q)$: 介于两者之间，但更 mode-covering 倾向

DPH-RL (Li et al. 2025) 用 forward KL / JSD 替代 reverse KL 来维持 broad solution coverage。HDPO 直接用 JSD，确保 distillation 不 collapse 到 single mode，而是覆盖 teacher 的多种解题策略。

参考资料：GKD (Agarwal et al. 2024) 提供 divergence choice 灵活性 https://arxiv.org/abs/2306.13649

## 8. 与相关工作的深入对比

### 8.1 Self-Distillation 家族

**OPSD (Zhao et al. 2026)** https://arxiv.org/abs/2601.18734
- 单个 LLM 同时作为 teacher 和 student
- Teacher 条件在 privileged info（verified reasoning traces 或 ground truth solutions）
- Student 只看到问题
- 在 student 的 on-policy rollouts 上最小化 per-token divergence
- HDPO 不同点：只在 cliff prompts 上 distill，用 R=1 filtering 选最优 target

**SDPO (Hübotter et al. 2026)** https://arxiv.org/abs/2601.20802
- 把 rich textual feedback（runtime errors, judge evaluations）转成 dense signal
- 蒸馏 feedback-conditioned predictions 回 unconditional policy
- 解决 credit-assignment bottleneck
- HDPO 不同点：用 ground truth 而非 feedback；target cliff prompts

### 8.2 Unified KD + RL

**KDRL (Xu et al. 2025)** https://arxiv.org/abs/2506.02208
- 同时 minimize reverse KL between student and teacher + maximize expected reward
- 发现 combination 比 single objective 更好
- HDPO 不同点：用 JSD 而非 reverse KL，且只在 cliff 上 trigger

**RLAD (Zhang et al. 2026)** https://arxiv.org/abs/2602.22495
- 把 importance ratio 换成 old policy 和 teacher 的 geometric mixture
- 直接 embed teacher 到 RL policy update
- HDPO 不同点：separate loss term，不修改 importance ratio

**G-OPD (Yang et al. 2026)** https://arxiv.org/abs/2602.12125
- 理论上证明 on-policy distillation 是 dense KL-constrained RL 的特例
- 引入 reward extrapolation 让 student 超越 teacher
- HDPO 不同点：focus 在 cliff 上，用 R=1 hard filter 实现 $\beta \to 0$ limit

### 8.3 Cliff Problem 的其他解法对比

| 方法 | 核心机制 | 复杂度 | 是否直接 learn cliff |
|------|---------|--------|---------------------|
| VCRL (Jiang et al. 2025) https://arxiv.org/abs/2509.19803 | variance-based curriculum | 调度器 | 否（跳过 cliff） |
| DAPO (Yu et al. 2025) https://arxiv.org/abs/2503.14476 | filter zero-variance prompts | filter + oversample | 否（跳过 cliff） |
| Scaf-GRPO (Zhang et al. 2025b) https://arxiv.org/abs/2510.19807 | tiered in-prompt hints | hint 设计 + fade schedule | 是（通过 hint） |
| HINT (Wang et al. 2025) https://arxiv.org/abs/2510.09388 | targeted guidance | guidance 策略学习 | 是 |
| EvoCoT (Liu et al. 2025) https://arxiv.org/abs/2508.07809 | self-generate + verify CoT | constraint + expansion | 部分 |
| Retrospective Replay (Dou et al. 2025) https://arxiv.org/abs/2504.14363 | replay early trajectories | buffer 管理 | 间接 |
| RLEP (Zhang et al. 2025a) https://arxiv.org/abs/2507.07451 | blend verified trajectories | buffer + sampling | 间接 |
| Le et al. 2025 https://arxiv.org/abs/2509.21880 | entropy-scaled gradient | advantage shaping | 是（信号弱） |
| PRIME (Cui et al. 2025) https://arxiv.org/abs/2502.01456 | online process reward model | 训练 PRM | 部分 |
| ReLIFT (Ma et al. 2025) https://arxiv.org/abs/2506.07527 | interleaved RL + SFT | 两阶段 + external solutions | 是 |
| **HDPO** | privileged self-distillation | **单 forward pass + JSD** | **是** |

HDPO 的 elegance：用最少的额外复杂度（append ground truth, generate, filter, JSD）实现 cliff 上的 learning signal，且有理论保证。

### 8.4 与 π-Distill 的联系

π-Distill (Penaloza et al. 2026) https://arxiv.org/abs/2602.04942 独立得到相同的 objective 结构：

$$J_{Teacher} = \mathbb{E}[\tilde{R}] - \beta \cdot D_{KL}(\pi_T(\cdot|x, I) \| sg[\pi_S(\cdot|x)])$$

其中 $\pi_{ref} = sg[\pi_S]$（stop-gradient student）。

不同点：π-Distill 在 finite $\beta$ 下对所有 prompts 做 gradient ascent；HDPO 在 $\beta \to 0$ limit 用 hard R=1 filtering 机械实现，只在 cliff prompts 上 trigger（gradient-based optimization fails 的地方）。

## 9. Architecture 图解析

```
┌─────────────────────────────────────────────────────────────────┐
│                        HDPO Training Step                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Prompt Batch B ──────► Generate K rollouts per prompt          │
│       │                  y^(k) ~ π_θ(·|x)                       │
│       │                        │                                │
│       │                        ▼                                │
│       │              ┌──────────────────┐                       │
│       │              │ Score R(x, y^k)  │                       │
│       │              └──────────────────┘                       │
│       │                        │                                │
│       │              ┌─────────┴─────────┐                      │
│       │              │                   │                      │
│       │     [Non-cliff: mixed R]   [Cliff: all R=0]              │
│       │              │                   │                      │
│       │              ▼                   ▼                      │
│       │     ┌──────────────┐    ┌───────────────────┐           │
│       │     │ L_GRPO       │    │ Privileged Gen:   │           │
│       │     │ (standard    │    │ ȳ ~ π_θ(·|x⊕y*)   │           │
│       │     │  policy grad)│    │                   │           │
│       │     └──────────────┘    │ Filter: R(x,ȳ)=1 │           │
│       │              │         └─────────┬─────────┘           │
│       │              │                   │                      │
│       │              │                   ▼                      │
│       │              │         ┌────────────────────┐            │
│       │              │         │ L_JSD = (1/N_tok)  │            │
│       │              │         │  Σ JSD_k(π_T||π_θ) │            │
│       │              │         │  over (x,ȳ) ∈ T   │            │
│       │              │         └─────────┬──────────┘            │
│       │              │                   │                      │
│       └──────────────┴───────────────────┘                      │
│                        │                                        │
│                        ▼                                        │
│         L_HDPO = L_GRPO + λ · L_JSD                             │
│                        │                                        │
│                        ▼                                        │
│               θ ← θ - α ∇L_HDPO                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

关键架构特征：
- **Shared weights**: $\pi_T$ 和 $\pi_\theta$ 是同一个 $\theta$，只是输入 prefix 不同
- **Single forward pass overhead**: privileged generation 是唯一的额外开销
- **No auxiliary models**: 没有 PRM, no critic, no external teacher
- **Conditional trigger**: 只在 cliff prompts 上 activate distillation

## 10. 深层 Intuition 与联想

### 10.1 Cliff Problem 的信息论视角

Cliff problem 本质上是 **reference policy 在 correct trajectory space 上 entropy 为 0** 的极端情形。从信息论看：

- $\pi_{ref}(\cdot | R=1)$ 的 entropy $H(\pi_{ref}(\cdot|R=1)) = H(\pi_{ref}) - H(\pi_{ref}, R) \cdot$ 之类
- 当 $P_{\pi_{ref}}(R=1) \to 0$，conditional distribution 不再 well-defined
- 这相当于 "no information about correct solutions in the reference"

HDPO 通过 ground truth 注入，相当于在 input side 引入额外信息 $\Delta(g)$（用 Proposition 1 的符号），这个 information content 直接转化为 reference 在 correct space 上的 non-degenerate support。Information injection → distributional support expansion → effective learning target。

### 10.2 Self-Distillation 与 Mean Teacher 的联系

Drifting teacher 类比 semi-supervised learning 中的 Mean Teacher (Tarvainen & Valpola 2017)：https://arxiv.org/abs/1703.01780

- Mean Teacher: 用 student 当前 weights 的 EMA 作为 teacher，提供 consistency loss
- HDPO drifting: 用 student 当前 weights（无 EMA），不同 input 作为 teacher
- 共同点：teacher 和 student 同源，gap 来自某种 "perturbation"（mean teacher 是 weight perturbation，HDPO 是 input perturbation）

Frozen teacher 类比知识蒸馏中的 fixed teacher，但仍有 input-perturbation gap 而非 architecture gap。

### 10.3 与 DPO 的有趣联系

DPO (Rafailov et al. 2023) https://arxiv.org/abs/2305.18290 直接 optimize KL-regularized RL objective 的 closed form：

$$\pi^*(\tau) = \pi_{ref}(\tau) \cdot \frac{\exp(R(\tau)/\beta)}{Z(\beta)}$$

DPO 推导出 loss 形式，避免显式 RL。HDPO 的 Proposition 2 用 DPO 的 closed form 推出 hard-threshold limit $\beta \to 0$：optimal = $\pi_{ref}(\cdot|R=1)$，机械实现就是 R=1 rejection sampling。

可以说 HDPO 在 cliff prompts 上用 rejection sampling 代替了 gradient-based optimization，因为 cliff 上 gradient fails，但 sampling 在 privileged proxy 上可行。这是 RLVR + rejection sampling 的 hybrid。

### 10.4 Curriculum 的 Expand-then-Sharpen 联想

论文 Section 6 提出 expand-then-sharpen curriculum：
1. HDPO 阶段：broaden strategy support on cliff prompts
2. RL 阶段：sharpen dominant mode，保留 long tail

这个 cycle 类比：
- **Mixture of Experts 训练**: 先 expand experts（diversity），再 sharpen gating（specialization）
- **Simulated annealing**: 高温 explore，低温 exploit
- **GAN 训练 dynamics**: generator diversity vs discriminator sharpness

更深层联想：这跟 biological evolution 中的 speciation + selection 类似——先 broaden genotypes（distillation 引入新 modes），再 sharpen via selection（RL mode-seeking）。

### 10.5 Capacity Bottleneck 的 Open Question

论文承认 1.5B 模型上 pass@1 vs pass@k tradeoff 明显。Hypothesis: 更大模型 capacity 充足，可能不损失 pass@1。

这与 mixture model capacity 分析一致：
- 小 model：K modes 竞争 N parameters，每个 mode 分到 N/K
- 大 model：N >> K，每个 mode 独立 fit

如果这个 hypothesis 成立，HDPO 在大 model 上可能是 pure win——提升 coverage 不牺牲 greedy。这是 future work 的关键方向。

### 10.6 Ground Truth 的形式 Generalization

论文用 math（ground truth = solution），但 privileged information 形式可以 generalize：
- **Code generation**: ground truth = test cases passing → privileged = 看到 test cases
- **Scientific reasoning**: ground truth = 最终结论 → privileged = 知道结论
- **Tool use**: ground truth = 正确工具调用序列 → privileged = 看到调用 schema
- **Multi-step planning**: ground truth = goal state → privileged = 看到 goal

关键是只要 ground truth 可以 inject 进 prompt 且不破坏 solution 的 validity，HDPO 框架就适用。

### 10.7 与 Offline RL 的联系

HDPO 在 cliff prompts 上本质是 offline RL：从 privileged generation（off-policy data）+ R=1 filter 得到 demonstrations，然后蒸馏。这类似：
- **AWAC** (Nair et al. 2020): advantage-weighted regression from offline data
- **CQL** (Kumar et al. 2020): conservative Q-learning
- **IQL** (Kostrikov et al. 2022): implicit Q-learning

不同点：HDPO 的 "offline data" 是 on-the-fly 生成的（online distillation from privileged rollouts），且用 JSD 而非 regression。

### 10.8 Lipschitz Bound 的 Loose 但 Useful 性质

Proposition 1 的 bound 严格说很 loose：deep transformer 的 $L_\theta$ 可能很大（per-layer $O(C\sqrt{n})$，L 层 composition 指数级）。

但 paper Section A.5 的 Remark 关键洞察：**比较 同一 bound 下的两个 case，looseness 同样 affect 两者，不改变比较结论**。这是 inequality reasoning 的常见技巧——absolute magnitude 不准，但 relative comparison 仍然有效。

类似情况在 differential privacy 的 composition bound、generalization bound 等理论分析中常见：loose constants 不影响 qualitative 结论。

### 10.9 R=1 Filtering 与 Importance Sampling 的等价性

R=1 rejection sampling 在 binary reward 下实际上是 importance sampling 的 extreme form：

- Importance weight: $w(\tau) = \exp(R(\tau)/\beta) / Z(\beta)$
- $\beta \to 0$ limit: $w(\tau) \to \mathbb{1}[R(\tau)=1] / P(R=1)$
- 这就是 rejection sampling：接受概率 $\propto \mathbb{1}[R(\tau)=1]$

更精细的 view：HDPO 用 hard filter 是 $\beta \to 0$ limit，而 general $\beta$ 对应 soft filter（weighted sampling with $\exp(R/\beta)$）。未来工作可以探索 finite $\beta$ soft filter 是否更 smooth。

## 11. Limitations 与 Critical Analysis

### 11.1 Single Model Scale
只在 1.5B 参数上验证。Larger models：
- 可能 fewer cliffs（baseline 成功率高）→ HDPO 边际收益降低
- 或更大 capacity → coverage gain 不损失 pass@1
- 这是 open question

### 11.2 Single Dataset
只在 OpenMathInstruct-2 (MATH + GSM8K) 上验证。Generalization 到 code、logic、scientific reasoning 需要 experiment。

### 11.3 Computational Overhead
- Privileged generation: 一个额外 forward pass per cliff prompt
- Top-k teacher logits: 额外 logits 处理
- JSD loss: 额外 backward pass
- Overhead ∝ cliff prompts per step

对 small batch / many cliffs，overhead 可能显著。

### 11.4 Frozen vs Drifting 的 Tradeoff
实验显示 frozen teacher 在 λ=0.1 时 pass@8 最高，但 λ=0.01 时 drifting 更好。这暗示：
- Drifting: realizability gap 小，但 diversity 受 RL mode-seeking 侵蚀
- Frozen: diversity 高，但 realizability gap 大
- 理想可能是 EMA teacher（between frozen and drifting），未探索

### 11.5 Tail Correction 的 Validity
Top-64 logits renormalize 假设 teacher 分布 mass 集中在 top-64。对 rare token（e.g., 数字、变量名）可能 violated。Tail correction $P_{rest} \ln 2$ 假设 student rest mass 是 uniform，实际 student rest distribution 也 non-trivial。

更精确做法：full softmax JSD，但计算开销大。

## 12. Future Directions 的联想

### 12.1 Expand-then-Sharpen Curriculum
论文 Section 6 提议：
1. HDPO 阶段：expand strategy support on cliffs
2. Re-inject solved cliffs back to RL training
3. RL mode-seeking sharpen dominant mode

关键：re-injection 要 delay，让 distilled strategies 稳定编码后再 sharpen，避免 transient accessibility。

### 12.2 Multi-step Privileged Information
当前 privileged info 是 final answer。可以扩展到：
- **Step-level hints**: 提供 reasoning chain 的关键步骤
- **Subgoal decomposition**: 提供 subgoals
- **Constraint hints**: 提供 constraints（不变量、边界条件）

这对应 process reward model 思路，但用 privileged injection 而非显式 PRM。

### 12.3 Privileged Information 的 Curriculum
不一定每次注入完整 ground truth。可以：
- 早期：完整 ground truth（max privileged）
- 中期：partial hint（intermediate privileged）
- 后期：无 hint（vanilla RL）

这是 privileged info 的 fade-out curriculum。

### 12.4 与 Process Reward Models 结合
HDPO 在 outcome level 工作（R=1 filter）。可以与 process reward (Lightman et al. 2024) https://arxiv.org/abs/2305.20050 结合：
- Non-cliff prompts: process reward densify signal
- Cliff prompts: HDPO privileged distillation
- 两者 complementary

### 12.5 Cross-Model HDPO
虽然 same-model 理论上更优，但实践中可能用 stronger model 作为 teacher 在 cliff 上更 effective。Proposition 1 的 cross-model bound 给了 mismatch term，但 strong teacher 可能提供 higher quality trajectories 抵消 mismatch。

### 12.6 Theoretical Extensions
- Proposition 1: 从 KL 推广到 f-divergence
- Proposition 2: 从 binary reward 推广到 continuous reward
- Finite $\beta$ analysis: hard filter 是 $\beta \to 0$，soft filter 下的 optimal target

## 13. 总结：HDPO 的 Conceptual Elegance

HDPO 在我看来有几个层次的美：

**第一层（mechanism）**：用 ground truth 让 model 教自己解 cliff prompts。简单到令人怀疑——为什么之前没人这么干？

**第二层（theory）**：Proposition 1 证明 same-model 严格 tighter than cross-model；Proposition 2 证明 R=1 filter = KL-regularized RL optimal 在 $\beta \to 0$ limit。两个 proposition 把 mechanism 锚定在 well-established theory 上。

**第三层（philosophical）**：HDPO 实际上在做的是 **information-theoretic regularization**。Cliff prompts 上 reference policy 的 information content 是 0（没有正确 trajectory 的信息）。注入 ground truth 引入 $\Delta(g)$ 的 information content，转化为 reference 在 correct space 上的 entropy。Distillation 就是把这个 entropy 传递回 student 的参数。

**第四层（生态意义）**：HDPO 提示了一种新的 RL+distillation 范式——不是用更强 model 蒸馏，而是用同一个 model 在 privileged state 下蒸馏。这绕开了"必须有大 model teacher"的限制，让 self-improvement loop 可能在任何 scale 闭环。

**第五层（与 reasoning model 进化的联系）**：DeepSeek-R1、Qwen-Math 等 reasoning model 的 RL training 都遇到 cliff 问题。HDPO 提供了一个 principled solution。如果 expand-then-sharpen curriculum work，可能成为 self-improving reasoning model 的标准组件。

参考资料：
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen2.5-Math: https://arxiv.org/abs/2409.12122
- OpenMathInstruct-2: https://arxiv.org/abs/2410.01560

## 14. 一个 Personal Reflection

读完这篇 paper，几个想法：

1. **Simple ideas often win**: HDPO 的核心机制（append answer, generate, filter, distill）简单到一行能说完。但 simple 不等于 obvious——理论分析让 simple idea 变得 principled。

2. **Self-distillation 是 underexplored**: 传统 distillation 假设 stronger teacher。HDPO 展示 same-model + privileged info 也能 work，且有理论优势。这暗示很多 "strong teacher" 任务实际上可以用 "same model + privileged info" 替代。

3. **Theory + Practice 的 balance**: Proposition 1 和 2 都不是 tight bound（Lipschitz constant loose, hard-threshold 是 limit case），但它们 provide qualitative guarantees that guide design decisions。这是好的 theory 的标志——inform design 而非精确预测。

4. **Open question on scale**: 1.5B 上的结果虽然 positive，但 magnitude 不大（pass@8 +1.4-1.7%）。Scale 到 7B、70B 是否保持或放大？这是关键 empirical question。

5. **LLM-assisted research**: 论文 LLM Usage Statement 透明承认 Claude 参与了 math formalization、writing、brainstorming。这本身是 interesting 的 meta-observation——AI 协助的 AI 研究，recursive self-improvement 的一种 form。

HDPO 是一篇简洁、理论扎实、empirically validated 的工作。它解决的 cliff problem 是 RLVR 的 fundamental bottleneck，提出的 solution 优雅且 minimal。如果 expand-then-sharpen curriculum 在 future work 中验证有效，HDPO 可能成为 self-improving reasoning model 的 key component。
