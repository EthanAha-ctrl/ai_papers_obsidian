---
source_pdf: Agent Explorative Policy Optimization for.pdf
paper_sha256: 06e7b6929df0e7cdf2811e035803c993d4054dd2ce6a2e530d3ca6c1018fcf91
processed_at: '2026-07-18T03:44:40-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

<tool_call>边界的prefix
- $p(t_1^{src})$: 从prefix $t_1^{src}$ 开始的continuation的成功概率
- $q$: policy $p_\theta$选择tool use的概率
- $p^{tool}$: 每个tool-using rollout的平均成功率
- $N$: 采样数

关键机制：$q \to 1$ by construction，每个resampled continuation都一定是tool-using

## 4. Advantage calculation 的细节
公式(2): per-prefix GRPO advantage on resampled continuations
$$\hat{A}_k^{res}(t_1^{src}) = \frac{r_k^{res} - mean(\{r_j^{res}\}_{j=1}^K)}{std(\{r_j^{res}\}_{j=1}^K)}$$
- $K$: 每个prefix的resample数量
- $r_k^{res}$: 第k个resampled trajectory的binary reward
- 应用到continuation tokens $y_k^{res}$，prefix thinking tokens被masked

公式(3): binary recovery reward
$$r^{prefix}(t_1^{src}) = \mathbb{1}[\exists k \in \{1,...,K\}: r_k^{res} = 1]$$
- 只有至少一个resample成功时为1，否则为0
- 替换source rollout的原始（zero）reward

公式(4): per-prefix advantage for source
$$\hat{A}^{prefix}(t_1^{src}) = \frac{r^{prefix}(t_1^{src}) - mean(\{\tilde{r}_j\}_{j=1}^N)}{std(\{\tilde{r}_j\}_{j=1}^N)}$$
- $\tilde{r}_j$: source index时用$r^{prefix}$，其他用$r_j$
- 应用到source的prefix tokens，source continuation被masked

公式(5): combined AXPO loss
$$\mathcal{L}_{AXPO}(t_1^{src}) = \mathcal{L}_{clip}(t_1^{src}; \hat{A}^{prefix}(t_1^{src})) + \sum_{k=1}^K \mathcal{L}_{clip}(y_k^{res}; \hat{A}_k^{res}(t_1^{src}))$$
- 第一项：prefix上的loss，只用source
- 第二项：resampled continuations上的loss

公式(6): PPO-clip surrogate
$$\mathcal{L}_{clip}(y; \hat{A}) = \mathbb{E}_t[\min(\rho_t(\theta)\hat{A}, clip(\rho_t(\theta), 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}) - \beta D_{KL}[\pi_\theta(\cdot|x, y_{<t}) || \pi_{ref}(\cdot|x, y_{<t})]]$$
- $\rho_t(\theta)$: per-token importance ratio
- $\epsilon_{low} = 0.2, \epsilon_{high} = 0.4$: DAPO asymmetric clipping
- $\beta = 10^{-3}$: KL coefficient

## 5. 实验结果分析
- 9个multimodal benchmarks, 3个scale (2B/4B/8B)
- SFT + AXPO > SFT + GRPO at average
  - 2B: +1.1 pp Pass@1, +2.8 pp Pass@4
  - 4B: +1.4 pp Pass@1, +2.3 pp Pass@4
  - 8B: +1.8 pp Pass@1, +1.8 pp Pass@4
- 8B AXPO surpasses 32B Base on Pass@4 (75.8 vs 75.1)

## 6. 消融实验
所有组件都重要：
- 去掉prefix fix: 52.1 (接近GRPO baseline 51.9)
- 用no-tool rollout resampling: 51.7
- 去掉uncertainty rank: 51.0
- 去掉prefix credit: 51.4
- 去掉separated advantage group: 51.2
- Full AXPO: 53.9

## 7. 与其他RL方法对比
- Reward shaping (tool penalty/bonus): 都低于GRPO baseline
- Rollout 2×: 52.4 (still below AXPO 53.9)
- RLTF: 43.3
- CISPO: 47.8
- ARPO: 51.3
- AXPO: 53.9

让我把这些内容组织成一个深入的回答。我会尽量多用英文专业术语，用中文做逻辑连接。

我应该包括一些相关的研究和web links来扩展视野。让我想想相关的work：
- GRPO: https://arxiv.org/abs/2402.03300
- BroRL: https://arxiv.org/abs/2510.01180
- ARPO: https://openreview.net/forum?id=TX4k7BF6aO
- DAPO: https://arxiv.org/abs/2503.14476
- PPO: https://arxiv.org/abs/1707.06347
- ReAct: https://openreview.net/forum?id=WE_vluYUL-X
- SFT memorizes, RL generalizes: https://arxiv.org/abs/2501.17161
- DeepSeek-R1: https://arxiv.org/abs/2501.12948

让我开始写回答。我会用Karpathy的风格，build intuition, 技术细节丰富# AXPO 深度解析：Tool-call Resampling 如何修复 Agentic RL 的信号坍塌

Andrej, 这篇 paper 抓住了一个非常真实且被严重低估的问题：当我们把 GRPO 这类 group-relative RL 直接套用到 agentic reasoning 上时，tool use 这个行为会系统性地被"饿死"。让我从底层机制开始 build intuition，然后一层层剥开 AXPO 的设计。

---

## 1. Thinking-Acting Gap：为什么这不是一个普通的 RL 问题

### 1.1 Asymmetry 的本质

Thinking mode 之下，每一个 token 都在 policy $\pi_\theta$ 的完全控制之内——错误可以通过重新采样下一个 token 修正，整个 trajectory 是 self-contained 的。Tool use 完全不同：一个短的 JSON token sequence（`<tool_call>...</tool_call>`）触发一个外部 execution $\text{exec}(\mathbf{a}_t)$，返回一个 observation $\mathbf{o}_t$，这个 observation 的 distribution 完全在 policy 的 control 之外。一个 indent 错误、一个 bbox 坐标偏差 50 pixel、一个 search query 的措辞差异，都可能让整条 trajectory 从 correct 翻转到 wrong。

这种 asymmetry 在 RL 训练中具体表现为两个 quantifiable symptom（Figure 3）：

**Symptom 1: Tool use is under-attempted**  
在一个 group of $N=8$ rollouts 里，tool-using rollout 的比例全程卡在 20–35%，thinking-only 是 majority。GRPO 对 group 内的 minority 行为施加 zero pressure 去增长——因为 advantage 是 group-normalized 的，minority 的 contribution 被 majority 稀释。

**Symptom 2: Tool-using subgroup is disproportionately all-wrong**  
Conditional on a question where tool use is attempted，tool-using subgroup（group 内那些 emit 了 tool call 的 rollout）all-wrong rate ≈ 40%，而 no-tool subgroup 只有 25%。这意味着在 GRPO 的 advantage normalization 下：

- **Mixed group 场景**：no-tool rollouts 成功、tool-using rollouts 失败 → tool-call tokens 获得 negative advantage，**主动惩罚 tool use**
- **Fully all-wrong 场景**：$r_i - \text{mean}(\{r_j\}) = 0$ → advantage 归零，**tool-call tokens 拿不到任何梯度信号**

这两种情况加起来，tool-call tokens 在那些"最需要学 tool use"的问题上系统性拿到 non-positive advantage。这是 GRPO 在 agentic setting 下的 fundamental failure mode，而 paper 第 2.2 节用三个 diagnostic measurement 把它 ground 得很扎实。

### 1.2 为什么 naive rollout scaling 不能解决

直觉上，把 $N$ 从 8 调到 16、32，tool-using rollout 的绝对数量就会变多，signal 就会恢复。但 paper 在 Section 3.1 指出：tool-use rate $q \approx 0.3$，所以 raw rollout 里有 $(1-q) \approx 70\%$ 的 budget 浪费在 non-tool rollout 上，这些 rollout 对 tool-call tokens 的学习贡献为零。Table 3 的 "rollout 2×" 实验证证了这一点：把 rollout budget 翻倍只到 52.4 avg，仍然低于 AXPO 的 53.9，而且用了 4× 于 AXPO 的额外 compute。这是 paper 最有说服力的 ablation 之一。

参考 BroRL (https://arxiv.org/abs/2510.01180) 的 Theorem 1：scaling $N$ 减少 unsampled blind spots 的速率是 $p^2(1-p)^N$，对 high-$p$ region 衰减快，对 low-$p$ region 衰减慢。Tool-call tokens 恰好坐在 low-$p$ region（$q \cdot p^{tool}$ 很小），所以 naive scaling 是**结构性低效**的，而非 budget 不够。

---

## 2. AXPO 的核心机制：Tool-call Resampling

### 2.1 关键 insight：在哪里 resample

paper 的 key insight 可以一句话概括：**Thinking prefix 是 sound 的概率不低，bottleneck 在 tool call 本身，所以 resample 应该 fix 住 thinking prefix，只在 tool-call boundary 之后重采**。

这不是显然的。一个自然的疑问是：怎么知道 thinking prefix 不是 bottleneck？Figure 3c 给了 empirical 证据——fix 住一个 failed rollout 的 prefix $t_1^{src}$，从 $\pi_\theta(\cdot | \mathbf{x}, t_1^{src})$ 采 16 个 continuation，cluster 出来的 tool call semantic cluster 数量是 2.9–3.4 个。也就是说，**同一个 thinking prefix 下，policy 自己会 emit 多种语义不同的 tool call**。Tool call 是一个 under-explored divergence point，prefix 并没有 commit 到具体 action。

这个测量非常重要，它排除了"prefix 已经决定 action，resample 无意义"的反驳。

### 2.2 触发条件：只对 all-wrong tool-using subgroup 操作

AXPO 不盲目 resample 所有 tool-using prefix，只对满足两个条件的 subgroup 触发：
1. Tool-using subgroup non-empty（至少有一个 rollout 尝试了 tool）
2. Tool-using subgroup entirely wrong（没有 correct 的 tool-using rollout）

这两个条件下，tool-call tokens 在 GRPO 下拿到的 advantage 是 non-positive（zero 或 negative）。一个 recovered continuation 能把 advantage 从 non-positive 翻转到 positive，gradient lift per resampled continuation 最大。

paper 在 Section 3.2 明确指出：Proposition 1 原则上也允许有正确 tool-using rollout 的 group 参与 resampling，但那些 group 已经产生 positive advantage on tool-call tokens，marginal recovery 收益小。这是一个**budget allocation 的工程判断**，很合理。

### 2.3 Uncertainty-based prefix ranking

不是所有 admitted prefix 都值得 resample。Figure 3c 显示约 30% 的 prefix 在 16 次 resample 后 collapse 到单个 semantic cluster——意味着 policy 已经 converge 到一个 wrong commitment，再采也是同一个 wrong answer。

AXPO 用 mean policy probability over tool-call tokens 作为 uncertainty proxy（Section 3.2, Appendix B.9）：
- 低 confidence → policy 还在犹豫 → resample 有概率探索到不同 action
- 高 confidence → policy 已经 commit → resample 浪费 budget

Appendix B.9 验证了这个 proxy 和 exact policy entropy 高度相关（Pearson $\rho = 0.843$, Spearman $\rho = 0.835$），而且 vLLM 直接输出 per-token probability，不需要额外 forward pass——这是一个很实用的工程选择。FSDP 全 vocabulary forward pass 来算 entropy 会破坏 verl 的 rollout/gradient 交替 pipeline。

---

## 3. Proposition 1 的技术细节

让我把公式(1)的每个变量和 intuition 讲透：

$$\underbrace{1 - (1 - p(t_1^{src}))^N}_{\text{resampling: } N \text{ tool-using continuations}} \geq \underbrace{1 - (1 - q p^{tool})^N}_{\text{raw: } N \text{ rollouts, } q \text{ fraction tool-using}}$$

**变量定义**：
- $t_1^{src}$: 一个已经包含 opening `<tool_call>` tag 的 thinking prefix，即已经跨越 tool-call boundary
- $p(t_1^{src}) := \Pr_{\pi_\theta}[R = 1 | \mathbf{x}, t_1^{src}]$: 从这个 prefix 开始采 continuation 的成功概率
- $q := \Pr_{\pi_\theta}[\text{tool use} | \mathbf{x}]$: policy 在这个问题上 emit tool call 的概率
- $p^{tool} := \mathbb{E}_{t_1 \sim \pi_\theta(\cdot | \mathbf{x}, t_1 \in \mathcal{T})}[p(t_1)]$: 每个 tool-using rollout 的平均成功率
- $N$: 采样数

**两边都在测同一个事件**：group 里至少有一个 correct tool-using rollout 的概率。

**Left side (resampling)**: $N$ 个 continuation 全部 tool-using by construction（因为 prefix 已经过了 `<tool_call>` boundary），每个成功的概率是 $p(t_1^{src})$，全 fail 的概率是 $(1 - p(t_1^{src}))^N$，至少一个 success 的概率是 $1 - (1 - p(t_1^{src}))^N$。

**Right side (raw)**: $N$ 个 rollout 每个 tool-using 的概率是 $q$，conditional on tool-using 成功概率是 $p^{tool}$，所以每个 rollout 既 tool-using 又 correct 的概率是 $q \cdot p^{tool}$，至少一个 correct tool-using 的概率是 $1 - (1 - q p^{tool})^N$。

**关键机制**：resampling 把 $q \to 1$ by construction，消除了 raw sampling 的 $(1-q)$ waste factor。阈值从 $p^{tool}$ 降到 $q p^{tool}$，所以**任何 success probability $\geq q p^{tool}$ 的 prefix 都满足不等式**。这意味着即使挑了一个 below-average 的 prefix（$p(t_1^{src}) < p^{tool}$），只要它的 success probability 不低于 $q p^{tool}$，resampling 仍然 dominate raw sampling。

**严格不等式条件**：$p(t_1^{src}) > q p^{tool}$ 且 $q p^{tool} \in (0, 1)$。前者是说 prefix 稍微好于"raw sampling能触及的 floor"，后者是说 raw sampling 不是 trivially 100% 成功。

**对 GRPO 的 consequence**：在 binary reward 下，$1 - (1 - p_{eff})^N$ 就是 group 产生 positive (reinforcing) gradient on tool-call tokens 的概率。Proposition 1 说 resampling 在任何 fixed $N$ 下严格 break all-wrong tool-using subgroups。

**为什么 threshold 一定满足**：由 $p^{tool}$ 的定义，$\mathbb{E}_{t_1}[p(t_1)] = p^{tool} > q p^{tool}$ 只要 $q < 1$。所以期望意义上的 prefix 就在 threshold 之上，大多数实际 prefix 都满足假设。Empirical 上，fixing 一个 failed rollout 的 prefix 并 resample tool call，能 recover $\sim 15\%$ 的原本 all-wrong tool-using subgroup。

---

## 4. Advantage Calculation：解耦两条 gradient stream

这是 AXPO 最精妙的设计。如果直接把 resampled trajectories 当成 independent rollouts 和 source rollout 一起塞进 GRPO，shared prefix tokens 会收到 contradictory advantage：failed source 给 prefix 负信号，successful resample 给 prefix 正信号，互相抵消。

AXPO 的解法是**把 advantage stream 彻底拆开**，每个 token 只从一个 source 拿梯度。

### 4.1 Resampled continuations 的 advantage（公式 2）

$$\hat{A}_k^{res}(t_1^{src}) = \frac{r_k^{res} - \text{mean}(\{r_j^{res}\}_{j=1}^K)}{\text{std}(\{r_j^{res}\}_{j=1}^K)}, \quad k = 1, \ldots, K$$

- $K$: 每个被选中 prefix 的 resample 数量（paper 用 $K=4$）
- $r_k^{res} \in \{0,1\}$: 第 $k$ 个 resampled trajectory $\tau_k^{res} = (t_1^{src}, y_k^{res})$ 的 binary outcome reward
- $\hat{A}_k^{res}$: 在 $K$ 个 resample 内部做 GRPO-style normalization

**关键**：这个 advantage 只应用到 continuation tokens $\mathbf{y}_k^{res}$（tool call + tool output + 后续 thinking + answer），prefix thinking tokens 被 mask 掉。这就避免了 prefix 上 conflicting signal 的问题。

### 4.2 Source prefix 的 recovery reward（公式 3 和 4）

$$r^{prefix}(t_1^{src}) = \mathbb{1}[\exists k \in \{1, \ldots, K\} : r_k^{res} = 1]$$

这是一个**binary recovery indicator**：只要 $K$ 个 resample 里至少有一个成功，就给 source prefix reward 1，否则 0。这替换了 source rollout 原本的 reward（all-wrong 场景下是 0）。

然后把它塞进 source group 的 GRPO normalization：

$$\hat{A}^{prefix}(t_1^{src}) = \frac{r^{prefix}(t_1^{src}) - \text{mean}(\{\tilde{r}_j\}_{j=1}^N)}{\text{std}(\{\tilde{r}_j\}_{j=1}^N)}, \quad \tilde{r}_j = \begin{cases} r^{prefix}(t_1^{src}), & j = \text{source index} \\ r_j, & \text{otherwise} \end{cases}$$

- $N$: 原始 GRPO group size（8）
- $\tilde{r}_j$: source index 位置用 recovery reward 替换，其他 rollout 用原始 reward
- $\hat{A}^{prefix}$: 在 modified group 上重算的 advantage

**这个 advantage 只应用到 source trajectory 的 prefix tokens $t_1^{src}$**，source 的 continuation 被 mask 掉（因为 continuation 已经 fail 了，不应该再被 reinforce）。

### 4.3 合并 loss（公式 5）

$$\mathcal{L}_{AXPO}(t_1^{src}) = \underbrace{\mathcal{L}_{clip}(t_1^{src}; \hat{A}^{prefix}(t_1^{src}))}_{\text{prefix, source only}} + \sum_{k=1}^K \underbrace{\mathcal{L}_{clip}(y_k^{res}; \hat{A}_k^{res}(t_1^{src}))}_{\text{resampled continuation}}$$

每个被选中的 prefix 贡献两项：
1. Prefix 上的 PPO-clip loss，用 recovery advantage
2. $K$ 个 resampled continuations 上的 PPO-clip loss，用 per-prefix group advantage

### 4.4 PPO-clip surrogate（公式 6）

$$\mathcal{L}_{clip}(\mathbf{y}; \hat{A}) = \mathbb{E}_t\left[\min\left(\rho_t(\theta)\hat{A}, \text{clip}(\rho_t(\theta), 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}\right) - \beta D_{KL}[\pi_\theta(\cdot|\mathbf{x}, \mathbf{y}_{<t}) \| \pi_{ref}(\cdot|\mathbf{x}, \mathbf{y}_{<t})]\right]$$

- $\rho_t(\theta) = \pi_\theta(y_t | \mathbf{x}, \mathbf{y}_{<t}) / \pi_{\theta_{old}}(y_t | \mathbf{x}, \mathbf{y}_{<t})$: per-token importance ratio，当前 policy 相对于 rollout policy
- $\epsilon_{low} = 0.2, \epsilon_{high} = 0.4$: DAPO-style asymmetric clipping（https://arxiv.org/abs/2503.14476），lower bound 收紧防止 negative advantage 下过度 penalize，upper bound 放松允许 positive advantage 下更大 step
- $\beta = 10^{-3}$: KL penalty coefficient，anchor 在 SFT-initialized reference policy $\pi_{ref}$
- $\hat{A}$: broadcast 到 trajectory 的每个 token

**这个设计的精妙之处**：recovery indicator $r^{prefix}$ 是 monotone 的——只要有一个 resample 成功就 credit prefix。这把 "coverage gain"（resample 拓展了 correct trajectory 的可达集合）直接转化为 "gradient signal on prefix"。Policy 被明确地 incentivize 去产生那些"虽然这次 fail 但有可能 resample 成功"的 prefix，而不是只产生"一次就成功"的 prefix。这是一个非常聪明的 credit assignment。

---

## 5. 架构图解析

Figure 2 是 paper 的概念图，我把它拆解成 dataflow：

```
Input (x, image)
       │
       ▼
  ┌─────────────────────┐
  │ Standard GRPO group │
  │  N=8 rollouts       │
  └─────────────────────┘
       │
       ├── Thinking-only rollouts (majority, ~70%)
       │
       └── Tool-using rollouts (minority, ~30%)
                │
                ▼
       ┌────────────────────────┐
       │ Subgroup analysis      │
       │ Is tool-using subgroup │
       │ all-wrong?             │
       └────────────────────────┘
                │
        ┌───────┴───────┐
        │ No            │ Yes
        ▼               ▼
   Standard GRPO    AXPO trigger:
   advantage        1. Fix thinking prefix t_1^src
                   2. Rank by uncertainty (low confidence first)
                   3. Resample K=4 continuations
                      {y_k^res} ~ π_θ(·|x, t_1^src)
                   4. Execute tool calls, roll forward to <answer>
                          │
                          ▼
              ┌─────────────────────────┐
              │ Advantage decomposition  │
              │ • Source prefix ←        │
              │   recovery indicator     │
              │ • Resampled cont. ←      │
              │   per-prefix GRPO       │
              └─────────────────────────┘
                          │
                          ▼
              Combined PPO-clip loss
              (Eq. 5) + standard GRPO loss
              on non-triggered groups
```

Figure 4 则是 training dynamics 的实证：AXPO 训练过程中 tool-use rate 单调上升（+28 pp），all-wrong rate 比 GRPO 低 17 pp，per-step recovery rate ≈ 12%。这三个指标共同验证了"AXPO 在 training 时真的把 dead subgroup 救活了"。

---

## 6. 实验数据深度分析

### 6.1 Main Results (Table 1, Pass@1)

| Model | Method | MathVision | DynaMath | Math-VR | V* | VisualProbe | HR-Bench-4K | HR-Bench-8K | HR-MMSearch | MM-Search | Avg |
|-------|--------|-----------|----------|---------|-----|------------|-------------|-------------|-------------|-----------|-----|
| 8B | SFT+GRPO | 55.3 | 78.2 | 60.4 | 87.7 | 40.1 | 79.5 | 74.9 | 24.4 | 44.0 | 60.5 |
| 8B | SFT+AXPO | 56.1 | 79.0 | 60.6 | 87.8 | 45.8 | 83.3 | 77.0 | 25.9 | 45.0 | **62.3** |
| Δ | | +0.8 | +0.8 | +0.2 | +0.1 | **+5.7** | **+3.8** | **+2.1** | +1.5 | +1.0 | +1.8 |
| 32B | Base | 56.5 | 83.3 | 64.1 | 89.1 | 40.3 | 85.2 | 78.9 | 22.8 | 46.1 | 62.9 |

观察：
1. **8B AXPO (62.3) 几乎追平 32B Base (62.9) Pass@1**，在 Pass@4 上反超（75.8 vs 75.1）
2. **增益集中在 Perception category**：VisualProbe +5.7, HR-Bench-4K +3.8, HR-Bench-8K +2.1。这些 benchmark 的 bottleneck 是 image zoom-in 这个 tool call 本身，tool-using rollout 在 GRPO 下大量 all-wrong
3. **Reasoning category 增益小**：MathVision +0.8, DynaMath +0.8。这些 benchmark 的 bottleneck 在 thinking 而不是 tool call，AXPO 的 targeted intervention 不在这里
4. **Search category 有增益但不夸张**：HR-MMSearch +1.5，因为 multi-hop search 的 tool call 确实是 failure locus

这个 pattern 非常符合 Thinking-Acting Gap 的诊断：哪里 tool call 是 bottleneck，AXPO 就在哪 gain；哪里 thinking 是 bottleneck，AXPO gain 小（但没 regress）。

### 6.2 Pass@4 Results (Table 5)

Pass@4 的 gain 更能说明 AXPO 扩展了"reachable correct trajectory set"，而不仅仅 sharpen：

- 2B: +2.8 pp (66.8 → 69.6)
- 4B: +2.3 pp (71.7 → 74.1)
- 8B: +1.8 pp (74.0 → 75.8)

Pass@4 > Pass@1 gain 在每个 scale 都成立，这是 AXPO 操作在不同 axis 的 signature——SFT-then-GRPO 只 sharpen within existing ceiling，AXPO 同时 expand ceiling 和 sharpen。

### 6.3 Tool Utilization (Table 4)

8B 上的 tool utilization：
| Benchmark | SFT+GRPO | SFT+AXPO | Δ |
|-----------|----------|----------|---|
| MathVision | 46.1 | 75.7 | +29.6 |
| DynaMath | 30.7 | 63.9 | +33.2 |
| Math-VR | 7.6 | 29.1 | +21.5 |
| V* | 92.1 | 99.0 | +6.9 |
| VisualProbe | 99.1 | 100.0 | +0.9 |
| HR-Bench-4K | 73.0 | 90.0 | +17.0 |
| HR-Bench-8K | 78.0 | 92.0 | +14.0 |
| HR-MMSearch | 97.0 | 98.4 | +1.4 |
| MM-Search | 97.7 | 98.3 | +0.6 |

在 tool-saturated benchmark (V*, VisualProbe, MM-Search) AXPO 几乎不动 utilization（已经 ~100%），但在 under-attempted benchmark (MathVision, DynaMath) AXPO 把 utilization 拉高 30+ pp。这和 Figure 5a 的"AXPO 同时推进 tool-attempt rate 和 conditional pass@1 两个轴"完全一致——其他阶段只能在两轴之间 trade-off，只有 AXPO 能同时推进。

### 6.4 Ablation (Table 2)

这是理解 AXPO 每个设计 choice 的关键：

| Variant | Avg P@1 | Avg P@4 | 相对 AXPO 退化 |
|---------|---------|---------|----------------|
| AXPO w/o prefix fix (from scratch) | 52.1 | 69.1 | -1.8 / -1.1 |
| w/ no-tool rollout resampling | 51.7 | 65.9 | -2.2 / -4.3 |
| w/o uncertainty rank | 51.0 | 65.8 | -2.9 / -4.4 |
| w/o prefix credit | 51.4 | 67.7 | -2.5 / -2.5 |
| w/o separated advantage group | 51.2 | 67.0 | -2.7 / -3.2 |
| **Full AXPO** | **53.9** | **70.2** | — |
| SFT+GRPO baseline | 51.9 | 68.3 | -2.0 / -1.9 |

逐行分析：

1. **w/o prefix fix**：退化到 52.1，接近 GRPO baseline。每个 resample 变成 from-scratch rollout，$(1-q)$ waste factor 回来，Proposition 1 的优势消失。这验证了 prefix fixing 是 AXPO 的核心。

2. **w/ no-tool rollout resampling**：退化到 51.7，比 GRPO 还差。因为 no-tool subgroup 在 GRPO 下已经 cover 了，再 resample 无意义；而且浪费了本该用于 tool-using subgroup 的 budget。

3. **w/o uncertainty rank**：退化到 51.0，是最大退化。Random ranking 把 budget 路由到 policy 已经 converge 到 wrong commitment 的 prefix 上，resample 采到的还是同一个 wrong answer。这验证了"不是所有 prefix 都值得 resample"，uncertainty 是有效的 filter。

4. **w/o prefix credit**：退化到 51.4。Prefix tokens 拿不到 positive signal 即使 resample 成功，policy 不会被 steer 去产生 high-yield prefix。这验证了 recovery indicator 的 credit assignment 是必要的。

5. **w/o separated advantage group**：退化到 51.2。Shared prefix 收到 contradictory advantage（source fail 给负、resample success 给正），同时 no-tool success 在 source group 里稀释 recovered signal。这验证了 advantage stream 解耦的必要性。

这五个 ablation 加起来，清晰说明 AXPO 不是"任何 trick 都行"，而是一个**每一个设计 choice 都 load-bearing 的组合**。

### 6.5 对比其他 RL recipe (Table 3)

| Method | Avg P@1 | Avg P@4 |
|--------|---------|---------|
| GRPO baseline | 51.9 | 68.3 |
| + Tool penalty | 46.0 | 59.1 |
| + Tool bonus | 50.8 | 65.6 |
| + rollout 2× | 52.4 | 68.2 |
| RLTF | 43.3 | 56.0 |
| CISPO | 47.8 | 62.2 |
| ARPO | 51.3 | 65.6 |
| **AXPO** | **53.9** | **70.2** |

几个关键对比：

**vs reward shaping**: Tool penalty 把 utilization 压下去（penalize tool use），tool bonus 鼓励无差别 tool use 但不区分 helpful/unhelpful call。两者都没解决 coverage 问题，反而干扰 signal。AXPO 不动 reward，只动 rollout distribution。

**vs rollout 2×**: 这是最直接的 "more compute" control。2× budget 只到 52.4，AXPO 用 25% 额外 budget 到 53.9。"gain comes from where compute is spent, not from how much" 这句话是 paper 最犀利的 claim。

**vs ARPO**: ARPO (https://openreview.net/forum?id=TX4k7BF6aO) 是最接近的竞品，它 branch 在 tool observation 之后（post-observation entropy 高时 resample continuation）。但 ARPO 无法 recover "tool call 本身 wrong" 的场景——wrong sub-image bbox、wrong python code、wrong search query 都已经 committed，后续 continuation 怎么采都救不回来。AXPO branch 在 tool-call boundary 之前，exploration 的对象是 tool call 本身。这 2.6 pp 的差距（51.3 vs 53.9）正好对应 "tool call 本身是 failure locus" 这个诊断。

**vs RLTF / CISPO**: RLTF (https://arxiv.org/abs/2602.02482) 用 text feedback 训练，但 hint leakage 导致 tool-call collapse（utilization 掉到 40%）。CISPO (https://arxiv.org/abs/2506.13585) 是 importance sampling 变种，修 gradient bias 但不碰 all-wrong subgroup 的 zero advantage 问题。

---

## 7. 为什么 SFT-then-RL pipeline 需要 AXPO（Appendix B.4 的深入解读）

这部分 paper 论证得很精彩。把 4 个 baseline 排起来看：

| 阶段 | 作用 | Pass@4 ceiling 效果 |
|------|------|--------------------|
| Non-agent (no tools) | 内部 reasoning only | 最低 baseline |
| Base (tools, no training) | Inference-time tool access | 中小模型反而下降（agent prompt overhead > gain） |
| GRPO from base | RL without SFT | Pass@1 部分恢复，Pass@4 ceiling 被牺牲（collapse 到 narrow mode） |
| SFT only | 模仿 teacher trajectories | 拓宽 Pass@4 ceiling，但 exposure bias |
| SFT + GRPO | 标准 recipe | 在 SFT ceiling 内 sharpen，但不 expand |
| SFT + AXPO | 加 tool-call resampling | **同时 expand ceiling + sharpen** |

关键 insight：
- **SFT sets the Pass@4 ceiling**，因为它把 policy 的 cold-start distribution 拓宽到 tool-using trajectory 上，但 SFT 只能 imitate teacher 的 distribution
- **GRPO sharpens within the ceiling**，因为它在 policy 自己的 rollout 上做 group-relative normalization，但 GRPO 的 sharpening bias 到 SFT manifold
- **AXPO 同时 expand + sharpen**，因为它把 dead subgroup 救活，这些救活的 trajectory 是 SFT manifold 之外的新 reachable correct trajectory

Appendix B.4 引用 "SFT memorizes, RL generalizes" (https://arxiv.org/abs/2501.17161) 来说明 RL 在 SFT 之上的价值，但 paper 进一步指出 standard GRPO 的 generalization 被 Thinking-Acting Gap 限制，AXPO 是 generalization 的"放大器"。

---

## 8. Generalization to Unseen Tool (Appendix B.5)

这是一个很巧妙的实验。训练时只用 Python / Tavily search / image zoom-in 三个 tool，eval 时塞一个训练时没见过的 image search tool。

| Configuration | SFT+GRPO P@1 | SFT+AXPO P@1 |
|---------------|-------------|--------------|
| w/o image search | 44.0 | 45.0 |
| w/ image search | 46.8 (+2.8) | 49.6 (+4.6) |

AXPO 在 unseen tool 上 gain 比 GRPO 大 1.8 pp。Pass@4 两者 gain 接近（+6.0 vs +5.7），但 Pass@1 AXPO 显著更好。这说明 AXPO 训出来的 policy 不只是"会用更多 tool"，而是"更可靠地把 tool use 转化为 first-shot correct answer"。

我读这个结果的方式：AXPO 训练时 resample 让 policy 学到了"给定 thinking prefix，tool call 可以有多种 implementation"这种 meta-pattern，这个 pattern 在 unseen tool 上依然适用——policy 知道对同一个 intent 应该探索不同的 tool call realization。GRPO 训出来的 policy 只是 narrow 地 fit 到训练 tool inventory 的 specific call format 上。

---

## 9. 局限与可能的扩展方向

paper 自己在 Appendix E 提到了几个：

1. **Verifiable reward 依赖**：AXPO 的 trigger "all-wrong tool-using subgroup" 和 per-prefix advantage 都依赖 binary verifiable reward。Open-ended generation（story writing, code from spec）没有清晰 correctness signal，需要重新定义"failed subgroup"。但 resampling 机制本身是 reward-agnostic 的。

2. **8B 训练上限**：32B 是 inference-only baseline，不是 trainable。32B+ 的 Thinking-Acting Gap 是否更小（base policy 更强、tool-use ceiling 更高）是 open question。如果 32B 上 all-wrong subgroup fraction 已经很低，AXPO 的 marginal gain 可能衰减。

3. **Tool inventory 限制**：只有三个 tool family。长 latency tool（browser agent, GUI control）和高 per-call cost tool（LLM-as-tool）会引入新问题——长 latency 意味着 resample 的 wallclock cost 高，LLM-as-tool 意味着 reward noise 大。

我会额外加几个 paper 没提的：

4. **Multi-turn tool use**：paper 把 trajectory 限制在 3 turn。真实 agentic scenario 可能 10+ turn，第 $t$ turn 的 tool-call resampling 如何和第 $t-1$ turn 的 prefix 互动？Proposition 1 假设 single tool-call boundary，multi-turn 下需要 sequential resampling 或 trajectory-level tree search。

5. **Prefix selection 的更优 proxy**：mean policy probability 是 tractable 但粗糙。Predictive entropy 更精确但 cost 高。Might be worth exploring Monte Carlo entropy estimate 或 ensemble disagreement 作为中间方案。

6. **和 tree search 的关系**：AXPO 本质上是 MCTS 风格的 expansion at tool-call boundary，但用 GRPO advantage 而不是 UCB 做 selection。把 AXPO 嵌入 AlphaGo-style MCTS（value network + policy prior）可能进一步放大 effect。参考 Tree search for LLMs 的一些工作（https://arxiv.org/abs/2411.11904）。

7. **Reward hacking 风险**：Recovery indicator $r^{prefix}$ 是 binary 的，policy 可能学到"产生 resample 友好的 prefix"而不是"产生直接 correct 的 prefix"。Paper 没讨论这个，但理论上 AXPO 有 reward hacking 的 surface。

---

## 10. 和其他工作的联系

### 10.1 GRPO 家族

GRPO (https://arxiv.org/abs/2402.03300) 是 DeepSeekMath 提出的 group-relative advantage normalization。DAPO (https://arxiv.org/abs/2503.14476) 加了 asymmetric clipping 和 dynamic sampling。ProRL (https://openreview.net/forum?id=YPsJha5HXQ) 强调 prolonged RL expands reasoning boundary。BroRL (https://arxiv.org/abs/2510.01180) 分析 scaling $N$ 的 coverage guarantee。AXPO 是这一脉在 agentic setting 的延伸——指出 group-relative 在 tool use 这种 optional minority behavior 上的 fundamental failure，并用 targeted resampling 修复。

### 10.2 ReAct 和 agentic reasoning 起源

ReAct (https://openreview.net/forum?id=WE_vluYUL-X) 是 reasoning + acting 交替的鼻祖。Agentic reasoning (https://arxiv.org/abs/2502.04644) 把它推广到 deep research。但这些 prompting-era 的工作没有 RL training signal 的问题，AXPO 是在 post-training RL 阶段解决 ReAct-style agent 的 learning 信号问题。

### 10.3 Tool-augmented RL

ToRA (https://openreview.net/forum?id=Ep0TtjVoap)、Retool (https://arxiv.org/abs/2504.11536)、ToolRL (https://arxiv.org/abs/2504.13958)、ToRL (https://arxiv.org/abs/2503.23383) 都是 tool-integrated RL，但它们主要关注 thinking-only trajectory 上的 tool use，没有诊断 Thinking-Acting Gap。AXPO 的诊断（Figure 3 的三个 measurement）是这个领域第一个 quantitatively 把 asymmetry characterizing 出来的。

### 10.4 Distillation 和 exposure bias

paper Section 5 引用 DAgger (https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) 的 compounding error 观点，说明 SFT alone 不能保证 on-policy correctness。Kang et al. 的 distill LLM agent (https://openreview.net/forum?id=VkicTqszOn) 也是这个方向。AXPO 通过 RL 修复 exposure bias，但 standard GRPO 的 RL 受 SFT manifold 约束，AXPO 通过 resampling 跳出 SFT manifold。

### 10.5 Entropy 和 exploration

Confidence as uncertainty proxy (Appendix B.9) 让我想到 Deep Think with Confidence (https://arxiv.org/abs/2508.15260)。POPE (https://arxiv.org/abs/2601.18779) 做 privileged on-policy exploration。AXPO 的 uncertainty-based prefix ranking 和这条线相关，但 AXPO 的 exploration 是在 tool-call boundary 上 targeted，不是全局 entropy maximization。

---

## 11. Qualitative Examples 的 intuition

Appendix G 的三个例子很能说明问题：

**Perception (VisualProbe)**：Question "What is the number to the left of 1000?"，图里有 "Tel: 2563 1000"。GRPO rollout 里 thinking 一直 mention "image_zoom_in" 但**从未 emit tool call**，最后猜 999（numerical order）。AXPO rollout emit `image_zoom_in` 直接 zoom 到 telephone sign，看到 2563。这是 Thinking-Acting Gap 的完美 illustration——thinking 知道该 zoom，但 acting 从未发生。

**Reasoning (MathVision)**：三个 semicircle 几何题。GRPO rollout 纯 algebraic 推导得到 4（wrong sub-region 面积）。AXPO rollout emit `python` 画 matplotlib 图，可视化让 area-swap 结构变得可见，修正到 8。Tool call 提供了 algebraic reasoning 缺失的 geometric intuition。

**Search (HR-MMSearch)**：Multi-hop "designer → birthplace → mayor → birth year"。GRPO rollout 做完第一 hop search 后**不再 emit 第二 hop**，最后猜 designer 自己的 birth year。AXPO rollout sequential emit 两个 search call，正确解出 1984。Tool call 的 multi-hop decomposition 是 GRPO 学不到的，因为第二 hop 的 tool call 在 GRPO 下几乎总是 all-wrong（第一 hop 没返回 mayor 信息时，第二 hop 的 query 写不出来）。

这三个例子共同说明：**AXPO 不是让 model 变聪明，而是让 model 在 thinking 已经知道该 act 的情况下，真的去 act**。这正是 Thinking-Acting Gap 的本质。

---

## 12. 最终评价与 open questions

AXPO 是一个**诊断驱动、机制清晰、ablation 扎实**的工作。它不是"我们加了一个 trick 然后数字涨了"的 paper，而是先 quantitatively 诊断 GRPO 在 agentic setting 的 failure mode（Section 2.2 的三个 measurement），再用 Proposition 1 给出 resampling 为什么 dominate raw sampling 的 formal argument，最后用 9 个 benchmark × 3 个 scale 的实验验证。

我特别欣赏的几点：

1. **诊断先行**：Figure 3 的三个 measurement（tool use rate、all-wrong rate、semantic cluster count）是整个方法的 foundation。没有这些诊断，AXPO 看起来就是"加 resampling"，有了诊断才知道为什么 resampling 必须在 tool-call boundary。

2. **Compute efficiency claim 非常硬**：25% 额外 budget 打 100% 额外 budget（Table 3 的 rollout 2×），这个对比直接说明"问题在哪 spend compute 比 spend 多少 compute 重要"。

3. **Ablation 完整**：Table 2 的 5 行 ablation 每一行都对应一个具体 design choice，而且每行退化量都不小（1.8-2.9 pp），说明没有冗余设计。

4. **Pass@4 > Pass@1 gain 的 signature**：这说明 AXPO 真的扩展了 reachable correct trajectory set，而不是只 sharpen existing distribution。这是一个很 non-trivial 的 axis 区分。

我会继续追踪的几个方向：

- **Multi-turn 扩展**：Proposition 1 是 single tool-call boundary，multi-turn tree search 需要 sequential decision。可能结合 AlphaZero-style MCTS。
- **32B+ trainable**：AXPO 在 8B 上打平 32B inference-only，但 32B trainable + AXPO 会怎么样？all-wrong subgroup fraction 是否显著下降？
- **Non-verifiable reward**：Open-ended generation 下如何定义"failed subgroup"？可能需要 LLM-as-judge 的 soft recovery signal。
- **Reward hacking**：Binary recovery indicator 是否会让 policy 学到"产生容易 resample 的 prefix"而不是"一次 correct 的 prefix"？
- **和 process reward model 的结合**：PRM 可能在 tool-call boundary 提供更细粒度的 signal，和 AXPO 的 recovery indicator 互补。

paper 的 project page 在 https://link （paper 里说 available at link，但具体 URL 没给全），code 应该会开源。这个工作对 agentic RL community 的影响应该是结构性的——它指出 GRPO 在 optional behavior 上的 fundamental limitation，这个 limitation 不仅适用于 tool use，也可能适用于任何"minority but critical"的行为模式（e.g. safety intervention, long-tail skill activation）。

Andrej, 我觉得这篇 paper 最值得吸收的 intuition 是：**group-relative advantage normalization 对 minority behavior 系统性不利，而 targeted resampling at the divergence boundary 是修复这个 bias 的 principled 方式**。这个 intuition 应该可以推广到很多 agentic RL 的其他场景。
