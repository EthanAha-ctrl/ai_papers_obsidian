---
source_pdf: Agent vs. Parametric World Models.pdf
paper_sha256: 73d28fa670bd2527794ad998b9413e1431a96d7b57733f7aee4b8912d41a927a
processed_at: '2026-07-18T04:20:26-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Agent vs. Parametric World Models: Hybrid Planning 讲解

## 1. Core Insight: 两种 World Model 的"互补失败模式"

这篇 paper 的核心 insight 可以浓缩成一句话: **LLM agent 在做 planning 时,本质上同时扮演了两个角色——一个是 action selector,另一个是 implicit world model**。当 LLM 在 ReAct/Reflexion 风格的 reasoning 中写出 "task 3 is now completed" 这种话时,它实际上是在预测 environment 的 transition。问题是这个 prediction 没有 ground truth 监督,而且一旦出错,这个 false state atom 就会被写进 context window,被后续每一 step 的 attention 重新 attend 到,从而产生 **hallucination propagation**。

这跟 traditional model-based RL 里的 world model 形成了鲜明对比。parametric world model 比如 Dreamer (Hafner et al., 2020) 或 PETS (Chua et al., 2018) 的 transition error 可以用 NodeMSE 直接度量,但它缺乏 language reasoning 的 semantic flexibility。

Paper 把这个 tradeoff 形式化为一个"two-world-model comparison":
- **Agent world model**: 灵活、能做 semantic planning,但 error 是 semantic hallucination,难以用 single MSE 度量,且 history-dependent compounding
- **Parametric world model**: error 可测、local、不 propagate,但 standalone planning 能力弱 (0.565 SR vs 0.668)

Hybrid-WM 的设计哲学: 让 LLM 继续做它擅长的 semantic reasoning,只用一个小的 parametric backbone 提供一个 **auditable transition signal**,在 LLM draft 出 hallucinated delta 时 trigger 一次 targeted revision。

参考链接:
- ReAct paper: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- Dreamer V3: https://arxiv.org/abs/2301.04104
- PETS: https://arxiv.org/abs/1805.00909
- LLM-modulo (Kambhampati): https://arxiv.org/abs/2402.01817

---

## 2. Problem Formulation: Graph-Structured Planning

### 2.1 Task 定义

Task 定义为四元组 $(G_0, g, T^\star, H)$:
- $G_0$: initial world state
- $g$: natural language goal
- $T^\star$: true transition function (deterministic up to stochastic failures)
- $H$: oracle horizon

World state 暴露为 typed graph $G_t = (V_t, E_t, X_t, Z_t)$:
- $V_t$: entities (subtasks, tools, resources, components)
- $E_t$: dependency edges
- $X_t$: per-node attributes,包含 type 和 5 类 status: pending/active/completed/failed/skipped
- $Z_t \in \{0,1\}^{|V_t|}$: binary goal mask,$Z_t[v]=1$ 表示该 node 必须达到 completed 才算成功

**State serialisation matters**: 同一个 $G_t$ 可以 serialize 成 flat node table、nested adjacency list 或 Markdown checklist。Paper 在 Appendix B 里 ablate 三种格式,发现 serialisation alone 就让 HSR 变化 1.6×——因为 longer serialisation 挤占 attention budget,shorter serialisation 遗漏 dependencies。这个观察本身很有价值,说明 LLM agent 的 failure 不全是 "reasoning 能力不足",一部分是 context engineering 问题。

### 2.2 Agent World Model 的生成过程

给定 context $c_t = (\text{serialise}(G_t), g, \mathcal{A}_t)$,LLM 采样出:

$$
(\hat{a}_t, \tilde{\Delta}_t, \hat{z}_t) \sim P_{\text{LLM}}(\cdot | c_t)
$$

变量含义:
- $\hat{a}_t$: selected action at step $t$
- $\tilde{\Delta}_t \in \{-1, 0, +1\}^{|V_t|}$: imagined node-status delta。每个 entry 表示该 node 的 status 变化方向(-1 = 退化,0 = 不变,+1 = 推进)
- $\hat{z}_t \in [0,1]$: agent 自报的 confidence

Environment 然后执行真实 transition $G_{t+1} = T^\star(G_t, \hat{a}_t)$。注意: agent imagines 的 $\tilde{\Delta}_t$ 只是写进 context,environment 完全无视它。这就是 propagation 的根源——**agent 自己的 imagination 成了下一 step 的 "fact"**。

### 2.3 Hallucination 的形式化

定义 ground-truth delta:
$$
\Delta_t^\star = \text{delta}(G_t, T^\star(G_t, \hat{a}_t))
$$

A **hallucinated state atom** 就是任何 $\tilde{\Delta}_t[v] \neq \Delta_t^\star[v]$ 且 agent 声称要 change 的 entity $v$。

这里 paper 用 "delta" 而非 full state,是个很聪明的设计: delta 稀疏、直接 verifiable,而且把"agent 错了"这件事从 fuzzy semantic claim 变成了可以枚举的 atom 集合。

---

## 3. 三大核心 Metrics: HSR, PD, EES

这是 paper 的一个亮点——它把 agent world model 的 error 从"难以度量"变成了"可以横切多个维度"。

### 3.1 Hallucinated-State Rate (HSR)

$$
\text{HSR} = \frac{|\{(t, v) : \tilde{\Delta}_t[v] \neq \Delta_t^\star[v], \tilde{\Delta}_t[v] \neq 0\}|}{|\{(t, v) : \tilde{\Delta}_t[v] \neq 0\}|}
$$

即: agent 声称要 change 的 atoms 中,有多少是错的。Agent-Replan 的 HSR = 0.205,意味着 agent 每次声称要 change 5 个 status,就有 1 个是 hallucinated。

### 3.2 Propagation Depth (PD)

$$
\text{PD} = \mathbb{E}[\text{steps until recovery or episode end} | \text{atom was hallucinated}]
$$

Mean 2.45 steps。这个 metric 抓住了"hallucination 的传染性"——一个 false atom 平均要污染 2.45 step 的 context 才被纠正(或 episode 结束)。如果 H=15,这就意味着一个早期 hallucination 可以污染 16% 的整个 trajectory。

**Intuition**: PD > 1 是 agent world model 区别于 parametric world model 的关键。parametric model 的预测错了,只影响当前 step 的 action;agent 的 imagination 错了,会进入 KV cache 影响后续所有 token 的 attention。

### 3.3 Error-Explosion Slope (EES)

$$
\text{EES} = \text{least-squares slope of } \log(\text{error magnitude}) \text{ over steps } 1 \ldots 8
$$

捕捉 compounding 的速度。EES > 0 表示 error 在指数级增长;EES 接近 0 表示 error 稳定不爆发。

### 3.4 Long-Horizon Proxies

Paper 还定义了两个 horizon-level 量:

$$
\text{IndepBound}(H) = 1 - \prod_{k=1}^{H}(1 - p_{\text{err}}(k))
$$

这是在 stepwise independence 假设下,"至少一步出错"的概率。变量:
- $p_{\text{err}}(k)$: step $k$ 的 per-step error probability
- $H$: total horizon

$$
\text{ExpectedErrors}(H) = \sum_{k=1}^{H} p_{\text{err}}(k)
$$

expected number of erroneous steps,可能 > 1。

**重要 caveat**: paper 在 Appendix E 明确指出 IndepBound 是 **lower bound** 而非 upper bound。因为 PD > 0 说明 errors 是 positively serially correlated——一个 hallucinated atom 会推高 $\Pr[\mathcal{E}_{k+1}]$。在正相关下,$1 - \prod(1-p_k)$ underestimates 真实的 union probability。所以 Hybrid-WM vs baseline 的真实 gap **至少**和 IndepBound 显示的一样大。

---

## 4. Hybrid-WM 算法详解: 四个 Phase

### Phase 1: Parameterized Skeleton Scoring

Parametric backbone $F_\theta$ 对每个候选 action $a \in \mathcal{A}_t$ 输出:

$$
b_\theta(G_t, a) = (\hat{p}_{\text{valid}}, \widehat{\Delta G}, \hat{r}, \hat{p}_{\text{done}}, \hat{\rho}, \hat{U}, \hat{J}_K)
$$

变量含义:
- $\hat{p}_{\text{valid}} \in [0,1]$: action validity probability
- $\widehat{\Delta G}$: predicted next-state delta
- $\hat{r}$: predicted reward
- $\hat{p}_{\text{done}}$: predicted done probability
- $\hat{\rho}$: predicted risk
- $\hat{U}$: affected entities set
- $\hat{J}_K$: short-horizon value estimate (J 联想到 J-function,即有限 horizon value)

**关键设计**: 这些预测被压缩成 skeleton $B_t$,只保留 top-k by value 和 top-k by risk,k=4。这步 compression 必不可少,因为长 horizon 时 $|\mathcal{A}_t|$ 可能 > 50,full skeleton 会 overflow context window。Empirically k=4 保留了 96% 的 value,但 skeleton block 缩小 5×。

### Phase 2: Agent Draft

LLM 收到 system instruction + serialized state + goal + skeleton $B_t$ + candidate actions,被要求返回一个 JSON:

```json
{
  "selected_action": "execute(node_3)",
  "imagined_next_state": {"changed_nodes": [3], "node_3_new_status": "completed"},
  "reasoning": "...",
  "confidence": 0.9
}
```

形式化: $(\hat{a}_t, \tilde{\Delta}_t, \hat{z}_t) = \text{LLM}(G_t, g, \mathcal{A}_t, B_t)$

注意这里 skeleton $B_t$ 已经在 context 里——它给 agent 一个 grounded prior,在 LLM sample 之前就已经"锚定"了 transition 期望。这是和 post-hoc verification 的重要区别。

### Phase 3: Consistency Gate (核心创新)

定义两个 change-set:
- $S_t = \{v : \tilde{\Delta}_t[v] \neq 0\}$ (agent 想象要改的 nodes)
- $\hat{S}_t = \{v : \widehat{\Delta G}_t[v] \neq 0\}$ (parametric model 预测要改的 nodes)

Jaccard similarity:

$$
\text{cons}(\tilde{\Delta}_t, \widehat{\Delta G}_t) = \frac{|S_t \cap \hat{S}_t|}{|S_t \cup \hat{S}_t|}
$$

三个 branch:
- **cons ≥ τ_high = 0.70**: accept $\hat{a}_t$ directly
- **τ_low = 0.30 ≤ cons < τ_high**: accept 但在 ranking 时把 $\hat{\rho}$ 翻倍作为 penalty
- **cons < τ_low = 0.30**: trigger Phase 3b targeted revision

为什么用 Jaccard 而非精确 set equality? 因为 parametric model 自己也有 error,要求完全一致会 over-trigger。Jaccard 给了一个 graceful 的 disagreement measure。

### Phase 3b: Targeted Revision

当 gate trip 时,发出第二个 LLM call,内容是显式的 discrepancy list:

```
Node v: backbone predicts s_p but you imagined s_a.
```

Agent 被要求 revise 它的 imagined state 或 selected action。`max_corrections = 1` per step 来 bound cost。Empirically 单次 revision 解决 ~93% 的 triggered cases。

### Phase 4: Risk Gate

如果 $\hat{\rho}(\hat{a}_t) > \rho_{\text{th}} = 0.65$,触发第三个 LLM call,只 re-present low-risk subset of $\mathcal{A}_t$。Fires on ~8% of steps,主要拦截 catastrophic mistake 比如 retry 一个未解决 root cause 的 failed task。

### Cost 分析

Expected LLM calls per step:
$$
1 + 0.22 + 0.08 \approx 1.30
$$

22% 的 step 触发 correction gate,8% 触发 risk gate,所以平均每 step 1.3 次 LLM call。这是个非常便宜的 overhead,特别是相对 Agent-Verifier 的 26.29 calls/task。

---

## 5. Error Reduction Guarantee: 形式化的 contraction

Paper 给了一个简单的概率 bound,我把它逐项拆解。

### 定义

- $E_k$: Phase-2 agent draft 在 step $k$ 包含 semantic transition hallucination 的事件
- $D_k$: Jaccard gate 检测到这个 hallucination 的事件
- $R_k$: corrective revision 修复这个 hallucination 的事件

定义两个 conditional probability:
$$
\alpha_k = \Pr[D_k | E_k] \quad \text{(gate recall on erroneous drafts)}
$$
$$
\beta_k = \Pr[R_k | D_k, E_k] \quad \text{(repair success given detection)}
$$

两者都可以 < 1,因为 parametric model 也有 error。

### Proposition 1 (One-step hallucination contraction)

在 **non-adversarial correction** 假设下(即修正本身不会在修复原 error 的同时引入 new error):

$$
\Pr[E_k^{\text{Hybrid-WM}}] = \Pr[E_k](1 - \alpha_k \beta_k) \leq \Pr[E_k] \quad (1)
$$

推导:
$$
\Pr[E_k^{\text{Hybrid-WM}}] = \Pr[E_k] \Pr[\bar{D}_k \cup (D_k \cap \bar{R}_k) | E_k]
$$

(remaining erroneous 当且仅当 gate miss 或 gate detect 但 repair fail)

$$
= \Pr[E_k]\{1 - \Pr[D_k|E_k]\Pr[R_k|D_k,E_k]\} = \Pr[E_k](1-\alpha_k\beta_k)
$$

### 多步累积 bound

如果 $\alpha_k\beta_k \geq \gamma > 0$ 对所有 $k \leq H$ 成立:

$$
\mathbb{E}\left[\sum_{k=1}^{H}\mathbf{1}\{E_k^{\text{Hybrid-WM}}\}\right] \leq (1-\gamma)\mathbb{E}\left[\sum_{k=1}^{H}\mathbf{1}\{E_k\}\right] \quad (2)
$$

**Intuition**: 这个 bound 的妙处在于它 **不要求 parametric model perfect**。即使 $\alpha_k = 0.83$ (gate miss 17%),$\beta_k = 0.93$ (repair fail 7%),$\alpha_k\beta_k \approx 0.77$,hybrid 仍然把 expected hallucination count 压缩到原来的 23%。

Empirical 数据吻合: Hybrid-WM 的 p_err@10 = 0.164,vs Hybrid-Full 的 0.213,vs Agent-Replan 的 0.393。

---

## 6. 实验结果深度解析

### 6.1 主表 (Table 1) 关键对比

| Method | SR | SR-long | IAR | HSR | PD | Tok/Succ |
|---|---|---|---|---|---|---|
| Agent-Replan | 0.668 | 0.471 | 0.169 | 0.205 | 2.45 | 16389 |
| Parametric-WM-MCTS | 0.625 | 0.559 | 0.080 | 0.060 | 1.10 | 262 |
| Hybrid-Full | 0.750 | 0.636 | 0.091 | 0.111 | 1.65 | 13503 |
| **Hybrid-WM** | **0.838** | **0.758** | **0.065** | **0.079** | **1.51** | 15021 |

几个值得玩味的点:

1. **Parametric-WM-MCTS 的 Tok/Succ = 262**: 极便宜但 SR 只 0.625。说明 parametric model 在 graph-structured transition 上很准,但缺 semantic reasoning——它不知道 "为什么" 要选某 action。

2. **Agent-Verifier (Lightman et al. PRM)** 把 IAR 从 0.169 压到 0.062 但 HSR 还是 0.193: process reward model 能 catch invalid action 但 catch 不了 imagined state hallucination。这印证了 paper 的核心论点——**hallucination 和 invalid action 是两种不同的 failure mode**。

3. **Hybrid-WM vs Hybrid-Full**: 加了 consistency gate (Phase 3b 的 corrective revision)后,SR 从 0.750 升到 0.838,HSR 从 0.111 降到 0.079。这 0.088 的 SR gain 几乎全部来自"修复了 hallucinated delta,从而后续 action 不基于 corrupt state"。

4. **Hybrid-WM+Verifier 反而 SR 略低 (0.825)**: 加 verifier 引入更多 invalid action 修复但扰动 action selection。这是个 interesting finding: 不是所有 verification 都 helpful,过度 verification 可以 backfire。

### 6.2 Horizon-Resolved Analysis (Table 2)

| Method | H≤3 | H4-6 | H7-10 | H>10 | H>10 HSR |
|---|---|---|---|---|---|
| Agent-Replan | 0.957 | 0.899 | 0.762 | 0.471 | 0.222 |
| Parametric-WM-MPC | 0.639 | 0.628 | 0.607 | 0.502 | 0.074 |
| Hybrid-WM | 0.976 | 0.933 | 0.872 | 0.758 | 0.085 |

**这是 paper 最 compelling 的 result**。短 horizon 时三者差距小,因为 agent 即使 hallucinate 也来不及 propagate。但 H>10 时:
- Agent-Replan 崩到 0.471 (halo of compounding hallucination)
- Parametric 稳定但 low (0.502,ceiling 由 semantic reasoning 缺失决定)
- Hybrid-WM 达到 0.758

**Intuition**: long horizon 是 hybrid 真正发挥的地方。每一步 gate 把 hallucinated atom catch 在它进入 next context 之前,阻止了 compounding。这就像 rolling dice——单次掷骰子两次方法差别不大,但掷 100 次时,任何微小的 bias 都会放大。

### 6.3 Backbone Strength Ablation (Table 4)

最 surprising 的 finding:

| Backbone | TransAcc | Standalone SR | Hybrid SR | Hybrid HSR |
|---|---|---|---|---|
| MLP-small | 0.843 | 0.509 | 0.768 | 0.064 |
| MPNN-small | 0.992 | 0.560 | 0.771 | 0.063 |
| GPS | 0.981 | 0.571 | 0.765 | 0.061 |

**MLP 自己只能解 50.9% 的 task,但作为 skeleton provider 就能把 hybrid SR 拉到 76.8%**。这是个非常 practical 的发现:你不需要 train 一个 strong planner,你只需要 train 一个 **transition-faithful** 的小模型。因为 validity 和 delta prediction 比 full planning 简单得多。

这让我联想到 distillation literature 里的 "small model as verifier" 思路,也联想到 InstructGPT (Ouyang et al., 2022) 里 reward model 比 policy 小很多但仍然 effective 的现象。**任务难度和 verification 难度是 decoupled 的**。

### 6.4 Hybrid-WM Ablation (Table 5)

把每个 skeleton field 单独 ablate:

| Ablation | SR | HSR | 关键发现 |
|---|---|---|---|
| NoBackbone | 0.649 | 0.205 | baseline |
| Hybrid-ValidityOnly | 0.660 | 0.192 | validity 主要降 IAR 不降 HSR |
| Hybrid-DeltaOnly | 0.703 | 0.121 | **delta 是降 HSR 的主要 driver** |
| Hybrid-RiskOnly | 0.657 | 0.183 | risk 降 RWF 不降 HSR |
| Hybrid-ValueOnly | 0.678 | 0.167 | value 改善 ranking |
| Hybrid-WM-NoCorrectionGate | 0.772 | 0.108 | 没 gate 退化到 Hybrid-Full + |
| Hybrid-WM-NoRiskGate | 0.782 | 0.081 | SR 持平但 RWF 翻倍 |
| Hybrid-WM-tau0 (always correct) | 0.823 | 0.067 | 多花 token 小幅 gain |
| Hybrid-WM-tau1 (never correct) | 0.759 | 0.113 | 退化到 Hybrid-Full |
| **Hybrid-WM (τ=0.30)** | **0.838** | **0.079** | sweet spot |

关键 insights:
1. **Delta field 是降 HSR 的核心**(0.205 → 0.121),validity 主要降 invalid action rate,risk 降 risk-weighted failure。这说明 hallucinated state 是独立于 invalid action 的 failure mode。

2. **τ_low = 0.30 是 sweet spot**:τ 太低(<0.25)会 over-trigger 在 minor discrepancy 上,τ 太高(>0.40)会 miss 真正的 hallucination。

3. **Correction gate 是关键的"bridge"**:没有它,Hybrid-WM-NoCorrectionGate 退化到 0.772 SR,HSR 0.108。这个 gate 把"parametric model 有 error signal"变成"agent 实际 revise 它的 imagination"。

### 6.5 Live GPT-4o-mini Validation (Table 7)

这是 paper 的"reality check": 在真实 GPT-4o-mini 上跑 n=80 tasks,而非 simulator:

| Benchmark | Agent HSR | Hybrid HSR | Reduction |
|---|---|---|---|
| TaskGraph | 0.142 [0.111, 0.172] | 0.040 [0.024, 0.055] | -72% |
| ToolChain | 0.193 [0.148, 0.238] | 0.024 [0.011, 0.036] | -88% |
| ResourceAlloc | 0.170 [0.136, 0.204] | 0.046 [0.024, 0.069] | -73% |
| RepairFlow | 0.200 [0.171, 0.229] | 0.029 [0.010, 0.049] | -86% |
| **Pooled** | **0.176** | **0.035** | **-80%** |

所有 four benchmark 的 95% CI 都 **non-overlapping**,统计上非常 solid。而且 SR=1.000 for both arms——说明 H≤8 task 都能被 GPT-4o-mini 解,真正的 bottleneck 是 hallucinated state content,不是 task complexity。

### 6.6 Multi-Backbone Generalization (Table 8)

| Backbone | Agent SR | Agent HSR | Hybrid SR | Hybrid HSR | Corr Rate |
|---|---|---|---|---|---|
| GPT-4o-mini | 0.755 | 0.171 | 0.795 | 0.014 | 20.2% |
| Claude-3-Haiku | 0.620 | 0.204 | 0.790 | 0.074 | 24.8% |
| Gemini-1.5-Flash | 0.600 | 0.231 | 0.780 | 0.088 | 27.1% |
| Llama-3-8B | 0.470 | 0.318 | 0.730 | 0.109 | 31.9% |

**关键 finding**: Hybrid-WM **equalize backbones**。Llama-3-8B 在 agent-only 模式下 SR 只有 0.470 (大量 JSON parse failure),但加 Hybrid-WM 后 SR 升到 0.730,HSR 降到 0.109。这暗示 consistency gate 吸收了 backbone-specific noise。

Correction rate (20-32%) 和 agent-only HSR (17-32%) **monotonic 相关**——这是 gate 作为 per-step hallucination diagnostic 的有力证据。

### 6.7 Knowledge Graph Traversal (Table 9)

在 FB15k-237 上做 multi-hop KG traversal,作为 graph-structured planning 之外的 generalization test:

| Method | SR | HSR |
|---|---|---|
| Agent-only | 0.833 (10/12) | 0.888 ± 0.197 |
| Hybrid-WM | 0.750 (9/12) | 0.801 ± 0.223 |

n=12 太小,McNemar exact p = 1.000,SR 和 HSR 的差异都 not significant。但有个有意思的 qualitative finding: agent 在 KG 上几乎总是 hallucinate result entity IDs (HSR 0.888),因为 Freebase ID 对任何 LLM 都是 opaque。Hybrid-WM 把 HSR 降了 8.7pp 但 SR 略降,因为 backbone 在 500 trajectory 上 under-calibrated。

**这是 paper 的 honest moment**: 作者承认这个实验 underpowered,需要 n≥120 才能在 80% power 下检测 10pp 的 HSR 差异。这告诉读者 Hybrid-WM 的边界——backbone calibration 质量是关键。

---

## 7. Parametric Backbone 的 Multi-task Loss

Backbone 训练用 multi-task loss:

$$
\mathcal{L}_{\text{WM}} = \mathcal{L}_{\text{state}} + \lambda_r \mathcal{L}_{\text{reward}} + \lambda_d \mathcal{L}_{\text{done}} + \lambda_m \mathcal{L}_{\text{mask}} + \lambda_\rho \mathcal{L}_{\text{risk}}
$$

具体配置(Appendix D):
$$
\mathcal{L} = \mathcal{L}_{\text{BCE}}(p_{\text{valid}}) + 0.5\mathcal{L}_{\text{CE}}(\Delta G) + 0.3\mathcal{L}_{\text{MSE}}(\hat{r}) + 0.3\mathcal{L}_{\text{BCE}}(\hat{d}) + 0.2\mathcal{L}_{\text{BCE}}(\hat{\rho})
$$

变量:
- $\mathcal{L}_{\text{BCE}}(p_{\text{valid}})$: binary cross-entropy on action validity
- $\mathcal{L}_{\text{CE}}(\Delta G)$: cross-entropy on state delta (multi-class,{-1,0,+1})
- $\mathcal{L}_{\text{MSE}}(\hat{r})$: MSE on reward
- $\mathcal{L}_{\text{BCE}}(\hat{d})$: BCE on done signal
- $\mathcal{L}_{\text{BCE}}(\hat{\rho})$: BCE on risk

State vector = per-node status one-hot (4 classes) ⊕ edge-type embedding (8 classes)。MLP 2 层 128 unit;GCN/MPNN 2 层 64-dim message passing;GPS 加 Transformer attention。**所有模型 <100k 参数**——非常小。

NodeMSE 定义:
$$
\text{NodeMSE} = \frac{1}{|V_t|}\sum_{v \in V_t}\|\hat{x}_{t+1}(v) - x_{t+1}^\star(v)\|_2^2
$$

变量:
- $\hat{x}_{t+1}(v)$: backbone 预测的 node $v$ 在 step $t+1$ 的 attribute vector
- $x_{t+1}^\star(v)$: ground truth attribute vector
- $|V_t|$: graph 中 node 数量

---

## 8. Cost Model

Per-task dollar cost:
$$
C(\pi) = \sum_{t=0}^{\text{steps}}\left(T_{\text{in}}(t)p_{\text{in}} + T_{\text{out}}(t)p_{\text{out}}\right) \cdot 10^{-6} \text{ USD}
$$

变量:
- $T_{\text{in}}(t)$, $T_{\text{out}}(t)$: step $t$ 的 input/output token count
- $p_{\text{in}}, p_{\text{out}}$: per-million-token 价格

GPT-4o-mini: ($0.15, $0.60) USD/MTok
Claude-3-Haiku: ($0.25, $1.25)
Llama-3-8B self-hosted: marginal cost = 0

Hybrid-WM 在 GPT-4o-mini 上 $1.06 per 1k tasks,Gemini-1.5-Flash 最便宜 $0.29。Llama-3-8B 完全免费但 SR 损失 5pp。

---

## 9. 与 Related Work 的精微对比

### 9.1 vs LLM+P (Liu et al., 2023b)

LLM+P 把 planning 完全路由到 PDDL classical solver,LLM 只做 translation。优点是 PDDL solver 给出最优 plan,缺点是要求 domain 能被 PDDL 表达,且 LLM 完全不做 reasoning。

Hybrid-WM 让 LLM 保持 reasoning 主导,backbone 只是 advisory。这让 Hybrid-WM 适用于 PDDL 难以表达的 graph-structured domain。

### 9.2 vs LLM-modulo (Kambhampati et al., 2024)

LLM-modulo 提出 LLM + symbolic verifier 框架,verifier 在 LLM 出 action 后做 post-hoc check。Hybrid-WM 的区别是它在 **两个 stage** intervene:
1. **Pre-draft**: skeleton 进 context,ground LLM 的 imagination a priori
2. **Per-step**: consistency gate 在同一 step 内 trigger revision,而非等下一个 step 才发现错

这种 "ante-hoc + per-step" 的 intervention pattern 比 post-hoc reranking 更 efficient。

### 9.3 vs WorldCoder (Tang and Ellis, 2024) 和 WALL-E (Zhou et al., 2024b)

WorldCoder 让 LLM synthesize executable code 作为 world model,从 interaction 中 iterative refine。WALL-E 从 rollout 中 induce symbolic rules。

两者都是 "let LLM build its own world model" 路线。Hybrid-WM 是 "train a separate small model as world model"。前者更 general(generalize to new domain via code synthesis),后者更 sample-efficient(5k 参数 MLP 就够)。

### 9.4 vs Process Reward Models (Lightman et al., 2023)

PRM 用 large learned verifier 给 reasoning step 打分。Hybrid-WM 的 backbone 不打分,它做 structured prediction (delta, validity, risk) 然后 Jaccard gate 比较。前者需要 train verifier 在 reasoning trace 上,后者 train 在 transition 上——transition 数据比 reasoning trace 更易收集。

### 9.5 vs RAP (Hao et al., 2023)

RAP 把 LLM 当 transition model 嵌入 MCTS。但 RAP 直接 trust LLM 的 transition prediction,所以 LLM hallucination 会直接 propagate 进 MCTS tree。Hybrid-WM 的 consistency gate 可以看作 RAP 的 "transition model verification layer"。

参考链接:
- LLM+P: https://arxiv.org/abs/2304.11477
- WorldCoder: https://arxiv.org/abs/2402.12275
- WALL-E: https://arxiv.org/abs/2410.07484
- PRM (Let's verify step by step): https://arxiv.org/abs/2305.20050
- RAP: https://arxiv.org/abs/2305.14992

---

## 10. 一些值得深挖的联想

### 10.1 KV Cache 视角下的 Propagation Depth

PD = 2.45 的本质是:一个 hallucinated atom 进入 KV cache 后,后续 step 的 attention 会 attend 到它。如果我们把 LLM 的 context 看作一个 Markovian state,那么 hallucination propagation 就是这个 state 的 "corruption dynamics"。

这让人联想到 RNN 的 vanishing/exploding gradient——error 在 time dimension 上 propagate,要么 decay 要么 amplify。PD > 1 意味着 corruption 不 decay,所以 long horizon 必然崩。Hybrid-WM 的 consistency gate 是个 "error correction code" 在 time dimension 上。

### 10.2 LLM as World Model 的 Implicit Learning

最近有 work 表明 LLM 本身 implicitly encode 了 web environment dynamics (Gu et al., 2025, https://arxiv.org/abs/2411.06559)。这给 Hybrid-WM 提供了一个有趣的解读:LLM 已经是 world model,只是它的 implicit prediction 不可靠。Hybrid-WM 用 explicit parametric model 来 "verify & ground" implicit prediction。

这跟 mechanistic interpretability 里的 "linear probe" 思路异曲同工——你不需要 rewrite model,只需要 probe + verify 它的 implicit state。

### 10.3 Hallucination ≠ Invalid Action

这是 paper 的一个隐性 contribution,值得 explicit 强调。Agent-Verifier 把 IAR 从 0.169 压到 0.062 但 HSR 几乎不变 (0.205 → 0.193)。这说明 invalid action detection 和 hallucinated state detection 是两个 **正交** 的 failure mode:

- Invalid action: action 不被 environment 接受,immediate feedback
- Hallucinated state: agent 的 imagination 错了,可能 action 本身 valid,但 agent 基于错的 state 做后续 plan

Hybrid-WM 同时处理两者(validity field 处理前者,delta field 处理后者),这是为什么 Hybrid-Full-Verifier 的 SR 反而比 Hybrid-WM 低——verifier 只补了 invalid action,没补 hallucinated state。

### 10.4 类比 Chain-of-Verification (Dhuliawala et al., 2023)

CoVe 让 LLM 自己 generate verification question 自己 answer。Hybrid-WM 的 consistency gate 是个 "external" verification,用 parametric model 而非 LLM 自身。这避免了 self-verification 的 "confirmation bias"——LLM verify 自己的 output 倾向于 agree。

### 10.5 联想到 OpenAI 的 o1 / reasoning model

o1 风格的 reasoning model 在内部做 long CoT,可能也面临 hallucination propagation 问题。Hybrid-WM 的设计哲学——用一个 external signal 在 reasoning trace 中 trigger correction——可以扩展到 test-time compute scaling。可以想象一个 "reasoning step verifier" 在 o1-style reasoning 中 trigger backtrack。

### 10.6 联想到 Tool-Augmented Self-Correction

CRITIC (Gou et al., 2024) 让 LLM 用 tools 自己 critique。Hybrid-WM 可以看作 CRITIC 的"minimal tool":一个 forward pass 的 parametric model 作为最便宜的 critique tool。

### 10.7 关于 Backbone Calibration 的开放问题

KG experiment 的失败提示:Hybrid-WM 高度依赖 backbone 的 calibration。如果 backbone 在 OOD domain 上 under-calibrated(像 KG 上的 500 trajectory MLP),correction 可能 perturb 本来对的 action selection。一个 open question 是: 能否 train 一个 backbone 来 **estimate its own uncertainty**,在高不确定性时 disable gate?

这让人联想到 Bayesian deep learning 和 deep ensembles (Lakshminarayanan et al., 2017) 的 calibration 方法。

### 10.8 Token Cost 作为 Attention Budget

Paper 提到 serialisation format 改变 HSR 1.6×——longer serialisation 挤占 attention budget。这跟 recent work on "lost in the middle" (Liu et al., 2023, https://arxiv.org/abs/2307.03172) 一致。一个推论: Hybrid-WM 的 skeleton block 也会占用 attention budget,所以 k=4 的 compression 是必要的。如果未来 LLM 的 effective attention 大幅提升,可能可以 relax k。

### 10.9 联想到 Model Predictive Control (MPC)

Parametric-WM-MPC 把 world model 嵌入 MPC 框架做 planning。MPC 的局限是它需要 model 在多 step rollout 上保持准确——而 parametric model 的 NodeMSE 会随 horizon compounding。Hybrid-WM 把 multi-step rollout 的责任交给 LLM agent(它的 semantic reasoning 可以 recovery from small error),parametric model 只负责 single-step delta prediction——这是个非常 decoupled 的 division of labor。

### 10.10 一个可能的 extension: Learned Consistency Function

Jaccard 是个 hand-crafted metric。可以想象 train 一个 small neural network $f_\phi(\tilde{\Delta}_t, \widehat{\Delta G}_t, G_t) \to [0,1]$ 来决定何时 trigger correction。这能 learn 什么 type of disagreement 是 dangerous 的 vs benign 的。这可能比固定 τ_low = 0.30 更 robust。

### 10.11 联想到 Speculative Decoding

在 inference 加速领域,speculative decoding 用 small model draft,large model verify。Hybrid-WM 反过来:large model (LLM) draft,small model (parametric backbone) verify。这种 "anti-speculative" pattern 在哪里有用? 当 small model 在 structured prediction (delta) 上比 LLM 更准但 free-form reasoning 不如 LLM 时。

### 10.12 联想到 "Tool use is an attention sink"

最近有 work (https://arxiv.org/abs/2402.04960) 表明 tool use 在 LLM 里充当 attention sink。Hybrid-WM 的 skeleton block 可能也扮演类似 role——给 LLM 一个 structured "anchor" 来 attend,减少 hallucination。

---

## 11. 局限与 Future Work

Paper 在 Section 7 诚实承认的局限:

1. **Simulator-based evaluation**: 主表 n=3200 trajectories × 12 methods 是 simulator 产出,只在 n=80 real GPT-4o-mini episodes 上校准。Simulator 倾向 over-predict HSR 和 over-estimate token cost 3-4×,所以主表是 conservative。Claude/Gemini/Llama rows 是从 published reports calibrated 的,不是 live measurement。

2. **Graph-structured scope**: 四个 benchmark 都是 graph-structured。Free text、continuous values、partial observability 的 domain 可能改变 skeleton field 的重要性。

3. **Oracle transitions**: backbone train 在 oracle transition 上,某些 domain 可能 noisy 或 unavailable。

4. **Backbone 可以 confidently wrong**: 如果 backbone 在某 domain 系统性 bias,会 mislead agent。Paper 的 prompt frame skeleton 为 advisory,允许 justified override,但没 test adversarial backbone。

我能想到的额外 future direction:
- **Self-improving backbone**: 让 backbone 从 agent 实际 rollout 中持续 fine-tune
- **Multi-modal skeleton**: 当 state 包含 image 时,backbone 需要 visual encoder
- **Hierarchical Hybrid-WM**: high-level parametric model 做 subgoal planning,low-level 做 action planning
- **Backbone uncertainty estimation**: ensemble 或 BNN 给 gate 一个 confidence-aware trigger
- **Adversarial backbone robustness**: 如果 backbone 被 poison,Hybrid-WM 如何 degrade gracefully

---

## 12. 总结:为什么这篇 paper 重要

这篇 paper 的核心贡献,我觉得有三层:

**Layer 1 (技术层)**: 一个 simple, cheap, effective 的 hybrid planning 算法,5k 参数 MLP + 一个 Jaccard gate 就能把 LLM agent 的 hallucinated state rate 降 80%。

**Layer 2 (conceptual 层)**: 它把 "LLM agent 的 hallucination" 从一个 fuzzy 概念变成了可量化的 HSR/PD/EES 三维度,并把它和 parametric world model 的 supervised error 形式化对比。这给未来 agent reliability 研究提供了一个 measurement framework。

**Layer 3 (philosophical 层)**: 它印证了一个 emerging pattern——**LLM + small structured model** 比 pure LLM 或 pure symbolic 都强。LLM 负责模糊的、semantic 的、compositional 的 reasoning;small structured model 负责可验证的、局部的、可监督的 prediction。这种 "neuro-symbolic" pattern 在 Anthropic 的 Constitutional AI、OpenAI 的 verifier model、DeepMind 的 AlphaProof 都有 echo。

Hybrid-WM 可能是 LLM agent reliability 这条线上一个重要的 "bridge paper"——它把 model-based RL 的传统智慧 (small world model for grounding) 和 LLM agent 的 modern paradigm (LLM as flexible planner) 连接起来,并给出了一个 **操作上可实施** 的方案。

代码: https://github.com/Hik289/Agent-vs-param.git

---

## Reference Links

- Paper GitHub: https://github.com/Hik289/Agent-vs-param.git
- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- LATS: https://arxiv.org/abs/2310.04406
- RAP: https://arxiv.org/abs/2305.14992
- Plan-and-Solve: https://arxiv.org/abs/2305.04091
- Dreamer V3: https://arxiv.org/abs/2301.04104
- DreamerV2: https://arxiv.org/abs/2010.02193
- PlaNet: https://arxiv.org/abs/1811.02551
- PETS: https://arxiv.org/abs/1805.00909
- Dyna (Sutton 1991): https://www.cs.toronto.edu/~hinton/csc321/papers/sutton-dyna.pdf
- MuZero: https://www.nature.com/articles/s41586-020-03051-4
- LLM+P: https://arxiv.org/abs/2304.11477
- LLM-modulo: https://arxiv.org/abs/2402.01817
- WorldCoder: https://arxiv.org/abs/2402.12275
- WALL-E: https://arxiv.org/abs/2410.07484
- WALL-E 2.0: https://arxiv.org/abs/2504.15785
- PRM (Let's verify step by step): https://arxiv.org/abs/2305.20050
- CRITIC: https://arxiv.org/abs/2305.11738
- Chain-of-Verification: https://arxiv.org/abs/2309.11495
- Faithful CoT: https://arxiv.org/abs/2303.05173
- Self-Consistency: https://arxiv.org/abs/2203.11171
- LLMs as World Models of Internet (Gu et al.): https://arxiv.org/abs/2411.06559
- AgentBench: https://arxiv.org/abs/2308.03688
- FB15k-237: https://workshop-proceedings.icwsm.org/abstract.php?id=22
- Lost in the Middle: https://arxiv.org/abs/2307.03172
- Voyager: https://arxiv.org/abs/2305.16291
- ExpeL: https://arxiv.org/abs/2308.10144
- CodeAct: https://arxiv.org/abs/2402.01030
- GPT-4 Technical Report: https://arxiv.org/abs/2303.08774
- InstructGPT: https://arxiv.org/abs/2203.02155
- Grammar-constrained decoding: https://arxiv.org/abs/2305.13971
- SWE-bench: https://arxiv.org/abs/2310.06770
- WebArena: https://arxiv.org/abs/2307.13854
- AgentTuning: https://arxiv.org/abs/2310.12823
- FireAct: https://arxiv.org/abs/2310.05915
- Cognitive Architectures for Language Agents (Sumers et al.): https://arxiv.org/abs/2306.05831

如果你想 build deeper intuition,我特别推荐先看 Section 5.2 (long horizon scaling), Table 5 (ablation), Table 4 (backbone strength)——这三个 result 加起来构成了 paper 的 intellectual core。然后再回头看 Section 4 的算法设计和 Section 4.7 的 theoretical guarantee,就会发现所有这些 design choice (Jaccard gate, τ=0.30, delta field as primary signal, k=4 skeleton) 都有 empirical grounding。
