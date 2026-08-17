---
source_pdf: Budget-Constrained Agentic Large Language Models Intention-Based Planning
  for Costly Tool Use.pdf
paper_sha256: c52238a695828f0301e9556dec11a7000f7ce8286518c1cc91d1d5c4ecf275f9
processed_at: '2026-08-03T14:33:33-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 INTENT

好的 Karpathy，我换个讲法，用大白话。

---

## 这 paper 在说什么

想象你给一个 intern 50 块钱，让他去查 Google 的财报。API 市场上有 30 个 tool，有的贵（38 块一次），有的便宜（7 块一次），有的准有的烂。

**问题**：intern 经常管不住手。他花 38 块查了 cash flow，剩 12 块，然后又非要花 23 块查 balance sheet——结果超支 11 块，task 直接 fail。

你跟他说"你只剩 12 块了"，他**听见了**，但**下一步还是该花就花**。这就是 GPT-4.1-mini 加 PROMPT 的真实表现：32.8% 的任务超支。

更糟的是 reasoning model（GPT-5-nano）：它倒是不超支了，但变得过度胆小，干脆啥也不调，pass rate 远低于理论可能值。

paper 就在问：**能不能让 agent 既能花钱又不超支，并且尽量贴近"budget 下能拿到的最高分"**？

---

## 为什么 naive 解法都不行

### Prompt 告诉它"剩多少钱"没用

因为 LLM 是 next-token predictor，每次只看当前 context。它**没有真正的 multi-step lookahead**。它每次决策都 myopic："这个 tool 看起来有用，调一下"，然后下一步再 myopic 一次。

你跟我（Karpathy）都讲过：**autoregressive policy 的 fundamental limitation 就是 path dependence**。一旦 agent 走上某条路径，它不会 backtrack 去考虑 cheaper substitute。它就像个不会回头看的 chess player。

### RL 训练也不行

理论上你可以 PPO fine-tune agent 学会 budget-aware。但是：
- 训练时 agent 只能见到 historical tool data，**真实 service quality 在 inference 时才显现**（API 改版了、价格变了、新 tool 上线了）
- 想追上 market 动态要不停 retrain，对 frontier model 极贵
- on-policy data collection 本身要花钱调真 API

### MCTS 也不行

classical MCTS 要 branching + 多次 rollout。这里 action space 是 free-form text（tool argument），branching factor 爆炸；每次 real tool call 几百毫秒， prohibitive latency。参考 [POMCP](https://papers.nips.cc/paper/2010/hash/edfbe28fb28d296c9c2532dca5b62dc3-Abstract.html) 都假设 cheap environment，这里完全 fail。

---

## INTENT 的核心 trick

我直接讲 intuition，不堆公式。

### Trick 1: 别预测 tool 返回什么，预测它"够不够用"

agent 决定要不要改 plan，主要 driven by 一个 binary 信号："这个 tool 满足我的 intention 吗？" 不 driven by 具体返回内容。

比如 agent 想查 Google 的 net income：
- tool 返回正确数字 → satisfied → 继续下一步
- tool 返回 404 → not satisfied → retry
- tool 返回 schema 对但 data 错 → satisfied（agent 不知道是错的）→ 继续下一步

下一步的 decision 主要看这个 binary state。所以 world model 不用预测具体 observation，只要预测 $z_t \in \{0,1\}$：intention 是否满足。

paper 用 total probability 分解：

$$P_w(o_t \mid r_t, a_t) = \sum_{z_t \in \{0,1\}} P(o_t \mid a_t, z_t) \cdot P(z_t \mid r_t, a_t)$$

变量解释：
- $o_t$：tool 返回的 observation
- $r_t$：reasoning trace，承载 agent 的 intention
- $a_t$：tool call（tool $T_t$ + arguments $u_t$）
- $z_t$：binary latent，1 表示 intention 满足

这一下把 simulation 变得 tractable——只要估 $\tilde{\rho}_t = P(z_t=1 \mid r_t, T_t, u_t)$，不用模拟具体 observation content。

### Trick 2: 跑一条"理想 trajectory"，看 agent 心里想干嘛

MCO（简化版）做 stochastic rollout，high variance：可能恰巧走了 happy path，cost 估小了，真超支。

INTENT 改成 **ideal trajectory simulation**：每一步强制 $z_k = 1$（intention 满足），看 agent 在"完美世界"下会走什么路径。

$$\tilde{\tau}^* = (\tilde{a}_t, \tilde{o}_t^{(z=1)}, \tilde{r}_{t+1}, \tilde{a}_{t+1}, \ldots, \tilde{a}_K)$$

这相当于 **maximum likelihood latent plan extraction**。agent 自己不知道自己 latent 的 plan 长什么样，oracle 通过强制 ideal rollout 把它"逼出来"。

这点跟你讲过 ["LLMs have latent plans but they're not explicit"](https://www.youtube.com/watch?v=l8pRSuU81PU) 完全吻合。INTENT 提供了 mechanism 把 latent plan 显式化。

### Trick 3: 用 geometric distribution 估真实 cost

agent 在真实世界不会一次失败就放弃，它会 retry 直到 intention 满足。假设每次 success probability 是 $\tilde{\rho}_k$，retry 服从 [geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution)，期望尝试次数是 $\frac{1}{\tilde{\rho}_k}$。

所以 expected cost：

$$\tilde{c}_k = \frac{\text{COST}(a_k)}{\tilde{\rho}_k}$$

变量解释：
- $\text{COST}(a_k)$：单次 call 的 price $c^{(j)}$
- $\tilde{\rho}_k$：calibrated intention satisfaction probability
- $\tilde{c}_k$：pessimistic 估计的 expected cost

举个具体例子，paper case study 里：
- cost=38 的 cash flow tool，$\tilde{\rho}=0.35$ → expected cost = 38/0.35 = **108.6**
- cost=7 的便宜 tool，$\tilde{\rho}=0.02$ → expected cost = 7/0.02 = **350**

便宜的那个反而 expected cost 爆表——因为它大概率失败要 retry。

**这个 insight 是 INTENT 最聪明的地方**：agent 看到 price 7 觉得便宜，但 oracle 看到 price 7 + success prob 0.02 知道它"期望"要花 350。这种 second-order cost reasoning 是 LLM 通过 prompt 学不会的。

### Trick 4: Risk factor γ 控制 conservative 程度

加 risk preference $\gamma$：

$$\gamma \cdot \sum_{\tilde{a}_k \in \tilde{\tau}^*} \tilde{c}_k \leq B_t$$

- $\gamma \geq 1$：非常保守，宁可少调 tool
- $\gamma < 1$：激进，愿意赌一把
- paper 用 $\gamma=0.5$，Figure 7 显示 $\gamma \in [0.5, 0.7]$ 是 sweet spot

---

## INTENT 跑起来什么样

我直接走 paper Appendix E 的 case study（task 70610），让 intuition 落地：

**Query**: 查 Google quarterly cash flow + balance sheet + income statement + ratios。Budget=50。

### 第 1 轮：agent 提议贵 tool

agent thought: "我想先抓 cash flow，cost=38，剩 12。"

INTENT 在背后跑 ideal simulation：
- 假设 cash flow tool 成功（z=1）
- 假设下一步 income statement 也成功
- 算 expected cost = 38/0.35 + 11/0.97 = 108.6 + 11.3 = 119.9
- γ=0.5 → 60 > 50 → **REJECT**

feedback 给 agent："你这 tool price=38 但 success prob 只有 0.35，重试 expected cost 是 108。Budget 不够。"

### 第 2 轮：agent 切换到便宜 tool

agent 学乖了，换成 cost=7 的 Freedom Finance tool。

INTENT 再 simulation：
- price=7, $\tilde{\rho}=0.02$ → expected cost = 350 → **REJECT**

feedback："便宜是便宜，但 success prob 0.02，等于要 retry 50 次才能成功，expected cost 350。"

**Blacklist 机制**：把这俩高风险 tool 暂时拉黑，agent 后面不会再考虑。

### 第 3 轮：agent 找到 cost-effective tool

agent 找到 Holistic Finance Stock Data (cost=9, $\tilde{\rho}=0.71$)。

INTENT simulation：
- ideal trajectory: cash flow (9) + income statement (11) + balance sheet (23)
- expected cost = 9/0.71 + 11/0.94 + 23/0.90 = 12.7 + 11.7 + 25.6 = 50.0
- γ=0.5 → 25 < 50 → **ACCEPT**

### 第 4-5 轮：cache hit，顺畅执行

ACCEPT 后，oracle 把 ideal trajectory 的 action sequence 缓存。下一步 agent 提议的 action 跟 cache 一致，直接批准，不再 simulation。

最终成功完成 task，total cost = 43，预算剩 7。

---

## 结果如何

Table 1 我直接贴关键数据：

| Backbone | Method | Pass Rate | Budget-Optimal Rate | Feasible Rate | Latency |
|----------|--------|-----------|---------------------|---------------|---------|
| GPT-4.1-mini | RAW | 19.1 | 23.3 | 34.5 | 1.00× |
| GPT-4.1-mini | PROMPT | 30.9 | 37.7 | 67.2 | 0.55× |
| GPT-4.1-mini | DFSDT | 44.9 | 54.7 | 100 | 0.99× |
| GPT-4.1-mini | BTP | 46.4 | 56.8 | 100 | 0.76× |
| GPT-4.1-mini | BATS | 53.0 | 64.6 | 100 | 3.55× |
| GPT-4.1-mini | MCO | 58.9 | 71.8 | 100 | 2.05× |
| GPT-4.1-mini | **INTENT** | **63.8** | **77.8** | **100** | 1.76× |
| GPT-5-nano | **INTENT** | **76.0** | **92.6** | **100** | 2.16× |

几个直觉解读：

1. **PROMPT 的 0.55× latency 是假象**——它早 stop 所以快，但其实超支 32.8%
2. **Enforce baselines FR=100** 是因为他们直接 block 超支 action，但 PR 低（conservative）
3. **BATS 在 reasoning model 上 latency 10.1×**——经典 MCTS 在大 action space 上的代价
4. **INTENT 在 GPT-5-nano 上达到 OR=92.6**，几乎触到 budget 下的理论上限
5. **INTENT 比 MCO 高 4-5 个点**——intention decomposition 的 variance reduction 真有用

---

## Robustness 实验：market 变了怎么办

paper Section 4.3 的 robustness 实验很 thoughtful：

### New tools 上线（Figure 3）

用不同 fraction 的 training data 训 oracle，模拟新 tool 不断出现。结果显示 **log-linear scaling**：1/32 数据已经能拿到不错 PR。这说明 oracle 学的是"intention alignment 的 pattern"，不是 memorize 具体 tool spec。

### 价格变动（Figure 4）

把 reference tool 价格 ±50%：
- PROMPT 极敏感，PR 大幅波动——因为 path-dependent，没 substitution reasoning
- INTENT 稳定——它做 risk-adjusted cost estimation 时自然考虑 alternative

**关键 insight**：INTENT 把 cost-awareness 从 prompt-level 移到 algorithmic-level。agent 自己不必理解"价格变了我要换 tool"，oracle 在背后做这个 reasoning。

### Budget 缩放（Figure 5）

INTENT 在小 budget 下也能维持高 PR，PROMPT 在小 budget 下崩。这对 production 很有价值——你可以根据业务 SLA 调 budget 而 expect PR 平滑变化。

---

## 跟你工作的联想

### 跟 [World Models (Ha & Schmidhuber)](https://worldmodels.github.io/)

INTENT 的 $\mathcal{W}_\phi$ 是 Language World Model，跟 visual world model 思路一致——用 cheap learned simulator 替代 expensive real interaction。但 INTENT 引入 $z_t$ latent 把 simulation 节约到 intention 层面，是聪明 abstraction。

这跟你 Tesla 时代推的 [drive world model](https://www.youtube.com/watch?v=j0z4FweC4RM) 概念相似：你不可能让真实 fleet 反复试错训练 planner，必须用 learned simulator。INTENT 在 tool-use 场景做了类似的事。

### 跟 [nanoGPT](https://github.com/karpathy/nanoGPT) 精神

INTENT 的 implementation 用 Qwen2.5-3B + Qwen3-0.6B，**单卡 RTX Pro 6000 训完**。这跟你"小而美"工程哲学一致——不需要 frontier-scale 就能 demo 一个 strong idea。

### 跟你讲 ["don't ask LLMs to do everything, build systems around them"](https://karpathy.bearblog.dev/)

INTENT 的核心 design philosophy：**Budget-aware 不应该作为 LLM 的 capability，而应该作为 inference-time 的 planner overlay**。LLM 还做它最擅长的 ReAct，oracle 在背后 enforce budget constraint。

这跟你常说的 ["don't put everything in the model, put it in the system"](https://www.youtube.com/watch?v=zjkBMFhNjjc) 完全一致。

### 跟 [System 2 thinking / verifier-based reasoning](https://arxiv.org/abs/2406.14525)

你最近关注的 verifier-based reasoning 跟 INTENT 的 oracle 有点像——都是 inference-time verifier/planner。INTENT 验证 budget feasibility 而非 logical correctness，但 mechanism 类似。

### 跟 [speculative decoding](https://arxiv.org/abs/2211.17151) 思路像

都是 cheap model anticipate，verify 后 commit。差异：speculative decoding 验证 token-level match，INTENT 验证 intention satisfaction。

---

## 这 paper 的妙处

1. **Problem-driven 而非 technique-driven**：它没发明新架构、新 RL algorithm，但精准诊断了 agentic API economics 痛点，用 minimal 且 principled 方法解决
2. **Intention-based decomposition 是 deep insight**：从 cognitive science 借的——agent 改 plan 的 trigger 是 semantic alignment 而非 concrete content
3. **Inference-time only** 严格守住——这对 deployment 必须
4. **Geometric cost calibration** 数学简单但 effective，把 stochastic retry 转成 expected cost
5. **Case study** 极具说服力，把 mechanism 讲透

## 这 paper 的不足

1. **没 theoretical analysis**：没 regret bound、没 sample complexity。能不能证明 INTENT 的 cost estimate 是 true cost 的 upper bound with high probability？
2. **World model hallucination 风险**：$\mathcal{W}_\psi$ 生成 ideal observation 时可能 hallucinate 不存在的 data。对 high-stakes 场景（金融、医疗）risky
3. **Geometric distribution 假设强**：真实 retry 之间有 information gain，success probability 应该单调上升而非 constant
4. **Intention 是 binary 太粗糙**：可以扩展为 categorical（partial / full / mismatch）让 agent 得到更细 risk signal
5. **Single-trajectory lookahead 弱**：multi-modal intent distribution 下可能 fail，可以考虑 [beam search variant](https://arxiv.org/abs/2305.14992)

---

## 一句话总结

**让 agent 自由发挥 ReAct，但在它每次 propose action 前，一个 lightweight learned world model 在背后做 risk-aware feasibility check，并通过 feedback 让 agent 看见自己 latent plan 的 risk structure。**

这个 separation of concerns 是论文最重要的 design philosophy。LLM 做它最擅长的（reasoning + tool use），oracle 做它最擅长的（calibrated risk estimation + budget enforcement）。两者 decoupled 但协同。

---

## References

- [INTENT paper](https://arxiv.org/abs/2509.14157)
- [ReAct (Yao et al. 2022)](https://arxiv.org/abs/2210.03629)
- [ToolLLM (Qin et al. 2023)](https://arxiv.org/abs/2307.16789)
- [StableToolBench (Guo et al. 2024)](https://arxiv.org/abs/2406.04427)
- [RAP (Hao et al. 2023)](https://arxiv.org/abs/2305.14992)
- [Dreamer / World Models](https://danijar.com/worldmodels/)
- [Ha & Schmidhuber, World Models](https://worldmodels.github.io/)
- [nanoGPT — Karpathy](https://github.com/karpathy/nanoGPT)
- [Karpathy: State of GPT](https://www.youtube.com/watch?v=zjkBMFhNjjc)
- [Karpathy: Deep Dive into LLMs](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Karpathy blog](https://karpathy.bearblog.dev/)
- [MCP — Anthropic](https://www.anthropic.com/news/model-context-protocol)
- [BATS](https://arxiv.org/abs/2511.17006)
- [BTP](https://aclanthology.org/2024.acl-long.163/)
- [PPO](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [On Calibration of Neural Networks](https://arxiv.org/abs/1706.04599)
- [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)
- [Speculative Decoding](https://arxiv.org/abs/2211.17151)
- [MuZero](https://arxiv.org/abs/1911.08265)
- [POMCP (Silver & Veness)](https://papers.nips.cc/paper/2010/hash/edfbe28fb28d296c9c2532dca5b62dc3-Abstract.html)
- [Online Knapsack](https://www.cs.technion.ac.il/~danid/PDFs/md-online.pdf)
- [System Verifiers](https://arxiv.org/abs/2406.14525)
- [Geometric Distribution](https://en.wikipedia.org/wiki/Geometric_distribution)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [vLLM](https://arxiv.org/abs/2309.06180)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)

希望这版讲清楚了直觉。如果你想再深挖某个点（比如 geometric distribution 的 alternative formulation，或者 intention calibration 的 ECE 实验细节），告诉我具体哪个方向。

---

# INTENT: Budget-Constrained Agentic LLMs 的 Intention-Based Planning

Karpathy 你好，这篇 paper 触到了一个非常实际的痛点：当 agent 调用 API 真的要花钱的时候，怎么让它"理性消费"？我尽量从 RL / planning / world model 的角度把这件事拆解透彻。

---

## 1. Problem Setting：为什么这是真的难题

paper 设定一个非常具体但很有代表性的场景：一个 agentic LLM 在一个动态的 tool marketplace $\mathcal{M} = \{(T^{(j)}, c^{(j)})\}_{j=1}^m$ 中工作，其中 $T^{(j)}$ 是 tool spec（描述、input schema、example），$c^{(j)} \in \mathbb{R}_{\geq 0}$ 是单次调用价格。task instance 是三元组 $\mathcal{I} = (q, B, \mathcal{M})$，$q$ 是 user query，$B$ 是 hard budget。

agent 走 ReAct loop（[Yao et al. 2022, ReAct](https://arxiv.org/abs/2210.03629)），history 单调增长：

$$h_{t+1} = [h_t, r_t, a_t, o_t]$$

其中 $r_t$ 是 reasoning trace，$a_t$ 是 action（CALL 或 ANSWER），$o_t \sim \mathcal{E}(\cdot | T_t, u_t)$ 是 stochastic observation。

reward function 是 piecewise 的：

$$R(\tau) = J(a_K, q) \cdot \mathbb{I}\left(\sum_{t=1}^{K} \text{COST}(a_t) \leq B\right)$$

变量解释：$J(a_K, q) \in [0,1]$ 是 external judge（人或 judge model）对最终答案的评分，$\mathbb{I}(\cdot)$ 是 indicator function，超 budget 直接归零，**这是一阶 step function 的 hard constraint**，没有 slack，没有 soft penalty——这点很关键，意味着 budget 一旦越过，整个 trajectory 价值归零，**credit assignment 在 boundary 上不连续**。

### 为什么这个 setting 比一般 RL 难

1. **Action space unbounded**：tool argument 是 free-form text，token space $\mathcal{V}^*$，没法 enumeration
2. **Transition stochastic**：同一个 tool call 可能返回 noise / irrelevant results，触发 retry loop
3. **Cost 是 irretrievable**：钱花了就花了，不像 RL gym 可以 reset
4. **Non-stationary market**：tools 和 prices 随时间变，新 tool 上线没有 prior
5. **Inference-time only**：post-training RL 太贵且 mismatch，因为真实 service quality 在 inference time 才显现

paper Section 1 列了三个 "natural 但失败" 的 alternative：online knapsack、RL post-training、MCTS。我下面会逐个解释为什么这些都不 work。

---

## 2. 为什么 Naive Methods 都 Fail

### 2.1 PROMPT 的失败（Table 1 中 FR=67.2 for GPT-4.1-mini）

paper 在 Section 1 给出一个 brutal 的发现：你即使把 cost 信息和剩余 budget 显式塞进 prompt，模型仍然在 32.8% 的任务中超支。

原因在 paper 的 Case Study（Appendix E）非常清楚：
- agent 选了一个 cost=38 的 tool，剩 12 budget
- 它**完全知道**只剩 12，却仍执意调用 cost=23 的下一个 tool
- 超 budget 11，然后说 "oops"

这是经典的 **myopic sequential decision**：每一步看似合理，但累计起来就爆。它跟 LLM 的 autoregressive 生成本质相关——模型每次只看 current context，没有真正的 multi-step lookahead。这跟你（Karpathy）老说 "LLM 是 stateless next-token predictor" 完全一致。

更深一层：**path dependence**。一旦 agent 选了某条路径（比如先抓 cash flow），它就 trapped 在那条路径上，不会 backtrack 去考虑 cheaper substitute。这是 autoregressive policy 的 fundamental limitation。

### 2.2 Reasoning Model 也只是"过度保守"

paper Figure 1 说 GPT-5-nano（reasoning backbone）确实 budget compliance 更好，但是通过"过度保守"实现的——它干脆不调用，导致 pass rate 远低于 budget-optimal 上界。这是另一类失败：**under-exploration**。

这在 RL 中是经典的 exploration-exploitation tension，但在这里被 hard budget 加倍放大：exploration 本身就是 expensive 的，而 budget 是 hard constraint。

### 2.3 Online Knapsack 不适用

paper Section 1 提到 knapsack/LP 失败。原因是它假设 utility additive independent，但 tool 之间有 **sequential dependency**：tool B 的 value 取决于 tool A 给了什么信息。LLM agent 的 value 是 information-theoretic 的，不是 piecewise-linear 的。

参考 [Buchbinder & Naor 2009, Primal-Dual Online Algorithms](https://www.cs.technion.ac.il/~danid/PDFs/md-online.pdf) 和 [Agrawal et al. 2009, Online LP](https://arxiv.org/abs/0906.1220)，这些 classical online optimization 都假设可分离的 utility，跟 agent 的 sequential information gain 是 mismatch 的。

### 2.4 RL Post-Training 不适用

[PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300) 这类 RLHF-style 训练有几个 fundamental problems：
- 训练时 agent 把 tool knowledge 固化到 parameters，但 service quality / coverage / reliability 在 inference 时才显现
- 真要 capture 动态需要频繁 retrain，对 frontier model 极其昂贵
- on-policy data collection 本身要花真钱调 real APIs

这跟你 Tesla 时代做 [world model for driving](https://arxiv.org/abs/2406.18081) 类似——你不可能让真实 fleet 反复试错来训练 planner，必须用 learned simulator。

### 2.5 MCTS 太贵

classical [MCTS](https://www.cs.utah.edu/~tch/MDP-survey/Meuleau-Braunan-1999.pdf) 在这里不适用有几个原因：
- **action space unbounded**：free-form argument 让 branching factor 爆炸，rollout 不可能 exhaustive
- **tool execution 高 latency**：每个节点要调真 tool，几百毫秒起
- **state 不能 reuse**：因为 stochastic + 长 context，每次 trajectory 都是新的

参考 [Silver & Veness 2010, POMCP](https://papers.nips.cc/paper/2010/hash/edfbe28fb28d296c9c2532dca5b62dc3-Abstract.html), [Lee et al. 2018, CC-POMCP](https://arxiv.org/abs/1802.04409), [Hao et al. 2023, RAP](https://arxiv.org/abs/2305.14992) 都假设 cheap / free environment interaction，这里完全 fail。

paper 的核心 insight 就是：**world model 只做 single-trajectory lookahead，不做 tree search**。这点跟你 [nanoGPT](https://github.com/karpathy/nanoGPT) 路线的精神一致——minimal but principled intervention。

---

## 3. INTENT 的核心 Idea：Intention-Based Decomposition

这是 paper 的核心 contribution，也是最有意思的地方。

### 3.1 Key Observation

agent 决定**改变 high-level plan** 的 trigger，主要是"tool 是否满足我的 intention"，**不是** tool 返回的具体 content。

举个例子：你查 Google 财报，intention 是"拿到 net income 数字"。tool 可能返回
- (a) 正确数字 → 满足 intention
- (b) 404 error → 不满足，retry
- (c) 正确 schema 但 data is wrong → 满足 intention（agent 不知道是错的）

agent 下一步的 decision 主要 driven by (a)/(b)/(c) 这个 binary state，**不** driven by 具体的 net income 数值。这个 abstraction 让 cost 估计变得 tractable。

### 3.2 Probabilistic Factorization

引入 binary latent $z_t \in \{0, 1\}$，$z_t = 1$ 表示 intention satisfied。用 total probability 分解：

$$P_w(o_t \mid r_t, a_t) = \sum_{z_t \in \{0,1\}} \underbrace{P(o_t \mid a_t, z_t)}_{\text{Generation}} \cdot \underbrace{P(z_t \mid r_t, a_t)}_{\text{Intention}}$$

变量解释：
- $o_t$：tool observation
- $r_t$：reasoning trace，承载 agent 的 intention
- $a_t$：tool call（包含 tool $T_t$ 和 arguments $u_t$）
- $z_t$：binary intention satisfaction latent

**重要 assumption**：在 generation term 中 omit $r_t$，因为一旦 $z_t$ 确定，$o_t$ 主要取决于 tool semantics。这跟 [VLM 中的 latent variable decomposition](https://arxiv.org/abs/2110.08864) 思路类似，把 high-level semantic 从 low-level generation 中 decouple。

这个 factorization 让 world model 分成两个模块：

| 模块 | 输入 | 输出 | 实现 |
|------|------|------|------|
| Intention Predictor $\mathcal{W}_\rho$ | $[r_t, T_t, u_t]$ | $\tilde{\rho}_t \in [0,1]$ | Qwen3-0.6B-Embedding + classification head |
| Conditional Generator $\mathcal{W}_\psi$ | $[T_t, u_t, z_t]$ | $\tilde{o}_t$ | Qwen2.5-3B-Instruct fine-tuned |

### 3.3 Ideal Trajectory Simulation

这是 INTENT 最巧妙的地方。MCO 的 rollout 是 stochastic 的，high variance。INTENT 改成 **deterministic ideal trajectory**：每一步都强制 $z_k = 1$，看 agent 在"理想世界"下会走什么路径。

$$\tilde{\tau}^* = (\tilde{a}_t, \tilde{o}_t^{(z=1)}, \tilde{r}_{t+1}, \tilde{a}_{t+1}, \ldots, \tilde{a}_K)$$

这相当于 **maximum likelihood latent plan extraction**。agent 自己不知道自己 latent 的 plan 长什么样，oracle 通过强制 ideal rollout 把它"逼出来"。

这点跟你以前讲的 ["LLMs have latent plans but they're not explicit"](https://www.youtube.com/watch?v=l8pRSuU81PU) 的 observation 完全吻合。INTENT 提供了一个 mechanism 把 latent plan 显式化。

### 3.4 Geometric Cost Calibration（关键公式）

这是 paper 里最 math-heavy 的部分。Intuition 是：agent 在真实执行时不会一次性 stop on first failure，它会 retry 直到 intention met。如果 retry 服从 geometric distribution（每次 success probability $\tilde{\rho}_k$），那么期望尝试次数：

$$\mathbb{E}[\text{tries}] = \frac{1}{\tilde{\rho}_k}$$

所以 expected cost：

$$\tilde{c}_k = \frac{\text{COST}(a_k)}{\tilde{\rho}_k}$$

变量解释：
- $\text{COST}(a_k)$：单次 tool call 的 price $c^{(j)}$
- $\tilde{\rho}_k$：calibrated intention satisfaction probability
- $\tilde{c}_k$：悲观估计的 expected cost

**这里有个 implicit assumption**：retry 之间没有 information gain（$\tilde{\rho}_k$ 是 constant）。这在实践中不完全对——agent 可能 refine arguments 后 success 概率上升——但作为 upper bound 估计是合理的。

最后引入 risk preference $\gamma \in \mathbb{R}_+$，做 risk-adjusted 检查：

$$\gamma \cdot \sum_{\tilde{a}_k \in \tilde{\tau}^*} \tilde{c}_k \leq B_t$$

变量解释：
- $\gamma$：risk factor，$\gamma \geq 1$ 严格保守，$\gamma < 1$ 激进
- $B_t$：当前剩余 budget
- 求和 over 整条 ideal trajectory 的 expected cost

paper 实验 $\gamma = 0.5$，Figure 7 显示 $\gamma \in [0.5, 0.7]$ 是 sweet spot。

这个 $\gamma$ 的作用类似于 [Dreamer](https://danijar.com/worldmodels/) 里的 $\lambda$-return 或 [MuZero](https://arxiv.org/abs/1911.08265) 里的 temperature，是 exploration-exploitation knob。

---

## 4. Architecture 图解

paper Figure 2 对比了三种 paradigm：

### 4.1 (a) MCTS Paradigm
```
            Root
           / | \
         b1 b2 b3   ← branching factor huge (free-form args)
        /|  |  |\
       ... ... ...   ← each node real tool call (expensive)
       ↑
    prohibitive latency
```

### 4.2 (b) MCO Paradigm
```
   [Current h_t]
        ↓
   agent proposes a_t
        ↓
   World Model W_φ simulates
        ↓
   single stochastic rollout τ̃
        ↓
   COST(τ̃) ≤ B_t ? ACCEPT : REJECT + feedback
        ↓
   (high variance: single lucky sample)
```

### 4.3 (c) INTENT Paradigm（核心）
```
   [Current h_t, r_t, a_t]
        ↓
   ┌──────────────────────────────────┐
   │  IDEAL TRAJECTORY SIMULATION      │
   │  for k = t, t+1, ..., K:          │
   │    1. ρ̃_k = W_ρ(z=1|r_k, T_k, u_k)│  ← Intention Predict
   │    2. c̃_k = COST(a_k) / ρ̃_k       │  ← Geometric Calibrate
   │    3. õ_k ~ W_ψ(·|T_k, u_k, z=1)  │  ← Ideal Generation
   │    4. r̃_{k+1}, ã_{k+1} ~ π_θ      │  ← Agent Step
   └──────────────────────────────────┘
        ↓
   γ · Σ c̃_k ≤ B_t ? ACCEPT : REJECT
        ↓
   If ACCEPT: cache [ã_{t+1}, ..., ã_K]  ← Rollout Reuse
   If REJECT: feedback ô_t = [a_t, ρ̃_t, ..., ã_K]
        ↓
   agent re-plans based on risk structure
```

---

## 5. Training Pipeline

paper Appendix A.1 给了详细 training 细节：

**Data**：
- Language World Model $\mathcal{W}_\phi$ & Conditional Generator $\mathcal{W}_\psi$：用 [StableToolBench MirrorAPI-Cache](https://arxiv.org/abs/2501.18125)，~100k real trajectories from RapidAPI
- Intention Predictor $\mathcal{W}_\rho$：从 [ToolBench Reproduction Data](https://arxiv.org/abs/2307.16789) 构造 86k $(r, a, o)$ triples
- $z_t$ label：用 [GPT-4.1-mini as judge](https://arxiv.org/abs/2306.05685) 标注，保留 28k "successful" trajectories for conditional generator

**Important**：Intention Predictor 用 [post-hoc temperature scaling](https://arxiv.org/abs/1706.04599) 做 calibration，因为 neural network 倾向 overconfidence。Figure 6 显示 calibration 前后 reliability diagram 改善明显。这点很关键——$\tilde{\rho}_k$ 直接进 denominator，miscalibration 会让 $\tilde{c}_k$ 严重低估或高估。

**Hyperparameters**：
- AdamW optimizer, [cosine schedule with warmup](https://arxiv.org/abs/1608.03983)
- World Model / Conditional Generator: bs=64, lr=7e-5, Qwen2.5-3B-Instruct
- Intention Predictor: bs=32, lr=5e-5, Qwen3-0.6B-Embedding
- BF16 [full fine-tuning](https://arxiv.org/abs/1710.03740), 2 epochs
- [FlashAttention-2](https://arxiv.org/abs/2307.08691), [vLLM](https://arxiv.org/abs/2309.06180) serving

**Compute**: single NVIDIA RTX Pro 6000 GPU。这跟你的 [LLM training 从 0 到 hero](https://karpathy.ai/zero-to-hero.html) 哲学一致——这工作不需要 frontier-scale compute。

---

## 6. Experimental Results 解读

### 6.1 Main Results (Table 1)

| Backbone | Method | PR↑ | OR↑ | FR↑ | Time | Tok |
|----------|--------|-----|-----|-----|------|-----|
| GPT-4.1-mini | RAW | 19.1 | 23.3 | 34.5 | 1.00× | 1.00× |
| | PROMPT | 30.9 | 37.7 | 67.2 | 0.65× | 0.56× |
| | DFSDT | 44.9 | 54.7 | 100 | 0.97× | 1.14× |
| | BTP | 46.4 | 56.8 | 100 | 1.01× | 0.84× |
| | BATS | 53.0 | 64.6 | 100 | 1.96× | 4.13× |
| | MCO (ours) | 58.9 | 71.8 | 100 | 1.90× | 2.15× |
| | **INTENT** | **63.8** | **77.8** | **100** | **1.23×** | **1.70×** |
| GPT-5-nano | **INTENT** | **76.0** | **92.6** | **100** | 1.79× | 2.40× |

几个关键 takeaways：

1. **PROMPT vs RAW**：PROMPT 看似快（0.65×）实际是因为它早 stop（被 budget scared off），FR=67.2 说明它经常超支
2. **Enforce baselines 全 FR=100**：DFSDT/BTP/BATS 都强制不超支，但 PR 差异大
3. **BATS 很贵**：reasoning model 上 BATS 要 10.1× latency！这印证了 paper Section 1 的论断——classical planning 在大 action space 上 prohibitive
4. **INTENT 在 reasoning backbone 上达到 OR=92.6**：这意味着它几乎触到了 budget 下的理论上限
5. **INTENT vs MCO**：INTENT 比 MCO 高 4-5 个点，证明 intention decomposition 的 variance reduction 真的有效

### 6.2 Why INTENT beats MCO

MCO 是 INTENT 的简化版，用 single stochastic rollout 估计 cost。Table 1 显示 MCO 也不错（71.8 vs INTENT 77.8 on OR），但有两个 failure modes：

1. **Lucky sample bias**：single rollout 可能恰好走了"happy path"，cost 估计偏小，导致 budget 真超
2. **No latent plan extraction**：MCO 的 feedback 只是 action sequence，agent 不知道哪些步骤是 risk bottleneck

INTENT 的 ideal trajectory + geometric calibration 同时解决这两个问题。

### 6.3 Ablation Studies (Table 3, 4)

**Intention Predictor ablation**:
- OPTIMISTIC ($\tilde{\rho}=1$): PR=59.3（vs INTENT 63.8）— 退化为只看 deterministic cost，没用
- UNCALIBRATED: PR=59.7 — calibration 重要
- EMPIRICAL MEAN: PR=62.7 — 比 OPTIMISTIC 好，但忽略了 contextual info

**Conditional Generator ablation**:
- STOCHASTIC: PR=62.0 — 退化为 MCO-style
- NO ROLLOUT: PR=60.2 — 没有 lookahead，纯 immediate check
- NO TRACE: PR=62.7 — rejection 时不给 trace，agent 失去 risk structure 信息

**Efficiency mechanisms** (Table 4): rollout cache, last-call cache, blacklist 主要省 time，对 PR 影响小（PR 从 63.8 → 65.3/63.4/62.5），证明它们是工程优化不改变核心决策。

---

## 7. Robustness Experiments

paper Section 4.3 的 robustness 实验设计得非常 thoughtful。

### 7.1 New Tools (Figure 3)

模拟新 tool 不断上线，用不同 fraction 的 training data 训 oracle。结果显示 **log-linear scaling**：用 1/32 数据已经能拿到不错的 PR。这跟你常说 ["It's the data, stupid"](https://www.youtube.com/watch?v=L8pRSuU81PU) 哲学一致，也表明 oracle 学的是"intention alignment 模式"，不是 memorize 具体 tool。

### 7.2 Price Perturbation (Figure 4)

把 reference tool 的价格 ±50% 变动，看 model 怎么应对：
- PROMPT 极敏感（PR 大幅波动），因为 path-dependent，没有 substitution reasoning
- INTENT 稳定，因为它做 risk-adjusted cost estimation 时自然考虑 alternative

这点的关键 insight：**INTENT 把 cost-awareness 从 prompt-level 移到 algorithmic-level**，agent 不必理解"价格变了我要换 tool"——oracle 在背后做这个 reasoning。

### 7.3 Budget Scaling (Figure 5)

budget 从默认 50 缩放 ±若干倍。INTENT 在小 budget 下也能维持较高 PR，PROMPT 在小 budget 下崩。这个 figure 强烈建议 INTENT 在 production deployment 中的价值——你可以根据业务 SLA 调 budget 而 expect PR 平滑变化。

---

## 8. Case Study 深度解析 (Appendix E)

paper 的 case study (task 70610) 非常清楚地展示了 INTENT 怎么 work。

**Query**: 获取 Google 的 quarterly cash flow, balance sheet, income statement, ratios。Budget = 50。

**PROMPT 的失败路径**：
- Thought 1: 想抓 cash flow，cost=38，剩 12
- Thought 2: **明知只剩 12**，仍调 cost=23 的 balance sheet
- Thought 3: "Oops! Budget exceeded by 11" → ANSWER with incomplete data

**INTENT 的成功路径**：
- Thought 1: 提议同样的 cash flow（cost=38, p=0.35）
- **INTENT SIMULATION**：ideal trajectory 算出 expected cost = 38/0.35 + 11/0.97 = 108.7 + 11.3 = 120，γ=0.5 → 60 > 50 → REJECT
- Feedback 给 agent：暴露 $\tilde{\rho}_t = 0.35$ 这个 bottleneck
- Thought 2: agent 切换到 Freedom Finance tool（cost=7, p=0.02）— **更便宜但 intention 概率太低**
- **INTENT REJECT again**: 7/0.02 = 350，huge expected cost
- Blacklist 加入这俩 tool
- Thought 3: 找到 Holistic Finance Stock Data (cost=9, p=0.71)
- **INTENT SIMULATION**: 9/0.71 + 11/0.94 + 23/0.90 = 12.7 + 11.7 + 25.6 = 50，γ=0.5 → 25 < 50 → ACCEPT
- 后续 cache hit，顺畅执行完

这 case 极好地展示了 INTENT 的核心价值：**它不只是 budget checker，它是 risk structure expositor**。让 agent 看到"你以为 X tool 便宜，但失败概率高，expected cost 反而爆表"。

---

## 9. 与相关工作的 Comparison

### 9.1 vs [LLM-as-Planner / RAP](https://arxiv.org/abs/2305.14992)

RAP 用 LLM 自己当 world model + MCTS。INTENT 区别：
1. **专门的 intention predictor**（calibrated classifier）替代 stochastic LLM simulation
2. **Single-trajectory lookahead** 替代 MCTS tree
3. **Geometric cost calibration** 处理 stochastic retry

### 9.2 vs [BATS](https://arxiv.org/abs/2511.17006)

BATS（Liu et al. 2025）做 budget-aware tool use 但用 prompt-based budget tracker，且每步 self-verification，latency 爆炸（reasoning model 10.1×）。INTENT 用 model-based planning，一次性 anticipate 整条 trajectory。

### 9.3 vs [BTP](https://aclanthology.org/2024.acl-long.163/) (Budget-constrained Tool Planning)

BTP 把 tool selection 形式化为 multi-knapsack，用 GPT evaluator + embedding similarity 静态分配 quota。它根本不 reasoning sequential dependency。INTENT 用 online lookahead 动态调整。

### 9.4 vs [Dreamer / World Models](https://danijar.com/worldmodels/)

跟你在 Tesla 推动的 [Drive world model](https://www.youtube.com/watch?v=j0z4FweC4RM) 有概念相似：
- 都用 learned world model 替代 expensive 真实 interaction
- 都做 latent state inference
- 都做 planning in imagination

差异：Dreamer 用 RSSM 连续 latent state，INTENT 用 discrete $z_t$ binary；Dreamer 用 policy learning in dream，INTENT 用 lookahead evaluation 不改 policy。

### 9.5 vs [MCP](https://www.anthropic.com/news/model-context-protocol)

MCP 让 tool 标准化接入但**完全没 cost awareness**。INTENT 可以视为 MCP 之上的一层 cost-aware planning layer。这个方向未来很有 product 价值——想象一下 Anthropic 自己的 [Claude with MCP](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use) 加上 INTENT-style budget enforcement。

---

## 10. Limitations 和 Open Questions

paper 没明说但我觉得有几个：

1. **Geometric distribution assumption 强**：真实 retry 之间有 information gain，success probability 应该单调上升。可以建模为 [Bernoulli with learning](https://arxiv.org/abs/1906.06109)，但需要更复杂 estimator
2. **World model hallucination**：$\mathcal{W}_\psi$ 生成 ideal observation 但可能 hallucinate 不存在的 data。paper 说"format 估计足够 eliciting latent plan"，但在 high-stakes 场景（如金融决策）这 risky
3. **Intention annotation 用 LLM-judge**：自 bootstrapping 可能有 systematic bias，不同 judge 模型给出不同 $z_t$
4. **Rollout cache assumption 强**：cache hit 要求 $a_{t+1} \approx \tilde{a}_{t+1}$，但 argument 可能在 fine-grained 层面 mismatch，paper 用 "tolerant for some argument mismatches" 一笔带过
5. **No theoretical bound**：paper 没有 regret bound 或 sample complexity，全 empirical。能不能证明 INTENT 的 cost estimate 是某 ideal estimator 的 upper bound with high probability？这跟 [UCB / Thompson Sampling](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf) 的形式分析可以 connect 起来

---

## 11. 跟你（Karpathy）工作的联想

几个直觉上的 connection：

### 11.1 跟 [World Models (Ha & Schmidhuber)](https://worldmodels.github.io/)

INTENT 的 $\mathcal{W}_\phi$ 是 Language World Model，跟 visual world model 思路一致——把 expensive environment 替换成 cheap learned simulator。但 INTENT 引入 $z_t$ latent 把 simulation 节约到只在 intention 层面，是个聪明的 abstraction。

### 11.2 跟 [nanoGPT / minBPE](https://github.com/karpathy/nanoGPT) 的精神

INTENT 的 implementation 是 Qwen2.5-3B + Qwen3-0.6B，单卡 GPU 训完。这跟你"小而美"的工程哲学契合——不需要 frontier-scale 就能 demo 一个 strong idea。

### 11.3 跟你讲 ["State of GPT" / agentic AI](https://www.youtube.com/watch?v=zjkBMFhNjjc)

你常讲 LLM agent 当前的瓶颈是 reasoning + tool use 之间的 coupling。INTENT 给了一种思路：**把 high-level planning 和 low-level execution 解耦**——agent 还是用 ReAct 做 execution，但 oracle 在背后 enforce 高层 budget constraint。

### 11.4 跟 ["Deep Learning: System Verifier"](https://arxiv.org/abs/2406.14525)

你最近关注的 [System 2 thinking / verifier-based reasoning](https://www.youtube.com/watch?v=l8pRSuU81PU) 跟 INTENT 的 oracle 有点像——都是 inference-time 的 verifier/planner。INTENT 验证 budget feasibility 而非 logical correctness，但 mechanism 类似。

---

## 12. 个人评价

这篇 paper 给我的感觉是 **problem-driven** 而非 technique-driven。它没发明新架构，没发明新 RL algorithm，但它精准地诊断了一个真实工业场景（agentic API economics）的痛点，然后用一个 minimal 且 principled 的方法解决。

几个真正 strong 的地方：
1. **Hard budget formulation** 让 reward 是 step function，这个 abstraction 抓住了工业 reality
2. **Intention-based decomposition** 是很 deep insight，从 cognitive science 借的——agent 改 plan 的 trigger 是 semantic alignment 而非 concrete content
3. **Geometric cost calibration** 数学上 simple 但 effective，把 stochastic retry 转成 expected cost
4. **Inference-time only** 是 practical deployment 必须的，paper 严格守住这个边界
5. **Robustness experiments** 设计得很好，模拟 market non-stationarity

几个觉得不够的地方：
1. **没有 theoretical analysis**，全 empirical。Sample complexity, regret bound 这些可以加
2. **World model hallucination** 风险没充分讨论，对 high-stakes 应用可能是 deal-breaker
3. **Single-trajectory lookahead** 太弱，可能在 multi-modal intent distribution 下 fail。可以考虑 [beam search variant](https://arxiv.org/abs/2305.14992)
4. **Intention 是 binary** 太粗糙。可以扩展为 categorical（partial / full / mismatch）让 agent 得到更细 risk signal
5. **没有 explore-exploit formalization**：$\gamma$ 是 ad-hoc knob，没跟 [PAC bounds](https://en.wikipedia.org/wiki/Probably_approximately_correct_learning) 联系起来

---

## 13. 你可能会问的几个问题

我 anticipate 你可能的几个 follow-up：

**Q1**: intention predictor 怎么 generalize to new tools？
A: 因为它 input 是 $[r_t, T_t, u_t]$，对 tool spec 文本编码，cross-tool 泛化来自 transformer 的 in-context learning。Figure 3 的 log-linear scaling 验证了这点。

**Q2**: 为什么不直接 RL fine-tune 一个 budget-aware policy？
A: Section 1 论证过——service quality / coverage / reliability 在 inference time 才显现，offline policy 没法 capture。而且 frontier LLM 重训太贵。

**Q3**: world model 本身错误怎么 propagate？
A: paper 没充分回答。我觉得这是最大 open issue。Possible mitigation 是 online uncertainty estimation + adaptive lookahead depth。

**Q4**: 可以用在 multi-agent 场景吗？
A: 应该可以。每个 agent 独立 oracle，coordinator 维护 total budget。但 sequential dependency across agents 复杂。

**Q5**: 跟 [speculative decoding](https://arxiv.org/abs/2211.17151) 思路像吗？
A: 概念上像——都是用 cheap model anticipate，verify 后才 commit。但 speculative decoding 验证 token-level match，INTENT 验证 intention satisfaction。

---

## References

- [Paper (arXiv)](https://arxiv.org/abs/2509.14157) — INTENT 原文
- [ReAct (Yao et al. 2022)](https://arxiv.org/abs/2210.03629)
- [ToolLLM (Qin et al. 2023)](https://arxiv.org/abs/2307.16789)
- [StableToolBench (Guo et al. 2024)](https://arxiv.org/abs/2406.04427)
- [RAP: Reasoning with LLM via Planning (Hao et al. 2023)](https://arxiv.org/abs/2305.14992)
- [MCTS for POMDPs (Silver & Veness 2010)](https://papers.nips.cc/paper/2010/hash/edfbe28fb28d296c9c2532dca5b62dc3-Abstract.html)
- [Dreamer (Hafner et al.)](https://danijar.com/worldmodels/)
- [Ha & Schmidhuber, World Models](https://worldmodels.github.io/)
- [PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [MCP — Anthropic](https://www.anthropic.com/news/model-context-protocol)
- [RapidAPI](https://rapidapi.com/hub)
- [BATS Budget-Aware Tool Scaling](https://arxiv.org/abs/2511.17006)
- [BTP Budget-Constrained Tool Planning](https://aclanthology.org/2024.acl-long.163/)
- [On Calibration of Modern Neural Networks (Guo et al. 2017)](https://arxiv.org/abs/1706.04599)
- [LLM-as-a-Judge (Zheng et al. 2023)](https://arxiv.org/abs/2306.05685)
- [LLaMA-Factory](https://arxiv.org/abs/2403.13372)
- [vLLM (Kwon et al. 2023)](https://arxiv.org/abs/2309.06180)
- [FlashAttention-2 (Dao 2023)](https://arxiv.org/abs/2307.08691)
- [nanoGPT — Karpathy](https://github.com/karpathy/nanoGPT)
- [Karpathy: State of GPT](https://www.youtube.com/watch?v=zjkBMFhNjjc)
- [Karpathy: Deep Dive into LLMs](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Karpathy: Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNjjc)
- [Speculative Decoding](https://arxiv.org/abs/2211.17151)
- [MuZero](https://arxiv.org/abs/1911.08265)
- [Online Knapsack (Buchbinder & Naor)](https://www.cs.technion.ac.il/~danid/PDFs/md-online.pdf)
- [System Verifiers](https://arxiv.org/abs/2406.14525)
- [SWE-Gym (Pan et al. 2024)](https://arxiv.org/abs/2412.21139)
- [UI-TARS (Qin et al. 2025)](https://arxiv.org/abs/2501.12326)
- [WebSailor](https://arxiv.org/abs/2507.02592)
- [WebThinker](https://arxiv.org/abs/2504.21776)
- [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

---

## 14. Final Takeaway

如果让我用一句话总结这篇 paper 给你的核心 intuition：**Budget-aware agentic planning 不应该作为 LLM 的 capability，而应该作为 inference-time 的 planner overlay**。LLM 仍然做它最擅长的 ReAct，但在它每次 propose action 前，一个 lightweight learned world model 在背后做 risk-aware feasibility check，并通过 feedback 让 agent 看见自己的 latent plan 的 risk structure。

这个 separation of concerns 是论文最重要的 design philosophy，跟你一直讲的 ["don't ask LLMs to do everything, build systems around them"](https://karpathy.bearblog.dev/) 完全一致。

希望这些直觉对你有用。如果想要更深挖某个 component（比如 intention calibration 的 ECE analysis、geometric distribution 的 alternative formulation、或者 rollout cache 的 cache hit rate analysis），告诉我具体哪个方向，我可以再展开。
